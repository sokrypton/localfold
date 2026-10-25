/**
 * Detailed GPU kernel + host pipeline profiler for RFD4-Proteina (`Rfd4pTrunkGpu`, `Rfd4pDecoderGpu`,
 * `Rfd4pEncoderGpu`, and full 60-step `Rfd4pGeneratorGpu`).
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/profile-rfd4p.js
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/profile-rfd4p.js --lengths=100,200,400 --e2e=0
 *
 * 🔴 `--lengths=N,...` EXISTS BECAUSE THE PRESETS ARE ALL SHORT. The longest one in the
 * repository is `ppi_pdl1` at 191 tokens, and `webgpu/rfd4p/trunk.js` changes shape at
 * 362: past there the pair scratch no longer fits one chunk, `triF16` turns off, and the
 * chunked arm of `triangle()` runs a different set of kernels in f32. Nothing measured by
 * a preset reaches that arm, so every number ever recorded for this port describes the
 * short path only - and the designs people run are 100-400. A synthetic unconditioned
 * segment is the one way to profile at the length it will run.
 */

import {
  Rfd4pGeneratorGpu,
  buildBatch,
  fetchShardedWeights,
  getRfd4pPresetById,
  pairCondFeatures,
  pairFeatures,
  resolveRfd4pShardBase,
  seededRng,
  trunkCondFeatures,
  trunkSeqFeatures,
} from '../../webgpu/rfd4p/index.js';

export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  if (opt['no-submat'] === '1') device.noSubmat = true;
  const half = opt.half !== '0';
  // 🔴 NOT THE LOCAL DIR UNCONDITIONALLY. The shards are gitignored, so hard-coding the
  // local base made this tool unrunnable for anyone who had not run the exporter - which
  // is the common case. `resolveRfd4pShardBase` asks a weight file, not the manifest.
  const shardBase = await resolveRfd4pShardBase();
  const { tensors: weights, config } = await fetchShardedWeights(shardBase, { includeEncoder: true });
  const gen = new Rfd4pGeneratorGpu(device, weights, config, { half });
  const deviceCapabilities = {
    // 🔴 WHICH GPU, BECAUSE THE SAME MILLISECONDS MEAN DIFFERENT THINGS ON DIFFERENT ONES.
    // A pass at n = 400 running every large kernel at ~2.9 TFLOP/s is a kernel problem on
    // a card with ten times that and a ceiling on a laptop; the report cannot be read at
    // all without the adapter beside it, and this one did not print it.
    adapter: {
      vendor: device.adapterInfo?.vendor ?? null,
      architecture: device.adapterInfo?.architecture ?? null,
      device: device.adapterInfo?.device ?? null,
      description: device.adapterInfo?.description ?? null,
    },
    submat8: gen.trunk.engine.submat8,
    hasF16: gen.trunk.engine.hasF16,
    hasSubgroupMatrix: gen.trunk.engine.hasSubgroupMatrix,
    subgroupMatrixConfigs: gen.trunk.engine.subgroupMatrixConfigs,
  };

  async function loadPresetBatch(presetId) {
    const preset = getRfd4pPresetById(presetId);
    const resp = await fetch(`/${preset.cifUrl}`);
    const bytes = new Uint8Array(await resp.arrayBuffer());
    return buildBatch(bytes, { seed: 42 });
  }

  /**
   * One unconditioned segment of `len` residues, built from a contig alone.
   *
   * No structure is involved: a spec with a contig and no input starts from an empty atom
   * array, so the length is free and the batch is the same shape the presets produce.
   */
  function loadLengthBatch(len) {
    return buildBatch({ contig: `${len}-${len}` }, { seed: 42 });
  }

  // A case is a preset id or a synthetic length; `--lengths` adds the second kind.
  const cases = [];
  if (opt.lengths) {
    for (const raw of opt.lengths.split(',')) {
      const len = Number(raw);
      if (!Number.isInteger(len) || len < 2) throw new Error(`--lengths: ${raw} is not a length`);
      cases.push({ key: `n${len}`, load: () => loadLengthBatch(len) });
    }
  }
  if (!opt.lengths || opt.presets) {
    for (const presetId of (opt.presets ? opt.presets.split(',') : ['uncond_short', 'ppi_pdl1'])) {
      cases.push({ key: presetId, load: () => loadPresetBatch(presetId) });
    }
  }

  const kernelProfiles = {};
  for (const { key: presetId, load } of cases) {
    const { batch, n } = await load();
    const rng = seededRng(42);
    const nnCfg = config.nn;
    const distCfg = { dim: nnCfg.xt_pair_dist_dim, min: nnCfg.xt_pair_dist_min, max: nnCfg.xt_pair_dist_max };
    const x = {
      bb_ca: new Float32Array(n * 3),
      local_latents: new Float32Array(n * 32),
    };
    const tt = { bb_ca: 0.5, local_latents: 0.5 };
    const featCache = {};
    // Warm up caches
    trunkSeqFeatures(batch, {
      n, xt: x, feats: nnCfg.feats_seq, weights, randperm: (len) => rng.randperm(len, 0), noiseAlignChirality: nnCfg.noise_align_chirality, cache: featCache,
    });
    trunkCondFeatures(batch, { n, t: tt, recycleRemaining: null, feats: nnCfg.feats_cond_seq, weights, cache: featCache });
    trunkCondFeatures(batch, { n, t: tt, recycleRemaining: 1, feats: nnCfg.feats_cond_seq, weights, cache: featCache });
    pairFeatures(batch, { n, caNm: x.bb_ca, feats: nnCfg.feats_pair_repr, dist: distCfg, seqSepDim: nnCfg.seq_sep_dim, cache: featCache });

    const tSeq0 = performance.now();
    const seq = trunkSeqFeatures(batch, {
      n, xt: x, feats: nnCfg.feats_seq, weights, randperm: (len) => rng.randperm(len, 0), noiseAlignChirality: nnCfg.noise_align_chirality, cache: featCache,
    }).data;
    const seqMs = performance.now() - tSeq0;

    const tCond0 = performance.now();
    const cond0 = trunkCondFeatures(batch, { n, t: tt, recycleRemaining: null, feats: nnCfg.feats_cond_seq, weights, cache: featCache }).data;
    const cond1 = trunkCondFeatures(batch, { n, t: tt, recycleRemaining: 1, feats: nnCfg.feats_cond_seq, weights, cache: featCache }).data;
    const condMs = performance.now() - tCond0;

    const tPair0 = performance.now();
    const pair = pairFeatures(batch, { n, caNm: x.bb_ca, feats: nnCfg.feats_pair_repr, dist: distCfg, seqSepDim: nnCfg.seq_sep_dim, cache: featCache }).data;
    const pairMs = performance.now() - tPair0;

    const pairCond = pairCondFeatures(tt, nnCfg.feats_pair_cond);
    const f = { seq, cond0, cond1, pair, pairCond };

    // Warmup GPU trunk + record
    await gen.trunk.forward(n, f, 2, { insertion: true });

    // Measure wall-clock forward(n, f, 3) over 5 iterations (also boosts GPU to P0 clock before kernel profiling)
    const tFwd0 = performance.now();
    const ITERS = 5;
    for (let it = 0; it < ITERS; it += 1) {
      await gen.trunk.forward(n, f, 3, { insertion: false });
    }
    const fwd3WallMs = Number(((performance.now() - tFwd0) / ITERS).toFixed(2));

    // Profile initProgram, passPrograms[0], and headProgram using timestamp queries
    const initMap = await gen.trunk.engine.profile(gen.trunk.initProgram);
    const passMap = await gen.trunk.engine.profile(gen.trunk.passPrograms[0]);
    const headMap = await gen.trunk.engine.profile(gen.trunk.headProgram);

    const summarizeMap = (m) => {
      const rows = [...m.entries()]
        .map(([label, v]) => ({ label, ms: Number(v.ms.toFixed(3)), calls: v.calls, perCallUs: Number(((v.ms * 1000) / v.calls).toFixed(1)) }))
        .sort((a, b) => b.ms - a.ms);
      const totalMs = Number(rows.reduce((s, r) => s + r.ms, 0).toFixed(3));
      return { totalMs, topKernels: rows.slice(0, 20) };
    };

    // Profile decoder
    await gen.decoder.decode(batch, { n, caNm: x.bb_ca, z: x.local_latents, seq: new Int32Array(n).fill(1) });
    const decMap = await gen.decoder.engine.profile(gen.decoder.decodeProgram);

    kernelProfiles[presetId] = {
      n,
      // 🔴 THE SHAPE THE TRUNK ACTUALLY CHOSE, recorded beside the timings. Two numbers at
      // different lengths are not comparable when one ran the f16 single-chunk kernels and
      // the other ran the f32 chunked ones, and nothing in a kernel name says which.
      shape: { triF16: gen.trunk.triF16, chunks: gen.trunk.chunks.length, chunkRows: gen.trunk.R },
      cpuFeaturesMs: {
        seqMs: Number(seqMs.toFixed(3)),
        condMs: Number(condMs.toFixed(3)),
        pairCachedMs: Number(pairMs.toFixed(3)),
      },
      fwd3WallMs,
      initGpu: summarizeMap(initMap),
      passGpu: summarizeMap(passMap),
      headGpu: summarizeMap(headMap),
      decoderGpu: summarizeMap(decMap),
    };
  }

  // End-to-end 60-step benchmark on uncond_short (low & medium) and ppi_pdl1 (low)
  const e2e = {};
  if (opt.e2e !== '0') {
    for (const [presetId, schedule] of [
      ['uncond_short', 'low'],
      ['uncond_short', 'medium'],
      ['ppi_pdl1', 'low'],
    ]) {
      const { batch, n, atoms } = await loadPresetBatch(presetId);
      const t0 = performance.now();
      await gen.generate(batch, { nsteps: 60, recycleSchedule: schedule, seed: 42, inputAtoms: atoms });
      const wallMs = Number((performance.now() - t0).toFixed(1));
      e2e[`${presetId}_${schedule}_60steps`] = { n, schedule, wallMs };
    }
  }

  gen.destroy();
  return { deviceCapabilities, kernelProfiles, e2e };
}
