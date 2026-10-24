/**
 * Detailed GPU kernel + host pipeline profiler for RFD4-Proteina (`Rfd4pTrunkGpu`, `Rfd4pDecoderGpu`,
 * `Rfd4pEncoderGpu`, and full 60-step `Rfd4pGeneratorGpu`).
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/profile-rfd4p.js
 */

import {
  DEFAULT_LOCAL_SHARDED_DIR,
  Rfd4pGeneratorGpu,
  buildBatch,
  fetchShardedWeights,
  getRfd4pPresetById,
  pairCondFeatures,
  pairFeatures,
  seededRng,
  trunkCondFeatures,
  trunkSeqFeatures,
} from '../../webgpu/rfd4p/index.js';

export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  if (opt['no-submat'] === '1') device.noSubmat = true;
  const half = opt.half !== '0';
  const { tensors: weights, config } = await fetchShardedWeights(DEFAULT_LOCAL_SHARDED_DIR, { includeEncoder: true });
  const gen = new Rfd4pGeneratorGpu(device, weights, config, { half });
  const deviceCapabilities = {
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

  const kernelProfiles = {};
  for (const presetId of (opt.presets ? opt.presets.split(',') : ['uncond_short', 'ppi_pdl1'])) {
    const { batch, n } = await loadPresetBatch(presetId);
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
