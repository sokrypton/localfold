/**
 * Does the CHUNKED pair track compute what the single-chunk one computes?
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-rfd4p-chunked.js
 *   ... tools/gpu/check-rfd4p-chunked.js --n=200 --arms=f32,f16 --chunks=2,3,5
 *
 * 🔴 NOTHING HAD EVER CHECKED IT. Every rfd4p oracle case is 54 to 191 tokens and
 * `chunkPlan` keeps one chunk to n = 362 in f32 and 512 in f16, so `check-rfd4p.js`'s
 * whole suite - trunk, decoder, encoder, end-to-end CPU parity - exercises the
 * single-chunk arm and only that arm. The chunked arm runs a DIFFERENT set of kernels:
 * it norms without transposing, projects into a per-chunk `proj`, scatters that into
 * aT/bT through a `tri-gate-*` map, and then re-norms and re-projects a second time per
 * chunk to rebuild the output gate. None of those four had a reference.
 *
 * 🔴 AND `chunkBytes` IS WHAT MAKES THE QUESTION ASKABLE AT A LENGTH THAT FITS. It is a
 * constructor option, so the same n can be forced down either arm: one chunk at the
 * default bound, several at a small one. The two then differ ONLY in the slicing, so the
 * single-chunk arm is the reference and the bar is float non-associativity rather than a
 * tolerance anybody picked. A chunked f32 pass reorders no sum the unchunked one does not
 * - the GEMM is global either way - so f32 against f32 is held near zero; f16 against the
 * f32 reference gets the arithmetic's own bar.
 *
 * 🔴 IT COMPARES THE TRUNK'S OUTPUT, NOT A TRAJECTORY, so it is seconds and not a
 * generation: one `forward` an arm at one set of features. `check-rfd4p-length.js` is the
 * end-to-end question and costs a quarter of an hour an arm.
 */

import {
  Rfd4pTrunkGpu,
  buildBatch,
  fetchShardedWeights,
  pairCondFeatures,
  pairFeatures,
  resolveRfd4pShardBase,
  seededRng,
  trunkCondFeatures,
  trunkSeqFeatures,
} from '../../webgpu/rfd4p/index.js';
import { relativeRms } from './relative-rms.js';

/** The features one `forward` needs, built once and shared by every arm. */
function buildFeatures(batch, n, config, weights) {
  const rng = seededRng(42);
  const nnCfg = config.nn;
  const distCfg = { dim: nnCfg.xt_pair_dist_dim, min: nnCfg.xt_pair_dist_min, max: nnCfg.xt_pair_dist_max };
  // A fixed, non-zero x_t: an all-zero pair distance makes the pair track degenerate and
  // a degenerate reference agrees with anything.
  const xt = { bb_ca: new Float32Array(n * 3), local_latents: new Float32Array(n * 32) };
  for (let i = 0; i < n * 3; i += 1) xt.bb_ca[i] = Math.sin(i * 0.7) * 1.5;
  for (let i = 0; i < n * 32; i += 1) xt.local_latents[i] = Math.cos(i * 0.31);
  const tt = { bb_ca: 0.5, local_latents: 0.5 };
  const cache = {};
  return {
    seq: trunkSeqFeatures(batch, {
      n, xt, feats: nnCfg.feats_seq, weights, randperm: (len) => rng.randperm(len, 0),
      noiseAlignChirality: nnCfg.noise_align_chirality, cache,
    }).data,
    cond0: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: null, feats: nnCfg.feats_cond_seq, weights, cache }).data,
    cond1: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: 1, feats: nnCfg.feats_cond_seq, weights, cache }).data,
    pair: pairFeatures(batch, {
      n, caNm: xt.bb_ca, feats: nnCfg.feats_pair_repr, dist: distCfg, seqSepDim: nnCfg.seq_sep_dim, cache,
    }).data,
    pairCond: pairCondFeatures(tt, nnCfg.feats_pair_cond),
  };
}

export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const n = Number(opt.n ?? 200);
  const arms = (opt.arms ?? 'f32,f16').split(',');
  // How many chunks to force, as a divisor of the pair rows.
  const chunkCounts = (opt.chunks ?? '2,3,5').split(',').map(Number);
  // When set, every entry of `--chunks` uses this R instead of deriving one from a count.
  const explicitRows = opt.rows === undefined ? null : Number(opt.rows);
  // 🔴 THE BAR FOLLOWS THE ARITHMETIC. The contraction is one global GEMM either way, so
  // what chunking changes in f32 is the per-chunk epilogue's rounding and, at a small R,
  // the tile geometry some of the row-wise GEMMs resolve to - a reordering, which is what
  // CLAUDE.md's own band for this is: "kSplits under 1e-5, outKSplits 2.5e-5 ... all
  // correct, all float non-associativity". 1e-5 is that band and nothing looser.
  //
  // 🔴 AND IT IS A BOUND, NOT A SHRUG, because the fault it was built for is five orders
  // away: a chunk starting off a four-row boundary reads `local_latents` at relRMS 0.72 to
  // 0.98. Two f32 levels are observed and both are flat in the chunk COUNT - 1.3e-6 for
  // R >= 900 and 2.4e-6 for R <= 512, the latter identical at 8, 9 and 17 chunks - and a
  // residual that does not grow with the number of chunks is not an accumulating error.
  const bound = { f32: 1e-5, f16: 6e-3 };

  const shardBase = await resolveRfd4pShardBase();
  const { tensors: weights, config } = await fetchShardedWeights(shardBase);
  const { batch } = buildBatch({ contig: `${n}-${n}` }, { seed: 42 });
  const f = buildFeatures(batch, n, config, weights);
  const nn = n * n;

  /** One `forward` at a given arm and chunk bound; returns the three heads. */
  // 🔴 THE INTERMEDIATES, BECAUSE "THE OUTPUT IS WRONG" DOES NOT SAY WHICH LOOP. Each of
  // these is nn-indexed and so has the same length either way; the R-indexed scratch (zc,
  // xc, gz, phid) does not and cannot be compared across slicings. `zInitStatic` and
  // `zInit` are built by their OWN chunk loops before any triangle runs, so a difference
  // there is the pair linear's offsets and a difference only below it is the triangle's.
  const probe = (opt.buffers ?? 'zInitStatic,zInit,z').split(',').filter(Boolean);

  async function run(half, chunkBytes) {
    const trunk = new Rfd4pTrunkGpu(device, weights, config.nn, { half, chunkBytes });
    try {
      const out = await trunk.forward(n, f, 2, {});
      // 🔴 ONLY IN THE f32 ENCODING, because a packed-f16 buffer and an f32 one are not the
      // same bytes and reinterpreting one as the other produced 4.6e17 - a diagnostic
      // reporting nonsense is worse than one that declines. Both arms of a live comparison
      // share `triF16` (the caller enforces it), so this is never the interesting case
      // being skipped.
      const buffers = {};
      if (!trunk.triF16) {
        for (const label of probe) buffers[label] = await trunk.read(label, nn * 384);
      }
      return {
        chunks: trunk.chunks.length,
        chunkRows: trunk.R,
        triF16: trunk.triF16,
        buffers,
        bb_ca: out.bb_ca.slice(),
        local_latents: out.local_latents.slice(),
        logits: out.logits.slice(),
      };
    } finally {
      trunk.engine.destroy();
    }
  }

  const rows = [];
  let failures = 0;
  for (const arm of arms) {
    const half = arm === 'f16';
    const ref = await run(half, undefined);
    if (ref.chunks !== 1) throw new Error(`the ${arm} reference is ${ref.chunks} chunks at n=${n} - raise the bound or lower n`);
    for (const want of chunkCounts) {
      // 🔴 `--rows=` SETS R DIRECTLY, because `--chunks=` CONFOUNDS TWO HYPOTHESES. Every
      // chunk count that divides nn evenly also happened to give an 8-aligned R here, so
      // "the tail is short" and "R is not a multiple of the workgroup's row count" fit the
      // same six measurements. Naming R separates them: ask for one that is 8-aligned and
      // still ragged, and one that divides nn and is not.
      const rowsWanted = explicitRows ?? Math.max(1, Math.ceil(nn / want));
      const chunkedRowFloats = 384 + 256 + 384 + 2 * 384 + 2 * 256;
      const got = await run(half, rowsWanted * chunkedRowFloats * 4);
      // 🔴 A ROW IS ONLY A SLICING COMPARISON WHEN BOTH ARMS HOLD THE SAME ARITHMETIC, and
      // in the f16 arm today they do not: the single-chunk reference narrows the pair track
      // (`triF16`) and the chunked arm cannot, so comparing them measures f16-against-f32
      // activations and not the slicing. Measured, it reads 7.9e-3 on `local_latents` -
      // about the f16 activation error and nothing to do with chunks. Reporting that as a
      // failure would be the gate lying about what it tested, and skipping it silently
      // would hide that the arm is unreachable, so it is named and counted.
      if (got.triF16 !== ref.triF16) {
        rows.push({
          arm,
          wantChunks: want,
          chunks: got.chunks,
          chunkRows: got.chunkRows,
          comparable: false,
          why: `the chunked arm is ${got.triF16 ? 'f16' : 'f32'} where its single-chunk reference is ${ref.triF16 ? 'f16' : 'f32'} - not a slicing comparison`,
        });
        continue;
      }
      const dev = {
        bb_ca: relativeRms(got.bb_ca, ref.bb_ca),
        local_latents: relativeRms(got.local_latents, ref.local_latents),
        logits: relativeRms(got.logits, ref.logits),
      };
      for (const label of probe) {
        if (got.buffers[label] && ref.buffers[label]) dev[label] = relativeRms(got.buffers[label], ref.buffers[label]);
      }
      const worst = Math.max(dev.bb_ca, dev.local_latents, dev.logits);
      const ok = Number.isFinite(worst) && worst <= bound[arm];
      if (!ok) failures += 1;
      rows.push({
        arm,
        wantChunks: want,
        chunks: got.chunks,
        chunkRows: got.chunkRows,
        // 🔴 WHETHER THE LAST CHUNK IS SHORT, which is the discriminator this found: an
        // exact division agrees at 1.9e-7 and a ragged tail at 0.279, and a real length is
        // always ragged (n = 513 is 101944 + 101944 + 59281).
        ragged: nn % got.chunkRows !== 0,
        tailRows: nn - got.chunkRows * (got.chunks - 1),
        triF16: got.triF16,
        refTriF16: ref.triF16,
        bb_ca: Number(dev.bb_ca.toExponential(2)),
        local_latents: Number(dev.local_latents.toExponential(2)),
        logits: Number(dev.logits.toExponential(2)),
        ...Object.fromEntries(probe.filter((label) => dev[label] !== undefined)
          .map((label) => [label, Number(dev[label].toExponential(2))])),
        comparable: true,
        bound: bound[arm],
        ok,
      });
    }
  }

  const compared = rows.filter((r) => r.comparable).length;
  const report = { n, bound, compared, skipped: rows.length - compared, rows, ok: failures === 0 && compared > 0 };
  if (failures) throw new Error(`rfd4p chunked check: ${failures} arm(s) out of band: ${JSON.stringify(report)}`);
  // 🔴 A GATE THAT COMPARED NOTHING IS NOT A PASSING GATE, which is how every skip-based
  // arm here eventually goes quiet.
  if (!compared) throw new Error(`rfd4p chunked check: every row was skipped - nothing was compared: ${JSON.stringify(report)}`);
  return report;
}
