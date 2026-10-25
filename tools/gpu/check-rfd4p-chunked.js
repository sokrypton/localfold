/**
 * Does the CHUNKED pair track compute what the single-chunk one computes?
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-rfd4p-chunked.js
 *   ... tools/gpu/check-rfd4p-chunked.js --n=200 --arms=f32,f16 --chunks=2,3,5
 *
 * 🔴 NOTHING HAD EVER CHECKED IT. Every rfd4p oracle case is 54 to 191 tokens and
 * `chunkPlan` keeps one chunk to n = 362 in f32 and 512 in f16, so `check-rfd4p.js`'s
 * whole suite - trunk, decoder, encoder, end-to-end CPU parity - exercises the
 * single-chunk arm and only that arm. The chunked arm runs a DIFFERENT set of kernels: it
 * norms without transposing, projects into a per-chunk `proj` and scatters that into aT/bT
 * through a `tri-gate-*` map. None of them had a reference.
 *
 * 🔴 AND IT HAS FOUND THREE REAL FAULTS, each invisible to every other gate here:
 *   1. a chunk starting off a four-row boundary (`transposeLnOut` reads `x` four pair rows
 *      a word): relRMS ~1.0 on the pair track, and the default R was 4-aligned by luck;
 *   2. the recycle's own norm reading an f16 pair track as f32 - `inF16` missing where
 *      `outF16` was given, which was RIGHT until f16 stopped implying one chunk;
 *   3. the chunked triangle contracting over k = n rather than the 16-padded nPad the
 *      subgroup-matrix GEMM wants, which the single-chunk arm has always padded for.
 * Each produced a plausible structure rather than an error.
 *
 * 🔴 AND THE TELL THAT SEPARATED (2) FROM AN ACCUMULATION was that its residual was
 * IDENTICAL for two equal chunks and for four ragged ones, to four decimal places. An
 * error that grows with the number of chunks is an accumulation; one that does not is a
 * code path taken on the `chunks.length > 1` branch, and that narrowed six candidate
 * kernels to the two that live there.
 *
 * 🔴 AND `chunkBytes` IS WHAT MAKES THE QUESTION ASKABLE AT A LENGTH THAT FITS. It is a
 * constructor option, so the same n can be forced down either arm: one chunk at the
 * default bound, several at a small one. The two then differ ONLY in the slicing, so the
 * single-chunk arm is the reference and the bar is float non-associativity rather than a
 * tolerance anybody picked. A chunked f32 pass reorders no sum the unchunked one does not
 * - the GEMM is global either way - and measured it is EXACTLY ZERO where the chunks
 * divide evenly, so the f32 arm is a bit-for-bit comparison and not an approximate one.
 *
 * 🔴 THE f16 ARM IS NOT, AND THE RESIDUAL IS STATED RATHER THAN EXPLAINED AWAY: the two
 * f16 arms agree to 2.0e-3 on the alpha carbons and 4.4e-3 on the latents, where their f32
 * counterparts are bit-identical. It is flat in the chunk count and it survives
 * `--recycles=1`, so it is neither an accumulation nor the recycle seam - the two
 * formulations round at different points somewhere in the pass, and WHERE is not yet
 * identified. It is inside the f16 band, and at these lengths the alternative is the f32
 * chunked arm, which differs from f16 by far more; but a clean account of those 2e-3 is
 * missing and this comment is not a substitute for one.
 *
 * 🔴 IT COMPARES THE TRUNK'S OUTPUT, NOT A TRAJECTORY, so it is seconds and not a
 * generation: one `forward` an arm at one set of features. `check-rfd4p-length.js` is the
 * end-to-end question and costs a quarter of an hour an arm.
 */

import {
  Rfd4pTrunkGpu,
  chunkPlan,
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

/**
 * The byte bound that makes `chunkPlan` choose about `rows` rows a chunk.
 *
 * 🔴 ASKED OF `chunkPlan` RATHER THAN DERIVED, because this file had the row cost written
 * out by hand and the cost CHANGED underneath it - gz moved out of the R-indexed set and
 * the chunked shape learned f16 - so the bound that used to force three chunks silently
 * stopped forcing any, and the f16 arm compared a single-chunk run against itself and
 * reported a perfect zero. A checker that restates the thing it is checking goes quiet
 * exactly when that thing moves. R is monotone in the bound, so bisection finds it.
 */
function boundForRows(nn, wantF16, rows) {
  let lo = 16;
  let hi = 8 * 1024 * 1024 * 1024;
  for (let i = 0; i < 64; i += 1) {
    const mid = Math.floor((lo + hi) / 2);
    const plan = chunkPlan(nn, { wantF16, chunkBytes: mid });
    if (plan.chunks === 1 || plan.R > rows) hi = mid; else lo = mid;
  }
  return lo;
}

/**
 * Two f16 values a word, decoded by hand.
 *
 * Not `Float16Array`: it is a recent type, this runs in whatever Chrome the harness found,
 * and a diagnostic that throws on an old browser is a diagnostic nobody can use.
 */
function decodeHalves(words) {
  const out = new Float32Array(words.length * 2);
  for (let i = 0; i < words.length; i += 1) {
    for (let h = 0; h < 2; h += 1) {
      const bits = (words[i] >>> (h * 16)) & 0xffff;
      const sign = bits & 0x8000 ? -1 : 1;
      const exp = (bits >> 10) & 0x1f;
      const frac = bits & 0x3ff;
      out[i * 2 + h] = exp === 0
        ? sign * frac * 2 ** -24
        : (exp === 31 ? (frac ? NaN : sign * Infinity) : sign * (1 + frac / 1024) * 2 ** (exp - 15));
    }
  }
  return out;
}

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
  // 🔴 RECYCLES, BECAUSE THE SECOND CYCLE TAKES A DIFFERENT ROUTE IN EACH ARM. With one
  // chunk the last layer FUSES the next cycle's norm (`outZc`, `zCycNormScaled`); sliced,
  // `recycleIn` does it separately with `zCycNorm`. The two are equivalent and round
  // differently, so `--recycles=1` is the arm that excludes them and says whether a
  // residual belongs to the pair track or to that seam.
  const recycles = Number(opt.recycles ?? 2);
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
      const out = await trunk.forward(n, f, recycles, {});
      // Decoded per arm, because a packed-f16 buffer and an f32 one are not the same bytes
      // and reinterpreting one as the other produced 4.6e17. Both arms of a live comparison
      // share `triF16` (the caller enforces it), so this always compares like with like.
      // Each probed buffer's own element count, because they are not all nn x 384: the
      // triangle's operands are channel-major and row-padded, and its output has its own
      // batch stride. Reading one with another's length is how a diagnostic starts
      // comparing padding.
      const nPad = Math.ceil(n / 16) * 16;
      const scalars = {
        zInitStatic: nn * 384, zInit: nn * 384, z: nn * 384,
        aT: 256 * n * (trunk.triF16 ? nPad : n),
        bT: 256 * n * (trunk.triF16 ? nPad : n),
        x: 256 * Math.ceil(nn / 4) * 4,
      };
      const buffers = {};
      for (const label of probe) {
        const count = scalars[label];
        if (!count) throw new Error(`no element count known for ${label}`);
        buffers[label] = trunk.triF16
          ? decodeHalves(await trunk.read(label, Math.ceil(count / 2), 0, Uint32Array))
          : await trunk.read(label, count);
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
      const got = await run(half, boundForRows(nn, half, rowsWanted));
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
      // 🔴 A NaN-PROOF LOCALISER BESIDE THE relRMS, because an f16 buffer with one
      // non-finite half anywhere makes every relRMS over it NaN and a NaN localises
      // nothing. These say, per buffer, how much DIFFERS and whether each arm is finite at
      // all - which separates "this stage diverged" from "my decode is reading padding".
      const words = {};
      for (const label of probe) {
        const a = got.buffers[label];
        const b = ref.buffers[label];
        if (!a || !b) continue;
        let differ = 0;
        let badGot = 0;
        let badRef = 0;
        const len = Math.min(a.length, b.length);
        for (let i = 0; i < len; i += 1) {
          if (!Number.isFinite(a[i])) badGot += 1;
          if (!Number.isFinite(b[i])) badRef += 1;
          if (a[i] !== b[i] && !(Number.isNaN(a[i]) && Number.isNaN(b[i]))) differ += 1;
        }
        words[label] = {
          differFrac: Number((differ / Math.max(len, 1)).toFixed(4)),
          nonFiniteGot: badGot,
          nonFiniteRef: badRef,
          len,
        };
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
        words,
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
