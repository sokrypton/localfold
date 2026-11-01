/**
 * Which alignment rows a trunk pass reads: AlphaFold 3's per-pass MSA draw.
 *
 * 🔴 A FRESH DRAW EVERY PASS, NOT ONE FOR THE FOLD. AF3's Evoformer calls `shuffle_msa` and then `truncate_msa_batch`
 * inside `_embed_process_msa`, which runs once per trunk pass with that pass's own key - so each of AF3's 11 passes
 * (10 recycles) reads a different `num_msa` (1024) of the up to 16384 rows its featuriser kept. This port drew 1024
 * rows ONCE in the featuriser and fed the same ones to every pass, and on a complex whose small chain has a handful of
 * homologs that is not a detail: 8S83's chain B has ~10 rows in 5630, one draw holds 1-4 of them and eleven see nearly
 * all. Stock AF3 docked 13 seeds of 20 there and 6 of 20 with its shuffle frozen to one draw - exactly this port's 6.
 *
 * So the featuriser now exports the whole pool (msa-features.js af3MsaFromA3m) and every backend's trunk gathers
 * `msaDrawRows(...)` before its MSA embedding: cuda/af3 (trunk.cuh drawMsa), metal/af3 (trunk.mm drawMsa) and the page
 * (webgpu/af3/fold.js). cuda/featurise/msa_draw.h is the C++ twin, integer for integer - test/msa-draw.test.js pins
 * values both must produce.
 *
 * 🔴 THE QUERY IS DRAWN LIKE ANY OTHER ROW. gumbel_argsort_sample_idx gives every unmasked row logit 0, the query
 * included, so on a deep alignment AF3's own query is absent from most passes (1 - 1024/5630 of them on 8S83). The
 * sequence still reaches every row through target_feat. A draw is returned SORTED, so the query - pool row 0 - is
 * row 0 of a pass exactly when it was drawn; boltz2's and rf3's is_paired flag follows it (af3-any-model's
 * shuffle_msa_with_order: the flag is on the row whose ORIGIN is 0).
 *
 * 🔴 THE ORDER IS NOT AF3's, THE SET IS. AF3 leaves its rows permuted; the MSA stack is row-permutation-equivariant
 * (the outer product mean sums over rows, the pair-weighted averaging treats each alone), so only WHICH rows matter,
 * and file order keeps two runs comparable by eye. The PRNG is not JAX's either - a seed here names a draw, not AF3's.
 *
 * Modes (`--msa-draw=` on the native ports, `msaDraw` on foldBatch):
 *   "pass"   - a fresh draw every pass (AF3's; the default)
 *   "once"   - pass 0's draw for every pass (this port's old behaviour, kept as the control arm)
 *   "prefix" - the pool's first rows, every pass (chai-1, which does not shuffle: af3-any-model's `if not chai1`)
 *   "native" - protenix's: a pass reads a COUNT of rows drawn uniformly from 1 to the whole pool, and that many rows
 *              drawn from it - no num_msa cap (`msaNativeSample` in dialect.js; msaNativeRows below)
 *
 * 🔴 AND PROTENIX's TRUNK IS RANDOM HOWEVER SHALLOW THE ALIGNMENT. af3-any-model replays native Protenix's own
 * inference randomness for protenix2 by default (model_registry `native_rng`, pipeline/protenix_rng.py): every pass
 * samples `torch.randint(1, n + 1)` rows by `torch.randperm`, with no 1024 cut (its test cutoff is 16384), and once
 * a seed, with odds 0.4, MC dropout at 0.4 on the recycled pair's projection, a fresh mask every pass
 * (`recycleMcDropout`, `recycleMcDropoutOdds`; mcDropout* below). Native's stream is Philox on the CUDA generator,
 * laid out by the device's SM count, so no second backend can replay its DRAWS; these are its DISTRIBUTION, one
 * integer hash on every backend so a seed names the same fold on all of them.
 */

/** lowbias32 (Chris Wellons): a 32-bit integer hash, exact in JS through Math.imul */
function mix32(x) {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

/**
 * A pass's rows: `count` distinct indices of `rows`, ascending. Every index when `rows <= count`.
 * @param {number} seed the fold's seed (a non-negative integer below 2^53; the native ports' is a uint64, and the two
 *   agree wherever both can hold it)
 * @param {number} pass the trunk pass, 0-based
 */
export function msaDraw(seed, pass, rows, count) {
  if (rows <= count) return Int32Array.from({ length: rows }, (_, row) => row);
  const low = seed % 4294967296, high = Math.floor(seed / 4294967296) % 4294967296;
  let state = mix32((mix32(low ^ mix32(high)) + Math.imul(pass + 1, 0x9e3779b9)) >>> 0);
  const next = () => { state = (state + 0x9e3779b9) >>> 0; return mix32(state); };
  const order = Int32Array.from({ length: rows }, (_, row) => row);
  // a partial Fisher-Yates: the first `count` slots are a uniform draw without replacement. `next() * span` is below
  // 2^46, so the product and the division by 2^32 are exact and agree with the C++'s 64-bit multiply and shift
  for (let at = 0; at < count; at += 1) {
    const span = rows - at;
    const swap = at + Math.floor((next() * span) / 4294967296);
    const held = order[at]; order[at] = order[swap]; order[swap] = held;
  }
  return order.slice(0, count).sort();
}

/** the 32-bit key a seed names, and a stream of it (low and high words hashed together, as the C++ does) */
function seedKey(seed) {
  const low = seed % 4294967296, high = Math.floor(seed / 4294967296) % 4294967296;
  return mix32(low ^ mix32(high));
}

/**
 * protenix's rows for a pass: a count uniform in [1, rows], then that many distinct rows, ascending (msaNativeRows in
 * msa_draw.h is the twin). Its own stream, salted apart from msaDraw's.
 */
export function msaNativeRows(seed, pass, rows) {
  let state = mix32((seedKey(seed) + Math.imul(pass + 1, 0x9e3779b9) + 0x632be5ab) >>> 0);
  const next = () => { state = (state + 0x9e3779b9) >>> 0; return mix32(state); };
  const count = 1 + Math.floor((next() * rows) / 4294967296);
  const order = Int32Array.from({ length: rows }, (_, row) => row);
  for (let at = 0; at < count; at += 1) {
    const swap = at + Math.floor((next() * (rows - at)) / 4294967296);
    const held = order[at]; order[at] = order[swap]; order[swap] = held;
  }
  return order.slice(0, count).sort();
}

/** the rows pass `pass` reads, under a mode (see the header) */
export function msaDrawRows(mode, seed, pass, rows, count) {
  if (mode === "native") return msaNativeRows(seed, pass, rows);
  const take = Math.min(rows, count);
  if (mode === "prefix" || rows <= count) return Int32Array.from({ length: take }, (_, row) => row);
  if (mode !== "pass" && mode !== "once") throw new Error(`msaDraw: "${mode}" is not pass, once, prefix or native`);
  return msaDraw(seed, mode === "once" ? 0 : pass, rows, count);
}

/** whether a seed's fold applies protenix's MC dropout at all - once a run, at `odds` (dialect recycleMcDropoutOdds) */
export function mcDropoutApplies(seed, odds) {
  return odds > 0 && mix32((seedKey(seed) ^ 0x2545f491) >>> 0) < Math.floor(odds * 4294967296);
}

/** a pass's dropout key, which the trunks' kernels hash with each element's pair row and channel */
export function mcDropoutKey(seed, pass) {
  return mix32((seedKey(seed) + Math.imul(pass + 1, 0x9e3779b9) + 0x6d2b79f5) >>> 0);
}

/** the integer a hash must reach for its element to be kept, at dropout `rate` */
export function mcDropoutThreshold(rate) {
  return Math.floor(rate * 4294967296) >>> 0;
}

/**
 * Whether element (pair row `row`, channel `channel`) survives a pass's dropout - the hash every backend's kernel
 * computes (cuda/af3 mcDropoutK, metal/af3 af3_outer_sum, webgpu/af3 embedder.js). Kept elements are scaled by
 * 1 / (1 - rate), as torch's dropout scales them.
 */
export function mcDropoutKeep(key, row, channel, threshold) {
  return mix32((mix32((key ^ row) >>> 0) + channel) >>> 0) >= threshold;
}

/** a row-major [rows x width] array's chosen rows, as a new array of its type */
export function gatherRows(array, width, chosen) {
  const out = new array.constructor(chosen.length * width);
  for (let at = 0; at < chosen.length; at += 1) {
    out.set(array.subarray(chosen[at] * width, (chosen[at] + 1) * width), at * width);
  }
  return out;
}

/**
 * The pool as an exported input carries it: one int32 a cell, `code | deletion << 8 | mask << 16` (`batch.msaPacked`,
 * in place of `msa`, `msaMask` and `deletionMatrix` - cuda/featurise/entries.h writes the same). A third of the three
 * entries' 12 bytes, and the native trunks unpack only the rows a pass reads (cuda/af3 trunk.cuh, metal/af3 trunk.mm).
 * A code is AF3's 0-31, a deletion count is already AF3's int8 (<= 127), a mask 0 or 1.
 */
export function packMsa(msa, msaMask, deletionMatrix) {
  const packed = new Int32Array(msa.length);
  for (let at = 0; at < msa.length; at += 1) {
    packed[at] = (msa[at] & 0xff) | (Math.min(Math.max(deletionMatrix[at], 0), 255) << 8) | ((msaMask[at] !== 0 ? 1 : 0) << 16);
  }
  return packed;
}
