/**
 * AF3's confidence head on the GPU: pLDDT, PAE and PDE.
 *
 * This is the only part of AF3 that reads the SAMPLED COORDINATES back in.
 * Everything else runs before there are any; this looks at the atoms alongside
 * the trunk and predicts its own error.
 *
 *     pair += target_feat (both axes) + a distogram of the predicted structure
 *     4 x pairformer block - the same module the trunk runs 48 of
 *     pair   -> distance-error logits, SYMMETRISED -> PDE
 *           -> aligned-error logits, NOT symmetrised -> PAE
 *     single -> per-atom pLDDT logits, and per-atom resolved logits
 *
 * 🔴 pLDDT IS PER ATOM SLOT, NOT PER TOKEN. Its projection is (384, 24, 50):
 * one 50-bin distribution for each of a token's 24 dense atom slots. Reading it
 * as (384, 50) and broadcasting runs, produces plausible per-residue numbers,
 * and throws away the side-chain resolution that is the point of an atom-level
 * model.
 *
 * 🔴 "left" AND "right" ARE THE OTHER WAY ROUND. AF3 writes
 * left_target_feat_project(tf) with no axis expansion and
 * right_target_feat_project(tf)[:, None] with one - so the LEFT projection is
 * indexed by j and the right by i. Swapping them transposes a term nothing
 * downstream complains about.
 *
 * 🔴 PDE IS SYMMETRISED AND PAE IS NOT. PDE adds the logits to their own
 * transpose (one projection used twice; AF3 sets right = left). PAE is
 * directional - "how wrong is j when aligned on i" - and symmetrising it is a
 * plausible-looking tidy-up that destroys what it measures.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { Af3PairformerStackGpu } from "./pairformer-block-webgpu.js";
import { GRID_WIDTH } from "./pair-track-gpu.js";
import { tmPerBinFor, tmScoreD0 } from "../heads/tm-score.js";

const NUM_BINS = 64;
const MAX_ERROR_BIN = 31.0;
const PLDDT_BINS = 50;
const DGRAM_BINS = 39;
const DGRAM_MIN = 3.25;
const DGRAM_MAX = 50.75;

const EMBED_ORDER = ["leftTargetFeatProject", "rightTargetFeatProject", "distogramFeatProject"];
const HEAD_ORDER = [
  "logitsLnScale", "logitsLnOffset", "leftHalfDistanceLogits",
  "paeLogitsLnScale", "paeLogitsLnOffset", "paeLogits",
  "plddtLnScale", "plddtLnOffset", "plddtLogits",
  "resolvedLnScale", "resolvedLnOffset", "experimentallyResolvedLogits",
];

function pack(weights, order, label) {
  const offsets = {};
  let total = 0;
  for (const name of order) {
    if (weights[name] === undefined) throw new Error(`${label} weights missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of order) data.set(weights[name], offsets[name]);
  return { data, offsets };
}

/** Bin centres for an error head: `bins - 1` edges, plus a catch-all. */
function errorBinCentres() {
  const step = MAX_ERROR_BIN / (NUM_BINS - 2);
  const centres = new Float64Array(NUM_BINS);
  for (let bin = 0; bin < NUM_BINS - 1; bin += 1) centres[bin] = bin * step + step / 2;
  centres[NUM_BINS - 1] = centres[NUM_BINS - 2] + step;
  return centres;
}

export function createConfidenceShaders(shape, embedOffsets, headOffsets, epsilon, variance) {
  const { tokens, pairChannels, singleChannels, targetFeatWidth, dense } = shape;
  const pairs = tokens * tokens;
  const centres = errorBinCentres();
  const plddtCentres = [];
  for (let bin = 0; bin < PLDDT_BINS; bin += 1) {
    plddtCentres.push((0.5 / PLDDT_BINS + bin / PLDDT_BINS).toString());
  }
  // Squared lower edges, which is AF3's own spelling and avoids a square root.
  const lower = [];
  for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
    const edge = DGRAM_MIN + (DGRAM_MAX - DGRAM_MIN) * bin / (DGRAM_BINS - 1);
    lower.push((edge * edge).toString());
  }

  const common = `
const TOKENS: u32 = ${tokens}u;
const PAIRS: u32 = ${pairs}u;
const C_Z: u32 = ${pairChannels}u;
const C_S: u32 = ${singleChannels}u;
const TARGET_WIDTH: u32 = ${targetFeatWidth}u;
const DENSE: u32 = ${dense}u;
const BINS: u32 = ${NUM_BINS}u;
const PLDDT_BINS: u32 = ${PLDDT_BINS}u;
const DGRAM_BINS: u32 = ${DGRAM_BINS}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
`;

  const varianceCode = (count, read) => variance === "fast"
    ? `let variance = squares / f32(${count}) - mean * mean;`
    : `var variance = 0.0;
  for (var c = 0u; c < ${count}; c += 1u) {
    let d = ${read} - mean;
    variance += d * d;
  }
  variance /= f32(${count});`;

  // pair += left_j + right_i + the predicted structure's distogram.
  // 🔴 THE TARGET-FEATURE PROJECTION IS PER TOKEN AND WAS COMPUTED PER PAIR.
  // `embed` gave one thread each (i, j) and had it contract all 447 target
  // features into all 128 pair channels, twice - once for i and once for j -
  // so the same per-token projection was recomputed for every pair that token
  // appears in. That is TOKENS times too much work: at 150 tokens, 2.57 GMAC
  // where 17 M would do, and it grows as L^3 where the rest of the head grows
  // as L^2. It measured 64 ms of a 289 ms head at 150 tokens and would have
  // been most of it at 300.
  //
  // src/af3/confidence-reference.js has always done it this way - two calls to
  // `linear` over TOKENS rows, then broadcast into the pair - so this brings
  // the GPU TOWARDS the reference rather than away from it.
  const embedProject = `${common}
const W_LEFT: u32 = ${embedOffsets.leftTargetFeatProject}u;
const W_RIGHT: u32 = ${embedOffsets.rightTargetFeatProject}u;

@group(0) @binding(0) var<storage, read> target_feat: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> left: array<f32>;
@group(0) @binding(3) var<storage, read_write> right: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x + id.y * GRID_WIDTH * 64u;
  if (slot >= TOKENS * C_Z) { return; }
  let token = slot / C_Z;
  let c = slot % C_Z;
  var left_total = 0.0;
  var right_total = 0.0;
  // ...one read of the feature serving both projections, which is why they
  // share a kernel rather than being two dispatches.
  for (var f = 0u; f < TARGET_WIDTH; f += 1u) {
    let feature = target_feat[token * TARGET_WIDTH + f];
    left_total += feature * weights[W_LEFT + f * C_Z + c];
    right_total += feature * weights[W_RIGHT + f * C_Z + c];
  }
  left[slot] = left_total;
  right[slot] = right_total;
}`;

  const embed = `${common}
const LOWER = array<f32, ${DGRAM_BINS}>(${lower.join(", ")});
const W_LEFT: u32 = ${embedOffsets.leftTargetFeatProject}u;
const W_RIGHT: u32 = ${embedOffsets.rightTargetFeatProject}u;
const W_DGRAM: u32 = ${embedOffsets.distogramFeatProject}u;

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> pseudo_beta: array<f32>;
@group(0) @binding(2) var<storage, read> pair_mask: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> pair: array<f32>;
@group(0) @binding(5) var<storage, read> right: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let i = row / TOKENS;
  let j = row % TOKENS;

  // 🔴 THE COMPARISON IS ON SQUARED DISTANCES against squared edges, and it is
  // STRICTLY greater - so a pair inside 3.25 A satisfies no bin at all and its
  // row is entirely zero, indistinguishable from a masked pair. The diagonal is
  // always this case. Clamping it into bin 0 changes what the head is fed.
  var squared = 0.0;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let difference = pseudo_beta[i * 3u + axis] - pseudo_beta[j * 3u + axis];
    squared += difference * difference;
  }
  var bin = -1;
  for (var b = 0u; b < DGRAM_BINS; b += 1u) {
    // The final bin's top is 1e8, so everything past 50.75 A lands in it.
    var upper = 1.0e8;
    if (b + 1u < DGRAM_BINS) { upper = LOWER[b + 1u]; }
    if (squared > LOWER[b] && squared < upper) { bin = i32(b); }
  }
  let keep = pair_mask[row];

  for (var c = 0u; c < C_Z; c += 1u) {
    // ...left is indexed by j and right by i. See the note at the top.
    var value = pair[row * C_Z + c] + left[j * C_Z + c] + right[i * C_Z + c];
    if (bin >= 0) {
      value += keep * weights[W_DGRAM + u32(bin) * C_Z + c];
    }
    pair[row * C_Z + c] = value;
  }
}`;

  const centreList = Array.from(centres, (value) => value.toString()).join(", ");
  // 🔴 d0 IS BAKED IN, WHICH IS SOUND ONLY BECAUSE TOKENS ALREADY IS. The whole
  // shader is specialised on `tokens` (see TOKENS above) and d0 is a function of
  // nothing else, so it cannot go stale independently. Both pTM and ipTM use
  // this same global d0 - the interface score narrows which PAIRS are averaged,
  // not what d0 is.
  const tmPerBin = tmPerBinFor(centres, tmScoreD0(tokens));
  const tmList = Array.from(tmPerBin, (value) => value.toString()).join(", ");
  // 🔴 IT RECOMPUTED THE LAYERNORM FOR EVERY BIN, 192 TIMES A ROW. `half_logit`
  // took a (row, bin) and derived the row's mean, its variance and its whole
  // normalised activation before projecting ONE bin out of it - and it was
  // called three times per bin (the PDE's two halves and the PAE) across
  // sixty-four bins. That is three normalisations' worth of work done 64 times
  // each: about 246,000 operations a row where 70,000 would do, all of it on a
  // SINGLE thread, because the kernel gave one invocation the whole row.
  //
  // It is now shaped like `singleHeads` below, which had it right: a workgroup
  // a row, the reductions cooperative, the three normalised rows staged once,
  // and a lane to a bin. 33.6 -> a fraction of it, and the arithmetic is the
  // same arithmetic - each half still accumulates its own sum over channels in
  // the same order, and the two are added at the end as they were.
  const pairHeads = `${common}
const CENTRES = array<f32, ${NUM_BINS}>(${centreList});
const TM_PER_BIN = array<f32, ${NUM_BINS}>(${tmList});
const W_LN_SCALE: u32 = ${headOffsets.logitsLnScale}u;
const W_LN_OFFSET: u32 = ${headOffsets.logitsLnOffset}u;
const W_HALF: u32 = ${headOffsets.leftHalfDistanceLogits}u;
const W_PAE_SCALE: u32 = ${headOffsets.paeLogitsLnScale}u;
const W_PAE_OFFSET: u32 = ${headOffsets.paeLogitsLnOffset}u;
const W_PAE: u32 = ${headOffsets.paeLogits}u;

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> pair_mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> pde: array<f32>;
@group(0) @binding(4) var<storage, read_write> pae: array<f32>;
// 🔴 THE TM TERM, NOT THE LOGITS. pTM and ipTM need the whole PAE distribution,
// and this shader is the only place it exists - the head keeps the expectation
// alone. Reading the logits back would be tokens^2 * 64 floats; the term they
// reduce to is tokens^2, which is 64x smaller and is all either score wants.
@group(0) @binding(5) var<storage, read_write> tm_adjusted: array<f32>;

// The row's normalised activations, staged once: the PDE's own normalisation of
// this row and of its transpose, and the PAE's of this row.
var<workgroup> norm_row: array<f32, ${pairChannels}>;
var<workgroup> norm_transposed: array<f32, ${pairChannels}>;
var<workgroup> norm_pae: array<f32, ${pairChannels}>;
var<workgroup> distance: array<f32, ${NUM_BINS}>;
var<workgroup> aligned: array<f32, ${NUM_BINS}>;
var<workgroup> reduce_a: array<f32, 64>;
var<workgroup> reduce_b: array<f32, 64>;

/** Mean and inverse standard deviation of one pair row, cooperatively. */
fn row_statistics(base: u32, local: u32) -> vec2<f32> {
  var total = 0.0;
  var squares = 0.0;
  for (var c = local; c < C_Z; c += 64u) {
    let value = pair[base + c];
    total += value;
    squares += value * value;
  }
  reduce_a[local] = total;
  reduce_b[local] = squares;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) {
      reduce_a[local] += reduce_a[local + stride];
      reduce_b[local] += reduce_b[local + stride];
    }
    workgroupBarrier();
  }
  let mean = reduce_a[0] / f32(C_Z);
  let variance = reduce_b[0] / f32(C_Z) - mean * mean;
  // 🔴 A BARRIER BEFORE THE CALLER REUSES THE REDUCTION BUFFER, for the reason
  // singleHeads gives: every lane has just read slot 0, and a fast one writing
  // it again while a slow one still reads gives a wrong mean in some rows some
  // of the time.
  workgroupBarrier();
  return vec2<f32>(mean, inverseSqrt(variance + EPSILON));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= PAIRS) { return; }
  let i = row / TOKENS;
  let j = row % TOKENS;
  let transposed = j * TOKENS + i;
  let local = local_id.x;

  let own = row_statistics(row * C_Z, local);
  let other = row_statistics(transposed * C_Z, local);
  for (var c = local; c < C_Z; c += 64u) {
    let centred = (pair[row * C_Z + c] - own.x) * own.y;
    norm_row[c] = centred * weights[W_LN_SCALE + c] + weights[W_LN_OFFSET + c];
    norm_transposed[c] = (pair[transposed * C_Z + c] - other.x) * other.y
      * weights[W_LN_SCALE + c] + weights[W_LN_OFFSET + c];
    // ...the PAE normalises the SAME row with its own scale and offset, so the
    // centred value is shared and only the affine part differs.
    norm_pae[c] = centred * weights[W_PAE_SCALE + c] + weights[W_PAE_OFFSET + c];
  }
  workgroupBarrier();

  // A lane to a bin. The two halves of the PDE keep their own sums and are
  // added at the end, which is the order half_logit produced them in.
  for (var b = local; b < BINS; b += 64u) {
    var half_own = 0.0;
    var half_other = 0.0;
    var pae_total = 0.0;
    for (var c = 0u; c < C_Z; c += 1u) {
      let half_weight = weights[W_HALF + c * BINS + b];
      half_own += norm_row[c] * half_weight;
      half_other += norm_transposed[c] * half_weight;
      pae_total += norm_pae[c] * weights[W_PAE + c * BINS + b];
    }
    distance[b] = half_own + half_other;
    aligned[b] = pae_total;
  }
  workgroupBarrier();

  // 🔴 THE THREE EXPECTATIONS ARE COMPUTED ON ONE LANE, DELIBERATELY. They are
  // sixty-four values each and the reduction that would parallelise them costs
  // more barriers than the arithmetic is worth - but more than that, doing them
  // serially keeps the summation order the CPU reference uses, and this head's
  // checker compares against it at 1e-6 with a rounding envelope a fifth of
  // that.
  if (local == 0u) {
    var largest = -3.0e38;
    for (var b = 0u; b < BINS; b += 1u) { largest = max(largest, distance[b]); }
    var total = 0.0;
    var weighted = 0.0;
    for (var b = 0u; b < BINS; b += 1u) {
      let probability = exp(distance[b] - largest);
      total += probability;
      weighted += probability * CENTRES[b];
    }
    pde[row] = pair_mask[row] * (weighted / total);

    var pae_largest = -3.0e38;
    for (var b = 0u; b < BINS; b += 1u) { pae_largest = max(pae_largest, aligned[b]); }
    var pae_sum = 0.0;
    var pae_weighted = 0.0;
    for (var b = 0u; b < BINS; b += 1u) {
      let probability = exp(aligned[b] - pae_largest);
      pae_sum += probability;
      pae_weighted += probability * CENTRES[b];
    }
    pae[row] = pair_mask[row] * (pae_weighted / pae_sum);

    // 🔴 UNMASKED, because the reduction masks. Multiplying by pair_mask here
    // would fold masked pairs into the row mean as zeros rather than leaving
    // them out of it, which quietly lowers every score on a padded input.
    var tm_term = 0.0;
    for (var b = 0u; b < BINS; b += 1u) {
      tm_term += (exp(aligned[b] - pae_largest) / pae_sum) * TM_PER_BIN[b];
    }
    tm_adjusted[row] = tm_term;
  }
}`;

  const singleHeads = `${common}
const PLDDT_CENTRES = array<f32, ${PLDDT_BINS}>(${plddtCentres.join(", ")});
const W_PLDDT_SCALE: u32 = ${headOffsets.plddtLnScale}u;
const W_PLDDT_OFFSET: u32 = ${headOffsets.plddtLnOffset}u;
const W_PLDDT: u32 = ${headOffsets.plddtLogits}u;
const W_RESOLVED_SCALE: u32 = ${headOffsets.resolvedLnScale}u;
const W_RESOLVED_OFFSET: u32 = ${headOffsets.resolvedLnOffset}u;
const W_RESOLVED: u32 = ${headOffsets.experimentallyResolvedLogits}u;

@group(0) @binding(0) var<storage, read> single: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> plddt: array<f32>;
@group(0) @binding(3) var<storage, read_write> resolved: array<f32>;

var<workgroup> plddt_norm: array<f32, ${singleChannels}>;
var<workgroup> resolved_norm: array<f32, ${singleChannels}>;
var<workgroup> reduce_a: array<f32, 64>;
var<workgroup> reduce_b: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x;
  if (token >= TOKENS) { return; }
  let local = local_id.x;
  let base = token * C_S;

  var total = 0.0;
  var squares = 0.0;
  for (var c = local; c < C_S; c += 64u) {
    let value = single[base + c];
    total += value;
    squares += value * value;
  }
  reduce_a[local] = total;
  reduce_b[local] = squares;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) {
      reduce_a[local] += reduce_a[local + stride];
      reduce_b[local] += reduce_b[local + stride];
    }
    workgroupBarrier();
  }
  let mean = reduce_a[0] / f32(C_S);
  let squares_total = reduce_b[0];
  ${variance === "fast"
    ? "let variance = squares_total / f32(C_S) - mean * mean;"
    : `var centered = 0.0;
  for (var c = local; c < C_S; c += 64u) {
    let d = single[base + c] - mean;
    centered += d * d;
  }
  // 🔴 A BARRIER BEFORE REUSING THE REDUCTION BUFFER. Every invocation has
  // just read reduce_a[0] for the mean; writing reduce_a[local] without a
  // barrier lets a fast lane clobber slot 0 while a slow one is still reading
  // it. The result is a WRONG MEAN in some rows, some of the time - which
  // reads as a numerical problem, not a race.
  workgroupBarrier();
  reduce_a[local] = centered;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_a[local] += reduce_a[local + stride]; }
    workgroupBarrier();
  }
  let variance = reduce_a[0] / f32(C_S);`}
  let inverse_std = inverseSqrt(variance + EPSILON);
  workgroupBarrier();
  for (var c = local; c < C_S; c += 64u) {
    let centered = (single[base + c] - mean) * inverse_std;
    plddt_norm[c] = centered * weights[W_PLDDT_SCALE + c] + weights[W_PLDDT_OFFSET + c];
    resolved_norm[c] = centered * weights[W_RESOLVED_SCALE + c] + weights[W_RESOLVED_OFFSET + c];
  }
  workgroupBarrier();

  // 🔴 ONE DISTRIBUTION PER ATOM SLOT. The projection is (C_S, DENSE, BINS).
  for (var slot = local; slot < DENSE; slot += 64u) {
    var values: array<f32, ${PLDDT_BINS}>;
    var largest = -3.0e38;
    for (var b = 0u; b < PLDDT_BINS; b += 1u) {
      var logit = 0.0;
      for (var c = 0u; c < C_S; c += 1u) {
        logit += plddt_norm[c] * weights[W_PLDDT + c * DENSE * PLDDT_BINS + slot * PLDDT_BINS + b];
      }
      values[b] = logit;
      largest = max(largest, logit);
    }
    var total_p = 0.0;
    var weighted = 0.0;
    for (var b = 0u; b < PLDDT_BINS; b += 1u) {
      let probability = exp(values[b] - largest);
      total_p += probability;
      weighted += probability * PLDDT_CENTRES[b];
    }
    plddt[token * DENSE + slot] = 100.0 * weighted / total_p;

    for (var b = 0u; b < 2u; b += 1u) {
      var logit = 0.0;
      for (var c = 0u; c < C_S; c += 1u) {
        logit += resolved_norm[c] * weights[W_RESOLVED + c * DENSE * 2u + slot * 2u + b];
      }
      resolved[(token * DENSE + slot) * 2u + b] = logit;
    }
  }
}`;

  return { embedProject, embed, pairHeads, singleHeads };
}

export class Af3ConfidenceHeadGpu {
  constructor(device, options = {}) {
    this.device = device;
    // 🔴 THIS HEAD'S FOUR BLOCKS STAY IN f32, AND THAT IS A DELIBERATE EXCEPTION
    // TO THE TRUNK'S DEFAULT. pLDDT and PAE are the numbers the page puts in
    // front of a user, and they are a softmax over 50 and 64 bins - the most
    // amplifying thing either model emits. Measured on check-af3-confidence,
    // pLDDT goes 1.16e-4 to 2.32e-2 with f16 weights and PAE 5.75e-6 to
    // 2.51e-3, where the trunk's own outputs move by a factor.
    //
    // Four blocks of 52 is about 14 MiB of a 740 MiB fold, so f32 here costs
    // almost nothing and keeps the two user-facing numbers checked at the
    // tolerance the f32 arithmetic actually reaches. A caller that wants the
    // memory can still ask.
    this.options = {
      stagedPrecision: "f32", weightPrecision: "f32", accumulatePrecision: "f32", ...options,
    };
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {{pair, single, targetFeat, pseudoBeta, seqMask, tokens, dense}} input
   *   `pseudoBeta` is the representative atom per token, already gathered.
   * @param {object} weights the embed and head tensors, plus `blocks`
   * @param {{swapTransposedBias: boolean}} dialect
   */
  async run(input, weights, dialect, options = {}) {
    const { tokens, dense, seqMask } = input;
    const pairChannels = weights.pairChannels;
    const singleChannels = weights.singleChannels;
    const targetFeatWidth = weights.targetFeatWidth;
    const pairs = tokens * tokens;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";

    const pairMask = new Float32Array(pairs);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
    }

    const embedPacked = pack(weights, EMBED_ORDER, "confidence embed");
    const headPacked = pack(weights, HEAD_ORDER, "confidence head");
    const shape = { tokens, pairChannels, singleChannels, targetFeatWidth, dense };
    const sources = createConfidenceShaders(
      shape, embedPacked.offsets, headPacked.offsets, epsilon, variance);
    const base = `af3-confidence:${tokens}:${dense}:${epsilon}:${variance}`;
    const compiled = {};
    for (const [name, source] of Object.entries(sources)) {
      compiled[name] = await this.pipelines.get(`${base}:${name}`, source);
    }

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    let embeddedPair;
    try {
      const pair = keep(this.allocator.upload("af3-conf.pair", Float32Array.from(input.pair),
        storage | GPUBufferUsage.COPY_SRC));
      const targetFeat = keep(this.allocator.upload("af3-conf.target", input.targetFeat, storage));
      const pseudoBeta = keep(this.allocator.upload("af3-conf.beta", input.pseudoBeta, storage));
      const maskBuffer = keep(this.allocator.upload("af3-conf.mask", pairMask, storage));
      const embedWeights = keep(this.allocator.upload("af3-conf.embed-w", embedPacked.data, storage));
      const readback = keep(this.allocator.allocate("af3-conf.rb-pair", pairs * pairChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      // ...one per TOKEN, not one per pair; see the note on the kernel.
      const left = keep(this.allocator.allocate(
        "af3-conf.left", tokens * pairChannels * 4, storage));
      const right = keep(this.allocator.allocate(
        "af3-conf.right", tokens * pairChannels * 4, storage));
      const encoder = this.device.createCommandEncoder({ label: "af3-confidence-embed" });
      const project = encoder.beginComputePass({ label: "embed-project" });
      project.setPipeline(compiled.embedProject);
      project.setBindGroup(0, this.device.createBindGroup({
        layout: compiled.embedProject.getBindGroupLayout(0),
        entries: [targetFeat, embedWeights, left, right].map(
          (allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
      }));
      const projectGroups = Math.ceil((tokens * pairChannels) / 64);
      project.dispatchWorkgroups(
        Math.min(projectGroups, GRID_WIDTH), Math.ceil(projectGroups / GRID_WIDTH));
      project.end();
      const pass = encoder.beginComputePass({ label: "embed" });
      pass.setPipeline(compiled.embed);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: compiled.embed.getBindGroupLayout(0),
        entries: [left, pseudoBeta, maskBuffer, embedWeights, pair, right].map(
          (allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
      }));
      const groups = Math.ceil(pairs / 64);
      pass.dispatchWorkgroups(Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH));
      pass.end();
      encoder.copyBufferToBuffer(pair.buffer, 0, readback.buffer, 0, pairs * pairChannels * 4);
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      embeddedPair = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }

    // The four confidence pairformer blocks: the same stack the trunk runs.
    const stack = await new Af3PairformerStackGpu(this.device, this.options).run(
      { pair: embeddedPair, single: Float32Array.from(input.single),
        pairMask, seqMask, tokens }, weights.blocks, dialect, options);

    return { ...(await this.#heads(stack, pairMask, input, weights, headPacked, compiled)),
             pair: stack.pair, single: stack.single, embeddedPair };
  }

  async #heads(stack, pairMask, input, weights, headPacked, compiled) {
    const { tokens, dense } = input;
    const pairs = tokens * tokens;
    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const pair = keep(this.allocator.upload("af3-conf.h-pair", stack.pair, storage));
      const single = keep(this.allocator.upload("af3-conf.h-single", stack.single, storage));
      const maskBuffer = keep(this.allocator.upload("af3-conf.h-mask", pairMask, storage));
      const weightBuffer = keep(this.allocator.upload("af3-conf.h-w", headPacked.data, storage));
      const pde = keep(this.allocator.allocate("af3-conf.pde", pairs * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const pae = keep(this.allocator.allocate("af3-conf.pae", pairs * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const tmAdjusted = keep(this.allocator.allocate("af3-conf.tm", pairs * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const plddt = keep(this.allocator.allocate("af3-conf.plddt", tokens * dense * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const resolved = keep(this.allocator.allocate("af3-conf.resolved", tokens * dense * 2 * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const readbacks = {};
      for (const [name, source, bytes] of [["pde", pde, pairs * 4], ["pae", pae, pairs * 4],
        ["tmAdjusted", tmAdjusted, pairs * 4],
        ["plddt", plddt, tokens * dense * 4],
        ["resolved", resolved, tokens * dense * 2 * 4]]) {
        readbacks[name] = { allocation: keep(this.allocator.allocate(`af3-conf.rb-${name}`, bytes,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)), source, bytes };
      }

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-confidence-heads" });
      const run = (label, pipeline, buffers, x, y = 1) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(x, y);
        pass.end();
      };
      // ...ONE WORKGROUP A PAIR ROW, not one thread. The kernel gave a single
      // invocation the whole row and re-derived its LayerNorm for every bin;
      // it now stages the row once and gives a lane to each bin, so the
      // dispatch counts rows. Folded through x and y as every pair grid here
      // is - 300 tokens is 90,000 of them.
      run("pair-heads", compiled.pairHeads, [pair, maskBuffer, weightBuffer, pde, pae, tmAdjusted],
          Math.min(pairs, GRID_WIDTH), Math.ceil(pairs / GRID_WIDTH));
      run("single-heads", compiled.singleHeads, [single, weightBuffer, plddt, resolved], tokens);
      for (const { allocation, source, bytes } of Object.values(readbacks)) {
        encoder.copyBufferToBuffer(source.buffer, 0, allocation.buffer, 0, bytes);
      }
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);

      const output = {};
      for (const [name, { allocation }] of Object.entries(readbacks)) {
        await allocation.buffer.mapAsync(GPUMapMode.READ);
        output[name] = new Float32Array(allocation.buffer.getMappedRange().slice(0));
        allocation.buffer.unmap();
      }
      return output;
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
