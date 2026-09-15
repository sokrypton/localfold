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
import { GpuBufferAllocator } from "../../runtime/allocator.js";
import { pipelineCacheForDevice } from "../../runtime/pipeline-cache.js";
import { Af3PairformerStackGpu } from "../trunk/pairformer-block-webgpu.js";
import { GRID_WIDTH } from "../trunk/pair-track-gpu.js";

/** The same values as a Float32Array, without copying one that already is. */
const asFloats = (values) =>
  (values instanceof Float32Array ? values : Float32Array.from(values));
import { tmPerBinFor, tmScoreD0 } from "../../heads/tm-score.js";
// 🔴 THE ONE THING THIS FILE SHARES WITH THE CPU REFERENCE, and it is a
// STATISTIC rather than a kernel: rf3's confidence inputs are normalised on the
// host before any of this runs. Sharing a kernel would make the differential
// agree with itself; sharing a mean and a variance cannot.
import { maskedGlobalNorm, RF3_S_INPUTS_WIDTH } from "./confidence-reference.js";

const NUM_BINS = 64;
const MAX_ERROR_BIN = 31.0;
const PLDDT_BINS = 50;
const DGRAM_BINS = 39;
const DGRAM_MIN = 3.25;
const DGRAM_MAX = 50.75;

const EMBED_ORDER = ["leftTargetFeatProject", "rightTargetFeatProject", "distogramFeatProject"];
// 🔴 protenix2 ADDS A SECOND, UNBINNED DISTANCE TERM and normalises the trunk
// single before any use. Both are chosen by the tensors being there.
const embedOrderFor = (weights) => (weights.distanceFeatProject === undefined
  ? EMBED_ORDER : [...EMBED_ORDER, "distanceFeatProject"]);
// 🔴 boltz2 REBUILDS z RATHER THAN ADDING TO IT, from nine terms under its own
// scope. See `boltz2Reembed` in confidence-reference.js, which this mirrors.
const REEMBED_ORDER = [
  "sInputsNormScale", "sInputsNormOffset", "sNormScale", "sNormOffset", "sInputToS",
  "zNormScale", "zNormOffset", "relPosProject", "tokenBondsProject",
  "tokenBondsTypeEmbed", "contactEncodingUnspecified",
  "leftTargetFeatProject", "rightTargetFeatProject",
  "sToZProdIn1", "sToZProdIn2", "sToZProdOut", "distogramFeatProject",
];
const BOLTZ2_DGRAM_BINS = 64;
const MAX_RELATIVE_IDX = 32;
const MAX_RELATIVE_CHAIN = 2;
const POSITION_BINS = 2 * MAX_RELATIVE_IDX + 2;
const HEAD_ORDER = [
  "logitsLnScale", "logitsLnOffset", "leftHalfDistanceLogits",
  "paeLogitsLnScale", "paeLogitsLnOffset", "paeLogits",
  "plddtLnScale", "plddtLnOffset", "plddtLogits",
  "resolvedLnScale", "resolvedLnOffset", "experimentallyResolvedLogits",
];
/**
 * 🔴 A HEAD LayerNorm A BUNDLE DOES NOT CARRY IS NO LayerNorm, not one at scale
 * one: normalising still re-centres and rescales. boltz2 calls every logit head
 * directly on z and s, which is why those six tensors have no source in its
 * checkpoint - and why this cannot be a zero weight.
 *
 * 🔴 AND IT SPLITS EACH PAIR HEAD IN TWO, intra-chain and inter-chain, masked
 * hard rather than blended. On a MONOMER the inter head never fires, so a
 * single-chain gate cannot see whether it is implemented at all.
 */
const headOrderFor = (weights) => [
  ...(weights.logitsLnScale === undefined
    ? ["leftHalfDistanceLogits", "paeLogits", "plddtLogits",
       "experimentallyResolvedLogits"]
    : HEAD_ORDER),
  ...(weights.interHalfDistanceLogits === undefined
    ? [] : ["interHalfDistanceLogits", "paeInterLogits"]),
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

/**
 * protenix2's trunk single, clamped and LayerNormed before ANY use.
 *
 * 🔴 ON THE HOST, DELIBERATELY. It is tokens x 384 - 26k values at 68 tokens,
 * 115k at 300 - and it is the only thing between the trunk and the confidence
 * pairformer, which is a 4-block stack. A dispatch for it would cost a pass and
 * a readback to save a loop that does not appear in any profile. Every other
 * model gets the trunk's single unchanged, which is what `undefined` means.
 */
function normalisedTrunkSingle(input, weights, tokens) {
  const single = asFloats(input.single);
  if (weights.inputSingleNormScale === undefined) return single;
  const channels = weights.singleChannels;
  const output = new Float32Array(single.length);
  for (let token = 0; token < tokens; token += 1) {
    const base = token * channels;
    let total = 0;
    for (let c = 0; c < channels; c += 1) {
      total += Math.min(512, Math.max(-512, single[base + c]));
    }
    const mean = total / channels;
    let variance = 0;
    for (let c = 0; c < channels; c += 1) {
      const d = Math.min(512, Math.max(-512, single[base + c])) - mean;
      variance += d * d;
    }
    const inverse = 1 / Math.sqrt(variance / channels + 1e-5);
    for (let c = 0; c < channels; c += 1) {
      output[base + c] = (Math.min(512, Math.max(-512, single[base + c])) - mean) * inverse
        * weights.inputSingleNormScale[c] + weights.inputSingleNormOffset[c];
    }
  }
  return output;
}

export function createConfidenceShaders(shape, embedOffsets, headOffsets, epsilon, variance,
                                        reembedOffsets = null) {
  // Which of the three head shapes this bundle asks for; see `headOrderFor`.
  const headNorm = headOffsets.logitsLnScale !== undefined;
  const splitHeads = headOffsets.interHalfDistanceLogits !== undefined;
  const preSymmetrised = shape.preSymmetrisedPde === true;
  const { tokens, pairChannels, singleChannels, targetFeatWidth, dense } = shape;
  const pairs = tokens * tokens;
  const centres = errorBinCentres();
  const plddtCentres = [];
  for (let bin = 0; bin < PLDDT_BINS; bin += 1) {
    plddtCentres.push((0.5 / PLDDT_BINS + bin / PLDDT_BINS).toString());
  }
  // 🔴 TWO BINNINGS, AND THE BIN COUNT COMES OFF THE WEIGHT. AlphaFold 3 has 39
  // EDGES and a bin per edge, the last catching everything past 50.75; rf3 has
  // 39 BOUNDARIES and 40 bins, the index being how many of them the distance
  // exceeds - so it has a bin BELOW 3.25 that AF3 has no equivalent of. See
  // caDistogramFeatures in confidence-reference.js.
  const caDgram = shape.confidenceCaDgram === true;
  const dgramBins = caDgram ? 40 : DGRAM_BINS;
  // Squared lower edges, which is AF3's own spelling and avoids a square root.
  const lower = [];
  for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
    const edge = DGRAM_MIN + (DGRAM_MAX - DGRAM_MIN) * bin / (DGRAM_BINS - 1);
    lower.push((edge * edge).toString());
  }
  // ...and rf3's, squared the same way. Its boundaries divide by 39 where AF3's
  // divide by 38, so they are NOT a prefix of the table above.
  const caBounds = [];
  for (let at = 0; at < dgramBins - 1; at += 1) {
    const edge = DGRAM_MIN + at * ((DGRAM_MAX - DGRAM_MIN) / (dgramBins - 1));
    caBounds.push((edge * edge).toString());
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
const DGRAM_BINS: u32 = ${dgramBins}u;
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
  // src/af3/confidence/confidence-reference.js has always done it this way - two calls to
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
${caDgram
  ? `const CA_BOUNDS = array<f32, ${caBounds.length}>(${caBounds.join(", ")});`
  : `const LOWER = array<f32, ${DGRAM_BINS}>(${lower.join(", ")});`}
const W_LEFT: u32 = ${embedOffsets.leftTargetFeatProject}u;
const W_RIGHT: u32 = ${embedOffsets.rightTargetFeatProject}u;
const W_DGRAM: u32 = ${embedOffsets.distogramFeatProject}u;
${embedOffsets.distanceFeatProject === undefined ? ""
  : `const W_DISTANCE: u32 = ${embedOffsets.distanceFeatProject}u;`}

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
${caDgram ? `  // rf3: the bin is how many boundaries the distance is past, so EVERY pair
  // lands in one - there is no -1 here and no top-bin special case.
  var bin = 0;
  for (var b = 0u; b < DGRAM_BINS - 1u; b += 1u) {
    if (squared > CA_BOUNDS[b]) { bin += 1; }
  }` : `  var bin = -1;
  for (var b = 0u; b < DGRAM_BINS; b += 1u) {
    // The final bin's top is 1e8, so everything past 50.75 A lands in it.
    var upper = 1.0e8;
    if (b + 1u < DGRAM_BINS) { upper = LOWER[b + 1u]; }
    if (squared > LOWER[b] && squared < upper) { bin = i32(b); }
  }`}
  let keep = pair_mask[row];

  for (var c = 0u; c < C_Z; c += 1u) {
    // ...left is indexed by j and right by i. See the note at the top.
    var value = pair[row * C_Z + c] + left[j * C_Z + c] + right[i * C_Z + c];
    if (bin >= 0) {
      value += keep * weights[W_DGRAM + u32(bin) * C_Z + c];
    }
${embedOffsets.distanceFeatProject === undefined ? ""
  : `    // protenix2's second, UNBINNED distance term - and it is not masked,
    // exactly as the binned one above is.
    value += sqrt(squared + 1.0e-10) * weights[W_DISTANCE + c];`}
    pair[row * C_Z + c] = value;
  }
}`;

  // 🔴 boltz2's RE-EMBEDDING, IN TWO PASSES FOR THE REASON `embedProject` GIVES.
  // Five of its nine terms are per-TOKEN projections of one LayerNormed
  // s_inputs; computing them per PAIR would redo each of them TOKENS times, and
  // the outer-product term would redo two of them twice over. This pass writes
  // s_inputs, the two pair-axis projections, the two outer-product factors and
  // the rebuilt single; the pair pass below reads them.
  const reembedProject = reembedOffsets === null ? null : `${common}
const W_SI_SCALE: u32 = ${reembedOffsets.sInputsNormScale}u;
const W_SI_OFFSET: u32 = ${reembedOffsets.sInputsNormOffset}u;
const W_S_SCALE: u32 = ${reembedOffsets.sNormScale}u;
const W_S_OFFSET: u32 = ${reembedOffsets.sNormOffset}u;
const W_S_IN_TO_S: u32 = ${reembedOffsets.sInputToS}u;
const W_LEFT: u32 = ${reembedOffsets.leftTargetFeatProject}u;
const W_RIGHT: u32 = ${reembedOffsets.rightTargetFeatProject}u;
const W_PROD1: u32 = ${reembedOffsets.sToZProdIn1}u;
const W_PROD2: u32 = ${reembedOffsets.sToZProdIn2}u;

@group(0) @binding(0) var<storage, read> target_feat: array<f32>;
@group(0) @binding(1) var<storage, read> trunk_single: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
// 🔴 FOUR PLANES IN ONE BINDING, because WebGPU's default limit is EIGHT
// storage buffers a stage and the pair pass below needs eleven things. Four of
// them are the same shape and are written together here, so one buffer of
// 4 * TOKENS * C_Z costs nothing and takes the pair pass to exactly eight.
@group(0) @binding(3) var<storage, read_write> projections: array<f32>;
@group(0) @binding(4) var<storage, read_write> single: array<f32>;

var<workgroup> s_inputs: array<f32, ${targetFeatWidth}>;
var<workgroup> reduce_a: array<f32, 64>;
var<workgroup> reduce_b: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x;
  if (token >= TOKENS) { return; }
  let local = local_id.x;

  var total = 0.0;
  var squares = 0.0;
  for (var f = local; f < TARGET_WIDTH; f += 64u) {
    let value = target_feat[token * TARGET_WIDTH + f];
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
  let mean = reduce_a[0] / f32(TARGET_WIDTH);
  let variance = reduce_b[0] / f32(TARGET_WIDTH) - mean * mean;
  let inverse = inverseSqrt(variance + EPSILON);
  workgroupBarrier();
  for (var f = local; f < TARGET_WIDTH; f += 64u) {
    s_inputs[f] = (target_feat[token * TARGET_WIDTH + f] - mean) * inverse
      * weights[W_SI_SCALE + f] + weights[W_SI_OFFSET + f];
  }
  workgroupBarrier();

  for (var c = local; c < C_Z; c += 64u) {
    var l = 0.0; var r = 0.0; var p1 = 0.0; var p2 = 0.0;
    for (var f = 0u; f < TARGET_WIDTH; f += 1u) {
      let value = s_inputs[f];
      l += value * weights[W_LEFT + f * C_Z + c];
      r += value * weights[W_RIGHT + f * C_Z + c];
      p1 += value * weights[W_PROD1 + f * C_Z + c];
      p2 += value * weights[W_PROD2 + f * C_Z + c];
    }
    let plane = TOKENS * C_Z;
    projections[token * C_Z + c] = l;
    projections[plane + token * C_Z + c] = r;
    projections[2u * plane + token * C_Z + c] = p1;
    projections[3u * plane + token * C_Z + c] = p2;
  }

  // ...and the single track: its own LayerNorm plus a projection of s_inputs.
  var s_total = 0.0;
  var s_squares = 0.0;
  for (var c = local; c < C_S; c += 64u) {
    let value = trunk_single[token * C_S + c];
    s_total += value;
    s_squares += value * value;
  }
  workgroupBarrier();
  reduce_a[local] = s_total;
  reduce_b[local] = s_squares;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) {
      reduce_a[local] += reduce_a[local + stride];
      reduce_b[local] += reduce_b[local + stride];
    }
    workgroupBarrier();
  }
  let s_mean = reduce_a[0] / f32(C_S);
  let s_variance = reduce_b[0] / f32(C_S) - s_mean * s_mean;
  let s_inverse = inverseSqrt(s_variance + EPSILON);
  for (var c = local; c < C_S; c += 64u) {
    var value = (trunk_single[token * C_S + c] - s_mean) * s_inverse
      * weights[W_S_SCALE + c] + weights[W_S_OFFSET + c];
    for (var f = 0u; f < TARGET_WIDTH; f += 1u) {
      value += s_inputs[f] * weights[W_S_IN_TO_S + f * C_S + c];
    }
    single[token * C_S + c] = value;
  }
}`;

  // The pair half: z_norm(trunk z) plus eight terms. One workgroup per pair row,
  // because the outer-product term contracts C_Z into C_Z and is the only thing
  // here that is not a gather or a broadcast.
  const reembedPair = reembedOffsets === null ? null : `${common}
const W_Z_SCALE: u32 = ${reembedOffsets.zNormScale}u;
const W_Z_OFFSET: u32 = ${reembedOffsets.zNormOffset}u;
const W_RELPOS: u32 = ${reembedOffsets.relPosProject}u;
const W_BOND: u32 = ${reembedOffsets.tokenBondsProject}u;
const W_BOND_TYPE: u32 = ${reembedOffsets.tokenBondsTypeEmbed}u;
const W_CONTACT: u32 = ${reembedOffsets.contactEncodingUnspecified}u;
const W_PROD_OUT: u32 = ${reembedOffsets.sToZProdOut}u;
const W_DGRAM: u32 = ${reembedOffsets.distogramFeatProject}u;
const MAX_RELATIVE_IDX: i32 = ${MAX_RELATIVE_IDX};
const MAX_RELATIVE_CHAIN: i32 = ${MAX_RELATIVE_CHAIN};
const POSITION_BINS: u32 = ${POSITION_BINS}u;
const DGRAM64: u32 = ${BOLTZ2_DGRAM_BINS}u;

@group(0) @binding(0) var<storage, read> pair_in: array<f32>;
// The four per-token projections, in planes; see the pass above.
@group(0) @binding(1) var<storage, read> projections: array<f32>;
@group(0) @binding(2) var<storage, read> features: array<i32>;
@group(0) @binding(3) var<storage, read> bonds: array<f32>;
@group(0) @binding(4) var<storage, read> pseudo_beta: array<f32>;
@group(0) @binding(5) var<storage, read> pair_mask: array<f32>;
@group(0) @binding(6) var<storage, read> weights: array<f32>;
@group(0) @binding(7) var<storage, read_write> pair_out: array<f32>;
const PLANE: u32 = ${tokens * pairChannels}u;
fn left_at(t: u32, c: u32) -> f32 { return projections[t * C_Z + c]; }
fn right_at(t: u32, c: u32) -> f32 { return projections[PLANE + t * C_Z + c]; }
fn prod1_at(t: u32, c: u32) -> f32 { return projections[2u * PLANE + t * C_Z + c]; }
fn prod2_at(t: u32, c: u32) -> f32 { return projections[3u * PLANE + t * C_Z + c]; }

fn residue_index(t: u32) -> i32 { return features[t]; }
fn token_index(t: u32) -> i32 { return features[TOKENS + t]; }
fn asym_id(t: u32) -> i32 { return features[2u * TOKENS + t]; }
fn entity_id(t: u32) -> i32 { return features[3u * TOKENS + t]; }
fn sym_id(t: u32) -> i32 { return features[4u * TOKENS + t]; }
fn clamp_bin(value: i32, high: i32) -> i32 { return min(max(value, 0), high); }

var<workgroup> product: array<f32, ${pairChannels}>;
var<workgroup> reduce_a: array<f32, 64>;
var<workgroup> reduce_b: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= PAIRS) { return; }
  let i = row / TOKENS;
  let j = row % TOKENS;
  let local = local_id.x;

  var total = 0.0;
  var squares = 0.0;
  for (var c = local; c < C_Z; c += 64u) {
    let value = pair_in[row * C_Z + c];
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
  let inverse = inverseSqrt(variance + EPSILON);
  workgroupBarrier();

  for (var c = local; c < C_Z; c += 64u) {
    product[c] = prod1_at(i, c) * prod2_at(j, c);
  }
  workgroupBarrier();

  // The relative encoding's four active columns, the same buckets the embedder
  // resolves - one-hot, so the projection is a gather of rows.
  let same_chain = asym_id(i) == asym_id(j);
  let same_entity = entity_id(i) == entity_id(j);
  var bin_a = u32(2 * MAX_RELATIVE_IDX + 1);
  if (same_chain) {
    bin_a = u32(clamp_bin(residue_index(i) - residue_index(j) + MAX_RELATIVE_IDX,
                          2 * MAX_RELATIVE_IDX));
  }
  var bin_b = u32(2 * MAX_RELATIVE_IDX + 1);
  if (same_chain && residue_index(i) == residue_index(j)) {
    bin_b = u32(clamp_bin(token_index(i) - token_index(j) + MAX_RELATIVE_IDX,
                          2 * MAX_RELATIVE_IDX));
  }
  var bin_c = u32(2 * MAX_RELATIVE_CHAIN + 1);
  if (same_entity) {
    bin_c = u32(clamp_bin(sym_id(i) - sym_id(j) + MAX_RELATIVE_CHAIN,
                          2 * MAX_RELATIVE_CHAIN));
  }
  let row_b = POSITION_BINS + bin_b;
  let row_entity = POSITION_BINS * 2u;
  let row_c = POSITION_BINS * 2u + 1u + bin_c;

  // boltz2's own 64-bin distance embedding: 63 edges evenly over 2..22 A.
  var squared = 1.0e-10;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let difference = pseudo_beta[i * 3u + axis] - pseudo_beta[j * 3u + axis];
    squared += difference * difference;
  }
  let distance = sqrt(squared);
  var dbin = 0u;
  for (var edge = 0u; edge + 1u < DGRAM64; edge += 1u) {
    if (distance > 2.0 + 20.0 * f32(edge) / f32(DGRAM64 - 2u)) { dbin += 1u; }
  }
  let keep = pair_mask[row];
  let bond = bonds[row];
  let bond_row = u32(clamp(i32(bonds[PAIRS + row]), 0, 6));

  for (var c = local; c < C_Z; c += 64u) {
    var value = (pair_in[row * C_Z + c] - mean) * inverse * weights[W_Z_SCALE + c]
      + weights[W_Z_OFFSET + c]
      + weights[W_RELPOS + bin_a * C_Z + c]
      + weights[W_RELPOS + row_b * C_Z + c]
      + weights[W_RELPOS + row_c * C_Z + c]
      + bond * weights[W_BOND + c]
      + weights[W_BOND_TYPE + bond_row * C_Z + c]
      + weights[W_CONTACT + c]
      + right_at(i, c) + left_at(j, c)
      + keep * weights[W_DGRAM + dbin * C_Z + c];
    if (same_entity) { value += weights[W_RELPOS + row_entity * C_Z + c]; }
    for (var e = 0u; e < C_Z; e += 1u) {
      value += product[e] * weights[W_PROD_OUT + e * C_Z + c];
    }
    pair_out[row * C_Z + c] = value;
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
${headNorm ? `const W_LN_SCALE: u32 = ${headOffsets.logitsLnScale}u;
const W_LN_OFFSET: u32 = ${headOffsets.logitsLnOffset}u;
const W_PAE_SCALE: u32 = ${headOffsets.paeLogitsLnScale}u;
const W_PAE_OFFSET: u32 = ${headOffsets.paeLogitsLnOffset}u;` : ""}
const W_HALF: u32 = ${headOffsets.leftHalfDistanceLogits}u;
const W_PAE: u32 = ${headOffsets.paeLogits}u;
${splitHeads ? `const W_HALF_INTER: u32 = ${headOffsets.interHalfDistanceLogits}u;
const W_PAE_INTER: u32 = ${headOffsets.paeInterLogits}u;` : ""}

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
${splitHeads ? `// The chain identity, for the intra/inter split. Five rows of TOKENS, the
// layout the embedder uses; only asym_id is read here.
@group(0) @binding(6) var<storage, read> features: array<i32>;` : ""}

// The row's normalised activations, staged once: the PDE's own normalisation of
// this row and of its transpose, and the PAE's of this row.
var<workgroup> norm_row: array<f32, ${pairChannels}>;
var<workgroup> norm_transposed: array<f32, ${pairChannels}>;
var<workgroup> norm_pae: array<f32, ${pairChannels}>;
var<workgroup> distance: array<f32, ${NUM_BINS}>;
var<workgroup> aligned: array<f32, ${NUM_BINS}>;
var<workgroup> reduce_a: array<f32, 64>;
var<workgroup> reduce_b: array<f32, 64>;

/** The same, over a row already staged in workgroup memory. */
fn row_statistics_of(values: ptr<workgroup, array<f32, ${pairChannels}>>,
                     local: u32) -> vec2<f32> {
  var total = 0.0;
  var squares = 0.0;
  for (var c = local; c < C_Z; c += 64u) {
    let value = (*values)[c];
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
  workgroupBarrier();
  return vec2<f32>(mean, inverseSqrt(variance + EPSILON));
}

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

${preSymmetrised ? `  // 🔴 SYMMETRISED BEFORE THE HEAD, NOT AFTER IT. protenix2 and boltz2 add the
  // transpose to the ACTIVATION and project once; AF3 projects once and adds the
  // transpose of the RESULT. The two agree only where the projection is the same
  // on both halves, which under split heads it is not.
  for (var c = local; c < C_Z; c += 64u) {
    norm_row[c] = pair[row * C_Z + c] + pair[transposed * C_Z + c];
    norm_pae[c] = pair[row * C_Z + c];
  }
  workgroupBarrier();
${headNorm ? `  let sym = row_statistics_of(&norm_row, local);
  let own = row_statistics(row * C_Z, local);
  for (var c = local; c < C_Z; c += 64u) {
    norm_row[c] = (norm_row[c] - sym.x) * sym.y * weights[W_LN_SCALE + c]
      + weights[W_LN_OFFSET + c];
    norm_pae[c] = (pair[row * C_Z + c] - own.x) * own.y * weights[W_PAE_SCALE + c]
      + weights[W_PAE_OFFSET + c];
  }
  workgroupBarrier();` : ""}` : `  ${headNorm ? `let own = row_statistics(row * C_Z, local);
  let other = row_statistics(transposed * C_Z, local);` : ""}
  for (var c = local; c < C_Z; c += 64u) {
${headNorm ? `    let centred = (pair[row * C_Z + c] - own.x) * own.y;
    norm_row[c] = centred * weights[W_LN_SCALE + c] + weights[W_LN_OFFSET + c];
    norm_transposed[c] = (pair[transposed * C_Z + c] - other.x) * other.y
      * weights[W_LN_SCALE + c] + weights[W_LN_OFFSET + c];
    // ...the PAE normalises the SAME row with its own scale and offset, so the
    // centred value is shared and only the affine part differs.
    norm_pae[c] = centred * weights[W_PAE_SCALE + c] + weights[W_PAE_OFFSET + c];`
  : `    norm_row[c] = pair[row * C_Z + c];
    norm_transposed[c] = pair[transposed * C_Z + c];
    norm_pae[c] = pair[row * C_Z + c];`}
  }
  workgroupBarrier();`}

${splitHeads ? `  let same_chain = features[2u * TOKENS + i] == features[2u * TOKENS + j];` : ""}
  // A lane to a bin. The two halves of the PDE keep their own sums and are
  // added at the end, which is the order half_logit produced them in.
  for (var b = local; b < BINS; b += 64u) {
    var half_own = 0.0;
    var half_other = 0.0;
    var pae_total = 0.0;
${splitHeads ? `    var half_pick = W_HALF;
    var pae_pick = W_PAE;
    if (!same_chain) { half_pick = W_HALF_INTER; pae_pick = W_PAE_INTER; }` : ""}
    for (var c = 0u; c < C_Z; c += 1u) {
      let half_weight = weights[${splitHeads ? "half_pick" : "W_HALF"} + c * BINS + b];
      half_own += norm_row[c] * half_weight;
${preSymmetrised ? "" : "      half_other += norm_transposed[c] * half_weight;"}
      pae_total += norm_pae[c] * weights[${splitHeads ? "pae_pick" : "W_PAE"} + c * BINS + b];
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
${headNorm ? `const W_PLDDT_SCALE: u32 = ${headOffsets.plddtLnScale}u;
const W_PLDDT_OFFSET: u32 = ${headOffsets.plddtLnOffset}u;
const W_RESOLVED_SCALE: u32 = ${headOffsets.resolvedLnScale}u;
const W_RESOLVED_OFFSET: u32 = ${headOffsets.resolvedLnOffset}u;` : ""}
const W_PLDDT: u32 = ${headOffsets.plddtLogits}u;
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
${headNorm ? `    let centered = (single[base + c] - mean) * inverse_std;
    plddt_norm[c] = centered * weights[W_PLDDT_SCALE + c] + weights[W_PLDDT_OFFSET + c];
    resolved_norm[c] = centered * weights[W_RESOLVED_SCALE + c] + weights[W_RESOLVED_OFFSET + c];`
  : `    // boltz2 calls both heads directly on s; see headOrderFor.
    plddt_norm[c] = single[base + c];
    resolved_norm[c] = single[base + c];`}
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

  return { ...(reembedOffsets === null ? { embedProject, embed }
    : { reembedProject, reembedPair }),
    pairHeads, singleHeads };
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
    //
    // 🔴 AND THE MATRIX PAIR KERNELS ARE THE FOURTH AXIS, WHICH THIS PINNED
    // THREE OF FOR A YEAR. `triangleProjectMatrix`, `gridProjectMatrix` and
    // `gridAttendMatrix` replace the pair track's projections with kernels that
    // issue on f16 matrix units, and no precision option above reaches them -
    // so a head that had declared itself f32 was running three of its six
    // updates in f16 whenever the device prior turned them on. Measured on
    // check-af3-confidence: all four heads FAIL with them (pLDDT 1902x, PAE
    // 3463x, PDE 3718x, resolved 522x their conditioning envelope) and all four
    // pass without (7.0x and 7.3x on PAE and PDE). The trunk keeps them and
    // keeps the speed; see docs/AF3.md for the block-level ladder.
    this.options = {
      stagedPrecision: "f32", weightPrecision: "f32", accumulatePrecision: "f32",
      pairMatrixKernels: false, ...options,
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

    // 🔴 RoseTTAFold3's GLOBAL NORM, ON THE HOST, AND DELIBERATELY. It reduces
    // the WHOLE tensor to two scalars - one mean and one variance across every
    // token and every channel - so a GPU version is a full reduction, a
    // readback and a second pass, three dispatches to save an O(n) loop the
    // host runs once per FOLD rather than once per block. The pair is the big
    // one at `tokens^2 * 128`, which is 590k floats on a 68-mer: about a
    // millisecond here, against a denoiser that is seconds. Reuses the CPU
    // reference's own function, which is the one place this port and its
    // differential are allowed to share code - it is a statistic, not a kernel.
    if (dialect?.confidenceGlobalNorm === true) {
      input = { ...input,
                pair: maskedGlobalNorm(input.pair, pairMask, pairs, pairChannels),
                single: maskedGlobalNorm(input.single, seqMask, tokens, singleChannels),
                targetFeat: maskedGlobalNorm(input.targetFeat, seqMask, tokens,
                                             targetFeatWidth, RF3_S_INPUTS_WIDTH) };
    }

    // 🔴 THE HEAD'S SHAPE IS THE BUNDLE'S. boltz2 rebuilds z under its own scope,
    // normalises before no logit head, and splits both pair heads by chain; each
    // of the three is chosen by whether the tensors are there, and each changes
    // the generated WGSL, so all three are in the pipeline key.
    const reembedding = weights.reembed !== undefined;
    const embedPacked = reembedding
      ? pack(weights.reembed, REEMBED_ORDER, "confidence re-embed")
      : pack(weights, embedOrderFor(weights), "confidence embed");
    const headPacked = pack(weights, headOrderFor(weights), "confidence head");
    const shape = { tokens, pairChannels, singleChannels, targetFeatWidth, dense,
                    preSymmetrisedPde: dialect?.preSymmetrisedPde === true,
                    // rf3's 40-bin CA-CA embedding; see caDistogramFeatures.
                    confidenceCaDgram: dialect?.confidenceCaDgram === true };
    const sources = createConfidenceShaders(
      shape, reembedding ? {} : embedPacked.offsets, headPacked.offsets, epsilon, variance,
      reembedding ? embedPacked.offsets : null);
    const base = `af3-confidence:${tokens}:${dense}:${epsilon}:${variance}`
      + `:re${reembedding}:hn${headPacked.offsets.logitsLnScale !== undefined}`
      + `:sh${headPacked.offsets.interHalfDistanceLogits !== undefined}`
      + `:ps${shape.preSymmetrisedPde}:cd${shape.confidenceCaDgram}`;
    const compiled = {};
    for (const [name, source] of Object.entries(sources)) {
      compiled[name] = await this.pipelines.get(`${base}:${name}`, source);
    }

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    let embeddedPair;
    // boltz2 rebuilds the SINGLE as well as the pair; undefined elsewhere, and
    // the stack is then fed the trunk's own.
    let embeddedSingle;
    try {
      // 🔴 COPIED ONLY IF IT IS NOT ALREADY THE RIGHT ARRAY. `upload` writes
      // through queue.writeBuffer and does not mutate what it is given, so a
      // Float32Array can go straight in - and at 300 tokens this copy was 43.9
      // MiB of host allocation for nothing. The fallback stays because a
      // checker may hand this a plain array.
      const pair = keep(this.allocator.upload("af3-conf.pair", asFloats(input.pair),
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
      if (reembedding) {
        // boltz2's two passes: the per-token projections, then the pair.
        const projections = keep(this.allocator.allocate(
          "af3-conf.projections", 4 * tokens * pairChannels * 4, storage));
        const trunkSingle = keep(this.allocator.upload(
          "af3-conf.trunk-single", asFloats(input.single), storage));
        const rebuiltSingle = keep(this.allocator.allocate(
          "af3-conf.re-single", tokens * singleChannels * 4,
          storage | GPUBufferUsage.COPY_SRC));
        const featureData = new Int32Array(5 * tokens);
        ["residueIndex", "tokenIndex", "asymId", "entityId", "symId"]
          .forEach((name, index) => {
            const source = input.features?.[name];
            if (source === undefined) {
              throw new Error(`the boltz2 confidence re-embedding needs features.${name}`);
            }
            for (let t = 0; t < tokens; t += 1) featureData[index * tokens + t] = source[t];
          });
        const features = keep(this.allocator.upload("af3-conf.features", featureData, storage));
        // Two planes: the contact flag, then the bond ORDER - the same layout
        // the embedder's binding uses.
        const bondData = new Float32Array(pairs * 2);
        if (input.bondMatrix !== undefined) bondData.set(input.bondMatrix, 0);
        if (input.bondOrderMatrix !== undefined) bondData.set(input.bondOrderMatrix, pairs);
        const bonds = keep(this.allocator.upload("af3-conf.bonds", bondData, storage));
        const rebuiltPair = keep(this.allocator.allocate(
          "af3-conf.re-pair", pairs * pairChannels * 4,
          storage | GPUBufferUsage.COPY_SRC));
        const singleReadback = keep(this.allocator.allocate(
          "af3-conf.rb-single", tokens * singleChannels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
        const tokenPass = encoder.beginComputePass({ label: "reembed-project" });
        tokenPass.setPipeline(compiled.reembedProject);
        tokenPass.setBindGroup(0, this.device.createBindGroup({
          layout: compiled.reembedProject.getBindGroupLayout(0),
          entries: [targetFeat, trunkSingle, embedWeights, projections, rebuiltSingle].map(
            (allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
        }));
        tokenPass.dispatchWorkgroups(tokens);
        tokenPass.end();
        const pairPass = encoder.beginComputePass({ label: "reembed-pair" });
        pairPass.setPipeline(compiled.reembedPair);
        pairPass.setBindGroup(0, this.device.createBindGroup({
          layout: compiled.reembedPair.getBindGroupLayout(0),
          entries: [pair, projections, features, bonds, pseudoBeta,
                    maskBuffer, embedWeights, rebuiltPair].map(
            (allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
        }));
        pairPass.dispatchWorkgroups(
          Math.min(pairs, GRID_WIDTH), Math.ceil(pairs / GRID_WIDTH));
        pairPass.end();
        encoder.copyBufferToBuffer(rebuiltPair.buffer, 0, readback.buffer, 0,
                                   pairs * pairChannels * 4);
        encoder.copyBufferToBuffer(rebuiltSingle.buffer, 0, singleReadback.buffer, 0,
                                   tokens * singleChannels * 4);
        this.device.queue.submit([encoder.finish()]);
        const failure = await this.device.popErrorScope();
        if (failure !== null) throw new Error(`WebGPU validation failed: ${failure.message}`);
        await readback.buffer.mapAsync(GPUMapMode.READ);
        embeddedPair = new Float32Array(readback.buffer.getMappedRange().slice(0));
        readback.buffer.unmap();
        await singleReadback.buffer.mapAsync(GPUMapMode.READ);
        embeddedSingle = new Float32Array(singleReadback.buffer.getMappedRange().slice(0));
        singleReadback.buffer.unmap();
      } else {
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
      }
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }

    // The four confidence pairformer blocks: the same stack the trunk runs.
    const stack = await new Af3PairformerStackGpu(this.device, this.options).run(
      { pair: embeddedPair,
        single: embeddedSingle ?? normalisedTrunkSingle(input, weights, tokens),
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
      // ...and the chain identity where the heads split by it; a seventh
      // binding the other dialects' shader does not declare.
      const splitHeads = headPacked.offsets.interHalfDistanceLogits !== undefined;
      const headFeatures = splitHeads ? (() => {
        const data = new Int32Array(5 * tokens);
        const asymId = input.features?.asymId;
        if (asymId === undefined) {
          throw new Error("the boltz2 confidence heads split by chain and need features.asymId");
        }
        for (let t = 0; t < tokens; t += 1) data[2 * tokens + t] = asymId[t];
        return keep(this.allocator.upload("af3-conf.h-features", data, storage));
      })() : null;
      run("pair-heads", compiled.pairHeads,
          [pair, maskBuffer, weightBuffer, pde, pae, tmAdjusted,
           ...(splitHeads ? [headFeatures] : [])],
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
