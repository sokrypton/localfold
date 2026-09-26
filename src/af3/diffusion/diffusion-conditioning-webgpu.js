/**
 * AF3's diffusion conditioning: what the denoiser knows besides the atoms.
 *
 *     pair   = proj(LayerNorm([trunk pair | relative encoding]))  then 2 transitions
 *     single = proj(LayerNorm([trunk single | target_feat]))
 *              + proj(LayerNorm(fourier(noise level)))            then 2 transitions
 *
 * 🔴 THE RELATIVE ENCODING IS CONCATENATED RAW HERE, NOT PROJECTED. In the
 * embedder its 139 columns are projected and can be gathered; here they are
 * glued to the trunk pair's 128 and the LayerNorm runs over all 267 together.
 * So the one-hot columns change the NORMALISATION STATISTICS of the trunk pair
 * columns beside them - the two uses of the same feature are not
 * interchangeable, and treating this one as a projection drops that coupling.
 *
 * It still does not have to be materialised. Only three or four of the 139 are
 * ever set, so the row's sum, its sum of squares and its projection all have
 * closed forms:
 *
 *     count      = 3 + sameEntity
 *     sum        = sum(pair) + count
 *     sum sq dev = sum((pair - mean)^2) + count*(1-mean)^2 + (139-count)*mean^2
 *     relative's contribution to output o
 *                = inv * (sum over SET bins of scale*W  -  mean * S[o])
 *
 * where S[o] = sum over ALL 139 of scale*W is a constant of the weights,
 * computed once at setup. That keeps a tokens^2 x 267 tensor (96 MB at 300
 * tokens) from ever existing, and unlike the embedder's gather it is exact
 * rather than an optimisation of a sparse matmul.
 *
 * 🔴 THE NOISE LEVEL IS DIVIDED BY SIGMA_DATA BEFORE THE LOG. Feeding raw
 * angstroms gives a Fourier embedding of a number three orders too large, which
 * aliases across the schedule instead of separating it.
 *
 * 🔴 THE TWO TRANSITIONS ARE UNCONDITIONED - `conditionedTransition(x, null)` -
 * so they are an ordinary LayerNorm-SwiGLU pair with a bias-carrying norm, not
 * AdaLN. They reuse the trunk's transition shader with two-pass variance.
 */
import { createAddShader } from "../../runtime/execution.js";
import { GpuBufferAllocator } from "../../runtime/allocator.js";
import { pipelineCacheForDevice, settleAll } from "../../runtime/pipeline-cache.js";
import { residentWeightBuffer } from "../../runtime/resident.js";
import { deviceTuning } from "../../runtime/device-profile.js";
import { singleCondPadding, singleCondPaddingWgsl } from "../dialect.js";
import { createTransitionShader, packTransitionWeights, transitionRowTile, transitionWidth }
  from "../trunk/transition-webgpu.js";

/**
 * The host-side packing this module does to its weights, kept per weight
 * bundle rather than redone per call.
 *
 * 🔴 A SAMPLER CALLS THIS TWO HUNDRED TIMES DOWN ONE SCHEDULE, and none of
 * `relativeColumnSums`, `packTransitionWeights` or the noise packing reads the
 * noise level - they are functions of the WEIGHTS alone. Recomputing them per
 * step copied four transition weight bundles into fresh Float32Arrays and ran
 * a 128x128 column sum, once per step, for the identical answer. The key is
 * the weights object's identity: a fold holds one bundle for its whole life,
 * and a different bundle is a different model.
 */
const PREPARED_WEIGHTS = new WeakMap();

function prepareWeights(weights) {
  const cached = PREPARED_WEIGHTS.get(weights);
  if (cached !== undefined) return cached;
  const scale = weights.noiseEmbeddingInitialNormScale;
  const projection = weights.noiseEmbeddingInitialProjection;
  const norm = packNormWeights(scale, weights.noiseEmbeddingInitialNormOffset);
  const noiseData = new Float32Array(norm.length + projection.length);
  noiseData.set(norm, 0);
  noiseData.set(projection, norm.length);
  // 🔴 THE CLOSED FORM IS A PROPERTY OF THE JOINT LayerNorm, SO THE SPLIT
  // DIALECT HAS NO USE FOR IT - and asking for it there reads `scale[128+c]`
  // off the end of a 256-long norm and hands the shader a buffer of NaN.
  // OpenDDE's pair conditioning normalises each term separately; see the
  // `split` branch of createConditioningShaders.
  const split = weights.zTrunkProjection !== undefined;
  // ...and the third shape, where only the relpos is projected.
  const projectedRelpos = !split && weights.relpeProjection !== undefined;
  const prepared = {
    // Two vectors: the closed form's S[o], then the LayerNorm OFFSET's own
    // constant contribution over the same 139 columns. Zero where the bundle
    // carries no offset, which is every model but boltz2's family.
    columnSums: split ? new Float32Array(2 * weights.pairChannels)
      : (() => {
        const sums = relativeColumnSums(weights.pairCondInitialNormScale,
                                        weights.pairCondInitialProjection,
                                        weights.pairChannels, weights.pairChannels);
        const offsets = weights.pairCondInitialNormOffset == null
          ? new Float32Array(weights.pairChannels)
          : relativeOffsetSums(weights.pairCondInitialNormOffset,
                               weights.pairCondInitialProjection,
                               weights.pairChannels, weights.pairChannels);
        const packed = new Float32Array(sums.length + offsets.length);
        packed.set(sums, 0); packed.set(offsets, sums.length);
        return packed;
      })(),
    noisePacked: {
      data: noiseData,
      offsets: { noiseEmbeddingInitialNormScale: 0,
                 noiseEmbeddingInitialNormOffset: scale.length,
                 noiseEmbeddingInitialProjection: norm.length },
    },
    pairTransitions: weights.pairTransitions.map(
      (w) => packTransitionWeights(asTransitionWeights(w))),
    singleTransitions: weights.singleTransitions.map(
      (w) => packTransitionWeights(asTransitionWeights(w))),
  };
  PREPARED_WEIGHTS.set(weights, prepared);
  return prepared;
}

/**
 * A LayerNorm's weights as one buffer: the scale, then the offset.
 *
 * 🔴 ALWAYS BOTH, WITH ZEROS WHERE THE BUNDLE CARRIES NO OFFSET. An offset of
 * zero IS the scale-only LayerNorm - it is not a fallback, it is the identity -
 * so this needs no shader variant and no dialect flag, and a second bundle that
 * turns out to be affine cannot silently read the scale alone. Ten of boltz2's
 * diffusion LayerNorms are affine where AlphaFold 3's are scale-only; see
 * `offsetOf` in diffusion-weights.js.
 */
export function packNormWeights(scale, offset) {
  const packed = new Float32Array(scale.length * 2);
  packed.set(scale, 0);
  if (offset != null) {
    if (offset.length !== scale.length) {
      throw new Error(`LayerNorm offset is ${offset.length} channels and its `
        + `scale is ${scale.length}`);
    }
    packed.set(offset, scale.length);
  }
  return packed;
}

const GRID_WIDTH = 32_768;
const SIGMA_DATA = 16.0;
const MAX_RELATIVE_IDX = 32;
const MAX_RELATIVE_CHAIN = 2;
const POSITION_BINS = 2 * MAX_RELATIVE_IDX + 2;
const RELATIVE_WIDTH = POSITION_BINS * 2 + 1 + (2 * MAX_RELATIVE_CHAIN + 2);

/**
 * S[o] = sum over all 139 relative columns of scale[c] * W[c][o]. A constant of
 * the weights, so it is computed once here rather than per pair on the GPU.
 */
/**
 * O[o] = sum over the 139 relative columns of offset[c] * W[c][o].
 *
 * The LayerNorm offset is added AFTER the rescale, so its contribution through
 * the projection is a constant vector - no mean, no inverse-std. That is why it
 * folds here where the scale needs the closed form's two terms.
 */
export function relativeOffsetSums(offset, projection, pairChannels, outChannels) {
  const sums = new Float32Array(outChannels);
  for (let c = 0; c < RELATIVE_WIDTH; c += 1) {
    const row = pairChannels + c;
    for (let out = 0; out < outChannels; out += 1) {
      sums[out] += offset[row] * projection[row * outChannels + out];
    }
  }
  return sums;
}

export function relativeColumnSums(scale, projection, pairChannels, outChannels) {
  const sums = new Float32Array(outChannels);
  for (let c = 0; c < RELATIVE_WIDTH; c += 1) {
    const row = pairChannels + c;
    const weight = scale[row];
    for (let out = 0; out < outChannels; out += 1) {
      sums[out] += weight * projection[row * outChannels + out];
    }
  }
  return sums;
}

export function createConditioningShaders(shape, offsets) {
  const { tokens, pairChannels, seqChannels, trunkSingleChannels, targetFeatWidth, noiseChannels,
          padding, split = false, projectedRelpos = false, singleBias = false,
          trunkPairChannels = pairChannels } = shape;
  const pairs = tokens * tokens;
  const pairWidth = pairChannels + RELATIVE_WIDTH;
  const singleWidth = trunkSingleChannels + targetFeatWidth + padding.length;

  const pairInitial = `
const TOKENS: u32 = ${tokens}u;
const PAIRS: u32 = ${pairs}u;
const C_PAIR: u32 = ${pairChannels}u;
const WIDTH: u32 = ${pairWidth}u;
const RELATIVE_WIDTH: u32 = ${RELATIVE_WIDTH}u;
const POSITION_BINS: u32 = ${POSITION_BINS}u;
const MAX_RELATIVE_IDX: i32 = ${MAX_RELATIVE_IDX};
const MAX_RELATIVE_CHAIN: i32 = ${MAX_RELATIVE_CHAIN};
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1.0e-5;

@group(0) @binding(0) var<storage, read> trunk_pair: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<i32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> projection: array<f32>;
// 🔴 TWO VECTORS IN ONE BINDING: the closed form's S[o] in the first C_PAIR
// slots, then the contribution of the LayerNorm OFFSET over the 139 relative
// columns in the next - a constant of the weights exactly as S is. The
// trunk-pair half's offset stays in the loop below, where its projection row is
// read anyway. (No backticks in this comment: it is inside a JS template
// literal, and one would end the string.)
@group(0) @binding(4) var<storage, read> column_sums: array<f32>;
@group(0) @binding(5) var<storage, read_write> pair: array<f32>;

fn residue_index(t: u32) -> i32 { return features[t]; }
fn token_index(t: u32) -> i32 { return features[TOKENS + t]; }
fn asym_id(t: u32) -> i32 { return features[2u * TOKENS + t]; }
fn entity_id(t: u32) -> i32 { return features[3u * TOKENS + t]; }
fn sym_id(t: u32) -> i32 { return features[4u * TOKENS + t]; }
fn clamp_bin(value: i32, high: i32) -> i32 { return min(max(value, 0), high); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let i = row / TOKENS;
  let j = row % TOKENS;
  let base = row * C_PAIR;

  // Which relative columns are set.
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
  var count = 3.0;
  if (same_entity) { count = 4.0; }

  // The 267-wide statistics, in closed form.
  var total = count;
  for (var c = 0u; c < C_PAIR; c += 1u) { total += trunk_pair[base + c]; }
  let mean = total / f32(WIDTH);
  var variance = count * (1.0 - mean) * (1.0 - mean)
    + (f32(RELATIVE_WIDTH) - count) * mean * mean;
  for (var c = 0u; c < C_PAIR; c += 1u) {
    let d = trunk_pair[base + c] - mean;
    variance += d * d;
  }
  let inverse_std = inverseSqrt(variance / f32(WIDTH) + EPSILON);

  let row_a = C_PAIR + bin_a;
  let row_b = C_PAIR + POSITION_BINS + bin_b;
  let row_entity = C_PAIR + POSITION_BINS * 2u;
  let row_c = C_PAIR + POSITION_BINS * 2u + 1u + bin_c;

  for (var out = 0u; out < C_PAIR; out += 1u) {
    var value = 0.0;
    for (var c = 0u; c < C_PAIR; c += 1u) {
      value += ((trunk_pair[base + c] - mean) * inverse_std * scale[c]
        + scale[WIDTH + c]) * projection[c * C_PAIR + out];
    }
    // The set bins, minus the mean times every column's contribution.
    var gathered = scale[row_a] * projection[row_a * C_PAIR + out]
      + scale[row_b] * projection[row_b * C_PAIR + out]
      + scale[row_c] * projection[row_c * C_PAIR + out];
    if (same_entity) {
      gathered += scale[row_entity] * projection[row_entity * C_PAIR + out];
    }
    value += inverse_std * (gathered - mean * column_sums[out])
      + column_sums[C_PAIR + out];
    pair[row * C_PAIR + out] = value;
  }
}`;

  const singleInitial = `
const TOKENS: u32 = ${tokens}u;
const C_SEQ: u32 = ${seqChannels}u;
// 🔴 THE SINGLE THIS READS AND THE ONE IT WRITES ARE TWO WIDTHS. AlphaFold 3
// projects [831 -> 384] and its trunk single is also 384, so C_SEQ served as
// both the input boundary in feature() and the output extent below. boltz2
// projects [768 -> 768] from a trunk single of 384: read as one number,
// feature() takes 768 columns from a 384-wide row and the conditioning came
// out 8.2e-1 against af3-any-model.
const C_TRUNK_SEQ: u32 = ${trunkSingleChannels}u;
const TARGET_WIDTH: u32 = ${targetFeatWidth}u;
const WIDTH: u32 = ${singleWidth}u;
const NOISE_CHANNELS: u32 = ${noiseChannels}u;
const EPSILON: f32 = 1.0e-5;
const W_NOISE_SCALE: u32 = ${offsets.noiseEmbeddingInitialNormScale}u;
const W_NOISE_OFFSET: u32 = ${offsets.noiseEmbeddingInitialNormOffset}u;
const W_NOISE_PROJECT: u32 = ${offsets.noiseEmbeddingInitialProjection}u;

@group(0) @binding(0) var<storage, read> trunk_single: array<f32>;
@group(0) @binding(1) var<storage, read> target_feat: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> projection: array<f32>;
@group(0) @binding(4) var<storage, read> noise: array<f32>;
@group(0) @binding(5) var<storage, read> noise_weights: array<f32>;
@group(0) @binding(6) var<storage, read_write> single: array<f32>;

// 🔴 ONE WORKGROUP PER TOKEN, NOT ONE THREAD. This dispatched ceil(TOKENS/64)
// workgroups - ONE for a 59-residue protein - and each thread then computed 384
// output channels over an 831-wide concatenation plus the 256-wide noise
// embedding: 417k multiply-adds on a single lane, with the LayerNorm statistics
// recomputed for every one of those 384 outputs. The normalised vectors are
// computed once into workgroup memory here and the output range is strided
// across the lanes.
var<workgroup> normalised: array<f32, ${singleWidth}>;
var<workgroup> noise_norm: array<f32, ${noiseChannels}>;
var<workgroup> reduce_s: array<f32, 64>;

fn reduce_sum(local: u32, value: f32) -> f32 {
  reduce_s[local] = value;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_s[local] += reduce_s[local + stride]; }
    workgroupBarrier();
  }
  return reduce_s[0];
}

/** The concatenation [trunk single | target_feat], read as one row.
 *
 * The dialect may re-insert OpenFold3's two always-zero unknown-DNA columns,
 * which shift every later column's source - see singleCondPadding. The body
 * below is generated from the same list the CPU reference walks.
 */
fn feature(token: u32, index: u32) -> f32 {
  if (index < C_TRUNK_SEQ) { return trunk_single[token * C_TRUNK_SEQ + index]; }
  var source = index;
${singleCondPaddingWgsl(padding)}  return target_feat[token * TARGET_WIDTH + source - C_TRUNK_SEQ];
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x;
  if (token >= TOKENS) { return; }
  let local = local_id.x;

  var total = 0.0;
  for (var c = local; c < WIDTH; c += 64u) { total += feature(token, c); }
  let mean = reduce_sum(local, total) / f32(WIDTH);
  workgroupBarrier();
  var centred = 0.0;
  for (var c = local; c < WIDTH; c += 64u) {
    let d = feature(token, c) - mean;
    centred += d * d;
  }
  let inverse_std = inverseSqrt(reduce_sum(local, centred) / f32(WIDTH) + EPSILON);
  workgroupBarrier();
  for (var c = local; c < WIDTH; c += 64u) {
    normalised[c] = (feature(token, c) - mean) * inverse_std * scale[c] + scale[WIDTH + c];
  }

  // The noise embedding is one row, shared by every token: normalise and
  // project it here rather than in its own pass.
  var noise_total = 0.0;
  for (var c = local; c < NOISE_CHANNELS; c += 64u) { noise_total += noise[c]; }
  let noise_mean = reduce_sum(local, noise_total) / f32(NOISE_CHANNELS);
  workgroupBarrier();
  var noise_centred = 0.0;
  for (var c = local; c < NOISE_CHANNELS; c += 64u) {
    let d = noise[c] - noise_mean;
    noise_centred += d * d;
  }
  let noise_inverse = inverseSqrt(reduce_sum(local, noise_centred)
    / f32(NOISE_CHANNELS) + EPSILON);
  workgroupBarrier();
  for (var c = local; c < NOISE_CHANNELS; c += 64u) {
    noise_norm[c] = (noise[c] - noise_mean) * noise_inverse
      * noise_weights[W_NOISE_SCALE + c] + noise_weights[W_NOISE_OFFSET + c];
  }
  workgroupBarrier();

  for (var out = local; out < C_SEQ; out += 64u) {
    var value = 0.0;
    for (var c = 0u; c < WIDTH; c += 1u) {
      value += normalised[c] * projection[c * C_SEQ + out];
    }
    for (var c = 0u; c < NOISE_CHANNELS; c += 1u) {
      value += noise_norm[c] * noise_weights[W_NOISE_PROJECT + c * C_SEQ + out];
    }
${singleBias ? "    value += projection[C_SEQ * WIDTH + out];" : ""}
    single[token * C_SEQ + out] = value;
  }
}`;

  // 🔴 THE SPLIT DIALECT IS A DIFFERENT FUNCTION, NOT A RESHAPE. OpenDDE
  // normalises the trunk pair on its OWN width and projects it to the pair
  // width, projects the relative encoding to the pair width separately, and
  // only then concatenates and runs the joint LayerNorm - so the closed form
  // above, which exists because the raw one-hot columns share a LayerNorm with
  // the trunk pair, has nothing to be a closed form OF. See
  // diffusion-reference.js's `split` branch, which this mirrors line for line.
  //
  // 🔴 AND IT IS ONE DISPATCH, because the 2*C_PAIR concatenation is 1 KiB and
  // fits in workgroup memory. Materialising it would be `tokens^2 x 256`
  // floats - 92 MB at 300 tokens - written and read back for no reason.
  //
  // One workgroup per pair row rather than one lane, because the trunk
  // compression alone is C_TRUNK*C_PAIR multiply-accumulates: 49,152 at
  // OpenDDE's widths against the joint path's 16,384, and a lane-per-row
  // kernel reads the trunk row once per output column.
  const pairInitialSplit = `
const PAIRS: u32 = ${pairs}u;
const TOKENS: u32 = ${tokens}u;
const C_PAIR: u32 = ${pairChannels}u;
const C_TRUNK: u32 = ${trunkPairChannels}u;
const WIDTH: u32 = ${2 * pairChannels}u;
const RELATIVE_WIDTH: u32 = ${RELATIVE_WIDTH}u;
const POSITION_BINS: u32 = ${POSITION_BINS}u;
const MAX_RELATIVE_IDX: i32 = ${MAX_RELATIVE_IDX};
const MAX_RELATIVE_CHAIN: i32 = ${MAX_RELATIVE_CHAIN};
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const LANES: u32 = 64u;
const EPSILON: f32 = 1.0e-5;

@group(0) @binding(0) var<storage, read> trunk_pair: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<i32>;
@group(0) @binding(2) var<storage, read> z_scale: array<f32>;
@group(0) @binding(3) var<storage, read> z_projection: array<f32>;
@group(0) @binding(4) var<storage, read> relpe_projection: array<f32>;
@group(0) @binding(5) var<storage, read> scale: array<f32>;
@group(0) @binding(6) var<storage, read> projection: array<f32>;
@group(0) @binding(7) var<storage, read_write> pair: array<f32>;

var<workgroup> reduce: array<f32, LANES>;
var<workgroup> concatenated: array<f32, WIDTH>;

fn residue_index(t: u32) -> i32 { return features[t]; }
fn token_index(t: u32) -> i32 { return features[TOKENS + t]; }
fn asym_id(t: u32) -> i32 { return features[2u * TOKENS + t]; }
fn entity_id(t: u32) -> i32 { return features[3u * TOKENS + t]; }
fn sym_id(t: u32) -> i32 { return features[4u * TOKENS + t]; }
fn clamp_bin(value: i32, high: i32) -> i32 { return min(max(value, 0), high); }

// A whole-workgroup sum. The caller puts a barrier between two of these: the
// read of reduce[0] below is ordered before that barrier, so the next call's
// write cannot race it.
fn total_of(lane: u32, value: f32) -> f32 {
  reduce[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { reduce[lane] += reduce[lane + stride]; }
    workgroupBarrier();
  }
  return reduce[0];
}

@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= PAIRS) { return; }
  let lane = local.x;
  let base = row * C_TRUNK;

  // LayerNorm over the trunk pair's own width.
  var partial = 0.0;
  for (var c = lane; c < C_TRUNK; c += LANES) { partial += trunk_pair[base + c]; }
  let mean = total_of(lane, partial) / f32(C_TRUNK);
  workgroupBarrier();
  var squares = 0.0;
  for (var c = lane; c < C_TRUNK; c += LANES) {
    let d = trunk_pair[base + c] - mean;
    squares += d * d;
  }
  let inverse_std = inverseSqrt(total_of(lane, squares) / f32(C_TRUNK) + EPSILON);
  workgroupBarrier();

  // ...projected to the pair width, into the first half of the concatenation.
  for (var out = lane; out < C_PAIR; out += LANES) {
    var value = 0.0;
    for (var c = 0u; c < C_TRUNK; c += 1u) {
      value += ((trunk_pair[base + c] - mean) * inverse_std * z_scale[c]
        + z_scale[C_TRUNK + c]) * z_projection[c * C_PAIR + out];
    }
    concatenated[out] = value;
  }

  // The relative encoding, projected on its own and with no LayerNorm. It is
  // one-hot in three or four of 139 columns, so the projection is a gather of
  // those rows rather than a matrix multiply - exact, not an approximation.
  let i = row / TOKENS;
  let j = row % TOKENS;
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
  for (var out = lane; out < C_PAIR; out += LANES) {
    var value = relpe_projection[bin_a * C_PAIR + out]
      + relpe_projection[row_b * C_PAIR + out]
      + relpe_projection[row_c * C_PAIR + out];
    if (same_entity) { value += relpe_projection[row_entity * C_PAIR + out]; }
    concatenated[C_PAIR + out] = value;
  }
  workgroupBarrier();

  // The joint LayerNorm over the two halves, and the output projection.
  var joint = 0.0;
  for (var c = lane; c < WIDTH; c += LANES) { joint += concatenated[c]; }
  let joint_mean = total_of(lane, joint) / f32(WIDTH);
  workgroupBarrier();
  var joint_squares = 0.0;
  for (var c = lane; c < WIDTH; c += LANES) {
    let d = concatenated[c] - joint_mean;
    joint_squares += d * d;
  }
  let joint_inverse = inverseSqrt(total_of(lane, joint_squares) / f32(WIDTH) + EPSILON);

  for (var out = lane; out < C_PAIR; out += LANES) {
    var value = 0.0;
    for (var c = 0u; c < WIDTH; c += 1u) {
      value += ((concatenated[c] - joint_mean) * joint_inverse * scale[c]
        + scale[WIDTH + c]) * projection[c * C_PAIR + out];
    }
    pair[row * C_PAIR + out] = value;
  }
}`;

  // 🔴 THE THIRD SHAPE: THE RELPOS PROJECTED, THE TRUNK PAIR PASSED THROUGH.
  // protenix2 and boltz2 both take it, and with only `split` and the joint arm
  // to choose from they fell into AlphaFold 3's raw-139 one - which read a
  // PREFIX of protenix2's 512-long scale and quietly agreed with an equally
  // wrong reference, and read PAST the end of boltz2's 256-long one and made
  // 73728 NaNs. See the note in src/af3/diffusion/diffusion-reference.js.
  //
  // It is the split kernel with the trunk compression replaced by a copy: no
  // z_norm, no z_projection, and WIDTH is C_TRUNK + C_PAIR rather than
  // 2 * C_PAIR. Derived from that text rather than written beside it, so the
  // two cannot drift - the joint LayerNorm and the output projection below the
  // concatenation are the same lines in both.
  const pairInitialProjectedRelpos = pairInitialSplit
    .replace(`const WIDTH: u32 = ${2 * pairChannels}u;`,
             `const WIDTH: u32 = ${trunkPairChannels + pairChannels}u;`)
    .replace(`@group(0) @binding(2) var<storage, read> z_scale: array<f32>;
@group(0) @binding(3) var<storage, read> z_projection: array<f32>;
@group(0) @binding(4) var<storage, read> relpe_projection: array<f32>;
@group(0) @binding(5) var<storage, read> scale: array<f32>;
@group(0) @binding(6) var<storage, read> projection: array<f32>;
@group(0) @binding(7) var<storage, read_write> pair: array<f32>;`,
             `@group(0) @binding(2) var<storage, read> relpe_projection: array<f32>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;
@group(0) @binding(4) var<storage, read> projection: array<f32>;
@group(0) @binding(5) var<storage, read_write> pair: array<f32>;`)
    .replace(`  // LayerNorm over the trunk pair's own width.
  var partial = 0.0;
  for (var c = lane; c < C_TRUNK; c += LANES) { partial += trunk_pair[base + c]; }
  let mean = total_of(lane, partial) / f32(C_TRUNK);
  workgroupBarrier();
  var squares = 0.0;
  for (var c = lane; c < C_TRUNK; c += LANES) {
    let d = trunk_pair[base + c] - mean;
    squares += d * d;
  }
  let inverse_std = inverseSqrt(total_of(lane, squares) / f32(C_TRUNK) + EPSILON);
  workgroupBarrier();

  // ...projected to the pair width, into the first half of the concatenation.
  for (var out = lane; out < C_PAIR; out += LANES) {
    var value = 0.0;
    for (var c = 0u; c < C_TRUNK; c += 1u) {
      value += ((trunk_pair[base + c] - mean) * inverse_std * z_scale[c]
        + z_scale[C_TRUNK + c]) * z_projection[c * C_PAIR + out];
    }
    concatenated[out] = value;
  }`,
             `  // The trunk pair, copied in RAW: no LayerNorm of its own and no
  // projection. The joint LayerNorm below is the only one it sees.
  for (var c = lane; c < C_TRUNK; c += LANES) {
    concatenated[c] = trunk_pair[base + c];
  }`)
    .replace(`    concatenated[C_PAIR + out] = value;`,
             `    concatenated[C_TRUNK + out] = value;`);
  if (projectedRelpos && pairInitialProjectedRelpos === pairInitialSplit) {
    throw new Error("the projected-relpos shader derivation matched nothing: "
      + "the split kernel's text moved and the two have drifted apart");
  }

  return { pairInitial: projectedRelpos ? pairInitialProjectedRelpos
             : split ? pairInitialSplit : pairInitial, singleInitial };
}

/**
 * The reference reads an unconditioned transition's weights as `ffwLayerNorm*`
 * and `ffwTransition*`, which is adaptiveLayerNorm's naming convention carried
 * over; the transition kernel names them for its own shader. Mapped here rather
 * than renamed in either, because both names are load-bearing where they are.
 */
function asTransitionWeights(block) {
  return {
    inputLayerNormScale: block.ffwLayerNormScale,
    inputLayerNormOffset: block.ffwLayerNormOffset,
    transition1: block.ffwTransition1,
    transition2: block.ffwTransition2,
  };
}

/** `residual += transition(residual)`, for the four unconditioned transitions.
 *  One definition, in src/runtime/execution.js. */
/** AF3's Fourier noise embedding: cos(2 pi (w log(sigma)/4 + b)). */
export function noiseEmbedding(scaledNoiseLevel, weight, bias) {
  if (weight === undefined || bias === undefined) {
    throw new Error("the Fourier constants are required; see tools/export_af3_model.py");
  }
  const output = new Float32Array(weight.length);
  const logLevel = Math.log(scaledNoiseLevel) / 4;
  for (let index = 0; index < weight.length; index += 1) {
    output[index] = Math.cos(2 * Math.PI * (weight[index] * logLevel + bias[index]));
  }
  return output;
}

export class Af3DiffusionConditioningGpu {
  /**
   * @param {GPUDevice} device
   * @param {{pool?: boolean}} [options] pool the allocator. 🔴 A CALLER THAT
   *   TAKES THE RESULT WITHOUT WAITING FOR THE GPU MUST POOL. `release()`
   *   DESTROYS a buffer when the allocator does not pool, and returning before
   *   the submitted work has finished would destroy tensors that work is still
   *   reading - which WebGPU reports as "used in submit while destroyed" at
   *   some later, unrelated submit. Pooling hands them back instead, and the
   *   next call to reuse one is a step later, behind queue ordering.
   */
  constructor(device, options = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device, options.pool ?? false);
    this.pipelines = pipelineCacheForDevice(device);
  }


  /**
   * This shape's pipelines, without any of the fold's tensors.
   *
   * 🔴 SO THEY CAN BE COMPILED WHILE THE TRUNK RUNS. A cold denoiser call at
   * 130 tokens is 1360 ms against 16 warm, and `conditioning` is 189 of that -
   * all of it `createComputePipelineAsync`, which compiles off the main thread
   * and needs only a token count, the weights and the dialect. See
   * Af3DiffusionHeadGpu.warm, which already does this for the transformer.
   */
  async warm(tokens, weights, dialect) {
    await this.#compileFor(tokens, weights, dialect, undefined);
  }

  async #compileFor(tokens, weights, dialect, reusePair) {
    const pairs = tokens * tokens;
    const pairChannels = weights.pairChannels;
    const seqChannels = weights.seqChannels;
    // ...and the single it READS, which is a different number under boltz2.
    const trunkSingleChannels = weights.trunkSingleChannels ?? seqChannels;
    const targetFeatWidth = weights.targetFeatWidth;
    const noiseChannels = weights.fourierWeight.length;
    const prepared = prepareWeights(weights);
    const noisePacked = prepared.noisePacked;
    const padding = singleCondPadding(dialect, trunkSingleChannels);
    // 🔴 THE TRUNK PAIR'S WIDTH IS NOT THE CONDITIONING'S under the split
    // dialect - OpenDDE hands 384 channels to a 128-channel conditioning - so
    // the two are separate here and the cache key carries both.
    const split = weights.zTrunkProjection !== undefined;
    const projectedRelpos = !split && weights.relpeProjection !== undefined;
    // 🔴 THE TRUNK PAIR'S WIDTH MATTERS IN BOTH PROJECTED SHAPES, not only the
    // split one - it is the first half of the concatenation either way.
    const trunkPairChannels = (split || projectedRelpos)
      ? (weights.trunkPairChannels ?? pairChannels) : pairChannels;
    // 🔴 boltz2's SINGLE PROJECTION CARRIES A BIAS AND NOBODY ELSE'S DOES, so
    // it is appended to that buffer and read past its matrix - which keeps the
    // binding count the same for every model.
    const singleBias = weights.singleCondInitialProjectionBias != null;
    const shape = { tokens, pairChannels, seqChannels, trunkSingleChannels,
                    targetFeatWidth, noiseChannels, singleBias,
                    padding, split, projectedRelpos, trunkPairChannels };
    const sources = createConditioningShaders(shape, noisePacked.offsets);
    const base = `af3-diffcond:${tokens}:${pairChannels}:${seqChannels}:${targetFeatWidth}`
      // 🔴 THE MODE IS IN THE KEY. Two of the three shapes can produce the same
      // dimensions with different kernels, and a cache indexed on dimensions
      // alone would hand one caller the other's - the collision this repository
      // has now paid for four times.
      + `:${noiseChannels}:${padding.join(",")}:${split ? trunkPairChannels : 0}`
      + `:${projectedRelpos ? `pr${trunkPairChannels}` : ""}`
      + `:ts${trunkSingleChannels}${singleBias ? ":sb" : ""}`;
    // Every pipeline here is ASKED FOR before any is awaited, so the browser
    // builds them together; awaited one by one they were ten serial compiles
    // on a cold denoiser call.
    const compiling = {
      pairInitial: reusePair !== undefined ? undefined
        : this.pipelines.get(`${base}:pair-initial`, sources.pairInitial),
      singleInitial: this.pipelines.get(`${base}:single-initial`, sources.singleInitial),
      addPair: reusePair !== undefined ? undefined
        : this.pipelines.get(`${base}:add-pair`, createAddShader(pairs * pairChannels)),
      addSingle: this.pipelines.get(`${base}:add-single`,
        createAddShader(tokens * seqChannels)),
    };
    // The four unconditioned transitions: the trunk's shader, two-pass variance.
    const transitionPipelines = { pair: [], single: [] };
    for (let index = 0; index < 2; index += 1) {
      transitionPipelines.pair.push(reusePair !== undefined ? undefined
        : this.pipelines.get(`${base}:pair-transition:${index}`,
            createTransitionShader({ rows: pairs, channels: pairChannels, factor: 2 },
                                   prepared.pairTransitions[index].offsets, 1e-5, "two-pass")));
      // 🔴 THE SINGLE TRANSITION IS `tokens` ROWS AND THE KERNEL DISPATCHES
      // ROWS ONLY, so at 68 tokens it runs 68 workgroups of the default 128
      // threads - 8,704 on a device that holds 221,184. `transitionThreadTarget`
      // is the rule that widens it, and device-profile.js prices it at 4.19x on
      // the TRUNK's single transition, which is the same shape. This call site
      // never asked for it: boltz2's two conditioning transitions were 44% of
      // its whole denoiser call's GPU time, 8.4 ms of 21.4, because its seq
      // channel is 768 where AlphaFold 3's is 384 and the row's arithmetic goes
      // as the square.
      //
      // The width is BAKED into the shader, so it is in the key.
      const singleWidth_ = transitionWidth(
        tokens, transitionRowTile(tokens, seqChannels),
        deviceTuning(this.device).transitionThreadTarget, undefined, seqChannels * 2,
        // ...and never wider than this device will run. See transitionWidth.
        this.device.limits.maxComputeWorkgroupSizeX);
      transitionPipelines.single.push(this.pipelines.get(
        `${base}:single-transition:${index}:w${singleWidth_}`,
        createTransitionShader({ rows: tokens, channels: seqChannels, factor: 2,
                                 width: singleWidth_ },
                               prepared.singleTransitions[index].offsets, 1e-5, "two-pass")));
    }
    const compiled = await settleAll(compiling);
    transitionPipelines.pair = await Promise.all(transitionPipelines.pair);
    transitionPipelines.single = await Promise.all(transitionPipelines.single);
    return { shape, base, compiled, transitionPipelines };
  }

  /**
   * @param {{tokens: number, trunkSingle, trunkPair, targetFeat, noiseLevel,
   *          features: {residueIndex, tokenIndex, asymId, entityId, symId}}} input
   * @param {object} weights
   */
  async run(input, weights, options = {}) {
    const tokens = input.tokens;
    const pairs = tokens * tokens;
    const pairChannels = weights.pairChannels;
    const seqChannels = weights.seqChannels;
    // ...and the single it READS, which is a different number under boltz2.
    const trunkSingleChannels = weights.trunkSingleChannels ?? seqChannels;
    const targetFeatWidth = weights.targetFeatWidth;
    const noiseChannels = weights.fourierWeight.length;

    // 🔴 THE ONLY THING HERE THAT READS THE NOISE LEVEL. Everything else this
    // call needs from the weights is packed once per bundle; see
    // prepareWeights.
    const embedded = noiseEmbedding(input.noiseLevel / SIGMA_DATA,
                                    weights.fourierWeight, weights.fourierBias);
    const prepared = prepareWeights(weights);
    const columnSums = prepared.columnSums;
    const noisePacked = prepared.noisePacked;
    // Hoisted above the uploads: with the pair handed back from a previous
    // call, the whole pair track - its 1.8 MB trunk upload at 59 tokens, its
    // three scratch tensors and its two transition weight bundles - is work
    // this call must not do.
    const reusePair = options.reusePair;

    // 🔴 THE PADDING IS PART OF THE CACHE KEY, because it changes the generated
    // `feature()` body while every dimension in the key stays put - the one
    // shape a shader cache cannot see.
    const padding = singleCondPadding(input.dialect, trunkSingleChannels);
    const singleWidth = trunkSingleChannels + targetFeatWidth + padding.length;
    if (weights.singleCondInitialNormScale.length !== singleWidth) {
      throw new Error(`single conditioning is ${singleWidth} channels but its `
        + `LayerNorm scale is ${weights.singleCondInitialNormScale.length}; `
        + "the dialect and the weights disagree about the unknown-DNA columns");
    }

    const { shape, base, compiled, transitionPipelines } =
      await this.#compileFor(tokens, weights, input.dialect, reusePair);
    // ...and the third mode, carried on the shape with it so `run` and the
    // shader factory cannot disagree about which kernel is compiled.
    const { split, projectedRelpos } = shape;
    const featureData = new Int32Array(5 * tokens);
    ["residueIndex", "tokenIndex", "asymId", "entityId", "symId"].forEach((name, index) => {
      const source = input.features[name];
      if (source === undefined) throw new Error(`features.${name} is required`);
      for (let t = 0; t < tokens; t += 1) featureData[index * tokens + t] = source[t];
    });

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const up = (label, data) => keep(this.allocator.upload(label, data, storage));
      // 🔴 EVERY LINE GUARDED BY `onlyIfNew` IS PAIR-TRACK WORK A REUSED PAIR
      // MAKES POINTLESS - and `cond.trunk-pair` alone is tokens^2 x 128 floats
      // written across the bus, 1.8 MB at 59 tokens and 34 MB at 256, once per
      // sampler step, into a buffer no dispatch was going to read.
      const onlyIfNew = (build) => (reusePair === undefined ? build() : undefined);
      // 🔴 THE CALLER'S BUFFER WHEN IT HAS ONE. On the structural-token path
      // the trunk pair comes straight out of a pairformer stack that can leave
      // it on the device, and this was a `tokens^2 x 384` upload of a tensor
      // that had just been read back for the purpose.
      const trunkPair = onlyIfNew(() => (input.trunkPairBuffer !== undefined
        ? { buffer: input.trunkPairBuffer } : up("cond.trunk-pair", input.trunkPair)));
      const trunkSingle = up("cond.trunk-single", input.trunkSingle);
      const targetFeat = up("cond.target", input.targetFeat);
      const features = onlyIfNew(() => up("cond.features", featureData));
      // 🔴 THE WEIGHT-DERIVED ONES ARE RESIDENT, NOT UPLOADED PER STEP. Six of
      // these eight are functions of the weight bundle - a sampler was writing
      // them across the bus two hundred times, and the two transition bundles
      // alone are 6.8 MiB. `cond.noise` is the exception that makes the module:
      // it IS the noise level.
      const resident = (label, build) =>
        ({ buffer: residentWeightBuffer(this.device, weights, label, build) });
      const pairScale = onlyIfNew(() => resident("cond.pair-scale",
        () => packNormWeights(weights.pairCondInitialNormScale,
                              weights.pairCondInitialNormOffset)));
      const pairProjection = onlyIfNew(() => resident("cond.pair-projection",
        () => weights.pairCondInitialProjection));
      const sums = onlyIfNew(() => resident("cond.column-sums", () => columnSums));
      const zScale = split
        ? onlyIfNew(() => resident("cond.z-scale",
            () => packNormWeights(weights.zTrunkNormScale, weights.zTrunkNormOffset)))
        : undefined;
      const zProjection = split
        ? onlyIfNew(() => resident("cond.z-projection", () => weights.zTrunkProjection))
        : undefined;
      const relpeProjection = (split || projectedRelpos)
        ? onlyIfNew(() => resident("cond.relpe-projection", () => weights.relpeProjection))
        : undefined;
      const singleScale = resident("cond.single-scale",
        () => packNormWeights(weights.singleCondInitialNormScale,
                              weights.singleCondInitialNormOffset));
      const singleProjection = resident("cond.single-projection", () => {
        const matrix = weights.singleCondInitialProjection;
        const bias = weights.singleCondInitialProjectionBias;
        if (bias == null) return matrix;
        const packed = new Float32Array(matrix.length + bias.length);
        packed.set(matrix, 0);
        packed.set(bias, matrix.length);
        return packed;
      });
      const noise = up("cond.noise", embedded);
      const noiseWeights = resident("cond.noise-weights", () => noisePacked.data);

      // 🔴 AND THE PAIR, WHEN THE CALLER OFFERS A BUFFER FOR IT. The first call
      // of a fold computes the pair conditioning and read it back so the head
      // could key its caches on the host array - 32 MiB at 255 tokens, a drain,
      // and then the transformer and the encoder uploaded it straight back. The
      // head now keeps it on the device and keys on the buffer instead.
      const outPair = reusePair === undefined ? options.outputs?.pair : undefined;
      const pair = onlyIfNew(() => (outPair !== undefined ? { buffer: outPair }
        : keep(this.allocator.allocate("cond.pair",
          pairs * pairChannels * 4, storage | GPUBufferUsage.COPY_SRC))));
      // 🔴 THE CALLER'S BUFFER WHEN IT OFFERS ONE, AND NO READBACK THEN. The
      // single track is the only thing here that moves with the noise level,
      // and the diffusion head's next two stages both read it - so a sampler
      // was draining the pipeline to copy tokens x 384 floats to the host and
      // writing them straight back. `options.outputs.single` lets the head keep
      // it on the device; see the note on the head's #chain.
      const outSingle = options.outputs?.single;
      const single = outSingle === undefined
        ? keep(this.allocator.allocate("cond.single", tokens * seqChannels * 4,
            storage | GPUBufferUsage.COPY_SRC))
        : { buffer: outSingle };
      const pairScratch = onlyIfNew(() => keep(this.allocator.allocate("cond.pair-scratch",
        pairs * pairChannels * 4, storage)));
      const singleScratch = keep(this.allocator.allocate("cond.single-scratch",
        tokens * seqChannels * 4, storage));
      const readPair = outPair !== undefined ? undefined
        : onlyIfNew(() => keep(this.allocator.allocate("cond.rb-pair",
          pairs * pairChannels * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)));
      const readSingle = outSingle !== undefined ? undefined
        : keep(this.allocator.allocate("cond.rb-single", tokens * seqChannels * 4,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      const transitionWeights = {
        pair: prepared.pairTransitions.map((packed, index) =>
          onlyIfNew(() => resident(`cond.pair-transition-w${index}`, () => packed.data))),
        single: prepared.singleTransitions.map((packed, index) =>
          resident(`cond.single-transition-w${index}`, () => packed.data)),
      };

      // The head's, when it has one; otherwise this call's own scope, awaited
      // below the way it always was.
      const deferred = outSingle === undefined ? undefined : options.validation;
      if (deferred === undefined) this.device.pushErrorScope("validation");
      else deferred.begin();
      const encoder = this.device.createCommandEncoder({ label: "af3-diffusion-conditioning" });
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
      const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
      // 🔴 THE PAIR CONDITIONING DOES NOT DEPEND ON THE NOISE LEVEL. It is the
      // trunk's pair and the relative encoding, projected and twice
      // transitioned - and none of that reads sigma, which enters only through
      // the Fourier embedding added to the SINGLE. A sampler calls this two
      // hundred times down one schedule and got the identical pair every time.
      // `reusePair` hands back the one a previous call already computed and
      // skips three of the five pipelines here; the head owns the caching,
      // because only the head knows the trunk has not changed underneath it -
      // and it is read at the top of this method, because the uploads and the
      // allocations above it are pair-track work too.
      // The split kernel is a workgroup per pair row; the joint one is a lane.
      // The two projected kernels are a workgroup per pair row; the joint one is
      // a lane.
      const pairLinear = spread((split || projectedRelpos) ? pairs
        : Math.ceil(pairs / 64));
      if (reusePair === undefined) {
        run("pair-initial", compiled.pairInitial,
            split
              ? [trunkPair, features, zScale, zProjection, relpeProjection,
                 pairScale, pairProjection, pair]
              : projectedRelpos
                ? [trunkPair, features, relpeProjection, pairScale, pairProjection, pair]
                : [trunkPair, features, pairScale, pairProjection, sums, pair],
            pairLinear[0], pairLinear[1]);
      }
      run("single-initial", compiled.singleInitial,
          [trunkSingle, targetFeat, singleScale, singleProjection, noise, noiseWeights, single],
          tokens);

      const pairAdd = spread(Math.ceil(pairs * pairChannels / 64));
      const singleAdd = spread(Math.ceil(tokens * seqChannels / 64));
      // `transitions: 0` stops after the initial projections, which is how the
      // closed-form relative-encoding path is checked on its own.
      const transitionCount = options.transitions ?? 2;
      for (let index = 0; index < transitionCount; index += 1) {
        if (reusePair === undefined) {
          const perPair = spread(Math.ceil(pairs / transitionRowTile(pairs, pairChannels)));
          run(`pair-transition-${index}`, transitionPipelines.pair[index],
              [pair, transitionWeights.pair[index], pairScratch], perPair[0], perPair[1]);
          run(`pair-add-${index}`, compiled.addPair, [pair, pairScratch], pairAdd[0], pairAdd[1]);
        }
        run(`single-transition-${index}`, transitionPipelines.single[index],
            [single, transitionWeights.single[index], singleScratch],
            Math.ceil(tokens / transitionRowTile(tokens, seqChannels)));
        run(`single-add-${index}`, compiled.addSingle, [single, singleScratch],
            singleAdd[0], singleAdd[1]);
      }
      if (reusePair === undefined && readPair !== undefined) {
        encoder.copyBufferToBuffer(pair.buffer, 0, readPair.buffer, 0, pairs * pairChannels * 4);
      }
      if (outSingle === undefined) {
        encoder.copyBufferToBuffer(single.buffer, 0, readSingle.buffer, 0,
                                   tokens * seqChannels * 4);
      }

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      // 🔴 NOT AWAITED WHEN NOTHING IS READ BACK. popErrorScope resolves when
      // the submitted work has FINISHED, so awaiting it is a full pipeline
      // drain - the same one the readback would cost, and pointless when the
      // caller is about to encode three more stages on top of this one. The
      // head hands in its own DeferredValidation and settles it once, at the
      // boundary that already synchronises.
      if (deferred !== undefined) {
        deferred.end("diffusion conditioning");
        return {
          pair: reusePair, single: undefined, singleBuffer: outSingle,
          elapsedMilliseconds: performance.now() - start,
          memory: this.allocator.snapshot(),
        };
      }
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      const read = async (allocation) => {
        await allocation.buffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(allocation.buffer.getMappedRange().slice(0));
        allocation.buffer.unmap();
        return copy;
      };
      return {
        pair: reusePair ?? (outPair !== undefined ? pair : await read(readPair)),
        single: await read(readSingle),
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
