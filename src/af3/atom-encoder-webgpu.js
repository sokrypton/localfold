/**
 * AF3's atom cross-attention encoder on the GPU.
 *
 * Atoms are attended in WINDOWS, not globally: 32 queries attend over 128 keys
 * within each of `subsets` overlapping subsets, which is what keeps an
 * atom-level model from being quadratic in atoms. Five gathers, all precomputed
 * by the featuriser, move between the three layouts (token-atoms, queries,
 * keys).
 *
 * 🔴 THE ORDER OF MASK AND GATHER IS LOAD-BEARING. The queries' conditioning is
 * masked BEFORE the keys are gathered from it. That is a no-op for the trunk's
 * own encoder, whose per-atom conditioning is already masked, and it is NOT a
 * no-op once the trunk single conditioning is added - that puts non-zero values
 * into padded atom slots, and gathering first carries them into the keys, where
 * two thirds of the slots are padding. Measured at 8.4e-2 on the encoder's
 * output when it was the wrong way round: a subtly wrong kernel, not a crash.
 *
 * 🔴 THE ATTENTION MASK BIAS IS A PRODUCT, NOT A SUM. `1e9 * (qmask-1) *
 * (kmask-1)` penalises a pair only when the query AND the key are padded, so a
 * real query can still attend to a padded key. Adding the two terms instead is
 * an OR, and the difference is large in a mostly-empty window. (RoseTTAFold3
 * does add them.)
 *
 * 🔴 PAIR VALIDITY IS "SAME REFERENCE SPACE", NOT "BOTH ATOMS REAL". Two atoms
 * only have a meaningful offset if they came from the same reference conformer.
 * And the validity FLAG itself is added ungated - "these two atoms are
 * unrelated" is information the model uses.
 *
 * 🔴 tokens_to_keys COMES FROM THE BATCH. Deriving it by carrying
 * tokens_to_queries through the queries-to-keys gather looks equivalent, but
 * its MASK is not: a derived one folds in the query's mask where AF3's is the
 * key's own.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const GRID_WIDTH = 32_768;

const BLOCK_ORDER = [
  "qSingleCondLayerNormScale", "qSingleCondScaleWeights", "qSingleCondScaleBias",
  "qSingleCondBias", "kSingleCondLayerNormScale", "kSingleCondScaleWeights",
  "kSingleCondScaleBias", "kSingleCondBias",
  "qProjection", "qBias", "kProjection", "vProjection", "gatingQuery",
  "Transition2", "AdaptiveZeroCondWeights", "AdaptiveZeroCondBias",
  "ffwSingleCondLayerNormScale", "ffwSingleCondScaleWeights", "ffwSingleCondScaleBias",
  "ffwSingleCondBias", "ffwTransition1", "ffwTransition2",
  "ffwAdaptiveZeroCondWeights", "ffwAdaptiveZeroCondBias",
];

export function packAtomBlockWeights(block) {
  const offsets = {};
  let total = 0;
  for (const name of BLOCK_ORDER) {
    if (block[name] === undefined) throw new Error(`atom block missing ${name}`);
    offsets[name] = total;
    total += block[name].length;
  }
  const data = new Float32Array(total);
  for (const name of BLOCK_ORDER) data.set(block[name], offsets[name]);
  return { data, offsets };
}

const PAIR_ORDER = [
  "singleToPairCondRow", "singleToPairCondCol", "embedPairOffsets",
  "embedPairDistances", "embedPairOffsetsValid", "pairMlp1", "pairMlp2", "pairMlp3",
  "pairInputLayerNormScale", "pairLogitsProjection",
  "lnormTrunkSingleCondScale", "embedTrunkSingleCond",
  "lnormTrunkPairCondScale", "embedTrunkPairCond",
  "atomPositionsToFeatures", "projectAtomFeaturesForAggr",
];

export function packAtomPairWeights(weights) {
  const offsets = {};
  let total = 0;
  for (const name of PAIR_ORDER) {
    if (weights[name] === undefined) throw new Error(`atom encoder missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of PAIR_ORDER) data.set(weights[name], offsets[name]);
  return { data, offsets };
}

/**
 * The constant preamble every atom shader shares. Exported because the DECODER
 * runs the same cross-attention blocks over the same layout - only its weights
 * and its two end passes differ.
 */
export function createAtomCommon(shape, pairOffsets, blockOffsets) {
  const { tokens, dense, subsets, queries, keys, channels, pairChannels,
          heads, dimension, perTokenChannels, trunkSingleChannels, trunkPairChannels,
          blocks } = shape;
  const width = heads * dimension;
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;
  const pairRows = subsets * queries * keys;
  const intermediate = channels * 2;

  return `
const TOKENS: u32 = ${tokens}u;
const DENSE: u32 = ${dense}u;
const SUBSETS: u32 = ${subsets}u;
const QUERIES: u32 = ${queries}u;
const KEYS: u32 = ${keys}u;
const QUERY_ROWS: u32 = ${queryRows}u;
const KEY_ROWS: u32 = ${keyRows}u;
const PAIR_ROWS: u32 = ${pairRows}u;
const C: u32 = ${channels}u;
const C_PAIR: u32 = ${pairChannels}u;
const C_TOKEN: u32 = ${perTokenChannels}u;
const C_TRUNK_SINGLE: u32 = ${trunkSingleChannels}u;
const C_TRUNK_PAIR: u32 = ${trunkPairChannels}u;
const HEADS: u32 = ${heads}u;
const DIMENSION: u32 = ${dimension}u;
const WIDTH: u32 = ${width}u;
const INTERMEDIATE: u32 = ${intermediate}u;
const BLOCKS: u32 = ${blocks}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1.0e-5;
const SCALE: f32 = ${1 / Math.sqrt(dimension)};
${Object.entries(pairOffsets).map(([n, v]) => `const P_${n}: u32 = ${v}u;`).join("\n")}
${Object.entries(blockOffsets).map(([n, v]) => `const W_${n}: u32 = ${v}u;`).join("\n")}

fn logistic(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }
fn swish(v: f32) -> f32 { return v / (1.0 + exp(-v)); }
fn relu(v: f32) -> f32 { return max(v, 0.0); }

// 🔴 EVERY GATHER IN ONE BUFFER, because WebGPU only GUARANTEES eight storage
// buffers per stage and this pass wanted nine. The adapter here allows ten,
// which is exactly the kind of headroom that makes a kernel work on the machine
// it was written on and fail elsewhere.
const G_TA_IDX: u32 = 0u;
const G_TA_MASK: u32 = ${queryRows}u;
const G_TQ_IDX: u32 = ${2 * queryRows}u;
const G_TQ_MASK: u32 = ${3 * queryRows}u;
const G_QK_IDX: u32 = ${4 * queryRows}u;
const G_QK_MASK: u32 = ${4 * queryRows + keyRows}u;
const G_TK_IDX: u32 = ${4 * queryRows + 2 * keyRows}u;
const G_TK_MASK: u32 = ${4 * queryRows + 3 * keyRows}u;
const G_QTA_IDX: u32 = ${4 * queryRows + 4 * keyRows}u;
const G_QTA_MASK: u32 = ${4 * queryRows + 4 * keyRows + tokens * dense}u;
const G_QSPACE: u32 = ${4 * queryRows + 4 * keyRows + 2 * tokens * dense}u;
const G_KSPACE: u32 = ${5 * queryRows + 4 * keyRows + 2 * tokens * dense}u;
`;
}

export function createAtomEncoderShaders(shape, pairOffsets, blockOffsets) {
  const { tokens, dense, subsets, queries, keys, channels, pairChannels,
          heads, dimension, perTokenChannels, trunkSingleChannels, trunkPairChannels,
          blocks } = shape;
  const width = heads * dimension;
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;
  const pairRows = subsets * queries * keys;
  const intermediate = channels * 2;
  const common = createAtomCommon(shape, pairOffsets, blockOffsets);

  // The trunk's single conditioning, per token: LayerNorm then project to 128.
  const trunkSingle = `${common}
@group(0) @binding(0) var<storage, read> trunk_single: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> projected: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let token = id.x;
  if (token >= TOKENS) { return; }
  let base = token * C_TRUNK_SINGLE;
  var total = 0.0;
  for (var c = 0u; c < C_TRUNK_SINGLE; c += 1u) { total += trunk_single[base + c]; }
  let mean = total / f32(C_TRUNK_SINGLE);
  var variance = 0.0;
  for (var c = 0u; c < C_TRUNK_SINGLE; c += 1u) {
    let d = trunk_single[base + c] - mean;
    variance += d * d;
  }
  let inverse = inverseSqrt(variance / f32(C_TRUNK_SINGLE) + EPSILON);
  for (var out = 0u; out < C; out += 1u) {
    var value = 0.0;
    for (var c = 0u; c < C_TRUNK_SINGLE; c += 1u) {
      value += (trunk_single[base + c] - mean) * inverse
        * weights[P_lnormTrunkSingleCondScale + c]
        * weights[P_embedTrunkSingleCond + c * C + out];
    }
    projected[token * C + out] = value;
  }
}`;

  // The trunk's pair conditioning, per token pair: LayerNorm then project to 16.
  const trunkPair = `${common}
@group(0) @binding(0) var<storage, read> trunk_pair: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> projected: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= TOKENS * TOKENS) { return; }
  let base = row * C_TRUNK_PAIR;
  var total = 0.0;
  for (var c = 0u; c < C_TRUNK_PAIR; c += 1u) { total += trunk_pair[base + c]; }
  let mean = total / f32(C_TRUNK_PAIR);
  var variance = 0.0;
  for (var c = 0u; c < C_TRUNK_PAIR; c += 1u) {
    let d = trunk_pair[base + c] - mean;
    variance += d * d;
  }
  let inverse = inverseSqrt(variance / f32(C_TRUNK_PAIR) + EPSILON);
  for (var out = 0u; out < C_PAIR; out += 1u) {
    var value = 0.0;
    for (var c = 0u; c < C_TRUNK_PAIR; c += 1u) {
      value += (trunk_pair[base + c] - mean) * inverse
        * weights[P_lnormTrunkPairCondScale + c]
        * weights[P_embedTrunkPairCond + c * C_PAIR + out];
    }
    projected[row * C_PAIR + out] = value;
  }
}`;

  // queriesCond, queriesMask, queriesAct - and the masking BEFORE the keys are
  // gathered from it.
  const buildQueries = `${common}
@group(0) @binding(0) var<storage, read> conditioning: array<f32>;
@group(0) @binding(1) var<storage, read> atom_mask: array<f32>;
@group(0) @binding(2) var<storage, read> gathers: array<i32>;
@group(0) @binding(3) var<storage, read> trunk_projected: array<f32>;
@group(0) @binding(4) var<storage, read_write> queries_cond: array<f32>;
@group(0) @binding(5) var<storage, read_write> queries_mask: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= QUERY_ROWS) { return; }
  let atom_live = gathers[G_TA_MASK + row] != 0;
  let atom_from = u32(max(gathers[G_TA_IDX + row], 0));
  let token_live = gathers[G_TQ_MASK + row] != 0;
  let token_from = u32(max(gathers[G_TQ_IDX + row], 0));

  var mask_value = 0.0;
  if (atom_live) { mask_value = atom_mask[atom_from]; }
  queries_mask[row] = mask_value;

  for (var c = 0u; c < C; c += 1u) {
    var value = 0.0;
    if (atom_live) { value = conditioning[atom_from * C + c]; }
    // The trunk's single conditioning, broadcast per token.
    if (token_live) { value += trunk_projected[token_from * C + c]; }
    // 🔴 MASKED HERE, before the keys gather from it.
    queries_cond[row * C + c] = value * mask_value;
  }
}`;

  const buildKeys = `${common}
@group(0) @binding(0) var<storage, read> queries_cond: array<f32>;
@group(0) @binding(1) var<storage, read> queries_mask: array<f32>;
@group(0) @binding(2) var<storage, read> gathers: array<i32>;
@group(0) @binding(3) var<storage, read_write> keys_cond: array<f32>;
@group(0) @binding(4) var<storage, read_write> keys_mask: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= KEY_ROWS) { return; }
  let live = gathers[G_QK_MASK + row] != 0;
  let source = u32(max(gathers[G_QK_IDX + row], 0));
  var mask_value = 0.0;
  if (live) { mask_value = queries_mask[source]; }
  keys_mask[row] = mask_value;
  for (var c = 0u; c < C; c += 1u) {
    var value = 0.0;
    if (live) { value = queries_cond[source * C + c]; }
    keys_cond[row * C + c] = value;
  }
}`;

  // The query activation: the conditioning plus the projected noisy positions.
  const buildAct = `${common}
@group(0) @binding(0) var<storage, read> queries_cond: array<f32>;
@group(0) @binding(1) var<storage, read> queries_mask: array<f32>;
@group(0) @binding(2) var<storage, read> positions: array<f32>;
@group(0) @binding(3) var<storage, read> gathers: array<i32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> act: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= QUERY_ROWS) { return; }
  var gathered = array<f32, 3>(0.0, 0.0, 0.0);
  if (gathers[G_TA_MASK + row] != 0) {
    let source = u32(max(gathers[G_TA_IDX + row], 0)) * 3u;
    gathered[0] = positions[source];
    gathered[1] = positions[source + 1u];
    gathered[2] = positions[source + 2u];
  }
  for (var c = 0u; c < C; c += 1u) {
    var value = 0.0;
    for (var axis = 0u; axis < 3u; axis += 1u) {
      value += gathered[axis] * weights[P_atomPositionsToFeatures + axis * C + c];
    }
    act[row * C + c] = queries_cond[row * C + c] + value * queries_mask[row];
  }
}`;

  // The atom pair representation: row + column, the reference-conformer offset
  // terms, the trunk pair, and then a three-layer MLP with a residual.
  const buildPair = `${common}
@group(0) @binding(0) var<storage, read> queries_cond: array<f32>;
@group(0) @binding(1) var<storage, read> keys_cond: array<f32>;
@group(0) @binding(2) var<storage, read> queries_ref: array<f32>;
@group(0) @binding(3) var<storage, read> keys_ref: array<f32>;
@group(0) @binding(4) var<storage, read> gathers: array<i32>;
@group(0) @binding(5) var<storage, read> trunk_pair: array<f32>;
@group(0) @binding(6) var<storage, read> weights: array<f32>;
@group(0) @binding(7) var<storage, read_write> pair: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIR_ROWS) { return; }
  let key = row % KEYS;
  let query_index = row / KEYS;
  let subset = query_index / QUERIES;
  let key_index = subset * KEYS + key;

  // 🔴 SAME REFERENCE SPACE, not "both real".
  var valid = 0.0;
  if (gathers[G_QSPACE + query_index] == gathers[G_KSPACE + key_index]) { valid = 1.0; }
  var offsets = array<f32, 3>(0.0, 0.0, 0.0);
  var squared = 0.0;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let d = queries_ref[query_index * 3u + axis] - keys_ref[key_index * 3u + axis];
    offsets[axis] = d;
    squared += d * d;
  }

  let use_trunk = gathers[G_TQ_MASK + query_index] != 0 && gathers[G_TK_MASK + key_index] != 0;
  var trunk_base = 0u;
  if (use_trunk) {
    trunk_base = (u32(max(gathers[G_TQ_IDX + query_index], 0)) * TOKENS
      + u32(max(gathers[G_TK_IDX + key_index], 0))) * C_PAIR;
  }

  for (var c = 0u; c < C_PAIR; c += 1u) {
    var value = 0.0;
    for (var d = 0u; d < C; d += 1u) {
      value += relu(queries_cond[query_index * C + d])
        * weights[P_singleToPairCondRow + d * C_PAIR + c];
      value += relu(keys_cond[key_index * C + d])
        * weights[P_singleToPairCondCol + d * C_PAIR + c];
    }
    var offset_term = 0.0;
    for (var axis = 0u; axis < 3u; axis += 1u) {
      offset_term += offsets[axis] * weights[P_embedPairOffsets + axis * C_PAIR + c];
    }
    value += valid * (offset_term + weights[P_embedPairDistances + c] / (1.0 + squared))
      + valid * weights[P_embedPairOffsetsValid + c];
    if (use_trunk) { value += trunk_pair[trunk_base + c]; }
    pair[row * C_PAIR + c] = value;
  }

  // The three-layer MLP, on a relu of the value just written, plus a residual.
  var hidden1 = array<f32, ${pairChannels}>();
  var hidden2 = array<f32, ${pairChannels}>();
  for (var c = 0u; c < C_PAIR; c += 1u) {
    var value = 0.0;
    for (var d = 0u; d < C_PAIR; d += 1u) {
      value += relu(pair[row * C_PAIR + d]) * weights[P_pairMlp1 + d * C_PAIR + c];
    }
    hidden1[c] = value;
  }
  for (var c = 0u; c < C_PAIR; c += 1u) {
    var value = 0.0;
    for (var d = 0u; d < C_PAIR; d += 1u) {
      value += relu(hidden1[d]) * weights[P_pairMlp2 + d * C_PAIR + c];
    }
    hidden2[c] = value;
  }
  for (var c = 0u; c < C_PAIR; c += 1u) {
    var value = 0.0;
    for (var d = 0u; d < C_PAIR; d += 1u) {
      value += relu(hidden2[d]) * weights[P_pairMlp3 + d * C_PAIR + c];
    }
    pair[row * C_PAIR + c] = pair[row * C_PAIR + c] + value;
  }
}`;

  // Per-block head biases from the atom pair representation.
  const pairLogits = `${common}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIR_ROWS) { return; }
  let base = row * C_PAIR;
  var total = 0.0;
  for (var c = 0u; c < C_PAIR; c += 1u) { total += pair[base + c]; }
  let mean = total / f32(C_PAIR);
  var variance = 0.0;
  for (var c = 0u; c < C_PAIR; c += 1u) {
    let d = pair[base + c] - mean;
    variance += d * d;
  }
  let inverse = inverseSqrt(variance / f32(C_PAIR) + EPSILON);

  let key = row % KEYS;
  let query_index = row / KEYS;
  let subset = query_index / QUERIES;
  let query = query_index % QUERIES;
  for (var block = 0u; block < BLOCKS; block += 1u) {
    for (var head = 0u; head < HEADS; head += 1u) {
      var value = 0.0;
      for (var c = 0u; c < C_PAIR; c += 1u) {
        value += (pair[base + c] - mean) * inverse
          * weights[P_pairInputLayerNormScale + c]
          * weights[P_pairLogitsProjection + c * BLOCKS * HEADS + block * HEADS + head];
      }
      let out = ((block * SUBSETS + subset) * HEADS + head) * QUERIES * KEYS
        + query * KEYS + key;
      logits[out] = value;
    }
  }
}`;

  return { trunkSingle, trunkPair, buildQueries, buildKeys, buildAct, buildPair, pairLogits,
           ...createAtomBlockShaders(common, shape) };
}

/** The three cross-attention blocks. */
export function createAtomBlockShaders(common, shape) {
  const { channels, keys } = shape;
  const intermediate = channels * 2;

  // AdaLN on queries and keys (different prefixes), then q/k/v/gate.
  const project = `${common}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> queries_cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read_write> gate: array<f32>;

var<workgroup> xq: array<f32, ${channels}>;
var<workgroup> qcond: array<f32, ${channels}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= QUERY_ROWS) { return; }
  let local = local_id.x;
  var total = 0.0;
  for (var c = 0u; c < C; c += 1u) { total += act[row * C + c]; }
  let mean = total / f32(C);
  var variance = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    let d = act[row * C + c] - mean;
    variance += d * d;
  }
  let inverse = inverseSqrt(variance / f32(C) + EPSILON);

  var cond_total = 0.0;
  for (var c = 0u; c < C; c += 1u) { cond_total += queries_cond[row * C + c]; }
  let cond_mean = cond_total / f32(C);
  var cond_variance = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    let d = queries_cond[row * C + c] - cond_mean;
    cond_variance += d * d;
  }
  let cond_inverse = inverseSqrt(cond_variance / f32(C) + EPSILON);
  for (var c = local; c < C; c += 64u) {
    qcond[c] = (queries_cond[row * C + c] - cond_mean) * cond_inverse
      * weights[W_qSingleCondLayerNormScale + c];
  }
  workgroupBarrier();
  for (var c = local; c < C; c += 64u) {
    var scale_value = weights[W_qSingleCondScaleBias + c];
    var shift = 0.0;
    for (var d = 0u; d < C; d += 1u) {
      scale_value += qcond[d] * weights[W_qSingleCondScaleWeights + d * C + c];
      shift += qcond[d] * weights[W_qSingleCondBias + d * C + c];
    }
    xq[c] = logistic(scale_value) * ((act[row * C + c] - mean) * inverse) + shift;
  }
  workgroupBarrier();

  for (var out = local; out < WIDTH; out += 64u) {
    var q_total = weights[W_qBias + out];
    var gate_total = 0.0;
    for (var c = 0u; c < C; c += 1u) {
      q_total += xq[c] * weights[W_qProjection + c * WIDTH + out];
      gate_total += xq[c] * weights[W_gatingQuery + c * WIDTH + out];
    }
    q[row * WIDTH + out] = q_total;
    gate[row * WIDTH + out] = gate_total;
  }
}`;

  // The key side is a separate dispatch because there are more key rows than
  // query rows, and they gather the activation through queries_to_keys.
  const projectKeys = `${common}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> keys_cond: array<f32>;
@group(0) @binding(2) var<storage, read> gathers: array<i32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> k: array<f32>;
@group(0) @binding(5) var<storage, read_write> v: array<f32>;

var<workgroup> xk: array<f32, ${channels}>;
var<workgroup> kcond: array<f32, ${channels}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= KEY_ROWS) { return; }
  let local = local_id.x;
  let live = gathers[G_QK_MASK + row] != 0;
  let source = u32(max(gathers[G_QK_IDX + row], 0));

  var total = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    var value = 0.0;
    if (live) { value = act[source * C + c]; }
    total += value;
  }
  let mean = total / f32(C);
  var variance = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    var value = 0.0;
    if (live) { value = act[source * C + c]; }
    let d = value - mean;
    variance += d * d;
  }
  let inverse = inverseSqrt(variance / f32(C) + EPSILON);

  var cond_total = 0.0;
  for (var c = 0u; c < C; c += 1u) { cond_total += keys_cond[row * C + c]; }
  let cond_mean = cond_total / f32(C);
  var cond_variance = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    let d = keys_cond[row * C + c] - cond_mean;
    cond_variance += d * d;
  }
  let cond_inverse = inverseSqrt(cond_variance / f32(C) + EPSILON);
  for (var c = local; c < C; c += 64u) {
    kcond[c] = (keys_cond[row * C + c] - cond_mean) * cond_inverse
      * weights[W_kSingleCondLayerNormScale + c];
  }
  workgroupBarrier();
  for (var c = local; c < C; c += 64u) {
    var scale_value = weights[W_kSingleCondScaleBias + c];
    var shift = 0.0;
    for (var d = 0u; d < C; d += 1u) {
      scale_value += kcond[d] * weights[W_kSingleCondScaleWeights + d * C + c];
      shift += kcond[d] * weights[W_kSingleCondBias + d * C + c];
    }
    var value = 0.0;
    if (live) { value = act[source * C + c]; }
    xk[c] = logistic(scale_value) * ((value - mean) * inverse) + shift;
  }
  workgroupBarrier();

  for (var out = local; out < WIDTH; out += 64u) {
    var k_total = 0.0;
    var v_total = 0.0;
    for (var c = 0u; c < C; c += 1u) {
      k_total += xk[c] * weights[W_kProjection + c * WIDTH + out];
      v_total += xk[c] * weights[W_vProjection + c * WIDTH + out];
    }
    k[row * WIDTH + out] = k_total;
    v[row * WIDTH + out] = v_total;
  }
}`;
  const projectKeysAtoms = `${common}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> queries_cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

var<workgroup> xk: array<f32, ${channels}>;
var<workgroup> kcond: array<f32, ${channels}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= QUERY_ROWS) { return; }
  let local = local_id.x;
  // ...every atom is its own source here, and there is no dead slot: the key
  // layout's padding is applied when this is expanded into it.
  let live = true;
  let source = row;

  var total = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    var value = 0.0;
    if (live) { value = act[source * C + c]; }
    total += value;
  }
  let mean = total / f32(C);
  var variance = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    var value = 0.0;
    if (live) { value = act[source * C + c]; }
    let d = value - mean;
    variance += d * d;
  }
  let inverse = inverseSqrt(variance / f32(C) + EPSILON);

  var cond_total = 0.0;
  for (var c = 0u; c < C; c += 1u) { cond_total += queries_cond[row * C + c]; }
  let cond_mean = cond_total / f32(C);
  var cond_variance = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    let d = queries_cond[row * C + c] - cond_mean;
    cond_variance += d * d;
  }
  let cond_inverse = inverseSqrt(cond_variance / f32(C) + EPSILON);
  for (var c = local; c < C; c += 64u) {
    kcond[c] = (queries_cond[row * C + c] - cond_mean) * cond_inverse
      * weights[W_kSingleCondLayerNormScale + c];
  }
  workgroupBarrier();
  for (var c = local; c < C; c += 64u) {
    var scale_value = weights[W_kSingleCondScaleBias + c];
    var shift = 0.0;
    for (var d = 0u; d < C; d += 1u) {
      scale_value += kcond[d] * weights[W_kSingleCondScaleWeights + d * C + c];
      shift += kcond[d] * weights[W_kSingleCondBias + d * C + c];
    }
    var value = 0.0;
    if (live) { value = act[source * C + c]; }
    xk[c] = logistic(scale_value) * ((value - mean) * inverse) + shift;
  }
  workgroupBarrier();

  for (var out = local; out < WIDTH; out += 64u) {
    var k_total = 0.0;
    var v_total = 0.0;
    for (var c = 0u; c < C; c += 1u) {
      k_total += xk[c] * weights[W_kProjection + c * WIDTH + out];
      v_total += xk[c] * weights[W_vProjection + c * WIDTH + out];
    }
    k[row * WIDTH + out] = k_total;
    v[row * WIDTH + out] = v_total;
  }
}`;

  // 🔴 THE KEY ROWS ARE A GATHER OF THE QUERY ROWS, SO THEIR PROJECTION IS ONE
  // TOO. queries_to_keys maps 45x128 key slots onto 1440 atoms - about four
  // slots an atom - and projectKeys recomputed the identical LayerNorm, AdaLN
  // and k/v projection for every one of them. Everything it reads for a key row
  // is a function of that row's SOURCE atom: keys_cond is built as a gather of
  // queries_cond, and the activation is read through the same index. So the
  // projection runs once per atom and this expands it: a quarter of the work
  // for the same numbers.
  //
  // 🔴 AND A DEAD SLOT MUST WRITE ZERO, not the atom it happens to point at.
  // The old kernel got that from `live` gating its reads; here the mask lives
  // in the expansion, and dropping it would feed the attention real keys where
  // it expects padding.
  const expandKeys = `${common}
@group(0) @binding(0) var<storage, read> k_atoms: array<f32>;
@group(0) @binding(1) var<storage, read> v_atoms: array<f32>;
@group(0) @binding(2) var<storage, read> gathers: array<i32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x + id.y * GRID_WIDTH * 64u;
  if (slot >= KEY_ROWS * WIDTH) { return; }
  let row = slot / WIDTH;
  let out = slot % WIDTH;
  var k_value = 0.0;
  var v_value = 0.0;
  if (gathers[G_QK_MASK + row] != 0) {
    let source = u32(max(gathers[G_QK_IDX + row], 0));
    k_value = k_atoms[source * WIDTH + out];
    v_value = v_atoms[source * WIDTH + out];
  }
  k[slot] = k_value;
  v[slot] = v_value;
}`;

  const attendFor = (block) => `${common}
const BLOCK: u32 = ${block}u;
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> pair_logits: array<f32>;
@group(0) @binding(4) var<storage, read> queries_mask: array<f32>;
@group(0) @binding(5) var<storage, read> keys_mask: array<f32>;
@group(0) @binding(6) var<storage, read_write> gathered: array<f32>;

var<workgroup> logits: array<f32, ${keys}>;
var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let slot = group.x + group.y * GRID_WIDTH;
  if (slot >= QUERY_ROWS * HEADS) { return; }
  let head = slot % HEADS;
  let query_index = slot / HEADS;
  let subset = query_index / QUERIES;
  let query = query_index % QUERIES;
  let local = local_id.x;
  let query_base = query_index * WIDTH + head * DIMENSION;

  for (var key = local; key < KEYS; key += 64u) {
    let key_index = subset * KEYS + key;
    var dot = 0.0;
    for (var d = 0u; d < DIMENSION; d += 1u) {
      dot += q[query_base + d] * k[key_index * WIDTH + head * DIMENSION + d];
    }
    // 🔴 A PRODUCT, NOT A SUM: only a padded query AND a padded key is penalised.
    let bias = 1.0e9 * (queries_mask[query_index] - 1.0) * (keys_mask[key_index] - 1.0);
    logits[key] = dot * SCALE + bias
      + pair_logits[(((BLOCK * SUBSETS + subset) * HEADS + head) * QUERIES + query) * KEYS + key];
  }
  workgroupBarrier();

  var local_max = -3.0e38;
  for (var key = local; key < KEYS; key += 64u) { local_max = max(local_max, logits[key]); }
  reduce[local] = local_max;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
    workgroupBarrier();
  }
  let largest = reduce[0];
  workgroupBarrier();

  var local_sum = 0.0;
  for (var key = local; key < KEYS; key += 64u) {
    let value = exp(logits[key] - largest);
    logits[key] = value;
    local_sum += value;
  }
  reduce[local] = local_sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let total = reduce[0];
  workgroupBarrier();

  for (var d = local; d < DIMENSION; d += 64u) {
    var sum = 0.0;
    for (var key = 0u; key < KEYS; key += 1u) {
      sum += logits[key] * v[(subset * KEYS + key) * WIDTH + head * DIMENSION + d];
    }
    gathered[query_index * WIDTH + head * DIMENSION + d] = sum / total;
  }
}`;

  // The zero-init gate, the residual, and the conditioned transition - all of
  // which read the query conditioning RAW rather than normalised.
  const output = `${common}
@group(0) @binding(0) var<storage, read> gathered: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> queries_cond: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> act: array<f32>;

var<workgroup> gated: array<f32, ${channels}>;
var<workgroup> after: array<f32, ${channels}>;
var<workgroup> cond_norm: array<f32, ${channels}>;
var<workgroup> x: array<f32, ${channels}>;
var<workgroup> wide: array<f32, ${intermediate}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= QUERY_ROWS) { return; }
  let local = local_id.x;

  for (var w = local; w < WIDTH; w += 64u) {
    gated[w] = gathered[row * WIDTH + w] * logistic(gate[row * WIDTH + w]);
  }
  workgroupBarrier();

  for (var c = local; c < C; c += 64u) {
    var projected = 0.0;
    for (var w = 0u; w < WIDTH; w += 1u) {
      projected += gated[w] * weights[W_Transition2 + w * C + c];
    }
    var zero_gate = weights[W_AdaptiveZeroCondBias + c];
    for (var d = 0u; d < C; d += 1u) {
      zero_gate += queries_cond[row * C + d] * weights[W_AdaptiveZeroCondWeights + d * C + c];
    }
    after[c] = act[row * C + c] + projected * logistic(zero_gate);
  }
  workgroupBarrier();

  // The transition reads the POST-attention activation.
  var total = 0.0;
  for (var c = 0u; c < C; c += 1u) { total += after[c]; }
  let mean = total / f32(C);
  var variance = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    let d = after[c] - mean;
    variance += d * d;
  }
  let inverse = inverseSqrt(variance / f32(C) + EPSILON);

  var cond_total = 0.0;
  for (var c = 0u; c < C; c += 1u) { cond_total += queries_cond[row * C + c]; }
  let cond_mean = cond_total / f32(C);
  var cond_variance = 0.0;
  for (var c = 0u; c < C; c += 1u) {
    let d = queries_cond[row * C + c] - cond_mean;
    cond_variance += d * d;
  }
  let cond_inverse = inverseSqrt(cond_variance / f32(C) + EPSILON);
  for (var c = local; c < C; c += 64u) {
    cond_norm[c] = (queries_cond[row * C + c] - cond_mean) * cond_inverse
      * weights[W_ffwSingleCondLayerNormScale + c];
  }
  workgroupBarrier();
  for (var c = local; c < C; c += 64u) {
    var scale_value = weights[W_ffwSingleCondScaleBias + c];
    var shift = 0.0;
    for (var d = 0u; d < C; d += 1u) {
      scale_value += cond_norm[d] * weights[W_ffwSingleCondScaleWeights + d * C + c];
      shift += cond_norm[d] * weights[W_ffwSingleCondBias + d * C + c];
    }
    x[c] = logistic(scale_value) * ((after[c] - mean) * inverse) + shift;
  }
  workgroupBarrier();

  let doubled = INTERMEDIATE * 2u;
  for (var i = local; i < INTERMEDIATE; i += 64u) {
    var gate_value = 0.0;
    var value = 0.0;
    for (var c = 0u; c < C; c += 1u) {
      let column = W_ffwTransition1 + c * doubled;
      gate_value += x[c] * weights[column + i];
      value += x[c] * weights[column + INTERMEDIATE + i];
    }
    wide[i] = swish(gate_value) * value;
  }
  workgroupBarrier();

  for (var c = local; c < C; c += 64u) {
    var projected = 0.0;
    for (var i = 0u; i < INTERMEDIATE; i += 1u) {
      projected += wide[i] * weights[W_ffwTransition2 + i * C + c];
    }
    var zero_gate = weights[W_ffwAdaptiveZeroCondBias + c];
    for (var d = 0u; d < C; d += 1u) {
      zero_gate += queries_cond[row * C + d]
        * weights[W_ffwAdaptiveZeroCondWeights + d * C + c];
    }
    act[row * C + c] = after[c] + projected * logistic(zero_gate);
  }
}`;

  // 🔴 THE ACTIVATION IS MASKED AFTER THE BLOCKS AND BEFORE THE SKIP CONNECTION.
  // The decoder reads that skip connection, so a padded row carrying a value
  // here is not a cosmetic difference - it re-enters the model downstream.
  const maskAct = `${common}
@group(0) @binding(0) var<storage, read> queries_mask: array<f32>;
@group(0) @binding(1) var<storage, read_write> act: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= QUERY_ROWS * C) { return; }
  act[index] = act[index] * queries_mask[index / C];
}`;

  // Project to the token width, gather back to token-atom layout, relu,
  // and average over each token's REAL atoms only.
  const aggregate = `${common}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> queries_mask: array<f32>;
@group(0) @binding(2) var<storage, read> gathers: array<i32>;
@group(0) @binding(3) var<storage, read> atom_mask: array<f32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> token_act: array<f32>;

// 🔴 ONE THREAD PER (TOKEN, CHANNEL), NOT PER TOKEN. This used to dispatch
// ceil(TOKENS/64) workgroups with a thread to a token - ONE workgroup for a
// 59-residue protein - and each of those threads then walked 768 output
// channels x 24 atoms x 128 input channels, 2.4M multiply-adds on a single
// lane. It was 43 ms of the atom encoder's 82: more than the three
// cross-attention blocks put together, in the pass that only pools their
// output. Splitting the channel loop across the grid gives 45k work items
// where there were 59, and consecutive threads read consecutive weight
// columns, so the reads coalesce as well.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x + id.y * GRID_WIDTH * 64u;
  if (slot >= TOKENS * C_TOKEN) { return; }
  let token = slot / C_TOKEN;
  let c = slot % C_TOKEN;
  var count = 0.0;
  for (var atom = 0u; atom < DENSE; atom += 1u) { count += atom_mask[token * DENSE + atom]; }

  {
    var total = 0.0;
    for (var atom = 0u; atom < DENSE; atom += 1u) {
      let dense_slot = token * DENSE + atom;
      if (atom_mask[dense_slot] == 0.0) { continue; }
      if (gathers[G_QTA_MASK + dense_slot] == 0) { continue; }
      let source = u32(max(gathers[G_QTA_IDX + dense_slot], 0));
      var value = 0.0;
      for (var d = 0u; d < C; d += 1u) {
        value += act[source * C + d] * queries_mask[source]
          * weights[P_projectAtomFeaturesForAggr + d * C_TOKEN + c];
      }
      total += relu(value);
    }
    var scaled = 0.0;
    if (count > 0.0) { scaled = total / count; }
    token_act[token * C_TOKEN + c] = scaled;
  }
}`;

  return { project, projectKeys, projectKeysAtoms, expandKeys,
           attendFor, output, maskAct, aggregate };
}

export class Af3AtomEncoderGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {object} input shape, conditioning, atomMask, refPos, refSpaceUid,
   *   the five gathers, tokenAtomsAct, trunkSingleCond, trunkPairCond
   * @param {object} weights the pair tensors plus `blocks`
   */
  async run(input, weights, options = {}) {
    const { tokens, dense, subsets, queries, keys } = input.shape;
    const channels = weights.channels;
    const pairChannels = weights.pairChannels;
    const heads = weights.heads;
    const dimension = weights.dimension;
    const width = heads * dimension;
    const queryRows = subsets * queries;
    const keyRows = subsets * keys;
    const pairRows = subsets * queries * keys;
    const perTokenChannels = weights.perTokenChannels;

    const pairPacked = packAtomPairWeights(weights);
    const blockPacked = weights.blocks.map(packAtomBlockWeights);
    const shape = {
      tokens, dense, subsets, queries, keys, channels, pairChannels, heads, dimension,
      perTokenChannels, trunkSingleChannels: weights.trunkSingleChannels,
      trunkPairChannels: weights.trunkPairChannels, blocks: weights.blocks.length,
    };
    const sources = createAtomEncoderShaders(shape, pairPacked.offsets, blockPacked[0].offsets);
    const base = `af3-atom:${tokens}:${dense}:${subsets}:${queries}:${keys}`
      + `:${channels}:${pairChannels}:${heads}:${dimension}:${perTokenChannels}`;
    const compiled = {};
    for (const [name, source] of Object.entries(sources)) {
      if (name === "attendFor") continue;
      compiled[name] = await this.pipelines.get(`${base}:${name}`, source);
    }
    // 🔴 ONE attend PIPELINE PER BLOCK. All three blocks' head biases live in
    // one buffer, and the block index selects a slice - baked in, because the
    // pipeline cache takes no override constants.
    compiled.attend = [];
    for (let index = 0; index < weights.blocks.length; index += 1) {
      compiled.attend.push(await this.pipelines.get(
        `${base}:attend:${index}`, sources.attendFor(index)));
    }

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (a) => { allocations.push(a); return a; };
    const up = (label, data) => keep(this.allocator.upload(label, data, storage));
    const alloc = (label, bytes, extra = 0) =>
      keep(this.allocator.allocate(label, bytes, storage | extra));
    const ints = (source) => Int32Array.from(source, (v) => Number(v));
    const floats = (source) => Float32Array.from(source, (v) => Number(v));

    try {
      const conditioning = up("atom.cond", input.conditioning);
      const atomMask = up("atom.mask", floats(input.atomMask));
      const pairWeights = up("atom.pair-weights", pairPacked.data);
      const trunkSingleCond = up("atom.trunk-single", input.trunkSingleCond);
      const trunkPairCond = up("atom.trunk-pair", input.trunkPairCond);
      const positions = up("atom.positions", input.tokenAtomsAct);
      const refPos = up("atom.ref-pos", floats(input.refPos));
      const refSpaceUid = up("atom.ref-space", ints(input.refSpaceUid));

      // Every gather, and the two reference-space columns, in one i32 buffer -
      // see the note in the shader preamble about the eight-buffer guarantee.
      const flatSpace = ints(input.refSpaceUid);
      const queriesSpace = new Int32Array(queryRows);
      const keysSpace = new Int32Array(keyRows);
      // 🔴 A MASKED SLOT'S REFERENCE SPACE IS ZERO, NOT A SENTINEL. AF3 gathers
      // with a zero-filling convert, so two PADDED atoms both read 0, compare
      // equal, and are treated as sharing a reference conformer - which makes
      // their offset term live. Using -1 and -2 here to mark them "unrelated"
      // is the tidier choice and a different model; it cost 3.1e-2 on the atom
      // pair representation.
      for (let index = 0; index < queryRows; index += 1) {
        queriesSpace[index] = input.tokenAtomsToQueries.mask[index]
          ? flatSpace[Number(input.tokenAtomsToQueries.indices[index])] : 0;
      }
      for (let index = 0; index < keyRows; index += 1) {
        keysSpace[index] = input.queriesToKeys.mask[index]
          ? queriesSpace[Number(input.queriesToKeys.indices[index])] : 0;
      }
      const gathers = new Int32Array(5 * queryRows + 5 * keyRows + 2 * tokens * dense);
      let at = 0;
      const place = (source) => { gathers.set(ints(source), at); at += source.length; };
      place(input.tokenAtomsToQueries.indices);
      place(input.tokenAtomsToQueries.mask);
      place(input.tokensToQueries.indices);
      place(input.tokensToQueries.mask);
      place(input.queriesToKeys.indices);
      place(input.queriesToKeys.mask);
      place(input.tokensToKeys.indices);
      place(input.tokensToKeys.mask);
      place(input.queriesToTokenAtoms.indices);
      place(input.queriesToTokenAtoms.mask);
      gathers.set(queriesSpace, at); at += queryRows;
      gathers.set(keysSpace, at);
      const gatherBuffer = up("atom.gathers", gathers);

      // The reference positions in query and key layout, gathered on the host:
      // three floats each, and the gathers are integer indirection the GPU has
      // no reason to redo.
      const queriesRef = new Float32Array(queryRows * 3);
      const keysRef = new Float32Array(keyRows * 3);
      const flatRef = floats(input.refPos);
      for (let index = 0; index < queryRows; index += 1) {
        if (!input.tokenAtomsToQueries.mask[index]) continue;
        const from = Number(input.tokenAtomsToQueries.indices[index]) * 3;
        for (let axis = 0; axis < 3; axis += 1) queriesRef[index * 3 + axis] = flatRef[from + axis];
      }
      for (let index = 0; index < keyRows; index += 1) {
        if (!input.queriesToKeys.mask[index]) continue;
        const from = Number(input.queriesToKeys.indices[index]) * 3;
        for (let axis = 0; axis < 3; axis += 1) keysRef[index * 3 + axis] = queriesRef[from + axis];
      }
      const queriesRefBuffer = up("atom.q-ref", queriesRef);
      const keysRefBuffer = up("atom.k-ref", keysRef);

      const trunkSingleProjected = alloc("atom.trunk-single-p", tokens * channels * 4);
      const trunkPairProjected = alloc("atom.trunk-pair-p", tokens * tokens * pairChannels * 4);
      const queriesCond = alloc("atom.q-cond", queryRows * channels * 4, GPUBufferUsage.COPY_SRC);
      const queriesMask = alloc("atom.q-mask", queryRows * 4, GPUBufferUsage.COPY_SRC);
      const keysCond = alloc("atom.k-cond", keyRows * channels * 4, GPUBufferUsage.COPY_SRC);
      const keysMask = alloc("atom.k-mask", keyRows * 4, GPUBufferUsage.COPY_SRC);
      const act = alloc("atom.act", queryRows * channels * 4, GPUBufferUsage.COPY_SRC);
      const pair = alloc("atom.pair", pairRows * pairChannels * 4, GPUBufferUsage.COPY_SRC);
      const logits = alloc("atom.logits", weights.blocks.length * subsets * heads
        * queries * keys * 4);
      const q = alloc("atom.q", queryRows * width * 4);
      const k = alloc("atom.k", keyRows * width * 4);
      const v = alloc("atom.v", keyRows * width * 4);
      // ...one row an ATOM, expanded into the key layout below.
      const kAtoms = alloc("atom.k-atoms", queryRows * width * 4);
      const vAtoms = alloc("atom.v-atoms", queryRows * width * 4);
      const gate = alloc("atom.gate", queryRows * width * 4);
      const gathered = alloc("atom.gathered", queryRows * width * 4);
      const tokenAct = alloc("atom.token-act", tokens * perTokenChannels * 4,
        GPUBufferUsage.COPY_SRC);

      const readbacks = {
        tokenAct: keep(this.allocator.allocate("atom.rb-token", tokens * perTokenChannels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        skipConnection: keep(this.allocator.allocate("atom.rb-skip", queryRows * channels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        pairCond: keep(this.allocator.allocate("atom.rb-pair", pairRows * pairChannels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        // The decoder reads all four of these, so the head can chain the two
        // without a second encoder run.
        queriesCond: keep(this.allocator.allocate("atom.rb-qcond", queryRows * channels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        keysCond: keep(this.allocator.allocate("atom.rb-kcond", keyRows * channels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        queriesMask: keep(this.allocator.allocate("atom.rb-qmask", queryRows * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        keysMask: keep(this.allocator.allocate("atom.rb-kmask", keyRows * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
      };

      const blockBuffers = blockPacked.map((packed, index) =>
        up(`atom.block-${index}`, packed.data));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-atom-encoder" });
      const run = (label, pipeline, buffers, x, y = 1) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((a, binding) => ({ binding, resource: { buffer: a.buffer } })),
        }));
        pass.dispatchWorkgroups(x, y);
        pass.end();
      };
      const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
      const lin = (count) => spread(Math.ceil(count / 64));

      run("trunk-single", compiled.trunkSingle,
          [trunkSingleCond, pairWeights, trunkSingleProjected], Math.ceil(tokens / 64));
      const tp = lin(tokens * tokens);
      run("trunk-pair", compiled.trunkPair,
          [trunkPairCond, pairWeights, trunkPairProjected], tp[0], tp[1]);
      const qr = lin(queryRows);
      run("build-queries", compiled.buildQueries,
          [conditioning, atomMask, gatherBuffer, trunkSingleProjected, queriesCond,
           queriesMask], qr[0], qr[1]);
      const kr = lin(keyRows);
      run("build-keys", compiled.buildKeys,
          [queriesCond, queriesMask, gatherBuffer, keysCond, keysMask], kr[0], kr[1]);
      run("build-act", compiled.buildAct,
          [queriesCond, queriesMask, positions, gatherBuffer, pairWeights, act], qr[0], qr[1]);
      const pr = lin(pairRows);
      run("build-pair", compiled.buildPair,
          [queriesCond, keysCond, queriesRefBuffer, keysRefBuffer, gatherBuffer,
           trunkPairProjected, pairWeights, pair], pr[0], pr[1]);
      run("pair-logits", compiled.pairLogits, [pair, pairWeights, logits], pr[0], pr[1]);

      for (let index = 0; index < weights.blocks.length; index += 1) {
        const w = blockBuffers[index];
        const perQuery = spread(queryRows);
        run(`project-${index}`, compiled.project,
            [act, queriesCond, w, q, gate], perQuery[0], perQuery[1]);
        run(`project-keys-${index}`, compiled.projectKeysAtoms,
            [act, queriesCond, w, kAtoms, vAtoms], perQuery[0], perQuery[1]);
        const expand = lin(keyRows * width);
        run(`expand-keys-${index}`, compiled.expandKeys,
            [kAtoms, vAtoms, gatherBuffer, k, v], expand[0], expand[1]);
        // The per-block slice of the logits.
        const slots = spread(queryRows * heads);
        run(`attend-${index}`, compiled.attend[index],
            [q, k, v, logits, queriesMask, keysMask, gathered], slots[0], slots[1]);
        run(`output-${index}`, compiled.output,
            [gathered, gate, queriesCond, w, act], perQuery[0], perQuery[1]);
      }

      const maskGroups = lin(queryRows * channels);
      run("mask-act", compiled.maskAct, [queriesMask, act], maskGroups[0], maskGroups[1]);
      const aggregateGroups = lin(tokens * perTokenChannels);
      run("aggregate", compiled.aggregate,
          [act, queriesMask, gatherBuffer, atomMask, pairWeights, tokenAct],
          aggregateGroups[0], aggregateGroups[1]);

      encoder.copyBufferToBuffer(tokenAct.buffer, 0, readbacks.tokenAct.buffer, 0,
                                 tokens * perTokenChannels * 4);
      encoder.copyBufferToBuffer(act.buffer, 0, readbacks.skipConnection.buffer, 0,
                                 queryRows * channels * 4);
      // 🔴 FIVE OF THE SEVEN READBACKS ARE THE SAME EVERY CALL. pairCond,
      // queriesCond, keysCond and the two masks are built from the reference
      // conformers, the gathers and the trunk - not from the noisy positions
      // and not from the noise level - so a 200-step sampler copied ~14 MB back
      // from the device two hundred times to get identical arrays, and handed
      // them straight back to the decoder. `reuseStatic` is the head saying it
      // still has them. The GPU still COMPUTES them, because the attention
      // blocks below read the buffers; only the copy back is skipped.
      const reuseStatic = options.reuseStatic;
      if (reuseStatic === undefined) {
        encoder.copyBufferToBuffer(pair.buffer, 0, readbacks.pairCond.buffer, 0,
                                   pairRows * pairChannels * 4);
        encoder.copyBufferToBuffer(queriesCond.buffer, 0, readbacks.queriesCond.buffer, 0,
                                   queryRows * channels * 4);
        encoder.copyBufferToBuffer(keysCond.buffer, 0, readbacks.keysCond.buffer, 0,
                                   keyRows * channels * 4);
        encoder.copyBufferToBuffer(queriesMask.buffer, 0, readbacks.queriesMask.buffer, 0,
                                   queryRows * 4);
        encoder.copyBufferToBuffer(keysMask.buffer, 0, readbacks.keysMask.buffer, 0, keyRows * 4);
      }

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      const read = async (a) => {
        await a.buffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(a.buffer.getMappedRange().slice(0));
        a.buffer.unmap();
        return copy;
      };
      return {
        tokenAct: await read(readbacks.tokenAct),
        skipConnection: await read(readbacks.skipConnection),
        pairCond: reuseStatic?.pairCond ?? await read(readbacks.pairCond),
        queriesCond: reuseStatic?.queriesCond ?? await read(readbacks.queriesCond),
        keysCond: reuseStatic?.keysCond ?? await read(readbacks.keysCond),
        queriesMask: reuseStatic?.queriesMask ?? await read(readbacks.queriesMask),
        keysMask: reuseStatic?.keysMask ?? await read(readbacks.keysMask),
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
