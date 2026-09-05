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
import { residentWeightBuffer } from "../runtime/resident.js";
import { noteAllocation, noteDestroy } from "../runtime/device-memory.js";

/**
 * Which labels in a caller's `staticCache` already hold their contents.
 *
 * Kept beside the cache rather than in it: the head owns that object and
 * destroys every VALUE in it when a fold ends, so a bookkeeping set stored
 * there would be asked to destroy itself.
 */
const STATIC_UPLOADS = new WeakMap();

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


/**
 * 🔴 PACKED ONCE PER WEIGHT OBJECT, NOT ONCE PER CALL. Packing allocates and
 * memcpies the whole bundle, and a 200-step sampler ran this two hundred times
 * over weights that never change. The offsets are still needed on every call,
 * because the shader sources are generated from them, so the whole result is
 * cached rather than only the data.
 */
const packedOnce = new WeakMap();

export function packCached(key, label, pack) {
  let forKey = packedOnce.get(key);
  if (forKey === undefined) {
    forKey = new Map();
    packedOnce.set(key, forKey);
  }
  let found = forKey.get(label);
  if (found === undefined) {
    found = pack();
    forKey.set(label, found);
  }
  return found;
}

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
/**
 * How many query rows one `output` workgroup carries, given how many there are.
 *
 * 🔴 A FUNCTION OF THE ROW COUNT, LIKE EVERY OTHER TILE HERE. It divides the
 * 655 KB of weights each workgroup reads - see the note on the kernel - and it
 * multiplies nothing else, so the only thing pulling the other way is the
 * workgroup count: this kernel is 64 lanes wide, so a 68-mer's 576 query rows
 * are only 9,216 invocations at a tile of four. Measured on a denoiser call at
 * that size: tile 2 gives 15 ms of atom encoder and 21 of decoder, tile 4 gives
 * 16 and 22, tile 8 gives 20 and 26. A real protein has thousands of atoms and
 * wants the larger tile.
 */
export function outputRowTileFor(queryRows) {
  for (const tile of [8, 4, 2]) {
    if (queryRows / tile >= 256) return tile;
  }
  return 1;
}

export function createAtomBlockShaders(common, shape) {
  const outputRowTile = shape.outputRowTile
    ?? outputRowTileFor(shape.subsets * shape.queries);
  const { channels, keys } = shape;
  const intermediate = channels * 2;
  const rowWidth = Math.min(4, outputRowTile);
  const rowGroups = outputRowTile / rowWidth;
  const rowVector = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[rowWidth];
  if (rowVector === undefined || !Number.isInteger(rowGroups)) {
    throw new Error(`outputRowTile ${outputRowTile} is not 1, 2 or a multiple of 4`);
  }
  const rowLane = (t) => rowWidth === 1 ? "" : `.${"xyzw"[t % rowWidth]}`;
  const rowGroup = (t) => Math.floor(t / rowWidth);
  const overRows = (body) =>
    Array.from({ length: outputRowTile }, (_, t) => body(t)).join("\n    ");
  const overRowGroups = (body) =>
    Array.from({ length: rowGroups }, (_, g) => body(g)).join("\n    ");
  /** A per-row quantity gathered into the tile's vectors. */
  const gather = (name, expression) => `${overRowGroups((g) => `var ${name}${g} = ${rowVector}(0.0);`)}
    ${overRows((t) => `{
      let row = min(base_row + ${t}u, QUERY_ROWS - 1u);
      ${name}${rowGroup(t)}${rowLane(t)} = ${expression};
    }`)}`;

  /**
   * AdaLN on a row's activation, conditioned by that row's own conditioning,
   * then two projections of the result. The query side and the atom-wise key
   * side are the same operation with different weights and different outputs,
   * so they are generated from here.
   *
   * 🔴 A TILE OF ROWS, for the reason the `output` kernel gives: with one
   * workgroup a row each of the four matrices is read whole to produce a single
   * row - 262 KB of weights for 128 output values.
   */
  const conditionedProject = (prefix, options) => `${common}
const ROW_TILE: u32 = ${outputRowTile}u;
${options.bindings}

// 🔴 EACH HOLDS ITS RAW TENSOR FIRST AND ITS NORMALISED ONE AFTER, IN PLACE.
// act and queries_cond were both read from GLOBAL memory in four loops -
// the two passes of a shared LayerNorm and the two normalisations - and two of
// those run over every channel on EVERY lane rather than a strided share, so a
// workgroup of 64 issued tens of thousands of loads for 2 * C * ROW_TILE
// distinct values. Staged once they are that many loads and the rest are
// workgroup reads, and the loops collapse from ROW_TILE scalar operations to
// ROW_TILE/4 vector ones.
//
// 🔴 IN PLACE AND NOT IN TWO MORE ARRAYS, WHICH IS THE DIFFERENCE FROM THE
// output KERNEL. That one already held 24 of the device's 32 KiB, so a fifth
// array cost no residency; this one holds 8, and two more would take it to 16 -
// halving how many workgroups a core can keep. Neither raw tensor is wanted
// after its own normalisation, so each is overwritten where it lies. The
// barrier below is what makes that safe.
var<workgroup> xq: array<${rowVector}, ${rowGroups * channels}>;
var<workgroup> qcond: array<${rowVector}, ${rowGroups * channels}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_row = (group.x + group.y * GRID_WIDTH) * ROW_TILE;
  if (base_row >= QUERY_ROWS) { return; }
  let local = local_id.x;

  // ...a row past the end is clamped rather than skipped: every lane reaches
  // the barriers, and its lane of each vector is dropped at the write.
  for (var c = local; c < C; c += 64u) {
    ${overRows((t) => `{
      let row = min(base_row + ${t}u, QUERY_ROWS - 1u);
      xq[${rowGroup(t)}u * C + c]${rowLane(t)} = act[row * C + c];
      qcond[${rowGroup(t)}u * C + c]${rowLane(t)} = queries_cond[row * C + c];
    }`)}
  }
  workgroupBarrier();

  ${overRowGroups((g) => `var total${g} = ${rowVector}(0.0);
  var cond_total${g} = ${rowVector}(0.0);`)}
  for (var c = 0u; c < C; c += 1u) {
    ${overRowGroups((g) => `total${g} += xq[${g}u * C + c];
    cond_total${g} += qcond[${g}u * C + c];`)}
  }
  ${overRowGroups((g) => `let mean${g} = total${g} / ${rowVector}(f32(C));
  let cond_mean${g} = cond_total${g} / ${rowVector}(f32(C));
  var variance${g} = ${rowVector}(0.0);
  var cond_variance${g} = ${rowVector}(0.0);`)}
  for (var c = 0u; c < C; c += 1u) {
    ${overRowGroups((g) => `{
      let d = xq[${g}u * C + c] - mean${g};
      variance${g} += d * d;
      let e = qcond[${g}u * C + c] - cond_mean${g};
      cond_variance${g} += e * e;
    }`)}
  }
  ${overRowGroups((g) => `let inverse${g} =
    inverseSqrt(variance${g} / ${rowVector}(f32(C)) + ${rowVector}(EPSILON));
  let cond_inverse${g} =
    inverseSqrt(cond_variance${g} / ${rowVector}(f32(C)) + ${rowVector}(EPSILON));`)}

  // ...the two loops above read every slot on every lane, so nothing may be
  // overwritten until all of them are past it.
  workgroupBarrier();
  for (var c = local; c < C; c += 64u) {
    let scale = weights[W_${prefix}SingleCondLayerNormScale + c];
    ${overRowGroups((g) => `qcond[${g}u * C + c] =
      (qcond[${g}u * C + c] - cond_mean${g}) * cond_inverse${g} * scale;`)}
  }
  workgroupBarrier();
  for (var c = local; c < C; c += 64u) {
    ${overRowGroups((g) =>
      `var scale_value${g} = ${rowVector}(weights[W_${prefix}SingleCondScaleBias + c]);
    var shift${g} = ${rowVector}(0.0);`)}
    for (var d = 0u; d < C; d += 1u) {
      let ws = weights[W_${prefix}SingleCondScaleWeights + d * C + c];
      let wb = weights[W_${prefix}SingleCondBias + d * C + c];
      ${overRowGroups((g) => `{
        let cn = qcond[${g}u * C + d];
        scale_value${g} += cn * ws;
        shift${g} += cn * wb;
      }`)}
    }
    // ...written out rather than through logistic(), which takes an f32; this
    // is its definition applied to the whole vector, so the arithmetic is the
    // same one. The output kernel above spells it the same way.
    ${overRowGroups((g) => `xq[${g}u * C + c] =
      ${rowVector}(1.0) / (${rowVector}(1.0) + exp(-scale_value${g}))
      * ((xq[${g}u * C + c] - mean${g}) * inverse${g}) + shift${g};`)}
  }
  workgroupBarrier();

  for (var out = local; out < WIDTH; out += 64u) {
    ${overRowGroups((g) =>
      `var a${g} = ${rowVector}(${options.biasA ? `weights[${options.biasA} + out]` : "0.0"});
    var b${g} = ${rowVector}(0.0);`)}
    for (var c = 0u; c < C; c += 1u) {
      // ...read once, used by every row of the tile.
      let wa = weights[${options.weightA} + c * WIDTH + out];
      let wb = weights[${options.weightB} + c * WIDTH + out];
      ${overRowGroups((g) => `{
        let x = xq[${g}u * C + c];
        a${g} += x * wa;
        b${g} += x * wb;
      }`)}
    }
    ${overRows((t) => `{
      let row = base_row + ${t}u;
      if (row < QUERY_ROWS) {
        ${options.outA}[row * WIDTH + out] = a${rowGroup(t)}${rowLane(t)};
        ${options.outB}[row * WIDTH + out] = b${rowGroup(t)}${rowLane(t)};
      }
    }`)}
  }
}`;

  const project = conditionedProject("q", {
    bindings: `@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> queries_cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read_write> gate: array<f32>;`,
    weightA: "W_qProjection", biasA: "W_qBias", outA: "q",
    weightB: "W_gatingQuery", outB: "gate",
  });

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
  const projectKeysAtoms = conditionedProject("k", {
    bindings: `@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> queries_cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;`,
    weightA: "W_kProjection", outA: "k",
    weightB: "W_vProjection", outB: "v",
  });

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
  //
  // 🔴 A TILE OF QUERY ROWS, BECAUSE ONE WORKGROUP A ROW READ 655 KB OF WEIGHTS
  // TO PRODUCE ONE. This kernel fuses five matmuls - the attention's output
  // projection, the zero-init gate, the conditioned scale and shift, the
  // widening and the way back - and with a workgroup per row every one of them
  // read its whole matrix for a single row: 377 MB a block, and the three
  // blocks of a decoder call were 10.5 ms of 24. The rows share every weight
  // and share nothing else, so a tile of them is exactly the vector: one read,
  // one vector multiply-add, four rows.
  const output = `${common}
const ROW_TILE: u32 = ${outputRowTile}u;
@group(0) @binding(0) var<storage, read> gathered: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> queries_cond: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> act: array<f32>;

var<workgroup> gated: array<${rowVector}, ${rowGroups * channels}>;
var<workgroup> after: array<${rowVector}, ${rowGroups * channels}>;
// 🔴 THE RAW CONDITIONING, STAGED ONCE, AND IT IS NOT cond_norm. It is read
// in FIVE places - the two adaptive-zero projections, the two passes of its own
// LayerNorm, and the normalisation itself - and it was read from GLOBAL memory
// in every one, once per row of the tile. Two of those loops run over every
// channel on EVERY lane rather than a strided share, so a workgroup of 64 was
// issuing tens of thousands of global loads for the C * ROW_TILE distinct
// values it needed. Staged, that is C * ROW_TILE loads and the rest are
// workgroup reads - and because the stage is a vector over the tile's rows, the
// loops reading it collapse from ROW_TILE scalar operations to ROW_TILE/4
// vector ones.
//
// 🔴 IT CANNOT SHARE cond_norm's SLOTS, which is the first thing to try and
// is wrong. The SECOND adaptive-zero projection reads the RAW conditioning -
// see the note where it does - and it runs after the normalised form has been
// written. Normalising in place would feed it the wrong tensor silently, and
// nothing here would fail. It costs 4 KiB and no residency: this kernel already
// holds 24 of the device's 32 KiB, so it is one workgroup a core either way.
var<workgroup> cond_raw: array<${rowVector}, ${rowGroups * channels}>;
var<workgroup> cond_norm: array<${rowVector}, ${rowGroups * channels}>;
var<workgroup> x: array<${rowVector}, ${rowGroups * channels}>;
var<workgroup> wide: array<${rowVector}, ${rowGroups * intermediate}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_row = (group.x + group.y * GRID_WIDTH) * ROW_TILE;
  if (base_row >= QUERY_ROWS) { return; }
  let local = local_id.x;

  // ...a row past the end is clamped rather than skipped: every lane reaches
  // the barriers, and its lane of each vector is dropped at the write.
  for (var w = local; w < WIDTH; w += 64u) {
    ${overRows((t) => `{
      let row = min(base_row + ${t}u, QUERY_ROWS - 1u);
      gated[${rowGroup(t)}u * WIDTH + w]${rowLane(t)} =
        gathered[row * WIDTH + w] * logistic(gate[row * WIDTH + w]);
    }`)}
  }
  for (var c = local; c < C; c += 64u) {
    ${overRows((t) => `{
      let row = min(base_row + ${t}u, QUERY_ROWS - 1u);
      cond_raw[${rowGroup(t)}u * C + c]${rowLane(t)} = queries_cond[row * C + c];
    }`)}
  }
  workgroupBarrier();

  for (var c = local; c < C; c += 64u) {
    ${overRowGroups((g) => `var projected${g} = ${rowVector}(0.0);`)}
    for (var w = 0u; w < WIDTH; w += 1u) {
      // ...read once, used by every row of the tile.
      let weight = weights[W_Transition2 + w * C + c];
      ${overRowGroups((g) => `projected${g} += gated[${g}u * WIDTH + w] * weight;`)}
    }
    ${overRowGroups((g) => `var zero${g} = ${rowVector}(weights[W_AdaptiveZeroCondBias + c]);`)}
    for (var d = 0u; d < C; d += 1u) {
      let weight = weights[W_AdaptiveZeroCondWeights + d * C + c];
      ${overRowGroups((g) => `zero${g} += cond_raw[${g}u * C + d] * weight;`)}
    }
    ${overRows((t) => `{
      let row = min(base_row + ${t}u, QUERY_ROWS - 1u);
      after[${rowGroup(t)}u * C + c]${rowLane(t)} = act[row * C + c]
        + projected${rowGroup(t)}${rowLane(t)} * logistic(zero${rowGroup(t)}${rowLane(t)});
    }`)}
  }
  workgroupBarrier();

  // The transition reads the POST-attention activation.
  ${overRowGroups((g) => `var total${g} = ${rowVector}(0.0);`)}
  for (var c = 0u; c < C; c += 1u) {
    ${overRowGroups((g) => `total${g} += after[${g}u * C + c];`)}
  }
  ${overRowGroups((g) => `let mean${g} = total${g} / ${rowVector}(f32(C));`)}
  ${overRowGroups((g) => `var variance${g} = ${rowVector}(0.0);`)}
  for (var c = 0u; c < C; c += 1u) {
    ${overRowGroups((g) => `{
      let d = after[${g}u * C + c] - mean${g};
      variance${g} += d * d;
    }`)}
  }
  ${overRowGroups((g) => `let inverse${g} =
    inverseSqrt(variance${g} / ${rowVector}(f32(C)) + ${rowVector}(EPSILON));`)}

  ${overRowGroups((g) => `var cond_total${g} = ${rowVector}(0.0);`)}
  for (var c = 0u; c < C; c += 1u) {
    ${overRowGroups((g) => `cond_total${g} += cond_raw[${g}u * C + c];`)}
  }
  ${overRowGroups((g) => `let cond_mean${g} = cond_total${g} / ${rowVector}(f32(C));`)}
  ${overRowGroups((g) => `var cond_variance${g} = ${rowVector}(0.0);`)}
  for (var c = 0u; c < C; c += 1u) {
    ${overRowGroups((g) => `{
      let d = cond_raw[${g}u * C + c] - cond_mean${g};
      cond_variance${g} += d * d;
    }`)}
  }
  ${overRowGroups((g) => `let cond_inverse${g} =
    inverseSqrt(cond_variance${g} / ${rowVector}(f32(C)) + ${rowVector}(EPSILON));`)}
  for (var c = local; c < C; c += 64u) {
    let scale = weights[W_ffwSingleCondLayerNormScale + c];
    ${overRowGroups((g) => `cond_norm[${g}u * C + c] =
      (cond_raw[${g}u * C + c] - cond_mean${g}) * cond_inverse${g} * scale;`)}
  }
  workgroupBarrier();
  for (var c = local; c < C; c += 64u) {
    ${overRowGroups((g) =>
      `var scale_value${g} = ${rowVector}(weights[W_ffwSingleCondScaleBias + c]);
    var shift${g} = ${rowVector}(0.0);`)}
    for (var d = 0u; d < C; d += 1u) {
      let ws = weights[W_ffwSingleCondScaleWeights + d * C + c];
      let wb = weights[W_ffwSingleCondBias + d * C + c];
      ${overRowGroups((g) => `{
        let cn = cond_norm[${g}u * C + d];
        scale_value${g} += cn * ws;
        shift${g} += cn * wb;
      }`)}
    }
    ${overRowGroups((g) => `x[${g}u * C + c] =
      ${rowVector}(1.0) / (${rowVector}(1.0) + exp(-scale_value${g}))
      * ((after[${g}u * C + c] - mean${g}) * inverse${g}) + shift${g};`)}
  }
  workgroupBarrier();

  let doubled = INTERMEDIATE * 2u;
  for (var i = local; i < INTERMEDIATE; i += 64u) {
    ${overRowGroups((g) => `var gate_value${g} = ${rowVector}(0.0);
    var value${g} = ${rowVector}(0.0);`)}
    for (var c = 0u; c < C; c += 1u) {
      let column = W_ffwTransition1 + c * doubled;
      let wg = weights[column + i];
      let wv = weights[column + INTERMEDIATE + i];
      ${overRowGroups((g) => `{
        let xc = x[${g}u * C + c];
        gate_value${g} += xc * wg;
        value${g} += xc * wv;
      }`)}
    }
    ${overRowGroups((g) => `wide[${g}u * INTERMEDIATE + i] =
      gate_value${g} / (${rowVector}(1.0) + exp(-gate_value${g})) * value${g};`)}
  }
  workgroupBarrier();

  for (var c = local; c < C; c += 64u) {
    ${overRowGroups((g) => `var projected${g} = ${rowVector}(0.0);`)}
    for (var i = 0u; i < INTERMEDIATE; i += 1u) {
      let weight = weights[W_ffwTransition2 + i * C + c];
      ${overRowGroups((g) => `projected${g} += wide[${g}u * INTERMEDIATE + i] * weight;`)}
    }
    ${overRowGroups((g) =>
      `var zero${g} = ${rowVector}(weights[W_ffwAdaptiveZeroCondBias + c]);`)}
    for (var d = 0u; d < C; d += 1u) {
      let weight = weights[W_ffwAdaptiveZeroCondWeights + d * C + c];
      ${overRowGroups((g) => `zero${g} += cond_raw[${g}u * C + d] * weight;`)}
    }
    ${overRows((t) => `{
      let row = base_row + ${t}u;
      if (row < QUERY_ROWS) {
        act[row * C + c] = after[${rowGroup(t)}u * C + c]${rowLane(t)}
          + projected${rowGroup(t)}${rowLane(t)}
            * logistic(zero${rowGroup(t)}${rowLane(t)});
      }
    }`)}
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
           attendFor, output, maskAct, aggregate, outputRowTile };
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

    const pairPacked = packCached(weights, "atom.pair", () => packAtomPairWeights(weights));
    const blockPacked = weights.blocks.map(
      (block) => packCached(block, "atom.block", () => packAtomBlockWeights(block)));
    const shape = {
      tokens, dense, subsets, queries, keys, channels, pairChannels, heads, dimension,
      perTokenChannels, trunkSingleChannels: weights.trunkSingleChannels,
      trunkPairChannels: weights.trunkPairChannels, blocks: weights.blocks.length,
    };
    const sources = createAtomEncoderShaders(shape, pairPacked.offsets, blockPacked[0].offsets);
    const base = `af3-atom:${tokens}:${dense}:${subsets}:${queries}:${keys}`
      + `:${channels}:${pairChannels}:${heads}:${dimension}:${perTokenChannels}`;
    const compiled = {};
    // 🔴 COMPILED CONCURRENTLY - see the note in pair-track-gpu.js.
    const compiling = [];
    for (const [name, source] of Object.entries(sources)) {
      // ...the factory also returns the row tile the dispatch needs, which is a
      // number rather than a shader.
      if (name === "attendFor" || typeof source !== "string") continue;
      compiling.push(this.pipelines.get(`${base}:${name}`, source)
        .then((pipeline) => { compiled[name] = pipeline; }));
    }
    // 🔴 ONE attend PIPELINE PER BLOCK. All three blocks' head biases live in
    // one buffer, and the block index selects a slice - baked in, because the
    // pipeline cache takes no override constants.
    compiled.attend = [];
    for (let index = 0; index < weights.blocks.length; index += 1) {
      const at = index;
      compiling.push(this.pipelines.get(`${base}:attend:${at}`, sources.attendFor(at))
        .then((pipeline) => { compiled.attend[at] = pipeline; }));
    }
    await Promise.all(compiling);

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (a) => { allocations.push(a); return a; };
    const up = (label, data) => keep(this.allocator.upload(label, data, storage));
    const alloc = (label, bytes, extra = 0) =>
      keep(this.allocator.allocate(label, bytes, storage | extra));
    // 🔴 THE STATIC HALF OF THIS ENCODER SURVIVES THE CALL, WHEN THE CALLER
    // ASKS. Everything built from the reference conformers, the gathers and the
    // trunk - the projected trunk tensors, the query and key conditioning and
    // their masks, the atom pair conditioning and its logits - has nothing to
    // do with the noisy positions or the noise level, and a sampler calls this
    // two hundred times down one schedule. `build-pair` alone measured 14.6 ms
    // of a 147 ms denoiser call, rebuilding the identical tensor every time.
    //
    // These are created OUTSIDE the pooled allocator, because a pooled buffer
    // is recycled at the end of the run that made it, and released only when
    // the cache is dropped.
    const staticCache = options.staticCache;
    // 🔴 BUILD IF ANY OF THEM HAD TO BE CREATED, not if the last one was. They
    // are created together, so a partially populated cache means a shape
    // changed underneath it - and skipping the build then would run the blocks
    // against one molecule's conditioning and another's gathers.
    let buildStatic = staticCache === undefined;
    const persistent = (label, bytes, extra = 0) => {
      if (staticCache === undefined) return alloc(label, bytes, extra);
      const size = Math.ceil(bytes / 4) * 4;
      const found = staticCache[label];
      if (found !== undefined && found.size === size) return { buffer: found };
      if (found !== undefined) { found.destroy(); noteDestroy(this.device, found.size, label); }
      buildStatic = true;
      noteAllocation(this.device, label, size);
      const buffer = this.device.createBuffer({
        label, size, usage: storage | extra | GPUBufferUsage.COPY_DST,
      });
      staticCache[label] = buffer;
      return { buffer };
    };
    // 🔴 AND THE CONTENTS ARE STATIC TOO, NOT ONLY THE BUFFER. `persistent`
    // keeps a tensor the blocks WRITE; this keeps one they READ. The per-atom
    // conditioning, the reference conformer, the ten gathers and the trunk's
    // two conditioned tensors are functions of the FOLD and not of the step,
    // and they were rebuilt on the host and written across the bus once per
    // sampler step - 2.6 MB a step at 59 tokens, growing as tokens^2 through
    // `atom.trunk-pair`, into buffers already holding the identical bytes.
    //
    // The build closure is not called on a hit, so the host-side gathering
    // above it - which walks every query and key row - does not run either.
    const uploaded = staticCache === undefined ? undefined
      : (STATIC_UPLOADS.get(staticCache) ?? new Set());
    if (staticCache !== undefined) STATIC_UPLOADS.set(staticCache, uploaded);
    const persistentUpload = (label, build, extra = 0) => {
      if (staticCache === undefined) return up(label, build());
      const found = staticCache[label];
      if (found !== undefined && uploaded.has(label)) return { buffer: found };
      const data = build();
      const size = Math.ceil(data.byteLength / 4) * 4;
      let buffer = found;
      if (buffer !== undefined && buffer.size !== size) {
        buffer.destroy();
        noteDestroy(this.device, buffer.size, label);
        buffer = undefined;
      }
      if (buffer === undefined) {
        // A shape that moved under the cache invalidates the computed statics
        // as well; see the note on buildStatic.
        buildStatic = true;
        noteAllocation(this.device, label, size);
        buffer = this.device.createBuffer({
          label, size, usage: storage | extra | GPUBufferUsage.COPY_DST,
        });
        staticCache[label] = buffer;
      }
      this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
      uploaded.add(label);
      return { buffer };
    };
    const ints = (source) => Int32Array.from(source, (v) => Number(v));
    const floats = (source) => Float32Array.from(source, (v) => Number(v));

    try {
      const conditioning = persistentUpload("atom.cond", () => input.conditioning);
      const atomMask = persistentUpload("atom.mask", () => floats(input.atomMask));
      const pairWeights = { buffer: residentWeightBuffer(this.device, weights,
        "atom.pair-weights", () => pairPacked.data) };
      const trunkSingleCond = persistentUpload("atom.trunk-single",
        () => input.trunkSingleCond);
      const trunkPairCond = persistentUpload("atom.trunk-pair", () => input.trunkPairCond);
      // 🔴 THE ONE INPUT THAT MOVES. Everything else this encoder reads is the
      // molecule or the trunk; the noisy coordinates are the step.
      const positions = up("atom.positions", input.tokenAtomsAct);
      const refPos = persistentUpload("atom.ref-pos", () => floats(input.refPos));
      const refSpaceUid = persistentUpload("atom.ref-space", () => ints(input.refSpaceUid));

      // Every gather, and the two reference-space columns, in one i32 buffer -
      // see the note in the shader preamble about the eight-buffer guarantee.
      const queriesSpace = new Int32Array(queryRows);
      const keysSpace = new Int32Array(keyRows);
      // 🔴 A MASKED SLOT'S REFERENCE SPACE IS ZERO, NOT A SENTINEL. AF3 gathers
      // with a zero-filling convert, so two PADDED atoms both read 0, compare
      // equal, and are treated as sharing a reference conformer - which makes
      // their offset term live. Using -1 and -2 here to mark them "unrelated"
      // is the tidier choice and a different model; it cost 3.1e-2 on the atom
      // pair representation.
      const buildQueriesSpace = () => {
        const flatSpace = ints(input.refSpaceUid);
        for (let index = 0; index < queryRows; index += 1) {
          queriesSpace[index] = input.tokenAtomsToQueries.mask[index]
            ? flatSpace[Number(input.tokenAtomsToQueries.indices[index])] : 0;
        }
        for (let index = 0; index < keyRows; index += 1) {
          keysSpace[index] = input.queriesToKeys.mask[index]
            ? queriesSpace[Number(input.queriesToKeys.indices[index])] : 0;
        }
      };
      const gatherBuffer = persistentUpload("atom.gathers", () => {
        buildQueriesSpace();
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
        return gathers;
      });

      // The reference positions in query and key layout, gathered on the host:
      // three floats each, and the gathers are integer indirection the GPU has
      // no reason to redo. Built once per fold - the key layout depends on the
      // molecule and the query one on nothing else either.
      let referenceLayouts;
      const layouts = () => {
        if (referenceLayouts !== undefined) return referenceLayouts;
        const queriesRef = new Float32Array(queryRows * 3);
        const keysRef = new Float32Array(keyRows * 3);
        const flatRef = floats(input.refPos);
        for (let index = 0; index < queryRows; index += 1) {
          if (!input.tokenAtomsToQueries.mask[index]) continue;
          const from = Number(input.tokenAtomsToQueries.indices[index]) * 3;
          for (let axis = 0; axis < 3; axis += 1) {
            queriesRef[index * 3 + axis] = flatRef[from + axis];
          }
        }
        for (let index = 0; index < keyRows; index += 1) {
          if (!input.queriesToKeys.mask[index]) continue;
          const from = Number(input.queriesToKeys.indices[index]) * 3;
          for (let axis = 0; axis < 3; axis += 1) {
            keysRef[index * 3 + axis] = queriesRef[from + axis];
          }
        }
        referenceLayouts = { queriesRef, keysRef };
        return referenceLayouts;
      };
      const queriesRefBuffer = persistentUpload("atom.q-ref", () => layouts().queriesRef);
      const keysRefBuffer = persistentUpload("atom.k-ref", () => layouts().keysRef);

      const trunkSingleProjected = persistent("atom.trunk-single-p", tokens * channels * 4);
      const trunkPairProjected = persistent("atom.trunk-pair-p",
        tokens * tokens * pairChannels * 4);
      const queriesCond = persistent("atom.q-cond", queryRows * channels * 4,
        GPUBufferUsage.COPY_SRC);
      const queriesMask = persistent("atom.q-mask", queryRows * 4, GPUBufferUsage.COPY_SRC);
      const keysCond = persistent("atom.k-cond", keyRows * channels * 4,
        GPUBufferUsage.COPY_SRC);
      const keysMask = persistent("atom.k-mask", keyRows * 4, GPUBufferUsage.COPY_SRC);
      const act = alloc("atom.act", queryRows * channels * 4, GPUBufferUsage.COPY_SRC);
      const pair = persistent("atom.pair", pairRows * pairChannels * 4,
        GPUBufferUsage.COPY_SRC);
      const logits = persistent("atom.logits", weights.blocks.length * subsets * heads
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

      // 🔴 READ BEFORE THE READBACKS ARE SIZED, NOT AFTER THE PASSES ARE
      // ENCODED: it decides which of them exist at all.
      const reuseStatic = options.reuseStatic;
      const readbacks = {
        tokenAct: keep(this.allocator.allocate("atom.rb-token", tokens * perTokenChannels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        skipConnection: keep(this.allocator.allocate("atom.rb-skip", queryRows * channels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        // 🔴 ALLOCATED ONLY WHEN THEY ARE COPIED INTO. The five below are the
        // static ones; with `reuseStatic` nothing writes them, and
        // `atom.rb-pair` alone is 12.8 MiB standing in a sampler's peak for
        // the length of a call that never touches it.
        pairCond: reuseStatic !== undefined ? undefined
          : keep(this.allocator.allocate("atom.rb-pair", pairRows * pairChannels * 4,
              GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        // The decoder reads all four of these, so the head can chain the two
        // without a second encoder run.
        queriesCond: reuseStatic !== undefined ? undefined
          : keep(this.allocator.allocate("atom.rb-qcond", queryRows * channels * 4,
              GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        keysCond: reuseStatic !== undefined ? undefined
          : keep(this.allocator.allocate("atom.rb-kcond", keyRows * channels * 4,
              GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        queriesMask: reuseStatic !== undefined ? undefined
          : keep(this.allocator.allocate("atom.rb-qmask", queryRows * 4,
              GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        keysMask: reuseStatic !== undefined ? undefined
          : keep(this.allocator.allocate("atom.rb-kmask", keyRows * 4,
              GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
      };

      const blockBuffers = weights.blocks.map((block, index) => ({
        buffer: residentWeightBuffer(this.device, block, "atom.block",
                                     () => blockPacked[index].data),
      }));

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

      const qr = lin(queryRows);
      const pr = lin(pairRows);
      // 🔴 EVERYTHING HERE EXCEPT build-act IS THE SAME ON EVERY CALL. See
      // `persistent` above: with a staticCache these run once per fold.
      if (buildStatic) {
        run("trunk-single", compiled.trunkSingle,
            [trunkSingleCond, pairWeights, trunkSingleProjected], Math.ceil(tokens / 64));
        const tp = lin(tokens * tokens);
        run("trunk-pair", compiled.trunkPair,
            [trunkPairCond, pairWeights, trunkPairProjected], tp[0], tp[1]);
        run("build-queries", compiled.buildQueries,
            [conditioning, atomMask, gatherBuffer, trunkSingleProjected, queriesCond,
             queriesMask], qr[0], qr[1]);
        const kr = lin(keyRows);
        run("build-keys", compiled.buildKeys,
            [queriesCond, queriesMask, gatherBuffer, keysCond, keysMask], kr[0], kr[1]);
        run("build-pair", compiled.buildPair,
            [queriesCond, keysCond, queriesRefBuffer, keysRefBuffer, gatherBuffer,
             trunkPairProjected, pairWeights, pair], pr[0], pr[1]);
        run("pair-logits", compiled.pairLogits, [pair, pairWeights, logits], pr[0], pr[1]);
      }
      // ...and this one reads the noisy positions, so it runs every time.
      run("build-act", compiled.buildAct,
          [queriesCond, queriesMask, positions, gatherBuffer, pairWeights, act], qr[0], qr[1]);

      for (let index = 0; index < weights.blocks.length; index += 1) {
        const w = blockBuffers[index];
        // ...one workgroup per TILE of query rows; see the note on `output`.
        const perOutput = spread(Math.ceil(queryRows / sources.outputRowTile));
        run(`project-${index}`, compiled.project,
            [act, queriesCond, w, q, gate], perOutput[0], perOutput[1]);
        run(`project-keys-${index}`, compiled.projectKeysAtoms,
            [act, queriesCond, w, kAtoms, vAtoms], perOutput[0], perOutput[1]);
        const expand = lin(keyRows * width);
        run(`expand-keys-${index}`, compiled.expandKeys,
            [kAtoms, vAtoms, gatherBuffer, k, v], expand[0], expand[1]);
        // The per-block slice of the logits.
        const slots = spread(queryRows * heads);
        run(`attend-${index}`, compiled.attend[index],
            [q, k, v, logits, queriesMask, keysMask, gathered], slots[0], slots[1]);
        run(`output-${index}`, compiled.output,
            [gathered, gate, queriesCond, w, act], perOutput[0], perOutput[1]);
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
        // 🔴 THE SAME FIVE TENSORS, AS DEVICE BUFFERS. They are already on the
        // GPU and the decoder's next act was to upload its own copy of them -
        // 17 MiB of a 59-residue fold held twice, and read back across the bus
        // once to make the second copy. Offered only when a staticCache keeps
        // them alive past this call; without one they belong to the pooled
        // allocator and are recycled the moment this returns.
        deviceStatics: staticCache === undefined ? undefined : {
          pairCond: pair.buffer, queriesCond: queriesCond.buffer,
          keysCond: keysCond.buffer, queriesMask: queriesMask.buffer,
          keysMask: keysMask.buffer,
        },
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
