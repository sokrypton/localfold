import {
  ATTENTION_NORMALIZE_SHADER,
  createAttentionNormalizeShader,
  attentionOutputTileColumns,
  attentionProjectTileColumns,
  attentionProjectTileRows,
  attentionOutputTileRows,
  createAttentionPairBiasShader,
  selectAttentionProjectKernel,
  selectAttentionOutputKernel,
  createAttentionNormParameters,
  createAttentionParameters,
  attentionPackOrder,
  packAttentionWeights,
  buildAttentionFlashKernel,
  ATTENTION_VALUE_STORAGE,
  selectAttentionFlashKernel,

} from "./attention.js";
import {
  attentionOutputMatrixDispatch, attentionProjectMatrixConfig,
  attentionProjectMatrixDispatch, attentionProjectMatrixFits,
  createAttentionOutputMatrixShader, createAttentionProjectMatrixShader,
} from "./attention-project-matrix.js";
import {
  createOuterProductMeanParameters,
  OUTER_PRODUCT_MEAN_TILE_INTERMEDIATE_SHADER,
  OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_SHADER,
  OUTER_PRODUCT_MEAN_FINALIZE_SHADER,
  createOuterProductMeanContractShader,
  outerProductMeanTileCapacity,
  createOuterProductMeanProjectOutputShader,
  OUTER_PRODUCT_MEAN_NORMALIZE_SHADER,
  OUTER_PRODUCT_MEAN_PROJECT_SHADER,
  opmProjectOutputPairs,
  OUTER_PRODUCT_MEAN_SCALE_SHADER,
  createOuterProductMeanMatrixOutputShader,
  opmMatrixContract,
  createOuterProductMeanMatrixContractShader,
  createOuterProductMeanProjectShader,
  opmContractPrecision,
  opmProjectTileRows,
  opmProjectTileColumns,
  packOuterProductMeanWeights,
  OUTER_PRODUCT_MEAN_PACK_ORDER,
  useOuterFirstContraction,
  outerFirstLimitBytes,
  outerFirstPairBlocks,

} from "./outer-product-mean.js";
import {
  createTransitionNormalizeParameters,
  createTransitionShaders,
  chooseLinearKernel,
  linearKernelRows,
  linearKernelColumns,
  linearTileColumns,
  linearTileRows,
  packTransitionWeights,
  transitionChunkRows,
  TRANSITION_PACK_ORDER,
  TRANSITION_TILE_ROWS,

} from "./transition.js";
import { SOURCES } from "../reference/alphafold-fixture.js";
import { WebGpuExecution } from "../runtime/execution.js";
import { deviceTuning } from "../runtime/device-profile.js";
import { createTriangleShaders, LINEAR_GRID_WIDTH } from "../triangle/shaders.js";
import {
  createTriangleContractMatrixShader, createTriangleProjectMatrixShader,
  createTriangleProjectOutMatrixShaders, triangleContractMatrixDispatch,
  triangleProjectMatrixDispatch, triangleProjectMatrixFits,
  triangleProjectOutMatrixDispatch, TRIANGLE_PROJECT_MATRIX_GEOMETRY,
  triangleProjectMatrixConfig,
} from "../triangle/project-matrix.js";

import { packWeights as packTriangleWeights, trianglePackOrder }
  from "../triangle/weights.js";

const GLOBAL_ATTENTION_COMMON = `
struct Parameters {
  length: u32, sequences: u32, channels: u32, heads: u32, head_dim: u32,
  query_weight: u32, key_weight: u32, value_weight: u32, gating_weight: u32,
  gating_bias: u32, output_weight: u32, output_bias: u32,
};
// 🔴 THE SAME 32768 execution.linearGrid AND execution.rowGrid FOLD AT. A grid
// is folded into y past this many workgroups in x, so every shader those two
// dispatch has to put the y term back. See the note on the kv kernel.
const GRID_WIDTH: u32 = 32768u;
`;

const GLOBAL_ATTENTION_KV_SHADER = `${GLOBAL_ATTENTION_COMMON}
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> keys: array<f32>;
@group(0) @binding(4) var<storage, read_write> values: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  // 🔴 THE y TERM, WITHOUT WHICH THIS KERNEL SILENTLY STOPPED AT 2,097,152
  // ELEMENTS. Its dispatch is execution.linearGrid, which folds into y past
  // 32768 workgroups, and this read id.x alone: every workgroup with a nonzero
  // y recomputed the FIRST 32768 * 64 elements instead of its own, so the keys
  // and values past that index were never written at all and the global column
  // attention read whatever the recycled scratch held. head_dim is 8 here, so
  // the cap is length * sequences = 262,144 - reached by an extra alignment of
  // 1024 rows at 256 residues. It cost an 825-residue fold its whole structure:
  // the chain collapsed into a ball two angstroms across while pLDDT rose to
  // 69.31. docs/AF2.md has the measurements.
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.sequences * p.head_dim) { return; }
  let d = index % p.head_dim; let row = index / p.head_dim;
  var key = 0.0; var value = 0.0;
  for (var c = 0u; c < p.channels; c += 1u) {
    let x = normalized[row * p.channels + c];
    key += x * weights[p.key_weight + c * p.head_dim + d];
    value += x * weights[p.value_weight + c * p.head_dim + d];
  }
  keys[index] = key; values[index] = value;
}`;

/**
 * The global column attention's query: the masked mean over sequences,
 * projected onto the heads.
 *
 * 🔴 THE SAME FAULT THE OUTPUT PROJECTION HAD, one pass earlier. A thread owned
 * one (column, head, d) and recomputed the column's MEAN OVER EVERY SEQUENCE for
 * each of the `channels` it contracts - but that mean depends on (column, c)
 * alone, so all `heads * head_dim` threads of a column computed the same
 * `channels` means, each a sweep of the whole alignment. At 825 residues and
 * 1024 rows that is 3.46 G multiply-accumulates issued for 57 M of work,
 * **60x**, and it measured 34.66 ms.
 *
 * A workgroup owns a COLUMN now: it reduces the means once into workgroup
 * memory - a lane per channel, sweeping the sequences - and then a lane per
 * (head, d) contracts them. Generated for one shape, like the output kernel.
 */
export function createGlobalAttentionQueryShader(channels, heads, headDim) {
  const gates = heads * headDim;
  return `${GLOBAL_ATTENTION_COMMON}
const CHANNELS: u32 = ${channels}u;
const GATES: u32 = ${gates}u;
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> query: array<f32>;

// The column's masked mean, one per channel, computed once for every head.
var<workgroup> means: array<f32, ${channels}>;
var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let column = group.x;
  if (column >= p.length) { return; }
  let local = local_id.x;

  // ...the denominator, cooperatively and once, where every thread of the
  // column used to sweep the whole mask for it.
  var count = 0.0;
  for (var sequence = local; sequence < p.sequences; sequence += 64u) {
    count += mask[sequence * p.length + column];
  }
  reduce[local] = count;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let denominator = 1e-10 + reduce[0];

  // 🔴 A LANE PER CHANNEL, SWEEPING THE SEQUENCES. The alternative - a lane per
  // sequence, reducing per channel - is CHANNELS barrier trees where this is
  // none, and the read is the same either way: consecutive lanes take
  // consecutive channels, which is the contiguous axis of the normalised MSA.
  for (var c = local; c < CHANNELS; c += 64u) {
    var total = 0.0;
    for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
      total += normalized[(column * p.sequences + sequence) * CHANNELS + c]
        * mask[sequence * p.length + column];
    }
    means[c] = total / denominator;
  }
  workgroupBarrier();

  let scale = inverseSqrt(f32(p.head_dim));
  for (var slot = local; slot < GATES; slot += 64u) {
    let head = slot / ${headDim}u;
    let d = slot % ${headDim}u;
    var result = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      result += means[c] * weights[p.query_weight + (c * p.heads + head) * p.head_dim + d];
    }
    query[column * GATES + slot] = result * scale;
  }
}`;
}

const GLOBAL_ATTENTION_FLASH_SHADER = `${GLOBAL_ATTENTION_COMMON}
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x; let head = id.y;
  if (column >= p.length || head >= p.heads) { return; }
  var maximum = -1e30; var denominator = 0.0;
  var accumulated: array<f32, 32>;
  for (var d = 0u; d < p.head_dim; d += 1u) { accumulated[d] = 0.0; }
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
    var logit = 0.0;
    for (var d = 0u; d < p.head_dim; d += 1u) {
      logit += query[(column * p.heads + head) * p.head_dim + d]
        * keys[(column * p.sequences + sequence) * p.head_dim + d];
    }
    if (mask[sequence * p.length + column] == 0.0) { logit = -1e9; }
    let next_maximum = max(maximum, logit);
    let previous_scale = exp(maximum - next_maximum);
    let weight = exp(logit - next_maximum);
    denominator = denominator * previous_scale + weight;
    for (var d = 0u; d < p.head_dim; d += 1u) {
      accumulated[d] = accumulated[d] * previous_scale
        + weight * values[(column * p.sequences + sequence) * p.head_dim + d];
    }
    maximum = next_maximum;
  }
  for (var d = 0u; d < p.head_dim; d += 1u) {
    output[(column * p.heads + head) * p.head_dim + d] = accumulated[d] / denominator;
  }
}`;

/**
 * The global column attention's gated output projection.
 *
 * 🔴 IT RECOMPUTED THE GATE ONCE PER OUTPUT CHANNEL, and the gate is a whole
 * `channels`-long dot product. A thread owned one (row, c_out) and ran
 *
 *     for head, for d:  gate = bias + sum_c normalized[row][c] * W[c][head][d]
 *
 * inside its own loop - but `gate` does not depend on `c_out`, so all `channels`
 * threads of a row computed the same `heads * head_dim` gates, each of them
 * `channels` multiply-adds. At AF2's extra-MSA widths that is 64 threads x 64
 * gates x 64 terms where 64 gates x 64 terms is the whole job: **33x the
 * arithmetic**, and it measured **289.93 ms of a 681.61 ms extra-MSA block at
 * 825 residues and 1024 rows** - 42.5%, the largest kernel in the whole fold and
 * 450 GFLOP issued to compute 13.8.
 *
 * A WORKGROUP owns a row now. It stages the row once, computes one gated value
 * per lane, and then each lane takes one output channel out of the shared
 * result. Two 64-term dot products a lane instead of sixty-five.
 *
 * 🔴 GENERATED FOR ONE SHAPE, because the staged arrays have to be sized and the
 * loops want constant bounds - see CLAUDE.md on what a runtime trip count costs
 * in a short body. The shape is in the pipeline cache key.
 */
export function createGlobalAttentionOutputShader(channels, heads, headDim, residual = false) {
  for (const [name, value] of [["channels", channels], ["heads", heads], ["headDim", headDim]]) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new RangeError(`global attention ${name} must be a positive integer; got ${value}`);
    }
  }
  const gates = heads * headDim;
  // GRID_WIDTH comes from GLOBAL_ATTENTION_COMMON now; it used to be declared
  // here, which is how the kernel beside this one came to be missing it.
  return `${GLOBAL_ATTENTION_COMMON}
const CHANNELS: u32 = ${channels}u;
const GATES: u32 = ${gates}u;
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> attended: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// The row, read once for every gate and every output channel that wants it.
var<workgroup> staged: array<f32, ${channels}>;
// ...and the gated value per (head, head_dim), which is what the projection
// contracts over. Computed once instead of once per output channel.
var<workgroup> gated: array<f32, ${gates}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= p.sequences * p.length) { return; }
  let local = local_id.x;
  let column = row % p.length;
  let sequence = row / p.length;
  // ...the normalised tensor is column-major; see the query kernel.
  let normalized_row = column * p.sequences + sequence;

  for (var c = local; c < CHANNELS; c += 64u) {
    staged[c] = normalized[normalized_row * CHANNELS + c];
  }
  workgroupBarrier();

  for (var slot = local; slot < GATES; slot += 64u) {
    let head = slot / ${headDim}u;
    let d = slot % ${headDim}u;
    var gate = weights[p.gating_bias + slot];
    for (var c = 0u; c < CHANNELS; c += 1u) {
      gate += staged[c] * weights[p.gating_weight + (c * p.heads + head) * p.head_dim + d];
    }
    gated[slot] = attended[(column * p.heads + head) * p.head_dim + d] / (1.0 + exp(-gate));
  }
  workgroupBarrier();

  for (var c_out = local; c_out < CHANNELS; c_out += 64u) {
    var result = weights[p.output_bias + c_out];
    for (var slot = 0u; slot < GATES; slot += 1u) {
      result += gated[slot] * weights[p.output_weight + slot * CHANNELS + c_out];
    }
    output[row * CHANNELS + c_out] ${residual ? "+=" : "="} result;
  }
}`;
}

function uniform(execution, label, data) {
  return execution.upload(label, data, GPUBufferUsage.UNIFORM);
}

async function encodeTransition(
  execution,
  encoder,
  source,
  rows,
  channels,
  weightsValue,
  label,
  residualTarget,
) {
  const hiddenChannels = weightsValue.firstBias.length;
  const descriptor = {
    activations: new Float32Array(0), rows, channels, hiddenChannels, weights: weightsValue,
  };
  // 🔴 THE TILE IS PART OF THE CACHE KEY, because it is part of the shader. The
  // deep MSA transitions want the wide tile and a 59-residue structure module
  // wants the narrow one, and a key that named neither would hand the second
  // shape the first shape's pipeline - dispatched with the wrong column stride,
  // which leaves columns unprojected and reads as a speedup.
  // 🔴 THE TILE AND THE PRECISION ARE ONE CHOICE - see chooseLinearKernel.
  const { tile, precision, weightPrecision, matrix } = chooseLinearKernel({
    rows,
    columns: Math.max(channels, hiddenChannels),
    inner: Math.min(channels, hiddenChannels),
    device: execution.device,
  });
  // Packed once and left on the device; see uploadResident - and DECODED there
  // too when the fixture could hand over the codes, which on AF2's int8 bundle
  // is the 445 ms of widening and the 118 of concatenating that this one label
  // costs a first fold. See src/runtime/device-pack.js.
  const { weights, offsets: packedOffsets } = await execution.uploadResidentPacked(
    `${label}.weights`, descriptor.weights,
    () => packTransitionWeights(descriptor, weightPrecision), weightPrecision,
    { sources: descriptor.weights[SOURCES], order: TRANSITION_PACK_ORDER,
      precision: weightPrecision });
  // 🔴 THE GRID IS THE CHOSEN KERNEL'S, NOT THE TILE'S. The matrix path owns a
  // 128x128 region per workgroup and has no lane tile at all, so reading the
  // grid off `tile` would dispatch a 128-wide kernel four times per 32 columns
  // - which recomputes rather than corrupts, and so reads as a slowdown with
  // every checker still passing. See the note on tileRows below, which is the
  // same mistake made once already.
  const tileColumns = linearKernelColumns({ tile, matrix });
  // 🔴 AND THE ROW COUNT IS THE TILE'S TOO. This was TRANSITION_TILE_ROWS, a
  // module constant equal to LINEAR_TILE's 32 - safe only because both shipped
  // tiles had 32 rows, which the comment beside it said outright. A taller tile
  // under that constant is dispatched ceil(rows/32) times and recomputes its own
  // rows: correct, and redundant by the ratio. It measured as a 3.2x regression
  // on the transitions before this line existed. See docs/A100.md.
  const tileRows = linearKernelRows({ tile, matrix });
  // 🔴 THE HIDDEN ACTIVATION IS THE BIGGEST THING A TRANSITION HOLDS and the
  // shortest-lived: four times the channels, written by the first pass and
  // read by the second and by nothing else. At 512 MSA rows it was 32 MiB of a
  // 572 MiB fold, capped there only because TRANSITION_CHUNK_TARGET_BYTES cuts
  // it into chunks. The store owns whole quads of columns, so a packed word is
  // never shared between lanes; a column count that is not a multiple of four
  // has no quad to own, which is what the guard below says.
  const hiddenStorage = hiddenChannels % 4 === 0 ? "f16" : "f32";
  const shaders = createTransitionShaders(
    descriptor, packedOffsets, tile, precision, weightPrecision, hiddenStorage, matrix ?? null);
  // The geometry is part of the shader, so it is part of the key - a matrix
  // pipeline handed to a vector dispatch is the corruption the note above is
  // about.
  const key = `${precision}:${weightPrecision}:${tileColumns}:${hiddenStorage}`
    + `:${matrix ? `m${matrix.blockRows}x${matrix.blockColumns}x${matrix.blockInner}`
      + `x${matrix.subgroupRows}x${matrix.subgroupColumns}${matrix.vectorStaging ? "v" : ""}` : "t"}`;
  const [normalize, linearFirst, linear, linearResidual] = await Promise.all([
    execution.pipelines.get(`block:transition:normalize:${weightPrecision}`, shaders[0]),
    execution.pipelines.get(`block:transition:linear-first:${key}`, shaders[1]),
    execution.pipelines.get(`block:transition:linear:${key}`, shaders[2]),
    execution.pipelines.get(`block:transition:linear-residual:${key}`, shaders[3]),
  ]);
  // 🔴 A WARM STOPS HERE. `execution.warming` means the caller wants this
  // operation's PIPELINES built and nothing encoded - see Execution.warm, and
  // the 95 compiles AF2's first fold used to serialise behind its own progress.
  // Everything above is pure derivation plus the resident weight upload, which
  // is cached and needed either way; everything below touches the encoder.
  if (execution.warming) return undefined;
  const output = residualTarget ?? execution.allocate(`${label}.output`, rows * channels);
  // ...HOW MANY ROWS FIT IN ONE BINDING. Returns `rows` whenever everything
  // already binds, which is every short input, so the single-dispatch path
  // below is byte for byte the one that was there before chunking existed.
  const chunkRows = transitionChunkRows(
    rows, channels, hiddenChannels, execution.transitionBufferLimit,
    execution.device.limits.minStorageBufferOffsetAlignment,
    tileRows,
  );

  if (chunkRows === rows) {
    const normalizeParams = uniform(execution, `${label}.normalize-parameters`,
      createTransitionNormalizeParameters(descriptor, packedOffsets));
    const firstParams = uniform(execution, `${label}.first-parameters`, new Uint32Array([
      rows, channels, hiddenChannels, packedOffsets[2], packedOffsets[3], 1, 0, 0,
    ]));
    const secondParams = uniform(execution, `${label}.second-parameters`, new Uint32Array([
      rows, hiddenChannels, channels, packedOffsets[4], packedOffsets[5], 0, 0, 0,
    ]));
    const normalized = execution.allocate(`${label}.normalized`, rows * channels);
    const hidden = execution.allocate(
      `${label}.hidden`, rows * hiddenChannels, GPUBufferUsage.STORAGE, hiddenStorage);
    const transitionNormGrid = execution.rowGrid(rows);
    execution.dispatch(encoder, normalize, [source, weights, normalizeParams, normalized],
      transitionNormGrid[0], transitionNormGrid[1], 1, `${label}.normalize`);
    execution.dispatch(encoder, linearFirst, [normalized, weights, firstParams, hidden],
      Math.ceil(hiddenChannels / tileColumns), Math.ceil(rows / tileRows), 1,
      `${label}.first`);
    execution.dispatch(encoder, residualTarget === undefined ? linear : linearResidual,
      [hidden, weights, secondParams, output],
      Math.ceil(channels / tileColumns), Math.ceil(rows / tileRows), 1,
      `${label}.second`);
    return output;
  }

  // 🔴 ONE SCRATCH PAIR FOR THE WHOLE LOOP, viewed per chunk rather than
  // reallocated. The scratch is sized for a chunk, and what changes each time
  // round is which rows of the SOURCE and the OUTPUT are bound - so the big
  // tensors are never bound whole and never need to be bindable whole.
  const normalized = execution.allocate(`${label}.normalized-chunk`, chunkRows * channels);
  const hidden = execution.allocate(
    `${label}.hidden-chunk`, chunkRows * hiddenChannels, GPUBufferUsage.STORAGE, hiddenStorage);
  for (let rowOffset = 0; rowOffset < rows; rowOffset += chunkRows) {
    const count = Math.min(chunkRows, rows - rowOffset);
    // ...the row count in the uniforms is the CHUNK's, not the tensor's: the
    // shaders bound-check against it, and the last chunk is usually short.
    const chunkDescriptor = { ...descriptor, rows: count };
    const normalizeParams = uniform(execution, `${label}.normalize-parameters-${rowOffset}`,
      createTransitionNormalizeParameters(chunkDescriptor, packedOffsets));
    const firstParams = uniform(execution, `${label}.first-parameters-${rowOffset}`, new Uint32Array([
      count, channels, hiddenChannels, packedOffsets[2], packedOffsets[3], 1, 0, 0,
    ]));
    const secondParams = uniform(execution, `${label}.second-parameters-${rowOffset}`, new Uint32Array([
      count, hiddenChannels, channels, packedOffsets[4], packedOffsets[5], 0, 0, 0,
    ]));
    const sourceChunk = execution.view(source, rowOffset * channels, count * channels);
    const outputChunk = execution.view(output, rowOffset * channels, count * channels);
    const normalizedChunk = execution.view(normalized, 0, count * channels);
    const hiddenChunk = execution.view(hidden, 0, count * hiddenChannels);
    const chunkNormGrid = execution.rowGrid(count);
    execution.dispatch(encoder, normalize, [sourceChunk, weights, normalizeParams, normalizedChunk],
      chunkNormGrid[0], chunkNormGrid[1], 1, `${label}.normalize-${rowOffset}`);
    execution.dispatch(encoder, linearFirst, [normalizedChunk, weights, firstParams, hiddenChunk],
      Math.ceil(hiddenChannels / tileColumns), Math.ceil(count / tileRows), 1,
      `${label}.first-${rowOffset}`);
    execution.dispatch(encoder, residualTarget === undefined ? linear : linearResidual,
      [hiddenChunk, weights, secondParams, outputChunk],
      Math.ceil(channels / tileColumns), Math.ceil(count / tileRows), 1,
      `${label}.second-${rowOffset}`);
  }
  return output;
}

async function encodeAttention(
  execution,
  encoder,
  options,
) {
  const descriptor = {
    activations: new Float32Array(0), mask: new Float32Array(0), batch: options.batch,
    queryLength: options.queries, channels: options.channels, heads: options.heads,
    transpose: options.transpose, weights: options.weights,
    ...(options.pairBias === undefined ? {} : { pairBias: options.pairBias }),
  };
  // 🔴 PACKED ONCE AND LEFT ON THE DEVICE. Packing an evoformer stack's weights
  // is 136.6 ms of host time a pass at 400 residues, measured, and it ran on
  // every pass over weights that never change. `uploadResident` returns the
  // OFFSETS as well as the buffer, because every shader and uniform below needs
  // them and fetching them any other way would run the pack anyway. See the
  // note in src/runtime/execution.js, which measured this on an M2, found
  // nothing, and asked for the revisit on a discrete GPU.
  const { weights, offsets: packedOffsets } = await execution.uploadResidentPacked(
    `${options.label}.weights`, descriptor.weights,
    () => packAttentionWeights(descriptor), "",
    { order: attentionPackOrder(descriptor, (holder) => holder?.[SOURCES]),
      precision: "f32" });
  // 🔴 BUILT SEPARATELY SO A REFUSED SUBGROUP PIPELINE CAN FALL BACK. See
  // buildAttentionFlashKernel: a device can carry both subgroup features and
  // still reject `@subgroup_size(32)`, and inside a Promise.all that rejection
  // is just a failed fold.
  // 🔴 THE STORAGE IS THE FLASH KERNEL'S TO REFUSE, AND IT IS ASKED FIRST.
  // Only the register-resident kernel reads the projected tensors packed - it
  // already reads them four floats at a time, and four halves are two words at
  // the same index. Every other variant is f32, and a caller that allocated
  // packed anyway would hand it four bindings of the right byte length holding
  // twice the values they should. So this is awaited BEFORE anything is
  // allocated, and what comes back decides, not what was asked for.
  const built = await buildAttentionFlashKernel(
    execution, execution.device, options.channels / options.heads, undefined, undefined,
    { input: "f16", value: options.valueStorage ?? ATTENTION_VALUE_STORAGE, output: "f16" });
  const flashKernel = built.kernel;
  const flash = built.pipeline;
  // A packed pair is two adjacent channels of one row, so a row has to hold a
  // whole number of them. Every attention here projects to 256 or 128.
  const projectedStorage = flashKernel.packedStorageSupported === true
    && options.channels % 2 === 0 ? "f16" : "f32";
  // 🔴 THE VALUE IS ALLOCATED AT ITS OWN WIDTH, AND THE KERNEL DECIDES IT.
  // Reading this off what was ASKED FOR rather than off what came back is the
  // failure the line above already guards against: a fallback kernel that
  // refused packed storage would be handed a binding holding twice the values
  // it will read, which WebGPU cannot see. See ATTENTION_VALUE_STORAGE for why
  // the value is not simply the same as its three siblings.
  const valueStorage = projectedStorage === "f32"
    ? "f32" : (flashKernel.valueStorage ?? projectedStorage);
  // 🔴 THE PROJECTION'S TILE TRAVELS WITH ITS SHADER, because the dispatch
  // below divides by it and the two shapes differ by precision - see
  // selectAttentionProjectKernel.
  // 🔴 THE NORMALISED ACTIVATION IS PACKED, AND NOTHING ELSE HERE IS YET. It
  // is the one tensor in this operation with exactly two touchers - the layer
  // norm writes it and the projection reads it - and both were already
  // generated shaders, so it is where the packed storage can be measured
  // without a five-kernel change. At 512 MSA rows it is 29.5 MiB of a fold
  // whose peak is 603; see src/runtime/storage.js for what a packed word costs.
  const normalizedStorage = "f16";
  const pairBiasStorage = options.pairBias?.source === "separate" ? "f32" : normalizedStorage;
  const projectKernel = selectAttentionProjectKernel(
    execution.device, options.projectPrecision ?? "auto", normalizedStorage, projectedStorage,
    valueStorage);
  // 🔴 THE SAME SEVEN BINDINGS, A DIFFERENT KERNEL AND A DIFFERENT UNIFORM. The
  // matrix projection reads MatmulParameters where the vector one reads the
  // attention's own struct, so it gets its own small buffer; everything else -
  // source, weights, and the four targets - is what the vector kernel bound.
  //
  // It declines whenever the VALUE is stored differently from its three
  // siblings, because one GEMM writes one element type and a binding holding
  // twice the values a shader will read is not something WebGPU can see.
  const projectMatrix = valueStorage === projectedStorage
    ? attentionProjectMatrixConfig(execution.device) : false;
  const projectMatrixFits = projectMatrix !== false
    && attentionProjectMatrixFits(projectMatrix,
      execution.device.limits?.maxComputeWorkgroupStorageSize ?? 49152);
  const outputKernel = selectAttentionOutputKernel(
    execution.device, options.residualTarget !== undefined,
    options.outputPrecision ?? "auto", projectedStorage);
  // ...and the output projection, which is the same operation one kernel later
  // and the plainest GEMM in AF2 - no interleave, no group, no gate, and the
  // one thing a plain projection lacks: the column attention's TRANSPOSED
  // store. Same knob, because it is the same question about the same units.
  const outputMatrix = attentionProjectMatrixConfig(execution.device);
  const outputMatrixFits = outputMatrix !== false
    && attentionProjectMatrixFits(outputMatrix,
      execution.device.limits?.maxComputeWorkgroupStorageSize ?? 49152);
  const [normalize, packedNormalize, project, pairProject, outputProject] = await Promise.all([
    execution.pipelines.get("block:attention:normalize", ATTENTION_NORMALIZE_SHADER),
    execution.pipelines.get(`block:attention:normalize:${normalizedStorage}`,
      createAttentionNormalizeShader(normalizedStorage)),
    projectMatrixFits
      ? execution.pipelines.get(
        `block:attention:project-matrix:${options.channels}:${options.heads}`
        + `:${normalizedStorage}${projectedStorage}:${JSON.stringify(projectMatrix)}`,
        createAttentionProjectMatrixShader(
          { channels: options.channels, heads: options.heads },
          { source: normalizedStorage, weight: "f32", output: projectedStorage },
          projectMatrix))
      : execution.pipelines.get(projectKernel.cacheKey, projectKernel.shader),
    // The bias reads `normalized` itself unless the caller gave a separate
    // source, so its storage is the normalised one in exactly that case.
    // ...the HEAD COUNT is in the key because the shader unrolls it; see
    // createAttentionPairBiasShader.
    execution.pipelines.get(`block:attention:pair-bias:${pairBiasStorage}:${options.heads}`,
      createAttentionPairBiasShader(pairBiasStorage, options.heads)),
    outputMatrixFits
      ? execution.pipelines.get(
        `block:attention:output-matrix:${options.channels}:${projectedStorage}`
        + `:${options.transpose === true}:${options.residualTarget !== undefined}`
        + `:${JSON.stringify(outputMatrix)}`,
        createAttentionOutputMatrixShader(
          { channels: options.channels, transpose: options.transpose === true },
          { source: projectedStorage, weight: "f32" }, outputMatrix,
          options.residualTarget !== undefined))
      : execution.pipelines.get(outputKernel.cacheKey, outputKernel.shader),
  ]);
  // 🔴 A WARM STOPS HERE. `execution.warming` means the caller wants this
  // operation's PIPELINES built and nothing encoded - see Execution.warm, and
  // the 95 compiles AF2's first fold used to serialise behind its own progress.
  // Everything above is pure derivation plus the resident weight upload, which
  // is cached and needed either way; everything below touches the encoder.
  if (execution.warming) return undefined;
  const rows = options.batch * options.queries;
  const elements = rows * options.channels;
  // 🔴 PACKED ONCE AND LEFT ON THE DEVICE. Packing an evoformer stack's weights
  // is 136.6 ms of host time a pass at 400 residues, measured, and this ran it
  // on every pass over weights that never change - see uploadResident in
  // src/runtime/execution.js, and the note below it that measured this on an M2
  // and asked for the revisit.
  const params = uniform(execution, `${options.label}.parameters`,
    createAttentionParameters(descriptor, packedOffsets));
  const projectMatrixParams = !projectMatrixFits ? undefined
    : uniform(execution, `${options.label}.project-matrix-parameters`, new Uint32Array([
      rows, options.channels, 4 * options.channels,
      packedOffsets[2], packedOffsets[6], 0, 0, 0,
    ]));
  // ...and the output projection's, whose two spare words carry `queries` and
  // `batch` because the transposed store needs both at runtime.
  const outputMatrixParams = !outputMatrixFits ? undefined
    : uniform(execution, `${options.label}.output-matrix-parameters`, new Uint32Array([
      rows, options.channels, options.channels,
      packedOffsets[7], packedOffsets[8], 0, options.queries, options.batch,
    ]));
  const normParams = uniform(execution, `${options.label}.norm-parameters`, createAttentionNormParameters(
    rows, options.channels, packedOffsets[0], packedOffsets[1], options.transpose,
    options.batch, options.queries, 1e-5,
  ));
  const normalized = execution.allocate(
    `${options.label}.normalized`, elements, GPUBufferUsage.STORAGE, normalizedStorage);
  const projected = (name) => execution.allocate(
    `${options.label}.${name}`, elements, GPUBufferUsage.STORAGE, projectedStorage);
  const query = projected("query");
  const key = projected("key");
  const value = execution.allocate(
    `${options.label}.value`, elements, GPUBufferUsage.STORAGE, valueStorage);
  const gate = projected("gate");
  // 🔴 THE ATTENTION WRITES BACK INTO `normalized`. Read the dispatches below:
  // the projection is the last pass that reads it - the pair bias, when it
  // shares the tensor, runs before that - and the flash kernel is the next
  // pass to write. So the two can be one buffer, and at 512 MSA rows that is
  // 14.8 MiB per attention, twice a block, of a 396 MiB peak.
  //
  // 🔴 ONLY WHEN THEY AGREE ABOUT THE ELEMENT, THOUGH. `normalized` is always
  // packed and `projected` is packed only where the register-resident flash
  // kernel accepts it; where it does not, one is half the bytes of the other
  // and sharing would hand a shader a buffer of the wrong length, which is not
  // something WebGPU can catch. Falling back to a second tensor there costs
  // the memory and keeps the fold.
  const weighted = normalizedStorage === projectedStorage
    ? normalized : projected("weighted");
  const output = options.residualTarget ?? execution.allocate(`${options.label}.output`, elements);
  const attentionNormGrid = execution.rowGrid(rows);
  execution.dispatch(encoder, packedNormalize, [options.source, weights, normParams, normalized],
    attentionNormGrid[0], attentionNormGrid[1], 1,
    `${options.label}.normalize`);

  let normalizedPair = normalized;
  if (options.pairBias?.source === "separate") {
    if (options.pairSource === undefined) throw new Error("separate attention pair bias requires a GPU source");
    normalizedPair = execution.allocate(
      `${options.label}.pair-normalized`, options.queries * options.queries * options.pairBias.channels,
    );
    const pairNormParams = uniform(execution, `${options.label}.pair-norm-parameters`,
      createAttentionNormParameters(
        options.queries * options.queries, options.pairBias.channels, packedOffsets[9], packedOffsets[10],
        false, 1, options.queries * options.queries, 1e-5,
      ));
    const pairNormGrid = execution.rowGrid(options.queries * options.queries);
    execution.dispatch(encoder, normalize, [options.pairSource, weights, pairNormParams, normalizedPair],
      pairNormGrid[0], pairNormGrid[1], 1, `${options.label}.pair-normalize`);
  }
  const pairBiasElements = options.pairBias === undefined ? 1 : options.heads * options.queries * options.queries;
  const pairBias = execution.allocate(`${options.label}.pair-bias`, pairBiasElements);
  if (options.pairBias !== undefined) {
    const grid = execution.linearGrid(pairBiasElements);
    execution.dispatch(encoder, pairProject, [normalizedPair, weights, params, pairBias],
      grid[0], grid[1], 1, `${options.label}.pair-bias`);
  }
  if (projectMatrixFits) {
    const dispatch = attentionProjectMatrixDispatch(
      { rows, channels: options.channels }, projectMatrix);
    execution.dispatch(encoder, project,
      [normalized, weights, projectMatrixParams, query, key, value, gate],
      dispatch.x, dispatch.y, 1, `${options.label}.project`);
  } else {
    execution.dispatch(encoder, project, [normalized, weights, params, query, key, value, gate],
      Math.ceil(options.channels / attentionProjectTileColumns(projectKernel.tile)),
      Math.ceil(rows / attentionProjectTileRows(projectKernel.tile)), 1,
      `${options.label}.project`);
  }
  execution.dispatch(encoder, flash, [query, key, value, gate, options.mask, pairBias, params, weighted],
    Math.ceil(options.queries / flashKernel.queryTile),
    options.batch, options.heads, `${options.label}.flash`);
  if (outputMatrixFits) {
    const dispatch = attentionOutputMatrixDispatch(
      { rows, channels: options.channels }, outputMatrix);
    execution.dispatch(encoder, outputProject,
      [weighted, weights, outputMatrixParams, output],
      dispatch.x, dispatch.y, 1, `${options.label}.output`);
  } else {
    execution.dispatch(encoder, outputProject, [weighted, weights, params, output],
      Math.ceil(options.channels / attentionOutputTileColumns(outputKernel.tile)),
      Math.ceil(rows / attentionOutputTileRows(outputKernel.tile)), 1,
      `${options.label}.output`);
  }
  return output;
}

async function encodeGlobalAttention(
  execution,
  encoder,
  source,
  mask,
  shape,
  weightsValue,
  label,
  residualTarget,
) {
  const w = weightsValue;
  const tensors = [w.queryNormScale, w.queryNormOffset, w.queryWeight, w.keyWeight, w.valueWeight,
    w.gatingWeight, w.gatingBias, w.outputWeight, w.outputBias];
  const offsets = [];
  let size = 0;
  for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
  const packed = new Float32Array(size);
  tensors.forEach((tensor, index) => packed.set(tensor, offsets[index]));
  const headDim = w.gatingBias.length / w.heads;
  const params = new Uint32Array([
    shape.length, shape.sequences, shape.cM, w.heads, headDim,
    offsets[2], offsets[3], offsets[4], offsets[5], offsets[6], offsets[7], offsets[8],
  ]);
  const [normalize, kvPipeline, queryPipeline, flashPipeline, outputPipeline] = await Promise.all([
    execution.pipelines.get("block:global-attention:normalize", ATTENTION_NORMALIZE_SHADER),
    execution.pipelines.get("block:global-attention:kv", GLOBAL_ATTENTION_KV_SHADER),
    execution.pipelines.get(`block:global-attention:query:${shape.cM}:${w.heads}:${headDim}`,
      createGlobalAttentionQueryShader(shape.cM, w.heads, headDim)),
    execution.pipelines.get("block:global-attention:flash", GLOBAL_ATTENTION_FLASH_SHADER),
    // ...the SHAPE is in the key because the kernel is generated for it; see
    // createGlobalAttentionOutputShader.
    execution.pipelines.get(
      `block:global-attention:output${residualTarget === undefined ? "" : "-residual"}`
        + `:${shape.cM}:${w.heads}:${headDim}`,
      createGlobalAttentionOutputShader(shape.cM, w.heads, headDim, residualTarget !== undefined),
    ),
  ]);
  // 🔴 A WARM STOPS HERE. `execution.warming` means the caller wants this
  // operation's PIPELINES built and nothing encoded - see Execution.warm, and
  // the 95 compiles AF2's first fold used to serialise behind its own progress.
  // Everything above is pure derivation plus the resident weight upload, which
  // is cached and needed either way; everything below touches the encoder.
  if (execution.warming) return undefined;
  const weights = execution.upload(`${label}.weights`, packed);
  const parameters = uniform(execution, `${label}.parameters`, params);
  const normParameters = uniform(execution, `${label}.norm-parameters`, createAttentionNormParameters(
    shape.length * shape.sequences, shape.cM, offsets[0], offsets[1], true,
    shape.length, shape.sequences, 1e-5,
  ));
  const normalized = execution.allocate(`${label}.normalized`, shape.length * shape.sequences * shape.cM);
  const keys = execution.allocate(`${label}.keys`, shape.length * shape.sequences * headDim);
  const values = execution.allocate(`${label}.values`, shape.length * shape.sequences * headDim);
  const query = execution.allocate(`${label}.query`, shape.length * w.heads * headDim);
  const attended = execution.allocate(`${label}.attended`, shape.length * w.heads * headDim);
  const output = residualTarget ?? execution.allocate(`${label}.output`, shape.sequences * shape.length * shape.cM);
  const globalNormGrid = execution.rowGrid(shape.length * shape.sequences);
  execution.dispatch(encoder, normalize, [source, weights, normParameters, normalized],
    globalNormGrid[0], globalNormGrid[1], 1, `${label}.normalize`);
  let grid = execution.linearGrid(shape.length * shape.sequences * headDim);
  execution.dispatch(encoder, kvPipeline, [normalized, weights, parameters, keys, values],
    grid[0], grid[1], 1, `${label}.kv`);
  // ...ONE WORKGROUP A COLUMN now, not one thread per (column, head, d).
  execution.dispatch(encoder, queryPipeline, [normalized, mask, weights, parameters, query],
    shape.length, 1, 1, `${label}.query`);
  execution.dispatch(encoder, flashPipeline, [query, keys, values, mask, parameters, attended],
    shape.length, w.heads, 1, `${label}.flash`);
  // ...ONE WORKGROUP A ROW now, not one thread an element; see the kernel.
  const outputRows = shape.sequences * shape.length;
  const outputGrid = execution.rowGrid(outputRows);
  execution.dispatch(encoder, outputPipeline, [normalized, attended, weights, parameters, output],
    outputGrid[0], outputGrid[1], 1, `${label}.output`);
  return output;
}

async function encodeOuterProductMean(
  execution,
  encoder,
  msa,
  msaMask,
  input,
  weightsValue,
  residualTarget,
) {
  const descriptor = {
    activations: new Float32Array(0), mask: new Float32Array(0), sequences: input.sequences,
    length: input.length, cM: input.cM, cOuter: input.cOuter, cZ: input.cZ,
    weights: weightsValue,
  };
  // Packed once and left on the device, and DECODED there - the outer product
  // mean's eight tensors are 22 ms of a first fold across the two stacks and
  // not one of them is reshaped, so the pack is the order and nothing else.
  const { weights, offsets: packedOffsets } = await execution.uploadResidentPacked(
    "opm.weights", descriptor.weights, () => packOuterProductMeanWeights(descriptor), "",
    { sources: descriptor.weights[SOURCES], order: OUTER_PRODUCT_MEAN_PACK_ORDER,
      precision: "f32" });
  // ...against what THIS device will bind, not a 64 MiB constant; see the
  // note in outer-product-mean.js. Worth 6.8x on a block at 400 residues.
  const outerFirst = useOuterFirstContraction(
    descriptor, outerFirstLimitBytes(execution.device));
  const outputPairs = opmProjectOutputPairs(execution.device);
  const contractPrecision = opmContractPrecision(execution.device);
  // 🔴 THE CONTRACTION ON THE MATRIX UNITS, or null for the hand-tiled vector
  // kernel; see opmMatrixContract. It only applies on the outer-first path,
  // where the contraction is the GEMM it needs to be.
  const matrixContract = outerFirst ? opmMatrixContract(execution.device, input) : null;
  // ...the output projection follows the contraction unless it is turned off on
  // its own; they are one geometry and two independent kernels.
  const matrixOutput = deviceTuning(execution.device).opmMatrixOutput === false
    ? null : matrixContract;
  const [normalize, project, intermediatePipeline, accumulatePipeline, finalizePipeline,
    scalePipeline, contractPipeline, projectOutputPipeline] = await Promise.all([
    execution.pipelines.get("block:opm:normalize", OUTER_PRODUCT_MEAN_NORMALIZE_SHADER),
    matrixContract === null
      ? execution.pipelines.get("block:opm:project", OUTER_PRODUCT_MEAN_PROJECT_SHADER)
      : execution.pipelines.get("block:opm:project-transposed",
        createOuterProductMeanProjectShader(undefined, true)),
    execution.pipelines.get("block:opm:tile-intermediate", OUTER_PRODUCT_MEAN_TILE_INTERMEDIATE_SHADER),
    execution.pipelines.get("block:opm:tile-accumulate", OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_SHADER),
    execution.pipelines.get("block:opm:finalize", OUTER_PRODUCT_MEAN_FINALIZE_SHADER),
    execution.pipelines.get("block:opm:scale", OUTER_PRODUCT_MEAN_SCALE_SHADER),
    matrixContract === null
      ? execution.pipelines.get(`block:opm:contract:${input.cOuter}:${contractPrecision}`,
        createOuterProductMeanContractShader(input.cOuter, contractPrecision))
      : execution.pipelines.get(
        `block:opm:contract-matrix:${input.cOuter}:${matrixContract.blockRows}`
          + `x${matrixContract.blockColumns}x${matrixContract.blockInner}`,
        createOuterProductMeanMatrixContractShader(input.cOuter, matrixContract)),
    matrixOutput === null
      ? execution.pipelines.get(
        outerFirst && residualTarget !== undefined
          ? `block:opm:project-output-residual:${input.cOuter}:${outputPairs}`
          : `block:opm:project-output:${input.cOuter}:${outputPairs}`,
        createOuterProductMeanProjectOutputShader(
          input.cOuter, outerFirst && residualTarget !== undefined, outputPairs),
      )
      : execution.pipelines.get(
        `block:opm:project-output-matrix:${input.cOuter}:${matrixContract.blockRows}`
          + `:${residualTarget !== undefined}`,
        createOuterProductMeanMatrixOutputShader(
          matrixContract, residualTarget !== undefined),
      ),
  ]);
  // 🔴 A WARM STOPS HERE. `execution.warming` means the caller wants this
  // operation's PIPELINES built and nothing encoded - see Execution.warm.
  if (execution.warming) return undefined;
  const rows = input.sequences * input.length;
  const pairElements = input.length * input.length * input.cZ;
  const params = uniform(execution, "opm.parameters",
    createOuterProductMeanParameters(descriptor, packedOffsets));
  const normalized = execution.allocate("opm.normalized", rows * input.cM);
  const left = execution.allocate("opm.left", rows * input.cOuter);
  const right = execution.allocate("opm.right", rows * input.cOuter);
  const tileCapacity = outerProductMeanTileCapacity(
    input, execution.device.limits.maxStorageBufferBindingSize);
  // 🔴 A BLOCK OF PAIRS, NOT ALL OF THEM. The intermediate is sized by
  // outerFirstPairBlocks and no longer by the protein - which is what lets the
  // fast path run at 825 residues on a device that cannot bind its 2.79 GB.
  const pairBlocks = outerFirstPairBlocks(
    descriptor, outerFirstLimitBytes(execution.device),
    deviceTuning(execution.device).opmPairBlockBytes,
    matrixContract === null ? 1 : input.length);
  const intermediateElements = outerFirst
    ? pairBlocks[0][1] * input.cOuter * input.cOuter
    : tileCapacity * input.length * input.cOuter * input.cZ;
  const intermediate = execution.allocate("opm.intermediate", intermediateElements);
  const output = outerFirst && residualTarget !== undefined ? residualTarget
    : execution.allocate("opm.output", pairElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const opmNormGrid = execution.rowGrid(rows);
  execution.dispatch(encoder, normalize, [msa, weights, params, normalized],
    opmNormGrid[0], opmNormGrid[1], 1, "opm.normalize");
  // 🔴 A TILE GRID, NOT A THREAD GRID. The projection used to give one thread
  // each (row, channel); it now gives one WORKGROUP a tile of both, so the
  // dispatch counts tiles. Row tiles are folded through x and y for the same
  // reason every other row grid here is - a row per sequence and residue passes
  // 32,768 at any real MSA depth - and the channel tile is z.
  const projectRowTiles = Math.ceil(rows / opmProjectTileRows());
  let grid = execution.rowGrid(projectRowTiles);
  execution.dispatch(encoder, project, [normalized, msaMask, weights, params, left, right],
    grid[0], grid[1], Math.ceil(input.cOuter / opmProjectTileColumns()),
    "opm.project");
  const outputGrid = execution.linearGrid(pairElements);
  if (outerFirst) {
    // ...the denominator once for every pair, where project-output used to
    // compute it per workgroup; see OUTER_PRODUCT_MEAN_SCALE_SHADER.
    const scale = matrixOutput === null ? undefined
      : execution.allocate("opm.scale", input.length * input.length);
    if (scale !== undefined) {
      const scaleGrid = execution.linearGrid(input.length * input.length);
      execution.dispatch(encoder, scalePipeline, [msaMask, params, scale],
        scaleGrid[0], scaleGrid[1], 1, "opm.scale");
    }
    for (const [offset, count] of pairBlocks) {
      const blk = uniform(execution, `opm.pair-block-${offset}`,
        new Uint32Array([offset, count, 0, 0]));
      if (matrixContract === null) {
        // ...both are one workgroup per PAIR; see outer-product-mean.js.
        const pairGrid = execution.linearGrid(count * 64);
        execution.dispatch(encoder, contractPipeline, [left, right, params, intermediate, blk],
          pairGrid[0], pairGrid[1], 1, "opm.contract");
      } else {
        // 🔴 A GEMM OVER THE BLOCK'S ROWS OF `i`. The block is a whole number of
        // them, so `left` is bound as a VIEW starting at this block's first
        // residue and the kernel's row index is local - which is what lets the
        // store index by pair and leaves project-output alone.
        const blockRows = (count / input.length) * input.cOuter;
        const columns = input.length * input.cOuter;
        const matrixParams = uniform(execution, `opm.matmul-${offset}`, new Uint32Array([
          blockRows, input.sequences, columns, 0, 0, 0, input.length, 0,
        ]));
        const first = (offset / input.length) * input.cOuter * input.sequences;
        execution.dispatch(encoder, contractPipeline,
          [execution.view(left, first, blockRows * input.sequences), right, matrixParams,
            intermediate],
          Math.ceil(columns / matrixContract.blockColumns),
          Math.ceil(blockRows / matrixContract.blockRows), 1, "opm.contract");
      }
      if (matrixOutput === null) {
        // ...its OWN grid, because it carries several pairs a workgroup where
        // the contraction carries one; they shared `pairGrid` when both were one.
        const projectOutputGrid = execution.linearGrid(
          Math.ceil(count / outputPairs) * 64);
        execution.dispatch(encoder, projectOutputPipeline,
          [intermediate, msaMask, weights, params, output, blk],
          projectOutputGrid[0], projectOutputGrid[1], 1, "opm.project-output");
      } else {
        // 🔴 THE SAME GEMM SHAPE AS THE CONTRACTION, one step on: rows are this
        // block's pairs, the inner extent is c_outer^2, the columns are c_z.
        const matrixOutputParams = uniform(execution, `opm.matmul-out-${offset}`,
          new Uint32Array([count, input.cOuter * input.cOuter, input.cZ,
            packedOffsets[6], packedOffsets[7], 0, offset, 0]));
        execution.dispatch(encoder, projectOutputPipeline,
          [intermediate, weights, matrixOutputParams, output, scale],
          Math.ceil(input.cZ / matrixContract.blockColumns),
          Math.ceil(count / matrixContract.blockRows), 1, "opm.project-output");
      }
    }
  } else {
    execution.endComputePass(encoder);
    encoder.clearBuffer(output.allocation.buffer);
    for (let offset = 0; offset < input.sequences; offset += tileCapacity) {
      const count = Math.min(tileCapacity, input.sequences - offset);
      const tileParams = uniform(execution, `opm.tile-${offset}`, new Uint32Array([offset, count, 0, 0]));
      grid = execution.linearGrid(count * input.length * input.cOuter * input.cZ);
      execution.dispatch(encoder, intermediatePipeline, [left, weights, params, tileParams, intermediate],
        grid[0], grid[1], 1, `opm.intermediate-${offset}`);
      execution.dispatch(encoder, accumulatePipeline,
        [right, intermediate, params, tileParams, output],
        outputGrid[0], outputGrid[1], 1, `opm.accumulate-${offset}`);
    }
    execution.dispatch(encoder, finalizePipeline, [msaMask, weights, params, output],
      outputGrid[0], outputGrid[1], 1, "opm.finalize");
  }
  return output;
}

async function encodeTriangleMultiplication(
  execution,
  encoder,
  pair,
  pairMask,
  input,
  weightsValue,
  direction,
  residualTarget,
) {
  const shape = { length: input.length, cZ: input.cZ, cHidden: input.triangleHidden };
  // 🔴 THE SAME THREE MATRIX KERNELS AF3's PAIR TRACK RUNS, on AF2's own
  // triangle. This operation is 9.5 ms of a 62.73 ms block at 400 residues -
  // its projection, its contraction and its gated output projection are all
  // vector - and src/triangle/ is ONE module: the shaders AF3 moved onto the
  // units in this file's earlier sections take AF2's widths unchanged. What
  // this had to grow is the caller: an interleaved weight pack, four uniforms,
  // and a second buffer for the output projection's gate.
  const triangleMatrix = triangleProjectMatrixConfig(execution.device, input.cZ);
  const matrixFits = triangleMatrix !== false && triangleProjectMatrixFits(
    { ...TRIANGLE_PROJECT_MATRIX_GEOMETRY, ...triangleMatrix },
    execution.device.limits?.maxComputeWorkgroupStorageSize ?? 49152);
  // 🔴 THE LAYOUT MUST MATCH THE KERNEL THAT WAS COMPILED. `interleaved` swaps
  // the four projection matrices for one transposed, interleaved block, which
  // is what the matrix projection reads and the vector one cannot. Same
  // element count either way; a pack and an offset table that disagree is a
  // finite, plausible tensor. See src/triangle/weights.js.
  // Packed once and left on the device; the LAYOUT is the cache variant,
  // because an interleaved pack has different offsets as well as different
  // bytes and handing one kernel the other's buffer is a plausible tensor.
  const layout = matrixFits
    ? { abLayout: "interleaved", zgLayout: "transposed",
        cHidden: input.triangleHidden, cZ: input.cZ }
    : {};
  // ...and DECODED on the device when the fixture can hand over the codes: this
  // pack is 87 ms of a first fold across the two stacks' 96 triangle
  // multiplications, all of it `code * scale` the GPU is idle for. The
  // interleave and both transposes are the decoder's mapping now; see
  // trianglePackOrder, where three of them cancel.
  const { weights, offsets: packedOffsets } = await execution.uploadResidentPacked(
    `triangle.${direction}.weights`, weightsValue,
    () => packTriangleWeights(weightsValue, "f32", layout),
    matrixFits ? "interleaved" : "blocked",
    { sources: weightsValue[SOURCES], order: trianglePackOrder(layout),
      precision: "f32", named: true });
  // 🔴 THE PROJECTION TILE IS THE DEVICE'S, WHICH AF3 HAS ASKED FOR SINCE THE
  // KNOB EXISTED AND AF2 NEVER DID. It is an occupancy choice - the same two
  // kernels, the same arithmetic in the same order, relRMS 0 - and this path
  // was taking src/triangle/shaders.js's default on every device while the
  // pairformer beside it took Ampere's 32x32. `undefined` keeps that default,
  // so a device with no prior is unchanged.
  const projectTile = deviceTuning(execution.device).trianglePairProjectTile ?? undefined;
  const shaders = createTriangleShaders(
    shape, "f32", packedOffsets, 1e-5, direction, "two-pass", projectTile);
  // 🔴 THE RESIDUAL FORM IS GENERATED, NOT PATCHED. It used to be a string
  // replacement on the finished WGSL; when the kernel's writeback was rewritten
  // the pattern stopped matching, and a replacement that matches nothing throws
  // nothing - the block would have OVERWRITTEN the pair representation instead
  // of adding to it, on the shipped AF2 path only.
  const residualShaders = createTriangleShaders(
    shape, "f32", packedOffsets, 1e-5, direction, "two-pass", shaders.projectTile, true);
  const pipelineKey = `block:triangle:${direction}:${input.length}:${input.cZ}:${input.triangleHidden}`
    + `:${shaders.projectTile.rows}x${shaders.projectTile.columns}`;
  const matrixKey = `${pipelineKey}:matrix:${JSON.stringify(triangleMatrix)}`;
  const outMatrixSources = matrixFits ? createTriangleProjectOutMatrixShaders(
    { cZ: input.cZ, cHidden: input.triangleHidden },
    { normalized: "f32", hidden: "f32", gate: "f32", weight: "f32" },
    triangleMatrix) : null;
  const [normalizeInput, projectAB, contract, normalizeHidden, projectOutput, projectOutGate]
    = await Promise.all([
      execution.pipelines.get(`${pipelineKey}:normalize-input`, shaders.normalizeInput),
      matrixFits
        ? execution.pipelines.get(`${matrixKey}:project-ab`,
          createTriangleProjectMatrixShader(
            { cZ: input.cZ, cHidden: input.triangleHidden },
            { normalized: "f32", ab: "f32", weight: "f32" }, triangleMatrix))
        : execution.pipelines.get(`${pipelineKey}:project-ab`, shaders.projectAB),
      matrixFits
        ? execution.pipelines.get(`${matrixKey}:contract:${direction}`,
          createTriangleContractMatrixShader(
            { length: input.length, channels: input.triangleHidden }, direction,
            { ab: "f32" }, triangleMatrix))
        : execution.pipelines.get(`${pipelineKey}:contract`, shaders.contract),
      execution.pipelines.get(`${pipelineKey}:normalize-hidden`, shaders.normalizeHidden),
      matrixFits
        ? execution.pipelines.get(`${matrixKey}:project-out`, outMatrixSources.project)
        : execution.pipelines.get(
          `${pipelineKey}:project-output${residualTarget === undefined ? "" : "-residual"}`,
          residualTarget === undefined ? shaders.projectOutput : residualShaders.projectOutput),
      matrixFits
        ? execution.pipelines.get(`${matrixKey}:project-out-gate`, outMatrixSources.gate)
        : Promise.resolve(null),
    ]);
  // 🔴 A WARM STOPS HERE. `execution.warming` means the caller wants this
  // operation's PIPELINES built and nothing encoded - see Execution.warm, and
  // the 95 compiles AF2's first fold used to serialise behind its own progress.
  // Everything above is pure derivation plus the resident weight upload, which
  // is cached and needed either way; everything below touches the encoder.
  if (execution.warming) return undefined;
  const pairs = input.length * input.length;
  const normalized = execution.allocate(`triangle.${direction}.normalized`, pairs * input.cZ);
  const a = execution.allocate(`triangle.${direction}.a`, pairs * input.triangleHidden);
  const b = execution.allocate(`triangle.${direction}.b`, pairs * input.triangleHidden);
  const contracted = execution.allocate(`triangle.${direction}.contracted`, pairs * input.triangleHidden);
  const hiddenNormalized = execution.allocate(`triangle.${direction}.hidden-normalized`, pairs * input.triangleHidden);
  const output = residualTarget ?? execution.allocate(`triangle.${direction}.output`, pairs * input.cZ);
  const projectParams = !matrixFits ? undefined
    : uniform(execution, `triangle.${direction}.project-parameters`, new Uint32Array([
      pairs, input.cZ, 4 * input.triangleHidden,
      packedOffsets.linearABWeight, packedOffsets.linearABBias, 0, 0, 0,
    ]));
  const contractParams = !matrixFits ? undefined
    : uniform(execution, `triangle.${direction}.contract-parameters`,
      new Uint32Array([input.length, input.length, input.length, 0, 0, 0, 0, 0]));
  const outGateParams = !matrixFits ? undefined
    : uniform(execution, `triangle.${direction}.out-gate-parameters`, new Uint32Array([
      pairs, input.cZ, input.cZ,
      packedOffsets.linearGWeight, packedOffsets.linearGBias, 0, 0, 0,
    ]));
  const outProjectParams = !matrixFits ? undefined
    : uniform(execution, `triangle.${direction}.out-project-parameters`, new Uint32Array([
      pairs, input.triangleHidden, input.cZ,
      packedOffsets.linearZWeight, packedOffsets.linearZBias, 0, 0, 0,
    ]));
  // 🔴 FOLDED OVER x AND y, for the same reason the projection below is folded
  // over y and z: there are n^2 pair rows and `ceil(pairs / normalizeRows)`
  // passes 65535 at 825 residues. The kernel reads
  // `base_row = (group.x + group.y * LINEAR_GRID_WIDTH) * NORMALIZE_ROWS`, and
  // rowGrid's width is that same 32768.
  const normalizeGrid = execution.rowGrid(Math.ceil(pairs / shaders.normalizeRows));
  execution.dispatch(encoder, normalizeInput, [pair, weights, normalized],
    normalizeGrid[0], normalizeGrid[1], 1,
    `triangle.${direction}.normalize-input`);
  let grid = execution.linearGrid(pairs * input.triangleHidden);
  // 🔴 ROWS OVER y AND z, WHICH src/multimer/block.js HAS DONE ALL ALONG AND
  // THIS DID NOT. x is the channel tile, so n^2 pair rows have nowhere else to
  // go, and `ceil(pairs / projectTile.rows)` passes the 65535 a dimension
  // allows at **725 residues** with a row tile of 8 - so an 825-residue monomer
  // refused with "needs 85079 workgroups in y". The SHADER has always folded
  // (`row0 = (group.y + group.z * PROJECT_GRID_WIDTH) * TILE_ROWS`); only this
  // caller had not, and its multimer twin twenty files away had. The two paths
  // share these shaders and diverged in the dispatch, which is the hazard
  // AGENTS.md names in the other direction.
  const projectWidth = shaders.projectGridWidth ?? LINEAR_GRID_WIDTH;
  const projectTiles = Math.ceil(pairs / shaders.projectTile.rows);
  const projectRows = [Math.min(projectTiles, projectWidth),
                       Math.ceil(projectTiles / projectWidth)];
  if (matrixFits) {
    // The same three buffers the vector kernel used; only the weight LAYOUT
    // and the dispatch change. `a` and `b` come out CHANNEL-MAJOR, which is
    // what the matrix contraction reads and what the AF3 pair track already
    // does with the same pair of kernels.
    const project = triangleProjectMatrixDispatch(
      { rows: pairs, cHidden: input.triangleHidden }, triangleMatrix);
    execution.dispatch(encoder, projectAB,
      [normalized, weights, projectParams, a, pairMask, b],
      project.x, project.y, 1, `triangle.${direction}.project`);
  } else {
    execution.dispatch(encoder, projectAB, [normalized, pairMask, weights, a, b],
      Math.ceil(input.triangleHidden / shaders.projectTile.columns),
      projectRows[0], projectRows[1],
      `triangle.${direction}.project`);
  }
  if (matrixFits) {
    // 🔴 a IS THE LEFT OPERAND OUTGOING AND THE RIGHT ONE INCOMING, which is
    // the whole difference between the two directions - the shader transposes
    // whichever of the two it has to, and swapping them returns a finite
    // tensor of the same shape.
    const swap = direction === "incoming";
    const c = triangleContractMatrixDispatch(
      { length: input.length, channels: input.triangleHidden }, triangleMatrix);
    execution.dispatch(encoder, contract,
      [swap ? b : a, swap ? a : b, contractParams, contracted],
      c.x, c.y, c.z, `triangle.${direction}.contract`);
  } else {
    execution.dispatch(encoder, contract, [a, b, contracted],
      Math.ceil(input.length / shaders.contractTile.columns),
      Math.ceil(input.length / shaders.contractTile.rows),
      input.triangleHidden, `triangle.${direction}.contract`);
  }
  execution.dispatch(encoder, normalizeHidden, [contracted, weights, hiddenNormalized],
    normalizeGrid[0], normalizeGrid[1], 1, `triangle.${direction}.normalize-hidden`);
  if (matrixFits) {
    // 🔴 THE GATE IS ITS OWN PASS AND THE PROJECTION READS IT, because
    // `logistic(z . Wg) * (x . Wz)` is two contractions over two DIFFERENT
    // sources and no column trick makes it one GEMM. `a` is dead by here - the
    // contraction was its last reader - so the gate's target costs nothing,
    // exactly as it does in AF3's pair track.
    //
    // 🔴 AND THE PROJECTION ALWAYS ADDS INTO ITS TARGET, so a caller with no
    // residual gets a cleared one. That is 82 MB of clear at 400 residues
    // against a second shader, and every AF2 caller passes a residual.
    if (residualTarget === undefined) encoder.clearBuffer(output);
    const out = triangleProjectOutMatrixDispatch({ rows: pairs, cZ: input.cZ }, triangleMatrix);
    execution.dispatch(encoder, projectOutGate, [normalized, weights, outGateParams, a],
      out.x, out.y, 1, `triangle.${direction}.output.gate`);
    execution.dispatch(encoder, projectOutput,
      [hiddenNormalized, weights, outProjectParams, output, a],
      out.x, out.y, 1, `triangle.${direction}.output`);
  } else {
    execution.dispatch(encoder, projectOutput, [normalized, hiddenNormalized, weights, output],
      Math.ceil(input.cZ / shaders.projectTile.columns),
      projectRows[0], projectRows[1],
      `triangle.${direction}.output`);
  }
  return output;
}

export async function encodeEvoformerBlock(
  execution,
  encoder,
  input,
  msa,
  pair,
  msaMask,
  pairMask,
) {
  const row = input.weights.msaRowAttention;
  await staged(execution, () => encodeAttention(execution, encoder, {
    source: msa, mask: msaMask, pairSource: pair, batch: input.sequences, queries: input.length,
    channels: input.cM, heads: row.heads, transpose: false, weights: row.attention,
    // 🔴 DERIVED, NOT COPIED. Writing the three tensors into a literal reads
    // them, and reading a lazy leaf decodes it - which is the 445 ms this whole
    // path exists to skip. `Object.create` keeps the getters AND the `SOURCES`
    // symbol reachable, so the device packer can still bind codes.
    pairBias: Object.assign(Object.create(row.pairBias), {
      source: "separate", activations: new Float32Array(0), channels: input.cZ,
    }),
    label: "msa-row-attention", residualTarget: msa,
  }));

  const column = input.weights.msaColumnAttention;
  await staged(execution, () => encodeAttention(execution, encoder, {
    source: msa, mask: msaMask, batch: input.length, queries: input.sequences,
    channels: input.cM, heads: column.heads, transpose: true, weights: column.attention,
    label: "msa-column-attention", residualTarget: msa,
  }));

  await staged(execution, () => encodeTransition(
    execution, encoder, msa, input.sequences * input.length, input.cM,
    input.weights.msaTransition, "msa-transition", msa,
  ));

  await staged(execution, async() => {
    const update = await encodeOuterProductMean(
      execution, encoder, msa, msaMask, input, input.weights.outerProductMean, pair,
    );
    // ...and a warm returns nothing, because it encoded nothing.
    if (!execution.warming && update !== pair) await execution.addInPlace(encoder, pair, update, "outer-product-mean.residual");
  });

  await staged(execution, () => encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, input, input.weights.triangleMultiplicationOutgoing, "outgoing", pair,
  ));
  await staged(execution, () => encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, input, input.weights.triangleMultiplicationIncoming, "incoming", pair,
  ));

  const starting = input.weights.triangleAttentionStarting;
  await staged(execution, () => encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: input.length, queries: input.length,
    channels: input.cZ, heads: starting.heads, transpose: false, weights: starting.attention,
    pairBias: Object.assign(Object.create(starting.pairBias), { source: "normalized-input" }),
    label: "triangle-attention-starting", residualTarget: pair,
  }));

  const ending = input.weights.triangleAttentionEnding;
  await staged(execution, () => encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: input.length, queries: input.length,
    channels: input.cZ, heads: ending.heads, transpose: true, weights: ending.attention,
    pairBias: Object.assign(Object.create(ending.pairBias), { source: "normalized-input" }),
    label: "triangle-attention-ending", residualTarget: pair,
  }));

  await staged(execution, () => encodeTransition(
    execution, encoder, pair, input.length * input.length, input.cZ,
    input.weights.pairTransition, "pair-transition", pair,
  ));
}

/**
 * A sub-layer's scratch, released the moment its residual write is encoded.
 *
 * 🔴 THE PAIR TRACK HELD TEN 332 MiB SCRATCH TENSORS AT 825 RESIDUES AND NEEDED
 * FIVE. Both triangle multiplications write their result into `pair` and run one
 * after the other, so the outgoing direction's `normalized`, `a`, `b`,
 * `contracted` and `hidden-normalized` are dead before the incoming direction
 * allocates its own five - and nothing released them, so the pool grew to hold
 * both. The same is true of the two triangle attentions and of every residual
 * sub-layer in the block: each one's scratch dies at its own write.
 *
 * 🔴 AND RELEASING INTO THE POOL MID-ENCODER IS SAFE FOR THE REASON THE BLOCK
 * LOOP ALREADY RELIES ON. A pooled buffer is reused, not destroyed, and the
 * dispatches that read it were encoded BEFORE the ones that overwrite it - in
 * the same compute pass, where WebGPU orders dispatches and makes each one's
 * writes visible to the next. src/evoformer/stack.js does exactly this between
 * blocks and says so; this does it between sub-layers.
 *
 * src/af3/pair-track-gpu.js reached the same conclusion from the other side and
 * counted its scratch down to five, one of which is shared - see
 * PAIR_SCRATCH_COUNT.
 */
export async function staged(execution, body) {
  // 🔴 A WARM DOES NOT STAGE AND DOES NOT WAIT. There is no encoder to order and
  // no scratch to release - the body returns as soon as its pipelines are
  // requested - so handing the promise to the execution and returning is what
  // puts the block's ten operations in flight at once. See Execution.warm.
  if (execution.warming) { execution.notePending(body()); return undefined; }
  const checkpoint = execution.checkpoint();
  try { return await body(); } finally { execution.releaseScratchSince(checkpoint); }
}

export async function encodeEvoformerPairBlock(
  execution,
  encoder,
  shape,
  weights,
  msa,
  pair,
  msaMask,
  pairMask,
) {
  await staged(execution, async() => {
    const update = await encodeOuterProductMean(
      execution, encoder, msa, msaMask, shape, weights.outerProductMean, pair,
    );
    // ...and a warm returns nothing, because it encoded nothing.
    if (!execution.warming && update !== pair) await execution.addInPlace(encoder, pair, update, "extra.outer-product-mean.residual");
  });
  await staged(execution, () => encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationOutgoing, "outgoing", pair,
  ));
  await staged(execution, () => encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationIncoming, "incoming", pair,
  ));
  await staged(execution, () => encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionStarting.heads, transpose: false,
    weights: weights.triangleAttentionStarting.attention,
    pairBias: Object.assign(Object.create(weights.triangleAttentionStarting.pairBias),
      { source: "normalized-input" }),
    label: "extra.triangle-attention-starting", residualTarget: pair,
  }));
  await staged(execution, () => encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionEnding.heads, transpose: true,
    weights: weights.triangleAttentionEnding.attention,
    pairBias: Object.assign(Object.create(weights.triangleAttentionEnding.pairBias),
      { source: "normalized-input" }),
    label: "extra.triangle-attention-ending", residualTarget: pair,
  }));
  await staged(execution, () => encodeTransition(
    execution, encoder, pair, shape.length * shape.length, shape.cZ,
    weights.pairTransition, "extra.pair-transition", pair,
  ));
}

export async function encodeExtraMsaBlock(
  execution,
  encoder,
  shape,
  weights,
  msa,
  pair,
  msaMask,
  pairMask,
) {
  const row = weights.msaRowAttention;
  await staged(execution, () => encodeAttention(execution, encoder, {
    source: msa, mask: msaMask, pairSource: pair, batch: shape.sequences, queries: shape.length,
    channels: shape.cM, heads: row.heads, transpose: false, weights: row.attention,
    pairBias: Object.assign(Object.create(row.pairBias), {
      source: "separate", activations: new Float32Array(0), channels: shape.cZ,
    }),
    label: "extra.msa-row-attention", residualTarget: msa,
  }));
  await staged(execution, () => encodeGlobalAttention(
    execution, encoder, msa, msaMask, shape, weights.msaColumnGlobalAttention,
    "extra.msa-column-global-attention", msa,
  ));
  await staged(execution, () => encodeTransition(
    execution, encoder, msa, shape.sequences * shape.length, shape.cM, weights.msaTransition,
    "extra.msa-transition", msa,
  ));
  await encodeEvoformerPairBlock(execution, encoder, shape, weights, msa, pair, msaMask, pairMask);
}

export async function encodeTemplatePairBlock(
  execution,
  encoder,
  shape,
  weights,
  pair,
  pairMask,
) {
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionStarting.heads, transpose: false,
    weights: weights.triangleAttentionStarting.attention,
    pairBias: Object.assign(Object.create(weights.triangleAttentionStarting.pairBias),
      { source: "normalized-input" }),
    label: "template.triangle-attention-starting", residualTarget: pair,
  });
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionEnding.heads, transpose: true,
    weights: weights.triangleAttentionEnding.attention,
    pairBias: Object.assign(Object.create(weights.triangleAttentionEnding.pairBias),
      { source: "normalized-input" }),
    label: "template.triangle-attention-ending", residualTarget: pair,
  });
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationOutgoing, "outgoing", pair,
  );
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationIncoming, "incoming", pair,
  );
  await encodeTransition(
    execution, encoder, pair, shape.length * shape.length, shape.cZ,
    weights.pairTransition, "template.pair-transition", pair,
  );
}

export class EvoformerBlockGpu {
  device;

  constructor(device) { this.device = device; }

  async run(input) {
    const execution = new WebGpuExecution(this.device);
    try {
      const msaElements = input.sequences * input.length * input.cM;
      const pairElements = input.length * input.length * input.cZ;
      if (input.msa.length !== msaElements || input.pair.length !== pairElements) {
        throw new RangeError("Evoformer block activation shape mismatch");
      }
      const msa = execution.upload("block.msa", input.msa, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const pair = execution.upload("block.pair", input.pair, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const msaMask = execution.upload("block.msa-mask", input.msaMask);
      const pairMask = execution.upload("block.pair-mask", input.pairMask);
      const encoder = this.device.createCommandEncoder({ label: "evoformer-block" });
      this.device.pushErrorScope("validation");

      await encodeEvoformerBlock(execution, encoder, input, msa, pair, msaMask, pairMask);

      const msaReadback = execution.createReadback("block.msa-readback", msa, encoder);
      const pairReadback = execution.createReadback("block.pair-readback", pair, encoder);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      const [msaOutput, pairOutput] = await Promise.all([
        execution.mapFloat32(msaReadback), execution.mapFloat32(pairReadback),
      ]);
      return {
        msa: msaOutput,
        pair: pairOutput,
        elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot(),
      };
    } finally {
      execution.release();
    }
  }
}
