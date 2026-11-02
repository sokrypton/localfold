/**
 * AF3's MSA attention: pair-weighted averaging over the row.
 *
 * 🔴 THERE IS NO QUERY AND NO KEY. The attention weights come from the PAIR
 * representation alone - one softmax per (head, i) over j, shared by every
 * sequence in the MSA - and the MSA only supplies the values. So the cost is
 * tokens^2 for the weights plus sequences*tokens^2 for the averaging, and there
 * is no q.k anywhere. Writing this as ordinary attention over the MSA rows is
 * the obvious mistake and produces a working, wrong module.
 *
 *     w[h][i][j] = softmax_j(pair_logits(LayerNorm(pair))[i][j][h]
 *                            + 1e9 * (keyMask[j] - 1))
 *     out[s][i]  = outputProjection(sum_j w[h][i][j] * v[s][j] * sigmoid(gate[s][i]))
 *
 * 🔴 THE KEY MASK IS THE MAXIMUM OVER MSA DEPTH, one row for all sequences: a
 * token that ANY sequence covers is attendable by EVERY sequence. Using each
 * sequence's own mask would be per-sequence softmaxes - a different operation
 * and a much more expensive one.
 */
import { packNamedWeights } from "../../../shared/weights/weight-pack.js";
import { GpuBufferAllocator } from "../../runtime/allocator.js";
import { pipelineCacheForDevice } from "../../runtime/pipeline-cache.js";

const GRID_WIDTH = 32_768;

const ORDER = [
  "actNormScale", "actNormOffset", "pairNormScale", "pairNormOffset",
  "pairLogits", "vProjection", "gatingQuery", "outputProjection",
];

export function packMsaAttentionWeights(weights) {
  return packNamedWeights(weights, { label: "msa attention", order: ORDER });
}

/**
 * The part of a pipeline key that the shaders above actually vary on.
 *
 * 🔴 EVERY KERNEL HERE SHARES `common`, SO EVERY KERNEL EMBEDS `HEADS`,
 * `DIMENSION` AND `WIDTH` - including `keyMask`, which is thirteen lines that
 * read only `TOKENS` and `SEQUENCES`. A caller that keys on the channel widths
 * alone therefore collides the moment two models share `msaChannels` and differ
 * in the head shape, and AlphaFold 3 and RoseTTAFold3 do exactly that: both are
 * msaChannels 64 with 8 heads, of dimension **8** and **32**. openbind0 sits
 * with AF3 and boltz2 with rf3, so any process that runs one family and then
 * the other at the same token count and depth hits it - which is a model switch
 * on the page, and is why no single-model CLI run ever showed it.
 *
 * Reported as `af3-msa:59:128:64:128:0.00001:fast:false:msa:keyMask - line 8 of
 * 35: "const DIMENSION: u32 = 8u;" against "const DIMENSION: u32 = 32u;"`.
 * Line 8 of 35 is `common`'s seventh constant: the cache indexes by SOURCE as
 * well as by key, so it refused rather than handing rosettafold3 AlphaFold 3's
 * kernel. A key is not a summary of the call, it is a promise about the text.
 */
export function msaAttentionKeyPart({ heads, dimension }) {
  return `${heads}x${dimension}`;
}

export function createMsaAttentionShaders(shape, offsets, epsilon, variance) {
  const { sequences, tokens, msaChannels, pairChannels, heads, dimension } = shape;
  const width = heads * dimension;
  const rows = sequences * tokens;

  const common = `
const SEQUENCES: u32 = ${sequences}u;
const TOKENS: u32 = ${tokens}u;
const ROWS: u32 = ${rows}u;
const C_M: u32 = ${msaChannels}u;
const C_Z: u32 = ${pairChannels}u;
const HEADS: u32 = ${heads}u;
const DIMENSION: u32 = ${dimension}u;
const WIDTH: u32 = ${width}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
const W_ACT_SCALE: u32 = ${offsets.actNormScale}u;
const W_ACT_OFFSET: u32 = ${offsets.actNormOffset}u;
const W_PAIR_SCALE: u32 = ${offsets.pairNormScale}u;
const W_PAIR_OFFSET: u32 = ${offsets.pairNormOffset}u;
const W_PAIR_LOGITS: u32 = ${offsets.pairLogits}u;
const W_V: u32 = ${offsets.vProjection}u;
const W_GATE: u32 = ${offsets.gatingQuery}u;
const W_OUT: u32 = ${offsets.outputProjection}u;

fn logistic(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
`;

  const varianceCode = (count, read) => variance === "fast"
    ? `let variance = squares / f32(${count}) - mean * mean;`
    : `var variance = 0.0;
  for (var c = 0u; c < ${count}; c += 1u) {
    let d = ${read} - mean;
    variance += d * d;
  }
  variance /= f32(${count});`;

  // The key mask: the maximum over MSA depth, one value per token.
  const keyMask = `${common}
@group(0) @binding(0) var<storage, read> msa_mask: array<f32>;
@group(0) @binding(1) var<storage, read_write> key_mask: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let token = id.x;
  if (token >= TOKENS) { return; }
  var largest = 0.0;
  for (var s = 0u; s < SEQUENCES; s += 1u) {
    largest = max(largest, msa_mask[s * TOKENS + token]);
  }
  key_mask[token] = largest;
}`;

  // 🔴 THE LOGITS FOR EVERY HEAD FROM ONE NORMALISATION OF THE ROW. The softmax
  // below is one workgroup a (head, i), and it used to compute its own logits -
  // so every pair row (i, j) was layer-normed and read HEADS times over, each
  // lane walking its own C_Z-float row: ~3 ms a pass at 384-512 channels. One
  // thread a pair now normalises once and writes all HEADS logits, masked,
  // into `attention`, which the softmax then reads in place. Each logit is the
  // same expression summed in the same order, so this is bit-identical.
  const stageLogitWeights = (2 + heads) * pairChannels * 4 <= 16384;
  const attentionLogits = `${common}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> key_mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> attention: array<f32>;
${stageLogitWeights ? `
var<workgroup> w_scale: array<f32, ${pairChannels}>;
var<workgroup> w_offset: array<f32, ${pairChannels}>;
var<workgroup> w_logits: array<f32, ${pairChannels * heads}>;
` : ""}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
${stageLogitWeights ? `  for (var c = lane; c < C_Z; c += 64u) {
    w_scale[c] = weights[W_PAIR_SCALE + c];
    w_offset[c] = weights[W_PAIR_OFFSET + c];
  }
  for (var idx = lane; idx < C_Z * HEADS; idx += 64u) {
    w_logits[idx] = weights[W_PAIR_LOGITS + idx];
  }
  workgroupBarrier();
` : ""}  let pair_index = id.x + id.y * GRID_WIDTH * 64u;
  if (pair_index >= TOKENS * TOKENS) { return; }
  let i = pair_index / TOKENS;
  let j = pair_index % TOKENS;
  let base = pair_index * C_Z;
  var total = 0.0;
  var squares = 0.0;
  for (var c = 0u; c < C_Z; c += 1u) {
    let value = pair[base + c];
    total += value;
    squares += value * value;
  }
  let mean = total / f32(C_Z);
  ${varianceCode("C_Z", "pair[base + c]")}
  let inverse_std = inverseSqrt(variance + EPSILON);
  ${Array.from({ length: heads }, (_, h) => `var logit_${h} = 0.0;`).join("\n  ")}
  for (var c = 0u; c < C_Z; c += 1u) {
    let value = (pair[base + c] - mean) * inverse_std * ${stageLogitWeights ? "w_scale[c]" : "weights[W_PAIR_SCALE + c]"}
      + ${stageLogitWeights ? "w_offset[c]" : "weights[W_PAIR_OFFSET + c]"};
    let w_base = ${stageLogitWeights ? "c * HEADS" : "W_PAIR_LOGITS + c * HEADS"};
    ${Array.from({ length: heads }, (_, h) => `logit_${h} += value * ${stageLogitWeights ? `w_logits[w_base + ${h}u]` : `weights[w_base + ${h}u]`};`).join("\n    ")}
  }
  let mask_bias = 1.0e9 * (key_mask[j] - 1.0);
  ${Array.from({ length: heads }, (_, h) => `attention[(${h}u * TOKENS + i) * TOKENS + j] = logit_${h} + mask_bias;`).join("\n  ")}
}`;

  // ...and the softmax over j, one workgroup per (head, i), on those logits.
  const attentionWeights = `${common}
@group(0) @binding(0) var<storage, read_write> attention: array<f32>;

var<workgroup> logits: array<f32, ${tokens}>;
var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let slot = group.x + group.y * GRID_WIDTH;
  if (slot >= HEADS * TOKENS) { return; }
  let local = local_id.x;

  var local_max = -3.0e38;
  for (var j = local; j < TOKENS; j += 64u) {
    let l = attention[slot * TOKENS + j];
    logits[j] = l;
    local_max = max(local_max, l);
  }
  reduce[local] = local_max;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
    workgroupBarrier();
  }
  let largest = reduce[0];
  workgroupBarrier();

  var local_sum = 0.0;
  for (var j = local; j < TOKENS; j += 64u) {
    let value = exp(logits[j] - largest);
    logits[j] = value;
    local_sum += value;
  }
  reduce[local] = local_sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let total = reduce[0];

  for (var j = local; j < TOKENS; j += 64u) {
    attention[slot * TOKENS + j] = logits[j] / total;
  }
}`;

  // v and the gate, per MSA row.
  //
  // 🔴 ONE WORKGROUP A ROW, NOT ONE THREAD, AND THE NORMALISED ROW STAGED. This
  // had a single invocation walk WIDTH outputs by C_M channels and re-derive
  // the normalised activation inside BOTH loops - so the row's 64 values were
  // recomputed 64 times each, and two thirds of the kernel was that. It is
  // 62 ms of a 255 ms MSA stack at 1024 rows, which is 60,416 rows of it.
  const project = `${common}
@group(0) @binding(0) var<storage, read> msa: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> gate: array<f32>;

var<workgroup> normalized: array<f32, ${msaChannels}>;
var<workgroup> reduce_a: array<f32, 64>;
var<workgroup> reduce_b: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= ROWS) { return; }
  let local = local_id.x;
  let base = row * C_M;

  var total = 0.0;
  var squares = 0.0;
  for (var c = local; c < C_M; c += 64u) {
    let value = msa[base + c];
    normalized[c] = value;
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
  let mean = reduce_a[0] / f32(C_M);
  ${variance === "fast"
    ? "let variance = reduce_b[0] / f32(C_M) - mean * mean;"
    : `workgroupBarrier();
  var centred = 0.0;
  for (var c = local; c < C_M; c += 64u) {
    let d = normalized[c] - mean;
    centred += d * d;
  }
  reduce_a[local] = centred;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_a[local] += reduce_a[local + stride]; }
    workgroupBarrier();
  }
  let variance = reduce_a[0] / f32(C_M);`}
  let inverse_std = inverseSqrt(variance + EPSILON);
  workgroupBarrier();
  for (var c = local; c < C_M; c += 64u) {
    normalized[c] = (normalized[c] - mean) * inverse_std * weights[W_ACT_SCALE + c]
      + weights[W_ACT_OFFSET + c];
  }
  workgroupBarrier();

  for (var out = local; out < WIDTH; out += 64u) {
    var value_total = 0.0;
    var gate_total = 0.0;
    for (var c = 0u; c < C_M; c += 1u) {
      // ...normalised once for the row, read here by every output.
      let value = normalized[c];
      value_total += value * weights[W_V + c * WIDTH + out];
      gate_total += value * weights[W_GATE + c * WIDTH + out];
    }
    values[row * WIDTH + out] = value_total;
    gate[row * WIDTH + out] = gate_total;
  }
}`;

  // 🔴 AND THE SAME PROJECTION OVER `projectRows` ROWS A WORKGROUP (shape.projectRows; 1 keeps the kernel above). One
  // row a workgroup read the whole [C_M][2 WIDTH] value and gate matrices for one row's 2 x C_M x WIDTH multiply-adds -
  // a weight read for every two of them, 307,200 times at 300 tokens x 1024 rows. A tile of rows reads each weight
  // once for all of them. Bit-identical to the row-a-workgroup kernel: each row's statistics take the same reduction
  // tree in the same order, and each output the same channel order.
  const R = Math.max(1, shape.projectRows ?? 1);
  const projectTiled = R === 1 ? project : `${common}
@group(0) @binding(0) var<storage, read> msa: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> gate: array<f32>;

const PROJECT_ROWS: u32 = ${R}u;
var<workgroup> normalized: array<f32, ${R * msaChannels}>;
var<workgroup> reduce_a: array<f32, ${R * 64}>;
var<workgroup> reduce_b: array<f32, ${R * 64}>;
var<workgroup> stats: array<vec2<f32>, ${R}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row0 = (group.x + group.y * GRID_WIDTH) * PROJECT_ROWS;
  if (row0 >= ROWS) { return; }
  let local = local_id.x;

  for (var r = 0u; r < PROJECT_ROWS; r += 1u) {
    let base = min(row0 + r, ROWS - 1u) * C_M;
    var total = 0.0;
    var squares = 0.0;
    for (var c = local; c < C_M; c += 64u) {
      let value = msa[base + c];
      normalized[r * C_M + c] = value;
      total += value;
      squares += value * value;
    }
    reduce_a[r * 64u + local] = total;
    reduce_b[r * 64u + local] = squares;
  }
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) {
      for (var r = 0u; r < PROJECT_ROWS; r += 1u) {
        reduce_a[r * 64u + local] += reduce_a[r * 64u + local + stride];
        reduce_b[r * 64u + local] += reduce_b[r * 64u + local + stride];
      }
    }
    workgroupBarrier();
  }
${variance === "fast" ? `  if (local < PROJECT_ROWS) {
    let mean = reduce_a[local * 64u] / f32(C_M);
    let variance = reduce_b[local * 64u] / f32(C_M) - mean * mean;
    stats[local] = vec2<f32>(mean, inverseSqrt(variance + EPSILON));
  }
  workgroupBarrier();` : `  if (local < PROJECT_ROWS) { stats[local] = vec2<f32>(reduce_a[local * 64u] / f32(C_M), 0.0); }
  workgroupBarrier();
  for (var r = 0u; r < PROJECT_ROWS; r += 1u) {
    let mean = stats[r].x;
    var centred = 0.0;
    for (var c = local; c < C_M; c += 64u) {
      let d = normalized[r * C_M + c] - mean;
      centred += d * d;
    }
    reduce_a[r * 64u + local] = centred;
  }
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) {
      for (var r = 0u; r < PROJECT_ROWS; r += 1u) { reduce_a[r * 64u + local] += reduce_a[r * 64u + local + stride]; }
    }
    workgroupBarrier();
  }
  if (local < PROJECT_ROWS) {
    let variance = reduce_a[local * 64u] / f32(C_M);
    stats[local] = vec2<f32>(stats[local].x, inverseSqrt(variance + EPSILON));
  }
  workgroupBarrier();`}
  for (var r = 0u; r < PROJECT_ROWS; r += 1u) {
    for (var c = local; c < C_M; c += 64u) {
      normalized[r * C_M + c] = (normalized[r * C_M + c] - stats[r].x) * stats[r].y * weights[W_ACT_SCALE + c]
        + weights[W_ACT_OFFSET + c];
    }
  }
  workgroupBarrier();

  for (var out = local; out < WIDTH; out += 64u) {
    var value_total: array<f32, ${R}>;
    var gate_total: array<f32, ${R}>;
    for (var r = 0u; r < PROJECT_ROWS; r += 1u) { value_total[r] = 0.0; gate_total[r] = 0.0; }
    for (var c = 0u; c < C_M; c += 1u) {
      let wv = weights[W_V + c * WIDTH + out];
      let wg = weights[W_GATE + c * WIDTH + out];
      for (var r = 0u; r < PROJECT_ROWS; r += 1u) {
        let value = normalized[r * C_M + c];
        value_total[r] += value * wv;
        gate_total[r] += value * wg;
      }
    }
    for (var r = 0u; r < PROJECT_ROWS; r += 1u) {
      let row = row0 + r;
      if (row < ROWS) {
        values[row * WIDTH + out] = value_total[r];
        gate[row * WIDTH + out] = gate_total[r];
      }
    }
  }
}`;

  // The averaging and the output projection, one workgroup per MSA row.
  const average = `${common}
@group(0) @binding(0) var<storage, read> attention: array<f32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> gathered: array<f32, ${width}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= ROWS) { return; }
  let local = local_id.x;
  let s = row / TOKENS;
  let i = row % TOKENS;

  for (var w = local; w < WIDTH; w += 64u) {
    let head = w / DIMENSION;
    let d = w % DIMENSION;
    var total = 0.0;
    for (var j = 0u; j < TOKENS; j += 1u) {
      total += attention[(head * TOKENS + i) * TOKENS + j]
        * values[(s * TOKENS + j) * WIDTH + head * DIMENSION + d];
    }
    gathered[w] = total * logistic(gate[row * WIDTH + w]);
  }
  workgroupBarrier();

  for (var c = local; c < C_M; c += 64u) {
    var total = 0.0;
    for (var w = 0u; w < WIDTH; w += 1u) {
      total += gathered[w] * weights[W_OUT + w * C_M + c];
    }
    output[row * C_M + c] = total;
  }
}`;

  // 🔴 AND THE AVERAGING OVER `averageRows` CONSECUTIVE TOKENS OF ONE SEQUENCE A WORKGROUP (shape.averageRows; 1 keeps
  // the kernel above): their sums read the same value rows and their output projections the same weights, so a tile
  // reads each once where one row a workgroup read them once a row. The same order of operations as above per row.
  const RA = Math.max(1, shape.averageRows ?? 1);
  const averageTiled = RA === 1 ? average : `${common}
@group(0) @binding(0) var<storage, read> attention: array<f32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const AVERAGE_ROWS: u32 = ${RA}u;
const TILES: u32 = ${Math.ceil(tokens / RA)}u;
var<workgroup> gathered: array<f32, ${RA * width}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let tile = group.x + group.y * GRID_WIDTH;
  if (tile >= SEQUENCES * TILES) { return; }
  let local = local_id.x;
  let s = tile / TILES;
  let i0 = (tile % TILES) * AVERAGE_ROWS;

  for (var w = local; w < WIDTH; w += 64u) {
    let head = w / DIMENSION;
    let d = w % DIMENSION;
    var total: array<f32, ${RA}>;
    for (var r = 0u; r < AVERAGE_ROWS; r += 1u) { total[r] = 0.0; }
    for (var j = 0u; j < TOKENS; j += 1u) {
      let value = values[(s * TOKENS + j) * WIDTH + head * DIMENSION + d];
      for (var r = 0u; r < AVERAGE_ROWS; r += 1u) {
        let i = min(i0 + r, TOKENS - 1u);
        total[r] += attention[(head * TOKENS + i) * TOKENS + j] * value;
      }
    }
    for (var r = 0u; r < AVERAGE_ROWS; r += 1u) {
      let row = s * TOKENS + min(i0 + r, TOKENS - 1u);
      gathered[r * WIDTH + w] = total[r] * logistic(gate[row * WIDTH + w]);
    }
  }
  workgroupBarrier();

  for (var c = local; c < C_M; c += 64u) {
    var total: array<f32, ${RA}>;
    for (var r = 0u; r < AVERAGE_ROWS; r += 1u) { total[r] = 0.0; }
    for (var w = 0u; w < WIDTH; w += 1u) {
      let weight = weights[W_OUT + w * C_M + c];
      for (var r = 0u; r < AVERAGE_ROWS; r += 1u) { total[r] += gathered[r * WIDTH + w] * weight; }
    }
    for (var r = 0u; r < AVERAGE_ROWS; r += 1u) {
      if (i0 + r < TOKENS) { output[(s * TOKENS + i0 + r) * C_M + c] = total[r]; }
    }
  }
}`;

  return { keyMask, attentionLogits, attentionWeights, project: projectTiled, average: averageTiled };
}

export class Af3MsaAttentionGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {Float32Array} msa sequences*tokens*msaChannels
   * @param {Float32Array} msaMask sequences*tokens
   * @param {Float32Array} pair tokens*tokens*pairChannels
   * @param {{sequences: number, tokens: number, msaChannels: number,
   *          pairChannels: number}} shape
   * @param {object} weights heads, dimension, and the eight tensors in ORDER
   */
  async run(msa, msaMask, pair, shape, weights, options = {}) {
    const { sequences, tokens, msaChannels, pairChannels } = shape;
    const { heads, dimension } = weights;
    if (!Number.isInteger(heads) || !Number.isInteger(dimension)) {
      throw new Error("weights.heads and weights.dimension must be integers");
    }
    const width = heads * dimension;
    const rows = sequences * tokens;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    if (msa.length !== rows * msaChannels) {
      throw new Error(`msa has ${msa.length} elements; expected ${rows * msaChannels}`);
    }

    const packed = packMsaAttentionWeights(weights);
    const full = { sequences, tokens, msaChannels, pairChannels, heads, dimension };
    const sources = createMsaAttentionShaders(full, packed.offsets, epsilon, variance);
    // This site already named the head shape; `msaAttentionKeyPart` is the
    // same statement in one place, so the stack cannot drift from it again.
    const key = `af3-msa-attn:${sequences}:${tokens}:${msaChannels}:${pairChannels}`
      + `:${msaAttentionKeyPart(full)}:${epsilon}:${variance}`;
    // Compiled together, not one after another: the browser builds them in
    // parallel, and a serial loop put every one of them on a cold fold's path.
    const compiled = Object.fromEntries(await Promise.all(Object.entries(sources).map(
      async ([name, source]) => [name, await this.pipelines.get(`${key}:${name}`, source)])));

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const msaBuffer = keep(this.allocator.upload("af3-msa-attn.msa", msa, storage));
      const maskBuffer = keep(this.allocator.upload("af3-msa-attn.mask", msaMask, storage));
      const pairBuffer = keep(this.allocator.upload("af3-msa-attn.pair", pair, storage));
      const weightBuffer = keep(this.allocator.upload("af3-msa-attn.weights", packed.data, storage));
      const keyMask = keep(this.allocator.allocate("af3-msa-attn.key-mask", tokens * 4, storage));
      const attention = keep(this.allocator.allocate(
        "af3-msa-attn.attention", heads * tokens * tokens * 4, storage));
      const values = keep(this.allocator.allocate("af3-msa-attn.values", rows * width * 4, storage));
      const gate = keep(this.allocator.allocate("af3-msa-attn.gate", rows * width * 4, storage));
      const output = keep(this.allocator.allocate(
        "af3-msa-attn.output", rows * msaChannels * 4, storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "af3-msa-attn.readback", rows * msaChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-msa-attention" });
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

      run("msa.key-mask", compiled.keyMask, [maskBuffer, keyMask], Math.ceil(tokens / 64));
      const logitGroups = spread(Math.ceil(tokens * tokens / 64));
      run("msa.attention-logits", compiled.attentionLogits,
          [pairBuffer, keyMask, weightBuffer, attention], logitGroups[0], logitGroups[1]);
      const weightGroups = spread(heads * tokens);
      run("msa.attention-weights", compiled.attentionWeights,
          [attention], weightGroups[0], weightGroups[1]);
      const perRow = spread(rows);
      // ...one workgroup a row now; see the note on the kernel.
      run("msa.project", compiled.project, [msaBuffer, weightBuffer, values, gate],
          perRow[0], perRow[1]);
      run("msa.average", compiled.average, [attention, values, gate, weightBuffer, output],
          perRow[0], perRow[1]);
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, rows * msaChannels * 4);

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return {
        output: result,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
