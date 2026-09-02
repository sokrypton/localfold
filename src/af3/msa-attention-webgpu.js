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
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const GRID_WIDTH = 32_768;

const ORDER = [
  "actNormScale", "actNormOffset", "pairNormScale", "pairNormOffset",
  "pairLogits", "vProjection", "gatingQuery", "outputProjection",
];

export function packMsaAttentionWeights(weights) {
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    if (weights[name] === undefined) throw new Error(`msa attention missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of ORDER) data.set(weights[name], offsets[name]);
  return { data, offsets };
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

  // The attention weights, from the pair alone. One workgroup per (head, i).
  const attentionWeights = `${common}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> key_mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> attention: array<f32>;

var<workgroup> logits: array<f32, ${tokens}>;
var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let slot = group.x + group.y * GRID_WIDTH;
  if (slot >= HEADS * TOKENS) { return; }
  let head = slot / TOKENS;
  let i = slot % TOKENS;
  let local = local_id.x;

  for (var j = local; j < TOKENS; j += 64u) {
    let base = (i * TOKENS + j) * C_Z;
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
    var logit = 0.0;
    for (var c = 0u; c < C_Z; c += 1u) {
      let value = (pair[base + c] - mean) * inverse_std * weights[W_PAIR_SCALE + c]
        + weights[W_PAIR_OFFSET + c];
      logit += value * weights[W_PAIR_LOGITS + c * HEADS + head];
    }
    logits[j] = logit + 1.0e9 * (key_mask[j] - 1.0);
  }
  workgroupBarrier();

  var local_max = -3.0e38;
  for (var j = local; j < TOKENS; j += 64u) { local_max = max(local_max, logits[j]); }
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
  workgroupBarrier();

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
    let d = msa[base + c] - mean;
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
    normalized[c] = (msa[base + c] - mean) * inverse_std * weights[W_ACT_SCALE + c]
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

  return { keyMask, attentionWeights, project, average };
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
    const key = `af3-msa-attn:${sequences}:${tokens}:${msaChannels}:${pairChannels}`
      + `:${heads}:${dimension}:${epsilon}:${variance}`;
    const compiled = {};
    for (const [name, source] of Object.entries(sources)) {
      compiled[name] = await this.pipelines.get(`${key}:${name}`, source);
    }

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
      const weightGroups = spread(heads * tokens);
      run("msa.attention-weights", compiled.attentionWeights,
          [pairBuffer, keyMask, weightBuffer, attention], weightGroups[0], weightGroups[1]);
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
