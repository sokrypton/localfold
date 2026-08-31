/**
 * AF3's triangle ("grid") self-attention on the GPU.
 *
 * Attention runs WITHIN A ROW of the pair grid: for each row and each head,
 * every position i attends over every position j of that row. The column
 * direction is the same kernel over the transposed activation, which is how AF3
 * gets both directions from one module.
 *
 * 🔴 IT IS THE NORMALISED ACTIVATION THAT IS TRANSPOSED, NOT THE PAIR. AF3
 * normalises once at the top and q, k, v and the gate all read that. Reading
 * the raw pair instead is not a crash and not a small error: it restores the
 * ~450x that the LayerNorm had removed, and the op returns a finite tensor
 * about eighty times too large.
 *
 * 🔴 THE BIAS COMES FROM THE UNTRANSPOSED ACTIVATION IN EVERY DIALECT. In the
 * column direction q/k/v read index (j,i) while the bias still reads (i,j).
 * They are computed in separate passes here for exactly that reason - fusing
 * the bias into the projection pass, which is otherwise the obvious saving,
 * silently gives it the transposed source.
 *
 * 🔴 AND THE OPENFOLD3 LINEAGE TRANSPOSES THE BIAS TOO, WHERE STOCK AF3 DOES
 * NOT. Same weights, same shapes, different answer. `swapTransposedBias` picks,
 * and it has no default.
 *
 * 🔴 THE MASK IS OVER THE KEY, AND IT IS TRANSPOSED WITH THE ACTIVATION.
 * mask[transpose ? (j,row) : (row,j)]. Using the query's mask instead leaves
 * padding attending to real positions, which changes real answers rather than
 * padded ones.
 *
 * q and k carry AF3's `transpose_weights` and v does not; the exported shapes
 * say so, (heads, dimension, channels) against (channels, heads, dimension).
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const GRID_WIDTH = 32_768;

const ORDER = [
  "actNormScale", "actNormOffset", "pairBiasProjection",
  "qProjection", "kProjection", "vProjection", "gatingQuery", "outputProjection",
];

export function packGridAttentionWeights(weights) {
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    if (weights[name] === undefined) throw new Error(`grid attention weights missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of ORDER) data.set(weights[name], offsets[name]);
  return { data, offsets };
}

export function createGridAttentionShaders(shape, offsets, epsilon, variance, dialect) {
  const { n, channels, heads, dimension, transpose } = shape;
  const width = heads * dimension;
  const pairs = n * n;
  const swapBias = transpose && dialect.swapTransposedBias;

  const common = `
const N: u32 = ${n}u;
const PAIRS: u32 = ${pairs}u;
const CHANNELS: u32 = ${channels}u;
const HEADS: u32 = ${heads}u;
const DIMENSION: u32 = ${dimension}u;
const WIDTH: u32 = ${width}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
const SCALE: f32 = ${1 / Math.sqrt(dimension)};
const W_NORM_SCALE: u32 = ${offsets.actNormScale}u;
const W_NORM_OFFSET: u32 = ${offsets.actNormOffset}u;
const W_BIAS: u32 = ${offsets.pairBiasProjection}u;
const W_Q: u32 = ${offsets.qProjection}u;
const W_K: u32 = ${offsets.kProjection}u;
const W_V: u32 = ${offsets.vProjection}u;
const W_GATE: u32 = ${offsets.gatingQuery}u;
const W_OUT: u32 = ${offsets.outputProjection}u;

fn logistic(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
`;

  // The activation index a pair row reads: (i,j) becomes (j,i) in the column
  // direction.
  const sourceRow = transpose ? "(row % N) * N + row / N" : "row";

  const normalize = `${common}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> normalized: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * CHANNELS;
  var total = 0.0;
  var squares = 0.0;
  for (var c = 0u; c < CHANNELS; c += 1u) {
    let value = pair[base + c];
    total += value;
    squares += value * value;
  }
  let mean = total / f32(CHANNELS);
  ${variance === "fast"
    ? "let variance = squares / f32(CHANNELS) - mean * mean;"
    : `var variance = 0.0;
  for (var c = 0u; c < CHANNELS; c += 1u) {
    let d = pair[base + c] - mean;
    variance += d * d;
  }
  variance /= f32(CHANNELS);`}
  let inverse_std = inverseSqrt(variance + EPSILON);
  for (var c = 0u; c < CHANNELS; c += 1u) {
    normalized[base + c] = (pair[base + c] - mean) * inverse_std * weights[W_NORM_SCALE + c]
      + weights[W_NORM_OFFSET + c];
  }
}`;

  // bias[h][i][j]. Laid out head-major because the attention pass reads a whole
  // row of it per (row, i, head).
  const biasPass = `${common}
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  // 🔴 THE UNTRANSPOSED ROW unless the dialect says otherwise.
  let source = ${swapBias ? "(row % N) * N + row / N" : "row"};
  let base = source * CHANNELS;
  for (var h = 0u; h < HEADS; h += 1u) {
    var total = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      total += normalized[base + c] * weights[W_BIAS + c * HEADS + h];
    }
    bias[h * PAIRS + row] = total;
  }
}`;

  // One workgroup per pair row; thread w owns output channel w of q, k, v and
  // the gate at once, so the normalised row is read from shared memory four
  // times instead of global memory four times.
  const project = `${common}
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> gate: array<f32>;

var<workgroup> act: array<f32, ${channels}>;

@compute @workgroup_size(${width})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= PAIRS) { return; }
  let local = local_id.x;
  let source = ${sourceRow};
  for (var c = local; c < CHANNELS; c += WIDTH) { act[c] = normalized[source * CHANNELS + c]; }
  workgroupBarrier();

  var q_total = 0.0;
  var k_total = 0.0;
  var v_total = 0.0;
  var gate_total = 0.0;
  for (var c = 0u; c < CHANNELS; c += 1u) {
    let x = act[c];
    // q, k and the gate are stored (out, channels); v is (channels, out).
    q_total += x * weights[W_Q + local * CHANNELS + c];
    k_total += x * weights[W_K + local * CHANNELS + c];
    gate_total += x * weights[W_GATE + local * CHANNELS + c];
    v_total += x * weights[W_V + c * WIDTH + local];
  }
  let index = row * WIDTH + local;
  q[index] = q_total;
  k[index] = k_total;
  v[index] = v_total;
  gate[index] = gate_total;
}`;

  // One workgroup per (row, i, head): softmax over that row's j, then the
  // weighted sum of v.
  const attend = `${common}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> gathered: array<f32>;

var<workgroup> logits: array<f32, ${n}>;
var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let slot = group.x + group.y * GRID_WIDTH;
  if (slot >= N * N * HEADS) { return; }
  let head = slot % HEADS;
  let i = (slot / HEADS) % N;
  let row = slot / (HEADS * N);
  let local = local_id.x;

  let query_base = (row * N + i) * WIDTH + head * DIMENSION;
  for (var j = local; j < N; j += 64u) {
    let key_base = (row * N + j) * WIDTH + head * DIMENSION;
    var dot = 0.0;
    for (var d = 0u; d < DIMENSION; d += 1u) {
      dot += q[query_base + d] * k[key_base + d];
    }
    // The KEY's mask, transposed with the activation.
    let masked = mask[${transpose ? "j * N + row" : "row * N + j"}];
    var value = dot * SCALE + bias[head * PAIRS + i * N + j];
    if (masked <= 0.0) { value = value - 1.0e9; }
    logits[j] = value;
  }
  workgroupBarrier();

  var local_max = -3.0e38;
  for (var j = local; j < N; j += 64u) { local_max = max(local_max, logits[j]); }
  reduce[local] = local_max;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
    workgroupBarrier();
  }
  let largest = reduce[0];
  workgroupBarrier();

  var local_sum = 0.0;
  for (var j = local; j < N; j += 64u) {
    let weight = exp(logits[j] - largest);
    logits[j] = weight;
    local_sum += weight;
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
    for (var j = 0u; j < N; j += 1u) {
      sum += logits[j] * v[(row * N + j) * WIDTH + head * DIMENSION + d];
    }
    gathered[(row * N + i) * WIDTH + head * DIMENSION + d] = sum / total;
  }
}`;

  // Gate, project down, and undo the transpose so the residual lands on the
  // orientation it came from.
  const project_out = `${common}
@group(0) @binding(0) var<storage, read> gathered: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> gated: array<f32, ${width}>;

@compute @workgroup_size(${width})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= PAIRS) { return; }
  let local = local_id.x;
  let index = row * WIDTH + local;
  gated[local] = gathered[index] * logistic(gate[index]);
  workgroupBarrier();

  let destination = ${transpose ? "(row % N) * N + row / N" : "row"};
  for (var c = local; c < CHANNELS; c += WIDTH) {
    var sum = 0.0;
    for (var w = 0u; w < WIDTH; w += 1u) {
      sum += gated[w] * weights[W_OUT + w * CHANNELS + c];
    }
    output[destination * CHANNELS + c] = sum;
  }
}`;

  return { normalize, bias: biasPass, project, attend, project_out };
}

export class Af3GridSelfAttentionGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {Float32Array} pair n*n*channels
   * @param {Float32Array} mask n*n
   * @param {{n: number, channels: number, transpose: boolean}} shape
   * @param {object} weights heads, dimension, and the eight tensors in ORDER
   * @param {{swapTransposedBias: boolean}} dialect
   * @param {{epsilon?: number, variance?: "fast"|"two-pass"}} options
   */
  async run(pair, mask, shape, weights, dialect, options = {}) {
    const { n, channels, transpose } = shape;
    const { heads, dimension } = weights;
    if (!Number.isInteger(heads) || !Number.isInteger(dimension)) {
      throw new Error("weights.heads and weights.dimension must be integers");
    }
    if (dialect?.swapTransposedBias === undefined) {
      throw new Error("dialect.swapTransposedBias has no default: stock AF3 is false, "
        + "the openfold3 lineage true");
    }
    const width = heads * dimension;
    const pairs = n * n;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    if (pair.length !== pairs * channels) {
      throw new Error(`pair has ${pair.length} elements; expected ${pairs * channels}`);
    }

    const packed = packGridAttentionWeights(weights);
    const sources = createGridAttentionShaders(
      { n, channels, heads, dimension, transpose }, packed.offsets, epsilon, variance, dialect);
    const key = `af3-grid:${n}:${channels}:${heads}:${dimension}:${transpose}`
      + `:${epsilon}:${variance}:${dialect.swapTransposedBias}`;
    const [normalize, bias, project, attend, projectOut] = await Promise.all([
      this.pipelines.get(`${key}:normalize`, sources.normalize),
      this.pipelines.get(`${key}:bias`, sources.bias),
      this.pipelines.get(`${key}:project`, sources.project),
      this.pipelines.get(`${key}:attend`, sources.attend),
      this.pipelines.get(`${key}:project-out`, sources.project_out),
    ]);

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    const linear2d = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
    try {
      const pairBuffer = keep(this.allocator.upload("af3-grid.pair", pair, storage));
      const maskBuffer = keep(this.allocator.upload("af3-grid.mask", mask, storage));
      const weightBuffer = keep(this.allocator.upload("af3-grid.weights", packed.data, storage));
      const normalized = keep(this.allocator.allocate("af3-grid.normalized", pairs * channels * 4, storage));
      const biasBuffer = keep(this.allocator.allocate("af3-grid.bias", heads * pairs * 4, storage));
      const q = keep(this.allocator.allocate("af3-grid.q", pairs * width * 4, storage));
      const k = keep(this.allocator.allocate("af3-grid.k", pairs * width * 4, storage));
      const v = keep(this.allocator.allocate("af3-grid.v", pairs * width * 4, storage));
      const gate = keep(this.allocator.allocate("af3-grid.gate", pairs * width * 4, storage));
      const gathered = keep(this.allocator.allocate("af3-grid.gathered", pairs * width * 4, storage));
      const output = keep(this.allocator.allocate(
        "af3-grid.output", pairs * channels * 4, storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "af3-grid.readback", pairs * channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-grid-attention" });
      const runPass = (label, pipeline, buffers, groups) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(groups[0], groups[1]);
        pass.end();
      };

      runPass("normalize", normalize, [pairBuffer, weightBuffer, normalized],
              linear2d(Math.ceil(pairs / 64)));
      runPass("bias", bias, [normalized, weightBuffer, biasBuffer],
              linear2d(Math.ceil(pairs / 64)));
      runPass("project", project, [normalized, weightBuffer, q, k, v, gate], linear2d(pairs));
      runPass("attend", attend, [q, k, v, biasBuffer, maskBuffer, gathered],
              linear2d(pairs * heads));
      runPass("project-out", projectOut, [gathered, gate, weightBuffer, output], linear2d(pairs));
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, pairs * channels * 4);

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
