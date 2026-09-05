// One ESM-C block on the GPU.
//
// Eight dispatches, deliberately: LayerNorm, qkv projection, QK-norm and RoPE,
// attention, output projection with the residual, FFN LayerNorm, the widening
// projection with its SwiGLU gate, and the contraction with the residual.
//
// 🔴 IT IS NOT FUSED AND THAT IS THE POINT, FOR NOW. `transition-webgpu.js`
// keeps a whole widened row in workgroup memory precisely so it never reaches
// global memory, and this block will want the same treatment - the qkv
// projection alone writes rows x 3456 floats that are read once. But a fused
// kernel that is wrong is hard to localise, and this repository's own scars are
// mostly about kernels that conformed in shape and computed something else. So:
// eight dispatches that can be checked one at a time against
// `oracle-dumps/esmc-*.json`, and a bench afterwards.
//
// 🔴 THE QK LayerNorm IS OVER THE FULL d_model AND HAPPENS BEFORE THE HEAD
// SPLIT, and RoPE pairs channel d with d + headDim/2. Both readings conform in
// shape; see src/esmc/tower-reference.js, where the alternatives are measured at
// 1.2e-1 and 2.8e-1 against 5.9e-7 for these.
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

export const LANES = 64;
export const GRID_WIDTH = 32768;

/**
 * LayerNorm over the last axis, one workgroup a row.
 *
 * 🔴 THE OFFSET BINDING IS DECLARED ONLY WHEN IT IS READ, for the reason the
 * linear kernel's residual is - and this one was written the other way and
 * shipped, because the block only ever used the offset form. ESM-C's final norm
 * is scale-only and the tower is its first caller: the shader compiled, the
 * compiler dropped the unread binding, and the bind group then had an entry the
 * layout did not. "binding index 2 not present in the bind group layout", at
 * dispatch, thirty-six blocks after anything a checker could see.
 */
export function createLayerNormShader({ rows, channels }, hasOffset, epsilon) {
  return `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
${hasOffset ? "@group(0) @binding(2) var<storage, read> offset: array<f32>;" : ""}
@group(0) @binding(${hasOffset ? 3 : 2}) var<storage, read_write> destination: array<f32>;

var<workgroup> sums: array<f32, ${LANES}>;
var<workgroup> squares: array<f32, ${LANES}>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x + group.y * ${GRID_WIDTH}u;
  if (row >= ${rows}u) { return; }
  let base = row * ${channels}u;
  var total = 0.0;
  var square = 0.0;
  for (var c = local.x; c < ${channels}u; c += ${LANES}u) {
    let value = source[base + c];
    total += value;
    square += value * value;
  }
  sums[local.x] = total;
  squares[local.x] = square;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      sums[local.x] += sums[local.x + stride];
      squares[local.x] += squares[local.x + stride];
    }
    workgroupBarrier();
  }
  let mean = sums[0] / ${channels}.0;
  // 🔴 E[x^2] - mean^2, which is what the reference's two-pass form computes to
  // within float32. The alternative - a second pass over the row - is exact and
  // costs a second read of a row already in registers nowhere.
  let variance = max(squares[0] / ${channels}.0 - mean * mean, 0.0);
  let inverse = inverseSqrt(variance + ${epsilon});
  for (var c = local.x; c < ${channels}u; c += ${LANES}u) {
    let normalised = (source[base + c] - mean) * inverse * scale[c];
    destination[base + c] = ${hasOffset ? "normalised + offset[c]" : "normalised"};
  }
}`;
}

/**
 * out = input @ weights^T (+ bias) (+ residual), with the input row staged.
 *
 * `weights` is (outer, inner) row-major, which is how torch stores a Linear and
 * how tools/export_esmc_model.py writes it.
 *
 * 🔴 THE RESIDUAL BINDING IS DECLARED ONLY WHEN IT IS READ. A binding a shader
 * declares and never uses can be eliminated by the compiler, and then the bind
 * group has an entry the layout does not - which fails validation at dispatch,
 * not at compile, and reads as a broken kernel. Binding a one-element dummy
 * instead is the same trap with an extra buffer.
 */
export function createLinearShader({ rows, inner, outer }, withResidual) {
  return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
${withResidual ? "@group(0) @binding(2) var<storage, read> residual: array<f32>;" : ""}
@group(0) @binding(${withResidual ? 3 : 2}) var<storage, read_write> destination: array<f32>;

var<workgroup> row_values: array<f32, ${inner}>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x + group.y * ${GRID_WIDTH}u;
  if (row >= ${rows}u) { return; }
  let base = row * ${inner}u;
  for (var c = local.x; c < ${inner}u; c += ${LANES}u) {
    row_values[c] = input[base + c];
  }
  workgroupBarrier();
  for (var o = local.x; o < ${outer}u; o += ${LANES}u) {
    let w = o * ${inner}u;
    var sum = 0.0;
    for (var i = 0u; i < ${inner}u; i += 1u) {
      sum += row_values[i] * weights[w + i];
    }
    let slot = row * ${outer}u + o;
    destination[slot] = ${withResidual ? "residual[slot] + sum" : "sum"};
  }
}`;
}

/**
 * Split the fused qkv, LayerNorm q and k over the FULL width, rotate both.
 *
 * One workgroup a row. The two norms are computed here rather than by the
 * LayerNorm kernel because they read from the packed (rows, 3 * model) tensor
 * and write into three separate ones; routing that through a generic kernel
 * means three copies of a tensor that is read once.
 */
export function createPrepareShader({ rows, model, heads }, epsilon, base) {
  const headDim = model / heads;
  return `
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> qScale: array<f32>;
@group(0) @binding(2) var<storage, read> kScale: array<f32>;
@group(0) @binding(3) var<storage, read_write> query: array<f32>;
@group(0) @binding(4) var<storage, read_write> key: array<f32>;
@group(0) @binding(5) var<storage, read_write> value: array<f32>;

var<workgroup> sums: array<f32, ${LANES}>;
var<workgroup> squares: array<f32, ${LANES}>;

fn reduce(lane: u32) {
  for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
      squares[lane] += squares[lane + stride];
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x + group.y * ${GRID_WIDTH}u;
  if (row >= ${rows}u) { return; }
  let packed = row * ${3 * model}u;
  let out = row * ${model}u;

  // v needs neither norm nor rotation.
  for (var c = local.x; c < ${model}u; c += ${LANES}u) {
    value[out + c] = qkv[packed + ${2 * model}u + c];
  }

  // 🔴 THE NORM IS OVER ALL ${model} CHANNELS, not over each head's ${headDim}.
  for (var side = 0u; side < 2u; side += 1u) {
    // 'from' is a WGSL reserved keyword, hence origin.
    let origin = packed + side * ${model}u;
    var total = 0.0;
    var square = 0.0;
    for (var c = local.x; c < ${model}u; c += ${LANES}u) {
      let v = qkv[origin + c];
      total += v;
      square += v * v;
    }
    sums[local.x] = total;
    squares[local.x] = square;
    workgroupBarrier();
    reduce(local.x);
    let mean = sums[0] / ${model}.0;
    let variance = max(squares[0] / ${model}.0 - mean * mean, 0.0);
    let inverse = inverseSqrt(variance + ${epsilon});
    for (var c = local.x; c < ${model}u; c += ${LANES}u) {
      let scale = select(kScale[c], qScale[c], side == 0u);
      let normalised = (qkv[origin + c] - mean) * inverse * scale;
      if (side == 0u) { query[out + c] = normalised; } else { key[out + c] = normalised; }
    }
    workgroupBarrier();
  }

  // 🔴 SPLIT HALVES: channel d of a head pairs with d + ${headDim / 2}, not with
  // d + 1. Interleaving adjacent channels is the other common convention and
  // conforms in shape.
  for (var slot = local.x; slot < ${(model / 2)}u; slot += ${LANES}u) {
    let head = slot / ${headDim / 2}u;
    let d = slot % ${headDim / 2}u;
    let frequency = f32(row) / pow(${base}, f32(2u * d) / ${headDim}.0);
    let c = cos(frequency);
    let s = sin(frequency);
    let first = out + head * ${headDim}u + d;
    let second = first + ${headDim / 2}u;
    let q0 = query[first]; let q1 = query[second];
    query[first] = q0 * c - q1 * s;
    query[second] = q0 * s + q1 * c;
    let k0 = key[first]; let k1 = key[second];
    key[first] = k0 * c - k1 * s;
    key[second] = k0 * s + k1 * c;
  }
}`;
}

/**
 * Full self-attention, one workgroup per (row, head), one lane per channel.
 *
 * 🔴 THE LOGITS ARE COMPUTED BY LANE-OVER-KEYS, NOT BY REDUCING PER KEY. Giving
 * each lane a stride of keys and a whole dot product costs `rows` multiply-adds
 * a lane and two tree reductions in total; reducing across lanes for every key
 * costs `rows` reductions. The second is the obvious shape and it is
 * log2(lanes) times the barriers.
 */
export function createAttentionShader({ rows, model, heads }) {
  const headDim = model / heads;
  return `
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> destination: array<f32>;

var<workgroup> logits: array<f32, ${rows}>;
var<workgroup> staged: array<f32, ${headDim}>;
var<workgroup> partial: array<f32, ${LANES}>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let flat = group.x + group.y * ${GRID_WIDTH}u;
  if (flat >= ${rows * heads}u) { return; }
  let row = flat / ${heads}u;
  let head = flat % ${heads}u;
  let qBase = row * ${model}u + head * ${headDim}u;

  for (var d = local.x; d < ${headDim}u; d += ${LANES}u) { staged[d] = query[qBase + d]; }
  workgroupBarrier();

  var largest = -3.0e38;
  for (var j = local.x; j < ${rows}u; j += ${LANES}u) {
    let kBase = j * ${model}u + head * ${headDim}u;
    var dot = 0.0;
    for (var d = 0u; d < ${headDim}u; d += 1u) { dot += staged[d] * key[kBase + d]; }
    let logit = dot * ${(1 / Math.sqrt(headDim)).toPrecision(9)};
    logits[j] = logit;
    largest = max(largest, logit);
  }
  partial[local.x] = largest;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) { partial[local.x] = max(partial[local.x], partial[local.x + stride]); }
    workgroupBarrier();
  }
  let peak = partial[0];
  workgroupBarrier();

  var total = 0.0;
  for (var j = local.x; j < ${rows}u; j += ${LANES}u) {
    let w = exp(logits[j] - peak);
    logits[j] = w;
    total += w;
  }
  partial[local.x] = total;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let sum = partial[0];

  for (var d = local.x; d < ${headDim}u; d += ${LANES}u) {
    var accumulated = 0.0;
    for (var j = 0u; j < ${rows}u; j += 1u) {
      accumulated += logits[j] * value[j * ${model}u + head * ${headDim}u + d];
    }
    destination[row * ${model}u + head * ${headDim}u + d] = accumulated / sum;
  }
}`;
}

/** LayerNorm, widen to [gate | value], gate, in one pass over the row. */
export function createSwigluShader({ rows, model, ffn }, epsilon) {
  return `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> offset: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> destination: array<f32>;

var<workgroup> row_values: array<f32, ${model}>;
var<workgroup> sums: array<f32, ${LANES}>;
var<workgroup> squares: array<f32, ${LANES}>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x + group.y * ${GRID_WIDTH}u;
  if (row >= ${rows}u) { return; }
  let base = row * ${model}u;
  var total = 0.0;
  var square = 0.0;
  for (var c = local.x; c < ${model}u; c += ${LANES}u) {
    let v = source[base + c];
    total += v;
    square += v * v;
  }
  sums[local.x] = total;
  squares[local.x] = square;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      sums[local.x] += sums[local.x + stride];
      squares[local.x] += squares[local.x + stride];
    }
    workgroupBarrier();
  }
  let mean = sums[0] / ${model}.0;
  let variance = max(squares[0] / ${model}.0 - mean * mean, 0.0);
  let inverse = inverseSqrt(variance + ${epsilon});
  for (var c = local.x; c < ${model}u; c += ${LANES}u) {
    row_values[c] = (source[base + c] - mean) * inverse * scale[c] + offset[c];
  }
  workgroupBarrier();

  // 🔴 THE GATE HALF IS FIRST: silu(wide[c]) * wide[${ffn} + c]. Swapped, the
  // block still runs and returns a plausible tensor of the same shape.
  for (var c = local.x; c < ${ffn}u; c += ${LANES}u) {
    var gate = 0.0;
    var linear = 0.0;
    let gateRow = c * ${model}u;
    let linearRow = (c + ${ffn}u) * ${model}u;
    for (var i = 0u; i < ${model}u; i += 1u) {
      gate += row_values[i] * weights[gateRow + i];
      linear += row_values[i] * weights[linearRow + i];
    }
    destination[row * ${ffn}u + c] = (gate / (1.0 + exp(-gate))) * linear;
  }
}`;
}

export class EsmcBlockGpu {
  constructor(device, allocator = new GpuBufferAllocator(device)) {
    this.device = device;
    this.allocator = allocator;
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * One block. `input` is (rows, model); `weights` holds the ten tensors
   * tools/export_esmc_model.py writes for a block, under their leaf names.
   */
  async run(input, shape, weights, options = {}) {
    const { rows, model, heads, ffn } = shape;
    const residualScale = shape.residualScale ?? 1;
    const epsilon = (options.epsilon ?? 1e-5).toExponential();
    const ropeBase = (options.ropeBase ?? 10000).toFixed(1);
    if (input.length !== rows * model) {
      throw new Error(`input has ${input.length} elements; expected ${rows * model}`);
    }
    if (model % heads !== 0) throw new Error(`${model} channels over ${heads} heads`);

    const storage = GPUBufferUsage.STORAGE;
    const kept = [];
    const keep = (allocation) => { kept.push(allocation); return allocation; };
    const upload = (name, values) => keep(this.allocator.upload(`esmc.${name}`, values, storage));
    const scratch = (name, elements) => keep(this.allocator.allocate(
      `esmc.${name}`, elements * 4, storage | GPUBufferUsage.COPY_SRC));
    try {
      const x = upload("input", input);
      const normed = scratch("normed", rows * model);
      const qkv = scratch("qkv", rows * 3 * model);
      const query = scratch("query", rows * model);
      const key = scratch("key", rows * model);
      const value = scratch("value", rows * model);
      const context = scratch("context", rows * model);
      const afterAttention = scratch("afterAttention", rows * model);
      const gated = scratch("gated", rows * ffn);
      const output = scratch("output", rows * model);

      const w = (leaf) => upload(leaf, weights[leaf]);
      const attnScale = w("attn_norm/scale");
      const attnOffset = w("attn_norm/offset");
      const qkvWeights = w("qkv/weights");
      const qScale = w("q_norm/scale");
      const kScale = w("k_norm/scale");
      const attnOut = w("attn_out/weights");
      const ffnScale = w("ffn_norm/scale");
      const ffnOffset = w("ffn_norm/offset");
      const fc1 = w("fc1/weights");
      const fc2 = w("fc2/weights");

      // 🔴 THE RESIDUAL SCALE IS FOLDED INTO THE PROJECTION WEIGHTS, not applied
      // as a separate pass. It is sqrt(layers / 36), which is exactly 1 at 36
      // layers - so a port that dropped it entirely would agree here and diverge
      // on the 6B tower, where it is sqrt(80/36).
      const scaled = (values) => {
        if (residualScale === 1) return values;
        const out = new Float32Array(values.length);
        for (let i = 0; i < values.length; i += 1) out[i] = values[i] / residualScale;
        return out;
      };
      const attnOutScaled = residualScale === 1 ? attnOut
        : upload("attn_out/scaled", scaled(weights["attn_out/weights"]));
      const fc2Scaled = residualScale === 1 ? fc2
        : upload("fc2/scaled", scaled(weights["fc2/weights"]));

      const pipeline = async (name, source) => this.pipelines.get(name, source);
      const normPipeline = await pipeline(
        `esmc-ln:${rows}:${model}:${epsilon}`,
        createLayerNormShader({ rows, channels: model }, true, epsilon));
      const qkvPipeline = await pipeline(
        `esmc-linear:${rows}:${model}:${3 * model}:0`,
        createLinearShader({ rows, inner: model, outer: 3 * model }, false));
      const preparePipeline = await pipeline(
        `esmc-prepare:${rows}:${model}:${heads}:${epsilon}:${ropeBase}`,
        createPrepareShader({ rows, model, heads }, epsilon, ropeBase));
      const attentionPipeline = await pipeline(
        `esmc-attend:${rows}:${model}:${heads}`,
        createAttentionShader({ rows, model, heads }));
      const outPipeline = await pipeline(
        `esmc-linear:${rows}:${model}:${model}:1`,
        createLinearShader({ rows, inner: model, outer: model }, true));
      const swigluPipeline = await pipeline(
        `esmc-swiglu:${rows}:${model}:${ffn}:${epsilon}`,
        createSwigluShader({ rows, model, ffn }, epsilon));
      const downPipeline = await pipeline(
        `esmc-linear:${rows}:${ffn}:${model}:1`,
        createLinearShader({ rows, inner: ffn, outer: model }, true));

      const readback = keep(this.allocator.allocate(
        "esmc.readback", rows * model * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      // 🔴 THE qkv INPUT COMES BACK TOO, BECAUSE IT IS THE ONE POINT THE ORACLE
      // RECORDS INSIDE THE BLOCK. A block that disagrees only at its output
      // leaves eight dispatches to bisect; a block that agrees on `normed` and
      // disagrees at the output has narrowed it to seven, and the LayerNorm is
      // the dispatch most likely to be right.
      const normedReadback = keep(this.allocator.allocate(
        "esmc.readback.normed", rows * model * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "esmc-block" });
      const pass = encoder.beginComputePass({ label: "esmc-block" });
      const dispatch = (built, bindings, count) => {
        pass.setPipeline(built);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: built.getBindGroupLayout(0),
          entries: bindings.map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(Math.min(count, GRID_WIDTH), Math.ceil(count / GRID_WIDTH));
      };

      dispatch(normPipeline, [x, attnScale, attnOffset, normed], rows);
      dispatch(qkvPipeline, [normed, qkvWeights, qkv], rows);
      dispatch(preparePipeline, [qkv, qScale, kScale, query, key, value], rows);
      dispatch(attentionPipeline, [query, key, value, context], rows * heads);
      dispatch(outPipeline, [context, attnOutScaled, x, afterAttention], rows);
      dispatch(swigluPipeline, [afterAttention, ffnScale, ffnOffset, fc1, gated], rows);
      dispatch(downPipeline, [gated, fc2Scaled, afterAttention, output], rows);
      pass.end();
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, rows * model * 4);
      encoder.copyBufferToBuffer(normed.buffer, 0, normedReadback.buffer, 0, rows * model * 4);

      const started = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      await normedReadback.buffer.mapAsync(GPUMapMode.READ);
      const normedResult = new Float32Array(normedReadback.buffer.getMappedRange().slice(0));
      normedReadback.buffer.unmap();
      return {
        output: result, normed: normedResult,
        elapsedMilliseconds: performance.now() - started,
      };
    } finally {
      for (let index = kept.length - 1; index >= 0; index -= 1) kept[index].release();
    }
  }
}
