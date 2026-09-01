import { ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters } from "../evoformer/attention.js";
import {
  createTransitionShaders, TRANSITION_TILE_COLUMNS, TRANSITION_TILE_ROWS,
} from "../evoformer/transition.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const LINEAR_SHADER = createTransitionShaders({}, [])[1];
const GRID_WIDTH = 32_768;

function packWeights(input) {
  const w = input.weights;
  const values = [
    w.pairNormScale, w.pairNormOffset,
    w.queryScalarWeight, w.queryScalarBias,
    w.keyValueScalarWeight, w.keyValueScalarBias,
    w.queryPointWeight, w.queryPointBias,
    w.keyValuePointWeight, w.keyValuePointBias,
    w.trainablePointWeights,
    w.attention2dWeight, w.attention2dBias,
    w.outputWeight, w.outputBias,
  ];
  const offsets = [];
  let size = 0;
  for (const value of values) { offsets.push(size); size += value.length; }
  const data = new Float32Array(size);
  values.forEach((value, index) => data.set(value, offsets[index]));
  return { data, offsets };
}

function parameters(input, offsets) {
  const buffer = new ArrayBuffer(128);
  const view = new DataView(buffer);
  const featureChannels = input.heads * input.scalarV
    + 4 * input.heads * input.pointV + input.heads * input.pairChannels;
  const integers = [
    input.length, input.channels, input.pairChannels, input.heads,
    input.scalarQk, input.scalarV, input.pointQk, input.pointV, featureChannels,
    ...offsets,
  ];
  integers.forEach((value, index) => view.setUint32(index * 4, value, true));
  view.setFloat32(96, Math.sqrt(1 / (3 * input.scalarQk)), true);
  view.setFloat32(100, Math.sqrt(1 / (3 * input.pointQk * 4.5)), true);
  view.setFloat32(104, Math.sqrt(1 / 3), true);
  return new Uint8Array(buffer);
}

const COMMON = `
const GRID_WIDTH: u32 = 32768u;
struct Parameters {
  length: u32, channels: u32, pair_channels: u32, heads: u32,
  scalar_qk: u32, scalar_v: u32, point_qk: u32, point_v: u32, feature_channels: u32,
  pair_norm_scale: u32, pair_norm_offset: u32,
  query_scalar_weight: u32, query_scalar_bias: u32,
  kv_scalar_weight: u32, kv_scalar_bias: u32,
  query_point_weight: u32, query_point_bias: u32,
  kv_point_weight: u32, kv_point_bias: u32,
  trainable_point_weights: u32,
  attention_2d_weight: u32, attention_2d_bias: u32,
  output_weight: u32, output_bias: u32,
  scalar_factor: f32, point_factor: f32, attention_2d_factor: f32,
  padding_0: u32, padding_1: u32, padding_2: u32, padding_3: u32, padding_4: u32,
};
`;

const POINT_SHADER = `
const GRID_WIDTH: u32 = 32768u;
struct PointParameters { length: u32, heads: u32, points: u32, padding: u32 };
@group(0) @binding(0) var<storage, read> local_points: array<f32>;
@group(0) @binding(1) var<storage, read> affine: array<f32>;
@group(0) @binding(2) var<uniform> p: PointParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let u = q.yzw;
  return 2.0 * dot(u, v) * u + (q.x * q.x - dot(u, u)) * v + 2.0 * q.x * cross(u, v);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.heads * p.points) { return; }
  let point = index % p.points;
  let head = (index / p.points) % p.heads;
  let residue = index / (p.points * p.heads);
  let plane = p.heads * p.points;
  let base = residue * 3u * plane + head * p.points + point;
  let local = vec3<f32>(local_points[base], local_points[base + plane], local_points[base + 2u * plane]);
  let affine_base = residue * 7u;
  let q = vec4<f32>(affine[affine_base], affine[affine_base + 1u], affine[affine_base + 2u], affine[affine_base + 3u]);
  let translation = vec3<f32>(affine[affine_base + 4u], affine[affine_base + 5u], affine[affine_base + 6u]);
  let global = rotate(q, local) + translation;
  let output_base = index * 3u;
  output[output_base] = global.x;
  output[output_base + 1u] = global.y;
  output[output_base + 2u] = global.z;
}`;

const LOGITS_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> query_scalar: array<f32>;
@group(0) @binding(1) var<storage, read> kv_scalar: array<f32>;
@group(0) @binding(2) var<storage, read> query_point: array<f32>;
@group(0) @binding(3) var<storage, read> kv_point: array<f32>;
@group(0) @binding(4) var<storage, read> pair: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<f32>;
@group(0) @binding(6) var<storage, read> weights: array<f32>;
@group(0) @binding(7) var<uniform> p: Parameters;
@group(0) @binding(8) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.heads * p.length * p.length) { return; }
  let key_index = index % p.length;
  let query = (index / p.length) % p.length;
  let head = index / (p.length * p.length);
  var result = 0.0;
  let q_scalar_base = (query * p.heads + head) * p.scalar_qk;
  let kv_scalar_stride = p.scalar_qk + p.scalar_v;
  let k_scalar_base = (key_index * p.heads + head) * kv_scalar_stride;
  for (var c = 0u; c < p.scalar_qk; c += 1u) {
    result += p.scalar_factor * query_scalar[q_scalar_base + c] * kv_scalar[k_scalar_base + c];
  }
  var distance = 0.0;
  let q_point_base = (query * p.heads + head) * p.point_qk * 3u;
  let kv_points = p.point_qk + p.point_v;
  let k_point_base = (key_index * p.heads + head) * kv_points * 3u;
  for (var point = 0u; point < p.point_qk; point += 1u) {
    for (var coordinate = 0u; coordinate < 3u; coordinate += 1u) {
      let delta = query_point[q_point_base + point * 3u + coordinate]
        - kv_point[k_point_base + point * 3u + coordinate];
      distance += delta * delta;
    }
  }
  let point_weight = p.point_factor * log(1.0 + exp(weights[p.trainable_point_weights + head]));
  result -= 0.5 * point_weight * distance;
  var pair_bias = weights[p.attention_2d_bias + head];
  let pair_base = (query * p.length + key_index) * p.pair_channels;
  for (var c = 0u; c < p.pair_channels; c += 1u) {
    pair_bias += pair[pair_base + c] * weights[p.attention_2d_weight + c * p.heads + head];
  }
  result += p.attention_2d_factor * pair_bias;
  result -= 1e5 * (1.0 - mask[query] * mask[key_index]);
  output[index] = result;
}`;

const SOFTMAX_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<uniform> p: Parameters;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= p.heads * p.length) { return; }
  let base = row * p.length;
  var maximum = -1e30;
  for (var k = 0u; k < p.length; k += 1u) { maximum = max(maximum, logits[base + k]); }
  var sum = 0.0;
  for (var k = 0u; k < p.length; k += 1u) { sum += exp(logits[base + k] - maximum); }
  for (var k = 0u; k < p.length; k += 1u) { output[base + k] = exp(logits[base + k] - maximum) / sum; }
}`;

const SCALAR_FEATURE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> attention: array<f32>;
@group(0) @binding(1) var<storage, read> kv_scalar: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> features: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.heads * p.scalar_v) { return; }
  let value_channel = index % p.scalar_v;
  let head = (index / p.scalar_v) % p.heads;
  let query = index / (p.scalar_v * p.heads);
  var result = 0.0;
  for (var key_index = 0u; key_index < p.length; key_index += 1u) {
    let a = attention[(head * p.length + query) * p.length + key_index];
    let kv_base = (key_index * p.heads + head) * (p.scalar_qk + p.scalar_v);
    result += a * kv_scalar[kv_base + p.scalar_qk + value_channel];
  }
  features[query * p.feature_channels + head * p.scalar_v + value_channel] = result;
}`;

const POINT_FEATURE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> attention: array<f32>;
@group(0) @binding(1) var<storage, read> kv_point: array<f32>;
@group(0) @binding(2) var<storage, read> affine: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> features: array<f32>;

fn inverse_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  return 2.0 * dot(q.yzw, v) * q.yzw + (q.x * q.x - dot(q.yzw, q.yzw)) * v - 2.0 * q.x * cross(q.yzw, v);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.heads * p.point_v) { return; }
  let point = index % p.point_v;
  let head = (index / p.point_v) % p.heads;
  let query = index / (p.point_v * p.heads);
  var global = vec3<f32>(0.0);
  let kv_points = p.point_qk + p.point_v;
  for (var key_index = 0u; key_index < p.length; key_index += 1u) {
    let a = attention[(head * p.length + query) * p.length + key_index];
    let point_base = ((key_index * p.heads + head) * kv_points + p.point_qk + point) * 3u;
    global += a * vec3<f32>(kv_point[point_base], kv_point[point_base + 1u], kv_point[point_base + 2u]);
  }
  let affine_base = query * 7u;
  let q = vec4<f32>(affine[affine_base], affine[affine_base + 1u], affine[affine_base + 2u], affine[affine_base + 3u]);
  let translation = vec3<f32>(affine[affine_base + 4u], affine[affine_base + 5u], affine[affine_base + 6u]);
  let local = inverse_rotate(q, global - translation);
  let scalar_size = p.heads * p.scalar_v;
  let point_size = p.heads * p.point_v;
  let point_index = head * p.point_v + point;
  let base = query * p.feature_channels;
  features[base + scalar_size + point_index] = local.x;
  features[base + scalar_size + point_size + point_index] = local.y;
  features[base + scalar_size + 2u * point_size + point_index] = local.z;
  features[base + scalar_size + 3u * point_size + point_index] = sqrt(1e-8 + dot(local, local));
}`;

const PAIR_FEATURE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> attention: array<f32>;
@group(0) @binding(1) var<storage, read> pair: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> features: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.heads * p.pair_channels) { return; }
  let channel = index % p.pair_channels;
  let head = (index / p.pair_channels) % p.heads;
  let query = index / (p.pair_channels * p.heads);
  var result = 0.0;
  for (var key_index = 0u; key_index < p.length; key_index += 1u) {
    result += attention[(head * p.length + query) * p.length + key_index]
      * pair[(query * p.length + key_index) * p.pair_channels + channel];
  }
  let offset = p.heads * p.scalar_v + 4u * p.heads * p.point_v;
  features[query * p.feature_channels + offset + head * p.pair_channels + channel] = result;
}`;
/**
 * WHAT THE EIGHT ITERATIONS SHARE.
 *
 * The structure module runs invariant point attention eight times over the SAME
 * pair representation, the same weights, the same residue mask and the same
 * dimensions; only the activations and the backbone frames differ between them.
 * Every one of those calls used to upload the pair tensor and layer-normalise
 * it again - L*L*128 floats, 25 MiB at 221 residues - so a pass spent 200 MiB
 * of uploads and eight identical normalisations arriving at a buffer that was
 * byte for byte the one before it. The pair track is the largest thing in the
 * model and the structure module was the heaviest re-uploader of it anywhere.
 *
 * PREPARE ONCE, RUN EIGHT TIMES. Everything invariant lives in a handle the
 * caller holds for as long as its iterations run and then releases. What is
 * left in run() is what genuinely changes.
 *
 * THE HANDLE IS OPTIONAL. run() given none prepares its own and releases it
 * before returning, which is exactly what it did before: a standalone IPA call
 * is still a standalone IPA call, and the tests that make one are unaffected.
 */
class PreparedInvariantPointAttention {
  #allocations;
  #released = false;

  constructor(fields, allocations) {
    Object.assign(this, fields);
    this.#allocations = allocations;
  }

  /** Idempotent, because both the owner and a failed prepare may call it. */
  release() {
    if (this.#released) return;
    this.#released = true;
    for (let index = this.#allocations.length - 1; index >= 0; index -= 1) {
      this.#allocations[index] .release();
    }
    this.#allocations.length = 0;
  }
}

export class InvariantPointAttentionGpu {
  device;
  allocator;
  pipelines;
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * Upload and normalise everything the iterations hold constant.
   *
   * @param {object} input the same shape run() takes; only the dimensions, the
   *   weights, the mask and the pair representation are read.
   * @returns {Promise<PreparedInvariantPointAttention>} release it when the
   *   iterations are done - nothing else will.
   */
  async prepare(input) {
    const pipelines = await Promise.all([
      this.pipelines.get("ipa:normalize", ATTENTION_NORMALIZE_SHADER),
      this.pipelines.get("ipa:linear", LINEAR_SHADER),
      this.pipelines.get("ipa:point", POINT_SHADER),
      this.pipelines.get("ipa:logits", LOGITS_SHADER),
      this.pipelines.get("ipa:softmax", SOFTMAX_SHADER),
      this.pipelines.get("ipa:scalar-feature", SCALAR_FEATURE_SHADER),
      this.pipelines.get("ipa:point-feature", POINT_FEATURE_SHADER),
      this.pipelines.get("ipa:pair-feature", PAIR_FEATURE_SHADER),
    ]);
    const packed = packWeights(input);
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const upload = (label, value, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.upload(label, value, usage));
    const allocate = (label, elements, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    const queryScalarColumns = input.heads * input.scalarQk;
    const kvScalarColumns = input.heads * (input.scalarQk + input.scalarV);
    const queryPointColumns = input.heads * 3 * input.pointQk;
    const kvPointColumns = input.heads * 3 * (input.pointQk + input.pointV);
    const featureChannels = input.weights.outputWeight.length / input.channels;
    let prepared;
    try {
      const weights = upload("ipa.weights", packed.data);
      const params = upload("ipa.parameters", parameters(input, packed.offsets), GPUBufferUsage.UNIFORM);
      const mask = upload("ipa.mask", input.mask);
      const pair = allocate("ipa.pair-normalized", input.length * input.length * input.pairChannels);
      const linearParams = (label, columns, weight, bias) =>
        upload(label, new Uint32Array([input.length, input.channels, columns, weight, bias, 0, 0, 0]),
          GPUBufferUsage.UNIFORM);
      prepared = new PreparedInvariantPointAttention({
        pipelines, packed, weights, params, mask, pair,
        queryScalarColumns, kvScalarColumns, queryPointColumns, kvPointColumns, featureChannels,
        qScalarParams: linearParams("ipa.q-scalar-params", queryScalarColumns, packed.offsets[2], packed.offsets[3]),
        kvScalarParams: linearParams("ipa.kv-scalar-params", kvScalarColumns, packed.offsets[4], packed.offsets[5]),
        qPointParams: linearParams("ipa.q-point-params", queryPointColumns, packed.offsets[6], packed.offsets[7]),
        kvPointParams: linearParams("ipa.kv-point-params", kvPointColumns, packed.offsets[8], packed.offsets[9]),
        qPointTransformParams: upload("ipa.q-point-transform-params", new Uint32Array([
          input.length, input.heads, input.pointQk, 0,
        ]), GPUBufferUsage.UNIFORM),
        kvPointTransformParams: upload("ipa.kv-point-transform-params", new Uint32Array([
          input.length, input.heads, input.pointQk + input.pointV, 0,
        ]), GPUBufferUsage.UNIFORM),
        outputParams: upload("ipa.output-params", new Uint32Array([
          input.length, featureChannels, input.channels, packed.offsets[13], packed.offsets[14], 0, 0, 0,
        ]), GPUBufferUsage.UNIFORM),
      }, allocations);
      // 🔴 THE RAW PAIR IS NOT KEPT. It feeds the normalisation and nothing
      // else, so holding it would leave a second L*L*128 tensor - another
      // 25 MiB at 221 residues - resident for the whole structure module to no
      // purpose. Waiting for the normalisation to land is what makes releasing
      // it here unambiguous, and it is ONE wait in place of eight uploads.
      let pairNormParams;
      let pairSource;
      try {
        pairNormParams = this.allocator.upload("ipa.pair-norm-parameters", createAttentionNormParameters(
          input.length * input.length, input.pairChannels, packed.offsets[0], packed.offsets[1],
          false, 1, input.length * input.length, 1e-5,
        ), GPUBufferUsage.UNIFORM);
        pairSource = this.allocator.upload("ipa.pair", input.pair, GPUBufferUsage.STORAGE);
        const encoder = this.device.createCommandEncoder({ label: "ipa.prepare" });
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipelines[0]);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipelines[0] .getBindGroupLayout(0),
          entries: [pairSource, weights, pairNormParams, pair].map((buffer, binding) =>
            ({ binding, resource: { buffer: buffer.buffer } })),
        }));
        // ...FOLDED, because one workgroup per pair row is L*L of them and a
        // dispatch is 65535 wide. The normalise shader reads its row as
        // group.x + group.y * GRID_WIDTH and has done all along.
        const rows = input.length * input.length;
        compute.dispatchWorkgroups(Math.min(rows, GRID_WIDTH), Math.ceil(rows / GRID_WIDTH));
        compute.end();
        this.device.pushErrorScope("validation");
        this.device.queue.submit([encoder.finish()]);
        const error = await this.device.popErrorScope();
        if (error !== null) throw new Error(`WebGPU IPA preparation failed: ${error.message}`);
        await this.device.queue.onSubmittedWorkDone();
      } finally {
        pairSource?.release();
        pairNormParams?.release();
      }
      return prepared;
    } catch (error) {
      if (prepared === undefined) {
        for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
      } else {
        prepared.release();
      }
      throw error;
    }
  }

  /**
   * Scratch the encoded loop reuses across every iteration.
   *
   * 🔴 ALLOCATED ONCE, NOT EIGHT TIMES, AND THAT IS ONLY SAFE INSIDE ONE PASS.
   * Dispatches recorded into a single compute pass run in order with a memory
   * barrier between them, so iteration N+1 cannot start reading a scratch
   * buffer before iteration N has finished writing it. Reusing these across
   * SUBMISSIONS would not be safe, which is why they belong to the encoded path
   * rather than to run().
   */
  allocateScratch(input, shared) {
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const allocate = (label, elements, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    const attentionElements = input.heads * input.length * input.length;
    return {
      queryScalar: allocate("ipa.query-scalar", input.length * shared.queryScalarColumns),
      kvScalar: allocate("ipa.kv-scalar", input.length * shared.kvScalarColumns),
      queryPointLocal: allocate("ipa.query-point-local", input.length * shared.queryPointColumns),
      kvPointLocal: allocate("ipa.kv-point-local", input.length * shared.kvPointColumns),
      queryPoint: allocate("ipa.query-point", input.length * input.heads * input.pointQk * 3),
      kvPoint: allocate("ipa.kv-point",
        input.length * input.heads * (input.pointQk + input.pointV) * 3),
      logits: allocate("ipa.logits", attentionElements),
      attention: allocate("ipa.attention", attentionElements),
      features: allocate("ipa.features", input.length * shared.featureChannels),
      release: () => { for (let i = allocations.length - 1; i >= 0; i -= 1) allocations[i].release(); },
    };
  }

  /**
   * Record one invariant point attention into an open compute pass.
   *
   * Everything is GPU-resident: `source` and `affine` are buffers the caller
   * owns, and `output` is written in place. Nothing is uploaded and nothing is
   * read back, which is the whole point - see the note in structure/core.js.
   */
  encode(compute, input, shared, scratch, source, affine, output) {
    const { pipelines, weights, params, mask, pair } = shared;
    const dispatch = (pipeline, buffers, x, y = 1) => {
      compute.setPipeline(pipeline);
      compute.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer: buffer.buffer } })),
      }));
      compute.dispatchWorkgroups(x, y);
    };
    const linearGrid = (elements, workgroupSize = 64) => {
      const groups = Math.ceil(elements / workgroupSize);
      return [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
    };
    const linear = (paramsValue, result, columns) =>
      dispatch(pipelines[1], [source, weights, paramsValue, result],
        Math.ceil(columns / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
    linear(shared.qScalarParams, scratch.queryScalar, shared.queryScalarColumns);
    linear(shared.kvScalarParams, scratch.kvScalar, shared.kvScalarColumns);
    linear(shared.qPointParams, scratch.queryPointLocal, shared.queryPointColumns);
    linear(shared.kvPointParams, scratch.kvPointLocal, shared.kvPointColumns);
    dispatch(pipelines[2], [scratch.queryPointLocal, affine, shared.qPointTransformParams, scratch.queryPoint],
      Math.ceil(scratch.queryPoint.byteLength / 4 / 3 / 64));
    dispatch(pipelines[2], [scratch.kvPointLocal, affine, shared.kvPointTransformParams, scratch.kvPoint],
      Math.ceil(scratch.kvPoint.byteLength / 4 / 3 / 64));
    const attentionElements = input.heads * input.length * input.length;
    const logitsGrid = linearGrid(attentionElements);
    dispatch(pipelines[3],
      [scratch.queryScalar, scratch.kvScalar, scratch.queryPoint, scratch.kvPoint, pair, mask, weights, params,
        scratch.logits], logitsGrid[0], logitsGrid[1]);
    dispatch(pipelines[4], [scratch.logits, params, scratch.attention], input.heads * input.length);
    dispatch(pipelines[5], [scratch.attention, scratch.kvScalar, params, scratch.features],
      Math.ceil(input.length * input.heads * input.scalarV / 64));
    dispatch(pipelines[6], [scratch.attention, scratch.kvPoint, affine, params, scratch.features],
      Math.ceil(input.length * input.heads * input.pointV / 64));
    dispatch(pipelines[7], [scratch.attention, pair, params, scratch.features],
      Math.ceil(input.length * input.heads * input.pairChannels / 64));
    dispatch(pipelines[1], [scratch.features, weights, shared.outputParams, output],
      Math.ceil(input.channels / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
  }

  async run(input) {
    const shared = input.prepared ?? await this.prepare(input);
    const ownsShared = input.prepared === undefined;
    const {
      pipelines, weights, params, mask, pair, featureChannels,
      queryScalarColumns, kvScalarColumns, queryPointColumns, kvPointColumns,
    } = shared;
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const upload = (label, value, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.upload(label, value, usage));
    const allocate = (label, elements, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    try {
      // ...WHAT ACTUALLY CHANGES between iterations: the activations coming out
      // of the previous one, and the backbone frames it moved.
      const source = upload("ipa.source", input.activations);
      const affine = upload("ipa.affine", input.affine);
      const queryScalar = allocate("ipa.query-scalar", input.length * queryScalarColumns);
      const kvScalar = allocate("ipa.kv-scalar", input.length * kvScalarColumns);
      const queryPointLocal = allocate("ipa.query-point-local", input.length * queryPointColumns);
      const kvPointLocal = allocate("ipa.kv-point-local", input.length * kvPointColumns);
      const queryPoint = allocate("ipa.query-point", input.length * input.heads * input.pointQk * 3);
      const kvPoint = allocate(
        "ipa.kv-point", input.length * input.heads * (input.pointQk + input.pointV) * 3,
      );
      const attentionElements = input.heads * input.length * input.length;
      const logits = allocate("ipa.logits", attentionElements);
      const attention = allocate("ipa.attention", attentionElements);
      const features = allocate("ipa.features", input.length * featureChannels);
      const output = allocate("ipa.output", input.length * input.channels, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const encoder = this.device.createCommandEncoder({ label: "invariant-point-attention" });
      this.device.pushErrorScope("validation");
      const linearGrid = (elements, workgroupSize = 64) => {
        const groups = Math.ceil(elements / workgroupSize);
        return [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
      };
      const pass = (pipeline, buffers, x, y = 1) => {
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer: buffer.buffer } })),
        }));
        compute.dispatchWorkgroups(x, y);
        compute.end();
      };
      const linear = (paramsValue, result, columns) =>
        pass(pipelines[1], [source, weights, paramsValue, result],
          Math.ceil(columns / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
      linear(shared.qScalarParams, queryScalar, queryScalarColumns);
      linear(shared.kvScalarParams, kvScalar, kvScalarColumns);
      linear(shared.qPointParams, queryPointLocal, queryPointColumns);
      linear(shared.kvPointParams, kvPointLocal, kvPointColumns);
      pass(pipelines[2], [queryPointLocal, affine, shared.qPointTransformParams, queryPoint],
        Math.ceil(queryPoint.byteLength / 4 / 3 / 64));
      pass(pipelines[2], [kvPointLocal, affine, shared.kvPointTransformParams, kvPoint],
        Math.ceil(kvPoint.byteLength / 4 / 3 / 64));
      // ...FOLDED. heads * L * L elements over 64 lanes passes 65535 workgroups
      // at about 590 residues, which is inside the range this page will fold.
      const logitsGrid = linearGrid(attentionElements);
      pass(pipelines[3], [queryScalar, kvScalar, queryPoint, kvPoint, pair, mask, weights, params, logits],
        logitsGrid[0], logitsGrid[1]);
      pass(pipelines[4], [logits, params, attention], input.heads * input.length);
      pass(pipelines[5], [attention, kvScalar, params, features],
        Math.ceil(input.length * input.heads * input.scalarV / 64));
      pass(pipelines[6], [attention, kvPoint, affine, params, features],
        Math.ceil(input.length * input.heads * input.pointV / 64));
      pass(pipelines[7], [attention, pair, params, features],
        Math.ceil(input.length * input.heads * input.pairChannels / 64));
      pass(pipelines[1], [features, weights, shared.outputParams, output],
        Math.ceil(input.channels / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
      const outputElements = output.byteLength / 4;
      const readback = allocate("ipa.readback", outputElements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, output.byteLength);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return { output: result, elapsedMilliseconds: performance.now() - start, memory: this.allocator.snapshot() };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
      if (ownsShared) shared.release();
    }
  }
}
