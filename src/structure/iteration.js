import {
  createTransitionShaders, TRANSITION_TILE_COLUMNS, TRANSITION_TILE_ROWS,
} from "../evoformer/transition.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const LINEAR_SHADER = createTransitionShaders({}, [])[1];

function pack(input) {
  const w = input.weights;
  const tensors = [
    w.attentionNormScale, w.attentionNormOffset,
    w.transitionWeights[0], w.transitionBiases[0],
    w.transitionWeights[1], w.transitionBiases[1],
    w.transitionWeights[2], w.transitionBiases[2],
    w.transitionNormScale, w.transitionNormOffset,
    w.affineWeight, w.affineBias,
  ];
  const offsets = [];
  let size = 0;
  for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
  const data = new Float32Array(size);
  tensors.forEach((tensor, index) => data.set(tensor, offsets[index]));
  return { data, offsets };
}

function normParams(rows, channels, scale, offset) {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  view.setUint32(0, rows, true);
  view.setUint32(4, channels, true);
  view.setUint32(8, scale, true);
  view.setUint32(12, offset, true);
  view.setFloat32(16, 1e-5, true);
  return new Uint8Array(buffer);
}

const ADD_NORMALIZE_SHADER = `
struct Parameters {
  rows: u32, channels: u32, scale: u32, offset: u32, epsilon: f32,
  padding_0: u32, padding_1: u32, padding_2: u32,
};
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> mean_value: array<f32, 1>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.x;
  if (row >= p.rows) { return; }
  let base = row * p.channels;
  var sum = 0.0;
  for (var c = local.x; c < p.channels; c += 64u) { sum += left[base + c] + right[base + c]; }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { mean_value[0] = partial[0] / f32(p.channels); }
  workgroupBarrier();
  var squared = 0.0;
  for (var c = local.x; c < p.channels; c += 64u) {
    let centered = left[base + c] + right[base + c] - mean_value[0];
    squared += centered * centered;
  }
  partial[local.x] = squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_std = inverseSqrt(partial[0] / f32(p.channels) + p.epsilon);
  for (var c = local.x; c < p.channels; c += 64u) {
    let value = left[base + c] + right[base + c];
    output[base + c] = (value - mean_value[0]) * inverse_std * weights[p.scale + c] + weights[p.offset + c];
  }
}`;

const AFFINE_SHADER = `
@group(0) @binding(0) var<storage, read> affine: array<f32>;
@group(0) @binding(1) var<storage, read> update: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let u = q.yzw;
  return 2.0 * dot(u, v) * u + (q.x * q.x - dot(u, u)) * v + 2.0 * q.x * cross(u, v);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let residue = id.x;
  if (residue >= arrayLength(&update) / 6u) { return; }
  let affine_base = residue * 7u;
  let update_base = residue * 6u;
  let q = vec4<f32>(affine[affine_base], affine[affine_base + 1u], affine[affine_base + 2u], affine[affine_base + 3u]);
  let v = vec3<f32>(update[update_base], update[update_base + 1u], update[update_base + 2u]);
  let delta = vec4<f32>(-dot(q.yzw, v), q.x * v + cross(q.yzw, v));
  let new_q = normalize(q + delta);
  let translation = vec3<f32>(affine[affine_base + 4u], affine[affine_base + 5u], affine[affine_base + 6u]);
  let translation_update = vec3<f32>(update[update_base + 3u], update[update_base + 4u], update[update_base + 5u]);
  let new_translation = translation + rotate(q, translation_update);
  output[affine_base] = new_q.x;
  output[affine_base + 1u] = new_q.y;
  output[affine_base + 2u] = new_q.z;
  output[affine_base + 3u] = new_q.w;
  output[affine_base + 4u] = new_translation.x;
  output[affine_base + 5u] = new_translation.y;
  output[affine_base + 6u] = new_translation.z;
}`;

export class StructurePostAttentionGpu {
  device;
  allocator;
  pipelines;
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(input) {
    const packed = pack(input);
    const [addNormalize, linear, affinePipeline] = await Promise.all([
      this.pipelines.get("structure:add-normalize", ADD_NORMALIZE_SHADER),
      this.pipelines.get("structure:linear", LINEAR_SHADER),
      this.pipelines.get("structure:affine", AFFINE_SHADER),
    ]);
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const upload = (label, value, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.upload(label, value, usage));
    const allocate = (label, elements, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    try {
      const source = upload("structure.source", input.activations);
      const attention = upload("structure.attention", input.attentionUpdate);
      const affine = upload("structure.affine", input.affine);
      const weights = upload("structure.weights", packed.data);
      const attentionNormParams = upload("structure.attention-norm-params", normParams(
        input.length, input.channels, packed.offsets[0], packed.offsets[1],
      ), GPUBufferUsage.UNIFORM);
      const transitionNormParams = upload("structure.transition-norm-params", normParams(
        input.length, input.channels, packed.offsets[8], packed.offsets[9],
      ), GPUBufferUsage.UNIFORM);
      const linearParams = (label, weight, bias, activation, columns = input.channels) =>
        upload(label, new Uint32Array([
          input.length, input.channels, columns, weight, bias, activation, 0, 0,
        ]), GPUBufferUsage.UNIFORM);
      const transitionParams = [
        linearParams("structure.transition-0-params", packed.offsets[2], packed.offsets[3], 1),
        linearParams("structure.transition-1-params", packed.offsets[4], packed.offsets[5], 1),
        linearParams("structure.transition-2-params", packed.offsets[6], packed.offsets[7], 0),
      ];
      const affineParams = linearParams("structure.affine-params", packed.offsets[10], packed.offsets[11], 0, 6);
      const elements = input.length * input.channels;
      const normalized = allocate("structure.attention-normalized", elements);
      const transition0 = allocate("structure.transition-0", elements);
      const transition1 = allocate("structure.transition-1", elements);
      const transition2 = allocate("structure.transition-2", elements);
      const output = allocate("structure.output", elements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const affineUpdate = allocate("structure.affine-update", input.length * 6);
      const affineOutput = allocate("structure.affine-output", input.length * 7,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const encoder = this.device.createCommandEncoder({ label: "structure-post-attention" });
      this.device.pushErrorScope("validation");
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
      pass(addNormalize, [source, attention, weights, attentionNormParams, normalized], input.length);
      pass(linear, [normalized, weights, transitionParams[0], transition0],
        Math.ceil(input.channels / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
      pass(linear, [transition0, weights, transitionParams[1], transition1],
        Math.ceil(input.channels / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
      pass(linear, [transition1, weights, transitionParams[2], transition2],
        Math.ceil(input.channels / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
      pass(addNormalize, [normalized, transition2, weights, transitionNormParams, output], input.length);
      pass(linear, [output, weights, affineParams, affineUpdate],
        Math.ceil(6 / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
      pass(affinePipeline, [affine, affineUpdate, affineOutput], Math.ceil(input.length / 64));
      const actReadback = allocate("structure.act-readback", elements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const affineReadback = allocate(
        "structure.affine-readback", input.length * 7, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      );
      encoder.copyBufferToBuffer(output.buffer, 0, actReadback.buffer, 0, output.byteLength);
      encoder.copyBufferToBuffer(affineOutput.buffer, 0, affineReadback.buffer, 0, affineOutput.byteLength);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await Promise.all([actReadback.buffer.mapAsync(GPUMapMode.READ), affineReadback.buffer.mapAsync(GPUMapMode.READ)]);
      const activations = new Float32Array(actReadback.buffer.getMappedRange().slice(0));
      const affineResult = new Float32Array(affineReadback.buffer.getMappedRange().slice(0));
      actReadback.buffer.unmap(); affineReadback.buffer.unmap();
      return {
        activations, affine: affineResult,
        elapsedMilliseconds: performance.now() - start, memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
    }
  }
}
