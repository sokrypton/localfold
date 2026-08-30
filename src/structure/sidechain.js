import {
  createTransitionShaders, TRANSITION_TILE_COLUMNS, TRANSITION_TILE_ROWS,
} from "../evoformer/transition.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const LINEAR_SHADER = createTransitionShaders({}, [])[1];
const RELU_SHADER = `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { output[id.x] = max(source[id.x], 0.0); }
`;
const ADD_SHADER = `
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { output[id.x] = left[id.x] + right[id.x]; }
`;
const ANGLE_SHADER = `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let angle = id.x;
  if (angle >= arrayLength(&source) / 2u) { return; }
  let x = source[angle * 2u];
  let y = source[angle * 2u + 1u];
  let inverse_norm = inverseSqrt(max(1e-12, x * x + y * y));
  output[angle * 2u] = x * inverse_norm;
  output[angle * 2u + 1u] = y * inverse_norm;
}`;

export class SidechainAnglesGpu {
  device;
  allocator;
  pipelines;
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(
    activations,
    initialActivations,
    length,
    inputChannels,
    hiddenChannels,
    weightsValue,
  ) {
    const w = weightsValue;
    const tensors = [
      w.inputWeight, w.inputBias, w.initialInputWeight, w.initialInputBias,
      w.residual1Weights[0], w.residual1Biases[0], w.residual1Weights[1], w.residual1Biases[1],
      w.residual2Weights[0], w.residual2Biases[0], w.residual2Weights[1], w.residual2Biases[1],
      w.angleWeight, w.angleBias,
    ];
    const offsets = [];
    let size = 0;
    for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
    const packed = new Float32Array(size);
    tensors.forEach((tensor, index) => packed.set(tensor, offsets[index]));
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const upload = (label, value, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.upload(label, value, usage));
    const allocate = (label, elements, usage = GPUBufferUsage.STORAGE) =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    try {
      const [linear, relu, add, angleNormalize] = await Promise.all([
        this.pipelines.get("sidechain:linear", LINEAR_SHADER),
        this.pipelines.get("sidechain:relu", RELU_SHADER),
        this.pipelines.get("sidechain:add", ADD_SHADER),
        this.pipelines.get("sidechain:angles", ANGLE_SHADER),
      ]);
      const source = upload("sidechain.source", activations);
      const initial = upload("sidechain.initial", initialActivations);
      const weights = upload("sidechain.weights", packed);
      const inputElements = length * inputChannels;
      const hiddenElements = length * hiddenChannels;
      const reluSource = allocate("sidechain.relu-source", inputElements);
      const reluInitial = allocate("sidechain.relu-initial", inputElements);
      const projection = allocate("sidechain.projection", hiddenElements);
      const initialProjection = allocate("sidechain.initial-projection", hiddenElements);
      let act = allocate("sidechain.act-0", hiddenElements);
      const temporaries = [0, 1].map((block) => ({
        relu1: allocate(`sidechain.block-${block}.relu-1`, hiddenElements),
        linear1: allocate(`sidechain.block-${block}.linear-1`, hiddenElements),
        relu2: allocate(`sidechain.block-${block}.relu-2`, hiddenElements),
        linear2: allocate(`sidechain.block-${block}.linear-2`, hiddenElements),
        output: allocate(`sidechain.block-${block}.output`, hiddenElements),
      }));
      const reluFinal = allocate("sidechain.relu-final", hiddenElements);
      const unnormalized = allocate("sidechain.unnormalized", length * 14,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const angles = allocate("sidechain.angles", length * 14, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const linearParams = (label, inner, columns, weight, bias) =>
        upload(label, new Uint32Array([length, inner, columns, weight, bias, 0, 0, 0]), GPUBufferUsage.UNIFORM);
      const params = [
        linearParams("sidechain.input-params", inputChannels, hiddenChannels, offsets[0], offsets[1]),
        linearParams("sidechain.initial-params", inputChannels, hiddenChannels, offsets[2], offsets[3]),
        linearParams("sidechain.r10-params", hiddenChannels, hiddenChannels, offsets[4], offsets[5]),
        linearParams("sidechain.r11-params", hiddenChannels, hiddenChannels, offsets[6], offsets[7]),
        linearParams("sidechain.r20-params", hiddenChannels, hiddenChannels, offsets[8], offsets[9]),
        linearParams("sidechain.r21-params", hiddenChannels, hiddenChannels, offsets[10], offsets[11]),
        linearParams("sidechain.angle-params", hiddenChannels, 14, offsets[12], offsets[13]),
      ];
      const encoder = this.device.createCommandEncoder({ label: "sidechain-angles" });
      const pass = (pipeline, buffers, x, y = 1) => {
        const compute = encoder.beginComputePass(); compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer: buffer.buffer } })),
        }));
        compute.dispatchWorkgroups(x, y); compute.end();
      };
      pass(relu, [source, reluSource], Math.ceil(inputElements / 64));
      pass(relu, [initial, reluInitial], Math.ceil(inputElements / 64));
      pass(linear, [reluSource, weights, params[0], projection],
        Math.ceil(hiddenChannels / TRANSITION_TILE_COLUMNS), Math.ceil(length / TRANSITION_TILE_ROWS));
      pass(linear, [reluInitial, weights, params[1], initialProjection],
        Math.ceil(hiddenChannels / TRANSITION_TILE_COLUMNS), Math.ceil(length / TRANSITION_TILE_ROWS));
      pass(add, [projection, initialProjection, act], Math.ceil(hiddenElements / 64));
      for (let block = 0; block < 2; block += 1) {
        const temp = temporaries[block];
        pass(relu, [act, temp.relu1], Math.ceil(hiddenElements / 64));
        pass(linear, [temp.relu1, weights, params[2 + block * 2], temp.linear1],
          Math.ceil(hiddenChannels / TRANSITION_TILE_COLUMNS), Math.ceil(length / TRANSITION_TILE_ROWS));
        pass(relu, [temp.linear1, temp.relu2], Math.ceil(hiddenElements / 64));
        pass(linear, [temp.relu2, weights, params[3 + block * 2], temp.linear2],
          Math.ceil(hiddenChannels / TRANSITION_TILE_COLUMNS), Math.ceil(length / TRANSITION_TILE_ROWS));
        pass(add, [act, temp.linear2, temp.output], Math.ceil(hiddenElements / 64));
        act = temp.output;
      }
      pass(relu, [act, reluFinal], Math.ceil(hiddenElements / 64));
      pass(linear, [reluFinal, weights, params[6], unnormalized],
        Math.ceil(14 / TRANSITION_TILE_COLUMNS), Math.ceil(length / TRANSITION_TILE_ROWS));
      pass(angleNormalize, [unnormalized, angles], Math.ceil(length * 7 / 64));
      const readbacks = [unnormalized, angles].map((value, index) => {
        const output = allocate(`sidechain.readback-${index}`, value.byteLength / 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        encoder.copyBufferToBuffer(value.buffer, 0, output.buffer, 0, value.byteLength); return output;
      });
      this.device.queue.submit([encoder.finish()]);
      await Promise.all(readbacks.map((value) => value.buffer.mapAsync(GPUMapMode.READ)));
      const values = readbacks.map((value) => {
        const result = new Float32Array(value.buffer.getMappedRange().slice(0)); value.buffer.unmap(); return result;
      });
      return { unnormalizedAngles: values[0], angles: values[1] };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
    }
  }
}
