import { ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters } from "../evoformer/attention.js";
import {
  createTransitionShaders, TRANSITION_TILE_COLUMNS, TRANSITION_TILE_ROWS,
} from "../evoformer/transition.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const LINEAR_SHADER = createTransitionShaders({}, [])[1];
const AFFINE_INIT_SHADER = `
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let residue = id.x;
  if (residue >= arrayLength(&output) / 7u) { return; }
  let base = residue * 7u;
  output[base] = 1.0;
  for (var c = 1u; c < 7u; c += 1u) { output[base + c] = 0.0; }
}`;

export class StructureInitializeGpu {
  device;
  allocator;
  pipelines;
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(
    msaFirstRow,
    length,
    msaChannels,
    structureChannels,
    weightsValue,
  ) {
    const tensors = [
      weightsValue.singleProjectionWeight, weightsValue.singleProjectionBias,
      weightsValue.singleNormScale, weightsValue.singleNormOffset,
      weightsValue.initialProjectionWeight, weightsValue.initialProjectionBias,
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
      const [linear, normalize, affineInit] = await Promise.all([
        this.pipelines.get("structure-init:linear", LINEAR_SHADER),
        this.pipelines.get("structure-init:normalize", ATTENTION_NORMALIZE_SHADER),
        this.pipelines.get("structure-init:affine", AFFINE_INIT_SHADER),
      ]);
      const source = upload("structure-init.msa", msaFirstRow);
      const weights = upload("structure-init.weights", packed);
      const linearParams = (label, inner, columns, weight, bias) =>
        upload(label, new Uint32Array([length, inner, columns, weight, bias, 0, 0, 0]), GPUBufferUsage.UNIFORM);
      const singleParams = linearParams("structure-init.single-params", msaChannels, structureChannels,
        offsets[0], offsets[1]);
      const initialParams = linearParams("structure-init.initial-params", structureChannels, structureChannels,
        offsets[4], offsets[5]);
      const normParams = upload("structure-init.norm-params", createAttentionNormParameters(
        length, structureChannels, offsets[2], offsets[3], false, 1, length, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const elements = length * structureChannels;
      const single = allocate("structure-init.single", elements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const normalized = allocate("structure-init.normalized", elements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const activations = allocate("structure-init.activations", elements,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const affine = allocate("structure-init.affine", length * 7, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const encoder = this.device.createCommandEncoder({ label: "structure-initialize" });
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
      pass(linear, [source, weights, singleParams, single],
        Math.ceil(structureChannels / TRANSITION_TILE_COLUMNS), Math.ceil(length / TRANSITION_TILE_ROWS));
      pass(normalize, [single, weights, normParams, normalized], length);
      pass(linear, [normalized, weights, initialParams, activations],
        Math.ceil(structureChannels / TRANSITION_TILE_COLUMNS), Math.ceil(length / TRANSITION_TILE_ROWS));
      pass(affineInit, [affine], Math.ceil(length / 64));
      const readbacks = [single, normalized, activations, affine].map((value, index) => {
        const buffer = allocate(`structure-init.readback-${index}`, value.byteLength / 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        encoder.copyBufferToBuffer(value.buffer, 0, buffer.buffer, 0, value.byteLength);
        return buffer;
      });
      this.device.queue.submit([encoder.finish()]);
      await Promise.all(readbacks.map((buffer) => buffer.buffer.mapAsync(GPUMapMode.READ)));
      const values = readbacks.map((buffer) => {
        const value = new Float32Array(buffer.buffer.getMappedRange().slice(0));
        buffer.buffer.unmap();
        return value;
      });
      return {
        single: values[0], initialRepresentation: values[1], activations: values[2], affine: values[3],
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
    }
  }
}
