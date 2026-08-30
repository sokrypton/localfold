import { GpuBufferAllocator } from "./allocator.js";
import { pipelineCacheForDevice } from "./pipeline-cache.js";

const SHADER = `
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  output[index] = left[index] + right[index];
}`;

export class ElementwiseAddGpu {
  device;
  allocator;
  pipelines;
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(leftValue, rightValue) {
    if (leftValue.length !== rightValue.length || leftValue.length === 0) {
      throw new RangeError("elementwise add requires equal non-empty tensors");
    }
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    try {
      const left = keep(this.allocator.upload("add.left", leftValue, GPUBufferUsage.STORAGE));
      const right = keep(this.allocator.upload("add.right", rightValue, GPUBufferUsage.STORAGE));
      const bytes = leftValue.length * 4;
      const output = keep(this.allocator.allocate("add.output", bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "add.readback", bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      ));
      const pipeline = await this.pipelines.get("elementwise:add", SHADER);
      const encoder = this.device.createCommandEncoder({ label: "elementwise-add" });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [left, right, output].map((allocation, binding) => ({
          binding, resource: { buffer: allocation.buffer },
        })),
      }));
      const groups = Math.ceil(leftValue.length / 64);
      pass.dispatchWorkgroups(Math.min(groups, 32768), Math.ceil(groups / 32768));
      pass.end();
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return result;
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
    }
  }
}
