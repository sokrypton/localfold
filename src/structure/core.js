import {
  InvariantPointAttentionGpu,

} from "./ipa.js";
import {
  StructurePostAttentionGpu,

} from "./iteration.js";
import { throwIfAborted, withAbort } from "../runtime/abort.js";

export class StructureCoreGpu {
  device;
  constructor(device) { this.device = device; }

  async run(input) {
    throwIfAborted(input.signal);
    let activations = input.activations;
    let affine = input.affine;
    const start = performance.now();
    const iterations = input.iterations ?? 8;
    const ipa = new InvariantPointAttentionGpu(this.device);
    const post = new StructurePostAttentionGpu(this.device);
    const geometry = {
      pair: input.pair,
      mask: input.mask,
      length: input.length,
      channels: input.channels,
      pairChannels: input.pairChannels,
      heads: 12,
      scalarQk: 16,
      scalarV: 16,
      pointQk: 4,
      pointV: 8,
      weights: input.ipaWeights,
    };
    // 🔴 THE PAIR REPRESENTATION IS UPLOADED ONCE, NOT EIGHT TIMES.
    //
    // Every iteration reads the same pair track and normalises it with the same
    // weights, so eight of them used to push L*L*128 floats across the bus and
    // layer-normalise them to produce a buffer identical to the last. At 221
    // residues that is 25 MiB a time, 200 MiB a pass, plus seven normalisations
    // whose only output was a copy. Prepared once here, it is resident for all
    // eight; only the activations and the frames still travel per iteration.
    const prepared = await withAbort(ipa.prepare(geometry), input.signal);
    const postPrepared = await withAbort(post.prepare({
      length: input.length, channels: input.channels, weights: input.postAttentionWeights,
    }), input.signal);
    const allocator = ipa.allocator;
    const scratch = ipa.allocateScratch(geometry, prepared);
    const owned = [];
    const keep = (value) => { owned.push(value); return value; };
    try {
      // 🔴 THE WHOLE LOOP IS ONE COMMAND BUFFER AND ONE COMPUTE PASS.
      //
      // It used to be sixteen submissions a recycle: the attention submitted,
      // fenced and mapped its output back to the CPU, and the post-attention
      // update uploaded that same array again and did the same. Twenty-four
      // round trips per recycle, for work that takes microseconds at this size,
      // plus one compute pass per dispatch - 144 pass boundaries.
      //
      // 🔴 WHICH IS ONLY SAFE BECAUSE NOTHING INSIDE THE LOOP WRITES A BUFFER
      // THROUGH THE QUEUE. queue.writeBuffer is ordered against the queue, not
      // against the encoder, so an upload here would land BEFORE the dispatches
      // still reading the buffer it wrote. Everything that varies per iteration
      // is now GPU-resident and everything else was hoisted into the two
      // prepare() calls above; that is what the hoisting is for, not tidiness.
      //
      // Dispatches recorded into one pass execute in order with a barrier
      // between them, which is what lets both operations share scratch and lets
      // iteration N+1 read what iteration N wrote.
      const elements = input.length * input.channels;
      const storage = GPUBufferUsage.STORAGE;
      const readable = storage | GPUBufferUsage.COPY_SRC;
      // Ping-pong: an operation cannot read and write one buffer, so the state
      // alternates and the last iteration decides which half holds the answer.
      const actBuffers = [
        keep(allocator.upload("structure.act-a", activations, readable)),
        keep(allocator.allocate("structure.act-b", elements * 4, readable)),
      ];
      const affineBuffers = [
        keep(allocator.upload("structure.affine-a", affine, readable)),
        keep(allocator.allocate("structure.affine-b", input.length * 7 * 4, readable)),
      ];
      const attentionOutput = keep(allocator.allocate("structure.attention-update", elements * 4, storage));

      const encoder = this.device.createCommandEncoder({ label: "structure-core" });
      this.device.pushErrorScope("validation");
      const compute = encoder.beginComputePass({ label: "structure-iterations" });
      let current = 0;
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        ipa.encode(compute, geometry, prepared, scratch,
          actBuffers[current], affineBuffers[current], attentionOutput);
        post.encode(compute, { length: input.length, channels: input.channels }, postPrepared,
          actBuffers[current], attentionOutput, affineBuffers[current],
          actBuffers[1 - current], affineBuffers[1 - current]);
        current = 1 - current;
      }
      compute.end();

      const actReadback = keep(allocator.allocate("structure.act-readback", elements * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      const affineReadback = keep(allocator.allocate("structure.affine-readback", input.length * 7 * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      encoder.copyBufferToBuffer(actBuffers[current].buffer, 0, actReadback.buffer, 0, elements * 4);
      encoder.copyBufferToBuffer(affineBuffers[current].buffer, 0, affineReadback.buffer, 0, input.length * 7 * 4);
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      throwIfAborted(input.signal);
      await withAbort(Promise.all([
        actReadback.buffer.mapAsync(GPUMapMode.READ),
        affineReadback.buffer.mapAsync(GPUMapMode.READ),
      ]), input.signal);
      activations = new Float32Array(actReadback.buffer.getMappedRange().slice(0));
      affine = new Float32Array(affineReadback.buffer.getMappedRange().slice(0));
      actReadback.buffer.unmap(); affineReadback.buffer.unmap();
      // ...REPORTED ONCE THE WORK IS DONE, not per iteration. There is no longer
      // a per-iteration boundary to ride on, and inventing one by fencing inside
      // the loop would give the progress bar back the cost this removed.
      input.onIteration?.(iterations, iterations);
    } finally {
      for (let index = owned.length - 1; index >= 0; index -= 1) owned[index].release();
      scratch.release();
      postPrepared.release();
      prepared.release();
    }
    throwIfAborted(input.signal);
    return { activations, affine, elapsedMilliseconds: performance.now() - start };
  }
}
