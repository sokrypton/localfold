/**
 * ESMFold2's folding trunk on the GPU: AF3's pair track, minus two of its five.
 *
 *     pair += triangle_multiplication_outgoing
 *     pair += triangle_multiplication_incoming
 *     pair += transition
 *
 * 🔴 IT IS AF3's PAIRFORMER BLOCK WITH THE GRID ATTENTIONS AND THE SINGLE TRACK
 * REMOVED, AND THAT IS A MEASUREMENT RATHER THAN A READING.
 * tools/check-esmfold2-trunk.js runs the CPU reference's three pieces 24 times
 * against the values the native model recorded at each of its four recycles:
 * relRMS 1.4e-6 against a 2e-4 bound. So no new arithmetic is needed here, only
 * a different sequence of the same passes.
 *
 * 🔴 AND THE ATTENTION IS SKIPPED, NOT ZEROED. An attention whose output
 * projection is zero adds zero, so a zeroed AF3 block already IS this block and
 * needs no code - which is what tools/esmc/esmfold2_trunk_weights.py's
 * zero_attention exists to build. But `grid.attend` is the largest kernel in an
 * AF3 trunk, 34.6% of its GPU time at 700 tokens, and this trunk runs 24 blocks
 * four times over: that is 96 rounds of computing zero to add. The zeroed arm
 * survives as the thing this is checked bit-identical against, which is the
 * only way to say the skipping skipped nothing else.
 *
 * 🔴 NO RESIDENT WEIGHTS, DELIBERATELY. The whole trunk is 37.9M parameters -
 * 144 MiB in f32, against the 567 MiB an AF3 trunk keeps resident for its
 * single track alone - so the upload is small and the block loop is short.
 * Residency is a trade that pays when the same weights are re-read across
 * recycles; if this trunk's four loops make it pay, it is one call to
 * residentWeightBuffer, the way src/af3/pairformer-block-webgpu.js does it.
 */
import { DeferredValidation } from "../runtime/validation.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { storageBytes } from "../runtime/storage.js";
import {
  PAIR_SCRATCH_COUNT, UNPACKED_PAIR_SCRATCH,
  compilePairTrack, encodePairTrack, packPairTrackWeights,
} from "../af3/pair-track-gpu.js";

/** ESMFold2-Experimental-Fast's `d_pair`. */
export const PAIR_CHANNELS = 256;

export class Esmfold2TrunkGpu {
  constructor(device, options = {}) {
    this.device = device;
    this.options = options;
    this.pipelines = pipelineCacheForDevice(device);
    this.allocator = options.allocator ?? new GpuBufferAllocator(device);
  }

  /**
   * One pass of the trunk: `blocks.length` blocks over one pair representation.
   *
   * 🔴 THIS IS ONE LOOP, NOT THE RECURRENCE. ESMFold2 runs the trunk
   * `num_loops + 1` times with `z = z_init + pair_loop_proj(z)` between them -
   * see tools/esmc/dump-esmfold2-trunk.py, which records that the config's
   * `num_loops: 3` means four iterations. The projection and the recurrence
   * belong to the caller; what a trunk pass IS, is this.
   *
   * @param {{pair: Float32Array, pairMask: Float32Array}} state
   * @param {object[]} blocks each with the three sub-modules' weights
   */
  async run(state, blocks, options = {}) {
    const n = options.n ?? Math.round(Math.sqrt(state.pairMask.length));
    const pairs = n * n;
    const channels = options.channels ?? PAIR_CHANNELS;
    if (state.buffer === undefined && state.pair.length !== pairs * channels) {
      throw new Error(`pair has ${state.pair.length} elements; expected ${pairs * channels}`);
    }
    if (state.buffer !== undefined && options.n === undefined) {
      throw new Error("a borrowed pair buffer carries no shape; pass options.n");
    }
    if (blocks.length === 0) throw new Error("a trunk with no blocks");
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    const pairBytes = pairs * channels * 4;

    const hasF16 = this.device.features?.has("shader-f16") === true;
    const stagedPrecision = this.options.stagedPrecision ?? (hasF16 ? "f16" : "f32");
    const weightPrecision = this.options.weightPrecision ?? "f32";
    // 🔴 THE TRIANGLE'S ACCUMULATORS STAY f32 HERE, WHERE AF3's TRUNK NARROWS
    // THEM, AND THE REASON IS THE RATIO RATHER THAN THE ERROR. Separating the
    // two f16 knobs - the transition's staged tiles and the triangle
    // projection's eight vec4 of accumulators - prices them very differently.
    // Error against the native model at 40 residues, time swept over lengths
    // (tools/gpu/check-esmfold2-trunk-gpu.js and bench-esmfold2-trunk.js):
    //
    // | staged : accumulate | relRMS | 40 tok | 150 | 300 |
    // |---|---|---|---|---|
    // | f32 : f32 | **1.10e-6** | 1.000x | 1.000 | 1.000 |
    // | f16 : f32 | 5.46e-4 | 1.163 | 1.240 | **1.306** |
    // | f32 : f16 | 2.62e-3 | 1.102 | 1.088 | 1.151 |
    // | f16 : f16 | 2.68e-3 | 1.348 | 1.466 | 1.480 |
    //
    // The accumulator carries 96% of the error and returns the smaller half of
    // the speedup, at every length. So this takes 1.31x of the 1.48x available
    // for a fifth of the error. AF3's trunk answers differently because it was
    // priced on that kernel alone (1.55x on bench-triangle-project at 118
    // tokens) rather than against the other knob; here the two were separated
    // because 24 blocks four times over amplifies whatever they round.
    const accumulatePrecision = this.options.accumulatePrecision ?? "f32";
    // 🔴 THE ATTENTION IS AN AXIS OF THE CACHE KEY, not just of the dispatch.
    // A pipeline cache shared with an AF3 fold in the same page holds shaders
    // compiled for the same n and channels; without this the two stacks would
    // collide on keys that mean different sequences of passes.
    // 🔴 THE ZEROED ARM IS A RUNTIME OPTION, NOT A SECOND STACK. The only
    // honest way to say the skipping skipped nothing else is to run the same
    // weights, the same shapes and the same block loop with the attention
    // present-and-zero, and compare - and a second driver written to do that
    // would be checking itself. Nothing ships with this on; see
    // tools/gpu/check-esmfold2-trunk-gpu.js, which is its only caller.
    const gridAttention = options.gridAttention ?? false;
    const base = `esmfold2-trunk:${n}:${channels}:${epsilon}:${variance}:`
      + `${gridAttention ? "zeroed-grid" : "no-grid"}`;
    const pipelines = await compilePairTrack(this.pipelines, {
      n, sample: blocks[0], epsilon, variance, base, channels,
      // The grid attention is what needs a dialect; without it there is no
      // transposed bias to have a convention about.
      dialect: { swapTransposedBias: false },
      gridAttention,
      transitionFactor: options.transitionFactor ?? 4,
      stagedPrecision, weightPrecision, accumulatePrecision,
    });

    try {
      // 🔴 A CALLER MAY OWN THE PAIR BUFFER, AND A FOLD DOES. Four loops that
      // each upload the pair and read it back is 736 MB of traffic at 300
      // tokens for a tensor that never leaves the device between them - and the
      // recycle projection in between is itself a GPU pass. `state.buffer` is
      // that path; `state.pair` is the standalone one a checker wants.
      const borrowed = state.buffer !== undefined;
      const pair = borrowed ? state.buffer : keep(this.allocator.upload(
        "esmfold2-trunk.pair", state.pair, storage | GPUBufferUsage.COPY_SRC));
      const pairMask = state.maskBuffer ?? keep(this.allocator.upload(
        "esmfold2-trunk.pair-mask", state.pairMask, storage));
      const scratch = [];
      for (let index = 0; index < PAIR_SCRATCH_COUNT; index += 1) {
        scratch.push(keep(this.allocator.allocate(
          `esmfold2-trunk.scratch${index}`,
          storageBytes(pairs * channels, UNPACKED_PAIR_SCRATCH[index]), storage)));
      }
      const gridHeads = gridAttention ? blocks[0].pairAttention1.heads : 0;
      const biasBuffer = gridAttention
        ? keep(this.allocator.allocate("esmfold2-trunk.bias", gridHeads * pairs * 4, storage))
        : undefined;

      const submissionWindow = options.submissionWindow ?? 16;
      const validation = new DeferredValidation(this.device, "ESMFold2 trunk");
      const start = performance.now();
      for (let index = 0; index < blocks.length; index += 1) {
        const pending = [];
        validation.begin();
        this.#encodeBlock({
          block: blocks[index], n, channels, pipelines, storage, pending, weightPrecision,
          pair, pairMask, scratch, gridAttention, gridHeads, biasBuffer,
        });
        validation.end(`block ${index}`);
        for (let at = pending.length - 1; at >= 0; at -= 1) pending[at].release();
        if ((index + 1) % submissionWindow === 0 || index === blocks.length - 1) {
          await this.device.queue.onSubmittedWorkDone();
        }
        await options.onBlock?.(index);
      }
      await validation.settle();

      // ...allocated after the scratch is released, not beside it. See the
      // note in src/af3/pairformer-block-webgpu.js: this allocator does not
      // pool, so releasing DESTROYS and the peak actually moves.
      for (const allocation of scratch) allocation.release();
      biasBuffer?.release();
      if (options.readback === false) {
        return { pair: undefined, elapsedMilliseconds: performance.now() - start,
                 memory: this.allocator.snapshot() };
      }
      const readback = keep(this.allocator.allocate(
        "esmfold2-trunk.readback", pairBytes,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      const encoder = this.device.createCommandEncoder({ label: "esmfold2-trunk.readback" });
      encoder.copyBufferToBuffer(pair.buffer, 0, readback.buffer, 0, pairBytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();

      return {
        pair: out,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }

  /** One block, submitted as one command buffer. */
  #encodeBlock(context) {
    const { block, n, channels, pipelines, storage, pending, weightPrecision } = context;
    const { pair, pairMask, scratch, gridAttention, gridHeads, biasBuffer } = context;
    const upload = (label, data) => {
      const allocation = this.allocator.upload(label, data, storage);
      pending.push(allocation);
      return allocation;
    };
    const packed = packPairTrackWeights(block, channels, weightPrecision, gridAttention);
    const weights = {
      outgoing: upload("w.tri.out", packed.outgoing),
      incoming: upload("w.tri.in", packed.incoming),
      transition: upload("w.pair-transition", packed.transition),
      ...(gridAttention ? {
        grid1: upload("w.grid1", packed.grid1),
        grid2: upload("w.grid2", packed.grid2),
      } : {}),
    };

    const encoder = this.device.createCommandEncoder({ label: "esmfold2-trunk-block" });
    const run = (label, pipeline, buffers, x, y = 1, z = 1) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((allocation, binding) => ({
          binding,
          resource: allocation.byteOffset === undefined
            ? { buffer: allocation.buffer }
            : { buffer: allocation.buffer,
                offset: allocation.byteOffset, size: allocation.byteSize },
        })),
      }));
      pass.dispatchWorkgroups(x, y, z);
      pass.end();
    };
    encodePairTrack({
      run, pipelines, n, channels, pair, pairMask, scratch, weights,
      gridAttention, gridHeads, biasBuffer,
    });
    this.device.queue.submit([encoder.finish()]);
  }
}
