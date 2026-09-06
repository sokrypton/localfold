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
   * @param options `onBlock(index)` fires when a block is ENCODED and is
   *   awaited so a caller may yield; `onBlockDone(completed, total)` fires when
   *   the DEVICE has finished one and is what a progress bar should read.
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
    // 🔴 BOTH KNOBS ARE f16, AND THE SECOND ONE IS PRICED AGAINST THE SAMPLER
    // RATHER THAN AGAINST A TENSOR NORM. The transition's staged tiles and the
    // triangle projection's eight vec4 of accumulators are separate kernels and
    // separate knobs. Against the native model at 40 residues, timed by
    // bench-esmfold2-trunk.js at 150:
    //
    // | staged : accumulate | relRMS | 150 tokens | 300 |
    // |---|---|---|---|
    // | f32 : f32 | 1.10e-6 | 1.000x | 1.000 |
    // | f16 : f32 | 5.46e-4 | 1.074 | **0.951** |
    // | **f16 : f16** | **2.68e-3** | **1.279** | **1.195** |
    //
    // 🔴 AND THE OLD DEFAULT WAS A LOSS AT 300 TOKENS. `f16:f32` staged the
    // transition's tiles narrow and left its arithmetic wide, and after the
    // tiling fix of this session that tile is half the size it was - so
    // narrowing what is no longer the bottleneck costs the `f32()` at each read
    // and buys nothing. It measured 1.306x at 300 tokens before that fix and
    // 0.951 after. A precision trade priced before a tiling change is not a
    // trade priced after it, and this is the second time that sentence has been
    // true in this file.
    //
    // The accumulator carries 96% of the error, and for a long time that was
    // the reason to leave it alone. What settled it was measuring the
    // STRUCTURE instead: folding the same 150-residue sequence twice at the
    // same seed, changing only this, and superposing the two answers.
    //
    // | what changed | how far the structure moved |
    // |---|---|
    // | the SAMPLER'S SEED, nothing else | **6.32 A** |
    // | f32:f32 -> f16:f32 | 0.008-0.009 |
    // | f32:f32 -> f16:f16 | **0.034-0.041** |
    //
    // 0.04 A against a 6.3 A spread is a factor of 160. This is a stochastic
    // sampler: a change smaller than its own draw is not a cost, it is below
    // the resolution of the thing being predicted. Contact precision and recall
    // are identical across all three arms, to three decimals.
    //
    // 🔴 AND THE CHECKER STILL SEPARATES THEM, which is the other half of
    // taking this. tools/gpu/check-esmfold2-trunk-gpu.js holds each arm to the
    // bound its own arithmetic implies - 2e-4, 1e-3 and 4e-3 - rather than
    // raising one to cover all three, because a single loose bound would stop
    // the f32 path being checked at all.
    const accumulatePrecision = this.options.accumulatePrecision
      ?? (hasF16 ? "f16" : "f32");
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
        // 🔴 WHEN THE DEVICE REACHES THIS BLOCK, REPORTED WITHOUT WAITING FOR
        // IT. `onBlock` below fires when a block is ENCODED, and sixteen of
        // those happen in the time the GPU takes over one - so a bar driven by
        // it sprints to the end of the submission window and then sits still,
        // which is what "the bar is not smooth" is. `onSubmittedWorkDone`
        // resolves once everything submitted so far has finished, so one taken
        // HERE settles exactly when this block is done.
        //
        // 🔴 AND IT IS NOT AWAITED, WHICH IS THE WHOLE TRICK. The loop carries
        // on encoding and the pipelining that makes this stack fast is
        // untouched; awaiting per block costs 14% of the trunk, measured at 150
        // residues - 7294 ms against 6406 - for a bar that this gets for
        // nothing. They resolve in submission order, so the count cannot go
        // backwards. AF3's pairformer has done this since it had a status line,
        // and it took it from AF2's evoformer stack.
        const submitted = index + 1;
        void this.device.queue.onSubmittedWorkDone()
          .then(() => options.onBlockDone?.(submitted, blocks.length));
        if ((index + 1) % submissionWindow === 0 || index === blocks.length - 1) {
          await this.device.queue.onSubmittedWorkDone();
        }
        // 🔴 AWAITED, SO A CALLER CAN YIELD. Every await above resolves from a
        // GPU promise, which is a microtask - so a page that only moved a
        // progress bar here would write it and never paint it.
        await options.onBlock?.(index);
      }
      await validation.settle();

      // ...allocated after the scratch is released, not beside it. See the
      // note in src/af3/pairformer-block-webgpu.js: this allocator does not
      // pool, so releasing DESTROYS and the peak actually moves.
      for (const allocation of scratch) allocation.release();
      biasBuffer?.release();
      if (options.readback === false) {
        return { pair: undefined,
                 precision: { staged: stagedPrecision, accumulate: accumulatePrecision },
                 elapsedMilliseconds: performance.now() - start,
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
        // 🔴 WHAT IT ACTUALLY RAN, NOT WHAT WAS ASKED FOR. A caller that passes
        // nothing gets the stack's own choice, and a checker holding that arm
        // to a bound has to know which arithmetic it got - the alternative is
        // naming the defaults in the checker, which makes it agree with itself
        // when the defaults move.
        precision: { staged: stagedPrecision, accumulate: accumulatePrecision },
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
