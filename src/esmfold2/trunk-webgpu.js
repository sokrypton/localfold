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
 * 🔴 THIS SAID "NO RESIDENT WEIGHTS, DELIBERATELY" AND THE REASONING WAS WRONG.
 * It read: the whole trunk is 37.9M parameters, 144 MiB in f32 against the 567
 * an AF3 trunk keeps resident for its single track alone, so the upload is
 * small and the block loop is short. That is true of the UPLOAD and says
 * nothing about the PACK, which is where the time was: packing 24 blocks into
 * the interleaved layout the matrix projection reads costs **232 ms of host
 * time** at 256 channels, measured, and this ran it on every pass of every
 * recycle over weights that never change. A trunk pass was 275.7 ms of GPU in
 * 701.1 of wall; with residency it is 275.8 in **501.3**.
 *
 * A hypothesis written in the same voice as a measurement, and then cited -
 * which is the failure docs/A100.md exists to prevent and has now recorded
 * about itself twice.
 */
import {
  deviceMatrixConfig, deviceTuning, halfPrecisionAvailable,
} from "../runtime/device-profile.js";
import { stagedMatrixBlock } from "../runtime/matrix-linear.js";
import { residentWeightBuffer } from "../runtime/resident.js";
import { residencyAllowed } from "../runtime/device-memory.js";
import {
  allocateTransitionSplit, packTransitionWeights, TRANSITION_SPLIT_MIN_CHANNELS,
  TRANSITION_ORDER,
} from "../af3/transition-webgpu.js";
import {
  allocateTriangleProjectMatrix, TRIANGLE_PROJECT_MATRIX_MIN_CHANNELS,
} from "../triangle/project-matrix.js";
import { packWeights as packTriangleWeights } from "../triangle/weights.js";
import { residentPairTrackOnDevice } from "../af3/pair-track-device-weights.js";
import { residentPackedOnDevice } from "../af3/device-weights.js";
import { af3TriangleWeights } from "../af3/triangle-webgpu.js";
import { DeferredValidation } from "../runtime/validation.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { storageBytes } from "../runtime/storage.js";
import {
  pairScratchCount, UNPACKED_PAIR_SCRATCH,
  compilePairTrack, encodePairTrack, packPairTrackWeights,
} from "../af3/pair-track-gpu.js";

/** ESMFold2-Experimental-Fast's `d_pair`. */
export const PAIR_CHANNELS = 256;

export class Esmfold2TrunkGpu {
  constructor(device, options = {}) {
    this.device = device;
    this.options = options;
    // 🔴 THE SAME DEFAULT AF3's PAIRFORMER TAKES, and the same escape: a device
    // with a memory budget that refuses an allocation drops back to uploading
    // per pass. See residencyAllowed in src/runtime/device-memory.js.
    this.residentWeights = (options.residentWeights ?? true) && residencyAllowed(device);
    // 🔴 THE ARM. A device path with no host path beside it is a path nothing
    // can be compared against - `check-esmfold2-trunk-pack.js` runs the trunk
    // both ways over one input and holds them to ZERO differing elements,
    // because the decode is bit-identical and the layouts are the same.
    this.devicePairTrack = options.devicePairTrack ?? true;
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

    const hasF16 = halfPrecisionAvailable(this.device);
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
    // 🔴 THE SPLIT TRANSITION IS WHERE THIS TRUNK'S TIME IS. `pair-transition`
    // is 419.88 ms of a 743.8 ms trunk at 300 tokens - 56.5%, more than the
    // other five kernels together - and at this track's 256 channels the fused
    // kernel loses to the split by 2.89x. See src/af3/transition-webgpu.js.
    const transitionFactor = options.transitionFactor ?? 4;
    const tuning = deviceTuning(this.device);
    const splitConfig = deviceMatrixConfig(this.device, { element: "f16" });
    // ...and the block the staged GEMMs share, from the profile; see
    // `stagedMatrixBlock` in src/runtime/device-profile.js.
    const block = {
      ...stagedMatrixBlock(tuning.stagedMatrixBlock),
      // A request, not a decision: each kernel asks directWeightsAllowed
      // whether its own contracted extent and weight buffer can take it.
      directWeights: tuning.stagedMatrixDirectWeights === true,
      prefetch: tuning.stagedMatrixPrefetch === true,
    };
    // 🔴 THE ACCUMULATOR MAY NARROW WHERE K IS A CHANNEL COUNT AND NOT WHERE IT
    // IS THE PROTEIN'S LENGTH, and that rule was bisected out of a fold that
    // came back with every coordinate NaN. `stagedMatrixResult: "f16"` halves
    // the registers the results cost and is worth 1.19x - and applied to all
    // five staged GEMMs it destroys the structure. Folding 1QYS one kernel
    // group at a time: the transition's two projections are fine, the
    // triangle's three projections are fine, and `tri.contract` alone is 2178
    // NaN coordinates. Its K is the protein's LENGTH and its operands are two
    // gated projections multiplied against each other, so its partial sums are
    // the only ones in the track not bounded by a width - which is the same
    // distinction the outer product mean's f16 contraction draws.
    //
    // So the contraction keeps the device config's own result type whatever the
    // knob says, and `contractResult` is how that reaches it.
    const narrowed = (tuning.stagedMatrixResult === null
      || tuning.stagedMatrixResult === undefined)
      ? {} : { result: tuning.stagedMatrixResult,
               contractResult: splitConfig?.resultComponentType };
    // 🔴 ITS OWN KNOB, not derived from the transition's - see the note on
    // projectMatrixConfig in src/af3/pairformer-block-webgpu.js - and its own
    // width rule, which is about precision rather than memory.
    const triangleProjectMatrix = tuning.triangleProjectMatrix === true && splitConfig !== null
      && channels >= (tuning.triangleProjectMatrixMinChannels
        ?? TRIANGLE_PROJECT_MATRIX_MIN_CHANNELS)
      ? { result: splitConfig.resultComponentType, matrixElement: splitConfig.componentType,
          tile: { M: splitConfig.M, N: splitConfig.N, K: splitConfig.K }, ...block,
          ...narrowed }
      : false;
    const pairTransitionSplit = tuning.pairTransitionSplit === true && splitConfig !== null
      && channels >= (tuning.pairTransitionSplitMinChannels
        ?? TRANSITION_SPLIT_MIN_CHANNELS)
      ? { result: splitConfig.resultComponentType, matrixElement: splitConfig.componentType,
          tile: { M: splitConfig.M, N: splitConfig.N, K: splitConfig.K }, ...block,
          ...narrowed }
      : false;
    const pipelines = await compilePairTrack(this.pipelines, {
      n, sample: blocks[0], epsilon, variance, base, channels,
      // The grid attention is what needs a dialect; without it there is no
      // transposed bias to have a convention about.
      dialect: { swapTransposedBias: false },
      gridAttention,
      transitionFactor,
      pairTransitionSplit,
      pairTransitionChunkBytes: tuning.pairTransitionChunkBytes,
      triangleProjectMatrix,
      maxComputeWorkgroupStorageSize: this.device.limits.maxComputeWorkgroupStorageSize,
      maxStorageBufferBindingSize: this.device.limits.maxStorageBufferBindingSize,
      minStorageBufferOffsetAlignment: this.device.limits.minStorageBufferOffsetAlignment,
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
      // ...four of them without the grid attention, not five; see
      // pairScratchCount in src/af3/pair-track-gpu.js.
      for (let index = 0; index < pairScratchCount(gridAttention); index += 1) {
        scratch.push(keep(this.allocator.allocate(
          `esmfold2-trunk.scratch${index}`,
          storageBytes(pairs * channels, UNPACKED_PAIR_SCRATCH[index]), storage)));
      }
      // ...and the split transition's two intermediates, sized for one chunk of
      // rows. The widened one is 369 MiB at 300 tokens unchunked, which is the
      // tensor the fused kernel existed to avoid; see transitionSplitChunkRows.
      const transitionSplit = pipelines.transitionSplit === undefined ? undefined
        : allocateTransitionSplit(this.allocator, {
          rows: pairs, channels, factor: transitionFactor,
          chunkRows: pipelines.transitionSplit.chunkRows,
          offsets: packTransitionWeights(blocks[0].pairTransition).offsets,
          label: "esmfold2-trunk.transition",
        }, keep);
      const projectMatrix = pipelines.projectMatrix === undefined ? undefined
        : allocateTriangleProjectMatrix(this.allocator, {
          rows: pairs, cZ: channels, cHidden: channels,
          offsets: packTriangleWeights(
            af3TriangleWeights(blocks[0].triangleMultiplicationOutgoing, channels),
            weightPrecision,
            { abLayout: "interleaved", zgLayout: "transposed",
              cHidden: channels, cZ: channels }).offsets,
          label: "esmfold2-trunk.tri-project",
        }, keep);
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
        // eslint-disable-next-line no-await-in-loop
        await this.#encodeBlock({
          block: blocks[index], n, channels, pipelines, storage, pending, weightPrecision,
          pair, pairMask, scratch, gridAttention, gridHeads, biasBuffer, transitionSplit,
          projectMatrix,
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
  async #encodeBlock(context) {
    const { block, n, channels, pipelines, storage, pending, weightPrecision } = context;
    const { pair, pairMask, scratch, gridAttention, gridHeads, biasBuffer,
            transitionSplit, projectMatrix } = context;
    const upload = (label, data) => {
      const allocation = this.allocator.upload(label, data, storage);
      pending.push(allocation);
      return allocation;
    };
    // 🔴 THE LAYOUT MUST MATCH THE KERNEL THAT WAS COMPILED. `projectMatrix`
    // exists exactly when compilePairTrack chose the interleaved projection,
    // so it is the one thing that decides both.
    //
    // 🔴 AND IT IS PACKED ON DEMAND AND UPLOADED ONCE, EVER, which AF3's
    // pairformer has done for a long time and this trunk had not. Packing 24
    // blocks costs **232 ms of host time** on the interleaved layout - measured
    // on this box, at 256 channels - and this ran it on every pass of every
    // recycle over weights that never change. A trunk pass is 275.7 ms of GPU
    // in 701.1 of wall; that packing was more than half of the difference, and
    // the ~168 MB of writeBuffer behind it was the rest.
    //
    // 🔴 RESIDENT IS A TRADE AND THE BUDGET ANSWERS IT, not a guess made in
    // advance - see the long note in src/af3/pairformer-block-webgpu.js. A
    // device with no budget set never takes this path.
    const resident = this.residentWeights
      ? (label, pack, variant) => ({
        buffer: residentWeightBuffer(this.device, block, label, pack, variant),
      })
      : (label, pack) => upload(label, pack());
    // 🔴 PACKED LAZILY AND NOT HELD, because residentWeightBuffer calls pack()
    // only on a MISS: after a block's first encode nothing reads these arrays
    // again, and a WeakMap keeping them alive cost AF3's page 350 MiB of heap
    // for nothing.
    let packedPair;
    const want = { transition: true, triangles: true, grids: gridAttention };
    const packedFor = () => (packedPair ??= packPairTrackWeights(
      block, channels, weightPrecision, gridAttention,
      projectMatrix === undefined ? "blocked" : "interleaved", want));
    // The precision and the layout are the cache VARIANT rather than part of
    // the label, so a device-memory breakdown still reads one row per tensor
    // and a run that switched either cannot be handed the other's buffer.
    const variant = `${weightPrecision}:${projectMatrix === undefined ? "b" : "i"}`;
    // 🔴 THE SAME DEVICE DECODE AF3's PAIRFORMER TAKES, on the same packer.
    // This trunk shares `packPairTrackWeights` with three AF3 stacks and was
    // the one caller still building all five on the HOST - measured as 828 ms
    // in `trunk 0` against 32 for the same block warm, out of a 2.7 s first
    // fold. `residentPairTrackOnDevice` is that decode, and `want` is what it
    // could not take, because the packer returns all five from one call and a
    // caller that stopped binding a buffer was still paying to build it.
    const onDevice = await residentPairTrackOnDevice(this.device, block, {
      channels, pairWeightPrecision: weightPrecision,
      abLayout: projectMatrix === undefined ? "blocked" : "interleaved",
      resident: this.residentWeights && this.devicePairTrack,
    });
    want.triangles = onDevice.want.triangles;
    want.grids = gridAttention && onDevice.want.grids;
    // 🔴 AND THE TRANSITION, which is the last host packer this trunk had:
    // `residentPackStats` measures it at 159 ms of a 1.75 s ESMFold2 fold, over
    // 24 blocks. It lays four tensors end to end in TRANSITION_ORDER and
    // reshapes nothing, so it is the contiguous case the decoder started with.
    const transitionOnDevice = this.residentWeights && this.devicePairTrack
      ? await residentPackedOnDevice(this.device, {
          key: block.pairTransition, label: "w.pair-transition",
          order: TRANSITION_ORDER, weights: block.pairTransition, variant,
          destination: weightPrecision === "f16" ? "f16" : "f32",
        })
      : undefined;
    want.transition = transitionOnDevice === undefined;
    const pairTransitionOnDevice = transitionOnDevice === undefined
      ? undefined : { buffer: transitionOnDevice };
    const weights = {
      outgoing: onDevice.buffers.outgoing
        ?? resident("w.tri.out", () => packedFor().outgoing, variant),
      incoming: onDevice.buffers.incoming
        ?? resident("w.tri.in", () => packedFor().incoming, variant),
      transition: pairTransitionOnDevice ?? resident(
        "w.pair-transition", () => packedFor().transition, variant),
      ...(gridAttention ? {
        grid1: onDevice.buffers.grid1 ?? resident("w.grid1", () => packedFor().grid1, variant),
        grid2: onDevice.buffers.grid2 ?? resident("w.grid2", () => packedFor().grid2, variant),
      } : {}),
    };

    const encoder = this.device.createCommandEncoder({ label: "esmfold2-trunk-block" });
    // 🔴 ONE PASS A BLOCK, NOT ONE A DISPATCH. WebGPU orders dispatches inside
    // a compute pass and makes each one's writes visible to the next - which is
    // what AF2's evoformer stack has always relied on. A block is about thirty
    // dispatches and an OpenDDE trunk is 1723 of them a pass.
    // `batchComputePasses` off restores one pass a dispatch, which is what
    // profile.js needs to attribute anything; see profileDevice.
    const batchPasses = deviceTuning(this.device).batchComputePasses !== false;
    let openPass = null;
    const endPass = () => { if (openPass !== null) { openPass.end(); openPass = null; } };
    const run = (label, pipeline, buffers, x, y = 1, z = 1) => {
      const pass = batchPasses
        ? (openPass ??= encoder.beginComputePass({ label: `${label.split(".")[0]}.block` }))
        : encoder.beginComputePass({ label });
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
      if (!batchPasses) pass.end();
    };
    encodePairTrack({
      run, pipelines, n, channels, pair, pairMask, scratch, weights,
      gridAttention, gridHeads, biasBuffer, transitionSplit, projectMatrix,
    });
    endPass();
    this.device.queue.submit([encoder.finish()]);
  }
}
