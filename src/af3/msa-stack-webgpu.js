/**
 * AF3's MSA stack, resident on the GPU. Four blocks, ahead of the pairformer's
 * forty-eight.
 *
 *     pair += outer_product_mean(msa)          <- the msa AS IT ARRIVED
 *     msa  += msa_attention(msa, pair)         <- the pair the line above changed
 *     msa  += transition(msa)
 *     ...then the same five pair updates the pairformer runs
 *
 * 🔴 THE OUTER PRODUCT MEAN READS THE MSA BEFORE THE TWO MSA UPDATES, and the
 * MSA attention reads the pair AFTER the outer product mean has changed it. So
 * the two tracks are interleaved, not sequential: neither "do the MSA then the
 * pair" nor "do the pair then the MSA" is what AF3 does, and both run.
 *
 * The five pair updates are shared with the pairformer stack; see
 * src/af3/pair-track-gpu.js.
 */
import { deviceTuning, shapedKnob } from "../runtime/device-profile.js";
import { resolveGridAttendMatrix } from "./grid-attention-matrix.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { residentWeightBuffer } from "../runtime/resident.js";
import { residencyAllowed } from "../runtime/device-memory.js";
import { storageBytes } from "../runtime/storage.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import {
  GRID_WIDTH, PAIR_SCRATCH_COUNT, UNPACKED_PAIR_SCRATCH, compilePairTrack, createAddShader,
  encodePairTrack, packPairTrackWeights,
} from "./pair-track-gpu.js";
import { residentPairTrackOnDevice } from "./pair-track-device-weights.js";
import { residentPackedOnDevice } from "./device-weights.js";
import {
  createOuterProductMeanShaders, packOuterProductMeanWeights,
} from "./outer-product-mean-webgpu.js";
import { createMsaAttentionShaders, packMsaAttentionWeights } from "./msa-attention-webgpu.js";
import { allocateGridProjectMatrix, gridProjectMatrixConfig }
  from "./grid-project-matrix.js";
import {
  allocateTransitionSplit, createTransitionShader, packTransitionWeights,
  splitTransitionConfig, transitionRowTile, TRANSITION_ORDER,
} from "./transition-webgpu.js";

export class Af3MsaStackGpu {
  constructor(device, options = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
    // The same default and the same escape the pairformer takes.
    this.residentWeights = (options.residentWeights ?? true) && residencyAllowed(device);
  }

  /**
   * @param {{pair: Float32Array, msa: Float32Array, pairMask: Float32Array,
   *          msaMask: Float32Array, tokens: number, sequences: number}} state
   * @param {object[]} blocks one weight bundle per block
   * @param {{swapTransposedBias: boolean}} dialect
   * @param {{epsilon?: number, variance?: "fast"|"two-pass",
   *          msaChannels?: number, onBlock?: (index: number) => void}} options
   */
  async run(state, blocks, dialect, options = {}) {
    const n = state.tokens;
    const sequences = state.sequences;
    const pairs = n * n;
    // 🔴 BOTH WIDTHS ARE THE BLOCK'S. OpenDDE's MSA track is 128 channels and
    // its pair track 384, against AlphaFold 3's 64 and 128, and a declared
    // width loads either bundle without complaint - see src/af3/weights.js.
    // `options.msaChannels` survives for a caller that is checking a stack in
    // isolation; the weights win when they say anything.
    const msaChannels = blocks[0].msaChannels ?? options.msaChannels ?? 64;
    const pairChannels = blocks[0].pairChannels;
    if (dialect?.msaUpdateBeforeOuterProduct === undefined) {
      throw new Error("dialect.msaUpdateBeforeOuterProduct has no default: AF3 "
        + "takes the outer product off the pre-update MSA and OpenDDE off the "
        + "updated one");
    }
    const { msaUpdateBeforeOuterProduct } = dialect;
    // 🔴 ONE VARIABLE FOR THE SHADER AND FOR THE PACKING, because they are the
    // same decision made twice. `compilePairTrack` generates kernels that read
    // the weights at this element and `packPairTrackWeights` writes them at
    // it - and this stack compiled at one and packed at the other, so an f16
    // kernel read f32 bytes as pairs of halves. Every coordinate came back NaN.
    // Same shape as the dispatch that was sized for 128 while the kernels were
    // built for 384; a value resolved in two places is how they drift.
    const pairWeightPrecision = options.pairWeightPrecision ?? "f32";
    if (!(pairChannels > 0)) {
      throw new Error("MSA blocks carry no pairChannels; they are built by "
        + "src/af3/weights.js, which derives it from the weights");
    }
    const rows = sequences * n;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    if (dialect?.swapTransposedBias === undefined) {
      throw new Error("dialect.swapTransposedBias has no default");
    }
    if (state.msa.length !== rows * msaChannels) {
      throw new Error(`msa has ${state.msa.length} elements; expected ${rows * msaChannels}`);
    }
    if (state.pair.length !== pairs * pairChannels) {
      throw new Error(`pair has ${state.pair.length} elements; expected ${pairs * pairChannels}`);
    }

    const sample = blocks[0];
    const gridHeads = sample.pairAttention1.heads;
    const msaHeads = sample.msaAttention1.heads;
    const msaDimension = sample.msaAttention1.dimension;
    const msaWidth = msaHeads * msaDimension;
    const outerChannels = sample.outerProductMean.outerChannels;
    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };

    const base = `af3-msa:${n}:${sequences}:${msaChannels}:${pairChannels}:${epsilon}:${variance}`
      + `:${dialect.swapTransposedBias}`;
    const pipelines = await compilePairTrack(this.pipelines, {
      // The device's answer, or undefined for the shared default.
      triangleProjectTile: shapedKnob(deviceTuning(this.device).trianglePairProjectTile),
      attendMatrix: resolveGridAttendMatrix(
        this.device, sample.pairAttention1.dimension, deviceTuning(this.device)),
      scratchStorage: UNPACKED_PAIR_SCRATCH,
      // 🔴 THE TRACK'S WIDTH IS THIS STACK'S, NOT compilePairTrack's DEFAULT.
      // Omitting it fell back to AlphaFold 3's 128 and split OpenDDE's
      // [384, 768] triangle projection at AF3's stride.
      n, channels: pairChannels, sample, epsilon, variance, dialect, base,
      weightPrecision: pairWeightPrecision,
      // 🔴 THIS STACK RUNS THE SAME PAIR TRACK AND HAD NONE OF ITS KERNEL
      // CHOICES. `grid.project`'s 108 passes in an AF3 trunk are 96 pairformer,
      // 8 MSA and 4 template - so a knob wired only into the pairformer leaves
      // an eighth of that kernel on the vector path for no reason. It costs no
      // memory and reads the layout this pack already writes.
      gridProjectMatrix: gridProjectMatrixConfig(this.device),
      // 🔴 AND THE SPLIT TRANSITION, WHICH THIS STACK ALSO NEVER HAD. Four of
      // an AF3 trunk's `pair-transition` passes are this stack's, and at
      // OpenDDE's 384 channels the fused kernel loses to the split by 3.71x -
      // the whole reason the pairformer takes it. Same width rule, same knob.
      pairTransitionSplit: splitTransitionConfig(this.device, pairChannels),
      pairTransitionChunkBytes: deviceTuning(this.device).pairTransitionChunkBytes,
      maxComputeWorkgroupStorageSize: this.device.limits.maxComputeWorkgroupStorageSize,
      maxStorageBufferBindingSize: this.device.limits.maxStorageBufferBindingSize,
      minStorageBufferOffsetAlignment: this.device.limits.minStorageBufferOffsetAlignment,
    });
    const gridProjectMatrix = pipelines.gridProjectMatrix === undefined ? undefined
      : allocateGridProjectMatrix(this.allocator, {
        ...pipelines.gridProjectMatrix, label: "af3-msa.grid-project",
      }, keep);
    const transitionSplit = pipelines.transitionSplit === undefined ? undefined
      : allocateTransitionSplit(this.allocator, {
        rows: n * n, channels: pairChannels,
        factor: sample.pairTransition.transition2.length / (pairChannels * pairChannels),
        chunkRows: pipelines.transitionSplit.chunkRows,
        offsets: packTransitionWeights(sample.pairTransition).offsets,
        label: "af3-msa.transition",
      }, keep);
    // 🔴 COMPILED CONCURRENTLY - see the note in pair-track-gpu.js.
    const compile = (key, source) => this.pipelines.get(key, source);
    const compiling = [];
    const into = (slot, key, source) => {
      compiling.push(compile(key, source).then((pipeline) => { pipelines[slot] = pipeline; }));
    };

    const opmShape = { sequences, tokens: n, msaChannels, outerChannels,
                       pairChannels };
    const { blockI, blockJ, blocksPerRow, ...opmSources } = createOuterProductMeanShaders(
      opmShape, packOuterProductMeanWeights(sample.outerProductMean).offsets, epsilon, variance);
    // ...the contraction's dispatch is one workgroup per (i, j) block of token
    // pairs; see the note on its kernel.
    pipelines.opmBlocks = Math.ceil(n / blockI) * blocksPerRow;
    for (const [name, source] of Object.entries(opmSources)) {
      into(`opm:${name}`, `${base}:opm:${name}`, source);
    }
    const attentionSources = createMsaAttentionShaders(
      { sequences, tokens: n, msaChannels, pairChannels,
        heads: msaHeads, dimension: msaDimension },
      packMsaAttentionWeights(sample.msaAttention1).offsets, epsilon, variance);
    for (const [name, source] of Object.entries(attentionSources)) {
      into(`msa:${name}`, `${base}:msa:${name}`, source);
    }
    into("msaTransition", `${base}:msa-transition`,
      createTransitionShader({ rows, channels: msaChannels, factor: 4 },
                             packTransitionWeights(sample.msaTransition).offsets,
                             epsilon, variance));
    into("addMsa", `${base}:add-msa`, createAddShader(rows * msaChannels));
    await Promise.all(compiling);

    try {
      const pair = keep(this.allocator.upload("af3-msa.pair", state.pair,
                                              storage | GPUBufferUsage.COPY_SRC));
      const msa = keep(this.allocator.upload("af3-msa.msa", state.msa,
                                             storage | GPUBufferUsage.COPY_SRC));
      const pairMask = keep(this.allocator.upload("af3-msa.pair-mask", state.pairMask, storage));
      const msaMask = keep(this.allocator.upload("af3-msa.msa-mask", state.msaMask, storage));

      // 🔴 FIVE PAIR-SIZED SCRATCH BUFFERS, AND THIS STACK KEEPS THEM WHOLE.
      // See UNPACKED_PAIR_SCRATCH for the four checkers that say so. The
      // allocation and the shaders read the SAME array, because a buffer that
      // disagrees with a shader about its element is not something WebGPU can
      // catch.
      const scratch = [];
      for (let index = 0; index < PAIR_SCRATCH_COUNT; index += 1) {
        scratch.push(keep(this.allocator.allocate(
          `af3-msa.scratch${index}`,
          storageBytes(pairs * pairChannels, UNPACKED_PAIR_SCRATCH[index]), storage)));
      }
      const biasBuffer = keep(this.allocator.allocate(
        "af3-msa.bias", gridHeads * pairs * 4, storage));
      const left = keep(this.allocator.allocate("af3-msa.left", rows * outerChannels * 4, storage));
      const right = keep(this.allocator.allocate("af3-msa.right", rows * outerChannels * 4, storage));
      // ...the outer product's denominator, computed once per pass rather than
      // carried through its contraction; see outer-product-mean-webgpu.js.
      const opmCounts = keep(this.allocator.allocate("af3-msa.opm-counts", pairs * 4, storage));
      const keyMask = keep(this.allocator.allocate("af3-msa.key-mask", n * 4, storage));
      const attention = keep(this.allocator.allocate(
        "af3-msa.attention", msaHeads * pairs * 4, storage));
      const msaScratch = [];
      for (let index = 0; index < 3; index += 1) {
        msaScratch.push(keep(this.allocator.allocate(
          `af3-msa.msa-scratch${index}`, rows * Math.max(msaWidth, msaChannels) * 4, storage)));
      }


      const start = performance.now();
      for (let index = 0; index < blocks.length; index += 1) {
        await this.#encodeBlock({
          block: blocks[index], n, sequences, rows, pairs, msaChannels, msaHeads, gridHeads,
          pairChannels, msaUpdateBeforeOuterProduct, pairWeightPrecision,
          pipelines, storage, pair, msa, pairMask, msaMask, scratch, biasBuffer,
          left, right, opmCounts, keyMask, attention, msaScratch, gridProjectMatrix,
          transitionSplit,
        });
        options.onBlock?.(index);
      }

      // 🔴 THE READBACKS COME AFTER THE SCRATCH GOES, NOT BEFORE THE LOOP.
      // They are written once, by the copy below, and a pair-sized MAP_READ
      // buffer standing beside six pair-sized scratch tensors for the whole
      // stack is 19.5 MiB at 200 tokens of the peak this stage HOLDS. The
      // drain is free here: the mapAsync two lines down is one anyway.
      await this.device.queue.onSubmittedWorkDone();
      for (const allocation of [...scratch, ...msaScratch, biasBuffer, attention]) {
        allocation.release();
      }
      const readbackPair = keep(this.allocator.allocate(
        "af3-msa.readback-pair", pairs * pairChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      const readbackMsa = keep(this.allocator.allocate(
        "af3-msa.readback-msa", rows * msaChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      const encoder = this.device.createCommandEncoder({ label: "af3-msa.readback" });
      encoder.copyBufferToBuffer(pair.buffer, 0, readbackPair.buffer, 0, pairs * pairChannels * 4);
      encoder.copyBufferToBuffer(msa.buffer, 0, readbackMsa.buffer, 0, rows * msaChannels * 4);
      this.device.queue.submit([encoder.finish()]);
      await readbackPair.buffer.mapAsync(GPUMapMode.READ);
      const outPair = new Float32Array(readbackPair.buffer.getMappedRange().slice(0));
      readbackPair.buffer.unmap();
      await readbackMsa.buffer.mapAsync(GPUMapMode.READ);
      const outMsa = new Float32Array(readbackMsa.buffer.getMappedRange().slice(0));
      readbackMsa.buffer.unmap();

      return {
        pair: outPair, msa: outMsa,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }

  async #encodeBlock(context) {
    const { block, n, sequences, rows, pairs, msaChannels, msaHeads, gridHeads } = context;
    const { pairChannels } = context;
    const { pipelines, storage, pair, msa, pairMask, msaMask, scratch, biasBuffer } = context;
    const { left, right, opmCounts, keyMask, attention, msaScratch } = context;
    const { gridProjectMatrix, transitionSplit } = context;

    const blockAllocations = [];
    const upload = (label, data) => {
      const allocation = this.allocator.upload(label, data, storage);
      blockAllocations.push(allocation);
      return allocation;
    };
    // 🔴 PACKED ON DEMAND AND UPLOADED ONCE, EVER, which the pairformer beside
    // this has done for a long time and this stack had not. It is four blocks
    // rather than forty-eight, and the pair-track pack alone is about 10 ms of
    // host time a block at AF3's width and more at OpenDDE's - on every pass of
    // every recycle, over weights that never change. See the note in
    // src/af3/pairformer-block-webgpu.js for why residency is a trade the
    // BUDGET answers rather than a choice made in advance.
    const resident = this.residentWeights
      ? (label, key, pack, variant) => ({
        buffer: residentWeightBuffer(this.device, key, label, pack, variant),
      })
      : (label, key, pack) => upload(label, pack());
    // Lazily, and not held: residentWeightBuffer calls pack() only on a miss.
    // The four the device can pack itself; see src/af3/pair-track-device-weights.js
    // for why a transpose and an interleave could not go the contiguous way.
    const onDevice = await residentPairTrackOnDevice(this.device, block, {
      channels: pairChannels, pairWeightPrecision: context.pairWeightPrecision,
      resident: this.residentWeights,
    });
    let packedPair;
    const w = context.pairWeightPrecision;
    const pairTransitionOnDevice = this.residentWeights
      ? await residentPackedOnDevice(this.device, {
          key: block.pairTransition, label: "w.pair-transition",
          order: TRANSITION_ORDER, weights: block.pairTransition, variant: w,
          destination: w === "f16" ? "f16" : "f32",
        }).then((buffer) => (buffer === undefined ? undefined : { buffer }))
      : undefined;
    // ...and asked BEFORE the pack, because `packPairTrackWeights` builds all
    // five from one call: a caller that stopped binding one was still paying to
    // build it. See src/af3/pair-track-device-weights.js.
    const want = { ...onDevice.want, transition: pairTransitionOnDevice === undefined };
    const packedFor = () => (packedPair ??= packPairTrackWeights(
      block, pairChannels, context.pairWeightPrecision, true, "blocked", want));
    const pairTrackWeights = {
      outgoing: onDevice.buffers.outgoing
        ?? resident("w.tri.out", block, () => packedFor().outgoing, w),
      incoming: onDevice.buffers.incoming
        ?? resident("w.tri.in", block, () => packedFor().incoming, w),
      grid1: onDevice.buffers.grid1 ?? resident("w.grid1", block, () => packedFor().grid1),
      grid2: onDevice.buffers.grid2 ?? resident("w.grid2", block, () => packedFor().grid2),
      // 🔴 THE ONE PAIR-TRACK TENSOR THE CONTIGUOUS DECODER CAN TAKE, and the
      // pairformer has taken it for a while. This stack had not: measured with
      // `residentPackStats`, `w.pair-transition` is 56 calls and 92 ms of an
      // AF3 fold and 12 calls and 130 ms of an OpenDDE one, and four of those
      // calls are this loop.
      transition: pairTransitionOnDevice
        ?? resident("w.pair-transition", block, () => packedFor().transition, w),
    };
    const opmWeights = resident("w.opm", block,
      () => packOuterProductMeanWeights(block.outerProductMean).data);
    const attentionWeights = resident("w.msa-attn", block,
      () => packMsaAttentionWeights(block.msaAttention1).data);
    const msaTransitionWeights = resident("w.msa-transition", block,
      () => packTransitionWeights(block.msaTransition).data);

    this.device.pushErrorScope("validation");
    const encoder = this.device.createCommandEncoder({ label: "af3-msa-block" });
    const run = (label, pipeline, buffers, x, y = 1, z = 1) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        // 🔴 byteOffset AND byteSize ARE HONOURED, as they are in the other two
        // stacks that encode this track. Dropping them binds the WHOLE buffer
        // where the caller asked for a range, which is silent: every index the
        // shader forms is then relative to the wrong base. The split pair
        // transition binds the pair at a row offset per chunk and is the first
        // caller here to need it; the inconsistency predates it.
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
    const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
    const ceil = (value, divisor) => Math.ceil(value / divisor);

    // 🔴 THE TWO HALVES OF AN MSA BLOCK, AND WHICH RUNS FIRST IS THE MODEL.
    // AlphaFold 3 takes the outer product off the PRE-update MSA and then
    // updates the rows against the pair that outer product just changed;
    // OpenDDE updates the rows FIRST and feeds the UPDATED MSA to the outer
    // product (upstream's `MSABlock.forward`, gated as
    // `msaUpdateBeforeOuterProduct`). Neither ordering changes a shape and both
    // produce a plausible representation, so nothing but this flag distinguishes
    // them - and the difference compounds over every block of every pass.
    const outerProduct = () => {
      const rowGroups = spread(ceil(rows, 64));
      run("opm.project", pipelines["opm:project"],
          [msa, msaMask, opmWeights, left, right], rowGroups[0], rowGroups[1]);
      const countGroups = spread(ceil(pairs, 64));
      run("opm.counts", pipelines["opm:counts"], [msaMask, opmCounts],
          countGroups[0], countGroups[1]);
      const perBlock = spread(pipelines.opmBlocks);
      run("opm.contract", pipelines["opm:contract"],
          [left, right, opmCounts, opmWeights, scratch[0]], perBlock[0], perBlock[1]);
      const addPairGroups = spread(ceil(pairs * pairChannels, 64));
      run("opm.add", pipelines.addPair, [pair, scratch[0]], addPairGroups[0], addPairGroups[1]);
    };

    // The row update: the pair-weighted average, then the transition. Upstream
    // keeps the transition inside `_msa_update`, so it moves with it.
    const updateMsa = () => {
      const addMsaGroups = spread(ceil(rows * msaChannels, 64));
      run("msa.key-mask", pipelines["msa:keyMask"], [msaMask, keyMask], ceil(n, 64));
      const weightGroups = spread(msaHeads * n);
      run("msa.attention-weights", pipelines["msa:attentionWeights"],
          [pair, keyMask, attentionWeights, attention], weightGroups[0], weightGroups[1]);
      const perRow = spread(rows);
      // ...one workgroup a row now; see the note on the kernel.
      run("msa.project", pipelines["msa:project"],
          [msa, attentionWeights, msaScratch[0], msaScratch[1]], perRow[0], perRow[1]);
      run("msa.average", pipelines["msa:average"],
          [attention, msaScratch[0], msaScratch[1], attentionWeights, msaScratch[2]],
          perRow[0], perRow[1]);
      run("msa.add", pipelines.addMsa, [msa, msaScratch[2]], addMsaGroups[0], addMsaGroups[1]);

      const perTransition = spread(ceil(rows, transitionRowTile(rows, msaChannels)));
      run("msa-transition", pipelines.msaTransition, [msa, msaTransitionWeights, msaScratch[0]],
          perTransition[0], perTransition[1]);
      run("msa-transition.add", pipelines.addMsa, [msa, msaScratch[0]],
          addMsaGroups[0], addMsaGroups[1]);
    };

    if (context.msaUpdateBeforeOuterProduct) {
      updateMsa();
      outerProduct();
    } else {
      outerProduct();
      updateMsa();
    }

    encodePairTrack({
      run, pipelines, n, channels: pairChannels, gridHeads, pair, pairMask,
      scratch, biasBuffer, gridProjectMatrix, transitionSplit,
      weights: pairTrackWeights,
    });

    this.device.queue.submit([encoder.finish()]);
    const error = await this.device.popErrorScope();
    await this.device.queue.onSubmittedWorkDone();
    for (let index = blockAllocations.length - 1; index >= 0; index -= 1) {
      blockAllocations[index].release();
    }
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
  }
}
