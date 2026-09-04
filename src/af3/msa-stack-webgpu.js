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
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import {
  GRID_WIDTH, PAIR_CHANNELS, compilePairTrack, createAddShader, encodePairTrack,
  packPairTrackWeights,
} from "./pair-track-gpu.js";
import {
  createOuterProductMeanShaders, packOuterProductMeanWeights,
} from "./outer-product-mean-webgpu.js";
import { createMsaAttentionShaders, packMsaAttentionWeights } from "./msa-attention-webgpu.js";
import { createTransitionShader, packTransitionWeights, transitionRowTile }
  from "./transition-webgpu.js";

export class Af3MsaStackGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
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
    const msaChannels = options.msaChannels ?? 64;
    const rows = sequences * n;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    if (dialect?.swapTransposedBias === undefined) {
      throw new Error("dialect.swapTransposedBias has no default");
    }
    if (state.msa.length !== rows * msaChannels) {
      throw new Error(`msa has ${state.msa.length} elements; expected ${rows * msaChannels}`);
    }
    if (state.pair.length !== pairs * PAIR_CHANNELS) {
      throw new Error(`pair has ${state.pair.length} elements; expected ${pairs * PAIR_CHANNELS}`);
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

    const base = `af3-msa:${n}:${sequences}:${msaChannels}:${epsilon}:${variance}`
      + `:${dialect.swapTransposedBias}`;
    const pipelines = await compilePairTrack(this.pipelines, {
      n, sample, epsilon, variance, dialect, base,
    });
    // 🔴 COMPILED CONCURRENTLY - see the note in pair-track-gpu.js.
    const compile = (key, source) => this.pipelines.get(key, source);
    const compiling = [];
    const into = (slot, key, source) => {
      compiling.push(compile(key, source).then((pipeline) => { pipelines[slot] = pipeline; }));
    };

    const opmShape = { sequences, tokens: n, msaChannels, outerChannels,
                       pairChannels: PAIR_CHANNELS };
    const { blockI, blockJ, blocksPerRow, ...opmSources } = createOuterProductMeanShaders(
      opmShape, packOuterProductMeanWeights(sample.outerProductMean).offsets, epsilon, variance);
    // ...the contraction's dispatch is one workgroup per (i, j) block of token
    // pairs; see the note on its kernel.
    pipelines.opmBlocks = Math.ceil(n / blockI) * blocksPerRow;
    for (const [name, source] of Object.entries(opmSources)) {
      into(`opm:${name}`, `${base}:opm:${name}`, source);
    }
    const attentionSources = createMsaAttentionShaders(
      { sequences, tokens: n, msaChannels, pairChannels: PAIR_CHANNELS,
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

      const scratch = [];
      for (let index = 0; index < 7; index += 1) {
        scratch.push(keep(this.allocator.allocate(
          `af3-msa.scratch${index}`, pairs * PAIR_CHANNELS * 4, storage)));
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

      const readbackPair = keep(this.allocator.allocate(
        "af3-msa.readback-pair", pairs * PAIR_CHANNELS * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      const readbackMsa = keep(this.allocator.allocate(
        "af3-msa.readback-msa", rows * msaChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      const start = performance.now();
      for (let index = 0; index < blocks.length; index += 1) {
        await this.#encodeBlock({
          block: blocks[index], n, sequences, rows, pairs, msaChannels, msaHeads, gridHeads,
          pipelines, storage, pair, msa, pairMask, msaMask, scratch, biasBuffer,
          left, right, opmCounts, keyMask, attention, msaScratch,
        });
        options.onBlock?.(index);
      }

      const encoder = this.device.createCommandEncoder({ label: "af3-msa.readback" });
      encoder.copyBufferToBuffer(pair.buffer, 0, readbackPair.buffer, 0, pairs * PAIR_CHANNELS * 4);
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
    const { pipelines, storage, pair, msa, pairMask, msaMask, scratch, biasBuffer } = context;
    const { left, right, opmCounts, keyMask, attention, msaScratch } = context;

    const blockAllocations = [];
    const upload = (label, data) => {
      const allocation = this.allocator.upload(label, data, storage);
      blockAllocations.push(allocation);
      return allocation;
    };
    const packedPair = packPairTrackWeights(block);
    const pairTrackWeights = {
      outgoing: upload("w.tri.out", packedPair.outgoing),
      incoming: upload("w.tri.in", packedPair.incoming),
      grid1: upload("w.grid1", packedPair.grid1),
      grid2: upload("w.grid2", packedPair.grid2),
      transition: upload("w.pair-transition", packedPair.transition),
    };
    const opmWeights = upload("w.opm", packOuterProductMeanWeights(block.outerProductMean).data);
    const attentionWeights = upload("w.msa-attn",
      packMsaAttentionWeights(block.msaAttention1).data);
    const msaTransitionWeights = upload("w.msa-transition",
      packTransitionWeights(block.msaTransition).data);

    this.device.pushErrorScope("validation");
    const encoder = this.device.createCommandEncoder({ label: "af3-msa-block" });
    const run = (label, pipeline, buffers, x, y = 1, z = 1) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((allocation, binding) => ({
          binding, resource: { buffer: allocation.buffer },
        })),
      }));
      pass.dispatchWorkgroups(x, y, z);
      pass.end();
    };
    const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
    const ceil = (value, divisor) => Math.ceil(value / divisor);

    // 🔴 THE OUTER PRODUCT READS THE MSA FIRST, before either MSA update below.
    const rowGroups = spread(ceil(rows, 64));
    run("opm.project", pipelines["opm:project"],
        [msa, msaMask, opmWeights, left, right], rowGroups[0], rowGroups[1]);
    const countGroups = spread(ceil(pairs, 64));
    run("opm.counts", pipelines["opm:counts"], [msaMask, opmCounts],
        countGroups[0], countGroups[1]);
    const perBlock = spread(pipelines.opmBlocks);
    run("opm.contract", pipelines["opm:contract"],
        [left, right, opmCounts, opmWeights, scratch[0]], perBlock[0], perBlock[1]);
    const addPairGroups = spread(ceil(pairs * PAIR_CHANNELS, 64));
    run("opm.add", pipelines.addPair, [pair, scratch[0]], addPairGroups[0], addPairGroups[1]);

    // ...and the MSA update reads the pair the outer product just changed.
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

    const perTransition = spread(ceil(rows, transitionRowTile(rows)));
    run("msa-transition", pipelines.msaTransition, [msa, msaTransitionWeights, msaScratch[0]],
        perTransition[0], perTransition[1]);
    run("msa-transition.add", pipelines.addMsa, [msa, msaScratch[0]],
        addMsaGroups[0], addMsaGroups[1]);

    encodePairTrack({
      run, pipelines, n, gridHeads, pair, pairMask, scratch, biasBuffer,
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
