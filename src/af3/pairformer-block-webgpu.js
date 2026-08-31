/**
 * AF3's pairformer block, and a stack of them, resident on the GPU.
 *
 *     pair   += triangle_multiplication_outgoing
 *     pair   += triangle_multiplication_incoming
 *     pair   += grid_self_attention(row)
 *     pair   += grid_self_attention(column)
 *     pair   += transition
 *     single += single_attention(biased by the pair, AFTER all five updates)
 *     single += transition
 *
 * WHY THIS DOES NOT USE THE PER-OPERATION RUNNER CLASSES. Each of those owns an
 * upload and a readback, which is what a differential test wants and the exact
 * opposite of what a 48-block stack wants: the pair representation is 46 MB at
 * 300 tokens and 184 MB at 600, and moving it across the bus twice per
 * operation - seven times a block, 336 times a trunk - would dominate
 * everything else the kernels do. So the block encodes the same SHADERS, which
 * are where the correctness lives and which their own checkers still pin, over
 * buffers that never leave the GPU.
 *
 * 🔴 THE SCRATCH BUFFERS ARE SHARED BETWEEN OPERATIONS AND THAT IS DELIBERATE.
 * Triangle multiplication's a/b/contracted and grid attention's q/k/v are never
 * live at the same time, so they are the same memory. Seven pair-sized scratch
 * buffers serve the whole block, allocated once for the whole stack rather than
 * per block - which is what keeps a 48-block trunk's peak equal to a single
 * block's. Anything added here that outlives its operation breaks that, and the
 * symptom is a trunk that dies at block 40 rather than block 1.
 *
 * 🔴 THE SINGLE TRACK READS THE PAIR AFTER ALL FIVE PAIR UPDATES, not before.
 * Reading it earlier is a plausible-looking reordering that still converges.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import {
  GRID_WIDTH, PAIR_CHANNELS, compilePairTrack, createAddShader, encodePairTrack,
  packPairTrackWeights,
} from "./pair-track-gpu.js";
import { createTransitionShader, packTransitionWeights } from "./transition-webgpu.js";
import { createSingleAttentionShaders, packSingleAttentionWeights } from "./single-attention-webgpu.js";

const SINGLE_CHANNELS = 384;

/**
 * The pair logits that bias single attention: LayerNorm the pair, project to
 * one scalar per head, and write it HEAD-MAJOR, which is the layout the
 * attention kernel reads a row of.
 */
function createPairLogitsShader(n, channels, heads, offsets, epsilon, variance) {
  const pairs = n * n;
  return `
const PAIRS: u32 = ${pairs}u;
const CHANNELS: u32 = ${channels}u;
const HEADS: u32 = ${heads}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
const W_SCALE: u32 = ${offsets.scale}u;
const W_OFFSET: u32 = ${offsets.offset}u;
const W_PROJECT: u32 = ${offsets.projection}u;

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * CHANNELS;
  var total = 0.0;
  var squares = 0.0;
  for (var c = 0u; c < CHANNELS; c += 1u) {
    let value = pair[base + c];
    total += value;
    squares += value * value;
  }
  let mean = total / f32(CHANNELS);
  ${variance === "fast"
    ? "let variance = squares / f32(CHANNELS) - mean * mean;"
    : `var variance = 0.0;
  for (var c = 0u; c < CHANNELS; c += 1u) {
    let d = pair[base + c] - mean;
    variance += d * d;
  }
  variance /= f32(CHANNELS);`}
  let inverse_std = inverseSqrt(variance + EPSILON);
  for (var h = 0u; h < HEADS; h += 1u) {
    var sum = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      let value = (pair[base + c] - mean) * inverse_std * weights[W_SCALE + c]
        + weights[W_OFFSET + c];
      sum += value * weights[W_PROJECT + c * HEADS + h];
    }
    logits[h * PAIRS + row] = sum;
  }
}`;
}

function packPairLogitsWeights(weights) {
  const order = ["scale", "offset", "projection"];
  const offsets = {};
  let total = 0;
  for (const name of order) {
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of order) data.set(weights[name], offsets[name]);
  return { data, offsets };
}

export class Af3PairformerStackGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * Run a stack of pairformer blocks.
   *
   * @param {{pair: Float32Array, single: Float32Array, pairMask: Float32Array,
   *          seqMask: Float32Array, tokens: number}} state
   * @param {object[]} blocks one weight bundle per block, as the checkers build
   *   them for the CPU reference
   * @param {{swapTransposedBias: boolean}} dialect
   * @param {{epsilon?: number, variance?: "fast"|"two-pass",
   *          onBlock?: (index: number) => void}} options
   */
  async run(state, blocks, dialect, options = {}) {
    const n = state.tokens;
    const pairs = n * n;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    if (dialect?.swapTransposedBias === undefined) {
      throw new Error("dialect.swapTransposedBias has no default");
    }
    if (state.pair.length !== pairs * PAIR_CHANNELS) {
      throw new Error(`pair has ${state.pair.length} elements; expected ${pairs * PAIR_CHANNELS}`);
    }
    if (state.single.length !== n * SINGLE_CHANNELS) {
      throw new Error(`single has ${state.single.length} elements; expected ${n * SINGLE_CHANNELS}`);
    }

    const heads = blocks[0].singleAttention.heads;
    const gridHeads = blocks[0].pairAttention1.heads;
    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    const pairBytes = pairs * PAIR_CHANNELS * 4;

    // The pair track, shared with the MSA stack.
    const base = `af3-block:${n}:${epsilon}:${variance}:${dialect.swapTransposedBias}`;
    const pipelines = await compilePairTrack(this.pipelines, {
      n, sample: blocks[0], epsilon, variance, dialect, base,
    });
    const compile = (key, source) => this.pipelines.get(key, source);

    const singleTransitionOffsets = packTransitionWeights(blocks[0].singleTransition).offsets;
    const singleOffsets = packSingleAttentionWeights(blocks[0].singleAttention).offsets;
    const logitsOffsets = packPairLogitsWeights({
      scale: blocks[0].singlePairLogitsNormScale,
      offset: blocks[0].singlePairLogitsNormOffset,
      projection: blocks[0].singlePairLogitsProjection,
    }).offsets;
    pipelines.singleTransition = await compile(`${base}:single-transition`,
      createTransitionShader({ rows: n, channels: SINGLE_CHANNELS, factor: 4 },
                             singleTransitionOffsets, epsilon, variance));
    const singleSources = createSingleAttentionShaders(
      { n, channels: SINGLE_CHANNELS, heads, dimension: blocks[0].singleAttention.dimension },
      singleOffsets, epsilon, variance);
    for (const [name, source] of Object.entries(singleSources)) {
      pipelines[`single:${name}`] = await compile(`${base}:single:${name}`, source);
    }
    pipelines.pairLogits = await compile(`${base}:pair-logits`,
      createPairLogitsShader(n, PAIR_CHANNELS, heads, logitsOffsets, epsilon, variance));
    pipelines.addSingle = await compile(`${base}:add-single`, createAddShader(n * SINGLE_CHANNELS));

    try {
      // Resident state.
      const pair = keep(this.allocator.upload("af3-block.pair", state.pair, storage | GPUBufferUsage.COPY_SRC));
      const single = keep(this.allocator.upload("af3-block.single", state.single, storage | GPUBufferUsage.COPY_SRC));
      const pairMask = keep(this.allocator.upload("af3-block.pair-mask", state.pairMask, storage));
      const seqMask = keep(this.allocator.upload("af3-block.seq-mask", state.seqMask, storage));

      // Seven pair-sized scratch buffers, shared by every operation.
      const scratch = [];
      for (let index = 0; index < 7; index += 1) {
        scratch.push(keep(this.allocator.allocate(`af3-block.scratch${index}`, pairBytes, storage)));
      }
      const biasBuffer = keep(this.allocator.allocate(
        "af3-block.bias", gridHeads * pairs * 4, storage));
      const pairLogits = keep(this.allocator.allocate(
        "af3-block.pair-logits", heads * pairs * 4, storage));
      const singleWidth = heads * blocks[0].singleAttention.dimension;
      const singleScratch = [];
      for (let index = 0; index < 6; index += 1) {
        singleScratch.push(keep(this.allocator.allocate(
          `af3-block.single-scratch${index}`,
          Math.max(n * singleWidth, n * SINGLE_CHANNELS) * 4, storage)));
      }

      const readbackPair = keep(this.allocator.allocate(
        "af3-block.readback-pair", pairBytes,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      const readbackSingle = keep(this.allocator.allocate(
        "af3-block.readback-single", n * SINGLE_CHANNELS * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      const start = performance.now();
      for (let index = 0; index < blocks.length; index += 1) {
        await this.#encodeBlock({
          block: blocks[index], n, pairs, heads, gridHeads, pipelines, storage, keep,
          pair, single, pairMask, seqMask, scratch, biasBuffer, pairLogits, singleScratch,
        });
        options.onBlock?.(index);
      }

      const encoder = this.device.createCommandEncoder({ label: "af3-block.readback" });
      encoder.copyBufferToBuffer(pair.buffer, 0, readbackPair.buffer, 0, pairBytes);
      encoder.copyBufferToBuffer(single.buffer, 0, readbackSingle.buffer, 0, n * SINGLE_CHANNELS * 4);
      this.device.queue.submit([encoder.finish()]);
      await readbackPair.buffer.mapAsync(GPUMapMode.READ);
      const outPair = new Float32Array(readbackPair.buffer.getMappedRange().slice(0));
      readbackPair.buffer.unmap();
      await readbackSingle.buffer.mapAsync(GPUMapMode.READ);
      const outSingle = new Float32Array(readbackSingle.buffer.getMappedRange().slice(0));
      readbackSingle.buffer.unmap();

      return {
        pair: outPair, single: outSingle,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }

  /** One block, submitted as one command buffer. */
  async #encodeBlock(context) {
    const { block, n, pairs, heads, gridHeads, pipelines, storage } = context;
    const { pair, single, pairMask, seqMask, scratch, biasBuffer, pairLogits, singleScratch } = context;

    // 🔴 PER-BLOCK WEIGHTS ARE RELEASED PER BLOCK, not with the stack. Each
    // block's are about 12 MB; holding all 48 would add ~576 MB to a trunk for
    // no reason, and it would look like the scratch pool leaking rather than
    // the weights.
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
    const singleTransitionWeights = upload("w.single-transition",
      packTransitionWeights(block.singleTransition).data);
    const singleWeights = upload("w.single", packSingleAttentionWeights(block.singleAttention).data);
    const logitsWeights = upload("w.pair-logits", packPairLogitsWeights({
      scale: block.singlePairLogitsNormScale,
      offset: block.singlePairLogitsNormOffset,
      projection: block.singlePairLogitsProjection,
    }).data);

    this.device.pushErrorScope("validation");
    const encoder = this.device.createCommandEncoder({ label: "af3-pairformer-block" });
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

    encodePairTrack({
      run, pipelines, n, gridHeads, pair, pairMask, scratch, biasBuffer,
      weights: pairTrackWeights,
    });

    // 🔴 AFTER the five, never before.
    const linear = spread(ceil(pairs, 64));
    run("pair-logits", pipelines.pairLogits, [pair, logitsWeights, pairLogits], linear[0], linear[1]);

    run("single.project", pipelines["single:project"],
        [single, singleWeights, singleScratch[0], singleScratch[1], singleScratch[2],
         singleScratch[3]], n);
    run("single.attend", pipelines["single:attend"],
        [singleScratch[0], singleScratch[1], singleScratch[2], pairLogits, seqMask,
         singleScratch[4]], n * heads);
    run("single.project-out", pipelines["single:project_out"],
        [singleScratch[4], singleScratch[3], singleWeights, singleScratch[5]], n);
    const addSingle = spread(ceil(n * SINGLE_CHANNELS, 64));
    run("single.add", pipelines.addSingle, [single, singleScratch[5]], addSingle[0], addSingle[1]);

    run("single-transition", pipelines.singleTransition,
        [single, singleTransitionWeights, singleScratch[0]], n);
    run("single-transition.add", pipelines.addSingle, [single, singleScratch[0]],
        addSingle[0], addSingle[1]);

    this.device.queue.submit([encoder.finish()]);
    const error = await this.device.popErrorScope();
    // ...and wait for the GPU to finish with them before handing the memory
    // back, since release recycles rather than merely forgets.
    await this.device.queue.onSubmittedWorkDone();
    for (let index = blockAllocations.length - 1; index >= 0; index -= 1) {
      blockAllocations[index].release();
    }
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
  }
}
