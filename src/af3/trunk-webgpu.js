/**
 * AF3's whole trunk on the GPU.
 *
 *     embed -> template embedding -> 4 x msaBlock -> 48 x pairformerBlock
 *           -> distogram head
 *
 * This is assembly, not new arithmetic: every stage has its own differential
 * test in tools/gpu. What it adds is what no per-stage check can show - that
 * the stages fit in the order AF3 runs them, and that fifty-two blocks of a
 * residual stack do not turn correct blocks into a wrong answer.
 *
 * 🔴 THE TEMPLATE EMBEDDING READS THE PAIR PART-BUILT. It goes in after the
 * relative encoding and the bonds and before anything else, which is why the
 * embedder does not add it and this file sequences the two. Running the
 * template embedder on the finished pair, or on the pair after the MSA stack,
 * is a natural reading of "add the template embedding" and a different model.
 *
 * WHY THE STAGES READ BACK BETWEEN THEMSELVES, and the blocks do not. Within a
 * stack the pair representation crosses the bus twice per operation if you let
 * it - 336 round trips for the pairformer alone - so the stacks keep everything
 * resident. Between the four stages it is four round trips for the whole trunk,
 * against a run that takes tens of seconds. That is not worth the plumbing, and
 * the stage boundary is where a caller wants to be able to look anyway.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { Af3EmbedderGpu } from "./embedder-webgpu.js";
import { Af3MsaStackGpu } from "./msa-stack-webgpu.js";
import { Af3PairformerStackGpu } from "./pairformer-block-webgpu.js";
import { Af3TemplateEmbedderGpu } from "./template-webgpu.js";
import { GRID_WIDTH, PAIR_CHANNELS } from "./pair-track-gpu.js";

const NUM_BINS = 64;
const FIRST_BREAK = 2.3125;
const LAST_BREAK = 21.6875;
const CONTACT_THRESHOLD = 8.0 + 1e-3;

/** The distogram bin edges: 63 of them, evenly spaced. */
export function binEdges() {
  const breaks = new Float32Array(NUM_BINS - 1);
  for (let index = 0; index < NUM_BINS - 1; index += 1) {
    breaks[index] = FIRST_BREAK + (LAST_BREAK - FIRST_BREAK) * index / (NUM_BINS - 2);
  }
  return breaks;
}

/**
 * The distogram head.
 *
 * 🔴 SYMMETRISED BY A SUM, NOT A MEAN. AF3 computes one half and adds its own
 * transpose, so the logits are twice a symmetric average. Halving them looks
 * like a normalisation and moves every contact probability.
 *
 * 🔴 A BIN COUNTS AS CONTACT WHEN ITS TOP EDGE IS AT OR BELOW 8 A. The 63
 * breaks describe 64 bins, so the final bin is open-ended and its top has to be
 * extrapolated by one spacing rather than read from the array.
 */
export function createDistogramShader(tokens, channels, offset) {
  const contactBins = [];
  const breaks = binEdges();
  const spacing = breaks[breaks.length - 1] - breaks[breaks.length - 2];
  for (let bin = 0; bin < NUM_BINS; bin += 1) {
    const top = bin < NUM_BINS - 1 ? breaks[bin] : breaks[breaks.length - 1] + spacing;
    contactBins.push(top <= CONTACT_THRESHOLD ? "1.0" : "0.0");
  }
  return `
const TOKENS: u32 = ${tokens}u;
const PAIRS: u32 = ${tokens * tokens}u;
const CHANNELS: u32 = ${channels}u;
const BINS: u32 = ${NUM_BINS}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const W_HALF: u32 = ${offset}u;
const CONTACT = array<f32, ${NUM_BINS}>(${contactBins.join(", ")});

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> pair_mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;
@group(0) @binding(4) var<storage, read_write> contact: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let i = row / TOKENS;
  let j = row % TOKENS;
  let transposed = j * TOKENS + i;

  var largest = -3.0e38;
  var values: array<f32, ${NUM_BINS}>;
  for (var b = 0u; b < BINS; b += 1u) {
    var total = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      // ...one half plus its own transpose.
      total += (pair[row * CHANNELS + c] + pair[transposed * CHANNELS + c])
        * weights[W_HALF + c * BINS + b];
    }
    values[b] = total;
    logits[row * BINS + b] = total;
    largest = max(largest, total);
  }

  var sum = 0.0;
  var contact_total = 0.0;
  for (var b = 0u; b < BINS; b += 1u) {
    let probability = exp(values[b] - largest);
    sum += probability;
    contact_total += CONTACT[b] * probability;
  }
  contact[row] = pair_mask[row] * (contact_total / sum);
}`;
}

export class Af3TrunkGpu {
  /** @param {{residentWeights?: boolean}} [options] passed to the pairformer. */
  constructor(device, options = {}) {
    this.options = options;
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {object} input as the embedder takes it, plus pairMask, seqMask,
   *   msaMask and `templates`
   * @param {{embedder: object, template: object, msaBlocks: object[],
   *          pairformerBlocks: object[], distogram: object}} weights
   * @param {{swapTransposedBias: boolean}} dialect
   * @param {{onStage?: (name: string, elapsed: number) => void,
   *          onPairformerBlock?: (index: number, total: number) => void}} options
   */
  async run(input, weights, dialect, options = {}) {
    const tokens = input.tokens;
    const pairs = tokens * tokens;
    const timings = {};
    const stage = async (name, work) => {
      const start = performance.now();
      const value = await work();
      timings[name] = performance.now() - start;
      options.onStage?.(name, timings[name]);
      return value;
    };

    const embedded = await stage("embedder",
      () => new Af3EmbedderGpu(this.device).run(input, weights.embedder, options));

    // 🔴 ON THE PART-BUILT PAIR - see the note at the top.
    const template = await stage("template", () => new Af3TemplateEmbedderGpu(this.device).run(
      { pair: embedded.pair, pairMask: input.pairMask, tokens,
        templates: input.templates ?? 4, templateOccupied: false,
        templateAatype: input.templateAatype },
      weights.template, dialect, options));
    const pair = Float32Array.from(embedded.pair);
    for (let index = 0; index < pair.length; index += 1) pair[index] += template.output[index];

    const msa = await stage("msa-stack", () => new Af3MsaStackGpu(this.device).run(
      { pair, msa: embedded.msa, pairMask: input.pairMask, msaMask: input.msaMask,
        tokens, sequences: input.sequences },
      weights.msaBlocks, dialect, options));

    // 🔴 THE ONLY STAGE WORTH A PROGRESS BAR. The pairformer is 48 blocks and
    // the bulk of the trunk; the other four stages are each a fraction of it,
    // so a bar that only moved between stages would sit still for most of the
    // wait. onBlock is reported under its own name rather than through
    // `options`, which is passed to every sub-stack and would otherwise fire
    // for the template's blocks too.
    const pairformer = await stage("pairformer", () => new Af3PairformerStackGpu(this.device, this.options).run(
      { pair: msa.pair, single: embedded.single, pairMask: input.pairMask,
        seqMask: input.seqMask, tokens },
      weights.pairformerBlocks, dialect, {
        ...options,
        onBlock: (index) => options.onPairformerBlock?.(index,
                                                        weights.pairformerBlocks.length),
        // ...and the one that says the device GOT there, which is what a status
        // line should show. See the note in pairformer-block-webgpu.js.
        onBlockDone: (completed, total) => options.onPairformerBlockDone?.(completed, total),
      }));

    const head = await stage("distogram",
      () => this.#distogram(pairformer.pair, input.pairMask, tokens, weights.distogram));

    return {
      pair: pairformer.pair, single: pairformer.single, msa: msa.msa,
      ...head, binEdges: binEdges(), timings,
    };
  }

  async #distogram(pair, pairMask, tokens, weights) {
    const pairs = tokens * tokens;
    const packed = new Float32Array(weights.halfLogits.length);
    packed.set(weights.halfLogits, 0);
    const pipeline = await this.pipelines.get(
      `af3-distogram:${tokens}:${PAIR_CHANNELS}`,
      createDistogramShader(tokens, PAIR_CHANNELS, 0));

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const pairBuffer = keep(this.allocator.upload("af3-disto.pair", pair, storage));
      const maskBuffer = keep(this.allocator.upload("af3-disto.mask", pairMask, storage));
      const weightBuffer = keep(this.allocator.upload("af3-disto.weights", packed, storage));
      const logits = keep(this.allocator.allocate("af3-disto.logits", pairs * NUM_BINS * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const contact = keep(this.allocator.allocate("af3-disto.contact", pairs * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const readLogits = keep(this.allocator.allocate("af3-disto.rb-logits",
        pairs * NUM_BINS * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      const readContact = keep(this.allocator.allocate("af3-disto.rb-contact",
        pairs * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-distogram" });
      const pass = encoder.beginComputePass({ label: "af3-distogram" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [pairBuffer, maskBuffer, weightBuffer, logits, contact].map(
          (allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
      }));
      const groups = Math.ceil(pairs / 64);
      pass.dispatchWorkgroups(Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH));
      pass.end();
      encoder.copyBufferToBuffer(logits.buffer, 0, readLogits.buffer, 0, pairs * NUM_BINS * 4);
      encoder.copyBufferToBuffer(contact.buffer, 0, readContact.buffer, 0, pairs * 4);
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);

      const read = async (allocation) => {
        await allocation.buffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(allocation.buffer.getMappedRange().slice(0));
        allocation.buffer.unmap();
        return copy;
      };
      return { logits: await read(readLogits), contactProbs: await read(readContact) };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
