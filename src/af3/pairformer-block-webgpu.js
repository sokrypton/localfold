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
import { DeferredValidation } from "../runtime/validation.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { releaseResidentWeights, residentWeightBuffer } from "../runtime/resident.js";
import { storageBytes } from "../runtime/storage.js";
import { releaseWeights } from "./weights.js";
import { GpuMemoryBudgetError, noteResidencyRefused, residencyAllowed }
  from "../runtime/device-memory.js";
import { transitionRowTile } from "./transition-webgpu.js";
import {
  GRID_WIDTH, PAIR_CHANNELS, PAIR_SCRATCH_COUNT, UNPACKED_PAIR_SCRATCH,
  compilePairTrack, createAddShader, encodePairTrack,
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
  // 🔴 THE HEADS ARE CONTIGUOUS IN THE PROJECTION, SO THEY ARE THE VECTOR - and
  // the normalisation belongs outside them. This looped heads OUTSIDE channels
  // and re-derived the normalised value inside both, so a row cost
  // HEADS x CHANNELS x (four reads and four operations) where it needs
  // CHANNELS x (three reads and HEADS/4 vector multiply-adds). The same shape
  // in the diffusion transformer's pair-logits was worth 3.7x.
  const headVectors = Math.ceil(heads / 4);
  const overHeadVectors = (body) =>
    Array.from({ length: headVectors }, (_, h) => body(h)).join("\n    ");
  if (heads % 4 !== 0) throw new Error(`heads ${heads} is not a multiple of four`);
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
// ...as vec4, which is why W_PROJECT and HEADS must both be multiples of four.
@group(0) @binding(2) var<storage, read> projection: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;

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
  ${overHeadVectors((h) => `var sum${h} = vec4<f32>(0.0);`)}
  for (var c = 0u; c < CHANNELS; c += 1u) {
    // ...normalised once, used by every head.
    let value = (pair[base + c] - mean) * inverse_std * weights[W_SCALE + c]
      + weights[W_OFFSET + c];
    let column = (W_PROJECT + c * HEADS) / 4u;
    ${overHeadVectors((h) => `sum${h} += value * projection[column + ${h}u];`)}
  }
  ${overHeadVectors((h) => Array.from({ length: 4 }, (_, l) =>
    `logits[(${h * 4 + l}u) * PAIRS + row] = sum${h}.${"xyzw"[l]};`).join("\n  "))}
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
  /**
   * @param {{residentWeights?: boolean}} [options] whether a block's weights
   *   stay on the device between passes. See the note at the upload below.
   */
  constructor(device, options = {}) {
    this.device = device;
    this.options = options;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
    this.residentWeights = (options.residentWeights ?? true) && residencyAllowed(device);
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
   *          onBlock?: (index: number) => void,
   *          onBlockDone?: (completed: number, total: number) => void}} options
   */
  /**
   * Run the stack, and if the device cannot afford to keep the weights on it,
   * run it again without doing that.
   *
   * 🔴 THE FALLBACK CANNOT HAPPEN MID-STACK, WHICH IS WHY IT IS A RETRY. The
   * refusal arrives partway through, with sixteen blocks' weights already on
   * the device and their command buffers still in flight; freeing them there
   * gives "Buffer w.tri.out used in submit while destroyed", and NOT freeing
   * them leaves the per-pass path to fit inside whatever budget they left - at
   * a 200 MiB ceiling, ten megabytes. So the stack is abandoned instead: drain
   * the queue, hand back every resident weight buffer, and encode the whole
   * thing again uploading per block. That costs one partial stack, once, on a
   * machine that could not have finished the other way.
   *
   * 🔴 AND IT IS NOT PREDICTED. Deciding in advance needs an estimate of what
   * the stack will hold against what scratch will need, which is a number that
   * goes stale in silence - upstream's had drifted 3-5x before anyone looked.
   * The budget already knows the answer; this asks it.
   */
  async run(state, blocks, dialect, options = {}) {
    try {
      return await this.#runStack(state, blocks, dialect, options);
    } catch (error) {
      if (!(error instanceof GpuMemoryBudgetError) || !this.residentWeights) throw error;
      await this.device.queue.onSubmittedWorkDone();
      const reclaimed = releaseResidentWeights(this.device);
      this.residentWeights = false;
      noteResidencyRefused(this.device);
      this.degradedTo = `uploading weights per pass (${(reclaimed / (1024 * 1024)).toFixed(0)}`
        + ` MiB reclaimed): ${error.message}`;
      options.onStatus?.(this.degradedTo);
      return await this.#runStack(state, blocks, dialect, options);
    }
  }

  async #runStack(state, blocks, dialect, options = {}) {
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
    const hasF16 = this.device.features?.has("shader-f16") === true;
    const stagedPrecision = this.options?.stagedPrecision ?? (hasF16 ? "f16" : "f32");
    // 🔴 THE RESIDENT WEIGHTS ARE THE MEMORY, AND THE SINGLE TRACK IS THE
    // WEIGHTS. Broken down by label, the 567 MiB an AF3 TRUNK keeps resident is
    // w.single-transition 324 and w.single 135 - 81% of it in two tensors,
    // because the single track runs 384 channels against the pair track's 128
    // and its transition widens by four on top. Held in f16 those are 230 MiB
    // instead of 459. It buys no time, and is not meant to: these kernels read
    // their weights one scalar at a time and this machine is instruction-bound,
    // so halving the BYTES does not halve the read instructions. Measured on
    // the pairformer at 118 tokens, three interleaved pairs: 163, 163, 166 ms
    // in f32 against 166, 167, 168 in f16 - so it costs about 2%, from the
    // f32() on every read, rather than costing nothing. What it buys is a
    // device small enough to hold the model at all.
    const weightPrecision = this.options?.weightPrecision ?? (hasF16 ? "f16" : "f32");
    // 🔴 THE PAIR TRACK'S OWN WEIGHTS STAY f32 BY DEFAULT, AND THE REASON IS THE
    // RATIO. Narrowing them saves 38 MiB - the triangle's 20 and the pair
    // transition's 18 - which is 6% of the 608 MiB the single track and the
    // diffusion transformer give up between them. What it costs is the worst
    // amplification measured anywhere here: the block checker's pair goes from
    // 17x its rounding envelope to 51x, and the confidence head's pLDDT from
    // 3.0e-4 to 2.3e-2. 94% of the saving for none of that is the better half
    // of the trade.
    //
    // It is still offered, because "this device cannot hold the model" is a
    // real state that --budget already exists for, and 38 MiB is 38 MiB when
    // the alternative is not folding.
    const pairWeightPrecision = this.options?.pairWeightPrecision ?? "f32";
    // The triangle projection's accumulators; see the note in pair-track-gpu.js.
    const accumulatePrecision = this.options?.accumulatePrecision ?? (hasF16 ? "f16" : "f32");
    if ((weightPrecision === "f16" || stagedPrecision === "f16") && !hasF16) {
      throw new Error("f16 weights and staged tiles require the shader-f16 feature");
    }
    const pipelines = await compilePairTrack(this.pipelines, {
      n, sample: blocks[0], epsilon, variance, dialect, base, stagedPrecision,
      weightPrecision: pairWeightPrecision, accumulatePrecision,
    });
    // 🔴 COMPILED CONCURRENTLY - see the note in pair-track-gpu.js. `compile`
    // returns the cache's promise and the awaits are collected, so these
    // overlap instead of serialising on the trunk's first pass.
    const compile = (key, source) => this.pipelines.get(key, source);
    const compiling = [];
    const into = (slot, key, source) => {
      compiling.push(compile(key, source).then((pipeline) => { pipelines[slot] = pipeline; }));
    };

    const singleTransitionOffsets = packTransitionWeights(blocks[0].singleTransition).offsets;
    const singleOffsets = packSingleAttentionWeights(blocks[0].singleAttention).offsets;
    const logitsOffsets = packPairLogitsWeights({
      scale: blocks[0].singlePairLogitsNormScale,
      offset: blocks[0].singlePairLogitsNormOffset,
      projection: blocks[0].singlePairLogitsProjection,
    }).offsets;
    into("singleTransition", `${base}:single-transition:${weightPrecision}`,
      createTransitionShader({ rows: n, channels: SINGLE_CHANNELS, factor: 4, weightPrecision },
                             singleTransitionOffsets, epsilon, variance));
    const { projectSplits, ...singleSources } = createSingleAttentionShaders(
      { n, channels: SINGLE_CHANNELS, heads, dimension: blocks[0].singleAttention.dimension,
        weightPrecision },
      singleOffsets, epsilon, variance);
    // ...the dispatch multiplies by this; see the note on PROJECT_SPLITS.
    pipelines.singleProjectSplits = projectSplits;
    for (const [name, source] of Object.entries(singleSources)) {
      into(`single:${name}`, `${base}:single:${weightPrecision}:${name}`, source);
    }
    into("pairLogits", `${base}:pair-logits`,
      createPairLogitsShader(n, PAIR_CHANNELS, heads, logitsOffsets, epsilon, variance));
    into("addSingle", `${base}:add-single`, createAddShader(n * SINGLE_CHANNELS));
    await Promise.all(compiling);

    try {
      // Resident state.
      const pair = keep(this.allocator.upload("af3-block.pair", state.pair, storage | GPUBufferUsage.COPY_SRC));
      const single = keep(this.allocator.upload("af3-block.single", state.single, storage | GPUBufferUsage.COPY_SRC));
      const pairMask = keep(this.allocator.upload("af3-block.pair-mask", state.pairMask, storage));
      const seqMask = keep(this.allocator.upload("af3-block.seq-mask", state.seqMask, storage));

      // Six pair-sized scratch buffers, shared by every operation. Their
      // storage is UNPACKED_PAIR_SCRATCH's to say, and the shaders read the
      // same array - a buffer half the bytes of what a shader expects is not
      // something WebGPU can catch. See that constant for what packing them
      // cost: a factor of 1200 on the pair representation this stack produces,
      // which is what the confidence head reads.
      const scratch = [];
      for (let index = 0; index < PAIR_SCRATCH_COUNT; index += 1) {
        scratch.push(keep(this.allocator.allocate(
          `af3-block.scratch${index}`,
          storageBytes(pairs * PAIR_CHANNELS, UNPACKED_PAIR_SCRATCH[index]), storage)));
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

      // 🔴 SUBMISSION RUNS AHEAD OF THE DEVICE, ON PURPOSE. Every block used to
      // submit, await a validation error scope and await an empty queue before
      // releasing its weights - two full GPU stalls per block, 48 times, so
      // nothing ever pipelined and a 68-token pairformer took 72 ms a block
      // where AF2's fatter evoformer block takes about 10. The blocks are now
      // encoded and submitted back to back; the queue keeps them in order.
      //
      // The window exists so the queue does not grow without bound and so an
      // abort can land, not because the memory needs it - the weights are
      // released as soon as each block is submitted, on queue ordering.
      //
      // 🔴 SIXTEEN IS MEASURED. Sweeping it over one 59-token stack: 1 gives
      // 881 ms, 4 gives 662, 8 gives 622, 16 gives 609 and 48 gives 607. Each
      // wait is a full pipeline drain, so a narrow window spends most of the
      // stack refilling; past sixteen the curve is flat and the remaining
      // waits are cheap insurance against an unbounded queue.
      const submissionWindow = options.submissionWindow ?? 16;
      const validation = new DeferredValidation(this.device, "AF3 pairformer stack");
      const start = performance.now();
      // 🔴 WHERE THE STACK'S WALL TIME ACTUALLY GOES, REPORTED RATHER THAN
      // GUESSED. This was needed to settle whether the trunk is compute bound:
      // the labelled compute passes summed to well under the wall clock, which
      // looked like per-block overhead, and the answer is that it is not - the
      // host encodes a whole 48-block pass in about 5 ms and spends the rest
      // inside onSubmittedWorkDone. Kept because the next person to see that
      // gap will otherwise re-derive it.
      let encodeMilliseconds = 0;
      let waitMilliseconds = 0;
      let releaseMilliseconds = 0;
      for (let index = 0; index < blocks.length; index += 1) {
        const pending = [];
        validation.begin();
        const encodeStart = performance.now();
        await this.#encodeBlock({
          block: blocks[index], n, pairs, heads, gridHeads, pipelines, storage, keep, pending,
          weightPrecision, pairWeightPrecision,
          pair, single, pairMask, seqMask, scratch, biasBuffer, pairLogits, singleScratch,
        });
        encodeMilliseconds += performance.now() - encodeStart;
        validation.end(`block ${index}`);
        const releaseStart = performance.now();
        for (let at = pending.length - 1; at >= 0; at -= 1) pending[at].release();
        releaseMilliseconds += performance.now() - releaseStart;
        // 🔴 WHEN THE DEVICE REACHES THIS BLOCK, reported without waiting for it.
        // This is AF2's idiom, from src/evoformer/stack.js, and it is here for
        // the same reason: onBlock above fires when a block is ENCODED, and
        // sixteen of those happen in the time the GPU takes over one - so a
        // status line driven by it sprints to the end of the window and then
        // sits still. onSubmittedWorkDone resolves once everything submitted so
        // far has finished, so one taken here settles exactly when this block
        // is done. It is NOT awaited: the loop carries on encoding and the
        // pipelining that makes this stack fast is untouched. They resolve in
        // submission order, so the count cannot go backwards.
        const submitted = index + 1;
        void this.device.queue.onSubmittedWorkDone()
          .then(() => options.onBlockDone?.(submitted, blocks.length));
        if ((index + 1) % submissionWindow === 0 || index === blocks.length - 1) {
          const waitStart = performance.now();
          await this.device.queue.onSubmittedWorkDone();
          waitMilliseconds += performance.now() - waitStart;
        }
        // 🔴 AWAITED, SO A CALLER CAN YIELD. Every await above resolves from a
        // GPU promise, which is a microtask - so a page that only updates a
        // progress bar here would write it and never paint it. Awaiting lets
        // the caller hand control back to the event loop for a frame.
        await options.onBlock?.(index);
      }
      await validation.settle();

      // 🔴 THE READBACKS ARE ALLOCATED AFTER THE SCRATCH IS GONE, NOT BEFORE
      // THE LOOP. They are written once, here, by a copy this stack has
      // already waited for - and a pair-sized MAP_READ buffer standing beside
      // six pair-sized scratch tensors for the whole 48-block loop is 43.9 MiB
      // of a 699 MiB peak at 300 tokens, held for nothing. `settle()` above
      // means the device is idle, so releasing the scratch here is safe and
      // this allocator does not pool: release DESTROYS, which is what makes
      // the peak move.
      for (const allocation of [...scratch, ...singleScratch, biasBuffer, pairLogits]) {
        allocation.release();
      }
      const readbackPair = keep(this.allocator.allocate(
        "af3-block.readback-pair", pairBytes,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      const readbackSingle = keep(this.allocator.allocate(
        "af3-block.readback-single", n * SINGLE_CHANNELS * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

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
        split: {
          encodeMilliseconds: Number(encodeMilliseconds.toFixed(1)),
          waitMilliseconds: Number(waitMilliseconds.toFixed(1)),
          releaseMilliseconds: Number(releaseMilliseconds.toFixed(1)),
        },
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

    // 🔴 RELEASED IMMEDIATELY, AND THAT IS SAFE BECAUSE THE QUEUE IS ORDERED.
    // Each block's weights are about 12 MB, so they cannot be held for all 48.
    // Releasing them used to mean draining the GPU first, on the reasoning that
    // the allocator RECYCLES memory and reusing a buffer the device is still
    // reading would be a race. It is not: allocator.upload writes through
    // device.queue.writeBuffer, which is ordered against previously submitted
    // work, so the recycled buffer is only overwritten after the commands
    // reading it have run. This is what AF2's evoformer stack has always done -
    // see the note there about submission running ahead of the device.
    const blockAllocations = context.pending;
    const upload = (label, data) => {
      const allocation = this.allocator.upload(label, data, storage);
      blockAllocations.push(allocation);
      return allocation;
    };
    // 🔴 PACKED ONCE PER BLOCK, EVER, AND THEN LET GO OF. Concatenating a
    // block's tensors into one Float32Array does not depend on anything but the
    // block, so doing it inside the encode loop meant redoing 35 ms of CPU work
    // on every pass, every recycle and every fold, in between GPU submissions
    // where it stalls the encoding rather than overlapping anything.
    //
    // 🔴 BUT A WeakMap KEEPING IT ALIVE COST 350 MiB OF HEAP FOR NOTHING. The
    // packed arrays exist to fill the resident device buffers below, and
    // residentWeightBuffer calls pack() only on a MISS - so after a block's
    // first encode nothing ever reads them again. Holding them made the page's
    // heap 1.1 GiB for a 59-token fold. This packs on demand instead: at most
    // once per block, and only while the misses are being filled.
    let packedPair;
    const packedFor = () => (packedPair ??= packPairTrackWeights(
      block, PAIR_CHANNELS, context.pairWeightPrecision));
    // 🔴 UPLOADED ONCE PER BLOCK, EVER, LIKE THE PACKING ABOVE. The packing was
    // already cached and the WRITE was not: eight buffers a block, 48 blocks, on
    // every pass of every recycle of every fold, over weights that never change.
    // src/runtime/resident.js keeps them on the device for the model's lifetime.
    // 🔴 RESIDENT IS A TRADE, AND WHICH SIDE OF IT IS RIGHT DEPENDS ON THE
    // MACHINE. Keeping all 48 blocks' weights on the device costs 567 MiB - 40%
    // of what a fold holds - and buys 30 ms a recycle, measured here at 59
    // tokens: 398 ms resident against 428 ms uploading per pass, with the
    // device at 567 MiB against 4 MiB. On this Mac that is worth paying. On a
    // 4 GiB phone, whose whole budget is 1.3 GiB, it is the difference between
    // folding and not.
    //
    // 🔴 SO IT IS NOT A CHOICE MADE IN ADVANCE. A guess needs an estimate of
    // what the stack will hold, and an estimate is a thing that goes stale
    // silently. Instead the fast path is tried and the budget answers: the
    // first allocation that would cross it raises GpuMemoryBudgetError before
    // createBuffer, and the stack drops to uploading per pass from there on -
    // for the rest of the run, since what did not fit will not fit later
    // either. A device with no budget set never takes this path at all.
    // The refusal is NOT caught here. See run(), which restarts the stack.
    const resident = this.residentWeights
      ? (label, pack, variant) => ({
        buffer: residentWeightBuffer(this.device, block, label, pack, variant),
      })
      : (label, pack) => ({ buffer: upload(label, pack()).buffer });
    const pairTrackWeights = {
      outgoing: resident("w.tri.out", () => packedFor().outgoing, context.pairWeightPrecision),
      incoming: resident("w.tri.in", () => packedFor().incoming, context.pairWeightPrecision),
      grid1: resident("w.grid1", () => packedFor().grid1),
      grid2: resident("w.grid2", () => packedFor().grid2),
      transition: resident(
        "w.pair-transition", () => packedFor().transition, context.pairWeightPrecision),
    };
    // 🔴 THE PRECISION IS PART OF THE CACHE KEY, as a variant rather than as
    // part of the label - see residentWeightBuffer. A process running both arms
    // would otherwise hand an f16 pipeline the f32 buffer.
    const singleTransitionWeights = resident(
      "w.single-transition",
      () => packTransitionWeights(block.singleTransition, context.weightPrecision).data,
      context.weightPrecision);
    const singleWeights = resident("w.single",
      () => packSingleAttentionWeights(block.singleAttention, context.weightPrecision).data,
      context.weightPrecision);
    const logitsWeights = resident("w.pair-logits", () => packPairLogitsWeights({
      scale: block.singlePairLogitsNormScale,
      offset: block.singlePairLogitsNormOffset,
      projection: block.singlePairLogitsProjection,
    }).data);
    // 🔴 AND NOW LET GO OF THE HOST'S COPY. Every buffer this block needs is on
    // the device and stays there for the model's lifetime, so the float32 the
    // packing read is dead - 11.7 MiB a block, 562 MiB over the stack, which is
    // what a lazily loaded weight object was holding. Releasing is a cache drop
    // and nothing more: a field read after this decodes again from the shard,
    // so a CPU reference sharing the same weights still works, just slower.
    if (this.residentWeights) releaseWeights(block);

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
    // ...the weights are bound twice, as scalars and as the vec4 view the head
    // projection is read through.
    run("pair-logits", pipelines.pairLogits,
        [pair, logitsWeights, logitsWeights, pairLogits], linear[0], linear[1]);

    run("single.project", pipelines["single:project"],
        [single, singleWeights, singleScratch[0], singleScratch[1], singleScratch[2],
         singleScratch[3]], n * pipelines.singleProjectSplits);
    run("single.attend", pipelines["single:attend"],
        [singleScratch[0], singleScratch[1], singleScratch[2], pairLogits, seqMask,
         singleScratch[4]], n * heads);
    run("single.project-out", pipelines["single:project_out"],
        [singleScratch[4], singleScratch[3], singleWeights, singleScratch[5]], n);
    const addSingle = spread(ceil(n * SINGLE_CHANNELS, 64));
    run("single.add", pipelines.addSingle, [single, singleScratch[5]], addSingle[0], addSingle[1]);

    run("single-transition", pipelines.singleTransition,
        [single, singleTransitionWeights, singleScratch[0]], ceil(n, transitionRowTile(n)));
    run("single-transition.add", pipelines.addSingle, [single, singleScratch[0]],
        addSingle[0], addSingle[1]);

    // Submitted, not awaited: the queue keeps the work in order, and the batch
    // this block belongs to drains once for all of them.
    this.device.queue.submit([encoder.finish()]);
  }
}
