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
import { GpuBufferAllocator } from "../../runtime/allocator.js";
import { pipelineCacheForDevice } from "../../runtime/pipeline-cache.js";
import { Af3EmbedderGpu } from "./embedder-webgpu.js";
import { Af3MsaStackGpu } from "./msa-stack-webgpu.js";
import { Af3PairformerStackGpu } from "./pairformer-block-webgpu.js";
import { Af3TemplateEmbedderGpu } from "./template-webgpu.js";
import { GRID_WIDTH, PAIR_CHANNELS, createAddShader } from "./pair-track-gpu.js";
import { DeferredValidation } from "../../runtime/validation.js";
import { af3ContactBins } from "../featurise/contact-classes.js";

/**
 * AlphaFold 3's own bin count, and the DEFAULT rather than the rule.
 *
 * 🔴 OpenDDE'S DISTOGRAM HAS 96 BINS, and nothing but `half_logits`' own shape
 * says so. Upstream leaves the two BREAKS at AlphaFold 3's 2.3125 and 21.6875
 * and changes only the count, so OpenDDE's grid is 95 edges over the same span
 * - a finer grid of the same reach, not a longer one. That is upstream's
 * reading and it is the only stated source: OpenDDE's published config.json is
 * metadata and carries no distogram range. See docs/OPENDDE.md.
 */
const NUM_BINS = 64;
const FIRST_BREAK = 2.3125;
const LAST_BREAK = 21.6875;
const CONTACT_THRESHOLD = 8.0 + 1e-3;

/** The distogram bin edges: 63 of them, evenly spaced. */
export function binEdges(bins = NUM_BINS) {
  const breaks = new Float32Array(bins - 1);
  for (let index = 0; index < bins - 1; index += 1) {
    breaks[index] = FIRST_BREAK + (LAST_BREAK - FIRST_BREAK) * index / (bins - 2);
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
 *
 * 🔴 AND A TRAINED BIAS ENTERS THIS SUM TWICE, WHICH IS WHY IT IS NOT SIMPLY
 * ADDED. Stock AlphaFold 3's `half_logits` is bias-free; OpenDDE's carries one,
 * and this head computes `half(i,j) + half(j,i)` - so a bias that is added once
 * per half is added twice per logit. OpenDDE symmetrises AFTER its own linear
 * exactly as this does, so the doubling is what its training saw and the
 * converter passes the bias straight across. Two of the four families upstream
 * lists need their bias HALVED instead, because their natives symmetrise the
 * pair FIRST - so this is a fact about the checkpoint, not about the name of
 * the tensor.
 */
export function createDistogramShader(tokens, channels, offset,
                                      { bins = NUM_BINS, biasOffset = -1 } = {}) {
  return `
const TOKENS: u32 = ${tokens}u;
const PAIRS: u32 = ${tokens * tokens}u;
const CHANNELS: u32 = ${channels}u;
const BINS: u32 = ${bins}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const W_HALF: u32 = ${offset}u;

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> pair_mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
// 🔴 HOW MANY BINS COUNT AS CONTACT IS PER PAIR, NOT A CONSTANT, because 8 A
// is a pseudo-beta convention and AF3 tokenises ligands one heavy atom at a
// time. See ../heads/contact-threshold.js; the bins are ordered, so a prefix
// length says it.
@group(0) @binding(3) var<storage, read> contact_bins: array<i32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<storage, read_write> contact: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let i = row / TOKENS;
  let j = row % TOKENS;
  let transposed = j * TOKENS + i;

  var largest = -3.0e38;
  var values: array<f32, ${bins}>;
  for (var b = 0u; b < BINS; b += 1u) {
    var total = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      // ...one half plus its own transpose.
      total += (pair[row * CHANNELS + c] + pair[transposed * CHANNELS + c])
        * weights[W_HALF + c * BINS + b];
    }
${biasOffset >= 0 ? `    total += 2.0 * weights[${biasOffset}u + b];\n` : ""}\
    values[b] = total;
    logits[row * BINS + b] = total;
    largest = max(largest, total);
  }

  var sum = 0.0;
  var contact_total = 0.0;
  let near_bins = u32(contact_bins[row]);
  for (var b = 0u; b < BINS; b += 1u) {
    let probability = exp(values[b] - largest);
    sum += probability;
    if (b < near_bins) { contact_total += probability; }
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
    // 🔴 ANNOUNCED BEFORE IT RUNS, NOT ONLY AFTER IT FINISHES. `onStage` fires
    // with a duration, so it can only ever mark a stage that is over - and on a
    // large protein the MSA stack alone is seconds, which the page spent
    // sitting on "Trunk · 0%" with nothing to say. `onStageStart` is a separate
    // signal on purpose: every existing consumer of `onStage` reads the
    // millisecond figure (tools/gpu/fold.js prints `detail.ms.toFixed(0)`), and
    // firing that with no duration would break them rather than inform them.
    const stage = async (name, work) => {
      options.onStageStart?.(name);
      const start = performance.now();
      const value = await work();
      timings[name] = performance.now() - start;
      options.onStage?.(name, timings[name]);
      return value;
    };

    // 🔴 THE SEAMS, FOR A CHECKER THAT HAS THE REFERENCE'S OWN. Every trunk gate
    // here compares the GPU against this port's CPU reference, so the two can be
    // wrong together - which is how boltz2's `target_feat` stayed 1.00e+0 from
    // af3-any-model's while every checker passed. `onSeam` is off unless a
    // caller asks, and costs a readback when it is on; see `fold.js
    // --trunk-oracle=` and tools/oracle/dump_af3_trunk_taps.py, whose tap names
    // these match.
    const seam = (name, value) => options.onSeam?.(name, value);

    // 🔴 THE PAIR, MSA AND SINGLE STAY ON THE DEVICE BETWEEN STAGES. Each stage
    // used to take a host array, upload it, compute, and read it back - and a
    // readback is a DRAIN, so every stage boundary stopped the GPU, crossed the
    // bus twice with the pair (32 MiB at 255 tokens, 82 at 400) and started it
    // again. Measured at 255 tokens under stock flags: a pass was 1050 ms of
    // wall for 587 of GPU, the embedder, template, MSA stack and distogram
    // doing ~80 ms of GPU work in ~420 ms of wall. The arithmetic is untouched
    // - the same kernels, the same f32 adds, now on the device - so this path
    // is held to BIT-IDENTICAL output. Validation is collected rather than
    // awaited per stage, because in Dawn an awaited `popErrorScope` resolves
    // when the submitted work does, which is a drain by another name; it is
    // settled once, at the readback that already synchronises.
    //
    // 🔴 AND THE SEAMS READ BACK ONLY WHEN A CHECKER ASKS, from these same
    // buffers - so the oracle compares the path the page runs, not a second
    // host path kept alive beside it.
    const validation = new DeferredValidation(this.device, "AF3 trunk");
    const read = async (allocation, elements) => {
      const staging = this.allocator.allocate("af3-trunk.seam-readback", elements * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      try {
        const encoder = this.device.createCommandEncoder({ label: "af3-trunk.seam" });
        encoder.copyBufferToBuffer(allocation.buffer, 0, staging.buffer, 0, elements * 4);
        this.device.queue.submit([encoder.finish()]);
        await staging.buffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(staging.buffer.getMappedRange().slice(0));
        staging.buffer.unmap();
        return copy;
      } finally {
        staging.release();
      }
    };
    const readSeam = async (name, allocation, elements) => {
      if (options.onSeam === undefined) return undefined;
      const value = await read(allocation, elements);
      seam(name, value);
      return value;
    };
    const pairChannels = weights.embedder.pairChannels;
    const pairElements = pairs * pairChannels;

    const embedded = await stage("embedder",
      () => new Af3EmbedderGpu(this.device).run(input, weights.embedder,
                                                { ...options, keepOnDevice: true, validation }));
    const owned = [embedded.pairAllocation, embedded.msaAllocation, embedded.singleAllocation];
    try {
      const pair = embedded.pairAllocation;
      // 🔴 A SNAPSHOT, because the template stage adds INTO this same buffer.
      // The difference taken afterwards against the live pair would be the
      // tensor minus itself - it read rms 0.0000 once, on the host path.
      const zInitSnapshot = await readSeam("tap.z_init_generic", pair, pairElements);
      await readSeam("tap.trunk_in_single", embedded.singleAllocation,
                     tokens * weights.embedder.singleChannels);

      // 🔴 ON THE PART-BUILT PAIR - see the note at the top.
      // 🔴 THE FOUR PINS AlphaFold 3's CONFIDENCE HEAD CARRIES, AND FOR THE SAME
      // REASON: this stage is 1.7% of a trunk and its output is added to z, so it
      // is paid for once and inherited by all 48 pairformer blocks. boltz2's
      // fused embedder reads 5.11e-4 against its own CPU reference with the
      // shipped precision and **9.11e-7** with these - three orders - and
      // measured interleaved at 256 tokens the stage is 8.0 ms either way against
      // a 468 ms trunk, four runs, no arm above 8.1. Correctness that costs
      // nothing measurable is not a trade.
      //
      // 🔴 AND `--f16=off` AND `--tune=` CANNOT REACH THEM. The pins go through
      // the CONSTRUCTOR; three `--tune` arms read an identical 5.11e-4 and said
      // only that they had not run. See docs/AF3.md.
      await stage("template", () => new Af3TemplateEmbedderGpu(this.device, {
        ...this.options,
        stagedPrecision: "f32", weightPrecision: "f32", accumulatePrecision: "f32",
        pairMatrixKernels: false,
      }).run(
        { pairMask: input.pairMask, tokens,
          templates: input.templates ?? 4,
          // Absent, every slot is empty - which is what a de novo fold has, and
          // is still a quarter of what enters the MSA stack.
          slots: input.templateSlots,
          // 🔴 THE CHAIN IDS, NOT A MASK. AF3 masks the template's geometry
          // ACROSS chains - two chains' templates were never in one coordinate
          // frame - and the embedder derives that per slot. Passing nothing used
          // to mean "assume one chain", which on a two-chain fold scored relRMS
          // 1.09 against AF3.
          asymId: input.asymId,
          multichainMask2d: input.multichainMask2d },
        // 🔴 THE TEMPLATE TERM IS ADDED INTO THE EMBEDDER'S PAIR, on the device
        // now: `pairBuffer` is read as z and updated in place as z + term.
        weights.template, dialect, { ...options, pairBuffer: pair.buffer, validation }));
      const afterTemplate = await readSeam("tap.z_after_template", pair, pairElements);
      // ...and the template module's OWN output, which is what the reference
      // traces as `evoformer/template_embedding`. `z_after_template` is
      // `z_init + term`, so with an exact z_init the two say the same thing -
      // but only the difference can be compared against the module's scope, and
      // a term that is 4e-3 wrong inside a sum that is 3.9e-3 wrong is worth
      // stating as itself.
      if (options.onSeam !== undefined) {
        const term = new Float32Array(pairElements);
        for (let i = 0; i < term.length; i += 1) term[i] = afterTemplate[i] - zInitSnapshot[i];
        seam("tap.template_term", term);
      }

      // 🔴 THE MSA EMBEDDING, BEFORE THE STACK TOUCHES IT. `z_after_msa` is the
      // only MSA seam there was, so a wrong FEATURE and a wrong STACK were the
      // same number. The reference records `evoformer/msa_activations` at exactly
      // this point.
      const msaElements = input.sequences * tokens * weights.embedder.msaChannels;
      await readSeam("tap.msa_activations", embedded.msaAllocation, msaElements);

      // 🔴 boltz2 ADDS THE PRE-MSA PAIR BACK. See `msaDoubleAddPair` in
      // dialect.js: its MSAModule returns the updated z and its caller adds z to
      // that, so what reaches the pairformer is `2 * z_in + delta`. The stack
      // updates the pair IN PLACE now, so z_in is copied aside first.
      const doubleAdd = dialect.msaDoubleAddPair === true;
      const preMsa = doubleAdd ? this.allocator.allocate("af3-trunk.pre-msa-pair",
        pairElements * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) : undefined;
      if (preMsa !== undefined) owned.push(preMsa);
      if (doubleAdd) {
        const encoder = this.device.createCommandEncoder({ label: "af3-trunk.pre-msa-copy" });
        encoder.copyBufferToBuffer(pair.buffer, 0, preMsa.buffer, 0, pairElements * 4);
        this.device.queue.submit([encoder.finish()]);
      }
      await stage("msa-stack", () => new Af3MsaStackGpu(this.device, this.options).run(
        { pairMask: input.pairMask, msaMask: input.msaMask, tokens, sequences: input.sequences },
        weights.msaBlocks, dialect,
        { ...options, stopAfterOpm: options.stopAfterOpm === true,
          pairBuffer: pair.buffer, msaBuffer: embedded.msaAllocation.buffer, validation }));
      if (doubleAdd) {
        const add = await this.pipelines.get(`af3-trunk:add:${pairElements}`,
                                             createAddShader(pairElements));
        validation.begin();
        const encoder = this.device.createCommandEncoder({ label: "af3-trunk.msa-double-add" });
        const pass = encoder.beginComputePass({ label: "trunk.msa-double-add" });
        pass.setPipeline(add);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: add.getBindGroupLayout(0),
          entries: [pair, preMsa].map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer } })),
        }));
        const groups = Math.ceil(pairElements / 64);
        pass.dispatchWorkgroups(Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        validation.end("msa double add");
      }
      // ...and the MSA tensor the stack produced, which the reference taps as
      // `msa_block_msa_act`. With one block it is that block's updated MSA, and
      // it is what separates a wrong MSA update from a wrong outer product.
      await readSeam("tap.msa_block_msa_act", embedded.msaAllocation, msaElements);
      await readSeam("tap.z_after_msa", pair, pairElements);

      // 🔴 THE ONLY STAGE WORTH A PROGRESS BAR. The pairformer is 48 blocks and
      // the bulk of the trunk; the other four stages are each a fraction of it,
      // so a bar that only moved between stages would sit still for most of the
      // wait. onBlock is reported under its own name rather than through
      // `options`, which is passed to every sub-stack and would otherwise fire
      // for the template's blocks too.
      // 🔴 THE HEAD IS PREPARED BEFORE THE PAIRFORMER RUNS, because nothing it
      // packs depends on the pair's values - so its host work (weights, contact
      // bins, uploads) happens while the GPU is still busy rather than after
      // the pass has drained.
      const head = await this.#prepareDistogram(input.pairMask, tokens,
                                                weights.distogram, input.contactClasses);
      owned.push(...head.allocations);
      const pairformer = await stage("pairformer", () => new Af3PairformerStackGpu(this.device, this.options).run(
        { pairMask: input.pairMask, seqMask: input.seqMask, tokens },
        weights.pairformerBlocks, dialect, {
          ...options,
          pairBuffer: pair.buffer, singleBuffer: embedded.singleAllocation.buffer,
          deferReadback: true, validation,
          onBlock: (index) => options.onPairformerBlock?.(index,
                                                          weights.pairformerBlocks.length),
          // ...and the one that says the device GOT there, which is what a status
          // line should show. See the note in pairformer-block-webgpu.js.
          onBlockDone: (completed, total) => options.onPairformerBlockDone?.(completed, total),
        }));
      // The pairformer's own encode/wait split, carried out so a bench can report
      // where the stack's wall time went without re-instrumenting it.
      this.lastPairformerSplit = pairformer.split;

      // 🔴 ONE SUBMIT AND ONE DRAIN TO END THE PASS: the head's dispatch, then
      // the pair, single, logits and contacts copied out together. It used to
      // be the pairformer's readback, then the head's host work, then the
      // head's own readback - two drains with the GPU idle between them.
      const singleElements = tokens * weights.embedder.singleChannels;
      const out = await stage("distogram", () => head.run(pair, embedded.singleAllocation,
                                                          pairElements, singleElements));
      // Every scope collected above was submitted ahead of that readback, which
      // has completed, so these resolve without waiting.
      await validation.settle();
      seam("tap.trunk_out_pair", out.pair);

      return {
        pair: out.pair, single: out.single,
        logits: out.logits, contactProbs: out.contactProbs,
        binEdges: binEdges(weights.distogram.bins ?? NUM_BINS), timings,
      };
    } finally {
      for (const allocation of owned) allocation.release();
    }
  }

  /**
   * Everything the distogram head needs that does not depend on the pair's
   * values: its pipeline, packed weights, contact bins and output buffers.
   * `run` encodes the head over the trunk's pair and reads the pair, single,
   * logits and contacts back in ONE submit. The allocations are the caller's.
   */
  async #prepareDistogram(pairMask, tokens, weights, contactClasses) {
    const pairs = tokens * tokens;
    // 🔴 THE HEAD'S THREE NUMBERS ARE THE TENSOR'S. `half_logits` is
    // [pairChannels, bins], so AlphaFold 3's 128x64 and OpenDDE's 384x96 are
    // both read off it rather than declared - see src/af3/weights/weights.js. The bias
    // is the fourth, and it exists only where the checkpoint trained one.
    const channels = weights.pairChannels ?? PAIR_CHANNELS;
    const bins = weights.bins ?? NUM_BINS;
    const bias = weights.halfLogitsBias;
    if (bias !== undefined && bias.length !== bins) {
      throw new Error(`distogram bias has ${bias.length} entries; expected ${bins}`);
    }
    const biasOffset = bias === undefined ? -1 : weights.halfLogits.length;
    const packed = new Float32Array(weights.halfLogits.length + (bias?.length ?? 0));
    packed.set(weights.halfLogits, 0);
    if (bias !== undefined) packed.set(bias, biasOffset);
    const pipeline = await this.pipelines.get(
      `af3-distogram:${tokens}:${channels}:${bins}:${biasOffset}`,
      createDistogramShader(tokens, channels, 0, { bins, biasOffset }));

    // 🔴 REQUIRED, NOT DEFAULTED. A caller with no classes would silently get
    // 8 A everywhere back, which is the convention this exists to correct -
    // and the failure would be a plausible contact map.
    if (contactClasses === undefined || contactClasses.length !== tokens) {
      throw new Error("the distogram head needs contactClasses, one per token");
    }
    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    const maskBuffer = keep(this.allocator.upload("af3-disto.mask", pairMask, storage));
    // 🔴 THE EDGES ARE THIS HEAD'S, NOT THE DEFAULT'S. `af3ContactBins` turns
    // a per-pair angstrom threshold into a COUNT of bins, so handing it
    // AlphaFold 3's 64-bin grid for OpenDDE's 96-bin head returns a count
    // against the wrong ruler - and a count is what the shader compares, so
    // nothing would be out of range and the contact map would simply be
    // wrong. Same shape of mistake as the AF2 manifests that carried 2 and 22
    // where the head's breaks are 2.3125 and 21.6875.
    const binsBuffer = keep(this.allocator.upload("af3-disto.contact-bins",
      af3ContactBins(contactClasses, tokens, binEdges(bins)), storage));
    const weightBuffer = keep(this.allocator.upload("af3-disto.weights", packed, storage));
    const logits = keep(this.allocator.allocate("af3-disto.logits", pairs * bins * 4,
      storage | GPUBufferUsage.COPY_SRC));
    const contact = keep(this.allocator.allocate("af3-disto.contact", pairs * 4,
      storage | GPUBufferUsage.COPY_SRC));

    const run = async (pairAllocation, singleAllocation, pairElements, singleElements) => {
      const mapRead = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
      const readback = [
        [logits, pairs * bins], [contact, pairs],
        [pairAllocation, pairElements], [singleAllocation, singleElements],
      ].map(([source, elements]) => ({
        source, bytes: elements * 4,
        target: this.allocator.allocate("af3-disto.readback", elements * 4, mapRead),
      }));
      try {
      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-distogram" });
      const pass = encoder.beginComputePass({ label: "af3-distogram" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [pairAllocation, maskBuffer, weightBuffer, binsBuffer,
                  logits, contact].map(
          (allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
      }));
      const groups = Math.ceil(pairs / 64);
      pass.dispatchWorkgroups(Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH));
      pass.end();
      for (const { source, target, bytes } of readback) {
        encoder.copyBufferToBuffer(source.buffer, 0, target.buffer, 0, bytes);
      }
      this.device.queue.submit([encoder.finish()]);
      const scope = this.device.popErrorScope();
      await Promise.all(readback.map(({ target }) => target.buffer.mapAsync(GPUMapMode.READ)));
      const error = await scope;
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      const [outLogits, contactProbs, outPair, outSingle] = readback.map(({ target }) => {
        const copy = new Float32Array(target.buffer.getMappedRange().slice(0));
        target.buffer.unmap();
        return copy;
      });
      return { logits: outLogits, contactProbs, pair: outPair, single: outSingle };
      } finally {
        for (const { target } of readback) target.release();
      }
    };
    return { run, allocations };
  }
}
