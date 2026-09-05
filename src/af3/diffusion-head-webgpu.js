/**
 * AF3's diffusion head on the GPU: one denoising step.
 *
 *     conditioning(trunk, noise level)
 *     atom encoder(noisy positions * input scaling)
 *     + projected single conditioning
 *     24-block token transformer
 *     atom decoder
 *     -> skip * noisy + out * update, masked
 *
 * 🔴 THE OUTPUT IS A BLEND, NOT A PREDICTION. AF3 returns
 * `skip * positionsNoisy + out * update`, the EDM preconditioning - so at high
 * noise the network's update dominates and at low noise the input does.
 * Returning the update alone runs, produces coordinates, and denoises far too
 * aggressively at the end of the schedule.
 *
 * 🔴 THE POSITIONS ARE MASKED AND RESCALED BY THE NOISE LEVEL BEFORE THE
 * ENCODER SEES THEM. Feeding raw angstroms works at low noise and saturates at
 * high, which is the failure that looks like "the model is fine, the schedule
 * is wrong".
 *
 * 🔴 THE ATOMS ARE CONDITIONED BY THE TRUNK'S SINGLE, AND THE TOKEN TRANSFORMER
 * BY THE CONDITIONING MODULE'S. They are different tensors and AF3 uses both -
 * the encoder gets `trunkSingle`, the transformer gets `cond.single`. Passing
 * one where the other belongs type-checks and is a different model.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { Af3DiffusionConditioningGpu } from "./diffusion-conditioning-webgpu.js";
import { Af3AtomEncoderGpu } from "./atom-encoder-webgpu.js";
import { Af3AtomDecoderGpu } from "./atom-decoder-webgpu.js";
import { Af3DiffusionTransformerGpu } from "./diffusion-transformer-webgpu.js";
import { DeferredValidation } from "../runtime/validation.js";
import { noteAllocation, noteDestroy } from "../runtime/device-memory.js";
import { residentWeightBuffer } from "../runtime/resident.js";

const SIGMA_DATA = 16.0;

/** The EDM preconditioning weights for a noise level. */
export function scalings(noiseLevel) {
  const denominator = noiseLevel * noiseLevel + SIGMA_DATA * SIGMA_DATA;
  return {
    skip: SIGMA_DATA * SIGMA_DATA / denominator,
    out: noiseLevel * SIGMA_DATA / Math.sqrt(denominator),
    input: 1 / Math.sqrt(denominator),
  };
}

/**
 * layerNormSlow(x, scale, null) then a projection - used twice, for the single
 * conditioning going into the transformer and for the transformer's output.
 */
function normaliseAndProject(input, rows, channels, outChannels, scale, projection) {
  const output = new Float32Array(rows * (projection === null ? channels : outChannels));
  for (let row = 0; row < rows; row += 1) {
    const base = row * channels;
    let total = 0;
    for (let c = 0; c < channels; c += 1) total += input[base + c];
    const mean = total / channels;
    let variance = 0;
    for (let c = 0; c < channels; c += 1) {
      const d = input[base + c] - mean;
      variance += d * d;
    }
    const inverse = 1 / Math.sqrt(variance / channels + 1e-5);
    if (projection === null) {
      for (let c = 0; c < channels; c += 1) {
        output[base + c] = (input[base + c] - mean) * inverse * scale[c];
      }
      continue;
    }
    for (let out = 0; out < outChannels; out += 1) {
      let value = 0;
      for (let c = 0; c < channels; c += 1) {
        value += (input[base + c] - mean) * inverse * scale[c] * projection[c * outChannels + out];
      }
      output[row * outChannels + out] = value;
    }
  }
  return output;
}


/**
 * LayerNorm with a scale and no offset, then a projection. One workgroup a row.
 *
 * 🔴 THIS WAS 58 MS OF A 204 MS DENOISER CALL, IN JAVASCRIPT. The single
 * conditioning is 384 wide and the token transformer wants 768, so this is a
 * 59x384x768 matmul - 17M multiply-adds in a scalar loop on the main thread,
 * two hundred times a fold, while the GPU sat idle. It is the last big piece of
 * the head that was not a shader.
 */
const NORMALISE_AND_PROJECT = (rows, inChannels, outChannels, lanes) => `
const ROWS: u32 = ${rows}u;
const C_IN: u32 = ${inChannels}u;
const C_OUT: u32 = ${outChannels}u;
const LANES: u32 = ${lanes}u;
const EPSILON: f32 = 1.0e-5;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> projection: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> normalised: array<f32, ${inChannels}>;
var<workgroup> reduce_a: array<f32, ${lanes}>;

fn reduce_sum(local: u32, value: f32) -> f32 {
  reduce_a[local] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_a[local] += reduce_a[local + stride]; }
    workgroupBarrier();
  }
  return reduce_a[0];
}

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x;
  if (row >= ROWS) { return; }
  let local = local_id.x;
  let base = row * C_IN;

  var total = 0.0;
  for (var c = local; c < C_IN; c += LANES) { total += input[base + c]; }
  let mean = reduce_sum(local, total) / f32(C_IN);
  workgroupBarrier();
  var centred = 0.0;
  for (var c = local; c < C_IN; c += LANES) {
    let d = input[base + c] - mean;
    centred += d * d;
  }
  let inverse = inverseSqrt(reduce_sum(local, centred) / f32(C_IN) + EPSILON);
  workgroupBarrier();
  for (var c = local; c < C_IN; c += LANES) {
    normalised[c] = (input[base + c] - mean) * inverse * scale[c];
  }
  workgroupBarrier();

  for (var out = local; out < C_OUT; out += LANES) {
    var value = 0.0;
    for (var c = 0u; c < C_IN; c += 1u) {
      value += normalised[c] * projection[c * C_OUT + out];
    }
    output[row * C_OUT + out] = value;
  }
}`;

/**
 * The same LayerNorm with no projection after it: one workgroup a row.
 *
 * 🔴 IT WAS ON THE CPU AND MEASURED AT 0 MS, WHICH WAS NOT THE REASON TO MOVE
 * IT. 59 rows of 768 really is nothing; what it cost was the READBACK in front
 * of it - the transformer's output had to reach the host for this loop to run,
 * and then go back for the decoder. The arithmetic is free either way; the
 * round trip was not.
 */
const NORMALISE_ONLY = (rows, channels, lanes) => `
const ROWS: u32 = ${rows}u;
const C: u32 = ${channels}u;
const LANES: u32 = ${lanes}u;
const EPSILON: f32 = 1.0e-5;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> reduce_a: array<f32, ${lanes}>;

fn reduce_sum(local: u32, value: f32) -> f32 {
  reduce_a[local] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_a[local] += reduce_a[local + stride]; }
    workgroupBarrier();
  }
  return reduce_a[0];
}

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x;
  if (row >= ROWS) { return; }
  let local = local_id.x;
  let base = row * C;

  var total = 0.0;
  for (var c = local; c < C; c += LANES) { total += input[base + c]; }
  let mean = reduce_sum(local, total) / f32(C);
  workgroupBarrier();
  var centred = 0.0;
  for (var c = local; c < C; c += LANES) {
    let d = input[base + c] - mean;
    centred += d * d;
  }
  let inverse = inverseSqrt(reduce_sum(local, centred) / f32(C) + EPSILON);
  workgroupBarrier();
  for (var c = local; c < C; c += LANES) {
    output[base + c] = (input[base + c] - mean) * inverse * scale[c];
  }
}`;

const ADD_IN_PLACE = (elements) => `
const ELEMENTS: u32 = ${elements}u;
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read_write> accumulator: array<f32>;
@group(0) @binding(1) var<storage, read> delta: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= ELEMENTS) { return; }
  accumulator[index] = accumulator[index] + delta[index];
}`;

export class Af3DiffusionHeadGpu {
  /** The last trunk's pair conditioning, which no noise level changes. */
  #conditioningPair;

  /** ...and the atom encoder's outputs that depend on it rather than on sigma. */
  #encoderStatic;

  /** The GPU buffers behind those, kept across calls and dropped with them. */
  #encoderBuffers = {};

  /**
   * ...and the decoder's, which are the same molecule seen from the other end:
   * its gathers, the encoder's conditioning and masks, and the pair logits it
   * derives from them. Held and dropped with the encoder's, because both are
   * invalidated by exactly one thing - a new trunk, which is a new fold.
   */
  #decoderBuffers = {};

  /**
   * The transformer, kept rather than rebuilt per call.
   *
   * 🔴 IT HOLDS THE NORMALISED PAIR CONDITIONING, and that is the point: a
   * fresh instance per step has nothing to remember, so the stack re-uploaded
   * and re-normalised a tokens^2 x 128 tensor on every one of them. See
   * Af3DiffusionTransformerGpu's #pairNorm.
   */
  #transformer;

  /**
   * The tensors that pass between this head's stages, kept on the device.
   *
   * 🔴 THE HEAD USED TO CHAIN ITS FIVE STAGES THROUGH Float32Arrays, AND EVERY
   * JOINT WAS A PIPELINE DRAIN. Each stage submitted, awaited its own error
   * scope - which resolves when the submitted work has FINISHED - and mapped a
   * readback, then the next stage wrote the same numbers back. At 59 tokens
   * the stages summed to 71 ms against about 52 ms of labelled compute, and
   * the difference was four of those round trips: the conditioning's single
   * track, the projection of it, the encoder's token activations and its skip
   * connection. None of them is ever LOOKED at on the host.
   *
   * 🔴 AND THE PAIR IS NOT AMONG THEM, ON PURPOSE. It is a per-TRUNK tensor
   * that both the encoder and the transformer already cache by the identity of
   * the host array, so the first call of a fold reads it back once and every
   * step after that costs nothing. That first call is also why the chain is
   * only used once the pair cache is warm: a device-chained first call would
   * have no host array to key those caches on.
   */
  #chain;

  /** ...and the conditioning module, kept so its allocator can pool. */
  #conditioner;

  /** ...and the atom encoder, for the same reason. */
  #encoder;

  /** ...and one pooled allocator for the single projection's two uploads. */
  #projections;

  /**
   * Release what this head is holding on the device.
   *
   * 🔴 THE STATIC CACHE OUTLIVES A CALL BY DESIGN AND MUST NOT OUTLIVE THE
   * FOLD. It is about 25 MB on a 59-residue chain - the atom pair conditioning
   * and its logits are most of it - and a head is built per sampler run, so
   * without this every fold would leave that behind until the collector
   * happened to notice. Both samplers call it in a finally.
   */
  dispose() {
    // 🔴 noteDestroy, NOT JUST destroy. These buffers are created through the
    // atom encoder's and decoder's `persistentUpload`, which calls
    // noteAllocation - so destroying them without the matching note leaves the
    // accounting believing they are still on the device, for ever. At 1530
    // tokens that is about 2.2 GB a fold: atom.trunk-pair alone is 1197 MiB,
    // and a real trace showed it "held" after the fold had finished.
    //
    // 🔴 AND IT IS NOT ONLY A WRONG NUMBER. `residentBytes` is what the budget
    // compares against, so every fold made the device look permanently fuller
    // than it is and the NEXT one is refused on memory that was given back -
    // an out-of-memory that is an accounting error. It also inflates
    // `peakBytes`, which is the figure anyone tuning memory reads.
    this.#releasePersistent();
    this.#encoderStatic = undefined;
    this.#conditioningPair = undefined;
    this.#transformer?.dispose();
    this.#transformer = undefined;
    this.#conditioner?.allocator.destroyPooled();
    this.#conditioner = undefined;
    this.#encoder?.allocator.destroyPooled();
    this.#encoder = undefined;
    this.#projections?.destroyPooled();
    this.#projections = undefined;
    this.#releaseChain();
  }

  /**
   * Give the atom encoder's and decoder's persistent buffers back, and SAY so.
   *
   * 🔴 THEY WERE DESTROYED WITHOUT THE NOTE, IN TWO PLACES. These are created
   * through `persistentUpload`, which calls noteAllocation - so destroying them
   * without the matching noteDestroy leaves the accounting believing they are
   * still on the device, for ever. At 1530 tokens that is about 2.2 GB a fold;
   * `atom.trunk-pair` alone is 1197 MiB, and a real trace showed it "held"
   * after the fold had finished and again in the peak the fold reported.
   *
   * 🔴 AND IT IS NOT ONLY A WRONG NUMBER. `residentBytes` is what the memory
   * BUDGET compares against, so every fold made the device look permanently
   * fuller than it is and the next one is refused on memory that had been
   * given back - an out-of-memory that is an accounting error. Measured at 200
   * tokens over two folds in one process, the second fold's held figure went
   * 524 -> 392 MiB.
   *
   * 🔴 ONE RELEASER, BECAUSE THERE WERE TWO CALLERS AND THEY DISAGREED. dispose
   * runs at the end of a fold and the cache invalidation runs when a new trunk
   * arrives mid-page; both destroyed the same buffers and neither noted it.
   */
  #releasePersistent() {
    for (const set of [this.#encoderBuffers, this.#decoderBuffers]) {
      for (const buffer of Object.values(set)) {
        noteDestroy(this.device, buffer.size, buffer.label || "atom.persistent");
        buffer.destroy();
      }
    }
    this.#encoderBuffers = {};
    this.#decoderBuffers = {};
  }

  #releaseChain() {
    if (this.#chain === undefined) return;
    for (const [label, entry] of Object.entries(this.#chain.buffers)) {
      entry.buffer.destroy();
      noteDestroy(this.device, entry.bytes, label);
    }
    this.#chain = undefined;
  }

  /**
   * A chain tensor, sized by the shape and kept until the shape moves.
   *
   * They are created outside the pooled allocators the stages use, because a
   * pooled buffer belongs to the call that took it and these outlive one.
   */
  #chainBuffer(key, label, bytes, extra = 0) {
    if (this.#chain?.key !== key) {
      this.#releaseChain();
      this.#chain = { key, buffers: {} };
    }
    const found = this.#chain.buffers[label];
    if (found !== undefined) return found.buffer;
    const size = Math.ceil(bytes / 4) * 4;
    noteAllocation(this.device, label, size);
    const buffer = this.device.createBuffer({
      label, size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra,
    });
    this.#chain.buffers[label] = { buffer, bytes: size };
    return buffer;
  }

  /**
   * The transformer's weight object, with this head's precision on it.
   *
   * 🔴 MEMOISED ON THE SOURCE, BECAUSE #compile KEYS ON IDENTITY. A fresh
   * spread per call misses that memo every time, and what it rebuilds is the
   * largest WGSL in the model - the very thing #compile exists to stop. The
   * default path passes `weights.transformer` straight through and never had
   * the problem; the arm that sets a precision explicitly, which is how a bench
   * compares f16 against f32, did.
   */
  #transformerWeights(weights) {
    const precision = this.options?.weightPrecision;
    if (precision === undefined) return weights.transformer;
    if (this.#transformerWeightsFor?.source !== weights.transformer) {
      this.#transformerWeightsFor = {
        source: weights.transformer,
        value: { ...weights.transformer, weightPrecision: precision },
      };
    }
    return this.#transformerWeightsFor.value;
  }

  /**
   * Compile the sampler's pipelines now, so the trunk pays for them.
   *
   * 🔴 NOTHING USED TO. fold.js builds the head above the recycle loop and its
   * comment claimed that compiled the pipelines; the constructor compiles
   * nothing, so the whole cost landed in the first denoiser call with the
   * trunk already finished. `createComputePipelineAsync` runs off the main
   * thread and the trunk leaves it idle - bench-trunk reports 9.4 ms of
   * encoding against 2948 of waiting - so this is free where it stands.
   *
   * It needs only the token count and the weights, neither of which the trunk
   * produces. Not awaited by the caller; the sampler's first call awaits the
   * same memoised promise.
   */
  async warm(tokens, weights) {
    this.#transformer ??= new Af3DiffusionTransformerGpu(this.device);
    await this.#transformer.warm(tokens, this.#transformerWeights(weights));
  }

  #transformerWeightsFor;

  constructor(device, options = {}) {
    this.device = device;
    // Only the transformer's resident weights read a precision here; see the
    // note in diffusion-transformer-webgpu.js for why it is the one stage where
    // narrowing them buys TIME as well as memory.
    this.options = options;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * One denoising step.
   *
   * @param {object} input shape, noiseLevel, positionsNoisy, atomMask, seqMask,
   *   trunkSingle, trunkPair, targetFeat, conditioning, refPos, refSpaceUid,
   *   features, and the five gathers
   * @param {object} weights conditioning, encoder, transformer, decoder, plus
   *   singleCondEmbedding* and outputNormScale
   */

  /** LayerNorm a device tensor into another, with the head's deferred scope. */
  async #normaliseOnly(input, into, rows, channels, scale, validation) {
    const lanes = 256;
    const pipeline = await pipelineCacheForDevice(this.device).get(
      `af3-normalise-only:${rows}:${channels}:${lanes}`,
      NORMALISE_ONLY(rows, channels, lanes));
    const scaleBuffer = residentWeightBuffer(this.device, scale, "cond.output-norm",
                                             () => scale);
    validation.begin();
    const encoder = this.device.createCommandEncoder({ label: "af3-output-norm" });
    const pass = encoder.beginComputePass({ label: "output-norm" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [input, scaleBuffer, into].map((buffer, binding) => ({
        binding, resource: { buffer },
      })),
    }));
    pass.dispatchWorkgroups(rows);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    validation.end("output norm");
  }

  /** `accumulator += delta`, elementwise, for two tensors already on the device. */
  async #addInto(accumulator, delta, elements, validation) {
    const pipeline = await pipelineCacheForDevice(this.device).get(
      `af3-head-add:${elements}`, ADD_IN_PLACE(elements));
    validation.begin();
    const encoder = this.device.createCommandEncoder({ label: "af3-head.add" });
    const pass = encoder.beginComputePass({ label: "head-add" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [accumulator, delta].map((buffer, binding) => ({
        binding, resource: { buffer },
      })),
    }));
    const groups = Math.ceil(elements / 64);
    pass.dispatchWorkgroups(Math.min(groups, 32768), Math.ceil(groups / 32768));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    validation.end("head add");
  }

  /**
   * The GPU form of normaliseAndProject, for the one call that is hot.
   *
   * `input` may be a Float32Array or a GPUBuffer; `into` a GPUBuffer to write
   * instead of reading the answer back. With both, this stage costs a submit
   * and nothing else - see #chain.
   */
  async #normaliseAndProject(input, rows, inChannels, outChannels, scale, projection,
                             { into, validation } = {}) {
    // 🔴 POOLED AND KEPT, NOT POOLED AND THROWN AWAY. With `into` this returns
    // while the work is in flight, so release() must not destroy what that
    // work is reading - and a pooled allocator created PER CALL leaks its
    // whole pool, because nothing ever destroys it and the device accounting
    // has no matching free. That cost 21.4 MiB of `np.projection` in a
    // 68-token fold: eighteen copies of one 384x768 matrix, one per step.
    const allocator = into === undefined
      ? new GpuBufferAllocator(this.device)
      : (this.#projections ??= new GpuBufferAllocator(this.device, true));
    const storage = GPUBufferUsage.STORAGE;
    const held = [];
    try {
      const lanes = 256;
      const pipeline = await pipelineCacheForDevice(this.device).get(
        `af3-normalise-project:${rows}:${inChannels}:${outChannels}:${lanes}`,
        NORMALISE_AND_PROJECT(rows, inChannels, outChannels, lanes));
      const keep = (allocation) => { held.push(allocation); return allocation; };
      const inputBuffer = input instanceof GPUBuffer
        ? { buffer: input } : keep(allocator.upload("np.input", input, storage));
      const scaleBuffer = keep(allocator.upload("np.scale", scale, storage));
      const weightBuffer = keep(allocator.upload("np.projection", projection, storage));
      const output = into !== undefined ? { buffer: into }
        : keep(allocator.allocate("np.output", rows * outChannels * 4,
            storage | GPUBufferUsage.COPY_SRC));
      const readback = into !== undefined ? undefined
        : keep(allocator.allocate("np.readback", rows * outChannels * 4,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      if (validation === undefined) this.device.pushErrorScope("validation");
      else validation.begin();
      const encoder = this.device.createCommandEncoder({ label: "af3-normalise-project" });
      const pass = encoder.beginComputePass({ label: "normalise-project" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [inputBuffer, scaleBuffer, weightBuffer, output].map((a, binding) => ({
          binding, resource: { buffer: a.buffer },
        })),
      }));
      pass.dispatchWorkgroups(rows);
      pass.end();
      if (into === undefined) {
        encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, rows * outChannels * 4);
      }
      this.device.queue.submit([encoder.finish()]);
      if (validation !== undefined) {
        validation.end("single projection");
        return undefined;
      }
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return result;
    } finally {
      for (let index = held.length - 1; index >= 0; index -= 1) held[index].release();
    }
  }

  async run(input, weights, options = {}) {
    const { tokens, dense } = input.shape;
    const scale = scalings(input.noiseLevel);
    const timings = {};
    const stage = async (name, work) => {
      const start = performance.now();
      const value = await work();
      timings[name] = performance.now() - start;
      options.onStage?.(name, timings[name]);
      return value;
    };

    // 🔴 THE PAIR CONDITIONING IS COMPUTED ONCE PER TRUNK, NOT ONCE PER CALL.
    // Nothing in it reads the noise level, so a 200-step sampler was building
    // the same tokens x tokens x 128 tensor two hundred times. The cache is on
    // the head instance and keyed by the trunk pair OBJECT: sampleOnGpu and
    // flowOnGpu construct one head and spread the same input through every
    // step, so identity is exactly the question "is this the same fold", and a
    // new fold brings a new array. Nothing here could tell a MUTATED array
    // apart, which is why the key is identity rather than a hash - the callers
    // never mutate one, and a hash would cost more than the work it saves.
    const cachedPair = this.#conditioningPair?.trunkPair === input.trunkPair
      && this.#conditioningPair?.tokens === tokens
      ? this.#conditioningPair.pair : undefined;

    // 🔴 THE CHAIN ONLY RUNS ONCE THE PAIR CACHE IS WARM. See #chain: the
    // first call of a fold has to read the pair conditioning back, because it
    // is the host array the encoder's and the transformer's own caches are
    // keyed on. That call is one step in two hundred.
    const chained = cachedPair !== undefined;
    const { subsets, queries } = input.shape;
    const queryRows = subsets * queries;
    const shapeKey = `${tokens}:${dense}:${queryRows}:${weights.encoder.channels}`
      + `:${weights.perTokenChannels}:${weights.seqChannels}`;
    const chain = chained ? {
      condSingle: this.#chainBuffer(shapeKey, "head.cond-single",
                                    tokens * weights.seqChannels * 4),
      act: this.#chainBuffer(shapeKey, "head.act", tokens * weights.perTokenChannels * 4),
      tokenAct: this.#chainBuffer(shapeKey, "head.token-act",
                                  tokens * weights.perTokenChannels * 4),
      skip: this.#chainBuffer(shapeKey, "head.skip",
                              queryRows * weights.encoder.channels * 4),
      normalised: this.#chainBuffer(shapeKey, "head.normalised",
                                    tokens * weights.perTokenChannels * 4),
    } : undefined;
    // One scope over the whole chain, settled at the decoder's readback - the
    // one boundary a denoiser step already synchronises at.
    const deferred = chained ? new DeferredValidation(this.device, "diffusion head") : undefined;

    // 🔴 POOLED ONLY WHEN CHAINED, AND THAT IS ABOUT THE PEAK. A pooled
    // allocator holds its buffers until something drops the pool, so on the
    // unchained first call - the one that still allocates the seven readbacks
    // - the conditioning's and the encoder's working sets were still resident
    // while the decoder ran, and a 68-token fold's peak rose 48 MiB for a
    // moment that lasts one step in two hundred. The chained steps need the
    // pool because they return while the work is in flight; the first call
    // waits for everything and can use a throwaway allocator, exactly as it
    // did before.
    const conditioner = chained
      ? (this.#conditioner ??= new Af3DiffusionConditioningGpu(this.device, { pool: true }))
      : new Af3DiffusionConditioningGpu(this.device);
    const cond = await stage("conditioning", () =>
      conditioner.run({
        tokens, trunkSingle: input.trunkSingle, trunkPair: input.trunkPair,
        targetFeat: input.targetFeat, noiseLevel: input.noiseLevel,
        features: input.features,
      }, weights.conditioning, {
        reusePair: cachedPair,
        ...(chained ? { outputs: { single: chain.condSingle }, validation: deferred } : {}),
      }));
    if (cachedPair === undefined) {
      // A new trunk means a new fold: drop the encoder's device-side cache too,
      // or the next call reuses the previous molecule's conditioning.
      this.#encoderStatic = undefined;
      this.#releasePersistent();
    }
    this.#conditioningPair = { trunkPair: input.trunkPair, tokens, pair: cond.pair };

    // 🔴 MASKED AND RESCALED - see the note at the top.
    const scaled = new Float32Array(tokens * dense * 3);
    for (let atom = 0; atom < tokens * dense; atom += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        scaled[atom * 3 + axis] =
          input.positionsNoisy[atom * 3 + axis] * input.atomMask[atom] * scale.input;
      }
    }

    const atomEncoder = chained
      ? (this.#encoder ??= new Af3AtomEncoderGpu(this.device, { pool: true }))
      : new Af3AtomEncoderGpu(this.device);
    const encoded = await stage("atom-encoder", () =>
      atomEncoder.run({
        shape: input.shape, conditioning: input.conditioning, atomMask: input.atomMask,
        refPos: input.refPos, refSpaceUid: input.refSpaceUid,
        tokenAtomsToQueries: input.tokenAtomsToQueries,
        queriesToKeys: input.queriesToKeys,
        queriesToTokenAtoms: input.queriesToTokenAtoms,
        tokensToQueries: input.tokensToQueries,
        tokensToKeys: input.tokensToKeys,
        tokenAtomsAct: scaled,
        // 🔴 THE TRUNK'S single, not the conditioning module's.
        trunkSingleCond: input.trunkSingle,
        trunkPairCond: cond.pair,
      }, weights.encoder, {
        reuseStatic: this.#encoderStatic?.conditioning === input.conditioning
          ? this.#encoderStatic : undefined,
        // ...and the DEVICE-side half of the same idea: the encoder's static
        // tensors stay on the GPU between calls instead of being rebuilt.
        staticCache: this.#encoderBuffers,
        ...(chained
          ? { outputs: { tokenAct: chain.tokenAct, skipConnection: chain.skip },
              validation: deferred }
          : {}),
      }));
    // ...cached under the same identity rule as the pair conditioning above,
    // and invalidated by the same thing: a new fold brings a new trunk array.
    if (this.#encoderStatic?.conditioning !== input.conditioning) {
      // 🔴 KEYED ON THE PER-ATOM CONDITIONING TOO, NOT ONLY THE TRUNK. The
      // queries and keys are built from it, and a caller that folds a different
      // molecule against a trunk it happens to still hold would otherwise be
      // handed the previous molecule's atom conditioning.
      this.#encoderStatic = {
        conditioning: input.conditioning,
        pairCond: encoded.pairCond, queriesCond: encoded.queriesCond,
        keysCond: encoded.keysCond, queriesMask: encoded.queriesMask,
        keysMask: encoded.keysMask,
      };
    }

    const projected = await stage("single-projection", () => this.#normaliseAndProject(
      chained ? chain.condSingle : cond.single,
      tokens, weights.seqChannels, weights.perTokenChannels,
      weights.singleCondEmbeddingNormScale, weights.singleCondEmbeddingProjection,
      chained ? { into: chain.act, validation: deferred } : {}));
    let act;
    if (chained) {
      // 🔴 A DISPATCH, NOT A LOOP OVER A COPY. The two halves of the
      // transformer's input are produced by the two stages above, both on the
      // device; adding them on the host would mean reading both back.
      await this.#addInto(chain.act, chain.tokenAct, tokens * weights.perTokenChannels,
                          deferred);
      act = chain.act;
    } else {
      // ...in place: `encoded.tokenAct` is this call's own readback and
      // nothing reads it again.
      act = encoded.tokenAct;
      for (let index = 0; index < act.length; index += 1) act[index] += projected[index];
    }

    this.#transformer ??= new Af3DiffusionTransformerGpu(this.device);
    const transformed = await stage("transformer", () =>
      this.#transformer.run(
        act, chained ? chain.condSingle : cond.single, cond.pair, input.seqMask, tokens,
        this.#transformerWeights(weights),
        chained ? { validation: deferred, keepOnDevice: true } : {}));

    // 🔴 THIS ONE STAYS ON THE CPU, AND THE DIFFERENCE IS THE PROJECTION. With
    // `null` for it this is a LayerNorm and nothing else - 59 rows of 768, too
    // small to be worth a dispatch, and measured at 0 ms. The one above is a
    // 384x768 matmul and was 58.
    const normalised = await stage("output-norm", async () => {
      if (!chained) {
        return normaliseAndProject(
          transformed.output, tokens, weights.perTokenChannels, weights.perTokenChannels,
          weights.outputNormScale, null);
      }
      await this.#normaliseOnly(transformed.outputBuffer, chain.normalised, tokens,
                                weights.perTokenChannels, weights.outputNormScale, deferred);
      return undefined;
    });

    const decoded = await stage("atom-decoder", () =>
      new Af3AtomDecoderGpu(this.device).run(normalised, encoded, input, weights.decoder,
        { staticCache: this.#decoderBuffers,
          ...(chained
            ? { deviceInputs: { skipConnection: chain.skip, tokenAct: chain.normalised } }
            : {}) }));
    // 🔴 SETTLED HERE AND NOWHERE EARLIER. The decoder's readback is the one
    // boundary a denoiser step already synchronises at, so every scope the
    // chain opened is read for free; opening and awaiting them per stage is
    // exactly the round trip this chain exists to remove.
    await deferred?.settle();
    // 🔴 AND THE POOLS ARE DROPPED AT THE END OF EVERY STEP. They exist so
    // that a stage can return while its work is still in flight - a
    // non-pooling release() DESTROYS, and destroying a buffer the queue is
    // still reading is the error this whole chain would otherwise walk into.
    // What they must not do is carry every stage's working set into the next
    // step. By this line the decoder's readback and the settle above have both
    // drained the queue, so there is nothing in flight and nothing to protect.
    this.#conditioner?.allocator.destroyPooled();
    this.#encoder?.allocator.destroyPooled();
    this.#projections?.destroyPooled();

    // 🔴 A BLEND, NOT A PREDICTION.
    const output = new Float32Array(input.positionsNoisy.length);
    for (let atom = 0; atom < tokens * dense; atom += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        const index = atom * 3 + axis;
        output[index] = (scale.skip * input.positionsNoisy[index]
          + scale.out * decoded.update[index]) * input.atomMask[atom];
      }
    }
    return { positions: output, timings, scalings: scale };
  }
}
