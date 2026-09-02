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

export class Af3DiffusionHeadGpu {
  /** The last trunk's pair conditioning, which no noise level changes. */
  #conditioningPair;

  /** ...and the atom encoder's outputs that depend on it rather than on sigma. */
  #encoderStatic;

  constructor(device) {
    this.device = device;
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

  /** The GPU form of normaliseAndProject, for the one call that is hot. */
  async #normaliseAndProject(input, rows, inChannels, outChannels, scale, projection) {
    const allocator = new GpuBufferAllocator(this.device);
    const storage = GPUBufferUsage.STORAGE;
    const held = [];
    try {
      const lanes = 256;
      const pipeline = await pipelineCacheForDevice(this.device).get(
        `af3-normalise-project:${rows}:${inChannels}:${outChannels}:${lanes}`,
        NORMALISE_AND_PROJECT(rows, inChannels, outChannels, lanes));
      const keep = (allocation) => { held.push(allocation); return allocation; };
      const inputBuffer = keep(allocator.upload("np.input", input, storage));
      const scaleBuffer = keep(allocator.upload("np.scale", scale, storage));
      const weightBuffer = keep(allocator.upload("np.projection", projection, storage));
      const output = keep(allocator.allocate("np.output", rows * outChannels * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(allocator.allocate("np.readback", rows * outChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
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
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, rows * outChannels * 4);
      this.device.queue.submit([encoder.finish()]);
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
    const cond = await stage("conditioning", () =>
      new Af3DiffusionConditioningGpu(this.device).run({
        tokens, trunkSingle: input.trunkSingle, trunkPair: input.trunkPair,
        targetFeat: input.targetFeat, noiseLevel: input.noiseLevel,
        features: input.features,
      }, weights.conditioning, { reusePair: cachedPair }));
    if (cachedPair === undefined) this.#encoderStatic = undefined;
    this.#conditioningPair = { trunkPair: input.trunkPair, tokens, pair: cond.pair };

    // 🔴 MASKED AND RESCALED - see the note at the top.
    const scaled = new Float32Array(tokens * dense * 3);
    for (let atom = 0; atom < tokens * dense; atom += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        scaled[atom * 3 + axis] =
          input.positionsNoisy[atom * 3 + axis] * input.atomMask[atom] * scale.input;
      }
    }

    const encoded = await stage("atom-encoder", () =>
      new Af3AtomEncoderGpu(this.device).run({
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
      cond.single, tokens, weights.seqChannels, weights.perTokenChannels,
      weights.singleCondEmbeddingNormScale, weights.singleCondEmbeddingProjection));
    const act = Float32Array.from(encoded.tokenAct);
    for (let index = 0; index < act.length; index += 1) act[index] += projected[index];

    const transformed = await stage("transformer", () =>
      new Af3DiffusionTransformerGpu(this.device)
        .run(act, cond.single, cond.pair, input.seqMask, tokens, weights.transformer));

    // 🔴 THIS ONE STAYS ON THE CPU, AND THE DIFFERENCE IS THE PROJECTION. With
    // `null` for it this is a LayerNorm and nothing else - 59 rows of 768, too
    // small to be worth a dispatch, and measured at 0 ms. The one above is a
    // 384x768 matmul and was 58.
    const normalised = await stage("output-norm", async () => normaliseAndProject(
      transformed.output, tokens, weights.perTokenChannels, weights.perTokenChannels,
      weights.outputNormScale, null));

    const decoded = await stage("atom-decoder", () =>
      new Af3AtomDecoderGpu(this.device).run(normalised, encoded, input, weights.decoder));

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
