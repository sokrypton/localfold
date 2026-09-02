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

    const projected = normaliseAndProject(
      cond.single, tokens, weights.seqChannels, weights.perTokenChannels,
      weights.singleCondEmbeddingNormScale, weights.singleCondEmbeddingProjection);
    const act = Float32Array.from(encoded.tokenAct);
    for (let index = 0; index < act.length; index += 1) act[index] += projected[index];

    const transformed = await stage("transformer", () =>
      new Af3DiffusionTransformerGpu(this.device)
        .run(act, cond.single, cond.pair, input.seqMask, tokens, weights.transformer));

    const normalised = normaliseAndProject(
      transformed.output, tokens, weights.perTokenChannels, weights.perTokenChannels,
      weights.outputNormScale, null);

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
