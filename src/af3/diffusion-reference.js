/**
 * AF3's diffusion head: the part that produces coordinates.
 *
 * The trunk predicts a distogram; this predicts atoms. It is a DENOISER, not a
 * generator: given noisy positions and the noise level they carry, it returns
 * its estimate of the clean structure, and the sampler in
 * diffusion-sampler-reference.js calls it two hundred times down a noise
 * schedule.
 *
 *     conditioning   trunk single + target_feat -> 384,  trunk pair + relative
 *                    encoding -> 128, plus a Fourier embedding of the noise
 *     encoder        the noisy positions, through the SAME atom cross-attention
 *                    encoder the trunk uses, now with the trunk conditioning
 *     transformer    24 blocks over tokens at 768 channels
 *     decoder        back down to atoms, and out as a position update
 *
 * 🔴 THE 203 M PARAMETERS ARE NOT 203 M OF NEW IDEAS. 198 M of them - 98% - are
 * the 24-block transformer, which is the AdaLN-conditioned attention already
 * written for the atom stack at different widths. Its atom encoder and decoder
 * are 0.9 M each and are literally atom-encoder-reference.js with other
 * weights. What is genuinely new here is the conditioning and the scaling.
 *
 * 🔴 THE OUTPUT IS A BLEND, NOT A PREDICTION. AF3 returns
 * `skip * positions_noisy + out * update`, where the two coefficients depend on
 * the noise level: at high noise the update dominates, at low noise the input
 * does. Returning the update alone type-checks, runs, and produces a structure
 * that is wrong in a way that looks like a bad model rather than a bug.
 */
import { adaptiveLayerNorm, adaptiveZeroInit, layerNormSlow }
  from "./atom-encoder-reference.js";
import { linear } from "./pairformer-reference.js";
import { relativeEncoding } from "./embedder-reference.js";
import { noiseEmbedding } from "./noise-fourier.js";

/** AF3's assumed data scale, in angstroms. Every noise level is relative to it. */
export const SIGMA_DATA = 16.0;

const sigmoid = (value) => 1 / (1 + Math.exp(-value));
const swish = (value) => value * sigmoid(value);

/**
 * A transition block with AdaLN conditioning, or none.
 *
 * The trunk's transitions have a learned LayerNorm and no conditioning; these
 * take both their scale and their shift from `cond`, and gate the result. With
 * `cond` null it degrades to the plain form, which is what the two conditioning
 * transitions in _conditioning use.
 */
export function conditionedTransition(x, cond, rows, channels, factor, weights,
                                      prefix, condChannels = channels) {
  const intermediate = channels * factor;
  const normalised = cond === null
    ? layerNormSlow(x, rows, channels, weights[`${prefix}FfwLayerNormScale`],
                    weights[`${prefix}FfwLayerNormOffset`])
    : adaptiveLayerNorm(x, cond, rows, channels, weights, `${prefix}ffw`, condChannels);
  const wide = linear(normalised, rows, channels, intermediate * 2,
                      weights[`${prefix}FfwTransition1`]);
  const gated = new Float32Array(rows * intermediate);
  for (let row = 0; row < rows; row += 1) {
    for (let i = 0; i < intermediate; i += 1) {
      gated[row * intermediate + i] = swish(wide[row * intermediate * 2 + i])
        * wide[row * intermediate * 2 + intermediate + i];
    }
  }
  if (cond === null) {
    return linear(gated, rows, intermediate, channels,
                  weights[`${prefix}FfwTransition2`]);
  }
  return adaptiveZeroInit(gated, cond, rows, channels, weights, `${prefix}ffw`,
                          condChannels, intermediate);
}

/**
 * Self-attention over tokens, conditioned by AdaLN and biased by the pair.
 *
 * @param {Float32Array} act        tokens * channels
 * @param {Float32Array} cond       tokens * condChannels
 * @param {Float32Array} pairLogits heads * tokens * tokens
 * @param {Float32Array} mask       tokens
 */
export function conditionedSelfAttention(act, cond, pairLogits, mask, tokens,
                                         channels, condChannels, weights) {
  const heads = weights.heads;
  const dimension = weights.dimension;
  const width = heads * dimension;
  // 🔴 THE SCALE IS THE PER-HEAD DIMENSION, taken AFTER the division by the
  // head count - AF3 writes `key_dim = key_dim // num_head` and only then
  // `key_dim ** -0.5`. Using the full 768 instead of 48 is a factor of four on
  // every logit, which softmax turns into a much flatter attention.
  const scale = 1 / Math.sqrt(dimension);
  const x = adaptiveLayerNorm(act, cond, tokens, channels, weights, "", condChannels);
  const q = linear(x, tokens, channels, width, weights.qProjection, weights.qBias);
  const k = linear(x, tokens, channels, width, weights.kProjection);
  const v = linear(x, tokens, channels, width, weights.vProjection);

  const gathered = new Float32Array(tokens * width);
  const logits = new Float32Array(tokens);
  for (let head = 0; head < heads; head += 1) {
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        let dot = 0;
        for (let d = 0; d < dimension; d += 1) {
          dot += q[i * width + head * dimension + d] * k[j * width + head * dimension + d];
        }
        logits[j] = dot * scale + 1e9 * (mask[j] - 1)
          + pairLogits[(head * tokens + i) * tokens + j];
      }
      let largest = -Infinity;
      for (let j = 0; j < tokens; j += 1) if (logits[j] > largest) largest = logits[j];
      let total = 0;
      for (let j = 0; j < tokens; j += 1) {
        logits[j] = Math.exp(logits[j] - largest);
        total += logits[j];
      }
      for (let d = 0; d < dimension; d += 1) {
        let sum = 0;
        for (let j = 0; j < tokens; j += 1) {
          sum += logits[j] * v[j * width + head * dimension + d];
        }
        gathered[i * width + head * dimension + d] = sum / total;
      }
    }
  }

  const gate = linear(x, tokens, channels, width, weights.gatingQuery);
  for (let index = 0; index < gathered.length; index += 1) {
    gathered[index] *= sigmoid(gate[index]);
  }
  return adaptiveZeroInit(gathered, cond, tokens, channels, weights, "",
                          condChannels, width);
}

/**
 * The 24-block token transformer.
 *
 * 🔴 THE BLOCKS ARE NESTED SIX BY FOUR, AND THE PAIR LOGITS FOLLOW THAT NESTING.
 * The LayerNorm over the pair conditioning is computed ONCE and shared, but each
 * of the six SUPER-BLOCKS then projects it to its own four blocks' worth of
 * head biases. So there are six projections, not one and not twenty-four, and a
 * flat reading of the stack indexes the wrong weights for every block after the
 * fourth.
 */
export function diffusionTransformer(act, cond, pairCond, mask, tokens, weights) {
  const channels = weights.channels;
  const condChannels = weights.condChannels;
  const heads = weights.heads;
  const perSuper = weights.blocksPerSuperBlock;
  const pairs = tokens * tokens;

  const normalisedPair = layerNormSlow(pairCond, pairs, weights.pairChannels,
                                       weights.pairInputLayerNormScale, null);
  let current = act;
  for (let superBlock = 0; superBlock < weights.superBlocks.length; superBlock += 1) {
    const group = weights.superBlocks[superBlock];
    const flat = linear(normalisedPair, pairs, weights.pairChannels, perSuper * heads,
                        group.pairLogitsProjection);
    for (let inner = 0; inner < perSuper; inner += 1) {
      const pairLogits = new Float32Array(heads * pairs);
      for (let i = 0; i < tokens; i += 1) {
        for (let j = 0; j < tokens; j += 1) {
          const source = (i * tokens + j) * perSuper * heads + inner * heads;
          for (let head = 0; head < heads; head += 1) {
            pairLogits[(head * tokens + i) * tokens + j] = flat[source + head];
          }
        }
      }
      const block = group.blocks[inner];
      const attention = conditionedSelfAttention(current, cond, pairLogits, mask,
                                                 tokens, channels, condChannels, block);
      const afterAttention = new Float32Array(current.length);
      for (let index = 0; index < current.length; index += 1) {
        afterAttention[index] = current[index] + attention[index];
      }
      const transitioned = conditionedTransition(afterAttention, cond, tokens, channels,
                                                 weights.transitionFactor, block, "",
                                                 condChannels);
      const next = new Float32Array(current.length);
      for (let index = 0; index < current.length; index += 1) {
        next[index] = afterAttention[index] + transitioned[index];
      }
      current = next;
    }
  }
  return current;
}

/**
 * The diffusion head's conditioning: what the denoiser knows besides the atoms.
 *
 * @returns {{single: Float32Array, pair: Float32Array}}
 */
export function diffusionConditioning(input, weights) {
  const { tokens, trunkSingle, trunkPair, targetFeat, noiseLevel } = input;
  const pairs = tokens * tokens;
  const pairChannels = weights.pairChannels;
  const seqChannels = weights.seqChannels;

  // ...the trunk pair and the RAW relative encoding, concatenated. 128 + 139.
  const relative = relativeEncoding(tokens, input.features);
  const width = pairChannels + weights.relativeWidth;
  const features2d = new Float32Array(pairs * width);
  for (let index = 0; index < pairs; index += 1) {
    for (let c = 0; c < pairChannels; c += 1) {
      features2d[index * width + c] = trunkPair[index * pairChannels + c];
    }
    for (let c = 0; c < weights.relativeWidth; c += 1) {
      features2d[index * width + pairChannels + c] =
        relative[index * weights.relativeWidth + c];
    }
  }
  let pair = linear(layerNormSlow(features2d, pairs, width,
                                  weights.pairCondInitialNormScale, null),
                    pairs, width, pairChannels, weights.pairCondInitialProjection);
  for (let index = 0; index < 2; index += 1) {
    const delta = conditionedTransition(pair, null, pairs, pairChannels, 2,
                                        weights.pairTransitions[index], "");
    for (let i = 0; i < pair.length; i += 1) pair[i] += delta[i];
  }

  // ...and the trunk single with target_feat. 384 + 447.
  const singleWidth = seqChannels + weights.targetFeatWidth;
  const features1d = new Float32Array(tokens * singleWidth);
  for (let token = 0; token < tokens; token += 1) {
    for (let c = 0; c < seqChannels; c += 1) {
      features1d[token * singleWidth + c] = trunkSingle[token * seqChannels + c];
    }
    for (let c = 0; c < weights.targetFeatWidth; c += 1) {
      features1d[token * singleWidth + seqChannels + c] =
        targetFeat[token * weights.targetFeatWidth + c];
    }
  }
  const single = linear(layerNormSlow(features1d, tokens, singleWidth,
                                      weights.singleCondInitialNormScale, null),
                        tokens, singleWidth, seqChannels,
                        weights.singleCondInitialProjection);

  // 🔴 THE NOISE LEVEL IS SCALED BY SIGMA_DATA BEFORE THE LOG, and the Fourier
  // constants are stock AF3's. A ported checkpoint carries its own trained
  // embedding; see noise-fourier.js.
  const embedded = noiseEmbedding(noiseLevel / SIGMA_DATA);
  const noiseChannels = embedded.length;
  const projected = linear(layerNormSlow(embedded, 1, noiseChannels,
                                         weights.noiseEmbeddingInitialNormScale, null),
                           1, noiseChannels, seqChannels,
                           weights.noiseEmbeddingInitialProjection);
  for (let token = 0; token < tokens; token += 1) {
    for (let c = 0; c < seqChannels; c += 1) {
      single[token * seqChannels + c] += projected[c];
    }
  }
  for (let index = 0; index < 2; index += 1) {
    const delta = conditionedTransition(single, null, tokens, seqChannels, 2,
                                        weights.singleTransitions[index], "");
    for (let i = 0; i < single.length; i += 1) single[i] += delta[i];
  }

  return { single, pair };
}

/**
 * The two coefficients that turn a network output into a denoised structure.
 *
 * At a noise level far above SIGMA_DATA the skip term vanishes and the update
 * carries everything; far below, the reverse. This is what makes the same
 * network usable at every step of the schedule.
 */
export function scalings(noiseLevel) {
  const denominator = noiseLevel * noiseLevel + SIGMA_DATA * SIGMA_DATA;
  return {
    skip: SIGMA_DATA * SIGMA_DATA / denominator,
    out: noiseLevel * SIGMA_DATA / Math.sqrt(denominator),
    // ...and what the network's INPUT is divided by, so its scale is O(1)
    // whatever the noise level.
    input: 1 / Math.sqrt(denominator),
  };
}
