/**
 * One AF3 MSA block (an "evoformer iteration"): msa and pair in, both out.
 *
 *     pair += outerProductMean(msa)       <- the MSA BEFORE its update
 *     msa  += msaAttention(msa, pair)     <- the pair AFTER the outer product
 *     msa  += transition(msa)
 *     pair += triangleMultiplication(pair, "outgoing")
 *     pair += triangleMultiplication(pair, "incoming")
 *     pair += gridSelfAttention(pair, transpose = false)
 *     pair += gridSelfAttention(pair, transpose = true)
 *     pair += transition(pair)
 *
 * 🔴 THE FIRST TWO LINES CROSS, AND TWO MODELS OF THE LINEAGE SWAP THEM.
 * `cpu/af3/trunk/msa.js`: "AF3 takes the outer product of the MSA as it arrived
 * and then updates the MSA against the pair that outer product just changed;
 * OpenDDE and Boltz-2 update the MSA first and take the outer product of the
 * result. Same modules, same weights, different function - and the difference
 * compounds over four blocks. Reading either line alone tells you nothing; the
 * order is the content."
 *
 * 🔴 SO `msaUpdateBeforeOuterProduct` HAS NO DEFAULT, AND THIS REFUSES WITHOUT
 * IT - exactly as the CPU reference does. A default would be a guess about which
 * model is being run, and both guesses produce a plausible representation.
 *
 * 🔴 AND THE TRANSITION MOVES WITH THE ATTENTION, NOT WITH THE BLOCK. "Upstream
 * keeps the transition inside `_msa_update`, so it moves with the attention
 * rather than staying put between the two halves" - which matters only in the
 * swapped order, and is invisible in AF3's own.
 *
 * The whole pair half is the pairformer's, imported rather than copied: both
 * triangle multiplications, both grid attentions and the pair transition are the
 * same modules with the same conventions.
 */
import { add } from "../../runtime/elementwise.js";
import { gridSelfAttention, releaseGridAttentionWeights, uploadGridAttentionWeights }
  from "./grid-attention.js";
import {
  msaAttention, releaseMsaAttentionWeights, uploadMsaAttentionWeights,
} from "./msa-attention.js";
import {
  outerProductMean, releaseOuterProductMeanWeights, uploadOuterProductMeanWeights,
} from "../../kernels/outer-product-mean.js";
import {
  releaseTransitionWeights, transition, uploadTransitionWeights,
} from "./transition.js";
import {
  releaseTriangleWeights, triangleMultiplication, uploadTriangleWeights,
} from "./triangle.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const MSA_BLOCK_GROUPS = [
  "outerProductMean", "msaAttention1", "msaTransition",
  "triangleMultiplicationOutgoing", "triangleMultiplicationIncoming",
  "pairAttention1", "pairAttention2", "pairTransition",
];

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 *
 * 🔴 THE OUTER PRODUCT MEAN IS AF2'S KERNEL, AND ITS WEIGHT NAMES ARE AF2'S.
 * AF3 spells them `layerNormInputScale`, `leftProjection`, `outputW`; the kernel
 * reads `layerNormScale`, `leftWeight`, `outputWeight`. The arithmetic is the
 * same operation with the same conventions - mask on the projections, output
 * weight `[cl][cr][cZ]`, bias before the division, epsilon 1e-3 - so the names
 * are translated here rather than the kernel being forked. The one thing that
 * genuinely differs is the LayerNorm's variance, which the kernel takes as an
 * option.
 */
export function uploadMsaBlockWeights(execution, weights) {
  for (const name of MSA_BLOCK_GROUPS) {
    if (weights[name] === undefined) {
      throw new Error(`af3 msa block weights: ${name} is missing`);
    }
  }
  const opm = weights.outerProductMean;
  const translated = {
    layerNormScale: opm.layerNormInputScale,
    layerNormOffset: opm.layerNormInputOffset,
    leftWeight: opm.leftProjection,
    // 🔴 AF3 HAS NO PROJECTION BIASES AND rosettafold3 DOES, "added BEFORE the
    // mask... an unmasked row gains two cross terms in the bilinear product,
    // which is why dropping them is not a constant offset". The AF2 kernel
    // always reads a bias tensor, so an absent one becomes zeros HERE, where the
    // comment can say that zeros and absent are the same thing for a term added
    // before a mask - which is not true of the output bias below.
    leftBias: opm.leftProjectionBias ?? new Float32Array(opm.outerChannels),
    rightWeight: opm.rightProjection,
    rightBias: opm.rightProjectionBias ?? new Float32Array(opm.outerChannels),
    outputWeight: opm.outputW,
    outputBias: opm.outputB,
  };
  return {
    pairChannels: weights.pairChannels,
    msaChannels: weights.msaChannels,
    outerChannels: opm.outerChannels,
    outerProductMean: uploadOuterProductMeanWeights(execution, translated),
    msaAttention1: uploadMsaAttentionWeights(execution, weights.msaAttention1),
    msaTransition: uploadTransitionWeights(execution, weights.msaTransition),
    triangleMultiplicationOutgoing: uploadTriangleWeights(
      execution, weights.triangleMultiplicationOutgoing),
    triangleMultiplicationIncoming: uploadTriangleWeights(
      execution, weights.triangleMultiplicationIncoming),
    pairAttention1: uploadGridAttentionWeights(execution, weights.pairAttention1),
    pairAttention2: uploadGridAttentionWeights(execution, weights.pairAttention2),
    pairTransition: uploadTransitionWeights(execution, weights.pairTransition),
  };
}

export function releaseMsaBlockWeights(execution, weights) {
  releaseOuterProductMeanWeights(execution, weights.outerProductMean);
  releaseMsaAttentionWeights(execution, weights.msaAttention1);
  releaseTransitionWeights(execution, weights.msaTransition);
  releaseTriangleWeights(execution, weights.triangleMultiplicationOutgoing);
  releaseTriangleWeights(execution, weights.triangleMultiplicationIncoming);
  releaseGridAttentionWeights(execution, weights.pairAttention1);
  releaseGridAttentionWeights(execution, weights.pairAttention2);
  releaseTransitionWeights(execution, weights.pairTransition);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{msa: Tensor, pair: Tensor, msaMask: Tensor, pairMask: Tensor,
 *   sequences: number, tokens: number,
 *   weights: ReturnType<typeof uploadMsaBlockWeights>,
 *   msaUpdateBeforeOuterProduct: boolean, swapTransposedBias?: boolean,
 *   divideByLength?: boolean, opmBiasAfterNorm?: boolean, epsilon?: number}} call
 * @returns {{msa: Tensor, pair: Tensor}} new tensors; the caller's are untouched
 */
export function msaBlock(execution, call) {
  const {
    msaMask, pairMask, sequences, tokens, weights: w,
    swapTransposedBias = false, divideByLength = false, opmBiasAfterNorm = false,
    epsilon = 1e-5,
  } = call;
  if (typeof call.msaUpdateBeforeOuterProduct !== "boolean") {
    throw new Error(
      "msaBlock: msaUpdateBeforeOuterProduct has no default - AF3 takes the outer product"
      + " off the pre-update MSA and OpenDDE off the updated one, and both orders"
      + " produce a plausible representation");
  }
  const pairChannels = w.pairChannels;
  const msaChannels = w.msaChannels;
  const pairs = tokens * tokens;
  const rows = sequences * tokens;

  let msa = call.msa;
  let pair = call.pair;
  let ownMsa = false;
  let ownPair = false;
  const residual = (current, update, own) => {
    const next = add(execution, current, update);
    execution.release(update);
    if (own) execution.release(current);
    return next;
  };

  const outerProduct = () => {
    pair = residual(pair, outerProductMean(execution, {
      activations: msa, mask: msaMask, weights: w.outerProductMean,
      sequences, length: tokens, cM: msaChannels, cOuter: w.outerChannels, cZ: pairChannels,
      layerNormEpsilon: epsilon, variance: "fast", biasAfterNorm: opmBiasAfterNorm,
    }), ownPair);
    ownPair = true;
  };
  const updateMsa = () => {
    msa = residual(msa, msaAttention(execution, {
      msa, msaMask, pair, weights: w.msaAttention1,
      sequences, tokens, msaChannels, pairChannels, epsilon,
    }), ownMsa);
    ownMsa = true;
    {
      // ...the residual inside: the down projection's repack adds it
      const next = transition(execution, {
        x: msa, weights: w.msaTransition, rows, channels: msaChannels, epsilon, residualOf: msa,
      });
      if (ownMsa) execution.release(msa);
      msa = next;
    }
  };
  if (call.msaUpdateBeforeOuterProduct) {
    updateMsa();
    outerProduct();
  } else {
    outerProduct();
    updateMsa();
  }

  for (const direction of ["outgoing", "incoming"]) {
    const group = direction === "outgoing"
      ? w.triangleMultiplicationOutgoing : w.triangleMultiplicationIncoming;
    // ...the residual inside: the triangle's last draw adds it
    const next = triangleMultiplication(execution, {
      pair, mask: pairMask, weights: group,
      length: tokens, channels: pairChannels,
      direction: /** @type {any} */ (direction), divideByLength, epsilon, residualOf: pair,
    });
    if (ownPair) execution.release(pair);
    pair = next;
    ownPair = true;
  }
  for (const [group, transpose] of [[w.pairAttention1, false], [w.pairAttention2, true]]) {
    // ...the residual inside, where the column direction's transpose back folds into the add
    const next = gridSelfAttention(execution, {
      pair, mask: pairMask, weights: group,
      length: tokens, channels: pairChannels,
      transpose: /** @type {any} */ (transpose), swapTransposedBias, epsilon, residualOf: pair,
    });
    if (ownPair) execution.release(pair);
    pair = next;
    ownPair = true;
  }
  {
    // ...the residual inside: the down projection's repack adds it
    const next = transition(execution, {
      x: pair, weights: w.pairTransition, rows: pairs, channels: pairChannels, epsilon, residualOf: pair,
    });
    if (ownPair) execution.release(pair);
    pair = next;
  }

  msa.label = "af3.msa-block.msa";
  pair.label = "af3.msa-block.pair";
  return { msa, pair };
}
