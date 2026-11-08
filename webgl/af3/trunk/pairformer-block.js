/**
 * One AF3 pairformer block: pair and single in, pair and single out.
 *
 *     pair   += triangleMultiplication(pair, "outgoing")
 *     pair   += triangleMultiplication(pair, "incoming")
 *     pair   += gridSelfAttention(pair, transpose = false)
 *     pair   += gridSelfAttention(pair, transpose = true)
 *     pair   += transition(pair)
 *     single += singleAttention(single, pairLogits(pair))
 *     single += transition(single)
 *
 * 🔴 EACH UPDATE READS THE RUNNING ACTIVATION, NOT THE BLOCK INPUT. "AF3 threads
 * them sequentially where chai-1 sums parallel deltas, which is why the order
 * below is load-bearing" - so a block that summed seven deltas computed from one
 * input would be a different model with the same weights, and no operation-level
 * gate could see it.
 *
 * 🔴 AND THE SINGLE TRACK READS THE PAIR AFTER ALL FIVE PAIR UPDATES. Reading it
 * before, or from the block's input, costs no error and returns a plausible
 * single representation.
 *
 * 🔴 AND THE PAIR LOGITS ARE BROADCAST OVER HEADS FROM A ONE-PER-HEAD
 * PROJECTION. The projection is `[pairs][heads]` and the attention wants
 * `[heads][pairs]`, so the relayout is a draw - the same shape as the grid
 * attention's bias, and for the same reason.
 *
 * 🔴 AND OpenDDE'S REFINER ADDS A PRECOMPUTED BIAS THERE, broadcast over heads:
 * "it comes from the structural-token expander... and every block of the refiner
 * adds the same one. No other pairformer in this tree has it." `extraPairBias`
 * is that term, and absent means it does not exist.
 */
import { ONE_TARGET, PRELUDE, sized, memoSource } from "../../runtime/execution.js";
import { add } from "../../runtime/elementwise.js";
import { gridSelfAttention, releaseGridAttentionWeights, uploadGridAttentionWeights }
  from "./grid-attention.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import {
  releaseSingleAttentionWeights, singleAttention, uploadSingleAttentionWeights,
} from "./single-attention.js";
import {
  releaseTransitionWeights, transition, uploadTransitionWeights,
} from "./transition.js";
import {
  releaseTriangleWeights, triangleMultiplication, uploadTriangleWeights,
} from "./triangle.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/**
 * `[pairs][heads]` to `[heads][pairs]`, with OpenDDE's extra term added.
 *
 * The extra bias is per PAIR and is added to every head, which is why it folds
 * in here rather than being a separate draw: the relayout already visits every
 * (head, pair) once.
 */
const pairLogitsSourceUncached = (tokens, heads, hasExtra) => sized({ PAIRS: tokens * tokens, ELEMENTS: heads * tokens * tokens }, `${PRELUDE}${ONE_TARGET}
const int HEADS = ${heads};
uniform sampler2D flatTex;
${hasExtra ? "uniform sampler2D extraTex;" : ""}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 value = vec4(0.0);
  // Every lane resolves its own head: a texel straddles two head planes
  // whenever the pair count is not a multiple of four.
  for (int lane = 0; lane < 4; ++lane) {
    int element = base + lane;
    if (element >= ELEMENTS) break;
    int head = element / PAIRS;
    int pair = element - head * PAIRS;
    float v = fetch1(flatTex, pair * HEADS + head)${hasExtra ? " + fetch1(extraTex, pair)" : ""};
    if (lane == 0) value.x = v;
    else if (lane == 1) value.y = v;
    else if (lane == 2) value.z = v;
    else value.w = v;
  }
  result = value;
}`);

export const BLOCK_GROUPS = [
  "triangleMultiplicationOutgoing", "triangleMultiplicationIncoming",
  "pairAttention1", "pairAttention2", "pairTransition",
  "singleAttention", "singleTransition",
];

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadPairformerBlockWeights(execution, weights) {
  for (const name of BLOCK_GROUPS) {
    if (weights[name] === undefined) {
      throw new Error(`af3 pairformer block weights: ${name} is missing`);
    }
  }
  return {
    pairChannels: weights.pairChannels,
    singleChannels: weights.singleChannels,
    triangleMultiplicationOutgoing: uploadTriangleWeights(
      execution, weights.triangleMultiplicationOutgoing),
    triangleMultiplicationIncoming: uploadTriangleWeights(
      execution, weights.triangleMultiplicationIncoming),
    pairAttention1: uploadGridAttentionWeights(execution, weights.pairAttention1),
    pairAttention2: uploadGridAttentionWeights(execution, weights.pairAttention2),
    pairTransition: uploadTransitionWeights(execution, weights.pairTransition),
    singlePairLogitsNormScale: execution.uploadWeight(
      weights.singlePairLogitsNormScale, "w.af3.block.logitsNormScale"),
    singlePairLogitsNormOffset: execution.uploadWeight(
      weights.singlePairLogitsNormOffset, "w.af3.block.logitsNormOffset"),
    singlePairLogitsProjection: execution.uploadWeight(
      weights.singlePairLogitsProjection, "w.af3.block.logitsProjection"),
    singleAttention: uploadSingleAttentionWeights(execution, weights.singleAttention),
    singleTransition: uploadTransitionWeights(execution, weights.singleTransition),
  };
}

export function releasePairformerBlockWeights(execution, weights) {
  releaseTriangleWeights(execution, weights.triangleMultiplicationOutgoing);
  releaseTriangleWeights(execution, weights.triangleMultiplicationIncoming);
  releaseGridAttentionWeights(execution, weights.pairAttention1);
  releaseGridAttentionWeights(execution, weights.pairAttention2);
  releaseTransitionWeights(execution, weights.pairTransition);
  releaseSingleAttentionWeights(execution, weights.singleAttention);
  releaseTransitionWeights(execution, weights.singleTransition);
  for (const name of ["singlePairLogitsNormScale", "singlePairLogitsNormOffset",
    "singlePairLogitsProjection"]) {
    execution.release(weights[name]);
  }
}

/**
 * The five PAIR sub-layers, in order - which is also the whole of a TEMPLATE
 * stack block.
 *
 * 🔴 THE TEMPLATE STACK'S BLOCK IS THIS, NOT A FOURTH VARIANT. `templateBlock`
 * in `cpu/af3/trunk/template.js` runs exactly these five on exactly this order,
 * at the template stack's own channel width and its own transition factor - so
 * it is one function rather than a copy, and the one thing that differs (the
 * factor, 2 for the template stack against 4 for the trunk) is derived from the
 * weight inside `transition`.
 *
 * Note this is NOT the AF2 template stack's order: that one attends first. The
 * two models disagree, and `webgl/af2/evoformer/block.js` carries the switch.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, pairMask: Tensor, tokens: number, channels: number,
 *   weights: any, swapTransposedBias?: boolean, divideByLength?: boolean,
 *   epsilon?: number}} call
 * @param {boolean} [ownsPair] true when the caller's `pair` may be released
 * @returns {Tensor}
 */
export function pairHalf(execution, call, ownsPair = false) {
  const {
    pairMask, tokens, channels, weights: w,
    swapTransposedBias = false, divideByLength = false, epsilon = 1e-5,
  } = call;
  let pair = call.pair;
  let own = ownsPair;
  const residual = (current, update) => {
    const next = add(execution, current, update);
    execution.release(update);
    if (own) execution.release(current);
    own = true;
    return next;
  };
  for (const direction of ["outgoing", "incoming"]) {
    const group = direction === "outgoing"
      ? w.triangleMultiplicationOutgoing : w.triangleMultiplicationIncoming;
    pair = residual(pair, triangleMultiplication(execution, {
      pair, mask: pairMask, weights: group,
      length: tokens, channels,
      direction: /** @type {any} */ (direction), divideByLength, epsilon,
    }));
  }
  for (const [group, transpose] of [[w.pairAttention1, false], [w.pairAttention2, true]]) {
    // ...the residual inside, where the column direction's transpose back folds into the add
    const next = gridSelfAttention(execution, {
      pair, mask: pairMask, weights: group,
      length: tokens, channels,
      transpose: /** @type {any} */ (transpose), swapTransposedBias, epsilon, residualOf: pair,
    });
    if (own) execution.release(pair);
    own = true;
    pair = next;
  }
  pair = residual(pair, transition(execution, {
    x: pair, weights: w.pairTransition, rows: tokens * tokens, channels, epsilon,
  }));
  return pair;
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, single: Tensor, pairMask: Tensor, seqMask: Tensor,
 *   tokens: number, weights: ReturnType<typeof uploadPairformerBlockWeights>,
 *   extraPairBias?: Tensor, swapTransposedBias?: boolean,
 *   divideByLength?: boolean, epsilon?: number}} call
 * @returns {{pair: Tensor, single: Tensor}} new tensors; the caller's are untouched
 */
export function pairformerBlock(execution, call) {
  const {
    pairMask, seqMask, tokens, weights: w, extraPairBias,
    swapTransposedBias = false, divideByLength = false, epsilon = 1e-5,
  } = call;
  const pairChannels = w.pairChannels;
  const singleChannels = w.singleChannels;
  const pairs = tokens * tokens;

  let pair = call.pair;
  let ownPair = false;
  /** Thread the update into the running activation and release what it replaces. */
  const residual = (current, update, own) => {
    const next = add(execution, current, update);
    execution.release(update);
    if (own) execution.release(current);
    return next;
  };

  pair = pairHalf(execution, {
    pair, pairMask, tokens, channels: pairChannels, weights: w,
    swapTransposedBias, divideByLength, epsilon,
  }, ownPair);
  ownPair = true;

  // The single track reads the pair AFTER all five pair updates.
  const logitsSource = layerNorm(execution, {
    x: pair, weight: w.singlePairLogitsNormScale, bias: w.singlePairLogitsNormOffset,
    rows: pairs, channels: pairChannels, epsilon, variance: "fast",
  });
  const heads = w.singleAttention.heads;
  const flat = linear(execution, {
    x: logitsSource, weight: w.singlePairLogitsProjection, rows: pairs,
    inChannels: pairChannels, outChannels: heads, weightLayout: "in-major",
  });
  execution.release(logitsSource);
  const pairLogits = execution.allocate(heads * pairs, "af3.block.pair-logits");
  execution.dispatch({
    source: pairLogitsSource(tokens, heads, extraPairBias !== undefined),
    output: pairLogits,
    inputs: {
      flatTex: flat,
      ...(extraPairBias === undefined ? {} : { extraTex: extraPairBias }),
    },
    label: `af3.block.pair-logits:${tokens}x${heads}`
      + `${extraPairBias === undefined ? "" : ":extra"}`,
  });
  execution.release(flat);

  let single = call.single;
  let ownSingle = false;
  single = residual(single, singleAttention(execution, {
    single, pairLogits, seqMask, weights: w.singleAttention,
    tokens, channels: singleChannels, epsilon,
  }), ownSingle);
  ownSingle = true;
  execution.release(pairLogits);
  single = residual(single, transition(execution, {
    x: single, weights: w.singleTransition, rows: tokens, channels: singleChannels, epsilon,
  }), ownSingle);

  pair.label = "af3.block.pair";
  single.label = "af3.block.single";
  return { pair, single };
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const pairLogitsSource = memoSource(pairLogitsSourceUncached);
