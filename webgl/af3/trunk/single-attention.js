/**
 * AF3's single-track attention, biased by the pair representation.
 *
 *     normalised = LayerNorm(single)                 AF3's FAST variance
 *     q = Linear(normalised) + bias                  the ONE bias in the block
 *     k, v, gate = Linear(normalised)
 *     out = Linear(softmax_j(q.k * scale + pairLogits + maskPenalty) . v
 *                  * sigmoid(gate))
 *
 * Structurally this is AF2's row attention with a pair bias applied to one row,
 * which is why `kernels/attention-flash.js` serves it unchanged at `batch = 1`:
 * that kernel's mask index collapses to the key position, which is AF3's
 * `1e9 * (seqMask[j] - 1)`, and its bias layout is already `[head][i][j]`.
 *
 * 🔴 THE ONLY GENUINE BIAS PARAMETER IN THE WHOLE PAIRFORMER BLOCK IS ON THIS
 * MODULE'S QUERIES. Everything else in the block is bias-free - AF3 writes
 * `bias_init=1.0` on its gates, which reads as "this gate starts open" and is
 * inert because those Linears never materialise a parameter. "A reference that
 * helpfully added the missing biases would open every gate and be wrong
 * everywhere", and so would a port; so this one is passed explicitly and the
 * others are not accepted at all.
 *
 * 🔴 AND THE PAIR LOGITS ARE READ AFTER ALL FIVE PAIR UPDATES, which is the
 * block's business rather than this module's - see `pairformer-block.js`. The
 * logits arrive here already `[heads][tokens][tokens]`.
 */
import { attentionFlash } from "../../kernels/attention-flash.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const SINGLE_ATTENTION_WEIGHTS = [
  "layerNormScale", "layerNormOffset",
  "qProjection", "qBias", "kProjection", "vProjection",
  "gatingQuery", "outputProjection",
];

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadSingleAttentionWeights(execution, weights) {
  /** @type {any} */
  const out = { heads: weights.heads, dimension: weights.dimension };
  for (const name of SINGLE_ATTENTION_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) {
      throw new Error(`af3 single attention weights: ${name} is missing`);
    }
    out[name] = execution.uploadWeight(values, `w.af3.single.${name}`);
  }
  return out;
}

export function releaseSingleAttentionWeights(execution, weights) {
  for (const name of SINGLE_ATTENTION_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{single: Tensor, pairLogits: Tensor, seqMask: Tensor, weights: any,
 *   tokens: number, channels: number, epsilon?: number}} call
 * @returns {Tensor} `tokens * channels`, the update; the caller adds it
 */
export function singleAttention(execution, call) {
  const { single, pairLogits, seqMask, weights: w, tokens, channels, epsilon = 1e-5 } = call;
  const heads = w.heads;
  const dimension = w.dimension;
  const width = heads * dimension;
  if (single.elements !== tokens * channels) {
    throw new RangeError(
      `af3 single attention: single holds ${single.elements} floats, not`
      + ` ${tokens} x ${channels}`);
  }
  if (pairLogits.elements !== heads * tokens * tokens) {
    throw new RangeError(
      `af3 single attention: pairLogits hold ${pairLogits.elements} floats, not`
      + ` ${heads} x ${tokens}^2`);
  }

  const normalised = layerNorm(execution, {
    x: single, weight: w.layerNormScale, bias: w.layerNormOffset,
    rows: tokens, channels, epsilon, variance: "fast",
  });
  const project = (weight, bias, activation) => linear(execution, {
    x: normalised, weight, rows: tokens, inChannels: channels, outChannels: width,
    weightLayout: "in-major",
    ...(bias === undefined ? {} : { bias }),
    ...(activation === undefined ? {} : { activation }),
  });
  const query = project(w.qProjection, w.qBias);
  const key = project(w.kProjection);
  const value = project(w.vProjection);
  const gate = project(w.gatingQuery, undefined, "sigmoid");

  const { output: gathered } = attentionFlash(execution, {
    query, key, value, gate, mask: seqMask,
    pairBias: pairLogits,
    // 🔴 ONE BATCH, WHICH IS WHAT MAKES THE FLASH KERNEL'S MASK INDEX RIGHT
    // HERE. With `batch = 1` its `b * Q + k` collapses to `k`, the key position,
    // and that is exactly AF3's `seqMask[j]`.
    batch: 1, queries: tokens, heads, headDim: dimension,
  });
  for (const tensor of [query, key, value, gate]) execution.release(tensor);
  execution.release(normalised);

  const out = linear(execution, {
    x: gathered, weight: w.outputProjection, rows: tokens,
    inChannels: width, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(gathered);
  out.label = "af3.single-attention";
  return out;
}
