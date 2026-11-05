/**
 * The whole gated attention block - what an Evoformer or pairformer layer
 * actually calls, and what `AttentionGpu` is on the WGSL side.
 *
 *     x        = LayerNorm(activations)                 rows x channels
 *     q, k, v  = Linear(x)        (no bias)
 *     gate     = sigmoid(Linear(x) + gatingBias)
 *     attended = softmax(q.k * scale + mask + pairBias) . v  *  gate
 *     out      = Linear(attended) + outputBias
 *
 * 🔴 ITS WEIGHTS ARE `[in][out]`, WHERE THE TRIANGLE'S ARE `[out][in]`. Read
 * off the WGSL shader rather than assumed: `let weight_row = weight_c *
 * projected;` then `weight_row + output_hd`. Nothing in the tensors'
 * SHAPES distinguishes the two conventions - both are `in * out` floats - so
 * every projection here names its layout and `linear` has no default that would
 * silently pick one. A transposed weight returns a tensor of the right shape and
 * a plausible magnitude.
 *
 * 🔴 AND THE HEAD LAYOUT COSTS NOTHING, WHICH IS WORTH STATING BECAUSE IT LOOKS
 * LIKE IT SHOULD. The projections produce `[row][channels]` with channel
 * `c = head * headDim + d`, and the flash core wants
 * `[batch][query][head][channel]` - which for `row = (batch, query)` is the
 * same bytes in the same order. There is no reshape between them.
 *
 * 🔴 THE SCALE IS APPLIED TO THE SCORE HERE AND TO THE QUERY THERE. The WGSL
 * kernel multiplies the projected query by `inverseSqrt(head_dim)` as it writes
 * it (`acc_r_v.x * q_scale`); this scales the logit instead. The product is the
 * same number up to f32 rounding, and saying so is cheaper than someone later
 * finding the two kernels "disagree about where the scale goes".
 *
 * The pair bias is taken PRE-PROJECTED, as `[heads][queries][queries]`. The
 * WGSL block computes it from the pair activations with their own LayerNorm and
 * projection; that stage is not ported yet, and until it is, a caller with pair
 * activations has to project them itself. Nothing here guesses.
 */
import { layerNorm } from "./layer-norm.js";
import { linear } from "./linear.js";
import { attentionFlash } from "./attention-flash.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/** The weights this block reads, in the WGSL side's own spelling. */
export const ATTENTION_BLOCK_WEIGHTS = [
  "queryNormScale", "queryNormOffset",
  "queryWeight", "keyWeight", "valueWeight",
  "gatingWeight", "gatingBias",
  "outputWeight", "outputBias",
];

/**
 * @param {import("../runtime/execution.js").Execution} execution
 */
export function uploadAttentionBlockWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of ATTENTION_BLOCK_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`attention block weights: ${name} is missing`);
    out[name] = execution.upload(values, `w.attention.${name}`);
  }
  return out;
}

export function releaseAttentionBlockWeights(execution, weights) {
  for (const name of ATTENTION_BLOCK_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{activations: Tensor, mask: Tensor, weights: Record<string, Tensor>,
 *   batch: number, queryLength: number, channels: number, heads: number,
 *   pairBias?: Tensor, transpose?: boolean, epsilon?: number}} call
 * @returns {Tensor} `rows x channels`; the caller owns it
 */
export function attentionBlock(execution, call) {
  const {
    activations, mask, weights: w, batch, queryLength, channels, heads,
    pairBias, transpose = false, epsilon = 1e-5,
  } = call;
  if (channels % heads !== 0) {
    throw new RangeError(`attention block: ${channels} channels over ${heads} heads`);
  }
  const headDim = channels / heads;
  const rows = batch * queryLength;
  if (activations.elements !== rows * channels) {
    throw new RangeError(
      `attention block: activations hold ${activations.elements} floats, not `
      + `${batch} x ${queryLength} x ${channels}`);
  }

  const normalised = layerNorm(execution, {
    x: activations, weight: w.queryNormScale, bias: w.queryNormOffset,
    rows, channels, epsilon,
  });

  const project = (weight, bias, activation) => linear(execution, {
    x: normalised, weight, bias, rows, inChannels: channels, outChannels: channels,
    activation, weightLayout: "in-major",
  });

  const query = project(w.queryWeight, undefined, "none");
  const key = project(w.keyWeight, undefined, "none");
  const value = project(w.valueWeight, undefined, "none");
  const gate = project(w.gatingWeight, w.gatingBias, "sigmoid");
  execution.release(normalised);

  const { output: attended } = attentionFlash(execution, {
    query, key, value, gate, mask, pairBias,
    batch, queries: queryLength, heads, headDim, transpose,
  });
  for (const tensor of [query, key, value, gate]) execution.release(tensor);

  const output = linear(execution, {
    x: attended, weight: w.outputWeight, bias: w.outputBias,
    rows, inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(attended);
  output.label = "attention.block";
  return output;
}
