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
 * 🔴 THE PAIR BIAS ARRIVES TWO WAYS, WHICH IS WHAT SEPARATES AF2's MSA ROW
 * ATTENTION FROM ITS TRIANGLE ATTENTIONS.
 *
 *   - `pairBias`: already projected, `[heads][queries][queries]`. The MSA row
 *     attention's bias comes from the PAIR tensor while the attention runs over
 *     the MSA, so it is produced by `./pair-bias.js` with its own LayerNorm and
 *     handed in. The WGSL side calls this `source: "separate"`.
 *   - `pairBiasProjection`: a `[channels][heads]` weight, and the bias is
 *     projected from THIS block's own normalised activations. That is the
 *     triangle attentions, where the bias and the q/k/v come from the same
 *     tensor - which is why fusing a second LayerNorm there would compute the
 *     same statistics twice. The WGSL side's non-separate path, with
 *     `pair_channels` set to `input.channels`.
 *
 * The second only means anything when the activations are a SQUARE pair matrix,
 * because the bias it produces is indexed `[head][query][key]` over
 * `queries^2` - so `batch` must equal `queryLength` and that is refused rather
 * than silently producing a bias of the wrong length.
 */
import { layerNorm } from "./layer-norm.js";
import { linear } from "./linear.js";
import { attentionFlash } from "./attention-flash.js";
import { toHeadMajorSource } from "./pair-bias.js";

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
 *   pairBias?: Tensor, pairBiasProjection?: Tensor, transpose?: boolean,
 *   epsilon?: number, storagePrecision?: "f16" | "f32"}} call
 * @returns {Tensor} `rows x channels`; the caller owns it
 */
export function attentionBlock(execution, call) {
  const {
    activations, mask, weights: w, batch, queryLength, channels, heads,
    pairBias, pairBiasProjection, transpose = false, epsilon = 1e-5,
    storagePrecision = "f16",
  } = call;
  // 🔴 f16 BY DEFAULT, BECAUSE THE SHIPPED AF2 BLOCK IS. It sets
  // `normalizedStorage = "f16"` outright, derives `projectedStorage` as f16 for
  // any channel width that is even, and asks its flash kernel for
  // `{input: "f16", ..., output: "f16"}` - none of it behind
  // `halfPrecisionAvailable` or a knob. An f32 chain disagrees with it by ~1e-4
  // on every attention, which a whole-block comparison measured against a
  // 1.9e-8 floor. `"f32"` is the arm for pricing the rounding; it is not the
  // configuration the primary path runs.
  const half = storagePrecision === "f16";
  if (pairBias !== undefined && pairBiasProjection !== undefined) {
    throw new Error(
      "attention block: a bias is either pre-projected (pairBias) or projected from "
      + "this block's own normalised activations (pairBiasProjection), not both");
  }
  if (pairBiasProjection !== undefined && batch !== queryLength) {
    throw new RangeError(
      `attention block: a bias projected from the normalised input indexes `
      + `[head][query][key] over queryLength^2, so batch (${batch}) must equal `
      + `queryLength (${queryLength}) - which is true of a square pair matrix and `
      + "is what the triangle attentions pass");
  }
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
    rows, channels, epsilon, roundHalf: half,
  });

  const project = (weight, bias, activation) => linear(execution, {
    x: normalised, weight, bias, rows, inChannels: channels, outChannels: channels,
    activation, weightLayout: "in-major", roundHalf: half,
  });

  // ...projected off the SAME normalised tensor, before it is released.
  let ownBias;
  if (pairBiasProjection !== undefined) {
    // ...the block's `pairBiasStorage` is `normalizedStorage` for a bias
    // projected from the normalised input, and f32 only for a separate one.
    const pairMajor = linear(execution, {
      x: normalised, weight: pairBiasProjection,
      rows, inChannels: channels, outChannels: heads, weightLayout: "in-major",
      roundHalf: half,
    });
    ownBias = execution.allocate(rows * heads, "attention.bias");
    execution.dispatch({
      source: toHeadMajorSource(rows, heads),
      output: ownBias,
      inputs: { sourceTex: pairMajor },
      label: `attention.bias.head-major:${heads}x${rows}`,
    });
    execution.release(pairMajor);
  }

  const query = project(w.queryWeight, undefined, "none");
  const key = project(w.keyWeight, undefined, "none");
  const value = project(w.valueWeight, undefined, "none");
  const gate = project(w.gatingWeight, w.gatingBias, "sigmoid");
  execution.release(normalised);

  const { output: attended } = attentionFlash(execution, {
    query, key, value, gate, mask, pairBias: pairBias ?? ownBias,
    batch, queries: queryLength, heads, headDim, transpose, roundHalf: half,
  });
  for (const tensor of [query, key, value, gate]) execution.release(tensor);
  if (ownBias !== undefined) execution.release(ownBias);

  const output = linear(execution, {
    x: attended, weight: w.outputWeight, bias: w.outputBias,
    rows, inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(attended);
  output.label = "attention.block";
  return output;
}
