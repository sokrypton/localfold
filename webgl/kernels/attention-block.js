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

/**
 * Which tensors the shipped block keeps in f16, settled by SWEEPING each one
 * against the WGSL block rather than by reading its source.
 *
 * 🔴 THE BIAS IS f32 AND THE OTHER THREE ARE f16, AND READING THE SOURCE GOT
 * THIS WRONG. `pairBiasStorage = options.pairBias?.source === "separate" ?
 * "f32" : normalizedStorage` looks like it says a bias projected from the
 * normalised input is stored f16 - so this port rounded it, and a whole-block
 * comparison sat at **6.0e-5** with the two triangle attentions contributing
 * about 3e-5 each. Keeping the bias in f32 and rounding the other three takes
 * the same comparison to **9.2e-8**, which is its floor.
 *
 * Localising it took two bisects and no further reading: disabling sub-layers by
 * zeroing their output projections on BOTH sides (a residual whose update is
 * zero is an identity) put the gap in the triangle attentions and nowhere else -
 * the outer product mean and both triangle multiplications sit at 4e-8 to 7e-8 -
 * and then sweeping the four rounding points inside them named the bias. Of the
 * four, only the bias moves it to the floor: `attended` f32 gives 5.3e-5,
 * `normalised` f32 gives 9.2e-5, `projections` f32 gives 8.4e-5.
 *
 * So a per-tensor default, not a per-group one. Each entry here is a
 * measurement; `storagePrecision` still sets the group for a caller that wants
 * to price the whole thing.
 */
const DEFAULT_PART = { bias: "f32" };

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
 *   epsilon?: number, storagePrecision?: "f16" | "f32",
 *   normalisedPrecision?: "f16" | "f32", projectionsPrecision?: "f16" | "f32",
 *   biasPrecision?: "f16" | "f32", attendedPrecision?: "f16" | "f32"}} call
 * @returns {Tensor} `rows x channels`; the caller owns it
 */
export function attentionBlock(execution, call) {
  const {
    activations, mask, weights: w, batch, queryLength, channels, heads,
    pairBias, pairBiasProjection, transpose = false, epsilon = 1e-5,
    storagePrecision = "f16",
  } = call;
  // 🔴 EACH ROUNDING POINT SEPARATELY, BECAUSE THE SHIPPED BLOCK'S CHOICES ARE
  // PER TENSOR AND A BLOCK COMPARISON CANNOT SAY WHICH ONE IT DISAGREES ABOUT.
  // `storagePrecision` sets all four; the individual options are what a bisect
  // sweeps. Defaults follow the group, so a caller that does not care sees one
  // switch.
  const partPrecision = (part) => call[`${part}Precision`] ?? DEFAULT_PART[part]
    ?? storagePrecision;
  // 🔴 f16 BY DEFAULT, BECAUSE THE SHIPPED AF2 BLOCK IS. It sets
  // `normalizedStorage = "f16"` outright, derives `projectedStorage` as f16 for
  // any channel width that is even, and asks its flash kernel for
  // `{input: "f16", ..., output: "f16"}` - none of it behind
  // `halfPrecisionAvailable` or a knob. An f32 chain disagrees with it by ~1e-4
  // on every attention, which a whole-block comparison measured against a
  // 1.9e-8 floor. `"f32"` is the arm for pricing the rounding; it is not the
  // configuration the primary path runs.
  const half = storagePrecision === "f16";
  const halfNormalised = partPrecision("normalised") === "f16";
  const halfProjections = partPrecision("projections") === "f16";
  const halfBias = partPrecision("bias") === "f16";
  const halfAttended = partPrecision("attended") === "f16";
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
    rows, channels, epsilon, roundHalf: halfNormalised,
  });

  const project = (weight, bias, activation) => linear(execution, {
    x: normalised, weight, bias, rows, inChannels: channels, outChannels: channels,
    activation, weightLayout: "in-major", roundHalf: halfProjections,
  });

  // ...projected off the SAME normalised tensor, before it is released.
  let ownBias;
  if (pairBiasProjection !== undefined) {
    // ...the block's `pairBiasStorage` is `normalizedStorage` for a bias
    // projected from the normalised input, and f32 only for a separate one.
    const pairMajor = linear(execution, {
      x: normalised, weight: pairBiasProjection,
      rows, inChannels: channels, outChannels: heads, weightLayout: "in-major",
      roundHalf: halfBias,
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
    batch, queries: queryLength, heads, headDim, transpose, roundHalf: halfAttended,
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
