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
import { linear, planeFetchGlsl, planeInputs } from "./linear.js";
import { attentionFlash } from "./attention-flash.js";
import { toHeadMajorSource } from "./pair-bias.js";
import { ONE_TARGET, PRELUDE, memoSource, sized, targetDeclarations } from "../runtime/execution.js";

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
    out[name] = execution.uploadWeight(values, `w.attention.${name}`);
  }
  return out;
}

export function releaseAttentionBlockWeights(execution, weights) {
  for (const name of [...ATTENTION_BLOCK_WEIGHTS, "qkvg", "qkvgBias"]) {
    if (weights[name] !== undefined) execution.release(weights[name]);
  }
}

/**
 * The four `[C][C]` in-major projections - query, key, value, gate - as ONE `[C][4C (+ heads)]` in-major weight, with
 * a triangle attention's `[C][heads]` pair-bias projection appended where `heads` makes whole texels (it reads the
 * same normalised activation), and one bias: zeros but for the gate's. AF3's grid attention does the same
 * (`fuseGridProjections`, webgl/af3/trunk/grid-attention.js); AF2's are all in-major already, so it is a concatenation
 * along each input row. Null on a shape this does not expect.
 */
export function fuseAttentionProjections(weights, heads, pairBiasProjection) {
  const { queryWeight: q, keyWeight: k, valueWeight: v, gatingWeight: g, gatingBias: gb } = weights;
  if (![q, k, v, g].every((part) => part instanceof Float32Array && part.length === q.length)) return null;
  const channels = Math.round(Math.sqrt(q.length));
  if (channels * channels !== q.length || channels % 4 !== 0 || !(gb instanceof Float32Array)
    || gb.length !== channels) return null;
  const extra = pairBiasProjection instanceof Float32Array && heads % 4 === 0
    && pairBiasProjection.length === channels * heads ? heads : 0;
  const stride = 4 * channels + extra;
  const fused = new Float32Array(channels * stride);
  for (let c = 0; c < channels; c += 1) {
    const at = c * stride;
    const from = c * channels;
    fused.set(q.subarray(from, from + channels), at);
    fused.set(k.subarray(from, from + channels), at + channels);
    fused.set(v.subarray(from, from + channels), at + 2 * channels);
    fused.set(g.subarray(from, from + channels), at + 3 * channels);
    if (extra > 0) fused.set(pairBiasProjection.subarray(c * heads, (c + 1) * heads), at + 4 * channels);
  }
  const bias = new Float32Array(stride);
  bias.set(gb, 3 * channels);
  return { weight: fused, bias, extra, channels };
}

/**
 * Upload an attention block's weights the way `attentionBlock` reads them fastest: the four projections (and a
 * triangle attention's pair-bias projection, `pairBiasProjection`) fused (`fuseAttentionProjections`), unless
 * `execution.noAf2AttentionFused` asks for the separate form. Returns `{attention, pairBiasFused}`.
 *
 * 🔴 THE FUSED WEIGHT IS HELD IN float16 EXACTLY WHEN ITS PARTS WOULD HAVE BEEN: `halfWeights` decides by size, a
 * 128 x 128 part sits under the floor and its 128 x 516 fusion over it, so `uploadWeight` would have halved a float32
 * weight's precision without anyone asking.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 */
export function uploadAttentionGroup(execution, source, label, heads, pairBiasProjection) {
  let fused = execution.noAf2AttentionFused ? null : fuseAttentionProjections(source, heads, pairBiasProjection);
  // ...a bias projection that cannot ride in it keeps the whole block separate
  if (fused !== null && pairBiasProjection !== undefined && fused.extra === 0) fused = null;
  /** @type {Record<string, any>} */
  const out = {};
  const away = new Set(fused === null ? [] : ["queryWeight", "keyWeight", "valueWeight", "gatingWeight", "gatingBias"]);
  for (const name of ATTENTION_BLOCK_WEIGHTS) {
    if (away.has(name)) continue;
    const values = source[name];
    if (values === undefined) throw new Error(`${label}: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.${label}.${name}`);
  }
  if (fused !== null) {
    const storage = execution.storage;
    const partHalf = storage.halfWeights && fused.channels * fused.channels >= storage.halfWeightFloor;
    out.qkvg = partHalf && fused.extra === 0
      ? execution.uploadWeight(fused.weight, `w.${label}.qkvg`)
      : execution.upload(fused.weight, `w.${label}.qkvg`);
    out.qkvgBias = execution.upload(fused.bias, `w.${label}.qkvg-bias`);
    out.qkvgExtra = fused.extra;
  }
  return { attention: out, pairBiasFused: fused !== null && fused.extra > 0 };
}

/**
 * The fused projection's kept row planes split into query, key, value and gate (the logistic applied), each rounded
 * to float16 where the block keeps its projections that way - and, `TRANSPOSED`, gathered: attention row
 * `b * A + a` (A = `queryLength`, B = `batch`) reads source row `a * B + b`, so the block attends over the transposed
 * activation without a `transposeLeading` draw over it. A projection is per row, so that is the same arithmetic.
 */
const splitSourceUncached = (rows, channels, extra, blockRows, transposed, queries, batch, roundHalf) => sized(
  { ROWS: rows }, `${PRELUDE}${targetDeclarations(4)}
const int C4 = ${channels / 4};
const int A = ${queries};
const int B = ${batch};
${blockRows > 0 ? planeFetchGlsl(blockRows, channels + extra / 4)
    : `uniform sampler2D fusedTex;
vec4 planeFetch(int row, int group) { return fetch4(fusedTex, row * ${channels + extra / 4} + group); }`}

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * C4) {
    result0 = vec4(0.0); result1 = vec4(0.0); result2 = vec4(0.0); result3 = vec4(0.0);
    return;
  }
  int row = texel / C4;
  int group = texel - row * C4;
  ${transposed ? "row = (row % A) * B + row / A;" : ""}
  vec4 gate = vec4(1.0) / (vec4(1.0) + exp(-planeFetch(row, group + 3 * C4)));
${["planeFetch(row, group)", "planeFetch(row, group + C4)", "planeFetch(row, group + 2 * C4)", "gate"]
    .map((v, i) => `  result${i} = ${roundHalf ? `half4(${v})` : v};`).join("\n")}
}`);

/**
 * A triangle attention's pair bias out of the fused projection's last `heads` columns, head-major
 * (`[head][query][key]`, which is what the flash kernel reads) - `toHeadMajorSource` reading kept planes, with the
 * ending node's swap: its bias comes off the TRANSPOSED normalised pair, row `(q, k)` of which is row `k * L + q`.
 */
const biasSourceUncached = (length, heads, stride4, offset, blockRows, swap, roundHalf) => sized(
  { L: length, PAIRS: length * length, ELEMENTS: heads * length * length }, `${PRELUDE}${ONE_TARGET}
const int OFFSET = ${offset};
${blockRows > 0 ? planeFetchGlsl(blockRows, stride4)
    : `uniform sampler2D fusedTex;
vec4 planeFetch(int row, int group) { return fetch4(fusedTex, row * ${stride4} + group); }`}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 value = vec4(0.0);
  for (int lane = 0; lane < 4; ++lane) {
    int element = base + lane;
    if (element >= ELEMENTS) break;
    int head = element / PAIRS;
    int pair = element - head * PAIRS;
    ${swap ? "int i = pair / L;\n    int source = (pair - i * L) * L + i;" : "int source = pair;"}
    float v = planeFetch(source, (OFFSET + head) / 4)[(OFFSET + head) % 4];
    if (lane == 0) value.x = v;
    else if (lane == 1) value.y = v;
    else if (lane == 2) value.z = v;
    else value.w = v;
  }
  result = ${roundHalf ? "half4(value)" : "value"};
}`);

/**
 * `base[x] + update[swapped(x)]` over `[A][B][C]` against an update laid `[B][A][C]`: a transposed attention's residual
 * with the transpose back folded in, where it was a `transposeLeading` and an add (`transposedAddSource` in
 * webgl/af3/trunk/grid-attention.js is the square case).
 */
const transposedResidualSourceUncached = (rows, channels, queries, batch) => sized({ ROWS: rows },
  `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int A = ${queries};
const int B = ${batch};
uniform sampler2D baseTex;
uniform sampler2D updateTex;

void main() {
  int texel = linearTexel();
  if (texel >= ROWS * C4) { result = vec4(0.0); return; }
  int row = texel / C4;
  int swapped = (row % B) * A + row / B;
  result = fetch4(baseTex, texel) + fetch4(updateTex, swapped * C4 + texel % C4);
}`);

export const splitSource = memoSource(splitSourceUncached);
export const biasSource = memoSource(biasSourceUncached);
export const transposedResidualSource = memoSource(transposedResidualSourceUncached);

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

  if (call.transposeRows === true && w.qkvg === undefined) {
    throw new Error("attention block: transposeRows needs the fused projections (uploadAttentionGroup)");
  }
  const normalised = layerNorm(execution, {
    x: activations, weight: w.queryNormScale, bias: w.queryNormOffset,
    rows, channels, epsilon, roundHalf: halfNormalised,
  });

  if (w.qkvg !== undefined) {
    return fusedAttention(execution, call, normalised, {
      rows, headDim, halfProjections, halfBias, halfAttended,
    });
  }

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

  return outputProjection(execution, call, attended, rows);
}

/** The output projection, with the caller's residual (`residualOf`) where there is one - transposed back if need be. */
function outputProjection(execution, call, attended, rows) {
  const { weights: w, channels, residualOf } = call;
  const transposed = call.transposeRows === true;
  const output = linear(execution, {
    x: attended, weight: w.outputWeight, bias: w.outputBias,
    rows, inChannels: channels, outChannels: channels, weightLayout: "in-major",
    ...(residualOf !== undefined && !transposed ? { residual: residualOf } : {}),
  });
  execution.release(attended);
  if (residualOf !== undefined && transposed) {
    const next = execution.allocate(rows * channels, "attention.residual");
    execution.dispatch({
      source: transposedResidualSource(rows, channels, call.queryLength, call.batch),
      output: next,
      inputs: { baseTex: residualOf, updateTex: output },
      label: `attention.transposed-residual:${rows}x${channels}`,
    });
    execution.release(output);
    next.label = "attention.block";
    return next;
  }
  output.label = "attention.block";
  return output;
}

/**
 * One projection for the four (and a triangle attention's bias), split by one draw, where `uploadAttentionGroup`
 * fused them.
 */
function fusedAttention(execution, call, normalised, { rows, headDim, halfProjections, halfBias, halfAttended }) {
  const {
    mask, weights: w, batch, queryLength, channels, heads, pairBias, transpose = false,
  } = call;
  const transposed = call.transposeRows === true;
  const extra = w.qkvgExtra;
  if (call.pairBiasProjection !== undefined) {
    throw new Error("attention block: the fused projections carry their pair bias themselves");
  }
  if (extra > 0 && batch !== queryLength) {
    throw new RangeError("attention block: a fused pair bias needs a square pair matrix");
  }
  const fused = linear(execution, {
    x: normalised, weight: w.qkvg, bias: w.qkvgBias, rows, inChannels: channels,
    outChannels: 4 * channels + extra, weightLayout: "in-major", keepPlanes: true,
  });
  execution.release(normalised);
  const kept = /** @type {any} */ (fused).planes !== undefined ? /** @type {any} */ (fused) : null;
  const fusedInputs = kept === null ? { fusedTex: fused } : planeInputs(kept);
  const blockRows = kept?.blockRows ?? 0;
  let ownBias;
  if (extra > 0) {
    ownBias = execution.allocate(rows * heads, "attention.bias");
    execution.dispatch({
      source: biasSource(queryLength, heads, channels + extra / 4, 4 * channels, blockRows, transposed, halfBias),
      output: ownBias,
      inputs: fusedInputs,
      label: `attention.bias.fused:${heads}x${rows}${transposed ? ":swapped" : ""}`,
    });
  }
  const [query, key, value, gate] = ["q", "k", "v", "g"]
    .map((part) => execution.allocate(rows * channels, `attention.${part}`));
  execution.dispatch({
    source: splitSource(rows, channels, extra, blockRows, transposed, queryLength, batch, halfProjections),
    outputs: [query, key, value, gate],
    inputs: fusedInputs,
    label: `attention.split-qkvg:${rows}x${channels}${transposed ? ":transposed" : ""}`,
  });
  if (kept === null) execution.release(fused);
  else for (const plane of kept.planes) execution.release(plane);
  const { output: attended } = attentionFlash(execution, {
    query, key, value, gate, mask, pairBias: pairBias ?? ownBias,
    batch, queries: queryLength, heads, headDim, transpose, roundHalf: halfAttended,
  });
  for (const tensor of [query, key, value, gate]) execution.release(tensor);
  if (ownBias !== undefined) execution.release(ownBias);
  return outputProjection(execution, call, attended, rows);
}
