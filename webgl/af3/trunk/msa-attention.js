/**
 * AF3's MSA pair-weighted averaging: values from the MSA, weights from the pair.
 *
 *     normalised  = LayerNorm(msa)                     AF3's FAST variance
 *     flat        = Linear(LayerNorm(pair))            one logit a head a pair
 *     w[h][i][j]  = softmax_j(flat[i][j][h] + 1e9 * (keyMask[j] - 1))
 *     averaged    = sum_j w[h][i][j] * value[s][j][h][d]
 *     out         = Linear(averaged * sigmoid(gate))
 *
 * 🔴 IT HAS NO QUERIES AND NO KEYS, WHICH IS THE WHOLE POINT. "Every other
 * attention here builds its weights from the thing it attends over. This one
 * builds them ENTIRELY from the pair representation - `pair_logits` is the whole
 * score - and the MSA supplies only values and a gate. AF3 calls it
 * pair-weighted averaging, and it is the reason the MSA stack is 3 M parameters
 * against the pairformer's 147 M."
 *
 * So `kernels/attention-flash.js` does NOT serve this one: that kernel's inner
 * loop is a dot product of a query against a key, and here there is nothing to
 * dot. The weights are a softmax over a tensor that is already complete before
 * any MSA row is read, so they are computed once for all sequences and then
 * applied - which is also why this is the cheapest attention in the trunk.
 *
 * 🔴 AND THERE IS ONE KEY MASK FOR ALL SEQUENCES, taken as the MAXIMUM over the
 * MSA depth: "a token any sequence covers is attendable by every sequence".
 * Using each sequence's own mask row is the obvious reading and is wrong - it
 * would make the weights depend on the sequence, which is exactly what this
 * module is built not to do, and the softmax would then have to be recomputed
 * per row.
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import { multiply } from "../../runtime/elementwise.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const MSA_ATTENTION_WEIGHTS = [
  "actNormScale", "actNormOffset", "pairNormScale", "pairNormOffset",
  "pairLogits", "vProjection", "gatingQuery", "outputProjection",
];

/** `keyMask[t] = max over sequences of msaMask[s][t]`. */
export const keyMaskSource = (sequences, tokens) => `${PRELUDE}${ONE_TARGET}
const int SEQUENCES = ${sequences};
const int L = ${tokens};
uniform sampler2D maskTex;

float largestAt(int token) {
  float largest = 0.0;
  for (int s = 0; s < SEQUENCES; ++s) {
    largest = max(largest, fetch1(maskTex, s * L + token));
  }
  return largest;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= L) { result = vec4(0.0); return; }
  result = vec4(
    largestAt(base),
    base + 1 < L ? largestAt(base + 1) : 0.0,
    base + 2 < L ? largestAt(base + 2) : 0.0,
    base + 3 < L ? largestAt(base + 3) : 0.0);
}`;

/**
 * The softmaxed weights, `[heads][i][j]`, from the flat `[pairs][heads]` logits.
 *
 * A fragment owns four consecutive j of one (head, i), and the softmax needs the
 * whole j row - so it sweeps the row twice, once for the maximum and once for
 * the sum. Four j a fragment rather than one keeps that sweep shared.
 */
export const msaWeightsSource = (tokens, heads) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int HEADS = ${heads};
const int ELEMENTS = ${heads * tokens * tokens};
uniform sampler2D flatTex;
uniform sampler2D keyMaskTex;

float logitAt(int head, int i, int j) {
  return fetch1(flatTex, (i * L + j) * HEADS + head)
    + 1e9 * (fetch1(keyMaskTex, j) - 1.0);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  // A texel never straddles two ROWS here only if L is a multiple of four, so
  // every lane resolves its own (head, i, j) - the same rule the triangle's
  // operands kernel had to learn.
  vec4 value = vec4(0.0);
  for (int lane = 0; lane < 4; ++lane) {
    int element = base + lane;
    if (element >= ELEMENTS) break;
    int row = element / L;
    int j = element - row * L;
    int head = row / L;
    int i = row - head * L;
    float largest = -1e30;
    for (int k = 0; k < L; ++k) largest = max(largest, logitAt(head, i, k));
    float total = 0.0;
    for (int k = 0; k < L; ++k) total += exp(logitAt(head, i, k) - largest);
    float v = exp(logitAt(head, i, j) - largest) / total;
    if (lane == 0) value.x = v;
    else if (lane == 1) value.y = v;
    else if (lane == 2) value.z = v;
    else value.w = v;
  }
  result = value;
}`;

/**
 * `averaged[s][i][h][d] = sum_j weights[h][i][j] * value[s][j][h][d]`.
 *
 * One fragment owns four consecutive d of one (s, i, head), so the four share
 * the weight row they sweep.
 */
export const averageSource = (sequences, tokens, heads, dimension) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int HEADS = ${heads};
const int D = ${dimension};
const int D4 = ${dimension / 4};
const int WIDTH = ${heads * dimension};
const int ELEMENTS = ${sequences * tokens * heads * dimension};
uniform sampler2D weightsTex;
uniform sampler2D valueTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % D4;
  int rest = texel / D4;
  int head = rest % HEADS;
  int row = rest / HEADS;
  int i = row % L;
  int s = row / L;

  int weightBase = (head * L + i) * L;
  int valueChannel = head * D + group * 4;
  vec4 total = vec4(0.0);
  for (int j = 0; j < L; ++j) {
    float weight = fetch1(weightsTex, weightBase + j);
    int valueBase = (s * L + j) * WIDTH + valueChannel;
    total += weight * vec4(
      fetch1(valueTex, valueBase), fetch1(valueTex, valueBase + 1),
      fetch1(valueTex, valueBase + 2), fetch1(valueTex, valueBase + 3));
  }
  result = total;
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadMsaAttentionWeights(execution, weights) {
  /** @type {any} */
  const out = { heads: weights.heads, dimension: weights.dimension };
  for (const name of MSA_ATTENTION_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`af3 msa attention weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.af3.msa.${name}`);
  }
  return out;
}

export function releaseMsaAttentionWeights(execution, weights) {
  for (const name of MSA_ATTENTION_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{msa: Tensor, msaMask: Tensor, pair: Tensor, weights: any,
 *   sequences: number, tokens: number, msaChannels: number, pairChannels: number,
 *   epsilon?: number}} call
 * @returns {Tensor} `sequences * tokens * msaChannels`, the update
 */
export function msaAttention(execution, call) {
  const {
    msa, msaMask, pair, weights: w, sequences, tokens, msaChannels, pairChannels,
    epsilon = 1e-5,
  } = call;
  const heads = w.heads;
  const dimension = w.dimension;
  const width = heads * dimension;
  const rows = sequences * tokens;
  const pairs = tokens * tokens;
  if (msa.elements !== rows * msaChannels) {
    throw new RangeError(
      `af3 msa attention: msa holds ${msa.elements} floats, not`
      + ` ${sequences} x ${tokens} x ${msaChannels}`);
  }
  if (dimension % 4 !== 0) {
    throw new RangeError(
      `af3 msa attention: the head dimension must be a multiple of 4, got ${dimension}`);
  }

  const normalised = layerNorm(execution, {
    x: msa, weight: w.actNormScale, bias: w.actNormOffset,
    rows, channels: msaChannels, epsilon, variance: "fast",
  });
  const pairNormalised = layerNorm(execution, {
    x: pair, weight: w.pairNormScale, bias: w.pairNormOffset,
    rows: pairs, channels: pairChannels, epsilon, variance: "fast",
  });
  const flat = linear(execution, {
    x: pairNormalised, weight: w.pairLogits, rows: pairs,
    inChannels: pairChannels, outChannels: heads, weightLayout: "in-major",
  });
  execution.release(pairNormalised);

  const keyMask = execution.allocate(tokens, "af3.msa.key-mask");
  execution.dispatch({
    source: keyMaskSource(sequences, tokens),
    output: keyMask,
    inputs: { maskTex: msaMask },
    label: `af3.msa.key-mask:${sequences}x${tokens}`,
  });
  const weightsByHead = execution.allocate(heads * pairs, "af3.msa.weights");
  execution.dispatch({
    source: msaWeightsSource(tokens, heads),
    output: weightsByHead,
    inputs: { flatTex: flat, keyMaskTex: keyMask },
    label: `af3.msa.weights:${tokens}x${heads}`,
  });
  execution.release(flat);
  execution.release(keyMask);

  const value = linear(execution, {
    x: normalised, weight: w.vProjection, rows,
    inChannels: msaChannels, outChannels: width, weightLayout: "in-major",
  });
  const averaged = execution.allocate(rows * width, "af3.msa.averaged");
  execution.dispatch({
    source: averageSource(sequences, tokens, heads, dimension),
    output: averaged,
    inputs: { weightsTex: weightsByHead, valueTex: value },
    label: `af3.msa.average:${sequences}x${tokens}x${width}`,
  });
  execution.release(weightsByHead);
  execution.release(value);

  const gate = linear(execution, {
    x: normalised, weight: w.gatingQuery, rows,
    inChannels: msaChannels, outChannels: width, weightLayout: "in-major",
    activation: "sigmoid",
  });
  execution.release(normalised);
  const gated = multiply(execution, averaged, gate);
  execution.release(averaged);
  execution.release(gate);
  const out = linear(execution, {
    x: gated, weight: w.outputProjection, rows,
    inChannels: width, outChannels: msaChannels, weightLayout: "in-major",
  });
  execution.release(gated);
  out.label = "af3.msa-attention";
  return out;
}
