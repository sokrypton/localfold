/**
 * AF2's global column attention - what the extra-MSA stack uses instead of
 * ordinary column attention, and a different kernel rather than a configuration
 * of one.
 *
 * The extra stack runs over thousands of alignment rows, so attending over the
 * sequences pairwise is unaffordable. Instead each residue gets ONE query, the
 * masked mean of its column, and the keys and values are SINGLE-headed:
 *
 *     normalised = LayerNorm(msa transposed to [residue][sequence][channels])
 *     mean[i][c] = sum_s normalised[i][s][c] * mask[s][i] / (1e-10 + sum_s mask[s][i])
 *     q[i][h][d] = sum_c mean[i][c] * Wq[c][h][d] * inversesqrt(headDim)
 *     k[i][s][d] = sum_c normalised[i][s][c] * Wk[c][d]        (one head, shared)
 *     v[i][s][d] = sum_c normalised[i][s][c] * Wv[c][d]
 *     a[i][h][d] = softmax_s(q[i][h] . k[i][s]) . v[i][s]
 *     gate[s][i] = sigmoid(normalised[i][s] @ Wg + bg)
 *     out[s][i]  = (a[i] broadcast over s) * gate[s][i] @ Wo + bo
 *
 * 🔴 FOUR THINGS READ OFF THE SHADERS RATHER THAN ASSUMED, each of which would
 * otherwise be a plausible tensor:
 *
 *   - the mean's denominator is `1e-10 + sum_s mask`, not `sum_s mask` and not
 *     the outer product mean's 1e-3
 *   - the query's `inversesqrt(headDim)` is applied to the QUERY as it is
 *     written, so it is already in `q` before any logit is formed
 *   - a masked key sets the logit to **-1e9 outright** (`if (mask == 0.0)
 *     { logit = -1e9; }`), where the ordinary attention adds `1e9 * (mask - 1)`;
 *     the two agree for a 0/1 mask and not for anything else
 *   - `Wq` is `[c][head][d]`, `Wk` and `Wv` are `[c][d]` single-headed, and the
 *     OUTPUT rows are written back in the original `[sequence][residue]` order
 *     while everything in between is `[residue][sequence]`
 *
 * 🔴 AND THE ATTENDED RESULT IS PER RESIDUE, BROADCAST OVER THE SEQUENCES. One
 * value per (residue, head, d) multiplies every sequence's gate, which is the
 * whole reason this is affordable - and is why there is a broadcast kernel here
 * rather than an elementwise multiply.
 */
import { ONE_TARGET, PRELUDE } from "../runtime/execution.js";
import { layerNorm } from "./layer-norm.js";
import { linear } from "./linear.js";
import { transposeLeading } from "../runtime/transpose.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

export const GLOBAL_ATTENTION_WEIGHTS = [
  "queryNormScale", "queryNormOffset", "queryWeight", "keyWeight", "valueWeight",
  "gatingWeight", "gatingBias", "outputWeight", "outputBias",
];

/**
 * `mean[i][c] = sum_s x[i][s][c] * mask[s][i] / (1e-10 + sum_s mask[s][i])`,
 * with the scale the query wants folded in - the projection that reads this is
 * bias-free and homogeneous, so scaling here is the same number and one draw
 * fewer.
 */
export const columnMeanSource = (sequences, length, channels) => `${PRELUDE}${ONE_TARGET}
const int SEQUENCES = ${sequences};
const int L = ${length};
const int C4 = ${channels / 4};
uniform sampler2D xTex;
uniform sampler2D maskTex;
uniform float scale;

void main() {
  int texel = linearTexel();
  if (texel >= L * C4) { result = vec4(0.0); return; }
  int c4 = texel % C4;
  int column = texel / C4;

  vec4 total = vec4(0.0);
  float count = 0.0;
  for (int s = 0; s < SEQUENCES; ++s) {
    float keep = fetch1(maskTex, s * L + column);
    count += keep;
    total += keep * fetch4(xTex, (column * SEQUENCES + s) * C4 + c4);
  }
  result = total * (scale / (1e-10 + count));
}`;

/**
 * `out[i][h][d] = softmax_s(q[i][h] . k[i][s]) . v[i][s][d]`, one head's keys
 * and values shared by every head.
 */
export const globalAttendSource = (sequences, length, heads, headDim) => `${PRELUDE}${ONE_TARGET}
const int SEQUENCES = ${sequences};
const int L = ${length};
const int HEADS = ${heads};
const int HD = ${headDim};
const int HD4 = ${headDim / 4};
uniform sampler2D queryTex;
uniform sampler2D keyTex;
uniform sampler2D valueTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel >= L * HEADS * HD4) { result = vec4(0.0); return; }
  int d4 = texel % HD4;
  int rest = texel / HD4;
  int head = rest % HEADS;
  int column = rest / HEADS;
  int queryBase = (column * HEADS + head) * HD4;

  float runningMax = -1e30;
  float runningSum = 0.0;
  vec4 acc = vec4(0.0);
  for (int s = 0; s < SEQUENCES; ++s) {
    int row = (column * SEQUENCES + s) * HD4;
    float logit = 0.0;
    for (int g = 0; g < HD4; ++g) {
      logit += dot(fetch4(queryTex, queryBase + g), fetch4(keyTex, row + g));
    }
    // A masked key is -1e9 OUTRIGHT here, which is this kernel's convention and
    // not the ordinary attention's additive one.
    if (fetch1(maskTex, s * L + column) == 0.0) logit = -1e9;
    float newMax = max(runningMax, logit);
    float previousScale = exp(runningMax - newMax);
    float weight = exp(logit - newMax);
    runningSum = runningSum * previousScale + weight;
    acc = acc * previousScale + weight * fetch4(valueTex, row + d4);
    runningMax = newMax;
  }
  result = acc / runningSum;
}`;

/**
 * `out[i][s][slot] = attended[i][slot] * gate[i][s][slot]` - one value per
 * residue against every sequence's gate.
 */
export const broadcastOverSequencesSource = (sequences, length, gates) => `${PRELUDE}${ONE_TARGET}
const int SEQUENCES = ${sequences};
const int L = ${length};
const int G4 = ${gates / 4};
uniform sampler2D attendedTex;
uniform sampler2D gateTex;

void main() {
  int texel = linearTexel();
  if (texel >= L * SEQUENCES * G4) { result = vec4(0.0); return; }
  int g4 = texel % G4;
  int row = texel / G4;
  int column = row / SEQUENCES;
  result = fetch4(attendedTex, column * G4 + g4) * fetch4(gateTex, texel);
}`;

/**
 * @param {import("../runtime/execution.js").Execution} execution
 */
export function uploadGlobalAttentionWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of GLOBAL_ATTENTION_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`global attention weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.global.${name}`);
  }
  return out;
}

export function releaseGlobalAttentionWeights(execution, weights) {
  for (const name of GLOBAL_ATTENTION_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{msa: Tensor, mask: Tensor, weights: Record<string, Tensor>,
 *   sequences: number, length: number, channels: number, heads: number,
 *   epsilon?: number}} call
 * @returns {Tensor} `[sequence][residue][channels]`; the caller owns it
 */
export function globalAttention(execution, call) {
  const {
    msa, mask, weights: w, sequences, length, channels, heads, epsilon = 1e-5,
  } = call;
  const headDim = channels / heads;
  const gates = heads * headDim;
  const rows = sequences * length;
  if (!Number.isInteger(headDim) || headDim % 4 !== 0) {
    throw new RangeError(
      `globalAttention: ${channels} channels over ${heads} heads gives a head width of `
      + `${headDim}, which must be a whole multiple of 4 (AF2's extra stack is 64 over 8)`);
  }
  if (msa.elements !== rows * channels) {
    throw new RangeError(
      `globalAttention: msa holds ${msa.elements} floats, not `
      + `${sequences} x ${length} x ${channels}`);
  }

  // Everything between the transpose and the output is residue-major.
  const byResidue = transposeLeading(execution, {
    x: msa, outer: sequences, inner: length, channels,
  });
  const normalised = layerNorm(execution, {
    x: byResidue, weight: w.queryNormScale, bias: w.queryNormOffset,
    rows, channels, epsilon,
  });
  execution.release(byResidue);

  const means = execution.allocate(length * channels, "global.means");
  execution.dispatch({
    source: columnMeanSource(sequences, length, channels),
    output: means,
    inputs: { xTex: normalised, maskTex: mask },
    floats: { scale: 1 / Math.sqrt(headDim) },
    label: `global.column-mean:${length}x${channels}`,
  });
  const query = linear(execution, {
    x: means, weight: w.queryWeight,
    rows: length, inChannels: channels, outChannels: gates, weightLayout: "in-major",
  });
  execution.release(means);

  const key = linear(execution, {
    x: normalised, weight: w.keyWeight,
    rows, inChannels: channels, outChannels: headDim, weightLayout: "in-major",
  });
  const value = linear(execution, {
    x: normalised, weight: w.valueWeight,
    rows, inChannels: channels, outChannels: headDim, weightLayout: "in-major",
  });

  const attended = execution.allocate(length * gates, "global.attended");
  execution.dispatch({
    source: globalAttendSource(sequences, length, heads, headDim),
    output: attended,
    inputs: { queryTex: query, keyTex: key, valueTex: value, maskTex: mask },
    label: `global.attend:${length}x${heads}x${headDim}`,
  });
  for (const tensor of [query, key, value]) execution.release(tensor);

  const gate = linear(execution, {
    x: normalised, weight: w.gatingWeight, bias: w.gatingBias,
    rows, inChannels: channels, outChannels: gates,
    activation: "sigmoid", weightLayout: "in-major",
  });
  execution.release(normalised);

  const gated = execution.allocate(rows * gates, "global.gated");
  execution.dispatch({
    source: broadcastOverSequencesSource(sequences, length, gates),
    output: gated,
    inputs: { attendedTex: attended, gateTex: gate },
    label: `global.broadcast:${length}x${sequences}x${gates}`,
  });
  execution.release(attended);
  execution.release(gate);

  const projected = linear(execution, {
    x: gated, weight: w.outputWeight, bias: w.outputBias,
    rows, inChannels: gates, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(gated);

  // ...and back to sequence-major, which is the order the residual target is in.
  const output = transposeLeading(execution, {
    x: projected, outer: length, inner: sequences, channels,
  });
  execution.release(projected);
  output.label = "global-attention";
  return output;
}
