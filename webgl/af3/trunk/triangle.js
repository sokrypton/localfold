/**
 * AF3's triangle multiplication: outgoing (`ikc,jkc->ijc`) or incoming
 * (`kjc,kic->ijc`).
 *
 *     normalised = LayerNorm(pair)                      AF3's FAST variance
 *     a[c][i][j] = projection[.][c * 2]     * mask * sigmoid(gate[.][c * 2])
 *     b[c][i][j] = projection[.][c * 2 + 1] * mask * sigmoid(gate[.][c * 2 + 1])
 *     product    = contract(a, b)                       over k
 *     centred    = centreNorm(product)                  over the CHANNEL axis
 *     out        = Linear(centred) * sigmoid(Linear(normalised))
 *
 * 🔴 THE SPLIT IS INTERLEAVED, AND ITS TWIN IN THE TRANSITION IS BLOCKED.
 * `cpu/af3/trunk/pairformer.js` states both: "triangle mul INTERLEAVED.
 * projection is (in, 2c), but the result is TRANSPOSED to (2c, i, j) BEFORE
 * being reshaped to (c, 2, i, j) - so channel `ch`'s two halves are outputs
 * `ch * 2` and `ch * 2 + 1`, not `ch` and `ch + c`." Getting it backwards
 * "costs no error and no NaN. It permutes channels and returns a plausible
 * tensor."
 *
 * 🔴 AND THE CENTRE NORM NORMALISES OVER THE CHANNEL AXIS OF A CHANNEL-MAJOR
 * ARRAY. "AF3 spells it LayerNorm(axis=0, param_axis=0), which reads like
 * 'normalise over channels' only once you know the array is channel-major here.
 * Doing it over i or j instead still runs and still returns finite numbers." So
 * this port keeps the product channel-major - `[channels][pairs]`, the axis
 * order the einsum contracts in - and the centre norm gathers across planes,
 * writing its output pair-major for everything downstream.
 *
 * 🔴 AND THE CENTRE NORM USES AF3'S FAST VARIANCE TOO, written out here rather
 * than reusing `kernels/layer-norm.js`, because that kernel reads a row of
 * CONSECUTIVE elements and this one reads a stride of `pairs`.
 *
 * 🔴 AND rosettafold3 DIVIDES THE PRODUCT BY THE SEQUENCE LENGTH, which looks
 * like it cannot matter because a LayerNorm is scale-invariant. "It matters
 * because of the norm's EPSILON: dividing by L shrinks the variance by L^2 -
 * 4624 at 68 tokens - and once the pair track decays with depth, `var` stops
 * dominating `var + eps` and the normalisation lands somewhere else." It is a
 * per-model branch, `divideByLength`, and NOT a weight fold: "folding 1/L into
 * the projection would reproduce the pre-norm tensor, and a LayerNorm's eps
 * does not commute with a scale".
 *
 * 🔴 AND NOTHING HERE HAS A BIAS. One `bias` in the whole pairformer block, on
 * the single track's query projection; these four Linears are bias-free.
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import { multiply } from "../../runtime/elementwise.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const TRIANGLE_WEIGHTS = [
  "leftNormInputScale", "leftNormInputOffset", "projection", "gate",
  "centerNormScale", "centerNormOffset", "outputProjection", "gatingLinear",
];

/**
 * The two gated operands, channel-major: `a[c][i * n + j]` and `b[c][...]`.
 *
 * Channel-major because that is the axis the contraction sweeps `k` along, so a
 * fragment of the contraction reads consecutive texels of one plane. Both are
 * written by one draw to two draw buffers - the gate's sigmoid is computed once
 * for both halves of the pair of channels.
 */
export const operandsSource = (length, channels) => `${PRELUDE}
layout(location = 0) out vec4 aOut;
layout(location = 1) out vec4 bOut;
const int L = ${length};
const int PAIRS = ${length * length};
const int C = ${channels};
const int ELEMENTS = ${channels * length * length};
uniform sampler2D projectionTex;
uniform sampler2D gateTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { aOut = vec4(0.0); bOut = vec4(0.0); return; }
  // 🔴 EVERY LANE RESOLVES ITS OWN CHANNEL, BECAUSE A TEXEL STRADDLES TWO
  // PLANES WHENEVER L*L IS NOT A MULTIPLE OF FOUR - which is most lengths: 25
  // at L=5, 121 at L=11. The first version of this kernel read the channel once
  // from the texel's first element and broke out of the lane loop at the plane
  // boundary, leaving the lanes that belong to the NEXT channel as zero. It
  // measured relRMS 0.13 to 0.31 against both references while they agreed with
  // each other to 2.4e-7 - which is what says the instrument is right and the
  // kernel is not. Four divisions instead of one; the same shape of problem
  // kernels/linear.js documents for AF2's 50 pLDDT bins.
  int base = texel * 4;
  vec4 a = vec4(0.0);
  vec4 b = vec4(0.0);
  for (int lane = 0; lane < 4; ++lane) {
    int element = base + lane;
    if (element >= ELEMENTS) break;
    int c = element / PAIRS;
    int index = element - c * PAIRS;
    // The INTERLEAVED split: channel c's two halves are outputs 2c and 2c + 1.
    int wide = index * 2 * C + c * 2;
    float m = fetch1(maskTex, index);
    float gateA = fetch1(gateTex, wide);
    float gateB = fetch1(gateTex, wide + 1);
    float left = fetch1(projectionTex, wide) * m / (1.0 + exp(-gateA));
    float right = fetch1(projectionTex, wide + 1) * m / (1.0 + exp(-gateB));
    if (lane == 0) { a.x = left; b.x = right; }
    else if (lane == 1) { a.y = left; b.y = right; }
    else if (lane == 2) { a.z = left; b.z = right; }
    else { a.w = left; b.w = right; }
  }
  aOut = a;
  bOut = b;
}`;

/**
 * The contraction and the centre norm in one draw, writing pair-major.
 *
 * 🔴 KEPT ONLY AS THE CONTROL ARM FOR `splitContractSource` BELOW, BECAUSE IT
 * DOES SIXTY-FOUR TIMES THE WORK IT NEEDS TO. A fragment owns four consecutive
 * CHANNELS of one pair and the centre norm needs that pair's whole channel
 * axis, so this fragment contracts ALL `channels` of the pair and keeps four -
 * and the sixty-four fragments of one pair each contract the same 256 channels
 * again. At 68 tokens and 256 channels that is `C * L * 2` = 34,816 texel reads
 * a fragment over 295,936 fragments, **10.3 billion fetches**, which at this
 * context's ~10 G fetches a second is the 1055 ms a call a bioemu2 fold's own
 * profile measures - the single most expensive kernel in that fold, 35% of it.
 * The comment this replaces said the fused form was what let "the centre norm
 * see the channel axis at all without a second pass over a channel-major
 * intermediate", and the second pass costs one read and one write per texel.
 */
export const contractSource = (length, channels, direction, divideByLength) => `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int PAIRS = ${length * length};
const int C = ${channels};
const int C4 = ${channels / 4};
const int ELEMENTS = ${length * length * channels};
uniform sampler2D aTex;
uniform sampler2D bTex;
uniform sampler2D scaleTex;
uniform sampler2D offsetTex;
uniform float epsilon;

float contract(int c, int i, int j) {
  float total = 0.0;
  int plane = c * PAIRS;
  for (int k = 0; k < L; ++k) {
    ${direction === "outgoing"
    ? "total += fetch1(aTex, plane + i * L + k) * fetch1(bTex, plane + j * L + k);"
    : "total += fetch1(aTex, plane + k * L + j) * fetch1(bTex, plane + k * L + i);"}
  }
  return ${divideByLength ? "total / float(L)" : "total"};
}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;

  // The whole channel axis of this pair, for the centre norm's statistics.
  float total = 0.0;
  float totalSquares = 0.0;
  vec4 mine = vec4(0.0);
  for (int g = 0; g < C4; ++g) {
    vec4 v = vec4(contract(g * 4, i, j), contract(g * 4 + 1, i, j),
      contract(g * 4 + 2, i, j), contract(g * 4 + 3, i, j));
    total += v.x + v.y + v.z + v.w;
    totalSquares += dot(v, v);
    if (g == group) mine = v;
  }
  // AF3's fast variance, as everywhere else in this trunk.
  float mean = total / float(C);
  float inverseStd = inversesqrt(totalSquares / float(C) - mean * mean + epsilon);
  result = (mine - mean) * inverseStd * fetch4(scaleTex, group) + fetch4(offsetTex, group);
}`;

/**
 * The contraction alone: a fragment owns four channels of one pair and
 * contracts exactly those four.
 *
 * 🔴 FOUR CONTRACTIONS A FRAGMENT RATHER THAN `channels` OF THEM, which is the
 * whole saving - 544 texel reads where the fused kernel does 34,816, at 68
 * tokens and 256 channels. The sum over `k` is in the same order and the same
 * products, so the intermediate is exactly what the fused kernel computed and
 * then threw away sixty-three sixty-fourths of.
 */
export const splitContractSource = (length, channels, direction, divideByLength,
  coalesce = true) => {
  // 🔴 THE OUTGOING DIRECTION WALKS k ALONG THE CONTIGUOUS AXIS, SO IT CAN READ
  // WHOLE TEXELS - AND `fetch1` WAS ASKING FOR THE SAME TEXEL FOUR TIMES. Four
  // consecutive k of `a[c][i][k]` are four consecutive floats, which is one
  // RGBA texel; `fetch1` fetches that texel and selects one component, so the
  // loop issued four fetches where one would do. The texture cache absorbs the
  // BYTES, but this kernel is bound by the fetch RATE - about 10 G a second -
  // not by bandwidth, so the instruction count is the cost. One `dot` per four
  // k instead of four multiply-adds.
  //
  // `plane` is `c * L * L` and the row offset is `i * L`, so both are multiples
  // of four whenever L is, which is the only case this takes; the general loop
  // stays for an odd length.
  //
  // 🔴 THE INCOMING DIRECTION CANNOT: it walks `a[c][k][j]`, a stride of L
  // floats, so no two of its reads share a texel. Turning it into an outgoing
  // contraction over transposed planes would, and is not done here.
  const coalesced = coalesce && direction === "outgoing" && length % 4 === 0;
  const body = coalesced
    ? `  for (int k4 = 0; k4 < L / 4; ++k4) {
    total += dot(fetch4(aTex, (plane + i * L) / 4 + k4), fetch4(bTex, (plane + j * L) / 4 + k4));
  }`
    : `  for (int k = 0; k < L; ++k) {
    ${direction === "outgoing"
    ? "total += fetch1(aTex, plane + i * L + k) * fetch1(bTex, plane + j * L + k);"
    : "total += fetch1(aTex, plane + k * L + j) * fetch1(bTex, plane + k * L + i);"}
  }`;
  return `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int PAIRS = ${length * length};
const int C4 = ${channels / 4};
const int ELEMENTS = ${length * length * channels};
uniform sampler2D aTex;
uniform sampler2D bTex;

float contract(int c, int i, int j) {
  float total = 0.0;
  int plane = c * PAIRS;
${body}
  return ${divideByLength ? "total / float(L)" : "total"};
}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  int c = group * 4;
  result = vec4(contract(c, i, j), contract(c + 1, i, j),
    contract(c + 2, i, j), contract(c + 3, i, j));
}`;
};

/**
 * The centre norm over an already-contracted pair-major tensor.
 *
 * The statistics are accumulated group by group and then component by
 * component, which is the order the fused kernel used - so the mean, the fast
 * variance and the result are BIT-IDENTICAL and `--no-triangle-split` prices a
 * pass count rather than a precision.
 */
export const centreNormSource = (length, channels) => `${PRELUDE}${ONE_TARGET}
const int C = ${channels};
const int C4 = ${channels / 4};
const int ELEMENTS = ${length * length * channels};
uniform sampler2D xTex;
uniform sampler2D scaleTex;
uniform sampler2D offsetTex;
uniform float epsilon;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int base = texel - group;
  float total = 0.0;
  float totalSquares = 0.0;
  vec4 mine = vec4(0.0);
  for (int g = 0; g < C4; ++g) {
    vec4 v = fetch4(xTex, base + g);
    total += v.x + v.y + v.z + v.w;
    totalSquares += dot(v, v);
    if (g == group) mine = v;
  }
  // AF3's fast variance, as everywhere else in this trunk.
  float mean = total / float(C);
  float inverseStd = inversesqrt(totalSquares / float(C) - mean * mean + epsilon);
  result = (mine - mean) * inverseStd * fetch4(scaleTex, group) + fetch4(offsetTex, group);
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadTriangleWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of TRIANGLE_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`af3 triangle weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.af3.triangle.${name}`);
  }
  return out;
}

export function releaseTriangleWeights(execution, weights) {
  for (const name of TRIANGLE_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, mask: Tensor, weights: Record<string, Tensor>,
 *   length: number, channels: number, direction: "outgoing" | "incoming",
 *   divideByLength?: boolean, epsilon?: number}} call
 * @returns {Tensor} `length^2 * channels`, the update; the caller adds it
 */
export function triangleMultiplication(execution, call) {
  const {
    pair, mask, weights: w, length, channels, direction,
    divideByLength = false, epsilon = 1e-5,
  } = call;
  if (direction !== "outgoing" && direction !== "incoming") {
    throw new RangeError(`af3 triangle: direction is outgoing or incoming, not ${direction}`);
  }
  const pairs = length * length;
  if (pair.elements !== pairs * channels) {
    throw new RangeError(
      `af3 triangle: pair holds ${pair.elements} floats, not ${length}^2 x ${channels}`);
  }
  if (channels % 4 !== 0) {
    throw new RangeError(`af3 triangle: channels must be a multiple of 4, got ${channels}`);
  }

  const normalised = layerNorm(execution, {
    x: pair, weight: w.leftNormInputScale, bias: w.leftNormInputOffset,
    rows: pairs, channels, epsilon, variance: "fast",
  });
  const projection = linear(execution, {
    x: normalised, weight: w.projection, rows: pairs,
    inChannels: channels, outChannels: channels * 2, weightLayout: "in-major",
  });
  const gate = linear(execution, {
    x: normalised, weight: w.gate, rows: pairs,
    inChannels: channels, outChannels: channels * 2, weightLayout: "in-major",
  });
  const a = execution.allocate(channels * pairs, "af3.triangle.a");
  const b = execution.allocate(channels * pairs, "af3.triangle.b");
  execution.dispatch({
    source: operandsSource(length, channels),
    outputs: [a, b],
    inputs: { projectionTex: projection, gateTex: gate, maskTex: mask },
    label: `af3.triangle.operands:${length}x${channels}`,
  });
  execution.release(projection);
  execution.release(gate);

  const centred = execution.allocate(pairs * channels, "af3.triangle.centred");
  if (execution.noTriangleSplit) {
    execution.dispatch({
      source: contractSource(length, channels, direction, divideByLength),
      output: centred,
      inputs: {
        aTex: a, bTex: b, scaleTex: w.centerNormScale, offsetTex: w.centerNormOffset,
      },
      floats: { epsilon },
      label: `af3.triangle.contract:${direction}:${length}x${channels}`
        + `${divideByLength ? ":over-length" : ""}`,
    });
  } else {
    const contracted = execution.allocate(pairs * channels, "af3.triangle.contracted");
    execution.dispatch({
      source: splitContractSource(length, channels, direction, divideByLength,
        !execution.noTriangleCoalesce),
      output: contracted,
      inputs: { aTex: a, bTex: b },
      label: `af3.triangle.contract-only:${direction}:${length}x${channels}`
        + `${divideByLength ? ":over-length" : ""}`,
    });
    execution.dispatch({
      source: centreNormSource(length, channels),
      output: centred,
      inputs: {
        xTex: contracted, scaleTex: w.centerNormScale, offsetTex: w.centerNormOffset,
      },
      floats: { epsilon },
      label: `af3.triangle.centre-norm:${length}x${channels}`,
    });
    execution.release(contracted);
  }
  execution.release(a);
  execution.release(b);

  const projected = linear(execution, {
    x: centred, weight: w.outputProjection, rows: pairs,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(centred);
  const outputGate = linear(execution, {
    x: normalised, weight: w.gatingLinear, rows: pairs,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
    activation: "sigmoid",
  });
  execution.release(normalised);
  const out = multiply(execution, projected, outputGate);
  execution.release(projected);
  execution.release(outputGate);
  out.label = "af3.triangle";
  return out;
}
