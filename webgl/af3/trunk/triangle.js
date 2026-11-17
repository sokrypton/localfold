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
import { ONE_TARGET, PRELUDE, sized, memoSource } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear, planeFetchGlsl, planeInputs } from "../../kernels/linear.js";
import { multiply } from "../../runtime/elementwise.js";
import {
  CONTRACT_BLOCK_MAX_PAIR_FLOATS, blockedContractSource, contractInto, contractRepackSource, paddedPairLength,
  splitContractSource, widePairBlockSource, wideRepackSource,
} from "../../kernels/triangle-contract.js";

// The pair-blocked contraction moved to webgl/kernels/triangle-contract.js, where AF2's triangle multiplication shares
// it; re-exported so the tools that read it from here still do.
export {
  CONTRACT_BLOCK_MAX_PAIR_FLOATS, blockedContractSource, contractInto, contractRepackSource, paddedPairLength,
  splitContractSource, widePairBlockSource, wideRepackSource,
};

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
const operandsSourceUncached = (length, channels, fused = false, blockRows = 0) => sized({ L: length, PAIRS: length * length, ELEMENTS: channels * length * length }, `${PRELUDE}
layout(location = 0) out vec4 aOut;
layout(location = 1) out vec4 bOut;
const int C = ${channels};
// fused: projection and gate are one [pairs][4C] tensor (fuseTriangleProjections), the gate after the projection
const int ROWW = ${fused ? 4 * channels : 2 * channels};
const int GATE_AT = ${fused ? 2 * channels : 0};
${blockRows > 0 ? `${planeFetchGlsl(blockRows, channels)}
float projectionAt(int element) { int row = element / ROWW; int col = element - row * ROWW; return planeFetch(row, col / 4)[col % 4]; }
float gateAt(int element) { return projectionAt(element); }` : `uniform sampler2D projectionTex;
uniform sampler2D gateTex;
float projectionAt(int element) { return fetch1(projectionTex, element); }
float gateAt(int element) { return fetch1(gateTex, element); }`}
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
    int wide = index * ROWW + c * 2;
    float m = fetch1(maskTex, index);
    float gateA = gateAt(wide + GATE_AT);
    float gateB = gateAt(wide + GATE_AT + 1);
    float left = projectionAt(wide) * m / (1.0 + exp(-gateA));
    float right = projectionAt(wide + 1) * m / (1.0 + exp(-gateB));
    if (lane == 0) { a.x = left; b.x = right; }
    else if (lane == 1) { a.y = left; b.y = right; }
    else if (lane == 2) { a.z = left; b.z = right; }
    else { a.w = left; b.w = right; }
  }
  aOut = a;
  bOut = b;
}`);

/**
 * The two gated operands written straight into the layout the contraction reads: each plane padded to `LP` (zeros
 * past `L`), and, `transposed`, with i and j exchanged - what `padPairPlanesSource` and `planeTransposeSource` used to
 * copy them into in two more passes. The gating arithmetic is `operandsSource`'s.
 *
 * 🔴 TWO PASSES OVER BOTH OPERANDS, EVERY TRIANGLE: at 59 residues every length pads (59 -> 64), so the outgoing
 * direction drew `pad-planes` twice and the incoming `transpose-a` and `transpose-b`, over a pair-sized tensor each.
 * `--no-triangle-fused-layout` is the arm that keeps them.
 */
const operandsLaidOutSourceUncached = (length, channels, paddedLength, transposed, fused = false, blockRows = 0) => sized({ L: length, LP: paddedLength, PLANE: paddedLength * paddedLength, ELEMENTS: channels * paddedLength * paddedLength }, `${PRELUDE}
layout(location = 0) out vec4 aOut;
layout(location = 1) out vec4 bOut;
const int C = ${channels};
// fused: projection and gate are one [pairs][4C] tensor (fuseTriangleProjections), the gate after the projection
const int ROWW = ${fused ? 4 * channels : 2 * channels};
const int GATE_AT = ${fused ? 2 * channels : 0};
${blockRows > 0 ? `${planeFetchGlsl(blockRows, channels)}
float projectionAt(int element) { int row = element / ROWW; int col = element - row * ROWW; return planeFetch(row, col / 4)[col % 4]; }
float gateAt(int element) { return projectionAt(element); }` : `uniform sampler2D projectionTex;
uniform sampler2D gateTex;
float projectionAt(int element) { return fetch1(projectionTex, element); }
float gateAt(int element) { return fetch1(gateTex, element); }`}
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { aOut = vec4(0.0); bOut = vec4(0.0); return; }
  // LP and PLANE are multiples of four, so the four lanes share one plane and one row and only j advances.
  int element = texel * 4;
  int c = element / PLANE;
  int within = element - c * PLANE;
  int i = within / LP;
  int j0 = within - i * LP;
  vec4 a = vec4(0.0);
  vec4 b = vec4(0.0);
  if (i < L) {
    for (int lane = 0; lane < 4; ++lane) {
      int j = j0 + lane;
      if (j >= L) break;
      int index = ${transposed ? "j * L + i" : "i * L + j"};
      int wide = index * ROWW + c * 2;
      float m = fetch1(maskTex, index);
      float gateA = gateAt(wide + GATE_AT);
      float gateB = gateAt(wide + GATE_AT + 1);
      float left = projectionAt(wide) * m / (1.0 + exp(-gateA));
      float right = projectionAt(wide + 1) * m / (1.0 + exp(-gateB));
      if (lane == 0) { a.x = left; b.x = right; }
      else if (lane == 1) { a.y = left; b.y = right; }
      else if (lane == 2) { a.z = left; b.z = right; }
      else { a.w = left; b.w = right; }
    }
  }
  aOut = a;
  bOut = b;
}`);

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
const contractSourceUncached = (length, channels, direction, divideByLength) => sized({ L: length, PAIRS: length * length, ELEMENTS: length * length * channels }, `${PRELUDE}${ONE_TARGET}
const int C = ${channels};
const int C4 = ${channels / 4};
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
}`);

/**
 * The pair axes padded to a multiple of four with zeros: `[c][L][L]` laid out
 * as `[c][LP][LP]`.
 *
 * 🔴 THIS IS WHAT UNLOCKS EVERY FAST CONTRACTION AT AN ODD LENGTH, AND IT WAS
 * 66% OF AN ESMFold2 FOLD. Both fast paths need four to divide the length: the
 * coalesced `fetch4` along k because a row must start on a texel boundary, and
 * the pair block because a fragment owns four j. At 75 tokens neither is
 * reachable, so ESMFold2 ran the scalar one-float-at-a-time loop - 272 ms a
 * call, **0.4 GFLOP/s**, where the same kernel blocked at 68 tokens reaches
 * 10. Its own profile puts that one kernel at **15.3 s of a 23.2 s fold**.
 *
 * Padding to 76 costs `(76/75)^3` = 4% more multiply-adds and two passes over
 * the operands, and buys the whole factor. The padding is ZEROS, so the k loop
 * running to LP adds exactly `0.0 * 0.0` for the tail - not an approximation,
 * and the sum is the same sum.
 *
 * 🔴 AND IT IS NOT THE PADDED ROW STRIDE THAT WAS MEASURED AND DECLINED. That
 * one laid an ALREADY-ALIGNED row at a wider stride to buy cache locality and
 * lost (webgl/README.md, "A row that wraps"); this one makes an UNALIGNED row
 * aligned so that a different kernel becomes legal. Same machinery, different
 * question - and at a length divisible by four it does nothing at all, which is
 * why bioemu2's 68 tokens are untouched.
 */
const padPairPlanesSourceUncached = (length, channels, paddedLength) => sized({ L: length, LP: paddedLength, PAIRS: length * length, PLANE: paddedLength * paddedLength, ELEMENTS: channels * paddedLength * paddedLength }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  // LP is a multiple of four and so is PLANE, so all four lanes share one
  // plane and one row i - only j advances.
  int base = texel * 4;
  int c = base / PLANE;
  int rest = base - c * PLANE;
  int i = rest / LP;
  int j = rest - i * LP;
  if (i >= L) { result = vec4(0.0); return; }
  int from = c * PAIRS + i * L;
  result = vec4(
    j + 0 < L ? fetch1(xTex, from + j + 0) : 0.0,
    j + 1 < L ? fetch1(xTex, from + j + 1) : 0.0,
    j + 2 < L ? fetch1(xTex, from + j + 2) : 0.0,
    j + 3 < L ? fetch1(xTex, from + j + 3) : 0.0);
}`);

/**
 * Swap the two PAIR axes inside every channel plane: `out[c][i][j] = in[c][j][i]`.
 *
 * 🔴 THIS IS WHAT LETS THE INCOMING CONTRACTION READ WHOLE TEXELS. Outgoing
 * walks `a[c][i][k]` as k advances - a stride of one float, so four k are one
 * RGBA texel - while incoming walks `a[c][k][j]`, a stride of L, and no two of
 * its reads share anything. Transposed, incoming IS outgoing with i and j
 * exchanged: `sum_k a[c][k][j] * b[c][k][i]` is `sum_k aT[c][j][k] * bT[c][i][k]`,
 * which is the outgoing kernel handed bT and aT.
 *
 * One pass over the tensor pays for L passes of strided reads. A fragment owns
 * four consecutive j of one (c, i), which are four different source texels, so
 * this kernel has the bad pattern - once, over L^2 * C elements, where the
 * contraction had it L times over.
 */
const planeTransposeSourceUncached = (length, channels, paddedLength = length) =>
  sized({ L: length, LP: paddedLength, PAIRS: length * length, PLANE: paddedLength * paddedLength, ELEMENTS: channels * paddedLength * paddedLength }, `${PRELUDE}${ONE_TARGET}
// It reads the SOURCE at its own pair axes and writes at the PADDED ones, so
// the transpose and the padding are one pass rather than two. Reading the
// destination's stride here is what reddened the incoming direction at every
// padded length while the aligned ones stayed green.
const int C4 = ${channels / 4};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  // Four consecutive elements of the plane-major tensor are four consecutive j
  // of one (c, i), and each comes from its own row of the source plane. LP is a
  // multiple of four, so the four share one plane and one i.
  int element = texel * 4;
  int plane = element / PLANE;
  int within = element - plane * PLANE;
  int i = within / LP;
  int j = within - i * LP;
  if (i >= L) { result = vec4(0.0); return; }
  int base = plane * PAIRS + i;
  result = vec4(
    j + 0 < L ? fetch1(xTex, base + (j + 0) * L) : 0.0,
    j + 1 < L ? fetch1(xTex, base + (j + 1) * L) : 0.0,
    j + 2 < L ? fetch1(xTex, base + (j + 2) * L) : 0.0,
    j + 3 < L ? fetch1(xTex, base + (j + 3) * L) : 0.0);
}`);

/**
 * The centre norm over an already-contracted pair-major tensor.
 *
 * The statistics are accumulated group by group and then component by
 * component, which is the order the fused kernel used - so the mean, the fast
 * variance and the result are BIT-IDENTICAL and `--no-triangle-split` prices a
 * pass count rather than a precision.
 */
const centreNormSourceUncached = (length, channels) => sized({ ELEMENTS: length * length * channels }, `${PRELUDE}${ONE_TARGET}
const int C = ${channels};
const int C4 = ${channels / 4};
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
}`);

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadTriangleWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  const fused = execution.noTriangleFusedProjection ? null : fuseTriangleProjections(weights);
  for (const name of TRIANGLE_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`af3 triangle weights: ${name} is missing`);
    if (fused !== null && (name === "projection" || name === "gate")) continue;
    out[name] = execution.uploadWeight(values, `w.af3.triangle.${name}`);
  }
  if (fused !== null) out.projectionGate = execution.uploadWeight(fused, "w.af3.triangle.projection-gate");
  return out;
}

/**
 * The two `[C][2C]` in-major projections that both read the normalised pair, as ONE `[C][4C]` - the projection's
 * columns then the gate's - which `triangleMultiplication` runs as one projection. Null on a shape it does not expect.
 */
export function fuseTriangleProjections(weights) {
  const { projection: p, gate: g } = weights;
  if (!(p instanceof Float32Array) || !(g instanceof Float32Array) || p.length !== g.length) return null;
  const channels = Math.round(Math.sqrt(p.length / 2));
  if (2 * channels * channels !== p.length) return null;
  const fused = new Float32Array(4 * channels * channels);
  for (let c = 0; c < channels; c += 1) {
    fused.set(p.subarray(c * 2 * channels, (c + 1) * 2 * channels), c * 4 * channels);
    fused.set(g.subarray(c * 2 * channels, (c + 1) * 2 * channels), c * 4 * channels + 2 * channels);
  }
  return fused;
}

export function releaseTriangleWeights(execution, weights) {
  for (const name of [...TRIANGLE_WEIGHTS, "projectionGate"]) {
    if (weights[name] !== undefined) execution.release(weights[name]);
  }
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
    divideByLength = false, epsilon = 1e-5, residualOf,
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
  // ...one [C][4C] projection where the two were fused at upload (`fuseTriangleProjections`)
  const fusedProjection = w.projectionGate !== undefined;
  const projection = linear(execution, {
    x: normalised, weight: fusedProjection ? w.projectionGate : w.projection, rows: pairs,
    inChannels: channels, outChannels: channels * (fusedProjection ? 4 : 2), weightLayout: "in-major",
    keepPlanes: fusedProjection,
  });
  // ...the fused projection's row planes, where it kept them: the operand draw reads them with no repack between
  const kept = /** @type {any} */ (projection).planes !== undefined ? /** @type {any} */ (projection) : null;
  const gate = fusedProjection ? projection : linear(execution, {
    x: normalised, weight: w.gate, rows: pairs,
    inChannels: channels, outChannels: channels * 2, weightLayout: "in-major",
  });
  // ...straight into the contraction's padded (and, incoming, transposed) layout where the split path wants one
  // (`operandsLaidOutSource`), else as before
  const layoutPadded = paddedPairLength(execution, length, channels);
  const layoutTransposed = direction === "incoming" && layoutPadded % 4 === 0
    && !execution.noTriangleCoalesce && !execution.noTriangleTranspose;
  const laidOut = !execution.noTriangleSplit && !execution.noTriangleFusedLayout
    && (layoutTransposed || layoutPadded !== length);
  const planeElements = laidOut ? channels * layoutPadded * layoutPadded : channels * pairs;
  const a = execution.allocate(planeElements, "af3.triangle.a");
  const b = execution.allocate(planeElements, "af3.triangle.b");
  execution.dispatch({
    source: laidOut
      ? operandsLaidOutSource(length, channels, layoutPadded, layoutTransposed, fusedProjection, kept?.blockRows ?? 0)
      : operandsSource(length, channels, fusedProjection, kept?.blockRows ?? 0),
    outputs: [a, b],
    inputs: kept === null ? { projectionTex: projection, gateTex: gate, maskTex: mask }
      : { ...planeInputs(kept), maskTex: mask },
    label: `af3.triangle.operands${laidOut ? (layoutTransposed ? ":transposed" : ":padded") : ""}:${length}x${channels}`,
  });
  if (kept !== null) for (const plane of kept.planes) execution.release(plane);
  else execution.release(projection);
  if (!fusedProjection) execution.release(gate);

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
    // 🔴 THE INCOMING CONTRACTION IS RUN AS AN OUTGOING ONE OVER TRANSPOSED
    // PLANES, so it can read whole texels too - see `planeTransposeSource`. Two
    // transposes are two passes over the tensor against L passes of
    // single-float reads: measured, the incoming direction was the slower half
    // of this stage even after the outgoing one was coalesced.
    //
    // 🔴 AND THE PAIR AXES ARE PADDED TO A MULTIPLE OF FOUR FIRST, WHICH IS
    // WHAT MAKES EITHER FAST PATH LEGAL AT AN ODD LENGTH - see
    // `padPairPlanesSource`. Both of them need four to divide the length, so at
    // ESMFold2's 75 tokens this stage ran the scalar loop and was 66% of the
    // whole fold. The padding is zeros, so the k loop's tail adds exactly zero.
    const padded = paddedPairLength(execution, length, channels);
    const transposeIncoming = direction === "incoming" && padded % 4 === 0
      && !execution.noTriangleCoalesce && !execution.noTriangleTranspose;
    const paddedPlane = channels * padded * padded;
    if (laidOut) {
      // a and b already ARE the padded (transposed) planes; incoming-transposed is outgoing handed bT and aT
      contractInto(execution, contracted, transposeIncoming ? b : a, transposeIncoming ? a : b, length, channels,
        divideByLength, transposeIncoming ? "incoming-transposed" : direction, padded);
    } else if (transposeIncoming) {
      // The transpose writes at the padded axes in the same pass.
      const aT = execution.allocate(paddedPlane, "af3.triangle.aT");
      const bT = execution.allocate(paddedPlane, "af3.triangle.bT");
      /** @type {[Tensor, Tensor, string][]} */
      const planes = [[a, aT, "a"], [b, bT, "b"]];
      for (const [source, target, name] of planes) {
        execution.dispatch({
          source: planeTransposeSource(length, channels, padded),
          output: target,
          inputs: { xTex: source },
          label: `af3.triangle.transpose-${name}:${length}x${channels}`,
        });
      }
      contractInto(execution, contracted, bT, aT, length, channels, divideByLength,
        "incoming-transposed", padded);
      execution.release(aT);
      execution.release(bT);
    } else if (padded !== length) {
      /** @type {Tensor[]} */
      const pads = [a, b].map((operand, index) => {
        const target = execution.allocate(paddedPlane, `af3.triangle.padded${index}`);
        execution.dispatch({
          source: padPairPlanesSource(length, channels, padded),
          output: target,
          inputs: { xTex: operand },
          label: `af3.triangle.pad-planes:${length}->${padded}x${channels}`,
        });
        return target;
      });
      contractInto(execution, contracted, pads[0], pads[1], length, channels,
        divideByLength, direction, padded);
      for (const operand of pads) execution.release(operand);
    } else {
      contractInto(execution, contracted, a, b, length, channels, divideByLength, direction);
    }
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

  const outputGate = linear(execution, {
    x: normalised, weight: w.gatingLinear, rows: pairs,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
    activation: "sigmoid",
  });
  execution.release(normalised);
  // `residualOf`: `residualOf + projected * gate` in the output projection's own epilogue (`linear`'s `multiplier`),
  // where it was the projection, a multiply and the caller's add
  if (residualOf !== undefined && !execution.noLinearEpilogue) {
    const out = linear(execution, {
      x: centred, weight: w.outputProjection, rows: pairs,
      inChannels: channels, outChannels: channels, weightLayout: "in-major",
      residual: residualOf, multiplier: outputGate,
    });
    execution.release(centred);
    execution.release(outputGate);
    out.label = "af3.triangle";
    return out;
  }
  const projected = linear(execution, {
    x: centred, weight: w.outputProjection, rows: pairs,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(centred);
  let out;
  if (residualOf !== undefined) {
    out = execution.allocate(pairs * channels, "af3.triangle.residual");
    execution.dispatch({
      source: residualProductSource(pairs * channels),
      output: out,
      inputs: { baseTex: residualOf, aTex: projected, bTex: outputGate },
      label: `af3.triangle.residual-product:${length}x${channels}`,
    });
  } else {
    out = multiply(execution, projected, outputGate);
  }
  execution.release(projected);
  execution.release(outputGate);
  out.label = "af3.triangle";
  return out;
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const operandsSource = memoSource(operandsSourceUncached);
export const operandsLaidOutSource = memoSource(operandsLaidOutSourceUncached);
export const contractSource = memoSource(contractSourceUncached);
export const padPairPlanesSource = memoSource(padPairPlanesSourceUncached);
export const planeTransposeSource = memoSource(planeTransposeSourceUncached);
export const centreNormSource = memoSource(centreNormSourceUncached);

/** `base + a * b`, elementwise. */
const residualProductSourceUncached = (elements) => sized({ ELEMENTS: elements }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D baseTex;
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(baseTex, texel) + fetch4(aTex, texel) * fetch4(bTex, texel);
}`);
export const residualProductSource = memoSource(residualProductSourceUncached);
