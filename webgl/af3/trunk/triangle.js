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
import { ONE_TARGET, PRELUDE, targetDeclarations, sized, memoSource } from "../../runtime/execution.js";
import { TEXTURE_WIDTH } from "../../runtime/context.js";
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
const operandsSourceUncached = (length, channels, fused = false) => sized({ L: length, PAIRS: length * length, ELEMENTS: channels * length * length }, `${PRELUDE}
layout(location = 0) out vec4 aOut;
layout(location = 1) out vec4 bOut;
const int C = ${channels};
// fused: projection and gate are one [pairs][4C] tensor (fuseTriangleProjections), the gate after the projection
const int ROWW = ${fused ? 4 * channels : 2 * channels};
const int GATE_AT = ${fused ? 2 * channels : 0};
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
    int wide = index * ROWW + c * 2;
    float m = fetch1(maskTex, index);
    float gateA = fetch1(gateTex, wide + GATE_AT);
    float gateB = fetch1(gateTex, wide + GATE_AT + 1);
    float left = fetch1(projectionTex, wide) * m / (1.0 + exp(-gateA));
    float right = fetch1(projectionTex, wide + 1) * m / (1.0 + exp(-gateB));
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
const operandsLaidOutSourceUncached = (length, channels, paddedLength, transposed, fused = false) => sized({ L: length, LP: paddedLength, PLANE: paddedLength * paddedLength, ELEMENTS: channels * paddedLength * paddedLength }, `${PRELUDE}
layout(location = 0) out vec4 aOut;
layout(location = 1) out vec4 bOut;
const int C = ${channels};
// fused: projection and gate are one [pairs][4C] tensor (fuseTriangleProjections), the gate after the projection
const int ROWW = ${fused ? 4 * channels : 2 * channels};
const int GATE_AT = ${fused ? 2 * channels : 0};
uniform sampler2D projectionTex;
uniform sampler2D gateTex;
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
      float gateA = fetch1(gateTex, wide + GATE_AT);
      float gateB = fetch1(gateTex, wide + GATE_AT + 1);
      float left = fetch1(projectionTex, wide) * m / (1.0 + exp(-gateA));
      float right = fetch1(projectionTex, wide + 1) * m / (1.0 + exp(-gateB));
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
 * The contraction alone: a fragment owns four channels of one pair and
 * contracts exactly those four.
 *
 * 🔴 FOUR CONTRACTIONS A FRAGMENT RATHER THAN `channels` OF THEM, which is the
 * whole saving - 544 texel reads where the fused kernel does 34,816, at 68
 * tokens and 256 channels. The sum over `k` is in the same order and the same
 * products, so the intermediate is exactly what the fused kernel computed and
 * then threw away sixty-three sixty-fourths of.
 */
const splitContractSourceUncached = (length, channels, direction, divideByLength,
  coalesce = true, paddedLength = length) => {
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
  const coalesced = coalesce && direction === "outgoing" && paddedLength % 4 === 0;
  const body = coalesced
    ? `  for (int k4 = 0; k4 < LP / 4; ++k4) {
    total += dot(fetch4(aTex, (plane + i * LP) / 4 + k4), fetch4(bTex, (plane + j * LP) / 4 + k4));
  }`
    : `  for (int k = 0; k < L; ++k) {
    ${direction === "outgoing"
    ? "total += fetch1(aTex, plane + i * LP + k) * fetch1(bTex, plane + j * LP + k);"
    : "total += fetch1(aTex, plane + k * LP + j) * fetch1(bTex, plane + k * LP + i);"}
  }`;
  return sized({ L: length, LP: paddedLength, PLANE: paddedLength * paddedLength, ELEMENTS: length * length * channels }, `${PRELUDE}${ONE_TARGET}
// The operands' pair axes, padded to a multiple of four so the coalesced loop
// is legal; equal to L whenever nothing needed padding.
const int C4 = ${channels / 4};
uniform sampler2D aTex;
uniform sampler2D bTex;

float contract(int c, int i, int j) {
  float total = 0.0;
  int plane = c * PLANE;
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
}`);
};

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
 * The contraction with a PAIR BLOCK: two i and four j a fragment, into eight
 * render targets.
 *
 * 🔴 THE SPLIT CONTRACTION RUNS AT FOUR FLOPS A FETCH WHERE THE PROJECTION
 * REACHES TWENTY-THREE, AND SHARING THE OPERANDS IS THE ONLY WAY UP. A fragment
 * owning four channels of one pair reads `a[i][*][c]` and `b[j][*][c]` for each
 * of those four channels - and the four channels live in four different planes,
 * so that is 136 texel reads for 272 multiply-adds. Measured inside a 48-block
 * trunk the contraction is about half of it, at roughly **1.8 GFLOP/s** where
 * the same device's projections reach 70.
 *
 * Nothing about the sweep can be made cheaper; the operands can be REUSED. A
 * block of two i and four j computes eight pairs from two `a` rows and four `b`
 * rows, so a k-step costs six fetches instead of eight for four times the
 * arithmetic: **10.7 flops a fetch, 2.7x**. Eight pairs of four channels is
 * exactly `MAX_DRAW_BUFFERS` targets, and `blockRepackSource` interleaves them.
 *
 * The channel loop is unrolled so only eight accumulators are live at a time
 * rather than thirty-two, and so that a `vec4` is never indexed by a variable.
 */
/**
 * @param residentOperands a DIAGNOSTIC ONLY, and never a kernel: it forces every
 *   fragment onto pair block ZERO, so all of them read the same two rows and
 *   four columns and the operand working set collapses from 9.4 MiB to a few
 *   KiB. It computes the wrong answer and `bench-webgl-contract.js` never
 *   compares it for one.
 *
 *   🔴 IT EXISTS BECAUSE THE `aliased` ARM DOES NOT ASK WHAT IT CLAIMS TO.
 *   Handing the kernel one texture as both operands halves the DISTINCT
 *   footprint and leaves every fragment reading its own pair rows, so the
 *   working set is still 4.7 MiB and still past the cache - which is why that
 *   arm reads 1.05-1.14x and rules out far less than it appears to. The
 *   projection's equivalent (`x-aliased` in probe-webgl-linear-operand.js)
 *   makes every fragment read the SAME rows, and there it is 3.44x. This is
 *   that arm for the contraction.
 */
const blockedContractSourceUncached = (length, channels, direction, divideByLength,
  channelSlowest = true, paddedLength = length, tileMapped = false,
  residentOperands = false) => {
  const planes = 8;
  const component = ["x", "y", "z", "w"];
  // For the outgoing direction a row is `a[i][k]`, contiguous in k; for the
  // incoming one the caller has already transposed both operands, so it is the
  // same expression.
  const row = (tex, pair) => `(plane + ${pair} * LP) / 4 + k4`;
  const channelBlock = (c4) => `  {
    float s00 = 0.0, s01 = 0.0, s02 = 0.0, s03 = 0.0;
    float s10 = 0.0, s11 = 0.0, s12 = 0.0, s13 = 0.0;
    int plane = (cgroup * 4 + ${c4}) * PLANE;
    for (int k4 = 0; k4 < LP / 4; ++k4) {
      vec4 a0 = fetch4(aTex, ${row("aTex", "i0")});
      vec4 a1 = fetch4(aTex, ${row("aTex", "i1")});
      vec4 b0 = fetch4(bTex, ${row("bTex", "j0")});
      vec4 b1 = fetch4(bTex, ${row("bTex", "j1")});
      vec4 b2 = fetch4(bTex, ${row("bTex", "j2")});
      vec4 b3 = fetch4(bTex, ${row("bTex", "j3")});
      s00 += dot(a0, b0); s01 += dot(a0, b1); s02 += dot(a0, b2); s03 += dot(a0, b3);
      s10 += dot(a1, b0); s11 += dot(a1, b1); s12 += dot(a1, b2); s13 += dot(a1, b3);
    }
${[0, 1].map((di) => [0, 1, 2, 3].map((dj) =>
  `    result${di * 4 + dj}.${component[c4]} = s${di}${dj}${divideByLength ? " / float(L)" : ""};`)
  .join("\n")).join("\n")}
  }`;
  return sized({ L: length, LP: paddedLength, PLANE: paddedLength * paddedLength, JBLOCKS: paddedLength / 4, BLOCKS: (paddedLength / 2) * (paddedLength / 4), TILEW: Math.max(1, Math.floor(2048 / (channels / 4))) }, `${PRELUDE}${targetDeclarations(planes)}
// The padded pair axes: the block sweeps the PADDED grid, whose tail rows and
// columns are zeros, and the repack reads only the true L of them.
const int C4 = ${channels / 4};
const int IBLOCKS = ${paddedLength / 2};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (${tileMapped ? "(texel / W) * TILEW + (texel % TILEW) >= BLOCKS"
    : "texel >= IBLOCKS * JBLOCKS * C4"}) {
${Array.from({ length: planes }, (_, p) => `    result${p} = vec4(0.0);`).join("\n")}
    return;
  }
${tileMapped
  ? `  // 🔴 A TILE-AWARE MAPPING, BECAUSE A RASTERISATION TILE IS TWO-DIMENSIONAL
  // AND THE TEXEL INDEX IS NOT. A tile covers a square of (x, y) and the texel
  // index is y * W + x, so consecutive ROWS of a tile are W apart in the index -
  // and with BLOCKS of them per channel a 32-row tile spans 32 * W / BLOCKS
  // channel groups, about a hundred at 68 tokens. Ordering the channel slowest
  // fixes the one-dimensional neighbour and leaves that untouched.
  //
  // Here the channel group is chosen by x / TILEW with TILEW = W / C4, so every
  // fragment sharing an x RANGE shares a channel - and a tile is a range of x
  // crossed with a range of y, so a whole tile shares one channel's operands.
  // It needs C4 to divide W, which every power-of-two channel count does.
  int lane = texel % TILEW;
  int cgroup = (texel % W) / TILEW;
  int block = (texel / W) * TILEW + lane;`
  : channelSlowest
    ? `  int block = texel % BLOCKS;
  int cgroup = texel / BLOCKS;`
    : `  int cgroup = texel % C4;
  int block = texel / C4;`}
  int jblock = ${residentOperands ? "0" : "block % JBLOCKS"};
  int iblock = ${residentOperands ? "0" : "block / JBLOCKS"};
  int i0 = iblock * 2;
  int i1 = i0 + 1;
  int j0 = jblock * 4;
  int j1 = j0 + 1;
  int j2 = j0 + 2;
  int j3 = j0 + 3;
${[0, 1, 2, 3].map(channelBlock).join("\n")}
}`);
};

/**
 * The contraction with a 4 i x 8 j x ONE channel block, against the 2 x 4 x 4
 * one above.
 *
 * 🔴 CHANNELS A FRAGMENT COST LOCALITY AND BUY NO INTENSITY, which is the whole
 * reason this shape exists. `bench-webgl-contract.js`'s `resident-tiled` arm
 * puts the contraction at **3.91x off its own operand-resident rate** - 1.26 ms
 * against 4.94, 127 GFLOP/s against 32 - and the working set a rasterisation
 * tile touches is `rows * channels * LP`: 136 pair rows by 4 channels by LP
 * values is **148 KiB** at 68 tokens and 256 channels. One channel a fragment
 * makes it **37 KiB**. And the same 32-float output budget then buys a wider
 * pair block, whose intensity is `8 * I * J / (I + J)`: **21.3 flops a fetch4**
 * at 4 x 8 against the 2 x 4 block's 10.7. Two factors from one change.
 *
 * Measured in the bench, block AND repack against block and repack:
 *
 * | L | 2x4x4 | 4x8x1 | factor |
 * |---|---|---|---|
 * | 64 | 3.49 ms | **2.61** | **1.33x** |
 * | 72 | 6.63 | **3.30** | **2.01x** |
 * | 128 | 25.84 | **14.64** | **1.77x** |
 *
 * 🔴 ITS x AXIS IS THE CHANNEL AND NOT THE CHANNEL GROUP, so `TILEW` is
 * `W / channels` where the other kernel's is `W / (channels / 4)` - 8 rather
 * than 32 at 256 channels. A TILEW-wide run of texels is one channel and TILEW
 * pair blocks, and a tile then holds one channel's operands.
 *
 * 🔴 AND IT NEEDS THE PAIR AXES PADDED TO A MULTIPLE OF EIGHT, where
 * `paddedPairLength` pads to four: at 68 tokens `JBLOCKS` would be 8.5 and the
 * shader does not compile - "cannot convert from const float to const highp
 * int", which is what a non-integer in a `const int` looks like. So this kernel
 * asks for its own padding and 68 tokens become 72, which is 1.19x the padded
 * arithmetic and is already inside the figures above.
 */
const widePairBlockSourceUncached = (length, channels, divideByLength,
  paddedLength = length, tileWidth = 8) => {
  const I = 4;
  const J = 8;
  const slots = [];
  for (let i = 0; i < I; i += 1) {
    for (let j = 0; j < J; j += 1) slots.push(`s${i}_${j}`);
  }
  const value = (name) => (divideByLength ? `${name} / float(L)` : name);
  return sized({ L: length, LP: paddedLength, PLANE: paddedLength * paddedLength, JBLOCKS: paddedLength / J, BLOCKS: (paddedLength / I) * (paddedLength / J), TILEW: tileWidth }, `${PRELUDE}${targetDeclarations(8)}
const int C = ${channels};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  int lane = texel % TILEW;
  int channel = (texel % W) / TILEW;
  int block = (texel / W) * TILEW + lane;
  // The grid is whole texture rows, so its tail is real draws and must write
  // rather than wrap - a target left unwritten is a texel nothing wrote.
  if (block >= BLOCKS || channel >= C) {
${Array.from({ length: 8 }, (_, p) => `    result${p} = vec4(0.0);`).join("\n")}
    return;
  }
  int jblock = block % JBLOCKS;
  int iblock = block / JBLOCKS;
  int plane = channel * PLANE;
${slots.map((a) => `  float ${a} = 0.0;`).join("\n")}
  for (int k4 = 0; k4 < LP / 4; ++k4) {
${Array.from({ length: I }, (_, i) =>
  `    vec4 a${i} = fetch4(aTex, (plane + (iblock * ${I} + ${i}) * LP) / 4 + k4);`).join("\n")}
${Array.from({ length: J }, (_, j) =>
  `    vec4 b${j} = fetch4(bTex, (plane + (jblock * ${J} + ${j}) * LP) / 4 + k4);`).join("\n")}
${Array.from({ length: I }, (_, i) => Array.from({ length: J }, (_, j) =>
    `    s${i}_${j} += dot(a${i}, b${j});`).join("\n")).join("\n")}
  }
${Array.from({ length: 8 }, (_, p) =>
  `  result${p} = vec4(${[0, 1, 2, 3].map((c) => value(slots[p * 4 + c])).join(", ")});`).join("\n")}
}`);
};

/**
 * The wide block's planes back into `[pair][channel group]`.
 *
 * 🔴 IT IS FOUR `fetch4` WHERE THE 2 x 4 x 4 REPACK IS ONE, AND THAT IS THE
 * SHAPE'S REAL PRICE. An output texel is four CHANNELS of one pair, and a wide
 * fragment holds one channel of thirty-two pairs - so the four channels come
 * from four different fragments, which are four source texels of the SAME
 * plane, and (i, j) picks the plane and the component. That cost is inside the
 * bench figures above, which time block and repack together; timing the block
 * alone read 2.76x at L = 72 where the pair reads 2.01x.
 */
const wideRepackSourceUncached = (length, channels, paddedLength = length,
  tileWidth = 8) => sized({ L: length, JBLOCKS: paddedLength / 8, TILEW: tileWidth, ELEMENTS: length * length * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
${Array.from({ length: 8 }, (_, p) => `uniform sampler2D plane${p};`).join("\n")}

float pick(vec4 v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int cgroup = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  int block = (i / 4) * JBLOCKS + (j / 8);
  // Not named flat: that is a GLSL interpolation qualifier and a reserved word.
  int slot = (i % 4) * 8 + (j % 8);
  int plane = slot / 4;
  int component = slot - plane * 4;
  int lane = block % TILEW;
  int y = block / TILEW;
  vec4 gathered = vec4(0.0);
  for (int c = 0; c < 4; ++c) {
    int source = y * W + (cgroup * 4 + c) * TILEW + lane;
    vec4 v =
${Array.from({ length: 7 }, (_, p) =>
    `      plane == ${p} ? fetch4(plane${p}, source) :`).join("\n")}
      fetch4(plane7, source);
    float picked = pick(v, component);
    if (c == 0) gathered.x = picked;
    else if (c == 1) gathered.y = picked;
    else if (c == 2) gathered.z = picked;
    else gathered.w = picked;
  }
  result = gathered;
}`);

/**
 * The largest pair tensor, in FLOATS, for which the pair block beats the
 * one-pair-a-fragment kernel.
 *
 * 🔴 THE BLOCK HAD NO CEILING AND IS THE WRONG KERNEL ABOVE ONE, by up to 4.8x.
 * `tools/gpu/bench-webgl-contract.js` over a grid of length and channels, with
 * the two kernels' answers identical, gives the blocked kernel's time divided
 * by the one-pair kernel's - under one means the block LOSES:
 *
 * | `L * L * channels` | 296K | 590K | 592K | 1.05M | 1.18M | 2.10M | 2.36M | 4.19M |
 * |---|---|---|---|---|---|---|---|---|
 * | blocked / one-pair | 5.10 | 1.71 | 4.62 | 1.12 | 3.95 | **0.38** | **0.59** | **0.21** |
 *
 * Nine points over three lengths and three channel widths, and `L * L *
 * channels` SEPARATES THEM COMPLETELY: every one at or below 1,183,744 wins and
 * every one at or above 2,097,152 loses, with no overlap on either side. So the
 * threshold is the one quantity and not a pair of them, and it is set between
 * the two, where nothing is measured.
 *
 * 🔴 AND THE MECHANISM IS THE REPACK ROUND TRIP, WHICH IS WHY IT IS THE TENSOR
 * AND NOT THE LENGTH. The block's saving is fetches - 10.7 flops a fetch where
 * the one-pair kernel asks for the same texel four times - and its cost is that
 * it writes `L * L * channels` floats across eight planes and then reads all of
 * them back to interleave. That cost is proportional to the tensor while the
 * saving is not, so a big enough pair track spends more on the round trip than
 * the fetches were worth.
 *
 * bioemu2 at 68 tokens and 256 channels is **1,183,744** - the largest
 * measured win, and the top of the range - so this costs that fold nothing and
 * is worth 1.7x to 2.6x on the lengths above it. AF3's 128-channel pair track
 * crosses at 128 tokens and bioemu2's 256-channel one at about 86.
 */
export const CONTRACT_BLOCK_MAX_PAIR_FLOATS = 1572864;

/**
 * The eight pair planes back into `[pair][channel]`.
 *
 * Plane `(i & 1) * 4 + (j & 3)` holds the pair at
 * `((i >> 1) * JBLOCKS + (j >> 2)) * C4 + cgroup`.
 */
const contractRepackSourceUncached = (length, channels, channelSlowest = true,
  paddedLength = length, tileMapped = false) => sized({ L: length, JBLOCKS: paddedLength / 4, BLOCKS: (paddedLength / 2) * (paddedLength / 4), TILEW: Math.max(1, Math.floor(2048 / (channels / 4))), ELEMENTS: length * length * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
// Its OUTPUT is the true L * L pairs and its SOURCE is the padded grid, which
// is the one place both lengths appear.
${Array.from({ length: 8 }, (_, p) => `uniform sampler2D plane${p};`).join("\n")}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int cgroup = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  int block = (i >> 1) * JBLOCKS + (j >> 2);
  int source = ${tileMapped
    ? "(block / TILEW) * W + cgroup * TILEW + (block % TILEW)"
    : channelSlowest ? "cgroup * BLOCKS + block" : "block * C4 + cgroup"};
  int plane = (i & 1) * 4 + (j & 3);
${Array.from({ length: 7 }, (_, p) =>
  `  if (plane == ${p}) { result = fetch4(plane${p}, source); return; }`).join("\n")}
  result = fetch4(plane7, source);
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
 * One contraction into `output`: the pair-blocked kernel where the shape takes
 * it, the one-pair-a-fragment kernel otherwise.
 *
 * `direction` is only a label here - the caller has already transposed the
 * incoming case into an outgoing one.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 */
function contractInto(execution, output, a, b, length, channels, divideByLength,
  direction, padded = length) {
  // The pair block needs the PADDED length divisible by four - two i and four j
  // a fragment - and eight render targets; `MAX_DRAW_BUFFERS` is at least four
  // everywhere, so the one-pair kernel stays for a context that has only four.
  // The size ceiling is on the TRUE tensor, which is what the repack moves.
  const blocked = padded % 4 === 0 && execution.maxDrawBuffers >= 8
    && !execution.noContractBlock
    && (execution.noContractCeiling
      || length * length * channels <= CONTRACT_BLOCK_MAX_PAIR_FLOATS);
  const tag = padded === length ? "" : `:padded${padded}`;
  if (!blocked) {
    execution.dispatch({
      source: splitContractSource(length, channels, direction === "incoming" ? "incoming" : "outgoing",
        divideByLength, !execution.noTriangleCoalesce, padded),
      output,
      inputs: { aTex: a, bTex: b },
      label: `af3.triangle.contract-only:${direction}:${length}x${channels}${tag}`
        + `${divideByLength ? ":over-length" : ""}`,
    });
    return output;
  }
  // 🔴 THE WIDE PAIR BLOCK FIRST, WHERE IT IS LEGAL: 4 i x 8 j x ONE channel
  // instead of 2 x 4 x 4, which is 1.33-2.01x in the bench because channels a
  // fragment cost locality and buy no intensity. See `widePairBlockSource`. It
  // needs the padded length divisible by EIGHT and `W / channels` to be at
  // least one TILEW, so a channel width past 2048 falls back.
  const wideTile = Math.floor(TEXTURE_WIDTH / channels);
  const wide = !execution.noContractWide && padded % 8 === 0 && wideTile >= 1
    && execution.maxDrawBuffers >= 8;
  if (wide) {
    const wideBlocks = (padded / 4) * (padded / 8);
    const wideTexels = Math.ceil(wideBlocks / wideTile) * TEXTURE_WIDTH;
    const widePlanes = Array.from({ length: 8 },
      (_, p) => execution.allocate(wideTexels * 4, `af3.triangle.wide-plane${p}`));
    execution.dispatch({
      source: widePairBlockSource(length, channels, divideByLength, padded, wideTile),
      outputs: widePlanes,
      inputs: { aTex: a, bTex: b },
      label: `af3.triangle.contract-wide:${direction}:${length}x${channels}${tag}`
        + `${divideByLength ? ":over-length" : ""}`,
    });
    execution.dispatch({
      source: wideRepackSource(length, channels, padded, wideTile),
      output,
      inputs: Object.fromEntries(widePlanes.map((plane, p) => [`plane${p}`, plane])),
      label: `af3.triangle.wide-repack:${length}x${channels}${tag}`,
    });
    for (const plane of widePlanes) execution.release(plane);
    return output;
  }
  // 🔴 THE TILE-AWARE MAPPING WHERE THE CHANNEL GROUPS DIVIDE THE TEXTURE
  // WIDTH, which is 3.29x and is what the fragment ORDER was a one-dimensional
  // approximation of. It needs `channels / 4` to divide W, which every
  // power-of-two channel width does; `TILEW` is then `W / C4` and a whole
  // rasterisation tile shares one channel's operands.
  const groups = channels / 4;
  const tileMapped = !execution.noContractTile && TEXTURE_WIDTH % groups === 0;
  const tileWidth = TEXTURE_WIDTH / groups;
  const blocks = (padded / 2) * (padded / 4);
  // The tiled grid is whole texture rows, which is 5% more fragments at 68
  // tokens and the price of the mapping.
  const planeTexels = tileMapped
    ? Math.ceil(blocks / tileWidth) * TEXTURE_WIDTH
    : blocks * groups;
  const planes = Array.from({ length: 8 },
    (_, p) => execution.allocate(planeTexels * 4, `af3.triangle.pair-plane${p}`));
  execution.dispatch({
    source: blockedContractSource(length, channels, direction, divideByLength,
      !execution.noContractChannelSlowest, padded, tileMapped),
    outputs: planes,
    inputs: { aTex: a, bTex: b },
    label: `af3.triangle.contract-block:${direction}:${length}x${channels}${tag}`
      + `${tileMapped ? ":tiled" : ""}${divideByLength ? ":over-length" : ""}`,
  });
  execution.dispatch({
    source: contractRepackSource(length, channels,
      !execution.noContractChannelSlowest, padded, tileMapped),
    output,
    inputs: Object.fromEntries(planes.map((plane, p) => [`plane${p}`, plane])),
    label: `af3.triangle.contract-repack:${length}x${channels}${tag}`,
  });
  for (const plane of planes) execution.release(plane);
  return output;
}

/**
 * The pair axes rounded up to a multiple of four - or EIGHT where the wide pair
 * block will run, since its 8 j a fragment needs it.
 *
 * 🔴 PADDING TO EIGHT IS 1.19x THE PADDED ARITHMETIC AT 68 TOKENS (72 cubed
 * against 68 cubed) AND THE WIDE BLOCK PAYS FOR IT TWICE OVER - 3.30 ms at 72
 * against the 2 x 4 x 4 kernel's 4.85 at 68. Asked for only when that kernel
 * can actually run, so a context without eight draw buffers or with
 * `--no-contract-wide` keeps the four-padding and the smaller tensors.
 */
const paddedPairLength = (execution, length, channels = 0) => {
  if (execution.noContractPad) return length;
  const wideWanted = !execution.noContractWide && channels > 0
    && Math.floor(TEXTURE_WIDTH / channels) >= 1 && execution.maxDrawBuffers >= 8;
  const multiple = wideWanted ? 8 : 4;
  return length % multiple === 0 ? length : Math.ceil(length / multiple) * multiple;
};

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
  });
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
    source: laidOut ? operandsLaidOutSource(length, channels, layoutPadded, layoutTransposed, fusedProjection)
      : operandsSource(length, channels, fusedProjection),
    outputs: [a, b],
    inputs: { projectionTex: projection, gateTex: gate, maskTex: mask },
    label: `af3.triangle.operands${laidOut ? (layoutTransposed ? ":transposed" : ":padded") : ""}:${length}x${channels}`,
  });
  execution.release(projection);
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
  // `residualOf`: return `residualOf + projected * gate` from one draw where it was a multiply and the caller's add
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
export const splitContractSource = memoSource(splitContractSourceUncached);
export const planeTransposeSource = memoSource(planeTransposeSourceUncached);
export const blockedContractSource = memoSource(blockedContractSourceUncached);
export const widePairBlockSource = memoSource(widePairBlockSourceUncached);
export const wideRepackSource = memoSource(wideRepackSourceUncached);
export const contractRepackSource = memoSource(contractRepackSourceUncached);
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
