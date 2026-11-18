/**
 * BioEmu2's denoiser in WebGL2: a dual stream of sixteen layers.
 *
 *     c_in     = 1 / sqrt(sigma^2 + sigma_data^2)
 *     r        = mask(x * c_in)
 *     cond     = single_embed(LN[s_trunk | s_inputs]) + fourier(log(sigma/sd)/4)
 *     q        = ctx.q0 + r_to_q(r)                     the ATOM stream
 *     a        = s_trunk_to_a(s_trunk) + r_token_to_a(r at each centre atom)
 *     16 x       q = atomLayer(q, cond at each atom)    windowed, 4 heads
 *                a = tokenLayer(a, cond)                full, 16 heads
 *                and the two couple through every token's CENTRE ATOM
 *     out      = to_out(LN(q_to_hidden(q) + s_to_hidden(a) at each atom))
 *     D        = c_skip * x + c_out * out
 *
 * 🔴 THE TWO STREAMS COUPLE THROUGH THE CENTRE ATOM IN BOTH DIRECTIONS, and
 * each direction is its own gated mixer: `s_to_q` reads `[a at my token | q]`
 * and `q_to_s` reads `[q at my centre | a]`. Both are a THREE-way slice of one
 * conditioning projection - alpha, beta and a gate - so a port that read it as
 * two would scale by the wrong third of the matrix.
 *
 * 🔴 AND THE ATOM WINDOW IS THE VENDOR'S, NOT A SLIDING ONE: query block `win`
 * attends to atoms `win * 32 - 48 ... + 128`, clamped to the molecule, so a
 * query's keys are decided by its BLOCK and not by its own index. Every query in
 * a block shares one key list, which is what makes the pair bias
 * `(window, query, key, head)` rather than `(atom, atom, head)`.
 *
 * 🔴 AND THE KEY AND VALUE ARE PER-HEAD HALVES OF THE SECOND THIRD OF `qkv`:
 * "kv_raw reshaped (..., 2, heads, hd) - so the key is columns h*hd.. of the
 * first half PER HEAD and the value the second". Reading kv as [k | v] in
 * channel order gives a plausible tensor at the same shapes.
 *
 * 🔴 AND THE QUERY ALONE CARRIES A BIAS, and the gate is a sigmoid of its own
 * projection of the SAME adaLN-modulated input - not of the attention's output.
 *
 * 🔴 AND THE STATIC CONTEXT IS THE CALLER'S. `ctx.z` (the token pair), `ctx.q0`
 * (the atom features) and `ctx.p` (the windowed atom pair) are what a fold
 * computes ONCE and a sampler replays sixteen layers against at every step; this
 * module takes them as inputs, which is also what lets the f64 reference and
 * this port be handed the same ones.
 */
import { ONE_TARGET, PRELUDE, memoSource } from "../runtime/execution.js";
import { drain } from "../runtime/steps.js";
import { layerNorm, layerNormStatsSource } from "../kernels/layer-norm.js";
import { linear, linearPaddedStride4 } from "../kernels/linear.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/**
 * 🔴 EVERY WEIGHT HERE IS OUT-MAJOR, WHICH IS THE OPPOSITE OF EVERY OTHER MODEL
 * IN THIS TREE. `cpu/bioemu2/tape.js`'s `linear` reads `weight[o * inF + k]` -
 * torch's own `[out][in]`, untransposed at export - where AF3, AF2 and ESMFold2
 * all arrive as `[in][out]` because their exporters transpose. Nothing in a
 * tensor's SHAPE distinguishes the two, which is why `kernels/linear.js` has no
 * default; a port that assumed this tree's usual layout would compute a
 * plausible tensor of the right shape from the transpose of every matrix.
 */
export const SIGMA_DATA = 16;
export const WINDOW_QUERIES = 32;
export const WINDOW_KEYS = 128;
export const WINDOW_CONTEXT = (WINDOW_KEYS - WINDOW_QUERIES) / 2;

/**
 * The first line of a kernel that may write its rows at a padded stride: `texel` the LOGICAL texel of a
 * `[rows][c4 texels]` tensor laid at `s4` texels a row, the padding written zero (0 or `c4`: the plain layout). This is
 * how a producer hands a projection the x `linear` would otherwise re-lay itself (`linearPaddedStride4`,
 * `linear`'s `xStride4`): the same values, stored once - and into float16 where the pad pass would have narrowed
 * them, which is the same conversion on the same store.
 */
const stridedTexel = (c4, s4) => (s4 > 0 && s4 !== c4
  ? `int raw = linearTexel();
  int sRow = raw / ${s4};
  int sK = raw - sRow * ${s4};
  if (sK >= ${c4}) { result = vec4(0.0); return; }
  int texel = sRow * ${c4} + sK;`
  : "int texel = linearTexel();");

/** `out = x * factor`, a scalar the host computed from sigma. */
const scaleSourceUncached = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;
uniform float factor;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) * factor;
}`;

/**
 * `out = a + b`, or with `periodA` (elements, a multiple of four) `out[i] = a[i % periodA] + b[i]`: one sample's
 * `a` broadcast over a batch of samples stacked sample-major in `b` - the same add, element for element.
 */
const sumSourceUncached = (elements, periodA = elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(aTex, ${periodA === elements ? "texel" : `texel % ${periodA / 4}`}) + fetch4(bTex, texel);
}`;

/** `out = a * b`, elementwise over equal shapes (rows of `c4` texels written at `s4`: `stridedTexel`). */
const productSourceUncached = (elements, c4 = 0, s4 = 0) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  ${stridedTexel(c4, s4)}
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(aTex, texel) * fetch4(bTex, texel);
}`;

/** `out = silu(x)` (and `stridedTexel`'s layout). */
const siluSourceUncached = (elements, c4 = 0, s4 = 0) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;

void main() {
  ${stridedTexel(c4, s4)}
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 x = fetch4(xTex, texel);
  result = x / (vec4(1.0) + exp(-x));
}`;

/** `out = sigmoid(x)`. */
const sigmoidSourceUncached = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(1.0) / (vec4(1.0) + exp(-fetch4(xTex, texel)));
}`;

/** `out[row][c] = x[row][from + c]`, a channel range of every row. */
const sliceSourceUncached = (rows, channels, from, width) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int W4 = ${width / 4};
const int FROM4 = ${from / 4};
const int ELEMENTS = ${rows * width};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % W4;
  int row = texel / W4;
  result = fetch4(xTex, row * C4 + FROM4 + group);
}`;

/**
 * `out[row][c] = index[row] < 0 ? 0 : x[index[row]][from + c]` - a channel range of a GATHERED row, so one
 * conditioning projection made at the token rows serves every atom row (and every sample's rows) through a map,
 * in the draw the plain slice took. The copy is exact.
 */
const gatherSliceSourceUncached = (rows, channels, from, width) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int W4 = ${width / 4};
const int FROM4 = ${from / 4};
const int ELEMENTS = ${rows * width};
uniform sampler2D xTex;
uniform sampler2D indexTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % W4;
  int row = texel / W4;
  int from = int(fetch1(indexTex, row));
  if (from < 0) { result = vec4(0.0); return; }
  result = fetch4(xTex, from * C4 + FROM4 + group);
}`;

/**
 * `out[row] = [left[row] | right[row]]`, two widths that need not match.
 *
 * 🔴 AND IT HAS A RAGGED PATH, BECAUSE THE REAL SINGLE CONDITIONING IS 839 WIDE.
 * `[s_trunk | s_inputs]` is 384 + 455 - and 455 is the featuriser's 384 pooled
 * channels plus 71 one-hots, a number with no reason to be a multiple of
 * anything. A fragment owns four consecutive channels of ONE row only where
 * every width here is a multiple of four; at 455 the four cross a row boundary
 * and belong to two rows. The aligned path is one texel fetch a fragment and the
 * ragged one resolves each element on its own, the same split
 * `gatherRowsSource` below documents - and the aligned form alone compiled
 * `const int R4 = 113.75` and failed with "cannot convert from const float to
 * const highp int" on the first real fold, where every gate at synthetic widths
 * was green.
 */
const concatSourceUncached = (rows, leftWidth, rightWidth) => (
  leftWidth % 4 === 0 && rightWidth % 4 === 0
    ? `${PRELUDE}${ONE_TARGET}
const int L4 = ${leftWidth / 4};
const int R4 = ${rightWidth / 4};
const int W4 = ${(leftWidth + rightWidth) / 4};
const int ELEMENTS = ${rows * (leftWidth + rightWidth)};
uniform sampler2D leftTex;
uniform sampler2D rightTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % W4;
  int row = texel / W4;
  result = group < L4
    ? fetch4(leftTex, row * L4 + group)
    : fetch4(rightTex, row * R4 + group - L4);
}`
    : `${PRELUDE}${ONE_TARGET}
const int LEFT = ${leftWidth};
const int RIGHT = ${rightWidth};
const int WIDTH = ${leftWidth + rightWidth};
const int ELEMENTS = ${rows * (leftWidth + rightWidth)};
uniform sampler2D leftTex;
uniform sampler2D rightTex;

float at(int element) {
  int row = element / WIDTH;
  int c = element - row * WIDTH;
  return c < LEFT
    ? fetch1(leftTex, row * LEFT + c)
    : fetch1(rightTex, row * RIGHT + c - LEFT);
}

void main() {
  int texel = linearTexel();
  int base = texel * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`);

/**
 * `out[row][c] = index[row] < 0 ? 0 : source[index[row]][c]`.
 *
 * 🔴 AND IT HAS A RAGGED PATH, BECAUSE THE COORDINATES ARE THREE WIDE. A
 * fragment owns four consecutive channels of ONE row only where the width is a
 * multiple of four; at three they belong to two rows with two different source
 * indices. The aligned path is one texel fetch and the ragged one resolves each
 * element - the same split `kernels/layer-norm.js` documents, and the first
 * version of this kernel had only the aligned form, which compiled
 * `const int C4 = 0.75` and failed with "cannot convert from const float to
 * const highp int".
 */
const gatherRowsSourceUncached = (rows, channels) => (channels % 4 === 0
  ? `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D sourceTex;
uniform sampler2D indexTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  int from = int(fetch1(indexTex, row));
  if (from < 0) { result = vec4(0.0); return; }
  result = fetch4(sourceTex, from * C4 + group);
}`
  : `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
const int ELEMENTS = ${rows * channels};
uniform sampler2D sourceTex;
uniform sampler2D indexTex;

float at(int element) {
  int row = element / CHANNELS;
  int c = element - row * CHANNELS;
  int from = int(fetch1(indexTex, row));
  if (from < 0) return 0.0;
  return fetch1(sourceTex, from * CHANNELS + c);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`);

/**
 * `out[row][c] = x[row][c] * mask[row % period]`, with the same ragged split - `period` the rows of ONE sample, so a
 * batch stacked sample-major reads one sample's mask.
 */
const maskRowsSourceUncached = (rows, channels, period = rows) => (channels % 4 === 0
  ? `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) * fetch1(maskTex, ${period === rows ? "texel / C4" : `(texel / C4) % ${period}`});
}`
  : `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D maskTex;

float at(int element) {
  return fetch1(xTex, element) * fetch1(maskTex, ${period === rows ? "element / CHANNELS" : `(element / CHANNELS) % ${period}`});
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`);

/**
 * adaLN: `LN(x) * (1 + alpha) + beta`, the LayerNorm affine-free.
 *
 * 🔴 THE "ONE" IS ADDED TO alpha AND NOT TO THE NORMALISED VALUE, which is the
 * adaLN-Zero convention: a freshly initialised layer has alpha zero and so
 * passes its input through unscaled. Adding it on the other side is the same
 * arithmetic only when alpha is zero.
 */
/** Whether adaLN splits its statistics out at this width - see `splitAdaLnSource`. */
export const adaLnSplits = (execution, channels) =>
  channels % 4 === 0 && !execution.noNormSplit && channels >= 768;

/** adaLN in two passes: the row statistics once a row, then the modulation. */
export function adaLnSplit(execution, { x, alpha, beta, rows, channels, epsilon }, into) {
  const stats = execution.allocate(rows * 4, "bioemu2.adaln.stats");
  execution.dispatch({
    source: layerNormStatsSource(rows, channels, "two-pass"),
    output: stats,
    inputs: { xTex: x },
    floats: { epsilon },
    label: `bioemu2.adaln-stats:${rows}x${channels}`,
  });
  execution.dispatch({
    source: splitAdaLnSource(rows, channels),
    output: into,
    inputs: { xTex: x, statsTex: stats, alphaTex: alpha, betaTex: beta },
    label: `bioemu2.adaln-apply:${rows}x${channels}`,
  });
  execution.release(stats);
  return into;
}

/**
 * adaLN with the row statistics READ rather than swept, for the pass above.
 *
 * 🔴 THE SAME 1.03x MECHANISM AS `layerNorm`'s, AND THE SAME MEASURED BAND.
 * `adaLnSource` below has every fragment sweep its whole row twice - once for
 * the mean and once for the spread - and there are `C4` fragments a row, so the
 * row is read `2 * C4` times over: 448 at 896 channels. `bench-webgl-norm.js`
 * prices the identical shape of kernel at **4.84x** split at 768 channels, and
 * at **0.33x** at 128, because the saving scales with `C4` while the cost - one
 * extra draw and a statistics pass of one fragment a row - does not. So this
 * runs only where that bench measured no loss at any row count, 768 channels
 * and wider, and `--no-norm-split` is the arm for both.
 */
const splitAdaLnSourceUncached = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D statsTex;
uniform sampler2D alphaTex;
uniform sampler2D betaTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int row = texel / C4;
  vec2 stats = fetch4(statsTex, row).xy;
  vec4 normalised = (fetch4(xTex, texel) - vec4(stats.x)) * stats.y;
  result = normalised * (vec4(1.0) + fetch4(alphaTex, texel)) + fetch4(betaTex, texel);
}`;

const adaLnSourceUncached = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int CHANNELS = ${channels};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D alphaTex;
uniform sampler2D betaTex;
uniform float epsilon;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  float total = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 v = fetch4(xTex, row * C4 + i);
    total += v.x + v.y + v.z + v.w;
  }
  float mean = total / float(CHANNELS);
  float spread = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 centred = fetch4(xTex, row * C4 + i) - mean;
    spread += dot(centred, centred);
  }
  float inverse = inversesqrt(spread / float(CHANNELS) + epsilon);
  vec4 normalised = (fetch4(xTex, texel) - vec4(mean)) * inverse;
  result = normalised * (vec4(1.0) + fetch4(alphaTex, texel)) + fetch4(betaTex, texel);
}`;

/**
 * `out[row][head] = sum_c LN(pair[row])[c] * scale[c] * project[head][c]`.
 *
 * The norm has a SCALE and no offset, and the projection is `[head][channel]` -
 * out-major, which is why this is its own kernel rather than a LayerNorm and a
 * `linear`.
 */
const pairBiasSourceUncached = (rows, channels, heads, perHead = false) => {
  // 🔴 THE STATISTICS WERE COMPUTED ONCE A HEAD AND THE ROW IS SHARED BY FOUR,
  // so this kernel issued SEVEN TIMES the texelFetch it needs. `at` normalises
  // the pair row before projecting it, and the mean and the spread depend on
  // the ROW only - yet it was called once per output element, four times a
  // fragment, each time sweeping the channel axis twice over again. And with
  // `heads` a multiple of four all four of a fragment's elements are the same
  // row (`base` is a multiple of four and `head` is `base % HEADS`), so the row
  // is normalised ONCE here and projected onto the fragment's four heads.
  //
  // The projection loop also read every operand through `fetch1`, which is a
  // whole texelFetch and a component select: 256 channels times three operands
  // times four heads. Read as `fetch4` along the channel axis that is 6 fetches
  // a four-channel step instead of 12 - and the pair row and the scale are
  // fetched once for all four heads rather than once each.
  //
  //   per fragment, at 256 channels and 16 heads:
  //     4 * (64 + 64 + 768) = 3584  ->  64 + 64 + 384 = 512
  //
  // 🔴 AND IT IS BIT-IDENTICAL ON PURPOSE, WHICH TOOK TWO TRIES. The products
  // accumulate in ASCENDING CHANNEL ORDER, one scalar at a time inside each
  // four, exactly as the loop over `c` did - not as a `dot`, which would
  // associate four at a time. The normalisation is scalar per component for the
  // same reason: as a vec4 chain the compiler contracts it differently and a
  // 4-step fold's CA-CA moved in the fourth digit. The variance stays the
  // two-pass form the reference uses and is NOT replaced by the fast one.
  //
  // `--no-pair-bias-fast` is the control arm.
  const fast = !perHead && heads % 4 === 0 && channels % 4 === 0;
  const head = (h) => `    vec4 p${h} = fetch4(projectTex, (head0 + ${h}) * C4 + i);
    s${h} += n0 * p${h}.x; s${h} += n1 * p${h}.y;
    s${h} += n2 * p${h}.z; s${h} += n3 * p${h}.w;`;
  const body = fast
    ? `  int row = base / HEADS;
  int head0 = base - row * HEADS;
  float total = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 v = fetch4(pairTex, row * C4 + i);
    total += v.x + v.y + v.z + v.w;
  }
  float mean = total / float(CHANNELS);
  float spread = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 centred = fetch4(pairTex, row * C4 + i) - mean;
    spread += dot(centred, centred);
  }
  float inverse = inversesqrt(spread / float(CHANNELS) + epsilon);
  float s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  for (int i = 0; i < C4; ++i) {
    // 🔴 SCALAR, NOT A vec4 MULTIPLY. Written as one vec4 chain - subtract the
    // mean, scale by the inverse, multiply by the scale texel - this kernel
    // stopped being bit-identical, because the compiler contracts a vec4 chain
    // differently from the scalar one the loop over c wrote, and a 4-step
    // fold's CA-CA moved from 1327.714 to 1324.144. Per component it is the
    // same expression as before. (No backticks in here: it is inside a JS
    // template literal and one would end it. It has happened six times.)
    vec4 v = fetch4(pairTex, row * C4 + i);
    vec4 sc = fetch4(scaleTex, i);
    float n0 = (v.x - mean) * inverse * sc.x;
    float n1 = (v.y - mean) * inverse * sc.y;
    float n2 = (v.z - mean) * inverse * sc.z;
    float n3 = (v.w - mean) * inverse * sc.w;
${[0, 1, 2, 3].map(head).join("\n")}
  }
  result = vec4(s0,
    base + 1 < ELEMENTS ? s1 : 0.0,
    base + 2 < ELEMENTS ? s2 : 0.0,
    base + 3 < ELEMENTS ? s3 : 0.0);`
    : `  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);`;
  return `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
const int C4 = ${channels / 4};
const int HEADS = ${heads};
const int ELEMENTS = ${rows * heads};
uniform sampler2D pairTex;
uniform sampler2D scaleTex;
uniform sampler2D projectTex;
uniform float epsilon;
${fast ? "" : `
float at(int element) {
  int row = element / HEADS;
  int head = element - row * HEADS;
  float total = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 v = fetch4(pairTex, row * C4 + i);
    total += v.x + v.y + v.z + v.w;
  }
  float mean = total / float(CHANNELS);
  float spread = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 centred = fetch4(pairTex, row * C4 + i) - mean;
    spread += dot(centred, centred);
  }
  float inverse = inversesqrt(spread / float(CHANNELS) + epsilon);
  float sum = 0.0;
  for (int c = 0; c < CHANNELS; ++c) {
    float normalised = (fetch1(pairTex, row * CHANNELS + c) - mean) * inverse
      * fetch1(scaleTex, c);
    sum += normalised * fetch1(projectTex, head * CHANNELS + c);
  }
  return sum;
}
`}
void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
${body}
}`;
};

/**
 * Full attention over tokens, biased per (i, j, head) and masked by the pad.
 *
 * q is `(rows, heads, headDim)` and k and v are the second and third thirds of
 * one `3 * d` projection - in CHANNEL order here, because the token stream's
 * slices are `[q | k | v]`.
 */
/**
 * The head's dot product, read four channels at a time where the layout allows.
 *
 * 🔴 BOTH ATTENTIONS WALKED THEIR HEAD DIMENSION WITH `fetch1`, WHICH IS A WHOLE
 * `texelFetch` AND A COMPONENT SELECT - and they walked it TWICE, once for the
 * softmax's maximum and once for the weight. At 768 channels in 16 heads that is
 * 48 scalar fetches a key where twelve `fetch4` carry the same numbers, so the
 * dominant term of these kernels was four times the fetches it needed.
 *
 * `qBase` and `kBase` are both a row offset plus `head * HEAD_DIM`, and every
 * width in play - 768 and 128 channels, 48 and 32 head dimensions - is a
 * multiple of four, so the division is exact. The products still accumulate in
 * ascending `d` order, one scalar at a time, rather than as a `dot`.
 *
 * `--no-attend-dot4` is the control arm.
 */
const headDot = (keyTex, wide) => wide
  ? `    for (int d4 = 0; d4 < HEAD_DIM / 4; ++d4) {
      vec4 q4 = fetch4(qTex, qBase / 4 + d4);
      vec4 k4 = fetch4(${keyTex}, kBase / 4 + d4);
      dot += q4.x * k4.x; dot += q4.y * k4.y;
      dot += q4.z * k4.z; dot += q4.w * k4.w;
    }`
  : `    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += fetch1(qTex, qBase + d) * fetch1(${keyTex}, kBase + d);
    }`;

const fullAttentionSourceUncached = (rows, channels, heads, wide = true, samples = 1) => `${PRELUDE}${ONE_TARGET}
const int ROWS = ${rows};
const int CHANNELS = ${channels};
const int HEADS = ${heads};
const int HEAD_DIM = ${channels / heads};
const int ELEMENTS = ${samples * rows * channels};
const float SCALE = ${(1 / Math.sqrt(channels / heads)).toFixed(10)};
uniform sampler2D qTex;
uniform sampler2D kTex;
uniform sampler2D vTex;
uniform sampler2D biasTex;
uniform sampler2D padTex;

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  // A row of a batch attends within its own sample: AT is the row in the stack (q, k and v are sample-major),
  // I the token inside its sample, which is what the shared bias and pad are indexed by.
  int at = base / CHANNELS;
  int c = base - at * CHANNELS;
  int smp = at / ROWS;
  int i = at - smp * ROWS;
  int head = c / HEAD_DIM;
  int qBase = at * CHANNELS + head * HEAD_DIM;

  float largest = -1e30;
  for (int j = 0; j < ROWS; ++j) {
    if (fetch1(padTex, j) == 0.0) continue;
    int kBase = (smp * ROWS + j) * CHANNELS + head * HEAD_DIM;
    float dot = 0.0;
${headDot("kTex", wide)}
    largest = max(largest, dot * SCALE + fetch1(biasTex, (i * ROWS + j) * HEADS + head));
  }
  float sum = 0.0;
  vec4 gathered = vec4(0.0);
  for (int j = 0; j < ROWS; ++j) {
    if (fetch1(padTex, j) == 0.0) continue;
    int kBase = (smp * ROWS + j) * CHANNELS + head * HEAD_DIM;
    float dot = 0.0;
${headDot("kTex", wide)}
    float weight = exp(dot * SCALE
      + fetch1(biasTex, (i * ROWS + j) * HEADS + head) - largest);
    sum += weight;
    gathered += weight * fetch4(vTex, (kBase + c % HEAD_DIM) / 4);
  }
  result = gathered / sum;
}`;

/**
 * The vendor's windowed atom attention.
 *
 * 🔴 THE KEY AND VALUE ARE PER-HEAD HALVES OF ONE `2 * d` SLICE, so the key of
 * head h is at `d + h * headDim` and its value at `2 * d + h * headDim` - see
 * the note at the top.
 */
const windowAttentionSourceUncached = (atoms, channels, heads, wide = true, samples = 1) => `${PRELUDE}${ONE_TARGET}
const int ATOMS = ${atoms};
const int CHANNELS = ${channels};
const int HEADS = ${heads};
const int HEAD_DIM = ${channels / heads};
const int W_Q = ${WINDOW_QUERIES};
const int W_K = ${WINDOW_KEYS};
const int CTX = ${WINDOW_CONTEXT};
const int ELEMENTS = ${samples * atoms * channels};
const float SCALE = ${(1 / Math.sqrt(channels / heads)).toFixed(10)};
uniform sampler2D qTex;
uniform sampler2D kvTex;
uniform sampler2D biasTex;
uniform sampler2D padTex;

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  // a row of a batch reads keys of its own sample only: AT the row in the stack, I the atom inside its sample
  int at = base / CHANNELS;
  int c = base - at * CHANNELS;
  int smp = at / ATOMS;
  int i = at - smp * ATOMS;
  int head = c / HEAD_DIM;
  int qBase = at * CHANNELS + head * HEAD_DIM;
  int window = i / W_Q;
  int inWindow = i - window * W_Q;

  float largest = -1e30;
  for (int slot = 0; slot < W_K; ++slot) {
    int j = window * W_Q - CTX + slot;
    if (j < 0 || j >= ATOMS) continue;
    if (fetch1(padTex, j) == 0.0) continue;
    int kBase = (smp * ATOMS + j) * 2 * CHANNELS + head * HEAD_DIM;
    float dot = 0.0;
${headDot("kvTex", wide)}
    float bias = fetch1(biasTex,
      ((window * W_Q + inWindow) * W_K + slot) * HEADS + head);
    largest = max(largest, dot * SCALE + bias);
  }
  float sum = 0.0;
  vec4 gathered = vec4(0.0);
  for (int slot = 0; slot < W_K; ++slot) {
    int j = window * W_Q - CTX + slot;
    if (j < 0 || j >= ATOMS) continue;
    if (fetch1(padTex, j) == 0.0) continue;
    int kBase = (smp * ATOMS + j) * 2 * CHANNELS + head * HEAD_DIM;
    float dot = 0.0;
${headDot("kvTex", wide)}
    float bias = fetch1(biasTex,
      ((window * W_Q + inWindow) * W_K + slot) * HEADS + head);
    float weight = exp(dot * SCALE + bias - largest);
    sum += weight;
    // ...the VALUE half: a whole CHANNELS further along this atom's row.
    gathered += weight * fetch4(kvTex, (kBase + CHANNELS + c % HEAD_DIM) / 4);
  }
  result = gathered / sum;
}`;

/**
 * adaLN with its alpha and beta read straight out of the conditioning PROJECTION they are slices of - row
 * `index[row]` (or the row itself), channels `alphaAt` and `betaAt` of `condWidth` - where two slice draws made
 * them first. The same expression on the same values: `split` is `splitAdaLnSource`'s pass (the row statistics
 * read from `statsTex`), otherwise `adaLnSource`'s sweep.
 */
const condAdaLnSourceUncached = (rows, channels, condWidth, alphaAt, betaAt, mapped, split, s4 = 0) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int CHANNELS = ${channels};
const int CW4 = ${condWidth / 4};
const int A4 = ${alphaAt / 4};
const int B4 = ${betaAt / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D condTex;
${mapped ? "uniform sampler2D indexTex;" : ""}
${split ? "uniform sampler2D statsTex;" : "uniform float epsilon;"}

void main() {
  ${stridedTexel(channels / 4, s4)}
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  int from = ${mapped ? "int(fetch1(indexTex, row))" : "row"};
  vec4 alpha = from < 0 ? vec4(0.0) : fetch4(condTex, from * CW4 + A4 + group);
  vec4 beta = from < 0 ? vec4(0.0) : fetch4(condTex, from * CW4 + B4 + group);
${split ? `  vec2 stats = fetch4(statsTex, row).xy;
  vec4 normalised = (fetch4(xTex, texel) - vec4(stats.x)) * stats.y;`
    : `  float total = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 v = fetch4(xTex, row * C4 + i);
    total += v.x + v.y + v.z + v.w;
  }
  float mean = total / float(CHANNELS);
  float spread = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 centred = fetch4(xTex, row * C4 + i) - mean;
    spread += dot(centred, centred);
  }
  float inverse = inversesqrt(spread / float(CHANNELS) + epsilon);
  vec4 normalised = (fetch4(xTex, texel) - vec4(mean)) * inverse;`}
  result = normalised * (vec4(1.0) + alpha) + beta;
}`;

/**
 * `out = x + gate * branch`, the gate read out of the conditioning projection as `condAdaLnSource` reads alpha -
 * one draw where a gate slice, the product and the residual sum were three. (Not bit-identical: the add may be
 * contracted into the multiply where it was a stored product.)
 */
const condGatedResidualSourceUncached = (rows, channels, condWidth, gateAt, mapped) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int CW4 = ${condWidth / 4};
const int G4 = ${gateAt / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D branchTex;
uniform sampler2D condTex;
${mapped ? "uniform sampler2D indexTex;" : ""}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  int from = ${mapped ? "int(fetch1(indexTex, row))" : "row"};
  vec4 gate = from < 0 ? vec4(0.0) : fetch4(condTex, from * CW4 + G4 + group);
  result = fetch4(xTex, texel) + gate * fetch4(branchTex, texel);
}`;

/**
 * The padded x layout (`linearPaddedStride4`) that EVERY one of `consumers` - the projections that read a tensor,
 * `{outChannels, weight}` - would lay `rows x inChannels` at, or 0 (the plain layout) where any declines or they
 * disagree. `--no-prepadded-x` is the arm that has every projection pad its own x again.
 */
function sharedXStride4(execution, rows, inChannels, consumers) {
  if (execution.noPrepaddedX || consumers.length === 0) return 0;
  const strides = consumers.map(({ outChannels, weight }) => linearPaddedStride4(execution, {
    rows, inChannels, outChannels, weight }));
  return strides.every((stride) => stride === strides[0]) ? strides[0] : 0;
}

/** A `rows x channels` tensor laid at `stride4` (0: plain), float16 where the pad pass would have narrowed it. */
function allocateX(execution, rows, channels, stride4, label) {
  const tensor = stride4 === 0 ? execution.allocate(rows * channels, label)
    : execution.halfActivations ? execution.allocateHalf(rows * stride4 * 4, label)
      : execution.allocate(rows * stride4 * 4, label);
  tensor.xStride4 = stride4;
  return tensor;
}

/** The two projections an attention reads its input through (`projections`), as `sharedXStride4` consumers. */
const attentionReaders = (w, channels) => [
  { outChannels: 3 * channels, weight: w.projQkv }, { outChannels: channels, weight: w.projG }];

/** `linear`, told the layout its x was written at (`allocateX`). */
const project = (execution, call) => linear(execution, { ...call, xStride4: call.x.xStride4 ?? 0 });

/**
 * adaLN of `x` with alpha and beta channels `alphaAt`/`betaAt` of the `condWidth`-wide projection `cond`, read
 * through `map` (null: the same rows) - see `condAdaLnSource`.
 */
export function condAdaLn(execution, {
  x, cond, map, condWidth, alphaAt, betaAt, rows, channels, epsilon, label, consumers = [],
}) {
  const s4 = sharedXStride4(execution, rows, channels, consumers);
  const out = allocateX(execution, rows, channels, s4, "bioemu2.adaln");
  const mapped = map !== null;
  const split = adaLnSplits(execution, channels);
  let stats = null;
  if (split) {
    stats = execution.allocate(rows * 4, "bioemu2.adaln.stats");
    execution.dispatch({
      source: layerNormStatsSource(rows, channels, "two-pass"),
      output: stats,
      inputs: { xTex: x },
      floats: { epsilon },
      label: `bioemu2.adaln-stats:${rows}x${channels}`,
    });
  }
  execution.dispatch({
    source: condAdaLnSource(rows, channels, condWidth, alphaAt, betaAt, mapped, split, s4),
    output: out,
    inputs: { xTex: x, condTex: cond, ...(mapped ? { indexTex: map } : {}), ...(split ? { statsTex: stats } : {}) },
    ...(split ? {} : { floats: { epsilon } }),
    label: `${label}:${rows}x${channels}${split ? ":split" : ""}`,
  });
  if (stats !== null) execution.release(stats);
  return out;
}

/** `x + gate * branch`, the gate channels `gateAt` of `cond` through `map` - see `condGatedResidualSource`. */
export function condGatedResidual(execution, { x, branch, cond, map, condWidth, gateAt, rows, channels, label }) {
  const out = execution.allocate(rows * channels, "bioemu2.layer");
  const mapped = map !== null;
  execution.dispatch({
    source: condGatedResidualSource(rows, channels, condWidth, gateAt, mapped),
    output: out,
    inputs: { xTex: x, branchTex: branch, condTex: cond, ...(mapped ? { indexTex: map } : {}) },
    label: `${label}:${rows}x${channels}`,
  });
  return out;
}

/**
 * Either attention straight off its projections: q, k and v read out of the one `3 * d` projection (`[q | k | v]` in
 * channel order, the atom stream's k and v the same per-head columns `windowAttentionSource`'s kv slice holds), the
 * query bias added as q is read, and the output multiplied by sigmoid(gate) as it is written - at the stride the
 * output projection reads its x at (`stridedTexel`). Where the slices, the bias add, the sigmoid and the gate product
 * were six draws (five for the atoms' one kv slice) and the attention a seventh, this is one, on the same arithmetic
 * in the same order: q is the same sum and the gate the same sigmoid times the same quotient. `--no-attend-fuse` is
 * the arm. `window` true is the atoms' windowed form, false the tokens' full one.
 */
const fusedAttentionSourceUncached = (window, rows, channels, heads, samples, s4) => `${PRELUDE}${ONE_TARGET}
const int ROWS = ${rows};
const int CHANNELS = ${channels};
const int C3 = ${3 * channels};
const int HEADS = ${heads};
const int HEAD_DIM = ${channels / heads};
const int ELEMENTS = ${samples * rows * channels};
const float SCALE = ${(1 / Math.sqrt(channels / heads)).toFixed(10)};
${window ? `const int W_Q = ${WINDOW_QUERIES};
const int W_K = ${WINDOW_KEYS};
const int CTX = ${WINDOW_CONTEXT};` : ""}
uniform sampler2D qkvTex;
uniform sampler2D qBiasTex;
uniform sampler2D gateTex;
uniform sampler2D biasTex;
uniform sampler2D padTex;

void main() {
  ${stridedTexel(channels / 4, s4)}
  int base = texel * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  // AT the row in the stack, I the row inside its own sample: a row attends within its sample only
  int at = base / CHANNELS;
  int c = base - at * CHANNELS;
  int smp = at / ROWS;
  int i = at - smp * ROWS;
  int head = c / HEAD_DIM;
  vec4 qv[HEAD_DIM / 4];
  for (int d4 = 0; d4 < HEAD_DIM / 4; ++d4) {
    qv[d4] = fetch4(qkvTex, (at * C3 + head * HEAD_DIM) / 4 + d4)
      + fetch4(qBiasTex, (head * HEAD_DIM) / 4 + d4);
  }
${window ? `  int window = i / W_Q;
  int inWindow = i - window * W_Q;` : ""}
  float largest = -1e30;
  for (int slot = 0; slot < ${window ? "W_K" : "ROWS"}; ++slot) {
    int j = ${window ? "window * W_Q - CTX + slot" : "slot"};
${window ? "    if (j < 0 || j >= ROWS) continue;\n" : ""}    if (fetch1(padTex, j) == 0.0) continue;
    int kBase = (smp * ROWS + j) * C3 + CHANNELS + head * HEAD_DIM;
    float dot = 0.0;
    for (int d4 = 0; d4 < HEAD_DIM / 4; ++d4) {
      vec4 q4 = qv[d4];
      vec4 k4 = fetch4(qkvTex, kBase / 4 + d4);
      dot += q4.x * k4.x; dot += q4.y * k4.y;
      dot += q4.z * k4.z; dot += q4.w * k4.w;
    }
    float bias = fetch1(biasTex, ${window ? "((window * W_Q + inWindow) * W_K + slot) * HEADS + head" : "(i * ROWS + j) * HEADS + head"});
    largest = max(largest, dot * SCALE + bias);
  }
  float sum = 0.0;
  vec4 gathered = vec4(0.0);
  for (int slot = 0; slot < ${window ? "W_K" : "ROWS"}; ++slot) {
    int j = ${window ? "window * W_Q - CTX + slot" : "slot"};
${window ? "    if (j < 0 || j >= ROWS) continue;\n" : ""}    if (fetch1(padTex, j) == 0.0) continue;
    int kBase = (smp * ROWS + j) * C3 + CHANNELS + head * HEAD_DIM;
    float dot = 0.0;
    for (int d4 = 0; d4 < HEAD_DIM / 4; ++d4) {
      vec4 q4 = qv[d4];
      vec4 k4 = fetch4(qkvTex, kBase / 4 + d4);
      dot += q4.x * k4.x; dot += q4.y * k4.y;
      dot += q4.z * k4.z; dot += q4.w * k4.w;
    }
    float bias = fetch1(biasTex, ${window ? "((window * W_Q + inWindow) * W_K + slot) * HEADS + head" : "(i * ROWS + j) * HEADS + head"});
    float weight = exp(dot * SCALE + bias - largest);
    sum += weight;
    gathered += weight * fetch4(qkvTex, (kBase + CHANNELS + c % HEAD_DIM) / 4);
  }
  vec4 gate = vec4(1.0) / (vec4(1.0) + exp(-fetch4(gateTex, at * (CHANNELS / 4) + c / 4)));
  result = gate * (gathered / sum);
}`;

/**
 * `fusedAttentionSource`'s draw, then the output projection: what `fullAttention` and `windowAttention` do after
 * their projections, where the fused form applies (the head walked four channels at a time).
 */
function fusedAttention(execution, { window, x, rows, samples, channels, heads, weights: w, bias, pad }) {
  const qkv = project(execution, {
    x, weight: w.projQkv, rows: samples * rows, inChannels: channels, outChannels: 3 * channels,
    weightLayout: "out-major",
  });
  const rawGate = project(execution, {
    x, weight: w.projG, rows: samples * rows, inChannels: channels, outChannels: channels,
    weightLayout: "out-major",
  });
  const total = samples * rows;
  const s4 = sharedXStride4(execution, total, channels, [{ outChannels: channels, weight: w.projO }]);
  const gated = allocateX(execution, total, channels, s4, `bioemu2.${window ? "atom" : "token"}-gated`);
  execution.dispatch({
    source: fusedAttentionSource(window, rows, channels, heads, samples, s4),
    output: gated,
    inputs: { qkvTex: qkv, qBiasTex: w.projQBias, gateTex: rawGate, biasTex: bias, padTex: pad },
    label: `bioemu2.${window ? "atom" : "token"}-attend-fused:${total}x${channels}:${heads}`,
  });
  execution.release(qkv);
  execution.release(rawGate);
  const out = project(execution, {
    x: gated, weight: w.projO, rows: total, inChannels: channels, outChannels: channels, weightLayout: "out-major",
  });
  execution.release(gated);
  return out;
}

/** Whether `fusedAttention` applies: not under its arm, and only with the head read four channels at a time. */
const attendFuses = (execution, channels, heads) =>
  !execution.noAttendFuse && !execution.noAttendDot4 && channels % (heads * 4) === 0;

/** A weight that a layer stack holds one slice of, as the reference views it. */
export function layerSlice(weights, name, layer, layers) {
  const tensor = weights[name];
  if (tensor === undefined) throw new Error(`bioemu2: no weight ${name}`);
  void layers;
  return tensor[layer];
}

/**
 * `out = x + gate * op(adaLN(x, alpha, beta))`, twice: the attention and then
 * the transition, from ONE six-way conditioning projection.
 *
 * 🔴 THE SIX SLICES ARE alpha_a, beta_a, gate_a, alpha_f, beta_f, gate_f, IN
 * THAT ORDER, and each is `channels` wide. A port that read them as three pairs
 * in the other grouping scales the attention by the transition's gate.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, conditioning: Tensor, rows: number, channels: number,
 *   heads: number, weights: any, attend: (input: Tensor) => Tensor,
 *   epsilon?: number}} call
 *   `attend` is the stream's own attention - windowed for the atoms and full for
 *   the tokens - injected so this function is the SHAPE of a layer and not a
 *   switch on which stream it is.
 * @returns {Tensor} `rows * channels`
 */
export function conditionedLayer(execution, call) {
  const {
    x, conditioning, rows, channels, weights: w, attend, epsilon = 1e-5,
  } = call;
  // 🔴 THE CONDITIONING MAY ARRIVE AT FEWER ROWS THAN x, with `condMap` taking each of x's rows to one of them (-1
  // for none). The denoiser's atom stream conditions on its TOKEN's row and every sample of a batch on the same
  // rows, so the projection below - LN, silu and a 6 * channels GEMM, all row by row - is made once a token rather
  // than once an atom of every sample, and the slices gather what each row reads.
  const condRows = call.condRows ?? rows;
  const condMap = call.condMap ?? null;
  // silu(LN(c)) @ cond_2, which is 6 * channels wide.
  const normalised = layerNorm(execution, {
    x: conditioning, weight: w.cond0Scale, bias: w.cond0Offset,
    rows: condRows, channels: conditioning.elements / condRows, epsilon,
  });
  const condChannels = conditioning.elements / condRows;
  const siluStride4 = sharedXStride4(execution, condRows, condChannels,
    [{ outChannels: 6 * channels, weight: w.cond2Weight }]);
  const activated = allocateX(execution, condRows, condChannels, siluStride4, "bioemu2.cond-silu");
  execution.dispatch({
    source: siluSource(normalised.elements, condChannels / 4, siluStride4),
    output: activated,
    inputs: { xTex: normalised },
    label: `bioemu2.cond-silu:${normalised.elements}`,
  });
  execution.release(normalised);
  const cond = project(execution, {
    x: activated, weight: w.cond2Weight, bias: w.cond2Bias, rows: condRows,
    inChannels: conditioning.elements / condRows, outChannels: 6 * channels,
    weightLayout: "out-major",
  });
  execution.release(activated);
  const piece = (index) => {
    const out = execution.allocate(rows * channels, `bioemu2.cond.${index}`);
    execution.dispatch(condMap === null ? {
      source: sliceSource(rows, 6 * channels, index * channels, channels),
      output: out,
      inputs: { xTex: cond },
      label: `bioemu2.cond-slice:${rows}x${channels}:${index}`,
    } : {
      source: gatherSliceSource(rows, 6 * channels, index * channels, channels),
      output: out,
      inputs: { xTex: cond, indexTex: condMap },
      label: `bioemu2.cond-gather-slice:${rows}x${channels}:${index}`,
    });
    return out;
  };

  const modulated = (alphaIndex, betaIndex, input) => {
    const alpha = piece(alphaIndex);
    const beta = piece(betaIndex);
    const out = execution.allocate(rows * channels, "bioemu2.adaln");
    if (adaLnSplits(execution, channels)) {
      adaLnSplit(execution, { x: input, alpha, beta, rows, channels, epsilon }, out);
    } else {
      execution.dispatch({
        source: adaLnSource(rows, channels),
        output: out,
        inputs: { xTex: input, alphaTex: alpha, betaTex: beta },
        floats: { epsilon },
        label: `bioemu2.adaln:${rows}x${channels}`,
      });
    }
    execution.release(alpha);
    execution.release(beta);
    return out;
  };
  const gated = (gateIndex, input, branch) => {
    const gate = piece(gateIndex);
    const scaled = execution.allocate(rows * channels, "bioemu2.gated");
    execution.dispatch({
      source: productSource(rows * channels),
      output: scaled,
      inputs: { aTex: gate, bTex: branch },
      label: `bioemu2.gate:${rows}x${channels}`,
    });
    execution.release(gate);
    const out = execution.allocate(rows * channels, "bioemu2.layer");
    execution.dispatch({
      source: sumSource(rows * channels),
      output: out,
      inputs: { aTex: input, bTex: scaled },
      label: `bioemu2.residual:${rows}x${channels}`,
    });
    execution.release(scaled);
    return out;
  };

  // 🔴 alpha, beta and the gate are READ OUT OF THE PROJECTION by the adaLN and the gated residual, where six slice
  // draws, two products and two sums made and combined them: 10 draws a layer to 2 (`--no-cond-fuse`, the arm).
  if (!execution.noCondFuse) {
    const width = 6 * channels;
    // (each written as the x its projections read: the attention's q/k/v and gate, the transition's fc1 and fc2)
    const hidden = w.transition.fc1.elements / channels;
    const readers = [call.attendConsumers ?? [],
      [{ outChannels: hidden, weight: w.transition.fc1 }, { outChannels: hidden, weight: w.transition.fc2 }]];
    const modulatedFused = (alphaIndex, input) => condAdaLn(execution, {
      x: input, cond, map: condMap, condWidth: width, alphaAt: alphaIndex * channels,
      betaAt: (alphaIndex + 1) * channels, rows, channels, epsilon, label: "bioemu2.cond-adaln",
      consumers: readers[alphaIndex === 0 ? 0 : 1],
    });
    const gatedFused = (gateIndex, input, branch) => condGatedResidual(execution, {
      x: input, branch, cond, map: condMap, condWidth: width, gateAt: gateIndex * channels,
      rows, channels, label: "bioemu2.cond-gated",
    });
    const forAttentionFused = modulatedFused(0, x);
    const attendedFused = attend(forAttentionFused);
    execution.release(forAttentionFused);
    const afterAttentionFused = gatedFused(2, x, attendedFused);
    execution.release(attendedFused);
    const forTransitionFused = modulatedFused(3, afterAttentionFused);
    const transitionedFused = transitionNoNorm(execution, {
      x: forTransitionFused, rows, channels, weights: w.transition,
    });
    execution.release(forTransitionFused);
    const outFused = gatedFused(5, afterAttentionFused, transitionedFused);
    execution.release(afterAttentionFused);
    execution.release(transitionedFused);
    execution.release(cond);
    return outFused;
  }
  const forAttention = modulated(0, 1, x);
  const attended = attend(forAttention);
  execution.release(forAttention);
  const afterAttention = gated(2, x, attended);
  execution.release(attended);

  const forTransition = modulated(3, 4, afterAttention);
  const transitioned = transitionNoNorm(execution, {
    x: forTransition, rows, channels, weights: w.transition,
  });
  execution.release(forTransition);
  const out = gated(5, afterAttention, transitioned);
  execution.release(afterAttention);
  execution.release(transitioned);
  execution.release(cond);
  return out;
}

/**
 * `fc3(silu(fc1(x)) * fc2(x))` - the SwiGLU transition with its gate and value
 * as SEPARATE projections, and no norm of its own.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, rows: number, channels: number, weights: any,
 *   outChannels?: number}} call `outChannels` where the transition is NOT square
 *   - which the coupling's two mixers are not.
 * @returns {Tensor} `rows * (outChannels ?? channels)`
 */
export function transitionNoNorm(execution, { x, rows, channels, weights: w, outChannels }) {
  const hidden = w.fc1.elements / channels;
  // 🔴 THE COUPLING'S MIXERS ARE NOT SQUARE, which is the whole point of them:
  // `s_to_q_mixer` reads `[a | q]` - the two streams' widths added - and writes
  // the ATOM width, and `q_to_s_mixer` writes the TOKEN width. `fc3` is what
  // says so, and assuming square makes the product that follows a mismatch the
  // reference catches with "offset is out of bounds" and this port would not.
  const out = outChannels ?? channels;
  const a = project(execution, {
    x, weight: w.fc1, rows, inChannels: channels, outChannels: hidden,
    weightLayout: "out-major",
  });
  const b = project(execution, {
    x, weight: w.fc2, rows, inChannels: channels, outChannels: hidden,
    weightLayout: "out-major",
  });
  const activated = execution.allocate(rows * hidden, "bioemu2.transition-silu");
  execution.dispatch({
    source: siluSource(rows * hidden),
    output: activated,
    inputs: { xTex: a },
    label: `bioemu2.transition-silu:${rows}x${hidden}`,
  });
  execution.release(a);
  const gatedStride4 = sharedXStride4(execution, rows, hidden, [{ outChannels: out, weight: w.fc3 }]);
  const gated = allocateX(execution, rows, hidden, gatedStride4, "bioemu2.transition-gated");
  execution.dispatch({
    source: productSource(rows * hidden, hidden / 4, gatedStride4),
    output: gated,
    inputs: { aTex: activated, bTex: b },
    label: `bioemu2.transition-gate:${rows}x${hidden}`,
  });
  execution.release(activated);
  execution.release(b);
  const projected = project(execution, {
    x: gated, weight: w.fc3, rows, inChannels: hidden, outChannels: out,
    weightLayout: "out-major",
  });
  execution.release(gated);
  return projected;
}

/** The attention's q, its bias-carrying query, and its gate. */
function projections(execution, { x, rows, channels, weights: w }) {
  const qkv = project(execution, {
    x, weight: w.projQkv, rows, inChannels: channels, outChannels: 3 * channels,
    weightLayout: "out-major",
  });
  const query = execution.allocate(rows * channels, "bioemu2.query");
  execution.dispatch({
    source: sliceSource(rows, 3 * channels, 0, channels),
    output: query,
    inputs: { xTex: qkv },
    label: `bioemu2.slice-q:${rows}x${channels}`,
  });
  const biased = execution.allocate(rows * channels, "bioemu2.query-biased");
  execution.dispatch({
    source: addRowBiasSource(rows, channels),
    output: biased,
    inputs: { xTex: query, biasTex: w.projQBias },
    label: `bioemu2.query-bias:${rows}x${channels}`,
  });
  execution.release(query);
  const rawGate = project(execution, {
    x, weight: w.projG, rows, inChannels: channels, outChannels: channels,
    weightLayout: "out-major",
  });
  const gate = execution.allocate(rows * channels, "bioemu2.attn-gate");
  execution.dispatch({
    source: sigmoidSource(rows * channels),
    output: gate,
    inputs: { xTex: rawGate },
    label: `bioemu2.attn-gate:${rows}x${channels}`,
  });
  execution.release(rawGate);
  return { qkv, query: biased, gate };
}

/** `out[row][c] = x[row][c] + bias[c]`. */
const addRowBiasSourceUncached = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D biasTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) + fetch4(biasTex, texel % C4);
}`;

/**
 * The per-head bias a stream's pair representation gives.
 *
 * @returns {Tensor} `rows * heads`
 */
export function pairBias(execution, { pair, rows, channels, heads, weights: w, epsilon = 1e-5 }) {
  const out = execution.allocate(rows * heads, "bioemu2.pair-bias");
  execution.dispatch({
    source: pairBiasSource(rows, channels, heads, execution.noPairBiasFast),
    output: out,
    inputs: { pairTex: pair, scaleTex: w.normZ, projectTex: w.projZ },
    floats: { epsilon },
    label: `bioemu2.pair-bias:${rows}x${channels}:${heads}`,
  });
  return out;
}

/** The token stream's full attention, with its gate and output projection. */
export function fullAttention(execution, call) {
  const { x, pair, pad, tokens, channels, heads, weights: w, epsilon = 1e-5 } = call;
  // `samples` stacked sample-major in x, attending within their own sample over one shared bias; `bias` the
  // precomputed `pairBias` of `pair` (the denoiser's is a fold's constant, made once - see `bioemu2DenoiserPrepare`)
  const samples = call.samples ?? 1;
  const rows = samples * tokens;
  const given = call.bias;
  if (given !== undefined && attendFuses(execution, channels, heads)) {
    return fusedAttention(execution, {
      window: false, x, rows: tokens, samples, channels, heads, weights: w, bias: given, pad });
  }
  const { qkv, query, gate } = projections(execution, {
    x, rows, channels, weights: w,
  });
  const slice = (from, label) => {
    const out = execution.allocate(rows * channels, `bioemu2.${label}`);
    execution.dispatch({
      source: sliceSource(rows, 3 * channels, from, channels),
      output: out,
      inputs: { xTex: qkv },
      label: `bioemu2.slice:${rows}x${channels}:${label}`,
    });
    return out;
  };
  const key = slice(channels, "key");
  const value = slice(2 * channels, "value");
  execution.release(qkv);
  const bias = given ?? pairBias(execution, {
    pair, rows: tokens * tokens, channels: pair.elements / (tokens * tokens),
    heads, weights: w, epsilon,
  });
  const attended = execution.allocate(rows * channels, "bioemu2.token-attended");
  execution.dispatch({
    source: fullAttentionSource(tokens, channels, heads,
      !execution.noAttendDot4 && channels % (heads * 4) === 0, samples),
    output: attended,
    inputs: { qTex: query, kTex: key, vTex: value, biasTex: bias, padTex: pad },
    label: `bioemu2.token-attend:${rows}x${channels}:${heads}`,
  });
  for (const tensor of [query, key, value]) execution.release(tensor);
  if (bias !== given) execution.release(bias);
  const gatedStride4 = sharedXStride4(execution, rows, channels, [{ outChannels: channels, weight: w.projO }]);
  const gated = allocateX(execution, rows, channels, gatedStride4, "bioemu2.token-gated");
  execution.dispatch({
    source: productSource(rows * channels, channels / 4, gatedStride4),
    output: gated,
    inputs: { aTex: gate, bTex: attended },
    label: `bioemu2.token-gate:${rows}x${channels}`,
  });
  execution.release(attended);
  execution.release(gate);
  const out = project(execution, {
    x: gated, weight: w.projO, rows, inChannels: channels,
    outChannels: channels, weightLayout: "out-major",
  });
  execution.release(gated);
  return out;
}

/** The atom stream's windowed attention, with its gate and output projection. */
export function windowAttention(execution, call) {
  const { x, pair, pad, atoms, channels, heads, weights: w, epsilon = 1e-5 } = call;
  // 🔴 THE BIAS MAY ARRIVE READY-MADE, which is not an optimisation: the input
  // embedder's window bias is computed ONCE a fold on the host - the windowed
  // pair geometry and its three-layer MLP are per atom-window pair and happen
  // before any layer runs - while the denoiser reads its own `ctx.p` every
  // layer. One kernel either way; what differs is who built the bias.
  const given = call.bias;
  // `samples` stacked sample-major in x, as `fullAttention` takes them: `atoms` is ONE sample's
  const samples = call.samples ?? 1;
  const rows = samples * atoms;
  if (given !== undefined && attendFuses(execution, channels, heads)) {
    return fusedAttention(execution, {
      window: true, x, rows: atoms, samples, channels, heads, weights: w, bias: given, pad });
  }
  const { qkv, query, gate } = projections(execution, {
    x, rows, channels, weights: w,
  });
  // 🔴 THE KEY AND VALUE TRAVEL TOGETHER, because the kernel reads them as
  // per-head halves of this one `2 * channels` slice - see the note at the top.
  const kv = execution.allocate(rows * 2 * channels, "bioemu2.kv");
  execution.dispatch({
    source: sliceSource(rows, 3 * channels, channels, 2 * channels),
    output: kv,
    inputs: { xTex: qkv },
    label: `bioemu2.slice-kv:${rows}x${2 * channels}`,
  });
  execution.release(qkv);
  const windows = atoms / WINDOW_QUERIES;
  const bias = given ?? pairBias(execution, {
    pair, rows: windows * WINDOW_QUERIES * WINDOW_KEYS,
    channels: pair.elements / (windows * WINDOW_QUERIES * WINDOW_KEYS),
    heads, weights: w, epsilon,
  });
  const attended = execution.allocate(rows * channels, "bioemu2.atom-attended");
  execution.dispatch({
    source: windowAttentionSource(atoms, channels, heads,
      !execution.noAttendDot4 && channels % (heads * 4) === 0, samples),
    output: attended,
    inputs: { qTex: query, kvTex: kv, biasTex: bias, padTex: pad },
    label: `bioemu2.atom-attend:${rows}x${channels}:${heads}`,
  });
  for (const tensor of [query, kv]) execution.release(tensor);
  if (bias !== given) execution.release(bias);
  const gatedStride4 = sharedXStride4(execution, rows, channels, [{ outChannels: channels, weight: w.projO }]);
  const gated = allocateX(execution, rows, channels, gatedStride4, "bioemu2.atom-gated");
  execution.dispatch({
    source: productSource(rows * channels, channels / 4, gatedStride4),
    output: gated,
    inputs: { aTex: gate, bTex: attended },
    label: `bioemu2.atom-gate:${rows}x${channels}`,
  });
  execution.release(attended);
  execution.release(gate);
  const out = project(execution, {
    x: gated, weight: w.projO, rows, inChannels: channels,
    outChannels: channels, weightLayout: "out-major",
  });
  execution.release(gated);
  return out;
}

/**
 * Everything a denoiser call reads that is the same at every sampler step: the 16 layers' pair biases for both
 * streams, the single conditioning's sigma-free half `single_embed(LN[s_trunk | s_inputs])`, the token stream's
 * `s_trunk_to_a(s_trunk)`, and the row maps a batch of `samples` walks through.
 *
 * 🔴 THESE WERE ALL RECOMPUTED AT EVERY STEP, the biases at every LAYER of every step: 32 pair-bias draws over the
 * `T^2 x 256` token pair and the windowed atom pair, the 839-wide LayerNorm and its projection, the trunk's norm and
 * its projection - for tensors the sampler cannot change. webgpu/bioemu2/network.js makes the same ones once in
 * `denoiserProgram`. Each is the same kernel on the same inputs, so hoisting it is exact.
 *
 * 🔴 AND THE MAPS ARE WHAT LET ONE CONDITIONING PROJECTION SERVE EVERY ATOM OF EVERY SAMPLE. The conditioning is
 * per TOKEN - an atom's row was a gather of its token's - and every layer then normalised, activated and projected
 * that gathered `N x 768` row by row (`576x768x768` and `576x768x1920` GEMMs a layer at 68 tokens). Made at the
 * `T + 1` token rows instead (the last a zero row, which is what a padded atom's gather gave) and read through
 * `atomCond`, it is the same rows of the same projection. A batch is stacked SAMPLE-MAJOR: `atomToken` takes an atom
 * row to its own sample's token row, `centre` and `centreOrPad` a token row to its sample's centre atom row.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{features: any, context: any, sInputs: Tensor, sTrunk: Tensor, weights: any, dims: any,
 *   samples?: number, epsilon?: number}} call
 * @returns {any} the prepared context, with `release()`
 */
export function bioemu2DenoiserPrepare(execution, call) {
  const { features: f, context, sInputs, sTrunk, weights: w, dims, epsilon = 1e-5 } = call;
  const samples = call.samples ?? 1;
  const { atoms, tokens, layers } = dims;
  const tokenChannels = dims.tokenChannels;
  const held = [];
  const keep = (tensor) => {
    held.push(tensor);
    return tensor;
  };

  // --- the row maps, from the features the device already has -------------
  const atomTokenOrPad = execution.read(f.atomToTokenOrPad);
  const centreAtom = execution.read(f.centreAtom);
  const centreAtomOrPad = execution.read(f.centreAtomOrPad);
  const atomCond = new Float32Array(samples * atoms);
  const atomToken = new Float32Array(samples * atoms);
  const centre = new Float32Array(samples * tokens);
  const centreOrPad = new Float32Array(samples * tokens);
  const tokenCond = new Float32Array(samples * tokens);
  for (let s = 0; s < samples; s += 1) {
    for (let atom = 0; atom < atoms; atom += 1) {
      const token = atomTokenOrPad[atom];
      atomCond[s * atoms + atom] = token < 0 ? tokens : token;
      atomToken[s * atoms + atom] = token < 0 ? -1 : s * tokens + token;
    }
    for (let token = 0; token < tokens; token += 1) {
      centre[s * tokens + token] = s * atoms + centreAtom[token];
      centreOrPad[s * tokens + token] = centreAtomOrPad[token] < 0 ? -1 : s * atoms + centreAtomOrPad[token];
      tokenCond[s * tokens + token] = token;
    }
  }
  const withZeroRow = new Float32Array(tokens + 1);
  for (let token = 0; token <= tokens; token += 1) withZeroRow[token] = token < tokens ? token : -1;
  const maps = {
    atomCond: keep(execution.upload(atomCond, "bioemu2.map.atom-cond")),
    // the identity at one sample: a plain slice then, as before the maps
    tokenCond: samples === 1 ? null : keep(execution.upload(tokenCond, "bioemu2.map.token-cond")),
    // at one sample these are the features' own, so the draws read the tensors they always did
    atomToken: samples === 1 ? f.atomToTokenOrPad : keep(execution.upload(atomToken, "bioemu2.map.atom-token")),
    centre: samples === 1 ? f.centreAtom : keep(execution.upload(centre, "bioemu2.map.centre")),
    centreOrPad: samples === 1 ? f.centreAtomOrPad
      : keep(execution.upload(centreOrPad, "bioemu2.map.centre-or-pad")),
    withZeroRow: keep(execution.upload(withZeroRow, "bioemu2.map.zero-row")),
  };

  // --- the 16 layers' pair biases, both streams ---------------------------
  const windows = atoms / WINDOW_QUERIES;
  const tokenBias = [];
  const atomBias = [];
  for (let layer = 0; layer < layers; layer += 1) {
    tokenBias.push(keep(pairBias(execution, {
      pair: context.z, rows: tokens * tokens, channels: context.z.elements / (tokens * tokens),
      heads: dims.tokenHeads, weights: w.tokenBlocks[layer].attention, epsilon,
    })));
    atomBias.push(keep(pairBias(execution, {
      pair: context.p, rows: windows * WINDOW_QUERIES * WINDOW_KEYS,
      channels: context.p.elements / (windows * WINDOW_QUERIES * WINDOW_KEYS),
      heads: dims.atomHeads, weights: w.atomBlocks[layer].attention, epsilon,
    })));
  }

  // --- single_embed(LN[s_trunk | s_inputs]): the conditioning before sigma --
  const trunkChannels = sTrunk.elements / tokens;
  const inputChannels = sInputs.elements / tokens;
  const joined = execution.allocate(tokens * (trunkChannels + inputChannels), "bioemu2.single-cat");
  execution.dispatch({
    source: concatSource(tokens, trunkChannels, inputChannels),
    output: joined,
    inputs: { leftTex: sTrunk, rightTex: sInputs },
    label: `bioemu2.single-cat:${tokens}x${trunkChannels + inputChannels}`,
  });
  const normalisedSingle = layerNorm(execution, {
    x: joined, weight: w.singleConditioner.normScale, bias: w.singleConditioner.normOffset,
    rows: tokens, channels: trunkChannels + inputChannels, epsilon,
  });
  execution.release(joined);
  const single = keep(linear(execution, {
    x: normalisedSingle, weight: w.singleConditioner.singleEmbedWeight,
    bias: w.singleConditioner.singleEmbedBias, rows: tokens,
    inChannels: trunkChannels + inputChannels, outChannels: tokenChannels,
    weightLayout: "out-major",
  }));
  execution.release(normalisedSingle);

  // --- s_trunk_to_a(s_trunk) ----------------------------------------------
  const normalisedTrunk = layerNorm(execution, {
    x: sTrunk, weight: w.sTrunkToA0Scale, bias: w.sTrunkToA0Offset,
    rows: tokens, channels: trunkChannels, epsilon,
  });
  const aBase = keep(linear(execution, {
    x: normalisedTrunk, weight: w.sTrunkToA1, rows: tokens,
    inChannels: trunkChannels, outChannels: tokenChannels, weightLayout: "out-major",
  }));
  execution.release(normalisedTrunk);

  return {
    samples, maps, tokenBias, atomBias, single, aBase,
    release: () => {
      for (const tensor of held) execution.release(tensor);
      held.length = 0;
    },
  };
}

/**
 * D(x, sigma): the whole denoiser, forward.
 *
 * 🔴 THE CONDITIONING IS `[s_trunk | s_inputs]` AND THE WIDTHS ARE THE DATA'S -
 * 384 and 455 in the shipped model - "and a typed-in width misreads both".
 *
 * 🔴 AND sigma REACHES FOUR PLACES: `c_in` on the input, the Fourier embedding
 * of `log(sigma / sigma_data) / 4` in the conditioning, and `c_skip` and `c_out`
 * on the output. A port that scaled only the input would return a finite
 * structure at every noise level and the wrong one at all but one.
 *
 * 🔴 AND x MAY CARRY A BATCH: `S * atoms * 3`, SAMPLE-MAJOR, every sample conditioned alike and attending only
 * within itself - one call where S walks would each make one, the draws shared and the rows S times. `prepared`
 * (`bioemu2DenoiserPrepare`, at that S) is what a sampler makes once a fold; without it the call makes and drops
 * its own.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, sigma: number, features: any, context: any,
 *   sInputs: Tensor, sTrunk: Tensor, weights: any, dims: any,
 *   hostFourier: {weight: Float32Array, bias: Float32Array}, prepared?: any,
 *   epsilon?: number, onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {Tensor} `S * atoms * 3`; the caller's to release
 */
export function bioemu2Denoise(execution, call) {
  return drain(bioemu2DenoiseSteps(execution, call));
}

/**
 * `bioemu2Denoise` as a generator that yields after each of the 16 layers, so the page can let go of its thread
 * inside a denoiser call (~0.8 s at 68 tokens on an M2) - the same draws in the same order; `drain` is the call.
 */
export function* bioemu2DenoiseSteps(execution, call) {
  const {
    x, sigma, features: f, context, sInputs, sTrunk, weights: w, dims,
    epsilon = 1e-5,
  } = call;
  const { atoms, tokens, layers } = dims;
  const samples = x.elements / (atoms * 3);
  if (!Number.isInteger(samples) || samples < 1) {
    throw new RangeError(`bioemu2 denoiser: x holds ${x.elements} floats, not a whole number of ${atoms} x 3`);
  }
  const own = call.prepared === undefined;
  const prepared = own
    ? bioemu2DenoiserPrepare(execution, { features: f, context, sInputs, sTrunk, weights: w, dims, samples, epsilon })
    : call.prepared;
  if (prepared.samples !== samples) {
    throw new RangeError(`bioemu2 denoiser: prepared for ${prepared.samples} samples, handed ${samples}`);
  }
  const { maps } = prepared;
  const atomRows = samples * atoms;
  const tokenRows = samples * tokens;
  const sigmaData = dims.sigmaData ?? SIGMA_DATA;
  const atomChannels = dims.atomChannels;
  const tokenChannels = dims.tokenChannels;
  const temporary = [];
  const keep = (tensor) => {
    temporary.push(tensor);
    return tensor;
  };

  // c_in, and the masked scaled input.
  const cIn = 1 / Math.sqrt(sigma * sigma + sigmaData * sigmaData);
  const scaled = keep(execution.allocate(atomRows * 3, "bioemu2.r"));
  execution.dispatch({
    source: scaleSource(atomRows * 3),
    output: scaled,
    inputs: { xTex: x },
    floats: { factor: cIn },
    label: `bioemu2.c-in:${atomRows}`,
  });
  const r = keep(execution.allocate(atomRows * 3, "bioemu2.r-masked"));
  execution.dispatch({
    source: maskRowsSource(atomRows, 3, atoms),
    output: r,
    inputs: { xTex: scaled, maskTex: f.atomMask },
    label: `bioemu2.r-mask:${atomRows}`,
  });
  call.onStage?.("r", r);

  // The single conditioning: the prepared [s_trunk | s_inputs] embedding, then the Fourier term.
  // 🔴 t IS `0.25 * log(sigma / sigma_data)`, embedded on the HOST: it is one
  // row, and the alternative is a texture to hold a number the host has.
  const t = 0.25 * Math.log(sigma / sigmaData);
  const fourier = new Float32Array(call.hostFourier.weight.length);
  for (let i = 0; i < fourier.length; i += 1) {
    fourier[i] = Math.cos(
      2 * Math.PI * (t * call.hostFourier.weight[i] + call.hostFourier.bias[i]));
  }
  const fourierTensor = keep(execution.upload(fourier, "bioemu2.fourier"));
  const normalisedFourier = keep(layerNorm(execution, {
    x: fourierTensor, weight: w.singleConditioner.fourierNormScale,
    bias: w.singleConditioner.fourierNormOffset,
    rows: 1, channels: fourier.length, epsilon,
  }));
  const fourierTerm = keep(linear(execution, {
    x: normalisedFourier, weight: w.singleConditioner.fourierToSingle, rows: 1,
    inChannels: fourier.length, outChannels: tokenChannels, weightLayout: "out-major",
  }));
  let cond = keep(execution.allocate(tokens * tokenChannels, "bioemu2.cond"));
  execution.dispatch({
    source: broadcastRowSource(tokens, tokenChannels),
    output: cond,
    inputs: { xTex: prepared.single, rowTex: fourierTerm },
    label: `bioemu2.fourier-add:${tokens}x${tokenChannels}`,
  });
  for (const group of w.singleConditioner.transitions) {
    const normalised = layerNorm(execution, {
      x: cond, weight: group.normScale, bias: group.normOffset,
      rows: tokens, channels: tokenChannels, epsilon,
    });
    const delta = transitionNoNorm(execution, {
      x: normalised, rows: tokens, channels: tokenChannels, weights: group,
    });
    execution.release(normalised);
    const next = keep(execution.allocate(tokens * tokenChannels, "bioemu2.cond"));
    execution.dispatch({
      source: sumSource(tokens * tokenChannels),
      output: next,
      inputs: { aTex: cond, bTex: delta },
      label: `bioemu2.cond-residual:${tokens}x${tokenChannels}`,
    });
    execution.release(delta);
    cond = next;
  }
  call.onStage?.("cond", cond);
  // ...and with a zero row after the last token: the row a padded atom's conditioning reads (`maps.atomCond`)
  const condWithZero = keep(execution.allocate((tokens + 1) * tokenChannels, "bioemu2.cond-zero-row"));
  execution.dispatch({
    source: gatherRowsSource(tokens + 1, tokenChannels),
    output: condWithZero,
    inputs: { sourceTex: cond, indexTex: maps.withZeroRow },
    label: `bioemu2.cond-zero-row:${tokens + 1}x${tokenChannels}`,
  });

  // The two streams' initial states.
  const rToQ = keep(linear(execution, {
    x: r, weight: w.rToQ, rows: atomRows, inChannels: 3, outChannels: atomChannels,
    weightLayout: "out-major",
  }));
  let q = keep(execution.allocate(atomRows * atomChannels, "bioemu2.q"));
  execution.dispatch({
    source: sumSource(atomRows * atomChannels, atoms * atomChannels),
    output: q,
    inputs: { aTex: context.q0, bTex: rToQ },
    label: `bioemu2.q0:${atomRows}x${atomChannels}`,
  });
  // ...plus the scaled input AT EVERY TOKEN'S CENTRE ATOM, masked by the token pad.
  const centreR = keep(execution.allocate(tokenRows * 3, "bioemu2.centre-r"));
  execution.dispatch({
    source: gatherRowsSource(tokenRows, 3),
    output: centreR,
    inputs: { sourceTex: r, indexTex: maps.centre },
    label: `bioemu2.centre-r:${tokenRows}`,
  });
  const maskedCentre = keep(execution.allocate(tokenRows * 3, "bioemu2.centre-masked"));
  execution.dispatch({
    source: maskRowsSource(tokenRows, 3, tokens),
    output: maskedCentre,
    inputs: { xTex: centreR, maskTex: f.tokenPad },
    label: `bioemu2.centre-mask:${tokenRows}`,
  });
  const rToA = keep(linear(execution, {
    x: maskedCentre, weight: w.rTokenToA, rows: tokenRows, inChannels: 3,
    outChannels: tokenChannels, weightLayout: "out-major",
  }));
  let a = keep(execution.allocate(tokenRows * tokenChannels, "bioemu2.a"));
  execution.dispatch({
    source: sumSource(tokenRows * tokenChannels, tokens * tokenChannels),
    output: a,
    inputs: { aTex: prepared.aBase, bTex: rToA },
    label: `bioemu2.a0:${tokenRows}x${tokenChannels}`,
  });
  call.onStage?.("q0", q);
  call.onStage?.("a0", a);

  // 🔴 THE ATOM STREAM'S CONDITIONING IS PROJECTED AT THE TOKEN ROWS (`bioemu2DenoiserPrepare`'s note): the two
  // GEMMs a layer it feeds were 576 x 768 x 768 and 576 x 768 x 1920 at 68 tokens and are 69 rows now. Exact in f32
  // (this denoiser's gate reads the same relative RMS to four digits); with the fold's float16 x the narrowed
  // rounding differs by row count, worst atom 0.056 A over a 20-step fold (0.001 A with `--no-half-x`).
  // `--no-token-row-cond` is the control arm: the conditioning gathered to every atom row first, as it was.
  let atomCond = { source: condWithZero, rows: tokens + 1, map: maps.atomCond };
  if (execution.noTokenRowCond) {
    const atAtom = keep(execution.allocate(atomRows * tokenChannels, "bioemu2.cond-at-atom"));
    execution.dispatch({
      source: gatherRowsSource(atomRows, tokenChannels),
      output: atAtom,
      inputs: { sourceTex: condWithZero, indexTex: maps.atomCond },
      label: `bioemu2.cond-at-atom:${atomRows}x${tokenChannels}`,
    });
    atomCond = { source: atAtom, rows: atomRows, map: null };
  }
  for (let layer = 0; layer < layers; layer += 1) {
    const atomWeights = w.atomBlocks[layer];
    const tokenWeights = w.tokenBlocks[layer];
    const nextQ = conditionedLayer(execution, {
      x: q, conditioning: atomCond.source, condRows: atomCond.rows, condMap: atomCond.map,
      rows: atomRows, channels: atomChannels,
      heads: dims.atomHeads, weights: atomWeights, epsilon,
      attendConsumers: attentionReaders(atomWeights.attention, atomChannels),
      attend: (input) => windowAttention(execution, {
        x: input, pair: context.p, pad: f.atomPad, atoms, channels: atomChannels,
        heads: dims.atomHeads, weights: atomWeights.attention, epsilon,
        bias: prepared.atomBias[layer], samples,
      }),
    });
    const nextA = conditionedLayer(execution, {
      x: a, conditioning: cond, condRows: tokens, condMap: maps.tokenCond,
      rows: tokenRows, channels: tokenChannels,
      heads: dims.tokenHeads, weights: tokenWeights, epsilon,
      attendConsumers: attentionReaders(tokenWeights.attention, tokenChannels),
      attend: (input) => fullAttention(execution, {
        x: input, pair: context.z, pad: f.tokenPad, tokens, channels: tokenChannels,
        heads: dims.tokenHeads, weights: tokenWeights.attention, epsilon,
        bias: prepared.tokenBias[layer], samples,
      }),
    });
    execution.release(q);
    execution.release(a);
    q = nextQ;
    a = nextA;

    // 🔴 AND THE TWO COUPLE THROUGH THE CENTRE ATOM, BOTH WAYS, each direction
    // its own three-way slice: alpha, beta and a gate.
    const coupling = w.coupling[layer];
    const width = tokenChannels + atomChannels;
    // `condSource` at `condRows` rows, read by each of the `rows` rows through `condMap` (null: the same rows)
    const mix = (condSource, condRows, condMap, rows, gatherIndex, left, right, weights, outChannels) => {
      const normalised = layerNorm(execution, {
        x: condSource, weight: weights.cond0Scale, bias: weights.cond0Offset,
        rows: condRows, channels: condSource.elements / condRows, epsilon,
      });
      const condChannels = condSource.elements / condRows;
      const projectedWidth = 2 * width + outChannels;
      const siluStride4 = sharedXStride4(execution, condRows, condChannels,
        [{ outChannels: projectedWidth, weight: weights.cond2Weight }]);
      const activated = allocateX(execution, condRows, condChannels, siluStride4, "bioemu2.mix-silu");
      execution.dispatch({
        source: siluSource(normalised.elements, condChannels / 4, siluStride4),
        output: activated,
        inputs: { xTex: normalised },
        label: `bioemu2.mix-silu:${normalised.elements}`,
      });
      execution.release(normalised);
      const projected = project(execution, {
        x: activated, weight: weights.cond2Weight, bias: weights.cond2Bias, rows: condRows,
        inChannels: condSource.elements / condRows,
        outChannels: 2 * width + outChannels, weightLayout: "out-major",
      });
      execution.release(activated);
      const slice = (from, howWide, label) => {
        const out = execution.allocate(rows * howWide, `bioemu2.mix.${label}`);
        execution.dispatch(condMap === null ? {
          source: sliceSource(rows, 2 * width + outChannels, from, howWide),
          output: out,
          inputs: { xTex: projected },
          label: `bioemu2.mix-slice:${rows}x${howWide}:${label}`,
        } : {
          source: gatherSliceSource(rows, 2 * width + outChannels, from, howWide),
          output: out,
          inputs: { xTex: projected, indexTex: condMap },
          label: `bioemu2.mix-gather-slice:${rows}x${howWide}:${label}`,
        });
        return out;
      };
      const gathered = execution.allocate(rows * left.channels, "bioemu2.mix-gather");
      execution.dispatch({
        source: gatherRowsSource(rows, left.channels),
        output: gathered,
        inputs: { sourceTex: left.tensor, indexTex: gatherIndex },
        label: `bioemu2.mix-gather:${rows}x${left.channels}`,
      });
      const joinedMix = execution.allocate(rows * width, "bioemu2.mix-cat");
      execution.dispatch({
        source: concatSource(rows, left.channels, right.channels),
        output: joinedMix,
        inputs: { leftTex: gathered, rightTex: right.tensor },
        label: `bioemu2.mix-cat:${rows}x${width}`,
      });
      execution.release(gathered);
      if (!execution.noCondFuse) {
        // alpha and beta read by the adaLN, and the gate by the residual against the stream's own state - the
        // coupling's sum - where three slices, the gate product and the sum were five draws
        const modulatedFused = condAdaLn(execution, {
          x: joinedMix, cond: projected, map: condMap, condWidth: projectedWidth, alphaAt: 0, betaAt: width,
          rows, channels: width, epsilon, label: "bioemu2.mix-cond-adaln",
          consumers: [weights.fc1, weights.fc2].map((weight) => ({ outChannels: weight.elements / width, weight })),
        });
        execution.release(joinedMix);
        const mixedFused = transitionNoNorm(execution, {
          x: modulatedFused, rows, channels: width, weights, outChannels,
        });
        execution.release(modulatedFused);
        const outFused = condGatedResidual(execution, {
          x: right.tensor, branch: mixedFused, cond: projected, map: condMap, condWidth: projectedWidth,
          gateAt: 2 * width, rows, channels: outChannels, label: "bioemu2.mix-cond-gated",
        });
        execution.release(mixedFused);
        execution.release(projected);
        return { coupled: outFused };
      }
      const alpha = slice(0, width, "alpha");
      const beta = slice(width, width, "beta");
      const modulated = execution.allocate(rows * width, "bioemu2.mix-adaln");
      if (adaLnSplits(execution, width)) {
        adaLnSplit(execution,
          { x: joinedMix, alpha, beta, rows, channels: width, epsilon }, modulated);
      } else {
        execution.dispatch({
          source: adaLnSource(rows, width),
          output: modulated,
          inputs: { xTex: joinedMix, alphaTex: alpha, betaTex: beta },
          floats: { epsilon },
          label: `bioemu2.mix-adaln:${rows}x${width}`,
        });
      }
      execution.release(joinedMix);
      execution.release(alpha);
      execution.release(beta);
      const mixed = transitionNoNorm(execution, {
        x: modulated, rows, channels: width, weights, outChannels,
      });
      execution.release(modulated);
      const gate = slice(2 * width, outChannels, "gate");
      const out = execution.allocate(rows * outChannels, "bioemu2.mix");
      execution.dispatch({
        source: productSource(rows * outChannels),
        output: out,
        inputs: { aTex: gate, bTex: mixed },
        label: `bioemu2.mix-gate:${rows}x${outChannels}`,
      });
      execution.release(gate);
      execution.release(mixed);
      execution.release(projected);
      return { gated: out };
    };
    const sToQ = mix(atomCond.source, atomCond.rows, atomCond.map, atomRows, maps.atomToken,
      { tensor: a, channels: tokenChannels }, { tensor: q, channels: atomChannels },
      { ...coupling.qCoupling, fc1: coupling.sToQ.fc1, fc2: coupling.sToQ.fc2,
        fc3: coupling.sToQ.fc3 },
      atomChannels);
    const qToS = mix(cond, tokens, maps.tokenCond, tokenRows, maps.centreOrPad,
      { tensor: q, channels: atomChannels }, { tensor: a, channels: tokenChannels },
      { ...coupling.sCoupling, fc1: coupling.qToS.fc1, fc2: coupling.qToS.fc2,
        fc3: coupling.qToS.fc3 },
      tokenChannels);
    // (the fused mixers hand back the coupled state itself; the control arm the gated term, summed here)
    const coupled = (term, base, rows, channels, label) => {
      if (term.coupled !== undefined) return term.coupled;
      const out = execution.allocate(rows * channels, `bioemu2.${label[0]}`);
      execution.dispatch({
        source: sumSource(rows * channels),
        output: out,
        inputs: { aTex: base, bTex: term.gated },
        label: `bioemu2.${label}:${rows}x${channels}`,
      });
      execution.release(term.gated);
      return out;
    };
    const mixedQ = coupled(sToQ, q, atomRows, atomChannels, "q-couple");
    const mixedA = coupled(qToS, a, tokenRows, tokenChannels, "a-couple");
    execution.release(q);
    execution.release(a);
    q = mixedQ;
    a = mixedA;
    call.onStage?.(`layer${layer}.q`, q);
    call.onStage?.(`layer${layer}.a`, a);
    yield { stage: "denoiser-layer", layer };
  }

  // The readout: the atom stream plus the token stream at every atom.
  const hidden = w.readout.qToHidden.elements / atomChannels;
  const fromQ = linear(execution, {
    x: q, weight: w.readout.qToHidden, rows: atomRows, inChannels: atomChannels,
    outChannels: hidden, weightLayout: "out-major",
  });
  const fromA = linear(execution, {
    x: a, weight: w.readout.sToHidden, rows: tokenRows, inChannels: tokenChannels,
    outChannels: hidden, weightLayout: "out-major",
  });
  const atAtom = execution.allocate(atomRows * hidden, "bioemu2.readout-at-atom");
  execution.dispatch({
    source: gatherRowsSource(atomRows, hidden),
    output: atAtom,
    inputs: { sourceTex: fromA, indexTex: maps.atomToken },
    label: `bioemu2.readout-gather:${atomRows}x${hidden}`,
  });
  execution.release(fromA);
  const summed = execution.allocate(atomRows * hidden, "bioemu2.readout");
  execution.dispatch({
    source: sumSource(atomRows * hidden),
    output: summed,
    inputs: { aTex: fromQ, bTex: atAtom },
    label: `bioemu2.readout-add:${atomRows}x${hidden}`,
  });
  execution.release(fromQ);
  execution.release(atAtom);
  const normalised = layerNorm(execution, {
    x: summed, weight: w.readout.outNormScale, rows: atomRows, channels: hidden, epsilon,
  });
  execution.release(summed);
  const update = linear(execution, {
    x: normalised, weight: w.readout.toOut, rows: atomRows, inChannels: hidden,
    outChannels: 3, weightLayout: "out-major",
  });
  execution.release(normalised);
  call.onStage?.("readout", update);

  // 🔴 AND THE ANSWER IS THE EDM BLEND, not the update: c_skip * x + c_out * out.
  const cSkip = (sigmaData * sigmaData) / (sigma * sigma + sigmaData * sigmaData);
  const cOut = (sigma * sigmaData) / Math.sqrt(sigmaData * sigmaData + sigma * sigma);
  const out = execution.allocate(atomRows * 3, "bioemu2.denoised");
  execution.dispatch({
    source: blendSource(atomRows * 3),
    output: out,
    inputs: { aTex: x, bTex: update },
    floats: { keep: cSkip, take: cOut },
    label: `bioemu2.blend:${atomRows}`,
  });
  execution.release(update);
  execution.release(q);
  execution.release(a);
  for (const tensor of temporary) execution.release(tensor);
  if (own) prepared.release();
  out.label = "bioemu2.denoised";
  return out;
}


/** `out[row][c] = x[row][c] + row0[c]`. */
const broadcastRowSourceUncached = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D rowTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) + fetch4(rowTex, texel % C4);
}`;

/** `out = keep * a + take * b`. */
const blendSourceUncached = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;
uniform float keep;
uniform float take;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = keep * fetch4(aTex, texel) + take * fetch4(bTex, texel);
}`;

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const scaleSource = memoSource(scaleSourceUncached);
export const sumSource = memoSource(sumSourceUncached);
export const productSource = memoSource(productSourceUncached);
export const siluSource = memoSource(siluSourceUncached);
export const sigmoidSource = memoSource(sigmoidSourceUncached);
export const sliceSource = memoSource(sliceSourceUncached);
export const concatSource = memoSource(concatSourceUncached);
export const gatherRowsSource = memoSource(gatherRowsSourceUncached);
export const maskRowsSource = memoSource(maskRowsSourceUncached);
export const gatherSliceSource = memoSource(gatherSliceSourceUncached);
export const condAdaLnSource = memoSource(condAdaLnSourceUncached);
export const fusedAttentionSource = memoSource(fusedAttentionSourceUncached);
export const condGatedResidualSource = memoSource(condGatedResidualSourceUncached);
export const splitAdaLnSource = memoSource(splitAdaLnSourceUncached);
export const adaLnSource = memoSource(adaLnSourceUncached);
export const pairBiasSource = memoSource(pairBiasSourceUncached);
export const fullAttentionSource = memoSource(fullAttentionSourceUncached);
export const windowAttentionSource = memoSource(windowAttentionSourceUncached);
export const addRowBiasSource = memoSource(addRowBiasSourceUncached);
export const broadcastRowSource = memoSource(broadcastRowSourceUncached);
export const blendSource = memoSource(blendSourceUncached);
