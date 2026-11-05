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
import { ONE_TARGET, PRELUDE } from "../runtime/execution.js";
import { layerNorm } from "../kernels/layer-norm.js";
import { linear } from "../kernels/linear.js";

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

/** `out = x * factor`, a scalar the host computed from sigma. */
export const scaleSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;
uniform float factor;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) * factor;
}`;

/** `out = a + b`. */
export const sumSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(aTex, texel) + fetch4(bTex, texel);
}`;

/** `out = a * b`, elementwise over equal shapes. */
export const productSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(aTex, texel) * fetch4(bTex, texel);
}`;

/** `out = silu(x)`. */
export const siluSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 x = fetch4(xTex, texel);
  result = x / (vec4(1.0) + exp(-x));
}`;

/** `out = sigmoid(x)`. */
export const sigmoidSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(1.0) / (vec4(1.0) + exp(-fetch4(xTex, texel)));
}`;

/** `out[row][c] = x[row][from + c]`, a channel range of every row. */
export const sliceSource = (rows, channels, from, width) => `${PRELUDE}${ONE_TARGET}
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
export const concatSource = (rows, leftWidth, rightWidth) => (
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
export const gatherRowsSource = (rows, channels) => (channels % 4 === 0
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

/** `out[row][c] = x[row][c] * mask[row]`, with the same ragged split. */
export const maskRowsSource = (rows, channels) => (channels % 4 === 0
  ? `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) * fetch1(maskTex, texel / C4);
}`
  : `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D maskTex;

float at(int element) {
  return fetch1(xTex, element) * fetch1(maskTex, element / CHANNELS);
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
export const adaLnSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
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
export const pairBiasSource = (rows, channels, heads, perHead = false) => {
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
export const fullAttentionSource = (rows, channels, heads) => `${PRELUDE}${ONE_TARGET}
const int ROWS = ${rows};
const int CHANNELS = ${channels};
const int HEADS = ${heads};
const int HEAD_DIM = ${channels / heads};
const int ELEMENTS = ${rows * channels};
const float SCALE = ${(1 / Math.sqrt(channels / heads)).toFixed(10)};
uniform sampler2D qTex;
uniform sampler2D kTex;
uniform sampler2D vTex;
uniform sampler2D biasTex;
uniform sampler2D padTex;

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  int i = base / CHANNELS;
  int c = base - i * CHANNELS;
  int head = c / HEAD_DIM;
  int qBase = i * CHANNELS + head * HEAD_DIM;

  float largest = -1e30;
  for (int j = 0; j < ROWS; ++j) {
    if (fetch1(padTex, j) == 0.0) continue;
    int kBase = j * CHANNELS + head * HEAD_DIM;
    float dot = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += fetch1(qTex, qBase + d) * fetch1(kTex, kBase + d);
    }
    largest = max(largest, dot * SCALE + fetch1(biasTex, (i * ROWS + j) * HEADS + head));
  }
  float sum = 0.0;
  vec4 gathered = vec4(0.0);
  for (int j = 0; j < ROWS; ++j) {
    if (fetch1(padTex, j) == 0.0) continue;
    int kBase = j * CHANNELS + head * HEAD_DIM;
    float dot = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += fetch1(qTex, qBase + d) * fetch1(kTex, kBase + d);
    }
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
export const windowAttentionSource = (atoms, channels, heads) => `${PRELUDE}${ONE_TARGET}
const int ATOMS = ${atoms};
const int CHANNELS = ${channels};
const int HEADS = ${heads};
const int HEAD_DIM = ${channels / heads};
const int W_Q = ${WINDOW_QUERIES};
const int W_K = ${WINDOW_KEYS};
const int CTX = ${WINDOW_CONTEXT};
const int ELEMENTS = ${atoms * channels};
const float SCALE = ${(1 / Math.sqrt(channels / heads)).toFixed(10)};
uniform sampler2D qTex;
uniform sampler2D kvTex;
uniform sampler2D biasTex;
uniform sampler2D padTex;

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  int i = base / CHANNELS;
  int c = base - i * CHANNELS;
  int head = c / HEAD_DIM;
  int qBase = i * CHANNELS + head * HEAD_DIM;
  int window = i / W_Q;
  int inWindow = i - window * W_Q;

  float largest = -1e30;
  for (int slot = 0; slot < W_K; ++slot) {
    int j = window * W_Q - CTX + slot;
    if (j < 0 || j >= ATOMS) continue;
    if (fetch1(padTex, j) == 0.0) continue;
    int kBase = j * 2 * CHANNELS + head * HEAD_DIM;
    float dot = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += fetch1(qTex, qBase + d) * fetch1(kvTex, kBase + d);
    }
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
    int kBase = j * 2 * CHANNELS + head * HEAD_DIM;
    float dot = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += fetch1(qTex, qBase + d) * fetch1(kvTex, kBase + d);
    }
    float bias = fetch1(biasTex,
      ((window * W_Q + inWindow) * W_K + slot) * HEADS + head);
    float weight = exp(dot * SCALE + bias - largest);
    sum += weight;
    // ...the VALUE half: a whole CHANNELS further along this atom's row.
    gathered += weight * fetch4(kvTex, (kBase + CHANNELS + c % HEAD_DIM) / 4);
  }
  result = gathered / sum;
}`;

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
  // silu(LN(c)) @ cond_2, which is 6 * channels wide.
  const normalised = layerNorm(execution, {
    x: conditioning, weight: w.cond0Scale, bias: w.cond0Offset,
    rows, channels: conditioning.elements / rows, epsilon,
  });
  const activated = execution.allocate(normalised.elements, "bioemu2.cond-silu");
  execution.dispatch({
    source: siluSource(normalised.elements),
    output: activated,
    inputs: { xTex: normalised },
    label: `bioemu2.cond-silu:${normalised.elements}`,
  });
  execution.release(normalised);
  const cond = linear(execution, {
    x: activated, weight: w.cond2Weight, bias: w.cond2Bias, rows,
    inChannels: conditioning.elements / rows, outChannels: 6 * channels,
    weightLayout: "out-major",
  });
  execution.release(activated);
  const piece = (index) => {
    const out = execution.allocate(rows * channels, `bioemu2.cond.${index}`);
    execution.dispatch({
      source: sliceSource(rows, 6 * channels, index * channels, channels),
      output: out,
      inputs: { xTex: cond },
      label: `bioemu2.cond-slice:${rows}x${channels}:${index}`,
    });
    return out;
  };

  const modulated = (alphaIndex, betaIndex, input) => {
    const alpha = piece(alphaIndex);
    const beta = piece(betaIndex);
    const out = execution.allocate(rows * channels, "bioemu2.adaln");
    execution.dispatch({
      source: adaLnSource(rows, channels),
      output: out,
      inputs: { xTex: input, alphaTex: alpha, betaTex: beta },
      floats: { epsilon },
      label: `bioemu2.adaln:${rows}x${channels}`,
    });
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
  const a = linear(execution, {
    x, weight: w.fc1, rows, inChannels: channels, outChannels: hidden,
    weightLayout: "out-major",
  });
  const b = linear(execution, {
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
  const gated = execution.allocate(rows * hidden, "bioemu2.transition-gated");
  execution.dispatch({
    source: productSource(rows * hidden),
    output: gated,
    inputs: { aTex: activated, bTex: b },
    label: `bioemu2.transition-gate:${rows}x${hidden}`,
  });
  execution.release(activated);
  execution.release(b);
  const projected = linear(execution, {
    x: gated, weight: w.fc3, rows, inChannels: hidden, outChannels: out,
    weightLayout: "out-major",
  });
  execution.release(gated);
  return projected;
}

/** The attention's q, its bias-carrying query, and its gate. */
function projections(execution, { x, rows, channels, weights: w }) {
  const qkv = linear(execution, {
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
  const rawGate = linear(execution, {
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
export const addRowBiasSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
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
  const { qkv, query, gate } = projections(execution, {
    x, rows: tokens, channels, weights: w,
  });
  const slice = (from, label) => {
    const out = execution.allocate(tokens * channels, `bioemu2.${label}`);
    execution.dispatch({
      source: sliceSource(tokens, 3 * channels, from, channels),
      output: out,
      inputs: { xTex: qkv },
      label: `bioemu2.slice:${tokens}x${channels}:${label}`,
    });
    return out;
  };
  const key = slice(channels, "key");
  const value = slice(2 * channels, "value");
  execution.release(qkv);
  const bias = pairBias(execution, {
    pair, rows: tokens * tokens, channels: pair.elements / (tokens * tokens),
    heads, weights: w, epsilon,
  });
  const attended = execution.allocate(tokens * channels, "bioemu2.token-attended");
  execution.dispatch({
    source: fullAttentionSource(tokens, channels, heads),
    output: attended,
    inputs: { qTex: query, kTex: key, vTex: value, biasTex: bias, padTex: pad },
    label: `bioemu2.token-attend:${tokens}x${channels}:${heads}`,
  });
  for (const tensor of [query, key, value, bias]) execution.release(tensor);
  const gated = execution.allocate(tokens * channels, "bioemu2.token-gated");
  execution.dispatch({
    source: productSource(tokens * channels),
    output: gated,
    inputs: { aTex: gate, bTex: attended },
    label: `bioemu2.token-gate:${tokens}x${channels}`,
  });
  execution.release(attended);
  execution.release(gate);
  const out = linear(execution, {
    x: gated, weight: w.projO, rows: tokens, inChannels: channels,
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
  const { qkv, query, gate } = projections(execution, {
    x, rows: atoms, channels, weights: w,
  });
  // 🔴 THE KEY AND VALUE TRAVEL TOGETHER, because the kernel reads them as
  // per-head halves of this one `2 * channels` slice - see the note at the top.
  const kv = execution.allocate(atoms * 2 * channels, "bioemu2.kv");
  execution.dispatch({
    source: sliceSource(atoms, 3 * channels, channels, 2 * channels),
    output: kv,
    inputs: { xTex: qkv },
    label: `bioemu2.slice-kv:${atoms}x${2 * channels}`,
  });
  execution.release(qkv);
  const windows = atoms / WINDOW_QUERIES;
  const bias = given ?? pairBias(execution, {
    pair, rows: windows * WINDOW_QUERIES * WINDOW_KEYS,
    channels: pair.elements / (windows * WINDOW_QUERIES * WINDOW_KEYS),
    heads, weights: w, epsilon,
  });
  const attended = execution.allocate(atoms * channels, "bioemu2.atom-attended");
  execution.dispatch({
    source: windowAttentionSource(atoms, channels, heads),
    output: attended,
    inputs: { qTex: query, kvTex: kv, biasTex: bias, padTex: pad },
    label: `bioemu2.atom-attend:${atoms}x${channels}:${heads}`,
  });
  for (const tensor of [query, kv]) execution.release(tensor);
  if (bias !== given) execution.release(bias);
  const gated = execution.allocate(atoms * channels, "bioemu2.atom-gated");
  execution.dispatch({
    source: productSource(atoms * channels),
    output: gated,
    inputs: { aTex: gate, bTex: attended },
    label: `bioemu2.atom-gate:${atoms}x${channels}`,
  });
  execution.release(attended);
  execution.release(gate);
  const out = linear(execution, {
    x: gated, weight: w.projO, rows: atoms, inChannels: channels,
    outChannels: channels, weightLayout: "out-major",
  });
  execution.release(gated);
  return out;
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
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, sigma: number, features: any, context: any,
 *   sInputs: Tensor, sTrunk: Tensor, weights: any, dims: any,
 *   hostFourier: {weight: Float32Array, bias: Float32Array},
 *   epsilon?: number, onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {Tensor} `atoms * 3`; the caller's to release
 */
export function bioemu2Denoise(execution, call) {
  const {
    x, sigma, features: f, context, sInputs, sTrunk, weights: w, dims,
    epsilon = 1e-5,
  } = call;
  const { atoms, tokens, layers } = dims;
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
  const scaled = keep(execution.allocate(atoms * 3, "bioemu2.r"));
  execution.dispatch({
    source: scaleSource(atoms * 3),
    output: scaled,
    inputs: { xTex: x },
    floats: { factor: cIn },
    label: `bioemu2.c-in:${atoms}`,
  });
  const r = keep(execution.allocate(atoms * 3, "bioemu2.r-masked"));
  execution.dispatch({
    source: maskRowsSource(atoms, 3),
    output: r,
    inputs: { xTex: scaled, maskTex: f.atomMask },
    label: `bioemu2.r-mask:${atoms}`,
  });
  call.onStage?.("r", r);

  // The single conditioning: [s_trunk | s_inputs], then the Fourier term.
  const trunkChannels = sTrunk.elements / tokens;
  const inputChannels = sInputs.elements / tokens;
  const joined = keep(execution.allocate(
    tokens * (trunkChannels + inputChannels), "bioemu2.single-cat"));
  execution.dispatch({
    source: concatSource(tokens, trunkChannels, inputChannels),
    output: joined,
    inputs: { leftTex: sTrunk, rightTex: sInputs },
    label: `bioemu2.single-cat:${tokens}x${trunkChannels + inputChannels}`,
  });
  const normalisedSingle = keep(layerNorm(execution, {
    x: joined, weight: w.singleConditioner.normScale,
    bias: w.singleConditioner.normOffset,
    rows: tokens, channels: trunkChannels + inputChannels, epsilon,
  }));
  let cond = keep(linear(execution, {
    x: normalisedSingle, weight: w.singleConditioner.singleEmbedWeight,
    bias: w.singleConditioner.singleEmbedBias, rows: tokens,
    inChannels: trunkChannels + inputChannels, outChannels: tokenChannels,
    weightLayout: "out-major",
  }));
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
  const withFourier = keep(execution.allocate(
    tokens * tokenChannels, "bioemu2.cond"));
  execution.dispatch({
    source: broadcastRowSource(tokens, tokenChannels),
    output: withFourier,
    inputs: { xTex: cond, rowTex: fourierTerm },
    label: `bioemu2.fourier-add:${tokens}x${tokenChannels}`,
  });
  cond = withFourier;
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

  // The two streams' initial states.
  const rToQ = keep(linear(execution, {
    x: r, weight: w.rToQ, rows: atoms, inChannels: 3, outChannels: atomChannels,
    weightLayout: "out-major",
  }));
  let q = keep(execution.allocate(atoms * atomChannels, "bioemu2.q"));
  execution.dispatch({
    source: sumSource(atoms * atomChannels),
    output: q,
    inputs: { aTex: context.q0, bTex: rToQ },
    label: `bioemu2.q0:${atoms}x${atomChannels}`,
  });
  const normalisedTrunk = keep(layerNorm(execution, {
    x: sTrunk, weight: w.sTrunkToA0Scale, bias: w.sTrunkToA0Offset,
    rows: tokens, channels: trunkChannels, epsilon,
  }));
  const aBase = keep(linear(execution, {
    x: normalisedTrunk, weight: w.sTrunkToA1, rows: tokens,
    inChannels: trunkChannels, outChannels: tokenChannels, weightLayout: "out-major",
  }));
  // ...plus the scaled input AT EVERY TOKEN'S CENTRE ATOM, masked by the token pad.
  const centreR = keep(execution.allocate(tokens * 3, "bioemu2.centre-r"));
  execution.dispatch({
    source: gatherRowsSource(tokens, 3),
    output: centreR,
    inputs: { sourceTex: r, indexTex: f.centreAtom },
    label: `bioemu2.centre-r:${tokens}`,
  });
  const maskedCentre = keep(execution.allocate(tokens * 3, "bioemu2.centre-masked"));
  execution.dispatch({
    source: maskRowsSource(tokens, 3),
    output: maskedCentre,
    inputs: { xTex: centreR, maskTex: f.tokenPad },
    label: `bioemu2.centre-mask:${tokens}`,
  });
  const rToA = keep(linear(execution, {
    x: maskedCentre, weight: w.rTokenToA, rows: tokens, inChannels: 3,
    outChannels: tokenChannels, weightLayout: "out-major",
  }));
  let a = keep(execution.allocate(tokens * tokenChannels, "bioemu2.a"));
  execution.dispatch({
    source: sumSource(tokens * tokenChannels),
    output: a,
    inputs: { aTex: aBase, bTex: rToA },
    label: `bioemu2.a0:${tokens}x${tokenChannels}`,
  });
  // The conditioning at every atom, through its token - a padded atom gets zeros.
  const condAtAtom = keep(execution.allocate(
    atoms * tokenChannels, "bioemu2.cond-at-atom"));
  execution.dispatch({
    source: gatherRowsSource(atoms, tokenChannels),
    output: condAtAtom,
    inputs: { sourceTex: cond, indexTex: f.atomToTokenOrPad },
    label: `bioemu2.cond-at-atom:${atoms}x${tokenChannels}`,
  });
  call.onStage?.("q0", q);
  call.onStage?.("a0", a);

  for (let layer = 0; layer < layers; layer += 1) {
    const atomWeights = w.atomBlocks[layer];
    const tokenWeights = w.tokenBlocks[layer];
    const nextQ = conditionedLayer(execution, {
      x: q, conditioning: condAtAtom, rows: atoms, channels: atomChannels,
      heads: dims.atomHeads, weights: atomWeights, epsilon,
      attend: (input) => windowAttention(execution, {
        x: input, pair: context.p, pad: f.atomPad, atoms, channels: atomChannels,
        heads: dims.atomHeads, weights: atomWeights.attention, epsilon,
      }),
    });
    const nextA = conditionedLayer(execution, {
      x: a, conditioning: cond, rows: tokens, channels: tokenChannels,
      heads: dims.tokenHeads, weights: tokenWeights, epsilon,
      attend: (input) => fullAttention(execution, {
        x: input, pair: context.z, pad: f.tokenPad, tokens, channels: tokenChannels,
        heads: dims.tokenHeads, weights: tokenWeights.attention, epsilon,
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
    const mix = (condSource, rows, gatherIndex, left, right, weights, outChannels) => {
      const normalised = layerNorm(execution, {
        x: condSource, weight: weights.cond0Scale, bias: weights.cond0Offset,
        rows, channels: condSource.elements / rows, epsilon,
      });
      const activated = execution.allocate(normalised.elements, "bioemu2.mix-silu");
      execution.dispatch({
        source: siluSource(normalised.elements),
        output: activated,
        inputs: { xTex: normalised },
        label: `bioemu2.mix-silu:${normalised.elements}`,
      });
      execution.release(normalised);
      const projected = linear(execution, {
        x: activated, weight: weights.cond2Weight, bias: weights.cond2Bias, rows,
        inChannels: condSource.elements / rows,
        outChannels: 2 * width + outChannels, weightLayout: "out-major",
      });
      execution.release(activated);
      const slice = (from, howWide, label) => {
        const out = execution.allocate(rows * howWide, `bioemu2.mix.${label}`);
        execution.dispatch({
          source: sliceSource(rows, 2 * width + outChannels, from, howWide),
          output: out,
          inputs: { xTex: projected },
          label: `bioemu2.mix-slice:${rows}x${howWide}:${label}`,
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
      const alpha = slice(0, width, "alpha");
      const beta = slice(width, width, "beta");
      const modulated = execution.allocate(rows * width, "bioemu2.mix-adaln");
      execution.dispatch({
        source: adaLnSource(rows, width),
        output: modulated,
        inputs: { xTex: joinedMix, alphaTex: alpha, betaTex: beta },
        floats: { epsilon },
        label: `bioemu2.mix-adaln:${rows}x${width}`,
      });
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
      return out;
    };
    const sToQ = mix(condAtAtom, atoms, f.atomToTokenOrPad,
      { tensor: a, channels: tokenChannels }, { tensor: q, channels: atomChannels },
      { ...coupling.qCoupling, fc1: coupling.sToQ.fc1, fc2: coupling.sToQ.fc2,
        fc3: coupling.sToQ.fc3 },
      atomChannels);
    const qToS = mix(cond, tokens, f.centreAtomOrPad,
      { tensor: q, channels: atomChannels }, { tensor: a, channels: tokenChannels },
      { ...coupling.sCoupling, fc1: coupling.qToS.fc1, fc2: coupling.qToS.fc2,
        fc3: coupling.qToS.fc3 },
      tokenChannels);
    const mixedQ = execution.allocate(atoms * atomChannels, "bioemu2.q");
    execution.dispatch({
      source: sumSource(atoms * atomChannels),
      output: mixedQ,
      inputs: { aTex: q, bTex: sToQ },
      label: `bioemu2.q-couple:${atoms}x${atomChannels}`,
    });
    const mixedA = execution.allocate(tokens * tokenChannels, "bioemu2.a");
    execution.dispatch({
      source: sumSource(tokens * tokenChannels),
      output: mixedA,
      inputs: { aTex: a, bTex: qToS },
      label: `bioemu2.a-couple:${tokens}x${tokenChannels}`,
    });
    execution.release(sToQ);
    execution.release(qToS);
    execution.release(q);
    execution.release(a);
    q = mixedQ;
    a = mixedA;
    call.onStage?.(`layer${layer}.q`, q);
    call.onStage?.(`layer${layer}.a`, a);
  }

  // The readout: the atom stream plus the token stream at every atom.
  const hidden = w.readout.qToHidden.elements / atomChannels;
  const fromQ = linear(execution, {
    x: q, weight: w.readout.qToHidden, rows: atoms, inChannels: atomChannels,
    outChannels: hidden, weightLayout: "out-major",
  });
  const fromA = linear(execution, {
    x: a, weight: w.readout.sToHidden, rows: tokens, inChannels: tokenChannels,
    outChannels: hidden, weightLayout: "out-major",
  });
  const atAtom = execution.allocate(atoms * hidden, "bioemu2.readout-at-atom");
  execution.dispatch({
    source: gatherRowsSource(atoms, hidden),
    output: atAtom,
    inputs: { sourceTex: fromA, indexTex: f.atomToTokenOrPad },
    label: `bioemu2.readout-gather:${atoms}x${hidden}`,
  });
  execution.release(fromA);
  const summed = execution.allocate(atoms * hidden, "bioemu2.readout");
  execution.dispatch({
    source: sumSource(atoms * hidden),
    output: summed,
    inputs: { aTex: fromQ, bTex: atAtom },
    label: `bioemu2.readout-add:${atoms}x${hidden}`,
  });
  execution.release(fromQ);
  execution.release(atAtom);
  const normalised = layerNorm(execution, {
    x: summed, weight: w.readout.outNormScale, rows: atoms, channels: hidden, epsilon,
  });
  execution.release(summed);
  const update = linear(execution, {
    x: normalised, weight: w.readout.toOut, rows: atoms, inChannels: hidden,
    outChannels: 3, weightLayout: "out-major",
  });
  execution.release(normalised);
  call.onStage?.("readout", update);

  // 🔴 AND THE ANSWER IS THE EDM BLEND, not the update: c_skip * x + c_out * out.
  const cSkip = (sigmaData * sigmaData) / (sigma * sigma + sigmaData * sigmaData);
  const cOut = (sigma * sigmaData) / Math.sqrt(sigmaData * sigmaData + sigma * sigma);
  const out = execution.allocate(atoms * 3, "bioemu2.denoised");
  execution.dispatch({
    source: blendSource(atoms * 3),
    output: out,
    inputs: { aTex: x, bTex: update },
    floats: { keep: cSkip, take: cOut },
    label: `bioemu2.blend:${atoms}`,
  });
  execution.release(update);
  execution.release(q);
  execution.release(a);
  for (const tensor of temporary) execution.release(tensor);
  out.label = "bioemu2.denoised";
  return out;
}

/** `out[row][c] = x[row][c] + row0[c]`. */
export const broadcastRowSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
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
export const blendSource = (elements) => `${PRELUDE}${ONE_TARGET}
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
