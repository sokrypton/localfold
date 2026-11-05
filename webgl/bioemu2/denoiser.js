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

/** `out[row] = [left[row] | right[row]]`, two widths that need not match. */
export const concatSource = (rows, leftWidth, rightWidth) => `${PRELUDE}${ONE_TARGET}
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
}`;

/** `out[row][c] = index[row] < 0 ? 0 : source[index[row]][c]`. */
export const gatherRowsSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
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
}`;

/** `out[row][c] = x[row][c] * mask[row]`. */
export const maskRowsSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel) * fetch1(maskTex, texel / C4);
}`;

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
export const pairBiasSource = (rows, channels, heads) => `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
const int C4 = ${channels / 4};
const int HEADS = ${heads};
const int ELEMENTS = ${rows * heads};
uniform sampler2D pairTex;
uniform sampler2D scaleTex;
uniform sampler2D projectTex;
uniform float epsilon;

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

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`;

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
