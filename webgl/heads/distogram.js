/**
 * P(d < threshold) for every pair - the contact map the page draws.
 *
 * `shared/heads/distogram.js` is the definition and `webgpu/heads/distogram.js`
 * is the WebGPU kernel; this is the same arithmetic in two draws.
 *
 *     logits[i][j][bin] = 2 * bias[bin]
 *                       + sum_c (pair[i][j][c] + pair[j][i][c]) * W[c][bin]
 *     contact[i][j]     = sum of softmax(logits[i][j])[b] for b < counted
 *
 * 🔴 THE PAIR IS SYMMETRISED AND THE BIAS COUNTED TWICE, which falls out of the
 * host's own formulation: it computes `half(i, j) + half(j, i)` and each `half`
 * starts from the bias. Writing `bias + sum_c (...)` instead would be a
 * plausible contact map that is wrong everywhere by one bias.
 *
 * 🔴 AND A BIN COUNTS WHEN ITS UPPER EDGE IS AT OR BELOW THE THRESHOLD, so the
 * bin straddling 8 A is EXCLUDED - `breaks[b] <= threshold`, which is what
 * AlphaFold's own contact figures do. Counting it would shift every number up.
 *
 * Two draws rather than one: a fragment writing four bins of one pair shares its
 * channel sweep across them, where a fragment owning a whole pair's softmax
 * would redo that sweep `bins / 4` times. The intermediate is `L^2 * bins`
 * floats, released as soon as the second draw is done.
 */
import { ONE_TARGET, PRELUDE, sized } from "../runtime/execution.js";
import { distogramBreaks, FIRST_BREAK, LAST_BREAK } from "../../shared/heads/distogram.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

export const distogramLogitsSource = (length, channels, bins) => sized({ L: length }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int B4 = ${bins / 4};
uniform sampler2D pairTex;
uniform sampler2D weightTex;
uniform sampler2D biasTex;

void main() {
  int texel = linearTexel();
  if (texel >= L * L * B4) { result = vec4(0.0); return; }
  int b4 = texel % B4;
  int pair = texel / B4;
  int j = pair % L;
  int i = pair / L;
  int forward = (i * L + j) * C4;
  int backward = (j * L + i) * C4;

  // Twice the bias: the host adds it once in each half.
  vec4 sum = 2.0 * fetch4(biasTex, b4);
  for (int c4 = 0; c4 < C4; ++c4) {
    vec4 symmetrised = fetch4(pairTex, forward + c4) + fetch4(pairTex, backward + c4);
    int w = (c4 * 4) * B4 + b4;
    sum += symmetrised.x * fetch4(weightTex, w);
    sum += symmetrised.y * fetch4(weightTex, w + B4);
    sum += symmetrised.z * fetch4(weightTex, w + 2 * B4);
    sum += symmetrised.w * fetch4(weightTex, w + 3 * B4);
  }
  result = sum;
}`);

export const distogramContactSource = (length, bins, counted) => sized({ PAIRS: length * length }, `${PRELUDE}${ONE_TARGET}
const int BINS = ${bins};
const int B4 = ${bins / 4};
const int COUNTED = ${counted};
uniform sampler2D logitsTex;

float contactOf(int pair) {
  int base = pair * BINS;
  float top = -1e30;
  for (int b = 0; b < BINS; ++b) top = max(top, fetch1(logitsTex, base + b));
  float total = 0.0;
  float under = 0.0;
  for (int b = 0; b < BINS; ++b) {
    float weight = exp(fetch1(logitsTex, base + b) - top);
    total += weight;
    if (b < COUNTED) under += weight;
  }
  return under / total;
}

void main() {
  int texel = linearTexel();
  int base = texel * 4;
  if (base >= PAIRS) { result = vec4(0.0); return; }
  result = vec4(
    contactOf(base),
    base + 1 < PAIRS ? contactOf(base + 1) : 0.0,
    base + 2 < PAIRS ? contactOf(base + 2) : 0.0,
    base + 3 < PAIRS ? contactOf(base + 3) : 0.0);
}`);

/** How many bins lie strictly under the threshold, by the host's rule. */
export function countedBins(bins, threshold = 8, first = FIRST_BREAK, last = LAST_BREAK) {
  const breaks = distogramBreaks(first, last, bins);
  let counted = 0;
  while (counted < breaks.length && breaks[counted] <= threshold) counted += 1;
  return counted;
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, weights: Tensor, bias: Tensor, length: number,
 *   channels: number, bins?: number, threshold?: number,
 *   firstBreak?: number, lastBreak?: number}} call
 * @returns {Tensor} `length^2` contact probabilities; the caller owns it
 */
export function distogramContacts(execution, call) {
  const {
    pair, weights, bias, length, channels, bins = 64, threshold = 8,
    firstBreak = FIRST_BREAK, lastBreak = LAST_BREAK,
  } = call;
  if (bins % 4 !== 0) throw new RangeError(`distogram: bins must be a multiple of 4, got ${bins}`);
  if (channels % 4 !== 0) {
    throw new RangeError(`distogram: channels must be a multiple of 4, got ${channels}`);
  }
  if (pair.elements !== length * length * channels) {
    throw new RangeError(
      `distogram: pair holds ${pair.elements} floats, not ${length}^2 x ${channels}`);
  }
  if (weights.elements !== channels * bins) {
    throw new RangeError(
      `distogram: weights hold ${weights.elements} floats, not ${channels} x ${bins}`);
  }
  const counted = countedBins(bins, threshold, firstBreak, lastBreak);

  const logits = execution.allocate(length * length * bins, "distogram.logits");
  execution.dispatch({
    source: distogramLogitsSource(length, channels, bins),
    output: logits,
    inputs: { pairTex: pair, weightTex: weights, biasTex: bias },
    label: `distogram.logits:${length}x${channels}x${bins}`,
  });
  const contacts = execution.allocate(length * length, "distogram.contacts");
  execution.dispatch({
    source: distogramContactSource(length, bins, counted),
    output: contacts,
    inputs: { logitsTex: logits },
    label: `distogram.contacts:${length}x${bins}:${counted}`,
  });
  execution.release(logits);
  return contacts;
}
