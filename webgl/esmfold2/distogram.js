/**
 * ESMFold2's distogram head: the trunk's pair representation, symmetrised.
 *
 *     logits = Linear(z[i][j] + z[j][i])        256 -> 128, with a bias
 *
 * 🔴 THE SYMMETRISATION IS PART OF THE HEAD, NOT A CONVENIENCE. Upstream writes
 * `distogram_head(z + z.transpose(-2, -3))`, so the head sees each pair added to
 * its transpose - a distance is symmetric and the trunk's pair is not. "Feeding
 * it `z` alone conforms in shape, produces a plausible distogram, and is a
 * different model."
 *
 * 🔴 AND THE BIN EDGES ARE NOT IN THIS CHECKPOINT'S CONFIG. `distogram_bins` is
 * 128 and nothing states the range; the CONFIDENCE head's config carries
 * `min_dist` 2.0 and `max_dist` 52.0 for its own 128, and assuming the trunk's
 * are the same would be a guess presented as a fact. So this returns LOGITS,
 * which is what can be checked against the model, and leaves turning them into
 * distances to a caller that has the edges from somewhere real.
 */
import { ONE_TARGET, PRELUDE, memoSource } from "../runtime/execution.js";
import { linear } from "../kernels/linear.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/** `out[i][j][c] = pair[i][j][c] + pair[j][i][c]`. */
const symmetriseSourceUncached = (tokens, channels) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int C4 = ${channels / 4};
const int ELEMENTS = ${tokens * tokens * channels};
uniform sampler2D pairTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  result = fetch4(pairTex, (i * L + j) * C4 + group)
    + fetch4(pairTex, (j * L + i) * C4 + group);
}`;

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, tokens: number, channels: number, bins: number,
 *   weights: Tensor, bias?: Tensor, symmetrise?: boolean,
 *   weightLayout?: "in-major" | "out-major"}} call
 *   🔴 `weightLayout` IS A PARAMETER BECAUSE BIOEMU2 READS THIS HEAD TOO and its
 *   checkpoint is OUT-major throughout where ESMFold2's is in-major - the same
 *   `[64, 256]` against `[256, 128]` the rest of that port turns on. A head given
 *   the wrong layout returns a tensor of the right shape and a distogram that is
 *   a transpose of the arithmetic asked for, so the default is the caller this
 *   file is named for and the other one says so.
 * @returns {Tensor} `tokens^2 * bins` of logits
 */
export function distogramLogits(execution, call) {
  const {
    pair, tokens, channels, bins, weights, bias, symmetrise = true,
    weightLayout = "in-major",
  } = call;
  let source = pair;
  if (symmetrise) {
    source = execution.allocate(tokens * tokens * channels, "ef2.distogram.symmetric");
    execution.dispatch({
      source: symmetriseSource(tokens, channels),
      output: source,
      inputs: { pairTex: pair },
      label: `ef2.distogram.symmetrise:${tokens}x${channels}`,
    });
  }
  const logits = linear(execution, {
    x: source, weight: weights, ...(bias === undefined ? {} : { bias }),
    rows: tokens * tokens, inChannels: channels, outChannels: bins,
    weightLayout,
  });
  if (source !== pair) execution.release(source);
  logits.label = "ef2.distogram.logits";
  return logits;
}

/**
 * Softmax over the bins, so a caller can read a contact probability.
 *
 * 🔴 OVER THE BINS, WHICH IS THE LAST AXIS: "the logits are (i, j, bin) and a
 * softmax over the wrong one still sums to one somewhere".
 */
const probabilitiesSourceUncached = (pairs, bins) => `${PRELUDE}${ONE_TARGET}
const int BINS = ${bins};
const int B4 = ${bins / 4};
const int ELEMENTS = ${pairs * bins};
uniform sampler2D logitsTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % B4;
  int pair = texel / B4;
  int base = pair * BINS;
  float largest = -1e30;
  for (int b = 0; b < BINS; ++b) largest = max(largest, fetch1(logitsTex, base + b));
  float total = 0.0;
  for (int b = 0; b < BINS; ++b) total += exp(fetch1(logitsTex, base + b) - largest);
  result = exp(fetch4(logitsTex, texel) - vec4(largest)) / vec4(total);
}`;

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{logits: Tensor, pairs: number, bins: number}} call
 * @returns {Tensor} `pairs * bins`
 */
export function distogramProbabilities(execution, { logits, pairs, bins }) {
  const out = execution.allocate(pairs * bins, "ef2.distogram.probabilities");
  execution.dispatch({
    source: probabilitiesSource(pairs, bins),
    output: out,
    inputs: { logitsTex: logits },
    label: `ef2.distogram.probabilities:${pairs}x${bins}`,
  });
  return out;
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const symmetriseSource = memoSource(symmetriseSourceUncached);
export const probabilitiesSource = memoSource(probabilitiesSourceUncached);
