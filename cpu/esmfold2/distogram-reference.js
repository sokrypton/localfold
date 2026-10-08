/**
 * ESMFold2's distogram head: the trunk's pair representation, symmetrised.
 *
 *     logits = Linear(z[i][j] + z[j][i])        256 -> 128, with a bias
 *
 * 🔴 THE SYMMETRISATION IS PART OF THE HEAD, NOT A CONVENIENCE. Upstream writes
 * `distogram_head(z + z.transpose(-2, -3))`, so the head sees each pair added to
 * its transpose - a distance is symmetric and the trunk's pair is not. Feeding
 * it `z` alone conforms in shape, produces a plausible distogram, and is a
 * different model.
 *
 * 🔴 AND THE BIN EDGES ARE NOT IN THIS CHECKPOINT'S CONFIG. `distogram_bins` is
 * 128 and nothing states the range; the CONFIDENCE head's config carries
 * `min_dist` 2.0 and `max_dist` 52.0 for its own 128, and assuming the trunk's
 * are the same would be a guess presented as a fact. So this returns LOGITS,
 * which is what can be checked against the model, and leaves turning them into
 * distances to a caller that has the edges from somewhere real.
 */

/** rows x inChannels against (inChannels, outChannels), plus a bias. */
function linear(input, rows, inChannels, outChannels, weights, bias) {
  const out = new Float32Array(rows * outChannels);
  for (let row = 0; row < rows; row += 1) {
    const inBase = row * inChannels;
    const outBase = row * outChannels;
    if (bias !== undefined) {
      for (let o = 0; o < outChannels; o += 1) out[outBase + o] = bias[o];
    }
    for (let c = 0; c < inChannels; c += 1) {
      const value = input[inBase + c];
      if (value === 0) continue;
      const weightBase = c * outChannels;
      for (let o = 0; o < outChannels; o += 1) out[outBase + o] += value * weights[weightBase + o];
    }
  }
  return out;
}

/**
 * @param {Float32Array} pair n*n*channels, the trunk's final output
 * @returns {Float32Array} n*n*bins of logits
 */
export function distogramLogits(pair, n, channels, weights, bias, bins,
                                symmetrise = true) {
  if (!symmetrise) return linear(pair, n * n, channels, bins, weights, bias);
  const symmetric = new Float32Array(pair.length);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const forward = (i * n + j) * channels;
      const backward = (j * n + i) * channels;
      for (let c = 0; c < channels; c += 1) {
        symmetric[forward + c] = pair[forward + c] + pair[backward + c];
      }
    }
  }
  return linear(symmetric, n * n, channels, bins, weights, bias);
}

/**
 * Softmax over the bins, so a caller can read a contact probability.
 *
 * 🔴 OVER THE BINS, WHICH IS THE LAST AXIS. The logits are (i, j, bin) and a
 * softmax over the wrong one still sums to one somewhere.
 */
export function distogramProbabilities(logits, pairs, bins) {
  const out = new Float32Array(logits.length);
  for (let pair = 0; pair < pairs; pair += 1) {
    const base = pair * bins;
    let largest = -Infinity;
    for (let b = 0; b < bins; b += 1) largest = Math.max(largest, logits[base + b]);
    let total = 0;
    for (let b = 0; b < bins; b += 1) {
      out[base + b] = Math.exp(logits[base + b] - largest);
      total += out[base + b];
    }
    for (let b = 0; b < bins; b += 1) out[base + b] /= total;
  }
  return out;
}
