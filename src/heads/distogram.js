/**
 * AlphaFold 2's distogram head, and the contact map that falls out of it.
 *
 * WHAT IT IS. One linear projection of the pair representation to 64 distance
 * bins, symmetrised:
 *
 *     half   = pair @ W + b          (L, L, 64)
 *     logits = half + half^T         transposed over the PAIR axes only
 *
 * The bins are 63 breaks evenly spaced from 2 to 22 A, so bin b covers
 * [breaks[b-1], breaks[b]) with the first open below 2 and the last open above
 * 22. That is the whole head; AlphaFold has always had it, and it is what the
 * contact map in every AlphaFold figure is drawn from.
 *
 * 🔴 THE SYMMETRISATION IS OVER i AND j, NOT OVER THE BINS. `half` is not
 * symmetric - the projection sees pair[i][j], which differs from pair[j][i] -
 * and the head adds the transpose so the distance between i and j is one
 * number however it is asked for. Transposing the bin axis as well would mix
 * distances that have nothing to do with each other, and the result would
 * still look like a plausible distogram.
 *
 * 🔴 AND A CONTACT IS P(d < 8 A), WHICH IS A SUM OVER BINS AND NOT A BIN. AF3
 * computes the same quantity in its own distogram head and this repo already
 * reads it back there, so the two models' contact maps mean the same thing:
 * the model's own probability that two residues are within 8 angstrom. Not a
 * measurement of the structure it produced - that is a different question, and
 * a contact map drawn from coordinates would answer it while looking
 * identical.
 */

/** The bin edges: `count - 1` breaks, evenly spaced. */
export function distogramBreaks(first, last, count) {
  const breaks = new Float64Array(count - 1);
  for (let index = 0; index < count - 1; index += 1) {
    breaks[index] = first + ((last - first) * index) / (count - 2);
  }
  return breaks;
}

/**
 * P(d < threshold) for every pair, from the pair representation.
 *
 * 🔴 IT NEVER MATERIALISES THE LOGITS. A distogram is `L * L * 64` floats -
 * 37 MB at 384 residues - and every caller here wants the contact map rather
 * than the bins, so the softmax and the sum run per pair in registers and only
 * `L * L` floats come out. The logits are available from `distogramLogits`
 * when something really needs them.
 *
 * @param {ArrayLike<number>} pair  L * L * channels
 * @param {ArrayLike<number>} weights  channels * bins, the half_logits weights
 * @param {ArrayLike<number>} bias  bins
 * @param {number} length  L
 * @param {{channels?: number, bins?: number, first?: number, last?: number,
 *          threshold?: number}} [options]
 * @returns {Float32Array} L * L, each in [0, 1]
 */
export function distogramContactProbabilities(pair, weights, bias, length, options = {}) {
  const channels = options.channels ?? 128;
  const bins = options.bins ?? 64;
  const threshold = options.threshold ?? 8;
  const breaks = distogramBreaks(options.first ?? 2, options.last ?? 22, bins);
  if (pair.length !== length * length * channels) {
    throw new RangeError(`pair is ${pair.length}; expected ${length * length * channels}`);
  }
  // 🔴 A BIN COUNTS WHEN ITS UPPER EDGE IS AT OR BELOW THE THRESHOLD. Bin b
  // spans up to breaks[b], so the bins strictly under 8 A are those with
  // `breaks[b] <= threshold`; the bin straddling the threshold is excluded,
  // which is what AlphaFold's own contact figures do.
  let counted = 0;
  while (counted < breaks.length && breaks[counted] <= threshold) counted += 1;

  // `half` for one row at a time, so the transpose can be taken without
  // holding the whole distogram: computing half[i][j] and half[j][i] as
  // needed costs the projection twice and saves L*L*64 floats.
  const half = (i, j, out) => {
    const base = (i * length + j) * channels;
    for (let bin = 0; bin < bins; bin += 1) out[bin] = bias[bin];
    for (let channel = 0; channel < channels; channel += 1) {
      const value = pair[base + channel];
      if (value === 0) continue;
      const row = channel * bins;
      for (let bin = 0; bin < bins; bin += 1) out[bin] += value * weights[row + bin];
    }
    return out;
  };

  const forward = new Float64Array(bins);
  const backward = new Float64Array(bins);
  const logits = new Float64Array(bins);
  const contacts = new Float32Array(length * length);
  for (let i = 0; i < length; i += 1) {
    for (let j = i; j < length; j += 1) {
      half(i, j, forward);
      half(j, i, backward);
      let largest = Number.NEGATIVE_INFINITY;
      for (let bin = 0; bin < bins; bin += 1) {
        logits[bin] = forward[bin] + backward[bin];
        if (logits[bin] > largest) largest = logits[bin];
      }
      let total = 0;
      let under = 0;
      for (let bin = 0; bin < bins; bin += 1) {
        const weight = Math.exp(logits[bin] - largest);
        total += weight;
        if (bin < counted) under += weight;
      }
      const probability = total === 0 ? 0 : under / total;
      contacts[i * length + j] = probability;
      contacts[j * length + i] = probability;
    }
  }
  return contacts;
}

/**
 * The full symmetrised logits, for a checker that wants the bins themselves.
 *
 * Deliberately separate: it allocates `L * L * bins`, which is the thing
 * distogramContactProbabilities exists to avoid.
 */
export function distogramLogits(pair, weights, bias, length, options = {}) {
  const channels = options.channels ?? 128;
  const bins = options.bins ?? 64;
  const half = new Float64Array(length * length * bins);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      const base = (i * length + j) * channels;
      const out = (i * length + j) * bins;
      for (let bin = 0; bin < bins; bin += 1) half[out + bin] = bias[bin];
      for (let channel = 0; channel < channels; channel += 1) {
        const value = pair[base + channel];
        if (value === 0) continue;
        const row = channel * bins;
        for (let bin = 0; bin < bins; bin += 1) half[out + bin] += value * weights[row + bin];
      }
    }
  }
  const logits = new Float32Array(length * length * bins);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      const a = (i * length + j) * bins;
      const b = (j * length + i) * bins;
      for (let bin = 0; bin < bins; bin += 1) logits[a + bin] = half[a + bin] + half[b + bin];
    }
  }
  return logits;
}
