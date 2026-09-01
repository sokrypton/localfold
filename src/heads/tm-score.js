/**
 * The pTM reduction, shared by AlphaFold 2 and AlphaFold 3.
 *
 *     score = max over anchors i of  mean over selected j of  E[tm(PAE_ij)]
 *
 * Both models compute exactly this, and the parts that differ between them are
 * the arguments rather than the arithmetic: where the bin centres come from,
 * what d0 is drawn from, which pairs are selected, and how chains are
 * identified. Those stay in each model's own file; only the reduction lives
 * here, because a second copy of it is a second chance to disagree about it.
 *
 * 🔴 IT IS A MAXIMUM OVER ANCHORS, NOT A MEAN. TM-score is defined against the
 * BEST alignment frame - the token from whose frame everything else superposes
 * best - so averaging the rows instead produces a lower number that still looks
 * like a plausible confidence and is not this quantity.
 *
 * 🔴 AND THE MEAN IS OVER THE SELECTED PAIRS ONLY. For ipTM the row mean is
 * taken over the cross-chain pairs alone, so an anchor deep inside one chain
 * is scored on how well it places the OTHER chain and not on its own
 * neighbourhood, which is the whole point of the interface score.
 */

/**
 * The reduction over an already-computed per-pair TM term.
 *
 * Split out because the GPU confidence head evaluates the term in its own
 * shader - it already holds the PAE logits there, and `tokens^2` floats come
 * back where `tokens^2 * 64` logits would not - so both paths share the part
 * that decides what the score MEANS, and differ only in where the expectation
 * was taken.
 *
 * @param {ArrayLike<number>} term  tokens * tokens
 * @param {(anchor: number, other: number) => boolean} selects
 * @returns {number} the score, or NaN when nothing is selected
 */

export function reduceTmScore(term, tokens, selects) {
  let best = Number.NEGATIVE_INFINITY;
  let anyAnchor = false;
  for (let anchor = 0; anchor < tokens; anchor += 1) {
    let total = 0;
    let count = 0;
    for (let other = 0; other < tokens; other += 1) {
      if (!selects(anchor, other)) continue;
      total += term[anchor * tokens + other];
      count += 1;
    }
    if (count === 0) continue;
    anyAnchor = true;
    const alignment = total / count;
    if (alignment > best) best = alignment;
  }
  return anyAnchor ? best : Number.NaN;
}

/**
 * The same score, taking the PAE logits and doing the expectation here.
 *
 * @param {ArrayLike<number>} logits  tokens * tokens * bins, the PAE logits
 * @param {number} tokens
 * @param {ArrayLike<number>} tmPerBin  bins; `1 / (1 + centre^2 / d0^2)` already
 *   evaluated, because both callers know their own centres and d0
 * @param {(anchor: number, other: number) => boolean} selects
 * @returns {number} the score, or NaN when `selects` admits nothing for every
 *   anchor - which is what a monomer's ipTM is, and is not zero
 */
export function bestAlignmentTmScore(logits, tokens, tmPerBin, selects) {
  const bins = tmPerBin.length;
  if (logits.length !== tokens * tokens * bins) {
    throw new RangeError(`PAE logits should be ${tokens * tokens * bins} long,`
      + ` not ${logits.length}`);
  }
  let best = Number.NEGATIVE_INFINITY;
  let anyAnchor = false;
  for (let anchor = 0; anchor < tokens; anchor += 1) {
    let total = 0;
    let count = 0;
    for (let other = 0; other < tokens; other += 1) {
      if (!selects(anchor, other)) continue;
      const base = (anchor * tokens + other) * bins;
      // Softmax and expectation in one pass, against the largest logit, because
      // the logits are unbounded and exp() of them is not.
      let largest = Number.NEGATIVE_INFINITY;
      for (let bin = 0; bin < bins; bin += 1) {
        if (logits[base + bin] > largest) largest = logits[base + bin];
      }
      let denominator = 0;
      let numerator = 0;
      for (let bin = 0; bin < bins; bin += 1) {
        const probability = Math.exp(logits[base + bin] - largest);
        denominator += probability;
        numerator += probability * tmPerBin[bin];
      }
      total += numerator / denominator;
      count += 1;
    }
    if (count === 0) continue;
    anyAnchor = true;
    const alignment = total / count;
    if (alignment > best) best = alignment;
  }
  return anyAnchor ? best : Number.NaN;
}

/**
 * TM-score's d0: the distance at which a residue counts for half.
 *
 * 🔴 CLIPPED AT 19 RESIDUES, NOT AT 15. Below 15 the cube root of a negative
 * number is undefined and at 15 exactly d0 is -1.8, so the clip must sit above
 * it. Both AlphaFold 2 and AlphaFold 3 use 19, which puts d0 at about 0.14 A.
 */
export function tmScoreD0(tokenCount) {
  return 1.24 * Math.cbrt(Math.max(tokenCount, 19) - 15) - 1.8;
}

/** `1 / (1 + centre^2 / d0^2)` for each bin: the TM term the PAE is scored by. */
export function tmPerBinFor(binCentres, d0) {
  const d0Squared = d0 * d0;
  const output = new Float64Array(binCentres.length);
  for (let bin = 0; bin < binCentres.length; bin += 1) {
    output[bin] = 1 / (1 + (binCentres[bin] * binCentres[bin]) / d0Squared);
  }
  return output;
}
