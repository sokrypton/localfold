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

/**
 * ipTM for each PAIR OF CHAINS separately, rather than pooled over all of them.
 *
 * 🔴 THE POOLED ipTM ANSWERS THE WRONG QUESTION ON MORE THAN TWO CHAINS. Its
 * selector is `asymId[i] !== asymId[j]`, so every cross-chain pair counts
 * equally - and in an assembly that holds both a native homodimer and a
 * designed binder, the homodimer's interface is the easy one and lifts the
 * score for the interface anyone is actually asking about. PDB 27UH is exactly
 * that shape: two S100A4 and two VHH, where the S100A4 pair is a real
 * biological dimer the model places well and each VHH is a de novo design.
 *
 * 🔴 IT IS THE SAME REDUCTION, ONLY THE SELECTION MOVES, which is the whole
 * reason reduceTmScore takes a predicate. ipTM already differs from pTM only in
 * what it selects; a per-interface score differs from ipTM the same way, and a
 * second implementation of "max over anchors of mean over selected" would be a
 * second chance to disagree about it.
 *
 * 🔴 AND AN ANCHOR IS TAKEN FROM EITHER SIDE. The reduction maximises over
 * every anchor whose row has selected pairs, so both chains contribute anchors
 * and the score is symmetric in the pair - which is what AF3's own chain-pair
 * ipTM reports.
 *
 * @param {ArrayLike<number>} term  tokens * tokens, the per-pair TM term
 * @param {number} tokens
 * @param {ArrayLike<number>} asymId  tokens, the chain each token belongs to
 * @param {ArrayLike<number>} seqMask  tokens
 * @returns {{chains: number[], scores: Map<string, number>}} `scores` is keyed
 *   `"a|b"` with a < b, holding that interface's ipTM; a pair with no live
 *   cross pairs is absent rather than zero
 */
export function chainPairTmScores(term, tokens, asymId, seqMask) {
  const chains = [];
  for (let i = 0; i < tokens; i += 1) {
    if (seqMask[i] > 0 && !chains.includes(asymId[i])) chains.push(asymId[i]);
  }
  chains.sort((a, b) => a - b);
  const scores = new Map();
  for (let a = 0; a < chains.length; a += 1) {
    for (let b = a + 1; b < chains.length; b += 1) {
      const first = chains[a];
      const second = chains[b];
      const score = reduceTmScore(term, tokens, (i, j) => seqMask[i] > 0 && seqMask[j] > 0
        && ((asymId[i] === first && asymId[j] === second)
          || (asymId[i] === second && asymId[j] === first)));
      if (Number.isFinite(score)) scores.set(`${first}|${second}`, score);
    }
  }
  return { chains, scores };
}
