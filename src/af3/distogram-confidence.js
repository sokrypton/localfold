/**
 * A per-token confidence from the trunk's distogram and a set of coordinates.
 *
 * WHAT IT IS FOR. The real confidence head costs more than a denoiser call -
 * 226 ms against 211 at 150 tokens, and 958 ms at 300 - so colouring a
 * diffusion TRAJECTORY with it would more than double a fold. This is the cheap
 * stand-in: the trunk already predicts a distribution over every pair's
 * distance, and it is already read back to the host, so asking how well a frame
 * agrees with that prediction costs one table lookup per pair and no GPU work
 * at all.
 *
 * WHAT IT IS. lDDT's shape with the distogram standing in for the reference
 * structure. lDDT scores a model by the fraction of reference distances it
 * preserves within a tolerance; replace "the reference distance" with "the
 * distribution the trunk predicted" and the fraction becomes a probability
 * mass. A token's score is the mean, over the CONTACTS THAT PIN IT, of
 *
 *     P(|D - d_ij| <= 0.75 A)
 *
 * where D is distributed as the trunk's distogram for (i, j) and d_ij is the
 * frame's own distance. It is in [0, 1] and is scaled to pLDDT's 0-100.
 *
 * 🔴 WHAT IT IS TUNED AGAINST IS SETTLING, NOT pLDDT, AND THE TWO WANT
 * OPPOSITE SETTINGS. The distogram is FIXED for a fold; only the structure
 * moves. So what a coloured trajectory shows is how much of the structure has
 * settled, and the honest reference for a frame is the fold's own final answer
 * - per-token lDDT against it, which every fold can compute. pLDDT is a
 * different quantity about a different question, and tuning against it picked
 * the eight sharpest LONG-RANGE contacts (|i - j| >= 6, tolerance 0.75 A).
 * Tuning against settling picks the opposite - all pairs from |i - j| >= 1,
 * tolerance 1 A - because local structure settles first and short-range pairs
 * are what track it. Across an eight-target panel that is 0.489 mean rank
 * correlation with what has actually settled, against 0.411 for the
 * pLDDT-tuned settings.
 *
 * 🔴 AND 0.49 IS WHAT IT IS: A ROUGH SIGNAL. On targets that fold it reaches
 * 0.63 to 0.88; on a GS linker and trp-cage it is 0.19 to 0.28. The
 * property that holds everywhere is the GLOBAL one - the mean rises
 * monotonically and saturates on all eight targets, including the scramble,
 * the linker and the homopolymer. Use it to show a structure resolving. Do not
 * invite anyone to read a single residue's colour.
 *
 * 🔴 THREE ALTERNATIVES LOST AGAINST THE pLDDT OBJECTIVE and were not re-run
 * against settling, so they are recorded as leads rather than as settled. The
 * first is worth re-testing if anyone returns to this; the other two are
 * unlikely to change sign.
 *
 * 🔴 TAKING THE BEST-HELD CONTACTS RATHER THAN THE MEAN IS WORSE, which is
 * the obvious next thing to try. Scoring a token by its single best contact, or
 * the mean of its best three or five, measured 0.58, 0.58 and 0.61 worst-case
 * against the mean's 0.664 - one contact holding up does not mean a position is
 * placed, and it takes agreement across several to say so.
 *
 * 🔴 IT SELECTS ON CONFIDENCE, NOT ON CONTACTS MADE, AND THAT IS THE BETTER
 * ANSWER. Ranking the candidates by the trunk's own contact probability -
 * P(d <= 8 A), which the distogram head already computes - reaches only 0.52
 * worst-case, and tightening the inclusion radius from 15 A to a real contact
 * distance loses at every setting. A pair the trunk confidently places at 13 A
 * is still a strong geometric constraint on where a token can be; restricting
 * to contacts throws those constraints away.
 *
 * 🔴 AND IT IS A WINDOWED MASS, NOT A CROSS ENTROPY. Scoring the observed bin
 * by log P - the cross entropy of the frame under the prediction, which is the
 * textbook choice - measured 0.593 worst-case against 0.664, and lost at every
 * matched setting. The bins are 0.3125 A wide, so a single bin's probability is
 * mostly a statement about the discretisation; a window that spans a few of
 * them is tolerant of exactly that and of nothing else.
 *
 * 🔴 IT IS NOT pLDDT AND MUST NOT BE LABELLED AS IT. pLDDT is a learned head
 * predicting the lDDT of a structure against the unknown truth; this measures
 * agreement with another of the model's own predictions, so it cannot know
 * anything the trunk did not.
 *
 * 🔴 AND THE TEST THAT MATTERS IS ACROSS TARGETS, NOT WITHIN ONE. Ranking
 * residues inside a fold says whether the loop looks worse than the core.
 * Whether a BAD fold looks worse than a good one is a different question, and
 * it is the one somebody comparing two predictions is asking. On an
 * eight-target panel spanning three that fold, a miniprotein, a scramble, a
 * linker and a homopolymer (tools/gpu/probe-distogram-confidence.js
 * --sequences=panel):
 *
 *     across targets   rank 0.762, linear 0.712 against mean pLDDT
 *     within targets   0.29 to 0.83 where the real pLDDT actually varies
 *
 * 🔴 THE WITHIN-TARGET NUMBERS ARE ONLY MEANINGFUL WHERE THERE IS SOMETHING TO
 * RANK. The panel's three worst - a GS linker at -0.30, poly-alanine at 0.06,
 * trp-cage at 0.46 - all have a real pLDDT that barely varies (spread 3.1, 2.4
 * and 2.1 points), so the correlation there is noise against noise and a low
 * number is not a failure. The probe reports that spread beside the
 * correlation for exactly this reason.
 *
 * 🔴 AND THE TOP OF THE RANGE IS WHERE THE RAW SCORE IS WORST. trp-cage is the
 * panel's most confident target at a real 96.6 and scores 54.0: a small rigid
 * protein's DISTANCE uncertainty does not shrink the way its pLDDT rises. So
 * the raw number must not be compared between folds. `calibrateToPlddt` is the
 * answer to that and is not optional if the colour is meant to mean anything.
 *
 * 🔴 AND ITS COST IS ALL IN THE PREPARATION, WHICH IS ONCE PER FOLD. The table
 * below is pairs x bins and takes one softmax and one prefix sum per pair;
 * after that a frame is one distance and one lookup per pair.
 */

/**
 * The tolerance, the inclusion radius, the minimum sequence separation and how
 * many contacts a token is scored on. All four were swept together against real
 * pLDDT on three proteins; see tools/gpu/probe-distogram-confidence.js.
 */
const TOLERANCE = 1.0;
const INCLUSION_RADIUS = 15;
const MINIMUM_SEPARATION = 1;
const CONTACTS = 16;
/**
 * How many of the strongest cross-chain contacts the interface score means
 * over: `k * L^(2/3)` of the SMALLER chain, floored.
 *
 * 🔴 IT SCALES WITH THE SMALLER CHAIN BECAUSE THE INTERFACE DOES. A twenty
 * residue peptide cannot present as many contacts as a ninety residue domain,
 * so a FIXED count reaches past the real interface on the small complexes and
 * stops short on the large. Measured on fifteen two-chain folds, as the
 * correlation between the estimate's residual error and the smaller chain's
 * length - the size bias itself, with the constant offset every version of this
 * has already removed:
 *
 *     fixed 64   +0.258        0.5 * L        -0.074
 *     fixed 32   +0.194        1.5 * L^(2/3)  -0.005
 *
 * 🔴 AND IT IS SUB-LINEAR BECAUSE AN INTERFACE IS A SURFACE. Two large proteins
 * do not meet over a large fraction of themselves - they touch on a patch. A
 * globular chain of L residues has a surface going as L^(2/3), and measured
 * interfaces are famously narrow in range, tens of residues across complexes
 * that differ in size by an order of magnitude. Linear extrapolates absurdly:
 * at 500 residues `0.5 * L` asks for 250 contacts and `1.5 * L^(2/3)` for 94,
 * and at 1000 it is 500 against 150.
 *
 * 🔴 THE EXPONENT IS CHOSEN ON THAT ARGUMENT AND NOT ON THE DATA, WHICH CANNOT
 * SEE IT. The fifteen targets span 20 to 93 residues in the smaller chain - a
 * factor of 4.6 - and over that range every exponent from 0 to 1 can be made
 * flat by trading the constant against it: p=0.5 and p=0.67 both reach a bias
 * of 0.000, and even p=1 reaches -0.008 at k=0.15. Their rank correlations sit
 * between 0.804 and 0.875, which fifteen targets cannot separate either. What
 * separates them is where they go OUTSIDE that range, and only one of them is
 * a statement about interfaces rather than about this panel.
 */
const INTERFACE_CONTACT_EXPONENT = 2 / 3;
const INTERFACE_CONTACT_SCALE = 1.5;
const INTERFACE_CONTACT_FLOOR = 8;
/**
 * The sequence separation p(intra) starts counting at.
 *
 * 🔴 IT CANNOT BE 1, AND THAT IS NOT A TUNING CHOICE. Adjacent residues are in
 * contact in every chain, folded or not, at a probability of essentially one -
 * so at separation 1 the strongest intra-chain contacts are the neighbours, and
 * p(intra) reads 1.000 on ALL SEVENTEEN targets of the panel including the
 * homopolymer. A quantity with no variance is not a measurement.
 *
 * 🔴 PAST THAT IT BARELY MATTERS, so 12 is chosen for what it MEANS rather than
 * for a number. Contact prediction has called |i - j| >= 12 medium-and-long
 * range for decades, which is the range that says a chain comes back on itself.
 * Swept against real pTM across the panel, separations 6, 12 and 24 sit at
 * Pearson 0.128, 0.144 and 0.167 - a spread far inside what seventeen targets
 * can resolve.
 */
const INTRA_SEPARATION = 12;
/**
 * ...relaxed for a chain too short to have contacts at that separation at all.
 *
 * 🔴 DERIVED FROM MINIMUM_SEPARATION, BECAUSE IT WAS WRITTEN OUT AND THEY
 * DISAGREED. This was the literal [6, 3, 1] while MINIMUM_SEPARATION said 1,
 * and the selection loop takes the FIRST of these that yields candidates - so
 * every chain long enough used 6 and the constant naming the behaviour was
 * decorative. A retune that set MINIMUM_SEPARATION to 1 changed nothing and
 * was measured as though it had.
 */
const FALLBACK_SEPARATIONS = (() => {
  const out = [];
  for (let step = MINIMUM_SEPARATION; step >= 1; step = Math.floor(step / 2)) out.push(step);
  return out;
})();

/**
 * The per-bin agreement table, computed once from a trunk's distogram.
 *
 * @param {Float32Array} logits  pairs * bins, the distogram head's output
 * @param {Float32Array} binEdges  bins - 1 break points, ascending
 * @param {number} tokens
 * @param {Float32Array} seqMask  tokens
 * @returns {{tokens: number, bins: number, centres: Float32Array,
 *            agreement: Float32Array, included: Uint8Array}}
 */
export function distogramAgreementTable(logits, binEdges, tokens, seqMask) {
  const bins = binEdges.length + 1;
  const pairs = tokens * tokens;
  if (logits.length !== pairs * bins) {
    throw new RangeError(`distogram logits are ${logits.length}; expected ${pairs * bins}`);
  }
  // ...the outer bins are open, so their centres are extrapolated by one
  // spacing rather than read from the array. See createDistogramShader.
  const spacing = binEdges.length > 1 ? binEdges[1] - binEdges[0] : 1;
  const centres = new Float32Array(bins);
  centres[0] = binEdges[0] - spacing / 2;
  for (let b = 1; b < bins - 1; b += 1) centres[b] = (binEdges[b - 1] + binEdges[b]) / 2;
  centres[bins - 1] = binEdges[bins - 2] + spacing / 2;

  // One softmax a pair, with its mean and spread. The spread is what ranks the
  // contacts: a sharp distogram is a contact the trunk is sure of.
  const probability = new Float64Array(bins);
  const expected = new Float32Array(pairs);
  const spread = new Float32Array(pairs);
  const perPair = [];
  for (let pair = 0; pair < pairs; pair += 1) {
    const base = pair * bins;
    let largest = -Infinity;
    for (let b = 0; b < bins; b += 1) largest = Math.max(largest, logits[base + b]);
    let total = 0;
    for (let b = 0; b < bins; b += 1) {
      probability[b] = Math.exp(logits[base + b] - largest);
      total += probability[b];
    }
    let mean = 0;
    for (let b = 0; b < bins; b += 1) {
      probability[b] /= total;
      mean += probability[b] * centres[b];
    }
    let variance = 0;
    for (let b = 0; b < bins; b += 1) variance += probability[b] * (centres[b] - mean) ** 2;
    expected[pair] = mean;
    spread[pair] = Math.sqrt(variance);
    perPair.push(Float64Array.from(probability));
  }

  // The contacts each token is scored on, chosen once from the distogram.
  const neighbours = new Int32Array(tokens * CONTACTS).fill(-1);
  const counts = new Int32Array(tokens);
  for (let i = 0; i < tokens; i += 1) {
    if (seqMask[i] <= 0) continue;
    for (const separation of FALLBACK_SEPARATIONS) {
      const candidates = [];
      for (let j = 0; j < tokens; j += 1) {
        if (seqMask[j] <= 0 || Math.abs(i - j) < separation) continue;
        const pair = i * tokens + j;
        if (expected[pair] > INCLUSION_RADIUS) continue;
        candidates.push(j);
      }
      if (candidates.length === 0) continue;
      candidates.sort((a, b) => spread[i * tokens + a] - spread[i * tokens + b]);
      const take = Math.min(CONTACTS, candidates.length);
      for (let k = 0; k < take; k += 1) neighbours[i * CONTACTS + k] = candidates[k];
      counts[i] = take;
      break;
    }
  }

  // ...and the agreement table for THOSE pairs only, which is tokens * 8 * bins
  // rather than tokens^2 * bins: 307 KB at 150 tokens where the full table is
  // 5.8 MB, and the preparation shrinks with it.
  const agreement = new Float32Array(tokens * CONTACTS * bins);
  for (let i = 0; i < tokens; i += 1) {
    for (let k = 0; k < counts[i]; k += 1) {
      const j = neighbours[i * CONTACTS + k];
      const distribution = perPair[i * tokens + j];
      const slot = (i * CONTACTS + k) * bins;
      for (let observed = 0; observed < bins; observed += 1) {
        let mass = 0;
        for (let b = 0; b < bins; b += 1) {
          if (Math.abs(centres[b] - centres[observed]) <= TOLERANCE) mass += distribution[b];
        }
        agreement[slot + observed] = mass;
      }
    }
  }
  return {
    tokens, bins, contacts: CONTACTS, centres, agreement, neighbours, counts, binEdges,
    // ...kept for the TM estimate below, which needs EVERY pair rather than the
    // eight a token is coloured by: pTM means over all of them.
    expected, spread,
  };
}

/**
 * One frame's per-token confidence, from the table and its pseudo-beta atoms.
 *
 * @param {ReturnType<typeof distogramAgreementTable>} table
 * @param {Float32Array} pseudoBeta  tokens * 3
 * @returns {Float32Array} tokens, each 0-100
 */
export function distogramConfidence(table, pseudoBeta) {
  const { tokens, bins, contacts, agreement, neighbours, counts, binEdges } = table;
  if (pseudoBeta.length !== tokens * 3) {
    throw new RangeError(`pseudoBeta is ${pseudoBeta.length}; expected ${tokens * 3}`);
  }
  const first = binEdges[0];
  const spacing = binEdges.length > 1 ? binEdges[1] - binEdges[0] : 1;
  const scores = new Float32Array(tokens);
  for (let i = 0; i < tokens; i += 1) {
    const count = counts[i];
    if (count === 0) continue;
    let total = 0;
    for (let k = 0; k < count; k += 1) {
      const j = neighbours[i * contacts + k];
      let squared = 0;
      for (let axis = 0; axis < 3; axis += 1) {
        const delta = pseudoBeta[i * 3 + axis] - pseudoBeta[j * 3 + axis];
        squared += delta * delta;
      }
      // ...the edges are evenly spaced, so the bin is arithmetic rather than a
      // search: bin 0 is everything below the first break, the last is
      // everything above the last.
      let bin = Math.floor((Math.sqrt(squared) - first) / spacing) + 1;
      if (bin < 0) bin = 0;
      if (bin > bins - 1) bin = bins - 1;
      total += agreement[(i * contacts + k) * bins + bin];
    }
    scores[i] = (total / count) * 100;
  }
  return scores;
}

/**
 * Put the stand-in on the fold's OWN pLDDT scale, using the final frame.
 *
 * 🔴 THIS IS WHAT MAKES IT COMPARABLE, AND WITHOUT IT THE RAW SCORE IS NOT.
 * Measured across an eight-target panel, the raw score orders targets about
 * right - rank correlation 0.76 against their mean pLDDT - but its ABSOLUTE
 * range is compressed and length-dependent, and the top of the range is where
 * it is worst: trp-cage is the panel's most confident target at a real 96.6 and
 * scores 54.0. Colouring two folds with the raw number would say the wrong
 * thing about which was better.
 *
 * A fold computes the real per-token pLDDT of its FINAL structure anyway, so
 * every trajectory has an anchor. Matching the mean and spread of the stand-in
 * to that anchor costs nothing, is exact on the final frame by construction,
 * and leaves the ranking untouched - the map is affine and increasing, so it
 * moves no residue past another. What it cannot fix is how the INTERMEDIATE
 * frames map, which nothing can: there is no pLDDT for a half-formed
 * structure to check against.
 *
 * 🔴 AND IT NEEDS THE FOLD TO HAVE FINISHED, which is a constraint on the UI
 * and not on the arithmetic. A trajectory drawn live has no anchor yet; the
 * honest options are to colour live on the raw score and recolour once the
 * fold lands, or to draw the trajectory only on replay.
 *
 * @param {Float32Array} approxFinal  the stand-in on the final frame
 * @param {Float64Array|Float32Array} realFinal  per-token pLDDT of that frame
 * @param {Float32Array} mask  tokens; only live tokens are fitted
 * @returns {(scores: Float32Array) => Float32Array}
 */
export function calibrateToPlddt(approxFinal, realFinal, mask) {
  const live = [];
  for (let i = 0; i < mask.length; i += 1) if (mask[i] > 0) live.push(i);
  const mean = (pick) => live.reduce((sum, i) => sum + pick(i), 0) / Math.max(live.length, 1);
  const spread = (pick, m) => Math.sqrt(
    live.reduce((sum, i) => sum + (pick(i) - m) ** 2, 0) / Math.max(live.length, 1));
  const approxMean = mean((i) => approxFinal[i]);
  const realMean = mean((i) => realFinal[i]);
  const approxSd = spread((i) => approxFinal[i], approxMean);
  const realSd = spread((i) => realFinal[i], realMean);
  // ...a flat stand-in cannot be stretched to a varying pLDDT, and pretending
  // otherwise would amplify noise into a colour. Shift only.
  const gain = approxSd > 1e-3 ? realSd / approxSd : 0;
  return (scores) => {
    const out = new Float32Array(scores.length);
    for (let i = 0; i < scores.length; i += 1) {
      const value = realMean + gain * (scores[i] - approxMean);
      out[i] = Math.min(100, Math.max(0, value));
    }
    return out;
  };
}

/**
 * A per-pair TM term from the distogram and a frame, for pTM and ipTM.
 *
 * 🔴 THE SAME REDUCTION AS THE REAL SCORE, ONLY THE TERM IS ESTIMATED. pTM is
 * `max over anchors i of mean over selected j of E[tm(PAE_ij)]`, and
 * src/heads/tm-score.js already owns that reduction and its chain selection -
 * so this produces only the term, and both scores fall out of `reduceTmScore`
 * exactly as the confidence head's own do. ipTM differs from pTM in the
 * SELECTION, not here.
 *
 * 🔴 THE ERROR IS THE TRUNK'S UNCERTAINTY AND THE FRAME'S DISAGREEMENT, IN
 * QUADRATURE. PAE asks how far off j is when aligned on i. The distogram does
 * not answer that - it describes one distance, not a relative frame - but it
 * bounds it from two sides, and both are in angstroms:
 *
 *     e_ij = sqrt( spread_ij^2 + (d_ij - E[D_ij])^2 )
 *
 * A pair the trunk is unsure of cannot be well aligned however the frame falls,
 * and a pair the frame has put at the wrong distance is misaligned whatever the
 * trunk thought. Neither term alone is the error; the quadrature is the
 * smallest thing that respects both.
 *
 * 🔴 AND IT IS A FLOOR, NOT AN ESTIMATE OF PAE. A pair can sit at exactly the
 * predicted distance and still be rotated wrongly about it - the distogram
 * cannot see that, so this can only ever be optimistic. Measured on the
 * ten-target panel it is optimistic by +0.125 of pTM on average, high on nine
 * of the ten.
 *
 * 🔴 IT TRACKS ipTM TOO, AND AN EARLIER VERSION OF THIS COMMENT SAID IT COULD
 * NOT. That claim was measured on two complexes whose real ipTM was 0.129 and
 * 0.153 - both at the bottom of the range - so a constant offset read as a
 * total failure. Across ten complexes spanning 0.12 to 0.74 it correlates at
 * 0.734 (0.954 excluding two nonsense homodimers), biased high throughout. Two
 * points cannot tell a bias from a blindness, and I called it a blindness.
 *
 * 🔴 BUT `distogramInterfaceContact` BEATS IT FOR ipTM AND IS SIMPLER. See
 * below. This is kept for pTM, where it correlates at 0.843 across the mixed
 * panel and 0.997 across the complexes.
 *
 * It is deliberately NOT wired to the page: it is optimistic by about +0.16 of
 * pTM and the bias is not constant, so it would sit beside the real score at
 * the final frame and visibly step.
 *
 * @param {ReturnType<typeof distogramAgreementTable>} table
 * @param {Float32Array} pseudoBeta  tokens * 3
 * @param {number} d0  from tmScoreD0(tokens), the caller's own
 * @returns {Float32Array} tokens * tokens
 */
export function distogramTmTerm(table, pseudoBeta, d0) {
  const { tokens, expected, spread } = table;
  if (pseudoBeta.length !== tokens * 3) {
    throw new RangeError(`pseudoBeta is ${pseudoBeta.length}; expected ${tokens * 3}`);
  }
  const d0Squared = d0 * d0;
  const term = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const pair = i * tokens + j;
      let squared = 0;
      for (let axis = 0; axis < 3; axis += 1) {
        const delta = pseudoBeta[i * 3 + axis] - pseudoBeta[j * 3 + axis];
        squared += delta * delta;
      }
      const deviation = Math.sqrt(squared) - expected[pair];
      const error = spread[pair] * spread[pair] + deviation * deviation;
      term[pair] = 1 / (1 + error / d0Squared);
    }
  }
  return term;
}

/**
 * How strongly the trunk believes two chains touch, from its own contacts.
 *
 * 🔴 THE STRONGEST CROSS-CHAIN CONTACTS, NOT THE AVERAGE OF THEM, AND THAT IS
 * THE WHOLE ESTIMATOR. The distogram head already computes P(d <= 8 A) for
 * every pair, so an interface is a question it has effectively been asked
 * already: are there cross-chain pairs it is confident about? Averaging over
 * ALL cross-chain pairs answers a different question - most of them are far
 * apart in any complex, so the mean measures interface SIZE against total size
 * and collapses to nothing.
 *
 * Measured against the real ipTM on two-chain folds spanning 0.10 to 0.74, as
 * Pearson over ten of them and over the eight that are not nonsense
 * homodimers:
 *
 *     top 4    0.742 / 0.872      top 64   0.890 / 0.974
 *     top 16   0.809 / 0.923      top 128  0.863 / 0.972
 *     top 32   0.854 / 0.953      ALL     0.156 / 0.120
 *
 * The collapse at "all" against the peak around 64 is the result: averaging
 * every cross-chain pair measures interface size against total size and
 * carries almost nothing. It also beats the distance-agreement estimator above
 * on rank - 0.855 against 0.552 - which is what a score anyone compares
 * between folds needs.
 *
 * How many to take is not a constant; see INTERFACE_CONTACT_FRACTION.
 *
 * 🔴 IT IS FRAME-INDEPENDENT, WHICH IS THE PRICE. It reads the distogram alone,
 * so it is one number for a fold and cannot animate the way the pLDDT stand-in
 * does. What it buys for that is availability: it is ready the moment the trunk
 * is, before a single denoiser call.
 *
 * 🔴 AND IT IS NOT ON ipTM's SCALE. It is a mean probability - 0.99 where ipTM
 * is 0.74 - so it ranks well and cannot be shown as an ipTM without a
 * calibration nobody here has fitted.
 *
 * @param {ArrayLike<number>} contactProbs  tokens * tokens, P(d <= 8 A)
 * @param {ArrayLike<number>} asymId  tokens, the chain each token belongs to
 * @param {Float32Array} seqMask  tokens
 * @param {number} tokens
 * @param {number} [count]  how many of the strongest to mean over; by default
 *   `1.5 * L^(2/3)` of the smaller chain, floored at 8
 * @returns {number} 0 to 1, or NaN when there is no interface to score
 */
export function distogramInterfaceContact(contactProbs, asymId, seqMask, tokens, count) {
  const { cross, smallest } = crossChainContacts(contactProbs, asymId, seqMask, tokens);
  // ...NaN rather than zero for a single chain, as ipTM itself reports: there
  // is no interface to score, which is not the same as a bad one.
  return topMean(cross, count ?? contactCount(smallest));
}

/**
 * The two numbers the page actually shows: p(intra) and p(inter).
 *
 * 🔴 THEY ARE WHAT THEY SAY AND NOTHING ELSE. Both are a mean of the trunk's
 * own P(d <= 8 A) over the contacts it is most confident about - within a chain
 * for p(intra), across chains for p(inter). No calibration, no fitted mapping,
 * no borrowed name. The page used to show this machinery as an estimated pLDDT
 * and an interface WORD, and a number named after a quantity it is not is
 * worse than a number named after itself, however well it correlates.
 *
 * 🔴 AND THEY DO CORRELATE, WHICH IS WHY THEY ARE WORTH SHOWING. Measured
 * against the real confidence head across the panel in
 * tools/gpu/probe-contact-confidence.js, as Pearson and Spearman:
 *
 *     p(inter) vs ipTM   0.844 / 0.905   over 8 complexes
 *     p(inter) vs pTM    0.734 / 0.833   over the same 8
 *
 * p(inter) is the one that earns its place: it ranks complexes the way ipTM
 * does, and it is known the moment the trunk is - before a single denoiser
 * call, where the real score is minutes away.
 *
 * 🔴 p(intra) TRACKS NOTHING THE HEAD REPORTS, AND IS SHOWN ANYWAY BECAUSE IT
 * IS SHOWN AS ITSELF. Across the same seventeen targets it reaches 0.144 /
 * 0.075 against pTM and 0.293 / 0.157 against mean pLDDT - which is nothing,
 * and no slice of the panel rescues it: monomers alone 0.046, complexes alone
 * 0.115, excluding the nonsense targets 0.207.
 *
 * 🔴 THE FAILURE IS REAL AND IS NOT THE REDUCTION'S. The control is pTM's OWN
 * reduction run over the distogram's expected TM term with no coordinates in
 * it - same input, the head's own arithmetic - and it reaches only 0.531 /
 * 0.385. The trunk's distogram carries the interface; it does not carry a
 * single chain's pTM.
 *
 * 🔴 AND GCN4 IS WHY, WHICH IS WORTH KNOWING BEFORE READING p(intra). A coiled
 * coil is one long helix: real pLDDT 83.9, real pTM 0.506, and p(intra) 0.017,
 * because it has no long-range intra-chain contacts to be confident about. That
 * is p(intra) being correct about a different question. It answers "does this
 * chain come back on itself, and is the trunk sure?" - not "is this a good
 * prediction?".
 *
 * 🔴 p(intra) SKIPS THE SHORT SEPARATIONS AND HAS TO. Neighbouring residues are
 * in contact in any chain, folded or not, at a probability of essentially one,
 * so the strongest intra-chain contacts of a random coil and of a real domain
 * are the same contacts. Requiring |i - j| >= INTRA_SEPARATION asks the only
 * question that distinguishes them: does the chain come back on itself?
 *
 * 🔴 AND IT FALLS BACK FOR A CHAIN TOO SHORT TO HAVE ANY, rather than
 * returning nothing - a twenty residue miniprotein has few pairs that far
 * apart and would otherwise report NaN where it has a real answer.
 *
 * @param {ArrayLike<number>} contactProbs  tokens * tokens, P(d <= 8 A)
 * @param {ArrayLike<number>} asymId  tokens
 * @param {Float32Array} seqMask  tokens
 * @param {number} tokens
 * @param {{separation?: number}} [options]  the separation floor, for the probe
 *   that swept it; the default is the swept value
 * @returns {{intra: number, inter: number}} each 0 to 1, and `inter` NaN when
 *   there is only one chain
 */
export function distogramContactConfidence(contactProbs, asymId, seqMask, tokens,
    options = {}) {
  const separation = options.separation ?? INTRA_SEPARATION;
  const { cross, smallest, live } = crossChainContacts(
    contactProbs, asymId, seqMask, tokens);
  let within = [];
  for (const floor of separationsFrom(separation)) {
    within = withinChainContacts(contactProbs, asymId, seqMask, tokens, floor);
    if (within.length > 0) break;
  }
  return {
    intra: topMean(within, contactCount(live)),
    inter: topMean(cross, contactCount(smallest)),
  };
}

/**
 * Every cross-chain contact probability, and the sizes the count rule needs.
 *
 * 🔴 BOTH ORDERS OF EACH PAIR, WHICH IS DELIBERATE AND LOAD-BEARING. The list
 * holds (i, j) and (j, i), so a "top 64" is really the strongest 32 pairs. The
 * count rule above was swept against real ipTM on a list built this way, so
 * de-duplicating it here would halve the effective count and silently move
 * every number the sweep settled.
 */
function crossChainContacts(contactProbs, asymId, seqMask, tokens) {
  const cross = [];
  const sizes = new Map();
  for (let i = 0; i < tokens; i += 1) {
    if (seqMask[i] <= 0) continue;
    sizes.set(asymId[i], (sizes.get(asymId[i]) ?? 0) + 1);
    for (let j = 0; j < tokens; j += 1) {
      if (seqMask[j] <= 0 || asymId[i] === asymId[j]) continue;
      cross.push(contactProbs[i * tokens + j]);
    }
  }
  let live = 0;
  for (const size of sizes.values()) live += size;
  return { cross, live, smallest: sizes.size === 0 ? 0 : Math.min(...sizes.values()) };
}

/** The same, within a chain and past a sequence separation. Both orders too. */
function withinChainContacts(contactProbs, asymId, seqMask, tokens, separation) {
  const within = [];
  for (let i = 0; i < tokens; i += 1) {
    if (seqMask[i] <= 0) continue;
    for (let j = 0; j < tokens; j += 1) {
      if (seqMask[j] <= 0 || asymId[i] !== asymId[j]) continue;
      if (Math.abs(i - j) < separation) continue;
      within.push(contactProbs[i * tokens + j]);
    }
  }
  return within;
}

/** The mean of the strongest `count`, or NaN when there is nothing to mean. */
function topMean(values, count) {
  if (values.length === 0) return Number.NaN;
  values.sort((a, b) => b - a);
  const take = Math.min(count, values.length);
  let total = 0;
  for (let index = 0; index < take; index += 1) total += values[index];
  return total / take;
}

/** How many strong contacts a unit of `length` residues can present. */
function contactCount(length) {
  return Math.max(INTERFACE_CONTACT_FLOOR, Math.round(
    INTERFACE_CONTACT_SCALE * length ** INTERFACE_CONTACT_EXPONENT));
}

/** `separation`, then halves of it, then 1 - for a chain too short for it. */
function separationsFrom(separation) {
  const out = [];
  for (let step = Math.max(1, separation); step > 1; step = Math.floor(step / 2)) out.push(step);
  out.push(1);
  return out;
}
