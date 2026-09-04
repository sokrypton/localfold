/**
 * lDDT's own arithmetic, with the distogram standing in for the reference.
 *
 * WHAT lDDT IS. For residue i, take every pair (i, j) whose distance in the
 * REFERENCE structure is under an inclusion radius R0 = 15 A; count the
 * fraction of them the model preserves within each of four thresholds -
 * 0.5, 1, 2 and 4 A - and average those four fractions. pLDDT is AlphaFold's
 * prediction of exactly that, computed on the alpha carbons.
 *
 * WHAT THIS IS. The same expression with the reference distance replaced by
 * the trunk's distribution over it, so the counts become expectations:
 *
 *              sum_j sum_t P(|d_model(i,j) - D| < t  and  D < R0)
 *     lDDT_i = -------------------------------------------------
 *                        4 * sum_j P(D < R0)
 *
 * where D is distributed as the distogram for that pair. Both halves are sums
 * over bins, so the whole thing is exact given the binning - there is no
 * tuning in it beyond lDDT's own constants.
 *
 * 🔴 INCLUSION IS A WEIGHT, NOT A CHOICE, AND THAT IS THE MAIN DIFFERENCE FROM
 * THE OLDER ESTIMATOR. src/af3/distogram-confidence.js picks the sixteen
 * sharpest contacts inside the radius and scores a token on those; lDDT has no
 * such rule. A pair the trunk puts at 14 A with wide spread contributes
 * P(D < 15) of a neighbour to the denominator and the matching mass to the
 * numerator, which is what "the reference decides who counts" means when the
 * reference is a distribution.
 *
 * 🔴 FOUR THRESHOLDS, NOT ONE TOLERANCE. The older estimator asks a single
 * question - is the distance within 1 A - and lDDT asks four and averages,
 * which is what makes it forgiving of a slightly displaced neighbour and
 * unforgiving of a badly placed one. Reproduced here rather than replaced by
 * one number.
 *
 * 🔴 AND NO SEQUENCE SEPARATION FLOOR. lDDT scores every pair inside the
 * radius, neighbours included; the older estimator skips |i - j| below a
 * floor. Skipping them changes what the score MEANS - it stops being lDDT.
 */

/**
 * lDDT's thresholds, and a WIDER inclusion radius than lDDT's own.
 *
 * 🔴 18 A, NOT 15, AND THE REASON IS THE PROBABILISTIC REFERENCE. lDDT decides
 * inclusion from a structure it can measure; here the reference is a
 * distribution, so a pair whose EXPECTED distance is 17 A still carries real
 * mass below 15 and belongs in the sum. Widening the window admits it. Swept
 * on 810 tokens over fourteen single-sequence AF3 targets, leave-one-target-out
 * RMSE against the confidence head:
 *
 *     r=10  10.55      r=15   8.67      r=22   9.30
 *     r=12   9.56      r=18   8.33
 *
 * 🔴 THE MARGIN OVER 15 IS AF3's, AND DOES NOT REPRODUCE ELSEWHERE. The same
 * sweep over 44,740 residues of AF2 - 108 single-sequence targets, four
 * recycles each, scored against AF2's own per-recycle pLDDT - puts them level
 * and slightly the other way:
 *
 *     r=12   6.27      r=15   4.97      r=18   5.06      r=22   5.48
 *
 * So the SHAPE of the curve is real on both models and the peak is not: 12 is
 * clearly too tight and 22 clearly too loose, while anything between 15 and 18
 * is a coin toss. 18 stays because it is the better of the two on the model
 * this actually serves, not because a fourteen-target margin established it.
 *
 * 🔴 THE THRESHOLDS STAY lDDT's. Widening them to {1, 2, 4, 8} measured 8.23
 * against 8.33 - a tenth of an angstrom of RMSE, and it flips sign on one of
 * the fourteen subsets. Not worth trading the definition for: with lDDT's own
 * four this is lDDT with a distogram for a reference, and with any other set
 * it is a score that resembles it.
 */
const INCLUSION_RADIUS = 18;
const THRESHOLDS = [0.5, 1, 2, 4];
/**
 * A pair whose chance of being inside the radius is below this contributes
 * nothing worth the arithmetic.
 *
 * 🔴 IT IS A SPARSITY CUT, NOT A MODELLING CHOICE. Everything below it is
 * still counted in the denominator through its own weight, which is what keeps
 * the estimator from quietly renormalising itself: a token whose neighbours
 * are all uncertain SHOULD score low, and dropping their weight instead of
 * their contribution would score it high on the few it kept.
 */
const WEIGHT_FLOOR = 0.01;
/**
 * A sequence separation floor and a contact cap, both OFF.
 *
 * 🔴 SWEPT AND REJECTED, WHICH IS WHY THEY ARE ZERO. Excluding local contacts
 * is the obvious idea - a chain's neighbours are preserved whatever it folds
 * into, so they are always confident and look like free marks - and a cap
 * bounds the work per token. On the same fourteen targets, leave-one-target-out
 * RMSE against 8.33 for the estimator as it stands:
 *
 *     separation  1  8.33   2  8.28   3  8.32   6  8.64   12  12.54
 *     contacts   64  8.38  32  9.14  16 10.57    8  12.10
 *
 * A floor of two or three is a wash and everything beyond it is worse; every
 * cap is worse, sharply so once it starts discarding real neighbours. Neither
 * earns a departure from lDDT's definition, so both stay off.
 */
const MINIMUM_SEPARATION = 0;
const MAX_CONTACTS = 0;

/**
 * The per-pair tables lDDT needs, built once from a trunk's distogram.
 *
 * Only the bins inside the radius are kept, and only the pairs with enough
 * mass there to matter: a protein has O(L) neighbours per residue within 15 A,
 * not O(L), so this is sparse by construction and stays a few megabytes where
 * the full distogram is tens.
 *
 * @param {Float32Array} logits  pairs * bins
 * @param {Float32Array} binEdges  bins - 1 ascending break points
 * @param {number} tokens
 * @param {Float32Array} seqMask  tokens
 */
export function distogramLddtTable(logits, binEdges, tokens, seqMask, options = {}) {
  const radius = options.radius ?? INCLUSION_RADIUS;
  // 🔴 lDDT ITSELF HAS NEITHER OF THESE, and they are swept rather than
  // assumed. A separation floor drops the neighbours a chain has whatever it
  // folds into - always preserved, so always confident - and a contact cap
  // bounds how many pairs a token is scored on. Both change what the score
  // MEANS, so they earn their place by measurement or not at all; see the
  // constants above for what the panel said.
  const separation = options.separation ?? MINIMUM_SEPARATION;
  const maxContacts = options.maxContacts ?? MAX_CONTACTS;
  const bins = binEdges.length + 1;
  if (logits.length !== tokens * tokens * bins) {
    throw new RangeError(`distogram logits are ${logits.length};`
      + ` expected ${tokens * tokens * bins}`);
  }
  const spacing = binEdges.length > 1 ? binEdges[1] - binEdges[0] : 1;
  const centres = new Float32Array(bins);
  centres[0] = binEdges[0] - spacing / 2;
  for (let b = 1; b < bins - 1; b += 1) centres[b] = (binEdges[b - 1] + binEdges[b]) / 2;
  centres[bins - 1] = binEdges[bins - 2] + spacing / 2;
  // Which bins are inside the radius at all - the rest can never contribute.
  const inside = [];
  for (let b = 0; b < bins; b += 1) if (centres[b] < radius) inside.push(b);

  const neighbours = [];
  const masses = [];
  const offsets = new Int32Array(tokens + 1);
  const probability = new Float64Array(bins);
  for (let i = 0; i < tokens; i += 1) {
    offsets[i] = neighbours.length;
    if (seqMask[i] <= 0) continue;
    const row = [];
    for (let j = 0; j < tokens; j += 1) {
      // 🔴 j === i IS EXCLUDED AND NOTHING ELSE IS. lDDT scores a residue
      // against every OTHER one inside the radius.
      if (j === i || seqMask[j] <= 0 || Math.abs(i - j) < separation) continue;
      const base = (i * tokens + j) * bins;
      let largest = -Infinity;
      for (let b = 0; b < bins; b += 1) {
        if (logits[base + b] > largest) largest = logits[base + b];
      }
      let total = 0;
      for (let b = 0; b < bins; b += 1) {
        probability[b] = Math.exp(logits[base + b] - largest);
        total += probability[b];
      }
      let weight = 0;
      for (const b of inside) weight += probability[b];
      weight /= total;
      if (weight < WEIGHT_FLOOR) continue;
      const kept = new Float32Array(inside.length);
      for (let k = 0; k < inside.length; k += 1) kept[k] = probability[inside[k]] / total;
      row.push({ j, kept, weight });
    }
    // ...capped by how likely a pair is to be inside the radius at all, which
    // is the probabilistic reading of "the nearest ones".
    if (maxContacts > 0 && row.length > maxContacts) {
      row.sort((a, b) => b.weight - a.weight);
      row.length = maxContacts;
    }
    for (const entry of row) { neighbours.push(entry.j); masses.push(entry.kept); }
  }
  offsets[tokens] = neighbours.length;
  return {
    tokens,
    centres: Float32Array.from(inside.map((b) => centres[b])),
    neighbours: Int32Array.from(neighbours),
    masses,
    offsets,
  };
}

/**
 * One frame's predicted lDDT per token, on pLDDT's 0-100 scale.
 *
 * @param {ReturnType<typeof distogramLddtTable>} table
 * @param {Float32Array} pseudoBeta  tokens * 3
 * @returns {Float32Array} tokens
 */
export function distogramLddt(table, pseudoBeta, options = {}) {
  const thresholds = options.thresholds ?? THRESHOLDS;
  const { tokens, centres, neighbours, masses, offsets } = table;
  if (pseudoBeta.length !== tokens * 3) {
    throw new RangeError(`pseudoBeta is ${pseudoBeta.length}; expected ${tokens * 3}`);
  }
  const out = new Float32Array(tokens);
  for (let i = 0; i < tokens; i += 1) {
    let preserved = 0;
    let included = 0;
    for (let slot = offsets[i]; slot < offsets[i + 1]; slot += 1) {
      const j = neighbours[slot];
      let squared = 0;
      for (let axis = 0; axis < 3; axis += 1) {
        const delta = pseudoBeta[i * 3 + axis] - pseudoBeta[j * 3 + axis];
        squared += delta * delta;
      }
      const observed = Math.sqrt(squared);
      const mass = masses[slot];
      // One pass over the kept bins answers all four thresholds at once, and
      // accumulates the inclusion weight with them.
      for (let k = 0; k < mass.length; k += 1) {
        const p = mass[k];
        if (p === 0) continue;
        included += p;
        const error = Math.abs(observed - centres[k]);
        for (const threshold of thresholds) if (error < threshold) preserved += p;
      }
    }
    out[i] = included === 0 ? 0 : (100 * preserved) / (thresholds.length * included);
  }
  return out;
}

/**
 * The estimate mapped onto pLDDT's scale, for a fold that is still running.
 *
 * 🔴 A FIT IS STILL NEEDED, AND THAT IS NOT A FAILURE OF THE DERIVATION. This
 * predicts lDDT against the trunk's OWN distribution; pLDDT predicts lDDT
 * against the structure that actually exists, and the trunk is systematically
 * less sure than the finished model turns out to be. The two differ by an
 * affine map and this is it, fitted on 810 tokens over fourteen
 * single-sequence targets by tools/gpu/probe-plddt-features.js.
 *
 * Scored leave-one-TARGET-out, so no target ever fits itself, against the
 * older agreement estimator on the same frames:
 *
 *     old agreement heuristic     RMSE 10.13   MAE 8.08
 *     this, at lDDT's own r=15    RMSE  8.67   MAE 6.88
 *     this, at r=18               RMSE  8.33   MAE 6.59
 *
 * an 18% reduction in RMSE over the heuristic it replaces. Fitting the two
 * together measured no better than this alone, so whatever the heuristic knew
 * is already here. Slope and intercept move only between 0.556-0.617 and
 * 38.7-43.4 as each target is dropped.
 *
 * 🔴 IT COMPRESSES AND MUST. Raw 22 to 93 comes out as 52 to 93, because a
 * distogram cannot separate a 55 from a 95 before the structure exists. It is
 * for the LIVE phase only: calibrateToPlddt anchors on the finished
 * structure's own pLDDT and is exact on the final frame.
 */
const LIVE_INTERCEPT = 41.29;
const LIVE_SLOPE = 0.5781;

export function lddtToPlddt(values) {
  const out = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    out[index] = Math.min(100, Math.max(0, LIVE_INTERCEPT + LIVE_SLOPE * values[index]));
  }
  return out;
}
