/**
 * A predicted aligned error for a model that has no confidence head.
 *
 * WHAT IT IS. EF2-fast ships with `confidence_head.enabled: false` and carries
 * zero confidence tensors, so it has no pLDDT and no PAE. It does have a
 * distogram, and a PAE is what says whether two parts of a structure are placed
 * correctly relative to each other - the one question a contact map and a
 * per-residue score cannot answer. This estimates one.
 *
 * 🔴 A DISTOGRAM SEES ONE PROJECTION OF WHAT A PAE MEASURES. Aligning on token
 * i's frame and letting `D_ij = dx_j - dx_i` be the relative displacement
 * error,
 *
 *     PAE(i,j)^2  ~  E||D_ij||^2          the full 3-D magnitude
 *     sigma_ij^2  =  E[(u_ij . D_ij)^2]   ONE radial projection of it
 *
 * so the direction a distogram cannot see is the tangential one - which is
 * exactly how a domain ROTATION displaces things. That is why sigma alone
 * under-reads inter-domain error, and why this is a fit rather than a formula.
 *
 * 🔴 AND PAE^2 IS A SQUARED-DISTANCE MATRIX, WHICH SAYS WHAT THE FEATURES ARE.
 * `E||D_ij||^2 = g_ii + g_jj - 2 g_ij` for `g` the Gram matrix of displacement
 * covariances - a PER-TOKEN mobility plus a PAIR coupling. Measured on AF3
 * folds, PAE needs 6-10 components for 90% of its energy where its own sigma
 * needs 13-33: the target is a low-rank collective signal and the distogram is
 * a high-rank per-pair one. So the mobility terms below are not a decoration;
 * they are worth 0.745 median Spearman with no pair term at all, against a
 * geometry baseline of 0.658.
 *
 * HOW IT WAS FITTED, AND AGAINST WHAT. AlphaFold 3 emits a distogram AND a real
 * PAE from one fold, so the estimator was fitted and scored there first
 * (`tools/pae-from-distogram.py`, leave-one-target-out: median Spearman 0.885
 * against 0.658 for the geometry alone, and 0.450 with sigma attached to the
 * WRONG pairs - below using no distogram at all, which is what says the
 * correspondence is being used). These coefficients are the same fit run on
 * EF2-fast's OWN distogram against AF3's PAE for the same sequences
 * (`tools/pae-transfer.py`).
 *
 * 🔴 THE TARGET IS ANOTHER MODEL'S PAE, AND THAT BOUNDS WHAT THIS CAN CLAIM.
 * AF3's PAE is about AF3's structure; this estimate is about EF2-fast's. Where
 * the two fold a target the same way the difference is small; where they
 * disagree the comparison is invalid rather than the estimate wrong. Measured:
 * the correlation between the two models' distance matrices predicts the score
 * at Pearson 0.772, and splitting on it -
 *
 *     same fold (agreement >= 0.9)   this 0.790,  AF3's own head 0.900
 *     different fold                 this 0.548
 *
 * - so read it as "close to a real confidence head where the fold is the same",
 * not as a calibrated PAE in angstroms.
 */

/**
 * 🔴 IT ORDERS PAIRS INSIDE ONE FOLD AND SAYS ALMOST NOTHING ACROSS FOLDS, AND
 * THAT IS THE OPPOSITE WAY ROUND FROM THE CERTAINTY. Measured over the nine
 * matched targets, one point per fold - the mean estimate against the mean true
 * PAE - it reads Pearson **0.340** and Spearman **0.117**, and its range is
 * 8.68 to 9.50 A where the truth's is 3.04 to 12.71. It is nearly a constant
 * between folds. Within a fold it orders pairs at 0.746.
 *
 * The certainty in distogram-webgpu.js is the exact complement: 0.90 across
 * folds and a median 0.44 within one, which is why its per-residue colour was
 * measured and not shipped. **So the two answer different questions and neither
 * substitutes for the other** - the pAE for "which parts of THIS fold are
 * placed relative to which", the certainty for "is this fold worth looking at".
 * That is also how a real PAE is read: nobody compares the mean PAE of two
 * different targets, they look at the block structure of one.
 *
 * 🔴 SO THE ANGSTROMS ARE A REGRESSION ONTO ANGSTROMS AND NOT A CALIBRATION.
 * Per-target bias runs -3.88 to +5.99 A with a mean of +0.69: it regresses to
 * the global mean, so an easy target reads far too high and a hard one too low.
 * Report the MAP, not the number.
 */

/** Bin centres in ANGSTROMS. See the note on `CONTACT_EDGES`. */
function binCentres(bins, edges) {
  const width = (edges.maximum - edges.minimum) / bins;
  const centres = new Float64Array(bins);
  for (let b = 0; b < bins; b += 1) centres[b] = edges.minimum + width * (b + 0.5);
  return { centres, width };
}

/**
 * Per-pair moments of the distogram, in angstroms.
 *
 * 🔴 THE BIAS IS NOT IN THE LOGITS BUFFER. The projection has none and the
 * contact pass adds it as it reads, so a caller that softmaxes the logits alone
 * gets a distribution - just the wrong one.
 *
 * 🔴 AND EVERY MOMENT IS IN ANGSTROMS, NOT IN BINS. AF3's grid is 64 bins over
 * 2-22 A and this one is 128 over 2-52, so a spread read in bin INDICES differs
 * by a factor of two between them for the same physical uncertainty - and the
 * entropy is not even the same unit, since a uniform distribution over 128 bins
 * carries log 2 more nats than one over 64 for free. That is why the width
 * feature is `exp(H) * binWidth` and not `H`.
 */
export function distogramMoments(logits, bias, tokens, bins, edges) {
  const { centres, width } = binCentres(bins, edges);
  const pairs = tokens * tokens;
  const mean = new Float32Array(pairs);
  const sigma = new Float32Array(pairs);
  const effectiveWidth = new Float32Array(pairs);
  const probability = new Float64Array(bins);
  for (let index = 0; index < pairs; index += 1) {
    const base = index * bins;
    let largest = -Infinity;
    for (let b = 0; b < bins; b += 1) {
      const value = logits[base + b] + bias[b];
      if (value > largest) largest = value;
    }
    let total = 0;
    for (let b = 0; b < bins; b += 1) {
      probability[b] = Math.exp(logits[base + b] + bias[b] - largest);
      total += probability[b];
    }
    let first = 0, second = 0, entropy = 0;
    for (let b = 0; b < bins; b += 1) {
      const p = probability[b] / total;
      first += p * centres[b];
      second += p * centres[b] * centres[b];
      if (p > 1e-12) entropy -= p * Math.log(p);
    }
    mean[index] = first;
    sigma[index] = Math.sqrt(Math.max(0, second - first * first));
    effectiveWidth[index] = Math.exp(entropy) * width;
  }
  return { mean, sigma, effectiveWidth };
}

/**
 * The fitted weights, in the order `FEATURES` names, with the intercept last.
 *
 * 🔴 AND THE FIT IS CONSTRAINED SO A UNIFORM RISE IN sigma CANNOT LOWER THE
 * ANSWER. `sigma` and the mobility terms are strongly correlated - the mobility
 * IS a mean of sigma - so an unconstrained least squares gives them large
 * opposite signs, and the net effect of raising every pair's spread by 1 A came
 * out at **-0.513 A**: the estimate went DOWN when the model got less sure. That
 * is fragile off the training distribution and indefensible on screen. One
 * linear constraint - the aggregate sigma direction pinned to zero - removes it
 * and IMPROVES the fit, 0.738 -> **0.746** median Spearman, which is what
 * removing a spurious direction looks like. Ridge was the obvious alternative
 * and is a far worse trade: it only turns the slope positive at lambda 1e5, by
 * which point the median has fallen to 0.661.
 *
 * 🔴 AND THE CONSTRAINT HAS TO NAME EVERY FEATURE THAT MOVES, WHICH `effWidth`
 * DOES. It is `exp(H) * binWidth`, and a Gaussian's `exp(H)` is
 * `sigma * sqrt(2 pi e)` - so widening every distribution by 1 A moves it by
 * 4.13, not by nothing. A first constraint over sigma and the mobilities alone
 * left a residual slope of -0.15 A per angstrom, which the unit test caught:
 * small enough to look like rounding and still the wrong sign.
 *
 * 🔴 SO THE ESTIMATE IS SCALE-INVARIANT IN sigma, DELIBERATELY. Multiplying
 * every spread by a constant leaves it unchanged; what moves it is one pair's
 * spread against the rest. That is what the mobility terms make it - a
 * contrast, not an absolute - and it is the reason a coefficient set fitted on
 * one bundle survives a re-quantisation of that bundle.
 *
 * 🔴 THESE ARE EF2-fast's OWN, NOT AF3's. The two distograms are on different
 * grids, so a coefficient fitted on one does not carry to the other even with
 * every feature in angstroms - the SPREADS differ, because a 128-bin grid
 * reaching 52 A can express an uncertainty a 64-bin grid stopping at 22 cannot.
 * Refit with tools/pae-transfer.py if the bundle's distogram changes.
 */
export const FEATURES = ["sigma", "d", "effWidth", "|E[d]-d|", "min(d,22)",
  "mob_i+mob_j", "mobfar_i+mobfar_j", "max(mob)"];
export const WEIGHTS = [1.39871, 0.27169, 0.09798, -0.07524, 0.06195,
  0.06506, -1.74362, 1.55348];
export const INTERCEPT = 4.00842;

/** PAE is reported over this range by AF3 and clamped to it here. */
export const MAX_ANGSTROMS = 32;

/**
 * The estimate, one value per ordered pair, row-major `i * tokens + j`.
 *
 * @param {{mean: ArrayLike<number>, sigma: ArrayLike<number>,
 *          effectiveWidth: ArrayLike<number>}} moments  from `distogramMoments`
 * @param {ArrayLike<number>} distances  tokens * tokens, between the same
 *   REPRESENTATIVE atoms the distogram is defined on - `representativeAtoms`,
 *   not alpha carbons, or the pair the model scored is not the pair measured.
 * @param {number} tokens
 */
export function alignedErrorFromDistogram(moments, distances, tokens) {
  const pairs = tokens * tokens;
  if (moments.sigma.length !== pairs || distances.length !== pairs) {
    throw new RangeError(`expected ${pairs} pairs;`
      + ` got ${moments.sigma.length} moments and ${distances.length} distances`);
  }
  // ...the Gram terms: how uncertain a token's distances are in general, and
  // the same over FAR pairs only, which is where a collective motion shows and
  // a bond length does not.
  const mobility = new Float64Array(tokens);
  const mobilityFar = new Float64Array(tokens);
  for (let i = 0; i < tokens; i += 1) {
    let total = 0, far = 0, farCount = 0;
    for (let j = 0; j < tokens; j += 1) {
      const s = moments.sigma[i * tokens + j];
      total += s;
      if (distances[i * tokens + j] > 12) { far += s; farCount += 1; }
    }
    mobility[i] = total / tokens;
    mobilityFar[i] = farCount === 0 ? mobility[i] : far / farCount;
  }

  const out = new Float32Array(pairs);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const index = i * tokens + j;
      const d = distances[index];
      const feature = [
        moments.sigma[index], d, moments.effectiveWidth[index],
        Math.abs(moments.mean[index] - d), Math.min(d, 22),
        mobility[i] + mobility[j], mobilityFar[i] + mobilityFar[j],
        Math.max(mobility[i], mobility[j]),
      ];
      let value = INTERCEPT;
      for (let k = 0; k < WEIGHTS.length; k += 1) value += WEIGHTS[k] * feature[k];
      out[index] = Math.max(0, Math.min(MAX_ANGSTROMS, value));
    }
  }
  // A token has no aligned error against itself.
  for (let i = 0; i < tokens; i += 1) out[i * tokens + i] = 0;
  return out;
}
