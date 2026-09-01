/**
 * pTM and ipTM, from the confidence head's PAE logits.
 *
 * pTM says how well the whole prediction is expected to superpose on the truth;
 * ipTM says the same of the INTERFACE, and for a complex it is the number that
 * matters - two chains can each be folded perfectly and docked wrongly, which
 * pTM and pLDDT both report as a success.
 *
 * Nothing here is learned. The confidence head already produces the PAE logits;
 * this is the arithmetic AF3 applies to them afterwards, and it is written out
 * rather than approximated because every step of it is a place to be subtly
 * wrong in a way that still yields a plausible number between 0 and 1.
 *
 * 🔴 BOTH SCORES COME FROM THE *GLOBAL* ADJUSTED PAE. AF3 emits two tensors -
 * `tmscore_adjusted_pae_global` and `..._interface` - and the tempting reading
 * is that ipTM uses the interface one. It does not: `_compute_ptm` in AF3's
 * model.py reads the global tensor for both and differs only in the `interface`
 * flag, which masks the PAIRS rather than changing the adjustment. The
 * interface tensor is for the per-chain-pair breakdown alone.
 *
 * The reduction itself - max over anchors of the masked row mean - is shared
 * with AlphaFold 2 in src/heads/tm-score.js. Only the conventions differ, and
 * they are the arguments: AF3 takes its bin centres from its own error head,
 * masks by seq_mask, and identifies chains by asym_id rather than by assuming
 * they are contiguous blocks.
 */
import { bestAlignmentTmScore, tmPerBinFor, tmScoreD0 } from "../heads/tm-score.js";

/**
 * The TM-adjusted PAE: for each pair, the expected TM term over the PAE bins.
 *
 * @param {Float32Array} paeLogits  tokens * tokens * bins
 * @param {ArrayLike<number>} binCentres  bins
 * @param {ArrayLike<number>} tokenCounts tokens * tokens, the `num_res` each
 *   pair's d0 is computed from - a constant for the global form, and a function
 *   of the two chains' sizes for the interface one
 * @returns {Float32Array} tokens * tokens
 */
export function tmAdjustedPae(paeLogits, tokens, binCentres, tokenCounts) {
  const bins = binCentres.length;
  const output = new Float32Array(tokens * tokens);
  for (let pair = 0; pair < tokens * tokens; pair += 1) {
    const base = pair * bins;
    // The softmax is computed here rather than reused, because the confidence
    // head keeps only the expectation and this needs the whole distribution.
    let largest = -Infinity;
    for (let bin = 0; bin < bins; bin += 1) {
      if (paeLogits[base + bin] > largest) largest = paeLogits[base + bin];
    }
    let total = 0;
    for (let bin = 0; bin < bins; bin += 1) total += Math.exp(paeLogits[base + bin] - largest);

    const d0 = tmScoreD0(tokenCounts[pair]);
    const d0Squared = d0 * d0;
    let term = 0;
    for (let bin = 0; bin < bins; bin += 1) {
      const probability = Math.exp(paeLogits[base + bin] - largest) / total;
      term += probability / (1 + (binCentres[bin] * binCentres[bin]) / d0Squared);
    }
    output[pair] = term;
  }
  return output;
}

/**
 * The token count each pair's d0 is drawn from, in AF3's two forms.
 *
 * 🔴 THE INTERFACE COUNT IS NOT THE PAIR'S TWO CHAINS ADDED UP. AF3 sums the
 * two tokens' chain sizes and then, for pairs WITHIN one chain, subtracts half
 * - `num_interface -= same_chain * (num_interface // 2)` - with an integer
 * division, so a same-chain pair is scored against roughly one chain's length
 * rather than two. Using the plain sum makes every intra-chain d0 too large and
 * flatters the score.
 */
export function interfaceTokenCounts(tokens, asymId, pairMask) {
  const chainTokens = new Int32Array(tokens);
  for (let i = 0; i < tokens; i += 1) {
    let count = 0;
    for (let j = 0; j < tokens; j += 1) {
      if (asymId[i] === asymId[j] && pairMask[i * tokens + j]) count += 1;
    }
    chainTokens[i] = count;
  }
  const counts = new Int32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      let value = chainTokens[i] + chainTokens[j];
      if (asymId[i] === asymId[j]) value -= Math.floor(value / 2);
      counts[i * tokens + j] = pairMask[i * tokens + j] ? value : 0;
    }
  }
  return counts;
}

/**
 * pTM and ipTM for a prediction.
 *
 * @param {{paeLogits: Float32Array, tokens: number, binCentres: ArrayLike<number>,
 *          seqMask: ArrayLike<number>, asymId: ArrayLike<number>}} input
 * @returns {{ptm: number, iptm: number}} `iptm` is NaN for a single chain,
 *   because there is no interface to score - not zero, which would read as a
 *   confident failure.
 */
export function predictedTmScores(input) {
  const { paeLogits, tokens, binCentres, seqMask, asymId } = input;
  const pairMask = new Uint8Array(tokens * tokens);
  let sequenceTokens = 0;
  for (let i = 0; i < tokens; i += 1) if (seqMask[i] > 0) sequenceTokens += 1;
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      pairMask[i * tokens + j] = (seqMask[i] > 0 && seqMask[j] > 0) ? 1 : 0;
    }
  }
  // 🔴 d0 FROM THE WHOLE PREDICTION, FOR BOTH SCORES. AF3's `_compute_ptm`
  // reads `tmscore_adjusted_pae_global` for pTM and ipTM alike and differs only
  // in the `interface` flag, which narrows the PAIRS. The `_interface` tensor -
  // whose d0 varies per pair with the two chains' sizes - is for the
  // per-chain-pair breakdown, and using it here would be a different number.
  const tmPerBin = tmPerBinFor(binCentres, tmScoreD0(sequenceTokens));
  const selected = (i, j) => pairMask[i * tokens + j] === 1;
  return {
    ptm: bestAlignmentTmScore(paeLogits, tokens, tmPerBin, selected),
    // NaN for a single chain, because there is no interface to score. Zero
    // would read as a confident failure.
    iptm: bestAlignmentTmScore(paeLogits, tokens, tmPerBin,
      (i, j) => selected(i, j) && asymId[i] !== asymId[j]),
  };
}
