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
 * The per-pair TM term, from the PAE logits.
 *
 * 🔴 ONCE PER PREDICTION, NOT ONCE PER SCORE. Every score over one prediction -
 * pTM, ipTM, and one per chain pair - is the same `tokens^2` expectations
 * reduced over different pairs, and the expectation is the expensive half:
 * O(tokens^2 * bins) against the reduction's O(tokens^2), with 64 bins. Doing
 * it inside each score made AlphaFold 2 pay for it twice, and would have made a
 * five-chain complex pay for it twelve times.
 *
 * 🔴 AND IT IS float64, NOT float32. The values it replaces were doubles held
 * in a local, so storing them narrower would move published scores in the last
 * digits - and test/prediction-results.test.js pins AlphaFold 2's to twelve.
 *
 * @param {ArrayLike<number>} logits  tokens * tokens * bins
 * @param {number} tokens
 * @param {ArrayLike<number>} tmPerBin  bins
 * @returns {Float64Array} tokens * tokens
 */
/**
 * Several expectations of ONE softmax over the same logits.
 *
 * 🔴 THE PAE HEAD TOOK THE SAME SOFTMAX TWICE, AND IT IS THE SLOWEST THING ON
 * THE MAIN THREAD. `predictedAlignedError` is the expectation of the bin
 * CENTRES and the pTM term is the expectation of `1 / (1 + centre^2 / d0^2)` -
 * the same probabilities, weighted differently - and they were two passes over
 * `tokens^2 * bins` logits, each calling `Math.exp` on every one. At 825
 * residues and 64 bins that is **43.6 million exp() calls, twice**: measured
 * 726 ms and 729 ms, 1.46 s of a 26.4 s fold, on the thread that also has to
 * paint the page.
 *
 * One pass, several weight vectors. The arithmetic per output is unchanged -
 * same maximum, same denominator, same order - so every score is bit-identical.
 */
export function softmaxExpectations(logits, rows, bins, weightSets) {
  if (logits.length !== rows * bins) {
    throw new RangeError(`logits should be ${rows * bins} long, not ${logits.length}`);
  }
  const outputs = weightSets.map(() => new Float64Array(rows));
  const sets = weightSets.length;
  // 🔴 THE ACCUMULATORS ARE LOCALS, WHICH IS WORTH 19% AGAINST 100%. Written
  // generally - `outputs[set][row] += ...` inside the bin loop - fusing two
  // expectations cost 1224 ms against 994 for one, when the whole point is that
  // the second should be nearly free: the exp() calls are shared and only the
  // multiply-add is not. Typed-array element accumulation is what ate it. The
  // one and two cases are written out; nothing calls this with more.
  if (sets === 1 || sets === 2) {
    const first = outputs[0];
    const second = sets === 2 ? outputs[1] : first;
    const firstWeights = weightSets[0];
    const secondWeights = sets === 2 ? weightSets[1] : firstWeights;
    for (let row = 0; row < rows; row += 1) {
      const base = row * bins;
      // Against the largest logit, because the logits are unbounded and exp()
      // of them is not.
      let largest = Number.NEGATIVE_INFINITY;
      for (let bin = 0; bin < bins; bin += 1) {
        if (logits[base + bin] > largest) largest = logits[base + bin];
      }
      let denominator = 0;
      let firstTotal = 0;
      let secondTotal = 0;
      for (let bin = 0; bin < bins; bin += 1) {
        const probability = Math.exp(logits[base + bin] - largest);
        denominator += probability;
        firstTotal += probability * firstWeights[bin];
        secondTotal += probability * secondWeights[bin];
      }
      first[row] = firstTotal / denominator;
      if (sets === 2) second[row] = secondTotal / denominator;
    }
    return outputs;
  }
  for (let row = 0; row < rows; row += 1) {
    const base = row * bins;
    let largest = Number.NEGATIVE_INFINITY;
    for (let bin = 0; bin < bins; bin += 1) {
      if (logits[base + bin] > largest) largest = logits[base + bin];
    }
    let denominator = 0;
    for (let set = 0; set < sets; set += 1) outputs[set][row] = 0;
    for (let bin = 0; bin < bins; bin += 1) {
      const probability = Math.exp(logits[base + bin] - largest);
      denominator += probability;
      for (let set = 0; set < sets; set += 1) {
        outputs[set][row] += probability * weightSets[set][bin];
      }
    }
    for (let set = 0; set < sets; set += 1) outputs[set][row] /= denominator;
  }
  return outputs;
}

export function tmTermFromLogits(logits, tokens, tmPerBin) {
  const bins = tmPerBin.length;
  if (logits.length !== tokens * tokens * bins) {
    throw new RangeError(`PAE logits should be ${tokens * tokens * bins} long,`
      + ` not ${logits.length}`);
  }
  return softmaxExpectations(logits, tokens * tokens, bins, [tmPerBin])[0];
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
  // 🔴 THE EXPECTATION AND THE REDUCTION ARE THE SAME TWO HALVES AS EVERYWHERE
  // ELSE. This used to inline the softmax inside the reduction loop, which is a
  // second copy of arithmetic that must agree with tmTermFromLogits to the last
  // digit or two paths score the same prediction differently. Composed, they
  // cannot disagree. It evaluates every pair rather than only the selected
  // ones, which is more work for a narrow selection and the same work for a
  // broad one - and the callers that reduce a prediction more than once now
  // build the term themselves and skip this entirely.
  return reduceTmScore(tmTermFromLogits(logits, tokens, tmPerBin), tokens, selects);
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
/**
 * pTM and ipTM for each chain on its own, as AlphaFold 3 computes them.
 *
 * 🔴 ON THE INTERFACE TERM, NOT THE GLOBAL ONE. `term` here is AF3's second TM
 * term (tmscore_adjusted_pae's "interface" version: d0 from the pair's own
 * chains - the chain's tokens on the diagonal, the two chains' summed off it -
 * see chainTmInfo), from which AF3 reduces `chain_pair_iptm`; `chain_ptm[i]` is
 * its diagonal, the score over pairs both of which are chain i. Scored on the
 * global term (d0 from every token) a short chain in a large complex read ~0.10
 * high - a binder's pTM, reported by Yeqing Lin.
 *
 * 🔴 AND `chain_iptm[i]` IS THE MEAN OF CHAIN i's PAIR SCORES, AF3's
 * `iptm_xchain`: each other chain's `chain_pair_iptm` with it, averaged. It was
 * the best anchor's mean over every other chain's tokens pooled - the same for
 * two chains, not for three.
 *
 * `frame` (tokenFrameMask): the tokens that may be anchors, AF3's
 * `pae_single_mask` - an ion has no frame and aligns nothing.
 *
 * @returns {{chains: number[], chainPtm: Map<number, number>,
 *            chainIptm: Map<number, number>}}
 */
export function perChainTmScores(term, tokens, asymId, seqMask, frame = undefined) {
  const chains = [];
  for (let i = 0; i < tokens; i += 1) {
    if (seqMask[i] > 0 && !chains.includes(asymId[i])) chains.push(asymId[i]);
  }
  chains.sort((a, b) => a - b);
  const chainPtm = new Map();
  const chainIptm = new Map();
  const pairs = chainPairTmScores(term, tokens, asymId, seqMask, frame).scores;
  for (const chain of chains) {
    const within = reduceTmScore(term, tokens, (i, j) => seqMask[i] > 0 && seqMask[j] > 0
      && (frame === undefined || frame[i]) && asymId[i] === chain && asymId[j] === chain);
    if (Number.isFinite(within)) chainPtm.set(chain, within);
    let total = 0;
    let count = 0;
    for (const other of chains) {
      if (other === chain) continue;
      const score = pairs.get(chain < other ? `${chain}|${other}` : `${other}|${chain}`);
      if (score !== undefined && Number.isFinite(score)) { total += score; count += 1; }
    }
    if (count > 0) chainIptm.set(chain, total / count);
  }
  return { chains, chainPtm, chainIptm };
}

/**
 * AlphaFold 3's interface TM term's inputs, packed for a shader: each token's
 * dense chain index (-1: not live), the chain count, then a chain pair's bins'
 * weights `1 / (1 + centre^2 / d0^2)` at d0 of the pair's tokens - the chain's
 * own on the diagonal, the two chains' summed off it (confidence_head.py's
 * num_interface_tokens). `chainOf` any id a token (< 0: none), `live` the
 * tokens counted. [tokens + 1 + chains^2 bins] floats.
 */
export function chainTmInfo(chainOf, live, binCentres, counted = undefined) {
  // (counted: the tokens d0 counts, where they are not these - OpenDDE's head runs on structural tokens and its
  // scores are reported over the residues: {chainOf, live} of the residue tokens)
  const tokens = chainOf.length;
  const countOf = counted?.chainOf ?? chainOf;
  const countLive = counted?.live ?? live;
  const ids = [];
  for (let i = 0; i < countOf.length; i += 1) {
    if (countLive[i] > 0 && countOf[i] >= 0 && !ids.includes(countOf[i])) ids.push(countOf[i]);
  }
  ids.sort((a, b) => a - b);
  const chains = ids.length;
  const bins = binCentres.length;
  const info = new Float32Array(tokens + 1 + Math.max(chains, 1) ** 2 * bins);
  const count = new Array(chains).fill(0);
  for (let i = 0; i < countOf.length; i += 1) {
    if (countLive[i] > 0 && countOf[i] >= 0) count[ids.indexOf(countOf[i])] += 1;
  }
  for (let i = 0; i < tokens; i += 1) info[i] = live[i] > 0 && chainOf[i] >= 0 ? ids.indexOf(chainOf[i]) : -1;
  info[tokens] = chains;
  for (let a = 0; a < chains; a += 1) {
    for (let b = 0; b < chains; b += 1) {
      const perBin = tmPerBinFor(binCentres, tmScoreD0(a === b ? count[a] : count[a] + count[b]));
      info.set(perBin, tokens + 1 + (a * chains + b) * bins);
    }
  }
  return info;
}

/**
 * `tmTermFromLogits` and `interfaceTmTerm` in ONE sweep over the logits: each
 * pair's softmax is taken once and both expectations read it. The same
 * arithmetic in the same order as the two separately - the max by comparison,
 * one exp a bin, the sums in bin order - so both terms are bit-identical to
 * theirs; a confidence readout spent ~100 ms of host time taking the 64-bin
 * softmax over every pair twice.
 *
 * @returns {{adjusted: Float64Array, interface: Float32Array}}
 */
export function tmTermsFromLogits(logits, tokens, tmPerBin, info) {
  const bins = tmPerBin.length;
  if (logits.length !== tokens * tokens * bins) {
    throw new RangeError(`PAE logits should be ${tokens * tokens * bins} long,`
      + ` not ${logits.length}`);
  }
  const chains = info[tokens];
  const adjusted = new Float64Array(tokens * tokens);
  const interfaceTerm = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    const a = info[i];
    for (let j = 0; j < tokens; j += 1) {
      const b = info[j];
      const live = a >= 0 && b >= 0;
      const pair = i * tokens + j;
      const base = pair * bins;
      const table = live ? tokens + 1 + (a * chains + b) * bins : 0;
      let largest = Number.NEGATIVE_INFINITY;
      for (let bin = 0; bin < bins; bin += 1) {
        if (logits[base + bin] > largest) largest = logits[base + bin];
      }
      let denominator = 0;
      let global = 0;
      let local = 0;
      for (let bin = 0; bin < bins; bin += 1) {
        const probability = Math.exp(logits[base + bin] - largest);
        denominator += probability;
        global += probability * tmPerBin[bin];
        if (live) local += probability * info[table + bin];
      }
      adjusted[pair] = global / denominator;
      if (live) interfaceTerm[pair] = local / denominator;
    }
  }
  return { adjusted, interface: interfaceTerm };
}

/**
 * The interface TM term on the host, off PAE logits [tokens^2, bins] and a
 * chainTmInfo - the shaders' arithmetic, for the readouts' host reference.
 */
export function interfaceTmTerm(logits, tokens, info) {
  const chains = info[tokens];
  const bins = (info.length - tokens - 1) / Math.max(chains, 1) ** 2;
  const out = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const a = info[i];
      const b = info[j];
      if (a < 0 || b < 0) continue;
      const base = (i * tokens + j) * bins;
      const table = tokens + 1 + (a * chains + b) * bins;
      let largest = -Infinity;
      for (let k = 0; k < bins; k += 1) largest = Math.max(largest, logits[base + k]);
      let total = 0;
      let weighted = 0;
      for (let k = 0; k < bins; k += 1) {
        const e = Math.exp(logits[base + k] - largest);
        total += e;
        weighted += e * info[table + k];
      }
      out[i * tokens + j] = weighted / total;
    }
  }
  return out;
}

/**
 * AlphaFold 3's token frames' validity (features.py, Frames.compute_features),
 * the pTM scores' anchors (model.py's pae_single_mask, the rows alone): every
 * polymer token has a frame; a ligand token one only where its residue has three
 * tokens or more and the token with its two nearest by the reference conformer
 * is not within 25 degrees of colinear (1 - |cos| < 0.0937, or two at one
 * place). An ion, a two-atom ligand: no anchor. cuda/af3/src/frames.cuh's twin.
 *
 * @returns {Uint8Array} tokens, 1 where the token may anchor a pTM alignment
 */
export function tokenFrameMask({ tokens, dense, isLigand, asymId, residueIndex, refPos, refMask }) {
  const ok = new Uint8Array(tokens).fill(1);
  if (isLigand === undefined || refPos === undefined) return ok;
  const at = (t) => {     // a ligand token's one atom: its first live slot
    for (let s = 0; s < dense; s += 1) {
      const k = t * dense + s;
      if (refMask[k] > 0) return [refPos[k * 3], refPos[k * 3 + 1], refPos[k * 3 + 2]];
    }
    return null;
  };
  for (let t = 0; t < tokens; t += 1) {
    if (!isLigand[t]) continue;
    const local = [];
    for (let u = 0; u < tokens; u += 1) {
      if (isLigand[u] && asymId[u] === asymId[t] && residueIndex[u] === residueIndex[t]) local.push(u);
    }
    if (local.length < 3) { ok[t] = 0; continue; }
    const xt = at(t);
    const cost = local.map((u) => {
      const xu = at(u);
      let d = xu !== null && xt !== null ? Math.hypot(xu[0] - xt[0], xu[1] - xt[1], xu[2] - xt[2]) : 0;
      if (xu === null) d += 1e8;
      if (u === t) d += 1e8;
      return [d, u];
    });
    cost.sort((x, y) => x[0] - y[0]);      // (ties in token order; AF3's argsort leaves them to its sort - real distances do not tie)
    const xa = at(cost[0][1]) ?? [0, 0, 0];
    const xc = at(cost[1][1]) ?? [0, 0, 0];
    const base = xt ?? [0, 0, 0];
    const v1 = [0, 1, 2].map((d) => xa[d] - base[d]);
    const v2 = [0, 1, 2].map((d) => xc[d] - base[d]);
    const n1 = Math.hypot(...v1);
    const n2 = Math.hypot(...v2);
    if (n1 < 1e-8 || n2 < 1e-8) { ok[t] = 0; continue; }
    const cos = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2);
    ok[t] = 1 - Math.abs(cos) < 0.0937 ? 0 : 1;
  }
  return ok;
}

export function chainPairTmScores(term, tokens, asymId, seqMask, frame = undefined) {
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
        && (frame === undefined || frame[i])
        && ((asymId[i] === first && asymId[j] === second)
          || (asymId[i] === second && asymId[j] === first)));
      if (Number.isFinite(score)) scores.set(`${first}|${second}`, score);
    }
  }
  return { chains, scores };
}

/**
 * AlphaFold 3's five TM numbers from the two terms: pTM, ipTM, and the three
 * chain-level scores the AF3 server writes beside them.
 *
 * 🔴 ONE DEFINITION, BECAUSE THERE WERE THREE AND THE THIRD WAS THE OLD ONE.
 * The page's WebGPU fold, the page's WebGL2 fold and the WebGL2 fold TOOL each
 * wrote this reduction out by hand, and the tool's copy went through
 * `cpu/af3/confidence/ptm.js` - anchors over every live token and no chain
 * fields at all - so the chain-pTM fix (reported by Yeqing Lin) could not be
 * checked through the one tool that folds AF3 on that backend. The two page
 * copies agreed only because they were written on the same afternoon.
 *
 * 🔴 AND THE TWO TERMS ARE NOT INTERCHANGEABLE. `tmAdjusted` is the GLOBAL term
 * (one d0 over every token) and pTM and ipTM are reduced from it, which is what
 * AF3's `_compute_ptm` reads for both; `tmInterface` is the interface term (d0
 * from the pair's own chains, see chainTmInfo) and every CHAIN score is reduced
 * from that. Swapping them leaves a plausible number between 0 and 1: on the
 * global term a short chain in a large complex read ~0.10 high.
 *
 * `frame` is tokenFrameMask - the tokens that may be ANCHORS, AF3's
 * `pae_single_mask`. Pass it; `undefined` anchors on every live token, which is
 * what AlphaFold 2 does (every AF2 token is a residue, so every one has a
 * frame) and is wrong wherever an ion or a two-atom ligand is a token.
 *
 * @param {{tmAdjusted: ArrayLike<number>, tmInterface: ArrayLike<number>,
 *   tokens: number, asymId: ArrayLike<number>, seqMask: ArrayLike<number>,
 *   frame?: ArrayLike<number>}} input
 * @returns {{ptm: number, iptm: number, chainPtm: Object<string, number>,
 *   chainIptm: Object<string, number>, chainPairIptm: Object<string, number>}}
 *   ipTM is NaN for a single chain - there is no interface to score, and zero
 *   would read as a confident failure; a chain whose score is not finite is
 *   ABSENT from its map rather than null (an ion anchors nothing).
 */
export function af3TmScores({ tmAdjusted, tmInterface, tokens, asymId, seqMask, frame }) {
  const selected = (i, j) => seqMask[i] > 0 && seqMask[j] > 0
    && (frame === undefined || frame[i] === 1);
  const perChain = perChainTmScores(tmInterface, tokens, asymId, seqMask, frame);
  return {
    ptm: reduceTmScore(tmAdjusted, tokens, selected),
    iptm: reduceTmScore(tmAdjusted, tokens,
      (i, j) => selected(i, j) && asymId[i] !== asymId[j]),
    // ...and one score per INTERFACE, because the pooled ipTM averages them: an
    // assembly holding a native dimer and a designed binder reports the easy
    // interface's confidence for the hard one. See chainPairTmScores.
    chainPairIptm: Object.fromEntries(
      chainPairTmScores(tmInterface, tokens, asymId, seqMask, frame).scores),
    chainPtm: Object.fromEntries(perChain.chainPtm),
    chainIptm: Object.fromEntries(perChain.chainIptm),
  };
}
