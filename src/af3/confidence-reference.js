/**
 * AF3's confidence head: pLDDT, PAE and PDE.
 *
 * This is what tells you whether to believe the structure, and it is the only
 * part of AF3 that reads the SAMPLED COORDINATES back in. Everything else runs
 * before there are any: the trunk builds representations, the diffusion head
 * turns them into atoms, and this then looks at those atoms alongside the trunk
 * and predicts its own error.
 *
 *     pair  += target_feat (both ways) + a distogram of the predicted structure
 *     4 x pairformer block, the same module the trunk runs 48 of
 *     pair  -> distance-error logits (symmetrised) -> PDE
 *           -> aligned-error logits                -> PAE
 *     single -> per-atom pLDDT logits              -> pLDDT
 *            -> per-atom resolved logits
 *
 * 🔴 pLDDT IS PER ATOM SLOT, NOT PER TOKEN. Its projection is (384, 24, 50):
 * one 50-bin distribution for every one of a token's 24 dense atom slots, from
 * the token's single representation. Reading it as (384, 50) and broadcasting
 * would run, produce plausible per-residue numbers, and throw away the
 * side-chain resolution that is the point of an atom-level model.
 *
 * 🔴 "left" AND "right" ARE THE OTHER WAY ROUND. In _embed_features AF3 writes
 * `left_target_feat_project(tf)` with no axis expansion and
 * `right_target_feat_project(tf)[:, None]` with one - so the LEFT projection
 * broadcasts along the row and is indexed by j, and the right by i. Swapping
 * them transposes a term nothing downstream will complain about.
 */
import { layerNorm, linear } from "./pairformer-reference.js";
import { relativeEncoding } from "./embedder-reference.js";

const NUM_BINS = 64;
const MAX_ERROR_BIN = 31.0;
const PLDDT_BINS = 50;
const DGRAM_BINS = 39;
const DGRAM_MIN = 3.25;
const DGRAM_MAX = 50.75;

/**
 * A one-hot distogram of the predicted structure, 39 bins from 3.25 to 50.75 A.
 *
 * 🔴 THE COMPARISON IS ON SQUARED DISTANCES AGAINST SQUARED EDGES, which is
 * AF3's own spelling and avoids a square root per pair. The final bin catches
 * everything beyond 50.75.
 */
export function distogramFeatures(positions, pairMask, tokens) {
  const lower = new Float64Array(DGRAM_BINS);
  for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
    const edge = DGRAM_MIN + (DGRAM_MAX - DGRAM_MIN) * bin / (DGRAM_BINS - 1);
    lower[bin] = edge * edge;
  }
  const output = new Float32Array(tokens * tokens * DGRAM_BINS);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      let squared = 0;
      for (let axis = 0; axis < 3; axis += 1) {
        const difference = positions[i * 3 + axis] - positions[j * 3 + axis];
        squared += difference * difference;
      }
      const base = (i * tokens + j) * DGRAM_BINS;
      for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
        const upper = bin + 1 < DGRAM_BINS ? lower[bin + 1] : 1e8;
        output[base + bin] = (squared > lower[bin] && squared < upper)
          ? pairMask[i * tokens + j] : 0;
      }
    }
  }
  return output;
}

/**
 * Bin centres for an error head: `bins - 1` edges, plus a catch-all.
 *
 * Exported because pTM reads the same centres from the same PAE logits, and two
 * copies of this would be two chances to disagree about where the bins sit.
 */
export function errorBinCentres(bins, maxErrorBin) {
  const step = maxErrorBin / (bins - 2);
  const centres = new Float64Array(bins);
  for (let bin = 0; bin < bins - 1; bin += 1) centres[bin] = bin * step + step / 2;
  centres[bins - 1] = centres[bins - 2] + step;
  return centres;
}

/** softmax over the last axis, then the expectation against `centres`. */
function expectation(logits, rows, bins, centres) {
  const output = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const base = row * bins;
    let largest = -Infinity;
    for (let bin = 0; bin < bins; bin += 1) {
      if (logits[base + bin] > largest) largest = logits[base + bin];
    }
    let total = 0;
    let weighted = 0;
    for (let bin = 0; bin < bins; bin += 1) {
      const probability = Math.exp(logits[base + bin] - largest);
      total += probability;
      weighted += probability * centres[bin];
    }
    output[row] = weighted / total;
  }
  return output;
}

/**
 * The confidence head.
 *
 * @param {{pair: Float32Array, single: Float32Array, targetFeat: Float32Array,
 *          pseudoBeta: Float32Array, seqMask: Float32Array, tokens: number,
 *          dense: number}} input
 *   `pseudoBeta` is the representative atom per token, already gathered through
 *   the batch's token_atoms_to_pseudo_beta.
 * @param {object} weights
 * @param {(state: object, weights: object, dialect: object) => object} block
 *   pairformerBlock, injected so this file does not import the trunk.
 * @param {{swapTransposedBias: boolean}} dialect
 */
/**
 * boltz2's confidence re-embedding: z and s rebuilt from the trunk's outputs.
 *
 * 🔴 NINE TERMS, NOT AF3's TWO, AND IT REPLACES z RATHER THAN ADDING TO IT.
 * AF3 adds two projections of target_feat and a distogram to the trunk's pair;
 * boltz2 LayerNorms the trunk's pair and single and then builds z from
 * scratch - relative position, token bonds, bond types, contact conditioning,
 * both target-feat projections, an OUTER PRODUCT of two more of them, and a
 * 64-bin distogram of the predicted coordinates.
 *
 * 🔴 AND "left" AND "right" ARE THE OTHER WAY ROUND HERE TOO, in the opposite
 * direction from `_embed_features`: boltz adds `s_to_z(s)[:, :, None]` -
 * broadcast along j, so indexed by i - which is the RIGHT projection, and
 * `s_to_z_transpose` is the LEFT one. Getting it backwards transposes a term
 * nothing downstream objects to.
 *
 * 🔴 THE CONTACT CONDITIONING RUNS ON A PLACEHOLDER. With no distance
 * restraints the one-hot is the UNSPECIFIED class, `selected` is 1, and the
 * encoder term is multiplied by zero - so what is left is the learned constant
 * `contact_encoding_unspecified`. That is what the module computes on this
 * input, not an approximation of it; the coverage limit is that this featuriser
 * has no restraint field to carry.
 */
function boltz2Reembed(input, weights, tokens, pairChannels, singleChannels, pairMask) {
  const w = weights.reembed;
  const pairs = tokens * tokens;
  const sInputs = layerNorm(input.targetFeat, tokens, weights.targetFeatWidth,
                            w.sInputsNormScale, w.sInputsNormOffset);
  const single = layerNorm(input.single, tokens, singleChannels,
                           w.sNormScale, w.sNormOffset);
  const fromInputs = linear(sInputs, tokens, weights.targetFeatWidth, singleChannels,
                            w.sInputToS);
  for (let index = 0; index < single.length; index += 1) single[index] += fromInputs[index];

  const pair = layerNorm(input.pair, pairs, pairChannels, w.zNormScale, w.zNormOffset);
  const relative = relativeEncoding(tokens, input.features);
  const relativeWidth = w.relPosProject.length / pairChannels;
  const positioned = linear(relative, pairs, relativeWidth, pairChannels, w.relPosProject);
  const left = linear(sInputs, tokens, weights.targetFeatWidth, pairChannels,
                      w.leftTargetFeatProject);
  const right = linear(sInputs, tokens, weights.targetFeatWidth, pairChannels,
                       w.rightTargetFeatProject);
  const prodIn1 = linear(sInputs, tokens, weights.targetFeatWidth, pairChannels,
                         w.sToZProdIn1);
  const prodIn2 = linear(sInputs, tokens, weights.targetFeatWidth, pairChannels,
                         w.sToZProdIn2);
  const dgram = boltz2DistogramFeatures(input.pseudoBeta, pairMask, tokens);
  const embedded = linear(dgram, pairs, BOLTZ2_DGRAM_BINS, pairChannels,
                          w.distogramFeatProject);
  const product = new Float32Array(pairChannels);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const index = i * tokens + j;
      const base = index * pairChannels;
      const bondOrder = input.bondOrderMatrix === undefined
        ? 0 : (input.bondOrderMatrix[index] | 0);
      const bondRow = (bondOrder >= 0 && bondOrder < 7 ? bondOrder : 0) * pairChannels;
      const bond = input.bondMatrix === undefined ? 0 : input.bondMatrix[index];
      for (let c = 0; c < pairChannels; c += 1) {
        product[c] = prodIn1[i * pairChannels + c] * prodIn2[j * pairChannels + c];
      }
      for (let c = 0; c < pairChannels; c += 1) {
        let total = positioned[base + c] + embedded[base + c]
          + bond * w.tokenBondsProject[c]
          + w.tokenBondsTypeEmbed[bondRow + c]
          + w.contactEncodingUnspecified[c]
          + right[i * pairChannels + c] + left[j * pairChannels + c];
        for (let e = 0; e < pairChannels; e += 1) {
          total += product[e] * w.sToZProdOut[e * pairChannels + c];
        }
        pair[base + c] += total;
      }
    }
  }
  return { pair, single };
}

const BOLTZ2_DGRAM_BINS = 64;

/**
 * boltz2's own 64-bin distance embedding, 63 edges evenly over 2..22 A.
 *
 * AF3's is 39 bins over 3.25..50.75 through a projection; boltz2's is an
 * nn.Embedding over a much finer, much shorter range. Nothing about a fold
 * depends on it, which is why it could be wrong without any fold gate noticing.
 */
function boltz2DistogramFeatures(positions, pairMask, tokens) {
  const pairs = tokens * tokens;
  const output = new Float32Array(pairs * BOLTZ2_DGRAM_BINS);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const index = i * tokens + j;
      let squared = 1e-10;
      for (let axis = 0; axis < 3; axis += 1) {
        const d = positions[i * 3 + axis] - positions[j * 3 + axis];
        squared += d * d;
      }
      const distance = Math.sqrt(squared);
      let bin = 0;
      for (let edge = 0; edge < BOLTZ2_DGRAM_BINS - 1; edge += 1) {
        if (distance > 2.0 + (20.0 * edge) / (BOLTZ2_DGRAM_BINS - 2)) bin += 1;
      }
      output[index * BOLTZ2_DGRAM_BINS + bin] = pairMask[index];
    }
  }
  return output;
}

export function confidenceHead(input, weights, block, dialect) {
  const { tokens, dense, seqMask } = input;
  const pairChannels = weights.pairChannels;
  const singleChannels = weights.singleChannels;
  const pairs = tokens * tokens;

  const pairMask = new Float32Array(pairs);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }

  // 🔴 ONE DIALECT REBUILDS z RATHER THAN ADDING TO IT; see `boltz2Reembed`.
  let pair;
  let single;
  if (dialect?.reembedConfidencePair === true) {
    const rebuilt = boltz2Reembed(input, weights, tokens, pairChannels,
                                  singleChannels, pairMask);
    pair = rebuilt.pair;
    single = rebuilt.single;
  } else {
    // ...the target features, once along each axis. See the note at the top about
    // which of "left" and "right" is which.
    const left = linear(input.targetFeat, tokens, weights.targetFeatWidth, pairChannels,
                        weights.leftTargetFeatProject);
    const right = linear(input.targetFeat, tokens, weights.targetFeatWidth, pairChannels,
                         weights.rightTargetFeatProject);
    const dgram = distogramFeatures(input.pseudoBeta, pairMask, tokens);
    const embedded = linear(dgram, pairs, DGRAM_BINS, pairChannels,
                            weights.distogramFeatProject);

    pair = Float32Array.from(input.pair);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const base = (i * tokens + j) * pairChannels;
        for (let c = 0; c < pairChannels; c += 1) {
          pair[base + c] += left[j * pairChannels + c] + right[i * pairChannels + c]
            + embedded[base + c];
        }
      }
    }
    single = Float32Array.from(input.single);
  }
  for (let index = 0; index < weights.blocks.length; index += 1) {
    const next = block({ pair, single, pairMask, seqMask, tokens },
                       weights.blocks[index], dialect);
    pair = next.pair;
    single = next.single;
  }

  // ...the distance-error head, SYMMETRISED by adding its own transpose. One
  // projection, used twice: AF3 sets `right = left` explicitly.
  // 🔴 A HEAD LayerNorm THE BUNDLE DOES NOT CARRY IS NO LayerNorm, not one at
  // scale 1: normalising still re-centres and rescales. boltz2 calls every
  // logit head directly on z and s.
  const headNorm = (x, rows, channels, scale, offset) =>
    (scale === undefined ? x : layerNorm(x, rows, channels, scale, offset));
  // 🔴 AND boltz2 SPLITS EACH PAIR HEAD IN TWO, intra-chain and inter-chain,
  // masked hard rather than blended - so on a MONOMER the inter head never
  // fires and a single-chain gate cannot see whether it is there at all.
  const splitHeads = weights.interHalfDistanceLogits !== undefined;
  const asymId = input.features?.asymId;
  const perPair = (intraLogits, interLogits) => {
    if (!splitHeads) return intraLogits;
    const out = new Float32Array(intraLogits.length);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const same = asymId === undefined || asymId[i] === asymId[j];
        const base = (i * tokens + j) * NUM_BINS;
        for (let bin = 0; bin < NUM_BINS; bin += 1) {
          out[base + bin] = same ? intraLogits[base + bin] : interLogits[base + bin];
        }
      }
    }
    return out;
  };
  // 🔴 boltz2 SYMMETRISES BEFORE THE PROJECTION, not after it. AF3 projects
  // once and adds the transpose of the result; the two agree only when the
  // projection is the same on both halves, which under split heads it is not.
  const preSymmetrised = dialect?.preSymmetrisedPde === true;
  const pdeSource = preSymmetrised
    ? (() => {
      const sum = new Float32Array(pair.length);
      for (let i = 0; i < tokens; i += 1) {
        for (let j = 0; j < tokens; j += 1) {
          const forward = (i * tokens + j) * pairChannels;
          const backward = (j * tokens + i) * pairChannels;
          for (let c = 0; c < pairChannels; c += 1) {
            sum[forward + c] = pair[forward + c] + pair[backward + c];
          }
        }
      }
      return sum;
    })()
    : pair;
  const normalisedPde = headNorm(pdeSource, pairs, pairChannels,
                                 weights.logitsLnScale, weights.logitsLnOffset);
  const half = linear(normalisedPde, pairs, pairChannels, NUM_BINS,
                      weights.leftHalfDistanceLogits);
  let distanceLogits;
  if (preSymmetrised) {
    distanceLogits = perPair(half, splitHeads
      ? linear(normalisedPde, pairs, pairChannels, NUM_BINS,
               weights.interHalfDistanceLogits)
      : half);
  } else {
    distanceLogits = new Float32Array(pairs * NUM_BINS);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const forward = (i * tokens + j) * NUM_BINS;
        const backward = (j * tokens + i) * NUM_BINS;
        for (let bin = 0; bin < NUM_BINS; bin += 1) {
          distanceLogits[forward + bin] = half[forward + bin] + half[backward + bin];
        }
      }
    }
  }
  const centres = errorBinCentres(NUM_BINS, MAX_ERROR_BIN);
  const pde = expectation(distanceLogits, pairs, NUM_BINS, centres);
  for (let index = 0; index < pairs; index += 1) pde[index] *= pairMask[index];

  // ...and the aligned-error head, which is NOT symmetrised: PAE is directional,
  // "how wrong is j when aligned on i".
  const normalisedPae = headNorm(pair, pairs, pairChannels,
                                 weights.paeLogitsLnScale, weights.paeLogitsLnOffset);
  const paeLogits = perPair(
    linear(normalisedPae, pairs, pairChannels, NUM_BINS, weights.paeLogits),
    weights.paeInterLogits === undefined ? undefined
      : linear(normalisedPae, pairs, pairChannels, NUM_BINS, weights.paeInterLogits));
  const pae = expectation(paeLogits, pairs, NUM_BINS, centres);
  for (let index = 0; index < pairs; index += 1) pae[index] *= pairMask[index];

  // pLDDT, per atom slot. The projection's output is (dense, bins) flattened.
  const plddtLogits = linear(
    headNorm(single, tokens, singleChannels, weights.plddtLnScale, weights.plddtLnOffset),
    tokens, singleChannels, dense * PLDDT_BINS, weights.plddtLogits);
  const width = 1 / PLDDT_BINS;
  const plddtCentres = new Float64Array(PLDDT_BINS);
  for (let bin = 0; bin < PLDDT_BINS; bin += 1) plddtCentres[bin] = 0.5 * width + bin * width;
  const plddt = expectation(plddtLogits, tokens * dense, PLDDT_BINS, plddtCentres);
  for (let index = 0; index < plddt.length; index += 1) plddt[index] *= 100;

  const resolvedLogits = linear(
    headNorm(single, tokens, singleChannels, weights.resolvedLnScale,
             weights.resolvedLnOffset),
    tokens, singleChannels, dense * 2, weights.experimentallyResolvedLogits);

  return { plddt, pae, pde, distanceLogits, paeLogits, resolvedLogits };
}
