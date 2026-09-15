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
 * 🔴 RoseTTAFold3 BINS THE SAME RANGE ONE MORE TIME AND COUNTS DIFFERENTLY.
 * AF3 has 39 EDGES and a bin per edge, the last catching everything past
 * 50.75; rf3 has 39 BOUNDARIES and 40 bins, the index being how many of them
 * the distance exceeds - so it has a bin BELOW 3.25 that AF3 has no equivalent
 * of, and its top bin starts one boundary later. The two are not a
 * reparameterisation of each other and the bundle says which: rf3's
 * `distogram_feat_project` is [40, 128] against [39, 128].
 */
const CA_DGRAM_BINS = 40;

/**
 * RoseTTAFold3's s_inputs width, where this port's `target_feat` is 447.
 *
 * The two missing columns are residue-vocabulary classes our alphabet does not
 * carry. They are zero on every input built here, which a PER-FEATURE norm
 * would not care about - but `maskedGlobalNorm` reduces ACROSS the feature
 * axis, so each of them still contributes `mean^2` to the variance sum, once
 * per real token. Upstream measured the correction at pae max|d| 0.047 -> 0.022.
 */
export const RF3_S_INPUTS_WIDTH = 449;

/**
 * A parameter-free LayerNorm over a WHOLE tensor, real tokens only.
 *
 * 🔴 THE MASK IS WHY THIS IS NOT A ONE-LINER. A statistic that reduces over
 * more than the feature axis is padding-sensitive in a way a per-feature one is
 * not: upstream measured a 76-residue chain padded into a 128-token bucket at
 * PAE ~28 A everywhere and pTM 0.04 against 0.89 for the same fold, purely from
 * counting the padding into the mean.
 *
 * @param {Float32Array} values `rows * channels`
 * @param {Float32Array} mask   one per ROW, broadcast over the channels
 * @param {number} vendorWidth  the width to normalise OVER, where it is wider
 *   than `channels`; the extra columns are assumed zero and so contribute
 *   `mean^2` apiece.
 */
export function maskedGlobalNorm(values, mask, rows, channels,
                                 vendorWidth = channels) {
  let live = 0;
  let total = 0;
  for (let row = 0; row < rows; row += 1) {
    if (!(mask[row] > 0)) continue;
    live += 1;
    const base = row * channels;
    for (let c = 0; c < channels; c += 1) total += values[base + c];
  }
  const count = Math.max(live * vendorWidth, 1);
  const mean = total / count;
  let variance = 0;
  for (let row = 0; row < rows; row += 1) {
    if (!(mask[row] > 0)) continue;
    const base = row * channels;
    for (let c = 0; c < channels; c += 1) {
      const d = values[base + c] - mean;
      variance += d * d;
    }
  }
  // ...each dropped column is zero, so it contributes mean^2, once per real row.
  variance += (vendorWidth - channels) * live * mean * mean;
  const inverse = 1 / Math.sqrt(variance / count + 1e-5);
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = (values[index] - mean) * inverse;
  }
  return output;
}

/** rf3's boundaries, and the bin is how many of them the distance is past. */
export function caDistogramFeatures(positions, pairMask, tokens) {
  const bounds = new Float64Array(CA_DGRAM_BINS - 1);
  for (let at = 0; at < CA_DGRAM_BINS - 1; at += 1) {
    bounds[at] = DGRAM_MIN + at * ((DGRAM_MAX - DGRAM_MIN) / (CA_DGRAM_BINS - 1));
  }
  const output = new Float32Array(tokens * tokens * CA_DGRAM_BINS);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      // ...the reference takes a real square root here and adds 1e-10 under it,
      // rather than comparing squares as AF3's does. Kept, because the boundary
      // arithmetic is what decides a bin and squaring the bounds moves the
      // comparison by an ulp at the edges.
      let squared = 1e-10;
      for (let axis = 0; axis < 3; axis += 1) {
        const difference = positions[i * 3 + axis] - positions[j * 3 + axis];
        squared += difference * difference;
      }
      const distance = Math.sqrt(squared);
      let bin = 0;
      for (let at = 0; at < bounds.length; at += 1) if (distance > bounds[at]) bin += 1;
      output[(i * tokens + j) * CA_DGRAM_BINS + bin] = pairMask[i * tokens + j];
    }
  }
  return output;
}

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
function boltz2Reembed(input, weights, tokens, pairChannels, singleChannels, pairMask,
                       onStage) {
  const w = weights.reembed;
  const pairs = tokens * tokens;
  const sInputs = layerNorm(input.targetFeat, tokens, weights.targetFeatWidth,
                            w.sInputsNormScale, w.sInputsNormOffset);
  const single = layerNorm(input.single, tokens, singleChannels,
                           w.sNormScale, w.sNormOffset);
  const fromInputs = linear(sInputs, tokens, weights.targetFeatWidth, singleChannels,
                            w.sInputToS);
  onStage?.("reembed.sInputsNorm", sInputs);
  onStage?.("reembed.sNorm", Float32Array.from(single));
  onStage?.("reembed.sInputToS", fromInputs);
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
  onStage?.("reembed.zNorm", Float32Array.from(pair));
  onStage?.("reembed.relPos", positioned);
  onStage?.("reembed.left", left);
  onStage?.("reembed.right", right);
  onStage?.("reembed.prod1", prodIn1);
  onStage?.("reembed.prod2", prodIn2);
  onStage?.("reembed.dgram", embedded);
  const product = new Float32Array(pairChannels);
  const traceProdOut = onStage === undefined
    ? null : new Float32Array(pairs * pairChannels);
  const traceBondType = onStage === undefined
    ? null : new Float32Array(pairs * pairChannels);
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
        let prodOut = 0;
        for (let e = 0; e < pairChannels; e += 1) {
          prodOut += product[e] * w.sToZProdOut[e * pairChannels + c];
        }
        if (traceProdOut !== null) {
          traceProdOut[base + c] = prodOut;
          traceBondType[base + c] = w.tokenBondsTypeEmbed[bondRow + c];
        }
        pair[base + c] += total + prodOut;
      }
    }
  }
  onStage?.("reembed.prodOut", traceProdOut);
  onStage?.("reembed.bondType", traceBondType);
  onStage?.("reembed.pair", pair);
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

  // 🔴 RoseTTAFold3 NORMALISES EVERY DETACHED TRUNK INPUT OVER THE WHOLE TENSOR
  // FIRST. Parameter-free, over REAL TOKENS ONLY, and it runs before anything
  // else here reads them. See `maskedGlobalNorm` and the flag's note in
  // dialect.js.
  let { pair: pairIn, single: singleIn, targetFeat } = input;
  if (dialect?.confidenceGlobalNorm === true) {
    pairIn = maskedGlobalNorm(pairIn, pairMask, pairs, pairChannels);
    singleIn = maskedGlobalNorm(singleIn, seqMask, tokens, singleChannels);
    // ...and target_feat over the VENDOR's width, which is two columns wider
    // than ours. See the flag's note.
    targetFeat = maskedGlobalNorm(targetFeat, seqMask, tokens,
                                  weights.targetFeatWidth, RF3_S_INPUTS_WIDTH);
  }
  input = { ...input, pair: pairIn, single: singleIn, targetFeat };

  // 🔴 ONE DIALECT REBUILDS z RATHER THAN ADDING TO IT; see `boltz2Reembed`.
  let pair;
  let single;
  if (dialect?.reembedConfidencePair === true) {
    const rebuilt = boltz2Reembed(input, weights, tokens, pairChannels,
                                  singleChannels, pairMask, input.onStage);
    pair = rebuilt.pair;
    single = rebuilt.single;
  } else {
    // ...the target features, once along each axis. See the note at the top about
    // which of "left" and "right" is which.
    // (protenix2's second, unbinned distance term is added below.)
    const left = linear(input.targetFeat, tokens, weights.targetFeatWidth, pairChannels,
                        weights.leftTargetFeatProject);
    const right = linear(input.targetFeat, tokens, weights.targetFeatWidth, pairChannels,
                         weights.rightTargetFeatProject);
    // 🔴 THE BIN COUNT IS THE WEIGHT'S, so a bundle that disagrees with the
    // dialect is a shape error rather than a silent prefix. See
    // caDistogramFeatures for what rf3's forty bins are.
    const caDgram = dialect?.confidenceCaDgram === true;
    const dgramBins = weights.distogramFeatProject.length / pairChannels;
    if (dgramBins !== (caDgram ? CA_DGRAM_BINS : DGRAM_BINS)) {
      throw new Error(`this bundle's distogram_feat_project has ${dgramBins} bins `
        + `and its dialect asks for ${caDgram ? CA_DGRAM_BINS : DGRAM_BINS}`);
    }
    const dgram = caDgram
      ? caDistogramFeatures(input.pseudoBeta, pairMask, tokens)
      : distogramFeatures(input.pseudoBeta, pairMask, tokens);
    const embedded = linear(dgram, pairs, dgramBins, pairChannels,
                            weights.distogramFeatProject);

    pair = Float32Array.from(input.pair);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const base = (i * tokens + j) * pairChannels;
        // 🔴 protenix2's SECOND distance term: a bias-free Linear on the RAW
        // distance, alongside the binned one. Unbinned, so it carries the
        // sub-bin resolution the one-hot throws away.
        let distance = 0;
        if (weights.distanceFeatProject !== undefined) {
          let squared = 1e-10;
          for (let axis = 0; axis < 3; axis += 1) {
            const d = input.pseudoBeta[i * 3 + axis] - input.pseudoBeta[j * 3 + axis];
            squared += d * d;
          }
          distance = Math.sqrt(squared);
        }
        for (let c = 0; c < pairChannels; c += 1) {
          pair[base + c] += left[j * pairChannels + c] + right[i * pairChannels + c]
            + embedded[base + c]
            + (weights.distanceFeatProject === undefined
              ? 0 : distance * weights.distanceFeatProject[c]);
        }
      }
    }
    // 🔴 AND THE TRUNK SINGLE IS NORMALISED BEFORE ANY USE, clamped first; see
    // `inputSingleNormScale` in weights.js. Both the pairformer and every head
    // read the normalised one.
    single = weights.inputSingleNormScale === undefined
      ? Float32Array.from(input.single)
      : layerNorm(Float32Array.from(input.single, (v) => Math.min(512, Math.max(-512, v))),
                  tokens, singleChannels,
                  weights.inputSingleNormScale, weights.inputSingleNormOffset);
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
