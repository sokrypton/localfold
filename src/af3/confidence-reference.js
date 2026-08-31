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

/** Bin centres for an error head: `bins - 1` edges, plus a catch-all. */
function errorBinCentres(bins, maxErrorBin) {
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
export function confidenceHead(input, weights, block, dialect) {
  const { tokens, dense, seqMask } = input;
  const pairChannels = weights.pairChannels;
  const singleChannels = weights.singleChannels;
  const pairs = tokens * tokens;

  const pairMask = new Float32Array(pairs);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }

  // ...the target features, once along each axis. See the note at the top about
  // which of "left" and "right" is which.
  const left = linear(input.targetFeat, tokens, weights.targetFeatWidth, pairChannels,
                      weights.leftTargetFeatProject);
  const right = linear(input.targetFeat, tokens, weights.targetFeatWidth, pairChannels,
                       weights.rightTargetFeatProject);
  const dgram = distogramFeatures(input.pseudoBeta, pairMask, tokens);
  const embedded = linear(dgram, pairs, DGRAM_BINS, pairChannels,
                          weights.distogramFeatProject);

  let pair = Float32Array.from(input.pair);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const base = (i * tokens + j) * pairChannels;
      for (let c = 0; c < pairChannels; c += 1) {
        pair[base + c] += left[j * pairChannels + c] + right[i * pairChannels + c]
          + embedded[base + c];
      }
    }
  }

  let single = Float32Array.from(input.single);
  for (let index = 0; index < weights.blocks.length; index += 1) {
    const next = block({ pair, single, pairMask, seqMask, tokens },
                       weights.blocks[index], dialect);
    pair = next.pair;
    single = next.single;
  }

  // ...the distance-error head, SYMMETRISED by adding its own transpose. One
  // projection, used twice: AF3 sets `right = left` explicitly.
  const half = linear(layerNorm(pair, pairs, pairChannels, weights.logitsLnScale,
                                weights.logitsLnOffset),
                      pairs, pairChannels, NUM_BINS, weights.leftHalfDistanceLogits);
  const distanceLogits = new Float32Array(pairs * NUM_BINS);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const forward = (i * tokens + j) * NUM_BINS;
      const backward = (j * tokens + i) * NUM_BINS;
      for (let bin = 0; bin < NUM_BINS; bin += 1) {
        distanceLogits[forward + bin] = half[forward + bin] + half[backward + bin];
      }
    }
  }
  const centres = errorBinCentres(NUM_BINS, MAX_ERROR_BIN);
  const pde = expectation(distanceLogits, pairs, NUM_BINS, centres);
  for (let index = 0; index < pairs; index += 1) pde[index] *= pairMask[index];

  // ...and the aligned-error head, which is NOT symmetrised: PAE is directional,
  // "how wrong is j when aligned on i".
  const paeLogits = linear(layerNorm(pair, pairs, pairChannels, weights.paeLogitsLnScale,
                                     weights.paeLogitsLnOffset),
                           pairs, pairChannels, NUM_BINS, weights.paeLogits);
  const pae = expectation(paeLogits, pairs, NUM_BINS, centres);
  for (let index = 0; index < pairs; index += 1) pae[index] *= pairMask[index];

  // pLDDT, per atom slot. The projection's output is (dense, bins) flattened.
  const plddtLogits = linear(
    layerNorm(single, tokens, singleChannels, weights.plddtLnScale, weights.plddtLnOffset),
    tokens, singleChannels, dense * PLDDT_BINS, weights.plddtLogits);
  const width = 1 / PLDDT_BINS;
  const plddtCentres = new Float64Array(PLDDT_BINS);
  for (let bin = 0; bin < PLDDT_BINS; bin += 1) plddtCentres[bin] = 0.5 * width + bin * width;
  const plddt = expectation(plddtLogits, tokens * dense, PLDDT_BINS, plddtCentres);
  for (let index = 0; index < plddt.length; index += 1) plddt[index] *= 100;

  const resolvedLogits = linear(
    layerNorm(single, tokens, singleChannels, weights.resolvedLnScale,
              weights.resolvedLnOffset),
    tokens, singleChannels, dense * 2, weights.experimentallyResolvedLogits);

  return { plddt, pae, pde, distanceLogits, paeLogits, resolvedLogits };
}
