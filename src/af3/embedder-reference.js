/**
 * AF3's trunk embedder: what the two stacks are handed before they run.
 *
 *     pair    = left_single[i] + right_single[j]        from target_feat
 *     pair   += prev_embedding(LayerNorm(recycled pair))
 *     pair   += relative encoding
 *     pair   += bond embedding
 *     pair   += template embedding
 *     msa     = msa_activations(msa features) + extra_msa_target_feat(target_feat)
 *     single  = single_activations(target_feat)
 *     single += prev_single_embedding(LayerNorm(recycled single))
 *
 * 🔴 TWO INPUTS TO THIS ARE NOT COMPUTED HERE, AND BOTH ARE DELIBERATE STUBS.
 *
 *   target_feat        447 columns, of which 384 come from the atom transformer
 *                      encoder running over reference conformers. That is a
 *                      separate module with its own weights and its own
 *                      chemistry input, and it is the last real piece of AF3
 *                      still missing.
 *
 *   template embedding a two-block stack of its own.
 *
 * 🔴 AND THE TEMPLATE EMBEDDING IS NOT OPTIONAL, WHICH IS THE TRAP. On a de
 * novo protein with FOUR EMPTY TEMPLATE SLOTS its output still has std 13.1
 * against a pair whose own std is about 55 - because the summed embeddings run
 * through LayerNorms and biases that do not vanish when their inputs are
 * masked, and the stack divides by the template COUNT rather than by how many
 * are real. AF2-multimer had the identical trap and it cost this project a week:
 * "no templates" does not mean "no template contribution". Anything that treats
 * a missing template embedder as a zero will be about 25% wrong from the first
 * block and will still fold to something plausible.
 */
import { layerNorm, linear } from "./pairformer-reference.js";

/** AF3's relative encoding: 139 one-hot columns per token pair. */
export function relativeEncoding(tokens, features, maxRelativeIdx = 32,
                                 maxRelativeChain = 2) {
  const positionBins = 2 * maxRelativeIdx + 2;      // 66
  const chainBins = 2 * maxRelativeChain + 2;       // 6
  const width = positionBins * 2 + 1 + chainBins;   // 139
  const output = new Float32Array(tokens * tokens * width);
  const { residueIndex, tokenIndex, asymId, entityId, symId } = features;
  const clamp = (value, high) => Math.min(Math.max(value, 0), high);

  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const base = (i * tokens + j) * width;
      const sameChain = asymId[i] === asymId[j];
      const sameEntity = entityId[i] === entityId[j];

      // ...distance along the chain, with everything inter-chain sharing one
      // extra bin rather than being clipped into the far end of the range.
      const offset = clamp(residueIndex[i] - residueIndex[j] + maxRelativeIdx,
                           2 * maxRelativeIdx);
      output[base + (sameChain ? offset : 2 * maxRelativeIdx + 1)] = 1;

      // ...distance along the residue, which only means anything for two tokens
      // of the SAME residue of the same chain. For a protein every token is its
      // own residue, so this is the diagonal and one saturated bin elsewhere.
      const sameResidue = sameChain && residueIndex[i] === residueIndex[j];
      const tokenOffset = clamp(tokenIndex[i] - tokenIndex[j] + maxRelativeIdx,
                                2 * maxRelativeIdx);
      output[base + positionBins
        + (sameResidue ? tokenOffset : 2 * maxRelativeIdx + 1)] = 1;

      output[base + positionBins * 2] = sameEntity ? 1 : 0;

      // ...and which copy of a repeated chain this is, within its symmetry class.
      const relativeChain = clamp(symId[i] - symId[j] + maxRelativeChain,
                                  2 * maxRelativeChain);
      output[base + positionBins * 2 + 1
        + (sameEntity ? relativeChain : 2 * maxRelativeChain + 1)] = 1;
    }
  }
  return output;
}

/**
 * The 34 MSA feature columns: a 32-way one-hot plus two deletion channels.
 *
 * @param {Int32Array|Float32Array} rows      sequences * tokens, residue codes
 * @param {Float32Array} deletionMatrix       sequences * tokens
 */
export function msaFeatures(rows, deletionMatrix, sequences, tokens) {
  const width = 34;
  const output = new Float32Array(sequences * tokens * width);
  for (let index = 0; index < sequences * tokens; index += 1) {
    const base = index * width;
    const code = rows[index];
    if (code >= 0 && code < 32) output[base + code] = 1;
    const deletions = deletionMatrix[index];
    output[base + 32] = Math.min(Math.max(deletions, 0), 1);
    // ...arctan-squashed rather than clipped, so a column with many deletions
    // stays distinguishable from one with a few instead of saturating.
    output[base + 33] = Math.atan(deletions / 3) * (2 / Math.PI);
  }
  return output;
}

/**
 * Build the trunk's inputs.
 *
 * @param {{targetFeat: Float32Array, tokens: number, features: object,
 *          msaRows: ArrayLike<number>, deletionMatrix: ArrayLike<number>,
 *          sequences: number, templateEmbedding: Float32Array,
 *          previousPair?: Float32Array, previousSingle?: Float32Array}} input
 * @param {object} weights
 */
export function embed(input, weights) {
  const { tokens, targetFeat, sequences } = input;
  const pairChannels = weights.pairChannels;
  const singleChannels = weights.singleChannels;
  const msaChannels = weights.msaChannels;
  const featureWidth = weights.targetFeatWidth;
  const pairs = tokens * tokens;

  const left = linear(targetFeat, tokens, featureWidth, pairChannels,
                      weights.leftSingle);
  const right = linear(targetFeat, tokens, featureWidth, pairChannels,
                       weights.rightSingle);
  const pair = new Float32Array(pairs * pairChannels);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const base = (i * tokens + j) * pairChannels;
      for (let c = 0; c < pairChannels; c += 1) {
        pair[base + c] = left[i * pairChannels + c] + right[j * pairChannels + c];
      }
    }
  }

  // 🔴 THE RECYCLED TERM IS NOT ZERO ON THE FIRST PASS. AF3 starts from a zero
  // pair, but the LayerNorm ahead of the projection turns a zero input into its
  // OFFSET, and the projection of that offset is a constant this graph adds
  // every time. Skipping the whole branch "because there is nothing to recycle"
  // drops a term that is present on pass one.
  const previousPair = input.previousPair ?? new Float32Array(pairs * pairChannels);
  const recycled = linear(
    layerNorm(previousPair, pairs, pairChannels, weights.prevEmbeddingNormScale,
              weights.prevEmbeddingNormOffset),
    pairs, pairChannels, pairChannels, weights.prevEmbedding);
  for (let index = 0; index < pair.length; index += 1) pair[index] += recycled[index];

  const relative = relativeEncoding(tokens, input.features);
  const positioned = linear(relative, pairs, weights.relativeWidth, pairChannels,
                            weights.positionActivations);
  for (let index = 0; index < pair.length; index += 1) pair[index] += positioned[index];

  // ...the bond embedding reads a contact matrix that is identically zero for a
  // polymer with no covalent links, and its Linear is bias-free, so it adds
  // exactly nothing here. Measured, not assumed: the oracle's bond_embedding
  // output is 0.0000 on this input. It is a real term for ligands.
  if (input.bondMatrix !== undefined) {
    const bonds = linear(input.bondMatrix, pairs, 1, pairChannels, weights.bondEmbedding);
    for (let index = 0; index < pair.length; index += 1) pair[index] += bonds[index];
  }

  // ...and the template embedding, which is a stub. See the note at the top:
  // its contribution is large even when every template slot is empty.
  const template = input.templateEmbedding;
  if (template === undefined) {
    throw new Error("templateEmbedding is required: AF3's template embedder"
      + " contributes even with no templates, so omitting it is not the same as"
      + " passing zeros");
  }
  for (let index = 0; index < pair.length; index += 1) pair[index] += template[index];

  const rows = sequences * tokens;
  const features = msaFeatures(input.msaRows, input.deletionMatrix, sequences, tokens);
  const msa = linear(features, rows, 34, msaChannels, weights.msaActivations);
  const fromTarget = linear(targetFeat, tokens, featureWidth, msaChannels,
                            weights.extraMsaTargetFeat);
  for (let s = 0; s < sequences; s += 1) {
    for (let t = 0; t < tokens; t += 1) {
      const base = (s * tokens + t) * msaChannels;
      for (let c = 0; c < msaChannels; c += 1) msa[base + c] += fromTarget[t * msaChannels + c];
    }
  }

  const single = linear(targetFeat, tokens, featureWidth, singleChannels,
                        weights.singleActivations);
  const previousSingle = input.previousSingle
    ?? new Float32Array(tokens * singleChannels);
  const recycledSingle = linear(
    layerNorm(previousSingle, tokens, singleChannels,
              weights.prevSingleEmbeddingNormScale, weights.prevSingleEmbeddingNormOffset),
    tokens, singleChannels, singleChannels, weights.prevSingleEmbedding);
  for (let index = 0; index < single.length; index += 1) {
    single[index] += recycledSingle[index];
  }

  return { pair, msa, single };
}
