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
/**
 * 🔴 boltz2's MSA FEATURE IS 35 WIDE, NOT 34: it appends `is_paired`, which for
 * an unpaired alignment is 1 on the QUERY ROW and 0 everywhere else. Its
 * `msa_activations` is [35, 64] where every other model's is [34, 64], and
 * because this file built 34 columns the extra one was simply never read - a
 * silent prefix of the matrix, correct in its strides and missing a term. On a
 * single-sequence batch the query row IS the whole alignment, so the missing
 * term is the whole of what the MSA stack was given: `z_after_msa` read 3.16e-1
 * from af3-any-model's with the z-init exact at 5.05e-8.
 */
export function msaFeatures(rows, deletionMatrix, sequences, tokens, width = 34,
                            pairedQueryRow = false) {
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
    // The paired flag, where the model has one AND its convention puts the
    // query row in it. Row 0 is the query. rosettafold3 carries the column and
    // leaves it identically zero - see `msaPairedQueryRow` in dialect.js.
    if (width > 34 && pairedQueryRow && index < tokens) output[base + 34] = 1;
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

  // 🔴 OpenDDE INITIALISES THE PAIR FROM THE SINGLE EMBEDDING, NOT FROM
  // target_feat, so `single_activations` is computed HERE rather than after the
  // MSA stack and left/right_single are 384 -> pair rather than 447 -> pair.
  // Their shapes say which: AlphaFold 3's `left_single` is [447, 128] and
  // OpenDDE's is [384, 384]. Reading the dialect and the shape both, because a
  // bundle whose flag and weights disagree would otherwise multiply a 447-wide
  // feature by a 384-wide matrix and read off the end of neither.
  if (weights.dialect?.pairInitFromSingle === undefined) {
    throw new Error("weights.dialect.pairInitFromSingle has no default: stock "
      + "AF3 builds the pair from target_feat and OpenDDE from s_init");
  }
  const fromSingle = weights.dialect.pairInitFromSingle;
  const pairSource = fromSingle
    ? linear(targetFeat, tokens, featureWidth, singleChannels, weights.singleActivations)
    : targetFeat;
  const pairSourceWidth = fromSingle ? singleChannels : featureWidth;
  const left = linear(pairSource, tokens, pairSourceWidth, pairChannels,
                      weights.leftSingle);
  const right = linear(pairSource, tokens, pairSourceWidth, pairChannels,
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

  // 🔴 boltz2's TWO EXTRA z-INIT TERMS, BOTH CONSTANT ON AN UNCONSTRAINED
  // INPUT AND BOTH TRAINED NONZERO.
  //
  //   `token_bonds_type_embed` is an nn.Embedding over bond ORDER, and boltz
  //   numbers them 0 = no bond, 2 = single, 3 = double... So row 0 is a learned
  //   vector on every unbonded pair, which on a protein monomer is every pair.
  //   Applying row 0 EVERYWHERE is exact for a bond-free input and wrong by up
  //   to 8.1 on a bonded one, so the orders come from the bond matrix where
  //   there is one.
  //
  //   `contact_conditioning` is boltz's distance-restraint encoder. With no
  //   restraints the one-hot is the UNSPECIFIED class, `selected` is 1, and the
  //   encoder term is multiplied by zero: what is left is the learned constant
  //   `contact_encoding_unspecified`. That is not an approximation of the
  //   module - it is what the module computes on this input - but it IS a
  //   coverage limit, because this featuriser has no restraint field to carry.
  if (weights.tokenBondsTypeEmbed !== undefined) {
    const orders = input.bondOrderMatrix;
    for (let index = 0; index < pairs; index += 1) {
      const order = orders === undefined ? 0 : (orders[index] | 0);
      const row = (order >= 0 && order < 7 ? order : 0) * pairChannels;
      for (let c = 0; c < pairChannels; c += 1) {
        pair[index * pairChannels + c] += weights.tokenBondsTypeEmbed[row + c]
          + weights.contactEncodingUnspecified[c];
      }
    }
  }

  // ...and the template embedding, which reads the pair AS IT IS AT THIS POINT
  // - after the relative encoding and the bonds, before anything else. Passing
  // a function rather than an array is how src/af3/template-reference.js gets
  // that without this file having to know what a template is.
  const template = typeof input.templateEmbedding === "function"
    ? input.templateEmbedding(pair)
    : input.templateEmbedding;
  if (template === undefined) {
    throw new Error("templateEmbedding is required: AF3's template embedder"
      + " contributes even with no templates, so omitting it is not the same as"
      + " passing zeros");
  }
  for (let index = 0; index < pair.length; index += 1) pair[index] += template[index];

  const rows = sequences * tokens;
  // The width is the WEIGHT's, not a constant: boltz2's is 35 and everyone
  // else's is 34, and reading 34 off a [35, 64] matrix is a silent prefix.
  const msaFeatureWidth = weights.msaActivations.length / msaChannels;
  // ...and where that column exists, whether the QUERY ROW carries it. boltz2
  // says yes and rosettafold3 says no; see `msaPairedQueryRow` in dialect.js.
  if (msaFeatureWidth > 34 && weights.dialect?.msaPairedQueryRow === undefined) {
    throw new Error("weights.dialect.msaPairedQueryRow has no default: this "
      + "bundle carries an is_paired column and only the dialect says whether "
      + "the query row is in it");
  }
  const features = msaFeatures(input.msaRows, input.deletionMatrix, sequences, tokens,
                               msaFeatureWidth,
                               weights.dialect?.msaPairedQueryRow === true);
  const msa = linear(features, rows, msaFeatureWidth, msaChannels, weights.msaActivations);
  const fromTarget = linear(targetFeat, tokens, featureWidth, msaChannels,
                            weights.extraMsaTargetFeat);
  for (let s = 0; s < sequences; s += 1) {
    for (let t = 0; t < tokens; t += 1) {
      const base = (s * tokens + t) * msaChannels;
      for (let c = 0; c < msaChannels; c += 1) msa[base + c] += fromTarget[t * msaChannels + c];
    }
  }

  // ...and where the pair init already built it, it is the SAME tensor rather
  // than a second projection: upstream hoists the one call, it does not repeat
  // it. Recomputing would be correct arithmetically and wasteful; sharing is
  // what says the two really are one embedding.
  const single = fromSingle
    ? pairSource.slice()
    : linear(targetFeat, tokens, featureWidth, singleChannels,
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
