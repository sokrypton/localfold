/**
 * Which AF3-lineage graph a set of weights was trained for.
 *
 * 🔴 A DIALECT IS NOT A PREFERENCE, AND HAS NO DEFAULT. Every consumer of these
 * flags throws when one is missing rather than assuming stock AF3: a checkpoint
 * has to be read through the graph it was converted for, and each of these
 * differences is SILENT when wrong - the shapes all still agree, the fold still
 * comes out, and it is a slightly different model.
 *
 * 🔴 AND OPENBIND IS NOT OPENFOLD3, WHICH IS THE TRAP THIS TABLE EXISTS TO
 * STOP. OpenBind is OpenFold3's v0.5.0 release and it moved TOWARD AlphaFold 3
 * in two places its preview-2 weights differ:
 *
 *   - `swapTransposedBias`. OpenFold3 preview-2 computes a column attention's
 *     pair bias as `Linear(z[k, q])` - it transposes the pair representation
 *     BEFORE the projection - where AF3's Algorithm 15 says `Linear(z[q, k])`.
 *     v0.5.0 keeps that, so it stays TRUE for `openfold3` and would be wrong
 *     here. Upstream's own list is TRANSPOSED_COLUMN_PAIR_BIAS in
 *     ../alphafold3 `model_config.py`, and openbind is deliberately not in it.
 *   - the diffusion transformer's pair LayerNorm, which preview-2 runs once per
 *     block and v0.5.0 runs once for the whole stack, as AF3 does. Their
 *     release note: "Moved the pair layer norm in the diffusion transformer out
 *     of attention pair bias. The pair layer norm is run once to match the
 *     AlphaFold3 SI." So there is no flag for it - our transformer already does
 *     the AF3 thing, and `openfold3` is the release that would need one.
 *
 * Reading the OF3 porting notes and applying them wholesale to OpenBind gets
 * both of those backwards, which is why they are written down here rather than
 * left to be re-derived.
 *
 * What is NOT here is anything the weight converter can absorb. The residue
 * alphabet permutation, the i/j crossing between AF3's two pair-embedding
 * sites, the SwiGLU gate/value concatenation and the element index shift are
 * all row permutations of a weight matrix - `one_hot(e - 1) @ W` is exactly
 * `one_hot(e) @ W[max(0, arange - 1)]` - so they happen once, offline, and the
 * graph never learns about them.
 */

/** Stock AlphaFold 3, DeepMind's own parameters. */
export const ALPHAFOLD3 = Object.freeze({
  swapTransposedBias: false,
  symmetriseBonds: false,
  maskPaddedKeys: false,
  padSingleCondUnknownDna: false,
  pairInitFromSingle: false,
  msaUpdateBeforeOuterProduct: false,
  distogramBias: false,
  keyMaskedAtomAttention: false,
  perBlockPairLayerNorm: false,
  perBlockAtomPairLayerNorm: false,
  chainedAtomLayerNorm: false,
  splitPairConditioning: false,
  structuralTokens: false,
});

/**
 * OpenBind-0 - OpenFold3 v0.5.0, Apache 2.0.
 *
 * Three branches, and each one is a real difference in what the model computes:
 * the token bond matrix is symmetric, padded key atoms are excluded from the
 * atom-pair offset validity, and the diffusion single conditioning normalises
 * over 833 channels rather than 831. See each flag's use site.
 *
 * 🔴 THE RELEASE NUMBER IS IN THE NAME ON PURPOSE. Upstream calls this model
 * OpenBind-0, and their registry's bare `openbind` is a name a LATER release
 * would answer to as well - which is precisely how `openfold3` came to mean two
 * models with different forward conventions, and cost this port an afternoon of
 * reading notes about the wrong one. A dialect table is the last place that
 * ambiguity should be allowed to live.
 */
export const OPENBIND0 = Object.freeze({
  swapTransposedBias: false,
  symmetriseBonds: true,
  maskPaddedKeys: true,
  padSingleCondUnknownDna: true,
  pairInitFromSingle: false,
  msaUpdateBeforeOuterProduct: false,
  distogramBias: false,
  keyMaskedAtomAttention: false,
  perBlockPairLayerNorm: false,
  perBlockAtomPairLayerNorm: false,
  chainedAtomLayerNorm: false,
  splitPairConditioning: false,
  structuralTokens: false,
});


/**
 * OpenDDE (Aureka Research), Apache-2.0 - an independent PyTorch
 * reimplementation in the AlphaFold 3 family, with its own pairformer and
 * primitives rather than DeepMind's.
 *
 * 🔴 IT IS THE FIRST BUNDLE HERE WHOSE WIDTHS ARE NOT AlphaFold 3's, which is
 * why `src/af3/weights.js` derives every width from the tensor that states it
 * rather than declaring it. The pair track is 384 channels against AF3's 128,
 * the MSA 128 against 64, the triangle attention 12 heads against 4 (and 2 in
 * the template stack against 4), and the distogram 96 bins against 64. Every
 * one of those loads through a declared width without complaint.
 *
 * 🔴 AND ITS LINEAGE DOES NOT PREDICT ITS CONVENTIONS, WHICH IS THE TRAP THIS
 * TABLE EXISTS FOR - twice over, in opposite directions from OpenBind-0:
 *
 *   - `swapTransposedBias` is TRUE here and FALSE for OpenBind-0. Both are
 *     OpenFold3-lineage; upstream's `TRANSPOSED_COLUMN_PAIR_BIAS` lists
 *     opendde and deliberately omits openbind.
 *   - `padSingleCondUnknownDna` is FALSE here and TRUE for OpenBind-0, even
 *     though OpenDDE's native single conditioning is 833 wide exactly as
 *     OpenFold3's is. The converter collapses it to 831 by remapping the
 *     32-class vocabulary onto AF3's 31 rather than padding it (upstream's
 *     `converters/opendde.py`, `_remap_s_inputs_vec`), so what reaches this
 *     graph is 831 and stock AF3's arithmetic is correct. Reading "833 in the
 *     checkpoint" as "pad the conditioning" would LayerNorm over two columns
 *     the converter already folded away.
 *
 * Everything else here is a branch OpenBind-0 does not take. Each is silent
 * when wrong in the way this file's header describes - the shapes agree and a
 * structure comes out - so each is named at its use site and keyed into the
 * shader cache.
 */
export const OPENDDE = Object.freeze({
  // Upstream `TRANSPOSED_COLUMN_PAIR_BIAS`: a column attention's pair bias is
  // `Linear(z[k, q])`, the pair transposed BEFORE the projection.
  swapTransposedBias: true,
  // OPENFOLD3_LINEAGE, both of these.
  symmetriseBonds: true,
  maskPaddedKeys: true,
  // ...but NOT this one; see the note above.
  padSingleCondUnknownDna: false,
  // The pair track is initialised from the single embedding `s_init` rather
  // than from `target_feat`, so `single_activations` is computed BEFORE the
  // pair init instead of after the MSA stack, and left/right_single are
  // 384 -> pair rather than 447 -> pair. Their shapes say so.
  pairInitFromSingle: true,
  // An MSA block updates the MSA FIRST and feeds the UPDATED rows to the outer
  // product mean; AF3 takes the outer product off the pre-update MSA. The
  // difference compounds over the blocks.
  msaUpdateBeforeOuterProduct: true,
  // The distogram's half-logit projection carries a trained bias.
  distogramBias: true,
  // The atom attention's mask bias is an OR over the two masks rather than
  // AF3's AND, so a real query cannot attend to a padded key at all.
  keyMaskedAtomAttention: true,
  // The pair conditioning is LayerNormed and projected once per block in the
  // token transformer, and once per block in the atom transformer. AF3 runs
  // each once for the whole stack.
  perBlockPairLayerNorm: true,
  perBlockAtomPairLayerNorm: true,
  // The atom cross-attention's two adaptive LayerNorms are CHAINED: the keys'
  // normalisation reads the already-normalised queries, not the raw input.
  chainedAtomLayerNorm: true,
  // The diffusion conditioning compresses the trunk pair and the relative
  // encoding SEPARATELY to the pair width and concatenates them, rather than
  // projecting one concatenation of the pair and the RAW relative features.
  splitPairConditioning: true,
  // 🔴 AND THE DIFFUSION RUNS ON AN EXPANDED TOKEN SET, WHICH IS THE ONE
  // DIFFERENCE THAT IS NOT A BRANCH. Between the trunk and the diffusion
  // OpenDDE expands each residue into about two "structural tokens" - a
  // backbone token and a sidechain one, glycine staying single - and runs the
  // diffusion and its confidence head on those. That is a second token space
  // threaded through the atom layouts, not a flag, so this says only that the
  // bundle wants one. See `src/af3/structural-tokens.js`.
  structuralTokens: true,
});

export const DIALECTS = Object.freeze({
  alphafold3: ALPHAFOLD3,
  openbind0: OPENBIND0,
  opendde: OPENDDE,
});

/**
 * Names a bundle may carry that mean one of the dialects above.
 *
 * 🔴 UPSTREAM PUBLISHES THE BLOB AS `openbind`, and its own registry calls the
 * model that, so a bundle exported before this rename - or converted by
 * somebody following upstream's naming - says `openbind` in its manifest. That
 * has to keep resolving. It is an ALIAS and not a second dialect: both names
 * reach the same frozen object, so there is no way for them to drift apart.
 */
export const DIALECT_ALIASES = Object.freeze({ openbind: "openbind0" });

/**
 * The dialect a model name implies.
 *
 * 🔴 AN UNKNOWN NAME RAISES. The alternative is a new checkpoint silently
 * folding through stock AF3's graph, which produces a structure - a slightly
 * wrong one - rather than an error.
 */
export function dialectFor(model) {
  const dialect = DIALECTS[DIALECT_ALIASES[model] ?? model];
  if (dialect === undefined) {
    throw new Error(`no AF3 dialect for model ${JSON.stringify(model)}; `
      + `known: ${Object.keys(DIALECTS).join(", ")}`);
  }
  return dialect;
}


/** AF3's polymer restype classes: 20 amino acids, UNK, GAP, 4 RNA, 4 DNA, N. */
export const AF3_RESTYPES = 31;

/**
 * Where `features_1d` carries a column AF3 has no input for, in the
 * concatenation's own index space (`[trunk single | target_feat]`).
 *
 * 🔴 A ZERO COLUMN IS FREE BEFORE A BARE LINEAR AND IS NOT FREE BEFORE A
 * LAYERNORM, which is the whole reason this exists. OpenFold3's restype and
 * profile blocks carry 32 classes to AF3's 31 - AF3 folds unknown DNA into the
 * one unknown-nucleic class - and EVERYWHERE ELSE the extra class is simply
 * dropped from the converted weights, because a column that is always zero
 * contributes nothing to a matrix multiply. Here it cannot be: the diffusion
 * single conditioning LayerNorms this concatenation, and a LayerNorm maps a
 * zero input to -mean/std. So OpenFold3 always adds a trained contribution
 * through those two columns AND divides by 833 rather than 831. Upstream
 * measures dropping them at 2.2e-3 relative error against 3.4e-7 with them.
 *
 * The converter emits the projection's rows in this same padded order, so the
 * two must agree: the check that catches a disagreement is the LayerNorm
 * scale's own length, asserted at the use site.
 *
 * @param {{padSingleCondUnknownDna: boolean}} dialect
 * @param {number} seqChannels width of the trunk single block that comes first
 * @returns {number[]} padded indices that are always zero, ascending
 */
export function singleCondPadding(dialect, seqChannels) {
  if (dialect?.padSingleCondUnknownDna === undefined) {
    throw new Error("dialect.padSingleCondUnknownDna has no default: stock AF3 "
      + "is false, the openfold3 lineage true");
  }
  if (!dialect.padSingleCondUnknownDna) return [];
  // ...one after the restype block and one after the profile block, in the
  // padded index space - so the second already counts the first.
  return [seqChannels + AF3_RESTYPES, seqChannels + 2 * AF3_RESTYPES + 1];
}

/**
 * The source index a padded column reads, or -1 for one of the zero columns.
 *
 * Written as a loop over `singleCondPadding` rather than arithmetic so that the
 * CPU reference and the generated WGSL below cannot express it differently.
 */
export function singleCondSource(padding, index) {
  let source = index;
  for (const at of padding) {
    if (index === at) return -1;
    if (index > at) source -= 1;
  }
  return source;
}

/**
 * `singleCondSource` as WGSL, over the concatenation's reader.
 *
 * @param {number[]} padding from `singleCondPadding`
 */
export function singleCondPaddingWgsl(padding) {
  if (padding.length === 0) return "";
  return padding.map((at) => `  if (index == ${at}u) { return 0.0; }`).join("\n")
    + "\n"
    + padding.map((at) => `  if (index > ${at}u) { source -= 1u; }`).join("\n")
    + "\n";
}
