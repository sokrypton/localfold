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
  preTrunkQuery: false,
  sampler: null,
  // 🔴 THE MSA MODULE ADDS ITS INPUT PAIR TWICE. boltz2's MSAModule RETURNS the
  // updated z - every MSALayer residual-updates it in place - and its caller
  // then does `z = z + msa_module(z, ...)`, so what reaches the pairformer is
  // `2 * z_in + delta` where AF3's is `z_in + delta`. Whether upstream meant it
  // does not matter: the weights were trained with it. Measured here as the MSA
  // stage reading 3.23e-1 from af3-any-model with the z-init exact at 5.05e-8,
  // and ours 14.13 against native's 20.38 - almost exactly one z_init of 6.59
  // short.
  msaDoubleAddPair: false,
  targetFeatAtomOnly: false,
  // 🔴 THE RESTYPE AN EMPTY TEMPLATE SLOT CARRIES, in the NINE-PROJECTION
  // embedder. Null means zero, which is this model's featuriser. See OPENDDE.
  // 🔴 CENTRE_REF_CONFORMERS: the reference conformers are centred per
  // `ref_space_uid` by every family's featuriser except stock AlphaFold 3's,
  // which is the reference implementation and keeps the uncentred CCD
  // ideals. It moves the RAW ref_pos channel only - the atom encoder also
  // reads a translation-invariant pairwise difference - which is why it
  // hid. See src/af3/featurise/featurise.js.
  centreRefConformers: false,
  emptyTemplateAatype: null,
  // 🔴 THE FUSED EMBEDDER'S FEATURE LAYOUT, for the models that HAVE a fused
  // embedder. protenix2's 108 columns are 39 distogram + 1 pseudo-beta mask +
  // 32 restype_i + 32 restype_j + 3 unit vector + 1 backbone frame mask, every
  // one of them a feature `templateGeometry` already computes for the
  // nine-projection path. boltz2's are 109 with 38 bins and 33 restypes - a
  // DIFFERENT feature set whose bin edges nothing here has measured - so it is
  // null and a supplied template refuses rather than guessing.
  // 🔴 BOLTZ-2 BUILDS ITS OWN 109 CHANNELS and they are not AF3's six
  // concatenated: 38 distogram bins on different edges, a unit vector that is
  // the element-wise SIGN of R_j^T (ca_i - t_j), a restype vocabulary shifted
  // by two over 33 classes, and restype_i varying along i where protenix2's
  // varies along j. See boltz2TemplateFeatures in template-features.js.
  // 🔴 RoseTTAFold3's 66 TEMPLATE COLUMNS, WHICH ARE NOT A TEMPLATE IN THE
  // OTHER TWO'S SENSE. A 64-bin CA-CA distance histogram over boundaries
  // `concat(arange(1, 4, 0.1), arange(4, 20.5, 0.5))`, a coverage flag and a
  // noise level - distance-distribution CONDITIONING, not a geometry
  // embedding, and it stops at 20 A where every other distogram here runs to
  // 50.75. It rides the same fused scopes as boltz2 and protenix2, so only
  // `a_proj`'s first dimension separates the three: 66 against 109 and 108.
  rosettafold3TemplateFeatures: false,
  templateFeatureMeanOnePass: false,
  boltz2TemplateFeatures: false,
  fusedTemplateLayout: null,
  emptyTemplateRestypeColumns: null,
  templateStackOuterResidual: false,
  templateVisibilityByCoverage: false,
  noHeadNorm: false,
  opmBiasAfterNorm: false,
  opmRowCountNorm: false,
  reembedConfidencePair: false,
  rawRefCharge: false,
  fusedTemplateEmbedder: false,
  projectedRelpos: false,
  preSymmetrisedPde: false,
  templateMeanOverAllSlots: false,
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
  // 🔴 THE FLAT ATOM AXIS IS RE-PADDED INSIDE EVERY ATOM BLOCK, which is
  // chai-1's and IntelliFold-2's. if2 reshapes its atoms into windows
  // (`b (n w) -> b n w`) inside each attention call and so pads with
  // `pad_at_dim(a_row, ..., value=0.)` every time: a block therefore starts
  // from a FRESHLY ZEROED padding, where AF3 carries whatever the previous
  // block's transition and residual left in those slots. A masked QUERY still
  // produces an output, and the next block GATHERS it as a key - so the
  // difference is not masked away, it propagates. The reference's signature:
  // with the window alignment right, if2's atom pair is exact on every window
  // and blocks 2 and 3 blow up on windows 16 and 17 ALONE, the two whose key
  // sets reach past the last real atom.
  // 🔴 THE MSA `is_paired` COLUMN IS PRESENT IN TWO MODELS AND MEANS TWO
  // DIFFERENT THINGS. boltz2's marks the QUERY ROW paired, so on an unpaired
  // alignment column 34 is 1 on row 0 and 0 elsewhere; RoseTTAFold3's
  // `add_residue_is_paired_feature` marks rows that were MSA-PAIRED ACROSS
  // CHAINS, which an unpaired alignment - all this codebase builds - never has,
  // so it is 0 EVERYWHERE INCLUDING THE QUERY. Confirmed upstream against the
  // native featurised batch, whose `msa_stack[..., 34]` is identically zero;
  // setting the query to 1 adds a constant bias to every MSA embedding. The
  // COLUMN's existence still comes off `msa_activations`' width, which is 35
  // in both and 34 in everything else - this flag decides only its VALUE.
  // 🔴 THE DIFFUSION BLOCK'S TRANSITION READS THE PRE-ATTENTION ACTIVATION,
  // and attention and transition share ONE residual add:
  // `act = act + attn + transition(act)` where AlphaFold 3 does `act += attn`
  // and then `act += transition(act)`. RoseTTAFold3's
  // `no_residual_connection_between_attention_and_transition`; chai-1's
  // parallel block is the same shape. It is not a reordering - the transition
  // sees a different tensor - and it costs one saved copy of the activation
  // per block, which at 68 tokens and 768 channels is 52k floats.
  // 🔴 WHERE THE ATOM KEY WINDOW SITS, WHICH IS NOT THE SAME QUESTION AS
  // WHETHER ITS PADDED SLOTS ARE MASKED. rf3, opendde and protenix CLAMP the
  // window - centred on `subset * 32 + 16`, a fixed offset range around it,
  // `clamp(index, 0, L - 1)` on the gather and the out-of-range slots masked
  // out of the attention - where AlphaFold 3 SLIDES it bodily in bounds so
  // every subset sees a full window of real keys. It is an end effect and it
  // is not small: af3-any-model measured its own version at 6MRR 0.767 ->
  // 0.737 on opendde. Upstream missed it on rf3 because rf3 was already in
  // KEY_MASKED_ATOM_ATTENTION, and that list is about the MASK.
  // 🔴 IntelliFold-2 SLIDES ITS WINDOW AGAINST A PADDED EDGE, which is neither
  // of the other two rules. It reshapes the flat atom axis into windows
  // (`b (n w) -> b n w`) and so pads to a whole 32-atom query block FIRST, so
  // the last window starts at `ceil(atoms / 32) * 32 - keys` rather than at
  // `atoms - keys`. On 6MRR that is 448 against 446 - its last two subsets take
  // keys 448..575 where AlphaFold 3's slide gives 446..573 - and against the
  // reference's own gather it is 125 of 128 indices wrong in each of them.
  // 🔴 FOUR FAMILIES HAVE NO TERMINAL OXT AND NO 5' OP3, AND IT SHIFTS THE FLAT
  // ATOM AXIS. boltz's canonical atom table is fixed and does not list one
  // (`const.ref_atoms["GLU"]` ends at OE2), its own CCD mol flags OXT
  // `leaving_atom: True`, and IntelliFold-2 forks that table whole;
  // rosettafold3's prediction path calls atomworks'
  // `remove_protein_terminal_oxygen` and the matching nucleic OP3 filter; and
  // OpenFold3/OpenBind-0 drop both. protenix and opendde deliberately KEEP the
  // OXT - protenix indexes it per residue and opendde appends it explicitly -
  // so this is not an OpenFold-lineage question and cannot be derived from one.
  //
  // 🔴 AND THE NUCLEIC HALF MATTERS MORE THAN THE PROTEIN HALF: OP3 is the
  // FIRST atom of residue 1, so carrying it shifts the ENTIRE flat atom axis of
  // a nucleic chain by one, where OXT only displaces a protein chain's tail. On
  // 6MRR it is one atom - 574 against 573 - and it moved the last two atom
  // windows of every one of these four against the reference's own gather.
  // 🔴 RoseTTAFold3's CONFIDENCE HEAD EMBEDS CA-CA DISTANCES IN 40 BINS, not
  // AlphaFold 3's CB-CB distogram in 39. Its boundaries are
  // `arange(39) * ((50.75 - 3.25) / 39) + 3.25` and the bin is the COUNT of
  // boundaries the distance exceeds, so there is a bin BELOW 3.25 that AF3 has
  // no equivalent of - and the positions are the token-centre CA (dense atom
  // index 1) rather than the pseudo-beta gather. The bundle states the width:
  // its `distogram_feat_project` is [40, 128] where every other model's is
  // [39, 128], so a wrong answer here is a shape error rather than a silent one.
  // 🔴 RoseTTAFold3's CONFIDENCE HEAD NORMALISES ITS INPUTS OVER THE WHOLE
  // TENSOR, not along the feature axis - a parameter-free LayerNorm applied to
  // each detached trunk input before use, where AlphaFold 3 takes them raw.
  //
  // 🔴 AND THE STATISTICS ARE OVER REAL TOKENS ONLY, which matters BECAUSE the
  // reduction spans more than the feature axis: a per-feature statistic cannot
  // see padding and this one can. Upstream measured the difference at PAE ~28 A
  // everywhere and pTM 0.04 against 0.89, for the same fold, from 76 residues
  // padded into a 128-token bucket.
  //
  // 🔴 AND `target_feat` IS NORMALISED OVER 449 COLUMNS WHERE THIS PORT HAS
  // 447. The two missing ones are residue-vocabulary classes our alphabet does
  // not carry; they are ZERO on every input we build, and a per-feature norm
  // would not care - but this statistic spans the feature axis, so each of them
  // still contributes `mean^2` to the variance. Upstream measured it at
  // pae max|d| 0.047 -> 0.022.
  confidenceGlobalNorm: false,
  // 🔴 RoseTTAFold3's CHIRAL CENTRES, AND THEY ARE NOT A NO-OP ON A PROTEIN.
  // 6MRR's 68 residues carry 213 of them in the reference's own batch. Every
  // other input the network takes is invariant under a mirror - distances,
  // frames, distograms all are - so the gradient of the improper-dihedral error
  // is the only thing that prefers an L amino acid to a D one, and upstream
  // records it asserting the L enantiomer over eleven D residues. See
  // `chiralCentres` for the rule and src/af3/diffusion/chiral-gradient.js for the term.
  chiralCentres: false,
  confidenceCaDgram: false,
  // 🔴 WITH NO ALIGNMENT, THIS FAMILY GETS THE QUERY ONCE AND AlphaFold 3 GETS
  // IT TWICE. AF3 concatenates a PAIRED and an UNPAIRED MSA block, so a chain
  // with no homologs contributes its own sequence to each and the model sees
  // depth 2; boltz's featuriser - which protenix, IntelliFold-2 and
  // RoseTTAFold3 all fork or match - emits a depth-1 `dummy_msa` and finds
  // nothing to pair.
  //
  // 🔴 COUNTED IN THE REFERENCE'S OWN BATCHES, NOT INFERRED: on 6MRR,
  // alphafold3, openbind0 and opendde carry TWO live MSA rows and boltz2,
  // protenix2, intellifold2 and rosettafold3 carry ONE. This port gave all
  // seven ONE, so the three that want two were folding a single sequence a row
  // short - and the pair-weighted averaging and the row transition are DEPTH
  // -sensitive where the outer product mean is not. Upstream measured the same
  // convention at 4.3% of esmfold2's MSA injection.
  //
  // 🔴 AND IT IS NAMED THE REFERENCE'S WAY ROUND, which keeps stock AlphaFold 3
  // as the all-false baseline this table's own test asserts. The featuriser's
  // option is `duplicateQueryRow` and reads the other way; `af3BatchFromA3m`
  // does the inversion, once, where the two names meet.
  dedupeSelfMsa: false,
  dropTerminalAtoms: false,
  // 🔴 THE ATOMISED-TOKEN CONVENTIONS, all three false here, which keeps this
  // dialect the all-false baseline test/af3-dialect.test.js asserts. They fire
  // only on a MODIFIED RESIDUE or a LIGAND, so 6MRR cannot see any of them -
  // see tools/check-batch-fields.js --target=gol-sep3, which is the dump that
  // can.
  atomizedElementNames: false,
  atomizedUnknownRestype: false,
  atomizedUnknownMsa: false,
  atomizedBackboneBonds: false,
  // Every other checkpoint's flow walk reaches a structure; see rosettafold3's.
  noFlowSampler: false,
  qblockAtomKeys: false,
  paddedAtomKeys: false,
  diffusionNoResidual: false,
  msaPairedQueryRow: false,
  maskAtomActPerBlock: false,
  // 🔴 RoseTTAFold3 SCALES THE TRIANGLE CONTRACTION BY 1/L, and it is
  // observable ONLY because the centre LayerNorm's epsilon does not commute
  // with a scale - which is also why it cannot be folded into a weight. See
  // src/kernels/triangle/shaders.js for where the GPU applies it and why there rather
  // than in the contraction.
  triangleMulDivideByLength: false,
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
  preTrunkQuery: false,
  sampler: null,
  // 🔴 THE MSA MODULE ADDS ITS INPUT PAIR TWICE. boltz2's MSAModule RETURNS the
  // updated z - every MSALayer residual-updates it in place - and its caller
  // then does `z = z + msa_module(z, ...)`, so what reaches the pairformer is
  // `2 * z_in + delta` where AF3's is `z_in + delta`. Whether upstream meant it
  // does not matter: the weights were trained with it. Measured here as the MSA
  // stage reading 3.23e-1 from af3-any-model with the z-init exact at 5.05e-8,
  // and ours 14.13 against native's 20.38 - almost exactly one z_init of 6.59
  // short.
  msaDoubleAddPair: false,
  targetFeatAtomOnly: false,
  // 🔴 THE RESTYPE AN EMPTY TEMPLATE SLOT CARRIES, in the NINE-PROJECTION
  // embedder. Null means zero, which is this model's featuriser. See OPENDDE.
  // 🔴 CENTRE_REF_CONFORMERS: the reference conformers are centred per
  // `ref_space_uid` by every family's featuriser except stock AlphaFold 3's,
  // which is the reference implementation and keeps the uncentred CCD
  // ideals. It moves the RAW ref_pos channel only - the atom encoder also
  // reads a translation-invariant pairwise difference - which is why it
  // hid. See src/af3/featurise/featurise.js.
  centreRefConformers: true,
  emptyTemplateAatype: null,
  // 🔴 THE FUSED EMBEDDER'S FEATURE LAYOUT, for the models that HAVE a fused
  // embedder. protenix2's 108 columns are 39 distogram + 1 pseudo-beta mask +
  // 32 restype_i + 32 restype_j + 3 unit vector + 1 backbone frame mask, every
  // one of them a feature `templateGeometry` already computes for the
  // nine-projection path. boltz2's are 109 with 38 bins and 33 restypes - a
  // DIFFERENT feature set whose bin edges nothing here has measured - so it is
  // null and a supplied template refuses rather than guessing.
  // 🔴 BOLTZ-2 BUILDS ITS OWN 109 CHANNELS and they are not AF3's six
  // concatenated: 38 distogram bins on different edges, a unit vector that is
  // the element-wise SIGN of R_j^T (ca_i - t_j), a restype vocabulary shifted
  // by two over 33 classes, and restype_i varying along i where protenix2's
  // varies along j. See boltz2TemplateFeatures in template-features.js.
  rosettafold3TemplateFeatures: false,
  templateFeatureMeanOnePass: false,
  boltz2TemplateFeatures: false,
  fusedTemplateLayout: null,
  emptyTemplateRestypeColumns: null,
  templateStackOuterResidual: false,
  templateVisibilityByCoverage: false,
  noHeadNorm: false,
  opmBiasAfterNorm: false,
  opmRowCountNorm: false,
  reembedConfidencePair: false,
  rawRefCharge: false,
  fusedTemplateEmbedder: false,
  projectedRelpos: false,
  preSymmetrisedPde: false,
  templateMeanOverAllSlots: false,
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
  confidenceGlobalNorm: false,
  chiralCentres: false,
  confidenceCaDgram: false,
  dedupeSelfMsa: false,
  dropTerminalAtoms: true,
  atomizedElementNames: false,
  atomizedUnknownRestype: false,
  atomizedUnknownMsa: false,
  atomizedBackboneBonds: false,
  // Every other checkpoint's flow walk reaches a structure; see rosettafold3's.
  noFlowSampler: false,
  qblockAtomKeys: false,
  paddedAtomKeys: false,
  diffusionNoResidual: false,
  msaPairedQueryRow: false,
  maskAtomActPerBlock: false,
  triangleMulDivideByLength: false,
  structuralTokens: false,
});


/**
 * OpenDDE (Aureka Research), Apache-2.0 - an independent PyTorch
 * reimplementation in the AlphaFold 3 family, with its own pairformer and
 * primitives rather than DeepMind's.
 *
 * 🔴 IT IS THE FIRST BUNDLE HERE WHOSE WIDTHS ARE NOT AlphaFold 3's, which is
 * why `src/af3/weights/weights.js` derives every width from the tensor that states it
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
  preTrunkQuery: false,
  sampler: null,
  // 🔴 THE MSA MODULE ADDS ITS INPUT PAIR TWICE. boltz2's MSAModule RETURNS the
  // updated z - every MSALayer residual-updates it in place - and its caller
  // then does `z = z + msa_module(z, ...)`, so what reaches the pairformer is
  // `2 * z_in + delta` where AF3's is `z_in + delta`. Whether upstream meant it
  // does not matter: the weights were trained with it. Measured here as the MSA
  // stage reading 3.23e-1 from af3-any-model with the z-init exact at 5.05e-8,
  // and ours 14.13 against native's 20.38 - almost exactly one z_init of 6.59
  // short.
  msaDoubleAddPair: false,
  targetFeatAtomOnly: false,
  // 🔴 AN EMPTY TEMPLATE SLOT CARRIES THE GAP RESTYPE, AND ONLY THE FIRST ONE.
  // OpenDDE takes protenix's featuriser, which "fills its one empty template
  // with the GAP restype and zero-pads the rest" - so `template_aatype` on a
  // query with NO template is 21 across slot 0 and 0 across slots 1..3, which
  // the reference's own batch dump shows exactly. Ours wrote 0 everywhere, and
  // an empty slot is NOT a no-op: the aatype one-hot picks a row out of
  // `template_pair_embedding_2`/`_3`, so row 0 (ALA) went in where row 21
  // belongs. Measured against af3-any-model on 6MRR, a query with no template
  // at all: the template module alone 2.83e-1 -> 1.32e-7, and the trunk seam
  // `z_after_template` 2.07e-2 -> exact, which was the last open OpenDDE
  // defect. AlphaFold 3, openbind0 and boltz2 write 0 in every slot; protenix2
  // shares this convention and already had it on the FUSED path as
  // `emptyTemplateRestypeColumns`.
  // 🔴 CENTRE_REF_CONFORMERS: the reference conformers are centred per
  // `ref_space_uid` by every family's featuriser except stock AlphaFold 3's,
  // which is the reference implementation and keeps the uncentred CCD
  // ideals. It moves the RAW ref_pos channel only - the atom encoder also
  // reads a translation-invariant pairwise difference - which is why it
  // hid. See src/af3/featurise/featurise.js.
  centreRefConformers: true,
  emptyTemplateAatype: 21,
  // 🔴 THE FUSED EMBEDDER'S FEATURE LAYOUT, for the models that HAVE a fused
  // embedder. protenix2's 108 columns are 39 distogram + 1 pseudo-beta mask +
  // 32 restype_i + 32 restype_j + 3 unit vector + 1 backbone frame mask, every
  // one of them a feature `templateGeometry` already computes for the
  // nine-projection path. boltz2's are 109 with 38 bins and 33 restypes - a
  // DIFFERENT feature set whose bin edges nothing here has measured - so it is
  // null and a supplied template refuses rather than guessing.
  // 🔴 BOLTZ-2 BUILDS ITS OWN 109 CHANNELS and they are not AF3's six
  // concatenated: 38 distogram bins on different edges, a unit vector that is
  // the element-wise SIGN of R_j^T (ca_i - t_j), a restype vocabulary shifted
  // by two over 33 classes, and restype_i varying along i where protenix2's
  // varies along j. See boltz2TemplateFeatures in template-features.js.
  rosettafold3TemplateFeatures: false,
  templateFeatureMeanOnePass: false,
  boltz2TemplateFeatures: false,
  fusedTemplateLayout: null,
  emptyTemplateRestypeColumns: null,
  templateStackOuterResidual: false,
  templateVisibilityByCoverage: false,
  noHeadNorm: false,
  opmBiasAfterNorm: false,
  opmRowCountNorm: false,
  reembedConfidencePair: false,
  rawRefCharge: false,
  fusedTemplateEmbedder: false,
  projectedRelpos: false,
  preSymmetrisedPde: false,
  templateMeanOverAllSlots: false,
  // Upstream `TRANSPOSED_COLUMN_PAIR_BIAS`: a column attention's pair bias is
  // `Linear(z[k, q])`, the pair transposed BEFORE the projection.
  swapTransposedBias: true,
  // OPENFOLD3_LINEAGE, both of these.
  symmetriseBonds: true,
  maskPaddedKeys: true,
  // ...but NOT this one; see the note above.
  // 🔴 IT JOINED THE PADDED LIST, AND THE BUNDLE HAD TO BE RE-EXPORTED FOR IT.
  // OpenDDE's diffusion single conditioning normalises over the vendor's 833
  // channels, not AF3's 831 - the two residue classes AF3 lacks are re-inserted
  // as ZERO columns before `single_cond_initial_norm`, and a LayerNorm maps a
  // zero input to -mean/std, so they are not free the way a zero column into a
  // bias-free Linear is. Worth a uniform 1 - sqrt(831/833) = 0.12% on the whole
  // single conditioning, which is exactly what the denoise oracle measured
  // (ours 2.8959 against 2.8994) before this. The converter emits the padded
  // 833-row scale and projection; a bundle exported before 2026-09-10 carries
  // 831 and the width assertion below is what says so.
  padSingleCondUnknownDna: true,
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
  // bundle wants one. See `src/af3/featurise/structural-tokens.js`.
  confidenceGlobalNorm: false,
  chiralCentres: false,
  confidenceCaDgram: false,
  dedupeSelfMsa: false,
  dropTerminalAtoms: false,
  atomizedElementNames: false,
  atomizedUnknownRestype: false,
  atomizedUnknownMsa: false,
  atomizedBackboneBonds: false,
  // Every other checkpoint's flow walk reaches a structure; see rosettafold3's.
  noFlowSampler: false,
  qblockAtomKeys: false,
  paddedAtomKeys: true,
  diffusionNoResidual: false,
  msaPairedQueryRow: false,
  maskAtomActPerBlock: false,
  triangleMulDivideByLength: false,
  structuralTokens: true,
});

/**
 * Protenix-v2 (ByteDance, Apache 2.0). Best-A 0.703 in the reference's table -
 * the strongest model this port can legally serve.
 *
 * 🔴 IT IS OPENDDE'S DIALECT WITH FOUR FLIPS, AND THAT IS THE WHOLE PORT.
 * Protenix-v2 sits in OPENFOLD3_LINEAGE exactly as OpenDDE does, so the lineage
 * branches - bond symmetrisation, the element index shift, trained Fourier
 * weights - are already here. Against OpenDDE its convention membership differs
 * in four places and only four:
 *
 *     msaUpdateBeforeOuterProduct    opendde true,  protenix2 FALSE
 *     splitPairConditioning          opendde true,  protenix2 true (see below)
 *     preSymmetrisedPde              opendde false, protenix2 TRUE
 *     templateMeanOverAllSlots       opendde false, protenix2 TRUE
 *
 * ...plus `structuralTokens`, OpenDDE's one difference that is not a flag at
 * all, which this model does not have.
 *
 * 🔴 AND `pairInitFromSingle` WAS SETTLED BY THE TENSOR, NOT BY THE TABLE.
 * OpenDDE builds the pair from `s_init` and its `left_single` is [384, 384];
 * this bundle's is **[447, 256]**, a target_feat-wide input, so it builds the
 * pair AlphaFold 3's way. The reference has no convention list for this - the
 * shape is the statement - which is why src/af3/trunk/embedder-reference.js reads the
 * flag AND the shape and refuses to default either.
 *
 * Every width is the tensor's and none is written down here: 48 trunk blocks of
 * 8 triangle heads at c_z 256, a 2-block template stack of 2 heads at 64, four
 * MSA blocks at c_m 128 with value_dim 8, and a 64-bin distogram whose
 * half-logit projection carries a bias. All of it matches the reference's
 * PROTENIX2_SETTINGS, which is the check that the derivation works rather than
 * a table this file has to keep in step.
 */
export const PROTENIX2 = Object.freeze({
  preTrunkQuery: false,
  sampler: null,
  // 🔴 THE MSA MODULE ADDS ITS INPUT PAIR TWICE. boltz2's MSAModule RETURNS the
  // updated z - every MSALayer residual-updates it in place - and its caller
  // then does `z = z + msa_module(z, ...)`, so what reaches the pairformer is
  // `2 * z_in + delta` where AF3's is `z_in + delta`. Whether upstream meant it
  // does not matter: the weights were trained with it. Measured here as the MSA
  // stage reading 3.23e-1 from af3-any-model with the z-init exact at 5.05e-8,
  // and ours 14.13 against native's 20.38 - almost exactly one z_init of 6.59
  // short.
  msaDoubleAddPair: false,
  targetFeatAtomOnly: false,
  templateStackOuterResidual: false,
  templateVisibilityByCoverage: false,
  noHeadNorm: false,
  opmBiasAfterNorm: false,
  opmRowCountNorm: false,
  reembedConfidencePair: false,
  rawRefCharge: false,
  // TRANSPOSED_COLUMN_PAIR_BIAS.
  swapTransposedBias: true,
  // OPENFOLD3_LINEAGE.
  symmetriseBonds: true,
  maskPaddedKeys: true,
  // 🔴 PADDED_SINGLE_COND, AND UNLIKE OpenDDE. Copied from OpenDDE's false and
  // the loader caught it in one run: "single conditioning is 831 channels but
  // its LayerNorm scale is 833". protenix2 carries the two unknown-DNA columns
  // where OpenDDE does not, which is why this is a flag and not a lineage
  // property - both models are OPENFOLD3_LINEAGE.
  padSingleCondUnknownDna: true,
  // 🔴 THE TENSOR SAYS SO: left_single is [447, 256], not [384, 384].
  pairInitFromSingle: false,
  // NOT in MSA_UPDATE_BEFORE_OPM - the outer product comes off the PRE-update
  // MSA, AlphaFold 3's way and not OpenDDE's.
  msaUpdateBeforeOuterProduct: false,
  // distogram_head/half_logits carries a bias.
  distogramBias: true,
  keyMaskedAtomAttention: true,
  perBlockPairLayerNorm: true,
  perBlockAtomPairLayerNorm: true,
  // "The opendde/protenix CHAINED form applies a norm TWICE, and composition
  // does not commute away" - the reference's own note.
  chainedAtomLayerNorm: true,
  // 🔴 NOT `splitPairConditioning`, AND THE GUARD IS WHAT SAID SO. Reading the
  // reference's DIFFUSION_PROJECTED_RELPOS membership, this was set true and
  // diffusion-weights.js threw at once: "this bundle does not carry
  // z_trunk_projection and its dialect says otherwise". There are THREE
  // conditioning shapes here, not two, and the tensors spell them out:
  //
  //     AF3        raw 139 relpos, trunk pair passed through   [267, 128]
  //     OpenDDE    BOTH projected (z_trunk_projection)         [256, 128]
  //     protenix2  relpe projected, trunk pair passed through  [512, 256]
  //
  // 512 is 256 + 256: `relpe_projection` is [139, 256] and there is no
  // `z_trunk_projection` at all. So `splitPairConditioning` is OpenDDE's
  // both-projected case and this is its own flag.
  splitPairConditioning: false,
  projectedRelpos: true,
  // No structural-token expansion; the diffusion runs on the trunk's tokens.
  confidenceGlobalNorm: false,
  chiralCentres: false,
  confidenceCaDgram: false,
  dedupeSelfMsa: true,
  dropTerminalAtoms: false,
  atomizedElementNames: false,
  atomizedUnknownRestype: false,
  atomizedUnknownMsa: false,
  atomizedBackboneBonds: false,
  // Every other checkpoint's flow walk reaches a structure; see rosettafold3's.
  noFlowSampler: false,
  qblockAtomKeys: false,
  paddedAtomKeys: true,
  diffusionNoResidual: false,
  msaPairedQueryRow: false,
  maskAtomActPerBlock: false,
  triangleMulDivideByLength: false,
  structuralTokens: false,
  // 🔴 THESE TWO ARE DECLARED AND NOT YET IMPLEMENTED, and saying so is the
  // point. `preSymmetrisedPde` symmetrises the PDE logits BEFORE the head
  // rather than after - the reference found it with confidence_parity.py
  // reading pde corr 0.87 while pae, plddt and resolved were all at parity, and
  // records that NO FOLD CAUGHT IT, because a symmetric plausibly-scaled error
  // metric stays symmetric and plausible. `templateMeanOverAllSlots` divides
  // the template term by every slot rather than the occupied ones. Until the
  // confidence head and the template embedder read them, a protenix2 fold is
  // right in its trunk and wrong in those two places.
  preSymmetrisedPde: true,
  templateMeanOverAllSlots: true,
  // 🔴 AN EMPTY SLOT'S 108 COLUMNS ARE ZERO EXCEPT TWO, MEASURED NOT ASSUMED.
  // `EMPTY=1 tools/oracle/dump_af3_template.py protenix2` reads every geometry
  // feature at exactly zero and both restype blocks one-hot at column 31 - GAP,
  // where _AF3_TO_OF3 sends AlphaFold 3's index 21. The blocks begin at 40 and
  // 72 in this model's order [disto(39), pb(1), restype_i(32), restype_j(32),
  // uvec(3), frame(1)]. boltz2's are all zero and its order is its own; see
  // BOLTZ2.
  // The FUSED embedder is this model's path and reads
  // `emptyTemplateRestypeColumns`; this is the same convention for the
  // nine-projection one, stated so the two cannot drift apart.
  // 🔴 CENTRE_REF_CONFORMERS: the reference conformers are centred per
  // `ref_space_uid` by every family's featuriser except stock AlphaFold 3's,
  // which is the reference implementation and keeps the uncentred CCD
  // ideals. It moves the RAW ref_pos channel only - the atom encoder also
  // reads a translation-invariant pairwise difference - which is why it
  // hid. See src/af3/featurise/featurise.js.
  centreRefConformers: true,
  emptyTemplateAatype: 21,
  // 🔴 THE FUSED EMBEDDER'S FEATURE LAYOUT, for the models that HAVE a fused
  // embedder. protenix2's 108 columns are 39 distogram + 1 pseudo-beta mask +
  // 32 restype_i + 32 restype_j + 3 unit vector + 1 backbone frame mask, every
  // one of them a feature `templateGeometry` already computes for the
  // nine-projection path. boltz2's are 109 with 38 bins and 33 restypes - a
  // DIFFERENT feature set whose bin edges nothing here has measured - so it is
  // null and a supplied template refuses rather than guessing.
  // 🔴 BOLTZ-2 BUILDS ITS OWN 109 CHANNELS and they are not AF3's six
  // concatenated: 38 distogram bins on different edges, a unit vector that is
  // the element-wise SIGN of R_j^T (ca_i - t_j), a restype vocabulary shifted
  // by two over 33 classes, and restype_i varying along i where protenix2's
  // varies along j. See boltz2TemplateFeatures in template-features.js.
  rosettafold3TemplateFeatures: false,
  templateFeatureMeanOnePass: false,
  boltz2TemplateFeatures: false,
  fusedTemplateLayout: { distogramBins: 39, restypes: 32 },
  emptyTemplateRestypeColumns: [40 + 31, 72 + 31],
  // 🔴 AND THE TEMPLATE EMBEDDER IS A DIFFERENT MODULE, NOT DIFFERENT WIDTHS.
  // protenix2 runs boltz2's fused form: `v = z_proj(z_norm(z)) + a_proj(a)`,
  // two pairformer blocks, `v_norm`, aggregate over slots, `u_proj(relu(u))`.
  // AF3's nine `template_pair_embedding_*` become one `a_proj` over a 108-wide
  // concatenation. See docs/AF3.md for the feature order, which has a trap in
  // it worth reading before implementing.
  fusedTemplateEmbedder: true,
});

/**
 * Boltz-2 (MIT). Best-A **0.430** in the reference's table - the strongest model
 * in it, and the reason this port was worth doing at all.
 *
 * 🔴 IT IS A BIGGER PORT THAN protenix2 AND SHARES ITS HARDEST PIECE. Both run
 * the FUSED template embedder - boltz2's own module, which protenix2 inherited -
 * so `fusedTemplateEmbedding` is already written and held to an oracle at
 * 1.52e-7. What boltz2 adds on top is an OUTER residual around that stack, which
 * protenix2 does not have, and the reference's note on it is worth repeating:
 * "protenix inherited the shared forward and got the wrong convention; rf3
 * escaped by not inheriting it. Either a per-vendor convention is named -- as it
 * now is here -- or the next subclass gets whichever behaviour its parent
 * happened to have."
 *
 * 🔴 SEVEN CONVENTIONS HERE ARE DECLARED AND NOT IMPLEMENTED. Listing them is
 * the point: a flag nothing reads is documentation, and a flag something reads
 * WRONGLY is a silent wrong model.
 *
 *   `opmRowCountNorm`   the outer product mean divides by the row COUNT.
 *                       🔴 IT NEEDS MSA DEPTH > 1 TO BITE: at depth 1 the bias
 *                       term is (1 - 1/1) * b = 0 and the two normalisers agree,
 *                       which is why boltz2's single-sequence 6MRR fold was
 *                       exact while its MSA module was not. A single-sequence
 *                       gate cannot see this one.
 *   `opmBiasAfterNorm`  where that bias enters, which is a separate question
 *                       from what the divisor counts.
 *   `noHeadNorm`        no LayerNorm before ANY confidence head.
 *   `reembedConfidencePair`  the confidence head REBUILDS its pair rather than
 *                       reading the trunk's: z_norm(z) + relpos + bonds + row,
 *                       column and outer product of s_inputs, plus a distance
 *                       embedding of the PREDICTED coordinates.
 *   `templateVisibilityByCoverage`  templates are masked by what the template
 *                       COVERS rather than by chain.
 *   `rawRefCharge`      the reference charge is not normalised.
 *   `templateStackOuterResidual`  above.
 *
 * And its sampler is its own - gamma_0 0.605, gamma_min 1.107, noise_scale
 * 0.901, step_scale 1.638, rho 8, sigma 0.0004..160 - which is NOT a dialect
 * flag and belongs with the sampler; see docs/AF3.md. Running it on AF3's
 * constants anneals on the wrong schedule and nothing errors.
 */
export const BOLTZ2 = Object.freeze({
  swapTransposedBias: true,
  symmetriseBonds: true,
  maskPaddedKeys: true,
  padSingleCondUnknownDna: false,
  // Settled by the tensors at export, as protenix2's was.
  pairInitFromSingle: false,
  // MSA_UPDATE_BEFORE_OPM, as OpenDDE.
  msaUpdateBeforeOuterProduct: true,
  distogramBias: true,
  keyMaskedAtomAttention: true,
  perBlockPairLayerNorm: true,
  // ...but NOT the atom-pair one, where OpenDDE and protenix2 both have it.
  perBlockAtomPairLayerNorm: false,
  // 🔴 NO CHAINED FORM, AND THE REFERENCE EXPLAINS WHY IT NEEDS NO BRANCH:
  // boltz2 gathers the ALREADY NORMALISED queries and carries one `adaln` per
  // layer, and adaptive LayerNorm is POINTWISE per atom - so gathering before
  // or after it is the same computation. The opendde/protenix chained form is
  // different precisely because it applies a norm TWICE.
  chainedAtomLayerNorm: false,
  splitPairConditioning: false,
  projectedRelpos: true,
  // ...and boltz2's IS the query row; see the flag's note on ALPHAFOLD3.
  confidenceGlobalNorm: false,
  chiralCentres: false,
  confidenceCaDgram: false,
  dedupeSelfMsa: true,
  dropTerminalAtoms: true,
  atomizedElementNames: false,
  atomizedUnknownRestype: true,
  atomizedUnknownMsa: false,
  atomizedBackboneBonds: false,
  // Every other checkpoint's flow walk reaches a structure; see rosettafold3's.
  noFlowSampler: false,
  qblockAtomKeys: false,
  paddedAtomKeys: true,
  // ...and boltz2 CLIPS AND PADS, which reaches the same window: upstream
  // lists "boltz2 slid the atom key window where it CLIPS AND PADS" among its
  // own port bugs, and against its gather this takes boltz2 from 134 mask and
  // 378 index differences to 2 and 0 - the 2 being an atom count, not a window,
  diffusionNoResidual: false,
  msaPairedQueryRow: true,
  maskAtomActPerBlock: false,
  triangleMulDivideByLength: false,
  structuralTokens: false,
  preSymmetrisedPde: true,
  templateMeanOverAllSlots: false,
  fusedTemplateEmbedder: true,
  // 🔴 AN EMPTY SLOT'S 109 COLUMNS ARE ALL ZERO HERE, where protenix2's carry
  // GAP. The reference records the split - "protenix fills the first slot with
  // GAP, opendde fills all four, intellifold2 deliberately uses 0", and boltz2
  // is with the last - and it is measurable rather than inferable, which is why
  // it is a dialect entry and not a rule.
  // 🔴 ITS s_inputs IS THE ATOM ENCODER'S 384 COLUMNS AND NOTHING ELSE, where
  // every other model here prepends a restype one-hot, a profile and a deletion
  // mean for 447. The reference states it plainly - "s_trunk is concatenated
  // with it, not with a 449-channel target_feat" - and the shape says the same:
  // boltz2's single conditioning reads 768 = 384 + 384 where AF3's reads
  // 831 = 447 + 384.
  // 🔴 ITS OWN EDM SCHEDULE, AND NOTHING ERRORS IF IT IS NOT USED. The
  // reference keeps these per model and says why: "running them on AF3's
  // constants would anneal on the wrong schedule ... nothing errors, it just
  // anneals differently and returns a plausible structure". boltz2's are not
  // small differences - gamma_0 0.605 against 0.8, step_scale 1.638 against
  // 1.5, rho 8 against 7 - and on AF3's the fold comes out with 9.6 A backbone
  // bonds against an ideal 1.46.
  //
  // 🔴 AND THIS IS NOT A DIALECT BRANCH. It is a table of constants that rides
  // here because the dialect is what a bundle already resolves to; the sampler
  // reads it and the forward graph never sees it.
  sampler: Object.freeze({
    gamma0: 0.605, gammaMin: 1.107, noiseScale: 0.901, stepScale: 1.638,
    rho: 8.0, sigmaMin: 0.0004, sigmaMax: 160.0,
  }),
  // 🔴 THE MSA MODULE ADDS ITS INPUT PAIR TWICE. boltz2's MSAModule RETURNS the
  // updated z - every MSALayer residual-updates it in place - and its caller
  // then does `z = z + msa_module(z, ...)`, so what reaches the pairformer is
  // `2 * z_in + delta` where AF3's is `z_in + delta`. Whether upstream meant it
  // does not matter: the weights were trained with it. Measured here as the MSA
  // stage reading 3.23e-1 from af3-any-model with the z-init exact at 5.05e-8,
  // and ours 14.13 against native's 20.38 - almost exactly one z_init of 6.59
  // short.
  msaDoubleAddPair: true,
  targetFeatAtomOnly: true,
  // The FUSED embedder is this model's path and reads
  // `emptyTemplateRestypeColumns`; this is the same convention for the
  // nine-projection one, stated so the two cannot drift apart.
  // 🔴 CENTRE_REF_CONFORMERS: the reference conformers are centred per
  // `ref_space_uid` by every family's featuriser except stock AlphaFold 3's,
  // which is the reference implementation and keeps the uncentred CCD
  // ideals. It moves the RAW ref_pos channel only - the atom encoder also
  // reads a translation-invariant pairwise difference - which is why it
  // hid. See src/af3/featurise/featurise.js.
  centreRefConformers: true,
  emptyTemplateAatype: null,
  // 🔴 THE FUSED EMBEDDER'S FEATURE LAYOUT, for the models that HAVE a fused
  // embedder. protenix2's 108 columns are 39 distogram + 1 pseudo-beta mask +
  // 32 restype_i + 32 restype_j + 3 unit vector + 1 backbone frame mask, every
  // one of them a feature `templateGeometry` already computes for the
  // nine-projection path. boltz2's are 109 with 38 bins and 33 restypes - a
  // DIFFERENT feature set whose bin edges nothing here has measured - so it is
  // null and a supplied template refuses rather than guessing.
  // 🔴 BOLTZ-2 BUILDS ITS OWN 109 CHANNELS and they are not AF3's six
  // concatenated: 38 distogram bins on different edges, a unit vector that is
  // the element-wise SIGN of R_j^T (ca_i - t_j), a restype vocabulary shifted
  // by two over 33 classes, and restype_i varying along i where protenix2's
  // varies along j. See boltz2TemplateFeatures in template-features.js.
  rosettafold3TemplateFeatures: false,
  templateFeatureMeanOnePass: false,
  boltz2TemplateFeatures: true,
  fusedTemplateLayout: null,
  emptyTemplateRestypeColumns: [],
  templateStackOuterResidual: true,
  templateVisibilityByCoverage: true,
  noHeadNorm: true,
  opmBiasAfterNorm: true,
  opmRowCountNorm: true,
  reembedConfidencePair: true,
  rawRefCharge: true,
  // 🔴 ITS QUERIES ARE THE PER-ATOM FEATURES BEFORE s_trunk, WHILE ITS
  // CONDITIONING IS AFTER. AlphaFold 3 uses one array for both - the query
  // activation starts as a copy of the conditioning - and boltz2, rosettafold3
  // and chai1 need them split: q reads `a`, c reads `a + token_to_atom(s_trunk)`.
  preTrunkQuery: true,
});

/**
 * IntelliFold-2 - intelligenAI, Apache 2.0.
 *
 * 🔴 THE SMALLEST DIALECT IN THIS FILE, AND THAT IS THE FINDING. Its module
 * tree is stock AlphaFold 3's - the nine-projection template embedder, AF3's
 * MSA stack, AF3's confidence head, AF3's diffusion transformer - and the
 * reference's convention table puts it in exactly TWO lists:
 * KEY_MASKED_ATOM_ATTENTION and MASK_ATOM_ACT_PER_BLOCK. It is deliberately
 * NOT in OPENFOLD3_LINEAGE: it forks boltz's FEATURISER, not OpenFold's
 * network, which is why the bonds are not symmetrised here and the reference
 * conformers are not centred while three of its featuriser conventions are
 * boltz's. Read the featuriser flags below as the other half of the port.
 *
 * 🔴 AND THE TWO IT IS IN ARE ONE BUG BETWEEN THEM. The reference records the
 * pair: with the atom key window aligned and the key-side mask wrong, block 1
 * reads 9.5e-03 and blocks 2-3 blow up on windows 16 and 17 alone; with both,
 * the whole encoder is exact at full depth (a_token 2.33e-05, q_atom
 * 1.23e-05). Neither is visible without the other, so shipping one is worse
 * than shipping neither - a partial fix moves the error and hides its cause.
 *
 * 🔴 THREE FEATURISER CONVENTIONS, WHICH ARE NOT DIALECT FLAGS HERE AND ARE
 * STILL DIFFERENCES. They are recorded so nobody re-derives them:
 *
 *   qblock_keys       the atom key window's edge is the atom count ROUNDED UP
 *                     to a whole 32-atom query block, because if2 reshapes the
 *                     flat atom axis into windows and must pad first. On 6MRR
 *                     its last two blocks take keys 448..575 where AF3's slide
 *                     gives 446..573.
 *   dedupe_self_msa   a chain with no alignments gets a DEPTH-1 MSA, where AF3
 *                     concatenates a paired and an unpaired copy and so hands
 *                     the query TWICE. The outer product mean over duplicates
 *                     is unchanged; the pair-weighted averaging and the row
 *                     transition are depth-sensitive.
 *   drop_atoms        no terminal OXT and no 5' OP3 - boltz's fixed atom
 *                     tables, which if2 forks whole.
 */
export const INTELLIFOLD2 = Object.freeze({
  preTrunkQuery: false,
  sampler: null,
  msaDoubleAddPair: false,
  targetFeatAtomOnly: false,
  centreRefConformers: false,
  emptyTemplateAatype: null,
  rosettafold3TemplateFeatures: false,
  templateFeatureMeanOnePass: false,
  boltz2TemplateFeatures: false,
  fusedTemplateLayout: null,
  emptyTemplateRestypeColumns: null,
  templateStackOuterResidual: false,
  templateVisibilityByCoverage: false,
  noHeadNorm: false,
  opmBiasAfterNorm: false,
  opmRowCountNorm: false,
  reembedConfidencePair: false,
  rawRefCharge: false,
  fusedTemplateEmbedder: false,
  projectedRelpos: false,
  preSymmetrisedPde: false,
  templateMeanOverAllSlots: false,
  swapTransposedBias: false,
  // NOT in OPENFOLD3_LINEAGE - the bonds are AF3's.
  symmetriseBonds: false,
  maskPaddedKeys: false,
  padSingleCondUnknownDna: false,
  pairInitFromSingle: false,
  msaUpdateBeforeOuterProduct: false,
  distogramBias: false,
  // The atom attention's mask bias is an OR over the two masks, so a real
  // query cannot attend to a padded key at all.
  keyMaskedAtomAttention: true,
  perBlockPairLayerNorm: false,
  perBlockAtomPairLayerNorm: false,
  chainedAtomLayerNorm: false,
  splitPairConditioning: false,
  // ...and the other half of that pair: every atom block starts from a freshly
  // zeroed padding. See the flag's note on ALPHAFOLD3.
  confidenceGlobalNorm: false,
  chiralCentres: false,
  confidenceCaDgram: false,
  dedupeSelfMsa: true,
  dropTerminalAtoms: true,
  atomizedElementNames: false,
  atomizedUnknownRestype: false,
  atomizedUnknownMsa: false,
  atomizedBackboneBonds: false,
  // Every other checkpoint's flow walk reaches a structure; see rosettafold3's.
  noFlowSampler: false,
  qblockAtomKeys: true,
  paddedAtomKeys: false,
  diffusionNoResidual: false,
  msaPairedQueryRow: false,
  maskAtomActPerBlock: true,
  triangleMulDivideByLength: false,
  structuralTokens: false,
});

/**
 * RoseTTAFold3 - Baker lab / RosettaCommons foundry, BSD 3-Clause.
 *
 * 🔴 THE LARGEST DIALECT IN THIS FILE, AND MOST OF IT IS NOT A FLAG YET. Ten
 * of the reference's convention tuples name it, and those are the easy half:
 * eleven more divergences are gated on the model NAME directly, which means
 * they are branches nobody expressed as a convention because only rf3 has
 * them. They are listed here in full, because a flag nothing reads is
 * documentation and a MISSING flag is a silently different model.
 *
 * ITS SHAPES ARE STOCK AF3's - c_z 128, four triangle-attention heads, triangle
 * multiplication hidden 128 - so unlike protenix2 there is no trunk widening.
 * Two config divergences only: a 65-bin distogram (AF3's is 64) and an MSA
 * module that holds ONE set of weights and runs it four times. That second one
 * is weight TYING, not a single iteration; the converter replicates the block
 * across all four layers, so this port sees four blocks and needs no branch.
 *
 * IMPLEMENTED HERE:
 *
 *   triangleMulDivideByLength  the contraction is scaled by 1/L. See the flag's
 *                       note on ALPHAFOLD3 and src/kernels/triangle/shaders.js.
 *
 * DECLARED AND NOT IMPLEMENTED - every one of these is a real difference:
 *
 *   gridAttentionBias   rf3's triangle-attention `gating_query` and
 *                       `output_projection` carry TRAINED BIASES that stock
 *                       AF3's GridSelfAttention has no slot for. The gate's
 *                       bias initialises to 1.0 and its weight to zeros, so the
 *                       gate is bias-DOMINATED: dropping it roughly halves the
 *                       gate, and it compounds across 96 triangle attentions.
 *   outerProductBias    `left_projection` and `right_projection` are biased
 *                       too, and dropping them is not a constant offset: the
 *                       outer product is bilinear, so the two cross terms go
 *                       with the bias. Measured upstream at corr 0.924 for the
 *                       MSA module's outer product with exact inputs.
 *   msaIsPairedColumn   the MSA feature is 35 columns - onehot(32), has_del,
 *                       del_val, is_paired - where ours is 34. 🔴 AND THE FLAG
 *                       IS ZERO EVERYWHERE INCLUDING THE QUERY ROW, where
 *                       boltz2's is 1 on the query: rf3's marks rows PAIRED
 *                       ACROSS CHAINS, which an unpaired alignment never has.
 *                       Setting the query to 1 adds a constant bias to every
 *                       MSA embedding.
 *   rf3TemplateFeatures its template module IS the fused one structurally -
 *                       `a_proj` is in the bundle - but its 66 columns are a
 *                       64-bin CA-CA distance distribution plus has_condition
 *                       and noise_level, not the distogram/restype/unit-vector
 *                       set protenix2 and boltz2 concatenate. An empty slot is
 *                       exact; a supplied one refuses.
 *   diffusionKqNorm     a q/k LayerNorm inside the diffusion attention, in both
 *                       the token transformer and the atom one.
 *   conformerEmbeddingBias  the atom single embedding takes a constant vector
 *                       from an MLP whose input is all zeros here - the MLP has
 *                       biases and a LayerNorm tail, so a zero feature is NOT a
 *                       zero contribution, and it is two thirds the magnitude
 *                       of the ref-feature embedding.
 *   chiralQueryTerm     the diffusion atom encoder adds the gradient of the
 *                       chiral-centre dihedral error to the query. It is the
 *                       only reflection-asymmetric signal in the network.
 *                       🔴 AND NOT A NO-OP ON A PROTEIN, which this line
 *                       claimed and which was wrong. Backbone stereochemistry
 *                       is the bulk of it: 6MRR, a plain 68-mer, carries 213
 *                       centres - a CA with (N, C, CB) for each of the 62
 *                       non-glycines and a CB for the ILE and THR, each in
 *                       three permutations. Implemented and gated. Worth
 *                       nothing measurable on a structure that is ALREADY
 *                       L-chiral, which is a different claim: see
 *                       test/chiral-gradient.test.js, where a mirrored one
 *                       costs more.
 *   confidenceGlobalNorm  the confidence head applies a PARAMETER-FREE
 *                       LayerNorm over the whole tensor to each detached trunk
 *                       input. 🔴 OVER REAL TOKENS ONLY: a mean and a variance
 *                       taken over the padded tensor move with the bucket size.
 *   confidenceCaDgram   that head embeds CA-CA distances in 40 bins over 39
 *                       boundaries 3.25..50.75, not AF3's CB-CB dgram.
 *   distogramBins65     the distogram head has 65 bins, not 64.
 *
 * AND FOUR FEATURISER CONVENTIONS, which are the batch's rather than the
 * graph's: `chirals`, `atomized_element_names` (atomised atoms are renamed to
 * their element symbol), `restype_alignment` (DECLARED AND MEASURED INERT - see docs/AF3.md: the
 * reference rewrites profile and msa for a token whose alignment column is
 * gap in EVERY row, and its MSA is padded to 16384 rows with 0 rather than
 * the gap index 21, so the predicate is false on every batch it produces.
 * Not implemented here, which is why the batch matches), `padded_keys`
 * (the atom key window is CLAMPED and the out-of-range slots masked, where AF3
 * slides the window bodily in bounds - the same convention as opendde and
 * protenix, and it was missed upstream because rf3 was already in
 * KEY_MASKED_ATOM_ATTENTION and that list is about the MASK), plus the terminal
 * OXT/OP3 drop and the self-MSA dedupe that if2 also has.
 *
 * 🔴 AND A RETRACTION THAT WAS MADE EVERYWHERE BUT HERE. This block used to
 * say the reference's converter was "incomplete on the atom path", on the
 * strength of a one-line WIP note in its `model_registry.py`. That note is
 * STALE: af3-any-model's own PARITY.md gates rosettafold3 at L0/L1/L3/L4/L5/L6
 * and its L5 log folds 6MRR at best 0.967 and mean 1.546. The claim was
 * retracted in f4f6b9e and docs/AF3.md and CLAUDE.md were corrected; this copy
 * was missed, which is an argument for one description of a model rather than
 * four. An rf3 dump can be trusted throughout.
 */
export const ROSETTAFOLD3 = Object.freeze({
  preTrunkQuery: true,
  sampler: null,
  msaDoubleAddPair: false,
  targetFeatAtomOnly: false,
  // NOT in CENTRE_REF_CONFORMERS.
  centreRefConformers: false,
  emptyTemplateAatype: null,
  rosettafold3TemplateFeatures: true,
  // 🔴 ONE FORWARD OVER THE MEAN OF THE PRESENT TEMPLATES' FEATURES, where
  // every other family runs a forward per SLOT and averages the outputs.
  // The reference is explicit: `a_tij = einsum('t,tijc->ijc', present, feats)
  // / clip(present.sum(), 1)`, then a single pass with "no per-template loop
  // and no template gating". Averaging over four padded SLOTS instead makes
  // one real template a QUARTER of what the checkpoint expects. See
  // template-webgpu.js, and docs/AF3.md for what that cost the fold.
  templateFeatureMeanOnePass: true,
  boltz2TemplateFeatures: false,
  fusedTemplateLayout: null,
  emptyTemplateRestypeColumns: null,
  templateStackOuterResidual: false,
  templateVisibilityByCoverage: false,
  noHeadNorm: false,
  opmBiasAfterNorm: true,
  opmRowCountNorm: false,
  reembedConfidencePair: false,
  rawRefCharge: true,
  // 🔴 THE FUSED MODULE, WITH A FEATURE SET NOTHING HERE BUILDS. Its converter
  // emits `a_proj`, so structurally it is boltz2's and protenix2's embedder -
  // and `a_proj` is [66, 64] where protenix2's is [108, 64] and boltz2's
  // [109, 64], because rf3's 66 columns are a 64-bin CA-CA distance
  // distribution plus has_condition and noise_level rather than a distogram,
  // restype one-hots, a unit vector and a frame mask. So the STACK runs and an
  // EMPTY slot is exact; a supplied template has no featuriser and
  // `fusedTemplateLayout: null` makes it refuse rather than guess, which is
  // where boltz2 and protenix2 both started.
  fusedTemplateEmbedder: true,
  projectedRelpos: true,
  preSymmetrisedPde: false,
  templateMeanOverAllSlots: false,
  // NOT in TRANSPOSED_COLUMN_PAIR_BIAS, unlike the rest of the OF3 lineage.
  swapTransposedBias: false,
  // OPENFOLD3_LINEAGE.
  symmetriseBonds: true,
  maskPaddedKeys: true,
  padSingleCondUnknownDna: true,
  pairInitFromSingle: false,
  msaUpdateBeforeOuterProduct: false,
  distogramBias: true,
  keyMaskedAtomAttention: true,
  perBlockPairLayerNorm: true,
  perBlockAtomPairLayerNorm: true,
  // 🔴 AND NOT THE CHAINED FORM, though it shares the per-block atom-pair norm
  // with opendde and protenix. The reference is explicit: "rf3 keeps AF3's
  // parallel normalisation; opendde/protenix chain them". Reading the per-block
  // flag as implying the chained one is the mistake this line exists to stop.
  chainedAtomLayerNorm: false,
  splitPairConditioning: false,
  confidenceGlobalNorm: true,
  chiralCentres: true,
  confidenceCaDgram: true,
  dedupeSelfMsa: true,
  dropTerminalAtoms: true,
  atomizedElementNames: true,
  atomizedUnknownRestype: true,
  atomizedUnknownMsa: true,
  atomizedBackboneBonds: true,
  // 🔴 THIS CHECKPOINT HAS NO WORKING FLOW WALK, AND THE PAGE DEFAULTED TO ONE.
  // Measured on 6MRR: `--mode=flow` gives N-CA **6.94 A** against 1.46 and
  // consecutive CA collapsing to 0.23 A at worst - the geometry gate refuses it
  // outright - while `--mode=diffusion` gives 1.693 A and a clean backbone.
  // AlphaFold 3 and intellifold2 fold fine in flow (CA-CA 3.58 and 3.88), so it
  // is this checkpoint and not the sampler.
  //
  // 🔴 AND pLDDT SAID 81.47 ON THE BROKEN ONE, four tenths from the good fold's
  // 81.53. Nothing but the chain-geometry check can see this.
  //
  // Inverse polarity, like `dedupeSelfMsa`: the flag names the DIVERGENCE so
  // stock AlphaFold 3 stays all-false, which test/af3-dialect.test.js asserts.
  noFlowSampler: true,
  qblockAtomKeys: false,
  paddedAtomKeys: true,
  diffusionNoResidual: true,
  msaPairedQueryRow: false,
  maskAtomActPerBlock: false,
  triangleMulDivideByLength: true,
  structuralTokens: false,
});

export const DIALECTS = Object.freeze({
  protenix2: PROTENIX2,
  boltz2: BOLTZ2,
  alphafold3: ALPHAFOLD3,
  openbind0: OPENBIND0,
  opendde: OPENDDE,
  intellifold2: INTELLIFOLD2,
  rosettafold3: ROSETTAFOLD3,
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
/**
 * The dialect fields the FEATURISER reads, as one object.
 *
 * 🔴 BECAUSE THREE CALL SITES HAND-COPIED THEM AND A FOURTH CONVENTION WOULD
 * HAVE MADE IT FOUR PLACES TO FORGET. `tools/gpu/fold.js`,
 * `tools/gpu/fold-opendde.js` and `web/af3-model.js` each listed
 * `centreRefConformers`, `paddedAtomKeys`, `qblockAtomKeys`,
 * `dropTerminalAtoms` and `dedupeSelfMsa` by name - which is the allow-list
 * shape CLAUDE.md already records twice, once for `predictA3m` dropping
 * `pairHost` and taking the contact overlay off the shipped page, and once for
 * src/af2/multimer/model.js dropping the whole multimer regime. Both were fixed by
 * forwarding the object. This is that fix, before the third instance costs
 * anything: a convention added to the table reaches every caller, and a caller
 * that wants a subset can still spread and override.
 *
 * It deliberately does NOT include the graph conventions - a featuriser has no
 * business with `noResidual` - so adding a kernel flag to the table does not
 * silently change a batch.
 */
/**
 * The four dialect flags an ATOM BLOCK carries, as one object.
 *
 * 🔴 THE SAME LESSON AS `featuriserDialect`, ONE MODULE OVER. These were listed
 * by hand at every site that builds an atom block - the loader in
 * diffusion-weights.js and the hand-built weight dicts in
 * tools/gpu/check-af3-atom-decoder.js - and when `maskAtomActPerBlock` was added
 * the loader learned it and the checker did not. The checker then threw
 * "maskAtomActPerBlock has no default" on every run, so the ONE differential
 * that separates the atom decoder's GPU path from its CPU reference had been
 * dead for as long as that flag has existed. A decoder defect could not have
 * been seen by the only instrument pointed at it.
 *
 * The block carries them rather than the dialect object because the DECODER
 * never sees a dialect - it reads what its weights carry.
 * @param {object} dialect
 */
export function atomBlockDialect(dialect) {
  if (dialect === undefined || dialect === null) {
    throw new Error("atomBlockDialect needs a dialect: an atom block's flags have no defaults");
  }
  return {
    chainedAtomLayerNorm: dialect.chainedAtomLayerNorm,
    keyMaskedAtomAttention: dialect.keyMaskedAtomAttention,
    maskAtomActPerBlock: dialect.maskAtomActPerBlock,
    diffusionNoResidual: dialect.diffusionNoResidual,
  };
}

export function featuriserDialect(dialect) {
  if (dialect === undefined || dialect === null) return {};
  return {
    symmetriseBonds: dialect.symmetriseBonds,
    centreRefConformers: dialect.centreRefConformers,
    paddedAtomKeys: dialect.paddedAtomKeys,
    qblockAtomKeys: dialect.qblockAtomKeys,
    dropTerminalAtoms: dialect.dropTerminalAtoms,
    dedupeSelfMsa: dialect.dedupeSelfMsa,
    atomizedElementNames: dialect.atomizedElementNames,
    atomizedUnknownRestype: dialect.atomizedUnknownRestype,
    atomizedUnknownMsa: dialect.atomizedUnknownMsa,
    atomizedBackboneBonds: dialect.atomizedBackboneBonds,
  };
}

export const DIALECT_ALIASES = Object.freeze({
  openbind: "openbind0",
  // The reference's own short names, which a bundle exported through it may
  // carry: `if2`/`intellifold` and `rf3`.
  if2: "intellifold2",
  intellifold: "intellifold2",
  rf3: "rosettafold3",
});

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
