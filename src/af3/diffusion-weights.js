/**
 * The diffusion head's weight bundle, and the atom reference embeddings.
 *
 * Split from weights.js because the diffusion side is a different half of
 * the checkpoint - the trunk's loader is already long, and a typo in one leaf
 * name here surfaces as a numerical disagreement rather than a missing key.
 */
import { bind, stacked } from "./weights.js";

const HEAD = "diffuser/~/diffusion_head";
const ENCODER = `${HEAD}/diffusion_atom_transformer_encoder`;
const DECODER = `${HEAD}/diffusion_atom_transformer_decoder`;
const ENCODER_STACK =
  `${ENCODER}/__layer_stack_with_per_layer/diffusion_atom_transformer_encoder`;
const DECODER_STACK =
  `${DECODER}/__layer_stack_with_per_layer/diffusion_atom_transformer_decoder`;
const TX = `${HEAD}/transformer`;
const TX_STACK = `${TX}/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformer`;

/**
 * One AdaLN cross-attention block, from either atom stack - as a DESCRIPTOR
 * whose leaves decode when they are read. See src/af3/weights.js: the diffusion
 * head is 920 MiB of float32 and the device already holds all of it.
 */
function atomBlock(store, root, index) {
  const at = (leaf) => stacked(store, `${root}${leaf}`, index);
  return {
    qSingleCondLayerNormScale: at("qsingle_cond_layer_norm/scale"),
    qSingleCondScaleWeights: at("qsingle_cond_scale/weights"),
    qSingleCondScaleBias: at("qsingle_cond_scale/bias"),
    qSingleCondBias: at("qsingle_cond_bias/weights"),
    kSingleCondLayerNormScale: at("ksingle_cond_layer_norm/scale"),
    kSingleCondScaleWeights: at("ksingle_cond_scale/weights"),
    kSingleCondScaleBias: at("ksingle_cond_scale/bias"),
    kSingleCondBias: at("ksingle_cond_bias/weights"),
    qProjection: at("q_projection/weights"),
    qBias: at("q_projection/bias"),
    kProjection: at("k_projection/weights"),
    vProjection: at("v_projection/weights"),
    gatingQuery: at("gating_query/weights"),
    Transition2: at("transition2/weights"),
    AdaptiveZeroCondWeights: at("adaptive_zero_cond/weights"),
    AdaptiveZeroCondBias: at("adaptive_zero_cond/bias"),
    ffwSingleCondLayerNormScale: at("ffw_single_cond_layer_norm/scale"),
    ffwSingleCondScaleWeights: at("ffw_single_cond_scale/weights"),
    ffwSingleCondScaleBias: at("ffw_single_cond_scale/bias"),
    ffwSingleCondBias: at("ffw_single_cond_bias/weights"),
    ffwTransition1: at("ffw_transition1/weights"),
    ffwTransition2: at("ffw_transition2/weights"),
    ffwAdaptiveZeroCondWeights: at("ffw_adaptive_zero_cond/weights"),
    ffwAdaptiveZeroCondBias: at("ffw_adaptive_zero_cond/bias"),
  };
}

/**
 * The atom encoder that builds target_feat's 384 atom-derived columns.
 *
 * 🔴 A DIFFERENT ATOM ENCODER FROM THE DIFFUSION HEAD'S, sharing its shape and
 * none of its weights. This one lives under `evoformer_conditioning`, runs
 * ONCE per fold on the reference conformers alone - no noisy positions, no
 * trunk conditioning - and its pooled output is 384 wide where the diffusion
 * encoder's is 768. Passing either bundle where the other belongs type-checks
 * and is a different model.
 */
export async function targetFeatureWeights(store) {
  const root = "diffuser/evoformer_conditioning";
  const encoder = `${root}_atom_transformer_encoder`;
  const stack =
    `${encoder}/__layer_stack_with_per_layer/evoformer_conditioning_atom_transformer_encoder`;
  const W = (leaf) => store.tensor(`${root}_${leaf}/weights`);
  return {
    reference: {
      channels: 128,
      embedRefPos: await W("embed_ref_pos"),
      embedRefMask: await W("embed_ref_mask"),
      embedRefElement: await W("embed_ref_element"),
      embedRefCharge: await W("embed_ref_charge"),
      embedRefAtomName: await W("embed_ref_atom_name"),
    },
    encoder: {
      channels: 128, pairChannels: 16, heads: 4, dimension: 32, perTokenChannels: 384,
      // 🔴 THE _1 SUFFIX IS PART OF THE NAME. Four of these also exist under
      // the unsuffixed name with IDENTICAL shapes, so dropping the suffix loads
      // clean and gives the wrong target_feat. embed_pair_offsets_valid is the
      // one with no _1 form, which makes the set look like a typo and is not.
      singleToPairCondRow: await W("single_to_pair_cond_row_1"),
      singleToPairCondCol: await W("single_to_pair_cond_col_1"),
      embedPairOffsets: await W("embed_pair_offsets_1"),
      embedPairDistances: await W("embed_pair_distances_1"),
      embedPairOffsetsValid: await W("embed_pair_offsets_valid"),
      pairMlp1: await W("pair_mlp_1"),
      pairMlp2: await W("pair_mlp_2"),
      pairMlp3: await W("pair_mlp_3"),
      pairInputLayerNormScale: await store.tensor(`${encoder}/pair_input_layer_norm/scale`),
      pairLogitsProjection: await store.tensor(`${encoder}/pair_logits_projection/weights`),
      projectAtomFeaturesForAggr: await W("project_atom_features_for_aggr"),
      blocks: [await bind(store, atomBlock(store, stack, 0)),
               await bind(store, atomBlock(store, stack, 1)),
               await bind(store, atomBlock(store, stack, 2))],
      // 🔴 THREE WEIGHTS THIS ENCODER DOES NOT HAVE, AT THE RIGHT LENGTHS AND
      // FULL OF ZEROS. Af3AtomEncoderGpu is a superset of this module: it also
      // adds the trunk's single, the trunk's pair and an embedding of the noisy
      // positions. Each is a BIAS-FREE linear of a layer-normed input, so
      // feeding zero inputs contributes exactly zero and one kernel serves both
      // - but the shader still INDEXES these, so they have to exist. Their
      // values are irrelevant; zeros say so.
      // Checked at relRMS 8e-8 against the CPU reference by
      // tools/gpu/check-af3-target-feat-gpu.js, which is also where the 33x
      // comes from.
      trunkSingleChannels: 384,
      trunkPairChannels: 128,
      lnormTrunkSingleCondScale: new Float32Array(384),
      embedTrunkSingleCond: new Float32Array(384 * 128),
      lnormTrunkPairCondScale: new Float32Array(128),
      embedTrunkPairCond: new Float32Array(128 * 16),
      atomPositionsToFeatures: new Float32Array(3 * 128),
    },
  };
}

/** The five reference embeddings the atom conditioning sums. */
export async function atomReference(store) {
  const T = (name) => store.tensor(`${HEAD}/${name}`);
  return {
    channels: 128,
    embedRefPos: await T("diffusion_embed_ref_pos/weights"),
    embedRefMask: await T("diffusion_embed_ref_mask/weights"),
    embedRefElement: await T("diffusion_embed_ref_element/weights"),
    embedRefCharge: await T("diffusion_embed_ref_charge/weights"),
    embedRefAtomName: await T("diffusion_embed_ref_atom_name/weights"),
  };
}

export async function diffusionWeights(store, superBlocks = 6) {
  const T = (name) => store.tensor(`${HEAD}/${name}`);
  const transition = async (prefix) => ({
    ffwLayerNormScale: await T(`${prefix}ffw_layer_norm/scale`),
    ffwLayerNormOffset: await T(`${prefix}ffw_layer_norm/offset`),
    ffwTransition1: await T(`${prefix}ffw_transition1/weights`),
    ffwTransition2: await T(`${prefix}ffw_transition2/weights`),
  });

  const projectionName = `${TX}/__layer_stack_with_per_layer/pair_logits_projection/weights`;
  const projections = await store.tensor(projectionName);
  const projectionStride = projections.length / store.shape(projectionName)[0];
  const groups = [];
  for (let s = 0; s < superBlocks; s += 1) {
    const blocks = [];
    for (let inner = 0; inner < 4; inner += 1) {
      const at = (leaf) => stacked(store, `${TX_STACK}${leaf}`, s * 4 + inner, 2);
      blocks.push(await bind(store, {
        SingleCondLayerNormScale: at("single_cond_layer_norm/scale"),
        SingleCondScaleWeights: at("single_cond_scale/weights"),
        SingleCondScaleBias: at("single_cond_scale/bias"),
        SingleCondBias: at("single_cond_bias/weights"),
        qProjection: at("q_projection/weights"),
        qBias: at("q_projection/bias"),
        kProjection: at("k_projection/weights"),
        vProjection: at("v_projection/weights"),
        gatingQuery: at("gating_query/weights"),
        Transition2: at("transition2/weights"),
        AdaptiveZeroCondWeights: at("adaptive_zero_cond/weights"),
        AdaptiveZeroCondBias: at("adaptive_zero_cond/bias"),
        ffwSingleCondLayerNormScale: at("ffw_single_cond_layer_norm/scale"),
        ffwSingleCondScaleWeights: at("ffw_single_cond_scale/weights"),
        ffwSingleCondScaleBias: at("ffw_single_cond_scale/bias"),
        ffwSingleCondBias: at("ffw_single_cond_bias/weights"),
        ffwTransition1: at("ffw_transition1/weights"),
        ffwTransition2: at("ffw_transition2/weights"),
        ffwAdaptiveZeroCondWeights: at("ffw_adaptive_zero_cond/weights"),
        ffwAdaptiveZeroCondBias: at("ffw_adaptive_zero_cond/bias"),
      }));
    }
    groups.push({
      pairLogitsProjection: projections.subarray(s * projectionStride,
                                                 (s + 1) * projectionStride),
      blocks,
    });
  }

  return {
    seqChannels: 384, perTokenChannels: 768,
    singleCondEmbeddingNormScale: await T("single_cond_embedding_norm/scale"),
    singleCondEmbeddingProjection: await T("single_cond_embedding_projection/weights"),
    outputNormScale: await T("output_norm/scale"),
    conditioning: {
      pairChannels: 128, seqChannels: 384, targetFeatWidth: 447, relativeWidth: 139,
      pairCondInitialNormScale: await T("pair_cond_initial_norm/scale"),
      pairCondInitialProjection: await T("pair_cond_initial_projection/weights"),
      pairTransitions: [await transition("pair_transition_0"),
                        await transition("pair_transition_1")],
      singleCondInitialNormScale: await T("single_cond_initial_norm/scale"),
      singleCondInitialProjection: await T("single_cond_initial_projection/weights"),
      singleTransitions: [await transition("single_transition_0"),
                          await transition("single_transition_1")],
      fourierWeight: await T("fourier_embedding_weight"),
      fourierBias: await T("fourier_embedding_bias"),
      noiseEmbeddingInitialNormScale: await T("noise_embedding_initial_norm/scale"),
      noiseEmbeddingInitialProjection: await T("noise_embedding_initial_projection/weights"),
    },
    transformer: {
      channels: 768, condChannels: 384, pairChannels: 128, heads: 16, dimension: 48,
      transitionFactor: 2, blocksPerSuperBlock: 4,
      pairInputLayerNormScale: await store.tensor(`${TX}/pair_input_layer_norm/scale`),
      superBlocks: groups,
    },
    encoder: {
      channels: 128, pairChannels: 16, heads: 4, dimension: 32,
      perTokenChannels: 768, trunkSingleChannels: 384, trunkPairChannels: 128,
      // 🔴 THE _1 SUFFIX IS PART OF THE NAME, HERE TOO. The same four tensors
      // exist unsuffixed, at IDENTICAL shapes, and belong to the pair
      // conditioning computed over a token's own 24 dense atom slots - AF3
      // captures them as [tokens, 24, 24, 16] against these ones'
      // [subsets, 32, 128, 16]. This encoder works in the QUERIES-KEYS layout,
      // so it wants the _1 set; dropping the suffix loads clean, runs, folds a
      // protein, and is a different model. targetFeatureWeights above says the
      // same thing about the same trap and this loader had it wrong: it cost
      // 0.102 relRMS against AF3 on the head's own output, side chains about 8%
      // compressed, and nothing caught it because the only checker that reaches
      // the head builds its weights by hand.
      singleToPairCondRow: await T("diffusion_single_to_pair_cond_row_1/weights"),
      singleToPairCondCol: await T("diffusion_single_to_pair_cond_col_1/weights"),
      embedPairOffsets: await T("diffusion_embed_pair_offsets_1/weights"),
      embedPairDistances: await T("diffusion_embed_pair_distances_1/weights"),
      // ...and this one has no _1 form, which makes the set look like a typo.
      embedPairOffsetsValid: await T("diffusion_embed_pair_offsets_valid/weights"),
      pairMlp1: await T("diffusion_pair_mlp_1/weights"),
      pairMlp2: await T("diffusion_pair_mlp_2/weights"),
      pairMlp3: await T("diffusion_pair_mlp_3/weights"),
      pairInputLayerNormScale: await store.tensor(`${ENCODER}/pair_input_layer_norm/scale`),
      pairLogitsProjection: await store.tensor(`${ENCODER}/pair_logits_projection/weights`),
      lnormTrunkSingleCondScale: await T("diffusion_lnorm_trunk_single_cond/scale"),
      embedTrunkSingleCond: await T("diffusion_embed_trunk_single_cond/weights"),
      lnormTrunkPairCondScale: await T("diffusion_lnorm_trunk_pair_cond/scale"),
      embedTrunkPairCond: await T("diffusion_embed_trunk_pair_cond/weights"),
      atomPositionsToFeatures: await T("diffusion_atom_positions_to_features/weights"),
      projectAtomFeaturesForAggr: await T("diffusion_project_atom_features_for_aggr/weights"),
      blocks: [await bind(store, atomBlock(store, ENCODER_STACK, 0)),
               await bind(store, atomBlock(store, ENCODER_STACK, 1)),
               await bind(store, atomBlock(store, ENCODER_STACK, 2))],
    },
    decoder: {
      channels: 128, pairChannels: 16, heads: 4, dimension: 32, perTokenChannels: 768,
      pairInputLayerNormScale: await store.tensor(`${DECODER}/pair_input_layer_norm/scale`),
      pairLogitsProjection: await store.tensor(`${DECODER}/pair_logits_projection/weights`),
      projectTokenFeaturesForBroadcast:
        await T("diffusion_project_token_features_for_broadcast/weights"),
      atomFeaturesLayerNormScale: await T("diffusion_atom_features_layer_norm/scale"),
      atomFeaturesToPositionUpdate: await T("diffusion_atom_features_to_position_update/weights"),
      blocks: [await bind(store, atomBlock(store, DECODER_STACK, 0)),
               await bind(store, atomBlock(store, DECODER_STACK, 1)),
               await bind(store, atomBlock(store, DECODER_STACK, 2))],
    },
  };
}
