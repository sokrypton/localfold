/**
 * The diffusion head's weight bundle, and the atom reference embeddings.
 *
 * Split from weights.js because the diffusion side is a different half of
 * the checkpoint - the trunk's loader is already long, and a typo in one leaf
 * name here surfaces as a numerical disagreement rather than a missing key.
 */
import {
  af3Dialect, bind, dims, layer, stacked,
  trunkWeights, confidenceWeights,
  structuralExpanderWeights, structuralRefinerWeights, openddeConfidenceWeights,
} from "./weights.js";

const HEAD = "diffuser/~/diffusion_head";
const ENCODER = `${HEAD}/diffusion_atom_transformer_encoder`;
const DECODER = `${HEAD}/diffusion_atom_transformer_decoder`;
/**
 * The two diffusion atom stacks' block roots.
 *
 * 🔴 THE STACK'S NAME FOLLOWS THE PAIR-NORM CONVENTION, because haiku names a
 * layer stack for whether it carries per-layer inputs and OpenDDE's does not.
 * The same rename applies to the target_feat encoder and to the token
 * transformer; see `atomPairNorm` and `txStackFor`.
 */
const atomStackName = (perBlockPair) =>
  perBlockPair ? "__layer_stack_no_per_layer" : "__layer_stack_with_per_layer";
const encoderStackFor = (perBlockPair) =>
  `${ENCODER}/${atomStackName(perBlockPair)}/diffusion_atom_transformer_encoder`;
const decoderStackFor = (perBlockPair) =>
  `${DECODER}/${atomStackName(perBlockPair)}/diffusion_atom_transformer_decoder`;
const TX = `${HEAD}/transformer`;
/**
 * The token transformer's doubly-nested block stack.
 *
 * 🔴 THE NAME DEPENDS ON THE DIALECT, because haiku names a layer stack for
 * whether it carries per-layer inputs and OpenDDE's does not. See
 * `diffusionWeights`.
 */
const txStackFor = (perBlockPair) => {
  const name = perBlockPair ? "__layer_stack_no_per_layer" : "__layer_stack_with_per_layer";
  return `${TX}/${name}/${name}/transformer`;
};

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
/**
 * A loaded atom block, with the one dialect flag its arithmetic reads.
 *
 * 🔴 THE FLAG TRAVELS WITH THE BLOCK BECAUSE THE BLOCK IS WHAT THE REFERENCE
 * IS HANDED. `crossAttentionBlock` takes one block's weights and a state; the
 * dialect is neither, and threading it through every call site instead would
 * mean three signatures changing for a value that never varies within a stack.
 * A block that reaches the reference without it raises.
 */
/**
 * An atom stack's pair LayerNorm and logits projection, per block or shared.
 *
 * 🔴 THREE ATOM STACKS NEED THIS AND EACH WOULD HAVE COPIED IT. AlphaFold 3
 * normalises the atom-pair conditioning ONCE for a stack, so the scale is
 * [16] beside it; OpenDDE normalises inside every block, so the same tensor is
 * [3, 16] INSIDE the stack, and haiku names the enclosing stack
 * `__layer_stack_no_per_layer` rather than `__layer_stack_with_per_layer`
 * because of it. Both the path and the rank move together.
 *
 * Returned as an ARRAY of three either way, so a reader indexes by block with
 * no branch of its own; under AlphaFold 3 the three entries are one tensor,
 * which is what "shared" means.
 */
async function atomPairNorm(store, stackRoot, perBlock, blocks = 3) {
  const scaleName = `${stackRoot}/pair_input_layer_norm/scale`;
  const projectionName = `${stackRoot}/pair_logits_projection/weights`;
  if (!perBlock) {
    const scale = await store.tensor(scaleName);
    const projection = await store.tensor(projectionName);
    return { perBlock: false,
             scale: Array.from({ length: blocks }, () => scale),
             projection: Array.from({ length: blocks }, () => projection) };
  }
  const scale = [];
  const projection = [];
  for (let index = 0; index < blocks; index += 1) {
    scale.push(await layer(store, scaleName, index));
    projection.push(await layer(store, projectionName, index));
  }
  return { perBlock: true, scale, projection };
}

async function atomBlockWith(store, stack, index, dialect) {
  const block = await bind(store, atomBlock(store, stack, index));
  block.chainedAtomLayerNorm = dialect.chainedAtomLayerNorm;
  block.keyMaskedAtomAttention = dialect.keyMaskedAtomAttention;
  if (block.chainedAtomLayerNorm === undefined
      || block.keyMaskedAtomAttention === undefined) {
    throw new Error("an atom block's dialect flags have no defaults");
  }
  return block;
}

export async function targetFeatureWeights(store) {
  const root = "diffuser/evoformer_conditioning";
  const encoder = `${root}_atom_transformer_encoder`;
  const dialect = af3Dialect(store);
  // 🔴 THE PAIR LAYERNORM IS SHARED OR PER BLOCK, AND THE STACK'S NAME SAYS
  // WHICH. AlphaFold 3 normalises the atom-pair conditioning ONCE for the whole
  // stack, so `pair_input_layer_norm/scale` sits beside the stack, unstacked, at
  // [16]. OpenDDE normalises it inside every block, so the same tensor is
  // [3, 16] and lives INSIDE the layer stack - and haiku names the stack
  // `__layer_stack_no_per_layer` rather than `__layer_stack_with_per_layer`
  // because of it. Two different paths and two different ranks for one tensor;
  // reading either without the other loads nothing and reports a missing name.
  const perBlockPair = dialect.perBlockAtomPairLayerNorm;
  if (perBlockPair === undefined) {
    throw new Error("dialect.perBlockAtomPairLayerNorm has no default: AF3 "
      + "normalises the atom-pair conditioning once for the stack, OpenDDE "
      + "once per block");
  }
  const stack = perBlockPair
    ? `${encoder}/__layer_stack_no_per_layer/evoformer_conditioning_atom_transformer_encoder`
    : `${encoder}/__layer_stack_with_per_layer/evoformer_conditioning_atom_transformer_encoder`;
  const stackRoot = perBlockPair
    ? `${encoder}/__layer_stack_no_per_layer`
    : encoder;
  const W = (leaf) => store.tensor(`${root}_${leaf}/weights`);
  const pairNorm = await atomPairNorm(store, stackRoot, perBlockPair);
  return {
    dialect,
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
      // The first entry is the shared tensor under stock AF3 and block 0's
      // under OpenDDE; `pairNormPerBlock` beside them says which, and a caller
      // that ignores it gets AlphaFold 3's behaviour on an OpenDDE bundle -
      // which is a plausible encoder, so the encoder asserts on the flag.
      pairInputLayerNormScale: pairNorm.scale[0],
      pairLogitsProjection: pairNorm.projection[0],
      pairNormPerBlock: pairNorm.perBlock,
      pairInputLayerNormScales: pairNorm.scale,
      pairLogitsProjections: pairNorm.projection,
      projectAtomFeaturesForAggr: await W("project_atom_features_for_aggr"),
      blocks: [await atomBlockWith(store, stack, 0, dialect),
               await atomBlockWith(store, stack, 1, dialect),
               await atomBlockWith(store, stack, 2, dialect)],
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

/**
 * The diffusion conditioning's weights, on their own.
 *
 * 🔴 ITS OWN FUNCTION SO A CHECKER CAN GO THROUGH IT. This used to be an object
 * literal inside `diffusionWeights`, so the one tool that checks the
 * conditioning hand-built its weight dict with `pairChannels: 128` and
 * `targetFeatWidth: 447` typed in - which is exactly the fault CLAUDE.md
 * records costing months on the side chains, and here it cost the OpenDDE fold
 * ten seconds a fold: the arm that should have said "the GPU conditioning has
 * no split-pair branch" instead read every tensor at AF3's widths, produced
 * NaN on both sides, and passed.
 *
 * @param {object} store
 * @param {object} dialect from `af3Dialect(store)`
 */
export async function conditioningWeights(store, dialect) {
  const T = (name) => store.tensor(`${HEAD}/${name}`);
  const transition = async (prefix) => ({
    ffwLayerNormScale: await T(`${prefix}ffw_layer_norm/scale`),
    ffwLayerNormOffset: await T(`${prefix}ffw_layer_norm/offset`),
    ffwTransition1: await T(`${prefix}ffw_transition1/weights`),
    ffwTransition2: await T(`${prefix}ffw_transition2/weights`),
  });
  const splitPair = dialect.splitPairConditioning;
  if (splitPair === undefined) {
    throw new Error("dialect.splitPairConditioning has no default");
  }
  const hasSplit = store.manifest?.tensors?.[`${HEAD}/z_trunk_projection/weights`] !== undefined;
  if (hasSplit !== splitPair) {
    throw new Error(`this bundle ${hasSplit ? "carries" : "does not carry"} `
      + "z_trunk_projection and its dialect says otherwise");
  }
  return {
    // 🔴 EVERY WIDTH HERE IS THE TENSOR'S. `pair_cond_initial_projection` is
    // [267, 128] under AlphaFold 3 and [256, 128] under OpenDDE, because the
    // second compresses its two inputs separately before concatenating them -
    // and OpenDDE's trunk pair arriving here is 384 wide, not 128.
    pairChannels: dims(store, `${HEAD}/pair_cond_initial_projection/weights`)[1],
    seqChannels: dims(store, `${HEAD}/single_cond_initial_projection/weights`)[1],
    targetFeatWidth: 447, relativeWidth: 139,
    trunkPairChannels: splitPair
      ? dims(store, `${HEAD}/z_trunk_projection/weights`)[0]
      : dims(store, `${HEAD}/pair_cond_initial_projection/weights`)[0] - 139,
    pairCondInitialNormScale: await T("pair_cond_initial_norm/scale"),
    pairCondInitialProjection: await T("pair_cond_initial_projection/weights"),
    // OpenDDE's two separate compressions; absent under AlphaFold 3, and the
    // reference branches on their presence.
    ...(splitPair ? {
      zTrunkNormScale: await T("z_trunk_norm/scale"),
      zTrunkProjection: await T("z_trunk_projection/weights"),
      relpeProjection: await T("relpe_projection/weights"),
    } : {}),
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
  };
}

export async function diffusionWeights(store, superBlocks = 6) {
  const T = (name) => store.tensor(`${HEAD}/${name}`);
  // The atom stacks' dialect flags; see `atomBlockWith`.
  const dialect = af3Dialect(store);
  const transition = async (prefix) => ({
    ffwLayerNormScale: await T(`${prefix}ffw_layer_norm/scale`),
    ffwLayerNormOffset: await T(`${prefix}ffw_layer_norm/offset`),
    ffwTransition1: await T(`${prefix}ffw_transition1/weights`),
    ffwTransition2: await T(`${prefix}ffw_transition2/weights`),
  });

  // 🔴 THE TOKEN TRANSFORMER'S PAIR NORM IS SHARED OR PER BLOCK, AND THE STACK
  // IS NAMED FOR IT. AlphaFold 3 LayerNorms the pair conditioning ONCE for the
  // whole stack - `pair_input_layer_norm/scale` is [128], beside the stack -
  // and projects it once per SUPER block, [6, 128, 4, 16] covering that
  // super-block's four blocks at 16 heads each. OpenDDE does both PER BLOCK:
  // [6, 4, 128] and [6, 4, 128, 16], inside a stack haiku names
  // `__layer_stack_no_per_layer` rather than `__layer_stack_with_per_layer`.
  // So the path and the rank both move, and reading one convention without the
  // other finds no tensors at all.
  const perBlockPair = dialect.perBlockPairLayerNorm;
  if (perBlockPair === undefined) {
    throw new Error("dialect.perBlockPairLayerNorm has no default: AF3 "
      + "normalises the token transformer's pair conditioning once for the "
      + "stack, OpenDDE once per block");
  }
  const stackName = perBlockPair ? "__layer_stack_no_per_layer" : "__layer_stack_with_per_layer";
  // 🔴 THE ATOM STACKS FOLLOW THE ATOM FLAG, NOT THE TOKEN ONE. They are two
  // separate memberships upstream - PER_BLOCK_PAIR_LAYER_NORM is about the
  // TOKEN transformer and PER_BLOCK_ATOM_PAIR_LAYER_NORM about these - and
  // upstream records models that are per-block on one and shared on the other.
  // 🔴 THE PAIR CONDITIONING'S SHAPE IS A DIALECT QUESTION AND THE TENSORS
  // AGREE WITH IT. OpenDDE carries `z_trunk_projection` and `relpe_projection`
  // and AlphaFold 3 does not, so presence and flag are cross-checked - a
  // bundle where they disagreed would silently condition on the wrong thing.
  // The pair conditioning's shape is a dialect question; conditioningWeights
  // cross-checks the flag against the tensors.
  const atomPerBlock = dialect.perBlockAtomPairLayerNorm;
  if (atomPerBlock === undefined) {
    throw new Error("dialect.perBlockAtomPairLayerNorm has no default");
  }
  const atomStack = atomPerBlock ? "/__layer_stack_no_per_layer" : "";
  const encoderPairNorm = await atomPairNorm(store, `${ENCODER}${atomStack}`, atomPerBlock);
  const decoderPairNorm = await atomPairNorm(store, `${DECODER}${atomStack}`, atomPerBlock);
  const projectionName = perBlockPair
    ? `${TX}/${stackName}/${stackName}/pair_logits_projection/weights`
    : `${TX}/${stackName}/pair_logits_projection/weights`;
  const rawProjections = await store.tensor(projectionName);
  const projectionStride = rawProjections.length / store.shape(projectionName)[0];

  /**
   * OpenDDE's per-block pair norm, folded into its per-block projection.
   *
   * 🔴 A PER-BLOCK LAYERNORM SCALE IS EXACTLY A PER-BLOCK PROJECTION, because
   * this norm has no offset: `LN(z) * scale_b @ W_b` is `LN(z) @
   * (diag(scale_b) W_b)`, and the mean and variance the LayerNorm removes do
   * not depend on the scale. So OpenDDE's [6, 4, 128] scale and [6, 4, 128, 16]
   * projection collapse into AlphaFold 3's shared-scale, one-projection-per-
   * super-block form with NO kernel change at all - which is the same move the
   * weight converter makes for every other difference it can absorb.
   *
   * The layouts differ as well as the ranks: AlphaFold 3's slice is
   * [channels, blockInGroup, head] and OpenDDE's is [blockInGroup, channels,
   * head], so this transposes while it folds.
   */
  const foldPerBlockPairNorm = async () => {
    const scaleName = `${TX}/${stackName}/${stackName}/pair_input_layer_norm/scale`;
    const scales = await store.tensor(scaleName);
    const [, blocksPerGroup, channels] = store.shape(scaleName);
    const heads = store.shape(projectionName)[3];
    const out = new Float32Array(rawProjections.length);
    for (let group = 0; group < superBlocks; group += 1) {
      for (let block = 0; block < blocksPerGroup; block += 1) {
        for (let c = 0; c < channels; c += 1) {
          const scale = scales[(group * blocksPerGroup + block) * channels + c];
          for (let head = 0; head < heads; head += 1) {
            const from = ((group * blocksPerGroup + block) * channels + c) * heads + head;
            const to = ((group * channels + c) * blocksPerGroup + block) * heads + head;
            out[to] = scale * rawProjections[from];
          }
        }
      }
    }
    return { projections: out, channels };
  };
  const folded = perBlockPair ? await foldPerBlockPairNorm() : null;
  const projections = folded === null ? rawProjections : folded.projections;
  const groups = [];
  for (let s = 0; s < superBlocks; s += 1) {
    const blocks = [];
    for (let inner = 0; inner < 4; inner += 1) {
      const at = (leaf) => stacked(store, `${txStackFor(perBlockPair)}${leaf}`, s * 4 + inner, 2);
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
    dialect: af3Dialect(store),
    seqChannels: 384, perTokenChannels: 768,
    singleCondEmbeddingNormScale: await T("single_cond_embedding_norm/scale"),
    singleCondEmbeddingProjection: await T("single_cond_embedding_projection/weights"),
    outputNormScale: await T("output_norm/scale"),
    conditioning: await conditioningWeights(store, dialect),
    transformer: {
      channels: 768, condChannels: 384, pairChannels: 128, heads: 16, dimension: 48,
      transitionFactor: 2, blocksPerSuperBlock: 4,
      // Shared under AlphaFold 3 and per block under OpenDDE, where the scale
      // lives inside the doubly-nested stack at [6, 4, 128].
      // 🔴 ALL ONES UNDER OpenDDE, because its per-block scales are folded into
      // the per-block projections above - see foldPerBlockPairNorm. The
      // LayerNorm itself still runs; only its affine has moved.
      pairNormPerBlock: perBlockPair,
      pairInputLayerNormScale: perBlockPair
        ? new Float32Array(folded.channels).fill(1)
        : await store.tensor(`${TX}/pair_input_layer_norm/scale`),
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
      // Shared under AlphaFold 3, per block under OpenDDE - and under OpenDDE
      // the tensors live INSIDE the stack, which is why the root moves too.
      pairInputLayerNormScale: encoderPairNorm.scale[0],
      pairLogitsProjection: encoderPairNorm.projection[0],
      pairInputLayerNormScales: encoderPairNorm.scale,
      pairLogitsProjections: encoderPairNorm.projection,
      pairNormPerBlock: encoderPairNorm.perBlock,
      lnormTrunkSingleCondScale: await T("diffusion_lnorm_trunk_single_cond/scale"),
      embedTrunkSingleCond: await T("diffusion_embed_trunk_single_cond/weights"),
      lnormTrunkPairCondScale: await T("diffusion_lnorm_trunk_pair_cond/scale"),
      embedTrunkPairCond: await T("diffusion_embed_trunk_pair_cond/weights"),
      atomPositionsToFeatures: await T("diffusion_atom_positions_to_features/weights"),
      projectAtomFeaturesForAggr: await T("diffusion_project_atom_features_for_aggr/weights"),
      blocks: [await atomBlockWith(store, encoderStackFor(atomPerBlock), 0, dialect),
               await atomBlockWith(store, encoderStackFor(atomPerBlock), 1, dialect),
               await atomBlockWith(store, encoderStackFor(atomPerBlock), 2, dialect)],
    },
    decoder: {
      channels: 128, pairChannels: 16, heads: 4, dimension: 32, perTokenChannels: 768,
      // Shared under AlphaFold 3, per block under OpenDDE - and under OpenDDE
      // the tensors live INSIDE the stack, which is why the root moves too.
      pairInputLayerNormScale: decoderPairNorm.scale[0],
      pairLogitsProjection: decoderPairNorm.projection[0],
      pairInputLayerNormScales: decoderPairNorm.scale,
      pairLogitsProjections: decoderPairNorm.projection,
      pairNormPerBlock: decoderPairNorm.perBlock,
      projectTokenFeaturesForBroadcast:
        await T("diffusion_project_token_features_for_broadcast/weights"),
      atomFeaturesLayerNormScale: await T("diffusion_atom_features_layer_norm/scale"),
      atomFeaturesToPositionUpdate: await T("diffusion_atom_features_to_position_update/weights"),
      blocks: [await atomBlockWith(store, decoderStackFor(atomPerBlock), 0, dialect),
               await atomBlockWith(store, decoderStackFor(atomPerBlock), 1, dialect),
               await atomBlockWith(store, decoderStackFor(atomPerBlock), 2, dialect)],
    },
  };
}

/**
 * Every weight `foldBatch` needs, for whichever model the manifest describes.
 *
 * 🔴 A TOOL THAT BUILDS THIS BY HAND IS PINNED TO AlphaFold 3, AND READS AS IF
 * IT IS NOT. `--model=` takes a manifest, so every probe here LOOKS
 * model-agnostic; the five lines each of them wrote out - trunk, diffusion,
 * `confidenceWeights`, atom reference, target features - name AlphaFold 3's
 * confidence head unconditionally and know nothing of a structural stack. The
 * same shape as the per-module checkers pinned to `CHANNELS = 128`, which is
 * where OpenDDE's first bundle broke while every one of them passed.
 *
 * The dialect decides, once, here: a bundle whose dialect says `structuralTokens`
 * gets the expander, the refiner and OpenDDE's own confidence head; everything
 * else gets AlphaFold 3's.
 */
export async function foldWeights(store, options = {}) {
  const trunk = await trunkWeights(store, options.blocks ?? 48, options.msaBlocks ?? 4);
  return {
    trunk,
    targetFeat: await targetFeatureWeights(store),
    diffusion: await diffusionWeights(store),
    atomReference: await atomReference(store),
    ...(trunk.dialect.structuralTokens ? {
      expander: await structuralExpanderWeights(store),
      refiner: await structuralRefinerWeights(store),
      openddeConfidence: {
        ...await openddeConfidenceWeights(store),
        weightPrecision: options.confidenceWeightPrecision,
      },
    } : { confidence: await confidenceWeights(store) }),
    ...(options.refinerWeightPrecision === undefined
      ? {} : { refinerWeightPrecision: options.refinerWeightPrecision }),
  };
}
