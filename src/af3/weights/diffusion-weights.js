/**
 * The diffusion head's weight bundle, and the atom reference embeddings.
 *
 * Split from weights.js because the diffusion side is a different half of
 * the checkpoint - the trunk's loader is already long, and a typo in one leaf
 * name here surfaces as a numerical disagreement rather than a missing key.
 */
import {
  af3Dialect, bind, dims, layer, stacked, stackedIfPresent,
  trunkWeights, confidenceWeights,
  structuralExpanderWeights, structuralRefinerWeights, openddeConfidenceWeights,
} from "./weights.js";
import { atomBlockDialect } from "../dialect.js";

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
/**
 * The token transformer's pair width, off the tensor that states it.
 *
 * 🔴 THE TWO LAYOUTS NEST DIFFERENTLY AND ORDER THEIR AXES DIFFERENTLY, so one
 * expression cannot read both:
 *
 *     AF3        .../__layer_stack_with_per_layer/pair_logits_projection   [6, 128, 4, 16]
 *     protenix2  .../__layer_stack_no_per_layer/__layer_stack_no_per_layer/...  [6, 4, 256, 16]
 *
 * singly nested with the pair width at axis 1, against doubly nested with it at
 * axis 2. `txStackFor` cannot be reused either: it appends a trailing
 * `/transformer` because most leaves in that stack are named `transformer<leaf>`
 * CONCATENATED, and this one is not.
 */
const txPairChannels = (store, perBlockPair) => {
  const name = txStackName(perBlockPair);
  return perBlockPair
    ? dims(store, `${TX}/${name}/${name}/pair_logits_projection/weights`)[2]
    : dims(store, `${TX}/${name}/pair_logits_projection/weights`)[1];
};

/**
 * The token transformer's five shape numbers, off ONE tensor.
 *
 * 🔴 BECAUSE `channels: 768, heads: 16, dimension: 48, transitionFactor: 2,
 * blocksPerSuperBlock: 4` WERE TYPED IN, and this literal has already been
 * wrong twice for exactly that reason - see the note at the transformer
 * descriptor, where `pairChannels: 128` read protenix2's 256-wide pair through
 * a 128-wide stride and `condChannels: 384` allocated boltz2's conditioning at
 * half its size. Both of those are derived now; these five were the rest.
 *
 * `q_projection/weights` states all of them:
 *
 *     [superBlocks, blocksPerSuperBlock, channels, heads, dimension]
 *     af3/boltz2/protenix2/if2/rf3/openbind0/opendde: [6, 4, 768, 16, 48]
 *
 * and `ffw_transition1/weights` is `[..., channels, hidden]`, where the
 * transition is a SwiGLU so its hidden axis is `channels * factor * 2`.
 *
 * 🔴 ALL SEVEN CHECKPOINTS AGREE ON THESE TODAY, so this changes no fold. It is
 * the eighth that pays: a bundle whose transformer is not AlphaFold 3's shape
 * gets the right widths, or - if the tensor is missing or malformed - a named
 * failure here rather than a validation error four stages downstream.
 */
function txShape(store, perBlockPair) {
  const stack = txStackFor(perBlockPair);
  const q = dims(store, `${stack}q_projection/weights`);
  if (q.length !== 5) {
    throw new Error(`the token transformer's q_projection is rank ${q.length}, not the`
      + " [superBlocks, blocksPerSuperBlock, channels, heads, dimension] this reads"
      + ` (${q.join("x")})`);
  }
  const [, blocksPerSuperBlock, channels, heads, dimension] = q;
  if (heads * dimension !== channels) {
    throw new Error(`the token transformer's ${heads} heads of ${dimension} do not make`
      + ` its ${channels} channels`);
  }
  const hidden = dims(store, `${stack}ffw_transition1/weights`).at(-1);
  // A SwiGLU transition projects to `factor * channels` TWICE, gate and value,
  // and the two live in one tensor.
  if (hidden % (channels * 2) !== 0) {
    throw new Error(`the token transformer's transition hidden ${hidden} is not a whole`
      + ` SwiGLU factor of ${channels} channels`);
  }
  return { channels, heads, dimension, blocksPerSuperBlock,
           transitionFactor: hidden / (channels * 2) };
}

const txStackName = (perBlockPair) =>
  perBlockPair ? "__layer_stack_no_per_layer" : "__layer_stack_with_per_layer";
const txStackFor = (perBlockPair) => {
  const name = txStackName(perBlockPair);
  return `${TX}/${name}/${name}/transformer`;
};

/**
 * One AdaLN cross-attention block, from either atom stack - as a DESCRIPTOR
 * whose leaves decode when they are read. See src/af3/weights/weights.js: the diffusion
 * head is 920 MiB of float32 and the device already holds all of it.
 */
function atomBlock(store, root, index) {
  const at = (leaf) => stacked(store, `${root}${leaf}`, index);
  const maybe = (leaf) => stackedIfPresent(store, `${root}${leaf}`, index);
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
    // rosettafold3's kq_norm in the ATOM stacks - see the token transformer's.
    // Their tensors sit under the same root the rest of the block uses, because
    // rf3 is in PER_BLOCK_ATOM_PAIR_LAYER_NORM and so takes the
    // `__layer_stack_no_per_layer` name either way.
    queryLayerNormScale: maybe("query_layer_norm/scale"),
    queryLayerNormOffset: maybe("query_layer_norm/offset"),
    keyLayerNormScale: maybe("key_layer_norm/scale"),
    keyLayerNormOffset: maybe("key_layer_norm/offset"),
    Transition2: at("transition2/weights"),
    AdaptiveZeroCondWeights: at("adaptive_zero_cond/weights"),
    AdaptiveZeroCondBias: at("adaptive_zero_cond/bias"),
    ffwSingleCondLayerNormScale: at("ffw_single_cond_layer_norm/scale"),
    ffwSingleCondScaleWeights: at("ffw_single_cond_scale/weights"),
    ffwSingleCondScaleBias: at("ffw_single_cond_scale/bias"),
    ffwSingleCondBias: at("ffw_single_cond_bias/weights"),
    ffwTransition1: at("ffw_transition1/weights"),
    // 🔴 boltz2's CONDITIONED TRANSITION HAS A THIRD PROJECTION. Its
    // ConditionedTransitionBlock is `SwiGLU(a) * a_to_b(a)` where every other
    // family here is `SwiGLU(a)` alone, so the up-gate multiplies the whole
    // intermediate before the down-projection. Four stacks carry it - the
    // trunk's atom encoder, the diffusion atom encoder and decoder, and the
    // token transformer - and without it boltz2's atom stack came out at
    // relRMS 8.18e-1 against af3-any-model with its pair logits exact to
    // 9.11e-4, which is the signature of a term missing INSIDE the block.
    // Null everywhere else, and `conditionedTransition` skips it then.
    ffwAToB: maybe("ffw_a_to_b/weights"),
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
             projection: Array.from({ length: blocks }, () => projection),
             // Already [C_PAIR, BLOCKS, HEADS]; the shader wants exactly this.
             packedProjection: projection };
  }
  const scale = [];
  const projection = [];
  for (let index = 0; index < blocks; index += 1) {
    scale.push(await layer(store, scaleName, index));
    projection.push(await layer(store, projectionName, index));
  }
  // 🔴 THE PER-BLOCK SCALE IS FOLDED INTO THE PER-BLOCK PROJECTION, and the
  // shared scale becomes ones - exactly what foldPerBlockPairNorm does for the
  // token transformer, and for the same reason.
  //
  // The atom DECODER's shader reads the projection per block
  // (`c * BLOCKS * HEADS + block * HEADS + head`) and the LayerNorm scale as a
  // SINGLE shared vector, so on a per-block model it applied BLOCK 0's scale to
  // every block. AF3 never noticed because its scale is shared already; on
  // protenix2 the GPU decoder read 1.87e-2 against its own CPU decoder's answer
  // where AlphaFold 3's reads 4.66e-7, and that compounded to 4.48e-1 over a
  // whole denoise step and to a fold whose bonds came out at 0.73x ideal.
  //
  // 🔴 AND OpenDDE HAS THE SAME FLAG, so it had the same defect and its fold
  // has been slightly wrong for as long as it has existed - nothing measured
  // its denoiser against a reference until now.
  //
  // The fold is exact: `sum_c n[c] * scale_b[c] * proj_b[c, h]` is
  // `sum_c n[c] * (scale_b[c] * proj_b[c, h])`. The CPU reference reads the
  // same arrays, so it sees ones and the folded projection and agrees by
  // construction rather than by a second implementation.
  const heads = projection[0].length / scale[0].length;
  const folded = projection.map((weights, index) => {
    const out = Float32Array.from(weights);
    for (let c = 0; c < scale[index].length; c += 1) {
      for (let h = 0; h < heads; h += 1) out[c * heads + h] *= scale[index][c];
    }
    return out;
  });
  // 🔴 AND THE SINGULAR `projection` MUST BE PACKED THE WAY THE DECODER'S
  // SHADER READS IT, WHICH IS NOT HOW A PER-BLOCK CHECKPOINT STORES IT.
  //
  //     AlphaFold 3   pair_logits_projection  [C_PAIR, BLOCKS, HEADS]   (16, 3, 4)
  //     protenix2     the same leaf           [BLOCKS, C_PAIR, HEADS]   (3, 16, 4)
  //
  // The shader indexes `c * BLOCKS * HEADS + block * HEADS + head`, so AF3's
  // whole tensor is already in its layout and `projection[0]` - which for a
  // SHARED norm is that whole tensor - is right by construction. On a per-block
  // model `projection[0]` is one block's [C_PAIR, HEADS], sixty-four of the
  // hundred and ninety-two floats the shader reads, and the other two blocks
  // read whatever follows. Repacked here, so the singular field means the same
  // thing for both.
  const packed = new Float32Array(scale[0].length * blocks * heads);
  for (let c = 0; c < scale[0].length; c += 1) {
    for (let b = 0; b < blocks; b += 1) {
      for (let h = 0; h < heads; h += 1) {
        packed[(c * blocks + b) * heads + h] = folded[b][c * heads + h];
      }
    }
  }
  return { perBlock: true, projection: folded, packedProjection: packed,
           scale: scale.map((one) => new Float32Array(one.length).fill(1)) };
}

/**
 * The one constant vector added to every atom's embedding, under either name.
 *
 * boltz2 calls it `embed_atom_features_bias` (its features go through ONE
 * biased Linear where AF3 sums five bias-free ones) and rosettafold3
 * `conformer_embedding_bias` (a collapsed MLP subtree whose input is zero and
 * whose output is not). No checkpoint carries both, and a checkpoint carrying
 * NEITHER - which is AF3, openbind0, opendde, protenix2 and intellifold2 - gets
 * null and the term does not exist.
 */
async function constantAtomBias(store, ...names) {
  const found = names.filter((name) => store.manifest?.tensors?.[name] !== undefined);
  if (found.length > 1) {
    throw new Error(`this bundle carries ${found.join(" and ")}; they are the `
      + "same term and a checkpoint with both is a converter bug, not a sum");
  }
  return found.length === 0 ? null : store.tensor(found[0]);
}

async function atomBlockWith(store, stack, index, dialect) {
  const block = await bind(store, atomBlock(store, stack, index));
  // 🔴 ONE LIST, IN dialect.js. Listing these here and again in every hand-built
  // weight dict is how `maskAtomActPerBlock` reached the loader and not
  // check-af3-atom-decoder.js, killing that differential silently.
  Object.assign(block, atomBlockDialect(dialect));
  // 🔴 CHAI-1 AND IntelliFold-2 RE-ZERO THE PADDED ATOM SLOTS AT THE TOP OF
  // EVERY BLOCK, because they pad the flat atom axis INSIDE each attention
  // call rather than once for the stack. Carried per block, like the other
  // two, so the DECODER - which sees no dialect object - reads it off its
  // weights the same way.
  // 🔴 rosettafold3's ATOM blocks take the same no_residual wiring its TOKEN
  // transformer does, and only the dialect says so - there is no tensor whose
  // presence marks it. Carried per block so the DECODER, which sees no dialect
  // object, reads it the way it reads the other three.
  if (block.diffusionNoResidual === undefined) {
    throw new Error("an atom block carries no diffusionNoResidual: AF3 adds the "
      + "attention and the transition through two residuals, rosettafold3 one");
  }
  if (block.chainedAtomLayerNorm === undefined
      || block.keyMaskedAtomAttention === undefined
      || block.maskAtomActPerBlock === undefined) {
    throw new Error("an atom block's dialect flags have no defaults");
  }
  return block;
}

export async function targetFeatureWeights(store) {
  const root = "diffuser/evoformer_conditioning";
  // 🔴 boltz2's `target_feat` IS A SUM OF SEVEN TERMS, NOT A CONCATENATION.
  // Everything else here lays out [restype 31 | profile 31 | deletion 1 |
  // atoms 384]; boltz2's InputEmbedder ADDS six bias-free projections onto the
  // atom encoder's token activation, all at seq_channel. Taking the atom half
  // alone - which is what `targetFeatAtomOnly` used to mean - put `target_feat`
  // at relRMS 1.00e+0 against af3-any-model with 0.39x its magnitude, and since
  // every other thing the trunk builds is a function of it, the fold came out a
  // 5.9 A ball while the denoise step was exact.
  const sum = store.manifest?.tensors?.["diffuser/boltz2_res_type_encoding/weights"]
    === undefined ? null : {
      resType: await store.tensor("diffuser/boltz2_res_type_encoding/weights"),
      msaProfile: await store.tensor("diffuser/boltz2_msa_profile_encoding/weights"),
      molType: await store.tensor("diffuser/boltz2_mol_type_conditioning/weights"),
      cyclic: await store.tensor("diffuser/boltz2_cyclic_conditioning/weights"),
      method: await store.tensor("diffuser/boltz2_method_conditioning/weights"),
      modified: await store.tensor("diffuser/boltz2_modified_conditioning/weights"),
    };
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
      // 🔴 boltz2 BUILDS THIS AS ONE Linear OVER THE CONCATENATED ATOM FEATURE
      // VECTOR, and a plain Linear at that - so it carries a bias that AF3's
      // per-feature bias-free Linears have no slot for. The reference measured
      // dropping it: a constant 128-vector of std 0.134 off EVERY atom's
      // embedding, about a quarter of the conditioning's own std, taking
      // per-atom corr to 0.912 with byte-identical inputs and carrying into
      // everything downstream.
      // 🔴 AND RoseTTAFold3 REACHES THE SAME SHAPE BY A DIFFERENT ROUTE, under
      // a different name. Its atom single rep also takes
      // `process_atom_level_embedding(f['atom_level_embedding'])`, and without
      // conformer embeddings that input is all ZEROS - but the MLP has biases
      // and its tail is a LayerNorm, so it emits a fixed NONZERO vector, the
      // same for every atom and two thirds the magnitude of the ref-feature
      // embedding. A zero feature is not a zero contribution. The reference's
      // converter collapses that subtree to one [128] constant, exactly as
      // boltz2's Linear bias is one, so the two share this field and the
      // forward needs no second branch.
      embedAtomFeaturesBias: await constantAtomBias(
        store, `${root}_embed_atom_features_bias`,
        `${root}_conformer_embedding_bias`),
    },
    encoder: {
      channels: 128, pairChannels: 16, heads: 4, dimension: 32, perTokenChannels: 384,
      // 🔴 THE `_1` FORM, AND THE TRACE THAT SAID OTHERWISE WAS TAKEN ON A
      // REGRESSED REFERENCE. Four of these exist twice in AlphaFold 3's own
      // checkpoint, unsuffixed and `_1`, at identical shapes: haiku numbers a
      // module the second time its constructor runs, and TWO different call
      // sites in `atom_cross_attention.py` build a Linear called
      // `<root>_single_to_pair_cond_row`. The first is inside
      // `_per_atom_conditioning`, over a token's own 24 dense slots; the second
      // is this encoder's, in the QUERIES-KEYS layout. So the unsuffixed set
      // belongs to the first and `_1` to this one.
      //
      // This file used to read the unsuffixed set, on the strength of an
      // `hk.intercept_methods` trace showing `_1` never firing. That trace was
      // taken after af3-any-model's 041ab187 ("stop computing three things the
      // models then throw away"), which deleted the FIRST call because its
      // result is assigned to `_`. Its result is - and deleting it renames the
      // SECOND call, so the encoder silently claimed the first one's weights.
      // Bisected on the A10 over 395 commits: run_alphafold.py on Google's own
      // af3.bin.zst folds 6MRR with mean CA-CB 1.5315 at 041ab187^ and 1.2610
      // at 041ab187, and reverting that one file alone restores 1.5315.
      //
      // 🔴 AND THE UNSUFFIXED SET IS NOT A SECOND TRAINED TENSOR, IT IS
      // UNTRAINED. Its output is discarded, so it never receives a gradient and
      // was serialised at its random initialisation. Against haiku's
      // `VarianceScaling(1.0, fan_in)`, whose std is `sqrt(1/fan_in)`, at three
      // fan-ins two orders apart: row [128,16] init 0.0884 against an actual
      // 0.0877, offsets [3,16] init 0.5774 against 0.5657, distances [1,16]
      // init 1.0 against 0.7953 (one sample standard error low for sixteen
      // draws). Their `_1` twins are 0.4063, 0.0137 and 0.1548 - nowhere near
      // init. So reading the unsuffixed set fed this encoder Google's own
      // accidental noise, which is why side chains came out at 0.72x their
      // extent with the backbone intact: an untrained atom pair bias is one
      // term among several and the token transformer still carries the fold.
      //
      // Every PORTED bundle writes one tensor into both names, which is why
      // protenix2, boltz2, if2 and rf3 were unaffected either way - verified by
      // hashing the shard bytes, 4 of 4 identical for rf3 and if2 and 4 of 4
      // different for af3. `embed_pair_offsets_valid` is the one with no `_1`
      // form, which is what made the set look like a typo.
      //
      // 🔴 THE DEAD BRANCH IS ALPHAFOLD 3's OWN, AND THIS PORT DOES NOT COPY
      // IT. Verified against google-deepmind/alphafold3's unmodified
      // `network/atom_cross_attention.py`: line 141 is
      // `token_atoms_single_cond, _ = _per_atom_conditioning(...)` with no
      // `need_pair` parameter at all, and the same four names are built at
      // 78/81/88/102 and again at 207/215/293/301. So AF3 really does build a
      // (tokens, 24, 24, c) tensor inside the denoiser - once per sampling step
      // per sample, ~1000 times in a 200-step 5-sample fold - and throw it
      // away. Reading `_1` here gets AF3's ANSWER without AF3's waste, because
      // nothing on this side ever builds the discarded half.
      //
      // Left exactly as it is on purpose. If upstream ever does want the waste
      // gone, deleting the call is the one thing that cannot work: a haiku
      // module's NAME is allocated by construction order, so the dead code is
      // load-bearing for the naming and removing it renames this encoder's
      // Linears onto the untrained tensors. Construct the four Linears and skip
      // only the einsums, or give every one of them an explicit name that does
      // not depend on what ran before it.
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
      pairLogitsProjection: pairNorm.packedProjection ?? pairNorm.projection[0],
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
      targetFeatSum: sum,
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

/**
 * A LayerNorm's trained OFFSET, where this bundle carries one.
 *
 * 🔴 TEN OF boltz2'S DIFFUSION LayerNorms ARE AFFINE AND AlphaFold 3's ARE
 * SCALE-ONLY, and reading the scale alone put boltz2's whole score model at
 * relRMS 1.60 - two uncorrelated tensors - with the very first thing built, the
 * pair conditioning's initial projection, already at 1.45e-1. An offset is a
 * per-channel constant added after the rescale, so dropping it is not a small
 * error anywhere it feeds an adaLN.
 *
 * 🔴 AND IT IS READ FROM THE BUNDLE, NOT FROM A TABLE. af3-any-model states the
 * ten scopes per model in `AFFINE_LAYER_NORMS`; here the converter has already
 * answered the same question by emitting the tensor or not, so asking the store
 * cannot drift from the weights the way a second list can. AF3's own bundle
 * carries offsets on the transitions and the trunk norms and is unaffected -
 * those call sites already read them.
 */
const offsetOf = async (store, name) =>
  (store.manifest?.tensors?.[name] === undefined ? null : await store.tensor(name));

/** The five reference embeddings the atom conditioning sums. */
export async function atomReference(store) {
  const T = (name) => store.tensor(`${HEAD}/${name}`);
  const O = (name) => offsetOf(store, `${HEAD}/${name}`);
  return {
    channels: 128,
    embedRefPos: await T("diffusion_embed_ref_pos/weights"),
    embedRefMask: await T("diffusion_embed_ref_mask/weights"),
    embedRefElement: await T("diffusion_embed_ref_element/weights"),
    embedRefCharge: await T("diffusion_embed_ref_charge/weights"),
    embedRefAtomName: await T("diffusion_embed_ref_atom_name/weights"),
    // ...and the diffusion head's own copy of it; see targetFeatureWeights.
    embedAtomFeaturesBias: await constantAtomBias(
      store, `${HEAD}/diffusion_embed_atom_features_bias`,
      `${HEAD}/diffusion_conformer_embedding_bias`),
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
  const O = (name) => offsetOf(store, `${HEAD}/${name}`);
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
  // 🔴 AND THERE IS A THIRD SHAPE, WHICH IS protenix2's. It projects the
  // relative encoding to the pair width and passes the trunk pair through at
  // ITS width, so the initial projection folds [z_trunk(c_z), relpe(c_z)]:
  //
  //     AF3        raw 139 relpos, trunk pair through   [267, 128]
  //     OpenDDE    both projected (z_trunk_projection)  [256, 128]
  //     protenix2  relpe projected, trunk pair through  [512, 256]
  //
  // Reading DIFFUSION_PROJECTED_RELPOS as `splitPairConditioning` put a true
  // here and the guard above caught it in one run, which is the whole reason
  // that guard exists.
  const projectedRelpos = dialect.projectedRelpos;
  if (projectedRelpos === undefined) {
    throw new Error("dialect.projectedRelpos has no default: AF3 concatenates "
      + "the RAW 139 relative-position features and protenix2 projects them to "
      + "the pair width first");
  }
  const hasRelpe = store.manifest?.tensors?.[`${HEAD}/relpe_projection/weights`] !== undefined;
  if (hasRelpe !== (splitPair || projectedRelpos)) {
    throw new Error(`this bundle ${hasRelpe ? "carries" : "does not carry"} `
      + "relpe_projection and its dialect says otherwise");
  }
  return {
    // 🔴 EVERY WIDTH HERE IS THE TENSOR'S. `pair_cond_initial_projection` is
    // [267, 128] under AlphaFold 3 and [256, 128] under OpenDDE, because the
    // second compresses its two inputs separately before concatenating them -
    // and OpenDDE's trunk pair arriving here is 384 wide, not 128.
    pairChannels: dims(store, `${HEAD}/pair_cond_initial_projection/weights`)[1],
    seqChannels: dims(store, `${HEAD}/single_cond_initial_projection/weights`)[1],
    // 🔴 THE CONDITIONING'S OUTPUT WIDTH AND THE SINGLE IT READS ARE TWO
    // NUMBERS, and AlphaFold 3 hides that by having them equal. Its projection
    // is [831, 384] - 384 out, and the trunk single it concatenates is also
    // 384 - so `seqChannels + targetFeatWidth` happened to be the input width.
    // boltz2's is [768, 768]: 768 out, 384 in. Read as one number that gives
    // 1152 against a LayerNorm of 768.
    trunkSingleChannels:
      dims(store, "diffuser/evoformer/single_activations/weights")[1],
    // 🔴 447 WAS TYPED IN, UNDER A COMMENT NAMING THAT EXACT FAULT. boltz2's
    // target_feat is 384 wide, not AlphaFold 3's 447, and the checkers read
    // "targetFeat has 10728 elements; expected 9216" - 447 against 384 over 24
    // tokens. It is derivable and never had to be a constant: the single
    // conditioning's LayerNorm states its INPUT width, which is the target
    // features plus the trunk single, plus two more where the dialect pads the
    // unknown-DNA columns.
    //
    //     af3        831 - 0 - 384 = 447
    //     protenix2  833 - 2 - 384 = 447
    //     boltz2     768 - 0 - 384 = 384
    // 🔴 FROM THE TENSOR THAT STATES IT, NOT FROM THE NORM THIS IS CHECKED
    // AGAINST. Deriving it as `scale - pad - trunkSingle` made the width
    // assertion in the reference VACUOUS: whatever the scale was, the derived
    // width absorbed it and the two could never disagree. A stale OpenDDE
    // bundle with an 831 scale then folded silently at target_feat 445 instead
    // of 447 - the exact shape of error that assertion exists to catch.
    // `single_activations` is [447, 384] and says 447 outright.
    targetFeatWidth: dims(store, "diffuser/evoformer/single_activations/weights")[0],
    relativeWidth: 139,
    trunkPairChannels: splitPair
      ? dims(store, `${HEAD}/z_trunk_projection/weights`)[0]
      // ...and where only the RELPOS is projected, the concatenation is two
      // equal halves, so the trunk pair's width is what relpe was projected TO.
      : projectedRelpos
        ? dims(store, `${HEAD}/pair_cond_initial_projection/weights`)[0]
          - dims(store, `${HEAD}/relpe_projection/weights`)[1]
        : dims(store, `${HEAD}/pair_cond_initial_projection/weights`)[0] - 139,
    pairCondInitialNormScale: await T("pair_cond_initial_norm/scale"),
    pairCondInitialNormOffset: await O("pair_cond_initial_norm/offset"),
    pairCondInitialProjection: await T("pair_cond_initial_projection/weights"),
    // OpenDDE's two separate compressions; absent under AlphaFold 3, and the
    // reference branches on their presence.
    ...(splitPair ? {
      zTrunkNormScale: await T("z_trunk_norm/scale"),
      zTrunkNormOffset: await O("z_trunk_norm/offset"),
      zTrunkProjection: await T("z_trunk_projection/weights"),
      relpeProjection: await T("relpe_projection/weights"),
    } : projectedRelpos ? {
      relpeProjection: await T("relpe_projection/weights"),
    } : {}),
    pairTransitions: [await transition("pair_transition_0"),
                      await transition("pair_transition_1")],
    singleCondInitialNormScale: await T("single_cond_initial_norm/scale"),
    singleCondInitialNormOffset: await O("single_cond_initial_norm/offset"),
    singleCondInitialProjection: await T("single_cond_initial_projection/weights"),
    // 🔴 boltz2's PROJECTION CARRIES A BIAS AND NOBODY ELSE'S DOES. An absent
    // bias is not a zero one here only because nothing read it: the single
    // conditioning came out 8.19e-1 from af3-any-model's while the PAIR half
    // was 2.69e-7, which is the signature of a missing additive term rather
    // than a wrong width.
    singleCondInitialProjectionBias:
      store.manifest?.tensors?.[`${HEAD}/single_cond_initial_projection/bias`] === undefined
        ? null : await T("single_cond_initial_projection/bias"),
    singleTransitions: [await transition("single_transition_0"),
                        await transition("single_transition_1")],
    fourierWeight: await T("fourier_embedding_weight"),
    fourierBias: await T("fourier_embedding_bias"),
    noiseEmbeddingInitialNormScale: await T("noise_embedding_initial_norm/scale"),
    noiseEmbeddingInitialNormOffset: await O("noise_embedding_initial_norm/offset"),
    noiseEmbeddingInitialProjection: await T("noise_embedding_initial_projection/weights"),
  };
}

export async function diffusionWeights(store, superBlocks = 6) {
  const T = (name) => store.tensor(`${HEAD}/${name}`);
  const O = (name) => offsetOf(store, `${HEAD}/${name}`);
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
  const maybeTx = (leaf, index) =>
    stackedIfPresent(store, `${txStackFor(perBlockPair)}${leaf}`, index, 2);
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
        // 🔴 rosettafold3's kq_norm: A TRAINED LayerNorm ON q AND k, over the
        // FLATTENED num_head * key_dim axis rather than per head, applied after
        // the projection and before the key_dim scaling. Only the diffusion
        // score-model transformers set it - the trunk and confidence
        // pairformers call the same reference function with it off - and no
        // other checkpoint carries the tensors, so `maybe` returns null and the
        // kernel is generated without the term.
        queryLayerNormScale: maybeTx("query_layer_norm/scale", s * 4 + inner),
        queryLayerNormOffset: maybeTx("query_layer_norm/offset", s * 4 + inner),
        keyLayerNormScale: maybeTx("key_layer_norm/scale", s * 4 + inner),
        keyLayerNormOffset: maybeTx("key_layer_norm/offset", s * 4 + inner),
        Transition2: at("transition2/weights"),
        AdaptiveZeroCondWeights: at("adaptive_zero_cond/weights"),
        AdaptiveZeroCondBias: at("adaptive_zero_cond/bias"),
        ffwSingleCondLayerNormScale: at("ffw_single_cond_layer_norm/scale"),
        ffwSingleCondScaleWeights: at("ffw_single_cond_scale/weights"),
        ffwSingleCondScaleBias: at("ffw_single_cond_scale/bias"),
        ffwSingleCondBias: at("ffw_single_cond_bias/weights"),
        ffwTransition1: at("ffw_transition1/weights"),
        // See `atomBlock`: boltz2's transition up-gate, nested two deep here.
        ffwAToB: stackedIfPresent(store,
          `${txStackFor(perBlockPair)}ffw_a_to_b/weights`, s * 4 + inner, 2),
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

  // Loaded before the table below so the transformer can take its conditioning
  // width from it rather than from a constant.
  const conditioning = await conditioningWeights(store, dialect);
  return {
    dialect: af3Dialect(store),
    // ...and the head's own single width is the conditioning's too: boltz2
    // embeds 768 where AlphaFold 3 embeds 384.
    seqChannels: conditioning.seqChannels, perTokenChannels: 768,
    singleCondEmbeddingNormScale: await T("single_cond_embedding_norm/scale"),
    singleCondEmbeddingNormOffset: await O("single_cond_embedding_norm/offset"),
    singleCondEmbeddingProjection: await T("single_cond_embedding_projection/weights"),
    outputNormScale: await T("output_norm/scale"),
    outputNormOffset: await O("output_norm/offset"),
    conditioning: conditioning,
    transformer: {
      // 🔴 `pairChannels: 128` WAS TYPED IN AND IT IS THE MODEL'S, NOT AF3's.
      // The token transformer reads the diffusion conditioning's PAIR, and
      // protenix2 widens that to 256 (PROTENIX2_SETTINGS widens
      // heads.diffusion.conditioning.pair_channel with the trunk). Its
      // `pair_logits_projection` is [6, 4, 256, 16] where AF3's is
      // [6, 4, 128, 16] - so the stack was reading a 256-wide pair through a
      // 128-wide stride, and every stage ran without complaint on a fold whose
      // backbone bonds came out at 0.96 A against an ideal 1.46.
      // 🔴 `condChannels: 384` WAS TYPED IN AND IT IS THE CONDITIONING'S OUTPUT
      // WIDTH. AlphaFold 3's single_cond_initial_projection is [831, 384] and
      // boltz2's is [768, 768], so the token transformer's conditioning buffer
      // was allocated at half the size it needed: "Write range (size: 208896)
      // does not fit in [Buffer difftx.cond] size (104448)" - exactly 2x, and a
      // validation error rather than a wrong answer only because the shapes
      // happened to be checkable.
      // 🔴 AND THE REMAINING FIVE COME OFF THE BUNDLE TOO, for the same reason
      // the two above it do. `channels: 768, heads: 16, dimension: 48,
      // transitionFactor: 2, blocksPerSuperBlock: 4` were typed in; every one
      // is stated by `q_projection/weights`, which is
      // [superBlocks, blocksPerSuperBlock, channels, heads, dimension]. See
      // txShape. All seven checkpoints agree on them today, so this moves no
      // fold - it is the eighth that pays.
      ...txShape(store, perBlockPair),
      condChannels: conditioning.seqChannels,
      pairChannels: txPairChannels(store, perBlockPair),
      // Shared under AlphaFold 3 and per block under OpenDDE, where the scale
      // lives inside the doubly-nested stack at [6, 4, 128].
      // 🔴 ALL ONES UNDER OpenDDE, because its per-block scales are folded into
      // the per-block projections above - see foldPerBlockPairNorm. The
      // LayerNorm itself still runs; only its affine has moved.
      pairNormPerBlock: perBlockPair,
      // 🔴 rosettafold3's BLOCK WIRING. Carried on the weights rather than
      // passed as a dialect, because that is how every other structural flag
      // reaches these stacks - see `chainedAtomLayerNorm`. False everywhere
      // else, and the encoder generates the kernels it always did.
      noResidual: dialect.diffusionNoResidual === true,
      pairInputLayerNormScale: perBlockPair
        ? new Float32Array(folded.channels).fill(1)
        : await store.tensor(`${TX}/pair_input_layer_norm/scale`),
      superBlocks: groups,
    },
    encoder: {
      channels: 128, pairChannels: 16, heads: 4, dimension: 32,
      perTokenChannels: 768, trunkSingleChannels: 384,
      // ...and the atom encoder's trunk pair, stated by the tensor that embeds
      // it: [128, 16] under AF3 and [256, 16] under protenix2.
      trunkPairChannels:
        dims(store, `${HEAD}/diffusion_embed_trunk_pair_cond/weights`)[0],
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
      // The `_1` form, which is what the paragraph above says and what this
      // line used to contradict. See `targetFeatureWeights` for the bisect.
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
      pairLogitsProjection: encoderPairNorm.packedProjection ?? encoderPairNorm.projection[0],
      pairInputLayerNormScales: encoderPairNorm.scale,
      pairLogitsProjections: encoderPairNorm.projection,
      pairNormPerBlock: encoderPairNorm.perBlock,
      lnormTrunkSingleCondScale: await T("diffusion_lnorm_trunk_single_cond/scale"),
      lnormTrunkSingleCondOffset: await O("diffusion_lnorm_trunk_single_cond/offset"),
      embedTrunkSingleCond: await T("diffusion_embed_trunk_single_cond/weights"),
      lnormTrunkPairCondScale: await T("diffusion_lnorm_trunk_pair_cond/scale"),
      lnormTrunkPairCondOffset: await O("diffusion_lnorm_trunk_pair_cond/offset"),
      embedTrunkPairCond: await T("diffusion_embed_trunk_pair_cond/weights"),
      atomPositionsToFeatures: await T("diffusion_atom_positions_to_features/weights"),
      // 🔴 rosettafold3's CHIRALITY PROJECTION, and the DIFFUSION encoder's
      // alone. The trunk's input embedder passes no coordinates
      // (`token_atoms_act=None` there), so the term has nowhere to enter; only
      // this stack sees the noisy structure. [3, 128], beside the positions
      // projection it is added to. See src/af3/diffusion/chiral-gradient.js.
      atomChiralToFeatures:
        store.manifest?.tensors?.[`${HEAD}/diffusion_atom_chiral_to_features/weights`]
          === undefined ? null : await T("diffusion_atom_chiral_to_features/weights"),
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
      pairLogitsProjection: decoderPairNorm.packedProjection ?? decoderPairNorm.projection[0],
      pairInputLayerNormScales: decoderPairNorm.scale,
      pairLogitsProjections: decoderPairNorm.projection,
      pairNormPerBlock: decoderPairNorm.perBlock,
      projectTokenFeaturesForBroadcast:
        await T("diffusion_project_token_features_for_broadcast/weights"),
      atomFeaturesLayerNormScale: await T("diffusion_atom_features_layer_norm/scale"),
      atomFeaturesLayerNormOffset: await O("diffusion_atom_features_layer_norm/offset"),
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
  const trunk = await trunkWeights(store, options.blocks, options.msaBlocks,
    { allowPrefix: true });
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
