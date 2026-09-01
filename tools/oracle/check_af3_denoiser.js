/**
 * Run one AF3 denoising step and compare the coordinates it produces.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 48 --diffusion 1 --float32 \
 *       --capture 'diffusion_head/__call__$|evoformer/__call__$' \
 *       --capture-args 'diffusion_head/__call__$' \
 *       --out af3-oracle-denoiser-f32.json
 *     node tools/oracle/check_af3_denoiser.js
 *
 * 🔴 THE NOISY POSITIONS ARE AN INPUT AND CANNOT BE INFERRED. The sampler draws
 * them from a PRNG inside its own scan, and nothing downstream is an invertible
 * function of them - so the dumper records the diffusion head's ARGUMENTS as
 * well as its outputs (--capture-args), and this feeds back exactly what AF3
 * denoised. Anything else would be testing a different question.
 *
 * That also fixes the noise level: it arrives as argument 1 rather than being
 * derived from the schedule, and reads 2560 - which is what the schedule says
 * for a single-step run, so the two agree.
 */
import { join } from "node:path";

import { atomCrossAttentionEncoder } from "../../src/af3/atom-encoder-reference.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { diffusionHead } from "../../src/af3/diffusion-reference.js";
import { ROOT, captures, layer, loadDump, loadTensors, report } from "./af3-bundle.js";

const HEAD = "diffuser/~/diffusion_head";
const EVO = "diffuser/evoformer";

/** One block of a (superBlocks, perSuper, ...) doubly-stacked tensor. */
function nested(tensors, name, superBlock, inner) {
  const tensor = tensors.get(name);
  if (tensor === undefined) throw new Error(`no tensor named ${name}`);
  const perSuper = tensor.shape[1];
  const stride = tensor.data.length / (tensor.shape[0] * perSuper);
  const start = (superBlock * perSuper + inner) * stride;
  return tensor.data.subarray(start, start + stride);
}

/**
 * The AdaLN weights common to every attention-plus-transition block.
 *
 * The self-attention conditioning leaves (single_cond_*) are added by the
 * caller when they exist; see atomStack for why a cross-attention block's do
 * not.
 */
function adaptiveBlock(at, selfConditioned = false) {
  const block = {
    qProjection: at("q_projection/weights"), qBias: at("q_projection/bias"),
    kProjection: at("k_projection/weights"), vProjection: at("v_projection/weights"),
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
  if (selfConditioned) {
    block.SingleCondLayerNormScale = at("single_cond_layer_norm/scale");
    block.SingleCondScaleWeights = at("single_cond_scale/weights");
    block.SingleCondScaleBias = at("single_cond_scale/bias");
    block.SingleCondBias = at("single_cond_bias/weights");
  }
  return block;
}

/**
 * The three-block atom stacks, whose weights carry the module name twice.
 *
 * 🔴 A CROSS-ATTENTION BLOCK CONDITIONS ITS QUERIES AND KEYS SEPARATELY, so its
 * leaves are qsingle_cond_* and ksingle_cond_* where the token transformer's
 * self-attention has one single_cond_*. The two blocks are otherwise identical
 * and adaptiveBlock's names would resolve to nothing here rather than to the
 * wrong tensor - which is the good failure, but only because haiku happens to
 * name them differently.
 */
function atomStack(tensors, scope, blocks) {
  const inner = scope.split("/").pop();
  const prefix = `${scope}/__layer_stack_with_per_layer/${inner}`;
  return Array.from({ length: blocks }, (_, index) => {
    const at = (leaf) => layer(tensors, `${prefix}${leaf}`, index);
    const block = adaptiveBlock(at);
    for (const side of ["q", "k"]) {
      block[`${side}SingleCondLayerNormScale`] = at(`${side}single_cond_layer_norm/scale`);
      block[`${side}SingleCondScaleWeights`] = at(`${side}single_cond_scale/weights`);
      block[`${side}SingleCondScaleBias`] = at(`${side}single_cond_scale/bias`);
      block[`${side}SingleCondBias`] = at(`${side}single_cond_bias/weights`);
    }
    return block;
  });
}

async function main() {
  const model = process.argv.includes("--model")
    ? process.argv[process.argv.indexOf("--model") + 1] : "model-af3-full-f32";
  const dump = await loadDump("af3-oracle-denoiser-f32.json");
  const { tensors } = await loadTensors(join(ROOT, model));
  const at = captures(dump, "dump_af3_trunk.py with --capture-args"
    + " (see the header of this file)");
  const T = (name) => {
    const tensor = tensors.get(`${HEAD}/${name}`);
    if (tensor === undefined) {
      throw new Error(`no tensor ${HEAD}/${name}; export with --include diffuser`);
    }
    return tensor.data;
  };
  const input = (name) => dump.inputs[name].data;

  const tokens = dump.tokens;
  const dense = 24;
  // 🔴 DERIVED, NOT 9 - see check_af3_trunk.js. A constant here means the
  // denoiser is only ever checked on the 12-token dump the number was taken
  // from, and silently reads a fraction of the atoms on anything larger.
  const shape = { tokens, dense, queries: 32, keys: 128,
    subsets: dump.inputs["queries_to_keys:gather_idxs"].data.length / 128 };
  const gather = (name, count) => ({
    indices: input(`${name}:gather_idxs`), mask: input(`${name}:gather_mask`), count,
  });
  const reference = {
    positions: Float32Array.from(input("ref_pos")),
    mask: Float32Array.from(input("ref_mask")),
    element: input("ref_element"),
    charge: Float32Array.from(input("ref_charge")),
    atomNameChars: input("ref_atom_name_chars"),
  };

  const transitionWeights = (name) => ({
    ffwLayerNormScale: T(`${name}ffw_layer_norm/scale`),
    ffwLayerNormOffset: T(`${name}ffw_layer_norm/offset`),
    ffwTransition1: T(`${name}ffw_transition1/weights`),
    ffwTransition2: T(`${name}ffw_transition2/weights`),
  });

  const noiseLevel = at(`${HEAD}/__call__<1`)[0];
  const positionsNoisy = at(`${HEAD}/__call__<0`);

  // 🔴 A NaN IS NOT A TOLERANCE FAILURE and must not be reported as one. relRMS
  // comes back NaN and `worst` stays 0, because every comparison against a NaN
  // is false - so the summary line reads like a near-miss when the output is
  // not a number at all. Whatever is finite before the model runs is worth
  // saying so explicitly, since a NaN in an INPUT and a NaN produced by the
  // arithmetic are different bugs.
  if (process.argv.includes("--finite")) {
    const check = (name, values) => {
      let nan = 0;
      let big = 0;
      for (const value of values) {
        if (Number.isNaN(value)) nan += 1;
        else if (!Number.isFinite(value) || Math.abs(value) > 1e12) big += 1;
      }
      console.log(`    ${name.padEnd(22)} ${values.length} values`
        + `  NaN ${nan}  huge/inf ${big}`);
    };
    console.log("  inputs, before anything runs:");
    check("noise level", [noiseLevel]);
    check("positions noisy", positionsNoisy);
    check("ref_pos", reference.positions);
    check("ref_mask", reference.mask);
    check("ref_charge", reference.charge);
    for (const name of ["queries_to_keys", "queries_to_token_atoms",
      "token_atoms_to_queries", "tokens_to_queries", "tokens_to_keys"]) {
      if (dump.inputs[`${name}:gather_idxs`] !== undefined) {
        check(name, dump.inputs[`${name}:gather_idxs`].data);
      }
    }
  }

  const superBlocks = Array.from({ length: 6 }, (_, superBlock) => ({
    pairLogitsProjection: layer(
      tensors, `${HEAD}/transformer/__layer_stack_with_per_layer/pair_logits_projection/weights`,
      superBlock),
    blocks: Array.from({ length: 4 }, (_, inner) => adaptiveBlock((leaf) =>
      // 🔴 TWO NESTED STACKS, AND THE PAIR PROJECTION SITS UNDER ONLY ONE.
      // The block weights are (6, 4, ...) - super-blocks then blocks - so their
      // scope carries __layer_stack_with_per_layer twice; pair_logits_projection
      // is (6, 128, 4, 16), one stack axis and the 4 as a real dimension.
      nested(tensors,
             `${HEAD}/transformer/__layer_stack_with_per_layer`
             + `/__layer_stack_with_per_layer/transformer${leaf}`,
             superBlock, inner), true)),
  }));

  const atomEncoderWeights = {
    channels: 128, pairChannels: 16, heads: 4, dimension: 32, perTokenChannels: 768,
    trunkSingleChannels: 384, trunkPairChannels: 128,
    embedRefPos: T("diffusion_embed_ref_pos/weights"),
    embedRefMask: T("diffusion_embed_ref_mask/weights"),
    embedRefElement: T("diffusion_embed_ref_element/weights"),
    embedRefCharge: T("diffusion_embed_ref_charge/weights"),
    embedRefAtomName: T("diffusion_embed_ref_atom_name/weights"),
    singleToPairCondRow: T("diffusion_single_to_pair_cond_row_1/weights"),
    singleToPairCondCol: T("diffusion_single_to_pair_cond_col_1/weights"),
    embedPairOffsets: T("diffusion_embed_pair_offsets_1/weights"),
    embedPairDistances: T("diffusion_embed_pair_distances_1/weights"),
    embedPairOffsetsValid: T("diffusion_embed_pair_offsets_valid/weights"),
    pairMlp1: T("diffusion_pair_mlp_1/weights"),
    pairMlp2: T("diffusion_pair_mlp_2/weights"),
    pairMlp3: T("diffusion_pair_mlp_3/weights"),
    pairInputLayerNormScale: T("diffusion_atom_transformer_encoder/pair_input_layer_norm/scale"),
    pairLogitsProjection: T("diffusion_atom_transformer_encoder/pair_logits_projection/weights"),
    projectAtomFeaturesForAggr: T("diffusion_project_atom_features_for_aggr/weights"),
    atomPositionsToFeatures: T("diffusion_atom_positions_to_features/weights"),
    lnormTrunkSingleCondScale: T("diffusion_lnorm_trunk_single_cond/scale"),
    embedTrunkSingleCond: T("diffusion_embed_trunk_single_cond/weights"),
    lnormTrunkPairCondScale: T("diffusion_lnorm_trunk_pair_cond/scale"),
    embedTrunkPairCond: T("diffusion_embed_trunk_pair_cond/weights"),
    blocks: atomStack(tensors, `${HEAD}/diffusion_atom_transformer_encoder`, 3),
  };

  // The per-atom conditioning uses the DIFFUSION head's own copies of the five
  // reference embeddings, not the trunk's.
  const conditioning = perAtomConditioning(reference, tokens, dense, {
    channels: 128,
    embedRefPos: atomEncoderWeights.embedRefPos,
    embedRefMask: atomEncoderWeights.embedRefMask,
    embedRefElement: atomEncoderWeights.embedRefElement,
    embedRefCharge: atomEncoderWeights.embedRefCharge,
    embedRefAtomName: atomEncoderWeights.embedRefAtomName,
  });

  const ours = diffusionHead({
    shape,
    positionsNoisy,
    noiseLevel,
    atomMask: reference.mask,
    seqMask: Float32Array.from(input("seq_mask")),
    trunkSingle: at(`${EVO}/__call__:single`),
    trunkPair: at(`${EVO}/__call__:pair`),
    targetFeat: at(`${EVO}/__call__:target_feat`),
    conditioning,
    refPos: reference.positions,
    refSpaceUid: Float32Array.from(input("ref_space_uid")),
    // 🔴 shape.subsets, NOT 9. Fixing the constant in `shape` above and leaving
    // it here gave the encoder a 9-subset query layout while the decoder used
    // the real one, so the decoder indexed past the end of every mask the
    // encoder returned - and the coordinates came back NaN with the checker
    // reporting relRMS NaN and worst 0, which reads like a near-miss.
    tokenAtomsToQueries: gather("token_atoms_to_queries", shape.subsets * shape.queries),
    queriesToKeys: gather("queries_to_keys", shape.subsets * shape.keys),
    queriesToTokenAtoms: gather("queries_to_token_atoms", tokens * dense),
    tokensToQueries: gather("tokens_to_queries", shape.subsets * shape.queries),
    tokensToKeys: gather("tokens_to_keys", shape.subsets * shape.keys),
    features: {
      residueIndex: input("residue_index"), tokenIndex: input("token_index"),
      asymId: input("asym_id"), entityId: input("entity_id"), symId: input("sym_id"),
    },
  }, {
    seqChannels: 384, perTokenChannels: 768,
    conditioning: {
      pairChannels: 128, seqChannels: 384, targetFeatWidth: 447, relativeWidth: 139,
      pairCondInitialNormScale: T("pair_cond_initial_norm/scale"),
      pairCondInitialProjection: T("pair_cond_initial_projection/weights"),
      pairTransitions: [transitionWeights("pair_transition_0"),
                        transitionWeights("pair_transition_1")],
      singleCondInitialNormScale: T("single_cond_initial_norm/scale"),
      singleCondInitialProjection: T("single_cond_initial_projection/weights"),
      noiseEmbeddingInitialNormScale: T("noise_embedding_initial_norm/scale"),
      noiseEmbeddingInitialProjection: T("noise_embedding_initial_projection/weights"),
      singleTransitions: [transitionWeights("single_transition_0"),
                          transitionWeights("single_transition_1")],
      fourierWeight: T("fourier_embedding_weight"),
      fourierBias: T("fourier_embedding_bias"),
    },
    encoder: atomEncoderWeights,
    singleCondEmbeddingNormScale: T("single_cond_embedding_norm/scale"),
    singleCondEmbeddingProjection: T("single_cond_embedding_projection/weights"),
    transformer: {
      channels: 768, condChannels: 384, pairChannels: 128,
      heads: 16, dimension: 48, blocksPerSuperBlock: 4, transitionFactor: 2,
      pairInputLayerNormScale: T("transformer/pair_input_layer_norm/scale"),
      superBlocks,
    },
    outputNormScale: T("output_norm/scale"),
    decoder: {
      channels: 128, pairChannels: 16, heads: 4, dimension: 32, perTokenChannels: 768,
      projectTokenFeaturesForBroadcast: T("diffusion_project_token_features_for_broadcast/weights"),
      pairInputLayerNormScale: T("diffusion_atom_transformer_decoder/pair_input_layer_norm/scale"),
      pairLogitsProjection: T("diffusion_atom_transformer_decoder/pair_logits_projection/weights"),
      atomFeaturesLayerNormScale: T("diffusion_atom_features_layer_norm/scale"),
      atomFeaturesToPositionUpdate: T("diffusion_atom_features_to_position_update/weights"),
      blocks: atomStack(tensors, `${HEAD}/diffusion_atom_transformer_decoder`, 3),
    },
  }, atomCrossAttentionEncoder,
  // 🔴 WHICH STAGE, when the answer is NaN. relRMS comes back NaN and `worst`
  // stays zero, so the summary line looks like a near-miss rather than an
  // output that is not a number - and five stages produce it.
  process.argv.includes("--finite")
    ? (stage, values) => {
      let nan = 0;
      let worst = 0;
      for (const value of values) {
        if (Number.isNaN(value)) nan += 1;
        else if (Math.abs(value) > worst) worst = Math.abs(value);
      }
      console.log(`    ${stage.padEnd(22)} ${String(values.length).padStart(8)} values`
        + `  NaN ${String(nan).padStart(7)}  max ${worst.toExponential(2)}`);
    }
    : undefined);

  console.log(`${dump.model}, ${tokens} tokens, one denoising step at noise level`
    + ` ${noiseLevel}, weights from ${model}/`);
  report("atoms", at(`${HEAD}/__call__`), ours);
}

await main();
