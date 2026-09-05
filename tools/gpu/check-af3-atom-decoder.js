/**
 * AF3's atom decoder: GPU against src/af3/diffusion-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-atom-decoder.js
 *
 * 🔴 THE DECODER IS FED THE CPU ENCODER'S OUTPUTS, not the GPU encoder's. That
 * isolates it: a disagreement here is the decoder's, not something inherited.
 * The encoder has its own checker.
 *
 * 🔴 THE GATHERS COME FROM A REAL BATCH, not a synthetic layout. The five of
 * them - token_atoms_to_queries, queries_to_keys, queries_to_token_atoms,
 * tokens_to_queries, tokens_to_keys - encode which atoms share a window, which
 * share a reference conformer, and which slots are padding. A hand-built layout
 * would be regular in ways a real one is not, and the padding is where the
 * interesting disagreements live: two thirds of the key slots are empty.
 *
 * The dump is oracle-dumps/af3-oracle-atom-f32.json, produced by
 * tools/oracle/dump_af3_trunk.py --diffusion 1 --float32.
 *
 * 🔴 ONE TRAP HERE HAS NO DISCRIMINATING CONTROL ON THIS BATCH, and saying so
 * is better than implying otherwise. Masking the queries' conditioning BEFORE
 * the keys gather from it is documented as worth 8.4e-2 - but removing that
 * mask entirely changes NOTHING on this input, because the batch's
 * token_atoms_to_queries mask is zero exactly where the atom mask is, so the
 * multiplication is redundant here. The trap needs a batch where the trunk
 * conditioning lands in a slot the gather calls live and the atom mask calls
 * padding. The mask bias being a PRODUCT rather than a sum does have a control:
 * summing scores 4.3e-4 and 7.3e-4, 312x and 249x the envelope.
 */
import { atomCrossAttentionEncoder } from "../../src/af3/atom-encoder-reference.js";
import { atomDecoder } from "../../src/af3/diffusion-reference.js";
import { Af3AtomDecoderGpu } from "../../src/af3/atom-decoder-webgpu.js";
import { openAf3Store } from "../../src/af3/weights.js";

const DUMP = "/oracle-dumps/af3-oracle-atom-f32.json";
const HEAD = "diffuser/~/diffusion_head";
const ENCODER = `${HEAD}/diffusion_atom_transformer_encoder`;
const DECODER = `${HEAD}/diffusion_atom_transformer_decoder`;
const STACK = `${ENCODER}/__layer_stack_with_per_layer/diffusion_atom_transformer_encoder`;
const DECODER_STACK = `${DECODER}/__layer_stack_with_per_layer/diffusion_atom_transformer_decoder`;

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

function deterministic(length, seed) {
  let state = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    output[index] = (((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000) * 2 - 1;
  }
  return output;
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const difference = actual[index] - expected[index];
    error += difference * difference;
    scale += expected[index] * expected[index];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const response = await fetch(DUMP);
  if (!response.ok) throw new Error(`failed to load ${DUMP}: ${response.status}`);
  const dump = await response.json();
  const inputs = dump.inputs;
  const raw = (name) => inputs[name].data;
  const gather = (name, count) => ({
    indices: raw(`${name}:gather_idxs`),
    mask: raw(`${name}:gather_mask`),
    count,
  });

  const tokens = dump.tokens;
  const dense = 24;
  const subsets = 9;
  const queries = 32;
  const keys = 128;
  const store = await openAf3Store();
  const T = (name) => store.tensor(name);

  const blockSlice = async (leaf, index) => {
    const name = `${STACK}${leaf}`;
    const whole = await store.tensor(name);
    const stride = whole.length / store.shape(name)[0];
    return whole.subarray(index * stride, (index + 1) * stride);
  };
  const blockWeights = async (index) => {
    const at = (leaf) => blockSlice(leaf, index);
    return {
      qSingleCondLayerNormScale: await at("qsingle_cond_layer_norm/scale"),
      qSingleCondScaleWeights: await at("qsingle_cond_scale/weights"),
      qSingleCondScaleBias: await at("qsingle_cond_scale/bias"),
      qSingleCondBias: await at("qsingle_cond_bias/weights"),
      kSingleCondLayerNormScale: await at("ksingle_cond_layer_norm/scale"),
      kSingleCondScaleWeights: await at("ksingle_cond_scale/weights"),
      kSingleCondScaleBias: await at("ksingle_cond_scale/bias"),
      kSingleCondBias: await at("ksingle_cond_bias/weights"),
      qProjection: await at("q_projection/weights"),
      qBias: await at("q_projection/bias"),
      kProjection: await at("k_projection/weights"),
      vProjection: await at("v_projection/weights"),
      gatingQuery: await at("gating_query/weights"),
      Transition2: await at("transition2/weights"),
      AdaptiveZeroCondWeights: await at("adaptive_zero_cond/weights"),
      AdaptiveZeroCondBias: await at("adaptive_zero_cond/bias"),
      ffwSingleCondLayerNormScale: await at("ffw_single_cond_layer_norm/scale"),
      ffwSingleCondScaleWeights: await at("ffw_single_cond_scale/weights"),
      ffwSingleCondScaleBias: await at("ffw_single_cond_scale/bias"),
      ffwSingleCondBias: await at("ffw_single_cond_bias/weights"),
      ffwTransition1: await at("ffw_transition1/weights"),
      ffwTransition2: await at("ffw_transition2/weights"),
      ffwAdaptiveZeroCondWeights: await at("ffw_adaptive_zero_cond/weights"),
      ffwAdaptiveZeroCondBias: await at("ffw_adaptive_zero_cond/bias"),
    };
  };

  const weights = {
    channels: 128, pairChannels: 16, heads: 4, dimension: 32,
    perTokenChannels: 768, trunkSingleChannels: 384, trunkPairChannels: 128,
    singleToPairCondRow: await T(`${HEAD}/diffusion_single_to_pair_cond_row/weights`),
    singleToPairCondCol: await T(`${HEAD}/diffusion_single_to_pair_cond_col/weights`),
    embedPairOffsets: await T(`${HEAD}/diffusion_embed_pair_offsets/weights`),
    embedPairDistances: await T(`${HEAD}/diffusion_embed_pair_distances/weights`),
    embedPairOffsetsValid: await T(`${HEAD}/diffusion_embed_pair_offsets_valid/weights`),
    pairMlp1: await T(`${HEAD}/diffusion_pair_mlp_1/weights`),
    pairMlp2: await T(`${HEAD}/diffusion_pair_mlp_2/weights`),
    pairMlp3: await T(`${HEAD}/diffusion_pair_mlp_3/weights`),
    pairInputLayerNormScale: await T(`${ENCODER}/pair_input_layer_norm/scale`),
    pairLogitsProjection: await T(`${ENCODER}/pair_logits_projection/weights`),
    lnormTrunkSingleCondScale: await T(`${HEAD}/diffusion_lnorm_trunk_single_cond/scale`),
    embedTrunkSingleCond: await T(`${HEAD}/diffusion_embed_trunk_single_cond/weights`),
    lnormTrunkPairCondScale: await T(`${HEAD}/diffusion_lnorm_trunk_pair_cond/scale`),
    embedTrunkPairCond: await T(`${HEAD}/diffusion_embed_trunk_pair_cond/weights`),
    atomPositionsToFeatures: await T(`${HEAD}/diffusion_atom_positions_to_features/weights`),
    projectAtomFeaturesForAggr: await T(`${HEAD}/diffusion_project_atom_features_for_aggr/weights`),
    blocks: [await blockWeights(0), await blockWeights(1), await blockWeights(2)],
  };

  const input = {
    shape: { tokens, dense, subsets, queries, keys },
    // The per-atom conditioning is an INPUT to the encoder, built by
    // _per_atom_conditioning, so a deterministic stand-in exercises the kernel
    // without dragging that module in.
    conditioning: deterministic(tokens * dense * 128, 909),
    atomMask: raw("pred_dense_atom_mask"),
    refPos: raw("ref_pos"),
    refSpaceUid: raw("ref_space_uid"),
    tokenAtomsToQueries: gather("token_atoms_to_queries", subsets * queries),
    queriesToKeys: gather("queries_to_keys", subsets * keys),
    queriesToTokenAtoms: gather("queries_to_token_atoms", tokens * dense),
    tokensToQueries: gather("tokens_to_queries", subsets * queries),
    tokensToKeys: gather("tokens_to_keys", subsets * keys),
    tokenAtomsAct: deterministic(tokens * dense * 3, 707),
    trunkSingleCond: deterministic(tokens * 384, 606),
    trunkPairCond: deterministic(tokens * tokens * 128, 505),
  };

  const encoded = atomCrossAttentionEncoder(input, weights);

  const decoderSlice = async (leaf, index) => {
    const name = `${DECODER_STACK}${leaf}`;
    const whole = await store.tensor(name);
    const stride = whole.length / store.shape(name)[0];
    return whole.subarray(index * stride, (index + 1) * stride);
  };
  const decoderBlock = async (index) => {
    const at = (leaf) => decoderSlice(leaf, index);
    return {
      qSingleCondLayerNormScale: await at("qsingle_cond_layer_norm/scale"),
      qSingleCondScaleWeights: await at("qsingle_cond_scale/weights"),
      qSingleCondScaleBias: await at("qsingle_cond_scale/bias"),
      qSingleCondBias: await at("qsingle_cond_bias/weights"),
      kSingleCondLayerNormScale: await at("ksingle_cond_layer_norm/scale"),
      kSingleCondScaleWeights: await at("ksingle_cond_scale/weights"),
      kSingleCondScaleBias: await at("ksingle_cond_scale/bias"),
      kSingleCondBias: await at("ksingle_cond_bias/weights"),
      qProjection: await at("q_projection/weights"),
      qBias: await at("q_projection/bias"),
      kProjection: await at("k_projection/weights"),
      vProjection: await at("v_projection/weights"),
      gatingQuery: await at("gating_query/weights"),
      Transition2: await at("transition2/weights"),
      AdaptiveZeroCondWeights: await at("adaptive_zero_cond/weights"),
      AdaptiveZeroCondBias: await at("adaptive_zero_cond/bias"),
      ffwSingleCondLayerNormScale: await at("ffw_single_cond_layer_norm/scale"),
      ffwSingleCondScaleWeights: await at("ffw_single_cond_scale/weights"),
      ffwSingleCondScaleBias: await at("ffw_single_cond_scale/bias"),
      ffwSingleCondBias: await at("ffw_single_cond_bias/weights"),
      ffwTransition1: await at("ffw_transition1/weights"),
      ffwTransition2: await at("ffw_transition2/weights"),
      ffwAdaptiveZeroCondWeights: await at("ffw_adaptive_zero_cond/weights"),
      ffwAdaptiveZeroCondBias: await at("ffw_adaptive_zero_cond/bias"),
    };
  };
  const decoderWeights = {
    channels: 128, pairChannels: 16, heads: 4, dimension: 32, perTokenChannels: 768,
    pairInputLayerNormScale: await T(`${DECODER}/pair_input_layer_norm/scale`),
    pairLogitsProjection: await T(`${DECODER}/pair_logits_projection/weights`),
    projectTokenFeaturesForBroadcast:
      await T(`${HEAD}/diffusion_project_token_features_for_broadcast/weights`),
    atomFeaturesLayerNormScale: await T(`${HEAD}/diffusion_atom_features_layer_norm/scale`),
    atomFeaturesToPositionUpdate:
      await T(`${HEAD}/diffusion_atom_features_to_position_update/weights`),
    blocks: [await decoderBlock(0), await decoderBlock(1), await decoderBlock(2)],
  };

  const tokenActInput = deterministic(tokens * 768, 4321);
  const expected = atomDecoder(tokenActInput, encoded, input, decoderWeights);
  const gpu = await new Af3AtomDecoderGpu(device)
    .run(tokenActInput, encoded, input, decoderWeights);
  const relRms = relativeRms(gpu.update, expected);
  console.log(`position update\trelRMS ${relRms.toExponential(2)}`
    + `\t${gpu.elapsedMilliseconds.toFixed(1)} ms`);
  if (relRms > 1e-5) throw new Error(`relRMS ${relRms.toExponential(2)} exceeds 1e-5`);
  return { tokens, relRms };
}
