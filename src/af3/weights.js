/**
 * Loading AF3's weights, over http - for the checkers and for the page.
 *
 * The per-kernel checkers each build the one bundle they need inline, which
 * keeps them readable on their own. Anything that needs a WHOLE STAGE - the
 * trunk, and the diffusion and confidence heads after it - wants this instead:
 * the bundles are long, and a typo in one leaf name of one of them surfaces as
 * a numerical disagreement rather than a missing key.
 */
import { HttpTensorStore } from "../reference/http-tensor-store.js";

export const MANIFEST = "/model-af3-full-f32/manifest.json";
const EVO = "diffuser/evoformer";
const MSA_STACK = `${EVO}/__layer_stack_no_per_layer/msa_stack`;
const PAIRFORMER = `${EVO}/__layer_stack_no_per_layer_1/trunk_pairformer`;
const TEMPLATE = `${EVO}/template_embedding`;
const TEMPLATE_SINGLE = `${TEMPLATE}/single_template_embedding`;
const CONFIDENCE = "diffuser/confidence_head";
const CONFIDENCE_STACK = `${CONFIDENCE}/__layer_stack_no_per_layer/confidence_pairformer`;
const TEMPLATE_STACK =
  `${TEMPLATE_SINGLE}/__layer_stack_no_per_layer/template_embedding_iteration`;

/**
 * Quantise-dequantise a tensor in place, so the model runs at a storage
 * precision without needing a packed format or new kernels yet.
 *
 * 🔴 THIS MEASURES ACCURACY, NOT SIZE. The values come back as float32, so
 * nothing gets smaller here - what it answers is whether a scheme is USABLE,
 * which is the question that has to be settled before packing anything.
 *
 * @param {"sym"|"asym"} mode asymmetric carries a zero point per group
 * @param {number} bits
 * @param {number} group weights sharing one scale
 * @param {boolean} search pick each group's range to minimise its error rather
 *   than taking it from the extremes
 */
export function quantiseInPlace(values, { bits, group, mode = "asym", search = false }) {
  // 🔴 SYMMETRIC CODES ARE SIGNED AND ASYMMETRIC ONES ARE NOT. Clamping a
  // symmetric code to [0, levels] zeroes every negative weight - half of them -
  // which does not crash, does not NaN, and folds a protein into a hairball.
  // It showed up as the trunk disagreeing with AF3 by relRMS 20.
  const levels = mode === "sym" ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
  const lowest = mode === "sym" ? -levels - 1 : 0;
  const clips = search ? [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0] : [1.0];
  for (let start = 0; start < values.length; start += group) {
    const end = Math.min(start + group, values.length);
    let low = Infinity;
    let high = -Infinity;
    for (let i = start; i < end; i += 1) {
      if (values[i] < low) low = values[i];
      if (values[i] > high) high = values[i];
    }
    let bestError = Infinity;
    let bestScale = 0;
    let bestZero = 0;
    for (const clip of clips) {
      let scale;
      let zero;
      if (mode === "sym") {
        const extent = Math.max(Math.abs(low), Math.abs(high)) * clip;
        scale = extent / levels;
        zero = 0;
      } else {
        const mid = (high + low) / 2;
        const half = ((high - low) / 2) * clip;
        scale = (2 * half) / levels;
        zero = mid - half;
      }
      if (scale === 0) { scale = 1; }
      let error = 0;
      for (let i = start; i < end; i += 1) {
        const q = Math.max(lowest, Math.min(levels, Math.round((values[i] - zero) / scale)));
        const d = q * scale + zero - values[i];
        error += d * d;
      }
      if (error < bestError) { bestError = error; bestScale = scale; bestZero = zero; }
    }
    for (let i = start; i < end; i += 1) {
      const q = Math.max(lowest, Math.min(levels, Math.round((values[i] - bestZero) / bestScale)));
      values[i] = q * bestScale + bestZero;
    }
  }
  return values;
}

/**
 * 🔴 NORMS, OFFSETS AND BIASES STAY FLOAT32. They are a fraction of a percent
 * of the parameters and the worst possible thing to group-quantise: a 128-wide
 * LayerNorm scale is two groups, so two scales carry the whole tensor, and that
 * tensor's whole job is to set the scale of what follows.
 */
const KEEP_FLOAT32 = /\/(scale|offset|bias)$|_bias$|_weight$|\/output_b$/;

export async function openAf3Store(manifest = MANIFEST, quant = null, onProgress = undefined) {
  const store = await HttpTensorStore.open(manifest, onProgress);
  if (quant === null) return store;
  const cache = new Map();
  const original = store.tensor.bind(store);
  store.tensor = async (name) => {
    if (cache.has(name)) return cache.get(name);
    const values = await original(name);
    const output = KEEP_FLOAT32.test(name)
      ? values
      : quantiseInPlace(Float32Array.from(values), quant);
    cache.set(name, output);
    return output;
  };
  return store;
}

/** One tensor, whole. */
export function tensor(store, name) {
  return store.tensor(name);
}

/** One block's slice of a tensor stacked over blocks. */
export async function layer(store, name, index) {
  const whole = await store.tensor(name);
  const shape = store.shape(name);
  const stride = whole.length / shape[0];
  if (index >= shape[0]) throw new Error(`${name} has ${shape[0]} blocks; asked for ${index}`);
  return whole.subarray(index * stride, (index + 1) * stride);
}

/** The five pair-track modules, which every stack shares. */
async function pairTrack(store, root, index, gridHeads, gridDimension) {
  const at = (leaf) => layer(store, `${root}/${leaf}`, index);
  const triangle = async (direction) => ({
    leftNormInputScale: await at(`triangle_multiplication_${direction}/left_norm_input/scale`),
    leftNormInputOffset: await at(`triangle_multiplication_${direction}/left_norm_input/offset`),
    projection: await at(`triangle_multiplication_${direction}/projection/weights`),
    gate: await at(`triangle_multiplication_${direction}/gate/weights`),
    centerNormScale: await at(`triangle_multiplication_${direction}/center_norm/scale`),
    centerNormOffset: await at(`triangle_multiplication_${direction}/center_norm/offset`),
    outputProjection: await at(`triangle_multiplication_${direction}/output_projection/weights`),
    gatingLinear: await at(`triangle_multiplication_${direction}/gating_linear/weights`),
  });
  const grid = async (which) => ({
    heads: gridHeads, dimension: gridDimension,
    actNormScale: await at(`pair_attention${which}/act_norm/scale`),
    actNormOffset: await at(`pair_attention${which}/act_norm/offset`),
    pairBiasProjection: await at(`pair_attention${which}/pair_bias_projection/weights`),
    qProjection: await at(`pair_attention${which}/q_projection/weights`),
    kProjection: await at(`pair_attention${which}/k_projection/weights`),
    vProjection: await at(`pair_attention${which}/v_projection/weights`),
    gatingQuery: await at(`pair_attention${which}/gating_query/weights`),
    outputProjection: await at(`pair_attention${which}/output_projection/weights`),
  });
  return {
    triangleMultiplicationOutgoing: await triangle("outgoing"),
    triangleMultiplicationIncoming: await triangle("incoming"),
    pairAttention1: await grid(1),
    pairAttention2: await grid(2),
    pairTransition: {
      inputLayerNormScale: await at("pair_transition/input_layer_norm/scale"),
      inputLayerNormOffset: await at("pair_transition/input_layer_norm/offset"),
      transition1: await at("pair_transition/transition1/weights"),
      transition2: await at("pair_transition/transition2/weights"),
    },
  };
}

export async function embedderWeights(store) {
  const T = (name) => store.tensor(`${EVO}/${name}`);
  return {
    pairChannels: 128, singleChannels: 384, msaChannels: 64,
    targetFeatWidth: 447, relativeWidth: 139,
    leftSingle: await T("left_single/weights"),
    rightSingle: await T("right_single/weights"),
    prevEmbeddingNormScale: await T("prev_embedding_layer_norm/scale"),
    prevEmbeddingNormOffset: await T("prev_embedding_layer_norm/offset"),
    prevEmbedding: await T("prev_embedding/weights"),
    positionActivations: await T("~_relative_encoding/position_activations/weights"),
    // 🔴 [1, 128] - ONE input feature, the contact matrix. It was in the
    // shipped bundle and read by nothing, so every fold downloaded it and
    // multiplied it by no ligand bonds at all. See embedder-webgpu.js.
    bondEmbedding: await T("bond_embedding/weights"),
    msaActivations: await T("msa_activations/weights"),
    extraMsaTargetFeat: await T("extra_msa_target_feat/weights"),
    singleActivations: await T("single_activations/weights"),
    prevSingleEmbeddingNormScale: await T("prev_single_embedding_layer_norm/scale"),
    prevSingleEmbeddingNormOffset: await T("prev_single_embedding_layer_norm/offset"),
    prevSingleEmbedding: await T("prev_single_embedding/weights"),
  };
}

export async function templateWeights(store) {
  const T = (name) => store.tensor(name);
  // 🔴 THE TEMPLATE STACK'S GRID ATTENTION IS 4 HEADS OF 16, not the trunk's
  // 4 of 32: 64 channels rather than 128.
  const blocks = [await pairTrack(store, TEMPLATE_STACK, 0, 4, 16),
                  await pairTrack(store, TEMPLATE_STACK, 1, 4, 16)];
  return {
    queryChannels: 128, blocks,
    queryEmbeddingNormScale: await T(`${TEMPLATE_SINGLE}/query_embedding_norm/scale`),
    queryEmbeddingNormOffset: await T(`${TEMPLATE_SINGLE}/query_embedding_norm/offset`),
    templatePairEmbedding8: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_8/weights`),
    templatePairEmbedding2: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_2/weights`),
    templatePairEmbedding3: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_3/weights`),
    outputLayerNormScale: await T(`${TEMPLATE_SINGLE}/output_layer_norm/scale`),
    outputLayerNormOffset: await T(`${TEMPLATE_SINGLE}/output_layer_norm/offset`),
    outputLinear: await T(`${TEMPLATE}/output_linear/weights`),
  };
}

export async function msaBlockWeights(store, index) {
  const at = (leaf) => layer(store, `${MSA_STACK}/${leaf}`, index);
  return {
    pairChannels: 128, msaChannels: 64,
    ...(await pairTrack(store, MSA_STACK, index, 4, 32)),
    outerProductMean: {
      outerChannels: 32,
      layerNormInputScale: await at("outer_product_mean/layer_norm_input/scale"),
      layerNormInputOffset: await at("outer_product_mean/layer_norm_input/offset"),
      leftProjection: await at("outer_product_mean/left_projection/weights"),
      rightProjection: await at("outer_product_mean/right_projection/weights"),
      outputW: await at("outer_product_mean/output_w"),
      outputB: await at("outer_product_mean/output_b"),
    },
    msaAttention1: {
      heads: 8, dimension: 8,
      actNormScale: await at("msa_attention1/act_norm/scale"),
      actNormOffset: await at("msa_attention1/act_norm/offset"),
      pairNormScale: await at("msa_attention1/pair_norm/scale"),
      pairNormOffset: await at("msa_attention1/pair_norm/offset"),
      pairLogits: await at("msa_attention1/pair_logits/weights"),
      vProjection: await at("msa_attention1/v_projection/weights"),
      gatingQuery: await at("msa_attention1/gating_query/weights"),
      outputProjection: await at("msa_attention1/output_projection/weights"),
    },
    msaTransition: {
      inputLayerNormScale: await at("msa_transition/input_layer_norm/scale"),
      inputLayerNormOffset: await at("msa_transition/input_layer_norm/offset"),
      transition1: await at("msa_transition/transition1/weights"),
      transition2: await at("msa_transition/transition2/weights"),
    },
  };
}

export async function pairformerBlockWeights(store, index) {
  const at = (leaf) => layer(store, `${PAIRFORMER}/${leaf}`, index);
  return {
    pairChannels: 128, singleChannels: 384,
    ...(await pairTrack(store, PAIRFORMER, index, 4, 32)),
    singlePairLogitsNormScale: await at("single_pair_logits_norm/scale"),
    singlePairLogitsNormOffset: await at("single_pair_logits_norm/offset"),
    singlePairLogitsProjection: await at("single_pair_logits_projection/weights"),
    singleAttention: {
      heads: 16, dimension: 24,
      layerNormScale: await at("single_attention_layer_norm/scale"),
      layerNormOffset: await at("single_attention_layer_norm/offset"),
      qProjection: await at("single_attention_q_projection/weights"),
      qBias: await at("single_attention_q_projection/bias"),
      kProjection: await at("single_attention_k_projection/weights"),
      vProjection: await at("single_attention_v_projection/weights"),
      gatingQuery: await at("single_attention_gating_query/weights"),
      outputProjection: await at("single_attention_transition2/weights"),
    },
    singleTransition: {
      inputLayerNormScale: await at("single_transition/input_layer_norm/scale"),
      inputLayerNormOffset: await at("single_transition/input_layer_norm/offset"),
      transition1: await at("single_transition/transition1/weights"),
      transition2: await at("single_transition/transition2/weights"),
    },
  };
}

export async function distogramWeights(store) {
  return { halfLogits: await store.tensor("diffuser/distogram_head/half_logits/weights") };
}

export async function confidenceWeights(store) {
  const T = (name) => store.tensor(`${CONFIDENCE}/${name}`);
  const blocks = [];
  for (let index = 0; index < 4; index += 1) {
    const at = (leaf) => layer(store, `${CONFIDENCE_STACK}/${leaf}`, index);
    blocks.push({
      pairChannels: 128, singleChannels: 384,
      ...(await pairTrack(store, CONFIDENCE_STACK, index, 4, 32)),
      singlePairLogitsNormScale: await at("single_pair_logits_norm/scale"),
      singlePairLogitsNormOffset: await at("single_pair_logits_norm/offset"),
      singlePairLogitsProjection: await at("single_pair_logits_projection/weights"),
      singleAttention: {
        heads: 16, dimension: 24,
        layerNormScale: await at("single_attention_layer_norm/scale"),
        layerNormOffset: await at("single_attention_layer_norm/offset"),
        qProjection: await at("single_attention_q_projection/weights"),
        qBias: await at("single_attention_q_projection/bias"),
        kProjection: await at("single_attention_k_projection/weights"),
        vProjection: await at("single_attention_v_projection/weights"),
        gatingQuery: await at("single_attention_gating_query/weights"),
        outputProjection: await at("single_attention_transition2/weights"),
      },
      singleTransition: {
        inputLayerNormScale: await at("single_transition/input_layer_norm/scale"),
        inputLayerNormOffset: await at("single_transition/input_layer_norm/offset"),
        transition1: await at("single_transition/transition1/weights"),
        transition2: await at("single_transition/transition2/weights"),
      },
    });
  }
  return {
    pairChannels: 128, singleChannels: 384, targetFeatWidth: 447, blocks,
    leftTargetFeatProject: await T("~_embed_features/left_target_feat_project/weights"),
    rightTargetFeatProject: await T("~_embed_features/right_target_feat_project/weights"),
    distogramFeatProject: await T("~_embed_features/distogram_feat_project/weights"),
    logitsLnScale: await T("logits_ln/scale"),
    logitsLnOffset: await T("logits_ln/offset"),
    leftHalfDistanceLogits: await T("left_half_distance_logits/weights"),
    paeLogitsLnScale: await T("pae_logits_ln/scale"),
    paeLogitsLnOffset: await T("pae_logits_ln/offset"),
    paeLogits: await T("pae_logits/weights"),
    plddtLnScale: await T("plddt_logits_ln/scale"),
    plddtLnOffset: await T("plddt_logits_ln/offset"),
    plddtLogits: await T("plddt_logits/weights"),
    resolvedLnScale: await T("experimentally_resolved_ln/scale"),
    resolvedLnOffset: await T("experimentally_resolved_ln/offset"),
    experimentallyResolvedLogits: await T("experimentally_resolved_logits/weights"),
  };
}

/** Everything the trunk needs. `pairformerBlocks` is capped for quick checks. */
export async function trunkWeights(store, pairformerBlocks = 48, msaBlocks = 4) {
  const msa = [];
  for (let index = 0; index < msaBlocks; index += 1) msa.push(await msaBlockWeights(store, index));
  const pairformer = [];
  for (let index = 0; index < pairformerBlocks; index += 1) {
    pairformer.push(await pairformerBlockWeights(store, index));
  }
  return {
    embedder: await embedderWeights(store),
    template: await templateWeights(store),
    msaBlocks: msa,
    pairformerBlocks: pairformer,
    distogram: await distogramWeights(store),
  };
}
