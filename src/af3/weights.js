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
  if (quant === null || quant === undefined) return store;
  // 🔴 SEVEN TOOLS PASSED `{ fetchImplementation: fetch }` HERE. It is not a
  // quantisation spec, so quantiseInPlace ran with an undefined group, walked
  // no groups, and changed nothing - while Float32Array.from copied every
  // tensor in the model. The measured cost on the full manifest was heap 760
  // MiB -> 1340 MiB and a pass over 1.4 GiB of floats, for no effect at all.
  // Naming the mistake is cheaper than measuring it again.
  if (typeof quant !== "object" || !Number.isInteger(quant.bits) || !Number.isInteger(quant.group)) {
    throw new TypeError("openAf3Store's second argument is a quantisation spec"
      + " needing integer bits and group; pass null for the model as exported");
  }
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

/**
 * A block's slice of a stacked tensor, as a THUNK that decodes when it is read.
 *
 * 🔴 THE TRUNK'S WEIGHTS ARE 562 MiB OF FLOAT32 AND THE DEVICE ALREADY HAS THEM.
 * The 48 pairformer blocks are stored as one tensor each with the block as the
 * leading axis, and every block took its slice by decoding the whole tensor -
 * so loading a model put 562 MiB on the heap and left it there, beside the same
 * numbers resident on the GPU. Reading a range instead means one block's
 * weights exist while it is being packed for upload and not after.
 *
 * The thunk carries its tensor name so openFields can bring the shard in before
 * anything reads it; tensorRangeSync is synchronous so this can sit behind a
 * property getter.
 */
export function stacked(store, name, index, dims = 1) {
  const shape = store.shape(name);
  // ...`dims` is how many LEADING axes the stack occupies: the diffusion
  // transformer nests a stack of six inside a stack of four, so its blocks are
  // indexed over both.
  const count = dims === 2 ? shape[0] * shape[1] : shape[0];
  if (index >= count) throw new Error(`${name} has ${count} blocks; asked for ${index}`);
  const stride = shape.slice(dims).reduce((product, value) => product * value, 1);
  const thunk = () => store.tensorRangeSync(name, index * stride, stride);
  thunk.tensorName = name;
  thunk.blockIndex = index;
  thunk.blockDims = dims;
  return thunk;
}

/** Every tensor a descriptor will read, so their shards can be opened at once. */
function fieldNames(fields, into = []) {
  for (const value of Object.values(fields)) {
    if (typeof value === "function") into.push(value.tensorName);
    else if (value !== null && typeof value === "object" && !ArrayBuffer.isView(value)) {
      fieldNames(value, into);
    }
  }
  return into;
}

/**
 * A descriptor's thunks turned into properties that decode on first read.
 *
 * 🔴 MEMOISED, AND CLEARED ONLY ON REQUEST. A getter that decoded on EVERY read
 * would be a trap for the CPU reference paths, which read a weight inside a
 * loop over residues; with the memo they behave exactly as they did when the
 * arrays were eager. What changes is that a consumer which knows it is finished
 * - the GPU block encoder, once the weights are resident on the device - can
 * say so with releaseWeights and get the memory back.
 */
function materialise(fields, memo) {
  const object = {};
  for (const [key, value] of Object.entries(fields)) {
    if (typeof value === "function") {
      Object.defineProperty(object, key, {
        enumerable: true,
        get() {
          let held = memo.get(value);
          if (held === undefined) { held = value(); memo.set(value, held); }
          return held;
        },
      });
    } else if (value !== null && typeof value === "object" && !ArrayBuffer.isView(value)) {
      object[key] = materialise(value, memo);
    } else {
      object[key] = value;
    }
  }
  return object;
}

const RELEASE = Symbol("release decoded weights");

/**
 * Let go of everything a lazily loaded weight object has decoded.
 *
 * Safe at any time and on anything: a released field decodes again when it is
 * next read, and an object that was never lazy ignores this entirely. Call it
 * once a block's weights are on the device.
 */
export function releaseWeights(weights) {
  weights?.[RELEASE]?.();
}

/**
 * Open the shards a descriptor needs, then bind it to properties.
 *
 * Every store can decode a whole tensor; only the HTTP store can decode a range
 * of one. Where it cannot, the descriptor is materialised eagerly from whole
 * tensors, which is what this did before and is still correct - just larger.
 */
export async function bind(store, fields) {
  if (typeof store.tensorRangeSync !== "function" || typeof store.open !== "function") {
    const eager = {};
    for (const [key, value] of Object.entries(fields)) {
      if (typeof value === "function") eager[key] = await sliceOf(store, value);
      else if (value !== null && typeof value === "object" && !ArrayBuffer.isView(value)) {
        eager[key] = await bind(store, value);
      } else eager[key] = value;
    }
    return eager;
  }
  const names = fieldNames(fields);
  await Promise.all([...new Set(names)].map((name) => store.open(name)));
  const memo = new Map();
  const object = materialise(fields, memo);
  Object.defineProperty(object, RELEASE, { value: () => memo.clear() });
  return object;
}

/** The eager form of a thunk, for a store that cannot read a range. */
async function sliceOf(store, thunk) {
  const whole = await store.tensor(thunk.tensorName);
  const shape = store.shape(thunk.tensorName);
  const count = thunk.blockDims === 2 ? shape[0] * shape[1] : shape[0];
  const stride = whole.length / count;
  return whole.subarray(thunk.blockIndex * stride, (thunk.blockIndex + 1) * stride);
}

/** One block's slice of a tensor stacked over blocks. */
export async function layer(store, name, index) {
  const whole = await store.tensor(name);
  const shape = store.shape(name);
  const stride = whole.length / shape[0];
  if (index >= shape[0]) throw new Error(`${name} has ${shape[0]} blocks; asked for ${index}`);
  return whole.subarray(index * stride, (index + 1) * stride);
}

/**
 * The five pair-track modules, which every stack shares - as a DESCRIPTOR whose
 * leaves are thunks, not as decoded arrays. bind() turns one into an object.
 */
function pairTrack(store, root, index, gridHeads, gridDimension) {
  const at = (leaf) => stacked(store, `${root}/${leaf}`, index);
  const triangle = (direction) => ({
    leftNormInputScale: at(`triangle_multiplication_${direction}/left_norm_input/scale`),
    leftNormInputOffset: at(`triangle_multiplication_${direction}/left_norm_input/offset`),
    projection: at(`triangle_multiplication_${direction}/projection/weights`),
    gate: at(`triangle_multiplication_${direction}/gate/weights`),
    centerNormScale: at(`triangle_multiplication_${direction}/center_norm/scale`),
    centerNormOffset: at(`triangle_multiplication_${direction}/center_norm/offset`),
    outputProjection: at(`triangle_multiplication_${direction}/output_projection/weights`),
    gatingLinear: at(`triangle_multiplication_${direction}/gating_linear/weights`),
  });
  const grid = (which) => ({
    heads: gridHeads, dimension: gridDimension,
    actNormScale: at(`pair_attention${which}/act_norm/scale`),
    actNormOffset: at(`pair_attention${which}/act_norm/offset`),
    pairBiasProjection: at(`pair_attention${which}/pair_bias_projection/weights`),
    qProjection: at(`pair_attention${which}/q_projection/weights`),
    kProjection: at(`pair_attention${which}/k_projection/weights`),
    vProjection: at(`pair_attention${which}/v_projection/weights`),
    gatingQuery: at(`pair_attention${which}/gating_query/weights`),
    outputProjection: at(`pair_attention${which}/output_projection/weights`),
  });
  return {
    triangleMultiplicationOutgoing: triangle("outgoing"),
    triangleMultiplicationIncoming: triangle("incoming"),
    pairAttention1: grid(1),
    pairAttention2: grid(2),
    pairTransition: {
      inputLayerNormScale: at("pair_transition/input_layer_norm/scale"),
      inputLayerNormOffset: at("pair_transition/input_layer_norm/offset"),
      transition1: at("pair_transition/transition1/weights"),
      transition2: at("pair_transition/transition2/weights"),
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
  const blocks = [await bind(store, pairTrack(store, TEMPLATE_STACK, 0, 4, 16)),
                  await bind(store, pairTrack(store, TEMPLATE_STACK, 1, 4, 16))];
  return {
    queryChannels: 128, blocks,
    queryEmbeddingNormScale: await T(`${TEMPLATE_SINGLE}/query_embedding_norm/scale`),
    queryEmbeddingNormOffset: await T(`${TEMPLATE_SINGLE}/query_embedding_norm/offset`),
    templatePairEmbedding8: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_8/weights`),
    templatePairEmbedding2: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_2/weights`),
    templatePairEmbedding3: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_3/weights`),
    // 🔴 THE SIX GEOMETRY FEATURES' WEIGHTS WERE IN THE BUNDLE ALL ALONG AND
    // WERE NOT READ. Only 2, 3 and 8 were loaded, because only those three are
    // reachable when every template slot is empty - so a checkpoint that has
    // always carried nine projections was being read for three. Four of these
    // are [64] rather than [39, 64] or [31, 64]: AF3 builds them with
    // `num_input_dims=0`, which makes the input a SCALAR per pair and the
    // weight a per-channel vector.
    templatePairEmbedding0: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_0/weights`),
    templatePairEmbedding1: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_1/weights`),
    templatePairEmbedding4: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_4/weights`),
    templatePairEmbedding5: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_5/weights`),
    templatePairEmbedding6: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_6/weights`),
    templatePairEmbedding7: await T(`${TEMPLATE_SINGLE}/template_pair_embedding_7/weights`),
    outputLayerNormScale: await T(`${TEMPLATE_SINGLE}/output_layer_norm/scale`),
    outputLayerNormOffset: await T(`${TEMPLATE_SINGLE}/output_layer_norm/offset`),
    outputLinear: await T(`${TEMPLATE}/output_linear/weights`),
  };
}

export async function msaBlockWeights(store, index) {
  const at = (leaf) => stacked(store, `${MSA_STACK}/${leaf}`, index);
  return bind(store, {
    pairChannels: 128, msaChannels: 64,
    ...pairTrack(store, MSA_STACK, index, 4, 32),
    outerProductMean: {
      outerChannels: 32,
      layerNormInputScale: at("outer_product_mean/layer_norm_input/scale"),
      layerNormInputOffset: at("outer_product_mean/layer_norm_input/offset"),
      leftProjection: at("outer_product_mean/left_projection/weights"),
      rightProjection: at("outer_product_mean/right_projection/weights"),
      outputW: at("outer_product_mean/output_w"),
      outputB: at("outer_product_mean/output_b"),
    },
    msaAttention1: {
      heads: 8, dimension: 8,
      actNormScale: at("msa_attention1/act_norm/scale"),
      actNormOffset: at("msa_attention1/act_norm/offset"),
      pairNormScale: at("msa_attention1/pair_norm/scale"),
      pairNormOffset: at("msa_attention1/pair_norm/offset"),
      pairLogits: at("msa_attention1/pair_logits/weights"),
      vProjection: at("msa_attention1/v_projection/weights"),
      gatingQuery: at("msa_attention1/gating_query/weights"),
      outputProjection: at("msa_attention1/output_projection/weights"),
    },
    msaTransition: {
      inputLayerNormScale: at("msa_transition/input_layer_norm/scale"),
      inputLayerNormOffset: at("msa_transition/input_layer_norm/offset"),
      transition1: at("msa_transition/transition1/weights"),
      transition2: at("msa_transition/transition2/weights"),
    },
  });
}

export async function pairformerBlockWeights(store, index) {
  const at = (leaf) => stacked(store, `${PAIRFORMER}/${leaf}`, index);
  return bind(store, {
    pairChannels: 128, singleChannels: 384,
    ...pairTrack(store, PAIRFORMER, index, 4, 32),
    singlePairLogitsNormScale: at("single_pair_logits_norm/scale"),
    singlePairLogitsNormOffset: at("single_pair_logits_norm/offset"),
    singlePairLogitsProjection: at("single_pair_logits_projection/weights"),
    singleAttention: {
      heads: 16, dimension: 24,
      layerNormScale: at("single_attention_layer_norm/scale"),
      layerNormOffset: at("single_attention_layer_norm/offset"),
      qProjection: at("single_attention_q_projection/weights"),
      qBias: at("single_attention_q_projection/bias"),
      kProjection: at("single_attention_k_projection/weights"),
      vProjection: at("single_attention_v_projection/weights"),
      gatingQuery: at("single_attention_gating_query/weights"),
      outputProjection: at("single_attention_transition2/weights"),
    },
    singleTransition: {
      inputLayerNormScale: at("single_transition/input_layer_norm/scale"),
      inputLayerNormOffset: at("single_transition/input_layer_norm/offset"),
      transition1: at("single_transition/transition1/weights"),
      transition2: at("single_transition/transition2/weights"),
    },
  });
}

export async function distogramWeights(store) {
  return { halfLogits: await store.tensor("diffuser/distogram_head/half_logits/weights") };
}

export async function confidenceWeights(store) {
  const T = (name) => store.tensor(`${CONFIDENCE}/${name}`);
  const blocks = [];
  for (let index = 0; index < 4; index += 1) {
    const at = (leaf) => stacked(store, `${CONFIDENCE_STACK}/${leaf}`, index);
    blocks.push(await bind(store, {
      pairChannels: 128, singleChannels: 384,
      ...pairTrack(store, CONFIDENCE_STACK, index, 4, 32),
      singlePairLogitsNormScale: at("single_pair_logits_norm/scale"),
      singlePairLogitsNormOffset: at("single_pair_logits_norm/offset"),
      singlePairLogitsProjection: at("single_pair_logits_projection/weights"),
      singleAttention: {
        heads: 16, dimension: 24,
        layerNormScale: at("single_attention_layer_norm/scale"),
        layerNormOffset: at("single_attention_layer_norm/offset"),
        qProjection: at("single_attention_q_projection/weights"),
        qBias: at("single_attention_q_projection/bias"),
        kProjection: at("single_attention_k_projection/weights"),
        vProjection: at("single_attention_v_projection/weights"),
        gatingQuery: at("single_attention_gating_query/weights"),
        outputProjection: at("single_attention_transition2/weights"),
      },
      singleTransition: {
        inputLayerNormScale: at("single_transition/input_layer_norm/scale"),
        inputLayerNormOffset: at("single_transition/input_layer_norm/offset"),
        transition1: at("single_transition/transition1/weights"),
        transition2: at("single_transition/transition2/weights"),
      },
    }));
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
