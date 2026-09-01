/**
 * AF3's diffusion sampler on the GPU: does it actually fold anything?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-sampler-gpu.js --steps=20
 *
 * This is not a differential test - the trajectory depends on a PRNG draw that
 * neither AF3 nor this shares, so there is nothing to compare against. What it
 * checks is the properties a correct sampler must have whatever the draw:
 *
 *   - the structure CONTRACTS. It starts as a gaussian cloud 2560 A across and
 *     must end at protein scale, tens of angstroms.
 *   - it converges. The last steps must move the atoms far less than the first.
 *   - padded slots stay exactly zero throughout.
 *
 * A denoiser wired up wrongly - the blend inverted, the scalings swapped, the
 * conditioning crossed - still produces coordinates, and still produces them
 * step after step. It just does not do this.
 *
 * (What the head computes IS differentially tested, in
 * tools/gpu/check-af3-diffusion-head.js. This is the loop around it.)
 *
 * 🔴 THE EXTENT IS NOT A QUALITY MEASURE HERE. The conditioning fed in is
 * synthetic - random trunk representations, random target features - so the
 * model has no real protein to fold toward and the ~110 A it settles at means
 * nothing about accuracy. What it does mean is that the sampler contracted a
 * 2560 A cloud by a factor of twenty and then held still, which is the loop
 * working. Folding a real sequence needs the featuriser, which is not written.
 */
/*
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-diffusion-head.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-diffusion-head.js --noise=160
 *
 * One denoising step, end to end: conditioning, atom encoder, the 24-block
 * token transformer, atom decoder, and the EDM blend. Every stage has its own
 * checker; what this adds is that they compose - and in particular that the
 * ATOMS are conditioned by the trunk's single while the TRANSFORMER is
 * conditioned by the conditioning module's, which are different tensors of
 * different widths that AF3 uses in different places.
 *
 * 🔴 CHECK MORE THAN ONE NOISE LEVEL. The EDM blend weights depend on it, and
 * at sigma = 16 (SIGMA_DATA) skip and out are both 1/2 - so a swapped pair of
 * scalings is invisible there and nowhere else.
 */
import { atomCrossAttentionEncoder } from "../../src/af3/atom-encoder-reference.js";
import { sampleOnGpu } from "../../src/af3/diffusion-sampler-webgpu.js";
import { atomCrossAttentionEncoder as encodeCpu } from "../../src/af3/atom-encoder-reference.js";
import { openAf3Store } from "../../src/af3/weights.js";

const DUMP = "/af3-oracle-atom-f32.json";
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

  // The conditioning module's own weights.
  const transition = async (prefix) => ({
    ffwLayerNormScale: await T(`${HEAD}/${prefix}ffw_layer_norm/scale`),
    ffwLayerNormOffset: await T(`${HEAD}/${prefix}ffw_layer_norm/offset`),
    ffwTransition1: await T(`${HEAD}/${prefix}ffw_transition1/weights`),
    ffwTransition2: await T(`${HEAD}/${prefix}ffw_transition2/weights`),
  });
  const conditioning = {
    pairChannels: 128, seqChannels: 384, targetFeatWidth: 447, relativeWidth: 139,
    pairCondInitialNormScale: await T(`${HEAD}/pair_cond_initial_norm/scale`),
    pairCondInitialProjection: await T(`${HEAD}/pair_cond_initial_projection/weights`),
    pairTransitions: [await transition("pair_transition_0"),
                      await transition("pair_transition_1")],
    singleCondInitialNormScale: await T(`${HEAD}/single_cond_initial_norm/scale`),
    singleCondInitialProjection: await T(`${HEAD}/single_cond_initial_projection/weights`),
    singleTransitions: [await transition("single_transition_0"),
                        await transition("single_transition_1")],
    fourierWeight: await T(`${HEAD}/fourier_embedding_weight`),
    fourierBias: await T(`${HEAD}/fourier_embedding_bias`),
    noiseEmbeddingInitialNormScale: await T(`${HEAD}/noise_embedding_initial_norm/scale`),
    noiseEmbeddingInitialProjection:
      await T(`${HEAD}/noise_embedding_initial_projection/weights`),
  };

  // The token transformer, six super-blocks of four.
  const TX = `${HEAD}/transformer`;
  const TX_STACK = `${TX}/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformer`;
  const txSlice = async (leaf, superBlock, inner) => {
    const name = `${TX_STACK}${leaf}`;
    const whole = await store.tensor(name);
    const shape = store.shape(name);
    const stride = whole.length / (shape[0] * shape[1]);
    const index = superBlock * shape[1] + inner;
    return whole.subarray(index * stride, (index + 1) * stride);
  };
  const txBlock = async (superBlock, inner) => {
    const at = (leaf) => txSlice(leaf, superBlock, inner);
    return {
      SingleCondLayerNormScale: await at("single_cond_layer_norm/scale"),
      SingleCondScaleWeights: await at("single_cond_scale/weights"),
      SingleCondScaleBias: await at("single_cond_scale/bias"),
      SingleCondBias: await at("single_cond_bias/weights"),
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
  const supers = Number(option(args, "supers", "2"));
  const projectionName = `${TX}/__layer_stack_with_per_layer/pair_logits_projection/weights`;
  const projections = await store.tensor(projectionName);
  const projectionStride = projections.length / store.shape(projectionName)[0];
  const superBlocks = [];
  for (let s = 0; s < supers; s += 1) {
    const blocks = [];
    for (let inner = 0; inner < 4; inner += 1) blocks.push(await txBlock(s, inner));
    superBlocks.push({
      pairLogitsProjection: projections.subarray(s * projectionStride, (s + 1) * projectionStride),
      blocks,
    });
  }
  const transformer = {
    channels: 768, condChannels: 384, pairChannels: 128, heads: 16, dimension: 48,
    transitionFactor: 2, blocksPerSuperBlock: 4,
    pairInputLayerNormScale: await T(`${TX}/pair_input_layer_norm/scale`),
    superBlocks,
  };

  const headWeights = {
    seqChannels: 384, perTokenChannels: 768,
    singleCondEmbeddingNormScale: await T(`${HEAD}/single_cond_embedding_norm/scale`),
    singleCondEmbeddingProjection: await T(`${HEAD}/single_cond_embedding_projection/weights`),
    outputNormScale: await T(`${HEAD}/output_norm/scale`),
    conditioning, transformer, encoder: weights, decoder: decoderWeights,
  };

  const noiseLevel = Number(option(args, "noise", "56.0"));
  const headInput = {
    ...input, noiseLevel,
    positionsNoisy: deterministic(tokens * dense * 3, 1357),
    seqMask: Float32Array.from(raw("seq_mask"), (v) => Number(v)),
    trunkSingle: input.trunkSingleCond,
    trunkPair: input.trunkPairCond,
    targetFeat: deterministic(tokens * 447, 2468),
    features: (() => {
      const residueIndex = new Int32Array(tokens);
      const asymId = new Int32Array(tokens);
      for (let t = 0; t < tokens; t += 1) { residueIndex[t] = t; asymId[t] = t < tokens / 2 ? 0 : 1; }
      return { residueIndex, tokenIndex: residueIndex, asymId, entityId: asymId,
               symId: new Int32Array(tokens) };
    })(),
  };

  const steps = Number(option(args, "steps", "12"));
  let state = 9;
  const uniform = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  const normal = () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());

  const mask = Float32Array.from(raw("pred_dense_atom_mask"), (v) => Number(v));
  const extent = (positions) => {
    let low = [Infinity, Infinity, Infinity];
    let high = [-Infinity, -Infinity, -Infinity];
    for (let atom = 0; atom < tokens * dense; atom += 1) {
      if (mask[atom] === 0) continue;
      for (let axis = 0; axis < 3; axis += 1) {
        low[axis] = Math.min(low[axis], positions[atom * 3 + axis]);
        high[axis] = Math.max(high[axis], positions[atom * 3 + axis]);
      }
    }
    return Math.max(high[0] - low[0], high[1] - low[1], high[2] - low[2]);
  };
  // 🔴 A FRAME-INVARIANT METRIC, because randomAugmentation applies a fresh
  // random rotation and translation at the top of EVERY step. Comparing
  // coordinates between consecutive frames measures that tumbling, not the
  // structure: it reported the sampler "diverging" from 21 A to 62 A when
  // nothing about the shape was diverging at all. Pairwise distances do not
  // care which way the molecule is facing.
  const real = [];
  for (let atom = 0; atom < tokens * dense; atom += 1) if (mask[atom] !== 0) real.push(atom);
  const distances = (positions) => {
    const output = new Float64Array(real.length * real.length);
    for (let a = 0; a < real.length; a += 1) {
      for (let b = 0; b < real.length; b += 1) {
        let squared = 0;
        for (let axis = 0; axis < 3; axis += 1) {
          const d = positions[real[a] * 3 + axis] - positions[real[b] * 3 + axis];
          squared += d * d;
        }
        output[a * real.length + b] = Math.sqrt(squared);
      }
    }
    return output;
  };
  const moved = (a, b) => {
    const da = distances(a);
    const db = distances(b);
    let total = 0;
    for (let index = 0; index < da.length; index += 1) {
      const d = da[index] - db[index];
      total += d * d;
    }
    return Math.sqrt(total / da.length);
  };

  const frames = [];
  let previousDenoised = null;
  let firstMove = 0;
  let lastMove = 0;
  const finalPositions = await sampleOnGpu(device, headInput, headWeights, {
    steps, normal,
    onStep: ({ step, noiseLevel, denoised }) => {
      if (previousDenoised !== null) {
        const distance = moved(denoised, previousDenoised);
        if (step === 2) firstMove = distance;
        lastMove = distance;
      }
      previousDenoised = denoised;
      frames.push({ step, noiseLevel, extent: extent(denoised) });
      if (step === 1 || step === steps || step % Math.ceil(steps / 4) === 0) {
        console.log(`  step ${String(step).padStart(3)}  sigma `
          + `${noiseLevel.toFixed(2).padStart(9)}   extent ${extent(denoised).toFixed(1)} A`);
      }
    },
  });

  const finalExtent = extent(finalPositions);
  console.log(`final extent ${finalExtent.toFixed(1)} A over ${steps} steps`);
  console.log(`the guess's distance matrix moved ${firstMove.toFixed(2)} A early and `
    + `${lastMove.toFixed(2)} A at the end`);

  // 🔴 PADDED SLOTS MUST STAY EXACTLY ZERO. The head masks its output, so a
  // nonzero padded atom means the mask is not reaching the blend.
  for (let atom = 0; atom < tokens * dense; atom += 1) {
    if (mask[atom] !== 0) continue;
    for (let axis = 0; axis < 3; axis += 1) {
      if (finalPositions[atom * 3 + axis] !== 0) {
        throw new Error(`padded atom ${atom} is at ${finalPositions[atom * 3 + axis]}`);
      }
    }
  }
  if (!(finalExtent > 1 && finalExtent < 500)) {
    throw new Error(`final extent ${finalExtent.toFixed(1)} A is not protein scale`);
  }
  if (!(lastMove < firstMove)) {
    throw new Error(`the sampler did not converge: ${lastMove} at the end against ${firstMove}`);
  }
  return { tokens, steps, finalExtent, firstMove, lastMove };
}
