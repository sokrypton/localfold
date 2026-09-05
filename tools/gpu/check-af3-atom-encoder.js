/**
 * AF3's atom cross-attention encoder: GPU against src/af3/atom-encoder-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-atom-encoder.js
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
 *
 * 🔴 AND THE DIALECT IS AN AXIS, BECAUSE `maskPaddedKeys` IS IMPLEMENTED TWICE
 * AND DIFFERENTLY. The reference ANDs the key mask into the offset validity;
 * the GPU writes -1 into a padded key's reference space so the equality test
 * fails on its own, which needs no ninth storage binding and no shader change.
 * Those are the same model only if the sentinel can never collide with a real
 * reference space, so both arms run and both are held to the envelope. The
 * openbind0 arm also has to DIFFER from the stock one on this batch or it is
 * checking nothing - two thirds of the key slots here are padding, so it does.
 *
 * 🔴 AND IT DIFFERS IN ONE OUTPUT, NOT THREE, WHICH IS WORTH KNOWING. Measured
 * here, openbind against alphafold3:
 *
 *     pairCond         7.77e-2
 *     tokenAct         2.96e-6   (rounding envelope 1.38e-6)
 *     skipConnection   3.72e-6   (envelope 2.93e-6)
 *
 * The atom pair conditioning moves enormously, because that is where a padded
 * key's offset term lived. Almost none of it reaches the encoder's OUTPUT,
 * because the attention masks those same keys anyway - what leaks through is
 * the mask bias being a large finite negative rather than -infinity, so a
 * padded key keeps a softmax weight of about 1e-6 rather than zero. So the
 * control is "some output moved well past its envelope", not "every one did":
 * demanding all three would fail on a correct implementation.
 */
import { atomCrossAttentionEncoder } from "../../src/af3/atom-encoder-reference.js";
import { Af3AtomEncoderGpu } from "../../src/af3/atom-encoder-webgpu.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { ALPHAFOLD3, OPENBIND0 } from "../../src/af3/dialect.js";

const DUMP = "/oracle-dumps/af3-oracle-atom-f32.json";
const HEAD = "diffuser/~/diffusion_head";
const ENCODER = `${HEAD}/diffusion_atom_transformer_encoder`;
const STACK = `${ENCODER}/__layer_stack_with_per_layer/diffusion_atom_transformer_encoder`;

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
    // dialect is set per arm below.
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

  const results = {};
  const references = {};
  let failed = 0;
  for (const [label, dialect] of [["alphafold3", ALPHAFOLD3], ["openbind0", OPENBIND0]]) {
    const arm = { ...input, dialect };
    const expected = atomCrossAttentionEncoder(arm, weights);
    const gpu = await new Af3AtomEncoderGpu(device).run(arm, weights);
    references[label] = expected;

    // What rounding alone produces through three cross-attention blocks.
    const perturbed = { ...arm, conditioning: Float32Array.from(arm.conditioning) };
    for (let index = 0; index < perturbed.conditioning.length; index += 1) {
      perturbed.conditioning[index] *= 1 + 1e-7;
    }
    const control = atomCrossAttentionEncoder(perturbed, weights);

    results[label] = {};
    for (const name of ["tokenAct", "skipConnection", "pairCond"]) {
      const value = relativeRms(gpu[name], expected[name]);
      const envelope = relativeRms(control[name], expected[name]);
      const bound = Math.max(1e-5, envelope * 10);
      const ok = value <= bound;
      if (!ok) failed += 1;
      results[label][name] = { relRms: value, envelope };
      console.log(`${label}\t${name}\trelRMS ${value.toExponential(2)}`
        + `\t(envelope ${envelope.toExponential(2)},`
        + ` ${(value / Math.max(envelope, 1e-30)).toFixed(1)}x)\t${ok ? "" : "FAIL"}`);
    }
    console.log(`${label}\t${gpu.elapsedMilliseconds.toFixed(1)} ms`
      + `\t${(gpu.memory.peakBytes / 2 ** 20).toFixed(1)} MiB`);
  }

  // 🔴 THE DISCRIMINATING CONTROL. Both arms passing says the GPU matches the
  // reference; it does not say the flag reached either of them. A dialect that
  // changed nothing would pass this checker twice and ship a model that is
  // stock AF3 wearing OpenBind's name.
  const separation = {};
  for (const name of ["tokenAct", "skipConnection", "pairCond"]) {
    separation[name] = relativeRms(references.openbind0[name], references.alphafold3[name]);
    console.log(`openbind0 vs alphafold3\t${name}`
      + `\trelRMS ${separation[name].toExponential(2)}`);
  }
  const moved = Math.max(...Object.values(separation));

  if (failed > 0) throw new Error(`${failed} output(s) outside their conditioning envelope`);
  if (moved < 1e-3) {
    throw new Error(`the dialect moved no output by more than ${moved.toExponential(2)}: `
      + "maskPaddedKeys did not reach the encoder, so neither arm was checked "
      + "against anything");
  }
  return { tokens, results, separation };
}
