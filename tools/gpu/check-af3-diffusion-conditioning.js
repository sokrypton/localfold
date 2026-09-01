/**
 * AF3's diffusion conditioning: GPU against src/af3/diffusion-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-diffusion-conditioning.js
 *
 * 🔴 SEVERAL CHAINS, AND A NOISE LEVEL FROM THE MIDDLE OF THE SCHEDULE. The
 * relative encoding's inter-chain branches are unreachable on one chain, and a
 * noise level of 1.0 makes log(sigma/16) land near a value where a missing
 * SIGMA_DATA division is nearly invisible.
 */
import { diffusionConditioning } from "../../src/af3/diffusion-reference.js";
import { Af3DiffusionConditioningGpu } from "../../src/af3/diffusion-conditioning-webgpu.js";
import { relativeEncoding } from "../../src/af3/embedder-reference.js";
import { layerNormSlow } from "../../src/af3/atom-encoder-reference.js";
import { linear } from "../../src/af3/pairformer-reference.js";
import { openAf3Store } from "../../src/af3/weights.js";

const HEAD = "diffuser/~/diffusion_head";
const PAIR_CHANNELS = 128;
const SEQ_CHANNELS = 384;
const TARGET_WIDTH = 447;

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
  const tokens = Number(option(args, "n", "24"));
  const chains = Number(option(args, "chains", "3"));
  const noiseLevel = Number(option(args, "noise", "56.0"));
  const store = await openAf3Store();
  const T = (name) => store.tensor(`${HEAD}/${name}`);

  const transition = async (prefix) => ({
    ffwLayerNormScale: await T(`${prefix}ffw_layer_norm/scale`),
    ffwLayerNormOffset: await T(`${prefix}ffw_layer_norm/offset`),
    ffwTransition1: await T(`${prefix}ffw_transition1/weights`),
    ffwTransition2: await T(`${prefix}ffw_transition2/weights`),
  });

  const weights = {
    pairChannels: PAIR_CHANNELS, seqChannels: SEQ_CHANNELS,
    targetFeatWidth: TARGET_WIDTH, relativeWidth: 139,
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
  };

  const perChain = Math.ceil(tokens / chains);
  const residueIndex = new Int32Array(tokens);
  const asymId = new Int32Array(tokens);
  const entityId = new Int32Array(tokens);
  const symId = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) {
    const chain = Math.floor(t / perChain);
    asymId[t] = chain;
    entityId[t] = chain === 1 ? 0 : chain;
    symId[t] = chain === 1 ? 1 : 0;
    residueIndex[t] = t - chain * perChain;
  }

  const input = {
    tokens, noiseLevel,
    trunkPair: deterministic(tokens * tokens * PAIR_CHANNELS, 71 + tokens),
    trunkSingle: deterministic(tokens * SEQ_CHANNELS, 72 + tokens),
    targetFeat: deterministic(tokens * TARGET_WIDTH, 73 + tokens),
    features: { residueIndex, tokenIndex: residueIndex, asymId, entityId, symId },
  };

  // The pair's initial projection on its own. The reference runs a fixed two
  // transitions, so this rebuilds just the first step from its own pieces
  // rather than trying to switch them off.
  {
    const relative = relativeEncoding(tokens, input.features);
    const width = PAIR_CHANNELS + 139;
    const pairs = tokens * tokens;
    const features2d = new Float32Array(pairs * width);
    for (let index = 0; index < pairs; index += 1) {
      for (let c = 0; c < PAIR_CHANNELS; c += 1) {
        features2d[index * width + c] = input.trunkPair[index * PAIR_CHANNELS + c];
      }
      for (let c = 0; c < 139; c += 1) {
        features2d[index * width + PAIR_CHANNELS + c] = relative[index * 139 + c];
      }
    }
    const reference = linear(
      layerNormSlow(features2d, pairs, width, weights.pairCondInitialNormScale, null),
      pairs, width, PAIR_CHANNELS, weights.pairCondInitialProjection);
    const gpuBare = await new Af3DiffusionConditioningGpu(device)
      .run(input, weights, { transitions: 0 });
    console.log(`  initial pair   ${relativeRms(gpuBare.pair, reference).toExponential(2)}`);
  }

  const expected = diffusionConditioning(input, weights);
  const gpu = await new Af3DiffusionConditioningGpu(device).run(input, weights);

  const results = {
    pair: relativeRms(gpu.pair, expected.pair),
    single: relativeRms(gpu.single, expected.single),
  };
  for (const [name, value] of Object.entries(results)) {
    console.log(`${name}\trelRMS ${value.toExponential(2)}`);
  }
  console.log(`noise level ${noiseLevel}\t${gpu.elapsedMilliseconds.toFixed(1)} ms`
    + `\t${(gpu.memory.peakBytes / 2 ** 20).toFixed(1)} MiB`);

  const bound = 1e-5;
  const worst = Math.max(...Object.values(results));
  if (worst > bound) throw new Error(`relRMS ${worst.toExponential(2)} exceeds ${bound}`);
  return { tokens, chains, noiseLevel, ...results };
}
