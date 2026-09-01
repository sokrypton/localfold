/**
 * AF3's embedder: GPU against src/af3/embedder-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-embedder.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-embedder.js --chains=3
 *
 * 🔴 THE FEATURES MUST DESCRIBE MORE THAN ONE CHAIN. The relative encoding's
 * interesting branches - the inter-chain bin, the same-entity flag, the
 * symmetry-copy bin - are all unreachable on a single chain, where every pair is
 * same-chain and same-entity and the encoding collapses to one clamped offset.
 * A single-chain check passes on an implementation that ignores asym_id.
 *
 * 🔴 AND IT MUST CARRY A BOND MATRIX. The reference applies the bond embedding
 * only when one is supplied, so a fixture without one compared zero against
 * zero - and the GPU embedder had neither the weight nor the term for as long
 * as ligands have existed here, while this check passed. A feature that is
 * absent from BOTH sides of a differential test is not tested by it.
 *
 * The template term is compared as zeros on both sides: the GPU embedder does
 * not add it (it reads the pair at this point and is sequenced by the caller),
 * so the reference is given a zero template to match. That is a statement about
 * WHERE the term is added, not that it is zero - AF3's template embedder
 * contributes std 13.1 even with four empty slots.
 */
import { embed } from "../../src/af3/embedder-reference.js";
import { Af3EmbedderGpu } from "../../src/af3/embedder-webgpu.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const EVO = "diffuser/evoformer";
const FEATURE_WIDTH = 447;
const PAIR_CHANNELS = 128;
const SINGLE_CHANNELS = 384;
const MSA_CHANNELS = 64;

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

function generator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}

function deterministic(length, seed) {
  const next = generator(seed);
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) output[index] = next() * 2 - 1;
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
  const tokens = Number(option(args, "tokens", "48"));
  const sequences = Number(option(args, "sequences", "8"));
  const chains = Number(option(args, "chains", "3"));
  const store = await HttpTensorStore.open(MANIFEST);
  const T = async (name) => store.tensor(`${EVO}/${name}`);

  const weights = {
    pairChannels: PAIR_CHANNELS, singleChannels: SINGLE_CHANNELS,
    msaChannels: MSA_CHANNELS, targetFeatWidth: FEATURE_WIDTH,
    relativeWidth: 139,
    leftSingle: await T("left_single/weights"),
    rightSingle: await T("right_single/weights"),
    prevEmbeddingNormScale: await T("prev_embedding_layer_norm/scale"),
    prevEmbeddingNormOffset: await T("prev_embedding_layer_norm/offset"),
    prevEmbedding: await T("prev_embedding/weights"),
    positionActivations: await T("~_relative_encoding/position_activations/weights"),
    msaActivations: await T("msa_activations/weights"),
    extraMsaTargetFeat: await T("extra_msa_target_feat/weights"),
    singleActivations: await T("single_activations/weights"),
    prevSingleEmbeddingNormScale: await T("prev_single_embedding_layer_norm/scale"),
    prevSingleEmbeddingNormOffset: await T("prev_single_embedding_layer_norm/offset"),
    prevSingleEmbedding: await T("prev_single_embedding/weights"),
    bondEmbedding: await T("bond_embedding/weights"),
  };

  // Several chains, two of them the same entity, so every branch is live.
  const perChain = Math.ceil(tokens / chains);
  const residueIndex = new Int32Array(tokens);
  const tokenIndex = new Int32Array(tokens);
  const asymId = new Int32Array(tokens);
  const entityId = new Int32Array(tokens);
  const symId = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) {
    const chain = Math.floor(t / perChain);
    asymId[t] = chain;
    entityId[t] = chain === 1 ? 0 : chain;   // chains 0 and 1 share an entity
    symId[t] = chain === 1 ? 1 : 0;
    residueIndex[t] = t - chain * perChain;
    tokenIndex[t] = residueIndex[t];
  }

  const next = generator(4242);
  const msaRows = new Int32Array(sequences * tokens);
  const deletionMatrix = new Float32Array(sequences * tokens);
  for (let index = 0; index < msaRows.length; index += 1) {
    msaRows[index] = Math.floor(next() * 34) - 1;   // includes out-of-range codes
    deletionMatrix[index] = next() * 6;
  }

  // A SPARSE, ASYMMETRIC CONTACT MATRIX, which is the shape AF3's is: one
  // direction per bond from the CCD table, and [0,0] cleared. Sparse because a
  // dense one would let a wrong index still land on a 1 and agree by luck.
  const bondMatrix = new Float32Array(tokens * tokens);
  for (let t = perChain; t + 1 < tokens; t += 3) {
    bondMatrix[t * tokens + (t + 1)] = 1;
  }
  bondMatrix[0] = 0;

  const input = {
    tokens, sequences,
    targetFeat: deterministic(tokens * FEATURE_WIDTH, 11 + tokens),
    features: { residueIndex, tokenIndex, asymId, entityId, symId },
    msaRows, deletionMatrix, bondMatrix,
    previousPair: deterministic(tokens * tokens * PAIR_CHANNELS, 22 + tokens),
    previousSingle: deterministic(tokens * SINGLE_CHANNELS, 33 + tokens),
  };

  const expected = embed({ ...input,
    templateEmbedding: new Float32Array(tokens * tokens * PAIR_CHANNELS) }, weights);
  const gpu = await new Af3EmbedderGpu(device).run(input, weights);

  const results = {
    pair: relativeRms(gpu.pair, expected.pair),
    msa: relativeRms(gpu.msa, expected.msa),
    single: relativeRms(gpu.single, expected.single),
  };
  for (const [name, value] of Object.entries(results)) {
    console.log(`${name}\trelRMS ${value.toExponential(2)}`);
  }
  console.log(`${gpu.elapsedMilliseconds.toFixed(1)} ms`
    + `\t${(gpu.memory.peakBytes / 2 ** 20).toFixed(1)} MiB peak`);

  const bound = 1e-5;
  const worst = Math.max(...Object.values(results));
  if (worst > bound) throw new Error(`relRMS ${worst.toExponential(2)} exceeds ${bound}`);
  return { tokens, sequences, chains, ...results };
}
