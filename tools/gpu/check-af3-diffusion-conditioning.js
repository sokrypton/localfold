/**
 * AF3's diffusion conditioning: GPU against src/af3/diffusion-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-diffusion-conditioning.js
 *
 * 🔴 SEVERAL CHAINS, AND A NOISE LEVEL FROM THE MIDDLE OF THE SCHEDULE. The
 * relative encoding's inter-chain branches are unreachable on one chain, and a
 * noise level of 1.0 makes log(sigma/16) land near a value where a missing
 * SIGMA_DATA division is nearly invisible.
 *
 * 🔴 AND THE DIALECT IS AN AXIS, because `padSingleCondUnknownDna` is written
 * twice: as a loop over `singleCondSource` in the reference, and as GENERATED
 * WGSL in the shader's `feature()`. Those are the two halves that can disagree,
 * and nothing else in the repository compares them. The openbind arm splices
 * two extra rows into the LayerNorm scale and the projection exactly where the
 * converter puts them, which is what makes the concatenation 833 wide - so the
 * arm tests the real question: does the shader read the same source column the
 * reference does, past two inserted zeros?
 *
 * 🔴 SPLICED WITH NON-ZERO ROWS ON PURPOSE. OpenFold3 trained those columns, so
 * zeros would leave only the LayerNorm's width to distinguish the arms and a
 * projection indexing bug past column 415 would go unseen.
 *
 * 🔴 AND THE SEPARATION IS SMALL IN ABSOLUTE TERMS - 1.5e-3 measured, which is
 * two columns of 833 plus a 0.24% change in the LayerNorm's width. That is not
 * a weak check: it is about 2400x the error each arm is held to, so a shader
 * that ignored the padding would miss the 1e-5 bound by two orders of
 * magnitude. The control is written as a RATIO against the arms' own error for
 * exactly that reason - an absolute threshold near 1.5e-3 would have almost no
 * headroom, and would fail on a correct implementation at a different token
 * count.
 */
import { diffusionConditioning } from "../../src/af3/diffusion-reference.js";
import { Af3DiffusionConditioningGpu } from "../../src/af3/diffusion-conditioning-webgpu.js";
import { relativeEncoding } from "../../src/af3/embedder-reference.js";
import { layerNormSlow } from "../../src/af3/atom-encoder-reference.js";
import { linear } from "../../src/af3/pairformer-reference.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { ALPHAFOLD3, OPENBIND, singleCondPadding } from "../../src/af3/dialect.js";

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
    tokens, noiseLevel, dialect: ALPHAFOLD3,
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

  const results = {};
  const singles = {};
  for (const [label, dialect] of [["alphafold3", ALPHAFOLD3], ["openbind", OPENBIND]]) {
    const arm = { ...input, dialect };
    const armWeights = padWeights(weights, singleCondPadding(dialect, SEQ_CHANNELS));
    const expected = diffusionConditioning(arm, armWeights);
    const gpu = await new Af3DiffusionConditioningGpu(device).run(arm, armWeights);
    singles[label] = expected.single;
    results[label] = {
      pair: relativeRms(gpu.pair, expected.pair),
      single: relativeRms(gpu.single, expected.single),
    };
    for (const [name, value] of Object.entries(results[label])) {
      console.log(`${label}\t${name}\trelRMS ${value.toExponential(2)}`);
    }
    console.log(`${label}\tnoise level ${noiseLevel}`
      + `\t${gpu.elapsedMilliseconds.toFixed(1)} ms`
      + `\t${(gpu.memory.peakBytes / 2 ** 20).toFixed(1)} MiB`);
  }

  // 🔴 THE DISCRIMINATING CONTROL. Two passing arms say the GPU agrees with the
  // reference; they do not say the padding reached either. The single
  // conditioning is the only output the flag can move - the pair path does not
  // read target_feat - so that is where it is measured.
  const separation = relativeRms(singles.openbind, singles.alphafold3);
  console.log(`openbind vs alphafold3\tsingle\trelRMS ${separation.toExponential(2)}`);

  const bound = 1e-5;
  const worst = Math.max(...Object.values(results).flatMap((r) => Object.values(r)));
  console.log(`separation is ${(separation / Math.max(worst, 1e-30)).toFixed(0)}x `
    + "the error each arm is held to");
  if (worst > bound) throw new Error(`relRMS ${worst.toExponential(2)} exceeds ${bound}`);
  if (separation < worst * 100) {
    throw new Error(`the dialect moved the single conditioning by `
      + `${separation.toExponential(2)}, under 100x the ${worst.toExponential(2)} `
      + "each arm is held to: padSingleCondUnknownDna did not reach it, so "
      + "neither arm was checked against anything");
  }
  return { tokens, chains, noiseLevel, results, separation };
}

/**
 * The single conditioning's LayerNorm scale and projection, with a row spliced
 * in at each padded column - which is what a converted OpenFold3 bundle carries
 * and what makes the concatenation 833 wide rather than 831.
 *
 * An empty padding list returns the weights unchanged, so the stock arm is the
 * bundle exactly as the store served it.
 */
function padWeights(weights, padding) {
  if (padding.length === 0) return weights;
  const width = SEQ_CHANNELS + TARGET_WIDTH;
  const scale = new Float32Array(width + padding.length);
  const projection = new Float32Array((width + padding.length) * SEQ_CHANNELS);
  // 🔴 DETERMINISTIC STAND-INS AT THE SCALE OF THEIR NEIGHBOURS, NOT AT AN
  // ARBITRARY ONE. These stand in for rows OpenFold3 actually trained, and the
  // size of the arms' separation is what says the check discriminates - so a
  // row an order of magnitude too small would make a correct implementation
  // look like an inert flag. Both are drawn against the RMS of the tensor they
  // are spliced into.
  const rms = (values) => {
    let total = 0;
    for (const value of values) total += value * value;
    return Math.sqrt(total / Math.max(values.length, 1));
  };
  const scaleRms = rms(weights.singleCondInitialNormScale);
  const projectionRms = rms(weights.singleCondInitialProjection);
  const extraScale = deterministic(padding.length, 4243).map((v) => v * scaleRms);
  const extraRows = deterministic(padding.length * SEQ_CHANNELS, 4244)
    .map((v) => v * projectionRms);
  let source = 0;
  let extra = 0;
  for (let index = 0; index < width + padding.length; index += 1) {
    if (padding.includes(index)) {
      scale[index] = extraScale[extra];
      for (let c = 0; c < SEQ_CHANNELS; c += 1) {
        projection[index * SEQ_CHANNELS + c] = extraRows[extra * SEQ_CHANNELS + c];
      }
      extra += 1;
      continue;
    }
    scale[index] = weights.singleCondInitialNormScale[source];
    for (let c = 0; c < SEQ_CHANNELS; c += 1) {
      projection[index * SEQ_CHANNELS + c] =
        weights.singleCondInitialProjection[source * SEQ_CHANNELS + c];
    }
    source += 1;
  }
  return { ...weights, singleCondInitialNormScale: scale,
           singleCondInitialProjection: projection };
}
