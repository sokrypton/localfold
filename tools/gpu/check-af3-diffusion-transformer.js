/**
 * AF3's diffusion token transformer: GPU against src/af3/diffusion-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-diffusion-transformer.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-diffusion-transformer.js --supers=6
 *
 * 🔴 THE CHECK RUNS AT LEAST TWO SUPER-BLOCKS BY DEFAULT. The six-by-four
 * nesting means a flat reading of the stack is correct for the first four
 * blocks and wrong for every block after, so a one-super-block check passes on
 * an implementation that ignores the nesting entirely.
 */
import { diffusionTransformer } from "../../src/af3/diffusion-reference.js";
import { Af3DiffusionTransformerGpu } from "../../src/af3/diffusion-transformer-webgpu.js";
import { openAf3Store } from "../../src/af3/weights.js";

const TX = "diffuser/~/diffusion_head/transformer";
const STACK = `${TX}/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformer`;
const CHANNELS = 768;
const COND_CHANNELS = 384;
const PAIR_CHANNELS = 128;
const HEADS = 16;
const DIMENSION = 48;
const PER_SUPER = 4;

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
  const supers = Number(option(args, "supers", "2"));
  const store = await openAf3Store();

  // The stacked tensors are (6, 4, ...): flatten the first two dims.
  const blockSlice = async (leaf, superBlock, inner) => {
    const name = `${STACK}${leaf}`;
    const whole = await store.tensor(name);
    const shape = store.shape(name);
    const count = shape[0] * shape[1];
    const stride = whole.length / count;
    const index = superBlock * shape[1] + inner;
    return whole.subarray(index * stride, (index + 1) * stride);
  };

  const blockWeights = async (superBlock, inner) => {
    const at = (leaf) => blockSlice(leaf, superBlock, inner);
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

  const projectionName = `${TX}/__layer_stack_with_per_layer/pair_logits_projection/weights`;
  const projections = await store.tensor(projectionName);
  const projectionStride = projections.length / store.shape(projectionName)[0];

  // 🔴 THE SUPER-BLOCK IS THE SMALLEST UNIT THAT CAN BE TRIMMED. The reference
  // loops on `blocksPerSuperBlock` rather than the array length, and that same
  // number is the stride of the pair-logits projection - so a partial
  // super-block is not a smaller model, it is a differently-shaped one.
  const superBlocks = [];
  for (let s = 0; s < supers; s += 1) {
    const blocks = [];
    for (let inner = 0; inner < PER_SUPER; inner += 1) blocks.push(await blockWeights(s, inner));
    superBlocks.push({
      pairLogitsProjection: projections.subarray(s * projectionStride, (s + 1) * projectionStride),
      blocks,
    });
  }

  const weights = {
    channels: CHANNELS, condChannels: COND_CHANNELS, pairChannels: PAIR_CHANNELS,
    heads: HEADS, dimension: DIMENSION, transitionFactor: 2,
    blocksPerSuperBlock: PER_SUPER,
    pairInputLayerNormScale: await store.tensor(`${TX}/pair_input_layer_norm/scale`),
    superBlocks,
  };

  const act = deterministic(tokens * CHANNELS, 101 + tokens);
  const cond = deterministic(tokens * COND_CHANNELS, 202 + tokens);
  const pairCond = deterministic(tokens * tokens * PAIR_CHANNELS, 303 + tokens);
  const mask = new Float32Array(tokens);
  for (let t = 0; t < tokens; t += 1) mask[t] = t < Math.ceil(tokens * 0.8) ? 1 : 0;

  const expected = diffusionTransformer(act, cond, pairCond, mask, tokens, weights);
  // The conditioning envelope: what rounding alone produces. See
  // tools/gpu/check-af3-block.js for why a flat tolerance is the wrong
  // instrument once a residual stack starts amplifying.
  const perturbed = Float32Array.from(act);
  for (let index = 0; index < perturbed.length; index += 1) perturbed[index] *= 1 + 1e-7;
  const control = diffusionTransformer(perturbed, cond, pairCond, mask, tokens, weights);
  const envelope = relativeRms(control, expected);
  // 🔴 THE RESIDENT WEIGHTS' STORAGE FORMAT IS AN AXIS. This stack keeps 756
  // MiB of weights on the device - more than everything else a fold holds put
  // together - and holds them in f16 wherever the device allows it. The f32
  // path keeps its bound; the f16 one gets its own, derived from what half
  // precision does to a residual stack this deep.
  const weightPrecision = option(args, "weights",
    device.features.has("shader-f16") ? "f16" : "f32");
  const gpu = await new Af3DiffusionTransformerGpu(device)
    .run(act, cond, pairCond, mask, tokens, { ...weights, weightPrecision });
  const relRms = relativeRms(gpu.output, expected);

  console.log(`diffusion transformer\t${supers} super-block(s) = ${supers * PER_SUPER} blocks`
    + `\tn=${tokens}\trelRMS ${relRms.toExponential(2)}`
    + `\t(envelope ${envelope.toExponential(2)},`
    + ` ${(relRms / Math.max(envelope, 1e-30)).toFixed(1)}x)`
    + `\t${gpu.elapsedMilliseconds.toFixed(0)} ms`
    + `\t${(gpu.memory.peakBytes / 2 ** 20).toFixed(1)} MiB`);

  // 🔴 THE f16 ARM'S BOUND IS LOOSE AND THE STRUCTURAL CLAIM IS NOT MADE HERE.
  // Half-precision weights measure 1.88e-2 on this checker's output against the
  // f32 path's ~1e-6, which is a big number to bless, so it is worth saying
  // exactly what it is and is not evidence of.
  //
  // It is a residual stack: eight blocks of synthetic weights amplify a 4.9e-4
  // per-weight rounding into the output, the same way the pairformer block
  // checker reads 17x the envelope where the assembled 48-block trunk reads
  // 4.3e-6. What decides whether that matters is a fold, and two were run -
  // 6MRR at 71 tokens and 1QYS at 91, flow-8, one recycle, against the same
  // seeds on the all-f32 tree:
  //
  //     6MRR   CA RMSD 0.0104 A   pLDDT 70.18260 -> 70.18063   1406 -> 740 MiB
  //     1QYS   CA RMSD 0.0093 A   pLDDT 69.54053 -> 69.54726   1414 -> 748 MiB
  //
  // ...with N-CA, CA-C and CA-CA medians identical to five decimals. AF3's own
  // accuracy on these is 0.7-0.9 A and the sampler's seed-to-seed spread is
  // ~0.1, so 0.01 A is three orders under the thing being measured.
  //
  // So this arm is here to catch a BROKEN kernel, which moves relRMS by orders,
  // and 4e-2 does that while leaving 2.1x over the measurement. The f32 arm
  // keeps the envelope rule and is the one that holds the arithmetic exact.
  const bound = weightPrecision === "f16" ? 4e-2 : Math.max(1e-5, envelope * 10);
  if (relRms > bound) {
    throw new Error(`relRMS ${relRms.toExponential(2)} exceeds ${bound.toExponential(2)}`
      + ` (${(relRms / envelope).toFixed(1)}x the rounding envelope)`);
  }
  return { tokens, supers, weightPrecision, relRms, bound, envelope,
           ms: gpu.elapsedMilliseconds };
}
