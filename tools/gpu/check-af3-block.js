/**
 * AF3 pairformer blocks on the GPU, against src/af3/pairformer-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-block.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-block.js --n=64 --blocks=8
 *
 * The per-kernel checkers pin each operation. This pins what they cannot: the
 * ORDER, the residual chain, and the fact that the single track reads the pair
 * after all five pair updates rather than before. Every one of those is a
 * reordering that still runs and still returns a pair representation.
 *
 * It also answers whether error compounds down a stack, which is the question a
 * one-block check cannot: the CPU reference showed it does not over 48 blocks
 * (1.4e-7 at block 8 against 4.3e-7 at block 47), and the GPU should behave the
 * same way.
 */
import { pairformerBlock } from "../../src/af3/pairformer-reference.js";
import { Af3PairformerStackGpu } from "../../src/af3/pairformer-block-webgpu.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const STACK = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer";
const DIALECT = { swapTransposedBias: false };

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
  const n = Number(option(args, "n", "32"));
  const count = Number(option(args, "blocks", "4"));
  const store = await HttpTensorStore.open(MANIFEST);

  const layer = async (leaf, index) => {
    const name = `${STACK}/${leaf}`;
    const whole = await store.tensor(name);
    const stride = whole.length / store.shape(name)[0];
    return whole.subarray(index * stride, (index + 1) * stride);
  };

  const blockWeights = async (index) => {
    const at = (leaf) => layer(leaf, index);
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
      heads: 4, dimension: 32,
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
      pairChannels: 128, singleChannels: 384,
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
  };

  // --stack=confidence runs the SAME stack with the confidence head's four
  // blocks instead of the trunk's, which separates "these weights" from "that
  // input" for the single-track residual recorded in check-af3-confidence.js.
  const blocks = [];
  if (option(args, "stack", "trunk") === "confidence") {
    const { confidenceWeights } = await import("../../src/af3/weights.js");
    const confidence = await confidenceWeights(store);
    for (let index = 0; index < Math.min(count, 4); index += 1) blocks.push(confidence.blocks[index]);
  } else {
    for (let index = 0; index < count; index += 1) blocks.push(await blockWeights(index));
  }

  const sequence = new Float32Array(n);
  for (let i = 0; i < n; i += 1) sequence[i] = i < Math.ceil(n * 0.8) ? 1 : 0;
  const pairMask = new Float32Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) pairMask[i * n + j] = sequence[i] * sequence[j];
  }
  // The confidence head feeds this stack a pair of std ~6 rather than ~0.6, so
  // the scale is a knob here too.
  const pairScale = Number(option(args, "pair-scale", "1"));
  const scaledPair = deterministic(n * n * 128, 4242 + n);
  for (let index = 0; index < scaledPair.length; index += 1) scaledPair[index] *= pairScale;
  const state = {
    tokens: n,
    pair: scaledPair,
    single: deterministic(n * 384, 8484 + n),
    pairMask, seqMask: sequence,
  };

  // --no-check reports time and memory only. The CPU reference is O(N^3) in
  // JavaScript, so it is minutes per block by 256 tokens - useless for a
  // measurement and not what a measurement is asking anyway.
  if (args.includes("--no-check")) {
    const gpu = await new Af3PairformerStackGpu(device).run(state, blocks, DIALECT);
    console.log(`${count} block(s), n=${n}\t${gpu.elapsedMilliseconds.toFixed(1)} ms`
      + `\t${(gpu.elapsedMilliseconds / count).toFixed(1)} ms/block`
      + `\t${(gpu.memory.peakBytes / 2 ** 20).toFixed(1)} MiB peak`);
    return { n, blocks: count, msPerBlock: gpu.elapsedMilliseconds / count,
             peakMiB: gpu.memory.peakBytes / 2 ** 20 };
  }

  // The reference, block by block.
  let cpu = { pair: state.pair, single: state.single };
  for (const weights of blocks) {
    cpu = pairformerBlock({ ...cpu, pairMask, seqMask: sequence, tokens: n }, weights, DIALECT);
  }

  // 🔴 PAST ABOUT THREE BLOCKS AN ABSOLUTE TOLERANCE MEASURES THE STACK, NOT
  // THE KERNEL. On a RANDOM pair representation this stack is chaotic: perturb
  // the input by a relative 1e-7 - one kernel's worth of rounding - and run the
  // CPU reference against ITSELF, and the two runs diverge to 6e-4 by block 3
  // and stay there. The GPU reproduces those numbers to two digits, which is
  // the actual evidence that it agrees; a fixed 1e-5 bound would fail it for
  // being a different but equally valid rounding of the same function.
  //
  // This is a property of random input, not of AF3. On the real trunk the CPU
  // reference tracks DeepMind over all 48 blocks without compounding (1.4e-7 at
  // block 8, 4.3e-7 at block 47).
  //
  // So the check is: the GPU must land inside the envelope that rounding alone
  // produces. A wrong kernel scores ~1 and misses it by three orders.
  const perturbed = Float32Array.from(state.pair);
  for (let index = 0; index < perturbed.length; index += 1) {
    perturbed[index] += state.pair[index] * 1e-7;
  }
  let control = { pair: perturbed, single: state.single };
  for (const weights of blocks) {
    control = pairformerBlock({ ...control, pairMask, seqMask: sequence, tokens: n },
                              weights, DIALECT);
  }
  const envelope = relativeRms(control.pair, cpu.pair);

  // 🔴 THE STAGED ATTENTION TILE'S PRECISION IS AN AXIS. A block stages grid
  // attention's key and value in f16 wherever the device has shader-f16, so one
  // bound cannot hold both it and the f32 path - and raising the single bound
  // would stop the f32 path being checked at the 1.7e-6 it actually reaches.
  const stagedPrecision = option(args, "staged",
    device.features.has("shader-f16") ? "f16" : "f32");
  const f16 = stagedPrecision === "f16";
  const accumulatePrecision = option(args, "accumulate",
    device.features.has("shader-f16") ? "f16" : "f32");
  const accumulate16 = accumulatePrecision === "f16";
  // 🔴 THE SINGLE TRACK'S WEIGHTS ARE AN AXIS HERE TOO, and leaving them out
  // meant an arm asking for "f32" still ran f16 weights - its single track read
  // 3.14e-5 against a 1e-5 bound written for a path it was not on.
  const weightPrecision = option(args, "weights",
    device.features.has("shader-f16") ? "f16" : "f32");
  const weight16 = weightPrecision === "f16";
  const gpu = await new Af3PairformerStackGpu(
    device, { stagedPrecision, accumulatePrecision, weightPrecision },
  ).run(state, blocks, DIALECT);
  const pairRms = relativeRms(gpu.pair, cpu.pair);
  const singleRms = relativeRms(gpu.single, cpu.single);
  console.log(`${count} block(s), n=${n}, staged ${stagedPrecision},`
    + ` accumulate ${accumulatePrecision}, weights ${weightPrecision}`);
  console.log(`pair\trelRMS ${pairRms.toExponential(2)}`
    + `\t(rounding envelope ${envelope.toExponential(2)},`
    + ` ${(pairRms / Math.max(envelope, 1e-30)).toFixed(1)}x)`);
  console.log(`single\trelRMS ${singleRms.toExponential(2)}`);
  console.log(`${gpu.elapsedMilliseconds.toFixed(1)} ms`
    + `\t${(gpu.memory.peakBytes / 2 ** 20).toFixed(1)} MiB peak`
    + `\t${(gpu.elapsedMilliseconds / count).toFixed(1)} ms/block`);

  // The single track is not chaotic here, so it keeps an absolute bound.
  // f16's eleven significant bits on the staged key and value, carried through
  // `count` blocks of residual accumulation. Measured 1.8e-5 at two blocks.
  // The single track's own number is set by the SINGLE track's weights, and it
  // sees the triangle's accumulators through the pair logits on top: 6.01e-7
  // all-f32, 3.14e-5 with f16 weights, 9.87e-4 with the accumulators as well.
  const singleBound = accumulate16 ? 3e-3 : weight16 ? 2e-4 : 1e-5;
  if (singleRms > singleBound) {
    throw new Error(`single relRMS ${singleRms.toExponential(2)} exceeds ${singleBound}`);
  }
  // The pair gets the larger of a flat floor and ten times the envelope, so a
  // single block - where the envelope is small and the GPU injects error at
  // every operation rather than only at the input - is still held tightly.
  // 🔴 AND EACH f16 ARM GETS ITS OWN MULTIPLE, DERIVED RATHER THAN CONVENIENT.
  // The pair track is chaotic across RANDOM INPUT - the note above says why,
  // and says the real trunk is not - so its bound has always been a multiple of
  // the f32 rounding envelope rather than a flat number. At four blocks:
  //
  //     staged f32, acc f32    1.0x     staged f16, acc f32   17.1x
  //     staged f32, acc f16  200.2x     staged f16, acc f16  200.5x
  //
  // 🔴 THE TWO ARE NOT COMPARABLE NUMBERS, AND READING THEM AS ONE SCALE IS THE
  // MISTAKE TO AVOID. The envelope is built by perturbing the INPUT by 1e-7 and
  // measuring what four blocks do to it - a 5,740x amplification. The staged
  // tile perturbs the arithmetic by ~2e-4 and the triangle's accumulators by
  // ~2e-3, which is three to four orders more than the probe, so a proportional
  // response would be tens of thousands of times the envelope rather than 200.
  // What the ratio says is that this system damps those perturbations, not that
  // the accumulators are 12x worse than the staging.
  //
  // The number that decides is the assembled trunk on real input, where the
  // same change moves the pair from 6.18e-7 to 1.49e-5 and two whole folds by
  // 0.009 and 0.012 A. 250 keeps a margin over 200.5 while staying three orders
  // under the ~1 that the docstring above says a WRONG kernel scores, which is
  // what this check is for.
  const pairBound = Math.max(1e-5, envelope * (accumulate16 ? 250 : f16 ? 20 : 10));
  if (pairRms > pairBound) {
    throw new Error(`pair relRMS ${pairRms.toExponential(2)} exceeds ${pairBound.toExponential(2)}`
      + ` (${(pairRms / envelope).toFixed(1)}x the rounding envelope)`);
  }
  return {
    n, blocks: count, pairRms, singleRms, envelope,
    msPerBlock: gpu.elapsedMilliseconds / count,
    peakMiB: gpu.memory.peakBytes / 2 ** 20,
  };
}
