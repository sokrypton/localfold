/**
 * AF3's template embedder (empty-template path): GPU against
 * src/af3/trunk/template-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-template.js
 *
 * The check also reports the output's std against the input pair's, because the
 * number is the argument for the module existing at all: with four EMPTY slots
 * this is not a small residual correction.
 */
import { templateEmbedding } from "../../src/af3/trunk/template-reference.js";
import { Af3TemplateEmbedderGpu } from "../../src/af3/trunk/template-webgpu.js";
import { HttpTensorStore } from "../../src/bundles/http-tensor-store.js";
import { af3Dialect } from "../../src/af3/weights/weights.js";
import { deviceTuning } from "../../src/runtime/device-profile.js";

// 🔴 A DEFAULT, NOT A CONSTANT. This was hardcoded, so on a box that has the
// int5 bundle and not the f32 one the checker 404s instead of running - and
// two whole stacks' kernel choices went ungated here for exactly that reason.
// `--model=` picks the bundle; the bound follows, because an int5 bundle is
// quantised and its residue against a float32 oracle is not the f32 one's.
const MANIFEST = "/model-af3-full-f32/manifest.json";
const ROOT = "diffuser/evoformer/template_embedding";
const SINGLE = `${ROOT}/single_template_embedding`;
const STACK = `${SINGLE}/__layer_stack_no_per_layer/template_embedding_iteration`;
// 🔴 THE DIALECT IS THE BUNDLE'S, NOT A LITERAL. This file pinned
// `swapTransposedBias: false`, which is AlphaFold 3's - so on an OpenDDE
// bundle, whose column pair bias IS transposed, it compared two different
// models and read NaN, and on IntelliFold-2 it read 1.65e-1. Every other
// `--model=` checker derives it from `manifest.model.name`; this one now does
// too. See `af3Dialect`.

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

function standardDeviation(values) {
  let total = 0;
  for (const value of values) total += value;
  const mean = total / values.length;
  let squares = 0;
  for (const value of values) squares += (value - mean) ** 2;
  return Math.sqrt(squares / values.length);
}

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "32"));
  const templates = Number(option(args, "templates", "4"));
  const model = option(args, "model", MANIFEST);
  // 🔴 THE MATRIX GRID PROJECTION IS AN AXIS HERE TOO, AND IT WAS BLOCKING FOUR
  // ARMS OUT OF FIVE. This stack runs `gridProjectMatrix`, which issues on f16
  // matrix units, and the 2e-5 bound was written for the vector path - so the
  // FIRST arm (0 occupied slots) read 2.25e-5, threw, and the occupied and
  // SPANNING arms never ran at all. The spanning ones are the point of this
  // file: cross-chain template masking is where a permissive default once
  // scored relRMS 1.09 against AF3, and it had quietly stopped being checked.
  //
  //     matrix on    0 slots  2.25e-5   then nothing
  //     matrix off   0 slots  2.53e-7   1: 2.09e-7   4: 1.55e-7
  //                  1 spanning 2.13e-7   4 spanning 1.63e-7
  //
  // So the bound follows the KERNEL, as check-af3-trunk.js and
  // check-evoformer-attention.js do, and `--matrix=off` is the arm that gets
  // the vector path. 8e-5 is 3x the matrix measurement.
  const matrixWanted = option(args, "matrix", "auto") !== "off";
  const matrixLive = matrixWanted && deviceTuning(device).gridProjectMatrix !== undefined
    && deviceTuning(device).gridProjectMatrix !== null
    && deviceTuning(device).gridProjectMatrix !== false;
  const bound = Number(option(args, "bound",
    matrixLive ? (model === MANIFEST ? "8e-5" : "4e-4")
      : model === MANIFEST ? "2e-5" : "1e-4"));
  // 🔴 AND THE MATRIX ARM'S BOUND FOLLOWS THE STACK'S WIDTH. It is an f16
  // accumulation over the channel axis, so a 256-channel stack accumulates
  // four times the terms a 64-channel one does: IntelliFold-2 reads 5.72e-4
  // where OpenDDE reads 8.08e-5, and with `--matrix=off` if2 is 3.80e-7 - the
  // port, not the width. Scaled by sqrt of the ratio, which is what a sum of
  // independent roundings does, and it is not a licence to raise the bound
  // further: the SHIPPED trunk pins `pairMatrixKernels: false` on this stage
  // for exactly this reason (see src/af3/trunk/trunk-webgpu.js).
  const store = await HttpTensorStore.open(model);

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
    // 🔴 THE HEAD COUNT AND THE HEAD WIDTH ARE THE TENSOR'S, NOT AlphaFold 3's.
    // `heads: 4, dimension: 16` was typed in here - AF3's template stack - and
    // it is 2 x 32 under OpenDDE and 8 x 32 under IntelliFold-2, both of which
    // this checker then compared against a graph built for 4 x 16. OpenDDE read
    // NaN and IntelliFold-2 1.65e-1, and neither was the port: `q_projection`
    // is [blocks, heads, dimension, channels] and says so. This is CLAUDE.md's
    // standing note about hand-built weight dicts, one file later.
    const gridShape = (which) =>
      store.shape(`${STACK}/pair_attention${which}/q_projection/weights`);
    const grid = async (which) => ({
      heads: gridShape(which)[1], dimension: gridShape(which)[2],
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
  };

  const T = (name) => store.tensor(name);
  // 🔴 BOTH WIDTHS OFF THE BUNDLE, NOT OFF AlphaFold 3. `QUERY_CHANNELS = 128`
  // was typed in here and the stack's own width was a constant inside
  // template-webgpu.js; OpenDDE's query is 384 and IntelliFold-2's stack is
  // 256. This checker builds its weight dict by hand rather than through
  // `templateWeights` - the trap CLAUDE.md names - so it has to read them
  // itself, and these are the two tensors that state them.
  const dialect = af3Dialect(store);
  const queryChannels = (await T(`${SINGLE}/query_embedding_norm/scale`)).length;
  const channels = (await T(`${SINGLE}/output_layer_norm/scale`)).length;
  const widthScale = matrixLive && !args.some((a) => a.startsWith("--bound="))
    ? Math.max(1, Math.sqrt(channels / 64)) : 1;
  const weights = {
    queryChannels, channels,
    queryEmbeddingNormScale: await T(`${SINGLE}/query_embedding_norm/scale`),
    queryEmbeddingNormOffset: await T(`${SINGLE}/query_embedding_norm/offset`),
    outputLayerNormScale: await T(`${SINGLE}/output_layer_norm/scale`),
    outputLayerNormOffset: await T(`${SINGLE}/output_layer_norm/offset`),
    outputLinear: await T(`${ROOT}/output_linear/weights`),
    blocks: [await blockWeights(0), await blockWeights(1)],
  };
  // All nine projections. Six of them were unreachable while every slot was
  // empty and are the whole point of the second arm below.
  for (const index of [0, 1, 2, 3, 4, 5, 6, 7, 8]) {
    weights[`templatePairEmbedding${index}`] =
      await T(`${SINGLE}/template_pair_embedding_${index}/weights`);
  }

  const sequence = new Float32Array(tokens);
  for (let i = 0; i < tokens; i += 1) sequence[i] = i < Math.ceil(tokens * 0.8) ? 1 : 0;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = sequence[i] * sequence[j];
  }
  const pair = deterministic(tokens * tokens * queryChannels, 555 + tokens);

  // 🔴 TWO ARMS, BECAUSE THE EMPTY ONE REACHES THREE OF NINE FEATURES. With
  // every slot empty the six geometry projections multiply zero, so this
  // checker agreed to 1e-7 for months while `templatePairEmbedding0` was not
  // even being LOADED. The occupied arm is the one that exercises them, and it
  // fills only some slots so the empty and occupied branches run in the same
  // dispatch sequence rather than in two separate runs.
  const slotsFor = (occupied) => {
    if (occupied === 0) return undefined;
    const made = [];
    for (let slot = 0; slot < templates; slot += 1) {
      if (slot >= occupied) { made.push(undefined); continue; }
      // A backbone that is a real, if uninteresting, chain: a helix along x,
      // so the distogram spans many bins and every frame is well conditioned.
      const aatype = new Int32Array(tokens);
      const atomPositions = new Float32Array(tokens * 24 * 3);
      const atomMask = new Float32Array(tokens * 24);
      for (let token = 0; token < tokens; token += 1) {
        aatype[token] = (token * 7 + slot) % 20;
        const turn = token * 1.75 + slot;
        const centre = [token * 1.5, 2.3 * Math.cos(turn), 2.3 * Math.sin(turn)];
        // N, CA, C, O, CB - enough for a frame and a pseudo-beta.
        const offsets = [[-0.6, 0.6, 0], [0, 0, 0], [0.6, 0.6, 0], [0.9, 1.7, 0], [0, -0.5, 1.2]];
        for (const [index, offset] of offsets.entries()) {
          // ...glycine has no CB, which is what makes the pseudo-beta fall
          // back to CA for one residue in twenty here rather than never.
          if (index === 4 && aatype[token] === 7) continue;
          const at = (token * 24 + index) * 3;
          for (let axis = 0; axis < 3; axis += 1) {
            atomPositions[at + axis] = centre[axis] + offset[axis];
          }
          atomMask[token * 24 + index] = 1;
        }
      }
      made.push({ aatype, atomPositions, atomMask });
    }
    return made;
  };

  // 🔴 TWO CHAINS, BECAUSE ONE CHAIN CANNOT TELL THE MASKS APART. AF3 masks
  // the geometry across chains and this checker had a single-chain query, so
  // "mask per chain" and "mask nothing" were the same array - which is how a
  // permissive default survived until a two-chain oracle dump scored relRMS
  // 1.09 against AF3. Half the tokens are chain 1 and half chain 2.
  const asymId = new Int32Array(tokens);
  for (let token = 0; token < tokens; token += 1) asymId[token] = token < tokens / 2 ? 1 : 2;

  for (const [occupied, spanChains] of [[0, false], [1, false], [templates, false],
                                        [1, true], [templates, true]]) {
    const made = slotsFor(occupied);
    // Spanning is a property of the SLOT: it says these coordinates came from
    // one structure, so its cross-chain distances are real geometry rather
    // than two frames compared.
    if (made !== undefined && spanChains) {
      for (const slot of made) if (slot) slot.spanChains = true;
    }
    const input = { pair, pairMask, tokens, templates, asymId, slots: made };
    const expected = templateEmbedding(input, weights, dialect);
    const gpu = await new Af3TemplateEmbedderGpu(device, { pairMatrixKernels: matrixWanted })
      .run(input, weights, dialect, { pairMatrixKernels: matrixWanted });
    const relRms = relativeRms(gpu.output, expected);
    console.log(`template\ttokens=${tokens} slots=${templates}`
      + ` occupied=${occupied}${spanChains ? " spanning" : ""}`
      + `\trelRMS ${relRms.toExponential(2)}`
      + `\t${gpu.elapsedMilliseconds.toFixed(1)} ms`
      + `\tstd ${standardDeviation(gpu.output).toFixed(2)}`);
    // 🔴 THE BOUND FOLLOWS THE BUNDLE. 2e-5 is a float32 bundle's residue
    // against a float32 oracle; the published int5 bundle is quantised, and at
    // 0 occupied slots it reads 2.41e-5 - larger for a reason that has nothing
    // to do with this kernel. `--bound=` is what says so, rather than a number
    // raised until both pass, which would stop checking the f32 one.
    if (!(relRms < bound * widthScale)) {
      throw new Error(`template with ${occupied} occupied slots`
        + `${spanChains ? " spanning" : ""}: relRMS ${relRms}`);
    }
    // 🔴 AND SPANNING HAS TO CHANGE THE ANSWER, or the flag is decoration.
    // Cross-chain geometry is most of what a two-chain template knows, so an
    // arm that agrees with its masked twin means the mask never opened.
    if (spanChains) {
      const masked = templateEmbedding(
        { ...input, slots: slotsFor(occupied) }, weights, dialect);
      const moved = relativeRms(expected, masked);
      console.log(`  spanning moves the output by relRMS ${moved.toExponential(2)}`);
      if (!(moved > 1e-3)) {
        throw new Error(`spanChains changed nothing (relRMS ${moved})`);
      }
    }
  }
  const input = { pair, pairMask, tokens, templates, asymId };
  const gpu = await new Af3TemplateEmbedderGpu(device).run(input, weights, dialect);
  // The argument for the module existing: this is not a small correction.
  console.log(`output std ${standardDeviation(gpu.output).toFixed(2)}`
    + ` against an input pair std of ${standardDeviation(pair).toFixed(2)}`
    + ` - with ${templates} EMPTY slots`);

  // Each arm asserted its own bound in the loop above.
  return { tokens, templates, outputStd: standardDeviation(gpu.output) };
}
