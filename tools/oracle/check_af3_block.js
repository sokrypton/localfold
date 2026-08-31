/**
 * Run our AF3 pairformer block against the oracle's, with the real weights.
 *
 *     node tools/oracle/check_af3_block.js
 *
 * Both sides come from ONE dump, and that is the point:
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 2 \
 *         --capture 'trunk_pairformer/__call__$' --out af3-oracle-2block.json
 *
 * captures every block's output in execution order, so block 0's output IS
 * block 1's input and the two are guaranteed consistent. Nothing here builds
 * features or embeddings; the block is tested on the activations AF3 itself
 * produced, in isolation from everything that made them.
 *
 * 🔴 THE OBVIOUS VERSION OF THIS IS WRONG. Taking --blocks 0 as the input and
 * --blocks 1 as the output looks equivalent and is not: the 0-block trunk
 * returns a pair that is NOT the pairformer stack's input (measured: they
 * differ by 3.6e-3 relative, which is four percent of everything one block
 * does), so the comparison silently charges our block for a difference it did
 * not make. Two depths of the same run are not two points on one path.
 *
 * 🔴 THE WEIGHTS ARE THE SHIPPED int8 ONES, so a disagreement is our wiring
 * PLUS our quantisation and the two have to be told apart before either is
 * believed. Export the float32 master to a directory and pass --model to read
 * that instead; the graph is only worth arguing about once f32 agrees.
 *
 * Both inputs are gitignored artefacts, so this is a script rather than a test:
 * it needs a 150 MiB export and a JAX run that no checkout has by default.
 */
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { readTensor } from "../../src/reference/dtype.js";
import { pairformerBlock } from "../../src/af3/pairformer-reference.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const STACK = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer";

/** Every tensor of a model directory, by name, already widened to float32. */
async function loadTensors(directory) {
  const manifest = JSON.parse(await readFile(join(directory, "manifest.json"), "utf8"));
  const shards = new Map();
  const tensors = new Map();
  for (const [name, record] of Object.entries(manifest.tensors)) {
    if (!shards.has(record.file)) {
      const bytes = await readFile(join(directory, record.file));
      shards.set(record.file,
                 bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    }
    tensors.set(name, {
      shape: record.shape,
      data: readTensor(record, shards.get(record.file), record.byteOffset, true),
    });
  }
  return { manifest, tensors };
}

/**
 * One layer out of a stacked tensor.
 *
 * 🔴 THE STACK AXIS IS FIRST, so layer `index` is a contiguous slice - but only
 * because the export keeps AF3's own (num_layer, ...) layout. Reading it as the
 * last axis would still produce a correctly shaped tensor of the wrong numbers.
 */
function layer(tensors, name, index) {
  const tensor = tensors.get(name);
  if (tensor === undefined) throw new Error(`no tensor named ${name}`);
  const stride = tensor.data.length / tensor.shape[0];
  return tensor.data.subarray(index * stride, (index + 1) * stride);
}

/** The block's tensors, named as the reference expects them. */
function blockWeights(tensors, index) {
  const at = (leaf) => layer(tensors, `${STACK}/${leaf}`, index);
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
  const gridAttention = (which) => ({
    heads: 4,
    dimension: 32,
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
    pairChannels: 128,
    singleChannels: 384,
    triangleMultiplicationOutgoing: triangle("outgoing"),
    triangleMultiplicationIncoming: triangle("incoming"),
    pairAttention1: gridAttention(1),
    pairAttention2: gridAttention(2),
    pairTransition: {
      inputLayerNormScale: at("pair_transition/input_layer_norm/scale"),
      inputLayerNormOffset: at("pair_transition/input_layer_norm/offset"),
      transition1: at("pair_transition/transition1/weights"),
      transition2: at("pair_transition/transition2/weights"),
    },
    singlePairLogitsNormScale: at("single_pair_logits_norm/scale"),
    singlePairLogitsNormOffset: at("single_pair_logits_norm/offset"),
    singlePairLogitsProjection: at("single_pair_logits_projection/weights"),
    singleAttention: {
      heads: 16,
      dimension: 24,
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
  };
}

/** Relative RMS of a difference, and the reference's own RMS beside it.
 *
 * 🔴 A RELATIVE ERROR IS UNREADABLE WITHOUT ITS DENOMINATOR. The multimer work
 * lost an afternoon to a block that looked four times worse than its neighbour
 * and was not: its reference values were four times smaller.
 */
function compare(reference, ours) {
  let error = 0;
  let magnitude = 0;
  let worst = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const difference = ours[index] - reference[index];
    error += difference * difference;
    magnitude += reference[index] * reference[index];
    if (Math.abs(difference) > worst) worst = Math.abs(difference);
  }
  const rms = Math.sqrt(magnitude / reference.length);
  return { relative: Math.sqrt(error / magnitude), rms, worst };
}

const load = async(name) => JSON.parse(await readFile(join(ROOT, name), "utf8"));

async function main() {
  const model = process.argv.includes("--model")
    ? process.argv[process.argv.indexOf("--model") + 1] : "model-af3";
  // 🔴 THE FLOAT32 DUMP IS THE REFERENCE, NOT THE DEFAULT ONE. AF3's trunk
  // computes in BFLOAT16, whose relative epsilon is 2^-8 = 3.9e-3 - so a
  // bfloat16 dump cannot tell a correct block from one that is a third of a
  // percent wrong, and this block measured 4.2e-3 against it while being exact.
  // Dump it with --float32 --blocks 2. That run is not what the model does; it
  // is the only thing that can hold a reimplementation to account.
  const dump = await load(process.argv.includes("--bfloat16")
    ? "af3-oracle-2block.json" : "af3-oracle-2block-f32.json");
  const { manifest, tensors } = await loadTensors(join(ROOT, model));
  if (manifest.model.name !== dump.model) {
    throw new Error(`${model}/ holds ${manifest.model.name} but the oracle ran`
      + ` ${dump.model}; they must be the same weights`);
  }
  // Block `index` reads what block `index - 1` wrote, so the dump must go at
  // least one block deeper than the block being checked.
  const index = 1;
  const captured = (which, call) => {
    const key = `trunk_pairformer/__call__:[${which}]#${call}`;
    const record = dump.outputs[key];
    if (record === undefined) {
      throw new Error(`${key} is not in the dump; re-run dump_af3_trunk.py with`
        + ` --blocks ${index + 1} --capture 'trunk_pairformer/__call__$'`);
    }
    return Float32Array.from(record.data);
  };

  const tokens = dump.tokens;
  const seqMask = Float32Array.from(dump.inputs.seq_mask.data);
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }

  const result = pairformerBlock({
    pair: captured(0, index - 1),
    single: captured(1, index - 1),
    pairMask,
    seqMask,
    tokens,
  }, blockWeights(tensors, index), {
    // Only the OpenFold3 lineage transposes the column-wise pair bias.
    swapTransposedBias: dump.model !== "alphafold3",
  });

  console.log(`${dump.model}, ${tokens} tokens, block ${index},`
    + ` weights from ${model}/ (${manifest.bundle.encoding})`);
  for (const [name, which, ours] of [["pair", 0, result.pair],
                                     ["single", 1, result.single]]) {
    const reference = captured(which, index);
    const { relative, rms, worst } = compare(reference, ours);
    console.log(`  ${name.padEnd(7)} relRMS ${relative.toExponential(3)}`
      + `   worst ${worst.toExponential(2)}   reference RMS ${rms.toFixed(3)}`);
  }
}

await main();
