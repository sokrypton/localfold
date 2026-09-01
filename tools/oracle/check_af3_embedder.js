/**
 * Run the embedder and the whole MSA stack, and compare against AF3.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 0 --float32 \
 *         --capture 'evoformer/template_embedding/__call__$|evoformer/__call__$|msa_stack/__call__$' \
 *         --out af3-oracle-embed-f32.json
 *     node tools/oracle/check_af3_embedder.js
 *
 * This is the first check that starts from FEATURES rather than from an
 * activation AF3 produced, so it is the first that could be wrong about what
 * the trunk is fed rather than about what it does with it.
 *
 * 🔴 TWO INPUTS ARE STILL TAKEN FROM THE ORACLE, and the report says so on
 * every run rather than leaving it to be inferred: `target_feat`, whose 384
 * atom-derived columns need the atom transformer encoder, and the template
 * embedding, which is a two-block stack of its own. Neither is implemented. A
 * result that looks exact here is exact ABOUT THE REST.
 */
import { join } from "node:path";

import { embed } from "../../src/af3/embedder-reference.js";
import { msaBlock } from "../../src/af3/msa-reference.js";
import { ROOT, captures, layer, loadDump, loadTensors, report } from "./af3-bundle.js";

const EVO = "diffuser/evoformer";
const STACK = `${EVO}/__layer_stack_no_per_layer/msa_stack`;

function embedderWeights(tensors) {
  const at = (name) => {
    const tensor = tensors.get(`${EVO}/${name}`);
    if (tensor === undefined) throw new Error(`no tensor named ${EVO}/${name}`);
    return tensor.data;
  };
  return {
    pairChannels: 128,
    singleChannels: 384,
    msaChannels: 64,
    targetFeatWidth: 447,
    relativeWidth: 139,
    leftSingle: at("left_single/weights"),
    rightSingle: at("right_single/weights"),
    prevEmbedding: at("prev_embedding/weights"),
    prevEmbeddingNormScale: at("prev_embedding_layer_norm/scale"),
    prevEmbeddingNormOffset: at("prev_embedding_layer_norm/offset"),
    positionActivations: at("~_relative_encoding/position_activations/weights"),
    bondEmbedding: at("bond_embedding/weights"),
    msaActivations: at("msa_activations/weights"),
    extraMsaTargetFeat: at("extra_msa_target_feat/weights"),
    singleActivations: at("single_activations/weights"),
    prevSingleEmbedding: at("prev_single_embedding/weights"),
    prevSingleEmbeddingNormScale: at("prev_single_embedding_layer_norm/scale"),
    prevSingleEmbeddingNormOffset: at("prev_single_embedding_layer_norm/offset"),
  };
}

function msaBlockWeights(tensors, index) {
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
    msaChannels: 64,
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
      heads: 8,
      dimension: 8,
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
  };
}

async function main() {
  const model = process.argv.includes("--model")
    ? process.argv[process.argv.indexOf("--model") + 1] : "model-af3-f32";
  const dump = await loadDump("af3-oracle-embed-f32.json");
  const { manifest, tensors } = await loadTensors(join(ROOT, model));
  const at = captures(dump, "dump_af3_trunk.py --blocks 0 --float32 with the"
    + " embedder capture (see the header of this file)");

  const tokens = dump.tokens;
  const input = (name) => dump.inputs[name].data;
  const seqMask = Float32Array.from(input("seq_mask"));
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }

  // 🔴 THE TRUNK RUNS THE FIRST num_msa ROWS, not the whole featurised MSA. The
  // dump carries eight rows and the model was configured for one, so slicing
  // here is not a convenience - taking all eight would build a different msa
  // representation and disagree from the first block.
  const sequences = dump.outputs["msa_stack/__call__:msa#0"].shape[0];
  const msaRows = input("msa").slice(0, sequences * tokens);
  const deletionMatrix = input("deletion_matrix").slice(0, sequences * tokens);

  const embedded = embed({
    tokens,
    sequences,
    targetFeat: at(`${EVO}/__call__:target_feat`),
    templateEmbedding: at(`${EVO}/template_embedding/__call__`),
    msaRows,
    deletionMatrix,
    features: {
      residueIndex: input("residue_index"),
      tokenIndex: input("token_index"),
      asymId: input("asym_id"),
      entityId: input("entity_id"),
      symId: input("sym_id"),
    },
  }, embedderWeights(tensors));

  console.log(`${dump.model}, ${tokens} tokens, ${sequences} sequence(s),`
    + ` weights from ${model}/ (${manifest.bundle.encoding})`);
  console.log("  target_feat and the template embedding are taken from the oracle;"
    + " everything else is ours");

  const msaMask = Float32Array.from(input("msa_mask")).subarray(0, sequences * tokens);
  let state = {
    msa: embedded.msa,
    pair: embedded.pair,
    msaMask,
    pairMask,
    sequences,
    tokens,
  };
  const dialect = { swapTransposedBias: dump.model !== "alphafold3" };
  const depth = dump.outputs["msa_stack/__call__:pair#3"] === undefined ? 1 : 4;
  for (let index = 0; index < depth; index += 1) {
    const next = msaBlock(state, msaBlockWeights(tensors, index), dialect);
    state = { ...state, msa: next.msa, pair: next.pair };
    console.log(`  after MSA block ${index}`);
    report("msa", at(`msa_stack/__call__:msa#${index}`), next.msa);
    // 🔴 WHICH ROWS, because "the msa is 20% wrong" and "row 17 onwards is
    // wrong" are different bugs and the relRMS cannot tell them apart. The MSA
    // is the one tensor here with a depth axis, so a slicing or padding mistake
    // shows up as whole rows being wrong and the rest being exact.
    if (process.argv.includes("--where") && index === 0) {
      const theirs = at(`msa_stack/__call__:msa#${index}`);
      const width = theirs.length / sequences;
      const rows = [];
      for (let s = 0; s < sequences; s += 1) {
        let error = 0;
        let scale = 0;
        for (let k = 0; k < width; k += 1) {
          const a = next.msa[s * width + k];
          const b = theirs[s * width + k];
          error += (a - b) ** 2; scale += b * b;
        }
        rows.push({ s, rel: Math.sqrt(error / Math.max(scale, 1e-30)) });
      }
      const bad = rows.filter((r) => r.rel > 1e-4).map((r) => r.s);
      const good = rows.filter((r) => r.rel <= 1e-4).map((r) => r.s);
      console.log(`    rows exact (<=1e-4): ${good.length ? good.join(",") : "none"}`);
      console.log(`    rows wrong  (>1e-4): ${bad.length ? bad.join(",") : "none"}`);
      console.log("    per-row relRMS: " + rows.slice(0, 8)
        .map((r) => `${r.s}:${r.rel.toExponential(1)}`).join(" "));
    }
    report("pair", at(`msa_stack/__call__:pair#${index}`), next.pair);
  }
  console.log("  trunk output (the pairformer stack's input)");
  report("pair", at(`${EVO}/__call__:pair`), state.pair);
  report("single", at(`${EVO}/__call__:single`), embedded.single);
}

await main();
