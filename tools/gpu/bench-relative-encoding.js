/**
 * Is the gather actually faster than the dense projection?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-relative-encoding.js --tokens=256
 *
 * The embedder projects the relative encoding's 139 columns into 128 channels.
 * Those columns hold three one-hot bins and one binary flag, so the projection
 * can be a sum of four weight rows instead of a 139x128 product. That is 139x
 * less arithmetic, which is NOT the same claim as 139x faster - so measure.
 *
 * 🔴 THE DENSE ARM STILL BUILDS ITS ONE-HOT IN REGISTERS rather than reading a
 * materialised tokens^2 x 139 tensor from memory. That is the BEST case for
 * dense, and deliberately so: if the gather wins against this it wins against
 * the real alternative by more.
 *
 * 🔴 THE ARMS ARE INTERLEAVED, not run in two batches. Run-to-run drift on this
 * machine is several-fold, so two consecutive batches compare the drift.
 */
import { Af3EmbedderGpu } from "../../src/af3/embedder-webgpu.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const EVO = "diffuser/evoformer";
const FEATURE_WIDTH = 447;

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "192"));
  const sequences = Number(option(args, "sequences", "8"));
  const repeats = Number(option(args, "repeats", "7"));
  const store = await HttpTensorStore.open(MANIFEST);
  const T = async (name) => store.tensor(`${EVO}/${name}`);

  const weights = {
    pairChannels: 128, singleChannels: 384, msaChannels: 64, targetFeatWidth: FEATURE_WIDTH,
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
  };

  const residueIndex = new Int32Array(tokens);
  const zeros = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) residueIndex[t] = t;
  const input = {
    tokens, sequences,
    targetFeat: new Float32Array(tokens * FEATURE_WIDTH).fill(0.01),
    features: { residueIndex, tokenIndex: residueIndex, asymId: zeros,
                entityId: zeros, symId: zeros },
    msaRows: new Int32Array(sequences * tokens),
    deletionMatrix: new Float32Array(sequences * tokens),
    previousPair: new Float32Array(tokens * tokens * 128).fill(0.01),
    previousSingle: new Float32Array(tokens * 384).fill(0.01),
  };

  const runner = new Af3EmbedderGpu(device);
  const times = { gather: [], dense: [] };
  // Warm up both so shader compilation is outside the timing.
  for (const relative of ["gather", "dense"]) await runner.run(input, weights, { relative });
  for (let attempt = 0; attempt < repeats; attempt += 1) {
    for (const relative of ["gather", "dense"]) {
      const result = await runner.run(input, weights, { relative });
      times[relative].push(result.elapsedMilliseconds);
    }
  }

  const median = (values) => {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };
  const gather = median(times.gather);
  const dense = median(times.dense);
  console.log(`tokens=${tokens}, ${repeats} interleaved pairs`);
  console.log(`gather\t${gather.toFixed(1)} ms\t[${times.gather.map((t) => t.toFixed(0)).join(" ")}]`);
  console.log(`dense\t${dense.toFixed(1)} ms\t[${times.dense.map((t) => t.toFixed(0)).join(" ")}]`);
  console.log(`speedup\t${(dense / gather).toFixed(2)}x`);
  return { tokens, gatherMs: gather, denseMs: dense, speedup: dense / gather };
}
