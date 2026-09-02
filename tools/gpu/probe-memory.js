/**
 * Where the HOST memory of a fold goes.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-memory.js --blocks=48
 *
 * 🔴 THE GPU IS NOT THE ONLY PLACE A MODEL COSTS MEMORY. The weights arrive as
 * 265 MB of int5 and are decoded to float32 on the host - 1405 MiB for AF3's
 * full manifest - and every decoded tensor is retained by HttpTensorStore for
 * the life of the page. A block's weights are then PACKED into another
 * Float32Array, and that is retained too. None of it is visible in the GPU
 * allocator's snapshot, which is the only memory number the benches print.
 *
 * This walks the load in stages and reports usedJSHeapSize after each, so the
 * cost of each retention can be named rather than guessed at.
 */
import { openAf3Store, trunkWeights, confidenceWeights } from "../../src/af3/weights.js";
import { targetFeatureWeights, diffusionWeights, atomReference } from "../../src/af3/diffusion-weights.js";
import { featuriseProtein } from "../../src/af3/featurise.js";
import { buildTargetFeat, DIALECT } from "../../src/af3/fold.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const MIB = 1024 * 1024;

async function heap() {
  // 🔴 COLLECT FIRST OR THE NUMBER IS NOISE. usedJSHeapSize counts garbage that
  // has not been swept, so a reading taken straight after a phase can be
  // hundreds of MiB above what is actually retained - which is exactly the
  // difference this probe exists to measure. tools/gpu-chrome.mjs runs Chrome
  // with --expose-gc for this.
  for (let round = 0; round < 3; round += 1) {
    globalThis.gc?.();
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return performance.memory?.usedJSHeapSize ?? 0;
}

export async function main(device, args) {
  const blocks = Number(option(args, "blocks", "48"));
  const manifestUrl = option(args, "model", "/model-af3-full-f32/manifest.json");
  const stages = [];
  const note = async (label, extra = {}) => {
    stages.push({ label, heapMiB: Number(((await heap()) / MIB).toFixed(1)), ...extra });
  };

  await note("start");
  const store = await openAf3Store(manifestUrl);
  const manifest = store.manifest ?? {};
  const tensors = Object.values(manifest.tensors ?? {});
  const f32Bytes = tensors.reduce(
    (sum, record) => sum + record.shape.reduce((a, b) => a * b, 1) * 4, 0);
  await note("manifest open");

  const weights = { trunk: await trunkWeights(store, blocks, 4),
                    targetFeat: await targetFeatureWeights(store) };
  await note("trunk weights");
  // ...and the rest of what a fold needs, so the page's heap can be attributed
  // rather than guessed at from the trunk alone.
  weights.diffusion = await diffusionWeights(store);
  await note("+ diffusion head");
  weights.confidence = await confidenceWeights(store);
  await note("+ confidence");
  weights.atomReference = await atomReference(store);
  await note("+ atom reference");

  const tokens = Number(option(args, "tokens", "59"));
  const rows = Number(option(args, "msa", "32"));
  const sequence = Array.from({ length: tokens },
    (_, index) => "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"[index % 59]).join("");
  const batch = featuriseProtein(sequence, {});
  const targetFeat = await buildTargetFeat(batch, weights.targetFeat, device);
  const msa = new Int32Array(rows * tokens);
  const deletionMatrix = new Float32Array(rows * tokens);
  const msaMask = new Float32Array(rows * tokens).fill(1);
  for (let row = 0; row < rows; row += 1) {
    for (let token = 0; token < tokens; token += 1) {
      msa[row * tokens + token] = (batch.msa[token] + row) % 20;
    }
  }
  const seqMask = batch.seqMask;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }
  const trunkGpu = new Af3TrunkGpu(device);
  await trunkGpu.run({
    tokens, sequences: rows, templates: 4, targetFeat, features: batch.features,
    msaRows: msa, deletionMatrix, msaMask, bondMatrix: batch.bondMatrix,
    pairMask, seqMask,
    previousPair: new Float32Array(tokens * tokens * 128),
    previousSingle: new Float32Array(tokens * 384),
  }, weights.trunk, DIALECT, {});
  await note("one trunk pass");

  // What the store is holding: every tensor any loader has touched, as float32,
  // plus the compressed shards behind them.
  return {
    manifest: manifestUrl,
    tensorCount: tensors.length,
    decodedFloat32MiB: Number((f32Bytes / MIB).toFixed(0)),
    stages,
    heldTensors: store.cachedTensorCount?.() ?? null,
    weightsLoaded: weights.trunk !== undefined,
  };
}
