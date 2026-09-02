/**
 * When a fold does not fit, does it slow down or give up?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-budget.js --tokens=384
 *
 * 🔴 THE BENCHES ALL RUN WITH NO CEILING AND THE PAGE NEVER DOES. `--budget=0`
 * leaves the device merely counted, so every number in this repository was
 * measured on a machine that would accept anything - while web/model.js sets
 * budgetForDevice(), which is a third of what the browser admits to having. A
 * fold that fits in a bench can therefore be refused in the page, and nothing
 * measured which.
 *
 * This runs the same trunk at a ladder of ceilings and reports, for each, what
 * the fold did: finished at full speed, finished after giving up weight
 * residency (slower, and by how much), or refused with the tensor named.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { buildTargetFeat, DIALECT } from "../../src/af3/fold.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";
import { memorySnapshot, setMemoryBudget, budgetForDevice }
  from "../../src/runtime/device-memory.js";
import { openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { targetFeatureWeights } from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const MiB = 1048576;

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "384"));
  const rows = Number(option(args, "msa", "128"));
  const blocks = Number(option(args, "blocks", "48"));

  // What the PAGE would use on this machine, which is the number that matters.
  const reported = typeof navigator.deviceMemory === "number" ? navigator.deviceMemory : null;
  const pageBudget = budgetForDevice();

  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const weights = {
    trunk: await trunkWeights(store, blocks, 4),
    targetFeat: await targetFeatureWeights(store),
  };
  const sequence = Array.from({ length: tokens },
    (_, index) => "ACDEFGHIKLMNPQRSTVWY"[index % 20]).join("");
  const batch = featuriseProtein(sequence, {});
  const targetFeat = await buildTargetFeat(batch, weights.targetFeat, device);

  // A synthetic alignment of the requested depth; only the shape costs anything.
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

  const ladder = (option(args, "budgets", "") !== ""
    ? option(args, "budgets", "").split(",").map(Number)
    : [0, Math.round(pageBudget / MiB), 1200, 800, 500, 300, 200]);

  const results = [];
  for (const budgetMiB of ladder) {
    setMemoryBudget(device, budgetMiB > 0 ? budgetMiB * MiB : undefined);
    // 🔴 A FRESH STACK PER ARM. residencyAllowed is remembered on the DEVICE, so
    // an arm that gave up residency would leave every later arm degraded and the
    // ladder would read as though the ceiling above it had done it.
    const trunkGpu = new Af3TrunkGpu(device, { residentWeights: true });
    let note = "";
    const started = performance.now();
    let outcome;
    try {
      await trunkGpu.run({
        tokens, sequences: rows, templates: 4, targetFeat, features: batch.features,
        msaRows: msa, deletionMatrix, msaMask,
        bondMatrix: batch.bondMatrix, pairMask, seqMask,
        previousPair: new Float32Array(tokens * tokens * 128),
        previousSingle: new Float32Array(tokens * 384),
      }, weights.trunk, DIALECT, { onStatus: (text) => { note = text; } });
      outcome = "finished";
    } catch (error) {
      outcome = error.name === "GpuMemoryBudgetError" ? "REFUSED" : `threw ${error.name}`;
      note = error.message;
    }
    const elapsed = Math.round(performance.now() - started);
    const snapshot = memorySnapshot(device);
    results.push({
      budgetMiB: budgetMiB === 0 ? "none" : budgetMiB,
      outcome,
      seconds: Number((elapsed / 1000).toFixed(1)),
      peakMiB: Number((snapshot.peakBytes / MiB).toFixed(0)),
      ...(note.startsWith("uploading weights")
        ? { degraded: "gave up weight residency" } : {}),
      ...(outcome === "REFUSED" ? { why: note.slice(0, 130) } : {}),
    });
  }
  setMemoryBudget(device, undefined);

  return {
    tokens, msaRows: rows, blocks,
    navigatorDeviceMemoryGiB: reported,
    pageBudgetMiB: Number((pageBudget / MiB).toFixed(0)),
    results,
  };
}
