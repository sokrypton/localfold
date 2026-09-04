/**
 * Where a trunk pass's time goes.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-trunk.js --tokens=59 --msa=32
 *
 * 🔴 THE TRUNK IS THE OTHER HALF OF A FOLD AND IS RUN ONCE PER RECYCLE. AF3's
 * default is ten of them, so at a second a pass it dwarfs a sampler that now
 * costs 176 ms a call - and none of the diffusion side's optimisations have
 * been applied here.
 *
 * 🔴 THE MSA IS SYNTHETIC AND ITS DEPTH IS A DIAL. The MSA stack's cost is
 * linear in rows and the pairformer's is not, so a bench at depth 1 measures a
 * different machine from the one a real fold runs. --msa sets it.
 *
 * WHAT IT FOUND, on a 59-residue chain at 32 rows, after the block weights were
 * made resident:
 *
 *     pairformer 577   msa-stack 89   template 13   embedder 11   distogram 8
 *
 * and inside the pairformer, by disabling groups of passes:
 *
 *     pair-transition 129   grid.* 125   tri.* 105   single.* 41
 *
 * 🔴 AND THERE IS NO OVERHEAD LEFT TO RECLAIM HERE. Making #encodeBlock encode
 * nothing at all - the block loop, the deferred validation, the per-block
 * fences and the weight setup, and no dispatches - leaves 7 ms of the 577. The
 * diffusion side's wins were host-side waste of exactly that kind; this stack
 * has none, so anything further has to come out of the kernels themselves, and
 * src/triangle/shaders.js already records several tiling attempts that lost.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { buildTargetFeat, DIALECT } from "../../src/af3/fold.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";
import { profileDevice } from "./profile.js";
import { memorySnapshot, setMemoryBudget } from "../../src/runtime/device-memory.js";
import { openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { targetFeatureWeights } from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const ALPHABET = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "59"));
  const rows = Number(option(args, "msa", "32"));
  const passes = Number(option(args, "passes", "3"));
  const blocks = Number(option(args, "blocks", "48"));
  const sequence = Array.from({ length: tokens },
    (_, index) => ALPHABET[index % ALPHABET.length]).join("");

  const batch = featuriseProtein(sequence, {});
  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"));
  const weights = { trunk: await trunkWeights(store, blocks, 4),
                    targetFeat: await targetFeatureWeights(store) };
  const targetFeat = await buildTargetFeat(batch, weights.targetFeat, device);

  // A synthetic alignment of the requested depth: the query, then rows that
  // differ from it. Contents do not change the cost, only the shape does.
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

  // --profile times every compute pass by label; see tools/gpu/profile.js.
  const profile = args.includes("--profile") ? profileDevice(device) : null;
  // --no-resident uploads each block's weights per pass instead of keeping
  // them on the device, which is the 562 MiB half of what a fold holds.
  // --budget=<MiB> puts a ceiling on the device instead and lets the stack
  // discover it, which is how the page behaves on a machine too small for the
  // resident path - and the only way that fallback gets exercised here.
  const residentWeights = !args.includes("--no-resident");
  const budgetMiB = Number(option(args, "budget", "0"));
  if (budgetMiB > 0) setMemoryBudget(device, budgetMiB * 1024 * 1024);
  const trunkGpu = new Af3TrunkGpu(device, { residentWeights });
  let previousPair = new Float32Array(tokens * tokens * 128);
  let previousSingle = new Float32Array(tokens * 384);
  const perPass = [];
  for (let pass = 0; pass < passes; pass += 1) {
    const timings = {};
    const started = performance.now();
    const trunk = await trunkGpu.run({
      tokens, sequences: rows, templates: 4, targetFeat, features: batch.features,
      msaRows: msa, deletionMatrix, msaMask,
      bondMatrix: batch.bondMatrix, pairMask, seqMask, previousPair, previousSingle,
    }, weights.trunk, DIALECT, { onStage: (name, ms) => { timings[name] = Math.round(ms); } });
    previousPair = trunk.pair;
    previousSingle = trunk.single;
    perPass.push({ pass, whole: Math.round(performance.now() - started), ...timings });
    // ...the last pass only, so pipeline compilation is not in the numbers.
    if (profile !== null && pass < passes - 1) profile.reset();
  }
  const passes_ = profile === null ? undefined : await profile.report();
  profile?.restore();

  // 🔴 THE FIRST PASS COMPILES EVERY PIPELINE. The median of the rest is what a
  // recycle actually costs.
  const after = perPass.slice(1);
  const median = (pick) => {
    const values = after.map(pick).sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)];
  };
  return {
    pairformerSplit: trunkGpu.lastPairformerSplit,
    tokens, msaRows: rows, blocks, residentWeights, budgetMiB, perPass,
    deviceMemory: memorySnapshot(device),
    ...(passes_ === undefined ? {} : { gpuPasses: passes_.slice(0, 18),
      gpuTotalMs: Number(passes_.reduce((t, e) => t + e.ms, 0).toFixed(1)),
      gpuLabels: passes_.length,
      gpuDispatches: passes_.reduce((t, e) => t + e.passes, 0) }),
    steady: Object.fromEntries(Object.keys(perPass[0])
      .filter((key) => key !== "pass")
      .map((key) => [key, median((row) => row[key])])),
  };
}
