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
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { buildTargetFeat, DIALECT } from "../../src/af3/fold.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";
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
  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"),
                                   { fetchImplementation: fetch });
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

  const trunkGpu = new Af3TrunkGpu(device);
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
  }

  // 🔴 THE FIRST PASS COMPILES EVERY PIPELINE. The median of the rest is what a
  // recycle actually costs.
  const after = perPass.slice(1);
  const median = (pick) => {
    const values = after.map(pick).sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)];
  };
  return {
    tokens, msaRows: rows, blocks, perPass,
    steady: Object.fromEntries(Object.keys(perPass[0])
      .filter((key) => key !== "pass")
      .map((key) => [key, median((row) => row[key])])),
  };
}
