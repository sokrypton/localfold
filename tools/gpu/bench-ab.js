/**
 * Interleaved A/B for the AF3 pairformer, in one process.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-ab.js --skip=single
 *
 * 🔴 TWO NUMBERS FROM TWO INVOCATIONS CANNOT BE COMPARED ON THIS MACHINE. It
 * drifts by up to 3.2x between runs, so a difference smaller than the drift is
 * noise wearing a result's clothes - and every per-pass figure taken that way
 * has to be thrown out. A and B alternate here inside a single process, and the
 * medians are what is reported.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";
import { buildTargetFeat, DIALECT } from "../../src/af3/fold.js";
import { openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { targetFeatureWeights } from "../../src/af3/diffusion-weights.js";

const LENGTH = 59;
const FLAGS = ["TRICONTRACT", "TRIPROJECT", "TRIOUT", "TRINORM", "GNORM", "GBIAS",
               "GPROJ", "GATT", "GOUT", "TRANS", "ADD", "SINGLE"];

export async function main(device, args) {
  const which = (args.find((a) => a.startsWith("--skip=")) ?? "--skip=single").slice(7)
    .split(",").map((name) => name.trim().toUpperCase());
  const rounds = Number((args.find((a) => a.startsWith("--rounds=")) ?? "--rounds=7").slice(9));

  const store = await openAf3Store("/model-af3-int5/manifest.json");
  const weights = await trunkWeights(store, 48, 4);
  const batch = featuriseProtein("A".repeat(LENGTH));
  const targetFeat = await buildTargetFeat(batch, await targetFeatureWeights(store), device);
  const input = {
    tokens: LENGTH, sequences: 1, templates: 4, targetFeat, features: batch.features,
    msaRows: batch.msa.subarray(0, LENGTH),
    deletionMatrix: batch.deletionMatrix.subarray(0, LENGTH),
    msaMask: batch.msaMask.subarray(0, LENGTH),
    pairMask: new Float32Array(LENGTH * LENGTH).fill(1), seqMask: batch.seqMask,
    previousPair: new Float32Array(LENGTH * LENGTH * 128),
    previousSingle: new Float32Array(LENGTH * 384),
  };
  const once = async (skip) => {
    for (const flag of FLAGS) globalThis[`__SKIP_${flag}`] = skip && which.includes(flag);
    let pairformer = 0;
    await new Af3TrunkGpu(device).run(input, weights, DIALECT,
      { onStage: (name, ms) => { if (name === "pairformer") pairformer = ms; } });
    return pairformer;
  };
  await once(false);                                   // compile everything first

  const a = [];
  const b = [];
  for (let round = 0; round < rounds; round += 1) {
    a.push(await once(false));
    b.push(await once(true));
  }
  const median = (values) => [...values].sort((x, y) => x - y)[Math.floor(values.length / 2)];
  const full = median(a);
  const without = median(b);
  console.log(`${LENGTH} tokens, ${rounds} interleaved rounds, medians`);
  console.log(`  with     ${full.toFixed(0)} ms   [${a.map((v) => v.toFixed(0)).join(" ")}]`);
  console.log(`  without  ${without.toFixed(0)} ms   [${b.map((v) => v.toFixed(0)).join(" ")}]`);
  console.log(`  ${which.join(",")} costs ${(full - without).toFixed(0)} ms`
    + `   (${((full - without) / full * 100).toFixed(0)}% of the pairformer)`);
  return { full, without, cost: full - without, skipped: which };
}
