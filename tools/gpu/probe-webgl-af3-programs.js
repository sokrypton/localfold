/**
 * Which of the WebGL2 AF3 fold's programs depend on the sequence length: the programs a fold
 * draws with (`foldPrograms`, a dry run, no draws) at several lengths, and the kernels whose
 * source differs between them - each of which a page compiles again for every new length.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-af3-programs.js [--lengths=60,100,101]
 *
 * `compilesPerLength`: with length-generic programs (webgl/runtime/length-generic.js) learned as
 * the page's first fold learns them (its length and 67 longer), how many programs each length
 * in turn actually compiles (`Execution.prepare`).
 */
import { openAf3Store } from "../../shared/af3/weights/weights.js";
import { foldWeights } from "../../shared/af3/weights/diffusion-weights.js";
import { af3BatchFromA3m } from "../../shared/af3/featurise/batch.js";
import { createWebGL2Context } from "../../webgl/runtime/context.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { foldPrograms, uploadAf3Weights } from "../../webgl/af3/model/fold.js";

const SEQUENCE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVMKVLLTGAGSRAAELVRKAGH".repeat(4);

export async function main(_device, args) {
  const option = (name, fallback) => (args.find((a) => a.startsWith(`--${name}=`)) ?? `--${name}=${fallback}`).split("=")[1];
  const lengths = option("lengths", "60,100,101").split(",").map(Number);
  const weights = await foldWeights(await openAf3Store(`${option("bundle", "/model-openbind0-int5")}/manifest.json`), { blocks: 1, msaBlocks: 1 });
  const execution = new Execution(createWebGL2Context(), { halfWeights: true, halfActivations: true, int5Weights: true });
  const uploaded = uploadAf3Weights(execution, weights);
  const runs = [];
  for (const length of lengths) {
    const { batch } = af3BatchFromA3m(SEQUENCE.slice(0, length), null, {});
    const programs = await foldPrograms(execution, {
      batch, weights: uploaded, hostWeights: weights,
      msa: { codes: Float32Array.from(batch.msa), deletions: Float32Array.from(batch.deletionMatrix), mask: Float32Array.from(batch.msaMask), sequences: 1 },
      steps: 1, seed: 1,
    });
    runs.push({ length, programs });
  }
  const family = (label) => label.split(":")[0];
  const [first, ...rest] = runs;
  const comparisons = rest.map(({ length, programs }) => {
    const shared = [...programs.keys()].filter((source) => first.programs.has(source));
    const fresh = [...programs].filter(([source]) => !first.programs.has(source));
    const byFamily = {};
    for (const [, label] of fresh) byFamily[family(label)] = (byFamily[family(label)] ?? 0) + 1;
    return {
      against: first.length, length, programs: programs.size, shared: shared.length, recompiled: fresh.length,
      recompiledByKernel: Object.fromEntries(Object.entries(byFamily).sort((a, b) => b[1] - a[1])),
    };
  });
  const sizeIndependent = {};
  for (const [, label] of [...first.programs].filter(([source]) => runs.every((r) => r.programs.has(source)))) {
    sizeIndependent[family(label)] = (sizeIndependent[family(label)] ?? 0) + 1;
  }
  // The page's way: learn from the first length and 67 longer, then compile length after length.
  const callFor = (length) => {
    const { batch } = af3BatchFromA3m(SEQUENCE.slice(0, length), null, {});
    return { batch, weights: uploaded, hostWeights: weights, steps: 1, seed: 1,
      msa: { codes: Float32Array.from(batch.msa), deletions: Float32Array.from(batch.deletionMatrix), mask: Float32Array.from(batch.msaMask), sequences: 1 } };
  };
  const longer = await foldPrograms(execution, callFor(first.length + 67));
  execution.learnLengths(first.programs.keys(), longer.keys());
  const compilesPerLength = [];
  let previous = null;
  for (const { length, programs } of runs) {
    // ...and each length against the one before, as prepareFoldPrograms learns on later folds.
    if (previous !== null) execution.learnLengths(previous.keys(), programs.keys());
    previous = programs;
    const before = execution.compiles;
    const had = new Set(execution.programs.keys());
    const t = performance.now();
    await execution.prepare(programs.keys());
    const added = [...execution.programs.keys()].filter((key) => !had.has(key));
    compilesPerLength.push({ length, compiled: execution.compiles - before, ms: Math.round(performance.now() - t),
      refusedGroups: execution.genericRefused.size,
      specialised: added.map((key) => programs.get(key)).filter(Boolean),
      generic: added.filter((key) => !programs.has(key)).length });
  }
  return { lengths, programs: runs.map((r) => r.programs.size), comparisons: comparisons.map(({ recompiledByKernel, ...rest }) => rest),
    sharedByKernel: sizeIndependent, compilesPerLength };
}
