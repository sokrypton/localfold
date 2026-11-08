/**
 * Which of the WebGL2 AF3 fold's programs depend on the sequence length: the programs a fold
 * draws with (`foldPrograms`, a dry run, no draws) at several lengths, and the kernels whose
 * source differs between them - each of which a page compiles again for every new length.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-af3-programs.js [--lengths=60,100,101]
 *
 * `compilesPerLength`: how many programs each length in turn actually compiles
 * (`Execution.prepare`), with the kernels' lengths as uniforms (`sized`): the first length
 * compiles the fold's programs, a later one what it alone reaches (a size threshold's kernel).
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
    const { programs } = await foldPrograms(execution, {
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
  const compilesPerLength = [];
  for (const { length, programs } of runs) {
    const before = execution.compiles;
    const had = new Set(execution.programs.keys());
    const t = performance.now();
    await execution.prepare(programs.keys());
    const added = [...execution.programs.keys()].filter((key) => !had.has(key));
    compilesPerLength.push({ length, compiled: execution.compiles - before, ms: Math.round(performance.now() - t),
      kernels: added.map((key) => programs.get(key)) });
  }
  return { lengths, programs: runs.map((r) => r.programs.size), comparisons: comparisons.map(({ recompiledByKernel, ...rest }) => rest),
    sharedByKernel: sizeIndependent, compilesPerLength };
}
