/**
 * How much of AlphaFold 2's template stage is compiling its four pipelines?
 *
 * 🔴 WRITTEN BECAUSE A FOLD CANNOT SEPARATE THEM. `fold-af2.js`'s `phases` puts
 * the stage at 180 ms of a 1042 ms fold at 150 residues and 210 of 3374 at 400
 * - nearly flat in a length its kernels are quadratic in - which says fixed
 * cost, and the obvious fixed cost is the compile. "Obvious" is not measured:
 * this times the four `pipelines.get` calls on a cold cache, and then again on
 * a warm one, with nothing else in the process.
 */
import { WebGpuExecution } from "../../src/runtime/execution.js";
import { QueryOnlyTemplateGpu } from "../../src/af2/evoformer/template.js";

export async function main(device) {
  const execution = new WebGpuExecution(device);
  const cold = performance.now();
  await QueryOnlyTemplateGpu.warm(execution);
  const coldMs = performance.now() - cold;
  const warm = performance.now();
  await QueryOnlyTemplateGpu.warm(execution);
  const warmMs = performance.now() - warm;
  return {
    coldCompileMs: Number(coldMs.toFixed(1)),
    warmLookupMs: Number(warmMs.toFixed(1)),
    note: "the stage's whole cost at 150 residues is about 180 ms",
  };
}
