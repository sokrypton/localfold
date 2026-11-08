/**
 * Can the WebGL2 AF3 fold's shader compile hide behind its weight upload? The programs found by
 * a dry run on a VIRTUAL upload (shapes, no textures - so before any real upload), then: the
 * real upload alone; the real upload with those programs compiled between its pieces (each
 * piece's GPU wait); and the programs alone. Hidden means upload-with-compile < upload + compile.
 * Runs under tools/gpu-chrome.mjs and, through a page calling `main`, in Safari.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-upload-compile.js [--length=100]
 */
import { openAf3Store } from "../../shared/af3/weights/weights.js";
import { foldWeights } from "../../shared/af3/weights/diffusion-weights.js";
import { af3BatchFromA3m } from "../../shared/af3/featurise/batch.js";
import { createWebGL2Context } from "../../webgl/runtime/context.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { drainAsync, gpuIdle } from "../../webgl/runtime/steps.js";
import { foldPrograms, releaseAf3Weights, uploadAf3Weights, uploadAf3WeightsSteps } from "../../webgl/af3/model/fold.js";

export async function main(_device, args = []) {
  const option = (name, fallback) => (args.find((a) => a.startsWith(`--${name}=`)) ?? `--${name}=${fallback}`).split("=")[1];
  const mark = (stage) => { globalThis.__progress = stage; };
  mark("loading");
  const weights = await foldWeights(await openAf3Store(`${option("bundle", "/model-openbind0-int5")}/manifest.json`));
  const { batch } = af3BatchFromA3m("PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVMKVLLTGAGSRAAELVRKAGH".repeat(4).slice(0, Number(option("length", "100"))), null, {});
  const msa = { codes: Float32Array.from(batch.msa), deletions: Float32Array.from(batch.deletionMatrix), mask: Float32Array.from(batch.msaMask), sequences: 1 };
  const make = () => new Execution(createWebGL2Context(), { halfWeights: true, halfActivations: true, int5Weights: true });
  const ms = (t) => Math.round(performance.now() - t);

  // 1. the programs, from a virtual upload and a dry run - no texture made
  mark("virtual");
  const probe = make();
  let t = performance.now();
  probe.storage.virtual = true;
  const virtualTree = uploadAf3Weights(probe, weights);
  probe.storage.virtual = false;
  const virtualUploadMs = ms(t);
  t = performance.now();
  const { groups } = await foldPrograms(probe, { batch, weights: virtualTree, hostWeights: weights, steps: 1, seed: 1, msa });
  const dryRunMs = ms(t);
  const sources = groups.flat();

  // 2. the real upload alone
  mark("upload alone");
  const a = make();
  t = performance.now();
  const uploadedA = await drainAsync(uploadAf3WeightsSteps(a, weights), () => gpuIdle(a.gl));
  const uploadMs = ms(t);
  releaseAf3Weights(a, uploadedA);

  // 3. the real upload, the programs compiled in its pieces' waits (and one a piece regardless)
  mark("upload with compile");
  const b = make();
  let next = 0;
  const compileNext = () => {
    if (next >= sources.length) return false;
    b.program(sources[next]);
    next += 1;
    return true;
  };
  t = performance.now();
  const uploadedB = await drainAsync(uploadAf3WeightsSteps(b, weights), async () => {
    compileNext();
    await gpuIdle(b.gl, compileNext);
  });
  const interleavedMs = ms(t);
  const compiledDuring = next;
  t = performance.now();
  while (compileNext()) { /* the rest */ }
  const restMs = ms(t);
  releaseAf3Weights(b, uploadedB);

  // 4. the programs alone, in a fresh context
  mark("compile alone");
  const c = make();
  t = performance.now();
  for (const source of sources) c.program(source);
  const compileMs = ms(t);
  mark("done");
  return {
    renderer: c.gl.getParameter(c.gl.RENDERER), programs: sources.length, firstSlice: groups[0].length,
    virtualUploadMs, dryRunMs, uploadMs, compileMs,
    interleaved: { uploadMs: interleavedMs, compiledDuring, restMs },
    hiddenMs: uploadMs + compileMs - interleavedMs - restMs,
  };
}
