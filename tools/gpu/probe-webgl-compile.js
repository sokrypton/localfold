/**
 * What the WebGL2 AF3 fold's shader compile costs, program by program: the programs a fold at
 * `--length` draws with (`foldPrograms`, a dry run), each compiled and linked on its own and
 * timed, slowest first, with the dry run's own time. Runs under tools/gpu-chrome.mjs and, through
 * a page that calls `main(null, args)`, in Safari.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-compile.js [--length=100] [--top=25]
 */
import { openAf3Store } from "../../shared/af3/weights/weights.js";
import { foldWeights } from "../../shared/af3/weights/diffusion-weights.js";
import { af3BatchFromA3m } from "../../shared/af3/featurise/batch.js";
import { createWebGL2Context, linkProgram } from "../../webgl/runtime/context.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { foldPrograms, uploadAf3Weights } from "../../webgl/af3/model/fold.js";

export async function main(_device, args = []) {
  const option = (name, fallback) => (args.find((a) => a.startsWith(`--${name}=`)) ?? `--${name}=${fallback}`).split("=")[1];
  const length = Number(option("length", "100"));
  const weights = await foldWeights(await openAf3Store(`${option("bundle", "/model-openbind0-int5")}/manifest.json`), { blocks: 1, msaBlocks: 1 });
  const execution = new Execution(createWebGL2Context(), { halfWeights: true, halfActivations: true, int5Weights: true });
  const uploaded = uploadAf3Weights(execution, weights);
  const { batch } = af3BatchFromA3m("PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVMKVLLTGAGSRAAELVRKAGH".repeat(8).slice(0, length), null, {});
  let t = performance.now();
  const { programs, groups: sliceGroups } = await foldPrograms(execution, {
    batch, weights: uploaded, hostWeights: weights, steps: 1, seed: 1,
    msa: { codes: Float32Array.from(batch.msa), deletions: Float32Array.from(batch.deletionMatrix), mask: Float32Array.from(batch.msaMask), sequences: 1 },
  });
  const dryRunMs = Math.round(performance.now() - t);
  const gl = execution.gl;
  const timed = [];
  for (const [source, label] of programs) {
    t = performance.now();
    linkProgram(gl, source);
    timed.push({ label, ms: performance.now() - t, chars: source.length });
  }
  timed.sort((a, b) => b.ms - a.ms);
  const total = timed.reduce((a, x) => a + x.ms, 0);
  const byFamily = {};
  for (const { label, ms } of timed) { const f = label.split(":")[0]; byFamily[f] = (byFamily[f] ?? 0) + ms; }
  // How many programs would remain if every module-scope integer constant (the widths too) were a
  // uniform: the sources grouped with those values blanked.
  const skeleton = (source) => source.replace(/^const int (\w+) = -?\d+;$/gm, "const int $1 = ?;");
  const groups = new Map();
  for (const [source, label] of programs) {
    const key = skeleton(source);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(label.split(":")[0]);
  }
  const merged = {};
  for (const labels of groups.values()) if (labels.length > 1) merged[labels[0]] = (merged[labels[0]] ?? 0) + labels.length - 1;
  return {
    withWidthsAsUniforms: groups.size, mergedAway: merged,
    perFamily: Object.entries(timed.reduce((a, { label }) => { const f = label.split(":")[0]; a[f] = (a[f] ?? 0) + 1; return a; }, {}))
      .filter(([, n]) => n > 1).sort((a, b) => b[1] - a[1]).map(([f, n]) => `${n} ${f}`),
    labelsOf: Object.fromEntries((option("labels", "") ? option("labels", "").split(",") : []).map((f) => [f, timed.filter(({ label }) => label.split(":")[0] === f).map(({ label }) => label)])),
    renderer: gl.getParameter(gl.RENDERER), programs: timed.length, groups: sliceGroups.map((g) => g.length), dryRunMs, linkTotalMs: Math.round(total),
    top: timed.slice(0, Number(option("top", "25"))).map(({ label, ms, chars }) => `${ms.toFixed(1)} ms ${chars}c ${label}`),
    families: Object.entries(byFamily).sort((a, b) => b[1] - a[1]).slice(0, 15).map(([f, ms]) => `${Math.round(ms)} ms ${f}`),
  };
}
