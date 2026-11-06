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
import { VERTEX_SHADER, createWebGL2Context, linkProgram } from "../../webgl/runtime/context.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { foldPrograms, uploadAf3Weights } from "../../webgl/af3/model/fold.js";

export async function main(_device, args = []) {
  const option = (name, fallback) => (args.find((a) => a.startsWith(`--${name}=`)) ?? `--${name}=${fallback}`).split("=")[1];
  const length = Number(option("length", "100"));
  const mark = (stage) => { globalThis.__progress = `${stage} at ${Math.round(performance.now())} ms`; };
  mark("loading");
  const weights = await foldWeights(await openAf3Store(`${option("bundle", "/model-openbind0-int5")}/manifest.json`), { blocks: 1, msaBlocks: 1 });
  mark("loaded");
  const execution = new Execution(createWebGL2Context(), { halfWeights: true, halfActivations: true, int5Weights: true });
  const uploaded = uploadAf3Weights(execution, weights);
  const { batch } = af3BatchFromA3m("PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVMKVLLTGAGSRAAELVRKAGH".repeat(8).slice(0, length), null, {});
  mark("uploaded");
  let t = performance.now();
  const { programs, groups: sliceGroups } = await foldPrograms(execution, {
    batch, weights: uploaded, hostWeights: weights, steps: 1, seed: 1,
    msa: { codes: Float32Array.from(batch.msa), deletions: Float32Array.from(batch.deletionMatrix), mask: Float32Array.from(batch.msaMask), sequences: 1 },
  });
  const dryRunMs = Math.round(performance.now() - t);
  mark("dry run done");
  const gl = execution.gl;
  const timed = [];
  // `--shared-vs=1`: one compiled vertex shader attached to every program, instead of one each.
  let vertex = null;
  if (option("shared-vs", "0") === "1") {
    vertex = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertex, VERTEX_SHADER);
    gl.compileShader(vertex);
    gl.getShaderParameter(vertex, gl.COMPILE_STATUS);
  }
  for (const [source, label] of programs) {
    t = performance.now();
    if (vertex === null) {
      linkProgram(gl, source);
    } else {
      const program = gl.createProgram();
      const fragment = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(fragment, source);
      gl.compileShader(fragment);
      gl.attachShader(program, vertex);
      gl.attachShader(program, fragment);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    }
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
    renderer: gl.getParameter(gl.RENDERER), programs: timed.length, groups: sliceGroups.map((g) => g.length),
    firstGroup: sliceGroups[0].map((source) => programs.get(source)), dryRunMs, linkTotalMs: Math.round(total),
    top: timed.slice(0, Number(option("top", "25"))).map(({ label, ms, chars }) => `${ms.toFixed(1)} ms ${chars}c ${label}`),
    families: Object.entries(byFamily).sort((a, b) => b[1] - a[1]).slice(0, 15).map(([f, ms]) => `${Math.round(ms)} ms ${f}`),
  };
}
