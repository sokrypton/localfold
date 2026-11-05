/**
 * Does `mediump` reach this device's HALF arithmetic, and is it faster?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-precision.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-precision.js --rows=4624 --in=256 --out=512
 *
 * 🔴 THIS IS THE ONE LEVER THE WGSL BACKEND HAS THAT THIS ONE HAS NOT BEEN
 * ASKED FOR. webgl/README.md's own floor argument says the fragment path is
 * capped at 21.33 flops a `fetch4` by the 32-float output budget, and that the
 * WGSL side is past it by workgroup memory AND by f16 ARITHMETIC - where this
 * backend's RGBA16F is storage only. But GLSL ES 300 has PRECISION QUALIFIERS,
 * and a translator over Metal may map `mediump float` to `half`, which is real
 * 16-bit arithmetic with half the register pressure. Nothing here has ever
 * asked: `PRELUDE` pins `highp` on float, int and sampler, so every kernel in
 * this backend is f32 whatever the storage.
 *
 * 🔴 AND `relRms` IS THE TELL, NOT THE TIME. A translator is free to PROMOTE
 * `mediump` to `highp` - the spec's minimum range is a floor, not a ceiling -
 * and then the two arms are the same kernel: identical time and relRms EXACTLY
 * zero. A ratio near 1.00 means nothing on its own; a ratio near 1.00 WITH a
 * nonzero relRms means half arithmetic that is not faster, and a zero relRms
 * means the qualifier was ignored and this whole question is moot on this
 * device. Read them together or not at all.
 *
 * The shape is the shipped eight-row blocked projection's: eight rows times
 * four outputs into eight render targets, which is the geometry the floor
 * argument is about. A ring of distinct outputs, because an Apple GPU discards
 * the earlier of two identical full-viewport draws into one target - see
 * `bench-webgl-linear.js`.
 */
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const fill = (count, seed) => {
  const values = new Float32Array(count);
  let state = seed | 0;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    // Small, so a half accumulator over a long k does not simply overflow -
    // this probe is about the RATE, and an infinity would hide it.
    values[i] = ((state >>> 8) / 0x7fffff - 0.5) * 0.25;
  }
  return values;
};

/**
 * The blocked projection, with the precision of everything inside it switchable.
 *
 * Its own prelude rather than `PRELUDE`, which pins highp - that is the whole
 * variable under test.
 */
const source = (precision, inChannels, outChannels, rows) => {
  const targets = 8;
  const component = ["x", "y", "z", "w"];
  return `#version 300 es
precision ${precision} float;
precision ${precision} sampler2D;
precision highp int;
const int W = 2048;
const int IN4 = ${inChannels / 4};
const int OUT4 = ${outChannels / 4};
const int ROWS = ${rows};
const int RQ = ${rows / targets};
${Array.from({ length: targets }, (_, t) =>
    `layout(location = ${t}) out vec4 result${t};`).join("\n")}
uniform sampler2D xTex;
uniform sampler2D wTex;

vec4 fetch4(sampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % W, texel / W), 0);
}

void main() {
  int texel = int(gl_FragCoord.y) * W + int(gl_FragCoord.x);
  if (texel >= RQ * OUT4) {
${Array.from({ length: targets }, (_, t) => `    result${t} = vec4(0.0);`).join("\n")}
    return;
  }
  int group = texel % OUT4;
  int block = texel / OUT4;
  int row = block * ${targets};
${Array.from({ length: targets }, (_, t) =>
    `  vec4 acc${t} = vec4(0.0);`).join("\n")}
  for (int k4 = 0; k4 < IN4; ++k4) {
${Array.from({ length: targets }, (_, t) =>
    `    vec4 x${t} = fetch4(xTex, (row + ${t}) * IN4 + k4);`).join("\n")}
${[0, 1, 2, 3].map((o) => `    vec4 w${o} = fetch4(wTex, (group * 4 + ${o}) * IN4 + k4);`).join("\n")}
${Array.from({ length: targets }, (_, t) => [0, 1, 2, 3].map((o) =>
    `    acc${t}.${component[o]} += dot(x${t}, w${o});`).join(" ")).join("\n")}
  }
${Array.from({ length: targets }, (_, t) => `  result${t} = acc${t};`).join("\n")}
}`;
};

export async function main(device, args) {
  const iterations = Number(option(args, "iterations", "8"));
  const rows = Number(option(args, "rows", "4624"));
  const inChannels = Number(option(args, "in", "256"));
  const outChannels = Number(option(args, "out", "512"));
  const gl = createWebGL2Context();
  const execution = new Execution(gl, { checkErrors: false });

  const x = execution.upload(fill(rows * inChannels, 7), "probe.x");
  const weight = execution.upload(fill(inChannels * outChannels, 11), "probe.w");
  const planeTexels = (rows / 8) * (outChannels / 4);
  const flops = 2 * rows * inChannels * outChannels;
  // Eight `fetch4` of x and four of the weight, per k-step per fragment.
  const fetches = planeTexels * (inChannels / 4) * 12;

  /** @type {Record<string, Float32Array>} */
  const answers = {};
  /** @type {Record<string, number>} */
  const timings = {};
  for (const precision of ["highp", "mediump"]) {
    const ring = Array.from({ length: iterations }, () =>
      Array.from({ length: 8 }, () => execution.allocate(planeTexels * 4, "probe.out")));
    const dispatch = (outputs) => execution.dispatch({
      source: source(precision, inChannels, outChannels, rows),
      outputs,
      inputs: { xTex: x, wTex: weight },
      label: `probe.project:${precision}`,
    });
    for (let i = 0; i < 2; i += 1) {
      dispatch(ring[i % ring.length]);
      execution.read(ring[i % ring.length][0]);
    }
    const start = performance.now();
    for (let i = 0; i < iterations; i += 1) dispatch(ring[i]);
    answers[precision] = Float32Array.from(execution.read(ring[iterations - 1][0]));
    timings[precision] = (performance.now() - start) / iterations;
    for (const set of ring) for (const out of set) execution.release(out);
  }
  execution.release(x);
  execution.release(weight);

  const relRms = relativeRms(answers.highp, answers.mediump);
  const ratio = timings.highp / timings.mediump;
  const renderer = gl.getParameter(gl.RENDERER);
  execution.destroy();
  // 1e-7 is comfortably below an f32 reordering and far below f16's ~1e-3, so
  // it separates "the qualifier did nothing" from "the arithmetic narrowed".
  const ignored = relRms < 1e-7;
  return {
    renderer,
    shape: `${rows}x${inChannels}->${outChannels}`,
    iterations,
    highpMs: Number(timings.highp.toFixed(2)),
    mediumpMs: Number(timings.mediump.toFixed(2)),
    speedup: Number(ratio.toFixed(3)),
    highpGflops: Number((flops / timings.highp / 1e6).toFixed(1)),
    mediumpGflops: Number((flops / timings.mediump / 1e6).toFixed(1)),
    highpGfetches: Number((fetches / timings.highp / 1e6).toFixed(2)),
    flopsPerFetch: Number((flops / fetches).toFixed(1)),
    relRms: Number(relRms.toExponential(2)),
    qualifierIgnored: ignored,
    verdict: ignored
      ? `mediump is PROMOTED on this device: relRms ${relRms.toExponential(2)} is exactly an `
        + `f32 match and the ratio is ${ratio.toFixed(3)}, so the two arms are one kernel and `
        + `half arithmetic is not reachable from GLSL ES here`
      : `mediump is REAL here: relRms ${relRms.toExponential(2)} and ${ratio.toFixed(3)}x `
        + `(${timings.highp.toFixed(2)} ms against ${timings.mediump.toFixed(2)})`,
  };
}
