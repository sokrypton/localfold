/**
 * Does one program holding several small kernels (an "uber-shader", the kernel chosen by a
 * uniform) link faster than the kernels as separate programs? Safari 16.3 links each program in
 * ~19 ms on the page's thread whatever its size, so it would if that cost is per program. Small
 * elementwise kernels in the fold's own style, linked both ways in fresh contexts.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-uber.js [--kernels=8] [--reps=3]
 *
 * 🔴 EVERY SALT IS FRESH PER RUN, BECAUSE A REPEATED ONE IS A CACHE HIT AND NOT A LINK. The
 * salts were `rep * 0.001`, so the second run of this probe asked for shaders the first run had
 * already built - and Metal keeps its compiled shaders per USER, outside the fresh Chrome profile
 * `gpu-chrome.mjs` makes each run. Measured here: 20 kernels linked as one program read 50 ms on
 * the first run of the probe and 2 ms on the next, with nothing changed. A random base now, so a
 * run always asks for code no cache has seen; see probe-webgl-programs.js --cold.
 */
import { createWebGL2Context, linkProgram } from "../../webgl/runtime/context.js";
import { ONE_TARGET, PRELUDE } from "../../webgl/runtime/execution.js";

const body = (k, salt) => `  vec4 a${k} = fetch4(aTex, texel) * ${(1 + k * 0.125 + salt).toFixed(4)};
  vec4 b${k} = fetch4(bTex, texel / ${k + 2});
  return a${k} + b${k} * exp(-a${k}) + vec4(fetch1(cTex, texel % ${k + 7}));`;

const declarations = `uniform sampler2D aTex;
uniform sampler2D bTex;
uniform sampler2D cTex;
uniform int ELEMENTS;`;

const single = (k, salt) => `${PRELUDE}${ONE_TARGET}
${declarations}
vec4 kernel(int texel) {
${body(k, salt)}
}
void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = kernel(texel);
}`;

const uber = (n, salt) => `${PRELUDE}${ONE_TARGET}
${declarations}
uniform int KERNEL;
${Array.from({ length: n }, (_, k) => `vec4 kernel${k}(int texel) {\n${body(k, salt)}\n}`).join("\n")}
void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
${Array.from({ length: n }, (_, k) => `  ${k ? "else " : ""}if (KERNEL == ${k}) result = kernel${k}(texel);`).join("\n")}
  else result = vec4(0.0);
}`;

export async function main(_device, args = []) {
  const option = (name, fallback) => Number((args.find((a) => a.startsWith(`--${name}=`)) ?? `--${name}=${fallback}`).split("=")[1]);
  const n = option("kernels", 8);
  const reps = option("reps", 3);
  const separate = [];
  const merged = [];
  // The base every rep's salt is offset from: fresh per run, so no cache can answer.
  const base = Math.random() * 1e3;
  for (let rep = 0; rep < reps; rep += 1) {
    let gl = createWebGL2Context();
    let t = performance.now();
    for (let k = 0; k < n; k += 1) linkProgram(gl, single(k, base + rep * 0.001));
    separate.push(Math.round(performance.now() - t));
    gl = createWebGL2Context();
    t = performance.now();
    linkProgram(gl, uber(n, base + rep * 0.001 + 0.0005));
    merged.push(Math.round(performance.now() - t));
  }
  return { kernels: n, separateMs: separate, uberMs: merged };
}
