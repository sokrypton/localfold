/**
 * Whether reading a projection's x as EIGHT halves a texel (RGBA32UI, unpackHalf2x16) beats four (RGBA16F).
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-packed-x.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-packed-x.js --shapes=3481x128x256,3481x128x128
 *
 * 🔴 RGBA16F HALVES THE BYTES AND NOT THE FETCHES. webgl/README.md declined the sibling port's eight-a-texel packing
 * because RGBA16F "holds the same four values a texel" with no kernel change - true of the BYTES, and the same file
 * says these kernels are bound by fetch ISSUE (its best case is ~92 G fetch4/s at 21.33 flops a fetch). Eight halves a
 * texel is half the x fetches for the same arithmetic, which RGBA16F cannot give. This prices that before anything
 * is built on it: one in-major projection, one row and four outputs a fragment, the weight in RGBA16F both ways, x
 * either way - the same sum in the same order, so the two must agree to a reordering of nothing.
 *
 * Every iteration writes its own target (a ring), because a tile-based GPU discards the first of two identical
 * full-viewport draws into one target (README, "THE OBVIOUS FIX IS DEFEATED BY HIDDEN-SURFACE REMOVAL").
 */
import { createWebGL2Context } from "../../webgl/index.js";
import { float32ToFloat16Array } from "../../shared/weights/float16.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const W = 2048;
const TRIANGLE = new Float32Array([-1, -1, 3, -1, -1, 3]);
const VERTEX = "#version 300 es\nin vec2 position;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); }";

/** out[row][o] = sum_k x[row][k] * w[k][o], four outputs a fragment; x as RGBA16F (4 a texel) or RGBA32UI (8). */
const fragment = (rows, inC, outC, packed) => `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
const int W = ${W};
const int ROWS = ${rows};
const int IN = ${inC};
const int OUT4 = ${outC / 4};
uniform ${packed ? "usampler2D" : "sampler2D"} xTex;
uniform sampler2D wTex;
out vec4 result;
vec4 fetchW(int t) { return texelFetch(wTex, ivec2(t % W, t / W), 0); }
void main() {
  int texel = int(gl_FragCoord.y) * W + int(gl_FragCoord.x);
  if (texel >= ROWS * OUT4) { result = vec4(0.0); return; }
  int group = texel % OUT4;
  int row = texel / OUT4;
  vec4 acc = vec4(0.0);
${packed ? `  int xBase = row * ${inC / 8};
  for (int i = 0; i < ${inC / 8}; ++i) {
    int t = xBase + i;
    uvec4 q = texelFetch(xTex, ivec2(t % W, t / W), 0);
    vec2 a = unpackHalf2x16(q.x); vec2 b = unpackHalf2x16(q.y);
    vec2 c = unpackHalf2x16(q.z); vec2 d = unpackHalf2x16(q.w);
    int k = i * 8;
    acc += a.x * fetchW(k * OUT4 + group) + a.y * fetchW((k + 1) * OUT4 + group)
      + b.x * fetchW((k + 2) * OUT4 + group) + b.y * fetchW((k + 3) * OUT4 + group)
      + c.x * fetchW((k + 4) * OUT4 + group) + c.y * fetchW((k + 5) * OUT4 + group)
      + d.x * fetchW((k + 6) * OUT4 + group) + d.y * fetchW((k + 7) * OUT4 + group);
  }` : `  int xBase = row * ${inC / 4};
  for (int i = 0; i < ${inC / 4}; ++i) {
    int t = xBase + i;
    vec4 x = texelFetch(xTex, ivec2(t % W, t / W), 0);
    int k = i * 4;
    acc += x.x * fetchW(k * OUT4 + group) + x.y * fetchW((k + 1) * OUT4 + group)
      + x.z * fetchW((k + 2) * OUT4 + group) + x.w * fetchW((k + 3) * OUT4 + group);
  }`}
  result = acc;
}`;

function compile(gl, source) {
  const program = gl.createProgram();
  for (const [type, text] of [[gl.VERTEX_SHADER, VERTEX], [gl.FRAGMENT_SHADER, source]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, text);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
  }
  gl.bindAttribLocation(program, 0, "position");
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

function texture(gl, internal, format, type, texels, data) {
  const height = Math.ceil(texels / W);
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internal, W, height);
  if (data !== null) {
    const padded = new data.constructor(W * height * 4);
    padded.set(data);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, height, format, type, padded);
  }
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  return { t, height };
}

export async function main(_device, args) {
  const iterations = Number(option(args, "iterations", "16"));
  const rounds = Number(option(args, "rounds", "5"));
  const shapes = option(args, "shapes", "3481x128x256,3481x128x128,3481x512x128,59x768x768").split(",")
    .map((s) => s.split("x").map(Number));
  const gl = createWebGL2Context();
  gl.getExtension("EXT_color_buffer_float");
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, TRIANGLE, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const fb = gl.createFramebuffer();
  const out = [];
  for (const [rows, inC, outC] of shapes) {
    let seed = 7;
    const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
    const x = Float32Array.from({ length: rows * inC }, random);
    const w = Float32Array.from({ length: inC * outC }, random);
    const xHalf = float32ToFloat16Array(x);
    const x16 = texture(gl, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, rows * inC / 4, xHalf);
    const x32 = texture(gl, gl.RGBA32UI, gl.RGBA_INTEGER, gl.UNSIGNED_INT, rows * inC / 8,
      new Uint32Array(xHalf.buffer, xHalf.byteOffset, xHalf.length / 2));
    const wt = texture(gl, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, inC * outC / 4, float32ToFloat16Array(w));
    // ...and both operands in float32, which is what a fold holds for a weight under 65,536 floats and an x whose
    // row stride divides the texture width - the AF3 trunk's pair projections
    const xF = texture(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, rows * inC / 4, Float32Array.from(xHalf, (h, i) => x[i]));
    const wF = texture(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, inC * outC / 4, w);
    const ring = Array.from({ length: iterations }, () => texture(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, rows * outC / 4, null));
    const height = ring[0].height;
    const timings = {};
    const answers = {};
    const plain = compile(gl, fragment(rows, inC, outC, false));
    const packedProgram = compile(gl, fragment(rows, inC, outC, true));
    // arm: [program, x texture, weight texture]
    const arms = {
      rgba16f: [plain, x16, wt], packed8: [packedProgram, x32, wt],
      x32w32: [plain, xF, wF], x16w32: [plain, x16, wF], x32w16: [plain, xF, wt],
    };
    const samples = Object.fromEntries(Object.keys(arms).map((arm) => [arm, []]));
    const run = (arm) => {
      const [program, xt, wtex] = arms[arm];
      gl.useProgram(program);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, xt.t);
      gl.uniform1i(gl.getUniformLocation(program, "xTex"), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, wtex.t);
      gl.uniform1i(gl.getUniformLocation(program, "wTex"), 1);
      gl.viewport(0, 0, W, height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      for (const target of ring) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target.t, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      const pixels = new Float32Array(W * height * 4);
      gl.readPixels(0, 0, W, height, gl.RGBA, gl.FLOAT, pixels);
      return pixels.subarray(0, rows * outC);
    };
    for (const arm of Object.keys(arms)) run(arm); // warm, and the clock
    for (let round = 0; round < rounds; round += 1) {
      for (const arm of Object.keys(arms)) {
        const start = performance.now();
        answers[arm] = Float32Array.from(run(arm));
        samples[arm].push((performance.now() - start) / iterations);
      }
    }
    for (const arm of Object.keys(arms)) timings[arm] = Math.min(...samples[arm]);
    let worst = 0;
    let scale = 0;
    for (let i = 0; i < answers.rgba16f.length; i += 1) {
      worst = Math.max(worst, Math.abs(answers.rgba16f[i] - answers.packed8[i]));
      scale = Math.max(scale, Math.abs(answers.rgba16f[i]));
    }
    const flops = 2 * rows * inC * outC;
    out.push({
      shape: `${rows}x${inC}x${outC}`,
      ms: Object.fromEntries(Object.entries(timings).map(([arm, ms]) => [arm, Number(ms.toFixed(3))])),
      gflops: Object.fromEntries(Object.entries(timings).map(([arm, ms]) => [arm, Math.round(flops / ms / 1e6)])),
      packedAgainstRgba16f: Number((timings.rgba16f / timings.packed8).toFixed(2)),
      worstDifference: Number((worst / scale).toExponential(2)),
    });
    for (const t of [x16, x32, wt, xF, wF, ...ring]) gl.deleteTexture(t.t);
    gl.deleteProgram(plain);
    gl.deleteProgram(packedProgram);
  }
  return { rows: out };
}
