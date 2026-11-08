/**
 * A DEPENDENT CHAIN of projections - each reading the last one's output, as a fold's are - with the inner loop split
 * across instances and summed by the BLEND unit, against one fragment walking the whole loop.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-ksplit-chain.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-ksplit-chain.js --rows=59 --width=768 --splits=1,2,4,8,16
 *
 * 🔴 WHY A CHAIN. Every kernel bench in this tree draws into a RING of independent outputs, so the GPU may overlap
 * the draws and the bench measures THROUGHPUT. A fold is a chain: each projection reads the one before, so nothing
 * overlaps and a draw costs its LATENCY - for a 59-row projection ~1,500 fragments, each looping 192 times over
 * dependent fetches, which fills nothing. That is the reading under which kernel wins measured here (2x, 4x) kept
 * vanishing inside the fold.
 *
 * 🔴 WHY BLENDING. `drawArraysInstanced` draws the same full-screen triangle S times; instance s takes the inner
 * loop's s-th slice, and additive blending (EXT_float_blend on an RGBA32F target) sums the S partial results into one
 * texel - S times the fragments, each loop S times shorter, no reduction pass, one draw call. Blending also switches
 * off hidden-surface removal, so overlapping instances are not discarded. The split this file's README measured 1.5x
 * slower used an explicit reduction draw, and was timed as throughput.
 *
 * The chain alternates two weights (f16, in-major [in][out]) so a link's input is the previous link's output; every
 * arm's final output is compared with the one-instance arm's.
 */
import { createWebGL2Context } from "../../webgl/index.js";
import { float32ToFloat16Array } from "../../shared/weights/float16.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const W = 2048;

const VERTEX = `#version 300 es
in vec2 position;
flat out int instance;
void main() { instance = gl_InstanceID; gl_Position = vec4(position, 0.0, 1.0); }`;

/** out[row][o] += sum over this instance's slice of k: x[row][k] * w[k][o]; four outputs a fragment. */
const fragment = (rows, width, splits) => `#version 300 es
precision highp float;
precision highp int;
const int W = ${W};
const int ROWS = ${rows};
const int OUT4 = ${width / 4};
const int SLICE4 = ${width / 4 / splits};
uniform sampler2D xTex;
uniform sampler2D wTex;
flat in int instance;
out vec4 result;
vec4 at(sampler2D t, int i) { return texelFetch(t, ivec2(i % W, i / W), 0); }
void main() {
  int texel = int(gl_FragCoord.y) * W + int(gl_FragCoord.x);
  if (texel >= ROWS * OUT4) { result = vec4(0.0); return; }
  int group = texel % OUT4;
  int row = texel / OUT4;
  int first = instance * SLICE4;
  vec4 acc = vec4(0.0);
  for (int j = 0; j < SLICE4; ++j) {
    int i = first + j;
    vec4 x = at(xTex, row * OUT4 + i);
    int k = i * 4;
    acc += x.x * at(wTex, k * OUT4 + group) + x.y * at(wTex, (k + 1) * OUT4 + group)
      + x.z * at(wTex, (k + 2) * OUT4 + group) + x.w * at(wTex, (k + 3) * OUT4 + group);
  }
  result = acc;
}`;

function compile(gl, fragmentSource) {
  const program = gl.createProgram();
  for (const [type, text] of [[gl.VERTEX_SHADER, VERTEX], [gl.FRAGMENT_SHADER, fragmentSource]]) {
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
  const height = Math.max(1, Math.ceil(texels / W));
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
  const rows = Number(option(args, "rows", "59"));
  const width = Number(option(args, "width", "768"));
  const links = Number(option(args, "links", "48"));
  const rounds = Number(option(args, "rounds", "5"));
  const splitsList = option(args, "splits", "1,2,4,8,16").split(",").map(Number);
  const gl = createWebGL2Context();
  const floatBlend = gl.getExtension("EXT_float_blend") !== null;
  gl.getExtension("EXT_color_buffer_float");
  if (!floatBlend) return { floatBlend, verdict: "no EXT_float_blend: an RGBA32F target cannot be blended here" };
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const fb = gl.createFramebuffer();

  let seed = 11;
  const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return (seed / 2147483648 - 0.5) * 0.08; };
  const weights = [0, 1].map(() => texture(gl, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, width * width / 4,
    float32ToFloat16Array(Float32Array.from({ length: width * width }, random))));
  const x0 = Float32Array.from({ length: rows * width }, () => random() * 10);
  const input = texture(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, rows * width / 4, x0);
  // the chain's activations: a fresh target every link, so nothing is written twice
  const chain = Array.from({ length: links }, () => texture(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, rows * width / 4, null));
  const height = chain[0].height;

  const runChain = (program, splits) => {
    gl.useProgram(program);
    gl.uniform1i(gl.getUniformLocation(program, "xTex"), 0);
    gl.uniform1i(gl.getUniformLocation(program, "wTex"), 1);
    gl.viewport(0, 0, W, height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    if (splits > 1) {
      gl.enable(gl.BLEND);
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
    } else {
      gl.disable(gl.BLEND);
    }
    let previous = input;
    for (let link = 0; link < links; link += 1) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, chain[link].t, 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, previous.t);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, weights[link % 2].t);
      if (splits > 1) {
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 3, splits);
      } else {
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      previous = chain[link];
    }
    gl.disable(gl.BLEND);
    const pixels = new Float32Array(W * height * 4);
    gl.readPixels(0, 0, W, height, gl.RGBA, gl.FLOAT, pixels);
    return pixels.subarray(0, rows * width);
  };

  const arms = splitsList.filter((s) => (width / 4) % s === 0).map((splits) => ({
    splits, program: compile(gl, fragment(rows, width, splits)), samples: [], answer: null,
  }));
  // warm - and each arm's GL error, since a refused draw is a skipped (fast) one and its target stays unwritten
  for (const arm of arms) {
    while (gl.getError() !== gl.NO_ERROR);
    runChain(arm.program, arm.splits);
    arm.glError = gl.getError();
  }
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of arms) {
      const start = performance.now();
      arm.answer = Float32Array.from(runChain(arm.program, arm.splits));
      arm.samples.push(performance.now() - start);
    }
  }
  const reference = arms[0].answer;
  let scale = 0;
  for (const v of reference) scale = Math.max(scale, Math.abs(v));
  const flops = 2 * rows * width * width * links;
  return {
    floatBlend, rows, width, links,
    arms: arms.map((arm) => {
      let worst = 0;
      for (let i = 0; i < reference.length; i += 1) worst = Math.max(worst, Math.abs(arm.answer[i] - reference[i]));
      const ms = Math.min(...arm.samples);
      return {
        splits: arm.splits, glError: arm.glError, chainMs: Number(ms.toFixed(2)), perLinkMs: Number((ms / links).toFixed(3)),
        gflops: Math.round(flops / ms / 1e6), spread: Number((Math.max(...arm.samples) / ms).toFixed(2)),
        relativeToOneInstance: Number((worst / (scale || 1)).toExponential(2)),
      };
    }),
  };
}
