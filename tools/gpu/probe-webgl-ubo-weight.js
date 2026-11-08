/**
 * A projection whose WEIGHT sits in uniform blocks rather than a texture, in a dependent chain, against the texture
 * kernel - the shape of every pair projection in an AF3 trunk.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-ubo-weight.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-ubo-weight.js --rows=3481 --width=128 --links=48
 *
 * 🔴 webgl/README.md MEASURED A UNIFORM-BLOCK READ AT 2.1x A `texelFetch` AND DECLINED IT ON bioemu2's WEIGHTS, which
 * are 131 K to 1.6 M floats and do not fit the 65,536 floats of uniforms a draw can hold - every chunk an extra draw.
 * AF3's trunk pair weights are 128 x 128 = 16,384 floats: one draw holds the whole of one, and since a weight never
 * changes it can live in its own uniform BUFFER, uploaded once and only BOUND per draw - no extra draw at all. The
 * one-texel in-major kernel spends four of every five fetches on the weight; this moves those four to the constant
 * path and leaves only x on the texture unit.
 *
 * 🔴 AND IT DOES NOT WORK IN SAFARI, AND THE FIRST VERSION OF THIS PROBE SAID IT WAS 2.22x THERE. Safari 16.3 draws
 * with less than 16 KiB of uniform blocks IN TOTAL (probe-webgl-uniform-budget.js) - a 128 x 128 weight is 64 KiB - and
 * refuses the draw with GL_INVALID_OPERATION; this probe never called `getError`, a refused draw is a skipped draw and
 * so a FAST one, and the chain's values had decayed to ~3e-5, where an unwritten target compared as agreement. Every
 * arm's error is checked now, a refused arm voids the timing, and the weights are scaled to keep the chain at ~1.
 * Chrome, which draws with all of it: 1.22x at 128 -> 128 and 1.29x at 64 -> 64 - which a whole AF3 fold did not see.
 *
 * 🔴 GLSL ES 3.00 INDEXES AN ARRAY OF BLOCKS ONLY BY A CONSTANT, so the k loop is unrolled per block by the generator:
 * block b holds k rows [b * K_PER, (b + 1) * K_PER) of the in-major weight, 512 vec4s each (8 KiB, std140 - a
 * single 16 KiB block is refused by Safari even alone).
 *
 * A chain of `links` square projections alternating two weights, each reading the last one's output, as a fold's do;
 * both arms must agree to a reordering of nothing.
 */
import { createWebGL2Context } from "../../webgl/index.js";
import { float32ToFloat16Array } from "../../shared/weights/float16.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const W = 2048;
const VEC4_PER_BLOCK = 512;
const VERTEX = "#version 300 es\nin vec2 position;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); }";

const head = (rows, width) => `#version 300 es
precision highp float;
precision highp int;
const int W = ${W};
const int ROWS = ${rows};
const int OUT4 = ${width / 4};
const int IN4 = ${width / 4};
uniform sampler2D xTex;
out vec4 result;
vec4 at(sampler2D t, int i) { return texelFetch(t, ivec2(i % W, i / W), 0); }`;

const textureKernel = (rows, width) => `${head(rows, width)}
uniform sampler2D wTex;
void main() {
  int texel = int(gl_FragCoord.y) * W + int(gl_FragCoord.x);
  if (texel >= ROWS * OUT4) { result = vec4(0.0); return; }
  int group = texel % OUT4;
  int row = texel / OUT4;
  vec4 acc = vec4(0.0);
  for (int i = 0; i < IN4; ++i) {
    vec4 x = at(xTex, row * IN4 + i);
    int k = i * 4;
    acc += x.x * at(wTex, k * OUT4 + group) + x.y * at(wTex, (k + 1) * OUT4 + group)
      + x.z * at(wTex, (k + 2) * OUT4 + group) + x.w * at(wTex, (k + 3) * OUT4 + group);
  }
  result = acc;
}`;

const uboKernel = (rows, width) => {
  const kPer = VEC4_PER_BLOCK / (width / 4); // weight rows (k) a block holds
  const blocks = width / kPer;
  const declarations = Array.from({ length: blocks },
    (_, b) => `layout(std140) uniform Weight${b} { vec4 w${b}[${VEC4_PER_BLOCK}]; };`).join("\n");
  const sweeps = Array.from({ length: blocks }, (_, b) => `
  for (int i = 0; i < ${kPer / 4}; ++i) {
    vec4 x = at(xTex, row * IN4 + ${b * kPer / 4} + i);
    int k = i * 4;
    acc += x.x * w${b}[k * OUT4 + group] + x.y * w${b}[(k + 1) * OUT4 + group]
      + x.z * w${b}[(k + 2) * OUT4 + group] + x.w * w${b}[(k + 3) * OUT4 + group];
  }`).join("");
  return `${head(rows, width)}
${declarations}
void main() {
  int texel = int(gl_FragCoord.y) * W + int(gl_FragCoord.x);
  if (texel >= ROWS * OUT4) { result = vec4(0.0); return; }
  int group = texel % OUT4;
  int row = texel / OUT4;
  vec4 acc = vec4(0.0);${sweeps}
  result = acc;
}`;
};

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
  const rows = Number(option(args, "rows", "3481"));
  const width = Number(option(args, "width", "128"));
  const links = Number(option(args, "links", "48"));
  const rounds = Number(option(args, "rounds", "5"));
  const gl = createWebGL2Context();
  gl.getExtension("EXT_color_buffer_float");
  const limits = {
    maxUniformBlockSize: gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE),
    maxFragmentUniformBlocks: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_BLOCKS),
  };
  const blocksNeeded = (width * width / 4) / VEC4_PER_BLOCK;
  if (limits.maxUniformBlockSize < VEC4_PER_BLOCK * 16 || limits.maxFragmentUniformBlocks < blocksNeeded) {
    return { limits, verdict: `this context cannot hold a ${width} x ${width} weight in ${blocksNeeded} blocks` };
  }
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const fb = gl.createFramebuffer();

  let seed = 5;
  // weights of variance 1 / width, so a link keeps the chain's scale (at 0.25 it decayed to ~3e-5 by link 48)
  const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return (seed / 2147483648 - 0.5) * 2 * Math.sqrt(3 / width); };
  // f16-ROUNDED values in both arms, so the texture arm's RGBA16F and the uniform arm's f32 hold the same numbers
  const weightValues = [0, 1].map(() => {
    const halves = float32ToFloat16Array(Float32Array.from({ length: width * width }, random));
    const back = new Float32Array(halves.length);
    const view = new DataView(new ArrayBuffer(2));
    for (let i = 0; i < halves.length; i += 1) {
      view.setUint16(0, halves[i], true);
      const h = view.getUint16(0, true);
      const e = (h >> 10) & 31; const m = h & 1023; const s = h & 32768 ? -1 : 1;
      back[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? NaN : s * (1 + m / 1024) * 2 ** (e - 15);
    }
    return { halves, back };
  });
  const weightTextures = weightValues.map(({ halves }) =>
    texture(gl, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, width * width / 4, halves));
  const weightBuffers = weightValues.map(({ back }) => {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, buffer);
    gl.bufferData(gl.UNIFORM_BUFFER, back, gl.STATIC_DRAW);
    return buffer;
  });
  const input = texture(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, rows * width / 4,
    Float32Array.from({ length: rows * width }, random));
  const chain = Array.from({ length: links }, () => texture(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, rows * width / 4, null));
  const height = chain[0].height;

  const textureProgram = compile(gl, textureKernel(rows, width));
  const uboProgram = compile(gl, uboKernel(rows, width));
  for (let b = 0; b < blocksNeeded; b += 1) {
    gl.uniformBlockBinding(uboProgram, gl.getUniformBlockIndex(uboProgram, `Weight${b}`), b);
  }
  const run = (arm) => {
    const program = arm === "texture" ? textureProgram : uboProgram;
    gl.useProgram(program);
    gl.uniform1i(gl.getUniformLocation(program, "xTex"), 0);
    if (arm === "texture") gl.uniform1i(gl.getUniformLocation(program, "wTex"), 1);
    gl.viewport(0, 0, W, height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    let previous = input;
    for (let link = 0; link < links; link += 1) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, chain[link].t, 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, previous.t);
      if (arm === "texture") {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, weightTextures[link % 2].t);
      } else {
        for (let b = 0; b < blocksNeeded; b += 1) {
          gl.bindBufferRange(gl.UNIFORM_BUFFER, b, weightBuffers[link % 2], b * VEC4_PER_BLOCK * 16, VEC4_PER_BLOCK * 16);
        }
      }
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      previous = chain[link];
    }
    const pixels = new Float32Array(W * height * 4);
    gl.readPixels(0, 0, W, height, gl.RGBA, gl.FLOAT, pixels);
    return pixels.subarray(0, rows * width);
  };
  const samples = { texture: [], ubo: [] };
  const answers = {};
  // 🔴 EACH ARM'S GL ERROR IS CHECKED, AND A SILENT ONE VOIDS THE ARM: a draw refused with INVALID_OPERATION leaves its
  // target unwritten, and an unwritten chain of zeros compared with another can read as agreement.
  const errors = {};
  for (const arm of ["texture", "ubo"]) {
    while (gl.getError() !== gl.NO_ERROR);
    const sample = run(arm);
    errors[arm] = gl.getError();
    answers[`${arm}Warm`] = sample;
  }
  for (let round = 0; round < rounds; round += 1) {
    for (const arm of ["texture", "ubo"]) {
      const start = performance.now();
      answers[arm] = Float32Array.from(run(arm));
      samples[arm].push(performance.now() - start);
    }
  }
  let worst = 0; let scale = 0;
  for (let i = 0; i < answers.texture.length; i += 1) {
    worst = Math.max(worst, Math.abs(answers.texture[i] - answers.ubo[i]));
    scale = Math.max(scale, Math.abs(answers.texture[i]));
  }
  const flops = 2 * rows * width * width * links;
  const ms = Object.fromEntries(Object.entries(samples).map(([arm, s]) => [arm, Math.min(...s)]));
  if (Object.values(errors).some((error) => error !== gl.NO_ERROR)) {
    return {
      renderer: gl.getParameter(gl.RENDERER), limits, rows, width, links, blocks: blocksNeeded, glErrors: errors,
      verdict: "an arm's draws were REFUSED (GL error), so no timing is reported: a refused draw is a skipped one,"
        + " and a skipped draw is fast - see probe-webgl-uniform-budget.js",
    };
  }
  return {
    renderer: gl.getParameter(gl.RENDERER), limits, rows, width, links, blocks: blocksNeeded, glErrors: errors,
    outputScale: Number(scale.toExponential(2)),
    textureMs: Number(ms.texture.toFixed(2)), uboMs: Number(ms.ubo.toFixed(2)),
    speedup: Number((ms.texture / ms.ubo).toFixed(2)),
    textureGflops: Math.round(flops / ms.texture / 1e6), uboGflops: Math.round(flops / ms.ubo / 1e6),
    relativeDifference: Number((worst / (scale || 1)).toExponential(2)),
  };
}
