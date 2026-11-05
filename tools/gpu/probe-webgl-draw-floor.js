/**
 * What a draw costs before it computes anything, and what that is worth over a
 * whole fold.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-draw-floor.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-draw-floor.js --texels=1,256,65536
 *
 * 🔴 IT EXISTS BECAUSE A bioemu2 FOLD ISSUES 54,381 DRAWS AND NOBODY HAD PRICED
 * ONE. `--host` puts `drawMs` at 14947 ms over those draws - 275 us each on
 * average - and the denoiser's own per-step profile is **41 kernels and about
 * 1500 draws**, most of them small: `bioemu2.cond-slice` is 192 calls a step at
 * 0.37 ms each for 73,728 output floats, which is far more than that arithmetic
 * can cost. So the question is whether the small kernels are bound by their
 * WORK or by the fact that they are a draw at all - and if it is the draw, then
 * the draw COUNT is a lever that no kernel optimisation reaches, and fusing is
 * the only thing that moves it.
 *
 * 🔴 EVERY ARM WRITES ITS OWN TARGET, because this device discards the earlier
 * of two identical full-viewport draws into one target before shading them -
 * the trap that has cost three measurements in this tree. A ring of distinct
 * outputs is the fix.
 *
 * 🔴 AND IT REPORTS THE FOLD'S SHARE RATHER THAN LEAVING THE READER TO
 * MULTIPLY, because the whole point of the number is `floor x 54,381` against a
 * 21.7 s fold.
 */
import { createWebGL2Context } from "../../webgl/index.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const FULLSCREEN_TRIANGLE = new Float32Array([-1, -1, 3, -1, -1, 3]);
const VERTEX_SHADER = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`;
// The least a kernel can do: one texel in, one out, no loop.
const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const int W = 2048;
uniform sampler2D xTex;
out vec4 result;
void main() {
  int texel = int(gl_FragCoord.y) * W + int(gl_FragCoord.x);
  result = texelFetch(xTex, ivec2(texel % W, texel / W), 0) * 1.0001;
}`;

const compile = (gl, kind, source) => {
  const shader = gl.createShader(kind);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`${gl.getShaderInfoLog(shader)}\n${source}`);
  }
  return shader;
};

export async function main(device, args) {
  const texelList = (option(args, "texels", "1,64,1024,16384,65536,262144"))
    .split(",").map(Number);
  const rounds = Number(option(args, "rounds", "4"));
  const draws = Number(option(args, "draws", "512"));
  const foldDispatches = Number(option(args, "fold-dispatches", "54381"));
  const foldMs = Number(option(args, "fold-ms", "21710"));
  const gl = createWebGL2Context();
  const W = 2048;

  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`link: ${gl.getProgramInfoLog(program)}`);
  }
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, FULLSCREEN_TRIANGLE, gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

  const source = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, source);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, 256, 0, gl.RGBA, gl.FLOAT,
    new Float32Array(W * 256 * 4).fill(0.5));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, source);
  gl.uniform1i(gl.getUniformLocation(program, "xTex"), 0);

  const rows = [];
  for (const texels of texelList) {
    const width = Math.min(texels, W);
    const height = Math.max(1, Math.ceil(texels / W));
    // A ring so no target is written twice: hidden-surface removal would void
    // every draw but the last and the arm would time one.
    const ring = Array.from({ length: draws }, () => {
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      const out = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, out);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, out, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      return { framebuffer, out };
    });
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, source);
    const draw = (i) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, ring[i].framebuffer);
      gl.viewport(0, 0, width, height);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    const sync = () => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, ring[0].framebuffer);
      const pixel = new Float32Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, pixel);
      return pixel[0];
    };
    draw(0);
    const witness = sync();
    const samples = [];
    for (let round = 0; round < rounds; round += 1) {
      const start = performance.now();
      for (let i = 0; i < draws; i += 1) draw(i);
      sync();
      samples.push((performance.now() - start) / draws);
    }
    const ms = Math.min(...samples);
    rows.push({
      texels,
      outputFloats: texels * 4,
      microsecondsPerDraw: Number((ms * 1000).toFixed(1)),
      spread: Number((Math.max(...samples) / Math.min(...samples)).toFixed(2)),
      gigabytesPerSecond: Number((texels * 16 * 2 / ms / 1e6).toFixed(1)),
      witnessFinite: Number.isFinite(witness),
    });
    for (const { framebuffer, out } of ring) {
      gl.deleteFramebuffer(framebuffer);
      gl.deleteTexture(out);
    }
  }

  const floor = Math.min(...rows.map((r) => r.microsecondsPerDraw));
  const foldFloorMs = (floor * foldDispatches) / 1000;
  return {
    renderer: gl.getParameter(gl.RENDERER),
    draws,
    rows,
    microsecondsPerDrawFloor: floor,
    foldDispatches,
    foldMs,
    foldFloorMilliseconds: Number(foldFloorMs.toFixed(0)),
    foldFloorShare: Number((foldFloorMs / foldMs).toFixed(3)),
    verdict: `a draw costs at least ${floor.toFixed(1)} us before it computes anything, so a fold's `
      + `${foldDispatches} of them are ${foldFloorMs.toFixed(0)} ms - `
      + `${(100 * foldFloorMs / foldMs).toFixed(1)}% of a ${(foldMs / 1000).toFixed(1)} s fold, `
      + "which no kernel optimisation reaches and only a smaller draw COUNT does",
  };
}
