/**
 * How much UNIFORM-BLOCK data a draw may actually bind here - the largest single block, and all blocks together -
 * against what the context reports.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-uniform-budget.js
 *     python3 tools/run-in-safari.py tools/gpu/probe-webgl-uniform-budget.js
 *
 * 🔴 SAFARI 16.3 REPORTS 192 KiB AND DRAWS WITH LESS THAN 16. It reports MAX_UNIFORM_BLOCK_SIZE 16384 and
 * MAX_FRAGMENT_UNIFORM_BLOCKS 12, and refuses (GL_INVALID_OPERATION at the draw, nothing at link or bind) a single
 * 16,384-byte block and ANY set of blocks totalling 16,384 bytes: 1 x 16,000 bytes draws, 3 x 4,096 draws, 4 x 4,096
 * does not. Chrome draws with all of it. So a projection whose weight lives in uniform blocks
 * (probe-webgl-ubo-weight.js) can hold ~4,000 floats in Safari - not one of AF3's 16,384-float pair weights - and the
 * browser this backend exists for cannot use the path. Every case checks `getError` after its draw and reads the
 * answer back, because a refused draw leaves its target unwritten and nothing else says so.
 */
import { createWebGL2Context } from "../../webgl/index.js";

const VERTEX = "#version 300 es\nin vec2 position;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); }";

export async function main() {
  const gl = createWebGL2Context();
  gl.getExtension("EXT_color_buffer_float");
  const reported = {
    maxUniformBlockSize: gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE),
    maxFragmentUniformBlocks: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_BLOCKS),
    maxCombinedUniformBlocks: gl.getParameter(gl.MAX_COMBINED_UNIFORM_BLOCKS),
    offsetAlignment: gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT),
  };
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const target = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, target);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 64, 1);
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
  const vec4Cap = reported.maxUniformBlockSize / 16;
  const cases = [[1, 256], [1, 512], [1, 1000], [1, vec4Cap], [2, 256], [3, 256], [4, 256], [12, 64],
    [Math.min(4, reported.maxFragmentUniformBlocks), vec4Cap]];
  const results = [];
  for (const [blocks, vec4s] of cases) {
    const declarations = Array.from({ length: blocks }, (_, b) => `uniform B${b} { vec4 w${b}[${vec4s}]; };`).join("\n");
    const sum = Array.from({ length: blocks }, (_, b) => `w${b}[int(gl_FragCoord.x) % ${vec4s}]`).join(" + ");
    const program = gl.createProgram();
    for (const [type, text] of [[gl.VERTEX_SHADER, VERTEX],
      [gl.FRAGMENT_SHADER, `#version 300 es\nprecision highp float;\n${declarations}\nout vec4 result;\nvoid main() { result = ${sum}; }`]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, text);
      gl.compileShader(shader);
      gl.attachShader(program, shader);
    }
    gl.bindAttribLocation(program, 0, "position");
    gl.linkProgram(program);
    const row = { blocks, bytesEach: vec4s * 16, bytesTotal: blocks * vec4s * 16 };
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) { results.push({ ...row, draws: false, link: "failed" }); continue; }
    for (let b = 0; b < blocks; b += 1) gl.uniformBlockBinding(program, gl.getUniformBlockIndex(program, `B${b}`), b);
    gl.useProgram(program);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, buffer);
    gl.bufferData(gl.UNIFORM_BUFFER, new Float32Array(vec4s * 4 * blocks).fill(1), gl.STATIC_DRAW);
    for (let b = 0; b < blocks; b += 1) gl.bindBufferRange(gl.UNIFORM_BUFFER, b, buffer, b * vec4s * 16, vec4s * 16);
    gl.viewport(0, 0, 64, 1);
    while (gl.getError() !== gl.NO_ERROR);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const error = gl.getError();
    const pixels = new Float32Array(64 * 4);
    gl.readPixels(0, 0, 64, 1, gl.RGBA, gl.FLOAT, pixels);
    results.push({ ...row, draws: error === gl.NO_ERROR && pixels[0] === blocks, glError: error, value: pixels[0] });
    gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
  }
  const drawn = results.filter((r) => r.draws);
  const refused = results.filter((r) => !r.draws);
  return {
    renderer: gl.getParameter(gl.RENDERER), reported, results,
    largestTotalDrawn: Math.max(0, ...drawn.map((r) => r.bytesTotal)),
    smallestTotalRefused: refused.length === 0 ? null : Math.min(...refused.map((r) => r.bytesTotal)),
    reportedTotal: reported.maxUniformBlockSize * reported.maxFragmentUniformBlocks,
  };
}
