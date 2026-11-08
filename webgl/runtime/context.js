/**
 * A WebGL2 context used as a compute device, and the texture layout every
 * kernel in this tree shares.
 *
 * 🔴 THIS IS ONE COPY WHERE THERE WERE TWO. `createWebGL2Context`, `compile`,
 * `VERTEX_SHADER`, `TEXTURE_WIDTH` and `CHANNELS_PER_TEXEL` were duplicated
 * byte for byte between the triangle contraction and flash attention, because
 * each was written as a standalone spike against the WGSL kernel beside it and
 * neither had a backend to belong to. They have one now, so the layout is
 * stated once: a second copy of "four floats to a texel" is a second place for
 * it to stop being true.
 *
 * `textureShape` is the one of the five that was NOT identical - attention
 * floored the height at one row and did not return the texel count, the
 * triangle returned the count and did not floor. The union of the two is what
 * is here, and for every input either ever received (`elements >= 1`) both gave
 * the same width and height; only an EMPTY tensor differed, which neither
 * kernel passes.
 *
 * WebGPU's equivalent is `webgpu/runtime/device.js`, which asks an adapter for
 * features and limits. There is no adapter here: a WebGL2 context either has
 * the extension or it does not, so this module checks the one extension the
 * whole approach rests on and `./limits.js` reads the rest.
 */

/**
 * Texel rows this wide, and it MUST stay a power of two.
 *
 * A texel is addressed as (index % width, index / width), and with an arbitrary
 * width that is an integer division per access - twice per inner-loop step in
 * the contraction. A power of two lets the compiler shift and mask instead.
 */
export const TEXTURE_WIDTH = 2048;

/**
 * 🔴 FOUR FLOATS TO A TEXEL, AND NO REPACKING TO GET THERE. The tensors are
 * laid out with channel fastest, so four consecutive channels are already four
 * consecutive floats - exactly one RGBA32F texel. The same Float32Array uploads
 * as RGBA without being rearranged, and the shader gets four values per fetch.
 */
export const CHANNELS_PER_TEXEL = 4;

/** The rectangle `elements` floats occupy at four to a texel. */
export function textureShape(elements) {
  const texels = Math.ceil(elements / CHANNELS_PER_TEXEL);
  return { width: TEXTURE_WIDTH, height: Math.max(1, Math.ceil(texels / TEXTURE_WIDTH)), texels };
}

/**
 * The whole-framebuffer triangle every kernel here draws.
 *
 * There is no compute shader in WebGL2, so a "dispatch" is a draw that covers
 * the output rectangle and one fragment computes one output element. Three
 * vertices at (-1,-1), (3,-1), (-1,3) cover the viewport with no clipping and
 * no second triangle.
 */
export const VERTEX_SHADER = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`;

/** Vertices for the triangle above, ready for `bufferData`. */
export const FULLSCREEN_TRIANGLE = new Float32Array([-1, -1, 3, -1, -1, 3]);

export function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`WebGL2 shader failed to compile: ${log}`);
  }
  return shader;
}

/** Link a program from the shared vertex shader and a kernel's fragment source. */
export function linkProgram(gl, fragmentSource) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`WebGL2 program failed to link: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

/**
 * `linkProgram` without asking how it went: compile and link are only REQUESTED, and asking for
 * a status is what waits for them. With KHR_parallel_shader_compile the driver does them off the
 * thread until `finishProgram` asks; see `Execution.prepare`.
 */
export function startProgram(gl, fragmentSource) {
  const program = gl.createProgram();
  const shaders = [[gl.VERTEX_SHADER, VERTEX_SHADER], [gl.FRAGMENT_SHADER, fragmentSource]]
    .map(([type, source]) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      gl.attachShader(program, shader);
      return shader;
    });
  gl.linkProgram(program);
  return { program, shaders };
}

/** The status `startProgram` did not ask for; throws as `linkProgram` and `compile` would. */
export function finishProgram(gl, { program, shaders }) {
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const logs = shaders.map((shader) => gl.getShaderInfoLog(shader)).filter(Boolean).join(" ");
    throw new Error(`WebGL2 program failed to link: ${gl.getProgramInfoLog(program)} ${logs}`);
  }
  for (const shader of shaders) gl.deleteShader(shader);
  return program;
}

/**
 * A WebGL2 context that can render to float textures, or an explanation.
 *
 * EXT_color_buffer_float is what makes an R32F or RGBA32F framebuffer legal to
 * draw into. Without it the whole approach is unavailable, and saying so by
 * name is more use than a link error deep in a draw call. It is an EXTENSION
 * and not core WebGL2, which is exactly why it is checked here rather than
 * assumed - see `./limits.js` for what else a context has to have.
 */
export function createWebGL2Context(canvas) {
  // An OffscreenCanvas where there is one that gives WebGL2, else a detached document canvas:
  // Safari has no OffscreenCanvas before 16.4, and no WebGL on one before 17.
  let gl = canvas?.getContext("webgl2") ?? null;
  if (canvas === undefined && typeof OffscreenCanvas === "function") {
    gl = new OffscreenCanvas(1, 1).getContext("webgl2");
  }
  if (canvas === undefined && gl === null && typeof document !== "undefined") {
    const element = document.createElement("canvas");
    element.width = 1;
    element.height = 1;
    gl = element.getContext("webgl2");
  }
  if (gl === null) throw new Error("this browser has no WebGL2");
  if (gl.getExtension("EXT_color_buffer_float") === null) {
    throw new Error("WebGL2 here cannot render to float textures (EXT_color_buffer_float)");
  }
  return gl;
}
