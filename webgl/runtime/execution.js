/**
 * What stands in for a compute dispatch here: one draw, one fragment per output
 * texel, into a tensor from ./storage.js.
 *
 * `webgpu/runtime/execution.js` is the shape this follows - allocate, upload,
 * dispatch, read, release, with the allocations and dispatches counted so a
 * fold's cost is measurable. The differences are forced by the API:
 *
 *   - There is no pipeline, only a linked PROGRAM, and no bind group: inputs are
 *     texture units and the kernel names a `sampler2D` per input.
 *   - A grid is a viewport. The WebGPU side folds a linear index across
 *     `(group.x, group.y)` because a dispatch is at most 65535 workgroups wide;
 *     here the output texture is `W` texels wide by construction and the
 *     fragment's own coordinates ARE the fold, so `linearTexel()` is the whole
 *     of it and there is no `linearGrid`/`rowGrid` to get wrong.
 *   - A tall output is drawn in BANDS, because a viewport is at most
 *     `MAX_VIEWPORT_DIMS` high while a texture may be `MAX_TEXTURE_SIZE`. Each
 *     band is one draw whose viewport ORIGIN is its first row, so
 *     `gl_FragCoord.y` is the absolute row and a kernel needs no row uniform.
 *
 * 🔴 THE PROGRAM CACHE IS KEYED BY SOURCE, NOT BY A NAME OR A HASH. The WebGPU
 * side learned this the expensive way twice - two models' kernels collided
 * under one key because the key named neither the channel widths nor the head
 * dimension the source embeds, and a cache that hands a caller somebody else's
 * kernel is a wrong answer with no error. Here every kernel is GENERATED with
 * its shapes baked in as `const int`, so the source is the only thing that
 * distinguishes two instances, and it is what the cache indexes.
 */
import { FULLSCREEN_TRIANGLE, TEXTURE_WIDTH, linkProgram } from "./context.js";
import { Storage } from "./storage.js";

/**
 * The declarations and helpers every kernel here opens with.
 *
 * `linearTexel()` is the fragment's own texel index in the output. `fetch4`
 * reads a whole texel of an input; `fetch1` reads one float, which costs a
 * texel fetch and a select and is for the ragged cases (a mask, a bias) rather
 * than for anything in an inner loop.
 */
export const PRELUDE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const int W = ${TEXTURE_WIDTH};
out vec4 result;

int linearTexel() { return int(gl_FragCoord.y) * W + int(gl_FragCoord.x); }

vec4 fetch4(sampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % W, texel / W), 0);
}

float fetch1(sampler2D tex, int index) {
  vec4 texel = fetch4(tex, index >> 2);
  int component = index & 3;
  return component == 0 ? texel.x
       : component == 1 ? texel.y
       : component == 2 ? texel.z : texel.w;
}
`;

export class Execution {
  /** @param {WebGL2RenderingContext} gl */
  constructor(gl, options = {}) {
    this.gl = gl;
    this.storage = new Storage(gl);
    /** @type {Map<string, {program: WebGLProgram, uniforms: Map<string, WebGLUniformLocation>}>} */
    this.programs = new Map();
    this.dispatches = 0;
    this.draws = 0;
    this.compiles = 0;
    /** Each draw checks for a GL error. Cheap, and off for a benchmark. */
    this.checkErrors = options.checkErrors ?? true;
    this.maxBandRows = Math.min(
      gl.getParameter(gl.MAX_VIEWPORT_DIMS)[1], gl.getParameter(gl.MAX_TEXTURE_SIZE));

    this.vertices = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertices);
    gl.bufferData(gl.ARRAY_BUFFER, FULLSCREEN_TRIANGLE, gl.STATIC_DRAW);
    this.vao = gl.createVertexArray();
    this.target = gl.createFramebuffer();
  }

  allocate(elements, label) { return this.storage.allocate(elements, label); }
  upload(values, label) { return this.storage.upload(values, label); }
  read(tensor) { return this.storage.read(tensor); }
  release(tensor) { return this.storage.release(tensor); }

  /** The program for this source, compiled once. */
  program(source) {
    const cached = this.programs.get(source);
    if (cached !== undefined) return cached;
    const gl = this.gl;
    const program = linkProgram(gl, source);
    // The attribute is bound once per program, into this execution's one VAO.
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertices);
    const position = gl.getAttribLocation(program, "position");
    // `position` is the shared VERTEX shader's attribute, not the kernel's: a
    // fragment shader that declares it would be asking for a varying nothing
    // writes. A negative location means the vertex stage did not link.
    if (position < 0) throw new Error("the shared vertex shader lost its `position` attribute");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    const entry = { program, uniforms: new Map() };
    this.programs.set(source, entry);
    this.compiles += 1;
    return entry;
  }

  #uniform(entry, name) {
    if (!entry.uniforms.has(name)) {
      entry.uniforms.set(name, this.gl.getUniformLocation(entry.program, name));
    }
    return entry.uniforms.get(name);
  }

  /**
   * Run one kernel into `output`.
   *
   * `inputs` is an object of sampler name to tensor, bound to units in its own
   * key order. `ints` and `floats` are the scalar uniforms, kept APART rather
   * than inferred from the value: `bias: 0` is a float that looks like an int,
   * and `uniform1i` on a float uniform is a silent GL error on some drivers and
   * a wrong value on others.
   *
   * @param {{source: string, output: import("./storage.js").Tensor,
   *   inputs?: Record<string, import("./storage.js").Tensor>,
   *   ints?: Record<string, number>, floats?: Record<string, number>,
   *   label?: string}} call
   */
  dispatch(call) {
    const gl = this.gl;
    const { source, output, inputs = {}, ints = {}, floats = {}, label = "kernel" } = call;
    const entry = this.program(source);
    gl.useProgram(entry.program);
    gl.bindVertexArray(this.vao);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.target);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D,
      output.texture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`${label}: cannot draw into ${output.label} (0x${status.toString(16)})`);
    }

    let unit = 0;
    for (const [name, tensor] of Object.entries(inputs)) {
      // 🔴 AN UNDECLARED SAMPLER IS A REFUSAL, NOT A NO-OP. `getUniformLocation`
      // returns null for a name the program does not have - including one the
      // compiler removed as unused - and `uniform1i(null, ...)` is silently
      // ignored, so the kernel then reads whatever was bound to unit 0. That is
      // the "read what was bound from before" failure the triangle kernel's own
      // header records, and it returns a plausible wrong answer.
      const location = this.#uniform(entry, name);
      if (location === null) {
        throw new Error(`${label}: no sampler named \`${name}\` in this kernel`);
      }
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tensor.texture);
      gl.uniform1i(location, unit);
      unit += 1;
    }
    for (const [name, value] of Object.entries(ints)) {
      const location = this.#uniform(entry, name);
      if (location === null) throw new Error(`${label}: no int uniform \`${name}\``);
      gl.uniform1i(location, value);
    }
    for (const [name, value] of Object.entries(floats)) {
      const location = this.#uniform(entry, name);
      if (location === null) throw new Error(`${label}: no float uniform \`${name}\``);
      gl.uniform1f(location, value);
    }

    this.dispatches += 1;
    for (let row = 0; row < output.height; row += this.maxBandRows) {
      const rows = Math.min(this.maxBandRows, output.height - row);
      // The viewport's ORIGIN is the band's first row, so gl_FragCoord.y is
      // absolute and the band lands in the rows it belongs to - no row uniform
      // and no re-draw on readback.
      gl.viewport(0, row, output.width, rows);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.draws += 1;
    }
    if (this.checkErrors) {
      const error = gl.getError();
      if (error !== gl.NO_ERROR) {
        throw new Error(`${label}: GL error 0x${error.toString(16)} after the draw`);
      }
    }
    return output;
  }

  stats() {
    return {
      dispatches: this.dispatches, draws: this.draws, programs: this.programs.size,
      maxBandRows: this.maxBandRows, ...this.storage.stats(),
    };
  }

  destroy() {
    const gl = this.gl;
    this.storage.releaseAll();
    for (const { program } of this.programs.values()) gl.deleteProgram(program);
    this.programs.clear();
    gl.deleteBuffer(this.vertices);
    gl.deleteVertexArray(this.vao);
    gl.deleteFramebuffer(this.target);
  }
}
