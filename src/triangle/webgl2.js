/**
 * The triangle contraction on WebGL2, for comparison against the WebGPU one.
 *
 * WHY THIS EXISTS. Profiling puts 80% of an Evoformer block in two kernels, and
 * `triangle.*.contract` is the simpler of them - 22% of block time at 221
 * residues, and the one operation in the model that already has a CPU reference
 * to check against. If a WebGL2 port of the Evoformer is worth doing at all,
 * this is the kernel that says so, and it says it in a day rather than a week.
 *
 * 🔴 WEBGL2 HAS NO COMPUTE SHADERS. Everything below follows from that:
 *
 *   - The work is done by a FRAGMENT shader. One draw covers a rectangle of
 *     output texels and every fragment computes one output element, so the
 *     "dispatch" is a triangle that covers the framebuffer.
 *   - There are no storage buffers, so the tensors live in textures and every
 *     linear index has to be decoded into a texel coordinate.
 *   - There is no workgroup shared memory, so the tiling the WGSL kernel uses -
 *     8x8 staging in `var<workgroup>` with two barriers a step - has no
 *     equivalent. This reads straight from texture memory and leans on the
 *     texture cache instead. That is the honest shape of the comparison: not
 *     the same algorithm in two languages, but the same MATHS under two
 *     different sets of constraints.
 *
 * WIDTH IS A POWER OF TWO on purpose. A texel is addressed as
 * (index % width, index / width), and with an arbitrary width that is an
 * integer division per access, twice per inner-loop step. A power of two lets
 * the compiler shift and mask instead.
 */

/** Texel rows this wide; see the note above. Must be a power of two. */
const TEXTURE_WIDTH = 2048;

/**
 * 🔴 FOUR FLOATS TO A TEXEL, AND NO REPACKING TO GET THERE. The tensors are
 * laid out with channel fastest, so four consecutive channels are already four
 * consecutive floats - exactly one RGBA32F texel. The same Float32Array uploads
 * as RGBA without being rearranged, and the shader gets four values per fetch.
 */
const CHANNELS_PER_TEXEL = 4;

function textureShape(elements) {
  const texels = Math.ceil(elements / CHANNELS_PER_TEXEL);
  return { width: TEXTURE_WIDTH, height: Math.ceil(texels / TEXTURE_WIDTH), texels };
}

/**
 * Channel-minor [pair][channel] to channel-major [channel/4][pair], four to a
 * texel. Done once on the CPU here; in a pipeline the pass that produced a and
 * b would write this layout directly, exactly as projectAB now does on the
 * WGSL side.
 */
function toChannelMajor(values, pairs, channels) {
  const groups = channels / CHANNELS_PER_TEXEL;
  const out = new Float32Array(values.length);
  for (let pair = 0; pair < pairs; pair += 1) {
    for (let group = 0; group < groups; group += 1) {
      const source = pair * channels + group * CHANNELS_PER_TEXEL;
      const target = (group * pairs + pair) * CHANNELS_PER_TEXEL;
      for (let m = 0; m < CHANNELS_PER_TEXEL; m += 1) out[target + m] = values[source + m];
    }
  }
  return out;
}

function compile(gl, type, source) {
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

const VERTEX_SHADER = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`;

/**
 * The contraction, with the shape baked in.
 *
 * L and CH are compile-time constants rather than uniforms so the index
 * arithmetic folds: `h = index % CH` becomes a mask when CH is a power of two,
 * and the loop bound is known when the shader is optimised.
 */
function contractShader(length, channels, width, direction, layout = "channel-minor") {
  const aAt = (k) => direction === "outgoing" ? `texel(i, ${k}, h4)` : `texel(${k}, i, h4)`;
  const bAt = (k) => direction === "outgoing" ? `texel(j, ${k}, h4)` : `texel(${k}, j, h4)`;
  const loadA = aAt("k"), loadB = bAt("k");
  // 🔴 CHANNEL-MAJOR IS WORTH 14x ON THE WGSL SIDE AND HURTS HERE. It was added
  // to compare the two APIs on equal terms and the answer was that they are not
  // equal in this respect: WebGL2 never had the problem it fixes. A fragment
  // computes a vec4, so four consecutive channels already arrive in one texel,
  // and neighbouring fragments already cover consecutive channel groups - the
  // reads are coalesced by construction. Rearranging into per-channel planes
  // takes that away and puts a whole L*L plane between neighbouring fragments.
  // Measured on an M2, interleaved, bitwise-identical output:
  //   L=128  0.98 -> 1.25 ms     L=192  6.22 -> 8.32 ms
  //   L=256  16.62 -> 41.72 ms   (3.5x WORSE)
  // Kept because the negative result is the point: it is why the WGSL port
  // looked slow against this one before the WGSL layout was fixed.
  const index = layout === "channel-major"
    ? "int texel(int x, int y, int c) { return c * (L * L) + x * L + y; }"
    : "int texel(int x, int y, int c) { return (x * L + y) * CH4 + c; }";

  return `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

const int L = ${length};
const int CH4 = ${channels / 4};        // channels per pair, four to a texel
const int W = ${width};

uniform sampler2D aTex;
uniform sampler2D bTex;
uniform int rowOffset;      // which band of the output this draw covers
uniform float bias;         // 0 in use; a benchmark varies it, see dispatch()
out vec4 result;

${index}
vec4 fetch(sampler2D tex, int index) {
  return texelFetch(tex, ivec2(index % W, index / W), 0);
}

void main() {
  int linear = (int(gl_FragCoord.y) + rowOffset) * W + int(gl_FragCoord.x);
  if (linear >= L * L * CH4) { result = vec4(0.0); return; }
  int h4 = linear % CH4;
  int pair = linear / CH4;
  int j = pair % L;
  int i = pair / L;
  vec4 sum = vec4(0.0);
  // 🔴 NOT UNROLLED, AND THAT WAS MEASURED. Reading four steps of k before
  // multiplying any of them - so the fetches overlap instead of forming a
  // dependent chain - is the obvious latency hiding for a shader with no
  // workgroup to swap to. It made this kernel SLOWER at four of five lengths
  // (2.5x at 96 residues, 1.3x at 128), presumably because the extra live
  // registers cost more occupancy than the overlap bought. The straight loop
  // stays until something measures better.
  for (int k = 0; k < L; ++k) {
    sum += fetch(aTex, ${loadA}) * fetch(bTex, ${loadB});
  }
  result = sum + bias;
}`;
}

/**
 * A WebGL2 context that can render to float textures, or an explanation.
 *
 * EXT_color_buffer_float is what makes an R32F framebuffer legal to draw into.
 * Without it the whole approach is unavailable, and saying so by name is more
 * use than a link error deep in a draw call.
 */
export function createWebGL2Context(canvas) {
  const gl = (canvas ?? new OffscreenCanvas(1, 1)).getContext("webgl2");
  if (gl === null) throw new Error("this browser has no WebGL2");
  if (gl.getExtension("EXT_color_buffer_float") === null) {
    throw new Error("WebGL2 here cannot render to float textures (EXT_color_buffer_float)");
  }
  return gl;
}

export class TriangleContractWebGL2 {
  gl;
  #programs = new Map();

  constructor(gl) { this.gl = gl; }

  #program(length, channels, direction, layout) {
    const key = `${length}:${channels}:${direction}:${layout}`;
    const cached = this.#programs.get(key);
    if (cached !== undefined) return cached;
    const gl = this.gl;
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER,
      contractShader(length, channels, TEXTURE_WIDTH, direction, layout)));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`WebGL2 program failed to link: ${gl.getProgramInfoLog(program)}`);
    }
    const entry = {
      program,
      aTex: gl.getUniformLocation(program, "aTex"),
      bTex: gl.getUniformLocation(program, "bTex"),
      rowOffset: gl.getUniformLocation(program, "rowOffset"),
      bias: gl.getUniformLocation(program, "bias"),
    };
    this.#programs.set(key, entry);
    return entry;
  }

  #upload(values) {
    const gl = this.gl;
    const { width, height } = textureShape(values.length);
    // ...PADDED TO THE RECTANGLE. A texture is width * height texels and the
    // upload has to fill all of them; the shader's bounds check is what keeps
    // the padding from reaching the output.
    const needed = width * height * CHANNELS_PER_TEXEL;
    const padded = values.length === needed
      ? values : (() => { const p = new Float32Array(needed); p.set(values); return p; })();
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, padded);
    for (const parameter of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) {
      gl.texParameteri(gl.TEXTURE_2D, parameter, gl.NEAREST);
    }
    for (const parameter of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) {
      gl.texParameteri(gl.TEXTURE_2D, parameter, gl.CLAMP_TO_EDGE);
    }
    return { texture, width, height };
  }

  /**
   * Everything that does not change between dispatches: the inputs uploaded as
   * textures, the program, the target and the full-screen triangle.
   *
   * 🔴 SEPARATE FROM THE DRAW, and not only for tidiness. In the Evoformer the
   * inputs to this kernel are the output of the one before it - already on the
   * GPU, never travelling - so a measurement that re-uploads them each time is
   * timing 16 MiB of transfer against a WebGPU path whose buffers were created
   * once. Preparing and dispatching separately is what makes the two comparable,
   * and it is also how a pipeline would actually call this.
   */
  prepare(input) {
    const gl = this.gl;
    const { a, b, length, channels } = input;
    const direction = input.direction ?? "outgoing";
    const elements = length * length * channels;
    if (a.length !== elements || b.length !== elements) {
      throw new RangeError(`triangle contract expects ${elements} elements in a and b`);
    }
    if (channels % CHANNELS_PER_TEXEL !== 0) {
      throw new RangeError(`channels must be a multiple of ${CHANNELS_PER_TEXEL}, got ${channels}`);
    }
    const layout = input.layout ?? "channel-minor";
    const pack = (values) => layout === "channel-major"
      ? toChannelMajor(values, length * length, channels) : values;
    const aTexture = this.#upload(pack(a));
    const bTexture = this.#upload(pack(b));
    const out = textureShape(elements);
    // 🔴 THE OUTPUT IS DRAWN IN BANDS. A framebuffer is at most
    // MAX_TEXTURE_SIZE on a side, and L*L*CH / 2048 rows passes that at
    // moderate lengths - 48,841 pair rows at 221 residues. Each band is its own
    // draw with a row offset: the fragment-shader equivalent of the grid
    // folding the WGSL side does.
    const bandRows = Math.min(out.height, gl.getParameter(gl.MAX_TEXTURE_SIZE));
    const target = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, target);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, out.width, bandRows, 0, gl.RGBA, gl.FLOAT, null);
    for (const parameter of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) {
      gl.texParameteri(gl.TEXTURE_2D, parameter, gl.NEAREST);
    }
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);

    const shader = this.#program(length, channels, direction, layout);
    gl.useProgram(shader.program);
    const vertices = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
    // ONE TRIANGLE, not two: a single oversized triangle covers the viewport
    // without the seam down the diagonal that two of them share.
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(shader.program, "position");
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    gl.uniform1i(shader.aTex, 0);
    gl.uniform1i(shader.bTex, 1);
    return { shader, aTexture, bTexture, target, framebuffer, vertices, vao, out, bandRows,
      elements, layout };
  }

  /**
   * Make this state current.
   *
   * 🔴 BOTH THE DISPATCH AND THE READ NEED THIS. When only dispatch() bound the
   * input textures, read() drew with whatever was bound from before and
   * returned a plausible-looking array of the right shape that was entirely
   * wrong - relative RMS 1.0 against the reference, which is to say no signal
   * at all. A shared binding step is what stops the two paths from drifting.
   */
  #bind(state) {
    const gl = this.gl;
    gl.useProgram(state.shader.program);
    gl.bindVertexArray(state.vao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, state.framebuffer);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, state.aTexture.texture);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, state.bTexture.texture);
  }

  /**
   * Draw the contraction into the prepared target. Nothing is read back.
   *
   * `bias` is added to every element and is zero in use. It exists because a
   * driver may drop a draw whose state and target are identical to the last
   * one, which makes repeated dispatches unmeasurable; varying it makes each
   * draw observably distinct.
   */
  dispatch(state, bias = 0) {
    const gl = this.gl;
    this.#bind(state);
    gl.uniform1f(state.shader.bias, bias);
    for (let row = 0; row < state.out.height; row += state.bandRows) {
      const rows = Math.min(state.bandRows, state.out.height - row);
      gl.viewport(0, 0, state.out.width, rows);
      gl.uniform1i(state.shader.rowOffset, row);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }

  /** Read the prepared target back as a Float32Array. */
  read(state) {
    const gl = this.gl;
    this.#bind(state);
    gl.uniform1f(state.shader.bias, 0);
    const output = new Float32Array(state.out.width * state.out.height * CHANNELS_PER_TEXEL);
    const band = new Float32Array(state.out.width * state.bandRows * CHANNELS_PER_TEXEL);
    for (let row = 0; row < state.out.height; row += state.bandRows) {
      const rows = Math.min(state.bandRows, state.out.height - row);
      // ...re-drawn per band because the target only holds one band at a time.
      gl.viewport(0, 0, state.out.width, rows);
      gl.uniform1i(state.shader.rowOffset, row);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.readPixels(0, 0, state.out.width, rows, gl.RGBA, gl.FLOAT, band);
      output.set(band.subarray(0, state.out.width * rows * CHANNELS_PER_TEXEL),
        row * state.out.width * CHANNELS_PER_TEXEL);
    }
    return output.subarray(0, state.elements);
  }

  release(state) {
    const gl = this.gl;
    for (const texture of [state.aTexture.texture, state.bTexture.texture, state.target]) {
      gl.deleteTexture(texture);
    }
    gl.deleteFramebuffer(state.framebuffer);
    gl.deleteBuffer(state.vertices);
    gl.deleteVertexArray(state.vao);
  }

  /** Prepare, dispatch, read and release - the whole thing, for a single call. */
  run(input) {
    const state = this.prepare(input);
    try { return this.read(state); } finally { this.release(state); }
  }
}
