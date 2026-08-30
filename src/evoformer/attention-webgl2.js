/**
 * Flash attention on WebGL2, for a fallback backend.
 *
 * WHY THIS ONE. `attention.flash` is the largest single cost in an Evoformer
 * block - 58% of it at 221 residues, and more now that the triangle contraction
 * has been fixed. It is also the kernel whose FEASIBILITY on WebGL2 is in
 * genuine doubt, because it carries sequential state: an online softmax with a
 * running maximum and sum threaded through the key loop. A port that only
 * handled the dense, stateless kernels would prove nothing about whether a
 * WebGL2 fallback is possible at all.
 *
 * 🔴 EIGHT RENDER TARGETS ARE WHAT MAKES THIS WORK. A fragment writes one
 * vec4, which is four of the head's thirty-two channels, so the obvious port
 * needs eight fragments per query - and each of them would recompute the SAME
 * q.k dot product, eight times over. WebGL2 guarantees at least four draw
 * buffers and this machine reports eight, so one fragment can own all thirty-
 * two channels across eight attachments and compute each dot product exactly
 * once. That is the difference between a port that is 8x redundant and one
 * that does the same arithmetic as the compute shader.
 *
 * WHAT IT DOES NOT NEED. The WGSL kernel spends a 32-lane tree reduction with
 * a workgroup barrier at every step to form one dot product, then two more
 * barriers to publish the softmax state - roughly eight barriers per key, per
 * query. A fragment has no one to synchronise with: the dot product is a loop
 * in registers and the running state never leaves them. Whether that pays for
 * the loss of the shared K/V tile is the question this file exists to answer.
 */

const TEXTURE_WIDTH = 2048;
const CHANNELS_PER_TEXEL = 4;

/** Channels in a head, fixed: the model uses 32 and the eight-target trick assumes it. */
export const HEAD_DIM = 32;
const VECTORS_PER_HEAD = HEAD_DIM / CHANNELS_PER_TEXEL;   // 8

function textureShape(elements) {
  const texels = Math.ceil(elements / CHANNELS_PER_TEXEL);
  return { width: TEXTURE_WIDTH, height: Math.max(1, Math.ceil(texels / TEXTURE_WIDTH)) };
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

function flashShader(queries, heads, width) {
  const outputs = Array.from({ length: VECTORS_PER_HEAD },
    (_, t) => `layout(location = ${t}) out vec4 result${t};`).join("\n");
  // ...unrolled by generation rather than by `#pragma unroll`, so the eight
  // accumulators are eight named registers and never an indexed array. GLSL ES
  // permits indexing a local array only by a constant expression, and a driver
  // that spills one to memory would put the whole point of this kernel there.
  const declare = (name, init) => Array.from({ length: VECTORS_PER_HEAD },
    (_, t) => `  vec4 ${name}${t} = ${init};`).join("\n");
  const each = (body) => Array.from({ length: VECTORS_PER_HEAD },
    (_, t) => `    ${body(t)}`).join("\n");

  return `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

const int Q = ${queries};
const int HEADS = ${heads};
const int HD4 = ${VECTORS_PER_HEAD};
const int W = ${width};

uniform sampler2D queryTex;
uniform sampler2D keyTex;
uniform sampler2D valueTex;
uniform sampler2D gateTex;
uniform sampler2D maskTex;
uniform sampler2D pairTex;
uniform int hasPairBias;
uniform int transposeMask;
uniform int batchCount;
uniform float bias;

${outputs}

vec4 fetch4(sampler2D tex, int vectorIndex) {
  return texelFetch(tex, ivec2(vectorIndex % W, vectorIndex / W), 0);
}

float fetch1(sampler2D tex, int index) {
  vec4 texel = fetch4(tex, index >> 2);
  int component = index & 3;
  return component == 0 ? texel.x
       : component == 1 ? texel.y
       : component == 2 ? texel.z : texel.w;
}

void main() {
  int q = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  int head = row % HEADS;
  int batch = row / HEADS;
  int queryBase = ((batch * Q + q) * HEADS + head) * HD4;

${declare("q", "fetch4(queryTex, queryBase + 0)").split("\n")
    .map((line, t) => line.replace("+ 0)", `+ ${t})`)).join("\n")}
${declare("acc", "vec4(0.0)")}

  float runningMax = -1e30;
  float runningSum = 0.0;

  for (int k = 0; k < Q; ++k) {
    int keyBase = ((batch * Q + k) * HEADS + head) * HD4;
    float score = 0.0;
${each((t) => `score += dot(q${t}, fetch4(keyTex, keyBase + ${t}));`)}

    int maskIndex = transposeMask == 0 ? batch * Q + k : k * batchCount + batch;
    float logit = score + 1e9 * (fetch1(maskTex, maskIndex) - 1.0);
    if (hasPairBias != 0) logit += fetch1(pairTex, (head * Q + q) * Q + k);
    logit = clamp(logit, -1e8, 1e8);

    // ...the same online softmax as the compute shader, kept in registers.
    float newMax = max(runningMax, logit);
    float previousScale = exp(runningMax - newMax);
    float weight = exp(logit - newMax);
    runningSum = runningSum * previousScale + weight;
    runningMax = newMax;
${each((t) => `acc${t} = acc${t} * previousScale + weight * fetch4(valueTex, keyBase + ${t});`)}
  }

${each((t) => `result${t} = (acc${t} / runningSum) * fetch4(gateTex, queryBase + ${t}) + bias;`)}
}`;
}

export function createWebGL2Context(canvas) {
  const gl = (canvas ?? new OffscreenCanvas(1, 1)).getContext("webgl2");
  if (gl === null) throw new Error("this browser has no WebGL2");
  if (gl.getExtension("EXT_color_buffer_float") === null) {
    throw new Error("WebGL2 here cannot render to float textures (EXT_color_buffer_float)");
  }
  return gl;
}

export class AttentionFlashWebGL2 {
  gl;
  #programs = new Map();

  constructor(gl) {
    this.gl = gl;
    const targets = gl.getParameter(gl.MAX_DRAW_BUFFERS);
    if (targets < VECTORS_PER_HEAD) {
      throw new Error(
        `this WebGL2 needs ${VECTORS_PER_HEAD} draw buffers for a ${HEAD_DIM}-channel head, has ${targets}`,
      );
    }
  }

  #program(queries, heads) {
    const key = `${queries}:${heads}`;
    const cached = this.#programs.get(key);
    if (cached !== undefined) return cached;
    const gl = this.gl;
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, flashShader(queries, heads, TEXTURE_WIDTH)));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`WebGL2 program failed to link: ${gl.getProgramInfoLog(program)}`);
    }
    const uniform = (name) => gl.getUniformLocation(program, name);
    const entry = {
      program,
      samplers: ["queryTex", "keyTex", "valueTex", "gateTex", "maskTex", "pairTex"].map(uniform),
      hasPairBias: uniform("hasPairBias"),
      transposeMask: uniform("transposeMask"),
      batchCount: uniform("batchCount"),
      bias: uniform("bias"),
    };
    this.#programs.set(key, entry);
    return entry;
  }

  #upload(values) {
    const gl = this.gl;
    const { width, height } = textureShape(values.length);
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
    return texture;
  }

  /**
   * Upload the inputs and build the eight-target framebuffer.
   *
   * The output grid is one fragment per (batch, head, query): x is the query,
   * y is batch * heads + head. Attachment t carries channels 4t..4t+3, which is
   * why reading it back is a gather rather than a copy - see `read`.
   */
  prepare(input) {
    const gl = this.gl;
    const { query, key, value, gate, mask, batch, queries, heads } = input;
    const headDim = input.headDim ?? HEAD_DIM;
    if (headDim !== HEAD_DIM) throw new RangeError(`this kernel is specialised for head_dim ${HEAD_DIM}`);
    const elements = batch * queries * heads * headDim;
    for (const [name, tensor] of [["query", query], ["key", key], ["value", value], ["gate", gate]]) {
      if (tensor.length !== elements) {
        throw new RangeError(`${name} should hold ${elements} elements, has ${tensor.length}`);
      }
    }
    const width = queries;
    const height = batch * heads;
    const limit = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    if (width > limit || height > limit) {
      throw new RangeError(`attention grid ${width}x${height} exceeds MAX_TEXTURE_SIZE ${limit}`);
    }

    const textures = {
      query: this.#upload(query), key: this.#upload(key), value: this.#upload(value),
      gate: this.#upload(gate), mask: this.#upload(mask),
      pair: this.#upload(input.pairBias ?? new Float32Array(4)),
    };

    const targets = [];
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    for (let t = 0; t < VECTORS_PER_HEAD; t += 1) {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      for (const parameter of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) {
        gl.texParameteri(gl.TEXTURE_2D, parameter, gl.NEAREST);
      }
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + t, gl.TEXTURE_2D, texture, 0);
      targets.push(texture);
    }
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`eight-target framebuffer is incomplete (0x${status.toString(16)})`);
    }
    gl.drawBuffers(targets.map((_, t) => gl.COLOR_ATTACHMENT0 + t));

    const shader = this.#program(queries, heads);
    gl.useProgram(shader.program);
    const vertices = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const position = gl.getAttribLocation(shader.program, "position");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

    return {
      shader, textures, targets, framebuffer, vertices, vao, width, height,
      batch, queries, heads, headDim,
      hasPairBias: input.pairBias !== undefined ? 1 : 0,
      transpose: input.transpose ? 1 : 0,
    };
  }

  #bind(state) {
    const gl = this.gl;
    gl.useProgram(state.shader.program);
    gl.bindVertexArray(state.vao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, state.framebuffer);
    gl.drawBuffers(state.targets.map((_, t) => gl.COLOR_ATTACHMENT0 + t));
    const order = [state.textures.query, state.textures.key, state.textures.value,
      state.textures.gate, state.textures.mask, state.textures.pair];
    order.forEach((texture, unit) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.uniform1i(state.shader.samplers[unit], unit);
    });
    gl.uniform1i(state.shader.hasPairBias, state.hasPairBias);
    gl.uniform1i(state.shader.transposeMask, state.transpose);
    gl.uniform1i(state.shader.batchCount, state.batch);
    gl.viewport(0, 0, state.width, state.height);
  }

  /** Draw the attention into the prepared targets. Nothing is read back. */
  dispatch(state, bias = 0) {
    const gl = this.gl;
    this.#bind(state);
    gl.uniform1f(state.shader.bias, bias);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /**
   * Read back in the compute shader's layout: [batch][query][head][channel].
   *
   * 🔴 A GATHER, NOT A COPY. Attachment t holds channels 4t..4t+3 for every
   * (batch, head, query), so the eight reads have to be interleaved back
   * together. In a real fallback pipeline the pass downstream would read the
   * eight textures directly and this would not exist; it is here so the result
   * can be compared element for element against WebGPU.
   */
  read(state) {
    const gl = this.gl;
    this.#bind(state);
    gl.uniform1f(state.shader.bias, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const { width, height, batch, queries, heads, headDim } = state;
    const output = new Float32Array(batch * queries * heads * headDim);
    const band = new Float32Array(width * height * CHANNELS_PER_TEXEL);
    for (let t = 0; t < VECTORS_PER_HEAD; t += 1) {
      gl.readBuffer(gl.COLOR_ATTACHMENT0 + t);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, band);
      for (let row = 0; row < height; row += 1) {
        const head = row % heads;
        const batchIndex = (row - head) / heads;
        for (let q = 0; q < queries; q += 1) {
          const source = (row * width + q) * CHANNELS_PER_TEXEL;
          const target = ((batchIndex * queries + q) * heads + head) * headDim + t * CHANNELS_PER_TEXEL;
          for (let m = 0; m < CHANNELS_PER_TEXEL; m += 1) output[target + m] = band[source + m];
        }
      }
    }
    return output;
  }

  release(state) {
    const gl = this.gl;
    for (const texture of Object.values(state.textures)) gl.deleteTexture(texture);
    for (const texture of state.targets) gl.deleteTexture(texture);
    gl.deleteFramebuffer(state.framebuffer);
    gl.deleteBuffer(state.vertices);
    gl.deleteVertexArray(state.vao);
  }

  run(input) {
    const state = this.prepare(input);
    try { return this.read(state); } finally { this.release(state); }
  }
}
