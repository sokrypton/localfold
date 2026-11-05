/**
 * A GEMM in a VERTEX shader through transform feedback, against the same GEMM
 * in a fragment shader - because the output budget is what binds this backend.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-feedback-gemm.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-feedback-gemm.js --rows=8 --outs=12
 *
 * 🔴 THE OUTPUT BUDGET IS THE ONE FACTOR EVERYTHING ELSE HERE REDUCES TO. A
 * fragment invocation writes `MAX_DRAW_BUFFERS * 4` = **32 floats**, which caps
 * a register tile's intensity at `8 * R * C / (R + C)` = 21.33 flops a fetch4,
 * and webgl/README.md multiplies that by this device's measured issue rate at a
 * realistic working set (25.7 G fetch4/s) for a ceiling of about **550
 * GFLOP/s** where parity with the WGSL bioemu2 fold needs **1303**. Every
 * mechanism that campaign measured - fragment order, tile mappings, the output
 * aspect, k splits, the block shape, float16 operands, packed operands - moves
 * the kernel WITHIN that ceiling. This asks whether WebGL2 has a different
 * ceiling anywhere.
 *
 * 🔴 IT DOES, ON PAPER: `probe-webgl-limits.js` reads
 * `MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS` as **128** on this device,
 * four times the fragment budget, and `MAX_VERTEX_TEXTURE_IMAGE_UNITS` as 16 so
 * the vertex stage can read its operands. At 128 outputs the smallest `R + C`
 * is 24 at 8 by 16, so the cap is **42.67** flops a fetch4 and the ceiling
 * about **1100 GFLOP/s**. The varyings cap it in practice -
 * `MAX_VERTEX_OUTPUT_COMPONENTS` is 120 - so the usable tile is a little
 * smaller, and the probe takes `--rows` and `--outs` rather than assuming one.
 *
 * 🔴 AND THE OTHER HALF IS WHETHER A VERTEX INVOCATION ISSUES FETCHES AS FAST
 * AS A FRAGMENT ONE, which no arithmetic answers. Both arms do the SAME total
 * arithmetic over the SAME operands, so the ratio is the answer: the fragment
 * arm is `probe-webgl-gemm-ceiling.js`'s kernel at this tile, and the feedback
 * arm is the same k loop in a vertex shader with `RASTERIZER_DISCARD` on.
 *
 * 🔴 MEASURED: 1.06-1.13x AT A 64 KiB WORKING SET AND A LOSS EVERYWHERE ELSE.
 * Five runs at `--window=4096` read 1.13, 1.06, 1.07, 1.07 and 1.10 - feedback
 * 543-560 GFLOP/s at 14.6 G fetch4/s against the fragment path's 494-507 at
 * 23.1, each arm's own spread 1.04-1.16. With shared operands it is **0.54x**
 * (1074 against 1986, because the fragment stage reaches 93.1 G fetch4/s there
 * and the vertex stage 28.0), and at 256 KiB **0.41x** (197 against 482). The
 * intensity gain is real; the vertex stage's issue rate is where it goes, 1.6x
 * lower at 64 KiB and 4.4x at 256. The band where it wins is the one that
 * matters - the shipped projection's working set is about 75 KiB - so the
 * output budget IS the binding constraint, raising it is possible, and raising
 * it buys about a tenth before the buffer-to-texture copy every kernel would
 * owe, since nothing in WebGL2 samples a buffer.
 *
 * 🔴 ITS SHARED-OPERAND ARM IS TINY AND READ BY EVERY INVOCATION, as that
 * probe's is.
 * This measures the ISSUE CEILING, not a real kernel: a real GEMM's operands do
 * not fit in cache and webgl/README.md records the three points of that curve
 * (about 88 G fetch4/s shared, 34 through 4 KiB, 25.7 through 64 KiB). What
 * transfers is the RATIO between the two stages at the same locality.
 */
import { createWebGL2Context } from "../../webgl/index.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const compile = (gl, kind, source) => {
  const shader = gl.createShader(kind);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`${gl.getShaderInfoLog(shader)}\n${source}`);
  }
  return shader;
};

/**
 * The k loop, shared by both arms so the arithmetic cannot differ.
 *
 * 🔴 `window` IS THE WHOLE POINT OF THE SECOND COMPARISON. With it zero the
 * operands are a few KiB that every invocation reads, which is the regime
 * `probe-webgl-gemm-ceiling.js` measures and where the fragment stage reaches
 * about 88 G fetch4/s - a rate no real kernel sees, because real operands do
 * not fit in cache (the shipped projection measures **20.4**). With it set,
 * each invocation reads a different slice of a `window`-texel operand, so the
 * working set is `window` and the two stages are compared at a locality a fold
 * actually has. Comparing them only at shared operands would flatter whichever
 * stage caches better and say nothing about a fold.
 */
const loop = (k, rows, outs, window) => `
  for (int kk = 0; kk < ${k}; kk += 4) {
${Array.from({ length: rows }, (_, r) => (window > 0
    ? `    vec4 a${r} = texelFetch(xTex, ivec2((slot + ${r} * ${k} + kk) >> 2 & ${window - 1}, 0), 0);`
    : `    vec4 a${r} = texelFetch(xTex, ivec2((${r} * ${k} + kk) >> 2, 0), 0);`)).join("\n")}
${Array.from({ length: outs }, (_, c) => (window > 0
    ? `    vec4 b${c} = texelFetch(wTex, ivec2((slot + ${c} * ${k} + kk) >> 2 & ${window - 1}, 0), 0);`
    : `    vec4 b${c} = texelFetch(wTex, ivec2((${c} * ${k} + kk) >> 2, 0), 0);`)).join("\n")}
${Array.from({ length: rows }, (_, r) => Array.from({ length: outs }, (_, c) =>
    `    s${r}_${c} += dot(a${r}, b${c});`).join("\n")).join("\n")}
  }`;

export async function main(device, args) {
  const k = Number(option(args, "k", "256"));
  // A power of two so the shader can mask rather than divide; 0 shares the
  // operands, as the gemm-ceiling probe does.
  const window = Number(option(args, "window", "0"));
  if (window !== 0 && (window & (window - 1)) !== 0) {
    throw new Error(`--window=${window} is not a power of two`);
  }
  const rows = Number(option(args, "rows", "8"));
  const outs = Number(option(args, "outs", "12"));
  const rounds = Number(option(args, "rounds", "4"));
  const iterations = Number(option(args, "iterations", "8"));
  const gl = createWebGL2Context();
  const outputs = rows * outs;
  const varyingComponents = Math.ceil(outputs / 4) * 4;
  const maxVarying = gl.getParameter(gl.MAX_VERTEX_OUTPUT_COMPONENTS);
  const maxInterleaved = gl.getParameter(gl.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS);
  if (varyingComponents > Math.min(maxVarying, maxInterleaved)) {
    throw new Error(`${outputs} outputs needs ${varyingComponents} components, past `
      + `min(MAX_VERTEX_OUTPUT_COMPONENTS ${maxVarying}, `
      + `MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS ${maxInterleaved})`);
  }
  if (outs % 4 !== 0) throw new Error(`outs ${outs} is not a multiple of four`);

  const fill = (count, seed) => {
    const values = new Float32Array(count);
    let state = seed | 0;
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      values[i] = ((state >>> 8) / 0x7fffff - 0.5) * 0.25;
    }
    return values;
  };
  const texture = (values) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, values.length / 4, 1, 0,
      gl.RGBA, gl.FLOAT, values);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return t;
  };
  // Sized to the window where there is one, so the reads really do span it.
  const xTexels = window > 0 ? window : (rows * k) / 4;
  const wTexels = window > 0 ? window : (outs * k) / 4;
  const xTex = texture(fill(xTexels * 4, 7));
  const wTex = texture(fill(wTexels * 4, 11));

  // 🔴 EACH ARM AT ITS OWN BEST TILE, WHICH IS THE COMPARISON AND NOT A FLAW IN
  // IT. The fragment path cannot express 96 outputs an invocation - that is the
  // constraint under test - so it runs the 8 by 4 the rest of this backend runs
  // and simply needs more invocations for the same work. Both arms produce the
  // SAME total outputs and the SAME total flops; what differs is the FETCH
  // count, because a bigger tile amortises its operands over more arithmetic.
  // That difference is the mechanism, so the probe reports both.
  const totalOutputs = 512 * 512 * 32;
  const invocations = Math.floor(totalOutputs / outputs);
  const flops = invocations * outputs * k * 2;
  const fetches = invocations * (rows + outs) * (k / 4);
  const fragmentRows = 8;
  const fragmentOuts = 4;
  const fragmentTile = fragmentRows * fragmentOuts;
  const fragmentInvocations = Math.floor(totalOutputs / fragmentTile);
  const fragmentFetches = fragmentInvocations * (fragmentRows + fragmentOuts) * (k / 4);

  const accumulators = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < outs; c += 1) accumulators.push(`s${r}_${c}`);
  }
  const quads = Array.from({ length: varyingComponents / 4 }, (_, q) => `out${q}`);

  // ---- the feedback arm -------------------------------------------------
  const feedbackVertex = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D xTex;
uniform sampler2D wTex;
${quads.map((n) => `out vec4 ${n};`).join("\n")}
void main() {
${accumulators.map((a) => `  float ${a} = 0.0;`).join("\n")}
  // gl_VertexID keeps the operands shared and the compiler honest: without a
  // per-invocation term the whole loop hoists and the arm measures nothing.
  float bias = float(gl_VertexID & 1) * 1e-8;
  int slot = gl_VertexID * ${rows * k};
${loop(k, rows, outs, window)}
${quads.map((n, q) => `  ${n} = vec4(${[0, 1, 2, 3].map((c) => {
    const flat = q * 4 + c;
    return flat < accumulators.length ? `${accumulators[flat]} + bias` : "0.0";
  }).join(", ")});`).join("\n")}
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}`;
  const trivialFragment = `#version 300 es
precision highp float;
out vec4 result;
void main() { result = vec4(0.0); }`;

  const feedbackProgram = gl.createProgram();
  gl.attachShader(feedbackProgram, compile(gl, gl.VERTEX_SHADER, feedbackVertex));
  gl.attachShader(feedbackProgram, compile(gl, gl.FRAGMENT_SHADER, trivialFragment));
  gl.transformFeedbackVaryings(feedbackProgram, quads, gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(feedbackProgram);
  if (!gl.getProgramParameter(feedbackProgram, gl.LINK_STATUS)) {
    throw new Error(`link feedback: ${gl.getProgramInfoLog(feedbackProgram)}`);
  }

  // One buffer per iteration, as the fragment probes use a ring of targets.
  const perInvocationBytes = varyingComponents * 4;
  const chunk = Math.min(invocations, Math.floor((64 << 20) / perInvocationBytes));
  // 🔴 SIZED THROUGH `COPY_WRITE_BUFFER` AND LEFT BOUND TO NOTHING, because
  // WebGL2 refuses a buffer that is bound to a transform-feedback point AND to
  // a non-feedback point at the same time - `GL_INVALID_OPERATION`, a draw that
  // silently does not happen, and a probe that reports 171 TFLOP/s. Sizing them
  // through `ARRAY_BUFFER` and leaving them there is exactly that conflict.
  const buffers = Array.from({ length: iterations }, () => {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, buffer);
    gl.bufferData(gl.COPY_WRITE_BUFFER, chunk * perInvocationBytes, gl.STATIC_COPY);
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, null);
    return buffer;
  });
  const feedback = gl.createTransformFeedback();
  const passes = Math.ceil(invocations / chunk);

  const runFeedback = (i) => {
    gl.useProgram(feedbackProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, xTex);
    gl.uniform1i(gl.getUniformLocation(feedbackProgram, "xTex"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, wTex);
    gl.uniform1i(gl.getUniformLocation(feedbackProgram, "wTex"), 1);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, feedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buffers[i % buffers.length]);
    gl.enable(gl.RASTERIZER_DISCARD);
    for (let pass = 0; pass < passes; pass += 1) {
      const count = Math.min(chunk, invocations - pass * chunk);
      gl.beginTransformFeedback(gl.POINTS);
      gl.drawArrays(gl.POINTS, 0, count);
      gl.endTransformFeedback();
    }
    gl.disable(gl.RASTERIZER_DISCARD);
    // Released from the feedback point so the read below may bind it.
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    // 🔴 A SILENT GL ERROR HERE READS AS 171 TFLOP/s. The first version of this
    // probe reported 0.025 ms and 171,798 GFLOP/s with a spread of 11 - which
    // is not a fast GEMM, it is a draw that did not happen. Transform feedback
    // has several ways to be refused (a buffer too small, a varying list the
    // link did not honour, a begin without a bound buffer) and every one of
    // them is a no-op plus an error code.
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      throw new Error(`transform feedback: GL error 0x${error.toString(16)} after the draw `
        + `(${invocations} invocations, ${perInvocationBytes} bytes each, ${passes} passes)`);
    }
  };
  const syncFeedback = () => {
    const out = new Float32Array(4);
    gl.bindBuffer(gl.COPY_READ_BUFFER, buffers[0]);
    gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, out);
    gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    return out[0];
  };

  // ---- the fragment arm -------------------------------------------------
  const fragmentVertex = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`;
  const targets = fragmentTile / 4;
  const fragmentAccumulators = [];
  for (let r = 0; r < fragmentRows; r += 1) {
    for (let c = 0; c < fragmentOuts; c += 1) fragmentAccumulators.push(`s${r}_${c}`);
  }
  const fragmentShader = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D xTex;
uniform sampler2D wTex;
${Array.from({ length: targets }, (_, t) =>
  `layout(location = ${t}) out vec4 result${t};`).join("\n")}
void main() {
${fragmentAccumulators.map((a) => `  float ${a} = 0.0;`).join("\n")}
  float bias = float(int(gl_FragCoord.x) & 1) * 1e-8;
  int slot = (int(gl_FragCoord.y) * ${1 << 20} + int(gl_FragCoord.x)) * ${fragmentRows * k};
${loop(k, fragmentRows, fragmentOuts, window)}
${Array.from({ length: targets }, (_, t) =>
  `  result${t} = vec4(${[0, 1, 2, 3].map((c) =>
    `${fragmentAccumulators[t * 4 + c]} + bias`).join(", ")});`).join("\n")}
}`;
  const fragmentProgram = gl.createProgram();
  gl.attachShader(fragmentProgram, compile(gl, gl.VERTEX_SHADER, fragmentVertex));
  gl.attachShader(fragmentProgram, compile(gl, gl.FRAGMENT_SHADER, fragmentShader));
  gl.linkProgram(fragmentProgram);
  if (!gl.getProgramParameter(fragmentProgram, gl.LINK_STATUS)) {
    throw new Error(`link fragment: ${gl.getProgramInfoLog(fragmentProgram)}`);
  }
  const side = Math.ceil(Math.sqrt(fragmentInvocations));
  const ring = Array.from({ length: iterations }, () => {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const attachments = [];
    for (let t = 0; t < targets; t += 1) {
      const out = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, out);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, side, side, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + t, gl.TEXTURE_2D, out, 0);
      attachments.push(gl.COLOR_ATTACHMENT0 + t);
    }
    gl.drawBuffers(attachments);
    return framebuffer;
  });
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const runFragment = (i) => {
    gl.useProgram(fragmentProgram);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    const position = gl.getAttribLocation(fragmentProgram, "position");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, xTex);
    gl.uniform1i(gl.getUniformLocation(fragmentProgram, "xTex"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, wTex);
    gl.uniform1i(gl.getUniformLocation(fragmentProgram, "wTex"), 1);
    gl.viewport(0, 0, side, side);
    gl.bindFramebuffer(gl.FRAMEBUFFER, ring[i % ring.length]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
  const syncFragment = () => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, ring[0]);
    const pixel = new Float32Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, pixel);
    return pixel[0];
  };

  const time = (run, sync) => {
    run(0);
    const witness = sync();
    const samples = [];
    for (let round = 0; round < rounds; round += 1) {
      const start = performance.now();
      for (let i = 0; i < iterations; i += 1) run(i);
      sync();
      samples.push((performance.now() - start) / iterations);
    }
    return { ms: Math.min(...samples),
      spread: Math.max(...samples) / Math.min(...samples),
      witness };
  };

  const feedbackResult = time(runFeedback, syncFeedback);
  const fragmentResult = time(runFragment, syncFragment);
  const report = (name, r, armFetches, tile) => ({
    arm: name,
    tile,
    milliseconds: Number(r.ms.toFixed(3)),
    spread: Number(r.spread.toFixed(2)),
    gigaflops: Number((flops / r.ms / 1e6).toFixed(0)),
    gigaFetchesPerSecond: Number((armFetches / r.ms / 1e6).toFixed(1)),
    flopsPerFetch: Number((flops / armFetches).toFixed(2)),
    witnessFinite: Number.isFinite(r.witness),
  });
  const arms = [
    report("transform-feedback", feedbackResult, fetches, `${rows}x${outs}`),
    report("fragment", fragmentResult, fragmentFetches, `${fragmentRows}x${fragmentOuts}`),
  ];
  const ratio = fragmentResult.ms / feedbackResult.ms;
  const parity = 1303;
  return {
    renderer: gl.getParameter(gl.RENDERER),
    tile: `${rows} rows x ${outs} outputs = ${outputs} floats an invocation`,
    k,
    window,
    windowKib: window > 0 ? Number((window * 16 / 1024).toFixed(1)) : null,
    invocations,
    feedbackBytesPerInvocation: perInvocationBytes,
    feedbackPasses: passes,
    fragmentInvocations,
    arms,
    feedbackOverFragment: Number(ratio.toFixed(2)),
    parityNeedsGflops: parity,
    verdict: ratio >= 1.1
      ? `TRANSFORM FEEDBACK IS ${ratio.toFixed(2)}x THE FRAGMENT PATH at the same tile and the `
        + `same arithmetic, so the output budget really is the binding constraint and a vertex `
        + `invocation can carry a bigger one`
      : `transform feedback is ${ratio.toFixed(2)}x the fragment path, so the bigger output `
        + `budget does NOT buy a bigger ceiling here - the vertex stage issues too slowly to `
        + `use it`,
  };
}
