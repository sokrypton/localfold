/**
 * A WebGL2 matmul through TRANSFORM FEEDBACK, which has four times the
 * fragment path's output budget.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl2-tf-matmul.js
 *
 * WHY. bench-webgl2-matmul.js establishes that a fragment-shader GEMM here is
 * fetch-RATE bound and that its speed tracks arithmetic intensity to within 1%
 * over a 3.3x range: 1/4/8 rows a fragment give intensity 6.4/16.0/21.3 and
 * 1.01/2.50/3.32 TFLOP/s. Its 156 G fetches a second is exactly the
 * texture-unit limit for 128-bit RGBA32F reads, 609/4 = 152 G/s.
 *
 * Intensity there is capped by `MAX_DRAW_BUFFERS` = 8, an RGBA texel each, so
 * **32 output floats an invocation** - which makes 21.3 flops a fetch the
 * ceiling of the fragment path, reached by both 8x1 and 4x2 and beaten by
 * nothing.
 *
 * 🔴 BUT THE VERTEX PATH HAS A DIFFERENT BUDGET.
 * `MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS` is **128** on this device -
 * four times as many outputs an invocation - which doubles the reachable
 * intensity to 42.7. If the fetch-rate model holds, that is 2x.
 *
 * Rasterisation is discarded; this is the vertex shader used as a compute
 * kernel, which is what people did before compute shaders existed.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const VERTEX = (K, ROWS, GROUPS, N, M, ORDER) => {
  const vecs = ROWS * GROUPS;   // vec4 outputs an invocation
  return `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D matA;        // (K/4) x M
uniform sampler2D matB;        // (N/4) x K
const int N_BLOCKS = ${N / 4};
${Array.from({ length: vecs }, (_, i) => `out vec4 v${i};`).join("\n")}
void main() {
  int id = gl_VertexID;
  // 🔴 WHICH INDEX VARIES FASTEST DECIDES WHICH OPERAND BROADCASTS.
  // Neighbouring invocations share whichever index is the SLOW one, so that
  // operand's fetches are uniform across them and are served once. A is ROWS
  // fetches an iteration and B is 4*GROUPS, so with 8x1 it is A that is worth
  // making uniform - which means nGroup fastest and mBlock slowest.
${ORDER === "nFast"
  ? `  int nGroup = id % ${N / 4 / GROUPS};
  int mBlock = id / ${N / 4 / GROUPS};`
  : `  int mBlock = id % ${M / ROWS};
  int nGroup = id / ${M / ROWS};`}
${Array.from({ length: vecs }, (_, i) => `  vec4 acc${i} = vec4(0.0);`).join("\n")}
  for (int k = 0; k < ${K / 4}; k++) {
${Array.from({ length: GROUPS }, (_, c) => `    vec4 b${c}_0 = texelFetch(matB, ivec2(nGroup * ${GROUPS} + ${c}, k * 4 + 0), 0);
    vec4 b${c}_1 = texelFetch(matB, ivec2(nGroup * ${GROUPS} + ${c}, k * 4 + 1), 0);
    vec4 b${c}_2 = texelFetch(matB, ivec2(nGroup * ${GROUPS} + ${c}, k * 4 + 2), 0);
    vec4 b${c}_3 = texelFetch(matB, ivec2(nGroup * ${GROUPS} + ${c}, k * 4 + 3), 0);`).join("\n")}
${Array.from({ length: ROWS }, (_, r) => `    vec4 a${r} = texelFetch(matA, ivec2(k, mBlock * ${ROWS} + ${r}), 0);
${Array.from({ length: GROUPS }, (_, c) =>
  `    acc${r * GROUPS + c} += a${r}.x * b${c}_0 + a${r}.y * b${c}_1 + a${r}.z * b${c}_2 + a${r}.w * b${c}_3;`).join("\n")}`).join("\n")}
  }
${Array.from({ length: vecs }, (_, i) => `  v${i} = acc${i};`).join("\n")}
  // 🔴 REQUIRED EVEN WITH RASTERIZER_DISCARD. Without it ANGLE links a vertex
  // shader with no position and the feedback comes back zeroed.
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
  gl_PointSize = 1.0;
}`;
};

const FRAGMENT = `#version 300 es
precision highp float;
out vec4 unused;
void main() { unused = vec4(0.0); }`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`${gl.getShaderInfoLog(shader)}`);
  }
  return shader;
}

export async function main(device, args) {
  const rounds = Number(option(args, "rounds", "5"));
  const iterations = Number(option(args, "iterations", "8"));
  const gl = new OffscreenCanvas(4, 4).getContext("webgl2");
  if (gl === null) return { error: "no webgl2" };
  if (gl.getExtension("EXT_color_buffer_float") === null) return { error: "no float colour buffer" };

  const makeTexture = (width, height, data, internal) => {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) {
      gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    }
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) {
      gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    }
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, gl.RGBA, gl.FLOAT, data);
    return tex;
  };

  const median = (v) => [...v].sort((a, b) => a - b)[v.length >> 1];
  const rows = [];
  // 🔴 THE SHAPES THIS REPOSITORY ACTUALLY RUNS, not just a square. A square
  // matmul is the number everyone quotes and none of the trunk's kernels have.
  const shapes = [
    { name: "square 2048", m: 2048, k: 2048, n: 2048 },
    { name: "AF2 transition", m: 30208, k: 256, n: 1024 },
    { name: "difftx qkvg", m: 240, k: 768, n: 768 },
    { name: "AF3 pair-transition", m: 40000, k: 128, n: 512 },
  ];
  for (const { name: shapeName, m, k, n } of shapes) {
  const aData = new Float32Array(m * k).fill(1);
  const bData = new Float32Array(k * n).fill(1);
  for (const [storage, internal] of [["f16", gl.RGBA16F]]) {
    const texA = makeTexture(k / 4, m, aData, internal);
    const texB = makeTexture(n / 4, k, bData, internal);
    // 🔴 THE REACHABLE INTENSITY IS SET BY THREE THINGS AT ONCE: 31 vec4 of
      // varyings (MAX_VERTEX_OUTPUT_COMPONENTS 124, not the 128 the transform
      // feedback limit advertises), ROWS dividing M, and GROUPS dividing N/4.
      // At 2048 the best that satisfies all three is 8x2, intensity 32.0
      // against the fragment path's 21.3 - 1.5x, not the 2x the raw component
      // count suggests.
      for (const [ROWS, GROUPS, ORDER] of
           [[4, 1, "nFast"], [8, 1, "nFast"], [16, 1, "nFast"], [8, 2, "nFast"]]) {
        if (m % ROWS !== 0 || (n / 4) % GROUPS !== 0) continue;
        if (Math.max(k / 4, m, n / 4, k) > gl.getParameter(gl.MAX_TEXTURE_SIZE)) continue;
      const vecs = ROWS * GROUPS;
      const components = vecs * 4;
      const cap = gl.getParameter(gl.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS);
      if (components > cap) {
        rows.push({ shape: shapeName, arm: `${ROWS}x${GROUPS} ${ORDER}`, storage, skipped: `${components} > cap ${cap}` });
        continue;
      }
      const program = gl.createProgram();
      gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX(k, ROWS, GROUPS, n, m, ORDER)));
      gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
      // 🔴 BEFORE linkProgram, not after. Naming the varyings later links a
      // program with no feedback and the draw writes nothing.
      gl.transformFeedbackVaryings(program,
        Array.from({ length: vecs }, (_, i) => `v${i}`), gl.INTERLEAVED_ATTRIBS);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        rows.push({ shape: shapeName, arm: `${ROWS}x${GROUPS} ${ORDER}`, storage, skipped: gl.getProgramInfoLog(program) });
        continue;
      }
      gl.useProgram(program);
      gl.uniform1i(gl.getUniformLocation(program, "matA"), 0);
      gl.uniform1i(gl.getUniformLocation(program, "matB"), 1);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texB);

      const vertices = (m / ROWS) * (n / 4 / GROUPS);
      const bytes = vertices * components * 4;
      const feedback = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, feedback);
      gl.bufferData(gl.ARRAY_BUFFER, bytes, gl.STATIC_COPY);
      const tf = gl.createTransformFeedback();
      gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf);
      gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, feedback);
      gl.enable(gl.RASTERIZER_DISCARD);

      const draw = () => {
        gl.beginTransformFeedback(gl.POINTS);
        gl.drawArrays(gl.POINTS, 0, vertices);
        gl.endTransformFeedback();
      };
      // 🔴 CHECKED, because a feedback that writes nothing is very fast.
      while (gl.getError() !== gl.NO_ERROR) { /* drain stale errors */ }
      draw();
      const probe = new Float32Array(4);
      // 🔴 A BUFFER MAY NOT BE BOUND TO TRANSFORM_FEEDBACK_BUFFER AND
      // ARRAY_BUFFER AT THE SAME TIME - WebGL2 rejects it as two incompatible
      // targets, and the read then returns zeros with an INVALID_OPERATION
      // nobody looked at. Release the feedback binding first.
      const readBack = () => {
        // 🔴 UNBIND THE TRANSFORM FEEDBACK OBJECT, not just its buffer base.
        // The object holds the binding, so releasing base 0 while the object
        // is still current leaves the buffer attached to two incompatible
        // targets and getBufferSubData returns zeros behind an
        // INVALID_OPERATION nobody reads.
        // BOTH bindings, in this order - probe-webgl2-tf.js is the smallest
        // case that works and it releases the base and then the object.
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
        gl.bindBuffer(gl.ARRAY_BUFFER, feedback);
        gl.getBufferSubData(gl.ARRAY_BUFFER, 0, probe);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf);
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, feedback);
      };
      readBack();
      const error = gl.getError();

      const once = () => {
        const start = performance.now();
        for (let i = 0; i < iterations; i += 1) draw();
        readBack();
        return (performance.now() - start) / iterations;
      };
      once();
      const times = [];
      for (let i = 0; i < rounds; i += 1) times.push(once());
      const ms = median(times);
      // 🔴 CHECKED FROM THE PROBE THE TIMED LOOP LAST FILLED, not from one
      // taken before it. An earlier version verified against a probe read
      // before the loop had run and reported every correct arm as wrong.
      readBack();
      const correct = Math.abs(probe[0] - k)
        < Math.max(1, k * (storage === "f16" ? 2e-2 : 1e-4));
      const fetches = ROWS + 4 * GROUPS;
      const flops = 32 * ROWS * GROUPS;
      rows.push({ shape: shapeName, arm: `${ROWS}x${GROUPS} ${ORDER}`, storage, components,
        intensity: Number((flops / fetches).toFixed(1)),
        ms: Number(ms.toFixed(4)),
        tflops: Number((2 * m * k * n / ms / 1e6 / 1000).toFixed(2)),
        percentOfCeiling: Number((100 * 2 * m * k * n / ms / 1e6 / 1000 / 18.1).toFixed(1)),
        correct, sample: probe[0], glError: error === gl.NO_ERROR ? null : error });
      gl.disable(gl.RASTERIZER_DISCARD);
      gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
      gl.deleteTransformFeedback(tf); gl.deleteBuffer(feedback); gl.deleteProgram(program);
    }
    gl.deleteTexture(texA); gl.deleteTexture(texB);
  }
  }
  return { rows };
}
