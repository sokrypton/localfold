/**
 * What WebGL2 can ACTUALLY do for a matmul on this device.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl2-matmul.js
 *
 * 🔴 WHY THIS EXISTS AND WHY THE FIRST ANSWER WAS WRONG. An earlier version of
 * docs/A100.md concluded "WebGL2 is 165x slower" from jax-js's WebGL backend,
 * which its own authors describe as offered "on a best-effort basis". That
 * measures one implementation, not the API - and a GPU is not 165x slower at
 * arithmetic because the arithmetic arrived through a fragment shader. This is
 * a properly packed GEMM instead: RGBA32F textures so one fetch carries four
 * floats, four outputs per fragment, and the k loop unrolled four deep.
 *
 * The structure is the one every WebGL matmul uses, because it is the only one
 * available: there is no shared memory in a fragment shader, so each
 * invocation re-reads its operands from texture and leans on the texture cache
 * where a compute kernel would stage a tile once.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const VERTEX = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`;

// One fragment owns one row of A and four columns of B: out is RGBA32F, so a
// fragment writes four results. The k loop steps four at a time, and each step
// is one A fetch (four k values) against four B fetches (four n values each).
const FRAGMENT = (K) => `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D matA;   // (K/4) x M, RGBA = four consecutive k
uniform sampler2D matB;   // (N/4) x K, RGBA = four consecutive n
out vec4 result;
void main() {
  int m = int(gl_FragCoord.y);
  int nBlock = int(gl_FragCoord.x);
  vec4 acc = vec4(0.0);
  for (int k = 0; k < ${K / 4}; k++) {
    vec4 a = texelFetch(matA, ivec2(k, m), 0);
    vec4 b0 = texelFetch(matB, ivec2(nBlock, k * 4 + 0), 0);
    vec4 b1 = texelFetch(matB, ivec2(nBlock, k * 4 + 1), 0);
    vec4 b2 = texelFetch(matB, ivec2(nBlock, k * 4 + 2), 0);
    vec4 b3 = texelFetch(matB, ivec2(nBlock, k * 4 + 3), 0);
    acc += a.x * b0 + a.y * b1 + a.z * b2 + a.w * b3;
  }
  result = acc;
}`;

/**
 * 🔴 FOUR ROWS A FRAGMENT, THROUGH FOUR DRAW BUFFERS, which is the only
 * register blocking a fragment shader can have. One output per fragment reads
 * one A texel against four B texels for 32 flops - five fetches. Four rows
 * reuse the SAME four B texels, so it is eight fetches for 128 flops, and the
 * arithmetic intensity goes from 6.4 flops a fetch to 16. This is what a
 * compute kernel gets from staging a tile in workgroup memory, done the only
 * way the older API allows.
 */
/**
 * @param {"a"|"b"|null} drop replace an operand with a value the compiler
 *   CANNOT fold - varying with k AND with the row - so the arm prices the
 *   FETCH and not the arithmetic. `vec4(1.0)` does not work: with a = 1 the
 *   eight accumulators become the same expression and collapse to one, and the
 *   arm then reports the speed of a kernel a quarter the size.
 * @param {boolean} transposedA lay A out with m along X instead of Y. A warp
 *   of fragments spans a 2D tile, so both gl_FragCoord.x and .y vary across
 *   it - but B is indexed by x and A by y, and a texture is far more coalesced
 *   along x. This is the arm that asks whether that is the whole difference.
 */
const FRAGMENT_MRT = (K, ROWS, drop, transposedA) => `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D matA;
uniform sampler2D matB;
${Array.from({ length: ROWS }, (_, r) =>
  `layout(location = ${r}) out vec4 out${r};`).join("\n")}
void main() {
  int mBlock = int(gl_FragCoord.y);
  int nBlock = int(gl_FragCoord.x);
${Array.from({ length: ROWS }, (_, r) => `  vec4 acc${r} = vec4(0.0);`).join("\n")}
  for (int k = 0; k < ${K / 4}; k++) {
${drop === "b" ? `    vec4 b0 = vec4(float(k * 4 + 0) * 1e-9); vec4 b1 = vec4(float(k * 4 + 1) * 1e-9);
    vec4 b2 = vec4(float(k * 4 + 2) * 1e-9); vec4 b3 = vec4(float(k * 4 + 3) * 1e-9);`
  : `    vec4 b0 = texelFetch(matB, ivec2(nBlock, k * 4 + 0), 0);
    vec4 b1 = texelFetch(matB, ivec2(nBlock, k * 4 + 1), 0);
    vec4 b2 = texelFetch(matB, ivec2(nBlock, k * 4 + 2), 0);
    vec4 b3 = texelFetch(matB, ivec2(nBlock, k * 4 + 3), 0);`}
${Array.from({ length: ROWS }, (_, r) => {
  const fetch = drop === "a" ? `vec4(float(k * ${ROWS} + ${r}) * 1e-9)`
    : transposedA ? `texelFetch(matA, ivec2(mBlock * ${ROWS} + ${r}, k), 0)`
                  : `texelFetch(matA, ivec2(k, mBlock * ${ROWS} + ${r}), 0)`;
  return `    vec4 a${r} = ${fetch};
    acc${r} += a${r}.x * b0 + a${r}.y * b1 + a${r}.z * b2 + a${r}.w * b3;`;
}).join("\n")}
  }
${Array.from({ length: ROWS }, (_, r) => `  out${r} = acc${r};`).join("\n")}
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? "shader failed");
  }
  return shader;
}

export async function main(device, args) {
  const rounds = Number(option(args, "rounds", "5"));
  const iterations = Number(option(args, "iterations", "16"));
  const canvas = new OffscreenCanvas(4, 4);
  const gl = canvas.getContext("webgl2", { antialias: false });
  if (gl === null) return { error: "no webgl2" };
  // 🔴 WITHOUT THIS A FLOAT FRAMEBUFFER IS NOT RENDERABLE and the draw is a
  // silent no-op, which reads as an extremely fast matmul.
  if (gl.getExtension("EXT_color_buffer_float") === null) {
    return { error: "no EXT_color_buffer_float; a float target is not renderable" };
  }

  const shapes = [
    { name: "difftx qkvg", m: 240, k: 768, n: 768 },
    { name: "AF2 transition", m: 30208, k: 256, n: 1024 },
    { name: "square 2048", m: 2048, k: 2048, n: 2048 },
  ];

  const median = (v) => [...v].sort((a, b) => a - b)[v.length >> 1];
  const rows = [];
  for (const { name, m, k, n } of shapes) {
    // A is (k/4) x m, B is (n/4) x k, both RGBA32F.
    const aData = new Float32Array(m * k).fill(1);
    const bData = new Float32Array(k * n).fill(1);
    const makeTexture = (width, height, data, internal = gl.RGBA32F) => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      // 🔴 RGBA16F STORAGE, f32 ARITHMETIC. The fetch widens to float in the
      // shader either way; what halves is the bytes the texture cache moves,
      // and the 32-bit arm is pinned at 2.50 TB/s of logical texel traffic
      // against this card's 1.5 TB/s of HBM. Exactly the trade
      // src/runtime/storage.js makes with pack2x16float on the WebGPU side.
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, gl.RGBA, gl.FLOAT, data);
      return tex;
    };
    if (Math.max(k / 4, n / 4, m, k) > gl.getParameter(gl.MAX_TEXTURE_SIZE)) {
      rows.push({ shape: name, skipped: "exceeds MAX_TEXTURE_SIZE" });
      continue;
    }
    for (const [storage, internal] of [["f32", gl.RGBA32F], ["f16", gl.RGBA16F]]) {
    const texA = makeTexture(k / 4, m, aData, internal);
    const texAT = makeTexture(m, k / 4, aData, internal);
    const texB = makeTexture(n / 4, k, bData, internal);
    const made = [texA, texAT, texB];

    for (const [rowsPerFragment, drop, transposedA] of
         [[8, null, false], [8, null, true], [8, "a", false], [8, "b", false]]) {
      // 🔴 THE SURGERY ARMS COMPUTE THE WRONG ANSWER ON PURPOSE. Replacing an
      // operand with a constant prices its FETCHES; a large error is the
      // expected report and the millisecond is the number that matters. Same
      // device as bench-evoformer-linear.js's `:noload`.
      const arm = `${rowsPerFragment} rows, ${storage}`
        + (transposedA ? ", A along X" : "")
        + (drop === null ? "" : `, no ${drop} fetch`);
      const mrt = rowsPerFragment > 1;
      if (m % rowsPerFragment !== 0
        || gl.getParameter(gl.MAX_DRAW_BUFFERS) < rowsPerFragment) continue;
      const height = m / rowsPerFragment;
      const targets = [];
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      for (let t = 0; t < rowsPerFragment; t += 1) {
        const tex = makeTexture(n / 4, height, null);
        targets.push(tex);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + t, gl.TEXTURE_2D, tex, 0);
      }
      gl.drawBuffers(targets.map((_, t) => gl.COLOR_ATTACHMENT0 + t));
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        rows.push({ shape: name, arm, skipped: "framebuffer incomplete" });
        continue;
      }
      const program = gl.createProgram();
      gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
      gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER,
        mrt ? FRAGMENT_MRT(k, rowsPerFragment, drop, transposedA) : FRAGMENT(k)));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program) ?? "link failed");
      }
      gl.useProgram(program);
      const quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      const position = gl.getAttribLocation(program, "position");
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      gl.uniform1i(gl.getUniformLocation(program, "matA"), 0);
      gl.uniform1i(gl.getUniformLocation(program, "matB"), 1);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, transposedA ? texAT : texA);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texB);
      gl.viewport(0, 0, n / 4, height);

      const draw = () => { gl.drawArrays(gl.TRIANGLES, 0, 3); };
      const sync = new Float32Array(4);
      draw();
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, sync);
      const correct = drop !== null ? null
        : Math.abs(sync[0] - k) < Math.max(1, k * (storage === "f16" ? 2e-2 : 1e-4));
      const once = () => {
        const start = performance.now();
        for (let i = 0; i < iterations; i += 1) draw();
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, sync);
        return (performance.now() - start) / iterations;
      };
      once();
      const times = [];
      for (let i = 0; i < rounds; i += 1) times.push(once());
      const ms = median(times);
      const gflops = 2 * m * k * n / ms / 1e6;
      rows.push({ shape: name, arm, ms: Number(ms.toFixed(4)),
        tflops: Number((gflops / 1000).toFixed(2)),
        percentOfCeiling: Number((100 * gflops / 1000 / 18.1).toFixed(1)), correct });
      for (const t of targets) gl.deleteTexture(t);
      gl.deleteFramebuffer(fbo); gl.deleteProgram(program); gl.deleteBuffer(quad);
    }
    for (const t of made) gl.deleteTexture(t);
    }
  }
  return { renderer: gl.getParameter(gl.RENDERER), iterations, rows };
}
