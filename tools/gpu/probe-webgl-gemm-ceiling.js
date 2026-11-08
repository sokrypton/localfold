/**
 * The MOST a fragment-shader GEMM can do on this device, measured.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-gemm-ceiling.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-gemm-ceiling.js --k=512
 *
 * 🔴 THIS EXISTS BECAUSE webgl/README.md's FLOOR HAS TWICE BEEN AN ARGUMENT AND
 * NOT A MEASUREMENT. The arithmetic is sound - an invocation writes at most
 * `MAX_DRAW_BUFFERS * 4` = 32 output floats, a register tile of R by C costs
 * `R + C` fetches a k-step for 256 flops, and `R * C = 32` makes the smallest
 * `R + C` twelve, so 21.33 flops a `fetch4` - but every figure multiplied by it
 * was either inferred from the shipped kernels (6x too low, retracted) or taken
 * under locality no GEMM can have (over-read, also retracted). So this measures
 * the thing itself: a GEMM in the BEST CASE this backend admits.
 *
 * Best case means every advantage at once and nothing realistic about it:
 *
 *   - the optimal tile, 8 rows by 4 outputs into 8 render targets, 32 floats;
 *   - operands small enough to be L1-resident - 8 KiB of x, 4 KiB of weight -
 *     and EVERY FRAGMENT READS THE SAME ONES, which a real GEMM cannot do
 *     because different fragments must produce different outputs;
 *   - row strides that are powers of two, so nothing wraps;
 *   - a long k loop so the loop overhead vanishes;
 *   - hundreds of thousands of fragments, so the device is full.
 *
 * What comes out is an upper bound: no fragment GEMM here can beat it, and a
 * real one with disjoint rows will not reach it. Compare it against what parity
 * would need - a bioemu2 fold is 5,081 GFLOP of countable arithmetic and the
 * WGSL fold takes 3.9 s, so **1,303 GFLOP/s**.
 *
 * 🔴 AND IT USES A RING OF TARGETS. An Apple GPU discards the earlier of two
 * identical full-viewport draws into one target, which has now cost three
 * measurements in this tree.
 */
import { FULLSCREEN_TRIANGLE, VERTEX_SHADER, compile, createWebGL2Context }
  from "../../webgl/runtime/context.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const source = (k, rows, outs) => {
  const component = ["x", "y", "z", "w"];
  return `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const int K4 = ${k / 4};
${Array.from({ length: rows }, (_, t) =>
    `layout(location = ${t}) out vec4 result${t};`).join("\n")}
uniform sampler2D xTex;
uniform sampler2D wTex;

void main() {
${Array.from({ length: rows }, (_, t) => `  vec4 acc${t} = vec4(0.0);`).join("\n")}
  // 🔴 EVERY FRAGMENT READS THE SAME OPERANDS. That is the point: it is the
  // best locality a kernel could have and better than any GEMM can have, so
  // what this measures is an upper bound rather than an achievable rate.
  for (int k4 = 0; k4 < K4; ++k4) {
${Array.from({ length: rows }, (_, t) =>
    `    vec4 x${t} = fetch4(xTex, ${t} * K4 + k4);`).join("\n")}
${Array.from({ length: outs }, (_, o) =>
    `    vec4 w${o} = fetch4(wTex, ${o} * K4 + k4);`).join("\n")}
${Array.from({ length: rows }, (_, t) => Array.from({ length: outs }, (_, o) =>
    `    acc${t}.${component[o]} += dot(x${t}, w${o});`).join(" ")).join("\n")}
  }
${Array.from({ length: rows }, (_, t) => `  result${t} = acc${t};`).join("\n")}
}

vec4 fetch4(sampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % 2048, texel / 2048), 0);
}`.replace(/^(vec4 fetch4[\s\S]*)$/m, "");
};

/** The shader with `fetch4` declared before use, which GLSL requires. */
const shader = (k, rows, outs) => {
  const body = source(k, rows, outs);
  const head = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

vec4 fetch4(sampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % 2048, texel / 2048), 0);
}
`;
  return head + body.replace(/^#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n/, "");
};

export async function main(device, args) {
  const k = Number(option(args, "k", "256"));
  const rounds = Number(option(args, "rounds", "4"));
  const iterations = Number(option(args, "iterations", "16"));
  const rows = 8;
  const outs = 4;
  const gl = createWebGL2Context();
  const targets = gl.getParameter(gl.MAX_DRAW_BUFFERS);
  if (targets < rows) throw new Error(`${targets} draw buffers, need ${rows}`);

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
    const texels = values.length / 4;
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texels, 1, 0, gl.RGBA, gl.FLOAT, values);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return t;
  };
  const xTex = texture(fill(rows * k, 7));
  const wTex = texture(fill(outs * k, 11));

  // 🔴 `--side` IS HERE TO ASK WHETHER THE FRAGMENT COUNT IS THE CEILING. Both
  // of this backend's largest families dispatch about 32,000-37,000 fragments -
  // the dense projection 32,256 and the triangle contraction 36,992 - because
  // the count is the output's float count over `MAX_DRAW_BUFFERS * 4`, and this
  // probe reaches its 2000 GFLOP/s at 512 squared, which is **262,144**. If the
  // rate falls off steeply below that, then the output budget caps the fragment
  // count, the fragment count caps occupancy, and occupancy caps both families
  // at once - which is one structural statement instead of two unexplained
  // kernels. Nothing had swept it: the side was a literal.
  const side = Number(option(args, "side", "512"));
  const fragments = side * side;
  const ring = Array.from({ length: iterations }, () => {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const attachments = [];
    for (let t = 0; t < rows; t += 1) {
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

  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, shader(k, rows, outs)));
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
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, xTex);
  gl.uniform1i(gl.getUniformLocation(program, "xTex"), 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, wTex);
  gl.uniform1i(gl.getUniformLocation(program, "wTex"), 1);
  gl.viewport(0, 0, side, side);

  const draw = (i) => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, ring[i % ring.length]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
  const sync = () => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, ring[0]);
    const pixel = new Float32Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, pixel);
    return pixel[0];
  };
  draw(0);
  const witness = sync();
  const samples = [];
  for (let round = 0; round < rounds; round += 1) {
    const start = performance.now();
    for (let i = 0; i < iterations; i += 1) draw(i);
    sync();
    samples.push((performance.now() - start) / iterations);
  }
  const ms = Math.min(...samples);
  const flops = fragments * rows * outs * k * 2;
  const fetches = fragments * (rows + outs) * (k / 4);
  const renderer = gl.getParameter(gl.RENDERER);
  const gflops = flops / ms / 1e6;
  // A bioemu2 fold is 5,081 GFLOP of countable arithmetic; the WGSL fold is 3.9 s.
  const parity = 5081 / 3.9;
  return {
    renderer,
    maxDrawBuffers: targets,
    tile: `${rows} rows x ${outs} outputs = ${rows * outs} output floats`,
    k,
    fragments,
    witnessFinite: Number.isFinite(witness),
    milliseconds: Number(ms.toFixed(3)),
    spread: Number((Math.max(...samples) / Math.min(...samples)).toFixed(2)),
    gigaflops: Number(gflops.toFixed(0)),
    gigaFetchesPerSecond: Number((fetches / ms / 1e6).toFixed(2)),
    flopsPerFetch: Number((flops / fetches).toFixed(2)),
    parityNeedsGflops: Number(parity.toFixed(0)),
    headroomOverParity: Number((gflops / parity).toFixed(2)),
    verdict: gflops < parity
      ? `the BEST CASE a fragment GEMM reaches here is ${gflops.toFixed(0)} GFLOP/s, and parity with `
        + `the WGSL bioemu2 fold needs ${parity.toFixed(0)} - so parity is out of reach by `
        + `${(parity / gflops).toFixed(2)}x even with locality no real GEMM can have`
      : `the best case reaches ${gflops.toFixed(0)} GFLOP/s against parity's ${parity.toFixed(0)}, `
        + `so the arithmetic does NOT rule parity out and the gap is in the kernels`,
  };
}
