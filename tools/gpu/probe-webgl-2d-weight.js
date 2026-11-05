/**
 * Whether a tensor's TEXTURE SHAPE is what the projections have been waiting
 * for: one matrix row per texture row, against this tree's linear index.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-2d-weight.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-2d-weight.js --rows=576 --in=896 --out=1792
 *
 * 🔴 TWO TRANSFORM-FEEDBACK KERNELS AT ONE SHAPE AND ONE OUTPUT BUDGET CAME OUT
 * 6.3x APART, AND THAT IS WHAT THIS ASKS ABOUT. `bench-webgl2-tf-matmul.js`
 * reaches **1.6 TFLOP/s** at 2048^3 with eight rows and four columns an
 * invocation - 32 floats, the same budget a fragment has - and its timing is
 * real (12.8 ms at one iteration, 10.7 at sixteen, so nothing is being
 * collapsed). The same geometry written against this tree's tensors reaches
 * **254 GFLOP/s**. Same arithmetic, same intensity, same device.
 *
 * The one thing that differs is how a texel is addressed. That bench gives
 * `matA` the shape `(K/4) x M` and fetches `ivec2(k, m)`, so **each matrix row
 * is a texture row**. This tree addresses linearly - `Storage`'s first
 * paragraph, "a tensor is a linear index, not a shape" - so element `(row, k)`
 * lands at `(t % 2048, t / 2048)` for `t = row * IN4 + k`, and a matrix row
 * WRAPS every 2048 texels. A texture is stored swizzled for 2D locality, so a
 * row that wraps is a row whose neighbours are not neighbours.
 *
 * That linear layout buys real things and the file says what they are: a pair
 * tensor is as addressable as a vector, and the capacity bound becomes
 * `W * MAX_TEXTURE_SIZE * 4` rather than `sqrt(MAX_TEXTURE_SIZE)` on one axis.
 * What was never measured is what it COSTS, and this is that measurement - on
 * the WEIGHT alone, which is the cheap half: a weight is uploaded once by this
 * tree's own code and read by one kernel, so reshaping it cascades nowhere.
 *
 * Both arms run the same 8 x 4 blocked geometry into eight render targets and
 * are checked against each other elementwise, so a layout that is faster and
 * wrong cannot pass.
 */
import { TEXTURE_WIDTH } from "../../webgl/runtime/context.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const ROWS_PER_FRAGMENT = 8;

const VERTEX = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`;

/**
 * `shaped`: the weight is `(IN/4) x OUT`, fetched `ivec2(k, o)`.
 * Otherwise it is linear at width W, fetched `(t % W, t / W)`.
 *
 * x stays linear in both arms, so the only difference is the weight's
 * addressing - which is the question.
 */
const FRAGMENT = (rows, inChannels, outChannels, shaped, splits = 1) => `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const int W = ${TEXTURE_WIDTH};
const int ROWS = ${rows};
const int IN4 = ${inChannels / 4};
const int OUT4 = ${outChannels / 4};
const int RQ = ${Math.ceil(rows / ROWS_PER_FRAGMENT)};
const int SPLITS = ${splits};
const int KLEN = ${inChannels / 4 / splits};
uniform sampler2D xTex;
uniform sampler2D weightTex;
${Array.from({ length: ROWS_PER_FRAGMENT },
    (_, i) => `layout(location = ${i}) out vec4 result${i};`).join("\n")}

vec4 fetchLinear(sampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % W, texel / W), 0);
}
${shaped
    ? "vec4 weightAt(int o, int k) { return texelFetch(weightTex, ivec2(k, o), 0); }"
    : "vec4 weightAt(int o, int k) { return fetchLinear(weightTex, o * IN4 + k); }"}

void main() {
  int texel = int(gl_FragCoord.y) * W + int(gl_FragCoord.x);
  // 🔴 THE SPLIT INDEX IS THE SLOWEST AXIS, so each 1/S of the k range is a
  // contiguous plane set and the reduction reads them in order.
  int split = texel / (RQ * OUT4);
  int within = texel - split * (RQ * OUT4);
  int group = within / RQ;
  int q = within - group * RQ;
  int o0 = group * 4;
  int base = texel < SPLITS * RQ * OUT4 ? q * ${ROWS_PER_FRAGMENT} : 0;
  int kStart = split * KLEN;
${Array.from({ length: ROWS_PER_FRAGMENT },
    (_, i) => `  int xBase${i} = min(base + ${i}, ROWS - 1) * IN4;`).join("\n")}
${Array.from({ length: ROWS_PER_FRAGMENT },
    (_, i) => `  vec4 acc${i} = vec4(0.0);`).join("\n")}
  for (int k = kStart; k < kStart + KLEN; ++k) {
    vec4 w0 = weightAt(o0 + 0, k);
    vec4 w1 = weightAt(o0 + 1, k);
    vec4 w2 = weightAt(o0 + 2, k);
    vec4 w3 = weightAt(o0 + 3, k);
${Array.from({ length: ROWS_PER_FRAGMENT }, (_, i) => `    {
      vec4 x = fetchLinear(xTex, xBase${i} + k);
      acc${i} += vec4(dot(x, w0), dot(x, w1), dot(x, w2), dot(x, w3));
    }`).join("\n")}
  }
${Array.from({ length: ROWS_PER_FRAGMENT },
    (_, i) => `  result${i} = acc${i};`).join("\n")}
}`;

const fill = (count, seed) => {
  let state = seed >>> 0;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    out[i] = ((state / 0x1_0000_0000) * 2 - 1) * 0.5;
  }
  return out;
};

export async function main(device, args) {
  const rows = Number(option(args, "rows", "576"));
  const inChannels = Number(option(args, "in", "896"));
  const outChannels = Number(option(args, "out", "1792"));
  const iterations = Number(option(args, "iterations", "8"));
  const gl = createWebGL2Context();
  const maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  if (inChannels / 4 > maxTexture || outChannels > maxTexture) {
    throw new RangeError(
      `a ${inChannels / 4} x ${outChannels} weight is past MAX_TEXTURE_SIZE ${maxTexture} - `
      + "which is the capacity the linear layout exists to avoid");
  }

  const makeTexture = (width, height, values) => {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const padded = new Float32Array(width * height * 4);
    padded.set(values.subarray(0, Math.min(values.length, padded.length)));
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, padded);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) {
      gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    }
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) {
      gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    }
    return texture;
  };
  const linearShape = (elements) => {
    const texels = Math.ceil(elements / 4);
    return { width: TEXTURE_WIDTH, height: Math.max(1, Math.ceil(texels / TEXTURE_WIDTH)) };
  };

  const xValues = fill(rows * inChannels, 7);
  const weightValues = fill(inChannels * outChannels, 11);
  const xShape = linearShape(xValues.length);
  const xTexture = makeTexture(xShape.width, xShape.height, xValues);
  const weightLinearShape = linearShape(weightValues.length);
  const weightLinear = makeTexture(weightLinearShape.width, weightLinearShape.height, weightValues);
  // One matrix row per texture row: (IN/4) wide, OUT tall.
  const weightShaped = makeTexture(inChannels / 4, outChannels, weightValues);

  const rowQuads = Math.ceil(rows / ROWS_PER_FRAGMENT);
  const planeShape = linearShape(rowQuads * outChannels);
  const framebuffer = gl.createFramebuffer();
  const vertices = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

  const run = (shaped, splits = 1) => {
    const program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, VERTEX],
      [gl.FRAGMENT_SHADER, FRAGMENT(rows, inChannels, outChannels, shaped, splits)]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(`${shaped ? "shaped" : "linear"}: ${gl.getShaderInfoLog(shader)}`);
      }
      gl.attachShader(program, shader);
    }
    gl.bindAttribLocation(program, 0, "position");
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`${shaped ? "shaped" : "linear"}: ${gl.getProgramInfoLog(program)}`);
    }
    // A ring of plane sets, so no target is written twice - hidden-surface
    // removal discards an overwritten draw and the timing would be a lie.
    const splitShape = linearShape(splits * rowQuads * outChannels);
    const rings = Array.from({ length: iterations }, () => Array.from(
      { length: ROWS_PER_FRAGMENT },
      () => makeTexture(splitShape.width, splitShape.height, new Float32Array(0))));
    const draw = (planes) => {
      gl.useProgram(program);
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      planes.forEach((plane, i) => gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, plane, 0));
      gl.drawBuffers(planes.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, xTexture);
      gl.uniform1i(gl.getUniformLocation(program, "xTex"), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, shaped ? weightShaped : weightLinear);
      gl.uniform1i(gl.getUniformLocation(program, "weightTex"), 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.viewport(0, 0, splitShape.width, splitShape.height);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    for (let i = 0; i < 2; i += 1) draw(rings[i % rings.length]);
    const pixel = new Float32Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, pixel);
    const started = performance.now();
    for (let i = 0; i < iterations; i += 1) draw(rings[i]);
    const out = new Float32Array(splitShape.width * splitShape.height * 4);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D,
      rings[iterations - 1][0], 0);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, splitShape.width, splitShape.height, gl.RGBA, gl.FLOAT, out);
    const ms = (performance.now() - started) / iterations;
    for (const planes of rings) for (const plane of planes) gl.deleteTexture(plane);
    gl.deleteProgram(program);
    return { ms, plane0: out.subarray(0, rowQuads * outChannels) };
  };

  const linearArm = run(false);
  const shapedArm = run(true);
  const splitArms = [2, 4, 8].map((splits) => ({ splits, ...run(false, splits) }));
  const flops = 2 * rows * inChannels * outChannels;
  const residual = relativeRms(linearArm.plane0, shapedArm.plane0);
  gl.deleteTexture(xTexture);
  gl.deleteTexture(weightLinear);
  gl.deleteTexture(weightShaped);
  gl.deleteFramebuffer(framebuffer);
  gl.deleteBuffer(vertices);

  return {
    shape: `${rows}x${inChannels}x${outChannels}`,
    maxTextureSize: maxTexture,
    weightTexture: {
      linear: `${weightLinearShape.width}x${weightLinearShape.height}`,
      shaped: `${inChannels / 4}x${outChannels}`,
    },
    linearIndex: {
      ms: Number(linearArm.ms.toFixed(3)),
      gflops: Number((flops / linearArm.ms / 1e6).toFixed(1)),
    },
    rowPerTextureRow: {
      ms: Number(shapedArm.ms.toFixed(3)),
      gflops: Number((flops / shapedArm.ms / 1e6).toFixed(1)),
    },
    speedup: Number((linearArm.ms / shapedArm.ms).toFixed(2)),
    // 🔴 THE SPLIT-K ARM, which is the question the shape raised: this kernel
    // gets 23 GFLOP/s here and 254 at 2048^3, and the difference is that 32,256
    // fragments cannot hide their own fetch latency. Each split multiplies the
    // fragment count and shortens the chain. The reduction that adds the
    // partials is NOT timed here - it is one pass over S x the output.
    splitK: splitArms.map(({ splits, ms }) => ({
      splits,
      ms: Number(ms.toFixed(3)),
      gflops: Number((2 * rows * inChannels * outChannels / ms / 1e6).toFixed(1)),
      speedup: Number((linearArm.ms / ms).toFixed(2)),
      fragments: splits * Math.ceil(rows / ROWS_PER_FRAGMENT) * (outChannels / 4),
    })),
    relRms: Number(residual.toExponential(2)),
    verdict: residual > 1e-6
      ? `FAILED: the two layouts disagree at ${residual.toExponential(2)} - one of them is `
        + "addressing the wrong texel, and a layout that is faster and wrong is not a result"
      : `ok: one matrix row per texture row is ${(linearArm.ms / shapedArm.ms).toFixed(2)}x the `
        + `linear index at ${rows}x${inChannels}x${outChannels}, same answer to `
        + `${residual.toExponential(2)}`,
  };
}
