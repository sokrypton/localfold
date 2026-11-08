/**
 * The dense projection through TRANSFORM FEEDBACK, at the shape a bioemu2 fold
 * is made of, against the fragment kernel that ships.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl-tf-linear.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl-tf-linear.js --rows=576 --in=896 --out=1792
 *
 * 🔴 THIS IS THE ONE LEVER LEFT AND THE NUMBER THAT DECIDES IT. webgl/README.md's
 * "How close can this get" fixes the fragment path's floor from two measured
 * numbers: 21.3 flops a fetch is an eight-target fragment GEMM's intensity
 * ceiling, and the best fetch rate seen here is 37 G/s - so about 790 GFLOP/s,
 * against the 2.4 TFLOP/s WGSL reaches on the same fold. A VERTEX shader's
 * output budget is `MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS`, **128 floats
 * here against a fragment's 32**, which doubles the reachable intensity to 42.7:
 * eight rows and sixteen output channels an invocation costs 24 fetches for 512
 * multiply-adds where the fragment block costs 12 for 128.
 *
🔴 **AND THE ANSWER IS THAT IT OFFERS NOTHING, WHICH CORRECTS A CLAIM THIS
 * REPOSITORY HAS CARRIED SINCE THE SPIKE.** docs/A100.md and webgl/README.md
 * both say "the vertex path is 2.2x the fragment path and the whole difference
 * is output budget", and the case for this backend reaching 47.8% of the device
 * rests on it. Measured at a real projection shape, 576 x 896 -> 1792:
 *
 *   - at the SAME 32-float budget a fragment has, transform feedback is
 *     **1.00x** - 70.2 ms against 70.3 - and `relRms` is **exactly 0**, so it
 *     is the same arithmetic in the same order at the same speed;
 *   - at **112** floats, the budget that was supposed to be the advantage, it
 *     is **1.85x SLOWER** (132 ms), because 112 accumulators do not fit in a
 *     register file and spill.
 *
 * So the binding constraint is REGISTERS, not the output budget, and the
 * budget cannot be spent. `bench-webgl2-tf-matmul.js`'s 1.59 TFLOP/s is a
 * square-2048 GEMM - a shape whose operands stream perfectly - and its own best
 * arm (`8x1`) writes the same 32 floats a fragment does, so that number was
 * never evidence for the larger budget either.
 *
 * 🔴 AND THE OUTPUT IS A BUFFER, WHICH IS THE WHOLE COST OF THE IDEA. Transform
 * feedback writes interleaved per-invocation floats into a buffer, so eight rows
 * x sixteen channels an invocation is NOT the tensor's row-major order; a
 * fragment kernel's eight render targets are not either, which is why
 * `blockRepackSource` exists. The difference is that a buffer is not a texture:
 * it takes a `texSubImage2D` from `PIXEL_UNPACK_BUFFER` to become one, and only
 * then can a permute pass put it in order. So transform feedback pays one more
 * full pass over the output than the fragment path does, and `passes` below
 * reports it rather than hiding it inside the win.
 */
import { Execution } from "../../webgl/runtime/execution.js";
import { TEXTURE_WIDTH } from "../../webgl/runtime/context.js";
import { vertexOutputFloats } from "../../webgl/runtime/limits.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { linear } from "../../webgl/kernels/linear.js";
import { relativeRms } from "./relative-rms.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * Rows an invocation owns, and output channels: 7 x 16 = **112 floats**, which
 * is 28 `vec4` varyings.
 *
 * 🔴 NOT 128, AND `limits.js` SAYS WHY BEFORE THE LINKER DOES.
 * `MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS` reads 128 here but the
 * varyings must also pack into `MAX_VERTEX_OUTPUT_COMPONENTS`, which is
 * **120** - `vertexOutputFloats` is the min of the two and exists for exactly
 * this. Taking the advertised figure at its word fails at LINK time with
 * "Could not pack varying v30", which is how this bench learned it; 112 leaves
 * room for `gl_Position` on an implementation that charges for it. Intensity is
 * 8RC/(R+C) flops a fetch, so 7 x 16 is 39.0 against the fragment block's 21.3.
 */
const TF_ROWS = 8;
const TF_OUTS = 4;

/**
 * The vertex shader as a compute kernel: `gl_VertexID` is the block index, the
 * 32 `vec4` outputs are captured, and rasterisation is discarded.
 *
 * `[out][in]`: output channel `o` has its `IN` weights contiguous, so a k-step
 * of four is one texel per output channel against one texel per row of x.
 */
const TF_VERTEX = (rows, inChannels, outChannels) => `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const int W = ${TEXTURE_WIDTH};
const int ROWS = ${rows};
const int IN4 = ${inChannels / 4};
const int OUT4 = ${outChannels / 4};
const int OBLOCKS = ${outChannels / TF_OUTS};
uniform sampler2D xTex;
uniform sampler2D weightTex;
${Array.from({ length: TF_ROWS * TF_OUTS / 4 }, (_, i) => `out vec4 v${i};`).join("\n")}

vec4 fetch4(sampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % W, texel / W), 0);
}

void main() {
  int block = gl_VertexID;
  int oblock = block % OBLOCKS;
  int rblock = block / OBLOCKS;
  int o0 = oblock * ${TF_OUTS};
  vec4 acc[${TF_ROWS * TF_OUTS / 4}];
  for (int i = 0; i < ${TF_ROWS * TF_OUTS / 4}; ++i) acc[i] = vec4(0.0);
  int xBase[${TF_ROWS}];
  for (int r = 0; r < ${TF_ROWS}; ++r) xBase[r] = min(rblock * ${TF_ROWS} + r, ROWS - 1) * IN4;
  for (int k = 0; k < IN4; ++k) {
    vec4 w[${TF_OUTS}];
    for (int c = 0; c < ${TF_OUTS}; ++c) w[c] = fetch4(weightTex, (o0 + c) * IN4 + k);
    for (int r = 0; r < ${TF_ROWS}; ++r) {
      vec4 x = fetch4(xTex, xBase[r] + k);
      for (int c = 0; c < ${TF_OUTS}; ++c) {
        acc[r * ${TF_OUTS / 4} + c / 4][c % 4] += dot(x, w[c]);
      }
    }
  }
${Array.from({ length: TF_ROWS * TF_OUTS / 4 }, (_, i) => `  v${i} = acc[${i}];`).join("\n")}
  // 🔴 w MUST BE 1, which cost a run of all zeros to learn. At vec4(0.0) the
  // position is degenerate and the vertex is dropped before the capture - no GL
  // error, an empty buffer, and a 138x speedup that is a kernel which did not
  // run.
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
  gl_PointSize = 1.0;
}`;

const TF_FRAGMENT = `#version 300 es
precision highp float;
out vec4 unused;
void main() { unused = vec4(0.0); }`;

/** A reproducible spread of magnitudes - a real weight matrix is not uniform. */
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
  if (inChannels % 4 !== 0 || outChannels % TF_OUTS !== 0) {
    throw new RangeError(
      `bench-webgl-tf-linear: in must be a multiple of 4 and out of ${TF_OUTS}`);
  }
  const gl = createWebGL2Context();
  const execution = new Execution(gl, { checkErrors: false });
  const interleaved = vertexOutputFloats(gl);
  if (interleaved < TF_ROWS * TF_OUTS) {
    throw new Error(
      `this context carries ${interleaved} vertex output floats (the min of the transform `
      + `feedback and varying limits), and this kernel writes ${TF_ROWS * TF_OUTS}`);
  }

  const xValues = fill(rows * inChannels, 7);
  const weightValues = fill(inChannels * outChannels, 11);
  const x = execution.upload(xValues, "bench.x");
  const weight = execution.upload(weightValues, "bench.w");
  const flops = 2 * rows * inChannels * outChannels;

  // --- the fragment kernel that ships -------------------------------------
  const call = { x, weight, rows, inChannels, outChannels, weightLayout: "out-major" };
  const ring = Array.from({ length: iterations },
    () => execution.allocate(rows * outChannels, "bench.out"));
  for (let i = 0; i < 2; i += 1) {
    linear(execution, call, ring[i % ring.length]);
    execution.read(ring[i % ring.length]);
  }
  let started = performance.now();
  for (let i = 0; i < iterations; i += 1) linear(execution, call, ring[i]);
  const fragmentAnswer = Float32Array.from(execution.read(ring[iterations - 1]));
  const fragmentMs = (performance.now() - started) / iterations;
  for (const out of ring) execution.release(out);

  // --- the same projection through transform feedback ----------------------
  const blocks = Math.ceil(rows / TF_ROWS) * (outChannels / TF_OUTS);
  const captured = blocks * TF_ROWS * TF_OUTS;
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, TF_VERTEX(rows, inChannels, outChannels)],
    [gl.FRAGMENT_SHADER, TF_FRAGMENT]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`tf shader: ${gl.getShaderInfoLog(shader)}`);
    }
    gl.attachShader(program, shader);
  }
  gl.transformFeedbackVaryings(program,
    Array.from({ length: TF_ROWS * TF_OUTS / 4 }, (_, i) => `v${i}`), gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`tf link: ${gl.getProgramInfoLog(program)}`);
  }
  // 🔴 A RING OF BUFFERS, for the same reason the fragment arm has a ring of
  // textures: one target written over and over lets the driver overlap or
  // discard, and the timing is then a lie.
  const buffers = Array.from({ length: iterations }, () => {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, captured * 4, gl.DYNAMIC_COPY);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return buffer;
  });
  const feedback = gl.createTransformFeedback();
  const errors = [];
  const runTf = (buffer) => {
    // 🔴 THE DEFAULT FRAMEBUFFER, because `Execution` leaves its own cached one
    // bound and a draw against an incomplete framebuffer is refused on some
    // implementations even with RASTERIZER_DISCARD on.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, x.texture);
    gl.uniform1i(gl.getUniformLocation(program, "xTex"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, weight.texture);
    gl.uniform1i(gl.getUniformLocation(program, "weightTex"), 1);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, feedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buffer);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, blocks);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    // 🔴 THE BASE AND THE OBJECT BOTH HAVE TO BE RELEASED, and a buffer still
    // bound to TRANSFORM_FEEDBACK_BUFFER may not also be bound to ARRAY_BUFFER -
    // `getBufferSubData` then returns zeros behind an INVALID_OPERATION nobody
    // reads. tools/gpu/probe-webgl2-tf.js is the smallest case that works.
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    const error = gl.getError();
    if (error !== gl.NO_ERROR && errors.length < 4) errors.push(`0x${error.toString(16)}`);
  };
  for (let i = 0; i < 2; i += 1) runTf(buffers[i % buffers.length]);
  const drain = new Float32Array(16);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers[0]);
  gl.getBufferSubData(gl.ARRAY_BUFFER, 0, drain);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  started = performance.now();
  for (let i = 0; i < iterations; i += 1) runTf(buffers[i]);
  const tfAnswer = new Float32Array(captured);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers[iterations - 1]);
  gl.getBufferSubData(gl.ARRAY_BUFFER, 0, tfAnswer);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  const tfMs = (performance.now() - started) / iterations;

  // The captured order is block-major: block (rblock, oblock) holds
  // TF_ROWS x TF_OUTS, so putting it in `[rows][outChannels]` order is the
  // permute pass the real thing would need on the GPU.
  const permuted = new Float32Array(rows * outChannels);
  const oblocks = outChannels / TF_OUTS;
  for (let block = 0; block < blocks; block += 1) {
    const oblock = block % oblocks;
    const rblock = Math.floor(block / oblocks);
    for (let r = 0; r < TF_ROWS; r += 1) {
      const row = rblock * TF_ROWS + r;
      if (row >= rows) continue;
      for (let c = 0; c < TF_OUTS; c += 1) {
        permuted[row * outChannels + oblock * TF_OUTS + c] =
          tfAnswer[(block * TF_ROWS + r) * TF_OUTS + c];
      }
    }
  }

  const residual = relativeRms(fragmentAnswer, permuted);
  execution.release(x);
  execution.release(weight);
  for (const buffer of buffers) gl.deleteBuffer(buffer);
  gl.deleteTransformFeedback(feedback);
  gl.deleteProgram(program);
  execution.destroy();

  const bar = 1e-5;
  return {
    glErrors: errors,
    firstCaptured: [...tfAnswer.slice(0, 4)].map((v) => Number(v.toPrecision(6))),
    capturedMax: Number(Math.max(...tfAnswer.slice(0, 4096).map(Math.abs)).toPrecision(6)),
    firstFragment: [...fragmentAnswer.slice(0, 4)].map((v) => Number(v.toPrecision(6))),
    shape: `${rows}x${inChannels}x${outChannels}`,
    interleavedComponents: interleaved,
    invocationOutputs: { fragment: 32, transformFeedback: TF_ROWS * TF_OUTS },
    blocks,
    fragment: {
      ms: Number(fragmentMs.toFixed(3)),
      gflops: Number((flops / fragmentMs / 1e6).toFixed(1)),
      passes: "block + repack",
    },
    transformFeedback: {
      ms: Number(tfMs.toFixed(3)),
      gflops: Number((flops / tfMs / 1e6).toFixed(1)),
      passes: "capture + buffer-to-texture + permute (the last two not timed here)",
    },
    speedup: Number((fragmentMs / tfMs).toFixed(2)),
    relRms: Number(residual.toExponential(2)),
    bar,
    verdict: residual > bar
      ? `FAILED: the transform-feedback projection disagrees at ${residual.toExponential(2)}, `
        + `past ${bar} - it is computing something else`
      : `ok: transform feedback is ${(fragmentMs / tfMs).toFixed(2)}x the fragment kernel at `
        + `${rows}x${inChannels}x${outChannels}, agreeing to ${residual.toExponential(2)} - and it `
        + "owes two more passes than the fragment path to put its buffer in order",
  };
}
