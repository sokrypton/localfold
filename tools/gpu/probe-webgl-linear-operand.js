/**
 * Which OPERAND binds the WebGL2 dense projection - x, or the weight codes?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-linear-operand.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-linear-operand.js --rows=576 --in=896 --out=1792
 *
 * 🔴 IT EXISTS BECAUSE THE PROJECTION IS NOW THE FOLD AND NOTHING SAYS WHY.
 * Profiled at 68 tokens and 20 steps, `linear*` is 27.0 s of the 37.1 s of
 * labelled GPU time in a bioemu2 fold - 72.8% - at an aggregate 219 GFLOP/s
 * against the 1958-2003 GFLOP/s `probe-webgl-gemm-ceiling.js` measures for a
 * best-case fragment GEMM. The contraction that used to dominate is 11.8% now.
 * So the next factor is in this kernel, and the question is which of its two
 * streams it is waiting on.
 *
 * 🔴 AND IT IS THE SAME DIAGNOSTIC THAT ALREADY WORKED ONCE, POINTED AT THE
 * OTHER OPERAND. `linear.js`'s own comment records it: "make every chunk read
 * the SAME four code texels and the fold goes 6664 ms to 3476 - so this kernel
 * was bound by code-fetch MISSES, not by bandwidth". That is the weight-side
 * arm, and it is what chose `orderRowFastest`. Nothing has ever run the x-side
 * twin, and the traffic arithmetic says x is now the larger stream by 9x: a
 * fragment reads `blockRows` rows of x and 4 output channels of codes, so x is
 * re-read once per OUTPUT GROUP (448 times at 1792 outputs) and the codes once
 * per ROW BLOCK (72 times at 576 rows) - 926 MB against 100 MB at this shape.
 *
 * The three arms differ ONLY in which texels a fragment addresses. Same
 * fragment count, same fetch count, same decode, same multiply-adds:
 *
 *   shipped    `q` from the texel index, so the fragments cover the real grid.
 *   x-aliased  every fragment reads row block 0. x collapses to one block's
 *              worth and stays in cache; the codes stream exactly as before.
 *   w-aliased  every fragment reads output group 0. The codes collapse to four
 *              channels; x streams exactly as before.
 *
 * A big x-aliased speedup means x binds; a big w-aliased one means the codes
 * still bind despite `orderRowFastest`; both small means neither stream is the
 * constraint and the kernel is issue-bound.
 *
 * 🔴 MEASURED AT 576x896x1792, THE FOLD'S LARGEST ROW: x BINDS AT 3.44x AND
 * THE CODES COST NOTHING (1.09x). 10.06 ms shipped at 184 GFLOP/s against 2.93
 * ms x-aliased at 632 and 9.25 ms w-aliased at 200. So `orderRowFastest`
 * finished the weight side and every remaining factor is on the x side - and
 * two of them are taken: x in float16 is 1.31x on its own (7.68 ms) and the
 * tile-aware order below is 1.96x (5.13 ms), 2.56x together (3.94 ms at 470
 * GFLOP/s).
 *
 * 🔴 AND THE FIRST VERSION OF THESE FIGURES READ 6.62x AND IS RETRACTED - see
 * the straw-baseline note at the end of this header. So `orderRowFastest` finished the weight side and
 * every remaining factor in this kernel is on the x side.
 *
 * 🔴 AND THE OBVIOUS CONCLUSION FROM THAT IS WRONG, WHICH IS WHY THE ASPECT
 * SWEEP IS HERE. x is re-read once per OUTPUT GROUP, so a wider output tile of
 * the same 32-float budget reads proportionally less x - 882 MB at 8x4, 441 at
 * 4x8, 220 at 2x16, 110 at 1x32 - and the codes it reads more of were just
 * measured free. Every one of them is SLOWER, monotonically:
 *
 * | tile | ms | GFLOP/s | x MB | code MB | factor |
 * |---|---:|---:|---:|---:|---:|
 * | **8x4** (shipped) | **19.24** | **96** | 882 | 68.9 | **1.00** |
 * | 4x8 | 22.95 | 81 | 441 | 137.8 | 0.84 |
 * | 2x16 | 27.04 | 68 | 220.5 | 275.6 | 0.71 |
 * | 1x32 | 35.73 | 52 | 110.3 | 551.3 | 0.54 |
 *
 * **Halving x's bytes makes the kernel slower.** So what binds is not x's
 * VOLUME but its fetch LOCALITY: aliasing x does not merely read less, it makes
 * every concurrent fragment read the SAME eight rows, and a wider output tile
 * keeps the same scattered pattern while multiplying the decode work and the
 * live accumulators. The shipped 8x4 is already the best aspect of its budget,
 * and three of these four arms are measured nulls - the lever is a fragment
 * ORDER that puts nearby x under one rasterisation tile, which is what was
 * worth 3.29x on the contraction, and NOT a reshaped output tile.
 *
 * 🔴 AND ITS OWN BASELINE IS A STRAW ONE: READ THE RATIOS, NEVER THE
 * MILLISECONDS. This probe lays x out UNPADDED where a fold stride-pads it -
 * 576x896 is 224 texels a row and 224 does not divide 2048, so `linear`'s
 * `padStride` repacks it and that alone is the 4.08x this tree has measured. So
 * the shipped arm here reads 18.76 ms where the fold's own profile says 10.63
 * for the same shape, 1.77x apart, and its `tiled-2` arm's **1.87x was measured
 * against a baseline whose x reads were misaligned**: the tiling was recovering
 * an alignment the real kernel already has. Implemented in `linear.js` and
 * measured in the fold, the same mapping is **1.12x and 1.19x** on the two
 * kernels and **1.082x** on the whole bioemu2 fold - 30.40 s to 28.10 s,
 * bit-identical. Real, and half what this probe promised. A probe that does not
 * reproduce the shipped kernel's own addressing overstates every lever it finds.
 *
 * 🔴 THE CODE BITS ARE ARBITRARY AND THIS IS NOT A DIFFERENTIAL. It measures
 * TIME at the shipped kernel's exact traffic and arithmetic; the weights decode
 * to whatever those bits mean. `witnessFinite` is there so a run that decoded
 * to NaN - which could change the timing - is visible rather than silent. The
 * kernel's correctness is `check-webgl-kernels.js` and `bench-webgl-linear.js`.
 *
 * 🔴 AND IT USES A RING OF DISTINCT TARGETS, because this GPU does
 * hidden-surface removal: drawing the same full-viewport triangle repeatedly
 * into ONE target discards the earlier fragments before shading them and the
 * probe measures one iteration divided by the count. That trap has cost three
 * measurements in this tree.
 */
import { createWebGL2Context } from "../../webgl/index.js";
import {
  INT5_DECODE, INT5_DECODE8, INT5_GROUP, INT5_GROUP_WORDS,
} from "../../webgl/runtime/int5.js";

const TEXTURE_W = 2048;
const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const FULLSCREEN_TRIANGLE = new Float32Array([-1, -1, 3, -1, -1, 3]);
const VERTEX_SHADER = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`;

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
 * The shipped int5 blocked projection's inner loop, with the operand addressing
 * under a flag. `arm` is "shipped", "x-aliased" or "w-aliased".
 *
 * 🔴 NO BACKTICKS IN THE GLSL - this is a template literal, and one backtick in
 * a comment here ends the string. That has happened eight times in this tree.
 */
const shader = (inChannels, outChannels, rowsPerFragment, outsPerFragment,
  rowBlocks, arm, tileWidth = 0, xStride4 = inChannels / 4, halfX = false) => {
  const R = rowsPerFragment;
  const C = outsPerFragment;
  const quads = C / 4;
  const rows = (f) => Array.from({ length: R }, (_, i) => f(i)).join("\n");
  const chans = (f) => Array.from({ length: C }, (_, i) => f(i)).join("\n");
  return `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
const int W = 2048;
const int IN = ${inChannels};
const int GPR = ${inChannels / INT5_GROUP};
const int OUTC = ${outChannels / C};
const int RQ = ${rowBlocks};
const int XSTRIDE4 = ${xStride4};
uniform sampler2D xTex;
uniform highp usampler2D codeTex;
uniform sampler2D tableTex;
${Array.from({ length: R * quads }, (_, i) =>
  `layout(location = ${i}) out vec4 result${i};`).join("\n")}
${INT5_DECODE}
${INT5_DECODE8}

vec4 fetch4(sampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % W, texel / W), 0);
}

void main() {
  int texel = int(gl_FragCoord.y) * W + int(gl_FragCoord.x);
${tileWidth > 0
    ? `  // 🔴 TILE-AWARE: a rasterisation tile is 2D and linearTexel() is 1D, so a
  // TILEW-wide run covers TILEW consecutive ROW BLOCKS at ONE output group
  // instead of ${rowBlocks} of them - the same mapping that was worth 3.29x on
  // the contraction. ${tileWidth} row blocks is ${tileWidth} * ${rowsPerFragment} rows of x a tile.
  const int TILEW = ${tileWidth};
  int lane = texel % TILEW;
  int group = (texel % W) / TILEW;
  int q = (texel / W) * TILEW + lane;
  // 🔴 AND THIS GRID IS PADDED WHERE THE SHIPPED ONE IS NOT, so the arm must
  // PAY for its padding or it is not comparable: q advances by TILEW a texel
  // row, so covering ${rowBlocks} row blocks takes ceil(${rowBlocks}/TILEW) rows of W and
  // the leftover fragments are real draws. They return here instead of wrapping
  // with a modulo, because wrapping would make them do a second copy of real
  // work and flatter the arm from the other side.
  if (q >= RQ || group >= ${outChannels / outsPerFragment}) {
${Array.from({ length: rowsPerFragment * (outsPerFragment / 4) }, (_, i) =>
      `    result${i} = vec4(0.0);`).join("\n")}
    return;
  }`
    : `  // Row block fastest, as the int5 path ships.
  int group = texel / RQ;
  int q = texel - group * RQ;`}
  ${arm === "w-aliased" ? "group = 0;" : "group = group % OUTC;"}
  ${arm === "x-aliased" ? "q = 0;" : "q = q % RQ;"}
  int o0 = group * ${C};
  int base = q * ${R};
${rows((i) => `  int xBase${i} = (base + ${i}) * XSTRIDE4;`)}
${rows((i) => Array.from({ length: quads }, (_, c) =>
  `  vec4 acc${i}_${c} = vec4(0.0);`).join("\n"))}
  for (int g = 0; g < GPR; ++g) {
${chans((o) => `    vec2 sz${o} = int5ScaleZero(tableTex, (o0 + ${o}) * GPR + g);`)}
    for (int chunk = 0; chunk < ${INT5_GROUP / 8}; ++chunk) {
      int k = g * ${INT5_GROUP} + chunk * 8;
${chans((o) => `      vec4 w${o}a, w${o}b;
      decodeInt5x8(codeTex, (o0 + ${o}) * IN + k, sz${o}.x, sz${o}.y, w${o}a, w${o}b);`)}
      int kq = k >> 2;
${rows((i) => `      {
        vec4 xa = fetch4(xTex, xBase${i} + kq);
        vec4 xb = fetch4(xTex, xBase${i} + kq + 1);
${Array.from({ length: quads }, (_, c) => `        acc${i}_${c} += vec4(
          dot(xa, w${c * 4 + 0}a) + dot(xb, w${c * 4 + 0}b),
          dot(xa, w${c * 4 + 1}a) + dot(xb, w${c * 4 + 1}b),
          dot(xa, w${c * 4 + 2}a) + dot(xb, w${c * 4 + 2}b),
          dot(xa, w${c * 4 + 3}a) + dot(xb, w${c * 4 + 3}b));`).join("\n")}
      }`)}
    }
  }
${rows((i) => Array.from({ length: quads }, (_, c) =>
  `  result${i * quads + c} = acc${i}_${c};`).join("\n"))}
}`;
};

export async function main(device, args) {
  const rows = Number(option(args, "rows", "576"));
  const inChannels = Number(option(args, "in", "896"));
  const outChannels = Number(option(args, "out", "1792"));
  const rounds = Number(option(args, "rounds", "4"));
  const iterations = Number(option(args, "iterations", "8"));
  const gl = createWebGL2Context();
  const targets = gl.getParameter(gl.MAX_DRAW_BUFFERS);
  if (inChannels % INT5_GROUP !== 0) {
    throw new Error(`in ${inChannels} is not a multiple of the int5 group ${INT5_GROUP}`);
  }
  if (outChannels % 4 !== 0) throw new Error(`out ${outChannels} is not a multiple of 4`);
  // 🔴 THE TILES ARE THE ASPECTS OF ONE FIXED BUDGET, which is what makes them
  // comparable: `MAX_DRAW_BUFFERS * 4` output floats a fragment, so R rows by C
  // output channels with R*C = 32 and C a multiple of four. 8x4 is what ships.
  const budget = targets * 4;
  // 🔴 THE TILE-AWARE ARMS, which the aspect sweep's nulls point at. A tile must
  // hold every output group the mapping gives a texel ROW, so TILEW is at most
  // `W / outputGroups`; a wider one would alias two groups onto one texel.
  const tileWidths = (option(args, "tile-widths", "") !== ""
    ? option(args, "tile-widths", "").split(",").map(Number)
    : [2, 4, 8]).filter((t) => t > 0);
  // 🔴 AND THE BOUND IS ENFORCED, NOT STATED. Written as a comment alone it let
  // `tiled-8` through at 1792 outputs: W/8 is 256 group slots against 448
  // output groups, so the mapping capped `group` at 255 and the arm computed
  // 57% of the work and reported 104 GFLOP/s as though it were comparable. The
  // tell was `wastedFragments` going NEGATIVE - an arm whose grid is smaller
  // than its own useful work is not a measurement.
  const outputGroups = Math.ceil(outChannels / 4);
  const aspects = (option(args, "tiles", "") !== ""
    ? option(args, "tiles", "").split(",")
    : ["8x4", "4x8", "2x16", "1x32"]).map((spec) => {
    const [R, C] = spec.split("x").map(Number);
    if (!(R > 0) || !(C > 0) || C % 4 !== 0 || R * C !== budget) {
      throw new Error(`tile ${spec}: R*C must be ${budget} with C a multiple of four`);
    }
    if (outChannels % C !== 0) throw new Error(`tile ${spec}: ${C} does not divide out ${outChannels}`);
    return { spec, R, C };
  });

  const fill = (count, seed) => {
    const values = new Float32Array(count);
    let state = seed | 0;
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      values[i] = ((state >>> 8) / 0x7fffff - 0.5) * 0.25;
    }
    return values;
  };
  const words = (count, seed) => {
    const out = new Uint32Array(count);
    let state = seed | 0;
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = state >>> 0;
    }
    return out;
  };
  const W = 2048;
  const floatTexture = (values) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    const texels = Math.ceil(values.length / 4);
    const height = Math.max(1, Math.ceil(texels / W));
    const padded = new Float32Array(W * height * 4);
    padded.set(values);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, height, 0, gl.RGBA, gl.FLOAT, padded);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return t;
  };
  // x in float16, which is what the WGSL path keeps its intermediates in and
  // what this backend does NOT - half the bytes of the stream that binds.
  const halfTexture = (values) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    const texels = Math.ceil(values.length / 4);
    const height = Math.max(1, Math.ceil(texels / W));
    const padded = new Float32Array(W * height * 4);
    padded.set(values);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, W, height, 0, gl.RGBA, gl.FLOAT, padded);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return t;
  };
  const wordTexture = (values) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    const texels = Math.ceil(values.length / 4);
    const height = Math.max(1, Math.ceil(texels / W));
    const padded = new Uint32Array(W * height * 4);
    padded.set(values);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, W, height, 0,
      gl.RGBA_INTEGER, gl.UNSIGNED_INT, padded);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return t;
  };

  const groups = (inChannels / INT5_GROUP) * outChannels;
  // 🔴 x AT THE STRIDE A FOLD GIVES IT, which is what this probe got wrong and
  // what made its first `tiled-2` reading 1.87x where the fold's is 1.12x. A
  // row of `in/4` texels wraps onto the next texture row unless that divides W,
  // so `linear` runs `padStride` over x - 4.08x on its own - and a probe laying
  // x out unpadded measures a kernel nothing dispatches. `--no-stride-pad` is
  // the arm that prices it here as it does in a fold.
  const unpadded = inChannels / 4;
  const divides = TEXTURE_W % unpadded === 0;
  const padStride = !flag(args, "no-stride-pad") && !divides;
  const xStride4 = padStride
    ? 2 ** Math.ceil(Math.log2(unpadded))
    : unpadded;
  const xValues = new Float32Array(rows * xStride4 * 4);
  {
    const source = fill(rows * inChannels, 7);
    for (let r = 0; r < rows; r += 1) {
      xValues.set(source.subarray(r * inChannels, (r + 1) * inChannels), r * xStride4 * 4);
    }
  }
  const xTex = floatTexture(xValues);
  const xHalfTex = halfTexture(xValues);
  const codeTex = wordTexture(words(groups * INT5_GROUP_WORDS, 11));
  // The scale/zero table is two floats a group, as `int5ScaleZero` reads it.
  const tableTex = floatTexture(fill(groups * 2, 13));

  // A viewport that covers the widest arm's fragments in the backend's own
  // 2048 rows; a narrower arm draws the same grid and masks with `% RQ`.
  const extentFor = (R, C, tileWidth) => {
    const rowBlocks = Math.ceil(rows / R);
    const height = tileWidth > 0
      ? Math.ceil(rowBlocks / tileWidth)
      : Math.ceil(rowBlocks * (outChannels / C) / W);
    return { height, fragments: height * W, useful: rowBlocks * (outChannels / C) };
  };
  const height = Math.max(...aspects.map((a) => extentFor(a.R, a.C, 0).height),
    ...tileWidths.map((t) => extentFor(8, 4, t).height));
  if (!(outputGroups > 0)) throw new Error("no output groups");
  const ring = Array.from({ length: iterations }, () => {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const attachments = [];
    for (let t = 0; t < targets; t += 1) {
      const out = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, out);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, height, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + t, gl.TEXTURE_2D, out, 0);
      attachments.push(gl.COLOR_ATTACHMENT0 + t);
    }
    gl.drawBuffers(attachments);
    return framebuffer;
  });

  const arms = [];
  const refusals = [];
  for (const { spec, R, C } of aspects) {
    const rowBlocks = Math.ceil(rows / R);
    const xBytes = (outChannels / C) * rows * inChannels * 4;
    const codeBytes = rowBlocks * groups * INT5_GROUP_WORDS * 4;
    // The aliasing arms only ever run on the shipped aspect: they answer which
    // STREAM binds, and the sweep answers what to do about it.
    const legal = tileWidths.filter((t) => Math.floor(W / t) >= outChannels / C);
    const refused = tileWidths.filter((t) => !legal.includes(t))
      .map((t) => `tiled-${t}: W/${t} is ${Math.floor(W / t)} group slots against `
        + `${outChannels / C} output groups`);
    refusals.push(...(spec === "8x4" ? refused : []));
    const armNames = spec === "8x4"
      ? ["shipped", "x-aliased", "w-aliased", "half-x",
        ...legal.map((t) => `tiled-${t}`),
        ...legal.map((t) => `tiled-${t}-half-x`)]
      : ["shipped"];
   for (const arm of armNames) {
    const tileWidth = arm.startsWith("tiled-") ? Number(arm.slice(6).split("-")[0]) : 0;
    const halfX = arm.endsWith("half-x");
    const extent = extentFor(R, C, tileWidth);
    // `half-x` alone is the shipped order with a narrower x, so its grid is the
    // shipped one; `tiled-N-half-x` is the tiled grid with a narrower x.

    // 🔴 USEFUL FLOPS, NOT DISPATCHED ONES: a padded grid is charged its real
    // milliseconds and credited only the work the kernel would keep.
    const flops = extent.useful * R * C * inChannels * 2;
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER,
      shader(inChannels, outChannels, R, C, rowBlocks, arm, tileWidth, xStride4, halfX)));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`link ${spec} ${arm}: ${gl.getProgramInfoLog(program)}`);
    }
    gl.useProgram(program);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, FULLSCREEN_TRIANGLE, gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, halfX ? xHalfTex : xTex);
    gl.uniform1i(gl.getUniformLocation(program, "xTex"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, codeTex);
    gl.uniform1i(gl.getUniformLocation(program, "codeTex"), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, tableTex);
    gl.uniform1i(gl.getUniformLocation(program, "tableTex"), 2);
    gl.viewport(0, 0, W, extent.height);

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
    arms.push({
      tile: `${R}x${C}`,
      arm,
      fragments: extent.fragments,
      wastedFragments: extent.fragments - extent.useful,
      milliseconds: Number(ms.toFixed(3)),
      spread: Number((Math.max(...samples) / Math.min(...samples)).toFixed(2)),
      gigaflops: Number((flops / ms / 1e6).toFixed(0)),
      xMegabytes: Number((xBytes * (halfX ? 0.5 : 1) / 1048576).toFixed(1)),
      codeMegabytes: Number((codeBytes / 1048576).toFixed(1)),
      witnessFinite: Number.isFinite(witness),
    });
   }
  }

  const of = (tile, name) => arms.find((a) => a.tile === tile && a.arm === name)?.milliseconds;
  const shipped = of("8x4", "shipped");
  const xFactor = shipped / of("8x4", "x-aliased");
  const wFactor = shipped / of("8x4", "w-aliased");
  const binds = xFactor >= 1.3 || wFactor >= 1.3
    ? (xFactor >= wFactor ? "x" : "the weight codes")
    : "neither stream";
  const sweep = arms.filter((a) => a.arm === "shipped" || a.arm.startsWith("tiled-")
    || a.arm === "half-x")
    .map((a) => ({ arm: a.arm, tile: a.tile, milliseconds: a.milliseconds,
      gigaflops: a.gigaflops, wastedFragments: a.wastedFragments,
      factorOverShipped: Number((shipped / a.milliseconds).toFixed(2)) }));
  const best = sweep.reduce((a, b) => (b.milliseconds < a.milliseconds ? b : a));
  const bestName = best.arm === "shipped" ? `the shipped ${best.tile}` : `${best.arm} at ${best.tile}`;
  return {
    renderer: gl.getParameter(gl.RENDERER),
    maxDrawBuffers: targets,
    shape: `${rows}x${inChannels}x${outChannels}`,
    xStride4,
    xStridePadded: padStride,
    budget: `${budget} output floats a fragment`,
    arms,
    refusedTileWidths: refusals,
    sweep,
    xAliasedFactor: Number(xFactor.toFixed(2)),
    wAliasedFactor: Number(wFactor.toFixed(2)),
    binds,
    bestTile: best.tile,
    bestArm: best.arm,
    bestFactor: best.factorOverShipped,
    verdict: binds === "neither stream"
      ? `NEITHER OPERAND BINDS: aliasing x is ${xFactor.toFixed(2)}x and aliasing the codes `
        + `${wFactor.toFixed(2)}x, so this kernel is issue-bound`
      : `${binds} BINDS at ${xFactor.toFixed(2)}x against the codes' ${wFactor.toFixed(2)}x, `
        + `and the best arm of the same budget is ${bestName} at `
        + `${best.factorOverShipped.toFixed(2)}x the shipped 8x4`,
  };
}
