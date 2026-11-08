/**
 * Is a UNIFORM BLOCK read cheaper than a `texelFetch` on this device?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-uniform-reads.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-uniform-reads.js --reads=2048
 *
 * 🔴 THE ONE MEMORY PATH THIS BACKEND HAS NEVER USED, AND THE ONLY ONE THAT
 * COULD BREAK ITS CEILING. webgl/README.md's bound is arithmetic: a fragment
 * writes at most `MAX_DRAW_BUFFERS * 4` = 32 output floats, a register tile of
 * R by C costs `R + C` fetches a k-step, and the best `R + C` is 12 - so 21.33
 * flops a `fetch4`, and every kernel's operands come through the TEXTURE unit.
 *
 * But GLSL ES 300 has UNIFORM BLOCKS, and they are a different path: a constant
 * cache, broadcast-friendly, and dynamically indexable. This context reports
 * `MAX_UNIFORM_BLOCK_SIZE` **16384 bytes** - 1024 `vec4`s - and
 * `MAX_FRAGMENT_UNIFORM_BLOCKS` **16**, so up to **65,536 floats** of operand
 * can sit in uniforms for one draw. A projection whose SHARED operand lived
 * there would read it off the constant cache and fetch only the streaming one:
 * eight fetches a k-step instead of twelve, which is 32 flops a fetch rather
 * than 21.33 - past the bound this file has been quoting as fixed.
 *
 * 🔴 SO THE QUESTION IS WHETHER THE READ IS ACTUALLY CHEAPER, AND IT IS A
 * QUESTION BECAUSE A DYNAMICALLY INDEXED UNIFORM ARRAY IS NOT A CONSTANT. A
 * compiler may lower it to exactly the same memory the texture unit reads, or
 * spill it, and on some hardware a divergent uniform index is serialised. This
 * measures both paths over the same count of reads in the same shape of loop,
 * with its own GL setup so nothing of the runtime's is in the way.
 *
 * Read `uniformOverTexture`: above one, uniforms are the cheaper path and the
 * ceiling is not where webgl/README.md says it is. At or below one, it is.
 */
import { FULLSCREEN_TRIANGLE, VERTEX_SHADER, compile, createWebGL2Context }
  from "../../webgl/runtime/context.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** The loop body is identical in both arms; only where `v` comes from differs. */
const source = (reads, vectors, uniform) => `#version 300 es
precision highp float;
precision highp int;
const int READS = ${reads};
const int VECTORS = ${vectors};
layout(location = 0) out vec4 result;
${uniform ? `layout(std140) uniform Block { vec4 data[VECTORS]; } block;`
    : "uniform highp sampler2D dataTex;"}

void main() {
  int base = int(gl_FragCoord.x) + int(gl_FragCoord.y) * 2048;
  vec4 total = vec4(0.0);
  // A stride with period VECTORS, so no unrolling can hoist the reads - see
  // probe-webgl-fetch-rate.js, whose first version measured a register read.
  for (int i = 0; i < READS / 4; ++i) {
${[0, 1, 2, 3].map((o) => {
    const index = `((i * 53 + ${o * 257} + base) % VECTORS)`;
    return uniform
      ? `    total += block.data[${index}];`
      : `    total += texelFetch(dataTex, ivec2(${index} % 2048, ${index} / 2048), 0);`;
  }).join("\n")}
  }
  result = total;
}`;

export async function main(device, args) {
  const reads = Number(option(args, "reads", "1024"));
  const rounds = Number(option(args, "rounds", "4"));
  const iterations = Number(option(args, "iterations", "16"));
  const gl = createWebGL2Context();
  const limits = {
    maxUniformBlockSize: gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE),
    maxFragmentUniformBlocks: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_BLOCKS),
  };
  // One block's worth, which is the whole point: the shared operand has to fit.
  const vectors = Math.min(1024, limits.maxUniformBlockSize / 16);
  const floats = new Float32Array(vectors * 4);
  for (let i = 0; i < floats.length; i += 1) floats[i] = (i % 97) / 97 - 0.5;

  // 🔴 A RING OF TARGETS, NOT ONE. An Apple GPU does hidden-surface removal:
  // the same full-viewport triangle drawn twice into one target with no read
  // between has the earlier draw's fragments DISCARDED before they are shaded.
  // The first version of this probe drew sixteen times into one texture and
  // reported 894 G reads a second on BOTH arms - identical, because it was
  // timing one draw and dividing by sixteen. bench-webgl-linear.js's header
  // records the same trap costing two wrong measurements.
  const side = 256;
  const fragments = side * side;
  const ring = Array.from({ length: iterations }, () => {
    const target = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, target);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, side, side, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D,
      target, 0);
    return framebuffer;
  });

  // The data, as a texture for one arm and as a uniform block for the other.
  const dataTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, dataTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, vectors, 1, 0, gl.RGBA, gl.FLOAT, floats);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  const ubo = gl.createBuffer();
  gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
  gl.bufferData(gl.UNIFORM_BUFFER, floats, gl.STATIC_DRAW);
  gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, ubo);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, FULLSCREEN_TRIANGLE, gl.STATIC_DRAW);

  const rows = [];
  const errors = [];
  for (const uniform of [false, true]) {
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER,
      source(reads, vectors, uniform)));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      errors.push(`${uniform ? "uniform" : "texture"}: ${gl.getProgramInfoLog(program)}`);
      continue;
    }
    gl.useProgram(program);
    const position = gl.getAttribLocation(program, "position");
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    if (uniform) {
      const index = gl.getUniformBlockIndex(program, "Block");
      gl.uniformBlockBinding(program, index, 0);
    } else {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, dataTex);
      gl.uniform1i(gl.getUniformLocation(program, "dataTex"), 0);
    }
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
    draw(0); sync();
    const samples = [];
    for (let round = 0; round < rounds; round += 1) {
      const start = performance.now();
      for (let i = 0; i < iterations; i += 1) draw(i);
      sync();
      samples.push((performance.now() - start) / iterations);
    }
    const ms = Math.min(...samples);
    rows.push({
      arm: uniform ? "uniform block" : "texelFetch",
      milliseconds: Number(ms.toFixed(3)),
      gigaReadsPerSecond: Number((fragments * reads / ms / 1e6).toFixed(2)),
      spread: Number((Math.max(...samples) / Math.min(...samples)).toFixed(2)),
    });
  }

  const texture = rows.find((r) => r.arm === "texelFetch");
  const uniform = rows.find((r) => r.arm === "uniform block");
  const ratio = texture && uniform
    ? uniform.gigaReadsPerSecond / texture.gigaReadsPerSecond : null;
  const renderer = gl.getParameter(gl.RENDERER);
  return {
    renderer,
    limits,
    vectorsInBlock: vectors,
    floatsAvailable: vectors * 4 * limits.maxFragmentUniformBlocks,
    reads,
    rows,
    errors,
    uniformOverTexture: ratio === null ? null : Number(ratio.toFixed(3)),
    verdict: ratio === null
      ? `one arm did not link: ${errors.join("; ")}`
      : ratio > 1.15
        ? `a uniform block read is ${ratio.toFixed(2)}x a texelFetch here, so the shared `
          + `operand of a projection belongs in uniforms and the 21.33 flops a fetch in `
          + `webgl/README.md is NOT the ceiling`
        : `a uniform block read is ${ratio.toFixed(2)}x a texelFetch here, so uniforms are `
          + `not a cheaper path on this device and the texture unit really is the only one`,
  };
}
