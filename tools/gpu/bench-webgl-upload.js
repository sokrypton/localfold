/**
 * What it costs to get weight bytes into a WebGL2 texture, and to read any back.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl-upload.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl-upload.js --megabytes=512
 *
 * 🔴 THIS ANSWERS A HYPOTHESIS AND REFUTES IT, WHICH IS WHY IT IS KEPT. A
 * bioemu2 fold in WebGL2 at 68 tokens spends **20.4 s inside six readbacks of
 * two megabytes** - a readback cannot return until everything queued before it
 * has run, so that is work the GPU had not finished - and the only things that
 * fold queues are draws and **1.79 GB of `texSubImage2D`**, the f32 weights.
 * Since the draw-doubling instrument said the draws were 2% (it was wrong; see
 * `Execution.doubleLabels`), the weights were the suspect.
 *
 * They are not: this context takes them at **10 GB/s**, so all 1.79 GB is about
 * **0.2 s**. A readback the other way is **0.8 GB/s**, twelve times slower, but
 * a fold reads back two megabytes. Neither direction is where the seconds are -
 * `--skip-draws` on the fold driver is what finally located them, in the draws.
 *
 * `uploadGbPerSecond` is the number that matters. The arms say where it goes:
 * whether a fresh `texImage2D` with its data beats an empty one plus
 * `texSubImage2D`, whether the row width changes it, and what a readback of the
 * same bytes costs.
 */
import { createWebGL2Context } from "../../webgl/index.js";
import { TEXTURE_WIDTH } from "../../webgl/runtime/context.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * A one-texel read of a texture, which cannot return until everything queued
 * before it has run - the only barrier this API offers.
 */
const flush = (gl, framebuffer, texture, pixel) => {
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, pixel);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
};

export async function main(device, args) {
  const megabytes = Number(option(args, "megabytes", "256"));
  const gl = createWebGL2Context();
  const framebuffer = gl.createFramebuffer();
  const pixel = new Float32Array(4);
  const rows = [];

  // One texture of `width x height` RGBA32F per arm, filled `repeats` times to
  // reach `megabytes`. A weight tensor in this backend is exactly this shape.
  for (const width of [TEXTURE_WIDTH, 512, 128]) {
    const height = 256;
    const texels = width * height;
    const floats = texels * 4;
    const bytes = floats * 4;
    const repeats = Math.max(1, Math.round((megabytes * (1 << 20)) / bytes));
    const data = new Float32Array(floats);
    for (let i = 0; i < floats; i += 1) data[i] = i * 1e-6;
    for (const arm of ["texSubImage2D into an allocated texture", "texImage2D with the data",
      "readPixels back"]) {
      // Warm, and get the allocation out of the measurement.
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.FLOAT, data);
      flush(gl, framebuffer, texture, pixel);

      const readInto = new Float32Array(floats);
      const started = performance.now();
      for (let r = 0; r < repeats; r += 1) {
        if (arm.startsWith("texSubImage2D")) {
          gl.bindTexture(gl.TEXTURE_2D, texture);
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.FLOAT, data);
        } else if (arm.startsWith("texImage2D")) {
          gl.bindTexture(gl.TEXTURE_2D, texture);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
        } else {
          gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
          gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, readInto);
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        }
      }
      // 🔴 THE FLUSH IS THE MEASUREMENT. `texSubImage2D` returns long before the
      // bytes are on the GPU; timing it without a barrier measures the host's
      // copy into the command stream and reads like 40 GB/s.
      flush(gl, framebuffer, texture, pixel);
      const ms = performance.now() - started;
      gl.deleteTexture(texture);
      rows.push({
        arm, width, repeats,
        megabytes: Number(((bytes * repeats) / (1 << 20)).toFixed(1)),
        ms: Number(ms.toFixed(1)),
        gbPerSecond: Number(((bytes * repeats) / (ms / 1000) / 1e9).toFixed(2)),
      });
    }
  }

  const uploads = rows.filter((row) => row.arm.startsWith("texSubImage2D"));
  const best = Math.max(...uploads.map((row) => row.gbPerSecond));
  return {
    renderer: gl.getParameter(gl.RENDERER),
    megabytesAsked: megabytes,
    rows,
    uploadGbPerSecond: best,
    // bioemu2's weights, which is what the number is for.
    bioemu2WeightSeconds: Number((1.79 / best).toFixed(1)),
    verdict: `ok: this context takes weights at ${best} GB/s, so bioemu2's 1.79 GB of `
      + `f32 weights is about ${(1.79 / best).toFixed(1)} s of upload - not where a fold's `
      + "seconds are",
  };
}
