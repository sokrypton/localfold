/**
 * The WebGL2 limits that decide what shape a kernel in this backend can be.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-limits.js
 *
 * 🔴 THE ONE THAT BINDS EVERYTHING HERE IS THE OUTPUT BUDGET. An invocation
 * writes `MAX_DRAW_BUFFERS * 4` floats, which caps a register tile's arithmetic
 * intensity at `8 * R * C / (R + C)` with `R * C` fixed - 21.33 flops a fetch4
 * at 32 outputs - and webgl/README.md's "How close can this get" multiplies that
 * by this device's measured fetch issue rate to get about 600 GFLOP/s where
 * parity with the WGSL fold needs 1303. So the question this probe exists to ask
 * is whether anything in WebGL2 writes MORE than 32 floats an invocation.
 *
 * 🔴 TRANSFORM FEEDBACK IS THE CANDIDATE: a VERTEX shader writing to buffers,
 * where the limit is `MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS` - 64 in the
 * GLES 3.0 minimum, twice the fragment budget. At 64 outputs the smallest
 * `R + C` is 16 at 8 by 8, so the intensity cap becomes `8 * 64 / 16` = 32
 * flops a fetch4 against 21.33, which is **1.5x the ceiling** rather than 1.5x
 * of one kernel. Whether a vertex invocation ISSUES fetches as fast as a
 * fragment one is the other half and is what `probe-webgl-feedback-gemm.js`
 * measures; this one says whether the shape is legal at all.
 */
import { createWebGL2Context } from "../../webgl/index.js";

const NAMES = [
  "MAX_DRAW_BUFFERS",
  "MAX_COLOR_ATTACHMENTS",
  "MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS",
  "MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS",
  "MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS",
  "MAX_VERTEX_TEXTURE_IMAGE_UNITS",
  "MAX_VERTEX_OUTPUT_COMPONENTS",
  "MAX_VERTEX_UNIFORM_COMPONENTS",
  "MAX_TEXTURE_IMAGE_UNITS",
  "MAX_TEXTURE_SIZE",
  "MAX_FRAGMENT_UNIFORM_COMPONENTS",
  "MAX_UNIFORM_BLOCK_SIZE",
  "MAX_ARRAY_TEXTURE_LAYERS",
  "MAX_SAMPLES",
];

export async function main(device, args) {
  const gl = createWebGL2Context();
  /** @type {Record<string, number>} */
  const limits = {};
  for (const name of NAMES) {
    const value = gl.getParameter(gl[name]);
    limits[name] = typeof value === "number" ? value : Number(value);
  }
  const fragmentFloats = limits.MAX_DRAW_BUFFERS * 4;
  const feedbackFloats = limits.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS;
  // The best intensity a register tile of R by C reaches, over the divisors of
  // the budget: `8 * R * C / (R + C)`, maximised where R and C are closest.
  const bestIntensity = (outputs) => {
    let best = 0;
    for (let r = 1; r <= outputs; r += 1) {
      if (outputs % r !== 0) continue;
      const c = outputs / r;
      // C is a multiple of four for a fragment target; for feedback a component
      // is a float, so any C is addressable.
      best = Math.max(best, (8 * r * c) / (r + c));
    }
    return best;
  };
  const fragmentCap = bestIntensity(fragmentFloats);
  const feedbackCap = bestIntensity(feedbackFloats);
  // webgl/README.md's measured issue rate at a realistic 64 KiB working set.
  const issueRate = 25.7;
  const parity = 1303;
  return {
    renderer: gl.getParameter(gl.RENDERER),
    limits,
    fragmentOutputFloats: fragmentFloats,
    feedbackOutputFloats: feedbackFloats,
    fragmentIntensityCap: Number(fragmentCap.toFixed(2)),
    feedbackIntensityCap: Number(feedbackCap.toFixed(2)),
    fragmentCeilingGflops: Number((fragmentCap * issueRate).toFixed(0)),
    feedbackCeilingGflops: Number((feedbackCap * issueRate).toFixed(0)),
    parityNeedsGflops: parity,
    vertexCanSampleTextures: limits.MAX_VERTEX_TEXTURE_IMAGE_UNITS >= 2,
    verdict: feedbackFloats > fragmentFloats
      ? `transform feedback writes ${feedbackFloats} floats an invocation against the `
        + `fragment path's ${fragmentFloats}, so its intensity cap is `
        + `${feedbackCap.toFixed(2)} against ${fragmentCap.toFixed(2)} - a ceiling of `
        + `${(feedbackCap * issueRate).toFixed(0)} GFLOP/s against `
        + `${(fragmentCap * issueRate).toFixed(0)}, where parity needs ${parity}`
      : `transform feedback writes ${feedbackFloats} floats an invocation and the fragment `
        + `path ${fragmentFloats}, so it raises no ceiling here`,
  };
}
