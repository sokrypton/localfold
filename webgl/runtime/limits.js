/**
 * What a WebGL2 context has to provide for this backend, and the ceilings that
 * follow from what it reports.
 *
 * 🔴 THE LIMITS ARE THE WHOLE FEASIBILITY QUESTION HERE, WHICH IS NOT TRUE OF
 * WebGPU. A WebGPU buffer is one-dimensional and 2 GB on the cards measured, so
 * a tensor's SHAPE never decides whether a kernel can be written. A WebGL2
 * tensor lives in a TEXTURE with a maximum axis, so `MAX_TEXTURE_SIZE` decides
 * what is expressible at all - and AF3's pair-transition, 40000x128x512, is not
 * expressible on a card reporting 32,768 (docs/A100.md, "WebGL2 reaches 48% of
 * this device"). A kernel that is 1.6x slower is a trade; a kernel that cannot
 * be written is a wall, and this module is where the wall is computed rather
 * than discovered in a link error.
 *
 * 🔴 AND THE CARDS THAT NEED THIS BACKEND REPORT SMALLER NUMBERS THAN THE ONE
 * THAT MEASURED IT. Every figure in docs/A100.md comes off an A100 reporting
 * `MAX_TEXTURE_SIZE` 32,768. The machines a WebGL2 fallback EXISTS for are the
 * ones without WebGPU, which are older, and the WebGL2 specification floor is
 * 2,048 - a pair-track ceiling of 45 tokens rather than 181. So the ceiling is
 * a function of the context, never a constant, and `pairTokenCeiling` is it.
 *
 * `tools/gpu/probe-webgl2-limits.js` REPORTS what a context says. This module
 * DECIDES what that means for a fold. Two jobs, deliberately apart: a probe
 * that refuses is not a probe, and a backend that only prints is not a backend.
 */

/**
 * Extensions this backend cannot run without, with what each one buys.
 *
 * Both are extensions rather than core WebGL2, which is the thing most easily
 * assumed away: a context can be a perfectly good WebGL2 context, render a
 * scene, and still be unable to hold a float tensor.
 */
export const REQUIRED_EXTENSIONS = Object.freeze({
  EXT_color_buffer_float:
    "renders to RGBA32F, which is what makes a float tensor a legal draw target",
});

/**
 * Extensions worth having and not required.
 *
 * `EXT_color_buffer_half_float` is the fallback target precision - half the
 * bandwidth, and present on hardware too old for the f32 one. Nothing here
 * takes it yet; it is named so that a context survey records it.
 */
export const OPTIONAL_EXTENSIONS = Object.freeze({
  EXT_color_buffer_half_float: "an f16 render target, half the bandwidth of RGBA32F",
  OES_texture_float_linear: "filtered float sampling; this backend uses NEAREST and does not need it",
});

/**
 * The longest chain whose PAIR track fits one texture axis.
 *
 * The pair tensor has `tokens * tokens` rows, so the operand stops fitting at
 * `sqrt(MAX_TEXTURE_SIZE)`. Measured against docs/A100.md's own table:
 *
 *     MAX_TEXTURE_SIZE 32768 -> 181 tokens   (32,761 rows fit, 200 -> 40,000 does not)
 *                      16384 -> 128
 *                       8192 ->  90
 *                       2048 ->  45          (the WebGL2 specification floor)
 *
 * Past it every pair-track kernel would need its row index tiled across two
 * texture dimensions, with the address arithmetic that implies. That is work,
 * not an impossibility - but it is work no kernel here has done, so the ceiling
 * is real for anything in this tree today.
 */
export function pairTokenCeiling(maxTextureSize) {
  return Math.floor(Math.sqrt(maxTextureSize));
}

/**
 * Floats one fragment-shader invocation can write.
 *
 * `MAX_DRAW_BUFFERS` attachments of one RGBA texel each. This is the budget
 * that caps arithmetic intensity on the fragment path at 21.3 flops a fetch,
 * which was once written down as "the ceiling of the API" and is the ceiling of
 * one PATH - see `vertexOutputFloats`.
 */
export function fragmentOutputFloats(gl) {
  return gl.getParameter(gl.MAX_DRAW_BUFFERS) * 4;
}

/**
 * Floats one VERTEX invocation can write through transform feedback, which is
 * four times the fragment budget and is worth 2.2x.
 *
 * 🔴 THE ADVERTISED LIMIT IS NOT THE REAL ONE, AND TAKING IT AT ITS WORD FAILS
 * AT LINK TIME. `MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS` reads 128 on
 * the measured card, but the varyings must ALSO pack into
 * `MAX_VERTEX_OUTPUT_COMPONENTS`, which reads 124 - so 32 vec4 outputs fail
 * with "Could not pack varying v30" and 31 is the ceiling. The minimum of the
 * two is the only honest answer, and it is why this is a function and not a
 * `getParameter` at the call site.
 */
export function vertexOutputFloats(gl) {
  return Math.min(
    gl.getParameter(gl.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS),
    gl.getParameter(gl.MAX_VERTEX_OUTPUT_COMPONENTS),
  );
}

/**
 * Everything this backend needs to know about a context, in one object.
 *
 * `usable` is false with `missing` naming the reason rather than throwing,
 * because the caller deciding between backends wants an answer and not an
 * exception. `createWebGL2Context` in ./context.js is the throwing door, for
 * the caller that has already committed.
 *
 * @param {WebGL2RenderingContext} gl
 */
export function webgl2Capability(gl) {
  const missing = Object.keys(REQUIRED_EXTENSIONS).filter((name) => gl.getExtension(name) === null);
  const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  return {
    usable: missing.length === 0,
    missing,
    optional: Object.fromEntries(Object.keys(OPTIONAL_EXTENSIONS)
      .map((name) => [name, gl.getExtension(name) !== null])),
    maxTextureSize,
    pairTokenCeiling: pairTokenCeiling(maxTextureSize),
    fragmentOutputFloats: fragmentOutputFloats(gl),
    vertexOutputFloats: vertexOutputFloats(gl),
    renderer: (() => {
      const debug = gl.getExtension("WEBGL_debug_renderer_info");
      return debug === null ? gl.getParameter(gl.RENDERER)
        : gl.getParameter(debug.UNMASKED_RENDERER_WEBGL);
    })(),
  };
}
