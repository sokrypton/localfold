/**
 * What a WebGL2 context has to provide for this backend, and the ceilings that
 * follow from what it reports.
 *
 * 🔴 A CEILING HERE IS A PROPERTY OF A LAYOUT, NOT OF THE API - AND THE ONE
 * THIS REPOSITORY QUOTED IS THE WRONG ONE FOR ITS OWN KERNELS. docs/A100.md
 * concludes that the pair track "stops fitting a texture axis at 181 tokens"
 * and that AF3's pair-transition "cannot be expressed". That is measured on a
 * GEMM whose operand is a 2D texture of `(K/4) x M` with `M = tokens^2`, so the
 * pair COUNT has to fit one texture axis: a real ceiling for that layout, and
 * `pairTokenCeiling` below is it.
 *
 * It is not the ceiling for a tensor addressed LINEARLY, which is what
 * `webgl/runtime/storage.js` does and what the triangle kernel has always done.
 * There the bound is the total element count - `2048 * MAX_TEXTURE_SIZE * 4` -
 * which on the M5 measured here is 134,217,728 floats, or **1024 tokens** at
 * AF2's 128-channel pair width. Checked rather than argued: a 400-token pair
 * tensor of 20,480,000 floats round-trips with 0 of them differing
 * (`tools/gpu/check-webgl-runtime.js`). Both numbers are right; only one of
 * them bounds this tree.
 *
 * 🔴 AND THE CARDS THAT NEED THIS BACKEND REPORT SMALLER NUMBERS THAN THE ONE
 * THAT MEASURED IT, whichever layout is used. Every figure in docs/A100.md comes
 * off an A100 reporting `MAX_TEXTURE_SIZE` 32,768; this M5 reports 16,384, which
 * halves both ceilings. The machines a WebGL2 fallback EXISTS for are older
 * still, and the specification floor is 2,048.
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
 * The longest chain whose PAIR track fits one texture AXIS - the ceiling for a
 * kernel that wants a 2D operand, and NOT the ceiling for a linearly addressed
 * tensor, which is `tensorCapacity` in ./storage.js and is eight times longer
 * here. Read that module before quoting this number as a limit on a fold.
 *
 * The pair tensor has `tokens * tokens` rows, so a 2D operand stops fitting at
 * `sqrt(MAX_TEXTURE_SIZE)`. Measured against docs/A100.md's own table:
 *
 *     MAX_TEXTURE_SIZE 32768 -> 181 tokens   (32,761 rows fit, 200 -> 40,000 does not)
 *                      16384 -> 128
 *                       8192 ->  90
 *                       2048 ->  45          (the WebGL2 specification floor)
 *
 * Past it a 2D-operand kernel needs its row index tiled across two texture
 * dimensions, with the address arithmetic that implies - which is precisely
 * what the linear layout already does, and why the triangle kernel has never
 * had this ceiling.
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
