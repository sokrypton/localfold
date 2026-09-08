/**
 * What WebGL2 exposes on this device, and whether any of it is compute.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl2.js
 *
 * WHY THE QUESTION COMES UP. WebGPU on this platform withholds two things this
 * repository wants - `shader-f16` (a Dawn policy block on NVIDIA) and
 * subgroup matrices (no VK_KHR_cooperative_matrix on this driver) - so it is
 * fair to ask whether the older API reaches something the newer one does not.
 */
export async function main() {
  const canvas = new OffscreenCanvas(64, 64);
  const gl = canvas.getContext("webgl2");
  if (gl === null) return { webgl2: false };
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const has = (name) => gl.getExtension(name) !== null;
  return {
    webgl2: true,
    renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    version: gl.getParameter(gl.VERSION),
    shadingLanguage: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
    // 🔴 THERE IS NO COMPUTE SHADER IN WebGL2. The constant does not exist on
    // the context; WebGL 2.0 Compute was prototyped in Chrome and removed. All
    // GPGPU here is fragment shaders writing to float textures, or transform
    // feedback - which is why the limits below are the ones that matter.
    computeShaderConstantExists: "COMPUTE_SHADER" in gl,
    storageBuffers: "SHADER_STORAGE_BUFFER" in gl,
    limits: {
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS),
      maxFragmentUniformVectors: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS),
      maxTextureImageUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
      maxVaryingVectors: gl.getParameter(gl.MAX_VARYING_VECTORS),
      // The nearest thing to workgroup memory a fragment shader has, which is
      // nothing: there is no shared array, only per-invocation registers.
      maxUniformBlockSize: gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE),
    },
    floatSupport: {
      colorBufferFloat: has("EXT_color_buffer_float"),
      colorBufferHalfFloat: has("EXT_color_buffer_half_float"),
      textureFloatLinear: has("OES_texture_float_linear"),
      // 🔴 AND THE ONE THAT WOULD MATTER: half-precision ARITHMETIC. GLSL ES
      // 3.00's `mediump` is a hint, and on desktop NVIDIA it is fp32.
      shaderF16Analogue: null,
    },
    extensions: gl.getSupportedExtensions().sort(),
  };
}
