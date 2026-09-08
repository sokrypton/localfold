/** The WebGL2 limits that bound a GPGPU kernel, including the ones a fragment
 * shader does not have but transform feedback might. */
export async function main() {
  const gl = new OffscreenCanvas(4, 4).getContext("webgl2");
  if (gl === null) return { error: "no webgl2" };
  const p = (name) => gl.getParameter(gl[name]);
  return {
    drawBuffers: p("MAX_DRAW_BUFFERS"),
    colorAttachments: p("MAX_COLOR_ATTACHMENTS"),
    fragmentOutputFloats: p("MAX_DRAW_BUFFERS") * 4,
    // 🔴 THE VERTEX PATH HAS ITS OWN OUTPUT CAP AND IT IS NOT THE SAME ONE.
    // Transform feedback writes from the VERTEX shader with rasterizer discard,
    // and its interleaved component limit is what bounds outputs there.
    transformFeedbackInterleavedComponents: p("MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS"),
    transformFeedbackSeparateAttribs: p("MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS"),
    transformFeedbackSeparateComponents: p("MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS"),
    vertexOutputComponents: p("MAX_VERTEX_OUTPUT_COMPONENTS"),
    vertexUniformVectors: p("MAX_VERTEX_UNIFORM_VECTORS"),
    vertexTextureImageUnits: p("MAX_VERTEX_TEXTURE_IMAGE_UNITS"),
    fragmentUniformVectors: p("MAX_FRAGMENT_UNIFORM_VECTORS"),
    uniformBlockSize: p("MAX_UNIFORM_BLOCK_SIZE"),
    uniformBufferBindings: p("MAX_UNIFORM_BUFFER_BINDINGS"),
    textureSize: p("MAX_TEXTURE_SIZE"),
    array3DTextureSize: p("MAX_3D_TEXTURE_SIZE"),
    arrayTextureLayers: p("MAX_ARRAY_TEXTURE_LAYERS"),
  };
}
