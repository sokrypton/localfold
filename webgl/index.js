/**
 * The WebGL2 backend's public surface.
 *
 * `webgpu/index.js` is the shape this follows: one door per tree, so a caller
 * names a backend rather than a file. The list is short because this backend is
 * two kernels and a runtime - see `webgl/README.md` for what it is, what it is
 * measured at, and the two walls (no workgroup memory, and a pair track that
 * stops fitting a texture axis) that decide whether it ever becomes more.
 *
 * 🔴 NOTHING SELECTS THIS BACKEND YET. The page requires WebGPU and says so
 * (`web/design.js`, "WebGPU unavailable"); `tools/build_site.py` does not copy
 * this tree, so it is not on the site. Both are one line each when a fold can
 * actually run here, and neither should change before one can.
 */
export { createWebGL2Context, textureShape } from "./runtime/context.js";
export {
  OPTIONAL_EXTENSIONS, REQUIRED_EXTENSIONS, fragmentOutputFloats, pairTokenCeiling,
  vertexOutputFloats, webgl2Capability,
} from "./runtime/limits.js";

export { TriangleContractWebGL2 } from "./kernels/triangle.js";
export { AttentionFlashWebGL2, HEAD_DIM } from "./kernels/attention.js";
