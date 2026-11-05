/**
 * The WebGL2 backend's public surface.
 *
 * `webgpu/index.js` is the shape this follows: one door per tree, so a caller
 * names a backend rather than a file. See `webgl/README.md` for what is here,
 * what is measured, and how much of AF2 is still missing.
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
export { Storage, storageBytes, tensorCapacity } from "./runtime/storage.js";
export { Execution, PRELUDE } from "./runtime/execution.js";
export { add, maskRows, multiply, scale } from "./runtime/elementwise.js";
export { transposeLeading } from "./runtime/transpose.js";

export { TriangleContractWebGL2 } from "./kernels/triangle.js";
export { AttentionFlashWebGL2, HEAD_DIM } from "./kernels/attention.js";
export { layerNorm, layerNormSource } from "./kernels/layer-norm.js";
export { linear, linearSource } from "./kernels/linear.js";
export { triangleContract } from "./kernels/triangle-contract.js";
export { triangleMultiplication, uploadTriangleWeights } from "./kernels/triangle-multiplication.js";
export { attentionBlock, uploadAttentionBlockWeights } from "./kernels/attention-block.js";
export { attentionFlash, attentionFlashPlan } from "./kernels/attention-flash.js";
export { pairBias, uploadPairBiasWeights } from "./kernels/pair-bias.js";
export { transition, uploadTransitionWeights } from "./kernels/transition.js";
export { outerProductMean, uploadOuterProductMeanWeights } from "./kernels/outer-product-mean.js";
