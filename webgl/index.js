/**
 * The WebGL2 backend's public surface.
 *
 * `webgpu/index.js` is the shape this follows: one door per tree, so a caller
 * names a backend rather than a file. See `webgl/README.md` for what is here,
 * what is measured, and how much of AF2 is still missing.
 *
 * 🔴 THE PAGE SELECTS THIS BACKEND NOW, FOR SEVEN MODELS - every AF3-lineage family but OpenDDE, which re-tokenises
 * after the trunk and is refused by name before the download (`WEBGL2_FAMILIES`, held to the lineage by
 * `test/webgl2-families.test.js`). `web/model.js`'s `foldBackend` asks for WebGPU first and answers "webgl2" when there
 * is none, and `web/app.js` folds those families through `web/af3-model.js`'s `backend: "webgl2"`. The status line
 * ends in "· WebGL2" so a reader knows which one folded.
 *
 * `tools/build_site.py` copies this tree now (it is in DIRECTORIES): `web/af3-model.js` imports it
 * STATICALLY, so a dist/ without it is a page whose AF3 path does not load at all.
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
