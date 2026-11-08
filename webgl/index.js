/**
 * The WebGL2 backend's public surface.
 *
 * `webgpu/index.js` is the shape this follows: one door per tree, so a caller
 * names a backend rather than a file. See `webgl/README.md` for what is here,
 * what is measured, and how much of AF2 is still missing.
 *
 * 🔴 THE PAGE SELECTS THIS BACKEND NOW, FOR THREE MODELS. `web/model.js`'s `foldBackend` asks for
 * WebGPU first and answers "webgl2" when there is none, and `web/app.js` folds the families in
 * `WEBGL2_FAMILIES` - af3, openbind0, intellifold2 - through `web/af3-model.js`'s
 * `backend: "webgl2"`. Measured on a browser with `navigator.gpu` deleted
 * (`tools/fold-in-page.py --no-webgpu`): AlphaFold 3 at pLDDT 68.0 in 11 s against WebGPU's 68.2
 * in 1 s, OpenBind-0 60.1, IntelliFold-2 54.4. The status line ends in "· WebGL2" so a reader
 * knows which one folded.
 *
 * 🔴 AND THE OTHER FIVE LINEAGE BUNDLES ARE REFUSED BY NAME, BEFORE THE DOWNLOAD. boltz2,
 * protenix2, mol1 and rosettafold3 carry the FUSED template embedder, which `af3/trunk/template.js`
 * has not ported - protenix2 folds on WebGPU at pLDDT 62.2 and died here at "templatePairEmbedding8
 * is missing" after half a gigabyte of parameters - and opendde re-tokenises after the trunk.
 * `test/webgl2-families.test.js` holds that list to the manifests in both directions.
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
