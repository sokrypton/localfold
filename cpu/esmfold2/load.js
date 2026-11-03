/**
 * ESMFold2's two bundles, read into a CpuBackend's memory.
 *
 * The same stores and the same shard cache as the page's GPU path, so a model either
 * path has downloaded is already here for the other. Each tensor is decoded once,
 * moved into the backend and let go, because the backend is what reads it from then on.
 */
import { HttpTensorStore } from "../../shared/bundles/http-tensor-store.js";
import { MODEL_BUNDLES, bundleBaseUrl, loadManifest } from "../../shared/bundles/manifests/index.js";
import {
  atomDecoderWeights, atomEncoderWeights, confidenceHeadWeights, denoiserWeights,
  featuriserWeights, trunkBlockWeights,
} from "../../shared/esmfold2/weights.js";

/** The tensors one ESM-C block holds, under the names its exporter writes. */
const BLOCK_LEAVES = ["attn_norm/scale", "attn_norm/offset", "qkv/weights", "q_norm/scale",
  "k_norm/scale", "attn_out/weights", "ffn_norm/scale", "ffn_norm/offset", "fc1/weights", "fc2/weights"];
const TOWER_SHARED = ["embed/weights", "final_norm/scale", "lm/combine", "lm/norm/scale",
  "lm/norm/offset", "lm/projection/weights", "lm/downproject/weights", "lm/downproject/bias"];
const SHIM = ["lm/pair_mlp_1/weights", "lm/pair_mlp_1/bias", "lm/pair_mlp_2/weights",
  "lm/pair_mlp_2/bias", "lm/pair_norm/scale", "lm/pair_norm/offset", "lm/norm/offset",
  "lm/projection/weights", "lm/downproject/weights", "lm/downproject/bias"];

/**
 * @param {string} family `ef2-fast-300m` or `ef2-fast-600m`
 * @param {import("../runtime/backend.js").CpuBackend} backend
 * @param {(progress: {loadedBytes: number, totalBytes: number}) => void} [onProgress]
 */
export async function loadEsmfold2CpuWeights(family, backend, onProgress) {
  const towerFamily = MODEL_BUNDLES[family]?.companion;
  if (towerFamily === undefined) throw new Error(`${family} names no language model bundle`);
  const [foldManifest, towerManifest] = await Promise.all([loadManifest(family), loadManifest(towerFamily)]);
  if (towerManifest.languageModel?.shim !== foldManifest.trunk?.source) {
    throw new Error(`the language model's shim is for ${towerManifest.languageModel?.shim}`
      + ` and the folding bundle is ${foldManifest.trunk?.source}`);
  }
  const seen = new Map();
  const report = (key) => (progress) => {
    seen.set(key, progress);
    let loadedBytes = 0;
    let totalBytes = 0;
    for (const value of seen.values()) {
      loadedBytes += value?.loadedBytes ?? 0;
      totalBytes += value?.totalBytes ?? 0;
    }
    onProgress?.({ loadedBytes, totalBytes });
  };
  const [foldStore, towerStore] = await Promise.all([
    HttpTensorStore.fromManifest(bundleBaseUrl(family), foldManifest, report("fold")),
    HttpTensorStore.fromManifest(bundleBaseUrl(towerFamily), towerManifest, report("tower")),
  ]);
  foldStore.prefetch();
  towerStore.prefetch();

  const read = (name) => foldStore.tensor(name);
  const M = foldManifest.trunk;
  const [featuriser, inputsEmbedder, denoiser, encoder, decoder, confidence] = await Promise.all([
    featuriserWeights(read),
    atomEncoderWeights(read, "atom", M.atomBlocks),
    denoiserWeights(read, { tokenBlocks: M.tokenBlocks }),
    atomEncoderWeights(read, "diffusionAtomEncoder", M.atomBlocks, { withCoordinates: true }),
    atomDecoderWeights(read, "diffusionAtomDecoder", M.atomBlocks),
    confidenceHeadWeights(read, foldManifest),
  ]);
  if (confidence === null) throw new Error(`${family}'s bundle carries no confidence head`);
  denoiser.encoder = encoder;
  denoiser.decoder = decoder;
  const trunkBlocks = [];
  for (let layer = 0; layer < M.blocks; layer += 1) {
    trunkBlocks.push(await trunkBlockWeights(read, layer));
  }
  const weights = { featuriser, inputsEmbedder, denoiser, trunkBlocks };
  backend.adoptTree(weights);
  backend.adoptTree(confidence);
  foldStore.releaseDecoded();

  const shim = {};
  for (const name of SHIM) shim[name] = backend.adopt(await towerStore.tensor(name));
  const shared = {};
  for (const name of TOWER_SHARED) shared[name] = backend.adopt(await towerStore.tensor(name));
  towerStore.releaseDecoded();
  weights.shim = shim;

  const blocks = new Map();
  return {
    shape: { ...M, loops: (M.loops ?? 3) + 1 },
    weights,
    confidenceWeights: confidence,
    language: {
      manifest: towerManifest.languageModel,
      ffn: towerManifest.tensors["blocks/0/fc2/weights"].shape[0],
      shared,
      // ...one block at a time, the first fold reaching each, so a fold with no
      // protein never spends the memory.
      block: async (layer) => {
        let block = blocks.get(layer);
        if (block === undefined) {
          block = {};
          for (const leaf of BLOCK_LEAVES) {
            const name = `blocks/${layer}/${leaf}`;
            block[name] = backend.adopt(await towerStore.tensor(name));
          }
          towerStore.releaseDecoded();
          blocks.set(layer, block);
        }
        return block;
      },
    },
  };
}
