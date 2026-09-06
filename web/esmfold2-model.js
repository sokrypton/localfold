/**
 * ESMFold2 in the page: two bundles, one fold, no alignment.
 *
 * 🔴 IT IS THE FIRST MODEL HERE THAT IS TWO BUNDLES. The folding half -
 * the trunk, the inputs embedder and the whole structure head - is 122 MiB at
 * int5; the LANGUAGE MODEL it reads is a separate 224 MiB at int3, with its own
 * exporter and its own licence. The shim that joins them is trained against one
 * folding model, which is why the ESM-C manifest carries both names and why
 * `loadEsmfold2Weights` checks them rather than trusting the pairing.
 *
 * 🔴 AND IT HAS NO CONFIDENCE HEAD. `confidence_head.enabled` is false in this
 * checkpoint and there are ZERO `confidence_head.*` tensors in it - so there is
 * no pLDDT, no PAE, no pTM and no ipTM, and none of them is computed and hidden.
 * What the trunk does produce is a DISTOGRAM, and the contact map from it is
 * what the heatmap panel is given. A page that filled a pLDDT column with a
 * constant would be inventing the model's opinion of its own answer.
 *
 * 🔴 AND NO ALIGNMENT AND NO TEMPLATE. `disable_msa_features` is true, the MSA
 * encoder is disabled, and `grep -rn template` over the whole upstream package
 * returns nothing. The page's MSA row is hidden for this family rather than
 * ignored, because a search that runs and is discarded is a minute of somebody
 * else's server for no reason.
 */
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";
import { bundleBaseUrl, loadManifest } from "../src/reference/manifests/index.js";
import {
  atomDecoderWeights, atomEncoderWeights, denoiserWeights, featuriserWeights,
  trunkBlockWeights,
} from "../src/esmfold2/weights.js";
import { SHIM_PAIR_TENSORS } from "../src/esmfold2/language-pair-webgpu.js";
import { EsmcTowerGpu } from "../src/esmc/tower-webgpu.js";

/** The ten tensors an ESM-C block holds, under the names its exporter writes. */
const BLOCK_LEAVES = ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
  "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
  "ffn_norm/offset", "fc1/weights", "fc2/weights"];
/** What the tower needs besides its blocks, plus the shim's single half. */
const TOWER_SHARED = ["embed/weights", "final_norm/scale", "lm/combine", "lm/norm/scale",
  "lm/norm/offset", "lm/projection/weights", "lm/downproject/weights",
  "lm/downproject/bias"];

/**
 * The step counts the sampler dial offers, per mode.
 *
 * 🔴 SIX STEPS IS NOT "FASTER", IT IS BROKEN, so `diffusion-8` is measured and
 * not offered. See SAMPLER_PRESETS in src/esmfold2/fold.js: it gives a CA-CA
 * spacing of 58 A where a peptide bond is 3.8. The churn is what breaks -
 * gamma0 re-noises to sigma * 1.605 and step_scale 1.638 overshoots - and at
 * six steps the levels are too far apart for either to be corrected. The flow
 * arm re-noises not at all and is merely poor there, so it starts at 16 too.
 *
 * 🔴 AND THE NUMBER ON THE DIAL IS THE SCHEDULE'S LENGTH, NOT THE STEP COUNT.
 * `max_inference_sigma` drops every entry above 256 and prepends the cap, so
 * the checkpoint's 15 runs eleven. The dial says what the model's own config
 * says; the status line reports what actually ran.
 */
export const ESMFOLD2_COUNTS = {
  flow: { label: "Steps", values: [16, 32], preferred: 16 },
  diffusion: { label: "Steps", values: [15, 32, 64, 200], preferred: 15 },
};

let weightsPromise;

/**
 * 🔴 MEMOISED, AND KEYED ON NOTHING, BECAUSE THERE IS ONE OF THESE. AF3's
 * loader is a Map keyed by family because two bundles build that graph and the
 * first version handed the second family the first one's weights. There is one
 * ESMFold2 bundle; if a second appears, this becomes a Map on the same day.
 */
export function loadEsmfold2Weights(onProgress) {
  if (weightsPromise !== undefined) return weightsPromise;
  weightsPromise = (async () => {
    const [foldManifest, towerManifest] = await Promise.all([
      loadManifest("esmfold2"), loadManifest("esmc"),
    ]);
    // 🔴 THE SHIM IS PER FOLDING MODEL AND THE ARTEFACTS SAY SO. A tower is
    // interchangeable between releases and a shim is not, so a pairing that
    // conforms in every shape can still be two halves of different models.
    if (towerManifest.languageModel?.shim !== foldManifest.trunk?.source) {
      throw new Error(`the language model's shim is for `
        + `${towerManifest.languageModel?.shim} and the folding bundle is `
        + `${foldManifest.trunk?.source}`);
    }
    // 🔴 ONE PROGRESS STREAM OVER TWO STORES, or the dial resets to zero in the
    // middle of a 347 MiB download. The two are reported as one total.
    const seen = new Map();
    const report = (key) => (progress) => {
      seen.set(key, progress);
      let loadedBytes = 0, totalBytes = 0;
      for (const value of seen.values()) {
        loadedBytes += value?.loadedBytes ?? 0;
        totalBytes += value?.totalBytes ?? 0;
      }
      onProgress?.({ ...(seen.get(key) ?? {}), loadedBytes, totalBytes });
    };
    const [foldStore, towerStore] = await Promise.all([
      HttpTensorStore.fromManifest(bundleBaseUrl("esmfold2"), foldManifest,
                                   report("fold")),
      HttpTensorStore.fromManifest(bundleBaseUrl("esmc"), towerManifest,
                                   report("tower")),
    ]);
    foldStore.prefetch();
    towerStore.prefetch();

    const read = (name) => foldStore.tensor(name);
    const M = foldManifest.trunk;
    const [featuriser, inputsEmbedder, denoiser, encoder, decoder] = await Promise.all([
      featuriserWeights(read),
      atomEncoderWeights(read, "atom", M.atomBlocks),
      denoiserWeights(read, { tokenBlocks: M.tokenBlocks }),
      atomEncoderWeights(read, "diffusionAtomEncoder", M.atomBlocks,
                         { withCoordinates: true }),
      atomDecoderWeights(read, "diffusionAtomDecoder", M.atomBlocks),
    ]);
    denoiser.encoder = encoder;
    denoiser.decoder = decoder;
    const trunkBlocks = [];
    for (let layer = 0; layer < M.blocks; layer += 1) {
      trunkBlocks.push(await trunkBlockWeights(read, layer));
    }

    // 🔴 THE SHIM'S SINGLE HALF IS LOADED EVEN THOUGH THE TOWER COMPUTES IT.
    // A non-protein token never reaches the tower and its hidden state is ZERO
    // rather than absent, so the fold evaluates the shim at zero itself. Three
    // tensors, one row of arithmetic, once a fold.
    const shim = {};
    for (const name of [...SHIM_PAIR_TENSORS, "lm/norm/offset",
                        "lm/projection/weights", "lm/downproject/weights",
                        "lm/downproject/bias"]) {
      shim[name] = await towerStore.tensor(name);
    }
    const towerShared = {};
    for (const name of TOWER_SHARED) towerShared[name] = await towerStore.tensor(name);

    return {
      shape: { ...M, loops: (M.loops ?? 3) + 1 },
      weights: { featuriser, inputsEmbedder, trunkBlocks, denoiser, shim },
      language: {
        manifest: towerManifest.languageModel,
        shared: towerShared,
        ffn: towerManifest.tensors["blocks/0/fc2/weights"].shape[0],
        // 🔴 A BLOCK IS READ WHEN THE TOWER ASKS FOR IT. Reading all 36 first
        // is 2190 MiB of float32 in the tab, which is the thing the streaming
        // design exists to avoid.
        block: async (layer) => {
          const block = {};
          for (const leaf of BLOCK_LEAVES) {
            block[leaf] = await towerStore.tensor(`blocks/${layer}/${leaf}`);
          }
          return block;
        },
      },
    };
  })();
  return weightsPromise;
}

/**
 * The language model half, as the callback `foldEsmfold2` expects.
 *
 * 🔴 ONE PACKED RUN WITH A CHAIN MASK, NOT ONE RUN PER CHAIN. ESM-C's rotary
 * positions are absolute over the packed array with no per-chain reset, so
 * running the tower once per chain gives every chain after the first a
 * different phase - the same shapes, a plausible tensor, a different model.
 */
export function languageModelRunner(device, allocator, loaded, pairChannels) {
  return async (ids, sequenceId) => {
    const tower = new EsmcTowerGpu(device, allocator);
    const result = await tower.run(ids, {
      rows: ids.length,
      model: loaded.language.manifest.width,
      heads: loaded.language.manifest.heads ?? loaded.language.manifest.width / 64,
      ffn: loaded.language.ffn,
      layers: loaded.language.manifest.layers,
      pair: pairChannels,
      residualScale: loaded.language.manifest.residualScale ?? 1,
    }, loaded.language.block, loaded.language.shared, { sequenceId });
    return result.single;
  };
}
