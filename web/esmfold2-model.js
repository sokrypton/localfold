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
import { MODEL_BUNDLES, bundleBaseUrl, loadManifest } from "../src/reference/manifests/index.js";
import {
  atomDecoderWeights, atomEncoderWeights, denoiserWeights, featuriserWeights,
  trunkBlockWeights,
} from "../src/esmfold2/weights.js";
import { SHIM_PAIR_TENSORS } from "../src/esmfold2/language-pair-webgpu.js";
import { EsmcTowerGpu } from "../src/esmc/tower-webgpu.js";
import { SAMPLER_DEFAULTS, SAMPLER_PRESETS } from "../src/esmfold2/fold.js";
import { churnFactors, noiseLevels, noiseSchedule }
  from "../src/esmfold2/sampler-reference.js";

/** The ten tensors an ESM-C block holds, under the names its exporter writes. */
const BLOCK_LEAVES = ["attn_norm/scale", "attn_norm/offset", "qkv/weights",
  "q_norm/scale", "k_norm/scale", "attn_out/weights", "ffn_norm/scale",
  "ffn_norm/offset", "fc1/weights", "fc2/weights"];
/** What the tower needs besides its blocks, plus the shim's single half. */
/** The leaves worth decoding straight to half precision: the block's matrices. */
const NARROW_LEAVES = new Set(["qkv/weights", "attn_out/weights",
  "fc1/weights", "fc2/weights"]);
const TOWER_SHARED = ["embed/weights", "final_norm/scale", "lm/combine", "lm/norm/scale",
  "lm/norm/offset", "lm/projection/weights", "lm/downproject/weights",
  "lm/downproject/bias"];

/**
 * The step counts the sampler dial offers.
 *
 * 🔴 DIFFUSION ONLY, AND THE FLOW ARM IS NOT OFFERED BECAUSE IT SAVES NOTHING.
 * For AF3 the flow/diffusion switch earns its place: that model's diffusion
 * default is 200 steps and flow-16 is a twelve-fold saving. ESMFold2's own
 * sampler is ELEVEN steps - `inference_num_steps: 15` truncated by
 * `max_inference_sigma` - so there is nothing to escape from, and a step costs
 * the same either way. `gamma0 = 0` stops noise being re-injected; it does not
 * make a step cheaper. Counted:
 *
 * | preset | asked | steps actually run |
 * |---|---|---|
 * | diffusion-15 (the checkpoint's own) | 15 | **11** |
 * | flow-16 | 16 | **12** |
 * | diffusion-32 / flow-32 | 32 | 23 each |
 *
 * So the flow arm at its usual setting runs MORE steps than the shipped
 * sampler, for a sampler the model was not trained with. It stays in
 * SAMPLER_PRESETS - the measurements are worth keeping and a tool may ask for
 * it - and the page does not offer a choice whose every option is
 * equivalent-or-worse.
 *
 * 🔴 AND SIX STEPS IS NOT "FASTER", IT IS BROKEN, so `diffusion-8` is measured
 * and not offered: CA-CA 58 A where a peptide bond is 3.8. The churn overshoots
 * when the levels are too far apart to correct it.
 */
export const ESMFOLD2_COUNTS = {
  diffusion: { label: "Diffusion", values: [15, 32, 64, 200], preferred: 15 },
};

/** The sampler mode this model runs, whatever a shared control says. */
export const ESMFOLD2_SAMPLER_MODE = "diffusion";

/**
 * How many steps a preset ACTUALLY runs, which is not the number in its name.
 *
 * 🔴 `max_inference_sigma` DROPS EVERY SCHEDULE ENTRY ABOVE 256 AND PREPENDS
 * THE CAP, so the checkpoint's `inference_num_steps: 15` runs ELEVEN. The dial
 * offering "15" beside a status line reading "11 steps" is the page
 * contradicting itself, and the config number is the one with no operational
 * meaning. Computed from the schedule rather than tabulated, so it stays true
 * if the sampler constants move.
 */
export function actualSteps(preset) {
  const settings = { ...SAMPLER_DEFAULTS, ...(SAMPLER_PRESETS[preset] ?? {}) };
  const schedule = noiseSchedule({
    steps: settings.steps, sMax: settings.sMax, sMin: settings.sMin,
    p: settings.p, sigmaData: settings.sigmaData, maxSigma: settings.maxSigma,
  });
  return noiseLevels(schedule,
    churnFactors(schedule, settings.gammaMin, settings.gamma0)).length;
}

const weightsPromises = new Map();

/**
 * 🔴 MEMOISED, AND KEYED ON NOTHING, BECAUSE THERE IS ONE OF THESE. AF3's
 * loader is a Map keyed by family because two bundles build that graph and the
 * first version handed the second family the first one's weights. There is one
 * ESMFold2 bundle; a second has appeared, so this is a Map.
 *
 * 🔴 KEYED BY FAMILY, AND THE `languageModel` FLAG RIDES ON THE FIRST CALL.
 * That flag only decides whether the tower PREFETCHES, so a second fold of the
 * same family that does want it still gets its weights - the store fetches on
 * demand and the tower streams its blocks during the fold anyway. What it
 * costs is the head start and the download dial, in the order "fold a ligand,
 * then fold a protein". Left as it is because the alternative is a second key
 * on a flag that changes no weights.
 *
 * 🔴 KEYED BY FAMILY, BECAUSE A SECOND CHECKPOINT IS MISTAKEN FOR THE FIRST IN
 * A CACHE AND NOT IN A LOADER. `loadAf3Weights` memoised ONE promise and the
 * second family's fold got the first family's weights; the shapes agree, so
 * nothing errors. EF2-fast's two pairs have the same trunk shapes and different
 * shims, which is the same trap one model over.
 */
export function loadEsmfold2Weights(onProgress,
                                    { languageModel = true,
                                      family = "ef2-fast-600m" } = {}) {
  const memo = weightsPromises.get(family);
  if (memo !== undefined) return memo;
  const promise = (async () => {
    // 🔴 THE COMPANION IS READ FROM THE REGISTRY, NOT WRITTEN DOWN HERE. A
    // folding bundle names the tower whose shim belongs to it, so a third pair
    // - ESM-C 6B is published too - is a registry entry rather than a branch.
    const tower = MODEL_BUNDLES[family]?.companion;
    if (tower === undefined) {
      throw new Error(`${family} names no language model bundle`);
    }
    const [foldManifest, towerManifest] = await Promise.all([
      loadManifest(family), loadManifest(tower),
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
    // 🔴 REPORTING STOPS WHEN THE LOAD DOES, OR THE DIAL NEVER CLEARS. The
    // tower STREAMS - its 36 blocks are read one at a time during the fold, by
    // the callback the tower calls - so its store goes on firing progress long
    // after this promise resolves. `startModelPreload` clears the dial when the
    // promise settles, and the next shard read put it straight back, at
    // "346 / 346 MiB", where it stayed for the rest of the session. What the
    // dial means is the DOWNLOAD; the streaming is the fold's own business and
    // the status line already narrates it.
    let loading = true;
    const report = (key) => (progress) => {
      if (!loading) return;
      seen.set(key, progress);
      let loadedBytes = 0, totalBytes = 0;
      for (const value of seen.values()) {
        loadedBytes += value?.loadedBytes ?? 0;
        totalBytes += value?.totalBytes ?? 0;
      }
      onProgress?.({ ...(seen.get(key) ?? {}), loadedBytes, totalBytes });
    };
    const [foldStore, towerStore] = await Promise.all([
      HttpTensorStore.fromManifest(bundleBaseUrl(family), foldManifest,
                                   report("fold")),
      // 🔴 AND ITS PROGRESS IS NOT REGISTERED WHEN IT IS NOT BEING FETCHED, or
      // the dial promises 346 MiB and stops at a third of it. The reporter sums
      // `totalBytes` across both stores, so a store that never downloads has to
      // be absent from the sum rather than merely idle.
      HttpTensorStore.fromManifest(bundleBaseUrl(tower), towerManifest,
                                   languageModel ? report("tower") : undefined),
    ]);
    foldStore.prefetch();
    // 🔴 224 MiB FOR A TOWER THE FOLD WILL NEVER CALL. ESM-C sees PROTEIN
    // tokens only - `protein_mask = (mol_type == 0) & token_mask` - so a fold
    // whose input is a ligand, or DNA, or RNA and nothing else has no row to
    // give it, and `foldEsmfold2` already skips the call at
    // `lm.ids.length === 0`. What it could not skip was the download, because
    // this prefetch starts every shard the moment the model is chosen.
    //
    // 🔴 THE SHIM STILL COMES FROM THIS STORE AND IS STILL READ. Its LayerNorm
    // has an offset and its downprojection a bias, so `shim(0)` is a fixed
    // NON-ZERO vector and a fold with no protein still needs those tensors -
    // see shimSingleForZeroState. They are a few hundred KB fetched on demand;
    // the 36 blocks are the 224 MiB, and those are what go unfetched.
    if (languageModel) towerStore.prefetch();

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

    loading = false;
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
        // 🔴 THE FOUR BIG MATRICES COME BACK ALREADY NARROW. They are 99.6% of
        // a block and the tower stores them at half precision anyway, so
        // decoding them to float32 first is a whole extra pass over 573 M
        // parameters - 723 ms a fold and 2.3 GB of intermediate. The six
        // vectors stay float32: they are the norms, and the tower uploads them
        // as they are.
        block: async (layer) => {
          const block = {};
          for (const leaf of BLOCK_LEAVES) {
            const name = `blocks/${layer}/${leaf}`;
            block[leaf] = NARROW_LEAVES.has(leaf)
              ? await towerStore.tensorAsFloat16(name) : await towerStore.tensor(name);
          }
          return block;
        },
      },
    };
  })();
  weightsPromises.set(family, promise);
  return promise;
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
  return async (ids, sequenceId, onBlock) => {
    const tower = new EsmcTowerGpu(device, allocator);
    const result = await tower.run(ids, {
      rows: ids.length,
      model: loaded.language.manifest.width,
      heads: loaded.language.manifest.heads ?? loaded.language.manifest.width / 64,
      ffn: loaded.language.ffn,
      layers: loaded.language.manifest.layers,
      pair: pairChannels,
      residualScale: loaded.language.manifest.residualScale ?? 1,
    }, loaded.language.block, loaded.language.shared, { sequenceId, onBlock });
    return result.single;
  };
}
