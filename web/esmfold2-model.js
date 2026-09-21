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
import { tokenLayoutFrom } from "./fold-archive.js";
import { alphaCarbons, fittedPdb, toPoints } from "./af3-model.js";
import { toPdb } from "../src/af3/fold.js";
import { foldEsmfold2 } from "../src/esmfold2/fold.js";
import { spreadOverAtoms, toDensePositions } from "../src/esmfold2/featurise.js";
import { throwIfAborted } from "../src/runtime/abort.js";
import { GpuBufferAllocator } from "../src/runtime/allocator.js";
import { smilesComponent } from "../src/chem/component.js";
import { ccdUrl, parseCcdComponent } from "../src/af3/featurise/ccd-component.js";
import { HttpTensorStore } from "../src/bundles/http-tensor-store.js";
import { readTensor } from "../src/weights/dtype.js";
import { MODEL_BUNDLES, bundleBaseUrl, loadManifest } from "../src/bundles/manifests/index.js";
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
 * 🔴 KEYED BY FAMILY, AND THE `languageModel` FLAG DOES NOT BELONG IN THE KEY.
 * It changes no weights - only whether the tower's shards are fetched ahead of
 * time - and keying on it would download the FOLDING bundle twice for one
 * family. The decision can change after the promise is cached, so the loaded
 * object carries `language.prefetch()` for a caller that later finds it does
 * want the tower; see the note there.
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
    let towerPrefetched = false;
    if (languageModel) {
      towerPrefetched = true;
      towerStore.prefetch();
    }

    // 🔴 AND IT CARRIES THE TWO HALVES A LAZY LOADER NEEDS. `source` hands the
    // bytes over undecoded so the GPU can dequantise them, and `decode` is the
    // synchronous host fallback a getter can call. See
    // src/esmfold2/weights.js: without these the loaders behave exactly as they
    // did, and with them the denoiser's 1.17 s of int5 decoding and 880 ms of
    // narrowing both stop happening on the main thread.
    const read = (name) => foldStore.tensor(name);
    read.source = async (name) => {
      await foldStore.open(name);
      return foldStore.tensorSource(name);
    };
    read.decode = (source) => readTensor(source.record, source.buffer, source.byteOffset, true);
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
        // ...what the language band in src/esmfold2/cost.js is priced on, taken
        // from the store that already counted it rather than written down.
        megabytes: towerStore.totalBytes / 1048576,
        shared: towerShared,
        // 🔴 A LATE PREFETCH, BECAUSE THE MEMO OUTLIVES THE DECISION. Whether
        // the tower is worth downloading depends on the ENTITIES and the PLM
        // row, and both can change after this promise is built and cached -
        // fold a ligand, then a protein, and the second inherited the first
        // one's "no language model" and lost its head start. The store still
        // serves it either way, so this is a head start rather than a
        // correctness fix; it is idempotent, and after the load has finished
        // its progress goes unreported, which is what the dial means.
        prefetch: () => {
          if (towerPrefetched) return;
          towerPrefetched = true;
          towerStore.prefetch();
        },
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
        // 🔴 AND THE FOUR ARE OFFERED AS CODES, NOT DECODED. `sources` is what
        // lets the tower decode them on the DEVICE - measured on a first
        // ESMFold2 fold, the block reads were 5482 ms of a 7990 ms wall and
        // this takes the fold to 3.3 s. The getters stay as the fallback for a
        // bundle the device decoder refuses, and they must be getters or the
        // decode this exists to skip happens anyway.
        block: async (layer) => {
          const block = { sources: {} };
          for (const leaf of BLOCK_LEAVES) {
            const name = `blocks/${layer}/${leaf}`;
            if (!NARROW_LEAVES.has(leaf)) {
              block[leaf] = await towerStore.tensor(name);
              continue;
            }
            await towerStore.open(name);
            let source;
            try { source = towerStore.tensorSource(name); } catch { source = undefined; }
            if (source !== undefined) block.sources[leaf] = source;
            let decoded;
            Object.defineProperty(block, leaf, {
              enumerable: true,
              get() { return (decoded ??= towerStore.tensorAsFloat16Sync(name)); },
            });
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

/** Per-chain means, ignoring tokens with no value. */
export function meanByChain(asymId, values) {
  if (values === undefined) return undefined;
  const chains = [...new Set(asymId)].sort((a, b) => a - b);
  return chains.map((chain) => {
    let sum = 0, seen = 0;
    for (let token = 0; token < asymId.length; token += 1) {
      if (asymId[token] !== chain || !(values[token] >= 0)) continue;
      sum += values[token];
      seen += 1;
    }
    return seen === 0 ? null : Math.round((sum / seen) * 100) / 100;
  });
}

/**
 * A fold's RESULT turned into the prediction every surface reads.
 *
 * 🔴 ONE ASSEMBLY PER GRAPH, AND THE SAME SHAPE FOR EACH. `predictionFromAf3`
 * in web/af3-model.js is the other one, and they take the same two arguments
 * for the same reason: a runtime that folds WITHOUT a page has to produce
 * exactly what the page produces, and a second field-by-field rebuild is how
 * this project has lost `gpu`, `align`, `position_atoms`, `maps` and `pae_n`,
 * each in silence.
 *
 * What differs between the two is the MODEL, not the convention: this
 * checkpoint has no confidence head, so there is no `confidence` object at
 * all, its pAE is estimated rather than predicted and travels under its own
 * name, and its settings are its own - it reads neither the recycle dial nor
 * the MSA depth. Those three are the whole of the difference and each is
 * commented where it sits.
 *
 * 🔴 AND `settings` IS PASSED IN, BECAUSE THREE OF ITS FIELDS ARE CONTROLS.
 * `recycleCount()`, `plmLabel()` and `samplerPreset()` read the page; a
 * process has none, so the caller states them.
 *
 * @param {object} result what the ESMFold2 fold returned
 * @param {{chains: string[], stem: string, pdb: string, modelName: string,
 *          certainty?: ArrayLike<number>, settings: object, context?: object}} about
 */
export function predictionFromEsmfold2(result, about) {
  const { chains, stem, pdb, modelName, certainty } = about;
  const foldContext = about.context ?? {};
  return {
    stem, pdb, chains,
    chainLengths: chains.map((chain) => chain.length),
    // 🔴 NO `confidence`, AND THAT IS THE HONEST SHAPE. Everything that reads a
    // prediction's confidence - the scores card, the archive's summary, the PAE
    // panel - asks for fields this checkpoint has no head to compute. An object
    // carrying zeros would be read as the model's opinion.
    // 🔴 ONE FIELD FOR THE CONTACT MAP, WHATEVER PRODUCED IT. See the note on
    // `contactSource` at the download button: this used to be `contacts` here,
    // `confidence.contactProbs` on the AF3 path and `contactSource` on AF2's,
    // and the archive knew about two of the three - so the model whose contact
    // map is its ONLY score wrote an archive without one while the panel on
    // screen showed it.
    contactSource: { contactProbs: result.contacts },
    // 🔴 AND THE pAE, WHICH IS NOT A `confidence` FIELD AND MUST NOT BECOME
    // ONE. It is estimated from the distogram rather than predicted by a head -
    // see src/esmfold2/aligned-error.js - so putting it under `confidence`
    // would let every reader that tests for that object conclude this
    // checkpoint has one, and start looking for the pLDDT and pTM beside it.
    // It is its own field, named for what it is.
    alignedError: result.alignedError,
    // 🔴 WITHIN EACH CHAIN AND ACROSS IT, KEPT APART. The certainty a residue
    // wears is about its own chain; the interface is a different question and
    // averaging them gives a number that answers neither. Measured on a
    // two-chain fold: 0.712 within, 0.370 across, 0.630 mixed.
    chainCertainty: meanByChain(result.features.asymId, certainty),
    chainInterfaceCertainty: meanByChain(result.features.asymId,
                                         result.interfaceCertainty),
    model: modelName,
    // 🔴 THE TOKEN LAYOUT, because a ligand is one token per heavy atom and the
    // archive cannot infer that from the chain lengths - it refuses to guess
    // and throws. See tokenIdentifiers.
    tokens: tokenLayoutFrom(result.features.asymId, result.features.residueIndex),
    // ...and the entities and the templates, which the archive's job request is
    // made of. Without these "Download all" wrote a request naming no
    // sequences.
    ...foldContext,
    // 🔴 THIS MODEL'S OWN SETTINGS, OVER THE SHARED DIALS'. `foldContext` is
    // built before the branch and carries the recycle count and the MSA depth
    // that AF2 and AF3 read - and this model reads NEITHER: it folds from the
    // sequence alone, which is why the page hides its MSA row, and its trunk
    // loops a number of times the checkpoint fixes rather than the dial. The
    // archive said `recycles: 1` and `max msa: 128` for a fold that used one
    // value of neither.
    settings: {
      seed: foldContext.settings?.seed,
      // Stated by the caller: the first three read page controls, and a
      // process has none. `diffusion steps` is the fold's own answer, so it
      // is taken from the result rather than asked for.
      ...(about.settings ?? {}),
      "diffusion steps": result.steps,
    },
    // ...and no alignment or template line, rather than "none", which reads as
    // a choice. This model takes neither: `grep -rn template` over ESMFold2's
    // whole upstream package returns nothing, and `z_init` has five terms with
    // none of them one.
    msaOrigin: undefined,
    msas: {},
    templates: undefined,
  };
}

/**
 * One ESMFold2 fold, end to end, with no page in it.
 *
 * 🔴 THE THIRD GRAPH GETS THE ARRANGEMENT THE OTHER TWO HAVE. `foldAf3` is
 * this for the AlphaFold 3 graph; this was 383 lines inside `foldWithEsmfold2`
 * in web/app.js, with the compute and the display INTERLEAVED in four
 * alternating bands - so a runtime with no page could not fold this
 * checkpoint at all, and tools/colab_runtime.mjs refused it by name.
 *
 * What stays with the caller is everything that needs a document: the viewer,
 * the status line, the download panel, the trunk CACHE. What comes here is
 * the work.
 *
 * 🔴 AND THE FIVE CONTROLS IN THE TRUNK KEY ARE OPTIONS NOW, WHICH IS THE
 * PART THAT COULD HAVE GONE WRONG QUIETLY. `chosenFamily`, `recycleCount`,
 * `plmChoice`, `usesLanguageModel` and `randomSeed` all feed `trunkKey`, and
 * a trunk reused against the wrong key is not an error - it is a fold that
 * answers with somebody else's trunk. The key is computed HERE, from those
 * options, and the caller is asked `reuseFor(key)` rather than being trusted
 * to rebuild the same key beside its cache.
 *
 * @param {{chains: string[], chainKinds: string[], ligandCodes?: Array,
 *   modifications?: Array, signal: AbortSignal, device: GPUDevice,
 *   modelLoad?: Promise<object>, family: string, modelName: string,
 *   languageModel: boolean, plm: string, loops: number, sampler: string,
 *   seed: number, reuseFor?: (key: string) => object|undefined,
 *   onStatus?: (text: string) => void, onProgress?: (n: number) => void,
 *   onFrame?: (pdb: string, certainty: unknown) => void,
 *   onContacts?: (contacts: unknown, certainty: unknown) => void,
 *   onBatch?: (batch: object) => void}} options
 */
export async function foldEsmfold2Job(options) {
  const { chains, chainKinds, signal, device, modelLoad, family, modelName,
          languageModel, plm, loops, sampler, seed } = options;
  const ligandCodes = options.ligandCodes ?? [];
  const modifications = options.modifications ?? [];
  const onStatus = (text) => options.onStatus?.(text);
  const framePdbs = [];
  const sequence = chains.join(":");
  onStatus(`${modelName} · loading`);
  // ...the long name is for the download dial, where provenance matters; the
  // status line uses the short one, because it is written many times a fold.
  // 🔴 THE LIGAND DICTIONARY IS FETCHED, NOT BUNDLED, exactly as on the AF3
  // path - and from the same place, because these are the same components. A
  // fold touches only the codes its ligands name and the PDB serves each as one
  // small mmCIF; the 21 polymer components stay baked.
  const ligands = [];
  for (const entry of ligandCodes) {
    // A structure rather than a code; see the note in web/af3-model.js.
    if (typeof entry !== "string") {
      onStatus(`${modelName} · building ${entry.code ?? "ligand"}`);
      ligands.push(await smilesComponent(entry.smiles, { code: entry.code ?? "LIG" }));
      continue;
    }
    onStatus(`${modelName} · fetching ligand ${entry}`);
    const response = await fetch(ccdUrl(entry), { signal });
    if (!response.ok) {
      throw new Error(`No chemical component ${entry} at the PDB (${response.status})`);
    }
    ligands.push(parseCcdComponent(await response.text()));
  }
  // 🔴 AND A MODIFIED RESIDUE'S COMPONENT FROM THE SAME PLACE, for the reason
  // web/af3-model.js gives: the featuriser is synchronous, and this is the one
  // piece of a batch that cannot be computed from the sequence.
  const modifyWith = [];
  for (const modification of modifications) {
    onStatus(`${modelName} · fetching modified residue ${modification.code}`);
    const response = await fetch(ccdUrl(modification.code), { signal });
    if (!response.ok) {
      throw new Error(`No chemical component ${modification.code}`
        + ` at the PDB (${response.status})`);
    }
    modifyWith.push({ chain: modification.chain, position: modification.position,
                      ...parseCcdComponent(await response.text()) });
  }
  throwIfAborted(signal);
  const loaded = await (modelLoad
    ?? loadEsmfold2Weights(undefined, { languageModel, family }));
  // 🔴 ASKED AGAIN HERE, BECAUSE THE PRELOAD DECIDED IT EARLIER AND THE MEMO
  // OUTLIVES BOTH. `startModelPreload` skips the tower's 223.6 MiB when the
  // entities hold no protein or the PLM row says none - and either can have
  // changed since, or the promise can have been built for a previous fold that
  // did not want it. Idempotent, and it is a head start rather than a
  // correctness fix: the store serves the blocks on demand either way.
  if (languageModel) loaded.language.prefetch?.();
  throwIfAborted(signal);
  throwIfAborted(signal);

  // `centreRandomAugmentation` draws a fresh rotation and translation of the
  // whole system at the top of each step - it is how the sampler is equivariant
  // and the model was trained with it in the loop - so consecutive frames differ
  // by a rigid motion far larger than anything the denoiser did, and unfitted
  // playback is a protein tumbling. AF3's path has fitted its trajectory since
  // it had one; this is the same function, not a second one.
  //
  // 🔴 AND TO THE FIRST FRAME, NOT THE LAST, because the frames are drawn as
  // they are computed and there is no last one yet.
  let reference = null;
  let slots;
  // 🔴 THE COLOUR IS THE DISTOGRAM'S CERTAINTY, NOT A pLDDT, AND THE PDB SAYS
  // SO IN A REMARK. This checkpoint has no confidence head - 820 tensors and
  // not one named confidence, plddt, pae or pde - so what goes in the B-factor
  // is an ORDERING with nothing to calibrate a number against. It is written
  // there because that is the only column a viewer can colour from, and a
  // downloaded file that carried an uncommented pLDDT-shaped column would be
  // read as one. See CERTAINTY in src/esmfold2/distogram-webgpu.js for the
  // sweep that chose its three constants.
  let certainty;
  let lastFrameCertainty;
  const REMARK = "REMARK   1 B-FACTOR IS DISTOGRAM CERTAINTY (0-100), NOT pLDDT."
    + "\nREMARK   1 THIS ESMFOLD2 CHECKPOINT HAS NO CONFIDENCE HEAD.";
  const withRemark = (pdb) => `${REMARK}\n${pdb}`;

  const started = performance.now();
  // 🔴 THE SAME CACHE AlphaFold 3's PATH HAS, AND FOR THE SAME REASON: the trunk
  // is the fold, so changing only the sampler should cost only the sampler. The
  // key is what the TRUNK depends on and nothing else - the checkpoint, the
  // chains and their kinds, the ligands, the pass count, and which language
  // model, since "none" and ESM-C 600M share a family and produce different
  // pairs. The seed is in it only when masking is on, because that is the only
  // way the seed reaches the trunk: `lm_mask_pct` is zero in this checkpoint, so
  // asking for a different SAMPLE reuses the trunk here where AF3 re-runs it.
  // 🔴 AND THE MASK IS IN IT, WITH THE SEED BEHIND IT. `lm_mask_pct` replaces a
  // fraction of the residues with the mask token BEFORE the tower runs, drawn
  // from the seed - so with masking on, two seeds are two different trunk
  // inputs and a key without them hands the second fold the first one's pair.
  // This checkpoint sets the fraction to 0, so the seed never reaches the trunk
  // and changing it reuses; but the config class documents single-sequence
  // checkpoints as setting 0.1, so a future bundle turns this on by existing
  // and the key has to be right before that rather than after.
  const lmMask = (loaded.shape.lmMaskPct ?? 0);
  const trunkKey = JSON.stringify({
    family, chains, chainKinds, ligandCodes,
    // A modification changes what is folded, so a trunk cached for the plain
    // chain is not this fold's.
    modifications: modifications.map((one) => `${one.code}@${one.position}`),
    loops,
    plm, languageModel,
    lmMask, maskSeed: lmMask > 0 ? seed : null,
  });
  const reuse = options.reuseFor?.(trunkKey);

  const result = await foldEsmfold2(device, {
    reuse,
    wantReusable: true,
    sequence,
    entities: { sequence, chainKinds, ligands, modifications: modifyWith },
    // ...and which tokens the viewer will draw, so this path's contact map is
    // collapsed the way the AF3 one is. See viewerTokens.
    onBatch: (batch) => options.onBatch?.(batch),
    // 🔴 THE RECYCLE DIAL DRIVES THIS TRUNK TOO, AND USED NOT TO. Its loop
    // count came from the checkpoint and the control beside it did nothing -
    // the "quietly ignored control" syncModelControls exists to prevent, which
    // is why the MSA row is hidden here rather than left on screen. The mapping
    // is exact: upstream runs `range(num_loops + 1)` and this checkpoint's
    // `num_loops` is 3, which is the dial's own default, so the default fold is
    // the same four passes it always was.
    shape: { ...loaded.shape, loops },
    weights: loaded.weights,
    tower: languageModelRunner(device, new GpuBufferAllocator(device), loaded,
                               loaded.shape.pairChannels),
    sampler,
    seed,
    // 🔴 THIS MODEL'S "SINGLE SEQUENCE". Without ESM-C it has no evolutionary
    // information at all - measured on a 76-mer, the fold moves 10.96 A, the
    // distogram predicts NO long-range contact, and the certainty falls from
    // 0.95 to 0.42, which is the confidence estimate correctly reporting that
    // the answer is worthless.
    languageModel,
    // ...and how big it is, so the bar's language band is this tower's and not
    // the one the constants were fitted against.
    languageModelMiB: loaded.language.megabytes,
    // 🔴 EACH FRAME GETS ITS OWN COLOUR, WHICH NEEDS THE DISTOGRAM RESIDENT.
    // The trunk's own certainty is fixed for a fold, so every frame would wear
    // the same one - and the interesting thing about a trajectory is watching
    // it become confident. Scoring each frame against the distogram costs the
    // logits staying on the device, 46 MiB at 300 tokens, released with the
    // last frame.
    frameCertainty: true,
    // 🔴 THE LINE AND THE BAR ARE TWO CALLBACKS NOW, AS AF3's ARE. One phase
    // word plus a percentage on the line; the fraction drives the bar. The
    // first version wrote a stage name per stage, and a two-millisecond recycle
    // between two multi-second trunk passes made it flicker.
    onStatus: (text) => { if (!signal.aborted) onStatus(`${modelName} · ${text}`); },
    onProgress: (fraction) => { if (!signal.aborted) options.onProgress?.(fraction); },
    // 🔴 THE CONTACT MAP EXISTS BEFORE ANY STRUCTURE DOES, because the
    // distogram head runs off the trunk and the sampler has not started. It is
    // held until there is a frame to hang it on, exactly as the AF3 path holds
    // its own.
    onContacts: (contacts, trunkCertainty) => {
      certainty = trunkCertainty;
      options.onContacts?.(contacts, trunkCertainty);
    },
    // 🔴 `denoised` AND NOT `coordinates`, AND THE REASON IS THE CAMERA. The
    // sampler's own walk starts as Gaussian noise at sigma 411 and ends at a
    // protein, so no fixed camera holds both and the early frames are not a
    // picture of anything. `denoised` is the model's predicted structure at each
    // call - EDM preconditioning included, so at a large noise level it is
    // almost all network - and is protein-sized in every frame. AF3's path
    // records the same finding, measured: a radius of gyration of 1896 A at
    // step 4 against 11.1 at the end.
    onStep: ({ denoised, features, certainty: frameCertainty }) => {
      if (signal.aborted) return;
      const dense = toDensePositions(features, denoised);
      if (slots === undefined) slots = alphaCarbons(features.batch);
      if (reference === null) {
        reference = toPoints(dense, features.batch.tokens * features.batch.dense);
      }
      // ...this frame's OWN agreement with the distogram, so an early frame
      // that has not converged is coloured as one rather than wearing the
      // finished structure's confidence. The trunk's mode-based certainty is
      // the fallback, and it is the same quantity measured a different way -
      // the two scored a tie on the sweep.
      const shown = frameCertainty ?? certainty;
      const pdb = withRemark(fittedPdb(features.batch, dense, reference, slots,
        shown === undefined ? null : spreadOverAtoms(features, shown, 100)));
      framePdbs.push(pdb);
      options.onFrame?.(pdb, shown);
      lastFrameCertainty = shown;
    },
  });

  throwIfAborted(signal);

  // 🔴 THE ANSWER IS FITTED ONTO THE SAME REFERENCE AS THE TRAJECTORY, or the
  // last frame of the play bar jumps by a rigid motion the fold did not make.
  // It is still `result.coordinates` - the sampler's own answer, not the last
  // denoiser call - and at the bottom of the schedule the two agree to a
  // fraction of an angstrom anyway.
  // 🔴 THE FINISHED STRUCTURE KEEPS THE LAST FRAME'S SCORE, not the trunk's.
  // The two are the same quantity read two ways, but the play bar would step
  // from a per-frame colour to a different one on its last frame, which reads
  // as the fold changing its mind at the end.
  certainty = lastFrameCertainty ?? result.certainty ?? certainty;
  const bFactors = certainty === undefined
    ? null : spreadOverAtoms(result.features, certainty, 100);
  const finalDense = toDensePositions(result.features, result.coordinates);
  const pdb = withRemark(reference === null
    ? toPdb(result.features.batch, finalDense, bFactors)
    : fittedPdb(result.features.batch, finalDense, reference,
                slots ?? alphaCarbons(result.features.batch), bFactors));

  return { result, pdb, framePdbs, certainty, trunkKey,
           seconds: (performance.now() - started) / 1000 };
}
