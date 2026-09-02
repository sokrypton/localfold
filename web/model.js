/**
 * Getting the device and the parameters, once per page.
 *
 * LOADED ONCE PER PAGE, not once per fold. These pages exist to be poked at -
 * try a sequence, change a residue, try again - and re-reading the weights
 * between two attempts would make that unusable. The device is kept for the
 * same reason: requestAdapter is not free either.
 *
 * WHY THIS IS ITS OWN FILE. Two pages want it: the single-sequence page and the
 * MSA page. They differ in exactly one tensor group, and everything else about
 * loading - the store, the device, the memoisation, the progress reporting - is
 * the same code. Sharing it here is the difference between one loader and two
 * that drift.
 *
 * NO DOM IN HERE. Progress arrives as a callback, because the two pages report
 * it differently and neither one's markup belongs in a module about weights.
 */
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";
import { ScriptTensorStore } from "../src/reference/script-tensor-store.js";
import { MODEL_BUNDLES, bundleBaseUrl, loadManifest } from "../src/reference/manifests/index.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
import { withAbort } from "../src/runtime/abort.js";

const stores = new Map();

/**
 * The tensor store for one model family, on whatever this origin can use.
 *
 * 🔴 ONE WAY IN FOR EVERY MODEL. The monomer read a compiled-in manifest and
 * the multimer FETCHED one, so they failed differently: a site without multimer
 * weights 404ed on model-multimer/manifest.json and died there, before a single
 * shard was asked for. Both tables are now modules - see
 * src/reference/manifests/ - so neither can 404, and the first thing that can
 * fail is a shard, which is a failure about weights rather than about metadata.
 *
 * Over http the shards are fetched directly. On a file:// page fetch does not
 * work at all, so the weights come in as classic scripts carrying base64 data:
 * URLs - see ScriptTensorStore, and tools/export-js-weights.py, which writes
 * them.
 *
 * `?model=` overrides the monomer path, which is how a page is pointed at a
 * manifest somewhere else without editing it.
 *
 * @param {import("../src/reference/manifests/index.js").ModelFamily} family
 */
export function openStore(onProgress, family = "monomer") {
  const bundle = MODEL_BUNDLES[family];
  if (bundle === undefined) throw new RangeError(`unknown model family ${family}`);
  const override = family === "monomer"
    ? new URLSearchParams(location.search).get("model") : null;
  if (override !== null) return HttpTensorStore.open(override, onProgress);
  let store = stores.get(family);
  if (store === undefined) {
    store = (async () => {
      const manifest = await loadManifest(family);
      const offline = location.protocol === "file:";
      const Store = offline ? ScriptTensorStore : HttpTensorStore;
      // 🔴 THE SHARDS COME FROM THE BUNDLE'S BASE, WHICH MAY BE OFF-ORIGIN. A
      // model hosted on Hugging Face resolves to an absolute URL and the store
      // does not need to know the difference: shard paths are resolved against
      // whatever base it was opened with. See bundleBaseUrl.
      //
      // 🔴 EXCEPT OFFLINE, WHICH IS THE WHOLE POINT OF BEING OFFLINE. A file://
      // bundle carries its weights beside it as base64 scripts and reads them
      // with ScriptTensorStore; pointing that at a remote would make the one
      // build that must not need the network the only one that always does.
      const base = offline ? bundle.directory : bundleBaseUrl(family);
      const opened = await Store.fromManifest(base, manifest, onProgress);
      // ...every shard at once; see HttpTensorStore.prefetch. AF2's loaders read
      // the whole bundle too.
      opened.prefetch?.();
      return opened;
    })();
    stores.set(family, store);
  }
  return store;
}

let devicePromise;

/** The WebGPU device, with the optional features the fast paths look for. */
export function getDevice() {
  devicePromise ??= (async () => {
    if (navigator.gpu === undefined) throw new Error("This browser has no WebGPU. It ships in current Chrome, Edge, Safari and Firefox.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("No compatible WebGPU adapter was found");
    // 🔴 THE PAGE IS THE ONE CALLER THAT HAS TO SURVIVE BEING WRONG. A bench
    // that asks for too much should fail loudly and does; a page that asks for
    // too much takes the machine down with it, so this is where the ceiling is
    // set. `null` takes the guess in device-memory.js from navigator.deviceMemory.
    return requestAlphaFoldDevice(adapter, { memoryBudgetBytes: null });
  })();
  return devicePromise;
}

const loaded = new Map();

/**
 * The parameters for one of the two inference paths.
 *
 * 🔴 THE TWO VARIANTS DIFFER IN ONE TENSOR GROUP, and it is not a saving to be
 * clever about. With no MSA the extra stack has no rows to attend over, so the
 * single-sequence path uses PAIR-ONLY blocks and never loads the extra-MSA
 * attention parameters at all. Hand it `extraStackWeights` and the shapes still
 * fit; the blocks it builds simply are not the ones it runs. Hand the MSA path
 * `extraPairStackWeights` and it is missing the attention it needs. Naming the
 * variant at the call site is what keeps that straight.
 *
 * 🔴 NOTHING IS QUANTISED OR ROUNDED HERE. The shards arrive as int8 with a
 * float16 scale per 64-weight block wherever that is safe - see
 * tools/quantize_model.py, which keeps the structure module and the geometry
 * tables at float32 and records what each format costs - and src/reference/
 * dtype.js dequantises them on the way in. The values that reach this function
 * are the ones the page used to spend most of a fold computing.
 *
 * It used to round the whole tree on every press of Fold, on the reasoning that
 * redoing it beat holding a second copy of 335 MiB. Both halves were wrong. The
 * conversion itself is cheap; what was not cheap was allocating a second
 * 371 MiB tree in the middle of a prediction, which cost about six seconds a
 * fold in garbage collection. And there is no second copy to avoid if the
 * precision is simply what was downloaded - which also took that download from
 * 355 MiB to 97.
 *
 * @param {"single"|"msa"} variant which inference path the weights are for
 * @param {(p: {loadedBytes: number, totalBytes: number}) => void} [onProgress]
 */
export function loadModel(variant, onProgress, signal = undefined, family = "monomer") {
  if (variant !== "single" && variant !== "msa") {
    throw new RangeError(`unknown model variant ${variant}: expected "single" or "msa"`);
  }
  if (family !== "monomer" && family !== "multimer") {
    throw new RangeError(`unknown model family ${family}: expected "monomer" or "multimer"`);
  }
  const key = `${family}:${variant}`;
  const cached = loaded.get(key);
  if (cached !== undefined) return withAbort(cached, signal);
  const pending = (async () => {
    const multimer = family === "multimer";
    const store = await openStore(onProgress, family);
    const fixture = AlphaFoldFixture.fromStore(store);
    const extraStackWeights = variant === "msa"
      ? fixture.extraStackWeights() : fixture.extraPairStackWeights();
    // 🔴 TWO DIFFERENT TEMPLATE TRACKS, and multimer's is not optional. The
    // monomer's `template` is the query-only residual, skipped when there are
    // no templates. Multimer's embedder runs every recycle regardless -
    // `template.enabled` is True for model_1_multimer_v3 and its wrapper adds
    // the activation to the pair unconditionally - so its weights are loaded
    // for every multimer fold, templates or not.
    const templateWeights = multimer ? Promise.resolve(undefined) : fixture.templateWeights();
    const templateEmbeddingWeights = multimer
      ? fixture.templateEmbeddingWeights() : Promise.resolve(undefined);
    const [embedding, template, templateEmbedding, extraStack, mainStack, structure, confidence,
      geometry, featureTables, paeBreaks] = await Promise.all([
      fixture.embeddingWeights(), templateWeights, templateEmbeddingWeights, extraStackWeights,
      fixture.mainStackWeights(), fixture.structureWeights(), fixture.confidenceWeights(),
      fixture.geometryTables(), fixture.queryOnlyFeatureTables(), fixture.tensor("confidencePaeBreaks"),
    ]);
    return {
      featureTables,
      paeBreaks,
      weights: {
        embedding, template, templateEmbedding, extraStack, mainStack, structure,
        lddt: confidence.lddt, pae: confidence.pae, geometry,
      },
    };
  })();
  loaded.set(key, pending);
  return withAbort(pending, signal);
}
