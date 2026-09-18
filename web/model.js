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
import { AlphaFoldFixture } from "../src/bundles/alphafold-fixture.js";
import { HttpTensorStore } from "../src/bundles/http-tensor-store.js";
import { tensorByteLength } from "../src/weights/dtype.js";
import { ScriptTensorStore } from "../src/bundles/script-tensor-store.js";
import { MODEL_BUNDLES, bundleBaseUrl, graphFamily, loadManifest }
  from "../src/bundles/manifests/index.js";
import { DeltaTensorStore } from "../src/bundles/delta-tensor-store.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
import { devUseDevice } from "./dev-log.js";
import { withAbort } from "../src/runtime/abort.js";

const stores = new Map();
/** What `openStore` loads when no family is named, and the only one `?model=` overrides. */
const DEFAULT_FAMILY = "monomer";

/**
 * The tensor store for one model family, on whatever this origin can use.
 *
 * 🔴 ONE WAY IN FOR EVERY MODEL. The monomer read a compiled-in manifest and
 * the multimer FETCHED one, so they failed differently: a site without multimer
 * weights 404ed on model-multimer/manifest.json and died there, before a single
 * shard was asked for. Both tables are now modules - see
 * src/bundles/manifests/ - so neither can 404, and the first thing that can
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
 * 🔴 AND THE SAME PARAMETER NAMES A FAMILY, WHICH IS TWO READERS OF ONE
 * CONTROL. `applyModelFromUrl` in web/app.js takes `?model=` as the model row's
 * value - `?model=af3`, `?model=ef2-fast-300m` - and this took it as a manifest
 * URL whenever the family was monomer. So `?model=monomer`, which is the one
 * spelling that reaches both, selected AlphaFold 2 and then fetched
 * `https://localfold.org/monomer`, and the page said "failed to load model
 * manifest: 404" for a model it had loaded a second earlier by dropdown.
 *
 * A path is what this override was for, so a path is what it now requires: a
 * value with a slash in it, or one ending in `.json`. A bare family name is the
 * other reader's.
 *
 * @param {import("../src/bundles/manifests/index.js").ModelFamily} family
 */
export function openStore(onProgress, family = DEFAULT_FAMILY) {
  const bundle = MODEL_BUNDLES[family];
  if (bundle === undefined) throw new RangeError(`unknown model family ${family}`);
  // 🔴 THE DEFAULT FAMILY, NOT THE MONOMER GRAPH. `?model=` names a path to
  // load INSTEAD of a bundle, so it belongs to the one family that is asked
  // for when nobody asked - pointing it at `monomer-3` would fetch model_1's
  // export and call it model_3. Every other family, delta or not, ignores it.
  const asked = family === DEFAULT_FAMILY
    ? new URLSearchParams(location.search).get("model") : null;
  const override = asked !== null && (asked.includes("/") || asked.endsWith(".json"))
    ? asked : null;
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
      // 🔴 ONE PROGRESS STREAM OVER BOTH STORES, OR THE DIAL FLICKERS. A delta
      // family downloads TWO bundles - 43 MiB of difference and the 73 MiB base
      // it is added to - and handing each the caller's callback lets them take
      // turns owning the dial: it reads "4 of 43 MiB", then "20 of 73", then
      // back, for as long as they overlap. Reported as ONE total it is a single
      // bar over 116 MiB. It only shows when the base is NOT already cached,
      // which is exactly the visitor who picks model 2 first, and it is the
      // same failure web/esmfold2-model.js records for its fold bundle and its
      // language model - the two are not shared yet because that one also has
      // to stop reporting when the load ends, its tower going on streaming
      // through the fold, which nothing here does.
      const seen = new Map();
      // 🔴 AND THE BASE'S SIZE IS SEEDED BEFORE THE FIRST REPORT, or the arc
      // snaps back ONCE - not because the loaded count falls but because the
      // DENOMINATOR grows. The delta's store reports "0 / 43 MiB" the moment it
      // opens, before the base has a manifest, and the next update says
      // "0 / 116": same bytes, a third of the arc. A manifest is compiled in
      // rather than fetched, so the base's total is knowable up front without a
      // byte moving.
      // 🔴 AND ONLY WHEN THE BASE IS ACTUALLY GOING TO BE FETCHED. `stores`
      // caches by family, so a visitor who has already folded with model_1 gets
      // that store back with no download - and seeding its size would promise
      // 116 MiB and stop at 43. This is web/esmfold2-model.js's rule for its
      // language model, from the other direction: a store that never downloads
      // has to be absent from the sum.
      if (bundle.delta !== undefined && onProgress !== undefined
        && !stores.has(bundle.delta.base)) {
        const baseManifest = await loadManifest(bundle.delta.base);
        seen.set("base", { loadedBytes: 0, totalBytes: Object.values(baseManifest.tensors)
          .reduce((sum, record) => sum + tensorByteLength(record), 0) });
      }
      const report = (key) => (progress) => {
        seen.set(key, progress);
        let loadedBytes = 0;
        let totalBytes = 0;
        for (const value of seen.values()) {
          loadedBytes += value?.loadedBytes ?? 0;
          totalBytes += value?.totalBytes ?? 0;
        }
        onProgress?.({ ...progress, loadedBytes, totalBytes });
      };
      // ...and a bundle that is not a delta keeps the caller's callback
      // untouched, so the one-bundle path is the function it always was.
      const mine = bundle.delta === undefined || onProgress === undefined
        ? onProgress : report("self");
      const opened = await Store.fromManifest(base, manifest, mine);
      // ...every shard at once; see HttpTensorStore.prefetch. AF2's loaders read
      // the whole bundle too.
      opened.prefetch?.();
      // 🔴 A DELTA FAMILY IS HALF A MODEL AND OPENS THE OTHER HALF ITSELF.
      // AlphaFold 2's five models are one training run continued five ways, so
      // models 2 to 5 ship as 43 MiB of DIFFERENCE against model_1's 97 rather
      // than as five whole bundles - and a visitor who has already folded with
      // model_1 has the base in cache, so switching costs the delta alone. The
      // base is opened through this same function, which means its store is
      // SHARED with a plain model_1 fold rather than downloaded twice.
      if (bundle.delta === undefined) return opened;
      // 🔴 AND THE BASE'S SHARE IS ONLY IN THE SUM WHEN IT IS ACTUALLY BEING
      // FETCHED. `openStore` caches by family, so a visitor who has already
      // folded with model_1 gets that store back without a byte moving and the
      // reporter never fires for it - which is what the sum has to mean, or the
      // dial would promise 116 MiB and stop at 43.
      return new DeltaTensorStore(
        await openStore(mine === undefined ? undefined : report("base"), bundle.delta.base),
        opened);
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
    const device = await requestAlphaFoldDevice(adapter, { memoryBudgetBytes: null });
    // ...so the footer's timing panel can read this device's memory counters.
    // It reads them; it does not install anything on the device.
    devUseDevice(device);
    return device;
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
 * tables at float32 and records what each format costs - and src/bundles/
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
  // 🔴 A DELTA FAMILY IS ITS BASE'S GRAPH, so the check is on what it reduces
  // to and not on its name: "monomer-3" runs the monomer's code with model_3's
  // weights. Everything below reads the store, and DeltaTensorStore has already
  // made that store look like a whole model.
  const graph = graphFamily(family);
  if (graph !== "monomer" && graph !== "multimer") {
    throw new RangeError(`unknown model family ${family}: expected "monomer" or "multimer"`);
  }
  const key = `${family}:${variant}`;
  const cached = loaded.get(key);
  if (cached !== undefined) return withAbort(cached, signal);
  const pending = (async () => {
    const multimer = graph === "multimer";
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
      geometry, featureTables, paeBreaks, distogram] = await Promise.all([
      fixture.embeddingWeights(), templateWeights, templateEmbeddingWeights, extraStackWeights,
      fixture.mainStackWeights(), fixture.structureWeights(), fixture.confidenceWeights(),
      fixture.geometryTables(), fixture.queryOnlyFeatureTables(), fixture.tensor("confidencePaeBreaks"),
      // ...undefined on a bundle predating the head. See distogramHeadWeights.
      fixture.distogramHeadWeights(),
    ]);
    return {
      featureTables,
      paeBreaks,
      weights: {
        embedding, template, templateEmbedding, extraStack, mainStack, structure,
        lddt: confidence.lddt, pae: confidence.pae, geometry, distogram,
      },
    };
  })();
  loaded.set(key, pending);
  return withAbort(pending, signal);
}
