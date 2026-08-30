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
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

/**
 * The tensor store this origin can actually use.
 *
 * Over http the shards are fetched directly. On a file:// page fetch does not
 * work at all, so the weights come in as classic scripts carrying base64 data:
 * URLs - see ScriptTensorStore, and tools/export-js-weights.py, which writes
 * them.
 *
 * `?model=` overrides both, which is how a page is pointed at a manifest
 * somewhere else without editing it.
 */
export function openStore(onProgress) {
  const override = new URLSearchParams(location.search).get("model");
  if (override !== null) return HttpTensorStore.open(override, onProgress);
  if (location.protocol === "file:") return ScriptTensorStore.open("./model/", onProgress);
  return HttpTensorStore.open("./model/manifest.json", onProgress);
}

let devicePromise;

/** The WebGPU device, with the optional features the fast paths look for. */
export function getDevice() {
  devicePromise ??= (async () => {
    if (navigator.gpu === undefined) throw new Error("This browser has no WebGPU. It ships in current Chrome, Edge, Safari and Firefox.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("No compatible WebGPU adapter was found");
    return requestAlphaFoldDevice(adapter);
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
export function loadModel(variant, onProgress) {
  if (variant !== "single" && variant !== "msa") {
    throw new RangeError(`unknown model variant ${variant}: expected "single" or "msa"`);
  }
  const cached = loaded.get(variant);
  if (cached !== undefined) return cached;
  const pending = (async () => {
    const store = await openStore(onProgress);
    const fixture = AlphaFoldFixture.fromStore(store);
    const extraStackWeights = variant === "msa"
      ? fixture.extraStackWeights() : fixture.extraPairStackWeights();
    const [embedding, template, extraStack, mainStack, structure, confidence, geometry,
      featureTables, paeBreaks] = await Promise.all([
      fixture.embeddingWeights(), fixture.templateWeights(), extraStackWeights,
      fixture.mainStackWeights(), fixture.structureWeights(), fixture.confidenceWeights(),
      fixture.geometryTables(), fixture.queryOnlyFeatureTables(), fixture.tensor("confidencePaeBreaks"),
    ]);
    return {
      featureTables,
      paeBreaks,
      weights: {
        embedding, template, extraStack, mainStack, structure,
        lddt: confidence.lddt, pae: confidence.pae, geometry,
      },
    };
  })();
  loaded.set(variant, pending);
  return pending;
}
