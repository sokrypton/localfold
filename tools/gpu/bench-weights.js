/**
 * How long the AF3 checkpoint takes to reach usable Float32Arrays, and why.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-weights.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-weights.js --model=/model-af3-full-f32/manifest.json
 *
 * 🔴 THE PAGE PAYS THIS ONCE AND THE USER WAITS THROUGH ALL OF IT. It is a
 * quarter of a gigabyte over 26 shards, and none of the fold's own speed is
 * visible until it lands - so it belongs on the same footing as a denoiser
 * call, measured rather than assumed.
 *
 * Reported in three parts, because they have different fixes: fetching the
 * bytes, turning them into the tensors the loaders ask for (which for an int5
 * bundle means dequantising), and the loaders' own assembly.
 */
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { MODEL_BUNDLES, loadManifest } from "../../src/reference/manifests/index.js";
import { confidenceWeights, trunkWeights } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  // --family loads a bundle the page ships (monomer, multimer, af3) through the
  // same compiled-in manifest the page uses; --model takes a directory instead.
  const family = option(args, "family", "");
  const manifestUrl = family === ""
    ? option(args, "model", "/model-af3-int5/manifest.json")
    : MODEL_BUNDLES[family].directory;

  const manifestStart = performance.now();
  const manifest = family === ""
    ? await (await fetch(manifestUrl)).json()
    : await loadManifest(family);
  const manifestMs = performance.now() - manifestStart;

  const tensors = Object.values(manifest.tensors ?? {});
  const shards = new Set(tensors.map((entry) => entry.file));
  const dtypes = {};
  for (const entry of tensors) dtypes[entry.dtype] = (dtypes[entry.dtype] ?? 0) + 1;

  let lastFraction = 0;
  const progress = [];
  const openStart = performance.now();
  const store = await HttpTensorStore.fromManifest(manifestUrl, manifest, (report) => {
    const fraction = report.totalBytes > 0 ? report.loadedBytes / report.totalBytes : 0;
    if (fraction - lastFraction < 0.25) return;
    lastFraction = fraction;
    progress.push({ fraction: Number(fraction.toFixed(2)),
                    megabytes: Number((report.loadedBytes / 1e6).toFixed(0)),
                    ms: Math.round(performance.now() - openStart) });
  });
  const openMs = performance.now() - openStart;

  const timed = async (label, work) => {
    const started = performance.now();
    await work();
    return [label, Math.round(performance.now() - started)];
  };
  // 🔴 AF2's LOADERS ARE THE FIXTURE'S, NOT A WEIGHTS MODULE, so for those
  // families this times decoding EVERY tensor in the bundle instead - which is
  // what the page ends up doing and is the number the dequantiser moves.
  const loaders = family === "" || family === "af3"
    ? Object.fromEntries([
      await timed("trunk", () => trunkWeights(store, 48, 4)),
      await timed("diffusion", () => diffusionWeights(store)),
      await timed("confidence", () => confidenceWeights(store)),
      await timed("atomReference", () => atomReference(store)),
      await timed("targetFeat", () => targetFeatureWeights(store)),
    ])
    : Object.fromEntries([
      await timed("all tensors", async () => {
        for (const name of Object.keys(manifest.tensors)) await store.tensor(name);
      }),
    ]);

  const bytes = tensors.reduce((sum, entry) => sum + (entry.byteLength ?? 0), 0);
  return {
    manifestUrl, tensors: tensors.length, shards: shards.size, dtypes,
    manifestMs: Math.round(manifestMs),
    fetchAndDecodeMs: Math.round(openMs),
    loaders,
    totalMs: Math.round(manifestMs + openMs
      + Object.values(loaders).reduce((a, b) => a + b, 0)),
    progress: progress.slice(0, 6),
    ...(bytes > 0 ? { megabytes: Number((bytes / 1e6).toFixed(1)) } : {}),
  };
}
