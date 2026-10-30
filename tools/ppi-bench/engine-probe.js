// In-page (engine-probe.py): fold fixed [binder, ...target] complexes through the page's own AF3 (web/af3-model.js
// foldAf3, the target templated as bench.js does) on a chosen bundle - "int5" is the page's shipped one, read through
// loadAf3Weights; any other value is a bundle directory read the same way loadAf3Weights reads one - and return each
// fold's benchmark terms (binder pTM, min interface PAE).
window.__probe = async (items, bundle, calls, recycles) => {
  const { foldAf3, loadAf3Weights } = await import("/web/af3-model.js");
  const { getDevice } = await import("/web/model.js");
  const { HttpTensorStore } = await import("/shared/bundles/http-tensor-store.js");
  const { confidenceWeights, trunkWeights } = await import("/shared/af3/weights/weights.js");
  const { diffusionWeights, atomReference, targetFeatureWeights } = await import("/shared/af3/weights/diffusion-weights.js");
  let weights;
  if (bundle === "int5") {
    weights = await loadAf3Weights(() => {}, "af3");
  } else {
    const manifest = await (await fetch(`${bundle}manifest.json`)).json();
    const store = await HttpTensorStore.fromManifest(bundle, manifest);
    store.prefetch();
    weights = { trunk: await trunkWeights(store), diffusion: await diffusionWeights(store), confidence: await confidenceWeights(store),
      atomReference: await atomReference(store), targetFeat: await targetFeatureWeights(store) };
  }
  const out = [];
  for (const item of items) {
    const templates = item.sequences.map((_, i) => ({ text: item.pdb, chainId: "BCDEFG"[i], chain: i + 1 }));
    const r = await foldAf3({ sequence: [item.binder, ...item.sequences].join(":"), templates, weights, device: await getDevice(),
      seed: item.seed, recycles, calls, onStatus: () => {}, onProgress: () => {} });
    const asym = Array.from(r.batch.asymId), n = asym.length, pae = r.confidence.predictedAlignedError;
    const minPae = (a, b) => {
      let m = Infinity;
      for (let i = 0; i < n; i += 1) if (asym[i] === a) for (let j = 0; j < n; j += 1) if (asym[j] === b) m = Math.min(m, pae[i * n + j]);
      return m;
    };
    const coi = asym[0];
    const ipae = Math.min(...[...new Set(asym)].filter((c) => c !== coi).map((c) => (minPae(coi, c) + minPae(c, coi)) / 2));
    out.push({ key: item.key, seed: item.seed, ipae, binderPtm: r.confidence.chainPtm[String(coi)] });
  }
  return out;
};
