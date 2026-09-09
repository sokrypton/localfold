/**
 * Per-kernel GPU time inside one AF2 TEMPLATE pair block.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/profile-af2-template.js --length=825
 *
 * 🔴 THE TEMPLATE EMBEDDER IS 1.04 s OF A 20.4 s FOLD AT 825 RESIDUES and it
 * runs with NO TEMPLATE: `fold-af2.js` synthesises its alignment, so the
 * geometry is zeros and the stack transforms a constant. Two pair blocks at 64
 * channels should be about a fifth of what a 128-channel Evoformer block's pair
 * half costs, and it is not - so this is the caller the profiler needed.
 */
import { QueryOnlyTemplateGpu } from "../../src/evoformer/template.js";
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const length = Number(option(args, "length", "825"));
  const top = Number(option(args, "top", "20"));
  const { MODEL_BUNDLES, loadManifest } = await import("../../src/reference/manifests/index.js");
  const fixture = AlphaFoldFixture.fromStore(await HttpTensorStore.fromManifest(
    MODEL_BUNDLES.monomer.directory, await loadManifest("monomer")));
  const weights = await fixture.templateWeights();
  const result = await new QueryOnlyTemplateGpu(device).run({
    length, templateChannels: 64, pairChannels: 128,
    pairMask: new Float32Array(length * length).fill(1),
    weights,
    // ...block 1, not 0: the first compiles every pipeline.
    profileBlock: Math.min(1, weights.blockWeights.length - 1),
  });
  const totals = new Map();
  for (const { label, nanoseconds } of result.timestampProfile ?? []) {
    const found = totals.get(label) ?? { label, ms: 0, dispatches: 0 };
    found.ms += nanoseconds / 1e6;
    found.dispatches += 1;
    totals.set(label, found);
  }
  const kernels = [...totals.values()]
    .map((row) => ({ ...row, ms: Number(row.ms.toFixed(3)) }))
    .sort((a, b) => b.ms - a.ms);
  const blockMs = kernels.reduce((sum, row) => sum + row.ms, 0);
  return {
    length, blocks: weights.blockWeights.length,
    wallMs: Number(result.elapsedMilliseconds.toFixed(1)),
    blockMs: Number(blockMs.toFixed(2)),
    kernels: kernels.slice(0, top),
  };
}
