/**
 * Sample from AF3's OWN trunk, and see whether the side chains come out right.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-af3-trunk-sample.js \
 *       --dump=/af3-sample200.json --steps=200
 *
 * THE QUESTION. probe-sidechains.js found our side chains about 8% compressed
 * where AF3's are ideal, and the trunk on AF3's own batch disagrees with AF3's
 * by 3.7e-2 relative - which is near the tolerance the conformer difference is
 * worth, but is not zero. Those two facts leave the blame unassigned: a wrong
 * pair representation would give the sampler a wrong answer to converge to, and
 * a wrong sampler would spoil a right one.
 *
 * 🔴 SO SUBSTITUTE THE TRUNK RATHER THAN COMPARE IT. This feeds AF3's captured
 * `single`, `pair` and `target_feat` straight into our diffusion head through
 * foldBatch's `reuse` path, skipping our trunk entirely. What comes out is
 * OUR sampler's answer to AF3's OWN question, so the two possibilities separate:
 * geometry that is still compressed is the head's or the sampler's, and
 * geometry that comes out ideal means the trunk error is what matters and the
 * head is fine.
 *
 * 🔴 THE RECYCLE COUNT MUST MATCH THE DUMP'S. `reuse.recycles` is set to the
 * requested count so foldBatch's loop does not run at all; the captured trunk
 * is the LAST recycle's, and pairing it with a different number would be
 * comparing against a pass AF3 never took.
 */
import { foldBatch, backboneGeometry } from "../../src/af3/fold.js";
import { batchFromDump } from "./fold.js";
import { sidechainGeometry } from "./sidechain-geometry.js";
import { confidenceWeights, openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const floats = (source) => Float32Array.from(source, (v) => Number(v));

/** The last of `base`, `base#0`, `base#1`... - a recycled dump numbers them. */
function lastCapture(dump, base) {
  if (dump.outputs[base] !== undefined) return dump.outputs[base];
  return Object.keys(dump.outputs)
    .filter((key) => key.startsWith(`${base}#`))
    .sort((a, b) => Number(a.split("#")[1]) - Number(b.split("#")[1]))
    .map((key) => dump.outputs[key]).pop();
}

/** Straight RMSD, no superposition: only INTERNAL geometry is being compared. */
function atomRmsd(a, b, mask, dense) {
  let sum = 0;
  let count = 0;
  for (let atom = 0; atom < mask.length; atom += 1) {
    if (!mask[atom]) continue;
    count += 1;
    for (let axis = 0; axis < 3; axis += 1) {
      const d = a[atom * 3 + axis] - b[atom * 3 + axis];
      sum += d * d;
    }
  }
  return Math.sqrt(sum / Math.max(count, 1));
}

export async function main(device, args) {
  const dumpPath = option(args, "dump", "/af3-sample200.json");
  const steps = Number(option(args, "steps", "200"));
  const mode = option(args, "mode", "diffusion");
  const recycles = Number(option(args, "recycles", "3"));
  const ownTrunk = args.includes("--own-trunk");

  const response = await fetch(dumpPath);
  if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
  const dump = await response.json();
  const batch = batchFromDump(dump);

  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"),
                                   { fetchImplementation: fetch });
  const weights = {
    trunk: await trunkWeights(store, 48, 4), diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store), atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  const single = lastCapture(dump, "diffuser/evoformer/__call__:single");
  const pair = lastCapture(dump, "diffuser/evoformer/__call__:pair");
  const targetFeat = lastCapture(dump, "diffuser/evoformer/__call__:target_feat");
  const reuse = ownTrunk ? undefined : {
    trunk: { single: floats(single.data), pair: floats(pair.data) },
    targetFeat: floats(targetFeat.data),
    recycles,
  };

  const result = await foldBatch(device, batch, weights, {
    mode, steps, recycles, seed: Number(option(args, "seed", "1")), reuse,
  });

  const rows = {
    dump: dumpPath, sequence: batch.sequence, mode, steps, recycles,
    trunk: ownTrunk ? "ours" : "AF3's own, substituted",
    meanPlddt: Number(result.meanPlddt.toFixed(1)),
    backbone: backboneGeometry(batch, result.positions),
    ...sidechainGeometry(batch, result.positions),
  };

  // ...and AF3's own answer, scored the same way, as the number that says what
  // "right" looks like on this exact molecule.
  const theirs = dump.outputs["diffusion_samples/atom_positions"];
  if (theirs !== undefined) {
    const positions = floats(theirs.data);
    rows.af3 = sidechainGeometry(batch, positions);
    rows.af3.backbone = backboneGeometry(batch, positions);
    rows.rmsdToAf3NoSuperposition =
      Number(atomRmsd(result.positions, positions, batch.predDenseAtomMask, batch.dense)
        .toFixed(2));
  }
  return rows;
}
