/**
 * Are the side chains the right shape, and if not, which bonds are wrong?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-sidechains.js --steps=8
 *     node tools/gpu-chrome.mjs tools/gpu/probe-sidechains.js --mode=diffusion --steps=160
 *
 * Every residue type has a table of pairs the chemistry holds fixed - bonds and
 * 1-3 angle distances - in reference-conformers.js. The structure module is free
 * to choose torsions and nothing else, so a predicted residue must reproduce
 * those distances whatever it does with the rotamer. This measures that, per
 * residue type and per named pair.
 *
 * 🔴 PAIRS ARE MATCHED BY ATOM NAME, NOT BY SLOT INDEX. If the question is
 * whether atoms are mislabelled, then indexing both sides by position assumes
 * the answer: two swapped labels would compare a bond against the ideal of the
 * bond it was swapped with, and could look perfect. Reading the name out of the
 * batch's own ref_atom_name_chars and looking the ideal up by that name means a
 * mismatch shows up as a wrong distance rather than hiding as a right one.
 *
 * WHAT IT FOUND, on PIAQ...ASK (59 residues) against AF3's own 200-step sample
 * of the same sequence, as the median ratio of predicted to ideal distance:
 *
 *                    bond    1-3    PHE50 ring bonds
 *     AF3 itself     1.017  1.015   1.407 1.404 1.404 1.405 1.409 1.408
 *     this port      0.927  0.908   1.122 1.099 1.287 1.198 1.164 1.303
 *
 * 🔴 THE CAUSE WAS A WEIGHT NAME, AND THIS PROBE IS WHAT MADE IT VISIBLE.
 * src/af3/diffusion-weights.js loaded four of the atom encoder's pair tensors
 * under their unsuffixed names, which exist at identical shapes and belong to a
 * different module - see AF3.md, "Fixed: the side chains were compressed". The
 * port now scores 1.015 / 1.017 with textbook aromatic rings. The numbers above
 * are kept because they are what the failure LOOKED like: everything short,
 * worse with distance from the backbone, glycine nearly right, and no
 * improvement with more steps - a shape a scale factor cannot make and
 * under-convergence cannot either.
 *
 * 🔴 AND IT IS NOT ATOM LABELLING, which is why the pairs here are matched by
 * NAME out of the batch's own ref_atom_name_chars. A swap would show as a
 * bimodal pattern - one bond at another bond's ideal - rather than as
 * everything short at once.
 *
 * 🔴 A RING IS REPORTED WHOLE. An aromatic ring's six bonds are all about
 * 1.39 A, so a single worst-pair number cannot distinguish "the ring is a
 * little large" from "two atoms are swapped and it is a bowtie". The ring pairs
 * are named and printed together for the aromatic types.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { atomName, foldBatch } from "../../src/af3/fold.js";
import { REFERENCE_CONFORMERS } from "../../src/af3/reference-conformers.js";
import { confidenceWeights, openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** Every aromatic type, and the ring atoms whose closure is the question. */
const RINGS = {
  F: ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
  Y: ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
  H: ["CG", "ND1", "CE1", "NE2", "CD2"],
  W: ["CG", "CD1", "NE1", "CE2", "CD2"],
  P: ["N", "CA", "CB", "CG", "CD"],
};

export async function main(device, args) {
  const sequence = option(args, "sequence",
    // one of everything that has a ring, with a few plain residues between
    "GFAYAWAHAPGFYWHPAGLKESTNQDRIVMC");
  const mode = option(args, "mode", "flow");
  const steps = Number(option(args, "steps", "8"));

  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"),
                                   { fetchImplementation: fetch });
  const weights = {
    trunk: await trunkWeights(store), diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store), atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  const batch = featuriseProtein(sequence, {});
  const result = await foldBatch(device, batch, weights, {
    mode, steps, recycles: Number(option(args, "recycles", "3")), seed: 1,
  });
  const positions = result.positions;
  const { dense } = batch;

  // slot -> name, straight out of the batch the model was given
  const nameOf = (token, atom) => atomName(batch.refAtomNameChars, token * dense + atom);
  const coordOf = (token, atom) => {
    const slot = (token * dense + atom) * 3;
    return [positions[slot], positions[slot + 1], positions[slot + 2]];
  };
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

  const ratios = [];
  const perPair = new Map();     // "F CG-CD1" -> {errors: []}
  const perType = new Map();     // "F" -> {errors: []}
  const ringReport = [];

  for (let token = 0; token < sequence.length; token += 1) {
    const code = sequence[token];
    const conformer = REFERENCE_CONFORMERS[code];
    if (conformer === undefined) continue;
    // name -> dense slot, for this residue as the model actually holds it
    const slotByName = new Map();
    for (let atom = 0; atom < dense; atom += 1) {
      if (!batch.predDenseAtomMask[token * dense + atom]) continue;
      slotByName.set(nameOf(token, atom), atom);
    }
    const idealName = new Map(conformer.internal.map(([index, name]) => [index, name]));
    for (const [i, j, ideal] of conformer.rigid) {
      const a = slotByName.get(idealName.get(i));
      const b = slotByName.get(idealName.get(j));
      if (a === undefined || b === undefined) continue;
      const predicted = distance(coordOf(token, a), coordOf(token, b));
      // 🔴 THE RATIO, NOT ONLY THE ERROR. A uniform scale problem and a
      // placement problem both raise the error; only the ratio tells them
      // apart, and only split by BOND against 1-3 distance does it say whether
      // whatever is wrong is local to a bond or spread over the residue.
      ratios.push({ kind: ideal < 1.8 ? "bond" : "1-3", ratio: predicted / ideal });
      const error = Math.abs(predicted - ideal);
      const key = `${code} ${idealName.get(i)}-${idealName.get(j)}`;
      if (!perPair.has(key)) perPair.set(key, []);
      perPair.get(key).push(error);
      if (!perType.has(code)) perType.set(code, []);
      perType.get(code).push(error);
    }
    // ...and the ring closure, bond by bond around it
    const ring = RINGS[code];
    if (ring !== undefined && ring.every((name) => slotByName.has(name))) {
      const bonds = ring.map((name, index) => {
        const next = ring[(index + 1) % ring.length];
        return {
          bond: `${name}-${next}`,
          length: Number(distance(coordOf(token, slotByName.get(name)),
                                  coordOf(token, slotByName.get(next))).toFixed(3)),
        };
      });
      ringReport.push({ residue: `${code}${token + 1}`, bonds });
    }
  }

  const summarise = (errors) => ({
    n: errors.length,
    mean: Number((errors.reduce((a, b) => a + b, 0) / errors.length).toFixed(3)),
    max: Number(Math.max(...errors).toFixed(3)),
  });
  return {
    sequence, mode, steps,
    meanPlddt: Number(result.meanPlddt.toFixed(1)),
    caca: Number(result.geometry.caca.toFixed(2)),
    scale: ["bond", "1-3"].map((kind) => {
      const values = ratios.filter((r) => r.kind === kind).map((r) => r.ratio).sort((a, b) => a - b);
      return { kind, n: values.length,
        median: Number(values[Math.floor(values.length / 2)].toFixed(3)),
        mean: Number((values.reduce((a, b) => a + b, 0) / values.length).toFixed(3)) };
    }),
    byType: Object.fromEntries([...perType].map(([k, v]) => [k, summarise(v)])),
    worstPairs: [...perPair]
      .map(([key, errors]) => ({ pair: key, ...summarise(errors) }))
      .sort((a, b) => b.max - a.max).slice(0, 12),
    rings: ringReport.slice(0, 6),
  };
}
