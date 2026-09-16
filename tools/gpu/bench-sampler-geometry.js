/**
 * Which sampler keeps a molecule's GEOMETRY - across ligands and nucleic acids,
 * not just protein?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-sampler-geometry.js \
 *       --model=/model-af3-int5/manifest.json --seeds=1,2
 *
 * 🔴 GEOMETRY, AND DELIBERATELY NOT RMSD. A fold score needs a deposited
 * structure, so it confines a sweep to targets that have one and then reports a
 * number dominated by where the chain sits; the samplers differ by tenths of an
 * angstrom there and by seed noise of the same size, so it cannot separate
 * them. Bonds can: they are local, they have a right answer for ANY sequence,
 * and this repository has already measured a glycerol torn apart at 6.97 A
 * against a 1.43 ideal while pLDDT read 92.38 and RMSD said the fold was fine.
 *
 * 🔴 AND THE IDEALS COME FROM THE CCD, WHICH IS WHAT MAKES THE SET POSSIBLE.
 * `reference-conformers.json` is twenty amino acids and an X, so before
 * `componentBonds` a ligand needed its five bonds typed into two files by hand
 * and a nucleotide could not be scored by anything here at all. The dictionary
 * states bonds rather than inferring them from a distance cutoff, and it is the
 * same source the featuriser reads - not the prediction, so a fold is not
 * scored against itself.
 *
 * 🔴 AND rosettafold3 HAS NO FLOW SAMPLER. `noFlowSampler` makes `foldBatch`
 * throw rather than switch silently, so its flow arms are SKIPPED and said to
 * be skipped - a blank cell is not a pass.
 */
import { confidenceWeights, openAf3Store, trunkWeights }
  from "../../src/af3/weights/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/weights/diffusion-weights.js";
import { dialectFor, featuriserDialect } from "../../src/af3/dialect.js";
import { featuriseProtein } from "../../src/af3/featurise/featurise.js";
import { foldBatch, toPdb } from "../../src/af3/fold.js";
import { ccdUrl, parseCcdComponent } from "../../src/af3/featurise/ccd-component.js";
import { bondGeometry } from "./bond-geometry.js";

const PROTEIN = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE";

// 🔴 EVERY SYSTEM THE PAGE CAN BE ASKED FOR, not three proteins. The nucleic
// and ligand rows are the ones no sampler comparison here has ever covered.
const SET = [
  { name: "protein", sequence: PROTEIN },
  { name: "protein+GOL", sequence: PROTEIN, ligands: ["GOL"] },
  { name: "protein+ATP", sequence: PROTEIN, ligands: ["ATP"] },
  { name: "protein+SEP", sequence: PROTEIN, ptms: [["SEP", 3]] },
  { name: "rna", chains: ["ACGUACGUACGUACGU"], kinds: ["rna"] },
  { name: "dna", chains: ["ACGTACGTACGTACGT"], kinds: ["dna"] },
  { name: "protein+dna", chains: [PROTEIN, "ACGTACGTACGT"], kinds: ["protein", "dna"] },
];

// mode and its count. The page prefers diffusion 25 and flow 16, and those are
// the two a visitor actually gets; the others say how much of any difference is
// the sampler and how much is the truncation.
const ARMS = [["diffusion", 25], ["diffusion", 200], ["flow", 16], ["flow", 32]];

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const round = (value) => (value === null ? null : Number(value.toFixed(4)));

export async function main(device, args) {
  const seeds = option(args, "seeds", "1,2").split(",").map(Number);
  const only = option(args, "targets", "").split(",").filter(Boolean);
  const armsWanted = option(args, "arms", "").split(",").filter(Boolean);
  const store = await openAf3Store(
    option(args, "model", "/model-af3-int5/manifest.json"));
  const conformers = await (await fetch("/tools/oracle/reference-conformers.json")).json();
  const weights = {
    trunk: await trunkWeights(store, undefined, undefined, { allowPrefix: true }),
    diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };
  const dialect = weights.trunk?.dialect ?? dialectFor(weights);
  const noFlow = dialect.noFlowSampler === true;

  // Every component the set needs, fetched once: the ligands, and the four
  // bases of whichever nucleic chains are in it.
  const components = new Map();
  const need = new Set();
  for (const target of SET) {
    for (const code of target.ligands ?? []) need.add(code);
    for (const [code] of target.ptms ?? []) need.add(code);
    for (const kind of target.kinds ?? []) {
      if (kind === "rna") for (const c of ["A", "C", "G", "U"]) need.add(c);
      if (kind === "dna") for (const c of ["DA", "DC", "DG", "DT"]) need.add(c);
    }
  }
  for (const code of need) {
    components.set(code, parseCcdComponent(await (await fetch(ccdUrl(code))).text()));
  }

  const rows = [];
  for (const target of SET) {
    if (only.length > 0 && !only.includes(target.name)) continue;
    const ligands = target.ligands === undefined ? undefined
      : target.ligands.map((code) => components.get(code));
    // 🔴 CHAINS ARE COLON-JOINED, not an array - `featuriseProtein` splits a
    // string, and passing the array died in `sequence.split is not a function`.
    const batch = featuriseProtein((target.chains ?? [target.sequence]).join(":"), {
      ...featuriserDialect(dialect),
      ...(target.kinds ? { chainKinds: target.kinds } : {}),
      ...(ligands ? { ligands } : {}),
      // A modification is `{ chain, position, ...component }` - the component
      // SPREAD, not nested, which is what probe-modified.js and
      // check-batch-fields.js both pass.
      ...(target.ptms ? { modifications: target.ptms.map(([code, at]) =>
        ({ chain: 0, position: at, ...components.get(code) })) } : {}),
    });
    for (const [mode, steps] of ARMS) {
      if (armsWanted.length > 0 && !armsWanted.includes(`${mode}${steps}`)) continue;
      if (mode === "flow" && noFlow) {
        rows.push({ target: target.name, arm: `${mode}${steps}`, skipped: "noFlowSampler" });
        continue;
      }
      for (const seed of seeds) {
        const result = await foldBatch(device, batch, weights,
          { mode, steps, recycles: 0, seed });
        const scored = bondGeometry(toPdb(batch, result.positions), conformers,
          { components });
        rows.push({
          target: target.name, arm: `${mode}${steps}`, seed,
          plddt: Number((result.meanPlddt ?? 0).toFixed(2)),
          mainchain: round(scored.mainchain.rms), sidechain: round(scored.sidechain.rms),
          peptide: round(scored.peptide.rms), nucleic: round(scored.nucleic.rms),
          ligand: round(scored.ligand.rms),
          bonds: scored.all.bonds, worst: scored.worst[0],
        });
      }
    }
  }
  return { model: option(args, "model", ""), noFlow, rows };
}
