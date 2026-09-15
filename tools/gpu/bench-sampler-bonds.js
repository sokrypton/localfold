/**
 * Which sampler setting keeps a structure's BONDS, on proteins with little or
 * no alignment?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-sampler-bonds.js \
 *       --model=/model-rosettafold3-int5/manifest.json \
 *       --sigmas=160,40,16,8 --steps=8 --seeds=1,2,3
 *
 * 🔴 BOND GEOMETRY, NOT RMSD, AND THAT IS THE POINT. A sampler that has lost
 * local chemistry shows it in bonds long before it shows it in a fold score,
 * and it can show it while pLDDT is unmoved: rosettafold3's flow walk reads
 * 81.4 whether it returns a chain or a 2.2 A ball. RMSD also needs a deposited
 * structure, which confines a sweep to targets that have one; bonds are scored
 * against the reference conformer this port already folds against, so ANY
 * sequence can be in the set. See bond-geometry.js.
 *
 * 🔴 AND THE SET IS SINGLE-SEQUENCE ON PURPOSE. Flow exists to be the fast
 * option, and the fast option is what a visitor picks when there is no
 * alignment to wait for - a designed protein, an orphan, a binder that does not
 * exist yet. A sweep on deep-MSA targets measures the easy case and calls it
 * safe. `--msa` searches instead, for the comparison.
 *
 * 🔴 AND A LIGAND IS IN THE SET BECAUSE IT HAS NO BACKBONE TO HIDE BEHIND.
 * `chain-geometry.js` steps over a ligand by design and a protein's RMSD is
 * dominated by its 68 residues, so a glycerol torn apart at 6.97 A scored 92.38
 * pLDDT and passed everything this repository had.
 */
import { confidenceWeights, openAf3Store, trunkWeights }
  from "../../src/af3/weights/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/weights/diffusion-weights.js";
import { dialectFor, featuriserDialect } from "../../src/af3/dialect.js";
import { featuriseProtein } from "../../src/af3/featurise/featurise.js";
import { af3BatchFromA3m } from "../../src/af3/featurise/batch.js";
import { generateMmseqs2Msa } from "../../src/input/mmseqs2-api.js";
import { foldBatch, toPdb } from "../../src/af3/fold.js";
import { ccdUrl, parseCcdComponent } from "../../src/af3/featurise/ccd-component.js";
import { bondGeometry } from "./bond-geometry.js";

const SIGMA_DATA = 16;

/**
 * 🔴 LITTLE OR NO ALIGNMENT IS THE SELECTION CRITERION, and these are named
 * rather than searched for it: designed proteins and de novo folds have few
 * natural homologs by construction. Measured through the MMseqs2 API from this
 * box: 6MRR returns 3 rows and 1QYS 6, against 5CAJ's 7907 and 1BRS's 8341.
 * The two naturals are here as the CONTRAST, not as the subject.
 */
const SET = [
  { name: "6MRR", rows: 3, note: "designed mini-protein",
    sequence: "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE" },
  { name: "1QYS", rows: 6, note: "Top7, de novo",
    sequence: "DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLE" },
  { name: "6MRR+GOL", rows: 3, note: "the same fold carrying a glycerol",
    sequence: "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE",
    ligands: ["GOL"] },
];

// GLYCEROL's five bonds and their ideal lengths, the same table
// tools/check-ligand-path.mjs asserts on - a ligand has no reference conformer,
// and inferring its bonds from the PREDICTION would score the fold against
// itself and pass anything.
const GOL_BONDS = [["C1", "O1", 1.43], ["C1", "C2", 1.52], ["C2", "O2", 1.43],
                   ["C2", "C3", 1.52], ["C3", "O3", 1.43]];

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};

export async function main(device, args = []) {
  const sigmas = option(args, "sigmas", "160,40,16,8").split(",").map(Number);
  const steps = Number(option(args, "steps", "8"));
  const seeds = option(args, "seeds", "1,2,3").split(",").map(Number);
  const mode = option(args, "mode", "flow");
  const withMsa = args.includes("--msa");
  const only = option(args, "targets", "").split(",").filter(Boolean);
  const conformers = await (await fetch("/tools/oracle/reference-conformers.json")).json();

  const store = await openAf3Store(
    option(args, "model", "/model-af3-int5/manifest.json"));
  const weights = {
    trunk: await trunkWeights(store), diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store), atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };
  const dialect = weights.trunk?.dialect ?? dialectFor(weights);

  const rows = [];
  for (const target of SET) {
    if (only.length > 0 && !only.includes(target.name)) continue;
    const alignment = withMsa ? await generateMmseqs2Msa(target.sequence) : null;
    const text = alignment === null ? null
      : (typeof alignment === "string" ? alignment : alignment.a3m ?? alignment.msa);
    // A ligand reaches the featuriser as a resolved CCD COMPONENT, not a code -
    // `featuriseProtein` sizes its arrays from `ligand.atoms.length`.
    const ligands = target.ligands === undefined ? undefined
      : await Promise.all(target.ligands.map(async (code) =>
        parseCcdComponent(await (await fetch(ccdUrl(code))).text())));
    const batch = text === null
      ? featuriseProtein(target.sequence,
        { ...featuriserDialect(dialect), ...(ligands ? { ligands } : {}) })
      : af3BatchFromA3m(target.sequence, text,
        { ...featuriserDialect(dialect), ...(ligands ? { ligands } : {}) }).batch;

    for (const sigma0 of sigmas) {
      for (const seed of seeds) {
        const result = await foldBatch(device, batch, weights, {
          mode, steps, recycles: 0, seed,
          schedule: { sigmaMax: sigma0 / SIGMA_DATA },
        });
        const scored = bondGeometry(toPdb(batch, result.positions), conformers,
          { ligandBonds: ligands ? GOL_BONDS : [] });
        rows.push({
          target: target.name, rows: target.rows, mode, sigma0, seed,
          msa: withMsa, plddt: Number((result.meanPlddt ?? 0).toFixed(2)),
          mainchain: round(scored.mainchain.rms), sidechain: round(scored.sidechain.rms),
          peptide: round(scored.peptide.rms), ligand: round(scored.ligand.rms),
          worst: scored.worst[0],
        });
      }
    }
  }
  return rows;
}

const round = (v) => v === null ? null : Number(v.toFixed(4));
