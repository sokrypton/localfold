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
import { bondGeometry, parsePdbResidues } from "./bond-geometry.js";
// 🔴 A NON-CHAIN IS NOT A BOND-LENGTH QUESTION and would pass every column
// above: this repository has measured intellifold2 in flow returning a fold
// the chain rule REFUSES on 1 seed in 6 while pLDDT read 83.30. A default
// cannot be chosen without counting those.
import { chainGeometryOf, chainGeometryVerdict } from "./chain-geometry.js";

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

/**
 * Matched budgets: the same number of denoiser calls to each sampler.
 *
 * 🔴 THE PAGE'S TWO SETTINGS ARE NOT A FAIR COMPARISON AND CANNOT TEST THE
 * CLAIM FLOW WAS ADDED FOR. Flow is offered at 16 cycles and diffusion at 25
 * steps, so "flow16 beats diffusion25" confounds the sampler with a 1.6x
 * budget - and the hope was that flow needs FEWER calls than diffusion to get a
 * small molecule right, which is a statement about the CURVE and not about one
 * pair of points. A denoiser call costs the same either way, so equal calls is
 * the honest axis and `--budgets=8,16,32,64` is how to ask.
 */
const matchedBudgets = (counts) =>
  counts.flatMap((n) => [["diffusion", n], ["flow", n]]);

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const round = (value) => (value === null ? null : Number(value.toFixed(4)));

export async function main(device, args) {
  const seeds = option(args, "seeds", "1,2").split(",").map(Number);
  const only = option(args, "targets", "").split(",").filter(Boolean);
  // 🔴 `--ligands=` REPLACES THE SET WITH ONE ROW PER SMALL MOLECULE, because
  // two of them cannot answer "is a sampler better for ligands": the full set's
  // ligand column came out 6-6 over GOL and ATP alone, which is four coin
  // flips. The codes are CCD codes and are fetched, so any molecule with a
  // dictionary entry can be in the sweep.
  const ligandSweep = option(args, "ligands", "").split(",").filter(Boolean);
  if (ligandSweep.length > 0) {
    SET.length = 0;
    for (const code of ligandSweep) {
      SET.push({ name: `protein+${code}`, sequence: PROTEIN, ligands: [code] });
    }
  }
  const armsWanted = option(args, "arms", "").split(",").filter(Boolean);
  const budgets = option(args, "budgets", "").split(",").filter(Boolean).map(Number);
  if (budgets.length > 0) {
    ARMS.length = 0;
    for (const arm of matchedBudgets(budgets)) ARMS.push(arm);
    // ...and the converged reference last, so a curve has a floor to be read
    // against rather than only against the other sampler.
    ARMS.push(["diffusion", 200]);
  }
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
        const pdb = toPdb(batch, result.positions);
        const scored = bondGeometry(pdb, conformers, { components });
        // 🔴 `chainGeometryOf` TAKES SPACINGS, NOT A PDB - it is the shared rule
        // for a tool that already has the distances. Passing the text returned
        // `caca: null` and a cheerful `notAChain: false`, which is the "a gate
        // that cannot fail" shape: a wrong argument read as a clean fold.
        const alphas = parsePdbResidues(pdb)
          .map((r) => r.atoms.get("CA")).filter(Boolean);
        const spacings = alphas.slice(1).map((p2, i) =>
          Math.hypot(p2[0] - alphas[i][0], p2[1] - alphas[i][1], p2[2] - alphas[i][2]));
        const chain = chainGeometryOf(spacings);
        const verdict = chainGeometryVerdict(chain);
        rows.push({
          target: target.name, arm: `${mode}${steps}`, seed,
          plddt: Number((result.meanPlddt ?? 0).toFixed(2)),
          mainchain: round(scored.mainchain.rms), sidechain: round(scored.sidechain.rms),
          peptide: round(scored.peptide.rms), nucleic: round(scored.nucleic.rms),
          ligand: round(scored.ligand.rms),
          bonds: scored.all.bonds, worst: scored.worst[0],
          caca: Number((chain?.caca ?? NaN).toFixed(3)),
          notAChain: verdict?.ok === false,
        });
      }
    }
  }
  return { model: option(args, "model", ""), noFlow, rows };
}
