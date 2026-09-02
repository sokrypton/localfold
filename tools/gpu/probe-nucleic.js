/**
 * Does a folded DNA or RNA chain have the right chemistry, and is it a chain?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-nucleic.js --dna=ACGTACGT
 *     node tools/gpu-chrome.mjs tools/gpu/probe-nucleic.js --rna=ACGUACGU --steps=32
 *     node tools/gpu-chrome.mjs tools/gpu/probe-nucleic.js --dna=ACGT --sequence=ACDEFGHIKL
 *
 * 🔴 FEATURISING IT CORRECTLY IS NOT FOLDING IT CORRECTLY, which is the gap
 * that hid the compressed side chains for months and the reason probe-modified
 * exists. check_af3_featurise.js holds a nucleic BATCH to AF3 array by array;
 * without this nothing holds its STRUCTURE to anything, and the failure mode of
 * a new token type is a plausible-looking backbone.
 *
 * Three questions, the same three a modified residue gets:
 *
 *   1. Each base's own chemistry, against its baked conformer. This is the
 *      side-chain failure's shape - everything short at once - and the ratio is
 *      the headline.
 *   2. Is it a CHAIN? The phosphodiester bond from one residue's O3' to the
 *      next residue's P is implicit in residue_index, exactly as the peptide
 *      bond is, so a base the model parked elsewhere would featurise perfectly.
 *      1.6 A says it is bonded.
 *   3. Do protein residues in the same job still fold normally? That separates
 *      "the nucleic chain is wrong" from "this fold is wrong".
 *
 * 🔴 PAIRS ARE MATCHED BY ATOM NAME, out of the batch's own ref_atom_name_chars,
 * for the reason probe-sidechains.js gives: indexing both sides by position
 * assumes the labelling is right, which is one of the things being asked. It
 * matters more here than anywhere - a nucleotide has a C4 AND a C4', a C2 and a
 * C2', and the primes are the part a reader drops.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch, atomName } from "../../src/af3/fold.js";
import { REFERENCE_CONFORMERS } from "../../src/af3/reference-conformers.js";
import { nucleicConformers } from "../../src/af3/reference-conformers-nucleic.js";
import { openAf3Store, trunkWeights, confidenceWeights } from "../../src/af3/weights.js";
import { targetFeatureWeights, diffusionWeights, atomReference }
  from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const median = (values) => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};

export async function main(device, args) {
  const dna = option(args, "dna", "");
  const rna = option(args, "rna", "");
  const protein = option(args, "sequence", "");
  const mode = option(args, "mode", "flow");
  const steps = Number(option(args, "steps", "16"));
  if (dna === "" && rna === "") throw new Error("pass --dna= or --rna=");

  // Protein first, because featuriseProtein numbers asym straight through and
  // the polymers come before the ligands; the order here is the order of the
  // kinds below.
  const chains = [];
  const chainKinds = [];
  if (protein !== "") { chains.push(protein.toUpperCase()); chainKinds.push("protein"); }
  if (dna !== "") { chains.push(dna.toUpperCase()); chainKinds.push("dna"); }
  if (rna !== "") { chains.push(rna.toUpperCase()); chainKinds.push("rna"); }

  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"));
  const weights = {
    trunk: await trunkWeights(store), diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store), atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  const batch = featuriseProtein(chains.join(":"), { chainKinds });
  const result = await foldBatch(device, batch, weights, {
    mode, steps, recycles: Number(option(args, "recycles", "3")), seed: 1,
  });
  const positions = result.positions;
  const { dense } = batch;

  const coordOf = (slot) => [positions[slot * 3], positions[slot * 3 + 1], positions[slot * 3 + 2]];
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const nameOf = (slot) => atomName(batch.refAtomNameChars, slot);
  const slotsOfToken = (token) => {
    const found = new Map();
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (batch.predDenseAtomMask[slot]) found.set(nameOf(slot), slot);
    }
    return found;
  };

  const tokenOfResidue = new Map();
  for (let token = 0; token < batch.tokens; token += 1) {
    const residue = batch.residueOfToken[token];
    if (residue >= 0 && !tokenOfResidue.has(residue)) tokenOfResidue.set(residue, token);
  }
  const kindOfResidue = (residue) => chainKinds[batch.chainOfResidue[residue]];
  const residueCount = batch.chainOfResidue.length;

  // --- 1. each base's own chemistry, against its baked conformer -------------
  const ratios = [];
  const worstPairs = [];
  for (let residue = 0; residue < residueCount; residue += 1) {
    const kind = kindOfResidue(residue);
    if (kind === "protein") continue;
    const code = batch.sequence[residue];
    const entry = nucleicConformers(kind)?.[code];
    if (entry === undefined) continue;
    // 🔴 `rigid` INDEXES `internal`, AND THAT IS RIGHT FOR THE 5' FORM TOO.
    // Each atom carries its own dense slot as its first field, and the two
    // forms agree on every slot they share - the 5' form only adds OP3 at slot
    // 0, which no rigid pair names. So reading names and ideal coordinates out
    // of `internal` is correct at both ends of the chain, and picking the form
    // by position would index `rigid` into an array shifted by one.
    const atoms = entry.internal;
    const byName = new Map(atoms.map((atom) => [atom[1], atom]));
    const slots = slotsOfToken(tokenOfResidue.get(residue));
    for (const [i, j] of entry.rigid) {
      const from = atoms[i][1];
      const to = atoms[j][1];
      const a = slots.get(from);
      const b = slots.get(to);
      if (a === undefined || b === undefined) continue;
      const want = distance(byName.get(from).slice(4), byName.get(to).slice(4));
      const got = distance(coordOf(a), coordOf(b));
      ratios.push(got / want);
      worstPairs.push({ where: `${kind} ${code}${residue + 1} ${from}-${to}`,
                        ratio: Number((got / want).toFixed(3)) });
    }
  }

  // --- 2. is it a chain? the phosphodiester bonds ----------------------------
  // 🔴 THE BOND NOTHING IN THE BATCH ASSERTS. AF3 leaves backbone connectivity
  // implicit in residue_index, so a base the model placed in the next postcode
  // would featurise exactly as one that is bonded.
  const backbone = [];
  for (let residue = 1; residue < residueCount; residue += 1) {
    if (kindOfResidue(residue) === "protein") continue;
    if (batch.chainOfResidue[residue] !== batch.chainOfResidue[residue - 1]) continue;
    const before = slotsOfToken(tokenOfResidue.get(residue - 1));
    const here = slotsOfToken(tokenOfResidue.get(residue));
    const a = before.get("O3'");
    const b = here.get("P");
    if (a === undefined || b === undefined) continue;
    backbone.push({
      bond: `${batch.sequence[residue - 1]}${residue} O3' - ${batch.sequence[residue]}${residue + 1} P`,
      length: Number(distance(coordOf(a), coordOf(b)).toFixed(3)),
    });
  }

  // --- 3. the protein residues, as a control --------------------------------
  const controlRatios = [];
  for (let residue = 0; residue < residueCount; residue += 1) {
    if (kindOfResidue(residue) !== "protein") continue;
    const conformer = REFERENCE_CONFORMERS[batch.sequence[residue]];
    if (conformer === undefined) continue;
    const last = residue + 1 >= residueCount
      || batch.chainOfResidue[residue + 1] !== batch.chainOfResidue[residue];
    const atoms = last ? conformer.cTerminal : conformer.internal;
    const byName = new Map(atoms.map((atom) => [atom[1], atom]));
    const slots = slotsOfToken(tokenOfResidue.get(residue));
    for (const [i, j] of conformer.rigid) {
      const from = atoms[i][1];
      const to = atoms[j][1];
      const a = slots.get(from);
      const b = slots.get(to);
      if (a === undefined || b === undefined) continue;
      const want = distance(byName.get(from).slice(4), byName.get(to).slice(4));
      controlRatios.push(distance(coordOf(a), coordOf(b)) / want);
    }
  }

  worstPairs.sort((left, right) => Math.abs(left.ratio - 1) - Math.abs(right.ratio - 1));
  return {
    chains, chainKinds, mode, steps,
    tokens: batch.tokens,
    meanPlddt: Number(result.meanPlddt?.toFixed?.(1) ?? result.meanPlddt),
    // 🔴 THE MEDIAN RATIO IS THE HEADLINE, as in probe-modified: the side-chain
    // failure read 0.927 where AF3 itself reads 1.017. Anything near 1 is the
    // chemistry the conformer table asked for.
    nucleicBondRatio: Number((median(ratios) ?? 0).toFixed(3)),
    proteinBondRatio: Number((median(controlRatios) ?? 0).toFixed(3)),
    nucleicPairs: ratios.length,
    // ~1.6 A each, or the chain is not a chain.
    backboneBonds: backbone,
    worstNucleicPairs: worstPairs.slice(-6).reverse(),
  };
}
