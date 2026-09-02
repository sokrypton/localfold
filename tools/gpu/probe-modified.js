/**
 * Is a modified residue the right shape, and is it still attached to the chain?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-modified.js --code=SEP --at=3
 *     node tools/gpu-chrome.mjs tools/gpu/probe-modified.js --code=PTR --at=3 --steps=32
 *
 * 🔴 FEATURISING IT CORRECTLY IS NOT FOLDING IT CORRECTLY, and this repository
 * has been caught by exactly that gap before. The side chains were compressed
 * for months because the only checker reaching the diffusion head built its
 * weight dict by hand instead of through the loader, so the batch was right and
 * the fold was wrong and nothing said so. check_af3_featurise.js now holds a
 * modified residue's BATCH to AF3 array by array; nothing holds its STRUCTURE
 * to anything. This does.
 *
 * Three questions, because a modified residue can fail in three ways:
 *
 *   1. Its own chemistry - are its bonds the lengths the dictionary says?
 *      This is the side-chain failure's shape: everything short at once.
 *   2. Is it ATTACHED? Its ten tokens carry one atom each and its peptide
 *      bonds to its neighbours are implicit in residue_index rather than in
 *      the bond matrix, so a residue that drifted off the chain would still
 *      featurise perfectly. A C-N of 1.33 A says it is bonded; 8 A says the
 *      model put it somewhere else entirely.
 *   3. Do the residues AROUND it still fold normally? A wrong token layout
 *      would disturb its neighbours as much as itself, and comparing the two
 *      separates "this residue is wrong" from "this fold is wrong".
 *
 * 🔴 PAIRS ARE MATCHED BY ATOM NAME, out of the batch's own
 * ref_atom_name_chars, for the reason probe-sidechains.js gives: indexing both
 * sides by position assumes the labelling is right, which is one of the things
 * being asked.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch, atomName } from "../../src/af3/fold.js";
import { ccdUrl, parseCcdComponent, polymerResidue } from "../../src/af3/ccd-component.js";
import { REFERENCE_CONFORMERS } from "../../src/af3/reference-conformers.js";
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
  const code = option(args, "code", "SEP").toUpperCase();
  const at = Number(option(args, "at", "3"));
  const sequence = option(args, "sequence", "ACSEFGHIKLWY");
  const mode = option(args, "mode", "flow");
  const steps = Number(option(args, "steps", "8"));

  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"));
  const weights = {
    trunk: await trunkWeights(store), diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store), atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  const response = await fetch(ccdUrl(code));
  if (!response.ok) throw new Error(`could not fetch ${code}: ${response.status}`);
  const component = parseCcdComponent(await response.text());
  const batch = featuriseProtein(sequence, {
    modifications: [{ chain: 0, position: at, ...component }],
  });
  const result = await foldBatch(device, batch, weights, {
    mode, steps, recycles: Number(option(args, "recycles", "3")), seed: 1,
  });
  const positions = result.positions;
  const { dense } = batch;

  const coordOf = (slot) => [positions[slot * 3], positions[slot * 3 + 1], positions[slot * 3 + 2]];
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const nameOf = (slot) => atomName(batch.refAtomNameChars, slot);
  /** name -> flat slot, for one token that holds a whole residue. */
  const slotsOfResidueToken = (token) => {
    const found = new Map();
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (batch.predDenseAtomMask[slot]) found.set(nameOf(slot), slot);
    }
    return found;
  };

  const span = batch.modifiedSpans[0];
  if (span === undefined) throw new Error("no modified residue in the batch");
  // Each of its tokens holds one atom, in slot zero.
  const modifiedSlots = new Map();
  for (let index = 0; index < span.count; index += 1) {
    const slot = (span.from + index) * dense;
    modifiedSlots.set(nameOf(slot), slot);
  }

  // --- 1. the modified residue's own chemistry, against the dictionary -------
  const ideal = polymerResidue(component, at === sequence.length);
  const idealAt = new Map(ideal.atoms.map((atom, index) => [atom.name, index]));
  const idealCoord = (name) => {
    const atom = ideal.atoms[idealAt.get(name)];
    return [atom.x, atom.y, atom.z];
  };
  const ownBonds = [];
  for (const bond of ideal.bonds) {
    const from = ideal.atoms[bond.from].name;
    const to = ideal.atoms[bond.to].name;
    const a = modifiedSlots.get(from);
    const b = modifiedSlots.get(to);
    if (a === undefined || b === undefined) continue;
    const want = distance(idealCoord(from), idealCoord(to));
    const got = distance(coordOf(a), coordOf(b));
    ownBonds.push({ pair: `${from}-${to}`, ideal: Number(want.toFixed(3)),
                    predicted: Number(got.toFixed(3)), ratio: Number((got / want).toFixed(3)) });
  }

  // --- 2. is it attached? the peptide bonds either side ----------------------
  // 🔴 THESE ARE THE BONDS NOTHING IN THE BATCH ASSERTS. AF3 leaves backbone
  // connectivity implicit in residue_index, so a modified residue that the
  // model placed in the next postcode would featurise exactly as one that is
  // bonded. 1.33 A is a peptide bond.
  const peptide = [];
  const tokenOfResidue = new Map();
  for (let token = 0; token < batch.tokens; token += 1) {
    const residue = batch.residueOfToken[token];
    if (residue >= 0 && !tokenOfResidue.has(residue)) tokenOfResidue.set(residue, token);
  }
  const link = (fromResidue, fromName, toResidue, toName) => {
    if (fromResidue < 0 || toResidue >= sequence.length) return;
    const source = fromResidue === at - 1
      ? modifiedSlots : slotsOfResidueToken(tokenOfResidue.get(fromResidue));
    const target = toResidue === at - 1
      ? modifiedSlots : slotsOfResidueToken(tokenOfResidue.get(toResidue));
    const a = source.get(fromName);
    const b = target.get(toName);
    if (a === undefined || b === undefined) return;
    peptide.push({
      bond: `${sequence[fromResidue]}${fromResidue + 1} ${fromName}`
        + ` - ${sequence[toResidue]}${toResidue + 1} ${toName}`,
      length: Number(distance(coordOf(a), coordOf(b)).toFixed(3)),
    });
  };
  link(at - 2, "C", at - 1, "N");        // the residue before, into this one
  link(at - 1, "C", at, "N");            // this one, into the residue after

  // --- 3. the unmodified residues, as a control -----------------------------
  const controlRatios = [];
  for (let residue = 0; residue < sequence.length; residue += 1) {
    if (residue === at - 1) continue;
    const conformer = REFERENCE_CONFORMERS[sequence[residue]];
    if (conformer === undefined) continue;
    const slots = slotsOfResidueToken(tokenOfResidue.get(residue));
    const atoms = residue === sequence.length - 1 ? conformer.cTerminal : conformer.internal;
    const byName = new Map(atoms.map((atom) => [atom[1], atom]));
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

  return {
    code, at, sequence, mode, steps,
    tokens: batch.tokens,
    meanPlddt: Number(result.meanPlddt?.toFixed?.(1) ?? result.meanPlddt),
    // 🔴 THE MEDIAN RATIO IS THE HEADLINE. The side-chain failure showed as
    // 0.927 where AF3 itself scores 1.017; anything near 1 is the chemistry
    // the dictionary asked for.
    modifiedBondRatio: Number((median(ownBonds.map((b) => b.ratio)) ?? 0).toFixed(3)),
    controlBondRatio: Number((median(controlRatios) ?? 0).toFixed(3)),
    peptideBonds: peptide,
    ownBonds,
  };
}
