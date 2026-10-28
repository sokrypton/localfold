/**
 * A previous fold's coordinates laid out for a new batch, to start AF3's diffusion from.
 *
 * 🔴 WHY: PROTEIN HUNTER HAS NO HISTORY BUT THE BACKBONE MPNN READS (2026-10-28). Each cycle folds
 * its new sequence from pure noise, so nothing but the sequence carries the last structure forward
 * - and a sequence is a weak encoding of it. Starting the diffusion part-way down from the last
 * cycle's structure (the sampler's `start`, webgpu/af3/diffusion/diffusion-sampler.js) is the
 * history: RFdiffusion's partial diffusion, on AF3.
 *
 * 🔴 THE SEQUENCE CHANGED, SO THE ATOMS DID. AF3's dense layout is per residue type: a mutated
 * residue's slots hold atoms the previous structure never had. Matched by the names `toPdb` writes
 * (`pdbSlots`), so both sides are named by one rule:
 *
 *   - an unchanged residue copies every atom it has;
 *   - a mutated one copies its BACKBONE - and CB where both residues have one - and puts every
 *     other atom on its C-alpha (C1' for a nucleotide).
 *
 * The crude half is deliberate and costs nothing: the walk joins at several angstroms of noise,
 * which swamps where a side chain was placed, and the backbone is what carries the fold.
 */
import { pdbSlots } from "./pdb.js";

/** Atoms a residue keeps across a mutation: the protein backbone and CB, and a nucleotide's sugar-phosphate. */
const KEPT_ACROSS_A_MUTATION = new Set(["N", "CA", "C", "O", "OXT", "CB",
  "P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]);

/**
 * @param {object} batch the NEW fold's batch
 * @param {string} pdb the previous fold, as `toPdb` wrote it - same chains, same residue numbers
 * @returns {{positions: Float32Array, copied: number, kept: number, placed: number}} positions in
 *   the batch's dense layout (`tokens * dense * 3`); how many atoms were copied from an unchanged
 *   residue, kept across a mutation, and placed on an anchor
 */
export function startFromPdb(batch, pdb) {
  const atoms = new Map();
  const residues = new Map();
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const residue = `${line[21]}|${Number(line.slice(22, 26))}`;
    const at = [Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54))];
    atoms.set(`${residue}|${line.slice(12, 16).trim()}`, at);
    if (!residues.has(residue)) residues.set(residue, { name: line.slice(17, 20).trim(), first: at });
  }
  const positions = new Float32Array(batch.tokens * batch.dense * 3);
  let copied = 0;
  let kept = 0;
  let placed = 0;
  for (const { slot, atomName, resName, chain, resSeq } of pdbSlots(batch)) {
    const residue = `${chain}|${resSeq}`;
    const before = residues.get(residue);
    // 🔴 A RESIDUE THE PREVIOUS FOLD DID NOT HAVE IS A DIFFERENT JOB, not a mutation: refused, since
    // starting from somewhere else is the one way this can fold the wrong thing quietly
    if (before === undefined) throw new Error(`the previous fold has no residue ${resSeq} of chain ${chain}`);
    const same = before.name === resName;
    let at = (same || KEPT_ACROSS_A_MUTATION.has(atomName)) ? atoms.get(`${residue}|${atomName}`) : undefined;
    if (at !== undefined) {
      if (same) copied += 1; else kept += 1;
    } else {
      at = atoms.get(`${residue}|CA`) ?? atoms.get(`${residue}|C1'`) ?? before.first;
      placed += 1;
    }
    positions.set(at, slot * 3);
  }
  return { positions, copied, kept, placed };
}
