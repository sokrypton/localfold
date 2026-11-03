/**
 * A BioEmu2 sample as a PDB, from the featuriser's atom keys and topology.
 *
 * BioEmu2 predicts its OWN atoms - the CCD's heavy atoms, every polymer residue
 * without its OXT or OP3 - so a sample is written in that order, one record per
 * real atom, with the residue named as the vendor's topology names it (`UNK`
 * for an unknown residue, the component's code for a ligand or a modified one).
 */
import { ELEMENT_SYMBOLS } from "../af3/featurise/ccd-component.js";
import { pdbLinkRecords } from "../af3/structure/pdb.js";

const STANDARD = new Set(["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
  "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK", "A", "C", "G", "U", "N", "DA", "DC", "DG", "DT", "DN"]);

/**
 * Records for one sample's real atoms (`coords` the featuriser's order, padded or not), and a LINK record for each of
 * the job's declared bonds (`links`, its `bondedAtomPairs`: a crosslink's two, a covalent ligand's), which is what the
 * viewer draws the stick between a side chain and a ligand from (see shared/af3/structure/pdb.js).
 */
export function bioemu2Pdb(atomKeys, topology, coords, { bFactors, links } = {}) {
  const lines = [];
  let serial = 1;
  let previousChain = null;
  for (let a = 0; a < atomKeys.length; a += 1) {
    const [chain, residue, name] = atomKeys[a];
    if (previousChain !== null && chain !== previousChain) lines.push("TER");
    previousChain = chain;
    const resName = String(topology.labels[a] ?? "UNK").slice(0, 3);
    const record = STANDARD.has(resName) && topology.molType[a] !== 3 ? "ATOM  " : "HETATM";
    const element = (ELEMENT_SYMBOLS[topology.element[a] - 1] ?? "X");
    const symbol = element.length === 1 ? element : element[0] + element[1].toLowerCase();
    const atomName = name.length < 4 && symbol.length === 1 ? ` ${name}` : name;
    const x = coords[a * 3].toFixed(3).padStart(8);
    const y = coords[a * 3 + 1].toFixed(3).padStart(8);
    const z = coords[a * 3 + 2].toFixed(3).padStart(8);
    const b = (bFactors?.[a] ?? 0).toFixed(2).padStart(6);
    lines.push(`${record}${String(serial).padStart(5)} ${atomName.padEnd(4)} ${resName.padStart(3)} ${chain.slice(0, 1)}`
      + `${String(residue).padStart(4)}    ${x}${y}${z}  1.00${b}          ${symbol.padStart(2)}`);
    serial += 1;
  }
  lines.push("TER", "END");
  // (the record's chain column is the id's first character, so an end is named the same way)
  const bonds = (links ?? []).map((pair) => pair.map(([chain, residue, atom]) => [String(chain).slice(0, 1), residue, atom]));
  return `${[...pdbLinkRecords(lines, bonds), ...lines].join("\n")}\n`;
}

/** Each atom's (chain, residue) CA or C1' for the chain-geometry gate: the spacing between consecutive residues. */
export function alphaCarbonSpacings(atomKeys, topology, coords) {
  const centres = [];
  for (let a = 0; a < atomKeys.length; a += 1) {
    if (topology.molType[a] !== 0 || atomKeys[a][2] !== "CA") continue;
    centres.push({ chain: atomKeys[a][0], residue: atomKeys[a][1], at: a });
  }
  const spacing = [];
  for (let i = 1; i < centres.length; i += 1) {
    const p = centres[i - 1];
    const q = centres[i];
    if (p.chain !== q.chain || q.residue !== p.residue + 1) continue;
    spacing.push(Math.hypot(coords[q.at * 3] - coords[p.at * 3], coords[q.at * 3 + 1] - coords[p.at * 3 + 1],
                            coords[q.at * 3 + 2] - coords[p.at * 3 + 2]));
  }
  return { spacing, centres };
}
