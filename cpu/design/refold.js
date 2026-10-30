/**
 * refold.js
 * ---------
 * A finished design, read back as a fold's input, and the refold of it fitted onto it.
 *
 * 🔴 READ OFF THE STRUCTURE, NOT TRACKED ALONGSIDE IT. RFD4-Proteina produces coordinates and its
 * residue identities come out of its decoder, so the structure is the only place a generation's
 * sequence exists; reading a hunt's design the same way means the refold checks what was made
 * rather than what the page believes was.
 *
 * 🔴 AND EVERY CHAIN IS READ, NOT ONLY THE PROTEIN. The first reader joined every ATOM chain as
 * amino acids, so a DNA chain refolded as a run of X and a ligand was dropped. A residue this
 * cannot name is an error, not an X: an X would refold a different molecule and report on it.
 *
 * Pure: PDB strings in and out, so `test/design-refold.test.js` holds it without a GPU.
 */
import { THREE_TO_ONE } from "./mpnn/constants.js";
import { alignToReference } from "./align.js";

const DNA = { DA: "A", DC: "C", DG: "G", DT: "T" };
const RNA = { A: "A", C: "C", G: "G", U: "U" };
const WATER = new Set(["HOH", "WAT", "DOD"]);
// a residue's position: CA for an amino acid, C1' for a nucleotide
const ANCHORS = ["CA", "C1'"];

// a fold's template slots (webgpu/af3/fold.js pads to four)
const TEMPLATE_SLOTS = 4;

const isRecord = (line) => line.startsWith("ATOM  ") || line.startsWith("HETATM");

/**
 * The chains of a PDB in file order: each with its residues (name, number, anchor coordinates) and
 * whether it is a polymer (any ATOM record) or a ligand (HETATM only). Water is left out.
 */
function chainsOf(pdb) {
  const chains = new Map();
  let previous = null;
  for (const line of pdb.split("\n")) {
    if (!isRecord(line)) continue;
    const name = line.slice(17, 20).trim();
    if (WATER.has(name)) continue;
    const letter = line[21];
    if (!chains.has(letter)) chains.set(letter, { letter, polymer: false, residues: [] });
    const chain = chains.get(letter);
    if (line.startsWith("ATOM  ")) chain.polymer = true;
    const key = line.slice(21, 27);
    if (key !== previous) {
      chain.residues.push({ name, number: line.slice(22, 27).trim(), anchor: null });
      previous = key;
    }
    const atom = line.slice(12, 16).trim();
    const residue = chain.residues[chain.residues.length - 1];
    if (residue.anchor === null && ANCHORS.includes(atom)) {
      residue.anchor = [Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54))];
    }
  }
  return [...chains.values()];
}

/** A polymer chain's kind and one-letter sequence, or an error naming the residue it cannot read. */
function polymerOf(chain) {
  for (const [kind, table] of [["protein", THREE_TO_ONE], ["dna", DNA], ["rna", RNA]]) {
    if (chain.residues.every((residue) => table[residue.name] !== undefined)) {
      return { kind, sequence: chain.residues.map((residue) => table[residue.name]).join("") };
    }
  }
  const odd = chain.residues.find((residue) => THREE_TO_ONE[residue.name] === undefined
    && DNA[residue.name] === undefined && RNA[residue.name] === undefined);
  throw new Error(odd === undefined
    ? `chain ${chain.letter} mixes amino acids and nucleotides, which a refold cannot read`
    : `chain ${chain.letter} has ${odd.name} at ${odd.number}, which a refold cannot read yet`);
}

/**
 * What a fold needs to make this design again: the polymer chains joined by ":", each one's kind,
 * and each ligand chain's components.
 *
 * @param {string} pdb
 * @returns {{sequence: string, chainKinds: string[], ligandCodes: (string|{codes: string[]})[]}}
 */
export function foldInputFromPdb(pdb) {
  const chains = chainsOf(pdb);
  const polymers = chains.filter((chain) => chain.polymer).map(polymerOf);
  if (polymers.length === 0) throw new Error("the design has no polymer chain to refold");
  const ligandCodes = chains.filter((chain) => !chain.polymer).map((chain) => (chain.residues.length === 1
    ? chain.residues[0].name : { codes: chain.residues.map((residue) => residue.name) }));
  return {
    sequence: polymers.map((polymer) => polymer.sequence).join(":"),
    chainKinds: polymers.map((polymer) => polymer.kind),
    ligandCodes,
  };
}

/**
 * The design's kept protein chains as templates for its refold, one slot a chain.
 *
 * 🔴 A REFOLD FROM THE SEQUENCE ALONE MISFOLDS A NATURAL TARGET: AF3 put 5O45's PD-L1 8.3 A from its
 * crystal, so the check of the binder was a check of a target nobody designed. Templating what the
 * job KEPT holds the target to the structure the design was made against, and leaves the designed
 * chains to the model.
 *
 * 🔴 ONE SLOT A CHAIN, NEVER ONE SPANNING THEM (`spanChains` off): a slot over the complex would
 * hand the model where the binder sits, and where it sits is what the refold is asked.
 *
 * 🔴 AND PART OF A DESIGNED CHAIN, WHEN THE JOB KEPT PART OF IT (`partial`). An antibody whose CDR loops
 * were redesigned keeps its framework; folding that framework from sequence tests whether a framework
 * folds, which nobody designed. A partial template covers the kept residues only - every other
 * position is AF3's gap token - so the loops, and where the chain docks, are still the model's.
 * Which residues count as kept, and when a chain is worth a partial template at all, is the caller's
 * call: templating a small motif would hand the model the answer a motif-scaffolding refold asks for.
 *
 * @param {string} pdb
 * @param {string|string[]} [designed] the designed chain(s), which are not templated
 * @param {{chain: string, residues: number[]}[]} [partial] designed chains to template in part: each
 *   kept residue as its 0-based place among the chain's residues
 * @returns {{text: string, chainId: string, chain: number, mapping?: [number, number][]}[]} as
 *   foldAf3's `templates` takes them: `chain` is the chain's place among the polymers
 *   foldInputFromPdb hands over, and `mapping` (partial only) pairs each kept residue with itself
 */
export function keptTemplates(pdb, designed, partial = []) {
  const held = designed === undefined ? [] : Array.isArray(designed) ? designed : [designed];
  const polymers = chainsOf(pdb).filter((chain) => chain.polymer);
  const kept = polymers.map((chain, at) => ({ chain, at }))
    .filter(({ chain }) => !held.includes(chain.letter) && polymerOf(chain).kind === "protein");
  const parts = [];
  for (const { chain: letter, residues } of partial) {
    const at = polymers.findIndex((chain) => chain.letter === letter);
    // a kept chain is already templated whole, and a chain that is not here has nothing to give
    if (at < 0 || !held.includes(letter) || polymerOf(polymers[at]).kind !== "protein") continue;
    const length = polymers[at].residues.length;
    const positions = [...new Set(residues)].filter((index) => Number.isInteger(index) && index >= 0 && index < length)
      .sort((a, b) => a - b);
    if (positions.length > 0) parts.push({ chain: polymers[at], at, positions });
  }
  if (kept.length + parts.length > TEMPLATE_SLOTS) {
    throw new Error(`the design keeps ${kept.length + parts.length} protein chains and a fold takes`
      + ` ${TEMPLATE_SLOTS} templates`);
  }
  return [
    ...kept.map(({ chain, at }) => ({ text: pdb, chainId: chain.letter, chain: at })),
    ...parts.map(({ chain, at, positions }) => ({
      text: pdb, chainId: chain.letter, chain: at, mapping: positions.map((index) => [index, index]),
    })),
  ];
}

/** Rewrite every record's chain letter through `rename`, and renumber the atom serials from `first`. */
function relabel(pdb, rename, first = 1) {
  let serial = first;
  const out = [];
  for (const line of pdb.split("\n")) {
    if (!isRecord(line)) continue;
    const letter = rename.get(line[21]);
    if (letter === undefined) continue;
    out.push(`${line.slice(0, 6)}${String(serial).padStart(5)}${line.slice(11, 21)}${letter}${line.slice(22)}`);
    serial += 1;
  }
  return out;
}

/**
 * The refold fitted onto the design, in the design's chain letters, and how far each of its
 * residues sits from the design's.
 *
 * 🔴 THE REFOLD'S CHAINS ARE MATCHED BY ORDER, NOT BY LETTER. A fold letters its chains A, B, ...
 * in the order it was handed them, and RFD4-Proteina letters its own way (the 5O45 binder is
 * design B, target A), so the same letter can name different molecules in the two files. Polymers
 * pair in file order, ligands in file order - the order `foldInputFromPdb` handed them over in.
 *
 * 🔴 FITTED THE WAY THE RUN WAS (`alignToReference`): on every chain but the designed one, so a
 * binder that lands elsewhere on its target shows as moved rather than being fitted away; with
 * nothing else, on itself.
 *
 * 🔴 ITS OWN STRUCTURE, NOT CHAINS ADDED TO THE DESIGN'S (2026-10-27). The refold was first drawn
 * as the design plus the refold under spare letters, one structure of twice the chains; the page
 * draws it as a second viewer object over the design's now, so it keeps the design's letters.
 *
 * @param {string} designPdb
 * @param {string} refoldPdb
 * @param {string|string[]} [designed] the design's designed chain(s), held out of the fit
 * @returns {{refold: string, deviations: {index: number, chain: string, residue: number,
 *            distance: number}[]}} `refold` in the design's letters and frame; `deviations` each
 *   refold residue's anchor distance from the design's, by its index in file order and by its chain
 *   and residue number - 🔴 the second is what a viewer can address, since py2Dmol does not keep
 *   a file's chain order (a design written B then A is drawn A then B)
 */
export function refoldOverlay(designPdb, refoldPdb, designed) {
  const design = chainsOf(designPdb);
  const refold = chainsOf(refoldPdb);
  const rename = new Map();
  for (const polymer of [true, false]) {
    const ours = design.filter((chain) => chain.polymer === polymer);
    const theirs = refold.filter((chain) => chain.polymer === polymer);
    if (ours.length !== theirs.length) {
      throw new Error(`the refold has ${theirs.length} ${polymer ? "polymer" : "ligand"} chains`
        + ` where the design has ${ours.length}`);
    }
    theirs.forEach((chain, at) => rename.set(chain.letter, ours[at].letter));
  }
  const fitted = alignToReference(`${relabel(refoldPdb, rename).join("\n")}\nEND\n`, designPdb, designed);

  // each refold residue against the design's residue in the same place of the same chain
  const ours = new Map(design.map((chain) => [chain.letter, chain]));
  let index = 0;
  const deviations = [];
  for (const chain of chainsOf(fitted)) {
    chain.residues.forEach((residue, at) => {
      const a = ours.get(chain.letter).residues[at]?.anchor;
      const b = residue.anchor;
      if (a && b) {
        deviations.push({ index, chain: chain.letter, residue: Number(residue.number),
          distance: Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) });
      }
      index += 1;
    });
  }
  return { refold: fitted, deviations };
}
