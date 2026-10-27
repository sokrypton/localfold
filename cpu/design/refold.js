/**
 * refold.js
 * ---------
 * A finished design, read back as a fold's input, and the refold of it laid over it.
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
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

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
 * The refold fitted onto the design, and the two together as one structure to draw.
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
 * @param {string} designPdb
 * @param {string} refoldPdb
 * @param {string|string[]} [designed] the design's designed chain(s), held out of the fit
 * @returns {{refold: string, pdb: string, chains: Map<string, string>,
 *            deviations: {index: number, distance: number}[]}}
 *   `refold` in the design's letters and frame; `pdb` the design followed by the refold under
 *   letters of its own (`chains`: design letter to overlay letter); `deviations` each refold
 *   residue's anchor distance from the design's, by its index among the overlay's residues
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
  const inOurLetters = `${relabel(refoldPdb, rename).join("\n")}\nEND\n`;
  const fitted = alignToReference(inOurLetters, designPdb, designed);

  // the overlay's own letters for the refold: the first the design does not use
  const used = new Set(design.map((chain) => chain.letter));
  const free = [...ALPHABET].filter((letter) => !used.has(letter));
  if (free.length < design.length) throw new Error("no chain letters left to draw the refold under");
  const overlayLetter = new Map(design.map((chain, at) => [chain.letter, free[at]]));

  const designLines = designPdb.split("\n").filter(isRecord);
  const refoldLines = relabel(fitted, overlayLetter, designLines.length + 1);
  const pdb = `${[...designLines, "TER", ...refoldLines].join("\n")}\nEND\n`;

  // each refold residue against the design's residue in the same place of the same chain
  const placed = new Map(chainsOf(fitted).map((chain) => [chain.letter, chain]));
  let index = design.reduce((sum, chain) => sum + chain.residues.length, 0);
  const deviations = [];
  for (const chain of chainsOf(pdb).slice(design.length)) {
    const ours = design.find((one) => overlayLetter.get(one.letter) === chain.letter);
    const moved = placed.get(ours.letter);
    chain.residues.forEach((_, at) => {
      const a = ours.residues[at]?.anchor;
      const b = moved.residues[at]?.anchor;
      if (a && b) deviations.push({ index, distance: Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) });
      index += 1;
    });
  }
  return { refold: fitted, pdb, chains: overlayLetter, deviations };
}
