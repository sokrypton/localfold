/**
 * AlphaFold 3's batch as a PDB: the residue tables, the dense layout's atom names and elements, and the writer -
 * out of src/af3/fold.js (which re-exports them), so a featuriser and the CUDA port's exporters take it without the
 * WebGPU fold behind it.
 */
import { ELEMENT_SYMBOLS } from "../featurise/ccd-component.js";

export const THREE_LETTER = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY",
  H: "HIS", I: "ILE", L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER",
  T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};

/**
 * ...and back, which reading a template needs.
 *
 * 🔴 INVERTED RATHER THAN WRITTEN OUT, so the two directions cannot disagree.
 * A second table would be twenty more lines that have to be edited together
 * with this one, and the failure of getting one wrong is a residue silently
 * becoming UNK - a four-atom blank the model folds around.
 */
export const ONE_LETTER = Object.fromEntries(
  Object.entries(THREE_LETTER).map(([one, three]) => [three, one]));

/**
 * The element symbol for an atomic number.
 *
 * 🔴 THIS WAS FOUR ENTRIES - C, N, O, S - AND EVERYTHING ELSE FELL THROUGH TO
 * CARBON. Right for a protein, which has nothing else, and wrong for most
 * ligands: across a corpus of 51 distinct hetero components, TWENTY-EIGHT carry
 * an element it dropped. Every phosphate-bearing ligand (P), every heme (FE),
 * and every metal ion - a magnesium written as a carbon atom.
 *
 * It costs twice over, because a viewer with no CONECT records derives a
 * ligand's bonds from the DISTANCE between atoms of known elements: a
 * disulfide at 2.05 A read as C-C (whose ceiling is 1.8) vanishes, and a P-O
 * at 1.63 read as C-O survives its 1.65 ceiling by two hundredths.
 *
 * ccd-component.js already had the full list, indexed the same way, so this is
 * one question with one answer rather than a second short table beside it.
 */
function elementSymbol(atomicNumber) {
  return ELEMENT_SYMBOLS[atomicNumber - 1] ?? "C";
}

/** The four-character atom name AF3 stores as codes offset by 32. */
export function atomName(nameChars, slot) {
  let name = "";
  for (let character = 0; character < 4; character += 1) {
    const code = nameChars[slot * 4 + character];
    if (code > 0) name += String.fromCharCode(code + 32);
  }
  return name.trim();
}

/**
 * A PDB from the dense atom layout, with pLDDT in the B-factor column so the
 * viewer can colour by it.
 *
 * 🔴 AND A HEADER ONLY WHERE ONE IS PASSED, WHICH IS WHAT MAKES IT SAFE TO ADD.
 * This wrote no REMARK at all, so a file saved from the page named neither the
 * model that produced it nor the quantity in its B-factor column - seven
 * families share this writer, and the AlphaFold 2 path beside it has carried a
 * provenance REMARK since it was written. The header is a CALLER's, because the
 * caller that knows which checkpoint ran is the page, and because three tests
 * and several tools read this output in file order: emitting a line they did
 * not ask for would change what every one of them parses.
 */
export function toPdb(batch, positions, plddt, options = {}) {
  // 🔴 ONE TEMPLATE A BATCH, BECAUSE A TRAJECTORY IS 25 FILES OF THE SAME
  // RECORDS. Everything in an ATOM line but the coordinates and the B-factor -
  // serial, name, residue, chain, element, TER, CONECT - is fixed by the batch,
  // and it was rebuilt for every frame: 26 PDBs after each AF3 fold, ~100 ms at
  // 255 residues on the critical path, doubled on a Colab runtime's CPU.
  const { records, conect } = pdbTemplate(batch);
  const lines = [];
  // ...first, before any coordinate record: a reader that stops at the first
  // ATOM never sees anything written after one, and most readers do.
  for (const text of options.remark ?? []) lines.push(`REMARK   1 ${text}`);
  for (const record of records) {
    if (record === "TER") { lines.push("TER"); continue; }
    const { slot, head, tail } = record;
    const confidence = plddt ? plddt[slot] : 0;
    lines.push(head
      + positions[slot * 3].toFixed(3).padStart(8)
      + positions[slot * 3 + 1].toFixed(3).padStart(8)
      + positions[slot * 3 + 2].toFixed(3).padStart(8)
      + "  1.00" + confidence.toFixed(2).padStart(6) + tail);
  }
  for (const line of conect) lines.push(line);
  lines.push("END");
  return lines.join("\n");
}

const PDB_TEMPLATES = new WeakMap();

/** The position-free part of `toPdb`'s output for a batch, built once. */
function pdbTemplate(batch) {
  const cached = PDB_TEMPLATES.get(batch);
  if (cached !== undefined) return cached;
  const { tokens, dense, sequence } = batch;
  const records = [];
  const conect = [];
  let serial = 1;
  // 🔴 ONE LETTER PER CHAIN, NOT "A" FOR EVERYTHING. A complex written as one
  // chain is a single 126-residue protein as far as any viewer or scoring tool
  // is concerned, with a peptide bond implied across an interface that has
  // none.
  // 🔴 SIXTY-TWO, NOT TWENTY-SIX: a PDB chain is one character, and the 27th chain of a large
  // complex wrapped back onto A and was MERGED with it in every reader. Upper case, then lower case,
  // then digits - the convention viewers accept - before the format runs out (an mmCIF names them
  // all; native/af3 writes one with --out=*.cif).
  const CHAIN_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const chainLetter = (token) => {
    const asym = batch.asymId === undefined ? 1 : batch.asymId[token];
    return CHAIN_CHARACTERS[(asym - 1) % CHAIN_CHARACTERS.length];
  };
  // 🔴 A LIGAND IS HETATM, AND IT HAS A NAME. `sequence` covers the polymers,
  // so a ligand token indexed into it is undefined and used to be written as a
  // UNK residue - which a viewer draws as an unknown amino acid and a scoring
  // tool reads as part of the chain. The span table says which tokens belong to
  // which component, and the component's own code is its residue name.
  //
  // 🔴 A MODIFIED RESIDUE IS THE SAME PROBLEM ONE STEP FURTHER IN. Its tokens
  // are inside the chain, so `sequence` HAS a letter at that position - but
  // `sequence` is indexed by RESIDUE and this loop walks TOKENS, and the two
  // stopped being the same number the moment a residue could be several
  // tokens. Every residue from the first modification onwards was named by
  // whatever letter happened to sit at its token index: a chain that reads as
  // a real protein and is not the one that was folded. residueOfToken is the
  // map, and a modified residue takes its component's own code, which is what
  // a PDB calls one.
  const componentOf = new Map();
  for (const span of [...(batch.ligandSpans ?? []), ...(batch.modifiedSpans ?? [])]) {
    for (let offset = 0; offset < span.count; offset += 1) {
      componentOf.set(span.from + offset, span.code);
    }
  }
  const residueOf = (token) => batch.residueOfToken?.[token] ?? token;
  // 🔴 A NUCLEOTIDE'S RESIDUE NAME IS NOT ITS AMINO ACID'S. THREE_LETTER maps
  // the one-letter code through the amino-acid table, where `A` is ALA and `G`
  // is GLY - so a DNA chain came out of here as a poly-alanine peptide that
  // every viewer draws as a protein ribbon and every scoring tool reads as one.
  // A PDB names DNA " DA" and RNA "  A", right-justified in the field, which is
  // what distinguishes the two: the D is the only thing in the format that
  // says which.
  const nucleicName = (residue) => {
    const kind = batch.chainKinds?.[batch.chainOfResidue?.[residue]];
    if (kind !== "dna" && kind !== "rna") return undefined;
    const code = sequence[residue];
    if (code === undefined) return undefined;
    return kind === "dna" ? `D${code}` : code;
  };
  // ...and which serial each ligand token was written as, so CONECT can name
  // them. A ligand token is one heavy atom and it sits in slot zero, so the
  // token is the atom; a polymer token is many atoms and has no entry here.
  const serialOfToken = new Map();
  for (let token = 0; token < tokens; token += 1) {
    const ligandCode = componentOf.get(token);
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (!batch.predDenseAtomMask[slot]) continue;
      if (ligandCode !== undefined && atom === 0) serialOfToken.set(token, serial);
      const name = atomName(batch.displayAtomNameChars ?? batch.refAtomNameChars, slot);
      records.push({ slot, head:
        (ligandCode === undefined ? "ATOM  " : "HETATM")
        + String(serial).padStart(5) + " "
        + (name.length < 4 ? ` ${name}`.padEnd(4) : name.slice(0, 4)) + " "
        + (ligandCode
          ?? nucleicName(residueOf(token))?.padStart(3)
          ?? THREE_LETTER[sequence[residueOf(token)]] ?? "UNK").padEnd(3)
        + " " + chainLetter(token)
        // 🔴 residue_index IS ALREADY 1-BASED. Adding one here shifted the whole
        // chain by a residue, which against a helical protein reads as a 3.7 A
        // RMSD and a TM-score of 0.37 - a plausible "wrong fold" rather than an
        // obvious bug. The real number was 0.69 A.
        + String(batch.features.residueIndex[token]).padStart(4) + "    ",
      tail: "          " + elementSymbol(batch.refElement[slot]).padStart(2) });
      serial += 1;
    }
    if (batch.asymId !== undefined && token + 1 < tokens
        && batch.asymId[token + 1] !== batch.asymId[token]) {
      records.push("TER");
    }
  }
  // 🔴 CONECT, OR THE LIGAND IS A BAG OF ATOMS. A viewer handed no bonds
  // derives them from the DISTANCE between atoms - py2Dmol says so out loud
  // ("No bonds - will use distance calculation") - and on a diffusion
  // trajectory it re-derives them from EVERY frame's coordinates, which are
  // deliberately noisy until the last few steps. The sticks then appear, cross
  // and vanish frame to frame: the picture looks broken while the prediction
  // may be fine. The bonds are already known - they came out of the CCD - so
  // there is nothing to compute here, only to write down.
  //
  // Ligand-internal only. A covalent link between a ligand and a polymer is
  // not featurised yet (see featurise.js), so claiming one here would be the
  // writer inventing chemistry the fold never saw.
  //
  // Four partners per line, continued on another CONECT for an atom with more:
  // the record has room for exactly four and a fifth silently overruns into
  // the next field.
  const partners = new Map();
  for (const span of [...(batch.ligandSpans ?? []), ...(batch.modifiedSpans ?? [])]) {
    for (const bond of span.bonds ?? []) {
      const a = serialOfToken.get(span.from + bond.from);
      const b = serialOfToken.get(span.from + bond.to);
      if (a === undefined || b === undefined) continue;   // a masked atom
      if (!partners.has(a)) partners.set(a, []);
      if (!partners.has(b)) partners.set(b, []);
      partners.get(a).push(b);
      partners.get(b).push(a);
    }
  }
  for (const [atom, bonded] of [...partners].sort((x, y) => x[0] - y[0])) {
    for (let start = 0; start < bonded.length; start += 4) {
      let line = "CONECT" + String(atom).padStart(5);
      for (const other of bonded.slice(start, start + 4)) line += String(other).padStart(5);
      conect.push(line);
    }
  }
  const template = { records, conect };
  PDB_TEMPLATES.set(batch, template);
  return template;
}
