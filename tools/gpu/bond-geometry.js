/**
 * Are a predicted structure's BONDS the right length - mainchain, sidechain,
 * peptide and ligand, scored separately?
 *
 *     import { bondGeometry, bondReport } from "./bond-geometry.js";
 *
 * 🔴 WHY THIS EXISTS WHEN RMSD AND pLDDT ALREADY DO. Neither can see a bond.
 * RMSD is dominated by where the fold sits, so a chain with every side chain
 * torn open still scores well if its alpha carbons land; pLDDT is the model's
 * own opinion and this repository has measured it at 92.38 on a glycerol whose
 * C1-O1 was 6.97 A against a 1.43 ideal, and at 81.4 through a rosettafold3
 * flow walk that collapses 68 residues into a 2.2 A ball. `chain-geometry.js`
 * closed part of the gap - it checks N-CA, CA-C and consecutive CA - and stops
 * at the BACKBONE by design, stepping over side chains and ligands entirely.
 *
 * 🔴 AND THE IDEAL LENGTHS ARE NOT TYPED IN. They are measured from the
 * reference conformer this port already ships and already folds against
 * (`tools/oracle/reference-conformers.json`), so the bar is the same geometry
 * the featuriser hands the model rather than a second table that can disagree
 * with it. A pair is BONDED if the ideal conformer puts it within
 * `BONDED_ANGSTROMS`: in an idealised residue a covalent bond is 1.2-1.8 A and
 * the next-nearest contact is past 2.1, so the cutoff is not a close call. It
 * is derived once per residue type, not per structure.
 *
 * 🔴 AND THE FOUR CLASSES ARE REPORTED SEPARATELY BECAUSE THEY FAIL
 * SEPARATELY. rosettafold3's flow walk breaks N-CA by 4.2x while leaving CA-CA
 * within 1.2x - a mainchain-only score calls that a 4x failure and a whole-
 * structure score dilutes it to nothing. Side chains are where a sampler that
 * has lost local geometry shows first, and a ligand has no backbone to hide
 * behind at all.
 */

/** An ideal-conformer distance at or below this is a covalent bond. */
const BONDED_ANGSTROMS = 1.95;

/** The peptide C(i)-N(i+1), which no single residue's conformer contains. */
const PEPTIDE_ANGSTROMS = 1.329;
/** Past this, consecutive residues are not bonded - a chain break or a gap. */
const PEPTIDE_BROKEN = 2.5;

const MAINCHAIN = new Set(["N", "CA", "C", "O", "OXT"]);

/**
 * The standard nucleotides, so a base is not reported as a ligand.
 *
 * 🔴 A NUCLEOTIDE IS SCORED THROUGH THE COMPONENT PATH LIKE A LIGAND AND IS NOT
 * ONE. Both take their ideals from the CCD because `reference-conformers.json`
 * covers only the twenty amino acids, but calling a guanine a ligand in the
 * report makes an RNA row unreadable and hides a torn ligand in a complex that
 * also has RNA. The class is the chemistry, not the code path.
 */
const NUCLEIC = new Set(["A", "C", "G", "U", "DA", "DC", "DG", "DT", "I", "DI", "N"]);

const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/**
 * The bonded pairs of one residue type and their ideal lengths, from the
 * reference conformer. Memoised per residue type: the graph is a property of
 * the chemistry, not of the fold being scored.
 */
/**
 * 🔴 EXPORTED SO THERE IS ONE DEFINITION OF "WHAT A BOND IS". `clash-geometry.js`
 * needs the same graph to know which pairs sterics does NOT govern, and a
 * second derivation there would let the two instruments disagree about a bond -
 * one scoring it as too long while the other counts it as a clash.
 */
export function idealBonds(conformer, cache, code) {
  if (cache.has(code)) return cache.get(code);
  const atoms = conformer.internal ?? conformer.nTerminal ?? [];
  const bonds = [];
  for (let i = 0; i < atoms.length; i += 1) {
    for (let j = i + 1; j < atoms.length; j += 1) {
      const ideal = distance(atoms[i].pos, atoms[j].pos);
      if (ideal > BONDED_ANGSTROMS) continue;
      bonds.push({ a: atoms[i].name, b: atoms[j].name, ideal });
    }
  }
  cache.set(code, bonds);
  return bonds;
}

/** ATOM/HETATM records, in file order, grouped into residues. */
export function parsePdbResidues(text) {
  const residues = [];
  let current = null;
  for (const line of text.split("\n")) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const name = line.slice(12, 16).trim();
    const code = line.slice(17, 20).trim();
    const chain = line.slice(21, 22);
    const number = Number(line.slice(22, 26));
    const pos = [Number(line.slice(30, 38)), Number(line.slice(38, 46)),
                 Number(line.slice(46, 54))];
    if (current === null || current.number !== number || current.chain !== chain
        || current.code !== code) {
      current = { code, chain, number, hetatm: line.startsWith("HETATM"), atoms: new Map() };
      residues.push(current);
    }
    current.atoms.set(name, pos);
  }
  return residues;
}

/**
 * @param {string} pdb a predicted structure
 * @param {object} conformers tools/oracle/reference-conformers.json
 * @param {{ligandBonds?: Array<[string,string,number]>}} [options] ideal
 *   lengths for a non-standard component, which has no reference conformer
 * @returns per-class rms error in angstroms, counts, and the worst offenders
 */
export function bondGeometry(pdb, conformers, options = {}) {
  const residues = parsePdbResidues(pdb);
  const cache = new Map();
  const classes = { mainchain: [], sidechain: [], peptide: [], nucleic: [], ligand: [] };
  const offenders = [];

  const record = (kind, label, seen, ideal) => {
    classes[kind].push(seen - ideal);
    offenders.push({ kind, label, seen, ideal, error: Math.abs(seen - ideal) });
  };

  for (let index = 0; index < residues.length; index += 1) {
    const residue = residues[index];
    const conformer = conformers[oneLetter(residue.code)];
    if (conformer === undefined) {
      // 🔴 A COMPONENT WITH NO CONFORMER IS A LIGAND, AND ITS BONDS MUST BE
      // GIVEN. Guessing them from the PREDICTION would score the fold against
      // itself and always pass - which is the shape of the `relative-rms over a
      // non-array returns 0` trap this repository already records.
      // 🔴 A COMPONENT FROM THE DICTIONARY OUTRANKS A HAND-TYPED LIST, and
      // covers what no list did: nucleotides and modified residues, not only
      // ligands. `components` is code -> parseCcdComponent output.
      const supplied = options.components?.get?.(residue.code);
      const table = supplied !== undefined
        ? componentBonds(supplied) : (options.ligandBonds ?? []);
      for (const [a, b, ideal] of table) {
        const first = residue.atoms.get(a); const second = residue.atoms.get(b);
        if (first === undefined || second === undefined) continue;
        record(NUCLEIC.has(residue.code) ? "nucleic" : "ligand",
               `${residue.code}${residue.number} ${a}-${b}`,
               distance(first, second), ideal);
      }
      continue;
    }
    for (const bond of idealBonds(conformer, cache, residue.code)) {
      const first = residue.atoms.get(bond.a); const second = residue.atoms.get(bond.b);
      if (first === undefined || second === undefined) continue;
      const kind = MAINCHAIN.has(bond.a) && MAINCHAIN.has(bond.b)
        ? "mainchain" : "sidechain";
      record(kind, `${residue.code}${residue.number} ${bond.a}-${bond.b}`,
             distance(first, second), bond.ideal);
    }
    const next = residues[index + 1];
    if (next === undefined || next.chain !== residue.chain) continue;
    const carbon = residue.atoms.get("C"); const nitrogen = next.atoms.get("N");
    if (carbon === undefined || nitrogen === undefined) continue;
    const seen = distance(carbon, nitrogen);
    // A chain break is not a wrong bond length; scoring it as one would make
    // every multi-chain fold look broken at the join.
    if (seen > PEPTIDE_BROKEN) continue;
    record("peptide", `${residue.code}${residue.number}-${next.code}${next.number} C-N`,
           seen, PEPTIDE_ANGSTROMS);
  }

  const rms = (values) => values.length === 0 ? null
    : Math.sqrt(values.reduce((sum, v) => sum + v * v, 0) / values.length);
  offenders.sort((a, b) => b.error - a.error);
  // 🔴 THE SIGN, NOT ONLY THE MAGNITUDE - AND THIS FUNCTION DID NOT REPORT IT
  // WHILE ITS SIBLING BELOW ALWAYS HAS. An rms says how far the bonds are from
  // ideal; the MEAN says which way, and that is what distinguishes a conformer
  // that is merely imprecise from one that is systematically CONTRACTED. It is
  // the signal that identified the defect docs/AF3.md records - AlphaFold 3's
  // side chains at 0.339 A rms "with 100% of them SHORT" - and reading only the
  // rms there would have said "noisy" where the truth was "squashed". Two
  // functions answering the same question with different fields is how a
  // diagnostic gets lost.
  const meanOf = (list) => (list.length === 0 ? null
    : list.reduce((total, error) => total + error, 0) / list.length);
  const summary = (list) => ({
    rms: rms(list), mean: meanOf(list), bonds: list.length,
    short: list.filter((error) => error < 0).length,
  });
  return {
    mainchain: summary(classes.mainchain),
    sidechain: summary(classes.sidechain),
    peptide: summary(classes.peptide),
    nucleic: summary(classes.nucleic),
    ligand: summary(classes.ligand),
    all: { rms: rms([...classes.mainchain, ...classes.sidechain,
                     ...classes.peptide, ...classes.nucleic, ...classes.ligand]),
           bonds: offenders.length },
    worst: offenders.slice(0, 5).map((o) => ({
      ...o, seen: Number(o.seen.toFixed(3)), ideal: Number(o.ideal.toFixed(3)),
      error: Number(o.error.toFixed(3)) })),
  };
}

/** The three-letter code as the conformer table keys it. */
/** The one-letter code a conformer is keyed by, from a PDB's three. */
export function oneLetter(code) {
  const table = {
    ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E",
    GLY: "G", HIS: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M", PHE: "F",
    PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V", UNK: "X",
  };
  return table[code];
}

/** One line per class, for a sweep's table. */
export function bondReport(result) {
  const cell = (name) => result[name].rms === null ? `${name} -`
    : `${name} ${result[name].rms.toFixed(3)} (${result[name].bonds})`;
  return ["mainchain", "sidechain", "peptide", "nucleic", "ligand"].map(cell).join("  ");
}


/**
 * The bonded pairs of a CCD component and their ideal lengths, from the
 * dictionary's own bond list and ideal coordinates.
 *
 * 🔴 THIS IS WHY A LIGAND NO LONGER NEEDS A HAND-TYPED TABLE, AND WHY NUCLEIC
 * ACIDS CAN BE SCORED AT ALL. `reference-conformers.json` is twenty amino acids
 * and an X, so a glycerol's five bonds were typed into two files and a DA, DC,
 * DG, DT, A, C, G or U could not be scored by anything here. The CCD is not the
 * prediction - it is the authority the featuriser itself reads through
 * `parseCcdComponent` - so taking ideals from it does not score a fold against
 * itself, which is the trap `ligandBonds` existed to avoid.
 *
 * 🔴 AND IT USES THE DICTIONARY'S BOND LIST, NOT A DISTANCE CUTOFF. The
 * conformer path infers a bond from `BONDED_ANGSTROMS`, which is sound for an
 * amino acid and guesswork on a crowded ring; `_chem_comp_bond` states them,
 * with orders. A component whose ideal coordinates are all zero - the
 * dictionary carries some - yields no bonds rather than a table of noise.
 */
export function componentBonds(component) {
  const atoms = component.atoms ?? [];
  // 🔴 `parseCcdComponent` GIVES x/y/z, NOT A `pos` TRIPLE - the conformer set's
  // shape. Reading `atom.pos[0]` here threw on the first component tried.
  const at = (atom) => [atom.x, atom.y, atom.z];
  const spread = atoms.reduce((most, atom) =>
    Math.max(most, Math.abs(atom.x), Math.abs(atom.y), Math.abs(atom.z)), 0);
  if (spread === 0) return [];
  const bonds = [];
  for (const bond of component.bonds ?? []) {
    const from = atoms[bond.from], to = atoms[bond.to];
    if (from === undefined || to === undefined) continue;
    const ideal = distance(at(from), at(to));
    if (!(ideal > 0)) continue;
    bonds.push([from.name, to.name, ideal]);
  }
  return bonds;
}

/**
 * The same rule over AF3's DENSE `[tokens, maxAtoms, 3]` grid, scored against
 * the reference conformer positions the batch itself carries.
 *
 * 🔴 WHY A SECOND ENTRY POINT RATHER THAN A PDB. An oracle dump is a grid and a
 * set of `ref_*` features; turning it into a PDB needs residue names, a chain
 * map and the terminal-atom convention, and every one of those is a place for
 * the two sides of a comparison to differ for reasons that are not the model's.
 * Here the ideal length is `ref_pos`'s own distance - the same number the
 * featuriser handed the model - so native's answer and this port's are scored
 * by one rule with nothing in between.
 *
 * @param {Float32Array} positions [tokens * maxAtoms * 3]
 * @param {{refPos: Float32Array, refMask: Float32Array,
 *          nameChars: Float32Array, tokens: number, maxAtoms: number}} layout
 */
export function denseBondGeometry(positions, layout) {
  const { refPos, refMask, nameChars, tokens, maxAtoms } = layout;
  const name = (token, slot) => {
    let text = "";
    for (let c = 0; c < 4; c += 1) {
      const code = nameChars[(token * maxAtoms + slot) * 4 + c];
      if (code > 0) text += String.fromCharCode(code + 32);
    }
    return text.trim();
  };
  const at = (source, token, slot) => {
    const base = (token * maxAtoms + slot) * 3;
    return [source[base], source[base + 1], source[base + 2]];
  };
  const classes = { mainchain: [], sidechain: [] };
  const bonds = [];
  for (let token = 0; token < tokens; token += 1) {
    for (let i = 0; i < maxAtoms; i += 1) {
      if (refMask[token * maxAtoms + i] === 0) continue;
      for (let j = i + 1; j < maxAtoms; j += 1) {
        if (refMask[token * maxAtoms + j] === 0) continue;
        const ideal = distance(at(refPos, token, i), at(refPos, token, j));
        if (ideal > BONDED_ANGSTROMS || ideal === 0) continue;
        const a = name(token, i), b = name(token, j);
        const kind = MAINCHAIN.has(a) && MAINCHAIN.has(b) ? "mainchain" : "sidechain";
        const seen = distance(at(positions, token, i), at(positions, token, j));
        classes[kind].push(seen - ideal);
        bonds.push({ token, kind, label: `${a}-${b}`, ideal, seen });
      }
    }
  }
  const rms = (v) => (v.length === 0 ? null
    : Math.sqrt(v.reduce((s, d) => s + d * d, 0) / v.length));
  const mean = (v) => (v.length === 0 ? null : v.reduce((s, d) => s + d, 0) / v.length);
  return {
    mainchain: { rms: rms(classes.mainchain), mean: mean(classes.mainchain),
                 bonds: classes.mainchain.length },
    sidechain: { rms: rms(classes.sidechain), mean: mean(classes.sidechain),
                 bonds: classes.sidechain.length,
                 short: classes.sidechain.filter((d) => d < 0).length },
    bonds,
  };
}
