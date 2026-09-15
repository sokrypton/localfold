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

const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/**
 * The bonded pairs of one residue type and their ideal lengths, from the
 * reference conformer. Memoised per residue type: the graph is a property of
 * the chemistry, not of the fold being scored.
 */
function idealBonds(conformer, cache, code) {
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
  const classes = { mainchain: [], sidechain: [], peptide: [], ligand: [] };
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
      for (const [a, b, ideal] of options.ligandBonds ?? []) {
        const first = residue.atoms.get(a); const second = residue.atoms.get(b);
        if (first === undefined || second === undefined) continue;
        record("ligand", `${residue.code} ${a}-${b}`, distance(first, second), ideal);
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
  return {
    mainchain: { rms: rms(classes.mainchain), bonds: classes.mainchain.length },
    sidechain: { rms: rms(classes.sidechain), bonds: classes.sidechain.length },
    peptide: { rms: rms(classes.peptide), bonds: classes.peptide.length },
    ligand: { rms: rms(classes.ligand), bonds: classes.ligand.length },
    all: { rms: rms([...classes.mainchain, ...classes.sidechain,
                     ...classes.peptide, ...classes.ligand]),
           bonds: offenders.length },
    worst: offenders.slice(0, 5).map((o) => ({
      ...o, seen: Number(o.seen.toFixed(3)), ideal: Number(o.ideal.toFixed(3)),
      error: Number(o.error.toFixed(3)) })),
  };
}

/** The three-letter code as the conformer table keys it. */
function oneLetter(code) {
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
  return ["mainchain", "sidechain", "peptide", "ligand"].map(cell).join("  ");
}
