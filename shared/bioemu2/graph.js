/**
 * BioEmu2's bonded topology: the vendor's `topology_from_structure`, as
 * af3-any-model's pipeline/bioemu2_steering.Graph transcribes it.
 *
 * MDTraj's standard residue bonds - for every residue BioEmu2 builds, the CCD's
 * own heavy-atom bonds - plus O3'-P between adjacent nucleotides of a nucleic
 * chain, C-N between adjacent protein residues (whatever their template), and
 * the featuriser's template and declared bonds. Both the physicality report
 * and the steering potentials read it.
 *
 * `topology` is featuriseBioemu2's.
 */
import { loopOf } from "../af3/featurise/ccd-component.js";
import { STANDARD_COMPONENTS } from "../af3/featurise/standard-components.js";

export const PROTEIN = 0, DNA = 1, RNA = 2, NONPOLYMER = 3;
export const UNK_PROT = 22;
export const CANONICAL_PROTEIN = Object.freeze([...Array(20).keys()].map((i) => i + 2));
export const NUCLEIC_TO_CCD = Object.freeze({ 28: "DA", 30: "DC", 29: "DG", 31: "DT", 23: "A", 25: "C", 24: "G", 26: "U" });
export const TOKEN_NAMES = Object.freeze({
  2: "ALA", 3: "ARG", 4: "ASN", 5: "ASP", 6: "CYS", 7: "GLN", 8: "GLU", 9: "GLY", 10: "HIS", 11: "ILE",
  12: "LEU", 13: "LYS", 14: "MET", 15: "PHE", 16: "PRO", 17: "SER", 18: "THR", 19: "TRP", 20: "TYR", 21: "VAL",
  22: "UNK_PROT",
});
export const D_AMINO_ACID_TO_L = Object.freeze({
  DAL: "ALA", DAR: "ARG", DAS: "ASP", DCY: "CYS", DGL: "GLU", DGN: "GLN", DHI: "HIS", DIL: "ILE", DLE: "LEU",
  DLY: "LYS", DPN: "PHE", DPR: "PRO", DSG: "ASN", DSN: "SER", DTH: "THR", DTR: "TRP", DTY: "TYR", DVA: "VAL",
  MED: "MET",
});
export const NONBONDED_EXCLUSION_GRAPH_DISTANCE = 3;
/** MDTraj ships standard bonds for these residue names, and none for UNK, a modified residue or a ligand. */
const STANDARD = new Set([...Object.values(TOKEN_NAMES).filter((n) => n !== "UNK_PROT"),
                          "A", "C", "G", "U", "DA", "DC", "DG", "DT"]);

const standardBondCache = new Map();
/** The CCD's heavy-atom bond names of a standard component. */
function standardBondNames(label) {
  if (!standardBondCache.has(label)) {
    const loop = loopOf(STANDARD_COMPONENTS[label], "_chem_comp_bond");
    const a = loop.columns.indexOf("atom_id_1");
    const b = loop.columns.indexOf("atom_id_2");
    standardBondCache.set(label, loop.rows.map((row) => [row[a], row[b]]));
  }
  return standardBondCache.get(label);
}

/** (first, last + 1) per residue, in atom order. */
export function residueSpans(residueIdx) {
  const out = [];
  let lo = 0;
  for (let i = 1; i <= residueIdx.length; i += 1) {
    if (i === residueIdx.length || residueIdx[i] !== residueIdx[lo]) { out.push([lo, i]); lo = i; }
  }
  return out;
}

const key2 = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);

/** BondedGeometryGraph. */
export class BioEmu2Graph {
  constructor(topology) {
    const { names, labels, residueIdx: res, molType: mol, atomAsym: chain, residueType: rtype,
            tokenIdxAtAtom: tok } = topology;
    const n = names.length;
    const bonds = new Map();
    const add = (a, b) => { const lo = Math.min(a, b); const hi = Math.max(a, b); bonds.set(`${lo},${hi}`, [lo, hi]); };
    const residues = residueSpans(res);
    for (const [lo, hi] of residues) {
      if (!STANDARD.has(labels[lo])) continue;
      const where = new Map();
      for (let i = lo; i < hi; i += 1) where.set(names[i], i);
      for (const [a, b] of standardBondNames(labels[lo])) {
        if (where.has(a) && where.has(b)) add(where.get(a), where.get(b));
      }
    }
    for (const [a, b] of topology.bondIndex) add(a, b);
    const nucleicChains = new Set(residues.filter(([lo]) => mol[lo] === DNA || mol[lo] === RNA).map(([lo]) => chain[lo]));
    let prev = null;
    for (const [lo, hi] of residues) {
      if (prev !== null && chain[prev[0]] === chain[lo]) {
        const pa = new Map();
        for (let i = prev[0]; i < prev[1]; i += 1) pa.set(names[i], i);
        const ca = new Map();
        for (let i = lo; i < hi; i += 1) ca.set(names[i], i);
        if (nucleicChains.has(chain[lo]) && pa.has("O3'") && ca.has("P")) add(pa.get("O3'"), ca.get("P"));
        if (mol[prev[0]] === PROTEIN && mol[lo] === PROTEIN && pa.has("N") && pa.has("C") && ca.has("N")) {
          add(pa.get("C"), ca.get("N"));
        }
      }
      prev = [lo, hi];
    }
    this.bonds = [...bonds.values()].sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    this.bondSet = new Set(this.bonds.map(([a, b]) => `${a},${b}`));
    // atom labels: (reference residue name, atom name)
    this.labels = new Array(n);
    this.residue = new Array(n);
    this.mol = Array.from(mol);
    this.ordinaryX = new Set();
    for (const [lo, hi] of residues) {
      const token = mol[lo] === PROTEIN ? rtype[lo] : null;
      const tokens = new Set();
      for (let i = lo; i < hi; i += 1) tokens.add(tok[i]);
      const isX = token === UNK_PROT && tokens.size === 1;
      const name = NUCLEIC_TO_CCD[rtype[lo]] ?? D_AMINO_ACID_TO_L[labels[lo]] ?? labels[lo];
      for (let i = lo; i < hi; i += 1) {
        this.labels[i] = [isX ? TOKEN_NAMES[token] : name, names[i]];
        this.residue[i] = [chain[lo], res[lo]];
        if (isX) this.ordinaryX.add(i);
      }
    }
    this.neighbors = new Map();
    for (const [i, j] of this.bonds) {
      if (!this.neighbors.has(i)) this.neighbors.set(i, new Set());
      if (!this.neighbors.has(j)) this.neighbors.set(j, new Set());
      this.neighbors.get(i).add(j);
      this.neighbors.get(j).add(i);
    }
    // (i, k, centre) for every pair of a centre's neighbours, centres in order
    this.angles = [];
    for (const c of [...this.neighbors.keys()].sort((a, b) => a - b)) {
      const bonded = [...this.neighbors.get(c)].sort((a, b) => a - b);
      for (let x = 0; x < bonded.length; x += 1) {
        for (let y = x + 1; y < bonded.length; y += 1) this.angles.push([bonded[x], bonded[y], c]);
      }
    }
    this.peptideEdges = new Set(this.bonds.filter(([i, j]) => !this.sameResidue(i, j)
      && this.residue[i][0] === this.residue[j][0] && this.mol[i] === PROTEIN && this.mol[j] === PROTEIN
      && new Set([this.labels[i][1], this.labels[j][1]]).size === 2
      && [this.labels[i][1], this.labels[j][1]].includes("C") && [this.labels[i][1], this.labels[j][1]].includes("N")
      && Math.abs(this.residue[i][1] - this.residue[j][1]) === 1).map(([i, j]) => `${i},${j}`));
  }

  sameResidue(i, j) {
    return this.residue[i][0] === this.residue[j][0] && this.residue[i][1] === this.residue[j][1];
  }

  peptideFollowingResidue(i, j) {
    if (!this.peptideEdges.has(`${Math.min(i, j)},${Math.max(i, j)}`)) return null;
    const [resI, nameI] = this.labels[i];
    return nameI === "N" ? resI : this.labels[j][0];
  }

  peptideAngleFollowingResidue(oi, c, ok) {
    let following = null;
    if (this.sameResidue(c, ok)) following = this.peptideFollowingResidue(oi, c);
    if (following === null && this.sameResidue(c, oi)) following = this.peptideFollowingResidue(c, ok);
    return following;
  }

  isNaLinkBond(i, j) {
    const names = [this.labels[i][1], this.labels[j][1]];
    return !this.sameResidue(i, j) && this.mol[i] === this.mol[j] && (this.mol[i] === DNA || this.mol[i] === RNA)
      && names.includes("O3'") && names.includes("P") && names[0] !== names[1]
      && this.residue[i][0] === this.residue[j][0] && Math.abs(this.residue[i][1] - this.residue[j][1]) === 1;
  }

  isDisulfide(i, j) {
    return this.labels[i][0] === "CYS" && this.labels[i][1] === "SG" && this.labels[j][0] === "CYS"
      && this.labels[j][1] === "SG" && this.mol[i] === PROTEIN && this.mol[j] === PROTEIN;
  }

  isNaLinkAngle(oi, c, ok) {
    if (new Set([this.mol[oi], this.mol[c], this.mol[ok]]).size !== 1) return false;
    return [[oi, ok], [ok, oi]].some(([outer, other]) => this.sameResidue(c, other) && this.isNaLinkBond(outer, c));
  }

  /** Every pair within `maxBonds` bonds of each other, (lower, higher), sorted. */
  nonbondedExclusions(maxBonds = NONBONDED_EXCLUSION_GRAPH_DISTANCE) {
    const out = new Map();
    for (const source of this.neighbors.keys()) {
      const depth = new Map([[source, 0]]);
      const queue = [source];
      for (let q = 0; q < queue.length; q += 1) {
        const node = queue[q];
        if (depth.get(node) >= maxBonds) continue;
        for (const other of this.neighbors.get(node)) {
          if (!depth.has(other)) { depth.set(other, depth.get(node) + 1); queue.push(other); }
        }
      }
      for (const o of depth.keys()) if (source < o) out.set(`${source},${o}`, [source, o]);
    }
    return [...out.values()].sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  }
}

export { key2 };
