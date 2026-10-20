/**
 * BioEmu2's protein physicality check on its output samples.
 *
 * A transcription of af3-any-model's pipeline/bioemu2_physicality.py, itself
 * bioemu2 @7846769's physicality.py (check_physicality and what it calls). It
 * changes no coordinates: per sample, whether the protein chains pass the
 * preset's thresholds and the FIRST check that failed, as BioEmu2's
 * `physicality.json` reports. Non-protein chains are not checked, as in
 * BioEmu2. Held to the colab port by tools/check-bioemu2-topology.js.
 *
 * Nothing here needs a GPU or a model: it runs on the page after any backend's
 * fold, on featuriseBioemu2's topology and the BioEmu2Graph bonds.
 */
import { loopOf, ELEMENT_SYMBOLS } from "../af3/featurise/ccd-component.js";
import { STANDARD_COMPONENTS } from "../af3/featurise/standard-components.js";
import { residueSpans } from "./graph.js";

const PROT_BACKBONE = ["N", "CA", "C", "O"];
const BACKBONE_ATOMS = new Set([...PROT_BACKBONE, "OXT"]);
const MAINCHAIN_BOND_WINDOWS = [[["N", "CA"], [1.25, 1.70]], [["CA", "C"], [1.35, 1.70]],
                                [["C", "O"], [0.95, 1.50]], [["C", "OXT"], [0.95, 1.50]]];
const PEPTIDE_BOND_WINDOW = [1.05, 1.65];
const SIDECHAIN_BOND_WINDOWS = { "C,C": [1.10, 1.80], "C,N": [1.10, 1.70], "C,O": [1.00, 1.60],
                                 "C,S": [1.55, 2.10], "C,SE": [1.70, 2.25] };
const DISULFIDE_BOND_WINDOW = [1.80, 2.25];
const GENERIC_BOND_WINDOW = [1.00, 1.90];
const MAX_HEAVY_NEIGHBORS = { C: 4, N: 4, O: 2, S: 6, P: 5 };
const pairs = (...names) => names.map((p) => p.split("-"));
export const SIDECHAIN_BONDS = {
  ALA: pairs("CA-CB"),
  ARG: pairs("CA-CB", "CB-CG", "CG-CD", "CD-NE", "NE-CZ", "CZ-NH1", "CZ-NH2"),
  ASN: pairs("CA-CB", "CB-CG", "CG-OD1", "CG-ND2"),
  ASP: pairs("CA-CB", "CB-CG", "CG-OD1", "CG-OD2"),
  CYS: pairs("CA-CB", "CB-SG"),
  GLN: pairs("CA-CB", "CB-CG", "CG-CD", "CD-OE1", "CD-NE2"),
  GLU: pairs("CA-CB", "CB-CG", "CG-CD", "CD-OE1", "CD-OE2"),
  GLY: [],
  HIS: pairs("CA-CB", "CB-CG", "CG-ND1", "CG-CD2", "ND1-CE1", "CD2-NE2", "CE1-NE2"),
  ILE: pairs("CA-CB", "CB-CG1", "CB-CG2", "CG1-CD1"),
  LEU: pairs("CA-CB", "CB-CG", "CG-CD1", "CG-CD2"),
  LYS: pairs("CA-CB", "CB-CG", "CG-CD", "CD-CE", "CE-NZ"),
  MET: pairs("CA-CB", "CB-CG", "CG-SD", "SD-CE"),
  PHE: pairs("CA-CB", "CB-CG", "CG-CD1", "CG-CD2", "CD1-CE1", "CD2-CE2", "CE1-CZ", "CE2-CZ"),
  PRO: pairs("CA-CB", "CB-CG", "CG-CD", "CD-N"),
  SER: pairs("CA-CB", "CB-OG"),
  THR: pairs("CA-CB", "CB-OG1", "CB-CG2"),
  TRP: pairs("CA-CB", "CB-CG", "CG-CD1", "CG-CD2", "CD1-NE1", "NE1-CE2", "CD2-CE2", "CD2-CE3", "CE2-CZ2", "CE3-CZ3",
             "CZ2-CH2", "CZ3-CH2"),
  TYR: pairs("CA-CB", "CB-CG", "CG-CD1", "CG-CD2", "CD1-CE1", "CD2-CE2", "CE1-CZ", "CE2-CZ", "CZ-OH"),
  VAL: pairs("CA-CB", "CB-CG1", "CB-CG2"),
};
const LIGAND_RESNAMES = new Set(["LIG", "L01", "L02", "L03", "L04", "L05", "L06", "L07", "L08", "L09"]);
const CHECKS = ["bad_bonds_backbone", "bad_bonds_sidechain", "bad_bonds_sulfur", "within_residue_clashes",
                "valence_exceeded", "isolated_atoms", "same_chain_residue_clashes", "cross_chain_residue_clashes",
                "duplicate_atoms", "atoms_mismatch", "bonds_mismatch"];
const preset = (values, perResidue) => ({ ...Object.fromEntries(CHECKS.map((k, i) => [k, values[i]])), perResidue });
export const THRESHOLDS = {
  no_violations: preset([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], false),
  high_quality: preset([5, 10, 0, 5, 10, 1, 10, 5, 0, 0, 0], false),
  high_quality_per_residue: preset([0.05, 0.1, 0.0, 0.05, 0.1, 0.01, 0.1, 0.05, 0.0, 0.0, 0.0], true),
};
const CUTOFFS = { tooClose: 1.0, inferredBond: 1.9, isolatedAtom: 2.0, sidechainClash: 2.0, degenerate: 0.5 };

const symbol = (z) => (z ? ELEMENT_SYMBOLS[z - 1] : "X");
const key = (a, b) => (a <= b ? `${a}\u0000${b}` : `${b}\u0000${a}`);
const numKey = (a, b) => (a <= b ? [a, b] : [b, a]);

let refAtomsCache = null;
function refAtoms() {
  if (refAtomsCache === null) {
    refAtomsCache = {};
    for (const residue of Object.keys(SIDECHAIN_BONDS)) {
      const loop = loopOf(STANDARD_COMPONENTS[residue], "_chem_comp_atom");
      const id = loop.columns.indexOf("atom_id");
      const sym = loop.columns.indexOf("type_symbol");
      refAtomsCache[residue] = new Set(loop.rows.filter((r) => r[sym] !== "H" && r[id] !== "OXT").map((r) => r[id]));
    }
  }
  return refAtomsCache;
}

/** The checked atoms (multi-residue chains with a standard amino acid), as the vendor's atom_slice leaves them. */
class Protein {
  constructor(topology, bonds) {
    const { names, labels, residueIdx: res, atomAsym: chain } = topology;
    const chains = new Map();
    for (let i = 0; i < names.length; i += 1) {
      if (!chains.has(chain[i])) chains.set(chain[i], new Set());
      chains.get(chain[i]).add(res[i]);
    }
    const keep = new Set();
    for (const [c, rs] of chains) {
      if (rs.size > 1 && names.some((_, i) => chain[i] === c && labels[i] in SIDECHAIN_BONDS)) keep.add(c);
    }
    this.idx = [];
    for (let i = 0; i < names.length; i += 1) if (keep.has(chain[i])) this.idx.push(i);
    const where = new Map(this.idx.map((o, k) => [o, k]));
    this.names = this.idx.map((i) => names[i]);
    this.resName = this.idx.map((i) => labels[i]);
    this.res = this.idx.map((i) => res[i]);
    this.chain = this.idx.map((i) => chain[i]);
    this.element = this.idx.map((i) => symbol(topology.element[i]));
    const seen = new Map();
    for (const [a, b] of bonds) {
      if (!where.has(a) || !where.has(b)) continue;
      const [x, y] = numKey(where.get(a), where.get(b));
      seen.set(`${x},${y}`, [x, y]);
    }
    this.bonds = [...seen.values()].sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    this.residues = [];
    let lo = 0;
    for (let k = 1; k <= this.idx.length; k += 1) {
      if (k === this.idx.length || this.res[k] !== this.res[lo] || this.chain[k] !== this.chain[lo]) {
        this.residues.push([lo, k]);
        lo = k;
      }
    }
    this.resOf = new Int32Array(this.idx.length);
    this.residues.forEach(([a, b], k) => { for (let i = a; i < b; i += 1) this.resOf[i] = k; });
    this.refAtoms = refAtoms();
  }

  ligandResidues() {
    const chains = [...new Set(this.chain)];
    const ligChains = new Set(chains.filter((c) => !this.residues.some(([lo]) => this.chain[lo] === c
      && this.resName[lo] in SIDECHAIN_BONDS)));
    const out = new Set();
    this.residues.forEach(([lo], k) => { if (LIGAND_RESNAMES.has(this.resName[lo]) || ligChains.has(this.chain[lo])) out.add(k); });
    return out;
  }

  alphaBackboneResidues() {
    const bonded = new Map();
    for (const [a, b] of this.bonds) {
      if (this.resOf[a] !== this.resOf[b]) continue;
      if (!bonded.has(this.resOf[a])) bonded.set(this.resOf[a], new Set());
      bonded.get(this.resOf[a]).add(key(this.names[a], this.names[b]));
    }
    const need = [key("CA", "N"), key("C", "CA"), key("C", "O")];
    const out = new Set();
    this.residues.forEach(([lo], k) => {
      if (this.resName[lo] in SIDECHAIN_BONDS || need.every((n) => bonded.get(k)?.has(n))) out.add(k);
    });
    return out;
  }
}

function windowOf(p, i, j) {
  const e = [p.element[i], p.element[j]].sort();
  if (e[0] === "S" && e[1] === "S") return DISULFIDE_BOND_WINDOW;
  return SIDECHAIN_BOND_WINDOWS[`${e[0]},${e[1]}`] ?? GENERIC_BOND_WINDOW;
}

function checkTopology(p, extra) {
  const lig = p.ligandResidues();
  const bb = p.alphaBackboneResidues();
  let dup = 0;
  let mismatch = 0;
  const expected = new Set();
  const tag = (k, lo, n) => `${p.resName[lo]}${k}-${n}`;
  for (const c of [...new Set(p.chain)].sort((a, b) => a - b)) {
    let prevC = null;
    p.residues.forEach(([lo, hi], k) => {
      if (p.chain[lo] !== c) return;
      if (lig.has(k)) { prevC = null; return; }
      const names = p.names.slice(lo, hi);
      dup += names.length !== new Set(names).size ? 1 : 0;
      const std = p.resName[lo] in SIDECHAIN_BONDS;
      if (std) {
        const ref = p.refAtoms[p.resName[lo]];
        const have = new Set(names);
        mismatch += have.size === ref.size && [...have].every((n) => ref.has(n)) ? 0 : 1;
      }
      if (bb.has(k)) {
        const bonds = [["N", "CA"], ["CA", "C"], ["C", "O"], ...(std ? SIDECHAIN_BONDS[p.resName[lo]] : [])];
        for (const [a, b] of bonds) expected.add(key(tag(k, lo, a), tag(k, lo, b)));
        if (prevC !== null) expected.add(key(prevC, tag(k, lo, "N")));
        prevC = tag(k, lo, "C");
      } else {
        prevC = null;
      }
    });
  }
  const label = (a) => `${p.resName[a]}${p.resOf[a]}-${p.names[a]}`;
  for (const [a, b] of extra) {
    if (p.resName[a] in SIDECHAIN_BONDS && p.resName[b] in SIDECHAIN_BONDS) expected.add(key(label(a), label(b)));
  }
  const current = new Set();
  for (const [a, b] of p.bonds) {
    const names = key(label(a), label(b));
    if ((p.resName[a] in SIDECHAIN_BONDS && p.resName[b] in SIDECHAIN_BONDS
         && !lig.has(p.resOf[a]) && !lig.has(p.resOf[b])) || expected.has(names)) current.add(names);
  }
  let symmetric = 0;
  for (const n of current) if (!expected.has(n)) symmetric += 1;
  for (const n of expected) if (!current.has(n)) symmetric += 1;
  return { duplicate_atoms: dup, atoms_mismatch: mismatch, bonds_mismatch: symmetric };
}

function geometry(p) {
  const lig = p.ligandResidues();
  const bb = p.alphaBackboneResidues();
  const main = [];
  const staticBad = [];
  const ridxOf = new Map();
  const covered = new Set();
  const side = [];
  const degen = [];
  for (const c of [...new Set(p.chain)].sort((a, b) => a - b)) {
    let prevC = null;
    p.residues.forEach(([lo, hi], k) => {
      if (p.chain[lo] !== c) return;
      if (lig.has(k)) { prevC = null; return; }
      const ri = staticBad.length;
      ridxOf.set(k, ri);
      const by = new Map();
      for (let a = lo; a < hi; a += 1) by.set(p.names[a], a);
      const std = p.resName[lo] in SIDECHAIN_BONDS;
      const expected = std ? [...p.refAtoms[p.resName[lo]]].filter((n) => !PROT_BACKBONE.includes(n)) : null;
      staticBad.push(expected !== null && (expected.some((n) => !by.has(n)) || by.size !== hi - lo));
      if (bb.has(k)) {
        for (const [[a, b], w] of MAINCHAIN_BOND_WINDOWS) if (by.has(a) && by.has(b)) main.push([by.get(a), by.get(b), w]);
        if (prevC !== null && by.has("N")) main.push([prevC, by.get("N"), PEPTIDE_BOND_WINDOW]);
        prevC = by.get("C") ?? null;
      } else {
        prevC = null;
      }
      if (expected === null) return;
      for (const [a, b] of SIDECHAIN_BONDS[p.resName[lo]]) {
        if (by.has(a) && by.has(b)) {
          side.push([ri, by.get(a), by.get(b), windowOf(p, by.get(a), by.get(b))]);
          covered.add(numKey(by.get(a), by.get(b)).join(","));
        }
      }
      const atoms = [...by.values()];
      for (let x = 0; x < atoms.length - 1; x += 1) {
        for (let y = x + 1; y < atoms.length; y += 1) degen.push([ri, atoms[x], atoms[y]]);
      }
    });
  }
  for (const [a, b] of main) covered.add(numKey(a, b).join(","));
  for (const [a, b] of p.bonds) {
    const ri = ridxOf.get(p.resOf[a]);
    const rj = ridxOf.get(p.resOf[b]);
    if (covered.has(`${a},${b}`) || ri === undefined || rj === undefined) continue;
    covered.add(`${a},${b}`);
    const w = windowOf(p, a, b);
    for (const r of new Set([ri, rj])) side.push([r, a, b, w]);
  }
  side.sort((x, y) => x[0] - y[0]);
  return { main, staticBad, side, degen, residuesChecked: staticBad.length };
}

const dsq = (c, i, j) => {
  const dx = c[i * 3] - c[j * 3];
  const dy = c[i * 3 + 1] - c[j * 3 + 1];
  const dz = c[i * 3 + 2] - c[j * 3 + 2];
  return dx * dx + dy * dy + dz * dz;
};
const outside = (c, i, j, w) => { const d = Math.sqrt(dsq(c, i, j)); return d < w[0] || d > w[1]; };

/** Every pair (i < j) within `r` in any frame: cKDTree.query_pairs over the frames, unioned. */
function candidatePairs(frames, n, r) {
  const keys = new Set();
  const r2 = r * r;
  for (const c of frames) {
    const cells = new Map();
    const cellOf = (i) => [0, 1, 2].map((a) => Math.floor(c[i * 3 + a] / r));
    for (let i = 0; i < n; i += 1) {
      const k = cellOf(i).join(",");
      if (!cells.has(k)) cells.set(k, []);
      cells.get(k).push(i);
    }
    for (let i = 0; i < n; i += 1) {
      const [x, y, z] = cellOf(i);
      for (let dx = -1; dx <= 1; dx += 1) for (let dy = -1; dy <= 1; dy += 1) for (let dz = -1; dz <= 1; dz += 1) {
        for (const j of cells.get(`${x + dx},${y + dy},${z + dz}`) ?? []) {
          if (j > i && dsq(c, i, j) <= r2) keys.add(i * n + j);
        }
      }
    }
  }
  return [...keys].sort((a, b) => a - b).map((k) => [Math.floor(k / n), k % n]);
}

function geometryStats(p, frames) {
  const lig = p.ligandResidues();
  const isLig = p.idx.map((_, a) => lig.has(p.resOf[a]));
  const prot = isLig.map((x) => !x);
  const g = geometry(p);
  const bondIndex = new Map(p.bonds.map(([a, b], k) => [`${a},${b}`, k]));
  const sulfur = p.bonds.filter(([a, b]) => p.element[a] === "S" && p.element[b] === "S");
  const bondNb = p.bonds.map(([a, b]) => Math.max(CUTOFFS.isolatedAtom, windowOf(p, a, b)[1]));
  const cysS = p.idx.map((_, a) => p.resName[a] === "CYS" && p.names[a] === "SG" && p.element[a] === "S");
  const elig = p.idx.map((_, a) => !isLig[a] && !BACKBONE_ATOMS.has(p.names[a]) && (
    !(p.resName[a] in p.refAtoms) || (p.refAtoms[p.resName[a]].has(p.names[a]) && !PROT_BACKBONE.includes(p.names[a]))));
  const maxNb = p.element.map((e) => MAX_HEAVY_NEIGHBORS[e] ?? 4);
  const maxD = Math.max(CUTOFFS.tooClose, CUTOFFS.inferredBond, CUTOFFS.sidechainClash, CUTOFFS.isolatedAtom,
                        Math.max(0, ...bondNb), DISULFIDE_BOND_WINDOW[1]);
  const n = p.idx.length;
  const stats = Object.fromEntries(["bad_bonds_backbone", "bad_bonds_sidechain", "bad_bonds_sulfur",
    "within_residue_clashes", "valence_exceeded", "isolated_atoms", "same_chain_residue_clashes",
    "cross_chain_residue_clashes", "non_finite_atoms"].map((k) => [k, new Array(frames.length).fill(0)]));
  const finite = [];
  frames.forEach((c, f) => {
    let bad = 0;
    for (let a = 0; a < n; a += 1) {
      if (!(Number.isFinite(c[a * 3]) && Number.isFinite(c[a * 3 + 1]) && Number.isFinite(c[a * 3 + 2]))) bad += 1;
    }
    stats.non_finite_atoms[f] = bad;
    if (bad === 0) finite.push(f);
  });
  if (finite.length === 0) return stats;
  const pairList = candidatePairs(finite.map((f) => frames[f]), n, maxD);
  for (const f of finite) {
    const c = frames[f];
    stats.bad_bonds_backbone[f] = g.main.filter(([a, b, w]) => outside(c, a, b, w)).length;
    const bad = g.staticBad.slice();
    for (const [r, a, b, w] of g.side) if (outside(c, a, b, w)) bad[r] = true;
    for (const [r, a, b] of g.degen) if (dsq(c, a, b) < CUTOFFS.degenerate ** 2) bad[r] = true;
    stats.bad_bonds_sidechain[f] = bad.filter(Boolean).length;
    stats.bad_bonds_sulfur[f] = sulfur.filter(([a, b]) => outside(c, a, b, DISULFIDE_BOND_WINDOW)).length;
    const d2 = pairList.map(([i, j]) => dsq(c, i, j));
    const clash = d2.map((d) => d < CUTOFFS.sidechainClash ** 2);
    const ssPairs = [];
    pairList.forEach(([i, j], k) => {
      if (cysS[i] && cysS[j] && p.resOf[i] !== p.resOf[j]
          && d2[k] >= DISULFIDE_BOND_WINDOW[0] ** 2 && d2[k] <= DISULFIDE_BOND_WINDOW[1] ** 2) ssPairs.push(k);
    });
    if (ssPairs.length) {
      const partners = new Int32Array(n);
      for (const k of ssPairs) { partners[pairList[k][0]] += 1; partners[pairList[k][1]] += 1; }
      for (const k of ssPairs) if (partners[pairList[k][0]] === 1 && partners[pairList[k][1]] === 1) clash[k] = false;
    }
    let within = 0;
    let same = 0;
    let cross = 0;
    const inferred = new Int32Array(n);
    const neighbours = new Int32Array(n);
    pairList.forEach(([i, j], k) => {
      const pp = prot[i] && prot[j];
      const bi = bondIndex.get(`${i},${j}`);
      const nbD = bi === undefined ? CUTOFFS.isolatedAtom : bondNb[bi];
      if (pp && d2[k] < CUTOFFS.tooClose ** 2) within += 1;
      if (pp && d2[k] <= CUTOFFS.inferredBond ** 2) { inferred[i] += 1; inferred[j] += 1; }
      if (pp && d2[k] <= nbD ** 2) { neighbours[i] += 1; neighbours[j] += 1; }
      const clashPair = p.resOf[i] !== p.resOf[j] && bi === undefined;
      if (p.chain[i] === p.chain[j]) { if (clashPair && elig[i] && elig[j] && clash[k]) same += 1; }
      else if (clashPair && clash[k]) cross += 1;
    });
    stats.within_residue_clashes[f] = within;
    stats.valence_exceeded[f] = [...inferred].filter((v, a) => v > maxNb[a]).length;
    stats.isolated_atoms[f] = [...neighbours].filter((v, a) => prot[a] && v === 0).length;
    stats.same_chain_residue_clashes[f] = same;
    stats.cross_chain_residue_clashes[f] = cross;
  }
  return stats;
}

/**
 * Per sample: {ok, reason, stats}. `frames` are Float64Array/Float32Array of
 * (atoms * 3) angstrom in BioEmu2's atom order (NaN for an atom a sample lacks);
 * `bonds` BioEmu2Graph's; `declared` the job's declared bonds as atom-index pairs.
 *
 * @param {"high_quality"|"no_violations"|"high_quality_per_residue"|"no_filter"} mode
 */
export function checkPhysicality(topology, bonds, frames, mode = "high_quality", declared = []) {
  const notChecked = () => frames.map(() => ({ ok: true, reason: null, stats: {} }));
  if (mode === "no_filter") return notChecked();
  if (!(mode in THRESHOLDS)) throw new Error(`physicality ${mode}: want ${Object.keys(THRESHOLDS).join(", ")} or no_filter`);
  const p = new Protein(topology, bonds);
  if (p.idx.length === 0) return notChecked();
  const where = new Map(p.idx.map((o, k) => [o, k]));
  const extra = declared.filter(([a, b]) => where.has(a) && where.has(b)).map(([a, b]) => [where.get(a), where.get(b)]);
  const sliced = frames.map((c) => {
    const out = new Float64Array(p.idx.length * 3);
    p.idx.forEach((o, k) => { for (let a = 0; a < 3; a += 1) out[k * 3 + a] = c[o * 3 + a]; });
    return out;
  });
  const stats = geometryStats(p, sliced);
  const topologyStats = checkTopology(p, extra);
  for (const [k, v] of Object.entries(topologyStats)) stats[k] = frames.map(() => v);
  const { perResidue, ...thresholds } = THRESHOLDS[mode];
  const nRes = p.residues.length;
  return frames.map((_, f) => {
    let reason = stats.non_finite_atoms[f] > 0 ? "non_finite_atoms" : null;
    if (reason === null) {
      for (const [k, v] of Object.entries(thresholds)) {
        const value = perResidue ? stats[k][f] / nRes : stats[k][f];
        if (value > v) { reason = k; break; }
      }
    }
    return { ok: reason === null, reason, stats: Object.fromEntries(Object.entries(stats).map(([k, v]) => [k, v[f]])) };
  });
}

export { residueSpans };
