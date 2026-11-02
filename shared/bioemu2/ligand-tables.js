/**
 * The ligand half (Tier 2) of BioEmu2's steering tables, from the CCD - not from RDKit.
 *
 * 🔴 AN APPROXIMATION OF THE VENDOR'S TABLES, ON PURPOSE, AND SAID SO
 * EVERYWHERE IT IS USED. bioemu2 @7846769 builds these from Boltz-1's CCD
 * pickle of RDKit molecules (a 346 MB file of Python objects) with RDKit's
 * distance-geometry bounds matrix and CIP ranks, which no browser and no
 * native featuriser here has. What the potentials READ is:
 *
 *   * PoseBusters: a lower and upper bound on every pair of a ligand's heavy
 *     atoms, flagged bond / angle / neither, each widened by a 15-20% buffer;
 *   * ChiralAtom: (n1, n2, n3, centre) tuples and a sign for each;
 *   * StereoBond: (a, s, e, b) tuples across an E/Z double bond and E or Z;
 *   * PlanarBond: the six atoms of each non-aromatic C=C between trigonal carbons.
 *
 * So this builds each of them from the dictionary itself:
 *
 *   * bounds: the IDEAL conformer's 1-2 and 1-3 distances with RDKit's own
 *     tolerances (0.01 and 0.04 A), 1-4 between the cis and trans distances of
 *     the ideal bond lengths and angles, everything else a VDW lower bound
 *     (RDKit's radii, its 0.7 scale at five atoms), then triangle smoothing over
 *     the whole molecule, hydrogens included, as GetMoleculeBoundsMatrix does.
 *     Measured against the vendor on glycerol: the same 15 pairs and flags; the
 *     bonds sit at the CCD's 1.43 A C-O where RDKit's UFF says 1.394;
 *   * chirality and E/Z: the CCD's own `pdbx_stereo_config` says which centres
 *     and bonds have one; a tuple's ORIENTATION is read off the ideal geometry
 *     (the sign of its dihedral), which is the same constraint whatever order
 *     the neighbours come in - the vendor orders them by CIP rank, this by
 *     atom order with hydrogens last;
 *   * planarity: the C=C rule, aromatic bonds excluded as a sanitised molecule's are.
 *
 * Only a CCD-coded single-component ligand gets these, as in the vendor
 * (`ligand_ccd`): not a SMILES ligand, a glycan or a modified residue.
 */
import { loopOf } from "../af3/featurise/ccd-component.js";

/** RDKit's PeriodicTable rvdw, angstrom, by element symbol. */
const RVDW = { H: 1.2, B: 1.8, C: 1.7, N: 1.6, O: 1.55, F: 1.5, SI: 2.1, P: 1.8, S: 1.8, CL: 1.8, BR: 1.9, I: 2.1, SE: 1.9 };
const DIST12_TOL = 0.01;
const DIST13_TOL = 0.04;
const VDW_SCALE_15 = 0.7;
const MAX_UPPER = 1000;

function component(text) {
  const atoms = loopOf(text, "_chem_comp_atom");
  const col = (name) => atoms.columns.indexOf(name);
  const ideal = atoms.rows.every((r) => ["x", "y", "z"].every((a) => Number.isFinite(Number(r[col(`pdbx_model_Cartn_${a}_ideal`)]))));
  const names = atoms.rows.map((r) => r[col("atom_id")]);
  const element = atoms.rows.map((r) => (r[col("type_symbol")] ?? "C").toUpperCase());
  const xyz = atoms.rows.map((r) => ["x", "y", "z"].map((a) => Number(r[col(ideal ? `pdbx_model_Cartn_${a}_ideal` : `model_Cartn_${a}`)])));
  const chiral = atoms.rows.map((r) => ["R", "S"].includes(r[col("pdbx_stereo_config")]));
  const bondLoop = loopOf(text, "_chem_comp_bond");
  const bcol = (name) => bondLoop.columns.indexOf(name);
  const index = new Map(names.map((n, i) => [n, i]));
  const bonds = bondLoop.rows.map((r) => ({
    a: index.get(r[bcol("atom_id_1")]), b: index.get(r[bcol("atom_id_2")]), order: r[bcol("value_order")] ?? "SING",
    aromatic: (r[bcol("pdbx_aromatic_flag")] ?? "N") === "Y", stereo: ["E", "Z"].includes(r[bcol("pdbx_stereo_config")]),
  })).filter((bond) => bond.a !== undefined && bond.b !== undefined);
  const neighbours = names.map(() => []);
  for (const { a, b } of bonds) { neighbours[a].push(b); neighbours[b].push(a); }
  return { names, element, xyz, chiral, bonds, neighbours };
}

const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
const sub = (p, q) => [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
/** The vendor's _dihedral convention. */
function dihedral(p0, p1, p2, p3) {
  const rij = sub(p0, p1);
  const rkj = sub(p2, p1);
  const rkl = sub(p2, p3);
  const n1 = cross(rij, rkj);
  const n2 = cross(rkj, rkl);
  return Math.atan2(dot(rkj, cross(n1, n2)) / Math.sqrt(dot(rkj, rkj)), dot(n1, n2));
}

/** Graph distances (bond counts) from every atom, BFS. */
function topologicalDistances(neighbours) {
  return neighbours.map((_, s) => {
    const d = new Int32Array(neighbours.length).fill(-1);
    d[s] = 0;
    const queue = [s];
    for (let q = 0; q < queue.length; q += 1) for (const o of neighbours[queue[q]]) if (d[o] < 0) { d[o] = d[queue[q]] + 1; queue.push(o); }
    return d;
  });
}

/** The cis and trans distances of a-b-c-d from its bond lengths and its two angles. */
function cisTrans(l1, l2, l3, angle1, angle2) {
  // b at the origin, c on +x, a in the xy plane at angle1 from b->c, d at angle2 from c->b with torsion phi
  const a = [l1 * Math.cos(angle1), l1 * Math.sin(angle1), 0];
  const at = (phi) => [l2 - l3 * Math.cos(angle2), l3 * Math.sin(angle2) * Math.cos(phi), l3 * Math.sin(angle2) * Math.sin(phi)];
  return [dist(a, at(0)), dist(a, at(Math.PI))];
}

function boundsMatrix(c) {
  const n = c.names.length;
  const topo = topologicalDistances(c.neighbours);
  const lower = Array.from({ length: n }, () => new Float64Array(n));
  const upper = Array.from({ length: n }, () => new Float64Array(n).fill(MAX_UPPER));
  const angleAt = (i, j, k) => {
    const u = sub(c.xyz[i], c.xyz[j]);
    const v = sub(c.xyz[k], c.xyz[j]);
    return Math.acos(Math.max(-1, Math.min(1, dot(u, v) / Math.sqrt(dot(u, u) * dot(v, v)))));
  };
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const d = dist(c.xyz[i], c.xyz[j]);
      let lo;
      let hi;
      if (topo[i][j] === 1) { lo = d - DIST12_TOL; hi = d + DIST12_TOL; }
      else if (topo[i][j] === 2) { lo = d - DIST13_TOL; hi = d + DIST13_TOL; }
      else if (topo[i][j] === 3) {
        // the path a-b-c-d through the shared neighbours
        const b = c.neighbours[i].find((x) => topo[x][j] === 2);
        const cc = c.neighbours[j].find((x) => topo[x][i] === 2 && c.neighbours[b].includes(x));
        if (b === undefined || cc === undefined) { lo = d; hi = d; }
        else {
          const [cis, trans] = cisTrans(dist(c.xyz[i], c.xyz[b]), dist(c.xyz[b], c.xyz[cc]), dist(c.xyz[cc], c.xyz[j]),
                                        angleAt(i, b, cc), angleAt(b, cc, j));
          lo = Math.min(cis, trans);
          hi = Math.max(cis, trans);
        }
      } else {
        const vdw = (RVDW[c.element[i]] ?? 1.8) + (RVDW[c.element[j]] ?? 1.8);
        lo = topo[i][j] === 4 ? vdw * VDW_SCALE_15 : vdw;
        lo = Math.min(lo, d);           // never above the ideal conformer itself
        hi = MAX_UPPER;
      }
      lower[i][j] = lower[j][i] = lo;
      upper[i][j] = upper[j][i] = hi;
    }
  }
  // triangle smoothing (Floyd-Warshall on the upper bounds; lower from the upper of the other two sides)
  for (let k = 0; k < n; k += 1) {
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        if (i === j) continue;
        const u = upper[i][k] + upper[k][j];
        if (u < upper[i][j]) upper[i][j] = u;
        const l1 = lower[i][k] - upper[k][j];
        const l2 = lower[k][j] - upper[i][k];
        const l = Math.max(l1, l2);
        if (l > lower[i][j]) lower[i][j] = l;
      }
    }
  }
  return { lower, upper, topo };
}

/**
 * Tier 2 for every CCD-coded ligand residue in `topology`, keyed as the vendor's
 * feats are: rdkit_bounds_*, chiral_atom_*, stereo_bond_*, planar_bond_index.
 *
 * @param {(code: string) => string} ccd a component's mmCIF text
 */
export function ligandTables(topology, ccd) {
  const { names, residueIdx: res, molType: mol, atomAsym: chain, ligandCcd } = topology;
  const geo = [];
  const chir = [];
  const ster = [];
  const plan = [];
  let lo = 0;
  while (lo < names.length) {
    let hi = lo + 1;
    while (hi < names.length && res[hi] === res[lo] && chain[hi] === chain[lo]) hi += 1;
    const code = mol[lo] === 3 ? ligandCcd[chain[lo]] : undefined;
    if (code) {
      const c = component(ccd(code));
      const where = new Map();
      for (let k = lo; k < hi; k += 1) where.set(names[k].trim(), k);
      const map = new Map();
      c.names.forEach((nm, i) => { if (where.has(nm)) map.set(i, where.get(nm)); });
      if (map.size) {
        const { lower, upper, topo } = boundsMatrix(c);
        const n = c.names.length;
        for (let i = 0; i < n; i += 1) {
          for (let j = i + 1; j < n; j += 1) {
            if (!map.has(i) || !map.has(j)) continue;
            geo.push([map.get(i), map.get(j), topo[i][j] === 1, topo[i][j] === 2, upper[i][j], lower[i][j]]);
          }
        }
        // chiral centres the dictionary labels, sp3 with at most four neighbours; hydrogens order last
        c.chiral.forEach((isChiral, centre) => {
          if (!isChiral || c.neighbours[centre].length > 4 || c.neighbours[centre].length < 3) return;
          const nb = [...c.neighbours[centre]].sort((a, b) => (c.element[a] === "H") - (c.element[b] === "H") || a - b);
          const tuples = [[...nb.slice(0, 3), centre]];
          if (nb.length === 4) {
            for (let skip = 0; skip < 3; skip += 1) {
              const cs = nb.slice(0, 4).filter((_, i) => i !== skip);
              tuples.push([...(skip % 2 === 0 ? cs.slice(0, 3).reverse() : cs.slice(0, 3)), centre]);
            }
          }
          for (const ids of tuples) {
            if (!ids.every((i) => map.has(i))) continue;
            const phi = dihedral(...ids.map((i) => c.xyz[i]));
            chir.push([ids.map((i) => map.get(i)), phi > 0]);
          }
        });
        // E/Z double bonds the dictionary labels
        for (const bond of c.bonds) {
          if (!bond.stereo) continue;
          const sn = c.neighbours[bond.a].filter((x) => x !== bond.b).sort((p, q) => (c.element[p] === "H") - (c.element[q] === "H") || p - q);
          const en = c.neighbours[bond.b].filter((x) => x !== bond.a).sort((p, q) => (c.element[p] === "H") - (c.element[q] === "H") || p - q);
          const pairs = [[sn[0], en[0]]];
          if (sn.length === 2 && en.length === 2) pairs.push([sn[1], en[1]]);
          for (const [p, q] of pairs) {
            const ids = [p, bond.a, bond.b, q];
            if (p === undefined || q === undefined || !ids.every((i) => map.has(i))) continue;
            const phi = dihedral(...ids.map((i) => c.xyz[i]));
            ster.push([ids.map((i) => map.get(i)), Math.abs(phi) > Math.PI / 2]);
          }
        }
        // [C;X3;^2](*)(*)=[C;X3;^2](*)(*): a non-aromatic double bond between two trigonal carbons
        for (const bond of c.bonds) {
          if (bond.order !== "DOUB" || bond.aromatic) continue;
          const [a, b] = [bond.a, bond.b];
          if (c.element[a] !== "C" || c.element[b] !== "C" || c.neighbours[a].length !== 3 || c.neighbours[b].length !== 3) continue;
          const na = c.neighbours[a].filter((x) => x !== b);
          const nb = c.neighbours[b].filter((x) => x !== a);
          const ids = [a, ...na, b, ...nb];
          if (ids.every((i) => map.has(i))) plan.push(ids.map((i) => map.get(i)));
        }
      }
    }
    lo = hi;
  }
  const rows = (tuples, k) => Array.from({ length: k }, (_, r) => Int32Array.from(tuples, (t) => t[r]));
  return {
    rdkit_bounds_index: rows(geo.map((g) => [g[0], g[1]]), 2),
    rdkit_bounds_bond_mask: Uint8Array.from(geo, (g) => (g[2] ? 1 : 0)),
    rdkit_bounds_angle_mask: Uint8Array.from(geo, (g) => (g[3] ? 1 : 0)),
    rdkit_upper_bounds: Float32Array.from(geo, (g) => g[4]),
    rdkit_lower_bounds: Float32Array.from(geo, (g) => g[5]),
    chiral_atom_index: rows(chir.map((x) => x[0]), 4),
    chiral_atom_orientations: Uint8Array.from(chir, (x) => (x[1] ? 1 : 0)),
    stereo_bond_index: rows(ster.map((x) => x[0]), 4),
    stereo_bond_orientations: Uint8Array.from(ster, (x) => (x[1] ? 1 : 0)),
    planar_bond_index: rows(plan, 6),
  };
}
