/**
 * Chemically equivalent atoms and identical subunits, resolved as RFD4-Proteina's
 * `StructureComparisonMetric` resolves them before a motif RMSD.
 *
 * Mirrors `rfproteina/utils/atom_permutation.py` (`permute_gt_atoms`, `atom_permutation`),
 * `rfproteina/utils/subunit_permutation.py` (`permute_gt_subunits`) and the automorphism search
 * they rest on, atomworks' `find_automorphisms_with_networkx` (`atomworks/ml/transforms/symmetry.py`):
 * the array is split into units (a polymer residue each, a non-polymer `pn_unit` whole), and each
 * unit's automorphisms are its bond graph's, nodes coloured by element, whole-array heavy-atom
 * degree and polymer residue rank, edges by a resonance-normalised bond order.
 *
 * 🔴 THE BOND ORDERS ARE RDKit's PERCEPTION IN ATOMWORKS AND A PORT OF ITS RULE HERE. atomworks
 * hands each unit to RDKit (`atom_array_to_rdkit`, sanitised) and equalises every aromatic bond
 * and every CONJUGATED single or double one, then every bond from an atom to its terminal
 * same-element neighbours (a carboxylate's or phosphate's oxygens; never carbon, never a triple
 * bond). The conjugation half is RDKit's `markConjAtomBonds` on the unit's Kekulé bonds, with
 * RDKit's implicit hydrogens filled in from its valence tables - which matters, because a unit is
 * cut out of its chain and the cut is filled with hydrogens. tools/check-rfd4p-automorphisms.py
 * holds this file's automorphisms to atomworks' over the chemical component dictionary.
 *
 * Only the CHOICE of labelling follows from any of this: the choice is an argmin over squared
 * errors, so a candidate order that differs from networkx's VF2 order can only matter on an exact
 * tie, and the RMSD is then taken under the labelling chosen.
 */

import { BOND_TYPE } from './pipeline/cif.js';
import { superimposeCoords3D } from './output.js';

const RESONANT = 1.5;
const AROMATIC_TYPES = new Set([BOND_TYPE.AROMATIC, BOND_TYPE.AROMATIC_SINGLE, BOND_TYPE.AROMATIC_DOUBLE,
  BOND_TYPE.AROMATIC_TRIPLE]);
const ORDER_BY_TYPE = { [BOND_TYPE.SINGLE]: 1, [BOND_TYPE.DOUBLE]: 2, [BOND_TYPE.TRIPLE]: 3,
  [BOND_TYPE.QUADRUPLE]: 4 };
const HYDROGEN_LIKE = new Set(['H', 'D', 'T']);

/** `_bond_order`: every aromatic flavour one resonant label; anything unrecognised its own. */
function bondOrder(type) {
  if (AROMATIC_TYPES.has(type)) return RESONANT;
  return ORDER_BY_TYPE[type] ?? -1 - type;
}

// RDKit's periodic table, for the elements a design's components are made of: default valence,
// allowed valences and outer-shell electrons (`GetDefaultValence`, `GetValenceList`,
// `GetNOuterElecs`). An element it gives no default valence (a transition metal) is -1.
const PERIODIC = {
  H: [1, 1, [1], 1], B: [5, 3, [3], 3], C: [6, 4, [4], 4], N: [7, 3, [3], 5], O: [8, 2, [2], 6],
  F: [9, 1, [1], 7], NA: [11, 1, [1, -1], 1], MG: [12, 2, [2, -1], 2], AL: [13, 3, [3], 3],
  SI: [14, 4, [4], 4], P: [15, 3, [3, 5], 5], S: [16, 2, [2, 4, 6], 6], CL: [17, 1, [1], 7],
  K: [19, 1, [1, -1], 1], CA: [20, 2, [2, -1], 2], AS: [33, 3, [3, 5], 5], SE: [34, 2, [2, 4, 6], 6],
  BR: [35, 1, [1], 7], I: [53, 1, [1, 3, 5], 7],
};
// `isEarlyAtom`: hydrogens point the other way on these (NHYD = V - SBO - CHG).
const EARLY = new Set(['LI', 'BE', 'B', 'NA', 'MG', 'AL', 'K', 'CA']);
const periodic = (el) => PERIODIC[el] ?? [0, -1, [-1], 0];

/**
 * RDKit's conjugation (`setConjugation` / `markConjAtomBonds`) over a unit's bonds, given as
 * Kekulé orders the way `atom_array_to_rdkit` adds them, with RDKit's implicit hydrogens.
 * @returns {boolean[]} per bond
 */
function conjugatedBonds(element, charge, bonds) {
  const n = element.length;
  const contrib = bonds.map(([, , t]) => (t === BOND_TYPE.AROMATIC ? 1.5 : (t === BOND_TYPE.AROMATIC_SINGLE ? 1
    : t === BOND_TYPE.AROMATIC_DOUBLE ? 2 : t === BOND_TYPE.AROMATIC_TRIPLE ? 3
      : t === BOND_TYPE.COORDINATION ? 0 : (ORDER_BY_TYPE[t] ?? 1))));
  const incident = Array.from({ length: n }, () => []);
  bonds.forEach(([i, j], k) => { incident[i].push(k); incident[j].push(k); });
  const degree = incident.map((b) => b.filter((k) => bonds[k][2] !== BOND_TYPE.COORDINATION).length);
  // `calcExplicitValence`: the bonds' valence contributions, rounded after RDKit's +0.1.
  const explicit = incident.map((b) => Math.round(b.reduce((s, k) => s + contrib[k], 0) + 0.1));
  // `calcImplicitValence`: the hydrogens RDKit fills in, or -1 past every valence allowed.
  const implicit = (i, chgIn) => {
    const [z, dv, valences] = periodic(element[i]);
    if (dv === -1 || valences[valences.length - 1] === -1) return 0;
    let chg = chgIn;
    if (EARLY.has(element[i])) chg = -chg;
    if (z === 6 && chg > 0) chg = -chg;
    for (const v of valences) if (explicit[i] <= v + chg) return v + chg - explicit[i];
    return -1;
  };
  // An atom past every valence it is allowed fails RDKit's sanitisation. atomworks' `fix_mol`
  // then charges each such atom (valence - default valence) and keeps that only if every atom is
  // then legal - a four-bonded nitrogen becomes N+ - or else uses the molecule unsanitised, with
  // nothing perceived conjugated (a BODIPY's four-bonded, uncharged boron).
  charge = charge.map((c) => c ?? 0);
  const violators = [...Array(n).keys()].filter((i) => implicit(i, charge[i]) < 0);
  if (violators.length) {
    const fixed = charge.slice();
    for (const i of violators) fixed[i] = explicit[i] - periodic(element[i])[1];
    if ([...Array(n).keys()].some((i) => implicit(i, fixed[i]) < 0)) return bonds.map(() => false);
    charge = fixed;
  }
  const hydrogens = [...Array(n).keys()].map((i) => implicit(i, charge[i]));
  const electrons = (i) => {
    const [, dv, , nouter] = periodic(element[i]);
    if (dv <= 1) return -1;
    const deg = degree[i] + hydrogens[i];
    if (deg > 3) return -1;
    const nlp = Math.max(nouter - dv - (charge[i] ?? 0), 0);
    let res = (dv - deg) + nlp;
    if (res > 1 && explicit[i] - degree[i] > 1) res = 1;
    return res;
  };
  const candidate = (i) => {
    const [z, , , nouter] = periodic(element[i]);
    return (z <= 10 || (nouter !== 5 && nouter !== 6) || (nouter === 6 && z < 52)) && electrons(i) > 0;
  };
  const conj = bonds.map(([, , t]) => AROMATIC_TYPES.has(t));
  for (let i = 0; i < n; i += 1) {
    if (!candidate(i)) continue;
    const sbo = degree[i] + hydrogens[i];
    if (sbo < 2 || sbo > 3) continue;
    for (const b1 of incident[i]) {
      if (contrib[b1] < 1.5) continue;
      for (const b2 of incident[i]) {
        if (b1 === b2) continue;
        const other = bonds[b2][0] === i ? bonds[b2][1] : bonds[b2][0];
        if (degree[other] + hydrogens[other] > 3) continue;
        if (candidate(other)) { conj[b1] = true; conj[b2] = true; }
      }
    }
  }
  return conj;
}

/**
 * `_resonance_normalized_bond_orders`: each bond's label, equal across resonance forms.
 * @param {{ element: string[], charge: number[], bonds: number[][] }} unit local bonds `[i, j, type]`
 * @param {number[]} nHeavy whole-array heavy-atom degree of each unit atom
 */
export function normalizedBondOrders(unit, nHeavy) {
  const { element, charge, bonds } = unit;
  const raw = bonds.map(([, , t]) => bondOrder(t));
  const orders = raw.slice();
  const conj = conjugatedBonds(element, charge, bonds);
  bonds.forEach(([, , t], k) => {
    if (conj[k] && (raw[k] === 1 || raw[k] === 2 || AROMATIC_TYPES.has(t))) orders[k] = RESONANT;
  });
  const neighbours = new Map();
  bonds.forEach(([i, j], k) => {
    if (!neighbours.has(i)) neighbours.set(i, []);
    if (!neighbours.has(j)) neighbours.set(j, []);
    neighbours.get(i).push([j, k]);
    neighbours.get(j).push([i, k]);
  });
  for (const adjacent of neighbours.values()) {
    const byElement = new Map();
    for (const [nb, k] of adjacent) {
      if (nHeavy[nb] !== 1) continue;
      if (!byElement.has(element[nb])) byElement.set(element[nb], []);
      byElement.get(element[nb]).push(k);
    }
    for (const [el, ks] of byElement) {
      if (ks.length < 2 || el === 'C' || ks.some((k) => raw[k] === 3)) continue;
      for (const k of ks) orders[k] = RESONANT;
    }
  }
  return orders;
}

/**
 * Every automorphism of a coloured graph (`GraphMatcher(G, G).isomorphisms_iter()`), identity
 * first, at most `max` mappings examined (identity included, as atomworks counts them).
 * @param {Array<string|number>} colour per node
 * @param {number[][]} edges `[i, j, label]`
 * @returns {Int32Array[]} each `perm[i]` = the node i maps to
 */
export function automorphisms(colour, edges, max = 1000) {
  const n = colour.length;
  const label = new Map();
  const adj = Array.from({ length: n }, () => []);
  for (const [i, j, l] of edges) {
    label.set(i * n + j, l);
    label.set(j * n + i, l);
    adj[i].push(j);
    adj[j].push(i);
  }
  // Colour refinement: an automorphism preserves the stable colouring, so it bounds candidates.
  let cls = colour.map((c) => String(c));
  for (let round = 0; round < n; round += 1) {
    const sig = cls.map((c, i) => `${c}|${adj[i].map((j) => `${cls[j]}~${label.get(i * n + j)}`).sort().join(',')}`);
    const ids = new Map();
    const next = sig.map((s) => { if (!ids.has(s)) ids.set(s, `${ids.size}`); return ids.get(s); });
    const before = new Set(cls).size;
    cls = next;
    if (new Set(cls).size === before) break;
  }
  // Map nodes in breadth-first order so each new node has mapped neighbours to check against.
  const order = [];
  const seen = new Uint8Array(n);
  for (let s = 0; s < n; s += 1) {
    if (seen[s]) continue;
    seen[s] = 1;
    const queue = [s];
    while (queue.length) {
      const v = queue.shift();
      order.push(v);
      for (const w of adj[v]) if (!seen[w]) { seen[w] = 1; queue.push(w); }
    }
  }
  const out = [];
  const map = new Int32Array(n).fill(-1);
  const used = new Uint8Array(n);
  let examined = 0;
  const identity = Int32Array.from({ length: n }, (_, i) => i);
  const consistent = (v, w, depth) => {
    for (let d = 0; d < depth; d += 1) {
      const u = order[d];
      if (label.get(v * n + u) !== label.get(w * n + map[u])) return false;
    }
    return true;
  };
  const search = (depth) => {
    if (examined >= max) return;
    if (depth === n) {
      examined += 1;
      const perm = Int32Array.from(map);
      if (!perm.every((p, i) => p === i)) out.push(perm);
      return;
    }
    const v = order[depth];
    // `v` itself first, so the identity is the first mapping found, as VF2 finds it.
    for (let t = -1; t < n && examined < max; t += 1) {
      const w = t < 0 ? v : t;
      if ((t >= 0 && t === v) || used[w] || cls[w] !== cls[v] || !consistent(v, w, depth)) continue;
      map[v] = w;
      used[w] = 1;
      search(depth + 1);
      used[w] = 0;
      map[v] = -1;
    }
  };
  search(0);
  return [identity, ...out];
}

/** `count_heavy_neighbors`: non-hydrogen bond partners, over the whole array's bonds. */
export function heavyNeighbours(view) {
  const counts = new Int32Array(view.n);
  for (const [i, j] of view.bonds) {
    if (HYDROGEN_LIKE.has(view.element[i]) || HYDROGEN_LIKE.has(view.element[j])) continue;
    counts[i] += 1;
    counts[j] += 1;
  }
  return counts;
}

function residueIndex(view) {
  const idx = new Int32Array(view.n);
  let r = 0;
  for (let i = 1; i < view.n; i += 1) {
    if (view.chainId[i] !== view.chainId[i - 1] || view.resId[i] !== view.resId[i - 1]
      || view.insCode[i] !== view.insCode[i - 1]) r += 1;
    idx[i] = r;
  }
  return idx;
}

/**
 * `find_automorphisms_with_networkx` over a view: one entry per unit, `rows[k][m]` the global
 * atom that goes to `rows[0][m]` under automorphism k (row 0 the identity).
 * @param {object} view `{ n, element, charge, isPolymer, chainId, resId, insCode, pnUnit, bonds }`
 */
export function automorphismUnits(view, max = 1000) {
  const n = view.n;
  if (n === 0) return [];
  const heavy = heavyNeighbours(view);
  const res = residueIndex(view);
  // `build_automorphism_units`: an all-polymer pn_unit splits per residue, anything else is whole.
  const pnNonPolymer = new Map();
  for (let i = 0; i < n; i += 1) {
    if (!view.isPolymer[i]) pnNonPolymer.set(view.pnUnit[i], true);
  }
  const pnIds = [...new Set(view.pnUnit)].sort();
  const pnIndex = new Map(pnIds.map((p, k) => [p, k]));
  const unitOf = new Array(n);
  for (let i = 0; i < n; i += 1) {
    unitOf[i] = pnNonPolymer.get(view.pnUnit[i]) ? n + 1 + pnIndex.get(view.pnUnit[i]) : res[i];
  }
  const units = new Map();
  [...Array(n).keys()].sort((a, b) => unitOf[a] - unitOf[b] || a - b).forEach((i) => {
    if (!units.has(unitOf[i])) units.set(unitOf[i], []);
    units.get(unitOf[i]).push(i);
  });
  const out = [];
  for (const indices of units.values()) {
    const local = new Map(indices.map((g, k) => [g, k]));
    const ures = indices.map((g) => res[g]);
    const single = ures[0] === ures[ures.length - 1];
    const rankOf = new Map([...new Set(ures)].sort((a, b) => a - b).map((r, k) => [r, k]));
    const rank = indices.map((g, k) => (view.isPolymer[g] ? (single ? 0 : rankOf.get(ures[k])) : -1));
    const bonds = [];
    for (const [i, j, t] of view.bonds) {
      if (local.has(i) && local.has(j)) bonds.push([local.get(i), local.get(j), t]);
    }
    const unit = { element: indices.map((g) => view.element[g]), charge: indices.map((g) => view.charge[g] ?? 0), bonds };
    const orders = normalizedBondOrders(unit, indices.map((g) => heavy[g]));
    const colour = indices.map((g, k) => `${view.element[g]}/${heavy[g]}/${rank[k]}`);
    const perms = automorphisms(colour, bonds.map(([i, j], k) => [i, j, orders[k]]), max);
    out.push(perms.map((p) => Int32Array.from(p, (q) => indices[q])));
  }
  return out;
}

/** Fit `pred` onto `gt` over `frame`, and return every pred atom moved by that fit. */
function superimposeOn(pred, gt, frame) {
  const idx = [];
  for (let i = 0; i < frame.length; i += 1) if (frame[i]) idx.push(i);
  const fixed = new Float64Array(idx.length * 3);
  const mobile = new Float64Array(idx.length * 3);
  idx.forEach((i, k) => {
    for (let c = 0; c < 3; c += 1) { fixed[k * 3 + c] = gt[i * 3 + c]; mobile[k * 3 + c] = pred[i * 3 + c]; }
  });
  const { rotation: R, fixedCenter: fc, mobileCenter: mc } = superimposeCoords3D(fixed, mobile);
  const out = new Float64Array(pred.length);
  for (let i = 0; i < pred.length / 3; i += 1) {
    const x = pred[i * 3] - mc[0];
    const y = pred[i * 3 + 1] - mc[1];
    const z = pred[i * 3 + 2] - mc[2];
    out[i * 3] = x * R[0] + y * R[1] + z * R[2] + fc[0];
    out[i * 3 + 1] = x * R[3] + y * R[4] + z * R[5] + fc[1];
    out[i * 3 + 2] = x * R[6] + y * R[7] + z * R[8] + fc[2];
  }
  return out;
}

const finiteAt = (c, i) => Number.isFinite(c[i * 3]) && Number.isFinite(c[i * 3 + 1]) && Number.isFinite(c[i * 3 + 2]);

/**
 * `atom_permutation`: the ground-truth labelling that best matches the prediction, unit by unit.
 * @returns {Int32Array} `perm` such that `gt[perm]` is the resolved ground truth
 */
export function atomPermutation(pred, gt, gtMask, units, { atomClasses = null, frameMask = null } = {}) {
  const n = gtMask.length;
  const perm = Int32Array.from({ length: n }, (_, i) => i);
  let frame = frameMask ?? gtMask.map((m, i) => m && (!atomClasses || atomClasses[i] >= 0));
  frame = frame.map((f, i) => f && gtMask[i]);
  let x = pred;
  if (frame.filter(Boolean).length >= 3) x = superimposeOn(pred, gt, frame);
  for (const rows of units) {
    if (rows.length <= 1 || rows[0].length === 0) continue;
    const dest = rows[0];
    const m = dest.length;
    const moves = new Array(m).fill(false);
    for (let k = 0; k < m; k += 1) moves[k] = rows.some((r) => r[k] !== dest[k]);
    const active = Array.from({ length: m }, (_, k) => gtMask[dest[k]] && moves[k]
      && (!atomClasses || atomClasses[dest[k]] >= 0));
    const nActive = Math.max(active.filter(Boolean).length, 1);
    let best = 0;
    let bestErr = Infinity;
    rows.forEach((src, c) => {
      let legal = true;
      let err = 0;
      for (let k = 0; k < m; k += 1) {
        if (!active[k]) continue;
        if (!gtMask[src[k]] || (atomClasses && atomClasses[src[k]] !== atomClasses[dest[k]])) { legal = false; break; }
        const d = dest[k];
        const s = src[k];
        const dx = x[d * 3] - gt[s * 3];
        const dy = x[d * 3 + 1] - gt[s * 3 + 1];
        const dz = x[d * 3 + 2] - gt[s * 3 + 2];
        err += dx * dx + dy * dy + dz * dz;
      }
      err = legal && Number.isFinite(err) ? err / nActive : Infinity;
      if (err < bestErr) { bestErr = err; best = c; }
    });
    const chosen = rows[best];
    for (let k = 0; k < m; k += 1) if (active[k]) perm[dest[k]] = chosen[k];
  }
  return perm;
}

/**
 * `permute_gt_atoms`: a copy of `gt` with its symmetric atoms relabelled onto `pred`.
 * @param {Float64Array|Float32Array} pred flat coordinates
 * @param {Float64Array|Float32Array} gt flat coordinates
 * @param {object} view the ground truth's atoms and bonds (`automorphismUnits`)
 */
export function permuteGtAtoms(pred, gt, view, { atomClasses = null, frameMask = null } = {}) {
  const n = view.n;
  if (atomClasses) {
    const counts = new Map();
    for (const c of atomClasses) if (c >= 0) counts.set(c, (counts.get(c) ?? 0) + 1);
    if (Math.max(0, ...counts.values()) < 2) return gt;
  }
  const units = automorphismUnits(view);
  if (!units.some((u) => u.length > 1)) return gt;
  const gtMask = Array.from({ length: n }, (_, i) => finiteAt(gt, i));
  if (!gtMask.some(Boolean)) return gt;
  const perm = atomPermutation(pred, gt, gtMask, units, { atomClasses, frameMask });
  const out = new Float64Array(gt.length);
  for (let i = 0; i < n; i += 1) for (let c = 0; c < 3; c += 1) out[i * 3 + c] = gt[perm[i] * 3 + c];
  return out;
}

function* permutationsOf(items) {
  if (items.length <= 1) { yield items.slice(); return; }
  for (let k = 0; k < items.length; k += 1) {
    const rest = items.slice(0, k).concat(items.slice(k + 1));
    for (const p of permutationsOf(rest)) yield [items[k], ...p];
  }
}

/**
 * `_subunit_candidates`: destination-to-source atom maps that exchange identical molecules,
 * identity first. Molecules are the view's `molIid`; two are interchangeable when they share an
 * entity and the same chain and residue boundaries and ordered atoms, and carry the same mask
 * classes.
 */
function subunitCandidates(view, classes, limit) {
  const n = view.n;
  const identity = Int32Array.from({ length: n }, (_, i) => i);
  const molecules = new Map();
  for (let i = 0; i < n; i += 1) {
    if (!molecules.has(view.molIid[i])) molecules.set(view.molIid[i], []);
    molecules.get(view.molIid[i]).push(i);
  }
  const res = residueIndex(view);
  const groups = new Map();
  for (const indices of molecules.values()) {
    if (!indices.some((i) => classes[i])) continue;
    const chainStarts = [];
    const resStarts = [];
    indices.forEach((g, k) => {
      if (k === 0 || view.chainIid[g] !== view.chainIid[indices[k - 1]]) chainStarts.push(k);
      if (k === 0 || res[g] !== res[indices[k - 1]]) resStarts.push(k);
    });
    const signature = JSON.stringify([chainStarts, resStarts, indices.map((g) => view.resName[g]),
      indices.map((g) => view.atomName[g]), indices.map((g) => view.element[g])]);
    const key = JSON.stringify([view.molEntity[indices[0]], signature, indices.map((g) => classes[g])]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(indices);
  }
  let count = 1;
  for (const g of groups.values()) {
    for (let f = 2; f <= g.length; f += 1) {
      count *= f;
      if (count > limit) {
        throw new Error(`Subunit symmetry above ${limit} permutations is not supported, as upstream's is not`);
      }
    }
  }
  if (count === 1) return [identity];
  // `get_isomorphisms_from_symmetry_groups`: the Cartesian product of each group's orderings.
  let assignments = [[]];
  for (const g of groups.values()) {
    const orders = [...permutationsOf(g.map((_, k) => k))];
    const next = [];
    for (const o of orders) for (const a of assignments) next.push([...a, o.map((k) => g[k])]);
    assignments = next;
  }
  const slots = [...groups.values()].flat();
  return assignments.map((assignment) => {
    const cand = Int32Array.from(identity);
    assignment.flat().forEach((src, s) => {
      const dst = slots[s];
      for (let k = 0; k < dst.length; k += 1) cand[dst[k]] = src[k];
    });
    return cand;
  });
}

/**
 * `permute_gt_subunits`: relabel identical molecules of `gt` onto `pred`, by the least squared
 * error over `scoreMask` after a fit on `frameMask`. Coordinates stay in the ground truth's frame.
 */
export function permuteGtSubunits(pred, gt, view, { frameMask = null, scoreMask = null, align = true,
  maxPermutations = 720 } = {}) {
  const n = view.n;
  const frame = frameMask ?? new Array(n).fill(true);
  const score = scoreMask ?? new Array(n).fill(true);
  const classes = Array.from({ length: n }, (_, i) => (align && frame[i] ? 1 : 0) + 2 * (score[i] ? 1 : 0));
  const candidates = subunitCandidates(view, classes, maxPermutations);
  if (candidates.length === 1) return gt;
  const valid = Array.from({ length: n }, (_, i) => finiteAt(gt, i));
  const observed = valid.map((v, i) => v && finiteAt(pred, i));
  const fit = frame.map((f, i) => f && observed[i]);
  const scored = score.map((s, i) => s && observed[i]);
  if (!scored.some(Boolean) || (align && !fit.some(Boolean))) return gt;
  let best = candidates[0];
  let bestErr = Infinity;
  for (const cand of candidates) {
    if (!cand.every((s, i) => valid[s] === valid[i])) continue;
    const target = new Float64Array(n * 3);
    for (let i = 0; i < n; i += 1) for (let c = 0; c < 3; c += 1) target[i * 3 + c] = gt[cand[i] * 3 + c];
    const fitted = align ? superimposeOn(pred, target, fit) : pred;
    let err = 0;
    let count = 0;
    for (let i = 0; i < n; i += 1) {
      if (!scored[i]) continue;
      count += 1;
      for (let c = 0; c < 3; c += 1) {
        const d = fitted[i * 3 + c] - target[i * 3 + c];
        err += d * d;
      }
    }
    err /= count;
    if (err < bestErr) { bestErr = err; best = cand; }
  }
  const out = new Float64Array(gt.length);
  for (let i = 0; i < n; i += 1) for (let c = 0; c < 3; c += 1) out[i * 3 + c] = gt[best[i] * 3 + c];
  return out;
}
