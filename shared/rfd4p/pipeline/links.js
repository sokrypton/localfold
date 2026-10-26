/**
 * atomworks' link chemistry (atomworks/io/utils/link_chemistry.py), which upstream runs when it
 * PREPARES a raw structure (`RCSB_WITH_METALS`, `add_missing_atoms`): an authored inter-residue
 * bond turns two dictionary reactants into one product, and the product is built and checked.
 *
 * 🔴 IT CHANGES WHAT THE MODEL SEES. 5O45's macrocycle closes through CCS CE, whose dictionary
 * entry has no leaving atom there, so the carboxylate's C=O becomes C-O(-); 1DPN's TAF phosphate
 * keeps its P=O by moving it onto OP1; 1QFE's DHS-LYS amide drops a hydrogen and a charge; and a
 * product with no legal valence (1U9X, 2ZZ6, 8QIA) is refused. This port loaded every one of them
 * as two untouched reactants.
 *
 * The functions follow upstream's one for one and are named after them. Hydrogens are atoms here,
 * as they are upstream at this point (`hydrogen_policy="keep"`): the prepared structure carries
 * every dictionary hydrogen, resolved or not, until the link products are built. Bonds keep
 * biotite's row order (`BondRows`), because upstream takes the first match in several places.
 */
import { CCD_TEMPLATES, WATER_LIKE_CCDS } from './ccd-templates.js';
import { RDKIT_VALENCE, RDKIT_DEFAULT_VALENCE } from './valence.js';

// biotite BondType values (cif.js `BOND_TYPE`, which imports this module).
const ANY = 0;
const SINGLE = 1;
const DOUBLE = 2;
const COORDINATION = 8;

/** atomworks `BIOTITE_BOND_TYPE_TO_BOND_ORDER`, by BondType value; `.get(type, 1)`. */
const ORDER_OF = [1, 1, 2, 3, 4, 1, 2, 3, 0];
export const bondOrder = (type) => ORDER_OF[type] ?? 1;
/** `_BOND_ORDERS`, the weights `get_bond_degree_per_atom` sums: `.get(type, int(type))`. */
const degreeWeight = (type) => ORDER_OF[type] ?? type;

/** atomworks `DEFAULT_VALENCE`, looked up by an UPPERCASE element: its `Cl` and `Br` never match. */
const DEFAULT_VALENCE = { H: 1, C: 4, N: 3, O: 2, F: 1, B: 3 };
const HYDROGEN_LIKE = new Set(['H', 'H2', 'D', 'T']);
const isHydrogen = (element) => HYDROGEN_LIKE.has(element);

/** `_LINK_PRODUCT_CHARGES`: (element, charge, valence) -> charge. */
const LINK_PRODUCT_CHARGES = new Map([
  ['B,0,4', -1], ['C,1,4', 0], ['N,0,4', 1], ['O,-1,2', 0], ['O,0,1', -1],
  ['P,0,4', 1], ['S,0,3', 1], ['SE,0,1', -1], ['SE,0,3', 1],
]);

/** `_is_valid_rdkit_valence`: RDKit parses `[<El>H<valence><charge>]` with no radical. */
export function isValidRdkitValence(element, charge, valence) {
  const per = RDKIT_VALENCE[String(element).toUpperCase()];
  if (!per) return false;
  if (charge < -4 || charge > 4 || valence < 0 || valence > 16) {
    throw new Error(`RDKit valence probe [${element}H${valence}${charge}] is outside the generated table (valence.js)`);
  }
  const ok = per[String(charge)];
  return ok === 'any' || (Array.isArray(ok) && ok.includes(valence));
}

/** RDKit's `GetDefaultValence`, which raises for an element it does not know. */
function rdkitDefaultValence(element) {
  const v = RDKIT_DEFAULT_VALENCE[String(element).toUpperCase()];
  if (v === undefined) throw new Error(`Element '${element}' not found`);
  return v;
}

const DO_NOT_MATCH_CCD = new Set([...WATER_LIKE_CCDS, 'UNL']);

/**
 * A component as atomworks' `atom_array_from_ccd_code` reads it - every atom in the dictionary's
 * order, hydrogens included, and its bonds as biotite types - from the generated templates, else
 * a fetched entry (`ccdBlocks`). Null where neither has it, as the dictionary raises there.
 */
export function componentChemistry(code, ccdBlocks = null, cache = new Map()) {
  if (cache.has(code)) return cache.get(code);
  let out = null;
  const t = CCD_TEMPLATES[code];
  if (t) {
    out = {
      names: t.atoms.map((a) => a[0]), element: t.atoms.map((a) => String(a[2]).toUpperCase()),
      charge: t.atoms.map((a) => a[3]), leaving: t.atoms.map((a) => Boolean(a[4])),
      bonds: t.bonds.map((b) => b.slice()),
    };
  } else if (ccdBlocks?.get?.(code)) {
    out = chemistryFromBlock(ccdBlocks.get(code), code);
  }
  cache.set(code, out);
  return out;
}

const COMP_BOND = { 'SING:N': 1, 'DOUB:N': 2, 'TRIP:N': 3, 'QUAD:N': 4, 'SING:Y': 5, 'DOUB:Y': 6, 'TRIP:Y': 7, 'AROM:Y': 9, 'AROM:N': 9 };
const col = (cat, key) => {
  const v = cat?.[key];
  return v === undefined ? [] : Array.isArray(v) ? v : [v];
};
const unset = (v) => v === undefined || v === null || v === '.' || v === '?' || v === '';

function chemistryFromBlock(block, code) {
  const cat = block.chem_comp_atom;
  const rows = col(cat, 'comp_id').map((c, r) => [c, r]).filter(([c]) => c === code).map(([, r]) => r);
  if (!rows.length) return null;
  const names = rows.map((r) => col(cat, 'atom_id')[r]);
  const index = new Map(names.map((n, k) => [n, k]));
  const bc = block.chem_comp_bond;
  const bonds = [];
  col(bc, 'comp_id').forEach((c, r) => {
    if (c !== code) return;
    const i = index.get(col(bc, 'atom_id_1')[r]);
    const j = index.get(col(bc, 'atom_id_2')[r]);
    if (i === undefined || j === undefined) return;
    const key = `${String(col(bc, 'value_order')[r] || 'SING').toUpperCase()}:${String(col(bc, 'pdbx_aromatic_flag')[r] || 'N').toUpperCase()}`;
    bonds.push([Math.min(i, j), Math.max(i, j), COMP_BOND[key] ?? ANY]);
  });
  return {
    names,
    element: rows.map((r) => String(col(cat, 'type_symbol')[r]).toUpperCase()),
    charge: rows.map((r) => (unset(col(cat, 'charge')[r]) ? 0 : Number.parseInt(col(cat, 'charge')[r], 10) || 0)),
    leaving: rows.map((r) => col(cat, 'pdbx_leaving_atom_flag')[r] === 'Y'),
    bonds,
  };
}

/** A component's atoms' bonds, `[neighbour, type]` in the template's row order. */
function templateBonds(comp, k) {
  const out = [];
  for (const [i, j, t] of comp.bonds) {
    if (i === k) out.push([j, t]);
    else if (j === k) out.push([i, t]);
  }
  return out;
}

/**
 * `get_leaving_atom_groups` / `_leaving_groups_from_topology`: for each atom, the groups of atoms
 * that would come away with it removed and are all leaving (or, for a group of hydrogens only,
 * all flagged), in the dictionary's order.
 */
export function leavingGroups(comp) {
  if (!comp || !comp.leaving.some(Boolean)) return new Map();
  const n = comp.names.length;
  const adj = Array.from({ length: n }, () => []);
  // networkx adjacency: neighbours in the order their edges were added.
  const seen = Array.from({ length: n }, () => new Set());
  for (const [i, j] of comp.bonds) {
    if (!seen[i].has(j)) { seen[i].add(j); adj[i].push(j); }
    if (!seen[j].has(i)) { seen[j].add(i); adj[j].push(i); }
  }
  const out = new Map();
  for (let atom = 0; atom < n; atom++) {
    const groups = [];
    const unvisited = new Set(adj[atom]);
    while (unvisited.size) {
      const start = unvisited.values().next().value;
      unvisited.delete(start);
      const visited = new Set([atom, start]);
      const queue = [start];
      const component = [];
      while (queue.length) {
        const node = queue.shift();
        component.push(node);
        for (const nb of adj[node]) if (!visited.has(nb)) { visited.add(nb); queue.push(nb); }
      }
      for (const c of component) unvisited.delete(c);
      const heavy = component.filter((x) => comp.element[x] !== 'H');
      const leaving = heavy.length ? heavy.every((x) => comp.leaving[x]) : component.every((x) => comp.leaving[x]);
      if (leaving) groups.push(component.slice().sort((a, b) => a - b));
    }
    if (groups.length) {
      groups.sort((a, b) => a[0] - b[0]);
      out.set(comp.names[atom], groups.map((g) => g.map((x) => comp.names[x])));
    }
  }
  return out;
}

/**
 * The dictionary lookups link chemistry makes, cached for one structure: a component, its
 * leaving groups (`get_chem_comp_leaving_atom_groups`, empty for a component the dictionary does
 * not match) and the type of the bond an atom loses with them (`get_leaving_atom_bond_type`).
 * `polymerizationAtoms(resName, chainType)` is cif.js's `getPolymerizationAtoms`.
 */
export function linkDictionary(ccdBlocks, polymerizationAtoms) {
  const comps = new Map();
  const groups = new Map();
  const comp = (code) => componentChemistry(code, ccdBlocks, comps);
  const leaving = (code) => {
    if (!groups.has(code)) groups.set(code, DO_NOT_MATCH_CCD.has(code) ? new Map() : leavingGroups(comp(code)));
    return groups.get(code);
  };
  const leavingBondType = (code, atomName) => {
    const g = leaving(code).get(atomName);
    const c = comp(code);
    if (!g || !c) return null;
    const k = c.names.indexOf(atomName);
    if (k < 0) return null;
    const set = new Set(g.flat());
    for (const [nb, t] of templateBonds(c, k)) if (set.has(c.names[nb]) && t !== SINGLE) return t;
    return null;
  };
  return { comp, leaving, leavingBondType, polymerizationAtoms };
}

/**
 * `infer_link_order`: the order of an authored link whose `pdbx_value_order` is unset. The CCD's
 * own order for a bond inside one residue; else the higher of the two partners' leaving-bond
 * TYPES (upstream takes `int(BondType)`, so an aromatic double leaving bond is 6, as there),
 * capped by what each partner can still take.
 */
export function inferLinkOrder(atoms, a1, a2, dict, { allowMissingTemplates = false } = {}) {
  const partners = [a1, a2].map((i) => [atoms.res_name[i], atoms.atom_name[i]]);
  const show = `[${partners.map(([r, a]) => `('${r}', '${a}')`).join(', ')}]`;
  const keys = ['chain_id', 'res_id', 'ins_code', 'res_name', 'transformation_id'].filter((k) => Array.isArray(atoms[k]));
  const sameResidue = keys.every((k) => atoms[k][a1] === atoms[k][a2]);
  let order = Math.max(...partners.map(([r, a]) => dict.leavingBondType(r, a) || 1));
  for (const [resName, atomName] of partners) {
    const t = dict.comp(resName);
    if (!t) {
      if (allowMissingTemplates) return ANY;
      throw new Error(`Cannot infer link order for ${show}: missing CCD template for ${resName}`);
    }
    const matches = t.names.flatMap((n, k) => (n === atomName ? [k] : []));
    if (matches.length !== 1) {
      throw new Error(`Cannot infer link order for ${show}: missing CCD atom/bonds for ${resName}/${atomName}`);
    }
    const nbrs = templateBonds(t, matches[0]);
    if (sameResidue) {
      const existing = nbrs.find(([nb]) => t.names[nb] === partners[1][1]);
      if (existing) return existing[1];
    }
    if (order === 1) continue;
    const leaving = new Set((dict.leaving(resName).get(atomName) ?? []).flat());
    let all = 0;
    let retained = 0;
    for (const [nb, type] of nbrs) {
      all += bondOrder(type);
      if (!isHydrogen(t.element[nb]) && !leaving.has(t.names[nb])) retained += bondOrder(type);
    }
    const element = t.element[matches[0]];
    let capacity = Math.max(all, rdkitDefaultValence(element)) - retained;
    if (capacity < order && isValidRdkitValence(element, t.charge[matches[0]], retained + order)) capacity = order;
    order = Math.min(order, capacity);
  }
  if (order < 1) {
    throw new Error(`Cannot infer link order for ${show}: retained heavy-atom bonds leave no valence; provide explicit product connectivity`);
  }
  return order;
}

/**
 * biotite's BondList as far as link chemistry uses it: rows `[i, j, type]` with `i < j`, kept in
 * their order. Updating a bond keeps its row; removing one and adding it back puts it last, as
 * `remove_bond` then `add_bond` does upstream.
 */
export class BondRows {
  constructor(rows) {
    this.rows = new Map();
    this.adj = new Map();
    this.next = 0;
    this.version = 0;
    for (const [i, j, t] of rows) this.add(i, j, t);
  }

  static key(i, j) { return i < j ? `${i},${j}` : `${j},${i}`; }

  add(i, j, t) {
    const k = BondRows.key(i, j);
    const row = this.rows.get(k);
    this.version += 1;
    if (row) {
      row[2] = t;
      return;
    }
    const r = [Math.min(i, j), Math.max(i, j), t, this.next++];
    this.rows.set(k, r);
    for (const x of [i, j]) {
      if (!this.adj.has(x)) this.adj.set(x, new Set());
      this.adj.get(x).add(r);
    }
  }

  remove(i, j) {
    const k = BondRows.key(i, j);
    const r = this.rows.get(k);
    if (!r) return;
    this.version += 1;
    this.rows.delete(k);
    this.adj.get(i).delete(r);
    this.adj.get(j).delete(r);
  }

  type(i, j) { return this.rows.get(BondRows.key(i, j))?.[2]; }

  /** Rows in order. */
  list() { return [...this.rows.values()].sort((a, b) => a[3] - b[3]); }

  rowsOf(i) { return [...(this.adj.get(i) ?? [])].sort((a, b) => a[3] - b[3]); }

  /** `get_bonds(i)`: `[neighbour, type]` in row order. */
  of(i) { return this.rowsOf(i).map((r) => [r[0] === i ? r[1] : r[0], r[2]]); }

  /** `_find_bonded_hydrogens`: hydrogens in the first column's rows first, then the second's. */
  hydrogensOf(i, element) {
    const rows = this.rowsOf(i);
    return [...rows.filter((r) => r[0] === i).map((r) => r[1]), ...rows.filter((r) => r[1] === i).map((r) => r[0])]
      .filter((x) => isHydrogen(element[x]));
  }

  degrees(n) {
    const d = new Float64Array(n);
    for (const [i, j, t] of this.rows.values()) { d[i] += degreeWeight(t); d[j] += degreeWeight(t); }
    return d;
  }
}

/** `_validate_inferred_polymer_bond`: each end of a backbone bond must have room for it. */
export function validateInferredPolymerBond(atoms, i, j, degree, hydrogens, dict) {
  for (const idx of [i, j]) {
    const groups = dict.leaving(atoms.res_name[idx]);
    const expected = (DEFAULT_VALENCE[atoms.element[idx]] ?? 0) + atoms.charge[idx];
    if (degree[idx] >= expected && !groups.get(atoms.atom_name[idx])?.length && !hydrogens(idx)) {
      throw new Error(`Cannot infer polymer bond at ${atoms.res_name[idx]}/${atoms.atom_name[idx]}: `
        + 'no free valence or declared leaving group');
    }
  }
}

/**
 * `resolve_link_chemistry`: build the products of `atoms`' authored inter-residue bonds, apply
 * the charges they imply and validate every affected atom. Returns the array without the atoms
 * that left; raises where upstream raises.
 *
 * @param {object} atoms an atom array with hydrogens and `bonds` in upstream's row order
 * @param {object} dict `linkDictionary(...)`
 * @param {(atoms: object, keep: boolean[]) => object} keepAtoms the array without the dropped atoms
 */
export function resolveLinkChemistry(atoms, dict, keepAtoms) {
  const L = new LinkState(atoms, dict);
  const interBonds = L.interResidueBonds();
  if (!interBonds.length) return atoms;
  L.removeCcdLeavingAtoms(interBonds);
  L.restoreLeavingBondOrdersOrRaise();
  const impacted = L.resolveOvervalentAtoms();
  if (!impacted.some((x, i) => x && L.alive[i])) return L.result(keepAtoms);
  L.applyLinkProductCharges(impacted);
  L.normalizeLinkedAmides(impacted);
  L.validateLinkValence(impacted);
  return L.result(keepAtoms);
}

/**
 * The working state: the atoms, alive or gone (upstream compacts the array after each phase; here
 * an atom that leaves is marked and its bonds removed, which is the same thing), and the bonds.
 */
class LinkState {
  constructor(atoms, dict) {
    this.a = atoms;
    this.n = atoms.nAtoms;
    this.dict = dict;
    this.bonds = new BondRows(atoms.bonds);
    this.charge = atoms.charge.slice();
    this.alive = new Array(this.n).fill(true);
    // Residue position of each atom: biotite's residue starts.
    this.residue = new Int32Array(this.n);
    let r = 0;
    for (let i = 1; i < this.n; i++) {
      if (['chain_id', 'res_id', 'ins_code', 'res_name'].some((k) => atoms[k][i] !== atoms[k][i - 1])
        || (atoms.transformation_id && atoms.transformation_id[i] !== atoms.transformation_id[i - 1])) r += 1;
      this.residue[i] = r;
    }
  }

  el(i) { return this.a.element[i]; }

  kill(i) {
    this.alive[i] = false;
    for (const [nb] of this.bonds.of(i)) this.bonds.remove(i, nb);
  }

  degrees() { return this.bonds.degrees(this.n); }

  /** `_has_valid_rdkit_valence`, over the bonds as they stand. */
  validValence(i) {
    let v = 0;
    for (const [, t] of this.bonds.of(i)) v += bondOrder(t);
    return isValidRdkitValence(this.el(i), this.charge[i], v);
  }

  overvalent(i, degree) {
    let expected = DEFAULT_VALENCE[this.el(i)];
    if (expected === undefined) return false;
    if (this.el(i) !== 'C') expected += this.charge[i];
    return degree[i] > expected;
  }

  /** `_get_inter_residue_bonds`: non-coordination bonds between residues, in row order. */
  interResidueBonds() {
    const keys = ['chain_id', 'res_id', 'ins_code', ...(this.a.transformation_id ? ['transformation_id'] : [])];
    return this.bonds.list().filter(([i, j, t]) => t !== COORDINATION && keys.some((k) => this.a[k][i] !== this.a[k][j]));
  }

  interMask() {
    const m = new Array(this.n).fill(false);
    for (const [i, j] of this.interResidueBonds()) { m[i] = true; m[j] = true; }
    return m;
  }

  /** `_displace_hydrogens`: detach `count` hydrogens, unresolved ones first. */
  displaceHydrogens(i, count, removed) {
    const coordOk = (h) => this.a.coord[h].every(Number.isFinite);
    const explicit = this.bonds.hydrogensOf(i, this.a.element);
    const ordered = [...explicit.filter((h) => !coordOk(h)), ...explicit.filter(coordOk)].slice(0, Math.max(0, count));
    for (const h of ordered) { this.bonds.remove(i, h); removed.add(h); }
    return ordered.length;
  }

  /** `_remove_ccd_leaving_atoms`, with resolved atoms preserved (the default mask). */
  removeCcdLeavingAtoms(interBonds) {
    const a = this.a;
    const withInter = [...new Set(interBonds.flatMap(([i, j]) => [i, j]))].sort((x, y) => x - y);
    const interMask = new Array(this.n).fill(false);
    for (const i of withInter) interMask[i] = true;
    const polymerLink = new Array(this.n).fill(false);
    for (const [i, j] of interBonds) {
      const s = [i, j].map((x) => this.dict.polymerizationAtoms(a.res_name[x], a.chain_type?.[x]));
      const pair = [a.atom_name[i], a.atom_name[j]];
      if ((pair[0] === s[0][0] && pair[1] === s[1][1]) || (pair[0] === s[0][1] && pair[1] === s[1][0])) {
        polymerLink[i] = true;
        polymerLink[j] = true;
      }
    }
    const displaceable = new Float64Array(this.n);
    for (const [i, j, t] of interBonds) { displaceable[i] += bondOrder(t); displaceable[j] += bondOrder(t); }
    const preserve = (i) => a.coord[i].every(Number.isFinite);
    const resAtoms = new Map();
    for (let i = 0; i < this.n; i++) if (this.alive[i]) {
      if (!resAtoms.has(this.residue[i])) resAtoms.set(this.residue[i], []);
      resAtoms.get(this.residue[i]).push(i);
    }
    const leaving = new Array(this.n).fill(false);
    let current = -1;
    let present;
    let preserved;
    let linked;
    for (const i of withInter) {
      const groups = this.dict.leaving(a.res_name[i]).get(a.atom_name[i]) ?? [];
      if (!groups.length) continue;
      const members = resAtoms.get(this.residue[i]);
      if (this.residue[i] !== current) {
        present = new Set(members.map((x) => a.atom_name[x]));
        preserved = new Set(members.filter(preserve).map((x) => a.atom_name[x]));
        linked = new Set(members.filter((x) => interMask[x]).map((x) => a.atom_name[x]));
        current = this.residue[i];
      }
      const attached = groups.filter((g) => g.some((nm) => present.has(nm)));
      const removable = attached.filter((g) => (polymerLink[i] || !g.some((nm) => preserved.has(nm)))
        && !g.some((nm) => linked.has(nm)));
      const budget = Math.max(0, displaceable[i] - (groups.length - attached.length));
      for (const g of removable.slice(0, budget)) {
        const set = new Set(g);
        for (const x of members) if (set.has(a.atom_name[x])) leaving[x] = true;
      }
    }
    if (!leaving.some(Boolean)) return;
    // `_validate_leaving_atoms_have_no_unexpected_bonds`, each column in turn.
    const rows = this.bonds.list();
    for (const [lc, oc] of [[0, 1], [1, 0]]) {
      for (const r of rows) {
        if (!leaving[r[lc]] || interMask[r[oc]] || leaving[r[oc]]) continue;
        const l = r[lc];
        const o = r[oc];
        throw new Error(`Leaving atom '${a.atom_name[l]}' in residue ${a.res_name[l]} (index ${this.index(l)}) is bonded to `
          + `'${a.atom_name[o]}' (index ${this.index(o)}), which is neither a parent atom making inter-residue bonds nor another `
          + 'leaving atom. Cannot safely remove.');
      }
    }
    for (let i = 0; i < this.n; i++) if (leaving[i]) this.kill(i);
  }

  /** An atom's index in upstream's array as it stands: the living atoms before it. */
  index(i) {
    let k = 0;
    for (let j = 0; j < i; j++) if (this.alive[j]) k += 1;
    return k;
  }

  /** `_restore_leaving_bond_orders_or_raise`. */
  restoreLeavingBondOrdersOrRaise() {
    const a = this.a;
    const gained = new Float64Array(this.n);
    for (const [i, j, t] of this.interResidueBonds()) { gained[i] += bondOrder(t); gained[j] += bondOrder(t); }
    const degree = this.degrees();
    const removed = new Set();
    for (let i = 0; i < this.n; i++) {
      if (!gained[i]) continue;
      const type = this.dict.leavingBondType(a.res_name[i], a.atom_name[i]);
      const lost = type === null ? 0 : bondOrder(type);
      if (lost <= gained[i]) continue;
      const t = this.dict.comp(a.res_name[i]);
      const k = t.names.indexOf(a.atom_name[i]);
      const expected = templateBonds(t, k).reduce((s, [, bt]) => s + degreeWeight(bt), 0);
      const where = `${a.res_name[i]}/${a.atom_name[i]}`;
      const times = Math.min(lost - gained[i], expected - degree[i]);
      for (let step = 0; step < times; step++) {
        const candidates = [];
        for (const [nb, bt] of this.bonds.of(i)) {
          if (bt !== SINGLE || this.residue[nb] !== this.residue[i] || !['O', 'S', 'N'].includes(this.el(nb))) continue;
          const heavy = this.bonds.of(nb).map(([x]) => x).filter((x) => !isHydrogen(this.el(x)));
          const hasH = this.bonds.hydrogensOf(nb, a.element).length > 0;
          const charge = this.charge[nb] === -1 ? 0 : this.charge[nb];
          const valence = degree[nb] + 1 - (hasH ? 1 : 0);
          if (heavy.length === 1 && heavy[0] === i && (hasH || this.charge[nb] === -1)
            && isValidRdkitValence(this.el(nb), charge, valence)) {
            candidates.push({ state: `${this.el(nb)},${charge},${valence}`, nb, hasH, charge });
          }
        }
        if (!candidates.length) throw new Error(`Unsupported bond-order rearrangement at ${where}`);
        if (new Set(candidates.map((c) => c.state)).size !== 1) throw new Error(`Ambiguous bond-order rearrangement at ${where}`);
        const pick = candidates.find((c) => !c.hasH) ?? candidates[0];
        if (pick.hasH) this.displaceHydrogens(pick.nb, 1, removed);
        this.bonds.remove(i, pick.nb);
        this.bonds.add(i, pick.nb, DOUBLE);
        this.charge[pick.nb] = pick.charge;
        degree[i] += 1;
        degree[pick.nb] += pick.hasH ? 0 : 1;
      }
      if (degree[i] < expected) {
        throw new Error(`Unsupported bond-order rearrangement at ${where}: `
          + 'provide explicit product bond orders instead of inferring a replacement double bond');
      }
    }
    for (const h of removed) this.kill(h);
  }

  /** `_lower_multiple_bond`. */
  lowerMultipleBond(i, degree, requireOvervalentPartner, impacted) {
    for (const [p, bt] of this.bonds.of(i)) {
      const order = bondOrder(bt);
      const over = this.overvalent(p, degree);
      if (order <= 1 || !['C', 'N'].includes(this.el(p)) || this.residue[p] !== this.residue[i]
        || (requireOvervalentPartner && !over)) continue;
      // Without implicit hydrogens (`nhyd`) no donor can hand one over, so an addition to a
      // partner that is not itself overvalent is not a product here, as upstream.
      if (!over) continue;
      this.bonds.remove(i, p);
      this.bonds.add(i, p, order - 1);
      degree[i] -= 1;
      degree[p] -= 1;
      impacted[i] = true;
      impacted[p] = true;
      return true;
    }
    return false;
  }

  /** `_maybe_fix_overvalent_carbon`: returns a terminal group for the caller to delete. */
  maybeFixOvervalentCarbon(i, degree, removed, impacted) {
    const partners = this.bonds.of(i);
    if (this.displaceHydrogens(i, 1, removed)) {
      degree[i] -= 1;
      return [];
    }
    const own = partners.map(([p]) => this.residue[p] === this.residue[i]);
    const hasCarbonyl = partners.some(([p, t], k) => own[k] && this.el(p) === 'O' && t === DOUBLE);
    if (hasCarbonyl && partners.some(([, t], k) => !own[k] && t !== COORDINATION)) {
      for (let k = 0; k < partners.length; k++) {
        const [p, t] = partners[k];
        if (!own[k] || t !== SINGLE) continue;
        const heavyNbrs = this.bonds.of(p).filter(([x]) => !isHydrogen(this.el(x))).length;
        if (this.el(p) === 'N' && this.a.coord[p].some((c) => Number.isNaN(c)) && heavyNbrs === 1) {
          return [p, ...this.bonds.hydrogensOf(p, this.a.element)];
        }
      }
    }
    for (const [o, t] of partners) {
      if (this.el(o) === 'O' && t === DOUBLE) {
        this.bonds.remove(i, o);
        this.bonds.add(i, o, SINGLE);
        this.charge[o] = -1;
        degree[i] -= 1;
        degree[o] -= 1;
        impacted[o] = true;
        return [];
      }
    }
    this.lowerMultipleBond(i, degree, false, impacted);
    return [];
  }

  /** `_rebalance_terminal_heteroatoms`. */
  rebalanceTerminalHeteroatoms(impacted, degree, removed) {
    const linkedAtoms = impacted.flatMap((x, i) => (x && this.alive[i] ? [i] : []));
    for (const l of linkedAtoms) {
      const element = this.el(l);
      if (this.validValence(l)) continue;
      const nbrs = this.bonds.of(l);
      if (element === 'P' || element === 'S') {
        const candidates = [];
        for (const [o, t] of nbrs) {
          if (t !== DOUBLE || this.el(o) !== 'O') continue;
          const on = this.bonds.of(o);
          const covalent = on.filter(([, ot]) => ot !== COORDINATION).map(([x]) => x);
          if (this.residue[o] === this.residue[l] && covalent.length === 1 && covalent[0] === l
            && isValidRdkitValence(element, this.charge[l], degree[l] - 1)) {
            candidates.push([on.filter(([, ot]) => ot === COORDINATION).length, o]);
          }
        }
        if (candidates.length) {
          candidates.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
          const o = candidates[0][1];
          this.bonds.remove(l, o);
          this.bonds.add(l, o, SINGLE);
          this.charge[o] = -1;
          degree[l] -= 1;
          degree[o] -= 1;
          impacted[o] = true;
        }
        continue;
      }
      if (!['O', 'S', 'N'].includes(element)) continue;
      let candidates = [];
      for (const [center, t] of nbrs) {
        if (t !== DOUBLE || this.residue[center] !== this.residue[l]) continue;
        for (const [s, st] of this.bonds.of(center)) {
          if (st !== SINGLE || this.residue[s] !== this.residue[l] || this.el(s) !== element) continue;
          const sn = this.bonds.of(s);
          const heavyCovalent = sn.filter(([x, xt]) => xt !== COORDINATION && !isHydrogen(this.el(x))).map(([x]) => x);
          if (heavyCovalent.length !== 1 || heavyCovalent[0] !== center) continue;
          const explicitH = this.bonds.hydrogensOf(s, this.a.element).length;
          const hasH = explicitH > 0;
          const charge = this.charge[s] === -1 ? 0 : this.charge[s];
          if (isValidRdkitValence(element, charge, degree[s] + 1 - (hasH ? 1 : 0))) {
            candidates.push({
              coordination: sn.filter(([, xt]) => xt === COORDINATION).length,
              signature: `${this.charge[s]},0,${explicitH}`, center, s, hasH, charge,
            });
          }
        }
      }
      if (!candidates.length) continue;
      const least = Math.min(...candidates.map((c) => c.coordination));
      candidates = candidates.filter((c) => c.coordination === least);
      if (new Set(candidates.map((c) => c.signature)).size !== 1) continue;
      const { center, s, hasH, charge } = candidates[0];
      if (hasH) this.displaceHydrogens(s, 1, removed);
      this.bonds.remove(l, center);
      this.bonds.add(l, center, SINGLE);
      this.bonds.remove(center, s);
      this.bonds.add(center, s, DOUBLE);
      this.charge[s] = charge;
      degree[l] -= 1;
      degree[s] += hasH ? 0 : 1;
      impacted[s] = true;
    }
  }

  /** `_resolve_overvalent_atoms`: the product repairs, in upstream's order. */
  resolveOvervalentAtoms() {
    const impacted = this.interMask();
    const inter = impacted.flatMap((x, i) => (x ? [i] : []));
    const carbons = inter.filter((i) => this.el(i) === 'C');
    const others = inter.filter((i) => this.el(i) !== 'C');
    let degree = this.degrees();
    let seen = this.bonds.version;
    const removed = new Set();
    for (const i of carbons) if (this.overvalent(i, degree)) this.lowerMultipleBond(i, degree, true, impacted);
    this.rebalanceTerminalHeteroatoms(impacted, degree, removed);
    for (const i of [...carbons, ...others]) {
      if (removed.has(i)) continue;
      const element = this.el(i);
      if (element === 'S' && degree[i] === 3 && this.displaceHydrogens(i, 1, removed)) {
        degree[i] -= 1;
        continue;
      }
      if (!this.overvalent(i, degree)) continue;
      let displace = [];
      if (element === 'C') displace = this.maybeFixOvervalentCarbon(i, degree, removed, impacted);
      else if (element === 'N' || element === 'O') {
        this.displaceHydrogens(i, degree[i] - DEFAULT_VALENCE[element] - this.charge[i], removed);
      }
      for (const gone of displace) {
        for (const [nb] of this.bonds.of(gone)) this.bonds.remove(gone, nb);
        removed.add(gone);
      }
      // Upstream recounts every atom's degree here; the count only moves when a bond did.
      if (this.bonds.version !== seen) {
        degree = this.degrees();
        seen = this.bonds.version;
      }
    }
    for (const x of removed) { this.kill(x); impacted[x] = false; }
    return impacted;
  }

  /** `apply_link_product_charges`. */
  applyLinkProductCharges(toUpdate) {
    const a = this.a;
    if (!a.element.some((e, i) => this.alive[i] && isHydrogen(e))) return; // upstream warns and returns
    const indices = toUpdate.flatMap((x, i) => (x && this.alive[i] ? [i] : []));
    const degree = this.degrees();
    const inter = this.interMask();
    for (const i of indices) {
      if (degree[i] !== 1 || !inter[i]) continue;
      const size = this.alive.filter((al, j) => al && this.residue[j] === this.residue[i]).length;
      if (size > 1) {
        throw new Error(`Atom ${this.index(i)} (${a.element[i]}, atom_name=${a.atom_name[i]}, res_name=${a.res_name[i]}, `
          + `chain_id=${a.chain_id[i]}, res_id=${a.res_id[i]}) has inter-residue bonds but no intra-residue bonds `
          + '— usually a missing CCD template. Ensure the residue is not an unknown ligand (UNL) or otherwise lacking '
          + 'bonding information.');
      }
    }
    for (const i of indices) {
      const key = `${String(a.element[i]).toUpperCase()},${this.charge[i]},${degree[i]}`;
      if (LINK_PRODUCT_CHARGES.has(key)) this.charge[i] = LINK_PRODUCT_CHARGES.get(key);
      else if (key === 'O,0,3') {
        const covalent = this.bonds.of(i).filter(([, t]) => t !== COORDINATION).map(([x]) => x);
        if (!covalent.some((x) => this.el(x) === 'O')) this.charge[i] = 1;
      }
    }
  }

  /** `_has_amide_bond`: N-C(=O), the C=O read off the first row joining them. */
  hasAmideBond(n) {
    for (const [c] of this.bonds.of(n)) {
      if (this.el(c) !== 'C') continue;
      for (const [o] of this.bonds.of(c)) if (this.el(o) === 'O' && this.bonds.type(c, o) === DOUBLE) return true;
    }
    return false;
  }

  /** `_normalize_linked_amides`. */
  normalizeLinkedAmides(toUpdate) {
    const removed = new Set();
    for (let n = 0; n < this.n; n++) {
      if (!toUpdate[n] || !this.alive[n] || this.el(n) !== 'N' || this.charge[n] !== 1) continue;
      if (!this.hasAmideBond(n)) continue;
      if (!this.displaceHydrogens(n, 1, removed)) throw new Error(`Cannot neutralize amide nitrogen ${this.index(n)}: no removable hydrogen`);
      this.charge[n] -= 1;
    }
    for (const h of removed) { this.kill(h); toUpdate[h] = false; }
  }

  /** `_validate_link_valence` (no `nhyd`: an atom implicit hydrogens could complete is accepted). */
  validateLinkValence(impacted) {
    const a = this.a;
    const degree = this.degrees();
    for (let i = 0; i < this.n; i++) {
      if (!impacted[i] || !this.alive[i] || this.validValence(i)) continue;
      if ([1, 2, 3, 4].some((h) => isValidRdkitValence(a.element[i], this.charge[i], degree[i] + h))) continue;
      throw new Error(`Unresolved link valence at ${a.chain_id[i]}/${a.res_id[i]}/${a.res_name[i]}/${a.atom_name[i]} `
        + `(element=${a.element[i]}, charge=${this.charge[i]}, bond-order sum=${degree[i]})`);
    }
  }

  result(keepAtoms) {
    this.a.charge = this.charge;
    this.a.bonds = this.bonds.list().map(([i, j, t]) => [i, j, t]);
    return keepAtoms(this.a, this.alive);
  }
}
