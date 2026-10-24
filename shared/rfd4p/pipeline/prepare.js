/**
 * Structure preparation for unannotated imports, mirroring the AtomWorks `rcsb` ParseConfig preset that AtomWorks
 * Studio applies to an "Unannotated input" (`atomworks.io` + `atomworks_studio_editing.loading`):
 *
 * - build biological assembly 1 (`atomworks/io/utils/assembly.py`, biotite's operation-expression grammar);
 * - remove waters and common crystallization additives (`remove_waters`, `remove_ccds`);
 * - convert selenomethionine to methionine (`convert_mse_to_met`).
 *
 * - `altloc="first"` per chain;
 * - `add_missing_atoms` for polymer heavy atoms (unmodeled residues and truncated side chains), plus CCD charges
 *   for monoatomic ions. The hydrogens AtomWorks also adds never reach the RFD4 batch.
 *
 * `fix_arginines` is done by `parseCifToAtomArray`.
 */
import { getOrderedAtomNamesForResidue } from '../output.js';
import {
  CHAIN_TYPE,
  ELEMENT_TO_ATOMIC_NUMBER,
  NONSCALAR_1BODY_ANNOTATIONS,
  SCALAR_1BODY_ANNOTATIONS,
  STD_RESIDUE_AROMATIC,
  buildBondsForAtomArray,
  getResidueSegments,
} from './cif.js';
import { concatAtomArrays, sliceAtomArray } from './spec.js';

/** CCD formal charges of common monoatomic ions, for files that leave `pdbx_formal_charge` blank. */
const ION_CHARGES = {
  LI: 1, NA: 1, K: 1, RB: 1, CS: 1, AG: 1, CU1: 1, MG: 2, CA: 2, SR: 2, BA: 2, MN: 2, CO: 2, NI: 2, CU: 2, ZN: 2,
  CD: 2, HG: 2, PB: 2, FE2: 2, FE: 3, AL: 3, GA: 3, LA: 3, YB: 3, F: -1, CL: -1, BR: -1, IOD: -1,
};

const isProteinChain = (ct) => ct === CHAIN_TYPE.POLYPEPTIDE_L || ct === CHAIN_TYPE.POLYPEPTIDE_D;
const isNucleicChain = (ct) => ct === CHAIN_TYPE.DNA || ct === CHAIN_TYPE.RNA;

/**
 * `add_missing_atoms` for polymers (heavy atoms only; the hydrogens AtomWorks also adds never reach the batch):
 * residues present in `_pdbx_poly_seq_scheme` but not modeled are added with every template heavy atom, and modeled
 * residues missing template atoms (truncated side chains) are completed. Added atoms have no coordinates and zero
 * occupancy, so the model designs them. Bonds are rebuilt from the completed residues.
 */
export function addMissingPolymerAtoms(atoms, cifBlock) {
  const scheme = cifBlock?.pdbx_poly_seq_scheme;
  if (!scheme || atoms.nAtoms === 0) return atoms;
  const col = (k) => {
    const v = scheme[k];
    return v === undefined ? [] : Array.isArray(v) ? v : [v];
  };
  const asym = col('asym_id').map(String);
  const seqId = col('seq_id').map(Number);
  const monId = col('mon_id').map(String);
  const pdbSeq = col('pdb_seq_num').map(Number);
  const seqByChain = new Map();
  for (let k = 0; k < asym.length; k += 1) {
    const rows = seqByChain.get(asym[k]) ?? [];
    // Unmodeled MSE is added as MET, as convert_mse_to_met has already run on the modeled residues.
    const mon = monId[k] === 'MSE' ? 'MET' : monId[k];
    if (!rows.length || rows.at(-1).seqId !== seqId[k]) rows.push({ seqId: seqId[k], mon, auth: pdbSeq[k] });
    seqByChain.set(asym[k], rows);
  }

  // Residues of each polymer chain, keyed by res_id, in file order.
  const starts = getResidueSegments(atoms);
  const chainOrder = [];
  const chainRes = new Map();
  for (let r = 0; r < starts.length - 1; r += 1) {
    const i = starts[r];
    const chain = atoms.chain_id[i];
    if (!chainRes.has(chain)) {
      chainRes.set(chain, []);
      chainOrder.push(chain);
    }
    chainRes.get(chain).push({ start: i, stop: starts[r + 1], resId: atoms.res_id[i] });
  }

  const plan = []; // entries: number (source atom) or object (new atom fields)
  let changed = false;
  const emitted = new Set();
  for (let r = 0; r < starts.length - 1; r += 1) {
    const chain = atoms.chain_id[starts[r]];
    if (emitted.has(chain)) continue;
    emitted.add(chain);
    const residues = chainRes.get(chain);
    const ct = atoms.chain_type?.[residues[0].start];
    const rows = seqByChain.get(chain);
    if (!rows || !(isProteinChain(ct) || isNucleicChain(ct))) {
      for (const res of residues) for (let i = res.start; i < res.stop; i += 1) plan.push(i);
      continue;
    }
    const byId = new Map(residues.map((res) => [res.resId, res]));
    const ref = residues[0].start;
    const newAtom = (row, name) => ({ ref, resName: row.mon, resId: row.seqId, authSeq: row.auth, name });
    for (const row of rows) {
      const res = byId.get(row.seqId);
      // A modeled residue is completed from its own (possibly converted) name, not the scheme's.
      const template = getOrderedAtomNamesForResidue(res ? atoms.res_name[res.start] : row.mon);
      if (!res) {
        if (!template.length) continue;
        for (const name of template) plan.push(newAtom(row, name));
        changed = true;
        continue;
      }
      byId.delete(row.seqId);
      const present = new Map();
      for (let i = res.start; i < res.stop; i += 1) if (!present.has(atoms.atom_name[i])) present.set(atoms.atom_name[i], i);
      const missing = template.filter((name) => !present.has(name));
      if (!missing.length) {
        for (let i = res.start; i < res.stop; i += 1) plan.push(i);
        continue;
      }
      changed = true;
      // Template order, then any non-template atoms (OXT, hydrogens, ...) in file order.
      const used = new Set();
      for (const name of template) {
        if (present.has(name)) {
          plan.push(present.get(name));
          used.add(present.get(name));
        } else {
          plan.push(newAtom({ ...row, mon: atoms.res_name[res.start], auth: atoms.auth_seq_id[res.start] }, name));
        }
      }
      for (let i = res.start; i < res.stop; i += 1) if (!used.has(i)) plan.push(i);
    }
    // Modeled residues the scheme does not list (rare) keep their atoms, after the scheme residues.
    for (const res of byId.values()) for (let i = res.start; i < res.stop; i += 1) plan.push(i);
  }
  if (!changed) return atoms;

  const out = { nAtoms: plan.length, bonds: [], annotations: {} };
  const keys = Object.keys(atoms).filter((k) => Array.isArray(atoms[k]) && atoms[k].length === atoms.nAtoms);
  for (const key of keys) {
    out[key] = plan.map((p) => {
      if (typeof p === 'number') return key === 'coord' ? atoms.coord[p].slice() : atoms[key][p];
      const element = p.name[0];
      switch (key) {
        case 'coord': return [NaN, NaN, NaN];
        case 'atom_name': return p.name;
        case 'res_name': return p.resName;
        case 'res_id': return p.resId;
        case 'auth_seq_id': return p.authSeq;
        case 'element': return element;
        case 'atomic_number': return ELEMENT_TO_ATOMIC_NUMBER[element] ?? 0;
        case 'occupancy': return 0;
        case 'b_factor': return NaN;
        case 'charge': return 0;
        case 'hetero': return false;
        case 'ins_code': return '';
        case 'label_alt_id': return '.';
        case 'is_aromatic': return Boolean(STD_RESIDUE_AROMATIC[p.resName]?.has(p.name));
        case 'is_leaving_atom': return false;
        case 'atom_id': return 0;
        default: return atoms[key][p.ref];
      }
    });
  }
  for (const [k, arr] of Object.entries(atoms.annotations || {})) {
    const ns = NONSCALAR_1BODY_ANNOTATIONS[k];
    const sc = SCALAR_1BODY_ANNOTATIONS[k];
    out.annotations[k] = plan.map((p) => {
      if (typeof p === 'number') return Array.isArray(arr[p]) ? arr[p].slice() : arr[p];
      return ns ? new Array(ns.cols).fill(ns.default) : sc ? sc.default : arr[p.ref];
    });
  }
  out.atom_id = out.atom_id ? out.atom_id.map((_, i) => i + 1) : out.atom_id;
  buildBondsForAtomArray(out, cifBlock);
  return out;
}

/**
 * `altloc="first"`: keep the alphabetically first alternate-location letter of each chain, dropping atoms of other
 * letters (a residue modeled only in another conformer then becomes a missing residue). `parseCifToAtomArray`
 * already keeps one conformer per atom; this applies AtomWorks' chain-level choice on top.
 */
export function selectFirstAltlocPerChain(atoms) {
  if (!atoms.label_alt_id) return atoms;
  const isLetter = (v) => v && v !== '.' && v !== '?';
  const first = new Map();
  for (let i = 0; i < atoms.nAtoms; i += 1) {
    const alt = atoms.label_alt_id[i];
    if (!isLetter(alt)) continue;
    const cur = first.get(atoms.chain_id[i]);
    if (cur === undefined || alt < cur) first.set(atoms.chain_id[i], alt);
  }
  if (!first.size) return atoms;
  const keep = [];
  for (let i = 0; i < atoms.nAtoms; i += 1) {
    const alt = atoms.label_alt_id[i];
    if (!isLetter(alt) || alt === first.get(atoms.chain_id[i])) keep.push(i);
  }
  return keep.length === atoms.nAtoms ? atoms : sliceAtomArray(atoms, keep);
}

/** Fill blank formal charges of monoatomic ions from the CCD values. In place. */
export function assignIonCharges(atoms) {
  for (let i = 0; i < atoms.nAtoms; i += 1) {
    const q = ION_CHARGES[atoms.res_name[i]];
    if (q !== undefined && !atoms.charge[i]) atoms.charge[i] = q;
  }
  return atoms;
}

/** `ParseConfig.from_preset('rcsb').remove_ccds` (also `web/studio/schema.json` `loading.presets.rcsb`). */
export const RCSB_REMOVE_CCDS = [
  'SO4', 'GOL', 'EDO', 'PO4', 'ACT', 'PEG', 'DMS', 'TRS', 'PGE', 'PG4', 'FMT', 'EPE', 'MPD', 'MES', 'CD', 'IOD',
];
export const WATER_CCDS = ['HOH', 'DOD', 'WAT', 'H2O'];

/**
 * biotite `_parse_operation_expression`: `'1,2'`, `'1-60'`, `'(1-60)'`, `'(1,2)(3-4)'`, `'(X0)(1-5)'`. Parenthesized
 * groups combine right to left: each operation lists its steps in application order (rightmost group first).
 * @returns {string[][]}
 */
export function parseOperationExpression(expr) {
  const text = String(expr).trim();
  const groups = text.includes('(') ? [...text.matchAll(/\(([^)]*)\)/g)].map((m) => m[1]) : [text];
  const expanded = groups.map((g) => g.split(',').flatMap((part) => {
    const p = part.trim();
    const range = /^(\d+)-(\d+)$/.exec(p);
    if (!range) return p ? [p] : [];
    const out = [];
    for (let i = Number(range[1]); i <= Number(range[2]); i += 1) out.push(String(i));
    return out;
  }));
  let ops = [[]];
  for (const group of expanded.reverse()) {
    const next = [];
    for (const prefix of ops) for (const step of group) next.push([...prefix, step]);
    ops = next;
  }
  return ops;
}

const column = (cat, key) => {
  const v = cat?.[key];
  return v === undefined ? [] : Array.isArray(v) ? v : [v];
};

/**
 * Build one biological assembly (default the first listed, as `build_assembly="1"` resolves for RCSB entries).
 * Returns `null` when the file has no assembly records or the assembly would exceed `maxAtoms` (callers keep the
 * asymmetric unit then). Copies carry `transformation_id`; when a chain occurs in more than one copy, every chain is
 * renamed `<chain>_<transformation_id>` (AtomWorks' `chain_disambiguation="chain_iid"` export).
 */
export function buildAssembly(atoms, cifBlock, { assemblyId = '1', maxAtoms = 250000 } = {}) {
  const gen = cifBlock?.pdbx_struct_assembly_gen;
  const opers = cifBlock?.pdbx_struct_oper_list;
  if (!gen || !opers) return null;
  const ids = column(gen, 'assembly_id').map(String);
  const want = ids.includes(String(assemblyId)) ? String(assemblyId) : ids[0];
  const transforms = new Map();
  const operIds = column(opers, 'id').map(String);
  operIds.forEach((id, k) => {
    const m = (i, j) => Number(column(opers, `matrix[${i}][${j}]`)[k]);
    const v = (i) => Number(column(opers, `vector[${i}]`)[k]);
    transforms.set(id, {
      R: [m(1, 1), m(1, 2), m(1, 3), m(2, 1), m(2, 2), m(2, 3), m(3, 1), m(3, 2), m(3, 3)],
      t: [v(1), v(2), v(3)],
    });
  });
  const rows = ids.map((id, k) => ({ id, expr: column(gen, 'oper_expression')[k], asyms: column(gen, 'asym_id_list')[k] }))
    .filter((r) => r.id === want);
  const plan = rows.map((r) => {
    const asyms = new Set(String(r.asyms).split(',').map((s) => s.trim()));
    const indices = [];
    for (let i = 0; i < atoms.nAtoms; i += 1) if (asyms.has(atoms.chain_id[i])) indices.push(i);
    return { indices, operations: parseOperationExpression(r.expr) };
  });
  const total = plan.reduce((s, p) => s + p.indices.length * p.operations.length, 0);
  if (total === 0 || total > maxAtoms) return null;
  if (plan.some((p) => p.operations.some((op) => op.some((step) => !transforms.has(step))))) return null;

  const parts = [];
  for (const { indices, operations } of plan) {
    for (const operation of operations) {
      const part = sliceAtomArray(atoms, indices);
      for (const step of operation) {
        const { R, t } = transforms.get(step);
        for (const c of part.coord) {
          const [x, y, z] = c;
          c[0] = R[0] * x + R[1] * y + R[2] * z + t[0];
          c[1] = R[3] * x + R[4] * y + R[5] * z + t[1];
          c[2] = R[6] * x + R[7] * y + R[8] * z + t[2];
        }
      }
      part.transformation_id = new Array(part.nAtoms).fill(operation.join(''));
      parts.push(part);
    }
  }
  const out = concatAtomArrays(parts);
  const copies = new Map();
  for (let i = 0; i < out.nAtoms; i += 1) {
    const set = copies.get(out.chain_id[i]) ?? new Set();
    set.add(out.transformation_id[i]);
    copies.set(out.chain_id[i], set);
  }
  if ([...copies.values()].some((s) => s.size > 1)) {
    out.chain_id = out.chain_id.map((c, i) => `${c}_${out.transformation_id[i]}`);
  }
  return out;
}

/** Drop whole residues whose CCD code is in `ccds`. */
export function removeResidues(atoms, ccds) {
  const drop = new Set(ccds);
  const keep = [];
  for (let i = 0; i < atoms.nAtoms; i += 1) if (!drop.has(atoms.res_name[i])) keep.push(i);
  return keep.length === atoms.nAtoms ? atoms : sliceAtomArray(atoms, keep);
}

/** `convert_mse_to_met`: MSE -> MET, SE -> SD (sulfur), as a standard polymer residue. In place. */
export function convertMseToMet(atoms) {
  for (let i = 0; i < atoms.nAtoms; i += 1) {
    if (atoms.res_name[i] !== 'MSE') continue;
    atoms.res_name[i] = 'MET';
    atoms.hetero[i] = false;
    if (atoms.chem_comp_type) atoms.chem_comp_type[i] = 'L-PEPTIDE LINKING';
    if (atoms.atom_name[i] === 'SE') {
      atoms.atom_name[i] = 'SD';
      atoms.element[i] = 'S';
      if (atoms.atomic_number) atoms.atomic_number[i] = 16;
    }
  }
  return atoms;
}

/**
 * The `rcsb` preset's structural preparation, in AtomWorks' order: assembly, then removals, then MSE conversion.
 * @returns {{atoms: object, notes: string[]}}
 */
export function prepareRcsbStructure(atoms, cifBlock, { assemblyId = '1', maxAtoms = 250000 } = {}) {
  // AtomWorks order (`atomworks/io/_pipeline.py`): altloc at load, remove components, MSE -> MET, complete atoms
  // (CCD templates, which also set ion charges), then the assembly.
  const notes = [];
  let out = removeResidues(selectFirstAltlocPerChain(atoms), [...WATER_CCDS, ...RCSB_REMOVE_CCDS]);
  convertMseToMet(out);
  out = addMissingPolymerAtoms(out, cifBlock);
  assignIonCharges(out);
  const built = buildAssembly(out, cifBlock, { assemblyId, maxAtoms });
  if (built) {
    out = built;
  } else if (cifBlock?.pdbx_struct_assembly_gen) {
    notes.push(`Biological assembly ${assemblyId} is larger than ${maxAtoms} atoms; kept the asymmetric unit.`);
  }
  return { atoms: out, notes };
}
