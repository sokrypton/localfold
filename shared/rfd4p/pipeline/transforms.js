/**
 * RFD4-Proteina data pipeline transforms from AtomArray to model batch tensors.
 *
 * Ports the validation/inference transform sequence from:
 * - `rfproteina/datapipes/pipeline.py` (`build_design_validation_pipeline`)
 * - `atomworks/ml/transforms/realize_expandable_segments.py` (`RealizeExpandableSegments`)
 * - `atomworks/ml/transforms/covalent_modifications.py` (`AnnotateCovalentModifications`)
 * - `atomworks/ml/transforms/atomize.py` (`AtomizeByCCDName`)
 * - `rfproteina/transforms/sequence_conditioning.py` (`FinalizeSequenceConditioning`, `CanonicalizeSequenceProbabilityConditions`)
 * - `rfproteina/transforms/internal_coords.py` (`MaterializeInternalCoords`)
 * - `rfproteina/transforms/unindexing.py` (`UnindexFlaggedTokens`)
 * - `rfproteina/transforms/geometry.py` (`CenterOnCoordinateConditionedAtoms`)
 */

import {
  TokenType,
  TOKEN_TYPE_IDS,
  UNIFIED_ATOM37_ENCODING,
  UNIFIED_TOKEN_TO_IDX,
  RESIDUES_BY_TOKEN_TYPE,
  LOCAL_LATENT_RESIDUE_SET,
  STANDARD_AA,
  STANDARD_RNA,
  STANDARD_DNA,
  STANDARD_POLYMER_RESIDUES,
} from '../encoding.js';

import {
  CHAIN_TYPE,
  BOND_TYPE,
  SCALAR_1BODY_ANNOTATIONS,
  NONSCALAR_1BODY_ANNOTATIONS,
  getResidueSegments,
} from './cif.js';
import { parseCathCode } from '../conditions.js';

import { getDefaultIndexGroups } from './spec.js';
import { alignVectors, computeChiralityFeatures } from './chirality.js';
import { getTokenStarts, bondFlankingResidues, computeTokenBondFeatures } from './bonds.js';

const AA_AND_RNA_SET = new Set([
  ...RESIDUES_BY_TOKEN_TYPE.aa,
  ...RESIDUES_BY_TOKEN_TYPE.rna,
]);

const PROTEIN_BACKBONE_ATOMS = new Set(['N', 'CA', 'C', 'O']);
const RNA_BACKBONE_ATOMS = new Set([
  'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
]);
const DNA_BACKBONE_ATOMS = new Set([
  'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
]);
const NA_BACKBONE_ATOMS = new Set([...RNA_BACKBONE_ATOMS, ...DNA_BACKBONE_ATOMS]);

// Build Atom37 slot lookup map `<res_name>|<atom_name>` -> slot (0..36)
const ATOM37_KEY_TO_SLOT = (() => {
  const map = new Map();
  const stdAaSet = new Set(STANDARD_AA);
  for (const resName of LOCAL_LATENT_RESIDUE_SET) {
    if (resName === '<M>') continue;
    const atoms = UNIFIED_ATOM37_ENCODING.token_atoms[resName];
    if (!atoms) continue;
    for (let slot = 0; slot < atoms.length; slot++) {
      const atomName = atoms[slot].trim();
      if (atomName) {
        map.set(`${resName}|${atomName}`, slot);
        if (stdAaSet.has(resName)) {
          map.set(`UNK|${atomName}`, slot);
        }
      }
    }
  }
  return map;
})();

/**
 * Deterministic PRNG (Mulberry32) used as fallback when explicit replay arrays aren't passed.
 */
export function createSeededRng(seed = 0) {
  let a = seed >>> 0;
  const nextFloat = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    nextFloat,
    nextInt(lo, hiExclusive) {
      return lo + Math.floor(nextFloat() * (hiExclusive - lo));
    },
    permutation(n) {
      const p = new Int32Array(n);
      for (let i = 0; i < n; i++) p[i] = i;
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(nextFloat() * (i + 1));
        const tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
      }
      return p;
    },
    randomSO3() {
      const M = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const u1 = Math.max(1e-12, nextFloat());
          const u2 = nextFloat();
          M[i][j] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        }
      }
      const u0 = M[0];
      const n0 = Math.hypot(u0[0], u0[1], u0[2]) || 1;
      const e0 = [u0[0] / n0, u0[1] / n0, u0[2] / n0];

      const d10 = M[1][0] * e0[0] + M[1][1] * e0[1] + M[1][2] * e0[2];
      const v1 = [M[1][0] - d10 * e0[0], M[1][1] - d10 * e0[1], M[1][2] - d10 * e0[2]];
      const n1 = Math.hypot(v1[0], v1[1], v1[2]) || 1;
      const e1 = [v1[0] / n1, v1[1] / n1, v1[2] / n1];

      const e2 = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
      ];
      return [e0, e1, e2];
    },
  };
}

/**
 * Slice an `AtomArray` by an array of indices (or range `[start, stop)`), preserving all top-level
 * per-atom arrays, `annotations`, and remapped `bonds`.
 */
export function sliceAtoms(atoms, indicesOrStart, stop = undefined) {
  let indices;
  if (typeof indicesOrStart === 'number') {
    const len = Math.max(0, stop - indicesOrStart);
    indices = new Array(len);
    for (let i = 0; i < len; i++) indices[i] = indicesOrStart + i;
  } else {
    indices = indicesOrStart;
  }

  const nOrig = atoms.nAtoms ?? atoms.res_name.length;
  const nAtoms = indices.length;
  const oldToNew = new Map();
  for (let i = 0; i < nAtoms; i++) oldToNew.set(indices[i], i);

  const out = { nAtoms, bonds: [], annotations: {} };
  for (const [k, v] of Object.entries(atoms)) {
    if (k === 'nAtoms' || k === 'bonds' || k === 'annotations') continue;
    if (Array.isArray(v) && v.length === nOrig) {
      out[k] = indices.map((idx) => (Array.isArray(v[idx]) ? v[idx].slice() : v[idx]));
    }
  }
  for (const [annKey, annArr] of Object.entries(atoms.annotations || {})) {
    if (Array.isArray(annArr) && annArr.length === nOrig) {
      out.annotations[annKey] = indices.map((idx) =>
        Array.isArray(annArr[idx]) ? annArr[idx].slice() : annArr[idx]
      );
    }
  }
  for (const [u, v, bt] of atoms.bonds || []) {
    const nu = oldToNew.get(u);
    const nv = oldToNew.get(v);
    if (nu !== undefined && nv !== undefined) {
      out.bonds.push([Math.min(nu, nv), Math.max(nu, nv), bt]);
    }
  }
  return out;
}

export function cloneAtoms(atoms) {
  const n = atoms.nAtoms ?? atoms.res_name.length;
  return sliceAtoms(atoms, 0, n);
}

/**
 * Concatenate multiple `AtomArray` objects, preserving all per-atom arrays, `annotations`, and shifted `bonds`.
 */
export function concatAtoms(arrays) {
  if (arrays.length === 0) throw new Error('Cannot concatenate empty list of AtomArrays');
  if (arrays.length === 1) return cloneAtoms(arrays[0]);

  let totalAtoms = 0;
  const topKeys = new Set();
  const allAnnKeys = new Set();
  for (const a of arrays) {
    const len = a.nAtoms ?? a.res_name.length;
    totalAtoms += len;
    for (const [k, v] of Object.entries(a)) {
      if (k !== 'nAtoms' && k !== 'bonds' && k !== 'annotations' && Array.isArray(v) && v.length === len) {
        topKeys.add(k);
      }
    }
    for (const k of Object.keys(a.annotations || {})) allAnnKeys.add(k);
  }

  const out = { nAtoms: totalAtoms, bonds: [], annotations: {} };
  for (const key of topKeys) {
    out[key] = [];
    for (const a of arrays) {
      const len = a.nAtoms ?? a.res_name.length;
      const arr = a[key];
      for (let i = 0; i < len; i++) {
        const val = arr ? arr[i] : null;
        out[key].push(Array.isArray(val) ? val.slice() : val);
      }
    }
  }

  for (const annKey of allAnnKeys) {
    const scalarSpec = SCALAR_1BODY_ANNOTATIONS[annKey];
    const nonScalarSpec = NONSCALAR_1BODY_ANNOTATIONS[annKey];
    let defVal = null;
    if (annKey === '_expseg_uid') defVal = -1;
    else if (scalarSpec) defVal = scalarSpec.default;
    else if (nonScalarSpec) defVal = new Array(nonScalarSpec.cols).fill(nonScalarSpec.default);
    else if (annKey.startsWith('condition_sequence-probability-')) defVal = NaN;

    out.annotations[annKey] = [];
    for (const a of arrays) {
      const len = a.nAtoms ?? a.res_name.length;
      const src = a.annotations?.[annKey];
      for (let i = 0; i < len; i++) {
        const v = src ? src[i] : defVal;
        out.annotations[annKey].push(Array.isArray(v) ? v.slice() : v);
      }
    }
  }

  let offset = 0;
  for (const a of arrays) {
    const len = a.nAtoms ?? a.res_name.length;
    for (const [u, v, bt] of a.bonds || []) {
      out.bonds.push([u + offset, v + offset, bt]);
    }
    offset += len;
  }

  return out;
}

/**
 * Compute within-group 0-indexed residue index (`within_chain_source_res_idx`).
 */
export function computeWithinChainSourceResIdx(atoms) {
  const n = atoms.nAtoms;
  const out = new Array(n).fill(0);
  if (n === 0) return out;
  const resBounds = getResidueSegments(atoms);
  const countByChain = new Map();
  for (let r = 0; r < resBounds.length - 1; r++) {
    const s = resBounds[r];
    const e = resBounds[r + 1];
    const cid = atoms.chain_iid ? atoms.chain_iid[s] : atoms.chain_id[s];
    const idx = countByChain.get(cid) ?? 0;
    for (let i = s; i < e; i++) out[i] = idx;
    countByChain.set(cid, idx + 1);
  }
  return out;
}

/**
 * Refresh chemical residue/backbone boolean flags on `atoms`.
 */
export function ensureChemicalFlags(atoms) {
  const n = atoms.nAtoms;
  const stdAaSet = new Set(STANDARD_AA);
  const stdRnaSet = new Set(STANDARD_RNA);
  const stdDnaSet = new Set(STANDARD_DNA);

  atoms.is_protein = new Array(n);
  atoms.is_rna = new Array(n);
  atoms.is_dna = new Array(n);
  atoms.is_nucleic_acid = new Array(n);
  atoms.is_protein_backbone = new Array(n);
  atoms.is_rna_backbone = new Array(n);
  atoms.is_dna_backbone = new Array(n);
  atoms.is_nucleic_acid_backbone = new Array(n);
  atoms.is_standard_aa = new Array(n);
  atoms.is_standard_or_unknown_aa = new Array(n);
  atoms.is_standard_unknown_or_masked_aa = new Array(n);
  atoms.is_standard_rna = new Array(n);
  atoms.is_standard_or_unknown_rna = new Array(n);
  atoms.is_standard_unknown_or_masked_rna = new Array(n);
  atoms.is_standard_dna = new Array(n);
  atoms.is_standard_or_unknown_dna = new Array(n);
  atoms.is_standard_unknown_or_masked_dna = new Array(n);

  for (let i = 0; i < n; i++) {
    const resName = atoms.res_name[i];
    const atomName = atoms.atom_name[i];
    const ct = atoms.chain_type[i];

    const isProt = ct === CHAIN_TYPE.POLYPEPTIDE_L || ct === CHAIN_TYPE.POLYPEPTIDE_D;
    const isRna = ct === CHAIN_TYPE.RNA;
    const isDna = ct === CHAIN_TYPE.DNA;
    const isNa = isRna || isDna || ct === CHAIN_TYPE.DNA_RNA_HYBRID;

    atoms.is_protein[i] = isProt;
    atoms.is_rna[i] = isRna;
    atoms.is_dna[i] = isDna;
    atoms.is_nucleic_acid[i] = isNa;

    atoms.is_protein_backbone[i] = isProt && PROTEIN_BACKBONE_ATOMS.has(atomName);
    atoms.is_rna_backbone[i] = isRna && RNA_BACKBONE_ATOMS.has(atomName);
    atoms.is_dna_backbone[i] = isDna && DNA_BACKBONE_ATOMS.has(atomName);
    atoms.is_nucleic_acid_backbone[i] = isNa && NA_BACKBONE_ATOMS.has(atomName);

    const stdAa = stdAaSet.has(resName);
    const stdUnkAa = stdAa || resName === 'UNK';
    atoms.is_standard_aa[i] = stdAa;
    atoms.is_standard_or_unknown_aa[i] = stdUnkAa;
    atoms.is_standard_unknown_or_masked_aa[i] = stdUnkAa || (resName === '<M>' && isProt);

    const stdRna = stdRnaSet.has(resName);
    const stdUnkRna = stdRna || resName === 'N';
    atoms.is_standard_rna[i] = stdRna;
    atoms.is_standard_or_unknown_rna[i] = stdUnkRna;
    atoms.is_standard_unknown_or_masked_rna[i] = stdUnkRna || (resName === '<M>' && isRna);

    const stdDna = stdDnaSet.has(resName);
    const stdUnkDna = stdDna || resName === 'DN';
    atoms.is_standard_dna[i] = stdDna;
    atoms.is_standard_or_unknown_dna[i] = stdUnkDna;
    atoms.is_standard_unknown_or_masked_dna[i] = stdUnkDna || (resName === '<M>' && isDna);
  }
  return atoms;
}

/**
 * 1. `AddSrcComponentAnnotation`: Set `atoms.annotations['annotation_src-component_1_atom']`.
 */
export function addSrcComponentAnnotation(atoms) {
  const n = atoms.nAtoms;
  if (!atoms.annotations) atoms.annotations = {};
  const segMin = atoms.annotations['annotation_expsegmin_1_residue'];
  const segMax = atoms.annotations['annotation_expsegmax_1_residue'];
  const srcComp = new Array(n);
  const resBounds = getResidueSegments(atoms);
  for (let r = 0; r < resBounds.length - 1; r++) {
    const s = resBounds[r];
    const e = resBounds[r + 1];
    const label =
      segMin && segMin[s] !== -1
        ? `${segMin[s]}-${segMax[s]}`
        : `${atoms.chain_id[s]}${atoms.res_id[s]}`;
    for (let i = s; i < e; i++) srcComp[i] = label;
  }
  atoms.annotations['annotation_src-component_1_atom'] = srcComp;
  return atoms;
}

/**
 * 2. `RealizeExpandableSegments`: Replace expandable segment sentinels with `N` `<M>` stubs.
 */
export function realizeExpandableSegments(atoms, { segmentLengths = null, rng = null } = {}) {
  const n = atoms.nAtoms;
  if (n === 0 || !atoms.annotations) return atoms;
  const segMin = atoms.annotations['annotation_expsegmin_1_residue'];
  const segMax = atoms.annotations['annotation_expsegmax_1_residue'];
  if (!segMin || !segMax) return atoms;

  const sentinels = [];
  for (let i = 0; i < n; i++) {
    if (segMin[i] !== -1 && atoms.res_name[i] !== '<M>') {
      sentinels.push(i);
    }
  }
  if (sentinels.length === 0) return atoms;

  const prng = rng || createSeededRng(0);
  const sampledLengths = sentinels.map((idx, k) => {
    const lo = segMin[idx];
    const hi = segMax[idx];
    if (segmentLengths && k < segmentLengths.length) {
      const len = segmentLengths[k];
      if (len < lo || len > hi) {
        throw new Error(`Segment length ${len} outside [${lo}, ${hi}] for sentinel ${k}`);
      }
      return len;
    }
    return prng.nextInt(lo, hi + 1);
  });

  atoms.annotations._expseg_uid = new Array(n).fill(-1);
  let cur = atoms;

  // Replace right-to-left so earlier sentinel indices stay valid
  for (let k = sentinels.length - 1; k >= 0; k--) {
    const idx = sentinels[k];
    const segLen = sampledLengths[k];
    const uid = k;

    let prevNeighbor = -1;
    let nextNeighbor = -1;
    for (const [a, b] of cur.bonds) {
      const other = a === idx ? b : b === idx ? a : -1;
      if (other !== -1) {
        if (other < idx) prevNeighbor = other;
        else if (other > idx) nextNeighbor = other;
      }
    }

    const baseResId = cur.res_id[idx];
    const chainIid = cur.chain_iid ? cur.chain_iid[idx] : cur.chain_id[idx];

    const afterLen = cur.nAtoms - (idx + 1);
    const prefix = idx > 0 ? sliceAtoms(cur, 0, idx) : null;
    const suffix = afterLen > 0 ? sliceAtoms(cur, idx + 1, cur.nAtoms) : null;
    if (suffix && segLen !== 1) {
      for (let i = 0; i < suffix.nAtoms; i++) {
        const cIid = suffix.chain_iid ? suffix.chain_iid[i] : suffix.chain_id[i];
        if (cIid === chainIid) {
          suffix.res_id[i] += segLen - 1;
          if (suffix.auth_seq_id) suffix.auth_seq_id[i] += segLen - 1;
        }
      }
    }

    const single = sliceAtoms(cur, idx, idx + 1);
    single.res_name[0] = '<M>';
    single.atom_name[0] = 'UNK';
    single.element[0] = 'X';
    if (single.atomic_number) single.atomic_number[0] = 0;
    single.coord[0] = [NaN, NaN, NaN];
    if (single.occupancy) single.occupancy[0] = 0.0;
    if (single.annotations._expseg_uid) single.annotations._expseg_uid[0] = uid;

    const stubs = [];
    for (let s = 0; s < segLen; s++) {
      const st = cloneAtoms(single);
      st.res_id[0] = baseResId + s;
      if (st.auth_seq_id) st.auth_seq_id[0] = baseResId + s;
      stubs.push(st);
    }
    const stubArr = stubs.length > 0 ? concatAtoms(stubs) : null;
    if (stubArr) {
      stubArr.bonds = [];
      for (let s = 0; s + 1 < segLen; s++) {
        stubArr.bonds.push([s, s + 1, BOND_TYPE.SINGLE]);
      }
    }

    const parts = [];
    if (prefix) parts.push(prefix);
    if (stubArr) parts.push(stubArr);
    if (suffix) parts.push(suffix);
    cur = parts.length > 0 ? concatAtoms(parts) : sliceAtoms(cur, 0, 0);

    if (segLen > 0) {
      if (prevNeighbor !== -1) {
        cur.bonds.push([prevNeighbor, idx, BOND_TYPE.SINGLE]);
      }
      if (nextNeighbor !== -1) {
        const shiftedNext = nextNeighbor + segLen - 1;
        cur.bonds.push([idx + segLen - 1, shiftedNext, BOND_TYPE.SINGLE]);
      }
    } else if (prevNeighbor !== -1 && nextNeighbor !== -1) {
      const shiftedNext = nextNeighbor - 1;
      cur.bonds.push([prevNeighbor, shiftedNext, BOND_TYPE.SINGLE]);
    }
  }

  cur.bonds.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  cur.within_chain_source_res_idx = computeWithinChainSourceResIdx(cur);
  ensureChemicalFlags(cur);
  return cur;
}

/**
 * 3. Filter hydrogens and terminal leaving oxygens (`RemoveHydrogens`, `RemoveTerminalOxygen`, `RemoveNucleicAcidTerminalOxygen`).
 */
export function filterUnwantedAtoms(atoms) {
  ensureChemicalFlags(atoms);
  const n = atoms.nAtoms;
  const keep = [];
  for (let i = 0; i < n; i++) {
    const elem = (atoms.element[i] || '').toUpperCase();
    if (elem === 'H' || elem === 'D') continue;
    const atomName = atoms.atom_name[i];
    const isLeaving = atoms.is_leaving_atom ? atoms.is_leaving_atom[i] : false;
    if (atoms.is_protein[i] && atomName === 'OXT' && isLeaving) continue;
    if (atoms.is_nucleic_acid[i] && atomName === 'OP3') continue;
    keep.push(i);
  }
  const filtered = keep.length === n ? atoms : sliceAtoms(atoms, keep);
  ensureChemicalFlags(filtered);
  return filtered;
}

/**
 * 4. `AddGlobalAtomIdAnnotation`: `atom_id = 0..nAtoms-1`.
 */
export function addGlobalAtomIdAnnotation(atoms) {
  const n = atoms.nAtoms;
  atoms.atom_id = Array.from({ length: n }, (_, i) => i);
  return atoms;
}

/**
 * 5. `AnnotateCovalentModifications` & `AtomizeByCCDName`.
 */
export function annotateCovalentModificationsAndAtomize(atoms) {
  const n = atoms.nAtoms;
  const resBounds = getResidueSegments(atoms);
  const nRes = resBounds.length - 1;
  const atomToRes = new Int32Array(n);
  for (let r = 0; r < nRes; r++) {
    for (let i = resBounds[r]; i < resBounds[r + 1]; i++) atomToRes[i] = r;
  }

  const resIsCovMod = new Uint8Array(nRes);
  for (const [i, j] of atoms.bonds) {
    const r1 = atomToRes[i];
    const r2 = atomToRes[j];
    if (r1 === r2) continue;
    const an1 = atoms.atom_name[i];
    const an2 = atoms.atom_name[j];
    const sameChain = atoms.chain_iid[i] === atoms.chain_iid[j];
    const resDiff = Math.abs(atoms.res_id[i] - atoms.res_id[j]);
    const isStdPeptideBond =
      sameChain &&
      resDiff === 1 &&
      atoms.is_protein[i] &&
      atoms.is_protein[j] &&
      ((an1 === 'C' && an2 === 'N') || (an1 === 'N' && an2 === 'C'));
    const isStdPhosphodiesterBond =
      sameChain &&
      resDiff === 1 &&
      atoms.is_nucleic_acid[i] &&
      atoms.is_nucleic_acid[j] &&
      ((an1 === "O3'" && an2 === 'P') || (an1 === 'P' && an2 === "O3'"));

    if (!isStdPeptideBond && !isStdPhosphodiesterBond) {
      if (atoms.is_polymer[i]) resIsCovMod[r1] = 1;
      if (atoms.is_polymer[j]) resIsCovMod[r2] = 1;
    }
  }

  atoms.is_covalent_modification = new Array(n).fill(false);
  atoms.atomize = atoms.atomize ? atoms.atomize.slice() : new Array(n).fill(false);

  for (let r = 0; r < nRes; r++) {
    const s = resBounds[r];
    const e = resBounds[r + 1];
    const resName = atoms.res_name[s];
    const isCov = Boolean(resIsCovMod[r]);
    const shouldAtomize = !LOCAL_LATENT_RESIDUE_SET.has(resName) || Boolean(atoms.atomize[s]);
    for (let i = s; i < e; i++) {
      atoms.is_covalent_modification[i] = isCov;
      atoms.atomize[i] = shouldAtomize;
    }
  }
  return atoms;
}

/**
 * 6. `AddGlobalTokenIdAnnotation`: non-atomized residues get 1 token; atomized residues get 1 token per atom.
 */
export function addGlobalTokenIdAnnotation(atoms) {
  const n = atoms.nAtoms;
  const resBounds = getResidueSegments(atoms);
  const tokenId = new Array(n);
  let nextToken = 0;
  for (let r = 0; r < resBounds.length - 1; r++) {
    const s = resBounds[r];
    const e = resBounds[r + 1];
    if (atoms.atomize[s]) {
      for (let i = s; i < e; i++) tokenId[i] = nextToken++;
    } else {
      for (let i = s; i < e; i++) tokenId[i] = nextToken;
      nextToken++;
    }
  }
  atoms.token_id = tokenId;
  return atoms;
}

function spreadTokenWiseAny(atoms, boolMask) {
  const starts = getTokenStarts(atoms, true);
  const out = new Array(atoms.nAtoms).fill(false);
  for (let t = 0; t < starts.length - 1; t++) {
    const s = starts[t];
    const e = starts[t + 1];
    let any = false;
    for (let i = s; i < e; i++) {
      if (boolMask[i]) {
        any = true;
        break;
      }
    }
    if (any) {
      for (let i = s; i < e; i++) out[i] = true;
    }
  }
  return out;
}

function spreadTokenWiseAll(atoms, boolMask) {
  const starts = getTokenStarts(atoms, true);
  const out = new Array(atoms.nAtoms).fill(false);
  for (let t = 0; t < starts.length - 1; t++) {
    const s = starts[t];
    const e = starts[t + 1];
    let all = true;
    for (let i = s; i < e; i++) {
      if (!boolMask[i]) {
        all = false;
        break;
      }
    }
    if (all) {
      for (let i = s; i < e; i++) out[i] = true;
    }
  }
  return out;
}

/**
 * 7. `FinalizeSequenceConditioning` & `CanonicalizeSequenceProbabilityConditions`.
 */
export function finalizeAndCanonicalizeSequenceConditioning(atoms) {
  const n = atoms.nAtoms;
  const ann = atoms.annotations;
  const cSeq = ann['condition_sequence_1_residue']
    ? ann['condition_sequence_1_residue'].slice()
    : new Array(n).fill('<M>');
  const cCrd = ann['condition_coordinate_1_atom'];
  const cCrdGrp = ann['condition_CoordinateGroup_1_atom'];

  const sequence = new Array(n);
  const conditioned = new Array(n);
  const required = new Array(n);

  for (let i = 0; i < n; i++) {
    const allowed = AA_AND_RNA_SET.has(atoms.res_name[i]);
    const existing = cSeq[i] !== '<M>';
    sequence[i] = allowed ? cSeq[i] : atoms.res_name[i];
    conditioned[i] = existing || !allowed;

    let req = conditioned[i] || Boolean(atoms.atomize[i]);
    const hasCrd =
      (cCrd && Number.isFinite(cCrd[i][0]) && Number.isFinite(cCrd[i][1]) && Number.isFinite(cCrd[i][2])) ||
      (cCrdGrp && cCrdGrp[i] > 0);
    if (hasCrd) {
      const isBb =
        atoms.is_protein_backbone[i] || atoms.is_rna_backbone[i] || atoms.is_dna_backbone[i];
      if (!isBb) req = true;
    }
    required[i] = req;
  }

  const requiredSpread = spreadTokenWiseAny(atoms, required);
  for (let i = 0; i < n; i++) {
    if (requiredSpread[i] && !conditioned[i]) {
      sequence[i] = atoms.res_name[i];
    }
  }
  ann['condition_sequence_1_residue'] = sequence;

  // Canonicalize sequence probability conditions
  const resBounds = getResidueSegments(atoms);
  const seqAtoms = new Array(n).fill(false);
  for (let r = 0; r < resBounds.length - 1; r++) {
    const s = resBounds[r];
    const e = resBounds[r + 1];
    let anySeq = false;
    for (let i = s; i < e; i++) {
      if (sequence[i] !== '<M>') {
        anySeq = true;
        break;
      }
    }
    if (anySeq) {
      for (let i = s; i < e; i++) seqAtoms[i] = true;
    }
  }

  for (const tok of STANDARD_POLYMER_RESIDUES) {
    const key = `condition_sequence-probability-${tok}_1_residue`;
    const prob = ann[key] ? ann[key].slice() : new Array(n).fill(NaN);
    let anyMask = false;
    for (let i = 0; i < n; i++) {
      if (seqAtoms[i]) {
        prob[i] = sequence[i] === tok ? 1.0 : 0.0;
        anyMask = true;
      } else if (!Number.isFinite(prob[i])) {
        prob[i] = NaN;
      } else {
        anyMask = true;
      }
    }
    if (anyMask) {
      ann[key] = prob;
    } else {
      delete ann[key];
    }
  }

  return atoms;
}

/**
 * 8. `FeaturizeExpandedSegments`: remove unindexed expanded segments at inference; zero coords of remaining expanded segments.
 */
export function featurizeExpandedSegments(atoms, { isInference = true } = {}) {
  const n = atoms.nAtoms;
  const segMin = atoms.annotations['annotation_expsegmin_1_residue'];
  if (!segMin) return atoms;

  let hasExp = false;
  for (let i = 0; i < n; i++) {
    if (segMin[i] !== -1) {
      hasExp = true;
      break;
    }
  }
  if (!hasExp) return atoms;

  let cur = atoms;
  const cIdx = cur.annotations['condition_index_1_residue'];
  if (isInference && cIdx) {
    const keep = [];
    for (let i = 0; i < cur.nAtoms; i++) {
      const isExp = cur.annotations['annotation_expsegmin_1_residue'][i] !== -1;
      const isUnidx = !cIdx[i];
      if (!(isExp && isUnidx)) keep.push(i);
    }
    if (keep.length < cur.nAtoms) {
      cur = sliceAtoms(cur, keep);
    }
  }

  for (let i = 0; i < cur.nAtoms; i++) {
    if (cur.annotations['annotation_expsegmin_1_residue'][i] !== -1) {
      cur.coord[i] = [0.0, 0.0, 0.0];
    }
  }
  return cur;
}

/**
 * 9. `MaterializeInternalCoords`: fill unset positive `condition_CoordinateGroup_1_atom` groups.
 */
export function materializeInternalCoords(atoms, { internalCoordRotations = null, rng = null } = {}) {
  const n = atoms.nAtoms;
  const ann = atoms.annotations;
  if (!ann['condition_CoordinateGroup_1_atom']) {
    const cCrd = ann['condition_coordinate_1_atom'];
    ann['condition_CoordinateGroup_1_atom'] = new Array(n);
    for (let i = 0; i < n; i++) {
      const known =
        cCrd &&
        Number.isFinite(cCrd[i][0]) &&
        Number.isFinite(cCrd[i][1]) &&
        Number.isFinite(cCrd[i][2]);
      ann['condition_CoordinateGroup_1_atom'][i] = known ? 0 : -1;
    }
  }
  if (!ann['condition_InternalCoordinate_1_atom']) {
    ann['condition_InternalCoordinate_1_atom'] = Array.from({ length: n }, () => [NaN, NaN, NaN]);
  }

  const groups = ann['condition_CoordinateGroup_1_atom'];
  const payload = ann['condition_InternalCoordinate_1_atom'];
  const internalGroupIds = [...new Set(groups.filter((g) => g > 0))].sort((a, b) => a - b);
  if (internalGroupIds.length === 0) return atoms;

  const prng = rng || createSeededRng(0);
  for (const gid of internalGroupIds) {
    const members = [];
    let allFinite = true;
    for (let i = 0; i < n; i++) {
      if (groups[i] === gid) {
        members.push(i);
        if (
          !Number.isFinite(payload[i][0]) ||
          !Number.isFinite(payload[i][1]) ||
          !Number.isFinite(payload[i][2])
        ) {
          allFinite = false;
        }
      }
    }
    if (allFinite || members.length === 0) continue;

    let mx = 0, my = 0, mz = 0;
    for (const i of members) {
      mx += atoms.coord[i][0];
      my += atoms.coord[i][1];
      mz += atoms.coord[i][2];
    }
    mx /= members.length;
    my /= members.length;
    mz /= members.length;

    const R = (internalCoordRotations && internalCoordRotations[gid]) || prng.randomSO3();
    for (const i of members) {
      const cx = atoms.coord[i][0] - mx;
      const cy = atoms.coord[i][1] - my;
      const cz = atoms.coord[i][2] - mz;
      payload[i] = [
        cx * R[0][0] + cy * R[0][1] + cz * R[0][2],
        cx * R[1][0] + cy * R[1][1] + cz * R[1][2],
        cx * R[2][0] + cy * R[2][1] + cz * R[2][2],
      ];
    }
  }
  return atoms;
}

/**
 * 10. `UnindexFlaggedTokens`: move unindexed motifs (`!condition_index_1_residue`) to end of array.
 */
export function unindexFlaggedTokens(atoms, { isInference = true } = {}) {
  const n = atoms.nAtoms;
  const ann = atoms.annotations;
  const cIdx = ann['condition_index_1_residue'] || new Array(n).fill(true);
  let anyUnindexed = false;
  for (let i = 0; i < n; i++) {
    if (!cIdx[i]) {
      anyUnindexed = true;
      break;
    }
  }

  if (!anyUnindexed) {
    atoms.insertion_source_idx = new Array(n).fill(-1);
    atoms.is_unindexed_source = new Array(n).fill(false);
    addGlobalTokenIdAnnotation(atoms);
    addGlobalAtomIdAnnotation(atoms);
    return atoms;
  }

  if (!ann['condition_index-group_1_token']) {
    ann['condition_index-group_1_token'] = getDefaultIndexGroups(atoms);
  }
  const indexGroups = ann['condition_index-group_1_token'];

  const maxResIdByChain = new Map();
  for (let i = 0; i < n; i++) {
    const cid = atoms.chain_iid[i];
    const prev = maxResIdByChain.get(cid);
    if (prev === undefined || atoms.res_id[i] > prev) {
      maxResIdByChain.set(cid, atoms.res_id[i]);
    }
  }

  let nextCopyTokenId = Math.max(...atoms.token_id) + 1;
  const nextFreeResId = new Map();
  for (const [cid, maxRes] of maxResIdByChain.entries()) {
    nextFreeResId.set(cid, maxRes + 1);
  }

  const unindexedGroupIds = [
    ...new Set(indexGroups.filter((_, i) => !cIdx[i])),
  ].sort((a, b) => a - b);

  const unindexedSlices = [];
  for (const gid of unindexedGroupIds) {
    const memberIdxs = [];
    for (let i = 0; i < n; i++) {
      if (!cIdx[i] && indexGroups[i] === gid) memberIdxs.push(i);
    }
    const motif = sliceAtoms(atoms, memberIdxs);

    const motifChains = [...new Set(motif.chain_iid)];
    for (const cid of motifChains) {
      let minRes = Infinity;
      let maxRes = -Infinity;
      for (let i = 0; i < motif.nAtoms; i++) {
        if (motif.chain_iid[i] === cid) {
          if (motif.res_id[i] < minRes) minRes = motif.res_id[i];
          if (motif.res_id[i] > maxRes) maxRes = motif.res_id[i];
        }
      }
      const offset = nextFreeResId.get(cid) - minRes;
      for (let i = 0; i < motif.nAtoms; i++) {
        if (motif.chain_iid[i] === cid) {
          motif.res_id[i] += offset;
        }
      }
      nextFreeResId.set(cid, maxRes + offset + 2);
    }

    const resBounds = getResidueSegments(motif);
    motif.insertion_source_idx = new Array(motif.nAtoms);
    motif.token_id = new Array(motif.nAtoms);
    for (let r = 0; r < resBounds.length - 1; r++) {
      const s = resBounds[r];
      const e = resBounds[r + 1];
      const srcTok = isInference ? -1 : motif.token_id[s];
      for (let i = s; i < e; i++) {
        motif.insertion_source_idx[i] = srcTok;
        motif.token_id[i] = nextCopyTokenId;
      }
      nextCopyTokenId++;
    }
    unindexedSlices.push(motif);
  }

  const unindexed = concatAtoms(unindexedSlices);

  // Remove original unindexed tokens at inference
  const resBoundsOrig = getResidueSegments(atoms);
  const nResOrig = resBoundsOrig.length - 1;
  const isRemovedRes = new Array(nResOrig).fill(false);
  const isExpandableRes = new Array(nResOrig).fill(false);
  const chainOfRes = new Array(nResOrig);
  const segMin = ann['annotation_expsegmin_1_residue'];

  for (let r = 0; r < nResOrig; r++) {
    const s = resBoundsOrig[r];
    const e = resBoundsOrig[r + 1];
    chainOfRes[r] = atoms.chain_iid[s];
    isExpandableRes[r] = Boolean(segMin && segMin[s] !== -1);
    for (let i = s; i < e; i++) {
      if (!cIdx[i]) {
        isRemovedRes[r] = true;
        break;
      }
    }
  }

  const nCollapsedBefore = new Int32Array(nResOrig);
  const uniqueChains = [...new Set(chainOfRes)];
  for (const cid of uniqueChains) {
    const inChain = [];
    for (let r = 0; r < nResOrig; r++) {
      if (chainOfRes[r] === cid) inChain.push(r);
    }
    const isRem = inChain.map((r) => isRemovedRes[r]);
    const collapse = isRem.slice();

    let k = 0;
    while (k < isRem.length) {
      if (!isRem[k]) {
        k++;
        continue;
      }
      const runStart = k;
      while (k < isRem.length && isRem[k]) k++;
      const runStop = k;
      const atBoundary = runStart === 0 || runStop === inChain.length;
      const adjExpandable =
        !atBoundary &&
        (isExpandableRes[inChain[runStart - 1]] || isExpandableRes[inChain[runStop]]);
      if (!atBoundary && !adjExpandable) {
        for (let m = runStart; m < runStop; m++) collapse[m] = false;
      }
    }

    let cum = 0;
    for (let m = 0; m < inChain.length; m++) {
      nCollapsedBefore[inChain[m]] = cum;
      if (collapse[m]) cum++;
    }
  }

  const collapsedBeforeAtom = new Int32Array(n);
  const tokenToUnindex = new Array(n).fill(false);
  for (let r = 0; r < nResOrig; r++) {
    const s = resBoundsOrig[r];
    const e = resBoundsOrig[r + 1];
    for (let i = s; i < e; i++) {
      collapsedBeforeAtom[i] = nCollapsedBefore[r];
      tokenToUnindex[i] = isRemovedRes[r];
      atoms.res_id[i] -= nCollapsedBefore[r];
    }
  }

  const indexedKeep = [];
  const collapsedKept = [];
  for (let i = 0; i < n; i++) {
    if (!tokenToUnindex[i]) {
      indexedKeep.push(i);
      collapsedKept.push(collapsedBeforeAtom[i]);
    }
  }

  const indexedArr = sliceAtoms(atoms, indexedKeep);
  const resBounds = getResidueSegments(indexedArr);
  for (let r = 0; r + 1 < resBounds.length - 1; r++) {
    const sPrev = resBounds[r];
    const sNext = resBounds[r + 1];
    if (
      indexedArr.chain_iid[sPrev] === indexedArr.chain_iid[sNext] &&
      indexedArr.res_id[sNext] - indexedArr.res_id[sPrev] === 1 &&
      collapsedKept[sNext] > collapsedKept[sPrev]
    ) {
      bondFlankingResidues(indexedArr, r, r + 1, resBounds);
    }
  }

  indexedArr.within_chain_source_res_idx = computeWithinChainSourceResIdx(indexedArr);
  indexedArr.insertion_source_idx = new Array(indexedArr.nAtoms).fill(-1);

  const full = concatAtoms([indexedArr, unindexed]);
  const fullIdxGrp = full.annotations['condition_index-group_1_token'];
  const fullIsIdx = full.annotations['condition_index_1_residue'];
  let maxIdxGrp = -Infinity;
  let minUnidxGrp = Infinity;
  for (let i = 0; i < full.nAtoms; i++) {
    if (fullIsIdx[i]) {
      if (fullIdxGrp[i] > maxIdxGrp) maxIdxGrp = fullIdxGrp[i];
    } else {
      if (fullIdxGrp[i] < minUnidxGrp) minUnidxGrp = fullIdxGrp[i];
    }
  }
  const shift = (Number.isFinite(maxIdxGrp) ? maxIdxGrp : -1) - minUnidxGrp + 1;
  for (let i = 0; i < full.nAtoms; i++) {
    if (!fullIsIdx[i]) fullIdxGrp[i] += shift;
  }

  full.is_unindexed_source = new Array(full.nAtoms).fill(false);
  addGlobalTokenIdAnnotation(full);
  addGlobalAtomIdAnnotation(full);
  ensureChemicalFlags(full);
  return full;
}

/**
 * 11. `CenterOnCoordinateConditionedAtoms` & `FeaturizeCentroidOfAtomsWithoutCoordinateConditioning`.
 */
export function centerAndFeaturizeCentroid(atoms) {
  const n = atoms.nAtoms;
  const ann = atoms.annotations;
  const cCrd = ann['condition_coordinate_1_atom'];
  const coordMask = new Array(n).fill(false);
  let anyCoordCond = false;
  if (cCrd) {
    for (let i = 0; i < n; i++) {
      if (
        Number.isFinite(cCrd[i][0]) &&
        Number.isFinite(cCrd[i][1]) &&
        Number.isFinite(cCrd[i][2])
      ) {
        coordMask[i] = true;
        anyCoordCond = true;
      }
    }
  }

  let sx = 0, sy = 0, sz = 0, count = 0;
  for (let i = 0; i < n; i++) {
    if (anyCoordCond && !coordMask[i]) continue;
    const c = atoms.coord[i];
    if (Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2])) {
      sx += c[0];
      sy += c[1];
      sz += c[2];
      count++;
    }
  }
  const shift = count > 0 ? [sx / count, sy / count, sz / count] : [0, 0, 0];

  atoms.coord_to_be_noised = new Array(n);
  for (let i = 0; i < n; i++) {
    const c = atoms.coord[i];
    atoms.coord_to_be_noised[i] = [c[0] - shift[0], c[1] - shift[1], c[2] - shift[2]];
  }

  for (const key of ['condition_coordinate_1_atom', 'condition_ChainCentroid_1_atom']) {
    const arr = ann[key];
    if (!arr) continue;
    for (let i = 0; i < n; i++) {
      if (
        Number.isFinite(arr[i][0]) &&
        Number.isFinite(arr[i][1]) &&
        Number.isFinite(arr[i][2])
      ) {
        arr[i] = [arr[i][0] - shift[0], arr[i][1] - shift[1], arr[i][2] - shift[2]];
      }
    }
  }

  let ux = 0, uy = 0, uz = 0, uCount = 0;
  let anyNotCond = false;
  for (let i = 0; i < n; i++) {
    if (!coordMask[i]) {
      anyNotCond = true;
      const c = atoms.coord_to_be_noised[i];
      if (Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2])) {
        ux += c[0];
        uy += c[1];
        uz += c[2];
        uCount++;
      }
    }
  }
  const centroidWithoutCond = new Float32Array(3);
  if (anyNotCond && uCount > 0) {
    centroidWithoutCond[0] = ux / uCount;
    centroidWithoutCond[1] = uy / uCount;
    centroidWithoutCond[2] = uz / uCount;
  } else {
    centroidWithoutCond.fill(NaN);
  }

  return { atoms, centroidWithoutCond };
}

/**
 * 12. `AddSupportsLocalLatentAnnotation`.
 */
export function addSupportsLocalLatentAnnotation(atoms) {
  const n = atoms.nAtoms;
  const cIdx = atoms.annotations['condition_index_1_residue'];
  const unidxMask = cIdx ? cIdx.map((v) => !v) : new Array(n).fill(false);
  const inContext = spreadTokenWiseAny(atoms, unidxMask);
  atoms.supports_local_latent = new Array(n);
  for (let i = 0; i < n; i++) {
    atoms.supports_local_latent[i] =
      LOCAL_LATENT_RESIDUE_SET.has(atoms.res_name[i]) &&
      !atoms.atomize[i] &&
      !inContext[i];
  }
  return atoms;
}

/**
 * Map each atom to its Atom37 slot (`0..36`), or `-1` if unrepresented (e.g. `<M>:UNK` stub).
 */
export function buildAtom37Slots(atoms) {
  const n = atoms.nAtoms;
  const slots = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    if (atoms.atomize && atoms.atomize[i]) {
      slots[i] = 1;
      continue;
    }
    const resName = atoms.res_name[i];
    const atomName = (atoms.atom_name[i] || '').trim();
    const s = ATOM37_KEY_TO_SLOT.get(`${resName}|${atomName}`);
    if (s !== undefined) slots[i] = s;
  }
  return slots;
}

function encodeResNames(atoms, resNames, atomizeToken = '<A>') {
  const n = resNames.length;
  const out = new Int32Array(n);
  const atomizeIdx = UNIFIED_TOKEN_TO_IDX[atomizeToken];
  for (let i = 0; i < n; i++) {
    if (atoms.atomize && atoms.atomize[i]) {
      out[i] = atomizeIdx;
      continue;
    }
    const rn = resNames[i];
    if (UNIFIED_TOKEN_TO_IDX[rn] !== undefined) {
      out[i] = UNIFIED_TOKEN_TO_IDX[rn];
    } else if (atoms.is_protein && atoms.is_protein[i]) {
      out[i] = UNIFIED_TOKEN_TO_IDX.UNK;
    } else if (atoms.is_dna && atoms.is_dna[i]) {
      out[i] = UNIFIED_TOKEN_TO_IDX.DN;
    } else if (atoms.is_rna && atoms.is_rna[i]) {
      out[i] = UNIFIED_TOKEN_TO_IDX.N;
    } else {
      out[i] = UNIFIED_TOKEN_TO_IDX['<G>'];
    }
  }
  return out;
}

/**
 * 13. Convert AtomArray to token-level Atom37 Proteina batch tensors (`batch.*`).
 */
export function convertToProteinaBatch(
  atoms,
  {
    centroidWithoutCond,
    chainPerm = null,
    moleculePerm = null,
    oracleBatch = null,
    pathLengthCutoff = 64,
    isInference = true,
    rng = null,
  } = {}
) {
  const nAtoms = atoms.nAtoms;
  const ann = atoms.annotations;
  const tokenStarts = getTokenStarts(atoms, false);
  const nTokens = tokenStarts.length;
  const prng = rng || createSeededRng(0);

  // Unique sorted chain_iid and molecule_iid
  const uniqueChains = [...new Set(atoms.chain_iid)].sort();
  const chainToIdx = new Map(uniqueChains.map((c, i) => [c, i]));
  let resolvedChainPerm = chainPerm;
  if (!resolvedChainPerm && oracleBatch && oracleBatch.chain_iid) {
    const obChain = oracleBatch.chain_iid.data ?? oracleBatch.chain_iid;
    resolvedChainPerm = new Int32Array(uniqueChains.length);
    for (let t = 0; t < nTokens; t++) {
      const aIdx = tokenStarts[t];
      resolvedChainPerm[chainToIdx.get(atoms.chain_iid[aIdx])] = obChain[t];
    }
  }
  if (!resolvedChainPerm) {
    resolvedChainPerm = prng.permutation(uniqueChains.length);
  }

  const uniqueMols = [...new Set(atoms.molecule_iid)].sort((a, b) => a - b);
  const molToIdx = new Map(uniqueMols.map((m, i) => [m, i]));
  let resolvedMolPerm = moleculePerm;
  if (!resolvedMolPerm && oracleBatch && oracleBatch.molecule_iid) {
    const obMol = oracleBatch.molecule_iid.data ?? oracleBatch.molecule_iid;
    resolvedMolPerm = new Int32Array(uniqueMols.length);
    for (let t = 0; t < nTokens; t++) {
      const aIdx = tokenStarts[t];
      resolvedMolPerm[molToIdx.get(atoms.molecule_iid[aIdx])] = obMol[t];
    }
  }
  if (!resolvedMolPerm) {
    resolvedMolPerm = prng.permutation(uniqueMols.length);
  }

  // molecule_entity_id
  const uniqueEntities = [...new Set(atoms.molecule_entity)].sort((a, b) => a - b);
  const entityToIdx = new Map(uniqueEntities.map((e, i) => [e, i]));
  const entityIds = new Int16Array(nAtoms);
  for (let i = 0; i < nAtoms; i++) {
    entityIds[i] = entityToIdx.get(atoms.molecule_entity[i]);
  }
  const cSeqNames = ann['condition_sequence_1_residue'] || new Array(nAtoms).fill('<M>');
  if (isInference) {
    let nextEntity = uniqueEntities.length > 0 ? uniqueEntities.length : 0;
    const maskedChains = [];
    const seenMaskedChain = new Set();
    for (let i = 0; i < nAtoms; i++) {
      if (cSeqNames[i] === '<M>') {
        const cid = atoms.chain_iid[i];
        if (!seenMaskedChain.has(cid)) {
          seenMaskedChain.add(cid);
          maskedChains.push(cid);
        }
      }
    }
    maskedChains.sort();
    for (const cid of maskedChains) {
      for (let i = 0; i < nAtoms; i++) {
        if (atoms.chain_iid[i] === cid) entityIds[i] = nextEntity;
      }
      nextEntity++;
    }
  }

  // Atom-level token_type
  const atomTokenType = new Int32Array(nAtoms).fill(TokenType.UNKNOWN);
  for (let i = 0; i < nAtoms; i++) {
    if (atoms.is_standard_unknown_or_masked_dna[i]) atomTokenType[i] = TokenType.DNA;
    if (atoms.is_standard_unknown_or_masked_rna[i]) atomTokenType[i] = TokenType.RNA;
    if (atoms.is_standard_unknown_or_masked_aa[i]) atomTokenType[i] = TokenType.PROTEIN;
    const rn = atoms.res_name[i];
    const idx = UNIFIED_TOKEN_TO_IDX[rn];
    if (idx !== undefined) {
      const grp = UNIFIED_ATOM37_ENCODING.token_group_by_idx[idx];
      const tid = TOKEN_TYPE_IDS[grp];
      if (tid !== TokenType.UNKNOWN) atomTokenType[i] = tid;
    }
    if (atoms.atomize[i]) atomTokenType[i] = TokenType.ATOMIZED;
  }
  atoms.token_type = Array.from(atomTokenType);

  // Atom-level is_resolved_mask (spread token-wise with all)
  const atomResolved = new Array(nAtoms);
  for (let i = 0; i < nAtoms; i++) {
    const c = atoms.coord[i];
    atomResolved[i] =
      atoms.occupancy[i] > 0.0 &&
      !Number.isNaN(c[0]) &&
      !Number.isNaN(c[1]) &&
      !Number.isNaN(c[2]);
  }
  const tokenResolvedSpread = spreadTokenWiseAll(atoms, atomResolved);

  // Atom-level is_in_context_conditioning
  const cIdx = ann['condition_index_1_residue'];
  const unidxMask = cIdx ? cIdx.map((v) => !v) : new Array(nAtoms).fill(false);
  const inContextSpread = spreadTokenWiseAny(atoms, unidxMask);

  // Atom-level residue_type & condition_sequence_1_residue
  const atomResidueType = encodeResNames(atoms, atoms.res_name, '<A>');
  const atomCSeqEncoded = encodeResNames(atoms, cSeqNames, '<A>');
  for (let i = 0; i < nAtoms; i++) {
    if (cSeqNames[i] === '<M>' && !atoms.atomize[i]) {
      atomCSeqEncoded[i] = UNIFIED_TOKEN_TO_IDX['<M>'];
    }
  }

  // Chirality features (atom-level, pooled to token-level via tokenStarts)
  const chirFeats = computeChiralityFeatures(atoms);
  const chirData = chirFeats.chirality.data ?? chirFeats.chirality;
  const chirFrames = chirFeats.chiral_frame_atoms.data ?? chirFeats.chiral_frame_atoms;
  const chirOwners = chirFeats.chiral_slot_owner.data ?? chirFeats.chiral_slot_owner;
  const tetVerts = chirFeats.tetrahedral_vertices.data ?? chirFeats.tetrahedral_vertices;

  // Build token-level features by pooling at tokenStarts
  const residue_type = new Int32Array(nTokens);
  const cond_seq = new Int32Array(nTokens);
  const token_type = new Int32Array(nTokens);
  const tok_idx = new Int16Array(nTokens);
  const chain_iid = new Int16Array(nTokens);
  const molecule_iid = new Int16Array(nTokens);
  const molecule_entity_id = new Int16Array(nTokens);
  const supports_local_latent = new Uint8Array(nTokens);
  const is_in_context_conditioning = new Uint8Array(nTokens);
  const is_resolved_mask = new Uint8Array(nTokens);
  const insertion_source_idx = new Int32Array(nTokens);
  const loss_weight = new Float32Array(nTokens);
  const coord_cond_mask = new Uint8Array(nTokens);
  const cond_sec = new Int16Array(nTokens);
  const cond_idxgrp = new Int16Array(nTokens);
  const cond_cath = new Int32Array(nTokens * 3).fill(-1);
  const cond_centroid = new Float32Array(nTokens * 3).fill(NaN);
  const chirality = new Float32Array(nTokens * 18);
  const chiral_frame_atoms = new Int32Array(nTokens * 6 * 8);
  const chiral_slot_owner = new Int32Array(nTokens * 6);
  const tetrahedral_vertices = new Int32Array(nTokens * 4);

  const cCrd = ann['condition_coordinate_1_atom'];
  const cSec = ann['condition_secondary-structure_1_residue'];
  const cIdxGrp = ann['condition_index-group_1_token'] || getDefaultIndexGroups(atoms);
  const cCentroid = ann['condition_ChainCentroid_1_atom'];
  const cCath = ann['condition_cath-label_1_residue'];

  for (let t = 0; t < nTokens; t++) {
    const a = tokenStarts[t];
    residue_type[t] = atomResidueType[a];
    cond_seq[t] = atomCSeqEncoded[a];
    token_type[t] = atomTokenType[a];
    tok_idx[t] = atoms.token_id[a];
    chain_iid[t] = resolvedChainPerm[chainToIdx.get(atoms.chain_iid[a])];
    molecule_iid[t] = resolvedMolPerm[molToIdx.get(atoms.molecule_iid[a])];
    molecule_entity_id[t] = entityIds[a];
    supports_local_latent[t] = atoms.supports_local_latent[a] ? 1 : 0;
    is_in_context_conditioning[t] = inContextSpread[a] ? 1 : 0;
    is_resolved_mask[t] = tokenResolvedSpread[a] ? 1 : 0;
    insertion_source_idx[t] = atoms.insertion_source_idx ? atoms.insertion_source_idx[a] : -1;
    loss_weight[t] = atoms.loss_weight ? atoms.loss_weight[a] : 1.0;
    coord_cond_mask[t] =
      cCrd &&
      Number.isFinite(cCrd[a][0]) &&
      Number.isFinite(cCrd[a][1]) &&
      Number.isFinite(cCrd[a][2])
        ? 1
        : 0;
    cond_sec[t] = cSec ? cSec[a] : 0;
    cond_idxgrp[t] = cIdxGrp[a];
    if (cCath && cCath[a] !== undefined) {
      const triple = Array.isArray(cCath[a]) ? cCath[a] : parseCathCode(cCath[a]);
      cond_cath[t * 3] = triple[0];
      cond_cath[t * 3 + 1] = triple[1];
      cond_cath[t * 3 + 2] = triple[2];
    }
    if (cCentroid) {
      cond_centroid[t * 3] = cCentroid[a][0];
      cond_centroid[t * 3 + 1] = cCentroid[a][1];
      cond_centroid[t * 3 + 2] = cCentroid[a][2];
    }
    for (let k = 0; k < 18; k++) chirality[t * 18 + k] = chirData[a * 18 + k];
    for (let k = 0; k < 48; k++) chiral_frame_atoms[t * 48 + k] = chirFrames[a * 48 + k];
    for (let k = 0; k < 6; k++) chiral_slot_owner[t * 6 + k] = chirOwners[a * 6 + k];
    for (let k = 0; k < 4; k++) tetrahedral_vertices[t * 4 + k] = tetVerts[a * 4 + k];
  }

  // Atom37-level arrays [nTokens, 37, ...]
  const atomSlots = buildAtom37Slots(atoms);
  const atom37_mask = new Uint8Array(nTokens * 37);
  const coords = new Float32Array(nTokens * 37 * 3);
  const element = new Int16Array(nTokens * 37).fill(-1);
  const ground_truth_element = new Int16Array(nTokens * 37);
  const charge = new Int16Array(nTokens * 37);
  const incident_bond_type_counts = new Int16Array(nTokens * 37 * 10);
  const cond_crd = new Float32Array(nTokens * 37 * 3).fill(NaN);
  const cond_intcrd = new Float32Array(nTokens * 37 * 3).fill(NaN);
  const cond_idx = new Uint8Array(nTokens * 37).fill(1);
  const cond_crdgrp = new Int16Array(nTokens * 37).fill(-1);
  const cond_hot = new Uint8Array(nTokens * 37);
  const cond_ras = new Int16Array(nTokens * 37);
  const cond_hb = new Int16Array(nTokens * 37);
  const cond_pistack = new Uint8Array(nTokens * 37);
  const cond_pication = new Int16Array(nTokens * 37);
  const cond_salt = new Int16Array(nTokens * 37);
  const cond_hphob = new Uint8Array(nTokens * 37);
  const cond_mcoord = new Int16Array(nTokens * 37);

  const cIntCrd = ann['condition_InternalCoordinate_1_atom'];
  const cCrdGrp = ann['condition_CoordinateGroup_1_atom'];
  const cHot = ann['condition_hotspot_1_atom'];
  const cRas = ann['condition_rasa-binned_1_atom'];
  const cHb = ann['condition_hbond_1_atom'];
  const cPiStack = ann['condition_pistack_1_atom'];
  const cPiCation = ann['condition_pication_1_atom'];
  const cSalt = ann['condition_saltbridge_1_atom'];
  const cHphob = ann['condition_hydrophobic_1_atom'];
  const cMcoord = ann['condition_metal-coord_1_atom'];

  for (let i = 0; i < nAtoms; i++) {
    const s = atomSlots[i];
    if (s < 0) continue;
    const t = atoms.token_id[i];
    const ts = t * 37 + s;
    atom37_mask[ts] = 1;

    const c = atoms.coord_to_be_noised[i];
    coords[ts * 3] = c[0];
    coords[ts * 3 + 1] = c[1];
    coords[ts * 3 + 2] = c[2];

    const z = atoms.atomic_number[i];
    ground_truth_element[ts] = z;
    element[ts] = cSeqNames[i] !== '<M>' ? z : -1;
    charge[ts] = atoms.charge ? atoms.charge[i] : 0;

    if (cCrd) {
      cond_crd[ts * 3] = cCrd[i][0];
      cond_crd[ts * 3 + 1] = cCrd[i][1];
      cond_crd[ts * 3 + 2] = cCrd[i][2];
    }
    if (cIntCrd) {
      cond_intcrd[ts * 3] = cIntCrd[i][0];
      cond_intcrd[ts * 3 + 1] = cIntCrd[i][1];
      cond_intcrd[ts * 3 + 2] = cIntCrd[i][2];
    }
    cond_idx[ts] = cIdx ? (cIdx[i] ? 1 : 0) : 1;
    cond_crdgrp[ts] = cCrdGrp ? cCrdGrp[i] : -1;
    cond_hot[ts] = cHot && cHot[i] ? 1 : 0;
    cond_ras[ts] = cRas ? cRas[i] : 0;
    cond_hb[ts] = cHb ? cHb[i] : 0;
    cond_pistack[ts] = cPiStack && cPiStack[i] ? 1 : 0;
    cond_pication[ts] = cPiCation ? cPiCation[i] : 0;
    cond_salt[ts] = cSalt ? cSalt[i] : 0;
    cond_hphob[ts] = cHphob && cHphob[i] ? 1 : 0;
    cond_mcoord[ts] = cMcoord ? cMcoord[i] : 0;
  }

  // If oracleBatch is provided and has internal coordinate groups, align each internal coord group via Kabsch SVD
  if (oracleBatch && oracleBatch.condition_InternalCoordinate_1_atom) {
    const obIntCrd =
      oracleBatch.condition_InternalCoordinate_1_atom.data ??
      oracleBatch.condition_InternalCoordinate_1_atom;
    const internalGids = [...new Set(cond_crdgrp)].filter((g) => g > 0);
    for (const gid of internalGids) {
      const slotIndices = [];
      const srcPts = [];
      const tgtPts = [];
      for (let ts = 0; ts < nTokens * 37; ts++) {
        if (cond_crdgrp[ts] === gid) {
          slotIndices.push(ts);
          srcPts.push([cond_intcrd[ts * 3], cond_intcrd[ts * 3 + 1], cond_intcrd[ts * 3 + 2]]);
          tgtPts.push([obIntCrd[ts * 3], obIntCrd[ts * 3 + 1], obIntCrd[ts * 3 + 2]]);
        }
      }
      if (srcPts.length >= 3) {
        const { rotation: R } = alignVectors(tgtPts, srcPts);
        for (let m = 0; m < slotIndices.length; m++) {
          const ts = slotIndices[m];
          const p = srcPts[m];
          cond_intcrd[ts * 3] = p[0] * R[0][0] + p[1] * R[0][1] + p[2] * R[0][2];
          cond_intcrd[ts * 3 + 1] = p[0] * R[1][0] + p[1] * R[1][1] + p[2] * R[1][2];
          cond_intcrd[ts * 3 + 2] = p[0] * R[2][0] + p[1] * R[2][1] + p[2] * R[2][2];
        }
      }
    }
  }

  // Incident bond type counts [nTokens, 37, 10]
  for (const [a, b, bt] of atoms.bonds) {
    const sa = atomSlots[a];
    if (sa >= 0) {
      const ta = atoms.token_id[a];
      incident_bond_type_counts[(ta * 37 + sa) * 10 + bt]++;
    }
    const sb = atomSlots[b];
    if (sb >= 0) {
      const tb = atoms.token_id[b];
      incident_bond_type_counts[(tb * 37 + sb) * 10 + bt]++;
    }
  }

  // Token-level bond features
  const bondFeats = computeTokenBondFeatures(atoms, {
    pathLengthCutoff,
    fillResIdGaps: isInference,
  });

  // Derived coords_nm and coord_mask
  const coords_nm = new Float32Array(coords.length);
  for (let i = 0; i < coords.length; i++) coords_nm[i] = coords[i] / 10.0;
  const coord_mask = new Uint8Array(atom37_mask);
  const padding_mask = new Uint8Array(nTokens).fill(1);

  // 🔴 THE NUCLEIC SECONDARY STRUCTURE, as upstream's tokeniser assembles it
  // (`datapipes/pipeline.py`): the per-residue state at each token's first atom, present where it
  // is not UNSPECIFIED; a base pair between two tokens' first atoms in both directions; and a LOOP
  // token's whole row and column marked as specified-unpaired. All three were zeros here whatever
  // the input said - the model reads `na_ss_state` and `na_ss_pair` and was told nothing.
  const naSs = nucleicSecondaryStructure(ann, tokenStarts, nAtoms);

  const batch = {
    centroid_of_atoms_without_coordinate_conditioning: {
      data: centroidWithoutCond,
      shape: [1, 3],
      dtype: 'F32',
    },
    residue_type: { data: residue_type, shape: [1, nTokens], dtype: 'I64' },
    condition_sequence_1_residue: { data: cond_seq, shape: [1, nTokens], dtype: 'I64' },
    token_type: { data: token_type, shape: [1, nTokens], dtype: 'I32' },
    tok_idx: { data: tok_idx, shape: [1, nTokens], dtype: 'I16' },
    chain_iid: { data: chain_iid, shape: [1, nTokens], dtype: 'I16' },
    molecule_iid: { data: molecule_iid, shape: [1, nTokens], dtype: 'I16' },
    molecule_entity_id: { data: molecule_entity_id, shape: [1, nTokens], dtype: 'I16' },
    supports_local_latent: { data: supports_local_latent, shape: [1, nTokens], dtype: 'U8' },
    is_in_context_conditioning: { data: is_in_context_conditioning, shape: [1, nTokens], dtype: 'U8' },
    is_resolved_mask: { data: is_resolved_mask, shape: [1, nTokens], dtype: 'U8' },
    insertion_source_idx: { data: insertion_source_idx, shape: [1, nTokens], dtype: 'I32' },
    loss_weight: { data: loss_weight, shape: [1, nTokens], dtype: 'F32' },
    coord_cond_mask: { data: coord_cond_mask, shape: [1, nTokens], dtype: 'U8' },
    'condition_secondary-structure_1_residue': { data: cond_sec, shape: [1, nTokens], dtype: 'I16' },
    'condition_index-group_1_token': { data: cond_idxgrp, shape: [1, nTokens], dtype: 'I16' },
    na_ss_state: { data: naSs.state, shape: [1, nTokens], dtype: 'I16' },
    na_ss_state_mask: { data: naSs.stateMask, shape: [1, nTokens], dtype: 'U8' },
    padding_mask: { data: padding_mask, shape: [1, nTokens], dtype: 'U8' },
    'condition_cath-label_1_residue': { data: cond_cath, shape: [1, nTokens, 3], dtype: 'I32' },
    condition_ChainCentroid_1_atom: { data: cond_centroid, shape: [1, nTokens, 3], dtype: 'F32' },
    chirality: { data: chirality, shape: [1, nTokens, 18], dtype: 'F32' },
    chiral_frame_atoms: { data: chiral_frame_atoms, shape: [1, nTokens, 6, 8], dtype: 'I64' },
    chiral_slot_owner: { data: chiral_slot_owner, shape: [1, nTokens, 6], dtype: 'I64' },
    tetrahedral_vertices: { data: tetrahedral_vertices, shape: [1, nTokens, 4], dtype: 'I64' },
    coords: { data: coords, shape: [1, nTokens, 37, 3], dtype: 'F32' },
    coords_nm: { data: coords_nm, shape: [1, nTokens, 37, 3], dtype: 'F32' },
    element: { data: element, shape: [1, nTokens, 37], dtype: 'I16' },
    ground_truth_element: { data: ground_truth_element, shape: [1, nTokens, 37], dtype: 'I16' },
    charge: { data: charge, shape: [1, nTokens, 37], dtype: 'I16' },
    incident_bond_type_counts: {
      data: incident_bond_type_counts,
      shape: [1, nTokens, 37, 10],
      dtype: 'I16',
    },
    atom37_mask: { data: atom37_mask, shape: [1, nTokens, 37], dtype: 'U8' },
    coord_mask: { data: coord_mask, shape: [1, nTokens, 37], dtype: 'U8' },
    condition_coordinate_1_atom: { data: cond_crd, shape: [1, nTokens, 37, 3], dtype: 'F32' },
    condition_InternalCoordinate_1_atom: {
      data: cond_intcrd,
      shape: [1, nTokens, 37, 3],
      dtype: 'F32',
    },
    condition_index_1_residue: { data: cond_idx, shape: [1, nTokens, 37], dtype: 'U8' },
    condition_CoordinateGroup_1_atom: { data: cond_crdgrp, shape: [1, nTokens, 37], dtype: 'I16' },
    condition_hotspot_1_atom: { data: cond_hot, shape: [1, nTokens, 37], dtype: 'U8' },
    'condition_rasa-binned_1_atom': { data: cond_ras, shape: [1, nTokens, 37], dtype: 'I16' },
    condition_hbond_1_atom: { data: cond_hb, shape: [1, nTokens, 37], dtype: 'I16' },
    condition_pistack_1_atom: { data: cond_pistack, shape: [1, nTokens, 37], dtype: 'U8' },
    condition_pication_1_atom: { data: cond_pication, shape: [1, nTokens, 37], dtype: 'I16' },
    condition_saltbridge_1_atom: { data: cond_salt, shape: [1, nTokens, 37], dtype: 'I16' },
    condition_hydrophobic_1_atom: { data: cond_hphob, shape: [1, nTokens, 37], dtype: 'U8' },
    'condition_metal-coord_1_atom': { data: cond_mcoord, shape: [1, nTokens, 37], dtype: 'I16' },
    bond_mask: {
      data: bondFeats.bond_mask.data ?? bondFeats.bond_mask,
      shape: [1, nTokens, nTokens],
      dtype: 'U8',
    },
    bond_order: {
      data: bondFeats.bond_order.data ?? bondFeats.bond_order,
      shape: [1, nTokens, nTokens],
      dtype: 'I16',
    },
    is_aromatic: {
      data: bondFeats.is_aromatic.data ?? bondFeats.is_aromatic,
      shape: [1, nTokens, nTokens],
      dtype: 'U8',
    },
    na_ss_pair: {
      data: naSs.pair,
      shape: [1, nTokens, nTokens],
      dtype: 'I16',
    },
    na_ss_pair_mask: {
      data: naSs.pairMask,
      shape: [1, nTokens, nTokens],
      dtype: 'U8',
    },
    token_bond_path_length: {
      data: bondFeats.token_bond_path_length.data ?? bondFeats.token_bond_path_length,
      shape: [1, nTokens, nTokens],
      dtype: 'I16',
    },
    canonical_polymer_ordering: {
      data: bondFeats.canonical_polymer_ordering.data ?? bondFeats.canonical_polymer_ordering,
      shape: [1, nTokens, nTokens],
      dtype: 'I8',
    },
  };

  for (const tok of STANDARD_POLYMER_RESIDUES) {
    const key = `condition_sequence-probability-${tok}_1_residue`;
    const src = ann[key];
    const arr = new Float32Array(nTokens).fill(NaN);
    if (src) {
      for (let t = 0; t < nTokens; t++) arr[t] = src[tokenStarts[t]];
    }
    batch[key] = { data: arr, shape: [1, nTokens], dtype: 'F32' };
  }

  return { batch, n: nTokens };
}

/**
 * Recover realized expandable segment lengths from an oracle batch's `atom37_mask`.
 */
export function inferSegmentLengthsFromOracleBatch(oracleBatch) {
  if (!oracleBatch || !oracleBatch.atom37_mask) return null;
  const mask = oracleBatch.atom37_mask.data ?? oracleBatch.atom37_mask;
  const n = oracleBatch.atom37_mask.shape
    ? oracleBatch.atom37_mask.shape.at(-2)
    : mask.length / 37;
  const runs = [];
  let cur = 0;
  for (let t = 0; t < n; t++) {
    let any = false;
    for (let s = 0; s < 37; s++) {
      if (mask[t * 37 + s]) {
        any = true;
        break;
      }
    }
    if (!any) {
      cur++;
    } else if (cur > 0) {
      runs.push(cur);
      cur = 0;
    }
  }
  if (cur > 0) runs.push(cur);
  return runs;
}

/**
 * Run the full RFD4-Proteina validation/inference pipeline on a parsed `AtomArray`.
 */
export function runValidationPipeline(atomsInput, opts = {}) {
  const {
    seed = 0,
    segmentLengths = null,
    chainPerm = null,
    moleculePerm = null,
    internalCoordRotations = null,
    oracleBatch = null,
    pathLengthCutoff = 64,
    isInference = true,
  } = opts;

  const rng = createSeededRng(seed);
  const resolvedSegLengths =
    segmentLengths || (oracleBatch ? inferSegmentLengthsFromOracleBatch(oracleBatch) : null);

  let atoms = cloneAtoms(atomsInput);
  atoms = addSrcComponentAnnotation(atoms);
  if (isInference) {
    atoms = realizeExpandableSegments(atoms, { segmentLengths: resolvedSegLengths, rng });
  }
  atoms = filterUnwantedAtoms(atoms);
  atoms = addGlobalAtomIdAnnotation(atoms);
  atoms = annotateCovalentModificationsAndAtomize(atoms);
  atoms = addGlobalTokenIdAnnotation(atoms);
  atoms = finalizeAndCanonicalizeSequenceConditioning(atoms);
  atoms = addGlobalTokenIdAnnotation(atoms);
  atoms = featurizeExpandedSegments(atoms, { isInference });
  atoms = materializeInternalCoords(atoms, { internalCoordRotations, rng });
  atoms = unindexFlaggedTokens(atoms, { isInference });

  const { atoms: centeredAtoms, centroidWithoutCond } = centerAndFeaturizeCentroid(atoms);
  atoms = centeredAtoms;
  atoms = addSupportsLocalLatentAnnotation(atoms);

  const { batch, n } = convertToProteinaBatch(atoms, {
    centroidWithoutCond,
    chainPerm,
    moleculePerm,
    oracleBatch,
    pathLengthCutoff,
    isInference,
    rng,
  });

  return { batch, n, atoms, inputAtomArray: atomsInput };
}

/** `na_ss_state`, `na_ss_pair` and their masks (upstream `datapipes/pipeline.py`). */
function nucleicSecondaryStructure(ann, tokenStarts, nAtoms) {
  const nTokens = tokenStarts.length;
  const state = new Int16Array(nTokens);
  const stateMask = new Uint8Array(nTokens);
  const pair = new Int16Array(nTokens * nTokens);
  const pairMask = new Uint8Array(nTokens * nTokens);
  const stateAnn = ann?.['condition_na-ss-state_1_residue'];
  if (stateAnn) {
    for (let t = 0; t < nTokens; t++) {
      state[t] = Number(stateAnn[tokenStarts[t]] ?? 0);
      stateMask[t] = state[t] !== 0 ? 1 : 0;
    }
  }
  const srcIdx = ann?.['annotation_src-atom-index_1_atom'];
  const partners = ann?.['condition_basepair-partners_1_atom'];
  if (srcIdx && partners) {
    // Only a token's FIRST atom stands for it: `atom_to_token[token_starts] = arange(n_tokens)`.
    const tokenOfSrc = new Map();
    for (let t = 0; t < nTokens; t++) {
      const src = srcIdx[tokenStarts[t]];
      if (src != null && src >= 0) tokenOfSrc.set(src, t);
    }
    for (let t = 0; t < nTokens; t++) {
      const list = partners[tokenStarts[t]];
      if (!list) continue;
      for (const raw of String(list).split(',')) {
        if (raw === '') continue;
        const u = tokenOfSrc.get(Number(raw));
        if (u === undefined) continue;
        pair[t * nTokens + u] = 1;
        pairMask[t * nTokens + u] = 1;
      }
    }
  }
  // BasePairEnum.LOOP = 2: an explicitly unpaired token's row and column are SPECIFIED.
  for (let t = 0; t < nTokens; t++) {
    if (state[t] !== 2) continue;
    for (let u = 0; u < nTokens; u++) {
      pairMask[t * nTokens + u] = 1;
      pairMask[u * nTokens + t] = 1;
    }
  }
  return { state, stateMask, pair, pairMask };
}
