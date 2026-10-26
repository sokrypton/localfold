/**
 * `ValidateBatchInvariants` (rfproteina/transforms/invariants.py): the checks upstream runs over
 * every finished batch, at inference as at training, and RAISES on.
 *
 * 🔴 NEITHER PORT HAD ANY OF THEM, SO EIGHT KINDS OF JOB UPSTREAM REFUSES FOLDED HERE. A spec over
 * a structure whose designed chain has no coordinates, conditioning on an expandable segment, a
 * centroid on one chain of two, an unindexed residue with nothing to place it by, sequence
 * probabilities that sum past one or name another polymer's residues, a contig that joins two
 * residues 20 A apart with a peptide bond - each was a batch the model was handed and upstream's
 * never is. tools/check-rfd4p-invariants.py holds the verdicts, check by check, to upstream's on
 * a corpus of jobs good and bad; cuda/rfd4p/featurise/invariants.h is the native twin.
 *
 * Training-only checks (`check_conditioning_annotation_matches_ground_truth`, the training arms
 * of the others) are not ported: this pipeline is inference only.
 */

import { CHAIN_TYPE } from './cif.js';
import { getTokenStarts, TOKEN_TYPE, getPolymerizationAtomNamesForToken } from './bonds.js';
import { buildAtom37Slots, addGlobalTokenIdAnnotation } from './transforms.js';
import { LOCAL_LATENT_RESIDUE_SET, UNIFIED_TOKEN_TO_IDX } from '../encoding.js';

const MAX_BOND = 4.0;
const MAX_POLYMER_BOND = 2.0;
const SEQ_PROB_TOKENS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
  'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'];
const STANDARD_AA = new Set(SEQ_PROB_TOKENS.slice(0, 20));
const STANDARD_RNA = new Set(['A', 'C', 'G', 'U']);
const STANDARD_DNA = new Set(['DA', 'DC', 'DG', 'DT']);
const STANDARD_POLYMER = new Set(SEQ_PROB_TOKENS);
// `ATOM37_FEATURES`, less what is not polymer conditioning (coords, element, charge), with defaults.
const ATOM37_CONDITIONS = [
  ['condition_coordinate_1_atom', 'coord'], ['condition_InternalCoordinate_1_atom', 'coord'],
  ['condition_index_1_residue', true], ['condition_CoordinateGroup_1_atom', -1],
  ['condition_hotspot_1_atom', false], ['condition_rasa-binned_1_atom', 0], ['condition_hbond_1_atom', 0],
  ['condition_pistack_1_atom', false], ['condition_pication_1_atom', 0], ['condition_saltbridge_1_atom', 0],
  ['condition_hydrophobic_1_atom', false], ['condition_metal-coord_1_atom', 0],
];
// The 1-body StandardAnnotations above atom level (`STANDARD_ANNOTATIONS`), with their level.
const LEVELLED = [
  ['annotation_expsegmax_1_residue', 'residue'], ['annotation_expsegmin_1_residue', 'residue'],
  ['condition_c-terminus_1_residue', 'residue'], ['condition_cath-label_1_residue', 'residue'],
  ['condition_index-group_1_token', 'token'], ['condition_index_1_residue', 'residue'],
  ['condition_n-terminus_1_residue', 'residue'], ['condition_na-ss-state_1_residue', 'residue'],
  ['condition_secondary-structure_1_residue', 'residue'],
  ...SEQ_PROB_TOKENS.map((t) => [`condition_sequence-probability-${t}_1_residue`, 'residue']),
  ['condition_sequence_1_residue', 'residue'],
];
const PROTEIN_CHAINS = new Set([CHAIN_TYPE.POLYPEPTIDE_D, CHAIN_TYPE.POLYPEPTIDE_L, CHAIN_TYPE.CYCLIC_PSEUDO_PEPTIDE]);
const NUCLEIC_CHAINS = new Set([CHAIN_TYPE.DNA, CHAIN_TYPE.RNA, CHAIN_TYPE.DNA_RNA_HYBRID]);

const finite3 = (v) => Array.isArray(v) && Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
const anyFinite3 = (v) => Array.isArray(v) && v.some(Number.isFinite);
const allNaN = (v) => !Array.isArray(v) || v.every(Number.isNaN);
const sameValue = (a, b) => {
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((x, k) => sameValue(x, b[k]));
  }
  return a === b || (typeof a === 'number' && typeof b === 'number' && Number.isNaN(a) && Number.isNaN(b));
};
// `np.isclose`: rtol 1e-5, atol 1e-8, NaN equal only where asked.
const isClose = (a, b, equalNan = false) => (Number.isNaN(a) || Number.isNaN(b)
  ? equalNan && Number.isNaN(a) && Number.isNaN(b)
  : Math.abs(a - b) <= 1e-8 + 1e-5 * Math.abs(b));

/** A column of the finished array: an annotation, or a field. */
function column(atoms, name) {
  return atoms.annotations?.[name] ?? atoms[name];
}

/** atomworks' `get_residue_starts`: chain, residue, insertion code and transformation. */
function residueStarts(atoms) {
  const n = atoms.nAtoms;
  const out = [];
  for (let i = 0; i < n; i += 1) {
    if (i === 0 || atoms.chain_id[i] !== atoms.chain_id[i - 1] || atoms.res_id[i] !== atoms.res_id[i - 1]
      || (atoms.ins_code?.[i] ?? '') !== (atoms.ins_code?.[i - 1] ?? '')
      || (atoms.transformation_id?.[i] ?? '') !== (atoms.transformation_id?.[i - 1] ?? '')) out.push(i);
  }
  return out;
}

/** A batch field as a flat array and its per-token stride. */
function feat(batch, name) {
  const f = batch[name];
  if (f === undefined) return null;
  return f.data ?? f;
}

const atomized = (atoms, i) => Boolean(atoms.atomize?.[i]);

// ---- atom-array checks -----------------------------------------------------------------------

function moleculesOnSeparateChains(atoms) {
  const chain = atoms.chain_iid ?? atoms.chain_id;
  const molecule = atoms.molecule_iid ?? atoms.molecule_id;
  const byChain = new Map();
  for (let i = 0; i < atoms.nAtoms; i += 1) {
    if (!byChain.has(chain[i])) byChain.set(chain[i], new Set());
    byChain.get(chain[i]).add(molecule[i]);
  }
  return [...byChain.keys()].sort().flatMap((c) => (byChain.get(c).size > 1
    ? [{ check: 'multiple_molecules_on_chain', message: `Chain ${c} contains ${byChain.get(c).size} different molecules`, count: 1 }]
    : []));
}

function standardAnnotationLevels(atoms) {
  const out = [];
  const n = atoms.nAtoms;
  const residue = [...residueStarts(atoms), n];
  const token = getTokenStarts(atoms, true);
  for (const [name, level] of LEVELLED) {
    const values = column(atoms, name);
    if (!values) continue;
    const starts = level === 'residue' ? residue : token;
    let bad = 0;
    for (let s = 0; s + 1 < starts.length; s += 1) {
      for (let i = starts[s] + 1; i < starts[s + 1]; i += 1) {
        if (!sameValue(values[i], values[starts[s]])) { bad += 1; break; }
      }
    }
    if (bad) {
      out.push({ check: 'standard_annotation_level_violated',
        message: `'${name}' (level=${level}) has inconsistent values in ${bad} segment(s)`, count: bad });
    }
  }
  return out;
}

function globalTokenIdConsistent(atoms) {
  const before = Array.from(atoms.token_id);
  const copy = { ...atoms, token_id: undefined };
  addGlobalTokenIdAnnotation(copy);
  let bad = 0;
  for (let i = 0; i < atoms.nAtoms; i += 1) if (copy.token_id[i] !== before[i]) bad += 1;
  return bad ? [{ check: 'global_token_id_inconsistent',
    message: `'token_id' annotation is inconsistent with recomputed values in ${bad} atom(s)`, count: bad }] : [];
}

function centroidAllOrNothing(atoms) {
  const centroid = column(atoms, 'condition_ChainCentroid_1_atom');
  if (!centroid) return [];
  const chain = atoms.chain_iid ?? atoms.chain_id;
  const chains = [...new Set(chain)].sort();
  let conditioned = 0;
  const partial = [];
  for (const c of chains) {
    let some = false;
    let all = true;
    for (let i = 0; i < atoms.nAtoms; i += 1) {
      if (chain[i] !== c) continue;
      const m = anyFinite3(centroid[i]);
      some ||= m;
      all &&= m;
    }
    if (some) {
      conditioned += 1;
      if (!all) partial.push(String(c));
    }
  }
  const out = [];
  if (partial.length) {
    out.push({ check: 'centroid_conditioning_partial_chain',
      message: `chain(s) [${partial.map((c) => `'${c}'`).join(', ')}] have a centroid on only some of their atoms; `
        + 'the centroid is a per-chain value and must cover every atom of the chain', count: partial.length });
  }
  if (chains.length && conditioned > 0 && conditioned < chains.length) {
    out.push({ check: 'centroid_conditioning_not_all_or_nothing',
      message: `${conditioned}/${chains.length} chain(s) have centroid conditioning; must be applied to all chains or none`,
      count: chains.length });
  }
  return out;
}

function conditioningCoordsNotNan(atoms) {
  const out = [];
  const n = atoms.nAtoms;
  const groups = column(atoms, 'condition_CoordinateGroup_1_atom');
  const internal = (i) => groups && groups[i] !== -1 && groups[i] !== 0 && groups[i] != null;
  const intcrd = column(atoms, 'condition_InternalCoordinate_1_atom');
  const crd = column(atoms, 'condition_coordinate_1_atom');
  let orphan = 0;
  for (let i = 0; i < n; i += 1) if (intcrd && anyFinite3(intcrd[i]) && !internal(i)) orphan += 1;
  if (orphan) {
    out.push({ check: 'internal_coords_require_coord_group',
      message: `${orphan} atom(s) have a C_INTCRD payload but no internal C_CRDGRP; `
        + 'coordinate groups are the source of truth for internal-coordinate conditioning', count: orphan });
  }
  const pairs = [[(i) => crd && anyFinite3(crd[i]), crd, 'C_CRD'], [internal, intcrd, 'C_INTCRD']];
  for (const [mask, values, label] of pairs) {
    let bad = 0;
    for (let i = 0; i < n; i += 1) if (mask(i) && !finite3(values?.[i])) bad += 1;
    if (bad) {
      out.push({ check: 'conditioning_coord_not_finite',
        message: `${bad} ${label}-conditioned atom(s) have non-finite conditioning coordinates`, count: bad });
    }
  }
  return out;
}

function expandableSegmentConditioning(atoms) {
  const segMin = column(atoms, 'annotation_expsegmin_1_residue');
  if (!segMin || !segMin.some((v) => v !== -1)) return [];
  const names = [];
  const tokens = new Set();
  for (const [name, def] of ATOM37_CONDITIONS) {
    const values = column(atoms, name);
    if (!values) continue;
    let any = false;
    for (let i = 0; i < atoms.nAtoms; i += 1) {
      if (segMin[i] === -1) continue;
      const nondefault = def === 'coord' ? !allNaN(values[i]) : values[i] !== def;
      if (nondefault) { any = true; tokens.add(atoms.token_id[i]); }
    }
    if (any) names.push(name);
  }
  if (!tokens.size) return [];
  return [{ check: 'expandable_segment_atom37_conditioning',
    message: `${tokens.size} expandable-segment token(s) have atom37 conditioning that would be discarded: `
      + `${names.join(', ')}. Remove the conditioning or provide ordinary polymer residues.`, count: tokens.size }];
}

function bondLengths(atoms) {
  const bonds = atoms.bonds ?? [];
  if (!bonds.length) return [];
  const n = atoms.nAtoms;
  const segMin = column(atoms, 'annotation_expsegmin_1_residue');
  const resolved = (i) => finite3(atoms.coord[i]) && !(segMin && segMin[i] !== -1)
    && (atoms.occupancy ? atoms.occupancy[i] > 0 : true);
  const chain = atoms.chain_iid ?? atoms.chain_id;
  const tokenType = atoms.token_type ?? new Array(n).fill(TOKEN_TYPE.UNKNOWN);
  const inter = ([i, j]) => chain[i] !== chain[j] || atoms.res_id[i] !== atoms.res_id[j]
    || (atoms.ins_code?.[i] ?? '') !== (atoms.ins_code?.[j] ?? '');
  const measured = bonds.map(([i, j]) => resolved(i) && resolved(j));
  const outgoing = new Uint8Array(n);
  const incoming = new Uint8Array(n);
  const kinds = new Set();
  bonds.forEach((b, k) => { if (measured[k] && inter(b)) for (const e of [b[0], b[1]]) kinds.add(`${atoms.res_name[e]}\u0000${tokenType[e]}`); });
  for (const kind of kinds) {
    const [resName, rt] = kind.split('\u0000');
    const [exitAtom, entryAtom] = getPolymerizationAtomNamesForToken(resName, Number(rt));
    for (let i = 0; i < n; i += 1) {
      if (atoms.res_name[i] !== resName || String(tokenType[i]) !== rt) continue;
      if (exitAtom && atoms.atom_name[i] === exitAtom) outgoing[i] = 1;
      if (entryAtom && atoms.atom_name[i] === entryAtom) incoming[i] = 1;
    }
  }
  const tooLong = [];
  let worst = 0;
  bonds.forEach((b, k) => {
    if (!measured[k]) return;
    const [i, j] = b;
    const d = Math.hypot(atoms.coord[i][0] - atoms.coord[j][0], atoms.coord[i][1] - atoms.coord[j][1],
      atoms.coord[i][2] - atoms.coord[j][2]);
    const polymer = inter(b) && ((outgoing[i] && incoming[j]) || (incoming[i] && outgoing[j]));
    const limit = polymer ? MAX_POLYMER_BOND : MAX_BOND;
    if (d > limit) { tooLong.push([i, j, d, limit]); worst = Math.max(worst, d); }
  });
  if (!tooLong.length) return [];
  const label = (i) => `${chain[i]}:${atoms.res_name[i]}${atoms.res_id[i]}${atoms.ins_code?.[i] ?? ''}:${atoms.atom_name[i]}`;
  const details = tooLong.slice(0, 5).map(([i, j, d, l]) => `${label(i)} -- ${label(j)} = ${d.toFixed(2)} Å (limit ${l.toFixed(1)} Å)`);
  return [{ check: 'bond_length',
    message: `${tooLong.length} bond(s) exceed the polymerization limit of ${MAX_POLYMER_BOND.toFixed(1)} Å `
      + `or general limit of ${MAX_BOND.toFixed(1)} Å; maximum length is ${worst.toFixed(2)} Å. `
      + `First ${details.length}: ${details.join('; ')}. Check source coordinates and residue numbering; `
      + 'preserve missing-residue gaps instead of bonding across them.', count: tooLong.length }];
}

function unindexedCoordConditioning(atoms) {
  const idx = column(atoms, 'condition_index_1_residue');
  if (!idx || idx.every(Boolean)) return [];
  const crd = column(atoms, 'condition_coordinate_1_atom');
  const intcrd = column(atoms, 'condition_InternalCoordinate_1_atom');
  const starts = [...residueStarts(atoms), atoms.nAtoms];
  const bad = [];
  for (let r = 0; r + 1 < starts.length; r += 1) {
    let unindexed = false;
    let coord = false;
    for (let i = starts[r]; i < starts[r + 1]; i += 1) {
      unindexed ||= !idx[i];
      coord ||= (crd && anyFinite3(crd[i])) || (intcrd && anyFinite3(intcrd[i]));
    }
    if (unindexed && !coord) bad.push(starts[r]);
  }
  if (!bad.length) return [];
  const labels = bad.slice(0, 10).map((s) => `${atoms.chain_id[s]}/${atoms.res_name[s]}/${atoms.res_id[s]}`);
  return [{ check: 'unindexed_must_be_coord_conditioned',
    message: `${bad.length} unindexed residue(s) have no coordinate conditioning on any atom: `
      + `${labels.join(', ')}${bad.length > labels.length ? ' ...' : ''}. To pad an index group without `
      + 'conditioning it, leave a res_id gap instead of including an unconditioned spacer residue.', count: bad.length }];
}

// ---- feature checks --------------------------------------------------------------------------

function tokenAtom37(batch, name, nTok) {
  const data = feat(batch, name);
  return data ? { data, per: data.length / nTok } : null;
}

function conditioning(atoms, batch, nTok) {
  const out = [];
  const tokenType = feat(batch, 'token_type');
  const seq = feat(batch, 'condition_sequence_1_residue');
  const mask = feat(batch, 'atom37_mask');
  const groups = feat(batch, 'condition_CoordinateGroup_1_atom');
  const crd = feat(batch, 'condition_coordinate_1_atom');
  const M = UNIFIED_TOKEN_TO_IDX['<M>'];
  let atomizedNotSeq = 0;
  for (let t = 0; t < nTok; t += 1) if (tokenType[t] === TOKEN_TYPE.ATOMIZED && seq[t] === M) atomizedNotSeq += 1;
  if (atomizedNotSeq) {
    out.push({ check: 'atomized_must_be_seq_conditioned',
      message: `${atomizedNotSeq} atomized token(s) lack sequence conditioning`, count: atomizedNotSeq });
  }
  const backbone = new Uint8Array(nTok * 37);
  const slots = buildAtom37Slots(atoms);
  for (let i = 0; i < atoms.nAtoms; i += 1) {
    if (slots[i] < 0) continue;
    backbone[atoms.token_id[i] * 37 + slots[i]] = (atoms.is_protein_backbone?.[i] || atoms.is_rna_backbone?.[i]
      || atoms.is_dna_backbone?.[i]) ? 1 : 0;
  }
  let nonBb = 0;
  for (let t = 0; t < nTok; t += 1) {
    let has = false;
    for (let s = 0; s < 37 && !has; s += 1) {
      const at = t * 37 + s;
      const c = Number.isFinite(crd[at * 3]) || Number.isFinite(crd[at * 3 + 1]) || Number.isFinite(crd[at * 3 + 2])
        || (groups[at] !== -1 && groups[at] !== 0);
      has = c && Boolean(mask[at]) && !backbone[at];
    }
    if (has && seq[t] === M) nonBb += 1;
  }
  if (nonBb) {
    out.push({ check: 'coord_conditioned_must_be_seq_conditioned',
      message: `${nonBb} token(s) with non-backbone coordinate conditioning lack sequence conditioning`, count: nonBb });
  }
  return out;
}

function atomizedIndexed(batch, nTok) {
  const tokenType = feat(batch, 'token_type');
  const idx = feat(batch, 'condition_index_1_residue');
  if (!idx) return [];
  let bad = 0;
  for (let t = 0; t < nTok; t += 1) {
    if (tokenType[t] !== TOKEN_TYPE.ATOMIZED) continue;
    let indexed = false;
    for (let s = 0; s < 37; s += 1) indexed ||= Boolean(idx[t * 37 + s]);
    if (!indexed) bad += 1;
  }
  return bad ? [{ check: 'atomized_must_be_indexed', message: `${bad} atomized token(s) are unindexed`, count: bad }] : [];
}

function indexGroupConnectivity(atoms, batch, nTok) {
  const pathLength = feat(batch, 'token_bond_path_length');
  if (!pathLength) return [];
  const groups = column(atoms, 'condition_index-group_1_token');
  const idx = column(atoms, 'condition_index_1_residue');
  const n = atoms.nAtoms;
  const group = (i) => (groups ? groups[i] : 0);
  const indexed = (i) => (idx ? Boolean(idx[i]) : true) && !atomized(atoms, i);
  const segMin = column(atoms, 'annotation_expsegmin_1_residue');
  const tokenType = atoms.token_type ?? new Array(n).fill(TOKEN_TYPE.UNKNOWN);
  const link = (i) => getPolymerizationAtomNamesForToken(atoms.res_name[i], Number(tokenType[i]));
  const standsIn = (i) => Boolean(segMin && segMin[i] !== -1);
  const leaving = (i) => atoms.atom_name[i] === link(i)[0] || standsIn(i);
  const entering = (i) => atoms.atom_name[i] === link(i)[1] || standsIn(i);
  for (const [i, j] of atoms.bonds ?? []) {
    const backbone = (leaving(i) && entering(j)) || (entering(i) && leaving(j));
    if (backbone && indexed(i) && indexed(j) && group(i) !== group(j)) {
      let count = 0;
      for (const [p, q] of atoms.bonds) {
        if (((leaving(p) && entering(q)) || (entering(p) && leaving(q))) && indexed(p) && indexed(q) && group(p) !== group(q)) count += 1;
      }
      return [{ check: 'backbone_bond_index_group',
        message: `Backbone-bonded ${atoms.chain_id[i]}${atoms.res_id[i]}-${atoms.chain_id[j]}${atoms.res_id[j]} `
          + `have different index groups (${group(i)}, ${group(j)}), hiding connectivity from the model. `
          + 'Use one index group for the continuous indexed scaffold.', count }];
    }
  }
  const starts = getTokenStarts(atoms);
  const molecule = atoms.molecule_iid;
  const pairs = [];
  for (let a = 0; a < nTok; a += 1) {
    for (let b = a + 1; b < nTok; b += 1) {
      if (group(starts[a]) === group(starts[b]) && molecule[starts[a]] === molecule[starts[b]]
        && pathLength[a * nTok + b] === 0) pairs.push([a, b]);
    }
  }
  if (!pairs.length) return [];
  const labels = pairs.slice(0, 10).map(([a, b]) => `${atoms.chain_id[starts[a]]}/${atoms.res_id[starts[a]]}`
    + `-${atoms.chain_id[starts[b]]}/${atoms.res_id[starts[b]]}`);
  return [{ check: 'index_group_molecule_must_be_connected',
    message: `${pairs.length} token pair(s) of one index group have no path between them in the bond graph though `
      + `they are one molecule: ${labels.join(', ')}${pairs.length > labels.length ? ' ...' : ''}`, count: pairs.length }];
}

function coordinatesNotNan(batch, nTok) {
  const coords = feat(batch, 'coords');
  const mask = feat(batch, 'atom37_mask');
  const crd = feat(batch, 'condition_coordinate_1_atom');
  const tokenType = feat(batch, 'token_type');
  if (!coords || !mask) return [];
  let nan = false;
  let inf = false;
  let nonzero = 0;
  for (let t = 0; t < nTok; t += 1) {
    for (let s = 0; s < 37; s += 1) {
      const at = t * 37 + s;
      const v = [coords[at * 3], coords[at * 3 + 1], coords[at * 3 + 2]];
      if (!mask[at]) {
        if (v.some((x) => x !== 0)) nonzero += 1;
        continue;
      }
      const conditioned = (crd && [crd[at * 3], crd[at * 3 + 1], crd[at * 3 + 2]].some(Number.isFinite))
        || (s === 1 && tokenType[t] === TOKEN_TYPE.ATOMIZED);
      if (!conditioned) continue;
      nan ||= v.some(Number.isNaN);
      inf ||= v.some((x) => x === Infinity || x === -Infinity);
    }
  }
  const out = [];
  if (nan) out.push({ check: 'nan_in_valid_coords', message: 'NaN found in atom coordinates', count: 1 });
  if (inf) out.push({ check: 'inf_in_valid_coords', message: 'Inf found in atom coordinates', count: 1 });
  if (nonzero) {
    out.push({ check: 'nonzero_padding_coords', message: `${nonzero} invalid atom slot(s) have non-zero coordinates`,
      count: nonzero });
  }
  return out;
}

function elementEncoding(batch, nTok) {
  const element = feat(batch, 'element');
  const mask = feat(batch, 'atom37_mask');
  const seq = feat(batch, 'condition_sequence_1_residue');
  if (!element || !mask || !seq) return [];
  const M = UNIFIED_TOKEN_TO_IDX['<M>'];
  let leaked = 0;
  let missing = 0;
  for (let t = 0; t < nTok; t += 1) {
    for (let s = 0; s < 37; s += 1) {
      const at = t * 37 + s;
      if (seq[t] === M) { if (element[at] !== -1) leaked += 1; } else if (mask[at] && element[at] < 0) missing += 1;
    }
  }
  const out = [];
  if (leaked) {
    out.push({ check: 'element_leakage', message: `${leaked} atom slot(s) in non-seq-conditioned tokens expose element info`,
      count: leaked });
  }
  if (missing) {
    out.push({ check: 'missing_element_in_seq_conditioned',
      message: `${missing} valid atom(s) in seq-conditioned tokens lack element info`, count: missing });
  }
  return out;
}

function sequenceConditioning(batch, nTok) {
  const seq = feat(batch, 'condition_sequence_1_residue');
  const residueType = feat(batch, 'residue_type');
  const tokenType = feat(batch, 'token_type');
  if (!seq || !residueType || !feat(batch, 'atom37_mask')) return [];
  const M = UNIFIED_TOKEN_TO_IDX['<M>'];
  const A = UNIFIED_TOKEN_TO_IDX['<A>'];
  const nTypes = Object.keys(UNIFIED_TOKEN_TO_IDX).length;
  const out = [];
  let range = false;
  let mismatch = 0;
  let notA = 0;
  for (let t = 0; t < nTok; t += 1) {
    if (residueType[t] < 0 || residueType[t] >= nTypes) range = true;
    if (seq[t] !== M && residueType[t] !== seq[t]) mismatch += 1;
    if (tokenType[t] === TOKEN_TYPE.ATOMIZED && seq[t] !== A) notA += 1;
  }
  if (range) out.push({ check: 'residue_type_out_of_range', message: 'Ground-truth residue_type contains out-of-range indices', count: 1 });
  if (mismatch) {
    out.push({ check: 'seq_conditioned_residue_type_mismatch',
      message: `${mismatch} seq-conditioned token(s): condition_sequence != residue_type`, count: mismatch });
  }
  // `non_seq_conditioned_not_masked` cannot fire: a token is non-seq-conditioned exactly when its
  // condition_sequence is <M>.
  if (notA) {
    out.push({ check: 'atomized_condition_sequence_not_A', message: `${notA} atomized token(s) have condition_sequence != <A>`,
      count: notA });
  }
  return out;
}

function sequenceProbabilities(atoms) {
  const out = [];
  const starts = residueStarts(atoms);
  const stops = [...starts.slice(1), atoms.nAtoms];
  const R = starts.length;
  const probs = Array.from({ length: R }, () => new Array(SEQ_PROB_TOKENS.length).fill(0));
  const spec = Array.from({ length: R }, () => new Array(SEQ_PROB_TOKENS.length).fill(false));
  SEQ_PROB_TOKENS.forEach((tok, c) => {
    const values = column(atoms, `condition_sequence-probability-${tok}_1_residue`);
    if (!values) return;
    let nonuniformMask = 0;
    let nonuniformValue = 0;
    for (let r = 0; r < R; r += 1) {
      const m0 = !Number.isNaN(values[starts[r]]);
      let maskBad = false;
      let valueBad = false;
      for (let i = starts[r]; i < stops[r]; i += 1) {
        const m = !Number.isNaN(values[i]);
        if (m !== m0) maskBad = true;
        if (m && !isClose(values[i], values[starts[r]], true)) valueBad = true;
      }
      nonuniformMask += maskBad;
      nonuniformValue += valueBad;
      probs[r][c] = values[starts[r]];
      spec[r][c] = m0;
    }
    if (nonuniformMask) {
      out.push({ check: 'sequence_probability_nonuniform_residue_mask',
        message: `sequence-probability-${tok} has a nonuniform mask in ${nonuniformMask} residue(s)`, count: nonuniformMask });
    }
    if (nonuniformValue) {
      out.push({ check: 'sequence_probability_nonuniform_residue_value',
        message: `sequence-probability-${tok} has nonuniform values in ${nonuniformValue} residue(s)`, count: nonuniformValue });
    }
  });
  const cSeq = column(atoms, 'condition_sequence_1_residue');
  const seqConditioned = starts.map((s) => Boolean(cSeq && cSeq[s] !== '<M>'));
  if (!spec.some((row) => row.some(Boolean)) && !seqConditioned.some(Boolean)) return out;
  const push = (check, n, message) => { if (n) out.push({ check, message, count: n }); };
  let nonfinite = 0;
  let outOfRange = 0;
  let wrongType = 0;
  let exceeds = 0;
  let below = 0;
  let incomplete = 0;
  let conflicting = 0;
  const kind = (s) => {
    const r = atoms.res_name[s];
    return {
      aa: STANDARD_AA.has(r) || r === 'UNK' || r === '<M>',
      rna: STANDARD_RNA.has(r) || r === 'N' || r === '<M>',
      dna: STANDARD_DNA.has(r) || r === 'DN' || r === '<M>',
    };
  };
  for (let r = 0; r < R; r += 1) {
    const k = kind(starts[r]);
    const compatible = SEQ_PROB_TOKENS.map((t) => (STANDARD_AA.has(t) ? k.aa : STANDARD_RNA.has(t) ? k.rna : k.dna));
    let mass = 0;
    for (let c = 0; c < SEQ_PROB_TOKENS.length; c += 1) {
      const p = probs[r][c];
      const fin = Number.isFinite(p);
      if (spec[r][c] && !fin) nonfinite += 1;
      if (spec[r][c] && fin && (p < 0 || p > 1)) outOfRange += 1;
      if (spec[r][c] && fin && p > 0 && !compatible[c]) wrongType += 1;
      if (spec[r][c] && fin) mass += p;
    }
    if (mass > 1 + 1e-6) exceeds += 1;
    const fully = compatible.some(Boolean) && compatible.every((cmp, c) => spec[r][c] || !cmp);
    const target = cSeq ? cSeq[starts[r]] : '<M>';
    const noncanonical = seqConditioned[r] && !STANDARD_POLYMER.has(target);
    if (fully && !noncanonical && mass < 1 - 1e-6) below += 1;
    if (seqConditioned[r] && !spec[r].every(Boolean)) incomplete += 1;
    if (seqConditioned[r] && SEQ_PROB_TOKENS.some((t, c) => spec[r][c]
      && !isClose(probs[r][c], target === t ? 1 : 0))) conflicting += 1;
  }
  push('sequence_probability_nonfinite', nonfinite, `${nonfinite} specified sequence probability value(s) are not finite`);
  push('sequence_probability_out_of_range', outOfRange, `${outOfRange} specified sequence probability value(s) fall outside [0, 1]`);
  push('sequence_probability_wrong_token_type', wrongType,
    `${wrongType} positive sequence probability value(s) target identities from a different polymer type`);
  push('sequence_probability_sum_exceeds_one', exceeds, `specified sequence probabilities sum above 1 on ${exceeds} residue(s)`);
  push('sequence_probability_sum_below_one', below, `fully specified sequence probabilities sum below 1 on ${below} residue(s)`);
  push('sequence_probability_incomplete_for_c_seq', incomplete,
    `${incomplete} C_SEQ-conditioned residue(s) do not have all canonical-polymer probabilities explicitly specified`);
  push('sequence_probability_conflicts_with_c_seq', conflicting,
    `${conflicting} residue(s) have sequence probabilities that conflict with C_SEQ`);
  return out;
}

// ---- cross-cutting ---------------------------------------------------------------------------

function centering(atoms, batch, nTok) {
  const noised = atoms.coord_to_be_noised;
  const crd = column(atoms, 'condition_coordinate_1_atom');
  if (!noised || !crd) return [];
  const sum = [0, 0, 0];
  let count = 0;
  for (let i = 0; i < atoms.nAtoms; i += 1) {
    if (!anyFinite3(crd[i])) continue;
    for (let c = 0; c < 3; c += 1) sum[c] += noised[i][c];
    count += 1;
  }
  if (!count) return [];
  const out = [];
  const norm = Math.hypot(sum[0] / count, sum[1] / count, sum[2] / count);
  if (!(norm < 1e-3)) {
    out.push({ check: 'coordinates_not_centered', message: `Atom-array centroid norm ${norm.toExponential(2)} (threshold 1e-3)`, count: 1 });
  }
  const coords = feat(batch, 'coords');
  const fcrd = feat(batch, 'condition_coordinate_1_atom');
  const mask = feat(batch, 'atom37_mask');
  if (!coords || !fcrd || !mask) return out;
  const fsum = [0, 0, 0];
  let fn = 0;
  for (let at = 0; at < nTok * 37; at += 1) {
    if (!mask[at] || ![fcrd[at * 3], fcrd[at * 3 + 1], fcrd[at * 3 + 2]].some(Number.isFinite)) continue;
    for (let c = 0; c < 3; c += 1) fsum[c] += coords[at * 3 + c];
    fn += 1;
  }
  if (fn) {
    const fnorm = Math.hypot(fsum[0] / fn, fsum[1] / fn, fsum[2] / fn);
    if (!(fnorm < 1e-3)) {
      out.push({ check: 'feats_coordinates_not_centered', message: `Feats centroid norm ${fnorm.toExponential(2)} (threshold 1e-3)`, count: 1 });
    }
  }
  return out;
}

function supportsLocalLatent(atoms, batch, nTok) {
  const support = feat(batch, 'supports_local_latent');
  if (!support) return [];
  const starts = getTokenStarts(atoms, true);
  const idx = column(atoms, 'condition_index_1_residue');
  const segMin = column(atoms, 'annotation_expsegmin_1_residue');
  const mask = feat(batch, 'atom37_mask');
  let missingRef = 0;
  let invalid = 0;
  let missing = 0;
  for (let t = 0; t + 1 < starts.length; t += 1) {
    const s = starts[t];
    let inContext = false;
    for (let i = s; i < starts[t + 1]; i += 1) inContext ||= Boolean(idx && !idx[i]);
    const expected = LOCAL_LATENT_RESIDUE_SET.has(atoms.res_name[s]) && !atomized(atoms, s) && !inContext;
    const has = Boolean(support[t]);
    if (mask && has && !mask[t * 37 + 1] && !(segMin && segMin[s] !== -1)) missingRef += 1;
    if (has && !expected) invalid += 1;
    if (!has && expected) missing += 1;
  }
  const out = [];
  if (missingRef) {
    out.push({ check: 'local_latent_reference_atom_missing',
      message: `${missingRef} local-latent token(s) lack their atom37 slot-1 reference atom`, count: missingRef });
  }
  if (invalid) {
    out.push({ check: 'supports_local_latent_ineligible',
      message: `${invalid} token(s) have supports_local_latent=True but are not eligible`, count: invalid });
  }
  if (missing) {
    out.push({ check: 'supports_local_latent_missing', message: `${missing} eligible token(s) have supports_local_latent=False`,
      count: missing });
  }
  return out;
}

function noOxtOnLocalLatent(atoms, batch) {
  const support = feat(batch, 'supports_local_latent');
  if (!support) return [];
  const chain = atoms.chain_iid ?? atoms.chain_id;
  const residues = new Map();
  const starts = residueStarts(atoms);
  for (const s of starts) residues.set(chain[s], (residues.get(chain[s]) ?? 0) + 1);
  let bad = 0;
  for (let i = 0; i < atoms.nAtoms; i += 1) {
    if (atoms.atom_name[i] !== 'OXT' || atomized(atoms, i) || !support[atoms.token_id[i]]) continue;
    if (PROTEIN_CHAINS.has(atoms.chain_type?.[i]) && residues.get(chain[i]) > 1) bad += 1;
  }
  return bad ? [{ check: 'oxt_on_local_latent_token', message: `${bad} OXT atom(s) found on supports_local_latent non-atomized token(s)`,
    count: bad }] : [];
}

/**
 * `detect_batch_invariant_violations`, at inference.
 * @param {object} atoms the finished atom array
 * @param {object} batch the batch built from it
 * @returns {{check: string, message: string, count: number}[]}
 */
export function batchInvariantViolations(atoms, batch) {
  const nTok = feat(batch, 'token_type').length;
  return [
    ...moleculesOnSeparateChains(atoms),
    ...standardAnnotationLevels(atoms),
    ...globalTokenIdConsistent(atoms),
    ...centroidAllOrNothing(atoms),
    ...conditioningCoordsNotNan(atoms),
    ...expandableSegmentConditioning(atoms),
    ...bondLengths(atoms),
    ...unindexedCoordConditioning(atoms),
    ...conditioning(atoms, batch, nTok),
    ...atomizedIndexed(batch, nTok),
    ...indexGroupConnectivity(atoms, batch, nTok),
    ...coordinatesNotNan(batch, nTok),
    ...elementEncoding(batch, nTok),
    ...sequenceConditioning(batch, nTok),
    ...sequenceProbabilities(atoms),
    ...centering(atoms, batch, nTok),
    ...supportsLocalLatent(atoms, batch, nTok),
    ...noOxtOnLocalLatent(atoms, batch),
  ];
}

/** `ValidateBatchInvariants(on_violation="raise")`: upstream's message, check by check. */
export function assertBatchInvariants(atoms, batch, exampleId = 'design') {
  const violations = batchInvariantViolations(atoms, batch);
  if (!violations.length) return;
  const lines = violations.map((v) => `  [${v.check}] ${v.message}`).join('\n');
  throw new Error(`Batch invariant violation(s) for example '${exampleId}':\n${lines}`);
}
