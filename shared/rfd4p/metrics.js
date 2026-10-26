/**
 * A design's metrics, as upstream's inline metrics compute them, and the pre-oracle verdict.
 *
 * Mirrors RFD4-Proteina's `configs/metrics/inline_metrics/design_metrics.yaml` and
 * `configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml` - the metric classes in
 * `rfproteina/metrics/` (amino_acid_composition, design_metrics, sequence_adherence_metric,
 * unindexed_insertion_metric, sse_metrics, hotspots) on the array `process_single_sample_to_atom_array`
 * builds, keyed as upstream keys them (`<metric>.<value>`). tools/check-rfd4p-metrics.py holds
 * every value to upstream's own metric manager on the same sample.
 *
 * 🔴 UPSTREAM'S RELEASED GENERATOR COMPUTES NONE OF THIS BY DEFAULT. `rfd4-generate` builds a
 * metric manager only when a `metrics` config is passed - evaluation and training do - and then
 * gates every metric on the job's `metrics_tags`, so an untagged job gets the amino-acid
 * composition and the unknown-residue filter and nothing else. That gate is reproduced
 * (`metricsTags`), not widened: a number upstream would not have reported is not reported.
 *
 * 🔴 AND THE MOTIF RMSDs (`design_motif_allatom_rmsd`, its unindexed twin, `ca_rmsd_to_input`)
 * RESOLVE SYMMETRY FIRST, as upstream's `StructureComparisonMetric` does: identical subunits and
 * chemically equivalent atoms (a carboxylate's oxygens, a ring's flip) are relabelled onto the
 * design before the fit (shared/rfd4p/symmetry.js), so a flipped Asp is not a motif failure. The
 * reference is upstream's `SA_CWI`, staged per coordinate group (`coordsWithInsertion`). The
 * filters that read one (`motif_rmsd_filter`, `insertion_rmsd_filter`) name keys no upstream
 * metric produces (`motif_rmsd.rmsd`, `unindexed_metrics.insertion_rmsd`), so they stay
 * indeterminate on both sides.
 *
 * The array: `processSampleToAtoms`' atoms (which tools/check-rfd4p-metrics.py also holds to
 * upstream's, atom for atom), each reading the input's annotations through the input atom it
 * came from (`inputIndex`), its token's first (`tokenInputIndex`), or - on a grafted unindexed
 * token - the motif atom it was grafted from (`condSource`). Its bonds are upstream's: the
 * input's, among the atoms that survive, plus biotite's `connect_via_residue_names` whenever a
 * generated residue added atoms (`copy_generated_coordinates_to_input`).
 */

import { BOND_TYPE, CHAIN_TYPE, STD_RESIDUE_BONDS } from './pipeline/cif.js';
import { superimposeCoords3D } from './output.js';
import { permuteGtAtoms, permuteGtSubunits } from './symmetry.js';

const MASKED = '<M>';
const STANDARD_AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
  'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'];
/** `SEQUENCE_PROBABILITY_CONDITIONS`, in the unified encoding's channel order. */
export const SEQUENCE_PROBABILITY_TOKENS = [...STANDARD_AA, 'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'];
const SSE = { NONE: 0, HELIX: 1, SHEET: 2, LOOP: 3 };

/** The CCD's `_chem_comp.type` of the residues a design is made of (biotite's `link_type`). */
const STANDARD_LINK = Object.fromEntries([
  ...STANDARD_AA.map((r) => [r, r === 'GLY' ? 'PEPTIDE LINKING' : 'L-PEPTIDE LINKING']),
  ['UNK', 'L-PEPTIDE LINKING'],
  ...['A', 'C', 'G', 'U', 'N'].map((r) => [r, 'RNA LINKING']),
  ...['DA', 'DC', 'DG', 'DT', 'DN'].map((r) => [r, 'DNA LINKING']),
]);
const PEPTIDE_LINKS = new Set(['PEPTIDE LINKING', 'L-PEPTIDE LINKING', 'D-PEPTIDE LINKING']);
const NUCLEIC_LINKS = new Set(['RNA LINKING', 'DNA LINKING']);
/** biotite's `amino_acid_names()`: every CCD type it calls an amino acid. */
const AMINO_ACID_TYPES = new Set(['D-BETA-PEPTIDE, C-GAMMA LINKING', 'D-GAMMA-PEPTIDE, C-DELTA LINKING',
  'D-PEPTIDE LINKING', 'D-PEPTIDE NH3 AMINO TERMINUS', 'L-BETA-PEPTIDE, C-GAMMA LINKING',
  'L-GAMMA-PEPTIDE, C-DELTA LINKING', 'L-PEPTIDE COOH CARBOXY TERMINUS', 'L-PEPTIDE LINKING',
  'L-PEPTIDE NH3 AMINO TERMINUS', 'PEPTIDE LINKING']);

/**
 * Where each metric applies, from design_metrics.yaml: `any` (one of), `all`, `not`. A metric
 * runs when the job's `metrics_tags` pass its gate (`Metric.applies_to_tags`).
 */
export const METRIC_GATES = {
  amino_acid_composition: {},
  bond_length_metrics: { any: ['design'], not: ['small_molecule_only', 'structure_prediction'] },
  clash_metrics: { any: ['design', 'structure_prediction'], not: ['small_molecule_only'] },
  hotspot_obedience: { any: ['hotspot_conditioning'] },
  sequence_adherence: { any: ['design'] },
  sequence_probability_adherence: { any: ['design'] },
  sse_distribution: { any: ['design'], not: ['small_molecule_only', 'structure_prediction'] },
  sse_accuracy: { any: ['design'], not: ['small_molecule_only', 'structure_prediction'] },
  unindexed_insertion_mapping: { all: ['unindexed'] },
  sequence_similarity_to_input: { any: ['partial_diffusion'] },
  design_motif_allatom_rmsd: { any: ['3d_coordinate_conditioning', 'unindexed', 'internal_coordinate_conditioning'] },
  design_unindexed_motif_allatom_rmsd: { all: ['unindexed'] },
  ca_rmsd_to_input: { any: ['partial_diffusion'] },
};

/** all_pre_oracle_filters.yaml: each filter's gate and its thresholds (inclusive `max`). */
export const PRE_ORACLE_FILTERS = {
  unknown_residue_filter: {
    gate: {}, thresholds: { 'amino_acid_composition.n_generated_unknown_residues': 0 },
  },
  sequence_adherence_filter: {
    gate: { any: ['design'] }, thresholds: { 'sequence_adherence.n_violations': 0 },
  },
  unindexed_insertion_mapping_filter: {
    gate: { all: ['unindexed'] },
    thresholds: {
      'unindexed_insertion_mapping.n_unmapped_tokens': 0,
      'unindexed_insertion_mapping.n_sequence_mismatches': 0,
    },
  },
  insertion_rmsd_filter: {
    gate: { all: ['unindexed', 'insertion_rmsd_filter'] }, thresholds: { 'unindexed_metrics.insertion_rmsd': 1.0 },
  },
  motif_rmsd_filter: { gate: { all: ['motif_rmsd_filter'] }, thresholds: { 'motif_rmsd.rmsd': 1.0 } },
  ligand_clash_filter: {
    gate: { all: ['ligand_clash_filter'], not: ['no_ligand'] },
    thresholds: { 'clash_metrics.frac_atoms_clashing_inter_residue_polymer_ligand_1p5A': 0 },
  },
};

/**
 * A job's `metrics_tags`, with what upstream's generator adds itself: `partial_diffusion` for a
 * run that starts from the input (inverse folding, or a partial time / trajectory fraction).
 */
export function jobMetricsTags(job) {
  const tags = [...(job.metrics_tags ?? job.metricsTags ?? [])];
  const partial = Boolean(job.inverseFolding ?? job.inverse_folding)
    || [job.partialTBackbone ?? job.partial_t_backbone, job.partialTLatent ?? job.partial_t_latent,
      job.partialTrajectoryFractionSteps ?? job.partial_trajectory_fraction_steps].some((v) => v != null);
  if (partial && !tags.includes('partial_diffusion')) tags.push('partial_diffusion');
  return tags;
}

/**
 * The tags upstream's own validation sets give a task conditioned the way this one is
 * (`configs/datasets/val/`: motif scaffolding is `3d_coordinate_conditioning`, with
 * `internal_coordinate_conditioning` for internal groups and `unindexed` for an unindexed motif;
 * PPI adds `hotspot_conditioning`) - read off the batch's input, so a motif's RMSD is reported
 * where there is a motif and nowhere else (on a design with none, upstream's raises).
 * @param {object} input the batch's input atom table
 */
export function conditioningMetricsTags(input) {
  const n = input.atom_name.length;
  const crd = input.condition_coordinate_1_atom;
  const group = input.condition_CoordinateGroup_1_atom;
  const idx = input.condition_index_1_residue;
  const hot = input.condition_hotspot_1_atom;
  let absolute = false;
  let internal = false;
  let unindexed = false;
  let hotspot = false;
  for (let i = 0; i < n; i += 1) {
    absolute ||= Array.isArray(crd?.[i]) && crd[i].every(Number.isFinite);
    internal ||= group?.[i] != null && group[i] !== -1 && group[i] !== 0;
    unindexed ||= idx ? !idx[i] : false;
    hotspot ||= Boolean(hot?.[i]);
  }
  return [
    ...(absolute || internal ? ['3d_coordinate_conditioning'] : []),
    ...(internal ? ['internal_coordinate_conditioning'] : []),
    ...(unindexed ? ['unindexed'] : []),
    ...(hotspot ? ['hotspot_conditioning'] : []),
  ];
}

/**
 * Every tag a ported metric or filter gates on, in a fixed order: the native port carries a
 * job's tags as these bits (a tag outside the list gates nothing here, so dropping it loses
 * nothing).
 */
export const METRICS_TAGS = ['design', 'structure_prediction', 'small_molecule_only', 'hotspot_conditioning',
  'unindexed', 'partial_diffusion', 'insertion_rmsd_filter', 'motif_rmsd_filter', 'ligand_clash_filter',
  'no_ligand', '3d_coordinate_conditioning', 'internal_coordinate_conditioning'];
export function metricsTagBits(tags) {
  return METRICS_TAGS.reduce((bits, t, k) => (tags.includes(t) ? bits | (1 << k) : bits), 0);
}

export function appliesToTags(gate, tags) {
  const has = new Set(tags);
  return (!gate.all || gate.all.every((t) => has.has(t)))
    && (!gate.any || gate.any.some((t) => has.has(t)))
    && !(gate.not ?? []).some((t) => has.has(t));
}

const f32 = Math.fround;
/** `np.linalg.norm` of a float32 difference: squares summed in float32, then the root. */
function dist32(c, i, j) {
  const dx = f32(c[i * 3] - c[j * 3]);
  const dy = f32(c[i * 3 + 1] - c[j * 3 + 1]);
  const dz = f32(c[i * 3 + 2] - c[j * 3 + 2]);
  return f32(Math.sqrt(f32(f32(f32(dx * dx) + f32(dy * dy)) + f32(dz * dz))));
}

/** Segment starts where any of `keys` changes (biotite's `get_segment_starts`). */
function starts(arr, keys) {
  const out = arr.n ? [0] : [];
  for (let i = 1; i < arr.n; i += 1) {
    if (keys.some((k) => arr[k][i] !== arr[k][i - 1])) out.push(i);
  }
  return out;
}
/** atomworks' `get_residue_starts` (and the transformation id); biotite's has the residue name. */
const residueStarts = (arr) => starts(arr, ['chainId', 'resId', 'insCode', 'transformationId']);
const biotiteResidueStarts = (arr) => starts(arr, ['chainId', 'resId', 'insCode']);
const tokenStarts = (arr) => starts(arr, ['tokenId']);
const spread = (n, st, values) => {
  const out = new Array(n);
  for (let r = 0; r < st.length; r += 1) {
    for (let i = st[r]; i < (st[r + 1] ?? n); i += 1) out[i] = values[r];
  }
  return out;
};

const finite3 = (v) => Array.isArray(v) && v.length === 3 && v.every((x) => x !== null && Number.isFinite(Number(x)));

/**
 * The generated array, as columns, with the input's annotations read through each atom's source.
 * @param {object} structure `processSampleToAtoms`' result
 * @param {object} input the batch's input atom table (`buildBatch(...).atoms`)
 */
export function designArray(structure, input) {
  const atoms = structure.atoms;
  const n = atoms.length;
  const col = (name, at) => (at >= 0 ? input[name]?.[at] : undefined);
  const arr = { n, coord: new Float32Array(n * 3) };
  const fields = ['resName', 'atomName', 'element', 'chainId', 'resId', 'insCode', 'transformationId',
    'tokenId', 'chainIid', 'isPolymer', 'isProtein', 'atomize', 'chemCompType', 'crd', 'internal', 'hot',
    'cSeq', 'cSec', 'uii', 'inputIndex', 'charge', 'coordGroup', 'crdsrc', 'pnUnit', 'molIid', 'molEntity',
    'condSource', 'crdValue', 'intCrd'];
  for (const f of fields) arr[f] = new Array(n);
  arr.seqProb = SEQUENCE_PROBABILITY_TOKENS.map(() => new Float64Array(n));
  for (let i = 0; i < n; i += 1) {
    const a = atoms[i];
    const tok = a.tokenInputIndex ?? -1;
    // A grafted atom reads every 1-body condition from the motif atom it was grafted from;
    // otherwise an atom reads the input atom it is, a residue-level one its token's first.
    const atomSrc = a.condSource ?? a.inputIndex ?? -1;
    const resSrc = a.condSource ?? tok;
    arr.coord[i * 3] = a.x;
    arr.coord[i * 3 + 1] = a.y;
    arr.coord[i * 3 + 2] = a.z;
    arr.resName[i] = a.resName;
    arr.atomName[i] = a.atomName;
    arr.element[i] = String(a.element ?? '').toUpperCase();
    arr.chainId[i] = a.chainId;
    arr.resId[i] = a.resId;
    arr.insCode[i] = a.insCode ?? '';
    arr.transformationId[i] = String(col('transformation_id', tok) ?? '');
    arr.tokenId[i] = a.tokenId;
    arr.chainIid[i] = String(col('chain_iid', tok) ?? a.chainId);
    arr.isPolymer[i] = Boolean(col('is_polymer', tok));
    arr.isProtein[i] = Boolean(col('is_protein', tok));
    arr.atomize[i] = Boolean(col('atomize', tok));
    arr.chemCompType[i] = String(col('chem_comp_type', tok) ?? '').toUpperCase();
    arr.crd[i] = Boolean(a.crd);
    arr.internal[i] = a.coordGroup !== -1 && a.coordGroup !== 0 && a.coordGroup != null;
    arr.hot[i] = Boolean(col('condition_hotspot_1_atom', atomSrc));
    arr.cSeq[i] = String(col('condition_sequence_1_residue', resSrc) ?? MASKED);
    arr.cSec[i] = Number(col('condition_secondary-structure_1_residue', resSrc) ?? SSE.NONE);
    SEQUENCE_PROBABILITY_TOKENS.forEach((t, k) => {
      const v = col(`condition_sequence-probability-${t}_1_residue`, resSrc);
      arr.seqProb[k][i] = v == null ? NaN : Number(v);
    });
    arr.uii[i] = Boolean(a.isUnindexedInsertion);
    arr.inputIndex[i] = a.inputIndex ?? -1;
    arr.charge[i] = Number(a.charge ?? 0);
    arr.coordGroup[i] = a.coordGroup ?? -1;
    arr.condSource[i] = a.condSource ?? -1;
    // float32, as upstream's annotations hold them and the native port's tables carry them.
    arr.crdValue[i] = a.crd && finite3(a.crd) ? a.crd.map(Math.fround) : null;
    arr.intCrd[i] = a.intCrd && finite3(a.intCrd) ? a.intCrd.map(Math.fround) : null;
    // `SA_CRDSRC`: a grafted motif atom carries its unindexed token's source id, any other
    // coordinate-conditioned atom (and a grafted one whose token had none) its own `SA_SRC`
    // (`_merge_indexed_coord_conditioning_into_src`); "." is unconditioned.
    const own = arr.crd[i] || arr.internal[i] ? (a.srcComponent ?? '.') : '.';
    const grafted = a.crdSrcComponent ?? '.';
    arr.crdsrc[i] = (grafted !== '.' && a.condSource != null ? grafted : own) !== '.';
    arr.pnUnit[i] = String(col('pn_unit_iid', tok) ?? arr.chainIid[i]);
    arr.molIid[i] = String(col('molecule_iid', tok) ?? arr.chainIid[i]);
    arr.molEntity[i] = String(col('molecule_entity', tok) ?? '');
  }
  arr.hasSec = input['condition_secondary-structure_1_residue'] !== undefined;
  arr.bonds = designBonds(arr, input);
  return arr;
}

/** A residue's CCD type, by name (`link_type`); the input's annotation for one we do not table. */
function linkType(arr, i) {
  return STANDARD_LINK[arr.resName[i]] ?? arr.chemCompType[i];
}

/**
 * The array's bonds, as upstream's predicted array carries them: the input's own between atoms
 * that survive into the design, merged with biotite's `connect_via_residue_names` over the
 * non-atomised atoms whenever a generated residue added atoms the input lacked.
 */
function designBonds(arr, input) {
  const n = arr.n;
  const keys = new Map();
  const out = [];
  // A bond both lists carry takes the second list's type (biotite's `BondList.merge`).
  const add = (i, j, type) => {
    const [a, b] = i < j ? [i, j] : [j, i];
    const key = a * n + b;
    if (a === b) return;
    if (keys.has(key)) { out[keys.get(key)][2] = type; return; }
    keys.set(key, out.length);
    out.push([a, b, type]);
  };
  const outOf = new Map();
  for (let i = 0; i < n; i += 1) if (arr.inputIndex[i] >= 0) outOf.set(arr.inputIndex[i], i);
  for (const [p, q, type] of input.bonds ?? []) {
    const i = outOf.get(p);
    const j = outOf.get(q);
    if (i !== undefined && j !== undefined) add(i, j, type ?? BOND_TYPE.SINGLE);
  }
  let inserted = false;
  for (let i = 0; i < n; i += 1) if (arr.inputIndex[i] < 0 && !arr.atomize[i]) inserted = true;
  if (!inserted) return out;
  const sub = [];
  for (let i = 0; i < n; i += 1) if (!arr.atomize[i]) sub.push(i);
  const view = { n: sub.length };
  for (const k of ['chainId', 'resId', 'insCode']) view[k] = sub.map((i) => arr[k][i]);
  const st = biotiteResidueStarts(view);
  st.push(sub.length);
  for (let r = 0; r + 1 < st.length; r += 1) {
    const atomsIn = sub.slice(st[r], st[r + 1]);
    for (const [p, q, type] of STD_RESIDUE_BONDS[arr.resName[atomsIn[0]]] ?? []) {
      for (const i of atomsIn) {
        if (arr.atomName[i] !== p) continue;
        for (const j of atomsIn) if (arr.atomName[j] === q) add(i, j, type);
      }
    }
  }
  for (let r = 0; r + 2 < st.length; r += 1) {
    const cur = sub[st[r]];
    const nxt = sub[st[r + 1]];
    if (arr.chainId[nxt] !== arr.chainId[cur] || arr.resId[nxt] - arr.resId[cur] > 1) continue;
    const a = linkType(arr, cur);
    const b = linkType(arr, nxt);
    let names;
    if (PEPTIDE_LINKS.has(a) && PEPTIDE_LINKS.has(b)) names = ['C', 'N'];
    else if (NUCLEIC_LINKS.has(a) && NUCLEIC_LINKS.has(b)) names = ["O3'", 'P'];
    else continue;
    const i = sub.slice(st[r], st[r + 1]).find((k) => arr.atomName[k] === names[0]);
    const j = sub.slice(st[r + 1], st[r + 2]).find((k) => arr.atomName[k] === names[1]);
    if (i !== undefined && j !== undefined) add(i, j, BOND_TYPE.SINGLE);
  }
  return out;
}

/** `AminoAcidCompositionMetric` */
function aminoAcidComposition(arr) {
  const st = residueStarts(arr);
  let unknown = 0;
  const std = [];
  for (const s of st) {
    const name = arr.resName[s];
    const inputUnknown = arr.cSeq[s] !== MASKED && arr.cSeq[s] === 'UNK';
    if (name === 'UNK' && !inputUnknown) unknown += 1;
    if (STANDARD_AA.includes(name)) std.push(name);
  }
  const frac = (r) => (std.length ? std.filter((x) => x === r).length / std.length : NaN);
  return {
    n_generated_unknown_residues: unknown,
    n_standard_residues: std.length,
    ala_fraction: frac('ALA'),
    val_fraction: frac('VAL'),
    gly_fraction: frac('GLY'),
  };
}

/** `SequenceAdherenceMetric` */
function sequenceAdherence(arr) {
  const st = residueStarts(arr).filter((s) => arr.cSeq[s] !== MASKED);
  if (st.length === 0) return { sequence_adherence: NaN, n_violations: 0 };
  const bad = st.filter((s) => arr.cSeq[s] !== arr.resName[s]).length;
  return { sequence_adherence: 1 - bad / st.length, n_violations: bad };
}

/** `SequenceProbabilityAdherenceMetric`: the exact 0 and 1 constraints. */
function sequenceProbabilityAdherence(arr) {
  let nZero = 0;
  let nOne = 0;
  let zeroBad = 0;
  let oneBad = 0;
  for (const s of residueStarts(arr)) {
    SEQUENCE_PROBABILITY_TOKENS.forEach((t, k) => {
      const v = arr.seqProb[k][s];
      if (Number.isNaN(v)) return;
      const is = arr.resName[s] === t;
      if (v === 0) { nZero += 1; if (is) zeroBad += 1; }
      if (v === 1) { nOne += 1; if (!is) oneBad += 1; }
    });
  }
  return {
    zero_adherence: nZero ? 1 - zeroBad / nZero : NaN,
    one_adherence: nOne ? 1 - oneBad / nOne : NaN,
    n_zero_conditions: nZero,
    n_zero_violations: zeroBad,
    n_one_conditions: nOne,
    n_one_violations: oneBad,
  };
}

/**
 * `UnindexedInsertionMappingMetric`. 🔴 The sequence mismatches are counted over the PROPOSED
 * mapping even where it could not be applied - upstream hands the metric the mapping, not the
 * annotation's outcome - while the unmapped count reads which tokens were actually grafted.
 */
function unindexedInsertionMapping(arr, input, mapping) {
  const idx = input.condition_index_1_residue;
  const tokens = [];
  if (idx) {
    let prev = null;
    for (let i = 0; i < input.token_id.length; i += 1) {
      if (idx[i] || input.token_id[i] === prev) continue;
      prev = input.token_id[i];
      tokens.push(i);
    }
  }
  if (tokens.length === 0) return { n_unmapped_tokens: 0, n_sequence_mismatches: 0 };
  const mapped = new Set();
  for (let i = 0; i < arr.n; i += 1) if (arr.uii[i]) mapped.add(arr.tokenId[i]);
  const generated = new Map();
  for (const s of tokenStarts(arr)) generated.set(arr.tokenId[s], arr.resName[s]);
  let mismatches = 0;
  for (const a0 of tokens) {
    const cs = input.condition_sequence_1_residue?.[a0];
    if (cs == null || cs === MASKED) continue;
    const target = (mapping ?? {})[input.token_id[a0]];
    if (generated.get(target) !== input.res_name[a0]) mismatches += 1;
  }
  return { n_unmapped_tokens: tokens.length - mapped.size, n_sequence_mismatches: mismatches };
}

const CLASH_CATEGORIES = ['intra_residue_polymer', 'intra_residue_ligand', 'inter_residue_polymer_polymer',
  'inter_residue_polymer_ligand', 'inter_residue_ligand_ligand'];

/** `ClashMetric` at 1.5 A: non-bonded pairs, both-coordinate-conditioned pairs excluded. */
function clashMetrics(arr, threshold = 1.5) {
  const { n, coord } = arr;
  const bonded = new Set(arr.bonds.map(([i, j]) => i * n + j));
  const cell = threshold;
  const grid = new Map();
  const cellOf = (i) => [0, 1, 2].map((k) => Math.floor(coord[i * 3 + k] / cell));
  for (let i = 0; i < n; i += 1) {
    const key = cellOf(i).join(',');
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(i);
  }
  const pairs = [];
  for (let i = 0; i < n; i += 1) {
    const [x, y, z] = cellOf(i);
    for (let dx = -1; dx <= 1; dx += 1) for (let dy = -1; dy <= 1; dy += 1) for (let dz = -1; dz <= 1; dz += 1) {
      for (const j of grid.get(`${x + dx},${y + dy},${z + dz}`) ?? []) {
        if (j <= i || bonded.has(i * n + j) || (arr.crd[i] && arr.crd[j])) continue;
        const d = dist32(coord, i, j);
        if (d <= threshold) pairs.push([i, j, d]);
      }
    }
  }
  const sameRes = (i, j) => arr.chainIid[i] === arr.chainIid[j] && arr.resId[i] === arr.resId[j];
  const category = (i, j) => {
    const pi = arr.isPolymer[i];
    const pj = arr.isPolymer[j];
    if (sameRes(i, j)) return pi && pj ? 0 : !pi && !pj ? 1 : -1;
    return pi && pj ? 2 : pi !== pj ? 3 : 4;
  };
  const fraction = (keep) => {
    const best = new Float64Array(n).fill(Infinity);
    for (const [i, j, d] of pairs) {
      if (!keep(i, j)) continue;
      if (d < best[i]) best[i] = d;
      if (d < best[j]) best[j] = d;
    }
    let count = 0;
    for (let i = 0; i < n; i += 1) if (best[i] < threshold) count += 1;
    return count / n;
  };
  const key = `${threshold.toFixed(1).replace('.', 'p')}A`;
  const out = { [`frac_atoms_clashing_${key}`]: fraction(() => true) };
  CLASH_CATEGORIES.forEach((name, c) => {
    out[`frac_atoms_clashing_${name}_${key}`] = fraction((i, j) => category(i, j) === c);
  });
  return out;
}

const CHNO = new Set(['C', 'H', 'N', 'O']);
const CHNOPS = new Set(['C', 'H', 'N', 'O', 'P', 'S']);

/** `BondLengthMetric(verbose=True)`: bonds longer than atomworks' element thresholds. */
function bondLengthMetrics(arr) {
  if (arr.bonds.length === 0) return {};
  const long = arr.bonds.map(([i, j]) => {
    const e1 = arr.element[i];
    const e2 = arr.element[j];
    const limit = CHNO.has(e1) && CHNO.has(e2) ? 1.7 : CHNOPS.has(e1) && CHNOPS.has(e2) ? 2.4 : 3.6;
    return dist32(arr.coord, i, j) > limit;
  });
  const out = { 'bond_length.frac_unreasonable': long.filter(Boolean).length / long.length };
  const cats = {
    within_residue: (i, j) => arr.chainIid[i] === arr.chainIid[j] && arr.resId[i] === arr.resId[j],
    'between_residue.within_chain': (i, j) => arr.chainIid[i] === arr.chainIid[j] && arr.resId[i] !== arr.resId[j],
    'between_chain.polymer_polymer': (i, j) => arr.chainIid[i] !== arr.chainIid[j] && arr.isPolymer[i] && arr.isPolymer[j],
    'between_chain.nonpolymer_to_polymer': (i, j) => arr.chainIid[i] !== arr.chainIid[j] && arr.isPolymer[i] !== arr.isPolymer[j],
  };
  for (const [name, test] of Object.entries(cats)) {
    let total = 0;
    let bad = 0;
    arr.bonds.forEach(([i, j], k) => { if (test(i, j)) { total += 1; if (long[k]) bad += 1; } });
    out[`bond_length.frac_unreasonable.${name}`] = total ? bad / total : NaN;
  }
  return out;
}

// ---- P-SEA (biotite's `annotate_sse`), in float32 as biotite computes it -------------------
// numpy's `deg2rad`: x * (pi / 180), which is not (x * pi) / 180 in the last place.
const deg = (d) => d * (Math.PI / 180);
const R_HELIX = [deg(89 - 12), deg(89 + 12)];
const A_HELIX = [deg(50 - 20), deg(50 + 20)];
const D3_HELIX = [5.3 - 0.5, 5.3 + 0.5];
const D4_HELIX = [6.4 - 0.6, 6.4 + 0.6];
const R_STRAND = [deg(124 - 14), deg(124 + 14)];
const A_STRAND = [deg(-180), deg(-125), deg(145), deg(180)];
const D2_STRAND = [6.7 - 0.6, 6.7 + 0.6];
const D3_STRAND = [9.9 - 0.9, 9.9 + 0.9];
const D4_STRAND = [12.4 - 1.1, 12.4 + 1.1];
const inRange = (v, [lo, hi]) => v >= lo && v <= hi;

function sub3(p, q) { return [f32(q[0] - p[0]), f32(q[1] - p[1]), f32(q[2] - p[2])]; }
function dot3(a, b) { return f32(f32(f32(a[0] * b[0]) + f32(a[1] * b[1])) + f32(a[2] * b[2])); }
function unit3(v) { const l = f32(Math.sqrt(dot3(v, v))); return [f32(v[0] / l), f32(v[1] / l), f32(v[2] / l)]; }
function cross3(a, b) {
  return [f32(f32(a[1] * b[2]) - f32(a[2] * b[1])), f32(f32(a[2] * b[0]) - f32(a[0] * b[2])),
    f32(f32(a[0] * b[1]) - f32(a[1] * b[0]))];
}
const distance = (p, q) => { const d = sub3(p, q); return f32(Math.sqrt(dot3(d, d))); };
const angle = (a, b, c) => f32(Math.acos(dot3(unit3(sub3(b, a)), unit3(sub3(b, c)))));
function dihedral(a, b, c, d) {
  const v1 = unit3(sub3(a, b));
  const v2 = unit3(sub3(b, c));
  const v3 = unit3(sub3(c, d));
  const n1 = cross3(v1, v2);
  const n2 = cross3(v2, v3);
  return f32(Math.atan2(dot3(cross3(n1, n2), v2), dot3(n1, n2)));
}

function maskConsecutive(mask, number) {
  const out = new Array(mask.length).fill(false);
  for (let s = 0; s + number <= mask.length; s += 1) {
    let all = true;
    for (let k = 0; k < number; k += 1) all = all && mask[s + k];
    if (all) for (let k = 0; k < number; k += 1) out[s + k] = true;
  }
  return out;
}
function extendRegion(base, extension) {
  const n = base.length;
  const change = base.map((b, i) => b !== (i === 0 ? false : base[i - 1]));
  const left = new Array(n).fill(false);
  for (let i = 0; i + 1 < n; i += 1) left[i] = change[i + 1] && base[i + 1];
  return base.map((b, i) => b || ((left[i] || (change[i] && !b)) && extension[i]));
}
function maskRegionsWithContacts(coord, candidate, minContacts, minDistance, maxDistance) {
  const ok = coord.filter((c) => c.every(Number.isFinite));
  if (ok.length === 0) return new Array(candidate.length).fill(false);
  const contacts = new Array(coord.length).fill(0);
  candidate.forEach((c, i) => {
    if (!c) return;
    for (const q of ok) {
      const d = distance(coord[i], q);
      if (d <= maxDistance && d > minDistance) contacts[i] += 1;
    }
  });
  const out = new Array(candidate.length).fill(false);
  const edges = [];
  candidate.forEach((c, i) => { if (c !== (i === 0 ? false : candidate[i - 1])) edges.push(i); });
  edges.push(coord.length);
  for (let r = 0; r + 1 < edges.length; r += 1) {
    let total = 0;
    for (let i = edges[r]; i < edges[r + 1]; i += 1) total += contacts[i];
    if (total >= minContacts) for (let i = edges[r]; i < edges[r + 1]; i += 1) out[i] = true;
  }
  return out;
}

/** biotite `annotate_sse`: one code per residue ('a', 'b', 'c', '' for a residue with no CA). */
export function annotateSse(arr) {
  const st = biotiteResidueStarts(arr);
  const nan = [NaN, NaN, NaN];
  const ca = st.map(() => nan);
  for (let i = 0; i < arr.n; i += 1) {
    const isAa = STANDARD_LINK[arr.resName[i]]
      ? PEPTIDE_LINKS.has(STANDARD_LINK[arr.resName[i]])
      : AMINO_ACID_TYPES.has(arr.chemCompType[i]);
    if (!isAa || arr.atomName[i] !== 'CA') continue;
    let r = st.length - 1;
    while (st[r] > i) r -= 1;
    ca[r] = [arr.coord[i * 3], arr.coord[i * 3 + 1], arr.coord[i * 3 + 2]];
  }
  if (ca.length <= 5) return ca.map((c) => (c.some(Number.isNaN) ? '' : 'c'));
  // A gap or a decrease in res_id between consecutive atoms inserts a virtual residue.
  const breaks = [];
  for (let i = 1; i < arr.n; i += 1) {
    const d = arr.resId[i] - arr.resId[i - 1];
    if (d !== 0 && d !== 1) {
      let r = st.length - 1;
      while (st[r] > i) r -= 1;
      breaks.push(r);
    }
  }
  const coords = [];
  const real = [];
  let b = 0;
  for (let r = 0; r < ca.length; r += 1) {
    while (b < breaks.length && breaks[b] === r) { coords.push(nan); real.push(false); b += 1; }
    coords.push(ca[r]);
    real.push(true);
  }
  const L = coords.length;
  const d2 = new Array(L).fill(NaN);
  const d3 = new Array(L).fill(NaN);
  const d4 = new Array(L).fill(NaN);
  const ri = new Array(L).fill(NaN);
  const ai = new Array(L).fill(NaN);
  for (let i = 1; i < L - 1; i += 1) d2[i] = distance(coords[i - 1], coords[i + 1]);
  for (let i = 1; i < L - 2; i += 1) d3[i] = distance(coords[i - 1], coords[i + 2]);
  for (let i = 1; i < L - 3; i += 1) d4[i] = distance(coords[i - 1], coords[i + 3]);
  for (let i = 1; i < L - 1; i += 1) ri[i] = angle(coords[i - 1], coords[i], coords[i + 1]);
  for (let i = 1; i < L - 2; i += 1) ai[i] = dihedral(coords[i - 1], coords[i], coords[i + 1], coords[i + 2]);
  const relaxedHelix = d3.map((_, i) => inRange(d3[i], D3_HELIX) || inRange(ri[i], R_HELIX));
  const strictHelix = d3.map((_, i) => (inRange(d3[i], D3_HELIX) && inRange(d4[i], D4_HELIX))
    || (inRange(ri[i], R_HELIX) && inRange(ai[i], A_HELIX)));
  const relaxedStrand = d3.map((_, i) => inRange(d3[i], D3_STRAND));
  const strictStrand = d3.map((_, i) => (inRange(d2[i], D2_STRAND) && inRange(d3[i], D3_STRAND)
    && inRange(d4[i], D4_STRAND)) || (inRange(ri[i], R_STRAND)
    && (inRange(ai[i], [A_STRAND[0], A_STRAND[1]]) || inRange(ai[i], [A_STRAND[2], A_STRAND[3]]))));
  const helix = extendRegion(maskConsecutive(strictHelix, 5), relaxedHelix);
  const short = maskRegionsWithContacts(coords, maskConsecutive(strictStrand, 3), 5, 4.2, 5.2);
  const strandBase = maskConsecutive(strictStrand, 4).map((m, i) => m || short[i]);
  const strand = extendRegion(strandBase, relaxedStrand);
  const sse = coords.map((c, i) => (c.some(Number.isNaN) ? '' : strand[i] ? 'b' : helix[i] ? 'a' : 'c'));
  return sse.filter((_, i) => real[i]);
}

/** atomworks' `annotate_secondary_structure`: per atom, the residue's SSEnum and validity. */
function secondaryStructure(arr) {
  const st = residueStarts(arr);
  let sse = new Array(st.length).fill(SSE.NONE);
  let valid = new Array(st.length).fill(false);
  const codes = annotateSse(arr);
  if (codes.length === st.length) {
    sse = codes.map((c) => (c === 'a' ? SSE.HELIX : c === 'b' ? SSE.SHEET : c === 'c' ? SSE.LOOP : SSE.NONE));
    valid = codes.map((c) => c !== '');
  }
  return { sse: spread(arr.n, st, sse), valid: spread(arr.n, st, valid) };
}

/** `get_chain_of_interest`: the least sequence-conditioned protein chain, then the least placed. */
export function chainOfInterest(arr) {
  const chains = [...new Set(arr.chainIid)].sort();
  if (chains.length < 2) return chains[0] ?? null;
  const atomsOf = (c) => arr.chainIid.flatMap((x, i) => (x === c ? [i] : []));
  const protein = chains.filter((c) => atomsOf(c).every((i) => arr.isProtein[i]));
  if (protein.length === 1) return protein[0];
  if (protein.length === 0) return chains[0];
  const mean = (c, test) => { const at = atomsOf(c); return at.filter(test).length / at.length; };
  const seq = protein.map((c) => mean(c, (i) => arr.cSeq[i] !== MASKED));
  const least = protein.filter((_, k) => seq[k] === Math.min(...seq));
  if (least.length === 1) return least[0];
  const crd = least.map((c) => mean(c, (i) => arr.crd[i]) + mean(c, (i) => arr.internal[i]));
  return least.filter((_, k) => crd[k] === Math.min(...crd))[0];
}

function sseFractions(values, suffix) {
  if (values.length === 0) {
    return Object.fromEntries(['helix_fraction', 'sheet_fraction', 'loop_fraction', 'num_ss_elements']
      .map((k) => [`${k}${suffix}`, NaN]));
  }
  const frac = (v) => values.filter((x) => x === v).length / values.length;
  let elements = 0;
  values.forEach((v, i) => { if (i === 0 || v !== values[i - 1]) elements += 1; });
  return {
    [`helix_fraction${suffix}`]: frac(SSE.HELIX),
    [`sheet_fraction${suffix}`]: frac(SSE.SHEET),
    [`loop_fraction${suffix}`]: frac(SSE.LOOP),
    [`num_ss_elements${suffix}`]: elements,
  };
}

/** `SecondaryStructureDistribution`, over valid tokens and over the chain of interest's. */
function sseDistribution(arr, ss) {
  const ts = tokenStarts(arr).filter((s) => ss.valid[s]);
  const coi = chainOfInterest(arr);
  return {
    ...sseFractions(ts.map((s) => ss.sse[s]), ''),
    ...sseFractions(ts.filter((s) => arr.chainIid[s] === coi).map((s) => ss.sse[s]), '_coi'),
  };
}

/** `SecondaryStructureAccuracy`, per atom, against the job's own C_SEC. */
function sseAccuracy(arr, ss) {
  if (!arr.hasSec) return {};
  const conditioned = arr.cSec.map((v) => v !== SSE.NONE);
  if (!conditioned.some(Boolean)) return {};
  const at = conditioned.flatMap((c, i) => (c && ss.valid[i] ? [i] : []));
  if (at.length === 0) {
    return { overall_accuracy: NaN, helix_recall: NaN, sheet_recall: NaN, loop_recall: NaN,
      n_conditioned_atoms: conditioned.filter(Boolean).length };
  }
  const recall = (cls) => {
    const of = at.filter((i) => arr.cSec[i] === cls);
    return of.length ? of.filter((i) => ss.sse[i] === cls).length / of.length : NaN;
  };
  return {
    overall_accuracy: at.filter((i) => arr.cSec[i] === ss.sse[i]).length / at.length,
    helix_recall: recall(SSE.HELIX),
    sheet_recall: recall(SSE.SHEET),
    loop_recall: recall(SSE.LOOP),
    n_conditioned_atoms: at.length,
  };
}

/** `HotspotObedience(max_hotspot_distance=4.5)` */
function hotspotObedience(arr, limit = 4.5) {
  const hot = arr.hot.flatMap((h, i) => (h ? [i] : []));
  if (hot.length === 0) return { hotspot_obedience: NaN };
  let obeyed = 0;
  for (const chain of [...new Set(hot.map((i) => arr.chainIid[i]))].sort()) {
    const binder = [];
    for (let j = 0; j < arr.n; j += 1) if (arr.chainIid[j] !== chain && !arr.crd[j]) binder.push(j);
    for (const i of hot) {
      if (arr.chainIid[i] !== chain) continue;
      if (binder.some((j) => dist32(arr.coord, i, j) < limit)) obeyed += 1;
    }
  }
  return { hotspot_obedience: obeyed / hot.length };
}

/** `SequenceSimilarityToInput`: protein residue identity against the indexed input. */
function sequenceSimilarityToInput(arr, input) {
  const generated = residueStarts(arr).filter((s) => arr.isProtein[s]).map((s) => arr.resName[s]);
  const idx = input.condition_index_1_residue;
  const view = { n: 0, chainId: [], resId: [], insCode: [], transformationId: [], resName: [] };
  for (let i = 0; i < input.res_name.length; i += 1) {
    const name = input.atom_name[i];
    if ((idx && !idx[i]) || !input.is_protein?.[i]) continue;
    if (input.is_polymer?.[i] && name === 'OXT') continue;
    view.chainId.push(input.chain_id[i]);
    view.resId.push(Number(input.res_id[i]));
    view.insCode.push(input.ins_code?.[i] ?? '');
    view.transformationId.push(String(input.transformation_id?.[i] ?? ''));
    view.resName.push(input.res_name[i]);
    view.n += 1;
  }
  const given = residueStarts(view).map((s) => view.resName[s]);
  if (generated.length === 0 || given.length === 0 || generated.length !== given.length) {
    return { sequence_identity_to_input: NaN };
  }
  return { sequence_identity_to_input: generated.filter((r, k) => r === given[k]).length / generated.length };
}

const PROTEIN_CHAINS = new Set([CHAIN_TYPE.POLYPEPTIDE_D, CHAIN_TYPE.POLYPEPTIDE_L, CHAIN_TYPE.CYCLIC_PSEUDO_PEPTIDE]);
const NUCLEIC_CHAINS = new Set([CHAIN_TYPE.DNA, CHAIN_TYPE.RNA, CHAIN_TYPE.DNA_RNA_HYBRID]);
const isInternalGroup = (g) => g != null && g !== -1 && g !== 0;
const finiteAt = (c, i) => Number.isFinite(c[i * 3]) && Number.isFinite(c[i * 3 + 1]) && Number.isFinite(c[i * 3 + 2]);

/** biotite's `superimpose(fixed, mobile)` over `idx`, as a function of a whole coordinate set. */
function fitOnto(fixed, mobile, fixedIdx, mobileIdx) {
  const f = new Float64Array(fixedIdx.length * 3);
  const m = new Float64Array(mobileIdx.length * 3);
  fixedIdx.forEach((i, k) => { for (let c = 0; c < 3; c += 1) f[k * 3 + c] = fixed[i * 3 + c]; });
  mobileIdx.forEach((i, k) => { for (let c = 0; c < 3; c += 1) m[k * 3 + c] = mobile[i * 3 + c]; });
  const { rotation: R, fixedCenter: fc, mobileCenter: mc } = superimposeCoords3D(f, m);
  return (x, y, z) => {
    const a = x - mc[0];
    const b = y - mc[1];
    const c = z - mc[2];
    return [a * R[0] + b * R[1] + c * R[2] + fc[0], a * R[3] + b * R[4] + c * R[5] + fc[1],
      a * R[6] + b * R[7] + c * R[8] + fc[2]];
  };
}

/** Fit `reference` (per-atom triples) onto the array's coordinates over `atoms`, into `cwi`. */
function stageGroup(arr, cwi, atoms, reference) {
  const ref = new Float64Array(arr.n * 3);
  for (const i of atoms) for (let c = 0; c < 3; c += 1) ref[i * 3 + c] = reference(i)[c];
  const fit = fitOnto(arr.coord, ref, atoms, atoms);
  for (const i of atoms) {
    const [x, y, z] = fit(ref[i * 3], ref[i * 3 + 1], ref[i * 3 + 2]);
    cwi[i * 3] = x; cwi[i * 3 + 1] = y; cwi[i * 3 + 2] = z;
  }
}

/**
 * `SA_CWI` (`coords-with-insertion`), the motif RMSD's reference, staged as upstream stages it:
 * `process_single_sample_to_atom_array` (the coordinates, overlaid with C_CRD, each internal
 * group's C_INTCRD fitted onto the design) for an indexed motif, then `annotate_unindexed_insertion`
 * for an unindexed one (a grafted atom's motif coordinates, every internal group re-fitted whole
 * over its coordinate-conditioned atoms, and the grafted ones invalidated if not every motif atom
 * was placed). `null` where upstream stages none - and its metric raises.
 */
function coordsWithInsertion(arr, input, structure) {
  const n = arr.n;
  const idx = input.condition_index_1_residue;
  const unindexed = [];
  if (idx) for (let i = 0; i < idx.length; i += 1) if (!idx[i]) unindexed.push(i);
  const own = (i) => arr.condSource[i] < 0;
  const ownCrd = (i) => own(i) && arr.crd[i];
  const ownInternal = (i) => own(i) && arr.internal[i];
  const indexed = [...Array(n).keys()].some((i) => ownCrd(i) || ownInternal(i));
  if (!indexed && unindexed.length === 0) return null;
  const cwi = new Float64Array(n * 3).fill(NaN);
  if (indexed) {
    for (let i = 0; i < n * 3; i += 1) cwi[i] = arr.coord[i];
    for (let i = 0; i < n; i += 1) if (ownCrd(i)) for (let c = 0; c < 3; c += 1) cwi[i * 3 + c] = arr.crdValue[i][c];
    const groups = new Map();
    for (let i = 0; i < n; i += 1) {
      if (!ownInternal(i)) continue;
      if (!groups.has(arr.coordGroup[i])) groups.set(arr.coordGroup[i], []);
      groups.get(arr.coordGroup[i]).push(i);
    }
    for (const atoms of [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v)) {
      if (atoms.some((i) => !arr.intCrd[i])) throw new Error(`NaN values detected in input or generated coords for coord group ${arr.coordGroup[atoms[0]]}.`);
      stageGroup(arr, cwi, atoms, (i) => arr.intCrd[i]);
    }
  }
  if (unindexed.length === 0) return cwi;
  // A proposed mapping that could not be applied drops the unindexed region, and the staging
  // above is all there is; an applied (or empty) one is staged in full.
  const proposed = Object.keys(structure.predictedInsertionMapping ?? {}).length > 0;
  if (proposed && !arr.uii.some(Boolean)) return cwi;
  const motif = (i) => arr.condSource[i] >= 0 && (arr.crdValue[i] !== null
    || isInternalGroup(input.condition_CoordinateGroup_1_atom?.[arr.condSource[i]]));
  let placed = 0;
  for (let i = 0; i < n; i += 1) {
    if (!motif(i)) continue;
    placed += 1;
    if (arr.crdValue[i]) for (let c = 0; c < 3; c += 1) cwi[i * 3 + c] = arr.crdValue[i][c];
  }
  const groups = new Map();
  for (let i = 0; i < n; i += 1) {
    if (!arr.crdsrc[i] || !isInternalGroup(arr.coordGroup[i])) continue;
    if (!groups.has(arr.coordGroup[i])) groups.set(arr.coordGroup[i], []);
    groups.get(arr.coordGroup[i]).push(i);
  }
  for (const atoms of [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v)) {
    if (atoms.some((i) => !arr.intCrd[i])) continue;  // upstream skips such a group, with a warning
    stageGroup(arr, cwi, atoms, (i) => arr.intCrd[i]);
  }
  const crd = input.condition_coordinate_1_atom;
  const expected = unindexed.filter((i) => (Array.isArray(crd?.[i]) && crd[i].every(Number.isFinite))
    || isInternalGroup(input.condition_CoordinateGroup_1_atom?.[i])).length;
  if (expected > 0 && placed !== expected) {
    for (let i = 0; i < n; i += 1) if (motif(i)) for (let c = 0; c < 3; c += 1) cwi[i * 3 + c] = NaN;
  }
  return cwi;
}

/** A sub-array's atoms, bonds re-indexed: what a symmetry search over a slice of `arr` reads. */
function sliceView(arr, atoms) {
  const local = new Map(atoms.map((g, k) => [g, k]));
  const pick = (f) => atoms.map((g) => arr[f][g]);
  const bonds = [];
  for (const [i, j, t] of arr.bonds) if (local.has(i) && local.has(j)) bonds.push([local.get(i), local.get(j), t]);
  return { n: atoms.length, element: pick('element'), charge: pick('charge'), isPolymer: pick('isPolymer'),
    chainId: pick('chainId'), chainIid: pick('chainIid'), resId: pick('resId'), insCode: pick('insCode'),
    resName: pick('resName'), atomName: pick('atomName'), pnUnit: pick('pnUnit'), molIid: pick('molIid'),
    molEntity: pick('molEntity'), bonds };
}

/**
 * `structure_comparison_metric`: fit the prediction onto the ground truth over the align masks
 * (each side's own, finite coordinates only) and take the RMSD over the subset masks, pairing
 * the two sides' selected atoms in order. `null` where upstream returns nothing: the two sides
 * select different counts, or nothing to fit.
 */
function structureComparison(gt, pred, gtAlign, predAlign, gtSubset, predSubset) {
  const pick = (c, mask) => mask.flatMap((m, i) => (m && finiteAt(c, i) ? [i] : []));
  const ga = pick(gt, gtAlign);
  const pa = pick(pred, predAlign);
  if (ga.length !== pa.length || ga.length === 0) return null;
  const fit = fitOnto(gt, pred, ga, pa);
  const moved = new Float64Array(pred.length);
  for (let i = 0; i < pred.length / 3; i += 1) {
    const [x, y, z] = fit(pred[i * 3], pred[i * 3 + 1], pred[i * 3 + 2]);
    moved[i * 3] = x; moved[i * 3 + 1] = y; moved[i * 3 + 2] = z;
  }
  const gs = pick(gt, gtSubset);
  const ps = pick(moved, predSubset);
  if (gs.length !== ps.length) return null;
  if (gs.length === 0) return { rmsd: NaN, n: 0 };
  let sq = 0;
  gs.forEach((g, k) => {
    const p = ps[k];
    for (let c = 0; c < 3; c += 1) {
      const d = gt[g * 3 + c] - moved[p * 3 + c];
      sq += d * d;
    }
  });
  return { rmsd: Math.sqrt(sq / gs.length), n: gs.length };
}

/** `StructureComparisonMetric._atom_classes`: -1 outside the subset, when anything is. */
const atomClasses = (subset) => (subset.every(Boolean) ? null : subset.map((m) => (m ? 0 : -1)));

/**
 * `design_motif_allatom_rmsd` and its unindexed twin: per `CoordinateGroup`, the motif atoms'
 * ground truth (`SA_CWI`) resolved for identical subunits and equivalent atoms in the group's own
 * frame, fitted and scored; the groups' RMSDs pooled atom-weighted.
 */
function motifRmsd(arr, cwi, selected) {
  const values = [...new Set(arr.coordGroup)].sort((a, b) => a - b);
  let total = 0;
  let weighted = 0;
  for (const value of values) {
    const atoms = [];
    for (let i = 0; i < arr.n; i += 1) if (arr.coordGroup[i] === value) atoms.push(i);
    const view = sliceView(arr, atoms);
    const pred = new Float64Array(atoms.length * 3);
    let gt = new Float64Array(atoms.length * 3);
    atoms.forEach((g, k) => {
      for (let c = 0; c < 3; c += 1) { pred[k * 3 + c] = arr.coord[g * 3 + c]; gt[k * 3 + c] = cwi[g * 3 + c]; }
    });
    const mask = atoms.map((g) => selected(g));
    gt = permuteGtSubunits(pred, gt, view, { frameMask: mask, scoreMask: mask, align: true });
    gt = permuteGtAtoms(pred, gt, view, { atomClasses: atomClasses(mask), frameMask: mask });
    const r = structureComparison(gt, pred, mask, mask, mask, mask);
    if (!r || r.n === 0) continue;
    total += r.n;
    weighted += r.rmsd * r.rmsd * r.n;
  }
  return total > 0 ? { rmsd: Math.sqrt(weighted / total) } : {};
}

/**
 * The array upstream hands a metric as its ground truth (`input_atom_array`): the input's
 * indexed atoms, its multi-residue polymers' terminal OXT/OP3 dropped.
 */
function inputView(input) {
  const total = input.atom_name.length;
  const idx = input.condition_index_1_residue;
  const kept = [];
  for (let i = 0; i < total; i += 1) if (!idx || idx[i]) kept.push(i);
  const chainOf = (i) => String(input.chain_iid?.[i] ?? input.chain_id[i]);
  const residues = new Map();
  kept.forEach((i, k) => {
    const prev = kept[k - 1];
    if (k === 0 || input.chain_id[i] !== input.chain_id[prev] || input.res_id[i] !== input.res_id[prev]
      || (input.ins_code?.[i] ?? '') !== (input.ins_code?.[prev] ?? '')) {
      residues.set(chainOf(i), (residues.get(chainOf(i)) ?? 0) + 1);
    }
  });
  const atoms = kept.filter((i) => {
    const terminal = (input.atom_name[i] === 'OXT' && PROTEIN_CHAINS.has(input.chain_type?.[i]))
      || (input.atom_name[i] === 'OP3' && NUCLEIC_CHAINS.has(input.chain_type?.[i]));
    return !(terminal && residues.get(chainOf(i)) > 1 && !input.atomize?.[i]);
  });
  const local = new Map(atoms.map((g, k) => [g, k]));
  const bonds = [];
  for (const [i, j, t] of input.bonds ?? []) {
    if (local.has(i) && local.has(j)) bonds.push([local.get(i), local.get(j), t ?? BOND_TYPE.SINGLE]);
  }
  const pick = (f, d) => atoms.map((g) => input[f]?.[g] ?? d);
  const coord = new Float64Array(atoms.length * 3);
  atoms.forEach((g, k) => { for (let c = 0; c < 3; c += 1) coord[k * 3 + c] = Math.fround(Number(input.coord[g]?.[c] ?? NaN)); });
  return { n: atoms.length, coord, element: pick('element', '').map((e) => String(e).toUpperCase()),
    charge: pick('charge', 0).map(Number), isPolymer: pick('is_polymer', false).map(Boolean),
    isProtein: pick('is_protein', false).map(Boolean), chainId: pick('chain_id', ''),
    chainIid: atoms.map(chainOf), resId: pick('res_id', 0), insCode: pick('ins_code', ''),
    resName: pick('res_name', ''), atomName: pick('atom_name', ''), pnUnit: pick('pn_unit_iid', '').map(String),
    molIid: pick('molecule_iid', '').map(String), molEntity: pick('molecule_entity', '').map(String), bonds };
}

/** `ca_rmsd_to_input`: the design's protein CAs against the input's, after one fit. */
function caRmsdToInput(arr, input) {
  const gtView = inputView(input);
  const gtCa = gtView.atomName.map((nm, i) => gtView.isProtein[i] && nm === 'CA');
  const predCa = arr.atomName.map((nm, i) => arr.isProtein[i] && nm === 'CA');
  const pred = Float64Array.from(arr.coord);
  let gt = gtView.coord;
  if (gtView.n === arr.n) {
    gt = permuteGtSubunits(pred, gt, gtView, { frameMask: gtCa, scoreMask: gtCa, align: true });
    gt = permuteGtAtoms(pred, gt, gtView, { atomClasses: atomClasses(gtCa), frameMask: gtCa });
  }
  const r = structureComparison(gt, pred, gtCa, predCa, gtCa, predCa);
  return r ? { rmsd: r.rmsd } : {};
}

/**
 * Every metric this job's tags admit, keyed `<metric>.<value>` as upstream keys them, and the
 * pre-oracle filters over them.
 * @param {object} structure `processSampleToAtoms`' result
 * @param {object} input the batch's input atom table
 * @param {{ metricsTags?: string[] }} [options]
 */
export function designMetrics(structure, input, { metricsTags = [] } = {}) {
  const arr = designArray(structure, input);
  const want = (name) => appliesToTags(METRIC_GATES[name], metricsTags);
  let ss = null;
  const sse = () => (ss ??= secondaryStructure(arr));
  let cwi;
  const motifReference = () => {
    cwi ??= coordsWithInsertion(arr, input, structure);
    if (cwi === null) {
      throw new Error("a motif RMSD needs the design's coordinate-conditioned atoms, and this job has none: "
        + "upstream raises here too ('coords-with-insertion' is never staged), so its "
        + "'3d_coordinate_conditioning' / 'internal_coordinate_conditioning' / 'unindexed' tag does not apply");
    }
    return cwi;
  };
  const compute = {
    amino_acid_composition: () => aminoAcidComposition(arr),
    bond_length_metrics: () => bondLengthMetrics(arr),
    clash_metrics: () => clashMetrics(arr),
    hotspot_obedience: () => hotspotObedience(arr),
    sequence_adherence: () => sequenceAdherence(arr),
    sequence_probability_adherence: () => sequenceProbabilityAdherence(arr),
    sse_distribution: () => sseDistribution(arr, sse()),
    sse_accuracy: () => sseAccuracy(arr, sse()),
    unindexed_insertion_mapping: () => unindexedInsertionMapping(arr, input, structure.predictedInsertionMapping),
    sequence_similarity_to_input: () => sequenceSimilarityToInput(arr, input),
    design_motif_allatom_rmsd: () => motifRmsd(arr, motifReference(), (i) => arr.crdsrc[i]),
    design_unindexed_motif_allatom_rmsd: () => motifRmsd(arr, motifReference(), (i) => arr.crdsrc[i] && arr.uii[i]),
    ca_rmsd_to_input: () => caRmsdToInput(arr, input),
  };
  const metrics = {};
  for (const [name, fn] of Object.entries(compute)) {
    if (!want(name)) continue;
    for (const [k, v] of Object.entries(fn())) metrics[`${name}.${k}`] = v;
  }
  return { metrics, preOracle: preOracleVerdict(metrics, metricsTags) };
}

/**
 * `apply_pre_oracle_filters`: a design passes when no applicable filter fails. A filter whose
 * metric is ABSENT fails but is indeterminate (`allMetricsPresent: false`), which upstream keeps
 * on disk rather than deleting - and NaN is present, and fails nothing (`NaN <= 0` is false, but
 * upstream compares the same way, so it fails there too).
 */
export function preOracleVerdict(metrics, metricsTags = []) {
  const filters = {};
  let pass = true;
  for (const [name, { gate, thresholds }] of Object.entries(PRE_ORACLE_FILTERS)) {
    if (!appliesToTags(gate, metricsTags)) continue;
    let ok = true;
    let present = true;
    for (const [key, max] of Object.entries(thresholds)) {
      const v = metrics[key];
      if (v === undefined) { present = false; ok = false; continue; }
      if (!(v <= max)) ok = false;
    }
    filters[name] = { pass: ok, allMetricsPresent: present };
    pass &&= ok;
  }
  return { pass, filters };
}
