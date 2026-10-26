/**
 * Condition-spec (YAML/JSON) parser, selection query evaluator (`sel(...)` + boolean queries),
 * `applyContig`, `applyConditions`, and `buildAtomArrayFromConditionSpec`.
 *
 * Ports `rfproteina.inference_utils.contig_spec`, `rfproteina.inference_utils.condition_spec`,
 * `rfproteina.utils.mask_spec`, and `atomworks.io.utils.query`.
 */

import { normalizeCatLabel } from '../conditions.js';
import { prepareFreshBlock } from './fresh.js';
import { evalQuery, pathMask } from './query.js';
import {
  BOND_TYPE,
  CHAIN_TYPE,
  SCALAR_1BODY_ANNOTATIONS,
  NONSCALAR_1BODY_ANNOTATIONS,
  createEmptyAtomArray,
  parseCifToAtomArray,
  readStructureBlock,
  computeIdAndEntityAnnotations,
  getResidueSegments,
  getPolymerizationAtoms,
} from './cif.js';

export const EXPANDABLE_SEGMENT_RES_ID_SENTINEL = -(2 ** 31);

const SUFFIX_TO_CHAIN_TYPE = {
  P: CHAIN_TYPE.POLYPEPTIDE_L,
  R: CHAIN_TYPE.RNA,
  D: CHAIN_TYPE.DNA,
};

// Enum definitions for condition resolution
export const CONDITION_ENUMS = {
  RasaEnum: { UNSPECIFIED: 0, BURIED: 1, INTERMEDIATE: 2, EXPOSED: 3 },
  HBondEnum: { NONE: 0, DONOR: 1, ACCEPTOR: 2, BOTH: 3 },
  PiCationEnum: { NONE: 0, AROMATIC: 1, CATION: 2, BOTH: 3 },
  SaltBridgeEnum: { NONE: 0, POSITIVE: 1, NEGATIVE: 2 },
  MetalCoordEnum: { NONE: 0, METAL: 1, COORDINATING: 2 },
  // 🔴 UPSTREAM'S NAME IS SHEET (atomworks `SSEnum`: NONE, HELIX, SHEET, LOOP, NON_PROTEIN), and
  // this table said STRAND alone, so `C_SEC: {SHEET: ...}` - the spelling every upstream spec uses -
  // was refused as an invalid enum value. STRAND stays as an alias of the same 2. NON_PROTEIN (4) is
  // not here on purpose: upstream accepts the name and then refuses it at the model, whose
  // embedding is four wide ("expected 0-3"), so taking it would only move the refusal later.
  SSEnum: { NONE: 0, HELIX: 1, SHEET: 2, STRAND: 2, LOOP: 3 },
  BasePairEnum: { UNSPECIFIED: 0, PAIR: 1, LOOP: 2 },
};

// `CoordinateGroup.MAX_GROUP_ID`: model batches store group ids as int16.
const CRDGRP_MAX_GROUP_ID = 32767;

// Registry of 1-body conditions and standard annotations
export const CONDITION_REGISTRY = {
  C_CRD: { fullName: 'condition_coordinate_1_atom', dtype: 'coord', default: [NaN, NaN, NaN] },
  coordinate: { fullName: 'condition_coordinate_1_atom', dtype: 'coord', default: [NaN, NaN, NaN] },
  condition_coordinate_1_atom: { fullName: 'condition_coordinate_1_atom', dtype: 'coord', default: [NaN, NaN, NaN] },

  C_CRDGRP: { fullName: 'condition_CoordinateGroup_1_atom', dtype: 'crdgrp', default: -1 },
  CoordinateGroup: { fullName: 'condition_CoordinateGroup_1_atom', dtype: 'crdgrp', default: -1 },
  condition_CoordinateGroup_1_atom: { fullName: 'condition_CoordinateGroup_1_atom', dtype: 'crdgrp', default: -1 },

  C_INTCRD: { fullName: 'condition_InternalCoordinate_1_atom', dtype: 'intcrd', default: [NaN, NaN, NaN] },
  InternalCoordinate: { fullName: 'condition_InternalCoordinate_1_atom', dtype: 'intcrd', default: [NaN, NaN, NaN] },
  condition_InternalCoordinate_1_atom: { fullName: 'condition_InternalCoordinate_1_atom', dtype: 'intcrd', default: [NaN, NaN, NaN] },

  C_CENTROID: { fullName: 'condition_ChainCentroid_1_atom', dtype: 'centroid', default: [NaN, NaN, NaN] },
  ChainCentroid: { fullName: 'condition_ChainCentroid_1_atom', dtype: 'centroid', default: [NaN, NaN, NaN] },
  condition_ChainCentroid_1_atom: { fullName: 'condition_ChainCentroid_1_atom', dtype: 'centroid', default: [NaN, NaN, NaN] },

  C_SEQ: { fullName: 'condition_sequence_1_residue', dtype: 'seq', default: '<M>' },
  sequence: { fullName: 'condition_sequence_1_residue', dtype: 'seq', default: '<M>' },
  condition_sequence_1_residue: { fullName: 'condition_sequence_1_residue', dtype: 'seq', default: '<M>' },

  C_IDX: { fullName: 'condition_index_1_residue', dtype: 'bool', default: true },
  index: { fullName: 'condition_index_1_residue', dtype: 'bool', default: true },
  condition_index_1_residue: { fullName: 'condition_index_1_residue', dtype: 'bool', default: true },

  C_IDXGRP: { fullName: 'condition_index-group_1_token', dtype: 'int', default: 0 },
  'index-group': { fullName: 'condition_index-group_1_token', dtype: 'int', default: 0 },
  'condition_index-group_1_token': { fullName: 'condition_index-group_1_token', dtype: 'int', default: 0 },

  C_HOT: { fullName: 'condition_hotspot_1_atom', dtype: 'bool', default: false },
  hotspot: { fullName: 'condition_hotspot_1_atom', dtype: 'bool', default: false },
  condition_hotspot_1_atom: { fullName: 'condition_hotspot_1_atom', dtype: 'bool', default: false },

  C_RAS: { fullName: 'condition_rasa-binned_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.RasaEnum, default: 0 },
  'rasa-binned': { fullName: 'condition_rasa-binned_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.RasaEnum, default: 0 },
  'condition_rasa-binned_1_atom': { fullName: 'condition_rasa-binned_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.RasaEnum, default: 0 },

  C_HB: { fullName: 'condition_hbond_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.HBondEnum, default: 0 },
  hbond: { fullName: 'condition_hbond_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.HBondEnum, default: 0 },
  condition_hbond_1_atom: { fullName: 'condition_hbond_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.HBondEnum, default: 0 },

  C_PISTACK: { fullName: 'condition_pistack_1_atom', dtype: 'bool', default: false },
  pistack: { fullName: 'condition_pistack_1_atom', dtype: 'bool', default: false },
  condition_pistack_1_atom: { fullName: 'condition_pistack_1_atom', dtype: 'bool', default: false },

  C_PICATION: { fullName: 'condition_pication_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.PiCationEnum, default: 0 },
  pication: { fullName: 'condition_pication_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.PiCationEnum, default: 0 },
  condition_pication_1_atom: { fullName: 'condition_pication_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.PiCationEnum, default: 0 },

  C_SALT: { fullName: 'condition_saltbridge_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.SaltBridgeEnum, default: 0 },
  saltbridge: { fullName: 'condition_saltbridge_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.SaltBridgeEnum, default: 0 },
  condition_saltbridge_1_atom: { fullName: 'condition_saltbridge_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.SaltBridgeEnum, default: 0 },

  C_HPHOB: { fullName: 'condition_hydrophobic_1_atom', dtype: 'bool', default: false },
  hydrophobic: { fullName: 'condition_hydrophobic_1_atom', dtype: 'bool', default: false },
  condition_hydrophobic_1_atom: { fullName: 'condition_hydrophobic_1_atom', dtype: 'bool', default: false },

  C_MCOORD: { fullName: 'condition_metal-coord_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.MetalCoordEnum, default: 0 },
  'metal-coord': { fullName: 'condition_metal-coord_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.MetalCoordEnum, default: 0 },
  'condition_metal-coord_1_atom': { fullName: 'condition_metal-coord_1_atom', dtype: 'enum', enum: CONDITION_ENUMS.MetalCoordEnum, default: 0 },

  C_SEC: { fullName: 'condition_secondary-structure_1_residue', dtype: 'enum', enum: CONDITION_ENUMS.SSEnum, default: 0 },
  'secondary-structure': { fullName: 'condition_secondary-structure_1_residue', dtype: 'enum', enum: CONDITION_ENUMS.SSEnum, default: 0 },
  'condition_secondary-structure_1_residue': { fullName: 'condition_secondary-structure_1_residue', dtype: 'enum', enum: CONDITION_ENUMS.SSEnum, default: 0 },

  C_CATH: { fullName: 'condition_cath-label_1_residue', dtype: 'cath', default: 'x.x.x.x' },
  'cath-label': { fullName: 'condition_cath-label_1_residue', dtype: 'cath', default: 'x.x.x.x' },
  'condition_cath-label_1_residue': { fullName: 'condition_cath-label_1_residue', dtype: 'cath', default: 'x.x.x.x' },

  C_NA_SS_STATE: { fullName: 'condition_na-ss-state_1_residue', dtype: 'enum', enum: CONDITION_ENUMS.BasePairEnum, default: 0 },
  'na-ss-state': { fullName: 'condition_na-ss-state_1_residue', dtype: 'enum', enum: CONDITION_ENUMS.BasePairEnum, default: 0 },
  'condition_na-ss-state_1_residue': { fullName: 'condition_na-ss-state_1_residue', dtype: 'enum', enum: CONDITION_ENUMS.BasePairEnum, default: 0 },

  // 🔴 NAMES UPSTREAM RESOLVES THAT THIS DID NOT: the chain-end flags, the legacy `mask_*`
  // aliases of three boolean conditions, the base pairs (TWO-body, which a condition spec cannot
  // set - refused in upstream's words), and the pipeline's own annotations, which are refused
  // by name rather than written, since nothing here reads them back as a condition.
  C_NTR: { fullName: 'condition_n-terminus_1_residue', dtype: 'bool', default: false },
  'n-terminus': { fullName: 'condition_n-terminus_1_residue', dtype: 'bool', default: false },
  'condition_n-terminus_1_residue': { fullName: 'condition_n-terminus_1_residue', dtype: 'bool', default: false },
  C_CTR: { fullName: 'condition_c-terminus_1_residue', dtype: 'bool', default: false },
  'c-terminus': { fullName: 'condition_c-terminus_1_residue', dtype: 'bool', default: false },
  'condition_c-terminus_1_residue': { fullName: 'condition_c-terminus_1_residue', dtype: 'bool', default: false },
  mask_hotspot_1_atom: { fullName: 'condition_hotspot_1_atom', dtype: 'bool', default: false },
  mask_pistack_1_atom: { fullName: 'condition_pistack_1_atom', dtype: 'bool', default: false },
  mask_hydrophobic_1_atom: { fullName: 'condition_hydrophobic_1_atom', dtype: 'bool', default: false },
  C_BP: { fullName: 'condition_basepair_2_residue', dtype: 'twobody', default: 0 },
  basepair: { fullName: 'condition_basepair_2_residue', dtype: 'twobody', default: 0 },
  condition_basepair_2_residue: { fullName: 'condition_basepair_2_residue', dtype: 'twobody', default: 0 },
  SA_CRDSRC: { fullName: 'annotation_src-coordinate-conditioned-region_1_atom', dtype: 'internal', default: '' },
  SA_IDXGRPSRC: { fullName: 'annotation_src-index-group_1_atom', dtype: 'internal', default: 0 },
  SA_UII: { fullName: 'is_unindexed_insertion', dtype: 'internal', default: false },
  SA_CWI: { fullName: 'annotation_coords-with-insertion_1_atom', dtype: 'internal', default: 0 },
  S_SEGMIN: { fullName: 'annotation_expsegmin_1_residue', dtype: 'internal', default: -1 },
  S_SEGMAX: { fullName: 'annotation_expsegmax_1_residue', dtype: 'internal', default: -1 },
  S_ATM: { fullName: 'atomize', dtype: 'internal', default: false },

  SA_SRC: { fullName: 'annotation_src-component_1_atom', dtype: 'str', default: '.' },
  'src-component': { fullName: 'annotation_src-component_1_atom', dtype: 'str', default: '.' },
  'annotation_src-component_1_atom': { fullName: 'annotation_src-component_1_atom', dtype: 'str', default: '.' },
};

export function resolveConditionSpec(name) {
  if (CONDITION_REGISTRY[name]) return CONDITION_REGISTRY[name];
  if (name.startsWith('C_SEQ_PROB_')) {
    const tok = name.slice('C_SEQ_PROB_'.length);
    return { fullName: `condition_sequence-probability-${tok}_1_residue`, dtype: 'float', default: NaN };
  }
  if (name.startsWith('sequence-probability-')) {
    const tok = name.slice('sequence-probability-'.length);
    return { fullName: `condition_sequence-probability-${tok}_1_residue`, dtype: 'float', default: NaN };
  }
  if (name.startsWith('condition_sequence-probability-')) {
    return { fullName: name, dtype: 'float', default: NaN };
  }
  throw new Error(`Unrecognized Condition/StandardAnnotation name: ${name}`);
}

// ---------------------------------------------------------------------------
// YAML / JSON Parser for Condition Specs
// ---------------------------------------------------------------------------

function stripYamlComment(line) {
  let inSingle = false;
  let inDouble = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === "'" && !inDouble) {
      if (inSingle && line[i + 1] === "'") { i++; continue; }
      inSingle = !inSingle;
    } else if (ch === '"' && !inSingle) {
      if (ch === '\\') { i++; continue; }
      inDouble = !inDouble;
    } else if (ch === '#' && !inSingle && !inDouble) {
      if (i === 0 || /\s/.test(line[i - 1])) {
        return line.slice(0, i);
      }
    }
  }
  return line;
}

function parseYamlScalar(raw) {
  const s = raw.trim();
  if (s === '') return null;
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
    const q = s[0];
    const inner = s.slice(1, -1);
    if (q === "'") return inner.replace(/''/g, "'");
    return inner.replace(/\\(.)/g, '$1');
  }
  if (s === 'true' || s === 'True' || s === 'TRUE') return true;
  if (s === 'false' || s === 'False' || s === 'FALSE') return false;
  if (s === 'null' || s === 'Null' || s === 'NULL' || s === '~') return null;
  if (/^-?\d+$/.test(s)) return parseInt(s, 10);
  if (/^-?\d+\.\d*(?:[eE][+-]?\d+)?$/.test(s)) return parseFloat(s);
  return s;
}

function splitTopLevel(str, sep) {
  const parts = [];
  let buf = '';
  let depth = 0;
  let inSingle = false;
  let inDouble = false;
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (ch === "'" && !inDouble) {
      if (inSingle && str[i + 1] === "'") { buf += "''"; i++; continue; }
      inSingle = !inSingle;
      buf += ch;
      continue;
    }
    if (ch === '"' && !inSingle) {
      if (str[i - 1] !== '\\') inDouble = !inDouble;
      buf += ch;
      continue;
    }
    if (!inSingle && !inDouble) {
      if (ch === '[' || ch === '{' || ch === '(') depth++;
      else if (ch === ']' || ch === '}' || ch === ')') depth--;
      else if (ch === sep && depth === 0) {
        parts.push(buf);
        buf = '';
        continue;
      }
    }
    buf += ch;
  }
  parts.push(buf);
  return parts;
}

function findTopLevelColon(str) {
  let depth = 0;
  let inSingle = false;
  let inDouble = false;
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (ch === "'" && !inDouble) {
      if (inSingle && str[i + 1] === "'") { i++; continue; }
      inSingle = !inSingle;
      continue;
    }
    if (ch === '"' && !inSingle) {
      if (str[i - 1] !== '\\') inDouble = !inDouble;
      continue;
    }
    if (!inSingle && !inDouble) {
      if (ch === '[' || ch === '{' || ch === '(') depth++;
      else if (ch === ']' || ch === '}' || ch === ')') depth--;
      else if (ch === ':' && depth === 0) {
        if (i + 1 === str.length || /\s/.test(str[i + 1])) return i;
      }
    }
  }
  return -1;
}

function parseFlowValue(str) {
  const s = str.trim();
  if (s.startsWith('{') && s.endsWith('}')) {
    const inner = s.slice(1, -1).trim();
    const obj = {};
    if (!inner) return obj;
    for (const part of splitTopLevel(inner, ',')) {
      const p = part.trim();
      if (!p) continue;
      const colonIdx = findTopLevelColon(p);
      if (colonIdx < 0) throw new Error(`Invalid YAML flow mapping entry: ${p}`);
      const kRaw = p.slice(0, colonIdx).trim();
      const vRaw = p.slice(colonIdx + 1).trim();
      const key = String(parseYamlScalar(kRaw));
      obj[key] = parseFlowValue(vRaw);
    }
    return obj;
  }
  if (s.startsWith('[') && s.endsWith(']')) {
    const inner = s.slice(1, -1).trim();
    if (!inner) return [];
    const arr = [];
    for (const part of splitTopLevel(inner, ',')) {
      const p = part.trim();
      if (!p) continue;
      arr.push(parseFlowValue(p));
    }
    return arr;
  }
  return parseYamlScalar(s);
}

/**
 * Parse a condition-spec YAML or JSON string into a plain JS object.
 */
export function parseConditionSpecYaml(text) {
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) {
    try {
      return JSON.parse(trimmed);
    } catch {
      return parseFlowValue(trimmed);
    }
  }

  // Preprocess lines: strip comments, join multiline flow collections ({...} or [...])
  const rawLines = text.split(/\r?\n/);
  const lines = [];
  for (let i = 0; i < rawLines.length; i++) {
    const cleaned = stripYamlComment(rawLines[i]).trimEnd();
    if (!cleaned.trim()) continue;
    const indent = cleaned.match(/^ */)[0].length;
    let content = cleaned.slice(indent);

    // Check if content has unclosed { or [
    const countUnclosed = (s) => {
      let d = 0, inS = false, inD = false;
      for (let j = 0; j < s.length; j++) {
        const c = s[j];
        if (c === "'" && !inD) inS = !inS;
        else if (c === '"' && !inS && s[j - 1] !== '\\') inD = !inD;
        else if (!inS && !inD) {
          if (c === '{' || c === '[') d++;
          else if (c === '}' || c === ']') d--;
        }
      }
      return d;
    };

    while (countUnclosed(content) > 0 && i + 1 < rawLines.length) {
      i++;
      const nextClean = stripYamlComment(rawLines[i]).trim();
      if (nextClean) content += ' ' + nextClean;
    }

    lines.push({ indent, content });
  }

  let pos = 0;

  function parseBlock(minIndent) {
    if (pos >= lines.length || lines[pos].indent < minIndent) return null;
    if (lines[pos].content.startsWith('- ') || lines[pos].content === '-') {
      const arr = [];
      const listIndent = lines[pos].indent;
      while (pos < lines.length && lines[pos].indent === listIndent && (lines[pos].content.startsWith('- ') || lines[pos].content === '-')) {
        const rest = lines[pos].content.slice(1).trim();
        pos++;
        if (!rest) {
          arr.push(parseBlock(listIndent + 1));
        } else if (rest.startsWith('{') || rest.startsWith('[')) {
          arr.push(parseFlowValue(rest));
        } else {
          const colonIdx = findTopLevelColon(rest);
          if (colonIdx >= 0) {
            const k = String(parseYamlScalar(rest.slice(0, colonIdx)));
            const vStr = rest.slice(colonIdx + 1).trim();
            const itemObj = {};
            if (vStr) {
              itemObj[k] = vStr.startsWith('{') || vStr.startsWith('[') ? parseFlowValue(vStr) : parseYamlScalar(vStr);
            } else {
              itemObj[k] = parseBlock(listIndent + 1);
            }
            while (pos < lines.length && lines[pos].indent > listIndent) {
              const subLine = lines[pos].content;
              const subColon = findTopLevelColon(subLine);
              if (subColon < 0) break;
              const sk = String(parseYamlScalar(subLine.slice(0, subColon)));
              const sv = subLine.slice(subColon + 1).trim();
              const subIndent = lines[pos].indent;
              pos++;
              itemObj[sk] = sv
                ? (sv.startsWith('{') || sv.startsWith('[') ? parseFlowValue(sv) : parseYamlScalar(sv))
                : parseBlock(subIndent + 1);
            }
            arr.push(itemObj);
          } else {
            arr.push(parseYamlScalar(rest));
          }
        }
      }
      return arr;
    }

    const obj = {};
    const mapIndent = lines[pos].indent;
    while (pos < lines.length && lines[pos].indent === mapIndent && !lines[pos].content.startsWith('-')) {
      const line = lines[pos].content;
      const colonIdx = findTopLevelColon(line);
      if (colonIdx < 0) throw new Error(`Invalid YAML mapping line: ${line}`);
      const k = String(parseYamlScalar(line.slice(0, colonIdx)));
      const vStr = line.slice(colonIdx + 1).trim();
      pos++;
      if (vStr) {
        obj[k] = vStr.startsWith('{') || vStr.startsWith('[') ? parseFlowValue(vStr) : parseYamlScalar(vStr);
      } else {
        obj[k] = parseBlock(mapIndent + 1);
      }
    }
    return obj;
  }

  return parseBlock(0) || {};
}

// ---------------------------------------------------------------------------
// Path-Selection DSL (`sel('...')`) and QueryExpression Evaluator
// ---------------------------------------------------------------------------

/**
 * `sel('...')`'s path DSL, upstream's (`AtomSelectionStack.from_query`): see query.js.
 * Returns a boolean array of length `atoms.nAtoms`.
 */
export function evalPathSelection(atoms, selectionStr) {
  return pathMask(atoms, selectionStr);
}

/**
 * A selection: upstream's whole query language (`resolve_mask`), see query.js. The evaluator
 * that stood here was hand-written and wrong in five ways query.js lists.
 */
export function evalSelection(atoms, queryStr) {
  return evalQuery(atoms, queryStr, resolveConditionSpec);
}

// ---------------------------------------------------------------------------
// AtomArray slicing / concatenation / expandable-segment insertion helpers
// ---------------------------------------------------------------------------

const ARRAY_KEYS = [
  'hetero',
  'element',
  'atom_name',
  'res_name',
  'chain_id',
  'label_entity_id',
  'res_id',
  'ins_code',
  'auth_seq_id',
  'atom_id',
  'b_factor',
  'occupancy',
  'charge',
  'label_alt_id',
  'coord',
  'transformation_id',
  'is_polymer',
  'chain_type',
  'chem_comp_type',
  'atomic_number',
  'is_aromatic',
  'is_leaving_atom',
  'pn_unit_id',
  'molecule_id',
  'chain_entity',
  'pn_unit_entity',
  'molecule_entity',
  'chain_iid',
  'pn_unit_iid',
  'molecule_iid',
];

export function sliceAtomArray(atoms, indices) {
  const nAtoms = indices.length;
  const oldToNew = new Map();
  for (let i = 0; i < nAtoms; i++) oldToNew.set(indices[i], i);

  const out = { nAtoms, bonds: [], annotations: {} };
  for (const key of ARRAY_KEYS) {
    if (!atoms[key]) continue;
    if (key === 'coord') {
      out.coord = indices.map((idx) => atoms.coord[idx].slice());
    } else {
      out[key] = indices.map((idx) => atoms[key][idx]);
    }
  }
  for (const [annKey, annArr] of Object.entries(atoms.annotations || {})) {
    out.annotations[annKey] = indices.map((idx) =>
      Array.isArray(annArr[idx]) ? annArr[idx].slice() : annArr[idx]
    );
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

export function concatAtomArrays(arrays) {
  if (arrays.length === 0) throw new Error('Cannot concatenate empty list of AtomArrays');
  if (arrays.length === 1) return sliceAtomArray(arrays[0], Array.from({ length: arrays[0].nAtoms }, (_, i) => i));

  let totalAtoms = 0;
  const allAnnKeys = new Set();
  for (const a of arrays) {
    totalAtoms += a.nAtoms;
    for (const k of Object.keys(a.annotations || {})) allAnnKeys.add(k);
  }

  const out = { nAtoms: totalAtoms, bonds: [], annotations: {} };
  for (const key of ARRAY_KEYS) {
    if (arrays.some((a) => a[key] !== undefined)) {
      out[key] = [];
      for (const a of arrays) {
        const arr = a[key] || new Array(a.nAtoms).fill(null);
        for (let i = 0; i < a.nAtoms; i++) {
          out[key].push(key === 'coord' ? arr[i].slice() : arr[i]);
        }
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
      const src = a.annotations?.[annKey];
      for (let i = 0; i < a.nAtoms; i++) {
        const v = src ? src[i] : defVal;
        out.annotations[annKey].push(Array.isArray(v) ? v.slice() : v);
      }
    }
  }

  let offset = 0;
  for (const a of arrays) {
    for (const [u, v, bt] of a.bonds || []) {
      out.bonds.push([u + offset, v + offset, bt]);
    }
    offset += a.nAtoms;
  }

  return out;
}

function findFlankingBondAtomIdxs(atoms, sentinelIdx, chainKey) {
  const ct = atoms.chain_type[sentinelIdx];
  const [, startName] = getPolymerizationAtoms('', ct);
  const [termName] = getPolymerizationAtoms('', ct);
  if (!termName || !startName) return [null, null];

  const chainVals = atoms[chainKey] || atoms.chain_id;
  const chainVal = chainVals[sentinelIdx];
  const resBounds = getResidueSegments(atoms);
  let sentinelResIdx = 0;
  for (let r = 0; r < resBounds.length - 1; r++) {
    if (sentinelIdx >= resBounds[r] && sentinelIdx < resBounds[r + 1]) {
      sentinelResIdx = r;
      break;
    }
  }

  const segmin = atoms.annotations['annotation_expsegmin_1_residue'];
  const isExpandable = (idx) => Boolean(segmin && segmin[idx] !== -1);

  let prevTerminalIdx = null;
  if (sentinelResIdx > 0) {
    const ps = resBounds[sentinelResIdx - 1];
    const pe = resBounds[sentinelResIdx];
    if (chainVals[ps] === chainVal) {
      for (let ai = ps; ai < pe; ai++) {
        if (atoms.atom_name[ai] === termName || (pe - ps === 1 && isExpandable(ai))) {
          prevTerminalIdx = ai;
          break;
        }
      }
    }
  }

  let nextStartIdx = null;
  if (sentinelResIdx < resBounds.length - 2) {
    const ns = resBounds[sentinelResIdx + 1];
    const ne = resBounds[sentinelResIdx + 2];
    if (chainVals[ns] === chainVal) {
      for (let ai = ns; ai < ne; ai++) {
        if (atoms.atom_name[ai] === startName || (ne - ns === 1 && isExpandable(ai))) {
          nextStartIdx = ai;
          break;
        }
      }
    }
  }

  return [prevTerminalIdx, nextStartIdx];
}

export function installSentinelBonds(atoms, sentinelIdx, chainKey = 'chain_iid') {
  const [prevIdx, nextIdx] = findFlankingBondAtomIdxs(atoms, sentinelIdx, chainKey);
  if (prevIdx === null && nextIdx === null) return;

  if (prevIdx !== null && nextIdx !== null) {
    const u0 = Math.min(prevIdx, nextIdx);
    const v0 = Math.max(prevIdx, nextIdx);
    atoms.bonds = atoms.bonds.filter(([u, v]) => !(Math.min(u, v) === u0 && Math.max(u, v) === v0));
  }

  const hasBond = (a, b) => {
    const u0 = Math.min(a, b);
    const v0 = Math.max(a, b);
    return atoms.bonds.some(([u, v]) => Math.min(u, v) === u0 && Math.max(u, v) === v0);
  };

  if (prevIdx !== null && !hasBond(prevIdx, sentinelIdx)) {
    atoms.bonds.push([Math.min(prevIdx, sentinelIdx), Math.max(prevIdx, sentinelIdx), BOND_TYPE.SINGLE]);
  }
  if (nextIdx !== null && !hasBond(sentinelIdx, nextIdx)) {
    atoms.bonds.push([Math.min(sentinelIdx, nextIdx), Math.max(sentinelIdx, nextIdx), BOND_TYPE.SINGLE]);
  }
}

export function insertExpandableSegment(
  atoms,
  segMin,
  segMax,
  {
    atomIdx = null,
    insertBefore = true,
    newChain = false,
    newChainId = null,
    newChainType = null,
  } = {}
) {
  const n = atoms ? atoms.nAtoms : 0;
  if (n === 0 && !newChain) throw new Error('Cannot insert an expandable segment into an empty array without newChain=true.');

  const chainKey = atoms && atoms.chain_iid ? 'chain_iid' : 'chain_id';
  let insertPosition;
  let sourceAtomIdx;
  let chainId;
  let sentinelResId;
  let needsShift = false;
  let chainIdentifierValue;
  let chainType = null;

  if (newChain) {
    insertPosition = n;
    sourceAtomIdx = n > 0 ? n - 1 : -1;
    chainId = newChainId;
    sentinelResId = 1;
    chainIdentifierValue = chainId;
    chainType = newChainType ?? CHAIN_TYPE.POLYPEPTIDE_L;
  } else {
    insertPosition = insertBefore ? atomIdx : atomIdx + 1;
    sourceAtomIdx = atomIdx;
    chainId = String(atoms.chain_id[atomIdx]);
    chainIdentifierValue = atoms[chainKey][atomIdx];

    const chainIndices = [];
    for (let i = 0; i < n; i++) {
      if (atoms[chainKey][i] === chainIdentifierValue) chainIndices.push(i);
    }
    const nBefore = chainIndices.filter((idx) => idx < insertPosition).length;
    const nAfter = chainIndices.length - nBefore;
    if (nBefore > 0) {
      const prevAtomIdx = chainIndices[nBefore - 1];
      const prevResId = atoms.res_id[prevAtomIdx];
      if (nAfter > 0) {
        const nextAtomIdx = chainIndices[nBefore];
        const nextResId = atoms.res_id[nextAtomIdx];
        if (nextResId > prevResId + 1) {
          sentinelResId = nextResId - 1;
          needsShift = false;
        } else {
          sentinelResId = prevResId + 1;
          needsShift = true;
        }
      } else {
        sentinelResId = prevResId + 1;
        needsShift = true;
      }
    } else {
      const firstResId = atoms.res_id[chainIndices[0]];
      if (firstResId > 0) {
        sentinelResId = firstResId - 1;
        needsShift = false;
      } else {
        sentinelResId = firstResId;
        needsShift = true;
      }
    }
  }

  if (!atoms) {
    atoms = createEmptyAtomArray();
  }
  // Ensure expsegmin, expsegmax, _expseg_uid exist on atoms
  if (!atoms.annotations['annotation_expsegmin_1_residue']) {
    atoms.annotations['annotation_expsegmin_1_residue'] = new Array(n).fill(-1);
  }
  if (!atoms.annotations['annotation_expsegmax_1_residue']) {
    atoms.annotations['annotation_expsegmax_1_residue'] = new Array(n).fill(-1);
  }
  if (!atoms.annotations['_expseg_uid']) {
    atoms.annotations['_expseg_uid'] = new Array(n).fill(-1);
  }
  const maxUid = atoms.annotations['_expseg_uid'].reduce((m, v) => Math.max(m, v), -1);
  const segUid = maxUid + 1;

  // Build 1-atom sentinel copied from sourceAtomIdx (or fresh when n === 0)
  const sentinel =
    sourceAtomIdx >= 0
      ? sliceAtomArray(atoms, [sourceAtomIdx])
      : {
          ...createEmptyAtomArray(),
          nAtoms: 1,
          hetero: [true],
          element: ['X'],
          atom_name: ['UNK'],
          res_name: [`${segMin}-${segMax}`],
          chain_id: [chainId],
          label_entity_id: [1],
          res_id: [sentinelResId],
          ins_code: [''],
          auth_seq_id: [sentinelResId],
          atom_id: [1],
          b_factor: [NaN],
          occupancy: [0.0],
          charge: [0],
          label_alt_id: ['.'],
          coord: [[NaN, NaN, NaN]],
          transformation_id: ['1'],
          is_polymer: [(chainType ?? CHAIN_TYPE.POLYPEPTIDE_L) <= CHAIN_TYPE.RNA],
          chain_type: [chainType ?? CHAIN_TYPE.POLYPEPTIDE_L],
          chem_comp_type: ['NON-POLYMER'],
          atomic_number: [0],
          is_aromatic: [false],
          is_leaving_atom: [false],
          bonds: [],
          annotations: {},
        };
  sentinel.chain_id[0] = chainId;
  sentinel.res_name[0] = `${segMin}-${segMax}`;
  sentinel.atom_name[0] = 'UNK';
  sentinel.element[0] = 'X';
  sentinel.coord[0] = [NaN, NaN, NaN];
  sentinel.hetero[0] = true;
  sentinel.res_id[0] = sentinelResId;
  if (sentinel.ins_code) sentinel.ins_code[0] = '';
  if (sentinel.occupancy) sentinel.occupancy[0] = 0.0;
  if (sentinel.b_factor) sentinel.b_factor[0] = NaN;
  if (sentinel.charge) sentinel.charge[0] = 0;
  if (sentinel.atomic_number) sentinel.atomic_number[0] = 0;
  if (sentinel.is_aromatic) sentinel.is_aromatic[0] = false;
  if (sentinel.is_leaving_atom) sentinel.is_leaving_atom[0] = false;
  if (sentinel.chem_comp_type) sentinel.chem_comp_type[0] = 'NON-POLYMER';

  if (chainType !== null) {
    sentinel.chain_type[0] = chainType;
    sentinel.is_polymer[0] = chainType <= CHAIN_TYPE.RNA;
  }
  if (sentinel.label_entity_id && sourceAtomIdx >= 0 && chainId !== atoms.chain_id[sourceAtomIdx]) {
    const maxEid = atoms.label_entity_id.reduce((m, v) => Math.max(m, v), 0);
    sentinel.label_entity_id[0] = maxEid + 1;
  }

  sentinel.annotations['annotation_expsegmin_1_residue'] = [segMin];
  sentinel.annotations['annotation_expsegmax_1_residue'] = [segMax];
  sentinel.annotations['_expseg_uid'] = [segUid];

  // Reset any other present standard annotations on sentinel to defaults
  for (const annKey of Object.keys(sentinel.annotations)) {
    if (
      annKey === 'annotation_expsegmin_1_residue' ||
      annKey === 'annotation_expsegmax_1_residue' ||
      annKey === '_expseg_uid'
    ) {
      continue;
    }
    const sc = SCALAR_1BODY_ANNOTATIONS[annKey];
    const nsc = NONSCALAR_1BODY_ANNOTATIONS[annKey];
    if (sc) sentinel.annotations[annKey] = [sc.default];
    else if (nsc) sentinel.annotations[annKey] = [new Array(nsc.cols).fill(nsc.default)];
    else if (annKey.startsWith('condition_sequence-probability-')) sentinel.annotations[annKey] = [NaN];
  }

  // Insert sentinel at insertPosition
  const before = insertPosition > 0 ? sliceAtomArray(atoms, Array.from({ length: insertPosition }, (_, i) => i)) : null;
  const after = insertPosition < n ? sliceAtomArray(atoms, Array.from({ length: n - insertPosition }, (_, i) => insertPosition + i)) : null;

  // Note: sliceAtomArray drops cross-slice bonds, so we splice and remap bonds directly!
  const pieces = [before, sentinel, after].filter(Boolean);
  const result = concatAtomArrays(pieces);

  // Restore all original bonds with index >= insertPosition shifted by +1
  result.bonds = [];
  for (const [u, v, bt] of atoms.bonds || []) {
    const nu = u >= insertPosition ? u + 1 : u;
    const nv = v >= insertPosition ? v + 1 : v;
    result.bonds.push([nu, nv, bt]);
  }

  if (!newChain) {
    if (needsShift) {
      for (let i = insertPosition + 1; i < result.nAtoms; i++) {
        if (result[chainKey][i] === chainIdentifierValue && result.res_id[i] >= sentinelResId) {
          result.res_id[i] += 1;
        }
      }
    }
    installSentinelBonds(result, insertPosition, chainKey);
  }

  computeIdAndEntityAnnotations(result);
  return result;
}

// ---------------------------------------------------------------------------
// Contig Parser & `applyContig`
// ---------------------------------------------------------------------------

function* createChainIdGenerator(unavailableChainIds) {
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  for (let i = 0; i < letters.length; i++) {
    const cid = letters[i];
    if (!unavailableChainIds.has(cid)) yield cid;
  }
  for (let i = 0; i < letters.length; i++) {
    for (let j = 0; j < letters.length; j++) {
      const cid = letters[i] + letters[j];
      if (!unavailableChainIds.has(cid)) yield cid;
    }
  }
}

function parseLengthAndSuffix(body, token) {
  let chainType = null;
  let b = body;
  if (b && SUFFIX_TO_CHAIN_TYPE[b[b.length - 1]] !== undefined) {
    chainType = SUFFIX_TO_CHAIN_TYPE[b[b.length - 1]];
    b = b.slice(0, -1);
  }
  if (!b || !/^[0-9-]+$/.test(b)) {
    throw new Error(`Malformed expandable-segment token: ${token}`);
  }
  const parts = b.split('-');
  let segMin, segMax;
  if (parts.length === 1) {
    segMin = segMax = parseInt(parts[0], 10);
  } else if (parts.length === 2) {
    segMin = parseInt(parts[0], 10);
    segMax = parseInt(parts[1], 10);
  } else {
    throw new Error(`Malformed expandable-segment token: ${token}`);
  }
  if (segMax < segMin) {
    throw new Error(`Expandable segment ${token} has end < start (${segMax} < ${segMin}).`);
  }
  return { segMin, segMax, chainType };
}

function parseContigToken(rawToken) {
  const token = rawToken.trim();
  if (!token) throw new Error('Empty contig token (check for stray/trailing commas).');
  if (token === '/0') return { kind: 'chainbreak' };
  if (token.includes(':')) {
    const idx = token.indexOf(':');
    const name = token.slice(0, idx);
    const body = token.slice(idx + 1);
    if (!name || name === '<M>') {
      throw new Error(`Invalid named expandable-segment token: ${token}`);
    }
    const { segMin, segMax, chainType } = parseLengthAndSuffix(body, token);
    return { kind: 'expandable', name, segMin, segMax, chainType };
  }
  if (/^[A-Za-z]/.test(token)) {
    const m = token.match(/^([A-Za-z]+)(\d+)(?:-(\d+))?$/);
    if (!m) throw new Error(`Malformed preserved-segment token: ${token}`);
    const chainId = m[1];
    const start = parseInt(m[2], 10);
    const end = m[3] !== undefined ? parseInt(m[3], 10) : start;
    if (end < start) throw new Error(`Preserved segment ${token} has end < start (${end} < ${start}).`);
    return { kind: 'preserved', chainId, start, end };
  }
  const { segMin, segMax, chainType } = parseLengthAndSuffix(token, token);
  return { kind: 'expandable', name: null, segMin, segMax, chainType };
}

/**
 * Subset `atomArrayIn` and insert expandable segments per `contig`.
 * Returns `{ atomArray, originalResId }`.
 */
export function applyContig(atomArrayIn, contig) {
  const tokens = contig.split(',').map(parseContigToken);
  // 🔴 UPSTREAM'S CHECKS, IN UPSTREAM'S WORDS (`inference_utils/contig_spec.py`), which this port
  // did not have: a duplicated segment name, two preserved segments sharing residues (which
  // duplicates them, contiguously renumbered), a range running past its chain, a suffix that
  // disagrees with the chain it is anchored on, and an input with nothing to preserve. Each was a
  // job built silently into something its author did not write.
  const names = tokens.filter((t) => t.kind === 'expandable' && t.name !== null).map((t) => t.name);
  const dupes = [...new Set(names.filter((n, i) => names.indexOf(n) !== i))].sort();
  if (dupes.length > 0) {
    throw new Error(`Duplicate expandable-segment name(s) in contig: ${JSON.stringify(dupes)}`);
  }
  const runs = [[]];
  for (const tok of tokens) {
    if (tok.kind === 'chainbreak') runs.push([]);
    else runs[runs.length - 1].push(tok);
  }
  const nonEmptyRuns = runs.filter((r) => r.length > 0);
  if (nonEmptyRuns.length === 0) {
    throw new Error('Contig string produced no chain runs.');
  }
  const byChain = new Map();
  for (const run of nonEmptyRuns) {
    for (const tok of run) {
      if (tok.kind !== 'preserved') continue;
      if (!byChain.has(tok.chainId)) byChain.set(tok.chainId, []);
      byChain.get(tok.chainId).push(tok);
    }
  }
  for (const [chainId, segs] of byChain) {
    const sorted = segs.slice().sort((a, b) => a.start - b.start);
    for (let k = 1; k < sorted.length; k++) {
      const prev = sorted[k - 1];
      const curr = sorted[k];
      if (curr.start <= prev.end) {
        throw new Error(
          `Preserved segments on chain '${chainId}' overlap: ${chainId}${prev.start}-${prev.end} and ` +
            `${chainId}${curr.start}-${curr.end} both reference some of the same residues.`
        );
      }
    }
  }
  // Upstream requires an input, so a contig with nothing to preserve is refused there. With NO
  // input this port realises it into an empty structure (its pure-contig jobs), so only an input
  // with nothing preserved is refused.
  if ((atomArrayIn?.nAtoms ?? 0) > 0 && byChain.size === 0) {
    throw new Error(
      "Contig has no preserved segments -- condition_spec's `input` structure exists only to be " +
        'referenced by preserved segments, so a contig that generates everything from scratch ' +
        "doesn't need an `input` file at all."
    );
  }

  const claimedChainIds = new Set();
  for (const run of nonEmptyRuns) {
    for (const tok of run) {
      if (tok.kind === 'preserved') claimedChainIds.add(tok.chainId);
    }
  }
  const synthGen = createChainIdGenerator(claimedChainIds);
  const chainRuns = [];
  const seenChainIds = new Set();

  for (const runTokens of nonEmptyRuns) {
    const preserved = runTokens.filter((t) => t.kind === 'preserved');
    let chainId;
    if (preserved.length > 0) {
      const cids = new Set(preserved.map((t) => t.chainId));
      if (cids.size > 1) {
        throw new Error(`An output chain mixes preserved segments from multiple chains: ${Array.from(cids)}`);
      }
      chainId = preserved[0].chainId;
    } else {
      chainId = synthGen.next().value;
    }
    if (seenChainIds.has(chainId)) {
      throw new Error(`Contig produces two separate output chains both named ${chainId}`);
    }
    seenChainIds.add(chainId);
    chainRuns.push({ tokens: runTokens, chainId });
  }

  const buildRunPreserved = (run) => {
    const pieces = [];
    const origResPieces = [];
    const tokenOffsets = new Map();
    let cursor = 0;
    let nextResId = 1;

    for (let tokIdx = 0; tokIdx < run.tokens.length; tokIdx++) {
      const tok = run.tokens[tokIdx];
      if (tok.kind !== 'preserved') continue;
      const selected = [];
      for (let i = 0; i < atomArrayIn.nAtoms; i++) {
        if (atomArrayIn.chain_id[i] === tok.chainId && atomArrayIn.res_id[i] >= tok.start && atomArrayIn.res_id[i] <= tok.end) {
          selected.push(i);
        }
      }
      if (selected.length === 0) {
        throw new Error(`Preserved segment '${tok.chainId}${tok.start}-${tok.end}' matched no atoms.`);
      }
      let chainMin = Infinity;
      let chainMax = -Infinity;
      for (let i = 0; i < atomArrayIn.nAtoms; i++) {
        if (atomArrayIn.chain_id[i] !== tok.chainId) continue;
        chainMin = Math.min(chainMin, atomArrayIn.res_id[i]);
        chainMax = Math.max(chainMax, atomArrayIn.res_id[i]);
      }
      if (tok.start < chainMin || tok.end > chainMax) {
        throw new Error(
          `Preserved segment '${tok.chainId}${tok.start}-${tok.end}' runs outside chain ${tok.chainId}, ` +
            `whose residues are ${chainMin}-${chainMax}.`
        );
      }
      const segment = sliceAtomArray(atomArrayIn, selected);
      const resStarts = getResidueSegments(segment);
      for (let r = 0; r < resStarts.length - 1; r++) {
        for (let i = resStarts[r]; i < resStarts[r + 1]; i++) {
          segment.res_id[i] = nextResId + r;
        }
      }
      pieces.push(segment);
      origResPieces.push(...selected.map((idx) => atomArrayIn.res_id[idx]));
      tokenOffsets.set(tokIdx, [cursor, cursor + segment.nAtoms]);
      cursor += segment.nAtoms;
      nextResId += resStarts.length - 1;
    }

    if (pieces.length === 0) return { preservedAtoms: null, runOriginalResId: [], tokenOffsets };
    const preservedAtoms = concatAtomArrays(pieces);
    return { preservedAtoms, runOriginalResId: origResPieces, tokenOffsets };
  };

  const resolveRunChainType = (run) => {
    const preserved = run.tokens.find((t) => t.kind === 'preserved');
    let anchorType = null;
    if (preserved) {
      const types = new Set();
      for (let i = 0; i < atomArrayIn.nAtoms; i++) {
        if (atomArrayIn.chain_id[i] === preserved.chainId) types.add(atomArrayIn.chain_type[i]);
      }
      if (types.size > 1) {
        throw new Error(
          `Input chain '${preserved.chainId}' contains multiple ChainTypes -- a contig run anchored on ` +
            "this chain can't unambiguously default an unsuffixed expandable segment's type. Split this " +
            'chain into separately-named chains upstream.'
        );
      }
      if (types.size === 1) anchorType = [...types][0];
    }
    const requested = new Set(
      run.tokens.filter((t) => t.kind === 'expandable' && t.chainType !== null).map((t) => t.chainType)
    );
    if (requested.size > 1) throw new Error('Chain run requests conflicting chain types.');
    if (requested.size === 0) return anchorType !== null ? anchorType : CHAIN_TYPE.POLYPEPTIDE_L;
    const chainType = Array.from(requested)[0];
    if (anchorType !== null && chainType !== anchorType) {
      throw new Error(
        `An expandable segment explicitly specifies chain type ${chainType}, which disagrees with its ` +
          `run's anchor chain type ${anchorType}.`
      );
    }
    return chainType;
  };

  const processRun = (run, runPreservedAtoms, runOriginalResId, tokenOffsets, assembled, originalResId) => {
    const chainType = resolveRunChainType(run);
    const runStart = assembled ? assembled.nAtoms : 0;
    let currAssembled = assembled;
    let currOrigResId = originalResId.slice();

    if (runPreservedAtoms) {
      if (currAssembled && currAssembled.annotations['_expseg_uid'] && !runPreservedAtoms.annotations['_expseg_uid']) {
        runPreservedAtoms.annotations['_expseg_uid'] = new Array(runPreservedAtoms.nAtoms).fill(-1);
      }
      currAssembled = currAssembled ? concatAtomArrays([currAssembled, runPreservedAtoms]) : runPreservedAtoms;
      currOrigResId.push(...runOriginalResId);
    }

    let runTail = runStart + (runPreservedAtoms ? runPreservedAtoms.nAtoms : 0);
    const liveOffsets = new Map();
    for (const [idx, [s, e]] of tokenOffsets.entries()) {
      liveOffsets.set(idx, [runStart + s, runStart + e]);
    }

    let firstExpandableInRun = true;
    for (let tokIdx = 0; tokIdx < run.tokens.length; tokIdx++) {
      const tok = run.tokens[tokIdx];
      if (tok.kind !== 'expandable') continue;

      let nextIdx = null;
      for (let j = tokIdx + 1; j < run.tokens.length; j++) {
        if (run.tokens[j].kind === 'preserved') { nextIdx = j; break; }
      }

      let insertPosition;
      if (nextIdx !== null) {
        const atomIdx = liveOffsets.get(nextIdx)[0];
        currAssembled = insertExpandableSegment(currAssembled, tok.segMin, tok.segMax, {
          atomIdx,
          insertBefore: true,
        });
        insertPosition = atomIdx;
      } else if (liveOffsets.size > 0) {
        const atomIdx = runTail - 1;
        currAssembled = insertExpandableSegment(currAssembled, tok.segMin, tok.segMax, {
          atomIdx,
          insertBefore: false,
        });
        insertPosition = atomIdx + 1;
      } else if (firstExpandableInRun) {
        currAssembled = insertExpandableSegment(currAssembled, tok.segMin, tok.segMax, {
          newChain: true,
          newChainId: run.chainId,
          newChainType: chainType,
        });
        insertPosition = runTail;
      } else {
        currAssembled = insertExpandableSegment(currAssembled, tok.segMin, tok.segMax, {
          atomIdx: runTail - 1,
          insertBefore: false,
        });
        insertPosition = runTail;
      }

      firstExpandableInRun = false;
      currOrigResId.splice(insertPosition, 0, EXPANDABLE_SEGMENT_RES_ID_SENTINEL);
      for (const [k, [s, e]] of liveOffsets.entries()) {
        liveOffsets.set(k, [
          s >= insertPosition ? s + 1 : s,
          e >= insertPosition ? e + 1 : e,
        ]);
      }
      runTail += 1;

      const name = tok.name !== null ? tok.name : `${tok.segMin}-${tok.segMax}`;
      currAssembled.res_name[insertPosition] = name;
    }

    return { assembled: currAssembled, originalResId: currOrigResId };
  };

  let assembled = null;
  let originalResId = [];
  let pendingRuns = [];

  for (const run of chainRuns) {
    const { preservedAtoms, runOriginalResId, tokenOffsets } = buildRunPreserved(run);
    if (assembled === null && preservedAtoms === null) {
      pendingRuns.push({ run, runOriginalResId, tokenOffsets });
      continue;
    }

    ({ assembled, originalResId } = processRun(
      run,
      preservedAtoms,
      runOriginalResId,
      tokenOffsets,
      assembled,
      originalResId
    ));
    const unblockedRunLen = assembled.nAtoms;
    for (const pending of pendingRuns) {
      ({ assembled, originalResId } = processRun(
        pending.run,
        null,
        pending.runOriginalResId,
        pending.tokenOffsets,
        assembled,
        originalResId
      ));
    }
    if (pendingRuns.length > 0) {
      const reorder = [];
      for (let i = unblockedRunLen; i < assembled.nAtoms; i++) reorder.push(i);
      for (let i = 0; i < unblockedRunLen; i++) reorder.push(i);
      assembled = sliceAtomArray(assembled, reorder);
      originalResId = reorder.map((i) => originalResId[i]);
    }
    pendingRuns = [];
  }

  // If the contig has only expandable segments (no preserved segments in any run), drain pendingRuns now
  if (assembled === null && pendingRuns.length > 0) {
    for (const pending of pendingRuns) {
      ({ assembled, originalResId } = processRun(
        pending.run,
        null,
        pending.runOriginalResId,
        pending.tokenOffsets,
        assembled,
        originalResId
      ));
    }
    pendingRuns = [];
  }

  // Uniformly offset each chain's res_id so its lowest value is 1
  const uniqueChains = Array.from(new Set(assembled.chain_id));
  for (const cid of uniqueChains) {
    let minResId = Infinity;
    for (let i = 0; i < assembled.nAtoms; i++) {
      if (assembled.chain_id[i] === cid && assembled.res_id[i] < minResId) {
        minResId = assembled.res_id[i];
      }
    }
    const offset = 1 - minResId;
    if (offset !== 0) {
      for (let i = 0; i < assembled.nAtoms; i++) {
        if (assembled.chain_id[i] === cid) assembled.res_id[i] += offset;
      }
    }
  }

  return { atomArray: assembled, originalResId };
}

// ---------------------------------------------------------------------------
// `applyConditions`
// ---------------------------------------------------------------------------

function coerceBool(val) {
  if (typeof val === 'boolean') return val;
  if (typeof val === 'string' && (val.toLowerCase() === 'true' || val.toLowerCase() === 'false')) {
    return val.toLowerCase() === 'true';
  }
  throw new Error(`Expected boolean condition value (true/false), got ${JSON.stringify(val)}`);
}

function getDefaultCoordinateGroups(atoms) {
  const n = atoms.nAtoms;
  const groups = new Array(n).fill(-1);
  const crd = atoms.annotations['condition_coordinate_1_atom'];
  if (crd) {
    for (let i = 0; i < n; i++) {
      if (Number.isFinite(crd[i][0]) && Number.isFinite(crd[i][1]) && Number.isFinite(crd[i][2])) {
        groups[i] = 0;
      }
    }
  }
  return groups;
}

export function getDefaultIndexGroups(atoms) {
  const n = atoms.nAtoms;
  if (n === 0) return [];
  if (!atoms.molecule_iid) computeIdAndEntityAnnotations(atoms);
  const molAnnot = atoms.molecule_iid;
  const isIndexed = atoms.annotations['condition_index_1_residue']
    ? atoms.annotations['condition_index_1_residue'].map(Boolean)
    : new Array(n).fill(true);

  const uniqueMols = Array.from(new Set(molAnnot)).sort((a, b) => a - b);
  const molToDense = new Map(uniqueMols.map((m, idx) => [m, idx]));
  const nMolecules = uniqueMols.length;

  const groups = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    if (isIndexed[i]) groups[i] = molToDense.get(molAnnot[i]);
  }

  if (isIndexed.some((x) => !x)) {
    const runIds = new Array(n).fill(0);
    let r = 0;
    for (let i = 1; i < n; i++) {
      if (molAnnot[i] !== molAnnot[i - 1] || isIndexed[i] !== isIndexed[i - 1]) r++;
      runIds[i] = r;
    }
    const unindexedRuns = [];
    const seenRun = new Set();
    for (let i = 0; i < n; i++) {
      if (!isIndexed[i] && !seenRun.has(runIds[i])) {
        seenRun.add(runIds[i]);
        unindexedRuns.push(runIds[i]);
      }
    }
    unindexedRuns.sort((a, b) => a - b);
    const runToDense = new Map(unindexedRuns.map((rid, idx) => [rid, idx]));
    for (let i = 0; i < n; i++) {
      if (!isIndexed[i]) groups[i] = runToDense.get(runIds[i]) + nMolecules;
    }
  }

  return groups;
}

/**
 * Apply the `conditions:` dictionary from a condition spec onto `atoms` in place.
 */
export function applyConditions(atoms, conditions, originalResId) {
  const n = atoms.nAtoms;
  const realResId = atoms.res_id.slice();
  atoms.res_id = originalResId.slice();

  const pendingIndexGroups = new Map();

  // 🔴 UPSTREAM'S CHECKS BEFORE ANYTHING IS WRITTEN (`condition_spec._apply_conditions`), in its
  // words. Each was a value this took and turned into something else: `C_CRDGRP: {abc: ...}`
  // became a NaN group and `{0: ...}` claimed the reserved absolute group, a direct `C_INTCRD`
  // wrote a payload upstream derives, and a two-body name has no selection-pair grammar here.
  for (const [annotationName, values] of Object.entries(conditions || {})) {
    const spec = resolveConditionSpec(annotationName);
    if (spec.dtype === 'intcrd') throw new Error('C_INTCRD is a saved payload; edit C_CRDGRP instead.');
    if (spec.dtype === 'twobody') {
      throw new Error(
        `'${annotationName}' resolves to 'basepair', a 2-body condition -- condition_spec has no ` +
          'selection-pair grammar, so only 1-body conditions (per-atom or per-residue) can be set here.'
      );
    }
    if (spec.dtype === 'internal') {
      throw new Error(
        `'${annotationName}' names ${spec.fullName}, which the pipeline writes itself; this port does not ` +
          'take it from a job.'
      );
    }
    if (spec.dtype === 'crdgrp') {
      for (const [value, entries] of Object.entries(values)) {
        if (value !== 'null') {
          if (!/^\d+$/.test(value) || !(Number(value) > 0 && Number(value) <= CRDGRP_MAX_GROUP_ID)) {
            throw new Error(
              `C_CRDGRP requires an integer ID from 1 to ${CRDGRP_MAX_GROUP_ID}; reserved IDs are not editable.`
            );
          }
        }
        if (!Array.isArray(entries) || entries.some((e) => !e || typeof e !== 'object'
          || Object.keys(e).length !== 1 || !('select' in e))) {
          throw new Error('C_CRDGRP entries must contain only select; the value is the group ID.');
        }
      }
    }
  }

  try {
    for (const [annotationName, values] of Object.entries(conditions || {})) {
      const spec = resolveConditionSpec(annotationName);
      const fullName = spec.fullName;

      if (spec.dtype === 'crdgrp') {
        if (!atoms.annotations[fullName]) {
          atoms.annotations[fullName] = getDefaultCoordinateGroups(atoms);
        }
        const groups = atoms.annotations[fullName].slice();
        const defaults = getDefaultCoordinateGroups(atoms);
        for (const [value, entries] of Object.entries(values)) {
          for (const entry of entries) {
            const mask = evalSelection(atoms, entry.select);
            for (let i = 0; i < n; i++) {
              if (mask[i]) {
                groups[i] = value === null || value === 'null' ? defaults[i] : parseInt(value, 10);
              }
            }
          }
        }
        atoms.annotations[fullName] = groups;
        continue;
      }

      for (const [value, entries] of Object.entries(values)) {
        const groupedIndices = new Map();
        for (const entry of entries) {
          const mask = evalSelection(atoms, entry.select);
          const indices = [];
          for (let i = 0; i < n; i++) if (mask[i]) indices.push(i);

          const isUnindexedGroup = fullName === 'condition_index_1_residue' && !coerceBool(value);
          if (!isUnindexedGroup && entry.group !== undefined && entry.group !== null) {
            throw new Error(
              `${annotationName}: \`group:\` is only meaningful for C_IDX (value false), not ${JSON.stringify(value)}.`
            );
          }
          if (isUnindexedGroup) {
            if (entry.group === undefined || entry.group === null) {
              throw new Error(`${annotationName}: value ${value} requires a 'group:' field alongside 'select:'`);
            }
            if (!groupedIndices.has(entry.group)) groupedIndices.set(entry.group, []);
            groupedIndices.get(entry.group).push(...indices);
          } else {
            if (spec.dtype === 'coord') {
              const active = coerceBool(value);
              if (!atoms.annotations['condition_coordinate_1_atom']) {
                atoms.annotations['condition_coordinate_1_atom'] = Array.from({ length: n }, () => [NaN, NaN, NaN]);
              }
              const crd = atoms.annotations['condition_coordinate_1_atom'];
              for (const idx of indices) {
                crd[idx] = active ? atoms.coord[idx].slice() : [NaN, NaN, NaN];
              }
              // Sync coordinate group 0
              const crdGrp = atoms.annotations['condition_CoordinateGroup_1_atom']
                ? atoms.annotations['condition_CoordinateGroup_1_atom'].slice()
                : getDefaultCoordinateGroups(atoms);
              const defGrp = getDefaultCoordinateGroups(atoms);
              for (const idx of indices) {
                if (active || crdGrp[idx] <= 0) crdGrp[idx] = defGrp[idx];
              }
              atoms.annotations['condition_CoordinateGroup_1_atom'] = crdGrp;
            } else if (spec.dtype === 'seq') {
              if (!atoms.annotations[fullName]) {
                atoms.annotations[fullName] = new Array(n).fill('<M>');
              }
              const seq = atoms.annotations[fullName];
              // 🔴 TRUE OR FALSE AND NOTHING ELSE: a sequence condition copies the residue's own
              // identity (upstream's `ground_truth_value`) or masks it. This also took a literal
              // residue name - `C_SEQ: {ALA: ...}` - which upstream refuses as not a boolean.
              const boolVal = coerceBool(value);
              for (const idx of indices) seq[idx] = boolVal ? atoms.res_name[idx] : '<M>';
            } else if (spec.dtype === 'bool') {
              if (!atoms.annotations[fullName]) {
                atoms.annotations[fullName] = new Array(n).fill(spec.default);
              }
              const b = coerceBool(value);
              for (const idx of indices) atoms.annotations[fullName][idx] = b;
            } else if (spec.dtype === 'enum') {
              if (!atoms.annotations[fullName]) {
                atoms.annotations[fullName] = new Array(n).fill(spec.default);
              }
              // `coerce_enum_value`: a member's name, or an integer that IS a member's value.
              const numeric = typeof value === 'number' || /^-?\d+$/.test(String(value));
              const enumVal = numeric ? parseInt(value, 10) : spec.enum[String(value)];
              if (enumVal === undefined || (numeric && !Object.values(spec.enum).includes(enumVal))) {
                throw new Error(`Invalid enum value ${value} for ${annotationName}`);
              }
              for (const idx of indices) atoms.annotations[fullName][idx] = enumVal;
            } else if (spec.dtype === 'centroid') {
              if (!atoms.annotations[fullName]) {
                atoms.annotations[fullName] = Array.from({ length: n }, () => [NaN, NaN, NaN]);
              }
              // 🔴 EXACTLY THREE (`parse_centroid_value`): this stored `slice(0, 3)` of however many
              // it was given, so two numbers became a two-element "point" and four were truncated.
              const parts = Array.isArray(value) ? value.map(Number) : String(value).split(',').map(Number);
              if (parts.length !== 3 || parts.some((v) => !Number.isFinite(v))) {
                throw new Error(
                  `ChainCentroid value must have exactly 3 components (x, y, z), got ${JSON.stringify(value)}.`
                );
              }
              for (const idx of indices) atoms.annotations[fullName][idx] = parts.slice();
            } else if (spec.dtype === 'cath') {
              if (!atoms.annotations[fullName]) {
                atoms.annotations[fullName] = new Array(n).fill('x.x.x.x');
              }
              // `set_cath_condition`: a C.A.T label from the model's own vocabulary, stored with a
              // trailing `.x`. This appended `.x` to anything, so `3.40.50.20` became a five-field
              // code and a typo became a label nothing maps.
              const cathStr = normalizeCatLabel(value);
              for (const idx of indices) atoms.annotations[fullName][idx] = cathStr;
            } else if (spec.dtype === 'float') {
              if (!atoms.annotations[fullName]) {
                atoms.annotations[fullName] = new Array(n).fill(spec.default);
              }
              // A number, which a boolean is not and NaN is not either - `Number('abc')` wrote NaN,
              // which for a sequence probability means "unspecified", silently.
              const fVal = Number(value);
              if (typeof value === 'boolean' || /^\s*(true|false)?\s*$/i.test(String(value)) || !Number.isFinite(fVal)) {
                throw new Error(`${annotationName} writes a number, so it cannot take ${JSON.stringify(value)}.`);
              }
              for (const idx of indices) atoms.annotations[fullName][idx] = fVal;
            } else {
              if (typeof value === 'boolean' || /^(true|false)$/i.test(String(value))) {
                throw new Error(`${spec.fullName} writes a value, so it cannot take ${JSON.stringify(value)}.`);
              }
              if (!atoms.annotations[fullName]) {
                atoms.annotations[fullName] = new Array(n).fill(spec.default);
              }
              for (const idx of indices) atoms.annotations[fullName][idx] = String(value);
            }
          }
        }

        if (groupedIndices.size > 0) {
          if (!atoms.annotations['condition_index_1_residue']) {
            atoms.annotations['condition_index_1_residue'] = new Array(n).fill(true);
          }
          for (const [label, idxs] of groupedIndices.entries()) {
            for (const idx of idxs) atoms.annotations['condition_index_1_residue'][idx] = false;
            if (!pendingIndexGroups.has(label)) pendingIndexGroups.set(label, []);
            pendingIndexGroups.get(label).push(...idxs);
          }
        }
      }
    }

    if (pendingIndexGroups.size > 0) {
      const groupIds = getDefaultIndexGroups(atoms);
      const isIndexed = atoms.annotations['condition_index_1_residue'];
      let nextId = groupIds.reduce((m, v) => Math.max(m, v), -1) + 1;
      for (const idxs of pendingIndexGroups.values()) {
        const unindexedIdxs = idxs.filter((idx) => !isIndexed[idx]);
        if (unindexedIdxs.length === 0) continue;
        for (const idx of unindexedIdxs) groupIds[idx] = nextId;
        nextId++;
      }
      atoms.annotations['condition_index-group_1_token'] = groupIds;
    }
  } finally {
    atoms.res_id = realResId;
  }

  computeIdAndEntityAnnotations(atoms);
  return atoms;
}

/**
 * Build an annotated AtomArray from a condition-spec object or YAML/JSON string.
 *
 * @param {object|string} specOrText - Condition spec object or YAML/JSON string.
 * @param {object} [opts]
 * @param {string|Uint8Array|object} [opts.inputCif] - Pre-loaded CIF content (or AtomArray) for `spec.input`.
 * @param {function(string): (string|Uint8Array)} [opts.loadInput] - Callback to load relative/absolute input path.
 * @param {string} [opts.exampleId] - Example key to select if `specOrText` has top-level example mappings.
 */
/**
 * `load_condition_spec_inputs` for one file: a condition spec's `input:` as upstream loads it.
 * With `asAnnotatedCif` the file is read as written and its conditions kept (`ANNOTATED_CIF`);
 * without, it is a raw structure and its conditions are discarded (`RCSB_WITH_METALS`).
 * tools/check-rfd4p-load.py holds this to atomworks atom for atom.
 *
 * @param {string|Uint8Array} input mmCIF text or bytes (plain or Zstandard)
 * @param {{asAnnotatedCif?: boolean, ccdBlocks?: Map<string, object>}} [opts] `ccdBlocks`: fetched
 *   dictionary entries (ccd.js) for components outside the generated templates
 */
/**
 * Whether a structure file is an ANNOTATED CIF - the output of a condition spec, carrying
 * `condition_*` / `annotation_*` / `mask_*` columns - rather than a raw structure. Upstream is told
 * which (`input_as_annotated_cif`); a page handed a file decides from what it carries.
 */
export function isAnnotatedCif(input) {
  const columns = Object.keys(readStructureBlock(input).block.atom_site ?? {});
  return columns.some((c) => /^(?:condition_|annotation_|mask_)/.test(c));
}

export function loadConditionSpecInput(input, { asAnnotatedCif = false, ccdBlocks } = {}) {
  let atoms;
  if (asAnnotatedCif) {
    atoms = parseCifToAtomArray(input, { ccdBlocks });
  } else {
    const block = prepareFreshBlock(readStructureBlock(input).block, { ccdBlocks });
    atoms = parseCifToAtomArray(null, { block, fixArginines: true, ccdBlocks });
    atoms.annotations = {};
  }
  computeIdAndEntityAnnotations(atoms);
  return atoms;
}

/** The fields one condition-spec example may carry (`_CONDITION_SPEC_EXAMPLE_KEYS`). */
export const CONDITION_SPEC_EXAMPLE_KEYS = Object.freeze(['input', 'input_as_annotated_cif', 'contig', 'conditions']);

/**
 * `split_global_args`: a spec FILE - examples by name, with `global_args` folded into each (an
 * example's own value wins) - as `{name: example}`. A flat object (this port's own job shape, with
 * an `input` or a `contig` at the top) is one example called `job`.
 */
export function conditionSpecExamples(spec) {
  if (spec === null || typeof spec !== 'object' || Array.isArray(spec)) {
    throw new Error('A condition spec must be a mapping.');
  }
  if ('input' in spec || 'contig' in spec) return { job: spec };
  const globalArgs = spec.global_args ?? {};
  if (globalArgs === null || typeof globalArgs !== 'object' || Array.isArray(globalArgs)) {
    throw new Error(`condition_spec \`global_args\` must be a mapping, got ${Array.isArray(globalArgs) ? 'list' : typeof globalArgs}.`);
  }
  const invalid = Object.entries(spec)
    .filter(([k, v]) => k !== 'global_args' && (v === null || typeof v !== 'object' || Array.isArray(v)))
    .map(([k, v]) => `${k} (${Array.isArray(v) ? 'list' : v === null ? 'NoneType' : typeof v})`);
  if (invalid.length) throw new Error(`condition_spec examples must be mappings; invalid entries: ${invalid.join(', ')}.`);
  const out = {};
  for (const [k, v] of Object.entries(spec)) if (k !== 'global_args') out[k] = { ...globalArgs, ...v };
  if (!Object.keys(out).length) throw new Error('Condition spec contains no examples.');
  return out;
}

/**
 * `separate_coordinate_groups`: each input's internal-coordinate groups renumbered into one
 * namespace, so two files' group 1 do not become one rigid group. In place.
 */
function separateCoordinateGroups(arrays) {
  let next = 1;
  for (const a of arrays) {
    const groups = (a.annotations?.condition_CoordinateGroup_1_atom ?? getDefaultCoordinateGroups(a)).slice();
    const old = groups.slice();
    const intcrd = a.annotations?.condition_InternalCoordinate_1_atom;
    if (intcrd && intcrd.some((c, i) => !(old[i] > 0) && c.some(Number.isFinite))) {
      throw new Error('Internal coordinate targets exist outside internal coordinate groups.');
    }
    for (const gid of [...new Set(old.filter((g) => g > 0))].sort((x, y) => x - y)) {
      for (let i = 0; i < old.length; i += 1) if (old[i] === gid) groups[i] = next;
      next += 1;
    }
    if (a.annotations?.condition_CoordinateGroup_1_atom) a.annotations.condition_CoordinateGroup_1_atom = groups;
  }
  return arrays;
}

/**
 * Build an annotated AtomArray from a condition-spec object or YAML/JSON string.
 *
 * @param {object|string} specOrText - Condition spec object or YAML/JSON string.
 * @param {object} [opts]
 * @param {string|Uint8Array|object} [opts.inputCif] - Pre-loaded CIF content (or AtomArray) for `spec.input`.
 * @param {function(string): (string|Uint8Array)} [opts.loadInput] - Callback to load relative/absolute input path.
 * @param {string} [opts.exampleId] - The example to build, when the spec names more than one.
 * @param {Map<string, object>} [opts.ccdBlocks] - Fetched dictionary entries (ccd.js) for a raw input's components.
 */
export function buildAtomArrayFromConditionSpec(specOrText, opts = {}) {
  const parsed = typeof specOrText === 'string' ? parseConditionSpecYaml(specOrText) : specOrText;
  const flat = parsed !== null && typeof parsed === 'object' && ('input' in parsed || 'contig' in parsed);
  const examples = conditionSpecExamples(parsed);
  const names = Object.keys(examples);
  // 🔴 A FILE OF SEVERAL EXAMPLES NAMES WHICH ONE. This built the first and said nothing, so a
  // spec upstream runs as five designs ran as one here.
  if (opts.exampleId !== undefined && !examples[opts.exampleId]) {
    throw new Error(`no example '${opts.exampleId}' in this condition spec; it has ${names.join(', ')}`);
  }
  if (opts.exampleId === undefined && names.length > 1) {
    throw new Error(`this condition spec has ${names.length} examples (${names.join(', ')}); name one`);
  }
  const exampleId = opts.exampleId ?? names[0];
  const rawSpec = examples[exampleId];
  // Upstream's own vocabulary, for a spec in upstream's shape: a typo'd field applies nothing.
  if (!flat) {
    const unknown = Object.keys(rawSpec).filter((k) => !CONDITION_SPEC_EXAMPLE_KEYS.includes(k)).sort();
    if (unknown.length) {
      throw new Error(`${exampleId}: unrecognized condition_spec field(s) ${JSON.stringify(unknown)}. `
        + `Recognized fields: ${JSON.stringify([...CONDITION_SPEC_EXAMPLE_KEYS].sort())}.`);
    }
  }

  const asAnnotatedCif = Boolean(rawSpec.input_as_annotated_cif);
  const load = (input) => loadConditionSpecInput(input, { asAnnotatedCif, ccdBlocks: opts.ccdBlocks });
  let atomArray;

  if (opts.inputAtomArray) {
    atomArray = opts.inputAtomArray;
  } else if (opts.inputCif) {
    atomArray = load(opts.inputCif);
  } else if (rawSpec.input) {
    const inputs = Array.isArray(rawSpec.input) ? rawSpec.input : [rawSpec.input];
    const loadedArrays = inputs.map((inp) => {
      if (typeof inp === 'object' && inp !== null && inp.nAtoms !== undefined) return inp;
      if (typeof inp === 'string' && (inp.includes('_atom_site.') || inp.startsWith('data_'))) return load(inp);
      if (opts.loadInput) return load(opts.loadInput(inp));
      throw new Error(
        `Cannot load condition_spec input path '${inp}' without opts.inputCif or opts.loadInput(path).`
      );
    });
    if (loadedArrays.length === 1) {
      atomArray = loadedArrays[0];
    } else {
      // `load_condition_spec_inputs` over several files: one chain id may come from one file only.
      const seen = new Map();
      loadedArrays.forEach((a, k) => {
        for (const c of [...new Set(a.chain_id)].sort()) {
          if (seen.has(c)) {
            throw new Error(`chain_id '${c}' appears in both '${seen.get(c)}' and '${typeof inputs[k] === 'string' ? inputs[k] : `input ${k}`}' -- `
              + 'condition_spec does not auto-rename colliding chain ids across input files; rename one of them so '
              + '`contig` selections stay unambiguous.');
          }
          seen.set(c, typeof inputs[k] === 'string' ? inputs[k] : `input ${k}`);
        }
      });
      atomArray = concatAtomArrays(separateCoordinateGroups(loadedArrays));
      computeIdAndEntityAnnotations(atomArray);
    }
  } else {
    atomArray = createEmptyAtomArray();
  }

  if (!asAnnotatedCif) {
    atomArray.annotations = {};
  }

  let originalResId;
  if (rawSpec.contig) {
    ({ atomArray, originalResId } = applyContig(atomArray, rawSpec.contig));
  } else {
    originalResId = atomArray.res_id.slice();
  }

  if (rawSpec.conditions) {
    applyConditions(atomArray, rawSpec.conditions, originalResId);
  }

  computeIdAndEntityAnnotations(atomArray);
  return atomArray;
}
