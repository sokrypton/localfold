/**
 * Condition-spec (YAML/JSON) parser, selection query evaluator (`sel(...)` + boolean queries),
 * `applyContig`, `applyConditions`, and `buildAtomArrayFromConditionSpec`.
 *
 * Ports `rfproteina.inference_utils.contig_spec`, `rfproteina.inference_utils.condition_spec`,
 * `rfproteina.utils.mask_spec`, and `atomworks.io.utils.query`.
 */

import {
  BOND_TYPE,
  CHAIN_TYPE,
  SCALAR_1BODY_ANNOTATIONS,
  NONSCALAR_1BODY_ANNOTATIONS,
  createEmptyAtomArray,
  parseCifToAtomArray,
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
  SSEnum: { NONE: 0, HELIX: 1, STRAND: 2, LOOP: 3 },
  BasePairEnum: { UNSPECIFIED: 0, PAIR: 1, LOOP: 2 },
};

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

function expandPathField(rawField, isResId = false) {
  const v = rawField ? rawField.trim() : '*';
  if (!v || v === '*') return ['*'];
  if (v.startsWith('[') && v.endsWith(']')) {
    const items = v.slice(1, -1).split(',').map((s) => s.trim()).filter(Boolean);
    const out = [];
    for (const item of items) {
      if (item === '*') {
        out.push('*');
      } else if (isResId) {
        const m = item.match(/^(-?\d+)-(-?\d+)$/);
        if (m) {
          const s = parseInt(m[1], 10);
          const e = parseInt(m[2], 10);
          const step = s <= e ? 1 : -1;
          for (let k = s; step > 0 ? k <= e : k >= e; k += step) out.push(k);
        } else {
          out.push(parseInt(item, 10));
        }
      } else {
        out.push(item);
      }
    }
    return out;
  }
  return [isResId ? parseInt(v, 10) : v];
}

/**
 * Evaluate a `sel('...')` path-selection string against `atoms`.
 * Returns a boolean array of length `atoms.nAtoms`.
 */
export function evalPathSelection(atoms, selectionStr) {
  const n = atoms.nAtoms;
  const result = new Array(n).fill(false);
  const tokens = splitTopLevel(selectionStr, ',').map((t) => t.trim()).filter(Boolean);

  for (const token of tokens) {
    const parts = splitTopLevel(token, '/');
    const chains = expandPathField(parts[0] ?? '*', false);
    const resNames = expandPathField(parts[1] ?? '*', false);
    const resIds = expandPathField(parts[2] ?? '*', true);
    const atomNames = expandPathField(parts[3] ?? '*', false);
    const transIds = expandPathField(parts[4] ?? '*', false);

    const matchChain = chains.includes('*') ? null : new Set(chains);
    const matchResName = resNames.includes('*') ? null : new Set(resNames);
    const matchResId = resIds.includes('*') ? null : new Set(resIds);
    const matchAtomName = atomNames.includes('*') ? null : new Set(atomNames);
    const matchTransId = transIds.includes('*') ? null : new Set(transIds.map(String));

    for (let i = 0; i < n; i++) {
      if (result[i]) continue;
      if (matchChain && !matchChain.has(atoms.chain_id[i])) continue;
      if (matchResName && !matchResName.has(atoms.res_name[i])) continue;
      if (matchResId && !matchResId.has(atoms.res_id[i])) continue;
      if (matchAtomName && !matchAtomName.has(atoms.atom_name[i])) continue;
      if (matchTransId && !matchTransId.has(String(atoms.transformation_id?.[i] ?? '1'))) continue;
      result[i] = true;
    }
  }
  return result;
}

function getAtomAttribute(atoms, name) {
  if (name in atoms && Array.isArray(atoms[name])) return atoms[name];
  if (atoms.annotations && name in atoms.annotations) return atoms.annotations[name];
  const n = atoms.nAtoms;
  if (name === 'x') return atoms.coord.map((c) => c[0]);
  if (name === 'y') return atoms.coord.map((c) => c[1]);
  if (name === 'z') return atoms.coord.map((c) => c[2]);
  if (name === 'all') return new Array(n).fill(true);
  if (name === 'is_ligand') return atoms.is_polymer.map((p) => !p);
  if (name === 'is_protein') return atoms.chain_type.map((ct) => ct === CHAIN_TYPE.POLYPEPTIDE_L || ct === CHAIN_TYPE.POLYPEPTIDE_D);
  if (name === 'is_dna') return atoms.chain_type.map((ct) => ct === CHAIN_TYPE.DNA);
  if (name === 'is_rna') return atoms.chain_type.map((ct) => ct === CHAIN_TYPE.RNA);
  if (name === 'is_nucleic_acid') return atoms.chain_type.map((ct) => ct === CHAIN_TYPE.DNA || ct === CHAIN_TYPE.RNA);
  if (atoms.res_name && atoms.res_name.includes(name)) return atoms.res_name.map((r) => r === name);
  if (atoms.chain_id && atoms.chain_id.includes(name)) return atoms.chain_id.map((c) => c === name);
  throw new Error(`Attribute '${name}' is not defined on AtomArray`);
}

// Evaluate a query expression string against `atoms`, returning a boolean array of length `atoms.nAtoms`.
export function evalSelection(atoms, queryStr) {
  const n = atoms.nAtoms;
  // Tokenize query expression
  const tokens = [];
  let i = 0;
  while (i < queryStr.length) {
    const ch = queryStr[i];
    if (/\s/.test(ch)) { i++; continue; }
    if (ch === "'" || ch === '"') {
      const q = ch;
      let s = '';
      i++;
      while (i < queryStr.length && queryStr[i] !== q) {
        if (queryStr[i] === '\\' && i + 1 < queryStr.length) {
          i++;
        }
        s += queryStr[i++];
      }
      i++; // closing quote
      tokens.push({ type: 'str', value: s });
      continue;
    }
    const two = queryStr.slice(i, i + 2);
    if (['==', '!=', '<=', '>='].includes(two)) {
      tokens.push({ type: 'op', value: two });
      i += 2;
      continue;
    }
    if (['(', ')', '[', ']', ',', '&', '|', '~', '<', '>'].includes(ch)) {
      tokens.push({ type: ch, value: ch });
      i++;
      continue;
    }
    if (/[-0-9]/.test(ch) && (ch !== '-' || /[0-9]/.test(queryStr[i + 1] || ''))) {
      let num = ch;
      i++;
      while (i < queryStr.length && /[0-9.eE+-]/.test(queryStr[i])) num += queryStr[i++];
      tokens.push({ type: 'num', value: Number(num) });
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      let id = ch;
      i++;
      while (i < queryStr.length && /[A-Za-z0-9_-]/.test(queryStr[i])) id += queryStr[i++];
      if (id === 'not' && tokens.length > 0 && tokens[tokens.length - 1].value === 'not') {
        // handled normally
      }
      tokens.push({ type: 'id', value: id });
      continue;
    }
    throw new Error(`Unexpected character '${ch}' in query: ${queryStr}`);
  }

  let pos = 0;
  const peek = () => tokens[pos] || null;
  const consume = () => tokens[pos++];

  const toBoolArray = (v) => {
    if (Array.isArray(v)) return v.map(Boolean);
    return new Array(n).fill(Boolean(v));
  };

  function parseOr() {
    let left = parseAnd();
    while (peek() && ((peek().type === 'id' && peek().value === 'or') || peek().type === '|')) {
      consume();
      const right = parseAnd();
      const l = toBoolArray(left);
      const r = toBoolArray(right);
      left = l.map((x, idx) => x || r[idx]);
    }
    return left;
  }

  function parseAnd() {
    let left = parseNot();
    while (peek() && ((peek().type === 'id' && peek().value === 'and') || peek().type === '&')) {
      consume();
      const right = parseNot();
      const l = toBoolArray(left);
      const r = toBoolArray(right);
      left = l.map((x, idx) => x && r[idx]);
    }
    return left;
  }

  function parseNot() {
    if (peek() && ((peek().type === 'id' && peek().value === 'not') || peek().type === '~')) {
      consume();
      const operand = toBoolArray(parseNot());
      return operand.map((x) => !x);
    }
    return parseCompare();
  }

  function parseCompare() {
    const left = parsePrimary();
    const t = peek();
    if (!t) return left;
    if (t.type === 'op' || (t.type === 'id' && (t.value === 'in' || t.value === 'not')) || t.type === '<' || t.type === '>') {
      let op = consume().value;
      if (op === 'not') {
        const next = consume();
        if (!next || next.value !== 'in') throw new Error(`Expected 'in' after 'not' in query: ${queryStr}`);
        op = 'not in';
      }
      const right = parsePrimary();
      const evalPair = (a, b) => {
        if (op === '==') return a === b;
        if (op === '!=') return a !== b;
        if (op === '<') return a < b;
        if (op === '<=') return a <= b;
        if (op === '>') return a > b;
        if (op === '>=') return a >= b;
        if (op === 'in') return Array.isArray(b) && b.includes(a);
        if (op === 'not in') return Array.isArray(b) && !b.includes(a);
        throw new Error(`Unsupported operator ${op}`);
      };
      if (Array.isArray(left) && (op === 'in' || op === 'not in' || !Array.isArray(right))) {
        return left.map((x) => evalPair(x, right));
      }
      if (Array.isArray(left) && Array.isArray(right)) {
        return left.map((x, idx) => evalPair(x, right[idx]));
      }
      return evalPair(left, right);
    }
    return left;
  }

  function parsePrimary() {
    const t = consume();
    if (!t) throw new Error(`Unexpected end of query: ${queryStr}`);
    if (t.type === '(') {
      const val = parseOr();
      const close = consume();
      if (!close || close.type !== ')') throw new Error(`Missing ')' in query: ${queryStr}`);
      return val;
    }
    if (t.type === '[') {
      const list = [];
      while (peek() && peek().type !== ']') {
        list.push(parseOr());
        if (peek() && peek().type === ',') consume();
      }
      consume(); // ']'
      return list;
    }
    if (t.type === 'str' || t.type === 'num') return t.value;
    if (t.type === 'id') {
      if (t.value === 'True' || t.value === 'true') return true;
      if (t.value === 'False' || t.value === 'false') return false;
      if (peek() && peek().type === '(') {
        consume(); // '('
        const args = [];
        while (peek() && peek().type !== ')') {
          args.push(parseOr());
          if (peek() && peek().type === ',') consume();
        }
        consume(); // ')'
        if (t.value === 'sel') {
          return evalPathSelection(atoms, String(args[0]));
        }
        if (t.value === 'has_nan_coord') {
          return atoms.coord.map((c) => !Number.isFinite(c[0]) || !Number.isFinite(c[1]) || !Number.isFinite(c[2]));
        }
        if (t.value === 'has_bonds') {
          const bonded = new Set();
          for (const [u, v] of atoms.bonds || []) {
            bonded.add(u);
            bonded.add(v);
          }
          return Array.from({ length: n }, (_, idx) => bonded.has(idx));
        }
        if (t.value === 'sa_mask') {
          const cond = resolveConditionSpec(String(args[0]));
          const arr = atoms.annotations[cond.fullName];
          if (!arr) return new Array(n).fill(false);
          if (cond.dtype === 'coord' || cond.dtype === 'intcrd' || cond.dtype === 'centroid') {
            return arr.map((c) => Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2]));
          }
          if (cond.dtype === 'seq') return arr.map((v) => v !== '<M>');
          if (cond.dtype === 'bool') return arr.map(Boolean);
          if (cond.dtype === 'enum') return arr.map((v) => v !== 0);
          return arr.map((v) => v !== cond.default);
        }
        throw new Error(`Unsupported query function: ${t.value}`);
      }
      return getAtomAttribute(atoms, t.value);
    }
    throw new Error(`Unexpected token ${JSON.stringify(t)} in query: ${queryStr}`);
  }

  return toBoolArray(parseOr());
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
  const runs = [[]];
  for (const tok of tokens) {
    if (tok.kind === 'chainbreak') runs.push([]);
    else runs[runs.length - 1].push(tok);
  }
  const nonEmptyRuns = runs.filter((r) => r.length > 0);
  if (nonEmptyRuns.length === 0) {
    throw new Error('Contig string produced no chain runs.');
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
      const idx = atomArrayIn.chain_id.indexOf(preserved.chainId);
      if (idx >= 0) anchorType = atomArrayIn.chain_type[idx];
    }
    const requested = new Set(
      run.tokens.filter((t) => t.kind === 'expandable' && t.chainType !== null).map((t) => t.chainType)
    );
    if (requested.size > 1) throw new Error('Chain run requests conflicting chain types.');
    if (requested.size === 0) return anchorType !== null ? anchorType : CHAIN_TYPE.POLYPEPTIDE_L;
    return Array.from(requested)[0];
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
              let isBool = false;
              let boolVal = false;
              try {
                boolVal = coerceBool(value);
                isBool = true;
              } catch {
                isBool = false;
              }
              for (const idx of indices) {
                seq[idx] = isBool ? (boolVal ? atoms.res_name[idx] : '<M>') : String(value);
              }
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
              const enumVal =
                typeof value === 'number' || /^-?\d+$/.test(String(value))
                  ? parseInt(value, 10)
                  : spec.enum[String(value)];
              if (enumVal === undefined) {
                throw new Error(`Invalid enum value ${value} for ${annotationName}`);
              }
              for (const idx of indices) atoms.annotations[fullName][idx] = enumVal;
            } else if (spec.dtype === 'centroid') {
              if (!atoms.annotations[fullName]) {
                atoms.annotations[fullName] = Array.from({ length: n }, () => [NaN, NaN, NaN]);
              }
              const parts = Array.isArray(value) ? value.map(Number) : String(value).split(',').map(Number);
              for (const idx of indices) atoms.annotations[fullName][idx] = parts.slice(0, 3);
            } else if (spec.dtype === 'cath') {
              if (!atoms.annotations[fullName]) {
                atoms.annotations[fullName] = new Array(n).fill('x.x.x.x');
              }
              let cathStr = String(value);
              if (!cathStr.endsWith('.x')) cathStr += '.x';
              for (const idx of indices) atoms.annotations[fullName][idx] = cathStr;
            } else if (spec.dtype === 'float') {
              if (!atoms.annotations[fullName]) {
                atoms.annotations[fullName] = new Array(n).fill(spec.default);
              }
              const fVal = Number(value);
              for (const idx of indices) atoms.annotations[fullName][idx] = fVal;
            } else {
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
export function buildAtomArrayFromConditionSpec(specOrText, opts = {}) {
  let rawSpec = typeof specOrText === 'string' ? parseConditionSpecYaml(specOrText) : specOrText;

  // Check if rawSpec is a multi-example mapping (like `{ binder: { input: ..., contig: ..., conditions: ... } }`)
  if (!('input' in rawSpec) && !('contig' in rawSpec)) {
    const globalArgs = rawSpec.global_args || {};
    const exampleKeys = Object.keys(rawSpec).filter((k) => k !== 'global_args');
    if (exampleKeys.length === 0) {
      throw new Error('Condition spec contains no examples.');
    }
    const chosenKey = opts.exampleId && rawSpec[opts.exampleId] ? opts.exampleId : exampleKeys[0];
    rawSpec = { ...globalArgs, ...rawSpec[chosenKey] };
  }

  const asAnnotatedCif = Boolean(rawSpec.input_as_annotated_cif);
  let atomArray;

  if (opts.inputAtomArray) {
    atomArray = opts.inputAtomArray;
  } else if (opts.inputCif) {
    atomArray = parseCifToAtomArray(opts.inputCif, { fixArginines: !asAnnotatedCif });
  } else if (rawSpec.input) {
    const inputs = Array.isArray(rawSpec.input) ? rawSpec.input : [rawSpec.input];
    const loadedArrays = inputs.map((inp) => {
      if (typeof inp === 'object' && inp !== null && inp.nAtoms !== undefined) return inp;
      if (typeof inp === 'string' && (inp.includes('_atom_site.') || inp.startsWith('data_'))) {
        return parseCifToAtomArray(inp, { fixArginines: !asAnnotatedCif });
      }
      if (opts.loadInput) {
        const bytesOrText = opts.loadInput(inp);
        return parseCifToAtomArray(bytesOrText, { fixArginines: !asAnnotatedCif });
      }
      throw new Error(
        `Cannot load condition_spec input path '${inp}' without opts.inputCif or opts.loadInput(path).`
      );
    });
    if (!asAnnotatedCif) {
      for (const a of loadedArrays) a.annotations = {};
    }
    atomArray = loadedArrays.length === 1 ? loadedArrays[0] : concatAtomArrays(loadedArrays);
    computeIdAndEntityAnnotations(atomArray);
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
