/**
 * RFD4-Proteina Condition definitions, enums, CATH fold mappings, and
 * `sel('<chain>/<resname>/<resid>/<atom>')` selection expression evaluator.
 *
 * Matches `rfproteina/conditions.py`, `rfproteina/utils/mask_spec.py`, and
 * `rfproteina/inference_utils/condition_spec.py`.
 */

import { STANDARD_POLYMER_RESIDUES, UNIFIED_ATOM37_ENCODING } from "./encoding.js";

export const RasaEnum = Object.freeze({
  UNSPECIFIED: 0,
  BURIED: 1,
  INTERMEDIATE: 2,
  PARTIALLY_BURIED: 2,
  EXPOSED: 3,
});

export const HBondEnum = Object.freeze({
  NONE: 0,
  DONOR: 1,
  ACCEPTOR: 2,
  BOTH: 3,
});

export const PiCationEnum = Object.freeze({
  NONE: 0,
  AROMATIC: 1,
  PI: 1,
  CATION: 2,
  BOTH: 3,
});

export const SaltBridgeEnum = Object.freeze({
  NONE: 0,
  POSITIVE: 1,
  CATION: 1,
  NEGATIVE: 2,
  ANION: 2,
  BOTH: 1,
});

export const MetalCoordEnum = Object.freeze({
  NONE: 0,
  METAL: 1,
  COORDINATING: 2,
  LIGAND: 2,
});

export const SSEnum = Object.freeze({
  NONE: 0,
  UNSPECIFIED: 0,
  HELIX: 1,
  H: 1,
  SHEET: 2,
  STRAND: 2,
  E: 2,
  LOOP: 3,
  COIL: 3,
  C: 3,
  L: 3,
});

export const BasePairEnum = Object.freeze({
  UNSPECIFIED: 0,
  PAIR: 1,
  PAIRED: 1,
  LOOP: 2,
});

export const CoordGroupConst = Object.freeze({
  UNKNOWN_COORDS: -1,
  KNOWN_ABSOLUTE_COORDS: 0,
  MAX_GROUP_ID: 32767,
});

export const C_CRD = Object.freeze({
  alias: "C_CRD",
  full_name: "condition_coordinate_1_atom",
});

export const C_INTCRD = Object.freeze({
  alias: "C_INTCRD",
  full_name: "condition_InternalCoordinate_1_atom",
});

export const C_CRDGRP = Object.freeze({
  alias: "C_CRDGRP",
  full_name: "condition_CoordinateGroup_1_atom",
  ...CoordGroupConst,
});

export const C_IDX = Object.freeze({
  alias: "C_IDX",
  full_name: "condition_index_1_token",
});

export const C_IDXGRP = Object.freeze({
  alias: "C_IDXGRP",
  full_name: "condition_index-group_1_token",
});

export const C_SEQ = Object.freeze({
  alias: "C_SEQ",
  full_name: "condition_sequence_1_residue",
});

export const C_HOT = Object.freeze({
  alias: "C_HOT",
  full_name: "condition_hotspot_1_atom",
});

export const C_RAS = Object.freeze({
  alias: "C_RAS",
  full_name: "condition_rasa-binned_1_atom",
  enum: RasaEnum,
});

export const C_SEC = Object.freeze({
  alias: "C_SEC",
  full_name: "condition_secondary-structure_1_residue",
  enum: SSEnum,
});

export const C_HB = Object.freeze({
  alias: "C_HB",
  full_name: "condition_hbond_1_atom",
  enum: HBondEnum,
});

export const C_PISTACK = Object.freeze({
  alias: "C_PISTACK",
  full_name: "condition_pistack_1_atom",
});

export const C_PICATION = Object.freeze({
  alias: "C_PICATION",
  full_name: "condition_pication_1_atom",
  enum: PiCationEnum,
});

export const C_SALT = Object.freeze({
  alias: "C_SALT",
  full_name: "condition_saltbridge_1_atom",
  enum: SaltBridgeEnum,
});

export const C_HPHOB = Object.freeze({
  alias: "C_HPHOB",
  full_name: "condition_hydrophobic_1_atom",
});

export const C_MCOORD = Object.freeze({
  alias: "C_MCOORD",
  full_name: "condition_metal-coord_1_atom",
  enum: MetalCoordEnum,
});

export const C_CATH = Object.freeze({
  alias: "C_CATH",
  full_name: "condition_cath-label_1_residue",
  NULL_VALUE: "x.x.x.x",
  UNCONDITIONED_VALUE: -1,
});

export const C_CENTROID = Object.freeze({
  alias: "C_CENTROID",
  full_name: "condition_ChainCentroid_1_atom",
});

export const C_BP = Object.freeze({
  alias: "C_BP",
  full_name: "condition_basepair_2_residue",
  enum: BasePairEnum,
});

export const C_NA_SS_STATE = Object.freeze({
  alias: "C_NA_SS_STATE",
  full_name: "condition_na-ss-state_1_residue",
  enum: BasePairEnum,
});

export const SEQUENCE_PROBABILITY_CONDITIONS = Object.freeze(
  STANDARD_POLYMER_RESIDUES.map((tokenName, channelIdx) =>
    Object.freeze({
      alias: `C_SEQ_PROB_${tokenName}`,
      full_name: `condition_sequence-probability-${tokenName}_1_residue`,
      token_name: tokenName,
      token_index: UNIFIED_ATOM37_ENCODING.token_to_idx[tokenName],
      channel: channelIdx,
    })
  )
);

export const SEQUENCE_PROBABILITY_BY_TOKEN = Object.freeze(
  Object.fromEntries(SEQUENCE_PROBABILITY_CONDITIONS.map((c) => [c.token_name, c]))
);

/**
 * Helper to look up a condition tensor in `batch` by either `alias` or `full_name`.
 */
export function getCondTensor(batch, condObj) {
  if (!batch) return undefined;
  if (condObj.alias in batch) return batch[condObj.alias];
  if (condObj.full_name in batch) return batch[condObj.full_name];
  return undefined;
}

// 🔴 THE CATH LABEL -> EMBEDDING-ROW MAPPING IS UPSTREAM'S, GENERATED, NOT A HAND-PICKED LIST.
// These were 5 classes, 32 architectures and 12 topologies typed in as "common presets", and 26
// of the 32 architecture rows and all 12 topology rows disagreed with the pickle the checkpoint
// was trained with - `3.40.50` went in as topology 890 where the model reads 1056. The model
// embeds whatever index it is handed, so every CATH condition the page ever applied was a
// DIFFERENT fold label. See tools/gen-rfd4p-cath-mapping.py.
export { CATH_CLASS_MAP, CATH_ARCHITECTURE_MAP, CATH_TOPOLOGY_MAP } from './cath-mapping.js';
import {
  CATH_CLASS_MAP as CATH_CLASS_MAP_,
  CATH_ARCHITECTURE_MAP as CATH_ARCHITECTURE_MAP_,
  CATH_TOPOLOGY_MAP as CATH_TOPOLOGY_MAP_,
} from './cath-mapping.js';

/**
 * A CATH label's code at one level, as upstream's `extract_cath_code_by_level` cuts it: a
 * right-split, so a stored `C.A.T.x` gives `C.A.T`, `C.A` and `C`.
 */
function cathCodeByLevel(code, level) {
  const k = { H: 0, T: 1, A: 2, C: 3 }[level];
  const parts = String(code).split('.');
  const cut = Math.min(k, parts.length - 1);
  return parts.slice(0, parts.length - cut).join('.');
}

/**
 * The C.A.T labels a job may name (`set_cath_condition`): `x.x.x`, `C.x.x`, `C.A.x` and `C.A.T`
 * from the model's own vocabulary. A fourth `.x` level is accepted and dropped.
 */
export function normalizeCatLabel(value) {
  if (typeof value !== 'string') {
    throw new Error("CAT must be a quoted C.A.T label, e.g. '1.x.x' or '2.60.40'; only C/A/T are supported.");
  }
  let label = value;
  if (label.split('.').length === 4 && label.endsWith('.x')) label = label.slice(0, -2);
  const parts = label.split('.');
  const valid = label === 'x.x.x'
    || (parts.length === 3 && parts[1] === 'x' && parts[2] === 'x' && parts[0] in CATH_CLASS_MAP_)
    || (parts.length === 3 && parts[2] === 'x' && `${parts[0]}.${parts[1]}` in CATH_ARCHITECTURE_MAP_)
    || (parts.length === 3 && label in CATH_TOPOLOGY_MAP_);
  if (!valid) {
    throw new Error(
      `Unknown CAT label '${value}'; choose a C/A/T label from the model vocabulary, or 'x.x.x' to clear.`
    );
  }
  return `${label}.x`;
}

/**
 * Parse a CATH string (e.g. `"2.60.40"`, `"1.20"`, `"1"`, or `"x.x.x.x"`) into `[c, a, t]`
 * integer indices where `-1` denotes unconditioned.
 */
export function parseCathCode(codeStr) {
  if (Array.isArray(codeStr) && codeStr.length === 3) {
    return [Number(codeStr[0]), Number(codeStr[1]), Number(codeStr[2])];
  }
  // Upstream's own conversion (`datapipes/pipeline.py`): each level cut by
  // `extract_cath_code_by_level` and looked up, -1 where the map has no such code.
  const code = codeStr === null || codeStr === undefined ? 'x.x.x.x' : String(codeStr);
  return [
    CATH_CLASS_MAP_[cathCodeByLevel(code, 'C')] ?? -1,
    CATH_ARCHITECTURE_MAP_[cathCodeByLevel(code, 'A')] ?? -1,
    CATH_TOPOLOGY_MAP_[cathCodeByLevel(code, 'T')] ?? -1,
  ];
}

function matchFieldSpec(value, specStr, isNumeric = false) {
  if (!specStr || specStr === "*") return true;
  const trimmed = specStr.trim();
  let items = [trimmed];
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    items = trimmed
      .slice(1, -1)
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  } else if (trimmed.includes(",")) {
    items = trimmed.split(",").map((s) => s.trim()).filter(Boolean);
  }
  for (const item of items) {
    if (item === "*") return true;
    if (isNumeric) {
      const numVal = Number(value);
      const dashIdx = item.indexOf("-", item.startsWith("-") ? 1 : 0);
      if (dashIdx > 0) {
        const lo = Number(item.slice(0, dashIdx));
        const hi = Number(item.slice(dashIdx + 1));
        if (numVal >= lo && numVal <= hi) return true;
      } else if (numVal === Number(item)) {
        return true;
      }
    } else if (String(value) === item) {
      return true;
    }
  }
  return false;
}

/**
 * Evaluate a single `sel('<chain>/<resname>/<resid>/<atom>')` path or predicate against an atom list.
 * Returns a `Uint8Array(atoms.length)` boolean mask (1 = selected, 0 = not selected).
 */
export function evaluateSelection(atoms, expr) {
  const n = atoms.length;
  const mask = new Uint8Array(n);
  if (!expr) return mask;
  const s = String(expr).trim();

  // Handle compound `|`, `&`, `~`, `and`, `or`, `not`
  if (s.includes(" | ") || s.includes(" or ")) {
    const parts = s.split(/\s+(?:\||or)\s+/);
    for (const part of parts) {
      const sub = evaluateSelection(atoms, part);
      for (let i = 0; i < n; i++) mask[i] |= sub[i];
    }
    return mask;
  }
  if (s.includes(" & ") || s.includes(" and ")) {
    const parts = s.split(/\s+(?:&|and)\s+/);
    mask.fill(1);
    for (const part of parts) {
      const sub = evaluateSelection(atoms, part);
      for (let i = 0; i < n; i++) mask[i] &= sub[i];
    }
    return mask;
  }
  if (s.startsWith("~") || s.startsWith("not ")) {
    const rest = s.startsWith("~") ? s.slice(1).trim() : s.slice(4).trim();
    const sub = evaluateSelection(atoms, rest);
    for (let i = 0; i < n; i++) mask[i] = sub[i] ? 0 : 1;
    return mask;
  }

  if (s === "has_nan_coord()") {
    for (let i = 0; i < n; i++) {
      const c = atoms[i].coord;
      mask[i] = !c || !Number.isFinite(c[0]) || !Number.isFinite(c[1]) || !Number.isFinite(c[2]) ? 1 : 0;
    }
    return mask;
  }
  if (s === "is_ligand") {
    for (let i = 0; i < n; i++) mask[i] = atoms[i].is_ligand ? 1 : 0;
    return mask;
  }
  if (s === "is_polymer") {
    for (let i = 0; i < n; i++) mask[i] = atoms[i].is_polymer ? 1 : 0;
    return mask;
  }
  if (s === "is_protein") {
    for (let i = 0; i < n; i++) mask[i] = atoms[i].is_protein ? 1 : 0;
    return mask;
  }
  if (s === "all" || s === "*") {
    mask.fill(1);
    return mask;
  }

  // Parse `sel('...')` or direct `'chain/resname/resid/atom'`
  let pathStr = s;
  const selMatch = s.match(/^sel\(\s*['"]([^'"]*)['"]\s*\)$/);
  if (selMatch) {
    pathStr = selMatch[1];
  }

  const fields = pathStr.split("/");
  const chainSpec = fields[0] ?? "*";
  const resNameSpec = fields[1] ?? "*";
  const resIdSpec = fields[2] ?? "*";
  const atomNameSpec = fields[3] ?? "*";

  for (let i = 0; i < n; i++) {
    const a = atoms[i];
    if (!matchFieldSpec(a.chain_id, chainSpec, false)) continue;
    const matchResName =
      matchFieldSpec(a.res_name, resNameSpec, false) ||
      (a.segment_name && matchFieldSpec(a.segment_name, resNameSpec, false));
    if (!matchResName) continue;
    const resIdToCheck = a.original_res_id !== undefined && a.original_res_id > -1000000 ? a.original_res_id : a.res_id;
    if (!matchFieldSpec(resIdToCheck, resIdSpec, true) && !matchFieldSpec(a.res_id, resIdSpec, true)) continue;
    if (!matchFieldSpec(a.atom_name, atomNameSpec, false)) continue;
    mask[i] = 1;
  }
  return mask;
}

export const HBondConditioningEnum = HBondEnum;
export const SecondaryStructureEnum = SSEnum;
export const parseCathCodeTriple = parseCathCode;

export const ALL_CONDITIONS = Object.freeze([
  C_CRD,
  C_INTCRD,
  C_CRDGRP,
  C_IDX,
  C_IDXGRP,
  C_SEQ,
  C_HOT,
  C_RAS,
  C_SEC,
  C_HB,
  C_PISTACK,
  C_PICATION,
  C_SALT,
  C_HPHOB,
  C_MCOORD,
  C_CATH,
  C_CENTROID,
  C_BP,
  C_NA_SS_STATE,
  ...SEQUENCE_PROBABILITY_CONDITIONS.map((c) => ({ ...c, residueName: c.token_name })),
]);

export function resolveConditionSpecName(name) {
  const clean = String(name || "").trim();
  const found = ALL_CONDITIONS.find((c) => c.alias === clean || c.full_name === clean);
  if (!found) {
    throw new Error(`Unknown condition name: ${JSON.stringify(name)}`);
  }
  return found;
}

export function coerceEnumConditionValue(enumObj, rawValue) {
  if (typeof rawValue === "number" && Number.isFinite(rawValue)) {
    return Math.trunc(rawValue);
  }
  const str = String(rawValue ?? "").trim().toUpperCase();
  if (str && Object.prototype.hasOwnProperty.call(enumObj, str)) {
    return enumObj[str];
  }
  const parsed = Number.parseInt(str, 10);
  if (Number.isFinite(parsed)) return parsed;
  throw new Error(`Invalid enum condition value ${JSON.stringify(rawValue)}`);
}

export function parseCentroidValue(rawValue) {
  if (Array.isArray(rawValue) && rawValue.length === 3) {
    return [Number(rawValue[0]), Number(rawValue[1]), Number(rawValue[2])];
  }
  if (typeof rawValue === "string") {
    const parts = rawValue.replace(/^\[|\]$/g, "").split(",").map((s) => Number.parseFloat(s.trim()));
    if (parts.length === 3 && parts.every(Number.isFinite)) {
      return parts;
    }
  }
  return [Number.NaN, Number.NaN, Number.NaN];
}
