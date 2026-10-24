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

/**
 * Common CATH C/A/T index presets matching the checkpoint's
 * `embedding_C` (6), `embedding_A` (44), `embedding_T` (1473) tables.
 * Remember `FoldEmbeddingSeqFeat` adds +1 to stored indices (where -1 is null -> index 0).
 */
export const CATH_CLASS_MAP = Object.freeze({
  "1": 0, // Mainly Alpha
  "2": 1, // Mainly Beta
  "3": 2, // Alpha Beta
  "4": 3, // Few Secondary Structures
  "6": 4, // Special
});

export const CATH_ARCHITECTURE_MAP = Object.freeze({
  "1.10": 0, // Orthogonal Bundle
  "1.20": 1, // Up-down Bundle
  "1.25": 2, // Alpha Horseshoe
  "1.40": 3, // Alpha solenoid
  "1.50": 4, // Alpha/alpha barrel
  "2.10": 5, // Ribbon
  "2.20": 6, // Single Sheet
  "2.30": 7, // Roll
  "2.40": 8, // Beta Barrel
  "2.60": 9, // Sandwich (Immunoglobulin-like)
  "2.70": 10, // Distorted Sandwich
  "2.80": 11, // Trefoil
  "2.100": 12, // Aligned Prism
  "2.110": 13, // 4-Bladed Beta-Propeller
  "2.115": 14, // 5-Bladed Beta-Propeller
  "2.120": 15, // 6-Bladed Beta-Propeller
  "2.130": 16, // 7-Bladed Beta-Propeller
  "2.140": 17, // 8-Bladed Beta-Propeller
  "2.160": 18, // 3 Solenoid
  "2.170": 19, // Beta Complex
  "3.10": 20, // Roll (Alpha-Beta)
  "3.20": 21, // Alpha-Beta Barrel (TIM Barrel)
  "3.30": 22, // 2-Layer Sandwich
  "3.40": 23, // 3-Layer (aba) Sandwich (Rossmann)
  "3.50": 24, // 3-Layer (bba) Sandwich
  "3.55": 25, // 3-Layer (bab) Sandwich
  "3.60": 26, // 4-Layer Sandwich
  "3.65": 27, // Alpha-beta prism
  "3.70": 28, // Box
  "3.75": 29, // 5-stranded Propeller
  "3.80": 30, // Alpha-Beta Horseshoe
  "3.90": 31, // Alpha-Beta Complex
});

export const CATH_TOPOLOGY_MAP = Object.freeze({
  "1.10.8": 12,
  "1.10.10": 15,
  "1.20.5": 88,
  "1.20.120": 110, // Four-helix up-down bundle
  "1.25.40": 142, // TPR-like repeat
  "2.40.50": 295, // OB-fold beta barrel
  "2.40.128": 325, // Lipocalin / GFP beta barrel
  "2.60.40": 395, // Immunoglobulin-like beta-sandwich
  "2.130.10": 480, // 7-bladed beta-propeller
  "3.20.20": 620, // TIM barrel
  "3.30.70": 710, // Ferredoxin-like 2-layer sandwich
  "3.40.50": 890, // Rossmann fold
});

/**
 * Parse a CATH string (e.g. `"2.60.40"`, `"1.20"`, `"1"`, or `"x.x.x.x"`) into `[c, a, t]`
 * integer indices where `-1` denotes unconditioned.
 */
export function parseCathCode(codeStr) {
  if (!codeStr || codeStr === "x.x.x.x" || codeStr === "null") {
    return [-1, -1, -1];
  }
  if (Array.isArray(codeStr) && codeStr.length === 3) {
    return [Number(codeStr[0]), Number(codeStr[1]), Number(codeStr[2])];
  }
  const parts = String(codeStr).trim().split(".");
  const cKey = parts[0] && parts[0] !== "x" ? parts[0] : null;
  const aKey = cKey && parts[1] && parts[1] !== "x" ? `${cKey}.${parts[1]}` : null;
  const tKey = aKey && parts[2] && parts[2] !== "x" ? `${aKey}.${parts[2]}` : null;

  const cIdx = cKey ? (CATH_CLASS_MAP[cKey] ?? -1) : -1;
  const aIdx = aKey ? (CATH_ARCHITECTURE_MAP[aKey] ?? -1) : -1;
  const tIdx = tKey ? (CATH_TOPOLOGY_MAP[tKey] ?? -1) : -1;
  return [cIdx, aIdx, tIdx];
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
