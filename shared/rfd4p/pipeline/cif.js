// Pure ES module mmCIF & Annotated-CIF parser for RFD4-Proteina pipeline.
// Mirrors atomworks.io.parse(..., config=ANNOTATED_CIF / RCSB_WITH_METALS).

import { isBinaryCif, parseBinaryCifBlock } from './bcif.js';
import { decodeTextOrZstd, unwrapBytes } from './zstd.js';
import { CCD_TEMPLATES } from './ccd-templates.js';
import {
  BondRows, inferLinkOrder, linkDictionary, resolveLinkChemistry, validateInferredPolymerBond,
} from './links.js';

export const BOND_TYPE = {
  ANY: 0,
  SINGLE: 1,
  DOUBLE: 2,
  TRIPLE: 3,
  QUADRUPLE: 4,
  AROMATIC_SINGLE: 5,
  AROMATIC_DOUBLE: 6,
  AROMATIC_TRIPLE: 7,
  COORDINATION: 8,
  AROMATIC: 9,
};

export const CHAIN_TYPE = {
  CYCLIC_PSEUDO_PEPTIDE: 0,
  OTHER_POLYMER: 1,
  PEPTIDE_NUCLEIC_ACID: 2,
  DNA: 3,
  DNA_RNA_HYBRID: 4,
  POLYPEPTIDE_D: 5,
  POLYPEPTIDE_L: 6,
  RNA: 7,
  NON_POLYMER: 8,
  WATER: 9,
  BRANCHED: 10,
  MACROLIDE: 11,
};

const MMCIF_CHAIN_TYPE_MAP = {
  'cyclic-pseudo-peptide': CHAIN_TYPE.CYCLIC_PSEUDO_PEPTIDE,
  'other': CHAIN_TYPE.OTHER_POLYMER,
  'peptide nucleic acid': CHAIN_TYPE.PEPTIDE_NUCLEIC_ACID,
  'polydeoxyribonucleotide': CHAIN_TYPE.DNA,
  'polydeoxyribonucleotide/polyribonucleotide hybrid': CHAIN_TYPE.DNA_RNA_HYBRID,
  'polypeptide(d)': CHAIN_TYPE.POLYPEPTIDE_D,
  'polypeptide(l)': CHAIN_TYPE.POLYPEPTIDE_L,
  'polyribonucleotide': CHAIN_TYPE.RNA,
  'branched': CHAIN_TYPE.BRANCHED,
  'macrolide': CHAIN_TYPE.MACROLIDE,
  'non-polymer': CHAIN_TYPE.NON_POLYMER,
  'water': CHAIN_TYPE.WATER,
};

const COMP_BOND_ORDER_TO_TYPE = {
  'SING:N': BOND_TYPE.SINGLE,
  'DOUB:N': BOND_TYPE.DOUBLE,
  'TRIP:N': BOND_TYPE.TRIPLE,
  'QUAD:N': BOND_TYPE.QUADRUPLE,
  'SING:Y': BOND_TYPE.AROMATIC_SINGLE,
  'DOUB:Y': BOND_TYPE.AROMATIC_DOUBLE,
  'TRIP:Y': BOND_TYPE.AROMATIC_TRIPLE,
  'AROM:Y': BOND_TYPE.AROMATIC,
  'AROM:N': BOND_TYPE.AROMATIC,
};

const STRUCT_CONN_ORDER_TO_TYPE = {
  sing: BOND_TYPE.SINGLE,
  doub: BOND_TYPE.DOUBLE,
  trip: BOND_TYPE.TRIPLE,
  quad: BOND_TYPE.QUADRUPLE,
};

export const ELEMENT_TO_ATOMIC_NUMBER = {
  H: 1, D: 1, T: 1, HE: 2, LI: 3, BE: 4, B: 5, C: 6, N: 7, O: 8, F: 9, NE: 10,
  NA: 11, MG: 12, AL: 13, SI: 14, P: 15, S: 16, CL: 17, AR: 18, K: 19, CA: 20,
  SC: 21, TI: 22, V: 23, CR: 24, MN: 25, FE: 26, CO: 27, NI: 28, CU: 29, ZN: 30,
  GA: 31, GE: 32, AS: 33, SE: 34, BR: 35, KR: 36, RB: 37, SR: 38, Y: 39, ZR: 40,
  NB: 41, MO: 42, TC: 43, RU: 44, RH: 45, PD: 46, AG: 47, CD: 48, IN: 49, SN: 50,
  SB: 51, TE: 52, I: 53, XE: 54, CS: 55, BA: 56, LA: 57, CE: 58, PR: 59, ND: 60,
  PM: 61, SM: 62, EU: 63, GD: 64, TB: 65, DY: 66, HO: 67, ER: 68, TM: 69, YB: 70,
  LU: 71, HF: 72, TA: 73, W: 74, RE: 75, OS: 76, IR: 77, PT: 78, AU: 79, HG: 80,
  TL: 81, PB: 82, BI: 83, PO: 84, AT: 85, RN: 86, FR: 87, RA: 88, AC: 89, TH: 90,
  PA: 91, U: 92,
};

export const METAL_ELEMENTS = new Set([
  'LI', 'BE', 'NA', 'MG', 'AL', 'K', 'CA', 'SC', 'TI', 'V', 'CR', 'MN', 'FE',
  'CO', 'NI', 'CU', 'ZN', 'GA', 'RB', 'SR', 'Y', 'ZR', 'NB', 'MO', 'TC', 'RU',
  'RH', 'PD', 'AG', 'CD', 'IN', 'SN', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND', 'PM',
  'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'W',
  'RE', 'OS', 'IR', 'PT', 'AU', 'HG', 'TL', 'PB', 'BI',
]);

// Canonical heavy-atom bonds for the 28 standard amino acid and RNA/DNA residues.
export const STD_RESIDUE_BONDS = {
  ALA: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['N','CA',1]],
  ARG: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CD','NE',1],['CG','CD',1],['CZ','NH1',1],['CZ','NH2',2],['N','CA',1],['NE','CZ',1]],
  ASN: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CG','ND2',1],['CG','OD1',2],['N','CA',1]],
  ASP: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CG','OD1',2],['CG','OD2',1],['N','CA',1]],
  CYS: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','SG',1],['N','CA',1]],
  GLN: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CD','NE2',1],['CD','OE1',2],['CG','CD',1],['N','CA',1]],
  GLU: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CD','OE1',2],['CD','OE2',1],['CG','CD',1],['N','CA',1]],
  GLY: [['C','O',2],['C','OXT',1],['CA','C',1],['N','CA',1]],
  HIS: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CD2','NE2',5],['CE1','NE2',5],['CG','CD2',6],['CG','ND1',5],['N','CA',1],['ND1','CE1',6]],
  ILE: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG1',1],['CB','CG2',1],['CG1','CD1',1],['N','CA',1]],
  LEU: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CG','CD1',1],['CG','CD2',1],['N','CA',1]],
  LYS: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CD','CE',1],['CE','NZ',1],['CG','CD',1],['N','CA',1]],
  MET: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CG','SD',1],['N','CA',1],['SD','CE',1]],
  PHE: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CD1','CE1',5],['CD2','CE2',6],['CE1','CZ',6],['CE2','CZ',5],['CG','CD1',6],['CG','CD2',5],['N','CA',1]],
  PRO: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CG','CD',1],['N','CA',1],['N','CD',1]],
  SER: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','OG',1],['N','CA',1]],
  THR: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG2',1],['CB','OG1',1],['N','CA',1]],
  TRP: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CD1','NE1',5],['CD2','CE2',6],['CD2','CE3',5],['CE2','CZ2',5],['CE3','CZ3',6],['CG','CD1',6],['CG','CD2',5],['CZ2','CH2',6],['CZ3','CH2',5],['N','CA',1],['NE1','CE2',5]],
  TYR: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG',1],['CD1','CE1',5],['CD2','CE2',6],['CE1','CZ',6],['CE2','CZ',5],['CG','CD1',6],['CG','CD2',5],['CZ','OH',1],['N','CA',1]],
  VAL: [['C','O',2],['C','OXT',1],['CA','C',1],['CA','CB',1],['CB','CG1',1],['CB','CG2',1],['N','CA',1]],
  A: [["C1'",'N9',1],['C2','N3',6],["C2'","C1'",1],["C2'","O2'",1],["C3'","C2'",1],["C3'","O3'",1],["C4'","C3'",1],["C4'","O4'",1],['C5','C4',6],['C5','C6',5],["C5'","C4'",1],['C6','N1',6],['C6','N6',1],['C8','N7',6],['N1','C2',5],['N3','C4',5],['N7','C5',5],['N9','C4',5],['N9','C8',5],["O4'","C1'",1],["O5'","C5'",1],['OP3','P',1],['P',"O5'",1],['P','OP1',2],['P','OP2',1]],
  C: [["C1'",'N1',1],['C2','N3',1],['C2','O2',2],["C2'","C1'",1],["C2'","O2'",1],["C3'","C2'",1],["C3'","O3'",1],['C4','C5',1],['C4','N4',1],["C4'","C3'",1],["C4'","O4'",1],['C5','C6',2],["C5'","C4'",1],['N1','C2',1],['N1','C6',1],['N3','C4',2],["O4'","C1'",1],["O5'","C5'",1],['OP3','P',1],['P',"O5'",1],['P','OP1',2],['P','OP2',1]],
  G: [["C1'",'N9',1],['C2','N2',1],['C2','N3',2],["C2'","C1'",1],["C2'","O2'",1],["C3'","C2'",1],["C3'","O3'",1],["C4'","C3'",1],["C4'","O4'",1],['C5','C4',6],['C5','C6',1],["C5'","C4'",1],['C6','N1',1],['C6','O6',2],['C8','N7',6],['N1','C2',1],['N3','C4',1],['N7','C5',5],['N9','C4',5],['N9','C8',5],["O4'","C1'",1],["O5'","C5'",1],['OP3','P',1],['P',"O5'",1],['P','OP1',2],['P','OP2',1]],
  U: [["C1'",'N1',1],['C2','N3',1],['C2','O2',2],["C2'","C1'",1],["C2'","O2'",1],["C3'","C2'",1],["C3'","O3'",1],['C4','C5',1],['C4','O4',2],["C4'","C3'",1],["C4'","O4'",1],['C5','C6',2],["C5'","C4'",1],['N1','C2',1],['N1','C6',1],['N3','C4',1],["O4'","C1'",1],["O5'","C5'",1],['OP3','P',1],['P',"O5'",1],['P','OP1',2],['P','OP2',1]],
  DA: [["C1'",'N9',1],['C2','N3',6],["C2'","C1'",1],["C3'","C2'",1],["C3'","O3'",1],["C4'","C3'",1],["C4'","O4'",1],['C5','C4',6],['C5','C6',5],["C5'","C4'",1],['C6','N1',6],['C6','N6',1],['C8','N7',6],['N1','C2',5],['N3','C4',5],['N7','C5',5],['N9','C4',5],['N9','C8',5],["O4'","C1'",1],["O5'","C5'",1],['OP3','P',1],['P',"O5'",1],['P','OP1',2],['P','OP2',1]],
  DC: [["C1'",'N1',1],['C2','N3',1],['C2','O2',2],["C2'","C1'",1],["C3'","C2'",1],["C3'","O3'",1],['C4','C5',1],['C4','N4',1],["C4'","C3'",1],["C4'","O4'",1],['C5','C6',2],["C5'","C4'",1],['N1','C2',1],['N1','C6',1],['N3','C4',2],["O4'","C1'",1],["O5'","C5'",1],['OP3','P',1],['P',"O5'",1],['P','OP1',2],['P','OP2',1]],
  DG: [["C1'",'N9',1],['C2','N2',1],['C2','N3',2],["C2'","C1'",1],["C3'","C2'",1],["C3'","O3'",1],["C4'","C3'",1],["C4'","O4'",1],['C5','C4',6],['C5','C6',1],["C5'","C4'",1],['C6','N1',1],['C6','O6',2],['C8','N7',6],['N1','C2',1],['N3','C4',1],['N7','C5',5],['N9','C4',5],['N9','C8',5],["O4'","C1'",1],["O5'","C5'",1],['OP3','P',1],['P',"O5'",1],['P','OP1',2],['P','OP2',1]],
  DT: [["C1'",'N1',1],['C2','N3',1],['C2','O2',2],["C2'","C1'",1],["C3'","C2'",1],["C3'","O3'",1],['C4','C5',1],['C4','O4',2],["C4'","C3'",1],["C4'","O4'",1],['C5','C6',2],['C5','C7',1],["C5'","C4'",1],['N1','C2',1],['N1','C6',1],['N3','C4',1],["O4'","C1'",1],["O5'","C5'",1],['OP3','P',1],['P',"O5'",1],['P','OP1',2],['P','OP2',1]],
};

export const STD_RESIDUE_AROMATIC = {
  HIS: new Set(['CG', 'ND1', 'CD2', 'CE1', 'NE2']),
  PHE: new Set(['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']),
  TRP: new Set(['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']),
  TYR: new Set(['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']),
  A: new Set(['N9', 'C8', 'N7', 'C5', 'C6', 'N1', 'C2', 'N3', 'C4']),
  G: new Set(['N9', 'C8', 'N7', 'C5', 'C4']),
  DA: new Set(['N9', 'C8', 'N7', 'C5', 'C6', 'N1', 'C2', 'N3', 'C4']),
  DG: new Set(['N9', 'C8', 'N7', 'C5', 'C4']),
};

export const AA_RESIDUES = new Set([
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK',
]);
export const RNA_RESIDUES = new Set(['A', 'C', 'G', 'U', 'N']);
export const DNA_RESIDUES = new Set(['DA', 'DC', 'DG', 'DT', 'DN']);

// 1-body scalar StandardAnnotation specifications
export const SCALAR_1BODY_ANNOTATIONS = {
  'annotation_expsegmin_1_residue': { type: 'int', default: -1 },
  'annotation_expsegmax_1_residue': { type: 'int', default: -1 },
  'annotation_src-component_1_atom': { type: 'str', default: '' },
  'annotation_src-index-group_1_atom': { type: 'int', default: 0 },
  'annotation_src-coordinate-conditioned-region_1_atom': { type: 'int', default: -1 },
  'condition_sequence_1_residue': { type: 'str', default: '<M>' },
  'condition_index_1_residue': { type: 'bool', default: true },
  'condition_index-group_1_token': { type: 'int', default: 0 },
  'condition_hotspot_1_atom': { type: 'bool', default: false },
  'condition_hbond_1_atom': { type: 'int', default: 0 },
  'condition_rasa-binned_1_atom': { type: 'int', default: 0 },
  'condition_pistack_1_atom': { type: 'bool', default: false },
  'condition_saltbridge_1_atom': { type: 'int', default: 0 },
  'condition_hydrophobic_1_atom': { type: 'bool', default: false },
  'condition_pication_1_atom': { type: 'int', default: 0 },
  'condition_metal-coord_1_atom': { type: 'int', default: 0 },
  'condition_CoordinateGroup_1_atom': { type: 'int', default: -1 },
  'condition_secondary-structure_1_residue': { type: 'int', default: 0 },
  // 🔴 THE NUCLEIC STATE (BasePairEnum: UNSPECIFIED, PAIR, LOOP), which upstream's annotated
  // inputs carry and the model reads as `na_ss_state` - and which this table did not list, so the
  // reader never kept it and the batch wrote zeros whatever the file said.
  'condition_na-ss-state_1_residue': { type: 'int', default: 0 },
  'condition_n-terminus_1_residue': { type: 'bool', default: false },
  'condition_c-terminus_1_residue': { type: 'bool', default: false },
  // 🔴 AND THE BASE PAIRS, a TWO-body annotation (`_condition_basepair_2_residue`, an atom pair
  // a row), carried as two per-atom columns so that every slice, concatenation and insertion
  // keeps them without learning about pairs: each atom's index in the file it was read from, and
  // the file indices of its partners, comma-joined. `convertToProteinaBatch` resolves them to
  // tokens as upstream's tokeniser does.
  'annotation_src-atom-index_1_atom': { type: 'int', default: -1 },
  'condition_basepair-partners_1_atom': { type: 'str', default: '' },
  'mask_sequence_1_residue': { type: 'bool', default: false },
  'mask_coordinate_1_atom': { type: 'bool', default: false },
  'mask_index_1_residue': { type: 'bool', default: true },
};

export const NONSCALAR_1BODY_ANNOTATIONS = {
  'condition_coordinate_1_atom': { cols: 3, type: 'float', default: NaN },
  'condition_InternalCoordinate_1_atom': { cols: 3, type: 'float', default: NaN },
  'condition_ChainCentroid_1_atom': { cols: 3, type: 'float', default: NaN },
  'condition_cath-label_1_residue': { cols: 3, type: 'int', default: -1 },
  'annotation_coords-with-insertion_1_atom': { cols: 3, type: 'float', default: NaN },
};

/**
 * Tokenize an mmCIF text into an array of { value, quoted } tokens.
 */
function tokenizeCif(text) {
  const tokens = [];
  const n = text.length;
  let i = 0;
  let atLineStart = true;

  while (i < n) {
    // Multiline semicolon string at start of line
    if (atLineStart && text[i] === ';') {
      let end = i + 1;
      while (end < n) {
        if ((text[end] === '\n' || text[end] === '\r') && end + 1 < n) {
          let next = end + 1;
          if (text[end] === '\r' && text[next] === '\n') next++;
          if (text[next] === ';') {
            const val = text.slice(i + 1, end);
            tokens.push({ value: val, quoted: true });
            i = next + 1;
            atLineStart = false;
            break;
          }
        }
        end++;
      }
      if (end >= n) break;
      continue;
    }

    const ch = text[i];
    if (ch === '\n' || ch === '\r') {
      atLineStart = true;
      i++;
      continue;
    }
    if (ch === ' ' || ch === '\t') {
      i++;
      continue;
    }
    atLineStart = false;

    // Comment
    if (ch === '#') {
      while (i < n && text[i] !== '\n' && text[i] !== '\r') i++;
      continue;
    }

    // Quoted string ('...' or "...")
    if (ch === "'" || ch === '"') {
      const quote = ch;
      let j = i + 1;
      while (j < n) {
        if (text[j] === quote) {
          const nextCh = j + 1 < n ? text[j + 1] : ' ';
          if (nextCh === ' ' || nextCh === '\t' || nextCh === '\n' || nextCh === '\r') {
            break;
          }
        }
        j++;
      }
      tokens.push({ value: text.slice(i + 1, j), quoted: true });
      i = j + 1;
      continue;
    }

    // Unquoted token
    let j = i;
    while (j < n) {
      const c = text[j];
      if (c === ' ' || c === '\t' || c === '\n' || c === '\r' || c === '#') break;
      j++;
    }
    tokens.push({ value: text.slice(i, j), quoted: false });
    i = j;
  }

  return tokens;
}

/**
 * Parse an mmCIF string into a block dictionary:
 *   block[categoryName][fieldName] = Array<string>
 */
export function parseCifBlock(cifText) {
  const tokens = tokenizeCif(cifText);
  const block = {};
  let i = 0;
  const n = tokens.length;

  while (i < n) {
    const tok = tokens[i];
    if (!tok.quoted && tok.value.startsWith('data_')) {
      i++;
      continue;
    }
    if (!tok.quoted && tok.value === 'loop_') {
      i++;
      const headers = [];
      while (i < n && !tokens[i].quoted && tokens[i].value.startsWith('_')) {
        const dot = tokens[i].value.indexOf('.');
        const cat = tokens[i].value.slice(1, dot);
        const col = tokens[i].value.slice(dot + 1);
        headers.push({ cat, col });
        if (!block[cat]) block[cat] = {};
        if (!block[cat][col]) block[cat][col] = [];
        i++;
      }
      const nCols = headers.length;
      if (nCols === 0) continue;
      while (i < n) {
        const t = tokens[i];
        if (!t.quoted && (t.value === 'loop_' || t.value.startsWith('_') || t.value.startsWith('data_'))) {
          break;
        }
        for (let c = 0; c < nCols && i < n; c++, i++) {
          const h = headers[c];
          block[h.cat][h.col].push(tokens[i].value);
        }
      }
      continue;
    }
    if (!tok.quoted && tok.value.startsWith('_')) {
      const dot = tok.value.indexOf('.');
      const cat = tok.value.slice(1, dot);
      const col = tok.value.slice(dot + 1);
      if (!block[cat]) block[cat] = {};
      const val = i + 1 < n ? tokens[i + 1].value : '';
      block[cat][col] = [val];
      i += 2;
      continue;
    }
    i++;
  }

  return block;
}

function safeInt(s, fallback = 0) {
  if (s === undefined || s === null || s === '.' || s === '?' || s === '') return fallback;
  const v = parseInt(s, 10);
  return Number.isNaN(v) ? fallback : v;
}

function safeFloat(s, fallback = NaN) {
  if (s === undefined || s === null || s === '.' || s === '?' || s === '' || s === 'nan' || s === 'NaN') return fallback;
  const v = parseFloat(s);
  return Number.isNaN(v) ? fallback : v;
}

function safeBool(s, fallback = false) {
  if (s === undefined || s === null || s === '.' || s === '?' || s === '') return fallback;
  if (s === 'True' || s === 'true' || s === '1' || s === 'Y') return true;
  if (s === 'False' || s === 'false' || s === '0' || s === 'N') return false;
  return fallback;
}

export function getResidueSegments(atoms) {
  const n = atoms.nAtoms;
  if (n === 0) return [0];
  const starts = [0];
  const hasTid = Array.isArray(atoms.transformation_id);
  for (let i = 1; i < n; i++) {
    if (
      atoms.chain_id[i] !== atoms.chain_id[i - 1] ||
      atoms.res_id[i] !== atoms.res_id[i - 1] ||
      atoms.res_name[i] !== atoms.res_name[i - 1] ||
      atoms.ins_code[i] !== atoms.ins_code[i - 1] ||
      (hasTid && atoms.transformation_id[i] !== atoms.transformation_id[i - 1])
    ) {
      starts.push(i);
    }
  }
  starts.push(n);
  return starts;
}

export function getPolymerizationAtoms(resName, chainType) {
  if (
    chainType === CHAIN_TYPE.POLYPEPTIDE_L ||
    chainType === CHAIN_TYPE.POLYPEPTIDE_D ||
    chainType === CHAIN_TYPE.CYCLIC_PSEUDO_PEPTIDE ||
    AA_RESIDUES.has(resName)
  ) {
    return ['C', 'N'];
  }
  if (
    chainType === CHAIN_TYPE.DNA ||
    chainType === CHAIN_TYPE.RNA ||
    chainType === CHAIN_TYPE.DNA_RNA_HYBRID ||
    RNA_RESIDUES.has(resName) ||
    DNA_RESIDUES.has(resName)
  ) {
    return ["O3'", 'P'];
  }
  return [null, null];
}

/**
 * Compute pn_unit_id, molecule_id, chain_entity, pn_unit_entity, molecule_entity,
 * chain_iid, pn_unit_iid, molecule_iid on `atoms` in place (mirroring atomworks).
 */
export function computeIdAndEntityAnnotations(atoms) {
  const n = atoms.nAtoms;
  if (n === 0) {
    atoms.pn_unit_id = [];
    atoms.pn_unit_iid = [];
    atoms.molecule_id = [];
    atoms.molecule_iid = [];
    atoms.chain_entity = [];
    atoms.pn_unit_entity = [];
    atoms.molecule_entity = [];
    atoms.chain_iid = [];
    return atoms;
  }

  const nonCoordBonds = (atoms.bonds || []).filter((b) => b[2] !== BOND_TYPE.COORDINATION);

  // 1. pn_unit_id: start with chain_id; merge connected non-polymer chains
  const pn_unit_id = atoms.chain_id.slice();
  const nonPolyChains = Array.from(new Set(atoms.chain_id.filter((_, i) => !atoms.is_polymer[i]))).sort();
  if (nonPolyChains.length > 0) {
    const chainToIdx = new Map(nonPolyChains.map((c, idx) => [c, idx]));
    const adj = Array.from({ length: nonPolyChains.length }, (_, idx) => new Set([idx]));
    for (const [u, v] of nonCoordBonds) {
      if (!atoms.is_polymer[u] && !atoms.is_polymer[v]) {
        const cu = chainToIdx.get(atoms.chain_id[u]);
        const cv = chainToIdx.get(atoms.chain_id[v]);
        if (cu !== undefined && cv !== undefined && cu !== cv) {
          adj[cu].add(cv);
          adj[cv].add(cu);
        }
      }
    }
    const visited = new Uint8Array(nonPolyChains.length);
    for (let i = 0; i < nonPolyChains.length; i++) {
      if (visited[i]) continue;
      const comp = [];
      const q = [i];
      visited[i] = 1;
      while (q.length > 0) {
        const cur = q.shift();
        comp.push(nonPolyChains[cur]);
        const nbrs = Array.from(adj[cur]).sort((a, b) => a - b);
        for (const nb of nbrs) {
          if (!visited[nb]) {
            visited[nb] = 1;
            q.push(nb);
          }
        }
      }
      const joined = comp.slice().sort().join(',');
      const compSet = new Set(comp);
      for (let a = 0; a < n; a++) {
        if (compSet.has(atoms.chain_id[a])) pn_unit_id[a] = joined;
      }
    }
  }
  atoms.pn_unit_id = pn_unit_id;

  // 2. molecule_id: connected components of pn_unit_id (sorted unique nodes)
  const uniquePnUnits = Array.from(new Set(pn_unit_id)).sort();
  const pnToIdx = new Map(uniquePnUnits.map((p, idx) => [p, idx]));
  const pnAdj = Array.from({ length: uniquePnUnits.length }, (_, idx) => new Set([idx]));
  for (const [u, v] of nonCoordBonds) {
    const pu = pnToIdx.get(pn_unit_id[u]);
    const pv = pnToIdx.get(pn_unit_id[v]);
    if (pu !== undefined && pv !== undefined && pu !== pv) {
      pnAdj[pu].add(pv);
      pnAdj[pv].add(pu);
    }
  }
  const pnVisited = new Uint8Array(uniquePnUnits.length);
  const pnToMolId = new Map();
  let molCounter = 0;
  for (let i = 0; i < uniquePnUnits.length; i++) {
    if (pnVisited[i]) continue;
    const q = [i];
    pnVisited[i] = 1;
    while (q.length > 0) {
      const cur = q.shift();
      pnToMolId.set(uniquePnUnits[cur], molCounter);
      const nbrs = Array.from(pnAdj[cur]).sort((a, b) => a - b);
      for (const nb of nbrs) {
        if (!pnVisited[nb]) {
          pnVisited[nb] = 1;
          q.push(nb);
        }
      }
    }
    molCounter++;
  }
  atoms.molecule_id = pn_unit_id.map((p) => pnToMolId.get(p));

  // Helper to compute entity IDs at a given level ("chain", "pn_unit", "molecule")
  function annotateLevelEntities(levelCol, lowerIdCol, lowerEntityCol, includeInterBonds) {
    const levelVals = atoms[levelCol];
    const uniqueInstances = Array.from(new Set(levelVals)).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    const hashToEntity = new Map();
    let nextEntity = 0;
    const out = new Array(n).fill(0);

    for (const inst of uniqueInstances) {
      const atomIdxs = [];
      for (let i = 0; i < n; i++) {
        if (levelVals[i] === inst) atomIdxs.push(i);
      }
      const globalToLocal = new Map(atomIdxs.map((g, l) => [g, l]));
      const lowerIds = atomIdxs.map((g) => atoms[lowerIdCol][g]);
      const lowerEnts = atomIdxs.map((g) => atoms[lowerEntityCol][g]);

      // Unique sorted lower-level nodes
      const uniqueNodes = Array.from(new Set(lowerIds)).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
      const nodeToIdx = new Map(uniqueNodes.map((nd, idx) => [nd, idx]));
      const nodeLabels = uniqueNodes.map((nd) => {
        const firstLocal = lowerIds.indexOf(nd);
        return String(lowerEnts[firstLocal]);
      });
      const nodeAdj = Array.from({ length: uniqueNodes.length }, (_, idx) => new Set([idx]));
      const interBonds = [];

      for (const [u, v] of nonCoordBonds) {
        const lu = globalToLocal.get(u);
        const lv = globalToLocal.get(v);
        if (lu === undefined || lv === undefined) continue;
        const nu = nodeToIdx.get(lowerIds[lu]);
        const nv = nodeToIdx.get(lowerIds[lv]);
        if (nu !== nv) {
          nodeAdj[nu].add(nv);
          nodeAdj[nv].add(nu);
          if (includeInterBonds) {
            const t1 = `${lowerEnts[lu]}:${atoms.res_id[u]}:${atoms.res_name[u]}:${atoms.atom_name[u]}`;
            const t2 = `${lowerEnts[lv]}:${atoms.res_id[v]}:${atoms.res_name[v]}:${atoms.atom_name[v]}`;
            interBonds.push(t1 < t2 ? `${t1}|${t2}` : `${t2}|${t1}`);
          }
        }
      }

      // 3-iteration Weisfeiler-Lehman refinement
      let curLabels = nodeLabels.slice();
      for (let iter = 0; iter < 3; iter++) {
        const nextLabels = new Array(uniqueNodes.length);
        for (let k = 0; k < uniqueNodes.length; k++) {
          const nbLabels = Array.from(nodeAdj[k]).map((nb) => curLabels[nb]).sort();
          nextLabels[k] = `${curLabels[k]}(${nbLabels.join(',')})`;
        }
        curLabels = nextLabels;
      }
      const wlPart = curLabels.slice().sort().join(';');
      const countMap = new Map();
      for (const lbl of nodeLabels) countMap.set(lbl, (countMap.get(lbl) || 0) + 1);
      const countPart = Array.from(countMap.entries())
        .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
        .map(([k, c]) => `${k}:${c}`)
        .join(',');
      interBonds.sort();
      const fullHash = `${wlPart}_${uniqueNodes.length}_${countPart}_${interBonds.join('&')}`;

      let eid = hashToEntity.get(fullHash);
      if (eid === undefined) {
        eid = nextEntity++;
        hashToEntity.set(fullHash, eid);
      }
      for (const g of atomIdxs) out[g] = eid;
    }
    return out;
  }

  atoms.chain_entity = annotateLevelEntities('chain_id', 'res_id', 'res_name', false);
  atoms.pn_unit_entity = annotateLevelEntities('pn_unit_id', 'chain_id', 'chain_entity', true);
  atoms.molecule_entity = annotateLevelEntities('molecule_id', 'pn_unit_id', 'pn_unit_entity', true);

  // Instance IDs (chain_iid, pn_unit_iid, molecule_iid)
  const tid = atoms.transformation_id || new Array(n).fill('1');
  atoms.transformation_id = tid;
  atoms.chain_iid = atoms.chain_id.map((c, i) => `${c}_${tid[i]}`);
  atoms.pn_unit_iid = atoms.pn_unit_id.map((p, i) =>
    p.split(',').map((u) => `${u}_${tid[i]}`).join(',')
  );
  const molIidStr = atoms.molecule_id.map((m, i) => `${m}${tid[i]}`);
  const uniqueMolIidStr = Array.from(new Set(molIidStr)).sort();
  const molIidMap = new Map(uniqueMolIidStr.map((s, idx) => [s, idx]));
  atoms.molecule_iid = molIidStr.map((s) => molIidMap.get(s));

  return atoms;
}

/**
 * atomworks' `add_polymer_bonds`: the backbone bond between consecutive residues of one chain
 * instance, where neither site already reaches the other residue (an authored link, either way
 * round) and the numbering does not jump - by MORE than one, so 52 to 52A (an insertion code,
 * the same number) is bonded, which a `== 1` test here did not.
 * @returns {number[][]} the `[out, in]` pairs to add, in residue order
 */
function polymerBondsToAdd(atoms, bonds, resStarts, atomToRes, dict) {
  // `_validate_inferred_polymer_bond` reads the degree before any backbone bond is added.
  const degree = new BondRows(bonds).degrees(atoms.nAtoms);
  const isH = (i) => ['H', 'H2', 'D', 'T'].includes(atoms.element[i]);
  const neighbours = new Map();
  for (const [u, v, bt] of bonds) {
    if (bt === BOND_TYPE.COORDINATION) continue;
    if (!neighbours.has(u)) neighbours.set(u, []);
    if (!neighbours.has(v)) neighbours.set(v, []);
    neighbours.get(u).push(v);
    neighbours.get(v).push(u);
  }
  const find = (s, e, name) => {
    for (let i = s; i < e; i++) if (atoms.atom_name[i] === name) return i;
    return -1;
  };
  const out = [];
  for (let r = 0; r < resStarts.length - 2; r++) {
    const s1 = resStarts[r];
    const e1 = resStarts[r + 1];
    const s2 = resStarts[r + 1];
    const e2 = resStarts[r + 2];
    if (atoms.chain_id[s1] !== atoms.chain_id[s2]) continue;
    if (atoms.transformation_id && atoms.transformation_id[s1] !== atoms.transformation_id[s2]) continue;
    if (atoms.res_id[s2] - atoms.res_id[s1] > 1) continue;
    const atoms1 = getPolymerizationAtoms(atoms.res_name[s1], atoms.chain_type[s1]);
    const atoms2 = getPolymerizationAtoms(atoms.res_name[s2], atoms.chain_type[s2]);
    if (!atoms1[0] || !atoms2[1]) continue;
    const idxIn = find(s2, e2, atoms2[1]);
    const idxOut = find(s1, e1, atoms1[0]);
    if (idxIn < 0 || idxOut < 0) continue;
    // `_has_existing_polymer_attachment`: either site already bonded into the other residue.
    if ((neighbours.get(idxOut) ?? []).some((x) => atomToRes[x] === r + 1)
      || (neighbours.get(idxIn) ?? []).some((x) => atomToRes[x] === r)) continue;
    // An explicit backbone link may run opposite to the numbering.
    const reverseIn = atoms1[1] ? find(s1, e1, atoms1[1]) : -1;
    const reverseOut = atoms2[0] ? find(s2, e2, atoms2[0]) : -1;
    if (reverseIn >= 0 && reverseOut >= 0 && (neighbours.get(reverseIn) ?? []).includes(reverseOut)) continue;
    validateInferredPolymerBond(atoms, idxOut, idxIn, degree, (i) => (neighbours.get(i) ?? []).some(isH), dict);
    out.push([idxOut, idxIn]);
  }
  return out;
}

/**
 * `add_polymer_bonds` over a finished array, as upstream's `parse_atom_array` re-prepares the
 * structure a condition spec built: two preserved segments a contig joins directly ARE bonded -
 * and refused if the bond is impossible (`bond_length`), where this port left them unbonded.
 */
export function addPolymerBonds(atoms, { ccdBlocks = null } = {}) {
  if (!atoms.bonds || !atoms.chain_type || atoms.nAtoms === 0) return atoms;
  const dict = linkDictionary(ccdBlocks, getPolymerizationAtoms);
  const resStarts = getResidueSegments(atoms);
  const atomToRes = new Int32Array(atoms.nAtoms);
  for (let r = 0; r < resStarts.length - 1; r++) {
    for (let i = resStarts[r]; i < resStarts[r + 1]; i++) atomToRes[i] = r;
  }
  for (const [u, v] of polymerBondsToAdd(atoms, atoms.bonds, resStarts, atomToRes, dict)) {
    atoms.bonds.push([Math.min(u, v), Math.max(u, v), BOND_TYPE.SINGLE]);
  }
  return atoms;
}

/**
 * Add intra-residue bonds, struct_conn bonds, polymer backbone bonds, and retype metal bonds -
 * atomworks' `_add_bonds`, in its order. With `sanitize` (a raw structure being prepared,
 * upstream's `add_missing_atoms`) the authored links' products are then built (links.js), and
 * the hydrogens that held them are dropped.
 *
 * 🔴 THE BONDS ARE KEPT IN biotite's ROW ORDER, because link chemistry takes the first match in
 * several places: `merge` puts the INCOMING list first, so the backbone bonds come first, then each
 * residue's dictionary bonds, then the `struct_conn` links they do not repeat - and where a link
 * repeats a residue's own bond (1FYL's BRU C5-BR), the dictionary's order wins.
 *
 * @param {object} atoms
 * @param {object|null} cifBlock
 * @param {object|null} customBondDict
 * @param {{sanitize?: boolean, ccdBlocks?: Map<string, object>}} [opts]
 */
export function buildBondsForAtomArray(atoms, cifBlock = null, customBondDict = null, opts = {}) {
  const n = atoms.nAtoms;
  const sanitize = Boolean(opts.sanitize);
  const dict = linkDictionary(opts.ccdBlocks ?? null, getPolymerizationAtoms);

  // Each residue's bonds: the dictionary's (hydrogens too), overlaid by a custom dictionary and the
  // file's own `_chem_comp_bond`.
  const resBondLookup = new Map();
  for (const [res, t] of Object.entries(CCD_TEMPLATES)) {
    const m = new Map();
    for (const [i, j, bt] of t.bonds) {
      const a1 = t.atoms[i][0];
      const a2 = t.atoms[j][0];
      m.set(a1 < a2 ? `${a1}:${a2}` : `${a2}:${a1}`, bt);
    }
    resBondLookup.set(res, m);
  }
  if (customBondDict) {
    for (const [res, pairs] of Object.entries(customBondDict)) {
      if (!resBondLookup.has(res)) resBondLookup.set(res, new Map());
      const m = resBondLookup.get(res);
      for (const [a1, a2, bt] of pairs) {
        const k = a1 < a2 ? `${a1}:${a2}` : `${a2}:${a1}`;
        m.set(k, bt);
      }
    }
  }
  if (cifBlock && cifBlock.chem_comp_bond) {
    const ccb = cifBlock.chem_comp_bond;
    const compIds = ccb.comp_id || [];
    const atom1s = ccb.atom_id_1 || [];
    const atom2s = ccb.atom_id_2 || [];
    const orders = ccb.value_order || [];
    const aroms = ccb.pdbx_aromatic_flag || [];
    for (let i = 0; i < compIds.length; i++) {
      const res = compIds[i];
      if (!resBondLookup.has(res)) resBondLookup.set(res, new Map());
      const m = resBondLookup.get(res);
      const a1 = atom1s[i];
      const a2 = atom2s[i];
      const k = a1 < a2 ? `${a1}:${a2}` : `${a2}:${a1}`;
      const orderKey = `${(orders[i] || 'SING').toUpperCase()}:${(aroms[i] || 'N').toUpperCase()}`;
      const bt = COMP_BOND_ORDER_TO_TYPE[orderKey] ?? BOND_TYPE.ANY;
      m.set(k, bt);
    }
  }

  const resStarts = getResidueSegments(atoms);
  const atomToRes = new Int32Array(n);
  for (let r = 0; r < resStarts.length - 1; r++) {
    for (let i = resStarts[r]; i < resStarts[r + 1]; i++) atomToRes[i] = r;
  }

  // 1. `add_bonds_from_struct_conn`: covalent links and metal coordination, an unset order inferred
  // from the two reactants (`infer_link_order`; a component the dictionary lacks keeps an unknown
  // order when the structure is not being prepared, and is refused when it is).
  const linkRows = [];
  if (cifBlock && cifBlock.struct_conn) {
    const sc = cifBlock.struct_conn;
    const nRows = (sc.conn_type_id || []).length;
    const hasTid = Array.isArray(atoms.transformation_id);
    const useTid = hasTid && sc.ptnr1_transformation_id && sc.ptnr2_transformation_id;

    for (let row = 0; row < nRows; row++) {
      // `add_bond_types_from_struct_conn=("covale", "metalc")` in both of upstream's presets: a
      // metal's coordination is a bond there (retyped to COORDINATION below), and was dropped here.
      const connType = sc.conn_type_id[row];
      if (connType !== 'covale' && connType !== 'metalc') continue;
      if (sc.ptnr1_symmetry && sc.ptnr1_symmetry[row] !== '1_555') continue;
      if (sc.ptnr2_symmetry && sc.ptnr2_symmetry[row] !== '1_555') continue;

      const findPartner = (p) => {
        const asym = sc[`ptnr${p}_label_asym_id`]?.[row] ?? '';
        const comp = sc[`ptnr${p}_label_comp_id`]?.[row] ?? '';
        let seq = sc[`ptnr${p}_label_seq_id`]?.[row] ?? '';
        if (seq === '.' && sc[`ptnr${p}_auth_seq_id`]) {
          seq = sc[`ptnr${p}_auth_seq_id`][row];
        }
        const seqInt = safeInt(seq, -999999);
        const atomName = sc[`ptnr${p}_label_atom_id`]?.[row] ?? '';
        const tid = useTid ? sc[`ptnr${p}_transformation_id`][row] : null;

        let match = -1;
        for (let i = 0; i < n; i++) {
          if (
            atoms.chain_id[i] === asym &&
            atoms.res_name[i] === comp &&
            atoms.res_id[i] === seqInt &&
            atoms.atom_name[i] === atomName &&
            (!useTid || atoms.transformation_id[i] === tid)
          ) {
            match = i;
          }
        }
        return match;
      };

      const idx1 = findPartner(1);
      const idx2 = findPartner(2);
      if (idx1 >= 0 && idx2 >= 0 && idx1 !== idx2) {
        let bt;
        if (connType === 'metalc') bt = BOND_TYPE.COORDINATION;
        else {
          // The order as written: upstream matches the mmCIF vocabulary exactly (lower case).
          const order = sc.pdbx_value_order?.[row] ?? '?';
          if (Object.hasOwn(STRUCT_CONN_ORDER_TO_TYPE, order)) bt = STRUCT_CONN_ORDER_TO_TYPE[order];
          else if (order === '?' || order === '.' || order === '') {
            bt = inferLinkOrder(atoms, idx1, idx2, dict, { allowMissingTemplates: !sanitize });
          } else throw new Error(`Unsupported struct_conn bond order: '${order}'`);
        }
        // Chemical preparation needs concrete orders.
        if (sanitize && bt === BOND_TYPE.ANY) bt = BOND_TYPE.SINGLE;
        linkRows.push([idx1, idx2, bt]);
      }
    }
  }

  // 2. Each residue's own bonds, merged over the links (the dictionary's type wins a repeat).
  const residueRows = [];
  for (let r = 0; r < resStarts.length - 1; r++) {
    const s = resStarts[r];
    const e = resStarts[r + 1];
    const m = resBondLookup.get(atoms.res_name[s]);
    if (!m) continue;
    for (let i = s; i < e; i++) {
      const ai = atoms.atom_name[i];
      for (let j = i + 1; j < e; j++) {
        const aj = atoms.atom_name[j];
        const bt = m.get(ai < aj ? `${ai}:${aj}` : `${aj}:${ai}`);
        if (bt !== undefined) residueRows.push([i, j, bt]);
      }
    }
  }
  let bonds = new BondRows([...residueRows, ...(atoms.bonds ?? []), ...linkRows.filter(([u, v]) => !residueRows.some(
    ([a, b]) => (a === Math.min(u, v) && b === Math.max(u, v))))]);

  // 3. `add_polymer_bonds`, merged in front.
  const current = bonds.list().map(([u, v, bt]) => [u, v, bt]);
  const polymer = polymerBondsToAdd(atoms, current, resStarts, atomToRes, dict);
  bonds = new BondRows([...polymer.map(([u, v]) => [u, v, BOND_TYPE.SINGLE]), ...current]);
  atoms.bonds = bonds.list().map(([u, v, bt]) => [u, v, bt]);

  // 4. The authored links' products (`resolve_link_chemistry`), then the hydrogens go.
  let out = atoms;
  if (sanitize) {
    out = resolveLinkChemistry(atoms, dict, keepAtoms);
    out = keepAtoms(out, out.element.map((e) => !HYDROGEN_ELEMENTS.has(e)));
  }

  // 5. Retype any metal-incident bond to COORDINATION.
  out.bonds = out.bonds.map(([u, v, bt]) => {
    const eu = (out.element[u] || '').toUpperCase();
    const ev = (out.element[v] || '').toUpperCase();
    return METAL_ELEMENTS.has(eu) || METAL_ELEMENTS.has(ev) ? [u, v, BOND_TYPE.COORDINATION] : [u, v, bt];
  });
  return out;
}

const HYDROGEN_ELEMENTS = new Set(['H', 'H2', 'D', 'T']);

/** The array without the atoms `keep` is false for: every per-atom field and the bonds, reindexed. */
function keepAtoms(atoms, keep) {
  if (keep.every(Boolean)) return atoms;
  const n = atoms.nAtoms;
  const index = new Int32Array(n).fill(-1);
  let k = 0;
  for (let i = 0; i < n; i++) if (keep[i]) index[i] = k++;
  const pick = (arr) => arr.filter((_, i) => keep[i]);
  for (const [key, v] of Object.entries(atoms)) {
    if (key !== 'bonds' && Array.isArray(v) && v.length === n) atoms[key] = pick(v);
  }
  for (const [key, v] of Object.entries(atoms.annotations ?? {})) {
    if (Array.isArray(v) && v.length === n) atoms.annotations[key] = pick(v);
  }
  atoms.bonds = (atoms.bonds ?? []).filter(([u, v]) => keep[u] && keep[v]).map(([u, v, bt]) => [index[u], index[v], bt]);
  atoms.nAtoms = k;
  return atoms;
}

/**
 * Create an empty AtomArray (0 atoms) for pure de novo contigs without an input structure.
 */
export function createEmptyAtomArray() {
  return {
    nAtoms: 0,
    hetero: [],
    element: [],
    atom_name: [],
    res_name: [],
    chain_id: [],
    label_entity_id: [],
    res_id: [],
    ins_code: [],
    auth_seq_id: [],
    atom_id: [],
    b_factor: [],
    occupancy: [],
    charge: [],
    label_alt_id: [],
    coord: [],
    transformation_id: [],
    is_polymer: [],
    chain_type: [],
    chem_comp_type: [],
    atomic_number: [],
    is_aromatic: [],
    is_leaving_atom: [],
    pn_unit_id: [],
    molecule_id: [],
    chain_entity: [],
    pn_unit_entity: [],
    molecule_entity: [],
    chain_iid: [],
    pn_unit_iid: [],
    molecule_iid: [],
    bonds: [],
    annotations: {},
  };
}

/**
 * Convert standard PDB `ATOM  ` / `HETATM` lines into a minimal mmCIF `_atom_site` loop.
 */
export function pdbTextToCifText(pdbText) {
  const lines = pdbText.split(/\r?\n/);
  const rows = [];
  let serial = 0;
  for (const line of lines) {
    if (line.startsWith('ENDMDL')) break;
    if (!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) continue;
    serial += 1;
    const group = line.slice(0, 6).trim() || 'ATOM';
    const atomName = line.slice(12, 16).trim() || 'CA';
    const altId = line.slice(16, 17).trim() || '.';
    const resName = line.slice(17, 20).trim() || 'ALA';
    const chainId = line.slice(21, 22).trim() || 'A';
    const resId = line.slice(22, 26).trim() || '1';
    const insCode = line.slice(26, 27).trim() || '.';
    const x = line.slice(30, 38).trim() || '0.0';
    const y = line.slice(38, 46).trim() || '0.0';
    const z = line.slice(46, 54).trim() || '0.0';
    const occ = line.slice(54, 60).trim() || '1.00';
    const bfac = line.slice(60, 66).trim() || '0.00';
    let elem = line.slice(76, 78).trim();
    if (!elem) elem = atomName.replace(/[0-9']/g, '').slice(0, 1) || 'C';
    const qAtom = atomName.includes("'") ? `"${atomName}"` : atomName;
    rows.push(
      `${group} ${serial} ${elem.toUpperCase()} ${qAtom} ${altId} ${resName} ${chainId} 1 ${resId} ${insCode} ${x} ${y} ${z} ${occ} ${bfac} 0 ${resId} ${resName} ${chainId} ${qAtom} 1`
    );
  }
  return [
    'data_pdb',
    'loop_',
    '_atom_site.group_PDB',
    '_atom_site.id',
    '_atom_site.type_symbol',
    '_atom_site.label_atom_id',
    '_atom_site.label_alt_id',
    '_atom_site.label_comp_id',
    '_atom_site.label_asym_id',
    '_atom_site.label_entity_id',
    '_atom_site.label_seq_id',
    '_atom_site.pdbx_PDB_ins_code',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.occupancy',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.pdbx_formal_charge',
    '_atom_site.auth_seq_id',
    '_atom_site.auth_comp_id',
    '_atom_site.auth_asym_id',
    '_atom_site.auth_atom_id',
    '_atom_site.pdbx_PDB_model_num',
    ...rows,
    '#',
  ].join('\n');
}

/**
 * Parse a CIF string, PDB string, or Zstandard-compressed CIF bytes into an AtomArray object.
 */
/** Row view of a parsed CIF category (columns may be scalars when the category has one row). */
export const catRows = (cat) => {
  const cols = cat ? Object.keys(cat) : [];
  const n = cols.length ? (Array.isArray(cat[cols[0]]) ? cat[cols[0]].length : 1) : 0;
  return { cols, n, get: (c, r) => (Array.isArray(cat[c]) ? cat[c][r] : cat[c]) };
};

/**
 * Append the fetched components' `_chem_comp_atom` / `_chem_comp_bond` rows to `cifBlock` (in place), so bond
 * building and the Studio export see them exactly as if the file had carried them. Columns missing on either side
 * are filled with '?'.
 */
export function mergeCcdIntoBlock(cifBlock, blocks) {
  for (const cat of ['chem_comp_atom', 'chem_comp_bond']) {
    const parts = [...blocks.values()].map((b) => catRows(b[cat])).filter((p) => p.n > 0);
    if (!parts.length) continue;
    const base = catRows(cifBlock[cat]);
    const cols = [...new Set([...base.cols, ...parts.flatMap((p) => p.cols)])];
    const out = Object.fromEntries(cols.map((c) => [c, []]));
    for (const src of [{ part: base, cat: cifBlock[cat] }, ...parts.map((p) => ({ part: p }))]) {
      for (let r = 0; r < src.part.n; r += 1) {
        for (const c of cols) out[c].push(src.part.cols.includes(c) ? String(src.part.get(c, r)) : '?');
      }
    }
    cifBlock[cat] = out;
  }
  return cifBlock;
}

/**
 * A structure file as a parsed block: mmCIF text, BinaryCIF, or a PDB file converted - plain, gzip
 * or Zstandard, as upstream takes one. `fromPdb` says it was a PDB file, whose insertion codes are
 * its numbering.
 * @returns {{block: object, fromPdb: boolean}}
 */
export function readStructureBlock(input) {
  if (typeof input !== 'string') {
    const bytes = unwrapBytes(input);
    if (isBinaryCif(bytes)) return { block: parseBinaryCifBlock(bytes), fromPdb: false };
    input = new TextDecoder('utf-8').decode(bytes);
  }
  const fromPdb = !input.includes('_atom_site.') && /(?:^|\n)(?:ATOM  |HETATM)/.test(input);
  return { block: parseCifBlock(fromPdb ? pdbTextToCifText(input) : input), fromPdb };
}

export function parseCifToAtomArray(cifInput, opts = {}) {
  // A parsed block is taken as it is (fresh.js hands one over); text or bytes are read here.
  const { block, fromPdb } = opts.block ? { block: opts.block, fromPdb: false } : readStructureBlock(cifInput);
  // Fetched Chemical Component Dictionary entries (`ccd.js`) for components the file carries no tables for.
  if (opts.ccdBlocks?.size) mergeCcdIntoBlock(block, opts.ccdBlocks);
  const atomSite = block.atom_site;
  if (!atomSite) {
    throw new Error('CIF block does not contain _atom_site category');
  }

  const nRows = (atomSite.label_atom_id || atomSite.auth_atom_id || []).length;

  // The first model, and atomworks' `altloc="first"`: per chain, the alphabetically first
  // alternate-location LETTER, with every atom that has none. An atom whose only conformer is
  // another letter is dropped, and a non-letter location (`1`, `2`) is never chosen - both
  // upstream's, where this kept the first conformer of each ATOM in file order.
  const modelNums = atomSite.pdbx_PDB_model_num;
  const firstModelNum = modelNums && modelNums.length > 0 ? modelNums[0] : null;
  const altIds = atomSite.label_alt_id;
  const isDefaultAlt = (alt) => alt === undefined || alt === '.' || alt === '?' || alt === ' ' || alt === '';
  const chainOf = (i) => atomSite.label_asym_id?.[i] || atomSite.auth_asym_id?.[i] || 'A';
  const chosenAlt = new Map();
  if (altIds) {
    for (let i = 0; i < nRows; i++) {
      if (firstModelNum !== null && modelNums[i] !== firstModelNum) continue;
      const alt = altIds[i];
      if (isDefaultAlt(alt) || !/^[A-Za-z]+$/.test(alt)) continue;
      const cur = chosenAlt.get(chainOf(i));
      if (cur === undefined || alt < cur) chosenAlt.set(chainOf(i), alt);
    }
  }
  const selectedRows = [];
  for (let i = 0; i < nRows; i++) {
    if (firstModelNum !== null && modelNums[i] !== firstModelNum) continue;
    const alt = altIds ? altIds[i] : '.';
    if (!isDefaultAlt(alt) && alt !== chosenAlt.get(chainOf(i))) continue;
    selectedRows.push(i);
  }

  const nAtoms = selectedRows.length;

  // Extract _chem_comp metadata
  const compTypeMap = new Map();
  if (block.chem_comp && block.chem_comp.id) {
    const ids = block.chem_comp.id;
    const types = block.chem_comp.type || block.chem_comp.pdbx_type || [];
    for (let i = 0; i < ids.length; i++) {
      if (types[i] && types[i] !== '?' && types[i] !== '.') {
        compTypeMap.set(ids[i], types[i].toUpperCase());
      }
    }
  }

  // Extract _chem_comp_atom metadata (aromatic, leaving)
  const ccaAromMap = new Map();
  const ccaLeavingMap = new Map();
  if (block.chem_comp_atom && block.chem_comp_atom.comp_id) {
    const cids = block.chem_comp_atom.comp_id;
    const aids = block.chem_comp_atom.atom_id;
    const aroms = block.chem_comp_atom.pdbx_aromatic_flag;
    const leavs = block.chem_comp_atom.pdbx_leaving_atom_flag;
    for (let i = 0; i < cids.length; i++) {
      const key = `${cids[i]}:${aids[i]}`;
      // An unset flag says nothing: a raw RCSB file's `_chem_comp_atom` has no flag columns, and a
      // fetched entry merged into it pads them with '?' - which read as "not leaving" took every
      // terminal OXT's flag away.
      if (aroms && (aroms[i] === 'Y' || aroms[i] === 'N')) ccaAromMap.set(key, aroms[i] === 'Y');
      if (leavs && (leavs[i] === 'Y' || leavs[i] === 'N')) ccaLeavingMap.set(key, leavs[i] === 'Y');
    }
  }

  // Extract _entity and _entity_poly for chain_info
  const entityTypeMap = new Map();
  if (block.entity && block.entity.id && block.entity.type) {
    for (let i = 0; i < block.entity.id.length; i++) {
      entityTypeMap.set(block.entity.id[i], block.entity.type[i].toLowerCase());
    }
  }
  const entityPolyTypeMap = new Map();
  const chainToPolyType = new Map();
  if (block.entity_poly && block.entity_poly.entity_id) {
    const eids = block.entity_poly.entity_id;
    const ptypes = block.entity_poly.type || [];
    const strands = block.entity_poly.pdbx_strand_id || [];
    for (let i = 0; i < eids.length; i++) {
      const pt = (ptypes[i] || '').toLowerCase();
      entityPolyTypeMap.set(eids[i], pt);
      if (strands[i]) {
        for (const s of strands[i].split(',')) {
          const cid = s.trim();
          if (cid) chainToPolyType.set(cid, pt);
        }
      }
    }
  }

  const atoms = {
    nAtoms,
    hetero: new Array(nAtoms),
    element: new Array(nAtoms),
    atom_name: new Array(nAtoms),
    res_name: new Array(nAtoms),
    chain_id: new Array(nAtoms),
    label_entity_id: new Array(nAtoms),
    res_id: new Array(nAtoms),
    ins_code: new Array(nAtoms),
    auth_seq_id: new Array(nAtoms),
    atom_id: new Array(nAtoms),
    b_factor: new Array(nAtoms),
    occupancy: new Array(nAtoms),
    charge: new Array(nAtoms),
    label_alt_id: new Array(nAtoms),
    coord: new Array(nAtoms),
    transformation_id: new Array(nAtoms).fill('1'),
    is_polymer: new Array(nAtoms),
    chain_type: new Array(nAtoms),
    chem_comp_type: new Array(nAtoms),
    atomic_number: new Array(nAtoms),
    is_aromatic: new Array(nAtoms),
    is_leaving_atom: new Array(nAtoms),
    bonds: [],
    annotations: {},
    // Upstream's selection namespace has no `b_factor` when the file had no B column; this
    // reader fills NaN either way, so the absence is recorded rather than inferred.
    noBFactorColumn: !atomSite.B_iso_or_equiv,
  };

  for (let k = 0; k < nAtoms; k++) {
    const r = selectedRows[k];
    atoms.hetero[k] = (atomSite.group_PDB?.[r] || 'ATOM') === 'HETATM';
    atoms.element[k] = atomSite.type_symbol?.[r] || 'C';
    atoms.atom_name[k] = atomSite.label_atom_id?.[r] || atomSite.auth_atom_id?.[r] || '';
    atoms.res_name[k] = atomSite.label_comp_id?.[r] || atomSite.auth_comp_id?.[r] || '';
    atoms.chain_id[k] = atomSite.label_asym_id?.[r] || atomSite.auth_asym_id?.[r] || 'A';
    atoms.label_entity_id[k] = safeInt(atomSite.label_entity_id?.[r], 0);
    atoms.res_id[k] = safeInt(atomSite.label_seq_id?.[r], -1);
    // atomworks reads no insertion code from an mmCIF (`ins_code[:] = ""`): a CIF numbers its
    // residues uniquely in `label_seq_id`. A PDB file's is kept, since there it is the numbering.
    const ins = fromPdb ? atomSite.pdbx_PDB_ins_code?.[r] : '';
    atoms.ins_code[k] = ins === '.' || ins === '?' || !ins ? '' : ins;
    atoms.auth_seq_id[k] = safeInt(atomSite.auth_seq_id?.[r], atoms.res_id[k]);
    atoms.atom_id[k] = safeInt(atomSite.id?.[r], k + 1);
    atoms.b_factor[k] = safeFloat(atomSite.B_iso_or_equiv?.[r], NaN);
    atoms.occupancy[k] = safeFloat(atomSite.occupancy?.[r], 1.0);
    atoms.label_alt_id[k] = atomSite.label_alt_id?.[r] || '.';
    atoms.coord[k] = [
      safeFloat(atomSite.Cartn_x?.[r], NaN),
      safeFloat(atomSite.Cartn_y?.[r], NaN),
      safeFloat(atomSite.Cartn_z?.[r], NaN),
    ];

    const ccaKey = `${atoms.res_name[k]}:${atoms.atom_name[k]}`;
    // 🔴 THE FILE'S CHARGE, AND NOTHING ELSE. This filled a blank one from the file's own
    // `_chem_comp_atom` and then from a table of standard-residue charges (LYS NZ +1, HIS ND1
    // +1, ...); upstream's annotated reader keeps the column as written, so a raw RCSB file read
    // as annotated took 13-69 charges atomworks does not give it. A raw structure's charges are
    // the dictionary's, and fresh.js writes them in explicitly.
    atoms.charge[k] = safeInt(atomSite.pdbx_formal_charge?.[r], 0);
    atoms.is_aromatic[k] = ccaAromMap.has(ccaKey)
      ? ccaAromMap.get(ccaKey)
      : Boolean(STD_RESIDUE_AROMATIC[atoms.res_name[k]]?.has(atoms.atom_name[k]));
    atoms.is_leaving_atom[k] = ccaLeavingMap.has(ccaKey)
      ? ccaLeavingMap.get(ccaKey)
      : (AA_RESIDUES.has(atoms.res_name[k]) && atoms.atom_name[k] === 'OXT') ||
        ((RNA_RESIDUES.has(atoms.res_name[k]) || DNA_RESIDUES.has(atoms.res_name[k])) && atoms.atom_name[k] === 'OP3');

    const elUpper = atoms.element[k].toUpperCase();
    atoms.atomic_number[k] = ELEMENT_TO_ATOMIC_NUMBER[elUpper] ?? 0;

    let cct = compTypeMap.get(atoms.res_name[k]);
    if (!cct) {
      if (atoms.res_name[k] === 'GLY') cct = 'PEPTIDE LINKING';
      else if (AA_RESIDUES.has(atoms.res_name[k])) cct = 'L-PEPTIDE LINKING';
      else if (DNA_RESIDUES.has(atoms.res_name[k])) cct = 'DNA LINKING';
      else if (RNA_RESIDUES.has(atoms.res_name[k])) cct = 'RNA LINKING';
      else cct = 'NON-POLYMER';
    }
    atoms.chem_comp_type[k] = cct;
  }

  // Determine chain_type and is_polymer per chain_id
  const uniqueChains = Array.from(new Set(atoms.chain_id));
  const chainTypeByChain = new Map();
  for (const cid of uniqueChains) {
    const firstIdx = atoms.chain_id.indexOf(cid);
    const eidStr = String(atoms.label_entity_id[firstIdx]);
    let polyStr = chainToPolyType.get(cid) || entityPolyTypeMap.get(eidStr) || entityTypeMap.get(eidStr);
    let ct = polyStr ? MMCIF_CHAIN_TYPE_MAP[polyStr] : undefined;
    if (ct === undefined) {
      const resName = atoms.res_name[firstIdx];
      if (DNA_RESIDUES.has(resName)) ct = CHAIN_TYPE.DNA;
      else if (RNA_RESIDUES.has(resName)) ct = CHAIN_TYPE.RNA;
      else if (AA_RESIDUES.has(resName) || /^\d+-\d+$/.test(resName) || resName === '<M>') ct = CHAIN_TYPE.POLYPEPTIDE_L;
      else ct = CHAIN_TYPE.NON_POLYMER;
    }
    chainTypeByChain.set(cid, ct);
  }

  for (let k = 0; k < nAtoms; k++) {
    const ct = chainTypeByChain.get(atoms.chain_id[k]);
    atoms.chain_type[k] = ct;
    atoms.is_polymer[k] = ct <= CHAIN_TYPE.RNA;
    // update_nonpoly_seq_ids: non-polymer chains use auth_seq_id for res_id
    if (!atoms.is_polymer[k]) {
      atoms.res_id[k] = atoms.auth_seq_id[k];
    }
  }

  // Optional fix_arginines (used when loading unannotated structures with RCSB_WITH_METALS)
  if (opts.fixArginines) {
    const resStarts = getResidueSegments(atoms);
    for (let r = 0; r < resStarts.length - 1; r++) {
      const s = resStarts[r];
      const e = resStarts[r + 1];
      if (atoms.res_name[s] !== 'ARG') continue;
      let iCD = -1, iNH1 = -1, iNH2 = -1;
      for (let i = s; i < e; i++) {
        if (atoms.atom_name[i] === 'CD') iCD = i;
        else if (atoms.atom_name[i] === 'NH1') iNH1 = i;
        else if (atoms.atom_name[i] === 'NH2') iNH2 = i;
      }
      if (iCD >= 0 && iNH1 >= 0 && iNH2 >= 0) {
        const cCD = atoms.coord[iCD];
        const c1 = atoms.coord[iNH1];
        const c2 = atoms.coord[iNH2];
        const d1 = Math.hypot(cCD[0] - c1[0], cCD[1] - c1[1], cCD[2] - c1[2]);
        const d2 = Math.hypot(cCD[0] - c2[0], cCD[1] - c2[1], cCD[2] - c2[2]);
        if (Number.isFinite(d1) && Number.isFinite(d2) && d1 > d2) {
          atoms.coord[iNH1] = c2;
          atoms.coord[iNH2] = c1;
        }
      }
    }
  }

  // Load 1-body scalar StandardAnnotations from _atom_site
  for (const [colName, colValues] of Object.entries(atomSite)) {
    let spec = SCALAR_1BODY_ANNOTATIONS[colName];
    if (!spec && colName.startsWith('condition_sequence-probability-')) {
      spec = { type: 'float', default: NaN };
    }
    if (!spec && colName.startsWith('mask_sequence-probability-')) {
      spec = { type: 'bool', default: false };
    }
    if (!spec && colName === 'condition_cath-label_1_residue' && !block['condition_cath-label_1_residue']) {
      spec = { type: 'str', default: 'x.x.x.x' };
    }
    if (!spec) continue;

    const arr = new Array(nAtoms);
    for (let k = 0; k < nAtoms; k++) {
      const raw = colValues[selectedRows[k]];
      if (spec.type === 'int') arr[k] = safeInt(raw, spec.default);
      else if (spec.type === 'float') arr[k] = safeFloat(raw, spec.default);
      else if (spec.type === 'bool') arr[k] = safeBool(raw, spec.default);
      else arr[k] = raw === '.' || raw === '?' ? spec.default : raw;
    }
    atoms.annotations[colName] = arr;
  }

  // Load nonscalar 1-body StandardAnnotations from dedicated CIF categories
  for (const [catName, spec] of Object.entries(NONSCALAR_1BODY_ANNOTATIONS)) {
    const cat = block[catName];
    if (!cat) continue;
    const arr = Array.from({ length: nAtoms }, () => new Array(spec.cols).fill(spec.default));
    const idx0 = cat.idx0;
    const nCatRows = idx0 ? idx0.length : (cat.val0 ? cat.val0.length : 0);
    for (let r = 0; r < nCatRows; r++) {
      const atomIdx = idx0 ? safeInt(idx0[r], -1) : r;
      if (atomIdx < 0 || atomIdx >= nAtoms) continue;
      const rowVals = new Array(spec.cols);
      for (let c = 0; c < spec.cols; c++) {
        const raw = cat[`val${c}`]?.[r];
        rowVals[c] = spec.type === 'int' ? safeInt(raw, spec.default) : safeFloat(raw, spec.default);
      }
      arr[atomIdx] = rowVals;
    }
    atoms.annotations[catName] = arr;
  }

  // The base pairs: `_condition_basepair_2_residue`, atom indices `idx0`/`idx1` and a value. The
  // condition is SYMMETRIC, and upstream's reader stores each row in both directions (W05: 37
  // rows, 74 pairs), so each atom lists the other.
  const bp = block['condition_basepair_2_residue'];
  if (bp && bp.idx0 && bp.idx1) {
    const partners = Array.from({ length: nAtoms }, () => []);
    for (let r = 0; r < bp.idx0.length; r++) {
      const i = safeInt(bp.idx0[r], -1);
      const j = safeInt(bp.idx1[r], -1);
      if (i < 0 || j < 0 || i >= nAtoms || j >= nAtoms) continue;
      partners[i].push(j);
      partners[j].push(i);
    }
    atoms.annotations['annotation_src-atom-index_1_atom'] = Array.from({ length: nAtoms }, (_, k) => k);
    atoms.annotations['condition_basepair-partners_1_atom'] = partners.map((list) => list.join(','));
  }

  // Handle legacy masks (mask_sequence_1_residue, mask_coordinate_1_atom, mask_index_1_residue)
  if (atoms.annotations['mask_index_1_residue']) {
    const mask = atoms.annotations['mask_index_1_residue'];
    atoms.annotations['condition_index_1_residue'] = mask.map(Boolean);
    delete atoms.annotations['mask_index_1_residue'];
  }
  if (atoms.annotations['mask_sequence_1_residue']) {
    const mask = atoms.annotations['mask_sequence_1_residue'];
    const existing = atoms.annotations['condition_sequence_1_residue'];
    atoms.annotations['condition_sequence_1_residue'] = mask.map((m, i) =>
      m ? (existing && existing[i] !== '<M>' ? existing[i] : atoms.res_name[i]) : '<M>'
    );
    delete atoms.annotations['mask_sequence_1_residue'];
  }
  if (atoms.annotations['mask_coordinate_1_atom']) {
    const mask = atoms.annotations['mask_coordinate_1_atom'];
    const existing = atoms.annotations['condition_coordinate_1_atom'];
    atoms.annotations['condition_coordinate_1_atom'] = mask.map((m, i) =>
      m ? (existing ? existing[i] : atoms.coord[i].slice()) : [NaN, NaN, NaN]
    );
    delete atoms.annotations['mask_coordinate_1_atom'];
  }

  const built = buildBondsForAtomArray(atoms, block, null, { sanitize: opts.sanitize, ccdBlocks: opts.ccdBlocks });
  computeIdAndEntityAnnotations(built);

  return built;
}
