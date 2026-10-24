/**
 * RFD4-Proteina canonical 147-class UNIFIED_ATOM37_ENCODING vocabulary
 * and chemical metadata.
 *
 * Matches `rfproteina/encoding.py`, `rfproteina/resources/aa_rna_dna_glycan_ncaa.yaml`,
 * `rfproteina/utils/constants.py`, and `rfproteina/partial_autoencoder/decoder.py`.
 */

export const TokenType = Object.freeze({
  PROTEIN: 0,
  POLYPEPTIDE_L: 0,
  RNA: 1,
  DNA: 2,
  ATOMIZED: 3,
  UNKNOWN: 4,
  GLYCAN: 5,
  NCAA: 6,
});

export const NUM_TOKEN_TYPES = 7;

export const TOKEN_TYPE_IDS = Object.freeze({
  aa: TokenType.PROTEIN,
  rna: TokenType.RNA,
  dna: TokenType.DNA,
  atomized: TokenType.ATOMIZED,
  unknown: TokenType.UNKNOWN,
  glycan: TokenType.GLYCAN,
  ncaa: TokenType.NCAA,
  mask: TokenType.UNKNOWN,
});

/**
 * Exact ordered 147-residue specification from `resources/aa_rna_dna_glycan_ncaa.yaml`.
 * Order of groups and entries determines class indices 0..146 in the checkpoint.
 */
export const RESIDUE_LAYOUTS_BY_GROUP = [
  ["mask", [
    ["<M>", "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"],
  ]],
  ["aa", [
    ["ALA", "N CA C CB O - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["ARG", "N CA C CB O CG - - - - - CD - - - - - - - - - - - NE - - - - - NH1 NH2 - CZ - - - OXT"],
    ["ASN", "N CA C CB O CG - - - - - - - - - ND2 OD1 - - - - - - - - - - - - - - - - - - - OXT"],
    ["ASP", "N CA C CB O CG - - - - - - - - - - OD1 OD2 - - - - - - - - - - - - - - - - - - OXT"],
    ["CYS", "N CA C CB O - - - - - SG - - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["GLN", "N CA C CB O CG - - - - - CD - - - - - - - - - - - - - NE2 OE1 - - - - - - - - - OXT"],
    ["GLU", "N CA C CB O CG - - - - - CD - - - - - - - - - - - - - - OE1 OE2 - - - - - - - - OXT"],
    ["GLY", "N CA C - O - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["HIS", "N CA C CB O CG - - - - - - - CD2 ND1 - - - - - CE1 - - - - NE2 - - - - - - - - - - OXT"],
    ["ILE", "N CA C CB O - CG1 CG2 - - - - CD1 - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["LEU", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["LYS", "N CA C CB O CG - - - - - CD - - - - - - - CE - - - - - - - - - - - - - - - NZ OXT"],
    ["MET", "N CA C CB O CG - - - - - - - - - - - - SD CE - - - - - - - - - - - - - - - - OXT"],
    ["PHE", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - CE1 CE2 - - - - - - - - - - CZ - - - OXT"],
    ["PRO", "N CA C CB O CG - - - - - CD - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["SER", "N CA C CB O - - - OG - - - - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["THR", "N CA C CB O - - CG2 - OG1 - - - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["TRP", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - - CE2 CE3 - NE1 - - - CH2 - - - - CZ2 CZ3 - OXT"],
    ["TYR", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - CE1 CE2 - - - - - - - - - OH CZ - - - OXT"],
    ["VAL", "N CA C CB O - CG1 CG2 - - - - - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["UNK", "N CA C - O - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"],
  ]],
  ["rna", [
    ["A", "P C1' C2' O2' C3' O3' C4' O4' C5' O5' OP1 OP2 N9 C8 N7 C5 C4 N3 C2 N1 C6 N6 - - - - - - - - - - - - - - OP3"],
    ["C", "P C1' C2' O2' C3' O3' C4' O4' C5' O5' OP1 OP2 - - - - - - - - - - - - N1 C2 O2 N3 C4 C5 C6 N4 - - - - OP3"],
    ["G", "P C1' C2' O2' C3' O3' C4' O4' C5' O5' OP1 OP2 N9 C8 N7 C5 C4 N3 C2 N1 C6 - N2 O6 - - - - - - - - - - - - OP3"],
    ["U", "P C1' C2' O2' C3' O3' C4' O4' C5' O5' OP1 OP2 - - - - - - - - - - - - N1 C2 O2 N3 C4 C5 C6 - O4 - - - OP3"],
    ["N", "P C1' C2' O2' C3' O3' C4' O4' C5' O5' OP1 OP2 - - - - - - - - - - - - - - - - - - - - - - - - -"],
  ]],
  ["dna", [
    ["DA", "P C1' C2' - C3' O3' C4' O4' C5' O5' OP1 OP2 N9 C8 N7 C5 C4 N3 C2 N1 C6 N6 - - - - - - - - - - - - - - OP3"],
    ["DC", "P C1' C2' - C3' O3' C4' O4' C5' O5' OP1 OP2 - - - - - - - - - - - - N1 C2 O2 N3 C4 C5 C6 N4 - - - - OP3"],
    ["DG", "P C1' C2' - C3' O3' C4' O4' C5' O5' OP1 OP2 N9 C8 N7 C5 C4 N3 C2 N1 C6 - N2 O6 - - - - - - - - - - - - OP3"],
    ["DT", "P C1' C2' - C3' O3' C4' O4' C5' O5' OP1 OP2 - - - - - - - - - - - - N1 C2 O2 N3 C4 C5 C6 - O4 C7 - - OP3"],
    ["DN", "P C1' C2' - C3' O3' C4' O4' C5' O5' OP1 OP2 - - - - - - - - - - - - - - - - - - - - - - - - -"],
  ]],
  ["atomized", [
    ["<A>", "- X - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"],
  ]],
  ["unknown", [
    ["<G>", "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"],
  ]],
  ["ncaa", [
    ["SEP", "N CA C CB O - - O1P OG - - - - - - - - - - - - - - - O2P - - - O3P P - - - - - - OXT"],
    ["TPO", "N CA C CB O - - CG2 - OG1 - - - - - O1P - - - - - - - - - - O2P - - O3P - - - - P - OXT"],
    ["CSO", "N CA C CB O - - - - - SG - - - - - - - - - - - - - - - - - - - - - - - - OD OXT"],
    ["PTR", "N CA C CB O CG - - - - - - CD1 CD2 O1P O2P - - O3P - CE1 CE2 - - - - - - - - P OH CZ - - - OXT"],
    ["CME", "N CA C CB O - - - - - SG - - - - - - - SD CE - - - - - - - - - - - OH CZ - - - OXT"],
    ["KCX", "N CA C CB O CG - - - - - CD - - CX - - - - CE - - - - - - - - - OQ1 OQ2 - - - - NZ OXT"],
    ["LLP", "N CA C CB O CG C2 - C2' C3 C4 CD - - C4' - - C5 C5' CE - - - - C6 N1 O3 OP1 OP2 OP3 - OP4 - P - NZ OXT"],
    ["CSD", "N CA C CB O - - - - - SG - - - - - OD1 OD2 - - - - - - - - - - - - - - - - - - OXT"],
    ["HIC", "N CA C CB O CG - - - - - - - CD2 ND1 - - - - - CE1 - - - - NE2 - - - - - - CZ - - - OXT"],
    ["ALY", "N CA C CB O CG CH - - - - CD - - - - - - - CE - - - - - - CH3 - - - - OH - - - NZ OXT"],
    ["HYP", "N CA C CB O CG - - - - - CD - - - - OD1 - - - - - - - - - - - - - - - - - - - OXT"],
    ["MLY", "N CA C CB O CG - - - - - CD - - - - - - - CE - - - - - - - - CH2 - - - - - CH1 NZ OXT"],
    ["TYS", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - CE1 CE2 - O1 O2 O3 - - - - - OH CZ - S - OXT"],
    ["M3L", "N CA C CB O CG CM1 - - - - CD - - - - - - - CE - - - - - - CM2 CM3 - - - - - - - NZ OXT"],
    ["OCS", "N CA C CB O - - - - - SG - - - - - OD1 OD2 - - - - - - OD3 - - - - - - - - - - - OXT"],
    ["CSS", "N CA C CB O - - - - - SG - - - - - - - SD - - - - - - - - - - - - - - - - - OXT"],
    ["CSX", "N CA C CB O - - - - - SG - - - - - - - - - - - - - - - - - - - OD - - - - - OXT"],
    ["CAS", "N CA C CB O - AS - - - SG - - - - - - - - - CE1 CE2 - - - - - - - - - - - - - - OXT"],
    ["AIB", "N CA C - O - - - - - - - - - - - - - - - - - - - - - CB1 - - CB2 - - - - - - OXT"],
    ["0TD", "N CA C CB O CG - - - CSB - - - - - - OD1 OD2 - - - - - - - SB - - - - - - - - - - OXT"],
    ["MLZ", "N CA C CB O CG - - - - - CD - - - - - - - CE - - - - - - - - CM - - - - - - NZ OXT"],
    ["NLE", "N CA C CB O CG - - - - - CD - - - - - - - CE - - - - - - - - - - - - - - - - OXT"],
    ["MEQ", "N CA C CB O CG - - - - - CD - - - - - - - CE - - - - - NE2 OE1 - - - - - - - - - OXT"],
    ["2MR", "N CA C CB O CG - - - - - CD - - - - - - CQ1 - - - - NE CQ2 - - - - NH1 NH2 - CZ - - - OXT"],
    ["4D4", "N CA C CB O CG - - OB - - CD - - - - - - - - - - - NE - - - - - NH1 NH2 - CZ - - - OXT"],
    ["D2T", "N CA C CB O CG - - CB1 - - - - - - - OD1 OD2 - - - - - - - - - - - - SB - - - - - OXT"],
    ["ABA", "N CA C CB O CG - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["YCM", "N CA C CB O - - - - - SG CD - - - - - - - CE - - - - - - - - - NZ2 - - - - OZ1 - OXT"],
    ["ORN", "N CA C CB O CG - - - - - CD - - - - - - - - - - - NE - - - - - - - - - - - - OXT"],
    ["SEC", "N CA C CB O - - - - - - - - - - SE - - - - - - - - - - - - - - - - - - - - OXT"],
    ["MEN", "N CA C CB O CG - - - - - - - - - ND2 OD1 - - - - CE2 - - - - - - - - - - - - - - OXT"],
    ["CGU", "N CA C CB O CG - - - - - - CD1 CD2 OE11 - - - - - - - OE12 OE21 - - - - - - - - - - OE22 - OXT"],
    ["KPI", "N CA C CB O CG - - C1 - - CD - - - CX1 - - - CE - - CX2 - - - - - O1 - - - - O2 - NZ OXT"],
    ["PHD", "N CA C CB O CG OP1 - - - - - - - - - OD1 OD2 - - - - - - - OP2 - - - - - - - OP3 P - OXT"],
    ["SMC", "N CA C CB O - - - CS - SG - - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["MHS", "N CA C CB O CG - - - - - - - CD2 ND1 - - CM - - CE1 - - - - NE2 - - - - - - - - - - OXT"],
    ["TPQ", "N CA C CB O - - - C1 - - - - - - - C2 C3 - - - - C4 - - - - C5 C6 - O2 - - O4 - O5 OXT"],
    ["NEP", "N CA C CB O CG - - - O1P - - - CD2 ND1 O2P - - - - CE1 - - O3P - NE2 - - - - - P - - - - OXT"],
    ["MK8", "N CA C CB O CG - - - - - CD - - - CB1 - - - CE - - - - - - - - - - - - - - - - OXT"],
    ["DPP", "N CA C CB O - - - - - - - - - NG - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["ALC", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - CE1 CE2 - - - - - - - - - - CZ - - - OXT"],
    ["CAF", "N CA C CB O - - - - - SG - - - - - - - - - CE1 CE2 - AS - - - O1 - - - - - - - - OXT"],
    ["DDE", "N CA C CB O CG - CAA CAB - - - - CD2 ND1 - - - CAC - CE1 - CAT CAU - NE2 CBI - - CBW - - - NAD NCB OAG OXT"],
    ["CIR", "N CA C - O - - C3 C4 - - - - - - C5 - - C7 - - - - - - - N6 N8 - - - - - O7 - - OXT"],
    ["PHI", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - CE1 CE2 - I - - - - - - - - CZ - - - OXT"],
    ["SCH", "N CA C CB O - - - - - SG - - - - - - - SD CE - - - - - - - - - - - - - - - - OXT"],
    ["DAB", "N CA C CB O CG - - ND - - - - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["SCY", "N CA C CB O - - - - - SG CD - - - - - - - CE - - - - - - OCD - - - - - - - - - OXT"],
    ["LYR", "N CA C CB O CG C1 C10 C11 C12 C13 CD - - C14 C15 C16 - C17 CE - - C18 C19 C2 C3 C4 C5 C6 - C7 C8 - C80 C9 NZ OXT"],
    ["NMM", "N CA C CB O CG - - - - - CD - - - - - - CAA - - - - NE - - - - - NH1 NH2 - CZ - - - OXT"],
    ["SNC", "N CA C CB O - - - - - SG - - - - - - - - - - - - - - - - ND OE - - - - - - - OXT"],
    ["SME", "N CA C CB O CG OE - - - - - - - - - - - - CE - - - - - - - - - - S - - - - - OXT"],
    ["TYQ", "N CA C CB O CG - - - - - - CD1 CD2 - N5 - - - - CE1 CE2 - - - - - - - OZ - OH CZ - - - OXT"],
    ["MHO", "N CA C CB O CG - - - - - - - - - - OD1 - SD CE - - - - - - - - - - - - - - - - OXT"],
    ["FTR", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - - CE2 CE3 - NE1 - - F CH2 - - - - CZ2 CZ3 - OXT"],
    ["TRQ", "N CA C CB O CG - O6 - - - - CD1 CD2 - - - - - - - CE2 CE3 - NE1 - O7 - CH2 - - - - CZ2 CZ3 - OXT"],
    ["EEP", "N CA C CB O CG - - - O1 - - CD1 CD2 - - - - - - - - - - O2 - - - - - - - - - - - OXT"],
    ["4HH", "N CA C CB O - CJ CK OG CL1 - - - - CL2 CL3 CM CO CP - - - CQ CS CT NN NR O1P O2P O3P OM ON - OR P SU OXT"],
    ["3FG", "N CA C CB O - CG1 CG2 - - - - CD1 CD2 - - OD1 OD2 - - - - - - - - - - - - - - CZ - - - OXT"],
    ["TOX", "N CA C CB O CG - - - - - - CD1 CD2 - O1 - - - - - CE2 CE3 - NE1 - O2 - CH2 - - - - CZ2 CZ3 - OXT"],
    ["AGM", "N CA C CB O CG - - - - - CD - - - - - - - - - CE2 - - NE1 - - - - NH1 NH2 - CZ - - - OXT"],
    ["CCS", "N CA C CB O - - - - OZ1 SG CD - - - - - - - CE - - OZ2 - - - - - - - - - - - - - OXT"],
    ["NIY", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - CE1 CE2 - NN - - O1 - - - O2 OH CZ - - - OXT"],
    ["V5N", "N CA C CB O CG - - - - - - - CD2 ND1 - - - O2 - CE1 - - - - NE2 - - - - - - - - - - OXT"],
    ["HY3", "N CA C - O - - - - - - - - - - C3 - - - - - - - C4 - - - C5 - - - O2 - - - - OXT"],
    ["OMY", "N CA C CB O CG - CL - - - - CD1 CD2 - - - - - - CE1 CE2 - - - - - - - OCZ - - CZ - - ODE OXT"],
    ["P1L", "N CA C CB O - C10 - - C11 SG - - - C12 C13 C14 - C15 - - - C16 C17 C18 C19 C20 C21 - C22 C7 C8 - C9 - O7 OXT"],
    ["4BF", "N CA C CB O CG - - - - - - CD1 CD2 - BR - - - - CE1 CE2 - - - - - - - - - - CZ - - - OXT"],
    ["MGN", "N CA C - O CG - - - - - CD - - CB1 - - - - - - - - CB2 - NE2 OE1 - - - - - - - - - OXT"],
    ["MP8", "N CA C CB O CG - - - - - CD - - - - - - - CE - - - - - - - - - - - - - - - - OXT"],
    ["004", "N CA C CB O - CG1 CG2 - - - - CD1 CD2 - - - - - CE - - - - - - - - - - - - - - - - OXT"],
    ["0AF", "N CA C CB O CG - - - O1 - - CD1 CD2 - - - - - - - CE2 CE3 - NE1 - - - CH2 - - - - CZ2 CZ3 - OXT"],
    ["5CT", "N CA C CB O CG C1 - C2 - - CD - - C3 - C4 - - CE - - - - N1 - - - - - - - - O1 - NZ OXT"],
    ["ASA", "N CA C CB O CG - - - - - - - - - - OD1 OD2"],
    ["IYR", "N CA C CB O CG - CC - - - CD - - - - - - - CE - - - - CF CH - - IE - - OF - - - - OXT"],
    ["TRX", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - - CE2 CE3 - NE1 OH2 - - CH2 - - - - CZ2 CZ3 - OXT"],
    ["5OH", "N CA C CB O - - - - CQ - - - - CR CS - NP - - - - - - NQ - - - NR - OS - - - - - OXT"],
    ["SEB", "N CA C CB O - - - OG - - - - - - CH1 OD1 OD2 SD CE - - CI1 - - - - CI2 CH2 - - - CZ CJ - - OXT"],
    ["DAH", "N CA C CB O CG - - - - - - CD1 CD2 - - - - - - CE1 CE2 - - - - - OE2 - - - OH CZ - - - OXT"],
    ["TBG", "N CA C CB O - CG1 CG2 - - - - - - - - - - - - - - - - - - - - CG3 - - - - - - - OXT"],
    ["ALO", "N CA C CB O - - CG2 - OG1 - - - - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["DDZ", "N CA C CB O - - - - OG1 - - - - - - - - - - - - - - - - - - - OG2 - - - - - - OXT"],
    ["KCR", "N CA C CB O CG - - CH - - CD - - - - - CH3 - CE - - - - - - - CX - - - OH - - CY NZ OXT"],
    ["GLZ", "N CA C - O"],
    ["IIL", "N CA C CB O - CG1 CG2 - - - - CD1 - - - - - - - - - - - - - - - - - - - - - - - OXT"],
    ["ML3", "N CA C CB O - - - - CM1 SG CD - - - - - - CM2 CE - - - - - - - CM3 - - - - - - - NZ OXT"],
    ["PPN", "N CA C CB O CG - - - N1 - - CD1 CD2 - - - O1 - - CE1 CE2 - - - - - - - - - - CZ - O2 - OXT"],
    ["SLL", "N CA C CB O CG - - - - - CD - - CK - - CL - CE - - - - - - CP - - CX OP1 OP2 - - OX NZ OXT"],
    ["TYI", "N CA C CB O CG I1 - - - - - CD1 CD2 - - - - - - CE1 CE2 - - - - - I2 - - - OH CZ - - - OXT"],
    ["CSU", "N CA C CB O - - - - - SG - - - - - OD1 OD2 - - - - - - - - - - OD3 S - - - - - - OXT"],
    ["CYG", "N CA C CB O - CG1 - C1 CA1 SG - CD1 - - CB1 - - N1 - - - - - - - - OE2 - - - - - O1 O2 - OXT"],
    ["ILX", "N CA C CB O - CG1 CG2 - OG1 - - CD1 - - - OD1 - - - - - - - - - - - - - - - - - - - OXT"],
    ["OAS", "N CA C CB O - - - OG - - - - - - - - - - - - - - - - - C1A C2A - - OAC - - - - - OXT"],
    ["HRG", "N CA C CB O CG - - - - - CD - - - - - - - - - - - NE - - - - - NH1 NH2 - CZ - - CG' OXT"],
    ["OMT", "N CA C CB O CG - - - - - - - - - - OD1 OD2 SD CE - - - - - - - - - - - - - - - - OXT"],
    ["PBF", "N CA C CB O CG CI1 CI2 CK1 - - - CD1 CD2 - - - - - - CE1 CE2 CK2 CL - CN1 CT - - - - ON2 CZ - - - OXT"],
    ["API", "N CA C - O - - C3 - C4 - - - - C5 - C6 - C7 - - - N6 - - - - - - - - O3 - - O4 - OXT"],
    ["HIP", "N CA C CB O CG - O1P - - - - - CD2 ND1 - - O2P - - CE1 - O3P P - NE2 - - - - - - - - - - OXT"],
    ["MIS", "N CA C CB O - - - OG - - - - - - - - - - - - - C1 - C2 C3 - O1P - - O2P O3P - P - - OXT"],
    ["ALN", "N CA C CB O - CG1 CG2 - CD3 - - CD1 CD2 - CZ1 - - - - CE1 CE2 CE3 - - - - - - - - - - CZ2 - - OXT"],
  ]],
  ["glycan", [
    ["NAG", "C2 C1 C3 C4 C5 C6 O1 - O3 O4 O5 O6 C7 C8 N2 O7"],
    ["BMA", "C2 C1 C3 C4 C5 C6 O1 O2 O3 O4 O5 O6"],
    ["MAN", "C2 C1 C3 C4 C5 C6 O1 O2 O3 O4 O5 O6"],
    ["FUC", "C2 C1 C3 C4 C5 C6 O1 O2 O3 O4 O5"],
    ["GLC", "C2 C1 C3 C4 C5 C6 O1 O2 O3 O4 O5 O6"],
    ["GAL", "C2 C1 C3 C4 C5 C6 O1 O2 O3 O4 O5 O6"],
    ["BGC", "C2 C1 C3 C4 C5 C6 O1 O2 O3 O4 O5 O6"],
    ["SIA", "C1 C2 C3 C4 C5 C6 - O2 - O4 - O6 C7 C8 - O7 C9 C10 C11 N5 O1A O1B O8 O9 O10"],
    ["NDG", "C2 C1 C3 C4 C5 C6 O1 - O3 O4 O5 O6 C7 C8 N2 O7"],
    ["XYP", "C2 C1 C3 C4 C5 - O1 O2 O3 O4 O5"],
    ["GLA", "C2 C1 C3 C4 C5 C6 O1 O2 O3 O4 O5 O6"],
    ["A2G", "C2 C1 C3 C4 C5 C6 O1 - O3 O4 O5 O6 C7 C8 N2 O7"],
    ["NGA", "C2 C1 C3 C4 C5 C6 O1 - O3 O4 O5 O6 C7 C8 N2 O7"],
  ]],
];

function buildVocabulary() {
  const tokens = [];
  const tokenToIdx = Object.create(null);
  const tokenAtoms = Object.create(null);
  const tokenSlotMap = Object.create(null);
  const residuesByTokenType = Object.create(null);
  const tokenGroupByIdx = [];

  for (const [group, entries] of RESIDUE_LAYOUTS_BY_GROUP) {
    const groupNames = [];
    for (const [name, layout] of entries) {
      const idx = tokens.length;
      tokens.push(name);
      tokenToIdx[name] = idx;
      tokenGroupByIdx.push(group);
      groupNames.push(name);

      const rawAtoms = layout.trim().split(/\s+/).map((a) => (a === "-" ? "" : a));
      const atoms37 = new Array(37).fill("");
      const slotMap = Object.create(null);
      for (let i = 0; i < rawAtoms.length && i < 37; i++) {
        atoms37[i] = rawAtoms[i];
        if (rawAtoms[i]) slotMap[rawAtoms[i]] = i;
      }
      tokenAtoms[name] = Object.freeze(atoms37);
      tokenSlotMap[name] = Object.freeze(slotMap);
    }
    residuesByTokenType[group] = Object.freeze(groupNames);
  }

  const nTokens = tokens.length;
  const restypeAtom37Mask = new Uint8Array(nTokens * 37);
  for (let t = 0; t < nTokens; t++) {
    const atoms = tokenAtoms[tokens[t]];
    for (let s = 0; s < 37; s++) {
      if (atoms[s]) restypeAtom37Mask[t * 37 + s] = 1;
    }
  }

  // Matches `get_allowed_residue_type_mask()` in `decoder.py`
  const allowedResidueTypes = new Uint8Array(NUM_TOKEN_TYPES * nTokens);
  for (const [group, names] of Object.entries(residuesByTokenType)) {
    const tid = TOKEN_TYPE_IDS[group];
    if (tid !== TokenType.UNKNOWN) {
      for (const name of names) {
        const idx = tokenToIdx[name];
        allowedResidueTypes[tid * nTokens + idx] = 1;
      }
    }
  }
  // TokenType.UNKNOWN = allowed.any(dim=0)
  for (let idx = 0; idx < nTokens; idx++) {
    let any = 0;
    for (let tid = 0; tid < NUM_TOKEN_TYPES; tid++) {
      if (tid !== TokenType.UNKNOWN && allowedResidueTypes[tid * nTokens + idx]) {
        any = 1;
        break;
      }
    }
    allowedResidueTypes[TokenType.UNKNOWN * nTokens + idx] = any;
  }

  return {
    encoding: Object.freeze({
      tokens: Object.freeze(tokens),
      n_tokens: nTokens,
      n_atoms: 37,
      token_to_idx: Object.freeze(tokenToIdx),
      idx_to_token: Object.freeze(tokens.slice()),
      token_atoms: Object.freeze(tokenAtoms),
      token_slot_map: Object.freeze(tokenSlotMap),
      token_group_by_idx: Object.freeze(tokenGroupByIdx),
    }),
    residuesByTokenType: Object.freeze(residuesByTokenType),
    restypeAtom37Mask,
    allowedResidueTypes,
  };
}

const _vocab = buildVocabulary();

export const UNIFIED_ATOM37_ENCODING = _vocab.encoding;
export const UNIFIED_ATOM37_TOKENS = UNIFIED_ATOM37_ENCODING.tokens;
export const UNIFIED_TOKEN_TO_IDX = UNIFIED_ATOM37_ENCODING.token_to_idx;
export const RESIDUES_BY_TOKEN_TYPE = _vocab.residuesByTokenType;

// Structured 2D views `[147][37]` and `[7][147]` alongside flat views
export const UNIFIED_RESTYPE_ATOM37_MASK = Object.freeze(
  Array.from({ length: _vocab.encoding.n_tokens }, (_, t) =>
    _vocab.restypeAtom37Mask.subarray(t * 37, (t + 1) * 37)
  )
);
export const UNIFIED_RESTYPE_ATOM37_MASK_FLAT = _vocab.restypeAtom37Mask;

export const ALLOWED_RESIDUE_TYPES = Object.freeze(
  Array.from({ length: NUM_TOKEN_TYPES }, (_, tid) =>
    _vocab.allowedResidueTypes.subarray(
      tid * _vocab.encoding.n_tokens,
      (tid + 1) * _vocab.encoding.n_tokens
    )
  )
);
export const ALLOWED_RESIDUE_TYPES_FLAT = _vocab.allowedResidueTypes;

export const NCAA_NAMES = RESIDUES_BY_TOKEN_TYPE.ncaa;
export const GLYCAN_NAMES = RESIDUES_BY_TOKEN_TYPE.glycan;

export const LOCAL_LATENT_RESIDUES = Object.freeze(
  UNIFIED_ATOM37_ENCODING.tokens.filter((name) => name !== "<A>" && name !== "<G>")
);
export const LOCAL_LATENT_RESIDUE_SET = new Set(LOCAL_LATENT_RESIDUES);

export const STANDARD_AA = Object.freeze(
  RESIDUES_BY_TOKEN_TYPE.aa.filter((name) => name !== "UNK")
);
export const STANDARD_RNA = Object.freeze(
  RESIDUES_BY_TOKEN_TYPE.rna.filter((name) => name !== "N")
);
export const STANDARD_DNA = Object.freeze(
  RESIDUES_BY_TOKEN_TYPE.dna.filter((name) => name !== "DN")
);

/**
 * The 28 canonical polymer tokens in unified-encoding order (`STANDARD_POLYMER_RESIDUES`),
 * used by `SEQUENCE_PROBABILITY_CONDITIONS` (`dim = 2 * 28 = 56`).
 */
export const STANDARD_POLYMER_RESIDUES = Object.freeze([
  ...STANDARD_AA,
  ...STANDARD_RNA,
  ...STANDARD_DNA,
]);

export const RESTYPE_1TO3 = Object.freeze({
  A: "ALA",
  R: "ARG",
  N: "ASN",
  D: "ASP",
  C: "CYS",
  Q: "GLN",
  E: "GLU",
  G: "GLY",
  H: "HIS",
  I: "ILE",
  L: "LEU",
  K: "LYS",
  M: "MET",
  F: "PHE",
  P: "PRO",
  S: "SER",
  T: "THR",
  W: "TRP",
  Y: "TYR",
  V: "VAL",
});

export const RESTYPE_3TO1 = Object.freeze(
  Object.fromEntries([
    ...Object.entries(RESTYPE_1TO3).map(([one, three]) => [three, one]),
    ["UNK", "X"],
    ["<M>", "X"],
    ["A", "A"],
    ["C", "C"],
    ["G", "G"],
    ["U", "U"],
    ["N", "N"],
    ["DA", "A"],
    ["DC", "C"],
    ["DG", "G"],
    ["DT", "T"],
    ["DN", "N"],
  ])
);

export const ATOMIC_NUMBER_BY_ELEMENT = Object.freeze({
  H: 1, HE: 2, LI: 3, BE: 4, B: 5, C: 6, N: 7, O: 8, F: 9, NE: 10,
  NA: 11, MG: 12, AL: 13, SI: 14, P: 15, S: 16, CL: 17, AR: 18,
  K: 19, CA: 20, SC: 21, TI: 22, V: 23, CR: 24, MN: 25, FE: 26, CO: 27, NI: 28, CU: 29, ZN: 30,
  GA: 31, GE: 32, AS: 33, SE: 34, BR: 35, KR: 36,
  RB: 37, SR: 38, Y: 39, ZR: 40, MO: 42, RU: 44, RH: 45, PD: 46, AG: 47, CD: 48,
  SN: 50, SB: 51, TE: 52, I: 53, XE: 54, CS: 55, BA: 56, W: 74, RE: 75, OS: 76, IR: 77, PT: 78, AU: 79, HG: 80, PB: 82, BI: 83,
});

export const ELEMENT_BY_ATOMIC_NUMBER = Object.freeze(
  Object.fromEntries(Object.entries(ATOMIC_NUMBER_BY_ELEMENT).map(([sym, z]) => [z, sym]))
);

/**
 * Infer element symbol from atom name and residue name when explicit element column is absent.
 */
export function inferElementSymbol(atomName, resName = "") {
  const clean = String(atomName || "").trim().toUpperCase();
  if (!clean || clean === "X") return "C";
  const upperRes = String(resName || "").trim().toUpperCase();
  if (clean === upperRes && ATOMIC_NUMBER_BY_ELEMENT[clean]) {
    return clean;
  }
  const stripped = clean.replace(/^[0-9]+/, "");
  if (stripped.startsWith("SE") && (upperRes === "SEC" || upperRes === "MSE" || upperRes === "SEB")) return "SE";
  if (stripped.startsWith("CL")) return "CL";
  if (stripped.startsWith("BR")) return "BR";
  if (stripped.startsWith("AS") && (upperRes === "CAS" || upperRes === "CAF")) return "AS";
  const first = stripped[0];
  if (ATOMIC_NUMBER_BY_ELEMENT[first]) return first;
  return "C";
}

export function getAtomicNumber(elementOrAtomName, resName = "") {
  const sym = String(elementOrAtomName || "").trim().toUpperCase();
  if (ATOMIC_NUMBER_BY_ELEMENT[sym]) return ATOMIC_NUMBER_BY_ELEMENT[sym];
  const inferred = inferElementSymbol(sym, resName);
  return ATOMIC_NUMBER_BY_ELEMENT[inferred] ?? 6;
}

export const ATOMIC_NUMBERS = ATOMIC_NUMBER_BY_ELEMENT;
export const ELEMENT_SYMBOLS_BY_Z = ELEMENT_BY_ATOMIC_NUMBER;

export const STANDARD_POLYMER_TO_PROB_IDX = Object.freeze(
  Object.fromEntries(STANDARD_POLYMER_RESIDUES.map((name, idx) => [name, idx]))
);

export const ATOM37_NAMES = Object.freeze([
  "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1",
  "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE",
  "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1",
  "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]);

export function inferTokenTypeFromResName(resName, isHetero = false) {
  const clean = String(resName || "").trim().toUpperCase();
  const idx = UNIFIED_TOKEN_TO_IDX[clean];
  if (idx !== undefined) {
    const group = UNIFIED_ATOM37_ENCODING.token_group_by_idx[idx];
    return TOKEN_TYPE_IDS[group] ?? TokenType.PROTEIN;
  }
  return isHetero ? TokenType.ATOMIZED : TokenType.PROTEIN;
}

export function lookupAtom37Slot(resName, atomName) {
  const cleanRes = String(resName || "").trim().toUpperCase();
  const cleanAtom = String(atomName || "").trim();
  const slotMap = UNIFIED_ATOM37_ENCODING.token_slot_map[cleanRes];
  if (slotMap && Object.prototype.hasOwnProperty.call(slotMap, cleanAtom)) {
    return slotMap[cleanAtom];
  }
  const fallback = ATOM37_NAMES.indexOf(cleanAtom);
  return fallback >= 0 ? fallback : -1;
}
