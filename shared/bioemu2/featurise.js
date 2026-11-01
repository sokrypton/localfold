/**
 * BioEmu2's input features, built from an AlphaFold 3 job.
 *
 * BioEmu2 (microsoft/bioemu2, checkpoint bioemu2-preview) is Boltz-derived but
 * reads NONE of AF3's batch: a transcription of af3-any-model's
 * `pipeline/bioemu2_features.py` (itself bioemu2 @7846769's System ->
 * AllAtomStructure -> batch path), field for field, held to that port by
 * `tools/check-bioemu2-features.js` against `tools/oracle/dump_bioemu2.py`.
 * The output keeps the vendor's own key names, so a dump and this are compared
 * by name.
 *
 * Its conventions, every one different from AlphaFold 3's somewhere:
 *
 *   * reference positions are the CCD's IDEAL coordinates, heavy atoms only,
 *     centred on the component's own mean - deterministic, no conformer draw.
 *     A polymer residue is centred over the WHOLE component and then loses OXT
 *     (protein) or OP3 (nucleic acid) at EVERY position, the C-terminus
 *     included, without re-centring;
 *   * a standard residue is one token (centre atom CA or C1'); an unknown
 *     residue is ALA's template typed UNK; a modified residue, a ligand and a
 *     glycan monomer are one token per atom, typed UNK_PROT (22);
 *   * a standard residue carries NO bonds - only atom-token residues keep their
 *     CCD bonds, plus the job's declared bonds; `token_bonds` is symmetric;
 *   * residue_index and ref_space_uid count residues across the WHOLE system;
 *   * chains with identical content are one entity, entities in order of first
 *     appearance, and the chains are LAID OUT entity by entity - not in job
 *     order;
 *   * the MSA is Boltz-1's: rows lined up by index across chains (paired rows
 *     first for a heteromer that has them), 4092 rows, deletion value
 *     pi/2 * atan(d / 3) - not AF3's 2/pi.
 *
 * 🔴 A SMILES LIGAND IS NOT EXACT AND CANNOT BE. The vendor draws its conformer
 * with an UNSEEDED ETKDG and names its atoms by RDKit's canonical rank, in the
 * InChI round trip's order, so two vendor runs disagree with each other. Here
 * the caller hands in a component (shared/chem's conformer), whose atoms keep
 * their SMILES order and element-and-counter names. The model reads the name
 * as an atom feature, so this is a modelling difference and it is the only one.
 */
import { loopOf, ELEMENT_SYMBOLS } from "../af3/featurise/ccd-component.js";
import { STANDARD_COMPONENTS } from "../af3/featurise/standard-components.js";

export const MAX_SEQS = 4092;
export const MAX_PAIRED_SEQS = 8192;
export const NUM_TOKENS = 33;
export const GAP = 1;
export const UNK_PROT = 22;
export const WINDOW = 32;
/** `method_feature`: BioEmu2's "md" and "non-md" conditioning tokens. */
export const METHOD = Object.freeze({ equilibrium: 4, structure: 7 });
export const CONTACT_CLASSES = Object.freeze({
  UNSPECIFIED: 0, UNSELECTED: 1, "POCKET>BINDER": 2, "BINDER>POCKET": 3, CONTACT: 4,
});
export const MOL_TYPE = Object.freeze({ protein: 0, dna: 1, rna: 2, ligand: 3 });

const PROT = (() => {
  const table = {};
  const letters = "ARNDCEQGHILKMFPSTWYV";
  const codes = [2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
  for (let i = 0; i < letters.length; i += 1) table[letters[i]] = codes[i];
  for (const letter of "XJBZOU") table[letter] = UNK_PROT;
  table["-"] = GAP;
  return table;
})();
const RNA = { A: 23, G: 24, C: 25, U: 26, N: 27 };
const DNA = { A: 28, G: 29, C: 30, T: 31, N: 32 };
const THREE = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", E: "GLU", Q: "GLN", G: "GLY", H: "HIS", I: "ILE",
  L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};
const UNKNOWN_CCD = { protein: "UNK", dna: "DN", rna: "N" };

/** `ord(c) - 32` a character, four of them, zero-padded. */
export function encodeAtomName(name) {
  const out = [0, 0, 0, 0];
  for (let i = 0; i < Math.min(4, name.length); i += 1) out[i] = name.charCodeAt(i) - 32;
  return out;
}

/** mdtraj's virtual site X is element 0; everything else its atomic number. */
function atomicNumber(symbol) {
  const upper = symbol.toUpperCase();
  if (upper === "X") return 0;
  const at = ELEMENT_SYMBOLS.indexOf(upper);
  if (at < 0) throw new Error(`BioEmu2: no atomic number for element ${symbol}`);
  return at + 1;
}

/**
 * template_from_cifwrapper: heavy atoms (minus leaving atoms when asked), ideal
 * coordinates - the model ones when ANY ideal value is unreadable, as the
 * vendor's ValueError fallback takes the whole component - centred on THEIR
 * mean, in float64 and rounded to float32 at the end, as numpy does.
 */
export function ccdTemplate(text, code, removeLeaving = false) {
  const loop = loopOf(text, "_chem_comp_atom");
  const column = (name) => loop.columns.indexOf(name);
  const get = (row, name) => { const at = column(name); return at < 0 ? undefined : row[at]; };
  const rows = loop.rows;
  if (rows.length === 0) throw new Error(`${code} has no atoms in the CCD`);
  const ideal = rows.every((row) => ["x", "y", "z"].every(
    (axis) => Number.isFinite(Number(get(row, `pdbx_model_Cartn_${axis}_ideal`)))
      && get(row, `pdbx_model_Cartn_${axis}_ideal`) !== "?" && get(row, `pdbx_model_Cartn_${axis}_ideal`) !== "."));
  const keep = rows.map((row) => {
    const symbol = (get(row, "type_symbol") ?? "").toUpperCase();
    if (symbol === "H" || symbol === "D") return false;
    if (removeLeaving && (get(row, "pdbx_leaving_atom_flag") ?? "N") === "Y") return false;
    return true;
  });
  const names = [];
  const element = [];
  const charge = [];
  const xyz = [];
  const index = new Map();
  rows.forEach((row, i) => {
    if (!keep[i]) return;
    const name = get(row, "atom_id");
    index.set(name, names.length);
    names.push(name);
    element.push(atomicNumber(get(row, "type_symbol") ?? "X"));
    const q = get(row, "charge");
    charge.push(q === undefined || q === "?" || q === "." ? 0 : Number(q));
    const prefix = ideal ? "pdbx_model_Cartn_" : "model_Cartn_";
    const point = ["x", "y", "z"].map((axis) => Number(get(row, `${prefix}${axis}${ideal ? "_ideal" : ""}`)));
    if (!point.every(Number.isFinite)) throw new Error(`${code}: atom ${name} has no usable coordinates`);
    xyz.push(point);
  });
  const mean = [0, 1, 2].map((axis) => xyz.reduce((sum, p) => sum + p[axis], 0) / xyz.length);
  const coords = new Float32Array(xyz.length * 3);
  xyz.forEach((p, i) => { for (let axis = 0; axis < 3; axis += 1) coords[i * 3 + axis] = p[axis] - mean[axis]; });
  const bondLoop = loopOf(text, "_chem_comp_bond");
  const b1 = bondLoop.columns.indexOf("atom_id_1");
  const b2 = bondLoop.columns.indexOf("atom_id_2");
  const bonds = [];
  for (const row of bondLoop.rows) {
    const a = index.get(row[b1]);
    const b = index.get(row[b2]);
    if (a !== undefined && b !== undefined) bonds.push([a, b]);
  }
  return { names, element, charge, coords, bonds };
}

/** A template without the named atoms; coordinates are NOT re-centred. */
function dropAtoms(t, drop) {
  const map = [];
  const out = { names: [], element: [], charge: [], coords: [], bonds: [] };
  const coords = [];
  t.names.forEach((name, i) => {
    if (drop.includes(name)) { map.push(-1); return; }
    map.push(out.names.length);
    out.names.push(name);
    out.element.push(t.element[i]);
    out.charge.push(t.charge[i]);
    coords.push(t.coords[i * 3], t.coords[i * 3 + 1], t.coords[i * 3 + 2]);
  });
  out.coords = Float32Array.from(coords);
  out.bonds = t.bonds.map(([a, b]) => [map[a], map[b]]).filter(([a, b]) => a >= 0 && b >= 0);
  return out;
}

/** is_leaving_atom_donor: does a bonded atom of this residue sit on a heavy leaving atom? */
function isLeavingAtomDonor(text, bondAtomNames) {
  const loop = loopOf(text, "_chem_comp_atom");
  const col = (name) => loop.columns.indexOf(name);
  const leaving = new Set(loop.rows.filter((row) => row[col("pdbx_leaving_atom_flag")] === "Y"
    && !["H", "D"].includes((row[col("type_symbol")] ?? "").toUpperCase())).map((row) => row[col("atom_id")]));
  const bondLoop = loopOf(text, "_chem_comp_bond");
  const b1 = bondLoop.columns.indexOf("atom_id_1");
  const b2 = bondLoop.columns.indexOf("atom_id_2");
  const names = new Set(bondAtomNames);
  return bondLoop.rows.some((row) => (names.has(row[b1]) && leaving.has(row[b2]))
    || (names.has(row[b2]) && leaving.has(row[b1])));
}

function componentText(ccd, code) {
  const text = STANDARD_COMPONENTS[code] ?? ccd?.(code);
  if (text == null) throw new Error(`${code} is not in the CCD`);
  return text;
}

/** inference_input_structure_from_{protein,rna,dna}_sequence, with modifications. */
function polymerResidues(ccd, kind, sequence, modifications) {
  const hasMods = modifications.length > 0;
  const mods = new Map(modifications.map(({ code, position }) => [position, code]));
  const table = { protein: PROT, rna: RNA, dna: DNA }[kind];
  const unknownLetter = kind === "protein" ? "X" : "N";
  const out = [];
  for (let at = 0; at < sequence.length; at += 1) {
    const position = at + 1;
    const letter = sequence[at];
    let code = mods.get(position) ?? null;
    if (code === null && letter === unknownLetter) code = UNKNOWN_CCD[kind];
    if (code !== null && code !== UNKNOWN_CCD[kind]) {
      const t = dropAtoms(ccdTemplate(componentText(ccd, code), code), ["OXT", "OP3"]);
      out.push({ t, restype: UNK_PROT, perAtom: true, centre: null, label: code });
      continue;
    }
    if (kind === "protein") {
      const ccdCode = code !== null || THREE[letter] === undefined ? "ALA" : THREE[letter];
      const t = dropAtoms(ccdTemplate(componentText(ccd, ccdCode), ccdCode), ["OXT"]);
      const restype = code === null ? (table[letter] ?? UNK_PROT) : UNK_PROT;
      out.push({ t, restype, perAtom: false, centre: "CA", label: restype === UNK_PROT ? "UNK" : ccdCode });
    } else {
      let ccdCode = kind === "rna" ? letter : `D${letter}`;
      if (code !== null) ccdCode = UNKNOWN_CCD[kind];
      const t = dropAtoms(ccdTemplate(componentText(ccd, ccdCode), ccdCode), ["OP3"]);
      const unknown = code !== null || letter === "N";
      const restype = unknown ? table.N : table[letter];
      if (restype === undefined) throw new Error(`BioEmu2: ${letter} is not a ${kind.toUpperCase()} residue`);
      const label = unknown ? (hasMods ? ccdCode : "UNK") : ccdCode;
      out.push({ t, restype, perAtom: false, centre: "C1'", label });
    }
  }
  return out;
}

/**
 * BioEmu2 entities from a job's chains: identical content is one entity, in
 * order of first appearance. A ligand with one CCD code and no declared bond is
 * a Ligand; one with several codes or any bond is a Branched entity.
 */
function entitiesOf(job) {
  const bonded = new Set((job.bondedAtomPairs ?? []).flatMap((pair) => pair.map((end) => end[0])));
  const entities = [];
  const index = new Map();
  for (const chain of job.chains) {
    let key;
    const mods = [...(chain.modifications ?? [])]
      .map((m) => [m.code, m.position]).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] - b[1]));
    if (["protein", "rna", "dna"].includes(chain.kind)) key = [chain.kind, chain.sequence, mods];
    else if (chain.smiles != null) key = ["smiles", chain.smiles, []];
    else if (chain.ccdCodes.length === 1 && !bonded.has(chain.id)) key = ["ccd", chain.ccdCodes[0], []];
    else key = ["branched", chain.ccdCodes, []];
    const text = JSON.stringify(key);
    if (!index.has(text)) { index.set(text, entities.length); entities.push({ key, chains: [] }); }
    entities[index.get(text)].chains.push(chain);
  }
  return entities;
}

/**
 * A job -> BioEmu2's feature dict (no batch axis) and the atom keys.
 *
 * @param {{chains: {id: string, kind: "protein"|"rna"|"dna"|"ligand", sequence?: string,
 *           modifications?: {code: string, position: number}[], ccdCodes?: string[],
 *           smiles?: string, unpairedMsa?: string, pairedMsa?: string}[],
 *          bondedAtomPairs?: [string, number, string][][]}} job
 * @param {{ccd?: (code: string) => string, mode?: "structure"|"equilibrium",
 *          contacts?: object[], smilesTemplates?: Object<string, object>}} options
 *   `ccd` returns a component's mmCIF text (the standard ones are built in);
 *   `smilesTemplates[chainId]` is a SMILES ligand's template, {names, element,
 *   charge, coords, bonds} - see the note at the top.
 * @returns {{features: object, atomKeys: [string, number, string][], topology: object}}
 */
export function featuriseBioemu2(job, options = {}) {
  const { ccd, mode = "structure", contacts = null, smilesTemplates = {} } = options;
  if (!(mode in METHOD)) throw new Error(`BioEmu2 mode ${mode}: want structure or equilibrium`);
  const bonds = job.bondedAtomPairs ?? [];
  const atoms = { coords: [], element: [], charge: [], name: [], residue: [], token: [], mol: [], restype: [] };
  const tok = { asym: [], entity: [], sym: [], centre: [] };
  const bondIndex = [];
  const atomKeys = [];
  const chainInfo = [];
  const labels = [];
  const ligandCcd = {};
  let nAtoms = 0;
  let nTokens = 0;
  let nRes = 0;
  const chainOrder = [];
  entitiesOf(job).forEach(({ key: [kind, payload, mods], chains }, entityId) => {
    chains.forEach((chain, symId) => {
      chainOrder.push(chain);
      let residues;
      let mol;
      if (["protein", "rna", "dna"].includes(kind)) {
        residues = polymerResidues(ccd, kind, payload, mods.map(([code, position]) => ({ code, position })));
        mol = MOL_TYPE[kind];
      } else if (kind === "ccd") {
        residues = [{ t: ccdTemplate(componentText(ccd, payload), payload), restype: UNK_PROT, perAtom: true,
                      centre: null, label: payload }];
        mol = MOL_TYPE.ligand;
        ligandCcd[chainOrder.length - 1] = payload;
      } else if (kind === "smiles") {
        const t = smilesTemplates[chain.id];
        if (t === undefined) throw new Error(`BioEmu2: SMILES chain ${chain.id} needs a template`);
        residues = [{ t, restype: UNK_PROT, perAtom: true, centre: null, label: "UNK" }];
        mol = MOL_TYPE.ligand;
      } else {
        // per entity, as all_atom_structure builds one Branched structure its chains share: a residue that donates in
        // any copy loses its leaving atoms in every copy
        const entityIds = new Set(chains.map((c) => c.id));
        const atomsByRes = new Map();
        for (const pair of bonds) {
          for (const [c, r, a] of pair) {
            if (!entityIds.has(c)) continue;
            if (!atomsByRes.has(r - 1)) atomsByRes.set(r - 1, []);
            atomsByRes.get(r - 1).push(a);
          }
        }
        residues = payload.map((code, i) => {
          const text = componentText(ccd, code);
          const remove = atomsByRes.has(i) ? isLeavingAtomDonor(text, atomsByRes.get(i)) : false;
          return { t: ccdTemplate(text, code, remove), restype: UNK_PROT, perAtom: true, centre: null, label: code };
        });
        mol = MOL_TYPE.ligand;
      }
      const chainTokenStart = nTokens;
      residues.forEach((res, rLocal) => {
        const t = res.t;
        const n = t.names.length;
        for (let i = 0; i < n; i += 1) {
          labels.push(res.label);
          atoms.coords.push(t.coords[i * 3], t.coords[i * 3 + 1], t.coords[i * 3 + 2]);
          atoms.element.push(t.element[i]);
          atoms.charge.push(t.charge[i]);
          atoms.name.push(t.names[i]);
          atoms.residue.push(nRes);
          atoms.mol.push(mol);
          atoms.restype.push(res.restype);
          atoms.token.push(res.perAtom ? nTokens + i : nTokens);
          atomKeys.push([chain.id, rLocal + 1, t.names[i]]);
        }
        // a standard residue is built with no bonds at all; only atom-token residues keep their CCD bonds
        if (res.perAtom) for (const [a, b] of t.bonds) bondIndex.push([a + nAtoms, b + nAtoms]);
        if (res.perAtom) {
          for (let i = 0; i < n; i += 1) tok.centre.push(nAtoms + i);
          nTokens += n;
        } else {
          const centre = t.names.indexOf(res.centre);
          if (centre < 0) throw new Error(`BioEmu2: ${res.label} has no ${res.centre}`);
          tok.centre.push(nAtoms + centre);
          nTokens += 1;
        }
        nAtoms += n;
        nRes += 1;
      });
      const count = nTokens - chainTokenStart;
      for (let i = 0; i < count; i += 1) {
        tok.asym.push(chainOrder.length - 1);
        tok.entity.push(entityId);
        tok.sym.push(symId);
      }
      chainInfo.push({ chain, kind, payload, residues, from: chainTokenStart, to: nTokens });
    });
  });

  // declared covalent bonds (System.bonds -> _add_extra_covalent_bonds)
  const keyIndex = new Map(atomKeys.map((k, i) => [k.join("\u0000"), i]));
  for (const [first, second] of bonds) {
    const a = keyIndex.get(first.join("\u0000"));
    const b = keyIndex.get(second.join("\u0000"));
    if (a === undefined || b === undefined) {
      throw new Error(`BioEmu2: the declared bond ${JSON.stringify([first, second])} names an atom that is not there`);
    }
    bondIndex.push([a, b]);
  }

  const T = nTokens;
  const centre = Int32Array.from(tok.centre);
  const tokenBonds = new Float32Array(T * T);
  for (const [a, b] of bondIndex) {
    const i = atoms.token[a];
    const j = atoms.token[b];
    tokenBonds[i * T + j] = 1;
    tokenBonds[j * T + i] = 1;
  }
  const features = {
    res_type: Int32Array.from(centre, (c) => atoms.restype[c]),
    residue_index: Int32Array.from(centre, (c) => atoms.residue[c]),
    asym_id: Int32Array.from(tok.asym),
    entity_id: Int32Array.from(tok.entity),
    sym_id: Int32Array.from(tok.sym),
    token_center_atom_index: centre,
    token_pad_mask: new Int32Array(T).fill(1),
    token_bonds: tokenBonds,
    method_feature: new Int32Array(T).fill(METHOD[mode]),
    pocket_feature: new Int32Array(T),
    contact_conditioning: (() => {
      const out = new Int32Array(T * T * 5);
      for (let p = 0; p < T * T; p += 1) out[p * 5] = 1;
      return out;
    })(),
    contact_threshold: new Float32Array(T * T),
    dt: 0,
    is_dynamics_conditioned: 0,
  };
  const residueOfAtom = Int32Array.from(atoms.residue);
  const tokenOfAtom = Int32Array.from(atoms.token);
  Object.assign(features, msaFeatures(chainInfo, residueOfAtom, tokenOfAtom, T));
  if (contacts) Object.assign(features, contactFeatures(contacts, chainInfo, residueOfAtom, tokenOfAtom, atoms.name, centre, T));

  // atoms, padded to the window (batch_collate)
  const nPad = Math.ceil(nAtoms / WINDOW) * WINDOW;
  const pad = (values, width, fill, Type) => {
    const out = new Type(nPad * width).fill(fill);
    for (let i = 0; i < values.length; i += 1) out[i] = values[i];
    return out;
  };
  const nameEncoding = [];
  for (const name of atoms.name) nameEncoding.push(...encodeAtomName(name));
  Object.assign(features, {
    template_coords: pad(atoms.coords, 3, 0, Float32Array),
    element: pad(atoms.element, 1, 0, Int32Array),
    charge: pad(atoms.charge, 1, 0, Float32Array),
    name_encoding: pad(nameEncoding, 4, 0, Int32Array),
    token_idx_at_atom: pad(atoms.token, 1, -1, Int32Array),
    ref_space_uid: pad(atoms.residue, 1, -1, Int32Array),
    atom_pad_mask: pad(new Array(nAtoms).fill(1), 1, 0, Int32Array),
    atom_mask: pad(new Array(nAtoms).fill(1), 1, 0, Int32Array),
    coords_cond: new Float32Array(nPad * 3),
  });
  const topology = {
    names: atoms.name, labels, element: atoms.element, residueIdx: atoms.residue, molType: atoms.mol,
    residueType: atoms.restype, tokenIdxAtAtom: atoms.token,
    atomAsym: atoms.token.map((t) => tok.asym[t]), bondIndex, ligandCcd, nPad,
  };
  return { features, atomKeys, tokens: T, atoms: nAtoms, atomsPadded: nPad, msaRows: features.msaRows, topology };
}

/**
 * Condition on a starting frame (propagation): add_propagation's coords_cond
 * (the frame CENTRED), dt in ps, and the flag. `x0` is the real atoms, in the
 * order featuriseBioemu2 lays them out.
 */
export function addPropagation(features, x0, lagNs) {
  if (!(lagNs > 0 && Number.isFinite(lagNs))) throw new Error("the lag must be finite and positive");
  const nPad = features.template_coords.length / 3;
  const n = x0.length / 3;
  const mean = [0, 1, 2].map((axis) => { let s = 0; for (let i = 0; i < n; i += 1) s += x0[i * 3 + axis]; return s / n; });
  const cond = new Float32Array(nPad * 3);
  for (let i = 0; i < n; i += 1) for (let axis = 0; axis < 3; axis += 1) cond[i * 3 + axis] = x0[i * 3 + axis] - mean[axis];
  return { ...features, coords_cond: cond, dt: Math.fround(lagNs * 1000), is_dynamics_conditioned: 1 };
}

// --- MSA (msa/colabfold.tokenize_msa + msa/features.process_msa_features) -----

function a3mRows(a3m) {
  return a3m.trim().split(/\r?\n/).map((line) => line.trim())
    .filter((line) => line && !line.startsWith(">") && !line.startsWith("#"));
}

function queryOf(row) {
  let out = "";
  for (const c of row) if (!(c >= "a" && c <= "z") && c !== "-" && c !== ".") out += c;
  return out.toUpperCase();
}

/** _build_msa_from_colabfold_seqs: (residue tokens, {position: deletions}) per row. */
function encodeRows(rows) {
  return rows.map((row) => {
    const res = [];
    const dels = new Map();
    let count = 0;
    for (const c of row) {
      if (c !== "-" && c >= "a" && c <= "z") { count += 1; continue; }
      if (count) { dels.set(res.length, count); count = 0; }
      const code = PROT[c];
      if (code === undefined) throw new Error(`BioEmu2: ${c} is not an MSA residue`);
      res.push(code);
    }
    return { res, dels };
  });
}

function msaFeatures(chainInfo, residueOfAtom, tokenOfAtom, T) {
  const protein = chainInfo.map((info, i) => (info.kind === "protein" ? i : -1)).filter((i) => i >= 0);
  const seqs = new Map(protein.map((i) => [i, chainInfo[i].payload]));
  const unpairedRows = (i) => {
    const a3m = chainInfo[i].chain.unpairedMsa;
    let rows = a3m ? a3mRows(a3m) : [];
    if (rows.length === 0) rows = [seqs.get(i)];
    if (queryOf(rows[0]) !== seqs.get(i).toUpperCase()) {
      throw new Error(`MSA query of chain ${chainInfo[i].chain.id} is not its sequence`);
    }
    return rows;
  };
  const distinct = new Set(seqs.values());
  const heteromeric = distinct.size > 1;
  const pairedGiven = heteromeric && protein.every((i) => chainInfo[i].chain.pairedMsa);
  const nChains = chainInfo.length;
  const msas = new Map();
  const pairing = [];
  const isPaired = [];
  const all = (value) => Array.from({ length: nChains }, () => value);
  if (pairedGiven) {
    const pairedRows = new Map(protein.map((i) => [i, a3mRows(chainInfo[i].chain.pairedMsa)
      .slice(0, MAX_PAIRED_SEQS + 1).slice(1).filter((r) => r.replaceAll("-", "")).slice(0, MAX_PAIRED_SEQS)]));
    const unpaired = new Map(protein.map((i) => [i, unpairedRows(i).slice(0, MAX_SEQS)]));
    const depth = Math.min(Math.min(...protein.map((i) => pairedRows.get(i).length)), MAX_SEQS - 1);
    for (const i of protein) {
      const combined = [seqs.get(i), ...pairedRows.get(i).slice(0, depth)];
      const u = unpaired.get(i);
      const start = u.length && u[0].replaceAll("-", "").toUpperCase() === seqs.get(i).toUpperCase() ? 1 : 0;
      combined.push(...u.slice(start, start + MAX_SEQS - combined.length));
      msas.set(i, encodeRows(combined));
    }
    pairing.push(all(0));
    isPaired.push(all(1));
    for (let k = 1; k <= depth; k += 1) {
      pairing.push(Array.from({ length: nChains }, (_, c) => (protein.includes(c) ? k : -1)));
      isPaired.push(Array.from({ length: nChains }, (_, c) => (protein.includes(c) ? 1 : 0)));
    }
    const offset = depth + 1;
    const maxUnpaired = Math.max(...protein.map((c) => msas.get(c).length - offset));
    for (let k = 0; k < maxUnpaired; k += 1) {
      pairing.push(Array.from({ length: nChains }, (_, c) => (protein.includes(c) && offset + k < msas.get(c).length
        ? offset + k : -1)));
      isPaired.push(all(0));
      if (pairing.length >= MAX_SEQS) break;
    }
  } else {
    for (const i of protein) msas.set(i, encodeRows(unpairedRows(i).slice(0, MAX_SEQS)));
    const homomeric = distinct.size === 1 && protein.length > 1;
    const nSeqs = Math.min(protein.length ? Math.min(...protein.map((c) => msas.get(c).length)) : 1, MAX_SEQS);
    for (let k = 0; k < nSeqs; k += 1) {
      const flag = homomeric ? 1 : (k === 0 ? 1 : 0);
      isPaired.push(all(flag));
      pairing.push(Array.from({ length: nChains }, (_, c) => (protein.includes(c) ? k : (k === 0 ? 0 : -1))));
    }
  }
  // dummy MSAs: the chain's own residue types, one row
  for (let c = 0; c < nChains; c += 1) {
    if (!msas.has(c)) msas.set(c, [{ res: chainInfo[c].residues.map((r) => r.restype), dels: new Map() }]);
  }
  const resAtToken = new Int32Array(T);
  for (let a = 0; a < tokenOfAtom.length; a += 1) resAtToken[tokenOfAtom[a]] = residueOfAtom[a];
  const asym = new Int32Array(T);
  const local = new Int32Array(T);
  chainInfo.forEach((info, c) => {
    for (let t = info.from; t < info.to; t += 1) { asym[t] = c; local[t] = resAtToken[t] - resAtToken[info.from]; }
  });
  const S = pairing.length;
  const msa = new Int32Array(S * T).fill(GAP);
  const deletion = new Float32Array(S * T);
  const paired = new Int32Array(S * T);
  for (let t = 0; t < T; t += 1) {
    const c = asym[t];
    const r = local[t];
    for (let s = 0; s < S; s += 1) {
      paired[s * T + t] = isPaired[s][c];
      const k = pairing[s][c];
      if (k !== -1) {
        const { res, dels } = msas.get(c)[k];
        msa[s * T + t] = res[r];
        deletion[s * T + t] = dels.get(r) ?? 0;
      }
    }
  }
  // (the vendor's pi/2 * arctan(d / 3) runs in float32, and so does this, a rounding at each step)
  const value = new Float32Array(S * T);
  const half = Math.fround(Math.PI / 2);
  for (let i = 0; i < S * T; i += 1) value[i] = Math.fround(half * Math.fround(Math.atan(Math.fround(deletion[i] / 3))));
  const hasDeletion = Int32Array.from(deletion, (d) => (d > 0 ? 1 : 0));
  const deletionMean = new Float32Array(T);
  const profile = new Float32Array(T * NUM_TOKENS);
  for (let t = 0; t < T; t += 1) {
    let sum = 0;
    const counts = new Float64Array(NUM_TOKENS);
    for (let s = 0; s < S; s += 1) { sum += value[s * T + t]; counts[msa[s * T + t]] += 1; }
    deletionMean[t] = sum / S;
    for (let k = 0; k < NUM_TOKENS; k += 1) profile[t * NUM_TOKENS + k] = counts[k] / S;
  }
  return {
    msa, msa_paired: paired, deletion_value: value, has_deletion: hasDeletion, deletion_mean: deletionMean,
    profile, msa_mask: new Int32Array(S * T).fill(1), msaRows: S,
  };
}

/** One-hot an index MSA ([S * T] -> [S * T * 33]), the network's input form. */
export function msaOneHot(msa) {
  const out = new Float32Array(msa.length * NUM_TOKENS);
  for (let i = 0; i < msa.length; i += 1) out[i * NUM_TOKENS + msa[i]] = 1;
  return out;
}

/**
 * Contacts in BioEmu2's Contact form: [{token1: {asym_id, res_id?, atom_name?},
 * token2: ..., max_distance}]. A token's res_id defaults to 1 only when it
 * names an atom (ContactToken.validate_explicit_res_id).
 */
export function parseContacts(raw) {
  return raw.map((c) => {
    const extra = Object.keys(c).filter((k) => !["token1", "token2", "max_distance"].includes(k));
    if (extra.length) throw new Error(`unknown contact field(s) ${extra.sort()}`);
    const tokens = ["token1", "token2"].map((k) => {
      const t = c[k];
      const bad = Object.keys(t).filter((key) => !["asym_id", "res_id", "atom_name"].includes(key));
      if (bad.length) throw new Error(`unknown contact token field(s) in ${JSON.stringify(t)}`);
      if (t.atom_name == null && !("res_id" in t)) {
        throw new Error(`${JSON.stringify(t)}: res_id must be explicitly set when atom_name is None`);
      }
      const resId = Number(t.res_id ?? 1);
      if (resId < 1) throw new Error(`${JSON.stringify(t)}: res_id is 1-based`);
      return [t.asym_id, resId, t.atom_name ?? null];
    });
    const d = Number(c.max_distance ?? 6.0);
    if (!(d >= 4.0 && d <= 20.0)) throw new Error(`max_distance ${d} is outside [4, 20] A, the range BioEmu2 was trained on`);
    return [tokens[0], tokens[1], d];
  });
}

/**
 * _add_contact_conditioning: every pair UNSELECTED, the named pairs CONTACT
 * (both ways) at their distance. A residue token is found by chain and
 * chain-local residue; a named atom picks the token centred on that atom; a
 * residue with several tokens is addressed by its CA or C1' token.
 */
function contactFeatures(contacts, chainInfo, residueOfAtom, tokenOfAtom, atomNames, centre, T) {
  const resAtToken = new Int32Array(T);
  for (let a = 0; a < tokenOfAtom.length; a += 1) resAtToken[tokenOfAtom[a]] = residueOfAtom[a];
  const names = Array.from(centre, (c) => atomNames[c]);
  const local = new Int32Array(T);
  const chainOf = new Map();
  chainInfo.forEach((info) => {
    chainOf.set(info.chain.id, info);
    for (let t = info.from; t < info.to; t += 1) local[t] = resAtToken[t] - resAtToken[info.from];
  });
  const resolve = (token) => {
    const [cid, resId, atom] = token;
    const info = chainOf.get(cid);
    if (info === undefined) throw new Error(`contact token ${JSON.stringify(token)}: no chain ${cid}`);
    if (info.kind === "ccd" || info.kind === "smiles") {
      if (!atom) throw new Error(`contact token ${JSON.stringify(token)} addresses a ligand and must name an atom`);
      if (resId !== 1) throw new Error(`contact token ${JSON.stringify(token)}: a ligand has one residue`);
    } else {
      if (info.kind === "branched" && !atom) {
        throw new Error(`contact token ${JSON.stringify(token)} addresses a branched entity and must name an atom`);
      }
      if (resId > info.payload.length) {
        throw new Error(`contact token ${JSON.stringify(token)}: res_id exceeds the chain length ${info.payload.length}`);
      }
    }
    let sel = [];
    for (let t = info.from; t < info.to; t += 1) if (local[t] === resId - 1) sel.push(t);
    if (atom) sel = sel.filter((t) => names[t] === atom);
    else if (sel.length > 1) sel = sel.filter((t) => names[t] === "CA" || names[t] === "C1'");
    if (sel.length !== 1) throw new Error(`contact token ${JSON.stringify(token)} does not match exactly one token`);
    return sel[0];
  };
  const cls = new Int32Array(T * T).fill(CONTACT_CLASSES.UNSELECTED);
  const threshold = new Float32Array(T * T);
  for (const [a, b, d] of contacts) {
    const i = resolve(a);
    const j = resolve(b);
    cls[i * T + j] = cls[j * T + i] = CONTACT_CLASSES.CONTACT;
    threshold[i * T + j] = threshold[j * T + i] = d;
  }
  const out = new Int32Array(T * T * 5);
  for (let p = 0; p < T * T; p += 1) out[p * 5 + cls[p]] = 1;
  return { contact_conditioning: out, contact_threshold: threshold };
}

/**
 * An AlphaFold 3 job JSON (either dialect, as tools/oracle reads it) -> the
 * job form featuriseBioemu2 takes. Copies expand: `id: ["A", "B"]` is two
 * chains with the same content.
 */
export function bioemu2JobFromAf3Json(json) {
  const chains = [];
  for (const entry of json.sequences) {
    const [[kind, e]] = Object.entries(entry);
    const ids = Array.isArray(e.id) ? e.id : [e.id];
    for (const id of ids) {
      if (kind === "protein") {
        chains.push({ id, kind, sequence: e.sequence, unpairedMsa: e.unpairedMsa ?? null, pairedMsa: e.pairedMsa ?? null,
                      modifications: (e.modifications ?? []).map((m) => ({ code: m.ptmType, position: m.ptmPosition })) });
      } else if (kind === "rna" || kind === "dna") {
        chains.push({ id, kind, sequence: e.sequence,
                      modifications: (e.modifications ?? []).map((m) => ({ code: m.modificationType, position: m.basePosition })) });
      } else if (kind === "ligand") {
        chains.push({ id, kind, ccdCodes: e.ccdCodes ?? null, smiles: e.smiles ?? null });
      } else {
        throw new Error(`BioEmu2: a ${kind} entry is not an input it takes`);
      }
    }
  }
  return { chains, bondedAtomPairs: json.bondedAtomPairs ?? [] };
}
