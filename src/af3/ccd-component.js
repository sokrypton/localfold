/**
 * A chemical component from the PDB's dictionary, for ligands.
 *
 *     const gol = parseCcdComponent(await (await fetch(ccdUrl("GOL"))).text());
 *
 * AF3's own featuriser reads a 515 MB CCD pickle. A browser cannot, and does not
 * need to: a fold touches the handful of components its ligands name, and each
 * is a small mmCIF file the PDB serves individually. The 21 polymer components
 * stay baked in reference-conformers.js, because every fold needs them.
 *
 * 🔴 THE COLUMN ORDER IS READ, NEVER ASSUMED. An mmCIF loop declares its columns
 * and the PDB has added some over the years - `pdbx_backbone_atom_flag` and the
 * two terminal flags are recent, and they sit in the MIDDLE of the atom loop,
 * between the stereo config and the coordinates. Code that counted fields from
 * the left read the coordinates as flags the day those appeared.
 *
 * 🔴 AND THE IDEAL COORDINATES ARE NOT AF3's. AF3 builds a fresh conformer per
 * instance with RDKit - fixed bond lengths and angles, random torsions - so its
 * reference positions for GOL bear no resemblance to the dictionary's, while
 * the atom names, elements and charges match exactly. This is the same thing
 * the 21 amino acids already do (see reference-conformers.js), it costs about
 * 0.01 A of structure, and it is why the checker compares BONDED DISTANCES
 * rather than coordinates. Using the dictionary's ideal conformer is therefore
 * correct in the only sense available.
 */

/** Where the PDB serves one component. */
export function ccdUrl(code) {
  const upper = code.trim().toUpperCase();
  if (!/^[A-Z0-9]{1,5}$/.test(upper)) {
    throw new Error(`${code} is not a CCD code: expected 1-5 letters or digits`);
  }
  return `https://files.rcsb.org/ligands/download/${upper}.cif`;
}

/** The column names of an mmCIF `loop_`, and the rows under it. */
function loopOf(text, prefix) {
  const lines = text.split(/\r?\n/);
  const columns = [];
  const rows = [];
  let inHeader = false;
  let inBody = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith(`${prefix}.`)) {
      if (inBody) break;              // a second loop with the same prefix
      columns.push(trimmed.slice(prefix.length + 1).split(/\s+/)[0]);
      inHeader = true;
      continue;
    }
    if (!inHeader) continue;
    if (trimmed === "" || trimmed === "#" || trimmed === "loop_") {
      if (inBody) break;
      continue;
    }
    if (trimmed.startsWith("_")) break;
    inBody = true;
    // Values may be quoted, and an atom name like O5' or "C1'" can carry one.
    rows.push(trimmed.match(/'[^']*'|"[^"]*"|\S+/g)?.map(
      (value) => value.replace(/^['"]|['"]$/g, "")) ?? []);
  }
  return { columns, rows };
}

/**
 * Parse one component's mmCIF into the fields AF3's featuriser needs.
 *
 * 🔴 HYDROGENS ARE DROPPED. AF3 tokenises a ligand one token per HEAVY atom -
 * a six-token glycerol, not fourteen - and its reference arrays carry no
 * hydrogens at all. Keeping them produces a batch of the wrong length whose
 * every downstream shape is still self-consistent.
 *
 * @returns {{code: string, atoms: {name: string, element: number, charge: number,
 *            x: number, y: number, z: number}[],
 *           bonds: {from: number, to: number, order: number}[]}}
 */
export function parseCcdComponent(text) {
  const code = text.match(/_chem_comp\.id\s+(\S+)/)?.[1]?.replace(/['"]/g, "");
  if (code === undefined) throw new Error("this mmCIF has no _chem_comp.id");

  const atomLoop = loopOf(text, "_chem_comp_atom");
  const at = (row, name) => {
    const index = atomLoop.columns.indexOf(name);
    return index < 0 ? undefined : row[index];
  };
  // The ideal conformer is the one with no crystal contacts in it. A component
  // occasionally lacks it, and then the model coordinates are what there is.
  const hasIdeal = atomLoop.columns.includes("pdbx_model_Cartn_x_ideal");
  const coordinate = (row, axis) => {
    const value = at(row, hasIdeal ? `pdbx_model_Cartn_${axis}_ideal` : `model_Cartn_${axis}`);
    const parsed = Number.parseFloat(value);
    if (!Number.isFinite(parsed)) throw new Error(`${code} has no usable ${axis} coordinate`);
    return parsed;
  };

  const byName = new Map();
  const atoms = [];
  for (const row of atomLoop.rows) {
    const symbol = (at(row, "type_symbol") ?? "").toUpperCase();
    if (symbol === "H" || symbol === "D") continue;
    const name = at(row, "atom_id") ?? "";
    byName.set(name, atoms.length);
    atoms.push({
      name,
      element: ELEMENTS.indexOf(symbol) + 1,
      charge: Number.parseFloat(at(row, "charge") ?? "0") || 0,
      x: coordinate(row, "x"), y: coordinate(row, "y"), z: coordinate(row, "z"),
    });
  }
  if (atoms.length === 0) throw new Error(`${code} has no heavy atoms`);

  const bondLoop = loopOf(text, "_chem_comp_bond");
  const bondAt = (row, name) => {
    const index = bondLoop.columns.indexOf(name);
    return index < 0 ? undefined : row[index];
  };
  const bonds = [];
  for (const row of bondLoop.rows) {
    const from = byName.get(bondAt(row, "atom_id_1"));
    const to = byName.get(bondAt(row, "atom_id_2"));
    if (from === undefined || to === undefined) continue;   // a bond to a hydrogen
    bonds.push({ from, to, order: BOND_ORDERS[bondAt(row, "value_order") ?? "SING"] ?? 1 });
  }
  return { code, atoms, bonds };
}

/**
 * AF3's element numbering: the atomic number, so carbon is 6.
 *
 * 🔴 ONE-BASED BY POSITION IN THIS LIST, which is the same thing only because
 * the list starts at hydrogen and omits nothing. It is written out rather than
 * computed so that a gap would be visible.
 */
const ELEMENTS = [
  "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
  "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
  "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
  "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
  "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
  "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
  "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
  "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
  "TL", "PB", "BI", "PO", "AT", "RN",
];

const BOND_ORDERS = { SING: 1, DOUB: 2, TRIP: 3, QUAD: 4, AROM: 1 };
