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
 * 🔴 NOR ARE THE CHARGES, ON SOME COMPONENTS. AF3 reads `GetFormalCharge()`
 * off the RDKit molecule it built the conformer from, and only falls back to
 * this file's `charge` column when no conformer could be generated
 * (features.py:1626). RDKit reperceives: heme's dictionary entry puts -1 on two
 * carboxylate oxygens and two porphyrin nitrogens, and AF3 records 0 for all
 * four. Without RDKit in the page the dictionary's value is the honest choice -
 * it is the published chemistry - but it is not always AF3's, and the checker
 * reports the difference rather than hiding it. It reaches the model through
 * one linear into the atom conditioning, alongside the element and the name.
 *
 * 🔴 AND THE IDEAL COORDINATES ARE NOT AF3's. AF3 builds a fresh conformer per
 * instance with RDKit - fixed bond lengths and angles, random torsions - so its
 * reference positions for GOL bear no resemblance to the dictionary's, while
 * the atom names and elements match exactly. This is the same thing
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

/**
 * The column names of an mmCIF category and the rows under it.
 *
 * 🔴 A CATEGORY WITH ONE ROW IS NOT WRITTEN AS A LOOP, and that is how mmCIF
 * has always worked: `loop_` exists to avoid repeating item names, so a single
 * row is written as plain `_category.item value` pairs instead. Every monatomic
 * ion is such a category - MG, ZN, CL, K, FE2 - and reading only the looped
 * form found no atoms in any of them.
 *
 * It did not fail cleanly either. The item lines were taken for column headers,
 * their values discarded, and the "# #" line that follows was then accepted as
 * a two-field row - so MG parsed as one atom with no name, element zero and no
 * coordinates, and folded. A wrong answer, quietly.
 */
function loopOf(text, prefix) {
  const lines = text.split(/\r?\n/);
  const columns = [];
  const rows = [];
  // The unlooped form: item -> its value, gathered from the same lines that
  // name the columns.
  const single = new Map();
  let inHeader = false;
  let inBody = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith(`${prefix}.`)) {
      if (inBody) break;              // a second loop with the same prefix
      const [item, ...value] = trimmed.slice(prefix.length + 1).split(/\s+/);
      columns.push(item);
      if (value.length > 0) single.set(item, value.join(" "));
      inHeader = true;
      continue;
    }
    if (!inHeader) continue;
    // 🔴 A COMMENT IS NOT ALWAYS A LONE "#". files.rcsb.org ends ATP's atom loop
    // with the line "# #", which passed this test, tokenised into a two-field
    // "row", and became an atom with no coordinates - so every ATP fold died on
    // "ATP has no usable x coordinate" and the same went for any component
    // written that way. Anything made only of hashes and spaces is a comment.
    if (trimmed === "" || trimmed === "loop_" || /^[#\s]+$/.test(trimmed)) {
      if (inBody) break;
      continue;
    }
    if (trimmed.startsWith("_")) break;
    // Values may be quoted, and an atom name like O5' or "C1'" can carry one.
    const fields = trimmed.match(/'[^']*'|"[^"]*"|\S+/g)?.map(
      (value) => value.replace(/^['"]|['"]$/g, "")) ?? [];
    // 🔴 AND A ROW HAS AS MANY FIELDS AS THERE ARE COLUMNS. A short line is not
    // a row of this loop whatever else it is; accepting one puts undefined in
    // every column and the error that follows names the wrong thing entirely.
    // In these files a row is always one line, so the first mismatch is the end.
    if (fields.length !== columns.length) {
      if (inBody) break;
      continue;
    }
    inBody = true;
    rows.push(fields);
  }
  if (rows.length === 0 && single.size > 0) {
    rows.push(columns.map((item) => (single.get(item) ?? "").replace(/^['"]|['"]$/g, "")));
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
  //
  // 🔴 PER ATOM, NOT PER COLUMN, AND AN ION HAS NEITHER. The choice used to be
  // made once from whether the ideal COLUMNS existed, but a component can carry
  // the columns and leave them as "?" - which every monatomic ion does, because
  // a single atom has no conformer to describe. MG has the columns and no
  // values in them, so a protein with one magnesium failed the whole fold with
  // "MG has no usable x coordinate".
  //
  // 🔴 AND THE ORIGIN IS THE RIGHT ANSWER FOR EXACTLY ONE ATOM, not for a
  // missing coordinate in general. A conformer is only defined up to a rigid
  // motion and the diffusion head places the component itself, so for a lone
  // atom every point is equally correct and the origin is one of them. For a
  // component with several atoms a zero would be a WRONG geometry rather than
  // an arbitrary one - it would collapse that atom onto the others - so that
  // stays an error.
  const monatomic = atomLoop.rows.filter(
    (row) => !["H", "D"].includes((at(row, "type_symbol") ?? "").toUpperCase())).length === 1;
  const coordinate = (row, axis) => {
    for (const column of [`pdbx_model_Cartn_${axis}_ideal`, `model_Cartn_${axis}`]) {
      const parsed = Number.parseFloat(at(row, column));
      if (Number.isFinite(parsed)) return parsed;
    }
    if (monatomic) return 0;
    throw new Error(`${code} has no usable ${axis} coordinate`);
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
      element: ELEMENT_SYMBOLS.indexOf(symbol) + 1,
      charge: Number.parseFloat(at(row, "charge") ?? "0") || 0,
      // 🔴 A LEAVING ATOM IS PRESENT IN THE COMPONENT AND ABSENT FROM THE
      // POLYMER. The dictionary describes a free amino acid, so a modified
      // residue carries the OXT it would lose on forming a peptide bond -
      // SEP has eleven heavy atoms here and AF3 gives it TEN tokens in the
      // middle of a chain and eleven at the C-terminus. The flag is kept
      // rather than acted on, because which end a residue sits at is not a
      // property of the component. See polymerResidue below.
      leaving: (at(row, "pdbx_leaving_atom_flag") ?? "N").toUpperCase() === "Y",
    });
    Object.assign(atoms[atoms.length - 1], {
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
 * A component as it appears INSIDE a polymer chain, with its leaving atoms
 * resolved and its bonds renumbered to match.
 *
 * 🔴 THE OXT COMES BACK ONLY AT A C-TERMINUS, which is what AF3 does: a
 * phosphoserine is ten tokens in the middle of a chain and eleven at the end,
 * the extra one an OXT sitting where the dictionary puts it - between O and the
 * modification's own atoms - rather than appended. Every other leaving atom
 * goes, since it is exactly the atom the peptide bond replaced.
 *
 * 🔴 AND THE BONDS ARE RENUMBERED, because they index the atom list. Dropping
 * an atom without remapping them silently rewires the residue: SEP's bonds
 * would move one place along from OXT onwards, which is a molecule that does
 * not exist and reads as a plausible one.
 *
 * @param {{code: string, atoms: object[], bonds: object[]}} component
 * @param {boolean} isCTerminal
 */
export function polymerResidue(component, isCTerminal) {
  const keep = component.atoms.map(
    (atom) => !atom.leaving || (isCTerminal && atom.name === "OXT"));
  const renumbered = [];
  let next = 0;
  for (let index = 0; index < component.atoms.length; index += 1) {
    renumbered.push(keep[index] ? next++ : -1);
  }
  return {
    code: component.code,
    atoms: component.atoms.filter((_, index) => keep[index]),
    bonds: component.bonds
      .filter((bond) => renumbered[bond.from] >= 0 && renumbered[bond.to] >= 0)
      .map((bond) => ({ ...bond, from: renumbered[bond.from], to: renumbered[bond.to] })),
  };
}

/**
 * AF3's element numbering: the atomic number, so carbon is 6.
 *
 * 🔴 ONE-BASED BY POSITION IN THIS LIST, which is the same thing only because
 * the list starts at hydrogen and omits nothing. It is written out rather than
 * computed so that a gap would be visible.
 */
export const ELEMENT_SYMBOLS = [
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
