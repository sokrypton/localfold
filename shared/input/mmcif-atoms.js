/**
 * The `_atom_site` loop of an mmCIF file, read into one record per atom.
 *
 * The page's one mmCIF reader: web/template-source.js parses an uploaded or
 * fetched template with it. Only the first model is kept, and a field the file
 * does not carry falls back to its label_* twin or a neutral default.
 *
 * PROVENANCE: copied from sokrypton/mpnn's `mpnn/pdb.js` (`parseCIFAtoms`, at
 * commit a80942f), which this tree mirrored whole while a design page used
 * ProteinMPNN. The page needed only this function, so the
 * mirror went (2026-11-16) and this is no longer kept in step with upstream.
 * One local addition: `labelSeq`, the position in the entity's full sequence
 * that an AF3 job's templateIndices count.
 */

/**
 * @typedef {object} RawAtom
 * @property {boolean} hetero
 * @property {string} name
 * @property {string} altLoc
 * @property {string} resName
 * @property {string} chain
 * @property {number} resSeq
 * @property {number} labelSeq
 * @property {string} iCode
 * @property {number} x
 * @property {number} y
 * @property {number} z
 * @property {number} occupancy
 * @property {string} element
 */

/** @returns {RawAtom[]} */
export function parseCIFAtoms(text) {
  const lines = text.split("\n");
  const atoms = [];
  let i = 0;
  while (i < lines.length) {
    if (lines[i].trim() !== "loop_") {
      i++;
      continue;
    }
    // Collect the column names of this loop.
    let j = i + 1;
    const columns = [];
    while (j < lines.length && lines[j].trim().startsWith("_")) {
      columns.push(lines[j].trim().split(/\s+/)[0]);
      j++;
    }
    if (!columns[0]?.startsWith("_atom_site.")) {
      i = j;
      continue;
    }
    const col = Object.fromEntries(columns.map((c, k) => [c.replace("_atom_site.", ""), k]));
    const pick = (row, key, fallback = "") => (col[key] === undefined ? fallback : row[col[key]]);

    let firstModel = null;
    for (; j < lines.length; j++) {
      const line = lines[j].trim();
      if (line === "" || line.startsWith("#") || line.startsWith("loop_")) break;
      const row = line.match(/'[^']*'|"[^"]*"|\S+/g)?.map((s) => s.replace(/^['"]|['"]$/g, ""));
      if (!row || row.length < columns.length) continue;

      const model = pick(row, "pdbx_PDB_model_num", "1");
      if (firstModel === null) firstModel = model;
      if (model !== firstModel) break;

      const iCode = pick(row, "pdbx_PDB_ins_code", "");
      const altLoc = pick(row, "label_alt_id", "");
      atoms.push({
        hetero: pick(row, "group_PDB", "ATOM") === "HETATM",
        name: pick(row, "auth_atom_id", pick(row, "label_atom_id", "")),
        altLoc: altLoc === "." || altLoc === "?" ? "" : altLoc,
        resName: pick(row, "auth_comp_id", pick(row, "label_comp_id", "")).toUpperCase(),
        chain: pick(row, "auth_asym_id", pick(row, "label_asym_id", "A")),
        resSeq: parseInt(pick(row, "auth_seq_id", pick(row, "label_seq_id", "0")), 10),
        // (the position in the entity's FULL sequence, unresolved residues counted - what an AF3 job's
        // templateIndices index; NaN where the file has none)
        labelSeq: parseInt(pick(row, "label_seq_id", "."), 10),
        iCode: iCode === "." || iCode === "?" ? "" : iCode,
        x: parseFloat(pick(row, "Cartn_x", "0")),
        y: parseFloat(pick(row, "Cartn_y", "0")),
        z: parseFloat(pick(row, "Cartn_z", "0")),
        occupancy: parseFloat(pick(row, "occupancy", "1")) || 0,
        element: pick(row, "type_symbol", "").toUpperCase(),
      });
    }
    i = j;
  }
  return atoms;
}
