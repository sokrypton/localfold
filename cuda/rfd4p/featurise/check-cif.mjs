/**
 * The native mmCIF reader against `parseCifToAtomArray`, column by column, on every structure the
 * page ships.
 *
 * 🔴 WHAT IS COMPARED IS WHAT COMES STRAIGHT FROM THE FILE. The reference's atom array also
 * carries chemistry it DERIVES - `is_polymer`, `chem_comp_type`, `atomic_number`, the leaving-atom
 * flags, the id and entity annotations and the bonds - and the native reader does not produce
 * those yet. Comparing only what it claims is the honest bar; claiming the rest by omission is
 * how a half-finished reader gets trusted.
 *
 *     node cuda/rfd4p/featurise/check-cif.mjs
 */
import { decodeTextOrZstd } from "../../../shared/rfd4p/pipeline/index.js";
import { parseCifToAtomArray } from "../../../shared/rfd4p/pipeline/cif.js";
import {
  ensureStereochemistryAnnotations, computeChiralityFeatures,
} from "../../../shared/rfd4p/pipeline/chirality.js";
import {
  ensureChemicalFlags, addSrcComponentAnnotation, computeWithinChainSourceResIdx,
  realizeExpandableSegments, createSeededRng, filterUnwantedAtoms,
  addGlobalAtomIdAnnotation, annotateCovalentModificationsAndAtomize,
  addGlobalTokenIdAnnotation,
  finalizeAndCanonicalizeSequenceConditioning,
  featurizeExpandedSegments,
  materializeInternalCoords,
} from "../../../shared/rfd4p/pipeline/transforms.js";
import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const repo = new URL("../../../", import.meta.url).pathname;
const binary = `${repo}cuda/rfd4p/featurise/rfd4p-featurise`;
const work = mkdtempSync(join(tmpdir(), "rfd4p-cif-"));
// 🔴 EXACT, NOT THROUGH `Math.fround`. Comparing the float32 of both sides is how a NARROWING
// hides: the reader kept coordinates as float where the reference keeps doubles, every column
// still "matched", and the difference only showed six stages later in the chirality fit.
const eq = (p, q) => (Number.isNaN(p) && Number.isNaN(q)) || p === q;
// The chirality tensors ARE float32 on both sides (the reference writes a Float32Array), so they
// take the narrower comparison - and say so.
const eqf = (p, q) => (Number.isNaN(p) && Number.isNaN(q)) || Math.fround(p) === Math.fround(q);

/**
 * The same structure with one extra atom in its first residue of `resNames`, copied from a row
 * that is already there so every other column stays legal.
 *
 * 🔴 BECAUSE `filterUnwantedAtoms` HAS THREE RULES AND THE SEVEN SHIPPED STRUCTURES REACH ONE.
 * Counted: 32 hydrogens in `sm_sam` and NOT ONE `OXT` or `OP3` anywhere in the corpus - so the
 * leaving-oxygen and the nucleic-terminal rules were both untested, and deleting either left
 * this gate green on all seven. These two cases are what can see them. Both sides read the SAME
 * synthesised text, so a difference is still the port's and never the fixture's.
 */
function withExtraAtom(text, resNames, atomName, element) {
  const lines = text.split("\n");
  const head = lines.findIndex((l) => l.startsWith("_atom_site."));
  if (head < 0) throw new Error("no _atom_site loop");
  let end = head;
  while (lines[end]?.startsWith("_atom_site.")) end += 1;
  const cols = lines.slice(head, end).map((l) => l.trim().slice("_atom_site.".length));
  const at = (k) => cols.indexOf(k);
  const tokenise = (l) => l.match(/"[^"]*"|'[^']*'|\S+/g) ?? [];
  for (let i = end; i < lines.length; i += 1) {
    if (!lines[i].trim() || lines[i].startsWith("#") || lines[i].startsWith("loop_")) break;
    const tok = tokenise(lines[i]);
    if (tok.length !== cols.length) continue;
    if (!resNames.includes(tok[at("label_comp_id")])) continue;
    for (const [key, value] of [["type_symbol", element], ["label_atom_id", atomName],
      ["auth_atom_id", atomName]]) {
      if (at(key) >= 0) tok[at(key)] = value;
    }
    lines.splice(i, 0, tok.join(" "));
    return lines.join("\n");
  }
  throw new Error(`no row for ${resNames.join("/")}`);
}

/**
 * The same structure with a `condition_CoordinateGroup_1_atom` column giving the first `count`
 * atoms group 1, the next `count` group 2, and everything else -1.
 *
 * 🔴 BECAUSE NOT ONE SHIPPED STRUCTURE HAS A POSITIVE COORDINATE GROUP, so
 * `materializeInternalCoords` never reaches its `randomSO3` draw - and that draw is the one the
 * README has been warning about since the stage table was written. Two groups rather than one,
 * because the draw happens PER GROUP in ascending order and a single group cannot tell a
 * correct stream from one that draws once for all of them.
 */
function withCoordinateGroups(text, count = 12) {
  const lines = text.split("\n");
  const head = lines.findIndex((l) => l.startsWith("_atom_site."));
  let end = head;
  while (lines[end]?.startsWith("_atom_site.")) end += 1;
  const cols = lines.slice(head, end).map((l) => l.trim().slice("_atom_site.".length));
  const tokenise = (l) => l.match(/"[^"]*"|'[^']*'|\S+/g) ?? [];
  const existing = cols.indexOf("condition_CoordinateGroup_1_atom");
  let seen = 0;
  for (let i = end; i < lines.length; i += 1) {
    if (!lines[i].trim() || lines[i].startsWith("#") || lines[i].startsWith("loop_")) break;
    const tok = tokenise(lines[i]);
    if (tok.length !== cols.length + (existing < 0 ? 0 : 0)) continue;
    const gid = seen < count ? "1" : seen < count * 2 ? "2" : "-1";
    if (existing >= 0) tok[existing] = gid;
    else tok.push(gid);
    lines[i] = tok.join(" ");
    seen += 1;
  }
  if (existing < 0) lines.splice(end, 0, "_atom_site.condition_CoordinateGroup_1_atom ");
  return lines.join("\n");
}

/**
 * The same structure with `condition_InternalCoordinate_1_atom` already SET for atoms
 * `0..count-1`, in the `idx0`/`val0..val2` loop the three-wide annotations take.
 *
 * 🔴 BECAUSE A GROUP THAT IS ALREADY COMPLETE DRAWS NOTHING, and that is the half of the RNG
 * order the two-group case cannot see. Paired with `withCoordinateGroups` on a structure that
 * does NOT expand (so the file's atom indices are still the stage's), group 1 is skipped and
 * group 2 takes the FIRST rotation - a port that drew before the `allFinite` test would hand it
 * the second, with geometry that is perfectly valid and a stream that is one rotation out.
 */
function withInternalCoords(text, count = 12) {
  const rows = [];
  for (let i = 0; i < count; i += 1) rows.push(`${i} ${i * 0.5} ${i * 0.25} ${i * 0.125}`);
  const block = ["#", "loop_",
    "_condition_InternalCoordinate_1_atom.idx0",
    "_condition_InternalCoordinate_1_atom.val0",
    "_condition_InternalCoordinate_1_atom.val1",
    "_condition_InternalCoordinate_1_atom.val2",
    ...rows, "#", ""];
  return `${text.replace(/\s*$/, "")}\n${block.join("\n")}`;
}

/**
 * The same structure with every residue's sequence condition set back to `<M>`.
 *
 * 🔴 BECAUSE THE BACKBONE EXEMPTION IS INVISIBLE WHILE THE SEQUENCE IS ALREADY CONDITIONED.
 * `finalizeAndCanonicalizeSequenceConditioning` spreads "required" over a token and then reveals
 * the residue name only where the atom is NOT already conditioned - so on the shipped motifs,
 * whose sequence is given, removing the `is_*_backbone` test changes nothing at all. Masked, the
 * same structure separates a residue whose conditioned coordinates are backbone-only (which must
 * stay `<M>`) from one with a conditioned side chain (which must be revealed).
 */
function withMaskedSequence(text) {
  const lines = text.split("\n");
  const head = lines.findIndex((l) => l.startsWith("_atom_site."));
  let end = head;
  while (lines[end]?.startsWith("_atom_site.")) end += 1;
  const cols = lines.slice(head, end).map((l) => l.trim().slice("_atom_site.".length));
  const col = cols.indexOf("condition_sequence_1_residue");
  if (col < 0) throw new Error("no condition_sequence column");
  const tokenise = (l) => l.match(/"[^"]*"|'[^']*'|\S+/g) ?? [];
  for (let i = end; i < lines.length; i += 1) {
    if (!lines[i].trim() || lines[i].startsWith("#") || lines[i].startsWith("loop_")) break;
    const tok = tokenise(lines[i]);
    if (tok.length !== cols.length) continue;
    tok[col] = "<M>";
    lines[i] = tok.join(" ");
  }
  return lines.join("\n");
}

/**
 * The same structure with one of its chains duplicated under a new `label_asym_id`.
 *
 * 🔴 BECAUSE NO SHIPPED STRUCTURE HAS TWO PROTEIN CHAINS. Surveyed: every one of the seven has
 * at most ONE chain carrying backbone `C` atoms, so a cross-chain peptide bond - the only shape
 * that can see the `sameChain` clause - cannot be built from the corpus at all.
 */
function withDuplicatedChain(text, newAsym = "Z") {
  const lines = text.split("\n");
  const head = lines.findIndex((l) => l.startsWith("_atom_site."));
  let end = head;
  while (lines[end]?.startsWith("_atom_site.")) end += 1;
  const cols = lines.slice(head, end).map((l) => l.trim().slice("_atom_site.".length));
  const at = (k) => cols.indexOf(k);
  const tokenise = (l) => l.match(/"[^"]*"|'[^']*'|\S+/g) ?? [];
  let last = end;
  const rows = [];
  for (let i = end; i < lines.length; i += 1) {
    if (!lines[i].trim() || lines[i].startsWith("#") || lines[i].startsWith("loop_")) break;
    const tok = tokenise(lines[i]);
    if (tok.length === cols.length) { rows.push(tok); last = i; }
  }
  const src = rows.find((r) => r[at("label_atom_id")] === "C")?.[at("label_asym_id")];
  if (!src) throw new Error("no chain with a backbone C");
  const copies = rows.filter((r) => r[at("label_asym_id")] === src).map((r) => {
    const t = r.slice();
    t[at("label_asym_id")] = newAsym;
    if (at("auth_asym_id") >= 0) t[at("auth_asym_id")] = newAsym;
    return t.join(" ");
  });
  lines.splice(last + 1, 0, ...copies);
  return lines.join("\n");
}

/**
 * The same structure with a covalent bond DECLARED between a `C` and an `N` that are not
 * neighbours - a crosslink, in the single-value `_struct_conn` form `motif_idx_1prw` ships.
 *
 * 🔴 BECAUSE THE ADJACENCY CLAUSE IS OTHERWISE UNREACHABLE. `annotateCovalentModifications`
 * exempts a C-N bond only between CONSECUTIVE residues of one chain, and dropping `resDiff === 1`
 * left all nine cases above green - no shipped structure declares a C-N bond across a gap, so
 * the rule that makes a crosslink a modification rather than a backbone link was never asked.
 */
function withDeclaredBond(text, { gap = 4, crossChain = false } = {}) {
  const lines = text.split("\n");
  const head = lines.findIndex((l) => l.startsWith("_atom_site."));
  let end = head;
  while (lines[end]?.startsWith("_atom_site.")) end += 1;
  const cols = lines.slice(head, end).map((l) => l.trim().slice("_atom_site.".length));
  const at = (k) => cols.indexOf(k);
  const tokenise = (l) => l.match(/"[^"]*"|'[^']*'|\S+/g) ?? [];
  const rows = [];
  for (let i = end; i < lines.length; i += 1) {
    if (!lines[i].trim() || lines[i].startsWith("#") || lines[i].startsWith("loop_")) break;
    const tok = tokenise(lines[i]);
    if (tok.length === cols.length) rows.push(tok);
  }
  const field = (r, k) => r[at(k)];
  const emit = (first, second) => {
    const block = ["#",
      "_struct_conn.id                        x1",
      "_struct_conn.conn_type_id              covale",
      "_struct_conn.pdbx_value_order          sing",
      `_struct_conn.ptnr1_label_asym_id       ${field(first, "label_asym_id")}`,
      `_struct_conn.ptnr2_label_asym_id       ${field(second, "label_asym_id")}`,
      `_struct_conn.ptnr1_label_comp_id       ${field(first, "label_comp_id")}`,
      `_struct_conn.ptnr2_label_comp_id       ${field(second, "label_comp_id")}`,
      `_struct_conn.ptnr1_label_seq_id        ${field(first, "label_seq_id")}`,
      `_struct_conn.ptnr2_label_seq_id        ${field(second, "label_seq_id")}`,
      "_struct_conn.ptnr1_label_atom_id       C",
      "_struct_conn.ptnr2_label_atom_id       N",
      "_struct_conn.pdbx_ptnr1_PDB_ins_code   .",
      "_struct_conn.pdbx_ptnr2_PDB_ins_code   .",
      "#", ""];
    return `${text.replace(/\s*$/, "")}\n${block.join("\n")}`;
  };
  if (crossChain) {
    // Any C and N in DIFFERENT chains whose residue numbers are one apart - which chain letters
    // happen to carry those numbers is the file's business, so both ends are searched.
    const cs = rows.filter((r) => field(r, "label_atom_id") === "C");
    const ns = rows.filter((r) => field(r, "label_atom_id") === "N");
    for (const c of cs) {
      const n = ns.find((r) => field(r, "label_asym_id") !== field(c, "label_asym_id")
        && Math.abs(Number(field(r, "label_seq_id")) - Number(field(c, "label_seq_id"))) === 1);
      if (n) return emit(c, n);
    }
    throw new Error("no cross-chain C/N pair one residue apart");
  }
  const first = rows.find((r) => field(r, "label_atom_id") === "C");
  if (!first) throw new Error("no C atom");
  const second = rows.find((r) => field(r, "label_atom_id") === "N"
    && field(r, "label_asym_id") === field(first, "label_asym_id")
    && Number(field(r, "label_seq_id")) - Number(field(first, "label_seq_id")) >= gap);
  if (!second) throw new Error("no distant N atom");
  return emit(first, second);
}

// 🔴 THE EXAMPLES ARE ZSTD AND THE READER TAKES A PLAIN FILE, which it says rather than
// guessing. Decompression is the one thing on this path that is still JavaScript's.
const cases = new Map();
for (const f of readdirSync(`${repo}web/examples`).filter((x) => x.endsWith(".cif.zst")).sort()) {
  cases.set(f.replace(".cif.zst", ""),
    decodeTextOrZstd(new Uint8Array(readFileSync(`${repo}web/examples/${f}`))));
}
cases.set("ppi_pdl1+OXT",
  withExtraAtom(cases.get("ppi_pdl1"), ["ALA", "GLY", "SER", "LEU", "VAL"], "OXT", "O"));
cases.set("dna_7m5w+OP3",
  withExtraAtom(cases.get("dna_7m5w"), ["DA", "DC", "DG", "DT"], "OP3", "O"));
cases.set("ppi_pdl1+crosslink", withDeclaredBond(cases.get("ppi_pdl1")));
cases.set("ppi_pdl1+interchain",
  withDeclaredBond(withDuplicatedChain(cases.get("ppi_pdl1")), { crossChain: true }));
cases.set("motif_idx_1prw+masked", withMaskedSequence(cases.get("motif_idx_1prw")));
cases.set("motif_idx_1prw+coordgroups", withCoordinateGroups(cases.get("motif_idx_1prw")));
cases.set("5o45_cropped+icgroups",
  withInternalCoords(withCoordinateGroups(cases.get("5o45_cropped"))));

let structures = 0;
let differing = 0;
for (const [name, text] of cases) {
  const path = join(work, `${name.replace("+", "_plus_")}.cif`);
  writeFileSync(path, text);

  const js = parseCifToAtomArray(text);
  const lines = execFileSync(binary, ["x", `--dump-cif=${path}`],
    { encoding: "utf8", maxBuffer: 1 << 28 }).trim().split("\n");
  const n = Number(lines[0].split(" ")[1]);
  const rows = lines.slice(1)
    .filter((l) => !/^(extra|bond|id|ann|chiral|stereobond|chirfeat|chircenter|chirframe|chirowner|chirtetra|flag|srccomp|withinchain)\t?/.test(l))
    .map((l) => l.split("\t"));
  const named = (tag) => {
    const l = lines.find((x) => x.startsWith(`${tag}\t`));
    return l ? l.split("\t").slice(1).map(Number) : [];
  };
  const bonds = lines.filter((l) => l.startsWith("bond\t"))
    .map((l) => l.split("\t").slice(1).map(Number));
  const ids = lines.filter((l) => l.startsWith("id\t")).map((l) => l.split("\t").slice(1));
  ensureStereochemistryAnnotations(js);
  const chir = computeChiralityFeatures(js);
  // 🔴 THE ANNOTATION SET IS SNAPSHOT BEFORE THE TRANSFORMS RUN. `addSrcComponentAnnotation`
  // ADDS one, and comparing afterwards asks the reader for a column the reader is not the stage
  // that produces - the native side dumps both, at the point each is made.
  // 🔴 THE VALUES TOO, NOT ONLY THE NAMES. An annotated structure CARRIES
  // `annotation_src-component_1_atom` and `addSrcComponentAnnotation` then OVERWRITES it, so
  // comparing afterwards asks the reader to have produced the transform's answer.
  const annotationsFromCif = Object.keys(js.annotations).sort().join(" ");
  const cifAnnotationValues = Object.fromEntries(
    Object.entries(js.annotations).map(([k, v]) => [k, Array.isArray(v[0]) ? v.map((r) => r.slice()) : v.slice()]));
  ensureChemicalFlags(js);
  addSrcComponentAnnotation(js);
  computeWithinChainSourceResIdx(js);
  const ann = new Map();
  for (const l of lines.filter((x) => x.startsWith("ann\t"))) {
    const p = l.split("\t");
    ann.set(p[2], { kind: p[1], values: p.slice(3) });
  }

  const wrong = [];
  if (n !== js.res_name.length) wrong.push(`n ${n} vs ${js.res_name.length}`);
  const columns = [
    ["element", 0, (v) => v, js.element],
    ["atom_name", 1, (v) => v, js.atom_name],
    ["res_name", 2, (v) => v, js.res_name],
    ["chain_id", 3, (v) => v, js.chain_id],
    ["ins_code", 4, (v) => v, js.ins_code],
    ["label_alt_id", 5, (v) => v, js.label_alt_id],
    ["res_id", 6, Number, js.res_id],
    ["auth_seq_id", 7, Number, js.auth_seq_id],
    ["atom_id", 8, Number, js.atom_id],
    ["label_entity_id", 9, Number, js.label_entity_id],
    ["hetero", 10, Number, js.hetero.map((v) => (v ? 1 : 0))],
    ["x", 11, Number, js.coord.map((v) => v[0]), eq],
    ["y", 12, Number, js.coord.map((v) => v[1]), eq],
    ["z", 13, Number, js.coord.map((v) => v[2]), eq],
    ["b_factor", 14, Number, js.b_factor, eq],
    ["occupancy", 15, Number, js.occupancy, eq],
    // The derived half: the charge after its fallbacks, the atomic number, the aromatic and
    // leaving flags, the chain type and the component type.
    ["charge", 16, Number, js.charge],
    ["atomic_number", 17, Number, js.atomic_number],
    ["is_aromatic", 18, Number, js.is_aromatic.map((v) => (v ? 1 : 0))],
    ["is_leaving_atom", 19, Number, js.is_leaving_atom.map((v) => (v ? 1 : 0))],
    ["chain_type", 20, Number, js.chain_type],
    ["chem_comp_type", 21, (v) => v, js.chem_comp_type],
  ];
  for (const [label, at, cast, want, how] of columns) {
    const same = how ?? (typeof want[0] === "number" ? eq : (p, q) => p === q);
    let bad = 0;
    for (let i = 0; i < Math.min(rows.length, want.length); i += 1) {
      if (!same(cast(rows[i][at]), want[i])) bad += 1;
    }
    if (rows.length !== want.length) wrong.push(`${label} length`);
    else if (bad) wrong.push(`${label} ${bad}`);
  }
  // 🔴 THE BOND LIST IS COMPARED IN ORDER, NOT AS A SET. Later stages index into it, so two
  // lists with the same contents in a different order are two different inputs.
  if (bonds.length !== js.bonds.length) {
    wrong.push(`bonds ${bonds.length} vs ${js.bonds.length}`);
  } else {
    const bad = bonds.filter((b, i) =>
      b[0] !== js.bonds[i][0] || b[1] !== js.bonds[i][1] || b[2] !== js.bonds[i][2]).length;
    if (bad) wrong.push(`bonds ${bad}`);
  }

  // The id and entity annotations, which the bonds decide.
  const idColumns = [
    ["pn_unit_id", 0, (v) => v, js.pn_unit_id],
    ["molecule_id", 1, Number, js.molecule_id],
    ["chain_entity", 2, Number, js.chain_entity],
    ["pn_unit_entity", 3, Number, js.pn_unit_entity],
    ["molecule_entity", 4, Number, js.molecule_entity],
    ["chain_iid", 5, (v) => v, js.chain_iid],
    ["pn_unit_iid", 6, (v) => v, js.pn_unit_iid],
    ["molecule_iid", 7, Number, js.molecule_iid],
  ];
  for (const [label, at, cast, want] of idColumns) {
    if (!want) continue;
    if (ids.length !== want.length) { wrong.push(`${label} length`); continue; }
    const bad = ids.filter((r, i) => cast(r[at]) !== want[i]).length;
    if (bad) wrong.push(`${label} ${bad}`);
  }

  // 🔴 AND THE ANNOTATION SET ITSELF, not only the values. A column silently absent is a
  // conditioning silently dropped, which is the failure this whole featuriser is written against.
  const wantNames = annotationsFromCif;
  const gotNames = [...ann.keys()].sort().join(" ");
  if (wantNames !== gotNames) wrong.push(`annotations {${gotNames}} vs {${wantNames}}`);
  else {
    for (const name of annotationsFromCif.split(" ").filter(Boolean)) {
      const want = cifAnnotationValues[name];
      const got = ann.get(name);
      const flat = Array.isArray(want[0]) ? want.flat() : want;
      if (got.values.length !== flat.length) { wrong.push(`${name} length`); continue; }
      let bad = 0;
      for (let i = 0; i < flat.length; i += 1) {
        const w = flat[i];
        const g = got.kind === "s" ? got.values[i] : Number(got.values[i]);
        if (typeof w === "boolean") { if ((w ? 1 : 0) !== g) bad += 1; }
        else if (typeof w === "number") { if (!eq(g, w)) bad += 1; }
        else if (String(w) !== String(g)) bad += 1;
      }
      if (bad) wrong.push(`${name} ${bad}`);
    }
  }

  // The stereochemistry: which atoms are chiral centres and which double bonds are stereogenic.
  for (const [label, got, want] of [
    ["chiral_type", named("chiral"), js.chiral_type],
    ["stereogenic_bond_atoms", named("stereobond"),
     js.stereogenic_bond_atoms.map((v) => (v ? 1 : 0))],
  ]) {
    if (got.length !== want.length) { wrong.push(`${label} length`); continue; }
    const bad = got.filter((v, i) => v !== want[i]).length;
    if (bad) wrong.push(`${label} ${bad}`);
  }

  // The four chirality tensors, plus the centre flag.
  for (const [label, got, want] of [
    ["chirality", named("chirfeat"), Array.from(chir.chirality.data)],
    ["is_chiral_center", named("chircenter"), Array.from(chir.is_chiral_center)],
    ["chiral_frame_atoms", named("chirframe"), Array.from(chir.chiral_frame_atoms.data)],
    ["chiral_slot_owner", named("chirowner"), Array.from(chir.chiral_slot_owner.data)],
    ["tetrahedral_vertices", named("chirtetra"), Array.from(chir.tetrahedral_vertices.data)],
  ]) {
    if (got.length !== want.length) { wrong.push(`${label} length ${got.length}/${want.length}`); continue; }
    const bad = got.filter((v, i) => !eqf(v, want[i])).length;
    if (bad) wrong.push(`${label} ${bad}`);
  }

  // `ensureChemicalFlags`' seventeen booleans, and the two annotations beside them.
  const flagLines = new Map();
  for (const l of lines.filter((x) => x.startsWith("flag\t"))) {
    const p = l.split("\t");
    flagLines.set(p[1], p.slice(2).map(Number));
  }
  for (const [label, want] of Object.entries(js).filter(([k]) => flagLines.has(k))) {
    const got = flagLines.get(label);
    if (got.length !== want.length) { wrong.push(`${label} length`); continue; }
    const bad = got.filter((v, i) => v !== (want[i] ? 1 : 0)).length;
    if (bad) wrong.push(`${label} ${bad}`);
  }
  if (flagLines.size !== 17) wrong.push(`flags ${flagLines.size}`);
  {
    const l = lines.find((x) => x.startsWith("srccomp\t"));
    const got = l ? l.split("\t").slice(1) : [];
    const want = js.annotations["annotation_src-component_1_atom"];
    if (got.length !== want.length) wrong.push("src-component length");
    else {
      const bad = got.filter((v, i) => v !== want[i]).length;
      if (bad) wrong.push(`src-component ${bad}`);
    }
    const w2 = js.within_chain_source_res_idx;
    const g2 = named("withinchain");
    if (w2 && (g2.length !== w2.length || g2.some((v, i) => v !== w2[i]))) {
      wrong.push("within_chain_source_res_idx");
    }
  }

  // 🔴 AND THE SLICE/CONCAT ROUND TRIP, at split points that cut through a residue and a bond.
  // Those two are the only primitives the rest of the pipeline has for inserting and dropping
  // atoms; a column they forget keeps its OLD values through an insertion, which is a wrong
  // answer in a field nothing else looks at. Cutting the array in two and putting it back must
  // reproduce every byte of the dump.
  for (const k of [1, 2, 17, Math.max(1, n - 1)]) {
    if (k >= n) continue;
    const again = execFileSync(binary, ["x", `--dump-cif=${path}`, `--split=${k}`],
      { encoding: "utf8", maxBuffer: 1 << 28 });
    if (again !== execFileSync(binary, ["x", `--dump-cif=${path}`],
      { encoding: "utf8", maxBuffer: 1 << 28 })) {
      wrong.push(`round trip at ${k}`);
    }
  }

  // `realizeExpandableSegments`: each sentinel becomes its run of masked residues, everything
  // after it shifts, and the bonds are rebuilt and re-sorted. Compared on the columns it TOUCHES.
  {
    const realized = realizeExpandableSegments(js, { rng: createSeededRng(0) });
    const out2 = execFileSync(binary, ["x", `--dump-cif=${path}`, "--realize"],
      { encoding: "utf8", maxBuffer: 1 << 28 }).split("\n");
    const got = out2.filter((l) => l.startsWith("rz\t")).map((l) => l.split("\t").slice(1));
    const gotBonds = out2.filter((l) => l.startsWith("rzbond\t"))
      .map((l) => l.split("\t").slice(1).map(Number));
    if (got.length !== realized.nAtoms) {
      wrong.push(`realized ${got.length} vs ${realized.nAtoms}`);
    } else {
      const after = [
        ["rz res_name", 0, (v) => v, realized.res_name],
        ["rz atom_name", 1, (v) => v, realized.atom_name],
        ["rz element", 2, (v) => v, realized.element],
        ["rz res_id", 3, Number, realized.res_id],
        ["rz auth_seq_id", 4, Number, realized.auth_seq_id],
        ["rz _expseg_uid", 5, Number, realized.annotations._expseg_uid],
        ["rz x", 6, Number, realized.coord.map((c) => c[0])],
        ["rz y", 7, Number, realized.coord.map((c) => c[1])],
        ["rz z", 8, Number, realized.coord.map((c) => c[2])],
        ["rz occupancy", 9, Number, realized.occupancy],
      ];
      for (const [label, at, cast, want] of after) {
        // 🔴 AN ABSENT ANNOTATION IS NOT A ZERO ONE. `_expseg_uid` is never created where there
        // was nothing to expand, because the stage returns first - so a structure without an
        // expandable segment has no column, not a column of -1, and comparing one against the
        // other would report every atom.
        if (!want) continue;
        const bad = got.filter((r, i) => !eq(cast(r[at]), want[i])).length;
        if (bad) wrong.push(`${label} ${bad}`);
      }
      if (gotBonds.length !== realized.bonds.length) wrong.push("rz bonds length");
      else {
        const bad = gotBonds.filter((b, i) => b[0] !== realized.bonds[i][0]
          || b[1] !== realized.bonds[i][1] || b[2] !== realized.bonds[i][2]).length;
        if (bad) wrong.push(`rz bonds ${bad}`);
      }
    }
  }

  // `filterUnwantedAtoms`, `addGlobalAtomIdAnnotation`,
  // `annotateCovalentModificationsAndAtomize` and `addGlobalTokenIdAnnotation`, on top of the
  // realised array - the four stages that decide which atoms survive and where the tokens fall.
  {
    const rng = createSeededRng(0);
    const realized = realizeExpandableSegments(js, { rng });
    const filtered = filterUnwantedAtoms(realized);
    addGlobalAtomIdAnnotation(filtered);
    annotateCovalentModificationsAndAtomize(filtered);
    addGlobalTokenIdAnnotation(filtered);
    finalizeAndCanonicalizeSequenceConditioning(filtered);
    // `featurizeExpandedSegments` can DELETE atoms, so it runs before the count is compared and
    // its result is what every column below is held to.
    const expanded = featurizeExpandedSegments(filtered);
    // 🔴 THE SAME GENERATOR THE REALISATION USED. `materializeInternalCoords` draws from the
    // pipeline's one stream, so the gate must advance a single `rng` across both stages or it
    // agrees only where neither draws.
    materializeInternalCoords(expanded, { rng });
    const out3 = execFileSync(binary, ["x", `--dump-cif=${path}`, "--atomize"],
      { encoding: "utf8", maxBuffer: 1 << 28 }).split("\n");
    const got = out3.filter((l) => l.startsWith("az\t")).map((l) => l.split("\t").slice(1));
    const gotBonds = out3.filter((l) => l.startsWith("azbond\t"))
      .map((l) => l.split("\t").slice(1).map(Number));
    if (got.length !== expanded.nAtoms) {
      wrong.push(`atomized ${got.length} vs ${expanded.nAtoms}`);
    } else {
      const after = [
        ["az res_name", 0, (v) => v, expanded.res_name],
        ["az atom_name", 1, (v) => v, expanded.atom_name],
        ["az element", 2, (v) => v, expanded.element],
        ["az res_id", 3, Number, expanded.res_id],
        ["az atom_id", 4, Number, expanded.atom_id],
        ["az token_id", 5, Number, expanded.token_id],
        ["az atomize", 6, Number, expanded.atomize.map((v) => (v ? 1 : 0))],
        ["az is_covalent_modification", 7, Number,
          expanded.is_covalent_modification.map((v) => (v ? 1 : 0))],
      ];
      for (const [label, at, cast, want] of after) {
        if (!want) continue;
        const bad = got.filter((r, i) => !eq(cast(r[at]), want[i])).length;
        if (bad) wrong.push(`${label} ${bad}`);
      }
      if (gotBonds.length !== expanded.bonds.length) wrong.push("az bonds length");
      else {
        const bad = gotBonds.filter((b, i) => b[0] !== expanded.bonds[i][0]
          || b[1] !== expanded.bonds[i][1] || b[2] !== expanded.bonds[i][2]).length;
        if (bad) wrong.push(`az bonds ${bad}`);
      }

      // `finalizeAndCanonicalizeSequenceConditioning`: the settled sequence, and the probability
      // columns that SURVIVE it - compared as a SET of names as well as values, because the
      // stage deletes a column that ends up entirely NaN and an absent column is not an empty one.
      // `featurizeExpandedSegments` zeroes every expanded segment's coordinates, so the
      // coordinates are compared here and not only at the realise stage.
      const xyz = out3.filter((l) => l.startsWith("azxyz\t"))
        .map((l) => l.split("\t").slice(1).map(Number));
      if (xyz.length !== expanded.nAtoms) wrong.push(`az xyz ${xyz.length}`);
      else {
        const bad = xyz.filter((c, i) => !eq(c[0], expanded.coord[i][0])
          || !eq(c[1], expanded.coord[i][1]) || !eq(c[2], expanded.coord[i][2])).length;
        if (bad) wrong.push(`az xyz ${bad}`);
      }
      // `materializeInternalCoords`: the group column it creates where there is none, and the
      // internal coordinates it fills - which is where the pipeline's RNG is next consumed.
      const ic = out3.filter((l) => l.startsWith("azic\t"))
        .map((l) => l.split("\t").slice(1).map(Number));
      const wantGroup = expanded.annotations["condition_CoordinateGroup_1_atom"];
      const wantIc = expanded.annotations["condition_InternalCoordinate_1_atom"];
      if (ic.length !== expanded.nAtoms) wrong.push(`az ic ${ic.length}`);
      else {
        const badG = ic.filter((r, i) => r[0] !== wantGroup[i]).length;
        if (badG) wrong.push(`az coord-group ${badG}`);
        const badI = ic.filter((r, i) => !eq(r[1], wantIc[i][0]) || !eq(r[2], wantIc[i][1])
          || !eq(r[3], wantIc[i][2])).length;
        if (badI) wrong.push(`az internal-coord ${badI}`);
      }
      const seq = out3.filter((l) => l.startsWith("azseq\t")).map((l) => l.split("\t")[1]);
      const wantSeq = expanded.annotations["condition_sequence_1_residue"];
      if (seq.length !== wantSeq.length) wrong.push(`az sequence ${seq.length} vs ${wantSeq.length}`);
      else {
        const bad = seq.filter((v, i) => v !== wantSeq[i]).length;
        if (bad) wrong.push(`az sequence ${bad}`);
      }
      const gotProb = new Map(out3.filter((l) => l.startsWith("azprob\t")).map((l) => {
        const f = l.split("\t");
        return [f[1], f.slice(2).map(Number)];
      }));
      const wantProb = new Map(Object.entries(expanded.annotations)
        .filter(([k]) => k.startsWith("condition_sequence-probability-"))
        .map(([k, v]) => [k.slice("condition_sequence-probability-".length, -"_1_residue".length), v]));
      const onlyGot = [...gotProb.keys()].filter((k) => !wantProb.has(k));
      const onlyWant = [...wantProb.keys()].filter((k) => !gotProb.has(k));
      if (onlyGot.length) wrong.push(`az prob extra ${onlyGot.join(",")}`);
      if (onlyWant.length) wrong.push(`az prob missing ${onlyWant.join(",")}`);
      for (const [k, want] of wantProb) {
        const got = gotProb.get(k);
        if (!got) continue;
        const bad = got.filter((v, i) => !eq(v, want[i])).length;
        if (bad) wrong.push(`az prob ${k} ${bad}`);
      }
    }
  }

  structures += 1;
  if (wrong.length) differing += 1;
  console.log(`  ${name.padEnd(26)} ${String(n).padStart(5)} atoms  `
    + (wrong.length ? `DIFFER: ${wrong.join(", ")}` : "exact"));
}
console.log(`${structures} structures, ${differing} differing`);
process.exit(differing ? 1 : 0);
