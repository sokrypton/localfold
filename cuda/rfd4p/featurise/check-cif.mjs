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

let structures = 0;
let differing = 0;
for (const name of readdirSync(`${repo}web/examples`).filter((f) => f.endsWith(".cif.zst")).sort()) {
  // 🔴 THE EXAMPLES ARE ZSTD AND THE READER TAKES A PLAIN FILE, which it says rather than
  // guessing. Decompression is the one thing on this path that is still JavaScript's.
  const text = decodeTextOrZstd(new Uint8Array(readFileSync(`${repo}web/examples/${name}`)));
  const path = join(work, name.replace(".cif.zst", ".cif"));
  writeFileSync(path, text);

  const js = parseCifToAtomArray(text);
  const lines = execFileSync(binary, ["x", `--dump-cif=${path}`],
    { encoding: "utf8", maxBuffer: 1 << 28 }).trim().split("\n");
  const n = Number(lines[0].split(" ")[1]);
  const rows = lines.slice(1)
    .filter((l) => !/^(extra|bond|id|ann|chiral|stereobond|chirfeat|chircenter|chirframe|chirowner|chirtetra)\t?/.test(l))
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
  const wantNames = Object.keys(js.annotations).sort().join(" ");
  const gotNames = [...ann.keys()].sort().join(" ");
  if (wantNames !== gotNames) wrong.push(`annotations {${gotNames}} vs {${wantNames}}`);
  else {
    for (const [name, want] of Object.entries(js.annotations)) {
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

  structures += 1;
  if (wrong.length) differing += 1;
  console.log(`  ${name.padEnd(26)} ${String(n).padStart(5)} atoms  `
    + (wrong.length ? `DIFFER: ${wrong.join(", ")}` : "exact"));
}
console.log(`${structures} structures, ${differing} differing`);
process.exit(differing ? 1 : 0);
