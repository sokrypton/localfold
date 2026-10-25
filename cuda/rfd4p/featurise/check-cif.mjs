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
import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const repo = new URL("../../../", import.meta.url).pathname;
const binary = `${repo}cuda/rfd4p/featurise/rfd4p-featurise`;
const work = mkdtempSync(join(tmpdir(), "rfd4p-cif-"));
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
  const rows = lines.slice(1).filter((l) => !/^(extra|bond|id)\t/.test(l)).map((l) => l.split("\t"));
  const bonds = lines.filter((l) => l.startsWith("bond\t"))
    .map((l) => l.split("\t").slice(1).map(Number));
  const ids = lines.filter((l) => l.startsWith("id\t")).map((l) => l.split("\t").slice(1));

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
    ["x", 11, Number, js.coord.map((v) => v[0])],
    ["y", 12, Number, js.coord.map((v) => v[1])],
    ["z", 13, Number, js.coord.map((v) => v[2])],
    ["b_factor", 14, Number, js.b_factor],
    ["occupancy", 15, Number, js.occupancy],
    // The derived half: the charge after its fallbacks, the atomic number, the aromatic and
    // leaving flags, the chain type and the component type.
    ["charge", 16, Number, js.charge],
    ["atomic_number", 17, Number, js.atomic_number],
    ["is_aromatic", 18, Number, js.is_aromatic.map((v) => (v ? 1 : 0))],
    ["is_leaving_atom", 19, Number, js.is_leaving_atom.map((v) => (v ? 1 : 0))],
    ["chain_type", 20, Number, js.chain_type],
    ["chem_comp_type", 21, (v) => v, js.chem_comp_type],
  ];
  for (const [label, at, cast, want] of columns) {
    const eq = typeof want[0] === "number" ? eqf : (p, q) => p === q;
    let bad = 0;
    for (let i = 0; i < Math.min(rows.length, want.length); i += 1) {
      if (!eq(cast(rows[i][at]), want[i])) bad += 1;
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

  structures += 1;
  if (wrong.length) differing += 1;
  console.log(`  ${name.padEnd(26)} ${String(n).padStart(5)} atoms  `
    + (wrong.length ? `DIFFER: ${wrong.join(", ")}` : "exact"));
}
console.log(`${structures} structures, ${differing} differing`);
process.exit(differing ? 1 : 0);
