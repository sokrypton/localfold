/**
 * BioEmu2 samples, scored as BioEmu2 scores them: its physicality report and its steering potentials' energies.
 *
 *     node tools/score-bioemu2-samples.mjs --job=<AF3 job.json> <sample.pdb|.cif> [...] [--json]
 *
 * Per sample: whether the protein passes the high_quality physicality thresholds and the first check that failed
 * (shared/bioemu2/physicality.js), and every default potential's energy at the end of sampling (t = 0,
 * shared/bioemu2/potentials.js on steering-tables.js's tables, the CCD's ligand half included) - zero where the
 * structure is inside every potential's flat bottom. Then the mean over samples. It reads PDB (BioEmu2's own records,
 * the native and the page's) or mmCIF (af3-any-model's), by (chain, residue, atom name). A missing atom is an error -
 * unless --allow-missing, for af3-any-model's mmCIF, which writes the AF3 atom set and so drops a glycan's O1 that
 * BioEmu2 predicts: then every potential term touching a missing atom is left out, in every file alike, and the line
 * says which atoms.
 */
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { featuriseBioemu2, bioemu2JobFromAf3Json } from "../shared/bioemu2/featurise.js";
import { smilesTemplates } from "../shared/bioemu2/inputs.js";
import { BioEmu2Graph } from "../shared/bioemu2/graph.js";
import { checkPhysicality } from "../shared/bioemu2/physicality.js";
import { steeringTables, parseStereoTables, STEREO_COMMIT } from "../shared/bioemu2/steering-tables.js";
import { energies, hostTables, stepParameters, DEFAULT_SPECS } from "../shared/bioemu2/potentials.js";

const args = process.argv.slice(2);
const option = (name) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const files = args.filter((a) => !a.startsWith("--"));
const repo = new URL("..", import.meta.url).pathname;
const ccd = (code) => {
  for (const dir of [join(repo, "tools/fixtures/ccd"), join(homedir(), ".cache/localfold/ccd")]) {
    if (existsSync(join(dir, `${code}.cif`))) return readFileSync(join(dir, `${code}.cif`), "utf8");
  }
  throw new Error(`${code}: not in tools/fixtures/ccd or ~/.cache/localfold/ccd (fold it once natively to cache it)`);
};
const job = bioemu2JobFromAf3Json(JSON.parse(readFileSync(option("job"), "utf8")));
const featurised = featuriseBioemu2(job, { ccd, smilesTemplates: await smilesTemplates(job) });
const { atomKeys, topology, features } = featurised;
const stereo = parseStereoTables(readFileSync(join(homedir(), `.cache/alphafold3/bioemu2/stereo_chemistry.${STEREO_COMMIT}.yaml`), "utf8"));
const raw = steeringTables(topology, stereo, { asymId: features.asym_id, entityId: features.entity_id }, { ccd });
const tables = hostTables(raw);
const nPad = topology.nPad;

function coordinates(path) {
  const text = readFileSync(path, "utf8");
  const where = new Map();
  if (/^data_/m.test(text) && /_atom_site\./.test(text)) {
    const columns = [...text.matchAll(/^_atom_site\.(\S+)/gm)].map((m) => m[1]);
    const col = (name) => columns.indexOf(name);
    for (const line of text.split("\n")) {
      if (!/^(ATOM|HETATM)\s/.test(line)) continue;
      const v = line.trim().split(/\s+/);
      const chain = v[col("auth_asym_id") >= 0 ? col("auth_asym_id") : col("label_asym_id")];
      const seq = v[col("label_seq_id")] !== "." && col("label_seq_id") >= 0 ? v[col("label_seq_id")] : v[col("auth_seq_id")];
      where.set(`${chain}|${Number(seq)}|${v[col("label_atom_id")].replace(/^"|"$/g, "")}`,
                [Number(v[col("Cartn_x")]), Number(v[col("Cartn_y")]), Number(v[col("Cartn_z")])]);
    }
  } else {
    for (const line of text.split("\n")) {
      if (!/^(ATOM  |HETATM)/.test(line)) continue;
      where.set(`${line[21]}|${Number(line.slice(22, 26))}|${line.slice(12, 16).trim()}`,
                [Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54))]);
    }
  }
  const x = new Float64Array(nPad * 3);
  const missing = atomKeys.filter((k) => !where.has(k.join("|")));
  if (missing.length && !args.includes("--allow-missing")) {
    throw new Error(`${path} lacks ${missing.length} atom(s), e.g. ${JSON.stringify(missing.slice(0, 3))}`);
  }
  atomKeys.forEach((k, a) => { if (where.has(k.join("|"))) x.set(where.get(k.join("|")), a * 3); else absent.add(a); });
  return x;
}
const absent = new Set();
// (--leave-out=B:1:O1,...: the same terms left out of files that have the atoms, so two sets compare like for like)
for (const key of (option("leave-out") ?? "").split(",").filter(Boolean)) {
  const at = atomKeys.findIndex((k) => k.join(":") === key);
  if (at < 0) throw new Error(`--leave-out=${key}: no such atom (chain:residue:name)`);
  absent.add(at);
}

// every term touching an absent atom left out (--allow-missing): index tables and the per-term arrays beside them
function withoutAbsent(t) {
  if (absent.size === 0) return t;
  const out = { ...t };
  const keep = (rows) => Array.from(rows[0] ?? [], (_, k) => rows.every((r) => !absent.has(r[k])));
  const filterRows = (key, beside = []) => {
    const rows = t[key];
    if (!rows?.length || !rows[0]?.length) return;
    const ok = keep(rows);
    out[key] = rows.map((r) => r.filter((_, k) => ok[k]));
    for (const b of beside) if (t[b]) out[b] = t[b].filter((_, k) => ok[k]);
  };
  filterRows("bond_index", Object.keys(t).filter((k) => /^bond_(lower|upper)\.\d+$/.test(k)));
  filterRows("angle_index", Object.keys(t).filter((k) => /^angle_(lower|upper)\.\d+$/.test(k)));
  filterRows("polymer_local_clash_index");
  filterRows("rdkit_bounds_index", ["rdkit_bounds_bond_mask", "rdkit_bounds_angle_mask", "rdkit_upper_bounds", "rdkit_lower_bounds"]);
  filterRows("chiral_atom_index", ["chiral_atom_orientations"]);
  filterRows("stereo_bond_index", ["stereo_bond_orientations"]);
  filterRows("planar_improper_index");
  for (const m of ["polymer_steric_atom_mask", "interchain_steric_atom_mask"]) out[m] = Uint8Array.from(t[m], (v, a) => (absent.has(a) ? 0 : v));
  return out;
}

const frames = files.map(coordinates);
const scored = withoutAbsent(tables);
if (absent.size) console.log(`(left out: every term touching ${[...absent].map((a) => atomKeys[a].join(":")).join(", ")}, absent from a file)`);
const physicality = checkPhysicality(topology, new BioEmu2Graph(topology).bonds, frames, "high_quality");
const p = stepParameters(0);
const rows = frames.map((x, f) => {
  const e = energies(x, scored, p);
  return { file: files[f], ok: physicality[f].ok, reason: physicality[f].reason, stats: physicality[f].stats,
           energies: Object.fromEntries(DEFAULT_SPECS.map(([kind], i) => [kind, e[i]])) };
});
const mean = Object.fromEntries(DEFAULT_SPECS.map(([kind]) => [kind, rows.reduce((s, r) => s + r.energies[kind], 0) / rows.length]));
const summary = { samples: rows.length, pass: rows.filter((r) => r.ok).length,
                  reasons: rows.map((r) => r.reason), meanEnergy: mean, ligandSteered: raw.ligandSteered };
if (args.includes("--json")) console.log(JSON.stringify({ rows, summary }));
else {
  for (const r of rows) {
    console.log(`${r.file}: ${r.ok ? "passes" : `rejected (${r.reason})`}  `
      + DEFAULT_SPECS.map(([kind]) => `${kind.replace("Potential", "")} ${r.energies[kind].toFixed(3)}`).join("  "));
  }
  console.log(`physicality ${summary.pass}/${summary.samples} pass; mean energies `
    + DEFAULT_SPECS.map(([kind]) => `${kind.replace("Potential", "")} ${mean[kind].toFixed(3)}`).join("  "));
}
