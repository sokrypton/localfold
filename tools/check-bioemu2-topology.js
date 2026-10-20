/**
 * BioEmu2's bonded graph, steering tables and physicality report against af3-any-model's port.
 *
 *     node tools/check-bioemu2-topology.js --dump=oracle-dumps/bioemu2-topology-6mrr.json --job=<job.json>
 *         [--features-dump=oracle-dumps/bioemu2-6mrr.json] [--stereo=<stereo_chemistry.yaml>]
 *
 * The dump is tools/oracle/dump_bioemu2_topology.py's: colab's Graph, its
 * `tables()` and `physicality.check` on a sample and three noised copies of it
 * under all four presets. The graph and the integer tables must be EXACT, the
 * bounds to float32, and every physicality stat, verdict and first-failed
 * reason must agree. The ligand half of the tables (rdkit_*, chiral_*, stereo_*,
 * planar_*) is REPORTED, not asserted: it is not built here yet (see
 * shared/bioemu2/steering-tables.js).
 */
import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { featuriseBioemu2, bioemu2JobFromAf3Json } from "../shared/bioemu2/featurise.js";
import { BioEmu2Graph } from "../shared/bioemu2/graph.js";
import { steeringTables, parseStereoTables, STEREO_COMMIT } from "../shared/bioemu2/steering-tables.js";
import { checkPhysicality } from "../shared/bioemu2/physicality.js";
import { readBioemu2Dump, smilesTemplatesOf } from "./bioemu2-dump.js";

const option = (name, fallback) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const ccd = (code) => {
  const path = new URL(`fixtures/ccd/${code}.cif`, import.meta.url);
  if (!existsSync(path)) throw new Error(`${code}: not in tools/fixtures/ccd`);
  return readFileSync(path, "utf8");
};
const LIGAND_KEYS = ["rdkit_bounds_index", "rdkit_bounds_bond_mask", "rdkit_bounds_angle_mask", "rdkit_upper_bounds",
  "rdkit_lower_bounds", "chiral_atom_index", "chiral_atom_orientations", "stereo_bond_index",
  "stereo_bond_orientations", "planar_bond_index"];

const flat = (x) => (Array.isArray(x) ? x.flatMap((v) => (typeof v === "object" && v !== null ? flat(v) : [Number(v)]))
  : Array.from(x, Number));

export function compareTopology(dumpPath, jobPath, featuresDump, stereoPath) {
  const dump = JSON.parse(readFileSync(dumpPath, "utf8").replace(/\bNaN\b/g, "null"));
  const job = bioemu2JobFromAf3Json(JSON.parse(readFileSync(jobPath, "utf8")));
  const smiles = featuresDump ? smilesTemplatesOf(readBioemu2Dump(featuresDump).stages) : {};
  const ours = featuriseBioemu2(job, { ccd, smilesTemplates: smiles });
  const g = new BioEmu2Graph(ours.topology);
  const rows = [];
  let failures = 0;
  const same = (name, a, b, tolerance = 0) => {
    const x = flat(a);
    const y = flat(b);
    let differ = x.length !== y.length ? Math.max(x.length, y.length) : 0;
    let worst = 0;
    if (!differ) {
      for (let i = 0; i < x.length; i += 1) {
        const d = Math.abs(x[i] - y[i]);
        if (!(d <= tolerance) && !(x[i] === y[i])) differ += 1;
        if (Number.isFinite(d)) worst = Math.max(worst, d);
      }
    }
    rows.push(`${name.padEnd(34)} ${differ ? `${differ} differ (${x.length} against ${y.length})` : `exact (${x.length})`}`);
    return differ === 0;
  };
  failures += !same("graph.bonds", g.bonds, dump.graph.bonds);
  failures += !same("graph.angles", g.angles, dump.graph.angles);
  failures += !same("graph.exclusions", g.nonbondedExclusions(), dump.graph.exclusions);
  failures += !same("graph.ordinary_x", [...g.ordinaryX].sort((a, b) => a - b), dump.graph.ordinary_x);
  const labelsAgree = g.labels.every((l, i) => l[0] === dump.graph.labels[i][0] && l[1] === dump.graph.labels[i][1]);
  rows.push(`${"graph.labels".padEnd(34)} ${labelsAgree ? "exact" : "DIFFER"}`);
  failures += !labelsAgree;

  const stereoText = readFileSync(stereoPath ?? `${homedir()}/.cache/alphafold3/bioemu2/stereo_chemistry.${STEREO_COMMIT}.yaml`, "utf8");
  const tables = steeringTables(ours.topology, parseStereoTables(stereoText),
                                { asymId: ours.features.asym_id, entityId: ours.features.entity_id });
  for (const [k, want] of Object.entries(dump.tables)) {
    if (LIGAND_KEYS.includes(k)) {
      rows.push(`${`tables.${k}`.padEnd(34)} (ligand tier: ${flat(want).length} values in the reference, not built here)`);
      continue;
    }
    if (!(k in tables)) { rows.push(`${`tables.${k}`.padEnd(34)} MISSING`); failures += 1; continue; }
    const tolerance = /lower|upper|radius|_ref$/.test(k) ? 1e-6 : 0;
    failures += !same(`tables.${k}`, tables[k], want, tolerance);
  }

  const frames = dump.frames.map((f) => Float64Array.from(f.flat()));
  const where = new Map(ours.atomKeys.map((k, i) => [k.join("|"), i]));
  const declared = (job.bondedAtomPairs ?? []).map(([a, b]) => [where.get(a.join("|")), where.get(b.join("|"))]);
  for (const [mode, want] of Object.entries(dump.physicality)) {
    const got = checkPhysicality(ours.topology, g.bonds, frames, mode, declared);
    const verdicts = got.every((r, f) => r.ok === want.ok[f] && r.reason === want.reasons[f]);
    let statsAgree = true;
    const bad = [];
    for (const [k, values] of Object.entries(want.stats)) {
      values.forEach((v, f) => { if (got[f].stats[k] !== v) { statsAgree = false; bad.push(`${k}[${f}] ${got[f].stats[k]} vs ${v}`); } });
    }
    rows.push(`${`physicality.${mode}`.padEnd(34)} ${verdicts && statsAgree ? "exact" : `DIFFER ${bad.slice(0, 4).join("; ")}`}`
      + `  (${got.map((r) => r.reason ?? "passed").join(", ")})`);
    failures += !(verdicts && statsAgree);
  }
  return { rows, failures };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { rows, failures } = compareTopology(option("dump"), option("job"), option("features-dump"), option("stereo"));
  for (const row of rows) console.log(`  ${row}`);
  if (failures) { console.error(`${failures} item(s) differ`); process.exit(1); }
  console.log("graph, polymer tables and physicality agree");
}
