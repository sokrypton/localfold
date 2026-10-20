/**
 * Is cuda/featurise/bioemu2_features.h the JavaScript featuriser, byte for byte?
 *
 *     node tools/check-native-bioemu2.mjs [--probe=<bioemu2-probe binary>]
 *
 * Builds cuda/featurise/bioemu2-probe.cpp (the native featuriser's own flags,
 * -ffp-contract=off) unless handed one, runs it and shared/bioemu2/featurise.js
 * on every case below, and holds every array to the same INTEGERS and the same
 * float BIT PATTERNS - the same rule npm run test:featurise holds every other
 * native featuriser to. A component the fixtures lack is read from the native
 * side's own cache (~/.cache/localfold/ccd), so both sides read one file.
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { featuriseBioemu2, bioemu2JobFromAf3Json, parseContacts } from "../shared/bioemu2/featurise.js";
import { readBioemu2Dump, smilesTemplatesOf } from "./bioemu2-dump.js";
import { steeringTables, parseStereoTables, STEREO_COMMIT } from "../shared/bioemu2/steering-tables.js";
import { hostTables } from "../shared/bioemu2/potentials.js";

const option = (name) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const repo = new URL("..", import.meta.url).pathname;
const fixtures = join(repo, "tools/fixtures/ccd");
const cache = process.env.LOCALFOLD_CCD_DIR ?? join(homedir(), ".cache/localfold/ccd");
const ccd = (code) => {
  for (const dir of [fixtures, cache]) if (existsSync(join(dir, `${code}.cif`))) return readFileSync(join(dir, `${code}.cif`), "utf8");
  throw new Error(`${code}: in neither ${fixtures} nor ${cache}`);
};

let probe = option("probe");
if (!probe) {
  probe = join(mkdtempSync(join(tmpdir(), "bioemu2-probe-")), "bioemu2-probe");
  execFileSync("g++", ["-std=c++17", "-O2", "-ffp-contract=off", "-pthread", join(repo, "cuda/featurise/bioemu2-probe.cpp"), "-o", probe],
               { stdio: "inherit" });
}

const jobs = join(repo, "tools/fixtures/bioemu2");
const CASES = [
  { name: "6mrr", job: `${jobs}/6mrr.json` },
  { name: "6mrr + SEP + GOL", job: `${jobs}/6mrr-gol-sep.json` },
  { name: "RNA + DNA", job: `${jobs}/rna-dna.json` },
  { name: "6mrr + NAG + retinal", job: `${jobs}/6mrr-nag-ret.json` },
  { name: "SMILES", job: `${jobs}/smiles.json`, smilesDump: join(repo, "oracle-dumps/bioemu2-smiles.json") },
  { name: "equilibrium + contacts", job: `${jobs}/6mrr.json`, mode: "equilibrium", contacts: join(repo, "tools/fixtures/bioemu2/contacts.json") },
  { name: "a glycan, branched, bonded", job: join(repo, "tools/fixtures/glycan-standalone.json") },
  { name: "a heteromer with paired MSAs", job: `${jobs}/heteromer.json` },
];

// the stereochemistry tables steering reads (CCP4 MON_LIB, fetched, never shipped): steering-tables.js's STEREO_URL
const stereoPath = join(homedir(), `.cache/alphafold3/bioemu2/stereo_chemistry.${STEREO_COMMIT}.yaml`);
const stereo = existsSync(stereoPath) ? parseStereoTables(readFileSync(stereoPath, "utf8")) : null;
if (!stereo) console.log(`  (no ${stereoPath}: the steering tables are not compared)`);

let failures = 0;
for (const c of CASES) {
  if (!existsSync(c.job)) { console.log(`  ${c.name}: SKIP (no ${c.job})`); continue; }
  const smiles = c.smilesDump && existsSync(c.smilesDump) ? smilesTemplatesOf(readBioemu2Dump(c.smilesDump).stages) : {};
  const ours = featuriseBioemu2(bioemu2JobFromAf3Json(JSON.parse(readFileSync(c.job, "utf8"))), {
    ccd, mode: c.mode ?? "structure", smilesTemplates: smiles,
    contacts: c.contacts ? parseContacts(JSON.parse(readFileSync(c.contacts, "utf8"))) : null,
  });
  const args = [`--job=${c.job}`, `--ccd-dir=${fixtures}`, `--mode=${c.mode ?? "structure"}`];
  if (existsSync(stereoPath)) args.push(`--steering=${stereoPath}`);
  if (c.contacts) args.push(`--contacts=${c.contacts}`);
  if (Object.keys(smiles).length) {
    const path = join(mkdtempSync(join(tmpdir(), "bioemu2-smiles-")), "templates.json");
    writeFileSync(path, JSON.stringify(Object.fromEntries(Object.entries(smiles).map(([k, t]) =>
      [k, { names: t.names, element: t.element, charge: t.charge, coords: Array.from(t.coords), bonds: t.bonds }]))));
    args.push(`--smiles-templates=${path}`);
  }
  let native;
  try {
    native = JSON.parse(execFileSync(probe, args, { maxBuffer: 1 << 30, env: { ...process.env, LOCALFOLD_CCD_DIR: cache } }).toString());
  } catch (error) {
    console.log(`  ${c.name}: the native featuriser failed: ${error.stderr ?? error.message}`);
    failures += 1;
    continue;
  }
  const differ = [];
  const names = new Set([...Object.keys(native.ints), ...Object.keys(native.floats)]);
  for (const [k, v] of Object.entries(ours.features)) if (v?.length !== undefined && k !== "msaRows") names.add(k);
  for (const k of names) {
    const mine = ours.features[k];
    if (k in native.ints) {
      const theirs = native.ints[k];
      if (mine === undefined || mine.length !== theirs.length || theirs.some((v, i) => v !== mine[i])) differ.push(k);
    } else if (k in native.floats) {
      const bits = new Uint32Array(Float32Array.from(mine ?? []).buffer);
      const theirs = native.floats[k];
      if (bits.length !== theirs.length || theirs.some((v, i) => v !== bits[i])) differ.push(k);
    } else {
      differ.push(`${k} (missing natively)`);
    }
  }
  const keys = native.atomKeys.length === ours.atomKeys.length && native.atomKeys.every((k, i) => k.join("|") === ours.atomKeys[i].join("|"));
  if (!keys) differ.push("atom keys");
  if (native.tokens !== ours.tokens || native.msaRows !== ours.msaRows) differ.push("counts");
  let steeringArrays = 0;
  if (stereo) {
    const t = steeringTables(ours.topology, stereo, { asymId: ours.features.asym_id, entityId: ours.features.entity_id }, { ccd });
    const host = hostTables(t);
    const mine = { ...t, ...host, bond_ref: t.bond_ref, angle_ref: t.angle_ref };
    const flatten = (v) => (Array.isArray(v) && v.length && (Array.isArray(v[0]) || ArrayBuffer.isView(v[0]))
      ? v.flatMap((r) => Array.from(r)) : Array.from(v));
    for (const [kind, table] of Object.entries(native.steering)) {
      for (const [k, theirs] of Object.entries(table)) {
        steeringArrays += 1;
        if (!(k in mine)) { differ.push(`steering.${k} (missing here)`); continue; }
        let values = flatten(mine[k]).map((v) => (v === true ? 1 : v === false ? 0 : v));
        if (kind === "floats") values = Array.from(new Uint32Array(Float32Array.from(values).buffer));
        if (values.length !== theirs.length || theirs.some((v, i) => v !== values[i])) differ.push(`steering.${k}`);
      }
    }
  }
  failures += differ.length ? 1 : 0;
  console.log(`  ${c.name}: ${ours.tokens} tokens, ${ours.atoms} atoms, ${ours.msaRows} MSA rows - ${differ.length
    ? `DIFFER: ${differ.join(", ")}` : `${names.size} feature and ${steeringArrays} steering arrays identical`}`);
}
if (failures) { console.error(`${failures} case(s) differ`); process.exit(1); }
console.log("the native BioEmu2 featuriser is the JavaScript's, byte for byte");
