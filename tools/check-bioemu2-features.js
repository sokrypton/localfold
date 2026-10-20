/**
 * Does shared/bioemu2/featurise.js build what af3-any-model's BioEmu2 port builds, field by field?
 *
 *     node tools/check-bioemu2-features.js --dump=oracle-dumps/bioemu2-6mrr.json --job=<job.json>
 *         [--mode=equilibrium] [--contacts=<contacts.json>]
 *
 * The dump is tools/oracle/dump_bioemu2.py's, whose `feats.*` are colab's own
 * `bioemu2_features.featurise` on the same job. Integer fields must be EXACT;
 * float fields (the centred CCD coordinates, the profile, the deletion value)
 * are held to 1e-6 absolute, since a mean in numpy's pairwise summation and one
 * in a JavaScript loop round differently in the last place. The MSA is
 * compared as an index - the dump's one-hot is decoded first - and the atom keys
 * are compared too, since they are how a prediction is mapped back to atoms.
 *
 * Components the standard table lacks are read from tools/fixtures/ccd/<code>.cif.
 */
import { readFileSync, existsSync } from "node:fs";
import { featuriseBioemu2, bioemu2JobFromAf3Json, parseContacts, NUM_TOKENS } from "../shared/bioemu2/featurise.js";
import { readBioemu2Dump, smilesTemplatesOf } from "./bioemu2-dump.js";

const option = (name, fallback) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const ccd = (code) => {
  const path = new URL(`fixtures/ccd/${code}.cif`, import.meta.url);
  if (!existsSync(path)) throw new Error(`${code}: not in tools/fixtures/ccd (fetch it from files.rcsb.org/ligands/download/${code}.cif)`);
  return readFileSync(path, "utf8");
};

export function compareFeatures(dumpPath, jobPath, { mode, contactsPath } = {}) {
  const { mode: dumpMode, stages } = readBioemu2Dump(dumpPath);
  const job = bioemu2JobFromAf3Json(JSON.parse(readFileSync(jobPath, "utf8")));
  const contacts = contactsPath ? parseContacts(JSON.parse(readFileSync(contactsPath, "utf8"))) : null;
  const ours = featuriseBioemu2(job, { ccd, mode: mode ?? dumpMode, contacts, smilesTemplates: smilesTemplatesOf(stages) });
  const rows = [];
  let failures = 0;
  for (const [key, want] of Object.entries(stages)) {
    if (!key.startsWith("feats.")) continue;
    const name = key.slice("feats.".length);
    let got = ours.features[name];
    let expected = want;
    if (name === "msa") {
      // the dump's one-hot, as an index
      const index = new Int32Array(want.length / NUM_TOKENS);
      for (let i = 0; i < index.length; i += 1) {
        for (let k = 0; k < NUM_TOKENS; k += 1) if (want[i * NUM_TOKENS + k]) index[i] = k;
      }
      expected = index;
    }
    if (typeof expected === "number" || expected.length === undefined) expected = [Number(expected)];
    if (typeof got === "number") got = [got];
    if (got === undefined) { rows.push(`${name.padEnd(24)} MISSING here`); failures += 1; continue; }
    if (got.length !== expected.length) {
      rows.push(`${name.padEnd(24)} LENGTH ${got.length} against ${expected.length}`);
      failures += 1;
      continue;
    }
    const float = want.constructor === Float32Array && name !== "msa";
    let differ = 0;
    let worst = 0;
    let first = -1;
    for (let i = 0; i < got.length; i += 1) {
      const d = Math.abs(got[i] - expected[i]);
      if (float ? d > 1e-6 : d !== 0) { differ += 1; if (first < 0) first = i; }
      worst = Math.max(worst, d);
    }
    if (differ) failures += 1;
    rows.push(`${name.padEnd(24)} ${differ ? `${differ} of ${got.length} differ (worst ${worst.toExponential(2)},`
      + ` first at ${first}: ${got[first]} against ${expected[first]})` : `exact${float ? ` (worst ${worst.toExponential(2)})` : ""}`}`);
  }
  const keys = stages.atom_keys;
  const keysDiffer = keys.length !== ours.atomKeys.length
    || keys.some((k, i) => k.join("|") !== ours.atomKeys[i].join("|"));
  if (keysDiffer) failures += 1;
  rows.push(`${"atom keys".padEnd(24)} ${keysDiffer ? `DIFFER (${ours.atomKeys.length} against ${keys.length})` : `exact (${keys.length})`}`);
  return { rows, failures, tokens: ours.tokens, atoms: ours.atoms };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { rows, failures, tokens, atoms } = compareFeatures(option("dump"), option("job"), {
    mode: option("mode"), contactsPath: option("contacts"),
  });
  console.log(`${tokens} tokens, ${atoms} atoms`);
  for (const row of rows) console.log(`  ${row}`);
  if (failures) { console.error(`${failures} field(s) differ`); process.exit(1); }
  console.log("all fields agree");
}
