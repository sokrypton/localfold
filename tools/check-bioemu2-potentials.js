/**
 * BioEmu2's steering potentials - every energy and its gradient - against af3-any-model's port.
 *
 *     node tools/check-bioemu2-potentials.js --dump=oracle-dumps/bioemu2-topology-6mrr.json [--job=<job.json>]
 *
 * The dump (tools/oracle/dump_bioemu2_topology.py) carries colab's host tables, the
 * step parameters at t = 0.9 / 0.5 / 0.1, and each default potential's energy
 * per particle and its jax.grad, on a sample and three noised copies. This
 * evaluates shared/bioemu2/potentials.js on THOSE tables, so a difference is in
 * the energy or its hand-written gradient and never in a table. With --job it
 * also rebuilds the host tables from our own steering tables and compares the
 * polymer half.
 */
import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { energies, stepParameters, guidanceInterval, hostTables, DEFAULT_SPECS, PARAMS } from "../shared/bioemu2/potentials.js";
import { featuriseBioemu2, bioemu2JobFromAf3Json } from "../shared/bioemu2/featurise.js";
import { steeringTables, parseStereoTables, STEREO_COMMIT } from "../shared/bioemu2/steering-tables.js";

const option = (name, fallback) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const dump = JSON.parse(readFileSync(option("dump"), "utf8").replace(/\bNaN\b/g, "null").replace(/-?\bInfinity\b/g, "null"));
const typed = (v) => (Array.isArray(v) && Array.isArray(v[0]) ? v.map((r) => Float64Array.from(r))
  : Array.isArray(v) ? Float64Array.from(v.map((x) => (x === true ? 1 : x === false ? 0 : x))) : v);
const tables = Object.fromEntries(Object.entries(dump.steer.host).map(([k, v]) => [k, typed(v)]));
const coords = dump.coords.map((frame) => Float64Array.from(frame.flat()));
let failures = 0;
for (const at of dump.steer.times) {
  const p = stepParameters(at.t);
  for (const k of ["guidance_weight", "resampling_weight", ...PARAMS]) {
    const ok = p[k].every((v, i) => (Number.isNaN(v) && at.params[k][i] === null) || Math.abs(v - at.params[k][i]) < 1e-7);
    if (!ok) { failures += 1; console.log(`t ${at.t} ${k}: ${p[k]} against ${at.params[k]}`); }
  }
  const interval = guidanceInterval(at.t);
  if (interval.join() !== at.interval.join()) { failures += 1; console.log(`t ${at.t} interval ${interval} vs ${at.interval}`); }
  const rows = [];
  DEFAULT_SPECS.forEach(([kind], i) => {
    let worstE = 0;
    let num = 0;
    let den = 0;
    coords.forEach((c, f) => {
      const grad = new Float64Array(c.length);
      const weights = DEFAULT_SPECS.map((_, j) => (j === i ? 1 : 0));
      const e = energies(c, tables, p, DEFAULT_SPECS, grad, weights)[i];
      const want = at.energies[i][f];
      worstE = Math.max(worstE, Math.abs(e - want) / Math.max(Math.abs(want), 1e-3));
      const g = at.grads[i][f].flat();
      for (let k = 0; k < g.length; k += 1) { num += (grad[k] - g[k]) ** 2; den += g[k] ** 2; }
    });
    const gRel = den > 0 ? Math.sqrt(num / den) : Math.sqrt(num);
    // (the reference sums the dense overlap in float32 and SUBTRACTS the bonded pairs, which cancels: intra-chain VDW
    // reads ~1e-4 against this exact sum while its gradient agrees to 1e-7)
    const bad = worstE > (kind === "IntraChainVDWOverlapPotential" ? 1e-3 : 1e-4) || gRel > 1e-3;
    failures += bad ? 1 : 0;
    rows.push(`  ${kind.padEnd(32)} energy rel ${worstE.toExponential(2)}   grad relRMS ${gRel.toExponential(2)}`
      + `${den === 0 ? " (zero)" : ""}${bad ? "   DIFFERS" : ""}`);
  });
  console.log(`t = ${at.t}`);
  for (const row of rows) console.log(row);
}

const jobPath = option("job");
if (jobPath) {
  const ccd = (code) => readFileSync(new URL(`fixtures/ccd/${code}.cif`, import.meta.url), "utf8");
  const ours = featuriseBioemu2(bioemu2JobFromAf3Json(JSON.parse(readFileSync(jobPath, "utf8"))), { ccd });
  const stereo = parseStereoTables(readFileSync(`${homedir()}/.cache/alphafold3/bioemu2/stereo_chemistry.${STEREO_COMMIT}.yaml`, "utf8"));
  const host = hostTables(steeringTables(ours.topology, stereo, { asymId: ours.features.asym_id, entityId: ours.features.entity_id }),
                          );
  for (const k of ["planar_improper_index", "symmetric_chain_pairs", ...Object.keys(host).filter((k) => /\.\d+$/.test(k))]) {
    const a = (Array.isArray(host[k]) ? host[k].flatMap((r) => Array.from(r)) : Array.from(host[k]));
    const b = (dump.steer.host[k] ?? []).flat();
    const same = a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= 1e-6);
    if (!same && !k.startsWith("planar")) failures += 1;
    console.log(`  host.${k.padEnd(28)} ${same ? "exact" : `${a.length} against ${b.length}`}`);
  }
}
if (failures) { console.error(`${failures} item(s) differ`); process.exit(1); }
console.log("every potential's energy and gradient agrees");
