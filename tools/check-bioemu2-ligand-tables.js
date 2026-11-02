/**
 * How far shared/bioemu2/ligand-tables.js (Tier 2 from the CCD) is from the vendor's RDKit tables. A REPORT.
 *
 *     node tools/check-bioemu2-ligand-tables.js --dump=oracle-dumps/bioemu2-topology-<t>.json --job=<job.json>
 *
 * It cannot be a gate: the vendor's bounds are RDKit's UFF distance geometry
 * and its tuples are ordered by CIP rank (see the header of ligand-tables.js).
 * What it reports is what the potentials would SEE: the pair set and its
 * flags, the bounds' differences, the tuple counts, and - the number that
 * matters - each Tier-2 potential's energy on the dump's frames under our
 * tables against under the vendor's, after the same host_tables step.
 */
import { readFileSync } from "node:fs";
import { featuriseBioemu2, bioemu2JobFromAf3Json } from "../shared/bioemu2/featurise.js";
import { ligandTables } from "../shared/bioemu2/ligand-tables.js";
import { energies, stepParameters, DEFAULT_SPECS, PARAMS, hostTables } from "../shared/bioemu2/potentials.js";

const option = (name) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const dump = JSON.parse(readFileSync(option("dump"), "utf8").replace(/\bNaN\b/g, "null"));
const ccd = (code) => readFileSync(new URL(`fixtures/ccd/${code}.cif`, import.meta.url), "utf8");
const ours = featuriseBioemu2(bioemu2JobFromAf3Json(JSON.parse(readFileSync(option("job"), "utf8"))), { ccd });
const mine = ligandTables(ours.topology, ccd);
const want = dump.tables;
const pairs = (idx) => (idx[0] ?? []).map((a, k) => `${Math.min(a, idx[1][k])},${Math.max(a, idx[1][k])}`);
const ourPairs = pairs(mine.rdkit_bounds_index.map((r) => Array.from(r)));
const theirPairs = pairs(want.rdkit_bounds_index);
const theirAt = new Map(theirPairs.map((p, k) => [p, k]));
let same = 0;
let flags = 0;
let dLo = 0;
let dHi = 0;
let finiteHi = 0;
ourPairs.forEach((p, k) => {
  const j = theirAt.get(p);
  if (j === undefined) return;
  same += 1;
  flags += (!!mine.rdkit_bounds_bond_mask[k] === !!want.rdkit_bounds_bond_mask[j]
            && !!mine.rdkit_bounds_angle_mask[k] === !!want.rdkit_bounds_angle_mask[j]) ? 1 : 0;
  dLo = Math.max(dLo, Math.abs(mine.rdkit_lower_bounds[k] - want.rdkit_lower_bounds[j]));
  if (want.rdkit_upper_bounds[j] < 100) { finiteHi += 1; dHi = Math.max(dHi, Math.abs(mine.rdkit_upper_bounds[k] - want.rdkit_upper_bounds[j])); }
});
console.log(`bounds: ${ourPairs.length} pairs here, ${theirPairs.length} in the vendor's, ${same} shared, flags agree on ${flags};`
  + ` worst |lower| ${dLo.toFixed(3)} A, worst |upper| ${dHi.toFixed(3)} A over ${finiteHi} finite`);
for (const k of ["chiral_atom_index", "stereo_bond_index", "planar_bond_index"]) {
  console.log(`${k}: ${mine[k][0]?.length ?? 0} tuples here, ${want[k][0]?.length ?? 0} in the vendor's`);
}
// each Tier-2 potential's energy on the frames, under our tables and under the vendor's
const typed = (v) => (Array.isArray(v) && Array.isArray(v[0]) ? v.map((r) => Float64Array.from(r))
  : Array.isArray(v) ? Float64Array.from(v.map((x) => (x === true ? 1 : x === false ? 0 : x))) : v);
const theirs = Object.fromEntries(Object.entries(dump.steer.host).map(([k, v]) => [k, typed(v)]));
const withOurs = hostTables({ ...Object.fromEntries(Object.entries(dump.tables).map(([k, v]) => [k, typed(v)])), ...mine,
                              bond_ref: dump.tables.bond_ref, angle_ref: dump.tables.angle_ref });
const p = stepParameters(0.5);
const tier2 = ["PoseBustersPotential", "ChiralAtomPotential", "StereoBondPotential", "PlanarBondPotential"];
for (const [f, frame] of dump.coords.entries()) {
  const c = Float64Array.from(frame.flat());
  const a = energies(c, withOurs, p);
  const b = energies(c, theirs, p);
  console.log(`frame ${f} (noise ${dump.noise[f]} A): ` + tier2.map((kind) => {
    const i = DEFAULT_SPECS.findIndex(([k]) => k === kind);
    return `${kind.replace("Potential", "")} ${a[i].toFixed(3)} vs ${b[i].toFixed(3)}`;
  }).join(", "));
}
void PARAMS;
