/**
 * BioEmu2's steered samplers (edm_gd, edm_fk, edm_fkc) against af3-any-model's port, on a toy denoiser.
 *
 *     node tools/check-bioemu2-steering.js --dump=oracle-dumps/bioemu2-steering-<t>.json
 *         --topology=oracle-dumps/bioemu2-topology-<t>.json
 *
 * tools/oracle/dump_bioemu2_steering.py ran colab's samplers with injected draws
 * and D(x, sigma) = c_skip x + (1 - c_skip) ref; this runs shared/bioemu2/sampler.js
 * with the same draws and denoiser on the same host tables. The resampling
 * INDICES must agree step for step (a flipped choice is a different lineage),
 * and the final particles to float32 accumulation over the steps.
 */
import { readFileSync } from "node:fs";
import { steeredSample, fkcSample } from "../shared/bioemu2/sampler.js";
import { energies, stepParameters, PARAMS } from "../shared/bioemu2/potentials.js";
import { weightedRigidAlign } from "../cpu/af3/diffusion/rigid-align.js";

const option = (name) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const read = (path) => JSON.parse(readFileSync(path, "utf8").replace(/\bNaN\b/g, "null"));
const dump = read(option("dump"));
const topology = read(option("topology"));
const typed = (v) => (Array.isArray(v) && Array.isArray(v[0]) ? v.map((r) => Float64Array.from(r))
  : Array.isArray(v) ? Float64Array.from(v.map((x) => (x === true ? 1 : x === false ? 0 : x === null ? NaN : x))) : v);
const tables = Object.fromEntries(Object.entries(topology.steer.host).map(([k, v]) => [k, typed(v)]));
const mask = tables.atom_pad_mask;
const ref = Float64Array.from(dump.ref.flat());
const P = dump.particles;
const SD = 16;
const cSkip = (s) => (SD * SD) / (s * s + SD * SD);
const toy = (x, sigma) => x.map((xp) => Float64Array.from(xp, (v, i) => cSkip(sigma) * v + (1 - cSkip(sigma)) * ref[i]));
const draws = {
  x0: () => dump.draws.x0.map((p) => Float64Array.from(p.flat())),
  step: (k) => ({ rot: dump.draws.rot[k].map((r) => Float64Array.from(r.flat())), trans: dump.draws.trans[k],
                  eps: dump.draws.eps[k].map((e) => Float64Array.from(e.flat())) }),
  uniform: (k) => dump.draws.uniform[k],
};
const align = (x, target, m) => Float64Array.from(weightedRigidAlign(x, target, Float64Array.from(m), m.length));
const rel = (a, b) => {
  let num = 0;
  let den = 0;
  a.forEach((p, i) => { for (let k = 0; k < p.length; k += 1) { num += (p[k] - b[i][k]) ** 2; den += b[i][k] ** 2; } });
  return Math.sqrt(num / den);
};
let failures = 0;
for (const [name, interval] of [["edm_gd", null], ["edm_fk", 3]]) {
  const trace = [];
  const x = await steeredSample({ denoise: toy, tables, mask, particles: P, draws, align, numSteps: dump.steps,
                                  stepScale: 1.5, resamplingInterval: interval, trace, numGdSteps: dump.gd_steps ?? 20 });
  const want = dump[name];
  const r = rel(x, want.x.map((p) => p.flat()));
  let indexDiffer = 0;
  want.trace.forEach((row, k) => { if (row[3].join() !== trace[k].idx.join()) indexDiffer += 1; });
  // 🔴 AGAINST THE REFERENCE'S OWN FLOOR, which the dump measures (its initial draw nudged by 1e-6): over many steps
  // edm_gd and edm_fk are CHAOTIC - the flat-bottom potentials' gradients are discontinuous, so float32 alone flips a
  // term, and a flipped resampling choice is another lineage - so a fixed bound would fail a correct port. At 2 and 4
  // steps the floor is ~1e-6 and this agrees there to 5e-7..1e-6 with every index the same.
  const floor = want.floor ?? 0;
  const bad = r > Math.max(1e-4, 10 * floor) || (indexDiffer > 0 && floor < 1e-2);
  failures += bad ? 1 : 0;
  console.log(`${name}: final particles relRMS ${r.toExponential(2)} (the reference's own floor ${floor.toExponential(2)}),`
    + ` resampling indices differ at ${indexDiffer} of ${want.trace.length} steps${bad ? "   DIFFERS" : ""}`);
}
// edm_fkc: the toy denoiser's vector-Jacobian product, written out
const denoiseWithGrad = async (x, sigma, comp, tHat) => {
  const den = toy(x, sigma);
  const prm = stepParameters(tHat);
  const buffers = Object.fromEntries(PARAMS.map((b) => [b, prm[b]]));
  const c = cSkip(sigma);
  const dc = (-2 * sigma * SD * SD) / (sigma * sigma + SD * SD) ** 2;
  let u = 0;
  const gx = [];
  const gs = [];
  den.forEach((d, p) => {
    const gD = new Float64Array(d.length);
    u += energies(d, tables, buffers, undefined, gD, prm.guidance_weight.map(() => 1)).reduce((s, e) => s + e, 0);
    gx.push(Float64Array.from(gD, (g) => g * c));
    let s = 0;
    for (let i = 0; i < d.length; i += 1) s += gD[i] * dc * (x[p][i] - ref[i]);
    gs.push(s);
  });
  return { den, gx, gs, u };
};
const fkcTrace = { x: [], logw: [] };
const { x, nonFiniteStep } = await fkcSample({ denoiseWithGrad, tables, mask, particles: P, draws, numSteps: dump.steps,
                                              trace: fkcTrace });
if (dump.edm_fkc.trace) {
  // the first step whose state or log-weights part from the reference's
  for (let k = 0; k < fkcTrace.x.length; k += 1) {
    const rx = rel(fkcTrace.x[k], dump.edm_fkc.trace.x[k].map((p) => p.flat()));
    const want = dump.edm_fkc.trace.logw[k];
    const dw = Math.max(...fkcTrace.logw[k].map((v, i) => Math.abs(v - want[i])));
    if (rx > 1e-5 || dw > 1e-4) {
      console.log(`edm_fkc: first parts at step ${k}: state relRMS ${rx.toExponential(2)}, log-weight max|d| ${dw.toExponential(2)}`
        + ` (ours ${fkcTrace.logw[k].map((v) => v.toFixed(5))} against ${want.map((v) => v.toFixed(5))})`);
      break;
    }
  }
}
const r = rel(x, dump.edm_fkc.x.map((p) => p.flat()));
const bad = r > Math.max(1e-4, 10 * (dump.edm_fkc.floor ?? 0)) || nonFiniteStep !== dump.edm_fkc.bad;
failures += bad ? 1 : 0;
console.log(`edm_fkc: final particles relRMS ${r.toExponential(2)}, non-finite step ${nonFiniteStep} against ${dump.edm_fkc.bad}`
  + `${bad ? "   DIFFERS" : ""}`);
if (failures) { console.error(`${failures} sampler(s) differ`); process.exit(1); }
console.log("all three steered samplers agree");
