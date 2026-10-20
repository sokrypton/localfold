/**
 * Is cuda/bioemu2/src/steer.h - BioEmu2's steering natively - the JavaScript's, and af3-any-model's?
 *
 *     node tools/check-native-bioemu2-steering.mjs [--probe=<steer-probe binary>]
 *
 * Builds cuda/bioemu2/steer-probe.cpp (the featuriser's flags, -ffp-contract=off) unless handed one, and holds:
 *
 *   - every default potential's ENERGY and GRADIENT (one-hot weights) on every topology dump's frames, at the dump's
 *     recorded times, to shared/bioemu2/potentials.js - counted bit for bit, bounded at 1e-12 relative. What can
 *     separate the two is Math.exp (V8's fdlibm against glibc) inside one schedule, rounded to float32 after;
 *   - the steered samplers (edm_gd, edm_fk) on the steering dumps' toy denoiser and injected draws to sampler.js's
 *     steeredSample - the resampling INDICES step for step, the particles at 1e-12 - and to af3-any-model's own
 *     recorded run, at the floor that dump measures (its initial draw nudged by 1e-6), as check-bioemu2-steering.js;
 *   - a `--potentials` file read natively (cuda/bioemu2/src/potentials_yaml.h) as potentials-yaml.js and potentialSpecs
 *     read it: the same potentials, the same parameters at every t to the bit, the same refusal.
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { energies, stepParameters, guidanceInterval, potentialSpecs, DEFAULT_SPECS, PARAMS } from "../shared/bioemu2/potentials.js";
import { parseSimpleYaml } from "../shared/bioemu2/potentials-yaml.js";
import { steeredSample, fkcSample } from "../shared/bioemu2/sampler.js";
import { weightedRigidAlign } from "../cpu/af3/diffusion/rigid-align.js";

const option = (name) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const repo = new URL("..", import.meta.url).pathname;
const dumps = join(repo, "oracle-dumps");
let probe = option("probe");
if (!probe) {
  probe = join(mkdtempSync(join(tmpdir(), "steer-probe-")), "steer-probe");
  execFileSync("g++", ["-std=c++17", "-O2", "-ffp-contract=off", "-pthread", join(repo, "cuda/bioemu2/steer-probe.cpp"), "-o", probe],
               { stdio: "inherit" });
}
const read = (path) => JSON.parse(readFileSync(path, "utf8").replace(/\bNaN\b/g, "null").replace(/-?\bInfinity\b/g, "null"));
const typed = (v) => (Array.isArray(v) && Array.isArray(v[0]) ? v.map((r) => Float64Array.from(r))
  : Array.isArray(v) ? Float64Array.from(v.map((x) => (x === true ? 1 : x === false ? 0 : x === null ? NaN : x))) : v);
const native = (...args) => JSON.parse(execFileSync(probe, args, { maxBuffer: 1 << 30 }).toString());
const relative = (a, b) => (a === b ? 0 : Math.abs(a - b) / Math.max(Math.abs(b), 1e-300));
let failures = 0;

// ---- the potentials
for (const name of ["6mrr", "6mrr-gol-sep", "rna-dna", "smiles", "6mrr-nag-ret"]) {
  const path = join(dumps, `bioemu2-topology-${name}.json`);
  if (!existsSync(path)) { console.log(`  potentials ${name}: SKIP (no ${path})`); continue; }
  const dump = read(path);
  if (!dump.steer?.times) { console.log(`  potentials ${name}: SKIP (the dump records no potentials)`); continue; }
  const tables = Object.fromEntries(Object.entries(dump.steer.host).map(([k, v]) => [k, typed(v)]));
  const coords = dump.coords.map((frame) => Float64Array.from(frame.flat()));
  const theirs = native("potentials", path);
  let values = 0;
  let exact = 0;
  let worst = 0;
  dump.steer.times.forEach((at, ti) => {
    const p = stepParameters(at.t);
    const n = theirs.times[ti];
    if (guidanceInterval(at.t).join() !== n.interval.join()) { failures += 1; console.log(`  ${name} t ${at.t}: interval differs`); }
    DEFAULT_SPECS.forEach((_, i) => {
      coords.forEach((c, f) => {
        const grad = new Float64Array(c.length);
        const weights = DEFAULT_SPECS.map((__, j) => (j === i ? 1 : 0));
        const e = energies(c, tables, p, DEFAULT_SPECS, grad, weights)[i];
        const ne = n.energies[i][f];
        const ng = n.grads[i * coords.length + f];
        values += 1 + grad.length;
        exact += (e === ne ? 1 : 0) + grad.reduce((s, g, k) => s + (g === ng[k] ? 1 : 0), 0);
        worst = Math.max(worst, relative(ne, e));
        // (a gradient element relative to the gradient's own scale: a component that cancels to ~0 is noise)
        const scale = Math.max(...grad.map(Math.abs), 1e-300);
        grad.forEach((g, k) => { worst = Math.max(worst, Math.abs(ng[k] - g) / scale); });
      });
    });
  });
  const bad = worst > 1e-12;
  failures += bad ? 1 : 0;
  console.log(`  potentials ${name.padEnd(13)} ${values} energies and gradient elements, ${exact} bit for bit,`
    + ` worst ${worst.toExponential(2)}${bad ? "   DIFFERS" : ""}`);
}

// ---- --potentials files: the vendor's presets, schedules of all three kinds, and what must be refused
const yamls = join(mkdtempSync(join(tmpdir(), "steer-yaml-")), "x");
const YAMLS = {
  "bonds-and-clashes": readFileSync(join(repo, "tools/fixtures/bioemu2/potentials/bonds-and-clashes.yaml"), "utf8"),
  chirality: readFileSync(join(repo, "tools/fixtures/bioemu2/potentials/chirality.yaml"), "utf8"),
  schedules: "potentials:\n  - _target_: bioemu2.denoising.steering.potentials.SymmetricChainCOMPotential  # a comment\n    parameters:\n"
    + "      guidance_weight: 0.5\n      buffer:\n        _target_: bioemu2.denoising.steering.potentials.ExponentialInterpolation\n"
    + "        start: 1.0\n        end: 5.0\n        alpha: -2.0\n  - _target_: bioemu2.denoising.steering.potentials.PolymerLocalClashPotential\n"
    + "    parameters:\n      minimum_distance: 2.0\n      guidance_weight:\n"
    + "        _target_: bioemu2.denoising.steering.potentials.LinearInterpolation\n        start_t: 0.9\n        end_t: 0.1\n"
    + "        start_val: 0.3\n        end_val: 0.05\n",
  piecewise: "potentials:\n  - _target_: a.PolymerLocalClashPotential\n    parameters:\n      minimum_distance: 2.5\n      guidance_weight:\n"
    + "        _target_: a.PiecewiseStepFunction\n        thresholds: [0.4, 0.7]\n        values: [0.2, 0.1, 0.0]\n",
  "unknown potential": "potentials:\n  - _target_: x.MadeUpPotential\n",
  "an anchor": "potentials:\n  - _target_: x.BondLengthPotential\n    sigma_tolerance: &a 4\n",
  "no tolerance": "potentials:\n  - _target_: x.BondLengthPotential\n    parameters: {guidance_weight: 0.1}\n",
  "a tab": "potentials:\n\t- _target_: x.BondLengthPotential\n",
  empty: "potentials: []\n",
};
for (const [name, text] of Object.entries(YAMLS)) {
  const path = `${yamls}-${name.replaceAll(" ", "-")}.yaml`;
  writeFileSync(path, text);
  let mine;
  try {
    const specs = potentialSpecs(parseSimpleYaml(text));
    mine = { kinds: specs.map((s) => s[0]), tolerance: specs.map((s) => s[2].sigma_tolerance ?? null),
             at: Array.from({ length: 21 }, (_, k) => {
               const p = stepParameters(k / 20, specs);
               return { ...Object.fromEntries(["guidance_weight", "resampling_weight", ...PARAMS].map((key) =>
                 [key, p[key].map((v) => (Number.isNaN(v) ? null : v))])), interval: guidanceInterval(k / 20, specs) };
             }) };
  } catch (error) {
    mine = { error: error.message };
  }
  const theirs = native("specs", path);
  const ok = "error" in mine || "error" in theirs ? mine.error === theirs.error : JSON.stringify(mine) === JSON.stringify(theirs);
  failures += ok ? 0 : 1;
  console.log(`  --potentials ${name.padEnd(18)} ${ok ? ("error" in mine ? `refused alike: ${mine.error}`
    : `${mine.kinds.length} potential(s), every parameter at 21 times the same`)
    : `DIFFERS: ${JSON.stringify(mine).slice(0, 300)} against ${JSON.stringify(theirs).slice(0, 300)}`}`);
}

// ---- the steered samplers
for (const name of ["6mrr-nag-ret-4", "6mrr-nag-ret"]) {
  const path = join(dumps, `bioemu2-steering-${name}.json`);
  const topologyPath = join(dumps, `bioemu2-topology-${name.replace(/-4$/, "")}.json`);
  if (!existsSync(path) || !existsSync(topologyPath)) { console.log(`  sampler ${name}: SKIP`); continue; }
  const dump = read(path);
  const topology = read(topologyPath);
  const tables = Object.fromEntries(Object.entries(topology.steer.host).map(([k, v]) => [k, typed(v)]));
  const mask = tables.atom_pad_mask;
  const ref = Float64Array.from(dump.ref.flat());
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
  const started = Date.now();
  const theirs = native("sample", path, topologyPath);
  const nativeSeconds = (Date.now() - started) / 1000;
  const rel = (a, b) => {
    let num = 0;
    let den = 0;
    a.forEach((p, i) => { for (let k = 0; k < p.length; k += 1) { num += (p[k] - b[i][k]) ** 2; den += b[i][k] ** 2; } });
    return Math.sqrt(num / den);
  };
  // edm_fkc: the toy denoiser's vector-Jacobian product, as check-bioemu2-steering.js writes it
  {
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
    const t0 = Date.now();
    const { x, nonFiniteStep } = await fkcSample({ denoiseWithGrad, tables, mask, particles: dump.particles, draws, numSteps: dump.steps });
    const jsSeconds = (Date.now() - t0) / 1000;
    const n = theirs.edm_fkc;
    const toJs = rel(n.x, x);
    const want = dump.edm_fkc;
    const toRef = rel(n.x, want.x.map((p) => p.flat()));
    const floor = want.floor ?? 0;
    const bad = toJs > 1e-9 || n.bad !== nonFiniteStep || toRef > Math.max(1e-4, 10 * floor) || n.bad !== want.bad;
    failures += bad ? 1 : 0;
    console.log(`  edm_fkc ${name.padEnd(14)} against sampler.js ${toJs.toExponential(2)} (non-finite step ${n.bad} / ${nonFiniteStep});`
      + ` against af3-any-model ${toRef.toExponential(2)} (its floor ${floor.toExponential(2)}, its non-finite step ${want.bad})`
      + `  [JS ${jsSeconds.toFixed(1)} s]${bad ? "   DIFFERS" : ""}`);
  }
  for (const [sampler, interval] of [["edm_gd", null], ["edm_fk", 3]]) {
    const trace = [];
    const t0 = Date.now();
    const x = await steeredSample({ denoise: toy, tables, mask, particles: dump.particles, draws, align, numSteps: dump.steps,
                                    stepScale: 1.5, resamplingInterval: interval, trace, numGdSteps: dump.gd_steps ?? 20 });
    const jsSeconds = (Date.now() - t0) / 1000;
    const n = theirs[sampler];
    const toJs = rel(n.x, x);
    const sameIdx = trace.every((row, k) => row.idx.join() === n.idx[k].join());
    const want = dump[sampler];
    const toRef = rel(n.x, want.x.map((p) => p.flat()));
    const refIdx = want.trace.filter((row, k) => row[3].join() !== n.idx[k].join()).length;
    const floor = want.floor ?? 0;
    const bad = toJs > 1e-12 || !sameIdx || toRef > Math.max(1e-4, 10 * floor) || (refIdx > 0 && floor < 1e-2);
    failures += bad ? 1 : 0;
    console.log(`  ${sampler} ${name.padEnd(15)} against sampler.js ${toJs.toExponential(2)} (indices ${sameIdx ? "the same" : "DIFFER"});`
      + ` against af3-any-model ${toRef.toExponential(2)} (its floor ${floor.toExponential(2)}, ${refIdx} resampling step(s) differ)`
      + `  [${dump.steps} steps: native ${nativeSeconds.toFixed(1)} s both samplers, JS ${jsSeconds.toFixed(1)} s]${bad ? "   DIFFERS" : ""}`);
  }
}
if (failures) { console.error(`${failures} item(s) differ`); process.exit(1); }
console.log("the native steering is the JavaScript's");
