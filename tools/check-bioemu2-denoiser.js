/**
 * cpu/bioemu2/denoiser.js - BioEmu2's denoiser and its vector-Jacobian product - against af3-any-model's port.
 *
 *     node --max-old-space-size=8192 tools/check-bioemu2-denoiser.js --dump=oracle-dumps/bioemu2-6mrr.json \
 *         --weights=<a float32 export of the blob's diffuser/~/bioemu2 scope>/manifest.json [--only=denoise1,vjp1]
 *
 * The dump (tools/oracle/dump_bioemu2.py, which records the denoiser on a REAL sample re-noised) carries the
 * features, the static context, s_inputs and the trunk's last single, and per noise level the input, the output
 * and - for vjp<k> - a cotangent w with jax.grad of w . D(x, sigma) w.r.t. x and sigma. relRMS per arm; the
 * gradient arms are what every backend's edm_fkc reverse pass is held to through this.
 */
import { FileTensorStore } from "../shared/bundles/tensor-store.js";
import { denoiserWeights, bioemu2Denoise as denoise, bioemu2DenoiseVjp as vjp } from "../cpu/bioemu2/denoiser.js";
import { readBioemu2Dump, relRms } from "./bioemu2-dump.js";

const option = (name) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const { stages } = readBioemu2Dump(option("dump"));
const store = await FileTensorStore.open(option("weights"));
const w = await denoiserWeights(store);
const feat = (k) => stages[`feats.${k}`];
const f = {
  atom_mask: feat("atom_mask"), atom_pad_mask: feat("atom_pad_mask"), token_idx_at_atom: feat("token_idx_at_atom"),
  token_center_atom_index: feat("token_center_atom_index"), token_pad_mask: feat("token_pad_mask"),
  is_dynamics_conditioned: 0, dt: 0, coords_cond: null,
};
const T = feat("res_type").length;
const N = feat("atom_mask").length;
const ctx = { z: stages["ctx.z"], q0: stages["ctx.q0"], p: stages["ctx.p"] };
const sInputs = stages.s_inputs;
const passes = Object.keys(stages).filter((k) => /^pass\d+\.single$/.test(k)).sort();
const sTrunk = stages[passes[passes.length - 1]];
const only = option("only")?.split(",");
let failures = 0;
for (let k = 0; stages[`denoise${k}.x`]; k += 1) {
  if (only && !only.includes(`denoise${k}`)) continue;
  const t0 = Date.now();
  const out = denoise(w, f, ctx, sInputs, sTrunk, stages[`denoise${k}.x`], stages[`denoise${k}.sigma`], { N, T });
  const r = relRms(out, stages[`denoise${k}.out`]);
  failures += r > 1e-4 ? 1 : 0;
  console.log(`denoise at sigma ${stages[`denoise${k}.sigma`]}: relRMS ${r.toExponential(2)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
for (let k = 0; stages[`vjp${k}.x`]; k += 1) {
  if (only && !only.includes(`vjp${k}`)) continue;
  const t0 = Date.now();
  const sigma = stages[`vjp${k}.sigma`];
  const { gx, gs } = vjp(w, f, ctx, sInputs, sTrunk, stages[`vjp${k}.x`], sigma, stages[`vjp${k}.w`], { N, T });
  // 🔴 THE NETWORK'S PART, NOT THE TOTAL: D = c_skip x + c_out F(x), and at sigma 16 the network's part of d/dx
  // (|.| 21) nearly CANCELS c_skip w (21), leaving a total of 2.6 - so a 1.4e-4 difference in the part reads 1.2e-3 in
  // the total. That 1.4e-4 is the reference's own float32 backward (uniform over every window and position, projection
  // 1.000003); this tape against itself in float64 is 4e-6. d/dsigma is a sum that cancels the same way.
  const cSkip = 256 / (sigma * sigma + 256);
  const net = (g) => Float64Array.from(g, (v, i) => v - cSkip * stages[`vjp${k}.w`][i]);
  const r = relRms(net(gx), net(stages[`vjp${k}.gx`]));
  const want = stages[`vjp${k}.gs`];
  const rs = Math.abs(gs - want) / Math.max(Math.abs(want), 1e-12);
  failures += r > 1e-3 || rs > 5e-3 ? 1 : 0;
  console.log(`vjp at sigma ${sigma}: d/dx (the network's part) relRMS ${r.toExponential(2)}, total ${relRms(gx, stages[`vjp${k}.gx`]).toExponential(2)};`
    + ` d/dsigma ${gs.toExponential(6)} against ${want.toExponential(6)} (rel ${rs.toExponential(2)}, ${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
if (failures) { console.error(`${failures} arm(s) differ`); process.exit(1); }
console.log("the CPU denoiser and its VJP agree with the reference");
