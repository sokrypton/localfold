// ESMFold2's EDM sampler, in the parts a port can actually be held to.
//
//     node tools/check-esmfold2-sampler.js
//
// 🔴 THE SAMPLER IS STOCHASTIC, SO "THE SAME STRUCTURE" IS NOT THE GATE. Every
// step centres, randomly rotates and translates the coordinates and then adds
// Gaussian noise - four draws from torch's global RNG per step - and nothing in
// JavaScript reproduces that stream. Checking coordinates against the model's
// would measure the RNG, not the port.
//
// What IS deterministic, and is checked here exactly:
//
//   * the noise schedule, including the truncation that turns fifteen
//     requested steps into eleven run ones;
//   * the churn factors and the noise level each step asks the denoiser about;
//   * the Kabsch alignment and the update algebra, on the LAST step - whose
//     output is returned unchanged, so the model's own answer is the oracle.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import { churnFactors, noiseLevels, noiseSchedule, samplerStep }
  from "../src/esmfold2/sampler-reference.js";

const ROOT = new URL("..", import.meta.url).pathname;
const dumpPath = process.argv[2] ?? join(ROOT, "oracle-dumps", "esmfold2-trunk-40-lm.json");
const dump = JSON.parse(readFileSync(dumpPath, "utf8"));
if (dump.sampler == null) throw new Error(`${dumpPath} has no sampler record`);
const s = dump.sampler;

const relative = (got, want) => {
  let error = 0, total = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    error += d * d;
    total += want[i] * want[i];
  }
  return Math.sqrt(error / total);
};

let failures = 0;
const report = (label, ok, detail) => {
  if (!ok) failures += 1;
  console.log(`  ${label.padEnd(38)} ${ok ? "ok" : "FAILED"}   ${detail}`);
};

console.log(`${dump.esmfold2}: ${s.steps} steps requested, `
  + `${s.perStep.length} run, sigma_data ${s.sigmaData}\n`);

// --- the schedule, untruncated, against the model's own.
{
  const got = noiseSchedule({ steps: s.steps, sMax: s.sMax, sMin: s.sMin, p: s.p,
                              sigmaData: s.sigmaData });
  const want = Float32Array.from(s.schedule);
  report("noise schedule", relative(got, want) < 1e-6,
         `relRMS ${relative(got, want).toExponential(3)} over ${want.length} entries`);
}

// --- the truncation, and the noise levels it implies.
{
  // 🔴 256 IS THE SHIPPED `max_inference_sigma` AND IT IS A DEFAULT ARGUMENT,
  // not a config field - `sample(..., max_inference_sigma: float | None = 256.0)`.
  // It is why fifteen steps run eleven times, and reading the step count off the
  // config gives a sampler that takes four extra steps at noise levels the model
  // never sees.
  const schedule = noiseSchedule({ steps: s.steps, sMax: s.sMax, sMin: s.sMin, p: s.p,
                                   sigmaData: s.sigmaData, maxSigma: 256 });
  const gammas = churnFactors(schedule, s.gammaMin, s.gamma0);
  const levels = noiseLevels(schedule, gammas);
  report("steps after truncation", levels.length === s.perStep.length,
         `${levels.length} against the model's ${s.perStep.length}`);
  let worst = 0;
  for (let i = 0; i < Math.min(levels.length, s.perStep.length); i += 1) {
    worst = Math.max(worst, Math.abs(levels[i] - s.perStep[i].tHat) / s.perStep[i].tHat);
  }
  // 🔴 THE BOUND IS float32's, TIMES EIGHT, AND THAT IS THE SCHEDULE'S OWN
  // EXPONENT. `sigma_data * (high + k/(n-1) * (low - high)) ** p` with p = 8
  // multiplies a relative error by eight, and the dump records float32 - so
  // 1.19e-7 in becomes about 1e-6 out. A 1e-6 bound failed a correct schedule
  // at 1.487e-6; the arithmetic says what to expect, rather than the number
  // being raised until it passes.
  report("t_hat per step", worst < 5e-6, `worst relative ${worst.toExponential(3)}`);
}

// --- the last step, whose output is the structure the model returned.
{
  const last = s.perStep[s.perStep.length - 1];
  const noisy = Float32Array.from(last.xNoisy);
  const denoised = Float32Array.from(last.xDenoised);
  const atoms = noisy.length / 3;
  const mask = Float32Array.from(
    dump.features.atom_attention_mask.values.slice(0, atoms));
  // The schedule's last entry is zero, which is what makes the final step land
  // on the answer rather than on another noisy iterate.
  const got = samplerStep(noisy, denoised, mask, atoms, last.tHat, 0, s.stepScale);
  const want = Float32Array.from(dump.coordinates);
  const score = relative(got, want);
  report("the last step, against the fold", score < 1e-5,
         `relRMS ${score.toExponential(3)}`);

  // 🔴 THE CONTROLS. Both of these run, converge to a structure, and are wrong:
  // aligning the answer onto the noisy copy instead of the other way round, and
  // taking the step without aligning at all.
  const unaligned = new Float32Array(noisy.length);
  const factor = s.stepScale * (0 - last.tHat) / last.tHat;
  for (let i = 0; i < unaligned.length; i += 1) {
    unaligned[i] = noisy[i] + factor * (noisy[i] - denoised[i]);
  }
  const without = relative(unaligned, want);
  // 🔴 AND IT DISCRIMINATES ONLY WEAKLY HERE, FOR A REASON WORTH KNOWING. By
  // the last step the structure has converged and the noisy copy is already
  // nearly in the answer's frame, so the alignment is close to the identity and
  // skipping it costs 1.4e-3 rather than the 1e-1 an early step would. This
  // arm says the check can SEE the alignment; it does not say the alignment is
  // unimportant, and an early-step version of this would say so far more
  // loudly.
  report("...without the alignment (control)", without > 1e-4,
         `relRMS ${without.toExponential(3)} - ${without > 1e-4 ? "discriminates" : "DOES NOT"}`);
}

console.log(failures === 0
  ? "\nthe sampler's schedule, churn and step agree with ESMFold2"
  : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
