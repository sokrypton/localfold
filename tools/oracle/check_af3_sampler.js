/**
 * Check the sampler's arithmetic against AF3, exactly.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 48 --diffusion 1 --float32 \
 *       --capture 'diffusion_head/__call__$' --capture-args 'diffusion_head/__call__$' \
 *       --out oracle-dumps/af3-oracle-denoiser-f32.json
 *     node tools/oracle/check_af3_sampler.js
 *
 * 🔴 THE TRAJECTORY IS NOT REPRODUCIBLE AND THE ARITHMETIC IS. Every step draws
 * from JAX's threefry PRNG for the augmentation and the injected noise, which
 * this repository does not implement, so our sample is a different draw from
 * the same distribution and comparing coordinates directly would mean nothing.
 *
 * What a ONE-STEP run gives instead is an exact test: with `steps = 1` the
 * sampler's single update produces the final answer, so
 *
 *     samplerStep(positions_noisy, denoised, t_hat, level) == the dumped sample
 *
 * holds elementwise. AF3 supplies the first three - two as the diffusion head's
 * captured arguments, one as its output - and `level` comes from our own
 * schedule, so a wrong schedule endpoint or a missing step_scale shows up here
 * rather than as a slightly worse structure.
 */
import { noiseLevels, samplerStep } from "../../src/af3/diffusion-sampler-reference.js";
import { captures, loadDump, report } from "./af3-bundle.js";

const HEAD = "diffuser/~/diffusion_head";

async function main() {
  const dump = await loadDump("oracle-dumps/af3-oracle-denoiser-f32.json");
  const at = captures(dump, "dump_af3_trunk.py --diffusion 1 --capture-args"
    + " 'diffusion_head/__call__$'");

  const levels = noiseLevels(1);
  const tHat = at(`${HEAD}/__call__<1`)[0];
  const positionsNoisy = at(`${HEAD}/__call__<0`);
  const denoised = at(`${HEAD}/__call__`);

  console.log(`${dump.model}, one step: schedule ${levels[0].toFixed(4)}`
    + ` -> ${levels[1].toFixed(6)}, AF3 denoised at ${tHat}`);
  // 🔴 THE FIRST LEVEL IS ALSO t_hat HERE, and that is a real check rather than
  // a tautology: gamma is 0.8 only where the level is above gamma_min, and the
  // level being stepped TO (0.0064) is below it - so the churn is off, t_hat
  // equals the previous level, and our schedule's start must equal the number
  // AF3 passed its head.
  if (Math.abs(levels[0] - tHat) > 1e-3) {
    throw new Error(`our schedule starts at ${levels[0]} but AF3 denoised at`
      + ` ${tHat}; the two must agree before the step means anything`);
  }

  const sample = at("diffusion_samples/atom_positions");
  report("sample", sample,
         samplerStep(positionsNoisy, denoised, tHat, levels[1]));
}

await main();
