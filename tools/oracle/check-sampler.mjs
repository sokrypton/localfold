/**
 * Replay an oracle trajectory through the JS sampler (`shared/rfd4p/flow.js`), using the
 * recorded network outputs and recorded noise: schedule, initial noise and every step's update
 * must reproduce the next recorded x_t. Also checks the recycle depths.
 *
 *   node tools/oracle/check-sampler.mjs [case=uncond_short]
 */

import {
  DEFAULT_SAMPLING,
  flowEmitMasks,
  recycleDepths,
  sampleInitialNoise,
  sampleSchedule,
  simulationStep,
} from '../../shared/rfd4p/flow.js';
import { compare, loadOracle } from './oracle-io.mjs';

const caseName = process.argv[2] ?? 'uncond_short';
const { t, meta } = loadOracle(new URL('../../oracle-dumps/', import.meta.url).pathname, caseName);
const n = t['batch.padding_mask'].shape[1];
let failed = false;
const ok = (label, cond, extra = '') => {
  if (!cond) failed = true;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${label} ${extra}`);
};

/** Replays recorded randn draws in order (skipping randperm, which features consume). */
function recordedRng() {
  const draws = meta.rng.filter((r) => r.fn === 'randn');
  let k = 0;
  return {
    randn(count) {
      const r = draws[k++];
      const d = r && t[`rng.${r.i}`].data;
      if (!d || d.length !== count) throw new Error(`randn replay mismatch at ${k - 1}: want ${count}, have ${d?.length}`);
      return d;
    },
    get used() {
      return k;
    },
    total: draws.length,
  };
}

const sched = sampleSchedule(DEFAULT_SAMPLING);
for (const dm of ['bb_ca', 'local_latents']) {
  const ts = compare(sched[dm].ts, t[`schedule.ts.${dm}`].data);
  const gt = compare(sched[dm].gt, t[`schedule.gt.${dm}`].data);
  ok(`schedule ${dm}`, ts.maxAbs < 1e-6 && gt.relRms < 1e-5, `ts maxAbs=${ts.maxAbs.toExponential(2)} gt relRms=${gt.relRms.toExponential(2)}`);
}
const depths = recycleDepths('medium', DEFAULT_SAMPLING.nsteps);
ok('recycle depths', JSON.stringify(depths) === JSON.stringify(meta.recycle_depths));

const B = (k) => t[`batch.${k}`].data;
const emit = flowEmitMasks(
  { padding_mask: B('padding_mask'), is_in_context_conditioning: B('is_in_context_conditioning'), supports_local_latent: B('supports_local_latent') },
  n
);
const rng = recordedRng();
const x = sampleInitialNoise({ n, emit, coordCondMask: B('coord_cond_mask'), moleculeIid: B('molecule_iid'), rng });
for (const dm of ['bb_ca', 'local_latents']) {
  const r = compare(x[dm], t[`step.0.x_t.${dm}`].data);
  ok(`initial noise ${dm}`, r.maxAbs < 1e-6, `maxAbs=${r.maxAbs.toExponential(2)}`);
}

// Each step from the recorded x_t, so errors do not compound.
let worst = { bb_ca: 0, local_latents: 0 };
const nsteps = meta.steps.length;
for (let s = 0; s < nsteps; s += 1) {
  for (const dm of ['bb_ca', 'local_latents']) {
    const xs = Float32Array.from(t[`step.${s}.x_t.${dm}`].data);
    const ts = sched[dm].ts;
    simulationStep({
      x: xs, v: t[`step.${s}.out.${dm}.v`].data, t: Math.fround(ts[s]), dt: ts[s + 1] - ts[s], gt: sched[dm].gt[s],
      emit: emit[dm], dim: dm === 'bb_ca' ? 3 : 32, step: sched[dm].step, rng,
    });
    const want = s + 1 < nsteps ? t[`step.${s + 1}.x_t.${dm}`].data : t[`final.x.${dm}`].data;
    const r = compare(xs, want);
    worst[dm] = Math.max(worst[dm], r.relRms);
  }
}
for (const dm of ['bb_ca', 'local_latents']) ok(`steps ${dm}`, worst[dm] < 1e-5, `worst relRms=${worst[dm].toExponential(2)}`);
ok('all randn draws consumed', rng.used === rng.total, `${rng.used}/${rng.total}`);
console.log(failed ? 'FAILED' : 'sampler matches');
process.exitCode = failed ? 1 : 0;
