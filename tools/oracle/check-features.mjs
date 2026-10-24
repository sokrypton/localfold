/**
 * Check JS feature construction (`shared/rfd4p/features.js`) against the oracle's recorded
 * pre-`linear_out` feature matrices, given the oracle's own batch and x_t for call 0.
 *
 *   node tools/oracle/check-features.mjs [case=uncond_short]
 */

import { readFileSync } from 'node:fs';
import { pairCondFeatures, pairFeatures, trunkCondFeatures, trunkSeqFeatures } from '../../shared/rfd4p/features.js';
import { compare, loadOracle, readSafetensors, releasedWeightsPath } from './oracle-io.mjs';

const caseName = process.argv[2] ?? 'uncond_short';
const paths = releasedWeightsPath();
const cfg = JSON.parse(readFileSync(paths.config, 'utf8')).config.nn;
const weights = readSafetensors(paths.weights);
const { t, meta } = loadOracle(new URL('../../oracle-dumps/', import.meta.url).pathname, caseName);

/** Upstream-layout batch from the oracle's `batch.*` tensors. */
export function oracleBatch(tensors) {
  const batch = {};
  for (const [k, v] of Object.entries(tensors)) if (k.startsWith('batch.')) batch[k.slice(6)] = v;
  return batch;
}

/** Recorded randperm draws for `call`, replayed in order. */
export function recordedRandperm(tensors, rngLog, call) {
  const draws = rngLog.filter((r) => r.fn === 'randperm' && r.call === call).map((r) => tensors[`rng.${r.i}`].data);
  let k = 0;
  return (len) => {
    const d = draws[k++];
    if (!d || d.length !== len) throw new Error(`randperm replay mismatch: want ${len}, have ${d?.length}`);
    return d;
  };
}

const batch = oracleBatch(t);
const n = batch.padding_mask.shape[1];
const xt = { bb_ca: t['step.0.x_t.bb_ca'].data, local_latents: t['step.0.x_t.local_latents'].data };
const tt = { bb_ca: t['step.0.t.bb_ca'].data[0], local_latents: t['step.0.t.local_latents'].data[0] };

let failed = false;
function report(label, got, wantKey, width) {
  const want = t[wantKey].data;
  const r = compare(got, want);
  const ok = r.relRms < 1e-5 && r.maxAbs < 1e-4;
  if (!ok) failed = true;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label.padEnd(10)} relRms=${r.relRms.toExponential(2)} maxAbs=${r.maxAbs.toExponential(2)}`);
  if (!ok && width) {
    // Attribute the error to feature columns.
    const bad = new Map();
    for (let i = 0; i < want.length; i += 1) {
      if (Math.abs(got[i] - want[i]) > 1e-4) bad.set(i % width, (bad.get(i % width) ?? 0) + 1);
    }
    console.log('   bad columns:', [...bad.entries()].slice(0, 40).map(([c, k]) => `${c}x${k}`).join(' '));
  }
}

const seq = trunkSeqFeatures(batch, {
  n, xt, feats: cfg.feats_seq, weights, randperm: recordedRandperm(t, meta.rng, 0),
  noiseAlignChirality: cfg.noise_align_chirality,
});
report('seq', seq.data, 'call0.init_repr_factory.features', seq.dim);

const cond = trunkCondFeatures(batch, { n, t: tt, recycleRemaining: null, feats: cfg.feats_cond_seq, weights });
report('cond', cond.data, 'call0.cond_factory.features', cond.dim);

const pair = pairFeatures(batch, {
  n, caNm: xt.bb_ca, feats: cfg.feats_pair_repr,
  dist: { dim: cfg.xt_pair_dist_dim, min: cfg.xt_pair_dist_min, max: cfg.xt_pair_dist_max }, seqSepDim: cfg.seq_sep_dim,
});
report('pair', pair.data, 'call0.pair_repr_builder.init_repr_factory.features', pair.dim);

report('pair_cond', pairCondFeatures(tt, cfg.feats_pair_cond), 'call0.pair_repr_builder.cond_factory.features');

console.log(failed ? 'FAILED' : 'features match');
process.exitCode = failed ? 1 : 0;
