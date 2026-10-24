/**
 * Check the CPU autoencoder encoder (`cpu/rfd4p/encoder.js`) and partial-diffusion /
 * inverse-folding helpers (`shared/rfd4p/partial.js`) against `oracle-dumps/encoder_oracle.*`.
 *
 *   node tools/oracle/check-encoder.mjs
 */

import { readFileSync } from 'node:fs';
import { encodeLatentsCpu, encoderSeqFeatures } from '../../cpu/rfd4p/encoder.js';
import { DEFAULT_SAMPLING, flowEmitMasks } from '../../shared/rfd4p/flow.js';
import {
  buildPartialSchedule,
  corruptBatch,
  resolveInverseFoldingMethod,
  resolvePartialT,
  sampleNoisedTargetBackbone,
} from '../../shared/rfd4p/partial.js';
import { compare, loadOracle, readSafetensors, releasedWeightsPath } from './oracle-io.mjs';

const paths = releasedWeightsPath();
const cfg = JSON.parse(readFileSync(paths.config, 'utf8')).config.autoencoder_config.nn_ae.encoder;
const weights = readSafetensors(paths.weights);
const dumpDir = new URL('../../oracle-dumps/', import.meta.url).pathname;
const { t, meta } = loadOracle(dumpDir, 'encoder_oracle');

let failed = false;
const ok = (label, cond, extra = '') => {
  if (!cond) failed = true;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${label} ${extra}`);
};

for (const caseName of meta.cases) {
  const batch = {};
  const prefix = `${caseName}.batch.`;
  for (const [k, v] of Object.entries(t)) {
    if (k.startsWith(prefix)) batch[k.slice(prefix.length)] = v;
  }
  const n = batch.padding_mask.shape[1];

  const { data: feats, dim: fdim } = encoderSeqFeatures(batch, { n, weights, cfg });
  const rf = compare(feats, t[`${caseName}.enc.features`].data);
  ok(
    `[${caseName}] enc.features (dim=${fdim})`,
    fdim === 1587 && rf.maxAbs < 1e-6 && rf.nonFinite === 0,
    `relRms=${rf.relRms.toExponential(2)} maxAbs=${rf.maxAbs.toExponential(2)}`,
  );

  const eps = t[`${caseName}.enc.eps`].data;
  const encRng = { randn: () => eps };
  const out = encodeLatentsCpu(batch, { n, weights, cfg, rng: encRng });

  const wantEmit = t[`${caseName}.enc.emit_mask`].data;
  ok(
    `[${caseName}] enc.emit_mask`,
    out.emitMask.every((v, i) => v === wantEmit[i]),
  );

  for (const [field, key] of [
    ['mean', 'mean'],
    ['logScale', 'log_scale'],
    ['zLatent', 'z_latent'],
  ]) {
    const r = compare(out[field], t[`${caseName}.enc.${key}`].data);
    const cond = key === 'z_latent' ? r.relRms < 1e-5 : r.maxAbs < 1e-4 && r.relRms < 1e-5;
    ok(
      `[${caseName}] enc.${key}`,
      cond && r.nonFinite === 0,
      `relRms=${r.relRms.toExponential(2)} maxAbs=${r.maxAbs.toExponential(2)}`,
    );
  }

  // Corrupt batch check
  const emit = flowEmitMasks(
    {
      padding_mask: batch.padding_mask.data,
      is_in_context_conditioning: batch.is_in_context_conditioning.data,
      supports_local_latent: batch.supports_local_latent.data,
    },
    n,
  );
  const nCorruptDraws = meta[`${caseName}_corrupt_n_draws`];
  let cIdx = 0;
  const corruptRng = {
    randn(count) {
      const d = t[`${caseName}.corrupt.rng.${cIdx++}`].data;
      if (d.length !== count) throw new Error(`corrupt rng length ${d.length} != ${count}`);
      return d;
    },
  };
  const caNm = new Float32Array(n * 3);
  const coorsNm = batch.coords_nm.data;
  for (let i = 0; i < n; i += 1) {
    const base = (i * 37 + 1) * 3;
    caNm[i * 3] = coorsNm[base];
    caNm[i * 3 + 1] = coorsNm[base + 1];
    caNm[i * 3 + 2] = coorsNm[base + 2];
  }
  const corrupted = corruptBatch({
    n,
    x1: { bb_ca: caNm, local_latents: t[`${caseName}.enc.z_latent`].data },
    partialTDict: meta[`${caseName}_corrupt_t`],
    emit,
    coordCondMask: batch.coord_cond_mask.data,
    moleculeIid: batch.molecule_iid.data,
    rng: corruptRng,
  });
  ok(`[${caseName}] corrupt rng draws consumed`, cIdx === nCorruptDraws, `${cIdx}/${nCorruptDraws}`);
  for (const [part, key] of [
    ['x0', 'x_0'],
    ['x1', 'x_1'],
    ['xt', 'x_t'],
  ]) {
    for (const dm of ['bb_ca', 'local_latents']) {
      const r = compare(corrupted[part][dm], t[`${caseName}.corrupt.${key}.${dm}`].data);
      ok(
        `[${caseName}] corrupt.${key}.${dm}`,
        r.maxAbs < 1e-6 && r.nonFinite === 0,
        `maxAbs=${r.maxAbs.toExponential(2)}`,
      );
    }
  }

  // sampleNoisedTargetBackbone check
  const nNoisedDraws = meta[`${caseName}_noised_bb_n_draws`];
  let nbIdx = 0;
  const noisedRng = {
    randn(count) {
      const d = t[`${caseName}.noised_bb.rng.${nbIdx++}`].data;
      if (d.length !== count) throw new Error(`noised_bb rng length ${d.length} != ${count}`);
      return d;
    },
  };
  const noisedBb = sampleNoisedTargetBackbone({
    n,
    targetBackbone: t[`${caseName}.corrupt.x_1.bb_ca`].data,
    tBb: meta[`${caseName}_noised_bb_t`],
    emitBbCa: emit.bb_ca,
    coordCondMask: batch.coord_cond_mask.data,
    moleculeIid: batch.molecule_iid.data,
    rng: noisedRng,
  });
  const rNb = compare(noisedBb, t[`${caseName}.noised_bb.out`].data);
  ok(
    `[${caseName}] sampleNoisedTargetBackbone`,
    nbIdx === nNoisedDraws && rNb.maxAbs < 1e-6 && rNb.nonFinite === 0,
    `draws=${nbIdx}/${nNoisedDraws} maxAbs=${rNb.maxAbs.toExponential(2)}`,
  );
}

for (const [sname, sinfo] of Object.entries(meta.schedules)) {
  const ifMethod = resolveInverseFoldingMethod(sinfo.cfg_args);
  ok(`[sched:${sname}] inverseFoldingMethod`, ifMethod === sinfo.inverse_folding_method);

  const pt = resolvePartialT(sinfo.cfg_args, DEFAULT_SAMPLING);
  const ptOk =
    Math.abs((pt.bb_ca ?? 0) - sinfo.partial_t.bb_ca) < 1e-6 &&
    Math.abs((pt.local_latents ?? 0) - sinfo.partial_t.local_latents) < 1e-6;
  ok(
    `[${sname}] resolvePartialT`,
    ptOk,
    `got=${JSON.stringify(pt)} want=${JSON.stringify(sinfo.partial_t)}`,
  );

  const sched = buildPartialSchedule(pt, sinfo.actual_nsteps, DEFAULT_SAMPLING, {
    fullNsteps: sinfo.full_nsteps,
  });
  for (const dm of ['bb_ca', 'local_latents']) {
    const rTs = compare(sched[dm].ts, t[`sched.${sname}.ts.${dm}`].data);
    const rGt = compare(sched[dm].gt, t[`sched.${sname}.gt.${dm}`].data);
    ok(
      `[sched:${sname}] ${dm}`,
      rTs.maxAbs < 1e-6 && rGt.relRms < 1e-5,
      `ts maxAbs=${rTs.maxAbs.toExponential(2)} gt relRms=${rGt.relRms.toExponential(2)}`,
    );
  }
}

console.log(failed ? 'FAILED' : 'encoder & partial diffusion match');
process.exitCode = failed ? 1 : 0;
