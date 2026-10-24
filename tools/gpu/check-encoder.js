/**
 * GPU autoencoder encoder (`webgpu/rfd4p/encoder.js`) vs the PyTorch oracle (`oracle-dumps/encoder_oracle.*`).
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-encoder.js [--case=all] [--half=0]
 */

import { Rfd4pEncoderGpu } from '../../webgpu/rfd4p/encoder.js';
import { loadOracleCase, loadReleased } from './check-trunk.js';

function compare(got, want) {
  let se = 0;
  let sw = 0;
  let maxAbs = 0;
  let bad = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    if (!Number.isFinite(got[i])) bad += 1;
    se += d * d;
    sw += want[i] * want[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return { relRms: Math.sqrt(se / Math.max(sw, 1e-30)), maxAbs, nonFinite: bad };
}

export async function main(device, args) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const caseArg = opt.case ?? 'all';
  const half = opt.half === '1';

  const t0 = performance.now();
  const { cfg, weights } = await loadReleased();
  const encCfg = cfg.autoencoder_config.nn_ae.encoder;
  const encoder = new Rfd4pEncoderGpu(device, weights, encCfg, { half });
  const tLoad = performance.now();

  const { meta, t } = await loadOracleCase('encoder_oracle');
  const cases = caseArg === 'all' ? meta.cases : caseArg.split(',');

  const tol = half ? 1e-2 : 1e-4;
  const results = [];
  for (const caseName of cases) {
    const batch = {};
    const prefix = `${caseName}.batch.`;
    for (const [k, v] of Object.entries(t)) {
      if (k.startsWith(prefix)) batch[k.slice(prefix.length)] = v;
    }
    const n = batch.padding_mask.shape[1];
    const eps = t[`${caseName}.enc.eps`].data;
    const rng = { randn: () => eps };

    const tStart = performance.now();
    const out = await encoder.encode(batch, { n, rng });
    const ms = Math.round((performance.now() - tStart) * 10) / 10;

    const wantEmit = t[`${caseName}.enc.emit_mask`].data;
    let emitDiff = 0;
    for (let i = 0; i < n; i += 1) if (out.emitMask[i] !== wantEmit[i]) emitDiff += 1;

    const rMean = compare(out.mean, t[`${caseName}.enc.mean`].data);
    const rLogScale = compare(out.logScale, t[`${caseName}.enc.log_scale`].data);
    const rZ = compare(out.zLatent, t[`${caseName}.enc.z_latent`].data);

    const ok =
      emitDiff === 0 &&
      rMean.relRms < tol &&
      rLogScale.relRms < tol &&
      rZ.relRms < tol &&
      (half || (rMean.maxAbs < tol && rLogScale.maxAbs < tol)) &&
      rMean.nonFinite === 0 &&
      rLogScale.nonFinite === 0 &&
      rZ.nonFinite === 0;

    results.push({
      case: caseName,
      n,
      na: encoder.na,
      ok,
      emitDiff,
      meanRelRms: Number(rMean.relRms.toExponential(2)),
      meanMaxAbs: Number(rMean.maxAbs.toExponential(2)),
      logScaleRelRms: Number(rLogScale.relRms.toExponential(2)),
      logScaleMaxAbs: Number(rLogScale.maxAbs.toExponential(2)),
      zRelRms: Number(rZ.relRms.toExponential(2)),
      zMaxAbs: Number(rZ.maxAbs.toExponential(2)),
      ms,
    });
  }

  const mb = (x) => Math.round((x / 2 ** 20) * 10) / 10;
  const memMB = { weights: mb(encoder.engine.weightBytes), activations: mb(encoder.engine.activationBytes) };
  encoder.destroy();

  return {
    half,
    loadMs: Math.round(tLoad - t0),
    memMB,
    allOk: results.every((r) => r.ok),
    results,
  };
}
