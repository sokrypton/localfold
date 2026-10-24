/**
 * GPU autoencoder decoder (`webgpu/rfd4p/decoder.js`) vs the PyTorch oracle across all 6 cases.
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-decoder.js [--case=all] [--half=0]
 */

import { sequenceForDecoder } from '../../cpu/rfd4p/decoder.js';
import { Rfd4pDecoderGpu } from '../../webgpu/rfd4p/decoder.js';
import { loadOracleCase, loadReleased } from './check-trunk.js';

const ALL_CASES = ['uncond_short', 'ppi_pdl1', 'sm_sam', 'dna_7m5w', 'motif_idx_1prw', 'motif_unidx_1prw'];

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
  const cases = caseArg === 'all' ? ALL_CASES : caseArg.split(',');
  const half = opt.half === '1';

  const t0 = performance.now();
  const { cfg, weights } = await loadReleased();
  const decCfg = cfg.autoencoder_config.nn_ae.decoder;
  const decoder = new Rfd4pDecoderGpu(device, weights, decCfg, { half });
  const tLoad = performance.now();

  const tol = half ? 1e-2 : 1e-4;
  const results = [];
  for (const caseName of cases) {
    const { meta, t } = await loadOracleCase(caseName);
    const batch = {};
    for (const [k, v] of Object.entries(t)) if (keyStartsWithBatch(k)) batch[k.slice(6)] = v;
    const n = batch.padding_mask.shape[1];
    const last = meta.steps.length - 1;

    const seq = sequenceForDecoder(batch, n, t[`step.${last}.out.sequence.logits`].data);
    const wantSeq = t['decode.in.C_SEQ'].data;
    const seqOk = seq.every((v, i) => v === wantSeq[i]);

    const tStart = performance.now();
    const out = await decoder.decode(batch, {
      n,
      caNm: t['decode.in.ca_coors_nm'].data,
      z: t['decode.in.z_latent'].data,
      seq: Int32Array.from(wantSeq),
    });
    const ms = Math.round((performance.now() - tStart) * 10) / 10;

    const wantRt = t['decode.out.residue_type'].data;
    let rtDiff = 0;
    for (let i = 0; i < n; i += 1) if (out.residueType[i] !== wantRt[i]) rtDiff += 1;

    const wantMask = t['decode.out.atom_mask'].data;
    let maskDiff = 0;
    for (let i = 0; i < wantMask.length; i += 1) if (out.atomMask[i] !== wantMask[i]) maskDiff += 1;

    const r = compare(out.coorsNm, t['decode.out.coors_nm'].data);
    const ok = seqOk && rtDiff === 0 && maskDiff === 0 && r.maxAbs < tol && r.nonFinite === 0;
    results.push({
      case: caseName,
      n,
      na: decoder.na,
      ok,
      seqOk,
      rtDiff,
      maskDiff,
      relRms: Number(r.relRms.toExponential(2)),
      maxAbsNm: Number(r.maxAbs.toExponential(2)),
      ms,
    });
  }

  const mb = (x) => Math.round((x / 2 ** 20) * 10) / 10;
  const memMB = { weights: mb(decoder.engine.weightBytes), activations: mb(decoder.engine.activationBytes) };
  decoder.destroy();

  return {
    half,
    loadMs: Math.round(tLoad - t0),
    memMB,
    allOk: results.every((r) => r.ok),
    results,
  };
}

function keyStartsWithBatch(k) {
  return k.startsWith('batch.');
}
