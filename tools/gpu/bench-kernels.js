/**
 * A/B GEMM kernel variants inside the real trunk: per-kernel GPU time of one trunk pass (`passPrograms[0]`) and the
 * trunk output against the baseline variant, interleaved over rounds (laptops throttle; never trust back-to-back).
 *
 *   node tools/gpu-chrome.mjs tools/gpu/bench-kernels.js --cases=uncond_short,ppi_pdl1 --rounds=3 \
 *     --variants='[{},{"smemF32":true}]'
 *
 * A variant is an `Rfd4pEngine.gemmVariant` object. Needs `timestamp-query`.
 */
import { pairCondFeatures, pairFeatures, trunkCondFeatures, trunkSeqFeatures } from '../../shared/rfd4p/features.js';
import { Rfd4pTrunkGpu } from '../../webgpu/rfd4p/trunk.js';
import { loadOracleCase, loadReleased } from './check-trunk.js';

function features(t, meta, nnCfg, weights) {
  const batch = {};
  for (const [key, v] of Object.entries(t)) if (key.startsWith('batch.')) batch[key.slice(6)] = v;
  const n = batch.padding_mask.shape[1];
  const xt = { bb_ca: t['step.0.x_t.bb_ca'].data, local_latents: t['step.0.x_t.local_latents'].data };
  const tt = { bb_ca: t['step.0.t.bb_ca'].data[0], local_latents: t['step.0.t.local_latents'].data[0] };
  const draws = meta.rng.filter((r) => r.fn === 'randperm' && r.call === 0).map((r) => t[`rng.${r.i}`].data);
  let d = 0;
  return {
    n,
    f: {
      seq: trunkSeqFeatures(batch, { n, xt, feats: nnCfg.feats_seq, weights, randperm: () => draws[d++], noiseAlignChirality: nnCfg.noise_align_chirality }).data,
      cond0: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: null, feats: nnCfg.feats_cond_seq, weights }).data,
      cond1: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: 1, feats: nnCfg.feats_cond_seq, weights }).data,
      pair: pairFeatures(batch, { n, caNm: xt.bb_ca, feats: nnCfg.feats_pair_repr,
        dist: { dim: nnCfg.xt_pair_dist_dim, min: nnCfg.xt_pair_dist_min, max: nnCfg.xt_pair_dist_max }, seqSepDim: nnCfg.seq_sep_dim }).data,
      pairCond: pairCondFeatures(tt, nnCfg.feats_pair_cond),
    },
  };
}

const relRms = (a, b) => {
  let se = 0, sw = 0;
  for (let i = 0; i < b.length; i += 1) { se += (a[i] - b[i]) ** 2; sw += b[i] ** 2; }
  return Math.sqrt(se / Math.max(sw, 1e-30));
};

export async function main(device, args) {
  if (!device.features?.has('timestamp-query')) throw new Error('needs timestamp-query');
  const opt = Object.fromEntries(args.map((a) => { const i = a.indexOf('='); return [a.slice(2, i), a.slice(i + 1)]; }));
  const variants = JSON.parse(opt.variants ?? '[{},{"smemF32":true}]');
  const rounds = Number(opt.rounds ?? 3);
  const { cfg, weights } = await loadReleased();
  const trunk = new Rfd4pTrunkGpu(device, weights, cfg.nn, { half: true });
  const e = trunk.engine;
  const out = [];
  for (const caseName of (opt.cases ?? 'uncond_short,ppi_pdl1').split(',')) {
    const { meta, t } = await loadOracleCase(caseName);
    const { n, f } = features(t, meta, cfg.nn, weights);
    const best = variants.map(() => ({ total: Infinity, kernels: new Map() }));
    const outputs = [];
    for (let r = 0; r < rounds; r += 1) {
      for (let v = 0; v < variants.length; v += 1) {
        e.gemmVariant = variants[v];
        trunk.n = -1; // re-record with this variant's pipelines
        await trunk.forward(n, f, 1);
        if (r === 0) outputs[v] = { s: await trunk.read('s', n * 768), z: await trunk.read('z', n * n * 384) };
        const prof = await e.profile(trunk.passPrograms[0]);
        let total = 0;
        for (const { ms } of prof.values()) total += ms;
        if (total < best[v].total) best[v] = { total, kernels: prof };
      }
    }
    const base = best[0];
    out.push({
      case: caseName,
      n,
      variants: variants.map((v, k) => ({
        variant: v,
        passMs: Number(best[k].total.toFixed(2)),
        speedup: Number((base.total / best[k].total).toFixed(3)),
        sRel: Number(relRms(outputs[k].s, outputs[0].s).toExponential(2)),
        zRel: Number(relRms(outputs[k].z, outputs[0].z).toExponential(2)),
        top: [...best[k].kernels].sort((a, b) => b[1].ms - a[1].ms).slice(0, 6)
          .map(([label, { ms }]) => `${label} ${ms.toFixed(1)} (${((base.kernels.get(label)?.ms ?? ms) / ms).toFixed(2)}x)`),
      })),
    });
  }
  trunk.destroy();
  return out;
}
