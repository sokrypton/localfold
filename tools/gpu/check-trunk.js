/**
 * GPU trunk vs the PyTorch oracle, with the released weights.
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-trunk.js --case=uncond_short
 *
 * Needs `weights/` (the bundle's config.json + model.safetensors) and `oracle-dumps/<case>.*` served from the repo.
 * Feeds the oracle's call-0 batch and x_t through `shared/rfd4p/features.js`, runs one trunk pass (compared with the
 * oracle's first-cycle states) and then the full recycle depth (compared with the oracle's network outputs).
 */

import { DEFAULT_HF_CONFIG_URL, DEFAULT_HF_WEIGHTS_URL, parseSafetensors } from '../../shared/rfd4p/safetensors.js';
import { pairCondFeatures, pairFeatures, trunkCondFeatures, trunkSeqFeatures } from '../../shared/rfd4p/features.js';
import { Rfd4pTrunkGpu } from '../../webgpu/rfd4p/trunk.js';

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

async function fetchBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.arrayBuffer();
}

/**
 * The released FLOAT weights, which the parity checks compare the GPU against.
 *
 * 🔴 THESE ARE NOT THE INT8 SHARDS THE PAGE RUNS ON, and they are not in this
 * repository either. Parity has to be measured against full precision, so this
 * wants `model.safetensors` - which lives in NVIDIA's own Hugging Face repo
 * under NVIDIA's own terms. A local `weights/model.safetensors` is used when
 * someone has exported or downloaded one; otherwise this reaches for
 * Hugging Face and the download is large.
 *
 * So every check that calls this needs that file. See weights/SOURCE.md.
 */
export async function loadReleased() {
  const local = await fetch('/weights/model.safetensors', { method: 'HEAD' }).catch(() => null);
  const weightsUrl = local && local.ok ? '/weights/model.safetensors' : DEFAULT_HF_WEIGHTS_URL;
  const configUrl = await fetch('/weights/config.json', { method: 'HEAD' })
    .then((r) => (r && r.ok ? '/weights/config.json' : DEFAULT_HF_CONFIG_URL))
    .catch(() => DEFAULT_HF_CONFIG_URL);
  const [cfg, buf] = await Promise.all([
    fetch(configUrl).then((r) => r.json()),
    fetchBuffer(weightsUrl),
  ]);
  return { cfg: cfg.config ?? cfg, weights: parseSafetensors(buf).tensors };
}

export async function loadOracleCase(name) {
  const [meta, buf] = await Promise.all([
    fetch(`/oracle-dumps/${name}.json`).then((r) => r.json()),
    fetchBuffer(`/oracle-dumps/${name}.safetensors`),
  ]);
  return { meta, t: parseSafetensors(buf).tensors };
}

export async function main(device, args) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  if (opt['no-submat'] === '1') device.noSubmat = true;
  const caseName = opt.case ?? 'uncond_short';
  const half = opt.half === '1';
  const t0 = performance.now();
  const { cfg, weights } = await loadReleased();
  const { meta, t } = await loadOracleCase(caseName);
  const nnCfg = cfg.nn;
  const batch = {};
  for (const [key, v] of Object.entries(t)) if (key.startsWith('batch.')) batch[key.slice(6)] = v;
  const n = batch.padding_mask.shape[1];
  const xt = { bb_ca: t['step.0.x_t.bb_ca'].data, local_latents: t['step.0.x_t.local_latents'].data };
  const tt = { bb_ca: t['step.0.t.bb_ca'].data[0], local_latents: t['step.0.t.local_latents'].data[0] };
  const draws = meta.rng.filter((r) => r.fn === 'randperm' && r.call === 0).map((r) => t[`rng.${r.i}`].data);
  let d = 0;
  const randperm = () => draws[d++];
  const f = {
    seq: trunkSeqFeatures(batch, { n, xt, feats: nnCfg.feats_seq, weights, randperm, noiseAlignChirality: nnCfg.noise_align_chirality }).data,
    cond0: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: null, feats: nnCfg.feats_cond_seq, weights }).data,
    cond1: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: 1, feats: nnCfg.feats_cond_seq, weights }).data,
    pair: pairFeatures(batch, { n, caNm: xt.bb_ca, feats: nnCfg.feats_pair_repr,
      dist: { dim: nnCfg.xt_pair_dist_dim, min: nnCfg.xt_pair_dist_min, max: nnCfg.xt_pair_dist_max }, seqSepDim: nnCfg.seq_sep_dim }).data,
    pairCond: pairCondFeatures(tt, nnCfg.feats_pair_cond),
  };
  const tLoad = performance.now();
  const trunk = new Rfd4pTrunkGpu(device, weights, nnCfg, { half });
  const results = [];
  const check = (label, got, key, tol) => {
    const r = compare(got, t[key].data);
    results.push({ label, ok: r.relRms < tol && r.nonFinite === 0, relRms: Number(r.relRms.toExponential(2)), maxAbs: Number(r.maxAbs.toExponential(2)) });
  };
  // One pass: init states and the first cycle against the oracle's per-layer dumps.
  await trunk.forward(n, f, 1);
  const tol = half ? 2e-2 : 1e-4;
  check('c', await trunk.read('c0', n * 256), 'call0.c', tol);
  check('s_init', await trunk.read('sInit', n * 768), 'call0.s_init', tol);
  check('z_init', await trunk.read('zInit', n * n * 384), 'call0.z_init', tol);
  if (t['call0.cycle0.attn.13']) {
    check('cycle0 s', await trunk.read('s', n * 768), 'call0.cycle0.attn.13', tol * 10);
    check('cycle0 z', await trunk.read('z', n * n * 384), 'call0.cycle0.pair.12', tol * 10);
  }
  // Full recycle depth of call 0: network outputs.
  const depth = meta.steps[0].n_recycle;
  await device.queue.onSubmittedWorkDone();
  const tf = performance.now();
  const out = await trunk.forward(n, f, depth, { insertion: true });
  const tRun = performance.now() - tf;
  // Upstream zeroes the flow outputs outside each modality's emit mask (in-context tokens; latents also need a
  // latent-carrying token); the sequence logits are only padding-masked.
  const ctx = batch.is_in_context_conditioning.data;
  const lat = batch.supports_local_latent.data;
  for (let i = 0; i < n; i += 1) {
    if (ctx[i]) out.bb_ca.fill(0, i * 3, i * 3 + 3);
    if (ctx[i] || !lat[i]) out.local_latents.fill(0, i * 32, i * 32 + 32);
  }
  check(`v bb_ca (x${depth})`, out.bb_ca, 'step.0.out.bb_ca.v', tol * 10);
  check(`v latents (x${depth})`, out.local_latents, 'step.0.out.local_latents.v', tol * 10);
  check(`seq logits (x${depth})`, out.logits, 'step.0.out.sequence.logits', tol * 10);
  if (t['step.0.out.insertion.logits']) {
    check(`insertion logits (x${depth})`, out.insertion, 'step.0.out.insertion.logits', tol * 10);
  }
  // Timing: one more call at the same depth (pipelines warm).
  const tw = performance.now();
  await trunk.forward(n, f, depth);
  const tWarm = performance.now() - tw;
  let profileTop = undefined;
  if (opt.profile === '1' && device.features?.has('timestamp-query')) {
    const prof = await trunk.engine.profile(trunk.passPrograms[0]);
    profileTop = [...prof.entries()]
      .sort((a, b) => b[1].ms - a[1].ms)
      .slice(0, 12)
      .map(([k, v]) => ({ op: k, ms: Math.round(v.ms * 100) / 100, calls: v.calls }));
  }
  const mb = (x) => Math.round(x / 2 ** 20);
  const mem = { weights: mb(trunk.engine.weightBytes), activations: mb(trunk.engine.activationBytes) };
  trunk.destroy();
  return {
    case: caseName, n, half, depth,
    results,
    allOk: results.every((r) => r.ok),
    ms: { load: Math.round(tLoad - t0), firstCall: Math.round(tRun), warmCall: Math.round(tWarm), perPass: Math.round((tWarm / depth) * 10) / 10 },
    memMB: mem,
    profileTop,
  };
}
