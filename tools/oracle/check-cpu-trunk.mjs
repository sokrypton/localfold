/**
 * Stage-wise check of the CPU reference against a PyTorch oracle, using the released weights.
 * Every stage is fed the oracle's own input, so an error is attributed to one module.
 *
 *   node tools/oracle/check-cpu-trunk.mjs [case=uncond_short] [layers=all|N]
 */

import { readFileSync } from 'node:fs';
import {
  adaptiveLayerNorm,
  applyNorm,
  linearLayer,
  multiheadAttnAndTransition,
  pairReprUpdate,
  swiGluTransition,
} from '../../cpu/rfd4p/reference.js';
import { compare, loadOracle, readSafetensors, releasedWeightsPath } from './oracle-io.mjs';

const caseName = process.argv[2] ?? 'uncond_short';
const layerArg = process.argv[3] ?? 'all';
const TOL = 1e-4;

const paths = releasedWeightsPath();
const cfg = JSON.parse(readFileSync(paths.config, 'utf8')).config.nn;
console.time('load weights');
const weights = readSafetensors(paths.weights);
console.timeEnd('load weights');
const { t } = loadOracle(new URL('../../oracle-dumps/', import.meta.url).pathname, caseName);

const n = t['batch.padding_mask'].shape[1];
const mask = Uint8Array.from(t['batch.padding_mask'].data);
const D = cfg.token_dim;
const P = cfg.pair_repr_dim;
const C = cfg.dim_cond;
const norm = 'rmsnorm_torch';
const W = (k) => weights[k].data;
const results = [];
let failed = false;

function check(label, got, wantKey) {
  const r = compare(got, t[wantKey].data);
  const ok = r.relRms < TOL && r.nonFinite === 0;
  if (!ok) failed = true;
  results.push({ stage: label, relRms: r.relRms.toExponential(2), maxAbs: r.maxAbs.toExponential(2), ok });
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label.padEnd(28)} relRms=${r.relRms.toExponential(2)} maxAbs=${r.maxAbs.toExponential(2)}`);
}

// c = transition_c_2(transition_c_1(Linear(cond feats)))
{
  const f = t['call0.cond_factory.features'];
  let c = linearLayer(f.data, n, f.shape[2], W('nn.cond_factory.linear_out.weight'), C);
  c = swiGluTransition(c, mask, n, C, 2, weights, 'nn.transition_c_1');
  c = swiGluTransition(c, mask, n, C, 2, weights, 'nn.transition_c_2');
  check('c', c, 'call0.c');
}

// s_init = Linear(seq feats) (mask applied by the caller upstream; the hook sees pre-mask)
{
  const f = t['call0.init_repr_factory.features'];
  const s = linearLayer(f.data, n, f.shape[2], W('nn.init_repr_factory.linear_out.weight'), D);
  check('s_init', s, 'call0.s_init');
}

// z_init = AdaLN(RMSNorm(Linear(pair feats)), RMSNorm(Linear(pair cond)))
{
  const pairMask = new Uint8Array(n * n);
  for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) pairMask[i * n + j] = mask[i] && mask[j];
  const f = t['call0.pair_repr_builder.init_repr_factory.features'];
  let z = linearLayer(f.data, n * n, f.shape[3], W('nn.pair_repr_builder.init_repr_factory.linear_out.weight'), P);
  z = applyNorm(z, n * n, P, norm, W('nn.pair_repr_builder.init_repr_factory.ln_out.weight'));
  const fc = t['call0.pair_repr_builder.cond_factory.features'];
  let zc = linearLayer(fc.data, 1, fc.shape[1], W('nn.pair_repr_builder.cond_factory.linear_out.weight'), 256);
  zc = applyNorm(zc, 1, 256, norm, W('nn.pair_repr_builder.cond_factory.ln_out.weight'));
  z = adaptiveLayerNorm(z, zc, pairMask, n * n, P, 1, 256, norm, weights, 'nn.pair_repr_builder.adaln');
  check('z_init', z, 'call0.z_init');
}

// Trunk layers of the first recycle cycle, each from the oracle's own input.
const c = t['call0.c'].data;
const nLayers = layerArg === 'all' ? cfg.nlayers : Number(layerArg);
let sIn = t['call0.cycle0.s_in'].data;
let zIn = t['call0.cycle0.z_in'].data;
for (let i = 0; i < nLayers; i += 1) {
  const s = multiheadAttnAndTransition(sIn, zIn, c, mask, n, D, P, cfg.nheads, C, true, true, norm, weights,
    `nn.transformer_layers.${i}`);
  check(`attn.${i}`, s, `call0.cycle0.attn.${i}`);
  sIn = t[`call0.cycle0.attn.${i}`].data;
  const pk = `call0.cycle0.pair.${i}`;
  if (t[pk]) {
    const z = pairReprUpdate(sIn, zIn, mask, n, D, P, cfg.tri_mult_c, true, 'hadamard', weights,
      `nn.pair_update_layers.${i}`);
    check(`pair.${i}`, z, pk);
    zIn = t[pk].data;
  }
}

console.log(failed ? 'FAILED' : 'all stages within tolerance');
process.exitCode = failed ? 1 : 0;
