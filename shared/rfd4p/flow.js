/**
 * RFD4-Proteina flow-matching sampler, ported from upstream
 * `flow_matching/product_space_flow_matcher.py` and `flow_matching/rdn_flow_matcher.py`.
 *
 * Two modalities are integrated from t=0 (noise) to t=1 (data): `bb_ca` (CA coordinates, nm, 3
 * per token) and `local_latents` (32 per token). The network predicts the velocity `v` for both.
 *
 * Randomness goes through an injectable `rng` so a recorded PyTorch trajectory can be replayed:
 *   rng.randn(count) -> Float32Array of standard normals, in upstream's draw order.
 */

/** Released inference sampling arguments (`configs/generation/inference_base.yaml`). */
export const DEFAULT_SAMPLING = {
  nsteps: 60,
  bb_ca: {
    schedule: { mode: 'log', p: 2.0, tail_steps: 10, tail_step_multiplier: 2 },
    gt: { mode: '1/t', p: 1.0, clamp_val: null },
    step: { sampling_mode: 'vf_tsr', sc_scale_noise: 1.0, t_lim_ode: 0.95, t_lim_ode_below: 0.02, tsr_k: 16.0, tsr_sigma: 3.0 },
  },
  local_latents: {
    schedule: { mode: 'power', p: 2.0, tail_steps: 10, tail_step_multiplier: 2 },
    gt: { mode: 'tan', p: 1.0, clamp_val: null },
    step: { sampling_mode: 'vf_tsr', sc_scale_noise: 1.0, t_lim_ode: 0.95, t_lim_ode_below: 0.02, tsr_k: 4.0, tsr_sigma: 12.0 },
  },
  /** `product_flowmatcher.bb_ca` stochastic centering (bundle config). */
  centering: { scale: 0.2 },
};

/**
 * Recycle ("thinking") schedules, every `configs/recycle_strategy/<budget>/<shape>.yaml`:
 * `[width, depth]` bands over the 60 steps. Depth is the total number of trunk passes per step.
 * A bare budget (`low`, `medium`, `high`) and `thinking_<budget>` are `front_loaded`, as upstream's
 * `thinking_*.yaml` say; `thinking_medium` is upstream's default (`configs/inference.yaml`).
 *
 * 🔴 `high` WAS `[[15,6],[15,5],[15,4],[15,3]]` HERE AND IN cuda/rfd4p, which is no upstream file:
 * 270 passes at depths 3-6 where upstream's is 300 at 2-9. `low` and `medium` were right and every
 * preset names `low`, so nothing that ran could see it. The table is now every file, transcribed.
 */
const RECYCLE_BANDS = {
  low: {
    flat: [[60, 2]],
    front_loaded: [[10, 4], [10, 3], [10, 2], [30, 1]],
    midpeak: [[10, 1], [5, 2], [5, 3], [10, 4], [5, 3], [5, 2], [20, 1]],
  },
  medium: {
    flat: [[60, 3]],
    front_loaded: [[5, 6], [10, 5], [5, 4], [10, 3], [20, 2], [10, 1]],
    midpeak: [[5, 1], [5, 2], [5, 2], [5, 4], [10, 5], [5, 4], [5, 2], [5, 2], [15, 1]],
  },
  high: {
    flat: [[60, 5]],
    front_loaded: [[10, 9], [10, 8], [10, 5], [20, 3], [10, 2]],
    midpeak: [[5, 1], [5, 2], [5, 3], [5, 5], [10, 6], [5, 5], [5, 3], [5, 2], [15, 1]],
  },
};
export const RECYCLE_SCHEDULES = { none: [[60, 1]] };
for (const [budget, shapes] of Object.entries(RECYCLE_BANDS)) {
  RECYCLE_SCHEDULES[budget] = shapes.front_loaded;
  RECYCLE_SCHEDULES[`thinking_${budget}`] = shapes.front_loaded;
  for (const [shape, bands] of Object.entries(shapes)) RECYCLE_SCHEDULES[`${budget}/${shape}`] = bands;
}

/** Upstream's default recycle strategy (`configs/inference.yaml`: `thinking_medium`). */
export const DEFAULT_RECYCLE_SCHEDULE = 'medium';

/** Expand `[width, depth]` bands into one depth per step (`Proteina.recycle_depths`). */
export function recycleDepths(schedule, nsteps, fullNsteps = nsteps) {
  if (typeof schedule === 'number') return new Array(nsteps).fill(Math.max(1, schedule));
  if (typeof schedule === 'string' && /^\d+$/.test(schedule)) {
    return new Array(nsteps).fill(Math.max(1, Number(schedule)));
  }
  if (Array.isArray(schedule) && typeof schedule[0] === 'number') {
    if (schedule.length !== fullNsteps) throw new Error(`recycle depths length ${schedule.length}, want ${fullNsteps}`);
    return schedule.slice(schedule.length - nsteps);
  }
  const bands = typeof schedule === 'string' ? RECYCLE_SCHEDULES[schedule] : schedule;
  if (!bands) throw new Error(`unknown recycle schedule ${schedule}`);
  const depths = bands.flatMap(([w, d]) => new Array(w).fill(d));
  if (depths.length === fullNsteps) {
    return depths.slice(depths.length - nsteps);
  }
  if (depths.length > 0 && fullNsteps >= nsteps) {
    const fullDepths = Array.from({ length: fullNsteps }, (_, i) =>
      depths[Math.min(depths.length - 1, Math.floor((i * depths.length) / fullNsteps))]
    );
    return fullDepths.slice(fullDepths.length - nsteps);
  }
  throw new Error(`recycle schedule expands to ${depths.length}, want ${fullNsteps}`);
}

const f32 = Math.fround;

function linspace(a, b, k) {
  const out = new Float64Array(k);
  for (let i = 0; i < k; i += 1) out[i] = k === 1 ? a : a + ((b - a) * i) / (k - 1);
  return out;
}

/** `get_schedule_tsr_safe` (eps 1e-3) without the tail multiplier: `nsteps + 1` points. */
function baseScheduleTsrSafe(mode, nsteps, p, eps = 1e-3) {
  if (mode === 'uniform') return linspace(eps, 1, nsteps + 1);
  if (mode === 'power') return linspace(eps ** (1 / p), 1, nsteps + 1).map((v) => v ** p);
  if (mode === 'log') {
    const e = linspace(-p, eps, nsteps + 1);
    const t = new Float64Array(nsteps + 1);
    for (let i = 0; i <= nsteps; i += 1) t[i] = 1 - 10 ** e[nsteps - i];
    let mn = Infinity;
    for (const v of t) mn = Math.min(mn, v);
    for (let i = 0; i <= nsteps; i += 1) t[i] -= mn - eps;
    let mx = -Infinity;
    for (const v of t) mx = Math.max(mx, v);
    for (let i = 0; i <= nsteps; i += 1) t[i] /= mx;
    return t;
  }
  throw new Error(`schedule mode ${mode} not supported`);
}

/** `get_schedule_tsr_safe` with `_schedule_with_tail_multiplier`. */
export function scheduleTsrSafe({ mode, p, tail_steps = 0, tail_step_multiplier = 1 }, nsteps) {
  const m = tail_step_multiplier;
  const tsSteps = nsteps >= tail_steps * m ? tail_steps : Math.floor(nsteps / (2 * Math.max(1, m)));
  if (!tsSteps) return baseScheduleTsrSafe(mode, nsteps, p);
  const base = nsteps - (m - 1) * tsSteps;
  const fine = base * m;
  const tailStart = (base - tsSteps) * m;
  const full = baseScheduleTsrSafe(mode, fine, p);
  const idx = [];
  for (let i = 0; i < tailStart; i += m) idx.push(i);
  for (let i = tailStart; i <= fine; i += 1) idx.push(i);
  return Float64Array.from(idx, (i) => full[i]);
}

/** `compute_tsr_ratio`. */
export function tsrRatio(t, k, sigma) {
  const snr = (t * t) / ((1 - t) * (1 - t));
  return (snr * sigma * sigma + 1) / ((snr * sigma * sigma) / k + 1);
}

/** `get_gt` (f_pow 1, eps 1e-2), divided by tsr_ratio^2 for `vf_tsr`. */
export function noiseSchedule(ts, { mode, p, clamp_val }, step) {
  if (p !== 1) throw new Error('gt transform p != 1 not implemented');
  const g = new Float64Array(ts.length - 1);
  for (let i = 0; i < g.length; i += 1) {
    const t = Math.min(Math.max(ts[i], 0), 1 - 1e-5);
    let v;
    if (mode === '1/t') v = 1 / (t + 1e-2);
    else if (mode === '1-t/t') v = (1 - t) / (t + 1e-2);
    else if (mode === 'tan') v = ((Math.PI / 2) * Math.sin(((1 - t) * Math.PI) / 2)) / (Math.cos(((1 - t) * Math.PI) / 2) + 1e-2);
    else if (mode === 'ode') v = 0;
    else throw new Error(`gt mode ${mode} not implemented`);
    if (clamp_val != null) v = Math.min(v, clamp_val);
    if (step.sampling_mode === 'vf_tsr') v /= tsrRatio(ts[i], step.tsr_k ?? 1, step.tsr_sigma ?? 1) ** 2;
    g[i] = v;
  }
  return g;
}

/** Per-modality time and noise schedules (`ProductSpaceFlowMatcher.sample_schedule`). */
export function sampleSchedule(sampling = DEFAULT_SAMPLING) {
  const out = {};
  for (const dm of ['bb_ca', 'local_latents']) {
    const a = sampling[dm];
    if (a.step.sampling_mode !== 'vf_tsr') throw new Error('only vf_tsr sampling is ported');
    const ts = scheduleTsrSafe(a.schedule, sampling.nsteps);
    out[dm] = { ts, gt: noiseSchedule(ts, a.gt, a.step), step: a.step };
  }
  return out;
}

/**
 * Emit masks (`compute_flow_emit_masks`).
 * @param {{padding_mask: ArrayLike<number>, is_in_context_conditioning: ArrayLike<number>,
 *   supports_local_latent: ArrayLike<number>}} m
 */
export function flowEmitMasks(m, n) {
  const bb = new Uint8Array(n);
  const ll = new Uint8Array(n);
  for (let i = 0; i < n; i += 1) {
    bb[i] = m.padding_mask[i] && !m.is_in_context_conditioning[i] ? 1 : 0;
    ll[i] = bb[i] && m.supports_local_latent[i] ? 1 : 0;
  }
  return { bb_ca: bb, local_latents: ll };
}

/**
 * Initial noise (`sample_noise` for both modalities, inference): standard normal, emit-masked;
 * bb_ca zero-centred over its emitted tokens, then one N(0, scale^2) offset per molecule
 * instance (`per_group` over `molecule_iid`, ascending) for tokens without absolute
 * coordinate conditioning. Draw order matches upstream: bb_ca noise, its group offsets, latents.
 * @param {object} o
 * @param {number} o.n
 * @param {{bb_ca: Uint8Array, local_latents: Uint8Array}} o.emit
 * @param {ArrayLike<number>} o.coordCondMask
 * @param {ArrayLike<number>} o.moleculeIid
 * @param {{randn: (k: number) => ArrayLike<number>}} o.rng
 * @param {number} [o.centeringScale]
 */
export function sampleInitialNoise({ n, emit, coordCondMask, moleculeIid, rng, centeringScale = 0.2 }) {
  const bb = Float32Array.from(rng.randn(n * 3));
  let cnt = 0;
  const mean = [0, 0, 0];
  for (let i = 0; i < n; i += 1) {
    if (!emit.bb_ca[i]) {
      bb.fill(0, i * 3, i * 3 + 3);
      continue;
    }
    cnt += 1;
    for (let k = 0; k < 3; k += 1) mean[k] += bb[i * 3 + k];
  }
  for (let k = 0; k < 3; k += 1) mean[k] = f32(mean[k] / Math.max(cnt, 1));
  for (let i = 0; i < n; i += 1) {
    if (!emit.bb_ca[i]) continue;
    for (let k = 0; k < 3; k += 1) bb[i * 3 + k] = f32(bb[i * 3 + k] - mean[k]);
  }
  const offsetMask = (i) => emit.bb_ca[i] && !coordCondMask[i];
  const groups = [...new Set(Array.from({ length: n }, (_, i) => i).filter(offsetMask).map((i) => moleculeIid[i]))]
    .sort((a, b) => a - b);
  for (const g of groups) {
    const off = rng.randn(3);
    for (let i = 0; i < n; i += 1) {
      if (!offsetMask(i) || moleculeIid[i] !== g) continue;
      for (let k = 0; k < 3; k += 1) bb[i * 3 + k] = f32(bb[i * 3 + k] + f32(centeringScale * off[k]));
    }
  }
  const ll = Float32Array.from(rng.randn(n * 32));
  for (let i = 0; i < n; i += 1) if (!emit.local_latents[i]) ll.fill(0, i * 32, i * 32 + 32);
  return { bb_ca: bb, local_latents: ll };
}

/**
 * One `vf_tsr` Euler(-Maruyama) step for one modality (`simulation_step`), in place.
 * Noise is drawn only inside the stochastic window, as upstream does.
 * @returns {Float32Array} the clean-sample estimate x1 = x_t + (1 - t) v (emit-masked)
 */
export function simulationStep({ x, v, t, dt, gt, emit, dim, step, rng }) {
  const { tsr_k: k, tsr_sigma: sigma, t_lim_ode: hi, t_lim_ode_below: lo, sc_scale_noise: scn } = step;
  const n = emit.length;
  const x1 = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) x1[i] = emit[(i / dim) | 0] ? f32(x[i] + f32((1 - t) * v[i])) : 0;
  if (t === 0) return x1; // v_scaled = 0
  const r = tsrRatio(t, k, sigma);
  const stochastic = !(t > hi || t < lo);
  const eps = stochastic ? rng.randn(x.length) : null;
  const std = stochastic ? Math.sqrt(2 * gt * scn * dt) : 0;
  for (let i = 0; i < x.length; i += 1) {
    const score = (t * v[i] - x[i]) / (1 - t);
    const ss = score * r;
    const vs = (x[i] + (1 - t) * ss) / t;
    const delta = stochastic ? (vs + gt * ss) * dt + std * eps[i] : vs * dt;
    x[i] = emit[(i / dim) | 0] ? f32(x[i] + delta) : 0;
  }
  void n;
  return x1;
}

/** A seeded standard-normal source (xorshift128+ / Box–Muller) for interactive sampling. */
export function seededRng(seed = 0) {
  let s0 = BigInt.asUintN(64, BigInt(seed) * 0x9e3779b97f4a7c15n + 1n);
  let s1 = BigInt.asUintN(64, s0 ^ 0xbf58476d1ce4e5b9n);
  const next = () => {
    let a = s0;
    const b = s1;
    s0 = b;
    a ^= BigInt.asUintN(64, a << 23n);
    s1 = a ^ b ^ (a >> 17n) ^ (b >> 26n);
    return Number(BigInt.asUintN(53, s1 + b)) / 2 ** 53;
  };
  return {
    randn(count) {
      const out = new Float32Array(count);
      for (let i = 0; i < count; i += 2) {
        const u = Math.max(next(), 1e-300);
        const w = next();
        const r = Math.sqrt(-2 * Math.log(u));
        out[i] = r * Math.cos(2 * Math.PI * w);
        if (i + 1 < count) out[i + 1] = r * Math.sin(2 * Math.PI * w);
      }
      return out;
    },
    randperm(k) {
      const p = Array.from({ length: k }, (_, i) => i);
      for (let i = k - 1; i > 0; i -= 1) {
        const j = Math.floor(next() * (i + 1));
        [p[i], p[j]] = [p[j], p[i]];
      }
      return p;
    },
  };
}
