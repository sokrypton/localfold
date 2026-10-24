/**
 * Partial diffusion and inverse folding schedule & corruption helpers for RFD4-Proteina,
 * ported from upstream `search/vanilla.py` (`resolve_inverse_folding_method`, `resolve_partial_t`,
 * `build_partial_schedule`) and `flow_matching/product_space_flow_matcher.py` (`corrupt_batch`,
 * `sample_noised_target_backbone`).
 */

import { DEFAULT_SAMPLING, sampleSchedule } from './flow.js';

export const RESAMPLED_FORWARD_MARGINAL = 'resampled_forward_marginal';
const INVERSE_FOLDING_METHODS = new Set([RESAMPLED_FORWARD_MARGINAL]);

const f32 = Math.fround;

/** Python round-half-to-even for exact step-count parity. */
function pyRound(x) {
  const fl = Math.floor(x);
  const diff = x - fl;
  if (Math.abs(diff - 0.5) < 1e-12) return fl % 2 === 0 ? fl : fl + 1;
  return Math.round(x);
}

/**
 * Denoising steps a partial trajectory runs (`round(fraction * nsteps)`, half to even). The schedule check in
 * `buildPartialSchedule` needs every caller to agree on this, so use it rather than `Math.round`, which differs on
 * exact halves (50% of 5 steps).
 */
export function partialStepCount(frac, nsteps) {
  return pyRound(frac * nsteps);
}

function validatePartialT(partialTDict) {
  const bad = Object.entries(partialTDict).filter(([, t]) => t >= 1.0);
  if (bad.length > 0) {
    throw new Error(
      `Partial-diffusion start time must be < 1 (t=1 is the clean state, and the score conversion divides by 1 - t); ` +
        `resolved ${JSON.stringify(Object.fromEntries(bad))}.`,
    );
  }
  return partialTDict;
}

/**
 * Validate and return the configured inverse-folding method (`resolve_inverse_folding_method`).
 * @param {object} [opts]
 * @returns {string|null}
 */
export function resolveInverseFoldingMethod(opts = {}) {
  const enabled = opts.inverseFolding ?? opts.inverse_folding ?? false;
  if (!enabled) return null;
  const method = opts.inverseFoldingMethod ?? opts.inverse_folding_method ?? RESAMPLED_FORWARD_MARGINAL;
  if (!INVERSE_FOLDING_METHODS.has(method)) {
    throw new Error(`inverse_folding_method must be one of ${[...INVERSE_FOLDING_METHODS].join(', ')}, got ${method}`);
  }
  return method;
}

/**
 * Extract and validate partial diffusion / inverse folding config (`resolve_partial_t`),
 * returning per-modality starting `t` values (`{ bb_ca, local_latents }`) or `{}` for full generation.
 *
 * Accepts either camelCase (`inverseFolding`, `partialTrajectoryFractionSteps`, `partialTBackbone`,
 * `partialTLatent`, `couplePartialT`) or upstream snake_case keys.
 *
 * @param {object} [opts]
 * @param {object} [sampling]
 * @param {number} [refNsteps]
 * @returns {{bb_ca?: number, local_latents?: number}}
 */
export function resolvePartialT(opts = {}, sampling = DEFAULT_SAMPLING, refNsteps = 10000) {
  const ptBackbone = opts.partialTBackbone ?? opts.partial_t_backbone ?? null;
  const ptLatent = opts.partialTLatent ?? opts.partial_t_latent ?? null;
  const doCouple = Boolean(opts.couplePartialT ?? opts.couple_partial_t ?? false);
  const frac = opts.partialTrajectoryFractionSteps ?? opts.partial_trajectory_fraction_steps ?? null;
  const inverseFoldingMethod = resolveInverseFoldingMethod(opts);

  if (inverseFoldingMethod !== null) {
    if (ptBackbone != null || ptLatent != null || doCouple || frac != null) {
      throw new Error(
        'inverse_folding is mutually exclusive with partial_t_backbone, partial_t_latent, ' +
          'couple_partial_t, and partial_trajectory_fraction_steps.',
      );
    }
    return { bb_ca: 0.0, local_latents: 0.0 };
  }

  if (frac != null) {
    if (ptBackbone != null || ptLatent != null || doCouple) {
      throw new Error(
        'partial_trajectory_fraction_steps is mutually exclusive with partial_t_backbone, ' +
          'partial_t_latent, and couple_partial_t.',
      );
    }
    if (!(frac > 0.0 && frac <= 1.0)) {
      throw new Error(`partial_trajectory_fraction_steps must be in (0, 1], got ${frac}.`);
    }
    const nsteps = opts.nsteps ?? sampling.nsteps;
    const remainingSteps = partialStepCount(frac, nsteps);
    if (remainingSteps < 1) {
      throw new Error(
        `partial_trajectory_fraction_steps=${frac} with nsteps=${nsteps} rounds to zero denoising steps.`,
      );
    }
    const refSched = sampleSchedule({ ...sampling, nsteps });
    const k = nsteps - remainingSteps;
    return validatePartialT({
      bb_ca: f32(refSched.bb_ca.ts[k]),
      local_latents: f32(refSched.local_latents.ts[k]),
    });
  }

  const nSet = (ptBackbone != null ? 1 : 0) + (ptLatent != null ? 1 : 0);
  if (nSet === 0) return {};

  if (nSet === 2) {
    if (doCouple) {
      throw new Error(
        'couple_partial_t=true requires exactly one of partial_t_backbone or partial_t_latent; got both.',
      );
    }
    return validatePartialT({ bb_ca: Number(ptBackbone), local_latents: Number(ptLatent) });
  }

  if (!doCouple) {
    const which = ptBackbone != null ? 'partial_t_backbone' : 'partial_t_latent';
    throw new Error(
      `Only ${which} is set. Either provide both partial_t_backbone and partial_t_latent, or set couple_partial_t=true.`,
    );
  }

  const refSched = sampleSchedule({ ...sampling, nsteps: refNsteps });
  const refTs = ptBackbone != null ? refSched.bb_ca.ts : refSched.local_latents.ts;
  const refVal = ptBackbone != null ? Number(ptBackbone) : Number(ptLatent);
  let k = refNsteps;
  for (let i = 0; i <= refNsteps; i += 1) {
    if (f32(refTs[i]) >= refVal - 1e-6) {
      k = i;
      break;
    }
  }
  return validatePartialT({
    bb_ca: f32(refSched.bb_ca.ts[k]),
    local_latents: f32(refSched.local_latents.ts[k]),
  });
}

/**
 * Build per-modality time and noise schedules covering `[partial_t, 1.0]` with `nsteps` steps
 * (`build_partial_schedule`).
 *
 * Returns `{ bb_ca: { ts, gt, step }, local_latents: { ts, gt, step }, ts, gt }`.
 *
 * @param {{bb_ca?: number, local_latents?: number}} partialTDict
 * @param {number} nsteps
 * @param {object} [sampling]
 * @param {{fullNsteps?: number|null}} [opts]
 */
export function buildPartialSchedule(partialTDict, nsteps, sampling = DEFAULT_SAMPLING, { fullNsteps = null } = {}) {
  const modes = ['bb_ca', 'local_latents'];
  if (fullNsteps != null) {
    if (!(nsteps >= 1 && nsteps <= fullNsteps)) {
      throw new Error(`Partial step count must be in [1, ${fullNsteps}], got ${nsteps}.`);
    }
    const full = sampleSchedule({ ...sampling, nsteps: fullNsteps });
    const out = { ts: {}, gt: {} };
    for (const dm of modes) {
      const ts = full[dm].ts.slice(full[dm].ts.length - (nsteps + 1));
      const gt = full[dm].gt.slice(full[dm].gt.length - nsteps);
      if (Math.abs(f32(ts[0]) - f32(partialTDict[dm])) > 1e-6) {
        throw new Error(`${dm}: partial corruption time must match the first time of the full schedule tail.`);
      }
      out[dm] = { ts, gt, step: full[dm].step };
      out.ts[dm] = ts;
      out.gt[dm] = gt;
    }
    return out;
  }

  const nstepsDense = Math.max(nsteps * 100, 1000);
  const dense = sampleSchedule({ ...sampling, nsteps: nstepsDense });
  const out = { ts: {}, gt: {} };
  for (const dm of modes) {
    const pt = Number(partialTDict[dm] ?? 0.0);
    const denseTs = dense[dm].ts;
    let iStart = nstepsDense;
    for (let i = 0; i <= nstepsDense; i += 1) {
      if (f32(denseTs[i]) >= pt - 1e-6) {
        iStart = i;
        break;
      }
    }
    const nAbove = Math.max(nstepsDense - iStart, 1);
    const nstepsExpanded = pyRound((nsteps * nstepsDense) / nAbove);
    const full = sampleSchedule({ ...sampling, nsteps: nstepsExpanded });
    const ts = full[dm].ts.slice(full[dm].ts.length - (nsteps + 1));
    const gt = full[dm].gt.slice(full[dm].gt.length - nsteps);
    out[dm] = { ts, gt, step: full[dm].step };
    out.ts[dm] = ts;
    out.gt[dm] = gt;
  }
  return out;
}

function zeroComBbNoise(n, emitBbCa, rng) {
  const bb = Float32Array.from(rng.randn(n * 3));
  let cnt = 0;
  const mean = [0, 0, 0];
  for (let i = 0; i < n; i += 1) {
    if (!emitBbCa[i]) {
      bb.fill(0, i * 3, i * 3 + 3);
      continue;
    }
    cnt += 1;
    for (let k = 0; k < 3; k += 1) mean[k] += bb[i * 3 + k];
  }
  for (let k = 0; k < 3; k += 1) mean[k] = f32(mean[k] / Math.max(cnt, 1));
  for (let i = 0; i < n; i += 1) {
    if (!emitBbCa[i]) continue;
    for (let k = 0; k < 3; k += 1) bb[i * 3 + k] = f32(bb[i * 3 + k] - mean[k]);
  }
  return bb;
}

function applyPerGroupOffset(x, n, emitBbCa, coordCondMask, moleculeIid, rng, centeringScale) {
  const out = Float32Array.from(x);
  const offsetMask = (i) => Boolean(emitBbCa[i] && (!coordCondMask || !coordCondMask[i]));
  const groups = [...new Set(Array.from({ length: n }, (_, i) => i).filter(offsetMask).map((i) => moleculeIid[i]))]
    .sort((a, b) => a - b);
  for (const g of groups) {
    const off = rng.randn(3);
    for (let i = 0; i < n; i += 1) {
      if (!offsetMask(i) || moleculeIid[i] !== g) continue;
      for (let k = 0; k < 3; k += 1) out[i * 3 + k] = f32(out[i * 3 + k] + f32(centeringScale * off[k]));
    }
  }
  for (let i = 0; i < n; i += 1) {
    if (!emitBbCa[i]) out.fill(0, i * 3, i * 3 + 3);
  }
  return out;
}

/**
 * Draw the target backbone from its forward marginal at `tBb` (`sample_noised_target_backbone`),
 * used at each step of `resampled_forward_marginal` inverse folding.
 *
 * @param {object} o
 * @param {number} o.n
 * @param {Float32Array} o.targetBackbone `[n, 3]` clean target CA coordinates (nm)
 * @param {number} o.tBb scalar time in `[0, 1)`
 * @param {Uint8Array} o.emitBbCa `[n]`
 * @param {ArrayLike<number>} [o.coordCondMask] `[n]`
 * @param {ArrayLike<number>} o.moleculeIid `[n]`
 * @param {{randn: (k: number) => ArrayLike<number>}} o.rng
 * @param {number} [o.centeringScale]
 * @returns {Float32Array} `[n, 3]`
 */
export function sampleNoisedTargetBackbone({
  n,
  targetBackbone,
  tBb,
  emitBbCa,
  coordCondMask,
  moleculeIid,
  rng,
  centeringScale = 0.2,
}) {
  const x0 = zeroComBbNoise(n, emitBbCa, rng);
  const x0Off = applyPerGroupOffset(x0, n, emitBbCa, coordCondMask, moleculeIid, rng, centeringScale);
  const xt = new Float32Array(n * 3);
  const t = f32(tBb);
  const oneMinusT = f32(1 - t);
  for (let i = 0; i < n; i += 1) {
    if (!emitBbCa[i]) continue;
    for (let k = 0; k < 3; k += 1) {
      const x1v = targetBackbone[i * 3 + k];
      xt[i * 3 + k] = f32(f32(oneMinusT * x0Off[i * 3 + k]) + f32(t * x1v));
    }
  }
  return xt;
}

/**
 * Corrupt a clean sample `x1 = { bb_ca, local_latents }` to starting times `partialTDict`
 * (`ProductSpaceFlowMatcher.corrupt_batch`).
 *
 * RNG draw order matches upstream `corrupt_batch`:
 *   1. `sample_noise(training=True)` for `bb_ca` (`n * 3` normals, zero-COM over `emit.bb_ca`)
 *   2. `sample_noise(training=True)` for `local_latents` (`n * 32` normals)
 *   3. `interpolate` for `bb_ca`: per-`molecule_iid` stochastic offset (`3` normals per group),
 *      then `(1 - t_bb) * x0_offset + t_bb * x1.bb_ca`
 *   4. `interpolate` for `local_latents`: `(1 - t_ll) * x0.local_latents + t_ll * x1.local_latents`
 *
 * @param {object} o
 * @param {number} o.n
 * @param {{bb_ca: Float32Array, local_latents: Float32Array}} o.x1
 * @param {{bb_ca?: number, local_latents?: number}} o.partialTDict
 * @param {{bb_ca: Uint8Array, local_latents: Uint8Array}} o.emit
 * @param {ArrayLike<number>} [o.coordCondMask]
 * @param {ArrayLike<number>} o.moleculeIid
 * @param {{randn: (k: number) => ArrayLike<number>}} o.rng
 * @param {number} [o.centeringScale]
 * @returns {{x0: {bb_ca: Float32Array, local_latents: Float32Array}, x1: {bb_ca: Float32Array, local_latents: Float32Array}, xt: {bb_ca: Float32Array, local_latents: Float32Array}, t: {bb_ca: number, local_latents: number}}}
 */
export function corruptBatch({
  n,
  x1,
  partialTDict,
  emit,
  coordCondMask,
  moleculeIid,
  rng,
  centeringScale = 0.2,
}) {
  const tBb = f32(partialTDict.bb_ca ?? 0.0);
  const tLl = f32(partialTDict.local_latents ?? 0.0);

  const x1Bb = Float32Array.from(x1.bb_ca);
  const x1Ll = Float32Array.from(x1.local_latents);
  for (let i = 0; i < n; i += 1) {
    if (!emit.bb_ca[i]) x1Bb.fill(0, i * 3, i * 3 + 3);
    if (!emit.local_latents[i]) x1Ll.fill(0, i * 32, i * 32 + 32);
  }

  // 1. sample_noise(training=True) for bb_ca and local_latents
  const x0Bb = zeroComBbNoise(n, emit.bb_ca, rng);
  const x0Ll = Float32Array.from(rng.randn(n * 32));
  for (let i = 0; i < n; i += 1) {
    if (!emit.local_latents[i]) x0Ll.fill(0, i * 32, i * 32 + 32);
  }

  // 2. interpolate for bb_ca (applies x_0 stochastic offset) and local_latents
  const x0BbOff = applyPerGroupOffset(x0Bb, n, emit.bb_ca, coordCondMask, moleculeIid, rng, centeringScale);
  const xtBb = new Float32Array(n * 3);
  const oneMinusTBb = f32(1 - tBb);
  for (let i = 0; i < n; i += 1) {
    if (!emit.bb_ca[i]) continue;
    for (let k = 0; k < 3; k += 1) {
      xtBb[i * 3 + k] = f32(f32(oneMinusTBb * x0BbOff[i * 3 + k]) + f32(tBb * x1Bb[i * 3 + k]));
    }
  }

  const xtLl = new Float32Array(n * 32);
  const oneMinusTLl = f32(1 - tLl);
  for (let i = 0; i < n; i += 1) {
    if (!emit.local_latents[i]) continue;
    for (let k = 0; k < 32; k += 1) {
      xtLl[i * 32 + k] = f32(f32(oneMinusTLl * x0Ll[i * 32 + k]) + f32(tLl * x1Ll[i * 32 + k]));
    }
  }

  return {
    x0: { bb_ca: x0Bb, local_latents: x0Ll },
    x1: { bb_ca: x1Bb, local_latents: x1Ll },
    xt: { bb_ca: xtBb, local_latents: xtLl },
    t: { bb_ca: tBb, local_latents: tLl },
  };
}
