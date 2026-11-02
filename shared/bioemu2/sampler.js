/**
 * BioEmu2's steered samplers - edm_gd, edm_fk and edm_fkc - generic over the denoiser.
 *
 * A transcription of af3-any-model's network/bioemu2_steering.steered_sample
 * and fkc_sample (bioemu2 @7846769's edm_steered_sampler and
 * edm_fkc_sampler). Every particle is a sample; Feynman-Kac resampling couples
 * them, so all particles step together. The potentials are potentials.js's,
 * evaluated on the host; the denoiser is the caller's (a GPU or native
 * network), handed in as
 *
 *     denoise(x: Float64Array[] (P particles of nPad * 3), sigma: number, companion) -> Float64Array[]
 *
 * and, for edm_fkc, `denoiseWithGrad(x, sigma, companion, seed)`: the denoised
 * coordinates plus the vector-Jacobian product of the potential U(D(x, sigma))
 * with respect to x and to sigma - the reverse pass through the denoiser that
 * the Feynman-Kac corrector needs.
 *
 * The rigid alignment is the caller's too - `align(x, target, mask)`, the
 * weighted Kabsch the other realigning samplers use (cpu/af3/diffusion/rigid-align.js).
 *
 * Every random draw comes from `draws`: {x0(shape), step(k) -> {rot, trans,
 * eps}, uniform(k, groups)}, so a parity harness can replay the reference's
 * and a fold can use its own generator.
 */
import { energies, stepParameters, guidanceInterval, PARAMS, DEFAULT_SPECS, SIGMA_DATA, SIGMA_MIN, SIGMA_MAX, RHO }
  from "./potentials.js";

/** edm_sample_schedule: num_steps Karras levels (x sigma_data), then 0, in float32 as the vendor's. */
export function schedule(numSteps) {
  if (numSteps < 2) throw new Error(`BioEmu2 samples with at least 2 steps, got ${numSteps}`);
  const f = Math.fround;
  const inv = 1.0 / RHO;
  const hi = f(f(SIGMA_MAX) ** inv);
  const lo = f(f(SIGMA_MIN) ** inv);
  const out = new Float32Array(numSteps + 1);
  for (let k = 0; k < numSteps; k += 1) {
    const s = f(f(hi + f(f(k / f(numSteps - 1)) * f(lo - hi))) ** RHO);
    out[k] = f(s * f(SIGMA_DATA));
  }
  return out;
}

const centreAndRotate = (x, mask, rot, trans) => {
  const n = mask.length;
  let w = 0;
  const centre = [0, 0, 0];
  for (let a = 0; a < n; a += 1) if (mask[a]) { w += 1; for (let d = 0; d < 3; d += 1) centre[d] += x[a * 3 + d]; }
  for (let d = 0; d < 3; d += 1) centre[d] /= w;
  const out = new Float64Array(x.length);
  for (let a = 0; a < n; a += 1) {
    const v = [x[a * 3] - centre[0], x[a * 3 + 1] - centre[1], x[a * 3 + 2] - centre[2]];
    for (let s = 0; s < 3; s += 1) out[a * 3 + s] = v[0] * rot[s] + v[1] * rot[3 + s] + v[2] * rot[6 + s] + trans[s];
  }
  return out;
};

/** x @ R, per atom (the score-guidance update rotates with its particle; no centring). */
const rotateOnly = (x, rot) => {
  const out = new Float64Array(x.length);
  for (let a = 0; a < x.length / 3; a += 1) {
    for (let s = 0; s < 3; s += 1) out[a * 3 + s] = x[a * 3] * rot[s] + x[a * 3 + 1] * rot[3 + s] + x[a * 3 + 2] * rot[6 + s];
  }
  return out;
};

/** softmax within each group of `group` consecutive particles. */
export function groupedSoftmax(logw, group) {
  const out = new Float64Array(logw.length);
  for (let g = 0; g < logw.length; g += group) {
    let max = -Infinity;
    for (let i = g; i < g + group; i += 1) max = Math.max(max, logw[i]);
    let sum = 0;
    for (let i = g; i < g + group; i += 1) { out[i] = Math.exp(logw[i] - max); sum += out[i]; }
    for (let i = g; i < g + group; i += 1) out[i] /= sum;
  }
  return out;
}

/** systematic_resample with one uniform per group. */
export function systematicResample(w, group, u) {
  const out = new Int32Array(w.length);
  for (let g = 0; g < w.length / group; g += 1) {
    const cum = [];
    let s = 0;
    for (let i = 0; i < group; i += 1) { s += w[g * group + i]; cum.push(s); }
    for (let i = 0; i < group; i += 1) {
      const pos = (i + u[g]) / group;
      let idx = 0;
      while (idx < group && cum[idx] < pos) idx += 1;               // searchsorted, side='left'
      out[g * group + i] = Math.min(idx, group - 1) + g * group;
    }
  }
  return out;
}

const PARTICLE_SUM = (energiesBySpec, weights) => energiesBySpec.reduce((s, e, i) => s + weights[i] * e, 0);

/**
 * edm_steered_sampler: presets edm_gd (resamplingInterval null) and edm_fk (3).
 * Returns the particles (Float64Array[]) after the last step.
 */
export async function steeredSample({ denoise, tables, mask, particles: P, draws, align, numSteps = 200,
  stepScale = 1.0, noiseScale = 1.0, gamma0 = 0.8, gammaMin = 1.0, resamplingInterval = 3, groupSize = null,
  fkLambda = 4.0, numGdSteps = 20, companion = null, specs = DEFAULT_SPECS, trace = null }) {
  const group = groupSize ?? P;
  const sig = schedule(numSteps);
  const gam = Array.from(sig, (s) => (s > gammaMin ? Math.fround(gamma0) : 0));
  const last = numSteps - 1;
  const n = mask.length;
  let x = draws.x0().map((v) => { const out = new Float64Array(v.length); for (let i = 0; i < v.length; i += 1) out[i] = Math.fround(sig[0] * v[i]); return out; });
  let sgu = Array.from({ length: P }, () => new Float64Array(n * 3));
  let ePrev = new Float64Array(P);
  let hasPrev = false;
  let comp = companion ? companion.map((c) => Float64Array.from(c)) : null;
  let uniformIndex = -1;
  for (let k = 0; k < numSteps; k += 1) {
    const sTm = sig[k];
    const sT = sig[k + 1];
    const g = gam[k + 1];
    const tHat = Math.fround(sTm * (1 + g));
    const noiseVar = noiseScale ** 2 * Math.max(tHat ** 2 - sTm ** 2, 0);
    const noiseStd = Math.fround(Math.sqrt(noiseVar));
    const resample = resamplingInterval !== null && (k % resamplingInterval === 0 || k === last);
    if (resample) uniformIndex += 1;
    const tNorm = 1.0 - (k + 1) / numSteps;
    const prm = stepParameters(tNorm, specs);
    const interval = guidanceInterval(tNorm, specs);
    const { rot, trans, eps } = draws.step(k);
    x = x.map((xp, p) => centreAndRotate(xp, mask, rot[p], trans[p]));
    if (comp) comp = comp.map((cp, p) => centreAndRotate(cp, mask, rot[p], trans[p]));
    sgu = sgu.map((v, p) => rotateOnly(v, rot[p]));
    const noise = eps.map((e) => Float64Array.from(e, (v) => noiseStd * v));
    let noisy = x.map((xp, p) => Float64Array.from(xp, (v, i) => v + noise[p][i]));
    let x0 = await denoise(noisy, tHat, comp);
    const buffers = Object.fromEntries(PARAMS.map((b) => [b, prm[b]]));
    let w = new Float64Array(P).fill(1 / group);
    let energy = ePrev;
    if (resample) {
      energy = new Float64Array(P);
      const un = new Float64Array(P);
      for (let p = 0; p < P; p += 1) {
        energy[p] = PARTICLE_SUM(energies(x0[p], tables, buffers, specs), prm.resampling_weight);
        const logG = hasPrev ? ePrev[p] - energy[p] : -energy[p];
        let ll = 0;
        if (numGdSteps > 0 && noiseVar > 0) {
          for (let i = 0; i < noise[p].length; i += 1) ll += noise[p][i] ** 2 - (noise[p][i] + sgu[p][i]) ** 2;
          ll /= 2 * Math.max(noiseVar, 1e-30);
        }
        const value = ll + fkLambda * logG;
        un[p] = Number.isFinite(value) ? value : -1e4;
      }
      w = groupedSoftmax(un, group);
    }
    if (k < last && numGdSteps > 0) {
      const upd = x0.map((v) => new Float64Array(v.length));
      for (let gs = 0; gs < numGdSteps; gs += 1) {
        const coef = specs.map((_, i) => (prm.guidance_weight[i] !== 0 && gs % interval[i] === 0 ? prm.guidance_weight[i] : 0));
        if (!coef.some((c) => c !== 0)) continue;
        for (let p = 0; p < P; p += 1) {
          const at = Float64Array.from(x0[p], (v, i) => v + upd[p][i]);
          const grad = new Float64Array(at.length);
          energies(at, tables, buffers, specs, grad, coef);
          for (let i = 0; i < grad.length; i += 1) upd[p][i] -= grad[i];
        }
      }
      const factor = (-stepScale * (sT - tHat)) / Math.max(tHat, 1e-12);
      x0 = x0.map((v, p) => Float64Array.from(v, (a, i) => a + upd[p][i]));
      sgu = upd.map((u) => Float64Array.from(u, (v) => v * factor));
    }
    const idx = resample ? systematicResample(w, group, draws.uniform(uniformIndex, P / group)) : Int32Array.from({ length: P }, (_, i) => i);
    trace?.push({ k, resample, w: Array.from(w), idx: Array.from(idx), energy: Array.from(energy) });
    x = Array.from(idx, (i) => x[i]);
    noisy = Array.from(idx, (i) => noisy[i]);
    x0 = Array.from(idx, (i) => x0[i]);
    sgu = Array.from(idx, (i) => sgu[i]);
    energy = Float64Array.from(idx, (i) => energy[i]);
    if (comp) comp = Array.from(idx, (i) => comp[i]);
    hasPrev = hasPrev || resample;
    ePrev = energy;
    const step = (stepScale * (sT - tHat)) / Math.max(tHat, 1e-12);
    x = noisy.map((nz, p) => {
      const aligned = align(nz, x0[p], mask);
      return Float64Array.from(aligned, (v, i) => v + step * (v - x0[p][i]));
    });
  }
  return x;
}

const karrasSigma = (t) => {
  const lo = SIGMA_MIN ** (1 / RHO);
  const hi = SIGMA_MAX ** (1 / RHO);
  return SIGMA_DATA * (lo + t * (hi - lo)) ** RHO;
};
const tFromSigma = (sigma) => {
  const s = Math.max(sigma / SIGMA_DATA, SIGMA_MIN);
  const lo = SIGMA_MIN ** (1 / RHO);
  const hi = SIGMA_MAX ** (1 / RHO);
  return Math.min(Math.max((s ** (1 / RHO) - lo) / (hi - lo), 0), 1);
};
const beta = (sigma, power) => {
  const t = tFromSigma(sigma);
  const sAbs = karrasSigma(t);
  return ((1 + 0.1 * (SIGMA_DATA * SIGMA_MIN) ** 2) * (1 - t) ** power) / (1 + 0.1 * sAbs ** 2);
};

/**
 * edm_fkc_sampler (preset edm_fkc: weight_update 'guidance', reheat 'sigma',
 * ess_threshold 0.9). `denoiseWithGrad(x, sigma, companion, weightsFor)` returns
 * {den, gx, gs, u}: D(x, sigma), and the gradient of sum_p U_hat(D(x_p, sigma))
 * with respect to every x_p and to sigma (one number a particle), where U_hat
 * is the potentials at the step's `hat` parameters. Returns {x, nonFiniteStep}.
 */
export async function fkcSample({ denoiseWithGrad, tables, mask, particles: P, draws, numSteps = 200, groupSize = null,
  essThreshold = 0.9, betaPower = 4.0, companion = null, specs = DEFAULT_SPECS, trace = null }) {
  const group = groupSize ?? P;
  const sig = Array.from(schedule(numSteps));
  const nt = sig.length - 1;
  const n = mask.length;
  let x = draws.x0().map((v) => Float64Array.from(v, (a) => Math.fround(sig[0] * a)));
  let logw = new Float64Array(P);
  let comp = companion ? companion.map((c) => Float64Array.from(c)) : null;
  let bad = -1;
  const total = (den, t) => {
    const prm = stepParameters(t, specs);
    const buffers = Object.fromEntries(PARAMS.map((b) => [b, prm[b]]));
    return den.map((d) => energies(d, tables, buffers, specs).reduce((s, e) => s + e, 0));
  };
  for (let k = 0; k < nt; k += 1) {
    // (every per-step constant reaches the vendor's scan as float32, so it is rounded here as there)
    const f = Math.fround;
    const sC = sig[k];
    const sN = sig[k + 1];
    const sHat = k > 0 ? sig[k - 1] : sC;
    const variance = f(sHat ** 2 - sC ** 2);
    const tC = Math.max(1 - k / nt, 0);
    const tN = Math.max(1 - (k + 1) / nt, 0);
    const tHat = tFromSigma(sHat);
    const { rot, trans, eps } = draws.step(k);
    trace?.x.push(x.map((v) => Array.from(v)));
    x = x.map((xp, p) => centreAndRotate(xp, mask, rot[p], trans[p]));
    if (comp) comp = comp.map((cp, p) => centreAndRotate(cp, mask, rot[p], trans[p]));
    const reheated = x.map((xp, p) => Float64Array.from(xp, (v, i) => v + Math.sqrt(variance) * eps[p][i]));
    const { den, gx, gs, u } = await denoiseWithGrad(reheated, sHat, comp, tHat);
    const eN = total(den, tN);
    const eC = total(den, tC);
    let finite = Number.isFinite(u) && eN.every(Number.isFinite) && eC.every(Number.isFinite);
    const bHat = f(beta(sHat, betaPower));
    const bN = f(beta(sN, betaPower));
    const bC = f(beta(sC, betaPower));
    const next = reheated.map((r, p) => {
      let align = 0;
      const out = new Float64Array(r.length);
      for (let i = 0; i < r.length; i += 1) {
        const score = (den[p][i] - r[i]) / sHat ** 2;
        align += -bHat * gx[p][i] * score * (sC ** 2 - sN ** 2) / 2;
        out[i] = r[i] + sHat * (sHat - sN) * score - 0.5 * variance * bHat * gx[p][i];
      }
      logw[p] = logw[p] - bN * eN[p] + bC * eC[p] + align + bHat * gs[p] * (sC - sN);
      return out;
    });
    finite = finite && logw.every(Number.isFinite);
    if (bad < 0 && !finite) bad = k;
    trace?.logw.push(Array.from(logw));
    const w = groupedSoftmax(logw, group);
    let resample = k === nt - 1;
    for (let g0 = 0; g0 < P; g0 += group) {
      let sq = 0;
      for (let i = g0; i < g0 + group; i += 1) sq += w[i] ** 2;
      if (1 / (group * sq) < essThreshold) resample = true;
    }
    const idx = resample ? systematicResample(w, group, draws.uniform(k, P / group)) : Int32Array.from({ length: P }, (_, i) => i);
    x = Array.from(idx, (i) => next[i]);
    if (comp) comp = Array.from(idx, (i) => comp[i]);
    if (resample) logw = new Float64Array(P);
    void n;
  }
  return { x, nonFiniteStep: bad };
}
