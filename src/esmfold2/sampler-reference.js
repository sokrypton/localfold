/**
 * ESMFold2's EDM sampler: a denoiser, a noise schedule, and a structure.
 *
 * 🔴 IT IS STOCHASTIC, SO A PORT CANNOT REPRODUCE ITS COORDINATES. Every step
 * centres `x`, rotates it by a random rotation, translates it by a random
 * vector and adds Gaussian noise - four draws from torch's global RNG per step,
 * eleven steps. Nothing in JavaScript reproduces that stream, so the honest
 * gate is not "the same structure" but the DETERMINISTIC parts, each exactly:
 * the schedule, the gammas, the Kabsch alignment and the update algebra, with
 * the draws taken from the model rather than made again.
 *
 * 🔴 AND THE RANDOM AUGMENTATION IS NOT COSMETIC. It is AF3's Algorithm 19 and
 * the model was trained with it: the denoiser sees a differently oriented copy
 * of the structure at every step, which is what makes its answer rotation
 * equivariant in practice rather than in principle. Dropping it makes a sampler
 * that runs and folds worse.
 */

/**
 * The Karras power-law schedule, with a zero appended.
 *
 * 🔴 AND `max_inference_sigma` TRUNCATES IT AND THEN PREPENDS ITSELF, so asking
 * for fifteen steps runs ELEVEN. Entries above the cap are dropped and the cap
 * becomes the new first entry - at the shipped 256 that removes four of the
 * sixteen. A port that runs the requested count is running a different
 * schedule; the number of steps is an OUTPUT of this function, not an input.
 */
export function noiseSchedule({ steps, sMax, sMin, p, sigmaData, maxSigma }) {
  const values = [];
  if (steps === 1) {
    values.push(sMax * sigmaData, 0);
  } else {
    const inverse = 1 / p;
    const high = Math.pow(sMax, inverse);
    const low = Math.pow(sMin, inverse);
    for (let k = 0; k < steps; k += 1) {
      values.push(sigmaData * Math.pow(high + (k / (steps - 1)) * (low - high), p));
    }
    values.push(0);
  }
  if (maxSigma === undefined || maxSigma === null) return Float32Array.from(values);
  const kept = values.filter((value) => value <= maxSigma);
  return Float32Array.from([maxSigma, ...kept]);
}

/**
 * The churn per entry.
 *
 * 🔴 STEP i TAKES `gammas[i + 1]`, NOT `gammas[i]`. Upstream zips
 * `schedule[:-1]` with `schedule[1:]` and `gammas[1:]`, so the churn applied to
 * `sigma_tm` is decided by the NEXT noise level. Off by one it still runs, and
 * the first step's t_hat is the tell: 256 x 1.605 = 410.88 is what the model
 * records, and 256 alone is what the wrong indexing gives.
 */
export function churnFactors(schedule, gammaMin, gamma0) {
  const out = new Float32Array(schedule.length);
  for (let i = 0; i < schedule.length; i += 1) {
    out[i] = schedule[i] > gammaMin ? gamma0 : 0;
  }
  return out;
}

/** The noise level the denoiser is actually asked about, per step. */
export function noiseLevels(schedule, gammas) {
  const out = [];
  for (let i = 0; i + 1 < schedule.length; i += 1) {
    out.push(schedule[i] * (1 + gammas[i + 1]));
  }
  return out;
}

/** Symmetric 3x3 eigendecomposition by cyclic Jacobi. -> {values, vectors} */
function symmetricEigen(matrix) {
  const a = Float64Array.from(matrix);
  const v = Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let sweep = 0; sweep < 24; sweep += 1) {
    let off = 0;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) off += a[p * 3 + q] * a[p * 3 + q];
    if (off < 1e-30) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = a[p * 3 + q];
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (a[q * 3 + q] - a[p * 3 + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k += 1) {
        const akp = a[k * 3 + p], akq = a[k * 3 + q];
        a[k * 3 + p] = c * akp - s * akq;
        a[k * 3 + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k += 1) {
        const apk = a[p * 3 + k], aqk = a[q * 3 + k];
        a[p * 3 + k] = c * apk - s * aqk;
        a[q * 3 + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k += 1) {
        const vkp = v[k * 3 + p], vkq = v[k * 3 + q];
        v[k * 3 + p] = c * vkp - s * vkq;
        v[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }
  return { values: [a[0], a[4], a[8]], vectors: v };
}

/**
 * The rotation that best takes `x` onto `target`, weighted.
 *
 * 🔴 THE SVD IS OF A 3x3 AND IS DONE THROUGH `H^T H`, which squares the
 * condition number and is fine here and nowhere else: three singular values of
 * a well-scaled covariance, and the answer is a ROTATION rather than a
 * reconstruction. The determinant correction is what stops it returning a
 * reflection - without it a mirrored structure scores perfectly and folds
 * inside out.
 */
export function weightedRigidAlign(x, target, weights, atoms) {
  let total = 0;
  const centre = [0, 0, 0];
  const centreTarget = [0, 0, 0];
  for (let atom = 0; atom < atoms; atom += 1) {
    const w = weights[atom];
    total += w;
    for (let axis = 0; axis < 3; axis += 1) {
      centre[axis] += w * x[atom * 3 + axis];
      centreTarget[axis] += w * target[atom * 3 + axis];
    }
  }
  const by = 1 / Math.max(total, 1e-8);
  for (let axis = 0; axis < 3; axis += 1) {
    centre[axis] *= by;
    centreTarget[axis] *= by;
  }

  // H = sum w * (target - mu_target) (x - mu_x)^T
  const h = new Float64Array(9);
  for (let atom = 0; atom < atoms; atom += 1) {
    const w = weights[atom];
    if (w === 0) continue;
    for (let i = 0; i < 3; i += 1) {
      const a = target[atom * 3 + i] - centreTarget[i];
      for (let j = 0; j < 3; j += 1) {
        h[i * 3 + j] += w * a * (x[atom * 3 + j] - centre[j]);
      }
    }
  }

  // H = U S V^T, from the eigenvectors of H^T H (= V) and H V = U S.
  const hth = new Float64Array(9);
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      let sum = 0;
      for (let k = 0; k < 3; k += 1) sum += h[k * 3 + i] * h[k * 3 + j];
      hth[i * 3 + j] = sum;
    }
  }
  const { values, vectors } = symmetricEigen(hth);
  const order = [0, 1, 2].sort((a, b) => values[b] - values[a]);
  const vv = new Float64Array(9);
  const uu = new Float64Array(9);
  const sigmas = [];
  for (let column = 0; column < 3; column += 1) {
    const from = order[column];
    for (let row = 0; row < 3; row += 1) vv[row * 3 + column] = vectors[row * 3 + from];
    sigmas.push(Math.sqrt(Math.max(values[from], 0)));
  }

  // 🔴 A SINGULAR VALUE IS ZERO WHEN ITS EIGENVALUE IS, AND TESTING THE ROOT
  // INSTEAD IS THE BUG py2Dmol's `svd3` RECORDS HAVING BEEN BITTEN BY. The
  // square root HALVES the exponent, so a numerically-zero eigenvalue of 8e-15
  // against 196 comes out as a singular value of 9e-8 - which clears any
  // sensible absolute floor, is then divided by, and leaves that column of U
  // not merely wrong but non-orthonormal. The floor belongs in the space where
  // the noise actually lives, and it is RELATIVE to the largest eigenvalue,
  // because "small" means nothing on its own when a structure's coordinates can
  // be 1 or 1e4.
  //
  // This routine is py2Dmol's `svd3` (src/io/math.js) in this file's idiom: the
  // same Jacobi-on-H^T-H, the same eigenvalue floor, the same unit-vector
  // verification and orthonormal completion. A flat or linear point cloud is
  // exactly what a structure viewer meets constantly and what a diffusion
  // sampler's early steps can produce.
  const floor = 1e-12 * Math.abs(values[order[0]] || 1);
  const good = [];
  for (let column = 0; column < 3; column += 1) {
    if (values[order[column]] <= floor) { sigmas[column] = 0; continue; }
    for (let row = 0; row < 3; row += 1) {
      let sum = 0;
      for (let k = 0; k < 3; k += 1) sum += h[row * 3 + k] * vv[k * 3 + column];
      uu[row * 3 + column] = sum / sigmas[column];
    }
    // ...and if the division did not give a unit vector it was not a real
    // direction after all. Cheap, and it does not trust the floor.
    const length = Math.hypot(uu[column], uu[3 + column], uu[6 + column]);
    if (Math.abs(length - 1) > 1e-6) {
      uu[column] = uu[3 + column] = uu[6 + column] = 0;
      sigmas[column] = 0;
    } else {
      good.push(column);
    }
  }
  // Any column the division could not give, completed orthonormally against
  // the ones it could - taking whichever axis has the most left after the
  // projection, so the completion is never near-degenerate itself.
  for (let column = 0; column < 3; column += 1) {
    if (good.includes(column)) continue;
    let best = null;
    let bestLength = -1;
    for (const seed of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      const w = seed.slice();
      for (const k of good) {
        const dot = w[0] * uu[k] + w[1] * uu[3 + k] + w[2] * uu[6 + k];
        w[0] -= dot * uu[k];
        w[1] -= dot * uu[3 + k];
        w[2] -= dot * uu[6 + k];
      }
      const length = Math.hypot(w[0], w[1], w[2]);
      if (length > bestLength) { bestLength = length; best = w; }
    }
    for (let row = 0; row < 3; row += 1) uu[row * 3 + column] = best[row] / bestLength;
    good.push(column);
  }

  // R = U diag(1, 1, det(U V^T)) V^T.
  const uvt = new Float64Array(9);
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      let sum = 0;
      for (let k = 0; k < 3; k += 1) sum += uu[i * 3 + k] * vv[j * 3 + k];
      uvt[i * 3 + j] = sum;
    }
  }
  const determinant =
    uvt[0] * (uvt[4] * uvt[8] - uvt[5] * uvt[7])
    - uvt[1] * (uvt[3] * uvt[8] - uvt[5] * uvt[6])
    + uvt[2] * (uvt[3] * uvt[7] - uvt[4] * uvt[6]);
  const sign = determinant < 0 ? -1 : 1;
  const r = new Float64Array(9);
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      let sum = 0;
      for (let k = 0; k < 3; k += 1) {
        sum += uu[i * 3 + k] * (k === 2 ? sign : 1) * vv[j * 3 + k];
      }
      r[i * 3 + j] = sum;
    }
  }

  // x_c @ R^T + mu_target.
  const out = new Float32Array(x.length);
  for (let atom = 0; atom < atoms; atom += 1) {
    for (let i = 0; i < 3; i += 1) {
      let sum = 0;
      for (let j = 0; j < 3; j += 1) sum += (x[atom * 3 + j] - centre[j]) * r[i * 3 + j];
      out[atom * 3 + i] = sum + centreTarget[i];
    }
  }
  return out;
}

/**
 * One sampler step, after the denoiser has answered.
 *
 *     x_noisy = align(x_noisy, x_denoised)
 *     x       = x_noisy + eta * (sigma_t - t_hat) * (x_noisy - x_denoised) / t_hat
 *
 * 🔴 THE ALIGNMENT COMES FIRST AND IT MOVES THE NOISY COPY, NOT THE ANSWER. The
 * denoiser's output is the reference; the noisy input is rotated onto it, and
 * the step is then taken between two structures in one frame. Aligning the
 * other way round is the same Kabsch call with its arguments swapped and it
 * walks the structure away from the answer.
 */
export function samplerStep(noisy, denoised, weights, atoms, tHat, sigmaT, stepScale) {
  const aligned = weightedRigidAlign(noisy, denoised, weights, atoms);
  const out = new Float32Array(aligned.length);
  const factor = stepScale * (sigmaT - tHat) / tHat;
  for (let i = 0; i < out.length; i += 1) {
    out[i] = aligned[i] + factor * (aligned[i] - denoised[i]);
  }
  return out;
}
