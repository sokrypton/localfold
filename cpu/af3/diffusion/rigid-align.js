// The weighted rigid alignment the samplers that realign take (ESMFold2's, boltz2's): moved here from
// cpu/esmfold2/sampler.js, which re-exports it, so cpu/af3 can read it without an import cycle.
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
  // This routine is py2Dmol's `svd3` (that project's own `io/math.js`) in this file's idiom: the
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
