// The sampler's rigid alignment, on the shapes that break a naive 3x3 SVD.
//
// 🔴 A DEGENERATE POINT CLOUD IS NOT A CORNER CASE HERE. The alignment runs
// once per sampler step against a structure that starts as pure noise and ends
// folded, and a flat or linear arrangement is exactly what py2Dmol's `svd3`
// records meeting constantly in a viewer. The routine in
// src/esmfold2/sampler-reference.js is that one's algorithm - the eigenvalue
// floor, the unit-vector check, the orthonormal completion - and this pins the
// behaviour those exist for, so a later "simplification" back to a bare
// H^T H / sqrt / divide has something to fail.
import { describe, expect, it } from "./harness.js";
import { weightedRigidAlign } from "../src/esmfold2/sampler-reference.js";

/** Rotate `points` by the rotation taking x->y, y->-x (90 degrees about z). */
function quarterTurn(points, atoms) {
  const out = new Float32Array(points.length);
  for (let a = 0; a < atoms; a += 1) {
    out[a * 3] = -points[a * 3 + 1];
    out[a * 3 + 1] = points[a * 3];
    out[a * 3 + 2] = points[a * 3 + 2];
  }
  return out;
}

const worstError = (got, want) => {
  let worst = 0;
  for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(got[i] - want[i]));
  return worst;
};

describe("the sampler's rigid alignment", () => {
  const atoms = 8;
  const ones = new Float32Array(atoms).fill(1);

  it("recovers a rotation from a generic cloud", () => {
    const x = new Float32Array(atoms * 3);
    for (let a = 0; a < atoms; a += 1) {
      x[a * 3] = Math.cos(a);
      x[a * 3 + 1] = Math.sin(a);
      x[a * 3 + 2] = a * 0.37;
    }
    const target = quarterTurn(x, atoms);
    expect(worstError(weightedRigidAlign(x, target, ones, atoms), target) < 1e-5).toBe(true);
  });

  it("recovers a rotation from a FLAT cloud, where one singular value is zero", () => {
    const x = new Float32Array(atoms * 3);
    for (let a = 0; a < atoms; a += 1) {
      x[a * 3] = Math.cos(a);
      x[a * 3 + 1] = Math.sin(a);
    }
    const target = quarterTurn(x, atoms);
    expect(worstError(weightedRigidAlign(x, target, ones, atoms), target) < 1e-5).toBe(true);
  });

  it("recovers a rotation from a LINEAR cloud, where two are", () => {
    const x = new Float32Array(atoms * 3);
    const target = new Float32Array(atoms * 3);
    for (let a = 0; a < atoms; a += 1) {
      x[a * 3] = a;
      target[a * 3 + 1] = a;
    }
    expect(worstError(weightedRigidAlign(x, target, ones, atoms), target) < 1e-5).toBe(true);
  });

  it("holds at every scale, because the floor is relative", () => {
    // 🔴 THE POINT OF THE RELATIVE FLOOR. At coordinates of 1e4 the covariance's
    // eigenvalues are ~1e16 and its numerical zero is ~1, whose square root is
    // 1 - past any absolute threshold anyone would write.
    for (const scale of [1e-3, 1, 1e2, 1e4]) {
      const x = new Float32Array(atoms * 3);
      for (let a = 0; a < atoms; a += 1) {
        x[a * 3] = scale * Math.cos(a);
        x[a * 3 + 1] = scale * Math.sin(a);
      }
      const target = quarterTurn(x, atoms);
      const relative = worstError(weightedRigidAlign(x, target, ones, atoms), target) / scale;
      expect(relative < 1e-5).toBe(true);
    }
  });

  it("never returns a reflection", () => {
    // The determinant correction. Without it a mirrored structure scores
    // perfectly and the fold comes out inside out.
    const x = new Float32Array(atoms * 3);
    const target = new Float32Array(atoms * 3);
    for (let a = 0; a < atoms; a += 1) {
      x[a * 3] = Math.cos(a);
      x[a * 3 + 1] = Math.sin(a);
      x[a * 3 + 2] = a * 0.21;
      target[a * 3] = Math.cos(a);
      target[a * 3 + 1] = Math.sin(a);
      target[a * 3 + 2] = -a * 0.21;             // mirrored in z
    }
    const aligned = weightedRigidAlign(x, target, ones, atoms);
    // A rotation cannot reach the mirror image, so the fit must be imperfect -
    // a routine that allowed a reflection would land on it exactly.
    expect(worstError(aligned, target) > 1e-3).toBe(true);
    // ...and it must still be a rigid motion: pairwise distances preserved.
    const distance = (p, i, j) => Math.hypot(
      p[i * 3] - p[j * 3], p[i * 3 + 1] - p[j * 3 + 1], p[i * 3 + 2] - p[j * 3 + 2]);
    let worst = 0;
    for (let i = 0; i < atoms; i += 1) {
      for (let j = i + 1; j < atoms; j += 1) {
        worst = Math.max(worst, Math.abs(distance(aligned, i, j) - distance(x, i, j)));
      }
    }
    expect(worst < 1e-5).toBe(true);
  });
});
