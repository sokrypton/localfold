// The page's Kabsch (`superimposeCoords3D`), held to Horn's quaternion method, which shares none
// of its code. cuda/rfd4p/gate.py holds the native Kabsch to the page's byte for byte, so a
// mistake the two share passes there: both once flipped the third axis by det(H)'s sign and
// returned a REFLECTION for every reflected point set.
import test from "node:test";
import assert from "node:assert/strict";
import { superimposeCoords3D } from "../shared/rfd4p/output.js";

/** Horn (1987): the best rotation's squared residual, from the largest eigenvalue of a 4x4. */
function hornRmsd(fixed, mobile) {
  const n = fixed.length / 3;
  const centre = (c) => [0, 1, 2].map((k) => {
    let s = 0;
    for (let i = 0; i < n; i += 1) s += c[i * 3 + k];
    return s / n;
  });
  const cf = centre(fixed);
  const cm = centre(mobile);
  const S = Array.from({ length: 3 }, () => [0, 0, 0]);
  let e = 0;
  for (let i = 0; i < n; i += 1) {
    const a = [0, 1, 2].map((k) => mobile[i * 3 + k] - cm[k]);
    const b = [0, 1, 2].map((k) => fixed[i * 3 + k] - cf[k]);
    for (let r = 0; r < 3; r += 1) for (let c = 0; c < 3; c += 1) S[r][c] += a[r] * b[c];
    e += a.reduce((s, v) => s + v * v, 0) + b.reduce((s, v) => s + v * v, 0);
  }
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const N = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ];
  // Cyclic Jacobi on the symmetric 4x4: its largest eigenvalue.
  const a = N.map((row) => row.slice());
  for (let sweep = 0; sweep < 100; sweep += 1) {
    let off = 0;
    for (let p = 0; p < 4; p += 1) for (let q = p + 1; q < 4; q += 1) off += a[p][q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < 4; p += 1) {
      for (let q = p + 1; q < 4; q += 1) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k += 1) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k += 1) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  const lambda = Math.max(a[0][0], a[1][1], a[2][2], a[3][3]);
  return Math.sqrt(Math.max(e - 2 * lambda, 0) / n);
}

const det = (R) => R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6])
  + R[2] * (R[3] * R[7] - R[4] * R[6]);

test("superimposeCoords3D is the best PROPER rotation, reflected sets included", () => {
  let s = 12345;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  let reflectedCases = 0;
  for (const n of [3, 4, 8, 40, 200]) {
    for (const noise of [0, 0.5, 4]) {
      for (const flip of [1, -1]) {
        for (let trial = 0; trial < 4; trial += 1) {
          const fixed = new Float64Array(n * 3);
          const mobile = new Float64Array(n * 3);
          for (let i = 0; i < n * 3; i += 1) fixed[i] = next() * 40 - 20;
          for (let i = 0; i < n; i += 1) {
            const [x, y, z] = [fixed[i * 3], fixed[i * 3 + 1], fixed[i * 3 + 2] * flip];
            mobile[i * 3] = 0.8253 * x - 0.5646 * y + 3.5 + (next() - 0.5) * noise;
            mobile[i * 3 + 1] = 0.5646 * x + 0.8253 * y - 7.25 + (next() - 0.5) * noise;
            mobile[i * 3 + 2] = z + 11 + (next() - 0.5) * noise;
          }
          const fit = superimposeCoords3D(fixed, mobile);
          assert.ok(Math.abs(det(fit.rotation) - 1) < 1e-9, `det ${det(fit.rotation)} at n=${n} flip=${flip}`);
          const want = hornRmsd(fixed, mobile);
          assert.ok(Math.abs(fit.rmsd - want) <= 1e-7 * Math.max(1, want),
            `rmsd ${fit.rmsd} against Horn's ${want} at n=${n} noise=${noise} flip=${flip}`);
          if (flip < 0) reflectedCases += 1;
        }
      }
    }
  }
  assert.ok(reflectedCases > 0);
});
