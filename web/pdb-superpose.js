/**
 * One PDB's coordinates superposed onto another's of the same atoms, the text otherwise untouched.
 *
 * 🔴 A NATIVE FOLD'S RESULT IS NOT IN ITS FRAMES' ORIENTATION. The native ports stream each sampler step superposed onto
 * the first one written (the prediction settles in place rather than tumbling with the walk's random rotations), and
 * then hand back the structure in the model's own frame - the same atoms as the last frame to the hundredth of an
 * angstrom once fitted, and 4-60 A away from it as drawn. Put at the end of the streamed trajectory as it came, the
 * picture jumped on its last frame. The page draws the result moved onto that frame; the file it downloads is the
 * model's own, as the command line writes it.
 *
 * Horn's closed form (the rotation is the top eigenvector of a 4x4 quaternion matrix), over every ATOM/HETATM record
 * in order. Two texts with different atom counts come back unmoved.
 */

const isAtom = (line) => line.startsWith("ATOM") || line.startsWith("HETATM");
const coordinates = (lines) => lines.filter(isAtom).map((line) => [
  Number.parseFloat(line.slice(30, 38)), Number.parseFloat(line.slice(38, 46)), Number.parseFloat(line.slice(46, 54))]);

// the eigenvector of the largest eigenvalue of a symmetric 4x4 (cyclic Jacobi)
function topEigenvector(a) {
  const m = a.map((row) => row.slice());
  const v = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep += 1) {
    let off = 0;
    for (let p = 0; p < 4; p += 1) for (let q = p + 1; q < 4; q += 1) off += m[p][q] * m[p][q];
    if (off < 1e-24) break;
    for (let p = 0; p < 4; p += 1) {
      for (let q = p + 1; q < 4; q += 1) {
        if (Math.abs(m[p][q]) < 1e-300) continue;
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k += 1) {
          const kp = m[k][p];
          const kq = m[k][q];
          m[k][p] = c * kp - s * kq;
          m[k][q] = s * kp + c * kq;
        }
        for (let k = 0; k < 4; k += 1) {
          const pk = m[p][k];
          const qk = m[q][k];
          m[p][k] = c * pk - s * qk;
          m[q][k] = s * pk + c * qk;
        }
        for (let k = 0; k < 4; k += 1) {
          const kp = v[k][p];
          const kq = v[k][q];
          v[k][p] = c * kp - s * kq;
          v[k][q] = s * kp + c * kq;
        }
      }
    }
  }
  let best = 0;
  for (let k = 1; k < 4; k += 1) if (m[k][k] > m[best][best]) best = k;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}

/** `text`'s atoms moved onto `reference`'s (the same atoms, in the same order). */
export function superposePdbOnto(text, reference) {
  const lines = text.split("\n");
  const moving = coordinates(lines);
  const fixed = coordinates(reference.split("\n"));
  if (moving.length < 3 || moving.length !== fixed.length) return text;
  const centre = (points) => points.reduce((sum, p) => sum.map((value, k) => value + p[k] / points.length), [0, 0, 0]);
  const cm = centre(moving);
  const cf = centre(fixed);
  const s = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];           // S[i][j] = sum (moving - cm)_i (fixed - cf)_j
  for (let a = 0; a < moving.length; a += 1) {
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) s[i][j] += (moving[a][i] - cm[i]) * (fixed[a][j] - cf[j]);
    }
  }
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = s;
  const [w, x, y, z] = topEigenvector([
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ]);
  const r = [
    [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
  ];
  let a = 0;
  return lines.map((line) => {
    if (!isAtom(line)) return line;
    const p = moving[a].map((value, k) => value - cm[k]);
    a += 1;
    const moved = r.map((row, k) => row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + cf[k]);
    return line.slice(0, 30) + moved.map((value) => value.toFixed(3).padStart(8)).join("") + line.slice(54);
  }).join("\n");
}
