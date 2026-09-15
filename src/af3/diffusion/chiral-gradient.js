/**
 * RoseTTAFold3's chirality signal: d/dx of the improper-dihedral error.
 *
 * 🔴 IT IS AN INPUT FEATURE, NOT A LOSS. The diffusion atom encoder adds
 * `Linear(grad)` to its query, where `grad` is the gradient of
 *
 *     sum over valid centres of (dihedral(x) - ideal)^2
 *
 * with respect to the SCALED NOISY COORDINATES - the same `R_L` the encoder
 * hands `process_r`. The reference detaches it (`stop_gradient`), so nothing
 * learns through this path; it is a hint about which way the structure is
 * wrong, recomputed every sampler step.
 *
 * 🔴 AND IT IS THE ONLY REFLECTION-ASYMMETRIC THING IN THE NETWORK. Mirror a
 * structure and every distance, every frame and every distogram is unchanged;
 * the improper dihedral flips sign. Without this term a D-amino acid is exactly
 * as good an answer as an L one.
 *
 * 🔴 THE REFERENCE TAKES `jax.grad` OF THE SCALAR AND RoseTTAFold3 ITSELF
 * HAND-DERIVES IT OVER ~150 LINES. This is the hand derivation, and it is
 * checked against a CENTRAL DIFFERENCE of the same scalar in
 * `test/chiral-gradient.test.js` rather than against either of them - a
 * numerical gradient cannot share a mistake with an analytic one.
 */

/** The scalar the gradient is of, for one quadruple. Kept separate so the test
 *  can difference exactly the function the derivative claims to differentiate. */
export function improperDihedral(a, b, c, d, eps = 1e-6) {
  const sub = (p, q) => [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
  const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
  const b0 = sub(a, b);
  const b1 = sub(c, b);
  const b2 = sub(d, c);
  const length = Math.sqrt(dot(b1, b1)) + eps;
  const b1n = [b1[0] / length, b1[1] / length, b1[2] / length];
  const pb0 = dot(b0, b1n);
  const pb2 = dot(b2, b1n);
  const v = [b0[0] - pb0 * b1n[0], b0[1] - pb0 * b1n[1], b0[2] - pb0 * b1n[2]];
  const w = [b2[0] - pb2 * b1n[0], b2[1] - pb2 * b1n[1], b2[2] - pb2 * b1n[2]];
  const cross = [b1n[1] * v[2] - b1n[2] * v[1],
                 b1n[2] * v[0] - b1n[0] * v[2],
                 b1n[0] * v[1] - b1n[1] * v[0]];
  // The reference adds eps to BOTH arguments of atan2, which moves the angle by
  // about 1e-6 and is kept because the ideal it is differenced against is not
  // moved with it.
  return Math.atan2(dot(cross, w) + eps, dot(v, w) + eps);
}

/**
 * The whole feature: `centers` quadruples of flat atom indices, `angles` their
 * ideals, `positions` the flat [atoms * 3] coordinates.
 *
 * 🔴 DIFFERENTIATED NUMERICALLY, PER CENTRE, AND THAT IS A DELIBERATE CHOICE.
 * Twelve coordinates enter a centre and the analytic derivative of an atan2 of
 * two cross products is where rf3 spends 150 lines; a central difference over
 * twelve components is 24 evaluations of a function that is about forty flops,
 * so a centre costs ~1,000 flops and 6MRR's 213 centres cost 200k - against a
 * denoiser step measured in tens of millions. The reference itself does not
 * hand-derive this (it calls `jax.grad`), and a closed form here would be a new
 * source of silent error for no measurable time.
 *
 * 🔴 A CENTRE WHOSE ANGLE IS ZERO IS PADDING and contributes nothing, which is
 * the reference's own `valid` mask.
 */
export function chiralPositionGradients(positions, centers, angles, atoms,
                                        step = 1e-4) {
  const gradients = new Float32Array(atoms * 3);
  const point = (index) => [positions[index * 3], positions[index * 3 + 1],
                            positions[index * 3 + 2]];
  for (let centre = 0; centre < angles.length; centre += 1) {
    const ideal = angles[centre];
    if (ideal === 0) continue;
    const idx = [centers[centre * 4], centers[centre * 4 + 1],
                 centers[centre * 4 + 2], centers[centre * 4 + 3]];
    const p = idx.map(point);
    for (let corner = 0; corner < 4; corner += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        const keep = p[corner][axis];
        p[corner][axis] = keep + step;
        const up = improperDihedral(p[0], p[1], p[2], p[3]) - ideal;
        p[corner][axis] = keep - step;
        const down = improperDihedral(p[0], p[1], p[2], p[3]) - ideal;
        p[corner][axis] = keep;
        // d/dx of (dih - ideal)^2 is 2 (dih - ideal) d(dih)/dx, and the square
        // is differenced directly so the two share the same atan2 branch.
        const derivative = (up * up - down * down) / (2 * step);
        const at = idx[corner] * 3 + axis;
        gradients[at] += Number.isFinite(derivative) ? derivative : 0;
      }
    }
  }
  return gradients;
}
