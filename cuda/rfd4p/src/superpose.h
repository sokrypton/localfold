// `superimposeCoords3D` and `align_atom_array_to_input_frame`: putting a generated structure
// back into the input's own frame.
//
// 🔴 WITHOUT IT A MOTIF DESIGN IS IN THE MODEL'S CENTRED FRAME, NOT THE TARGET'S. The sampler
// works on centred coordinates, so the output is a correct structure in the wrong place - it
// superposes on nothing, and no bond length, clash score or chain check can see it. The anchors
// are the atoms whose coordinates were CONDITIONED, because those are the ones whose position in
// the input frame is known.
//
// 🔴 AND IT IS INVISIBLE ON EVERY SHIPPED EXAMPLE. `web/examples/*.cif.zst` are already centred
// (chain B of `ppi_pdl1` has centroid -0.19, 0.13, -0.53), so the two frames coincide by
// accident and the whole transform is a no-op. A structure somebody deposits is not centred.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace rfd4p {
namespace superpose {

using std::vector;

struct Fit {
  double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double fixedCenter[3] = {0, 0, 0};
  double mobileCenter[3] = {0, 0, 0};
  bool ok = false;
};

/**
 * `jacobiEigen3x3`, exactly as `output.js` has it.
 *
 * Three things here are the reference's and not the textbook's, and all three put together
 * disagree on 125 of the gate's 240 point sets - which is where this comment's old claim that
 * "the textbook version disagrees on more than half" came from. Mutated ONE AT A TIME against
 * the same sweep, they are nothing like each other:
 *
 * 🔴 **FORCING THE EIGENVECTOR MATRIX RIGHT-HANDED IS THE ONE THAT MATTERS: 114 of 240**, at
 * every point count, a rotation element moving by up to 1.94. Negating the third column when
 * det(V) < 0 is not a tidy-up.
 *
 * **The partial rotation update is 29 of 240 AND ALL 29 ARE n = 2.** Writing it as two full
 * matrix products is the same arithmetic in a different order, and the only inputs it moves are
 * the rank-deficient ones - two centred points are collinear, so the rotation about that line is
 * not determined by the data and either answer is as good. It must still be ported exactly,
 * because the two sides are held to the same bytes; it is not a convention that decides a
 * structure.
 *
 * 🔴 **AND CLAMPING A NEGATIVE EIGENVALUE TO ZERO IS INERT - 0 of 240, and so is removing the
 * sqrt-site guard beside it, and so is removing BOTH.** Not because it never happens: a negative
 * eigenvalue comes up 41 times in that sweep. It lands in slot 2 thirty-seven times, whose vector
 * `superimpose` never reads because U's third column is a CROSS PRODUCT - the two conventions are
 * connected, and the second is what kills the first - and in slot 1 four times, where the clamped
 * 0 and the unclamped NaN both fail `sigma > 1e-9` and take the same complement branch. Kept
 * because the reference has it, written down as measured rather than as a 🔴 nobody can falsify.
 */
inline void jacobiEigen(const double in[9], double values[3], double vectors[9]) {
  double a[9];
  for (int i = 0; i < 9; ++i) a[i] = in[i];
  double v[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  for (int sweep = 0; sweep < 25; ++sweep) {
    double maxOff = 0;
    for (const auto& pq : pairs) maxOff = std::max(maxOff, std::fabs(a[pq[0] * 3 + pq[1]]));
    if (maxOff < 1e-13) break;
    for (const auto& pq : pairs) {
      const int p = pq[0], q = pq[1];
      const double apq = a[p * 3 + q];
      if (std::fabs(apq) < 1e-13) continue;
      const double app = a[p * 3 + p], aqq = a[q * 3 + q];
      const double tau = (aqq - app) / (2 * apq);
      const double t = tau >= 0 ? 1 / (tau + std::sqrt(1 + tau * tau))
                                : -1 / (-tau + std::sqrt(1 + tau * tau));
      const double c = 1 / std::sqrt(1 + t * t), s = t * c;
      a[p * 3 + p] = app - t * apq;
      a[q * 3 + q] = aqq + t * apq;
      a[p * 3 + q] = 0;
      a[q * 3 + p] = 0;
      const int r = 3 - p - q;
      const double arp = a[r * 3 + p], arq = a[r * 3 + q];
      a[r * 3 + p] = c * arp - s * arq;
      a[p * 3 + r] = a[r * 3 + p];
      a[r * 3 + q] = s * arp + c * arq;
      a[q * 3 + r] = a[r * 3 + q];
      for (int k = 0; k < 3; ++k) {
        const double vkp = v[k * 3 + p], vkq = v[k * 3 + q];
        v[k * 3 + p] = c * vkp - s * vkq;
        v[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int i, int j) { return a[j * 3 + j] < a[i * 3 + i]; });
  for (int col = 0; col < 3; ++col) {
    const int src = order[col];
    values[col] = std::max(0.0, a[src * 3 + src]);
    for (int row = 0; row < 3; ++row) vectors[row * 3 + col] = v[row * 3 + src];
  }
  const double d = vectors[0] * (vectors[4] * vectors[8] - vectors[5] * vectors[7]) -
                   vectors[1] * (vectors[3] * vectors[8] - vectors[5] * vectors[6]) +
                   vectors[2] * (vectors[3] * vectors[7] - vectors[4] * vectors[6]);
  if (d < 0) for (int row = 0; row < 3; ++row) vectors[row * 3 + 2] = -vectors[row * 3 + 2];
}

inline double det3(const double m[9]) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
         m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/**
 * `superimposeCoords3D`: the rotation and the two centroids that put `mobile` onto `fixed`.
 *
 * 🔴 THE THIRD COLUMN OF U IS A CROSS PRODUCT, NOT A THIRD SINGULAR VECTOR, and the reflection
 * is fixed by the SIGN OF det(H) applied to V's third column. Building all three from the
 * eigendecomposition and then "fixing the determinant" is the usual recipe and is NOT this one:
 * where the smallest singular value is near zero its vector is numerical noise, and the cross
 * product is defined there while the eigenvector is not.
 *
 * Both halves measured one at a time on the gate's 240 point sets, at every point count from 2
 * to 200: the third singular vector in place of the cross product is **99 differing**, and
 * taking det(H)'s sign as always positive is **90**, each moving a rotation element by up to
 * 1.92. These two are load-bearing where two of jacobiEigen's three are not.
 */
inline Fit superimpose(const vector<double>& fixed, const vector<double>& mobile) {
  Fit out;
  const size_t n = std::min(fixed.size(), mobile.size()) / 3;
  if (n == 0) return out;
  for (size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      out.fixedCenter[k] += fixed[i * 3 + k];
      out.mobileCenter[k] += mobile[i * 3 + k];
    }
  }
  for (int k = 0; k < 3; ++k) { out.fixedCenter[k] /= (double)n; out.mobileCenter[k] /= (double)n; }

  double h[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    const double m[3] = {mobile[i * 3] - out.mobileCenter[0], mobile[i * 3 + 1] - out.mobileCenter[1],
                         mobile[i * 3 + 2] - out.mobileCenter[2]};
    const double f[3] = {fixed[i * 3] - out.fixedCenter[0], fixed[i * 3 + 1] - out.fixedCenter[1],
                         fixed[i * 3 + 2] - out.fixedCenter[2]};
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) h[r * 3 + c] += m[r] * f[c];
  }

  double hth[9];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      hth[r * 3 + c] = h[0 * 3 + r] * h[0 * 3 + c] + h[1 * 3 + r] * h[1 * 3 + c] +
                       h[2 * 3 + r] * h[2 * 3 + c];
    }
  }
  double values[3], V[9];
  jacobiEigen(hth, values, V);

  double U[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int col = 0; col < 2; ++col) {
    const double sigma = std::sqrt(values[col] > 0 ? values[col] : 0.0);
    if (sigma > 1e-9) {
      for (int row = 0; row < 3; ++row) {
        U[row * 3 + col] = (h[row * 3 + 0] * V[0 * 3 + col] + h[row * 3 + 1] * V[1 * 3 + col] +
                            h[row * 3 + 2] * V[2 * 3 + col]) / sigma;
      }
    } else if (col == 0) {
      U[0] = 1;
    } else {
      const double u0x = U[0], u0y = U[3], u0z = U[6];
      double vx = std::fabs(u0x) < 0.9 ? 1.0 : 0.0;
      double vy = std::fabs(u0x) < 0.9 ? 0.0 : 1.0;
      double vz = 0.0;
      const double dot = vx * u0x + vy * u0y + vz * u0z;
      vx -= dot * u0x; vy -= dot * u0y; vz -= dot * u0z;
      double norm = std::sqrt(vx * vx + vy * vy + vz * vz);
      if (norm == 0) norm = 1;
      U[1] = vx / norm; U[4] = vy / norm; U[7] = vz / norm;
    }
  }
  U[0 * 3 + 2] = U[1 * 3 + 0] * U[2 * 3 + 1] - U[2 * 3 + 0] * U[1 * 3 + 1];
  U[1 * 3 + 2] = U[2 * 3 + 0] * U[0 * 3 + 1] - U[0 * 3 + 0] * U[2 * 3 + 1];
  U[2 * 3 + 2] = U[0 * 3 + 0] * U[1 * 3 + 1] - U[1 * 3 + 0] * U[0 * 3 + 1];

  const double sign = det3(h) >= 0 ? 1.0 : -1.0;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out.R[r * 3 + c] = V[r * 3 + 0] * U[c * 3 + 0] + V[r * 3 + 1] * U[c * 3 + 1] +
                         sign * V[r * 3 + 2] * U[c * 3 + 2];
    }
  }
  out.ok = true;
  return out;
}

}  // namespace superpose
}  // namespace rfd4p
