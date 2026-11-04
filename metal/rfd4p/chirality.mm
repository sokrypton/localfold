// `align_chirality`: the one sequence feature that is recomputed every step rather than replayed.
//
// A chiral centre's packed vectors are the CLEAN structure's, and what the model is shown is those vectors rotated
// onto the directions its neighbours have in the CURRENT noisy state - so a small molecule's handedness follows the
// trajectory instead of describing a structure that no longer exists. `shared/rfd4p/features.js`'s `alignChirality`
// is the spec and `cuda/rfd4p/src/chirality.cuh` is the other port's reading.
//
// 🔴 ON THE HOST, AND THAT IS NOT A COMPROMISE. A frame is two origin atoms and up to six neighbours, and a job
// has a handful of frames: the whole thing is a few hundred flops a step against a trunk pass of billions. What
// it must not be is approximate - an eigenvector of a 4x4, which is where the proper rotation comes from, and a
// float32 Jacobi sweep would put a rotation's third decimal in play.
#include "rfd4p.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace {

// the symmetric 4x4's eigenvalues and eigenvectors, cyclic Jacobi - `jacobiEigen4`
void jacobiEigen4(double A[4][4], double values[4], double V[4][4]) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) V[i][j] = i == j ? 1.0 : 0.0;
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = 0;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) off += A[p][q] * A[p][q];
    if (off < 1e-30) break;
    for (int p = 0; p < 4; ++p) {
      for (int q = p + 1; q < 4; ++q) {
        if (std::abs(A[p][q]) < 1e-300) continue;
        const double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        // 🔴 `Math.sign(theta || 1)`: zero takes the POSITIVE branch, which is what the reference's `|| 1` does
        const double sgn = theta > 0 ? 1.0 : (theta < 0 ? -1.0 : 1.0);
        const double t = sgn / (std::abs(theta) + std::sqrt(theta * theta + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < 4; ++k) {
          const double akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 4; ++k) {
          const double apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 4; ++k) {
          const double vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (int i = 0; i < 4; ++i) values[i] = A[i][i];
}

// the proper rotation that best takes the source directions onto the target ones (`properRotation`): the largest
// eigenvector of the 4x4 built from the correlation matrix, as a quaternion, written row-major for ROW vectors
void properRotation(const double C[9], double R[9]) {
  const double Sxx = C[0], Sxy = C[1], Sxz = C[2], Syx = C[3], Syy = C[4], Syz = C[5], Szx = C[6], Szy = C[7],
               Szz = C[8];
  double N[4][4] = {{Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx},
                    {Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz},
                    {Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy},
                    {Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz}};
  double values[4], V[4][4];
  jacobiEigen4(N, values, V);
  int best = 0;
  for (int i = 1; i < 4; ++i) if (values[i] > values[best]) best = i;
  const double w = V[0][best], x = V[1][best], y = V[2][best], z = V[3][best];
  const double Q[3][3] = {{w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)},
                          {2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)},
                          {2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z}};
  // the row-vector rotation is Q transposed
  R[0] = Q[0][0]; R[1] = Q[1][0]; R[2] = Q[2][0];
  R[3] = Q[0][1]; R[4] = Q[1][1]; R[5] = Q[2][1];
  R[6] = Q[0][2]; R[7] = Q[1][2]; R[8] = Q[2][2];
}

}  // namespace

void alignChirality(int n, const float* caNm, const float* chirClean, const int* frames, const int* owners,
                    const float* pad, int frameWidth, float* out) {
  const int S = 6;                       // MAX_CHIRAL_SLOTS
  const int K = frameWidth - 2;          // the neighbours a frame names
  const int W = 18;                      // the packed vectors a token: S slots of three
  std::vector<double> rot((size_t)n * S * 9, 0.0);
  std::vector<uint8_t> have((size_t)n * S, 0);
  for (int ci = 0; ci < n; ++ci) {
    if (pad[ci] == 0) continue;
    for (int b = 0; b < S; ++b) {
      const size_t fr = ((size_t)ci * S + b) * frameWidth;
      if (frames[fr] < 0) continue;
      const int o0 = frames[fr], o1 = frames[fr + 1];
      const double ox = ((double)caNm[o0 * 3] + caNm[o1 * 3]) / 2;
      const double oy = ((double)caNm[o0 * 3 + 1] + caNm[o1 * 3 + 1]) / 2;
      const double oz = ((double)caNm[o0 * 3 + 2] + caNm[o1 * 3 + 2]) / 2;
      double C[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
      for (int k = 0; k < K; ++k) {
        const int nb = frames[fr + 2 + k];
        if (nb < 0) continue;
        const double sv[3] = {chirClean[(size_t)nb * W + b * 3], chirClean[(size_t)nb * W + b * 3 + 1],
                              chirClean[(size_t)nb * W + b * 3 + 2]};
        double t[3] = {caNm[nb * 3] - ox, caNm[nb * 3 + 1] - oy, caNm[nb * 3 + 2] - oz};
        const double norm = std::max(std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]), 1e-8);
        for (int j = 0; j < 3; ++j) t[j] /= norm;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) C[i * 3 + j] += sv[i] * t[j];
      }
      properRotation(C, rot.data() + ((size_t)ci * S + b) * 9);
      have[(size_t)ci * S + b] = 1;
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int s = 0; s < S; ++s) {
      const int owner = owners[(size_t)i * S + s];
      // 🔴 A SLOT WITH NO OWNER KEEPS THE CLEAN VECTOR, and a padded token keeps zeros: the reference starts from
      // a copy of the clean array and overwrites only the slots a frame owns.
      if (owner < 0 || !have[(size_t)owner]) continue;
      const double* R = rot.data() + (size_t)owner * 9;
      const size_t o = (size_t)i * W + s * 3;
      const double v[3] = {chirClean[o], chirClean[o + 1], chirClean[o + 2]};
      out[o] = (float)(v[0] * R[0] + v[1] * R[3] + v[2] * R[6]);
      out[o + 1] = (float)(v[0] * R[1] + v[1] * R[4] + v[2] * R[7]);
      out[o + 2] = (float)(v[0] * R[2] + v[1] * R[5] + v[2] * R[8]);
    }
  }
  for (int i = 0; i < n; ++i)
    if (pad[i] == 0)
      for (int k = 0; k < W; ++k) out[(size_t)i * W + k] = 0.f;
}
