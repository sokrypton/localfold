// `align_chirality`: the one sequence feature besides the noised state itself that MOVES between
// sampler steps.
//
// A chiral centre's packed vectors are written in the frame of the clean structure. At each step
// the noisy CA positions define a new frame, so each centre's neighbours are re-read out of `x_t`,
// a proper rotation is fitted onto the stored directions, and the centre's vectors are rotated by
// it. Without this the model is told a handedness measured against coordinates it no longer has.
//
// 🔴 THIS IS WHY `--generate` USED TO REFUSE THE SMALL-MOLECULE PRESET. The exporter writes the
// sequence features once and the port overwrites the thirty-five state columns each step; a feature
// that moves for any other reason would silently be frozen at step zero. `sm_sam` is the only
// shipped preset with chiral frames (6 of them, 84 owners over 165 tokens) and it was refused for
// exactly that, rather than folded with a stale handedness.
//
// Ported from `alignChirality` and `properRotation` in shared/rfd4p/features.js. One thread a
// frame, in DOUBLE: the fit is a 4x4 Jacobi eigendecomposition of a quaternion matrix, the
// reference runs it in JS doubles, and there are six of them.
#pragma once
#include "trunk.cuh"

namespace rfd4p {

constexpr int MAX_CHIRAL_SLOTS = 6;

// `properRotation`: the rotation that best carries the stored directions onto the observed ones,
// as the largest eigenvector of the 4x4 quaternion form of the correlation matrix. Returns the
// ROW-vector rotation, the nine entries in the reference's own order.
__device__ inline void properRotation(const double* C, double* R) {
  const double Sxx = C[0], Sxy = C[1], Sxz = C[2];
  const double Syx = C[3], Syy = C[4], Syz = C[5];
  const double Szx = C[6], Szy = C[7], Szz = C[8];
  double A[4][4] = {
      {Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx},
      {Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz},
      {Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy},
      {Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz}};
  double V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = 0;
    for (int p = 0; p < 4; ++p) for (int q = p + 1; q < 4; ++q) off += A[p][q] * A[p][q];
    if (off < 1e-30) break;
    for (int p = 0; p < 4; ++p) {
      for (int q = p + 1; q < 4; ++q) {
        if (fabs(A[p][q]) < 1e-300) continue;
        const double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        // JS `Math.sign(theta || 1)`: zero takes the positive branch, as `0 || 1` is 1.
        const double sgn = theta > 0 ? 1.0 : (theta < 0 ? -1.0 : 1.0);
        const double t = sgn / (fabs(theta) + sqrt(theta * theta + 1));
        const double c = 1 / sqrt(t * t + 1);
        const double s = t * c;
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
  int best = 0;
  for (int i = 1; i < 4; ++i) if (A[i][i] > A[best][best]) best = i;
  const double w = V[0][best], x = V[1][best], y = V[2][best], z = V[3][best];
  const double Q[3][3] = {
      {w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)},
      {2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)},
      {2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z}};
  R[0] = Q[0][0]; R[1] = Q[1][0]; R[2] = Q[2][0];
  R[3] = Q[0][1]; R[4] = Q[1][1]; R[5] = Q[2][1];
  R[6] = Q[0][2]; R[7] = Q[1][2]; R[8] = Q[2][2];
}

/**
 * One rotation a frame. `frames` is `[n, S, F]` with the two origin atoms first and `F - 2`
 * neighbours after, `-1` where a slot is empty; a frame whose first entry is negative is absent.
 */
__global__ void chiralFramesK(const float* __restrict__ chir, const int* __restrict__ frames,
                              const float* __restrict__ xtBbCa, const float* __restrict__ mask,
                              double* __restrict__ rot, int* __restrict__ live, int n, int S,
                              int F) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n * S) return;
  const int ci = id / S, b = id % S;
  live[id] = 0;
  const int fr = id * F;
  if (mask[ci] == 0.f || frames[fr] < 0) return;
  const int o0 = frames[fr], o1 = frames[fr + 1];
  const float ox = (xtBbCa[o0 * 3] + xtBbCa[o1 * 3]) / 2;
  const float oy = (xtBbCa[o0 * 3 + 1] + xtBbCa[o1 * 3 + 1]) / 2;
  const float oz = (xtBbCa[o0 * 3 + 2] + xtBbCa[o1 * 3 + 2]) / 2;
  double C[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int k = 0; k < F - 2; ++k) {
    const int nb = frames[fr + 2 + k];
    if (nb < 0) continue;
    const double sx = chir[nb * 18 + b * 3], sy = chir[nb * 18 + b * 3 + 1];
    const double sz = chir[nb * 18 + b * 3 + 2];
    double tx = xtBbCa[nb * 3] - ox, ty = xtBbCa[nb * 3 + 1] - oy, tz = xtBbCa[nb * 3 + 2] - oz;
    // `Math.hypot`, which is not `sqrt(x*x+y*y+z*z)` in the last place - but this is a direction
    // that is immediately normalised and fed a 4x4 eigensolve, so the port takes the plain form.
    const double norm = fmax(sqrt(tx * tx + ty * ty + tz * tz), 1e-8);
    tx /= norm; ty /= norm; tz /= norm;
    C[0] += sx * tx; C[1] += sx * ty; C[2] += sx * tz;
    C[3] += sy * tx; C[4] += sy * ty; C[5] += sy * tz;
    C[6] += sz * tx; C[7] += sz * ty; C[8] += sz * tz;
  }
  properRotation(C, rot + (size_t)id * 9);
  live[id] = 1;
}

// Each owned slot's three packed numbers, through its frame's rotation. A padded token is zeroed
// whole, as the reference zeroes it.
__global__ void chiralApplyK(const float* __restrict__ chir, const int* __restrict__ owners,
                             const double* __restrict__ rot, const int* __restrict__ live,
                             const float* __restrict__ mask, float* __restrict__ seqF, int n, int S,
                             int fdim, int chirOff) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n * S) return;
  const int i = id / S, s = id % S;
  float* dst = seqF + (size_t)i * fdim + chirOff + s * 3;
  const int o = i * 18 + s * 3;
  if (mask[i] == 0.f) { dst[0] = dst[1] = dst[2] = 0.f; return; }
  const int owner = owners[id];
  const float vx = chir[o], vy = chir[o + 1], vz = chir[o + 2];
  if (owner < 0 || !live[owner]) { dst[0] = vx; dst[1] = vy; dst[2] = vz; return; }
  const double* R = rot + (size_t)owner * 9;
  dst[0] = (float)(vx * R[0] + vy * R[3] + vz * R[6]);
  dst[1] = (float)(vx * R[1] + vy * R[4] + vz * R[7]);
  dst[2] = (float)(vx * R[2] + vy * R[5] + vz * R[8]);
}

/** Both passes, into the sequence feature block at `chirOff`. */
inline void alignChirality(const float* chir, const int* frames, const int* owners,
                           const float* xtBbCa, const float* mask, float* seqF, int n, int F,
                           int fdim, int chirOff) {
  const int S = MAX_CHIRAL_SLOTS;
  double* rot = scratch<double>("rfd4p.chir.rot", (size_t)n * S * 9);
  int* live = scratch<int>("rfd4p.chir.live", (size_t)n * S);
  chiralFramesK<<<blocks((size_t)n * S), 128, 0, STREAM>>>(chir, frames, xtBbCa, mask, rot, live, n,
                                                           S, F);
  chiralApplyK<<<blocks((size_t)n * S), 128, 0, STREAM>>>(chir, owners, rot, live, mask, seqF, n, S,
                                                          fdim, chirOff);
}

}  // namespace rfd4p
