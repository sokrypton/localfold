// BioEmu2's EDM sampler: af3-any-model's diffusion_head.sample under the bioemu2 constants (model_registry's
// _SAMPLER_CONSTANTS: bioemu2's EDMSampler with the checkpoint's EDMSchedule) - boltz's loop:
//   levels: sigma_data (sMax^(1/rho) + k/(n-1) (sMin^(1/rho) - sMax^(1/rho)))^rho, k < n, then a 0 appended
//   x = levels[0] N(0, 1)
//   each step (sigma -> sigma'): centre on the real atoms, rotate at random, translate by N(0, 1), zero the padding;
//     gamma = gamma0 where the NEXT level exceeds gammaMin;  t = sigma (1 + gamma)
//     x_noisy = x + noiseScale sqrt(t^2 - sigma^2) N(0, 1);  d = D(x_noisy, t)
//     x_noisy = Kabsch(x_noisy onto d) (REALIGN_SAMPLER);  x = x_noisy + stepScale (sigma' - t) (x_noisy - d) / t
//   each sample centred on its real atoms at the end
// With a COMPANION (propagation's conditioning frame) the frame takes the state's rotation and translation every
// step, each centred on its own mean, and is the denoiser's coords_cond.
// The draws are this port's own (a seeded mt19937 per particle), so a sample is a different sample from the
// reference's; the denoiser is held to the reference's recorded inputs instead (oracle.py).
//
// 🔴 ALL PARTICLES STEP TOGETHER, WHICH IS THE SEAM STEERING NEEDS. BioEmu2's edm_gd/edm_fk/edm_fkc presets couple
// the samples (Feynman-Kac resampling reorders particles mid-trajectory), so the loop is particle-inner: every
// particle's denoised estimate exists before any takes its step. `StepHook` is where they would go - `guide` may
// move each denoised estimate (gradient guidance on the energy of D), and `resample` may permute the particles
// (and their companions) after the step. Neither is implemented here; a null hook is the plain sampler.
#pragma once
#include <random>
#include "denoise_call.cuh"
#include "../../af3/src/rigid.cuh"

namespace be2 {

// at most this many particles a denoiser call (bioemu2.cu sets it from the room)
inline int BATCH_SAMPLES = 64;
struct SamplerSettings { int steps = 200; double sMax = 160, sMin = 4e-4, rho = 7, gamma0 = 0.8, gammaMin = 1.0,
                         noiseScale = 1.0, stepScale = 1.5; };
struct StepHook {
  // (step, sigma', t, the particles' denoised estimates [n][N * 3], host): may move the estimates
  std::function<void(int, double, double, std::vector<std::vector<float>>&)> guide;
  // (step, the particles' states and companions after the step): may permute / replace them
  std::function<void(int, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&)> resample;
};

// the levels as af3-any-model computes them: float32 times and a float32 power, then the 0
inline std::vector<float> schedule(const SamplerSettings& s) {
  std::vector<float> v;
  const float hi = powf((float)s.sMax, 1.f / (float)s.rho), lo = powf((float)s.sMin, 1.f / (float)s.rho);
  for (int k = 0; k < s.steps; ++k) {
    float t = s.steps == 1 ? 0.f : (float)k / (float)(s.steps - 1);
    v.push_back(SIGMA_DATA * powf(hi + t * (lo - hi), (float)s.rho));
  }
  v.push_back(0.f);
  return v;
}

// a uniform random rotation (a normalised Gaussian quaternion) and a N(0, 1) translation, applied to x
// centred on its masked mean; padding zeroed
inline void augment(std::vector<float>& x, const std::vector<float>& mask, const double R[9], const double sh[3]) {
  const size_t N = mask.size();
  double tot = 0, c[3] = {0, 0, 0};
  for (size_t a = 0; a < N; ++a) { tot += mask[a]; for (int k = 0; k < 3; ++k) c[k] += mask[a] * x[a * 3 + k]; }
  for (int k = 0; k < 3; ++k) c[k] /= tot + 1e-6;
  for (size_t a = 0; a < N; ++a) {
    double p[3] = {x[a * 3] - c[0], x[a * 3 + 1] - c[1], x[a * 3 + 2] - c[2]};
    for (int k = 0; k < 3; ++k)
      x[a * 3 + k] = mask[a] ? (float)(p[0] * R[0 * 3 + k] + p[1] * R[1 * 3 + k] + p[2] * R[2 * 3 + k] + sh[k]) : 0.f;
  }
}
inline void randomRigid(std::mt19937_64& rng, double R[9], double sh[3]) {
  std::normal_distribution<double> Nd(0, 1);
  double q[4]; for (auto& v : q) v = Nd(rng);
  double qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  double w = q[0] / qn, i = q[1] / qn, j = q[2] / qn, k = q[3] / qn;
  double r[9] = {1 - 2 * (j * j + k * k), 2 * (i * j - k * w), 2 * (i * k + j * w),
                 2 * (i * j + k * w), 1 - 2 * (i * i + k * k), 2 * (j * k - i * w),
                 2 * (i * k - j * w), 2 * (j * k + i * w), 1 - 2 * (i * i + j * j)};
  for (int a = 0; a < 9; ++a) R[a] = r[a];
  for (int a = 0; a < 3; ++a) sh[a] = Nd(rng);
}
inline void centreMasked(std::vector<float>& x, const std::vector<float>& mask) {
  const size_t N = mask.size(); double tot = 0, c[3] = {0, 0, 0};
  for (size_t a = 0; a < N; ++a) { tot += mask[a]; for (int k = 0; k < 3; ++k) c[k] += mask[a] * x[a * 3 + k]; }
  for (int k = 0; k < 3; ++k) c[k] /= std::max(tot, 1e-6);
  for (size_t a = 0; a < N; ++a) for (int k = 0; k < 3; ++k) x[a * 3 + k] = mask[a] ? (float)(x[a * 3 + k] - c[k]) : 0.f;
}

// GPU-resident per-step augmentation + optional noise injection:
// One block of 256 threads per particle p = blockIdx.x.
// Computes the masked centroid in double precision, applies the random rigid transform
// (R[9], sh[3]) in double precision, zeros padded atoms, and adds noiseAll (if non-null).
__global__ void __launch_bounds__(256)
be2AugmentK(float* __restrict__ xAll, const float* __restrict__ mask,
            const double* __restrict__ rshAll, const float* __restrict__ noiseAll, int N) {
  __shared__ double sRed[8][4];
  __shared__ double sC[3];
  const int p = blockIdx.x, tx = threadIdx.x, warp = tx >> 5, lane = tx & 31;
  float* x = xAll + (size_t)p * N * 3;
  double tot = 0, c0 = 0, c1 = 0, c2 = 0;
  for (int a = tx; a < N; a += 256) {
    double m = mask[a];
    tot += m;
    c0 += m * x[a * 3];
    c1 += m * x[a * 3 + 1];
    c2 += m * x[a * 3 + 2];
  }
  for (int o = 16; o; o >>= 1) {
    tot += __shfl_xor_sync(~0u, tot, o);
    c0  += __shfl_xor_sync(~0u, c0, o);
    c1  += __shfl_xor_sync(~0u, c1, o);
    c2  += __shfl_xor_sync(~0u, c2, o);
  }
  if (lane == 0) {
    sRed[warp][0] = tot; sRed[warp][1] = c0; sRed[warp][2] = c1; sRed[warp][3] = c2;
  }
  __syncthreads();
  if (tx == 0) {
    double t = 0, x0 = 0, x1 = 0, x2 = 0;
    for (int w = 0; w < 8; ++w) {
      t += sRed[w][0]; x0 += sRed[w][1]; x1 += sRed[w][2]; x2 += sRed[w][3];
    }
    double inv = 1.0 / (t + 1e-6);
    sC[0] = x0 * inv; sC[1] = x1 * inv; sC[2] = x2 * inv;
  }
  __syncthreads();
  const double mc0 = sC[0], mc1 = sC[1], mc2 = sC[2];
  const double* R = rshAll + (size_t)p * 12;
  const double r00 = R[0], r01 = R[1], r02 = R[2];
  const double r10 = R[3], r11 = R[4], r12 = R[5];
  const double r20 = R[6], r21 = R[7], r22 = R[8];
  const double sh0 = R[9], sh1 = R[10], sh2 = R[11];
  const float* np = noiseAll ? (noiseAll + (size_t)p * N * 3) : nullptr;
  for (int a = tx; a < N; a += 256) {
    bool live = mask[a] != 0.f;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f;
    if (live) {
      double d0 = x[a * 3] - mc0, d1 = x[a * 3 + 1] - mc1, d2 = x[a * 3 + 2] - mc2;
      v0 = (float)(d0 * r00 + d1 * r10 + d2 * r20 + sh0);
      v1 = (float)(d0 * r01 + d1 * r11 + d2 * r21 + sh1);
      v2 = (float)(d0 * r02 + d1 * r12 + d2 * r22 + sh2);
    }
    if (np) {
      v0 += np[a * 3];
      v1 += np[a * 3 + 1];
      v2 += np[a * 3 + 2];
    }
    x[a * 3] = v0; x[a * 3 + 1] = v1; x[a * 3 + 2] = v2;
  }
}

// GPU-resident Kabsch rigid alignment of xAll onto targetAll (dd) + Euler step:
//   x_aligned = Kabsch(x, d)
//   x_next    = x_aligned + f * (x_aligned - d)
__global__ void __launch_bounds__(256)
be2RigidAlignEulerK(float* __restrict__ xAll, const float* __restrict__ targetAll,
                    const float* __restrict__ w, int n, double f) {
  __shared__ double sRed[8][9];
  __shared__ double sCx[3], sCt[3], sR[9];
  float* x = xAll + (size_t)blockIdx.x * n * 3;
  const float* target = targetAll + (size_t)blockIdx.x * n * 3;
  int tx = threadIdx.x, warp = tx >> 5, lane = tx & 31;
  double tot = 0, cx0 = 0, cx1 = 0, cx2 = 0, ct0 = 0, ct1 = 0, ct2 = 0;
  float rx[16][3], rt[16][3], rw[16];
  const bool reg = n <= 4096;
  if (reg) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      int a = tx + k * 256;
      if (a >= n) break;
      float wa = w[a];
      rw[k] = wa;
      float x0 = x[a * 3], x1 = x[a * 3 + 1], x2 = x[a * 3 + 2];
      float t0 = target[a * 3], t1 = target[a * 3 + 1], t2 = target[a * 3 + 2];
      rx[k][0] = x0; rx[k][1] = x1; rx[k][2] = x2;
      rt[k][0] = t0; rt[k][1] = t1; rt[k][2] = t2;
      double dw = wa;
      tot += dw;
      cx0 += dw * x0; cx1 += dw * x1; cx2 += dw * x2;
      ct0 += dw * t0; ct1 += dw * t1; ct2 += dw * t2;
    }
  } else {
    for (int a = tx; a < n; a += 256) {
      double wa = w[a];
      tot += wa;
      cx0 += wa * x[a * 3];     cx1 += wa * x[a * 3 + 1];     cx2 += wa * x[a * 3 + 2];
      ct0 += wa * target[a * 3]; ct1 += wa * target[a * 3 + 1]; ct2 += wa * target[a * 3 + 2];
    }
  }
  for (int o = 16; o; o >>= 1) {
    tot += __shfl_xor_sync(~0u, tot, o);
    cx0 += __shfl_xor_sync(~0u, cx0, o); cx1 += __shfl_xor_sync(~0u, cx1, o); cx2 += __shfl_xor_sync(~0u, cx2, o);
    ct0 += __shfl_xor_sync(~0u, ct0, o); ct1 += __shfl_xor_sync(~0u, ct1, o); ct2 += __shfl_xor_sync(~0u, ct2, o);
  }
  if (lane == 0) {
    sRed[warp][0] = tot;
    sRed[warp][1] = cx0; sRed[warp][2] = cx1; sRed[warp][3] = cx2;
    sRed[warp][4] = ct0; sRed[warp][5] = ct1; sRed[warp][6] = ct2;
  }
  __syncthreads();
  if (tx == 0) {
    double t = 0, x0 = 0, x1 = 0, x2 = 0, y0 = 0, y1 = 0, y2 = 0;
    for (int k = 0; k < 8; ++k) {
      t  += sRed[k][0];
      x0 += sRed[k][1]; x1 += sRed[k][2]; x2 += sRed[k][3];
      y0 += sRed[k][4]; y1 += sRed[k][5]; y2 += sRed[k][6];
    }
    if (t < 1e-8) t = 1e-8;
    double invT = 1.0 / t;
    sCx[0] = x0 * invT; sCx[1] = x1 * invT; sCx[2] = x2 * invT;
    sCt[0] = y0 * invT; sCt[1] = y1 * invT; sCt[2] = y2 * invT;
  }
  __syncthreads();
  const double cx[3] = {sCx[0], sCx[1], sCx[2]}, ct[3] = {sCt[0], sCt[1], sCt[2]};
  double h[9] = {0};
  if (reg) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      int a = tx + k * 256;
      if (a >= n) break;
      double wa = rw[k];
      if (wa == 0) continue;
      double dt0 = rt[k][0] - ct[0], dt1 = rt[k][1] - ct[1], dt2 = rt[k][2] - ct[2];
      double dx0 = rx[k][0] - cx[0], dx1 = rx[k][1] - cx[1], dx2 = rx[k][2] - cx[2];
      h[0] += wa * dt0 * dx0; h[1] += wa * dt0 * dx1; h[2] += wa * dt0 * dx2;
      h[3] += wa * dt1 * dx0; h[4] += wa * dt1 * dx1; h[5] += wa * dt1 * dx2;
      h[6] += wa * dt2 * dx0; h[7] += wa * dt2 * dx1; h[8] += wa * dt2 * dx2;
    }
  } else {
    for (int a = tx; a < n; a += 256) {
      double wa = w[a];
      if (wa == 0) continue;
      double dt0 = target[a * 3] - ct[0], dt1 = target[a * 3 + 1] - ct[1], dt2 = target[a * 3 + 2] - ct[2];
      double dx0 = x[a * 3] - cx[0],      dx1 = x[a * 3 + 1] - cx[1],      dx2 = x[a * 3 + 2] - cx[2];
      h[0] += wa * dt0 * dx0; h[1] += wa * dt0 * dx1; h[2] += wa * dt0 * dx2;
      h[3] += wa * dt1 * dx0; h[4] += wa * dt1 * dx1; h[5] += wa * dt1 * dx2;
      h[6] += wa * dt2 * dx0; h[7] += wa * dt2 * dx1; h[8] += wa * dt2 * dx2;
    }
  }
  for (int i = 0; i < 9; ++i) {
    for (int o = 16; o; o >>= 1) h[i] += __shfl_xor_sync(~0u, h[i], o);
    if (lane == 0) sRed[warp][i] = h[i];
  }
  __syncthreads();
  if (tx == 0) {
    for (int i = 0; i < 9; ++i) {
      double s = 0;
      for (int k = 0; k < 8; ++k) s += sRed[k][i];
      h[i] = s;
    }
    double* m = sRed[0];
    double* v = sRed[1];
    double* vv = sRed[2];
    double* uu = sRed[3];
    double* uvt = sRed[4];
    double* ev = sRed[5];
    for (int i = 0; i < 9; ++i) { v[i] = (i % 4 == 0) ? 1.0 : 0.0; uu[i] = 0.0; }
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      double s = 0; for (int k = 0; k < 3; ++k) s += h[k * 3 + i] * h[k * 3 + j];
      m[i * 3 + j] = s;
    }
    for (int sweep = 0; sweep < 32; ++sweep) {
      double off = m[1] * m[1] + m[2] * m[2] + m[5] * m[5];
      if (off <= 1e-30 * (m[0] * m[0] + m[4] * m[4] + m[8] * m[8])) break;
      const int P[3][2] = {{0, 1}, {0, 2}, {1, 2}};
#pragma unroll
      for (int pi = 0; pi < 3; ++pi) {
        int p = P[pi][0], q = P[pi][1];
        double apq = m[p * 3 + q];
        if (fabs(apq) < 1e-300) continue;
        double theta = (m[q * 3 + q] - m[p * 3 + p]) / (2 * apq);
        double t = (theta >= 0 ? 1 : -1) / (fabs(theta) + sqrt(theta * theta + 1));
        double c = rsqrt(t * t + 1), s = t * c;
#pragma unroll
        for (int k = 0; k < 3; ++k) { double akp = m[k * 3 + p], akq = m[k * 3 + q]; m[k * 3 + p] = c * akp - s * akq; m[k * 3 + q] = s * akp + c * akq; }
#pragma unroll
        for (int k = 0; k < 3; ++k) { double apk = m[p * 3 + k], aqk = m[q * 3 + k]; m[p * 3 + k] = c * apk - s * aqk; m[q * 3 + k] = s * apk + c * aqk; }
#pragma unroll
        for (int k = 0; k < 3; ++k) { double vkp = v[k * 3 + p], vkq = v[k * 3 + q]; v[k * 3 + p] = c * vkp - s * vkq; v[k * 3 + q] = s * vkp + c * vkq; }
      }
    }
    ev[0] = m[0]; ev[1] = m[4]; ev[2] = m[8];
    int o0 = 0, o1 = 1, o2 = 2;
    if (ev[o1] > ev[o0]) { int tmp = o0; o0 = o1; o1 = tmp; }
    if (ev[o2] > ev[o1]) { int tmp = o1; o1 = o2; o2 = tmp; }
    if (ev[o1] > ev[o0]) { int tmp = o0; o0 = o1; o1 = tmp; }
    int order[3] = {o0, o1, o2};
    bool good[3] = {false, false, false};
    double floor = 1e-12 * fabs(ev[o0] ? ev[o0] : 1);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      int oc = order[c];
      for (int r = 0; r < 3; ++r) vv[r * 3 + c] = v[r * 3 + oc];
      if (ev[oc] <= floor) continue;
      double invSig = rsqrt(ev[oc]);
      for (int r = 0; r < 3; ++r) { double s = 0; for (int k = 0; k < 3; ++k) s += h[r * 3 + k] * vv[k * 3 + c]; uu[r * 3 + c] = s * invSig; }
      double lenSq = uu[c] * uu[c] + uu[3 + c] * uu[3 + c] + uu[6 + c] * uu[6 + c];
      if (fabs(lenSq - 1.0) > 2e-6) { uu[c] = uu[3 + c] = uu[6 + c] = 0; } else good[c] = true;
    }
    for (int c = 0; c < 3; ++c) {
      if (good[c]) continue;
      double best0 = 0, best1 = 0, best2 = 0, bestLen = -1;
      for (int sd = 0; sd < 3; ++sd) {
        double w0 = sd == 0 ? 1 : 0, w1 = sd == 1 ? 1 : 0, w2 = sd == 2 ? 1 : 0;
        for (int k = 0; k < 3; ++k) if (good[k]) {
          double dot = w0 * uu[k] + w1 * uu[3 + k] + w2 * uu[6 + k];
          w0 -= dot * uu[k]; w1 -= dot * uu[3 + k]; w2 -= dot * uu[6 + k];
        }
        double len = sqrt(w0 * w0 + w1 * w1 + w2 * w2);
        if (len > bestLen) { bestLen = len; best0 = w0; best1 = w1; best2 = w2; }
      }
      double invBest = 1.0 / bestLen;
      uu[c] = best0 * invBest; uu[3 + c] = best1 * invBest; uu[6 + c] = best2 * invBest;
      good[c] = true;
    }
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { double s = 0; for (int k = 0; k < 3; ++k) s += uu[i * 3 + k] * vv[j * 3 + k]; uvt[i * 3 + j] = s; }
    double det = uvt[0] * (uvt[4] * uvt[8] - uvt[5] * uvt[7]) - uvt[1] * (uvt[3] * uvt[8] - uvt[5] * uvt[6]) + uvt[2] * (uvt[3] * uvt[7] - uvt[4] * uvt[6]);
    double sign = det < 0 ? -1 : 1;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { double s = 0; for (int k = 0; k < 3; ++k) s += uu[i * 3 + k] * (k == 2 ? sign : 1) * vv[j * 3 + k]; sR[i * 3 + j] = s; }
  }
  __syncthreads();
  const double r00 = sR[0], r01 = sR[1], r02 = sR[2];
  const double r10 = sR[3], r11 = sR[4], r12 = sR[5];
  const double r20 = sR[6], r21 = sR[7], r22 = sR[8];
  if (reg) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      int a = tx + k * 256;
      if (a >= n) break;
      double d0 = rx[k][0] - cx[0], d1 = rx[k][1] - cx[1], d2 = rx[k][2] - cx[2];
      float ax0 = (float)(d0 * r00 + d1 * r01 + d2 * r02 + ct[0]);
      float ax1 = (float)(d0 * r10 + d1 * r11 + d2 * r12 + ct[1]);
      float ax2 = (float)(d0 * r20 + d1 * r21 + d2 * r22 + ct[2]);
      x[a * 3]     = (float)(ax0 + f * (ax0 - rt[k][0]));
      x[a * 3 + 1] = (float)(ax1 + f * (ax1 - rt[k][1]));
      x[a * 3 + 2] = (float)(ax2 + f * (ax2 - rt[k][2]));
    }
  } else {
    for (int a = tx; a < n; a += 256) {
      double d0 = x[a * 3] - cx[0], d1 = x[a * 3 + 1] - cx[1], d2 = x[a * 3 + 2] - cx[2];
      float ax0 = (float)(d0 * r00 + d1 * r01 + d2 * r02 + ct[0]);
      float ax1 = (float)(d0 * r10 + d1 * r11 + d2 * r12 + ct[1]);
      float ax2 = (float)(d0 * r20 + d1 * r21 + d2 * r22 + ct[2]);
      x[a * 3]     = (float)(ax0 + f * (ax0 - target[a * 3]));
      x[a * 3 + 1] = (float)(ax1 + f * (ax1 - target[a * 3 + 1]));
      x[a * 3 + 2] = (float)(ax2 + f * (ax2 - target[a * 3 + 2]));
    }
  }
}

// what each step's predictions are handed to (--frames): (particle, step, steps, the denoised estimate on the host)
inline std::function<void(int, int, int, const std::vector<float>&)> FRAME_HOOK;

// The step conditioning sample() computed last, kept for the next call of the SAME fold (a propagation's steps and a
// steered sampler's rounds condition every call alike). 🔴 FORGOTTEN AT EVERY FOLD (foldInput): it is recognised by
// the conditioning's ADDRESS, and the next fold's - another job, another mode - is recomputed into the same scratch,
// so a resident server's equilibrium job handed its conditioning to the structure job after it, and three structure
// jobs in a row came back with the third's samples 19 A from the first's (tools/check-native-worker.py, "bioemu2 6mrr
// again (the resident server, warm)")
struct SamplerCondCache {
  const float* sc0 = nullptr; int T = 0, layers = 0; float dynGate = -1.f; std::vector<float> tEmb;
  void forget() { *this = SamplerCondCache{}; }
};
inline SamplerCondCache SAMPLER_COND;
// n particles from noise (companions: one centred conditioning frame each, or empty); returns their final states
inline std::vector<std::vector<float>> sample(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int n,
                                              std::vector<std::vector<float>> companions = {},
                                              const StepHook* hook = nullptr, bool centreEnd = true) {
  const int N = d.N;
  std::vector<float> mask(N);
  { const int* m = M.i("atom_mask"); for (int a = 0; a < N; ++a) mask[a] = m[a] ? 1.f : 0.f; }
  const std::vector<float> levels = schedule(s);
  const int steps = (int)levels.size() - 1;
  std::vector<float> hTEmb(steps);
  for (int i = 0; i < steps; ++i) {
    const double sigma = levels[i], next = levels[i + 1];
    const double gamma = next > s.gammaMin ? s.gamma0 : 0;
    const double t = sigma * (1 + gamma);
    hTEmb[i] = 0.25f * logf((float)t / SIGMA_DATA);
  }
  SamplerCondCache& cached = SAMPLER_COND;
  const bool sameCond = (steps > 0 && cached.sc0 == d.sc0 && cached.T == d.T &&
                         cached.layers == d.layers && cached.dynGate == d.dynGate && cached.tEmb == hTEmb);
  float* cndAll = sameCond ? scratch<float>("be2.sc.cndAll", (size_t)steps * (d.T + 1) * 768)
                           : precomputeAllSingleCond(d, hTEmb);
  const int Ca = 128, Dt = 768, Dc = Dt + Ca;
  int condBatch = std::max(1, steps);
  const int rowsT1 = condBatch == 1 ? d.T : (d.T + 1);
  const size_t condAStride = (size_t)(d.T + 1) * (6 * Ca);
  const size_t condTStride = (size_t)rowsT1 * (6 * Dt);
  const size_t cqcStride   = (size_t)(d.T + 1) * (2 * Dc + Ca);
  const size_t cscStride   = (size_t)rowsT1 * (2 * Dc + Dt);
  const size_t perStepCondBytes = (size_t)d.layers * (condAStride + condTStride + cqcStride + cscStride) * 4;
  while (condBatch > 1 && !roomFor((size_t)condBatch * perStepCondBytes + (1ull << 30))) condBatch >>= 1;
  float* condAAll = scratch<float>("be2.dn.condA", (size_t)d.layers * condBatch * condAStride);
  float* condTAll = scratch<float>("be2.dn.condT", (size_t)d.layers * condBatch * condTStride);
  float* cqcAll   = scratch<float>("be2.dn.cqc",   (size_t)d.layers * condBatch * cqcStride);
  float* cscAll   = scratch<float>("be2.dn.csc",   (size_t)d.layers * condBatch * cscStride);
  const bool reuseAllCond = (condBatch == steps && sameCond);
  int curCondBase = reuseAllCond ? 0 : -1;
  int curCondB = reuseAllCond ? steps : 0;
  if (steps > 0 && !reuseAllCond) {
    curCondBase = 0;
    curCondB = std::min(condBatch, steps);
    computeStepCondBatch(d, cndAll, curCondB, condAAll, condTAll, cqcAll, cscAll);
    if (condBatch == steps) {
      cached.sc0 = d.sc0;
      cached.T = d.T;
      cached.layers = d.layers;
      cached.dynGate = d.dynGate;
      cached.tEmb = hTEmb;
    }
  }
  const bool comp = !companions.empty();
  const size_t per = (size_t)N * 3;
  // the particles go through the denoiser together, `batch` at a time (all of them where there is the room)
  const int batch = std::max(1, std::min(n, BATCH_SAMPLES));
  float* dx = scratch<float>("be2.smp.dx", per * batch);
  float* dd = scratch<float>("be2.smp.dd", per * batch);
  float* dc = comp ? scratch<float>("be2.smp.dc", per * batch) : nullptr;
  static float* pinnedBuf = nullptr;
  static size_t pinnedCap = 0;
  size_t needPinned = per * batch * (comp ? 3 : 2);
  if (needPinned > pinnedCap) {
    if (pinnedBuf) CK(cudaFreeHost(pinnedBuf));
    CK(cudaHostAlloc(&pinnedBuf, needPinned * sizeof(float), cudaHostAllocDefault));
    pinnedCap = needPinned;
  }
  float* hx = pinnedBuf;
  float* hd = pinnedBuf + per * batch;
  float* hc = comp ? (pinnedBuf + 2 * per * batch) : nullptr;

  std::vector<std::mt19937_64> rng;
  for (int p = 0; p < n; ++p) rng.emplace_back(seed * 1000003ull + (uint64_t)p);
  std::normal_distribution<double> Nd(0, 1);
  std::vector<std::vector<float>> x(n, std::vector<float>((size_t)N * 3)), den(n, std::vector<float>((size_t)N * 3));
  for (int p = 0; p < n; ++p) for (auto& v : x[p]) v = (float)(levels[0] * Nd(rng[p]));

  if (!hook && !FRAME_HOOK && batch >= n && steps > 0) {
    constexpr int RNG_RING = 8;
    const size_t rshBytes = (size_t)n * 12 * sizeof(double);
    const size_t noiseBytes = per * (size_t)n * sizeof(float);
    const size_t slotBytes = (rshBytes + noiseBytes + 63) & ~size_t(63);
    static char* pinnedRing = nullptr;
    static size_t pinnedRingCap = 0;
    static cudaEvent_t slotEv[RNG_RING] = {};
    static bool slotEvInit = false;
    if (!slotEvInit) {
      for (int s0 = 0; s0 < RNG_RING; ++s0) CK(cudaEventCreateWithFlags(&slotEv[s0], cudaEventDisableTiming));
      slotEvInit = true;
    }
    if (slotBytes * RNG_RING > pinnedRingCap) {
      if (pinnedRing) CK(cudaFreeHost(pinnedRing));
      CK(cudaHostAlloc(&pinnedRing, slotBytes * RNG_RING, cudaHostAllocDefault));
      pinnedRingCap = slotBytes * RNG_RING;
    }
    auto slotRsh = [&](int sl) { return (double*)(pinnedRing + sl * slotBytes); };
    auto slotNoise = [&](int sl) { return (float*)(pinnedRing + sl * slotBytes + rshBytes); };
    auto fillSlot = [&](int stepIdx, int sl) {
      const double sigma = levels[stepIdx], next = levels[stepIdx + 1];
      const double gamma = next > s.gammaMin ? s.gamma0 : 0;
      const double t = sigma * (1 + gamma);
      const double eps = s.noiseScale * sqrt(std::max(t * t - sigma * sigma, 0.0));
      double* rsh = slotRsh(sl);
      float* npAll = slotNoise(sl);
      for (int p = 0; p < n; ++p) {
        randomRigid(rng[p], rsh + p * 12, rsh + p * 12 + 9);
        float* np = npAll + per * p;
        for (size_t a = 0; a < per; ++a) np[a] = (float)(eps * Nd(rng[p]));
      }
    };
    double* dRsh = scratch<double>("be2.smp.rsh", (size_t)n * 12);
    float* dNoise = scratch<float>("be2.smp.noise", per * (size_t)n);
    float* dMask = scratch<float>("be2.smp.mask", N);
    CK(cudaMemcpyAsync(dMask, mask.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice, STREAM));
    for (int p = 0; p < n; ++p) {
      std::copy(x[p].begin(), x[p].end(), hx + per * p);
      if (comp) std::copy(companions[p].begin(), companions[p].end(), hc + per * p);
    }
    CK(cudaMemcpyAsync(dx, hx, per * (size_t)n * sizeof(float), cudaMemcpyHostToDevice, STREAM));
    if (comp) CK(cudaMemcpyAsync(dc, hc, per * (size_t)n * sizeof(float), cudaMemcpyHostToDevice, STREAM));

    fillSlot(0, 0);
    for (int i = 0; i < steps; ++i) {
      const double sigma = levels[i], next = levels[i + 1];
      const double gamma = next > s.gammaMin ? s.gamma0 : 0;
      const double t = sigma * (1 + gamma);
      if (i < curCondBase || i >= curCondBase + curCondB) {
        curCondBase = i;
        curCondB = std::min(condBatch, steps - i);
        computeStepCondBatch(d, cndAll + (size_t)i * (d.T + 1) * Dt, curCondB,
                             condAAll, condTAll, cqcAll, cscAll);
      }
      const StepCond stepCond{condAAll, condTAll, cqcAll, cscAll, curCondB, i - curCondBase};
      const int sl = i % RNG_RING;
      CK(cudaMemcpyAsync(dRsh, slotRsh(sl), rshBytes, cudaMemcpyHostToDevice, STREAM));
      CK(cudaMemcpyAsync(dNoise, slotNoise(sl), noiseBytes, cudaMemcpyHostToDevice, STREAM));
      CK(cudaEventRecord(slotEv[sl], STREAM));
      be2AugmentK<<<n, 256, 0, STREAM>>>(dx, dMask, dRsh, dNoise, N);
      if (comp) be2AugmentK<<<n, 256, 0, STREAM>>>(dc, dMask, dRsh, nullptr, N);
      denoise(d, dx, (float)t, dd, dc, n, &stepCond);
      const double f = s.stepScale * (next - t) / t;
      be2RigidAlignEulerK<<<n, 256, 0, STREAM>>>(dx, dd, dMask, N, f);
      if (i + 1 < steps) {
        const int nextSl = (i + 1) % RNG_RING;
        if (i + 1 >= RNG_RING) CK(cudaEventSynchronize(slotEv[nextSl]));
        fillSlot(i + 1, nextSl);
      }
    }
    CK(cudaMemcpyAsync(hx, dx, per * (size_t)n * sizeof(float), cudaMemcpyDeviceToHost, STREAM));
    CK(cudaStreamSynchronize(STREAM));
    for (int p = 0; p < n; ++p) std::copy(hx + per * p, hx + per * (p + 1), x[p].begin());
    if (centreEnd) for (auto& v : x) centreMasked(v, mask);
    else for (auto& v : x) for (int a = 0; a < N; ++a) if (!mask[a]) v[a * 3] = v[a * 3 + 1] = v[a * 3 + 2] = 0.f;
    return x;
  }

  std::vector<std::array<double, 9>> nextR(n);
  std::vector<std::array<double, 3>> nextSh(n);
  std::vector<std::vector<float>> nextNoise(n, std::vector<float>(per));
  bool compReady = false;
  auto genStepRng = [&](int stepIdx) {
    const double sigma = levels[stepIdx], next = levels[stepIdx + 1];
    const double gamma = next > s.gammaMin ? s.gamma0 : 0;
    const double t = sigma * (1 + gamma);
    const double eps = s.noiseScale * sqrt(std::max(t * t - sigma * sigma, 0.0));
    const bool preComp = comp && !(hook && hook->resample);
    for (int p = 0; p < n; ++p) {
      randomRigid(rng[p], nextR[p].data(), nextSh[p].data());
      float* np = nextNoise[p].data();
      for (int a = 0; a < N * 3; ++a) np[a] = (float)(eps * Nd(rng[p]));
      if (preComp) {
        augment(companions[p], mask, nextR[p].data(), nextSh[p].data());
        if (batch >= n) std::copy(companions[p].begin(), companions[p].end(), hc + per * p);
      }
    }
    compReady = preComp;
  };
  if (steps > 0) genStepRng(0);

  for (int i = 0; i < steps; ++i) {
    const double sigma = levels[i], next = levels[i + 1];
    const double gamma = next > s.gammaMin ? s.gamma0 : 0;
    const double t = sigma * (1 + gamma);
    if (i < curCondBase || i >= curCondBase + curCondB) {
      curCondBase = i;
      curCondB = std::min(condBatch, steps - i);
      computeStepCondBatch(d, cndAll + (size_t)i * (d.T + 1) * Dt, curCondB,
                           condAAll, condTAll, cqcAll, cscAll);
    }
    const StepCond stepCond{condAAll, condTAll, cqcAll, cscAll, curCondB, i - curCondBase};
    for (int p = 0; p < n; ++p) {
      augment(x[p], mask, nextR[p].data(), nextSh[p].data());
      if (comp && !compReady) augment(companions[p], mask, nextR[p].data(), nextSh[p].data());
      const float* np = nextNoise[p].data();
      for (int a = 0; a < N * 3; ++a) x[p][a] += np[a];
    }
    for (int p0 = 0; p0 < n; p0 += batch) {
      const int b = std::min(batch, n - p0);
      for (int k = 0; k < b; ++k) {
        std::copy(x[p0 + k].begin(), x[p0 + k].end(), hx + per * k);
        if (comp && !(compReady && batch >= n))
          std::copy(companions[p0 + k].begin(), companions[p0 + k].end(), hc + per * k);
      }
      CK(cudaMemcpyAsync(dx, hx, per * b * 4, cudaMemcpyHostToDevice, STREAM));
      if (comp) CK(cudaMemcpyAsync(dc, hc, per * b * 4, cudaMemcpyHostToDevice, STREAM));
      denoise(d, dx, (float)t, dd, dc, b, &stepCond);
      CK(cudaMemcpyAsync(hd, dd, per * b * 4, cudaMemcpyDeviceToHost, STREAM));
      if (p0 + batch >= n && i + 1 < steps) genStepRng(i + 1);
      CK(cudaStreamSynchronize(STREAM));
      for (int k = 0; k < b; ++k) std::copy(hd + per * k, hd + per * (k + 1), den[p0 + k].begin());
    }
    if (hook && hook->guide) hook->guide(i, next, t, den);
    for (int p = 0; p < n; ++p) {
      if (FRAME_HOOK) FRAME_HOOK(p, i + 1, steps, den[p]);
      rigidAlign(x[p], den[p], mask, N);
      const double f = s.stepScale * (next - t) / t;
      for (int k = 0; k < N * 3; ++k) x[p][k] = (float)(x[p][k] + f * (x[p][k] - den[p][k]));
    }
    if (hook && hook->resample) hook->resample(i, x, companions);
  }
  if (centreEnd) for (auto& v : x) centreMasked(v, mask);
  else for (auto& v : x) for (int a = 0; a < N; ++a) if (!mask[a]) v[a * 3] = v[a * 3 + 1] = v[a * 3 + 2] = 0.f;
  return x;
}

// propagation (bioemu2.propagation): each particle a trajectory of `chain` lag-time steps from x0. A step samples
// conditioned on the previous frame (centred, the companion), then superposes the endpoint on that frame. Returns
// [particle][step] frames
inline std::vector<std::vector<std::vector<float>>> propagate(const Denoiser& d, const SamplerSettings& s, uint64_t seed,
                                                             int n, int chain, const std::vector<float>& x0,
                                                             const StepHook* hook = nullptr) {
  const int N = d.N;
  std::vector<float> mask(N);
  { const int* m = M.i("atom_mask"); for (int a = 0; a < N; ++a) mask[a] = m[a] ? 1.f : 0.f; }
  std::vector<std::vector<float>> prev(n, x0);
  std::vector<std::vector<std::vector<float>>> frames(n);
  for (int k = 0; k < chain; ++k) {
    std::vector<std::vector<float>> comps = prev;
    for (auto& c : comps) centreMasked(c, mask);
    // (the reference folds the step into its key; here the step is part of the seed)
    std::vector<std::vector<float>> end = sample(d, s, seed * 7919ull + (uint64_t)k, n, comps, hook, false);
    for (int p = 0; p < n; ++p) {
      rigidAlign(end[p], prev[p], mask, N);
      for (int a = 0; a < N; ++a) if (!mask[a]) end[p][a * 3] = end[p][a * 3 + 1] = end[p][a * 3 + 2] = 0.f;
      frames[p].push_back(end[p]);
      prev[p] = end[p];
    }
  }
  return frames;
}

// the records of pdb.template (export_input.mjs) with these coordinates
inline void writePdb(const std::string& templatePath, const std::string& out, const std::vector<float>& x) {
  std::ifstream in(templatePath);
  if (!in) { fprintf(stderr, "no %s (export_input.mjs writes it)\n", templatePath.c_str()); exit(1); }
  FILE* f = fopen(out.c_str(), "w");
  if (!f) { fprintf(stderr, "cannot write %s: %s\n", out.c_str(), strerror(errno)); exit(1); }
  fprintf(f, "REMARK   1 BioEmu2 (bioemu2-preview): no confidence head - the B-factor column is zero, not a pLDDT\n");
  std::string line;
  while (std::getline(in, line)) {
    if ((line.rfind("ATOM", 0) == 0 || line.rfind("HETATM", 0) == 0) && line.size() >= 66) {
      int atom = (int)lround(atof(line.substr(30, 8).c_str())) * 1000 + (int)lround(atof(line.substr(38, 8).c_str()));
      char coords[64];
      snprintf(coords, sizeof coords, "%8.3f%8.3f%8.3f", x[atom * 3], x[atom * 3 + 1], x[atom * 3 + 2]);
      line = line.substr(0, 30) + coords + line.substr(54);
    }
    fprintf(f, "%s\n", line.c_str());
  }
  fclose(f);
}

}  // namespace be2
