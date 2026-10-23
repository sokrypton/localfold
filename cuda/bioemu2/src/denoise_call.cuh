// One call of BioEmu2's denoiser, several samples at once (see denoise.cuh for the model). Everything that depends on
// sigma alone - the single conditioning and every layer's conditioning projections, on the token side AND the atom
// side - is computed once a call and shared by every sample; each atom row reads its token's (CondLayer's map).
#pragma once
#include "denoise.cuh"

namespace be2 {

// the row maps of `samples` stacked samples: each atom to its token's conditioning row (padding atoms to 0), each
// token to its own
struct RowMaps { int samples = 0; int* atomTok = nullptr; int* tok = nullptr; };
inline RowMaps ROW_MAPS;
inline const RowMaps& rowMaps(int samples, int T, int N) {
  if (ROW_MAPS.samples == samples && ROW_MAPS.atomTok) return ROW_MAPS;
  if (ROW_MAPS.atomTok) { CK(cudaFree(ROW_MAPS.atomTok)); CK(cudaFree(ROW_MAPS.tok)); }
  const int* tok = M.i("token_idx_at_atom");
  std::vector<int> a((size_t)samples * N), t((size_t)samples * T);
  for (size_t r = 0; r < a.size(); ++r) a[r] = tok[r % N] >= 0 ? tok[r % N] : T;   // (a padding atom: the zero row T)
  for (size_t r = 0; r < t.size(); ++r) t[r] = (int)(r % T);
  ROW_MAPS = {samples, upload(a.data(), a.size()), upload(t.data(), t.size())};
  return ROW_MAPS;
}
inline void forgetRowMaps() {
  if (ROW_MAPS.atomTok) { CK(cudaStreamSynchronize(STREAM)); CK(cudaFree(ROW_MAPS.atomTok)); CK(cudaFree(ROW_MAPS.tok)); }
  ROW_MAPS = {};
}

__global__ void scaleMaskedBatchK(const float* x, const int* mask, float* r, size_t N, size_t n, float s) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n * 3) r[t] = mask[(t / 3) % N] ? x[t] * s : 0.f;
}
__global__ void repeatRowsK(const float* __restrict__ src, float* __restrict__ dst, size_t elems, size_t total) {   // dst = src, src, ...
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((elems & 3) == 0 && (total & 3) == 0 && (((uintptr_t)src | (uintptr_t)dst) & 15) == 0) {
    size_t e4 = elems >> 2, tot4 = total >> 2;
    if (t < tot4) {
      reinterpret_cast<float4*>(dst)[t] = __ldg(reinterpret_cast<const float4*>(src) + (t % e4));
    }
    return;
  }
  if (t < total) dst[t] = src[t % elems];
}
inline void repeatRows(const float* src, float* dst, size_t elems, size_t total) {
  size_t nt = ((elems & 3) == 0 && (total & 3) == 0 && (((uintptr_t)src | (uintptr_t)dst) & 15) == 0) ? (total >> 2) : total;
  repeatRowsK<<<blocks(nt), 256, 0, STREAM>>>(src, dst, elems, total);
}

__global__ void initCondK(const float* __restrict__ sc0, const float* __restrict__ f768,
                          float* __restrict__ cnd, int T, int Dt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int d4 = Dt >> 2;
  size_t n4 = (size_t)(T + 1) * d4;
  if (t >= n4) return;
  size_t r = t / (size_t)d4;
  int c4 = (int)(t - r * (size_t)d4);
  if (r < (size_t)T) {
    float4 sv = __ldg(reinterpret_cast<const float4*>(sc0) + t);
    float4 fv = __ldg(reinterpret_cast<const float4*>(f768) + c4);
    reinterpret_cast<float4*>(cnd)[t] = make_float4(sv.x + fv.x, sv.y + fv.y, sv.z + fv.z, sv.w + fv.w);
  } else {
    reinterpret_cast<float4*>(cnd)[t] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
}

__global__ void cosEmbedBatchK(const float* __restrict__ w, const float* __restrict__ b,
                               float* __restrict__ out, const float* __restrict__ tArr, int steps, int F) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= steps * F) return;
  int s = idx / F, k = idx - s * F;
  float t = __ldg(tArr + s);
  out[idx] = fastCosF(2.f * (float)M_PI * (t * __ldg(w + k) + __ldg(b + k)));
}

__global__ void initCondPackedBatchK(const float* __restrict__ sc0, const float* __restrict__ f768,
                                     float* __restrict__ cndPacked, int steps, int T, int Dt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int d4 = Dt >> 2;
  size_t td4 = (size_t)T * d4;
  size_t n4 = (size_t)steps * td4;
  if (t >= n4) return;
  size_t s = t / td4, rem = t - s * td4;
  int c4 = (int)(rem % (size_t)d4);
  float4 sv = __ldg(reinterpret_cast<const float4*>(sc0) + rem);
  float4 fv = __ldg(reinterpret_cast<const float4*>(f768) + s * (size_t)d4 + c4);
  reinterpret_cast<float4*>(cndPacked)[t] = make_float4(sv.x + fv.x, sv.y + fv.y, sv.z + fv.z, sv.w + fv.w);
}

__global__ void padCondBatchK(const float* __restrict__ cndPacked, float* __restrict__ cndAll,
                              int steps, int T, int Dt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int d4 = Dt >> 2;
  size_t stride4 = (size_t)(T + 1) * d4;
  size_t n4 = (size_t)steps * stride4;
  if (t >= n4) return;
  size_t s = t / stride4, rem = t - s * stride4;
  size_t r = rem / (size_t)d4;
  if (r < (size_t)T) {
    reinterpret_cast<float4*>(cndAll)[t] = __ldg(reinterpret_cast<const float4*>(cndPacked) + s * (size_t)T * d4 + rem);
  } else {
    reinterpret_cast<float4*>(cndAll)[t] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
}

struct StepCond {
  const float* condAAll = nullptr;
  const float* condTAll = nullptr;
  const float* cqcAll   = nullptr;
  const float* cscAll   = nullptr;
  int stepBatch = 1;
  int stepIdx = 0;
};

inline float* precomputeAllSingleCond(const Denoiser& d, const std::vector<float>& hTEmb) {
  const int steps = (int)hTEmb.size(), T = d.T, Dt = 768;
  const std::string sc = SM + "single_conditioner/";
  const int F = dimP(sc + "fourier_embed/proj/bias", 0);
  float* tDev = scratch<float>("be2.sc.tDev", steps);
  CK(cudaMemcpyAsync(tDev, hTEmb.data(), (size_t)steps * 4, cudaMemcpyHostToDevice, STREAM));
  float* four = scratch<float>("be2.sc.fourB", (size_t)steps * F);
  cosEmbedBatchK<<<blocks((size_t)steps * F), 256, 0, STREAM>>>(P(sc + "fourier_embed/proj/weight"),
                                                                P(sc + "fourier_embed/proj/bias"),
                                                                four, tDev, steps, F);
  layerNorm(four, four, steps, F, P(sc + "norm_fourier/weight"), P(sc + "norm_fourier/bias"));
  float* f768 = scratch<float>("be2.sc.f768B", (size_t)steps * Dt);
  linear(four, P(sc + "fourier_to_single/weight"), f768, steps, F, Dt);
  float* cndPacked = scratch<float>("be2.sc.cndPacked", (size_t)steps * T * Dt);
  initCondPackedBatchK<<<blocks(((size_t)steps * T * Dt) >> 2), 256, 0, STREAM>>>(d.sc0, f768, cndPacked, steps, T, Dt);
  for (int i = 0; i < 2; ++i)
    transition(cndPacked, cndPacked, (size_t)steps * T, Dt, sc + "transitions_" + std::to_string(i) + "/");
  float* cndAll = scratch<float>("be2.sc.cndAll", (size_t)steps * (T + 1) * Dt);
  padCondBatchK<<<blocks(((size_t)steps * (T + 1) * Dt) >> 2), 256, 0, STREAM>>>(cndPacked, cndAll, steps, T, Dt);
  return cndAll;
}

inline void computeStepCondBatch(const Denoiser& d, const float* cndBatch, int B,
                                 float* condAAll, float* condTAll, float* cqcAll, float* cscAll) {
  const int T = d.T, Ca = 128, Dt = 768, Dc = Dt + Ca;
  const std::string L = SM + "layers/";
  const std::string cp = L + "coupling/";
  const std::string ap = L + "atom_block/transformer_layer/", tp = L + "token_block/";
  Cond0Tables c0Tab{
    {P(ap + "cond_0/weight"),
     P(tp + "cond_0/weight"),
     P(cp + "q_coupling_cond_0/weight"),
     P(cp + "s_coupling_cond_0/weight")},
    {Popt(ap + "cond_0/bias"),
     Popt(tp + "cond_0/bias"),
     Popt(cp + "q_coupling_cond_0/bias"),
     Popt(cp + "s_coupling_cond_0/bias")}
  };
  const int rows = B * (T + 1);
  const int rowsT = B == 1 ? T : rows;
  const size_t condAStride = (size_t)rows * (6 * Ca);
  const size_t condTStride = (size_t)rowsT * (6 * Dt);
  const size_t cqcStride   = (size_t)rows * (2 * Dc + Ca);
  const size_t cscStride   = (size_t)rowsT * (2 * Dc + Dt);
  const size_t cnhStride   = (size_t)rows * Dt;
  const size_t streamStride = (size_t)d.layers * cnhStride;
  dim3 c0Grid((unsigned)((rows + 7) / 8), (unsigned)d.layers);
  if (GEMM_HALF) {
    half* cnhAll = scratch<half>("be2.dn.cnhAll", 4 * streamStride);
    lnToCond4K<half><<<c0Grid, 256, 0, STREAM>>>(cndBatch, cnhAll, T, rows, c0Tab);
    linearBatchPre<half>(cnhAll + 0 * streamStride, (long long)cnhStride,
                         halfMirror<half>(P(ap + "cond_2/weight"), (size_t)d.layers * (6 * Ca) * Dt),
                         (long long)(6 * Ca) * Dt, condAAll, (long long)condAStride, rows, Dt, 6 * Ca, d.layers);
    linearBatchPre<half>(cnhAll + 1 * streamStride, (long long)cnhStride,
                         halfMirror<half>(P(tp + "cond_2/weight"), (size_t)d.layers * (6 * Dt) * Dt),
                         (long long)(6 * Dt) * Dt, condTAll, (long long)condTStride, rowsT, Dt, 6 * Dt, d.layers);
    linearBatchPre<half>(cnhAll + 2 * streamStride, (long long)cnhStride,
                         halfMirror<half>(P(cp + "q_coupling_cond_2/weight"), (size_t)d.layers * (2 * Dc + Ca) * Dt),
                         (long long)(2 * Dc + Ca) * Dt, cqcAll, (long long)cqcStride, rows, Dt, 2 * Dc + Ca, d.layers);
    linearBatchPre<half>(cnhAll + 3 * streamStride, (long long)cnhStride,
                         halfMirror<half>(P(cp + "s_coupling_cond_2/weight"), (size_t)d.layers * (2 * Dc + Dt) * Dt),
                         (long long)(2 * Dc + Dt) * Dt, cscAll, (long long)cscStride, rowsT, Dt, 2 * Dc + Dt, d.layers);
  } else {
    float* cnAll = scratch<float>("be2.dn.cnAll", 4 * streamStride);
    lnToCond4K<float><<<c0Grid, 256, 0, STREAM>>>(cndBatch, cnAll, T, rows, c0Tab);
    linearBatch(cnAll + 0 * streamStride, (long long)cnhStride, P(ap + "cond_2/weight"),
                (long long)(6 * Ca) * Dt, condAAll, (long long)condAStride, rows, Dt, 6 * Ca, d.layers);
    linearBatch(cnAll + 1 * streamStride, (long long)cnhStride, P(tp + "cond_2/weight"),
                (long long)(6 * Dt) * Dt, condTAll, (long long)condTStride, rowsT, Dt, 6 * Dt, d.layers);
    linearBatch(cnAll + 2 * streamStride, (long long)cnhStride, P(cp + "q_coupling_cond_2/weight"),
                (long long)(2 * Dc + Ca) * Dt, cqcAll, (long long)cqcStride, rows, Dt, 2 * Dc + Ca, d.layers);
    linearBatch(cnAll + 3 * streamStride, (long long)cnhStride, P(cp + "s_coupling_cond_2/weight"),
                (long long)(2 * Dc + Dt) * Dt, cscAll, (long long)cscStride, rowsT, Dt, 2 * Dc + Dt, d.layers);
  }
  addBiasBatchRows(condAAll, Popt(ap + "cond_2/bias"), rows, 6 * Ca, d.layers);
  addBiasBatchRows(condTAll, Popt(tp + "cond_2/bias"), rowsT, 6 * Dt, d.layers);
  addBiasBatchRows(cqcAll, Popt(cp + "q_coupling_cond_2/bias"), rows, 2 * Dc + Ca, d.layers);
  addBiasBatchRows(cscAll, Popt(cp + "s_coupling_cond_2/bias"), rowsT, 2 * Dc + Dt, d.layers);
}

// Fused atom stream initialization:
//   r[row] = mask[al] ? cIn * x[row] : 0
//   q[row] = q0[al] + r[row] @ wR2Q^T (+ dynGate * cond[row_or_al] @ wC2Q^T)
//   and when DO_ADALN is true (GEMM_HALF), computes layer-0's adaLN(q) directly in registers into hh0 [RN, 128].
template <bool DO_ADALN>
__global__ void __launch_bounds__(128, 8)
qInitK(const float* __restrict__ q0, const float* __restrict__ x, const int* __restrict__ mask,
       const float* __restrict__ wR2Q, const float* __restrict__ cond, const float* __restrict__ wC2Q,
       float* __restrict__ q, half* __restrict__ hh0, const float* __restrict__ condA0,
       const int* __restrict__ mapA, size_t RN, int N, float cIn, float dynGate, bool condPerSample) {
  size_t row = (size_t)blockIdx.x * 4 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (row >= RN) return;
  int al = (int)(row % (size_t)N);
  int m = __ldg(mask + al);
  float rx = m ? __ldg(x + row * 3 + 0) * cIn : 0.f;
  float ry = m ? __ldg(x + row * 3 + 1) * cIn : 0.f;
  float rz = m ? __ldg(x + row * 3 + 2) * cIn : 0.f;
  int c0 = lane * 4;
  float4 qv = __ldg(reinterpret_cast<const float4*>(q0) + (size_t)al * 32 + lane);
  const float4* wr4 = reinterpret_cast<const float4*>(wR2Q + c0 * 3);
  float4 w0 = __ldg(wr4 + 0), w1 = __ldg(wr4 + 1), w2 = __ldg(wr4 + 2);
  qv.x += w0.x * rx + w0.y * ry + w0.z * rz;
  qv.y += w0.w * rx + w1.x * ry + w1.y * rz;
  qv.z += w1.z * rx + w1.w * ry + w2.x * rz;
  qv.w += w2.y * rx + w2.z * ry + w2.w * rz;
  if (cond && wC2Q) {
    size_t crow = condPerSample ? row : (size_t)al;
    float cx = __ldg(cond + crow * 3 + 0);
    float cy = __ldg(cond + crow * 3 + 1);
    float cz = __ldg(cond + crow * 3 + 2);
    const float4* wc4 = reinterpret_cast<const float4*>(wC2Q + c0 * 3);
    float4 u0 = __ldg(wc4 + 0), u1 = __ldg(wc4 + 1), u2 = __ldg(wc4 + 2);
    qv.x += dynGate * (u0.x * cx + u0.y * cy + u0.z * cz);
    qv.y += dynGate * (u0.w * cx + u1.x * cy + u1.y * cz);
    qv.z += dynGate * (u1.z * cx + u1.w * cy + u2.x * cz);
    qv.w += dynGate * (u2.y * cx + u2.z * ry * 0.f + u2.z * cy + u2.w * cz);
  }
  reinterpret_cast<float4*>(q + row * 128)[lane] = qv;
  if constexpr (DO_ADALN) {
    float s = (qv.x + qv.y) + (qv.z + qv.w);
    #pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s * (1.f / 128.f);
    float dx = qv.x - mean, dy = qv.y - mean, dz = qv.z - mean, dw = qv.w - mean;
    float var = (dx * dx + dy * dy) + (dz * dz + dw * dw);
    #pragma unroll
    for (int o = 16; o; o >>= 1) var += __shfl_xor_sync(~0u, var, o);
    float inv = rsqrtf(var * (1.f / 128.f) + 1e-5f);
    size_t cr = (size_t)__ldg(mapA + row);
    const float4* ar4 = reinterpret_cast<const float4*>(condA0 + cr * (6 * 128));
    const float4* br4 = reinterpret_cast<const float4*>(condA0 + cr * (6 * 128) + 128);
    float4 av = __ldg(ar4 + lane), bv = __ldg(br4 + lane);
    storeVec4<half>(hh0 + row * 128, lane,
                    dx * inv * (1.f + av.x) + bv.x,
                    dy * inv * (1.f + av.y) + bv.y,
                    dz * inv * (1.f + av.z) + bv.z,
                    dw * inv * (1.f + av.w) + bv.w);
  }
}

// Fused token stream initialization:
//   rc[row] = r[s * N + centre[il]]
//   a[row]  = a0[il] + rc[row] @ wR2A^T
//   and when DO_ADALN is true (GEMM_HALF), computes layer-0's adaLN(a) directly in registers into thh0 [RT, 768].
template <bool DO_ADALN>
__global__ void __launch_bounds__(128, 4)
aInitK(const float* __restrict__ a0, const float* __restrict__ x, const int* __restrict__ mask,
       const int* __restrict__ centre, const float* __restrict__ wR2A,
       float* __restrict__ a, half* __restrict__ thh0, const float* __restrict__ condT0,
       const int* __restrict__ mapT, size_t RT, int T, int N, float cIn) {
  size_t row = (size_t)blockIdx.x * 4 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (row >= RT) return;
  size_t s = row / (size_t)T;
  int il = (int)(row - s * (size_t)T);
  int ca = __ldg(centre + il);
  int m = __ldg(mask + ca);
  size_t aRow = s * (size_t)N + (size_t)ca;
  float rx = m ? __ldg(x + aRow * 3 + 0) * cIn : 0.f;
  float ry = m ? __ldg(x + aRow * 3 + 1) * cIn : 0.f;
  float rz = m ? __ldg(x + aRow * 3 + 2) * cIn : 0.f;
  const float4* a0r = reinterpret_cast<const float4*>(a0 + (size_t)il * 768);
  float4* ar = reinterpret_cast<float4*>(a + row * 768);
  float4 xv[6];
  float sum = 0.f;
  #pragma unroll
  for (int k = 0; k < 6; ++k) {
    int c4 = k * 32 + lane;
    int c0 = c4 * 4;
    float4 av = __ldg(a0r + c4);
    const float4* wr4 = reinterpret_cast<const float4*>(wR2A + c0 * 3);
    float4 w0 = __ldg(wr4 + 0), w1 = __ldg(wr4 + 1), w2 = __ldg(wr4 + 2);
    av.x += w0.x * rx + w0.y * ry + w0.z * rz;
    av.y += w0.w * rx + w1.x * ry + w1.y * rz;
    av.z += w1.z * rx + w1.w * ry + w2.x * rz;
    av.w += w2.y * rx + w2.z * ry + w2.w * rz;
    ar[c4] = av;
    xv[k] = av;
    if constexpr (DO_ADALN) sum += (av.x + av.y) + (av.z + av.w);
  }
  if constexpr (DO_ADALN) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    float mean = sum * (1.f / 768.f), var = 0.f;
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
      float dx = xv[k].x - mean, dy = xv[k].y - mean, dz = xv[k].z - mean, dw = xv[k].w - mean;
      var += (dx * dx + dy * dy) + (dz * dz + dw * dw);
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) var += __shfl_xor_sync(~0u, var, o);
    float inv = rsqrtf(var * (1.f / 768.f) + 1e-5f);
    size_t cr = (size_t)__ldg(mapT + row);
    const float4* ar4 = reinterpret_cast<const float4*>(condT0 + cr * (6 * 768));
    const float4* br4 = reinterpret_cast<const float4*>(condT0 + cr * (6 * 768) + 768);
    half* yr = thh0 + row * 768;
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
      int c4 = k * 32 + lane;
      float4 av = __ldg(ar4 + c4), bv = __ldg(br4 + c4);
      storeVec4<half>(yr, c4,
                      (xv[k].x - mean) * inv * (1.f + av.x) + bv.x,
                      (xv[k].y - mean) * inv * (1.f + av.y) + bv.y,
                      (xv[k].z - mean) * inv * (1.f + av.z) + bv.z,
                      (xv[k].w - mean) * inv * (1.f + av.w) + bv.w);
    }
  }
}

// Layer 15's final gatedAdd fused with f32->f16 conversion for readout's q_to_hidden and s_to_hidden GEMMs.
__global__ void gatedAddToHalfK(const float* __restrict__ x, const float* __restrict__ h,
                                const float* __restrict__ gate, half* __restrict__ xh,
                                size_t rows, int C, int ldg, const int* __restrict__ map) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int c4 = C >> 2, ldg4 = ldg >> 2;
  size_t n4 = rows * (size_t)c4;
  if (t >= n4) return;
  size_t r; int col4;
  if ((c4 & (c4 - 1)) == 0) {
    col4 = (int)(t & (size_t)(c4 - 1));
    r = t >> (__ffs(c4) - 1);
  } else if (c4 == 192 && n4 <= UINT32_MAX) {
    unsigned u = (unsigned)t, q = u / 192u; r = q; col4 = (int)(u - q * 192u);
  } else {
    r = t / c4; col4 = (int)(t % c4);
  }
  size_t gr = map ? (size_t)__ldg(map + r) : r;
  float4 xv = __ldg(reinterpret_cast<const float4*>(x) + t);
  float4 hv = __ldg(reinterpret_cast<const float4*>(h) + t);
  float4 gv = __ldg(reinterpret_cast<const float4*>(gate) + gr * ldg4 + col4);
  storeVec4<half>(xh, (int)t,
                  xv.x + gv.x * hv.x, xv.y + gv.y * hv.y,
                  xv.z + gv.z * hv.z, xv.w + gv.w * hv.w);
}

// Fused readout tail: tokensToAtoms(ah, h) + LayerNorm(384, to_out_0) + Linear(384->3, to_out_1) + edmOut(x, out, den),
// one warp per atom row. Replaces 4 launches and 3 global memory passes over h [RN, 384] per denoiser step.
__global__ void __launch_bounds__(256, 4)
readoutTailK(const float* __restrict__ h, const float* __restrict__ ah,
             const int* __restrict__ atomTok, const int* __restrict__ atomPad,
             const float* __restrict__ normW, const float* __restrict__ wOut,
             const float* __restrict__ x, float* __restrict__ den,
             size_t RN, int N, int T, float skip, float cout) {
  constexpr int Hd = 384, K4 = Hd / 4;
  __shared__ float4 sNorm[K4];
  __shared__ float4 sW0[K4], sW1[K4], sW2[K4];
  for (int idx = threadIdx.x; idx < K4; idx += blockDim.x) {
    sNorm[idx] = __ldg(reinterpret_cast<const float4*>(normW) + idx);
    sW0[idx]   = __ldg(reinterpret_cast<const float4*>(wOut) + 0 * K4 + idx);
    sW1[idx]   = __ldg(reinterpret_cast<const float4*>(wOut) + 1 * K4 + idx);
    sW2[idx]   = __ldg(reinterpret_cast<const float4*>(wOut) + 2 * K4 + idx);
  }
  __syncthreads();
  size_t row = (size_t)blockIdx.x * 8 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (row >= RN) return;
  unsigned urow = (unsigned)row, uN = (unsigned)N;
  int s = (int)(urow / uN), al = (int)(urow - (unsigned)s * uN);
  int pad = __ldg(atomPad + al);
  int tokIdx = pad ? __ldg(atomTok + al) : 0;
  const float4* hr4 = reinterpret_cast<const float4*>(h + row * Hd);
  const float4* ahr4 = reinterpret_cast<const float4*>(ah + ((size_t)s * T + tokIdx) * Hd);
  float4 xv[3];
  float sum = 0.f;
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    int c4 = k * 32 + lane;
    float4 hv = __ldg(hr4 + c4);
    float4 av = pad ? __ldg(ahr4 + c4) : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 v = make_float4(hv.x + av.x, hv.y + av.y, hv.z + av.z, hv.w + av.w);
    xv[k] = v;
    sum += (v.x + v.y) + (v.z + v.w);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  float mean = sum * (1.f / 384.f), var = 0.f;
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    float dx = xv[k].x - mean, dy = xv[k].y - mean, dz = xv[k].z - mean, dw = xv[k].w - mean;
    var += (dx * dx + dy * dy) + (dz * dz + dw * dw);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) var += __shfl_xor_sync(~0u, var, o);
  float inv = rsqrtf(var * (1.f / 384.f) + 1e-5f);
  float ox = 0.f, oy = 0.f, oz = 0.f;
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    int c4 = k * 32 + lane;
    float4 sc = sNorm[c4], w0 = sW0[c4], w1 = sW1[c4], w2 = sW2[c4];
    float nx = (xv[k].x - mean) * inv * sc.x;
    float ny = (xv[k].y - mean) * inv * sc.y;
    float nz = (xv[k].z - mean) * inv * sc.z;
    float nw = (xv[k].w - mean) * inv * sc.w;
    ox += nx * w0.x + ny * w0.y + nz * w0.z + nw * w0.w;
    oy += nx * w1.x + ny * w1.y + nz * w1.z + nw * w1.w;
    oz += nx * w2.x + ny * w2.y + nz * w2.z + nw * w2.w;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) {
    ox += __shfl_xor_sync(~0u, ox, o);
    oy += __shfl_xor_sync(~0u, oy, o);
    oz += __shfl_xor_sync(~0u, oz, o);
  }
  if (lane == 0) {
    const float* xr = x + row * 3;
    float* dr = den + row * 3;
    dr[0] = skip * __ldg(xr + 0) + cout * ox;
    dr[1] = skip * __ldg(xr + 1) + cout * oy;
    dr[2] = skip * __ldg(xr + 2) + cout * oz;
  }
}

// x [samples, N, 3] (noisy, angstrom) at sigma -> den [samples, N, 3]. `cond` (optional, [samples, N, 3]) is each
// sample's conditioning frame (propagation's companion, augmented with the state); null takes the featurised
// coords_cond for every sample
inline void denoise(const Denoiser& d, const float* x, float sigma, float* den, const float* cond = nullptr,
                    int samples = 1, const StepCond* preCond = nullptr) {
  const int T = d.T, N = d.N, Ca = 128, Dt = 768;
  const size_t RN = (size_t)samples * N, RT = (size_t)samples * T;
  const RowMaps& maps = rowMaps(samples, T, N);
  const std::string sc = SM + "single_conditioner/", af = SM + "atom_feature_embedder/";
  const float cIn = 1.f / sqrtf(sigma * sigma + SIGMA_DATA * SIGMA_DATA), t = 0.25f * logf(sigma / SIGMA_DATA);
  const std::string L = SM + "layers/";
  const std::string ab = "attention/attention_layer/pair_bias_attn/", tb = "attention/pair_bias_attn/";
  const size_t Pn = (size_t)T * T, wb = (size_t)N * WK * 4, tbn = 16 * Pn;
  const int Dc = Dt + Ca;                                  // the coupling's width, 896
  StepCond localCond;
  if (!preCond) {
    // the single conditioning (shared)
    const int F = dimP(sc + "fourier_embed/proj/bias", 0);
    float* four = scratch<float>("be2.dn.four", F);
    cosEmbedK<<<blocks(F), 256, 0, STREAM>>>(P(sc + "fourier_embed/proj/weight"), P(sc + "fourier_embed/proj/bias"), four, t, F);
    layerNorm(four, four, 1, F, P(sc + "norm_fourier/weight"), P(sc + "norm_fourier/bias"));
    float* f768 = scratch<float>("be2.dn.f768", Dt);
    linear(four, P(sc + "fourier_to_single/weight"), f768, 1, F, Dt);
    // cond, and a ZERO row after its T: what a padding atom of the reference conditions on (c_to_q is 0 there)
    float* cnd = scratch<float>("be2.dn.cond", (size_t)(T + 1) * Dt);
    initCondK<<<blocks(((size_t)(T + 1) * Dt) >> 2), 256, 0, STREAM>>>(d.sc0, f768, cnd, T, Dt);
    for (int i = 0; i < 2; ++i) transition(cnd, cnd, T, Dt, sc + "transitions_" + std::to_string(i) + "/");
    const size_t condAStride1 = (size_t)(T + 1) * (6 * Ca);
    const size_t condTStride1 = (size_t)T * (6 * Dt);
    const size_t cqcStride1   = (size_t)(T + 1) * (2 * Dc + Ca);
    const size_t cscStride1   = (size_t)T * (2 * Dc + Dt);
    float* condAAll = scratch<float>("be2.dn.condA", (size_t)d.layers * condAStride1);
    float* condTAll = scratch<float>("be2.dn.condT", (size_t)d.layers * condTStride1);
    float* cqcAll   = scratch<float>("be2.dn.cqc",   (size_t)d.layers * cqcStride1);
    float* cscAll   = scratch<float>("be2.dn.csc",   (size_t)d.layers * cscStride1);
    computeStepCondBatch(d, cnd, 1, condAAll, condTAll, cqcAll, cscAll);
    localCond = {condAAll, condTAll, cqcAll, cscAll, 1, 0};
    preCond = &localCond;
  }
  const int rowsT1 = preCond->stepBatch == 1 ? T : (T + 1);
  const size_t condAStride = (size_t)(T + 1) * (6 * Ca);
  const size_t condTStride = (size_t)rowsT1 * (6 * Dt);
  const size_t cqcStride   = (size_t)(T + 1) * (2 * Dc + Ca);
  const size_t cscStride   = (size_t)rowsT1 * (2 * Dc + Dt);
  const int* atomMaskDev = Idev("atom_mask");
  const int* atomTokDev  = Idev("token_idx_at_atom");
  const int* atomPadDev  = Idev("atom_pad_mask");
  const int* centreDev   = Idev("token_center_atom_index");
  // the atom and token streams' initial states (+ layer 0's adaLN fused in registers when GEMM_HALF)
  float* q = scratch<float>("be2.dn.q", RN * Ca);
  float* a = scratch<float>("be2.dn.a", RT * Dt);
  const float* condSrc = d.dyn ? (cond ? cond : d.coordsCond) : nullptr;
  const float* wC2Q    = d.dyn ? P(af + "cond_to_q_trans/weight") : nullptr;
  const float* condA0  = preCond->condAAll + (size_t)preCond->stepIdx * condAStride;
  const float* condT0  = preCond->condTAll + (size_t)preCond->stepIdx * condTStride;
  half* hh0  = GEMM_HALF ? scratch<half>("be2.dn.hh0",  RN * Ca) : nullptr;
  half* thh0 = GEMM_HALF ? scratch<half>("be2.dn.thh0", RT * Dt) : nullptr;
  if (GEMM_HALF) {
    qInitK<true><<<(unsigned)((RN + 3) / 4), 128, 0, STREAM>>>(
        d.q0, x, atomMaskDev, P(af + "r_to_q_trans/weight"), condSrc, wC2Q,
        q, hh0, condA0, maps.atomTok, RN, N, cIn, d.dynGate, cond != nullptr);
    aInitK<true><<<(unsigned)((RT + 3) / 4), 128, 0, STREAM>>>(
        d.a0, x, atomMaskDev, centreDev, P(SM + "r_token_to_a/weight"),
        a, thh0, condT0, maps.tok, RT, T, N, cIn);
  } else {
    qInitK<false><<<(unsigned)((RN + 3) / 4), 128, 0, STREAM>>>(
        d.q0, x, atomMaskDev, P(af + "r_to_q_trans/weight"), condSrc, wC2Q,
        q, nullptr, nullptr, nullptr, RN, N, cIn, d.dynGate, cond != nullptr);
    aInitK<false><<<(unsigned)((RT + 3) / 4), 128, 0, STREAM>>>(
        d.a0, x, atomMaskDev, centreDev, P(SM + "r_token_to_a/weight"),
        a, nullptr, nullptr, nullptr, RT, T, N, cIn);
  }
  // the layers
  float* tbias = d.tbias ? nullptr : scratch<float>("be2.dn.tbias", tbn);
  float* dq = scratch<float>("be2.dn.dq", RN * Ca);
  float* da = scratch<float>("be2.dn.da", RT * Dt);
  float* tAtom = scratch<float>("be2.dn.tAtom", RN * Ca);
  float* tTok  = scratch<float>("be2.dn.tTok",  RT * Dt);
  const std::string cp = L + "coupling/";
  const std::string ap = L + "atom_block/transformer_layer/", tp = L + "token_block/";
  const bool debugHalf = getenv("BE2_DEBUG_HALF") != nullptr;
  half* qhLast = (GEMM_HALF && !debugHalf) ? scratch<half>("be2.dn.qhL", RN * Ca) : nullptr;
  half* ahLast = (GEMM_HALF && !debugHalf) ? scratch<half>("be2.dn.ahL", RT * Dt) : nullptr;
  for (int l = 0; l < d.layers; ++l) {
    const size_t lb = (size_t)l * preCond->stepBatch + preCond->stepIdx;
    const size_t lbPrev = l > 0 ? ((size_t)(l - 1) * preCond->stepBatch + preCond->stepIdx) : 0;
    const float* condA = preCond->condAAll + lb * condAStride;
    const float* condT = preCond->condTAll + lb * condTStride;
    const float* cqc   = preCond->cqcAll   + lb * cqcStride;
    const float* csc   = preCond->cscAll   + lb * cscStride;
    const float* preGateQ = l > 0 ? (preCond->cqcAll + lbPrev * cqcStride + 2 * Dc) : nullptr;
    const float* preGateA = l > 0 ? (preCond->cscAll + lbPrev * cscStride + 2 * Dc) : nullptr;
    const CondLayer atomL{ap, ab, l, true, 4};
    condLayer(atomL, q, (int)RN, Ca, condA, maps.atomTok, d.wbias + wb * l, N, tAtom,
              l > 0 ? dq : nullptr, preGateQ, 2 * Dc + Ca, l == 0 ? hh0 : nullptr);
    const float* bias = d.tbias ? d.tbias + tbn * l : tbias;
    if (!d.tbias) tokenBias(d.z, PL(tp + tb + "norm_z/weight", l), PL(tp + tb + "proj_z/weight", l), tbias, T, 256, 16);
    const CondLayer tokL{tp, tb, l, false, 16};
    condLayer(tokL, a, (int)RT, Dt, condT, maps.tok, bias, T, tTok,
              l > 0 ? da : nullptr, preGateA, 2 * Dc + Dt, l == 0 ? thh0 : nullptr);
    if (GEMM_HALF) {
      // s_to_q = [a broadcast | q]; q_to_s = [q centre | a], fused with post-transition gatedAdd directly into adaLN
      const int Rtok = (int)RT + 2 * (T + 1);
      half* qsh = scratch<half>("be2.dn.qsh", RT * Dc);
      half* uvb = scratch<half>("be2.s2q.uvb", (size_t)Rtok * 768);
      float4* tokStats = scratch<float4>("be2.s2q.tstat", RT);
      adaLnQ2SK<half><<<(unsigned)((Rtok + 3) / 4), 128, 0, STREAM>>>(q, tAtom, condA + 5 * Ca, 6 * Ca, atomTokDev,
                                                                      a, tTok, condT + 5 * Dt, 6 * Dt, centreDev,
                                                                      qsh, RT, N, T, csc, csc + Dc, 2 * Dc + Dt, maps.tok,
                                                                      uvb, tokStats, cqc, cqc + Dc, 2 * Dc + Ca);
      s2qMixerHalf(a, q, tAtom, condA + 5 * Ca, 6 * Ca, atomTokDev, atomPadDev,
                   dq, RN, N, T, RT, cqc, cqc + Dc, 2 * Dc + Ca, maps.atomTok, cp + "s_to_q_mixer/", l,
                   uvb, tokStats);
      transitionHalf(nullptr, qsh, da, RT, Dc, cp + "q_to_s_mixer/", l, false, false, 0);
    } else {
      float* sqn = scratch<float>("be2.dn.sqn", RN * Dc); float* qsn = scratch<float>("be2.dn.qsn", RT * Dc);
      adaLnQ2SK<float><<<(unsigned)((RT + 3) / 4), 128, 0, STREAM>>>(q, tAtom, condA + 5 * Ca, 6 * Ca, atomTokDev,
                                                                     a, tTok, condT + 5 * Dt, 6 * Dt, centreDev,
                                                                     qsn, RT, N, T, csc, csc + Dc, 2 * Dc + Dt, maps.tok);
      adaLnS2QK<float><<<(unsigned)((RN + 3) / 4), 128, 0, STREAM>>>(a, q, tAtom, condA + 5 * Ca, 6 * Ca,
                                                                     atomTokDev, atomPadDev,
                                                                     sqn, RN, N, T, cqc, cqc + Dc, 2 * Dc + Ca, maps.atomTok);
      transition(sqn, dq, RN, Dc, cp + "s_to_q_mixer/", l, false, false);
      transition(qsn, da, RT, Dc, cp + "q_to_s_mixer/", l, false, false);
    }
    if (l + 1 == d.layers) {
      if (qhLast && ahLast) {
        gatedAddToHalfK<<<blocks((RN * Ca) >> 2), 256, 0, STREAM>>>(q, dq, cqc + 2 * Dc, qhLast, RN, Ca, 2 * Dc + Ca, maps.atomTok);
        gatedAddToHalfK<<<blocks((RT * Dt) >> 2), 256, 0, STREAM>>>(a, da, csc + 2 * Dc, ahLast, RT, Dt, 2 * Dc + Dt, maps.tok);
      } else {
        gatedAdd(q, dq, cqc + 2 * Dc, RN, Ca, 2 * Dc + Ca, maps.atomTok);
        gatedAdd(a, da, csc + 2 * Dc, RT, Dt, 2 * Dc + Dt, maps.tok);
      }
    }
  }
  // the readout
  const std::string ro = SM + "readout/";
  const int Hd = dimP(ro + "q_to_hidden/weight", 0);
  float* h = scratch<float>("be2.dn.h", RN * Hd);
  float* ah = scratch<float>("be2.dn.ah", RT * Hd);
  if (qhLast && ahLast) {
    linearPre<half>(qhLast, halfMirror<half>(P(ro + "q_to_hidden/weight"), (size_t)Hd * Ca), h, RN, Ca, Hd);
    linearPre<half>(ahLast, halfMirror<half>(P(ro + "s_to_hidden/weight"), (size_t)Hd * Dt), ah, RT, Dt, Hd);
  } else {
    linear(q, P(ro + "q_to_hidden/weight"), h, RN, Ca, Hd);
    linear(a, P(ro + "s_to_hidden/weight"), ah, RT, Dt, Hd);
  }
  const float s2 = sigma * sigma, sd2 = SIGMA_DATA * SIGMA_DATA;
  const float skip = sd2 / (s2 + sd2), cout = sigma * SIGMA_DATA / sqrtf(sd2 + s2);
  if (!debugHalf && Hd == 384) {
    readoutTailK<<<(unsigned)((RN + 7) / 8), 256, 0, STREAM>>>(
        h, ah, atomTokDev, atomPadDev, P(ro + "to_out_0/weight"), P(ro + "to_out_1/weight"),
        x, den, RN, N, T, skip, cout);
    return;
  }
  tokensToAtoms(ah, h, N, Hd, 0, 1.f, samples, T);
  layerNorm(h, h, RN, Hd, P(ro + "to_out_0/weight"), nullptr);
  float* out = scratch<float>("be2.dn.out", RN * 3);
  linear(h, P(ro + "to_out_1/weight"), out, RN, Hd, 3);
  edmOutK<<<blocks(RN * 3), 256, 0, STREAM>>>(x, out, den, RN * 3, skip, cout);
  if (debugHalf) {
    auto bad = [&](const char* n, const float* p, size_t k) {
      auto h = download(p, k); size_t b = 0; for (float v : h) b += !std::isfinite(v);
      printf("  %s: %zu of %zu nonfinite\n", n, b, k);
    };
    bad("q", q, RN * Ca); bad("a", a, RT * Dt); bad("h", h, RN * Hd); bad("out", out, RN * 3); bad("x", x, RN * 3);
  }
}

}  // namespace be2
