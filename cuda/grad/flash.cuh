// FlashAttention-2 on the tensor cores for sdpaForward / sdpaBackward (float builds; D a multiple of 8, at most 64).
// The logits are never materialised: the forward keeps a running max and sum per query row and writes the log-sum-
// exp beside the output; the backward recomputes each 64 x 64 tile of probabilities from it, keeps dK and dV for its
// 64 keys in registers, and accumulates dQ and every bias gradient with atomics. q, k, v, dO, dq, dk, dv are read and
// written in place at their own leading dimensions (head h at columns h D .. h D + D), so nothing is transposed into
// a per-head layout first. The semantics are softmaxBiasK's and softmaxBwdK's exactly: the same mask, biases, clip
// and chain exclusion, and the clip passes the gradient through.
//
// mma.sync.m16n8k8 with tf32 operands and f32 accumulation. With TF32 off (the gates, float32 training) every
// product is three of them - the operands split into a tf32 head and a tf32 tail, hi*hi + hi*lo + lo*hi - which is
// float32's accuracy (the dropped lo*lo is 2^-22 of the product) at well past the SIMT throughput.
#pragma once
#include <cstdint>
#include <type_traits>

namespace grad {
namespace flash {

constexpr int BR = 64, BC = 64, THREADS = 128;   // 4 warps, 16 rows each
constexpr int PADK = 4;                          // smem row pad: conflict-free for the [g][t] fragment reads

__device__ __forceinline__ uint32_t tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ void mmaTf32(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// an A fragment (g, t) (g + 8, t) (g, t + 4) (g + 8, t + 4) as tf32, and with SPLIT its tail: split once per k-step
// and reused across the n-tiles it multiplies
// MODE 0: tf32; 1: 3xTF32 (SPLIT, float32's precision); 2: bf16 (the training precision). bf16's m16n8k8 holds a
// thread's k pair as 2t, 2t + 1 where tf32's holds t, t + 4 - the same pairing in A and B, and k is summed, so the
// tf32 fragment loads feed it unchanged, packed in twos
__device__ __forceinline__ uint32_t bf2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ void mmaBf16(float* c, uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b));
}
template <int MODE>
struct AFrag {
  uint32_t h[4], l[MODE == 1 ? 4 : 1];
  __device__ __forceinline__ explicit AFrag(const float* a) {
    if (MODE == 2) { h[0] = bf2(a[0], a[2]); h[1] = bf2(a[1], a[3]); return; }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[i] = tf32(a[i]);
      if (MODE == 1) l[i] = tf32(a[i] - __uint_as_float(h[i]));
    }
  }
};

// a B operand tile split when it is loaded into shared memory: hi as tf32 bits, and with SPLIT lo beside it (one
// array of the tile's size further on), so a fragment read is two loads and no conversion. NS 3 adds a third part
// (the score product's operands: hi + mid + lo carries float32's 24 bits where hi + lo carries ~22, and the softmax
// turns a logit's absolute error into the probabilities' relative one)
template <int NS>
__device__ __forceinline__ void storeSplit(float* s, size_t lo, float x) {
  if (NS == 0) { s[0] = x; return; }      // (bf16: rounded where the fragment is packed)
  uint32_t h = tf32(x);
  s[0] = __uint_as_float(h);
  if (NS >= 2) {
    float r = x - __uint_as_float(h);
    uint32_t m = tf32(r);
    s[lo] = __uint_as_float(m);
    if (NS >= 3) s[2 * lo] = __uint_as_float(tf32(r - __uint_as_float(m)));
  }
}
// the score product at float32's precision: A and B in three parts, the six products above 2^-33 of the result
struct AFrag3 {
  uint32_t h[4], m[4], l[4];
  __device__ __forceinline__ explicit AFrag3(const float* a) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[i] = tf32(a[i]);
      float r = a[i] - __uint_as_float(h[i]);
      m[i] = tf32(r);
      l[i] = tf32(r - __uint_as_float(m[i]));
    }
  }
};
__device__ __forceinline__ void mma3(float* c, const AFrag3& A, const float* s, size_t lo, int i0, int i1) {
  uint32_t bh[2] = {__float_as_uint(s[i0]), __float_as_uint(s[i1])};
  uint32_t bm[2] = {__float_as_uint(s[lo + i0]), __float_as_uint(s[lo + i1])};
  uint32_t bl[2] = {__float_as_uint(s[2 * lo + i0]), __float_as_uint(s[2 * lo + i1])};
  float t[4] = {0.f, 0.f, 0.f, 0.f};
  mmaTf32(t, A.l[0], A.l[1], A.l[2], A.l[3], bh[0], bh[1]);
  mmaTf32(t, A.h[0], A.h[1], A.h[2], A.h[3], bl[0], bl[1]);
  mmaTf32(t, A.m[0], A.m[1], A.m[2], A.m[3], bm[0], bm[1]);
  mmaTf32(t, A.m[0], A.m[1], A.m[2], A.m[3], bh[0], bh[1]);
  mmaTf32(t, A.h[0], A.h[1], A.h[2], A.h[3], bm[0], bm[1]);
  mmaTf32(t, A.h[0], A.h[1], A.h[2], A.h[3], bh[0], bh[1]);
  c[0] += t[0]; c[1] += t[1]; c[2] += t[2]; c[3] += t[3];
}
// c += A B for one m16n8k8 step, b at (k t, n g) and (k t + 4, n g): s[i0] and s[i1] of a split tile
template <int MODE>
__device__ __forceinline__ void mma(float* c, const AFrag<MODE>& A, const float* s, size_t lo, int i0, int i1) {
  if (MODE == 2) { mmaBf16(c, A.h[0], A.h[1], bf2(s[i0], s[i1])); return; }
  uint32_t bh[2] = {__float_as_uint(s[i0]), __float_as_uint(s[i1])};
  if (MODE == 1) {
    uint32_t bl[2] = {__float_as_uint(s[lo + i0]), __float_as_uint(s[lo + i1])};
    // the k-step's three products summed on the tensor cores from zero, then added in float32: the tensor cores'
    // own accumulation is not IEEE (it does not round to nearest), and over a long sum that, not the operand
    // split, is what separates 3xTF32 from float32
    float t[4] = {0.f, 0.f, 0.f, 0.f};
    mmaTf32(t, A.l[0], A.l[1], A.l[2], A.l[3], bh[0], bh[1]);
    mmaTf32(t, A.h[0], A.h[1], A.h[2], A.h[3], bl[0], bl[1]);
    mmaTf32(t, A.h[0], A.h[1], A.h[2], A.h[3], bh[0], bh[1]);
    c[0] += t[0]; c[1] += t[1]; c[2] += t[2]; c[3] += t[3];
    return;
  }
  mmaTf32(c, A.h[0], A.h[1], A.h[2], A.h[3], bh[0], bh[1]);
}

// the tile's additive logit terms, [64 queries][64 keys] at row stride BC + PADK: the key mask and both biases,
// read along the key axis (coalesced), -inf where the key is past K, the query past Q, or the chain mask excludes it.
// (mask + bias0 + bias1, then added to the scaled dot product: the reference adds them one at a time onto it,
// which differs in rounding only where both biases are present.)
__device__ __forceinline__ void loadAdd(float* T, const Sdpa& a, int b, int h, int q0, int k0) {
  static_assert(THREADS == 128 && BC == 64, "a thread owns 4 keys of a 64-key row, 8 rows a pass");
  const int c4 = (threadIdx.x & 15) * 4, r0 = threadIdx.x >> 4;
  float km[4];
  int sk[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {          // this thread's keys: the same for every row it writes
    int k = k0 + c4 + j;
    bool in = k < a.K;
    km[j] = !in ? -INFINITY : a.keyMask ? 1e9f * (a.keyMask[(size_t)b * a.K + k] - 1.f) : 0.f;
    sk[j] = in && a.seqK ? a.seqK[k] : 0;
  }
  const float* bp[2];
  bool vec[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    bp[j] = a.bias[j].p ? a.bias[j].p + b * a.bias[j].sb + h * a.bias[j].sh + k0 + c4 : nullptr;
    vec[j] = bp[j] && k0 + c4 + 4 <= a.K && a.bias[j].sq % 4 == 0 && ((uintptr_t)bp[j] & 15) == 0;
  }
#pragma unroll
  for (int r = r0; r < BR; r += THREADS / 16) {
    const int q = q0 + r;
    float v[4];
    if (q >= a.Q) {
      v[0] = v[1] = v[2] = v[3] = -INFINITY;
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] = km[j];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        if (!bp[i]) continue;
        const float* p = bp[i] + (long long)q * a.bias[i].sq;
        if (vec[i]) {
          float4 x = *reinterpret_cast<const float4*>(p);
          v[0] += x.x; v[1] += x.y; v[2] += x.z; v[3] += x.w;
        } else {
#pragma unroll
          for (int j = 0; j < 4; ++j) if (k0 + c4 + j < a.K) v[j] += p[j];
        }
      }
      if (a.seqQ) {
        const int sq = a.seqQ[q];
#pragma unroll
        for (int j = 0; j < 4; ++j) if (sk[j] != sq) v[j] = -INFINITY;
      }
    }
    *reinterpret_cast<float4*>(T + r * (BC + PADK) + c4) = make_float4(v[0], v[1], v[2], v[3]);
  }
}
__device__ __forceinline__ float logit(float s, float add, const Sdpa& a) {
  if (add == -INFINITY) return -INFINITY;
  float v = s * a.scale + add;
  return a.clip ? fminf(fmaxf(v, -1e8f), 1e8f) : v;
}

// rows [r0, r0 + 64) of x[b] (row stride ld) into s[64][D + PADK]; zero past `rows`; split in NS parts for the B
// operand
template <int D, int NS>
__device__ __forceinline__ void loadTile(float* s, const float* x, long long ld, int r0, int rows) {
  constexpr size_t LO = 64 * (D + PADK);
  for (int i = threadIdx.x; i < 64 * D; i += THREADS) {
    int r = i / D, d = i % D;
    storeSplit<NS>(s + r * (D + PADK) + d, LO, r0 + r < rows ? x[(long long)(r0 + r) * ld + d] : 0.f);
  }
}

// grid: (query blocks) x H x B flattened; o [B, Q, ldo] at column h D; lse [B, H, Q]
template <int D, int MODE>
__global__ void __launch_bounds__(THREADS) fwdK(Sdpa a, const float* __restrict__ q, int ldq,
                                                const float* __restrict__ k, int ldk, const float* __restrict__ v,
                                                int ldv, float* __restrict__ o, int ldo, float* __restrict__ lse) {
  constexpr bool SPLIT = MODE == 1, BF = MODE == 2;
  extern __shared__ __align__(16) float sm[];
  constexpr size_t LO = BC * (D + PADK);           // a split tile's lo half, after its hi
  float* Ks = sm;                                  // [BC][D + PADK], split
  float* Vs = Ks + (SPLIT ? 3 : 1) * LO;           // (K in three parts: the score product's)
  float* T = Vs + (SPLIT ? 2 : 1) * LO;            // [BR][BC + PADK]: the additive terms, then P
  const int nqb = (a.Q + BR - 1) / BR;
  const int qb = blockIdx.x % nqb, h = (blockIdx.x / nqb) % a.H, b = blockIdx.x / (nqb * a.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int q0 = qb * BR, qr0 = q0 + warp * 16;   // the block's and this warp's first query
  const float* qb_ = q + (size_t)b * a.Q * ldq + h * D;
  const float* kb_ = k + (size_t)b * a.K * ldk + h * D;
  const float* vb_ = v + (size_t)b * a.K * ldv + h * D;
  float qa[D / 8][4];
#pragma unroll
  for (int s = 0; s < D / 8; ++s)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int r = qr0 + g + (i & 1) * 8, c = s * 8 + t + (i >> 1) * 4;
      qa[s][i] = r < a.Q ? qb_[(long long)r * ldq + c] : 0.f;
    }
  float oacc[D / 8][4];
#pragma unroll
  for (int n = 0; n < D / 8; ++n) oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f;
  float mx[2] = {-INFINITY, -INFINITY}, sum[2] = {0.f, 0.f};
  float* P = T + warp * 16 * (BC + PADK);          // this warp's 16 rows
  for (int k0 = 0; k0 < a.K; k0 += BC) {
    __syncthreads();
    loadTile<D, SPLIT ? 3 : BF ? 0 : 1>(Ks, kb_, ldk, k0, a.K);
    loadTile<D, SPLIT ? 2 : BF ? 0 : 1>(Vs, vb_, ldv, k0, a.K);
    loadAdd(T, a, b, h, q0, k0);
    __syncthreads();
    float sacc[BC / 8][4];
#pragma unroll
    for (int j = 0; j < BC / 8; ++j) sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0.f;
#pragma unroll
    for (int s = 0; s < D / 8; ++s) {
      if constexpr (SPLIT) {
        const AFrag3 A(qa[s]);
#pragma unroll
        for (int j = 0; j < BC / 8; ++j)
          mma3(sacc[j], A, Ks, LO, (j * 8 + g) * (D + PADK) + s * 8 + t, (j * 8 + g) * (D + PADK) + s * 8 + t + 4);
      } else {
        const AFrag<MODE> A(qa[s]);
#pragma unroll
        for (int j = 0; j < BC / 8; ++j)
          mma<MODE>(sacc[j], A, Ks, LO, (j * 8 + g) * (D + PADK) + s * 8 + t, (j * 8 + g) * (D + PADK) + s * 8 + t + 4);
      }
    }
    float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int j = 0; j < BC / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int r = e >> 1;
        float x = logit(sacc[j][e], P[(g + r * 8) * (BC + PADK) + j * 8 + 2 * t + (e & 1)], a);
        sacc[j][e] = x;
        rmax[r] = fmaxf(rmax[r], x);
      }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(~0u, rmax[r], 1));
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(~0u, rmax[r], 2));
    }
    float alpha[2], nm[2], rs[2] = {0.f, 0.f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      nm[r] = fmaxf(mx[r], rmax[r]);
      alpha[r] = nm[r] == -INFINITY ? 1.f : expf(mx[r] - nm[r]);
      mx[r] = nm[r];
    }
#pragma unroll
    for (int j = 0; j < BC / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int r = e >> 1;
        float p = sacc[j][e] == -INFINITY ? 0.f : expf(sacc[j][e] - nm[r]);
        rs[r] += p;
        P[(g + r * 8) * (BC + PADK) + j * 8 + 2 * t + (e & 1)] = p;
      }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      rs[r] += __shfl_xor_sync(~0u, rs[r], 1);
      rs[r] += __shfl_xor_sync(~0u, rs[r], 2);
      sum[r] = sum[r] * alpha[r] + rs[r];
    }
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      oacc[n][0] *= alpha[0]; oacc[n][1] *= alpha[0]; oacc[n][2] *= alpha[1]; oacc[n][3] *= alpha[1];
    }
    __syncwarp();
#pragma unroll
    for (int s = 0; s < BC / 8; ++s) {
      float pa[4] = {P[g * (BC + PADK) + s * 8 + t], P[(g + 8) * (BC + PADK) + s * 8 + t],
                     P[g * (BC + PADK) + s * 8 + t + 4], P[(g + 8) * (BC + PADK) + s * 8 + t + 4]};
      const AFrag<MODE> A(pa);
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        mma<MODE>(oacc[n], A, Vs, LO, (s * 8 + t) * (D + PADK) + n * 8 + g, (s * 8 + t + 4) * (D + PADK) + n * 8 + g);
      }
    }
  }
  float* ob = o + (size_t)b * a.Q * ldo + h * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int qq = qr0 + g + r * 8;
    if (qq >= a.Q) continue;
    float inv = sum[r] > 0.f ? 1.f / sum[r] : 0.f;
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      ob[(long long)qq * ldo + n * 8 + 2 * t] = oacc[n][r * 2] * inv;
      ob[(long long)qq * ldo + n * 8 + 2 * t + 1] = oacc[n][r * 2 + 1] * inv;
    }
    // the row's max and the log of its sum apart (lse [2][B, H, Q]): summed into one float the log is lost
    // whenever the logits are large - a row masked whole is -1e8 everywhere, and -1e8 + log K is -1e8 in float32,
    // which made every p of that row 1 instead of 1 / K
    if (t == 0) {
      const size_t i = ((size_t)b * a.H + h) * a.Q + qq, N = (size_t)a.B * a.H * a.Q;
      lse[i] = sum[r] > 0.f ? mx[r] : -INFINITY;
      lse[N + i] = sum[r] > 0.f ? logf(sum[r]) : 0.f;
    }
  }
}

// Di[b, h, q] = sum_d dO o (the softmax backward's row term); 8 lanes a row, a float4 each where D allows
__global__ void rowDotK(const float* __restrict__ dO, int lddo, const float* __restrict__ o, int ldo, float* Di, int B,
                        int H, int Q, int D) {
  size_t row = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 8;
  int lane = threadIdx.x & 7;
  if (row >= (size_t)B * H * Q) return;
  int q = row % Q, h = (row / Q) % H, b = (int)(row / ((size_t)Q * H));
  const float* x = dO + ((size_t)b * Q + q) * lddo + h * D;
  const float* y = o + ((size_t)b * Q + q) * ldo + h * D;
  float s = 0.f;
  if (D % 4 == 0 && lddo % 4 == 0 && ldo % 4 == 0 && ((uintptr_t)x & 15) == 0 && ((uintptr_t)y & 15) == 0) {
    for (int d = lane * 4; d < D; d += 32) {
      float4 u = *reinterpret_cast<const float4*>(x + d), w = *reinterpret_cast<const float4*>(y + d);
      s += u.x * w.x + u.y * w.y + u.z * w.z + u.w * w.w;
    }
  } else {
    for (int d = lane; d < D; d += 8) s += x[d] * y[d];
  }
  for (int off = 4; off; off >>= 1) s += __shfl_xor_sync(~0u, s, off);
  if (!lane) Di[row] = s;
}
inline void rowDot(const float* dO, int lddo, const float* o, int ldo, float* Di, int B, int H, int Q, int D) {
  rowDotK<<<blocks((size_t)B * H * Q * 8), 256, 0, STREAM>>>(dO, lddo, o, ldo, Di, B, H, Q, D);
}
__global__ void zeroColsK(float* x, int ld, size_t rows, int C) {   // x[r, 0:C] = 0
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * C) x[(i / C) * ld + i % C] = 0.f;
}

// grid: (key blocks) x H x B flattened. Each warp owns 16 keys; per query block of 64, in two halves of 32 queries:
//   S^T = K Q^T, P^T = exp(S^T - lse), dP^T = V dO^T, dS^T = P^T (dP^T - Di)
//   dV += P^T dO, dK += scale dS^T Q (registers)
// then for the whole block: dQ += scale dS K and the bias gradients += dS, both through shared memory and coalesced
// atomics. T holds the tile [query][key]: the additive terms, overwritten in place by P, then by dS. (Double-
// buffering the query blocks with cp.async was measured: the second stage costs the third block an SM, and the
// overlap does not buy it back - 1.48 -> 1.66 ms on AF2's triangle attention, TF32.)
template <int D, int MODE>
__global__ void __launch_bounds__(THREADS, MODE == 1 ? 2 : 3) bwdK(Sdpa a, const float* __restrict__ q, int ldq,
                                                   const float* __restrict__ k, int ldk, const float* __restrict__ v,
                                                   int ldv, const float* __restrict__ dO, int lddo,
                                                   const float* __restrict__ lse, const float* __restrict__ Di,
                                                   float* __restrict__ dq, int lddq, float* __restrict__ dk, int lddk,
                                                   float* __restrict__ dv, int lddv, bool accK, bool accV) {
  constexpr bool SPLIT = MODE == 1, BF = MODE == 2;
  extern __shared__ __align__(16) float sm[];
  constexpr int LD = D + PADK, LT = BC + PADK;
  constexpr size_t LO = 64 * LD;                   // a split tile's lo half, after its hi
  float* Ks = sm;                                  // [BC][LD], split
  float* Qs = Ks + (SPLIT ? 2 : 1) * LO;           // [BR][LD], split (in three parts: the score product's)
  float* Os = Qs + (SPLIT ? 3 : 1) * LO;           // dO [BR][LD], split; then the block's dQ (hi half, plain floats)
  float* T = Os + (SPLIT ? 2 : 1) * LO;            // [BR][LT]
  float* Ls = T + BR * LT;                         // lse [BR]
  float* Ds = Ls + BR;                             // Di [BR]
  float* Lg = Ds + BR;                             // log sum [BR]
  const int nkb = (a.K + BC - 1) / BC;
  const int kb = blockIdx.x % nkb, h = (blockIdx.x / nkb) % a.H, b = blockIdx.x / (nkb * a.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int k0 = kb * BC, kl0 = warp * 16;         // the block's first key; this warp's first, within the block
  const float* qb_ = q + (size_t)b * a.Q * ldq + h * D;
  const float* ob_ = dO + (size_t)b * a.Q * lddo + h * D;
  const float* vb_ = v + (size_t)b * a.K * ldv + h * D;
  const float* kb_ = k + (size_t)b * a.K * ldk + h * D;
  loadTile<D, SPLIT ? 2 : BF ? 0 : 1>(Ks, kb_, ldk, k0, a.K);
  __syncthreads();
  float ka[D / 8][4], va[D / 8][4];
#pragma unroll
  for (int s = 0; s < D / 8; ++s)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int r = kl0 + g + (i & 1) * 8, c = s * 8 + t + (i >> 1) * 4;
      ka[s][i] = k0 + r < a.K ? kb_[(long long)(k0 + r) * ldk + c] : 0.f;
      va[s][i] = k0 + r < a.K ? vb_[(long long)(k0 + r) * ldv + c] : 0.f;
    }
  float dka[D / 8][4], dva[D / 8][4];
#pragma unroll
  for (int n = 0; n < D / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dka[n][e] = dva[n][e] = 0.f;
  const float* lseb = lse + ((size_t)b * a.H + h) * a.Q;
  const float* lgb = lseb + (size_t)a.B * a.H * a.Q;
  const float* Dib = Di + ((size_t)b * a.H + h) * a.Q;
  float* dqb = dq + (size_t)b * a.Q * lddq + h * D;
  real* g0 = a.bias[0].grad ? a.bias[0].grad + b * a.bias[0].sb + h * a.bias[0].sh + k0 : nullptr;
  real* g1 = a.bias[1].grad ? a.bias[1].grad + b * a.bias[1].sb + h * a.bias[1].sh + k0 : nullptr;
  for (int q0 = 0; q0 < a.Q; q0 += BR) {
    __syncthreads();
    loadTile<D, SPLIT ? 3 : BF ? 0 : 1>(Qs, qb_, ldq, q0, a.Q);
    loadTile<D, SPLIT ? 2 : BF ? 0 : 1>(Os, ob_, lddo, q0, a.Q);
    loadAdd(T, a, b, h, q0, k0);
    for (int i = threadIdx.x; i < BR; i += THREADS) {
      Ls[i] = q0 + i < a.Q ? lseb[q0 + i] : INFINITY;
      Lg[i] = q0 + i < a.Q ? lgb[q0 + i] : 0.f;
      Ds[i] = q0 + i < a.Q ? Dib[q0 + i] : 0.f;
    }
    __syncthreads();
#pragma unroll 1
    for (int hq = 0; hq < 2; ++hq) {
      float st[4][4], dpt[4][4];   // 16 keys x 32 queries
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) st[jj][e] = dpt[jj][e] = 0.f;
#pragma unroll
      for (int s = 0; s < D / 8; ++s) {
        const AFrag<MODE> Va(va[s]);
        if constexpr (SPLIT) {
          const AFrag3 Ka(ka[s]);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const int j = hq * 4 + jj;
            mma3(st[jj], Ka, Qs, LO, (j * 8 + g) * LD + s * 8 + t, (j * 8 + g) * LD + s * 8 + t + 4);
          }
        } else {
          const AFrag<MODE> Ka(ka[s]);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const int j = hq * 4 + jj;
            mma<MODE>(st[jj], Ka, Qs, LO, (j * 8 + g) * LD + s * 8 + t, (j * 8 + g) * LD + s * 8 + t + 4);
          }
        }
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int j = hq * 4 + jj;
          mma<MODE>(dpt[jj], Va, Os, LO, (j * 8 + g) * LD + s * 8 + t, (j * 8 + g) * LD + s * 8 + t + 4);
        }
      }
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          int kl = kl0 + g + (e >> 1) * 8, ql = (hq * 4 + jj) * 8 + 2 * t + (e & 1);
          float x = logit(st[jj][e], T[ql * LT + kl], a);
          float p = x == -INFINITY ? 0.f : expf((x - Ls[ql]) - Lg[ql]);
          T[ql * LT + kl] = p;
          dpt[jj][e] = p * (dpt[jj][e] - Ds[ql]);
        }
      __syncwarp();
      // dV += P^T dO over this half's queries: A = P^T (keys x queries), B = dO (queries x D)
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int qs = hq * 32 + s * 8;
        float pa[4] = {T[(qs + t) * LT + kl0 + g], T[(qs + t) * LT + kl0 + g + 8], T[(qs + t + 4) * LT + kl0 + g],
                       T[(qs + t + 4) * LT + kl0 + g + 8]};
        const AFrag<MODE> A(pa);
#pragma unroll
        for (int n = 0; n < D / 8; ++n) {
          mma<MODE>(dva[n], A, Os, LO, (qs + t) * LD + n * 8 + g, (qs + t + 4) * LD + n * 8 + g);
        }
      }
      __syncwarp();
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          T[((hq * 4 + jj) * 8 + 2 * t + (e & 1)) * LT + kl0 + g + (e >> 1) * 8] = dpt[jj][e];
      __syncwarp();
      // dK += dS^T Q
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int qs = hq * 32 + s * 8;
        float pa[4] = {T[(qs + t) * LT + kl0 + g], T[(qs + t) * LT + kl0 + g + 8], T[(qs + t + 4) * LT + kl0 + g],
                       T[(qs + t + 4) * LT + kl0 + g + 8]};
        const AFrag<MODE> A(pa);
#pragma unroll
        for (int n = 0; n < D / 8; ++n) {
          mma<MODE>(dka[n], A, Qs, LO, (qs + t) * LD + n * 8 + g, (qs + t + 4) * LD + n * 8 + g);
        }
      }
    }
    __syncthreads();
    // the bias gradients, coalesced along the key axis
    if (g0 || g1)
      for (int i = threadIdx.x; i < BR * BC; i += THREADS) {
        int ql = i / BC, kl = i % BC;
        float ds = T[ql * LT + kl];
        if (ds != 0.f) {
          long long qq = q0 + ql;
          if (g0) atomicAdd(g0 + qq * a.bias[0].sq + kl, ds);
          if (g1) atomicAdd(g1 + qq * a.bias[1].sq + kl, ds);
        }
      }
    // dQ for this warp's 16 queries over the block's 64 keys: A = dS (queries x keys), B = K (keys x D), staged in Os
    const int qw = warp * 16;
    float dqa[D / 8][4];
#pragma unroll
    for (int n = 0; n < D / 8; ++n) dqa[n][0] = dqa[n][1] = dqa[n][2] = dqa[n][3] = 0.f;
#pragma unroll
    for (int s = 0; s < BC / 8; ++s) {
      float pa[4] = {T[(qw + g) * LT + s * 8 + t], T[(qw + g + 8) * LT + s * 8 + t], T[(qw + g) * LT + s * 8 + t + 4],
                     T[(qw + g + 8) * LT + s * 8 + t + 4]};
      const AFrag<MODE> A(pa);
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        mma<MODE>(dqa[n], A, Ks, LO, (s * 8 + t) * LD + n * 8 + g, (s * 8 + t + 4) * LD + n * 8 + g);
      }
    }
#pragma unroll
    for (int n = 0; n < D / 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) Os[(qw + g + (e >> 1) * 8) * LD + n * 8 + 2 * t + (e & 1)] = dqa[n][e] * a.scale;
    __syncwarp();
    for (int i = lane; i < 16 * D; i += 32) {
      int r = i / D, d = i % D, qq = q0 + qw + r;
      if (qq < a.Q) atomicAdd(dqb + (long long)qq * lddq + d, Os[(qw + r) * LD + d]);
    }
  }
  // this warp's 16 keys of dK and dV
  float* dkb = dk + (size_t)b * a.K * lddk + h * D;
  float* dvb = dv + (size_t)b * a.K * lddv + h * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int kk = k0 + kl0 + g + r * 8;
    if (kk >= a.K) continue;
#pragma unroll
    for (int n = 0; n < D / 8; ++n)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        float* pk = dkb + (long long)kk * lddk + n * 8 + 2 * t + c;
        float* pv = dvb + (long long)kk * lddv + n * 8 + 2 * t + c;
        float xk = dka[n][r * 2 + c] * a.scale, xv = dva[n][r * 2 + c];
        *pk = accK ? *pk + xk : xk;
        *pv = accV ? *pv + xv : xv;
      }
  }
}

// ---- bf16: its own kernels, not MODE 2 of the ones above - FlashAttention-2's structure. q, k, v (and dO) are first
// copied to bf16, head-major and padded ([B, H, N rounded to 64, D rounded to 16], zeros past the ends), so a tile
// is one contiguous block that cp.async moves into shared memory with no conversion and no bounds checks; the next
// tile and its pair-bias tile load while this one computes (two stages). The [q][k] biases are summed into one bf16
// tile the same way (bf16 is what a bf16 run's bias is: autocast's projection makes it so), and the key mask and any
// per-key bias are added per key from a 64-entry vector. The products are m16n8k16; P and dS go from the
// accumulators straight into the next product's A operand (an m16n8k16 C pair of n-tiles is an A fragment), so P
// never touches shared memory; a B operand contracted over a tile's rows is read with ldmatrix.trans from the one
// [row][D] copy. Shared memory is what decides three CTAs an SM or two, and two cannot hide the MMA latency.
__device__ __forceinline__ void mma16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t word(const uint16_t* s, int i) { return *reinterpret_cast<const uint32_t*>(s + i); }
// the B fragment of rows k .. k + 15 and columns n .. n + 7 of a row-major bf16 tile (row stride ld): ldmatrix.trans
__device__ __forceinline__ void ldT(uint32_t& b0, uint32_t& b1, const uint16_t* s, int k, int n, int ld) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(s + (k + (threadIdx.x & 15)) * ld + n);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(a));
}
// four 8 x 8 bf16 tiles transposed (lane l addresses row l % 8 of tile l / 8; tile j lands in r[j])
__device__ __forceinline__ void ldT4(uint32_t* r, const uint16_t* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void cp16(void* d, const void* s, bool on) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(d);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(a), "l"(s), "r"(on ? 16 : 0));
}
__device__ __forceinline__ void cp4(void* d, const void* s, bool on) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(d);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(a), "l"(s), "r"(on ? 4 : 0));
}
__device__ __forceinline__ void cpCommit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cpWait1() { asm volatile("cp.async.wait_group 1;"); }
// two accumulator n-tiles (8 columns each) as one A fragment over their 16 columns
__device__ __forceinline__ void cToA(uint32_t* f, const float* c0, const float* c1) {
  f[0] = bf2(c0[0], c0[1]); f[1] = bf2(c0[2], c0[3]); f[2] = bf2(c1[0], c1[1]); f[3] = bf2(c1[2], c1[3]);
}
// rows r, r + 8 and columns c .. c + 15 of a head-major [rows][DP] bf16 block, as an A fragment
template <int DP>
__device__ __forceinline__ void aFragH(uint32_t* f, const uint16_t* x, int r, int c) {
#pragma unroll
  for (int i = 0; i < 4; ++i) f[i] = word(x, (r + (i & 1) * 8) * DP + c + (i >> 1) * 8);
}
// where a [B, N, H, D] operand's rows are: a head-major padded copy (headsBf), or the projection's own row-major
// bf16 output read in place (16-byte rows: ld a multiple of 8) - element (b, h, n, c) at p + b sB + h sH + n sN + c,
// zero where n >= N or c >= W
struct HSrc { const uint16_t* p = nullptr; long long sB = 0, sH = 0, sN = 0; int N = 0, W = 0; };
__device__ __forceinline__ const uint16_t* hrow(const HSrc& s, int b, int h, int n) {
  return s.p + b * s.sB + h * s.sH + (long long)n * s.sN;
}
// a 64-row tile of it into shared memory [64][ld] (cp.async, zero-filled past the ends)
template <int DP, int ROWS = 64, bool UNROLL = true>
__device__ __forceinline__ void loadH(uint16_t* dst, int ld, const HSrc& s, int b, int h, int n0) {
  if constexpr (!UNROLL) {   // (the forward's: unrolled, its D 32 tile lost 2% at n 768)
    for (int i = threadIdx.x; i < ROWS * DP / 8; i += THREADS) {
      const int r = i / (DP / 8), c = (i % (DP / 8)) * 8;
      const bool on = n0 + r < s.N && c < s.W;
      cp16(dst + r * ld + c, on ? hrow(s, b, h, n0 + r) + c : s.p, on);
    }
    return;
  }
  // (unrolled over a thread's chunks from one row pointer: the address arithmetic was 6% of the backward's issue)
  constexpr int CH = DP / 8, NS = (ROWS * CH + THREADS - 1) / THREADS;
  const uint16_t* base = hrow(s, b, h, n0);
#pragma unroll
  for (int j = 0; j < NS; ++j) {
    const int i = threadIdx.x + j * THREADS, r = i / CH, c = (i % CH) * 8;
    if (ROWS * CH % THREADS != 0 && i >= ROWS * CH) break;
    const bool on = n0 + r < s.N && c < s.W;
    cp16(dst + r * ld + c, on ? base + (long long)r * s.sN + c : s.p, on);
  }
}
// rows r, r + 8 and columns c .. c + 15 of it as an A fragment
__device__ __forceinline__ void aFragS(uint32_t* f, const HSrc& s, int b, int h, int r, int c) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int rr = r + (i & 1) * 8, cc = c + (i >> 1) * 8;
    f[i] = rr < s.N && cc < s.W ? *reinterpret_cast<const uint32_t*>(hrow(s, b, h, rr) + cc) : 0u;
  }
}
template <int D> __host__ __device__ constexpr int padD() { return (D + 15) / 16 * 16; }
template <int D> __host__ __device__ constexpr int ldBf() { return padD<D>() + 8; }   // a tile's row stride (bf16)
// x [B, N, ld] at head h's D columns -> out [B, H, Np, DP] bf16, zero past N and D (two columns a thread)
__device__ __forceinline__ uint32_t bfPair(const float* p) { return bf2(p[0], p[1]); }
__device__ __forceinline__ uint32_t bfPair(const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); }
template <class X>   // (X: float, or bf16 where the projections wrote it)
__global__ void headsBfK(const X* __restrict__ x, int ld, uint32_t* __restrict__ out, int N, int H, int D, int Np,
                         int DP2, size_t total) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int c = 2 * (int)(i % DP2);
  const size_t row = i / DP2;
  const int n = (int)(row % Np), h = (int)((row / Np) % H);
  const size_t b = row / ((size_t)Np * H);
  uint32_t r = 0;
  if (n < N && c < D) r = bfPair(x + (b * N + n) * ld + (size_t)h * D + c);
  out[i] = r;
}
template <class X>
inline const uint16_t* headsBf(const X* x, int ld, int B, int N, int H, int D, int Np, int DP) {
  const size_t total = (size_t)B * H * Np * (DP / 2);
  uint32_t* out = reinterpret_cast<uint32_t*>(ARENA.get(total));
  headsBfK<<<blocks(total), 256, 0, STREAM>>>(x, ld, out, N, H, D, Np, DP / 2, total);
  return reinterpret_cast<const uint16_t*>(out);
}
// the [q][k] biases (sq != 0) summed into one bf16 tile [Bb][Hh][Qp][Kp] (Bb, Hh 1 where every such bias
// broadcasts), zero past Q and K
struct BiasTile { const uint16_t* p = nullptr; long long sb = 0, sh = 0; int ld = 0; };
__global__ void biasBfK(Sdpa a, uint16_t* __restrict__ out, int Bb, int Hh, int Qp, int Kp) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)Bb * Hh * Qp * Kp) return;
  const int k = (int)(i % Kp), q = (int)((i / Kp) % Qp), h = (int)((i / ((size_t)Kp * Qp)) % Hh);
  const int b = (int)(i / ((size_t)Kp * Qp * Hh));
  float v = 0.f;
  if (q < a.Q && k < a.K)
    for (int j = 0; j < 2; ++j) {
      const Bias& B = a.bias[j];
      if (B.p && B.sq != 0) v += B.p[b * B.sb + h * B.sh + (long long)q * B.sq + k];
    }
  out[i] = (uint16_t)(bf2(v, 0.f) & 0xffff);
}
inline BiasTile biasBf(const Sdpa& a, int Qp, int Kp) {
  BiasTile t;
  bool any = false, perB = false, perH = false;
  for (int j = 0; j < 2; ++j) {
    const Bias& B = a.bias[j];
    if (B.p && B.sq != 0) { any = true; perB |= B.sb != 0; perH |= B.sh != 0; }
  }
  if (!any) return t;
  const int Bb = perB ? a.B : 1, Hh = perH ? a.H : 1;
  const size_t n = (size_t)Bb * Hh * Qp * Kp;
  uint16_t* out = reinterpret_cast<uint16_t*>(ARENA.get((n + 1) / 2));
  biasBfK<<<blocks(n), 256, 0, STREAM>>>(a, out, Bb, Hh, Qp, Kp);
  t.p = out; t.sb = perB ? (long long)Hh * Qp * Kp : 0; t.sh = perH ? (long long)Qp * Kp : 0; t.ld = Kp;
  return t;
}
// 2^x in one MUFU instruction (2^-inf = 0); the exponentials below are taken in base 2
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
constexpr float LOG2E = 1.4426950408889634f;
__device__ __forceinline__ float bfToF(uint16_t x) { return __uint_as_float((uint32_t)x << 16); }
// key k's additive term (the mask and every per-key bias; -inf past K)
__device__ __forceinline__ float keyAdd(const Sdpa& a, int b, int h, int k) {
  if (k >= a.K) return -INFINITY;
  float v = a.keyMask ? 1e9f * (a.keyMask[(size_t)b * a.K + k] - 1.f) : 0.f;
  for (int j = 0; j < 2; ++j) {
    const Bias& B = a.bias[j];
    if (B.p && B.sq == 0) v += B.p[b * B.sb + h * B.sh + k];
  }
  return v;
}
constexpr int LB = BC + 8;   // a bf16 [q][k] tile's row stride

__device__ __forceinline__ float ldOut(const float* p) { return *p; }
__device__ __forceinline__ float ldOut(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void stOut(float* p, float v) { *p = v; }
__device__ __forceinline__ void stOut(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
// two adjacent elements as one store where the address allows (an even column of an even row stride)
__device__ __forceinline__ void st2(float* p, float a, float b) {
  if (((uintptr_t)p & 7) == 0) *reinterpret_cast<float2*>(p) = make_float2(a, b); else { p[0] = a; p[1] = b; }
}
__device__ __forceinline__ void st2(__nv_bfloat16* p, float a, float b) {
  if (((uintptr_t)p & 3) == 0) *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
  else { p[0] = __float2bfloat16(a); p[1] = __float2bfloat16(b); }
}
template <int D, class OT = float>   // (OT: o written float, or bf16 for a module that keeps it so)
__global__ void __launch_bounds__(THREADS) fwdBf(Sdpa a, BiasTile bt, HSrc qs_, HSrc ks_, HSrc vs_,
                                                 OT* __restrict__ o, int ldo,
                                                 float* __restrict__ lse) {
  constexpr int DP = padD<D>(), LK = ldBf<D>();
  extern __shared__ __align__(16) float sm[];
  uint16_t* Kb = reinterpret_cast<uint16_t*>(sm);                 // [2][BC][LK]
  uint16_t* Vb = Kb + 2 * BC * LK;                                // [2][BC][LK]
  uint16_t* Tb = Vb + 2 * BC * LK;                                // [2][BR][LB]: the bias tile
  float* KA = reinterpret_cast<float*>(Tb + 2 * BR * LB);         // [2][BC]: each key's additive term
  int* KS = reinterpret_cast<int*>(KA + 2 * BC);                  // [2][BC]: seqK
  const int nqb = (a.Q + BR - 1) / BR;
  const int qb = blockIdx.x % nqb, h = (blockIdx.x / nqb) % a.H, b = blockIdx.x / (nqb * a.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int q0 = qb * BR, qr0 = q0 + warp * 16;
  const size_t bh = (size_t)b * a.H + h;
  const uint16_t* tp = bt.p ? bt.p + b * bt.sb + h * bt.sh + (size_t)q0 * bt.ld : nullptr;
  auto issue = [&](int st, int k0) {
    loadH<DP, BC, false>(Kb + st * BC * LK, LK, ks_, b, h, k0);
    loadH<DP, BC, false>(Vb + st * BC * LK, LK, vs_, b, h, k0);
    if (tp) {
      uint16_t* T = Tb + st * BR * LB;
      for (int i = threadIdx.x; i < BR * BC / 8; i += THREADS) {
        const int r = i / (BC / 8), c = (i % (BC / 8)) * 8;
        cp16(T + r * LB + c, tp + (size_t)r * bt.ld + k0 + c, true);
      }
    }
  };
  issue(0, 0);
  cpCommit();
  if (threadIdx.x < BC) {
    KA[threadIdx.x] = keyAdd(a, b, h, threadIdx.x);
    KS[threadIdx.x] = a.seqK && threadIdx.x < a.K ? a.seqK[threadIdx.x] : 0;
  }
  uint32_t qa[DP / 16][4];
#pragma unroll
  for (int s = 0; s < DP / 16; ++s) aFragS(qa[s], qs_, b, h, qr0 + g, s * 16 + 2 * t);
  int qs[2] = {0, 0};
  if (a.seqQ)
    for (int r = 0; r < 2; ++r) qs[r] = qr0 + g + r * 8 < a.Q ? a.seqQ[qr0 + g + r * 8] : 0;
  float oacc[D / 8][4];
#pragma unroll
  for (int n = 0; n < D / 8; ++n) oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f;
  float mx[2] = {-INFINITY, -INFINITY}, sum[2] = {0.f, 0.f};
  const int nk = (a.K + BC - 1) / BC;
  for (int it = 0; it < nk; ++it) {
    const int st = it & 1, k0 = it * BC;
    float nka = 0.f;
    int nks = 0;
    if (it + 1 < nk) {
      issue(st ^ 1, k0 + BC);
      if (threadIdx.x < BC) {   // the next tile's keys: loaded now, stored after this tile's arithmetic
        const int k = k0 + BC + threadIdx.x;
        nka = keyAdd(a, b, h, k);
        nks = a.seqK && k < a.K ? a.seqK[k] : 0;
      }
    }
    cpCommit();
    cpWait1();
    __syncthreads();
    const uint16_t* Ks = Kb + st * BC * LK;
    const uint16_t* Vs = Vb + st * BC * LK;
    const uint16_t* T = Tb + st * BR * LB + warp * 16 * LB;
    const float* ka = KA + st * BC;
    const int* ks = KS + st * BC;
    float sacc[BC / 8][4];
#pragma unroll
    for (int j = 0; j < BC / 8; ++j) sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0.f;
#pragma unroll
    for (int s = 0; s < DP / 16; ++s)
#pragma unroll
      for (int j = 0; j < BC / 8; ++j) {
        const int i = (j * 8 + g) * LK + s * 16 + 2 * t;
        mma16(sacc[j], qa[s], word(Ks, i), word(Ks, i + 8));
      }
    float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int j = 0; j < BC / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = e >> 1, kl = j * 8 + 2 * t + (e & 1);
        float add = ka[kl];
        if (add != -INFINITY) {
          if (a.seqQ && ks[kl] != qs[r]) add = -INFINITY;
          else if (tp) add += bfToF(T[(g + r * 8) * LB + kl]);
        }
        const float x = logit(sacc[j][e], add, a) * LOG2E;
        sacc[j][e] = x;
        rmax[r] = fmaxf(rmax[r], x);
      }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(~0u, rmax[r], 1));
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(~0u, rmax[r], 2));
    }
    float alpha[2], nm[2], rs[2] = {0.f, 0.f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      nm[r] = fmaxf(mx[r], rmax[r]);
      alpha[r] = nm[r] == -INFINITY ? 1.f : ex2(mx[r] - nm[r]);
      mx[r] = nm[r];
    }
#pragma unroll
    for (int j = 0; j < BC / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float p = sacc[j][e] == -INFINITY ? 0.f : ex2(sacc[j][e] - nm[e >> 1]);
        rs[e >> 1] += p;
        sacc[j][e] = p;
      }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      rs[r] += __shfl_xor_sync(~0u, rs[r], 1);
      rs[r] += __shfl_xor_sync(~0u, rs[r], 2);
      sum[r] = sum[r] * alpha[r] + rs[r];
    }
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      oacc[n][0] *= alpha[0]; oacc[n][1] *= alpha[0]; oacc[n][2] *= alpha[1]; oacc[n][3] *= alpha[1];
    }
#pragma unroll
    for (int u = 0; u < BC / 16; ++u) {
      uint32_t pa[4];
      cToA(pa, sacc[2 * u], sacc[2 * u + 1]);
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        uint32_t b0, b1;
        ldT(b0, b1, Vs, u * 16, n * 8, LK);
        mma16(oacc[n], pa, b0, b1);
      }
    }
    if (it + 1 < nk && threadIdx.x < BC) { KA[(st ^ 1) * BC + threadIdx.x] = nka; KS[(st ^ 1) * BC + threadIdx.x] = nks; }
    __syncthreads();   // this stage is the one the next iteration's issue overwrites
  }
  OT* ob = o + (size_t)b * a.Q * ldo + h * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int qq = qr0 + g + r * 8;
    if (qq >= a.Q) continue;
    const float inv = sum[r] > 0.f ? 1.f / sum[r] : 0.f;
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      st2(ob + (long long)qq * ldo + n * 8 + 2 * t, oacc[n][r * 2] * inv, oacc[n][r * 2 + 1] * inv);
    }
    if (t == 0) {
      const size_t i = bh * a.Q + qq, N = (size_t)a.B * a.H * a.Q;
      lse[i] = sum[r] > 0.f ? mx[r] / LOG2E : -INFINITY;   // (back in natural units)
      lse[N + i] = sum[r] > 0.f ? logf(sum[r]) : 0.f;
    }
  }
}
template <int D>
inline size_t fwdBfSmem() { return 2 * (4 * BC * ldBf<D>() + 2 * BR * LB) + 4 * (4 * BC); }

// the backward as bwdK lays it out (a CTA a key block, a warp 16 keys, the queries in halves of 32), in bf16:
// S^T = K Q^T and dP^T = V dO^T (A = K, V from registers; B = Q, dO read as words), dV += P^T dO and dK += dS^T Q
// (A = P^T, dS^T from the accumulators; B = dO, Q by ldmatrix.trans), dQ = dS K (A = dS, B = K by ldmatrix.trans).
// dS takes the bias tile's place in shared memory, element by element (the thread that reads a bias writes its dS),
// as bf16 - what a bf16 run's logits gradient is - for dQ and the bias gradients.
__global__ void narrowDqK(const float* x, int ldx, __nv_bfloat16* y, int ldy, int C, size_t rows) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * C) y[i / C * ldy + i % C] = __float2bfloat16(x[i / C * ldx + i % C]);
}
constexpr double FIXED = 1099511627776.0;   // 2^40: |a gradient| to 8e6, resolved to 1e-12
__global__ void fromFixedK(float* g, const unsigned long long* f, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && f[i]) g[i] += (float)((double)(long long)f[i] / FIXED);
}
// a thread-block cluster's rank, its barrier (split: arrive, then wait) and a 16-byte store into another CTA's shared
// memory (sm_90 and later; bwdBf's CR > 1)
__device__ __forceinline__ unsigned clusterRank() {
  unsigned r = 0;
#if __CUDA_ARCH__ >= 900
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
#endif
  return r;
}
__device__ __forceinline__ void clusterArrive(bool release = true) {
#if __CUDA_ARCH__ >= 900
  if (release) asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  else asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
#else
  __trap();
#endif
}
__device__ __forceinline__ void clusterWait() {
#if __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
}
// st.async: the store completes on the receiving CTA's mbarrier as transaction bytes - no fence on the sender
// (a release barrier waits out every atomic the warp has in flight: 14% of bwdBf's stalls when it carried the data)
__device__ __forceinline__ void stAsync4(const void* local, const void* mbar, unsigned rank, float x, float y, float z,
                                         float w) {
#if __CUDA_ARCH__ >= 900
  unsigned r, m;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"((unsigned)__cvta_generic_to_shared(local)), "r"(rank));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(m) : "r"((unsigned)__cvta_generic_to_shared(mbar)), "r"(rank));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(r), "f"(x), "f"(y), "f"(z), "f"(w), "r"(m) : "memory");
#endif
}
__device__ __forceinline__ void mbarInit(void* mbar, unsigned count) {
#if __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"((unsigned)__cvta_generic_to_shared(mbar)), "r"(count) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}
__device__ __forceinline__ void mbarExpect(void* mbar, unsigned bytes) {
#if __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"((unsigned)__cvta_generic_to_shared(mbar)), "r"(bytes) : "memory");
#endif
}
__device__ __forceinline__ void mbarWait(void* mbar, unsigned parity) {
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n .reg .pred p;\n MBW%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra MBW%=;\n}"
               ::"r"((unsigned)__cvta_generic_to_shared(mbar)), "r"(parity) : "memory");
#endif
}
template <int D, class DKV = float, int QT = 64, bool GEN = true, int CR = 1>   // (DKV: dK and dV written in bf16 for a
                                                  // module that consumes them so; QT: the query tile, 32 or 64 rows; GEN
                                                  // false: no chain ids and no fixed-point bias sums, compiled out - 5%
                                                  // of it; CR > 1: a cluster of CR CTAs, CR consecutive rows at one head
                                                  // and key block, sums the bias gradients over its rows through shared
                                                  // memory before the atomics - every bias with a gradient the same for
                                                  // every row, sb 0)
__global__ void __launch_bounds__(THREADS, 3) bwdBf(Sdpa a, BiasTile bt, HSrc qs_, HSrc ks_, HSrc vs_, HSrc os_,
                                                    const float* __restrict__ lse, const float* __restrict__ Di,
                                                    float* __restrict__ dq, int lddq, DKV* __restrict__ dk, int lddk,
                                                    DKV* __restrict__ dv, int lddv, bool accK, bool accV,
                                                    unsigned long long* fx0, unsigned long long* fx1,
                                                    uint16_t* __restrict__ dqo, int ldqo, unsigned* __restrict__ cnt) {
  const int* const seqQ = GEN ? a.seqQ : nullptr;
  const int* const seqK = GEN ? a.seqK : nullptr;
  if constexpr (!GEN) { fx0 = nullptr; fx1 = nullptr; }
  constexpr int DP = padD<D>(), LK = ldBf<D>();
  extern __shared__ __align__(16) float sm[];
  uint16_t* Kb = reinterpret_cast<uint16_t*>(sm);                 // [BC][LK]
  uint16_t* Qb = Kb + BC * LK;                                    // [2][QT][LK]
  uint16_t* Ob = Qb + 2 * QT * LK;                                // dO [2][QT][LK]
  uint16_t* Tb = Ob + 2 * QT * LK;                                // [2][QT][LB]: the bias tile
  uint16_t* Sb = Tb + 2 * QT * LB;                                // dS^T [BC][QT], 16-byte chunks XOR-swizzled by row
  float* QV = reinterpret_cast<float*>(Sb + BC * QT);             // [2][QT][4]: row max, log sum, Di, seqQ
  float4* Red = reinterpret_cast<float4*>(QV + 2 * 4 * QT);       // CR > 1: [CR][4 / CR][THREADS], what the cluster's
                                                                  // other CTAs send this one of the bias gradients
  unsigned long long* mbar = reinterpret_cast<unsigned long long*>(Red + 4 * THREADS);   // (its arrival: their bytes)
  const int nkb = (a.K + BC - 1) / BC;
  const int bx = blockIdx.x / CR;   // (CR > 1: the cluster's ranks the fastest index - its rows)
  const int kb = bx % nkb, h = (bx / nkb) % a.H, b = bx / (nkb * a.H) * CR + (int)(blockIdx.x % CR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int k0 = kb * BC, kl0 = warp * 16;
  const size_t bh = (size_t)b * a.H + h;
  const uint16_t* tp = bt.p ? bt.p + b * bt.sb + h * bt.sh + k0 : nullptr;
  const float* lseb = lse + bh * a.Q;
  const float* lgb = lseb + (size_t)a.B * a.H * a.Q;
  const float* Dib = Di + bh * a.Q;
  loadH<DP>(Kb, LK, ks_, b, h, k0);
  auto issue = [&](int st, int q0) {
    loadH<DP, QT>(Qb + st * QT * LK, LK, qs_, b, h, q0);
    loadH<DP, QT>(Ob + st * QT * LK, LK, os_, b, h, q0);
    if (tp) {
      uint16_t* T = Tb + st * QT * LB;
      for (int i = threadIdx.x; i < QT * BC / 8; i += THREADS) {
        const int r = i / (BC / 8), c = (i % (BC / 8)) * 8;
        cp16(T + r * LB + c, tp + (size_t)(q0 + r) * bt.ld + c, true);
      }
    }
    if (threadIdx.x < QT) {
      float* qv = QV + st * 4 * QT;
      const int j = threadIdx.x, q = q0 + j;
      const bool in = q < a.Q;
      cp4(qv + 4 * j, in ? lseb + q : lseb, in);
      cp4(qv + 4 * j + 1, in ? lgb + q : lgb, in);
      cp4(qv + 4 * j + 2, in ? Dib + q : Dib, in);
      if (seqQ) cp4(qv + 4 * j + 3, in ? seqQ + q : seqQ, in);
    }
  };
  issue(0, 0);
  cpCommit();
  float kaR[2];
  int ksR[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int k = k0 + kl0 + g + r * 8;
    kaR[r] = keyAdd(a, b, h, k);
    ksR[r] = seqK && k < a.K ? seqK[k] : 0;
  }
  // the clip as two bounds, the lower one -inf for a key that is out (so that the clip cannot bring it back)
  const float hiC = a.clip ? 1e8f : INFINITY;
  const float loC[2] = {kaR[0] == -INFINITY || !a.clip ? -INFINITY : -1e8f, kaR[1] == -INFINITY || !a.clip ? -INFINITY : -1e8f};
  const float ckL[2] = {kaR[0] * LOG2E, kaR[1] * LOG2E}, sl2 = a.scale * LOG2E;
  // dS^T's element (k, q): a row a key, its 16-byte chunks permuted by the row (the stores and ldmatrix both clean)
  auto sOff = [](int k, int q) { return k * QT + (((q >> 3) ^ (k & (QT / 8 - 1))) << 3) + (q & 7); };
  uint32_t ka[DP / 16][4], va[DP / 16][4];
#pragma unroll
  for (int s = 0; s < DP / 16; ++s) {
    aFragS(ka[s], ks_, b, h, k0 + kl0 + g, s * 16 + 2 * t);
    aFragS(va[s], vs_, b, h, k0 + kl0 + g, s * 16 + 2 * t);
  }
  float dka[D / 8][4], dva[D / 8][4];
#pragma unroll
  for (int n = 0; n < D / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dka[n][e] = dva[n][e] = 0.f;
  float* dqb = dq + (size_t)b * a.Q * lddq + h * D;
  real* g0 = a.bias[0].grad ? a.bias[0].grad + b * a.bias[0].sb + h * a.bias[0].sh + k0 : nullptr;
  real* g1 = a.bias[1].grad ? a.bias[1].grad + b * a.bias[1].sb + h * a.bias[1].sh + k0 : nullptr;
  // (the bias gradients in fixed point - bindcraft2's nbbFixed: an integer sum is the same in any order, so the
  // gradient is deterministic where float atomics differ run to run)
  unsigned long long* f0 = fx0 && g0 ? fx0 + (g0 - a.bias[0].grad) : nullptr;
  unsigned long long* f1 = fx1 && g1 ? fx1 + (g1 - a.bias[1].grad) : nullptr;
  const int nq = (a.Q + QT - 1) / QT;
  const bool red = CR > 1 && (g0 || g1);   // (the same for the whole cluster)
  const unsigned rk = CR > 1 ? clusterRank() : 0u;
  // (the mbarrier initialised and every CTA of the cluster running before any shared memory is written; then the
  // first of the relaxed arrivals that say "this CTA has read its buffer" - a sender waits for them before it writes)
  if (red) {
    if (threadIdx.x == 0) mbarInit(mbar, 1);
    clusterArrive(); clusterWait();
    clusterArrive(false);
  }
  int rc = 0;
  for (int it = 0; it < nq; ++it) {
    const int st = it & 1, q0 = it * QT;
    if (it + 1 < nq) issue(st ^ 1, q0 + QT);
    cpCommit();
    cpWait1();
    __syncthreads();
    const uint16_t* Qs = Qb + st * QT * LK;
    const uint16_t* Os = Ob + st * QT * LK;
    uint16_t* T = Tb + st * QT * LB;
    const float* qv = QV + st * 4 * QT;
#pragma unroll 1
    for (int hq = 0; hq < QT / 32; ++hq) {
      float sa[4][4], dpt[4][4];   // 16 keys x 32 queries
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) sa[jj][e] = dpt[jj][e] = 0.f;
#pragma unroll
      for (int s = 0; s < DP / 16; ++s)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int i = ((hq * 4 + jj) * 8 + g) * LK + s * 16 + 2 * t;
          mma16(sa[jj], ka[s], word(Qs, i), word(Qs, i + 8));
          mma16(dpt[jj], va[s], word(Os, i), word(Os, i + 8));
        }
      // the bias pairs (queries 2t, 2t + 1 at key g) of the warp's 16 keys: tile jj * 2 + hk, transposed
      uint32_t bw[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
      if (tp)
#pragma unroll
        for (int x = 0; x < 2; ++x) {
          const int m = x * 4 + (lane >> 3);
          ldT4(bw + 4 * x, T + ((hq * 4 + (m >> 1)) * 8 + (lane & 7)) * LB + kl0 + (m & 1) * 8);
        }
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int qe = (hq * 4 + jj) * 8 + 2 * t;   // the thread's two queries qe, qe + 1
        float Lq[2], Lgq[2], Dq[2];
        int sq[2];
#pragma unroll
        for (int par = 0; par < 2; ++par) {
          const float4 v = *reinterpret_cast<const float4*>(qv + 4 * (qe + par));
          // (a padded query, or one whose every key is out, takes +inf: its every p is then exactly 0)
          Lq[par] = q0 + qe + par < a.Q && v.x != -INFINITY ? v.x : INFINITY;
          Lgq[par] = LOG2E * v.y;
          Dq[par] = v.z;
          sq[par] = __float_as_int(v.w);
        }
        float cq[2];   // (!GEN: the query's log-sum-exp in base 2, folded into one subtraction a score)
        if constexpr (!GEN) {
#pragma unroll
          for (int par = 0; par < 2; ++par) cq[par] = fmaf(Lq[par], LOG2E, Lgq[par]);
        }
#pragma unroll
        for (int hk = 0; hk < 2; ++hk) {
          const int kl = kl0 + g + hk * 8;
          const uint32_t bb = bw[jj * 2 + hk];
          float dsp[2];
#pragma unroll
          for (int par = 0; par < 2; ++par) {
            const int e = hk * 2 + par;
            const float bias = __uint_as_float(par ? bb & 0xffff0000u : bb << 16);
            float p;
            if constexpr (!GEN) {   // (no clip, no chain ids: 2^(s scale log2e + (key term + bias) log2e - c_q), two FMAs)
              // (a row whose every key is masked, max ~-1e9: the key term and c_q, both ~-1.4e9, are 128 apart in
              // float32 and the fused form lands 2^128 off - that row takes the general form, in natural units)
              if (Lq[par] > -1e8f) p = ex2(fmaf(sa[jj][e], sl2, fmaf(bias, LOG2E, ckL[hk] - cq[par])));
              else p = ex2(fmaf(fmaf(sa[jj][e], a.scale, kaR[hk] + bias) - Lq[par], LOG2E, -Lgq[par]));
            } else {
              float x = fminf(fmaxf(fmaf(sa[jj][e], a.scale, kaR[hk] + bias), loC[hk]), hiC);
              if (seqQ && ksR[hk] != sq[par]) x = -INFINITY;
              p = ex2(fmaf(x - Lq[par], LOG2E, -Lgq[par]));
            }
            const float ds = p * (dpt[jj][e] - Dq[par]);
            sa[jj][e] = p;
            dpt[jj][e] = ds;
            dsp[par] = ds;
          }
          *reinterpret_cast<uint32_t*>(Sb + sOff(kl, qe)) = bf2(dsp[0], dsp[1]);
        }
      }
      // the bias gradients straight from the registers, together (a warp's atomic covers four rows' 32-byte
      // sectors whole). Unconditional in a full tile: a masked element's dS is exactly 0 (a branch an element was a
      // fifth of the issue)
      // (offsets in 32 bits from the slab's (query 0, key k0): within one (b, h) slab, Q sq < 2^31)
      const int qb = q0 + hq * 32 + 2 * t;
      const bool full = qb - 2 * t + 32 <= a.Q && k0 + BC <= a.K;   // (the whole half-tile in range: no checks)
      // the thread's 8-query group jj (four values: queries 2t, 2t + 1 at keys g and g + 8)
      auto addJ = [&](int jj, const float* v) {
        auto add = [&](auto* gp, int sq) {
          const int base = qb * sq + kl0 + g;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int qo = jj * 8 + (e & 1), ko = (e >> 1) * 8;
            const int o = base + qo * sq + ko;
            // (predicated, not redirected: pointing every out-of-range element at one address serialised a
            // partial key block's atomics on it - 7x the kernel at a length 522, 10 keys past a multiple of 64)
            if (!full && !(qb + qo < a.Q && k0 + kl0 + g + ko < a.K)) continue;
            if constexpr (sizeof(*gp) == 8)
              atomicAdd(gp + o, (unsigned long long)__double2ll_rn((double)v[e] * FIXED));
            else
              atomicAdd(gp + o, v[e]);
          }
        };
        if (f0) add(f0, (int)a.bias[0].sq); else if (g0) add(g0, (int)a.bias[0].sq);
        if (f1) add(f1, (int)a.bias[1].sq); else if (g1) add(g1, (int)a.bias[1].sq);
      };
      // CR > 1: each CTA sends the groups jj another rank owns (jj % CR) to that rank, and adds the cluster's sum of
      // its own after the products below (the barrier's arrive here, its wait there)
      float4* R = Red;
      if constexpr (CR > 1) {
        if (red) {
          clusterWait();   // (every CTA has read the previous round's sums out of its buffer)
          if (threadIdx.x == 0) mbarExpect(mbar, (CR - 1) * (4 / CR) * THREADS * 16);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj)
            if (jj % CR != (int)rk)
              stAsync4(R + ((int)rk * (4 / CR) + jj / CR) * THREADS + threadIdx.x, mbar, jj % CR, dpt[jj][0], dpt[jj][1],
                       dpt[jj][2], dpt[jj][3]);
        }
      } else if (g0 || g1) {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) addJ(jj, dpt[jj]);
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t pa[4], da[4];
        cToA(pa, sa[2 * u], sa[2 * u + 1]);
        cToA(da, dpt[2 * u], dpt[2 * u + 1]);
#pragma unroll
        for (int n = 0; n < D / 8; ++n) {
          uint32_t b0, b1;
          ldT(b0, b1, Os, hq * 32 + u * 16, n * 8, LK);
          mma16(dva[n], pa, b0, b1);
          ldT(b0, b1, Qs, hq * 32 + u * 16, n * 8, LK);
          mma16(dka[n], da, b0, b1);
        }
      }
      if constexpr (CR > 1) {
        if (red) {
          mbarWait(mbar, rc & 1);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            if (jj % CR != (int)rk) continue;
            float v[4] = {dpt[jj][0], dpt[jj][1], dpt[jj][2], dpt[jj][3]};
#pragma unroll
            for (int src = 0; src < CR; ++src) {   // (a fixed order, the owner fixed: deterministic run to run)
              if (src == (int)rk) continue;
              const float4 x = R[(src * (4 / CR) + jj / CR) * THREADS + threadIdx.x];
              v[0] += x.x; v[1] += x.y; v[2] += x.z; v[3] += x.w;
            }
            addJ(jj, v);
          }
          clusterArrive(false);
        }
      }
      ++rc;
    }
    __syncthreads();
    // dQ for this warp's 16 queries over the block's 64 keys, added straight from the accumulators
    // (QT 32: warps 0, 1 and 2, 3 take the tile's two 16-query halves, each pair splitting the n-tiles)
    const int qw = QT == 64 ? warp * 16 : (warp & 1) * 16;
    constexpr int NT = D / 8;
    const int n0 = QT == 64 ? 0 : (warp >> 1) * (NT / 2), n1 = QT == 64 ? NT : (warp >> 1 ? NT : NT / 2);
    float dqa[D / 8][4];
#pragma unroll
    for (int n = 0; n < D / 8; ++n) dqa[n][0] = dqa[n][1] = dqa[n][2] = dqa[n][3] = 0.f;
#pragma unroll
    for (int s = 0; s < BC / 16; ++s) {
      uint32_t A[4];   // (tile m: queries qw + (m & 1) * 8, keys s * 16 + (m >> 1) * 8, transposed back)
      {
        const int m = lane >> 3, k = s * 16 + (m >> 1) * 8 + (lane & 7);
        ldT4(A, Sb + sOff(k, qw + (m & 1) * 8));
      }
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        if (n < n0 || n >= n1) continue;
        uint32_t b0, b1;
        ldT(b0, b1, Kb, s * 16, n * 8, LK);
        mma16(dqa[n], A, b0, b1);
      }
    }
    // (rows g and g ^ 1 trade a value, so that each atomic covers four rows' eight columns - 32-byte sectors
    // whole - where a fragment's own pairs put four lanes in each of eight)
    const int odd = g & 1;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int qe = q0 + qw + (g & ~1) + r * 8;
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        if (n < n0 || n >= n1) continue;
        const float c0 = dqa[n][r * 2] * a.scale, c1 = dqa[n][r * 2 + 1] * a.scale;
        const float x = __shfl_xor_sync(~0u, odd ? c0 : c1, 4);
        float* p = dqb + (long long)qe * lddq + n * 8 + 2 * t + odd;
        if (qe < a.Q) atomicAdd(p, odd ? x : c0);
        if (qe + 1 < a.Q) atomicAdd(p + lddq, odd ? c1 : x);
      }
    }
    __syncthreads();   // this stage is the one the next iteration's issue overwrites
  }
  if (red) clusterWait();   // (the last arrival's pair: no CTA leaves while another may still write into it)
  // dqo: the last of a (b, h)'s key blocks to finish narrows its dQ into the bf16 copy the projections read, while
  // it is in L2 (a separate pass re-read the whole float dQ; counted per tile instead, the fences cost more)
  if (dqo) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(cnt + bh, 1u) == (unsigned)(nkb - 1);
    __syncthreads();
    if (last) {
      __threadfence();
      for (int i = threadIdx.x; i < a.Q * (D / 4); i += THREADS) {
        const int qq = i / (D / 4), c = (i % (D / 4)) * 4;
        const float4 v = __ldcg(reinterpret_cast<const float4*>(dqb + (long long)qq * lddq + c));
        *reinterpret_cast<uint2*>(dqo + ((size_t)b * a.Q + qq) * ldqo + h * D + c) = make_uint2(bf2(v.x, v.y), bf2(v.z, v.w));
      }
    }
  }
  DKV* dkb = dk + (size_t)b * a.K * lddk + h * D;
  DKV* dvb = dv + (size_t)b * a.K * lddv + h * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int kk = k0 + kl0 + g + r * 8;
    if (kk >= a.K) continue;
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      DKV* pk = dkb + (long long)kk * lddk + n * 8 + 2 * t;
      DKV* pv = dvb + (long long)kk * lddv + n * 8 + 2 * t;
      float xk0 = dka[n][r * 2] * a.scale, xk1 = dka[n][r * 2 + 1] * a.scale, xv0 = dva[n][r * 2], xv1 = dva[n][r * 2 + 1];
      if (accK) { xk0 += ldOut(pk); xk1 += ldOut(pk + 1); }
      if (accV) { xv0 += ldOut(pv); xv1 += ldOut(pv + 1); }
      st2(pk, xk0, xk1);   // (the pair as one store: 4.65 -> 4.59 ms for AF3 block n 384 two calls)
      st2(pv, xv0, xv1);
    }
  }
}
template <int D, int QT = 64, int CR = 1>
inline size_t bwdBfSmem() {
  return 2 * (BC * ldBf<D>() + 4 * QT * ldBf<D>() + 2 * QT * LB + BC * QT) + 4 * (2 * 4 * QT) + (CR > 1 ? 4 * THREADS * 16 + 16 : 0);
}
// an operand's source: bf16 rows read in place where they are 16-byte aligned, else a head-major bf16 copy
template <int D, class X>
inline HSrc hsrc(const X* x, int ld, int B, int N, int H) {
  HSrc s;
  if (sizeof(X) == 2 && ld % 8 == 0 && ((uintptr_t)x & 15) == 0) {
    s.p = reinterpret_cast<const uint16_t*>(x); s.sB = (long long)N * ld; s.sH = D; s.sN = ld; s.N = N; s.W = D;
    return s;
  }
  constexpr int DP = padD<D>();
  const int Np = (N + 63) / 64 * 64;
  s.p = headsBf(x, ld, B, N, H, D, Np, DP); s.sB = (long long)H * Np * DP; s.sH = (long long)Np * DP; s.sN = DP;
  s.N = Np; s.W = DP;
  return s;
}
template <int D, class X, class OT = float>
inline void forwardBf(const Sdpa& a, const X* q, int ldq, const X* k, int ldk, const X* v, int ldv,
                      OT* o, int ldo, float* lse) {
  Scratch scratch;
  const int Npq = (a.Q + 63) / 64 * 64, Npk = (a.K + 63) / 64 * 64;
  const HSrc qs_ = hsrc<D>(q, ldq, a.B, a.Q, a.H), ks_ = hsrc<D>(k, ldk, a.B, a.K, a.H), vs_ = hsrc<D>(v, ldv, a.B, a.K, a.H);
  static bool attr = false;
  const size_t smem = fwdBfSmem<D>();
  if (!attr) { GK(cudaFuncSetAttribute(fwdBf<D, OT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  const BiasTile bt = biasBf(a, Npq, Npk);
  const size_t grid = (size_t)((a.Q + BR - 1) / BR) * a.H * a.B;
  fwdBf<D, OT><<<(unsigned)grid, THREADS, smem, STREAM>>>(a, bt, qs_, ks_, vs_, o, ldo, lse);
}
// ---- the grid ("triangle") attention's forward: fwdBf with the per-score work cut to what it needs. fwdBf issued ~37
// instructions a score (a shared load of the key term and of the bias for each, the clip and -inf branches, the scale
// and log2(e) as separate multiplies) and was bound by issue, not by its MMAs - 1.51 ms a call at 512 tokens, 4 heads
// of 32. Here a score is two FMAs into base 2 (the key terms pre-multiplied, built once a call by keyTermsK and copied
// with each key tile), a max, a subtraction, an exponential and a sum; the bias tile goes by ldmatrix straight into
// the accumulators' layout, K and V by ldmatrix; every copy walks a running pointer. The epilogue takes the gate:
// g = sigmoid(raw) written back over the raw gate (gateSigLdBfK's), og = o g (gateMulBfK's). 1.33 ms with both
// gate passes inside, against 2.0 for the three kernels; float scores as before (the logits' roundings move by an
// ulp: the scale folded into one FMA). Needs: no chain ids, no clip, the [q][k] bias the same for every row (sb 0).
__device__ __forceinline__ void ldsm2(uint32_t* r, const uint16_t* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldsm4(uint32_t* r, const uint16_t* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
template <int D, int GR, int WQ>
inline size_t gridFwdSmem() { return 2 * ((size_t)2 * GR * BC * ldBf<D>() * 2 + (size_t)16 * WQ * LB * 2 + GR * BC * 4); }
// GR: grid rows a block, WQ: warps a row (16 queries each; the row's query tile GQ = 16 WQ)
template <int D, int GR, int WQ>
__global__ void __launch_bounds__(32 * GR * WQ, 512 / (32 * GR * WQ)) gridFwdBf(Sdpa a, BiasTile bt, HSrc qs_, HSrc ks_, HSrc vs_,
                                                        __nv_bfloat16* __restrict__ o, int ldo, float* __restrict__ lse,
                                                        __nv_bfloat16* __restrict__ gate, int ldg,
                                                        __nv_bfloat16* __restrict__ og, int ldog,
                                                        const float* __restrict__ ka2, int kpad) {
  constexpr int DP = padD<D>(), LK = ldBf<D>();
  constexpr size_t KV = (size_t)BC * LK;                                   // one row's K (or V) tile, in halves
  constexpr int GQ = 16 * WQ, NT = 32 * GR * WQ, MT = 1;
  constexpr size_t STAGE = (2 * GR * KV * 2 + (size_t)GQ * LB * 2 + GR * BC * 4);   // bytes
  extern __shared__ __align__(16) unsigned char gsm[];
  auto Ks = [&](int st, int r) { return reinterpret_cast<uint16_t*>(gsm + st * STAGE) + r * 2 * KV; };
  auto Vs = [&](int st, int r) { return Ks(st, r) + KV; };
  auto Ts = [&](int st) { return reinterpret_cast<uint16_t*>(gsm + st * STAGE) + GR * 2 * KV; };
  auto KAs = [&](int st) { return reinterpret_cast<float*>(gsm + st * STAGE + 2 * GR * KV * 2 + (size_t)GQ * LB * 2); };
  const int nqb = (a.Q + GQ - 1) / GQ;
  const int qb = blockIdx.x % nqb, h = (blockIdx.x / nqb) % a.H, bp = blockIdx.x / (nqb * a.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int rg = warp / WQ, wq = warp % WQ;
  const int b = bp * GR + rg, bl = b < a.B ? b : a.B - 1;                  // (a last block's spare row: computes, stores nothing)
  const int q0 = qb * GQ, qw = q0 + wq * 16 * MT;
  const float sl2 = (float)a.scale * LOG2E;
  const uint16_t* tp = bt.p ? bt.p + h * bt.sh + (size_t)q0 * bt.ld : nullptr;
  // each thread's copies, as running pointers a key tile on (the generic loaders' address arithmetic was most of
  // the loop's integer issue): KC chunks of K and of V a row, 4 of the bias tile, and the row's key terms
  constexpr int CH = DP / 8, KC = (BC * CH + NT - 1) / NT;
  const uint16_t *kp[GR][KC], *vp[GR][KC];
  int kr[KC];
#pragma unroll
  for (int j = 0; j < KC; ++j) {
    const int i = threadIdx.x + j * NT, r = i / CH, c = (i % CH) * 8;
    kr[j] = (BC * CH % NT == 0 || i < BC * CH) && c < ks_.W ? r : 1 << 30;
#pragma unroll
    for (int g2 = 0; g2 < GR; ++g2) {
      const int br = bp * GR + g2 < a.B ? bp * GR + g2 : a.B - 1;
      kp[g2][j] = hrow(ks_, br, h, r) + c; vp[g2][j] = hrow(vs_, br, h, r) + c;
    }
  }
  const uint16_t* bsrc = tp ? tp + (size_t)(threadIdx.x / (BC / 8)) * bt.ld + (threadIdx.x % (BC / 8)) * 8 : nullptr;
  const int kaR = threadIdx.x / (BC / 4), kaC = (threadIdx.x % (BC / 4)) * 4;   // (threads 0 .. GR * 16 - 1)
  const float* kasrc = ka2 + (size_t)(bp * GR + kaR < a.B ? bp * GR + kaR : a.B - 1) * kpad + kaC;
  auto issue = [&](int st, int k0) {
#pragma unroll
    for (int g2 = 0; g2 < GR; ++g2)
#pragma unroll
      for (int j = 0; j < KC; ++j) {
        if (kr[j] == 1 << 30) continue;
        const int i = threadIdx.x + j * NT, off = (i / CH) * LK + (i % CH) * 8;
        const bool on = k0 + kr[j] < ks_.N;
        cp16(Ks(st, g2) + off, on ? kp[g2][j] + (long long)k0 * ks_.sN : ks_.p, on);
        cp16(Vs(st, g2) + off, on ? vp[g2][j] + (long long)k0 * vs_.sN : vs_.p, on);
      }
    if (tp) {
      uint16_t* T = Ts(st) + (threadIdx.x / (BC / 8)) * LB + (threadIdx.x % (BC / 8)) * 8;
#pragma unroll
      for (int j = 0; j < GQ * BC / 8 / NT; ++j) cp16(T + j * (NT / (BC / 8)) * LB, bsrc + (size_t)j * (NT / (BC / 8)) * bt.ld + k0, true);
    }
    if (threadIdx.x < GR * (BC / 4)) cp16(KAs(st) + kaR * BC + kaC, kasrc + k0, true);
  };
  issue(0, 0);
  cpCommit();
  uint32_t qa[MT][DP / 16][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int s = 0; s < DP / 16; ++s) aFragS(qa[mt][s], qs_, bl, h, qw + mt * 16 + g, s * 16 + 2 * t);
  float oacc[MT][D / 8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int n = 0; n < D / 8; ++n) oacc[mt][n][0] = oacc[mt][n][1] = oacc[mt][n][2] = oacc[mt][n][3] = 0.f;
  float mx[MT][2], sum[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) mx[mt][0] = mx[mt][1] = -INFINITY, sum[mt][0] = sum[mt][1] = 0.f;
  const int nk = (a.K + BC - 1) / BC;
  for (int it = 0; it < nk; ++it) {
    const int st = it & 1, k0 = it * BC;
    if (it + 1 < nk) issue(st ^ 1, k0 + BC);
    cpCommit();
    cpWait1();
    __syncthreads();
    const uint16_t *K = Ks(st, rg), *V = Vs(st, rg), *T = Ts(st);
    const float* ka = KAs(st) + rg * BC;
    // a tile's 16 x 64 scores at a time, one query tile after the other (both at once held 64 more registers and spilled)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      float sacc[BC / 8][4];
#pragma unroll
      for (int j = 0; j < BC / 8; ++j) sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0.f;
#pragma unroll
      for (int j = 0; j < BC / 8; ++j)
#pragma unroll
        for (int s = 0; s < DP / 16; ++s) {
          uint32_t kb[2];
          ldsm2(kb, K + (j * 8 + (lane & 7)) * LK + s * 16 + ((lane >> 3) & 1) * 8);
          mma16(sacc[j], qa[mt][s], kb[0], kb[1]);
        }
      // the logits in base 2: x = s scale log2(e) + (key term + bias) log2(e) - two FMAs (a key out is -inf in the key
      // term, and stays so); the bias tile read straight into the accumulators' layout (pairs of columns 2t, 2t + 1)
      const uint16_t* rowp = T + (wq * 16 * MT + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LB + (lane >> 4) * 8;
      float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int j = 0; j < BC / 8; j += 2) {
        uint32_t r4[4] = {0u, 0u, 0u, 0u};
        if (tp) ldsm4(r4, rowp + j * 8);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const float2 kk = *reinterpret_cast<const float2*>(ka + (j + jj) * 8 + 2 * t);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint32_t w = r4[jj * 2 + (e >> 1)];
            const float bias = __uint_as_float(e & 1 ? w & 0xffff0000u : w << 16);
            const float x = fmaf(sacc[j + jj][e], sl2, fmaf(bias, LOG2E, (e & 1) ? kk.y : kk.x));
            sacc[j + jj][e] = x;
            rmax[e >> 1] = fmaxf(rmax[e >> 1], x);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(~0u, rmax[r], 1));
        rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(~0u, rmax[r], 2));
      }
      float alpha[2], nm[2], rs[2] = {0.f, 0.f};
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float m = fmaxf(mx[mt][r], rmax[r]);
        alpha[r] = m == -INFINITY ? 1.f : ex2(mx[mt][r] - m);
        mx[mt][r] = m;
        nm[r] = m == -INFINITY ? 0.f : m;   // (a row with every key out so far: its p are 2^-inf = 0)
      }
#pragma unroll
      for (int j = 0; j < BC / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float p = ex2(sacc[j][e] - nm[e >> 1]);
          rs[e >> 1] += p;
          sacc[j][e] = p;
        }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        rs[r] += __shfl_xor_sync(~0u, rs[r], 1);
        rs[r] += __shfl_xor_sync(~0u, rs[r], 2);
        sum[mt][r] = sum[mt][r] * alpha[r] + rs[r];
      }
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        oacc[mt][n][0] *= alpha[0]; oacc[mt][n][1] *= alpha[0]; oacc[mt][n][2] *= alpha[1]; oacc[mt][n][3] *= alpha[1];
      }
#pragma unroll
      for (int u = 0; u < BC / 16; ++u) {
        uint32_t pa[4];
        cToA(pa, sacc[2 * u], sacc[2 * u + 1]);
#pragma unroll
        for (int n = 0; n < D / 8; ++n) {
          uint32_t b0, b1;
          ldT(b0, b1, V, u * 16, n * 8, LK);
          mma16(oacc[mt][n], pa, b0, b1);
        }
      }
    }
    __syncthreads();   // this stage is the one the next iteration's issue overwrites
  }
  if (b >= a.B) return;
  const size_t bh = (size_t)b * a.H + h, N = (size_t)a.B * a.H * a.Q;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int qq = qw + mt * 16 + g + r * 8;
      if (qq >= a.Q) continue;
      const float inv = sum[mt][r] > 0.f ? 1.f / sum[mt][r] : 0.f;
      const size_t row = (size_t)b * a.Q + qq;
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        const int c = h * D + n * 8 + 2 * t;
        const uint32_t ow = bf2(oacc[mt][n][r * 2] * inv, oacc[mt][n][r * 2 + 1] * inv);
        *reinterpret_cast<uint32_t*>(o + row * ldo + c) = ow;
        if (gate) {
          uint32_t* gp = reinterpret_cast<uint32_t*>(gate + row * ldg + c);
          const uint32_t gw = *gp;
          const float s0 = __bfloat162float(__float2bfloat16(1.f / (1.f + __expf(-__uint_as_float(gw << 16))))),
                      s1 = __bfloat162float(__float2bfloat16(1.f / (1.f + __expf(-__uint_as_float(gw & 0xffff0000u)))));
          *gp = bf2(s0, s1);
          *reinterpret_cast<uint32_t*>(og + row * ldog + c) = bf2(__uint_as_float(ow << 16) * s0, __uint_as_float(ow & 0xffff0000u) * s1);
        }
      }
      if (t == 0) {
        const size_t i = bh * a.Q + qq;
        lse[i] = sum[mt][r] > 0.f ? mx[mt][r] / LOG2E : -INFINITY;
        lse[N + i] = sum[mt][r] > 0.f ? logf(sum[mt][r]) : 0.f;
      }
    }
}
// every row's per-key terms (keyAdd - the key mask, -inf past K - in base 2) [B][Kp], for gridFwdBf's tiles
__global__ void keyTermsK(Sdpa a, float* __restrict__ out, int Kp) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)a.B * Kp) out[i] = keyAdd(a, (int)(i / Kp), 0, (int)(i % Kp)) * LOG2E;
}
// the grid attention's forward through gridFwdBf where its shape allows (false: the caller takes the general path)
inline bool gridForwardBf(const Sdpa& a, const __nv_bfloat16* q, int ldq, const __nv_bfloat16* k, int ldk,
                          const __nv_bfloat16* v, int ldv, __nv_bfloat16* o, int ldo, float* lse,
                          __nv_bfloat16* gate = nullptr, int ldg = 0, __nv_bfloat16* og = nullptr, int ldog = 0) {
  if (a.seqQ || a.seqK || a.clip || (a.D != 16 && a.D != 32) || a.bias[1].p) return false;
  if (a.bias[0].p && (a.bias[0].sb != 0 || a.bias[0].sq == 0)) return false;
  if (ldo % 2 || (gate && (ldg % 2 || ldog % 2))) return false;
  Scratch scratch;
  const int Npq = (a.Q + 63) / 64 * 64, Npk = (a.K + 63) / 64 * 64;
  const BiasTile bt = biasBf(a, Npq, Npk);
  float* ka2 = ARENA.get((size_t)a.B * Npk);   // (no per-key bias here: keyAdd is the same for every head)
  keyTermsK<<<blocks((size_t)a.B * Npk), 256, 0, STREAM>>>(a, ka2, Npk);
  // (one row of 64 queries a block: two rows sharing the bias tile, or 128 queries sharing K and V, measured the same
  // at 512 tokens - 11.6, 11.6 and 11.5 ms a pairformer block's forward - though the kernel reads L2 at 74% of its
  // throughput: what is left is latency, ~15 instructions a score at 4 blocks of 4 warps an SM)
  constexpr int GR = 1, WQ = 4, GQ = 16 * WQ;
  const size_t grid = (size_t)((a.Q + GQ - 1) / GQ) * a.H * ((a.B + GR - 1) / GR);
  auto run = [&](auto Dc) {
    constexpr int DD = decltype(Dc)::value;
    const HSrc qs_ = hsrc<DD>(q, ldq, a.B, a.Q, a.H), ks_ = hsrc<DD>(k, ldk, a.B, a.K, a.H), vs_ = hsrc<DD>(v, ldv, a.B, a.K, a.H);
    static bool attr = false;
    const size_t smem = gridFwdSmem<DD, GR, WQ>();
    if (!attr) { GK(cudaFuncSetAttribute(gridFwdBf<DD, GR, WQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
    gridFwdBf<DD, GR, WQ><<<(unsigned)grid, 32 * GR * WQ, smem, STREAM>>>(a, bt, qs_, ks_, vs_, o, ldo, lse, gate, ldg, og, ldog, ka2, Npk);
  };
  if (a.D == 32) run(std::integral_constant<int, 32>{}); else run(std::integral_constant<int, 16>{});
  return true;
}

template <int D, class X, class DKV = float, class DOT = float>
inline void backwardBf(const Sdpa& a, const X* q, int ldq, const X* k, int ldk, const X* v, int ldv,
                       const DOT* dO, int lddo, const float* lse, const float* Di, float* dq, int lddq, DKV* dk,
                       int lddk, DKV* dv, int lddv, bool accK, bool accV, __nv_bfloat16* dqo = nullptr, int ldqo = 0) {
  Scratch scratch;
  const int Npq = (a.Q + 63) / 64 * 64, Npk = (a.K + 63) / 64 * 64;
  const HSrc qs_ = hsrc<D>(q, ldq, a.B, a.Q, a.H), ks_ = hsrc<D>(k, ldk, a.B, a.K, a.H), vs_ = hsrc<D>(v, ldv, a.B, a.K, a.H),
             os_ = hsrc<D>(dO, lddo, a.B, a.Q, a.H);
  // (a 32-row query tile halves the shared memory, and lost both ways at the triangle shape on the A100: 907 us
  // capped at 128 registers for four CTAs an SM, 966 us at three, against 761 at 64 rows. Where the 64-row tile's
  // 54 KB leaves the SM one CTA - an RTX PRO 6000's ~100 KB - the 32-row one runs instead: narrow)
  constexpr int QT = 64, CR = 2;
  static bool attr = false, clusters = false, narrow = false;
  const size_t smem = bwdBfSmem<D, QT>(), smemC = bwdBfSmem<D, QT, CR>(), smemN = bwdBfSmem<D, 32>();
  if (!attr) {
    for (auto f : {bwdBf<D, DKV, QT, true>, bwdBf<D, DKV, QT, false>}) {
      GK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      GK(cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    int per = 0;
    GK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, bwdBf<D, DKV, QT, false>, THREADS, smem));
    narrow = per < 2;
    if (narrow)
      for (auto f : {bwdBf<D, DKV, 32, true>, bwdBf<D, DKV, 32, false>}) {
        GK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemN));
        GK(cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      }
    // clusters on sm_90 alone, where the device and this binary's code for it have them: measured there (the
    // grid512 bench's backward 2.75 -> 2.55 ms on the GH200); on an RTX PRO 6000 (sm_120) the same kernel lost, 3.47
    // -> 4.20 ms)
    int dev = 0, major = 0;
    GK(cudaGetDevice(&dev));
    GK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    cudaFuncAttributes fa;
    GK(cudaFuncGetAttributes(&fa, bwdBf<D, DKV, QT, false, CR>));
    clusters = major == 9 && fa.binaryVersion >= 90 && !narrow;
    if (clusters)
      for (auto f : {bwdBf<D, DKV, QT, true, CR>, bwdBf<D, DKV, QT, false, CR>}) {
        GK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemC));
        GK(cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      }
    attr = true;
  }
  const BiasTile bt = biasBf(a, Npq, Npk);
  // the fixed-point accumulators: each bias gradient's whole extent, zeroed, then added back into it
  unsigned long long* fx[2] = {nullptr, nullptr};
  size_t ext[2] = {0, 0};
  for (int j = 0; j < 2; ++j) {
    const Bias& B = a.bias[j];
    if (!B.grad || !a.deterministic) continue;
    ext[j] = (size_t)((B.sb ? a.B - 1 : 0) * B.sb + (B.sh ? a.H - 1 : 0) * B.sh + (long long)(a.Q - 1) * B.sq + a.K);
    fx[j] = reinterpret_cast<unsigned long long*>(ARENA.get(2 * ext[j]));
    GK(cudaMemsetAsync(fx[j], 0, ext[j] * 8, STREAM));
  }
  const size_t grid = (size_t)((a.K + BC - 1) / BC) * a.H * a.B;
  const bool gen = a.seqQ || a.seqK || fx[0] || fx[1] || a.clip;   // (!gen: bwdBf's short form, which has no clip)
  // dqo: dQ narrowed into bf16 in the kernel, a tile at a time by the last of its key blocks (counted); where the
  // vector accesses do not align, by a pass after
  const bool inK = dqo && lddq % 4 == 0 && ldqo % 4 == 0 && ((uintptr_t)dq & 15) == 0 && ((uintptr_t)dqo & 7) == 0;
  unsigned* cnt = nullptr;
  if (inK) {
    const size_t nc = (size_t)a.B * a.H;
    cnt = reinterpret_cast<unsigned*>(ARENA.get(nc));
    GK(cudaMemsetAsync(cnt, 0, nc * 4, STREAM));
  }
  uint16_t* dqoK = inK ? reinterpret_cast<uint16_t*>(dqo) : nullptr;
  // the bias gradients summed over CR rows on the chip first (a cluster of CR rows' CTAs): at the triangle's shape,
  // every [q][k] element took one atomic a row - 537 M a call at 512 tokens, the kernel L2-bound at 80% of it
  bool rowShared = a.bias[0].grad || a.bias[1].grad;
  for (int j = 0; j < 2; ++j)
    if (a.bias[j].grad && a.bias[j].sb != 0) rowShared = false;
  if (clusters && rowShared && a.B % CR == 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = smemC; cfg.stream = STREAM;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CR; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    GK(cudaLaunchKernelEx(&cfg, gen ? bwdBf<D, DKV, QT, true, CR> : bwdBf<D, DKV, QT, false, CR>, a, bt, qs_, ks_, vs_,
                          os_, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV, fx[0], fx[1], dqoK, ldqo, cnt));
  } else if (narrow) {
    (gen ? bwdBf<D, DKV, 32, true> : bwdBf<D, DKV, 32, false>)<<<(unsigned)grid, THREADS, smemN, STREAM>>>(a, bt, qs_, ks_,
        vs_, os_, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV, fx[0], fx[1], dqoK, ldqo, cnt);
  } else {
    (gen ? bwdBf<D, DKV, QT, true> : bwdBf<D, DKV, QT, false>)<<<(unsigned)grid, THREADS, smem, STREAM>>>(a, bt, qs_, ks_,
        vs_, os_, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV, fx[0], fx[1], dqoK, ldqo, cnt);
  }
  if (dqo && !inK) narrowDqK<<<blocks((size_t)a.B * a.Q * a.H * D), 256, 0, STREAM>>>(dq, lddq, dqo, ldqo, a.H * D, (size_t)a.B * a.Q);
  for (int j = 0; j < 2; ++j)
    if (fx[j]) fromFixedK<<<blocks(ext[j]), 256, 0, STREAM>>>(a.bias[j].grad, fx[j], ext[j]);
}

// The TF32 forward in its common case - one bias, no clip, no chain ids (the diffusion transformer's token attention,
// 16 heads of 48 over [S n] rows): fwdK<D, 0> was latency-bound at 13 TFLOP/s (7.9 cycles an issue, three warps a
// scheduler for its 44 KB of shared memory, every tile loaded and rounded by plain loads before the barrier, P handed
// between the products through shared memory). Here K and V are double-buffered by cp.async and rounded to tf32 in
// shared memory once, by the thread that loaded them; each warp fetches its own 16 rows of the bias tile, one tile
// ahead; the key mask is a term a key; the logits stay in natural units (exp2 of (x - max) log2 e: a max in log2
// units is off by ~100 at a row masked whole, -1.4e9, which the backward's natural replay turns into e^100); and P
// stays in registers: the P V product sums over keys, so within each 8 keys the A fragment's column t is key 2t and
// t + 4 is 2t + 1 - which is where the score accumulator already holds them - and V's rows are read in that order.
template <int D>
__global__ void __launch_bounds__(THREADS, 2) fwdT(Sdpa a, const float* __restrict__ q, int ldq, const float* __restrict__ k,
                                                   int ldk, const float* __restrict__ v, int ldv, float* __restrict__ o,
                                                   int ldo, float* __restrict__ lse) {
  constexpr int DK = D + PADK, BS = BC + 8;        // K and V's row stride; the bias tile's (a float2 a half-warp conflict-free)
  constexpr int KV = 2 * BC * DK + BC;             // a stage: K, V, the key terms
  constexpr int C4 = D / 4;
  extern __shared__ __align__(16) float sm[];
  float* Bs = sm + 2 * KV;                         // [BR][BS]
  const int nqb = (a.Q + BR - 1) / BR;
  const int qb = blockIdx.x % nqb, h = (blockIdx.x / nqb) % a.H, b = blockIdx.x / (nqb * a.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int q0 = qb * BR, qr0 = q0 + warp * 16;
  const float* kb_ = k + (size_t)b * a.K * ldk + h * D;
  const float* vb_ = v + (size_t)b * a.K * ldv + h * D;
  const float* bb_ = a.bias[0].p + b * a.bias[0].sb + h * a.bias[0].sh;
  const long long bsq = a.bias[0].sq;
  const float* km_ = a.keyMask ? a.keyMask + (size_t)b * a.K : nullptr;
  const float sc = a.scale;
  const int nkt = (a.K + BC - 1) / BC;
  auto issueKV = [&](int kt) {
    float* Ks = sm + (kt & 1) * KV;
    float* Vs = Ks + BC * DK;
    const int k0 = kt * BC;
    for (int i = threadIdx.x; i < BC * C4; i += THREADS) {
      const int r = i / C4, c = (i % C4) * 4;
      const bool in = k0 + r < a.K;
      const long long rr = in ? k0 + r : 0;
      cp16(Ks + r * DK + c, kb_ + rr * ldk + c, in);
      cp16(Vs + r * DK + c, vb_ + rr * ldv + c, in);
    }
    if (threadIdx.x < BC) cp4(Vs + BC * DK + threadIdx.x, km_ ? km_ + k0 + threadIdx.x : kb_, km_ && k0 + threadIdx.x < a.K);
  };
  // this warp's 16 rows of the bias tile (rows past Q read row Q - 1: written nowhere)
  auto issueBias = [&](int kt) {
    const int k0 = kt * BC;
    for (int i = lane; i < 16 * BC / 4; i += 32) {
      const int r = warp * 16 + i / (BC / 4), c = (i % (BC / 4)) * 4;
      const bool in = k0 + c < a.K;
      cp16(Bs + r * BS + c, bb_ + (long long)min(q0 + r, a.Q - 1) * bsq + (in ? k0 + c : 0), in);
    }
  };
  issueKV(0);
  issueBias(0);
  cpCommit();
  uint32_t qa[D / 8][4];
  {
    const float* qb_ = q + (size_t)b * a.Q * ldq + h * D;
#pragma unroll
    for (int s = 0; s < D / 8; ++s)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = qr0 + g + (i & 1) * 8, c = s * 8 + t + (i >> 1) * 4;
        qa[s][i] = tf32(r < a.Q ? qb_[(long long)r * ldq + c] : 0.f);
      }
  }
  float oacc[D / 8][4];
#pragma unroll
  for (int n = 0; n < D / 8; ++n) oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f;
  float mx[2] = {-INFINITY, -INFINITY}, sum[2] = {0.f, 0.f};
  for (int kt = 0; kt < nkt; ++kt) {
    if (kt + 1 < nkt) issueKV(kt + 1);
    cpCommit();
    cpWait1();
    float* Ks = sm + (kt & 1) * KV;
    float* Vs = Ks + BC * DK;
    float* Ms = Vs + BC * DK;
    {   // (what this thread loaded: rounded to tf32; the key terms)
      const int k0 = kt * BC;
      for (int i = threadIdx.x; i < BC * C4; i += THREADS) {
        const int r = i / C4, c = (i % C4) * 4;
        float4* pk = reinterpret_cast<float4*>(Ks + r * DK + c);
        float4* pv = reinterpret_cast<float4*>(Vs + r * DK + c);
        float4 x = *pk, y = *pv;
        x.x = __uint_as_float(tf32(x.x)); x.y = __uint_as_float(tf32(x.y)); x.z = __uint_as_float(tf32(x.z)); x.w = __uint_as_float(tf32(x.w));
        y.x = __uint_as_float(tf32(y.x)); y.y = __uint_as_float(tf32(y.y)); y.z = __uint_as_float(tf32(y.z)); y.w = __uint_as_float(tf32(y.w));
        *pk = x; *pv = y;
      }
      if (threadIdx.x < BC)
        Ms[threadIdx.x] = k0 + threadIdx.x >= a.K ? -INFINITY : km_ ? 1e9f * (Ms[threadIdx.x] - 1.f) : 0.f;
    }
    __syncthreads();
    float sacc[BC / 8][4];
#pragma unroll
    for (int j = 0; j < BC / 8; ++j) sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0.f;
#pragma unroll
    for (int s = 0; s < D / 8; ++s)
#pragma unroll
      for (int j = 0; j < BC / 8; ++j)
        mmaTf32(sacc[j], qa[s][0], qa[s][1], qa[s][2], qa[s][3], __float_as_uint(Ks[(j * 8 + g) * DK + s * 8 + t]),
                __float_as_uint(Ks[(j * 8 + g) * DK + s * 8 + t + 4]));
    float rmax[2] = {-INFINITY, -INFINITY};
    {
      const float* B0 = Bs + (warp * 16 + g) * BS + 2 * t;
#pragma unroll
      for (int j = 0; j < BC / 8; ++j) {
        const float2 m2 = *reinterpret_cast<const float2*>(Ms + j * 8 + 2 * t);
        const float2 c0 = *reinterpret_cast<const float2*>(B0 + j * 8);
        const float2 c1 = *reinterpret_cast<const float2*>(B0 + 8 * BS + j * 8);
        sacc[j][0] = fmaf(sacc[j][0], sc, m2.x + c0.x);
        sacc[j][1] = fmaf(sacc[j][1], sc, m2.y + c0.y);
        sacc[j][2] = fmaf(sacc[j][2], sc, m2.x + c1.x);
        sacc[j][3] = fmaf(sacc[j][3], sc, m2.y + c1.y);
        rmax[0] = fmaxf(rmax[0], fmaxf(sacc[j][0], sacc[j][1]));
        rmax[1] = fmaxf(rmax[1], fmaxf(sacc[j][2], sacc[j][3]));
      }
    }
    __syncwarp();
    if (kt + 1 < nkt) issueBias(kt + 1);   // (into this warp's rows, read above; waited on with the next K and V)
    cpCommit();
    float alpha[2], nm[2], rs[2] = {0.f, 0.f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(~0u, rmax[r], 1));
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(~0u, rmax[r], 2));
      nm[r] = fmaxf(mx[r], rmax[r]);
      alpha[r] = ex2((mx[r] - nm[r]) * LOG2E);   // (the first tile: ex2(-inf) = 0, onto zeros)
      mx[r] = nm[r];
    }
#pragma unroll
    for (int j = 0; j < BC / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        sacc[j][e] = ex2((sacc[j][e] - nm[e >> 1]) * LOG2E);
        rs[e >> 1] += sacc[j][e];
      }
#pragma unroll
    for (int r = 0; r < 2; ++r) sum[r] = sum[r] * alpha[r] + rs[r];   // (the quad's partial sums: reduced at the end)
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      oacc[n][0] *= alpha[0]; oacc[n][1] *= alpha[0]; oacc[n][2] *= alpha[1]; oacc[n][3] *= alpha[1];
    }
#pragma unroll
    for (int s = 0; s < BC / 8; ++s) {
      const uint32_t a0 = tf32(sacc[s][0]), a1 = tf32(sacc[s][2]), a2 = tf32(sacc[s][1]), a3 = tf32(sacc[s][3]);
#pragma unroll
      for (int n = 0; n < D / 8; ++n)
        mmaTf32(oacc[n], a0, a1, a2, a3, __float_as_uint(Vs[(s * 8 + 2 * t) * DK + n * 8 + g]),
                __float_as_uint(Vs[(s * 8 + 2 * t + 1) * DK + n * 8 + g]));
    }
    __syncthreads();   // (this stage's K and V are the next issue's)
  }
  float* ob = o + (size_t)b * a.Q * ldo + h * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    sum[r] += __shfl_xor_sync(~0u, sum[r], 1);
    sum[r] += __shfl_xor_sync(~0u, sum[r], 2);
    const int qq = qr0 + g + r * 8;
    if (qq >= a.Q) continue;
    const float inv = sum[r] > 0.f ? 1.f / sum[r] : 0.f;
#pragma unroll
    for (int n = 0; n < D / 8; ++n)
      *reinterpret_cast<float2*>(ob + (long long)qq * ldo + n * 8 + 2 * t) = make_float2(oacc[n][r * 2] * inv, oacc[n][r * 2 + 1] * inv);
    if (t == 0) {
      const size_t i = ((size_t)b * a.H + h) * a.Q + qq, N = (size_t)a.B * a.H * a.Q;
      lse[i] = sum[r] > 0.f ? mx[r] : -INFINITY;
      lse[N + i] = sum[r] > 0.f ? logf(sum[r]) : 0.f;
    }
  }
}
template <int D>
inline bool forwardT(const Sdpa& a, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv, float* o,
                     int ldo, float* lse) {
  const Bias& bi = a.bias[0];
  auto al = [](const void* p) { return ((uintptr_t)p & 15) == 0; };
  if (a.clip || a.seqQ || a.seqK || a.bias[1].p || !bi.p || D % 8 || a.K % 4 || bi.sq % 4 || bi.sb % 4 || bi.sh % 4 ||
      ldq % 4 || ldk % 4 || ldv % 4 || ldo % 2 || !al(bi.p) || !al(k) || !al(v))
    return false;
  constexpr size_t smem = sizeof(float) * (2 * (2 * BC * (D + PADK) + BC) + BR * (BC + 8));
  static bool attr = false;
  if (!attr) { GK(cudaFuncSetAttribute(fwdT<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  const size_t grid = (size_t)((a.Q + BR - 1) / BR) * a.H * a.B;
  fwdT<D><<<(unsigned)grid, THREADS, smem, STREAM>>>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse);
  return true;
}

// fwdT's backward (the same case): a CTA 64 keys, a warp 16 of them, the queries in tiles of 32 - Q, dO, the tile's
// bias and its rows' log-sum-exp and Di double-buffered by cp.async, Q and dO rounded to tf32 in shared memory by the
// thread that loaded them; K and V as tf32 A fragments in registers. P and dS go from the S^T and dP^T accumulators
// straight into dV += P^T dO and dK += dS^T Q (fwdT's key permutation, here over queries); dS also replaces the bias
// tile in place, for dQ += dS K (its keys permuted the same way, which keeps K's fragment reads conflict-free) and
// the bias gradient's coalesced atomics. bwdK<D, 0> was 4.0 ms at the diffusion transformer's shape: 12.5% occupancy
// for 58 KB of shared memory, every tile loaded by plain loads, P and dS through shared memory.
template <int D>
__global__ void __launch_bounds__(THREADS, 2) bwdT(Sdpa a, const float* __restrict__ q, int ldq, const float* __restrict__ k,
                                                   int ldk, const float* __restrict__ v, int ldv,
                                                   const float* __restrict__ dO, int lddo, const float* __restrict__ lse,
                                                   const float* __restrict__ Di, float* __restrict__ dq, int lddq,
                                                   float* __restrict__ dk, int lddk, float* __restrict__ dv, int lddv,
                                                   bool accK, bool accV) {
  constexpr int LD = D + PADK, LT = BC + PADK, QT = 32, C4 = D / 4, NT = D / 8;
  constexpr int ST = 2 * QT * LD + QT * LT + 3 * QT;   // a stage: Q, dO, the bias tile (then dS), lse, log-sum, Di
  extern __shared__ __align__(16) float sm[];
  float* Ks = sm + 2 * ST;                             // [BC][LD], tf32
  const int nkb = (a.K + BC - 1) / BC;
  const int kb = blockIdx.x % nkb, h = (blockIdx.x / nkb) % a.H, b = blockIdx.x / (nkb * a.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int k0 = kb * BC, kl0 = warp * 16;
  const float* qb_ = q + (size_t)b * a.Q * ldq + h * D;
  const float* ob_ = dO + (size_t)b * a.Q * lddo + h * D;
  const float* kb_ = k + (size_t)b * a.K * ldk + h * D;
  const float* vb_ = v + (size_t)b * a.K * ldv + h * D;
  const float* bb_ = a.bias[0].p + b * a.bias[0].sb + h * a.bias[0].sh + k0;
  const long long bsq = a.bias[0].sq;
  const size_t bh = (size_t)b * a.H + h;
  const float* lseb = lse + bh * a.Q;
  const float* lgb = lseb + (size_t)a.B * a.H * a.Q;
  const float* Dib = Di + bh * a.Q;
  float* dqb = dq + (size_t)b * a.Q * lddq + h * D;
  float* g0 = a.bias[0].grad ? a.bias[0].grad + b * a.bias[0].sb + h * a.bias[0].sh + k0 : nullptr;
  const int nqt = (a.Q + QT - 1) / QT;
  auto issue = [&](int qt) {
    float* Qs = sm + (qt & 1) * ST;
    float* Os = Qs + QT * LD;
    float* Ts = Os + QT * LD;
    float* Ls = Ts + QT * LT;
    const int q0 = qt * QT;
    for (int i = threadIdx.x; i < QT * C4; i += THREADS) {
      const int r = i / C4, c = (i % C4) * 4;
      const bool in = q0 + r < a.Q;
      const long long rr = in ? q0 + r : 0;
      cp16(Qs + r * LD + c, qb_ + rr * ldq + c, in);
      cp16(Os + r * LD + c, ob_ + rr * lddo + c, in);
    }
    for (int i = threadIdx.x; i < QT * BC / 4; i += THREADS) {
      const int r = i / (BC / 4), c = (i % (BC / 4)) * 4;
      const bool in = q0 + r < a.Q && k0 + c < a.K;
      cp16(Ts + r * LT + c, in ? bb_ + (long long)(q0 + r) * bsq + c : bb_, in);
    }
    if (threadIdx.x < QT) {
      const bool in = q0 + threadIdx.x < a.Q;
      cp4(Ls + threadIdx.x, lseb + (in ? q0 + threadIdx.x : 0), in);
      cp4(Ls + QT + threadIdx.x, lgb + (in ? q0 + threadIdx.x : 0), in);
      cp4(Ls + 2 * QT + threadIdx.x, Dib + (in ? q0 + threadIdx.x : 0), in);
    }
  };
  for (int i = threadIdx.x; i < BC * C4; i += THREADS) {
    const int r = i / C4, c = (i % C4) * 4;
    const bool in = k0 + r < a.K;
    cp16(Ks + r * LD + c, kb_ + (long long)(in ? k0 + r : 0) * ldk + c, in);
  }
  issue(0);
  cpCommit();
  uint32_t ka[NT][4], va[NT][4];
  float kmR[2];
#pragma unroll
  for (int s = 0; s < NT; ++s)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = k0 + kl0 + g + (i & 1) * 8, c = s * 8 + t + (i >> 1) * 4;
      ka[s][i] = tf32(r < a.K ? kb_[(long long)r * ldk + c] : 0.f);
      va[s][i] = tf32(r < a.K ? vb_[(long long)r * ldv + c] : 0.f);
    }
#pragma unroll
  for (int hk = 0; hk < 2; ++hk) {
    const int kk = k0 + kl0 + g + hk * 8;
    kmR[hk] = kk >= a.K ? -INFINITY : a.keyMask ? 1e9f * (a.keyMask[(size_t)b * a.K + kk] - 1.f) : 0.f;
  }
  float dka[NT][4], dva[NT][4];
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dka[n][e] = dva[n][e] = 0.f;
  const float sc = a.scale;
  for (int qt = 0; qt < nqt; ++qt) {
    if (qt + 1 < nqt) issue(qt + 1);
    cpCommit();
    cpWait1();
    float* Qs = sm + (qt & 1) * ST;
    float* Os = Qs + QT * LD;
    float* Ts = Os + QT * LD;
    float* Ls = Ts + QT * LT;
    const int q0 = qt * QT;
    {   // (what this thread loaded, rounded to tf32: Q and dO, and K the first time)
      auto round4 = [](float* p) {
        float4 x = *reinterpret_cast<float4*>(p);
        x.x = __uint_as_float(tf32(x.x)); x.y = __uint_as_float(tf32(x.y)); x.z = __uint_as_float(tf32(x.z)); x.w = __uint_as_float(tf32(x.w));
        *reinterpret_cast<float4*>(p) = x;
      };
      for (int i = threadIdx.x; i < QT * C4; i += THREADS) {
        const int r = i / C4, c = (i % C4) * 4;
        round4(Qs + r * LD + c);
        round4(Os + r * LD + c);
      }
      if (qt == 0)
        for (int i = threadIdx.x; i < BC * C4; i += THREADS) round4(Ks + (i / C4) * LD + (i % C4) * 4);
    }
    __syncthreads();
    float st[4][4], dpt[4][4];   // 16 keys x 32 queries: S^T and dP^T
#pragma unroll
    for (int jj = 0; jj < 4; ++jj)
#pragma unroll
      for (int e = 0; e < 4; ++e) st[jj][e] = dpt[jj][e] = 0.f;
#pragma unroll
    for (int s = 0; s < NT; ++s)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int i0 = (jj * 8 + g) * LD + s * 8 + t;
        mmaTf32(st[jj], ka[s][0], ka[s][1], ka[s][2], ka[s][3], __float_as_uint(Qs[i0]), __float_as_uint(Qs[i0 + 4]));
        mmaTf32(dpt[jj], va[s][0], va[s][1], va[s][2], va[s][3], __float_as_uint(Os[i0]), __float_as_uint(Os[i0 + 4]));
      }
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int ql = jj * 8 + 2 * t;
      const float2 L = *reinterpret_cast<const float2*>(Ls + ql), Lg = *reinterpret_cast<const float2*>(Ls + QT + ql),
                   Dq = *reinterpret_cast<const float2*>(Ls + 2 * QT + ql);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int hk = e >> 1, par = e & 1;
        float* tp = Ts + (ql + par) * LT + kl0 + g + hk * 8;
        const float x = fmaf(st[jj][e], sc, kmR[hk] + *tp);
        const float p = ex2(fmaf(x - (par ? L.y : L.x), LOG2E, -LOG2E * (par ? Lg.y : Lg.x)));
        const float ds = p * (dpt[jj][e] - (par ? Dq.y : Dq.x));
        st[jj][e] = p;
        dpt[jj][e] = ds;
        *tp = ds;
      }
    }
    // dV += P^T dO, dK += dS^T Q: A from the accumulators (query column t is query 2t, t + 4 is 2t + 1)
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const uint32_t p0 = tf32(st[jj][0]), p1 = tf32(st[jj][2]), p2 = tf32(st[jj][1]), p3 = tf32(st[jj][3]);
      const uint32_t d0 = tf32(dpt[jj][0]), d1 = tf32(dpt[jj][2]), d2 = tf32(dpt[jj][1]), d3 = tf32(dpt[jj][3]);
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const int i0 = (jj * 8 + 2 * t) * LD + n * 8 + g;
        mmaTf32(dva[n], p0, p1, p2, p3, __float_as_uint(Os[i0]), __float_as_uint(Os[i0 + LD]));
        mmaTf32(dka[n], d0, d1, d2, d3, __float_as_uint(Qs[i0]), __float_as_uint(Qs[i0 + LD]));
      }
    }
    __syncthreads();   // (the whole tile's dS in Ts; Q and dO read for the last time)
    if (g0)
      for (int i = threadIdx.x; i < QT * BC; i += THREADS) {
        const int ql = i / BC, kl = i % BC;
        const float ds = Ts[ql * LT + kl];
        if (ds != 0.f && q0 + ql < a.Q) atomicAdd(g0 + (long long)(q0 + ql) * a.bias[0].sq + kl, ds);
      }
    {   // dQ for 16 of the tile's queries and every other n-tile, into Qs (scaled), then coalesced atomics
      const int qr = (warp & 1) * 16;
      constexpr int NH = (NT + 1) / 2;
      float dqa[NH][4];
#pragma unroll
      for (int u = 0; u < NH; ++u) dqa[u][0] = dqa[u][1] = dqa[u][2] = dqa[u][3] = 0.f;
#pragma unroll
      for (int s = 0; s < BC / 8; ++s) {
        const int c = s * 8 + 2 * t;
        const uint32_t a0 = tf32(Ts[(qr + g) * LT + c]), a1 = tf32(Ts[(qr + g + 8) * LT + c]),
                       a2 = tf32(Ts[(qr + g) * LT + c + 1]), a3 = tf32(Ts[(qr + g + 8) * LT + c + 1]);
#pragma unroll
        for (int u = 0; u < NH; ++u) {
          const int n = (warp >> 1) + 2 * u;
          if (n < NT)
            mmaTf32(dqa[u], a0, a1, a2, a3, __float_as_uint(Ks[c * LD + n * 8 + g]), __float_as_uint(Ks[(c + 1) * LD + n * 8 + g]));
        }
      }
#pragma unroll
      for (int u = 0; u < NH; ++u) {
        const int n = (warp >> 1) + 2 * u;
        if (n >= NT) continue;
        *reinterpret_cast<float2*>(Qs + (qr + g) * LD + n * 8 + 2 * t) = make_float2(dqa[u][0] * sc, dqa[u][1] * sc);
        *reinterpret_cast<float2*>(Qs + (qr + g + 8) * LD + n * 8 + 2 * t) = make_float2(dqa[u][2] * sc, dqa[u][3] * sc);
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < QT * D; i += THREADS) {
      const int r = i / D, d = i % D;
      if (q0 + r < a.Q) atomicAdd(dqb + (long long)(q0 + r) * lddq + d, Qs[r * LD + d]);
    }
    __syncthreads();   // (this stage is the next issue's)
  }
  float* dkb = dk + (size_t)b * a.K * lddk + h * D;
  float* dvb = dv + (size_t)b * a.K * lddv + h * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int kk = k0 + kl0 + g + r * 8;
    if (kk >= a.K) continue;
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        float* pk = dkb + (long long)kk * lddk + n * 8 + 2 * t + c;
        float* pv = dvb + (long long)kk * lddv + n * 8 + 2 * t + c;
        const float xk = dka[n][r * 2 + c] * sc, xv = dva[n][r * 2 + c];
        *pk = accK ? *pk + xk : xk;
        *pv = accV ? *pv + xv : xv;
      }
  }
}
template <int D>
inline bool backwardT(const Sdpa& a, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
                      const float* dO, int lddo, const float* lse, const float* Di, float* dq, int lddq, float* dk,
                      int lddk, float* dv, int lddv, bool accK, bool accV) {
  const Bias& bi = a.bias[0];
  auto al = [](const void* p) { return ((uintptr_t)p & 15) == 0; };
  if (a.clip || a.seqQ || a.seqK || a.bias[1].p || !bi.p || D % 8 || a.K % 4 || bi.sq % 4 || bi.sb % 4 || bi.sh % 4 ||
      ldq % 4 || lddo % 4 || !al(bi.p) || !al(q) || !al(dO))
    return false;
  constexpr size_t smem = sizeof(float) * (2 * (2 * 32 * (D + PADK) + 32 * (BC + PADK) + 3 * 32) + BC * (D + PADK));
  static bool attr = false;
  if (!attr) { GK(cudaFuncSetAttribute(bwdT<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  const size_t grid = (size_t)((a.K + BC - 1) / BC) * a.H * a.B;
  bwdT<D><<<(unsigned)grid, THREADS, smem, STREAM>>>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk,
                                                     dv, lddv, accK, accV);
  return true;
}

template <int D, int MODE>
inline size_t bwdSmem() { return sizeof(float) * ((MODE == 1 ? 7 : 3) * 64 * (D + PADK) + BC * (BR + PADK) + 3 * BR); }

template <int D, int MODE>
inline void forward(const Sdpa& a, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv, float* o,
                    int ldo, float* lse) {
  constexpr bool SPLIT = MODE == 1;
  static bool attr = false;
  size_t smem = sizeof(float) * ((SPLIT ? 5 : 2) * BC * (D + PADK) + BR * (BC + PADK));
  if (!attr) { GK(cudaFuncSetAttribute(fwdK<D, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  size_t grid = (size_t)((a.Q + BR - 1) / BR) * a.H * a.B;
  fwdK<D, MODE><<<(unsigned)grid, THREADS, smem, STREAM>>>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse);
}
template <int D, int MODE>
inline void backward(const Sdpa& a, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
                     const float* dO, int lddo, const float* lse, const float* Di, float* dq, int lddq, float* dk,
                     int lddk, float* dv, int lddv, bool accK, bool accV) {
  constexpr bool SPLIT = MODE == 1;
  static bool attr = false;
  size_t smem = bwdSmem<D, MODE>();
  if (!attr) {
    GK(cudaFuncSetAttribute(bwdK<D, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    GK(cudaFuncSetAttribute(bwdK<D, MODE>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    attr = true;
  }
  size_t grid = (size_t)((a.K + BC - 1) / BC) * a.H * a.B;
  bwdK<D, MODE><<<(unsigned)grid, THREADS, smem, STREAM>>>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq,
                                                            dk, lddk, dv, lddv, accK, accV);
}

// dispatch on D (a multiple of 8, <= 64) and the precision; false when the shape has no kernel here
#define FLASH_DISPATCH(D_, CALL)                                         \
  switch (D_) {                                                          \
    case 8: { constexpr int DD = 8; CALL; } return true;                 \
    case 16: { constexpr int DD = 16; CALL; } return true;               \
    case 24: { constexpr int DD = 24; CALL; } return true;               \
    case 32: { constexpr int DD = 32; CALL; } return true;               \
    case 48: { constexpr int DD = 48; CALL; } return true;               \
    case 64: { constexpr int DD = 64; CALL; } return true;               \
    default: return false;                                               \
  }
inline bool supported(int D) { return D == 8 || D == 16 || D == 24 || D == 32 || D == 48 || D == 64; }
inline bool runForward(const Sdpa& a, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
                       float* o, int ldo, float* lse) {
  if (BF16 && a.D == 8) { forward<8, 2>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse); return true; }   // (m16n8k8 fits D 8 unpadded)
  if (BF16) { FLASH_DISPATCH(a.D, (forwardBf<DD>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse))) }
  else if (TF32) { FLASH_DISPATCH(a.D, (forwardT<DD>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse) || (forward<DD, 0>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse), true))) }
  else { FLASH_DISPATCH(a.D, (forward<DD, 1>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse))) }
}
inline bool runBackward(const Sdpa& a, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
                        const float* dO, int lddo, const float* lse, const float* Di, float* dq, int lddq, float* dk,
                        int lddk, float* dv, int lddv, bool accK, bool accV) {
  if (BF16 && a.D == 8) { backward<8, 2>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV); return true; }
  if (BF16) { FLASH_DISPATCH(a.D, (backwardBf<DD>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV))) }
  else if (TF32) { FLASH_DISPATCH(a.D, (backwardT<DD>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV) || (backward<DD, 0>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV), true))) }
  else { FLASH_DISPATCH(a.D, (backward<DD, 1>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV))) }
}
// q, k, v already bf16 (a module on bf16 storage): the bf16 kernels, whatever D they support (8 included, padded)
template <class OT>
inline bool runForwardBfIn(const Sdpa& a, const __nv_bfloat16* q, int ldq, const __nv_bfloat16* k, int ldk,
                           const __nv_bfloat16* v, int ldv, OT* o, int ldo, float* lse) {
  FLASH_DISPATCH(a.D, (forwardBf<DD>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse)))
}
template <class DKV, class DOT>
inline bool runBackwardBfIn(const Sdpa& a, const __nv_bfloat16* q, int ldq, const __nv_bfloat16* k, int ldk,
                            const __nv_bfloat16* v, int ldv, const DOT* dO, int lddo, const float* lse,
                            const float* Di, float* dq, int lddq, DKV* dk, int lddk, DKV* dv, int lddv, bool accK,
                            bool accV, __nv_bfloat16* dqo = nullptr, int ldqo = 0) {
  FLASH_DISPATCH(a.D, (backwardBf<DD>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV, dqo, ldqo)))
}
#undef FLASH_DISPATCH

}  // namespace flash
}  // namespace grad
