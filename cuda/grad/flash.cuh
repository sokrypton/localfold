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

// ---- bf16: its own kernels, not MODE 2 of the ones above. The tiles are rounded to bf16 once, as they are staged
// (MODE 2 kept them float and rounded at every fragment read), the products are m16n8k16 (twice m16n8k8's k a
// fragment), and P and dS go from the accumulators straight into the next product's A operand - an m16n8k16 C pair
// of n-tiles is exactly an A fragment - so P never touches shared memory. Every B operand is read as one 32-bit
// word, which needs its contraction axis contiguous: a tile contracted over D is staged [row][D], one contracted
// over its rows [D][row]. D pads to a multiple of 16 with zeros where it is contracted.
__device__ __forceinline__ void mma16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t word(const uint16_t* s, int i) { return *reinterpret_cast<const uint32_t*>(s + i); }
// the A fragment of rows r, r + 8 and columns c .. c + 15 of x [rows][ld] (zero past `rows` and past D)
template <int D>
__device__ __forceinline__ void aFrag(uint32_t* f, const float* x, long long ld, int r, int rows, int c) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int rr = r + (i & 1) * 8, cc = c + (i >> 1) * 8;
    f[i] = rr < rows && cc < D ? bf2(x[(long long)rr * ld + cc], x[(long long)rr * ld + cc + 1]) : 0u;
  }
}
// two accumulator n-tiles (8 columns each) as one A fragment over their 16 columns
__device__ __forceinline__ void cToA(uint32_t* f, const float* c0, const float* c1) {
  f[0] = bf2(c0[0], c0[1]); f[1] = bf2(c0[2], c0[3]); f[2] = bf2(c1[0], c1[1]); f[3] = bf2(c1[2], c1[3]);
}
constexpr int LR = 64 + 8;                         // a [D][64 rows] tile's row stride, in bf16
template <int D> __host__ __device__ constexpr int ldBf() { return (D + 15) / 16 * 16 + 8; }   // a [64][D] tile's, padded D + 8
// rows [r0, r0 + 64) of x (row stride ld, D columns): S [64][ldBf] (zero past D to the pad) and/or St [D][LR],
// two rows by two columns a thread so every store is a word
template <int D>
__device__ __forceinline__ void loadBf(uint16_t* S, uint16_t* St, const float* x, long long ld, int r0, int rows) {
  constexpr int DP = (D + 15) / 16 * 16, LK = ldBf<D>();
  const bool v2 = (ld & 1) == 0 && ((uintptr_t)x & 7) == 0;
  constexpr int N = 32 * (DP / 2) / THREADS;       // every load issued before the first store
  static_assert(32 * (DP / 2) % THREADS == 0, "whole passes");
  float2 u[N], w[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    const int i = threadIdx.x + j * THREADS, r = 2 * (i / (DP / 2)), c = 2 * (i % (DP / 2));
    u[j] = w[j] = make_float2(0.f, 0.f);
    if (c < D) {
      const float* p = x + (long long)(r0 + r) * ld + c;
      if (r0 + r < rows) u[j] = v2 ? *reinterpret_cast<const float2*>(p) : make_float2(p[0], p[1]);
      if (r0 + r + 1 < rows) w[j] = v2 ? *reinterpret_cast<const float2*>(p + ld) : make_float2(p[ld], p[ld + 1]);
    }
  }
#pragma unroll
  for (int j = 0; j < N; ++j) {
    const int i = threadIdx.x + j * THREADS, r = 2 * (i / (DP / 2)), c = 2 * (i % (DP / 2));
    if (S) {
      *reinterpret_cast<uint32_t*>(S + r * LK + c) = bf2(u[j].x, u[j].y);
      *reinterpret_cast<uint32_t*>(S + (r + 1) * LK + c) = bf2(w[j].x, w[j].y);
    }
    if (St && c < D) {
      *reinterpret_cast<uint32_t*>(St + c * LR + r) = bf2(u[j].x, w[j].x);
      *reinterpret_cast<uint32_t*>(St + (c + 1) * LR + r) = bf2(u[j].y, w[j].y);
    }
  }
}

template <int D>
__global__ void __launch_bounds__(THREADS) fwdBf(Sdpa a, const float* __restrict__ q, int ldq,
                                                 const float* __restrict__ k, int ldk, const float* __restrict__ v,
                                                 int ldv, float* __restrict__ o, int ldo, float* __restrict__ lse) {
  constexpr int DP = (D + 15) / 16 * 16, LK = ldBf<D>(), LT = BC + PADK;
  extern __shared__ __align__(16) float sm[];
  float* T = sm;                                                  // [BR][LT]: the additive terms
  uint16_t* Ks = reinterpret_cast<uint16_t*>(T + BR * LT);        // [BC][LK]
  uint16_t* Vt = Ks + BC * LK;                                    // [D][LR]
  const int nqb = (a.Q + BR - 1) / BR;
  const int qb = blockIdx.x % nqb, h = (blockIdx.x / nqb) % a.H, b = blockIdx.x / (nqb * a.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int q0 = qb * BR, qr0 = q0 + warp * 16;
  const float* qb_ = q + (size_t)b * a.Q * ldq + h * D;
  const float* kb_ = k + (size_t)b * a.K * ldk + h * D;
  const float* vb_ = v + (size_t)b * a.K * ldv + h * D;
  uint32_t qa[DP / 16][4];
#pragma unroll
  for (int s = 0; s < DP / 16; ++s) aFrag<D>(qa[s], qb_, ldq, qr0 + g, a.Q, s * 16 + 2 * t);
  float oacc[D / 8][4];
#pragma unroll
  for (int n = 0; n < D / 8; ++n) oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f;
  float mx[2] = {-INFINITY, -INFINITY}, sum[2] = {0.f, 0.f};
  const float* P = T + warp * 16 * LT;
  for (int k0 = 0; k0 < a.K; k0 += BC) {
    __syncthreads();
    loadBf<D>(Ks, nullptr, kb_, ldk, k0, a.K);
    loadBf<D>(nullptr, Vt, vb_, ldv, k0, a.K);
    loadAdd(T, a, b, h, q0, k0);
    __syncthreads();
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
        int r = e >> 1;
        float x = logit(sacc[j][e], P[(g + r * 8) * LT + j * 8 + 2 * t + (e & 1)], a);
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
        float p = sacc[j][e] == -INFINITY ? 0.f : expf(sacc[j][e] - nm[e >> 1]);
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
        const int i = (n * 8 + g) * LR + u * 16 + 2 * t;
        mma16(oacc[n], pa, word(Vt, i), word(Vt, i + 8));
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
    if (t == 0) {
      const size_t i = ((size_t)b * a.H + h) * a.Q + qq, N = (size_t)a.B * a.H * a.Q;
      lse[i] = sum[r] > 0.f ? mx[r] : -INFINITY;
      lse[N + i] = sum[r] > 0.f ? logf(sum[r]) : 0.f;
    }
  }
}
template <int D>
inline size_t fwdBfSmem() { return sizeof(float) * BR * (BC + PADK) + 2 * (BC * ldBf<D>() + D * LR); }

// the backward as bwdK lays it out (a CTA a key block, a warp 16 keys, the queries in halves of 32), in bf16:
// S^T = K Q^T and dP^T = V dO^T (A = K, V from registers; B = Q, dO [query][D]); dV += P^T dO and dK += dS^T Q
// (A = P^T, dS^T from the accumulators; B = dO^T, Q^T [D][query]); dQ = dS K (A = dS from T in float, B = K^T).
// dS is kept in float in T for the bias gradients.
template <int D>
__global__ void __launch_bounds__(THREADS, D <= 32 ? 3 : 2) bwdBf(Sdpa a, const float* __restrict__ q, int ldq,
                                                   const float* __restrict__ k, int ldk, const float* __restrict__ v,
                                                   int ldv, const float* __restrict__ dO, int lddo,
                                                   const float* __restrict__ lse, const float* __restrict__ Di,
                                                   float* __restrict__ dq, int lddq, float* __restrict__ dk, int lddk,
                                                   float* __restrict__ dv, int lddv, bool accK, bool accV) {
  constexpr int DP = (D + 15) / 16 * 16, LK = ldBf<D>(), LT = BC + PADK, LS = D + PADK;
  extern __shared__ __align__(16) float sm[];
  float* T = sm;                                   // [BR][LT]: the additive terms, then dS
  float* Ls = T + BR * LT;                         // lse [BR]
  float* Ds = Ls + BR;                             // Di [BR]
  float* Lg = Ds + BR;                             // log sum [BR]
  float* St = Lg + BR;                             // [4 warps][16][LS]: dQ staged for coalesced atomics
  uint16_t* Qs = reinterpret_cast<uint16_t*>(St + 4 * 16 * LS);   // [BR][LK]
  uint16_t* Os = Qs + BR * LK;                                    // dO [BR][LK]
  uint16_t* Qt = Os + BR * LK;                                    // [D][LR]
  uint16_t* Ot = Qt + D * LR;                                     // [D][LR]
  uint16_t* Kt = Ot + D * LR;                                     // [D][LR]
  const int nkb = (a.K + BC - 1) / BC;
  const int kb = blockIdx.x % nkb, h = (blockIdx.x / nkb) % a.H, b = blockIdx.x / (nkb * a.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int k0 = kb * BC, kl0 = warp * 16;
  const float* qb_ = q + (size_t)b * a.Q * ldq + h * D;
  const float* ob_ = dO + (size_t)b * a.Q * lddo + h * D;
  const float* vb_ = v + (size_t)b * a.K * ldv + h * D;
  const float* kb_ = k + (size_t)b * a.K * ldk + h * D;
  loadBf<D>(nullptr, Kt, kb_, ldk, k0, a.K);
  uint32_t ka[DP / 16][4], va[DP / 16][4];
#pragma unroll
  for (int s = 0; s < DP / 16; ++s) {
    aFrag<D>(ka[s], kb_, ldk, k0 + kl0 + g, a.K, s * 16 + 2 * t);
    aFrag<D>(va[s], vb_, ldv, k0 + kl0 + g, a.K, s * 16 + 2 * t);
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
    loadBf<D>(Qs, Qt, qb_, ldq, q0, a.Q);
    loadBf<D>(Os, Ot, ob_, lddo, q0, a.Q);
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
      for (int s = 0; s < DP / 16; ++s)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int i = ((hq * 4 + jj) * 8 + g) * LK + s * 16 + 2 * t;
          mma16(st[jj], ka[s], word(Qs, i), word(Qs, i + 8));
          mma16(dpt[jj], va[s], word(Os, i), word(Os, i + 8));
        }
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          int kl = kl0 + g + (e >> 1) * 8, ql = (hq * 4 + jj) * 8 + 2 * t + (e & 1);
          float x = logit(st[jj][e], T[ql * LT + kl], a);
          float p = x == -INFINITY ? 0.f : expf((x - Ls[ql]) - Lg[ql]);
          float ds = p * (dpt[jj][e] - Ds[ql]);
          T[ql * LT + kl] = ds;
          st[jj][e] = p;
          dpt[jj][e] = ds;
        }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t pa[4], sa[4];
        cToA(pa, st[2 * u], st[2 * u + 1]);
        cToA(sa, dpt[2 * u], dpt[2 * u + 1]);
#pragma unroll
        for (int n = 0; n < D / 8; ++n) {
          const int i = (n * 8 + g) * LR + hq * 32 + u * 16 + 2 * t;
          mma16(dva[n], pa, word(Ot, i), word(Ot, i + 8));
          mma16(dka[n], sa, word(Qt, i), word(Qt, i + 8));
        }
      }
    }
    __syncthreads();
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
    const int qw = warp * 16;
    float dqa[D / 8][4];
#pragma unroll
    for (int n = 0; n < D / 8; ++n) dqa[n][0] = dqa[n][1] = dqa[n][2] = dqa[n][3] = 0.f;
#pragma unroll
    for (int s = 0; s < BC / 16; ++s) {
      uint32_t A[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 x = *reinterpret_cast<const float2*>(T + (qw + g + (i & 1) * 8) * LT + s * 16 + 2 * t + (i >> 1) * 8);
        A[i] = bf2(x.x, x.y);
      }
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        const int i = (n * 8 + g) * LR + s * 16 + 2 * t;
        mma16(dqa[n], A, word(Kt, i), word(Kt, i + 8));
      }
    }
    float* stw = St + warp * 16 * LS;
#pragma unroll
    for (int n = 0; n < D / 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) stw[(g + (e >> 1) * 8) * LS + n * 8 + 2 * t + (e & 1)] = dqa[n][e] * a.scale;
    __syncwarp();
    for (int i = lane; i < 16 * D; i += 32) {
      int r = i / D, d = i % D, qq = q0 + qw + r;
      if (qq < a.Q) atomicAdd(dqb + (long long)qq * lddq + d, stw[r * LS + d]);
    }
  }
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
template <int D>
inline size_t bwdBfSmem() {
  return sizeof(float) * (BR * (BC + PADK) + 3 * BR + 4 * 16 * (D + PADK)) + 2 * (2 * BR * ldBf<D>() + 3 * D * LR);
}
template <int D>
inline void forwardBf(const Sdpa& a, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
                      float* o, int ldo, float* lse) {
  static bool attr = false;
  size_t smem = fwdBfSmem<D>();
  if (!attr) { GK(cudaFuncSetAttribute(fwdBf<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  size_t grid = (size_t)((a.Q + BR - 1) / BR) * a.H * a.B;
  fwdBf<D><<<(unsigned)grid, THREADS, smem, STREAM>>>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse);
}
template <int D>
inline void backwardBf(const Sdpa& a, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
                       const float* dO, int lddo, const float* lse, const float* Di, float* dq, int lddq, float* dk,
                       int lddk, float* dv, int lddv, bool accK, bool accV) {
  static bool attr = false;
  size_t smem = bwdBfSmem<D>();
  if (!attr) {
    GK(cudaFuncSetAttribute(bwdBf<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    GK(cudaFuncSetAttribute(bwdBf<D>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    attr = true;
  }
  size_t grid = (size_t)((a.K + BC - 1) / BC) * a.H * a.B;
  bwdBf<D><<<(unsigned)grid, THREADS, smem, STREAM>>>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk,
                                                       lddk, dv, lddv, accK, accV);
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
  else if (TF32) { FLASH_DISPATCH(a.D, (forward<DD, 0>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse))) }
  else { FLASH_DISPATCH(a.D, (forward<DD, 1>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse))) }
}
inline bool runBackward(const Sdpa& a, const float* q, int ldq, const float* k, int ldk, const float* v, int ldv,
                        const float* dO, int lddo, const float* lse, const float* Di, float* dq, int lddq, float* dk,
                        int lddk, float* dv, int lddv, bool accK, bool accV) {
  if (BF16 && a.D == 8) { backward<8, 2>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV); return true; }
  if (BF16) { FLASH_DISPATCH(a.D, (backwardBf<DD>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV))) }
  else if (TF32) { FLASH_DISPATCH(a.D, (backward<DD, 0>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV))) }
  else { FLASH_DISPATCH(a.D, (backward<DD, 1>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV))) }
}
#undef FLASH_DISPATCH

}  // namespace flash
}  // namespace grad
