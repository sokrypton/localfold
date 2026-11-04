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

template <int D>
__global__ void __launch_bounds__(THREADS) fwdBf(Sdpa a, BiasTile bt, const uint16_t* __restrict__ qh,
                                                 const uint16_t* __restrict__ kh, const uint16_t* __restrict__ vh,
                                                 int Npq, int Npk, float* __restrict__ o, int ldo,
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
  const uint16_t* kp = kh + bh * Npk * DP;
  const uint16_t* vp = vh + bh * Npk * DP;
  const uint16_t* tp = bt.p ? bt.p + b * bt.sb + h * bt.sh + (size_t)q0 * bt.ld : nullptr;
  auto issue = [&](int st, int k0) {
    uint16_t* Ks = Kb + st * BC * LK;
    uint16_t* Vs = Vb + st * BC * LK;
    for (int i = threadIdx.x; i < BC * DP / 8; i += THREADS) {
      const int r = i / (DP / 8), c = (i % (DP / 8)) * 8;
      cp16(Ks + r * LK + c, kp + (size_t)(k0 + r) * DP + c, true);
      cp16(Vs + r * LK + c, vp + (size_t)(k0 + r) * DP + c, true);
    }
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
  for (int s = 0; s < DP / 16; ++s) aFragH<DP>(qa[s], qh + bh * Npq * DP, qr0 + g, s * 16 + 2 * t);
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
  float* ob = o + (size_t)b * a.Q * ldo + h * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int qq = qr0 + g + r * 8;
    if (qq >= a.Q) continue;
    const float inv = sum[r] > 0.f ? 1.f / sum[r] : 0.f;
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      ob[(long long)qq * ldo + n * 8 + 2 * t] = oacc[n][r * 2] * inv;
      ob[(long long)qq * ldo + n * 8 + 2 * t + 1] = oacc[n][r * 2 + 1] * inv;
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
template <int D>
__global__ void __launch_bounds__(THREADS, 3) bwdBf(Sdpa a, BiasTile bt, const uint16_t* __restrict__ qh,
                                                    const uint16_t* __restrict__ kh, const uint16_t* __restrict__ vh,
                                                    const uint16_t* __restrict__ doh, int Npq, int Npk,
                                                    const float* __restrict__ lse, const float* __restrict__ Di,
                                                    float* __restrict__ dq, int lddq, float* __restrict__ dk, int lddk,
                                                    float* __restrict__ dv, int lddv, bool accK, bool accV) {
  constexpr int DP = padD<D>(), LK = ldBf<D>();
  extern __shared__ __align__(16) float sm[];
  uint16_t* Kb = reinterpret_cast<uint16_t*>(sm);                 // [BC][LK]
  uint16_t* Qb = Kb + BC * LK;                                    // [2][BR][LK]
  uint16_t* Ob = Qb + 2 * BR * LK;                                // dO [2][BR][LK]
  uint16_t* Tb = Ob + 2 * BR * LK;                                // [2][BR][LB]: the bias tile, then dS
  float* QV = reinterpret_cast<float*>(Tb + 2 * BR * LB);         // [2][4][BR]: row max, log sum, Di, seqQ
  const int nkb = (a.K + BC - 1) / BC;
  const int kb = blockIdx.x % nkb, h = (blockIdx.x / nkb) % a.H, b = blockIdx.x / (nkb * a.H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int k0 = kb * BC, kl0 = warp * 16;
  const size_t bh = (size_t)b * a.H + h;
  const uint16_t* qp = qh + bh * Npq * DP;
  const uint16_t* op = doh + bh * Npq * DP;
  const uint16_t* kp = kh + bh * Npk * DP;
  const uint16_t* tp = bt.p ? bt.p + b * bt.sb + h * bt.sh + k0 : nullptr;
  const float* lseb = lse + bh * a.Q;
  const float* lgb = lseb + (size_t)a.B * a.H * a.Q;
  const float* Dib = Di + bh * a.Q;
  for (int i = threadIdx.x; i < BC * DP / 8; i += THREADS) {
    const int r = i / (DP / 8), c = (i % (DP / 8)) * 8;
    cp16(Kb + r * LK + c, kp + (size_t)(k0 + r) * DP + c, true);
  }
  auto issue = [&](int st, int q0) {
    uint16_t* Qs = Qb + st * BR * LK;
    uint16_t* Os = Ob + st * BR * LK;
    for (int i = threadIdx.x; i < BR * DP / 8; i += THREADS) {
      const int r = i / (DP / 8), c = (i % (DP / 8)) * 8;
      cp16(Qs + r * LK + c, qp + (size_t)(q0 + r) * DP + c, true);
      cp16(Os + r * LK + c, op + (size_t)(q0 + r) * DP + c, true);
    }
    if (tp) {
      uint16_t* T = Tb + st * BR * LB;
      for (int i = threadIdx.x; i < BR * BC / 8; i += THREADS) {
        const int r = i / (BC / 8), c = (i % (BC / 8)) * 8;
        cp16(T + r * LB + c, tp + (size_t)(q0 + r) * bt.ld + c, true);
      }
    }
    if (threadIdx.x < BR) {
      float* qv = QV + st * 4 * BR;
      const int j = threadIdx.x, q = q0 + j;
      const bool in = q < a.Q;
      cp4(qv + j, in ? lseb + q : lseb, in);
      cp4(qv + BR + j, in ? lgb + q : lgb, in);
      cp4(qv + 2 * BR + j, in ? Dib + q : Dib, in);
      if (a.seqQ) cp4(qv + 3 * BR + j, in ? a.seqQ + q : a.seqQ, in);
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
    ksR[r] = a.seqK && k < a.K ? a.seqK[k] : 0;
  }
  uint32_t ka[DP / 16][4], va[DP / 16][4];
#pragma unroll
  for (int s = 0; s < DP / 16; ++s) {
    aFragH<DP>(ka[s], kp, k0 + kl0 + g, s * 16 + 2 * t);
    aFragH<DP>(va[s], vh + bh * Npk * DP, k0 + kl0 + g, s * 16 + 2 * t);
  }
  float dka[D / 8][4], dva[D / 8][4];
#pragma unroll
  for (int n = 0; n < D / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dka[n][e] = dva[n][e] = 0.f;
  float* dqb = dq + (size_t)b * a.Q * lddq + h * D;
  real* g0 = a.bias[0].grad ? a.bias[0].grad + b * a.bias[0].sb + h * a.bias[0].sh + k0 : nullptr;
  real* g1 = a.bias[1].grad ? a.bias[1].grad + b * a.bias[1].sb + h * a.bias[1].sh + k0 : nullptr;
  const int nq = (a.Q + BR - 1) / BR;
  for (int it = 0; it < nq; ++it) {
    const int st = it & 1, q0 = it * BR;
    if (it + 1 < nq) issue(st ^ 1, q0 + BR);
    cpCommit();
    cpWait1();
    __syncthreads();
    const uint16_t* Qs = Qb + st * BR * LK;
    const uint16_t* Os = Ob + st * BR * LK;
    uint16_t* T = Tb + st * BR * LB;
    const float* qv = QV + st * 4 * BR;
#pragma unroll 1
    for (int hq = 0; hq < 2; ++hq) {
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
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int kl = kl0 + g + (e >> 1) * 8, ql = (hq * 4 + jj) * 8 + 2 * t + (e & 1);
          float add = kaR[e >> 1];
          if (add != -INFINITY) {
            if (a.seqQ && ksR[e >> 1] != __float_as_int(qv[3 * BR + ql])) add = -INFINITY;
            else if (tp) add += bfToF(T[ql * LB + kl]);
          }
          const float x = logit(sa[jj][e], add, a);
          const float p = x == -INFINITY || q0 + ql >= a.Q ? 0.f : ex2(fmaf(x - qv[ql], LOG2E, -LOG2E * qv[BR + ql]));
          const float ds = p * (dpt[jj][e] - qv[2 * BR + ql]);
          T[ql * LB + kl] = (uint16_t)(bf2(ds, 0.f) & 0xffff);
          sa[jj][e] = p;
          dpt[jj][e] = ds;
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
    }
    __syncthreads();
    if (g0 || g1)
      for (int i = threadIdx.x; i < BR * BC; i += THREADS) {
        const int ql = i / BC, kl = i % BC;
        const float ds = bfToF(T[ql * LB + kl]);
        if (ds != 0.f) {
          const long long qq = q0 + ql;
          if (g0) atomicAdd(g0 + qq * a.bias[0].sq + kl, ds);
          if (g1) atomicAdd(g1 + qq * a.bias[1].sq + kl, ds);
        }
      }
    // dQ for this warp's 16 queries over the block's 64 keys, added straight from the accumulators (a warp's
    // atomics cover eight rows' 32-byte sectors whole)
    const int qw = warp * 16;
    float dqa[D / 8][4];
#pragma unroll
    for (int n = 0; n < D / 8; ++n) dqa[n][0] = dqa[n][1] = dqa[n][2] = dqa[n][3] = 0.f;
#pragma unroll
    for (int s = 0; s < BC / 16; ++s) {
      uint32_t A[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) A[i] = word(T, (qw + g + (i & 1) * 8) * LB + s * 16 + 2 * t + (i >> 1) * 8);
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        uint32_t b0, b1;
        ldT(b0, b1, Kb, s * 16, n * 8, LK);
        mma16(dqa[n], A, b0, b1);
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int qq = q0 + qw + g + r * 8;
      if (qq >= a.Q) continue;
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        atomicAdd(dqb + (long long)qq * lddq + n * 8 + 2 * t, dqa[n][r * 2] * a.scale);
        atomicAdd(dqb + (long long)qq * lddq + n * 8 + 2 * t + 1, dqa[n][r * 2 + 1] * a.scale);
      }
    }
    __syncthreads();   // this stage is the one the next iteration's issue overwrites
  }
  float* dkb = dk + (size_t)b * a.K * lddk + h * D;
  float* dvb = dv + (size_t)b * a.K * lddv + h * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int kk = k0 + kl0 + g + r * 8;
    if (kk >= a.K) continue;
#pragma unroll
    for (int n = 0; n < D / 8; ++n)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        float* pk = dkb + (long long)kk * lddk + n * 8 + 2 * t + c;
        float* pv = dvb + (long long)kk * lddv + n * 8 + 2 * t + c;
        const float xk = dka[n][r * 2 + c] * a.scale, xv = dva[n][r * 2 + c];
        *pk = accK ? *pk + xk : xk;
        *pv = accV ? *pv + xv : xv;
      }
  }
}
template <int D>
inline size_t bwdBfSmem() { return 2 * (5 * BR * ldBf<D>() + 2 * BR * LB) + 4 * (2 * 4 * BR); }
template <int D, class X>
inline void forwardBf(const Sdpa& a, const X* q, int ldq, const X* k, int ldk, const X* v, int ldv,
                      float* o, int ldo, float* lse) {
  Scratch scratch;
  constexpr int DP = padD<D>();
  const int Npq = (a.Q + 63) / 64 * 64, Npk = (a.K + 63) / 64 * 64;
  const uint16_t* qh = headsBf(q, ldq, a.B, a.Q, a.H, D, Npq, DP);
  const uint16_t* kh = headsBf(k, ldk, a.B, a.K, a.H, D, Npk, DP);
  const uint16_t* vh = headsBf(v, ldv, a.B, a.K, a.H, D, Npk, DP);
  static bool attr = false;
  const size_t smem = fwdBfSmem<D>();
  if (!attr) { GK(cudaFuncSetAttribute(fwdBf<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  const BiasTile bt = biasBf(a, Npq, Npk);
  const size_t grid = (size_t)((a.Q + BR - 1) / BR) * a.H * a.B;
  fwdBf<D><<<(unsigned)grid, THREADS, smem, STREAM>>>(a, bt, qh, kh, vh, Npq, Npk, o, ldo, lse);
}
template <int D, class X>
inline void backwardBf(const Sdpa& a, const X* q, int ldq, const X* k, int ldk, const X* v, int ldv,
                       const float* dO, int lddo, const float* lse, const float* Di, float* dq, int lddq, float* dk,
                       int lddk, float* dv, int lddv, bool accK, bool accV) {
  Scratch scratch;
  constexpr int DP = padD<D>();
  const int Npq = (a.Q + 63) / 64 * 64, Npk = (a.K + 63) / 64 * 64;
  const uint16_t* qh = headsBf(q, ldq, a.B, a.Q, a.H, D, Npq, DP);
  const uint16_t* oh = headsBf(dO, lddo, a.B, a.Q, a.H, D, Npq, DP);
  const uint16_t* kh = headsBf(k, ldk, a.B, a.K, a.H, D, Npk, DP);
  const uint16_t* vh = headsBf(v, ldv, a.B, a.K, a.H, D, Npk, DP);
  static bool attr = false;
  const size_t smem = bwdBfSmem<D>();
  if (!attr) {
    GK(cudaFuncSetAttribute(bwdBf<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    GK(cudaFuncSetAttribute(bwdBf<D>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    attr = true;
  }
  const BiasTile bt = biasBf(a, Npq, Npk);
  const size_t grid = (size_t)((a.K + BC - 1) / BC) * a.H * a.B;
  bwdBf<D><<<(unsigned)grid, THREADS, smem, STREAM>>>(a, bt, qh, kh, vh, oh, Npq, Npk, lse, Di, dq, lddq, dk,
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
// q, k, v already bf16 (a module on bf16 storage): the bf16 kernels, whatever D they support (8 included, padded)
inline bool runForwardBfIn(const Sdpa& a, const __nv_bfloat16* q, int ldq, const __nv_bfloat16* k, int ldk,
                           const __nv_bfloat16* v, int ldv, float* o, int ldo, float* lse) {
  FLASH_DISPATCH(a.D, (forwardBf<DD>(a, q, ldq, k, ldk, v, ldv, o, ldo, lse)))
}
inline bool runBackwardBfIn(const Sdpa& a, const __nv_bfloat16* q, int ldq, const __nv_bfloat16* k, int ldk,
                            const __nv_bfloat16* v, int ldv, const float* dO, int lddo, const float* lse,
                            const float* Di, float* dq, int lddq, float* dk, int lddk, float* dv, int lddv, bool accK,
                            bool accV) {
  FLASH_DISPATCH(a.D, (backwardBf<DD>(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV)))
}
#undef FLASH_DISPATCH

}  // namespace flash
}  // namespace grad
