// The full ESMFold2's MSA encoder (ESMFold2-Fast has none), float32. biohub's modeling_esmfold2.py MSAEncoder,
// MSAEncoderBlock, OuterProductMean and MSAPairWeightedAveraging (modeling_esmfold2_common.py) are the reading:
//
//   m = embed([one-hot(33) * mask | has_deletion | deletion_value]) + project_inputs(s_inputs)[token]
//   per block:  pair += OPM(m)                     Wout(outer(a, b)) / max(pair count, 1), the bias divided too
//               (not the last)  m += PWA(m, pair)  softmax over j of a head's LN(pair) logits, weighting
//                                                  each row's values, gated; m += SwiGLU transition(m)
//               pair += tri out, tri in, pair transition      (the trunk's pair-only block)
//
// Its input pair is z_init and its answer REPLACES the injection every pass (msa_encoder_overwrite) - and
// since neither the alignment nor z_init changes between passes (the subsample and the column mask are drawn
// once a fold here, where biohub draws the column mask once a fold and the subsample every pass), it runs ONCE
// a fold, in place over z_init. Layout [row][token][channel] (cuda/af3's), which no einsum here minds.
#pragma once
#include "trunk.cuh"
#include <random>

inline bool hasMsaEncoder() { return M.has("meta/msaBlocks") && M.meta("meta/msaBlocks") > 0; }

// one-hot rows of the embedding where the column is live, the two deletion features always (biohub masks only
// the one-hot: "bias-free MSAEncoder.embed requires zeroed padding"), plus the projected s_inputs
__global__ void msaEmbedEf2K(const int* rows, const float* del, const float* mask, const float* E, const float* fromInputs,
                             float* m, size_t count, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count * C) return;
  int c = (int)(t % C); size_t row = t / C; int token = (int)(row % T);
  float d = del[row];
  float v = mask[row] > 0.f ? E[(size_t)rows[row] * C + c] : 0.f;
  if (d > 0.f) v += E[(size_t)33 * C + c];
  v += 1.57079632679489662f * atanf(d / 3.f) * E[(size_t)34 * C + c];
  m[t] = v + fromInputs[(size_t)token * C + c];
}
// x [rows][a | b] (2H) times the row's mask, split into a [rows][H] and b [rows][H]
__global__ void opmSplitK(const float* x, const float* mask, float* a, float* b, size_t rows, int Hh) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * Hh) return;
  size_t r = t / Hh; int c = (int)(t % Hh); float k = mask[r];
  a[t] = x[r * 2 * Hh + c] * k; b[t] = x[r * 2 * Hh + Hh + c] * k;
}
// P [(i, c)][(j, e)] -> Pp [(i, j)][(c, e)]
__global__ void opmPermuteEf2K(const float* P, float* Pp, int bi, int T, int Hh) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)bi * T * Hh * Hh) return;
  int e = (int)(t % Hh); size_t r = t / Hh; int c = (int)(r % Hh); r /= Hh; int j = (int)(r % T); int i = (int)(r / T);
  Pp[t] = P[(((size_t)i * Hh + c) * T + j) * Hh + e];
}
// pair[(i0 + i) j] += (X + b) / max(count, 1)
__global__ void opmAddClampedK(float* pair, const float* X, const float* b, const float* count, int i0, int bi, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)bi * T * C) return;
  int c = (int)(t % C); size_t ij = t / C;
  size_t at = (size_t)i0 * T + ij;
  pair[at * C + c] += (X[t] + b[c]) / fmaxf(count[at], 1.f);
}
inline void msaOuterProductMean(float* pair, const float* m, const float* mask, int S, int T, int Cm, int C,
                                const std::string& Bn) {
  const int Hh = (int)M.meta("meta/opmHidden");
  size_t rows = (size_t)S * T;
  float* ln = scratch<float>("msa.ln", rows * Cm);
  layerNorm(m, ln, rows, Cm, F(Bn + "normScale"), F(Bn + "normOffset"));
  float* x = scratch<float>("msa.opmx", rows * 2 * Hh);
  gemm(ln, F(Bn + "projection"), x, rows, Cm, 2 * Hh);
  float* a = scratch<float>("msa.opma", rows * Hh); float* b = scratch<float>("msa.opmb", rows * Hh);
  opmSplitK<<<blocks(rows * Hh), 256, 0, STREAM>>>(x, mask, a, b, rows, Hh);
  const float one = 1.f, zero = 0.f;
  float* count = scratch<float>("msa.count", (size_t)T * T);      // count[i][j] = sum_s mask[s][i] mask[s][j]
  CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, T, T, S, &one, mask, T, mask, T, &zero, count, T));
  size_t per = (size_t)T * (2 * Hh * Hh + C) * 4;                  // a query row's outer products and its update
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(T, ((size_t)256 << 20) / per));
  float* P = scratch<float>("msa.opmP", (size_t)Bi * T * Hh * Hh);
  float* Pp = scratch<float>("msa.opmPp", (size_t)Bi * T * Hh * Hh);
  float* X = scratch<float>("msa.opmX", (size_t)Bi * T * C);
  for (int i0 = 0; i0 < T; i0 += Bi) {
    int bi = std::min(Bi, T - i0);
    // row-major P [(i, c)][(j, e)] = sum_s a[s, i, c] b[s, j, e]; col-major P^T = b (T*H x S) a_blk^T
    CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, T * Hh, bi * Hh, S, &one, b, T * Hh, a + (size_t)i0 * Hh, T * Hh, &zero,
                   P, T * Hh));
    opmPermuteEf2K<<<blocks((size_t)bi * T * Hh * Hh), 256, 0, STREAM>>>(P, Pp, bi, T, Hh);
    gemm(Pp, F(Bn + "output"), X, (size_t)bi * T, Hh * Hh, C);
    opmAddClampedK<<<blocks((size_t)bi * T * C), 256, 0, STREAM>>>(pair, X, F(Bn + "outputBias"), count, i0, bi, T, C);
  }
}
// w[h][i][j] = softmax over j of flat[(i j)][h] (every token is real: no mask)
__global__ void pwaSoftmaxK(const float* flat, float* w, int T, int heads) {
  size_t rowId = blockIdx.x; int h = (int)(rowId / T), i = (int)(rowId % T);
  float* out = w + rowId * T;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < T; j += blockDim.x) { float v = flat[((size_t)i * T + j) * heads + h]; out[j] = v; mx = fmaxf(mx, v); }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); mx = red[0]; __syncthreads();
  float s = 0;
  for (int j = threadIdx.x; j < T; j += blockDim.x) { float e = expf(out[j] - mx); out[j] = e; s += e; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < T; j += blockDim.x) out[j] *= inv;
}
// v [s][j][h d + e] -> [h][j][s][e]
template <class TI, class TO>
__global__ void pwaToHeadsK(const TI* v, TO* out, int S, int T, int heads, int d) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * T * heads * d) return;
  int e = (int)(t % d); size_t r = t / d; int s = (int)(r % S); r /= S; int j = (int)(r % T); int h = (int)(r / T);
  out[t] = v[((size_t)s * T + j) * heads * d + h * d + e];
}
// o [h][i][s][e] -> [s][i][h d + e], times sigmoid(gate)
__device__ __forceinline__ float pwaF(float x) { return x; }
__device__ __forceinline__ float pwaF(half x) { return __half2float(x); }
__device__ __forceinline__ void pwaPut(float* p, float v) { *p = v; }
__device__ __forceinline__ void pwaPut(half* p, float v) { *p = __float2half(v); }
template <class TV>
__global__ void pwaFromHeadsK(const TV* o, const TV* gate, TV* out, int S, int T, int heads, int d) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * T * heads * d) return;
  int c = (int)(t % (heads * d)); size_t si = t / (heads * d); int i = (int)(si % T), s = (int)(si / T);
  int h = c / d, e = c % d;
  pwaPut(out + t, pwaF(o[(((size_t)h * T + i) * S + s) * d + e]) / (1.f + expf(-pwaF(gate[t]))));
}
inline void msaPairWeightedAveraging(float* m, const float* pair, int S, int T, int Cm, int C, const std::string& Bn) {
  const int heads = (int)M.meta("meta/msaHeads"), d = (int)M.meta("meta/msaHeadWidth"), Wd = heads * d;
  size_t rows = (size_t)S * T, P = (size_t)T * T;
  float* ln = scratch<float>("msa.ln", rows * Cm);
  layerNorm(m, ln, rows, Cm, F(Bn + "normScale"), F(Bn + "normOffset"));
  float* flat = scratch<float>("msa.flat", P * heads);
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));
  float* pln = scratch<float>("msa.pln", chunk * C);
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    layerNorm(pair + r0 * C, pln, r, C, F(Bn + "pairNormScale"), F(Bn + "pairNormOffset"));
    gemm(pln, F(Bn + "pairLogits"), flat + r0 * heads, r, C, heads);
  }
  float* w = scratch<float>("msa.w", (size_t)heads * P);
  pwaSoftmaxK<<<(unsigned)(heads * T), 128, 0, STREAM>>>(flat, w, T, heads);
  // each row's own values, weighted sums, gate and output: a block of rows at a time
  int Sc = (int)std::max<size_t>(1, std::min<size_t>(S, ((size_t)128 << 20) / ((size_t)T * Wd * 4)));
  float* v = scratch<float>("msa.v", (size_t)Sc * T * Wd); float* vh = scratch<float>("msa.vh", (size_t)Sc * T * Wd);
  float* oh = scratch<float>("msa.oh", (size_t)Sc * T * Wd); float* g = scratch<float>("msa.g", (size_t)Sc * T * Wd);
  const float one = 1.f, zero = 0.f;
  for (int s0 = 0; s0 < S; s0 += Sc) {
    int sc = std::min(Sc, S - s0); size_t cr = (size_t)sc * T;
    const float* lnc = ln + (size_t)s0 * T * Cm;
    gemm(lnc, F(Bn + "value"), v, cr, Cm, Wd);
    pwaToHeadsK<<<blocks(cr * Wd), 256, 0, STREAM>>>(v, vh, sc, T, heads, d);
    // per head: O_h (T x sc d) = W_h (T x T) V_h (T x sc d); col-major O^T = V^T W^T
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, sc * d, T, T, &one, vh, sc * d, (long long)T * sc * d, w, T,
                                 (long long)P, &zero, oh, sc * d, (long long)T * sc * d, heads));
    gemm(lnc, F(Bn + "gate"), g, cr, Cm, Wd);
    pwaFromHeadsK<<<blocks(cr * Wd), 256, 0, STREAM>>>(oh, g, v, sc, T, heads, d);
    gemm(v, F(Bn + "output"), m + (size_t)s0 * T * Cm, cr, Wd, Cm, 1.f);
  }
}

// The same in f16 (MSA Pairformer's --fast; ESMFold2's encoder keeps the f32 form above): the LN'd rows, the values,
// gates and weighted sums f16 into tensor-core GEMMs with f32 accumulation (cuBLASLt), the softmax f32 and its weights
// rounded to f16 for the per-head GEMM, the output projection accumulated straight into the f32 rows
// [heads][T][T] f32 -> [heads][T][Tp] f16, the rows padded to a multiple of 8 with zeros: at an odd T (1,023) the
// per-head GEMM's leading dimension took cuBLAS onto an unaligned sm_75 kernel, 571 of 1,175 ms (2,048 x 1,023)
__global__ void pwaWeightsHK(const float* w, half* wh, int T, int Tp, size_t rowsN) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rowsN * Tp) return;
  size_t r = t / Tp; int j = (int)(t % Tp);
  wh[t] = j < T ? __float2half(w[r * T + j]) : __float2half(0.f);
}
// pair16: the pair's buffer holds bf16 (MSA Pairformer's PAIR_BF16; 256 channels)
// The per-head weighted sum and its gate in one kernel (msaPairWeightedAveragingFast): for each head h,
//   v[s][i][h d + e] = sigmoid(gate[s][i][h d + e]) sum_j w_h[i][j] vals[s][j][h d + e]
// as ONE GEMM a head - M the (sequence, channel) pairs, N the residues i, K the residues j - reading the values where the
// value-gate GEMM wrote them (rows ldv apart: [s][j][values | gates]) and writing the MSA's own layout, the gate applied
// on the way out. cuBLAS's form of it was a 32-column GEMM a (sequence, head) - 35 TFLOPS - and a gate pass after it.
// A block: 4 sequences x 32 channels (128 rows of M) by 64 residues, 4 warps of 64 x 32, K in steps of 32 through 4
// cp.async stages; the values [j][(s, e)] reach the tensor cores through ldmatrix.trans, w_h [i][j] directly.
// wh: [heads][T][Tp] f16 (Tp a multiple of 8, zero past T); d 32
constexpr int PH_SQ = 4, PH_BM = PH_SQ * 32, PH_BN = 64, PH_KT = 32, PH_STAGES = 4, PH_NTH = 128;
constexpr size_t PH_STAGE = (size_t)(PH_KT * PH_BM + PH_BN * PH_KT) * 2;
constexpr int PH_LDO = PH_BM + 8;
constexpr size_t PH_SMEM = std::max(PH_STAGES * PH_STAGE, (size_t)PH_BN * PH_LDO * 2);
__device__ __forceinline__ int phA(int r, int c) { return r * PH_BM + ((((c >> 3) ^ r) & 7) | ((c >> 3) & 8)) * 8 + (c & 7); }   // [32][128]
__device__ __forceinline__ int phB(int r, int c) { return r * PH_KT + ((((c >> 3) ^ (r >> 1)) & 3) << 3) + (c & 7); }          // [64][32]
__global__ void __launch_bounds__(PH_NTH) pwaHeadK(const half* __restrict__ vg, int ldv, const half* __restrict__ wh,
                                                   half* __restrict__ out, int S, int T, int Tp, int Wd) {
  constexpr int d = 32;
  extern __shared__ __align__(16) unsigned char smem[];
  auto As = [&](int st) { return (half*)(smem + st * PH_STAGE); };
  auto Bs = [&](int st) { return As(st) + PH_KT * PH_BM; };
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wm = warp & 1, wn = warp >> 1;
  const int i0 = blockIdx.x * PH_BN, s0 = blockIdx.y * PH_SQ, h = blockIdx.z;
  const int tiles = (T + PH_KT - 1) / PH_KT;
  const half* W = wh + (size_t)h * T * Tp;
  // the copies: A 512 chunks (j 32 x 4 sequences x 4), B 256 (i 64 x 4), every offset once
  constexpr int AIT = PH_KT * PH_BM / 8 / PH_NTH, BIT = PH_BN * PH_KT / 8 / PH_NTH;
  int aDst[AIT], aJ[AIT], bDst[BIT], bK[BIT]; const half* aSrc[AIT]; const half* bSrc[BIT]; bool aOk[AIT], bOk[BIT];
#pragma unroll
  for (int it = 0; it < AIT; ++it) {
    const int q = it * PH_NTH + threadIdx.x, jr = q / 16, rem = q % 16, sl = rem / 4, ec = (rem % 4) * 8;
    aDst[it] = phA(jr, sl * 32 + ec); aJ[it] = jr; aOk[it] = s0 + sl < S;
    aSrc[it] = vg + ((size_t)(aOk[it] ? s0 + sl : 0) * T + jr) * ldv + h * d + ec;
  }
#pragma unroll
  for (int it = 0; it < BIT; ++it) {
    const int q = it * PH_NTH + threadIdx.x, ir = q / 4, kc = (q % 4) * 8;
    bDst[it] = phB(ir, kc); bK[it] = kc; bOk[it] = i0 + ir < T;
    bSrc[it] = W + (size_t)(bOk[it] ? i0 + ir : 0) * Tp + kc;
  }
  auto issue = [&](int t, int st) {
    const int j0 = t * PH_KT;
    half *a = As(st), *b = Bs(st);
#pragma unroll
    for (int it = 0; it < AIT; ++it) {
      const bool ok = aOk[it] && j0 + aJ[it] < T;
      cpAsync16(a + aDst[it], ok ? aSrc[it] + (size_t)j0 * ldv : vg, ok);
    }
#pragma unroll
    for (int it = 0; it < BIT; ++it) {
      const bool ok = bOk[it] && j0 + bK[it] < Tp;
      cpAsync16(b + bDst[it], ok ? bSrc[it] + j0 : wh, ok);
    }
    cpCommit();
  };
  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
#pragma unroll
  for (int t = 0; t < PH_STAGES - 1; ++t) { if (t < tiles) issue(t, t); else cpCommit(); }
  int aFr[2][4], bFr[2][2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    const int q = lane >> 3;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) aFr[ks][mt] = phA(ks * 16 + (q >> 1) * 8 + (lane & 7), wm * 64 + mt * 16 + (q & 1) * 8);
#pragma unroll
    for (int n2 = 0; n2 < 2; ++n2) bFr[ks][n2] = PH_KT * PH_BM + phB(wn * 32 + n2 * 16 + (q >> 1) * 8 + (lane & 7), ks * 16 + (q & 1) * 8);
  }
  uint32_t fa[2][4][4], fb[2][2][4];
  auto load = [&](const half* stage, int ks, int buf) {
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) ldsm4t(fa[buf][mt], stage + aFr[ks][mt]);
#pragma unroll
    for (int n2 = 0; n2 < 2; ++n2) ldsm4(fb[buf][n2], stage + bFr[ks][n2]);
  };
  int st = 0, nst = PH_STAGES - 1;
  for (int t = 0; t < tiles; ++t) {
    cpWait<PH_STAGES - 2>();
    __syncthreads();
    if (t + PH_STAGES - 1 < tiles) issue(t + PH_STAGES - 1, nst); else cpCommit();
    nst = nst + 1 == PH_STAGES ? 0 : nst + 1;
    const half* stage = As(st);
    st = st + 1 == PH_STAGES ? 0 : st + 1;
    load(stage, 0, 0);
#pragma unroll
    for (int ks = 0; ks < PH_KT / 16; ++ks) {
      const int cur = ks & 1;
      if (ks + 1 < PH_KT / 16) load(stage, ks + 1, cur ^ 1);
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int n2 = 0; n2 < 2; ++n2) {
          mma16816(acc[mt][2 * n2], fa[cur][mt], fb[cur][n2][0], fb[cur][n2][1]);
          mma16816(acc[mt][2 * n2 + 1], fa[cur][mt], fb[cur][n2][2], fb[cur][n2][3]);
        }
    }
  }
  cpWait<0>();
  __syncthreads();
  // the tile [i][(s, e)] through shared memory, then each (s, i)'s 32 channels gated and stored as 64 contiguous bytes
  half* o = (half*)smem;
  const int g = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int m = wm * 64 + mt * 16 + g, n = wn * 32 + nt * 8 + tig * 2;
      o[n * PH_LDO + m] = __float2half(acc[mt][nt][0]); o[(n + 1) * PH_LDO + m] = __float2half(acc[mt][nt][1]);
      o[n * PH_LDO + m + 8] = __float2half(acc[mt][nt][2]); o[(n + 1) * PH_LDO + m + 8] = __float2half(acc[mt][nt][3]);
    }
  __syncthreads();
#pragma unroll
  for (int it = 0; it < PH_BN * PH_SQ * 4 / PH_NTH; ++it) {
    const int q = it * PH_NTH + threadIdx.x, il = q / 16, rem = q % 16, sl = rem / 4, ec = (rem % 4) * 8;
    const int s = s0 + sl, i = i0 + il;
    if (s >= S || i >= T) continue;
    uint4 ov = *reinterpret_cast<const uint4*>(o + il * PH_LDO + sl * 32 + ec);
    const size_t row = (size_t)s * T + i;
    uint4 gv = *reinterpret_cast<const uint4*>(vg + row * ldv + Wd + h * d + ec);
    const half *op = reinterpret_cast<const half*>(&ov), *gp = reinterpret_cast<const half*>(&gv);
    uint4 rv; half* rp = reinterpret_cast<half*>(&rv);
#pragma unroll
    for (int e = 0; e < 8; ++e) rp[e] = __float2half(__half2float(op[e]) * sigmH(__half2float(gp[e])));
    *reinterpret_cast<uint4*>(out + row * Wd + h * d + ec) = rv;
  }
}
inline void msaPairWeightedAveragingFast(float* m, const float* pair, int S, int T, int Cm, int C, const std::string& Bn,
                                         bool pair16 = false) {
  if (pair16 && C != 256) { fprintf(stderr, "a bf16 pair's pair-weighted averaging is 256 channels\n"); exit(1); }
  const int heads = (int)M.meta("meta/msaHeads"), d = (int)M.meta("meta/msaHeadWidth"), Wd = heads * d;
  size_t rows = (size_t)S * T, P = (size_t)T * T;
  // (msa.lnh: one f16 copy of the normed MSA rows - MSA Pairformer's outer product normalises into the same buffer)
  half* ln = scratch<half>("msa.lnh", rows * Cm);
  layerNormH(m, ln, rows, Cm, F(Bn + "normScale"), F(Bn + "normOffset"));
  float* flat = scratch<float>("msa.flat", P * heads);
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (2 * (size_t)C));
  half* pln = scratch<half>("msah.pln", chunk * C);
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    if (pair16)
      layerNormVK<256, __nv_bfloat16, half><<<(unsigned)((r + 7) / 8), 256, 0, STREAM>>>(
          reinterpret_cast<const __nv_bfloat16*>(pair) + r0 * C, pln, r, F(Bn + "pairNormScale"), F(Bn + "pairNormOffset"));
    else layerNormH(pair + r0 * C, pln, r, C, F(Bn + "pairNormScale"), F(Bn + "pairNormOffset"));
    ltGemm(pln, Fh(Bn + "pairLogits"), flat + r0 * heads, false, r, C, heads, nullptr, false, 0.f);
  }
  float* w = scratch<float>("msa.w", (size_t)heads * P);
  pwaSoftmaxK<<<(unsigned)(heads * T), 128, 0, STREAM>>>(flat, w, T, heads);
  const int Tp = (T + 7) / 8 * 8;
  half* wh = scratch<half>("msah.w", (size_t)heads * T * Tp);
  pwaWeightsHK<<<blocks((size_t)heads * T * Tp), 256, 0, STREAM>>>(w, wh, T, Tp, (size_t)heads * T);
  int Sc = (int)std::max<size_t>(1, std::min<size_t>(S, ((size_t)128 << 20) / ((size_t)T * Wd * 2)));
  // the values and gates in one GEMM (valueGate: [Cm][2 Wd]), rows 2 Wd wide
  if (d != 32) { fprintf(stderr, "the fused per-head kernel is 32 channels a head (%d)\n", d); exit(1); }
  half* vg = scratch<half>("msah.vg", (size_t)Sc * T * 2 * Wd);
  half* v = scratch<half>("msah.v", (size_t)Sc * T * Wd);
  static bool attr = false;
  if (!attr) { smemAttr(pwaHeadK, (int)PH_SMEM); attr = true; }
  for (int s0 = 0; s0 < S; s0 += Sc) {
    int sc = std::min(Sc, S - s0); size_t cr = (size_t)sc * T;
    const half* lnc = ln + (size_t)s0 * T * Cm;
    ltGemm(lnc, Fh(Bn + "valueGate"), vg, true, cr, Cm, 2 * Wd, nullptr, false, 0.f);
    pwaHeadK<<<dim3((T + PH_BN - 1) / PH_BN, (sc + PH_SQ - 1) / PH_SQ, heads), PH_NTH, PH_SMEM, STREAM>>>(vg, 2 * Wd, wh, v, sc, T, Tp, Wd);
    ltGemm(v, Fh(Bn + "output"), m + (size_t)s0 * T * Cm, false, cr, Wd, Cm, nullptr, false, 1.f);
  }
}

// z_init (in, out): the encoder's pair. The alignment's rows past 1024 are subsampled keeping the query and
// the a3m's order, and 10% of the non-query rows' columns masked - both drawn from the seed, as biohub's
// processor does at inference (msa_max_depth 1024, msa_column_mask_rate 0.1); EF2_DETERMINISTIC=1 takes the
// first rows and masks nothing (af3-any-model's reading, for a tensor comparison)
inline void msaEncode(float* zi, const float* sInputs, int T, int C, int Si, uint64_t seed) {
  static const bool deterministic = getenv("EF2_DETERMINISTIC") != nullptr;
  const int depth = (int)M.meta("meta/msa_depth"), Cm = (int)M.meta("meta/msaChannels");
  const int maxDepth = (int)M.meta("meta/msaMaxDepth");
  const float rate = deterministic ? 0.f : (float)M.meta("meta/msaColumnMaskRate");
  std::mt19937_64 rng(seed ^ 0x4d5341ull);
  std::vector<int> keep(depth);
  for (int s = 0; s < depth; ++s) keep[s] = s;
  if (depth > maxDepth) {
    if (!deterministic) std::shuffle(keep.begin() + 1, keep.end(), rng);
    keep.resize(maxDepth);
    std::sort(keep.begin(), keep.end());
  }
  const int S = (int)keep.size();
  const int* rowsH = M.i("msa/rows"); const float* delH = M.f("msa/deletion");
  std::vector<int> rows((size_t)S * T); std::vector<float> del((size_t)S * T), mask((size_t)S * T, 1.f);
  for (int s = 0; s < S; ++s)
    for (int t = 0; t < T; ++t) { rows[(size_t)s * T + t] = rowsH[(size_t)keep[s] * T + t]; del[(size_t)s * T + t] = delH[(size_t)keep[s] * T + t]; }
  if (S > 1 && rate > 0.f) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    for (int t = 0; t < T; ++t)
      if (u(rng) < rate) for (int s = 1; s < S; ++s) mask[(size_t)s * T + t] = 0.f;
  }
  size_t R = (size_t)S * T, P = (size_t)T * T;
  int* rowsD = scratch<int>("msa.rows", R); float* delD = scratch<float>("msa.del", R); float* maskD = scratch<float>("msa.mask", R);
  CK(cudaMemcpyAsync(rowsD, rows.data(), R * 4, cudaMemcpyHostToDevice, STREAM));
  CK(cudaMemcpyAsync(delD, del.data(), R * 4, cudaMemcpyHostToDevice, STREAM));
  CK(cudaMemcpyAsync(maskD, mask.data(), R * 4, cudaMemcpyHostToDevice, STREAM));
  float* fromInputs = scratch<float>("msa.fromInputs", (size_t)T * Cm);
  gemm(sInputs, F("msaEncoder/projectInputs"), fromInputs, T, Si, Cm);
  float* m = scratch<float>("msa.m", R * Cm);
  msaEmbedEf2K<<<blocks(R * Cm), 256, 0, STREAM>>>(rowsD, delD, maskD, F("msaEncoder/embed"), fromInputs, m, R, T, Cm);
  float* pmask = scratch<float>("msa.pmask", P);
  fillK<<<blocks(P), 256, 0, STREAM>>>(pmask, 1.f, P);
  const int blocksN = (int)M.meta("meta/msaBlocks");
  for (int b = 0; b < blocksN; ++b) {
    std::string Bn = "msaEncoder/blocks/" + std::to_string(b) + "/";
    msaOuterProductMean(zi, m, maskD, S, T, Cm, C, Bn + "outerProductMean/");
    if (M.has("f/" + Bn + "pairWeightedAveraging/value")) {
      msaPairWeightedAveraging(m, zi, S, T, Cm, C, Bn + "pairWeightedAveraging/");
      rowsTransition(m, R, Cm, Bn + "msaTransition/");
    }
    trunkBlock(zi, pmask, T, C, "msaEncoder/blocks", b);
  }
  releaseScratch({ "msa." });
}
