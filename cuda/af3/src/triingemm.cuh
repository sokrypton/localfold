// The triangle multiplication's input side as a tiled GEMM (Ampere on): LN(pair) -> [projection | gate] and the
// gating linear, the gated, masked a and b written channel-major and the gating linear's raw output t2 - what
// triIn256K computes, with the opposite trade. triIn256K keeps a warp's 16 LN'd rows in registers as A fragments
// and streams the weights 16 columns a step, so each weight fragment it loads from shared memory feeds one warp's
// 16 rows (Nsight Compute at 384 channels: shared memory 70% busy, the tensor pipe ~50%, 172 registers, two blocks
// an SM). Here the LN'd rows are written once as f16 (lnPlaneK, the padded plane's rows) and a block of 4 warps
// computes a 128 x 128 tile from them, each warp 64 x 64 - 32 MMA chains and 4 MMAs for every ldmatrix - with A and
// B both streamed through a three-stage cp.async pipeline. The grid runs a row tile's column tiles side by side, so
// its A rows are read from L2 after the first.
//
// The weights are packed once (triInGemmWeights) into [NT][C][128] tiles: for the first C/32 tiles,
// [proj 32 | gate 32 | proj 32 | gate 32] - each warp's 64 columns are 32 projection columns and their gates, so
// the gating happens in registers - then C/128 tiles of the gating linear. A projection column 2ch is a's channel
// ch, 2ch + 1 b's (the [projection | gate] matrix's interleave, as triIn256K reads it).
#pragma once
#include "fused256.cuh"

// LN of the padded plane's rows into f16 (a padding row is LN(0) = the offset, as triIn256K computes it)
template <int C, class PT>
__global__ void lnPlaneK(const float* __restrict__ pair, half* __restrict__ out, int n, int np, const float* __restrict__ sc,
                         const float* __restrict__ of) {
  constexpr int K = C / 32;
  size_t q = (size_t)blockIdx.x * 8 + (threadIdx.x >> 5), pp = (size_t)np * np;
  int lane = threadIdx.x & 31;
  if (q >= pp) return;
  unsigned u = (unsigned)q, i = u / (unsigned)np, j = u - i * (unsigned)np;
  bool real = i < (unsigned)n && j < (unsigned)n;
  size_t pr = (size_t)i * n + j;
  float v[K], s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { v[k] = real ? pairLd<PT>(pair, pr * C + lane + 32 * k) : 0.f; s += v[k]; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, qd = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { float d = v[k] - mean; qd += d * d; }
  for (int o = 16; o; o >>= 1) qd += __shfl_xor_sync(~0u, qd, o);
  float inv = rsqrtf(qd / C + 1e-5f);
#pragma unroll
  for (int k = 0; k < K; ++k) out[q * C + lane + 32 * k] = __float2half((v[k] - mean) * inv * sc[lane + 32 * k] + of[lane + 32 * k]);
}

// shared layouts: the A stage [128 rows][32 k] halves, 64-byte rows, chunk ^ ((row >> 1) & 3); the B stage
// [32 k][128 n] halves, chunk ^ (k & 7) - both conflict-free for ldmatrix and for the cp.async writes
__device__ __forceinline__ int tgA(int r, int c) { return r * 32 + ((((c >> 3) ^ ((r >> 1) & 3))) << 3) + (c & 7); }
__device__ __forceinline__ int tgB(int k, int nn) { return k * 128 + (((nn >> 3) ^ (k & 7)) << 3) + (nn & 7); }
#ifndef TG_STAGES_N
#define TG_STAGES_N 3
#endif
constexpr int TG_STAGES = TG_STAGES_N;
constexpr size_t TG_STAGE = (size_t)(128 * 32 + 32 * 128) * 2;
constexpr size_t triInGemmSmem() { return TG_STAGES * TG_STAGE; }      // (the epilogue's staging reuses it)
// The GEMM's main loop: acc (a warp's 64 x 64) = X[row0 .. row0 + 128][0 .. C] Wt[0 .. C][0 .. 128], X row-major with
// rows past pp read as zero, Wt one packed [C][128] tile. Leaves every cp.async group drained.
template <int C>
__device__ __forceinline__ void tgMain(const half* __restrict__ X, const half* __restrict__ Wt, size_t row0, size_t pp,
                                       unsigned char* smem, float (&acc)[4][8][4]) {
  constexpr int KC = C / 32;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int wm = warp & 1, wn = warp >> 1;
  auto As = [&](int s) { return (half*)(smem + s * TG_STAGE); };
  auto Bs = [&](int s) { return As(s) + 128 * 32; };
  // every address below from per-thread bases: the swizzles (tgA, tgB) reduce to an XOR with a constant across a
  // chunk's k16 steps and column groups, so with the k loop unrolled each ldmatrix and copy is a base plus an
  // immediate (written out because the generic form cost a third of the issue slots in integer arithmetic)
  const int tid = threadIdx.x, l4 = lane >> 4, ar = wm * 64 + (lane & 15), kl = ((lane >> 3) & 1) * 8 + (lane & 7);
  const int aBase = ar * 32 + ((l4 ^ ((ar >> 1) & 3)) << 3);            // + mi * 512, ^ (ks * 16)
  const int bBase = kl * 128 + wn * 64 + ((l4 ^ (lane & 7)) << 3);      // ^ (nj * 16), + ks * 2048
  const int dA = tgA(tid >> 2, (tid & 3) * 8), dB = tgB(tid >> 4, (tid & 15) * 8);   // + i * 1024 for the i-th copy
  const half* sA[4]; bool okA[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    size_t q = row0 + (tid >> 2) + 32 * i;
    okA[i] = q < pp;
    sA[i] = X + (okA[i] ? q : 0) * C + (tid & 3) * 8;
  }
  const half* sB = Wt + (size_t)(tid >> 4) * 128 + (tid & 15) * 8;
  auto issue = [&](int kc) {
    if (kc < KC) {
      half *A = As(kc % TG_STAGES), *B = Bs(kc % TG_STAGES);
#pragma unroll
      for (int i = 0; i < 4; ++i) cpAsync16(A + dA + i * 1024, sA[i] + kc * 32, okA[i]);
#pragma unroll
      for (int i = 0; i < 4; ++i) cpAsync16(B + dB + i * 1024, sB + (size_t)kc * 4096 + i * 1024, true);
    }
    cpCommit();
  };
#pragma unroll
  for (int s0 = 0; s0 < TG_STAGES - 1; ++s0) issue(s0);
  // fragments double-buffered in registers: the next half-step's loads in flight under this one's 32 MMAs (the
  // helpers are asm volatile, so the order written is the order issued); a chunk's barrier and its next copy sit
  // between its two half-steps, once every thread holds the chunk's second half in registers
  uint32_t af[2][4][4], bf[2][4][4];
  auto load = [&](int buf, int kc, int ks) {
    const half *A = As(kc % TG_STAGES), *B = Bs(kc % TG_STAGES);
    const half* a0 = A + (aBase ^ (ks * 16));
    const half* b0 = B + ks * 2048;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) ldsm4(af[buf][mi], a0 + mi * 512);
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) ldsm4t(bf[buf][nj], b0 + (bBase ^ (nj * 16)));
  };
  auto mmas = [&](int buf) {
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int nj = 0; nj < 4; ++nj) {
        mma16816(acc[mi][2 * nj], af[buf][mi], bf[buf][nj][0], bf[buf][nj][1]);
        mma16816(acc[mi][2 * nj + 1], af[buf][mi], bf[buf][nj][2], bf[buf][nj][3]);
      }
  };
  cpWait<TG_STAGES - 2>();
  __syncthreads();
  issue(TG_STAGES - 1);
  load(0, 0, 0);
#pragma unroll
  for (int kc = 0; kc < KC; ++kc) {
    load(1, kc, 1);
    mmas(0);
    if (kc + 1 < KC) {
      cpWait<TG_STAGES - 2>();          // chunk kc + 1 landed (every group but the newest TG_STAGES - 2)
      __syncthreads();                  // and every thread is done reading chunk kc's stage, which the next issue takes
      issue(kc + TG_STAGES);
      load(0, kc + 1, 0);
    }
    mmas(1);
  }
  cpWait<0>();
}
template <int C, class TA>
__global__ void __launch_bounds__(128) triInGemmK(const half* __restrict__ X, const float* __restrict__ pair,
    const float* __restrict__ mask, const half* __restrict__ Wg, TA* __restrict__ a, TA* __restrict__ b,
    half* __restrict__ t2, int n, int np, size_t cs) {
  constexpr int KC = C / 32, NAB = C / 32, LDT = 72;
  extern __shared__ __align__(16) unsigned char smem[];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  int wm = warp & 1, wn = warp >> 1;
  const int t = blockIdx.x;                                      // the column tile (fastest: a row tile's run together)
  const size_t pp = (size_t)np * np, row0 = (size_t)blockIdx.y * 128;
  const half* Wt = Wg + (size_t)t * C * 128;
  // the thread's eight rows' masks, read before the main loop (issued under the pipeline's fill)
  float msk[4][2];
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      size_t q = row0 + wm * 64 + mi * 16 + g + h * 8;
      float v = 0.f;
      if (t < NAB && q < pp) {
        unsigned u = (unsigned)q, i = u / (unsigned)np, j = u - i * (unsigned)np;
        if (i < (unsigned)n && j < (unsigned)n) v = mask[(size_t)i * n + j];
      }
      msk[mi][h] = v;
    }
  float acc[4][8][4] = {};
  tgMain<C>(X, Wt, row0, pp, smem, acc);
  __syncthreads();                      // the stages are free: the epilogue's staging takes them
  if (t < NAB) {
    // a and b: projection n8 tiles 0..3 against their gates 4..7, staged channel-major for 16-byte row stores
    TA* Ta = (TA*)smem + warp * 2 * 16 * LDT;
    TA* Tb = Ta + 16 * LDT;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
      int r0 = mi * 16 + g;
      const float m0 = msk[mi][0], m1 = msk[mi][1];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int chl = j * 4 + tig;
        const float* p = acc[mi][j]; const float* q = acc[mi][j + 4];
        Ta[chl * LDT + r0] = TA(p[0] * sigmH(q[0]) * m0);
        Tb[chl * LDT + r0] = TA(p[1] * sigmH(q[1]) * m0);
        Ta[chl * LDT + r0 + 8] = TA(p[2] * sigmH(q[2]) * m1);
        Tb[chl * LDT + r0 + 8] = TA(p[3] * sigmH(q[3]) * m1);
      }
    }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int idx = lane + 32 * i, chl = idx >> 3, r8 = (idx & 7) * 8;
      size_t q = row0 + wm * 64 + r8;
      if (q < cs) {
        size_t at = (size_t)(t * 32 + wn * 16 + chl) * cs + q;
        *reinterpret_cast<uint4*>(a + at) = *reinterpret_cast<const uint4*>(Ta + chl * LDT + r8);
        *reinterpret_cast<uint4*>(b + at) = *reinterpret_cast<const uint4*>(Tb + chl * LDT + r8);
      }
    }
  } else {
    int c0 = (t - NAB) * 128 + wn * 64 + 2 * tig;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
      size_t q0 = row0 + wm * 64 + mi * 16 + g, q1 = q0 + 8;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (q0 < pp) *reinterpret_cast<half2*>(t2 + q0 * C + c0 + j * 8) = __floats2half2_rn(acc[mi][j][0], acc[mi][j][1]);
        if (q1 < pp) *reinterpret_cast<half2*>(t2 + q1 * C + c0 + j * 8) = __floats2half2_rn(acc[mi][j][2], acc[mi][j][3]);
      }
    }
  }
}

// [NT][C][128] tiles from the [C][4C] projection|gate matrix and the [C][C] gating linear (see above)
__global__ void triInGemmPackK(const half* __restrict__ Wpg, const half* __restrict__ Wgl, int C, half* __restrict__ out) {
  size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int NAB = C / 32, NT = NAB + C / 128;
  if (e >= (size_t)NT * C * 128) return;
  int cl = (int)(e % 128); size_t rest = e / 128; int k = (int)(rest % C), t = (int)(rest / C);
  half v;
  if (t < NAB) {
    int seg = cl >> 5, i = cl & 31;
    int p = t * 64 + (seg >> 1) * 32 + i;                         // the projection column (seg 0, 2) or its gate's (1, 3)
    v = Wpg[(size_t)k * 4 * C + (seg & 1 ? 2 * C : 0) + p];
  } else {
    v = Wgl[(size_t)k * C + (t - NAB) * 128 + cl];
  }
  out[e] = v;
}
inline const half* triInGemmWeights(const std::string& key, const half* Wpg, const half* Wgl, int C) {
  static std::map<std::string, half*> cache;
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  size_t nel = (size_t)(C / 32 + C / 128) * C * 128;
  half* w = dallocT<half>(nel);
  triInGemmPackK<<<blocks(nel), 256, 0, STREAM>>>(Wpg, Wgl, C, w);
  return cache[key] = w;
}
// LOCALFOLD_TRIIN_GEMM=0: triIn256K instead
inline const bool TRIIN_GEMM = !getenv("LOCALFOLD_TRIIN_GEMM") || atoi(getenv("LOCALFOLD_TRIIN_GEMM"));
template <int C, class TA>
inline bool triInGemm(const float* pair, const float* mask, const float* sc, const float* of, const half* Wg, TA* a, TA* b,
                      half* t2, int n, int np, size_t cs) {
  // (from 384 channels: at 256 the LN pass it needs costs more than the GEMM saves - protenix2's trunk 713 -> 719 ms,
  // where OpenDDE's goes 1453 -> 1422 and IntelliFold-2's 2309 -> 2261)
  constexpr size_t smem = triInGemmSmem();
  if (C < 384 || !TRIIN_GEMM || !fitsSmem(smem)) return false;
  size_t pp = (size_t)np * np;
  half* X = scratch<half>("tri.lnplane", pp * C);
  WITH_PAIR_T(lnPlaneK<C, PT><<<(unsigned)((pp + 7) / 8), 256, 0, STREAM>>>(pair, X, n, np, sc, of));
  static bool attr = false;
  if (!attr) { smemAttr((triInGemmK<C, TA>), (int)smem); attr = true; }
  dim3 grid(C / 32 + C / 128, (unsigned)((pp + 127) / 128));
  triInGemmK<C, TA><<<grid, 128, smem, STREAM>>>(X, pair, mask, Wg, a, b, t2, n, np, cs);
  return true;
}

