// The MSA16 stack's attention output projection, with the NEXT module's LayerNorm emitted from it:
//   msa[r] = m16(o[r] Wout + b + msa[r]);  xn[r] = LN(msa[r]) (the rounded row, as the next module reads it)
// where cuBLASLt's GEMM wrote the MSA and the next module (the column attention after the row attention, the MSA
// transition after the column attention) ran a LayerNorm pass of its own over all of it. C = 256 = the projection's
// width and depth (AF2's MSA: 8 heads of 32). M the MSA's 16-bit type: bf16 on Ampere, f16 on a T4 (MSA16_HALF). One
// block 64 rows x all 256 columns (eight warps: four 16-row groups x two 128-column halves), the K loop in
// double-buffered 32-deep stages. Then the rows' residuals are read into the stages' memory, each accumulator adds its
// bias and residual and is rounded in place - the 16-bit sum, the value the MSA keeps - and a row pass writes whole
// rows back and LayerNorms them with layerNormVK's arithmetic. (M is a parameter for the measurement below.)
// (The sums staged as f32 accumulators, the residual added in the row pass, were the same arithmetic in 66.5 KB - past
// a T4's 64 KB a block, and two blocks an A100 SM where 44 KB is three. 32-row blocks fitted a T4 at 38 KB and lost
// there: each block reads all of Wout, and the kernel was 3.2 ms a call against cuBLAS's GEMM at ~1.3 plus the LN.)
// A T4 (f16, MSA16_HALF) does not take it: at 64 rows (44 KB) and its next stage loaded through registers (no cp.async
// there) the kernel was 0.9% behind cuBLAS's GEMM plus the LN pass over 5CAJ's fold, and at 32 rows 3% behind. Ampere:
// 101.8 -> 93.4 ms over the fold's 384 calls against the f32-staged form (three blocks an SM), byte-identical.
#pragma once
namespace oln {
constexpr int C = 256, R = 64, KC = 32, LDA = KC, LDW = C, LDV = C + 8, NTH = 256;
constexpr size_t STG = (size_t)R * LDA * 2 + (size_t)KC * LDW * 2;
constexpr size_t SMEM = 2 * STG > (size_t)R * LDV * 2 ? 2 * STG : (size_t)R * LDV * 2;
__device__ __forceinline__ int swA(int r, int c) { return r * LDA + (c ^ (((r >> 1) & 3) << 3)); }
__device__ __forceinline__ int swW(int k, int c) { return k * LDW + (c ^ ((k & 7) << 3)); }
}
__device__ __forceinline__ void mma16816bf(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
template <class M, class TO, class TI = M>
__global__ void __launch_bounds__(oln::NTH, 2) outLnK(const TI* __restrict__ O, const TI* __restrict__ W,
    const float* __restrict__ bias, M* __restrict__ msa, const float* __restrict__ lnS,
    const float* __restrict__ lnO, TO* __restrict__ xn, size_t rows) {
  using namespace oln;
  constexpr bool BF = std::is_same_v<M, __nv_bfloat16>;
  constexpr bool BFI = std::is_same_v<TI, __nv_bfloat16>;
  using M2 = std::conditional_t<BF, __nv_bfloat162, half2>;
  extern __shared__ __align__(16) unsigned char smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, rg = warp & 3, chh = warp >> 2;
  const size_t row0 = (size_t)blockIdx.x * R;
  auto As = [&](int s) { return (TI*)(smem + s * STG); };
  auto Ws = [&](int s) { return (TI*)(smem + s * STG) + R * LDA; };
  M* V = (M*)smem;
  constexpr int WPT = KC * C / 8 / NTH;
  auto issue = [&](int kc, int s) {
    {   // A: 64 rows x 32 k, one 16-byte piece a thread
      const int r = threadIdx.x >> 2, c = (threadIdx.x & 3) * 8;
      const size_t row = row0 + r;
      cpAsync16(As(s) + swA(r, c), O + (row < rows ? row : 0) * C + kc * KC + c, row < rows);
    }
#pragma unroll
    for (int i = 0; i < WPT; ++i) {   // W: 32 k x 256 columns, four a thread
      const int u = i * NTH + threadIdx.x, k = u / (C / 8), c = (u % (C / 8)) * 8;
      cpAsync16(Ws(s) + swW(k, c), W + (size_t)(kc * KC + k) * C + c, true);
    }
    cpCommit();
  };
  issue(0, 0);
  float acc[16][4] = {};
  constexpr int KSTEPS = C / KC;
#pragma unroll 1
  for (int kc = 0; kc < KSTEPS; ++kc) {
    const int st = kc & 1;
    if (kc + 1 < KSTEPS) { issue(kc + 1, st ^ 1); cpWait<1>(); }
    else {
      cpWait<0>();
      // Stage 0 is now idle while kc == KSTEPS - 1 reads Stage 1: prefetch the first 32 rows of msa into V[0..31]
      static_assert((R / 2) * LDV * 2 <= STG, "first half of V fits in idle stage 0");
#pragma unroll
      for (int i = 0; i < (R / 2) * C / 8 / NTH; ++i) {
        const int u = i * NTH + threadIdx.x, r = u / (C / 8), c = (u % (C / 8)) * 8;
        const size_t row = row0 + r;
        cpAsync16(V + r * LDV + c, msa + (row < rows ? row : 0) * C + c, row < rows);
      }
      cpCommit();
    }
    __syncthreads();
    const TI* A = As(st); const TI* Wt = Ws(st);
    uint32_t a[2][4];
    ldsm4(a[0], A + swA(rg * 16 + (lane & 15), (lane >> 4) * 8));
    ldsm4(a[1], A + swA(rg * 16 + (lane & 15), 16 + (lane >> 4) * 8));
#pragma unroll
    for (int ks = 0; ks < KC / 16; ++ks) {
      uint32_t f[2][4];
      const int kl = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), cl = chh * 128 + (lane >> 4) * 8;
      ldsm4t(f[0], Wt + swW(kl, cl));
#pragma unroll
      for (int n2 = 0; n2 < 8; ++n2) {
        const int cur = n2 & 1;
        if (n2 + 1 < 8) ldsm4t(f[cur ^ 1], Wt + swW(kl, cl + (n2 + 1) * 16));
        if constexpr (BFI) { mma16816bf(acc[2 * n2], a[ks], f[cur][0], f[cur][1]); mma16816bf(acc[2 * n2 + 1], a[ks], f[cur][2], f[cur][3]); }
        else { mma16816(acc[2 * n2], a[ks], f[cur][0], f[cur][1]); mma16816(acc[2 * n2 + 1], a[ks], f[cur][2], f[cur][3]); }
      }
    }
    __syncthreads();
  }
  // the second half of the residual rows into V[32..63] (every warp is past the last barrier), coalesced
#pragma unroll
  for (int i = (R / 2) * C / 8 / NTH; i < R * C / 8 / NTH; ++i) {
    const int u = i * NTH + threadIdx.x, r = u / (C / 8), c = (u % (C / 8)) * 8;
    const size_t row = row0 + r;
    cpAsync16(V + r * LDV + c, msa + (row < rows ? row : 0) * C + c, row < rows);
  }
  cpCommit(); cpWait<0>();
  __syncthreads();
  // each accumulator + its bias + its residual, rounded, in place
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    const int col = chh * 128 + nt * 8 + tig * 2;
    const float2 bf = *reinterpret_cast<const float2*>(bias + col);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      M2* p = reinterpret_cast<M2*>(V + (rg * 16 + g + 8 * h) * LDV + col);
      const float2 rf = ld2f(reinterpret_cast<const M*>(p));
      const float a = acc[nt][2 * h] + bf.x + rf.x, b = acc[nt][2 * h + 1] + bf.y + rf.y;
      if constexpr (BF) *p = __floats2bfloat162_rn(a, b); else *p = __floats2half2_rn(a, b);
    }
  }
  __syncthreads();
  // a warp a row at a time: lane's 8 columns lane*8 .. +7
  constexpr int RPW = R / (NTH / 32);
  const int c0 = lane * 8;
  float sc[8], of[8];
  *reinterpret_cast<float4*>(sc) = *reinterpret_cast<const float4*>(lnS + c0);
  *reinterpret_cast<float4*>(sc + 4) = *reinterpret_cast<const float4*>(lnS + c0 + 4);
  *reinterpret_cast<float4*>(of) = *reinterpret_cast<const float4*>(lnO + c0);
  *reinterpret_cast<float4*>(of + 4) = *reinterpret_cast<const float4*>(lnO + c0 + 4);
#pragma unroll
  for (int i = 0; i < RPW; ++i) {
    const int r = warp + i * (NTH / 32);
    const size_t row = row0 + r;
    if (row >= rows) break;
    const uint4 ob = *reinterpret_cast<const uint4*>(V + r * LDV + c0);
    *reinterpret_cast<uint4*>(msa + row * C + c0) = ob;
    float v[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float2 f = ld2f(reinterpret_cast<const M*>(&ob) + 2 * e); v[2 * e] = f.x; v[2 * e + 1] = f.y; }
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) s += v[e];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    const float mean = s / C;
    float q = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { const float d = v[e] - mean; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
    const float inv = rsqrtf(q / C + 1e-5f);
    uint4 xo;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float a = (v[2 * e] - mean) * inv * sc[2 * e] + of[2 * e];
      const float b = (v[2 * e + 1] - mean) * inv * sc[2 * e + 1] + of[2 * e + 1];
      if constexpr (std::is_same_v<TO, half>) reinterpret_cast<half2*>(&xo)[e] = __floats2half2_rn(a, b);
      else reinterpret_cast<__nv_bfloat162*>(&xo)[e] = __floats2bfloat162_rn(a, b);
    }
    *reinterpret_cast<uint4*>(xn + row * C + c0) = xo;
  }
}
template <class M, class TO, class TI = M>
inline void outLn256(const TI* O, const TI* W, const float* bias, M* msa,
                     const float* lnS, const float* lnO, TO* xn, size_t rows) {
  static bool attr = false;
  if (!attr) { smemAttr((outLnK<M, TO, TI>), (int)oln::SMEM); attr = true; }
  outLnK<M, TO, TI><<<(unsigned)((rows + oln::R - 1) / oln::R), oln::NTH, oln::SMEM, STREAM>>>(O, W, bias, msa, lnS, lnO, xn, rows);
}
// the LN'd rows a projection emitted for the next module: that module's LN prefix, its block, the MSA it normalised
struct EmittedLn { std::string ln; int blk = -1; const void* msa = nullptr; void* xn = nullptr; size_t rows = 0; };
inline EmittedLn EMITTED_LN;
// the next module's: its rows if they were emitted for exactly this LN of exactly this MSA, else null (and forgotten)
inline void* takeEmittedLn(const std::string& ln, int blk, const void* msa, size_t rows) {
  EmittedLn e = EMITTED_LN; EMITTED_LN = EmittedLn{};
  return e.ln == ln && e.blk == blk && e.msa == msa && e.rows == rows ? e.xn : nullptr;
}
