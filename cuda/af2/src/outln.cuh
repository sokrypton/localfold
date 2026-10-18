// The MSA16 stack's attention output projection, with the NEXT module's LayerNorm emitted from it:
//   msa[r] = bf16(o[r] Wout + b + msa[r]);  xn[r] = LN(msa[r]) (the bf16-rounded row, as the next module reads it)
// where cuBLASLt's GEMM wrote the MSA and the next module (the column attention after the row attention, the MSA
// transition after the column attention) ran a LayerNorm pass of its own over all of it. C = 256 = the projection's
// width and depth (AF2's MSA: 8 heads of 32). One block 64 rows x all 256 columns (eight warps: four 16-row groups x
// two 128-column halves), the K loop in double-buffered 32-deep stages; the f32 accumulators then staged in the
// stages' memory so a row pass reads whole rows: + bias + the residual, rounded, written back, LayerNormed with
// layerNormVK's arithmetic over the rounded values. Ampere on (the bf16 MMA).
#pragma once
namespace oln {
constexpr int C = 256, R = 64, KC = 32, LDA = KC + 8, LDW = C + 8, LDY = C + 4, NTH = 256;
constexpr size_t STG = (size_t)R * LDA * 2 + (size_t)KC * LDW * 2;
constexpr size_t SMEM = 2 * STG > (size_t)R * LDY * 4 ? 2 * STG : (size_t)R * LDY * 4;
}
__device__ __forceinline__ void mma16816bf(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
template <class TO>
__global__ void __launch_bounds__(oln::NTH) outLnK(const __nv_bfloat16* __restrict__ O, const __nv_bfloat16* __restrict__ W,
    const float* __restrict__ bias, __nv_bfloat16* __restrict__ msa, const float* __restrict__ lnS,
    const float* __restrict__ lnO, TO* __restrict__ xn, size_t rows) {
  using namespace oln;
  using B16 = __nv_bfloat16;
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ float sB[C], sS[C], sO[C];
  for (int t = threadIdx.x; t < C; t += NTH) { sB[t] = bias[t]; sS[t] = lnS[t]; sO[t] = lnO[t]; }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, rg = warp & 3, chh = warp >> 2;
  const size_t row0 = (size_t)blockIdx.x * R;
  auto As = [&](int s) { return (B16*)(smem + s * STG); };
  auto Ws = [&](int s) { return (B16*)(smem + s * STG) + R * LDA; };
  auto issue = [&](int kc, int s) {
    {   // A: 64 rows x 32 k, one 16-byte piece a thread
      const int r = threadIdx.x >> 2, c = (threadIdx.x & 3) * 8;
      const size_t row = row0 + r;
      cpAsync16(As(s) + r * LDA + c, O + (row < rows ? row : 0) * C + kc * KC + c, row < rows);
    }
#pragma unroll
    for (int i = 0; i < KC * C / 8 / NTH; ++i) {   // W: 32 k x 256 columns, four a thread
      const int u = i * NTH + threadIdx.x, k = u / (C / 8), c = (u % (C / 8)) * 8;
      cpAsync16(Ws(s) + k * LDW + c, W + (size_t)(kc * KC + k) * C + c, true);
    }
    cpCommit();
  };
  float acc[16][4] = {};
  issue(0, 0);
  constexpr int KSTEPS = C / KC;
#pragma unroll 1
  for (int kc = 0; kc < KSTEPS; ++kc) {
    const int st = kc & 1;
    if (kc + 1 < KSTEPS) { issue(kc + 1, st ^ 1); cpWait<1>(); } else cpWait<0>();
    __syncthreads();
    const B16* A = As(st); const B16* Wt = Ws(st);
#pragma unroll
    for (int ks = 0; ks < KC / 16; ++ks) {
      uint32_t a[4];
      ldsm4(a, A + (rg * 16 + (lane & 15)) * LDA + ks * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int n2 = 0; n2 < 8; ++n2) {
        uint32_t f[4];
        ldsm4t(f, Wt + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW + chh * 128 + n2 * 16 + (lane >> 4) * 8);
        mma16816bf(acc[2 * n2], a, f[0], f[1]); mma16816bf(acc[2 * n2 + 1], a, f[2], f[3]);
      }
    }
    __syncthreads();
  }
  // the accumulators into the stages' memory (every warp is past the last barrier)
  float* Y = (float*)smem;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    const int col = chh * 128 + nt * 8 + tig * 2;
    *reinterpret_cast<float2*>(Y + (rg * 16 + g) * LDY + col) = make_float2(acc[nt][0], acc[nt][1]);
    *reinterpret_cast<float2*>(Y + (rg * 16 + g + 8) * LDY + col) = make_float2(acc[nt][2], acc[nt][3]);
  }
  __syncthreads();
  // a warp a row at a time: lane's 8 columns lane*8 .. +7 - every row's residual loaded first, all in flight at once (one
  // at a time, each row waited on its own load: 44% of the kernel's stall samples)
  constexpr int RPW = R / (NTH / 32);
  const int c0 = lane * 8;
  uint4 resAll[RPW];
#pragma unroll
  for (int i = 0; i < RPW; ++i) {
    const size_t row = row0 + warp + i * (NTH / 32);
    resAll[i] = row < rows ? *reinterpret_cast<const uint4*>(msa + row * C + c0) : make_uint4(0, 0, 0, 0);
  }
#pragma unroll
  for (int i = 0; i < RPW; ++i) {
    const int r = warp + i * (NTH / 32);
    const size_t row = row0 + r;
    if (row >= rows) break;
    const float4 y0 = *reinterpret_cast<const float4*>(Y + r * LDY + c0), y1 = *reinterpret_cast<const float4*>(Y + r * LDY + c0 + 4);
    const uint4 res = resAll[i];
    const __nv_bfloat162* rh = reinterpret_cast<const __nv_bfloat162*>(&res);
    const float yv[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
    float v[8];
    uint4 ob; __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&ob);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 rf = __bfloat1622float2(rh[e]);
      const __nv_bfloat162 o2 = __floats2bfloat162_rn(yv[2 * e] + sB[c0 + 2 * e] + rf.x, yv[2 * e + 1] + sB[c0 + 2 * e + 1] + rf.y);
      oh[e] = o2;
      const float2 back = __bfloat1622float2(o2);
      v[2 * e] = back.x; v[2 * e + 1] = back.y;
    }
    *reinterpret_cast<uint4*>(msa + row * C + c0) = ob;
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
      const float a = (v[2 * e] - mean) * inv * sS[c0 + 2 * e] + sO[c0 + 2 * e];
      const float b = (v[2 * e + 1] - mean) * inv * sS[c0 + 2 * e + 1] + sO[c0 + 2 * e + 1];
      if constexpr (std::is_same_v<TO, half>) reinterpret_cast<half2*>(&xo)[e] = __floats2half2_rn(a, b);
      else reinterpret_cast<__nv_bfloat162*>(&xo)[e] = __floats2bfloat162_rn(a, b);
    }
    *reinterpret_cast<uint4*>(xn + row * C + c0) = xo;
  }
}
template <class TO>
inline void outLn256(const __nv_bfloat16* O, const __nv_bfloat16* W, const float* bias, __nv_bfloat16* msa,
                     const float* lnS, const float* lnO, TO* xn, size_t rows) {
  static bool attr = false;
  if (!attr) { smemAttr((outLnK<TO>), (int)oln::SMEM); attr = true; }
  outLnK<TO><<<(unsigned)((rows + oln::R - 1) / oln::R), oln::NTH, oln::SMEM, STREAM>>>(O, W, bias, msa, lnS, lnO, xn, rows);
}
// the LN'd rows a projection emitted for the next module: that module's LN prefix, its block, the MSA it normalised
struct EmittedLn { std::string ln; int blk = -1; const void* msa = nullptr; void* xn = nullptr; size_t rows = 0; };
inline EmittedLn EMITTED_LN;
// the next module's: its rows if they were emitted for exactly this LN of exactly this MSA, else null (and forgotten)
inline void* takeEmittedLn(const std::string& ln, int blk, const void* msa, size_t rows) {
  EmittedLn e = EMITTED_LN; EMITTED_LN = EmittedLn{};
  return e.ln == ln && e.blk == blk && e.msa == msa && e.rows == rows ? e.xn : nullptr;
}
