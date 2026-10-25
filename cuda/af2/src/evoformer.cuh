// The input embedder and the Evoformer block - af3-any-model's AF2 multimer graph
// (alphafold3/af2/model/modules.py: EmbeddingsAndEvoformer, EvoformerIteration), float32.
#pragma once
#include "fast.cuh"
#include "flash2.cuh"
#include "globalattn.cuh"
#include "outln.cuh"

struct Trunk {
  int L, N, E;                  // residues, MSA rows, extra MSA rows
  int T = 0;                    // template rows appended to the MSA (their single features), after N
  float* msa;                   // [N, L, 256]
  float* extra;                 // [E, L, 64]
  float* pair;                  // [L, L, 128]
  const float* msaMask;         // [N, L]
  const float* extraMask;       // [E, L]
  float* pairMask;              // [L, L]
  bool opmFirst;
  // all ones (no padding, no absent row): the flash kernel then skips the mask altogether
  bool msaOnes = false, extraOnes = false, extraZeros = false, pairOnes = false;
  float* msa16 = nullptr;       // the main stack's MSA in bf16 while its blocks run (MSA16), else null
};
// MSA16: the main stack's MSA held in bf16 through its blocks (AlphaFold 2's own precision, global_config.bfloat16):
// its LayerNorms read half the bytes and its three residual adds - the row and column attention's output projections
// and the transition's second layer, bf16 GEMMs accumulating in place - read and write half; converted in before the
// blocks and out after them (af2.cu). Set where every block's MSA path has a bf16 form: --fast, every MSA row live,
// no template rows, 8 heads of 32 on an Ampere part
inline bool MSA16 = false;
// ...and in f16 where there is no bf16 tensor core (a T4: sm_75). AF2's MSA stays below ~1,000 in magnitude (5CAJ with its
// alignment 700, single-sequence 493, barnase-barstar 937, a 1,000-residue deep fold 518 - f16's range is 65,504), and
// f16 holds three more mantissa bits than bf16. The bf16-only fusions (sm_12x's split transition; outln.cuh, which
// takes f16 too and was measured slower on a T4) are skipped there
inline bool MSA16_HALF = false;
template <class F> inline void withMsa16(F&& f) { if (MSA16_HALF) f(half{}); else f(__nv_bfloat16{}); }
// a weight as the MSA16 GEMMs take it: a bf16 copy, or the f16 weight itself
template <class M> inline const void* msaWeight(const half* w, size_t n) {
  if constexpr (std::is_same_v<M, __nv_bfloat16>) return bf16Of(w, n); else return w;
}
template <class M> __global__ void msa16ToF32K(const M* in, float* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) ptx_st_global_f32(out + i, toF(in[i]));
}
// a pair pointer `elems` elements on, in the pair's own storage: t.pair is a float* that holds bf16 under PAIR16, so a bare
// `pair + n` lands 2n bf16 elements on - past 2,048 residues (a second chunk of the row attention's bias, 4M pairs a chunk)
// that read past the pair and into whatever lay there, a crash or a silently wrong bias
inline const float* pairAdvance(const float* p, size_t elems) {
  return PAIR16 ? reinterpret_cast<const float*>(reinterpret_cast<const __nv_bfloat16*>(p) + elems) : p + elems;
}

// ---------------------------------------------------------------- the embedder
// pseudo-beta (CB; CA for glycine) from atom37 positions, then the 15-bin distogram AF2 recycles
__global__ void prevDgramK(const float* pos37, const int* aatype, float* out, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L) return;
  int i = (int)(t / L), j = (int)(t % L);
  auto pb = [&](int r, int k) { int a = ptx_ldg_s32(aatype + r) == 7 ? 1 : 3; return ptx_ldg_f32(pos37 + ((size_t)r * 37 + a) * 3 + k); };
  float d2 = 0.f;
  for (int k = 0; k < 3; ++k) { float d = ptx_sub_f32(pb(i, k), pb(j, k)); d2 = ptx_fma_f32(d, d, d2); }
  for (int b = 0; b < 15; ++b) {
    float lo = ptx_fma_f32((float)b, (20.75f - 3.25f) / 14.f, 3.25f);
    float lower = ptx_mul_f32(lo, lo);
    float hi = ptx_fma_f32((float)(b + 1), (20.75f - 3.25f) / 14.f, 3.25f);
    float upper = b + 1 < 15 ? ptx_mul_f32(hi, hi) : 1e8f;
    ptx_st_global_f32(out + t * 15 + b, (d2 > lower && d2 < upper) ? 1.f : 0.f);
  }
}
__global__ void targetFeatK(const int* aatype, float* tf, int L) {      // one-hot 20 (X as V), a zero 21st
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * 21) return;
  int i = (int)(t / 21), c = (int)(t % 21);
  int a = ptx_min_s32(ptx_max_s32(ptx_ldg_s32(aatype + i), 0), 19);
  ptx_st_global_f32(tf + t, c == a ? 1.f : 0.f);
}
inline size_t AF2_CHUNK = (size_t)64 << 20;     // elements a row-chunked tensor holds, on a card short of room
// ---------------------------------------------------------------- the pair in bf16 (AF2_P16)
// Where every update the stacks run has a bf16-pair form (af2Pair16Ok), the pair is held in bf16 from the embedder
// to the end of the Evoformer: the stacks' pair kernels take it through cuda/af3's PAIR16 (set around them in
// evoformerBlock), the embedder and the templates add into it here, and after the stacks it is converted once into
// the recycled pair (f32), which the structure module, the heads and the next pass's embedder read.
// LOCALFOLD_PAIR_F32=1 keeps the f32 pair.
inline bool AF2_P16 = false;
template <class PT, class TX>
__global__ void __launch_bounds__(256, 8)
addIntoPairK(float* __restrict__ pair, const TX* __restrict__ x, size_t n, const float* __restrict__ bias128 = nullptr, float biasScale = 1.f) {
  if constexpr (std::is_same_v<PT, __nv_bfloat16> && std::is_same_v<TX, half>) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 7 < n) {
      __nv_bfloat16* p16 = reinterpret_cast<__nv_bfloat16*>(pair);
      uint4 pu = ptx_ld_global_v4_u32(p16 + i);
      uint4 xu = ptx_ldg_v4_u32(x + i);
      float4 b0 = make_float4(0.f, 0.f, 0.f, 0.f), b1 = make_float4(0.f, 0.f, 0.f, 0.f);
      if (bias128) {
        const int c8 = (threadIdx.x & 15) << 3;
        b0 = ptx_ldg_v4_f32(bias128 + c8);
        b1 = ptx_ldg_v4_f32(bias128 + c8 + 4);
        asm("mul.ftz.f32 %0, %0, %8;\n\t"
            "mul.ftz.f32 %1, %1, %8;\n\t"
            "mul.ftz.f32 %2, %2, %8;\n\t"
            "mul.ftz.f32 %3, %3, %8;\n\t"
            "mul.ftz.f32 %4, %4, %8;\n\t"
            "mul.ftz.f32 %5, %5, %8;\n\t"
            "mul.ftz.f32 %6, %6, %8;\n\t"
            "mul.ftz.f32 %7, %7, %8;"
            : "+f"(b0.x), "+f"(b0.y), "+f"(b0.z), "+f"(b0.w),
              "+f"(b1.x), "+f"(b1.y), "+f"(b1.z), "+f"(b1.w)
            : "f"(biasScale));
      }
      float2 p0 = ptx_bf16x2_to_f32x2(pu.x), p1 = ptx_bf16x2_to_f32x2(pu.y);
      float2 p2 = ptx_bf16x2_to_f32x2(pu.z), p3 = ptx_bf16x2_to_f32x2(pu.w);
      float2 x0 = ptx_f16x2_to_f32x2(xu.x), x1 = ptx_f16x2_to_f32x2(xu.y);
      float2 x2 = ptx_f16x2_to_f32x2(xu.z), x3 = ptx_f16x2_to_f32x2(xu.w);
      uint4 out4;
      out4.x = ptx_cvt_bf16x2_f32(p0.x + (x0.x + b0.x), p0.y + (x0.y + b0.y));
      out4.y = ptx_cvt_bf16x2_f32(p1.x + (x1.x + b0.z), p1.y + (x1.y + b0.w));
      out4.z = ptx_cvt_bf16x2_f32(p2.x + (x2.x + b1.x), p2.y + (x2.y + b1.y));
      out4.w = ptx_cvt_bf16x2_f32(p3.x + (x3.x + b1.z), p3.y + (x3.y + b1.w));
      ptx_st_global_v4_u32(p16 + i, out4);
    }
    return;
  }
  size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < n) {
    float4 p = pairLd4<PT>(pair, i);
    float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
    if (bias128) {
      b = ptx_ldg_v4_f32(bias128 + ((threadIdx.x & 31) << 2));
      asm("mul.ftz.f32 %0, %0, %4;\n\t"
          "mul.ftz.f32 %1, %1, %4;\n\t"
          "mul.ftz.f32 %2, %2, %4;\n\t"
          "mul.ftz.f32 %3, %3, %4;"
          : "+f"(b.x), "+f"(b.y), "+f"(b.z), "+f"(b.w) : "f"(biasScale));
    }
    if constexpr (std::is_same_v<TX, float>) {
      float4 v = ptx_ldg_v4_f32(x + i);
      p.x += v.x + b.x; p.y += v.y + b.y; p.z += v.z + b.z; p.w += v.w + b.w;
    } else {
      uint2 u = ptx_ldg_v2_u32(x + i);
      float2 a = ptx_f16x2_to_f32x2(u.x);
      float2 c = ptx_f16x2_to_f32x2(u.y);
      p.x += a.x + b.x; p.y += a.y + b.y; p.z += c.x + b.z; p.w += c.y + b.w;
    }
    pairSt4<PT>(pair, i, p);
  } else {
    for (size_t k = i; k < n; ++k)
      pairSt<PT>(pair, k, pairLd<PT>(pair, k) + (toF(x[k]) + (bias128 ? ptx_ldg_f32(bias128 + (k & 127)) * biasScale : 0.f)));
  }
}
__global__ void __launch_bounds__(256, 8)
pairBf16ToF32K(const __nv_bfloat16* __restrict__ in, float* __restrict__ out, size_t n) {
  size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < n) ptx_st_global_v4_f32(out + i, pairLd4<__nv_bfloat16>(reinterpret_cast<const float*>(in), i));
  else for (size_t k = i; k < n; ++k) ptx_st_global_f32(out + k, toF(in[k]));
}
template <class PT, bool FAST_INV = false, class TX = float>
__global__ void __launch_bounds__(256, 6)
opmAdd128K(float* __restrict__ pair, const TX* __restrict__ x, const float* __restrict__ bias,
           const float* __restrict__ norm, size_t i0, int Bi, int L, float xs = 1.f, float normVal = 0.f) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (!std::is_same_v<PT, float> && !std::is_same_v<TX, float>) {
    if (t >= (size_t)Bi * L * 16) return;
    int f = (int)((t & 15) << 3);
    size_t p = i0 * L + (t >> 4);
    float denom = 1e-3f + (norm ? ptx_ldg_f32(norm + p) : normVal);
    uint4 ux = ptx_ldg_v4_u32(x + (t << 3));
    float4 b0 = ptx_ldg_v4_f32(bias + f);
    float4 b1 = ptx_ldg_v4_f32(bias + f + 4);
    PT* pp = reinterpret_cast<PT*>(pair) + (p << 7) + f;
    uint4 up = ptx_ld_global_v4_u32(pp);
    float2 a0 = ptx_f16x2_to_f32x2(ux.x), a1 = ptx_f16x2_to_f32x2(ux.y);
    float2 a2 = ptx_f16x2_to_f32x2(ux.z), a3 = ptx_f16x2_to_f32x2(ux.w);
    float2 q0 = std::is_same_v<PT, __nv_bfloat16> ? ptx_bf16x2_to_f32x2(up.x) : ptx_f16x2_to_f32x2(up.x);
    float2 q1 = std::is_same_v<PT, __nv_bfloat16> ? ptx_bf16x2_to_f32x2(up.y) : ptx_f16x2_to_f32x2(up.y);
    float2 q2 = std::is_same_v<PT, __nv_bfloat16> ? ptx_bf16x2_to_f32x2(up.z) : ptx_f16x2_to_f32x2(up.z);
    float2 q3 = std::is_same_v<PT, __nv_bfloat16> ? ptx_bf16x2_to_f32x2(up.w) : ptx_f16x2_to_f32x2(up.w);
    if constexpr (FAST_INV) {
      float inv = ptx_rcp_f32(denom);
      q0.x = ptx_fma_f32(ptx_fma_f32(xs, a0.x, b0.x), inv, q0.x);
      q0.y = ptx_fma_f32(ptx_fma_f32(xs, a0.y, b0.y), inv, q0.y);
      q1.x = ptx_fma_f32(ptx_fma_f32(xs, a1.x, b0.z), inv, q1.x);
      q1.y = ptx_fma_f32(ptx_fma_f32(xs, a1.y, b0.w), inv, q1.y);
      q2.x = ptx_fma_f32(ptx_fma_f32(xs, a2.x, b1.x), inv, q2.x);
      q2.y = ptx_fma_f32(ptx_fma_f32(xs, a2.y, b1.y), inv, q2.y);
      q3.x = ptx_fma_f32(ptx_fma_f32(xs, a3.x, b1.z), inv, q3.x);
      q3.y = ptx_fma_f32(ptx_fma_f32(xs, a3.y, b1.w), inv, q3.y);
    } else {
      q0.x += ptx_fma_f32(xs, a0.x, b0.x) / denom;
      q0.y += ptx_fma_f32(xs, a0.y, b0.y) / denom;
      q1.x += ptx_fma_f32(xs, a1.x, b0.z) / denom;
      q1.y += ptx_fma_f32(xs, a1.y, b0.w) / denom;
      q2.x += ptx_fma_f32(xs, a2.x, b1.x) / denom;
      q2.y += ptx_fma_f32(xs, a2.y, b1.y) / denom;
      q3.x += ptx_fma_f32(xs, a3.x, b1.z) / denom;
      q3.y += ptx_fma_f32(xs, a3.y, b1.w) / denom;
    }
    uint4 out4;
    if constexpr (std::is_same_v<PT, __nv_bfloat16>) {
      out4 = make_uint4(ptx_cvt_bf16x2_f32(q0.x, q0.y), ptx_cvt_bf16x2_f32(q1.x, q1.y),
                        ptx_cvt_bf16x2_f32(q2.x, q2.y), ptx_cvt_bf16x2_f32(q3.x, q3.y));
    } else {
      out4 = make_uint4(ptx_cvt_f16x2_f32(q0.x, q0.y), ptx_cvt_f16x2_f32(q1.x, q1.y),
                        ptx_cvt_f16x2_f32(q2.x, q2.y), ptx_cvt_f16x2_f32(q3.x, q3.y));
    }
    ptx_st_global_v4_u32(pp, out4);
    return;
  }
  if (t >= (size_t)Bi * L * 32) return;
  int f = (int)((t & 31) << 2);
  size_t p = i0 * L + (t >> 5);
  float denom = 1e-3f + (norm ? ptx_ldg_f32(norm + p) : normVal);
  float4 x4;
  if constexpr (std::is_same_v<TX, float>) {
    x4 = ptx_ldg_v4_f32(x + (t << 2));
  } else {
    uint2 u = ptx_ldg_v2_u32(x + (t << 2));
    float2 a = ptx_f16x2_to_f32x2(u.x);
    float2 c = ptx_f16x2_to_f32x2(u.y);
    x4 = make_float4(a.x, a.y, c.x, c.y);
  }
  float4 b4 = ptx_ldg_v4_f32(bias + f);
  float4 p4 = pairLd4<PT>(pair, (p << 7) + f);
  if constexpr (FAST_INV) {
    float inv = ptx_rcp_f32(denom);
    p4.x = ptx_fma_f32(ptx_fma_f32(xs, x4.x, b4.x), inv, p4.x);
    p4.y = ptx_fma_f32(ptx_fma_f32(xs, x4.y, b4.y), inv, p4.y);
    p4.z = ptx_fma_f32(ptx_fma_f32(xs, x4.z, b4.z), inv, p4.z);
    p4.w = ptx_fma_f32(ptx_fma_f32(xs, x4.w, b4.w), inv, p4.w);
  } else {
    p4.x += ptx_fma_f32(xs, x4.x, b4.x) / denom;
    p4.y += ptx_fma_f32(xs, x4.y, b4.y) / denom;
    p4.z += ptx_fma_f32(xs, x4.z, b4.z) / denom;
    p4.w += ptx_fma_f32(xs, x4.w, b4.w) / denom;
  }
  pairSt4<PT>(pair, (p << 7) + f, p4);
}
template <class PT>
__global__ void opmZeroMask128K(float* __restrict__ pair, const float* __restrict__ bias, size_t pairs, float denom = 1e-3f) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * 32) return;
  int f = (int)((t & 31) << 2);
  float4 b4 = ptx_ldg_v4_f32(bias + f);
  float4 p4 = pairLd4<PT>(pair, t << 2);
  p4.x += b4.x / denom; p4.y += b4.y / denom; p4.z += b4.z / denom; p4.w += b4.w / denom;
  pairSt4<PT>(pair, t << 2, p4);
}
// the pair as f32 for a reader with no bf16 form (the template embedders' query): itself, or a converted copy
inline const float* pairF32(const float* pair, size_t n) {
  if (!AF2_P16) return pair;
  float* f = scratch<float>("p16.view", n);
  pairBf16ToF32K<<<blocks((n + 3) / 4), 256, 0, STREAM>>>(reinterpret_cast<const __nv_bfloat16*>(pair), f, n);
  return f;
}
template <class PT = float>
__global__ void pairOuterSumK(float* pair, const float* left, const float* right, int L, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / L), j = (int)(ij % L);
  pairSt<PT>(pair, t, ptx_add_f32(ptx_ldg_f32(left + (size_t)i * C + c), ptx_ldg_f32(right + (size_t)j * C + c)));
}
__global__ void __launch_bounds__(256, 8)
msaEmbedAddK(float* __restrict__ msa, const float* __restrict__ bias, const int* __restrict__ aatype,
             const float* __restrict__ wP1d, const float* __restrict__ bP1d,
             const float* __restrict__ prevRowLn, size_t rows, int L, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (C == 256) {
    if (t >= rows * 64) return;
    int c = (int)(t & 63) << 2;
    size_t row = t >> 6;
    int i = (int)((unsigned)row % (unsigned)L);
    int a = ptx_min_s32(ptx_max_s32(ptx_ldg_s32(aatype + i), 0), 19);
    size_t idx = t << 2;
    float4 mv = ptx_ld_global_v4_f32(msa + idx);
    float4 bv = ptx_ldg_v4_f32(bias + c);
    float4 wv = ptx_ldg_v4_f32(wP1d + a * 256 + c);
    float4 bp = ptx_ldg_v4_f32(bP1d + c);
    mv.x = ptx_add_f32(ptx_add_f32(mv.x, bv.x), ptx_add_f32(wv.x, bp.x));
    mv.y = ptx_add_f32(ptx_add_f32(mv.y, bv.y), ptx_add_f32(wv.y, bp.y));
    mv.z = ptx_add_f32(ptx_add_f32(mv.z, bv.z), ptx_add_f32(wv.z, bp.z));
    mv.w = ptx_add_f32(ptx_add_f32(mv.w, bv.w), ptx_add_f32(wv.w, bp.w));
    if (row < (size_t)L) {
      float4 pv = ptx_ldg_v4_f32(prevRowLn + idx);
      mv.x = ptx_add_f32(mv.x, pv.x); mv.y = ptx_add_f32(mv.y, pv.y);
      mv.z = ptx_add_f32(mv.z, pv.z); mv.w = ptx_add_f32(mv.w, pv.w);
    }
    ptx_st_global_v4_f32(msa + idx, mv);
    return;
  }
  if (t >= rows * C) return;
  int c = (int)(t % C), i = (int)(t / C % L);
  int a = ptx_min_s32(ptx_max_s32(ptx_ldg_s32(aatype + i), 0), 19);
  float v = ptx_add_f32(ptx_ld_global_f32(msa + t), ptx_ldg_f32(bias + c));
  v = ptx_add_f32(v, ptx_add_f32(ptx_ldg_f32(wP1d + a * C + c), ptx_ldg_f32(bP1d + c)));
  if (t < (size_t)L * C) v = ptx_add_f32(v, ptx_ldg_f32(prevRowLn + t));
  ptx_st_global_f32(msa + t, v);
}
template <class PT>
__global__ void __launch_bounds__(256, 6)
embedPairK(const float* __restrict__ wLeft, const float* __restrict__ bLeft,
           const float* __restrict__ wRight, const float* __restrict__ bRight,
           const float* __restrict__ pos37, const int* __restrict__ aatype,
           const float* __restrict__ wDg, const float* __restrict__ bDg,
           const float* __restrict__ prevPair, const float* __restrict__ lnScale, const float* __restrict__ lnOffset,
           const int* __restrict__ ri, const int* __restrict__ asym, const int* __restrict__ entity, const int* __restrict__ sym,
           const float* __restrict__ wRel, const float* __restrict__ bRel,
           float* __restrict__ pair, int L, bool chainRows) {
  size_t ij = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (ij >= (size_t)L * L) return;
  int i = (int)((unsigned)ij / (unsigned)L), j = (int)((unsigned)ij % (unsigned)L);
  int rawAi = ptx_ldg_s32(aatype + i), rawAj = ptx_ldg_s32(aatype + j);
  int ai = ptx_min_s32(ptx_max_s32(rawAi, 0), 19), aj = ptx_min_s32(ptx_max_s32(rawAj, 0), 19);
  int pi = rawAi == 7 ? 1 : 3, pj = rawAj == 7 ? 1 : 3;
  float d2 = 0.f;
  for (int k = 0; k < 3; ++k) {
    float d = ptx_sub_f32(ptx_ldg_f32(pos37 + ((size_t)i * 37 + pi) * 3 + k),
                          ptx_ldg_f32(pos37 + ((size_t)j * 37 + pj) * 3 + k));
    d2 = ptx_fma_f32(d, d, d2);
  }
  float lo = ptx_fma_f32((float)lane, (20.75f - 3.25f) / 14.f, 3.25f);
  float hi = ptx_fma_f32((float)(lane + 1), (20.75f - 3.25f) / 14.f, 3.25f);
  uint32_t m0 = ptx_ballot_sync(lane < 15 && d2 > ptx_mul_f32(lo, lo) && d2 < (lane + 1 < 15 ? ptx_mul_f32(hi, hi) : 1e8f));
  int dgBin = m0 ? (ptx_ffs_u32(m0) - 1) : -1;
  int off = ptx_ldg_s32(ri + i) - ptx_ldg_s32(ri + j);
  int clipped = ptx_min_s32(ptx_max_s32(off + 32, 0), 64);
  int asymI = ptx_ldg_s32(asym + i), asymJ = ptx_ldg_s32(asym + j);
  int r0 = asymI == asymJ ? clipped : chainRows ? 65 : asymI > asymJ ? 64 : 0;
  bool sameEntity = ptx_ldg_s32(entity + i) == ptx_ldg_s32(entity + j);
  int rc = ptx_min_s32(ptx_max_s32(ptx_ldg_s32(sym + i) - ptx_ldg_s32(sym + j) + 2, 0), 4);
  int r2 = 67 + (sameEntity ? rc : 5);
  const float* pr = prevPair + ij * 128;
  float pv[4], s = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) { pv[k] = ptx_ldg_f32(pr + lane + 32 * k); s = ptx_add_f32(s, pv[k]); }
  s = ptx_warp_sum_f32(s);
  float mean = ptx_mul_f32(s, 1.f / 128.f), v = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) { float d = ptx_sub_f32(pv[k], mean); v = ptx_fma_f32(d, d, v); }
  v = ptx_warp_sum_f32(v);
  float inv = ptx_rsqrt_f32(ptx_fma_f32(v, 1.f / 128.f, 1e-5f));
  auto step = [](float x) {
    if constexpr (std::is_same_v<PT, float>) return x;
    else return toF(fromF<PT>(x));
  };
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    int c = lane + 32 * k;
    float left = ptx_add_f32(ptx_ldg_f32(wLeft + ai * 128 + c), ptx_ldg_f32(bLeft + c));
    float right = ptx_add_f32(ptx_ldg_f32(wRight + aj * 128 + c), ptx_ldg_f32(bRight + c));
    float acc = step(ptx_add_f32(left, right));
    float dg = ptx_add_f32(dgBin >= 0 ? ptx_ldg_f32(wDg + dgBin * 128 + c) : 0.f, ptx_ldg_f32(bDg + c));
    acc = step(ptx_add_f32(acc, dg));
    float ln = ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(pv[k], mean), inv), ptx_ldg_f32(lnScale + c), ptx_ldg_f32(lnOffset + c));
    acc = step(ptx_add_f32(acc, ln));
    float rel = ptx_add_f32(ptx_add_f32(ptx_ldg_f32(wRel + r0 * 128 + c), sameEntity ? ptx_ldg_f32(wRel + 66 * 128 + c) : 0.f),
                            ptx_add_f32(ptx_ldg_f32(wRel + r2 * 128 + c), ptx_ldg_f32(bRel + c)));
    pairSt<PT>(pair, ij * 128 + c, ptx_add_f32(acc, rel));
  }
}
__global__ void __launch_bounds__(256, 6)
extraEmbedK(const int* __restrict__ codes, const float* __restrict__ hasDel,
            const float* __restrict__ delVal, const float* __restrict__ w,
            const float* __restrict__ b, float* __restrict__ out, size_t rows) {
  __shared__ __align__(16) float sw[25 * 64], sb[64];
  for (int i = threadIdx.x; i < 25 * 16; i += blockDim.x) {
    ptx_st_shared_v4_f32(sw + (i << 2), ptx_ldg_v4_f32(w + (i << 2)));
  }
  if (threadIdx.x < 16) ptx_st_shared_v4_f32(sb + (threadIdx.x << 2), ptx_ldg_v4_f32(b + (threadIdx.x << 2)));
  __syncthreads();
  size_t t4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t4 >= rows * 16) return;
  size_t r = t4 >> 4; int c = (int)(t4 & 15) << 2;
  int code = ptx_ldg_s32(codes + r);
  float hd = ptx_ldg_f32(hasDel + r), dv = ptx_ldg_f32(delVal + r);
  float4 v = (code >= 0 && code < 23) ? ptx_ld_shared_v4_f32(sw + code * 64 + c) : make_float4(0.f, 0.f, 0.f, 0.f);
  float4 w23 = ptx_ld_shared_v4_f32(sw + 23 * 64 + c);
  float4 w24 = ptx_ld_shared_v4_f32(sw + 24 * 64 + c);
  float4 bv = ptx_ld_shared_v4_f32(sb + c);
  v.x = ptx_add_f32(ptx_fma_f32(dv, w24.x, ptx_fma_f32(hd, w23.x, v.x)), bv.x);
  v.y = ptx_add_f32(ptx_fma_f32(dv, w24.y, ptx_fma_f32(hd, w23.y, v.y)), bv.y);
  v.z = ptx_add_f32(ptx_fma_f32(dv, w24.z, ptx_fma_f32(hd, w23.z, v.z)), bv.z);
  v.w = ptx_add_f32(ptx_fma_f32(dv, w24.w, ptx_fma_f32(hd, w23.w, v.w)), bv.w);
  ptx_st_global_v4_f32(out + (t4 << 2), v);
}
__global__ void pairMaskK(const float* seqMask, float* out, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)L * L) ptx_st_global_f32(out + t, ptx_mul_f32(ptx_ldg_f32(seqMask + t / L), ptx_ldg_f32(seqMask + t % L)));
}

// a fold short of room (af2.cu): the chunked forms may engage, and no CUDA graph is captured, so a choice
// made from the free memory cannot differ between a pass and its capture
inline bool AF2_TIGHT = false;
// prev: prevMsaRow [L, 256], prevPair [L, L, 128] (null: the pair itself, re-embedded in place), prevPos [L, 37, 3]
inline void embed(Trunk& t, int pass, const float* prevMsaRow, const float* prevPair, const float* prevPos) {
  const std::string E = "evoformer/";
  int L = t.L, N = t.N;
  std::string f = "f" + std::to_string(pass) + "/";
  const int* aatype = Idev("aatype");
  // msa = preprocess_1d(target)[None] + preprocess_msa(msa_feat); row 0 += LN(prev msa first row)
  gemm(W(f + "msa_feat"), P(E + "preprocess_msa/weights", -1), t.msa, (size_t)N * L, 49, 256);
  float* prevRowLn = scratch<float>("emb.prevRow", (size_t)L * 256);
  layerNorm(prevMsaRow, prevRowLn, L, 256, E + "prev_msa_first_row_norm");
  msaEmbedAddK<<<blocks((size_t)N * L * 64), 256, 0, STREAM>>>(
    t.msa, P(E + "preprocess_msa/bias", -1), aatype, P(E + "preprocess_1d/weights", -1), P(E + "preprocess_1d/bias", -1),
    prevRowLn, (size_t)N * L, L, 256);
  // pair = left[i] + right[j] + prev_pos_linear(dgram) + LN(prev pair) + relpos
  size_t pairs = (size_t)L * L;
  if (!prevPair && AF2_P16) { fprintf(stderr, "the in-place re-embedding has no bf16-pair form\n"); exit(1); }
  WITH_PT(AF2_P16, embedPairK<PT><<<(unsigned)((pairs + 7) / 8), 256, 0, STREAM>>>(
    P(E + "left_single/weights", -1), P(E + "left_single/bias", -1),
    P(E + "right_single/weights", -1), P(E + "right_single/bias", -1),
    prevPos, aatype,
    P(E + "prev_pos_linear/weights", -1), P(E + "prev_pos_linear/bias", -1),
    prevPair ? prevPair : t.pair, P(E + "prev_pair_norm/scale", -1), P(E + "prev_pair_norm/offset", -1),
    Idev("residue_index"), Idev("asym_id"), Idev("entity_id"), Idev("sym_id"),
    P(E + "~_relative_encoding/position_activations/weights", -1), P(E + "~_relative_encoding/position_activations/bias", -1),
    t.pair, L, M.flag("meta/multimer")));
  // the extra MSA's activations
  size_t erows = (size_t)t.E * L;
  auto ones = [](const float* h, size_t n) { for (size_t i = 0; i < n; ++i) if (h[i] != 1.f) return false; return true; };
  auto zeros = [](const float* h, size_t n) { for (size_t i = 0; i < n; ++i) if (h[i] != 0.f) return false; return true; };
  auto tmplOnes = [&]() {
    if (t.T == 0) return true;
    if (M.flag("meta/multimer") || !M.has("t/mask")) return false;
    const float* tm = M.f("t/mask");
    for (size_t r = 0; r < (size_t)t.T * L; ++r) {
      const float* m = tm + r * 37;
      if (m[0] * m[1] * m[2] * m[4] != 1.f) return false;
    }
    return true;
  };
  t.msaOnes = ones(M.f(f + "msa_mask"), (size_t)N * L) && tmplOnes();     // (a template row's mask is the template's)
  t.extraOnes = ones(M.f(f + "extra_msa_mask"), erows);
  t.extraZeros = zeros(M.f(f + "extra_msa_mask"), erows);
  if (!t.extraZeros) {
    extraEmbedK<<<blocks(erows * 16), 256, 0, STREAM>>>(
      Idev(f + "extra_msa"), W(f + "extra_has_deletion"), W(f + "extra_deletion_value"),
      P(E + "extra_msa_activations/weights", -1), P(E + "extra_msa_activations/bias", -1), t.extra, erows);
  }
  // the masks into fixed buffers: the stacks after this are replayed as one CUDA graph from pass 1 on,
  // and a graph's pointers do not change between passes
  float* mm = scratch<float>("mask.msa", (size_t)(N + t.T) * L); float* em = scratch<float>("mask.extra", erows);
  CK(cudaMemcpyAsync(mm, W(f + "msa_mask"), (size_t)N * L * 4, cudaMemcpyDeviceToDevice, STREAM));
  if (!t.extraZeros) CK(cudaMemcpyAsync(em, W(f + "extra_msa_mask"), erows * 4, cudaMemcpyDeviceToDevice, STREAM));
  t.msaMask = mm; t.extraMask = em;
  if (pass == 0) {
    t.pairOnes = ones(M.f("seq_mask"), L);
    pairMaskK<<<blocks(pairs), 256, 0, STREAM>>>(W("seq_mask"), t.pairMask, L);
  }
}

// ---------------------------------------------------------------- the block's modules
__global__ void packBatched4K(const float* __restrict__ qw, const float* __restrict__ kw,
                              const float* __restrict__ vw, const float* __restrict__ gw,
                              float* __restrict__ out, int nblk, size_t slice) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)nblk * slice) return;
  int b = (int)(t / slice); size_t s = t % slice;
  float* o = out + (size_t)b * 4 * slice;
  ptx_st_global_f32(o + s, ptx_ldg_f32(qw + t));
  ptx_st_global_f32(o + slice + s, ptx_ldg_f32(kw + t));
  ptx_st_global_f32(o + 2 * slice + s, ptx_ldg_f32(vw + t));
  ptx_st_global_f32(o + 3 * slice + s, ptx_ldg_f32(gw + t));
}
// gated self-attention with its own q/k/v/gate/output weights, over [Bt, n, C] rows
inline void gatedAttention(const float* xn, int Bt, int n, int C, const std::string& A, int blk, int H, int D,
                           const float* keyMask, const float* pairBias, float* out,
                           float* addInto = nullptr, int swapA = 0, int swapB = 0) {
  size_t rows = (size_t)Bt * n; int Wd = H * D;
  size_t wSlice = (size_t)C * Wd, oSlice = rows * (size_t)Wd;
  static std::map<const float*, const float*> qkvgWCache;
  int b0 = blk < 0 ? -1 : 0;
  const float* qw0 = P(A + "/query_w", b0);
  auto it = qkvgWCache.find(qw0);
  if (it == qkvgWCache.end()) {
    int nblk = blk < 0 ? 1 : (int)dimW(A + "/query_w", 0);
    float* w4 = wpool<float>((size_t)nblk * 4 * wSlice);
    packBatched4K<<<blocks((size_t)nblk * wSlice), 256, 0, STREAM>>>(
      qw0, P(A + "/key_w", b0), P(A + "/value_w", b0), P(A + "/gating_w", b0), w4, nblk, wSlice);
    it = qkvgWCache.emplace(qw0, w4).first;
  }
  const float* w4 = it->second + (blk < 0 ? 0 : (size_t)blk * 4 * wSlice);
  float* q = scratch<float>("att.qkvg", 4 * oSlice);
  float *k = q + oSlice, *v = q + 2 * oSlice, *g = q + 3 * oSlice;
  float* o = scratch<float>("att.o", oSlice);
  const float one = 1.f, zero = 0.f;
  CB(cublasSgemmStridedBatched(::H, CUBLAS_OP_N, CUBLAS_OP_N, Wd, (int)rows, C, &one,
                               w4, Wd, (long long)wSlice, xn, C, 0, &zero, q, Wd, (long long)oSlice, 4));
  attention(D, q, k, v, g, keyMask, pairBias, o, Bt, n, H, P(A + "/gating_b", blk));
  gemm(o, P(A + "/output_w", blk), out, rows, Wd, C);
  if (addInto) {
    addBiasSwapInto(addInto, out, P(A + "/output_b", blk), swapA ? swapB : 1, swapA ? swapA : (int)rows, C, swapA != 0);
  } else {
    addBiasK<<<blocks((C & 3) == 0 ? rows * (size_t)(C >> 2) : rows * C), 256, 0, STREAM>>>(out, P(A + "/output_b", blk), rows, C);
  }
}
// [pairs, H] f32 pair-bias projection (+ 1e9 (mask - 1) when a mask is given) -> [H, L, stride] f16 in
// the flash kernel's log2 units
__global__ void biasFromProjK(const float* proj, const float* pairMask, half* out, int L, int H, int stride, bool transposed) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * H) return;
  int h = (int)(t % H); size_t ij = t / H; int i = (int)(ij / L), j = (int)(ij % L);
  float v = ptx_add_f32(ptx_ldg_f32(proj + t), (pairMask ? ptx_mul_f32(1e9f, ptx_sub_f32(ptx_ldg_f32(pairMask + ij), 1.f)) : 0.f));
  // transposed: the ending node's bias, b[h][q][k] = proj[k][q] (it attends along the first axis)
  if (transposed) { int x = i; i = j; j = x; }
  ptx_st_global_u16(out + ((size_t)h * L + i) * stride + j, ptx_f32_to_f16_raw(ptx_max_f32(ptx_mul_f32(v, LOG2E), -6e4f)));
  if (j == L - 1) for (int pad = L; pad < stride; ++pad) ptx_st_global_u16(out + ((size_t)h * L + i) * stride + pad, 0);
}
inline const half* zeroBias(int H, int n, int stride) {        // a bias-free attention's bias, once
  static half* z = nullptr; static size_t have = 0;
  size_t need = (size_t)H * n * stride;
  if (need > have) {
    if (z) CK(cudaFree(z));
    z = dallocT<half>(need); CK(cudaMemset(z, 0, need * 2)); have = need;
  }
  return z;
}
// the MSA16 stack's row (NB false: the pair bias) and column (NB: none) attention, bf16 out, with the pair track's
// speculative softmax (FLASH_SPEC), and the row attention its 64-key tiles where an SM holds two blocks of them
// (flash2RWideKeysFor: unless 48-key tiles pad n meaningfully less). A100, 1,000 residues, 508 rows: the row attention 914 -> 862 ms, the column 524 -> 511 (64 keys
// there 545: no bias tile for the wider tile to amortise)
template <bool NB, class B16>
inline void msaFlash(const half* qkvg, const half* bias, int stride, B16* o, int n, int heads, size_t rows, float scale,
                     size_t rowStride = 0, size_t posStride = 0, size_t outRowStride = 0, size_t outPosStride = 0) {
  if constexpr (std::is_same_v<B16, half>) {
    if (flashRegStaged()) {   // (a T4: the one-stage register-staged form, its tiles as flash2R1Strided's; MSA16_HALF is true there)
      flashGrid2RRun<32, 2, NB ? 48 : 32, 2, 2, NB, true, B16>(qkvg, bias, stride, o, n, heads, rows, scale, nullptr, rowStride, posStride, outRowStride, outPosStride);
      return;
    }
  }
  if constexpr (NB) {
    const int p64 = (n + 63) / 64 * 64, p48 = (n + 47) / 48 * 48;
    if (n >= 128 && p64 <= 1.025 * p48 && fitsSmem(2 * fa2Stage<32, 4, 64, 2, 1, true>())) {
      if (FLASH_SPEC) flashGrid2RRun<32, 4, 64, 2, 1, true, false, B16, true>(qkvg, bias, stride, o, n, heads, rows, scale, nullptr, rowStride, posStride, outRowStride, outPosStride);
      else flashGrid2RRun<32, 4, 64, 2, 1, true, false, B16, false>(qkvg, bias, stride, o, n, heads, rows, scale, nullptr, rowStride, posStride, outRowStride, outPosStride);
      return;
    }
    if (n >= 96 && fitsSmem(2 * fa2Stage<32, 4, 48, 2, 1, true>())) {
      if (FLASH_SPEC) flashGrid2RRun<32, 4, 48, 2, 1, true, false, B16, true>(qkvg, bias, stride, o, n, heads, rows, scale, nullptr, rowStride, posStride, outRowStride, outPosStride);
      else flashGrid2RRun<32, 4, 48, 2, 1, true, false, B16, false>(qkvg, bias, stride, o, n, heads, rows, scale, nullptr, rowStride, posStride, outRowStride, outPosStride);
      return;
    }
  }
  if (FLASH_SPEC && !NB && flash2RWideKeysFor(n))
    flashGrid2RRun<32, 2, 64, 2, 2, NB, false, B16, true>(qkvg, bias, stride, o, n, heads, rows, scale, nullptr, rowStride, posStride, outRowStride, outPosStride);
  else if (FLASH_SPEC)
    flashGrid2RRun<32, 2, 48, 2, 2, NB, false, B16, true>(qkvg, bias, stride, o, n, heads, rows, scale, nullptr, rowStride, posStride, outRowStride, outPosStride);
  else flashGrid2RRun<32, 2, 48, 2, 2, NB, false, B16>(qkvg, bias, stride, o, n, heads, rows, scale, nullptr, rowStride, posStride, outRowStride, outPosStride);
}
// --fast: from the normalised input xn (f16): one q/k/v/gate GEMM straight into the flash kernel's
// layout -> flash attention -> the output projection with its bias, added into `residual` (transposed
// back first when `transposedBack`, xn being [n, Bt] of the residual's [Bt, n])
// y[q * A + b] += x[(b - b0) * B + q] for rows b in [b0, b0 + cnt): a chunk of transposedBack's swapAdd
__global__ void swapAddRowsK(float* __restrict__ y, const float* __restrict__ x, int A, int B, int C, size_t b0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * B * C) return;
  int c = (int)(t % C); size_t r = t / C; size_t db = r % cnt, q = r / cnt, b = b0 + db;
  size_t dst = (q * A + b) * C + c;
  ptx_st_global_f32(y + dst, ptx_add_f32(ptx_ld_global_f32(y + dst), ptx_ldg_f32(x + (db * B + q) * C + c)));
}
inline void attentionCore(const half* xn, int Bt, int n, int C, const std::string& A, int blk, const float* keyMask,
                          const half* bias, float* residual, bool transposedBack) {
  size_t rows = (size_t)Bt * n;
  AttnW w = attnWeights(A, blk, C, flashNarrow(keyMask));
  int Wp = w.H * w.Dp, stride = (n + 7) / 8 * 8;
  if (!bias) bias = zeroBias(w.H, n, stride);
  // where the q/k/v/gate and the output of every row would not fit with room to spare, a chunk of rows at
  // a time (the flash kernel's row offset indexes only the key mask)
  size_t need = (rows + 128) * 4 * Wp * 2 + rows * Wp * 2 + (transposedBack ? rows * C * 4 : 0);
  size_t R = !AF2_TIGHT || roomFor(need, { "fatt.qkvg", "fatt.o", "fatt.tmp" }) ? Bt
           : std::max<size_t>(1, std::min<size_t>(Bt, AF2_CHUNK / ((size_t)n * 4 * Wp)));
  half* qkvg = scratch<half>("fatt.qkvg", (R * n + 128) * 4 * Wp);
  half* o = scratch<half>("fatt.o", R * n * Wp);
  float* tmp = transposedBack ? scratch<float>("fatt.tmp", R * n * C) : nullptr;
  for (size_t b0 = 0; b0 < (size_t)Bt; b0 += R) {
    size_t bc = std::min(R, (size_t)Bt - b0), rc = bc * n;
    ltGemm(xn + b0 * n * C, w.qkvg, qkvg, true, rc, C, 4 * Wp, w.qkvgBias, false, 0.f);
    flashGrid<half>(qkvg, bias, stride, keyMask, o, n, w.H, w.Dp, b0, bc, false, 1.f / sqrtf((float)w.D));
    if (!transposedBack) {
      ltGemm(o, w.out, residual + b0 * n * C, false, rc, Wp, C, P(A + "/output_b", blk), false, 1.f);
    } else if (R == (size_t)Bt) {
      ltGemm(o, w.out, tmp, false, rows, Wp, C, P(A + "/output_b", blk), false, 0.f);
      swapAddK<<<blocks(rows * C), 256, 0, STREAM>>>(residual, tmp, Bt, n, C);
    } else {
      ltGemm(o, w.out, tmp, false, rc, Wp, C, P(A + "/output_b", blk), false, 0.f);
      swapAddRowsK<<<blocks(rc * C), 256, 0, STREAM>>>(residual, tmp, Bt, n, C, b0, bc);
    }
  }
}
// the same, attending ACROSS the leading axis of xn [n][Bt][C] (an MSA's columns, the triangle's ending
// node) where it lies: the strided flash kernel reads rows and positions at strides, and the output
// lands in the residual's own layout, so nothing is transposed
inline void attentionCoreAcross(const half* xn, int Bt, int n, int C, const std::string& A, int blk, const half* bias,
                                float* residual, const float* keyMask = nullptr) {
  size_t rows = (size_t)Bt * n;
  AttnW w = attnWeights(A, blk, C);
  int Wp = w.H * w.Dp, stride = (n + 7) / 8 * 8;
  half* qkvg = scratch<half>("fatt.qkvg", (rows + 128) * 4 * Wp);
  ltGemm(xn, w.qkvg, qkvg, true, rows, C, 4 * Wp, w.qkvgBias, false, 0.f);
  // no bias (the column attention): the grid kernel takes none and loads no zeros; any other form wants them
  if (!bias && !(!keyMask && FLASH_2R && w.Dp == 32 && (!flashRegStaged() || flash2R1Takes()))) bias = zeroBias(w.H, n, stride);
  half* o = scratch<half>("fatt.o", rows * Wp);
  flashGridStrided(qkvg, bias, stride, keyMask, o, n, w.H, w.Dp, Bt, 1.f / sqrtf((float)w.D),
                   (size_t)4 * Wp, (size_t)Bt * 4 * Wp, (size_t)Wp, (size_t)Bt * Wp);
  ltGemm(o, w.out, residual, false, rows, Wp, C, P(A + "/output_b", blk), false, 1.f);
}
// the pair bias, --fast: LN(pair) in f16 -> [pairs, H] -> the flash layout
inline const half* pairBiasFast(const half* pn, int L, int C, int H, const float* w, const float* pairMask, bool transposed = false) {
  size_t pairs = (size_t)L * L; int stride = (L + 7) / 8 * 8;
  float* proj = scratch<float>("fbias.proj", pairs * H);
  static std::map<const float*, half*> wh;
  auto it = wh.find(w);
  if (it == wh.end()) {
    half* h = wpool<half>((size_t)C * H);
    toHalfK<<<blocks((size_t)C * H), 256, 0, STREAM>>>(w, h, (size_t)C * H);
    it = wh.emplace(w, h).first;
  }
  ltGemm(pn, it->second, proj, false, pairs, C, H, nullptr, false, 0.f);
  half* bias = scratch<half>("fbias.bias", (size_t)H * L * stride);
  biasFromProjK<<<blocks(pairs * H), 256, 0, STREAM>>>(proj, pairMask, bias, L, H, stride, transposed);
  return bias;
}

// pair bias [H, L, L] = LN(pair) W (+ 1e9 (pairMask - 1) for the MSA rows)
__global__ void pairBiasK(const float* proj, const float* pairMask, float* bias, int L, int H) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * H) return;
  int h = (int)(t % H); size_t ij = t / H;
  ptx_st_global_f32(bias + (size_t)h * L * L + ij,
                    ptx_add_f32(ptx_ldg_f32(proj + t), (pairMask ? ptx_mul_f32(1e9f, ptx_sub_f32(ptx_ldg_f32(pairMask + ij), 1.f)) : 0.f)));
}

// the bias from pair rows [r0, r0 + cnt) (biasFromProjK over a chunk)
__global__ void biasFromProjRowsK(const float* proj, half* out, int L, int H, int stride, bool transposed, size_t r0,
                                  size_t cnt, const float* pairMask = nullptr, bool headMajor = false) {   // proj [cnt][H], or [H][cnt]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * H) return;
  int h = (int)(t % H); size_t ij = r0 + t / H; size_t i = ij / L, j = ij % L;
  float v = ptx_add_f32(ptx_ldg_f32(proj + (headMajor ? (size_t)h * cnt + t / H : t)),
                        (pairMask ? ptx_mul_f32(1e9f, ptx_sub_f32(ptx_ldg_f32(pairMask + ij), 1.f)) : 0.f));
  if (transposed) { size_t x = i; i = j; j = x; }
  ptx_st_global_u16(out + ((size_t)h * L + i) * stride + j, ptx_f32_to_f16_raw(ptx_max_f32(ptx_mul_f32(v, LOG2E), -6e4f)));
  if (j == (size_t)L - 1) for (int pad = L; pad < stride; ++pad) ptx_st_global_u16(out + ((size_t)h * L + i) * stride + pad, 0);
}
// [C][H] f32 -> [C][16] f16, zero columns past H (gridInK's bias projection reads 16)
__global__ void padHeadsK(const float* w, half* out, int C, int H) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= C * 16) return;
  int k = t / 16, c = t % 16;
  ptx_st_global_u16(out + t, c < H ? ptx_f32_to_f16_raw(ptx_ldg_f32(w + k * H + c)) : 0);
}
inline const half* paddedFeat2dWeight(const float* wf, int C, int H) {
  static std::map<const float*, half*> wbCache;
  auto it = wbCache.find(wf);
  if (it == wbCache.end()) {
    half* h = wpool<half>((size_t)C * 16);
    padHeadsK<<<blocks((size_t)C * 16), 256, 0, STREAM>>>(wf, h, C, H);
    it = wbCache.emplace(wf, h).first;
  }
  return it->second;
}
inline void msaRowAttention(Trunk& t, const std::string& S, int blk, float* msa, int rowsN, int C, int H, int D,
                            const float* msaMask) {
  int L = t.L; size_t pairs = (size_t)L * L, rows = (size_t)rowsN * L;
  std::string R = S + "msa_row_attention_with_pair_bias";
  if (FAST) {
    // the pair bias straight from the pair: cuda/af3's lnHeadsK (LN + the projection to 16 columns, H of
    // them live) in chunks, then laid out with the pair mask - no LN'd copy of the pair, no GEMM
    const half* bias;
    {
      if (H > 16) { fprintf(stderr, "msa row attention: %d heads, the fused pair bias takes 16\n", H); exit(1); }
      int stride = (L + 7) / 8 * 8;
      const half* wb = paddedFeat2dWeight(P(R + "/feat_2d_weights", blk), 128, H);
      half* b = scratch<half>("fbias.bias", (size_t)H * L * stride);
      size_t per = std::max<size_t>(1, std::min(pairs, AF2_CHUNK / 16));
      const float *sc = P(R + "/feat_2d_norm/scale", blk), *of = P(R + "/feat_2d_norm/offset", blk);
      for (size_t r0 = 0; r0 < pairs; r0 += per) {
        size_t r = std::min(per, pairs - r0);
        lnHeadsRaw<16>(pairAdvance(t.pair, r0 * 128), sc, of, wb, nullptr, r,
                       b, L, H, stride, false, r0, t.pairOnes ? nullptr : t.pairMask);
      }
      bias = b;
    }
    if (MSA16 && msa == t.msa16) {
      withMsa16([&](auto tag) {
      using B16 = decltype(tag);
      constexpr bool BF = std::is_same_v<B16, __nv_bfloat16>;
      B16* m16 = reinterpret_cast<B16*>(msa);
      AttnW w = attnWeights(R + "/attention", blk, C);
      int Wp = w.H * w.Dp, stride = (L + 7) / 8 * 8;
      half* xn = scratch<half>("frow.xn", rows * C);
      if (takeEmittedLn(R + "/query_norm", blk, msa, rows) != xn)
        layerNormVK<256, B16, half><<<lnVGrid(rows), 256, 0, STREAM>>>(m16, xn, rows, P(R + "/query_norm/scale", blk),
                                                                                  P(R + "/query_norm/offset", blk));
      half* qkvg = scratch<half>("fatt.qkvg", (rows + 128) * 4 * Wp);
      ltGemm(xn, w.qkvg, qkvg, true, rows, C, 4 * Wp, w.qkvgBias, false, 0.f);
      const std::string ln = S + "msa_column_attention/query_norm";
      if (!t.msaOnes) {
        half* oh = scratch<half>("fatt.o", rows * Wp);
        flashGrid<half>(qkvg, bias, stride, msaMask, oh, L, w.H, w.Dp, 0, rowsN, false, 1.f / sqrtf((float)w.D));
        if (BF && C == oln::C && Wp == oln::C && fitsSmem(oln::SMEM)) {
          half* xn2 = scratch<half>("fcol.xn", rows * C);
          outLn256<B16, half, half>(oh, w.out, P(R + "/attention/output_b", blk), m16,
                                    P(ln + "/scale", blk), P(ln + "/offset", blk), xn2, rows);
          EMITTED_LN = EmittedLn{ ln, blk, msa, xn2, rows };
          return;
        }
        ltGemm(oh, msaWeight<B16>(w.out, (size_t)Wp * C), m16, !BF, rows, Wp, C, P(R + "/attention/output_b", blk), false, 1.f, 0, 0, BF);
        return;
      }
      B16* o = scratch<B16>("fatt.o16", rows * Wp);
      msaFlash<false>(qkvg, bias, stride, o, L, w.H, rowsN, 1.f / sqrtf((float)w.D));
      if (BF && C == oln::C && Wp == oln::C && fitsSmem(oln::SMEM)) {
        // the output projection writes the column attention's LayerNorm'd rows as well (outln.cuh): no LN pass there
        half* xn2 = scratch<half>("fcol.xn", rows * C);
        outLn256<B16, half>(o, (const B16*)msaWeight<B16>(w.out, (size_t)Wp * C), P(R + "/attention/output_b", blk), m16,
                            P(ln + "/scale", blk), P(ln + "/offset", blk), xn2, rows);
        EMITTED_LN = EmittedLn{ ln, blk, msa, xn2, rows };
        return;
      }
      ltGemm(o, msaWeight<B16>(w.out, (size_t)Wp * C), m16, !BF, rows, Wp, C, P(R + "/attention/output_b", blk), false, 1.f, 0, 0, BF);
      });
      return;
    }
    bool ones = msa == t.msa ? t.msaOnes : t.extraOnes;
    // the extra stack's shape (64 channels, eight unpadded heads of eight): the LN and the q/k/v/gate projection in
    // one pass, and the output projection added into the MSA in one (globalattn.cuh) - where room allows all its rows
    if (C == ga::C && H == ga::H && D == ga::D) {
      const float* keyMask = ones ? nullptr : msaMask;
      AttnW w = attnWeights(R + "/attention", blk, C, flashNarrow(keyMask));
      const int Wp = w.H * w.Dp, stride = (L + 7) / 8 * 8;
      const size_t need = (rows + 128) * 4 * Wp * 2 + rows * Wp * 2;
      if (Wp == ga::H * ga::D && (!AF2_TIGHT || roomFor(need, { "fatt.qkvg", "fatt.o", "fatt.tmp" }))) {
        half* qkvg = scratch<half>("fatt.qkvg", (rows + 128) * 4 * Wp);
        lnQkvg64(msa, P(R + "/query_norm/scale", blk), P(R + "/query_norm/offset", blk), w.qkvg, w.qkvgBias, qkvg, 0, rows);
        half* o = scratch<half>("fatt.o", rows * Wp);
        flashGrid<half>(qkvg, bias, stride, keyMask, o, L, w.H, w.Dp, 0, rowsN, false, 1.f / sqrtf((float)w.D));
        projAdd64(o, w.out, P(R + "/attention/output_b", blk), msa, rows);
        return;
      }
    }
    half* xn = scratch<half>("frow.xn", rows * C);
    layerNormH(msa, xn, rows, C, R + "/query_norm", blk);
    attentionCore(xn, rowsN, L, C, R + "/attention", blk, ones ? nullptr : msaMask, bias, msa, false);
    return;
  }
  needF32Pair("af2 msa row attention (unfused)");
  float* pn = scratch<float>("row.pn", pairs * 128);
  layerNorm(t.pair, pn, pairs, 128, R + "/feat_2d_norm", blk);
  float* proj = scratch<float>("row.proj", pairs * H);
  gemm(pn, P(R + "/feat_2d_weights", blk), proj, pairs, 128, H);
  float* bias = scratch<float>("row.bias", pairs * H);
  pairBiasK<<<blocks(pairs * H), 256, 0, STREAM>>>(proj, t.pairMask, bias, L, H);
  float* xn = scratch<float>("row.xn", rows * C);
  layerNorm(msa, xn, rows, C, R + "/query_norm", blk);
  float* out = scratch<float>("row.out", rows * C);
  gatedAttention(xn, rowsN, L, C, R + "/attention", blk, H, D, msaMask, bias, out, msa);
}
inline void msaColumnAttention(Trunk& t, const std::string& S, int blk, float* msa, int rowsN, int C, int H, int D,
                               const float* msaMask) {
  int L = t.L; size_t rows = (size_t)rowsN * L;
  std::string A = S + "msa_column_attention";
  if (MSA16 && msa == t.msa16) {
    // (as attentionCoreAcross, its MSA, output and output projection bf16 - or f16)
    withMsa16([&](auto tag) {
    using B16 = decltype(tag);
    constexpr bool BF = std::is_same_v<B16, __nv_bfloat16>;
    B16* m16 = reinterpret_cast<B16*>(msa);
    AttnW w = attnWeights(A + "/attention", blk, C);
    int Wp = w.H * w.Dp, stride = (rowsN + 7) / 8 * 8;
    half* xn = scratch<half>("fcol.xn", rows * C);
    if (takeEmittedLn(A + "/query_norm", blk, msa, rows) != xn)
      layerNormVK<256, B16, half><<<lnVGrid(rows), 256, 0, STREAM>>>(m16, xn, rows, P(A + "/query_norm/scale", blk),
                                                                                P(A + "/query_norm/offset", blk));
    const std::string ln = S + "msa_transition/input_layer_norm";
    if (L > 262 && rowsN == 1 && t.msaOnes && BF && C == oln::C && Wp == oln::C && fitsSmem(oln::SMEM) && (!AF2_TIGHT || roomFor(rows * C * 2, { "ftr.xnfull" }))) {
      half* vg = scratch<half>("fatt.qkvg", (rows + 128) * 2 * Wp);
      ltGemm(xn, attnVgWeight(w, C), vg, true, rows, C, 2 * Wp, w.qkvgBias + 2 * Wp, false, 0.f);
      B16* xt = scratch<B16>("ftr.xnfull", rows * C);
      outLn256<B16, B16, B16, true>((const B16*)vg, (const B16*)msaWeight<B16>(w.out, (size_t)Wp * C),
                                    P(A + "/attention/output_b", blk), m16, P(ln + "/scale", blk), P(ln + "/offset", blk), xt, rows);
      EMITTED_LN = EmittedLn{ ln, blk, msa, xt, rows };
      return;
    }
    half* qkvg = scratch<half>("fatt.qkvg", (rows + 128) * 4 * Wp);
    ltGemm(xn, w.qkvg, qkvg, true, rows, C, 4 * Wp, w.qkvgBias, false, 0.f);
    if (!t.msaOnes) {
      float* m = scratch<float>(AF2_TIGHT ? "col.mask" : "mask.colMsa", rows);
      if (AF2_TIGHT || blk == 0) swap01(msaMask, m, rowsN, L, 1);
      const half* zbias = zeroBias(w.H, rowsN, stride);
      half* oh = scratch<half>("fatt.o", rows * Wp);
      flashGridStrided(qkvg, zbias, stride, m, oh, rowsN, w.H, w.Dp, L, 1.f / sqrtf((float)w.D),
                       (size_t)4 * Wp, (size_t)L * 4 * Wp, (size_t)Wp, (size_t)L * Wp);
      if (BF && C == oln::C && Wp == oln::C && fitsSmem(oln::SMEM) && (!AF2_TIGHT || roomFor(rows * C * 2, { "ftr.xnfull" }))) {
        B16* xt = scratch<B16>("ftr.xnfull", rows * C);
        outLn256<B16, B16, half>(oh, w.out, P(A + "/attention/output_b", blk), m16,
                                 P(ln + "/scale", blk), P(ln + "/offset", blk), xt, rows);
        EMITTED_LN = EmittedLn{ ln, blk, msa, xt, rows };
        return;
      }
      ltGemm(oh, msaWeight<B16>(w.out, (size_t)Wp * C), m16, !BF, rows, Wp, C, P(A + "/attention/output_b", blk), false, 1.f, 0, 0, BF);
      return;
    }
    B16* o = scratch<B16>("fatt.o16", rows * Wp);
    msaFlash<true>(qkvg, nullptr, stride, o, rowsN, w.H, L, 1.f / sqrtf((float)w.D),
                                                     (size_t)4 * Wp, (size_t)L * 4 * Wp, (size_t)Wp, (size_t)L * Wp);
    // ...and its output projection writes the MSA transition's LayerNorm'd rows (outln.cuh), where room allows them whole
    if (BF && C == oln::C && Wp == oln::C && fitsSmem(oln::SMEM) && (!AF2_TIGHT || roomFor(rows * C * 2, { "ftr.xnfull" }))) {
      B16* xt = scratch<B16>("ftr.xnfull", rows * C);
      outLn256<B16, B16>(o, (const B16*)msaWeight<B16>(w.out, (size_t)Wp * C), P(A + "/attention/output_b", blk), m16,
                         P(ln + "/scale", blk), P(ln + "/offset", blk), xt, rows);
      EMITTED_LN = EmittedLn{ ln, blk, msa, xt, rows };
      return;
    }
    ltGemm(o, msaWeight<B16>(w.out, (size_t)Wp * C), m16, !BF, rows, Wp, C, P(A + "/attention/output_b", blk), false, 1.f, 0, 0, BF);
    });
    return;
  }
  if (FAST) {
    const float* mt = nullptr;
    if (!t.msaOnes) {
      float* m = scratch<float>(AF2_TIGHT ? "col.mask" : "mask.colMsa", rows);
      if (AF2_TIGHT || blk == 0) swap01(msaMask, m, rowsN, L, 1);
      mt = m;
    }
    half* xn = scratch<half>("fcol.xn", rows * C);
    layerNormH(msa, xn, rows, C, A + "/query_norm", blk);
    attentionCoreAcross(xn, L, rowsN, C, A + "/attention", blk, nullptr, msa, mt);
    return;
  }
  float* tr = scratch<float>("col.tr", rows * C);
  float* mt = scratch<float>("col.mask", rows);
  swap01(msa, tr, rowsN, L, C);                 // [L, N, C]
  swap01(msaMask, mt, rowsN, L, 1);
  float* xn = scratch<float>("col.xn", rows * C);
  layerNorm(tr, xn, rows, C, A + "/query_norm", blk);
  float* out = scratch<float>("col.out", rows * C);
  gatedAttention(xn, L, rowsN, C, A + "/attention", blk, H, D, mt, nullptr, out, msa, L, rowsN);
}
inline void msaColumnGlobalAttention(Trunk& t, const std::string& S, int blk, float* msa, int rowsN, int C,
                                     const float* msaMask) {
  int L = t.L; size_t rows = (size_t)rowsN * L;
  std::string A = S + "msa_column_global_attention";
  int H = (int)dimW(A + "/attention/query_w", 2), D = (int)dimW(A + "/attention/query_w", 3), Wd = H * D;
  // --fast at the extra stack's shape: fused, in the MSA's own layout (globalattn.cuh)
  if (FAST && C == ga::C && H == ga::H && D == ga::D) { globalAttentionFused(msa, msaMask, L, rowsN, A, blk); return; }
  float* tr = scratch<float>("gcol.tr", rows * C); float* xn = scratch<float>("gcol.xn", rows * C);
  float* mt = scratch<float>("gcol.mask", rows);
  swap01(msa, tr, rowsN, L, C);
  swap01(msaMask, mt, rowsN, L, 1);
  layerNorm(tr, xn, rows, C, A + "/query_norm", blk);
  float* avg = scratch<float>("gcol.avg", (size_t)L * Wd);
  size_t smem = (size_t)(C + Wd + H * rowsN + 2 * rowsN * D) * 4;
  if (fitsSmem(smem) || H != 8 || D != 8) {
    static bool attr = false;
    if (!attr) { smemAttr((globalAttentionK), std::min<size_t>(160 * 1024, smemLimit())); attr = true; }
    if (!fitsSmem(smem) || smem > 160 * 1024) { fprintf(stderr, "global attention: %d sequences do not fit one block\n", rowsN); exit(1); }
    globalAttentionK<<<L, 256, smem, STREAM>>>(xn, mt, P(A + "/attention/query_w", blk), P(A + "/attention/key_w", blk),
                                               P(A + "/attention/value_w", blk), avg, rowsN, C, H, D);
  } else {     // a device with less shared memory (a T4): the streamed form
    globalAttentionStreamK<8, 8><<<L, 256, (size_t)(C + Wd + 8 * H * (D + 2)) * 4, STREAM>>>(
      xn, mt, P(A + "/attention/query_w", blk), P(A + "/attention/key_w", blk), P(A + "/attention/value_w", blk), avg, rowsN, C);
  }
  float* gate = scratch<float>("gcol.gate", rows * Wd);
  gemm(xn, P(A + "/attention/gating_w", blk), gate, rows, C, Wd);
  float* gated = scratch<float>("gcol.gated", rows * Wd);
  globalGateK<<<blocks((Wd & 3) == 0 ? rows * (size_t)(Wd >> 2) : rows * Wd), 256, 0, STREAM>>>(avg, gate, gated, L, rowsN, Wd, P(A + "/attention/gating_b", blk));
  float* out = scratch<float>("gcol.out", rows * C);
  gemm(gated, P(A + "/attention/output_w", blk), out, rows, Wd, C);
  addBiasSwapInto(msa, out, P(A + "/attention/output_b", blk), rowsN, L, C, true);
}
inline void transition(float* x, size_t rows, int C, const std::string& T, int blk) {
  int I = (int)dimW(T + "/transition1/weights", blk < 0 ? 1 : 2);     // 4C in the stacks, 2C in the template's
  // at 128 channels (the pair stacks) cuda/af3's fused transition, its ReLU-with-bias form: the LayerNorm'd
  // rows and the widened ones never written
  if (FAST && fusedTransitionRaw<true>(x, rows, C, I, P(T + "/input_layer_norm/scale", blk), P(T + "/input_layer_norm/offset", blk),
                                       PH(T + "/transition1/weights", blk), PH(T + "/transition2/weights", blk),
                                       P(T + "/transition1/bias", blk), P(T + "/transition2/bias", blk)))
    return;
  needF32Pair("af2 transition (unfused)");     // (PAIR16 is set only around the pair's own updates)
  if (FAST) {
    // in row chunks of ~128 MB of the widened rows (the whole widened tensor was 1.26 GB of a pair track
    // at 783 residues; a chunk of 2^15+ rows keeps the GEMMs as fast)
    // (at 256 channels - the MSA stacks - the LayerNorm and the widening as one kernel was measured and is
    // slower: cuda/af3's transitionUpK in a ReLU-with-bias form, writing these augmented rows, 182.8 ms
    // over 5CAJ's 576 calls against 117.8 for the LN plus this GEMM it replaced, and 181.6 at two tiles a
    // warp. At 256 channels its m16-a-warp MMAs read each weight fragment for one MMA; cuBLAS does better.)
    size_t chunk = std::min(rows, std::max<size_t>(32768, (((size_t)128 << 20) / (2 * (size_t)(I + 8))) & ~(size_t)127));
    if (!AF2_TIGHT && rows <= 135168) chunk = rows;
    // (on sm_12x the MSA16 stack's MSA transition in one kernel, cuda/af3's fusedTransitionSplit256: 2.41 -> 2.00 ms a
    // call at 1,000 residues x 508 rows on an RTX PRO 6000; slower on an A100, which keeps the two GEMMs below)
    if (MSA16 && !MSA16_HALF && C == 256 && I == 4 * C && ccMajor() == 12 &&
        fusedTransitionSplit256(x, rows, I, P(T + "/input_layer_norm/scale", blk), P(T + "/input_layer_norm/offset", blk),
                                PH(T + "/transition1/weights", blk), PH(T + "/transition2/weights", blk),
                                P(T + "/transition1/bias", blk), P(T + "/transition2/bias", blk)))
      return;
    if (MSA16 && C == 256 && I == 4 * C) {     // (the MSA16 stack's MSA: every GEMM bf16 - or f16 - the residual in place)
      withMsa16([&](auto tag) {
      using B16 = decltype(tag);
      constexpr bool BF = std::is_same_v<B16, __nv_bfloat16>;
      B16* x16 = reinterpret_cast<B16*>(x);
      B16* xh = scratch<B16>("ftr.xn16", chunk * C);
      const size_t blocks128 = ((C + 127) / 128) * ((chunk + 127) / 128);
      if (((chunk >= 96 && blocks128 >= 72) || chunk <= 128) && !getenv("AF2_NO_PTX_GEMM")) {
        B16* mh = scratch<B16>("ftr.mid16" + std::to_string(I), chunk * I);
        const void* w1 = msaWeight<B16>(PH(T + "/transition1/weights", blk), (size_t)C * I);
        const void* w2 = msaWeight<B16>(PH(T + "/transition2/weights", blk), (size_t)I * C);
        const B16* emitted = (const B16*)takeEmittedLn(T + "/input_layer_norm", blk, x, rows);
        for (size_t r0 = 0; r0 < rows; r0 += chunk) {
          size_t r = std::min(chunk, rows - r0);
          const B16* xin = emitted ? emitted + r0 * C : xh;
          if (!emitted)
            layerNormVK<256, B16, B16><<<lnVGrid(r), 256, 0, STREAM>>>(x16 + r0 * C, xh, r, P(T + "/input_layer_norm/scale", blk),
                                                                                  P(T + "/input_layer_norm/offset", blk));
          ltGemm(xin, w1, mh, !BF, r, C, I, P(T + "/transition1/bias", blk), true, 0.f, 0, I, BF);
          ltGemm(mh, w2, x16 + r0 * C, !BF, r, I, C, P(T + "/transition2/bias", blk), false, 1.f, 0, 0, BF);
        }
        return;
      }
      B16* mh;
      if constexpr (BF) mh = augmentedInputBf16("ftr.mid16" + std::to_string(I), chunk, I);
      else mh = augmentedInput("ftr.mid16h" + std::to_string(I), chunk, I);
      const void* w1 = msaWeight<B16>(PH(T + "/transition1/weights", blk), (size_t)C * I);
      const void* w2 = msaWeight<B16>(augmentedWeight(T + "/transition2/weights", blk, P(T + "/transition2/bias", blk), I, C), (size_t)(I + 8) * C);
      const B16* emitted = (const B16*)takeEmittedLn(T + "/input_layer_norm", blk, x, rows);
      for (size_t r0 = 0; r0 < rows; r0 += chunk) {
        size_t r = std::min(chunk, rows - r0);
        const B16* xin = emitted ? emitted + r0 * C : xh;
        if (!emitted)
          layerNormVK<256, B16, B16><<<lnVGrid(r), 256, 0, STREAM>>>(x16 + r0 * C, xh, r, P(T + "/input_layer_norm/scale", blk),
                                                                                P(T + "/input_layer_norm/offset", blk));
        ltGemm(xin, w1, mh, !BF, r, C, I, P(T + "/transition1/bias", blk), true, 0.f, 0, I + 8, BF);
        ltGemm(mh, w2, x16 + r0 * C, !BF, r, I + 8, C, nullptr, false, 1.f, 0, 0, BF);
      }
      });
      return;
    }
    half* xh = scratch<half>("ftr.xn", chunk * C);
    half* mh = augmentedInput("ftr.mid" + std::to_string(I), chunk, I);     // [rows, I+8], a 1 at column I
    const half* w2 = augmentedWeight(T + "/transition2/weights", blk, P(T + "/transition2/bias", blk), I, C);
    for (size_t r0 = 0; r0 < rows; r0 += chunk) {
      size_t r = std::min(chunk, rows - r0);
      layerNormH(x + r0 * C, xh, r, C, T + "/input_layer_norm", blk);
      ltGemm(xh, PH(T + "/transition1/weights", blk), mh, true, r, C, I, P(T + "/transition1/bias", blk), true, 0.f, 0, I + 8);
      // the second layer's bias carried by the product (its epilogue with a residual add ran as a second
      // kernel over the whole MSA): the 1 at column I picks up the bias row of the augmented weight
      ltGemm(mh, w2, x + r0 * C, false, r, I + 8, C, nullptr, false, 1.f);
    }
    return;
  }
  float* xn = scratch<float>("tr.xn", rows * C);
  float* mid = scratch<float>("tr.mid", rows * I);
  float* out = scratch<float>("tr.out", rows * C);
  layerNorm(x, xn, rows, C, T + "/input_layer_norm", blk);
  linearB(xn, T + "/transition1", blk, mid, rows, C, I, true);
  gemm(mid, P(T + "/transition2/weights", blk), out, rows, I, C);
  addBiasSwapInto(x, out, P(T + "/transition2/bias", blk), (int)rows, 1, C, false);
}
// lt [s][i][c] -> [i][c][s], the left operand of the shallow form's contraction
__global__ void opmLeftToICS(const half* lt, half* out, int S, int L, int O, float scale = 1.f) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * L * O) return;
  int s = (int)(t % S); size_t r = t / S; int c = (int)(r % O); int i = (int)(r / O);
  unsigned short h = ptx_ldg_u16(lt + ((size_t)s * L + i) * O + c);
  ptx_st_global_u16(out + t, scale == 1.f ? h : ptx_f32_to_f16_raw(ptx_mul_f32(ptx_f16_raw_to_f32(h), scale)));
}
__global__ void opmLeftToICSF32K(const float* __restrict__ lt, float* __restrict__ out, int S, int L, int O) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * L * O) return;
  int s = (int)(t % S); size_t r = t / S; int c = (int)(r % O); int i = (int)(r / O);
  ptx_st_global_f32(out + t, ptx_ldg_f32(lt + ((size_t)s * L + i) * O + c));
}
// (the bias-augmented shallow OPM, sm_12x) Lt [L][K] -> [L][K+8], ones at columns K and K+1 and zeros after; and the GEMM's
// weight grown by 8 rows - the tiled bias as two f16 rows at K and K+1, zeros after - so the bias rides the GEMM itself
__global__ void opmAugLeftK(const half* Lt, half* La, int L, int K) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * (K + 8)) return;
  size_t j = t / (K + 8); int k = (int)(t % (K + 8));
  ptx_st_global_u16(La + t, k < K ? ptx_ldg_u16(Lt + j * K + k) : ptx_f32_to_f16_raw(k == K || k == K + 1 ? 1.f : 0.f));
}
// (two rows: the bias's f16 rounding and what that rounding left, so their f32 sum is the bias to ~2^-22)
__global__ void opmAugBiasRowsK(const float* bt, half* tail, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= 8 * n) return;
  if (t < n) ptx_st_global_u16(tail + t, ptx_f32_to_f16_raw(ptx_ldg_f32(bt + t)));
  else if (t < 2 * n) {
    float b = ptx_ldg_f32(bt + t - n);
    ptx_st_global_u16(tail + t, ptx_f32_to_f16_raw(ptx_sub_f32(b, ptx_f16_raw_to_f32(ptx_f32_to_f16_raw(b)))));
  }
  else ptx_st_global_u16(tail + t, (unsigned short)0);
}
// the output bias over a pair row's L positions, scaled: [L][C] of bias[c] * scale
__global__ void tileBiasK(const float* bias, float* out, int L, int C, float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)L * C) ptx_st_global_f32(out + t, ptx_mul_f32(ptx_ldg_f32(bias + t % C), scale));
}
__global__ void opmLeftToICSBf16K(const half* __restrict__ lt, __nv_bfloat16* __restrict__ La, int S, int L, int O, float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int K = S * O, Ka = 4 * K + 8;
  size_t total = (size_t)L * (K + 2);
  if (t >= total) return;
  int idx = (int)(t % (K + 2)), i = (int)(t / (K + 2));
  __nv_bfloat16* row = La + (size_t)i * Ka;
  if (idx < K) {
    int s = idx % S, c = idx / S;
    float v = ptx_f16_raw_to_f32(ptx_f32_to_f16_raw(ptx_mul_f32(ptx_f16_raw_to_f32(ptx_ldg_u16(lt + ((size_t)s * L + i) * O + c)), scale)));
    unsigned short l0 = ptx_f32_to_bf16_raw(v);
    float l1 = ptx_sub_f32(v, ptx_bf16_raw_to_f32(l0));
    uint32_t u = ptx_cvt_bf16x2_f32(ptx_bf16_raw_to_f32(l0), l1);
    ptx_st_global_v2_u32(row + 4 * idx, make_uint2(u, u));
  } else if (idx == K) {
    uint32_t p0 = ptx_cvt_bf16x2_f32(1.f, 1.f), p1 = ptx_cvt_bf16x2_f32(1.f, 0.f);
    ptx_st_global_v2_u32(row + 4 * K, make_uint2(p0, p1));
  } else {
    ptx_st_global_v2_u32(row + 4 * K + 4, make_uint2(0, 0));
  }
}
__global__ void opmPackTbfK(const half* __restrict__ T, const float* __restrict__ ob, float scale,
                            __nv_bfloat16* __restrict__ Tbf, int K, size_t M) {
  size_t m4 = M >> 2;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)(K + 1) * m4) return;
  size_t col4 = t % m4;
  int k = (int)(t / m4);
  size_t m = col4 << 2;
  if (k < K) {
    uint2 h4 = ptx_ldg_v2_u32(T + (size_t)k * M + m);
    float2 f0 = ptx_f16x2_to_f32x2(h4.x);
    float2 f1 = ptx_f16x2_to_f32x2(h4.y);
    uint32_t hi0 = ptx_cvt_bf16x2_f32(f0.x, f0.y), hi1 = ptx_cvt_bf16x2_f32(f1.x, f1.y);
    float2 hi0f = ptx_bf16x2_to_f32x2(hi0), hi1f = ptx_bf16x2_to_f32x2(hi1);
    uint32_t lo0 = ptx_cvt_bf16x2_f32(ptx_sub_f32(f0.x, hi0f.x), ptx_sub_f32(f0.y, hi0f.y));
    uint32_t lo1 = ptx_cvt_bf16x2_f32(ptx_sub_f32(f1.x, hi1f.x), ptx_sub_f32(f1.y, hi1f.y));
    uint2 uHi = make_uint2(hi0, hi1);
    uint2 uLo = make_uint2(lo0, lo1);
    size_t r0 = (size_t)(4 * k) * m4 + col4;
    ptx_st_global_v2_u32(reinterpret_cast<uint2*>(Tbf) + r0, uHi);
    ptx_st_global_v2_u32(reinterpret_cast<uint2*>(Tbf) + r0 + m4, uHi);
    ptx_st_global_v2_u32(reinterpret_cast<uint2*>(Tbf) + r0 + 2 * m4, uLo);
    ptx_st_global_v2_u32(reinterpret_cast<uint2*>(Tbf) + r0 + 3 * m4, uLo);
  } else {
    float4 b4 = ptx_ldg_v4_f32(ob + (m & 127));
    b4.x = ptx_mul_f32(b4.x, scale); b4.y = ptx_mul_f32(b4.y, scale);
    b4.z = ptx_mul_f32(b4.z, scale); b4.w = ptx_mul_f32(b4.w, scale);
    uint32_t b0_0 = ptx_cvt_bf16x2_f32(b4.x, b4.y), b0_1 = ptx_cvt_bf16x2_f32(b4.z, b4.w);
    float2 f0_0 = ptx_bf16x2_to_f32x2(b0_0), f0_1 = ptx_bf16x2_to_f32x2(b0_1);
    float4 r1 = make_float4(ptx_sub_f32(b4.x, f0_0.x), ptx_sub_f32(b4.y, f0_0.y), ptx_sub_f32(b4.z, f0_1.x), ptx_sub_f32(b4.w, f0_1.y));
    uint32_t b1_0 = ptx_cvt_bf16x2_f32(r1.x, r1.y), b1_1 = ptx_cvt_bf16x2_f32(r1.z, r1.w);
    float2 f1_0 = ptx_bf16x2_to_f32x2(b1_0), f1_1 = ptx_bf16x2_to_f32x2(b1_1);
    uint32_t b2_0 = ptx_cvt_bf16x2_f32(ptx_sub_f32(r1.x, f1_0.x), ptx_sub_f32(r1.y, f1_0.y));
    uint32_t b2_1 = ptx_cvt_bf16x2_f32(ptx_sub_f32(r1.z, f1_1.x), ptx_sub_f32(r1.w, f1_1.y));
    size_t rb = (size_t)(4 * K) * m4 + col4;
    ptx_st_global_v2_u32(reinterpret_cast<uint2*>(Tbf) + rb, make_uint2(b0_0, b0_1));
    ptx_st_global_v2_u32(reinterpret_cast<uint2*>(Tbf) + rb + m4, make_uint2(b1_0, b1_1));
    ptx_st_global_v2_u32(reinterpret_cast<uint2*>(Tbf) + rb + 2 * m4, make_uint2(b2_0, b2_1));
    uint2 z = make_uint2(0, 0);
    for (int r = 3; r < 8; ++r) ptx_st_global_v2_u32(reinterpret_cast<uint2*>(Tbf) + rb + r * m4, z);
  }
}
// The outer product mean's SHALLOW form, for an alignment of few rows: the output projection folded into the
// right operand first - T[c][s][j][f] = sum_e R[s][j][e] W[c O + e][f], a batched GEMM over c (W's rows for one
// c are contiguous) - then Y[i][j][f] = sum_{c,s} L[s][i][c] T[c][s][j][f], one GEMM with K = O x S. Its work
// is n^2 S O 128 against the standard form's n^2 O^2 (S + 128): a single sequence is ~30x less, and the forms
// cross in FLOPs near S = 43 - but measured on an A100 at 500 residues, whole fold, the shallow form is faster
// to 128 rows (1991 -> 1657 ms at 4, 2071 -> 1784 at 32, 2443 -> 2317 at 128) and slower from 256 (2912 against
// 3063): the standard form's permute and its O^2-wide product are memory, not arithmetic. LOCALFOLD_OPM_SHALLOW
// sets the deepest alignment it takes (0: never).
inline const int OPM_SHALLOW = getenv("LOCALFOLD_OPM_SHALLOW") ? atoi(getenv("LOCALFOLD_OPM_SHALLOW")) : 128;
#include "../../af3/src/tricontract.cuh"   // (bf16Gemms: past 1e10 multiply-adds a channel, the one-channel pick; fp8GemmTN)
// the deep contraction's operands transposed to [L*O][Sp], the rows contiguous (the one layout an FP8 GEMM takes), e4m3,
// the rows past rowsN zero
__global__ void opmToF8K(const half* in, __nv_fp8_e4m3* out, int rowsN, int Sp, int LO) {
  __shared__ float tile[32][33];
  int c0 = blockIdx.x * 32, s0 = blockIdx.y * 32;
  for (int y = threadIdx.y; y < 32; y += 8) {
    int s = s0 + y, c = c0 + threadIdx.x;
    ptx_st_shared_f32(&tile[y][threadIdx.x], s < rowsN && c < LO ? ptx_f16_raw_to_f32(ptx_ldg_u16(in + (size_t)s * LO + c)) : 0.f);
  }
  __syncthreads();
  for (int y = threadIdx.y; y < 32; y += 8) {
    int c = c0 + y, s = s0 + threadIdx.x;
    if (c < LO && s < Sp) out[(size_t)c * Sp + s] = __nv_fp8_e4m3(ptx_ld_shared_f32(&tile[threadIdx.x][y]));
  }
}
#include "../../af3/src/triingemm.cuh"
// The deep form's output projection read from the contraction's product where it lies, [i][a][j][b] (f16, scaled by
// 2^-k): a block is 128 flat (i, j) rows of the i-block, the k = (a, b) chunks a apart (tgMain's strides), and the epilogue adds
// (bias + 2^k y) / (1e-3 + norm) into the pair - no permuted copy, no f32 product, no opmAddK
template <class PT>
__global__ void __launch_bounds__(128, 2) opmOutK(const half* __restrict__ Pm, const half* __restrict__ W, const float* __restrict__ bias,
    const float* __restrict__ norm, float* __restrict__ pair, int i0, int bi, int L, float up, float invNorm = 0.f) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, wm = warp & 1, wn = warp >> 1;
  const size_t row0 = (size_t)blockIdx.x * 128, pp = (size_t)bi * L;
  float acc[4][8][4] = {};
  tgMain<1024>(Pm, W, row0, pp, smem, acc, 32, (size_t)L * 32, (unsigned)L);
  __syncthreads();
  constexpr int LDY = 64 + 4;
  float* Ys = reinterpret_cast<float*>(smem) + warp * 16 * LDY;
  const int cl = (lane & 15) * 4, f = wn * 64 + cl;
  const float4 b4 = ptx_ldg_v4_f32(bias + f);
  const bool full = row0 + 128 <= pp;
#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
    PairRaw4<PT> pv[8];
    const size_t ij0 = (size_t)i0 * L + row0 + wm * 64 + mi * 16 + (lane >> 4);
    if (full && !norm) {
#pragma unroll
      for (int ep = 0; ep < 8; ++ep) {
        pv[ep] = pairLdRaw4<PT>(pair, (ij0 + ep * 2) * 128 + f);
      }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      int c = t * 8 + 2 * tig;
      ptx_st_shared_v2_f32(Ys + g * LDY + c, acc[mi][t][0], acc[mi][t][1]);
      ptx_st_shared_v2_f32(Ys + (g + 8) * LDY + c, acc[mi][t][2], acc[mi][t][3]);
    }
    __syncwarp();
    if (full && !norm) {
#pragma unroll
      for (int ep = 0; ep < 8; ++ep) {
        float4 a4 = ptx_ld_shared_v4_f32(Ys + (ep * 2 + (lane >> 4)) * LDY + cl);
        float4 v = pairUnpack4<PT>(pv[ep]);
        v.x = ptx_fma_f32(ptx_fma_f32(up, a4.x, b4.x), invNorm, v.x);
        v.y = ptx_fma_f32(ptx_fma_f32(up, a4.y, b4.y), invNorm, v.y);
        v.z = ptx_fma_f32(ptx_fma_f32(up, a4.z, b4.z), invNorm, v.z);
        v.w = ptx_fma_f32(ptx_fma_f32(up, a4.w, b4.w), invNorm, v.w);
        pairSt4<PT>(pair, (ij0 + ep * 2) * 128 + f, v);
      }
    } else {
#pragma unroll
      for (int ep = 0; ep < 8; ++ep) {
        int rl = ep * 2 + (lane >> 4);
        size_t q = row0 + wm * 64 + mi * 16 + rl;
        if (q >= pp) continue;
        const size_t ij = (size_t)i0 * L + q;
        const float d = norm ? ptx_rcp_f32(ptx_add_f32(1e-3f, ptx_ldg_f32(norm + ij))) : invNorm;
        float4 a4 = ptx_ld_shared_v4_f32(Ys + rl * LDY + cl);
        float4 v = pairLd4<PT>(pair, ij * 128 + f);
        v.x = ptx_fma_f32(ptx_fma_f32(up, a4.x, b4.x), d, v.x);
        v.y = ptx_fma_f32(ptx_fma_f32(up, a4.y, b4.y), d, v.y);
        v.z = ptx_fma_f32(ptx_fma_f32(up, a4.z, b4.z), d, v.z);
        v.w = ptx_fma_f32(ptx_fma_f32(up, a4.w, b4.w), d, v.w);
        pairSt4<PT>(pair, ij * 128 + f, v);
      }
    }
    if (mi < 3) __syncwarp();
  }
}
inline std::pair<const half*, const float*> opmWeights(const std::string& Op, int blk, int C, int O = 32) {
  struct OpmW { const half* wt; const float* bias; };
  static std::map<std::pair<std::string, int>, OpmW> opmCache;
  auto it = opmCache.find({Op, blk});
  if (it == opmCache.end()) {
    half* wt = wpool<half>((size_t)C * 2 * O);
    tileColumns(PH(Op + "/left_projection/weights", blk), C, O, 0, O, 16, wt);
    tileColumns(PH(Op + "/right_projection/weights", blk), C, O, 0, O, 16, wt + (size_t)C * O);
    float* bb = wpool<float>(2 * O);
    CK(cudaMemcpyAsync(bb, P(Op + "/left_projection/bias", blk), O * 4, cudaMemcpyDeviceToDevice, STREAM));
    CK(cudaMemcpyAsync(bb + O, P(Op + "/right_projection/bias", blk), O * 4, cudaMemcpyDeviceToDevice, STREAM));
    it = opmCache.emplace(std::make_pair(Op, blk), OpmW{wt, bb}).first;
  }
  return {it->second.wt, it->second.bias};
}
inline void outerProductMean(Trunk& t, const std::string& S, int blk, const float* msa, int rowsN, int C,
                             const float* msaMask) {
  int L = t.L; size_t rows = (size_t)rowsN * L; const int O = 32;
  std::string Op = S + "outer_product_mean";
  const bool isMsa = (msa == t.msa || msa == t.msa16);
  if (!isMsa && t.extraZeros) {
    WITH_PAIR_T(opmZeroMask128K<PT><<<blocks((size_t)L * L * 32), 256, 0, STREAM>>>(t.pair, P(Op + "/output_b", blk), (size_t)L * L));
    return;
  }
  if (FAST) {
    half* lt = scratch<half>("fopm.left", rows * O); half* rt = scratch<half>("fopm.right", rows * O);
    static const bool lnFused = !getenv("LOCALFOLD_LNGEMM") || atoi(getenv("LOCALFOLD_LNGEMM"));
    bool ones = isMsa ? t.msaOnes : t.extraOnes;
    if ((lnFused || (MSA16 && msa == t.msa16)) && (C == 256 || C == 64)) {
      // the LayerNorm and both projections in one pass over the MSA (lnGemm): the LN'd MSA never written, read twice -
      // 5CAJ with its 7,907-row alignment 1382 -> 1354 ms of GPU a fold, 0.014 A. (The same kernel for the row
      // attention's q/k/v/gate, 1024 columns, loses - 130 ms against the LN's 31 and cuBLAS's 80: its two-tile warps
      // reach ~100 TFLOP/s where cuBLAS reaches 167, the transition's note below found the same)
      auto ow = opmWeights(Op, blk, C, O);
      const float *s0 = P(Op + "/layer_norm_input/scale", blk), *o0 = P(Op + "/layer_norm_input/offset", blk);
      const float* msk = ones ? nullptr : msaMask;
      if (MSA16 && msa == t.msa16) {
        int nextBlk = rows > 4096 ? (t.opmFirst ? blk : (blk + 1 < (int)dimW(S + "msa_transition/transition1/weights", 0) ? blk + 1 : -1)) : -1;
        const std::string rln = S + "msa_row_attention_with_pair_bias/query_norm";
        half* xnRow = nextBlk >= 0 ? scratch<half>("frow.xn", rows * C) : nullptr;
        const float* s2 = nextBlk >= 0 ? P(rln + "/scale", nextBlk) : nullptr;
        const float* o2 = nextBlk >= 0 ? P(rln + "/offset", nextBlk) : nullptr;
        withMsa16([&](auto tag) { lnGemm<256, decltype(tag)>(msa, rows, s0, o0, ow.first, ow.second, 2 * O, O, lt, rt, msk, s2, o2, xnRow); });
        if (nextBlk >= 0) EMITTED_LN = EmittedLn{ rln, nextBlk, msa, xnRow, rows };
      }
      else if (C == 256) lnGemm<256>(msa, rows, s0, o0, ow.first, ow.second, 2 * O, O, lt, rt, msk);
      else lnGemm<64>(msa, rows, s0, o0, ow.first, ow.second, 2 * O, O, lt, rt, msk);
    } else {
      half* xn = scratch<half>("fopm.xn", rows * C);
      layerNormH(msa, xn, rows, C, Op + "/layer_norm_input", blk);
      ltGemm(xn, PH(Op + "/left_projection/weights", blk), lt, true, rows, C, O, P(Op + "/left_projection/bias", blk), false, 0.f);
      ltGemm(xn, PH(Op + "/right_projection/weights", blk), rt, true, rows, C, O, P(Op + "/right_projection/bias", blk), false, 0.f);
      if (!ones) {
        scaleRowsHK<<<blocks(rows * O), 256, 0, STREAM>>>(lt, msaMask, rows, O);
        scaleRowsHK<<<blocks(rows * O), 256, 0, STREAM>>>(rt, msaMask, rows, O);
      }
    }
    const float one = 1.f, zero = 0.f;
    float* norm = nullptr;
    if (!ones) {
      if (AF2_TIGHT) {
        norm = scratch<float>("opm.norm", (size_t)L * L);
        CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, L, L, rowsN, &one, msaMask, L, msaMask, L, &zero, norm, L));
      } else {
        norm = scratch<float>(isMsa ? "mask.normMsa" : "mask.normExtra", (size_t)L * L);
        if (blk == 0) CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, L, L, rowsN, &one, msaMask, L, msaMask, L, &zero, norm, L));
      }
    }
    if (rowsN <= OPM_SHALLOW) {
      const half* W = PH(Op + "/output_w", blk);           // [O*O][128] f16
      // (8 more rows, the bias rows' - see below - on sm_12x with an f32 pair, and wherever the bias vector is past 2^17
      // elements: on Colab's RTX PRO 6000 a bf16 pair's fold from 8 rows at 1,500 residues 7.48 -> 7.07 s with it. A shorter
      // bias keeps the epilogue, which cuBLASLt fuses for a bf16 pair's f32 staging rows: there the augmented form was
      // 0.1-0.6% slower at 600 and 1,000 residues)
      const bool aug = ones && ((ccMajor() == 12 && !PAIR16) || (size_t)L * 128 > ((size_t)1 << 17));
      half* T = scratch<half>("fopm.T", (size_t)O * rows * 128 + (aug ? (size_t)8 * L * 128 : 0));
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, 128, (int)rows, O, &one, W, CUDA_R_16F, 128, (long long)O * 128,
                                    rt, CUDA_R_16F, O, 0, &zero, T, CUDA_R_16F, 128, (long long)rows * 128, O,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      const float inv = ones ? 1.f / (1e-3f + (float)rowsN) : 1.f;
      int K = O * rowsN;
      if (ones && PAIR16 && rowsN <= 4 && (size_t)L * L * 512 > ((size_t)36 << 20)) {
        int Ka = 4 * K + 8;
        size_t M = (size_t)L * 128;
        if (!AF2_TIGHT || roomFor(((size_t)L * Ka + (size_t)Ka * M) * 2, { "fopm.La", "fopm.Y" })) {
          __nv_bfloat16* La = scratch<__nv_bfloat16>("fopm.La", (size_t)L * Ka);
          __nv_bfloat16* Tbf = scratch<__nv_bfloat16>("fopm.Y", (size_t)Ka * M);
          opmLeftToICSBf16K<<<blocks((size_t)L * (K + 2)), 256, 0, STREAM>>>(lt, La, rowsN, L, O, inv);
          opmPackTbfK<<<blocks((size_t)(K + 1) * (M >> 2)), 256, 0, STREAM>>>(T, P(Op + "/output_b", blk), inv, Tbf, K, M);
          ltGemm(La, Tbf, reinterpret_cast<__nv_bfloat16*>(t.pair), false, L, Ka, L * 128, nullptr, false, 1.f, 0, 0, true);
          return;
        }
      }
      half* Lt = scratch<half>("fopm.Lt", rows * O);
      // with every MSA row live the norm is rowsN at every pair, so (Y + b) / (1e-3 + norm) folds into the
      // GEMM: L scaled by the constant, the bias tiled and scaled, accumulated straight into the pair (no
      // f32 Y written and read back, no opmAddK: 1.06 s of a 2,088-residue fold)
      opmLeftToICS<<<blocks(rows * O), 256, 0, STREAM>>>(lt, Lt, rowsN, L, O, inv);
      if (ones) {
        int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / ((size_t)L * 128)));
        if (PAIR16) {   // (into an f32 block of rows for a single unaligned sequence, where an f16 product moved 6MRR
                        // 1.898 -> 2.001 A; f16 when templates or MSA rows are present and L >= 128)
          const float* ob = P(Op + "/output_b", blk);
          if ((rowsN > 1 || M.meta("meta/templates", 0) > 0) && L >= 128) {
            half* Yb = scratch<half>("fopm.Y", (size_t)Bi * L * 128);
            for (int i0 = 0; i0 < L; i0 += Bi) {
              int bi = std::min(Bi, L - i0);
              ltGemm(Lt + (size_t)i0 * K, T, Yb, true, bi, K, L * 128, nullptr, false, 0.f);
              addIntoPairK<__nv_bfloat16, half><<<blocks((size_t)bi * L * 16), 256, 0, STREAM>>>(
                reinterpret_cast<float*>(reinterpret_cast<__nv_bfloat16*>(t.pair) + (size_t)i0 * L * 128), Yb, (size_t)bi * L * 32 * 4, ob, inv);
            }
            return;
          }
          float* Yb = scratch<float>("fopm.Y", (size_t)Bi * L * 128);
          for (int i0 = 0; i0 < L; i0 += Bi) {
            int bi = std::min(Bi, L - i0);
            ltGemm(Lt + (size_t)i0 * K, T, Yb, false, bi, K, L * 128, nullptr, false, 0.f);
            addIntoPairK<__nv_bfloat16, float><<<blocks((size_t)bi * L * 32), 256, 0, STREAM>>>(
              reinterpret_cast<float*>(reinterpret_cast<__nv_bfloat16*>(t.pair) + (size_t)i0 * L * 128), Yb, (size_t)bi * L * 128, ob, inv);
          }
          return;
        }
        float* bt = scratch<float>("fopm.btile", (size_t)L * 128);
        tileBiasK<<<blocks((size_t)L * 128), 256, 0, STREAM>>>(P(Op + "/output_b", blk), bt, L, 128, inv);
        // aug (sm_12x): the bias as the GEMM's K-th row against a column of ones, where the epilogue's L * 128-long bias
        // vector ran there as a second kernel over every chunk's output (cuBLASLt's globalKernel - on Colab's RTX PRO 6000
        // 887 ms, 12% of a 2,500-residue pass from 8 rows); the bias two f16 rows (opmAugBiasRowsK), the product relRMS
        // 1-4e-7 from the epilogue's (sm_120 emulated on an A100)
        const half* La = Lt; int Ka = K; const float* gb = bt;
        if (aug) {
          half* la = scratch<half>("fopm.La", (size_t)L * (K + 8));
          opmAugLeftK<<<blocks((size_t)L * (K + 8)), 256, 0, STREAM>>>(Lt, la, L, K);
          opmAugBiasRowsK<<<blocks((size_t)8 * L * 128), 256, 0, STREAM>>>(bt, T + (size_t)K * L * 128, (size_t)L * 128);
          La = la; Ka = K + 8; gb = nullptr;
        }
        for (int i0 = 0; i0 < L; i0 += Bi) {
          int bi = std::min(Bi, L - i0);
          ltGemm(La + (size_t)i0 * Ka, T, t.pair + (size_t)i0 * L * 128, false, bi, Ka, L * 128, gb, false, 1.f);
        }
        return;
      }
      int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / ((size_t)L * 128)));
      if (L >= 128) {
        half* Y = scratch<half>("fopm.Y", (size_t)Bi * L * 128);
        for (int i0 = 0; i0 < L; i0 += Bi) {
          int bi = std::min(Bi, L - i0);
          ltGemm(Lt + (size_t)i0 * K, T, Y, true, bi, K, L * 128, nullptr, false, 0.f);
          WITH_PAIR_T(opmAdd128K<PT, true, half><<<blocks((size_t)bi * L * (std::is_same_v<PT, float> ? 32 : 16)), 256, 0, STREAM>>>(t.pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L));
        }
        return;
      }
      float* Y = scratch<float>("fopm.Y", (size_t)Bi * L * 128);
      for (int i0 = 0; i0 < L; i0 += Bi) {
        int bi = std::min(Bi, L - i0);
        ltGemm(Lt + (size_t)i0 * K, T, Y, false, bi, K, L * 128, nullptr, false, 0.f);
        WITH_PAIR_T(opmAdd128K<PT, true><<<blocks((size_t)bi * L * 32), 256, 0, STREAM>>>(t.pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L));
      }
      return;
    }
    size_t per = (size_t)L * O * O;
    // i-blocks of 2^26 product elements (128 MB), 2^28 where the output projection reads the product in place (no
    // permuted copy, no f32 Y beside it), the card has the room (what the product holds counted, so each pass decides
    // alike), and its L2 does not hold a 2^26 block between the contraction writing it and opmOutK reading it. A100
    // (40 MB of L2) 600 / 1,000 / 1,500 residues 4.91 / 11.95 / 23.19 -> 4.84 / 11.83 / 22.95 s at 2^28, 2^27 and 2^29
    // level or slower, 2^23-2^25 slower; RTX PRO 6000 (128 MB of L2) 3.02 / 6.93 / 46.94 s (2,500) at 2^26 against
    // 3.05 / 6.98 / 47.14 at 2^28, opmOutK 104 -> 138 ms at 600, and 2^24-2^25 level or slower
    const bool directOut = ccOf() >= 80;
    static const bool l2HoldsBlock = [] { int d, l2; CK(cudaGetDevice(&d)); CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, d)); return l2 >= (96 << 20); }();
    const size_t budget = directOut && !l2HoldsBlock && roomFor(((size_t)256 << 20) * 2, {"fopm.P"}) ? (size_t)256 << 20 : (size_t)64 << 20;
    int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, budget / per));
    // the sum over the rows scaled by 2^-k - exact, as a power of two is - so it fits float16, and opmAddK scales it back:
    // unscaled, 7GYV's alignment at 1,945 extra rows overflowed it and NaN'd the fold (ColabFold's 512:5120 asks 5,120).
    // 2^k the row count / 64: the largest entry there was ~34 a row, so ~2,200 scaled, and the small ones stay clear of
    // float16's subnormals, which scaling to the mean (2^k >= the row count) reached - 7GYV's fold moved by it
    int k = 0; while ((64 << k) < rowsN) ++k;
    const float down = ldexpf(1.f, -k), up = ldexpf(1.f, k);
    half* Pm = scratch<half>("fopm.P", (size_t)Bi * per);
    // FP8 where the card has FP8 tensor instructions (fp8Tensor): both operands e4m3 over the rows padded to 16. From
    // Ampere on the output projection reads the product in place (opmOutK). RTX PRO 6000, 1,000 residues at 508 + 5,120
    // rows: 8.00 -> 7.60 s (FP8) -> 7.05 s (in place); A100 12.91 -> 12.23 s (README, "The deep outer product mean")
    const bool f8 = fp8Tensor(), direct = directOut;
    const int Sp = (rowsN + 15) / 16 * 16, LO = L * O;
    __nv_fp8_e4m3 *l8 = nullptr, *r8 = nullptr;
    if (f8) {
      l8 = scratch<__nv_fp8_e4m3>("fopm.l8", (size_t)LO * Sp); r8 = scratch<__nv_fp8_e4m3>("fopm.r8", (size_t)LO * Sp);
      dim3 g((LO + 31) / 32, (Sp + 31) / 32), b(32, 8);
      opmToF8K<<<g, b, 0, STREAM>>>(lt, l8, rowsN, Sp, LO);
      opmToF8K<<<g, b, 0, STREAM>>>(rt, r8, rowsN, Sp, LO);
    }
    half* X = direct ? nullptr : scratch<half>("fopm.X", (size_t)Bi * per);
    float* Y = direct ? nullptr : scratch<float>("fopm.Y", (size_t)Bi * L * 128);
    for (int i0 = 0; i0 < L; i0 += Bi) {
      int bi = std::min(Bi, L - i0);
      if (f8) fp8GemmTN(LO, bi * O, Sp, down, r8, Sp, 0, l8 + (size_t)i0 * O * Sp, Sp, 0, Pm, LO, 0, 1, CUDA_R_16F);
      else CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_T, L * O, bi * O, rowsN, &down, rt, CUDA_R_16F, L * O, lt + (size_t)i0 * O,
                           CUDA_R_16F, L * O, &zero, Pm, CUDA_R_16F, L * O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      if (direct) {
        WITH_PAIR_T(opmOutK<PT><<<(unsigned)(((size_t)bi * L + 127) / 128), 128, triInGemmSmem(), STREAM>>>(
          Pm, PH(Op + "/output_w", blk), P(Op + "/output_b", blk), norm, t.pair, i0, bi, L, up, 1.f / (1e-3f + (float)rowsN)));
        continue;
      }
      opmPermuteHK<<<blocks((size_t)bi * per / 8), 256, 0, STREAM>>>(Pm, X, bi, L, O);
      ltGemm(X, PH(Op + "/output_w", blk), Y, false, (size_t)bi * L, O * O, 128, nullptr, false, 0.f);
      WITH_PAIR_T(opmAdd128K<PT><<<blocks((size_t)bi * L * 32), 256, 0, STREAM>>>(t.pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L, up, (float)rowsN));
    }
    return;
  }
  needF32Pair("af2 outer product mean (unfused)");
  float* xn = scratch<float>("opm.xn", rows * C);
  layerNorm(msa, xn, rows, C, Op + "/layer_norm_input", blk);
  float* lt = scratch<float>("opm.left", rows * O); float* rt = scratch<float>("opm.right", rows * O);
  linearB(xn, Op + "/left_projection", blk, lt, rows, C, O, false, msaMask);
  linearB(xn, Op + "/right_projection", blk, rt, rows, C, O, false, msaMask);
  float* norm = scratch<float>("opm.norm", (size_t)L * L);
  const float one = 1.f, zero = 0.f;
  // norm[i, j] = sum_s mask[s, i] mask[s, j]
  CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, L, L, rowsN, &one, msaMask, L, msaMask, L, &zero, norm, L));
  if (rowsN <= 32) {
    const float* W = P(Op + "/output_w", blk);
    float* T = scratch<float>("opm.T", (size_t)O * rows * 128);
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, 128, (int)rows, O, &one, W, 128, (long long)O * 128,
                                 rt, O, 0, &zero, T, 128, (long long)rows * 128, O));
    float* Lt = scratch<float>("opm.Lt", rows * O);
    opmLeftToICSF32K<<<blocks(rows * O), 256, 0, STREAM>>>(lt, Lt, rowsN, L, O);
    int K = O * rowsN;
    int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / ((size_t)L * 128)));
    float* Y = scratch<float>("opm.Y", (size_t)Bi * L * 128);
    for (int i0 = 0; i0 < L; i0 += Bi) {
      int bi = std::min(Bi, L - i0);
      gemm(Lt + (size_t)i0 * K, T, Y, bi, K, L * 128);
      opmAdd128K<float><<<blocks((size_t)bi * L * 32), 256, 0, STREAM>>>(t.pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L);
    }
    return;
  }
  size_t per = (size_t)L * O * O;
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / per));
  float* Pm = scratch<float>("opm.P", (size_t)Bi * per);
  float* X = scratch<float>("opm.X", (size_t)Bi * per);
  float* Y = scratch<float>("opm.Y", (size_t)Bi * L * 128);
  for (int i0 = 0; i0 < L; i0 += Bi) {
    int bi = std::min(Bi, L - i0);
    // row-major P (bi*O x L*O) = left_blk^T right, as col-major P^T = right * left_blk^T
    CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, L * O, bi * O, rowsN, &one, rt, L * O, lt + (size_t)i0 * O, L * O,
                   &zero, Pm, L * O));
    if (O == 32) opmPermute32K<<<(unsigned)((size_t)bi * ((L + 3) >> 2)), 256, 0, STREAM>>>(Pm, X, bi, L);
    else opmPermuteK<float><<<blocks((O & 3) == 0 ? (size_t)bi * per / 4 : (size_t)bi * per), 256, 0, STREAM>>>(Pm, X, bi, L, O);
    gemm(X, P(Op + "/output_w", blk), Y, (size_t)bi * L, O * O, 128);
    opmAdd128K<float><<<blocks((size_t)bi * L * 32), 256, 0, STREAM>>>(t.pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L);
  }
}
__global__ void packBatched2K(const float* __restrict__ w0, const float* __restrict__ w1,
                              float* __restrict__ out, int nblk, size_t slice) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)nblk * slice) return;
  int b = (int)(t / slice); size_t s = t % slice;
  float* o = out + (size_t)b * 2 * slice;
  ptx_st_global_f32(o + s, ptx_ldg_f32(w0 + t));
  ptx_st_global_f32(o + slice + s, ptx_ldg_f32(w1 + t));
}
// a, b [c][i][k] (channel-major) from the projection's two halves, masked and gated
__global__ void __launch_bounds__(256, 8)
triSplitK(const float* __restrict__ proj, const float* __restrict__ gate, const float* __restrict__ mask,
          float* __restrict__ a, float* __restrict__ b, int L, int C,
          const float* __restrict__ projB = nullptr, const float* __restrict__ gateB = nullptr, int ld = 0) {
  __shared__ float A[32][33];
  __shared__ float B[32][33];
  if (!ld) ld = 2 * C;
  size_t pairs = (size_t)L * L;
  size_t ij0 = (size_t)blockIdx.x * 32;
  int c0 = (int)blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
  int c = c0 + tx;
  if (ij0 + 32 <= pairs && c0 + 32 <= C) {
    float pba = projB ? ptx_ldg_f32(projB + c) : 0.f, pbb = projB ? ptx_ldg_f32(projB + C + c) : 0.f;
    float gba = gateB ? ptx_ldg_f32(gateB + c) : 0.f, gbb = gateB ? ptx_ldg_f32(gateB + C + c) : 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      size_t ij = ij0 + ty + k * 8;
      size_t base = ij * (size_t)ld + c;
      float m = ptx_ldg_f32(mask + ij);
      float pa = ptx_add_f32(ptx_ldg_f32(proj + base), pba), ga = ptx_add_f32(ptx_ldg_f32(gate + base), gba);
      float pb = ptx_add_f32(ptx_ldg_f32(proj + base + C), pbb), gb = ptx_add_f32(ptx_ldg_f32(gate + base + C), gbb);
      ptx_st_shared_f32(&A[ty + k * 8][tx], ptx_mul_f32(ptx_mul_f32(pa, m), ptx_sigm_f32(ga)));
      ptx_st_shared_f32(&B[ty + k * 8][tx], ptx_mul_f32(ptx_mul_f32(pb, m), ptx_sigm_f32(gb)));
    }
    __syncthreads();
    size_t ij = ij0 + tx;
#pragma unroll
    for (int k = 0; k < 32; k += 8) {
      int co = c0 + ty + k;
      ptx_st_global_f32(a + (size_t)co * pairs + ij, ptx_ld_shared_f32(&A[tx][ty + k]));
      ptx_st_global_f32(b + (size_t)co * pairs + ij, ptx_ld_shared_f32(&B[tx][ty + k]));
    }
    return;
  }
  float pba = (c < C && projB) ? ptx_ldg_f32(projB + c) : 0.f, pbb = (c < C && projB) ? ptx_ldg_f32(projB + C + c) : 0.f;
  float gba = (c < C && gateB) ? ptx_ldg_f32(gateB + c) : 0.f, gbb = (c < C && gateB) ? ptx_ldg_f32(gateB + C + c) : 0.f;
#pragma unroll
  for (int k = 0; k < 32; k += 8) {
    size_t ij = ij0 + ty + k;
    if (ij < pairs && c < C) {
      float m = ptx_ldg_f32(mask + ij);
      size_t base = ij * (size_t)ld + c;
      ptx_st_shared_f32(&A[ty + k][tx], ptx_mul_f32(ptx_mul_f32(ptx_add_f32(ptx_ldg_f32(proj + base), pba), m), ptx_sigm_f32(ptx_add_f32(ptx_ldg_f32(gate + base), gba))));
      ptx_st_shared_f32(&B[ty + k][tx], ptx_mul_f32(ptx_mul_f32(ptx_add_f32(ptx_ldg_f32(proj + base + C), pbb), m), ptx_sigm_f32(ptx_add_f32(ptx_ldg_f32(gate + base + C), gbb))));
    }
  }
  __syncthreads();
  size_t ij = ij0 + tx;
#pragma unroll
  for (int k = 0; k < 32; k += 8) {
    int co = c0 + ty + k;
    if (co < C && ij < pairs) {
      ptx_st_global_f32(a + (size_t)co * pairs + ij, ptx_ld_shared_f32(&A[tx][ty + k]));
      ptx_st_global_f32(b + (size_t)co * pairs + ij, ptx_ld_shared_f32(&B[tx][ty + k]));
    }
  }
}
#include "../../af3/src/triblocked.cuh"
// one operand's [projection | gate] (C, 2C) f16 and its 2C bias, side 0 = a (the first C columns), 1 = b
__global__ void operand2K(const float* proj, const float* gate, const float* pb, const float* gb, half* w, float* bias,
                          int C, int side) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)C * 2 * C) {
    int k = (int)(t / (2 * C)), o = (int)(t % (2 * C));
    ptx_st_global_b16(w + t, ptx_f32_to_f16_raw(o < C ? ptx_ldg_f32(proj + (size_t)k * 2 * C + side * C + o)
                                                      : ptx_ldg_f32(gate + (size_t)k * 2 * C + side * C + o - C)));
  }
  if (t < (size_t)2 * C) ptx_st_global_f32(bias + t, (int)t < C ? ptx_ldg_f32(pb + side * C + t) : ptx_ldg_f32(gb + side * C + t - C));
}
inline void triangleBlocked2(float* pair, const float* mask, int L, int C, const std::string& T, int blk, bool outgoing) {
  struct Op { half* w; float* b; };
  static std::map<std::tuple<std::string, int, int>, Op> ops;
  auto opOf = [&](int side) {
    auto key = std::make_tuple(T, blk, side);
    auto it = ops.find(key);
    if (it != ops.end()) return it->second;
    Op o{ wpool<half>((size_t)C * 2 * C), wpool<float>(2 * C) };
    operand2K<<<blocks((size_t)C * 2 * C), 256, 0, STREAM>>>(P(T + "/projection/weights", blk), P(T + "/gate/weights", blk),
      P(T + "/projection/bias", blk), P(T + "/gate/bias", blk), o.w, o.b, C, side);
    return ops[key] = o;
  };
  Op a = opOf(0), b = opOf(1);
  TriBlockedW w{ P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), a.w, b.w, a.b, b.b,
                 P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                 PH(T + "/output_projection/weights", blk), P(T + "/output_projection/bias", blk),
                 PH(T + "/gating_linear/weights", blk), P(T + "/gating_linear/bias", blk) };
  triangleBlockedHalf(pair, mask, L, C, w, outgoing, AF2_CHUNK);
}
// 🔴 ON AN SM WITH AN A100's SHARED MEMORY, cuda/af3's streaming kernels as AF3 runs them at 128 channels (below)
inline bool af2Wide128() {
  static const bool wide128 = [] {
    if (const char* e = getenv("LOCALFOLD_TRI128")) return std::string(e) == "wide";
    int dev, perSm = 0; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    size_t lim = getenv("LOCALFOLD_SMEM_LIMIT") ? smemLimit() : (size_t)perSm;
    return lim >= 96 * 1024 && ccMajor() >= 8;     // (an RTX PRO 6000's ~100 KB too: cuda/af3's pairtrack.cuh)
  }();
  return wide128;
}
// the blocks on those streaming kernels (cuda/af3's triangleBlockedFused): the biased input kernel over the whole plane
// for b alone, then over each block's rectangle for a and its gating rows, a bf16 contraction, and the output kernel
// (its bias in the epilogue) over the rectangle - where triangleBlockedHalf ran an LN, a GEMM and a transposing gate a
// chunk and five passes on the way out, its f16 contraction on cuBLAS's 64-wide batched tiles
inline void triangleBlockedFused2(float* pair, const float* mask, int L, int C, const std::string& T, int blk, bool outgoing) {
  using B16 = __nv_bfloat16;
  const int Lp = (L + 7) / 8 * 8; const size_t plane = (size_t)Lp * Lp;
  TriFused tw = triFusedWeights(T, blk, C);
  const half* wt = tw.wt;
  B16* b = scratch<B16>("trib.bbf", plane * C);
  int width = (int)std::max<size_t>(8, std::min<size_t>(Lp, AF2_CHUNK / C / Lp / 8 * 8));
  {
    size_t f, t; deviceMemInfo(&f, &t);
    size_t perRow = (size_t)Lp * C * 6, spare = f > t / 16 ? f - t / 16 : 0;       // a, its gating rows, the product
    width = (int)std::max<size_t>(width, std::min<size_t>(Lp, std::min<size_t>(spare / perRow, 1024) / 8 * 8));
  }
  B16* a = scratch<B16>("trib.abf", (size_t)width * Lp * C);
  B16* prod = scratch<B16>("trib.pbf", (size_t)width * Lp * C);
  half* t2 = scratch<half>("trib.t2", (size_t)width * Lp * C);
  static const bool xr2 = [] {
    int dev, perSm = 0; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    return (size_t)perSm < 3 * (triIn256Smem<B16>(128, 8) + 1024);
  }();
  const size_t smem = xr2 ? triIn256Smem<B16>(128, 4, 2, 2) : triIn256Smem<B16>(128, 8);
  WITH_PAIR_T(
    static bool attr = false;
    if (!attr) {
      if (xr2) smemAttr((triIn256K<128, 4, B16, 2, true, PT, 2>), (int)smem);
      else smemAttr((triIn256K<128, 4, B16, 1, true, PT, 2>), (int)smem);
      attr = true;
    }
    auto in = [&](RectMap rm, size_t rows, B16* ao, B16* bo, half* t2o) {
      if (xr2)
        triIn256K<128, 4, B16, 2, true, PT, 2><<<(unsigned)((rows + 127) / 128), 128, smem, STREAM>>>(
          pair, mask, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), wt, ao, bo, t2o, L, Lp, plane,
          tw.bias, rm);
      else
        triIn256K<128, 4, B16, 1, true, PT, 2><<<(unsigned)((rows + 127) / 128), 128, smem, STREAM>>>(
          pair, mask, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), wt, ao, bo, t2o, L, Lp, plane,
          tw.bias, rm);
    };
    in(RectMap{}, plane, nullptr, b, nullptr);
    for (int k0 = 0; k0 < L; k0 += width) {
      int w = std::min(width, Lp - k0);
      RectMap rm = outgoing ? RectMap{k0, Lp, 0, (size_t)w * Lp} : RectMap{0, w, k0, (size_t)Lp * w};
      in(rm, rm.size, a, nullptr, t2);
      if (outgoing) bf16Gemms(CUBLAS_OP_T, CUBLAS_OP_N, Lp, w, Lp, 1.f, b, Lp, plane, a, Lp, rm.size, prod, Lp, rm.size, C);
      else bf16Gemms(CUBLAS_OP_N, CUBLAS_OP_T, w, Lp, Lp, 1.f, a, w, rm.size, b, Lp, plane, prod, w, rm.size, C);
      triangleOutRun<128, 4, B16>(prod, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                                  PH(T + "/output_projection/weights", blk), t2, pair, L, Lp, P(T + "/output_projection/bias", blk), rm, tw.wout);
    }
  );
  releaseScratch({ "trib." });
}
// the triangle's whole-form buffers (both precisions'): what a pass already holds counts toward its room
#define TRI_FAST_HELD { "ftri.a8", "ftri.b8", "ftri.a", "ftri.b", "ftri.t2", "ftri.prod", "ftri.xn", "ftri.og", "ftri.out", "ftri.abf", "ftri.bbf", "ftri.pbf" }
inline void triangleMultiplication(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk,
                                   bool outgoing, bool pairOnes = false) {
  size_t pairs = (size_t)L * L;
  std::string T = S + (outgoing ? "triangle_multiplication_outgoing" : "triangle_multiplication_incoming");
  // on a card short of room, in blocks of the output, where the whole form would not fit with room to spare
  if (FAST && shortPair(pairs, C)) {
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    if (!roomFor(5 * plane * C * 2, TRI_FAST_HELD)) {
      releaseScratch({ "ftri." });
      needF32Pair("af2 blocked triangle");
      static const bool fusedBlocks = !getenv("LOCALFOLD_TRIB_FUSED") || atoi(getenv("LOCALFOLD_TRIB_FUSED"));
      if (fusedBlocks && FUSED_TRIANGLE && C == 128 && !getenv("AF2_TRI_F32") && af2Wide128() && L >= 80 &&
          fitsSmem(triIn256Smem<__nv_bfloat16>(128, 8)))
        triangleBlockedFused2(pair, pairOnes ? nullptr : pairMask, L, C, T, blk, outgoing);
      else triangleBlocked2(pair, pairMask, L, C, T, blk, outgoing);
      return;
    }
  }
  // a, b and the contraction's product in bf16 (f32's range at half the bytes - f16's range is what overflows),
  // as cuda/af3 has them and as AlphaFold 2's own trunk runs (global_config.bfloat16): the output kernel then
  // stages half the product and takes its persistent form. AF2_TRI_F32=1 keeps the f32 product.
  static const bool triBf16 = !getenv("AF2_TRI_F32");
  const float* maskIn = pairOnes ? nullptr : pairMask;
  // 🔴 ON AN SM WITH AN A100's SHARED MEMORY, cuda/af3's streaming kernels as AF3 runs them at 128 channels: the input
  // kernel at two row tiles a warp (biased), a bf16 contraction, and the output kernel streaming its weight 16 columns
  // a stage with the bias in its epilogue - where the narrow output kernel holds the whole weight at one block an SM.
  // LOCALFOLD_TRI128=narrow|wide forces either (as cuda/af3's pairtrack.cuh)
  if (FAST && FUSED_TRIANGLE && C == 128 && triBf16 && af2Wide128() && fitsSmem(triIn256Smem<__nv_bfloat16>(128, 8))) {
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    TriFused w = triFusedWeights(T, blk, C);
    const half* wt = w.wt;
    // FP8 operands where the card has FP8 tensor instructions (fp8Tensor), as cuda/af3's triangle: both directions in the
    // outgoing layout (the incoming input kernel reading the pair transposed, RectMap's T), through an e4m3 GEMM into
    // the bf16 product, on a plane padded to 16 (an FP8 GEMM's leading dimensions)
    if (fp8Tensor() && L >= 80) {
      using F8 = __nv_fp8_e4m3;
      const int np8 = (L + 15) / 16 * 16; const size_t cs8 = (size_t)np8 * np8;
      F8* a8 = scratch<F8>("ftri.a8", cs8 * C); F8* b8 = scratch<F8>("ftri.b8", cs8 * C);
      half* t2 = scratch<half>("ftri.t2", cs8 * C);
      __nv_bfloat16* pb = scratch<__nv_bfloat16>("ftri.pbf", cs8 * C);
      RectMap rm; rm.T = !outgoing;
      const size_t smem = triIn256Smem<F8>(128, 8);
      WITH_PAIR_T(
        static bool attr = false;
        if (!attr) { smemAttr((triIn256K<128, 4, F8, 1, true, PT, 2>), (int)smem); attr = true; }
        triIn256K<128, 4, F8, 1, true, PT, 2><<<(unsigned)((cs8 + 127) / 128), 128, smem, STREAM>>>(
          pair, maskIn, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), wt, a8, b8, t2, L, np8, cs8, w.bias, rm));
      if (outgoing) triContractTN(np8, cs8, C, 1.f, a8, b8, pb, true);
      else triContractTN(np8, cs8, C, 1.f, b8, a8, pb, true);
      triangleOutRun<128, 4, __nv_bfloat16>(pb, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                                            PH(T + "/output_projection/weights", blk), t2, pair, L, np8, P(T + "/output_projection/bias", blk), {}, w.wout);
      return;
    }
    __nv_bfloat16* a = scratch<__nv_bfloat16>("ftri.abf", plane * C); __nv_bfloat16* b = scratch<__nv_bfloat16>("ftri.bbf", plane * C);
    half* t2 = scratch<half>("ftri.t2", plane * C);
    static const bool xr2 = [] {
      int dev, perSm = 0; CK(cudaGetDevice(&dev));
      CK(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
      return (size_t)perSm < 3 * (triIn256Smem<__nv_bfloat16>(128, 8) + 1024);
    }();
    if (L < 80) {
      triInRaw<__nv_bfloat16, true>(pair, maskIn, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk),
                                    w.wpg, PH(T + "/gating_linear/weights", blk), w.bias, a, b, t2, L, Lp, plane);
    } else if (xr2) {
      const size_t smem = triIn256Smem<__nv_bfloat16>(128, 4, 2, 2);
      WITH_PAIR_T(
        static bool attr = false;
        if (!attr) { smemAttr((triIn256K<128, 4, __nv_bfloat16, 2, true, PT, 2>), (int)smem); attr = true; }
        triIn256K<128, 4, __nv_bfloat16, 2, true, PT, 2><<<(unsigned)((plane + 127) / 128), 128, smem, STREAM>>>(
          pair, maskIn, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), wt, a, b, t2, L, Lp, plane, w.bias));
    } else {
      const size_t smem = triIn256Smem<__nv_bfloat16>(128, 8);
      WITH_PAIR_T(
        static bool attr = false;
        if (!attr) { smemAttr((triIn256K<128, 4, __nv_bfloat16, 1, true, PT, 2>), (int)smem); attr = true; }
        triIn256K<128, 4, __nv_bfloat16, 1, true, PT, 2><<<(unsigned)((plane + 127) / 128), 128, smem, STREAM>>>(
          pair, maskIn, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), wt, a, b, t2, L, Lp, plane, w.bias));
    }
    __nv_bfloat16* prod = scratch<__nv_bfloat16>("ftri.pbf", plane * C);
    triContractBf16(outgoing, Lp, plane, C, 1.f, a, b, prod);
    triangleOutRun<128, 4, __nv_bfloat16>(prod, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                                          PH(T + "/output_projection/weights", blk), t2, pair, L, Lp, P(T + "/output_projection/bias", blk), {}, w.wout);
    return;
  }
  if (FAST && FUSED_TRIANGLE && C == 64 && fitsSmem(std::max(triIn256Smem<half>(64, 8), triangleOutSmem<64, 4, float, 32, float>()))) {
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    TriFused w = triFusedWeights(T, blk, C);
    half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
    half* t2 = scratch<half>("ftri.t2", plane * C);
    const size_t smem = triIn256Smem<half>(64, 8);
    WITH_PAIR_T(
      static bool attr = false;
      if (!attr) { smemAttr((triIn256K<64, 4, half, 1, true, PT, 2>), (int)smem); attr = true; }
      triIn256K<64, 4, half, 1, true, PT, 2><<<(unsigned)((plane + 127) / 128), 128, smem, STREAM>>>(
        pair, maskIn, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), w.wt, a, b, t2, L, Lp, plane, w.bias));
    float* prod = scratch<float>("ftri.prod", plane * C);
    const float one = 1.f, zero = 0.f;
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16F, Lp, plane, a, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16F, Lp, plane, b, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    triangleOutRun<64, 4, float>(prod, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                                 PH(T + "/output_projection/weights", blk), t2, pair, L, Lp, P(T + "/output_projection/bias", blk), {}, w.wout);
    return;
  }
  if (FAST && FUSED_TRIANGLE && C == 128 && triBf16 && triFusedFits<__nv_bfloat16>()) {
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    TriFused w = triFusedWeights(T, blk, C);
    __nv_bfloat16* a = scratch<__nv_bfloat16>("ftri.abf", plane * C); __nv_bfloat16* b = scratch<__nv_bfloat16>("ftri.bbf", plane * C);
    half* t2 = scratch<half>("ftri.t2", plane * C);
    triInRaw<__nv_bfloat16, true>(pair, maskIn, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk),
                                  w.wpg, PH(T + "/gating_linear/weights", blk), w.bias, a, b, t2, L, Lp, plane);
    __nv_bfloat16* prod = scratch<__nv_bfloat16>("ftri.pbf", plane * C);
    triContractBf16(outgoing, Lp, plane, C, 1.f, a, b, prod);
    triOutRaw<__nv_bfloat16, true>(prod, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                                   PH(T + "/output_projection/weights", blk), P(T + "/output_projection/bias", blk), t2, pair, L, Lp,
                                   plane);
    return;
  }
  if (FAST && FUSED_TRIANGLE && C == 128 && triFusedFits<float>()) {
    // cuda/af3's fused kernels (fast.cuh, triFusedWeights): LN, the gated projections and the gating
    // linear in one kernel, AF2's contraction, then the centre norm, the output projection and the gate
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    TriFused w = triFusedWeights(T, blk, C);
    half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
    half* t2 = scratch<half>("ftri.t2", plane * C);
    triInRaw<half, true>(pair, maskIn, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), w.wpg,
                         PH(T + "/gating_linear/weights", blk), w.bias, a, b, t2, L, Lp, plane);   // (writes the padding)
    float* prod = scratch<float>("ftri.prod", plane * C);
    const float one = 1.f, zero = 0.f;
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16F, Lp, plane, a, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16F, Lp, plane, b, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    triOutRaw<float, true>(prod, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                           PH(T + "/output_projection/weights", blk), P(T + "/output_projection/bias", blk), t2, pair, L, Lp, plane);
    return;
  }
  // a T4, where neither fused form above fits (the output kernel holds the whole 128 x 128 weight, 71 KB):
  // cuda/af3's streaming triangle kernels at 128 channels, biased, the weight 16 columns a stage, MT=2 (two 16-row
  // tiles a warp, 128 rows a block; XROUNDS=2 -> 25.0 KB, 2 blocks/SM in 64 KB), and the contraction f16
  // into f32 (no bf16 MMA there) cleanly into triangleOutRun (32.0 KB, 2 blocks/SM in 64 KB)
  if (FAST && FUSED_TRIANGLE && C == 128 && L >= 80 &&
      fitsSmem(std::max(triIn256Smem<half>(128, 4, 2, 2), triangleOutSmem<128, 4, float, 32, float>()))) {
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    TriFused w = triFusedWeights(T, blk, C);
    const half* wt = w.wt;
    half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
    half* t2 = scratch<half>("ftri.t2", plane * C);
    const size_t smem = triIn256Smem<half>(128, 4, 2, 2);
    WITH_PAIR_T(
      static bool attr = false;
      if (!attr) { smemAttr((triIn256K<128, 4, half, 2, true, PT, 2>), (int)smem); attr = true; }
      triIn256K<128, 4, half, 2, true, PT, 2><<<(unsigned)((plane + 127) / 128), 128, smem, STREAM>>>(
        pair, maskIn, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), wt, a, b, t2, L, Lp, plane, w.bias));
    float* prod = scratch<float>("ftri.prod", plane * C);
    const float one = 1.f, zero = 0.f;
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16F, Lp, plane, a, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16F, Lp, plane, b, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    triangleOutRun<128, 4, float>(prod, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                                  PH(T + "/output_projection/weights", blk), t2, pair, L, Lp, P(T + "/output_projection/bias", blk), {}, w.wout);
    return;
  }
  needF32Pair("af2 triangle multiplication (unfused)");
  if (FAST) {
    TriW w = triWeights(T, blk, C);
    half* xn = scratch<half>("ftri.xn", pairs * C);
    layerNormH(pair, xn, pairs, C, T + "/left_norm_input", blk);
    // planes [Lp][Lp], the pad rows and columns zero (zeroPlanePad: at every use, the buffers' bytes being shared)
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
    zeroPlanePad(a, L, Lp, C); zeroPlanePad(b, L, Lp, C);
    // the five projections in row chunks of ~128 MB (whole, [pairs, 5C] was 0.78 GB at 783 residues):
    // a and b gated into their planes, the output gate kept [pairs, C] for the end
    size_t chunk = std::min(pairs, std::max<size_t>(32768, ((size_t)128 << 20) / (10 * (size_t)C)));
    half* pg = scratch<half>("ftri.pg", chunk * 5 * C);
    half* og = scratch<half>("ftri.og", pairs * C);
    for (size_t p0 = 0; p0 < pairs; p0 += chunk) {
      size_t r = std::min(chunk, pairs - p0);
      ltGemm(xn + p0 * C, w.w5, pg, true, r, C, 5 * C, w.b5, false, 0.f);
      triGateTK<<<dim3((unsigned)((r + 31) / 32), C / 32), dim3(32, 8), 0, STREAM>>>(pg, pairMask, a, b, r, C, L, Lp, p0);
      CK(cudaMemcpy2DAsync(og + p0 * C, C * 2, pg + 4 * C, 5 * C * 2, C * 2, r, cudaMemcpyDeviceToDevice, STREAM));
    }
    float* prod = scratch<float>("ftri.prod", plane * C);
    const float one = 1.f, zero = 0.f;
    // every extent Lp: the pad's zeros add nothing to a sum, and the pad's outputs are never read
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16F, Lp, plane, a, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16F, Lp, plane, b, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    half* cn = xn;                // (the normalised input is spent)
    centerNormHK<<<(unsigned)((pairs + 31) / 32), 256, (size_t)C * 33 * 4, STREAM>>>(prod, cn, pairs, C,
      P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk), L, Lp);
    float* out = scratch<float>("ftri.out", pairs * C);
    ltGemm(cn, w.out, out, false, pairs, C, C, P(T + "/output_projection/bias", blk), false, 0.f);
    gateMulAddHK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, out, og, pairs, C, C);
    return;
  }
  float* xn = scratch<float>("tri.xn", pairs * C);
  layerNorm(pair, xn, pairs, C, T + "/left_norm_input", blk);
  size_t wSlice = (size_t)C * 2 * C, oSlice = pairs * (size_t)2 * C;
  static std::map<const float*, const float*> pgWCache;
  int b0 = blk < 0 ? -1 : 0;
  const float* pw0 = P(T + "/projection/weights", b0);
  auto it = pgWCache.find(pw0);
  if (it == pgWCache.end()) {
    int nblk = blk < 0 ? 1 : (int)dimW(T + "/projection/weights", 0);
    float* w2 = wpool<float>((size_t)nblk * 2 * wSlice);
    packBatched2K<<<blocks((size_t)nblk * wSlice), 256, 0, STREAM>>>(
      pw0, P(T + "/gate/weights", b0), w2, nblk, wSlice);
    it = pgWCache.emplace(pw0, w2).first;
  }
  const float* w2 = it->second + (blk < 0 ? 0 : (size_t)blk * 2 * wSlice);
  float* proj = scratch<float>("tri.pg", 2 * oSlice);
  float* gate = proj + oSlice;
  const float one = 1.f, zero = 0.f;
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, 2 * C, (int)pairs, C, &one,
                               w2, 2 * C, (long long)wSlice, xn, C, 0, &zero, proj, 2 * C, (long long)oSlice, 2));
  float* a = scratch<float>("tri.a", pairs * C); float* b = scratch<float>("tri.b", pairs * C);
  triSplitK<<<dim3((unsigned)((pairs + 31) / 32), (unsigned)((C + 31) / 32)), dim3(32, 8), 0, STREAM>>>(
    proj, gate, pairMask, a, b, L, C, P(T + "/projection/bias", blk), P(T + "/gate/bias", blk));
  float* prod = scratch<float>("tri.prod", pairs * C);
  // per channel (row-major [i][k] planes): outgoing act[i,j] = sum_k a[i,k] b[j,k] = a b^T;
  // incoming act[i,j] = sum_k a[k,j] b[k,i] = (b^T a)[i,j]
  if (outgoing)
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, L, L, L, &one, b, L, pairs, a, L, pairs, &zero, prod, L,
                                 pairs, C));
  else
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_T, L, L, L, &one, a, L, pairs, b, L, pairs, &zero, prod, L,
                                 pairs, C));
  float* cn = scratch<float>("tri.cn", pairs * C);
  centerNormF32(prod, cn, pairs, C, T + "/center_norm", blk);
  float* out = scratch<float>("tri.out", pairs * C);
  gemm(cn, P(T + "/output_projection/weights", blk), out, pairs, C, C);
  float* g = scratch<float>("tri.g", pairs * C);
  gemm(xn, P(T + "/gating_linear/weights", blk), g, pairs, C, C);
  gateMulAddK<<<blocks((C & 3) == 0 ? pairs * (size_t)(C >> 2) : pairs * C), 256, 0, STREAM>>>(
    pair, out, g, pairs * C, P(T + "/output_projection/bias", blk), P(T + "/gating_linear/bias", blk), C);
}
// pair columns [b0, b0 + cnt) as rows: out[(b - b0) * L + q] = pair[q * L + b]
__global__ void gatherColumnsK(const float* pair, float* out, int L, int C, size_t b0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * L * C) return;
  int c = (int)(t % C); size_t r = t / C; size_t q = r % L, b = b0 + r / L;
  ptx_st_global_f32(out + t, ptx_ldg_f32(pair + (q * L + b) * C + c));
}
// pair[q * L + b] += x[(b - b0) * L + q], the columns [b0, b0 + cnt)
__global__ void addColumnsK(float* pair, const float* x, int L, int C, size_t b0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * L * C) return;
  int c = (int)(t % C); size_t r = t / C; size_t q = r % L, b = b0 + r / L;
  size_t dst = (q * L + b) * C + c;
  ptx_st_global_f32(pair + dst, ptx_add_f32(ptx_ld_global_f32(pair + dst), ptx_ldg_f32(x + t)));
}
// The triangle attention on a card short of room (--fast), in chunks of attention rows: the bias from the
// pair in chunks first, then per chunk its rows' LayerNorm, q/k/v/gate, the flash kernel and the output
// projection added back - never the LayerNorm'd pair, the q/k/v/gate or the output whole, nor the
// ending node's transposed copy of the pair (its rows are gathered from the columns, its mask read
// transposed). In place it is safe: a row (column) chunk reads and writes only its own rows (columns).
inline void triangleAttentionChunked(float* pair, const float* pairMask, int L, int C, const std::string& A, int blk,
                                     bool starting, bool pairOnes) {
  size_t pairs = (size_t)L * L;
  AttnW w = attnWeights(A + "/attention", blk, C);
  int H = w.H, Wp = w.H * w.Dp, stride = (L + 7) / 8 * 8;
  static std::map<const float*, half*> whCache;
  const float* wf = P(A + "/feat_2d_weights", blk);
  auto it = whCache.find(wf);
  if (it == whCache.end()) {
    half* h = wpool<half>((size_t)C * H);
    toHalfK<<<blocks((size_t)C * H), 256, 0, STREAM>>>(wf, h, (size_t)C * H);
    it = whCache.emplace(wf, h).first;
  }
  half* bias = scratch<half>("fbias.bias", (size_t)H * L * stride);
  size_t per = std::max<size_t>(1, std::min(pairs, AF2_CHUNK / C));
  half* lnc = scratch<half>("ftatt.lnc", per * C);
  float* projc = scratch<float>("fbias.projc", per * H);
  for (size_t r0 = 0; r0 < pairs; r0 += per) {
    size_t r = std::min(per, pairs - r0);
    layerNormH(pair + r0 * C, lnc, r, C, A + "/query_norm", blk);
    ltGemm(lnc, it->second, projc, false, r, C, H, nullptr, false, 0.f);
    biasFromProjRowsK<<<blocks(r * H), 256, 0, STREAM>>>(projc, bias, L, H, stride, !starting, r0, r);
  }
  // attention rows a chunk: a row is L positions, and its q/k/v/gate 4 Wp halves each
  size_t R = std::max<size_t>(1, std::min<size_t>(L, AF2_CHUNK / ((size_t)L * 4 * Wp)));
  half* xn = scratch<half>("ftatt.xn", R * L * C);
  float* gath = starting ? nullptr : scratch<float>("ftatt.cols", R * L * C);
  half* qkvg = scratch<half>("fatt.qkvg", (R * L + 128) * 4 * Wp);
  half* o = scratch<half>("fatt.o", R * L * Wp);
  float* tmp = starting ? nullptr : scratch<float>("fatt.tmp", R * L * C);
  for (size_t b0 = 0; b0 < (size_t)L; b0 += R) {
    size_t bc = std::min(R, (size_t)L - b0), rows = bc * L;
    const float* src = pair + b0 * L * C;
    if (!starting) { gatherColumnsK<<<blocks(rows * C), 256, 0, STREAM>>>(pair, gath, L, C, b0, bc); src = gath; }
    layerNormH(src, xn, rows, C, A + "/query_norm", blk);
    ltGemm(xn, w.qkvg, qkvg, true, rows, C, 4 * Wp, w.qkvgBias, false, 0.f);
    flashGrid<half>(qkvg, bias, stride, pairOnes ? nullptr : pairMask, o, L, H, w.Dp, b0, bc, !starting,
                    1.f / sqrtf((float)w.D));
    if (starting) {
      ltGemm(o, w.out, pair + b0 * L * C, false, rows, Wp, C, P(A + "/attention/output_b", blk), false, 1.f);
    } else {
      ltGemm(o, w.out, tmp, false, rows, Wp, C, P(A + "/attention/output_b", blk), false, 0.f);
      addColumnsK<<<blocks(rows * C), 256, 0, STREAM>>>(pair, tmp, L, C, b0, bc);
    }
  }
}
inline bool FUSED_GRID_AF2 = !getenv("LOCALFOLD_AF2_UNFUSED_GRID");
// The triangle attention on cuda/af3's fused grid kernels: LN + q/k/v/gate (+ the gate bias) + the pair
// bias projection in one kernel reading the ending node's rows transposed where they lie, the flash
// kernel, and the output projection (+ its bias) added into the pair - transposed in place for the ending
// node, so no gather, no scatter and no LN'd copy of the pair. One pass where the q/k/v/gate are a 32nd of
// the card (the bias written by the same kernel); else the bias first (LN + projection, chunked), then the
// rows a chunk at a time.
template <int C = 128>
inline void triangleAttentionGrid(float* pair, const float* pairMask, int L, const std::string& A, int blk,
                                  bool starting, bool pairOnes) {
  size_t pairs = (size_t)L * L;
  AttnW w = attnWeights(A + "/attention", blk, C);
  int H = w.H, Wp = H * w.Dp, stride = (L + 7) / 8 * 8;
  const bool tr = !starting;
  const float *lnS = P(A + "/query_norm/scale", blk), *lnO = P(A + "/query_norm/offset", blk);
  const float* ob = P(A + "/attention/output_b", blk);
  const half* Wb = paddedFeat2dWeight(P(A + "/feat_2d_weights", blk), C, H);
  size_t biasN = (size_t)H * L * stride;
  half* bias = scratch<half>("fbias.bias", biasN);
  static half* zeroedBias = nullptr; static size_t zeroedBiasN = 0;
  if (stride != L && (bias != zeroedBias || biasN > zeroedBiasN)) {
    CK(cudaMemsetAsync(bias, 0, biasN * 2, STREAM));                // the padding columns (once per buffer)
    zeroedBias = bias; zeroedBiasN = biasN;
  }
  const float* mask = pairOnes ? nullptr : pairMask;
  float scale = 1.f / sqrtf((float)w.D);
  // FP8 attention where the card has it (cuda/af3's flash8Run): k and v written e4m3 by the projection
  const bool f8 = w.Dp == 32 && !mask && fp8Attn();
  const size_t f8Row = f8 ? (size_t)3 * Wp : 0;            // (k8, v8 and the transposed v: bytes an element)
  // every row in one pass whenever the card has the room for its q/k/v/gate and output (what they already
  // hold counted, so each pass decides alike) - not below a fixed 32nd of the card: at 1,566 residues
  // 17.43 -> 16.43 s, at 2,088 35.80 -> 34.19 s (peak 7.9 -> 10.7 and 12.0 -> 17.1 GB on 40 GB); the
  // chunked form only where the memory is not there (LOCALFOLD_BIG=1 forces it, roomFor).
  // 🔴 The room asked for includes the triangle multiplication's whole form (five planes): these buffers
  // stay held while it runs, and starved of them it blocks - at 2,610 residues the one-pass grid took the
  // memory and the fold went 63.7 -> 104.4 s
  size_t triPlane = (size_t)((L + 7) / 8 * 8) * ((L + 7) / 8 * 8);
  if (roomFor(((pairs + 128) * 4 * Wp + pairs * Wp) * 2 + pairs * f8Row + 5 * triPlane * C * 2,
              {"fatt.qkvg", "fatt.o", "fatt.kv8", "attn.vt8", "ftri.a", "ftri.b", "ftri.t2", "ftri.prod", "ftri.xn", "ftri.og", "ftri.out", "ftri.abf", "ftri.bbf", "ftri.pbf"})) {
    half* qkvg = scratch<half>("fatt.qkvg", (pairs + 128) * 4 * Wp);
    uint8_t* kv8 = f8 ? scratch<uint8_t>("fatt.kv8", pairs * 2 * Wp) : nullptr;
    // (sm_12x: V written straight into flash8's transposed form, no fp8VtK pass - the same bytes)
    uint8_t* vt8 = f8 && ccMajor() == 12 ? scratch<uint8_t>("attn.vt8", (size_t)L * Wp * ((L + 63) / 64 * 64)) : nullptr;
    gridInRaw<C>(pair, lnS, lnO, w.qkvg, w.qkvgBias, qkvg, L, 0, pairs, tr, Wb, bias, H, stride, tr, kv8, ccMajor() == 12, vt8);
    half* o = scratch<half>("fatt.o", pairs * Wp);
    if (f8) flash8Run(qkvg, kv8, vt8 ? vt8 : kv8 + pairs * Wp, bias, stride, o, L, H, L, scale, nullptr, -1, vt8 != nullptr);
    else flashGrid<half>(qkvg, bias, stride, mask, o, L, H, w.Dp, 0, L, tr, scale);
    // (the row direction's GEMM accumulates into an f32 pair; a bf16 pair takes gridOutK in both directions -
    // an f16 product and an add pass were 6-10 ms slower a 494-residue fold)
    // (an f32 pair on sm_12x too: there cuBLASLt runs that GEMM's bias-and-accumulate epilogue as a second kernel over the
    // pair - on Colab's RTX PRO 6000 54 calls of cublasLt's globalKernel a 2,500-residue pass, and timing its other
    // candidates found none without it)
    if (!tr && !PAIR16 && ccMajor() != 12) ltGemm(o, w.out, pair, false, pairs, Wp, C, ob, false, 1.f);
    else gridOutRaw<C>(o, w.out, ob, pair, L, 0, pairs, tr);
    return;
  }
  {   // the bias, all of it before any row is attended (and before any is written): LN + the 16-column
      // projection in one kernel (lnHeadsK), laid out into `bias`
    size_t per = std::max<size_t>(1, std::min(pairs, AF2_CHUNK / 16));
    for (size_t r0 = 0; r0 < pairs; r0 += per) {
      size_t r = std::min(per, pairs - r0);
      lnHeadsRaw<16, C>(pairAdvance(pair, r0 * C), lnS, lnO, Wb, nullptr, r, bias, L, H, stride, tr, r0, nullptr);
    }
  }
  size_t R = std::max<size_t>(1, std::min<size_t>(L, AF2_CHUNK / ((size_t)L * 4 * Wp)));
  half* qkvg = scratch<half>("fatt.qkvg", (R * L + 128) * 4 * Wp);
  half* o = scratch<half>("fatt.o", R * L * Wp);
  uint8_t* kv8 = f8 ? scratch<uint8_t>("fatt.kv8", R * L * 2 * Wp) : nullptr;
  for (size_t b0 = 0; b0 < (size_t)L; b0 += R) {
    size_t bc = std::min(R, (size_t)L - b0), rows = bc * L;
    gridInRaw<C>(pair, lnS, lnO, w.qkvg, w.qkvgBias, qkvg, L, b0 * L, rows, tr, nullptr, nullptr, 0, 0, false, kv8, ccMajor() == 12);
    if (f8) flash8Run(qkvg, kv8, kv8 + rows * Wp, bias, stride, o, L, H, bc, scale, nullptr);
    else flashGrid<half>(qkvg, bias, stride, mask, o, L, H, w.Dp, b0, bc, tr, scale);
    if (!tr && !PAIR16 && ccMajor() != 12) ltGemm(o, w.out, pair + b0 * L * C, false, rows, Wp, C, ob, false, 1.f);
    else gridOutRaw<C>(o, w.out, ob, pair, L, b0 * L, rows, tr);
  }
}
inline void triangleAttention(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk,
                              bool starting, bool pairOnes = false) {
  size_t pairs = (size_t)L * L;
  std::string A = S + (starting ? "triangle_attention_starting_node" : "triangle_attention_ending_node");
  int H = (int)dimW(A + "/attention/query_w", 2), D = (int)dimW(A + "/attention/query_w", 3);
  if (FAST && FUSED_GRID_AF2 && C == 128 && H * (D < 16 ? 16 : D) == 128 && H <= 16 && gridFusedFits(128)) {
    triangleAttentionGrid<128>(pair, pairMask, L, A, blk, starting, pairOnes);
    return;
  }
  if (FAST && FUSED_GRID_AF2 && C == 64 && H * (D < 16 ? 16 : D) == 64 && H <= 16 && gridFusedFits(64)) {
    triangleAttentionGrid<64>(pair, pairMask, L, A, blk, starting, pairOnes);
    return;
  }
  needF32Pair("af2 triangle attention (unfused)");
  if (FAST && shortPair(pairs, C)) { triangleAttentionChunked(pair, pairMask, L, C, A, blk, starting, pairOnes); return; }
  if (FAST && pairOnes && !starting) {
    half* xn = scratch<half>("ftatt.xn", pairs * C);
    layerNormH(pair, xn, pairs, C, A + "/query_norm", blk);
    const half* bias = pairBiasFast(xn, L, C, H, P(A + "/feat_2d_weights", blk), nullptr, true);
    attentionCoreAcross(xn, L, L, C, A + "/attention", blk, bias, pair);
    return;
  }
  const float* x = pair; const float* mask = pairMask;
  float* tr = starting ? nullptr : scratch<float>("tatt.tr", pairs * C);
  float* mt = starting ? nullptr : scratch<float>("tatt.mask", pairs);
  if (!starting) { swap01(pair, tr, L, L, C); if (!FAST || !pairOnes) swap01(pairMask, mt, L, L, 1); x = tr; mask = mt; }
  if (FAST) {
    half* xn = scratch<half>("ftatt.xn", pairs * C);
    layerNormH(x, xn, pairs, C, A + "/query_norm", blk);
    const half* bias = pairBiasFast(xn, L, C, H, P(A + "/feat_2d_weights", blk), nullptr);
    attentionCore(xn, L, L, C, A + "/attention", blk, pairOnes ? nullptr : mask, bias, pair, !starting);
    return;
  }
  float* xn = scratch<float>("tatt.xn", pairs * C);
  layerNorm(x, xn, pairs, C, A + "/query_norm", blk);
  float* proj = scratch<float>("tatt.proj", pairs * H);
  gemm(xn, P(A + "/feat_2d_weights", blk), proj, pairs, C, H);
  float* bias = scratch<float>("tatt.bias", pairs * H);
  pairBiasK<<<blocks(pairs * H), 256, 0, STREAM>>>(proj, nullptr, bias, L, H);
  float* out = scratch<float>("tatt.out", pairs * C);
  gatedAttention(xn, L, L, C, A + "/attention", blk, H, D, mask, bias, out, pair, starting ? 0 : L, starting ? 0 : L);
}

inline void warmStackWeights(bool extraZeros = false, bool multimer = false, bool monomerTemplates = false) {
  if (!FAST) return;
  if (multimer || monomerTemplates) {
    const std::string TS = multimer ? "evoformer/template_embedding/single_template_embedding/template_embedding_iteration/"
                                    : "evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/";
    int nb = (int)dimW(TS + "pair_transition/transition1/weights", 0);
    int Ht = (int)dimW(TS + "triangle_attention_starting_node/attention/query_w", 2);
    int Ip = (int)dimW(TS + "pair_transition/transition1/weights", 2);
    for (int b = 0; b < nb; ++b) {
      triFusedWeights(TS + "triangle_multiplication_outgoing", b, 64);
      triFusedWeights(TS + "triangle_multiplication_incoming", b, 64);
      paddedFeat2dWeight(P(TS + "triangle_attention_starting_node/feat_2d_weights", b), 64, Ht);
      paddedFeat2dWeight(P(TS + "triangle_attention_ending_node/feat_2d_weights", b), 64, Ht);
      attnWeights(TS + "triangle_attention_starting_node/attention", b, 64);
      attnWeights(TS + "triangle_attention_ending_node/attention", b, 64);
      transitionW1Tiles<true>(PH(TS + "pair_transition/transition1/weights", b), 64, Ip, 16);
    }
    if (multimer) {
      PH("evoformer/template_embedding/single_template_embedding/template_pair_embedding_8/weights");
      PH("evoformer/template_embedding/output_linear/weights");
    } else {
      PH("evoformer/template_embedding/attention/value_w");
      PH("evoformer/template_embedding/attention/output_w");
    }
  }
  int extraBlocks = (int)dimW("evoformer/extra_msa_stack/msa_transition/transition1/weights", 0);
  int mainBlocks = (int)dimW("evoformer/evoformer_iteration/msa_transition/transition1/weights", 0);
  for (int s = 0; s < 2; ++s) {
    bool extra = s == 0;
    const std::string S = extra ? "evoformer/extra_msa_stack/" : "evoformer/evoformer_iteration/";
    int nblk = extra ? extraBlocks : mainBlocks, C = extra ? 64 : 256;
    int H = (int)dimW(S + "msa_row_attention_with_pair_bias/attention/query_w", 2);
    int Ht = (int)dimW(S + "triangle_attention_starting_node/attention/query_w", 2);
    int I = (int)dimW(S + "msa_transition/transition1/weights", 2);
    int Ip = (int)dimW(S + "pair_transition/transition1/weights", 2);
    for (int b = 0; b < nblk; ++b) {
      if (!extra || !extraZeros) {
        opmWeights(S + "outer_product_mean", b, C);
        paddedFeat2dWeight(P(S + "msa_row_attention_with_pair_bias/feat_2d_weights", b), 128, H);
        AttnW rw = attnWeights(S + "msa_row_attention_with_pair_bias/attention", b, C, extra);
        if (extra) {
          attnWeights(S + "msa_row_attention_with_pair_bias/attention", b, C, false);
          transitionW1Tiles<true>(PH(S + "msa_transition/transition1/weights", b), 64, I, 16);
        } else {
          const half* w2 = augmentedWeight(S + "msa_transition/transition2/weights", b, P(S + "msa_transition/transition2/bias", b), I, C);
          biasFor(P(S + "msa_transition/transition1/bias", b), I, true);
          AttnW cw = attnWeights(S + "msa_column_attention/attention", b, C);
          attnVgWeight(cw, C);
          if (!MSA16_HALF) {
            bf16Of(rw.out, (size_t)rw.H * rw.Dp * C);
            bf16Of(cw.out, (size_t)cw.H * cw.Dp * C);
            bf16Of(PH(S + "msa_transition/transition1/weights", b), (size_t)C * I);
            bf16Of(PH(S + "msa_transition/transition2/weights", b), (size_t)I * C);
            bf16Of(w2, (size_t)(I + 8) * C);
            biasForBf16(P(S + "msa_transition/transition1/bias", b), I);
          }
        }
      }
      triFusedWeights(S + "triangle_multiplication_outgoing", b, 128);
      triFusedWeights(S + "triangle_multiplication_incoming", b, 128);
      paddedFeat2dWeight(P(S + "triangle_attention_starting_node/feat_2d_weights", b), 128, Ht);
      paddedFeat2dWeight(P(S + "triangle_attention_ending_node/feat_2d_weights", b), 128, Ht);
      attnWeights(S + "triangle_attention_starting_node/attention", b, 128);
      attnWeights(S + "triangle_attention_ending_node/attention", b, 128);
      transitionW1Tiles<true>(PH(S + "pair_transition/transition1/weights", b), 128, Ip, 16);
    }
  }
}
// one Evoformer iteration of a stack: S is "evoformer/evoformer_iteration/" or ".../extra_msa_stack/"
inline void evoformerBlock(Trunk& t, bool extraStack, int blk) {
  const std::string S = extraStack ? "evoformer/extra_msa_stack/" : "evoformer/evoformer_iteration/";
  float* msa = extraStack ? t.extra : MSA16 ? t.msa16 : t.msa;
  int rowsN = extraStack ? t.E : t.N + t.T, C = extraStack ? 64 : 256;
  const float* mask = extraStack ? t.extraMask : t.msaMask;
  std::string R = S + "msa_row_attention_with_pair_bias/attention/query_w";
  int H = (int)dimW(R, 2), D = (int)dimW(R, 3);
  const bool skipMsa = extraStack && t.extraZeros;
  if (t.opmFirst) { PAIR16 = AF2_P16; outerProductMean(t, S, blk, msa, rowsN, C, mask); PAIR16 = false; }
  if (!skipMsa) {
    PAIR16 = AF2_P16; msaRowAttention(t, S, blk, msa, rowsN, C, H, D, mask); PAIR16 = false;
    if (extraStack) msaColumnGlobalAttention(t, S, blk, msa, rowsN, C, mask);
    else msaColumnAttention(t, S, blk, msa, rowsN, C, H, D, mask);
    transition(msa, (size_t)rowsN * t.L, C, S + "msa_transition", blk);
  }
  PAIR16 = AF2_P16;
  if (!t.opmFirst) outerProductMean(t, S, blk, msa, rowsN, C, mask);
  TRI_LT_TILE = (t.N + t.T > 1 || M.meta("meta/templates", 0) > 0);
  triangleMultiplication(t.pair, t.pairMask, t.L, 128, S, blk, true, t.pairOnes);
  triangleMultiplication(t.pair, t.pairMask, t.L, 128, S, blk, false, t.pairOnes);
  TRI_LT_TILE = true;
  triangleAttention(t.pair, t.pairMask, t.L, 128, S, blk, true, t.pairOnes);
  triangleAttention(t.pair, t.pairMask, t.L, 128, S, blk, false, t.pairOnes);
  transition(t.pair, (size_t)t.L * t.L, 128, S + "pair_transition", blk);
  PAIR16 = false;
}
// whether every update both stacks run takes a bf16 pair (each route below has a bf16 form; the f32-only ones
// refuse through needF32Pair): the fused triangle (either fused form, or a T4's streaming one), the fused grid
// attention, the fused transition, and not a card short of room (its blocked and in-place forms are f32)
inline bool af2Pair16Ok(int L) {
  // (Ampere on: a T4 has no f32 -> bf16 conversion instruction, and there the bf16 pair was 1-3% SLOWER - its
  // biased float-tile triangle output 1237 -> 1520 ms a 494-residue fold, gridOutK in both directions 563 -> 780)
  static const int major = ccMajor();
  if (major < 8 || !FAST || getenv("LOCALFOLD_PAIR_F32") || shortPair((size_t)L * L, 128) || !FUSED_TRIANGLE || !FUSED_GRID_AF2) return false;
  bool tri = triFusedFits<__nv_bfloat16>() || triFusedFits<float>() ||
             (L >= 80 && fitsSmem(std::max(triIn256Smem<half>(128, 8), triangleOutSmem<128, 4, float, 32, float>())));
  if (!tri || !gridFusedFits()) return false;
  for (const char* S : { "evoformer/extra_msa_stack/", "evoformer/evoformer_iteration/" }) {
    for (const char* A : { "triangle_attention_starting_node", "triangle_attention_ending_node" }) {
      std::string q = std::string(S) + A + "/attention/query_w";
      int H = (int)dimW(q, 2), D = (int)dimW(q, 3);
      if (H * (D < 16 ? 16 : D) != 128 || H > 16) return false;
    }
    int I = (int)dimW(std::string(S) + "pair_transition/transition1/weights", 2);
    if (!fusedTransitionFits((size_t)L * L, 128, I)) return false;
  }
  return true;
}
