// The pair track - triangle multiplication, grid ("triangle") attention, transition -
// shared by the pairformer, the MSA stack and the template stack. Transcribed from
// src/af3/trunk/pairformer-reference.js; generic in channels, heads and head width.
#pragma once
#include "common.cuh"

// Where a pair update adds its residual: its own input (nullptr, every model) or, for chai-1's parallel
// block (trunk.cuh's parallelPairUpdates), the block's running sum - so an update reads the pair ENTERING the
// block and adds into another buffer, with no copy of it. Every residual add of the three updates below
// goes through into(); a kernel that adds in place without it would write the block's input.
inline float* RESIDUAL_INTO = nullptr;
inline float* into(float* x) { return RESIDUAL_INTO ? RESIDUAL_INTO : x; }
// ...and chai's ending-node attention adds its difference TRANSPOSED (AF3's at (j, i) is chai's at (i, j)), which
// is the column pass's own residual without its transpose
inline bool RESIDUAL_UNTRANSPOSED = false;

// ---------------------------------------------------------------- elementwise kernels
// LayerNorm over the last axis with AF3's fast variance E[x^2] - E[x]^2. One warp a row.
template <class TI, class TO>
__global__ void layerNormK(const TI* in, TO* out, size_t rows, int C, const float* scale,
                           const float* offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const TI* x = in + row * C;
  float s = 0, ss = 0;
  for (int c = lane; c < C; c += 32) { float v = toF(x[c]); s += v; ss += v * v; }
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(~0u, s, o); ss += __shfl_xor_sync(~0u, ss, o); }
  float mean = s / C, inv = rsqrtf(ss / C - mean * mean + 1e-5f);
  for (int c = lane; c < C; c += 32)
    out[row * C + c] = fromF<TO>((toF(x[c]) - mean) * inv * scale[c] + offset[c]);
}
template <class TI, class TO>
void layerNorm(const TI* in, TO* out, size_t rows, int C, const std::string& w) {
  layerNormK<TI, TO><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(
    in, out, rows, C, W(w + "Scale"), W(w + "Offset"));
}
template <class TI, class TO>
void layerNorm2(const TI* in, TO* out, size_t rows, int C, const std::string& scale,
                const std::string& offset) {
  layerNormK<TI, TO><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(in, out, rows, C, W(scale), W(offset));
}
// gated = swish(a) * b (* u with `up`: boltz2's conditioned transition up-gate), a row of wide
// being [a | b] or [a | b | u], each I wide
template <class T>
__global__ void swigluK(const T* wide, T* gated, size_t rows, int I, bool up) {
  int L = up ? 3 * I : 2 * I;
  if constexpr (std::is_same_v<T, half>) {
    if (I % 8 == 0) {                 // eight halves (16 bytes) a thread
      size_t t = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
      if (t >= rows * I) return;
      size_t r = t / I; int i = (int)(t % I);
      uint4 a = *(const uint4*)(wide + r * L + i), b = *(const uint4*)(wide + r * L + I + i), o, u;
      if (up) u = *(const uint4*)(wide + r * L + 2 * I + i);
      const half2* a2 = (const half2*)&a; const half2* b2 = (const half2*)&b; half2* o2 = (half2*)&o;
      const half2* u2 = (const half2*)&u;
      for (int k = 0; k < 4; ++k) {
        float2 g = __half22float2(a2[k]), v = __half22float2(b2[k]);
        float x = g.x * sigm(g.x) * v.x, y = g.y * sigm(g.y) * v.y;
        if (up) { float2 w = __half22float2(u2[k]); x *= w.x; y *= w.y; }
        o2[k] = __floats2half2_rn(x, y);
      }
      *(uint4*)(gated + t) = o;
      return;
    }
  }
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int i = (int)(t % I);
  float g = toF(wide[r * L + i]);
  float v = g * sigm(g) * toF(wide[r * L + I + i]);
  if (up) v *= toF(wide[r * L + 2 * I + i]);
  gated[t] = fromF<T>(v);
}
// the launch: an eighth of the threads for the vector path
template <class T>
void swiglu(const T* wide, T* gated, size_t rows, int I, bool up = false) {
  size_t work = std::is_same_v<T, half> && I % 8 == 0 ? rows * I / 8 : rows * I;
  swigluK<T><<<blocks(work), 256, 0, STREAM>>>(wide, gated, rows, I, up);
}
// a conditioned transition's first weight: transition1, or [transition1 | ffwAToB] where the
// bundle carries boltz2's up-gate (one GEMM for both)
inline std::string upGatedTransition1(const std::string& B, int C, int I, bool& up) {
  up = hasW(B + ".ffwAToB");
  if (!up) return B + ".ffwTransition1";
  return concatColumns(B + ".ffwTransition1|aToB~", C, {{B + ".ffwTransition1", 2 * I, false}, {B + ".ffwAToB", I, false}});
}

// ---------------------------------------------------------------- fused weights
// The triangle's projection and gate as one (C, 4C) matrix: [projection (2C) | gate (2C)].
inline std::string projectionGate(const std::string& pre, int C) {
  return concatColumns(pre + ".projectionGate~", C, {{pre + ".projection", 2 * C, false}, {pre + ".gate", 2 * C, false}});
}
// q, k, v and the gate as one (C, 4W) matrix. The grid attention stores q, k and the gate
// (out, in) and v (in, out); `transposedQkg` says whether that holds.
inline std::string qkvgWeight(const std::string& pre, int C, int Wd, bool transposedQkg) {
  return concatColumns(pre + ".qkvg~", C, {{pre + ".qProjection", Wd, transposedQkg}, {pre + ".kProjection", Wd, transposedQkg},
                                          {pre + ".vProjection", Wd, false}, {pre + ".gatingQuery", Wd, transposedQkg}});
}

// ---------------------------------------------------------------- triangle multiplication
// a, b from [rows][4C] = [projection | gate], the INTERLEAVED split (channel ch's halves
// are 2ch and 2ch+1), masked, written channel-major (c, pairs) through shared memory.
template <class T>
__global__ void triGateK(const T* pg, const float* mask, T* a, T* b, size_t r0, size_t rows, int C,
                         size_t pairs, int n, int np) {
  __shared__ float A[32][33], B[32][33];
  size_t row0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    size_t local = row0 + ry; int c = c0 + tx;
    float va = 0, vb = 0;
    if (local < rows && c < C) {
      float m = mask[r0 + local];
      const T* p = pg + local * 4 * C; const T* g = p + 2 * C;
      if constexpr (std::is_same_v<T, half>) {        // channel c's two halves, one 4-byte load each
        float2 pv = __half22float2(*reinterpret_cast<const half2*>(p + c * 2));
        float2 gv = __half22float2(*reinterpret_cast<const half2*>(g + c * 2));
        va = pv.x * m * sigm(gv.x); vb = pv.y * m * sigm(gv.y);
      } else {
        va = toF(p[c * 2]) * m * sigm(toF(g[c * 2]));
        vb = toF(p[c * 2 + 1]) * m * sigm(toF(g[c * 2 + 1]));
      }
    }
    A[ry][tx] = va; B[ry][tx] = vb;
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t local = row0 + tx; int c = c0 + cy;
    if (local < rows && c < C) {
      unsigned p = (unsigned)(r0 + local), i = p / (unsigned)n;
      size_t q = (size_t)i * np + (p - i * (unsigned)n);     // the padded (np x np) position
      a[(size_t)c * pairs + q] = fromF<T>(A[tx][cy]);
      b[(size_t)c * pairs + q] = fromF<T>(B[tx][cy]);
    }
  }
}
// center_norm over the channels of a (c, pairs) array -> row-major, 32 pairs a block.
template <class TO>
__global__ void centerNormK(const float* prod, TO* out, size_t r0, size_t rows, int C, size_t pairs,
                            const float* scale, const float* offset, int n, int np) {
  extern __shared__ float T_[];
  __shared__ float mean[32], inv[32];
  size_t i0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
  for (int c = ty; c < C; c += 8) {
    size_t local = i0 + tx;
    unsigned p = (unsigned)(r0 + local), i = p / (unsigned)n;
    T_[c * 33 + tx] = local < rows ? prod[(size_t)c * pairs + (size_t)i * np + (p - i * (unsigned)n)] : 0.f;
  }
  __syncthreads();
  if (ty == 0) {
    float s = 0, ss = 0;
    for (int c = 0; c < C; ++c) { float v = T_[c * 33 + tx]; s += v; ss += v * v; }
    float m = s / C; mean[tx] = m; inv[tx] = rsqrtf(ss / C - m * m + 1e-5f);
  }
  __syncthreads();
  for (int ry = ty; ry < 32; ry += 8) {
    size_t local = i0 + ry; if (local >= rows) continue;
    for (int c = tx; c < C; c += 32)
      out[local * C + c] = fromF<TO>((T_[c * 33 + ry] - mean[ry]) * inv[ry] * scale[c] + offset[c]);
  }
}

// the same in two passes over prod (the statistics, then the normalised rows) when its C x 33 tile does not
// fit (IntelliFold-2's 512 channels on a T4: 67.5 KB)
template <class TO>
__global__ void centerNormStreamK(const float* prod, TO* out, size_t r0, size_t rows, int C, size_t pairs,
                                  const float* scale, const float* offset, int n, int np) {
  __shared__ float ps[8][33], pss[8][33], mean[32], inv[32];
  size_t i0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
  size_t local = i0 + tx;
  unsigned p = (unsigned)(r0 + local), i = p / (unsigned)n;
  size_t at = (size_t)i * np + (p - i * (unsigned)n);
  float s = 0, ss = 0;
  if (local < rows) for (int c = ty; c < C; c += 8) { float v = prod[(size_t)c * pairs + at]; s += v; ss += v * v; }
  ps[ty][tx] = s; pss[ty][tx] = ss;
  __syncthreads();
  if (ty == 0) {
    float a = 0, b = 0;
    for (int k = 0; k < 8; ++k) { a += ps[k][tx]; b += pss[k][tx]; }
    float m = a / C; mean[tx] = m; inv[tx] = rsqrtf(b / C - m * m + 1e-5f);
  }
  __syncthreads();
  for (int ry = ty; ry < 32; ry += 8) {
    size_t lr = i0 + ry; if (lr >= rows) continue;
    unsigned q = (unsigned)(r0 + lr), qi = q / (unsigned)n;
    size_t qa = (size_t)qi * np + (q - qi * (unsigned)n);
    for (int c = tx; c < C; c += 32)
      out[lr * C + c] = fromF<TO>((prod[(size_t)c * pairs + qa] - mean[ry]) * inv[ry] * scale[c] + offset[c]);
  }
}

inline bool TIGHT_STACK = false;          // the template stack on a trunk short of room (triangle, templateEmbedding)
inline bool FUSED_GRID = true;
inline bool TRI_BF16 = true;
inline int TRI_PAD = 8;           // the triangle's padded size is a multiple of this (0: none)
inline size_t CHUNK = (size_t)64 << 20;   // elements in a chunk tensor
#include "fusedtriangle.cuh"
#include "fused256.cuh"
template <class T, class PT = float>
__global__ void gatedAddK(float* pair, const T* proj, const T* gate, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pairSt<PT>(pair, i, pairLd<PT>(pair, i) + toF(proj[i]) * sigm(toF(gate[i])));
}
// ---- the pair's rows in either element type (PAIR16): a row's address, a LayerNorm of rows, a product added in
inline float* pairRow(float* pair, size_t row, int C) {
  return PAIR16 ? reinterpret_cast<float*>(reinterpret_cast<__nv_bfloat16*>(pair) + row * C) : pair + row * C;
}
inline const float* pairRow(const float* pair, size_t row, int C) { return pairRow(const_cast<float*>(pair), row, C); }
// pair16[r] += h[r] for rows of C halves (the bf16 pair's row-direction grid output, after its f16 GEMM)
__global__ void addHalfToBf16K(__nv_bfloat16* pair, const half* h, size_t n8) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n8) return;
  uint4 p = reinterpret_cast<uint4*>(pair)[i], a = reinterpret_cast<const uint4*>(h)[i];
  __nv_bfloat162* pp = reinterpret_cast<__nv_bfloat162*>(&p); const half2* aa = reinterpret_cast<const half2*>(&a);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 x = __bfloat1622float2(pp[k]), y = __half22float2(aa[k]);
    pp[k] = __floats2bfloat162_rn(x.x + y.x, x.y + y.y);
  }
  reinterpret_cast<uint4*>(pair)[i] = p;
}
inline void rowOut16(const half* gathered, float* pair, size_t rows, int Wd, int C, const std::string& w) {
  // (in chunks: a whole pass's product would be a pair-sized f16 tensor, the bytes the bf16 pair saves)
  size_t per = std::max<size_t>(1, std::min(rows, CHUNK / C));
  half* o = scratch<half>("grid.out16", per * C);
  for (size_t r0 = 0; r0 < rows; r0 += per) {
    size_t r = std::min(per, rows - r0);
    linear<half, half>(gathered + r0 * Wd, o, r, Wd, C, w);
    addHalfToBf16K<<<blocks(r * C / 8), 256, 0, STREAM>>>(reinterpret_cast<__nv_bfloat16*>(pair) + r0 * C, o, r * C / 8);
  }
}
// LN of pair rows [r0, r0 + rows) into out (the pair f32 or, under PAIR16, bf16)
template <class TO>
inline void lnPairRows(const float* pair, size_t r0, TO* out, size_t rows, int C, const std::string& scale,
                       const std::string& offset) {
  WITH_PAIR_T(layerNormK<PT, TO><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(
    reinterpret_cast<const PT*>(pair) + r0 * C, out, rows, C, W(scale), W(offset)));
}
// pair rows [r0, r0 + rows) += X W: the GEMM's own beta into an f32 pair, an f16 product and an add into a bf16 one
template <class T>
inline void linearIntoPairRows(const T* X, float* pair, size_t r0, size_t rows, int in, int C, const std::string& w) {
  if constexpr (std::is_same_v<T, half>) if (PAIR16) { rowOut16(X, pairRow(pair, r0, C), rows, in, C, w); return; }
  needF32Pair("a precise-path product into the pair");
  linear<T, float>(X, pair + r0 * C, rows, in, C, w, false, 1.f);
}

// The 256-channel pair track's fused kernels (fused256.cuh, ESMFold2's): native/af3's at 128 channels hold
// a whole output tile or weight on the chip, which at 256 is past the registers and shared memory a block
// gets; these stream their weights in narrower steps, two blocks an SM. Below ~80 tokens their tiles leave
// the device idle (ESMFold2's measurement), and a T4's 64 KB fits none of them.
inline bool FUSED_WIDE = true;
inline int FUSED_WIDE_MIN_TOKENS = 80;
// ...and at OpenDDE's 384 and IntelliFold-2's 512 too, now (LOCALFOLD_NO_WIDER=1: the unfused paths there). They fit
// one block an SM there and once lost to the unfused path (262 tokens: trunk 2160 against 1925 ms, 3360 against 2821);
// re-measured 2026-10-07 on the kernels as they stand, each wins on its own - see README, "The wider pair tracks"
inline const bool FUSED_WIDER = !getenv("LOCALFOLD_NO_WIDER");
constexpr size_t wideTriInSmem(int C) { return (size_t)128 * (C + 8) * 2; }                       // 8 warps
constexpr size_t wideTriOutSmem(int C) { return (size_t)C * 65 * 4 + 2 * 64 * 4; }               // 4 warps
constexpr size_t wideUpSmem(int C) {
  return C == 384 ? transitionUpSmem<384, 8>() : C == 512 ? transitionUpSmem<512, 8>() : transitionUpSmem<256, 8>();
}
inline bool wideFits(int C) {
  return (C == 256 || (FUSED_WIDER && (C == 384 || C == 512))) && fitsSmem(std::max({wideTriInSmem(C), wideTriOutSmem(C), wideUpSmem(C)}));
}
// ...and the TRIANGLE's two at 128 channels, where the 128-channel fused kernels do not fit (a T4's 64 KB: the
// output kernel holds the whole 128 x 128 weight, 71 KB, where these stream it 16 columns a stage, ~35 KB)
// the input kernel's forms: 8 warps holding all their rows, else 16 warps norming their rows 32 at a time
// (TRIIN_ROUNDED: a T4 at 256 channels, where 8 warps' rows are 67.6 KB), else 4 warps. A row's
// arithmetic does not depend on the form, so all three are byte-identical.
constexpr int TRIIN_ROUNDED = 16, TRIIN_XROUNDS = 8;
constexpr size_t wideTriInSmemW(int C, int warps) {
  return warps == TRIIN_ROUNDED ? triIn256Smem<__nv_bfloat16>(C, warps, TRIIN_XROUNDS) : triIn256Smem<__nv_bfloat16>(C, warps);
}
inline int wideTriInWarps(int C) {
  static const int forced = getenv("LOCALFOLD_TRIIN_WARPS") ? atoi(getenv("LOCALFOLD_TRIIN_WARPS")) : 0;
  if (forced) return forced;
  // (measured on a Colab T4: at 256 channels the rounded form is 1768 -> 1477 ms of protenix2's input kernel
  // against the 4-warp one; at 128, where 8 warps fit, 428 -> 420 against them - so only where 8 do not fit,
  // and an L4, unmeasured, keeps its 8)
  if (fitsSmem(wideTriInSmemW(C, 8))) return 8;
  return fitsSmem(wideTriInSmemW(C, TRIIN_ROUNDED)) ? TRIIN_ROUNDED : 4;
}
// (the output kernel as triangleOutRun launches it: the bf16 tile at 16-column stages, either product type)
inline size_t wideTriOutSmemReal(int C) {
  auto at = [](auto width) {
    constexpr int CC = decltype(width)::value;
    return std::max(triangleOutSmem<CC, 4, __nv_bfloat16, 16, __nv_bfloat16>(), triangleOutSmem<CC, 4, __nv_bfloat16, 16, float>());
  };
  switch (C) {
    case 128: return at(std::integral_constant<int, 128>{});
    case 384: return at(std::integral_constant<int, 384>{});
    case 512: return at(std::integral_constant<int, 512>{});
    default: return at(std::integral_constant<int, 256>{});
  }
}
inline size_t wideTriFitsSmem(int C) { return std::max(wideTriInSmemW(C, 4), wideTriOutSmemReal(C)); }
template <class F> void wideWidth(int C, F f) {
  switch (C) {
    case 128: f(std::integral_constant<int, 128>{}); break;
    case 384: f(std::integral_constant<int, 384>{}); break;
    case 512: f(std::integral_constant<int, 512>{}); break;
    default: f(std::integral_constant<int, 256>{});
  }
}
template <class F> void wideWarps(int C, F f) {
  int w = wideTriInWarps(C);
  if (w == 8) f(std::integral_constant<int, 8>{});
  else if (w == TRIIN_ROUNDED) f(std::integral_constant<int, TRIIN_ROUNDED>{});
  else f(std::integral_constant<int, 4>{});
}
// triIn256K's template for a warp count (the rounded form at TRIIN_ROUNDED)
template <int CC, int WI, class TA, class PT = float> constexpr auto triIn256For() {
  if constexpr (WI == TRIIN_ROUNDED) return triIn256K<CC, WI, TA, TRIIN_XROUNDS, false, PT>; else return triIn256K<CC, WI, TA, 1, false, PT>;
}
inline bool bf16Tensor() {         // bf16 MMA: Ampere on (a T4's contraction stays f16 into f32)
  static int major = [] { int d, m; CK(cudaGetDevice(&d)); CK(cudaDeviceGetAttribute(&m, cudaDevAttrComputeCapabilityMajor, d)); return m; }();
  return major >= 8;
}

#include "tricontract.cuh"

// ---------------------------------------------------------------- the triangle multiplication in blocks
// A RECTANGLE of the padded pair space - rows [i0, i0 + I), columns [j0, j0 + J) - with q = (i - i0) J
// + (j - j0) its own index: the operands of one block of the contraction and its output live there.
struct TriRect { int i0, I, j0, J; size_t size() const { return (size_t)I * J; } };
__device__ __forceinline__ size_t rectPair(const TriRect& r, size_t q, int n) {     // the pair at q, SIZE_MAX for padding
  size_t i = r.i0 + q / r.J, j = r.j0 + q % r.J;
  return i < (size_t)n && j < (size_t)n ? i * n + j : SIZE_MAX;
}
// LayerNorm of rows [q0, q0 + cnt) of a rectangle, read from the pair where they lie (zeros for padding),
// and their mask - a warp a row
template <class TO, class PT = float>
__global__ void rectLayerNormK(const float* pair, const float* mask, TO* out, float* m, TriRect r, size_t q0, size_t cnt,
                               int n, int C, const float* scale, const float* offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= cnt) return;
  size_t p = rectPair(r, q0 + row, n);
  const bool live = p != SIZE_MAX;
  float s = 0, ss = 0;
  for (int c = lane; c < C; c += 32) { float v = live ? pairLd<PT>(pair, p * C + c) : 0.f; s += v; ss += v * v; }
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(~0u, s, o); ss += __shfl_xor_sync(~0u, ss, o); }
  float mean = s / C, inv = rsqrtf(ss / C - mean * mean + 1e-5f);
  for (int c = lane; c < C; c += 32)
    out[row * C + c] = fromF<TO>(((live ? pairLd<PT>(pair, p * C + c) : 0.f) - mean) * inv * scale[c] + offset[c]);
  if (lane == 0 && m) m[row] = p != SIZE_MAX ? mask[p] : 0.f;
}
// one operand from a chunk's (rows, 2C) [projection | gate] rows, channel-major into the rectangle's
// buffer at q0 + row - through a 32 x 32 tile, so the reads run along the channels and the writes along
// the rows
template <class T>
__global__ void rectGateK(const T* pg, const float* m, T* out, size_t q0, size_t cnt, int C, size_t size) {
  __shared__ float A[32][33];
  size_t row0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    size_t row = row0 + ry; int c = c0 + tx;
    float v = 0;
    if (row < cnt && c < C) { const T* p = pg + row * 2 * C; v = toF(p[c]) * m[row] * sigm(toF(p[C + c])); }
    A[ry][tx] = v;
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t row = row0 + tx; int c = c0 + cy;
    if (row < cnt && c < C) out[(size_t)c * size + q0 + row] = fromF<T>(A[tx][cy]);
  }
}
// center_norm of rows [q0, q0 + cnt) of a rectangle's product (channel-major), row-major out: the
// channel-major tile read along the rows, 32 rows a block
template <class TO>
__global__ void rectCenterNormK(const float* prod, TO* out, size_t q0, size_t cnt, int C, size_t size,
                                const float* scale, const float* offset) {
  __shared__ float ps[8][33], pss[8][33], mean[32], inv[32];
  size_t row0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
  size_t row = row0 + tx;
  float s = 0, ss = 0;
  if (row < cnt) for (int c = ty; c < C; c += 8) { float v = prod[(size_t)c * size + q0 + row]; s += v; ss += v * v; }
  ps[ty][tx] = s; pss[ty][tx] = ss;
  __syncthreads();
  if (ty == 0) {
    float a = 0, b = 0;
    for (int k = 0; k < 8; ++k) { a += ps[k][tx]; b += pss[k][tx]; }
    float mu = a / C; mean[tx] = mu; inv[tx] = rsqrtf(b / C - mu * mu + 1e-5f);
  }
  __syncthreads();
  // out[row][c]: each warp row of the block writes a row's channels; the reads of prod are strided by
  // `size`, a channel a lane - the norm's pass above was the coalesced one
  for (int ry = ty; ry < 32; ry += 8) {
    size_t lr = row0 + ry; if (lr >= cnt) continue;
    for (int c = tx; c < C; c += 32)
      out[lr * C + c] = fromF<TO>((prod[(size_t)c * size + q0 + lr] - mean[ry]) * inv[ry] * scale[c] + offset[c]);
  }
}
// pair[at q] += t1 * sigmoid(t2) for the chunk's real rows
template <class T, class PT = float>
__global__ void rectGatedAddK(float* pair, const T* t1, const T* t2, TriRect r, size_t q0, size_t cnt, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  size_t row = t / C; int c = (int)(t % C);
  size_t p = rectPair(r, q0 + row, n);
  if (p != SIZE_MAX) pairSt<PT>(pair, p * C + c, pairLd<PT>(pair, p * C + c) + toF(t1[t]) * sigm(toF(t2[t])));
}

// [projection | gate] of ONE operand (side 0 = a, 1 = b) as a (C, 2C) matrix, from the interleaved (C, 2C)
// projection and gate (column 2ch is a's channel ch, 2ch + 1 b's)
__global__ void operandWeightK(const float* proj, const float* gate, float* out, int C, int side) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)C * 2 * C) return;
  int k = (int)(t / (2 * C)), o = (int)(t % (2 * C));
  out[t] = o < C ? proj[(size_t)k * 2 * C + 2 * o + side] : gate[(size_t)k * 2 * C + 2 * (o - C) + side];
}
inline std::string operandWeight(const std::string& pre, int C, int side) {
  std::string key = pre + ".operand" + std::to_string(side) + "~";
  if (!WF.count(key) && !WH.count(key)) {
    float* d = dalloc((size_t)C * 2 * C);
    operandWeightK<<<blocks((size_t)C * 2 * C), 256, 0, STREAM>>>(W(pre + ".projection"), W(pre + ".gate"), d, C, side);
    deviceWeight(key, d, (size_t)C * 2 * C);
  }
  return key;
}
// The whole triangle multiplication in BLOCKS, for a card short of room: the fixed operand b built whole
// (np^2 x C), then per block of OUTPUT rows (outgoing) or columns (incoming) the free operand a, the
// contraction and the output on that block alone - never a LayerNorm'd plane, a whole a or a whole
// product. Outgoing P[i, j] = sum_k a[i, k] b[j, k]: a row block reads pair rows the earlier blocks did
// not write. Incoming P[i, j] = sum_k a[k, j] b[k, i]: a COLUMN block reads pair columns the earlier
// blocks did not write - a row block would read rows they had. The reduction over k is never split.
template <class T>
void triangleBlocked(float* pair, const float* mask, int n, int C, const std::string& pre, bool outgoing,
                     bool divideByLength, int np) {
  size_t cs = (size_t)np * np;
  // each operand's own half of [projection | gate], so the two passes together project once
  std::string pgOf[2] = { operandWeight(pre, C, 0), operandWeight(pre, C, 1) };
  float alpha = divideByLength ? 1.f / n : 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  size_t per = std::max<size_t>(32, CHUNK / (4 * C));            // rows a chunk of the row-wise steps
  float* m = scratch<float>("trib.mask", per);
  T* ln = scratch<T>("trib.ln", per * C); T* pgOut = scratch<T>("trib.pg", per * 2 * C);
  // LN(pair) -> [projection | gate] -> one operand (side 0 = a, 1 = b), over a rectangle in chunks
  auto operands = [&](const TriRect& r, T* out, int side) {
    for (size_t q0 = 0; q0 < r.size(); q0 += per) {
      size_t cnt = std::min(per, r.size() - q0);
      WITH_PAIR_T(rectLayerNormK<T, PT><<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(pair, mask, ln, m, r, q0, cnt, n, C,
        W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset")));
      linear<T, T>(ln, pgOut, cnt, C, 2 * C, pgOf[side]);
      rectGateK<T><<<dim3((unsigned)((cnt + 31) / 32), (C + 31) / 32), dim3(32, 8), 0, STREAM>>>(pgOut, m, out, q0, cnt, C, r.size());
    }
  };
  T* b = scratch<T>("trib.b", cs * C);
  operands({0, np, 0, np}, b, 1);
  // a block of the free operand and its product: at least CHUNK / C rows of the rectangle, and as many more (up
  // to 1024) as what is free beside b allows - every block reads the whole of b (np^2 x C), so narrow blocks
  // are bound by re-reading it: at 10761 tokens 48-row blocks read 29.6 GB 224 times a triangle, 4.4 s of
  // memory traffic for 1.1 s of arithmetic
  int width = (int)std::max<size_t>(8, std::min<size_t>(np, (CHUNK / C) / np / 8 * 8));
  {
    size_t f, t; CK(cudaMemGetInfo(&f, &t));
    size_t perRow = (size_t)np * C * (sizeof(T) + 4), spare = f > t / 16 ? f - t / 16 : 0;    // a and prod, a row each
    width = (int)std::max<size_t>(width, std::min<size_t>(np, std::min<size_t>(spare / perRow, 1024) / 8 * 8));
  }
  T* a = scratch<T>("trib.a", (size_t)width * np * C);
  float* prod = scratch<float>("trib.prod", (size_t)width * np * C);
  T* t1 = scratch<T>("trib.t1", per * C); T* t2 = scratch<T>("trib.t2", per * C);
  for (int k0 = 0; k0 < n; k0 += width) {
    int w = std::min(width, np - k0);
    TriRect r = outgoing ? TriRect{k0, w, 0, np} : TriRect{0, np, k0, w};
    operands(r, a, 0);
    if (outgoing)          // (as the whole contraction, with w output rows)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, np, w, np, &alpha, b, cudaType<T>(), np, cs, a,
        cudaType<T>(), np, r.size(), &zero, prod, CUDA_R_32F, np, r.size(), C, CUBLAS_COMPUTE_32F, algo));
    else                   // (with w output columns)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, w, np, np, &alpha, a, cudaType<T>(), w, r.size(), b,
        cudaType<T>(), np, cs, &zero, prod, CUDA_R_32F, w, r.size(), C, CUBLAS_COMPUTE_32F, algo));
    for (size_t q0 = 0; q0 < r.size(); q0 += per) {
      size_t cnt = std::min(per, r.size() - q0);
      rectCenterNormK<T><<<(unsigned)((cnt + 31) / 32), dim3(32, 8), 0, STREAM>>>(prod, ln, q0, cnt, C, r.size(),
        W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"));
      linear<T, T>(ln, t1, cnt, C, C, pre + ".outputProjection");
      WITH_PAIR_T(rectLayerNormK<T, PT><<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(pair, mask, ln, nullptr, r, q0, cnt, n, C,
        W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset")));
      linear<T, T>(ln, t2, cnt, C, C, pre + ".gatingLinear");
      WITH_PAIR_T(rectGatedAddK<T, PT><<<blocks(cnt * C), 256, 0, STREAM>>>(into(pair), t1, t2, r, q0, cnt, n, C));
    }
  }
  // given back at once: the fixed operand is a plane, and the next stage (the MSA attention, the grid
  // attention) peaks beside the pair too - at these sizes a reallocation a call is nothing
  releaseScratch({ "trib." });
}
template <class T>
void triangle(float* pair, const float* mask, int n, int C, const std::string& pre, bool outgoing,
              bool divideByLength) {
  size_t pairs = (size_t)n * n;
  // a, b and their product live in a PADDED np x np space per channel, np a multiple of 8, the
  // padding zero: the contraction is unchanged and cuBLAS's GEMM runs twice as fast on an aligned
  // size (1044: 5.4 against 2.7 ms the pair of them; 522 the same). A multiple of 8 is enough for
  // the GEMM, and 32 padded the fused kernels' rows too (68 tokens: 96^2 against 72^2): trunk
  // 76.6 -> 72.3 ms at 68 tokens, 442 -> 438 at 261, 1617 -> 1606 at 522, flat at 150 and 1044
  int np = TRI_PAD ? (n + TRI_PAD - 1) / TRI_PAD * TRI_PAD : n;
  size_t cs = (size_t)np * np;
  // on a card short of room, in blocks of the output (triangleBlocked): no whole a, product or
  // LayerNorm'd plane - whichever kernels the device would otherwise run
  // (blocking costs time - a fifth of a trunk at 2096 tokens - so only where the whole form's operands,
  // product and gate, five planes, would not fit with room to spare)
  // (whichever form runs gives back the other's buffers first: scratch outlives the call, so a whole form
  // taken while there was room would otherwise sit beside the blocks of the next call, which had none)
  if (shortPair(pairs, C) || TIGHT_STACK) {
    // (TIGHT_STACK: a narrower stack - the template's 64 channels - on a trunk short of room, where its own pair is
    // under the threshold but the card is not: its whole form, 1.5 GB at 1530 tokens, was what ran out)
    if (TIGHT_STACK || !roomFor(5 * cs * C * 2, { "tri.a", "tri.b", "tri.prod", "tri.norm", "tri.abf", "tri.bbf", "tri.pbf", "tri.t2whole" })) {
      releaseScratch({ "tri.a", "tri.b", "tri.prod", "tri.norm", "tri.abf", "tri.bbf", "tri.pbf", "tri.t2whole" });
      triangleBlocked<T>(pair, mask, n, C, pre, outgoing, divideByLength, np);
      return;
    }
    releaseScratch({ "trib." });
  }
  std::string pg = projectionGate(pre, C);
  T *a = nullptr, *b = nullptr;
  float* prod = nullptr;          // f32: in f16 the contraction (a sum over n of products) overflows
  auto buffers = [&]() {
    a = scratch<T>("tri.a", cs * C); b = scratch<T>("tri.b", cs * C); prod = scratch<float>("tri.prod", cs * C);
    if (np != n) {                // the padding is written by nothing here
      CK(cudaMemsetAsync(a, 0, cs * C * sizeof(T), STREAM)); CK(cudaMemsetAsync(b, 0, cs * C * sizeof(T), STREAM));
    }
  };
  float alpha = divideByLength ? 1.f / n : 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  auto contract = [&]() {         // one n x n GEMM per channel: outgoing P = A B^T, incoming P = B^T A
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, np, np, np, &alpha, b, cudaType<T>(), np,
        cs, a, cudaType<T>(), np, cs, &zero, prod, CUDA_R_32F, np, cs, C, CUBLAS_COMPUTE_32F, algo));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, np, np, np, &alpha, a, cudaType<T>(), np,
        cs, b, cudaType<T>(), np, cs, &zero, prod, CUDA_R_32F, np, cs, C, CUBLAS_COMPUTE_32F, algo));
  };
  if constexpr (std::is_same_v<T, half>) {
    bool narrowFused = C == 128 && (TRI_BF16 ? triFusedFits<__nv_bfloat16>() : triFusedFits<float>());
    // (384 and 512 channels - OpenDDE, IntelliFold-2 - too: their unfused triangle was the LN, a [C, 4C] GEMM, the gate,
    // the centre norm, two GEMMs and a gated add; LOCALFOLD_NO_WIDER=1 keeps it)
    bool wide = (C == 256 || (FUSED_WIDER && (C == 384 || C == 512)) || (C == 128 && !narrowFused)) && fitsSmem(wideTriFitsSmem(C));
    if (FUSED_WIDE && FUSED_TRIANGLE && n >= FUSED_WIDE_MIN_TOKENS && wide) {
      // LN, the projection, the gate and the gating linear in one kernel (writing the padding), the f16
      // contraction into f32, then the centre norm, the output projection, the gate and the residual
      half* t2 = scratch<half>("tri.t2whole", cs * C);
      half* wt = scratch<half>("tri.wt256", triInTileHalves(C));
      tileTriIn(Wh(pg), Wh(pre + ".gatingLinear"), C, 16, wt);
      if (TRI_BF16 && bf16Tensor()) {
        // as the 128-channel path: a, b and the product in bf16, the product half the bytes both ways
        __nv_bfloat16* ab = scratch<__nv_bfloat16>("tri.abf", cs * C);
        __nv_bfloat16* bb = scratch<__nv_bfloat16>("tri.bbf", cs * C);
        __nv_bfloat16* pb = scratch<__nv_bfloat16>("tri.pbf", cs * C);
        wideWidth(C, [&](auto width) {
          constexpr int CC = decltype(width)::value, WO = 4;
          wideWarps(C, [&](auto warps) {
            constexpr int WI = decltype(warps)::value;
            WITH_PAIR_T(
              static bool attr = false;
              constexpr auto kern = triIn256For<CC, WI, __nv_bfloat16, PT>();
              if (!attr) { smemAttr(kern, (int)wideTriInSmemW(CC, WI)); attr = true; }
              kern<<<(unsigned)((cs + 16 * WI - 1) / (16 * WI)), 32 * WI, wideTriInSmemW(CC, WI), STREAM>>>(
                pair, mask, W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset"), wt, ab, bb, t2, n, np, cs, nullptr));
          });
          triContractBf16(outgoing, np, cs, C, alpha, ab, bb, pb);
          triangleOutRun<CC, WO, __nv_bfloat16>(pb, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"),
                                                Wh(pre + ".outputProjection"), t2, into(pair), n, np);
        });
        return;
      }
      a = scratch<T>("tri.a", cs * C); b = scratch<T>("tri.b", cs * C); prod = scratch<float>("tri.prod", cs * C);
      wideWidth(C, [&](auto width) {
        constexpr int CC = decltype(width)::value, WO = 4;
        wideWarps(C, [&](auto warps) {
          constexpr int WI = decltype(warps)::value;
          WITH_PAIR_T(
            static bool attr = false;
            constexpr auto kern = triIn256For<CC, WI, half, PT>();
            if (!attr) { smemAttr(kern, (int)wideTriInSmemW(CC, WI)); attr = true; }
            kern<<<(unsigned)((cs + 16 * WI - 1) / (16 * WI)), 32 * WI, wideTriInSmemW(CC, WI), STREAM>>>(
              pair, mask, W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset"), wt, a, b, t2, n, np, cs, nullptr));
        });
        contract();
        triangleOutRun<CC, WO>(prod, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"), Wh(pre + ".outputProjection"), t2, into(pair), n, np);
      });
      return;
    }
    if (FUSED_TRIANGLE && C == 128 && (TRI_BF16 ? triFusedFits<__nv_bfloat16>() : triFusedFits<float>())) {   // see fusedtriangle.cuh
      half* t2 = scratch<half>("tri.t2whole", cs * C);
      if (TRI_BF16) {
        // a, b and the contraction's product in bf16: f32's range at half the bytes (f16's
        // range is what overflowed), AF3's own activation precision
        __nv_bfloat16* ab = scratch<__nv_bfloat16>("tri.abf", cs * C);
        __nv_bfloat16* bb = scratch<__nv_bfloat16>("tri.bbf", cs * C);
        __nv_bfloat16* pb = scratch<__nv_bfloat16>("tri.pbf", cs * C);
        triIn128(pair, mask, pre, pg, ab, bb, t2, n, np, cs);
        triContractBf16(outgoing, np, cs, C, alpha, ab, bb, pb);
        triOut128(pb, pre, t2, into(pair), n, np, cs);
        return;
      }
      a = scratch<T>("tri.a", cs * C); b = scratch<T>("tri.b", cs * C); prod = scratch<float>("tri.prod", cs * C);
      triIn128(pair, mask, pre, pg, a, b, t2, n, np, cs);   // writes the padding itself
      contract();
      triOut128(prod, pre, t2, into(pair), n, np, cs);
      return;
    }
  }
  buffers();
  T* norm = scratch<T>("tri.norm", pairs * C);
  size_t rowsPer = std::max<size_t>(1, CHUNK / (4 * C));
  if (pairs > rowsPer && roomFor(pairs * 4 * C * sizeof(T), {"tri.pg"})) rowsPer = pairs;   // (as the transition's)
  T* pgOut = scratch<T>("tri.pg", std::min(rowsPer, pairs) * 4 * C);
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t rows = std::min(rowsPer, pairs - r0);
    lnPairRows<T>(pair, r0, norm + r0 * C, rows, C, pre + ".leftNormInputScale", pre + ".leftNormInputOffset");
    linear<T, T>(norm + r0 * C, pgOut, rows, C, 4 * C, pg);
    triGateK<T><<<dim3((unsigned)((rows + 31) / 32), (C + 31) / 32), dim3(32, 8), 0, STREAM>>>(
      pgOut, mask, a, b, r0, rows, C, cs, n, np);
  }
  contract();
  rowsPer = std::max<size_t>(1, CHUNK / C);
  T* centred = scratch<T>("tri.centred", std::min(rowsPer, pairs) * C);
  T* t1 = scratch<T>("tri.t1", std::min(rowsPer, pairs) * C);
  T* t2 = scratch<T>("tri.t2", std::min(rowsPer, pairs) * C);
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t rows = std::min(rowsPer, pairs - r0);
    if (fitsSmem((size_t)C * 33 * 4)) {   // C * 33 floats of shared memory: past the 48 KB default from C = 373 (IntelliFold-2's 512)
      static int granted = 0;
      if (C * 33 * 4 > granted) {
        smemAttr((centerNormK<T>), C * 33 * 4);
        granted = C * 33 * 4;
      }
      centerNormK<T><<<(unsigned)((rows + 31) / 32), dim3(32, 8), C * 33 * 4, STREAM>>>(prod, centred, r0,
        rows, C, cs, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"), n, np);
    } else {
      centerNormStreamK<T><<<(unsigned)((rows + 31) / 32), dim3(32, 8), 0, STREAM>>>(prod, centred, r0,
        rows, C, cs, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"), n, np);
    }
    linear<T, T>(centred, t1, rows, C, C, pre + ".outputProjection");
    linear<T, T>(norm + r0 * C, t2, rows, C, C, pre + ".gatingLinear");
    WITH_PAIR_T(gatedAddK<T, PT><<<blocks(rows * C), 256, 0, STREAM>>>(pairRow(into(pair), r0, C), t1, t2, rows * C));
  }
}

// ---------------------------------------------------------------- transition
#include "fusedtransition.cuh"
template <class T>
void transition(float* x, size_t rows, int C, int factor, const std::string& pre) {
  // the width is the weight's (OpenDDE's refiner is factor 2 where its trunk is 4)
  (void)factor;
  size_t w1 = lenW(pre + ".transition1");
  if (w1 % (2 * (size_t)C)) { fprintf(stderr, "%s.transition1 has %zu elements, not C %d x 2I\n", pre.c_str(), w1, C); exit(1); }
  int I = (int)(w1 / (2 * (size_t)C));
  // (the fused 128-channel kernel adds in place, so not under a redirected residual - nothing parallel is 128 wide)
  if constexpr (std::is_same_v<T, half>) if (!RESIDUAL_INTO && fusedTransition(x, rows, C, I, pre)) return;
  if constexpr (std::is_same_v<T, half>) {
    // 256 channels: LN, the widening and SwiGLU in one kernel (fused256.cuh), then the second GEMM with the
    // residual as its beta - the [rows, 2I] widening never written
    if (FUSED_WIDE && FUSED_TRANSITION && rows >= (size_t)FUSED_WIDE_MIN_TOKENS * FUSED_WIDE_MIN_TOKENS && wideFits(C)) {
      constexpr int WU = 8, R = 16 * WU;
      // whole waves of transitionUpK inside the same budget (transitionUpChunkRows)
      size_t rowsPer = transitionUpChunkRows<256, WU>(wideUpSmem(C), std::max<size_t>(R, CHUNK / (2 * I)));
      half* w1t = scratch<half>("tr.w1t", (size_t)2 * C * I);
      tileTransitionUp(Wh(pre + ".transition1"), C, I, w1t);
      wideWidth(C, [&](auto width) {
        constexpr int CC = decltype(width)::value;
        if (PAIR16) {
          // a bf16 pair: the gated rows in bf16 and the second GEMM bf16 throughout, accumulating straight into the
          // pair - no f16 product and add pass
          using B16 = __nv_bfloat16;
          B16* gated = scratch<B16>("tr.gatedbf", std::min(rowsPer, rows) * I);
          static bool attr = false;
          if (!attr) { smemAttr((transitionUpK<CC, WU, 32, 1, B16, B16>), (int)wideUpSmem(CC)); attr = true; }
          for (size_t r0 = 0; r0 < rows; r0 += rowsPer) {
            size_t r = std::min(rowsPer, rows - r0);
            transitionUpK<CC, WU, 32, 1, B16, B16><<<(unsigned)((r + R - 1) / R), 32 * WU, wideUpSmem(CC), STREAM>>>(
              pairRow(x, r0, C), W(pre + ".inputLayerNormScale"), W(pre + ".inputLayerNormOffset"), w1t, gated, r, I);
            linear<B16, B16>(gated, reinterpret_cast<B16*>(pairRow(into(x), r0, C)), r, I, C, pre + ".transition2", false, 1.f);
          }
          return;
        }
        half* gated = scratch<half>("tr.gated", std::min(rowsPer, rows) * I);
        static bool attr = false;
        if (!attr) { smemAttr((transitionUpK<CC, WU>), (int)wideUpSmem(CC)); attr = true; }
        for (size_t r0 = 0; r0 < rows; r0 += rowsPer) {
          size_t r = std::min(rowsPer, rows - r0);
          transitionUpK<CC, WU><<<(unsigned)((r + R - 1) / R), 32 * WU, wideUpSmem(CC), STREAM>>>(
            x + r0 * C, W(pre + ".inputLayerNormScale"), W(pre + ".inputLayerNormOffset"), w1t, gated, r, I);
          linear<half, float>(gated, into(x) + r0 * C, r, I, C, pre + ".transition2", false, 1.f);
        }
      });
      return;
    }
    // (a T4's form of it - transitionUpK<256, 16, 16, 8>, rows normed in rounds, ~49 KB - is what ESMFold2 takes
    // there; here it measured LEVEL on a Colab T4: protenix2's fused transition 1083 ms against the LN, GEMM and
    // SwiGLU passes' ~1100, the folds within the card's drift. Not taken.)
  }
  // every row in one pass where the card has the room (OpenDDE at 255 tokens: the trunk's transitions 370 -> 353
  // ms for 0.6 GB; chunks of 1-8k rows, small enough for L2 to hold the widening, are 397-615 - the GEMMs lose more)
  size_t rowsPer = std::max<size_t>(1, CHUNK / (2 * I));
  if (rows > rowsPer && roomFor(rows * (C + 3 * (size_t)I) * sizeof(T), {"tr.x", "tr.wide", "tr.gated"})) rowsPer = rows;
  T* xn = scratch<T>("tr.x", std::min(rowsPer, rows) * C);
  T* wide = scratch<T>("tr.wide", std::min(rowsPer, rows) * 2 * I);
  T* gated = scratch<T>("tr.gated", std::min(rowsPer, rows) * I);
  for (size_t r0 = 0; r0 < rows; r0 += rowsPer) {
    size_t r = std::min(rowsPer, rows - r0);
    lnPairRows<T>(x, r0, xn, r, C, pre + ".inputLayerNormScale", pre + ".inputLayerNormOffset");
    linear<T, T>(xn, wide, r, C, 2 * I, pre + ".transition1");
    swiglu<T>(wide, gated, r, I);
    linearIntoPairRows<T>(gated, into(x), r0, r, I, C, pre + ".transition2");
  }
}

// ---------------------------------------------------------------- grid attention kernels
#include "flash.cuh"

// act[r][j] = norm[j][r] for the column direction, 16 bytes a thread (C * elem a multiple of 16)
__global__ void gatherTransposedK(const void* normV, void* actV, int n, int C, size_t r0, size_t R,
                                  int elem) {
  int chunks = C * elem / 16;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * n * chunks) return;
  int c = (int)(t % chunks); size_t rest = t / chunks; size_t j = rest % n; size_t r = r0 + rest / n;
  reinterpret_cast<uint4*>(actV)[t] = reinterpret_cast<const uint4*>(normV)[(j * n + r) * chunks + c];
}
// bias[h][i][stride] (j padded to a multiple of 8, zeros past n), scaled by `scale`
template <class TB>
__global__ void biasLayoutK(const float* raw, TB* bias, int n, int stride, int heads, bool swap,
                            float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)heads * n * stride;
  if (t >= total) return;
  int j = (int)(t % stride); size_t rest = t / stride; int i = (int)(rest % n); int h = (int)(rest / n);
  float v = j < n ? scale * raw[(swap ? ((size_t)j * n + i) : ((size_t)i * n + j)) * heads + h] : 0.f;
  bias[t] = fromF<TB>(v);
}
// the same from head-major raw logits [hp][pairs] (hp >= heads, the padded projection's width)
template <class TB>
__global__ void biasLayoutHeadMajorK(const float* raw, TB* bias, int n, int stride, int heads, bool swap,
                                     float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)heads * n * stride, pairs = (size_t)n * n;
  if (t >= total) return;
  int j = (int)(t % stride); size_t rest = t / stride; int i = (int)(rest % n); int h = (int)(rest / n);
  float v = j < n ? scale * raw[(size_t)h * pairs + (swap ? ((size_t)j * n + i) : ((size_t)i * n + j))] : 0.f;
  bias[t] = fromF<TB>(v);
}
// the same from a chunk of pairs [r0, r0 + cnt) of raw (head-major over the chunk: [heads'][cnt]) - the
// padding columns are the caller's to zero
template <class TB>
__global__ void biasFromRawRowsK(const float* raw, TB* bias, size_t r0, size_t cnt, int n, int stride, int heads,
                                 bool swap, float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * cnt) return;
  size_t h = t / cnt, q = t % cnt, p = r0 + q, a = p / n, b = p % n;      // pair p is (a, b)
  size_t i = swap ? b : a, j = swap ? a : b;
  bias[(h * n + i) * stride + j] = fromF<TB>(scale * raw[h * cnt + q]);
}
// a (C, k) weight zero-padded to (C, kp) columns
inline std::string paddedColumns(const std::string& w, int C, int k, int kp) {
  return concatColumns(w + "~pad" + std::to_string(kp), C, {{w, k, false}, {"", kp - k, false}});
}
// pair[(r, j) or (j, r)] += out[r][j], four channels a thread (C a multiple of 4)
template <class PT = float, class TO = float>
__global__ void addGridK(float* pair, const TO* out, int n, int C, size_t r0, size_t R, bool tr) {
  int c4 = C / 4;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * n * c4) return;
  int c = (int)(t % c4); size_t rest = t / c4; size_t j = rest % n; size_t r = r0 + rest / n;
  size_t to = tr ? (j * n + r) : (r * n + j);
  float4 v = pairLd4<PT>(pair, (to * c4 + c) * 4), o;
  if constexpr (std::is_same_v<TO, float>) o = reinterpret_cast<const float4*>(out)[t];
  else { o.x = toF(out[t * 4]); o.y = toF(out[t * 4 + 1]); o.z = toF(out[t * 4 + 2]); o.w = toF(out[t * 4 + 3]); }
  v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
  pairSt4<PT>(pair, (to * c4 + c) * 4, v);
}
template <class T>
__global__ void addGateBiasK(T* qkvg, const float* bias, size_t rows, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * Wd) return;
  size_t r = t / Wd; int c = (int)(t % Wd);
  T& g = qkvg[r * 4 * Wd + 3 * Wd + c];
  g = fromF<T>(toF(g) + bias[c]);
}

// The unfused grid attention's LayerNorm and its pair-bias projection in one pass: the pair normed as
// layerNormK norms it (the same arithmetic, so `norm` is byte-identical) into `norm` AND shared memory, then
// projected on the tensor cores to 16 padded heads, head-major (out[h][row]) - where cuBLAS took the
// few-column GEMM as a 16x16 WMMA kernel that read the whole normed pair back.
template <int C, int WARPS, class PT = float>
__global__ void __launch_bounds__(WARPS * 32) lnNormHeadsK(const float* __restrict__ x, const float* __restrict__ scale,
    const float* __restrict__ offset, const half* __restrict__ Wp, half* __restrict__ norm, float* __restrict__ out,
    size_t rows, int heads) {
  constexpr int N = 16, R = 16 * WARPS, LDX = C + 8, LDW = N + 8, KS = C / 16, K = C / 32;
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem; half* Ws = Xs + R * LDX;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  for (int t = threadIdx.x; t < C * (N / 8); t += WARPS * 32) {
    int k = t / (N / 8), c = (t % (N / 8)) * 8;
    cpAsync16(Ws + k * LDW + c, Wp + (size_t)k * N + c, true);
  }
  cpCommit();
  float sc[K], of[K];
#pragma unroll
  for (int k = 0; k < K; ++k) { sc[k] = scale[lane + 32 * k]; of[k] = offset[lane + 32 * k]; }
  // the warp's own 16 rows, eight in flight (Chai-1 at 255 tokens: 32.8 -> 28.1 ms over a fold's 320 calls at
  // four, 16 no better; four at 512 channels, where eight rows are 128 registers)
  constexpr int B = C >= 512 ? 4 : 8;
#pragma unroll 1
  for (int i0 = 0; i0 < 16; i0 += B) {
    float v[B][K];
#pragma unroll
    for (int b = 0; b < B; ++b) {
      size_t row = row0 + warp * 16 + i0 + b;
#pragma unroll
      for (int k = 0; k < K; ++k) v[b][k] = row < rows ? pairLd<PT>(x, row * C + lane + 32 * k) : 0.f;
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
      int r = warp * 16 + i0 + b;
      size_t row = row0 + r;
      float s = 0, ss = 0;
#pragma unroll
      for (int k = 0; k < K; ++k) { s += v[b][k]; ss += v[b][k] * v[b][k]; }
      for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(~0u, s, o); ss += __shfl_xor_sync(~0u, ss, o); }
      float mean = s / C, inv = rsqrtf(ss / C - mean * mean + 1e-5f);
#pragma unroll
      for (int k = 0; k < K; ++k) {
        half h = __float2half((v[b][k] - mean) * inv * sc[k] + of[k]);
        if (row >= rows) h = __float2half(0.f);
        Xs[r * LDX + lane + 32 * k] = h;
        if (row < rows) norm[row * C + lane + 32 * k] = h;
      }
    }
  }
  cpWait<0>();
  __syncthreads();
  float acc[N / 8][4] = {};
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    uint32_t xa[4];
    ldsm4(xa, Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
    uint32_t f[4];
    ldsm4t(f, Ws + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW + (lane >> 4) * 8);
    mma16816(acc[0], xa, f[0], f[1]); mma16816(acc[1], xa, f[2], f[3]);
  }
  size_t r0 = row0 + warp * 16 + g, r1 = r0 + 8;
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    int h = nt * 8 + tig * 2;
    if (h < heads) { if (r0 < rows) out[(size_t)h * rows + r0] = acc[nt][0]; if (r1 < rows) out[(size_t)h * rows + r1] = acc[nt][2]; }
    if (h + 1 < heads) { if (r0 < rows) out[(size_t)(h + 1) * rows + r0] = acc[nt][1]; if (r1 < rows) out[(size_t)(h + 1) * rows + r1] = acc[nt][3]; }
  }
}
template <int C> constexpr size_t lnNormHeadsSmem(int warps) { return (size_t)16 * warps * (C + 8) * 2 + (size_t)C * 24 * 2; }
// norm + head-major raw bias for C 256/384/512 (false: not this width, or the device's shared memory)
inline bool LN_NORM_HEADS = true;
inline bool lnNormHeads(const float* pair, half* norm, float* raw, size_t rows, int C, int heads, const std::string& pre) {
  if (!LN_NORM_HEADS || heads > 16) return false;
  auto run = [&](auto width) -> bool {
    constexpr int CC = decltype(width)::value, WARPS = 4, R = 16 * WARPS;   // (8 warps: no faster)
    constexpr size_t smem = lnNormHeadsSmem<CC>(WARPS);
    if (!fitsSmem(smem)) return false;
    std::string wb = concatColumns(pre + ".pairBiasProjection~pad16", CC, {{pre + ".pairBiasProjection", heads, false}, {"", 16 - heads, false}});
    WITH_PAIR_T(
      static bool attr = false;
      if (!attr) { smemAttr((lnNormHeadsK<CC, WARPS, PT>), (int)smem); attr = true; }
      lnNormHeadsK<CC, WARPS, PT><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
        pair, W(pre + ".actNormScale"), W(pre + ".actNormOffset"), Wh(wb), norm, raw, rows, heads));
    return true;
  };
  switch (C) {
    case 256: return run(std::integral_constant<int, 256>{});
    case 384: return run(std::integral_constant<int, 384>{});
    case 512: return run(std::integral_constant<int, 512>{});
    default: return false;
  }
}
// the pair's row `row` (of C channels) as a float* base in whichever storage PAIR16 says it has
// the unfused column direction's two GEMMs strided in place (LOCALFOLD_GRID_STRIDED=0: gathered and scattered)
inline bool GRID_STRIDED = true;
// Grid attention over the pair, rows (tr = false) or columns (tr = true), residual added.
template <class T>
void gridAttention(float* pair, const float* mask, int n, int C, int heads, int D,
                   const std::string& pre, bool tr, bool swapBias) {
  size_t pairs = (size_t)n * n; int Wd = heads * D;
  if constexpr (std::is_same_v<T, half>) {
    // three fused kernels (fusedtriangle.cuh): LN + the bias projection (head-major), then per
    // chunk LN + q/k/v/gate (reading the column direction's rows transposed in place), the flash
    // kernel, and the output projection added into the pair
    if (FUSED_GRID && C == 128 && Wd == 128 && heads <= 16 && gridFusedFits() && !RESIDUAL_UNTRANSPOSED && !hasW(pre + ".gatingQueryBias") &&
        !hasW(pre + ".outputProjectionBias")) {
      int stride = (n + 7) / 8 * 8;
      half* bias = scratch<half>("grid.bias", (size_t)heads * n * stride);
      std::string qkvg = qkvgWeight(pre, C, Wd, true);
      std::string wb = paddedColumns(pre + ".pairBiasProjection", C, heads, 16);
      float scale = 1.f / sqrtf((float)D);
      // 🔴 every row in one pass whenever the card has the room for it (what its buffers already hold
      // counted, so every pass decides alike): whole is 3.3% of the trunk faster at 262 tokens and 4.8% at
      // 1048 (465 against 481 ms, 7.82 against 8.20 s on an A100) for 1.13 GB the chunks do not hold - it
      // was capped at a fixed 32nd of the card, which chunked from ~1,100 tokens on 40 GB with 25 GB free.
      // The chunks only where the memory is short (LOCALFOLD_BIG=1 forces them, roomFor)
      // (the room asked for includes the triangle multiplication's whole form, five planes, which these
      // buffers would otherwise starve into its blocked form - see native/af2's twin)
      size_t triPlane = (size_t)((n + 7) / 8 * 8) * ((n + 7) / 8 * 8);
      if (roomFor(((pairs + 128) * 4 * Wd + pairs * Wd) * 2 + 5 * triPlane * C * 2,
                  {"grid.qkvg", "grid.gathered", "tri.a", "tri.b", "tri.prod", "tri.norm", "tri.abf", "tri.bbf", "tri.pbf", "tri.t2whole"})) {
        // every row in one pass (this card has the memory): the bias written by the same kernel
        CK(cudaMemsetAsync(bias, 0, (size_t)heads * n * stride * 2, STREAM));    // the padding columns
        half* qkvgOut = scratch<half>("grid.qkvg", (pairs + 128) * 4 * Wd);
        gridIn128(pair, pre, qkvg, qkvgOut, n, 0, pairs, tr, Wh(wb), bias, heads, stride, tr && swapBias);
        half* gathered = scratch<half>("grid.gathered", pairs * Wd);
        flashGrid<half>(qkvgOut, bias, stride, MASK_ALL_ONES ? nullptr : mask, gathered, n, heads, D, 0, n, tr, scale);
        // (a bf16 pair: cuBLAS takes no f16-in, bf16-out GEMM, so the projection goes to f16 and one pass adds it)
        if (!tr && PAIR16) rowOut16(gathered, into(pair), pairs, Wd, C, pre + ".outputProjection");
        else if (!tr) linear<half, float>(gathered, into(pair), pairs, Wd, C, pre + ".outputProjection", false, 1.f);
        else gridOut128(gathered, pre + ".outputProjection", into(pair), n, 0, pairs, tr);
        return;
      }
      if (shortPair(pairs, C)) {
        // on a card short of room the 16-column projection in chunks of pairs, each laid into the bias
        size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / 16));
        float* raw = scratch<float>("grid.raw16", per * 16);
        CK(cudaMemsetAsync(bias, 0, (size_t)heads * n * stride * 2, STREAM));     // the padding columns
        for (size_t r0 = 0; r0 < pairs; r0 += per) {
          size_t r = std::min(per, pairs - r0);
          lnHeads128<16>(pairRow(pair, r0, C), pre + ".actNormScale", pre + ".actNormOffset", wb, raw, r);
          biasFromRawRowsK<half><<<blocks((size_t)heads * r), 256, 0, STREAM>>>(raw, bias, r0, r, n, stride, heads,
                                                                              tr && swapBias, LOG2E);
        }
      } else {
      float* raw = scratch<float>("grid.raw16", pairs * 16);
      lnHeads128<16>(pair, pre + ".actNormScale", pre + ".actNormOffset", wb, raw, pairs);
      biasLayoutHeadMajorK<half><<<blocks((size_t)heads * n * stride), 256, 0, STREAM>>>(
        raw, bias, n, stride, heads, tr && swapBias, LOG2E);
      }
      size_t R = std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)n * 4 * Wd)));
      for (size_t r0 = 0; r0 < (size_t)n; r0 += R) {
        size_t rows = std::min(R, (size_t)n - r0), prs = rows * n;
        half* qkvgOut = scratch<half>("grid.qkvg", (prs + 128) * 4 * Wd);   // padding: the last query block
        gridIn128(pair, pre, qkvg, qkvgOut, n, r0 * n, prs, tr);
        half* gathered = scratch<half>("grid.gathered", prs * Wd);
        flashGrid<half>(qkvgOut, bias, stride, MASK_ALL_ONES ? nullptr : mask, gathered, n, heads, D, r0, rows, tr, scale);
        if (!tr && PAIR16) rowOut16(gathered, pairRow(into(pair), r0 * n, C), prs, Wd, C, pre + ".outputProjection");
        else if (!tr) linear<half, float>(gathered, into(pair) + r0 * n * C, prs, Wd, C, pre + ".outputProjection", false, 1.f);
        else gridOut128(gathered, pre + ".outputProjection", into(pair), n, r0 * n, prs, tr);
      }
      return;
    }
  }
  // On a card short of room the LayerNorm'd pair is not kept: it is per pair position, so the bias pass and
  // each chunk's q/k/v/gate take it again from the pair (a column chunk gathered transposed first). In
  // place that is safe: a row chunk writes only its own rows, a column chunk only its own columns, and the
  // bias is all taken before any is written.
  const bool streamNorm = shortPair(pairs, C);
  if (streamNorm) needF32Pair("the unfused grid attention's streamed norm");
  T* norm = streamNorm ? nullptr : scratch<T>("grid.norm", pairs * C);
  float* raw = scratch<float>("grid.rawbias", pairs * heads);
  bool headMajor = false;
  if (!streamNorm) {
    if constexpr (std::is_same_v<T, half>) headMajor = lnNormHeads(pair, norm, raw, pairs, C, heads, pre);
    if (!headMajor) {
      lnPairRows<T>(pair, 0, norm, pairs, C, pre + ".actNormScale", pre + ".actNormOffset");
      linear<T, float>(norm, raw, pairs, C, heads, pre + ".pairBiasProjection");
    }
  } else {
    size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / C));
    T* lnc = scratch<T>("grid.normChunk", per * C);
    for (size_t r0 = 0; r0 < pairs; r0 += per) {
      size_t r = std::min(per, pairs - r0);
      layerNorm2<float, T>(pair + r0 * C, lnc, r, C, pre + ".actNormScale", pre + ".actNormOffset");
      linear<T, float>(lnc, raw + r0 * heads, r, C, heads, pre + ".pairBiasProjection");
    }
  }
  int stride = (n + 7) / 8 * 8;
  constexpr bool fast = std::is_same_v<T, half>;
  T* bias = scratch<T>("grid.bias", (size_t)heads * n * stride);
  if (headMajor)
    biasLayoutHeadMajorK<T><<<blocks((size_t)heads * n * stride), 256, 0, STREAM>>>(
      raw, bias, n, stride, heads, tr && swapBias, fast ? LOG2E : 1.f);
  else
    biasLayoutK<T><<<blocks((size_t)heads * n * stride), 256, 0, STREAM>>>(
      raw, bias, n, stride, heads, tr && swapBias, fast ? LOG2E : 1.f);
  std::string qkvg = qkvgWeight(pre, C, Wd, true);
  const float* gateBias = hasW(pre + ".gatingQueryBias") ? W(pre + ".gatingQueryBias") : nullptr;
  const float* outBias = hasW(pre + ".outputProjectionBias") ? W(pre + ".outputProjectionBias") : nullptr;
  size_t R = std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)n * 4 * Wd)));
  float scale = 1.f / sqrtf((float)D);
  for (size_t r0 = 0; r0 < (size_t)n; r0 += R) {
    size_t rows = std::min(R, (size_t)n - r0), prs = rows * n;
    const T* act = streamNorm ? nullptr : norm + r0 * n * C;
    if (streamNorm) {
      T* g = scratch<T>("grid.act", prs * C);
      const float* src = pair + r0 * n * C;
      if (tr) {
        float* g32 = scratch<float>("grid.act32", prs * C);
        if (C * 4 % 16) { fprintf(stderr, "grid attention: %d channels are not 16-byte rows\n", C); exit(1); }
        gatherTransposedK<<<blocks(prs * C * 4 / 16), 256, 0, STREAM>>>(pair, g32, n, C, r0, rows, 4);
        src = g32;
      }
      layerNorm2<float, T>(src, g, prs, C, pre + ".actNormScale", pre + ".actNormOffset");
      act = g;
    }
    T* qkvgOut = scratch<T>("grid.qkvg", (prs + 128) * 4 * Wd);   // padding: the last query block
    if (tr && !streamNorm) {
      // (read transposed in place by a strided-batched GEMM instead: 41.6 against 35 ms for the gather and
      // one GEMM, Chai-1 at 255 tokens - not taken)
      T* g = scratch<T>("grid.act", prs * C);
      if (C * sizeof(T) % 16) { fprintf(stderr, "grid attention: %d channels are not 16-byte rows\n", C); exit(1); }
      gatherTransposedK<<<blocks(prs * C * sizeof(T) / 16), 256, 0, STREAM>>>(norm, g, n, C, r0, rows, sizeof(T));
      act = g;
    }
    linear<T, T>(act, qkvgOut, prs, C, 4 * Wd, qkvg);
    if (gateBias) addGateBiasK<T><<<blocks(prs * Wd), 256, 0, STREAM>>>(qkvgOut, gateBias, prs, Wd);
    T* gathered = scratch<T>("grid.gathered", prs * Wd);
    flashGrid<T>(qkvgOut, bias, stride, MASK_ALL_ONES && std::is_same_v<T, half> ? nullptr : mask, gathered, n, heads, D,
                 r0, rows, tr, scale);       // (no mask when every token is real: the unmasked kernel)
    if (!tr && !outBias) {
      linearIntoPairRows<T>(gathered, into(pair), r0 * n, prs, Wd, C, pre + ".outputProjection");
      continue;
    }
    if (tr && !outBias && RESIDUAL_UNTRANSPOSED) {
      needF32Pair("the parallel block's untransposed residual");
      // the residual kept untransposed (the parallel pair block's): the GEMM adds into it itself
      linear<T, float>(gathered, into(pair) + r0 * n * C, prs, Wd, C, pre + ".outputProjection", false, 1.f);
      continue;
    }
    if constexpr (std::is_same_v<T, half>) {
      if (GRID_STRIDED && tr && !outBias && !PAIR16) {
        // the output projection added into the pair where it belongs, (j, r), by the GEMM itself
        linearStrided<T, float>(gathered, Wd, (size_t)n * Wd, into(pair) + r0 * C, (size_t)n * C, C, n, (int)rows, Wd, C,
                                pre + ".outputProjection", 1.f);
        continue;
      }
    }
    if constexpr (std::is_same_v<T, half>) {
      if (PAIR16 && !outBias) {     // (a bf16 pair: the product in f16, added where it belongs)
        T* o16 = scratch<T>("grid.out16", prs * C);
        linear<T, T>(gathered, o16, prs, Wd, C, pre + ".outputProjection");
        addGridK<__nv_bfloat16, T><<<blocks(prs * C / 4), 256, 0, STREAM>>>(into(pair), o16, n, C, r0, rows, tr);
        continue;
      }
    }
    float* o = scratch<float>("grid.out", prs * C);
    linear<T, float>(gathered, o, prs, Wd, C, pre + ".outputProjection");
    if (outBias) addBiasK<<<blocks(prs * C), 256, 0, STREAM>>>(o, outBias, prs, C);
    WITH_PAIR_T(addGridK<PT, float><<<blocks(prs * C / 4), 256, 0, STREAM>>>(into(pair), o, n, C, r0, rows, tr && !RESIDUAL_UNTRANSPOSED));
  }
}

// Whether a pairformer stack can hold its pair in bf16 (PAIR16): every update it would run has a bf16 form - the
// 128-channel fused triangle (or, where that does not fit, the streaming one from FUSED_WIDE_MIN_TOKENS), the fused
// grid attention and the fused transition - and nothing takes the big-input or parallel paths. LOCALFOLD_PAIR_F32=1
// keeps it f32 (the comparison arm)
inline bool pairBf16Ok(int n, int C, const std::string& B0) {
  static const bool off = getenv("LOCALFOLD_PAIR_F32") != nullptr;
  if (off || M.flag("trunk.dialect.parallelPairformer")) return false;
  // the wider tracks (256, 384, 512): their streaming triangle, the unfused one, the 256-channel and unfused
  // transitions and the unfused grid attention all take a bf16 pair - not their big-input forms
  // (Ampere on: on a Colab T4, which has no f32 -> bf16 conversion instruction, it was level or slower - protenix2's
  // trunk 5654/6045/6589 -> 5725/6102/6602 ms, OpenDDE's 14811/14693 -> 14603/14694 - where the 128-channel track's
  // was 2.7% faster)
  if (C != 128) return (C == 256 || C == 384 || C == 512) && bf16Tensor() && !shortPair((size_t)n * n, C);
  bool narrow = FUSED_TRIANGLE && (TRI_BF16 ? triFusedFits<__nv_bfloat16>() : triFusedFits<float>());
  bool wide = FUSED_WIDE && FUSED_TRIANGLE && n >= FUSED_WIDE_MIN_TOKENS && fitsSmem(wideTriFitsSmem(128));
  std::string A = B0 + ".pairAttention1";
  bool grid = FUSED_GRID && gridFusedFits() && !hasW(A + ".gatingQueryBias") && !hasW(A + ".outputProjectionBias") &&
              (int)M.meta(A + ".heads") * (int)M.meta(A + ".dimension") == 128 && (int)M.meta(A + ".heads") <= 16;
  int I = (int)(lenW(B0 + ".pairTransition.transition1") / (2 * (size_t)C));
  bool tr = FUSED_TRANSITION && fitsSmem((size_t)16 * 4 * 2 * (128 + 8) * 2 + 2 * ftStage<128, 16>()) && I % 16 == 0;
  return (narrow || wide) && grid && tr;
}
// The five pair updates of a pairformer/MSA/template block, in AF3's order.
template <class T>
// releaseBetween: each update's scratch given back before the next (the template stack on a card short of room,
// where the triangle's whole-form buffers - 1.8 GB at 1530 tokens - otherwise sat beside the grid attention's)
void pairUpdates(float* pair, const float* mask, int n, int C, const std::string& pre, bool swap,
                 bool divide, int transitionFactor, bool releaseBetween = false) {
  triangle<T>(pair, mask, n, C, pre + ".triangleMultiplicationOutgoing", true, divide); stage("tri.out");
  triangle<T>(pair, mask, n, C, pre + ".triangleMultiplicationIncoming", false, divide); stage("tri.in");
  if (releaseBetween) releaseScratch({ "tri.", "trib." });
  int heads = (int)M.meta(pre + ".pairAttention1.heads"), D = (int)M.meta(pre + ".pairAttention1.dimension");
  gridAttention<T>(pair, mask, n, C, heads, D, pre + ".pairAttention1", false, swap); stage("grid.row");
  gridAttention<T>(pair, mask, n, C, heads, D, pre + ".pairAttention2", true, swap); stage("grid.col");
  if (releaseBetween) releaseScratch({ "grid." });
  transition<T>(pair, (size_t)n * n, C, transitionFactor, pre + ".pairTransition"); stage("transition");
  if (releaseBetween) releaseScratch({ "tr." });
}
