// Kernel arguments, declared once for both sides: this header is C++ to the host and Metal Shading Language to the
// kernels (metal/build.sh prepends it to every Metal source), so a struct's layout cannot differ between the two.
// Pointers are GPU addresses (an allocation's MTLBuffer.gpuAddress plus an offset) - Metal 3's pointers in argument
// structs - so host code keeps plain pointer arithmetic. Only 4- and 8-byte scalars and pointers: no bool, no vectors.
#pragma once
#ifdef __METAL_VERSION__
#define DP(T) device T*
#define CP(T) device const T*
typedef ulong u64;
typedef long i64;
#else
#include <cstdint>
typedef _Float16 half;
typedef uint64_t u64;
typedef int64_t i64;
typedef unsigned int uint;
typedef unsigned char uchar;
#define DP(T) T*
#define CP(T) const T*
#endif

// ---------------------------------------------------------------- core (metal/core/common.metal)
struct CopyArgs { DP(uchar) dst; CP(uchar) src; u64 bytes; };
struct Copy2DArgs { DP(uchar) dst; CP(uchar) src; u64 dpitch, spitch, width, height; };
struct FillArgs { DP(uchar) dst; u64 bytes; uint value; uint pad; };
// a bundle tensor decoded on the device (metal/core/weights.mm): kind 0 float32, 1 float16, 3 intN asymmetric
// (code * scale + zero a group, scale and zero float16); out16 writes float16, else float32
// first: the bundle tensor's first element decoded (a slice); flags: 1 rounded to float16 (a delta's base), 2 added to
// what dst holds (a delta)
struct DecodeEntry { u64 src, scale, zero, dst, n, first; uint kind, bits, block, out16, flags, pad; };
// a weight walk's gathered part (metal/core/model.cpp): op 'v' a strided view of one source, 'x' the product of two,
// 'i' a source converted to integers, 'o' ones - n elements over `rank` dimensions, written float32 at dst
struct GatherPart { u64 dst, src0, src1, n; i64 dims[6], ds[6], s0[6], s1[6]; int rank, op; };
struct GatherArgs { CP(GatherPart) parts; };
struct DecodeArgs { CP(uchar) raw; CP(DecodeEntry) table; };
struct ConvArgs { CP(float) x; DP(half) y; u64 n; };          // f32 -> f16 / f16 -> f32
struct ConvBackArgs { CP(half) x; DP(float) y; u64 n; };
struct LayerNormArgs {      // a simdgroup a row; scale/offset may be null; x f32 or f16 (xh), y f32 or f16 (yh)
  CP(float) x; CP(half) xh; DP(float) y; DP(half) yh; CP(float) scale; CP(float) offset;
  u64 rows; int C, ldx, ldy; float eps;
};
struct AddArgs { DP(float) y; CP(float) x; u64 n; float a; uint pad; };   // y += a x
struct BiasArgs { DP(float) y; CP(float) b; u64 rows; int C; int act; };  // y += b per column, act 1 relu 2 gelu(erf)

// gated flash attention (metal/core/common.metal, lf_attention): per batch row b and head h, the queries' and keys'
// q | k | v | g at qkvg + b rowStride + position posStride (+ 0, W, 2W, 3W, then h D); out at b outRowStride +
// position outPosStride + h D. bias [H][n][biasStride] in log2 units, shared by every row; mask the key's,
// mask[(r0 + b) n + key] or (maskT) mask[key n + r0 + b]; qBias [W] added to the query
struct AttnArgs { CP(half) qkvg; DP(half) out; CP(half) bias; CP(float) mask; CP(float) qBias;
                  i64 rowStride, posStride, outRowStride, outPosStride, r0;
                  int n, heads, biasStride, maskT; float scale; int pad; };

// ---------------------------------------------------------------- GEMM (metal/core/gemm.metal)
struct GemmArgs {
  u64 A, B, C, D, bias;            // device addresses (D: the output, C: beta's input)
  i64 sa, sb, sc, sd;              // batch strides, elements
  int m, n, k, lda, ldb, ldc, ldd;
  int ta, tb, ptrs, epilogue, biasType;
  float alpha, beta;
  u64 aux; int ldaux, auxPad;      // epilogue 64: D = C + aux * sigmoid(alpha X W)
  u64 aux2, aux3; i64 tgR0, tgPairs; int tgN, tgNp, tgC, tgPad;   // the triangle's gate (epilogue 256)
};
