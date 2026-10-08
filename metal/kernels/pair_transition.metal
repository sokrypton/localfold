// AF3's pair transition, fused: LayerNorm, the up projection to two halves
// (a, b), swish(a) * b, the down projection, and the residual add - in place on
// the pair, one pass over it.
//
//   pair     [rows][C]   float, read and written (the residual)
//   gamma    [C], beta [C]   float
//   w1       [C][2F]     half; a is columns 0..F, b is columns F..2F
//   w2       [F][C]      half
//
// Measured on an M2 at 256 tokens (65,536 rows, metal/check/check-kernels):
// 9.8 ms against WebGPU's 19.2 a call in the trunk, 2.63 TFLOP/s - which is AT
// Apple's own MPS on a 4096-cube GEMM (2.4-2.8) on this part. There is nothing
// left in this kernel worth chasing on an M2.
//
// HOW. A threadgroup takes 32 rows. LayerNorm writes them to threadgroup memory
// as half; the intermediate is produced in chunks of 32 columns (a and b
// together), gated in registers, written as half, and immediately contracted
// into the output - so the [rows][F] intermediate never exists anywhere. Two
// simdgroups, each owning half the output columns; the output tile lives in
// registers until the residual.
//
// Measured and NOT taken (M2, 256 tokens): f16 accumulation of a and b (no
// faster - the M2's matrix units run f16 at f32's rate - and relRMS 2.6e-3
// against 4.6e-6), four or eight simdgroups (4-40% slower), 64-row tiles and
// 128-wide chunks (slower; 64 rows with four simdgroups cannot even launch 128
// threads, see check-kernels.mm). As in grid_attend.metal, every tile loop must
// be fully unrolled.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")

struct PairTransitionShape { uint rows; float eps; };

template <uint C, uint F, uint SG, uint R, uint CH>
void pair_transition_impl(device float* pair, device const float* gamma, device const float* beta,
                          device const half* w1, device const half* w2, constant PairTransitionShape& s,
                          uint tg, uint sg, uint lane, threadgroup half* X, threadgroup half* G) {
  static_assert(C == 128, "the LayerNorm reads four channels a lane");
  const uint r0 = tg * R;
  for (uint rr = sg; rr < R; rr += SG) {
    uint row = min(r0 + rr, s.rows - 1);
    float4 x = *(device const float4*)(pair + row * C + lane * 4);
    float mean = simd_sum(x.x + x.y + x.z + x.w) / C;
    float4 d = x - mean;
    float var = simd_sum(dot(d, d)) / C;
    float4 y = d * rsqrt(var + s.eps) * *(device const float4*)(gamma + lane * 4)
             + *(device const float4*)(beta + lane * 4);
    *(threadgroup half4*)(X + rr * C + lane * 4) = half4(y);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // This lane's place in every 8x8 tile: row sm, columns sn and sn + 1.
  const uint sm = (lane / 16) * 4 + (lane % 8) / 2;
  const uint sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  constexpr uint RT = R / 8, OC = C / SG / 8, UC = CH / SG / 8;
  simdgroup_float8x8 O[RT][OC];
  UNROLL for (uint i = 0; i < RT; ++i) UNROLL for (uint j = 0; j < OC; ++j) O[i][j] = simdgroup_float8x8(0);

  for (uint c0 = 0; c0 < F; c0 += CH) {
    simdgroup_float8x8 A[RT][UC], B[RT][UC];
    UNROLL for (uint i = 0; i < RT; ++i) UNROLL for (uint j = 0; j < UC; ++j) {
      A[i][j] = simdgroup_float8x8(0); B[i][j] = simdgroup_float8x8(0);
    }
    const uint col = c0 + sg * (CH / SG);
    UNROLL for (uint kk = 0; kk < C / 8; ++kk) {
      simdgroup_half8x8 Wa[UC], Wb[UC];
      UNROLL for (uint j = 0; j < UC; ++j) {
        simdgroup_load(Wa[j], w1, 2 * F, ulong2(col + j * 8, kk * 8));
        simdgroup_load(Wb[j], w1, 2 * F, ulong2(F + col + j * 8, kk * 8));
      }
      UNROLL for (uint i = 0; i < RT; ++i) {
        simdgroup_half8x8 Xm;
        simdgroup_load(Xm, X, C, ulong2(kk * 8, i * 8));
        UNROLL for (uint j = 0; j < UC; ++j) {
          simdgroup_multiply_accumulate(A[i][j], Xm, Wa[j], A[i][j]);
          simdgroup_multiply_accumulate(B[i][j], Xm, Wb[j], B[i][j]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous chunk's G is fully read
    UNROLL for (uint i = 0; i < RT; ++i) UNROLL for (uint j = 0; j < UC; ++j) {
      thread auto& a = A[i][j].thread_elements();
      thread auto& b = B[i][j].thread_elements();
      float g0 = a[0] / (1.0f + exp(-a[0])) * b[0];
      float g1 = a[1] / (1.0f + exp(-a[1])) * b[1];
      *(threadgroup half2*)(G + (i * 8 + sm) * CH + sg * (CH / SG) + j * 8 + sn) = half2(g0, g1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    UNROLL for (uint kk = 0; kk < CH / 8; ++kk) {
      simdgroup_half8x8 Wm[OC];
      UNROLL for (uint j = 0; j < OC; ++j) simdgroup_load(Wm[j], w2, C, ulong2(sg * (C / SG) + j * 8, c0 + kk * 8));
      UNROLL for (uint i = 0; i < RT; ++i) {
        simdgroup_half8x8 Gm;
        simdgroup_load(Gm, G, CH, ulong2(kk * 8, i * 8));
        UNROLL for (uint j = 0; j < OC; ++j) simdgroup_multiply_accumulate(O[i][j], Gm, Wm[j], O[i][j]);
      }
    }
  }
  UNROLL for (uint i = 0; i < RT; ++i) {
    uint row = r0 + i * 8 + sm;
    if (row >= s.rows) continue;
    UNROLL for (uint j = 0; j < OC; ++j) {
      thread auto& o = O[i][j].thread_elements();
      device float2* p = (device float2*)(pair + row * C + sg * (C / SG) + j * 8 + sn);
      *p = *p + float2(o[0], o[1]);
    }
  }
}

// Dispatch: ceil(rows / 32) threadgroups of 64 threads.
kernel void pair_transition_c128_f512(device float* pair [[buffer(0)]], device const float* gamma [[buffer(1)]],
                                      device const float* beta [[buffer(2)]], device const half* w1 [[buffer(3)]],
                                      device const half* w2 [[buffer(4)]], constant PairTransitionShape& s [[buffer(5)]],
                                      uint tg [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half X[32 * 128];
  threadgroup half G[32 * 32];
  pair_transition_impl<128, 512, 2, 32, 32>(pair, gamma, beta, w1, w2, s, tg, sg, lane, X, G);
}
