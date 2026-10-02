// The pair transition in one kernel on the f16 path: x += SwiGLU(LN(x) W1) W2, with the 4C-wide
// intermediate never leaving the chip. The unfused form writes and reads it (2 KB a pair at C 128,
// most of the transition's HBM traffic): LN -> GEMM -> SwiGLU -> GEMM is four passes over bytes
// this keeps in registers.
//
// A block is 64 rows on 4 warps, a warp 16 rows: its LayerNorm'd rows stay in registers as MMA A
// fragments for the whole kernel; the intermediate is walked in chunks of 32 columns (and their 32
// SwiGLU partners), each chunk's a and b computed on the tensor cores, gated in f32, packed
// straight into A fragments of the second GEMM (the flash kernel's P trick), and accumulated into
// the 16 x C output. The weights stream through shared memory, double-buffered.
#pragma once
#include "fusedtriangle.cuh"

constexpr int FT_NC = 32;

template <int C>
__host__ __device__ constexpr size_t ftStage() {   // W1 a and b chunks [C][NC+8], W2 chunk [NC][C+8]
  return (size_t)2 * C * (FT_NC + 8) * 2 + (size_t)FT_NC * (C + 8) * 2;
}

template <int C, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) fusedTransitionK(float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ W1, const half* __restrict__ W2, size_t rows, int I) {
  constexpr int FT_ROWS = 16 * WARPS, NTH = 32 * WARPS;
  constexpr int LDX = C + 8, LDA = FT_NC + 8, LDW2 = C + 8, KS = C / 16, NT = C / 8;
  constexpr size_t STAGE = ftStage<C>();
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem;                                    // [64][LDX]
  unsigned char* stages = smem + (size_t)FT_ROWS * LDX * 2;
  auto W1a = [&](int s) { return (half*)(stages + s * STAGE); };
  auto W1b = [&](int s) { return W1a(s) + C * LDA; };
  auto W2s = [&](int s) { return W1b(s) + C * LDA; };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * FT_ROWS;
  int chunks = I / FT_NC;
  auto issue = [&](int j, int st) {
    half *a = W1a(st), *b = W1b(st), *w2 = W2s(st);
    for (int t = threadIdx.x; t < C * (FT_NC / 8); t += NTH) {      // W1 rows k, 32 columns each half
      int k = t / (FT_NC / 8), c = (t % (FT_NC / 8)) * 8;
      cpAsync16(a + k * LDA + c, W1 + (size_t)k * 2 * I + j * FT_NC + c, true);
      cpAsync16(b + k * LDA + c, W1 + (size_t)k * 2 * I + I + j * FT_NC + c, true);
    }
    for (int t = threadIdx.x; t < FT_NC * (C / 8); t += NTH) {       // W2 rows j*NC .. , all C columns
      int k = t / (C / 8), c = (t % (C / 8)) * 8;
      cpAsync16(w2 + k * LDW2 + c, W2 + (size_t)(j * FT_NC + k) * C + c, true);
    }
    asm volatile("cp.async.commit_group;");
  };
  issue(0, 0);
  // LayerNorm, a warp a row (C / 32 floats a lane), into shared memory as f16
  lnRowsToShared<C, FT_ROWS, WARPS>(x, [&](int r) { size_t row = row0 + r; return row < rows ? row : SIZE_MAX; },
                                    lnScale, lnOffset, Xs, LDX, warp, lane);
  __syncthreads();
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  float acc[NT][4] = {};
  for (int j = 0; j < chunks; ++j) {
    int st = j & 1;
    if (j + 1 < chunks) { issue(j + 1, st ^ 1); asm volatile("cp.async.wait_group 1;"); }
    else asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    const half *a = W1a(st), *b = W1b(st), *w2 = W2s(st);
    float ha[FT_NC / 8][4] = {}, hb[FT_NC / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int n2 = 0; n2 < FT_NC / 16; ++n2) {
        uint32_t fa[4], fb[4];
        int k = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), c = n2 * 16 + (lane >> 4) * 8;
        ldsm4t(fa, a + k * LDA + c);
        ldsm4t(fb, b + k * LDA + c);
        mma16816(ha[2 * n2], xa[ks], fa[0], fa[1]); mma16816(ha[2 * n2 + 1], xa[ks], fa[2], fa[3]);
        mma16816(hb[2 * n2], xa[ks], fb[0], fb[1]); mma16816(hb[2 * n2 + 1], xa[ks], fb[2], fb[3]);
      }
    }
    // SwiGLU in f32, straight into the second GEMM's A fragments
#pragma unroll
    for (int t = 0; t < FT_NC / 16; ++t) {
      auto gate = [&](int nt, int e) { float v = ha[nt][e]; return v * sigm(v) * hb[nt][e]; };
      uint32_t pa[4] = { pack2(gate(2 * t, 0), gate(2 * t, 1)), pack2(gate(2 * t, 2), gate(2 * t, 3)),
                         pack2(gate(2 * t + 1, 0), gate(2 * t + 1, 1)), pack2(gate(2 * t + 1, 2), gate(2 * t + 1, 3)) };
#pragma unroll
      for (int et = 0; et < NT; et += 2) {
        uint32_t vb[4];
        ldsm4t(vb, w2 + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW2 + (et + (lane >> 4)) * 8);
        mma16816(acc[et], pa, vb[0], vb[1]);
        mma16816(acc[et + 1], pa, vb[2], vb[3]);
      }
    }
    __syncthreads();
  }
  // the residual: rows g and g + 8 of the warp's 16, columns et*8 + 2 tig (+1)
  size_t r0 = row0 + warp * 16 + g, r1 = r0 + 8;
#pragma unroll
  for (int et = 0; et < NT; ++et) {
    int c = et * 8 + tig * 2;
    if (r0 < rows) { float2* p = (float2*)(x + r0 * C + c); float2 v = *p; v.x += acc[et][0]; v.y += acc[et][1]; *p = v; }
    if (r1 < rows) { float2* p = (float2*)(x + r1 * C + c); float2 v = *p; v.x += acc[et][2]; v.y += acc[et][3]; *p = v; }
  }
}

inline bool FUSED_TRANSITION = true;
inline int FT_WARPS = 8;
template <int WARPS>
void fusedTransitionAt(float* x, size_t rows, int C, int I, const std::string& pre) {
  constexpr int R = 16 * WARPS;
  size_t smem = (size_t)R * (128 + 8) * 2 + 2 * ftStage<128>();
  static bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(fusedTransitionK<128, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  fusedTransitionK<128, WARPS><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
    x, W(pre + ".inputLayerNormScale"), W(pre + ".inputLayerNormOffset"), Wh(pre + ".transition1"),
    Wh(pre + ".transition2"), rows, I);
}
// x += transition(x) for C = 128, f16 weights; false if the shape is not this kernel's
inline bool fusedTransition(float* x, size_t rows, int C, int I, const std::string& pre) {
  if (!FUSED_TRANSITION || C != 128 || I % FT_NC) return false;
  if (FT_WARPS == 4) fusedTransitionAt<4>(x, rows, C, I, pre);
  else if (FT_WARPS == 16) fusedTransitionAt<16>(x, rows, C, I, pre);
  else fusedTransitionAt<8>(x, rows, C, I, pre);
  return true;
}
