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

template <int C, int NC = FT_NC>
__host__ __device__ constexpr size_t ftStage() {   // W1 a and b chunks [C][NC+8], W2 chunk [NC][C+8]
  return (size_t)2 * C * (NC + 8) * 2 + (size_t)NC * (C + 8) * 2;
}

// RELU: AlphaFold 2's transition - x += ReLU(LN(x) W1 + b1) W2 + b2, W1 [C][I] (one half, not SwiGLU's two);
// without it the kernel is AF3's. MT: 16-row tiles a warp - at 2 every weight fragment read from shared
// memory feeds two MMAs instead of one (the kernel is bound by those reads at MT 1); NC: intermediate
// columns a chunk (16 keeps MT 2's registers in bounds)
template <int C, int WARPS, bool RELU = false, int MT = 1, int NC = FT_NC>
__global__ void __launch_bounds__(WARPS * 32) fusedTransitionK(float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ W1, const half* __restrict__ W2, size_t rows, int I,
    const float* __restrict__ b1 = nullptr, const float* __restrict__ b2 = nullptr) {
  constexpr int FT_ROWS = 16 * WARPS * MT, NTH = 32 * WARPS;
  constexpr int LDX = C + 8, LDA = NC + 8, LDW2 = C + 8, KS = C / 16, NT = C / 8;
  constexpr size_t STAGE = ftStage<C, NC>();
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem;                                    // [FT_ROWS][LDX]
  unsigned char* stages = smem + (size_t)FT_ROWS * LDX * 2;
  auto W1a = [&](int s) { return (half*)(stages + s * STAGE); };
  auto W1b = [&](int s) { return W1a(s) + C * LDA; };
  auto W2s = [&](int s) { return W1b(s) + C * LDA; };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * FT_ROWS;
  int chunks = I / NC;
  auto issue = [&](int j, int st) {
    half *a = W1a(st), *b = W1b(st), *w2 = W2s(st);
    for (int t = threadIdx.x; t < C * (NC / 8); t += NTH) {      // W1 rows k, NC columns each half
      int k = t / (NC / 8), c = (t % (NC / 8)) * 8;
      if constexpr (RELU) {
        cpAsync16(a + k * LDA + c, W1 + (size_t)k * I + j * NC + c, true);
      } else {
        cpAsync16(a + k * LDA + c, W1 + (size_t)k * 2 * I + j * NC + c, true);
        cpAsync16(b + k * LDA + c, W1 + (size_t)k * 2 * I + I + j * NC + c, true);
      }
    }
    for (int t = threadIdx.x; t < NC * (C / 8); t += NTH) {       // W2 rows j*NC .. , all C columns
      int k = t / (C / 8), c = (t % (C / 8)) * 8;
      cpAsync16(w2 + k * LDW2 + c, W2 + (size_t)(j * NC + k) * C + c, true);
    }
    cpCommit();
  };
  issue(0, 0);
  // LayerNorm, a warp a row (C / 32 floats a lane), into shared memory as f16
  lnRowsToShared<C, FT_ROWS, WARPS>(x, [&](int r) { size_t row = row0 + r; return row < rows ? row : SIZE_MAX; },
                                    lnScale, lnOffset, Xs, LDX, warp, lane);
  __syncthreads();
  uint32_t xa[MT][KS][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int ks = 0; ks < KS; ++ks)
      ldsm4(xa[m][ks], Xs + ((warp * MT + m) * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  float acc[MT][NT][4] = {};
  for (int j = 0; j < chunks; ++j) {
    int st = j & 1;
    // chunk j has landed; the barrier also says every warp is done with the stage the next issue
    // overwrites (chunk j - 1's), so one barrier a chunk
    cpWait<0>();
    __syncthreads();
    if (j + 1 < chunks) issue(j + 1, st ^ 1);
    const half *a = W1a(st), *b = W1b(st), *w2 = W2s(st);
    float ha[MT][NC / 8][4] = {}, hb[MT][NC / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int n2 = 0; n2 < NC / 16; ++n2) {
        uint32_t fa[4], fb[4];
        int k = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), c = n2 * 16 + (lane >> 4) * 8;
        ldsm4t(fa, a + k * LDA + c);
#pragma unroll
        for (int m = 0; m < MT; ++m) { mma16816(ha[m][2 * n2], xa[m][ks], fa[0], fa[1]); mma16816(ha[m][2 * n2 + 1], xa[m][ks], fa[2], fa[3]); }
        if constexpr (!RELU) {
          ldsm4t(fb, b + k * LDA + c);
#pragma unroll
          for (int m = 0; m < MT; ++m) { mma16816(hb[m][2 * n2], xa[m][ks], fb[0], fb[1]); mma16816(hb[m][2 * n2 + 1], xa[m][ks], fb[2], fb[3]); }
        }
      }
    }
    // SwiGLU in f32, straight into the second GEMM's A fragments
#pragma unroll
    for (int t = 0; t < NC / 16; ++t) {
      uint32_t pa[MT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        auto gate = [&](int nt, int e) {
          if constexpr (RELU) return fmaxf(ha[m][nt][e] + b1[j * NC + nt * 8 + tig * 2 + (e & 1)], 0.f);
          else { float v = ha[m][nt][e]; return v * sigm(v) * hb[m][nt][e]; }
        };
        pa[m][0] = pack2(gate(2 * t, 0), gate(2 * t, 1)); pa[m][1] = pack2(gate(2 * t, 2), gate(2 * t, 3));
        pa[m][2] = pack2(gate(2 * t + 1, 0), gate(2 * t + 1, 1)); pa[m][3] = pack2(gate(2 * t + 1, 2), gate(2 * t + 1, 3));
      }
#pragma unroll
      for (int et = 0; et < NT; et += 2) {
        uint32_t vb[4];
        ldsm4t(vb, w2 + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW2 + (et + (lane >> 4)) * 8);
#pragma unroll
        for (int m = 0; m < MT; ++m) { mma16816(acc[m][et], pa[m], vb[0], vb[1]); mma16816(acc[m][et + 1], pa[m], vb[2], vb[3]); }
      }
    }
  }
  // the residual: rows g and g + 8 of each of the warp's 16-row tiles, columns et*8 + 2 tig (+1)
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    size_t r0 = row0 + (warp * MT + m) * 16 + g, r1 = r0 + 8;
#pragma unroll
    for (int et = 0; et < NT; ++et) {
      int c = et * 8 + tig * 2;
      float o0 = 0.f, o1 = 0.f;
      if constexpr (RELU) { o0 = b2[c]; o1 = b2[c + 1]; }
      if (r0 < rows) { float2* p = (float2*)(x + r0 * C + c); float2 v = *p; v.x += acc[m][et][0] + o0; v.y += acc[m][et][1] + o1; *p = v; }
      if (r1 < rows) { float2* p = (float2*)(x + r1 * C + c); float2 v = *p; v.x += acc[m][et][2] + o0; v.y += acc[m][et][3] + o1; *p = v; }
    }
  }
}

// one launch of a given form (the bench's arms; fusedTransitionRaw picks the form)
template <int WARPS, bool RELU = false, int MT = 1, int NC = FT_NC>
void fusedTransitionAt(float* x, size_t rows, int I, const float* lnScale, const float* lnOffset, const half* W1,
                       const half* W2, const float* b1, const float* b2) {
  constexpr int R = 16 * WARPS * MT;
  size_t smem = (size_t)R * (128 + 8) * 2 + 2 * ftStage<128, NC>();
  static bool attr = false;
  if (!attr) { smemAttr((fusedTransitionK<128, WARPS, RELU, MT, NC>), (int)smem); attr = true; }
  fusedTransitionK<128, WARPS, RELU, MT, NC><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
    x, lnScale, lnOffset, W1, W2, rows, I, b1, b2);
}
inline bool FUSED_TRANSITION = true;
inline int FT_WARPS = 8;
inline bool FT_TWO_TILES = !getenv("LOCALFOLD_FT_ONE_TILE");
// x += transition(x) for C = 128, f16 weights, on raw pointers (AlphaFold 2 calls it with RELU and its
// biases); false if the shape is not this kernel's or the device cannot hold its blocks
template <bool RELU = false>
inline bool fusedTransitionRaw(float* x, size_t rows, int C, int I, const float* lnScale, const float* lnOffset,
                               const half* W1, const half* W2, const float* b1 = nullptr, const float* b2 = nullptr) {
  if (!FUSED_TRANSITION || C != 128 || I % FT_NC) return false;
  // two 16-row tiles a warp on 4 warps (128-row blocks, 68 KB): 1.14x the 8-warp one-tile form at 1,044
  // tokens (--bench-trans), bit-identical; where it does not fit (a T4) or the input is small, the one-tile forms
  if (FT_TWO_TILES && (rows + 127) / 128 >= MIN_BLOCKS && fitsSmem((size_t)128 * (128 + 8) * 2 + 2 * ftStage<128, 16>())) {
    fusedTransitionAt<4, RELU, 2, 16>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
    return true;
  }
  int warps = FT_WARPS == 8 && (rows + 127) / 128 < MIN_BLOCKS ? 4 : FT_WARPS;   // a small input: more, smaller blocks
  // no more than the device's shared memory (a T4's 64 KB takes the 4-warp form, or the unfused one)
  auto smemFor = [](int w) { return (size_t)16 * w * (128 + 8) * 2 + 2 * ftStage<128>(); };
  while (warps > 4 && !fitsSmem(smemFor(warps))) warps /= 2;
  if (!fitsSmem(smemFor(warps))) return false;
  if (warps == 4) fusedTransitionAt<4, RELU>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
  else if (warps == 16) fusedTransitionAt<16, RELU>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
  else fusedTransitionAt<8, RELU>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
  return true;
}
inline bool fusedTransition(float* x, size_t rows, int C, int I, const std::string& pre) {
  if (!FUSED_TRANSITION || C != 128 || I % FT_NC) return false;
  return fusedTransitionRaw(x, rows, C, I, W(pre + ".inputLayerNormScale"), W(pre + ".inputLayerNormOffset"),
                            Wh(pre + ".transition1"), Wh(pre + ".transition2"));
}
