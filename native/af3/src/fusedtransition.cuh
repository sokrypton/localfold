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
__host__ __device__ constexpr size_t ftStage() {   // W1 a and b chunks [C][NC] (stageSw), W2 chunk [NC][C] (w2Sw)
  return (size_t)2 * C * NC * 2 + (size_t)NC * C * 2;
}
// a W2 stage row is C = 128 halves unpadded, its sixteen 16-byte chunks XOR-swizzled by the row's low three
// bits: ldmatrix.trans reads eight consecutive rows at one chunk
template <int C>
__device__ __forceinline__ int w2Sw(int k, int c) {
  static_assert(C == 128, "sixteen chunks a row");
  return k * C + (c ^ ((k & 7) << 3));
}
// W1 as fusedTransitionK streams it: NC-column tiles [C][NC] contiguous (tileColumns), a's then b's
// (RELU: the one half). From W1 itself each 16-byte piece of a stage was a row from its neighbour's,
// and a cp.async write takes a shared wavefront per global row: 50% of the kernel's shared wavefronts
// were excess, nearly all of them those writes (Nsight Compute, AF3 at 1,044 tokens)
// Made once a weight and kept (128-256 KB each): tiled per call it was two launches a transition, 1.2 ms
// of a 68-token AF3 fold's 117. Forgotten with the derived weights (FORGET_HOOKS), which rebuild W1 itself.
template <bool RELU>
inline const half* transitionW1Tiles(const half* W1, int C, int I, int NC) {
  static std::map<std::pair<const half*, int>, half*> tiles;
  static bool hooked = false;
  if (!hooked) { FORGET_HOOKS.push_back([] { for (auto& [k, p] : tiles) CK(cudaFree(p)); tiles.clear(); }); hooked = true; }
  auto it = tiles.find({W1, NC});
  if (it != tiles.end()) return it->second;
  half* out = dallocT<half>((size_t)C * I * (RELU ? 1 : 2));
  if constexpr (RELU) tileColumns(W1, C, I, 0, I, NC, out);
  else { tileColumns(W1, C, 2 * I, 0, I, NC, out); tileColumns(W1, C, 2 * I, I, I, NC, out + (size_t)C * I); }
  return tiles[{W1, NC}] = out;
}

// RELU: AlphaFold 2's transition - x += ReLU(LN(x) W1 + b1) W2 + b2, W1 [C][I] (one half, not SwiGLU's two);
// without it the kernel is AF3's. MT: 16-row tiles a warp - at 2 every weight fragment read from shared
// memory feeds two MMAs instead of one (the kernel is bound by those reads at MT 1); NC: intermediate
// columns a chunk (16 keeps MT 2's registers in bounds)
template <int C, int WARPS, bool RELU = false, int MT = 1, int NC = FT_NC>
__global__ void __launch_bounds__(WARPS * 32) fusedTransitionK(float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ W1t, const half* __restrict__ W2, size_t rows, int I,
    const float* __restrict__ b1 = nullptr, const float* __restrict__ b2 = nullptr) {
  constexpr int FT_ROWS = 16 * WARPS * MT, NTH = 32 * WARPS;
  constexpr int LDX = C + 8, KS = C / 16, NT = C / 8;
  constexpr size_t STAGE = ftStage<C, NC>();
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem;                                    // [FT_ROWS][LDX]
  unsigned char* stages = smem + (size_t)FT_ROWS * LDX * 2;
  auto W1a = [&](int s) { return (half*)(stages + s * STAGE); };
  auto W1b = [&](int s) { return W1a(s) + C * NC; };
  auto W2s = [&](int s) { return W1b(s) + C * NC; };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * FT_ROWS;
  int chunks = I / NC;
  auto issue = [&](int j, int st) {
    half *a = W1a(st), *b = W1b(st), *w2 = W2s(st);
    for (int t = threadIdx.x; t < C * (NC / 8); t += NTH) {      // W1 rows k, NC columns each half
      int k = t / (NC / 8), c = (t % (NC / 8)) * 8;
      const half* src = W1t + ((size_t)j * C + k) * NC + c;
      cpAsync16(a + stageSw<NC>(k, c), src, true);
      if constexpr (!RELU) cpAsync16(b + stageSw<NC>(k, c), src + (size_t)C * I, true);
    }
    for (int t = threadIdx.x; t < NC * (C / 8); t += NTH) {       // W2 rows j*NC .. , all C columns
      int k = t / (C / 8), c = (t % (C / 8)) * 8;
      cpAsync16(w2 + w2Sw<C>(k, c), W2 + (size_t)(j * NC + k) * C + c, true);
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
        ldsm4t(fa, a + stageSw<NC>(k, c));
#pragma unroll
        for (int m = 0; m < MT; ++m) { mma16816(ha[m][2 * n2], xa[m][ks], fa[0], fa[1]); mma16816(ha[m][2 * n2 + 1], xa[m][ks], fa[2], fa[3]); }
        if constexpr (!RELU) {
          ldsm4t(fb, b + stageSw<NC>(k, c));
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
          else { float v = ha[m][nt][e]; return v * sigmH(v) * hb[m][nt][e]; }   // packed to f16 next
        };
        pa[m][0] = pack2(gate(2 * t, 0), gate(2 * t, 1)); pa[m][1] = pack2(gate(2 * t, 2), gate(2 * t, 3));
        pa[m][2] = pack2(gate(2 * t + 1, 0), gate(2 * t + 1, 1)); pa[m][3] = pack2(gate(2 * t + 1, 2), gate(2 * t + 1, 3));
      }
#pragma unroll
      for (int et = 0; et < NT; et += 2) {
        uint32_t vb[4];
        ldsm4t(vb, w2 + w2Sw<C>(t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), (et + (lane >> 4)) * 8));
#pragma unroll
        for (int m = 0; m < MT; ++m) { mma16816(acc[m][et], pa[m], vb[0], vb[1]); mma16816(acc[m][et + 1], pa[m], vb[2], vb[3]); }
      }
    }
  }
  // the residual, through the warp's own Xs rows (only it read them, as its A fragments; the loop's
  // barriers are behind every warp) 64 f32 columns at a time, so the read-modify-write is 16 bytes a lane
  // of whole rows, where a fragment's layout was 8 bytes in each of 8 rows
  constexpr int RW = 16 * MT, LDY = C / 2 + 4;
  static_assert(C == 128 && RW * LDY * 4 <= RW * LDX * 2, "a half of the warp's accumulators fits in its rows");
  float* Ys = reinterpret_cast<float*>(Xs + warp * RW * LDX);
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int e8 = 0; e8 < NT / 2; ++e8) {
        int et = hh * (NT / 2) + e8, c = e8 * 8 + tig * 2;
        *reinterpret_cast<float2*>(Ys + (m * 16 + g) * LDY + c) = make_float2(acc[m][et][0], acc[m][et][1]);
        *reinterpret_cast<float2*>(Ys + (m * 16 + g + 8) * LDY + c) = make_float2(acc[m][et][2], acc[m][et][3]);
      }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < RW / 2; ++i) {                          // row i * 2 + lane / 16, columns (lane % 16) * 4
      int rl = i * 2 + (lane >> 4), cl = (lane & 15) * 4, c = hh * (C / 2) + cl;
      size_t row = row0 + warp * RW + rl;
      if (row >= rows) continue;
      float4 a4 = *reinterpret_cast<const float4*>(Ys + rl * LDY + cl);
      float4* p = (float4*)(x + row * C + c); float4 v = *p;
      float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
      if constexpr (RELU) { o0 = b2[c]; o1 = b2[c + 1]; o2 = b2[c + 2]; o3 = b2[c + 3]; }
      v.x += a4.x + o0; v.y += a4.y + o1; v.z += a4.z + o2; v.w += a4.w + o3; *p = v;
    }
    __syncwarp();
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
  const half* w1t = transitionW1Tiles<RELU>(W1, 128, I, NC);
  fusedTransitionK<128, WARPS, RELU, MT, NC><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
    x, lnScale, lnOffset, w1t, W2, rows, I, b1, b2);
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
