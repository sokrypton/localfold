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

template <int C, int NC = FT_NC, bool RELU = false>
__host__ __device__ constexpr size_t ftStage() {   // W1 a (and b) chunks [C][NC] (stageSw), W2 chunk [NC][C] (w2Sw)
  return (size_t)(RELU ? 1 : 2) * C * NC * 2 + (size_t)NC * C * 2;
}
template <int C, int WARPS, int MT = 1, int NC = FT_NC, bool RELU = false>
__host__ __device__ constexpr size_t ftSmem() {
  constexpr size_t xs = (size_t)16 * WARPS * MT * (C + 8) * 2, st = ftStage<C, NC, RELU>();
  return (xs > st ? xs : st) + st;
}
// a W2 stage row is C halves unpadded (128 or 256: 16 or 32 16-byte chunks), the chunk index XOR-swizzled by
// the row's low three bits: ldmatrix.trans reads eight consecutive rows at one chunk
template <int C>
__device__ __forceinline__ int w2Sw(int k, int c) {
  static_assert(C == 64 || C == 128 || C == 256, "8, 16 or 32 chunks a row");
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
  static std::vector<half*> chunks; static size_t used = 0, have = 0;
  static bool hooked = false;
  if (!hooked) {
    FORGET_HOOKS.push_back([] { for (half* p : chunks) CK(cudaFreeAsync(p, STREAM)); chunks.clear(); tiles.clear(); used = have = 0; });
    hooked = true;
  }
  auto it = tiles.find({W1, NC});
  if (it != tiles.end()) return it->second;
  size_t n = ((size_t)C * I * (RELU ? 1 : 2) + 127) / 128 * 128;
  if (used + n > have) {
    have = std::max<size_t>(n, (size_t)8 << 20);
    chunks.push_back(poolAllocT<half>(have)); used = 0;
  }
  half* out = chunks.back() + used; used += n;
  if constexpr (RELU) tileColumns(W1, C, I, 0, I, NC, out);
  else { tileColumns(W1, C, 2 * I, 0, I, NC, out); tileColumns(W1, C, 2 * I, I, I, NC, out + (size_t)C * I); }
  return tiles[{W1, NC}] = out;
}

// RELU: AlphaFold 2's transition - x += ReLU(LN(x) W1 + b1) W2 + b2, W1 [C][I] (one half, not SwiGLU's two);
// without it the kernel is AF3's. MT: 16-row tiles a warp - at 2 every weight fragment read from shared
// memory feeds two MMAs instead of one (the kernel is bound by those reads at MT 1); NC: intermediate
// columns a chunk (16 keeps MT 2's registers in bounds)
// NST 1: ONE weight stage, a barrier before each chunk's store into it - for a T4, whose register-staged prefetch (LF_REG_STAGES)
// holds the next chunk in registers anyway: the second stage was 12 KB that kept its 64 KB SM to ONE block (4 warps)
template <int C, int WARPS, bool RELU = false, int MT = 1, int NC = FT_NC, class PT = float, int NST = 2,
          int MINB = ((C == 128 || C == 64) && WARPS == 4 && MT == 1 && NC == 16 && NST == 2) ? 3 : 1>
#if LF_REG_STAGES
__global__ void __launch_bounds__(WARPS * 32, (C == 64 && WARPS == 4 && MT == 1 && NC == 16) ? 3 : 1)
#else
__global__ void __launch_bounds__(WARPS * 32, MINB)
#endif
fusedTransitionK(float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ W1t, const half* __restrict__ W2, size_t rows, int I,
    const float* __restrict__ b1 = nullptr, const float* __restrict__ b2 = nullptr) {
  constexpr int FT_ROWS = 16 * WARPS * MT, NTH = 32 * WARPS;
  constexpr int LDX = C + 8, KS = C / 16, NT = C / 8;
  constexpr size_t STAGE = ftStage<C, NC, RELU>(), FT_XS = (size_t)FT_ROWS * LDX * 2;
  constexpr size_t OFF0 = FT_XS > STAGE ? FT_XS : STAGE;
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem;                                    // [FT_ROWS][LDX]; stage 1 aliases Xs once xa is loaded
  auto W1a = [&](int s) { return (half*)(smem + (s == 0 ? OFF0 : 0)); };
  auto W1b = [&](int s) { return W1a(s) + (RELU ? 0 : C * NC); };
  auto W2s = [&](int s) { return W1b(s) + C * NC; };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * FT_ROWS;
  int chunks = I / NC;
  constexpr int IT1 = (C * (NC / 8) + NTH - 1) / NTH, IT2 = (NC * (C / 8) + NTH - 1) / NTH;
  static_assert((NTH / (NC / 8)) % 8 == 0 && (NTH / (C / 8)) % 8 == 0, "stage row steps preserve swizzle bits");
  const int k0_1 = (int)threadIdx.x / (NC / 8), c0_1 = ((int)threadIdx.x % (NC / 8)) * 8, sw1 = stageSw<NC>(k0_1, c0_1);
  const int k0_2 = (int)threadIdx.x / (C / 8), c0_2 = ((int)threadIdx.x % (C / 8)) * 8, sw2 = w2Sw<C>(k0_2, c0_2);
  auto stage = [&](int j, int st, auto&& op) {
    half *a = W1a(st) + sw1, *b = W1b(st) + sw1, *w2 = W2s(st) + sw2;
    const half* src1 = W1t + ((size_t)j * C + k0_1) * NC + c0_1;
    const half* src2 = W2 + (size_t)(j * NC + k0_2) * C + c0_2;
#pragma unroll
    for (int it = 0; it < IT1; ++it) {                            // W1 rows k, NC columns each half
      if ((C * (NC / 8)) % NTH != 0 && it * NTH + (int)threadIdx.x >= C * (NC / 8)) break;
      int off = it * (NTH / (NC / 8)) * NC;
      const half* src = src1 + off;
      op(it, a + off, src);
      if constexpr (!RELU) op(IT1 + it, b + off, src + (size_t)C * I);
    }
#pragma unroll
    for (int it = 0; it < IT2; ++it) {                            // W2 rows j*NC .. , all C columns
      if ((NC * (C / 8)) % NTH != 0 && it * NTH + (int)threadIdx.x >= NC * (C / 8)) break;
      int off = it * (NTH / (C / 8)) * C;
      op(2 * IT1 + it, w2 + off, src2 + off);
    }
  };
  auto issue = [&](int j, int st) { stage(j, st, [](int, half* d, const half* s) { cpAsync16(d, s, true); }); cpCommit(); };
#if LF_REG_STAGES
  RegStage<2 * IT1 + IT2> next;       // (sm_75: the next chunk's weights held in registers across this chunk)
#endif
  issue(0, 0);
  // LayerNorm, a warp a row (C / 32 floats a lane), into shared memory as f16
  lnRowsToShared<C, FT_ROWS, WARPS, PT, (!RELU && C == 64)>(x, [&](int r) { size_t row = row0 + r; return row < rows ? row : SIZE_MAX; },
                                    lnScale, lnOffset, Xs, LDX, warp, lane);
#if !LF_REG_STAGES
  cpWait<0>();
#endif
  __syncthreads();
  uint32_t xa[MT][KS][4];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int ks = 0; ks < KS; ++ks)
      ldsm4(xa[m][ks], Xs + ((warp * MT + m) * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  }
  __syncthreads();
#if !LF_REG_STAGES
  if (NST == 2 && 1 < chunks) issue(1, 1);
#endif
  float acc[MT][NT][4] = {};
  for (int j = 0; j < chunks; ++j) {
    int st = NST == 1 ? 0 : j & 1;
    // chunk j has landed; the barrier also says every warp is done with the stage the next issue
    // overwrites (chunk j - 1's), so one barrier a chunk
    float2 b1v[NC / 8];
    if constexpr (RELU) {
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) b1v[nt] = ptx_ldg_v2_f32(b1 + j * NC + nt * 8 + tig * 2);
    }
#if LF_REG_STAGES
    if (j + 1 < chunks) stage(j + 1, st ^ 1, [&](int i, half*, const half* src) { next.load(i, src); });
    __syncthreads();
#else
    if (j > 0) {
      cpWait<0>();
      __syncthreads();
    }
#endif
    const half *a = W1a(st), *b = W1b(st), *w2 = W2s(st);
    float ha[MT][NC / 8][4] = {}, hb[MT][NC / 8][4] = {};
    constexpr int N2 = NC / 16, STEPS1 = KS * N2;
    const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
    uint32_t vb[2][4];
    {
      uint32_t fa[2][4], fb[2][4];
      ldsm4t(fa[0], a + stageSw<NC>(kl, cl));
      if constexpr (!RELU) ldsm4t(fb[0], b + stageSw<NC>(kl, cl));
#if !LF_REG_STAGES
      if (j > 0 && NST == 2 && j + 1 < chunks) issue(j + 1, st ^ 1);
#endif
#pragma unroll
      for (int s = 0; s < STEPS1; ++s) {
        const int cur = s & 1, ks = s / N2, n2 = s % N2;
        if (s + 1 < STEPS1) {
          const int nks = (s + 1) / N2, nn2 = (s + 1) % N2;
          ldsm4t(fa[cur ^ 1], a + stageSw<NC>(nks * 16 + kl, nn2 * 16 + cl));
          if constexpr (!RELU) ldsm4t(fb[cur ^ 1], b + stageSw<NC>(nks * 16 + kl, nn2 * 16 + cl));
        } else {
          ldsm4t(vb[0], w2 + w2Sw<C>(kl, cl));
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          mma16816(ha[m][2 * n2], xa[m][ks], fa[cur][0], fa[cur][1]);
          mma16816(ha[m][2 * n2 + 1], xa[m][ks], fa[cur][2], fa[cur][3]);
        }
        if constexpr (!RELU) {
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            mma16816(hb[m][2 * n2], xa[m][ks], fb[cur][0], fb[cur][1]);
            mma16816(hb[m][2 * n2 + 1], xa[m][ks], fb[cur][2], fb[cur][3]);
          }
        }
      }
    }
    // SwiGLU in f32, straight into the second GEMM's A fragments
#pragma unroll
    for (int t = 0; t < NC / 16; ++t) {
      if (t > 0) ldsm4t(vb[0], w2 + w2Sw<C>(t * 16 + kl, cl));
      uint32_t pa[MT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        auto gate = [&](int nt, int e) {
          if constexpr (RELU) return ptx_max_f32(ptx_add_f32(ha[m][nt][e], (e & 1) ? b1v[nt].y : b1v[nt].x), 0.f);
          else { float v = ha[m][nt][e]; return ptx_mul_f32(ptx_mul_f32(v, sigmH(v)), hb[m][nt][e]); }   // packed to f16 next
        };
        pa[m][0] = pack2(gate(2 * t, 0), gate(2 * t, 1)); pa[m][1] = pack2(gate(2 * t, 2), gate(2 * t, 3));
        pa[m][2] = pack2(gate(2 * t + 1, 0), gate(2 * t + 1, 1)); pa[m][3] = pack2(gate(2 * t + 1, 2), gate(2 * t + 1, 3));
      }
#pragma unroll
      for (int et = 0; et < NT; et += 2) {
        const int cur = (et >> 1) & 1;
        if (et + 2 < NT) ldsm4t(vb[cur ^ 1], w2 + w2Sw<C>(t * 16 + kl, (et + 2) * 8 + cl));
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          mma16816(acc[m][et], pa[m], vb[cur][0], vb[cur][1]);
          mma16816(acc[m][et + 1], pa[m], vb[cur][2], vb[cur][3]);
        }
      }
    }
    if constexpr (NST == 1) __syncthreads();      // (every warp done with the one stage before the next chunk takes it)
#if LF_REG_STAGES
    if (j + 1 < chunks) stage(j + 1, NST == 1 ? 0 : st ^ 1, [&](int i, half* d, const half*) { next.store(i, d); });
#else
    if (NST == 1 && j + 1 < chunks) issue(j + 1, 0);
#endif
  }
  __syncthreads();
  // the residual, through the warp's own Xs rows (only it read them, as its A fragments; the loop's
  // barriers are behind every warp) 64 f32 columns at a time, so the read-modify-write is 16 bytes a lane
  // of whole rows, where a fragment's layout was 8 bytes in each of 8 rows
  constexpr int RW = 16 * MT, LDY = C / 2 + 4, LPR = 32 / (C / 8);
  static_assert((C == 64 || C == 128) && RW * LDY * 4 <= RW * LDX * 2, "a half of the warp's accumulators fits in its rows");
  float* Ys = reinterpret_cast<float*>(Xs + warp * RW * LDX);
  const int rl0 = lane / (C / 8), cl = (lane % (C / 8)) * 4;
  const size_t baseRow = row0 + warp * RW + rl0;
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const int c = hh * (C / 2) + cl;
    float4 b4;
    if constexpr (RELU) b4 = ptx_ldg_v4_f32(b2 + c);
    PairRaw4<PT> pv[MT == 1 ? RW / LPR : 1];
    if constexpr (MT == 1) {
#pragma unroll
      for (int i = 0; i < RW / LPR; ++i) {
        size_t row = baseRow + i * LPR;
        if (row < rows) pv[i] = pairLdRaw4<PT>(x, row * C + c);
      }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
      for (int e8 = 0; e8 < NT / 2; ++e8) {
        int et = hh * (NT / 2) + e8, c8 = e8 * 8 + tig * 2;
        ptx_st_shared_v2_f32(Ys + (m * 16 + g) * LDY + c8, acc[m][et][0], acc[m][et][1]);
        ptx_st_shared_v2_f32(Ys + (m * 16 + g + 8) * LDY + c8, acc[m][et][2], acc[m][et][3]);
      }
    }
    __syncwarp();
    if constexpr (MT == 1) {
      float4 a4All[RW / LPR];
#pragma unroll
      for (int i = 0; i < RW / LPR; ++i)
        a4All[i] = ptx_ld_shared_v4_f32(Ys + (rl0 + i * LPR) * LDY + cl);
#pragma unroll
      for (int i = 0; i < RW / LPR; ++i) {
        size_t row = baseRow + i * LPR;
        if (row >= rows) continue;
        float4 a4 = a4All[i];
        float4 v = pairUnpack4<PT>(pv[i]);
        if constexpr (RELU) {
          a4.x = ptx_add_f32(a4.x, b4.x); a4.y = ptx_add_f32(a4.y, b4.y); a4.z = ptx_add_f32(a4.z, b4.z); a4.w = ptx_add_f32(a4.w, b4.w);
        }
        v.x = ptx_add_f32(v.x, a4.x); v.y = ptx_add_f32(v.y, a4.y); v.z = ptx_add_f32(v.z, a4.z); v.w = ptx_add_f32(v.w, a4.w);
        pairSt4<PT>(x, row * C + c, v);
      }
    } else {
      constexpr int N_EPI = RW / LPR;
      constexpr int EPI_B = (N_EPI % 4 == 0) ? 4 : 1;
#pragma unroll
      for (int i0 = 0; i0 < N_EPI; i0 += EPI_B) {
        PairRaw4<PT> pvr[EPI_B];
        float4 a4v[EPI_B];
#pragma unroll
        for (int d = 0; d < EPI_B; ++d) {
          int i = i0 + d;
          size_t row = baseRow + i * LPR;
          if (row < rows) {
            a4v[d] = ptx_ld_shared_v4_f32(Ys + (rl0 + i * LPR) * LDY + cl);
            pvr[d] = pairLdRaw4<PT>(x, row * C + c);
          }
        }
#pragma unroll
        for (int d = 0; d < EPI_B; ++d) {
          int i = i0 + d;
          size_t row = baseRow + i * LPR;
          if (row >= rows) continue;
          float4 a4 = a4v[d];
          float4 v = pairUnpack4<PT>(pvr[d]);
          if constexpr (RELU) {
            a4.x = ptx_add_f32(a4.x, b4.x); a4.y = ptx_add_f32(a4.y, b4.y); a4.z = ptx_add_f32(a4.z, b4.z); a4.w = ptx_add_f32(a4.w, b4.w);
          }
          v.x = ptx_add_f32(v.x, a4.x); v.y = ptx_add_f32(v.y, a4.y); v.z = ptx_add_f32(v.z, a4.z); v.w = ptx_add_f32(v.w, a4.w);
          pairSt4<PT>(x, row * C + c, v);
        }
      }
    }
    if (hh == 0) __syncwarp();
  }
}

// AlphaFold 2's transition at C 256 (its MSA, bf16 rows - the MSA16 stack's) in one kernel. fusedTransitionK holds a
// warp's whole 16 x C output, which at C 256 is 128 accumulators a lane beside the 64 of its LayerNorm'd rows (255
// registers). Here a PAIR of warps shares 16 rows and splits the output's columns: each computes half of a 32-column
// chunk of the intermediate (ReLU'd, into shared memory), the pair meets at a named barrier, and each takes the whole
// chunk against its half of W2 - 64 accumulators a lane. WM row tiles (2 * WM warps). The LayerNorm'd rows go to
// registers once; their shared memory then holds the chunk's intermediate, and all of it the residual's staging.
// On an RTX PRO 6000 (sm_120) at WM 4, 1,000 residues x 508 rows: 2.41 -> 2.00 ms a call against the LayerNorm and two
// cuBLASLt GEMMs it replaces; on an A100 3.78 against 3.57, so it is sm_12x's (af2 evoformer.cuh)
constexpr int FTS_NC = 32;
template <int C, int WM>
__host__ __device__ constexpr size_t ftSplitSmem() {
  constexpr size_t xs = (size_t)16 * WM * (C + 8) * 2, hs = (size_t)2 * 16 * WM * (FTS_NC + 8) * 2;
  return (xs > hs ? xs : hs) + 2 * (size_t)2 * C * FTS_NC * 2;
}
template <int C, int WM, class PT>
__global__ void __launch_bounds__(WM * 64) fusedTransitionSplitK(float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ W1t, const half* __restrict__ W2, size_t rows, int I,
    const float* __restrict__ b1, const float* __restrict__ b2) {
  constexpr int NC = FTS_NC, WARPS = 2 * WM, NTH = 32 * WARPS, R = 16 * WM, LDX = C + 8, KS = C / 16, NTO = C / 16, LDH = NC + 8;
  constexpr size_t XS = (size_t)R * LDX * 2 > (size_t)2 * R * LDH * 2 ? (size_t)R * LDX * 2 : (size_t)2 * R * LDH * 2;
  constexpr size_t STAGE = (size_t)2 * C * NC * 2;
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem;                                    // [R][LDX], then the intermediate [2][R][LDH]
  unsigned char* stages = smem + XS;
  auto W1s = [&](int s) { return (half*)(stages + s * STAGE); };
  auto W2s = [&](int s) { return W1s(s) + C * NC; };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, wm = warp >> 1, wn = warp & 1;
  size_t row0 = (size_t)blockIdx.x * R;
  int chunks = I / NC;
  constexpr int IT1 = C * (NC / 8) / NTH, IT2 = NC * (C / 8) / NTH;
  static_assert(C * (NC / 8) % NTH == 0 && NC * (C / 8) % NTH == 0, "whole stage loads");
  auto issue = [&](int j, int st) {
    half *a = W1s(st), *w2 = W2s(st);
#pragma unroll
    for (int it = 0; it < IT1; ++it) {
      const int t = it * NTH + (int)threadIdx.x, k = t / (NC / 8), c = (t % (NC / 8)) * 8;
      cpAsync16(a + stageSw<NC>(k, c), W1t + ((size_t)j * C + k) * NC + c, true);
    }
#pragma unroll
    for (int it = 0; it < IT2; ++it) {
      const int t = it * NTH + (int)threadIdx.x, k = t / (C / 8), c = (t % (C / 8)) * 8;
      cpAsync16(w2 + w2Sw<C>(k, c), W2 + (size_t)(j * NC + k) * C + c, true);
    }
    cpCommit();
  };
  issue(0, 0);
  lnRowsToShared<C, R, WARPS, PT>(x, [&](int r) { size_t row = row0 + r; return row < rows ? row : SIZE_MAX; },
                                  lnScale, lnOffset, Xs, LDX, warp, lane);
  __syncthreads();
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (wm * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  float acc[NTO][4] = {};
  for (int j = 0; j < chunks; ++j) {
    int st = j & 1;
    // chunk j's weights have landed, every warp has its rows (j 0) and is done with chunk j - 1's stage and the
    // intermediate buffer the chunk after next writes
    cpWait<0>();
    __syncthreads();
    if (j + 1 < chunks) issue(j + 1, st ^ 1);
    const half *a = W1s(st), *w2 = W2s(st);
    half* Hs = Xs + st * R * LDH;
    float ha[2][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {           // this warp's 16 of the chunk's 32 columns
      uint32_t fa[4];
      ldsm4t(fa, a + stageSw<NC>(ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), wn * 16 + (lane >> 4) * 8));
      mma16816(ha[0], xa[ks], fa[0], fa[1]); mma16816(ha[1], xa[ks], fa[2], fa[3]);
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      int c = wn * 16 + nt * 8 + tig * 2;
      float2 bb = ptx_ldg_v2_f32(b1 + j * NC + c);
      ptx_st_shared_u32(Hs + (wm * 16 + g) * LDH + c, ptx_cvt_f16x2_f32(ptx_max_f32(ptx_add_f32(ha[nt][0], bb.x), 0.f), ptx_max_f32(ptx_add_f32(ha[nt][1], bb.y), 0.f)));
      ptx_st_shared_u32(Hs + (wm * 16 + g + 8) * LDH + c, ptx_cvt_f16x2_f32(ptx_max_f32(ptx_add_f32(ha[nt][2], bb.x), 0.f), ptx_max_f32(ptx_add_f32(ha[nt][3], bb.y), 0.f)));
    }
    asm volatile("bar.sync %0, 64;" ::"r"(1 + wm) : "memory");     // the pair's two halves
#pragma unroll
    for (int t = 0; t < NC / 16; ++t) {
      uint32_t pa[4];
      ldsm4(pa, Hs + (wm * 16 + (lane & 15)) * LDH + t * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int et = 0; et < NTO; et += 2) {
        uint32_t vb[4];
        ldsm4t(vb, w2 + w2Sw<C>(t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), wn * (C / 2) + (et + (lane >> 4)) * 8));
        mma16816(acc[et], pa, vb[0], vb[1]); mma16816(acc[et + 1], pa, vb[2], vb[3]);
      }
    }
  }
  // the residual, 64 columns a pass through the shared memory (free once every warp is past its last chunk)
  __syncthreads();
  constexpr int LDY = 64 + 4;
  float* Ys = reinterpret_cast<float*>(smem) + warp * 16 * LDY;
  static_assert((size_t)WARPS * 16 * LDY * 4 <= XS + 2 * STAGE, "the staging fits");
#pragma unroll
  for (int q = 0; q < C / 128; ++q) {
#pragma unroll
    for (int e8 = 0; e8 < 8; ++e8) {
      int c = e8 * 8 + tig * 2;
      ptx_st_shared_v2_f32(Ys + g * LDY + c, acc[q * 8 + e8][0], acc[q * 8 + e8][1]);
      ptx_st_shared_v2_f32(Ys + (g + 8) * LDY + c, acc[q * 8 + e8][2], acc[q * 8 + e8][3]);
    }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int rl = i * 2 + (lane >> 4), cl = (lane & 15) * 4, c = wn * (C / 2) + q * 64 + cl;
      size_t row = row0 + wm * 16 + rl;
      if (row >= rows) continue;
      float4 a4 = ptx_ld_shared_v4_f32(Ys + rl * LDY + cl);
      float4 b4 = ptx_ldg_v4_f32(b2 + c);
      float4 v = pairLd4<PT>(x, row * C + c);
      v.x = ptx_add_f32(v.x, ptx_add_f32(a4.x, b4.x));
      v.y = ptx_add_f32(v.y, ptx_add_f32(a4.y, b4.y));
      v.z = ptx_add_f32(v.z, ptx_add_f32(a4.z, b4.z));
      v.w = ptx_add_f32(v.w, ptx_add_f32(a4.w, b4.w));
      pairSt4<PT>(x, row * C + c, v);
    }
    __syncwarp();
  }
}
// x += ReLU(LN(x) W1 + b1) W2 + b2 at C 256 on bf16 rows, the split form above at WM 4; false where the device cannot
// hold its block (99 KB)
inline bool fusedTransitionSplit256(float* x, size_t rows, int I, const float* lnScale, const float* lnOffset, const half* W1,
                                    const half* W2, const float* b1, const float* b2) {
  constexpr int C = 256, WM = 4, R = 16 * WM;
  constexpr size_t smem = ftSplitSmem<C, WM>();
  if (I % FTS_NC || !fitsSmem(smem)) return false;
  using PT = __nv_bfloat16;
  const half* w1t = transitionW1Tiles<true>(W1, C, I, FTS_NC);
  static bool attr = false;
  if (!attr) { smemAttr((fusedTransitionSplitK<C, WM, PT>), (int)smem); attr = true; }
  fusedTransitionSplitK<C, WM, PT><<<(unsigned)((rows + R - 1) / R), 64 * WM, smem, STREAM>>>(x, lnScale, lnOffset, w1t, W2, rows, I, b1, b2);
  return true;
}

// one launch of a given form (the bench's arms; fusedTransitionRaw picks the form)
template <int WARPS, bool RELU = false, int MT = 1, int NC = FT_NC, int NST = 2, int C = 128>
void fusedTransitionAt(float* x, size_t rows, int I, const float* lnScale, const float* lnOffset, const half* W1,
                       const half* W2, const float* b1, const float* b2) {
  constexpr int R = 16 * WARPS * MT;
  constexpr size_t smem = ftSmem<C, WARPS, MT, NC, RELU>();
  const half* w1t = transitionW1Tiles<RELU>(W1, C, I, NC);
  WITH_PAIR_T(
    static bool attr = false;
    if (!attr) { smemAttr((fusedTransitionK<C, WARPS, RELU, MT, NC, PT, NST>), (int)smem); attr = true; }
    fusedTransitionK<C, WARPS, RELU, MT, NC, PT, NST><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
      x, lnScale, lnOffset, w1t, W2, rows, I, b1, b2));
}
inline bool FUSED_TRANSITION = true;
#ifndef FT_ONE_STAGE
#define FT_ONE_STAGE 1
#endif
inline int FT_WARPS = 8;
inline bool FT_TWO_TILES = !getenv("LOCALFOLD_FT_ONE_TILE");
// x += transition(x) for C = 128, f16 weights, on raw pointers (AlphaFold 2 calls it with RELU and its
// biases); false if the shape is not this kernel's or the device cannot hold its blocks
// whether fusedTransitionRaw has a form for this shape on this device (the two-tile form, else the one-tile at 4 warps)
inline bool fusedTransitionFits(size_t rows, int C, int I) {
  if (!FUSED_TRANSITION) return false;
  if (C == 64) return (I % 16 == 0) && fitsSmem(ftSmem<64, 4, 1, 16, true>());
  if (C != 128 || I % FT_NC) return false;
  if (FT_TWO_TILES && (rows + 127) / 128 >= MIN_BLOCKS && fitsSmem(ftSmem<128, 4, 2, 16>())) return true;
  return fitsSmem(ftSmem<128, 4, 1, FT_NC>());
}
template <bool RELU = false>
inline bool fusedTransitionRaw(float* x, size_t rows, int C, int I, const float* lnScale, const float* lnOffset,
                               const half* W1, const half* W2, const float* b1 = nullptr, const float* b2 = nullptr) {
  if (!fusedTransitionFits(rows, C, I)) return false;
  if (C == 64) {
    fusedTransitionAt<4, RELU, 1, 16, 2, 64>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
    return true;
  }
  // two 16-row tiles a warp on 4 warps (128-row blocks, 68 KB): 1.14x the 8-warp one-tile form at 1,044
  // tokens (--bench-trans), bit-identical; where it does not fit (a T4) or the input is small, the one-tile forms
  // AF2's (ReLU) on sm_12x: one tile a warp on SIXTEEN warps, 256 rows a block (94 KB), each weight stage spread over
  // twice the rows - an RTX PRO 6000's 1,000-residue pair transition 1.20 -> 0.94 ms a call (--bench-pair: two tiles
  // on 4 warps 1.19, one tile on 8 1.03), bit-identical. AF3's SwiGLU form there is unmeasured and keeps the rule below
  if constexpr (RELU) {
    if (ccMajor() == 12 && (rows + 255) / 256 >= MIN_BLOCKS && fitsSmem(ftSmem<128, 16, 1, 16, RELU>())) {
      fusedTransitionAt<16, RELU, 1, 16>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
      return true;
    }
  }
  // a part without cp.async (a T4: register-staged, LF_REG_STAGES): ONE weight stage, so two 64-row blocks of the one-tile
  // form fit its 64 KB SM (29 KB each) where every two-stage form held one block - 4 warps an SM
  if (FT_ONE_STAGE && ccMajor() < 8 && (rows + 63) / 64 >= MIN_BLOCKS && fitsSmem(ftSmem<128, 4, 1, 16, RELU>())) {
    fusedTransitionAt<4, RELU, 1, 16, 1>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
    return true;
  }
  if (FT_TWO_TILES && (rows + 127) / 128 >= MIN_BLOCKS && fitsSmem(ftSmem<128, 4, 2, 16, RELU>())) {
    static const bool oneTile = [] {
      if (const char* e = getenv("LOCALFOLD_FT_FORM")) return atoi(e) == 1;
      int dev, perSm = 0; CK(cudaGetDevice(&dev));
      CK(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
      return (size_t)perSm >= 3 * (ftSmem<128, 4, 1, 16, RELU>() + 1024);
    }();
    if (oneTile) fusedTransitionAt<4, RELU, 1, 16>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
    else fusedTransitionAt<4, RELU, 2, 16>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
    return true;
  }
  if ((rows + 63) / 64 >= MIN_BLOCKS && fitsSmem(ftSmem<128, 4, 1, 16, RELU>())) {
    fusedTransitionAt<4, RELU, 1, 16>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
    return true;
  }
  int warps = FT_WARPS == 8 && (rows + 127) / 128 < MIN_BLOCKS ? 4 : FT_WARPS;   // a small input: more, smaller blocks
  // no more than the device's shared memory (a T4's 64 KB takes the 4-warp form, or the unfused one)
  auto smemFor = [](int w) { return (size_t)16 * w * (128 + 8) * 2 + ftStage<128, FT_NC, RELU>(); };
  while (warps > 4 && !fitsSmem(smemFor(warps))) warps /= 2;
  if (!fitsSmem(smemFor(warps))) return false;
  if (warps == 4) fusedTransitionAt<4, RELU>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
  else fusedTransitionAt<8, RELU>(x, rows, I, lnScale, lnOffset, W1, W2, b1, b2);
  return true;
}
inline bool fusedTransition(float* x, size_t rows, int C, int I, const std::string& pre) {
  if (!FUSED_TRANSITION || C != 128 || I % FT_NC) return false;
  return fusedTransitionRaw(x, rows, C, I, W(pre + ".inputLayerNormScale"), W(pre + ".inputLayerNormOffset"),
                            Wh(pre + ".transition1"), Wh(pre + ".transition2"));
}
