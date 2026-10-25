// Fused kernels for a 256-channel pair track (ESMFold2's, protenix2's), on the f16 tensor cores (mma.sync m16n8k16).
// cuda/af3's fused kernels hold a whole 256-wide output tile or weight on the chip, which at 256
// channels is 255 registers a thread or more shared memory than a block gets; these stream instead.
//
//   transitionUpK:   gated[r] = SwiGLU(LN(x[r]) W1)            the [P, 2I] widening never leaves the chip
//   triangleOutK:    pair[r] += (LN_c(prod[:, r]) Wout) * sigmoid(t2[r])
//                    the centre norm, the output projection, the gate and the residual in one pass,
//                    the product read channel-major (the contraction's layout) and Wout streamed
//                    through shared memory 32 output columns at a time
#pragma once
#include "fusedtriangle.cuh"

// ---------------------------------------------------------------- the transition's widening
// A block is 16 WARPS rows; the LN'd rows stay in registers as MMA A fragments; W1's [gate | value]
// halves are walked 32 columns at a time (each chunk and its SwiGLU partner), double-buffered.
// W1t: W1 [C][2I] as tileTransitionUp lays it out - the gate half's 32-column tiles, then the value half's
// NC, XROUNDS: the output columns a stage holds (W1t's tiles stay 32 wide; a 16-column stage is half of
// one) and, past 1, triIn256K's rounds - a T4's form: 16 warps, their rows normed 32 at a time beside 16-column
// stages, ~49 KB where the 8-warp form is 67.6 KB. Every output's sum over k runs in the same order in all
// forms, so they are byte-identical.
// TG: the gated rows' type (bf16 for a second GEMM that accumulates straight into a bf16 pair)
// MINB: blocks an SM the registers are held to. The 8-warp form at 256 channels is 67.6 KB of shared memory, two
// blocks an SM on an A100 - as transitionUpWaveRows and the launchers were written for - but it had grown to 160
// registers, so the registers held it to ONE (Nsight Compute at 988 tokens: occupancy 12.5%, the tensor pipe 48%).
// Held to 128 (no spills: LOCAL 0) it runs two: ESMFold2's 676 -> 601 ms of a 988-token fold, byte-identical.
// Not past 256 channels (the rows' fragments alone are 96-128 registers there, and two blocks do not fit anyway).
template <int C, int WARPS, int NC = 32, int XROUNDS = 1, class PT = float, class TG = half,
          int MINB = (C <= 256 && WARPS == 8 && XROUNDS == 1) ? 2 : 1>
#if LF_REG_STAGES
__global__ void __launch_bounds__(WARPS * 32, 1)
#else
__global__ void __launch_bounds__(WARPS * 32, MINB)
#endif
transitionUpK(const float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ W1t, TG* __restrict__ gated, size_t rows, int I) {
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, KS = C / 16;
  constexpr size_t STAGE = (size_t)2 * C * NC * 2;
  auto sw = [](int k, int c) { return stageSw<NC>(k, c); };
  extern __shared__ __align__(16) unsigned char smem[];
  // the LN'd rows and the weight stages share memory: the rows go into registers (A fragments) before
  // the first stage is issued, so a block's footprint is the larger of the two, not their sum (in rounds,
  // only stage 0 is issued before the round loop, so the rows share stage 1's memory at smem + STAGE)
  half* Xs = XROUNDS == 1 ? (half*)smem : (half*)(smem + STAGE);
  unsigned char* stages = smem;
  auto Wa = [&](int s) { return (half*)(stages + s * STAGE); };
  auto Wb = [&](int s) { return Wa(s) + C * NC; };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  int chunks = I / NC;
  constexpr int ITER = (C * (NC / 8) + NTH - 1) / NTH;
  const int k0 = (int)threadIdx.x / (NC / 8), c0_ = ((int)threadIdx.x % (NC / 8)) * 8, sw0 = sw(k0, c0_);
  auto stage = [&](int j, int st, auto&& op) {
    half *a = Wa(st) + sw0, *b = Wb(st) + sw0;
    const half* base = W1t + ((size_t)(j * NC / 32) * C + k0) * 32 + (j * NC) % 32 + c0_;   // (32-column tiles)
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
      if ((C * (NC / 8)) % NTH != 0 && it * NTH + (int)threadIdx.x >= C * (NC / 8)) break;
      const half* src = base + (size_t)it * (NTH / (NC / 8)) * 32;
      int off = it * (NTH / (NC / 8)) * NC;
      op(2 * it, a + off, src);
      op(2 * it + 1, b + off, src + (size_t)C * I);
    }
  };
  auto issue = [&](int j, int st) { stage(j, st, [](int, half* d, const half* s) { cpAsync16(d, s, true); }); cpCommit(); };
#if LF_REG_STAGES
  RegStage<2 * ITER> next;            // (sm_75: the next chunk's weights held in registers across this chunk)
#endif
  uint32_t xa[KS][4];
  if constexpr (XROUNDS == 1) {
    lnRowsToShared<C, R, WARPS, PT>(x, [&](int r) { size_t row = row0 + r; return row < rows ? row : SIZE_MAX; },
                                    lnScale, lnOffset, Xs, LDX, warp, lane);
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
    __syncthreads();                    // every warp has its fragments: the stages may overwrite the rows
    issue(0, 0);
  } else {
    static_assert(R % XROUNDS == 0 && (R / XROUNDS) % 16 == 0 && (R / XROUNDS) % WARPS == 0, "whole warps' rows a round");
    constexpr int XR = R / XROUNDS;
    issue(0, 0);
#pragma unroll 1
    for (int rd = 0; rd < XROUNDS; ++rd) {
      lnRowsToShared<C, XR, WARPS, PT>(x, [&](int r) { size_t row = row0 + rd * XR + r; return row < rows ? row : SIZE_MAX; },
                                   lnScale, lnOffset, Xs, LDX, warp, lane);
      __syncthreads();
      if (warp * 16 / XR == rd) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (warp * 16 - rd * XR + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
      }
      __syncthreads();
    }
  }
  for (int j = 0; j < chunks; ++j) {
    int st = j & 1;
#if LF_REG_STAGES
    if (j + 1 < chunks) stage(j + 1, st ^ 1, [&](int i, half*, const half* src) { next.load(i, src); });
    __syncthreads();
#else
    cpWait<0>();
    __syncthreads();
    if (j + 1 < chunks) issue(j + 1, st ^ 1);
#endif
    const half *a = Wa(st), *b = Wb(st);
    float ha[NC / 8][4] = {}, hb[NC / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int n2 = 0; n2 < NC / 16; ++n2) {
        uint32_t fa[4], fb[4];
        int k = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), c = n2 * 16 + (lane >> 4) * 8;
        ldsm4t(fa, a + sw(k, c));
        ldsm4t(fb, b + sw(k, c));
        mma16816(ha[2 * n2], xa[ks], fa[0], fa[1]); mma16816(ha[2 * n2 + 1], xa[ks], fa[2], fa[3]);
        mma16816(hb[2 * n2], xa[ks], fb[0], fb[1]); mma16816(hb[2 * n2 + 1], xa[ks], fb[2], fb[3]);
      }
    }
    // (staging these through shared memory for 16-byte stores measured SLOWER, 92.6 -> 112 ms: it took
    // the block past the 82 KB at which two fit an SM, and added a barrier a chunk)
    size_t r0 = row0 + warp * 16 + g, r1 = r0 + 8;
#pragma unroll
    for (int nt = 0; nt < NC / 8; ++nt) {
      auto gate = [&](int e) { float v = ha[nt][e]; return ptx_mul_f32(ptx_mul_f32(v, sigmH(v)), hb[nt][e]); };   // rounded to f16 next
      int c = j * NC + nt * 8 + tig * 2;
      if constexpr (std::is_same_v<TG, half>) {
        if (r0 < rows) ptx_st_global_u32(gated + r0 * I + c, ptx_cvt_f16x2_f32(gate(0), gate(1)));
        if (r1 < rows) ptx_st_global_u32(gated + r1 * I + c, ptx_cvt_f16x2_f32(gate(2), gate(3)));
      } else {
        if (r0 < rows) ptx_st_global_u32(gated + r0 * I + c, ptx_cvt_bf16x2_f32(gate(0), gate(1)));
        if (r1 < rows) ptx_st_global_u32(gated + r1 * I + c, ptx_cvt_bf16x2_f32(gate(2), gate(3)));
      }
    }
#if LF_REG_STAGES
    if (j + 1 < chunks) stage(j + 1, st ^ 1, [&](int i, half* d, const half*) { next.store(i, d); });
    __syncthreads();
#endif
  }
}

// transitionUpK's weights as it streams them (see tileColumns): 2 C I halves, each half's
// 32-column tiles [C][32] contiguous
inline void tileTransitionUp(const half* W1, int C, int I, half* out) {
  tileColumns(W1, C, 2 * I, 0, I, 32, out);
  tileColumns(W1, C, 2 * I, I, I, 32, out + (size_t)C * I);
}
// transitionUpK's dynamic shared memory: the larger of the LN'd rows and the two weight stages
// (in rounds, Xs shares stage 1 at smem + STAGE while stage 0 is in flight during the prologue)
template <int C, int WARPS, int NC = 32, int XROUNDS = 1>
constexpr size_t transitionUpSmem() {
  constexpr size_t stage = (size_t)2 * C * NC * 2, rows = (size_t)16 * WARPS / XROUNDS * (C + 8) * 2;
  if constexpr (XROUNDS > 1) return stage + (rows > stage ? rows : stage);
  else return rows > 2 * stage ? rows : 2 * stage;
}

// The rows one full wave of transitionUpK<C, WARPS> covers - the blocks every multiprocessor holds at
// once, times the multiprocessors, times a block's 16 * WARPS rows - so a caller that chunks its rows can
// chunk in whole waves. A chunk of 32768 rows was 256 blocks on an A100 at two a multiprocessor, 1.19
// waves: two rounds with the second nearly empty, then a 25-block tail at 0.12 waves - five rounds of the
// device where three do (Nsight Compute, ESMFold2 at 262 tokens).
template <int C, int WARPS, int NC = 32, int XROUNDS = 1>
size_t transitionUpWaveRows(size_t smem) {
  static size_t rows = [smem] {
    smemAttr((transitionUpK<C, WARPS, NC, XROUNDS>), (int)smem);
    int perSm = 0, dev = 0, sms = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, transitionUpK<C, WARPS, NC, XROUNDS>, 32 * WARPS, smem));
    CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    return (size_t)std::max(1, perSm) * sms * 16 * WARPS;
  }();
  return rows;
}
// ...and the largest whole number of those waves inside a budget of rows (at least one wave)
template <int C, int WARPS, int NC = 32, int XROUNDS = 1>
size_t transitionUpChunkRows(size_t smem, size_t budgetRows) {
  size_t wave = transitionUpWaveRows<C, WARPS, NC, XROUNDS>(smem);
  return std::max<size_t>(1, budgetRows / wave) * wave;
}

// ---------------------------------------------------------------- the triangle's output side
// RectMap: the kernels' rows as a RECTANGLE of the padded pair space (triangleBlocked's blocks) - rows [i0, i0 + size / J),
// columns [j0, j0 + J), q = (i - i0) J + (j - j0) a row's own index, and every operand and output plane `size` long; J 0:
// the whole padded plane, as before
// T (the input kernel): its rows read the pair TRANSPOSED - row (u, v) takes pair (v, u) - so the
// incoming triangle's operands come out with the contraction's index contiguous, as the outgoing one's do (an FP8 GEMM
// takes only that layout); its t2 rows still land at their own pair's place, where the output kernel reads them
// ab0, ab1: the input kernel's a/b steps [ab0, ab1) alone (triIn256K; -1 all of them)
// stride: the output kernel's product planes' length where it is not `size` (a chunk of a rectangle's rows, its product
// handed from the chunk's first row: sharded.cuh's gate made a chunk at a time)
struct RectMap { int i0 = 0, J = 0, j0 = 0; size_t size = 0; bool T = false; int ab0 = 0, ab1 = -1; size_t stride = 0; };
// prod: channel-major [C][Lp][Lp] f32 (the padded contraction's output); t2: the gating linear's raw
// output, [Lp * Lp][C] f16 (triInK's); pair: [L * L][C] f32. A block is 16 WARPS pairs: the product
// tile is staged channel-major (stride R + 1: the per-row reductions read a column, conflict-free), the
// centre norm's statistics taken per row, and each warp's A fragments built straight from the tile,
// normalised on the way; the tile's memory then holds Wout's stages, 32 output columns at a time.
// TT, NC: the tile's type and the output columns a weight stage holds. A float tile at 32 columns is 66.5 KB a
// block (two an SM on an A100); a bf16 tile at 16 is 34 KB (four, the registers' limit) - the product held
// at AF3's activation precision, which is what AF3's own triangle runs in
template <int C, int WARPS, class TT = float, int NC = 32, class TP = float>
__host__ __device__ constexpr size_t triangleOutSmem() {
  constexpr int R = 16 * WARPS;
  constexpr bool VEC = sizeof(TT) == 2 && sizeof(TP) == 2;      // (triangleOutK's 16-byte product loads)
  constexpr size_t tile = (size_t)C * (R + (VEC ? 8 : 0)) * sizeof(TT), stages = (size_t)2 * C * NC * 2 + (size_t)R * (NC + 4) * 4;
  return (tile > stages ? tile : stages) + (VEC ? 2 * (size_t)R * 4 + 2 * (size_t)C * 4 : 0);
}
// TP: the product's type in memory - f32, or bf16 (ESMFold2's: the contraction writes half the bytes and
// this reads half; the tile is bf16 either way)
template <int C, int WARPS, class TT = float, int NC = 32, class TP = float, bool BIAS = false, class PT = float,
          int MINB = (C <= 128 && WARPS == 4 && NC == 16) ? 4 : (C == 256 && WARPS == 4 && NC == 16) ? 3 : (C <= 128 && WARPS == 4) ? 2 : 1>
#if LF_REG_STAGES
__global__ void __launch_bounds__(WARPS * 32, (C <= 128 && WARPS == 4) ? 2 : 1)
#else
__global__ void __launch_bounds__(WARPS * 32, MINB)
#endif
triangleOutK(const TP* __restrict__ prod, const float* __restrict__ cnScale,
    const float* __restrict__ cnOffset, const half* __restrict__ Woutt, const half* __restrict__ t2,
    float* __restrict__ pair, int L, int Lp, const float* __restrict__ ob = nullptr, RectMap rm = {}) {   // ob: AF2's output bias
  // Woutt: Wout's NC-column tiles (tileColumns), the stages unpadded and swizzled (stageSw)
  // VEC (a 16-bit product into a 16-bit tile - ESMFold2's and protenix2's bf16): the block's rows are the
  // PADDED pair space, so eight consecutive rows are 16 contiguous, aligned bytes of a channel and the tile
  // fills by cp.async - where an unpadded row's product was two bytes a thread (Nsight Compute: long
  // scoreboard 41% of the kernel's stalls, on those loads); a padding row computes and stores nothing
  constexpr bool VEC = sizeof(TT) == 2 && sizeof(TP) == 2;
  static_assert(!VEC || std::is_same_v<TT, TP>, "the tile takes the product's bytes as they are");
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDP = R + (VEC ? 8 : 0), KS = C / 16;
  static_assert(VEC || (R & 31) == 0, "the unpadded tile's XOR swizzle permutes 32-row banks");
  constexpr size_t STAGE = (size_t)C * NC * 2;
  constexpr size_t PS = triangleOutSmem<C, WARPS, TT, NC, TP>() - (VEC ? 2 * (size_t)R * 4 + 2 * (size_t)C * 4 : 0);
  extern __shared__ __align__(16) unsigned char smem[];
  TT* Ps = (TT*)smem;                                         // [C][LDP], then the weight stages
  float* stat = VEC ? (float*)(smem + PS) : nullptr;          // [R] mean, [R] 1/sd (VEC; !VEC keeps them in registers per warp)
  // the centre norm's scale and offset, staged once for VEC (when !VEC, Ps is 32.0 KB at C=128 so two blocks fit a 64 KB T4 SM)
  float* nsc = VEC ? stat + 2 * R : nullptr;                  // [C] scale, [C] offset
  if constexpr (VEC) {
    for (int c = threadIdx.x; c < C; c += NTH) {
      float sc_v = ptx_ldg_f32(cnScale + c), of_v = ptx_ldg_f32(cnOffset + c);
      ptx_st_shared_f32(nsc + c, sc_v);
      ptx_st_shared_f32(nsc + C + c, of_v);
    }
  }
  auto pIdx = [&](int c, int r) { return c * LDP + (VEC ? r : (r ^ (c & 31))); };
  auto Ws = [&](int s) { return (half*)(smem + s * STAGE); };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const size_t P = (size_t)L * L, plane = rm.J ? rm.size : (size_t)Lp * Lp, stride = rm.J && rm.stride ? rm.stride : plane;
  const unsigned Jr = rm.J ? (unsigned)rm.J : (unsigned)Lp;
  size_t row0 = (size_t)blockIdx.x * R;
  auto padded = [&](size_t r) { unsigned u = (unsigned)r, i = u / (unsigned)L; return (size_t)i * Lp + (u - i * L); };
  // a row's pair (the unpadded index) and its padded index, in whichever space the rows are
  auto pairAt = [&](size_t row) -> size_t {
    if constexpr (VEC) {
      if (row >= plane) return SIZE_MAX;
      unsigned u = (unsigned)row, iq = u / Jr, i = rm.i0 + iq, j = rm.j0 + (u - iq * Jr);
      return i < (unsigned)L && j < (unsigned)L ? (size_t)i * L + j : SIZE_MAX;
    } else return row < P ? row : SIZE_MAX;
  };
  auto paddedAt = [&](size_t row) -> size_t { if constexpr (VEC) return row; else return padded(row); };
  if constexpr (VEC) {
    const int c0_p = (int)threadIdx.x / (R / 8), r0_p = ((int)threadIdx.x % (R / 8)) * 8;
    const size_t row = row0 + r0_p;
    const bool ok = row < plane;
    const TP* srcBase = prod + (size_t)c0_p * stride + (ok ? row : 0);
    TT* dstBase = Ps + c0_p * LDP + r0_p;
#pragma unroll
    for (int it = 0; it < (C * (R / 8)) / NTH; ++it) {
      constexpr int C_STEP = NTH / (R / 8);
      cpAsync16(dstBase + it * C_STEP * LDP, srcBase + (size_t)it * C_STEP * stride, ok);
    }
    cpCommit(); cpWait<0>();
  } else {
    static_assert(NTH % R == 0, "a thread keeps one pair of the tile");
    int r = threadIdx.x % R;   // a thread keeps one pair (its padded index computed once) and walks the channels
    size_t row = row0 + r;
    bool live = row < P;
    const TP* src = prod + (live ? padded(row) : 0);
#pragma unroll 32
    for (int c = threadIdx.x / R; c < C; c += NTH / R) Ps[pIdx(c, r)] = fromF<TT>(live ? toF(src[(size_t)c * stride]) : 0.f);
  }
  __syncthreads();
  // the centre norm's mean and 1/sd, a warp a row; at VEC's LDP (R + 8) a warp's reads across one row
  // conflict four ways, so it takes two rows a word (a bf16 pair) - the same sums in the same order, so
  // byte-identical (a thread pair a row reads conflict-free and sums in another order: 0.7% more of the
  // trunk, and a different fold). ESMFold2 trunk 3462 -> 3415 ms at 1,044 tokens, 223.6 -> 220.5 at 261
  auto rowStats2 = [&](int r, float (&va)[C / 32], float (&vb)[C / 32]) {
    float sa = 0.f, sb = 0.f;
#pragma unroll
    for (int k = 0; k < C / 32; ++k) { sa = ptx_add_f32(sa, va[k]); sb = ptx_add_f32(sb, vb[k]); }
    ptx_warp_sum2_f32(sa, sb);
    float ma_ = ptx_div_f32(sa, (float)C), mb_ = ptx_div_f32(sb, (float)C), qa = 0.f, qb = 0.f;
#pragma unroll
    for (int k = 0; k < C / 32; ++k) {
      float da = ptx_sub_f32(va[k], ma_), db = ptx_sub_f32(vb[k], mb_);
      qa = ptx_fma_f32(da, da, qa); qb = ptx_fma_f32(db, db, qb);
    }
    ptx_warp_sum2_f32(qa, qb);
    if (lane == 0) {
      float ia_ = ptx_rsqrt_f32(ptx_add_f32(ptx_div_f32(qa, (float)C), 1e-5f));
      float ib_ = ptx_rsqrt_f32(ptx_add_f32(ptx_div_f32(qb, (float)C), 1e-5f));
      ptx_st_shared_v2_f32(stat + r, ma_, mb_);
      ptx_st_shared_v2_f32(stat + R + r, ia_, ib_);
    }
  };
  int ra = warp * 16 + g, rb = ra + 8;
  float ma = 0.f, ia = 0.f, mb = 0.f, ib = 0.f;
  if constexpr (VEC) {
#pragma unroll
    for (int r = 2 * warp; r < R; r += 2 * WARPS) {
      float va[C / 32], vb[C / 32];
#pragma unroll
      for (int k = 0; k < C / 32; ++k) {
        uint32_t w = ptx_ld_shared_u32(Ps + (lane + 32 * k) * LDP + r);
        if constexpr (std::is_same_v<TT, __nv_bfloat16>) ptx_bf16x2_to_f32x2(w, va[k], vb[k]);
        else ptx_f16x2_to_f32x2(w, va[k], vb[k]);
      }
      rowStats2(r, va, vb);
    }
    __syncthreads();
    ma = ptx_ld_shared_f32(stat + ra); ia = ptx_ld_shared_f32(stat + R + ra);
    mb = ptx_ld_shared_f32(stat + rb); ib = ptx_ld_shared_f32(stat + R + rb);
  } else {
    for (int m = 0; m < 16; ++m) {
      int r = warp * 16 + m;
      float s = 0.f, v[C / 32];
#pragma unroll
      for (int k = 0; k < C / 32; ++k) { int c = lane + 32 * k; v[k] = toF(Ps[pIdx(c, r)]); s = ptx_add_f32(s, v[k]); }
      s = ptx_warp_sum_f32(s);
      float mean = ptx_div_f32(s, (float)C), q = 0.f;
#pragma unroll
      for (int k = 0; k < C / 32; ++k) { float d = ptx_sub_f32(v[k], mean); q = ptx_fma_f32(d, d, q); }
      q = ptx_warp_sum_f32(q);
      float inv = ptx_rsqrt_f32(ptx_add_f32(ptx_div_f32(q, (float)C), 1e-5f));
      if (g == m) { ma = mean; ia = inv; }
      if (g + 8 == m) { mb = mean; ib = inv; }
    }
  }
  auto nrmVal = [&](int r, float m, float inv, int k, float sc, float of) {
    return ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(toF(Ps[pIdx(k, r)]), m), inv), sc, of);
  };
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks0 = 0; ks0 < KS; ks0 += 8) {
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      int ks = ks0 + d, k = ks * 16 + 2 * tig;
      float2 sc01 = VEC ? ptx_ld_shared_v2_f32(nsc + k) : ptx_ldg_v2_f32(cnScale + k);
      float2 of01 = VEC ? ptx_ld_shared_v2_f32(nsc + C + k) : ptx_ldg_v2_f32(cnOffset + k);
      float2 sc89 = VEC ? ptx_ld_shared_v2_f32(nsc + k + 8) : ptx_ldg_v2_f32(cnScale + k + 8);
      float2 of89 = VEC ? ptx_ld_shared_v2_f32(nsc + C + k + 8) : ptx_ldg_v2_f32(cnOffset + k + 8);
      xa[ks][0] = pack2(nrmVal(ra, ma, ia, k, sc01.x, of01.x), nrmVal(ra, ma, ia, k + 1, sc01.y, of01.y));
      xa[ks][1] = pack2(nrmVal(rb, mb, ib, k, sc01.x, of01.x), nrmVal(rb, mb, ib, k + 1, sc01.y, of01.y));
      xa[ks][2] = pack2(nrmVal(ra, ma, ia, k + 8, sc89.x, of89.x), nrmVal(ra, ma, ia, k + 9, sc89.y, of89.y));
      xa[ks][3] = pack2(nrmVal(rb, mb, ib, k + 8, sc89.x, of89.x), nrmVal(rb, mb, ib, k + 9, sc89.y, of89.y));
    }
    if constexpr (C == 512) __syncwarp();
  }
  __syncthreads();                                            // Ps is dead from here: the weight stages take it
  constexpr int ITER = (C * (NC / 8) + NTH - 1) / NTH;
  const int k0 = (int)threadIdx.x / (NC / 8), c0_ = ((int)threadIdx.x % (NC / 8)) * 8, sw0 = stageSw<NC>(k0, c0_);
  auto stage = [&](int n, int st, auto&& op) {
    half* w = Ws(st) + sw0;
    const half* s = Woutt + ((size_t)n * C + k0) * NC + c0_;
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
      if ((C * (NC / 8)) % NTH != 0 && it * NTH + (int)threadIdx.x >= C * (NC / 8)) break;
      op(it, w + it * (NTH / (NC / 8)) * NC, s + (size_t)it * (NTH / (NC / 8)) * NC);
    }
  };
  auto issue = [&](int n, int st) { stage(n, st, [](int, half* d, const half* s) { cpAsync16(d, s, true); }); cpCommit(); };
#if LF_REG_STAGES
  RegStage<ITER> next;                // (sm_75: the next chunk's weights held in registers across this chunk)
#endif
  issue(0, 0);
  // the output chunk, staged so the gate and the residual go out 16 bytes a thread, a row's 32 columns
  // contiguous (a fragment's own stores are 8 bytes a row, eight rows a warp)
  constexpr int LDO = NC + 4, LPR = 32 / (NC / 4), PER = 16 / LPR;
  static_assert(R * (NC / 4) % NTH == 0, "whole items a thread");
  float* Os = (float*)(smem + 2 * STAGE) + warp * 16 * LDO;
  const int q0_ = (lane % (NC / 4)) * 4;
  unsigned pr_[C < 512 ? PER : 1], pad_[C < 512 ? PER : 1];
  if constexpr (C < 512) {
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int r = warp * 16 + i * LPR + lane / (NC / 4);
      size_t row = row0 + r;
      size_t p = pairAt(row);
      pr_[i] = p != SIZE_MAX ? (unsigned)p : ~0u;
      pad_[i] = p != SIZE_MAX ? (unsigned)paddedAt(row) : 0u;
    }
  }
  constexpr int chunks = C / NC;
  for (int n = 0; n < chunks; ++n) {
    int st = n & 1;
#if LF_REG_STAGES
    if (n + 1 < chunks) stage(n + 1, st ^ 1, [&](int i, half*, const half* src) { next.load(i, src); });
#endif
    // the chunk's residual and gate, loaded before its MMAs so they cover the latency (ncu: long scoreboard
    // 39% of the kernel's stalls, on these loads issued after the GEMM)
    PairRaw4<PT> pv[C < 512 ? PER : 1]; uint2 gv[C < 512 ? PER : 1];
    float4 b4 = {0.f, 0.f, 0.f, 0.f};
    if constexpr (BIAS) b4 = ptx_ldg_v4_f32(ob + n * NC + q0_);
    if constexpr (C < 512) {
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        if (pr_[i] != ~0u) {
          gv[i] = ptx_ld_global_v2_u32(t2 + (size_t)pad_[i] * C + n * NC + q0_);
          pv[i] = pairLdRaw4<PT>(pair, (size_t)pr_[i] * C + n * NC + q0_);
        }
      }
    }
#if LF_REG_STAGES
    __syncthreads();
#else
    cpWait<0>();
    __syncthreads();
#endif
    const half* w = Ws(st);
    float acc[NC / 8][4] = {};
    constexpr int N2 = NC / 16, STEPS = KS * N2;
    const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
    if constexpr (C == 512 && N2 == 1) {
      uint32_t f0, f1, f2, f3;
      uint32_t fb[4];
      ldsm4t(fb, w + stageSw<NC>(kl, cl));
#if !LF_REG_STAGES
      if (n + 1 < chunks) issue(n + 1, st ^ 1);
#endif
      f0 = fb[0]; f1 = fb[1]; f2 = fb[2]; f3 = fb[3];
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        mma16816(acc[0], xa[ks], f0, f1);
        if (ks + 1 < KS) {
          ldsm4t(fb, w + stageSw<NC>((ks + 1) * 16 + kl, cl));
          f0 = fb[0]; f1 = fb[1];
        }
        mma16816(acc[1], xa[ks], f2, f3);
        f2 = fb[2]; f3 = fb[3];
      }
    } else {
      uint32_t fb[2][4];
      ldsm4t(fb[0], w + stageSw<NC>(kl, cl));
#if !LF_REG_STAGES
      if (n + 1 < chunks) issue(n + 1, st ^ 1);
#endif
#pragma unroll
      for (int s = 0; s < STEPS; ++s) {
        const int cur = s & 1, ks = s / N2, n2 = s % N2;
        if (s + 1 < STEPS) {
          const int nks = (s + 1) / N2, nn2 = (s + 1) % N2;
          ldsm4t(fb[cur ^ 1], w + stageSw<NC>(nks * 16 + kl, nn2 * 16 + cl));
        }
        mma16816(acc[2 * n2], xa[ks], fb[cur][0], fb[cur][1]);
        mma16816(acc[2 * n2 + 1], xa[ks], fb[cur][2], fb[cur][3]);
      }
    }
#pragma unroll
    for (int nt = 0; nt < NC / 8; ++nt) {
      int c = nt * 8 + tig * 2;
      ptx_st_shared_v2_f32(Os + g * LDO + c, acc[nt][0], acc[nt][1]);
      ptx_st_shared_v2_f32(Os + (g + 8) * LDO + c, acc[nt][2], acc[nt][3]);
    }
    __syncwarp();
    float4 oAll[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int rl = i * LPR + lane / (NC / 4);
      oAll[i] = ptx_ld_shared_v4_f32(Os + rl * LDO + q0_);
    }
    int c = n * NC + q0_;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      size_t pr;
      if constexpr (C < 512) {
        if (pr_[i] == ~0u) continue;
        pr = (size_t)pr_[i];
      } else {
        size_t row = row0 + warp * 16 + i * LPR + lane / (NC / 4);
        pr = pairAt(row);
        if (pr == SIZE_MAX) continue;
      }
      float4 o = oAll[i];
      uint2 gw = C < 512 ? gv[i] : ptx_ld_global_v2_u32(t2 + paddedAt(row0 + warp * 16 + i * LPR + lane / (NC / 4)) * C + c);
      float2 g01 = ptx_f16x2_to_f32x2(gw.x), g23 = ptx_f16x2_to_f32x2(gw.y);
      float4 v = C < 512 ? pairUnpack4<PT>(pv[i]) : pairLd4<PT>(pair, pr * C + c);
      if constexpr (BIAS) {
        o.x = ptx_add_f32(o.x, b4.x); o.y = ptx_add_f32(o.y, b4.y); o.z = ptx_add_f32(o.z, b4.z); o.w = ptx_add_f32(o.w, b4.w);
      }
      v.x = ptx_fma_f32(o.x, sigmH(g01.x), v.x); v.y = ptx_fma_f32(o.y, sigmH(g01.y), v.y);
      v.z = ptx_fma_f32(o.z, sigmH(g23.x), v.z); v.w = ptx_fma_f32(o.w, sigmH(g23.y), v.w);
      pairSt4<PT>(pair, pr * C + c, v);
    }
#if LF_REG_STAGES
    __syncwarp();
    if (n + 1 < chunks) stage(n + 1, st ^ 1, [&](int i, half* d, const half*) { next.store(i, d); });
#endif
  }
}

// ---------------------------------------------------------------- the triangle's input side
// cuda/af3's triInK, re-laid for 256 channels: at its 32-column steps the weight stages (82 KB) sit
// beside the LN'd rows (68 KB), one block an SM. Here the steps are 16 columns and the stages take the
// rows' memory once they are A fragments, so a block is the rows' 68 KB and two fit an SM.
// Output as triInK's: a, b channel-major padded planes (interleaved split, masked), t2 the gating
// linear's raw output over the padded rows.
// TA: a and b's type - f16, or bf16 so the contraction can write a bf16 product (an f16 one overflows)

// its dynamic shared memory: the LN'd rows, or the two weight stages and the a/b/t2 staging after them;
// in ROUNDS (xrounds > 1), one round's 16 * warps / xrounds rows share smem + 2 * STAGE with Ta/Tb/Ys
// (Xs is consumed during the prologue's round loop before the main step loop ever writes Ta/Tb/Ys),
// keeping triIn256Smem(256, 16, 8) at 49,664 B (<= 64 KB on T4 sm_75, and 2 blocks/SM in 100 KB on sm_89/sm_120)
template <class TA = half> constexpr size_t triIn256Smem(int C, int warps, int xrounds = 1, int mt = 1) {
  size_t stage = (size_t)2 * C * 16 * 2;
  size_t rows = (size_t)16 * (warps * mt / xrounds) * (C + 8) * 2;
  size_t ab = (size_t)2 * 8 * (16 * warps * mt + 8) * sizeof(TA), t2s = (size_t)warps * 16 * 32 * 2;
  size_t tail = ab > t2s ? ab : t2s;
  if (mt > 1 && xrounds == mt) {
    size_t main = 2 * stage + tail;
    return rows > main ? rows : main;
  }
  size_t stages = 2 * stage + (xrounds > 1 && rows > tail ? rows : tail);
  return xrounds == 1 && rows > stages ? rows : stages;
}
// Wt: tileTriIn's layout at 16 columns a tile
// XROUNDS > 1: the block's rows LayerNorm'd a round at a time into a buffer of their own, each round's warps
// taking their fragments from it before the next - so a block of 16 warps needs one round's rows, not all of
// them, beside its stages. On a T4 (64 KB an SM) the 4-warp form held a whole SM for 4 warps; this holds it
// for 16. Every row is normed exactly as before (lnRowsToShared's per-row arithmetic), so byte-identical.
// BIAS: AlphaFold 2's projections carry biases (triInK's layout: [4C, projection | gate in Wpg's order][C, the
// gating linear's]); without it the kernel is the one AF3's lineage and ESMFold2 run
// MT: a warp's row tiles of 16 (its LN'd rows held as MT tiles of A fragments): each weight fragment read from
// shared memory then feeds MT MMAs where it fed one - the kernel was shared-memory bound (Nsight Compute at 988
// tokens: the LSU's shared wavefronts 68% of peak, the tensor pipe 57%). Each output's k order is unchanged.
template <int C, int WARPS, class TA = half, int XROUNDS = 1, bool BIAS = false, class PT = float, int MT = 1,
          int MINB = (((C == 128 || C == 64) && WARPS == 4 && (MT == 2 || MT == 1)) || (C == 256 && WARPS == 4 && XROUNDS == 1 && MT == 1)) ? 4
                   : (C == 256 && WARPS == 8 && XROUNDS == 1 && MT == 1) ? 2 : 1>
#if LF_REG_STAGES
__global__ void __launch_bounds__(WARPS * 32, fitBlocks((C == 128 && WARPS == 4 && MT == 2) ? 2 : 1, triIn256Smem<TA>(C, WARPS, XROUNDS, MT)))
#else
__global__ void __launch_bounds__(WARPS * 32, fitBlocks(MINB, triIn256Smem<TA>(C, WARPS, XROUNDS, MT)))
#endif
triIn256K(const float* __restrict__ pair, const float* __restrict__ mask,
    const float* __restrict__ lnScale, const float* __restrict__ lnOffset, const half* __restrict__ Wt,
    TA* __restrict__ a, TA* __restrict__ b, half* __restrict__ t2, int n, int np, size_t cs,
    const float* __restrict__ bias = nullptr, RectMap rm = {}) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  if constexpr (sizeof(TA) == 1) return;
#endif
  // [rm.ab0, rm.ab1): the a/b steps (8 channels each) this launch makes - a and b written at their channels' places, so
  // a caller holding a few channels hands planes based that many channels back (sharded.cuh's rounds); -1, all of them
  // mask: null for every pair real (ESMFold2's sharded trunk: a whole pair of ones is 1.1 GB a rank at 16,549 tokens)
  // rm: a rectangle's rows (RectMap), its planes `size` long; a null a, b or t2 is not written (the blocks' passes each
  // want one operand)
  // the weight stage is unpadded, [k][NC], its two 16-byte halves swapped on rows with bit 2 of k set
  // (sw): the padded 48-byte rows kept ldmatrix conflict-free but serialised the cp.async writes ~6.7x
  // (ncu: 40% of the kernel's shared wavefronts excessive, all of them those four LDGSTS); swizzled,
  // both are conflict-free - 87.9 -> 84.7 ms of ESMFold2's 5CAJ trunk, byte-identical
  static_assert(MT == 1 || XROUNDS == 1 || XROUNDS == MT, "row tiles a warp, or rounds");
  constexpr int NC = 16, CH = NC / 2, R = 16 * WARPS * MT, NTH = 32 * WARPS, LDX = C + 8, LDW = NC, KS = C / 16, LDT = R + 8;
  auto sw = [](int k, int c) { return stageSw<NC>(k, c); };
  constexpr size_t STAGE = (size_t)2 * C * LDW * 2;
  // (the stages and the a/b staging take the rows' memory once the rows are fragments: the launch gives the
  // larger of the two - at 8 warps the rows, at 4 the stages, triIn256Smem)
  const unsigned pp = rm.J ? (unsigned)rm.size : (unsigned)np * (unsigned)np;
  if (rm.J) cs = rm.size;
  const unsigned Jr = rm.J ? (unsigned)rm.J : (unsigned)np;
  auto pairOf = [&](unsigned q) -> size_t {
    if (q >= pp) return SIZE_MAX;
    unsigned u = q, iq = u / Jr, i = rm.i0 + iq, j = rm.j0 + (u - iq * Jr);
    if (rm.T) { unsigned x = i; i = j; j = x; }
    return i < (unsigned)n && j < (unsigned)n ? (size_t)i * n + j : SIZE_MAX;
  };
  // (a t2 row's place: its own pair's padded row - transposed under rm.T)
  auto t2Row = [&](unsigned q) -> size_t {
    if (!rm.T) return q;
    unsigned u = q, iq = u / Jr; return (size_t)(u - iq * Jr) * (pp / Jr) + iq;
  };
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (XROUNDS == 1 || (MT > 1 && XROUNDS == MT)) ? (half*)smem : (half*)(smem + 2 * STAGE);
  auto W0 = [&](int s) { return (half*)(smem + s * STAGE); };
  auto W1 = [&](int s) { return W0(s) + C * LDW; };
  TA* Ta = (TA*)(smem + 2 * STAGE);
  TA* Tb = Ta + CH * LDT;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const unsigned row0 = blockIdx.x * R;
  // the gating linear's steps take TWO of its tiles, one a stage (w1 idles there otherwise): four accumulator
  // chains a step where they had two, and half the steps (ncu had the kernel waiting on its MMA chains)
  const int abSteps = C / CH;
  const int ab0 = rm.ab0, nAb = (rm.ab1 < 0 ? abSteps : rm.ab1) - rm.ab0;
  const int steps = nAb + (t2 ? C / NC / 2 : 0);                          // (no t2: the gating linear skipped)
  auto stepOf = [&](int i) { return i < nAb ? ab0 + i : abSteps + (i - nAb); };   // (the weight tiles' step)
  constexpr int ITER = (C * (NC / 8) + NTH - 1) / NTH;
  const int k0 = (int)threadIdx.x / (NC / 8), c0_ = ((int)threadIdx.x % (NC / 8)) * 8, sw0 = sw(k0, c0_);
  // step j's two weight tiles, each 16-byte piece handed to op(index, its place in stage st, its source)
  auto stage = [&](int j, int st, auto&& op) {
    half *w0 = W0(st) + sw0, *w1 = W1(st) + sw0;
    const half* base = j < abSteps ? Wt + ((size_t)j * C + k0) * NC + c0_
                                   : Wt + ((size_t)(abSteps + 2 * (j - abSteps)) * C + k0) * NC + c0_ + (size_t)2 * C * C;
    const size_t w1Off = j < abSteps ? (size_t)2 * C * C : (size_t)C * NC;
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
      if ((C * (NC / 8)) % NTH != 0 && it * NTH + (int)threadIdx.x >= C * (NC / 8)) break;
      int off = it * (NTH / (NC / 8)) * NC;
      const half* src = base + off;
      op(2 * it, w0 + off, src);
      op(2 * it + 1, w1 + off, src + w1Off);
    }
  };
  auto issue = [&](int j, int st) { stage(j, st, [](int, half* d, const half* s) { cpAsync16(d, s, true); }); cpCommit(); };
#if LF_REG_STAGES
  RegStage<2 * ITER> next;            // (sm_75: the next step's tiles held in registers across this step's MMAs)
#endif
  uint32_t xa[MT][KS][4];
  if constexpr (XROUNDS == 1) {
    lnRowsToShared<C, R, WARPS, PT>(pair, [&](int r) { return pairOf(row0 + r); }, lnScale, lnOffset, Xs, LDX, warp, lane);
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) ldsm4(xa[mt][ks], Xs + ((warp * MT + mt) * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
    }
    __syncthreads();                    // every warp has its fragments: the stages take the rows' memory
    issue(stepOf(0), 0);
  } else if constexpr (MT > 1 && XROUNDS == MT) {
    constexpr int XR = 16 * WARPS;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      lnRowsToShared<C, XR, WARPS, PT>(pair, [&](int r) { return pairOf(row0 + ((r >> 4) * MT + mt) * 16 + (r & 15)); }, lnScale, lnOffset, Xs, LDX, warp, lane);
      __syncthreads();
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) ldsm4(xa[mt][ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
      __syncthreads();                  // the round's fragments are in registers: the next round or stage 0 takes Xs
    }
    issue(stepOf(0), 0);
  } else {
    static_assert(R % XROUNDS == 0 && (R / XROUNDS) % 16 == 0 && (R / XROUNDS) % WARPS == 0, "whole warps' rows a round");
    constexpr int XR = R / XROUNDS;
    issue(stepOf(0), 0);                // (the stages have memory of their own here)
#pragma unroll 1
    for (int rd = 0; rd < XROUNDS; ++rd) {
      lnRowsToShared<C, XR, WARPS, PT, false, 1>(pair, [&](int r) { return pairOf(row0 + rd * XR + r); }, lnScale, lnOffset, Xs, LDX, warp, lane);
      __syncthreads();
      if (warp * 16 / XR == rd) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) ldsm4(xa[0][ks], Xs + (warp * 16 - rd * XR + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
      }
      __syncthreads();                  // the round's warps have their fragments: the next round's rows go there
    }
  }
  const int lrBase = warp * (16 * MT) + g;
  const int ch0 = (int)threadIdx.x / (R / 8), r8 = ((int)threadIdx.x % (R / 8)) * 8;
  float m0[MT], m1[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    int lr0 = lrBase + mt * 16, lr1 = lr0 + 8;
    size_t pr0 = pairOf(row0 + lr0), pr1 = pairOf(row0 + lr1);
    m0[mt] = pr0 != SIZE_MAX ? (mask ? ptx_ldg_f32(mask + pr0) : 1.f) : 0.f;
    m1[mt] = pr1 != SIZE_MAX ? (mask ? ptx_ldg_f32(mask + pr1) : 1.f) : 0.f;
  }
  for (int it = 0; it < steps; ++it) {
    const int j = stepOf(it), st = it & 1;
    bool gating = j >= abSteps;
#if LF_REG_STAGES
    if (it + 1 < steps) stage(stepOf(it + 1), st ^ 1, [&](int i, half*, const half* s) { next.load(i, s); });
#endif
    float2 bp[NC / 8], bq[NC / 8];
    if constexpr (BIAS) {
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) {
        int col = gating ? 4 * C + 2 * (j - abSteps) * NC + nt * 8 + tig * 2 : j * NC + nt * 8 + tig * 2;
        int qcol = gating ? col + NC : 2 * C + col;          // the second gating tile, or the gate's columns
        bp[nt] = ptx_ldg_v2_f32(bias + col);
        bq[nt] = ptx_ldg_v2_f32(bias + qcol);
      }
    }
#if LF_REG_STAGES
    __syncthreads();
#else
    cpWait<0>();
    __syncthreads();
    if (it + 1 < steps) issue(stepOf(it + 1), st ^ 1);
#endif
    const half *w0 = W0(st), *w1 = W1(st);
    float p[MT][NC / 8][4] = {}, q[MT][NC / 8][4] = {};
    const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
    // each k step's weight fragment loaded ahead of the previous half-step's MMAs (8 regs instead of 16,
    // keeping 4-warp MT=2 at C=128 and 16-warp XROUNDS=8 at C=256 inside 128 registers); each sum's k order unchanged
    uint32_t f0[4], f1[4];
    ldsm4t(f0, w0 + sw(kl, cl));
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      ldsm4t(f1, w1 + sw(ks * 16 + kl, cl));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) { mma16816(p[mt][0], xa[mt][ks], f0[0], f0[1]); mma16816(p[mt][1], xa[mt][ks], f0[2], f0[3]); }
      if (ks + 1 < KS) ldsm4t(f0, w0 + sw((ks + 1) * 16 + kl, cl));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) { mma16816(q[mt][0], xa[mt][ks], f1[0], f1[1]); mma16816(q[mt][1], xa[mt][ks], f1[2], f1[3]); }
    }
    if constexpr (BIAS) {
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) {
        float b0 = bp[nt].x, b1 = bp[nt].y, g0 = bq[nt].x, g1 = bq[nt].y;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          p[mt][nt][0] = ptx_add_f32(p[mt][nt][0], b0); p[mt][nt][1] = ptx_add_f32(p[mt][nt][1], b1);
          p[mt][nt][2] = ptx_add_f32(p[mt][nt][2], b0); p[mt][nt][3] = ptx_add_f32(p[mt][nt][3], b1);
          q[mt][nt][0] = ptx_add_f32(q[mt][nt][0], g0); q[mt][nt][1] = ptx_add_f32(q[mt][nt][1], g1);
          q[mt][nt][2] = ptx_add_f32(q[mt][nt][2], g0); q[mt][nt][3] = ptx_add_f32(q[mt][nt][3], g1);
        }
      }
    }
    if (gating) {
      int c0 = 2 * (j - abSteps) * NC;                         // p the first tile's columns, q the second's
      if (!rm.T) {
        half* Ys = reinterpret_cast<half*>(smem + 2 * STAGE) + warp * 16 * 32;
        const int sw0 = (g >> 1) & 3;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
          for (int nt = 0; nt < NC / 8; ++nt) {
            int c8p = (nt ^ sw0) * 8 + tig * 2, c8q = ((nt + 2) ^ sw0) * 8 + tig * 2;
            ptx_st_shared_u32(Ys + g * 32 + c8p, ptx_cvt_f16x2_f32(p[mt][nt][0], p[mt][nt][1]));
            ptx_st_shared_u32(Ys + (g + 8) * 32 + c8p, ptx_cvt_f16x2_f32(p[mt][nt][2], p[mt][nt][3]));
            ptx_st_shared_u32(Ys + g * 32 + c8q, ptx_cvt_f16x2_f32(q[mt][nt][0], q[mt][nt][1]));
            ptx_st_shared_u32(Ys + (g + 8) * 32 + c8q, ptx_cvt_f16x2_f32(q[mt][nt][2], q[mt][nt][3]));
          }
          __syncwarp();
          const int c = tig * 8, csw = (tig ^ sw0) * 8;
          uint4 y0 = ptx_ld_shared_v4_u32(Ys + g * 32 + csw);
          uint4 y1 = ptx_ld_shared_v4_u32(Ys + (g + 8) * 32 + csw);
          unsigned qa0 = row0 + (warp * MT + mt) * 16 + g;
          if (qa0 < pp) ptx_st_global_v4_u32(t2 + (size_t)qa0 * C + c0 + c, y0);
          if (qa0 + 8 < pp) ptx_st_global_v4_u32(t2 + (size_t)(qa0 + 8) * C + c0 + c, y1);
          if (mt + 1 < MT) __syncwarp();
        }
      } else {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const int lr0 = lrBase + mt * 16, lr1 = lr0 + 8;
#pragma unroll
          for (int nt = 0; nt < NC / 8; ++nt) {
            int c = c0 + nt * 8 + tig * 2;
            unsigned qa = row0 + lr0, qb = row0 + lr1;
            if (!t2) continue;
            size_t ra = t2Row(qa), rb = t2Row(qb);
            if (qa < pp) ptx_st_global_u32(t2 + ra * C + c, ptx_cvt_f16x2_f32(p[mt][nt][0], p[mt][nt][1]));
            if (qb < pp) ptx_st_global_u32(t2 + rb * C + c, ptx_cvt_f16x2_f32(p[mt][nt][2], p[mt][nt][3]));
            if (qa < pp) ptx_st_global_u32(t2 + ra * C + c + NC, ptx_cvt_f16x2_f32(q[mt][nt][0], q[mt][nt][1]));
            if (qb < pp) ptx_st_global_u32(t2 + rb * C + c + NC, ptx_cvt_f16x2_f32(q[mt][nt][2], q[mt][nt][3]));
          }
        }
      }
    } else {
      // column 2ch is a's channel ch, 2ch+1 b's: the thread's columns nt*8 + 2 tig are channel nt*4 + tig
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int lr0 = lrBase + mt * 16, lr1 = lr0 + 8;
#pragma unroll
        for (int nt = 0; nt < NC / 8; ++nt) {
          int ch = nt * 4 + tig;
          Ta[ch * LDT + lr0] = fromF<TA>(ptx_mul_f32(ptx_mul_f32(p[mt][nt][0], sigmH(q[mt][nt][0])), m0[mt]));
          Tb[ch * LDT + lr0] = fromF<TA>(ptx_mul_f32(ptx_mul_f32(p[mt][nt][1], sigmH(q[mt][nt][1])), m0[mt]));
          Ta[ch * LDT + lr1] = fromF<TA>(ptx_mul_f32(ptx_mul_f32(p[mt][nt][2], sigmH(q[mt][nt][2])), m1[mt]));
          Tb[ch * LDT + lr1] = fromF<TA>(ptx_mul_f32(ptx_mul_f32(p[mt][nt][3], sigmH(q[mt][nt][3])), m1[mt]));
        }
      }
      __syncthreads();
#pragma unroll
      for (int it_ = 0; it_ < (CH * (R / 8) + NTH - 1) / NTH; ++it_) {
        if ((CH * (R / 8)) % NTH != 0 && it_ * NTH + (int)threadIdx.x >= CH * (R / 8)) break;
        int ch = ch0 + it_ * (NTH / (R / 8));
        unsigned row = row0 + r8;
        if (row < cs) {
          size_t at = (size_t)(j * CH + ch) * cs + row;
          if constexpr (sizeof(TA) == 1) {
            uint2 va = a ? ptx_ld_shared_v2_u32(Ta + ch * LDT + r8) : make_uint2(0, 0);
            uint2 vb = b ? ptx_ld_shared_v2_u32(Tb + ch * LDT + r8) : make_uint2(0, 0);
            if (a) ptx_st_global_v2_u32(a + at, va);
            if (b) ptx_st_global_v2_u32(b + at, vb);
          } else {
            uint4 va = a ? ptx_ld_shared_v4_u32(Ta + ch * LDT + r8) : make_uint4(0, 0, 0, 0);
            uint4 vb = b ? ptx_ld_shared_v4_u32(Tb + ch * LDT + r8) : make_uint4(0, 0, 0, 0);
            if (a) ptx_st_global_v4_u32(a + at, va);
            if (b) ptx_st_global_v4_u32(b + at, vb);
          }
        }
      }
    }
#if LF_REG_STAGES
    if (it + 1 < steps) stage(stepOf(it + 1), st ^ 1, [&](int i, half* d, const half*) { next.store(i, d); });   // (stage st^1 was
    __syncthreads();
#endif                                                                         // last read before this step's barrier)
  }
}

inline const half* triangleOutTiles(const half* Wout, int C, int NC) {
  static std::map<std::pair<const half*, int>, half*> tiles;
  static std::vector<half*> chunks; static size_t used = 0, have = 0;
  static bool hooked = false;
  if (!hooked) {
    FORGET_HOOKS.push_back([] { for (half* p : chunks) CK(cudaFreeAsync(p, STREAM)); chunks.clear(); tiles.clear(); used = have = 0; });
    hooked = true;
  }
  auto it = tiles.find({Wout, NC});
  if (it != tiles.end()) return it->second;
  size_t n = ((size_t)C * C + 127) / 128 * 128;
  if (used + n > have) {
    have = std::max<size_t>(n, (size_t)8 << 20);
    chunks.push_back(poolAllocT<half>(have)); used = 0;
  }
  half* out = chunks.back() + used; used += n;
  tileColumns(Wout, C, C, 0, C, NC, out);
  return tiles[{Wout, NC}] = out;
}

// the triangle's output side, launched: the bf16 tile at 16-column stages (LOCALFOLD_TRIOUT_F32=1: the float tile)
template <int C, int WARPS, class TP = float>
void triangleOutRun(const TP* prod, const float* sc, const float* of, const half* Wout, const half* t2, float* pair,
                    int L, int Lp, const float* ob = nullptr, RectMap rm = {}, const half* wtIn = nullptr) {
  static const bool f32 = getenv("LOCALFOLD_TRIOUT_F32") != nullptr;
  if (rm.J && (sizeof(TP) != 2 || f32)) { fprintf(stderr, "triangleOutRun: a rectangle wants the bf16 product and tile\n"); exit(1); }
  constexpr int R = 16 * WARPS;
  size_t P = (size_t)L * L;
  const half* wt = wtIn ? wtIn : triangleOutTiles(Wout, C, f32 || (ob && sizeof(TP) == 4) ? 32 : 16);
  if constexpr (C <= 256) {
    if (f32 && !ob) {
      constexpr size_t smem = triangleOutSmem<C, WARPS, float, 32>();
      WITH_PAIR_T(
        static bool attr = false;
        if (!attr) { smemAttr((triangleOutK<C, WARPS, float, 32, TP, false, PT>), (int)smem); attr = true; }
        triangleOutK<C, WARPS, float, 32, TP, false, PT><<<(unsigned)((P + R - 1) / R), 32 * WARPS, smem, STREAM>>>(prod, sc, of, wt, t2, pair, L, Lp));
      return;
    }
  }
  if constexpr (C <= 128) {
    if (ob) {
      // AF2's output bias - and its f32 product kept f32 in the tile, as AF2's own kernels keep it (the bf16 tile
      // read the triangle's update 2e-3 off theirs; a float tile at 128 channels is ~34 KB, inside a T4)
      if constexpr (sizeof(TP) == 4) {
        constexpr size_t smem = triangleOutSmem<C, WARPS, float, 32, TP>();
        WITH_PAIR_T(
          static bool attr = false;
          if (!attr) { smemAttr((triangleOutK<C, WARPS, float, 32, TP, true, PT>), (int)smem); attr = true; }
          triangleOutK<C, WARPS, float, 32, TP, true, PT><<<(unsigned)((P + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
            prod, sc, of, wt, t2, pair, L, Lp, ob));
      } else {
        // ...or a bf16 product (AF2's streaming triangle on an A100, as AF3's: its narrow path's product is bf16 too) - the
        // bf16 tile at 16-column stages, the bias in the same epilogue
        constexpr size_t smem = triangleOutSmem<C, WARPS, __nv_bfloat16, 16, TP>();
        constexpr bool vec = sizeof(TP) == 2;
        size_t rows = rm.J ? rm.size : vec ? (size_t)Lp * Lp : P;
        WITH_PAIR_T(
          static bool attr = false;
          if (!attr) { smemAttr((triangleOutK<C, WARPS, __nv_bfloat16, 16, TP, true, PT>), (int)smem); attr = true; }
          triangleOutK<C, WARPS, __nv_bfloat16, 16, TP, true, PT><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
            prod, sc, of, wt, t2, pair, L, Lp, ob, rm));
      }
      return;
    }
  }
  {
    constexpr size_t smem = triangleOutSmem<C, WARPS, __nv_bfloat16, 16, TP>();
    constexpr bool vec = sizeof(TP) == 2;                       // the padded rows (triangleOutK's VEC)
    size_t rows = rm.J ? rm.size : vec ? (size_t)Lp * Lp : P;
    WITH_PAIR_T(
      static bool attr = false;
      if (!attr) { smemAttr((triangleOutK<C, WARPS, __nv_bfloat16, 16, TP, false, PT>), (int)smem); attr = true; }
      triangleOutK<C, WARPS, __nv_bfloat16, 16, TP, false, PT><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(prod, sc, of, wt, t2, pair, L, Lp, nullptr, rm));
  }
}

