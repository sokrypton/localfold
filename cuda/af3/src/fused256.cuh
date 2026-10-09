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
__global__ void __launch_bounds__(WARPS * 32, MINB) transitionUpK(const float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ W1t, TG* __restrict__ gated, size_t rows, int I) {
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, KS = C / 16;
  constexpr size_t STAGE = (size_t)2 * C * NC * 2;
  auto sw = [](int k, int c) { return stageSw<NC>(k, c); };
  extern __shared__ __align__(16) unsigned char smem[];
  // the LN'd rows and the weight stages share memory: the rows go into registers (A fragments) before
  // the first stage is issued, so a block's footprint is the larger of the two, not their sum (in rounds, the
  // rows have a buffer of their own after the stages)
  half* Xs = XROUNDS == 1 ? (half*)smem : (half*)(smem + 2 * STAGE);
  unsigned char* stages = smem;
  auto Wa = [&](int s) { return (half*)(stages + s * STAGE); };
  auto Wb = [&](int s) { return Wa(s) + C * NC; };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  int chunks = I / NC;
  constexpr int ITER = (C * (NC / 8) + NTH - 1) / NTH;
  auto stage = [&](int j, int st, auto&& op) {
    half *a = Wa(st), *b = Wb(st);
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
      const int t = it * NTH + (int)threadIdx.x;
      if ((C * (NC / 8)) % NTH != 0 && t >= C * (NC / 8)) break;
      int k = t / (NC / 8), c = (t % (NC / 8)) * 8;
      const half* src = W1t + ((size_t)(j * NC / 32) * C + k) * 32 + (j * NC) % 32 + c;   // (32-column tiles)
      op(2 * it, a + sw(k, c), src);
      op(2 * it + 1, b + sw(k, c), src + (size_t)C * I);
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
#else
    if (j + 1 < chunks) { issue(j + 1, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
#endif
    __syncthreads();
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
      auto gate = [&](int e) { float v = ha[nt][e]; return v * sigmH(v) * hb[nt][e]; };   // rounded to f16 next
      int c = j * NC + nt * 8 + tig * 2;
      if constexpr (std::is_same_v<TG, half>) {
        if (r0 < rows) *reinterpret_cast<uint32_t*>(gated + r0 * I + c) = pack2(gate(0), gate(1));
        if (r1 < rows) *reinterpret_cast<uint32_t*>(gated + r1 * I + c) = pack2(gate(2), gate(3));
      } else {
        if (r0 < rows) *reinterpret_cast<__nv_bfloat162*>(gated + r0 * I + c) = __floats2bfloat162_rn(gate(0), gate(1));
        if (r1 < rows) *reinterpret_cast<__nv_bfloat162*>(gated + r1 * I + c) = __floats2bfloat162_rn(gate(2), gate(3));
      }
    }
#if LF_REG_STAGES
    if (j + 1 < chunks) stage(j + 1, st ^ 1, [&](int i, half* d, const half*) { next.store(i, d); });
#endif
    __syncthreads();
  }
}

// transitionUpK's weights as it streams them (see tileColumns): 2 C I halves, each half's
// 32-column tiles [C][32] contiguous
inline void tileTransitionUp(const half* W1, int C, int I, half* out) {
  tileColumns(W1, C, 2 * I, 0, I, 32, out);
  tileColumns(W1, C, 2 * I, I, I, 32, out + (size_t)C * I);
}
// transitionUpK's dynamic shared memory: the larger of the LN'd rows and the two weight stages
template <int C, int WARPS, int NC = 32, int XROUNDS = 1>
constexpr size_t transitionUpSmem() {
  if constexpr (XROUNDS > 1) return (size_t)2 * 2 * C * NC * 2 + (size_t)16 * WARPS / XROUNDS * (C + 8) * 2;
  else return std::max((size_t)16 * WARPS * (C + 8) * 2, (size_t)2 * 2 * C * NC * 2);
}

// The rows one full wave of transitionUpK<C, WARPS> covers - the blocks every multiprocessor holds at
// once, times the multiprocessors, times a block's 16 * WARPS rows - so a caller that chunks its rows can
// chunk in whole waves. A chunk of 32768 rows was 256 blocks on an A100 at two a multiprocessor, 1.19
// waves: two rounds with the second nearly empty, then a 25-block tail at 0.12 waves - five rounds of the
// device where three do (Nsight Compute, ESMFold2 at 262 tokens).
template <int C, int WARPS>
size_t transitionUpWaveRows(size_t smem) {
  static size_t rows = [smem] {
    smemAttr((transitionUpK<C, WARPS>), (int)smem);
    int perSm = 0, dev = 0, sms = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, transitionUpK<C, WARPS>, 32 * WARPS, smem));
    CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    return (size_t)std::max(1, perSm) * sms * 16 * WARPS;
  }();
  return rows;
}
// ...and the largest whole number of those waves inside a budget of rows (at least one wave)
template <int C, int WARPS>
size_t transitionUpChunkRows(size_t smem, size_t budgetRows) {
  size_t wave = transitionUpWaveRows<C, WARPS>(smem);
  return std::max<size_t>(1, budgetRows / wave) * wave;
}

// ---------------------------------------------------------------- the triangle's output side
// RectMap: the kernels' rows as a RECTANGLE of the padded pair space (triangleBlocked's blocks) - rows [i0, i0 + size / J),
// columns [j0, j0 + J), q = (i - i0) J + (j - j0) a row's own index, and every operand and output plane `size` long; J 0:
// the whole padded plane, as before
// T (the input kernel): its rows read the pair TRANSPOSED - row (u, v) takes pair (v, u) - so the
// incoming triangle's operands come out with the contraction's index contiguous, as the outgoing one's do (an FP8 GEMM
// takes only that layout); its t2 rows still land at their own pair's place, where the output kernel reads them
struct RectMap { int i0 = 0, J = 0, j0 = 0; size_t size = 0; bool T = false; };
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
  constexpr size_t tile = (size_t)C * (R + (VEC ? 8 : 1)) * sizeof(TT), stages = (size_t)2 * C * NC * 2 + (size_t)R * (NC + 4) * 4;
  return (tile > stages ? tile : stages) + 2 * (size_t)R * 4 + 2 * (size_t)C * 4;
}
// TP: the product's type in memory - f32, or bf16 (ESMFold2's: the contraction writes half the bytes and
// this reads half; the tile is bf16 either way)
template <int C, int WARPS, class TT = float, int NC = 32, class TP = float, bool BIAS = false, class PT = float>
__global__ void __launch_bounds__(WARPS * 32) triangleOutK(const TP* __restrict__ prod, const float* __restrict__ cnScale,
    const float* __restrict__ cnOffset, const half* __restrict__ Woutt, const half* __restrict__ t2,
    float* __restrict__ pair, int L, int Lp, const float* __restrict__ ob = nullptr, RectMap rm = {}) {   // ob: AF2's output bias
  // Woutt: Wout's NC-column tiles (tileColumns), the stages unpadded and swizzled (stageSw)
  // VEC (a 16-bit product into a 16-bit tile - ESMFold2's and protenix2's bf16): the block's rows are the
  // PADDED pair space, so eight consecutive rows are 16 contiguous, aligned bytes of a channel and the tile
  // fills by cp.async - where an unpadded row's product was two bytes a thread (Nsight Compute: long
  // scoreboard 41% of the kernel's stalls, on those loads); a padding row computes and stores nothing
  constexpr bool VEC = sizeof(TT) == 2 && sizeof(TP) == 2;
  static_assert(!VEC || std::is_same_v<TT, TP>, "the tile takes the product's bytes as they are");
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDP = R + (VEC ? 8 : 1), KS = C / 16;
  constexpr size_t STAGE = (size_t)C * NC * 2;
  constexpr size_t PS = triangleOutSmem<C, WARPS, TT, NC, TP>() - 2 * (size_t)R * 4 - 2 * (size_t)C * 4;
  extern __shared__ __align__(16) unsigned char smem[];
  TT* Ps = (TT*)smem;                                         // [C][LDP], then the weight stages
  float* stat = (float*)(smem + PS);                          // [R] mean, [R] 1/sd
  // the centre norm's scale and offset, staged once (they were 128 scalar loads a thread from global memory)
  float* nsc = stat + 2 * R;                                  // [C] scale, [C] offset
  for (int c = threadIdx.x; c < C; c += NTH) { nsc[c] = cnScale[c]; nsc[C + c] = cnOffset[c]; }
  auto Ws = [&](int s) { return (half*)(smem + s * STAGE); };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const size_t P = (size_t)L * L, plane = rm.J ? rm.size : (size_t)Lp * Lp;
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
#pragma unroll
    for (int t0_ = 0; t0_ < C * (R / 8); t0_ += NTH) {
      int t = t0_ + (int)threadIdx.x, c = t / (R / 8), r = (t % (R / 8)) * 8;
      size_t row = row0 + r;
      cpAsync16(Ps + c * LDP + r, prod + (size_t)c * plane + (row < plane ? row : 0), row < plane);
    }
    cpCommit(); cpWait<0>();
  } else {
    static_assert(NTH % R == 0, "a thread keeps one pair of the tile");
    int r = threadIdx.x % R;   // a thread keeps one pair (its padded index computed once) and walks the channels
    size_t row = row0 + r;
    bool live = row < P;
    const TP* src = prod + (live ? padded(row) : 0);
#pragma unroll 32
    for (int c = threadIdx.x / R; c < C; c += NTH / R) Ps[c * LDP + r] = TT(live ? (float)src[(size_t)c * plane] : 0.f);
  }
  __syncthreads();
  // the centre norm's mean and 1/sd, a warp a row; at VEC's LDP (R + 8) a warp's reads across one row
  // conflict four ways, so it takes two rows a word (a bf16 pair) - the same sums in the same order, so
  // byte-identical (a thread pair a row reads conflict-free and sums in another order: 0.7% more of the
  // trunk, and a different fold). ESMFold2 trunk 3462 -> 3415 ms at 1,044 tokens, 223.6 -> 220.5 at 261
  auto rowStats = [&](int r, float (&v)[C / 32]) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < C / 32; ++k) s += v[k];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / C, q = 0.f;
#pragma unroll
    for (int k = 0; k < C / 32; ++k) { float d = v[k] - mean; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
    if (lane == 0) { stat[r] = mean; stat[R + r] = rsqrtf(q / C + 1e-5f); }
  };
  if constexpr (VEC) {
    for (int r = 2 * warp; r < R; r += 2 * WARPS) {
      float va[C / 32], vb[C / 32];
#pragma unroll
      for (int k = 0; k < C / 32; ++k) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(Ps + (lane + 32 * k) * LDP + r);
        va[k] = (float)reinterpret_cast<const TT*>(&w)[0]; vb[k] = (float)reinterpret_cast<const TT*>(&w)[1];
      }
      rowStats(r, va); rowStats(r + 1, vb);
    }
  } else {
    for (int r = warp; r < R; r += WARPS) {
      float v[C / 32];
#pragma unroll
      for (int k = 0; k < C / 32; ++k) v[k] = (float)Ps[(lane + 32 * k) * LDP + r];
      rowStats(r, v);
    }
  }
  __syncthreads();
  // A fragments of m16n8k16: a0 (row g, k 2tig..+1), a1 (row g+8, ...), a2 (row g, k+8), a3 (row g+8, k+8)
  // (the gate computed here instead of read - sigmoid(LN(pair row) Wg) on a second set of A fragments, the input kernel
  // writing no t2 - was measured on the A100: the input kernel -1% (256 channels) / -17% (128), this one +31% / +41%,
  // the folds +4.7% / +1.4%; not taken)
  // (the centre norm's scale folded into Wout's rows and its offset into an output bias, the tile then read by
  // ldmatrix.trans - four 8x8 matrices an instruction for these 2-byte loads and the scale/offset reads - was measured:
  // -1% of this kernel at 988 tokens in AF3, +3.7% in ESMFold2's 256-channel one; not taken)
  int ra = warp * 16 + g, rb = ra + 8;
  float ma = stat[ra], ia = stat[R + ra], mb = stat[rb], ib = stat[R + rb];
  auto nrm = [&](int r, float m, float inv, int k) { return ((float)Ps[k * LDP + r] - m) * inv * nsc[k] + nsc[C + k]; };
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    int k = ks * 16 + 2 * tig;
    xa[ks][0] = pack2(nrm(ra, ma, ia, k), nrm(ra, ma, ia, k + 1));
    xa[ks][1] = pack2(nrm(rb, mb, ib, k), nrm(rb, mb, ib, k + 1));
    xa[ks][2] = pack2(nrm(ra, ma, ia, k + 8), nrm(ra, ma, ia, k + 9));
    xa[ks][3] = pack2(nrm(rb, mb, ib, k + 8), nrm(rb, mb, ib, k + 9));
  }
  __syncthreads();                                            // Ps is dead from here: the weight stages take it
  constexpr int ITER = (C * (NC / 8) + NTH - 1) / NTH;
  auto stage = [&](int n, int st, auto&& op) {
    half* w = Ws(st);
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
      const int t = it * NTH + (int)threadIdx.x;
      if ((C * (NC / 8)) % NTH != 0 && t >= C * (NC / 8)) break;
      int k = t / (NC / 8), c = (t % (NC / 8)) * 8;
      op(it, w + stageSw<NC>(k, c), Woutt + ((size_t)n * C + k) * NC + c);
    }
  };
  auto issue = [&](int n, int st) { stage(n, st, [](int, half* d, const half* s) { cpAsync16(d, s, true); }); cpCommit(); };
#if LF_REG_STAGES
  RegStage<ITER> next;                // (sm_75: the next chunk's weights held in registers across this chunk)
#endif
  issue(0, 0);
  // the output chunk, staged so the gate and the residual go out 16 bytes a thread, a row's 32 columns
  // contiguous (a fragment's own stores are 8 bytes a row, eight rows a warp)
  constexpr int LDO = NC + 4;
  float* Os = (float*)(smem + 2 * STAGE);
  constexpr int chunks = C / NC;
  for (int n = 0; n < chunks; ++n) {
    int st = n & 1;
#if LF_REG_STAGES
    if (n + 1 < chunks) stage(n + 1, st ^ 1, [&](int i, half*, const half* src) { next.load(i, src); });
#else
    if (n + 1 < chunks) { issue(n + 1, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
#endif
    // the chunk's residual and gate, loaded before its MMAs so they cover the latency (ncu: long scoreboard
    // 39% of the kernel's stalls, on these loads issued after the GEMM)
    constexpr int PER = R * (NC / 4) / NTH;
    static_assert(R * (NC / 4) % NTH == 0, "whole items a thread");
    float4 pv[PER]; uint2 gv[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int t = threadIdx.x + i * NTH, r = t / (NC / 4), q = (t % (NC / 4)) * 4;
      size_t row = row0 + r, pr = pairAt(row);
      if (pr != SIZE_MAX) {
        gv[i] = *reinterpret_cast<const uint2*>(t2 + paddedAt(row) * C + n * NC + q);
        pv[i] = pairLd4<PT>(pair, pr * C + n * NC + q);
      }
    }
    __syncthreads();
    const half* w = Ws(st);
    float acc[NC / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int n2 = 0; n2 < NC / 16; ++n2) {
        uint32_t fb[4];
        ldsm4t(fb, w + stageSw<NC>(ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), n2 * 16 + (lane >> 4) * 8));
        mma16816(acc[2 * n2], xa[ks], fb[0], fb[1]);
        mma16816(acc[2 * n2 + 1], xa[ks], fb[2], fb[3]);
      }
    }
#pragma unroll
    for (int nt = 0; nt < NC / 8; ++nt) {
      int c = nt * 8 + tig * 2;
      *reinterpret_cast<float2*>(Os + ra * LDO + c) = make_float2(acc[nt][0], acc[nt][1]);
      *reinterpret_cast<float2*>(Os + rb * LDO + c) = make_float2(acc[nt][2], acc[nt][3]);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int t = threadIdx.x + i * NTH, r = t / (NC / 4), q = (t % (NC / 4)) * 4;
      size_t row = row0 + r, pr = pairAt(row);
      if (pr == SIZE_MAX) continue;
      int c = n * NC + q;
      float4 o = *reinterpret_cast<const float4*>(Os + r * LDO + q);
      uint2 gw = gv[i];
      float2 g01 = __half22float2(*reinterpret_cast<half2*>(&gw.x)), g23 = __half22float2(*reinterpret_cast<half2*>(&gw.y));
      float4 v = pv[i];
      if constexpr (BIAS) { o.x += ob[c]; o.y += ob[c + 1]; o.z += ob[c + 2]; o.w += ob[c + 3]; }
      v.x += o.x * sigmH(g01.x); v.y += o.y * sigmH(g01.y); v.z += o.z * sigmH(g23.x); v.w += o.w * sigmH(g23.y);
      pairSt4<PT>(pair, pr * C + c, v);
    }
#if LF_REG_STAGES
    if (n + 1 < chunks) stage(n + 1, st ^ 1, [&](int i, half* d, const half*) { next.store(i, d); });
#endif
    __syncthreads();
  }
}

// ---------------------------------------------------------------- the triangle's input side
// cuda/af3's triInK, re-laid for 256 channels: at its 32-column steps the weight stages (82 KB) sit
// beside the LN'd rows (68 KB), one block an SM. Here the steps are 16 columns and the stages take the
// rows' memory once they are A fragments, so a block is the rows' 68 KB and two fit an SM.
// Output as triInK's: a, b channel-major padded planes (interleaved split, masked), t2 the gating
// linear's raw output over the padded rows.
// TA: a and b's type - f16, or bf16 so the contraction can write a bf16 product (an f16 one overflows)

// its dynamic shared memory: the LN'd rows, or the two weight stages and the a/b staging after them
// ...or, in ROUNDS (xrounds > 1), the stages, the staging and one round's 16 * warps / xrounds rows, none aliased
template <class TA = half> constexpr size_t triIn256Smem(int C, int warps, int xrounds = 1) {
  size_t rows = (size_t)16 * warps * (C + 8) * 2, stages = (size_t)2 * 2 * C * 16 * 2 + (size_t)2 * 8 * (16 * warps + 8) * sizeof(TA);
  if (xrounds > 1) return stages + (size_t)16 * warps / xrounds * (C + 8) * 2;
  return rows > stages ? rows : stages;
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
template <int C, int WARPS, class TA = half, int XROUNDS = 1, bool BIAS = false, class PT = float, int MT = 1>
__global__ void __launch_bounds__(WARPS * 32) triIn256K(const float* __restrict__ pair, const float* __restrict__ mask,
    const float* __restrict__ lnScale, const float* __restrict__ lnOffset, const half* __restrict__ Wt,
    TA* __restrict__ a, TA* __restrict__ b, half* __restrict__ t2, int n, int np, size_t cs,
    const float* __restrict__ bias = nullptr, RectMap rm = {}) {
  // rm: a rectangle's rows (RectMap), its planes `size` long; a null a, b or t2 is not written (the blocks' passes each
  // want one operand)
  // the weight stage is unpadded, [k][NC], its two 16-byte halves swapped on rows with bit 2 of k set
  // (sw): the padded 48-byte rows kept ldmatrix conflict-free but serialised the cp.async writes ~6.7x
  // (ncu: 40% of the kernel's shared wavefronts excessive, all of them those four LDGSTS); swizzled,
  // both are conflict-free - 87.9 -> 84.7 ms of ESMFold2's 5CAJ trunk, byte-identical
  static_assert(MT == 1 || XROUNDS == 1, "row tiles a warp, or rounds - not both");
  constexpr int NC = 16, CH = NC / 2, R = 16 * WARPS * MT, NTH = 32 * WARPS, LDX = C + 8, LDW = NC, KS = C / 16, LDT = R + 8;
  auto sw = [](int k, int c) { return stageSw<NC>(k, c); };
  constexpr size_t STAGE = (size_t)2 * C * LDW * 2;
  // (the stages and the a/b staging take the rows' memory once the rows are fragments: the launch gives the
  // larger of the two - at 8 warps the rows, at 4 the stages, triIn256Smem)
  const size_t pp = rm.J ? rm.size : (size_t)np * np;
  if (rm.J) cs = rm.size;
  const unsigned Jr = rm.J ? (unsigned)rm.J : (unsigned)np;
  auto pairOf = [&](size_t q) -> size_t {
    if (q >= pp) return SIZE_MAX;
    unsigned u = (unsigned)q, iq = u / Jr, i = rm.i0 + iq, j = rm.j0 + (u - iq * Jr);
    if (rm.T) { unsigned x = i; i = j; j = x; }
    return i < (unsigned)n && j < (unsigned)n ? (size_t)i * n + j : SIZE_MAX;
  };
  // (a t2 row's place: its own pair's padded row - transposed under rm.T)
  auto t2Row = [&](size_t q) -> size_t {
    if (!rm.T) return q;
    unsigned u = (unsigned)q, iq = u / Jr; return (size_t)(u - iq * Jr) * (pp / Jr) + iq;
  };
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = XROUNDS == 1 ? (half*)smem : (half*)(smem + 2 * STAGE + (size_t)2 * CH * LDT * sizeof(TA));
  auto W0 = [&](int s) { return (half*)(smem + s * STAGE); };
  auto W1 = [&](int s) { return W0(s) + C * LDW; };
  TA* Ta = (TA*)(smem + 2 * STAGE);
  TA* Tb = Ta + CH * LDT;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  // the gating linear's steps take TWO of its tiles, one a stage (w1 idles there otherwise): four accumulator
  // chains a step where they had two, and half the steps (ncu had the kernel waiting on its MMA chains)
  const int abSteps = C / CH, steps = abSteps + (t2 ? C / NC / 2 : 0);   // (no t2: the gating linear skipped)
  constexpr int ITER = (C * (NC / 8) + NTH - 1) / NTH;
  // step j's two weight tiles, each 16-byte piece handed to op(index, its place in stage st, its source)
  auto stage = [&](int j, int st, auto&& op) {
    half *w0 = W0(st), *w1 = W1(st);
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
      const int t = it * NTH + (int)threadIdx.x;
      if ((C * (NC / 8)) % NTH != 0 && t >= C * (NC / 8)) break;
      int k = t / (NC / 8), c = (t % (NC / 8)) * 8;
      const half* src = j < abSteps ? Wt + ((size_t)j * C + k) * NC + c   // the tiles run on from the projection into the gating linear
                                    : Wt + ((size_t)(abSteps + 2 * (j - abSteps)) * C + k) * NC + c + (size_t)2 * C * C;
      op(2 * it, w0 + sw(k, c), src);
      op(2 * it + 1, w1 + sw(k, c), src + (j < abSteps ? (size_t)2 * C * C : (size_t)C * NC));
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
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) ldsm4(xa[mt][ks], Xs + ((warp * MT + mt) * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
    __syncthreads();                    // every warp has its fragments: the stages take the rows' memory
    issue(0, 0);
  } else {
    static_assert(R % XROUNDS == 0 && (R / XROUNDS) % 16 == 0 && (R / XROUNDS) % WARPS == 0, "whole warps' rows a round");
    constexpr int XR = R / XROUNDS;
    issue(0, 0);                        // (the stages have memory of their own here)
#pragma unroll 1
    for (int rd = 0; rd < XROUNDS; ++rd) {
      lnRowsToShared<C, XR, WARPS, PT>(pair, [&](int r) { return pairOf(row0 + rd * XR + r); }, lnScale, lnOffset, Xs, LDX, warp, lane);
      __syncthreads();
      if (warp * 16 / XR == rd) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) ldsm4(xa[0][ks], Xs + (warp * 16 - rd * XR + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
      }
      __syncthreads();                  // the round's warps have their fragments: the next round's rows go there
    }
  }
  int lr0[MT], lr1[MT]; float m0[MT], m1[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    lr0[mt] = (warp * MT + mt) * 16 + g; lr1[mt] = lr0[mt] + 8;
    size_t pr0 = pairOf(row0 + lr0[mt]), pr1 = pairOf(row0 + lr1[mt]);
    m0[mt] = pr0 != SIZE_MAX ? mask[pr0] : 0.f; m1[mt] = pr1 != SIZE_MAX ? mask[pr1] : 0.f;
  }
  for (int j = 0; j < steps; ++j) {
    int st = j & 1;
#if LF_REG_STAGES
    if (j + 1 < steps) stage(j + 1, st ^ 1, [&](int i, half*, const half* s) { next.load(i, s); });
#else
    if (j + 1 < steps) { issue(j + 1, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
#endif
    __syncthreads();
    bool gating = j >= abSteps;
    const half *w0 = W0(st), *w1 = W1(st);
    float p[MT][NC / 8][4] = {}, q[MT][NC / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      int k = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), c = (lane >> 4) * 8;
      uint32_t f0[4];
      ldsm4t(f0, w0 + sw(k, c));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) { mma16816(p[mt][0], xa[mt][ks], f0[0], f0[1]); mma16816(p[mt][1], xa[mt][ks], f0[2], f0[3]); }
      uint32_t f1[4];
      ldsm4t(f1, w1 + sw(k, c));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) { mma16816(q[mt][0], xa[mt][ks], f1[0], f1[1]); mma16816(q[mt][1], xa[mt][ks], f1[2], f1[3]); }
    }
    if constexpr (BIAS) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) {
        int col = gating ? 4 * C + 2 * (j - abSteps) * NC + nt * 8 + tig * 2 : j * NC + nt * 8 + tig * 2;
        int qcol = gating ? col + NC : 2 * C + col;          // the second gating tile, or the gate's columns
        float b0 = bias[col], b1 = bias[col + 1], g0 = bias[qcol], g1 = bias[qcol + 1];
        p[mt][nt][0] += b0; p[mt][nt][1] += b1; p[mt][nt][2] += b0; p[mt][nt][3] += b1;
        q[mt][nt][0] += g0; q[mt][nt][1] += g1; q[mt][nt][2] += g0; q[mt][nt][3] += g1;
      }
    }
    if (gating) {
      int c0 = 2 * (j - abSteps) * NC;                         // p the first tile's columns, q the second's
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) {
        int c = c0 + nt * 8 + tig * 2;
        size_t qa = row0 + lr0[mt], qb = row0 + lr1[mt];
        if (!t2) continue;
        size_t ra = t2Row(qa), rb = t2Row(qb);
        if (qa < pp) *reinterpret_cast<half2*>(t2 + ra * C + c) = __floats2half2_rn(p[mt][nt][0], p[mt][nt][1]);
        if (qb < pp) *reinterpret_cast<half2*>(t2 + rb * C + c) = __floats2half2_rn(p[mt][nt][2], p[mt][nt][3]);
        if (qa < pp) *reinterpret_cast<half2*>(t2 + ra * C + c + NC) = __floats2half2_rn(q[mt][nt][0], q[mt][nt][1]);
        if (qb < pp) *reinterpret_cast<half2*>(t2 + rb * C + c + NC) = __floats2half2_rn(q[mt][nt][2], q[mt][nt][3]);
      }
    } else {
      // column 2ch is a's channel ch, 2ch+1 b's: the thread's columns nt*8 + 2 tig are channel nt*4 + tig
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) {
        int ch = nt * 4 + tig;
        Ta[ch * LDT + lr0[mt]] = TA(p[mt][nt][0] * sigmH(q[mt][nt][0]) * m0[mt]);
        Tb[ch * LDT + lr0[mt]] = TA(p[mt][nt][1] * sigmH(q[mt][nt][1]) * m0[mt]);
        Ta[ch * LDT + lr1[mt]] = TA(p[mt][nt][2] * sigmH(q[mt][nt][2]) * m1[mt]);
        Tb[ch * LDT + lr1[mt]] = TA(p[mt][nt][3] * sigmH(q[mt][nt][3]) * m1[mt]);
      }
      __syncthreads();
#pragma unroll
      for (int t0_ = 0; t0_ < CH * (R / 8); t0_ += NTH) {
        const int t = t0_ + (int)threadIdx.x;
        if ((CH * (R / 8)) % NTH != 0 && t >= CH * (R / 8)) break;
        int ch = t / (R / 8), r = (t % (R / 8)) * 8;
        size_t row = row0 + r;
        if (row < cs) {
          size_t at = (size_t)(j * CH + ch) * cs + row;
          // (8 rows: 16 bytes of a 16-bit operand, 8 of an FP8 one)
          using V8 = std::conditional_t<sizeof(TA) == 1, uint2, uint4>;
          if (a) *reinterpret_cast<V8*>(a + at) = *reinterpret_cast<const V8*>(Ta + ch * LDT + r);
          if (b) *reinterpret_cast<V8*>(b + at) = *reinterpret_cast<const V8*>(Tb + ch * LDT + r);
        }
      }
    }
#if LF_REG_STAGES
    if (j + 1 < steps) stage(j + 1, st ^ 1, [&](int i, half* d, const half*) { next.store(i, d); });   // (stage st^1 was
#endif                                                                         // last read before this step's barrier)
    __syncthreads();
  }
}

// the triangle's output side, launched: the bf16 tile at 16-column stages (LOCALFOLD_TRIOUT_F32=1: the float tile)
template <int C, int WARPS, class TP = float>
void triangleOutRun(const TP* prod, const float* sc, const float* of, const half* Wout, const half* t2, float* pair,
                    int L, int Lp, const float* ob = nullptr, RectMap rm = {}) {
  static const bool f32 = getenv("LOCALFOLD_TRIOUT_F32") != nullptr;
  if (rm.J && (sizeof(TP) != 2 || f32)) { fprintf(stderr, "triangleOutRun: a rectangle wants the bf16 product and tile\n"); exit(1); }
  constexpr int R = 16 * WARPS;
  size_t P = (size_t)L * L;
  half* wt = scratch<half>("triout.wt", (size_t)C * C);
  tileColumns(Wout, C, C, 0, C, f32 || (ob && sizeof(TP) == 4) ? 32 : 16, wt);
  if (f32 && !ob) {
    constexpr size_t smem = triangleOutSmem<C, WARPS, float, 32>();
    WITH_PAIR_T(
      static bool attr = false;
      if (!attr) { smemAttr((triangleOutK<C, WARPS, float, 32, TP, false, PT>), (int)smem); attr = true; }
      triangleOutK<C, WARPS, float, 32, TP, false, PT><<<(unsigned)((P + R - 1) / R), 32 * WARPS, smem, STREAM>>>(prod, sc, of, wt, t2, pair, L, Lp));
  } else if (ob) {
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
  } else {
    constexpr size_t smem = triangleOutSmem<C, WARPS, __nv_bfloat16, 16, TP>();
    constexpr bool vec = sizeof(TP) == 2;                       // the padded rows (triangleOutK's VEC)
    size_t rows = rm.J ? rm.size : vec ? (size_t)Lp * Lp : P;
    WITH_PAIR_T(
      static bool attr = false;
      if (!attr) { smemAttr((triangleOutK<C, WARPS, __nv_bfloat16, 16, TP, false, PT>), (int)smem); attr = true; }
      triangleOutK<C, WARPS, __nv_bfloat16, 16, TP, false, PT><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(prod, sc, of, wt, t2, pair, L, Lp, nullptr, rm));
  }
}

