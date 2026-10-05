// Fused kernels for a 256-channel pair track (ESMFold2's, protenix2's), on the f16 tensor cores (mma.sync m16n8k16).
// native/af3's fused kernels hold a whole 256-wide output tile or weight on the chip, which at 256
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
template <int C, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) transitionUpK(const float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ W1, half* __restrict__ gated, size_t rows, int I) {
  constexpr int NC = 32, R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, LDA = NC + 8, KS = C / 16;
  constexpr size_t STAGE = (size_t)2 * C * LDA * 2;
  extern __shared__ __align__(16) unsigned char smem[];
  // the LN'd rows and the weight stages share memory: the rows go into registers (A fragments) before
  // the first stage is issued, so a block's footprint is the larger of the two, not their sum
  half* Xs = (half*)smem;
  unsigned char* stages = smem;
  auto Wa = [&](int s) { return (half*)(stages + s * STAGE); };
  auto Wb = [&](int s) { return Wa(s) + C * LDA; };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  int chunks = I / NC;
  auto issue = [&](int j, int st) {
    half *a = Wa(st), *b = Wb(st);
    for (int t = threadIdx.x; t < C * (NC / 8); t += NTH) {
      int k = t / (NC / 8), c = (t % (NC / 8)) * 8;
      cpAsync16(a + k * LDA + c, W1 + (size_t)k * 2 * I + j * NC + c, true);
      cpAsync16(b + k * LDA + c, W1 + (size_t)k * 2 * I + I + j * NC + c, true);
    }
    cpCommit();
  };
  lnRowsToShared<C, R, WARPS>(x, [&](int r) { size_t row = row0 + r; return row < rows ? row : SIZE_MAX; },
                              lnScale, lnOffset, Xs, LDX, warp, lane);
  __syncthreads();
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  __syncthreads();                    // every warp has its fragments: the stages may overwrite the rows
  issue(0, 0);
  for (int j = 0; j < chunks; ++j) {
    int st = j & 1;
    if (j + 1 < chunks) { issue(j + 1, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
    __syncthreads();
    const half *a = Wa(st), *b = Wb(st);
    float ha[NC / 8][4] = {}, hb[NC / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int n2 = 0; n2 < NC / 16; ++n2) {
        uint32_t fa[4], fb[4];
        int k = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), c = n2 * 16 + (lane >> 4) * 8;
        ldsm4t(fa, a + k * LDA + c);
        ldsm4t(fb, b + k * LDA + c);
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
      if (r0 < rows) *reinterpret_cast<uint32_t*>(gated + r0 * I + c) = pack2(gate(0), gate(1));
      if (r1 < rows) *reinterpret_cast<uint32_t*>(gated + r1 * I + c) = pack2(gate(2), gate(3));
    }
    __syncthreads();
  }
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
// prod: channel-major [C][Lp][Lp] f32 (the padded contraction's output); t2: the gating linear's raw
// output, [Lp * Lp][C] f16 (triInK's); pair: [L * L][C] f32. A block is 16 WARPS pairs: the product
// tile is staged channel-major (stride R + 1: the per-row reductions read a column, conflict-free), the
// centre norm's statistics taken per row, and each warp's A fragments built straight from the tile,
// normalised on the way; the tile's memory then holds Wout's stages, 32 output columns at a time.
// TT, NC: the tile's type and the output columns a weight stage holds. A float tile at 32 columns is 66.5 KB a
// block (two an SM on an A100); a bf16 tile at 16 is 34 KB (four, the registers' limit) - the product held
// at AF3's activation precision, which is what AF3's own triangle runs in
template <int C, int WARPS, class TT = float, int NC = 32>
__host__ __device__ constexpr size_t triangleOutSmem() {
  constexpr int R = 16 * WARPS;
  constexpr size_t tile = (size_t)C * (R + 1) * sizeof(TT), stages = (size_t)2 * C * (NC + 8) * 2 + (size_t)R * (NC + 4) * 4;
  return (tile > stages ? tile : stages) + 2 * (size_t)R * 4;
}
template <int C, int WARPS, class TT = float, int NC = 32>
__global__ void __launch_bounds__(WARPS * 32) triangleOutK(const float* __restrict__ prod, const float* __restrict__ cnScale,
    const float* __restrict__ cnOffset, const half* __restrict__ Wout, const half* __restrict__ t2,
    float* __restrict__ pair, int L, int Lp) {
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDP = R + 1, LDW = NC + 8, KS = C / 16;
  constexpr size_t STAGE = (size_t)C * LDW * 2;
  constexpr size_t PS = triangleOutSmem<C, WARPS, TT, NC>() - 2 * (size_t)R * 4;
  extern __shared__ __align__(16) unsigned char smem[];
  TT* Ps = (TT*)smem;                                         // [C][LDP], then the weight stages
  float* stat = (float*)(smem + PS);                          // [R] mean, [R] 1/sd
  auto Ws = [&](int s) { return (half*)(smem + s * STAGE); };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const size_t P = (size_t)L * L, plane = (size_t)Lp * Lp;
  size_t row0 = (size_t)blockIdx.x * R;
  auto padded = [&](size_t r) { unsigned u = (unsigned)r, i = u / (unsigned)L; return (size_t)i * Lp + (u - i * L); };
  static_assert(NTH % R == 0, "a thread keeps one pair of the tile");
  {   // a thread keeps one pair (its padded index computed once) and walks the channels
    int r = threadIdx.x % R;
    size_t row = row0 + r;
    bool live = row < P;
    const float* src = prod + (live ? padded(row) : 0);
#pragma unroll 32
    for (int c = threadIdx.x / R; c < C; c += NTH / R) Ps[c * LDP + r] = TT(live ? src[(size_t)c * plane] : 0.f);
  }
  __syncthreads();
  for (int r = warp; r < R; r += WARPS) {                     // the centre norm's mean and 1/sd, a warp a row
    float v[C / 32], s = 0.f;
#pragma unroll
    for (int k = 0; k < C / 32; ++k) { v[k] = (float)Ps[(lane + 32 * k) * LDP + r]; s += v[k]; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / C, q = 0.f;
#pragma unroll
    for (int k = 0; k < C / 32; ++k) { float d = v[k] - mean; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
    if (lane == 0) { stat[r] = mean; stat[R + r] = rsqrtf(q / C + 1e-5f); }
  }
  __syncthreads();
  // A fragments of m16n8k16: a0 (row g, k 2tig..+1), a1 (row g+8, ...), a2 (row g, k+8), a3 (row g+8, k+8)
  int ra = warp * 16 + g, rb = ra + 8;
  float ma = stat[ra], ia = stat[R + ra], mb = stat[rb], ib = stat[R + rb];
  auto nrm = [&](int r, float m, float inv, int k) { return ((float)Ps[k * LDP + r] - m) * inv * cnScale[k] + cnOffset[k]; };
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
  auto issue = [&](int n, int st) {
    half* w = Ws(st);
    for (int t = threadIdx.x; t < C * (NC / 8); t += NTH) {
      int k = t / (NC / 8), c = (t % (NC / 8)) * 8;
      cpAsync16(w + k * LDW + c, Wout + (size_t)k * C + n * NC + c, true);
    }
    cpCommit();
  };
  issue(0, 0);
  // the output chunk, staged so the gate and the residual go out 16 bytes a thread, a row's 32 columns
  // contiguous (a fragment's own stores are 8 bytes a row, eight rows a warp)
  constexpr int LDO = NC + 4;
  float* Os = (float*)(smem + 2 * STAGE);
  constexpr int chunks = C / NC;
  for (int n = 0; n < chunks; ++n) {
    int st = n & 1;
    if (n + 1 < chunks) { issue(n + 1, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
    __syncthreads();
    const half* w = Ws(st);
    float acc[NC / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int n2 = 0; n2 < NC / 16; ++n2) {
        uint32_t fb[4];
        ldsm4t(fb, w + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW + n2 * 16 + (lane >> 4) * 8);
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
    for (int t = threadIdx.x; t < R * (NC / 4); t += NTH) {
      int r = t / (NC / 4), q = (t % (NC / 4)) * 4;
      size_t row = row0 + r;
      if (row >= P) continue;
      int c = n * NC + q;
      float4 o = *reinterpret_cast<const float4*>(Os + r * LDO + q);
      uint2 gw = *reinterpret_cast<const uint2*>(t2 + padded(row) * C + c);
      float2 g01 = __half22float2(*reinterpret_cast<half2*>(&gw.x)), g23 = __half22float2(*reinterpret_cast<half2*>(&gw.y));
      float4* d = reinterpret_cast<float4*>(pair + row * C + c); float4 v = *d;
      v.x += o.x * sigmH(g01.x); v.y += o.y * sigmH(g01.y); v.z += o.z * sigmH(g23.x); v.w += o.w * sigmH(g23.y);
      *d = v;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------- the triangle's input side
// native/af3's triInK, re-laid for 256 channels: at its 32-column steps the weight stages (82 KB) sit
// beside the LN'd rows (68 KB), one block an SM. Here the steps are 16 columns and the stages take the
// rows' memory once they are A fragments, so a block is the rows' 68 KB and two fit an SM.
// Output as triInK's: a, b channel-major padded planes (interleaved split, masked), t2 the gating
// linear's raw output over the padded rows.
template <int C, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) triIn256K(const float* __restrict__ pair, const float* __restrict__ mask,
    const float* __restrict__ lnScale, const float* __restrict__ lnOffset, const half* __restrict__ Wpg,
    const half* __restrict__ Wg, half* __restrict__ a, half* __restrict__ b, half* __restrict__ t2, int n, int np,
    size_t cs) {
  // the weight stage is unpadded, [k][NC], its two 16-byte halves swapped on rows with bit 2 of k set
  // (sw): the padded 48-byte rows kept ldmatrix conflict-free but serialised the cp.async writes ~6.7x
  // (ncu: 40% of the kernel's shared wavefronts excessive, all of them those four LDGSTS); swizzled,
  // both are conflict-free - 87.9 -> 84.7 ms of ESMFold2's 5CAJ trunk, byte-identical
  constexpr int NC = 16, CH = NC / 2, R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, LDW = NC, KS = C / 16, LDT = R + 8;
  auto sw = [](int k, int c) { return k * LDW + (c ^ (((k >> 2) & 1) << 3)); };
  constexpr size_t STAGE = (size_t)2 * C * LDW * 2;
  static_assert(2 * STAGE + (size_t)2 * CH * LDT * 2 <= (size_t)R * LDX * 2, "the stages and the a/b staging fit in the rows");
  const size_t pp = (size_t)np * np;
  auto pairOf = [&](size_t q) -> size_t {
    if (q >= pp) return SIZE_MAX;
    unsigned u = (unsigned)q, i = u / (unsigned)np, j = u - i * (unsigned)np;
    return i < (unsigned)n && j < (unsigned)n ? (size_t)i * n + j : SIZE_MAX;
  };
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem;
  auto W0 = [&](int s) { return (half*)(smem + s * STAGE); };
  auto W1 = [&](int s) { return W0(s) + C * LDW; };
  half* Ta = (half*)(smem + 2 * STAGE);
  half* Tb = Ta + CH * LDT;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  const int abSteps = C / CH, steps = abSteps + C / NC;
  auto issue = [&](int j, int st) {
    half *w0 = W0(st), *w1 = W1(st);
#pragma unroll
    for (int t0_ = 0; t0_ < C * (NC / 8); t0_ += NTH) {
      const int t = t0_ + (int)threadIdx.x;
      if ((C * (NC / 8)) % NTH != 0 && t >= C * (NC / 8)) break;
      int k = t / (NC / 8), c = (t % (NC / 8)) * 8;
      if (j < abSteps) {
        cpAsync16(w0 + sw(k, c), Wpg + (size_t)k * 4 * C + j * NC + c, true);
        cpAsync16(w1 + sw(k, c), Wpg + (size_t)k * 4 * C + 2 * C + j * NC + c, true);
      } else {
        cpAsync16(w0 + sw(k, c), Wg + (size_t)k * C + (j - abSteps) * NC + c, true);
      }
    }
    cpCommit();
  };
  lnRowsToShared<C, R, WARPS>(pair, [&](int r) { return pairOf(row0 + r); }, lnScale, lnOffset, Xs, LDX, warp, lane);
  __syncthreads();
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  __syncthreads();                    // every warp has its fragments: the stages take the rows' memory
  issue(0, 0);
  int lr0 = warp * 16 + g, lr1 = lr0 + 8;
  size_t pr0 = pairOf(row0 + lr0), pr1 = pairOf(row0 + lr1);
  float m0 = pr0 != SIZE_MAX ? mask[pr0] : 0.f, m1 = pr1 != SIZE_MAX ? mask[pr1] : 0.f;
  for (int j = 0; j < steps; ++j) {
    int st = j & 1;
    if (j + 1 < steps) { issue(j + 1, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
    __syncthreads();
    bool gating = j >= abSteps;
    const half *w0 = W0(st), *w1 = W1(st);
    float p[NC / 8][4] = {}, q[NC / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      int k = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), c = (lane >> 4) * 8;
      uint32_t f0[4];
      ldsm4t(f0, w0 + sw(k, c));
      mma16816(p[0], xa[ks], f0[0], f0[1]); mma16816(p[1], xa[ks], f0[2], f0[3]);
      if (!gating) {
        uint32_t f1[4];
        ldsm4t(f1, w1 + sw(k, c));
        mma16816(q[0], xa[ks], f1[0], f1[1]); mma16816(q[1], xa[ks], f1[2], f1[3]);
      }
    }
    if (gating) {
      int c0 = (j - abSteps) * NC;
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) {
        int c = c0 + nt * 8 + tig * 2;
        if (row0 + lr0 < pp) *reinterpret_cast<half2*>(t2 + (row0 + lr0) * C + c) = __floats2half2_rn(p[nt][0], p[nt][1]);
        if (row0 + lr1 < pp) *reinterpret_cast<half2*>(t2 + (row0 + lr1) * C + c) = __floats2half2_rn(p[nt][2], p[nt][3]);
      }
    } else {
      // column 2ch is a's channel ch, 2ch+1 b's: the thread's columns nt*8 + 2 tig are channel nt*4 + tig
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) {
        int ch = nt * 4 + tig;
        Ta[ch * LDT + lr0] = __float2half(p[nt][0] * sigmH(q[nt][0]) * m0);
        Tb[ch * LDT + lr0] = __float2half(p[nt][1] * sigmH(q[nt][1]) * m0);
        Ta[ch * LDT + lr1] = __float2half(p[nt][2] * sigmH(q[nt][2]) * m1);
        Tb[ch * LDT + lr1] = __float2half(p[nt][3] * sigmH(q[nt][3]) * m1);
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
          *reinterpret_cast<uint4*>(a + at) = *reinterpret_cast<const uint4*>(Ta + ch * LDT + r);
          *reinterpret_cast<uint4*>(b + at) = *reinterpret_cast<const uint4*>(Tb + ch * LDT + r);
        }
      }
    }
    __syncthreads();
  }
}

// the triangle's output side, launched: the bf16 tile at 16-column stages (LOCALFOLD_TRIOUT_F32=1: the float tile)
template <int C, int WARPS>
void triangleOutRun(const float* prod, const float* sc, const float* of, const half* Wout, const half* t2, float* pair,
                    int L, int Lp) {
  static const bool f32 = getenv("LOCALFOLD_TRIOUT_F32") != nullptr;
  constexpr int R = 16 * WARPS;
  size_t P = (size_t)L * L;
  if (f32) {
    constexpr size_t smem = triangleOutSmem<C, WARPS, float, 32>();
    static bool attr = false;
    if (!attr) { smemAttr((triangleOutK<C, WARPS, float, 32>), (int)smem); attr = true; }
    triangleOutK<C, WARPS, float, 32><<<(unsigned)((P + R - 1) / R), 32 * WARPS, smem, STREAM>>>(prod, sc, of, Wout, t2, pair, L, Lp);
  } else {
    constexpr size_t smem = triangleOutSmem<C, WARPS, __nv_bfloat16, 16>();
    static bool attr = false;
    if (!attr) { smemAttr((triangleOutK<C, WARPS, __nv_bfloat16, 16>), (int)smem); attr = true; }
    triangleOutK<C, WARPS, __nv_bfloat16, 16><<<(unsigned)((P + R - 1) / R), 32 * WARPS, smem, STREAM>>>(prod, sc, of, Wout, t2, pair, L, Lp);
  }
}

