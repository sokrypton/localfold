// AF2's MSAColumnGlobalAttention on the extra MSA, fused: four kernels on the MSA in its own [N][L][C] layout,
// where the unfused form transposed it to [L][N][C] (and back), LayerNormed, projected and gated in f32 passes
// of their own - eleven passes over a 5,120-row extra MSA, 7.9 ms a call at 261 residues on an A100, 8% of a
// fold (nvfold's: ~0.7 ms). C = 64, eight heads of eight (the extra stack's shape).
//
//   gaInK:    LN(row) on the way in (four lanes... two a row: 32 channels a lane), then on the tensor cores
//             [k | v | gate] = LN(row) [Wk | Wv | Wg]: k, v to kv [L][N][16] (f32, a column's keys contiguous),
//             sigmoid(gate + bg) to gate [N][L][64] (f16); the masked column sums of LN(row) and the mask's
//             sums into qsum [L][64], msum [L] (a slab a row block, summed in gaQueryK in the blocks' order: atomics
//             made the fold depend on which block finished first - 5CAJ's pLDDT 95.05-95.51 over four runs)
//   gaQueryK: q[b] = (qsum[b] / msum[b]) Wq / sqrt(D)
//   gaAttnK:  a block a column: the online softmax over its N rows, per head, of q.k (+ the mask's -1e9),
//             avg[b] = softmax . v
//   gaOutK:   msa[s, b] += (gate[s, b] * avg[b]) Wo + bo
//
// The arithmetic is the reference's but for the projections' operands in f16 (f32 accumulation), as every other
// FAST projection here.
namespace ga {
constexpr int C = 64, H = 8, D = 8, NIN = 2 * D + H * D;   // [k | v | gate] columns: 80
}

// msa [N][L][C] f32, mask [N][L]; Wk, Wv [C][D], Wg [C][H*D] (f16); one block a 16-column group and a run of rows
template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32, 4) gaInK(const float* __restrict__ msa, const float* __restrict__ mask,
    const float* __restrict__ lnS, const float* __restrict__ lnO, const half* __restrict__ wk, const half* __restrict__ wv,
    const half* __restrict__ wg, const float* __restrict__ gb, float* __restrict__ kv, half* __restrict__ gate,
    float* __restrict__ qsum, float* __restrict__ msum, int L, int N, int rowsPerBlock) {
  constexpr int C = ga::C, H = ga::H, D = ga::D, NIN = ga::NIN;
  constexpr int LDW = NIN + 8, LDX = C + 8;
  __shared__ __align__(16) union {
    struct {
      half Ws[C * LDW];
      half Xs[WARPS][16 * LDX];
    };
    struct {
      float red[WARPS][32][33];
      float mred[WARPS][16];
    };
  } sm;
  __shared__ __align__(16) float sS[C], sO[C], sGb[H * D];
  for (int t = threadIdx.x; t < C * (NIN / 8); t += WARPS * 32) {
    int k = t / (NIN / 8), c = (t % (NIN / 8)) * 8;
    uint4 wv_ = c < D ? ptx_ldg_v4_u32(wk + k * D + c)
                      : c < 2 * D ? ptx_ldg_v4_u32(wv + k * D + c - D)
                                  : ptx_ldg_v4_u32(wg + k * H * D + c - 2 * D);
    ptx_st_shared_v4_u32(sm.Ws + k * LDW + c, wv_);
  }
  if (threadIdx.x < C / 4) {
    int c4 = threadIdx.x * 4;
    ptx_st_shared_v4_f32(sS + c4, ptx_ldg_v4_f32(lnS + c4));
    ptx_st_shared_v4_f32(sO + c4, ptx_ldg_v4_f32(lnO + c4));
    ptx_st_shared_v4_f32(sGb + c4, ptx_ldg_v4_f32(gb + c4));
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int b0 = blockIdx.x * 16, tok = lane >> 1, hf = lane & 1, b = b0 + tok;
  const bool valid = b < L;
  const int s0 = blockIdx.y * rowsPerBlock, s1 = min(N, s0 + rowsPerBlock);
  float qacc[32], macc = 0.f;
#pragma unroll
  for (int c = 0; c < 32; ++c) qacc[c] = 0.f;
  half* xs = sm.Xs[warp];
  const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
  for (int s = s0 + warp; s < s1; s += WARPS) {
    float v[32], m = 0.f;
    if (valid) {
      const float4* p = reinterpret_cast<const float4*>(msa + ((size_t)s * L + b) * C + hf * 32);
#pragma unroll
      for (int i = 0; i < 8; ++i) { float4 f = ptx_ldg_v4_f32(p + i); v[4 * i] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w; }
      m = ptx_ldg_f32(mask + (size_t)s * L + b);
    } else {
#pragma unroll
      for (int c = 0; c < 32; ++c) v[c] = 0.f;
    }
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < 32; ++c) sum = ptx_add_f32(sum, v[c]);
    sum = ptx_add_f32(sum, ptx_shfl_xor_f32(sum, 1));
    const float mean = ptx_div_f32(sum, (float)C);
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < 32; ++c) { float d = ptx_sub_f32(v[c], mean); q = ptx_fma_f32(d, d, q); }
    q = ptx_add_f32(q, ptx_shfl_xor_f32(q, 1));
    const float inv = ptx_rsqrt_f32(ptx_add_f32(ptx_div_f32(q, (float)C), 1e-5f));
    macc = ptx_add_f32(macc, m);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c0 = 8 * i, ch0 = hf * 32 + c0;
      const float4 s4_0 = ptx_ld_shared_v4_f32(sS + ch0), s4_1 = ptx_ld_shared_v4_f32(sS + ch0 + 4);
      const float4 o4_0 = ptx_ld_shared_v4_f32(sO + ch0), o4_1 = ptx_ld_shared_v4_f32(sO + ch0 + 4);
      const float sv[8] = {s4_0.x, s4_0.y, s4_0.z, s4_0.w, s4_1.x, s4_1.y, s4_1.z, s4_1.w};
      const float ov[8] = {o4_0.x, o4_0.y, o4_0.z, o4_0.w, o4_1.x, o4_1.y, o4_1.z, o4_1.w};
      uint4 o4; uint32_t* oh = reinterpret_cast<uint32_t*>(&o4);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int c = c0 + 2 * e;
        const float x0 = ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(v[c], mean), inv), sv[2 * e], ov[2 * e]);
        const float x1 = ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(v[c + 1], mean), inv), sv[2 * e + 1], ov[2 * e + 1]);
        qacc[c] = ptx_fma_f32(m, x0, qacc[c]); qacc[c + 1] = ptx_fma_f32(m, x1, qacc[c + 1]);
        oh[e] = ptx_cvt_f16x2_f32(x0, x1);
      }
      ptx_st_shared_v4_u32(xs + tok * LDX + hf * 32 + 8 * i, o4);
    }
    __syncwarp();
    uint32_t xa[C / 16][4];
#pragma unroll
    for (int ks = 0; ks < C / 16; ++ks) ldsm4(xa[ks], xs + (lane & 15) * LDX + ks * 16 + (lane >> 4) * 8);
    float acc[NIN / 8][4] = {};
    uint32_t f[2][4];
    ldsm4t(f[0], sm.Ws + kl * LDW + cl);
#pragma unroll
    for (int ks = 0; ks < C / 16; ++ks)
#pragma unroll
      for (int n2 = 0; n2 < NIN / 16; ++n2) {
        const int step = ks * (NIN / 16) + n2, cur = step & 1;
        if (n2 + 1 < NIN / 16) ldsm4t(f[cur ^ 1], sm.Ws + (ks * 16 + kl) * LDW + (n2 + 1) * 16 + cl);
        else if (ks + 1 < C / 16) ldsm4t(f[cur ^ 1], sm.Ws + ((ks + 1) * 16 + kl) * LDW + cl);
        mma16816(acc[2 * n2], xa[ks], f[cur][0], f[cur][1]); mma16816(acc[2 * n2 + 1], xa[ks], f[cur][2], f[cur][3]);
      }
    float2 b2All[8];
#pragma unroll
    for (int nt = 2; nt < NIN / 8; ++nt) {
      b2All[nt - 2] = ptx_ld_shared_v2_f32(sGb + (nt - 2) * 8 + tig * 2);
    }
    __syncwarp();                         // (every lane has its fragments: the next row's LN may take the tile)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int bt = b0 + g + 8 * hh;
      if (bt >= L) continue;
#pragma unroll
      for (int nt = 0; nt < NIN / 8; ++nt) {
        const int col = nt * 8 + tig * 2;
        const float a0 = acc[nt][2 * hh], a1 = acc[nt][2 * hh + 1];
        if (nt < 2) ptx_st_global_v2_f32(kv + ((size_t)bt * N + s) * 2 * D + col, make_float2(a0, a1));
        else {
          const int gc = col - 2 * D;
          const float2 b2 = b2All[nt - 2];
          ptx_st_global_u32(gate + ((size_t)s * L + bt) * H * D + gc,
              ptx_cvt_f16x2_f32(ptx_sigm_f32(ptx_add_f32(a0, b2.x)), ptx_sigm_f32(ptx_add_f32(a1, b2.y))));
        }
      }
    }
  }
  __syncthreads();
  // the block's masked sums: across its warps in shared memory, into this row block's slab
  qsum += (size_t)blockIdx.y * ((size_t)L * C + L); msum = qsum + (size_t)L * C;
#pragma unroll
  for (int c = 0; c < 32; ++c) ptx_st_shared_f32(&sm.red[warp][lane][c], qacc[c]);
  if (hf == 0) ptx_st_shared_f32(&sm.mred[warp][tok], macc);
  __syncthreads();
#pragma unroll
  for (int it = 0; it < (32 * 32) / (WARPS * 32); ++it) {
    const int ln = it * WARPS + warp, c = lane, bb = b0 + (ln >> 1);
    if (bb >= L) continue;
    float r[WARPS];
#pragma unroll
    for (int w = 0; w < WARPS; ++w) r[w] = ptx_ld_shared_f32(&sm.red[w][ln][c]);
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) a = ptx_add_f32(a, r[w]);
    ptx_st_global_f32(qsum + (size_t)bb * C + (ln & 1) * 32 + c, a);
  }
  if (threadIdx.x < 16 && b0 + (int)threadIdx.x < L) {
    const int t = threadIdx.x;
    float r[WARPS];
#pragma unroll
    for (int w = 0; w < WARPS; ++w) r[w] = ptx_ld_shared_f32(&sm.mred[w][t]);
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) a = ptx_add_f32(a, r[w]);
    ptx_st_global_f32(msum + b0 + t, a);
  }
}

// q[b][h*D + d] = (qsum[b] / (msum[b] + 1e-10)) Wq / sqrt(D); Wq [C][H*D] f32. A block (C threads) a column; qsum,
// msum the row blocks' slabs ([Y][L*C + L]), summed in order
__global__ void __launch_bounds__(64) gaQueryK(const float* __restrict__ part, int L, int Y,
    const float* __restrict__ wq, float* __restrict__ q) {
  constexpr int C = ga::C, H = ga::H, D = ga::D;
  __shared__ float qa[C];
  const int b = blockIdx.x, t = threadIdx.x;
  const size_t slab = (size_t)L * C + L;
  float qs = 0.f, ms = 0.f;
  for (int y = 0; y < Y; ++y) {
    float qy = ptx_ldg_f32(part + y * slab + (size_t)b * C + t);
    float my = ptx_ldg_f32(part + y * slab + (size_t)L * C + b);
    qs = ptx_add_f32(qs, qy);
    ms = ptx_add_f32(ms, my);
  }
  ptx_st_shared_f32(qa + t, ptx_div_f32(qs, ptx_add_f32(ms, 1e-10f)));
  __syncthreads();
  float a = 0.f;
#pragma unroll
  for (int c = 0; c < C; c += 4) {
    float4 q4 = ptx_ld_shared_v4_f32(qa + c);
    float w0 = ptx_ldg_f32(wq + (size_t)c * H * D + t);
    float w1 = ptx_ldg_f32(wq + (size_t)(c + 1) * H * D + t);
    float w2 = ptx_ldg_f32(wq + (size_t)(c + 2) * H * D + t);
    float w3 = ptx_ldg_f32(wq + (size_t)(c + 3) * H * D + t);
    a = ptx_fma_f32(q4.x, w0, a);
    a = ptx_fma_f32(q4.y, w1, a);
    a = ptx_fma_f32(q4.z, w2, a);
    a = ptx_fma_f32(q4.w, w3, a);
  }
  ptx_st_global_f32(q + (size_t)b * H * D + t, ptx_mul_f32(a, ptx_rsqrt_f32((float)D)));
}

// a block a column: avg[b][h][d] = sum_s softmax_s(q[b][h] . k[b][s] + 1e9 (mask - 1)) v[b][s][d]
__global__ void __launch_bounds__(256, 2) gaAttnK(const float* __restrict__ kv, const float* __restrict__ mask,
    const float* __restrict__ qg, float* __restrict__ avg, int L, int N) {
  constexpr int H = ga::H, D = ga::D;
  __shared__ __align__(16) float qh[H * D];
  __shared__ float part[8][H][D + 2];
  const int b = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (threadIdx.x < H * D) ptx_st_shared_f32(qh + threadIdx.x, ptx_ldg_f32(qg + (size_t)b * H * D + threadIdx.x));
  __syncthreads();
  float m[H], l[H], acc[H][D];
#pragma unroll
  for (int h = 0; h < H; ++h) {
    m[h] = -INFINITY; l[h] = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d) acc[h][d] = 0.f;
  }
  for (int s = threadIdx.x; s < N; s += 256) {
    const float4* p = reinterpret_cast<const float4*>(kv + ((size_t)b * N + s) * 2 * D);
    const float4 k0 = ptx_ldg_v4_f32(p), k1 = ptx_ldg_v4_f32(p + 1), v0 = ptx_ldg_v4_f32(p + 2), v1 = ptx_ldg_v4_f32(p + 3);
    const float m_raw = ptx_ldg_f32(mask + (size_t)s * L + b);
    const float k[D] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w}, v[D] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    const float bias = ptx_mul_f32(1e9f, ptx_sub_f32(m_raw, 1.f));
#pragma unroll
    for (int h = 0; h < H; ++h) {
      float4 q0 = ptx_ld_shared_v4_f32(qh + h * D), q1 = ptx_ld_shared_v4_f32(qh + h * D + 4);
      float lg = bias;
      lg = ptx_fma_f32(q0.x, k[0], lg); lg = ptx_fma_f32(q0.y, k[1], lg);
      lg = ptx_fma_f32(q0.z, k[2], lg); lg = ptx_fma_f32(q0.w, k[3], lg);
      lg = ptx_fma_f32(q1.x, k[4], lg); lg = ptx_fma_f32(q1.y, k[5], lg);
      lg = ptx_fma_f32(q1.z, k[6], lg); lg = ptx_fma_f32(q1.w, k[7], lg);
      const float mn = ptx_max_f32(m[h], lg), cs = ptx_exp_f32(ptx_sub_f32(m[h], mn)), e = ptx_exp_f32(ptx_sub_f32(lg, mn));
      l[h] = ptx_fma_f32(l[h], cs, e);
#pragma unroll
      for (int d = 0; d < D; ++d) acc[h][d] = ptx_fma_f32(acc[h][d], cs, ptx_mul_f32(e, v[d]));
      m[h] = mn;
    }
  }
  auto merge = [&](int h, float m2, float l2, const float* a2) {
    const float M = ptx_max_f32(m[h], m2);
    if (M == -INFINITY) return;
    const float ca = ptx_exp_f32(ptx_sub_f32(m[h], M)), cb = ptx_exp_f32(ptx_sub_f32(m2, M));
    l[h] = ptx_fma_f32(l[h], ca, ptx_mul_f32(l2, cb));
#pragma unroll
    for (int d = 0; d < D; ++d) acc[h][d] = ptx_fma_f32(acc[h][d], ca, ptx_mul_f32(a2[d], cb));
    m[h] = M;
  };
  for (int o = 16; o; o >>= 1) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
      float a2[D];
      const float m2 = ptx_shfl_xor_f32(m[h], o), l2 = ptx_shfl_xor_f32(l[h], o);
#pragma unroll
      for (int d = 0; d < D; ++d) a2[d] = ptx_shfl_xor_f32(acc[h][d], o);
      merge(h, m2, l2, a2);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
      ptx_st_shared_f32(&part[warp][h][0], m[h]); ptx_st_shared_f32(&part[warp][h][1], l[h]);
#pragma unroll
      for (int d = 0; d < D; ++d) ptx_st_shared_f32(&part[warp][h][2 + d], acc[h][d]);
    }
  }
  __syncthreads();
  if (threadIdx.x < H * D) {
    const int h = threadIdx.x / D, d = threadIdx.x % D;
    float pw0[8], M = -INFINITY;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      pw0[w] = ptx_ld_shared_f32(&part[w][h][0]);
      M = ptx_max_f32(M, pw0[w]);
    }
    float Ls = 0.f, A = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      if (pw0[w] == -INFINITY) continue;
      const float c = ptx_exp_f32(ptx_sub_f32(pw0[w], M));
      Ls = ptx_fma_f32(ptx_ld_shared_f32(&part[w][h][1]), c, Ls);
      A = ptx_fma_f32(ptx_ld_shared_f32(&part[w][h][2 + d]), c, A);
    }
    ptx_st_global_f32(avg + (size_t)b * H * D + threadIdx.x, ptx_div_f32(A, Ls));
  }
}

// msa[s][b] += (gate[s][b] * avg[b]) Wo + bo; Wo [H*D][C] f16. One block a 16-column group and a run of rows
template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32, 4) gaOutK(const half* __restrict__ gate, const float* __restrict__ avg,
    const half* __restrict__ wo, const float* __restrict__ bo, float* __restrict__ msa, int L, int N, int rowsPerBlock) {
  constexpr int C = ga::C, H = ga::H, D = ga::D;
  constexpr int K = H * D, LDW = C + 8, LDX = K + 8;
  __shared__ __align__(16) half Ws[K * LDW];
  __shared__ __align__(16) half Xs[WARPS][16 * LDX];
  __shared__ __align__(16) float sBo[C];
  __shared__ __align__(16) float sAvg[16 * K];
  const int b0 = blockIdx.x * 16;
  for (int t = threadIdx.x; t < K * (C / 8); t += WARPS * 32) {
    int k = t / (C / 8), c = (t % (C / 8)) * 8;
    ptx_st_shared_v4_u32(Ws + k * LDW + c, ptx_ldg_v4_u32(wo + k * C + c));
  }
  if (threadIdx.x < C / 4) ptx_st_shared_v4_f32(sBo + threadIdx.x * 4, ptx_ldg_v4_f32(bo + threadIdx.x * 4));
  for (int t = threadIdx.x; t < 16 * (K / 4); t += WARPS * 32) {
    int tb = t / (K / 4), tc = (t % (K / 4)) * 4;
    float4 fa = (b0 + tb < L) ? ptx_ldg_v4_f32(avg + (size_t)(b0 + tb) * K + tc) : make_float4(0.f, 0.f, 0.f, 0.f);
    ptx_st_shared_v4_f32(sAvg + tb * K + tc, fa);
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int tok = lane >> 1, hf = lane & 1, b = b0 + tok;
  const int s0 = blockIdx.y * rowsPerBlock, s1 = min(N, s0 + rowsPerBlock);
  half* xs = Xs[warp];
  const float* avp = sAvg + tok * K + hf * 32;
  const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
  uint4 gn[4];
  if (s0 + warp < s1 && b < L) {
#pragma unroll
    for (int i = 0; i < 4; ++i) gn[i] = ptx_ldg_v4_u32(gate + ((size_t)(s0 + warp) * L + b) * K + hf * 32 + 8 * i);
  }
  for (int s = s0 + warp; s < s1; s += WARPS) {
    // this lane's token's 32 gate columns times its column's average, f16 into the warp's tile
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint4 o4 = make_uint4(0, 0, 0, 0);
      if (b < L) {
        const uint4 g4 = gn[i];
        const float4 af0 = ptx_ld_shared_v4_f32(avp + 8 * i), af1 = ptx_ld_shared_v4_f32(avp + 8 * i + 4);
        const float2 g0 = ptx_f16x2_to_f32x2(g4.x), g1 = ptx_f16x2_to_f32x2(g4.y);
        const float2 g2 = ptx_f16x2_to_f32x2(g4.z), g3 = ptx_f16x2_to_f32x2(g4.w);
        o4.x = ptx_cvt_f16x2_f32(ptx_mul_f32(g0.x, af0.x), ptx_mul_f32(g0.y, af0.y));
        o4.y = ptx_cvt_f16x2_f32(ptx_mul_f32(g1.x, af0.z), ptx_mul_f32(g1.y, af0.w));
        o4.z = ptx_cvt_f16x2_f32(ptx_mul_f32(g2.x, af1.x), ptx_mul_f32(g2.y, af1.y));
        o4.w = ptx_cvt_f16x2_f32(ptx_mul_f32(g3.x, af1.z), ptx_mul_f32(g3.y, af1.w));
      }
      ptx_st_shared_v4_u32(xs + tok * LDX + hf * 32 + 8 * i, o4);
    }
    __syncwarp();
    uint32_t xa[K / 16][4];
#pragma unroll
    for (int ks = 0; ks < K / 16; ++ks) ldsm4(xa[ks], xs + (lane & 15) * LDX + ks * 16 + (lane >> 4) * 8);
    __syncwarp();
    if (s + WARPS < s1 && b < L) {
#pragma unroll
      for (int i = 0; i < 4; ++i) gn[i] = ptx_ldg_v4_u32(gate + ((size_t)(s + WARPS) * L + b) * K + hf * 32 + 8 * i);
    }
    const int bt0 = b0 + g, bt1 = b0 + g + 8;
    float* row0 = msa + ((size_t)s * L + bt0) * C;
    float* row1 = msa + ((size_t)s * L + bt1) * C;
    float2 r0[C / 8], r1[C / 8];
    if (bt0 < L) {
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt) r0[nt] = ptx_ld_global_v2_f32(row0 + nt * 8 + tig * 2);
    }
    float acc[C / 8][4] = {};
    uint32_t f[2][4];
    ldsm4t(f[0], Ws + kl * LDW + cl);
#pragma unroll
    for (int ks = 0; ks < K / 16; ++ks) {
      if (ks == 2 && bt1 < L) {
#pragma unroll
        for (int nt = 0; nt < C / 8; ++nt) r1[nt] = ptx_ld_global_v2_f32(row1 + nt * 8 + tig * 2);
      }
#pragma unroll
      for (int n2 = 0; n2 < C / 16; ++n2) {
        const int step = ks * (C / 16) + n2, cur = step & 1;
        if (n2 + 1 < C / 16) ldsm4t(f[cur ^ 1], Ws + (ks * 16 + kl) * LDW + (n2 + 1) * 16 + cl);
        else if (ks + 1 < K / 16) ldsm4t(f[cur ^ 1], Ws + ((ks + 1) * 16 + kl) * LDW + cl);
        mma16816(acc[2 * n2], xa[ks], f[cur][0], f[cur][1]); mma16816(acc[2 * n2 + 1], xa[ks], f[cur][2], f[cur][3]);
      }
    }
    float2 b2All[C / 8];
#pragma unroll
    for (int nt = 0; nt < C / 8; ++nt) b2All[nt] = ptx_ld_shared_v2_f32(sBo + nt * 8 + tig * 2);
    if (bt0 < L) {
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt) {
        const int col = nt * 8 + tig * 2;
        const float2 b2 = b2All[nt];
        r0[nt].x = ptx_add_f32(r0[nt].x, ptx_add_f32(acc[nt][0], b2.x));
        r0[nt].y = ptx_add_f32(r0[nt].y, ptx_add_f32(acc[nt][1], b2.y));
        ptx_st_global_v2_f32(row0 + col, r0[nt]);
      }
    }
    if (bt1 < L) {
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt) {
        const int col = nt * 8 + tig * 2;
        const float2 b2 = b2All[nt];
        r1[nt].x = ptx_add_f32(r1[nt].x, ptx_add_f32(acc[nt][2], b2.x));
        r1[nt].y = ptx_add_f32(r1[nt].y, ptx_add_f32(acc[nt][3], b2.y));
        ptx_st_global_v2_f32(row1 + col, r1[nt]);
      }
    }
  }
}

// the whole of it: msa [N][L][64] += MSAColumnGlobalAttention(msa); A the module's weight prefix
inline void globalAttentionFused(float* msa, const float* msaMask, int L, int N, const std::string& A, int blk) {
  constexpr int WARPS = 4, ROWS = 64, ROWS_OUT = 32;
  float* kv = scratch<float>("gaf.kv", (size_t)L * N * 2 * ga::D);
  half* gate = scratch<half>("gaf.gate", (size_t)N * L * ga::H * ga::D);
  const int Y = (N + ROWS - 1) / ROWS, YOut = (N + ROWS_OUT - 1) / ROWS_OUT;
  float* qsum = scratch<float>("gaf.qsum", ((size_t)L * ga::C + L) * Y);
  float* msum = qsum + (size_t)L * ga::C;
  float* q = scratch<float>("gaf.q", (size_t)L * ga::H * ga::D);
  float* avg = scratch<float>("gaf.avg", (size_t)L * ga::H * ga::D);
  dim3 grid((L + 15) / 16, Y), gridOut((L + 15) / 16, YOut);
  gaInK<WARPS><<<grid, WARPS * 32, 0, STREAM>>>(msa, msaMask, P(A + "/query_norm/scale", blk), P(A + "/query_norm/offset", blk),
      PH(A + "/attention/key_w", blk), PH(A + "/attention/value_w", blk), PH(A + "/attention/gating_w", blk),
      P(A + "/attention/gating_b", blk), kv, gate, qsum, msum, L, N, ROWS);
  gaQueryK<<<L, ga::C, 0, STREAM>>>(qsum, L, Y, P(A + "/attention/query_w", blk), q);
  gaAttnK<<<L, 256, 0, STREAM>>>(kv, msaMask, q, avg, L, N);
  gaOutK<WARPS><<<gridOut, WARPS * 32, 0, STREAM>>>(gate, avg, PH(A + "/attention/output_w", blk), P(A + "/attention/output_b", blk),
                                                     msa, L, N, ROWS_OUT);
}

// The extra stack's row attention, its two projections fused the same way (C = 64, 8 heads of 8, unpadded):
//   lnQkvg64:   LN(msa row) and its q/k/v/gate projection in one pass - cuda/af3's gridInK at 64 channels and 256
//               columns, the gate's bias in its epilogue - where a LayerNorm pass and cuBLAS's GEMM wrote and read
//               the LN'd rows (0.38 + 0.81 ms a call at 261 residues)
//   projAdd64K: y[r] += x[r] Wo + bo, the attention's output projection added into the f32 MSA in one pass, where
//               cuBLASLt's f32-output GEMM ran a GEMM and then its epilogue over the whole MSA (0.62 + 0.93 ms)
inline void lnQkvg64(const float* x, const float* lnS, const float* lnO, const half* W, const float* qb, half* out,
                     size_t q0, size_t rows) {
  constexpr int C = 64, NQ = 256, WARPS = 4, NC = 64, MT = 2, R = 16 * WARPS * MT;
  const size_t smem = gridInSmem(WARPS, NC, C);
  static LF_TLS bool attr = false;
  if (!attr) { smemAttr((gridInK<C, NQ, WARPS, float, NC, MT, 3>), (int)smem); attr = true; }
  gridInK<C, NQ, WARPS, float, NC, MT, 3><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
      x, lnS, lnO, W, out, 1, q0, rows, false, nullptr, nullptr, 0, 0, false, qb);
}
template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32, 4) projAdd64K(const half* __restrict__ x, const half* __restrict__ wo,
    const float* __restrict__ bo, float* __restrict__ y, size_t rows) {
  constexpr int K = 64, C = 64, LDW = C + 8, LDX = K + 8;
  __shared__ __align__(16) half Ws[K * LDW];
  __shared__ __align__(16) half Xs[WARPS][16 * LDX];
  __shared__ __align__(16) float sBo[C];
  for (int t = threadIdx.x; t < K * (C / 8); t += WARPS * 32) {
    int k = t / (C / 8), c = (t % (C / 8)) * 8;
    ptx_st_shared_v4_u32(Ws + k * LDW + c, ptx_ldg_v4_u32(wo + k * C + c));
  }
  if (threadIdx.x < C / 4) ptx_st_shared_v4_f32(sBo + threadIdx.x * 4, ptx_ldg_v4_f32(bo + threadIdx.x * 4));
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, tok = lane >> 1, hf = lane & 1;
  const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
  half* xs = Xs[warp];
  size_t tile0 = (size_t)blockIdx.x * WARPS + warp, stepT = (size_t)gridDim.x * WARPS;
  uint4 xn[4];
  if (tile0 * 16 < rows) {
    const size_t r = tile0 * 16 + tok;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      xn[i] = r < rows ? ptx_ldg_v4_u32(x + r * K + hf * 32 + 8 * i) : make_uint4(0, 0, 0, 0);
  }
  for (size_t tile = tile0; tile * 16 < rows; tile += stepT) {
    const size_t r0 = tile * 16;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      ptx_st_shared_v4_u32(xs + tok * LDX + hf * 32 + 8 * i, xn[i]);
    __syncwarp();
    uint32_t xa[K / 16][4];
#pragma unroll
    for (int ks = 0; ks < K / 16; ++ks) ldsm4(xa[ks], xs + (lane & 15) * LDX + ks * 16 + (lane >> 4) * 8);
    __syncwarp();
    if ((tile + stepT) * 16 < rows) {
      const size_t rn = (tile + stepT) * 16 + tok;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        xn[i] = rn < rows ? ptx_ldg_v4_u32(x + rn * K + hf * 32 + 8 * i) : make_uint4(0, 0, 0, 0);
    }
    const size_t rr0 = r0 + g, rr1 = r0 + g + 8;
    float* row0 = y + rr0 * C;
    float* row1 = y + rr1 * C;
    float2 v0[C / 8], v1[C / 8];
    if (rr0 < rows) {
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt) v0[nt] = ptx_ld_global_v2_f32(row0 + nt * 8 + tig * 2);
    }
    float acc[C / 8][4] = {};
    uint32_t f[2][4];
    ldsm4t(f[0], Ws + kl * LDW + cl);
#pragma unroll
    for (int ks = 0; ks < K / 16; ++ks) {
      if (ks == 2 && rr1 < rows) {
#pragma unroll
        for (int nt = 0; nt < C / 8; ++nt) v1[nt] = ptx_ld_global_v2_f32(row1 + nt * 8 + tig * 2);
      }
#pragma unroll
      for (int n2 = 0; n2 < C / 16; ++n2) {
        const int step = ks * (C / 16) + n2, cur = step & 1;
        if (n2 + 1 < C / 16) ldsm4t(f[cur ^ 1], Ws + (ks * 16 + kl) * LDW + (n2 + 1) * 16 + cl);
        else if (ks + 1 < K / 16) ldsm4t(f[cur ^ 1], Ws + ((ks + 1) * 16 + kl) * LDW + cl);
        mma16816(acc[2 * n2], xa[ks], f[cur][0], f[cur][1]); mma16816(acc[2 * n2 + 1], xa[ks], f[cur][2], f[cur][3]);
      }
    }
    float2 b2All[C / 8];
#pragma unroll
    for (int nt = 0; nt < C / 8; ++nt) b2All[nt] = ptx_ld_shared_v2_f32(sBo + nt * 8 + tig * 2);
    if (rr0 < rows) {
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt) {
        const int col = nt * 8 + tig * 2;
        const float2 b2 = b2All[nt];
        v0[nt].x = ptx_add_f32(v0[nt].x, ptx_add_f32(acc[nt][0], b2.x));
        v0[nt].y = ptx_add_f32(v0[nt].y, ptx_add_f32(acc[nt][1], b2.y));
        ptx_st_global_v2_f32(row0 + col, v0[nt]);
      }
    }
    if (rr1 < rows) {
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt) {
        const int col = nt * 8 + tig * 2;
        const float2 b2 = b2All[nt];
        v1[nt].x = ptx_add_f32(v1[nt].x, ptx_add_f32(acc[nt][2], b2.x));
        v1[nt].y = ptx_add_f32(v1[nt].y, ptx_add_f32(acc[nt][3], b2.y));
        ptx_st_global_v2_f32(row1 + col, v1[nt]);
      }
    }
  }
}
inline void projAdd64(const half* x, const half* wo, const float* bo, float* y, size_t rows) {
  constexpr int WARPS = 4;
  const size_t tiles = (rows + 15) / 16;
  static const int sms = [] { int dev, n; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev)); return n; }();
  const unsigned grid = (unsigned)std::min<size_t>((tiles + WARPS - 1) / WARPS, (size_t)sms * 16);
  projAdd64K<WARPS><<<grid, WARPS * 32, 0, STREAM>>>(x, wo, bo, y, rows);
}
