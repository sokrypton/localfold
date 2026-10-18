// AF2's MSAColumnGlobalAttention on the extra MSA, fused: four kernels on the MSA in its own [N][L][C] layout,
// where the unfused form transposed it to [L][N][C] (and back), LayerNormed, projected and gated in f32 passes
// of their own - eleven passes over a 5,120-row extra MSA, 7.9 ms a call at 261 residues on an A100, 8% of a
// fold (nvfold's: ~0.7 ms). C = 64, eight heads of eight (the extra stack's shape).
//
//   gaInK:    LN(row) on the way in (four lanes... two a row: 32 channels a lane), then on the tensor cores
//             [k | v | gate] = LN(row) [Wk | Wv | Wg]: k, v to kv [L][N][16] (f32, a column's keys contiguous),
//             sigmoid(gate + bg) to gate [N][L][64] (f16); the masked column sums of LN(row) and the mask's
//             sums into qsum [L][64], msum [L] (per block, then one atomic a column-channel)
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
__global__ void __launch_bounds__(WARPS * 32) gaInK(const float* __restrict__ msa, const float* __restrict__ mask,
    const float* __restrict__ lnS, const float* __restrict__ lnO, const half* __restrict__ wk, const half* __restrict__ wv,
    const half* __restrict__ wg, const float* __restrict__ gb, float* __restrict__ kv, half* __restrict__ gate,
    float* __restrict__ qsum, float* __restrict__ msum, int L, int N, int rowsPerBlock) {
  constexpr int C = ga::C, H = ga::H, D = ga::D, NIN = ga::NIN;
  constexpr int LDW = NIN + 8, LDX = C + 8;
  __shared__ __align__(16) half Ws[C * LDW];
  __shared__ __align__(16) half Xs[WARPS][16 * LDX];
  __shared__ float sS[C], sO[C], sGb[H * D];
  __shared__ float red[WARPS][32][33];
  __shared__ float mred[WARPS][16];
  for (int t = threadIdx.x; t < C * NIN; t += WARPS * 32) {
    int k = t / NIN, c = t % NIN;
    Ws[k * LDW + c] = c < D ? wk[k * D + c] : c < 2 * D ? wv[k * D + c - D] : wg[k * H * D + c - 2 * D];
  }
  for (int t = threadIdx.x; t < C; t += WARPS * 32) { sS[t] = lnS[t]; sO[t] = lnO[t]; sGb[t] = gb[t]; }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int b0 = blockIdx.x * 16, tok = lane >> 1, hf = lane & 1, b = b0 + tok;
  const bool valid = b < L;
  const int s0 = blockIdx.y * rowsPerBlock, s1 = min(N, s0 + rowsPerBlock);
  float qacc[32], macc = 0.f;
#pragma unroll
  for (int c = 0; c < 32; ++c) qacc[c] = 0.f;
  half* xs = Xs[warp];
  for (int s = s0 + warp; s < s1; s += WARPS) {
    float v[32];
    if (valid) {
      const float4* p = reinterpret_cast<const float4*>(msa + ((size_t)s * L + b) * C + hf * 32);
#pragma unroll
      for (int i = 0; i < 8; ++i) { float4 f = p[i]; v[4 * i] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w; }
    } else {
#pragma unroll
      for (int c = 0; c < 32; ++c) v[c] = 0.f;
    }
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < 32; ++c) sum += v[c];
    sum += __shfl_xor_sync(~0u, sum, 1);
    const float mean = sum / C;
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < 32; ++c) { float d = v[c] - mean; q += d * d; }
    q += __shfl_xor_sync(~0u, q, 1);
    const float inv = rsqrtf(q / C + 1e-5f), m = valid ? mask[(size_t)s * L + b] : 0.f;
    macc += m;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint4 o4; half2* oh = reinterpret_cast<half2*>(&o4);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int c = 8 * i + 2 * e, ch = hf * 32 + c;
        const float x0 = (v[c] - mean) * inv * sS[ch] + sO[ch], x1 = (v[c + 1] - mean) * inv * sS[ch + 1] + sO[ch + 1];
        qacc[c] += m * x0; qacc[c + 1] += m * x1;
        oh[e] = __floats2half2_rn(x0, x1);
      }
      *reinterpret_cast<uint4*>(xs + tok * LDX + hf * 32 + 8 * i) = o4;
    }
    __syncwarp();
    uint32_t xa[C / 16][4];
#pragma unroll
    for (int ks = 0; ks < C / 16; ++ks) ldsm4(xa[ks], xs + (lane & 15) * LDX + ks * 16 + (lane >> 4) * 8);
    float acc[NIN / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < C / 16; ++ks)
#pragma unroll
      for (int n2 = 0; n2 < NIN / 16; ++n2) {
        uint32_t f[4];
        ldsm4t(f, Ws + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW + n2 * 16 + (lane >> 4) * 8);
        mma16816(acc[2 * n2], xa[ks], f[0], f[1]); mma16816(acc[2 * n2 + 1], xa[ks], f[2], f[3]);
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
        if (nt < 2) *reinterpret_cast<float2*>(kv + ((size_t)bt * N + s) * 2 * D + col) = make_float2(a0, a1);
        else {
          const int gc = col - 2 * D;
          *reinterpret_cast<half2*>(gate + ((size_t)s * L + bt) * H * D + gc) =
              __floats2half2_rn(1.f / (1.f + __expf(-(a0 + sGb[gc]))), 1.f / (1.f + __expf(-(a1 + sGb[gc + 1]))));
        }
      }
    }
  }
  // the block's masked sums: across its warps in shared memory, then one atomic a column-channel
#pragma unroll
  for (int c = 0; c < 32; ++c) red[warp][lane][c] = qacc[c];
  if (hf == 0) mred[warp][tok] = macc;
  __syncthreads();
  for (int t = threadIdx.x; t < 32 * 32; t += WARPS * 32) {
    const int ln = t / 32, c = t % 32, bb = b0 + (ln >> 1);
    if (bb >= L) continue;
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) a += red[w][ln][c];
    atomicAdd(qsum + (size_t)bb * C + (ln & 1) * 32 + c, a);
  }
  for (int t = threadIdx.x; t < 16; t += WARPS * 32) {
    if (b0 + t >= L) continue;
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) a += mred[w][t];
    atomicAdd(msum + b0 + t, a);
  }
}

// q[b][h*D + d] = (qsum[b] / (msum[b] + 1e-10)) Wq / sqrt(D); Wq [C][H*D] f32. A block (C threads) a column
__global__ void __launch_bounds__(64) gaQueryK(const float* __restrict__ qsum, const float* __restrict__ msum,
    const float* __restrict__ wq, float* __restrict__ q) {
  constexpr int C = ga::C, H = ga::H, D = ga::D, NIN = ga::NIN;
  __shared__ float qa[C];
  const int b = blockIdx.x, t = threadIdx.x;
  qa[t] = qsum[(size_t)b * C + t] / (msum[b] + 1e-10f);
  __syncthreads();
  float a = 0.f;
  for (int c = 0; c < C; ++c) a += qa[c] * wq[(size_t)c * H * D + t];
  q[(size_t)b * H * D + t] = a / sqrtf((float)D);
}

// a block a column: avg[b][h][d] = sum_s softmax_s(q[b][h] . k[b][s] + 1e9 (mask - 1)) v[b][s][d]
__global__ void __launch_bounds__(256) gaAttnK(const float* __restrict__ kv, const float* __restrict__ mask,
    const float* __restrict__ qg, float* __restrict__ avg, int L, int N) {
  constexpr int C = ga::C, H = ga::H, D = ga::D, NIN = ga::NIN;
  __shared__ float qh[H * D];
  __shared__ float part[8][H][D + 2];
  const int b = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (threadIdx.x < H * D) qh[threadIdx.x] = qg[(size_t)b * H * D + threadIdx.x];
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
    const float4 k0 = p[0], k1 = p[1], v0 = p[2], v1 = p[3];
    const float k[D] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w}, v[D] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    const float bias = 1e9f * (mask[(size_t)s * L + b] - 1.f);
#pragma unroll
    for (int h = 0; h < H; ++h) {
      float lg = bias;
#pragma unroll
      for (int d = 0; d < D; ++d) lg += qh[h * D + d] * k[d];
      const float mn = fmaxf(m[h], lg), cs = __expf(m[h] - mn), e = __expf(lg - mn);
      l[h] = l[h] * cs + e;
#pragma unroll
      for (int d = 0; d < D; ++d) acc[h][d] = acc[h][d] * cs + e * v[d];
      m[h] = mn;
    }
  }
  auto merge = [&](int h, float m2, float l2, const float* a2) {
    const float M = fmaxf(m[h], m2);
    if (M == -INFINITY) return;
    const float ca = __expf(m[h] - M), cb = __expf(m2 - M);
    l[h] = l[h] * ca + l2 * cb;
#pragma unroll
    for (int d = 0; d < D; ++d) acc[h][d] = acc[h][d] * ca + a2[d] * cb;
    m[h] = M;
  };
  for (int o = 16; o; o >>= 1) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
      float a2[D];
      const float m2 = __shfl_xor_sync(~0u, m[h], o), l2 = __shfl_xor_sync(~0u, l[h], o);
#pragma unroll
      for (int d = 0; d < D; ++d) a2[d] = __shfl_xor_sync(~0u, acc[h][d], o);
      merge(h, m2, l2, a2);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
      part[warp][h][0] = m[h]; part[warp][h][1] = l[h];
#pragma unroll
      for (int d = 0; d < D; ++d) part[warp][h][2 + d] = acc[h][d];
    }
  }
  __syncthreads();
  if (threadIdx.x < H * D) {
    const int h = threadIdx.x / D, d = threadIdx.x % D;
    float M = -INFINITY;
    for (int w = 0; w < 8; ++w) M = fmaxf(M, part[w][h][0]);
    float Ls = 0.f, A = 0.f;
    for (int w = 0; w < 8; ++w) {
      if (part[w][h][0] == -INFINITY) continue;
      const float c = __expf(part[w][h][0] - M);
      Ls += part[w][h][1] * c; A += part[w][h][2 + d] * c;
    }
    avg[(size_t)b * H * D + threadIdx.x] = A / Ls;
  }
}

// msa[s][b] += (gate[s][b] * avg[b]) Wo + bo; Wo [H*D][C] f16. One block a 16-column group and a run of rows
template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) gaOutK(const half* __restrict__ gate, const float* __restrict__ avg,
    const half* __restrict__ wo, const float* __restrict__ bo, float* __restrict__ msa, int L, int N, int rowsPerBlock) {
  constexpr int C = ga::C, H = ga::H, D = ga::D, NIN = ga::NIN;
  constexpr int K = H * D, LDW = C + 8, LDX = K + 8;
  __shared__ __align__(16) half Ws[K * LDW];
  __shared__ __align__(16) half Xs[WARPS][16 * LDX];
  __shared__ float sA[16][K], sB[C];
  const int b0 = blockIdx.x * 16;
  for (int t = threadIdx.x; t < K * C; t += WARPS * 32) Ws[(t / C) * LDW + t % C] = wo[t];
  for (int t = threadIdx.x; t < 16 * K; t += WARPS * 32) sA[t / K][t % K] = b0 + t / K < L ? avg[(size_t)(b0 + t / K) * K + t % K] : 0.f;
  for (int t = threadIdx.x; t < C; t += WARPS * 32) sB[t] = bo[t];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int tok = lane >> 1, hf = lane & 1, b = b0 + tok;
  const int s0 = blockIdx.y * rowsPerBlock, s1 = min(N, s0 + rowsPerBlock);
  half* xs = Xs[warp];
  for (int s = s0 + warp; s < s1; s += WARPS) {
    // this lane's token's 32 gate columns times its column's average, f16 into the warp's tile
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint4 o4 = make_uint4(0, 0, 0, 0);
      if (b < L) {
        const uint4 g4 = *reinterpret_cast<const uint4*>(gate + ((size_t)s * L + b) * K + hf * 32 + 8 * i);
        const half2* gh = reinterpret_cast<const half2*>(&g4);
        half2* oh = reinterpret_cast<half2*>(&o4);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c = hf * 32 + 8 * i + 2 * e;
          const float2 gf = __half22float2(gh[e]);
          oh[e] = __floats2half2_rn(gf.x * sA[tok][c], gf.y * sA[tok][c + 1]);
        }
      }
      *reinterpret_cast<uint4*>(xs + tok * LDX + hf * 32 + 8 * i) = o4;
    }
    __syncwarp();
    uint32_t xa[K / 16][4];
#pragma unroll
    for (int ks = 0; ks < K / 16; ++ks) ldsm4(xa[ks], xs + (lane & 15) * LDX + ks * 16 + (lane >> 4) * 8);
    float acc[C / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < K / 16; ++ks)
#pragma unroll
      for (int n2 = 0; n2 < C / 16; ++n2) {
        uint32_t f[4];
        ldsm4t(f, Ws + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW + n2 * 16 + (lane >> 4) * 8);
        mma16816(acc[2 * n2], xa[ks], f[0], f[1]); mma16816(acc[2 * n2 + 1], xa[ks], f[2], f[3]);
      }
    __syncwarp();
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int bt = b0 + g + 8 * hh;
      if (bt >= L) continue;
      float* row = msa + ((size_t)s * L + bt) * C;
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt) {
        const int col = nt * 8 + tig * 2;
        float2 r = *reinterpret_cast<float2*>(row + col);
        r.x += acc[nt][2 * hh] + sB[col]; r.y += acc[nt][2 * hh + 1] + sB[col + 1];
        *reinterpret_cast<float2*>(row + col) = r;
      }
    }
  }
}

// the whole of it: msa [N][L][64] += MSAColumnGlobalAttention(msa); A the module's weight prefix
inline void globalAttentionFused(float* msa, const float* msaMask, int L, int N, const std::string& A, int blk) {
  constexpr int WARPS = 4, ROWS = 128;
  float* kv = scratch<float>("gaf.kv", (size_t)L * N * 2 * ga::D);
  half* gate = scratch<half>("gaf.gate", (size_t)N * L * ga::H * ga::D);
  float* qsum = scratch<float>("gaf.qsum", (size_t)L * ga::C + L);
  float* msum = qsum + (size_t)L * ga::C;
  float* q = scratch<float>("gaf.q", (size_t)L * ga::H * ga::D);
  float* avg = scratch<float>("gaf.avg", (size_t)L * ga::H * ga::D);
  CK(cudaMemsetAsync(qsum, 0, ((size_t)L * ga::C + L) * 4, STREAM));
  dim3 grid((L + 15) / 16, (N + ROWS - 1) / ROWS);
  gaInK<WARPS><<<grid, WARPS * 32, 0, STREAM>>>(msa, msaMask, P(A + "/query_norm/scale", blk), P(A + "/query_norm/offset", blk),
      PH(A + "/attention/key_w", blk), PH(A + "/attention/value_w", blk), PH(A + "/attention/gating_w", blk),
      P(A + "/attention/gating_b", blk), kv, gate, qsum, msum, L, N, ROWS);
  gaQueryK<<<L, ga::C, 0, STREAM>>>(qsum, msum, P(A + "/attention/query_w", blk), q);
  gaAttnK<<<L, 256, 0, STREAM>>>(kv, msaMask, q, avg, L, N);
  gaOutK<WARPS><<<grid, WARPS * 32, 0, STREAM>>>(gate, avg, PH(A + "/attention/output_w", blk), P(A + "/attention/output_b", blk),
                                                 msa, L, N, ROWS);
}

// The extra stack's row attention, its two projections fused the same way (C = 64, 8 heads of 8, unpadded):
//   lnQkvg64:   LN(msa row) and its q/k/v/gate projection in one pass - cuda/af3's gridInK at 64 channels and 256
//               columns, the gate's bias in its epilogue - where a LayerNorm pass and cuBLAS's GEMM wrote and read
//               the LN'd rows (0.38 + 0.81 ms a call at 261 residues)
//   projAdd64K: y[r] += x[r] Wo + bo, the attention's output projection added into the f32 MSA in one pass, where
//               cuBLASLt's f32-output GEMM ran a GEMM and then its epilogue over the whole MSA (0.62 + 0.93 ms)
inline void lnQkvg64(const float* x, const float* lnS, const float* lnO, const half* W, const float* qb, half* out,
                     size_t q0, size_t rows) {
  constexpr int C = 64, NQ = 256, WARPS = 8, NC = 64, R = 16 * WARPS;
  const size_t smem = (size_t)R * (C + 8) * 2 + (size_t)2 * C * (NC + 8) * 2 + (size_t)C * 24 * 2;
  static bool attr = false;
  if (!attr) { smemAttr((gridInK<C, NQ, WARPS, float, NC>), (int)smem); attr = true; }
  gridInK<C, NQ, WARPS, float, NC><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
      x, lnS, lnO, W, out, 1, q0, rows, false, nullptr, nullptr, 0, 0, false, qb);
}
template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) projAdd64K(const half* __restrict__ x, const half* __restrict__ wo,
    const float* __restrict__ bo, float* __restrict__ y, size_t rows) {
  constexpr int K = 64, C = 64, LDW = C + 8, LDX = K + 8;
  __shared__ __align__(16) half Ws[K * LDW];
  __shared__ __align__(16) half Xs[WARPS][16 * LDX];
  __shared__ float sB[C];
  for (int t = threadIdx.x; t < K * C; t += WARPS * 32) Ws[(t / C) * LDW + t % C] = wo[t];
  for (int t = threadIdx.x; t < C; t += WARPS * 32) sB[t] = bo[t];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, tok = lane >> 1, hf = lane & 1;
  half* xs = Xs[warp];
  for (size_t tile = (size_t)blockIdx.x * WARPS + warp; tile * 16 < rows; tile += (size_t)gridDim.x * WARPS) {
    const size_t r0 = tile * 16, r = r0 + tok;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      *reinterpret_cast<uint4*>(xs + tok * LDX + hf * 32 + 8 * i) =
          r < rows ? *reinterpret_cast<const uint4*>(x + r * K + hf * 32 + 8 * i) : make_uint4(0, 0, 0, 0);
    __syncwarp();
    uint32_t xa[K / 16][4];
#pragma unroll
    for (int ks = 0; ks < K / 16; ++ks) ldsm4(xa[ks], xs + (lane & 15) * LDX + ks * 16 + (lane >> 4) * 8);
    float acc[C / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < K / 16; ++ks)
#pragma unroll
      for (int n2 = 0; n2 < C / 16; ++n2) {
        uint32_t f[4];
        ldsm4t(f, Ws + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW + n2 * 16 + (lane >> 4) * 8);
        mma16816(acc[2 * n2], xa[ks], f[0], f[1]); mma16816(acc[2 * n2 + 1], xa[ks], f[2], f[3]);
      }
    __syncwarp();
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const size_t rr = r0 + g + 8 * hh;
      if (rr >= rows) continue;
      float* row = y + rr * C;
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt) {
        const int col = nt * 8 + tig * 2;
        float2 v = *reinterpret_cast<float2*>(row + col);
        v.x += acc[nt][2 * hh] + sB[col]; v.y += acc[nt][2 * hh + 1] + sB[col + 1];
        *reinterpret_cast<float2*>(row + col) = v;
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
