#pragma once
// One fold across several GPUs, phase 2: the pair updates on a SHARDED pair - rank r holds pair rows [lo_r, hi_r) only
// (its slab, n_r x n x C), shares of the 16-padded length so every slab boundary suits the FP8 GEMM's leading
// dimensions. A kernel that touches only its own rows runs unchanged on a slab through a shifted base pointer
// (base = slab - lo * n * C): they index the pair by global row.
//   triangle outgoing - each rank builds the rows of b it owns, all-gathered into the whole b the blocked form keeps
//     on one GPU too, then this rank's blocks of a against it (triangleBlockedTN's outgoing branch on a slab);
//   triangle incoming - the same on z^T: p[i][j] = sum_k a(z[k][j]) b(z[k][i]) = sum_k a(zT[j][k]) b(zT[i][k]) is the
//     outgoing form on z^T with the incoming weights, its output at zT[j][i] = z[i][j] (the gate read there too);
//   row attention - the bias (every row's, every rank needs it) from this rank's rows then all-gathered; the rows in
//     chunks; column attention is row attention on z^T with the bias flipped (swap -> !swap);
//   the transition - local.
// z^T comes from an all-to-all transpose (transpose()), four a block. multigpu.cuh has the transport.
#include "pairtrack.cuh"

namespace sh {
inline bool ON = getenv("LOCALFOLD_MG_SHARDED") && atoi(getenv("LOCALFOLD_MG_SHARDED"));
inline int padded(int n) { return (n + 15) / 16 * 16; }
// rank r's rows of the pair: its share of the padded length, its stored (real) rows clipped to n
inline void rowsOf(int n, int r, int& lo, int& hi) { mg::shareOf(padded(n), 16, r, lo, hi); }
inline int storedRows(int n, int r) { int lo, hi; rowsOf(n, r, lo, hi); return std::max(0, std::min(hi, n) - lo); }
inline int maxStored(int n) { int m = 0; for (int r = 0; r < mg::WORLD; ++r) m = std::max(m, storedRows(n, r)); return m; }
inline size_t elem() { return PAIR16 ? 2 : 4; }
inline float* baseOf(void* slab, int lo, int n, int C) { return (float*)((char*)slab - (size_t)lo * n * C * elem()); }

// dst[(j) * n + i0 + i] = src[i * cols + j]: a staged block of the pair (rows x cols pairs, `bytes` a pair), transposed
__global__ void transposeBlockK(const uint4* __restrict__ src, uint4* __restrict__ dst, int rows, int cols, int n, int i0,
                                int vecs) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)rows * cols * vecs) return;
  size_t p = t / vecs; int v = (int)(t % vecs);
  int i = (int)(p / cols), j = (int)(p % cols);
  dst[((size_t)j * n + i0 + i) * vecs + v] = src[p * vecs + v];
}
// zT (this rank's rows of z^T: z's columns [lo, hi)) from every rank's slab of z
inline void transpose(const mg::Shared& z, const mg::Shared& zT, int n, int C) {
  const size_t pb = (size_t)C * elem(); const int vecs = (int)(pb / 16);
  int lo, hi; rowsOf(n, mg::RANK, lo, hi); const int cols = storedRows(n, mg::RANK);
  char* stage = scratch<char>("sh.tstage", std::max<size_t>(1, (size_t)maxStored(n) * cols * pb));
  mg::fence();
  for (int p = 0; p < mg::WORLD; ++p) {
    int plo, phi; rowsOf(n, p, plo, phi); const int rows = storedRows(n, p);
    if (!rows || !cols) continue;
    // rank p's rows, this rank's columns: rows x cols pairs, its slab's rows from plo
    CK(cudaMemcpy2DAsync(stage, (size_t)cols * pb, (const char*)z.peer[p] + (size_t)lo * pb, (size_t)n * pb,
                         (size_t)cols * pb, rows, cudaMemcpyDefault, STREAM));
    transposeBlockK<<<blocks((size_t)rows * cols * vecs), 256, 0, STREAM>>>((const uint4*)stage, (uint4*)zT.local, rows,
                                                                        cols, n, plo, vecs);
  }
  mg::fence();
}

// the triangle multiplication's outgoing form on a slab (z, or z^T with the incoming weights)
template <class TQ>
inline void triangleOn(const mg::Shared& z, const float* mask, int n, int C, const std::string& pre, bool divideByLength) {
  using B16 = __nv_bfloat16;
  constexpr bool F8 = sizeof(TQ) == 1;
  const int np = padded(n); const size_t cs = (size_t)np * np;
  const float alpha = divideByLength ? 1.f / n : 1.f;
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  float* base = baseOf(z.local, lo, n, C);
  std::string pg = projectionGate(pre, C);
  half* wt = scratch<half>("trib.wt", triInTileHalves(C));
  tileTriIn(Wh(pg), Wh(pre + ".gatingLinear"), C, 16, wt);
  // the rows of b this rank owns ([C][rows][np], shared), then every rank's into the whole b ([C][np][np])
  int maxShare = 0; for (int r = 0; r < mg::WORLD; ++r) { int a0, a1; rowsOf(n, r, a0, a1); maxShare = std::max(maxShare, a1 - a0); }
  mg::Shared& bmine = mg::shared("sh.bmine", (size_t)maxShare * np * C * sizeof(TQ));
  TQ* b = scratch<TQ>("trib.bq", cs * C);
  int width = (int)std::max<size_t>(16, std::min<size_t>(np, (CHUNK / C) / np / 16 * 16));
  {
    size_t f, t; deviceMemInfo(&f, &t);
    size_t perRow = (size_t)np * C * (sizeof(TQ) + 4), spare = f > t / 16 ? f - t / 16 : 0;
    width = (int)std::max<size_t>(width, std::min<size_t>(np, std::min<size_t>(spare / perRow, 1024) / 16 * 16));
  }
  width = std::min(width, std::max(16, hi - lo));
  TQ* a = scratch<TQ>("trib.aq", (size_t)width * np * C);
  B16* prod = scratch<B16>("trib.pbf", (size_t)width * np * C);
  half* t2 = scratch<half>("trib.t2", (size_t)width * np * C);
  const float* lnS = W(pre + ".leftNormInputScale"); const float* lnO = W(pre + ".leftNormInputOffset");
  auto gemm = [&](int m, int nn, const TQ* A, int lda, long long sA, const TQ* B, long long sB, int ldc, long long sC) {
    if constexpr (F8) fp8GemmTN(m, nn, np, alpha, A, lda, sA, B, np, sB, prod, ldc, sC, C);
    else bf16Gemms(CUBLAS_OP_T, CUBLAS_OP_N, m, nn, np, alpha, A, lda, sA, B, np, sB, prod, ldc, sC, C);
  };
  wideWidth(C, [&](auto cw) {
    constexpr int CC = decltype(cw)::value, WO = 4;
    wideWarps(C, [&](auto warps) {
      constexpr int WI = decltype(warps)::value;
      WITH_PAIR_T(
        static bool attr = false;
        constexpr auto kern = triIn256For<CC, WI, TQ, PT>();
        if (!attr) { smemAttr(kern, (int)wideTriInSmemW(CC, WI)); attr = true; }
        auto in = [&](RectMap rm, size_t rows, TQ* ao, TQ* bo, half* t2o) {
          kern<<<(unsigned)((rows + triInRowsOf(WI) - 1) / triInRowsOf(WI)), 32 * triInWarpsOf(WI), wideTriInSmemW(CC, WI), STREAM>>>(
            base, mask, lnS, lnO, wt, ao, bo, t2o, n, np, cs, nullptr, rm);
        };
        if (hi > lo) in(RectMap{lo, np, 0, (size_t)(hi - lo) * np}, (size_t)(hi - lo) * np, nullptr, (TQ*)bmine.local, nullptr);
        mg::fence();
        for (int r = 0; r < mg::WORLD; ++r) {       // rank r's rows of every channel's plane
          int rlo, rhi; rowsOf(n, r, rlo, rhi);
          if (rhi <= rlo) continue;
          const size_t rb = (size_t)(rhi - rlo) * np * sizeof(TQ);
          CK(cudaMemcpy2DAsync((char*)b + (size_t)rlo * np * sizeof(TQ), cs * sizeof(TQ), bmine.peer[r], rb, rb, C,
                               cudaMemcpyDefault, STREAM));
        }
        mg::fence();
        for (int k0 = lo; k0 < std::min(hi, n); k0 += width) {
          int w = std::min(width, hi - k0);
          const size_t rs = (size_t)w * np;
          RectMap rm{k0, np, 0, rs};
          in(rm, rs, a, nullptr, t2);
          gemm(np, w, b, np, (long long)cs, a, (long long)rs, np, (long long)rs);
          triangleOutRun<CC, WO, B16>(prod, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"),
                                      Wh(pre + ".outputProjection"), t2, into(base), n, np, nullptr, rm);
        }
      );
    });
  });
  releaseScratch({ "trib." });
}
template <class T>
inline void triangleSharded(const mg::Shared& z, const float* mask, int n, int C, const std::string& pre, bool divide) {
  if (C > 256) { fprintf(stderr, "sharded pair: the triangle takes 128 or 256 channels, not %d\n", C); exit(1); }
  if (fp8Tensor()) triangleOn<__nv_fp8_e4m3>(z, mask, n, C, pre, divide);
  else triangleOn<__nv_bfloat16>(z, mask, n, C, pre, divide);
}

// row attention on a slab (z; or z^T, with the bias's swap flipped - column attention)
inline void rowAttention(const mg::Shared& z, const float* mask, int n, int C, int heads, int D, const std::string& pre,
                         bool swap) {
  const int Wd = heads * D;
  if (!(C == 128 && Wd == 128 && heads <= 16 && D == 32) || hasW(pre + ".gatingQueryBias") || hasW(pre + ".outputProjectionBias")) {
    fprintf(stderr, "sharded pair: the grid attention's fused 128-channel form only (%s)\n", pre.c_str()); exit(1);
  }
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  const int rowsHere = storedRows(n, mg::RANK);
  float* base = baseOf(z.local, lo, n, C);
  const int stride = (n + 7) / 8 * 8;
  mg::Shared& biasS = mg::shared("sh.bias", (size_t)heads * n * stride * 2);
  half* bias = (half*)biasS.local;
  std::string qkvg = qkvgWeight(pre, C, Wd, true);
  std::string wb = paddedColumns(pre + ".pairBiasProjection", C, heads, 16);
  const float scale = 1.f / sqrtf((float)D);
  // this rank's rows' bias, in chunks of pairs
  CK(cudaMemsetAsync(bias, 0, (size_t)heads * n * stride * 2, STREAM));
  const size_t pairsHere = (size_t)rowsHere * n;
  size_t per = std::max<size_t>(1, std::min(pairsHere, CHUNK / 16));
  float* raw = scratch<float>("grid.raw16", per * 16);
  for (size_t q0 = 0; q0 < pairsHere; q0 += per) {
    size_t r = std::min(per, pairsHere - q0);
    lnHeads128<16>(pairRow((float*)z.local, q0, C), pre + ".actNormScale", pre + ".actNormOffset", wb, raw, r);
    biasFromRawRowsK<half><<<blocks((size_t)heads * r), 256, 0, STREAM>>>(raw, bias, (size_t)lo * n + q0, r, n, stride,
                                                                        heads, swap, LOG2E);
  }
  // every rank's: rows of each head's plane, or columns when swapped
  mg::fence();
  for (int r = 0; r < mg::WORLD; ++r) {
    if (r == mg::RANK) continue;
    int rlo, rhi; rowsOf(n, r, rlo, rhi); rhi = std::min(rhi, n);
    if (rhi <= rlo) continue;
    for (int h = 0; h < heads; ++h) {
      const size_t plane = (size_t)h * n * stride * 2;
      if (!swap) CK(cudaMemcpyAsync((char*)bias + plane + (size_t)rlo * stride * 2, (const char*)biasS.peer[r] + plane +
                                    (size_t)rlo * stride * 2, (size_t)(rhi - rlo) * stride * 2, cudaMemcpyDefault, STREAM));
      else CK(cudaMemcpy2DAsync((char*)bias + plane + (size_t)rlo * 2, (size_t)stride * 2, (const char*)biasS.peer[r] + plane +
                                (size_t)rlo * 2, (size_t)stride * 2, (size_t)(rhi - rlo) * 2, n, cudaMemcpyDefault, STREAM));
    }
  }
  mg::fence();
  // this rank's rows, in chunks
  size_t R = std::max<size_t>(1, std::min<size_t>(std::max(rowsHere, 1), CHUNK / ((size_t)n * 4 * Wd)));
  {
    size_t f, t; deviceMemInfo(&f, &t);
    size_t perRow = (size_t)n * 5 * Wd * 2, spare = f > t / 16 ? f - t / 16 : 0;
    R = std::max<size_t>(R, std::min<size_t>(rowsHere, std::min<size_t>(spare / perRow, 256)));
  }
  const bool f8 = (MASK_ALL_ONES || !mask) && fp8Attn();
  for (size_t r0 = lo; r0 < (size_t)(lo + rowsHere); r0 += R) {
    size_t rows = std::min(R, (size_t)(lo + rowsHere) - r0), prs = rows * n;
    half* qkvgOut = scratch<half>("grid.qkvg", (std::min<size_t>(R, rowsHere) * n + 128) * 4 * Wd);
    uint8_t* kv8 = f8 ? scratch<uint8_t>("grid.kv8", std::min<size_t>(R, rowsHere) * n * 2 * Wd) : nullptr;
    gridIn128(base, pre, qkvg, qkvgOut, n, r0 * n, prs, false, nullptr, nullptr, 0, 0, false, kv8);
    half* gathered = scratch<half>("grid.gathered", std::min<size_t>(R, rowsHere) * n * Wd);
    if (f8) flash8Run(qkvgOut, kv8, kv8 + prs * Wd, bias, stride, gathered, n, heads, rows, scale, nullptr);
    else flashGrid<half>(qkvgOut, bias, stride, MASK_ALL_ONES ? nullptr : mask, gathered, n, heads, D, r0, rows, false, scale);
    gridOut128(gathered, pre + ".outputProjection", into(base), n, r0 * n, prs, false);
  }
  releaseScratch({ "grid.qkvg", "grid.gathered", "grid.kv8", "attn.vt8" });
}

// A pairformer block's pair updates on the sharded pair z (this rank's slab), zT a slab-sized buffer for z^T
template <class T>
inline void pairUpdates(const mg::Shared& z, const mg::Shared& zT, const float* mask, int n, int C, const std::string& pre,
                        bool swap, bool divide, int transitionFactor) {
  if (!PAIR16) { fprintf(stderr, "sharded pair: the bf16 pair only\n"); exit(1); }
  triangleSharded<T>(z, mask, n, C, pre + ".triangleMultiplicationOutgoing", divide); stage("tri.out");
  transpose(z, zT, n, C);
  triangleSharded<T>(zT, mask, n, C, pre + ".triangleMultiplicationIncoming", divide); stage("tri.in");
  transpose(zT, z, n, C);
  int heads = (int)M.meta(pre + ".pairAttention1.heads"), D = (int)M.meta(pre + ".pairAttention1.dimension");
  rowAttention(z, mask, n, C, heads, D, pre + ".pairAttention1", false); stage("grid.row");
  transpose(z, zT, n, C);
  rowAttention(zT, mask, n, C, heads, D, pre + ".pairAttention2", !swap); stage("grid.col");
  transpose(zT, z, n, C);
  const int rowsHere = storedRows(n, mg::RANK);
  if (rowsHere) transition<T>((float*)z.local, (size_t)rowsHere * n, C, transitionFactor, pre + ".pairTransition");
  stage("transition");
}

// phase 2's check while the rest of the trunk still holds the whole pair: this rank's rows cut from the replicated
// pair into its slab, `work(z, zT, lo)` on the slab, and every rank's rows gathered back (mg::exchange)
template <class F>
inline void viaShards(float* pair, int n, int C, F work) {
  const size_t slabBytes = (size_t)maxStored(n) * n * C * elem();
  mg::Shared& z = mg::shared("sh.z", slabBytes);
  mg::Shared& zT = mg::shared("sh.zT", slabBytes);
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  const size_t row = (size_t)n * C * elem();
  const int rowsHere = storedRows(n, mg::RANK);
  if (rowsHere) CK(cudaMemcpyAsync(z.local, (char*)pair + (size_t)lo * row, (size_t)rowsHere * row, cudaMemcpyDefault, STREAM));
  work(z, zT, lo);
  if (rowsHere) CK(cudaMemcpyAsync((char*)pair + (size_t)lo * row, z.local, (size_t)rowsHere * row, cudaMemcpyDefault, STREAM));
  mg::exchange(false, n, C, elem(), padded(n), 16);
}
}  // namespace sh
