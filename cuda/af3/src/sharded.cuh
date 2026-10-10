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
  mg::Shared& bmine = mg::shared("sh.bmine" + std::to_string(C), (size_t)maxShare * np * C * sizeof(TQ));
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
  mg::Shared& biasS = mg::shared("sh.bias" + std::to_string(heads), (size_t)heads * n * stride * 2);
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


// ---- the generic forms (any width, an f32 or bf16 pair: the template stack's 64 channels, rf3's biased attention,
// the 256-channel models), after triangleBlocked's and gridAttention's own generic paths
template <class T>
inline void triangleGenericOn(const mg::Shared& z, const float* mask, int n, int C, const std::string& pre, bool divideByLength) {
  const int np = padded(n); const size_t cs = (size_t)np * np;
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  float* pair = baseOf(z.local, lo, n, C);
  std::string pgOf[2] = { operandWeight(pre, C, 0), operandWeight(pre, C, 1) };
  float alpha = divideByLength ? 1.f / n : 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  size_t per = std::max<size_t>(32, CHUNK / (4 * C));
  float* m = scratch<float>("trib.mask", per);
  T* ln = scratch<T>("trib.ln", per * C); T* pgOut = scratch<T>("trib.pg", per * 2 * C);
  auto operands = [&](const TriRect& r, T* out, int side) {
    for (size_t q0 = 0; q0 < r.size(); q0 += per) {
      size_t cnt = std::min(per, r.size() - q0);
      WITH_PAIR_T(rectLayerNormK<T, PT><<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(pair, mask, ln, m, r, q0, cnt, n, C,
        W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset")));
      linear<T, T>(ln, pgOut, cnt, C, 2 * C, pgOf[side]);
      rectGateK<T><<<dim3((unsigned)((cnt + 31) / 32), (C + 31) / 32), dim3(32, 8), 0, STREAM>>>(pgOut, m, out, q0, cnt, C, r.size());
    }
  };
  int maxShare = 0; for (int r = 0; r < mg::WORLD; ++r) { int a0, a1; rowsOf(n, r, a0, a1); maxShare = std::max(maxShare, a1 - a0); }
  mg::Shared& bmine = mg::shared(std::string("sh.bg") + (sizeof(T) == 2 ? "16." : "32.") + std::to_string(C), (size_t)maxShare * np * C * sizeof(T));
  if (hi > lo) operands({lo, hi - lo, 0, np}, (T*)bmine.local, 1);
  T* b = scratch<T>("trib.b", cs * C);
  mg::fence();
  for (int r = 0; r < mg::WORLD; ++r) {
    int rlo, rhi; rowsOf(n, r, rlo, rhi);
    if (rhi <= rlo) continue;
    const size_t rb = (size_t)(rhi - rlo) * np * sizeof(T);
    CK(cudaMemcpy2DAsync((char*)b + (size_t)rlo * np * sizeof(T), cs * sizeof(T), bmine.peer[r], rb, rb, C, cudaMemcpyDefault, STREAM));
  }
  mg::fence();
  int width = (int)std::max<size_t>(8, std::min<size_t>(np, (CHUNK / C) / np / 8 * 8));
  {
    size_t f, t; deviceMemInfo(&f, &t);
    size_t perRow = (size_t)np * C * (sizeof(T) + 4), spare = f > t / 16 ? f - t / 16 : 0;
    width = (int)std::max<size_t>(width, std::min<size_t>(np, std::min<size_t>(spare / perRow, 1024) / 8 * 8));
  }
  width = std::min(width, std::max(8, hi - lo));
  T* a = scratch<T>("trib.a", (size_t)width * np * C);
  float* prod = scratch<float>("trib.prod", (size_t)width * np * C);
  T* t1 = scratch<T>("trib.t1", per * C); T* t2 = scratch<T>("trib.t2", per * C);
  for (int k0 = lo; k0 < std::min(hi, n); k0 += width) {
    int w = std::min(width, hi - k0);
    TriRect r{k0, w, 0, np};
    operands(r, a, 0);
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, np, w, np, &alpha, b, cudaType<T>(), np, cs, a,
      cudaType<T>(), np, r.size(), &zero, prod, CUDA_R_32F, np, r.size(), C, CUBLAS_COMPUTE_32F, algo));
    for (size_t q0 = 0; q0 < r.size(); q0 += per) {
      size_t cnt = std::min(per, r.size() - q0);
      rectCenterNormK<T><<<(unsigned)((cnt + 31) / 32), dim3(32, 8), 0, STREAM>>>(prod, ln, q0, cnt, C, r.size(),
        W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"));
      linear<T, T>(ln, t1, cnt, C, C, pre + ".outputProjection");
      WITH_PAIR_T(rectLayerNormK<T, PT><<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(pair, mask, ln, nullptr, r, q0, cnt, n, C,
        W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset")));
      linear<T, T>(ln, t2, cnt, C, C, pre + ".gatingLinear");
      WITH_PAIR_T(rectGatedAddK<T, PT><<<blocks(cnt * C), 256, 0, STREAM>>>(into(pair), t1, t2, r, q0, cnt, n, C));
    }
  }
  releaseScratch({ "trib." });
}
// bias[h][i][j] (or [h][j][i] when swapped) = scale * raw[q][h] for the pairs p0 + q, raw pair-major (a linear's)
template <class TB>
__global__ void biasFromFlatRowsK(const float* raw, TB* bias, size_t p0, size_t cnt, int n, int stride, int heads, bool swap,
                                  float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * cnt) return;
  size_t q = t / heads; int h = (int)(t % heads);
  size_t p = p0 + q, a = p / n, b = p % n, i = swap ? b : a, j = swap ? a : b;
  bias[((size_t)h * n + i) * stride + j] = fromF<TB>(scale * raw[t]);
}
template <class T>
inline void rowAttentionGeneric(const mg::Shared& z, const float* mask, int n, int C, int heads, int D, const std::string& pre,
                                bool swap) {
  const int Wd = heads * D;
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  const int rowsHere = storedRows(n, mg::RANK);
  float* pair = baseOf(z.local, lo, n, C);
  constexpr bool fast = std::is_same_v<T, half>;
  const int stride = (n + 7) / 8 * 8;
  mg::Shared& biasS = mg::shared(std::string("sh.biasg") + (fast ? "16." : "32.") + std::to_string(heads), (size_t)heads * n * stride * sizeof(T));
  T* bias = (T*)biasS.local;
  CK(cudaMemsetAsync(bias, 0, (size_t)heads * n * stride * sizeof(T), STREAM));
  const size_t pairsHere = (size_t)rowsHere * n;
  size_t per = std::max<size_t>(1, std::min(std::max<size_t>(pairsHere, 1), CHUNK / C));
  T* lnc = scratch<T>("grid.normChunk", per * C);
  float* raw = scratch<float>("grid.rawbias", per * heads);
  for (size_t q0 = 0; q0 < pairsHere; q0 += per) {
    size_t r = std::min(per, pairsHere - q0);
    lnPairRows<T>((float*)z.local, q0, lnc, r, C, pre + ".actNormScale", pre + ".actNormOffset");
    linear<T, float>(lnc, raw, r, C, heads, pre + ".pairBiasProjection");
    biasFromFlatRowsK<T><<<blocks((size_t)heads * r), 256, 0, STREAM>>>(raw, bias, (size_t)lo * n + q0, r, n, stride, heads, swap,
                                                                        fast ? LOG2E : 1.f);
  }
  mg::fence();
  const size_t e = sizeof(T);
  for (int r = 0; r < mg::WORLD; ++r) {
    if (r == mg::RANK) continue;
    int rlo, rhi; rowsOf(n, r, rlo, rhi); rhi = std::min(rhi, n);
    if (rhi <= rlo) continue;
    for (int h = 0; h < heads; ++h) {
      const size_t plane = (size_t)h * n * stride * e;
      if (!swap) CK(cudaMemcpyAsync((char*)bias + plane + (size_t)rlo * stride * e, (const char*)biasS.peer[r] + plane +
                                    (size_t)rlo * stride * e, (size_t)(rhi - rlo) * stride * e, cudaMemcpyDefault, STREAM));
      else CK(cudaMemcpy2DAsync((char*)bias + plane + (size_t)rlo * e, (size_t)stride * e, (const char*)biasS.peer[r] + plane +
                                (size_t)rlo * e, (size_t)stride * e, (size_t)(rhi - rlo) * e, n, cudaMemcpyDefault, STREAM));
    }
  }
  mg::fence();
  std::string qkvg = qkvgWeight(pre, C, Wd, true);
  const float* gateBias = hasW(pre + ".gatingQueryBias") ? W(pre + ".gatingQueryBias") : nullptr;
  const float* outBias = hasW(pre + ".outputProjectionBias") ? W(pre + ".outputProjectionBias") : nullptr;
  const float scale = 1.f / sqrtf((float)D);
  size_t R = std::max<size_t>(1, std::min<size_t>(std::max(rowsHere, 1), CHUNK / ((size_t)n * 4 * Wd)));
  for (size_t r0 = lo; r0 < (size_t)(lo + rowsHere); r0 += R) {
    size_t rows = std::min(R, (size_t)(lo + rowsHere) - r0), prs = rows * n;
    T* act = scratch<T>("grid.act", std::min<size_t>(R, rowsHere) * n * C);
    lnPairRows<T>((float*)z.local, (r0 - lo) * n, act, prs, C, pre + ".actNormScale", pre + ".actNormOffset");
    T* qkvgOut = scratch<T>("grid.qkvg", (std::min<size_t>(R, rowsHere) * n + 128) * 4 * Wd);
    linear<T, T>(act, qkvgOut, prs, C, 4 * Wd, qkvg);
    if (gateBias) addGateBiasK<T><<<blocks(prs * Wd), 256, 0, STREAM>>>(qkvgOut, gateBias, prs, Wd);
    T* gathered = scratch<T>("grid.gathered", std::min<size_t>(R, rowsHere) * n * Wd);
    flashGrid<T>(qkvgOut, bias, stride, MASK_ALL_ONES && fast ? nullptr : mask, gathered, n, heads, D, r0, rows, false, scale);
    float* o = scratch<float>("grid.o", std::min<size_t>(R, rowsHere) * n * C);
    linear<T, float>(gathered, o, prs, Wd, C, pre + ".outputProjection");
    if (outBias) addBiasK<<<blocks(prs * C), 256, 0, STREAM>>>(o, outBias, prs, C);
    WITH_PAIR_T(addGridK<PT, float><<<blocks(prs * C / 4), 256, 0, STREAM>>>(into(pair), o, n, C, r0, rows, false));
  }
  releaseScratch({ "grid." });
}
// A pairformer block's pair updates on the sharded pair z (this rank's slab), zT a slab-sized buffer for z^T
template <class T>
inline void pairUpdates(const mg::Shared& z, const mg::Shared& zT, const float* mask, int n, int C, const std::string& pre,
                        bool swap, bool divide, int transitionFactor) {
  // the fused forms where they apply (the 128/256-channel streaming triangle on a bf16 pair, the 128-channel 4 x 32
  // grid attention without biases), the generic ones otherwise
  auto tri = [&](const mg::Shared& s, const std::string& p) {
    if constexpr (std::is_same_v<T, half>) {
      if (PAIR16 && (C == 128 || C == 256) && blockedFused(n, C)) { triangleSharded<T>(s, mask, n, C, p, divide); return; }
    }
    triangleGenericOn<T>(s, mask, n, C, p, divide);
  };
  int heads = (int)M.meta(pre + ".pairAttention1.heads"), D = (int)M.meta(pre + ".pairAttention1.dimension");
  auto att = [&](const mg::Shared& s, const std::string& p, bool sw) {
    if constexpr (std::is_same_v<T, half>) {
      if (C == 128 && heads * D == 128 && D == 32 && !hasW(p + ".gatingQueryBias") && !hasW(p + ".outputProjectionBias")) {
        rowAttention(s, mask, n, C, heads, D, p, sw); return;
      }
    }
    rowAttentionGeneric<T>(s, mask, n, C, heads, D, p, sw);
  };
  tri(z, pre + ".triangleMultiplicationOutgoing"); stage("tri.out");
  transpose(z, zT, n, C);
  tri(zT, pre + ".triangleMultiplicationIncoming"); stage("tri.in");
  transpose(zT, z, n, C);
  att(z, pre + ".pairAttention1", false); stage("grid.row");
  transpose(z, zT, n, C);
  att(zT, pre + ".pairAttention2", !swap); stage("grid.col");
  transpose(zT, z, n, C);
  const int rowsHere = storedRows(n, mg::RANK);
  if (rowsHere) transition<T>((float*)z.local, (size_t)rowsHere * n, C, transitionFactor, pre + ".pairTransition");
  stage("transition");
}

}  // namespace sh
