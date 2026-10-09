// The triangle multiplication's bf16 contraction (cuda/af3's pair track and ESMFold2's 256-channel one)
#pragma once
#include "common.cuh"

// The bf16 contraction, one np x np GEMM per channel. cuBLAS's default picks a 64x256 tile from
// np 152 to 392 (cuBLAS 12, A100), where a 128x128 tile is up to a quarter faster - 66.5 against
// 84.7 us at np 264 (261 tokens), 83 against 111 at 336, 31 against 39 at 200 - and level at 360
// and up, where the default takes it itself; below ~192 the default wins. So there, cuBLASLt with
// the heuristic list's first 128x128 candidate: a fixed rule, never a timing, so the choice (and
// the output) is the same every run.
inline bool TRI_LT_TILE = true;
// planAll (ESMFold2, whose single trunk pass has no CUDA graph to hide host time): a cached cuBLASLt plan
// at every np - cuBLASLt's own first candidate outside the 128x128 window - because cuBLAS's per-call
// choice for a bf16 batched GEMM cost ~60 us of host a call (ESMFold2 at 261 tokens: 11.5 ms of idle GPU)
// A bf16 batched GEMM (bf16 out, f32 accumulation), as cublasGemmStridedBatchedEx - but past 1e10 multiply-adds a
// channel with the algorithm cuBLASLt's heuristic picks for ONE channel's GEMM, run over the batch: there cuBLAS 12's
// batched heuristic (A100) turns to 64x64 tiles at 80-150 TFLOP/s, where the single-matrix pick (a 256x128 or 128x256
// tile, 15 stages) runs the whole batch at 260-277 - a 6,916-token fold's blocks (6928 x 872 x 6928, 128 channels) 135
// -> 39 ms, np 5000 whole 88 -> 29 ms per 32 channels, level at np 2976. A plan a shape, cached; the choice is
// cuBLASLt's own list, never a timing
inline void bf16Gemms(cublasOperation_t ta, cublasOperation_t tb, int m, int nn, int k, float alpha,
                      const __nv_bfloat16* A, int lda, long long sA, const __nv_bfloat16* B, int ldb, long long sB,
                      __nv_bfloat16* Cm, int ldc, long long sC, int batch) {
  const float zero = 0.f;
  if ((double)m * nn * k >= 1e10) {
    struct Plan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc; cublasLtMatmulAlgo_t algo; bool ok; };
    static cublasLtHandle_t lt = nullptr;
    static std::map<std::tuple<int, int, int, int, int, int, int, int, long long, long long, long long, int>, Plan> plans;
    if (!lt) CB(cublasLtCreate(&lt));
    auto key = std::make_tuple((int)ta, (int)tb, m, nn, k, lda, ldb, ldc, sA, sB, sC, batch);
    auto it = plans.find(key);
    if (it == plans.end()) {
      Plan pl{};
      CB(cublasLtMatmulDescCreate(&pl.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      CB(cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
      CB(cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
      auto layout = [&](int rows, int cols, int ld, long long stride, int count) {
        cublasLtMatrixLayout_t l; CB(cublasLtMatrixLayoutCreate(&l, CUDA_R_16BF, rows, cols, ld));
        if (count > 1) {
          CB(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
          CB(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
        }
        return l;
      };
      const int ar = ta == CUBLAS_OP_N ? m : k, ac = ta == CUBLAS_OP_N ? k : m;
      const int br = tb == CUBLAS_OP_N ? k : nn, bc = tb == CUBLAS_OP_N ? nn : k;
      cublasLtMatrixLayout_t a1 = layout(ar, ac, lda, 0, 1), b1 = layout(br, bc, ldb, 0, 1), c1 = layout(m, nn, ldc, 0, 1);
      pl.la = layout(ar, ac, lda, sA, batch); pl.lb = layout(br, bc, ldb, sB, batch); pl.lc = layout(m, nn, ldc, sC, batch);
      cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
      size_t ws = 0;     // no workspace: captured in a CUDA graph as it is
      CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
      cublasLtMatmulHeuristicResult_t res[8]; int got = 0;
      CB(cublasLtMatmulAlgoGetHeuristic(lt, pl.op, a1, b1, c1, c1, pref, 8, res, &got));
      for (int i = 0; i < got && !pl.ok; ++i) {
        cublasLtMatmulHeuristicResult_t chk;
        if (res[i].state == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulAlgoCheck(lt, pl.op, pl.la, pl.lb, pl.lc, pl.lc, &res[i].algo, &chk) == CUBLAS_STATUS_SUCCESS &&
            chk.workspaceSize == 0) { pl.algo = res[i].algo; pl.ok = true; }
      }
      CB(cublasLtMatmulPreferenceDestroy(pref));
      cublasLtMatrixLayoutDestroy(a1); cublasLtMatrixLayoutDestroy(b1); cublasLtMatrixLayoutDestroy(c1);
      it = plans.emplace(key, pl).first;
    }
    if (it->second.ok) {          // (no candidate valid over the batch: cuBLAS's own pick, below)
      const Plan& pl = it->second;
      CB(cublasLtMatmul(lt, pl.op, &alpha, A, pl.la, B, pl.lb, &zero, Cm, pl.lc, Cm, pl.lc, &pl.algo, nullptr, 0, STREAM));
      return;
    }
  }
  CB(cublasGemmStridedBatchedEx(H, ta, tb, m, nn, k, &alpha, A, CUDA_R_16BF, lda, sA, B, CUDA_R_16BF, ldb, sB, &zero,
    Cm, CUDA_R_16BF, ldc, sC, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
#include <cuda_fp8.h>
// FP8 tensor instructions (Ada and Blackwell: compute capability 8.9 on; an A100 has none). LOCALFOLD_FP8=0 keeps bf16
inline bool fp8Tensor() {
  static const bool on = [] {
    if (getenv("LOCALFOLD_FP8") && !atoi(getenv("LOCALFOLD_FP8"))) return false;
    int d, ma, mi; CK(cudaGetDevice(&d));
    CK(cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, d)); CK(cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, d));
    return ma * 10 + mi >= 89;
  }();
  return on;
}
// The contraction in the OUTGOING layout for either direction (the incoming one's operands written transposed, RectMap's
// T): p[c][i][j] = sum_k a[c][i][k] b[c][j][k], the contracted index contiguous in both - the only layout an FP8 GEMM
// takes. fp8: a and b e4m3 (unscaled: the operands sit well inside its range - folds within 0.03 A of bf16's, emulated),
// through cuBLASLt into a bf16 product; else bf16 (bf16Gemms)
inline void triContractTN(int np, size_t cs, int C, float alpha, const void* a, const void* b, __nv_bfloat16* p, bool fp8) {
  if (!fp8) {
    bf16Gemms(CUBLAS_OP_T, CUBLAS_OP_N, np, np, np, alpha, (const __nv_bfloat16*)b, np, cs, (const __nv_bfloat16*)a, np, cs, p, np, cs, C);
    return;
  }
  struct Plan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc; cublasLtMatmulAlgo_t algo; size_t ws; };
  static cublasLtHandle_t lt = nullptr;
  static std::map<std::tuple<int, size_t, int>, Plan> plans;
  if (!lt) CB(cublasLtCreate(&lt));
  auto key = std::make_tuple(np, cs, C);
  auto it = plans.find(key);
  if (it == plans.end()) {
    Plan pl{};
    CB(cublasLtMatmulDescCreate(&pl.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CB(cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CB(cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    int batch = C; long long stride = (long long)cs;
    auto layout = [&](cudaDataType t) {
      cublasLtMatrixLayout_t l; CB(cublasLtMatrixLayoutCreate(&l, t, np, np, np));
      CB(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
      CB(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
      return l;
    };
    pl.la = layout(CUDA_R_8F_E4M3); pl.lb = layout(CUDA_R_8F_E4M3); pl.lc = layout(CUDA_R_16BF);
    cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
    cublasLtMatmulHeuristicResult_t res[4]; int got = 0;
    for (size_t ws : { (size_t)0, (size_t)32 << 20 }) {      // (no workspace where one suffices: a CUDA graph captures it as it is)
      CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
      if (cublasLtMatmulAlgoGetHeuristic(lt, pl.op, pl.la, pl.lb, pl.lc, pl.lc, pref, 4, res, &got) == CUBLAS_STATUS_SUCCESS && got) {
        pl.algo = res[0].algo; pl.ws = res[0].workspaceSize; break;
      }
    }
    CB(cublasLtMatmulPreferenceDestroy(pref));
    if (!got) { fprintf(stderr, "no cuBLASLt FP8 algorithm for the contraction at np %d\n", np); exit(1); }
    it = plans.emplace(key, pl).first;
  }
  const Plan& pl = it->second;
  const float zero = 0.f;
  void* ws = pl.ws ? scratch<unsigned char>("tri.fp8ws", pl.ws) : nullptr;
  CB(cublasLtMatmul(lt, pl.op, &alpha, b, pl.la, a, pl.lb, &zero, p, pl.lc, p, pl.lc, &pl.algo, ws, pl.ws, STREAM));
}
inline void triContractBf16(bool outgoing, int np, size_t cs, int C, float alpha, const __nv_bfloat16* a,
                            const __nv_bfloat16* b, __nv_bfloat16* p, bool planAll = false) {
  const float zero = 0.f;
  const bool window = TRI_LT_TILE && np >= 200 && np <= 352;
  if (window || (planAll && (double)np * np * np < 1e10)) {
    struct Plan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t l; cublasLtMatmulAlgo_t algo; bool ok; };
    static cublasLtHandle_t lt = nullptr;
    static std::map<std::tuple<int, bool, int, size_t>, Plan> plans;     // the plan holds the batch and its stride
    if (!lt) CB(cublasLtCreate(&lt));
    auto it = plans.find({np, outgoing, C, cs});
    if (it == plans.end()) {
      // (the key needs no planAll: inside the window both take the 128x128 candidate, outside only planAll plans)
      Plan pl{}; 
      CB(cublasLtMatmulDescCreate(&pl.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t ta = outgoing ? CUBLAS_OP_T : CUBLAS_OP_N, tb = outgoing ? CUBLAS_OP_N : CUBLAS_OP_T;
      CB(cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
      CB(cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
      CB(cublasLtMatrixLayoutCreate(&pl.l, CUDA_R_16BF, np, np, np));
      int batch = C; long long stride = (long long)cs;
      CB(cublasLtMatrixLayoutSetAttribute(pl.l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
      CB(cublasLtMatrixLayoutSetAttribute(pl.l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
      cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
      size_t ws = 0;     // no workspace: the plan is captured in the trunk's CUDA graph as it is
      CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
      cublasLtMatmulHeuristicResult_t res[32]; int got = 0;
      CB(cublasLtMatmulAlgoGetHeuristic(lt, pl.op, pl.l, pl.l, pl.l, pl.l, pref, 32, res, &got));
      CB(cublasLtMatmulPreferenceDestroy(pref));
      for (int i = 0; i < got && !pl.ok; ++i) {
        int tile = 0, splitk = 1; size_t sz;
        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &sz);
        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &sz);
        if (res[i].state == CUBLAS_STATUS_SUCCESS && splitk <= 1 && (!window || tile == CUBLASLT_MATMUL_TILE_128x128)) {
          pl.algo = res[i].algo; pl.ok = true;
        }
      }
      it = plans.emplace(std::make_tuple(np, outgoing, C, cs), pl).first;
    }
    if (it->second.ok) {          // (a cuBLAS without that candidate keeps its own pick, below)
      const Plan& pl = it->second;
      CB(cublasLtMatmul(lt, pl.op, &alpha, outgoing ? b : a, pl.l, outgoing ? a : b, pl.l, &zero, p, pl.l, p, pl.l,
                        &pl.algo, nullptr, 0, STREAM));
      return;
    }
  }
  if (outgoing) bf16Gemms(CUBLAS_OP_T, CUBLAS_OP_N, np, np, np, alpha, b, np, cs, a, np, cs, p, np, cs, C);
  else bf16Gemms(CUBLAS_OP_N, CUBLAS_OP_T, np, np, np, alpha, a, np, cs, b, np, cs, p, np, cs, C);
}


