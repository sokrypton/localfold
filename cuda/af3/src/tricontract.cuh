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
inline void triContractBf16(bool outgoing, int np, size_t cs, int C, float alpha, const __nv_bfloat16* a,
                            const __nv_bfloat16* b, __nv_bfloat16* p, bool planAll = false) {
  const float zero = 0.f;
  const bool window = TRI_LT_TILE && np >= 200 && np <= 352;
  if (window || planAll) {
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
  if (outgoing)
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, np, np, np, &alpha, b, CUDA_R_16BF, np, cs, a,
      CUDA_R_16BF, np, cs, &zero, p, CUDA_R_16BF, np, cs, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  else
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, np, np, np, &alpha, a, CUDA_R_16BF, np, cs, b,
      CUDA_R_16BF, np, cs, &zero, p, CUDA_R_16BF, np, cs, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}


