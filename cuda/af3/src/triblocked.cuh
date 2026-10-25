// The triangle multiplication in output blocks on f16 operands, shared by cuda/af2 and cuda/ef2 (each
// includes it after its own ltGemm): the f32 port of the same scheme is cuda/af3's triangleBlocked.
#pragma once
// ---------------------------------------------------------------- the triangle multiplication in blocks
// (cuda/af3's triangleBlocked, with AlphaFold 2's biases and its projection halves one after the other)
struct TriRect2 { int i0, I, j0, J; size_t size() const { return (size_t)I * J; } };
__device__ __forceinline__ size_t rect2Pair(const TriRect2& r, size_t q, int n) {
  size_t i = r.i0 + q / r.J, j = r.j0 + q % r.J;
  return i < (size_t)n && j < (size_t)n ? i * n + j : SIZE_MAX;
}
__global__ void rect2LayerNormK(const float* pair, const float* mask, half* out, float* m, TriRect2 r, size_t q0, size_t cnt,
                                int n, int C, const float* scale, const float* offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= cnt) return;
  size_t p = rect2Pair(r, q0 + row, n);
  const float* x = p != SIZE_MAX ? pair + p * C : nullptr;
  float s = 0, ss = 0;
  for (int c = lane; c < C; c += 32) {
    float v = x ? ptx_ldg_f32(x + c) : 0.f;
    s = ptx_add_f32(s, v);
    ss = ptx_fma_f32(v, v, ss);
  }
  s = ptx_warp_sum_f32(s);
  ss = ptx_warp_sum_f32(ss);
  float mean = ptx_div_f32(s, (float)C), inv = ptx_rsqrt_f32(ptx_add_f32(ptx_fma_f32(-mean, mean, ptx_div_f32(ss, (float)C)), 1e-5f));
  for (int c = lane; c < C; c += 32) {
    float v = x ? ptx_ldg_f32(x + c) : 0.f;
    ptx_st_global_b16(out + row * C + c, ptx_f32_to_f16_raw(ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(v, mean), inv), ptx_ldg_f32(scale + c), ptx_ldg_f32(offset + c))));
  }
  if (lane == 0 && m) ptx_st_global_f32(m + row, p != SIZE_MAX ? (mask ? ptx_ldg_f32(mask + p) : 1.f) : 0.f);
}
__global__ void rect2GateK(const half* pg, const float* m, half* out, size_t q0, size_t cnt, int C, size_t size) {
  __shared__ float A[32][33];
  size_t row0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    size_t row = row0 + ry; int c = c0 + tx;
    float v = 0;
    if (row < cnt && c < C) {
      const half* p = pg + row * 2 * C;
      v = ptx_mul_f32(ptx_mul_f32(ptx_f16_raw_to_f32(ptx_ldg_u16(p + c)), ptx_ldg_f32(m + row)), ptx_sigm_f32(ptx_f16_raw_to_f32(ptx_ldg_u16(p + C + c))));
    }
    ptx_st_shared_f32(&A[ry][tx], v);
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t row = row0 + tx; int c = c0 + cy;
    if (row < cnt && c < C) ptx_st_global_b16(out + (size_t)c * size + q0 + row, ptx_f32_to_f16_raw(ptx_ld_shared_f32(&A[tx][cy])));
  }
}
__global__ void rect2CenterNormK(const float* prod, half* out, size_t q0, size_t cnt, int C, size_t size,
                                 const float* scale, const float* offset) {
  __shared__ float ps[8][33], pss[8][33], mean[32], inv[32];
  size_t row0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
  size_t row = row0 + tx;
  float s = 0, ss = 0;
  if (row < cnt) for (int c = ty; c < C; c += 8) {
    float v = ptx_ldg_f32(prod + (size_t)c * size + q0 + row);
    s = ptx_add_f32(s, v);
    ss = ptx_fma_f32(v, v, ss);
  }
  ptx_st_shared_f32(&ps[ty][tx], s);
  ptx_st_shared_f32(&pss[ty][tx], ss);
  __syncthreads();
  if (ty == 0) {
    float a = 0, b = 0;
    for (int k = 0; k < 8; ++k) { a = ptx_add_f32(a, ptx_ld_shared_f32(&ps[k][tx])); b = ptx_add_f32(b, ptx_ld_shared_f32(&pss[k][tx])); }
    float mu = ptx_div_f32(a, (float)C);
    ptx_st_shared_f32(mean + tx, mu);
    ptx_st_shared_f32(inv + tx, ptx_rsqrt_f32(ptx_add_f32(ptx_fma_f32(-mu, mu, ptx_div_f32(b, (float)C)), 1e-5f)));
  }
  __syncthreads();
  for (int ry = ty; ry < 32; ry += 8) {
    size_t lr = row0 + ry; if (lr >= cnt) continue;
    float mu = ptx_ld_shared_f32(mean + ry), iv = ptx_ld_shared_f32(inv + ry);
    for (int c = tx; c < C; c += 32)
      ptx_st_global_b16(out + lr * C + c, ptx_f32_to_f16_raw(ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(ptx_ldg_f32(prod + (size_t)c * size + q0 + lr), mu), iv), ptx_ldg_f32(scale + c), ptx_ldg_f32(offset + c))));
  }
}
__global__ void rect2GatedAddK(float* pair, const float* t1, const half* t2, TriRect2 r, size_t q0, size_t cnt, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  size_t row = t / C; int c = (int)(t % C);
  size_t p = rect2Pair(r, q0 + row, n);
  if (p != SIZE_MAX) {
    float cur = ptx_ld_global_f32(pair + p * C + c);
    ptx_st_global_f32(pair + p * C + c, ptx_fma_f32(ptx_ldg_f32(t1 + t), ptx_sigm_f32(ptx_f16_raw_to_f32(ptx_ldg_u16(t2 + t))), cur));
  }
}

// The driver, on the caller's weights: the LayerNorm in, one [projection | gate] (C, 2C) f16 matrix and its
// optional 2C bias per operand (side 0 = a), the centre norm, the output projection and the gating linear
// (f16, optional biases). The fixed operand b is built whole, then per block of OUTPUT rows (outgoing) or
// columns (incoming) the free operand, the contraction and the output - see cuda/af3's triangleBlocked
// for why the incoming blocks are columns.
struct TriBlockedW {
  const float *lnScale, *lnOffset; const half *wA, *wB; const float *bA, *bB;
  const float *cnScale, *cnOffset; const half* wOut; const float* bOut; const half* wGate; const float* bGate;
};
inline void triangleBlockedHalf(float* pair, const float* mask, int L, int C, const TriBlockedW& w, bool outgoing,
                                size_t chunk) {
  int np = (L + 7) / 8 * 8; size_t cs = (size_t)np * np;
  const float one = 1.f, zero = 0.f;
  size_t per = std::max<size_t>(32, chunk / (4 * C));
  float* m = scratch<float>("trib.mask", per);
  half* ln = scratch<half>("trib.ln", per * C); half* pgOut = scratch<half>("trib.pg", per * 2 * C);
  auto operands = [&](const TriRect2& r, half* out, const half* wt, const float* bias) {
    for (size_t q0 = 0; q0 < r.size(); q0 += per) {
      size_t cnt = std::min(per, r.size() - q0);
      rect2LayerNormK<<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(pair, mask, ln, m, r, q0, cnt, L, C, w.lnScale, w.lnOffset);
      ltGemm(ln, wt, pgOut, true, cnt, C, 2 * C, bias, false, 0.f);
      rect2GateK<<<dim3((unsigned)((cnt + 31) / 32), (C + 31) / 32), dim3(32, 8), 0, STREAM>>>(pgOut, m, out, q0, cnt, C, r.size());
    }
  };
  half* b = scratch<half>("trib.b", cs * C);
  operands({0, np, 0, np}, b, w.wB, w.bB);
  int width = (int)std::max<size_t>(8, std::min<size_t>(np, (chunk / C) / np / 8 * 8));
  half* a = scratch<half>("trib.a", (size_t)width * np * C);
  float* prod = scratch<float>("trib.prod", (size_t)width * np * C);
  float* t1 = scratch<float>("trib.t1", per * C); half* t2 = scratch<half>("trib.t2", per * C);
  for (int k0 = 0; k0 < L; k0 += width) {
    int wd = std::min(width, np - k0);
    TriRect2 r = outgoing ? TriRect2{k0, wd, 0, np} : TriRect2{0, np, k0, wd};
    operands(r, a, w.wA, w.bA);
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, np, wd, np, &one, b, CUDA_R_16F, np, cs, a, CUDA_R_16F, np,
                                    r.size(), &zero, prod, CUDA_R_32F, np, r.size(), C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, wd, np, np, &one, a, CUDA_R_16F, wd, r.size(), b, CUDA_R_16F, np,
                                    cs, &zero, prod, CUDA_R_32F, wd, r.size(), C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    for (size_t q0 = 0; q0 < r.size(); q0 += per) {
      size_t cnt = std::min(per, r.size() - q0);
      rect2CenterNormK<<<(unsigned)((cnt + 31) / 32), dim3(32, 8), 0, STREAM>>>(prod, ln, q0, cnt, C, r.size(), w.cnScale, w.cnOffset);
      ltGemm(ln, w.wOut, t1, false, cnt, C, C, w.bOut, false, 0.f);
      rect2LayerNormK<<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(pair, mask, ln, nullptr, r, q0, cnt, L, C, w.lnScale, w.lnOffset);
      ltGemm(ln, w.wGate, t2, true, cnt, C, C, w.bGate, false, 0.f);
      rect2GatedAddK<<<blocks(cnt * C), 256, 0, STREAM>>>(pair, t1, t2, r, q0, cnt, L, C);
    }
  }
  releaseScratch({ "trib." });
}
