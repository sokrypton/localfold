// ESM2 3B, the language model Chai-1's tokens read (alphafold3/model/esm.py, af3-any-model, is the reading):
//
//   x = embed(ids) * (1 - 0.15 * 0.8)                       token dropout's inference constant
//   per block:  h = LN(x; attn_norm);  q, k, v = h @ W + b   (no q/k norms)
//               q, k = RoPE(q, k)                           head 64, base 10000, split halves, position = row
//               x = x + attn(q, k, v) @ attn_out + b
//               x = x + gelu(LN(x; ffn_norm) @ fc1 + b) @ fc2 + b      (exact gelu, no residual scale)
//   out = LN(x; final_norm), the LAST state only, one chain at a time as [BOS, residues, EOS], BOS/EOS stripped
//
// The weights are af3-any-model's own lm/esm2.bin.zst, read as published (common.cuh's blob reader): its matrices
// [in, out] int8 with a float32 scale per output channel, kept RESIDENT as their codes (2.7 GB, not 11) and each
// expanded to float16 for its GEMM; biases and norms float16. q, k and v are three matrices there, expanded side
// by side into one [in, 3C] so the attention's projection stays one GEMM.
// Loaded under `e/` (Model::loadBundle(dir, "e", nullptr, "", "esm2/blocks/")).
#pragma once
#include "../af3/src/common.cuh"

namespace esm2 {
// a resident [in, out] int8 matrix (a blob's: a float32 scale per channel of `out` and per block of rows) expanded
// to float16 into columns [col0, col0 + out) of a [in, ld] matrix. A row a blockIdx.y and columns across the
// threads: per element there is no division left (the 64-bit ones of a flat index made this most of the
// tower's time - 64 ms of Chai-1's target features on 6MRR)
__global__ void expandColumnsK(const signed char* q, const float* scale, __half* w, int out, int g, int ld, int col0) {
  int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (col >= out) return;
  w[(size_t)row * ld + col0 + col] = __float2half((float)q[(size_t)row * out + col] * scale[(size_t)(row / g) * out + col]);
}
__global__ void embedK(const int* ids, const float* table, float* x, int rows, int C, float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)rows * C) x[t] = table[(size_t)ids[t / C] * C + t % C] * scale;
}
// LayerNorm over C, a warp a row, float32
__global__ void layerNormK(const float* x, float* y, int rows, int C, const float* scale, const float* offset) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + (size_t)row * C;
  float s = 0; for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) y[(size_t)row * C + c] = (xr[c] - mean) * inv * scale[c] + offset[c];
}
__global__ void biasK(float* __restrict__ y, const float* __restrict__ b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t < n4) {
      float4 yv = reinterpret_cast<const float4*>(y)[t];
      float4 bv = __ldg(reinterpret_cast<const float4*>(b) + (t % (size_t)c4));
      reinterpret_cast<float4*>(y)[t] = make_float4(yv.x + bv.x, yv.y + bv.y, yv.z + bv.z, yv.w + bv.w);
    }
    return;
  }
  if (t < rows * C) y[t] += b[t % C];
}
__global__ void biasGeluK(float* __restrict__ y, const float* __restrict__ b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t < n4) {
      float4 yv = reinterpret_cast<const float4*>(y)[t];
      float4 bv = __ldg(reinterpret_cast<const float4*>(b) + (t % (size_t)c4));
      auto g = [](float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); };
      reinterpret_cast<float4*>(y)[t] = make_float4(g(yv.x + bv.x), g(yv.y + bv.y), g(yv.z + bv.z), g(yv.w + bv.w));
    }
    return;
  }
  if (t < rows * C) { float v = y[t] + b[t % C]; y[t] = 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }
}
// rotary on q and k inside the packed [rows, 3C] buffer: channel d with d + 32 of each 64-wide head
__device__ __constant__ float ESM2_INV_FREQ[32] = {
  0x1.000000p+0f,  0x1.800000p-1f,  0x1.1fe000p-1f,  0x1.afc000p-2f,
  0x1.43c000p-2f,  0x1.e5c000p-3f,  0x1.6c4000p-3f,  0x1.110000p-3f,
  0x1.998000p-4f,  0x1.334000p-4f,  0x1.ccc000p-5f,  0x1.598000p-5f,
  0x1.030000p-5f,  0x1.848000p-6f,  0x1.234000p-6f,  0x1.b50000p-7f,
  0x1.47c000p-7f,  0x1.eb8000p-8f,  0x1.708000p-8f,  0x1.144000p-8f,
  0x1.9e8000p-9f,  0x1.36c000p-9f,  0x1.d24000p-10f, 0x1.5d8000p-10f,
  0x1.064000p-10f, 0x1.894000p-11f, 0x1.26c000p-11f, 0x1.ba4000p-12f,
  0x1.4b8000p-12f, 0x1.f14000p-13f, 0x1.750000p-13f, 0x1.17c000p-13f
};
__device__ __forceinline__ void esm2SincosD(double angle, float* sOut, float* cOut) {
  constexpr double INV_PIO2 = 0x1.45f306dc9c883p-1;
  constexpr double PIO2_HI  = 0x1.921fb54442d18p+0;
  constexpr double PIO2_LO  = 0x1.1a62633145c07p-54;
  double kf = nearbyint(angle * INV_PIO2);
  long long q = (long long)kf;
  double r = fma(-kf, PIO2_HI, angle) - kf * PIO2_LO;
  double r2 = r * r;
  double s = (((((-0x1.ae64567f544e4p-26 * r2 + 0x1.71de3a556c734p-19) * r2
                 - 0x1.a01a01a01a01ap-13) * r2 + 0x1.1111111111111p-7) * r2
               - 0x1.5555555555555p-3) * r2) * r + r;
  double c = (((((-0x1.27e4fb7789f5cp-22 * r2 + 0x1.a01a01a01a01ap-16) * r2
                 - 0x1.6c16c16c16c17p-10) * r2 + 0x1.5555555555555p-5) * r2
               - 0x1.0000000000000p-1) * r2) + 1.0;
  double sinA = (q & 1) ? c : s;
  double cosA = (q & 1) ? -s : c;
  if (q & 2) { sinA = -sinA; cosA = -cosA; }
  *sOut = (float)sinA;
  *cOut = (float)cosA;
}
__global__ void esm2RopeTableK(float2* __restrict__ cosSin, int rows) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * 32) return;
  int d = t & 31, row = t >> 5;
  float s, c;
  esm2SincosD((double)((float)row * ESM2_INV_FREQ[d]), &s, &c);
  cosSin[t] = make_float2(c, s);
}
__global__ void ropeK(float* __restrict__ qkv, const float2* __restrict__ cosSin, int rows, int heads, int C) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * heads * 32) return;
  int d = t & 31, h = (t >> 5) % heads, row = t / (32 * heads);
  float2 cs = __ldg(cosSin + row * 32 + d);
  float* pq = qkv + (size_t)row * 3 * C + h * 64 + d;
  float aq = pq[0], bq = pq[32];
  pq[0] = aq * cs.x - bq * cs.y;
  pq[32] = aq * cs.y + bq * cs.x;
  float* pk = pq + C;
  float ak = pk[0], bk = pk[32];
  pk[0] = ak * cs.x - bk * cs.y;
  pk[32] = ak * cs.y + bk * cs.x;
}
__global__ void softmaxK(float* __restrict__ S, int rows, float scale) {
  float* s = S + (size_t)blockIdx.x * rows;
  __shared__ float red[32];
  if (rows <= 1024) {
    float sv[4];
    float m = -INFINITY;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int j = threadIdx.x + k * 256;
      if (j < rows) { float v = s[j] * scale; sv[k] = v; m = fmaxf(m, v); }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
      #pragma unroll
      for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o));
      if (threadIdx.x == 0) red[0] = v; }
    __syncthreads(); m = red[0];
    float sum = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int j = threadIdx.x + k * 256;
      if (j < rows) { float e = expf(sv[k] - m); sv[k] = e; sum += e; }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
      #pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (threadIdx.x == 0) red[0] = v; }
    __syncthreads();
    float inv = 1.f / red[0];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int j = threadIdx.x + k * 256;
      if (j < rows) s[j] = sv[k] * inv;
    }
    return;
  }
  float m = -INFINITY;
  for (int j = threadIdx.x; j < rows; j += blockDim.x) { float v = s[j] * scale; s[j] = v; m = fmaxf(m, v); }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); m = red[0]; __syncthreads();
  float sum = 0;
  for (int j = threadIdx.x; j < rows; j += blockDim.x) { float e = expf(s[j] - m); s[j] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < rows; j += blockDim.x) s[j] *= inv;
}
__global__ void toHalfRowsK(const float* __restrict__ x, __half* __restrict__ y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && ((uintptr_t)x & 15) == 0 && ((uintptr_t)y & 7) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 v = __ldg(reinterpret_cast<const float4*>(x) + t);
      __half2 h0 = __floats2half2_rn(v.x, v.y);
      __half2 h1 = __floats2half2_rn(v.z, v.w);
      reinterpret_cast<uint2*>(y)[t] = make_uint2(*reinterpret_cast<uint32_t*>(&h0), *reinterpret_cast<uint32_t*>(&h1));
    }
    return;
  }
  if (t < n) y[t] = __float2half(x[t]);
}

struct Tower { int layers, C, heads, ffn; float embedScale; };
// a block's tensor: esm2/blocks/<module>/<layer>/<leaf>, the blob's own naming
inline std::string blockName(int layer, const std::string& module, const std::string& leaf) {
  return "esm2/blocks/" + module + "/" + std::to_string(layer) + "/" + leaf;
}
// the shapes say it (the blob carries no header): 36 blocks x 2560, heads of 64, FFN 10240
inline Tower tower() {
  int layers = 0; while (M.has("e/" + blockName(layers, "q", "weights"))) ++layers;
  int C = (int)M.meta("e/esm2/embed/weights#1"), ffn = (int)M.meta("e/" + blockName(0, "fc1", "weights") + "#1");
  if (layers == 0 || C % 64) { fprintf(stderr, "e/: not ESM2's blob (%d blocks, width %d)\n", layers, C); exit(1); }
  return { layers, C, C / 64, ffn, 1.f - 0.15f * 0.8f };       // (token dropout's inference constant)
}
// Y [rows, out] = X [rows, in] W (+ beta Y), W the resident [in, out] matrices named, side by side: f16 inputs, f32
// accumulation
inline void gemm(const float* X, const std::vector<std::string>& names, float* Y, int rows, int in, int out, float beta) {
  __half* w = scratch<__half>("esm2.w16", (size_t)in * out);
  int col0 = 0;
  for (const std::string& name : names) {
    ResidentInt8 r = M.residentInt8("e/" + name);
    if (!r.scales32 || r.rows != (size_t)in || col0 + r.block > out) {
      fprintf(stderr, "e/%s is not a [%d, *] int8 matrix of the blob's\n", name.c_str(), in); exit(1);
    }
    int g = (int)((r.rows + r.rowBlocks - 1) / r.rowBlocks);
    expandColumnsK<<<dim3((unsigned)((r.block + 255) / 256), (unsigned)r.rows), 256, 0, STREAM>>>(r.codes, r.scales32, w,
                                                                                               r.block, g, out, col0);
    col0 += r.block;
  }
  if (col0 != out) { fprintf(stderr, "e/%s...: %d columns, not %d\n", names[0].c_str(), col0, out); exit(1); }
  __half* xh = scratch<__half>("esm2.xh", (size_t)rows * in);
  size_t nIn = (size_t)rows * in;
  toHalfRowsK<<<blocks((nIn & 3) == 0 ? (nIn >> 2) : nIn), 256, 0, STREAM>>>(X, xh, nIn);
  const float one = 1.f;
  CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, out, rows, in, &one, w, CUDA_R_16F, out, xh, CUDA_R_16F, in, &beta, Y,
                  CUDA_R_32F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
inline const float* Wf(const std::string& name) { return W("e/" + name); }
// q | k | v's biases as one [3C], once a layer
inline const float* qkvBias(int layer, int C) {
  static std::map<int, float*> made;     // (a derived weight: forgotten with the others)
  static bool hooked = false;
  if (!hooked) { FORGET_HOOKS.push_back([] { for (auto& [l, p] : made) CK(cudaFree(p)); made.clear(); }); hooked = true; }
  float*& b = made[layer];
  if (!b) {
    CK(devMalloc(&b, 3 * (size_t)C * 4));
    for (int m = 0; m < 3; ++m)
      CK(cudaMemcpyAsync(b + m * C, Wf(blockName(layer, std::string(1, "qkv"[m]), "bias")), (size_t)C * 4,
                         cudaMemcpyDeviceToDevice, STREAM));
  }
  return b;
}

// one chain: ids [n] (its residues, ESM2's alphabet) -> out [n, C], the last state after the final LayerNorm
inline void embedChain(const Tower& t, const int* idsDevWrapped, int n, float* out) {
  int R = n + 2, C = t.C;
  float* x = scratch<float>("esm2.x", (size_t)R * C);
  float* xn = scratch<float>("esm2.xn", (size_t)R * C);
  float* qkv = scratch<float>("esm2.qkv", (size_t)R * 3 * C);
  float* ctx = scratch<float>("esm2.ctx", (size_t)R * C);
  float* h = scratch<float>("esm2.h", (size_t)R * t.ffn);
  float* S = scratch<float>("esm2.scores", (size_t)t.heads * R * R);
  float2* cosSin = reinterpret_cast<float2*>(scratch<float>("esm2.rope", (size_t)R * 64));
  esm2RopeTableK<<<blocks((size_t)R * 32), 256, 0, STREAM>>>(cosSin, R);
  embedK<<<blocks((size_t)R * C), 256, 0, STREAM>>>(idsDevWrapped, Wf("esm2/embed/weights"), x, R, C, t.embedScale);
  const float one = 1.f, zero = 0.f;
  size_t rcVec = (C & 3) == 0 ? (size_t)R * (C >> 2) : (size_t)R * C;
  size_t r3cVec = (C & 3) == 0 ? (size_t)R * 3 * (C >> 2) : (size_t)R * 3 * C;
  size_t rFfnVec = (t.ffn & 3) == 0 ? (size_t)R * (t.ffn >> 2) : (size_t)R * t.ffn;
  for (int l = 0; l < t.layers; ++l) {
    auto N = [&](const char* module, const char* leaf) { return blockName(l, module, leaf); };
    layerNormK<<<(R + 7) / 8, 256, 0, STREAM>>>(x, xn, R, C, Wf(N("attn_norm", "scale")), Wf(N("attn_norm", "offset")));
    gemm(xn, { N("q", "weights"), N("k", "weights"), N("v", "weights") }, qkv, R, C, 3 * C, 0.f);
    biasK<<<blocks(r3cVec), 256, 0, STREAM>>>(qkv, qkvBias(l, C), R, 3 * C);
    ropeK<<<blocks((size_t)R * t.heads * 32), 256, 0, STREAM>>>(qkv, cosSin, R, t.heads, C);
    // per head: S = K^T Q (col-major [keys, queries]), softmax over keys, ctx = V S
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, R, R, 64, &one, qkv + C, 3 * C, 64, qkv, 3 * C, 64, &zero,
                                 S, R, (long long)R * R, t.heads));
    softmaxK<<<(unsigned)(t.heads * R), 256, 0, STREAM>>>(S, R, 0.125f);
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, 64, R, R, &one, qkv + 2 * C, 3 * C, 64, S, R,
                                 (long long)R * R, &zero, ctx, C, 64, t.heads));
    gemm(ctx, { N("attn_out", "weights") }, x, R, C, C, 1.f);
    biasK<<<blocks(rcVec), 256, 0, STREAM>>>(x, Wf(N("attn_out", "bias")), R, C);
    layerNormK<<<(R + 7) / 8, 256, 0, STREAM>>>(x, xn, R, C, Wf(N("ffn_norm", "scale")), Wf(N("ffn_norm", "offset")));
    gemm(xn, { N("fc1", "weights") }, h, R, C, t.ffn, 0.f);
    biasGeluK<<<blocks(rFfnVec), 256, 0, STREAM>>>(h, Wf(N("fc1", "bias")), R, t.ffn);
    gemm(h, { N("fc2", "weights") }, x, R, t.ffn, C, 1.f);
    biasK<<<blocks(rcVec), 256, 0, STREAM>>>(x, Wf(N("fc2", "bias")), R, C);
  }
  // the final LayerNorm over the residues only (BOS and EOS dropped)
  layerNormK<<<(n + 7) / 8, 256, 0, STREAM>>>(x + C, out, n, C, Wf("esm2/final_norm/scale"), Wf("esm2/final_norm/offset"));
}
}  // namespace esm2
