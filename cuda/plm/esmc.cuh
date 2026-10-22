// ESM-C's tower (cpu/esmc/tower.js is the reading) - the protein language model ESMFold2 reads, here so that any
// model can (cuda/plm: the language models, apart from the folding networks that read them):
//
//   h    = LN(x; attn_norm) @ qkv -> [q | k | v]
//   q, k = LN(q; q_norm), LN(k; k_norm)      over the FULL width, no offset, before the heads exist
//   q, k = RoPE(q, k)                         head 64, base 10000, split halves, position = row
//   x    = x + attn(q, k, v) @ attn_out       (attention within one chain: sequence ids)
//   x    = x + swiglu(LN(x; ffn_norm) @ fc1) @ fc2
//
// esmcTower hands every hidden state to its caller - the embedding, each block's, and the last one after the final
// LayerNorm - which is what ESMFold2's shim mixes (cuda/ef2/src/shim.cuh). Weights under "c/" (the ESM-C
// bundle, or ESM-C 6B's resident int8 codes); its float32 building blocks are cuda/ef2's ops.cuh.
#pragma once
#include "../ef2/src/ops.cuh"

__global__ void embedK(const int* __restrict__ ids, const float* __restrict__ table, float* __restrict__ x, int rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)table | (uintptr_t)x) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = (size_t)rows * c4;
    if (t < n4) {
      int r = (int)(t / (size_t)c4), c = (int)(t % (size_t)c4);
      reinterpret_cast<float4*>(x)[t] = __ldg(reinterpret_cast<const float4*>(table + (size_t)ids[r] * C) + c);
    }
    return;
  }
  if (t < (size_t)rows * C) x[t] = table[(size_t)ids[t / C] * C + t % C];
}
// Precomputed exact pow(10000.0, -(2*d)/64.0) for d = 0..31 (avoids libdevice pow(double, double) in ropeK).
__device__ __constant__ double ESMC_INV_FREQ[32] = {
  0x1.0000000000000p+0,  0x1.7ff2224115d9ap-1,  0x1.1feb33c1c381ep-1,  0x1.afd1354c40d50p-2,
  0x1.43d136248490fp-2,  0x1.e5a84719edcd2p-3,  0x1.6c310e3769f3fp-3,  0x1.111aedafb9a9dp-3,
  0x1.999999999999ap-4,  0x1.33281b6744ae1p-4,  0x1.ccab8602d2697p-5,  0x1.59742aa36710dp-5,
  0x1.030dc4ea03a72p-5,  0x1.84869f47f170fp-6,  0x1.235a71c5ee5ccp-6,  0x1.b4f7e2b2c2a95p-7,
  0x1.47ae147ae147bp-7,  0x1.eb735f0ba1168p-8,  0x1.7089380241edfp-8,  0x1.145ceee91f40bp-8,
  0x1.9e7c6e43390b7p-9,  0x1.36d219065ac0cp-9,  0x1.d22a4fa316facp-10, 0x1.5d931bc235544p-10,
  0x1.0624dd2f1a9fcp-10, 0x1.892918d61a787p-11, 0x1.26d42cce9b24cp-11, 0x1.ba2e4b0e98677p-12,
  0x1.4b96be9c2da2cp-12, 0x1.f150280a2ace0p-13, 0x1.74eea61c12623p-13, 0x1.17a8e301c4436p-13
};
// rotary, split halves: channel d with d + 32 of each 64-wide head, position = row
__global__ void ropeK(float* x, int rows, int heads, int ld) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * heads * 32) return;
  int d = t & 31, h = (t >> 5) % heads, row = t / (32 * heads);
  float s, c;
  fastSincosD((double)row * ESMC_INV_FREQ[d], &s, &c);
  float* p = x + (size_t)row * ld + h * 64 + d;
  float a = p[0], b = p[32];
  p[0] = a * c - b * s;
  p[32] = a * s + b * c;
}
__global__ void esmcRopeTableK(float2* __restrict__ cosSin, int rows) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * 32) return;
  int d = t & 31, row = t >> 5;
  float s, c;
  fastSincosD((double)row * ESMC_INV_FREQ[d], &s, &c);
  cosSin[t] = make_float2(c, s);
}
// Fused LayerNorm(q; q_norm) + LayerNorm(k; k_norm) + RoPE(q, k) over C = heads * 64 (MAX_K = 2 * heads).
// Lane d = 0..31 owns both h * 64 + d (step 2*h) and h * 64 + 32 + d (step 2*h + 1), so RoPE pairs stay in-lane.
template <int MAX_K>
__global__ void __launch_bounds__(256, 2)
qkNormRopeK(const float* __restrict__ qkv, float* __restrict__ qOut, float* __restrict__ kOut,
            size_t rows, const float* __restrict__ qScale, const float* __restrict__ kScale,
            const float2* __restrict__ cosSin, float eps) {
  constexpr int C = MAX_K * 32, HEADS = MAX_K / 2;
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  int isK = blockIdx.y;
  const float* xr = qkv + row * (3 * C) + isK * C;
  float* yr = (isK ? kOut : qOut) + row * C;
  const float* scale = isK ? kScale : qScale;
  float xv[MAX_K];
  float s = 0;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    float val = __ldg(xr + k * 32 + lane);
    xv[k] = val;
    s += val;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s * (1.f / (float)C), v = 0;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    float d = xv[k] - mean;
    v += d * d;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v * (1.f / (float)C) + eps);
  float2 cs = __ldg(cosSin + row * 32 + lane);
  #pragma unroll
  for (int h = 0; h < HEADS; ++h) {
    int c0 = h * 64 + lane, c1 = c0 + 32;
    float a = (xv[2 * h] - mean) * inv * __ldg(scale + c0);
    float b = (xv[2 * h + 1] - mean) * inv * __ldg(scale + c1);
    yr[c0] = a * cs.x - b * cs.y;
    yr[c1] = a * cs.y + b * cs.x;
  }
}
// softmax over a score row, keys of another chain (sequence id) excluded
__global__ void softmaxSeqK(float* __restrict__ S, const int* __restrict__ seq, int rows, float scale) {
  size_t row = (size_t)blockIdx.x;          // (head, query)
  int q = (int)(row % (size_t)rows);
  int sq = __ldg(seq + q);
  float* s = S + row * rows;
  __shared__ float red[32];
  if (rows <= 1024) {
    float sv[4];
    float m = -INFINITY;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int j = threadIdx.x + k * 256;
      if (j < rows) {
        float v = __ldg(seq + j) == sq ? s[j] * scale : -INFINITY;
        sv[k] = v;
        m = fmaxf(m, v);
      }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
      float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
      #pragma unroll
      for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o));
      if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    m = red[0];
    float sum = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int j = threadIdx.x + k * 256;
      if (j < rows) {
        float e = expf(sv[k] - m);
        sv[k] = e;
        sum += e;
      }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
      float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
      #pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (threadIdx.x == 0) red[0] = v;
    }
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
  for (int j = threadIdx.x; j < rows; j += blockDim.x) {
    float v = __ldg(seq + j) == sq ? s[j] * scale : -INFINITY;
    s[j] = v; m = fmaxf(m, v);
  }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  m = red[0];
  __syncthreads();
  float sum = 0;
  for (int j = threadIdx.x; j < rows; j += blockDim.x) { float e = expf(s[j] - m); s[j] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < rows; j += blockDim.x) s[j] *= inv;
}

struct Esmc { int rows, model, heads, ffn, layers, pair; float residualScale = 1.f; };

// --fast: the tower's four matrices a block run on their f16 mirror, and their f32 copy is dropped from
// the device (compactTowerWeights): 2.2 GB of the 2.9 GB of weights, never read again in f32
inline bool TOWER16 = true;      // --no-tower16: the tower's f32 copies kept, TF32 GEMMs
inline bool towerHalf(const std::string& name) {
  if (!TOWER16 || name.rfind("c/blocks/", 0)) return false;
  for (const char* m : {"/qkv/weights", "/attn_out/weights", "/fc1/weights", "/fc2/weights"})
    if (name.size() > strlen(m) && !name.compare(name.size() - strlen(m), strlen(m), m)) return true;
  return false;
}
// a RESIDENT int8 matrix (ESM-C 6B, tools/export_esmc6b.py): [out, in] codes with a float16 scale a row, expanded
// to float16 in scratch for one GEMM - the tower is 6.4 GB as codes and 12.7 GB as float16, which no T4 holds
__global__ void expandRowsInt8K(const signed char* __restrict__ q, const __half* __restrict__ scale,
                                __half* __restrict__ w, int block) {
  int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (col < block) {
    size_t idx = (size_t)row * block + col;
    w[idx] = __float2half(__half2float(__ldg(scale + row)) * (float)__ldg(q + idx));
  }
}
inline void towerGemm(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f,
                      float alpha = 1.f) {
  if (M.isResident("c/" + w + "T")) {
    ResidentInt8 r = M.residentInt8("c/" + w + "T");
    if (r.scales32) { fprintf(stderr, "c/%sT: the resident ESM-C tower reads int8 codes, a scale a row\n", w.c_str()); exit(1); }
    if (r.elements != (size_t)in * out || r.block != in) { fprintf(stderr, "c/%sT is not [%d, %d] a row a scale\n", w.c_str(), out, in); exit(1); }
    __half* w16 = scratch<__half>("tower.w16", r.elements);
    expandRowsInt8K<<<dim3((unsigned)((in + 255) / 256), (unsigned)out), 256, 0, STREAM>>>(r.codes, r.scales, w16, in);
    __half* xh = scratch<__half>("gemmh.x", rows * in);
    toHalfK<<<blocks(rows * in), 256, 0, STREAM>>>(X, xh, rows * in);
    CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha, w16, CUDA_R_16F, in, xh, CUDA_R_16F, in, &beta,
                    Y, CUDA_R_32F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return;
  }
  // (alpha: the residual scale, on every path - it was the resident one's alone, and 300M's 0.9129 then exited here)
  if (FAST && TOWER16) gemmH(X, Wh("c/" + w), Y, rows, in, out, beta, alpha);
  else gemm(X, Cw(w), Y, rows, in, out, beta, 0, 0, alpha);
}

inline void esmcBlock(const Esmc& e, float* x, const int* seq, int layer, const float2* cosSin = nullptr) {
  std::string B = "blocks/" + std::to_string(layer) + "/";
  size_t R = e.rows; int C = e.model;
  float* xn = scratch<float>("esmc.xn", R * C);
  float* qkv = scratch<float>("esmc.qkv", R * 3 * C);
  layerNorm(x, xn, R, C, Cw(B + "attn_norm/scale"), Cw(B + "attn_norm/offset"));
  towerGemm(xn, B + "qkv/weights", qkv, R, C, 3 * C);
  float* q = scratch<float>("esmc.q", R * C); float* k = scratch<float>("esmc.k", R * C);
  const float* qScale = Cw(B + "q_norm/scale");
  const float* kScale = Cw(B + "k_norm/scale");
  if (cosSin && C == 960 && e.heads == 15) {
    qkNormRopeK<30><<<dim3((unsigned)((R + 7) / 8), 2), 256, 0, STREAM>>>(qkv, q, k, R, qScale, kScale, cosSin, 1e-5f);
  } else if (cosSin && C == 1152 && e.heads == 18) {
    qkNormRopeK<36><<<dim3((unsigned)((R + 7) / 8), 2), 256, 0, STREAM>>>(qkv, q, k, R, qScale, kScale, cosSin, 1e-5f);
  } else {
    layerNorm(qkv, q, R, C, qScale, nullptr, 1e-5f, 3 * C, C);
    layerNorm(qkv + C, k, R, C, kScale, nullptr, 1e-5f, 3 * C, C);
    ropeK<<<blocks((size_t)R * e.heads * 32), 256, 0, STREAM>>>(q, (int)R, e.heads, C);
    ropeK<<<blocks((size_t)R * e.heads * 32), 256, 0, STREAM>>>(k, (int)R, e.heads, C);
  }
  float* ctx = scratch<float>("esmc.ctx", R * C);
  // v stays in the packed buffer (row stride 3C); q and k were normalised out to stride C
  {
    // the heads a group at a time where every head's scores would not fit with room to spare - each head its own, so the
    // same arithmetic (ESM-C 6B's 40 heads at 16,683 rows were 44.5 GB of scores at once)
    int hg = e.heads;
    while (hg > 1 && !roomFor((size_t)hg * R * R * 4, { "esmc.scores" })) hg = (hg + 1) / 2;
    float* S = scratch<float>("esmc.scores", (size_t)hg * R * R);
    const float one = 1.f, zero = 0.f;
    for (int h0 = 0; h0 < e.heads; h0 += hg) {
      const int g = std::min(hg, e.heads - h0);
      CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, (int)R, (int)R, 64, &one, k + h0 * 64, C, 64, q + h0 * 64, C, 64, &zero,
                                   S, (int)R, (long long)R * R, g));
      softmaxSeqK<<<(unsigned)(g * R), 256, 0, STREAM>>>(S, seq, (int)R, 0.125f);
      CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, 64, (int)R, (int)R, &one, qkv + 2 * C + h0 * 64, 3 * C, 64, S,
                                   (int)R, (long long)R * R, &zero, ctx + h0 * 64, C, 64, g));
    }
  }
  towerGemm(ctx, B + "attn_out/weights", x, R, C, C, 1.f, 1.f / e.residualScale);   // x + f(x) / sqrt(layers / 36)
  layerNorm(x, xn, R, C, Cw(B + "ffn_norm/scale"), Cw(B + "ffn_norm/offset"));
  float* h = scratch<float>("esmc.h", R * 2 * e.ffn); float* g = scratch<float>("esmc.g", R * e.ffn);
  towerGemm(xn, B + "fc1/weights", h, R, C, 2 * e.ffn);
  swigluK<<<blocks((e.ffn & 3) == 0 ? R * (size_t)(e.ffn >> 2) : R * e.ffn), 256, 0, STREAM>>>(h, g, R, e.ffn);
  towerGemm(g, B + "fc2/weights", x, R, e.ffn, C, 1.f, 1.f / e.residualScale);
}

// the tower over ids [rows] (seq: each row's chain, attention stays within one): every hidden state handed to
// onState(k, x) - k 0 the embedding, k 1..layers-1 the blocks', k layers the last block's after final_norm
inline void esmcTower(const Esmc& e, const int* ids, const int* seq, const std::function<void(int, const float*)>& onState) {
  size_t R = e.rows; int C = e.model;
  float* x = scratch<float>("esmc.x", R * C);
  embedK<<<blocks((C & 3) == 0 ? R * (size_t)(C >> 2) : R * C), 256, 0, STREAM>>>(ids, W("c/embed/weights"), x, (int)R, C);
  onState(0, x);
  float2* cosSin = reinterpret_cast<float2*>(scratch<float>("esmc.rope", R * 64));
  esmcRopeTableK<<<blocks(R * 32), 256, 0, STREAM>>>(cosSin, (int)R);
  float* last = scratch<float>("esmc.last", R * C);
  for (int l = 0; l < e.layers; ++l) {
    esmcBlock(e, x, seq, l, cosSin);
    if (l + 1 < e.layers) onState(l + 1, x);
  }
  layerNorm(x, last, R, C, W("c/final_norm/scale"), nullptr);
  onState(e.layers, last);
}
