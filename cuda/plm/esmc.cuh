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

__global__ void embedK(const int* ids, const float* table, float* x, int rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)rows * C) x[t] = table[(size_t)ids[t / C] * C + t % C];
}
// rotary, split halves: channel d with d + 32 of each 64-wide head, position = row
__global__ void ropeK(float* x, int rows, int heads, int ld) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * heads * 32) return;
  int d = t % 32, h = (t / 32) % heads, row = t / (32 * heads);
  double s, c; sincos((double)row * pow(10000.0, -(double)(2 * d) / 64.0), &s, &c);
  float* p = x + (size_t)row * ld + h * 64 + d;
  float a = p[0], b = p[32];
  p[0] = a * (float)c - b * (float)s;
  p[32] = a * (float)s + b * (float)c;
}
// softmax over a score row, keys of another chain (sequence id) excluded
__global__ void softmaxSeqK(float* S, const int* seq, int rows, float scale) {
  size_t row = (size_t)blockIdx.x;          // (head, query)
  int q = (int)(row % rows);
  float* s = S + row * rows;
  __shared__ float red[32];
  float m = -INFINITY;
  for (int j = threadIdx.x; j < rows; j += blockDim.x) {
    float v = seq[j] == seq[q] ? s[j] * scale : -INFINITY;
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
__global__ void expandRowsInt8K(const signed char* q, const __half* scale, __half* w, size_t n, int block) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) w[t] = __float2half(__half2float(scale[t / block]) * (float)q[t]);
}
inline void towerGemm(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f,
                      float alpha = 1.f) {
  if (M.isResident("c/" + w + "T")) {
    ResidentInt8 r = M.residentInt8("c/" + w + "T");
    if (r.scales32) { fprintf(stderr, "c/%sT: the resident ESM-C tower reads int8 codes, a scale a row\n", w.c_str()); exit(1); }
    if (r.elements != (size_t)in * out || r.block != in) { fprintf(stderr, "c/%sT is not [%d, %d] a row a scale\n", w.c_str(), out, in); exit(1); }
    __half* w16 = scratch<__half>("tower.w16", r.elements);
    expandRowsInt8K<<<blocks(r.elements), 256, 0, STREAM>>>(r.codes, r.scales, w16, r.elements, r.block);
    __half* xh = scratch<__half>("gemmh.x", rows * in);
    toHalfK<<<blocks(rows * in), 256, 0, STREAM>>>(X, xh, rows * in);
    CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha, w16, CUDA_R_16F, in, xh, CUDA_R_16F, in, &beta,
                    Y, CUDA_R_32F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return;
  }
  if (alpha != 1.f) { fprintf(stderr, "towerGemm: a residual scale needs the resident path\n"); exit(1); }
  if (FAST && TOWER16) gemmH(X, Wh("c/" + w), Y, rows, in, out, beta);
  else gemm(X, Cw(w), Y, rows, in, out, beta);
}

inline void esmcBlock(const Esmc& e, float* x, const int* seq, int layer) {
  std::string B = "blocks/" + std::to_string(layer) + "/";
  size_t R = e.rows; int C = e.model;
  float* xn = scratch<float>("esmc.xn", R * C);
  float* qkv = scratch<float>("esmc.qkv", R * 3 * C);
  layerNorm(x, xn, R, C, Cw(B + "attn_norm/scale"), Cw(B + "attn_norm/offset"));
  towerGemm(xn, B + "qkv/weights", qkv, R, C, 3 * C);
  float* q = scratch<float>("esmc.q", R * C); float* k = scratch<float>("esmc.k", R * C);
  layerNorm(qkv, q, R, C, Cw(B + "q_norm/scale"), nullptr, 1e-5f, 3 * C, C);
  layerNorm(qkv + C, k, R, C, Cw(B + "k_norm/scale"), nullptr, 1e-5f, 3 * C, C);
  ropeK<<<blocks((size_t)R * e.heads * 32), 256, 0, STREAM>>>(q, (int)R, e.heads, C);
  ropeK<<<blocks((size_t)R * e.heads * 32), 256, 0, STREAM>>>(k, (int)R, e.heads, C);
  float* ctx = scratch<float>("esmc.ctx", R * C);
  // v stays in the packed buffer (row stride 3C); q and k were normalised out to stride C
  {
    float* S = scratch<float>("esmc.scores", (size_t)e.heads * R * R);
    const float one = 1.f, zero = 0.f;
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, (int)R, (int)R, 64, &one, k, C, 64, q, C, 64, &zero,
                                 S, (int)R, (long long)R * R, e.heads));
    softmaxSeqK<<<(unsigned)(e.heads * R), 256, 0, STREAM>>>(S, seq, (int)R, 0.125f);
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, 64, (int)R, (int)R, &one, qkv + 2 * C, 3 * C, 64, S,
                                 (int)R, (long long)R * R, &zero, ctx, C, 64, e.heads));
  }
  towerGemm(ctx, B + "attn_out/weights", x, R, C, C, 1.f, 1.f / e.residualScale);   // x + f(x) / sqrt(layers / 36)
  layerNorm(x, xn, R, C, Cw(B + "ffn_norm/scale"), Cw(B + "ffn_norm/offset"));
  float* h = scratch<float>("esmc.h", R * 2 * e.ffn); float* g = scratch<float>("esmc.g", R * e.ffn);
  towerGemm(xn, B + "fc1/weights", h, R, C, 2 * e.ffn);
  swigluK<<<blocks(R * e.ffn), 256, 0, STREAM>>>(h, g, R, e.ffn);
  towerGemm(g, B + "fc2/weights", x, R, e.ffn, C, 1.f, 1.f / e.residualScale);
}

// the tower over ids [rows] (seq: each row's chain, attention stays within one): every hidden state handed to
// onState(k, x) - k 0 the embedding, k 1..layers-1 the blocks', k layers the last block's after final_norm
inline void esmcTower(const Esmc& e, const int* ids, const int* seq, const std::function<void(int, const float*)>& onState) {
  size_t R = e.rows; int C = e.model;
  float* x = scratch<float>("esmc.x", R * C);
  embedK<<<blocks(R * C), 256, 0, STREAM>>>(ids, W("c/embed/weights"), x, (int)R, C);
  onState(0, x);
  float* last = scratch<float>("esmc.last", R * C);
  for (int l = 0; l < e.layers; ++l) {
    esmcBlock(e, x, seq, l);
    if (l + 1 < e.layers) onState(l + 1, x);
  }
  layerNorm(x, last, R, C, W("c/final_norm/scale"), nullptr);
  onState(e.layers, last);
}
