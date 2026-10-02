// ESM-C's tower and ESMFold2's language-model shim (src/esmc/tower-reference.js is the reading):
//
//   h    = LN(x; attn_norm) @ qkv -> [q | k | v]
//   q, k = LN(q; q_norm), LN(k; k_norm)      over the FULL width, no offset, before the heads exist
//   q, k = RoPE(q, k)                         head 64, base 10000, split halves, position = row
//   x    = x + attn(q, k, v) @ attn_out       (attention within one chain: sequence ids)
//   x    = x + swiglu(LN(x; ffn_norm) @ fc1) @ fc2
//
// and ESMFold2's shim mixes all 37 states (the embedding, 36 blocks, the last one post the final
// LayerNorm): single = (sum_k softmax(combine)_k LN(h_k) @ projection) @ downproject + bias, then
// the pair [a*b | a-b] -> 512 -> gelu -> 256 -> 256 -> LN.
#pragma once
#include "ops.cuh"

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

struct Esmc { int rows, model, heads, ffn, layers, pair; };

inline void esmcBlock(const Esmc& e, float* x, const int* seq, int layer) {
  std::string B = "blocks/" + std::to_string(layer) + "/";
  size_t R = e.rows; int C = e.model;
  float* xn = scratch<float>("esmc.xn", R * C);
  float* qkv = scratch<float>("esmc.qkv", R * 3 * C);
  layerNorm(x, xn, R, C, Cw(B + "attn_norm/scale"), Cw(B + "attn_norm/offset"));
  gemm(xn, Cw(B + "qkv/weights"), qkv, R, C, 3 * C);
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
  gemm(ctx, Cw(B + "attn_out/weights"), x, R, C, C, 1.f);
  layerNorm(x, xn, R, C, Cw(B + "ffn_norm/scale"), Cw(B + "ffn_norm/offset"));
  float* h = scratch<float>("esmc.h", R * 2 * e.ffn); float* g = scratch<float>("esmc.g", R * e.ffn);
  gemm(xn, Cw(B + "fc1/weights"), h, R, C, 2 * e.ffn);
  swigluK<<<blocks(R * e.ffn), 256, 0, STREAM>>>(h, g, R, e.ffn);
  gemm(g, Cw(B + "fc2/weights"), x, R, e.ffn, C, 1.f);
}

__global__ void axpyK(float* y, const float* x, float a, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] += a * x[t];
}
// the tower's rows onto the tokens; a token the tower never saw takes the shim at a zero state
__global__ void scatterRowsK(const float* rows, const int* tokenToRow, const float* zero, float* out, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int token = (int)(t / C), c = (int)(t % C), r = tokenToRow[token];
  out[t] = r < 0 ? zero[c] : rows[(size_t)r * C + c];
}
__global__ void pairJoinK(const float* s, float* out, int T, int i0, int bi, int C) {   // [bi*T, 2C]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)bi * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = i0 + (int)(ij / T), j = (int)(ij % T);
  float a = s[(size_t)i * C + c], b = s[(size_t)j * C + c];
  out[ij * 2 * C + c] = a * b;
  out[ij * 2 * C + C + c] = a - b;
}
__global__ void biasGeluK(float* y, const float* b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) y[t] = geluF(y[t] + b[t % C]);
}

// the tower and the shim: ids [rows] -> the language model's pair term lm_z [T, T, pair]
// onState(k, x): every one of the 37 hidden states, for a check
inline void languageModel(const Esmc& e, const int* ids, const int* seq, const int* tokenToRow, int T, float* lmZ,
                          const std::function<void(int, const float*)>& onState) {
  size_t R = e.rows; int C = e.model, P = e.pair;
  // no protein token (a nucleic or ligand-only input): the tower has nothing to read, and every token
  // takes the shim at a zero state - below, through tokenToRow's -1
  float* single = scratch<float>("shim.single", std::max<size_t>(R, 1) * P);
  if (R > 0) {
  float* x = scratch<float>("esmc.x", R * C);
  embedK<<<blocks(R * C), 256, 0, STREAM>>>(ids, Cw("embed/weights"), x, (int)R, C);
  // the mix weights, a constant
  std::vector<float> mix(M.len("c/lm/combine"));
  { const float* cmb = M.f("c/lm/combine"); float mx = -INFINITY, s = 0;
    for (size_t i = 0; i < mix.size(); ++i) mx = std::max(mx, cmb[i]);
    for (size_t i = 0; i < mix.size(); ++i) { mix[i] = expf(cmb[i] - mx); s += mix[i]; }
    for (auto& m : mix) m /= s; }
  float* acc = scratch<float>("shim.acc", R * P); CK(cudaMemsetAsync(acc, 0, R * P * 4, STREAM));
  float* xn = scratch<float>("shim.xn", R * C); float* proj = scratch<float>("shim.proj", R * P);
  auto mixIn = [&](int k, const float* state) {
    if (onState) onState(k, state);
    layerNorm(state, xn, R, C, Cw("lm/norm/scale"), Cw("lm/norm/offset"));
    gemm(xn, Cw("lm/projection/weights"), proj, R, C, P);
    axpyK<<<blocks(R * P), 256, 0, STREAM>>>(acc, proj, mix[k], R * P);
  };
  mixIn(0, x);
  float* last = scratch<float>("esmc.last", R * C);
  for (int l = 0; l < e.layers; ++l) {
    esmcBlock(e, x, seq, l);
    if (l + 1 < e.layers) mixIn(l + 1, x);
  }
  layerNorm(x, last, R, C, Cw("final_norm/scale"), nullptr);
  mixIn(e.layers, last);
  gemm(acc, Cw("lm/downproject/weights"), single, R, P, P);
  addBias(single, Cw("lm/downproject/bias"), R, P);
  }
  // a non-protein token's state is zero: LN(0) is the offset, the mix sums to one
  float* zero = scratch<float>("shim.zero", P);
  float* z1 = scratch<float>("shim.z1", P);
  gemm(Cw("lm/norm/offset"), Cw("lm/projection/weights"), z1, 1, C, P);
  gemm(z1, Cw("lm/downproject/weights"), zero, 1, P, P);
  addBias(zero, Cw("lm/downproject/bias"), 1, P);
  float* s = scratch<float>("shim.tokens", (size_t)T * P);
  scatterRowsK<<<blocks((size_t)T * P), 256, 0, STREAM>>>(single, tokenToRow, zero, s, T, P);
  // the pair, a block of rows at a time
  int bi = std::max(1, std::min(T, (int)(((size_t)32 << 20) / ((size_t)T * 2 * P))));
  float* join = scratch<float>("shim.join", (size_t)bi * T * 2 * P); float* hid = scratch<float>("shim.hid", (size_t)bi * T * P);
  for (int i0 = 0; i0 < T; i0 += bi) {
    int b = std::min(bi, T - i0); size_t cells = (size_t)b * T;
    pairJoinK<<<blocks(cells * P), 256, 0, STREAM>>>(s, join, T, i0, b, P);
    gemm(join, Cw("lm/pair_mlp_1/weights"), hid, cells, 2 * P, P);
    biasGeluK<<<blocks(cells * P), 256, 0, STREAM>>>(hid, Cw("lm/pair_mlp_1/bias"), cells, P);
    float* o = lmZ + (size_t)i0 * T * P;
    gemm(hid, Cw("lm/pair_mlp_2/weights"), o, cells, P, P);
    addBias(o, Cw("lm/pair_mlp_2/bias"), cells, P);
    layerNorm(o, o, cells, P, Cw("lm/pair_norm/scale"), Cw("lm/pair_norm/offset"));
  }
}
