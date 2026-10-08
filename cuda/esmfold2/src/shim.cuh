// ESMFold2's language-model shim over ESM-C's tower (cuda/plm/esmc.cuh): it mixes all 37 hidden states (the
// embedding, 36 blocks, the last one post the final LayerNorm): single = (sum_k softmax(combine)_k LN(h_k) @
// projection) @ downproject + bias, then the pair [a*b | a-b] -> 512 -> gelu -> 256 -> 256 -> LN.
#pragma once
#include "../../plm/esmc.cuh"

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

// the language model's pair term for pair rows [i0, i0 + b) (i of [T, T, pair]), from the per-token states
// s [T, P]: join, the pair MLP, its LayerNorm - in blocks of `bi` rows, the size languageModel uses
inline int lmPairBlock(int T, int P) { return std::max(1, std::min(T, (int)(((size_t)32 << 20) / ((size_t)T * 2 * P)))); }
inline void lmPairRows(const float* s, int T, int P, int i0, int b, float* o) {
  int bi = lmPairBlock(T, P);
  float* join = scratch<float>("shim.join", (size_t)bi * T * 2 * P); float* hid = scratch<float>("shim.hid", (size_t)bi * T * P);
  size_t cells = (size_t)b * T;
  pairJoinK<<<blocks(cells * P), 256, 0, STREAM>>>(s, join, T, i0, b, P);
  gemm(join, Cw("lm/pair_mlp_1/weights"), hid, cells, 2 * P, P);
  biasGeluK<<<blocks(cells * P), 256, 0, STREAM>>>(hid, Cw("lm/pair_mlp_1/bias"), cells, P);
  gemm(hid, Cw("lm/pair_mlp_2/weights"), o, cells, P, P);
  addBias(o, Cw("lm/pair_mlp_2/bias"), cells, P);
  layerNorm(o, o, cells, P, Cw("lm/pair_norm/scale"), Cw("lm/pair_norm/offset"));
}
// the tower and the shim: ids [rows] -> the language model's pair term lm_z [T, T, pair] (null lmZ: only
// the per-token states, in scratch "shim.tokens" - a streamed z_init makes the pair as it needs it)
// onState(k, x): every one of the 37 hidden states, for a check
inline void languageModel(const Esmc& e, const int* ids, const int* seq, const int* tokenToRow, int T, float* lmZ,
                          const std::function<void(int, const float*)>& onState) {
  size_t R = e.rows; int C = e.model, P = e.pair;
  // no protein token (a nucleic or ligand-only input): the tower has nothing to read, and every token
  // takes the shim at a zero state - below, through tokenToRow's -1
  float* single = scratch<float>("shim.single", std::max<size_t>(R, 1) * P);
  if (R > 0) {
  // the mix weights, a constant
  std::vector<float> mix(M.len(shimKey("lm/combine")));
  { const float* cmb = M.f(shimKey("lm/combine")); float mx = -INFINITY, s = 0;
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
  esmcTower(e, ids, seq, mixIn);
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
  if (!lmZ) return;
  int bi = lmPairBlock(T, P);
  for (int i0 = 0; i0 < T; i0 += bi) lmPairRows(s, T, P, i0, std::min(bi, T - i0), lmZ + (size_t)i0 * T * P);
}
