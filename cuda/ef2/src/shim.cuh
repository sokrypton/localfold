// ESMFold2's language-model shim over ESM-C's tower (cuda/plm/esmc.cuh): it mixes all 37 hidden states (the
// embedding, 36 blocks, the last one post the final LayerNorm): single = (sum_k softmax(combine)_k LN(h_k) @
// projection) @ downproject + bias, then the pair [a*b | a-b] -> 512 -> gelu -> 256 -> 256 -> LN.
#pragma once
#include "../../plm/esmc.cuh"

__global__ void axpyK(float* __restrict__ y, const float* __restrict__ x, float a, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)y | (uintptr_t)x) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 yv = reinterpret_cast<const float4*>(y)[t];
      float4 xv = __ldg(reinterpret_cast<const float4*>(x) + t);
      reinterpret_cast<float4*>(y)[t] = make_float4(yv.x + a * xv.x, yv.y + a * xv.y, yv.z + a * xv.z, yv.w + a * xv.w);
    }
    return;
  }
  if (t < n) y[t] += a * x[t];
}
// the tower's rows onto the tokens; a token the tower never saw takes the shim at a zero state
__global__ void scatterRowsK(const float* __restrict__ rows, const int* __restrict__ tokenToRow, const float* __restrict__ zero,
                             float* __restrict__ out, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)rows | (uintptr_t)zero | (uintptr_t)out) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = (size_t)T * c4;
    if (t >= n4) return;
    int token = (int)(t / (size_t)c4), c = (int)(t % (size_t)c4), r = tokenToRow[token];
    reinterpret_cast<float4*>(out)[t] = r < 0 ? __ldg(reinterpret_cast<const float4*>(zero) + c)
                                              : __ldg(reinterpret_cast<const float4*>(rows + (size_t)r * C) + c);
    return;
  }
  if (t >= (size_t)T * C) return;
  int token = (int)(t / C), c = (int)(t % C), r = tokenToRow[token];
  out[t] = r < 0 ? zero[c] : rows[(size_t)r * C + c];
}
__global__ void pairJoinK(const float* __restrict__ s, float* __restrict__ out, int T, int i0, int bi, int C) {   // [bi*T, 2C]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)s | (uintptr_t)out) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = (size_t)bi * T * c4;
    if (t >= n4) return;
    int c; size_t ij; int i, j;
    if (n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, uc = (unsigned)c4, ut = (unsigned)T;
      unsigned q = ((uc & (uc - 1)) == 0) ? (u >> (__ffs(c4) - 1)) : (u / uc);
      c = (int)(u - q * uc);
      ij = q;
      unsigned qi = q / ut;
      i = i0 + (int)qi;
      j = (int)(q - qi * ut);
    } else {
      c = (int)(t % c4); ij = t / c4; i = i0 + (int)(ij / T); j = (int)(ij % T);
    }
    float4 a = __ldg(reinterpret_cast<const float4*>(s + (size_t)i * C) + c);
    float4 b = __ldg(reinterpret_cast<const float4*>(s + (size_t)j * C) + c);
    float4* dst = reinterpret_cast<float4*>(out + ij * 2 * C);
    dst[c] = make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
    dst[c4 + c] = make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
    return;
  }
  if (t >= (size_t)bi * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = i0 + (int)(ij / T), j = (int)(ij % T);
  float a = s[(size_t)i * C + c], b = s[(size_t)j * C + c];
  out[ij * 2 * C + c] = a * b;
  out[ij * 2 * C + C + c] = a - b;
}
__global__ void biasGeluK(float* __restrict__ y, const float* __restrict__ b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t >= n4) return;
    int c = ((c4 & (c4 - 1)) == 0) ? (int)(t & (size_t)(c4 - 1)) : (int)(t % (size_t)c4);
    float4 yv = reinterpret_cast<const float4*>(y)[t];
    float4 bv = __ldg(reinterpret_cast<const float4*>(b) + c);
    reinterpret_cast<float4*>(y)[t] = make_float4(geluF(yv.x + bv.x), geluF(yv.y + bv.y), geluF(yv.z + bv.z), geluF(yv.w + bv.w));
    return;
  }
  if (t < rows * C) y[t] = geluF(y[t] + b[t % C]);
}

// the language model's pair term for pair rows [i0, i0 + b) (i of [T, T, pair]), from the per-token states
// s [T, P]: join, the pair MLP, its LayerNorm - in blocks of `bi` rows, the size languageModel uses
inline int lmPairBlock(int T, int P) { return std::max(1, std::min(T, (int)(((size_t)32 << 20) / ((size_t)T * 2 * P)))); }
inline void lmPairRows(const float* s, int T, int P, int i0, int b, float* o) {
  int bi = lmPairBlock(T, P);
  float* join = scratch<float>("shim.join", (size_t)bi * T * 2 * P); float* hid = scratch<float>("shim.hid", (size_t)bi * T * P);
  size_t cells = (size_t)b * T;
  size_t nVec = (P & 3) == 0 ? cells * (size_t)(P >> 2) : cells * P;
  pairJoinK<<<blocks(nVec), 256, 0, STREAM>>>(s, join, T, i0, b, P);
  gemm(join, Cw("lm/pair_mlp_1/weights"), hid, cells, 2 * P, P);
  biasGeluK<<<blocks(nVec), 256, 0, STREAM>>>(hid, Cw("lm/pair_mlp_1/bias"), cells, P);
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
    size_t n = R * P;
    axpyK<<<blocks((n & 3) == 0 ? (n >> 2) : n), 256, 0, STREAM>>>(acc, proj, mix[k], n);
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
  scatterRowsK<<<blocks((P & 3) == 0 ? (size_t)T * (P >> 2) : (size_t)T * P), 256, 0, STREAM>>>(single, tokenToRow, zero, s, T, P);
  // the pair, a block of rows at a time
  if (!lmZ) return;
  int bi = lmPairBlock(T, P);
  for (int i0 = 0; i0 < T; i0 += bi) lmPairRows(s, T, P, i0, std::min(bi, T - i0), lmZ + (size_t)i0 * T * P);
}
