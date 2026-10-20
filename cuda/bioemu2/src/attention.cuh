// BioEmu2's two attentions and its conditioned transformer layer (network/bioemu2.py's _window_attention,
// _attention and _conditioned_layer), several samples stacked on the rows.
#pragma once
#include "layers.cuh"

namespace be2 {

// ---------------------------------------------------------------- attentions
// AttentionPairBiasSlidingWindow: a block per (window, head), a thread per query; online softmax over the 128
// keys. qkv [rows, 3D] as the vendor fuses it: q (+ q_bias) at [0, D), then k at [D, 2D) and v at [2D, 3D) - its
// (2, heads, hd) reshape of the key/value half. out [rows, D], head-major. Several samples are stacked on the
// rows, N atoms each: a window's keys stay inside its sample, and every sample reads one bias.
template <int HD>
__global__ void windowAttentionK(const float* qkv, const float* qBias, const float* bias, const int* pad, float* out,
                                 int N, int H) {
  const int wg = blockIdx.x, h = blockIdx.y, i = threadIdx.x, D = H * HD;
  const int NW = N / WQ, smp = wg / NW, w = wg % NW;
  const size_t base = (size_t)smp * N;
  __shared__ float sk[WK][HD], sv[WK][HD];
  __shared__ int sm[WK];
  for (int e = threadIdx.x; e < WK * HD; e += blockDim.x) {
    int j = e / HD, d = e % HD, ka = w * WQ - CTX + j;
    bool ok = ka >= 0 && ka < N;
    sk[j][d] = ok ? qkv[(base + ka) * 3 * D + D + h * HD + d] : 0.f;
    sv[j][d] = ok ? qkv[(base + ka) * 3 * D + 2 * D + h * HD + d] : 0.f;
  }
  for (int j = threadIdx.x; j < WK; j += blockDim.x) { int ka = w * WQ - CTX + j; sm[j] = ka >= 0 && ka < N && pad[ka]; }
  __syncthreads();
  const size_t qa = base + w * WQ + i;
  float q[HD];
  const float scale = rsqrtf((float)HD);
  for (int d = 0; d < HD; ++d) q[d] = (qkv[qa * 3 * D + h * HD + d] + qBias[h * HD + d]) * scale;
  const float* b = bias + (((size_t)w * H + h) * WQ + i) * WK;
  float mx = -INFINITY, sum = 0, acc[HD];
  for (int d = 0; d < HD; ++d) acc[d] = 0;
  for (int j = 0; j < WK; ++j) {
    float l = 0; for (int d = 0; d < HD; ++d) l += q[d] * sk[j][d];
    l = sm[j] ? l + b[j] : NEG_INF;
    float nm = fmaxf(mx, l), f = expf(mx - nm), e = expf(l - nm);
    sum = sum * f + e;
    for (int d = 0; d < HD; ++d) acc[d] = acc[d] * f + e * sv[j][d];
    mx = nm;
  }
  for (int d = 0; d < HD; ++d) out[qa * D + h * HD + d] = acc[d] / sum;
}

// ...the same with four lanes a query, each a quarter of the 128 keys (its own online softmax), merged by shuffles:
// 128 threads a (window, head) block where the form above has 32 - 9.5 ms of an 82 ms step at 494 tokens
template <int HD>
__global__ void windowAttention4K(const float* qkv, const float* qBias, const float* bias, const int* pad, float* out,
                                  int N, int H) {
  constexpr int SPLIT = 4, KS = WK / SPLIT;
  const int wg = blockIdx.x, h = blockIdx.y, D = H * HD;
  const int NW = N / WQ, smp = wg / NW, w = wg % NW;
  const int i = threadIdx.x / SPLIT, part = threadIdx.x % SPLIT;
  const size_t base = (size_t)smp * N;
  __shared__ float sk[WK][HD + 1], sv[WK][HD + 1];
  __shared__ int sm[WK];
  for (int e = threadIdx.x; e < WK * HD; e += blockDim.x) {
    int j = e / HD, d = e % HD, ka = w * WQ - CTX + j;
    bool ok = ka >= 0 && ka < N;
    sk[j][d] = ok ? qkv[(base + ka) * 3 * D + D + h * HD + d] : 0.f;
    sv[j][d] = ok ? qkv[(base + ka) * 3 * D + 2 * D + h * HD + d] : 0.f;
  }
  for (int j = threadIdx.x; j < WK; j += blockDim.x) { int ka = w * WQ - CTX + j; sm[j] = ka >= 0 && ka < N && pad[ka]; }
  __syncthreads();
  const size_t qa = base + w * WQ + i;
  float q[HD];
  const float scale = rsqrtf((float)HD);
  for (int d = 0; d < HD; ++d) q[d] = (qkv[qa * 3 * D + h * HD + d] + qBias[h * HD + d]) * scale;
  const float* b = bias + (((size_t)w * H + h) * WQ + i) * WK;
  float mx = -INFINITY, sum = 0, acc[HD];
  for (int d = 0; d < HD; ++d) acc[d] = 0;
  for (int jj = 0; jj < KS; ++jj) {
    const int j = jj * SPLIT + part;            // (interleaved: the four lanes read neighbouring keys)
    float l = 0;
#pragma unroll
    for (int d = 0; d < HD; ++d) l += q[d] * sk[j][d];
    l = sm[j] ? l + b[j] : NEG_INF;
    float nm = fmaxf(mx, l), f = expf(mx - nm), e = expf(l - nm);
    sum = sum * f + e;
#pragma unroll
    for (int d = 0; d < HD; ++d) acc[d] = acc[d] * f + e * sv[j][d];
    mx = nm;
  }
  // the four parts' online softmaxes merged (lanes 4i .. 4i + 3 of a warp)
#pragma unroll
  for (int o = 1; o < SPLIT; o <<= 1) {
    float om = __shfl_xor_sync(~0u, mx, o), os = __shfl_xor_sync(~0u, sum, o);
    float nm = fmaxf(mx, om), f = expf(mx - nm), fo = expf(om - nm);
    sum = sum * f + os * fo;
#pragma unroll
    for (int d = 0; d < HD; ++d) { float oa = __shfl_xor_sync(~0u, acc[d], o); acc[d] = acc[d] * f + oa * fo; }
    mx = nm;
  }
  if (part == 0) for (int d = 0; d < HD; ++d) out[qa * D + h * HD + d] = acc[d] / sum;
}

// bias + key mask + softmax over the last axis of [H, T, T] logits, in place; a warp a row
__global__ void softmaxBiasK(float* logits, const float* bias, const int* keyMask, size_t rows, int T) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  float* l = logits + row * T; const float* b = bias + row * T;
  float mx = -INFINITY;
  for (int j = lane; j < T; j += 32) { float v = keyMask[j] ? l[j] + b[j] : NEG_INF; l[j] = v; mx = fmaxf(mx, v); }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  float s = 0;
  for (int j = lane; j < T; j += 32) { float e = expf(l[j] - mx); l[j] = e; s += e; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float inv = 1.f / s;
  for (int j = lane; j < T; j += 32) l[j] *= inv;
}
// AttentionPairBias over all T tokens of each of `samples` stacked samples: qkv [samples * T, 3D] (q, k, v
// contiguous thirds), bias [H, T, T] shared; out [samples * T, D]
inline void tokenAttention(const float* qkv, const float* qBias, const float* bias, float* out, int T, int D, int heads,
                           int samples = 1) {
  const int HD = D / heads;
  float* q = scratch<float>("be2.ta.q", (size_t)samples * T * D);
  CK(cudaMemcpy2DAsync(q, D * 4, qkv, 3 * D * 4, D * 4, (size_t)samples * T, cudaMemcpyDeviceToDevice, STREAM));
  addBiasRows(q, qBias, (size_t)samples * T, D);
  float* logits = scratch<float>("be2.ta.logits", (size_t)heads * T * T);
  const float alpha = 1.f / sqrtf((float)HD), zero = 0.f, one = 1.f;
  for (int s = 0; s < samples; ++s) {
    const float* qs = q + (size_t)s * T * D; const float* kv = qkv + (size_t)s * T * 3 * D;
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HD, &alpha, kv + D, 3 * D, HD, qs, D, HD, &zero,
                                 logits, T, (long long)T * T, heads));
    softmaxBiasK<<<(unsigned)(((size_t)heads * T + 7) / 8), 256, 0, STREAM>>>(logits, bias, Idev("token_pad_mask"),
                                                                              (size_t)heads * T, T);
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, HD, T, T, &one, kv + 2 * D, 3 * D, HD, logits, T,
                                 (long long)T * T, &zero, out + (size_t)s * T * D, D, HD, heads));
  }
}

// ---------------------------------------------------------------- the conditioned transformer layer
// ConditionedTransformerLayer (adaLN-Zero), in place on x [R, D]:
//   cond = cond_2(silu(LN(c))) -> alpha_a, beta_a, gamma_a, alpha_m, beta_m, gamma_m
//   x += gamma_a * attention((1 + alpha_a) LN(x) + beta_a);  x += gamma_m * transition((1 + alpha_m) LN(x) + beta_m)
// `windowed`: the sliding-window atom attention over the windowed pair's bias; else the full token attention over
// [H, T, T]. `att` names the attention's parameters, `l` the layer of a stacked module (-1: unstacked).
// 🔴 THE CONDITIONING IS COMPUTED ON ITS OWN ROWS AND MAPPED: it has Rc rows and `map` sends each of x's R rows to
// one (null: the identity). The denoiser's atom stream is conditioned on its token's row of the single conditioning
// broadcast to atoms, so its cond_2 is the same for every atom of a token - and every sample of a step: computed once
// a token rather than once an atom of every sample, and exact, because a broadcast row is the row. (A padding atom
// reads row 0; nothing reads a padding atom's result.)
struct CondLayer { std::string p, att; int l; bool windowed; int heads; };
inline const float* lw(const CondLayer& L, const std::string& n) { return L.l < 0 ? P(L.p + n) : PL(L.p + n, L.l); }
inline const float* lwOpt(const CondLayer& L, const std::string& n) {
  return L.l < 0 ? Popt(L.p + n) : PLopt(L.p + n, L.l);
}
// a layer's conditioning, [Rc, 6D]
inline float* layerCond(const CondLayer& L, const float* c, int Rc, int Cc, int D, const char* name) {
  float* cn = scratch<float>("be2.cl.cn", (size_t)Rc * Cc);
  layerNorm(c, cn, Rc, Cc, lw(L, "cond_0/weight"), lwOpt(L, "cond_0/bias"));
  siluK<<<blocks((size_t)Rc * Cc), 256, 0, STREAM>>>(cn, cn, (size_t)Rc * Cc);
  float* cond = scratch<float>(name, (size_t)Rc * 6 * D);
  linear(cn, lw(L, "cond_2/weight"), cond, Rc, Cc, 6 * D);
  if (const float* b = lwOpt(L, "cond_2/bias")) addBiasRows(cond, b, Rc, 6 * D);
  return cond;
}
// the layer on x [R, D] (samples stacked, `perSample` rows each), its conditioning rows `cond` reached through `map`
// GEMM_HALF: every GEMM input written in f16 by the kernel before it (adaLN, the gated attention output)
inline void condLayerHalf(const CondLayer& L, float* x, int R, int D, const float* cond, const int* map, const float* bias,
                          int perSample) {
  const std::string a = L.att;
  half* hh = scratch<half>("be2.cl.hh", (size_t)R * D);
  adaLn<half>(x, hh, R, D, cond, cond + D, 6 * D, 0, 1.f, map);
  float* qkv = scratch<float>("be2.cl.qkv", (size_t)R * 3 * D);
  float* g = scratch<float>("be2.cl.g", (size_t)R * D);
  float* o = scratch<float>("be2.cl.o", (size_t)R * D);
  linearPre<half>(hh, halfMirror<half>(lw(L, a + "proj_qkv/weight"), (size_t)3 * D * D), qkv, R, D, 3 * D);
  linearPre<half>(hh, halfMirror<half>(lw(L, a + "proj_g/weight"), (size_t)D * D), g, R, D, D);
  if (L.windowed) {
    windowAttention4K<32><<<dim3(R / WQ, L.heads), WQ * 4, 0, STREAM>>>(qkv, lw(L, a + "proj_q_bias"), bias,
                                                                   Idev("atom_pad_mask"), o, perSample, L.heads);
  } else {
    tokenAttention(qkv, lw(L, a + "proj_q_bias"), bias, o, perSample, D, L.heads, R / perSample);
  }
  sigmoidMulToK<<<blocks((size_t)R * D), 256, 0, STREAM>>>(o, g, hh, (size_t)R * D);
  float* h = scratch<float>("be2.cl.h", (size_t)R * D);
  linearPre<half>(hh, halfMirror<half>(lw(L, a + "proj_o/weight"), (size_t)D * D), h, R, D, D);
  gatedAdd(x, h, cond + 2 * D, R, D, 6 * D, map);
  adaLn<half>(x, hh, R, D, cond + 3 * D, cond + 4 * D, 6 * D, 0, 1.f, map);
  float* t = scratch<float>("be2.cl.t", (size_t)R * D);
  transitionHalf(nullptr, hh, t, R, D, L.p + "transition/", L.l, false, false, 0);
  gatedAdd(x, t, cond + 5 * D, R, D, 6 * D, map);
}
inline void condLayer(const CondLayer& L, float* x, int R, int D, const float* cond, const int* map, const float* bias,
                      int perSample) {
  if (GEMM_HALF) { condLayerHalf(L, x, R, D, cond, map, bias, perSample); return; }
  float* h = scratch<float>("be2.cl.h", (size_t)R * D);
  adaLn(x, h, R, D, cond, cond + D, 6 * D, 0, 1.f, map);
  float* qkv = scratch<float>("be2.cl.qkv", (size_t)R * 3 * D);
  float* g = scratch<float>("be2.cl.g", (size_t)R * D);
  float* o = scratch<float>("be2.cl.o", (size_t)R * D);
  const std::string a = L.att;
  linear(h, lw(L, a + "proj_qkv/weight"), qkv, R, D, 3 * D);
  linear(h, lw(L, a + "proj_g/weight"), g, R, D, D);
  if (L.windowed) {
    if (D / L.heads != 32) { fprintf(stderr, "BioEmu2: an atom head of %d (this port is written for 32)\n", D / L.heads); exit(1); }
    windowAttention4K<32><<<dim3(R / WQ, L.heads), WQ * 4, 0, STREAM>>>(qkv, lw(L, a + "proj_q_bias"), bias,
                                                                   Idev("atom_pad_mask"), o, perSample, L.heads);
  } else {
    tokenAttention(qkv, lw(L, a + "proj_q_bias"), bias, o, perSample, D, L.heads, R / perSample);
  }
  sigmoidMulK<<<blocks((size_t)R * D), 256, 0, STREAM>>>(o, g, o, (size_t)R * D);
  linear(o, lw(L, a + "proj_o/weight"), h, R, D, D);
  gatedAdd(x, h, cond + 2 * D, R, D, 6 * D, map);
  adaLn(x, h, R, D, cond + 3 * D, cond + 4 * D, 6 * D, 0, 1.f, map);
  float* t = scratch<float>("be2.cl.t", (size_t)R * D);
  transition(h, t, R, D, L.p + "transition/", L.l, false, false);
  gatedAdd(x, t, cond + 5 * D, R, D, 6 * D, map);
}
// ...on its own conditioning, a row of c a row of x (the input embedder's)
inline void condLayer(const CondLayer& L, float* x, int R, int D, const float* c, int Cc, const float* bias) {
  float* cond = layerCond(L, c, R, Cc, D, "be2.cl.cond");
  condLayer(L, x, R, D, cond, nullptr, bias, R);
}

}  // namespace be2
