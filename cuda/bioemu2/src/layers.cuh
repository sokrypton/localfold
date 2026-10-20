// BioEmu2's atom side and its two attentions (network/bioemu2.py's _atom_features, _atom_pair_tail,
// _window_attention, _attention and _conditioned_layer).
//
// The atom windows: atoms are padded to a multiple of 32, a window is 32 query atoms, and its 128 keys are
// centred on it - key j of window w is atom w*32 - 48 + j, and a key outside [0, N) is padding (zero
// coordinates, uid -1, masked). Pair tensors over the windows are [NW, 32, 128, 16], row-major.
#pragma once
#include "net.cuh"

namespace be2 {

// ---------------------------------------------------------------- per-atom features
// c = proj_atom_continuous([ref, charge]) + embed_element[element] + sum_i embed_atom_name_chars_i[name[i]],
// zero on padding atoms
__global__ void atomFeaturesK(const float* ref, const float* charge, const int* element, const int* names,
                              const int* pad, const float* wCont, const float* wElem, const float* wN0,
                              const float* wN1, const float* wN2, const float* wN3, float* c, size_t N, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N * C) return;
  size_t a = t / C; int k = (int)(t % C);
  if (!pad[a]) { c[t] = 0.f; return; }
  const float in[4] = {ref[a * 3], ref[a * 3 + 1], ref[a * 3 + 2], charge[a]};
  float v = 0.f;
  for (int q = 0; q < 4; ++q) v += wCont[k * 4 + q] * in[q];
  // (an embedding's table is [rows, C]: row `index`)
  v += wElem[(size_t)element[a] * C + k];
  const float* tabs[4] = {wN0, wN1, wN2, wN3};
  for (int q = 0; q < 4; ++q) v += tabs[q][(size_t)names[a * 4 + q] * C + k];
  c[t] = v;
}
inline void atomFeatures(const std::string& p, float* c, size_t N) {
  const int C = dimP(p + "proj_atom_continuous/weight", 0);
  atomFeaturesK<<<blocks(N * C), 256, 0, STREAM>>>(Fdev("template_coords"), Fdev("charge"), Idev("element"),
      Idev("name_encoding"), Idev("atom_pad_mask"), P(p + "proj_atom_continuous/weight"), P(p + "embed_element/weight"),
      P(p + "embed_atom_name_chars_0/weight"), P(p + "embed_atom_name_chars_1/weight"),
      P(p + "embed_atom_name_chars_2/weight"), P(p + "embed_atom_name_chars_3/weight"), c, N, C);
}

// ---------------------------------------------------------------- the windowed atom pair
// windowed_atom_pair_geometry: pr = (W_pos disp + W_dist / (1 + |disp|^2) + W_mask same) * same, where disp is
// key minus query and `same` says both sit in one reference space (and the query is real)
__global__ void pairGeometryK(const float* ref, const int* uid, const int* pad, const float* wPos, const float* wDist,
                              const float* wMask, float* pr, int N, int CP) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)N * WK;
  if (t >= total) return;
  int j = (int)(t % WK); size_t qa = t / WK;       // (query atom qa = w * 32 + i)
  int w = (int)(qa / WQ);
  int ka = w * WQ - CTX + j;
  bool inRange = ka >= 0 && ka < N;
  float kx = inRange ? ref[ka * 3] : 0.f, ky = inRange ? ref[ka * 3 + 1] : 0.f, kz = inRange ? ref[ka * 3 + 2] : 0.f;
  float d[3] = {kx - ref[qa * 3], ky - ref[qa * 3 + 1], kz - ref[qa * 3 + 2]};
  float inv = 1.f / (1.f + d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  int ku = inRange && pad[ka] ? uid[ka] : -1;
  float same = pad[qa] && uid[qa] == ku ? 1.f : 0.f;
  float* o = pr + t * CP;
  for (int c = 0; c < CP; ++c)
    o[c] = (wPos[c * 3] * d[0] + wPos[c * 3 + 1] * d[1] + wPos[c * 3 + 2] * d[2] + wDist[c] * inv + wMask[c] * same) * same;
}
// broadcast_token_pair_to_atom_window: pr += zp[tok_q, tok_k], zero where either atom is padding
__global__ void pairTokenAddK(const float* zp, const int* atomTok, const int* pad, float* pr, int N, int T, int CP) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK) return;
  int j = (int)(t % WK); size_t qa = t / WK; int w = (int)(qa / WQ);
  int ka = w * WQ - CTX + j;
  if (ka < 0 || ka >= N || !pad[ka] || !pad[qa]) return;
  const float* z = zp + ((size_t)atomTok[qa] * T + atomTok[ka]) * CP;
  float* o = pr + t * CP;
  for (int c = 0; c < CP; ++c) o[c] += z[c];
}
// _atom_pair_tail: pr += cq[q] + ck[k] (ck zero past the edge), then pr += p_mlp(pr), three ReLU-Linear layers
// (16 channels, no biases)
__global__ void pairTailK(float* pr, const float* cq, const float* ck, const float* w1, const float* w3,
                          const float* w5, int N) {
  constexpr int CP = 16;
  __shared__ float sw[3][CP * CP];
  for (int k = threadIdx.x; k < CP * CP; k += blockDim.x) { sw[0][k] = w1[k]; sw[1][k] = w3[k]; sw[2][k] = w5[k]; }
  __syncthreads();
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK) return;
  int j = (int)(t % WK); size_t qa = t / WK; int w = (int)(qa / WQ);
  int ka = w * WQ - CTX + j;
  float p[CP], h[CP], g[CP];
  float* o = pr + t * CP;
  for (int c = 0; c < CP; ++c) p[c] = o[c] + cq[qa * CP + c] + (ka >= 0 && ka < N ? ck[(size_t)ka * CP + c] : 0.f);
  for (int l = 0; l < 3; ++l) {
    const float* src = l == 0 ? p : l == 1 ? h : g;
    float* dst = l == 0 ? h : l == 1 ? g : h;
    float in[CP]; for (int c = 0; c < CP; ++c) in[c] = fmaxf(src[c], 0.f);
    for (int r = 0; r < CP; ++r) { float s = 0; for (int c = 0; c < CP; ++c) s += sw[l][r * CP + c] * in[c]; dst[r] = s; }
  }
  for (int c = 0; c < CP; ++c) o[c] = p[c] + h[c];
}
inline void pairTail(const std::string& p, float* pr, const float* c, int N) {
  const int C = dimP(p + "c_to_p_trans_q_1/weight", 1), CP = dimP(p + "c_to_p_trans_q_1/weight", 0);
  if (CP != 16) { fprintf(stderr, "BioEmu2: an atom pair of %d channels (this port is written for 16)\n", CP); exit(1); }
  float* rc = scratch<float>("be2.pt.relu", (size_t)N * C);
  reluK<<<blocks((size_t)N * C), 256, 0, STREAM>>>(c, rc, (size_t)N * C);
  float* cq = scratch<float>("be2.pt.cq", (size_t)N * CP); float* ck = scratch<float>("be2.pt.ck", (size_t)N * CP);
  linear(rc, P(p + "c_to_p_trans_q_1/weight"), cq, N, C, CP);
  linear(rc, P(p + "c_to_p_trans_k_1/weight"), ck, N, C, CP);
  pairTailK<<<blocks((size_t)N * WK), 256, 0, STREAM>>>(pr, cq, ck, P(p + "p_mlp_1/weight"), P(p + "p_mlp_3/weight"),
                                                         P(p + "p_mlp_5/weight"), N);
}
// _atom_features' pair half: the geometry (no token term, no tail)
inline void pairGeometry(const std::string& p, float* pr, int N) {
  pairGeometryK<<<blocks((size_t)N * WK), 256, 0, STREAM>>>(Fdev("template_coords"), Idev("ref_space_uid"),
      Idev("atom_pad_mask"), P(p + "embed_atompair_ref_pos/weight"), P(p + "embed_atompair_ref_dist/weight"),
      P(p + "embed_atompair_mask/weight"), pr, N, dimP(p + "embed_atompair_ref_pos/weight", 0));
}

// ---------------------------------------------------------------- pair biases
// the windowed pair's bias, LN(pr) (scale, no offset) -> proj_z, heads first: out[w][h][i][j]
__global__ void windowBiasK(const float* pr, const float* normW, const float* projZ, float* out, int N, int H) {
  constexpr int CP = 16;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK) return;
  int j = (int)(t % WK); size_t qa = t / WK; int w = (int)(qa / WQ), i = (int)(qa % WQ);
  const float* x = pr + t * CP;
  float mean = 0; for (int c = 0; c < CP; ++c) mean += x[c]; mean /= CP;
  float var = 0; for (int c = 0; c < CP; ++c) { float d = x[c] - mean; var += d * d; }
  float inv = rsqrtf(var / CP + 1e-5f);
  float n[CP]; for (int c = 0; c < CP; ++c) n[c] = (x[c] - mean) * inv * normW[c];
  for (int h = 0; h < H; ++h) {
    float s = 0; for (int c = 0; c < CP; ++c) s += projZ[h * CP + c] * n[c];
    out[(((size_t)w * H + h) * WQ + i) * WK + j] = s;
  }
}
inline void windowBias(const float* pr, const float* normW, const float* projZ, float* out, int N, int H) {
  windowBiasK<<<blocks((size_t)N * WK), 256, 0, STREAM>>>(pr, normW, projZ, out, N, H);
}
// the token pair's bias: LN(z) (scale, no offset) -> proj_z [H, C], heads first: out[h][i][j]
__global__ void tokenBiasK(const float* zn, const float* biasHC, float* out, int T, int H) {
  // zn is already projected: [T*T, H]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * H) return;
  int h = (int)(t % H); size_t ij = t / H;
  out[(size_t)h * T * T + ij] = zn[t];
}
inline void tokenBias(const float* z, const float* normW, const float* projZ, float* out, int T, int C, int H) {
  size_t P2 = (size_t)T * T;
  float* zn = scratch<float>("be2.tb.zn", P2 * C);
  layerNorm(z, zn, P2, C, normW, nullptr);
  float* proj = scratch<float>("be2.tb.proj", P2 * H);
  linear(zn, projZ, proj, P2, C, H);
  tokenBiasK<<<blocks(P2 * H), 256, 0, STREAM>>>(proj, nullptr, out, T, H);
}

// ---------------------------------------------------------------- attentions
// AttentionPairBiasSlidingWindow: a block per (window, head), a thread per query; online softmax over the 128
// keys. qkv [N, 3D] as the vendor fuses it: q (+ q_bias) at [0, D), then k at [D, 2D) and v at [2D, 3D) - its
// (2, heads, hd) reshape of the key/value half. out [N, D], head-major.
template <int HD>
__global__ void windowAttentionK(const float* qkv, const float* qBias, const float* bias, const int* pad, float* out,
                                 int N, int H) {
  const int w = blockIdx.x, h = blockIdx.y, i = threadIdx.x, D = H * HD;
  __shared__ float sk[WK][HD], sv[WK][HD];
  __shared__ int sm[WK];
  for (int e = threadIdx.x; e < WK * HD; e += blockDim.x) {
    int j = e / HD, d = e % HD, ka = w * WQ - CTX + j;
    bool ok = ka >= 0 && ka < N;
    sk[j][d] = ok ? qkv[(size_t)ka * 3 * D + D + h * HD + d] : 0.f;
    sv[j][d] = ok ? qkv[(size_t)ka * 3 * D + 2 * D + h * HD + d] : 0.f;
  }
  for (int j = threadIdx.x; j < WK; j += blockDim.x) { int ka = w * WQ - CTX + j; sm[j] = ka >= 0 && ka < N && pad[ka]; }
  __syncthreads();
  const int qa = w * WQ + i;
  float q[HD];
  const float scale = rsqrtf((float)HD);
  for (int d = 0; d < HD; ++d) q[d] = (qkv[(size_t)qa * 3 * D + h * HD + d] + qBias[h * HD + d]) * scale;
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
  for (int d = 0; d < HD; ++d) out[(size_t)qa * D + h * HD + d] = acc[d] / sum;
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
// AttentionPairBias over all T tokens: qkv [T, 3D] (q, k, v contiguous thirds), bias [H, T, T]; out [T, D]
inline void tokenAttention(const float* qkv, const float* qBias, const float* bias, float* out, int T, int D, int heads) {
  const int HD = D / heads;
  float* q = scratch<float>("be2.ta.q", (size_t)T * D);
  CK(cudaMemcpy2DAsync(q, D * 4, qkv, 3 * D * 4, D * 4, T, cudaMemcpyDeviceToDevice, STREAM));
  addBiasRows(q, qBias, T, D);
  float* logits = scratch<float>("be2.ta.logits", (size_t)heads * T * T);
  const float alpha = 1.f / sqrtf((float)HD), zero = 0.f, one = 1.f;
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HD, &alpha, qkv + D, 3 * D, HD, q, D, HD, &zero,
                               logits, T, (long long)T * T, heads));
  softmaxBiasK<<<(unsigned)(((size_t)heads * T + 7) / 8), 256, 0, STREAM>>>(logits, bias, Idev("token_pad_mask"), (size_t)heads * T, T);
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, HD, T, T, &one, qkv + 2 * D, 3 * D, HD, logits, T,
                               (long long)T * T, &zero, out, D, HD, heads));
}

// ---------------------------------------------------------------- the conditioned transformer layer
// ConditionedTransformerLayer (adaLN-Zero), in place on x [R, D]:
//   cond = cond_2(silu(LN(c))) -> alpha_a, beta_a, gamma_a, alpha_m, beta_m, gamma_m
//   x += gamma_a * attention((1 + alpha_a) LN(x) + beta_a);  x += gamma_m * transition((1 + alpha_m) LN(x) + beta_m)
// `windowed`: the sliding-window atom attention over the windowed pair's bias; else the full token attention over
// [H, T, T]. `att` names the attention's parameters, `l` the layer of a stacked module (-1: unstacked).
struct CondLayer { std::string p, att; int l; bool windowed; int heads; };
inline const float* lw(const CondLayer& L, const std::string& n) { return L.l < 0 ? P(L.p + n) : PL(L.p + n, L.l); }
inline const float* lwOpt(const CondLayer& L, const std::string& n) {
  return L.l < 0 ? Popt(L.p + n) : PLopt(L.p + n, L.l);
}
inline void condLayer(const CondLayer& L, float* x, int R, int D, const float* c, int Cc, const float* bias) {
  float* cn = scratch<float>("be2.cl.cn", (size_t)R * Cc);
  layerNorm(c, cn, R, Cc, lw(L, "cond_0/weight"), lwOpt(L, "cond_0/bias"));
  siluK<<<blocks((size_t)R * Cc), 256, 0, STREAM>>>(cn, cn, (size_t)R * Cc);
  float* cond = scratch<float>("be2.cl.cond", (size_t)R * 6 * D);
  linear(cn, lw(L, "cond_2/weight"), cond, R, Cc, 6 * D);
  if (const float* b = lwOpt(L, "cond_2/bias")) addBiasRows(cond, b, R, 6 * D);
  float* h = scratch<float>("be2.cl.h", (size_t)R * D);
  adaLn(x, h, R, D, cond, cond + D, 6 * D);
  float* qkv = scratch<float>("be2.cl.qkv", (size_t)R * 3 * D);
  float* g = scratch<float>("be2.cl.g", (size_t)R * D);
  float* o = scratch<float>("be2.cl.o", (size_t)R * D);
  const std::string a = L.att;
  linear(h, lw(L, a + "proj_qkv/weight"), qkv, R, D, 3 * D);
  linear(h, lw(L, a + "proj_g/weight"), g, R, D, D);
  if (L.windowed) {
    if (D / L.heads != 32) { fprintf(stderr, "BioEmu2: an atom head of %d (this port is written for 32)\n", D / L.heads); exit(1); }
    windowAttentionK<32><<<dim3(R / WQ, L.heads), WQ, 0, STREAM>>>(qkv, lw(L, a + "proj_q_bias"), bias,
                                                                   Idev("atom_pad_mask"), o, R, L.heads);
  } else {
    tokenAttention(qkv, lw(L, a + "proj_q_bias"), bias, o, R, D, L.heads);
  }
  sigmoidMulK<<<blocks((size_t)R * D), 256, 0, STREAM>>>(o, g, o, (size_t)R * D);
  linear(o, lw(L, a + "proj_o/weight"), h, R, D, D);
  gatedAddK<<<blocks((size_t)R * D), 256, 0, STREAM>>>(x, h, cond + 2 * D, R, D, 6 * D);
  adaLn(x, h, R, D, cond + 3 * D, cond + 4 * D, 6 * D);
  float* t = scratch<float>("be2.cl.t", (size_t)R * D);
  transition(h, t, R, D, L.p + "transition/", L.l, false, false);
  gatedAddK<<<blocks((size_t)R * D), 256, 0, STREAM>>>(x, t, cond + 5 * D, R, D, 6 * D);
}

}  // namespace be2
