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
__global__ void pairTokenAddK(const float* __restrict__ zp, const int* __restrict__ atomTok,
                              const int* __restrict__ pad, float* __restrict__ pr, int N, int T, int CP) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK) return;
  int j = (int)(t & (WK - 1)); size_t qa = t / WK; int w = (int)(qa / WQ);
  int ka = w * WQ - CTX + j;
  if (ka < 0 || ka >= N || !pad[ka] || !pad[qa]) return;
  const float* z = zp + ((size_t)atomTok[qa] * T + atomTok[ka]) * CP;
  float* o = pr + t * CP;
  if (CP == 16 && (((uintptr_t)zp | (uintptr_t)pr) & 15) == 0) {
    const float4* z4 = reinterpret_cast<const float4*>(z);
    float4* o4 = reinterpret_cast<float4*>(o);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      float4 ov = o4[k], zv = __ldg(z4 + k);
      o4[k] = make_float4(ov.x + zv.x, ov.y + zv.y, ov.z + zv.z, ov.w + zv.w);
    }
    return;
  }
  for (int c = 0; c < CP; ++c) o[c] += z[c];
}
// _atom_pair_tail: pr += cq[q] + ck[k] (ck zero past the edge), then pr += p_mlp(pr), three ReLU-Linear layers
// (16 channels, no biases)
__global__ void pairTailK(float* __restrict__ pr, const float* __restrict__ cq, const float* __restrict__ ck,
                          const float* __restrict__ w1, const float* __restrict__ w3,
                          const float* __restrict__ w5, int N) {
  constexpr int CP = 16;
  __shared__ float4 sw4[3][CP * (CP / 4)];
  for (int k = threadIdx.x; k < CP * (CP / 4); k += blockDim.x) {
    sw4[0][k] = __ldg(reinterpret_cast<const float4*>(w1) + k);
    sw4[1][k] = __ldg(reinterpret_cast<const float4*>(w3) + k);
    sw4[2][k] = __ldg(reinterpret_cast<const float4*>(w5) + k);
  }
  __syncthreads();
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK) return;
  int j = (int)(t & (WK - 1)); size_t qa = t / WK; int w = (int)(qa / WQ);
  int ka = w * WQ - CTX + j;
  bool kOk = (ka >= 0 && ka < N);
  float4 p4[4], h4[4];
  float4* o4 = reinterpret_cast<float4*>(pr + t * CP);
  const float4* cq4 = reinterpret_cast<const float4*>(cq + qa * CP);
  const float4* ck4 = kOk ? reinterpret_cast<const float4*>(ck + (size_t)ka * CP) : nullptr;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    float4 ov = o4[k], qv = __ldg(cq4 + k);
    float4 kv = kOk ? __ldg(ck4 + k) : make_float4(0.f, 0.f, 0.f, 0.f);
    p4[k] = make_float4(ov.x + qv.x + kv.x, ov.y + qv.y + kv.y, ov.z + qv.z + kv.z, ov.w + qv.w + kv.w);
    h4[k] = p4[k];
  }
  #pragma unroll
  for (int l = 0; l < 3; ++l) {
    float4 in4[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      in4[k] = make_float4(fmaxf(h4[k].x, 0.f), fmaxf(h4[k].y, 0.f), fmaxf(h4[k].z, 0.f), fmaxf(h4[k].w, 0.f));
    }
    float out[CP];
    #pragma unroll
    for (int r = 0; r < CP; ++r) {
      float s = 0.f;
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        float4 wv = sw4[l][r * 4 + k];
        s += wv.x * in4[k].x + wv.y * in4[k].y + wv.z * in4[k].z + wv.w * in4[k].w;
      }
      out[r] = s;
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      h4[k] = make_float4(out[k * 4 + 0], out[k * 4 + 1], out[k * 4 + 2], out[k * 4 + 3]);
    }
  }
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    o4[k] = make_float4(p4[k].x + h4[k].x, p4[k].y + h4[k].y, p4[k].z + h4[k].z, p4[k].w + h4[k].w);
  }
}
inline void pairTail(const std::string& p, float* pr, const float* c, int N) {
  const int C = dimP(p + "c_to_p_trans_q_1/weight", 1), CP = dimP(p + "c_to_p_trans_q_1/weight", 0);
  if (CP != 16) { fprintf(stderr, "BioEmu2: an atom pair of %d channels (this port is written for 16)\n", CP); exit(1); }
  float* rc = scratch<float>("be2.pt.relu", (size_t)N * C);
  size_t nc = (((size_t)N * C & 3) == 0 && (((uintptr_t)c | (uintptr_t)rc) & 15) == 0) ? ((size_t)N * C) >> 2 : (size_t)N * C;
  reluK<<<blocks(nc), 256, 0, STREAM>>>(c, rc, (size_t)N * C);
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
__global__ void windowBiasK(const float* __restrict__ pr, const float* __restrict__ normW,
                            const float* __restrict__ projZ, float* __restrict__ out, int N, int H) {
  constexpr int CP = 16;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK) return;
  int j = (int)(t & (WK - 1)); size_t qa = t / WK; int w = (int)(qa / WQ), i = (int)(qa % WQ);
  const float4* x4 = reinterpret_cast<const float4*>(pr + t * CP);
  float4 v0 = __ldg(x4 + 0), v1 = __ldg(x4 + 1), v2 = __ldg(x4 + 2), v3 = __ldg(x4 + 3);
  float mean = (v0.x + v0.y + v0.z + v0.w + v1.x + v1.y + v1.z + v1.w
              + v2.x + v2.y + v2.z + v2.w + v3.x + v3.y + v3.z + v3.w) * (1.f / 16.f);
  float x[CP] = {v0.x - mean, v0.y - mean, v0.z - mean, v0.w - mean,
                 v1.x - mean, v1.y - mean, v1.z - mean, v1.w - mean,
                 v2.x - mean, v2.y - mean, v2.z - mean, v2.w - mean,
                 v3.x - mean, v3.y - mean, v3.z - mean, v3.w - mean};
  float var = 0.f;
  #pragma unroll
  for (int c = 0; c < CP; ++c) var += x[c] * x[c];
  float inv = rsqrtf(var * (1.f / 16.f) + 1e-5f);
  float n[CP];
  #pragma unroll
  for (int c = 0; c < CP; ++c) n[c] = x[c] * inv * __ldg(normW + c);
  for (int h = 0; h < H; ++h) {
    float s = 0.f;
    #pragma unroll
    for (int c = 0; c < CP; ++c) s += __ldg(projZ + h * CP + c) * n[c];
    out[(((size_t)w * H + h) * WQ + i) * WK + j] = s;
  }
}
inline void windowBias(const float* pr, const float* normW, const float* projZ, float* out, int N, int H) {
  windowBiasK<<<blocks((size_t)N * WK), 256, 0, STREAM>>>(pr, normW, projZ, out, N, H);
}
struct WinBiasPtrs {
  const float* normW[16];
  const float* projZ[16];
};
// Batched windowBias across `layers` (up to 16, H == 4, CP == 16): normalizes pr[t, 0..15] ONCE in registers and
// projects all `layers` using shared-memory cached normW [layers, 16] and projZ [layers, 4, 16].
__global__ void windowBiasBatchK(const float* __restrict__ pr, WinBiasPtrs ptrs,
                                 float* __restrict__ out, int N, int H, int layers) {
  constexpr int CP = 16;
  __shared__ float sNorm[16][CP];
  __shared__ float sProj[16][4 * CP];
  for (int idx = threadIdx.x; idx < layers * CP; idx += blockDim.x) {
    int l = idx / CP, c = idx % CP;
    sNorm[l][c] = __ldg(ptrs.normW[l] + c);
  }
  for (int idx = threadIdx.x; idx < layers * H * CP; idx += blockDim.x) {
    int l = idx / (H * CP), hc = idx % (H * CP);
    sProj[l][hc] = __ldg(ptrs.projZ[l] + hc);
  }
  __syncthreads();
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK) return;
  int j = (int)(t & (WK - 1)); size_t qa = t / WK; int w = (int)(qa / WQ), i = (int)(qa % WQ);
  const float4* x4 = reinterpret_cast<const float4*>(pr + t * CP);
  float4 v0 = __ldg(x4 + 0), v1 = __ldg(x4 + 1), v2 = __ldg(x4 + 2), v3 = __ldg(x4 + 3);
  float mean = (v0.x + v0.y + v0.z + v0.w + v1.x + v1.y + v1.z + v1.w
              + v2.x + v2.y + v2.z + v2.w + v3.x + v3.y + v3.z + v3.w) * (1.f / 16.f);
  float x[CP] = {v0.x - mean, v0.y - mean, v0.z - mean, v0.w - mean,
                 v1.x - mean, v1.y - mean, v1.z - mean, v1.w - mean,
                 v2.x - mean, v2.y - mean, v2.z - mean, v2.w - mean,
                 v3.x - mean, v3.y - mean, v3.z - mean, v3.w - mean};
  float var = 0.f;
  #pragma unroll
  for (int c = 0; c < CP; ++c) var += x[c] * x[c];
  float inv = rsqrtf(var * (1.f / 16.f) + 1e-5f);
  #pragma unroll
  for (int c = 0; c < CP; ++c) x[c] *= inv;
  const size_t wb = (size_t)N * WK * H;
  for (int l = 0; l < layers; ++l) {
    float n[CP];
    #pragma unroll
    for (int c = 0; c < CP; ++c) n[c] = x[c] * sNorm[l][c];
    float* outL = out + (size_t)l * wb;
    for (int h = 0; h < H; ++h) {
      float s = 0.f;
      #pragma unroll
      for (int c = 0; c < CP; ++c) s += sProj[l][h * CP + c] * n[c];
      outL[(((size_t)w * H + h) * WQ + i) * WK + j] = s;
    }
  }
}
inline void windowBiasBatch(const float* pr, const WinBiasPtrs& ptrs, float* out, int N, int H, int layers) {
  if (H == 4 && layers <= 16) {
    windowBiasBatchK<<<blocks((size_t)N * WK), 256, 0, STREAM>>>(pr, ptrs, out, N, H, layers);
    return;
  }
  const size_t wb = (size_t)N * WK * H;
  for (int l = 0; l < layers; ++l) windowBias(pr, ptrs.normW[l], ptrs.projZ[l], out + (size_t)l * wb, N, H);
}
// the token pair's bias: LN(z) (scale, no offset) -> proj_z [H, C], heads first: out[h][i][j]
__global__ void tokenBiasK(const float* zn, const float* biasHC, float* out, int T, int H, size_t r0 = 0, size_t rows = 0) {
  // zn is already projected: [rows, H]
  size_t P2 = (size_t)T * T;
  if (!rows) rows = P2;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * H) return;
  int h = (int)(t % H); size_t ij = r0 + t / H;
  out[(size_t)h * P2 + ij] = zn[t];
}
inline void tokenBias(const float* z, const float* normW, const float* projZ, float* out, int T, int C, int H) {
  size_t P2 = (size_t)T * T;
  size_t chunk = std::min<size_t>(P2, (size_t)256 << 10);
  float* zn = scratch<float>("be2.tb.zn", chunk * C);
  float* proj = scratch<float>("be2.tb.proj", chunk * H);
  for (size_t r0 = 0; r0 < P2; r0 += chunk) {
    size_t r = std::min(chunk, P2 - r0);
    layerNorm(z + r0 * C, zn, r, C, normW, nullptr);
    linear(zn, projZ, proj, r, C, H);
    tokenBiasK<<<blocks(r * H), 256, 0, STREAM>>>(proj, nullptr, out, T, H, r0, r);
  }
}
// Fused 16-layer tokenBias under GEMM_HALF: since norm_z has no bias offset, LN_l(z)_{ij,c} = z_hat_{ij,c} * norm_z[l, c].
// Pre-multiplying W_tb[l*H + h, c] = proj_z[l, h, c] * norm_z[l, c] lets us normalize d.z ONCE (affine-free into half)
// and project all 16 layers * 16 heads = 256 outputs in ONE [P, 256] x [256, 256] GEMM instead of 16 N=16 GEMMs.
__global__ void tokenBiasScaleProjHalfK(const float* __restrict__ normZ, const float* __restrict__ projZ,
                                        half* __restrict__ out, int layers, int H, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)layers * H * C) return;
  int c = (int)(t % C); size_t lh = t / C; int l = (int)(lh / H);
  out[t] = __float2half(projZ[t] * normZ[(size_t)l * C + c]);
}
__global__ void tokenBiasBatchK(const float* __restrict__ proj, float* __restrict__ out, size_t P2, int LH,
                                size_t r0 = 0, size_t rows = 0) {
  if (!rows) rows = P2;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * LH) return;
  int lh = (int)(t % LH); size_t ij = r0 + t / LH;
  out[(size_t)lh * P2 + ij] = proj[t];
}
inline void tokenBiasAll(const float* z, const float* normWAll, const float* projZAll,
                         float* out, int T, int C, int H, int layers) {
  size_t P2 = (size_t)T * T;
  if (!GEMM_HALF) {
    size_t tbn = (size_t)H * P2;
    for (int l = 0; l < layers; ++l)
      tokenBias(z, normWAll + (size_t)l * C, projZAll + (size_t)l * H * C, out + (size_t)l * tbn, T, C, H);
    return;
  }
  const int LH = layers * H;
  static std::map<const float*, half*> wCache;
  half* wh = nullptr;
  auto it = wCache.find(projZAll);
  if (it != wCache.end()) {
    wh = it->second;
  } else {
    wh = wpool<half>((size_t)LH * C);
    tokenBiasScaleProjHalfK<<<blocks((size_t)LH * C), 256, 0, STREAM>>>(normWAll, projZAll, wh, layers, H, C);
    wCache[projZAll] = wh;
  }
  size_t chunk = std::min<size_t>(P2, (size_t)256 << 10);
  half* znh = scratch<half>("be2.tb.znh", chunk * C);
  float* proj = scratch<float>("be2.tb.projAll", chunk * LH);
  for (size_t r0 = 0; r0 < P2; r0 += chunk) {
    size_t r = std::min(chunk, P2 - r0);
    lnTo<half>(z + r0 * C, znh, r, C, nullptr, nullptr);
    linearPre<half>(znh, wh, proj, r, C, LH);
    tokenBiasBatchK<<<blocks(r * LH), 256, 0, STREAM>>>(proj, out, P2, LH, r0, r);
  }
}


}  // namespace be2
