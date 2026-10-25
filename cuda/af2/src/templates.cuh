// AF2-multimer's template embedder (af3-any-model's TemplateEmbedding / SingleTemplateEmbedding /
// TemplateEmbeddingIteration), float32. It runs for a multimer checkpoint whether or not a template
// is given: the reference hands the network one blank template (af2/features.py blank_features), and
// the embedding of a blank template is not zero - its aatype and normalised-query terms survive.
#pragma once
#include "evoformer.cuh"

// The 9 template pair inputs, summed through their linears into act [L, L, 64]:
//   0 dgram(39) . 1 pseudo-beta mask . 2 aatype_j(22) . 3 aatype_i(22) . 4-6 unit vector . 7 backbone mask
//   (8, the normalised query, is a GEMM added after)
// tPos [L, 37, 3], tMask [L, 37], tAatype [L]; the multichain mask all ones (mask_template_interchain
// is off in the reference, which makes its where() keep everything)
__global__ void templateFramesK(const float* __restrict__ tPos, const float* __restrict__ tMask,
                                float* __restrict__ frames, float* __restrict__ bbMask, int L) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= L) return;
  const float* p = tPos + (size_t)r * 37 * 3;
  ptx_st_global_f32(bbMask + r, ptx_mul_f32(ptx_mul_f32(ptx_ldg_f32(tMask + (size_t)r * 37), ptx_ldg_f32(tMask + (size_t)r * 37 + 1)),
                                            ptx_ldg_f32(tMask + (size_t)r * 37 + 2)));
  float e0[3], e1[3], *R = frames + (size_t)r * 12, *t = R + 9;
  for (int k = 0; k < 3; ++k) {
    float p0 = ptx_ldg_f32(p + k), p3 = ptx_ldg_f32(p + 3 + k), p6 = ptx_ldg_f32(p + 6 + k);
    e0[k] = ptx_sub_f32(p6, p3); e1[k] = ptx_sub_f32(p0, p3); ptx_st_global_f32(t + k, p3);
  }
  float n0 = ptx_rsqrt_f32(ptx_max_f32(ptx_fma_f32(e0[0], e0[0], ptx_fma_f32(e0[1], e0[1], ptx_mul_f32(e0[2], e0[2]))), 1e-12f));
  for (int k = 0; k < 3; ++k) e0[k] = ptx_mul_f32(e0[k], n0);
  float c = ptx_fma_f32(e1[0], e0[0], ptx_fma_f32(e1[1], e0[1], ptx_mul_f32(e1[2], e0[2])));
  for (int k = 0; k < 3; ++k) e1[k] = ptx_fma_f32(-c, e0[k], e1[k]);
  float n1 = ptx_rsqrt_f32(ptx_max_f32(ptx_fma_f32(e1[0], e1[0], ptx_fma_f32(e1[1], e1[1], ptx_mul_f32(e1[2], e1[2]))), 1e-12f));
  for (int k = 0; k < 3; ++k) e1[k] = ptx_mul_f32(e1[k], n1);
  float e2[3] = {
    ptx_fma_f32(e0[1], e1[2], -ptx_mul_f32(e0[2], e1[1])),
    ptx_fma_f32(e0[2], e1[0], -ptx_mul_f32(e0[0], e1[2])),
    ptx_fma_f32(e0[0], e1[1], -ptx_mul_f32(e0[1], e1[0]))
  };
  // columns e0 e1 e2: R = [[e0x e1x e2x], [e0y e1y e2y], [e0z e1z e2z]]
  for (int k = 0; k < 3; ++k) {
    ptx_st_global_f32(R + k * 3, e0[k]);
    ptx_st_global_f32(R + k * 3 + 1, e1[k]);
    ptx_st_global_f32(R + k * 3 + 2, e2[k]);
  }
}
__global__ void templatePairInputK(const int* __restrict__ tAatype, const float* __restrict__ tPos, const float* __restrict__ tMask,
                                   const float* __restrict__ frames, const float* __restrict__ bbMask,
                                   const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2,
                                   const float* __restrict__ w3, const float* __restrict__ w4, const float* __restrict__ w5,
                                   const float* __restrict__ w6, const float* __restrict__ w7, const float* __restrict__ bsum,
                                   const float* __restrict__ qterm, float* __restrict__ act, int L, int C) {
  __shared__ float sw[6][64];
  if (threadIdx.x < 64) {
    int c = threadIdx.x;
    ptx_st_shared_f32(&sw[0][c], ptx_ldg_f32(bsum + c));
    ptx_st_shared_f32(&sw[1][c], ptx_ldg_f32(w1 + c));
    ptx_st_shared_f32(&sw[2][c], ptx_ldg_f32(w4 + c));
    ptx_st_shared_f32(&sw[3][c], ptx_ldg_f32(w5 + c));
    ptx_st_shared_f32(&sw[4][c], ptx_ldg_f32(w6 + c));
    ptx_st_shared_f32(&sw[5][c], ptx_ldg_f32(w7 + c));
  }
  __syncthreads();
  size_t ij = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (ij >= (size_t)L * L) return;
  int i = (int)(ij / L), j = (int)(ij % L);
  int rawAi = ptx_ldg_s32(tAatype + i), rawAj = ptx_ldg_s32(tAatype + j);
  int pi = rawAi == 7 ? 1 : 3, pj = rawAj == 7 ? 1 : 3;
  float pb2d = ptx_mul_f32(ptx_ldg_f32(tMask + (size_t)i * 37 + pi), ptx_ldg_f32(tMask + (size_t)j * 37 + pj));
  float d2 = 0.f;
  for (int k = 0; k < 3; ++k) {
    float d = ptx_sub_f32(ptx_ldg_f32(tPos + ((size_t)i * 37 + pi) * 3 + k), ptx_ldg_f32(tPos + ((size_t)j * 37 + pj) * 3 + k));
    d2 = ptx_fma_f32(d, d, d2);
  }
  int dbin = -1;
  if (pb2d != 0.f) {
    float lo0 = ptx_fma_f32((float)lane, (50.75f - 3.25f) / 38.f, 3.25f), hi0 = ptx_fma_f32((float)(lane + 1), (50.75f - 3.25f) / 38.f, 3.25f);
    uint32_t m0 = ptx_ballot_sync(d2 > ptx_mul_f32(lo0, lo0) && d2 < ptx_mul_f32(hi0, hi0));
    int b1 = 32 + lane;
    float lo1 = ptx_fma_f32((float)b1, (50.75f - 3.25f) / 38.f, 3.25f), hi1 = ptx_fma_f32((float)(b1 + 1), (50.75f - 3.25f) / 38.f, 3.25f);
    uint32_t m1 = ptx_ballot_sync(b1 < 39 && d2 > ptx_mul_f32(lo1, lo1) && d2 < (b1 + 1 < 39 ? ptx_mul_f32(hi1, hi1) : 1e8f));
    if (m0) dbin = ptx_ffs_u32(m0) - 1;
    else if (m1) dbin = 31 + ptx_ffs_u32(m1);
  }
  const float *Ri = frames + (size_t)i * 12, *ti = Ri + 9, *tj = frames + (size_t)j * 12 + 9;
  float d[3] = {
    ptx_sub_f32(ptx_ldg_f32(tj + 0), ptx_ldg_f32(ti + 0)),
    ptx_sub_f32(ptx_ldg_f32(tj + 1), ptx_ldg_f32(ti + 1)),
    ptx_sub_f32(ptx_ldg_f32(tj + 2), ptx_ldg_f32(ti + 2))
  }, v[3];
  for (int a = 0; a < 3; ++a)
    v[a] = ptx_fma_f32(ptx_ldg_f32(Ri + a), d[0], ptx_fma_f32(ptx_ldg_f32(Ri + 3 + a), d[1], ptx_mul_f32(ptx_ldg_f32(Ri + 6 + a), d[2])));
  float nv = ptx_rsqrt_f32(ptx_max_f32(ptx_fma_f32(v[0], v[0], ptx_fma_f32(v[1], v[1], ptx_mul_f32(v[2], v[2]))), 1e-12f));
  float bb = ptx_sqrt_f32(ptx_mul_f32(ptx_ldg_f32(bbMask + i), ptx_ldg_f32(bbMask + j)));
  float nv_bb = ptx_mul_f32(nv, bb);
  float f1 = ptx_mul_f32(v[0], nv_bb), f2 = ptx_mul_f32(v[1], nv_bb), f3 = ptx_mul_f32(v[2], nv_bb);
  int ai = ptx_min_s32(ptx_max_s32(rawAi, 0), 21), aj = ptx_min_s32(ptx_max_s32(rawAj, 0), 21);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    int c = lane + 32 * k;
    float a = ptx_add_f32(ptx_ld_shared_f32(&sw[0][c]), (dbin >= 0 ? ptx_mul_f32(pb2d, ptx_ldg_f32(w0 + dbin * C + c)) : 0.f));
    a = ptx_fma_f32(pb2d, ptx_ld_shared_f32(&sw[1][c]), a);
    a = ptx_add_f32(a, ptx_add_f32(ptx_ldg_f32(w2 + aj * C + c), ptx_ldg_f32(w3 + ai * C + c)));
    a = ptx_fma_f32(f1, ptx_ld_shared_f32(&sw[2][c]), a);
    a = ptx_fma_f32(f2, ptx_ld_shared_f32(&sw[3][c]), a);
    a = ptx_fma_f32(f3, ptx_ld_shared_f32(&sw[4][c]), a);
    a = ptx_fma_f32(bb, ptx_ld_shared_f32(&sw[5][c]), a);
    ptx_st_global_f32(act + ij * C + c, ptx_add_f32(a, ptx_ldg_f32(qterm + ij * C + c)));
  }
}
__global__ void sumBiasesK(float* out, const float* const* bs, int n, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s = 0.f; for (int k = 0; k < n; ++k) s = ptx_add_f32(s, ptx_ldg_f32(bs[k] + c));
  ptx_st_global_f32(out + c, s);
}
__global__ void reluScaleK(float* x, float s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) ptx_st_global_f32(x + t, ptx_relu_f32(ptx_mul_f32(ptx_ld_global_f32(x + t), s)));
}

// pair += TemplateEmbedding(pair, templates); T templates in t/aatype [T, L], t/positions, t/mask
// (none given: one blank template, as the reference's blank_features)
inline void templateEmbedding(float* pair, const float* pairMask, int L, bool pairOnes = false) {
  const std::string TE = "evoformer/template_embedding/", S = TE + "single_template_embedding/";
  const std::string IT = S + "template_embedding_iteration/";
  const int C = 64, Cz = 128;
  size_t pairs = (size_t)L * L;
  int T = (int)M.meta("meta/templates", 0);
  const int* tAatype; const float* tPos; const float* tMask;
  if (T == 0) {
    static int* zi = nullptr; static float* zp = nullptr; static float* zm = nullptr; static int zl = 0;
    if (zl < L) {
      zi = dallocT<int>(L); zp = dalloc((size_t)L * 37 * 3); zm = dalloc((size_t)L * 37);
      CK(cudaMemset(zi, 0, L * 4)); CK(cudaMemset(zp, 0, (size_t)L * 37 * 3 * 4)); CK(cudaMemset(zm, 0, (size_t)L * 37 * 4));
      zl = L;
    }
    tAatype = zi; tPos = zp; tMask = zm; T = 1;
  } else { tAatype = Idev("t/aatype"); tPos = W("t/positions"); tMask = W("t/mask"); }
  float* qterm = scratch<float>("tmpl.qterm", pairs * C);
  if (FAST) {
    half* qnh = scratch<half>("tmpl.qnh", pairs * Cz);
    WITH_PT(AF2_P16, layerNormVK<128, PT, half><<<lnVGrid(pairs), 256, 0, STREAM>>>(
      reinterpret_cast<const PT*>(pair), qnh, pairs, P(S + "query_embedding_norm/scale"), P(S + "query_embedding_norm/offset")));
    ltGemm(qnh, PH(S + "template_pair_embedding_8/weights"), qterm, false, pairs, Cz, C, nullptr, false, 0.f);
  } else {
    float* qn = scratch<float>("tmpl.qn", pairs * Cz);
    WITH_PT(AF2_P16, layerNormFastK<128, PT><<<lnVGrid(pairs), 256, 0, STREAM>>>(
      reinterpret_cast<const PT*>(pair), qn, pairs, P(S + "query_embedding_norm/scale"), P(S + "query_embedding_norm/offset")));
    gemm(qn, P(S + "template_pair_embedding_8/weights"), qterm, pairs, Cz, C);
  }
  // the nine linears' biases, summed once
  static float* bsum = nullptr;
  if (!bsum) {
    bsum = dalloc(C);
    std::vector<const float*> bs;
    for (int k = 0; k < 9; ++k) bs.push_back(P(S + "template_pair_embedding_" + std::to_string(k) + "/bias"));
    const float** dbs = (const float**)dallocT<float*>(9);
    CK(cudaMemcpy(dbs, bs.data(), 9 * sizeof(float*), cudaMemcpyHostToDevice));
    sumBiasesK<<<1, 64, 0, STREAM>>>(bsum, dbs, 9, C);
  }
  float* sum = (!FAST || T > 1) ? scratch<float>("tmpl.sum", pairs * C) : nullptr;
  float* act = scratch<float>("tmpl.act", pairs * C);
  float* actn = T > 1 ? scratch<float>("tmpl.actn", pairs * C) : nullptr;
  float* frames = scratch<float>("tmpl.frames", (size_t)L * 12);
  float* bbMask = scratch<float>("tmpl.bbm", L);
  auto w = [&](int k) { return P(S + "template_pair_embedding_" + std::to_string(k) + "/weights"); };
  for (int k = 0; k < T; ++k) {
    templateFramesK<<<blocks(L), 256, 0, STREAM>>>(tPos + (size_t)k * L * 37 * 3, tMask + (size_t)k * L * 37, frames, bbMask, L);
    templatePairInputK<<<(unsigned)((pairs + 7) / 8), 256, 0, STREAM>>>(tAatype + (size_t)k * L, tPos + (size_t)k * L * 37 * 3,
      tMask + (size_t)k * L * 37, frames, bbMask, w(0), w(1), w(2), w(3), w(4), w(5), w(6), w(7), bsum, qterm, act, L, C);
    int nb = (int)dimW(IT + "pair_transition/transition1/weights", 0);
    for (int b = 0; b < nb; ++b) {
      triangleMultiplication(act, pairMask, L, C, IT, b, true, pairOnes);
      triangleMultiplication(act, pairMask, L, C, IT, b, false, pairOnes);
      triangleAttention(act, pairMask, L, C, IT, b, true, pairOnes);
      triangleAttention(act, pairMask, L, C, IT, b, false, pairOnes);
      transition(act, pairs, C, IT + "pair_transition", b);
    }
    if (T == 1) {
      if (FAST) {
        half* sh = scratch<half>("tmpl.sh", pairs * C);
        layerNormVK<64, float, half, true><<<lnVGrid(pairs), 256, 0, STREAM>>>(
          act, sh, pairs, P(S + "output_layer_norm/scale"), P(S + "output_layer_norm/offset"));
        if (!AF2_P16) {
          ltGemm(sh, PH(TE + "output_linear/weights"), pair, false, pairs, C, Cz, P(TE + "output_linear/bias"), false, 1.f);
          return;
        }
        float* out = scratch<float>("tmpl.out", pairs * Cz);
        ltGemm(sh, PH(TE + "output_linear/weights"), out, false, pairs, C, Cz, nullptr, false, 0.f);
        addIntoPairK<__nv_bfloat16, float><<<blocks((pairs * Cz + 3) / 4), 256, 0, STREAM>>>(pair, out, pairs * Cz, P(TE + "output_linear/bias"));
        return;
      }
      layerNormFastK<64, float, true><<<lnVGrid(pairs), 256, 0, STREAM>>>(
        act, sum, pairs, P(S + "output_layer_norm/scale"), P(S + "output_layer_norm/offset"));
    } else {
      layerNorm(act, k == 0 ? sum : actn, pairs, C, S + "output_layer_norm");
      if (k > 0) addK2<<<blocks(pairs * (C >> 2)), 256, 0, STREAM>>>(sum, actn, pairs * C);
    }
  }
  if (T > 1) reluScaleK<<<blocks(pairs * C), 256, 0, STREAM>>>(sum, 1.f / T, pairs * C);
  float* out = scratch<float>("tmpl.out", pairs * Cz);
  gemm(sum, P(TE + "output_linear/weights"), out, pairs, C, Cz);
  WITH_PT(AF2_P16, addIntoPairK<PT, float><<<blocks((pairs * Cz + 3) / 4), 256, 0, STREAM>>>(pair, out, pairs * Cz, P(TE + "output_linear/bias")));
}

// ---------------------------------------------------------------- the monomer's template embedder
// (af3-any-model's MonomerTemplateEmbedding / MonomerSingleTemplateEmbedding / TemplatePairStack)
// The 88 template pair inputs - dgram(39) | mask_2d | aatype_j(22) | aatype_i(22) | unit vector(3, zeros:
// use_template_unit_vector is off in every shipped monomer config) | backbone mask - all multiplied by
// the backbone mask, through embedding2d (88 -> 64)
__global__ void templatePairInputMonomerK(const int* __restrict__ tAatype, const float* __restrict__ tPos,
                                          const float* __restrict__ tMask, const float* __restrict__ w,
                                          const float* __restrict__ b, float* __restrict__ act, int L, int C) {
  __shared__ float sb[64], sw39[64], sw87[64];
  if (threadIdx.x < 64) {
    int c = threadIdx.x;
    ptx_st_shared_f32(sb + c, ptx_ldg_f32(b + c));
    ptx_st_shared_f32(sw39 + c, ptx_ldg_f32(w + 39 * C + c));
    ptx_st_shared_f32(sw87 + c, ptx_ldg_f32(w + 87 * C + c));
  }
  __syncthreads();
  size_t ij = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (ij >= (size_t)L * L) return;
  int i = (int)(ij / L), j = (int)(ij % L);
  int rawAi = ptx_ldg_s32(tAatype + i), rawAj = ptx_ldg_s32(tAatype + j);
  int pi = rawAi == 7 ? 1 : 3, pj = rawAj == 7 ? 1 : 3;
  float m2 = ptx_mul_f32(ptx_ldg_f32(tMask + (size_t)i * 37 + pi), ptx_ldg_f32(tMask + (size_t)j * 37 + pj));
  float d2 = 0.f;
  for (int k = 0; k < 3; ++k) {
    float d = ptx_sub_f32(ptx_ldg_f32(tPos + ((size_t)i * 37 + pi) * 3 + k), ptx_ldg_f32(tPos + ((size_t)j * 37 + pj) * 3 + k));
    d2 = ptx_fma_f32(d, d, d2);
  }
  float lo0 = ptx_fma_f32((float)lane, (50.75f - 3.25f) / 38.f, 3.25f), hi0 = ptx_fma_f32((float)(lane + 1), (50.75f - 3.25f) / 38.f, 3.25f);
  uint32_t m0 = ptx_ballot_sync(d2 > ptx_mul_f32(lo0, lo0) && d2 < ptx_mul_f32(hi0, hi0));
  int b1 = 32 + lane;
  float lo1 = ptx_fma_f32((float)b1, (50.75f - 3.25f) / 38.f, 3.25f), hi1 = ptx_fma_f32((float)(b1 + 1), (50.75f - 3.25f) / 38.f, 3.25f);
  uint32_t m1 = ptx_ballot_sync(b1 < 39 && d2 > ptx_mul_f32(lo1, lo1) && d2 < (b1 + 1 < 39 ? ptx_mul_f32(hi1, hi1) : 1e8f));
  int dbin = m0 ? (ptx_ffs_u32(m0) - 1) : (m1 ? (31 + ptx_ffs_u32(m1)) : -1);
  auto bbm = [&](int r) {
    return ptx_mul_f32(ptx_mul_f32(ptx_ldg_f32(tMask + (size_t)r * 37), ptx_ldg_f32(tMask + (size_t)r * 37 + 1)),
                       ptx_ldg_f32(tMask + (size_t)r * 37 + 2));
  };
  float bb = ptx_mul_f32(bbm(i), bbm(j));
  int ai = ptx_min_s32(ptx_max_s32(rawAi, 0), 21), aj = ptx_min_s32(ptx_max_s32(rawAj, 0), 21);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    int c = lane + 32 * k;
    float a = ptx_fma_f32(m2, ptx_ld_shared_f32(sw39 + c), (dbin >= 0 ? ptx_ldg_f32(w + dbin * C + c) : 0.f));
    a = ptx_add_f32(a, ptx_add_f32(ptx_ldg_f32(w + (40 + aj) * C + c), ptx_ldg_f32(w + (62 + ai) * C + c)));
    a = ptx_add_f32(a, ptx_ld_shared_f32(sw87 + c));
    ptx_st_global_f32(act + ij * C + c, ptx_fma_f32(a, bb, ptx_ld_shared_f32(sb + c)));
  }
}
// the pointwise attention from the query pair over the templates: q [pairs, H*D] (scaled), k, v
// [T, pairs, H*D]; out [pairs, H*D] = softmax_t(q . k_t) v_t  (no gating; every template present)
__global__ void templatePointAttentionK(const float* q, const float* k, const float* v, float* out, size_t pairs, int T,
                                        int H, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * H) return;
  size_t p = t / H; int h = (int)(t % H), W = H * D;
  if (D == 16 && ((p * W + h * 16) & 3) == 0) {
    size_t base = p * W + h * 16;
    const float4* q4 = reinterpret_cast<const float4*>(q + base);
    float4 q0 = ptx_ldg_v4_f32(q4 + 0), q1 = ptx_ldg_v4_f32(q4 + 1), q2 = ptx_ldg_v4_f32(q4 + 2), q3 = ptx_ldg_v4_f32(q4 + 3);
    float logits[8]; float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (s < T) {
        const float4* k4 = reinterpret_cast<const float4*>(k + (size_t)s * pairs * W + base);
        float4 k0 = ptx_ldg_v4_f32(k4 + 0), k1 = ptx_ldg_v4_f32(k4 + 1), k2 = ptx_ldg_v4_f32(k4 + 2), k3 = ptx_ldg_v4_f32(k4 + 3);
        float a = 0.f;
        a = ptx_fma_f32(q0.x, k0.x, a); a = ptx_fma_f32(q0.y, k0.y, a); a = ptx_fma_f32(q0.z, k0.z, a); a = ptx_fma_f32(q0.w, k0.w, a);
        a = ptx_fma_f32(q1.x, k1.x, a); a = ptx_fma_f32(q1.y, k1.y, a); a = ptx_fma_f32(q1.z, k1.z, a); a = ptx_fma_f32(q1.w, k1.w, a);
        a = ptx_fma_f32(q2.x, k2.x, a); a = ptx_fma_f32(q2.y, k2.y, a); a = ptx_fma_f32(q2.z, k2.z, a); a = ptx_fma_f32(q2.w, k2.w, a);
        a = ptx_fma_f32(q3.x, k3.x, a); a = ptx_fma_f32(q3.y, k3.y, a); a = ptx_fma_f32(q3.z, k3.z, a); a = ptx_fma_f32(q3.w, k3.w, a);
        logits[s] = a; mx = ptx_max_f32(mx, a);
      } else {
        logits[s] = 0.f;
      }
    }
    float sum = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (s < T) { logits[s] = ptx_exp_f32(ptx_sub_f32(logits[s], mx)); sum = ptx_add_f32(sum, logits[s]); }
    }
    float4 o0 = {0, 0, 0, 0}, o1 = {0, 0, 0, 0}, o2 = {0, 0, 0, 0}, o3 = {0, 0, 0, 0};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (s < T) {
        float w = logits[s];
        const float4* v4 = reinterpret_cast<const float4*>(v + (size_t)s * pairs * W + base);
        float4 v0 = ptx_ldg_v4_f32(v4 + 0), v1 = ptx_ldg_v4_f32(v4 + 1), v2 = ptx_ldg_v4_f32(v4 + 2), v3 = ptx_ldg_v4_f32(v4 + 3);
        o0.x = ptx_fma_f32(w, v0.x, o0.x); o0.y = ptx_fma_f32(w, v0.y, o0.y); o0.z = ptx_fma_f32(w, v0.z, o0.z); o0.w = ptx_fma_f32(w, v0.w, o0.w);
        o1.x = ptx_fma_f32(w, v1.x, o1.x); o1.y = ptx_fma_f32(w, v1.y, o1.y); o1.z = ptx_fma_f32(w, v1.z, o1.z); o1.w = ptx_fma_f32(w, v1.w, o1.w);
        o2.x = ptx_fma_f32(w, v2.x, o2.x); o2.y = ptx_fma_f32(w, v2.y, o2.y); o2.z = ptx_fma_f32(w, v2.z, o2.z); o2.w = ptx_fma_f32(w, v2.w, o2.w);
        o3.x = ptx_fma_f32(w, v3.x, o3.x); o3.y = ptx_fma_f32(w, v3.y, o3.y); o3.z = ptx_fma_f32(w, v3.z, o3.z); o3.w = ptx_fma_f32(w, v3.w, o3.w);
      }
    }
    float invSum = ptx_rcp_f32(sum);
    float4* out4 = reinterpret_cast<float4*>(out + base);
    ptx_st_global_v4_f32(out4 + 0, make_float4(ptx_mul_f32(o0.x, invSum), ptx_mul_f32(o0.y, invSum), ptx_mul_f32(o0.z, invSum), ptx_mul_f32(o0.w, invSum)));
    ptx_st_global_v4_f32(out4 + 1, make_float4(ptx_mul_f32(o1.x, invSum), ptx_mul_f32(o1.y, invSum), ptx_mul_f32(o1.z, invSum), ptx_mul_f32(o1.w, invSum)));
    ptx_st_global_v4_f32(out4 + 2, make_float4(ptx_mul_f32(o2.x, invSum), ptx_mul_f32(o2.y, invSum), ptx_mul_f32(o2.z, invSum), ptx_mul_f32(o2.w, invSum)));
    ptx_st_global_v4_f32(out4 + 3, make_float4(ptx_mul_f32(o3.x, invSum), ptx_mul_f32(o3.y, invSum), ptx_mul_f32(o3.z, invSum), ptx_mul_f32(o3.w, invSum)));
    return;
  }
  float logits[8]; float mx = -INFINITY;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (s < T) {
      float a = 0.f;
      for (int d = 0; d < D; ++d) a = ptx_fma_f32(ptx_ldg_f32(q + p * W + h * D + d), ptx_ldg_f32(k + ((size_t)s * pairs + p) * W + h * D + d), a);
      logits[s] = a; mx = ptx_max_f32(mx, a);
    } else {
      logits[s] = 0.f;
    }
  }
  float sum = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) if (s < T) { logits[s] = ptx_exp_f32(ptx_sub_f32(logits[s], mx)); sum = ptx_add_f32(sum, logits[s]); }
  for (int d = 0; d < D; ++d) {
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) if (s < T) a = ptx_fma_f32(logits[s], ptx_ldg_f32(v + ((size_t)s * pairs + p) * W + h * D + d), a);
    ptx_st_global_f32(out + p * W + h * D + d, ptx_div_f32(a, sum));
  }
}
inline void templateEmbeddingMonomer(float* pair, const float* pairMask, int L, int T, bool pairOnes = false) {
  const std::string TE = "evoformer/template_embedding/", S = TE + "single_template_embedding/";
  const std::string PS = S + "template_pair_stack/__layer_stack_no_state/";
  const int C = 64, Cz = 128;
  size_t pairs = (size_t)L * L;
  if (T > 8) { fprintf(stderr, "at most 8 templates\n"); exit(1); }
  float* reps = scratch<float>("mtmpl.reps", (size_t)T * pairs * C);
  int nb = (int)dimW(PS + "pair_transition/transition1/weights", 0);
  for (int k = 0; k < T; ++k) {
    float* act = reps + (size_t)k * pairs * C;
    templatePairInputMonomerK<<<(unsigned)((pairs + 7) / 8), 256, 0, STREAM>>>(Idev("t/aatype") + (size_t)k * L,
      W("t/positions") + (size_t)k * L * 37 * 3, W("t/mask") + (size_t)k * L * 37,
      P(S + "embedding2d/weights"), P(S + "embedding2d/bias"), act, L, C);
    for (int b = 0; b < nb; ++b) {        // the MONOMER's order: both attentions, then both multiplications
      triangleAttention(act, pairMask, L, C, PS, b, true, pairOnes);
      triangleAttention(act, pairMask, L, C, PS, b, false, pairOnes);
      triangleMultiplication(act, pairMask, L, C, PS, b, true, pairOnes);
      triangleMultiplication(act, pairMask, L, C, PS, b, false, pairOnes);
      transition(act, pairs, C, PS + "pair_transition", b);
    }
    if (FAST && T == 1) {
      const std::string A = TE + "attention/";
      int Hh = (int)dimW(A + "query_w", 1), D = (int)dimW(A + "query_w", 2), Wd = Hh * D;
      half* rh = scratch<half>("mtmpl.rh", pairs * C);
      layerNormVK<64, float, half><<<lnVGrid(pairs), 256, 0, STREAM>>>(
        act, rh, pairs, P(S + "output_layer_norm/scale"), P(S + "output_layer_norm/offset"));
      half* vvh = scratch<half>("mtmpl.vh", pairs * Wd);
      ltGemm(rh, PH(A + "value_w"), vvh, true, pairs, C, Wd, nullptr, false, 0.f);
      if (!AF2_P16) {
        ltGemm(vvh, PH(A + "output_w"), pair, false, pairs, Wd, Cz, P(A + "output_b"), false, 1.f);
        return;
      }
      float* out = scratch<float>("mtmpl.out", pairs * Cz);
      ltGemm(vvh, PH(A + "output_w"), out, false, pairs, Wd, Cz, nullptr, false, 0.f);
      addIntoPairK<__nv_bfloat16, float><<<blocks((pairs * Cz + 3) / 4), 256, 0, STREAM>>>(pair, out, pairs * Cz, P(A + "output_b"));
      return;
    }
    layerNorm(act, act, pairs, C, S + "output_layer_norm");
  }
  const std::string A = TE + "attention/";
  int Hh = (int)dimW(A + "query_w", 1), D = (int)dimW(A + "query_w", 2), Wd = Hh * D;
  float* vv = scratch<float>("mtmpl.v", (size_t)T * pairs * Wd);
  gemm(reps, P(A + "value_w"), vv, (size_t)T * pairs, C, Wd);
  float* o = vv;
  if (T > 1) {
    float* q = scratch<float>("mtmpl.q", pairs * Wd);
    float* kk = scratch<float>("mtmpl.k", (size_t)T * pairs * Wd);
    // (the query's 1/sqrt(D) as the GEMM's alpha: a cublasSscal over the plane took its count as an int, which
    // L^2 * Wd passes at 5,793 residues)
    gemm(pairF32(pair, pairs * Cz), P(A + "query_w"), q, pairs, Cz, Wd, 0.f, 1.f / sqrtf((float)D));
    gemm(reps, P(A + "key_w"), kk, (size_t)T * pairs, C, Wd);
    o = scratch<float>("mtmpl.o", pairs * Wd);
    templatePointAttentionK<<<blocks(pairs * Hh), 256, 0, STREAM>>>(q, kk, vv, o, pairs, T, Hh, D);
  }
  float* out = scratch<float>("mtmpl.out", pairs * Cz);
  gemm(o, P(A + "output_w"), out, pairs, Wd, Cz);
  WITH_PT(AF2_P16, addIntoPairK<PT, float><<<blocks((pairs * Cz + 3) / 4), 256, 0, STREAM>>>(pair, out, pairs * Cz, P(A + "output_b")));
}

// ---------------------------------------------------------------- the templates' MSA rows
#include "chi_tables.cuh"
template <class T, size_t N> inline const T* symbolPtr(const T (&sym)[N]) { void* p; CK(cudaGetSymbolAddress(&p, sym)); return (const T*)p; }
__device__ inline void sub3(const float* a, const float* b, float* o) { for (int k = 0; k < 3; ++k) o[k] = ptx_sub_f32(a[k], b[k]); }
__device__ inline float dot3(const float* a, const float* b) { return ptx_fma_f32(a[0], b[0], ptx_fma_f32(a[1], b[1], ptx_mul_f32(a[2], b[2]))); }
__device__ inline void cross3(const float* a, const float* b, float* o) {
  o[0] = ptx_fma_f32(a[1], b[2], -ptx_mul_f32(a[2], b[1]));
  o[1] = ptx_fma_f32(a[2], b[0], -ptx_mul_f32(a[0], b[2]));
  o[2] = ptx_fma_f32(a[0], b[1], -ptx_mul_f32(a[1], b[0]));
}
// monomer: aatype(22) | torsion sin,cos(14) | alt torsion sin,cos(14) | torsion mask(7) = 57, and the row's
// mask (the psi torsion's) - all_atom.atom37_to_torsion_angles, placeholder off (zero_init)
__global__ void templateTorsionFeatK(const int* tAatype, const float* tPos, const float* tMask, const int* chiIdx,
                                     const float* chiMaskT, const float* chiPi, float* feat, float* rowMask, int L) {
  __shared__ float zp[9];
  if (threadIdx.x < 9) ptx_st_shared_f32(zp + threadIdx.x, 0.f);
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  int aaRaw = ptx_ldg_s32(tAatype + i), a = ptx_min_s32(aaRaw, 20);
  const float* p = tPos + (size_t)i * 37 * 3; const float* m = tMask + (size_t)i * 37;
  const float* pp = i > 0 ? tPos + (size_t)(i - 1) * 37 * 3 : zp; const float* pm = i > 0 ? tMask + (size_t)(i - 1) * 37 : zp;
  const float* atoms[7][4]; float tm[7];
  atoms[0][0] = pp + 3; atoms[0][1] = pp + 6; atoms[0][2] = p; atoms[0][3] = p + 3;          // pre-omega
  tm[0] = ptx_mul_f32(ptx_mul_f32(pm[1], pm[2]), ptx_mul_f32(ptx_ldg_f32(m + 0), ptx_ldg_f32(m + 1)));
  atoms[1][0] = pp + 6; atoms[1][1] = p; atoms[1][2] = p + 3; atoms[1][3] = p + 6;          // phi
  tm[1] = ptx_mul_f32(pm[2], ptx_mul_f32(ptx_mul_f32(ptx_ldg_f32(m + 0), ptx_ldg_f32(m + 1)), ptx_ldg_f32(m + 2)));
  atoms[2][0] = p; atoms[2][1] = p + 3; atoms[2][2] = p + 6; atoms[2][3] = p + 12;          // psi
  tm[2] = ptx_mul_f32(ptx_mul_f32(ptx_ldg_f32(m + 0), ptx_ldg_f32(m + 1)), ptx_mul_f32(ptx_ldg_f32(m + 2), ptx_ldg_f32(m + 4)));
  for (int c = 0; c < 4; ++c) {
    float mm = ptx_ldg_f32(chiMaskT + a * 4 + c);
    for (int k = 0; k < 4; ++k) {
      int at = ptx_ldg_s32(chiIdx + (a * 4 + c) * 4 + k);
      atoms[3 + c][k] = p + at * 3;
      mm = ptx_mul_f32(mm, ptx_ldg_f32(m + at));
    }
    tm[3 + c] = mm;
  }
  float* f = feat + (size_t)i * 57;
  int aaClip = ptx_min_s32(ptx_max_s32(aaRaw, 0), 21);
  for (int k = 0; k < 22; ++k) ptx_st_global_f32(f + k, (k == aaClip) ? 1.f : 0.f);
  for (int t = 0; t < 7; ++t) {
    double e0[3], e1[3], e2[3], d[3];
    for (int k = 0; k < 3; ++k) {
      e0[k] = ptx_sub_f64((double)atoms[t][2][k], (double)atoms[t][1][k]);
      e1[k] = ptx_sub_f64((double)atoms[t][0][k], (double)atoms[t][2][k]);
      d[k] = ptx_sub_f64((double)atoms[t][3][k], (double)atoms[t][2][k]);
    }
    double n0 = ptx_sqrt_f64(ptx_fma_f64(e0[0], e0[0], ptx_fma_f64(e0[1], e0[1], ptx_fma_f64(e0[2], e0[2], 1e-8))));
    for (int k = 0; k < 3; ++k) e0[k] = ptx_div_f64(e0[k], n0);
    double c = ptx_fma_f64(e1[0], e0[0], ptx_fma_f64(e1[1], e0[1], ptx_mul_f64(e1[2], e0[2])));
    for (int k = 0; k < 3; ++k) e1[k] = ptx_fma_f64(-c, e0[k], e1[k]);
    double n1 = ptx_sqrt_f64(ptx_fma_f64(e1[0], e1[0], ptx_fma_f64(e1[1], e1[1], ptx_fma_f64(e1[2], e1[2], 1e-8))));
    for (int k = 0; k < 3; ++k) e1[k] = ptx_div_f64(e1[k], n1);
    e2[0] = ptx_fma_f64(e0[1], e1[2], -ptx_mul_f64(e0[2], e1[1]));
    e2[1] = ptx_fma_f64(e0[2], e1[0], -ptx_mul_f64(e0[0], e1[2]));
    e2[2] = ptx_fma_f64(e0[0], e1[1], -ptx_mul_f64(e0[1], e1[0]));
    double y = ptx_fma_f64(e1[0], d[0], ptx_fma_f64(e1[1], d[1], ptx_mul_f64(e1[2], d[2])));
    double z = ptx_fma_f64(e2[0], d[0], ptx_fma_f64(e2[1], d[1], ptx_mul_f64(e2[2], d[2])));
    double nn = ptx_sqrt_f64(ptx_fma_f64(z, z, ptx_fma_f64(y, y, 1e-8)));
    float sn = (float)ptx_div_f64(z, nn), cs = (float)ptx_div_f64(y, nn);
    if (t == 0 && i == 0) sn = cs = 0.f;
    if (t == 2) { sn = -sn; cs = -cs; }
    float alt = t >= 3 ? ptx_fma_f32(-2.f, ptx_ldg_f32(chiPi + a * 4 + t - 3), 1.f) : 1.f;
    ptx_st_global_f32(f + 22 + t * 2, sn); ptx_st_global_f32(f + 22 + t * 2 + 1, cs);
    ptx_st_global_f32(f + 36 + t * 2, ptx_mul_f32(sn, alt)); ptx_st_global_f32(f + 36 + t * 2 + 1, ptx_mul_f32(cs, alt));
    ptx_st_global_f32(f + 50 + t, tm[t]);
  }
  ptx_st_global_f32(rowMask + i, tm[2]);
}
// multimer (template_embedding_1d): aatype(22) | sin(chi) mask(4) | cos(chi) mask(4) | chi mask(4) = 34, the
// row's mask chi 1's
__global__ void templateChiFeatK(const int* tAatype, const float* tPos, const float* tMask, const int* chiIdx,
                                 const float* chiMaskT, float* feat, float* rowMask, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  int aaRaw = ptx_ldg_s32(tAatype + i), a = ptx_min_s32(ptx_max_s32(aaRaw, 0), 20);
  const float* p = tPos + (size_t)i * 37 * 3; const float* m = tMask + (size_t)i * 37;
  float* f = feat + (size_t)i * 34;
  int aaClip = ptx_min_s32(ptx_max_s32(aaRaw, 0), 21);
  for (int k = 0; k < 22; ++k) ptx_st_global_f32(f + k, (k == aaClip) ? 1.f : 0.f);
  for (int c = 0; c < 4; ++c) {
    const float* x[4]; float mm = ptx_ldg_f32(chiMaskT + a * 4 + c);
    for (int k = 0; k < 4; ++k) {
      int at = ptx_ldg_s32(chiIdx + (a * 4 + c) * 4 + k);
      x[k] = p + at * 3;
      mm = ptx_mul_f32(mm, ptx_ldg_f32(m + at));
    }
    float v1[3], v2[3], v3[3], c1[3], c2[3], c3[3];
    sub3(x[0], x[1], v1); sub3(x[1], x[2], v2); sub3(x[3], x[2], v3);
    cross3(v1, v2, c1); cross3(v3, v2, c2); cross3(c2, c1, c3);
    float v2m = ptx_sqrt_f32(ptx_max_f32(dot3(v2, v2), 1e-12f));
    float ang = atan2f(dot3(c3, v2), ptx_mul_f32(v2m, dot3(c1, c2)));
    ptx_st_global_f32(f + 22 + c, ptx_mul_f32(ptx_sin_f32(ang), mm));
    ptx_st_global_f32(f + 26 + c, ptx_mul_f32(ptx_cos_f32(ang), mm));
    ptx_st_global_f32(f + 30 + c, mm);
    if (c == 0) ptx_st_global_f32(rowMask + i, mm);
  }
}
// rows [T, L, 256] of the MSA from the templates' single features; their masks into rowMask [T, L]
inline void templateRows(int L, int T, bool multimer, float* rows, float* rowMask) {
  const std::string E = "evoformer/";
  int F = multimer ? 34 : 57;
  float* feat = scratch<float>("trow.feat", (size_t)T * L * F);
  for (int k = 0; k < T; ++k) {
    const int* aat = Idev("t/aatype") + (size_t)k * L;
    const float* pos = W("t/positions") + (size_t)k * L * 37 * 3; const float* msk = W("t/mask") + (size_t)k * L * 37;
    if (multimer)
      templateChiFeatK<<<blocks(L, 128), 128, 0, STREAM>>>(aat, pos, msk, symbolPtr(CHI_ATOM_INDICES), symbolPtr(CHI_ANGLES_MASK),
                                                           feat + (size_t)k * L * F, rowMask + (size_t)k * L, L);
    else
      templateTorsionFeatK<<<blocks(L, 128), 128, 0, STREAM>>>(aat, pos, msk, symbolPtr(CHI_ATOM_INDICES), symbolPtr(CHI_ANGLES_MASK),
                                                               symbolPtr(CHI_PI_PERIODIC), feat + (size_t)k * L * F, rowMask + (size_t)k * L, L);
  }
  float* hid = scratch<float>("trow.hid", (size_t)T * L * 256);
  linearB(feat, E + "template_single_embedding", -1, hid, (size_t)T * L, F, 256, true);
  linearB(hid, E + "template_projection", -1, rows, (size_t)T * L, 256, 256);
}
