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
__global__ void templatePairInputK(const int* tAatype, const float* tPos, const float* tMask, const float* w0,
                                   const float* w1, const float* w2, const float* w3, const float* w4, const float* w5,
                                   const float* w6, const float* w7, const float* bsum, float* act, int L, int C) {
  size_t ij = (size_t)blockIdx.x;
  if (ij >= (size_t)L * L) return;
  int i = (int)(ij / L), j = (int)(ij % L);
  __shared__ float feat[39 + 1 + 4 + 1];
  __shared__ int ai, aj;
  if (threadIdx.x == 0) {
    auto pbAtom = [&](int r) { return tAatype[r] == 7 ? 1 : 3; };
    int pi = pbAtom(i), pj = pbAtom(j);
    float mi = tMask[(size_t)i * 37 + pi], mj = tMask[(size_t)j * 37 + pj];
    float pb2d = mi * mj;
    float d2 = 0;
    for (int k = 0; k < 3; ++k) { float d = tPos[((size_t)i * 37 + pi) * 3 + k] - tPos[((size_t)j * 37 + pj) * 3 + k]; d2 += d * d; }
    for (int b = 0; b < 39; ++b) {
      float lo = 3.25f + (50.75f - 3.25f) * b / 38.f, lower = lo * lo;
      float hi = 3.25f + (50.75f - 3.25f) * (b + 1) / 38.f, upper = b + 1 < 39 ? hi * hi : 1e8f;
      feat[b] = ((d2 > lower && d2 < upper) ? 1.f : 0.f) * pb2d;
    }
    feat[39] = pb2d;
    // backbone frames: rotation from_two_vectors(C - CA, N - CA), translation CA
    auto frame = [&](int r, float* R, float* t, float& m) {
      const float* p = tPos + (size_t)r * 37 * 3;
      m = tMask[(size_t)r * 37] * tMask[(size_t)r * 37 + 1] * tMask[(size_t)r * 37 + 2];
      float e0[3], e1[3];
      for (int k = 0; k < 3; ++k) { e0[k] = p[6 + k] - p[3 + k]; e1[k] = p[k] - p[3 + k]; t[k] = p[3 + k]; }
      float n0 = rsqrtf(fmaxf(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2], 1e-12f));
      for (int k = 0; k < 3; ++k) e0[k] *= n0;
      float c = e1[0] * e0[0] + e1[1] * e0[1] + e1[2] * e0[2];
      for (int k = 0; k < 3; ++k) e1[k] -= c * e0[k];
      float n1 = rsqrtf(fmaxf(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2], 1e-12f));
      for (int k = 0; k < 3; ++k) e1[k] *= n1;
      float e2[3] = {e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0]};
      // columns e0 e1 e2: R = [[e0x e1x e2x], [e0y e1y e2y], [e0z e1z e2z]]
      for (int k = 0; k < 3; ++k) { R[k * 3] = e0[k]; R[k * 3 + 1] = e1[k]; R[k * 3 + 2] = e2[k]; }
    };
    float Ri[9], ti[3], mi_, Rj[9], tj[3], mj_;
    frame(i, Ri, ti, mi_); frame(j, Rj, tj, mj_);
    float d[3] = {tj[0] - ti[0], tj[1] - ti[1], tj[2] - ti[2]}, v[3];
    for (int a = 0; a < 3; ++a) v[a] = Ri[a] * d[0] + Ri[3 + a] * d[1] + Ri[6 + a] * d[2];     // R_i^T (CA_j - CA_i)
    float nv = rsqrtf(fmaxf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2], 1e-12f));
    float bb = sqrtf(mi_ * mj_);
    for (int a = 0; a < 3; ++a) feat[40 + a] = v[a] * nv * bb;
    feat[43] = bb;
    ai = min(max(tAatype[i], 0), 21); aj = min(max(tAatype[j], 0), 21);
  }
  __syncthreads();
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float a = bsum[c];
    for (int b = 0; b < 39; ++b) a += feat[b] * w0[b * C + c];
    a += feat[39] * w1[c];
    a += w2[aj * C + c] + w3[ai * C + c];
    a += feat[40] * w4[c] + feat[41] * w5[c] + feat[42] * w6[c] + feat[43] * w7[c];
    act[ij * C + c] = a;
  }
}
__global__ void sumBiasesK(float* out, const float* const* bs, int n, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s = 0; for (int k = 0; k < n; ++k) s += bs[k][c];
  out[c] = s;
}
__global__ void reluScaleK(float* x, float s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] = fmaxf(x[t] * s, 0.f);
}

// pair += TemplateEmbedding(pair, templates); T templates in t/aatype [T, L], t/positions, t/mask
// (none given: one blank template, as the reference's blank_features)
inline void templateEmbedding(float* pair, const float* pairMask, int L) {
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
  float* qn = scratch<float>("tmpl.qn", pairs * Cz);
  layerNorm(pair, qn, pairs, Cz, S + "query_embedding_norm");
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
  float* sum = scratch<float>("tmpl.sum", pairs * C);
  CK(cudaMemsetAsync(sum, 0, pairs * C * 4, STREAM));
  float* act = scratch<float>("tmpl.act", pairs * C);
  float* qterm = scratch<float>("tmpl.qterm", pairs * C);
  float* actn = scratch<float>("tmpl.actn", pairs * C);
  auto w = [&](int k) { return P(S + "template_pair_embedding_" + std::to_string(k) + "/weights"); };
  for (int k = 0; k < T; ++k) {
    templatePairInputK<<<(unsigned)pairs, 64, 0, STREAM>>>(tAatype + (size_t)k * L, tPos + (size_t)k * L * 37 * 3,
      tMask + (size_t)k * L * 37, w(0), w(1), w(2), w(3), w(4), w(5), w(6), w(7), bsum, act, L, C);
    gemm(qn, w(8), qterm, pairs, Cz, C);
    addK2<<<blocks(pairs * C), 256, 0, STREAM>>>(act, qterm, pairs * C);
    int nb = (int)dimW(IT + "pair_transition/transition1/weights", 0);
    for (int b = 0; b < nb; ++b) {
      triangleMultiplication(act, pairMask, L, C, IT, b, true);
      triangleMultiplication(act, pairMask, L, C, IT, b, false);
      triangleAttention(act, pairMask, L, C, IT, b, true);
      triangleAttention(act, pairMask, L, C, IT, b, false);
      transition(act, pairs, C, IT + "pair_transition", b);
    }
    layerNorm(act, actn, pairs, C, S + "output_layer_norm");
    addK2<<<blocks(pairs * C), 256, 0, STREAM>>>(sum, actn, pairs * C);
  }
  reluScaleK<<<blocks(pairs * C), 256, 0, STREAM>>>(sum, 1.f / T, pairs * C);
  float* out = scratch<float>("tmpl.out", pairs * Cz);
  linearB(sum, TE + "output_linear", -1, out, pairs, C, Cz);
  addK2<<<blocks(pairs * Cz), 256, 0, STREAM>>>(pair, out, pairs * Cz);
}
