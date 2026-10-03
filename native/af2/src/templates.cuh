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

// ---------------------------------------------------------------- the monomer's template embedder
// (af3-any-model's MonomerTemplateEmbedding / MonomerSingleTemplateEmbedding / TemplatePairStack)
// The 88 template pair inputs - dgram(39) | mask_2d | aatype_j(22) | aatype_i(22) | unit vector(3, zeros:
// use_template_unit_vector is off in every shipped monomer config) | backbone mask - all multiplied by
// the backbone mask, through embedding2d (88 -> 64)
__global__ void templatePairInputMonomerK(const int* tAatype, const float* tPos, const float* tMask, const float* w,
                                          const float* b, float* act, int L, int C) {
  size_t ij = (size_t)blockIdx.x;
  if (ij >= (size_t)L * L) return;
  int i = (int)(ij / L), j = (int)(ij % L);
  __shared__ float feat[40];
  __shared__ int ai, aj; __shared__ float bb;
  if (threadIdx.x == 0) {
    auto pbAtom = [&](int r) { return tAatype[r] == 7 ? 1 : 3; };
    int pi = pbAtom(i), pj = pbAtom(j);
    float m2 = tMask[(size_t)i * 37 + pi] * tMask[(size_t)j * 37 + pj];
    float d2 = 0;
    for (int k = 0; k < 3; ++k) { float d = tPos[((size_t)i * 37 + pi) * 3 + k] - tPos[((size_t)j * 37 + pj) * 3 + k]; d2 += d * d; }
    for (int q = 0; q < 39; ++q) {
      float lo = 3.25f + (50.75f - 3.25f) * q / 38.f, hi = 3.25f + (50.75f - 3.25f) * (q + 1) / 38.f;
      feat[q] = (d2 > lo * lo && d2 < (q + 1 < 39 ? hi * hi : 1e8f)) ? 1.f : 0.f;     // (the monomer's dgram is not masked)
    }
    feat[39] = m2;
    auto bbm = [&](int r) { return tMask[(size_t)r * 37] * tMask[(size_t)r * 37 + 1] * tMask[(size_t)r * 37 + 2]; };
    bb = bbm(i) * bbm(j);
    ai = min(max(tAatype[i], 0), 21); aj = min(max(tAatype[j], 0), 21);
  }
  __syncthreads();
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float a = 0;
    for (int q = 0; q < 40; ++q) a += feat[q] * w[q * C + c];
    a += w[(40 + aj) * C + c] + w[(62 + ai) * C + c];
    a += w[87 * C + c];                       // the backbone mask itself (=1 where bb is, scaled below)
    act[ij * C + c] = a * bb + b[c];          // every input times the backbone mask, then the bias
  }
}
// the pointwise attention from the query pair over the templates: q [pairs, H*D] (scaled), k, v
// [T, pairs, H*D]; out [pairs, H*D] = softmax_t(q . k_t) v_t  (no gating; every template present)
__global__ void templatePointAttentionK(const float* q, const float* k, const float* v, float* out, size_t pairs, int T,
                                        int H, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * H) return;
  size_t p = t / H; int h = (int)(t % H), W = H * D;
  float logits[8]; float mx = -INFINITY;
  for (int s = 0; s < T; ++s) {
    float a = 0;
    for (int d = 0; d < D; ++d) a += q[p * W + h * D + d] * k[((size_t)s * pairs + p) * W + h * D + d];
    logits[s] = a; mx = fmaxf(mx, a);
  }
  float sum = 0;
  for (int s = 0; s < T; ++s) { logits[s] = __expf(logits[s] - mx); sum += logits[s]; }
  for (int d = 0; d < D; ++d) {
    float a = 0;
    for (int s = 0; s < T; ++s) a += logits[s] * v[((size_t)s * pairs + p) * W + h * D + d];
    out[p * W + h * D + d] = a / sum;
  }
}
inline void templateEmbeddingMonomer(float* pair, const float* pairMask, int L, int T) {
  const std::string TE = "evoformer/template_embedding/", S = TE + "single_template_embedding/";
  const std::string PS = S + "template_pair_stack/__layer_stack_no_state/";
  const int C = 64, Cz = 128;
  size_t pairs = (size_t)L * L;
  if (T > 8) { fprintf(stderr, "at most 8 templates\n"); exit(1); }
  float* reps = scratch<float>("mtmpl.reps", (size_t)T * pairs * C);
  int nb = (int)dimW(PS + "pair_transition/transition1/weights", 0);
  for (int k = 0; k < T; ++k) {
    float* act = reps + (size_t)k * pairs * C;
    templatePairInputMonomerK<<<(unsigned)pairs, 64, 0, STREAM>>>(Idev("t/aatype") + (size_t)k * L,
      W("t/positions") + (size_t)k * L * 37 * 3, W("t/mask") + (size_t)k * L * 37,
      P(S + "embedding2d/weights"), P(S + "embedding2d/bias"), act, L, C);
    for (int b = 0; b < nb; ++b) {        // the MONOMER's order: both attentions, then both multiplications
      triangleAttention(act, pairMask, L, C, PS, b, true);
      triangleAttention(act, pairMask, L, C, PS, b, false);
      triangleMultiplication(act, pairMask, L, C, PS, b, true);
      triangleMultiplication(act, pairMask, L, C, PS, b, false);
      transition(act, pairs, C, PS + "pair_transition", b);
    }
    float* tmp = scratch<float>("mtmpl.ln", pairs * C);
    layerNorm(act, tmp, pairs, C, S + "output_layer_norm");
    CK(cudaMemcpyAsync(act, tmp, pairs * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  const std::string A = TE + "attention/";
  int Hh = (int)dimW(A + "query_w", 1), D = (int)dimW(A + "query_w", 2), Wd = Hh * D;
  float* q = scratch<float>("mtmpl.q", pairs * Wd);
  float* kk = scratch<float>("mtmpl.k", (size_t)T * pairs * Wd); float* vv = scratch<float>("mtmpl.v", (size_t)T * pairs * Wd);
  gemm(pair, P(A + "query_w"), q, pairs, Cz, Wd);
  float s = 1.f / sqrtf((float)D);
  CB(cublasSscal(H, (int)(pairs * Wd), &s, q, 1));
  gemm(reps, P(A + "key_w"), kk, (size_t)T * pairs, C, Wd);
  gemm(reps, P(A + "value_w"), vv, (size_t)T * pairs, C, Wd);
  float* o = scratch<float>("mtmpl.o", pairs * Wd);
  templatePointAttentionK<<<blocks(pairs * Hh), 256, 0, STREAM>>>(q, kk, vv, o, pairs, T, Hh, D);
  float* out = scratch<float>("mtmpl.out", pairs * Cz);
  gemm(o, P(A + "output_w"), out, pairs, Wd, Cz);
  addBiasK<<<blocks(pairs * Cz), 256, 0, STREAM>>>(out, P(A + "output_b"), pairs, Cz);
  addK2<<<blocks(pairs * Cz), 256, 0, STREAM>>>(pair, out, pairs * Cz);
}

// ---------------------------------------------------------------- the templates' MSA rows
#include "chi_tables.cuh"
template <class T, size_t N> inline const T* symbolPtr(const T (&sym)[N]) { void* p; CK(cudaGetSymbolAddress(&p, sym)); return (const T*)p; }
__device__ inline void sub3(const float* a, const float* b, float* o) { for (int k = 0; k < 3; ++k) o[k] = a[k] - b[k]; }
__device__ inline float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
__device__ inline void cross3(const float* a, const float* b, float* o) {
  o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
}
// monomer: aatype(22) | torsion sin,cos(14) | alt torsion sin,cos(14) | torsion mask(7) = 57, and the row's
// mask (the psi torsion's) - all_atom.atom37_to_torsion_angles, placeholder off (zero_init)
__global__ void templateTorsionFeatK(const int* tAatype, const float* tPos, const float* tMask, const int* chiIdx,
                                     const float* chiMaskT, const float* chiPi, float* feat, float* rowMask, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  int aaRaw = tAatype[i], a = min(aaRaw, 20);
  const float* p = tPos + (size_t)i * 37 * 3; const float* m = tMask + (size_t)i * 37;
  float zp[37 * 3] = {}; float zm[37] = {};
  const float* pp = i > 0 ? tPos + (size_t)(i - 1) * 37 * 3 : zp; const float* pm = i > 0 ? tMask + (size_t)(i - 1) * 37 : zm;
  const float* atoms[7][4]; float tm[7];
  atoms[0][0] = pp + 3; atoms[0][1] = pp + 6; atoms[0][2] = p; atoms[0][3] = p + 3;          // pre-omega
  tm[0] = pm[1] * pm[2] * m[0] * m[1];
  atoms[1][0] = pp + 6; atoms[1][1] = p; atoms[1][2] = p + 3; atoms[1][3] = p + 6;          // phi
  tm[1] = pm[2] * m[0] * m[1] * m[2];
  atoms[2][0] = p; atoms[2][1] = p + 3; atoms[2][2] = p + 6; atoms[2][3] = p + 12;          // psi
  tm[2] = m[0] * m[1] * m[2] * m[4];
  for (int c = 0; c < 4; ++c) {
    float mm = chiMaskT[a * 4 + c];
    for (int k = 0; k < 4; ++k) { int at = chiIdx[(a * 4 + c) * 4 + k]; atoms[3 + c][k] = p + at * 3; mm *= m[at]; }
    tm[3 + c] = mm;
  }
  float* f = feat + (size_t)i * 57;
  for (int k = 0; k < 22; ++k) f[k] = (k == min(max(aaRaw, 0), 21)) ? 1.f : 0.f;
  for (int t = 0; t < 7; ++t) {
    // frame: e0 = origin - neg_x, e1 = xy - origin (Gram-Schmidt), origin = atom 2; the fourth atom in it.
    // In double: at residue 0 the previous atoms are zero padding and Gram-Schmidt cancels to a vector
    // ~1e-14 of the backbone's, which the 1e-8 regulariser should leave near zero - in float the
    // cancellation's rounding was amplified into an O(1) pre-omega (the reference's is exactly 0)
    double e0[3], e1[3], e2[3], d[3];
    for (int k = 0; k < 3; ++k) { e0[k] = (double)atoms[t][2][k] - atoms[t][1][k]; e1[k] = (double)atoms[t][0][k] - atoms[t][2][k];
                                  d[k] = (double)atoms[t][3][k] - atoms[t][2][k]; }
    double n0 = sqrt(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2] + 1e-8); for (int k = 0; k < 3; ++k) e0[k] /= n0;
    double c = e1[0] * e0[0] + e1[1] * e0[1] + e1[2] * e0[2]; for (int k = 0; k < 3; ++k) e1[k] -= c * e0[k];
    double n1 = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2] + 1e-8); for (int k = 0; k < 3; ++k) e1[k] /= n1;
    e2[0] = e0[1] * e1[2] - e0[2] * e1[1]; e2[1] = e0[2] * e1[0] - e0[0] * e1[2]; e2[2] = e0[0] * e1[1] - e0[1] * e1[0];
    double y = e1[0] * d[0] + e1[1] * d[1] + e1[2] * d[2], z = e2[0] * d[0] + e2[1] * d[1] + e2[2] * d[2];
    double nn = sqrt(z * z + y * y + 1e-8);
    float sn = (float)(z / nn), cs = (float)(y / nn);
    // ...and the first residue's pre-omega is DEFINED as zero: its "previous" atoms are padding, the frame
    // degenerate, and its exact value the regulariser's ~0 - the reference's float arithmetic happens to
    // cancel to exactly 0 there, where any other rounding gives noise of either sign
    if (t == 0 && i == 0) sn = cs = 0.f;
    if (t == 2) { sn = -sn; cs = -cs; }
    float alt = t >= 3 ? 1.f - 2.f * chiPi[a * 4 + t - 3] : 1.f;
    f[22 + t * 2] = sn; f[22 + t * 2 + 1] = cs;
    f[36 + t * 2] = sn * alt; f[36 + t * 2 + 1] = cs * alt;
    f[50 + t] = tm[t];
  }
  rowMask[i] = tm[2];
}
// multimer (template_embedding_1d): aatype(22) | sin(chi) mask(4) | cos(chi) mask(4) | chi mask(4) = 34, the
// row's mask chi 1's
__global__ void templateChiFeatK(const int* tAatype, const float* tPos, const float* tMask, const int* chiIdx,
                                 const float* chiMaskT, float* feat, float* rowMask, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  int aaRaw = tAatype[i], a = min(max(aaRaw, 0), 20);
  const float* p = tPos + (size_t)i * 37 * 3; const float* m = tMask + (size_t)i * 37;
  float* f = feat + (size_t)i * 34;
  for (int k = 0; k < 22; ++k) f[k] = (k == min(max(aaRaw, 0), 21)) ? 1.f : 0.f;
  for (int c = 0; c < 4; ++c) {
    const float* x[4]; float mm = chiMaskT[a * 4 + c];
    for (int k = 0; k < 4; ++k) { int at = chiIdx[(a * 4 + c) * 4 + k]; x[k] = p + at * 3; mm *= m[at]; }
    float v1[3], v2[3], v3[3], c1[3], c2[3], c3[3];
    sub3(x[0], x[1], v1); sub3(x[1], x[2], v2); sub3(x[3], x[2], v3);
    cross3(v1, v2, c1); cross3(v3, v2, c2); cross3(c2, c1, c3);
    float v2m = sqrtf(fmaxf(dot3(v2, v2), 1e-12f));
    float ang = atan2f(dot3(c3, v2), v2m * dot3(c1, c2));
    f[22 + c] = sinf(ang) * mm; f[26 + c] = cosf(ang) * mm; f[30 + c] = mm;
    if (c == 0) rowMask[i] = mm;
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
