// BioEmu2's trunk (network/bioemu2.py's input_embedder, trunk_init, trunk_pass and distogram):
//
//   s_inputs = [mean_atoms(relu(W q)) | one-hot res_type | profile | deletion_mean | one-hot pocket]   (455)
//     q = c through 3 windowed conditioned layers (the atom encoder), c the per-atom features
//   s_init = W s_inputs + method[m];  z_init = W1 s_i + W2 s_j + method_pair[m_i] + method_pair[m_j] + relpos
//            + bonds W_b + contact conditioning
//   a pass: s = s_init + W LN(s_prev);  z = z_init + W LN(z_prev);
//           4 MSA layers (pair-weighted averaging, transition, outer product mean, the two triangle
//           multiplications, transition) over m = W msa_feat + W s_inputs, then 48 PairMixer blocks
//           (triangle out, triangle in, transition). The single is never updated inside a pass.
//   distogram = W (z + z^T) + b, 64 bins
#pragma once
#include "attention.cuh"

namespace be2 {

inline const std::string IE = "input_embedder/atom_attention_encoder/";
inline const std::string SE = "sequence_embedder/";

struct Dims { int T = 0, N = 0, S = 0, nValidRows = 1; };
inline Dims D_;

// each token's atoms: contiguous in BioEmu2's layout - [start[t], start[t + 1])
inline int* TOK_START = nullptr;
inline void buildTokenStarts(int T, int N) {
  const int* tok = M.i("token_idx_at_atom"); const int* pad = M.i("atom_pad_mask");
  std::vector<int> start(T + 1, -1);
  int last = -1;
  for (int a = 0; a < N; ++a) {
    if (!pad[a]) continue;
    if (tok[a] < last) { fprintf(stderr, "BioEmu2: a token's atoms are not contiguous\n"); exit(1); }
    if (tok[a] != last) { for (int t = last + 1; t <= tok[a]; ++t) start[t] = a; last = tok[a]; }
  }
  int real = 0; for (int a = 0; a < N; ++a) real += pad[a] != 0;
  for (int t = last + 1; t <= T; ++t) start[t] = real;
  if (TOK_START) CK(cudaFree(TOK_START));
  TOK_START = upload(start.data(), start.size());
}
// scatter_mean_atoms_to_tokens: per token, the mean over its real atoms (+1e-6 in the count)
__global__ void meanAtomsK(const float* x, const int* start, float* out, int T, int C, int ldo) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int i = (int)(t / C), c = (int)(t % C);
  float s = 0; int n = start[i + 1] - start[i];
  for (int a = start[i]; a < start[i + 1]; ++a) s += x[(size_t)a * C + c];
  out[(size_t)i * ldo + c] = s / ((float)n + 1e-6f);
}
// the rest of s_inputs after the atom half: one-hot res_type (33), profile (33), deletion_mean, one-hot pocket (4)
__global__ void sInputsTailK(float* s, const int* resType, const float* profile, const float* delMean, const int* pocket,
                             int T, int C0, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int W = 33 + 33 + 1 + 4;
  if (t >= (size_t)T * W) return;
  int i = (int)(t / W), k = (int)(t % W);
  float v;
  if (k < 33) v = resType[i] == k ? 1.f : 0.f;
  else if (k < 66) v = profile[(size_t)i * 33 + (k - 33)];
  else if (k == 66) v = delMean[i];
  else v = pocket[i] == k - 67 ? 1.f : 0.f;
  s[(size_t)i * ld + C0 + k] = v;
}

// the windowed atom encoder: s_inputs [T, 455]
inline void inputEmbedder(float* sInputs, int Si, const std::function<void(const char*, const float*, size_t)>& tap) {
  const int T = D_.T, N = D_.N, C = dimP(IE + "proj_atom_continuous/weight", 0);
  float* c = scratch<float>("be2.ie.c", (size_t)N * C);
  atomFeatures(IE, c, N);
  float* pr = scratch<float>("be2.ie.p", (size_t)N * WK * 16);
  pairGeometry(IE, pr, N);
  pairTail(IE, pr, c, N);
  float* q = scratch<float>("be2.ie.q", (size_t)N * C);
  CK(cudaMemcpyAsync(q, c, (size_t)N * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  float* bias = scratch<float>("be2.ie.bias", (size_t)N * WK * 4);
  for (int i = 0; i < 3; ++i) {
    std::string p = IE + "atom_encoder_layers_" + std::to_string(i) + "/transformer_layer/";
    std::string a = "attention/attention_layer/pair_bias_attn/";
    windowBias(pr, P(p + a + "norm_z/weight"), P(p + a + "proj_z/weight"), bias, N, dimP(p + a + "proj_z/weight", 0));
    condLayer({p, a, -1, true, dimP(p + a + "proj_z/weight", 0)}, q, N, C, c, C, bias);
  }
  const int A = dimP(IE + "atom_to_token_trans_0/weight", 0);
  float* h = scratch<float>("be2.ie.h", (size_t)N * A);
  linear(q, P(IE + "atom_to_token_trans_0/weight"), h, N, C, A);
  reluK<<<blocks((size_t)N * A), 256, 0, STREAM>>>(h, h, (size_t)N * A);
  meanAtomsK<<<blocks((size_t)T * A), 256, 0, STREAM>>>(h, TOK_START, sInputs, T, A, Si);
  sInputsTailK<<<blocks((size_t)T * 71), 256, 0, STREAM>>>(sInputs, Idev("res_type"), Fdev("profile"),
      Fdev("deletion_mean"), Idev("pocket_feature"), T, A, Si);
  if (tap) tap("s_inputs", sInputs, (size_t)T * Si);
}

// ---------------------------------------------------------------- z_init
// method pair (both axes), token bonds and contact conditioning added to z [P, C] (C = 256)
__global__ void zInitTermsK(float* z, const float* rows, const float* cols, const int* meth, const float* methPair,
                            const float* bonds, const float* wBond, const int* cls, const float* thr, const float* enc,
                            const float* encB, const float* unspec, const float* unsel, const float* fw, const float* fb,
                            int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  float v = rows[(size_t)i * C + c] + cols[(size_t)j * C + c];
  v += methPair[(size_t)meth[i] * C + c] + methPair[(size_t)meth[j] * C + c];
  v += bonds[ij] * wBond[c];
  // the contact conditioning: x = encoder([cc[2:], thr', cos 2 pi (w thr' + b)]) for a class past UNSELECTED, and
  // the class's own vector for UNSPECIFIED (0) and UNSELECTED (1)
  int k = cls[ij];
  if (k == 0) v += unspec[c];
  else if (k == 1) v += unsel[c];
  else {
    const int W = 3 + 1 + C;
    float th = (thr[ij] - 4.f) / 16.f;
    const float* row = enc + (size_t)c * W;
    float e = encB[c] + row[k - 2] + row[3] * th;
    for (int f = 0; f < C; ++f) e += row[4 + f] * cosf(2.f * (float)M_PI * (th * fw[f] + fb[f]));
    v += e;
  }
  z[t] = v;
}
inline void trunkInit(const float* sInputs, int Si, float* sInit, float* zInit,
                      const std::function<void(const char*, const float*, size_t)>& tap) {
  const int T = D_.T, Cs = dimP(SE + "s_init/weight", 0), C = dimP(SE + "z_init_1/weight", 0);
  linear(sInputs, P(SE + "s_init/weight"), sInit, T, Si, Cs);
  // s_init += method_conditioning.embedding[method] (no token padding here)
  {
    std::vector<int> meth(M.i("method_feature"), M.i("method_feature") + T);
    const float* emb = P(SE + "method_conditioning/embedding/weight");
    for (int i = 0; i < T; ++i) {
      const float one = 1.f;
      CB(cublasSaxpy(H, Cs, &one, emb + (size_t)meth[i] * Cs, 1, sInit + (size_t)i * Cs, 1));
    }
  }
  float* rows = scratch<float>("be2.zi.rows", (size_t)T * C); float* cols = scratch<float>("be2.zi.cols", (size_t)T * C);
  linear(sInputs, P(SE + "z_init_1/weight"), rows, T, Si, C);
  linear(sInputs, P(SE + "z_init_2/weight"), cols, T, Si, C);
  const std::string cc = SE + "contact_conditioning/";
  zInitTermsK<<<blocks((size_t)T * T * C), 256, 0, STREAM>>>(zInit, rows, cols, Idev("method_feature"),
      P(SE + "method_conditioning/pair_embedding/weight"), Fdev("token_bonds"), P(SE + "token_bonds/weight"),
      Idev("contact_class"), Fdev("contact_threshold"), P(cc + "encoder/weight"), P(cc + "encoder/bias"),
      P(cc + "encoding_unspecified"), P(cc + "encoding_unselected"), P(cc + "fourier_embedding/proj/weight"),
      P(cc + "fourier_embedding/proj/bias"), T, C);
  relPos(SE + "rel_pos/", zInit, T, C, true);
  if (tap) { tap("s_init", sInit, (size_t)T * Cs); tap("z_init", zInit, (size_t)T * T * C); }
}

// ---------------------------------------------------------------- the triangle multiplication
// TriangleMultiplicationFull (Boltz-2): zn = LN(z); proj = p_in(zn) sigmoid(g_in(zn)) (masked), a | b its halves;
// outgoing sum_k a[i,k] b[j,k], incoming sum_k a[k,i] b[k,j]; z += p_out(LN(out)) sigmoid(g_out(zn))
__global__ void triSplitK(const float* proj, const float* gate, float* a, float* b, size_t pairs, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * 2 * C) return;
  int c2 = (int)(t % (2 * C)); size_t ij = t / (2 * C);
  float v = proj[t] * sigm(gate[t]);
  if (c2 < C) a[(size_t)c2 * pairs + ij] = v; else b[(size_t)(c2 - C) * pairs + ij] = v;
}
inline void trimul(float* z, const std::string& p, int l, bool outgoing) {
  const int T = D_.T, C = 256; size_t Pn = (size_t)T * T;
  auto w = [&](const std::string& n) { return PL(p + n, l); };
  float* zn = scratch<float>("be2.tri.zn", Pn * C);
  layerNorm(z, zn, Pn, C, w("norm_in/weight"), w("norm_in/bias"));
  float* proj = scratch<float>("be2.tri.proj", Pn * 2 * C); float* gate = scratch<float>("be2.tri.gate", Pn * 2 * C);
  linear(zn, w("p_in/weight"), proj, Pn, C, 2 * C);
  linear(zn, w("g_in/weight"), gate, Pn, C, 2 * C);
  float* a = scratch<float>("be2.tri.a", Pn * C); float* b = scratch<float>("be2.tri.b", Pn * C);
  triSplitK<<<blocks(Pn * 2 * C), 256, 0, STREAM>>>(proj, gate, a, b, Pn, C);
  float* prod = proj;      // (the projection is spent: its first half holds the product)
  const float one = 1.f, zero = 0.f;
  // per channel, row-major [i][k] planes: out(j, i) column-major
  if (outgoing)
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, T, T, T, &one, b, T, Pn, a, T, Pn, &zero, prod, T, Pn, C));
  else
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_T, T, T, T, &one, b, T, Pn, a, T, Pn, &zero, prod, T, Pn, C));
  float* rowsP = gate;
  channelMajorToRowsK<<<blocks(Pn * C), 256, 0, STREAM>>>(prod, rowsP, Pn, C);
  layerNorm(rowsP, rowsP, Pn, C, w("norm_out/weight"), w("norm_out/bias"));
  float* out = a; float* g = b;
  linear(rowsP, w("p_out/weight"), out, Pn, C, C);
  linear(zn, w("g_out/weight"), g, Pn, C, C);
  gateMulAddK<<<blocks(Pn * C), 256, 0, STREAM>>>(z, out, g, Pn * C);
}

// ---------------------------------------------------------------- the MSA module
// m [S, T, 128] = msa_proj([one-hot msa | has_deletion | deletion_value | msa_paired]) + s_proj(s_inputs)[t]
__global__ void msaEmbedK(float* m, const int* msa, const int* hasDel, const float* delVal, const int* paired,
                          const float* w, const float* sp, size_t S, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= S * T * C) return;
  int c = (int)(t % C); size_t st = t / C; int i = (int)(st % T);
  const float* row = w + (size_t)c * 36;
  m[t] = row[msa[st]] + row[33] * (float)hasDel[st] + row[34] * delVal[st] + row[35] * (float)paired[st]
         + sp[(size_t)i * C + c];
}
// the pair-weighted averaging's weights: softmax over j of the [P, H] bias, written heads first [H][T][T]
__global__ void pwaSoftmaxK(const float* b, float* w, int T, int H) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;   // (h, i)
  int lane = threadIdx.x & 31;
  if (row >= (size_t)H * T) return;
  int h = (int)(row / T), i = (int)(row % T);
  float mx = -INFINITY;
  for (int j = lane; j < T; j += 32) mx = fmaxf(mx, b[((size_t)i * T + j) * H + h]);
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  float s = 0;
  for (int j = lane; j < T; j += 32) s += expf(b[((size_t)i * T + j) * H + h] - mx);
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  for (int j = lane; j < T; j += 32) w[row * T + j] = expf(b[((size_t)i * T + j) * H + h] - mx) / s;
}
inline void pairWeightedAveraging(float* m, const float* z, int l) {
  const int T = D_.T, S = D_.S, Cm = 128, C = 256, Hh = 8, Ch = 32;
  const std::string p = SE + "msa_module/layers/pair_weighted_averaging/";
  auto w = [&](const std::string& n) { return PL(p + n, l); };
  size_t R = (size_t)S * T, Pn = (size_t)T * T;
  float* mn = scratch<float>("be2.pwa.mn", R * Cm);
  layerNorm(m, mn, R, Cm, w("norm_m/weight"), w("norm_m/bias"));
  float* v = scratch<float>("be2.pwa.v", R * Hh * Ch); float* g = scratch<float>("be2.pwa.g", R * Hh * Ch);
  linear(mn, w("proj_m/weight"), v, R, Cm, Hh * Ch);
  linear(mn, w("proj_g/weight"), g, R, Cm, Hh * Ch);
  float* zn = scratch<float>("be2.pwa.zn", Pn * C);
  layerNorm(z, zn, Pn, C, w("norm_z/weight"), w("norm_z/bias"));
  float* b = scratch<float>("be2.pwa.b", Pn * Hh);
  linear(zn, w("proj_z/weight"), b, Pn, C, Hh);
  float* wts = scratch<float>("be2.pwa.w", (size_t)Hh * Pn);
  pwaSoftmaxK<<<(unsigned)(((size_t)Hh * T + 7) / 8), 256, 0, STREAM>>>(b, wts, T, Hh);
  float* o = scratch<float>("be2.pwa.o", R * Hh * Ch);
  const float one = 1.f, zero = 0.f;
  // o[s, i, h, :] = sum_j w[h, i, j] v[s, j, h, :], a batch over the rows s for each head
  for (int h = 0; h < Hh; ++h)
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, Ch, T, T, &one, v + h * Ch, Hh * Ch, (long long)T * Hh * Ch,
                                 wts + (size_t)h * Pn, T, 0, &zero, o + h * Ch, Hh * Ch, (long long)T * Hh * Ch, S));
  sigmoidMulK<<<blocks(R * Hh * Ch), 256, 0, STREAM>>>(o, g, o, R * Hh * Ch);
  linear(o, w("proj_o/weight"), m, R, Hh * Ch, Cm, 1.f);
}
// OuterProductMeanVanilla: the mean is over VALID ROWS (rows with any live entry), not per pair
__global__ void opmPackK(const float* proj, float* a, float* b, size_t R, int T, int Co) {   // [R, 2Co] -> a, b [S][T*Co]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * 2 * Co) return;
  int c = (int)(t % (2 * Co)); size_t r = t / (2 * Co);
  (c < Co ? a : b)[r * Co + (c % Co)] = proj[t];
}
// x [(i, c), (j, d)] for a block of i -> y [(i, j), (c, d)] / n
__global__ void opmPermuteRowsK(const float* x, float* y, int bi, int T, int Co, float inv) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)bi * T * Co * Co) return;
  int d = (int)(t % Co); size_t r = t / Co; int c = (int)(r % Co); r /= Co; int j = (int)(r % T); int i = (int)(r / T);
  y[t] = x[((size_t)i * Co + c) * ((size_t)T * Co) + (size_t)j * Co + d] * inv;
}
inline void outerProductMean(const float* m, float* z, int l) {
  const int T = D_.T, S = D_.S, Cm = 128, C = 256;
  const std::string p = SE + "msa_module/layers/outer_product_mean/";
  auto w = [&](const std::string& n) { return PL(p + n, l); };
  const int Co = dimP(p + "proj_ab/weight", 1) / 2;
  size_t R = (size_t)S * T;
  float* mn = scratch<float>("be2.opm.mn", R * Cm);
  layerNorm(m, mn, R, Cm, w("norm/weight"), w("norm/bias"));
  float* proj = scratch<float>("be2.opm.proj", R * 2 * Co);
  linear(mn, w("proj_ab/weight"), proj, R, Cm, 2 * Co);
  float* a = scratch<float>("be2.opm.a", R * Co); float* b = scratch<float>("be2.opm.b", R * Co);
  opmPackK<<<blocks(R * 2 * Co), 256, 0, STREAM>>>(proj, a, b, R, T, Co);
  const size_t TC = (size_t)T * Co;
  int bi = (int)std::max<size_t>(1, std::min<size_t>(T, ((size_t)32 << 20) / (Co * TC)));
  float* x = scratch<float>("be2.opm.x", (size_t)bi * Co * TC);
  float* y = scratch<float>("be2.opm.y", (size_t)bi * T * Co * Co);
  const float one = 1.f, zero = 0.f, inv = 1.f / (float)D_.nValidRows;
  for (int i0 = 0; i0 < T; i0 += bi) {
    int n = std::min(bi, T - i0);
    // x(jd, ic) column-major = sum_s b[s][jd] a[s][ic]
    CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, (int)TC, n * Co, S, &one, b, (int)TC, a + (size_t)i0 * Co, (int)TC, &zero,
                   x, (int)TC));
    opmPermuteRowsK<<<blocks((size_t)n * T * Co * Co), 256, 0, STREAM>>>(x, y, n, T, Co, inv);
    float* zr = z + (size_t)i0 * T * C;
    linear(y, w("proj_o/weight"), zr, (size_t)n * T, Co * Co, C, 1.f);
    addBiasRows(zr, w("proj_o/bias"), (size_t)n * T, C);
  }
}

// the pair-track pieces of a pass: the float32 forms here, cuda/ef2's fused ones under --fast (fasttrunk.cuh)
inline void pairTransitionF32(float* z, const std::string& p, int l) { transition(z, z, (size_t)D_.T * D_.T, 256, p, l); }
inline void (*TRIMUL)(float*, const std::string&, int, bool) = trimul;
inline void (*PAIR_TRANSITION)(float*, const std::string&, int) = pairTransitionF32;
// ---------------------------------------------------------------- a trunk pass
inline void trunkPass(const float* sInputs, int Si, const float* sInit, const float* zInit, float* s, float* z) {
  const int T = D_.T, S = D_.S, Cs = 384, C = 256, Cm = 128;
  size_t Pn = (size_t)T * T;
  // s = s_init + s_recycle(LN(s)); z = z_init + z_recycle(LN(z)), in place
  float* sn = scratch<float>("be2.tp.sn", (size_t)T * Cs);
  layerNorm(s, sn, T, Cs, P(SE + "s_norm/weight"), P(SE + "s_norm/bias"));
  CK(cudaMemcpyAsync(s, sInit, (size_t)T * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  linear(sn, P(SE + "s_recycle/weight"), s, T, Cs, Cs, 1.f);
  float* zn = scratch<float>("be2.tri.zn", Pn * C);       // (shares the triangle's: spent before it runs)
  layerNorm(z, zn, Pn, C, P(SE + "z_norm/weight"), P(SE + "z_norm/bias"));
  CK(cudaMemcpyAsync(z, zInit, Pn * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  linear(zn, P(SE + "z_recycle/weight"), z, Pn, C, C, 1.f);
  // the MSA module
  float* m = scratch<float>("be2.msa.m", (size_t)S * T * Cm);
  float* sp = scratch<float>("be2.msa.sp", (size_t)T * Cm);
  linear(sInputs, P(SE + "msa_module/s_proj/weight"), sp, T, Si, Cm);
  msaEmbedK<<<blocks((size_t)S * T * Cm), 256, 0, STREAM>>>(m, Idev("msa"), Idev("has_deletion"), Fdev("deletion_value"),
      Idev("msa_paired"), P(SE + "msa_module/msa_proj/weight"), sp, S, T, Cm);
  const std::string ml = SE + "msa_module/layers/";
  const int msaLayers = dimP(ml + "z_transition/fc1/weight", 0);
  for (int l = 0; l < msaLayers; ++l) {
    pairWeightedAveraging(m, z, l);
    transition(m, m, (size_t)S * T, Cm, ml + "msa_transition/", l);
    outerProductMean(m, z, l);
    TRIMUL(z, ml + "tri_mul_out/", l, true);
    TRIMUL(z, ml + "tri_mul_in/", l, false);
    PAIR_TRANSITION(z, ml + "z_transition/", l);
  }
  const std::string pm = SE + "pairmixer_module/layers/";
  const int mixers = dimP(pm + "transition_z/fc1/weight", 0);
  for (int l = 0; l < mixers; ++l) {
    TRIMUL(z, pm + "tri_mul_out/", l, true);
    TRIMUL(z, pm + "tri_mul_in/", l, false);
    PAIR_TRANSITION(z, pm + "transition_z/", l);
  }
}

// distogram logits [P, 64] = W (z + z^T) + b
inline void distogram(const float* z, float* logits) {
  const int T = D_.T, C = 256; size_t Pn = (size_t)T * T;
  const int B = dimP("distogram_module/distogram/weight", 0);
  float* zs = scratch<float>("be2.dg.zs", Pn * C);
  symmetriseK<<<blocks(Pn * C), 256, 0, STREAM>>>(z, zs, T, C);
  linear(zs, P("distogram_module/distogram/weight"), logits, Pn, C, B);
  addBiasRows(logits, P("distogram_module/distogram/bias"), Pn, B);
}

}  // namespace be2
