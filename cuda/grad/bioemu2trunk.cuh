// BioEmu2's trunk for training (cuda/bioemu2/src/trunk.cuh is the forward it follows: network/bioemu2.py's
// input_embedder, trunk_init, trunk_pass and distogram):
//
//   s_inputs = [mean_atoms(relu(W q)) | one-hot res_type | profile | deletion_mean | one-hot pocket]   (455)
//     q: the per-atom features c through three windowed conditioned layers (the atom encoder, on the tape - tape.cuh
//        and bioemu2den.cuh's ops, its weights under input_embedder/)
//   s_init = W s_inputs + method[m];  z_init = W1 s_i + W2 s_j + method_pair[m_i] + method_pair[m_j] + relpos
//            + bonds W_b + the contact conditioning (a class's own vector, or an encoder of its threshold's Fourier features)
//   a pass: s = s_init + W LN(s_prev);  z = z_init + W LN(z_prev); four MSA layers over m = W msa_feat + W s_inputs, then
//           the PairMixer layers (bioemu2.cuh's)
//   distogram = W (z + z^T) + b
// Saved for the backward: each pass's s and z before its recycle, and each layer's inputs; a layer is replayed before
// its backward. The tape holds only the input embedder.
#pragma once
#include "bioemu2.cuh"
#include "bioemu2den.cuh"

namespace grad {
namespace be2 {

const std::string IE = "input_embedder/atom_attention_encoder/", SE = "sequence_embedder/";

// ---------------------------------------------------------------- the input embedder, on the tape
struct InputBatch {
  const int *start;                    // [T + 1]: each token's (contiguous) real atoms
  const int *resType, *pocket;         // [T]
  const real *profile, *delMean;       // [T, 33], [T]
};
__global__ void meanReluK(const real* h, const int* start, real* out, int T, int C, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int i = t / C, c = t % C;
  real s = 0; int a0 = start[i], a1 = start[i + 1];
  for (int a = a0; a < a1; ++a) s += fmax(h[(size_t)a * C + c], (real)0);
  out[(size_t)i * ld + c] = s / ((real)(a1 - a0) + (real)1e-6);
}
__global__ void meanReluBK(const real* d, const real* h, const int* start, real* dh, int T, int C, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int i = t / C, c = t % C;
  int a0 = start[i], a1 = start[i + 1];
  real g = d[(size_t)i * ld + c] / ((real)(a1 - a0) + (real)1e-6);
  for (int a = a0; a < a1; ++a) if (h[(size_t)a * C + c] > 0) dh[(size_t)a * C + c] += g;
}
__global__ void sInputsTailK(real* s, const int* resType, const real* profile, const real* delMean, const int* pocket, int T, int C0, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int W = 33 + 33 + 1 + 4;
  if (t >= (size_t)T * W) return;
  int i = t / W, k = t % W;
  real v;
  if (k < 33) v = resType[i] == k ? 1 : 0;
  else if (k < 66) v = profile[(size_t)i * 33 + (k - 33)];
  else if (k == 66) v = delMean[i];
  else v = pocket[i] == k - 67 ? 1 : 0;
  s[(size_t)i * ld + C0 + k] = v;
}
// s_inputs [T, Si] (a node), from the batch's atoms
inline Node* inputEmbedder(Graph& g, const InputBatch& ib, int Si) {
  Tape& t = g.t; Weights& W = g.W; const Batch& b = g.b;
  const int T = b.T, N = b.N;
  const std::string root = g.root;
  g.root = "";
  const int Ca = W.dim(IE + "proj_atom_continuous/weight", 0);
  Param &wC = W.get(IE + "proj_atom_continuous/weight"), &wE = W.get(IE + "embed_element/weight");
  Param* wn[4];
  for (int q = 0; q < 4; ++q) wn[q] = &W.get(IE + "embed_atom_name_chars_" + std::to_string(q) + "/weight");
  Node* c = t.custom(N, Ca, [&](Node* y) {
    atomFeatK<<<blocks((size_t)N * Ca), 256, 0, STREAM>>>(b.ref, b.charge, b.element, b.names, b.atomPad, wC.w, wE.w, wn[0]->w, wn[1]->w,
                                                          wn[2]->w, wn[3]->w, y->v, N, Ca);
  }, [=, wn0 = wn[0], wn1 = wn[1], wn2 = wn[2], wn3 = wn[3], pC = &wC, pE = &wE](Node* y) {
    atomFeatBK<<<blocks((size_t)N * Ca), 256, 0, STREAM>>>(y->g, b.ref, b.charge, b.element, b.names, b.atomPad, pC->g, pE->g, wn0->g,
                                                           wn1->g, wn2->g, wn3->g, N, Ca);
  });
  const int CP = W.dim(IE + "embed_atompair_ref_pos/weight", 0);
  Param &wP = W.get(IE + "embed_atompair_ref_pos/weight"), &wD = W.get(IE + "embed_atompair_ref_dist/weight"),
        &wM = W.get(IE + "embed_atompair_mask/weight");
  Node* geo = t.custom((size_t)N * WK, CP, [&](Node* y) {
    pairGeomK<<<blocks((size_t)N * WK), 256, 0, STREAM>>>(b.ref, b.uid, b.atomPad, wP.w, wD.w, wM.w, y->v, N, CP);
  }, [=, pP = &wP, pD = &wD, pM = &wM](Node* y) {
    pairGeomBK<<<blocks(CP, 32), 32, 0, STREAM>>>(y->g, b.ref, b.uid, b.atomPad, pP->g, pD->g, pM->g, N, CP);
  });
  Node* rc = t.relu(c);
  Node* cq = t.linear(rc, W.get(IE + "c_to_p_trans_q_1/weight"), nullptr, CP);
  Node* ck = t.linear(rc, W.get(IE + "c_to_p_trans_k_1/weight"), nullptr, CP);
  Node* p2 = t.custom((size_t)N * WK, CP, [=](Node* y) {
    pairQKAddK<<<blocks((size_t)N * WK * CP), 256, 0, STREAM>>>(geo->v, cq->v, ck->v, y->v, N, CP);
  }, [=](Node* y) {
    tp::accK<<<blocks(y->n()), 256, 0, STREAM>>>(geo->g, y->g, y->n());
    pairQKAddBK<<<blocks((size_t)N * WK * CP), 256, 0, STREAM>>>(y->g, cq->g, ck->g, N, CP);
  });
  Node* h = p2;
  for (int k : {1, 3, 5}) h = t.linear(t.relu(h), W.get(IE + "p_mlp_" + std::to_string(k) + "/weight"), nullptr, CP);
  Node* p = t.add(p2, h);
  const std::string a = "attention/attention_layer/pair_bias_attn/";
  const int Ha = W.dim(IE + "atom_encoder_layers_0/transformer_layer/" + a + "proj_z/weight", 0);
  Node* q = c;
  for (int i = 0; i < 3; ++i) {
    const std::string lp = IE + "atom_encoder_layers_" + std::to_string(i) + "/transformer_layer/";
    Node* pb = t.linear(t.layerNorm(p, &W.get(lp + a + "norm_z/weight"), nullptr), W.get(lp + a + "proj_z/weight"), nullptr, Ha);
    Node* bias = t.custom((size_t)N * WK * Ha, 1, [=](Node* y) { winPermK<<<blocks((size_t)N * WK * Ha), 256, 0, STREAM>>>(pb->v, y->v, N, Ha, false); },
                          [=](Node* y) { winPermK<<<blocks((size_t)N * WK * Ha), 256, 0, STREAM>>>(y->g, pb->g, N, Ha, true); });
    Node* cond = layerCondition(g, c, lp, -1, 6 * Ca);
    q = conditioned(g, q, cond, lp, a, -1, bias, AttnShape{true, N, Ha, Ca / Ha, WK, b.atomPad});
  }
  const int A = W.dim(IE + "atom_to_token_trans_0/weight", 0);
  Node* hA = t.linear(q, W.get(IE + "atom_to_token_trans_0/weight"), nullptr, A);
  Node* sIn = t.custom(T, Si, [=](Node* y) {
    meanReluK<<<blocks((size_t)T * A), 256, 0, STREAM>>>(hA->v, ib.start, y->v, T, A, Si);
    sInputsTailK<<<blocks((size_t)T * 71), 256, 0, STREAM>>>(y->v, ib.resType, ib.profile, ib.delMean, ib.pocket, T, A, Si);
  }, [=](Node* y) {
    meanReluBK<<<blocks((size_t)T * A), 256, 0, STREAM>>>(y->g, hA->v, ib.start, hA->g, T, A, Si);
  });
  g.root = root;
  return sIn;
}

// ---------------------------------------------------------------- s_init, z_init, the passes, the distogram
struct TrunkBatch {
  const int *meth, *asym, *ent, *res, *sym, *cls;   // [T], [T]..., contact_class [T, T]
  const real *bonds, *thr;                          // [T, T]
  const int *msa, *hasDel, *paired;                 // [S, T]
  const real* delVal;                               // [S, T]
  int T, S;
  real validRows;
};
struct Trunk {
  Linear sInit, z1, z2, sRec, zRec, sProj, msaProj, dist;
  Param methEmb, methPair, bondW, ccEnc, ccEncB, unspec, unsel, fw, fb, rP, rT, rE, rC;
  LayerNorm sNorm, zNorm;
  std::vector<bioemu2::MsaLayer> msa;
  std::vector<bioemu2::PairHalf> mixers;
  int Si = 455, Cs = 384, C = 256, Cm = 128, B = 64;
};
// every parameter (the vendor's (out, in) names; a layer's slice "<tensor>@<l>")
inline void forEachParam(Trunk& m, int msaLayers, int mixers, int H, int D, int Co, int Im, int Iz, int Izm, real validRows,
                         int methods, const std::function<int(const std::string&, int)>& dim,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  auto lin = [&](const std::string& n, Linear& x, int K, int N, const std::string& bias = "") {
    x.K = K; x.N = N; x.trans = true; x.hasBias = !bias.empty(); f(n, x.w, (size_t)K * N); if (x.hasBias) f(bias, x.b, N);
  };
  const int Si = m.Si, Cs = m.Cs, C = m.C, Cm = m.Cm;
  lin(SE + "s_init/weight", m.sInit, Si, Cs);
  lin(SE + "z_init_1/weight", m.z1, Si, C); lin(SE + "z_init_2/weight", m.z2, Si, C);
  f(SE + "method_conditioning/embedding/weight", m.methEmb, (size_t)methods * Cs);
  f(SE + "method_conditioning/pair_embedding/weight", m.methPair, (size_t)methods * C);
  f(SE + "token_bonds/weight", m.bondW, C);
  const std::string cc = SE + "contact_conditioning/";
  f(cc + "encoder/weight", m.ccEnc, (size_t)C * (4 + C)); f(cc + "encoder/bias", m.ccEncB, C);
  f(cc + "encoding_unspecified", m.unspec, C); f(cc + "encoding_unselected", m.unsel, C);
  f(cc + "fourier_embedding/proj/weight", m.fw, C); f(cc + "fourier_embedding/proj/bias", m.fb, C);
  const std::string rp = SE + "rel_pos/";
  f(rp + "embed_rel_pos/weight", m.rP, (size_t)dim(rp + "embed_rel_pos/weight", 0) * C);
  f(rp + "embed_rel_token/weight", m.rT, (size_t)dim(rp + "embed_rel_token/weight", 0) * C);
  f(rp + "embed_same_entity/weight", m.rE, C);
  f(rp + "embed_rel_chain/weight", m.rC, (size_t)dim(rp + "embed_rel_chain/weight", 0) * C);
  m.sNorm.C = Cs; f(SE + "s_norm/weight", m.sNorm.g, Cs); f(SE + "s_norm/bias", m.sNorm.b, Cs);
  m.zNorm.C = C; f(SE + "z_norm/weight", m.zNorm.g, C); f(SE + "z_norm/bias", m.zNorm.b, C);
  lin(SE + "s_recycle/weight", m.sRec, Cs, Cs); lin(SE + "z_recycle/weight", m.zRec, C, C);
  lin(SE + "msa_module/s_proj/weight", m.sProj, Si, Cm);
  lin(SE + "msa_module/msa_proj/weight", m.msaProj, 36, Cm);
  const std::string ml = SE + "msa_module/layers/", pm = SE + "pairmixer_module/layers/";
  m.msa.resize(msaLayers);
  for (int l = 0; l < msaLayers; ++l) bioemu2::msaLayerParams(m.msa[l], ml, l, Cm, C, H, D, Co, Im, Iz, validRows, f);
  m.mixers.resize(mixers);
  for (int l = 0; l < mixers; ++l) bioemu2::mixerParams(m.mixers[l], pm, l, C, Izm, f);
  lin("distogram_module/distogram/weight", m.dist, C, m.B, "distogram_module/distogram/bias");
}
// z_init's per-pair terms (cuda/bioemu2's zInitTermsK), added to z: the method pair, the bonds, the contact conditioning
__global__ void zInitTermsK(real* z, const int* meth, const real* mp, const real* bonds, const real* wb, const int* cls, const real* thr,
                            const real* enc, const real* encB, const real* unspec, const real* unsel, const real* fw, const real* fb, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / T, j = ij % T;
  real v = mp[(size_t)meth[i] * C + c] + mp[(size_t)meth[j] * C + c] + bonds[ij] * wb[c];
  int k = cls[ij];
  if (k == 0) v += unspec[c];
  else if (k == 1) v += unsel[c];
  else {
    const int W = 4 + C;
    const real th = (thr[ij] - 4) / 16;
    const real* row = enc + (size_t)c * W;
    real e = encB[c] + row[k - 2] + row[3] * th;
    for (int q = 0; q < C; ++q) e += row[4 + q] * cos(2 * (real)M_PI * (th * fw[q] + fb[q]));
    v += e;
  }
  z[t] += v;
}
// their gradients: a thread a channel over every pair (the method rows, the bond, the two class vectors, the encoder's
// row and bias); the Fourier weights a thread a feature over every pair and channel
__global__ void zInitTermsBK(const real* d, const int* meth, const real* bonds, const int* cls, const real* thr, const real* fw, const real* fb,
                             real* gmp, real* gwb, real* genc, real* gencB, real* gun, real* gus, int T, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int W = 4 + C;
  real sb = 0, su = 0, ss = 0, se = 0, s3 = 0, sk[3] = {0, 0, 0};
  for (size_t ij = 0; ij < (size_t)T * T; ++ij) {
    real g = d[ij * C + c];
    int i = ij / T, j = ij % T;
    gmp[(size_t)meth[i] * C + c] += g; gmp[(size_t)meth[j] * C + c] += g;
    sb += bonds[ij] * g;
    int k = cls[ij];
    if (k == 0) su += g;
    else if (k == 1) ss += g;
    else {
      const real th = (thr[ij] - 4) / 16;
      se += g; sk[k - 2] += g; s3 += g * th;
      for (int q = 0; q < C; ++q) genc[(size_t)c * W + 4 + q] += g * cos(2 * (real)M_PI * (th * fw[q] + fb[q]));
    }
  }
  gwb[c] += sb; gun[c] += su; gus[c] += ss; gencB[c] += se; genc[(size_t)c * W + 3] += s3;
  for (int k = 0; k < 3; ++k) genc[(size_t)c * W + k] += sk[k];
}
__global__ void zInitFourierBK(const real* d, const int* cls, const real* thr, const real* enc, const real* fw, const real* fb,
                               real* gfw, real* gfb, int T, int C) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= C) return;
  const int W = 4 + C;
  real a = 0, b = 0;
  for (size_t ij = 0; ij < (size_t)T * T; ++ij) {
    if (cls[ij] < 2) continue;
    const real th = (thr[ij] - 4) / 16;
    real e = 0;
    for (int c = 0; c < C; ++c) e += d[ij * C + c] * enc[(size_t)c * W + 4 + q];
    real s = -e * sin(2 * (real)M_PI * (th * fw[q] + fb[q])) * 2 * (real)M_PI;
    a += s * th; b += s;
  }
  gfw[q] += a; gfb[q] += b;
}
// z[i, j] = rows[i] + cols[j]; and back: drows[i] = sum_j d[i, j], dcols[j] = sum_i d[i, j]
__global__ void outerK(const real* rows, const real* cols, real* z, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = t % C; size_t ij = t / C;
  z[t] = rows[(ij / T) * C + c] + cols[(ij % T) * C + c];
}
__global__ void outerBK(const real* d, real* dr, real* dc, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int a = t / C, c = t % C;
  real r = 0, k = 0;
  for (int o = 0; o < T; ++o) { r += d[((size_t)a * T + o) * C + c]; k += d[((size_t)o * T + a) * C + c]; }
  dr[t] = r; dc[t] = k;
}
// the method rows of s_init, and back
__global__ void methAddK(real* s, const int* meth, const real* emb, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)T * C) s[t] += emb[(size_t)meth[t / C] * C + t % C];
}
__global__ void methAddBK(const real* d, const int* meth, real* g, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)T * C) atomicAdd(&g[(size_t)meth[t / C] * C + t % C], d[t]);
}
// the MSA's 36 feature columns [S T, 36]: one-hot msa (33) | has_deletion | deletion_value | paired (an input)
__global__ void msaFeatK(const int* msa, const int* hasDel, const real* delVal, const int* paired, real* F, size_t rows) {
  size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  real* f = F + r * 36;
  for (int k = 0; k < 36; ++k) f[k] = 0;
  f[msa[r]] = 1; f[33] = hasDel[r]; f[34] = delVal[r]; f[35] = paired[r];
}
__global__ void addRowsK(real* m, const real* sp, size_t rows, int T, int C) {    // m[s, i] += sp[i]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) m[t] += sp[((t / C) % T) * C + t % C];
}
__global__ void sumOverSK(const real* d, real* out, int S, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  real a = 0; for (int s = 0; s < S; ++s) a += d[(size_t)s * T * C + t];
  out[t] = a;
}
__global__ void symAddK(const real* z, real* out, int T, int C) {    // out = z + z^T
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / T, j = ij % T;
  out[t] = z[t] + z[((size_t)j * T + i) * C + c];
}
struct TrunkRun {
  real *sInit, *zInit, *ones, *F;
  std::vector<real*> sPrev, zPrev;                       // [passes]: each pass's incoming s, z
  int passes = 0;
};
// z_init and s_init from s_inputs
inline void initForward(Trunk& m, const TrunkBatch& b, const real* sIn, real* sInit, real* zInit) {
  const int T = b.T, C = m.C; const size_t P = (size_t)T * T;
  m.sInit.forward(sIn, sInit, T);
  methAddK<<<blocks((size_t)T * m.Cs), 256, 0, STREAM>>>(sInit, b.meth, m.methEmb.w, T, m.Cs);
  Scratch sc;
  real *rows = ARENA.get((size_t)T * C), *cols = ARENA.get((size_t)T * C);
  m.z1.forward(sIn, rows, T); m.z2.forward(sIn, cols, T);
  outerK<<<blocks(P * C), 256, 0, STREAM>>>(rows, cols, zInit, T, C);
  zInitTermsK<<<blocks(P * C), 256, 0, STREAM>>>(zInit, b.meth, m.methPair.w, b.bonds, m.bondW.w, b.cls, b.thr, m.ccEnc.w, m.ccEncB.w,
                                                 m.unspec.w, m.unsel.w, m.fw.w, m.fb.w, T, C);
  real* rel = ARENA.get(P * C);
  relEmbK<<<blocks(P * C), 256, 0, STREAM>>>(b.asym, b.ent, b.res, b.sym, m.rP.w, m.rT.w, m.rE.w, m.rC.w, rel, T, C);
  add(zInit, rel, P * C);
}
inline void initBackward(Trunk& m, const TrunkBatch& b, const real* sIn, const real* dsInit, const real* dzInit, real* dsIn) {
  Scratch sc;
  const int T = b.T, C = m.C; const size_t P = (size_t)T * T;
  methAddBK<<<blocks((size_t)T * m.Cs), 256, 0, STREAM>>>(dsInit, b.meth, m.methEmb.g, T, m.Cs);
  m.sInit.backward(sIn, dsInit, dsIn, T, true);
  relEmbBK<<<blocks(P * C), 256, 0, STREAM>>>(dzInit, b.asym, b.ent, b.res, b.sym, m.rP.g, m.rT.g, m.rE.g, m.rC.g, T, C);
  zInitTermsBK<<<blocks(C, 64), 64, 0, STREAM>>>(dzInit, b.meth, b.bonds, b.cls, b.thr, m.fw.w, m.fb.w, m.methPair.g, m.bondW.g,
                                                 m.ccEnc.g, m.ccEncB.g, m.unspec.g, m.unsel.g, T, C);
  zInitFourierBK<<<blocks(C, 64), 64, 0, STREAM>>>(dzInit, b.cls, b.thr, m.ccEnc.w, m.fw.w, m.fb.w, m.fw.g, m.fb.g, T, C);
  real *dr = ARENA.get((size_t)T * C), *dc = ARENA.get((size_t)T * C);
  outerBK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(dzInit, dr, dc, T, C);
  m.z1.backward(sIn, dr, dsIn, T, true); m.z2.backward(sIn, dc, dsIn, T, true);
}
// one pass, in place on s and z (the layers kept: their inputs in `ms`, `zs` - each layer's m and z)
struct PassSave { real *sn, *zn, *m0; LnSave ls, lz; std::vector<real*> ms, zs; };
inline PassSave passForward(Trunk& m, const TrunkBatch& b, const TrunkRun& r, const real* sIn, real* s, real* z, bool keep) {
  PassSave p{};
  const int T = b.T, S = b.S, C = m.C, Cm = m.Cm; const size_t P = (size_t)T * T, R = (size_t)S * T;
  // (the norms read copies: s and z are overwritten below, and a LayerNorm's save points at its input)
  p.sn = ARENA.get((size_t)T * m.Cs); p.ls = lnForward(m.sNorm, cloneR(s, (size_t)T * m.Cs), p.sn, T);
  GK(cudaMemcpyAsync(s, r.sInit, (size_t)T * m.Cs * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  m.sRec.forward(p.sn, s, T, 1);
  p.zn = ARENA.get(P * C); p.lz = lnForward(m.zNorm, cloneR(z, P * C), p.zn, P);
  GK(cudaMemcpyAsync(z, r.zInit, P * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  m.zRec.forward(p.zn, z, P, 1);
  p.m0 = ARENA.get(R * Cm);
  m.msaProj.forward(r.F, p.m0, R);
  {
    Scratch sc;
    real* sp = ARENA.get((size_t)T * Cm); m.sProj.forward(sIn, sp, T);
    addRowsK<<<blocks(R * Cm), 256, 0, STREAM>>>(p.m0, sp, R, T, Cm);
  }
  real* mm_ = cloneR(p.m0, R * Cm);
  const int L = (int)m.msa.size(), Q = (int)m.mixers.size();
  if (keep) { p.ms.assign(L, nullptr); p.zs.assign(L + Q, nullptr); }
  for (int l = 0; l < L; ++l) {
    if (keep) { p.ms[l] = cloneR(mm_, R * Cm); p.zs[l] = cloneR(z, P * C); }
    Scratch sc;
    real *mo = ARENA.get(R * Cm), *zo = ARENA.get(P * C);
    bioemu2::msaLayerForward(m.msa[l], mm_, z, r.ones, mo, zo, S, T);
    GK(cudaMemcpyAsync(mm_, mo, R * Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(z, zo, P * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  for (int l = 0; l < Q; ++l) {
    if (keep) p.zs[L + l] = cloneR(z, P * C);
    Scratch sc;
    real* zo = ARENA.get(P * C);
    bioemu2::pairHalfForward(m.mixers[l], z, r.ones, zo, T);
    GK(cudaMemcpyAsync(z, zo, P * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  return p;
}
// ds, dz: the pass's outputs' gradients, replaced by its inputs'; dsInit, dzInit, dsIn added
inline void passBackward(Trunk& m, const TrunkBatch& b, const TrunkRun& r, const PassSave& p, const real* sIn, real* ds, real* dz,
                         real* dsInit, real* dzInit, real* dsIn) {
  Scratch scratch;
  const int T = b.T, S = b.S, C = m.C, Cm = m.Cm; const size_t P = (size_t)T * T, R = (size_t)S * T;
  const int L = (int)m.msa.size(), Q = (int)m.mixers.size();
  for (int l = Q; l-- > 0;) {
    Scratch sc;
    real* zo = ARENA.get(P * C);
    auto sv = bioemu2::pairHalfForward(m.mixers[l], p.zs[L + l], r.ones, zo, T);
    real* dzi = ARENA.get(P * C);
    bioemu2::pairHalfBackward(m.mixers[l], sv, r.ones, dz, dzi);
    GK(cudaMemcpyAsync(dz, dzi, P * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  real* dm = ARENA.zeros(R * Cm);
  for (int l = L; l-- > 0;) {
    Scratch sc;
    real *mo = ARENA.get(R * Cm), *zo = ARENA.get(P * C);
    auto sv = bioemu2::msaLayerForward(m.msa[l], p.ms[l], p.zs[l], r.ones, mo, zo, S, T);
    real *dmi = ARENA.get(R * Cm), *dzi = ARENA.get(P * C);
    bioemu2::msaLayerBackward(m.msa[l], sv, r.ones, dm, dz, dmi, dzi, S, T);
    GK(cudaMemcpyAsync(dm, dmi, R * Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(dz, dzi, P * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  // m0 = msaProj(F) + sProj(sIn)[i]
  m.msaProj.backward(r.F, dm, nullptr, R);
  real* dsp = ARENA.get((size_t)T * Cm);
  sumOverSK<<<blocks((size_t)T * Cm), 256, 0, STREAM>>>(dm, dsp, S, T, Cm);
  m.sProj.backward(sIn, dsp, dsIn, T, true);
  // z = zInit + zRec(LN(z_prev)); s = sInit + sRec(LN(s_prev))
  add(dzInit, dz, P * C);
  real* dzn = ARENA.get(P * C); m.zRec.backward(p.zn, dz, dzn, P);
  lnBackward(m.zNorm, p.lz, dzn, dz, P, false);
  add(dsInit, ds, (size_t)T * m.Cs);
  real* dsn = ARENA.get((size_t)T * m.Cs); m.sRec.backward(p.sn, ds, dsn, T);
  lnBackward(m.sNorm, p.ls, dsn, ds, T, false);
}
inline void distogramForward(Trunk& m, const real* z, int T, real* logits) {
  Scratch sc;
  const size_t P = (size_t)T * T;
  real* zs = ARENA.get(P * m.C); symAddK<<<blocks(P * m.C), 256, 0, STREAM>>>(z, zs, T, m.C);
  m.dist.forward(zs, logits, P);
}
inline void distogramBackward(Trunk& m, const real* z, int T, const real* dl, real* dz) {     // dz added
  Scratch sc;
  const size_t P = (size_t)T * T;
  real* zs = ARENA.get(P * m.C); symAddK<<<blocks(P * m.C), 256, 0, STREAM>>>(z, zs, T, m.C);
  real* dzs = ARENA.get(P * m.C); m.dist.backward(zs, dl, dzs, P);
  real* t = ARENA.get(P * m.C); symAddK<<<blocks(P * m.C), 256, 0, STREAM>>>(dzs, t, T, m.C);
  add(dz, t, P * m.C);
}

}  // namespace be2
}  // namespace grad
