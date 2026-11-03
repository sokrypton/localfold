// BioEmu2's denoiser for training: the static context (the pairwise conditioner's pair, the atom features and the
// windowed atom pair with its tail, every layer's window and token biases, the single conditioner's static half, a's
// trunk half) and one call D(x, sigma) - 16 dual-stream layers and the readout - on the tape (tape.cuh), into every
// weight. cuda/bioemu2/src/denoise.cuh's staticContext() and vjp.cuh's graph are what it follows op for op; the
// forward is held to the port's denoise() (and, through the port's vjp tape, its x gradient to af3-any-model's jax.grad).
//
// Not here: the dynamics-conditioned frame (cond_to_q_trans, the lag's time embedding).
#pragma once
#include <map>
#include "tape.cuh"

namespace grad {
namespace be2 {

using tp::Node; using tp::Tape; using tp::AttnShape;
constexpr int WQ = 32, WK = 128, CTX = (WK - WQ) / 2;

// the weights by name (and layer, for a layer-stacked tensor), each a Param, made from the bundle on first use
struct Weights {
  std::function<bool(const std::string&)> has;
  std::function<int(const std::string&, int)> dim;
  std::function<std::vector<float>(const std::string&, int)> fetch;      // (layer -1: the whole tensor)
  std::map<std::string, Param> p;
  Params all;
  Param& get(const std::string& n, int l = -1) {
    std::string key = l < 0 ? n : n + "@" + std::to_string(l);
    auto it = p.find(key);
    if (it != p.end()) return it->second;
    std::vector<float> h = fetch(n, l);
    Param& q = p[key];
    allocParam(q, key, h.size());
    std::vector<real> r(h.begin(), h.end());
    GK(cudaMemcpy(q.w, r.data(), r.size() * sizeof(real), cudaMemcpyHostToDevice));
    all.add(q);
    return q;
  }
  Param* opt(const std::string& n, int l = -1) { return has(n) ? &get(n, l) : nullptr; }
};
// the batch, on the device
struct Batch {
  int T = 0, N = 0, Si = 0;
  const int *atomPad, *tokAtAtom, *centre, *tokPad, *element, *names, *uid, *asym, *ent, *res, *sym;
  const real *ref, *charge;
  const int* atomRowCond;     // [N]: an atom's token, a padding atom T (the zero row of cond)
  const int* atomTok;         // [N]: an atom's token, a padding atom -1
};

// ---------------------------------------------------------------- the static context's own ops
// c = proj_atom_continuous([ref, charge]) + embed_element[element] + sum embed_atom_name_chars_i[name_i], zero on padding
__global__ void atomFeatK(const real* ref, const real* charge, const int* element, const int* names, const int* pad,
                          const real* wC, const real* wE, const real* n0, const real* n1, const real* n2, const real* n3, real* c,
                          size_t N, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N * C) return;
  size_t a = t / C; int k = t % C;
  if (!pad[a]) { c[t] = 0; return; }
  const real in[4] = {ref[a * 3], ref[a * 3 + 1], ref[a * 3 + 2], charge[a]};
  real v = 0;
  for (int q = 0; q < 4; ++q) v += wC[k * 4 + q] * in[q];
  v += wE[(size_t)element[a] * C + k];
  const real* tb[4] = {n0, n1, n2, n3};
  for (int q = 0; q < 4; ++q) v += tb[q][(size_t)names[a * 4 + q] * C + k];
  c[t] = v;
}
__global__ void atomFeatBK(const real* g, const real* ref, const real* charge, const int* element, const int* names, const int* pad,
                           real* gC, real* gE, real* g0, real* g1, real* g2, real* g3, size_t N, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N * C) return;
  size_t a = t / C; int k = t % C;
  if (!pad[a]) return;
  const real in[4] = {ref[a * 3], ref[a * 3 + 1], ref[a * 3 + 2], charge[a]};
  for (int q = 0; q < 4; ++q) atomicAdd(&gC[k * 4 + q], g[t] * in[q]);
  atomicAdd(&gE[(size_t)element[a] * C + k], g[t]);
  real* tb[4] = {g0, g1, g2, g3};
  for (int q = 0; q < 4; ++q) atomicAdd(&tb[q][(size_t)names[a * 4 + q] * C + k], g[t]);
}
// the windowed pair's geometry: (W_pos disp + W_dist / (1 + |disp|^2) + W_mask same) same
__device__ inline void geomOf(const real* ref, const int* uid, const int* pad, int N, size_t t, real d[3], real& inv, real& same) {
  int j = t % WK; size_t qa = t / WK; int w = qa / WQ;
  int ka = w * WQ - CTX + j;
  bool in = ka >= 0 && ka < N;
  real kx = in ? ref[ka * 3] : 0, ky = in ? ref[ka * 3 + 1] : 0, kz = in ? ref[ka * 3 + 2] : 0;
  d[0] = kx - ref[qa * 3]; d[1] = ky - ref[qa * 3 + 1]; d[2] = kz - ref[qa * 3 + 2];
  inv = 1 / (1 + d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  int ku = in && pad[ka] ? uid[ka] : -1;
  same = pad[qa] && uid[qa] == ku ? (real)1 : 0;
}
__global__ void pairGeomK(const real* ref, const int* uid, const int* pad, const real* wP, const real* wD, const real* wM, real* pr, int N, int CP) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK) return;
  real d[3], inv, same; geomOf(ref, uid, pad, N, t, d, inv, same);
  for (int c = 0; c < CP; ++c)
    pr[t * CP + c] = (wP[c * 3] * d[0] + wP[c * 3 + 1] * d[1] + wP[c * 3 + 2] * d[2] + wD[c] * inv + wM[c] * same) * same;
}
__global__ void pairGeomBK(const real* g, const real* ref, const int* uid, const int* pad, real* gP, real* gD, real* gM, int N, int CP) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;     // by channel
  if (c >= CP) return;
  real sp[3] = {0, 0, 0}, sd = 0, sm = 0;
  for (size_t t = 0; t < (size_t)N * WK; ++t) {
    real d[3], inv, same; geomOf(ref, uid, pad, N, t, d, inv, same);
    if (same == 0) continue;
    real gv = g[t * CP + c];
    for (int k = 0; k < 3; ++k) sp[k] += gv * d[k];
    sd += gv * inv; sm += gv;
  }
  for (int k = 0; k < 3; ++k) gP[c * 3 + k] += sp[k];
  gD[c] += sd; gM[c] += sm;
}
// pr += zp[tok_q, tok_k] where both atoms are real; and pr += cq[q] + ck[k] (ck zero past the edge)
__global__ void pairTokAddK(const real* zp, const int* atomTok, const int* pad, const real* in, real* pr, int N, int T, int CP) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK * CP) return;
  int c = t % CP; size_t r = t / CP; int j = r % WK; size_t qa = r / WK; int w = qa / WQ;
  int ka = w * WQ - CTX + j;
  real v = in[t];
  if (ka >= 0 && ka < N && pad[ka] && pad[qa]) v += zp[((size_t)atomTok[qa] * T + atomTok[ka]) * CP + c];
  pr[t] = v;
}
__global__ void pairTokAddBK(const real* g, const int* atomTok, const int* pad, real* gzp, int N, int T, int CP) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK * CP) return;
  int c = t % CP; size_t r = t / CP; int j = r % WK; size_t qa = r / WK; int w = qa / WQ;
  int ka = w * WQ - CTX + j;
  if (ka >= 0 && ka < N && pad[ka] && pad[qa]) atomicAdd(&gzp[((size_t)atomTok[qa] * T + atomTok[ka]) * CP + c], g[t]);
}
__global__ void pairQKAddK(const real* in, const real* cq, const real* ck, real* pr, int N, int CP) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK * CP) return;
  int c = t % CP; size_t r = t / CP; int j = r % WK; size_t qa = r / WK; int w = qa / WQ;
  int ka = w * WQ - CTX + j;
  pr[t] = in[t] + cq[qa * CP + c] + (ka >= 0 && ka < N ? ck[(size_t)ka * CP + c] : (real)0);
}
__global__ void pairQKAddBK(const real* g, real* gcq, real* gck, int N, int CP) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK * CP) return;
  int c = t % CP; size_t r = t / CP; int j = r % WK; size_t qa = r / WK; int w = qa / WQ;
  int ka = w * WQ - CTX + j;
  atomicAdd(&gcq[qa * CP + c], g[t]);
  if (ka >= 0 && ka < N) atomicAdd(&gck[(size_t)ka * CP + c], g[t]);
}
// [rows (qa WK + j), H] -> [NW, H, WQ, WK]; [P, H] -> [H, P]; and back
__global__ void winPermK(const real* x, real* y, int N, int H, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * WK * H) return;
  int h = t % H; size_t r = t / H; int j = r % WK; size_t qa = r / WK; int w = qa / WQ, i = qa % WQ;
  size_t o = (((size_t)w * H + h) * WQ + i) * WK + j;
  if (back) y[t] += x[o]; else y[o] = x[t];
}
__global__ void tokPermK(const real* x, real* y, size_t P, int H, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= P * H) return;
  size_t p = t / H; int h = t % H;
  if (back) y[t] += x[(size_t)h * P + p]; else y[(size_t)h * P + p] = x[t];
}
// the pairwise conditioner's relative encoding from its four tables
__device__ inline void relRows(const int* asym, const int* ent, const int* res, const int* sym, int i, int j, int& dRes, int& dTok,
                               int& dChain, bool& sameEnt) {
  const int rmax = 32, smax = 2;
  auto clip = [](int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; };
  bool sameChain = asym[i] == asym[j], sameRes = res[i] == res[j];
  sameEnt = ent[i] == ent[j];
  dRes = sameChain ? clip(res[i] - res[j] + rmax, 0, 2 * rmax) : 2 * rmax + 1;
  dTok = sameChain && sameRes ? clip(i - j + rmax, 0, 2 * rmax) : 2 * rmax + 1;
  dChain = sameEnt ? clip(sym[i] - sym[j] + smax, 0, 2 * smax) : 2 * smax + 1;
}
__global__ void relEmbK(const int* asym, const int* ent, const int* res, const int* sym, const real* wP, const real* wT,
                        const real* wE, const real* wC, real* out, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / T, j = ij % T;
  int dR, dT, dC; bool se; relRows(asym, ent, res, sym, i, j, dR, dT, dC, se);
  out[t] = wP[(size_t)dR * C + c] + wT[(size_t)dT * C + c] + (se ? wE[c] : (real)0) + wC[(size_t)dC * C + c];
}
__global__ void relEmbBK(const real* g, const int* asym, const int* ent, const int* res, const int* sym, real* gP, real* gT,
                         real* gE, real* gC, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / T, j = ij % T;
  int dR, dT, dC; bool se; relRows(asym, ent, res, sym, i, j, dR, dT, dC, se);
  atomicAdd(&gP[(size_t)dR * C + c], g[t]); atomicAdd(&gT[(size_t)dT * C + c], g[t]);
  if (se) atomicAdd(&gE[c], g[t]);
  atomicAdd(&gC[(size_t)dC * C + c], g[t]);
}
// cos(2 pi (t w + b)), one row; its parameters' gradient (t a constant here)
__global__ void fourK(const real* w, const real* b, real tt, real* y, int F) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < F) y[k] = cos(2 * (real)M_PI * (tt * w[k] + b[k]));
}
__global__ void fourBK(const real* w, const real* b, real tt, const real* g, real* gw, real* gb, int F) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= F) return;
  real s = -g[k] * sin(2 * (real)M_PI * (tt * w[k] + b[k])) * 2 * (real)M_PI;
  gw[k] += s * tt; gb[k] += s;
}

// ---------------------------------------------------------------- the graph
inline const std::string SM = "structure_module/";
struct Graph { Weights& W; const Batch& b; Tape& t; };
// fc3(silu(fc1 x) * fc2 x)
inline Node* swiglu(Graph& g, Node* x, const std::string& p, int l) {
  Weights& W = g.W;
  const int I = W.dim(SM + p + "fc1/weight", l < 0 ? 0 : 1), Co = W.dim(SM + p + "fc3/weight", l < 0 ? 0 : 1);
  return g.t.linear(g.t.mul(g.t.silu(g.t.linear(x, W.get(SM + p + "fc1/weight", l), nullptr, I)),
                            g.t.linear(x, W.get(SM + p + "fc2/weight", l), nullptr, I)),
                    W.get(SM + p + "fc3/weight", l), nullptr, Co);
}
inline Node* transition(Graph& g, Node* x, const std::string& p, int l) {      // x + swiglu(LN x)
  Weights& W = g.W;
  return g.t.add(x, swiglu(g, g.t.layerNorm(x, &W.get(SM + p + "norm/weight", l), W.opt(SM + p + "norm/bias", l)), p, l));
}
inline Node* layerCondition(Graph& g, Node* c, const std::string& p, int l, int width) {
  Weights& W = g.W;
  return g.t.linear(g.t.silu(g.t.layerNorm(c, &W.get(SM + p + "cond_0/weight", l), W.opt(SM + p + "cond_0/bias", l))),
                    W.get(SM + p + "cond_2/weight", l), W.opt(SM + p + "cond_2/bias", l), width);
}
inline Node* conditioned(Graph& g, Node* x, Node* cond, const std::string& p, const std::string& att, int l, Node* bias, AttnShape a) {
  Tape& t = g.t; Weights& W = g.W;
  const int D = (int)x->cols;
  Node* h = t.adaLn(x, cond, 0, D);
  Node* qkv = t.linear(h, W.get(SM + p + att + "proj_qkv/weight", l), nullptr, 3 * D);
  Node* gt = t.sigmoid(t.linear(h, W.get(SM + p + att + "proj_g/weight", l), nullptr, D));
  Node* o = t.attention(qkv, W.get(SM + p + att + "proj_q_bias", l), bias, a);
  Node* ao = t.linear(t.mul(gt, o), W.get(SM + p + att + "proj_o/weight", l), nullptr, D);
  x = t.gatedAdd(x, ao, cond, 2 * D);
  Node* m = swiglu(g, t.adaLn(x, cond, 3 * D, 4 * D), p + "transition/", l);
  return t.gatedAdd(x, m, cond, 5 * D);
}
// the windowed pair's bias for layer l: LN(p; norm_z, no offset) proj_z -> [NW, H, WQ, WK]
inline Node* windowBias(Graph& g, Node* p, int l, int N, int H) {
  const std::string ab = "layers/atom_block/transformer_layer/attention/attention_layer/pair_bias_attn/";
  Node* pb = g.t.linear(g.t.layerNorm(p, &g.W.get(SM + ab + "norm_z/weight", l), nullptr), g.W.get(SM + ab + "proj_z/weight", l), nullptr, H);
  return g.t.custom((size_t)N * WK * H, 1, [=](Node* y) { winPermK<<<blocks((size_t)N * WK * H), 256, 0, STREAM>>>(pb->v, y->v, N, H, false); },
                    [=](Node* y) { winPermK<<<blocks((size_t)N * WK * H), 256, 0, STREAM>>>(y->g, pb->g, N, H, true); });
}
inline Node* tokenBias(Graph& g, Node* z, int l, int T, int H) {
  const std::string tb = "layers/token_block/attention/pair_bias_attn/";
  Node* pb = g.t.linear(g.t.layerNorm(z, &g.W.get(SM + tb + "norm_z/weight", l), nullptr), g.W.get(SM + tb + "proj_z/weight", l), nullptr, H);
  const size_t P = (size_t)T * T;
  return g.t.custom(P * H, 1, [=](Node* y) { tokPermK<<<blocks(P * H), 256, 0, STREAM>>>(pb->v, y->v, P, H, false); },
                    [=](Node* y) { tokPermK<<<blocks(P * H), 256, 0, STREAM>>>(y->g, pb->g, P, H, true); });
}

struct Ctx { Node *z, *q0, *p, *sc0, *a0; std::vector<Node*> wbias, tbias; };
// the static context, on the tape: sTrunk [T, 384], zTrunk [T*T, 256], sInputs [T, Si] nodes
inline Ctx staticContext(Graph& g, Node* sTrunk, Node* zTrunk, Node* sInputs, int layers) {
  Tape& t = g.t; Weights& W = g.W; const Batch& b = g.b;
  const int T = b.T, N = b.N, C = 256;
  const size_t P = (size_t)T * T;
  const std::string pc = "pairwise_conditioner/", af = "atom_feature_embedder/", sc = "single_conditioner/";
  Ctx x{};
  // the atoms
  const int Ca = W.dim(SM + af + "proj_atom_continuous/weight", 0);
  Param &wC = W.get(SM + af + "proj_atom_continuous/weight"), &wE = W.get(SM + af + "embed_element/weight");
  Param* wn[4];
  for (int q = 0; q < 4; ++q) wn[q] = &W.get(SM + af + "embed_atom_name_chars_" + std::to_string(q) + "/weight");
  Node* c = t.custom(N, Ca, [&](Node* y) {
    atomFeatK<<<blocks((size_t)N * Ca), 256, 0, STREAM>>>(b.ref, b.charge, b.element, b.names, b.atomPad, wC.w, wE.w, wn[0]->w, wn[1]->w,
                                                          wn[2]->w, wn[3]->w, y->v, N, Ca);
  }, [=, wn0 = wn[0], wn1 = wn[1], wn2 = wn[2], wn3 = wn[3], pC = &wC, pE = &wE](Node* y) {
    atomFeatBK<<<blocks((size_t)N * Ca), 256, 0, STREAM>>>(y->g, b.ref, b.charge, b.element, b.names, b.atomPad, pC->g, pE->g, wn0->g,
                                                           wn1->g, wn2->g, wn3->g, N, Ca);
  });
  x.q0 = c;
  // c' = c + s_to_c(LN s_trunk) at each real atom
  Node* s2c = t.linear(t.layerNorm(sTrunk, &W.get(SM + af + "s_to_c_trans_0/weight"), W.opt(SM + af + "s_to_c_trans_0/bias")),
                       W.get(SM + af + "s_to_c_trans_1/weight"), nullptr, Ca);
  Node* cp = t.add(c, t.gather(s2c, b.atomTok, N));
  // the windowed pair: geometry, the token pair's projection, the tail
  const int CP = W.dim(SM + af + "embed_atompair_ref_pos/weight", 0);
  Param &wP = W.get(SM + af + "embed_atompair_ref_pos/weight"), &wD = W.get(SM + af + "embed_atompair_ref_dist/weight"),
        &wM = W.get(SM + af + "embed_atompair_mask/weight");
  Node* geo = t.custom((size_t)N * WK, CP, [&](Node* y) {
    pairGeomK<<<blocks((size_t)N * WK), 256, 0, STREAM>>>(b.ref, b.uid, b.atomPad, wP.w, wD.w, wM.w, y->v, N, CP);
  }, [=, pP = &wP, pD = &wD, pM = &wM](Node* y) {
    pairGeomBK<<<blocks(CP, 32), 32, 0, STREAM>>>(y->g, b.ref, b.uid, b.atomPad, pP->g, pD->g, pM->g, N, CP);
  });
  Node* z2p = t.linear(t.layerNorm(zTrunk, &W.get(SM + af + "z_to_p_trans_0/weight"), W.opt(SM + af + "z_to_p_trans_0/bias")),
                       W.get(SM + af + "z_to_p_trans_1/weight"), nullptr, CP);
  Node* p1 = t.custom((size_t)N * WK, CP, [=](Node* y) {
    pairTokAddK<<<blocks((size_t)N * WK * CP), 256, 0, STREAM>>>(z2p->v, b.tokAtAtom, b.atomPad, geo->v, y->v, N, T, CP);
  }, [=](Node* y) {
    tp::accK<<<blocks(y->n()), 256, 0, STREAM>>>(geo->g, y->g, y->n());
    pairTokAddBK<<<blocks((size_t)N * WK * CP), 256, 0, STREAM>>>(y->g, b.tokAtAtom, b.atomPad, z2p->g, N, T, CP);
  });
  Node* rc = t.relu(cp);
  Node* cq = t.linear(rc, W.get(SM + af + "c_to_p_trans_q_1/weight"), nullptr, CP);
  Node* ck = t.linear(rc, W.get(SM + af + "c_to_p_trans_k_1/weight"), nullptr, CP);
  Node* p2 = t.custom((size_t)N * WK, CP, [=](Node* y) {
    pairQKAddK<<<blocks((size_t)N * WK * CP), 256, 0, STREAM>>>(p1->v, cq->v, ck->v, y->v, N, CP);
  }, [=](Node* y) {
    tp::accK<<<blocks(y->n()), 256, 0, STREAM>>>(p1->g, y->g, y->n());
    pairQKAddBK<<<blocks((size_t)N * WK * CP), 256, 0, STREAM>>>(y->g, cq->g, ck->g, N, CP);
  });
  Node* h = p2;
  for (int k : {1, 3, 5}) h = t.linear(t.relu(h), W.get(SM + af + "p_mlp_" + std::to_string(k) + "/weight"), nullptr, CP);
  x.p = t.add(p2, h);
  // the pairwise conditioner
  Param &rP = W.get("rel_pos/embed_rel_pos/weight"), &rT = W.get("rel_pos/embed_rel_token/weight"),
        &rE = W.get("rel_pos/embed_same_entity/weight"), &rC = W.get("rel_pos/embed_rel_chain/weight");
  Node* rel = t.custom(P, C, [&](Node* y) {
    relEmbK<<<blocks(P * C), 256, 0, STREAM>>>(b.asym, b.ent, b.res, b.sym, rP.w, rT.w, rE.w, rC.w, y->v, T, C);
  }, [=, gP = &rP, gT = &rT, gE = &rE, gC = &rC](Node* y) {
    relEmbBK<<<blocks(P * C), 256, 0, STREAM>>>(y->g, b.asym, b.ent, b.res, b.sym, gP->g, gT->g, gE->g, gC->g, T, C);
  });
  Node* z = t.linear(t.layerNorm(t.concat(zTrunk, rel), &W.get(SM + pc + "dim_pairwise_init_proj_0/weight"),
                                 W.opt(SM + pc + "dim_pairwise_init_proj_0/bias")),
                     W.get(SM + pc + "dim_pairwise_init_proj_1/weight"), nullptr, C);
  for (int i = 0; i < 2; ++i) z = transition(g, z, pc + "transitions_" + std::to_string(i) + "/", -1);
  x.z = z;
  // the per-layer biases
  const std::string ab = "layers/atom_block/transformer_layer/attention/attention_layer/pair_bias_attn/",
                    tb = "layers/token_block/attention/pair_bias_attn/";
  const int Ha = W.dim(SM + ab + "proj_z/weight", 1), Ht = W.dim(SM + tb + "proj_z/weight", 1);
  for (int l = 0; l < layers; ++l) { x.wbias.push_back(windowBias(g, x.p, l, N, Ha)); x.tbias.push_back(tokenBias(g, z, l, T, Ht)); }
  // the single conditioner's static half; a's trunk half
  const int Dt = W.dim(SM + sc + "single_embed/weight", 0);
  x.sc0 = t.linear(t.layerNorm(t.concat(sTrunk, sInputs), &W.get(SM + sc + "norm_single/weight"), W.opt(SM + sc + "norm_single/bias")),
                   W.get(SM + sc + "single_embed/weight"), W.opt(SM + sc + "single_embed/bias"), Dt);
  x.a0 = t.linear(t.layerNorm(sTrunk, &W.get(SM + "s_trunk_to_a_0/weight"), W.opt(SM + "s_trunk_to_a_0/bias")),
                  W.get(SM + "s_trunk_to_a_1/weight"), nullptr, Dt);
  return x;
}

// D(x, sigma): x [N, 3] a node -> den [N, 3]
inline Node* denoise(Graph& g, const Ctx& c, Node* x, double sigma, int layers) {
  Tape& t = g.t; Weights& W = g.W; const Batch& b = g.b;
  const int T = b.T, N = b.N, Ca = 128, Dt = 768, Dc = Dt + Ca;
  const double sd = 16;
  const double cIn = 1 / sqrt(sigma * sigma + sd * sd), tt = 0.25 * log(sigma / sd), cSkip = sd * sd / (sigma * sigma + sd * sd),
               cOut = sigma * sd / sqrt(sd * sd + sigma * sigma);
  Node* r = t.maskRows(t.scale(x, (real)cIn), b.atomPad);
  // the single conditioning: sc0 (and a zero row after its T) + fourier, two transitions on the real rows
  const std::string sc = "single_conditioner/";
  const int F = W.dim(SM + sc + "fourier_embed/proj/bias", 0);
  Param &fw = W.get(SM + sc + "fourier_embed/proj/weight"), &fb = W.get(SM + sc + "fourier_embed/proj/bias");
  Node* four = t.custom(1, F, [&](Node* y) { fourK<<<blocks(F), 256, 0, STREAM>>>(fw.w, fb.w, (real)tt, y->v, F); },
                        [=, pw = &fw, pb = &fb](Node* y) { fourBK<<<blocks(F), 256, 0, STREAM>>>(pw->w, pb->w, (real)tt, y->g, pw->g, pb->g, F); });
  four = t.layerNorm(four, &W.get(SM + sc + "norm_fourier/weight"), W.opt(SM + sc + "norm_fourier/bias"));
  Node* fr = t.linear(four, W.get(SM + sc + "fourier_to_single/weight"), nullptr, Dt);
  std::vector<int> rowsH(T + 1), realH(T + 1, 1), zeroH(T + 1, 0);
  for (int i = 0; i <= T; ++i) rowsH[i] = i < T ? i : -1;
  realH[T] = 0;
  int* rowsD = (int*)ARENA.get(T + 1); int* realD = (int*)ARENA.get(T + 1); int* zeroD = (int*)ARENA.get(T + 1);
  GK(cudaMemcpyAsync(rowsD, rowsH.data(), (T + 1) * 4, cudaMemcpyHostToDevice, STREAM));
  GK(cudaMemcpyAsync(realD, realH.data(), (T + 1) * 4, cudaMemcpyHostToDevice, STREAM));
  GK(cudaMemcpyAsync(zeroD, zeroH.data(), (T + 1) * 4, cudaMemcpyHostToDevice, STREAM));
  Node* s0 = t.gather(c.sc0, rowsD, T + 1);                    // [T + 1, Dt], the last row zero
  Node* s = t.add(s0, t.maskRows(t.gather(fr, zeroD, T + 1), realD));
  for (int i = 0; i < 2; ++i) {
    const std::string p = sc + "transitions_" + std::to_string(i) + "/";
    Node* nn = t.layerNorm(s, &W.get(SM + p + "norm/weight"), W.opt(SM + p + "norm/bias"));
    s = t.add(s, t.maskRows(swiglu(g, nn, p, -1), realD));
  }
  Node* cond = s;
  // the two streams' initial states
  const std::string af = "atom_feature_embedder/";
  Node* q = t.add(c.q0, t.linear(r, W.get(SM + af + "r_to_q_trans/weight"), nullptr, Ca));
  Node* a = t.add(c.a0, t.linear(t.maskRows(t.gather(r, b.centre, T), b.tokPad), W.get(SM + "r_token_to_a/weight"), nullptr, Dt));
  std::vector<int> identH(T); for (int i = 0; i < T; ++i) identH[i] = i;
  int* identD = (int*)ARENA.get(T); GK(cudaMemcpyAsync(identD, identH.data(), T * 4, cudaMemcpyHostToDevice, STREAM));
  Node* condTok = t.gather(cond, identD, T);
  const std::string L = "layers/", ab = "attention/attention_layer/pair_bias_attn/", tb = "attention/pair_bias_attn/";
  for (int l = 0; l < layers; ++l) {
    const std::string ap = L + "atom_block/transformer_layer/", tpf = L + "token_block/";
    Node* condA = t.gather(layerCondition(g, cond, ap, l, 6 * Ca), b.atomRowCond, N);
    q = conditioned(g, q, condA, ap, ab, l, c.wbias[l], AttnShape{true, N, 4, 32, WK, b.atomPad});
    Node* cT = layerCondition(g, condTok, tpf, l, 6 * Dt);
    a = conditioned(g, a, cT, tpf, tb, l, c.tbias[l], AttnShape{false, T, 16, 48, T, b.tokPad});
    const std::string cp = L + "coupling/";
    Node* qc = t.gather(t.linear(t.silu(t.layerNorm(cond, &W.get(SM + cp + "q_coupling_cond_0/weight", l), W.opt(SM + cp + "q_coupling_cond_0/bias", l))),
                                 W.get(SM + cp + "q_coupling_cond_2/weight", l), W.opt(SM + cp + "q_coupling_cond_2/bias", l), 2 * Dc + Ca),
                        b.atomRowCond, N);
    Node* scn = t.linear(t.silu(t.layerNorm(condTok, &W.get(SM + cp + "s_coupling_cond_0/weight", l), W.opt(SM + cp + "s_coupling_cond_0/bias", l))),
                         W.get(SM + cp + "s_coupling_cond_2/weight", l), W.opt(SM + cp + "s_coupling_cond_2/bias", l), 2 * Dc + Dt);
    Node* sqIn = t.concat(t.gather(a, b.atomTok, N), q);
    Node* sToQ = swiglu(g, t.adaLn(sqIn, qc, 0, Dc), cp + "s_to_q_mixer/", l);
    Node* qsIn = t.concat(t.gather(q, b.centre, T), a);
    Node* qToS = swiglu(g, t.adaLn(qsIn, scn, 0, Dc), cp + "q_to_s_mixer/", l);
    a = t.gatedAdd(a, qToS, scn, 2 * Dc);
    q = t.gatedAdd(q, sToQ, qc, 2 * Dc);
  }
  const std::string ro = "readout/";
  const int Hd = W.dim(SM + ro + "q_to_hidden/weight", 0);
  Node* h = t.add(t.linear(q, W.get(SM + ro + "q_to_hidden/weight"), nullptr, Hd),
                  t.gather(t.linear(a, W.get(SM + ro + "s_to_hidden/weight"), nullptr, Hd), b.atomTok, N));
  Node* out = t.linear(t.layerNorm(h, &W.get(SM + ro + "to_out_0/weight"), nullptr), W.get(SM + ro + "to_out_1/weight"), nullptr, 3);
  return t.add(t.scale(x, (real)cSkip), t.scale(out, (real)cOut));
}

}  // namespace be2
}  // namespace grad
