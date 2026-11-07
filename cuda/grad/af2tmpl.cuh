// AlphaFold 2's template embedders for training (cuda/af2/src/templates.cuh is the forward it follows):
//
//   the multimer's:  act_t = F_t @ [W0..W7] + sum(b0..b8) + LN(pair) @ W8;  act_t = stack(act_t);
//                    pair += relu(sum_t LN(act_t) / T) @ Wout + bout
//   the monomer's:   act_t = F_t @ embedding2d + b;  rep_t = LN(stack(act_t));
//                    pair += (sum_t softmax_t(q . k_t) v_t) @ Wo + bo,  q = pair Wq / sqrt(D), k_t, v_t = rep_t Wk, Wv
//   their rows:      the template torsion features -> relu(F W1 + b1) W2 + b2 (the extra MSA rows)
//
// F_t [L L, 88] is the template's geometry as both checkpoints read it - the distogram's one-hot (39), the pseudo-beta
// mask, both residues' aatype one-hots (22 each), the unit vector (3) and the backbone mask - the multimer's weighted
// by the pseudo-beta mask where the checkpoint does, the monomer's by the backbone mask (its unit vector zero, as every
// shipped monomer config has it). A template is an input: F takes no gradient.
//
// The stack is the Evoformer's pair half (af2.cuh's modules): the multimer's order triangle out, in, attention start,
// end, transition; the monomer's both attentions first. Saved: each template's stack input and output; the blocks
// are replayed one at a time before their backward.
#pragma once
#include "af2.cuh"

namespace grad {
namespace af2 {

// ---------------------------------------------------------------- the pair-only block
struct PairBlock {
  TriMul triOut, triIn;
  TriAttn triStart, triEnd;
  Transition trans;
  bool attnFirst = false;      // the monomer's template stack
  int Cz = 64, H = 4;
};
// every parameter under `pre` (haiku's names); stacked weights are the caller's (its f slices them)
inline void forEachParam(PairBlock& b, const std::string& pre, int Cz, int H, int factor,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  auto ln = [&](const std::string& n, LayerNorm& l, int C) { l.C = C; f(pre + n + "/scale", l.g, C); f(pre + n + "/offset", l.b, C); };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; f(pre + n + "/weights", l.w, (size_t)K * N); f(pre + n + "/bias", l.b, N); };
  b.Cz = Cz; b.H = H;
  for (auto* t : {&b.triOut, &b.triIn}) {
    std::string n = t == &b.triOut ? "triangle_multiplication_outgoing" : "triangle_multiplication_incoming";
    t->C = Cz; t->I = Cz; t->outgoing = t == &b.triOut;
    ln(n + "/left_norm_input", t->lnIn, Cz);
    ln(n + "/center_norm", t->lnCenter, Cz);
    lin(n + "/projection", t->proj, Cz, 2 * Cz);
    lin(n + "/gate", t->gate, Cz, 2 * Cz);
    lin(n + "/output_projection", t->out, Cz, Cz);
    lin(n + "/gating_linear", t->gating, Cz, Cz);
  }
  for (auto* t : {&b.triStart, &b.triEnd}) {
    std::string n = t == &b.triStart ? "triangle_attention_starting_node" : "triangle_attention_ending_node";
    t->perColumn = t == &b.triEnd;
    ln(n + "/query_norm", t->qn, Cz);
    f(pre + n + "/feat_2d_weights", t->feat2d, (size_t)Cz * H);
    Attention& a = t->at;
    a.C = Cz; a.H = H; a.D = Cz / H; a.O = Cz; a.gating = true;
    const std::string A = pre + n + "/attention";
    f(A + "/query_w", a.wq, (size_t)Cz * Cz); f(A + "/key_w", a.wk, (size_t)Cz * Cz); f(A + "/value_w", a.wv, (size_t)Cz * Cz);
    f(A + "/gating_w", a.wg, (size_t)Cz * Cz); f(A + "/gating_b", a.bg, Cz);
    f(A + "/output_w", a.wo, (size_t)Cz * Cz); f(A + "/output_b", a.bo, Cz);
  }
  ln("pair_transition/input_layer_norm", b.trans.ln, Cz);
  lin("pair_transition/transition1", b.trans.l1, Cz, factor * Cz);
  lin("pair_transition/transition2", b.trans.l2, factor * Cz, Cz);
}
struct TaSave { LnSave q; real *xn, *nb, *maskT; AttnSave a; };
// out = pair + the triangle attention's update (starting node per row, ending node per column)
inline TaSave triAttnForward(TriAttn& t, int H, const real* pair, const real* pairMask, real* out, int L, int Cz) {
  TaSave s{};
  const size_t pr = (size_t)L * L;
  const real* x = pair; const real* mask = pairMask;
  if (t.perColumn) {
    real* xt = ARENA.get(pr * Cz); swap01(pair, xt, L, L, Cz); x = xt;
    s.maskT = ARENA.get(pr); transpose(pairMask, s.maskT, L, L); mask = s.maskT;
  }
  s.xn = ARENA.get(pr * Cz); s.q = lnForward(t.qn, x, s.xn, pr);
  s.nb = ARENA.get(pr * H); pairBias(s.xn, t.feat2d.w, s.nb, L, Cz, H);
  real* u = ARENA.get(pr * Cz);                 // (no scratch here: the attention's saves are allocated after it)
  s.a = attnForward(t.at, s.xn, mask, s.nb, u, L, L);
  if (out != pair) GK(cudaMemcpyAsync(out, pair, pr * Cz * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  if (t.perColumn) swap01(u, out, L, L, Cz, true); else add(out, u, pr * Cz);
  return s;
}
// dp: the output's gradient, the input's added to it (the residual is the identity)
inline void triAttnBackward(TriAttn& t, int H, const TaSave& s, const real* pairMask, real* dp, int L, int Cz) {
  Scratch sc;
  const size_t pr = (size_t)L * L;
  const real* du = dp;
  if (t.perColumn) { real* d = ARENA.get(pr * Cz); swap01(dp, d, L, L, Cz); du = d; }
  real* dnb = ARENA.zeros(pr * H);
  real* dxn = ARENA.get(pr * Cz);
  attnBackward(t.at, s.a, t.perColumn ? s.maskT : pairMask, s.nb, du, dxn, dnb, false);
  pairBiasBackward(s.xn, t.feat2d, dnb, dxn, L, Cz, H, true);
  if (t.perColumn) {
    real* dx = ARENA.get(pr * Cz); lnBackward(t.qn, s.q, dxn, dx, pr, false);
    swap01(dx, dp, L, L, Cz, true);
  } else lnBackward(t.qn, s.q, dxn, dp, pr, true);
}
struct PairBlockSave { TriMulSave tOut, tIn; TaSave st, en; TransSave tr; };
// z in place: one block
inline PairBlockSave pairBlockForward(PairBlock& b, real* z, const real* pairMask, int L) {
  PairBlockSave s{};
  const size_t pr = (size_t)L * L, PC = pr * b.Cz;
  auto mul = [&](TriMul& t, TriMulSave& sv) {
    real* in = cloneR(z, PC);
    real* u = ARENA.get(PC);
    sv = triMulForward(t, in, pairMask, u, L);
    add(z, u, PC);
  };
  auto att = [&](TriAttn& t, TaSave& sv) { real* in = cloneR(z, PC); sv = triAttnForward(t, b.H, in, pairMask, z, L, b.Cz); };
  if (b.attnFirst) { att(b.triStart, s.st); att(b.triEnd, s.en); mul(b.triOut, s.tOut); mul(b.triIn, s.tIn); }
  else { mul(b.triOut, s.tOut); mul(b.triIn, s.tIn); att(b.triStart, s.st); att(b.triEnd, s.en); }
  real* in = cloneR(z, PC);
  real* u = ARENA.get(PC);
  s.tr = transForward(b.trans, in, u, pr);
  add(z, u, PC);
  return s;
}
// dz: the block's output's gradient, replaced by its input's
inline void pairBlockBackward(PairBlock& b, const PairBlockSave& s, const real* pairMask, real* dz, int L) {
  const size_t PC = (size_t)L * L * b.Cz;
  transBackward(b.trans, s.tr, dz, dz, true);
  auto mul = [&](TriMul& t, const TriMulSave& sv) { Scratch sc; real* du = cloneR(dz, PC); triMulBackward(t, sv, pairMask, du, dz); };
  auto att = [&](TriAttn& t, const TaSave& sv) { triAttnBackward(t, b.H, sv, pairMask, dz, L, b.Cz); };
  if (b.attnFirst) { mul(b.triIn, s.tIn); mul(b.triOut, s.tOut); att(b.triEnd, s.en); att(b.triStart, s.st); }
  else { att(b.triEnd, s.en); att(b.triStart, s.st); mul(b.triIn, s.tIn); mul(b.triOut, s.tOut); }
}
inline void pairStackForward(std::vector<PairBlock>& bs, real* z, const real* pairMask, int L) {
  for (auto& b : bs) { Scratch sc; pairBlockForward(b, z, pairMask, L); }
}
// in: the stack's input; dz: its output's gradient, replaced by its input's
inline void pairStackBackward(std::vector<PairBlock>& bs, const real* in, real* dz, const real* pairMask, int L) {
  Scratch sc;
  const size_t PC = (size_t)L * L * bs[0].Cz;
  std::vector<real*> ins(bs.size());
  for (size_t k = 0; k < bs.size(); ++k) {
    ins[k] = k == 0 ? const_cast<real*>(in) : cloneR(ins[k - 1], PC);
    if (k > 0) { Scratch t; pairBlockForward(bs[k - 1], ins[k], pairMask, L); }
  }
  for (size_t k = bs.size(); k-- > 0;) {
    Scratch t;
    real* z = cloneR(ins[k], PC);
    PairBlockSave s = pairBlockForward(bs[k], z, pairMask, L);
    pairBlockBackward(bs[k], s, pairMask, dz, L);
  }
}

// ---------------------------------------------------------------- a template's pair features
// tAatype [L], tPos [L, 37, 3], tMask [L, 37] -> F [L L, 88], in either checkpoint's form
__device__ inline void tmplFrame(const real* p, real* R, real* t) {     // N, CA, C -> R (columns e0 e1 e2), t = CA
  real e0[3], e1[3];
  for (int k = 0; k < 3; ++k) { e0[k] = p[6 + k] - p[3 + k]; e1[k] = p[k] - p[3 + k]; t[k] = p[3 + k]; }
  real n0 = 1 / sqrt(fmax(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2], (real)1e-12));
  for (int k = 0; k < 3; ++k) e0[k] *= n0;
  real c = e1[0] * e0[0] + e1[1] * e0[1] + e1[2] * e0[2];
  for (int k = 0; k < 3; ++k) e1[k] -= c * e0[k];
  real n1 = 1 / sqrt(fmax(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2], (real)1e-12));
  for (int k = 0; k < 3; ++k) e1[k] *= n1;
  real e2[3] = {e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0]};
  for (int k = 0; k < 3; ++k) { R[k * 3] = e0[k]; R[k * 3 + 1] = e1[k]; R[k * 3 + 2] = e2[k]; }
}
__global__ void tmplPairFeatK(const int* tAatype, const real* tPos, const real* tMask, real* F, int L, bool multimer) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)L * L) return;
  int i = (int)(ij / L), j = (int)(ij % L);
  int rawAi = tAatype[i], rawAj = tAatype[j];
  int pi = rawAi == 7 ? 1 : 3, pj = rawAj == 7 ? 1 : 3;
  real pb = tMask[(size_t)i * 37 + pi] * tMask[(size_t)j * 37 + pj];
  real d2 = 0;
  for (int k = 0; k < 3; ++k) { real d = tPos[((size_t)i * 37 + pi) * 3 + k] - tPos[((size_t)j * 37 + pj) * 3 + k]; d2 += d * d; }
  int dbin = -1;
  if (!multimer || pb != 0)
    for (int b = 0; b < 39; ++b) {
      float lo = 3.25f + (50.75f - 3.25f) * b / 38.f, hi = 3.25f + (50.75f - 3.25f) * (b + 1) / 38.f;
      if ((float)d2 > lo * lo && (float)d2 < (b + 1 < 39 ? hi * hi : 1e8f)) { dbin = b; break; }
    }
  auto bbm = [&](int r) { return tMask[(size_t)r * 37] * tMask[(size_t)r * 37 + 1] * tMask[(size_t)r * 37 + 2]; };
  int ai = min(max(rawAi, 0), 21), aj = min(max(rawAj, 0), 21);
  real* f = F + ij * 88;
  for (int k = 0; k < 88; ++k) f[k] = 0;
  if (multimer) {
    real R[9], ti[3], tj[3], Rj[9];
    tmplFrame(tPos + (size_t)i * 111, R, ti); tmplFrame(tPos + (size_t)j * 111, Rj, tj);
    real d[3] = {tj[0] - ti[0], tj[1] - ti[1], tj[2] - ti[2]}, v[3];
    for (int a = 0; a < 3; ++a) v[a] = R[a] * d[0] + R[3 + a] * d[1] + R[6 + a] * d[2];
    real nv = 1 / sqrt(fmax(v[0] * v[0] + v[1] * v[1] + v[2] * v[2], (real)1e-12));
    real bb = sqrt(bbm(i) * bbm(j));
    if (dbin >= 0) f[dbin] = pb;
    f[39] = pb; f[40 + aj] = 1; f[62 + ai] = 1;
    f[84] = v[0] * nv * bb; f[85] = v[1] * nv * bb; f[86] = v[2] * nv * bb; f[87] = bb;
  } else {
    real bb = bbm(i) * bbm(j);
    if (dbin >= 0) f[dbin] = bb;
    f[39] = pb * bb; f[40 + aj] = bb; f[62 + ai] = bb; f[87] = bb;
  }
}

// ---------------------------------------------------------------- the embedders
struct Templates {
  bool multimer = true;
  // the multimer's
  LayerNorm qln;
  Linear w8, outLin;
  Param w[8], b[8];                  // template_pair_embedding_0..7: [39|1|22|22|1|1|1|1, C], their biases
  // the monomer's
  Linear emb;
  Param qw, kw, vw, ow, ob;          // the pointwise attention: [Cz, H D], [C, H D] x2, [H D, Cz], [Cz]
  int H = 0, D = 0;
  std::vector<PairBlock> blocks;
  LayerNorm outLn;
  int C = 64, Cz = 128;
};
inline const int TMPL_ROWS[8] = {39, 1, 22, 22, 1, 1, 1, 1};
// every parameter; block b's stacked tensors are named "<haiku name>@<b>"
inline void forEachParam(Templates& m, bool multimer, int nb, int H, int factor, int attH, int attD,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string TE = "evoformer/template_embedding/", S = TE + "single_template_embedding/";
  const int C = m.C, Cz = m.Cz;
  m.multimer = multimer;
  auto ln = [&](const std::string& n, LayerNorm& l, int Cc) { l.C = Cc; f(n + "/scale", l.g, Cc); f(n + "/offset", l.b, Cc); };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; f(n + "/weights", l.w, (size_t)K * N); f(n + "/bias", l.b, N); };
  m.blocks.resize(nb);
  const std::string stack = multimer ? S + "template_embedding_iteration/" : S + "template_pair_stack/__layer_stack_no_state/";
  for (int b = 0; b < nb; ++b) {
    m.blocks[b].attnFirst = !multimer;
    forEachParam(m.blocks[b], stack, C, H, factor, [&](const std::string& n, Param& p, size_t len) { f(n + "@" + std::to_string(b), p, len); });
  }
  ln(S + "output_layer_norm", m.outLn, C);
  if (multimer) {
    ln(S + "query_embedding_norm", m.qln, Cz);
    lin(S + "template_pair_embedding_8", m.w8, Cz, C);
    for (int k = 0; k < 8; ++k) {
      const std::string n = S + "template_pair_embedding_" + std::to_string(k);
      f(n + "/weights", m.w[k], (size_t)TMPL_ROWS[k] * C); f(n + "/bias", m.b[k], C);
    }
    lin(TE + "output_linear", m.outLin, C, Cz);
  } else {
    lin(S + "embedding2d", m.emb, 88, C);
    m.H = attH; m.D = attD;
    const std::string A = TE + "attention/";
    const size_t HD = (size_t)attH * attD;
    f(A + "query_w", m.qw, Cz * HD); f(A + "key_w", m.kw, C * HD); f(A + "value_w", m.vw, C * HD);
    f(A + "output_w", m.ow, HD * Cz); f(A + "output_b", m.ob, Cz);
  }
}
struct TemplatesSave {
  std::vector<real*> ins, outs, reps;     // per template: the stack's input, its output, (monomer) LN of it
  std::vector<LnSave> lo;
  real *qn = nullptr, *qterm = nullptr, *summed = nullptr, *wcat = nullptr;
  LnSave lq;
  real *q = nullptr, *k = nullptr, *v = nullptr, *w = nullptr, *o = nullptr;   // the monomer's attention
  int L = 0, T = 0;
};
__global__ void concatRowsK(const real* const* parts, const int* rows, int nParts, real* out, int C) {
  int r = blockIdx.x, c = threadIdx.x;
  if (c >= C) return;
  int at = r;
  for (int k = 0; k < nParts; ++k) { if (at < rows[k]) { out[(size_t)r * C + c] = parts[k][(size_t)at * C + c]; return; } at -= rows[k]; }
}
__global__ void reluMeanK(const real* x, real* y, real s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = fmax((real)0, x[t] * s);
}
__global__ void reluMeanBwdK(const real* x, real* d, real s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) d[t] = x[t] * s > 0 ? d[t] * s : (real)0;
}
// softmax over T templates per (pair, head): w [pairs, H, T], o [pairs, H D]
__global__ void tmplPointAttnK(const real* q, const real* k, const real* v, real* w, real* o, size_t pairs, int T, int H, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * H) return;
  size_t p = t / H; int h = (int)(t % H); const size_t W = (size_t)H * D;
  real* wr = w + t * T;
  real mx = -1e300;
  for (int s = 0; s < T; ++s) {
    real a = 0;
    for (int d = 0; d < D; ++d) a += q[p * W + h * D + d] * k[((size_t)s * pairs + p) * W + h * D + d];
    wr[s] = a; mx = fmax(mx, a);
  }
  real sum = 0;
  for (int s = 0; s < T; ++s) { wr[s] = exp(wr[s] - mx); sum += wr[s]; }
  for (int s = 0; s < T; ++s) wr[s] /= sum;
  for (int d = 0; d < D; ++d) {
    real a = 0;
    for (int s = 0; s < T; ++s) a += wr[s] * v[((size_t)s * pairs + p) * W + h * D + d];
    o[p * W + h * D + d] = a;
  }
}
__global__ void tmplPointAttnBwdK(const real* q, const real* k, const real* v, const real* w, const real* dout, real* dq,
                                  real* dk, real* dv, size_t pairs, int T, int H, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * H) return;
  size_t p = t / H; int h = (int)(t % H); const size_t W = (size_t)H * D;
  const real* wr = w + t * T;
  real dl[8], sw = 0;
  for (int s = 0; s < T; ++s) {
    real a = 0;
    for (int d = 0; d < D; ++d) {
      size_t at = ((size_t)s * pairs + p) * W + h * D + d;
      a += dout[p * W + h * D + d] * v[at];
      dv[at] = wr[s] * dout[p * W + h * D + d];
    }
    dl[s] = a; sw += wr[s] * a;
  }
  for (int d = 0; d < D; ++d) dq[p * W + h * D + d] = 0;
  for (int s = 0; s < T; ++s) {
    real g = wr[s] * (dl[s] - sw);
    for (int d = 0; d < D; ++d) {
      size_t at = ((size_t)s * pairs + p) * W + h * D + d;
      dq[p * W + h * D + d] += g * k[at];
      dk[at] = g * q[p * W + h * D + d];
    }
  }
}
// the eight pair-input weights stacked into one [88, C], and back
inline real* stackedPairWeights(Templates& m, bool grads) {
  real* out = ARENA.get((size_t)88 * m.C);
  std::vector<const real*> parts(8); std::vector<int> rows(TMPL_ROWS, TMPL_ROWS + 8);
  for (int k = 0; k < 8; ++k) parts[k] = grads ? m.w[k].g : m.w[k].w;
  const real** dp = (const real**)ARENA.get((8 * sizeof(real*) + sizeof(real) - 1) / sizeof(real));
  int* dr = (int*)ARENA.get(8);
  GK(cudaMemcpyAsync(dp, parts.data(), 8 * sizeof(real*), cudaMemcpyHostToDevice, STREAM));
  GK(cudaMemcpyAsync(dr, rows.data(), 8 * sizeof(int), cudaMemcpyHostToDevice, STREAM));
  concatRowsK<<<88, std::max(m.C, 32), 0, STREAM>>>(dp, dr, 8, out, m.C);
  GK(cudaStreamSynchronize(STREAM));           // (the pointer table is host memory)
  return out;
}
// F [T] of [L L, 88], pair [L L, Cz] -> out (+)= the template term (out may be pair)
inline TemplatesSave templatesForward(Templates& m, const std::vector<const real*>& F, const real* pair, const real* pairMask,
                                      int L, real* out) {
  TemplatesSave s{}; s.L = L; s.T = (int)F.size();
  const size_t pr = (size_t)L * L, PC = pr * m.C;
  const int T = s.T;
  s.ins.assign(T, nullptr); s.outs.assign(T, nullptr); s.lo.assign(T, LnSave{}); s.reps.assign(T, nullptr);
  if (m.multimer) {
    s.qn = ARENA.get(pr * m.Cz); s.lq = lnForward(m.qln, pair, s.qn, pr);
    s.qterm = ARENA.get(PC); m.w8.forward(s.qn, s.qterm, pr);
    for (int k = 0; k < 8; ++k) biasK<<<blocks(PC), 256, 0, STREAM>>>(s.qterm, m.b[k].w, pr, m.C);
    s.wcat = stackedPairWeights(m, false);
    s.summed = ARENA.zeros(PC);
  }
  real* y = m.multimer ? ARENA.get(PC) : nullptr;
  for (int t = 0; t < T; ++t) {
    s.ins[t] = ARENA.get(PC);
    if (m.multimer) {
      GK(cudaMemcpyAsync(s.ins[t], s.qterm, PC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
      mm(F[t], s.wcat, s.ins[t], (int)pr, 88, m.C, 1);
    } else m.emb.forward(F[t], s.ins[t], pr);
    s.outs[t] = cloneR(s.ins[t], PC);
    pairStackForward(m.blocks, s.outs[t], pairMask, L);
    if (m.multimer) {
      s.lo[t] = lnForward(m.outLn, s.outs[t], y, pr);
      add(s.summed, y, PC);
    } else {
      s.reps[t] = ARENA.get(PC);
      s.lo[t] = lnForward(m.outLn, s.outs[t], s.reps[t], pr);
    }
  }
  if (m.multimer) {
    Scratch sc;
    real* h = ARENA.get(PC); reluMeanK<<<blocks(PC), 256, 0, STREAM>>>(s.summed, h, (real)1 / T, PC);
    m.outLin.forward(h, out, pr, 1);
    return s;
  }
  // the monomer's attention over the templates (T = 1: the softmax is one, the output the value)
  const int HD = m.H * m.D;
  s.q = ARENA.get(pr * HD); mm(pair, m.qw.w, s.q, (int)pr, m.Cz, HD, 0, (real)(1 / sqrt((double)m.D)));
  s.k = ARENA.get((size_t)T * pr * HD); s.v = ARENA.get((size_t)T * pr * HD);
  for (int t = 0; t < T; ++t) {
    mm(s.reps[t], m.kw.w, s.k + (size_t)t * pr * HD, (int)pr, m.C, HD);
    mm(s.reps[t], m.vw.w, s.v + (size_t)t * pr * HD, (int)pr, m.C, HD);
  }
  s.w = ARENA.get(pr * m.H * T); s.o = ARENA.get(pr * HD);
  tmplPointAttnK<<<blocks(pr * m.H), 256, 0, STREAM>>>(s.q, s.k, s.v, s.w, s.o, pr, T, m.H, m.D);
  mm(s.o, m.ow.w, out, (int)pr, HD, m.Cz, 1);
  biasK<<<blocks(pr * m.Cz), 256, 0, STREAM>>>(out, m.ob.w, pr, m.Cz);
  return s;
}
// dout -> every parameter; dpair added (the query's path: the residual into the pair is the caller's)
inline void templatesBackward(Templates& m, const std::vector<const real*>& F, const TemplatesSave& s, const real* pair,
                              const real* pairMask, const real* dout, real* dpair) {
  Scratch scratch;
  const int L = s.L, T = s.T;
  const size_t pr = (size_t)L * L, PC = pr * m.C;
  std::vector<real*> dy(T);
  real* dWcat = m.multimer ? ARENA.zeros((size_t)88 * m.C) : nullptr;
  real* dq = m.multimer ? ARENA.zeros(PC) : nullptr;
  if (m.multimer) {
    real* h = ARENA.get(PC); reluMeanK<<<blocks(PC), 256, 0, STREAM>>>(s.summed, h, (real)1 / T, PC);
    real* ds = ARENA.get(PC); m.outLin.backward(h, dout, ds, pr);
    reluMeanBwdK<<<blocks(PC), 256, 0, STREAM>>>(s.summed, ds, (real)1 / T, PC);
    for (int t = 0; t < T; ++t) dy[t] = ds;
  } else {
    const int HD = m.H * m.D;
    colSum(m.ob.g, dout, pr, m.Cz);
    mmAT(s.o, dout, m.ow.g, (int)pr, HD, m.Cz);
    real* dO = ARENA.get(pr * HD); mmBT(dout, m.ow.w, dO, (int)pr, HD, m.Cz);
    real *dQ = ARENA.get(pr * HD), *dK = ARENA.get((size_t)T * pr * HD), *dV = ARENA.get((size_t)T * pr * HD);
    tmplPointAttnBwdK<<<blocks(pr * m.H), 256, 0, STREAM>>>(s.q, s.k, s.v, s.w, dO, dQ, dK, dV, pr, T, m.H, m.D);
    const real sc = (real)(1 / sqrt((double)m.D));
    mmAT(pair, dQ, m.qw.g, (int)pr, m.Cz, HD, 1, sc);
    mmBT(dQ, m.qw.w, dpair, (int)pr, m.Cz, HD, 1, sc);
    for (int t = 0; t < T; ++t) {
      dy[t] = ARENA.get(PC);
      const real *dk = dK + (size_t)t * pr * HD, *dv = dV + (size_t)t * pr * HD;
      mmAT(s.reps[t], dk, m.kw.g, (int)pr, m.C, HD); mmAT(s.reps[t], dv, m.vw.g, (int)pr, m.C, HD);
      mmBT(dk, m.kw.w, dy[t], (int)pr, m.C, HD); mmBT(dv, m.vw.w, dy[t], (int)pr, m.C, HD, 1);
    }
  }
  for (int t = 0; t < T; ++t) {
    Scratch sc;
    real* da = ARENA.get(PC); lnBackward(m.outLn, s.lo[t], dy[t], da, pr, false);
    pairStackBackward(m.blocks, s.ins[t], da, pairMask, L);
    if (m.multimer) { add(dq, da, PC); mmAT(F[t], da, dWcat, (int)pr, 88, m.C); }
    else m.emb.backward(F[t], da, nullptr, pr);
  }
  if (!m.multimer) return;
  // the stacked weights' gradient back into the eight, the nine biases', the query's
  real* dw = dWcat;
  for (int k = 0; k < 8; ++k) { add(m.w[k].g, dw, (size_t)TMPL_ROWS[k] * m.C); dw += (size_t)TMPL_ROWS[k] * m.C; colSum(m.b[k].g, dq, pr, m.C); }
  real* dqn = ARENA.get(pr * m.Cz); m.w8.backward(s.qn, dq, dqn, pr);
  lnBackward(m.qln, s.lq, dqn, dpair, pr, true);
}

// ---------------------------------------------------------------- the template rows
// feat [rows, F] (the torsion features: cuda/af2's templateTorsionFeatK / templateChiFeatK, inputs) -> out [rows, 256]
struct TemplateRows { Linear single, proj; int F = 0; };
inline void forEachParam(TemplateRows& r, int F, const std::function<void(const std::string&, Param&, size_t)>& f) {
  r.F = F;
  r.single.K = F; r.single.N = 256; f("evoformer/template_single_embedding/weights", r.single.w, (size_t)F * 256);
  f("evoformer/template_single_embedding/bias", r.single.b, 256);
  r.proj.K = 256; r.proj.N = 256; f("evoformer/template_projection/weights", r.proj.w, 256 * 256);
  f("evoformer/template_projection/bias", r.proj.b, 256);
}
struct RowsSave { real* h; };
inline RowsSave rowsForward(TemplateRows& r, const real* feat, real* out, size_t rows) {
  RowsSave s{ARENA.get(rows * 256)};
  r.single.forward(feat, s.h, rows);
  Scratch sc;
  real* a = ARENA.get(rows * 256); reluK<<<blocks(rows * 256), 256, 0, STREAM>>>(a, s.h, rows * 256);
  r.proj.forward(a, out, rows);
  return s;
}
inline void rowsBackward(TemplateRows& r, const RowsSave& s, const real* feat, const real* dout, size_t rows) {
  Scratch sc;
  real* a = ARENA.get(rows * 256); reluK<<<blocks(rows * 256), 256, 0, STREAM>>>(a, s.h, rows * 256);
  real* da = ARENA.get(rows * 256); r.proj.backward(a, dout, da, rows);
  reluBwdK<<<blocks(rows * 256), 256, 0, STREAM>>>(da, s.h, rows * 256);
  r.single.backward(feat, da, nullptr, rows);
}

}  // namespace af2
}  // namespace grad
