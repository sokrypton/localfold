// RFD4-Proteina's autoencoder for training: the layer both halves share (LayerNorm throughout, no pair bias, a
// conditioning that is identically zero - so each layer's gammas, betas and output gates are one row each, made from
// norm_cond's bias and the projections, and those weights train through that row), the decoder's ends (the per-token
// features with their two embedding tables, the three heads) and the encoder's (the per-atom features with their three
// embedding tables, the mean / log-scale head). cuda/rfd4p/src/decoder.cuh's decode() and encoder.cuh's encode() (the
// deterministic arm) are what they are held to.
#pragma once
#include "rfd4p.cuh"

namespace grad {
namespace rfd4p {

// ---------------------------------------------------------------- the shared layer
struct AeLayer {
  LayerNorm ncA, ncT;                 // norm_cond (its bias is the whole of the normed zero conditioning)
  Linear gA, bA, gT, bT, sA, sT;      // to_gamma.0 (bias), to_beta, the two scale_output gammas (bias)
  LayerNorm node, qn, kn;
  Linear qkv, g, out;                 // to_qkv [3D] and to_g with biases, to_out_node with its bias
  SwiGluR sg;
  int D = 0, C = 0, H = 0;
};
inline void aeLayerParams(AeLayer& l, const std::string& L, int D, int C, int H, const ParamFn& f) {
  auto lin = [&](const std::string& n, Linear& x, int K, int N, bool bias) {
    x.K = K; x.N = N; x.trans = true; x.hasBias = bias; f(n + ".weight", x.w, (size_t)K * N); if (bias) f(n + ".bias", x.b, N);
  };
  auto ln = [&](const std::string& n, LayerNorm& x, int C_) { x.C = C_; f(n + ".weight", x.g, C_); f(n + ".bias", x.b, C_); };
  l.D = D; l.C = C; l.H = H;
  ln(L + ".mhba.adaln.norm_cond", l.ncA, C); lin(L + ".mhba.adaln.to_gamma.0", l.gA, C, D, true); lin(L + ".mhba.adaln.to_beta", l.bA, C, D, false);
  ln(L + ".transition.adaln.norm_cond", l.ncT, C); lin(L + ".transition.adaln.to_gamma.0", l.gT, C, D, true);
  lin(L + ".transition.adaln.to_beta", l.bT, C, D, false);
  lin(L + ".mhba.scale_output.to_adaln_zero_gamma.0", l.sA, C, D, true);
  lin(L + ".transition.scale_output.to_adaln_zero_gamma.0", l.sT, C, D, true);
  const std::string A = L + ".mhba.mha";
  ln(A + ".node_norm", l.node, D);
  lin(A + ".to_qkv", l.qkv, D, 3 * D, true); lin(A + ".to_g", l.g, D, D, true);
  ln(A + ".q_layer_norm", l.qn, D); ln(A + ".k_layer_norm", l.kn, D);
  lin(A + ".to_out_node", l.out, D, D, true);
  l.sg.D = D; l.sg.I = 4 * D;
  lin(L + ".transition.transition.swish_linear.0", l.sg.w1, D, 8 * D, false);
  lin(L + ".transition.transition.linear_out", l.sg.out, 4 * D, D, false);
}
// the layer's six conditioning rows from a zero conditioning: LN(0) = norm_cond's bias; gamma = it W + b, beta = it W;
// the output scales a zero input's projection, their biases
struct AeCond { real *nA, *nT, *gA, *bA, *gT, *bT, *sA, *sT, *zero; LnSave lA, lT; };
inline AeCond aeCondForward(AeLayer& l) {
  AeCond c{};
  c.zero = ARENA.zeros(l.C);
  c.nA = ARENA.get(l.C); c.lA = lnForward(l.ncA, c.zero, c.nA, 1);
  c.nT = ARENA.get(l.C); c.lT = lnForward(l.ncT, c.zero, c.nT, 1);
  c.gA = ARENA.get(l.D); l.gA.forward(c.nA, c.gA, 1); c.bA = ARENA.get(l.D); l.bA.forward(c.nA, c.bA, 1);
  c.gT = ARENA.get(l.D); l.gT.forward(c.nT, c.gT, 1); c.bT = ARENA.get(l.D); l.bT.forward(c.nT, c.bT, 1);
  c.sA = ARENA.get(l.D); l.sA.forward(c.zero, c.sA, 1); c.sT = ARENA.get(l.D); l.sT.forward(c.zero, c.sT, 1);
  return c;
}
// the rows' gradients (each [D]) -> the conditioning weights
struct AeCondGrad { real *gA, *bA, *gT, *bT, *sA, *sT; };
inline void aeCondBackward(AeLayer& l, const AeCond& c, const AeCondGrad& d) {
  Scratch scratch;
  real *dnA = ARENA.get(l.C), *dnT = ARENA.get(l.C);
  l.gA.backward(c.nA, d.gA, dnA, 1); l.bA.backward(c.nA, d.bA, dnA, 1, true);
  l.gT.backward(c.nT, d.gT, dnT, 1); l.bT.backward(c.nT, d.bT, dnT, 1, true);
  l.sA.backward(c.zero, d.sA, nullptr, 1); l.sT.backward(c.zero, d.sT, nullptr, 1);
  real* t = ARENA.get(l.C);
  lnBackward(l.ncA, c.lA, dnA, t, 1, false); lnBackward(l.ncT, c.lT, dnT, t, 1, false);
}
// out = mask ? xn sigmoid(g) + b : 0, g and b one row; and back (dg, db summed over rows)
__global__ void aeAdaK(const real* xn, const real* g, const real* b, const real* mask, real* out, size_t rows, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * D) return;
  int c = t % D;
  out[t] = mask[t / D] == 0 ? (real)0 : xn[t] / (1 + exp(-g[c])) + b[c];
}
__global__ void aeAdaBwdK(const real* xn, const real* g, const real* mask, const real* d, real* dxn, real* dg, real* db, size_t rows, int D) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= D) return;
  real s = 1 / (1 + exp(-g[c])), sg = 0, sb = 0;
  for (size_t r = 0; r < rows; ++r) {
    size_t t = r * D + c;
    real v = mask[r] == 0 ? (real)0 : d[t];
    dxn[t] = v * s; sg += v * xn[t] * s * (1 - s); sb += v;
  }
  dg[c] = sg; db[c] = sb;
}
// s += mask ? x sigmoid(gamma) : 0, and back
__global__ void aeScaleAddK(const real* x, const real* g, const real* mask, real* s, size_t rows, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * D || mask[t / D] == 0) return;
  s[t] += x[t] / (1 + exp(-g[t % D]));
}
__global__ void aeScaleAddBwdK(const real* x, const real* g, const real* mask, const real* d, real* dx, real* dg, size_t rows, int D) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= D) return;
  real s = 1 / (1 + exp(-g[c])), sg = 0;
  for (size_t r = 0; r < rows; ++r) {
    size_t t = r * D + c;
    if (mask[r] == 0) { dx[t] = 0; continue; }
    dx[t] = d[t] * s; sg += d[t] * x[t] * s * (1 - s);
  }
  dg[c] = sg;
}
__global__ void gateSigK(const real* o, const real* g, real* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = o[t] / (1 + exp(-g[t]));
}
__global__ void gateSigBwdK(const real* o, const real* g, const real* d, real* dO, real* dg, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  real s = 1 / (1 + exp(-g[t]));
  dO[t] = d[t] * s; dg[t] = d[t] * o[t] * s * (1 - s);
}
// q, k out of qkv [n, 3D] (v stays strided in place); their gradients back into the first 2D columns
__global__ void splitColsK(const real* qkv, real* q, real* k, real* v, int n, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * 3 * D) return;
  size_t r = t / (3 * D); int c = t % (3 * D);
  if (c < D) q[r * D + c] = qkv[t]; else if (c < 2 * D) k[r * D + c - D] = qkv[t]; else if (v) v[r * D + c - 2 * D] = qkv[t];
}
__global__ void joinColsK(const real* dq, const real* dk, real* dqkv, int n, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * 3 * D) return;
  size_t r = t / (3 * D); int c = t % (3 * D);
  if (c < D) dqkv[t] = dq[r * D + c]; else if (c < 2 * D) dqkv[t] = dk[r * D + c - D];
}
struct AeSave {
  AeCond c; LnSave l1, ln, lq, lk, l2; SgSave sg;
  real *xn1, *ada1, *nodes, *qkv, *q, *k, *gpre, *o, *og, *attn, *s1, *xn2, *ada2, *tr;
  real* lse = nullptr;   // the attention's logit max and log-sum, kept for its backward
};
inline Sdpa aeSdpa(const AeLayer& l, const real* mask, int n) {
  Sdpa a; a.B = 1; a.H = l.H; a.Q = a.K = n; a.D = l.D / l.H; a.scale = 1 / sqrt((real)a.D); a.keyMask = mask;
  return a;
}
inline AeSave aeForward(AeLayer& l, const real* s, const real* mask, real* sOut, int n) {
  AeSave v{};
  const int D = l.D; const size_t nD = (size_t)n * D;
  LayerNorm plain; plain.C = D;
  v.c = aeCondForward(l);
  v.xn1 = ARENA.get(nD); v.l1 = lnForward(plain, s, v.xn1, n);
  v.ada1 = ARENA.get(nD); aeAdaK<<<blocks(nD), 256, 0, STREAM>>>(v.xn1, v.c.gA, v.c.bA, mask, v.ada1, n, D);
  v.nodes = ARENA.get(nD); v.ln = lnForward(l.node, v.ada1, v.nodes, n);
  v.qkv = ARENA.get(nD * 3); l.qkv.forward(v.nodes, v.qkv, n);
  v.gpre = ARENA.get(nD); l.g.forward(v.nodes, v.gpre, n);
  // q and k LayerNormed over the whole D (not per head)
  real *qr = ARENA.get(nD), *kr = ARENA.get(nD);
  splitColsK<<<blocks(nD * 3), 256, 0, STREAM>>>(v.qkv, qr, kr, nullptr, n, D);
  v.q = ARENA.get(nD); v.lq = lnForward(l.qn, qr, v.q, n);
  v.k = ARENA.get(nD); v.lk = lnForward(l.kn, kr, v.k, n);
  v.o = ARENA.get(nD);
  { const Sdpa sd_ = aeSdpa(l, mask, n); v.lse = ARENA.get((size_t)2 * sd_.B * sd_.H * sd_.Q); sdpaForward(sd_, v.q, D, v.k, D, v.qkv + 2 * D, 3 * D, v.o, D, v.lse); }
  v.og = ARENA.get(nD); gateSigK<<<blocks(nD), 256, 0, STREAM>>>(v.o, v.gpre, v.og, nD);
  v.attn = ARENA.get(nD); l.out.forward(v.og, v.attn, n);
  v.s1 = cloneR(s, nD); aeScaleAddK<<<blocks(nD), 256, 0, STREAM>>>(v.attn, v.c.sA, mask, v.s1, n, D);
  v.xn2 = ARENA.get(nD); v.l2 = lnForward(plain, v.s1, v.xn2, n);
  v.ada2 = ARENA.get(nD); aeAdaK<<<blocks(nD), 256, 0, STREAM>>>(v.xn2, v.c.gT, v.c.bT, mask, v.ada2, n, D);
  v.tr = ARENA.get(nD); v.sg = sgForward(l.sg, v.ada2, mask, v.tr, n);
  GK(cudaMemcpyAsync(sOut, v.s1, nD * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  aeScaleAddK<<<blocks(nD), 256, 0, STREAM>>>(v.tr, v.c.sT, mask, sOut, n, D);
  return v;
}
// dsOut -> ds (written)
inline void aeBackward(AeLayer& l, const AeSave& v, const real* mask, const real* dsOut, real* ds, int n) {
  Scratch scratch;
  const int D = l.D; const size_t nD = (size_t)n * D;
  LayerNorm plain; plain.C = D;
  AeCondGrad cg{ARENA.get(D), ARENA.get(D), ARENA.get(D), ARENA.get(D), ARENA.get(D), ARENA.get(D)};
  real* ds1 = cloneR(dsOut, nD);
  real* dtr = ARENA.get(nD);
  aeScaleAddBwdK<<<blocks(D, 64), 64, 0, STREAM>>>(v.tr, v.c.sT, mask, dsOut, dtr, cg.sT, n, D);
  real* dada2 = ARENA.zeros(nD); sgBackward(l.sg, v.sg, mask, dtr, dada2);
  real* dxn2 = ARENA.get(nD);
  aeAdaBwdK<<<blocks(D, 64), 64, 0, STREAM>>>(v.xn2, v.c.gT, mask, dada2, dxn2, cg.gT, cg.bT, n, D);
  lnBackward(plain, v.l2, dxn2, ds1, n, true);
  // s1 = s + scale_a attn
  real* dattn = ARENA.get(nD);
  aeScaleAddBwdK<<<blocks(D, 64), 64, 0, STREAM>>>(v.attn, v.c.sA, mask, ds1, dattn, cg.sA, n, D);
  real* dog = ARENA.get(nD); l.out.backward(v.og, dattn, dog, n);
  real *dO = ARENA.get(nD), *dg = ARENA.get(nD);
  gateSigBwdK<<<blocks(nD), 256, 0, STREAM>>>(v.o, v.gpre, dog, dO, dg, nD);
  real *dq = ARENA.get(nD), *dk = ARENA.get(nD), *dqkv = ARENA.zeros(nD * 3);
  sdpaBackward(aeSdpa(l, mask, n), v.q, D, v.k, D, v.qkv + 2 * D, 3 * D, dO, D, dq, D, dk, D, dqkv + 2 * D, 3 * D, false, false, false, v.o, D, v.lse);
  real *dqr = ARENA.get(nD), *dkr = ARENA.get(nD);
  lnBackward(l.qn, v.lq, dq, dqr, n, false); lnBackward(l.kn, v.lk, dk, dkr, n, false);
  joinColsK<<<blocks(nD * 3), 256, 0, STREAM>>>(dqr, dkr, dqkv, n, D);
  real* dnodes = ARENA.get(nD); l.qkv.backward(v.nodes, dqkv, dnodes, n); l.g.backward(v.nodes, dg, dnodes, n, true);
  real* dada1 = ARENA.get(nD); lnBackward(l.node, v.ln, dnodes, dada1, n, false);
  real* dxn1 = ARENA.get(nD);
  aeAdaBwdK<<<blocks(D, 64), 64, 0, STREAM>>>(v.xn1, v.c.gA, mask, dada1, dxn1, cg.gA, cg.bA, n, D);
  lnBackward(plain, v.l1, dxn1, ds1, n, true);
  GK(cudaMemcpyAsync(ds, ds1, nD * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  aeCondBackward(l, v.c, cg);
}

// ---------------------------------------------------------------- a stack with its input projection and heads
struct AeHead { LayerNorm ln; Linear w; };
struct AeStack {
  int D = 0, C = 0, H = 0, nlayers = 0, fdim = 0;
  Linear in;
  std::vector<AeLayer> layers;
  std::vector<AeHead> heads;
};
inline void aeStackParams(AeStack& a, const std::string& root, const std::vector<std::pair<std::string, int>>& heads, const ParamFn& f) {
  a.in.K = a.fdim; a.in.N = a.D; a.in.trans = true; a.in.hasBias = false;
  f("w/" + root + ".init_repr_factory.linear_out.weight", a.in.w, (size_t)a.fdim * a.D);
  a.layers.assign(a.nlayers, AeLayer{});
  for (int i = 0; i < a.nlayers; ++i) aeLayerParams(a.layers[i], "w/" + root + ".transformer_layers." + std::to_string(i), a.D, a.C, a.H, f);
  a.heads.assign(heads.size(), AeHead{});
  for (size_t k = 0; k < heads.size(); ++k) {
    const std::string h = "w/" + root + "." + heads[k].first;
    AeHead& x = a.heads[k];
    x.ln.C = a.D; f(h + ".0.weight", x.ln.g, a.D); f(h + ".0.bias", x.ln.b, a.D);
    x.w.K = a.D; x.w.N = heads[k].second; x.w.trans = true; x.w.hasBias = false; f(h + ".1.weight", x.w.w, (size_t)a.D * heads[k].second);
  }
}
struct AeRun { std::vector<real*> s; std::vector<LnSave> hl; std::vector<real*> hn, out; };
inline AeRun aeStackForward(AeStack& a, const real* feats, const real* mask, int n) {
  AeRun r;
  const size_t nD = (size_t)n * a.D;
  r.s.assign(a.nlayers + 1, nullptr);
  for (auto& p : r.s) p = ARENA.get(nD);
  a.in.forward(feats, r.s[0], n);
  maskRowsK<<<blocks(nD), 256, 0, STREAM>>>(r.s[0], mask, n, a.D);
  for (int i = 0; i < a.nlayers; ++i) { Scratch sc; aeForward(a.layers[i], r.s[i], mask, r.s[i + 1], n); }
  for (auto& h : a.heads) {
    real* hn = ARENA.get(nD); r.hl.push_back(lnForward(h.ln, r.s.back(), hn, n)); r.hn.push_back(hn);
    real* o = ARENA.get((size_t)n * h.w.N); h.w.forward(hn, o, n); r.out.push_back(o);
  }
  return r;
}
// the heads' cotangents (any null) -> every parameter; dfeats (written)
inline void aeStackBackward(AeStack& a, const AeRun& r, const real* feats, const real* mask, int n,
                            const std::vector<const real*>& dheads, real* dfeats) {
  Scratch scratch;
  const size_t nD = (size_t)n * a.D;
  real* ds = ARENA.zeros(nD);
  for (size_t k = 0; k < a.heads.size(); ++k) {
    if (!dheads[k]) continue;
    real* dn = ARENA.get(nD); a.heads[k].w.backward(r.hn[k], dheads[k], dn, n);
    lnBackward(a.heads[k].ln, r.hl[k], dn, ds, n, true);
  }
  for (int i = a.nlayers; i-- > 0;) {
    Scratch sc;
    real* o = ARENA.get(nD);
    AeSave v = aeForward(a.layers[i], r.s[i], mask, o, n);
    real* dIn = ARENA.get(nD);
    aeBackward(a.layers[i], v, mask, ds, dIn, n);
    GK(cudaMemcpyAsync(ds, dIn, nD * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  maskRowsK<<<blocks(nD), 256, 0, STREAM>>>(ds, mask, n, a.D);
  a.in.backward(feats, ds, dfeats, n);
}

// ---------------------------------------------------------------- the decoder's features
// [CA (3) | latent | the token type's embedding (32) | the residue type's (nTypes, 0 for <M>)], zero on padding
__global__ void decFeatsK(const real* ca, const real* z, const int* tt, const int* seq, const real* typeEmb, const real* seqEmb,
                          const real* pad, real* f, int n, int latent, int nTypes, int fdim) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * fdim) return;
  int i = t / fdim, c = t % fdim;
  real v = 0;
  if (pad[i] != 0) {
    if (c < 3) v = ca[i * 3 + c];
    else if (c < 3 + latent) v = z[(size_t)i * latent + c - 3];
    else if (c < 3 + latent + 32) v = typeEmb[(size_t)tt[i] * 32 + c - 3 - latent];
    else if (seq[i] != 0) v = seqEmb[(size_t)seq[i] * nTypes + c - 3 - latent - 32];
  }
  f[t] = v;
}
__global__ void decFeatsBwdK(const real* d, const int* tt, const int* seq, const real* pad, real* dz, real* dType, real* dSeq,
                             int n, int latent, int nTypes, int fdim) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * fdim) return;
  int i = t / fdim, c = t % fdim;
  if (pad[i] == 0) return;
  if (c < 3) return;
  if (c < 3 + latent) { if (dz) dz[(size_t)i * latent + c - 3] += d[t]; }
  else if (c < 3 + latent + 32) atomicAdd(&dType[(size_t)tt[i] * 32 + c - 3 - latent], d[t]);
  else if (seq[i] != 0) atomicAdd(&dSeq[(size_t)seq[i] * nTypes + c - 3 - latent - 32], d[t]);
}

// ---------------------------------------------------------------- the encoder's features
// encoder.cuh's encoderFeatsK, a thread a (token, column): the token type's embedding, the 37 slots' positions and mask
// (absolute, and relative to a present CA), the chain index, the bond counts, an element-plus-charge embedding a slot -
// every slot gated on its atom being there
struct EncLayout { int ttDim, chargeDim, offAbs, offRel, offChain, offBonds, offElCh; };
__device__ inline real encFeatOf(const real* coors, const real* am, const int* tt, const real* chain, const real* bonds,
                                 const int* el, const int* ch, const real* ttE, const real* elE, const real* chE,
                                 const EncLayout& L, int i, int c, int* table, int* row, int* col) {
  *table = -1;
  if (c < L.ttDim) { *table = 0; *row = tt[i]; *col = c; return ttE[(size_t)tt[i] * L.ttDim + c]; }
  const bool caLive = am[(size_t)i * 37 + 1] != 0;
  auto block = [&](int off, int width) { return off >= 0 && c >= off && c < off + width; };
  if (block(L.offAbs, 148) || block(L.offRel, 148)) {
    const bool rel = block(L.offRel, 148);
    int k = c - (rel ? L.offRel : L.offAbs);
    if (k >= 111) return am[(size_t)i * 37 + k - 111] != 0 ? (real)1 : 0;
    int a = k / 3, x = k % 3;
    if (am[(size_t)i * 37 + a] == 0) return 0;
    real v = coors[((size_t)i * 37 + a) * 3 + x];
    if (rel && caLive) v -= coors[((size_t)i * 37 + 1) * 3 + x];
    return v;
  }
  if (L.offChain >= 0 && c == L.offChain) return chain[i];
  if (block(L.offBonds, 370)) {
    int k = c - L.offBonds, a = k / 10;
    return am[(size_t)i * 37 + a] != 0 ? bonds[((size_t)i * 37 + a) * 10 + k % 10] : (real)0;
  }
  const int sd = 8 + L.chargeDim;
  if (block(L.offElCh, 37 * sd)) {
    int k = c - L.offElCh, a = k / sd, q = k % sd;
    size_t slot = (size_t)i * 37 + a;
    if (am[slot] == 0) return 0;
    if (q < 8) { *table = 1; *row = el[slot]; *col = q; return elE[(size_t)el[slot] * 8 + q]; }
    int cc = min(10, max(0, ch[slot] + 4));
    *table = 2; *row = cc; *col = q - 8; return chE[(size_t)cc * L.chargeDim + q - 8];
  }
  return 0;
}
__global__ void encFeatsK(const real* coors, const real* am, const int* tt, const real* chain, const real* bonds, const int* el,
                          const int* ch, const real* pad, const real* ttE, const real* elE, const real* chE, EncLayout L, real* f, int n, int fdim) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * fdim) return;
  int i = t / fdim, c = t % fdim, tb, r, cl;
  f[t] = pad[i] == 0 ? (real)0 : encFeatOf(coors, am, tt, chain, bonds, el, ch, ttE, elE, chE, L, i, c, &tb, &r, &cl);
}
__global__ void encFeatsBwdK(const real* d, const real* coors, const real* am, const int* tt, const real* chain, const real* bonds,
                             const int* el, const int* ch, const real* pad, const real* ttE, const real* elE, const real* chE, EncLayout L,
                             real* gTt, real* gEl, real* gCh, int n, int fdim) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * fdim) return;
  int i = t / fdim, c = t % fdim, tb, r, cl;
  if (pad[i] == 0) return;
  encFeatOf(coors, am, tt, chain, bonds, el, ch, ttE, elE, chE, L, i, c, &tb, &r, &cl);
  if (tb == 0) atomicAdd(&gTt[(size_t)r * L.ttDim + cl], d[t]);
  else if (tb == 1) atomicAdd(&gEl[(size_t)r * 8 + cl], d[t]);
  else if (tb == 2) atomicAdd(&gCh[(size_t)r * L.chargeDim + cl], d[t]);
}

}  // namespace rfd4p
}  // namespace grad
