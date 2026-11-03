// MSA Pairformer's whole model for training: the MSA's embedding (a table row a token), the pair's initial value (the
// relative-position rows and the bond), the layers (msap.cuh), the two contact heads at their layers and the LM head on
// the query row. cuda/msap/src/msap.cu's msaEmbedK, pairInitK, contacts() and lmHead() are what the new pieces are held
// to; the layers are test_msap's.
#pragma once
#include "msap.cuh"

namespace grad {
namespace msap {

// m[r] = table[token[r]]; and back
__global__ void embedRowsK(const int* tok, const real* W, real* m, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) m[t] = W[(size_t)tok[t / C] * C + t % C];
}
__global__ void embedRowsBwdK(const int* tok, const real* d, real* dW, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) atomicAdd(&dW[(size_t)tok[t / C] * C + t % C], d[t]);
}
// z = W[32] + W[66 + d] + W[132] + W[138] (+ bond where |i - j| = 1), d the shifted index's clipped offset; and back
__global__ void pairInitK2(const int* shifted, const real* W, const real* bond, real* z, int L, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / L, j = ij % L;
  int d = min(max(shifted[i] - shifted[j] + 32, 0), 64);
  real v = W[32 * C + c] + W[(66 + d) * C + c] + W[132 * C + c] + W[138 * C + c];
  if (abs(i - j) == 1) v += bond[c];
  z[t] = v;
}
__global__ void pairInitBwdK(const int* shifted, const real* dz, real* dW, real* dbond, int L, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;     // by channel
  if (c >= C) return;
  real s = 0, sb = 0;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j) {
      real g = dz[((size_t)i * L + j) * C + c];
      int d = min(max(shifted[i] - shifted[j] + 32, 0), 64);
      dW[(size_t)(66 + d) * C + c] += g;
      s += g;
      if (abs(i - j) == 1) sb += g;
    }
  dW[32 * C + c] += s; dW[132 * C + c] += s; dW[138 * C + c] += s; dbond[c] += sb;
}
// out[i, j] = (sigmoid(l_ij) + sigmoid(l_ji)) / 2, and back into the logits
__global__ void contactSymK(const real* l, real* out, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L) return;
  int i = t / L, j = t % L;
  out[t] = (1 / (1 + exp(-l[t])) + 1 / (1 + exp(-l[(size_t)j * L + i]))) / 2;
}
__global__ void contactSymBwdK(const real* l, const real* d, real* dl, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L) return;
  int i = t / L, j = t % L;
  real s = 1 / (1 + exp(-l[t]));
  dl[t] = (d[t] + d[(size_t)j * L + i]) / 2 * s * (1 - s);
}
// the LM head's gate: out = silu(h[:C] + b[:C]) * (h[C:] + b[C:])
__global__ void lmGateK2(const real* h, const real* b, real* out, int rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)rows * C) return;
  size_t r = t / C; int c = t % C;
  real g = h[r * 2 * C + c] + b[c], x = h[r * 2 * C + C + c] + b[C + c];
  out[t] = g / (1 + exp(-g)) * x;
}
__global__ void lmGateBwdK(const real* h, const real* b, const real* d, real* dh, int rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)rows * C) return;
  size_t r = t / C; int c = t % C;
  real g = h[r * 2 * C + c] + b[c], x = h[r * 2 * C + C + c] + b[C + c], s = 1 / (1 + exp(-g));
  dh[r * 2 * C + c] = d[t] * x * s * (1 + g * (1 - s));
  dh[r * 2 * C + C + c] = d[t] * g * s;
}

struct ContactHead { LayerNorm ln; Linear w; };       // w [C, 1] with its bias
struct LmHead { LayerNorm init, pre; Linear dense, logits; Param denseBias; int C = 464, V = 26; };
struct Model {
  int Cm = 464, C = 256, V = 0, depth = 0, cbLayer = 0, cfLayer = 0;
  Param msaInit, relpos, bond;
  std::vector<Layer> layers;
  ContactHead cb, cf;
  LmHead lm;
};
inline void forEachParam(Model& m, int H, int D, int I, const std::function<double(int)>& lambdaInit,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string P = "f/msap/";
  f(P + "msaInit", m.msaInit, (size_t)m.V * m.Cm);
  f(P + "relpos", m.relpos, (size_t)139 * m.C);
  f(P + "bond", m.bond, m.C);
  m.layers.assign(m.depth, Layer{});
  for (int l = 0; l < m.depth; ++l) forEachParam(m.layers[l], P + "blocks/" + std::to_string(l), m.Cm, m.C, H, D, I, lambdaInit(l), f);
  for (auto* h : {&m.cb, &m.cf}) {
    const std::string Hn = P + (h == &m.cb ? "cbContact/" : "confindContact/");
    h->ln.C = m.C; f(Hn + "normScale", h->ln.g, m.C); f(Hn + "normOffset", h->ln.b, m.C);
    h->w.K = m.C; h->w.N = 1; h->w.hasBias = true; f(Hn + "weight", h->w.w, m.C); f(Hn + "bias", h->w.b, 1);
  }
  LmHead& l = m.lm; l.C = m.Cm;
  l.init.C = l.C; f(P + "lm/initNormScale", l.init.g, l.C); f(P + "lm/initNormOffset", l.init.b, l.C);
  l.dense.K = l.C; l.dense.N = 2 * l.C; l.dense.hasBias = false; f(P + "lm/dense", l.dense.w, (size_t)2 * l.C * l.C);
  f(P + "lm/denseBias", l.denseBias, 2 * l.C);
  l.pre.C = l.C; f(P + "lm/preLogitNormScale", l.pre.g, l.C); f(P + "lm/preLogitNormOffset", l.pre.b, l.C);
  l.logits.K = l.C; l.logits.N = l.V; l.logits.hasBias = true; f(P + "lm/logits", l.logits.w, (size_t)l.C * l.V); f(P + "lm/logitsBias", l.logits.b, l.V);
}
struct HeadSave { LnSave ln; real *xn, *l; };
inline HeadSave contactForward(ContactHead& h, const real* z, int L, real* out) {
  HeadSave s{};
  const size_t P = (size_t)L * L;
  s.xn = ARENA.get(P * h.ln.C); s.ln = lnForward(h.ln, z, s.xn, P);
  s.l = ARENA.get(P); h.w.forward(s.xn, s.l, P);
  contactSymK<<<blocks(P), 256, 0, STREAM>>>(s.l, out, L);
  return s;
}
inline void contactBackward(ContactHead& h, const HeadSave& s, const real* dout, int L, real* dz) {     // dz added
  Scratch scratch;
  const size_t P = (size_t)L * L;
  real* dl = ARENA.get(P); contactSymBwdK<<<blocks(P), 256, 0, STREAM>>>(s.l, dout, dl, L);
  real* dxn = ARENA.get(P * h.ln.C); h.w.backward(s.xn, dl, dxn, P);
  lnBackward(h.ln, s.ln, dxn, dz, P, true);
}
struct LmSave { LnSave l1, l2; real *x1, *h, *g, *x2; };
inline LmSave lmForward(LmHead& l, const real* mq, int L, real* logits) {
  LmSave s{};
  s.x1 = ARENA.get((size_t)L * l.C); s.l1 = lnForward(l.init, mq, s.x1, L);
  s.h = ARENA.get((size_t)L * 2 * l.C); l.dense.forward(s.x1, s.h, L);
  s.g = ARENA.get((size_t)L * l.C); lmGateK2<<<blocks((size_t)L * l.C), 256, 0, STREAM>>>(s.h, l.denseBias.w, s.g, L, l.C);
  s.x2 = ARENA.get((size_t)L * l.C); s.l2 = lnForward(l.pre, s.g, s.x2, L);
  l.logits.forward(s.x2, logits, L);
  return s;
}
inline void lmBackward(LmHead& l, const LmSave& s, const real* dlogits, int L, real* dmq) {     // dmq added
  Scratch scratch;
  real* dx2 = ARENA.get((size_t)L * l.C); l.logits.backward(s.x2, dlogits, dx2, L);
  real* dg = ARENA.get((size_t)L * l.C); lnBackward(l.pre, s.l2, dx2, dg, L, false);
  real* dh = ARENA.get((size_t)L * 2 * l.C); lmGateBwdK<<<blocks((size_t)L * l.C), 256, 0, STREAM>>>(s.h, l.denseBias.w, dg, dh, L, l.C);
  colSum(l.denseBias.g, dh, L, 2 * l.C);
  real* dx1 = ARENA.get((size_t)L * l.C); l.dense.backward(s.x1, dh, dx1, L);
  lnBackward(l.init, s.l1, dx1, dmq, L, true);
}

struct Run { std::vector<real*> m, z; HeadSave cb, cf; LmSave lm; real *cbOut, *cfOut, *logits; };
// tokens [S, L], shifted [L] (the token index, 100 past each chain break); masks all ones (the release's form)
inline Run modelForward(Model& mo, const int* tok, const int* shifted, const real* msaMask, const real* pairMask, const real* maskT,
                        int S, int L) {
  Run r;
  const size_t R = (size_t)S * L, P = (size_t)L * L;
  r.m.assign(mo.depth + 1, nullptr); r.z.assign(mo.depth + 1, nullptr);
  for (int l = 0; l <= mo.depth; ++l) { r.m[l] = ARENA.get(R * mo.Cm); r.z[l] = ARENA.get(P * mo.C); }
  embedRowsK<<<blocks(R * mo.Cm), 256, 0, STREAM>>>(tok, mo.msaInit.w, r.m[0], R, mo.Cm);
  pairInitK2<<<blocks(P * mo.C), 256, 0, STREAM>>>(shifted, mo.relpos.w, mo.bond.w, r.z[0], L, mo.C);
  r.cbOut = ARENA.get(P); r.cfOut = ARENA.get(P); r.logits = ARENA.get((size_t)L * mo.lm.V);
  for (int l = 0; l < mo.depth; ++l) {
    { Scratch sc; layerForward(mo.layers[l], r.m[l], r.z[l], msaMask, pairMask, maskT, r.m[l + 1], r.z[l + 1], S, L, mo.C); }
    if (l == mo.cbLayer) r.cb = contactForward(mo.cb, r.z[l + 1], L, r.cbOut);
    if (l == mo.cfLayer) r.cf = contactForward(mo.cf, r.z[l + 1], L, r.cfOut);
  }
  r.lm = lmForward(mo.lm, r.m[mo.depth], L, r.logits);       // (the query row: the first L rows)
  return r;
}
// the heads' cotangents (any null) -> every parameter
inline void modelBackward(Model& mo, const Run& r, const int* tok, const int* shifted, const real* msaMask, const real* pairMask,
                          const real* maskT, int S, int L, const real* dcb, const real* dcf, const real* dlogits) {
  Scratch scratch;
  const size_t R = (size_t)S * L, P = (size_t)L * L;
  real* dm = ARENA.zeros(R * mo.Cm);
  real* dz = ARENA.zeros(P * mo.C);
  if (dlogits) lmBackward(mo.lm, r.lm, dlogits, L, dm);
  for (int l = mo.depth; l-- > 0;) {
    if (l == mo.cfLayer && dcf) contactBackward(mo.cf, r.cf, dcf, L, dz);
    if (l == mo.cbLayer && dcb) contactBackward(mo.cb, r.cb, dcb, L, dz);
    Scratch sc;
    LayerSave s = layerForward(mo.layers[l], r.m[l], r.z[l], msaMask, pairMask, maskT, ARENA.get(R * mo.Cm), ARENA.get(P * mo.C), S, L, mo.C);
    real *dmIn = ARENA.get(R * mo.Cm), *dzIn = ARENA.get(P * mo.C);
    layerBackward(mo.layers[l], s, msaMask, pairMask, maskT, dm, dz, dmIn, dzIn, S, L, mo.C);
    GK(cudaMemcpyAsync(dm, dmIn, R * mo.Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(dz, dzIn, P * mo.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  embedRowsBwdK<<<blocks(R * mo.Cm), 256, 0, STREAM>>>(tok, dm, mo.msaInit.g, R, mo.Cm);
  pairInitBwdK<<<blocks(mo.C, 64), 64, 0, STREAM>>>(shifted, dz, mo.relpos.g, mo.bond.g, L, mo.C);
}

}  // namespace msap
}  // namespace grad
