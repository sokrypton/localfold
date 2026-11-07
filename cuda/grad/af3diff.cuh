// The AF3 lineage's diffusion transformer for training (cpu/af3/diffusion/diffusion.js's diffusionTransformer;
// cuda/af3/src/diffusion.cuh's diffusionTransformer<float> is what the forward is held to):
//
//   per block:  x = AdaLN(act, cond); attention over tokens (q has a bias; an optional q/k LayerNorm) biased by the
//               pair logits and the token mask, gated, projected and AdaLN-zero gated on cond;
//               act += that; then a SwiGLU transition (Boltz-2: times an up-gate) on AdaLN(act or the block input),
//               AdaLN-zero gated, added. rf3 (noResidual): the transition reads the block's INPUT.
//   pair logits: LN(pairCond; scale only) once, then a projection per super-block to its blocks' heads
//
// AdaLN, AlphaFold 3's: sigmoid(LN(cond; scale) Ws + bs) * LN(x) + LN(cond; scale) Wb  (eps 1e-5)
//        chai-1's:      (cond Ws + 1) * LN(x; eps 0.1) + cond Wb, and its gating query zero, undone by 2x the output
#pragma once
#include "af3.cuh"

namespace grad {
namespace af3 {

struct AdaLn {
  LayerNorm xNorm;         // no affine
  LayerNorm cNorm;         // scale only (AlphaFold 3); unused for chai
  Linear scale, shift;     // [Cc, C]: scale with a bias (none for chai), shift without
  int C = 0, Cc = 0;
  bool chai = false;
};
struct AdaSave { LnSave lx, lc; real *xn, *cn, *sig; size_t rows; };
__global__ void adaCombineK(const real* xn, const real* sig, const real* sh, real* out, size_t n, real plus, bool raw) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  real s = raw ? sig[t] + plus : 1 / (1 + exp(-sig[t]));
  out[t] = s * xn[t] + sh[t];
}
inline AdaSave adaForward(const AdaLn& a, const real* x, const real* cond, real* out, size_t rows) {
  AdaSave s{}; s.rows = rows;
  s.xn = ARENA.get(rows * a.C);
  s.lx = lnForward(a.xNorm, x, s.xn, rows);
  if (a.chai) s.cn = const_cast<real*>(cond);
  else { s.cn = ARENA.get(rows * a.Cc); s.lc = lnForward(a.cNorm, cond, s.cn, rows); }
  s.sig = ARENA.get(rows * a.C);            // the scale's pre-activation
  a.scale.forward(s.cn, s.sig, rows);
  Scratch scratch;
  real* sh = ARENA.get(rows * a.C);
  a.shift.forward(s.cn, sh, rows);
  adaCombineK<<<blocks(rows * a.C), 256, 0, STREAM>>>(s.xn, s.sig, sh, out, rows * a.C, 1, a.chai);
  return s;
}
// dsig = dout xn act'(pre), dshift = dout, dxn = dout act(pre)
__global__ void adaBwdK(const real* dout, const real* xn, const real* pre, real* dpre, real* dxn, size_t n, real plus, bool raw) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  real s, ds;
  if (raw) { s = pre[t] + plus; ds = 1; } else { s = 1 / (1 + exp(-pre[t])); ds = s * (1 - s); }
  dpre[t] = dout[t] * xn[t] * ds;
  dxn[t] = dout[t] * s;
}
// dout -> dx (added), dcond (added)
inline void adaBackward(AdaLn& a, const AdaSave& s, const real* dout, real* dx, real* dcond) {
  Scratch scratch;
  size_t rows = s.rows;
  real *dpre = ARENA.get(rows * a.C), *dxn = ARENA.get(rows * a.C), *dcn = ARENA.get(rows * a.Cc);
  adaBwdK<<<blocks(rows * a.C), 256, 0, STREAM>>>(dout, s.xn, s.sig, dpre, dxn, rows * a.C, 1, a.chai);
  a.scale.backward(s.cn, dpre, dcn, rows);
  a.shift.backward(s.cn, dout, dcn, rows, true);
  lnBackward(a.xNorm, s.lx, dxn, dx, rows, true);
  if (a.chai) add(dcond, dcn, rows * a.Cc);
  else lnBackward(a.cNorm, s.lc, dcn, dcond, rows, true);
}

// AdaLN-zero: out (+)= (y W2) * sigmoid(cond Wz + bz) * mult
struct AdaZero {
  Linear proj, gate;       // proj [In, C] (Transition2 / ffwTransition2), gate [Cc, C] with its bias
  real mult = 1;
  bool identity = false;   // chai-1's atom blocks: no Transition2 (In == C), the input as it is
};
struct AzSave { const real* y; const real* cond; real *p, *g; size_t rows; size_t per = 0; };   // (per: azForwardGate's)
inline AzSave azForward(const AdaZero& z, const real* y, const real* cond, real* out, size_t rows) {
  AzSave s{y, cond, ARENA.get(rows * z.proj.N), ARENA.get(rows * z.proj.N), rows};
  if (z.identity) GK(cudaMemcpyAsync(s.p, y, rows * z.proj.N * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  else z.proj.forward(y, s.p, rows);
  if (z.mult != 1) scaleK<<<blocks(rows * z.proj.N), 256, 0, STREAM>>>(s.p, z.mult, rows * z.proj.N);
  z.gate.forward(cond, s.g, rows);
  sigmoidK<<<blocks(rows * z.proj.N), 256, 0, STREAM>>>(s.g, rows * z.proj.N);
  Scratch scratch;
  real* t = ARENA.get(rows * z.proj.N);
  mulK<<<blocks(rows * z.proj.N), 256, 0, STREAM>>>(t, s.p, s.g, rows * z.proj.N);
  add(out, t, rows * z.proj.N);
  return s;
}
// dout -> dy (written), dcond (added)
inline void azBackward(AdaZero& z, const AzSave& s, const real* dout, real* dy, real* dcond) {
  Scratch scratch;
  size_t rows = s.rows; const int C = z.proj.N;
  real *dp = ARENA.get(rows * C), *dg = ARENA.get(rows * C);
  gateBwdK<<<blocks(rows * C), 256, 0, STREAM>>>(dout, s.p, s.g, dp, dg, rows * C);
  z.gate.backward(s.cond, dg, dcond, rows, true);
  if (z.mult != 1) scaleK<<<blocks(rows * C), 256, 0, STREAM>>>(dp, z.mult, rows * C);
  if (z.identity) GK(cudaMemcpyAsync(dy, dp, rows * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  else z.proj.backward(s.y, dp, dy, rows);
}

// ---- the conditioning half of AdaLN and AdaLN-zero, once for every sample that shares the conditioning (a diffusion
// batch's atom stacks: their queries' and keys' conditioning is the encoder's, the same for all S): made once, and the
// samples' gradients into it summed and taken back once (adaPreBackward, azPreBackward) - linear in them, so the
// same gradient as each sample's own
struct AdaPre { LnSave lc; real *cn = nullptr, *sig = nullptr, *sh = nullptr; size_t rows = 0; };
inline AdaPre adaPre(const AdaLn& a, const real* cond, size_t rows) {
  AdaPre p; p.rows = rows;
  if (a.chai) p.cn = const_cast<real*>(cond);
  else { p.cn = ARENA.get(rows * a.Cc); p.lc = lnForward(a.cNorm, cond, p.cn, rows); }
  p.sig = ARENA.get(rows * a.C); a.scale.forward(p.cn, p.sig, rows);
  p.sh = ARENA.get(rows * a.C); a.shift.forward(p.cn, p.sh, rows);
  return p;
}
// (S: S samples' rows stacked over the one conditioning p was made from - x [S p.rows, C], its terms read modulo)
__global__ void adaCombinePK(const real* xn, const real* sig, const real* sh, real* out, size_t n, size_t per, real plus, bool raw) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const size_t u = t % per;
  real s = raw ? sig[u] + plus : 1 / (1 + exp(-sig[u]));
  out[t] = s * xn[t] + sh[u];
}
inline AdaSave adaForward(const AdaLn& a, const real* x, const AdaPre& p, real* out, int S = 1) {
  AdaSave s{}; s.rows = p.rows * S;
  s.xn = ARENA.get(s.rows * a.C);
  s.lx = lnForward(a.xNorm, x, s.xn, s.rows);
  s.cn = p.cn; s.sig = p.sig; s.lc = p.lc;
  adaCombinePK<<<blocks(s.rows * a.C), 256, 0, STREAM>>>(s.xn, p.sig, p.sh, out, s.rows * a.C, p.rows * a.C, 1, a.chai);
  return s;
}
// adaBwdK with the scale's pre-activation gradient and the shift's (dout) added into dsig and dsh - over S samples
// stacked on one conditioning [per], each element's S terms summed in sample order (a thread an element)
__global__ void adaBwdAccK(const real* dout, const real* xn, const real* pre, real* dsig, real* dsh, real* dxn, size_t per, int S,
                           real plus, bool raw) {
  size_t u = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= per) return;
  real s, ds;
  if (raw) { s = pre[u] + plus; ds = 1; } else { s = 1 / (1 + exp(-pre[u])); ds = s * (1 - s); }
  real gs = dsig[u], gh = dsh[u];
  for (int k = 0; k < S; ++k) {
    const size_t t = (size_t)k * per + u;
    gs += dout[t] * xn[t] * ds;
    gh += dout[t];
    dxn[t] = dout[t] * s;
  }
  dsig[u] = gs; dsh[u] = gh;
}
// dout -> dx (added); the conditioning's gradients into dsig, dsh (added; S: as adaForward's)
inline void adaBackward(AdaLn& a, const AdaSave& s, const real* dout, real* dx, real* dsig, real* dsh, int S = 1) {
  Scratch scratch;
  real* dxn = ARENA.get(s.rows * a.C);
  const size_t per = s.rows / S * a.C;
  adaBwdAccK<<<blocks(per), 256, 0, STREAM>>>(dout, s.xn, s.sig, dsig, dsh, dxn, per, S, 1, a.chai);
  lnBackward(a.xNorm, s.lx, dxn, dx, s.rows, true);
}
inline void adaPreBackward(AdaLn& a, const AdaPre& p, const real* dsig, const real* dsh, real* dcond) {
  Scratch scratch;
  real* dcn = ARENA.get(p.rows * a.Cc);
  a.scale.backward(p.cn, dsig, dcn, p.rows);
  a.shift.backward(p.cn, dsh, dcn, p.rows, true);
  if (a.chai) add(dcond, dcn, p.rows * a.Cc);
  else lnBackward(a.cNorm, p.lc, dcn, dcond, p.rows, true);
}
// the zero gate made once: sigmoid(cond Wz + bz)
inline real* azPre(const AdaZero& z, const real* cond, size_t rows) {
  real* g = ARENA.get(rows * z.proj.N);
  z.gate.forward(cond, g, rows);
  sigmoidK<<<blocks(rows * z.proj.N), 256, 0, STREAM>>>(g, rows * z.proj.N);
  return g;
}
__global__ void mulPK(real* y, const real* a, const real* b, size_t n, size_t per) {   // y = a b[t % per]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = a[t] * b[t % per];
}
// (S: S samples' rows stacked over the one gate - y and out [S rows / S ...])
inline AzSave azForwardGate(const AdaZero& z, const real* y, real* g, real* out, size_t rows, int S = 1) {
  AzSave s{y, nullptr, ARENA.get(rows * z.proj.N), g, rows, rows / S * z.proj.N};
  if (z.identity) GK(cudaMemcpyAsync(s.p, y, rows * z.proj.N * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  else z.proj.forward(y, s.p, rows);
  if (z.mult != 1) scaleK<<<blocks(rows * z.proj.N), 256, 0, STREAM>>>(s.p, z.mult, rows * z.proj.N);
  Scratch scratch;
  real* t = ARENA.get(rows * z.proj.N);
  mulPK<<<blocks(rows * z.proj.N), 256, 0, STREAM>>>(t, s.p, s.g, rows * z.proj.N, s.per);
  add(out, t, rows * z.proj.N);
  return s;
}
// (over S samples stacked on one gate [per]: each element's S terms summed in sample order)
__global__ void gateBwdAccK(const real* dy, const real* a, const real* sig, real* da, real* dpre, size_t per, int S) {
  size_t u = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= per) return;
  const real s = sig[u];
  real g = dpre[u];
  for (int k = 0; k < S; ++k) {
    const size_t t = (size_t)k * per + u;
    da[t] = dy[t] * s;
    g += dy[t] * a[t] * s * (1.f - s);
  }
  dpre[u] = g;
}
// dout -> dy (written); the gate's pre-activation gradient into dg (added)
inline void azBackwardGate(AdaZero& z, const AzSave& s, const real* dout, real* dy, real* dg) {
  Scratch scratch;
  size_t rows = s.rows; const int C = z.proj.N;
  real* dp = ARENA.get(rows * C);
  gateBwdAccK<<<blocks(s.per), 256, 0, STREAM>>>(dout, s.p, s.g, dp, dg, s.per, (int)(rows * C / s.per));
  if (z.mult != 1) scaleK<<<blocks(rows * C), 256, 0, STREAM>>>(dp, z.mult, rows * C);
  if (z.identity) GK(cudaMemcpyAsync(dy, dp, rows * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  else z.proj.backward(s.y, dp, dy, rows);
}
inline void azPreBackward(AdaZero& z, const real* cond, const real* dg, size_t rows, real* dcond) {
  z.gate.backward(cond, dg, dcond, rows, true);
}

struct DiffBlock {
  AdaLn ln1, ln2;
  Linear q, k, v, gate;            // [C, HD]; q with qBias
  LayerNorm qn, kn;                // the optional q/k LayerNorm over HD
  bool kqNorm = false;
  AdaZero z1, z2;                  // Transition2 + AdaptiveZeroCond; ffwTransition2 + ffwAdaptiveZeroCond
  Linear t1, up;                   // ffwTransition1 [C, 2I]; ffwAToB [C, I] (Boltz-2)
  bool hasUp = false;
  int C = 0, H = 0, D = 0, I = 0;
};
struct DiffTransformer {
  std::vector<DiffBlock> blocks;
  std::vector<Linear> superProj;   // [Cz, perSuper H]
  LayerNorm pairNorm;              // scale only
  int C = 0, Cc = 0, Cz = 0, H = 0, D = 0, perSuper = 4, factor = 2;
  bool noResidual = false, chai = false;
  Params params;
};

inline void forEachParam(DiffTransformer& t, const std::string& Tn, int nSuper, bool upGate, bool kqNorm,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const int C = t.C, Cc = t.Cc, HD = t.H * t.D, I = C * t.factor;
  t.pairNorm.C = t.Cz; f(Tn + ".pairInputLayerNormScale", t.pairNorm.g, t.Cz);
  t.superProj.assign(nSuper, Linear{});
  t.blocks.assign((size_t)nSuper * t.perSuper, DiffBlock{});
  auto lin = [&](const std::string& n, Linear& l, int K, int N, const std::string& bias = "") {
    l.K = K; l.N = N; l.hasBias = !bias.empty(); l.w.fanIn = K; f(n, l.w, (size_t)K * N); if (!bias.empty()) f(bias, l.b, N);
  };
  auto ada = [&](const std::string& B, AdaLn& a) {
    a.C = C; a.Cc = Cc; a.chai = t.chai; a.xNorm.C = C;
    if (t.chai) a.xNorm.eps = 0.1f;
    if (!t.chai) { a.cNorm.C = Cc; f(B + "SingleCondLayerNormScale", a.cNorm.g, Cc); }
    lin(B + "SingleCondScaleWeights", a.scale, Cc, C, t.chai ? "" : B + "SingleCondScaleBias");
    lin(B + "SingleCondBias", a.shift, Cc, C);
  };
  for (int sb = 0; sb < nSuper; ++sb) {
    const std::string S = Tn + ".superBlocks." + std::to_string(sb);
    lin(S + ".pairLogitsProjection", t.superProj[sb], t.Cz, t.perSuper * t.H);
    for (int k = 0; k < t.perSuper; ++k) {
      DiffBlock& b = t.blocks[(size_t)sb * t.perSuper + k];
      const std::string B = S + ".blocks." + std::to_string(k);
      b.C = C; b.H = t.H; b.D = t.D; b.I = I; b.kqNorm = kqNorm; b.hasUp = upGate;
      ada(B + ".", b.ln1);
      lin(B + ".qProjection", b.q, C, HD, B + ".qBias");
      lin(B + ".kProjection", b.k, C, HD);
      lin(B + ".vProjection", b.v, C, HD);
      lin(B + ".gatingQuery", b.gate, C, HD);
      if (kqNorm) {
        b.qn.C = HD; f(B + ".queryLayerNormScale", b.qn.g, HD); f(B + ".queryLayerNormOffset", b.qn.b, HD);
        b.kn.C = HD; f(B + ".keyLayerNormScale", b.kn.g, HD); f(B + ".keyLayerNormOffset", b.kn.b, HD);
      }
      lin(B + ".Transition2", b.z1.proj, HD, C);
      lin(B + ".AdaptiveZeroCondWeights", b.z1.gate, Cc, C, B + ".AdaptiveZeroCondBias");
      b.z1.mult = t.chai ? 2 : 1;
      ada(B + ".ffw", b.ln2);
      lin(B + ".ffwTransition1", b.t1, C, 2 * I);
      if (upGate) lin(B + ".ffwAToB", b.up, C, I);
      lin(B + ".ffwTransition2", b.z2.proj, I, C);
      lin(B + ".ffwAdaptiveZeroCondWeights", b.z2.gate, Cc, C, B + ".ffwAdaptiveZeroCondBias");
    }
  }
}

struct DiffBlockSave {
  AdaSave a1, a2;
  LnSave lq, lk;
  real *qkvg, *qraw, *kraw, *o, *og, *x1, *h, *u, *g;
  AzSave z1, z2;
  real* lse = nullptr;   // the attention's logit max and log-sum, kept for its backward
  // (kept across the transformer - DiffRun::keep - its forward's temporaries in the run's two scratch rows instead of
  // the arena, and og and g not kept but rebuilt by the backward from o and the gate, h and u: a sixth of a block's
  // saves, elementwise to redo)
  bool rebuild = false;
};
// (S samples batched: queries [S n] rows, the key mask [S n], the pair logits one set every sample reads)
inline Sdpa tokenSdpa(const DiffBlock& b, int n, const real* mask, const real* pl, real* dpl = nullptr, int S = 1) {
  Sdpa a;
  a.B = S; a.H = b.H; a.Q = n; a.K = n; a.D = b.D; a.scale = 1 / sqrt((real)b.D);
  a.keyMask = mask;
  a.bias[0] = Bias{pl, 0, (long long)n * n, n, dpl};
  return a;
}
// the gated SwiGLU (times the up-gate): wide [rows, 2I], up [rows, I] -> g [rows, I]
__global__ void swigluUpK(const real* wide, const real* up, real* g, size_t rows, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int i = t % I;
  real a = wide[r * 2 * I + i], b = wide[r * 2 * I + I + i];
  g[t] = a / (1 + exp(-a)) * b * (up ? up[t] : (real)1);
}
__global__ void swigluUpBwdK(const real* wide, const real* up, const real* dg, real* dwide, real* dup, size_t rows, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int i = t % I;
  real a = wide[r * 2 * I + i], b = wide[r * 2 * I + I + i], s = 1 / (1 + exp(-a)), u = up ? up[t] : (real)1;
  real d = dg[t];
  dwide[r * 2 * I + i] = d * b * u * (s + a * s * (1 - s));
  dwide[r * 2 * I + I + i] = d * a * s * u;
  if (dup) dup[t] = d * a * s * b;
}

// act [n, C] -> out [n, C]; S samples batched: act, cond and mask [S n, ...], one set of pair logits
// (tmpC [n C], tmpW [n max(HD, I)]: the scratch rows a kept forward's temporaries use - see DiffBlockSave::rebuild)
inline DiffBlockSave diffBlockForward(const DiffTransformer& t, const DiffBlock& b, const real* act, const real* cond,
                                      const real* mask, const real* pl, real* out, int nTok, int S = 1,
                                      real* tmpC = nullptr, real* tmpW = nullptr) {
  const int C = b.C, HD = b.H * b.D, I = b.I;
  const int n = nTok * S;     // (the rows every row-wise operation runs over)
  DiffBlockSave s{};
  s.rebuild = tmpC && tmpW;
  real* x = s.rebuild ? tmpC : ARENA.get((size_t)n * C);
  s.a1 = adaForward(b.ln1, act, cond, x, n);
  s.qkvg = ARENA.get((size_t)n * 4 * HD);
  real *q = s.qkvg, *k = q + (size_t)n * HD, *v = k + (size_t)n * HD, *g = v + (size_t)n * HD;
  if (b.kqNorm) {
    s.qraw = ARENA.get((size_t)n * HD); s.kraw = ARENA.get((size_t)n * HD);
    b.q.forward(x, s.qraw, n); b.k.forward(x, s.kraw, n);
    s.lq = lnForward(b.qn, s.qraw, q, n); s.lk = lnForward(b.kn, s.kraw, k, n);
  } else { b.q.forward(x, q, n); b.k.forward(x, k, n); }
  b.v.forward(x, v, n); b.gate.forward(x, g, n);
  sigmoidK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(g, (size_t)n * HD);
  s.o = ARENA.get((size_t)n * HD);
  { const Sdpa sd_ = tokenSdpa(b, nTok, mask, pl, nullptr, S); s.lse = ARENA.get((size_t)2 * sd_.B * sd_.H * sd_.Q); sdpaForward(sd_, q, HD, k, HD, v, HD, s.o, HD, s.lse); }
  s.og = s.rebuild ? tmpW : ARENA.get((size_t)n * HD);
  mulK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(s.og, s.o, g, (size_t)n * HD);
  s.x1 = ARENA.get((size_t)n * C);
  GK(cudaMemcpyAsync(s.x1, act, (size_t)n * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.z1 = azForward(b.z1, s.og, cond, s.x1, n);
  real* x2 = s.rebuild ? tmpC : ARENA.get((size_t)n * C);
  s.a2 = adaForward(b.ln2, t.noResidual ? act : s.x1, cond, x2, n);
  s.h = ARENA.get((size_t)n * 2 * I);
  b.t1.forward(x2, s.h, n);
  if (b.hasUp) { s.u = ARENA.get((size_t)n * I); b.up.forward(x2, s.u, n); }
  s.g = s.rebuild ? tmpW : ARENA.get((size_t)n * I);
  swigluUpK<<<blocks((size_t)n * I), 256, 0, STREAM>>>(s.h, b.hasUp ? s.u : nullptr, s.g, n, I);
  GK(cudaMemcpyAsync(out, s.x1, (size_t)n * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.z2 = azForward(b.z2, s.g, cond, out, n);
  return s;
}
// dout -> dact (written), dcond (added), dpl (added)
inline void diffBlockBackward(const DiffTransformer& t, DiffBlock& b, const DiffBlockSave& s, const real* act,
                              const real* cond, const real* mask, const real* pl, const real* dout, real* dact,
                              real* dcond, real* dpl, int nTok, int S = 1) {
  Scratch scratch;
  const int C = b.C, HD = b.H * b.D, I = b.I;
  const int n = nTok * S;
  real* dx1 = ARENA.get((size_t)n * C);
  GK(cudaMemcpyAsync(dx1, dout, (size_t)n * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  real* dact0 = ARENA.zeros((size_t)n * C);       // rf3: the transition's route to the block input
  // the transition
  real* dg = ARENA.get((size_t)n * I);
  if (s.rebuild) {   // (g rebuilt: the transition's output projection reads it)
    real* g2 = ARENA.get((size_t)n * I);
    swigluUpK<<<blocks((size_t)n * I), 256, 0, STREAM>>>(s.h, b.hasUp ? s.u : nullptr, g2, n, I);
    AzSave z2 = s.z2; z2.y = g2;
    azBackward(b.z2, z2, dout, dg, dcond);
  } else azBackward(b.z2, s.z2, dout, dg, dcond);
  real *dh = ARENA.get((size_t)n * 2 * I), *du = b.hasUp ? ARENA.get((size_t)n * I) : nullptr;
  swigluUpBwdK<<<blocks((size_t)n * I), 256, 0, STREAM>>>(s.h, b.hasUp ? s.u : nullptr, dg, dh, du, n, I);
  real* dx2 = ARENA.get((size_t)n * C);
  // (ln2's xn is x2's normalised input; the linear inputs are x2 = ada(ln2) - rebuilt from the save)
  real* x2 = ARENA.get((size_t)n * C);
  {
    real* sh = ARENA.get((size_t)n * C);
    b.ln2.shift.forward(s.a2.cn, sh, n);
    adaCombineK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(s.a2.xn, s.a2.sig, sh, x2, (size_t)n * C, 1, b.ln2.chai);
  }
  b.t1.backward(x2, dh, dx2, n);
  if (b.hasUp) b.up.backward(x2, du, dx2, n, true);
  adaBackward(b.ln2, s.a2, dx2, t.noResidual ? dact0 : dx1, dcond);
  // the attention
  real* dog = ARENA.get((size_t)n * HD);
  real *q = s.qkvg, *k = q + (size_t)n * HD, *v = k + (size_t)n * HD, *g = v + (size_t)n * HD;
  if (s.rebuild) {   // (og = o g rebuilt: the attention's output projection reads it)
    real* og = ARENA.get((size_t)n * HD);
    mulK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(og, s.o, g, (size_t)n * HD);
    AzSave z1 = s.z1; z1.y = og;
    azBackward(b.z1, z1, dx1, dog, dcond);
  } else azBackward(b.z1, s.z1, dx1, dog, dcond);
  real* dqkvg = ARENA.get((size_t)n * 4 * HD);
  real *dq = dqkvg, *dk = dq + (size_t)n * HD, *dv = dk + (size_t)n * HD, *dgt = dv + (size_t)n * HD;
  real* dO = ARENA.get((size_t)n * HD);
  gateBwdK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(dog, s.o, g, dO, dgt, (size_t)n * HD);
  sdpaBackward(tokenSdpa(b, nTok, mask, pl, dpl, S), q, HD, k, HD, v, HD, dO, HD, dq, HD, dk, HD, dv, HD, false, false, false, s.o, HD, s.lse);
  real* x = ARENA.get((size_t)n * C);
  {
    real* sh = ARENA.get((size_t)n * C);
    b.ln1.shift.forward(s.a1.cn, sh, n);
    adaCombineK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(s.a1.xn, s.a1.sig, sh, x, (size_t)n * C, 1, b.ln1.chai);
  }
  real* dx = ARENA.get((size_t)n * C);
  if (b.kqNorm) {
    real *dqr = ARENA.get((size_t)n * HD), *dkr = ARENA.get((size_t)n * HD);
    lnBackward(b.qn, s.lq, dq, dqr, n, false);
    lnBackward(b.kn, s.lk, dk, dkr, n, false);
    dq = dqr; dk = dkr;
  }
  b.q.backward(x, dq, dx, n);
  b.k.backward(x, dk, dx, n, true);
  b.v.backward(x, dv, dx, n, true);
  b.gate.backward(x, dgt, dx, n, true);
  add(dx1, dact0, (size_t)n * C);
  adaBackward(b.ln1, s.a1, dx, dx1, dcond);
  GK(cudaMemcpyAsync(dact, dx1, (size_t)n * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

// [nn, perSuper H] column block `inner` -> [H, n, n]  (back: accumulate into the flat's gradient)
__global__ void superBiasK(real* flat, real* pl, int n, int H, int perSuper, int inner, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t nn = (size_t)n * n;
  if (t >= nn * H) return;
  int h = t / nn; size_t ij = t % nn;
  size_t src = ij * perSuper * H + (size_t)inner * H + h;
  if (back) flat[src] += pl[t]; else pl[t] = flat[src];
}

// The whole transformer, every block's input kept and each replayed before its backward.
// (grow-only: a size change reallocated it - and every backward allocated and freed its three buffers behind a stream
// sync - which the device waited out each time; the caller gives it back with releaseRun. Stream-ordered: a training
// step grows it three times - the rollout's one sample, then each wider batch - and cudaFree's implicit device sync
// left the device idle 10-50 ms each time)
struct DiffRun {
  std::vector<real*> acts;         // [blocks + 1]
  real *pn = nullptr, *pl = nullptr;   // LN'd pair [nn, Cz]; every block's logits [blocks][H, n, n]
  real *dx = nullptr, *tmp = nullptr, *dpl = nullptr;   // the backward's
  LnSave pln;
  int n = 0, S = 1;     // tokens, and samples batched
  size_t capRows = 0, capNn = 0;
  // keep: every block's saves kept from the forward (in the caller's arena scope) for the backward, which then does
  // not replay the blocks - where the device has the room (train-of3's --diffusion-saves). keptBytes: what the last
  // kept forward held
  bool keep = false;
  std::vector<DiffBlockSave> saves;
  size_t keptBytes = 0;
  // timed: two events recorded around a forward's blocks - what a backward without keep replays (train-of3's
  // --sample-batch=auto prices keeping against it)
  cudaEvent_t* timed = nullptr;
};
inline void freeRun(DiffRun& r) {
  for (real* p : r.acts) GK(cudaFreeAsync(p, STREAM));
  for (real* p : {r.pn, r.pl, r.dx, r.tmp, r.dpl}) if (p) GK(cudaFreeAsync(p, STREAM));
  const bool keep = r.keep;   // (the caller's choices outlive a reallocation)
  cudaEvent_t* timed = r.timed;
  r = DiffRun{};
  r.keep = keep; r.timed = timed;
}
// (and the pool's bytes back to the driver: the pool keeps what it is given, and the trunk's backward wants the room)
inline void releaseRun(DiffRun& r) {
  freeRun(r);
  GK(cudaStreamSynchronize(STREAM));
  int dev; GK(cudaGetDevice(&dev));
  cudaMemPool_t pool; GK(cudaDeviceGetDefaultMemPool(&pool, dev));
  GK(cudaMemPoolTrimTo(pool, 0));
}
// S samples batched: act, cond, mask and out [S n, ...] (each sample's own noise level in its cond rows), the pair
// conditioning one for all - its logits computed once and read by every sample
// (logitsKept: r's pair logits are already pairCond's - a sampler's steps after its first, one pair for every step)
inline void transformerForward(DiffTransformer& t, DiffRun& r, const real* act, const real* cond, const real* mask,
                               const real* pairCond, real* out, int n, int S = 1, bool logitsKept = false) {
  const Arena::Mark m0 = ARENA.mark();   // the caller's arena is kept: this scopes its own use
  const size_t nn = (size_t)n * n, rows = (size_t)n * S;
  const int nb = (int)t.blocks.size();
  if (r.n != n) logitsKept = false;
  if (r.acts.empty() || rows > r.capRows || nn > r.capNn) {
    logitsKept = false;
    const size_t capRows = std::max(rows, r.capRows), capNn = std::max(nn, r.capNn);
    freeRun(r);
    r.acts.assign(nb + 1, nullptr);
    for (auto& p : r.acts) GK(cudaMallocAsync(&p, capRows * t.C * sizeof(real), STREAM));
    GK(cudaMallocAsync(&r.pn, capNn * t.Cz * sizeof(real), STREAM));
    GK(cudaMallocAsync(&r.pl, (size_t)nb * t.H * capNn * sizeof(real), STREAM));
    r.capRows = capRows; r.capNn = capNn;
  }
  r.n = n; r.S = S;
  r.saves.clear();
  ARENA.rewind(m0);
  if (!logitsKept) {
    real* mean = ARENA.get(nn); real* rstd = ARENA.get(nn);
    lnFwdK<<<blocks(nn, 8), 256, 0, STREAM>>>(pairCond, t.pairNorm.g.w, nullptr, r.pn, mean, rstd, nn, t.Cz, t.pairNorm.eps);
    real* flat = ARENA.get(nn * t.perSuper * t.H);
    for (int b = 0; b < nb; ++b) {
      if (b % t.perSuper == 0) t.superProj[b / t.perSuper].forward(r.pn, flat, nn);
      superBiasK<<<blocks(nn * t.H), 256, 0, STREAM>>>(flat, r.pl + (size_t)b * t.H * nn, n, t.H, t.perSuper, b % t.perSuper, false);
    }
  }
  GK(cudaMemcpyAsync(r.acts[0], act, rows * t.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  ARENA.rewind(m0);
  const size_t used0 = ARENA.used;
  // (kept: the blocks' temporaries in two scratch rows shared by every block, not in each block's saves)
  real *tmpC = nullptr, *tmpW = nullptr;
  if (r.keep) {
    int wide = 0; for (const auto& bk : t.blocks) wide = std::max({wide, bk.H * bk.D, bk.I});
    tmpC = ARENA.get(rows * t.C); tmpW = ARENA.get(rows * wide);
  }
  if (r.timed) GK(cudaEventRecord(r.timed[0], STREAM));
  for (int b = 0; b < nb; ++b) {
    if (!r.keep) ARENA.rewind(m0);
    DiffBlockSave sv = diffBlockForward(t, t.blocks[b], r.acts[b], cond, mask, r.pl + (size_t)b * t.H * nn, r.acts[b + 1], n, S, tmpC, tmpW);
    if (r.keep) r.saves.push_back(sv);
  }
  if (r.timed) GK(cudaEventRecord(r.timed[1], STREAM));
  if (r.keep) r.keptBytes = ARENA.used - used0;
  else ARENA.rewind(m0);
  GK(cudaMemcpyAsync(out, r.acts[nb], rows * t.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}
// dout -> dact (written), dcond (added), dpairCond (added)
inline void transformerBackward(DiffTransformer& t, DiffRun& r, const real* cond, const real* mask,
                                const real* pairCond, const real* dout, real* dact, real* dcond, real* dpairCond) {
  const Arena::Mark m0 = ARENA.mark();
  const int n = r.n, S = r.S, nb = (int)t.blocks.size();
  const size_t nn = (size_t)n * n, rows = (size_t)n * S;
  if (!r.dx) {
    GK(cudaMallocAsync(&r.dx, r.capRows * t.C * sizeof(real), STREAM));
    GK(cudaMallocAsync(&r.tmp, r.capRows * t.C * sizeof(real), STREAM));
    GK(cudaMallocAsync(&r.dpl, (size_t)nb * t.H * r.capNn * sizeof(real), STREAM));
  }
  real *dx = r.dx, *dpl = r.dpl, *tmp = r.tmp;
  GK(cudaMemsetAsync(dpl, 0, (size_t)nb * t.H * nn * sizeof(real), STREAM));
  GK(cudaMemcpyAsync(dx, dout, rows * t.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  for (int b = nb; b-- > 0;) {
    ARENA.rewind(m0);
    const real* pl = r.pl + (size_t)b * t.H * nn;
    const DiffBlockSave s = (int)r.saves.size() == nb ? r.saves[b] : diffBlockForward(t, t.blocks[b], r.acts[b], cond, mask, pl, tmp, n, S);
    diffBlockBackward(t, t.blocks[b], s, r.acts[b], cond, mask, pl, dx, dx, dcond, dpl + (size_t)b * t.H * nn, n, S);
  }
  ARENA.rewind(m0);
  {
    real* dflat = ARENA.get(nn * t.perSuper * t.H);
    real* dpn = ARENA.zeros(nn * t.Cz);
    for (int sb = 0; sb * t.perSuper < nb; ++sb) {
      GK(cudaMemsetAsync(dflat, 0, nn * t.perSuper * t.H * sizeof(real), STREAM));
      for (int k = 0; k < t.perSuper && sb * t.perSuper + k < nb; ++k)
        superBiasK<<<blocks(nn * t.H), 256, 0, STREAM>>>(dflat, dpl + (size_t)(sb * t.perSuper + k) * t.H * nn, n, t.H, t.perSuper, k, true);
      t.superProj[sb].backward(r.pn, dflat, dpn, nn, true);
    }
    real *mean = ARENA.get(nn), *rstd = ARENA.get(nn), *y = ARENA.get(nn * t.Cz);
    lnFwdK<<<blocks(nn, 8), 256, 0, STREAM>>>(pairCond, t.pairNorm.g.w, nullptr, y, mean, rstd, nn, t.Cz, t.pairNorm.eps);
    LnSave ls{pairCond, mean, rstd};
    lnBackward(t.pairNorm, ls, dpn, dpairCond, nn, true);
  }
  ARENA.rewind(m0);
  GK(cudaMemcpyAsync(dact, dx, rows * t.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  r.saves.clear();
}


// ---------------------------------------------------------------- the atom transformer (cross-attention over windows)
// cpu/af3/diffusion/atom-encoder.js's crossAttentionBlock: queries [subsets * 32, C], keys gathered from the query
// layout (queriesToKeys), [subsets * 128, C]. xq = AdaLN_q(act, qCond); keys = gather(chained ? xq : act),
// xk = AdaLN_k(keys, gather(qCond)); q = xq Wq + bq, k, v = xk Wk, xk Wv; logits + the block's pair logits + a mask
// bias (AF3: 1e9 (qm - 1)(km - 1), the product - only a pair of two padded atoms; keyMasked: -1e9 ((1 - qm) + (1 - km)));
// gated, AdaLN-zero on qCond, added; then the conditioned SwiGLU (Boltz-2's up-gate) on AdaLN_ffw(x1, or the block's
// input under noResidual), AdaLN-zero, added.
struct Gather { const int* idx; const real* mask; int count; };
__global__ void gatherK(const int* idx, const real* m, const real* src, real* out, int count, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)count * C) return;
  size_t r = t / C; int c = t % C;
  out[t] = m[r] != 0 ? src[(size_t)idx[r] * C + c] : (real)0;
}
__global__ void scatterK(const int* idx, const real* m, const real* d, real* dsrc, int count, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)count * C) return;
  size_t r = t / C; int c = t % C;
  if (m[r] != 0) atomicAdd(&dsrc[(size_t)idx[r] * C + c], d[t]);
}
inline void gather(const Gather& g, const real* src, real* out, int C) {
  gatherK<<<blocks((size_t)g.count * C), 256, 0, STREAM>>>(g.idx, g.mask, src, out, g.count, C);
}
inline void scatter(const Gather& g, const real* d, real* dsrc, int C) {
  scatterK<<<blocks((size_t)g.count * C), 256, 0, STREAM>>>(g.idx, g.mask, d, dsrc, g.count, C);
}
// the attention's mask bias [subsets, Q, K]
__global__ void atomMaskBiasK(const real* qm, const real* km, real* out, int subsets, int Q, int K, bool keyMasked) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)subsets * Q * K) return;
  int k = t % K; size_t r = t / K; int q = r % Q; int s = (int)(r / Q);
  real a = qm[(size_t)s * Q + q], b = km[(size_t)s * K + k];
  out[t] = keyMasked ? (real)-1e9 * ((1 - a) + (1 - b)) : (real)1e9 * (a - 1) * (b - 1);
}

struct AtomBlock {
  AdaLn lnQ, lnK, lnF;
  Linear q, k, v, gate, t1, up;
  AdaZero z1, z2;
  bool hasUp = false, chained = false, noResidual = false, keyMasked = false, maskActPerBlock = false;
  bool kqNorm = false;            // rf3: q = LN(xq Wq + bq), k = LN(xk Wk) over H D, with scale and offset
  LayerNorm qn, kn;
  int C = 0, H = 0, D = 0, I = 0;
};
struct AtomCtx {               // what every block of a stack shares
  Gather toKeys;               // queriesToKeys
  const real *qMask, *kMask, *qCond, *kCond, *maskBias;
  int subsets = 0, Q = 32, K = 128;
  // share: the subsets of one sample where a diffusion batch's samples are stacked (subsets = S share; 0: one sample) -
  // the pair logits, the mask bias and the blocks' conditioning terms (StackPre) then one sample's, read by all
  int share = 0;
  int samples() const { return share ? subsets / share : 1; }
};
struct AtomSave { AdaSave aq, ak, af; LnSave lq, lk; real *act0, *keys, *xq, *xk, *qkvg, *qraw, *kraw, *o, *og, *x1, *h, *u, *g; AzSave z1, z2; real* lse = nullptr; };
inline Sdpa atomSdpa(const AtomBlock& b, const AtomCtx& c, const real* pl, real* dpl = nullptr) {
  Sdpa a;
  a.B = c.subsets; a.H = b.H; a.Q = c.Q; a.K = c.K; a.D = b.D; a.scale = 1 / sqrt((real)b.D);
  a.bias[0] = Bias{pl, (long long)b.H * c.Q * c.K, (long long)c.Q * c.K, c.K, dpl, c.share};
  a.bias[1] = Bias{c.maskBias, (long long)c.Q * c.K, 0, c.K, nullptr, c.share};
  return a;
}
// a block's conditioning terms, made once for a batch (stackPre), and their gradients summed over it (stackGrads)
struct BlockPre { AdaPre q, k, f; real *g1 = nullptr, *g2 = nullptr; };
struct BlockGrads { real *sigQ, *shQ, *sigK, *shK, *sigF, *shF, *g1, *g2; };
// (bp: the conditioning terms made already - they are then not made here)
inline AtomSave atomForward(const AtomBlock& b, const AtomCtx& c, const real* act, const real* pl, real* out, const BlockPre* bp = nullptr) {
  const int C = b.C, HD = b.H * b.D, I = b.I;
  const size_t Rq = (size_t)c.subsets * c.Q, Rk = (size_t)c.subsets * c.K;
  AtomSave s{};
  s.act0 = const_cast<real*>(act);
  if (b.maskActPerBlock) { s.act0 = cloneR(act, Rq * C); rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(s.act0, c.qMask, Rq, C); }
  const int S = c.samples();   // (S > 1: bp's terms one sample's)
  s.xq = ARENA.get(Rq * C); s.aq = bp ? adaForward(b.lnQ, s.act0, bp->q, s.xq, S) : adaForward(b.lnQ, s.act0, c.qCond, s.xq, Rq);
  s.keys = ARENA.get(Rk * C); gather(c.toKeys, b.chained ? s.xq : s.act0, s.keys, C);
  s.xk = ARENA.get(Rk * C); s.ak = bp ? adaForward(b.lnK, s.keys, bp->k, s.xk, S) : adaForward(b.lnK, s.keys, c.kCond, s.xk, Rk);
  s.qkvg = ARENA.get(Rq * 2 * HD + Rk * 2 * HD);
  real *q = s.qkvg, *g = q + Rq * HD, *k = g + Rq * HD, *v = k + Rk * HD;
  if (b.kqNorm) {
    s.qraw = ARENA.get(Rq * HD); s.kraw = ARENA.get(Rk * HD);
    b.q.forward(s.xq, s.qraw, Rq); b.k.forward(s.xk, s.kraw, Rk);
    s.lq = lnForward(b.qn, s.qraw, q, Rq); s.lk = lnForward(b.kn, s.kraw, k, Rk);
  } else { b.q.forward(s.xq, q, Rq); b.k.forward(s.xk, k, Rk); }
  b.gate.forward(s.xq, g, Rq); b.v.forward(s.xk, v, Rk);
  sigmoidK<<<blocks(Rq * HD), 256, 0, STREAM>>>(g, Rq * HD);
  s.o = ARENA.get(Rq * HD);
  { const Sdpa sd_ = atomSdpa(b, c, pl); s.lse = ARENA.get((size_t)2 * sd_.B * sd_.H * sd_.Q); sdpaForward(sd_, q, HD, k, HD, v, HD, s.o, HD, s.lse); }
  s.og = ARENA.get(Rq * HD);
  mulK<<<blocks(Rq * HD), 256, 0, STREAM>>>(s.og, s.o, g, Rq * HD);
  s.x1 = cloneR(s.act0, Rq * C);
  s.z1 = bp ? azForwardGate(b.z1, s.og, bp->g1, s.x1, Rq, S) : azForward(b.z1, s.og, c.qCond, s.x1, Rq);
  real* xf = ARENA.get(Rq * C);
  s.af = bp ? adaForward(b.lnF, b.noResidual ? s.act0 : s.x1, bp->f, xf, S) : adaForward(b.lnF, b.noResidual ? s.act0 : s.x1, c.qCond, xf, Rq);
  s.h = ARENA.get(Rq * 2 * I); b.t1.forward(xf, s.h, Rq);
  if (b.hasUp) { s.u = ARENA.get(Rq * I); b.up.forward(xf, s.u, Rq); }
  s.g = ARENA.get(Rq * I);
  swigluUpK<<<blocks(Rq * I), 256, 0, STREAM>>>(s.h, b.hasUp ? s.u : nullptr, s.g, Rq, I);
  GK(cudaMemcpyAsync(out, s.x1, Rq * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.z2 = bp ? azForwardGate(b.z2, s.g, bp->g2, out, Rq, S) : azForward(b.z2, s.g, c.qCond, out, Rq);
  return s;
}
// dout -> dact (written), dqCond, dkCond (added), dpl (added)
// (bp, bg: the conditioning terms made once, and their gradients added into bg instead of taken back here)
inline void atomBackward(AtomBlock& b, const AtomCtx& c, const AtomSave& s, const real* pl, const real* dout, real* dact,
                         real* dqCond, real* dkCond, real* dpl, const BlockPre* bp = nullptr, BlockGrads* bg = nullptr) {
  Scratch scratch;
  const int C = b.C, HD = b.H * b.D, I = b.I;
  const size_t Rq = (size_t)c.subsets * c.Q, Rk = (size_t)c.subsets * c.K;
  real* dx1 = cloneR(dout, Rq * C);
  real* dact0 = ARENA.zeros(Rq * C);
  real* dg = ARENA.get(Rq * I);
  if (bg) azBackwardGate(b.z2, s.z2, dout, dg, bg->g2); else azBackward(b.z2, s.z2, dout, dg, dqCond);
  real *dh = ARENA.get(Rq * 2 * I), *du = b.hasUp ? ARENA.get(Rq * I) : nullptr;
  swigluUpBwdK<<<blocks(Rq * I), 256, 0, STREAM>>>(s.h, b.hasUp ? s.u : nullptr, dg, dh, du, Rq, I);
  real* xf = ARENA.get(Rq * C);
  const int S = c.samples();   // (S > 1: bp's terms one sample's, the gradients into them summed over the S)
  { const real* sh = bp ? bp->f.sh : nullptr;
    if (!sh) { real* t = ARENA.get(Rq * C); b.lnF.shift.forward(s.af.cn, t, Rq); sh = t; }
    adaCombinePK<<<blocks(Rq * C), 256, 0, STREAM>>>(s.af.xn, s.af.sig, sh, xf, Rq * C, Rq / S * C, 1, b.lnF.chai); }
  real* dxf = ARENA.get(Rq * C);
  b.t1.backward(xf, dh, dxf, Rq);
  if (b.hasUp) b.up.backward(xf, du, dxf, Rq, true);
  if (bg) adaBackward(b.lnF, s.af, dxf, b.noResidual ? dact0 : dx1, bg->sigF, bg->shF, S);
  else adaBackward(b.lnF, s.af, dxf, b.noResidual ? dact0 : dx1, dqCond);
  real* dog = ARENA.get(Rq * HD);
  if (bg) azBackwardGate(b.z1, s.z1, dx1, dog, bg->g1); else azBackward(b.z1, s.z1, dx1, dog, dqCond);
  real *q = s.qkvg, *g = q + Rq * HD, *k = g + Rq * HD, *v = k + Rk * HD;
  real* d = ARENA.get(Rq * 2 * HD + Rk * 2 * HD);
  real *dq = d, *dgt = dq + Rq * HD, *dk = dgt + Rq * HD, *dv = dk + Rk * HD;
  real* dO = ARENA.get(Rq * HD);
  gateBwdK<<<blocks(Rq * HD), 256, 0, STREAM>>>(dog, s.o, g, dO, dgt, Rq * HD);
  sdpaBackward(atomSdpa(b, c, pl, dpl), q, HD, k, HD, v, HD, dO, HD, dq, HD, dk, HD, dv, HD, false, false, false, s.o, HD, s.lse);
  real *dxq = ARENA.get(Rq * C), *dxk = ARENA.get(Rk * C);
  if (b.kqNorm) {
    real *dqr = ARENA.get(Rq * HD), *dkr = ARENA.get(Rk * HD);
    lnBackward(b.qn, s.lq, dq, dqr, Rq, false);
    lnBackward(b.kn, s.lk, dk, dkr, Rk, false);
    dq = dqr; dk = dkr;
  }
  b.q.backward(s.xq, dq, dxq, Rq);
  b.gate.backward(s.xq, dgt, dxq, Rq, true);
  b.k.backward(s.xk, dk, dxk, Rk);
  b.v.backward(s.xk, dv, dxk, Rk, true);
  real* dkeys = ARENA.zeros(Rk * C);
  if (bg) adaBackward(b.lnK, s.ak, dxk, dkeys, bg->sigK, bg->shK, S); else adaBackward(b.lnK, s.ak, dxk, dkeys, dkCond);
  if (b.chained) scatter(c.toKeys, dkeys, dxq, C); else scatter(c.toKeys, dkeys, dact0, C);
  if (bg) adaBackward(b.lnQ, s.aq, dxq, dact0, bg->sigQ, bg->shQ, S); else adaBackward(b.lnQ, s.aq, dxq, dact0, dqCond);
  add(dx1, dact0, Rq * C);
  if (b.maskActPerBlock) rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(dx1, c.qMask, Rq, C);
  GK(cudaMemcpyAsync(dact, dx1, Rq * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

// a stack of blocks with its pair logits: LN (scale only) of the atom pair, one projection to blocks * heads (or a norm
// and a projection a block: pairNormPerBlock), each block's [subsets, H, Q, K]
struct AtomStack {
  std::vector<AtomBlock> blocks;
  LayerNorm pn; Linear proj;                        // shared
  std::vector<LayerNorm> pns; std::vector<Linear> projs;   // per block
  bool perBlock = false;
  int C = 0, Cp = 0, H = 0, D = 0;
};
// [rows = subsets Q K, nb H] column (b, h) -> pl[b][s, h, q, k]
__global__ void atomLogitsK(real* flat, real* pl, int subsets, int H, int Q, int K, int nb, int b, int ldf, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)subsets * H * Q * K) return;
  int k = t % K; size_t r = t / K; int q = r % Q; r /= Q; int h = r % H; int s = (int)(r / H);
  size_t row = ((size_t)s * Q + q) * K + k;
  real* f = flat + row * ldf + (size_t)b * H + h;
  if (back) *f += pl[t]; else pl[t] = *f;
}
struct StackRun { std::vector<real*> acts; real* pl = nullptr; real* pn = nullptr; size_t Rq = 0; };
// the stack's pair logits [nb][subsets, H, Q, K] into pl (pn: [rows, Cp] scratch)
inline void stackLogits(AtomStack& st, const AtomCtx& c, const real* pair, real* pn, real* pl) {
  const int nb = (int)st.blocks.size(), H = st.H;
  const size_t rows = (size_t)c.subsets * c.Q * c.K, per = (size_t)c.subsets * H * c.Q * c.K;
  Scratch scratch;
  if (st.perBlock) {
    real* flat = ARENA.get(rows * H);
    for (int b = 0; b < nb; ++b) {
      Scratch sc;
      lnForward(st.pns[b], pair, pn, rows);
      st.projs[b].forward(pn, flat, rows);
      atomLogitsK<<<blocks(per), 256, 0, STREAM>>>(flat, pl + b * per, c.subsets, H, c.Q, c.K, 1, 0, H, false);
    }
  } else {
    lnForward(st.pn, pair, pn, rows);
    real* flat = ARENA.get(rows * nb * H);
    st.proj.forward(pn, flat, rows);
    for (int b = 0; b < nb; ++b)
      atomLogitsK<<<blocks(per), 256, 0, STREAM>>>(flat, pl + b * per, c.subsets, H, c.Q, c.K, nb, b, nb * H, false);
  }
}
// ...and their backward: dpl -> dpair (added)
inline void stackLogitsBackward(AtomStack& st, const AtomCtx& c, const real* pair, const real* dpl, real* dpair) {
  Scratch scratch;
  const int nb = (int)st.blocks.size(), H = st.H;
  const size_t rows = (size_t)c.subsets * c.Q * c.K, per = (size_t)c.subsets * H * c.Q * c.K;
  if (st.perBlock) {
    real* dflat = ARENA.get(rows * H);
    for (int b = 0; b < nb; ++b) {
      Scratch sc;
      real* pn = ARENA.get(rows * st.Cp);
      LnSave ls = lnForward(st.pns[b], pair, pn, rows);
      GK(cudaMemsetAsync(dflat, 0, rows * H * sizeof(real), STREAM));
      atomLogitsK<<<blocks(per), 256, 0, STREAM>>>(dflat, const_cast<real*>(dpl) + b * per, c.subsets, H, c.Q, c.K, 1, 0, H, true);
      real* d1 = ARENA.get(rows * st.Cp);
      st.projs[b].backward(pn, dflat, d1, rows);
      lnBackward(st.pns[b], ls, d1, dpair, rows, true);
    }
  } else {
    real* dflat = ARENA.zeros(rows * nb * H);
    for (int b = 0; b < nb; ++b)
      atomLogitsK<<<blocks(per), 256, 0, STREAM>>>(dflat, const_cast<real*>(dpl) + b * per, c.subsets, H, c.Q, c.K, nb, b, nb * H, true);
    real* pn = ARENA.get(rows * st.Cp);
    LnSave ls = lnForward(st.pn, pair, pn, rows);
    real* dpn = ARENA.zeros(rows * st.Cp);
    st.proj.backward(pn, dflat, dpn, rows);
    lnBackward(st.pn, ls, dpn, dpair, rows, true);
  }
}
// what every sample of a batch shares in a stack: the pair logits, and each block's conditioning terms (stackPre);
// the samples' gradients into them, summed (stackGrads), and taken back once (stackPreBackward). The stack's ~half of
// its small GEMMs a sample were these.
struct StackPre { std::vector<BlockPre> b; real* pl = nullptr; };
struct StackGrads { std::vector<BlockGrads> b; real* dpl = nullptr; };
inline StackPre stackPre(AtomStack& st, const AtomCtx& c, const real* pair) {
  const int nb = (int)st.blocks.size();
  const size_t Rq = (size_t)c.subsets * c.Q, Rk = (size_t)c.subsets * c.K, rows = Rq * c.K, per = (size_t)c.subsets * st.H * c.Q * c.K;
  StackPre p;
  p.pl = ARENA.get(per * nb);
  { Scratch sc; real* pn = ARENA.get(rows * st.Cp); stackLogits(st, c, pair, pn, p.pl); }
  p.b.resize(nb);
  for (int i = 0; i < nb; ++i) {
    const AtomBlock& k = st.blocks[i];
    p.b[i].q = adaPre(k.lnQ, c.qCond, Rq); p.b[i].k = adaPre(k.lnK, c.kCond, Rk); p.b[i].f = adaPre(k.lnF, c.qCond, Rq);
    p.b[i].g1 = azPre(k.z1, c.qCond, Rq); p.b[i].g2 = azPre(k.z2, c.qCond, Rq);
  }
  return p;
}
inline StackGrads stackGrads(const AtomStack& st, const AtomCtx& c) {
  const int nb = (int)st.blocks.size(), C = st.C;
  const size_t Rq = (size_t)c.subsets * c.Q, Rk = (size_t)c.subsets * c.K, per = (size_t)c.subsets * st.H * c.Q * c.K;
  StackGrads g;
  g.dpl = ARENA.zeros(per * nb);
  g.b.resize(nb);
  for (auto& x : g.b) {
    real* z = ARENA.zeros(6 * Rq * C + 2 * Rk * C);
    x.sigQ = z; x.shQ = z + Rq * C; x.sigF = z + 2 * Rq * C; x.shF = z + 3 * Rq * C; x.g1 = z + 4 * Rq * C; x.g2 = z + 5 * Rq * C;
    x.sigK = z + 6 * Rq * C; x.shK = x.sigK + Rk * C;
  }
  return g;
}
inline void stackPreBackward(AtomStack& st, const AtomCtx& c, const real* pair, const StackPre& p, const StackGrads& g,
                             real* dqCond, real* dkCond, real* dpair) {
  const size_t Rq = (size_t)c.subsets * c.Q;
  stackLogitsBackward(st, c, pair, g.dpl, dpair);
  for (size_t i = 0; i < st.blocks.size(); ++i) {
    AtomBlock& k = st.blocks[i];
    adaPreBackward(k.lnQ, p.b[i].q, g.b[i].sigQ, g.b[i].shQ, dqCond);
    adaPreBackward(k.lnK, p.b[i].k, g.b[i].sigK, g.b[i].shK, dkCond);
    adaPreBackward(k.lnF, p.b[i].f, g.b[i].sigF, g.b[i].shF, dqCond);
    azPreBackward(k.z1, c.qCond, g.b[i].g1, Rq, dqCond);
    azPreBackward(k.z2, c.qCond, g.b[i].g2, Rq, dqCond);
  }
}
// (pre: the batch's shared terms - the pair logits and the conditioning not made here)
inline void stackForward(AtomStack& st, const AtomCtx& c, const real* pair, const real* act, real* out, StackRun& r,
                         const StackPre* pre = nullptr) {
  const int nb = (int)st.blocks.size(), H = st.H;
  // (per: a block's logits - one sample's where samples are stacked over them)
  const size_t Rq = (size_t)c.subsets * c.Q, rows = (size_t)c.subsets * c.Q * c.K, per = (size_t)(c.share ? c.share : c.subsets) * H * c.Q * c.K;
  r.Rq = Rq;
  r.acts.assign(nb + 1, nullptr);
  for (auto& p : r.acts) p = ARENA.get(Rq * st.C);
  if (pre) r.pl = pre->pl;
  else {
    r.pl = ARENA.get(per * nb);
    r.pn = ARENA.get(rows * st.Cp);
    stackLogits(st, c, pair, r.pn, r.pl);
  }
  GK(cudaMemcpyAsync(r.acts[0], act, Rq * st.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  for (int b = 0; b < nb; ++b) { Scratch sc; atomForward(st.blocks[b], c, r.acts[b], r.pl + b * per, r.acts[b + 1], pre ? &pre->b[b] : nullptr); }
  GK(cudaMemcpyAsync(out, r.acts[nb], Rq * st.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}
// dout -> dact (written), dqCond, dkCond, dpair (added); every block replayed before its backward
// (pre, sg: the batch's shared terms, and their gradients added into sg - dqCond, dkCond and dpair then untouched)
inline void stackBackward(AtomStack& st, const AtomCtx& c, const real* pair, const StackRun& r, const real* dout, real* dact,
                          real* dqCond, real* dkCond, real* dpair, const StackPre* pre = nullptr, StackGrads* sg = nullptr) {
  Scratch scratch;
  const int nb = (int)st.blocks.size(), H = st.H;
  const size_t Rq = r.Rq, per = (size_t)(c.share ? c.share : c.subsets) * H * c.Q * c.K;
  real* dpl = sg ? sg->dpl : ARENA.zeros(per * nb);
  real* dx = cloneR(dout, Rq * st.C);
  real* tmp = ARENA.get(Rq * st.C);
  for (int b = nb; b-- > 0;) {
    Scratch sc;
    const BlockPre* bp = pre ? &pre->b[b] : nullptr;
    AtomSave s = atomForward(st.blocks[b], c, r.acts[b], r.pl + b * per, tmp, bp);
    atomBackward(st.blocks[b], c, s, r.pl + b * per, dx, dx, dqCond, dkCond, dpl + b * per, bp, sg ? &sg->b[b] : nullptr);
  }
  if (!sg) stackLogitsBackward(st, c, pair, dpl, dpair);
  GK(cudaMemcpyAsync(dact, dx, Rq * st.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

// every parameter of an atom stack under `P` ("diffusion.encoder"), with its nb blocks
inline void forEachParam(AtomStack& st, const std::string& P, int nb, bool perBlock, bool chai, bool upGate,
                         const std::function<void(const std::string&, Param&, size_t)>& f, bool kqNorm = false,
                         bool noTransition2 = false) {
  const int C = st.C, Cp = st.Cp, H = st.H, HD = H * st.D, I = 2 * C;
  st.perBlock = perBlock;
  auto lin = [&](const std::string& n, Linear& l, int K, int N, const std::string& bias = "") {
    l.K = K; l.N = N; l.hasBias = !bias.empty(); l.w.fanIn = K; f(n, l.w, (size_t)K * N); if (!bias.empty()) f(bias, l.b, N);
  };
  if (perBlock) {
    st.pns.assign(nb, LayerNorm{}); st.projs.assign(nb, Linear{});
    for (int b = 0; b < nb; ++b) {
      st.pns[b].C = Cp; f(P + ".pairInputLayerNormScales." + std::to_string(b), st.pns[b].g, Cp);
      lin(P + ".pairLogitsProjections." + std::to_string(b), st.projs[b], Cp, H);
    }
  } else {
    st.pn.C = Cp; f(P + ".pairInputLayerNormScale", st.pn.g, Cp);
    lin(P + ".pairLogitsProjection", st.proj, Cp, nb * H);
  }
  st.blocks.assign(nb, AtomBlock{});
  auto ada = [&](const std::string& B, AdaLn& a) {
    a.C = C; a.Cc = C; a.chai = chai; a.xNorm.C = C;
    if (chai) a.xNorm.eps = 0.1f;
    if (!chai) { a.cNorm.C = C; f(B + "SingleCondLayerNormScale", a.cNorm.g, C); }
    lin(B + "SingleCondScaleWeights", a.scale, C, C, chai ? "" : B + "SingleCondScaleBias");
    lin(B + "SingleCondBias", a.shift, C, C);
  };
  for (int b = 0; b < nb; ++b) {
    AtomBlock& k = st.blocks[b];
    const std::string B = P + ".blocks." + std::to_string(b);
    k.C = C; k.H = H; k.D = st.D; k.I = I; k.hasUp = upGate;
    ada(B + ".q", k.lnQ); ada(B + ".k", k.lnK); ada(B + ".ffw", k.lnF);
    lin(B + ".qProjection", k.q, C, HD, B + ".qBias");
    lin(B + ".kProjection", k.k, C, HD);
    lin(B + ".vProjection", k.v, C, HD);
    lin(B + ".gatingQuery", k.gate, C, HD);
    k.kqNorm = kqNorm;
    if (kqNorm) {
      k.qn.C = HD; f(B + ".queryLayerNormScale", k.qn.g, HD); f(B + ".queryLayerNormOffset", k.qn.b, HD);
      k.kn.C = HD; f(B + ".keyLayerNormScale", k.kn.g, HD); f(B + ".keyLayerNormOffset", k.kn.b, HD);
    }
    if (noTransition2) { k.z1.identity = true; k.z1.proj.N = C; }
    else lin(B + ".Transition2", k.z1.proj, HD, C);
    lin(B + ".AdaptiveZeroCondWeights", k.z1.gate, C, C, B + ".AdaptiveZeroCondBias");
    k.z1.mult = noTransition2 ? 2 : 1;
    lin(B + ".ffwTransition1", k.t1, C, 2 * I);
    if (upGate) lin(B + ".ffwAToB", k.up, C, I);
    lin(B + ".ffwTransition2", k.z2.proj, I, C);
    lin(B + ".ffwAdaptiveZeroCondWeights", k.z2.gate, C, C, B + ".ffwAdaptiveZeroCondBias");
  }
}

}  // namespace af3
}  // namespace grad
