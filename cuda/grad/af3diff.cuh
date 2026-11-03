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
};
struct AzSave { const real* y; const real* cond; real *p, *g; size_t rows; };
inline AzSave azForward(const AdaZero& z, const real* y, const real* cond, real* out, size_t rows) {
  AzSave s{y, cond, ARENA.get(rows * z.proj.N), ARENA.get(rows * z.proj.N), rows};
  z.proj.forward(y, s.p, rows);
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
  z.proj.backward(s.y, dp, dy, rows);
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
    l.K = K; l.N = N; l.hasBias = !bias.empty(); f(n, l.w, (size_t)K * N); if (!bias.empty()) f(bias, l.b, N);
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
};
inline Sdpa tokenSdpa(const DiffBlock& b, int n, const real* mask, const real* pl, real* dpl = nullptr) {
  Sdpa a;
  a.B = 1; a.H = b.H; a.Q = n; a.K = n; a.D = b.D; a.scale = 1 / sqrt((real)b.D);
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

// act [n, C] -> out [n, C]
inline DiffBlockSave diffBlockForward(const DiffTransformer& t, const DiffBlock& b, const real* act, const real* cond,
                                      const real* mask, const real* pl, real* out, int n) {
  const int C = b.C, HD = b.H * b.D, I = b.I;
  DiffBlockSave s{};
  real* x = ARENA.get((size_t)n * C);
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
  sdpaForward(tokenSdpa(b, n, mask, pl), q, HD, k, HD, v, HD, s.o, HD);
  s.og = ARENA.get((size_t)n * HD);
  mulK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(s.og, s.o, g, (size_t)n * HD);
  s.x1 = ARENA.get((size_t)n * C);
  GK(cudaMemcpyAsync(s.x1, act, (size_t)n * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.z1 = azForward(b.z1, s.og, cond, s.x1, n);
  real* x2 = ARENA.get((size_t)n * C);
  s.a2 = adaForward(b.ln2, t.noResidual ? act : s.x1, cond, x2, n);
  s.h = ARENA.get((size_t)n * 2 * I);
  b.t1.forward(x2, s.h, n);
  if (b.hasUp) { s.u = ARENA.get((size_t)n * I); b.up.forward(x2, s.u, n); }
  s.g = ARENA.get((size_t)n * I);
  swigluUpK<<<blocks((size_t)n * I), 256, 0, STREAM>>>(s.h, b.hasUp ? s.u : nullptr, s.g, n, I);
  GK(cudaMemcpyAsync(out, s.x1, (size_t)n * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.z2 = azForward(b.z2, s.g, cond, out, n);
  return s;
}
// dout -> dact (written), dcond (added), dpl (added)
inline void diffBlockBackward(const DiffTransformer& t, DiffBlock& b, const DiffBlockSave& s, const real* act,
                              const real* cond, const real* mask, const real* pl, const real* dout, real* dact,
                              real* dcond, real* dpl, int n) {
  Scratch scratch;
  const int C = b.C, HD = b.H * b.D, I = b.I;
  real* dx1 = ARENA.get((size_t)n * C);
  GK(cudaMemcpyAsync(dx1, dout, (size_t)n * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  real* dact0 = ARENA.zeros((size_t)n * C);       // rf3: the transition's route to the block input
  // the transition
  real* dg = ARENA.get((size_t)n * I);
  azBackward(b.z2, s.z2, dout, dg, dcond);
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
  azBackward(b.z1, s.z1, dx1, dog, dcond);
  real *q = s.qkvg, *k = q + (size_t)n * HD, *v = k + (size_t)n * HD, *g = v + (size_t)n * HD;
  real* dqkvg = ARENA.get((size_t)n * 4 * HD);
  real *dq = dqkvg, *dk = dq + (size_t)n * HD, *dv = dk + (size_t)n * HD, *dgt = dv + (size_t)n * HD;
  real* dO = ARENA.get((size_t)n * HD);
  gateBwdK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(dog, s.o, g, dO, dgt, (size_t)n * HD);
  sdpaBackward(tokenSdpa(b, n, mask, pl, dpl), q, HD, k, HD, v, HD, dO, HD, dq, HD, dk, HD, dv, HD);
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
struct DiffRun {
  std::vector<real*> acts;         // [blocks + 1]
  real *pn = nullptr, *pl = nullptr;   // LN'd pair [nn, Cz]; every block's logits [blocks][H, n, n]
  LnSave pln;
  int n = 0;
};
inline void transformerForward(DiffTransformer& t, DiffRun& r, const real* act, const real* cond, const real* mask,
                               const real* pairCond, real* out, int n) {
  const size_t nn = (size_t)n * n;
  const int nb = (int)t.blocks.size();
  if (r.n != n || r.acts.empty()) {
    for (real* p : r.acts) GK(cudaFree(p));
    if (r.pn) { GK(cudaFree(r.pn)); GK(cudaFree(r.pl)); }
    r.acts.assign(nb + 1, nullptr);
    for (auto& p : r.acts) GK(cudaMalloc(&p, (size_t)n * t.C * sizeof(real)));
    GK(cudaMalloc(&r.pn, nn * t.Cz * sizeof(real)));
    GK(cudaMalloc(&r.pl, (size_t)nb * t.H * nn * sizeof(real)));
    r.n = n;
  }
  ARENA.reset();
  {
    real* mean = ARENA.get(nn); real* rstd = ARENA.get(nn);
    lnFwdK<<<blocks(nn, 8), 256, 0, STREAM>>>(pairCond, t.pairNorm.g.w, nullptr, r.pn, mean, rstd, nn, t.Cz, t.pairNorm.eps);
    real* flat = ARENA.get(nn * t.perSuper * t.H);
    for (int b = 0; b < nb; ++b) {
      if (b % t.perSuper == 0) t.superProj[b / t.perSuper].forward(r.pn, flat, nn);
      superBiasK<<<blocks(nn * t.H), 256, 0, STREAM>>>(flat, r.pl + (size_t)b * t.H * nn, n, t.H, t.perSuper, b % t.perSuper, false);
    }
  }
  GK(cudaMemcpyAsync(r.acts[0], act, (size_t)n * t.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  for (int b = 0; b < nb; ++b) {
    ARENA.reset();
    diffBlockForward(t, t.blocks[b], r.acts[b], cond, mask, r.pl + (size_t)b * t.H * nn, r.acts[b + 1], n);
  }
  ARENA.reset();
  GK(cudaMemcpyAsync(out, r.acts[nb], (size_t)n * t.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}
// dout -> dact (written), dcond (added), dpairCond (added)
inline void transformerBackward(DiffTransformer& t, DiffRun& r, const real* cond, const real* mask,
                                const real* pairCond, const real* dout, real* dact, real* dcond, real* dpairCond) {
  const int n = r.n, nb = (int)t.blocks.size();
  const size_t nn = (size_t)n * n;
  real *dx, *dpl, *tmp;
  GK(cudaMalloc(&dx, (size_t)n * t.C * sizeof(real)));
  GK(cudaMalloc(&tmp, (size_t)n * t.C * sizeof(real)));
  GK(cudaMalloc(&dpl, (size_t)nb * t.H * nn * sizeof(real)));
  GK(cudaMemsetAsync(dpl, 0, (size_t)nb * t.H * nn * sizeof(real), STREAM));
  GK(cudaMemcpyAsync(dx, dout, (size_t)n * t.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  for (int b = nb; b-- > 0;) {
    ARENA.reset();
    const real* pl = r.pl + (size_t)b * t.H * nn;
    DiffBlockSave s = diffBlockForward(t, t.blocks[b], r.acts[b], cond, mask, pl, tmp, n);
    diffBlockBackward(t, t.blocks[b], s, r.acts[b], cond, mask, pl, dx, dx, dcond, dpl + (size_t)b * t.H * nn, n);
  }
  ARENA.reset();
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
  ARENA.reset();
  GK(cudaMemcpyAsync(dact, dx, (size_t)n * t.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  GK(cudaStreamSynchronize(STREAM));
  GK(cudaFree(dx)); GK(cudaFree(tmp)); GK(cudaFree(dpl));
}

}  // namespace af3
}  // namespace grad
