// BioEmu2's trunk for training (cuda/bioemu2/src/trunk.cuh is the reading: network/bioemu2.py's trunk_pass): Boltz's
// modules on the vendor's PyTorch layouts ((out, in), stacked [layers, ...] - a parameter here is one layer's slice,
// named "<tensor>@<layer>"):
//
//   MSA layer:   m += PWA(m, z)  (no mask);  m += T(m);  z += OPM(m) / valid rows + b;
//                z += triangle out (z);  z += triangle in (z);  z += T(z)
//   PairMixer:   z += triangle out;  z += triangle in;  z += T(z)
//   T(x) = fc3(silu(fc1 LN(x)) * fc2 LN(x)), no biases;  triangles blocked a | b, no mask, the incoming form
//   sum_k a_ki b_kj (Boltz's), every Linear bias-free but the outer product's output
#pragma once
#include "af3.cuh"

namespace grad {
namespace bioemu2 {

struct BTransition {
  LayerNorm ln;
  Linear fc1, fc2, fc3;     // [I, C], [I, C], [Co, I] (out, in)
  int C = 0, I = 0, Co = 0;
};
struct BtSave { LnSave ln; real *xn, *h1, *h2; size_t rows; };
__global__ void siluMulK(const real* a, const real* b, real* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = a[t] / (1 + exp(-a[t])) * b[t];
}
__global__ void siluMulBwdK(const real* a, const real* b, const real* dy, real* da, real* db, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  real s = 1 / (1 + exp(-a[t]));
  da[t] = dy[t] * b[t] * (s + a[t] * s * (1 - s));
  db[t] = dy[t] * a[t] * s;
}
// out (+)= T(x)
inline BtSave btForward(const BTransition& t, const real* x, real* out, size_t rows) {
  BtSave s{}; s.rows = rows;
  s.xn = ARENA.get(rows * t.C);
  s.ln = lnForward(t.ln, x, s.xn, rows);
  s.h1 = ARENA.get(rows * t.I); s.h2 = ARENA.get(rows * t.I);
  t.fc1.forward(s.xn, s.h1, rows); t.fc2.forward(s.xn, s.h2, rows);
  Scratch scratch;
  real* g = ARENA.get(rows * t.I);
  siluMulK<<<blocks(rows * t.I), 256, 0, STREAM>>>(s.h1, s.h2, g, rows * t.I);
  t.fc3.forward(g, out, rows, 1);
  return s;
}
inline void btBackward(BTransition& t, const BtSave& s, const real* dout, real* dx) {
  Scratch scratch;
  size_t rows = s.rows;
  real *g = ARENA.get(rows * t.I), *dg = ARENA.get(rows * t.I), *d1 = ARENA.get(rows * t.I), *d2 = ARENA.get(rows * t.I);
  siluMulK<<<blocks(rows * t.I), 256, 0, STREAM>>>(s.h1, s.h2, g, rows * t.I);
  t.fc3.backward(g, dout, dg, rows);
  siluMulBwdK<<<blocks(rows * t.I), 256, 0, STREAM>>>(s.h1, s.h2, dg, d1, d2, rows * t.I);
  real* dxn = ARENA.get(rows * t.C);
  t.fc1.backward(s.xn, d1, dxn, rows);
  t.fc2.backward(s.xn, d2, dxn, rows, true);
  lnBackward(t.ln, s.ln, dxn, dx, rows, true);
}

// the pair half every layer has
struct PairHalf { TriMul out, in; BTransition tr; };
struct PairHalfSave { TriMulSave o, i; BtSave t; real *z1, *z2; };
inline PairHalfSave pairHalfForward(PairHalf& p, const real* z, const real* ones, real* zOut, int T) {
  const int C = p.out.C; const size_t P = (size_t)T * T;
  PairHalfSave s{};
  real* u = ARENA.get(P * C);
  s.z1 = af3::cloneR(z, P * C);
  s.o = triMulForward(p.out, z, ones, u, T); add(s.z1, u, P * C);
  s.z2 = af3::cloneR(s.z1, P * C);
  s.i = triMulForward(p.in, s.z1, ones, u, T); add(s.z2, u, P * C);
  GK(cudaMemcpyAsync(zOut, s.z2, P * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.t = btForward(p.tr, s.z2, zOut, P);
  return s;
}
// dzOut -> dz (written; may alias)
inline void pairHalfBackward(PairHalf& p, const PairHalfSave& s, const real* ones, const real* dzOut, real* dz) {
  const int C = p.out.C; const size_t P = (size_t)s.t.rows;
  real* d = af3::cloneR(dzOut, P * C);
  btBackward(p.tr, s.t, dzOut, d);
  { Scratch sc; real* du = af3::cloneR(d, P * C); triMulBackward(p.in, s.i, ones, du, d); }
  { Scratch sc; real* du = af3::cloneR(d, P * C); triMulBackward(p.out, s.o, ones, du, d); }
  GK(cudaMemcpyAsync(dz, d, P * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

struct MsaLayer {
  af3::Pwa pwa;
  BTransition msaTr;
  Opm opm;
  PairHalf pair;
};
struct MsaLayerSave { af3::PwaSave pwa; BtSave tr; OpmSave opm; PairHalfSave ph; real *m1, *z1; };
inline MsaLayerSave msaLayerForward(MsaLayer& l, const real* m, const real* z, const real* ones, real* mOut, real* zOut,
                                    int S, int T) {
  const int Cm = l.pwa.Cm, C = l.pair.out.C;
  const size_t R = (size_t)S * T, P = (size_t)T * T;
  MsaLayerSave s{};
  s.m1 = af3::cloneR(m, R * Cm);
  s.pwa = af3::pwaForward(l.pwa, m, nullptr, nullptr, z, s.m1, S, T);
  GK(cudaMemcpyAsync(mOut, s.m1, R * Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.tr = btForward(l.msaTr, s.m1, mOut, R);
  s.z1 = af3::cloneR(z, P * C);
  s.opm = opmForward(l.opm, mOut, nullptr, s.z1, S, T);
  s.ph = pairHalfForward(l.pair, s.z1, ones, zOut, T);
  return s;
}
inline void msaLayerBackward(MsaLayer& l, const MsaLayerSave& s, const real* ones, const real* dmOut, const real* dzOut,
                             real* dm, real* dz, int S, int T) {
  const int Cm = l.pwa.Cm, C = l.pair.out.C;
  const size_t R = (size_t)S * T, P = (size_t)T * T;
  real* dz1 = ARENA.get(P * C);
  pairHalfBackward(l.pair, s.ph, ones, dzOut, dz1);
  real* d = af3::cloneR(dmOut, R * Cm);
  opmBackward(l.opm, s.opm, nullptr, dz1, d);                          // z1 = z + OPM(mOut)
  { Scratch sc; real* du = af3::cloneR(d, R * Cm); btBackward(l.msaTr, s.tr, du, d); }
  { Scratch sc; real* du = af3::cloneR(d, R * Cm); af3::pwaBackward(l.pwa, s.pwa, nullptr, du, d, dz1); }
  GK(cudaMemcpyAsync(dm, d, R * Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  GK(cudaMemcpyAsync(dz, dz1, P * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

// names: the vendor's, a layer's slice "<tensor>@<l>"; dims from the caller (C 256, Cm 128, 8 heads of 32, ...)
using ParamFn = std::function<void(const std::string&, Param&, size_t)>;
inline void triParams(TriMul& t, const std::string& p, int l, int C, bool outgoing, const ParamFn& f) {
  const std::string L = "@" + std::to_string(l);
  auto lin = [&](const std::string& n, Linear& x, int K, int N) { x.K = K; x.N = N; x.trans = true; x.hasBias = false; f(p + n + L, x.w, (size_t)K * N); };
  t.C = C; t.I = C; t.outgoing = outgoing; t.interleaved = false; t.swapAB = !outgoing; t.masked = false;
  t.lnIn.C = C; f(p + "norm_in/weight" + L, t.lnIn.g, C); f(p + "norm_in/bias" + L, t.lnIn.b, C);
  t.lnCenter.C = C; f(p + "norm_out/weight" + L, t.lnCenter.g, C); f(p + "norm_out/bias" + L, t.lnCenter.b, C);
  lin("p_in/weight", t.proj, C, 2 * C);
  lin("g_in/weight", t.gate, C, 2 * C);
  lin("p_out/weight", t.out, C, C);
  lin("g_out/weight", t.gating, C, C);
}
inline void trParams(BTransition& t, const std::string& p, int l, int C, int I, int Co, const ParamFn& f) {
  const std::string L = "@" + std::to_string(l);
  t.C = C; t.I = I; t.Co = Co;
  t.ln.C = C; f(p + "norm/weight" + L, t.ln.g, C); f(p + "norm/bias" + L, t.ln.b, C);
  auto lin = [&](const std::string& n, Linear& x, int K, int N) { x.K = K; x.N = N; x.trans = true; x.hasBias = false; f(p + n + L, x.w, (size_t)K * N); };
  lin("fc1/weight", t.fc1, C, I); lin("fc2/weight", t.fc2, C, I); lin("fc3/weight", t.fc3, I, Co);
}
inline void msaLayerParams(MsaLayer& m, const std::string& ml, int l, int Cm, int C, int H, int D, int Co, int Im, int Iz,
                           real validRows, const ParamFn& f) {
  const std::string L = "@" + std::to_string(l);
  const std::string pw = ml + "pair_weighted_averaging/";
  auto lin = [&](const std::string& n, Linear& x, int K, int N) { x.K = K; x.N = N; x.trans = true; x.hasBias = false; f(n + L, x.w, (size_t)K * N); };
  af3::Pwa& a = m.pwa;
  a.Cm = Cm; a.Cz = C; a.H = H; a.D = D; a.noKeyMask = true;
  a.ln.C = Cm; f(pw + "norm_m/weight" + L, a.ln.g, Cm); f(pw + "norm_m/bias" + L, a.ln.b, Cm);
  a.pn.C = C; f(pw + "norm_z/weight" + L, a.pn.g, C); f(pw + "norm_z/bias" + L, a.pn.b, C);
  lin(pw + "proj_z/weight", a.logits, C, H);
  lin(pw + "proj_m/weight", a.v, Cm, H * D);
  lin(pw + "proj_g/weight", a.gate, Cm, H * D);
  lin(pw + "proj_o/weight", a.out, H * D, Cm);
  trParams(m.msaTr, ml + "msa_transition/", l, Cm, Im, Cm, f);
  const std::string op = ml + "outer_product_mean/";
  Opm& o = m.opm;
  o.C = Cm; o.O = Co; o.Cz = C; o.fused = true; o.fixedCount = validRows;
  o.ln.C = Cm; f(op + "norm/weight" + L, o.ln.g, Cm); f(op + "norm/bias" + L, o.ln.b, Cm);
  lin(op + "proj_ab/weight", o.proj, Cm, 2 * Co);
  f(op + "proj_o/weight" + L, o.wo, (size_t)Co * Co * C);       // (out, in): [C, Co Co] - transposed when loaded
  f(op + "proj_o/bias" + L, o.bo, C);
  triParams(m.pair.out, ml + "tri_mul_out/", l, C, true, f);
  triParams(m.pair.in, ml + "tri_mul_in/", l, C, false, f);
  trParams(m.pair.tr, ml + "z_transition/", l, C, Iz, C, f);
}
inline void mixerParams(PairHalf& p, const std::string& pm, int l, int C, int Iz, const ParamFn& f) {
  triParams(p.out, pm + "tri_mul_out/", l, C, true, f);
  triParams(p.in, pm + "tri_mul_in/", l, C, false, f);
  trParams(p.tr, pm + "transition_z/", l, C, Iz, C, f);
}

}  // namespace bioemu2
}  // namespace grad
