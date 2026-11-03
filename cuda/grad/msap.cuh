// MSA Pairformer (cuda/msap) and ESMFold2's MSA encoder (cuda/ef2/src/msa.cuh) for training. They share their
// module set - the pair-weighted averaging without a key mask, the SwiGLU transition, the pair-only pairformer block
// (cuda/grad/af3.cuh) - and differ in their outer product:
//
//   MSA Pairformer's query-biased OPM (msap.cu's qOpmF32):
//     nm = LN(m); q = nm[query rows] Wq, k = nm Wk (256 = two halves of 128, each LayerNormed, one scale for both);
//     raw_t[s] = mean_l q_t[l] . k_t[s, l] / sqrt(128);  lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init;
//     w = softmax_s(raw_0 - lambda raw_1);  P = sum_s (w_s + 1e-32) a_s (x) b_s / (sum w + 1e-32);  a, b = nm Wl, nm Wr
//     pair += SwiGLU(P Wp + bp)   (the gate the first half)
//   ESMFold2's: one projection [Cm, 2H] split blocked, times the mask, Wout + b over max(count, 1) (pair.cuh's Opm,
//     fused and countClamp)
#pragma once
#include "af3.cuh"

namespace grad {
namespace msap {

struct QOpm {
  LayerNorm ln, qn, kn;            // qn, kn over 128, applied to both halves
  Linear q, k, left, right, toPair;
  Param lq1, lk1, lq2, lk2;        // [128]
  double lambdaInit = 0;
  int Cm = 464, O = 16, Cz = 256;
};
struct QOpmSave {
  LnSave ln, lq, lk;
  real *nm, *qraw, *kraw, *qn, *kn, *a, *b, *raw, *w, *outer, *h;
  double lambda, e1, e2, ws;
  int S, L;
};
// raw[t * S + s] = coef sum_l sum_c qn[(l, t), c] kn[s, (l, t), c]: a block a (s, t)
__global__ void seqScoresK(const real* qn, const real* kn, real* raw, int S, int L, real coef) {
  int s = blockIdx.x, t = blockIdx.y;
  real acc = 0;
  for (int l = threadIdx.x / 32; l < L; l += blockDim.x / 32) {
    const real* qv = qn + ((size_t)l * 2 + t) * 128;
    const real* kv = kn + (((size_t)s * L + l) * 2 + t) * 128;
    for (int c = threadIdx.x & 31; c < 128; c += 32) acc += qv[c] * kv[c];
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(~0u, acc, o);
  __shared__ real red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = acc;
  __syncthreads();
  if (threadIdx.x == 0) { real a = 0; for (int w = 0; w < (int)blockDim.x / 32; ++w) a += red[w]; raw[(size_t)t * S + s] = a * coef; }
}
// dkn[s, (l, t), c] = coef draw_t[s] qn[(l, t), c]; dqn[(l, t), c] = coef sum_s draw_t[s] kn[s, (l, t), c]
__global__ void seqScoresBwdKK(const real* qn, const real* draw, real* dkn, int S, int L, real coef) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)S * L * 256) return;
  int c = i % 128; size_t r = i / 128; int t = r % 2; r /= 2; int l = r % L; int s = (int)(r / L);
  dkn[i] = coef * draw[(size_t)t * S + s] * qn[((size_t)l * 2 + t) * 128 + c];
}
__global__ void seqScoresBwdQK(const real* kn, const real* draw, real* dqn, int S, int L, real coef) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)L * 256) return;
  int c = i % 128; size_t r = i / 128; int t = r % 2; int l = (int)(r / 2);
  real a = 0;
  for (int s = 0; s < S; ++s) a += draw[(size_t)t * S + s] * kn[(((size_t)s * L + l) * 2 + t) * 128 + c];
  dqn[i] = coef * a;
}
// x [S, L 16] scaled a row: y[s, :] = x[s, :] * f[s] (+ accumulate)
__global__ void rowScaleK(const real* x, const real* f, real* y, int S, int W, real eps) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * W) return;
  y[t] = x[t] * (f[t / W] + eps);
}
// out[s] = sum_c x[s, c] y[s, c]
__global__ void rowDotK(const real* x, const real* y, real* out, int S, int W) {
  int s = blockIdx.x;
  real a = 0;
  for (int c = threadIdx.x; c < W; c += blockDim.x) a += x[(size_t)s * W + c] * y[(size_t)s * W + c];
  for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(~0u, a, o);
  __shared__ real red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = a;
  __syncthreads();
  if (threadIdx.x == 0) { real t = 0; for (int w = 0; w < (int)blockDim.x / 32; ++w) t += red[w]; out[s] = t; }
}
// h [P, 2C] -> pair += silu(h[:, :C]) h[:, C:]  (and its backward)
__global__ void gluAddK(const real* h, real* pair, size_t P, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= P * C) return;
  size_t r = t / C; int c = t % C;
  real g = h[r * 2 * C + c], x = h[r * 2 * C + C + c];
  pair[t] += g / (1 + exp(-g)) * x;
}
__global__ void gluBwdK(const real* h, const real* dpair, real* dh, size_t P, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= P * C) return;
  size_t r = t / C; int c = t % C;
  real g = h[r * 2 * C + c], x = h[r * 2 * C + C + c], s = 1 / (1 + exp(-g));
  dh[r * 2 * C + c] = dpair[t] * x * (s + g * s * (1 - s));
  dh[r * 2 * C + C + c] = dpair[t] * g * s;
}

inline double hostDot(const real* a, const real* b, int n) {
  std::vector<real> x(n), y(n);
  GK(cudaMemcpyAsync(x.data(), a, n * sizeof(real), cudaMemcpyDeviceToHost, STREAM));
  GK(cudaMemcpyAsync(y.data(), b, n * sizeof(real), cudaMemcpyDeviceToHost, STREAM));
  GK(cudaStreamSynchronize(STREAM));
  double d = 0; for (int i = 0; i < n; ++i) d += (double)x[i] * y[i];
  return d;
}

// pair [L, L, Cz] += qopm(m [S, L, Cm]); row 0 is the query
inline QOpmSave qopmForward(const QOpm& o, const real* m, real* pair, int S, int L) {
  const size_t R = (size_t)S * L, P = (size_t)L * L, LC = (size_t)L * o.O;
  QOpmSave s{}; s.S = S; s.L = L;
  s.nm = ARENA.get(R * o.Cm);
  s.ln = lnForward(o.ln, m, s.nm, R);
  s.qraw = ARENA.get((size_t)L * 256); s.kraw = ARENA.get(R * 256);
  o.q.forward(s.nm, s.qraw, L); o.k.forward(s.nm, s.kraw, R);
  s.qn = ARENA.get((size_t)L * 256); s.kn = ARENA.get(R * 256);
  s.lq = lnForward(o.qn, s.qraw, s.qn, (size_t)L * 2);
  s.lk = lnForward(o.kn, s.kraw, s.kn, R * 2);
  s.a = ARENA.get(R * o.O); s.b = ARENA.get(R * o.O);
  o.left.forward(s.nm, s.a, R); o.right.forward(s.nm, s.b, R);
  s.raw = ARENA.get((size_t)2 * S);
  seqScoresK<<<dim3(S, 2), 256, 0, STREAM>>>(s.qn, s.kn, s.raw, S, L, 1 / sqrt((real)128) / L);
  s.e1 = std::exp(hostDot(o.lq1.w, o.lk1.w, 128)); s.e2 = std::exp(hostDot(o.lq2.w, o.lk2.w, 128));
  s.lambda = s.e1 - s.e2 + o.lambdaInit;
  std::vector<real> raw(2 * S), w(S);
  GK(cudaMemcpyAsync(raw.data(), s.raw, 2 * S * sizeof(real), cudaMemcpyDeviceToHost, STREAM));
  GK(cudaStreamSynchronize(STREAM));
  double mx = -INFINITY, sum = 0;
  for (int i = 0; i < S; ++i) mx = std::max(mx, (double)raw[i] - s.lambda * raw[S + i]);
  for (int i = 0; i < S; ++i) sum += std::exp((double)raw[i] - s.lambda * raw[S + i] - mx);
  s.ws = 0;
  for (int i = 0; i < S; ++i) { w[i] = (real)(std::exp((double)raw[i] - s.lambda * raw[S + i] - mx) / sum); s.ws += w[i]; }
  s.w = ARENA.get(S);
  GK(cudaMemcpyAsync(s.w, w.data(), S * sizeof(real), cudaMemcpyHostToDevice, STREAM));
  s.outer = ARENA.get(P * o.O * o.O);
  s.h = ARENA.get(P * 2 * o.Cz);
  {
    Scratch scratch;
    real *aw = ARENA.get(R * o.O), *p2 = ARENA.get(LC * LC);
    rowScaleK<<<blocks(R * o.O), 256, 0, STREAM>>>(s.a, s.w, aw, S, (int)LC, (real)1e-32);
    mmAT(aw, s.b, p2, S, (int)LC, (int)LC, 0, (real)(1 / (s.ws + 1e-32)));      // [(i c), (j e)]
    outerToPairK<<<blocks(P * o.O * o.O), 256, 0, STREAM>>>(p2, s.outer, L, o.O, false);
  }
  o.toPair.forward(s.outer, s.h, P);
  gluAddK<<<blocks(P * o.Cz), 256, 0, STREAM>>>(s.h, pair, P, o.Cz);
  return s;
}
// dpair -> dm (added); every parameter's gradient
inline void qopmBackward(QOpm& o, const QOpmSave& s, const real* dpair, real* dm) {
  Scratch scratch;
  const int S = s.S, L = s.L;
  const size_t R = (size_t)S * L, P = (size_t)L * L, LC = (size_t)L * o.O;
  const real inv = (real)(1 / (s.ws + 1e-32));
  real* dh = ARENA.get(P * 2 * o.Cz);
  gluBwdK<<<blocks(P * o.Cz), 256, 0, STREAM>>>(s.h, dpair, dh, P, o.Cz);
  real* douter = ARENA.get(P * o.O * o.O);
  o.toPair.backward(s.outer, dh, douter, P);
  // outer = sum_s (w_s + eps) a_s (x) b_s / ws: dws = -sum douter outer / ws
  const double dws = -dot(douter, s.outer, P * o.O * o.O) / (s.ws + 1e-32);
  real* dp2 = ARENA.get(LC * LC);
  outerToPairK<<<blocks(P * o.O * o.O), 256, 0, STREAM>>>(douter, dp2, L, o.O, true);
  real *X = ARENA.get(R * o.O), *Y = ARENA.get(R * o.O);
  mmBT(s.b, dp2, X, S, (int)LC, (int)LC, 0, inv);       // X = b dP^T / ws  [S, (i c)]
  mm(s.a, dp2, Y, S, (int)LC, (int)LC, 0, inv);         // Y = a dP / ws    [S, (j e)]
  real *da = ARENA.get(R * o.O), *db = ARENA.get(R * o.O), *dwv = ARENA.get(S);
  rowScaleK<<<blocks(R * o.O), 256, 0, STREAM>>>(X, s.w, da, S, (int)LC, (real)1e-32);
  rowScaleK<<<blocks(R * o.O), 256, 0, STREAM>>>(Y, s.w, db, S, (int)LC, (real)1e-32);
  rowDotK<<<S, 256, 0, STREAM>>>(s.a, X, dwv, S, (int)LC);
  // the softmax over s, and its two logits: z = raw0 - lambda raw1
  std::vector<real> w(S), dw(S), raw(2 * S);
  GK(cudaMemcpyAsync(w.data(), s.w, S * sizeof(real), cudaMemcpyDeviceToHost, STREAM));
  GK(cudaMemcpyAsync(dw.data(), dwv, S * sizeof(real), cudaMemcpyDeviceToHost, STREAM));
  GK(cudaMemcpyAsync(raw.data(), s.raw, 2 * S * sizeof(real), cudaMemcpyDeviceToHost, STREAM));
  GK(cudaStreamSynchronize(STREAM));
  double wdw = 0;
  for (int i = 0; i < S; ++i) { dw[i] += (real)dws; wdw += (double)w[i] * dw[i]; }
  std::vector<real> draw(2 * S);
  double dlambda = 0;
  for (int i = 0; i < S; ++i) {
    double dz = w[i] * (dw[i] - wdw);
    draw[i] = (real)dz; draw[S + i] = (real)(-s.lambda * dz);
    dlambda += -dz * raw[S + i];
  }
  real* drawD = ARENA.get(2 * S);
  GK(cudaMemcpyAsync(drawD, draw.data(), 2 * S * sizeof(real), cudaMemcpyHostToDevice, STREAM));
  axpyR<<<blocks(128), 256, 0, STREAM>>>(o.lq1.g, o.lk1.w, (real)(dlambda * s.e1), 128);
  axpyR<<<blocks(128), 256, 0, STREAM>>>(o.lk1.g, o.lq1.w, (real)(dlambda * s.e1), 128);
  axpyR<<<blocks(128), 256, 0, STREAM>>>(o.lq2.g, o.lk2.w, (real)(-dlambda * s.e2), 128);
  axpyR<<<blocks(128), 256, 0, STREAM>>>(o.lk2.g, o.lq2.w, (real)(-dlambda * s.e2), 128);
  const real coef = 1 / sqrt((real)128) / L;
  real *dkn = ARENA.get(R * 256), *dqn = ARENA.get((size_t)L * 256);
  seqScoresBwdKK<<<blocks(R * 256), 256, 0, STREAM>>>(s.qn, drawD, dkn, S, L, coef);
  seqScoresBwdQK<<<blocks((size_t)L * 256), 256, 0, STREAM>>>(s.kn, drawD, dqn, S, L, coef);
  real *dqr = ARENA.get((size_t)L * 256), *dkr = ARENA.get(R * 256);
  lnBackward(o.qn, s.lq, dqn, dqr, (size_t)L * 2, false);
  lnBackward(o.kn, s.lk, dkn, dkr, R * 2, false);
  real* dnm = ARENA.get(R * o.Cm);
  o.k.backward(s.nm, dkr, dnm, R);
  o.left.backward(s.nm, da, dnm, R, true);
  o.right.backward(s.nm, db, dnm, R, true);
  o.q.backward(s.nm, dqr, dnm, L, true);                // the query's rows lead
  lnBackward(o.ln, s.ln, dnm, dm, R, true);
}

// ---------------------------------------------------------------- MSA Pairformer's layer
//   msa += PWA(msa, pair); msa += T(msa); pair += qOPM(msa); pair += the pair-only block
struct Layer {
  af3::Pwa pwa;
  af3::GatedTransition trans;
  QOpm opm;
  af3::Pairformer pairBlock;
};
struct LayerSave { af3::PwaSave pwa; af3::GtSave tr; QOpmSave opm; af3::PairformerSave pb; real *m1, *p1; };
inline LayerSave layerForward(Layer& l, const real* m, const real* pair, const real* msaMask, const real* pairMask,
                              const real* maskT, real* mOut, real* pairOut, int S, int L, int Cz) {
  const size_t R = (size_t)S * L, P = (size_t)L * L;
  const int Cm = l.pwa.Cm;
  LayerSave s{};
  s.m1 = af3::cloneR(m, R * Cm);
  s.pwa = af3::pwaForward(l.pwa, m, msaMask, nullptr, pair, s.m1, S, L);
  GK(cudaMemcpyAsync(mOut, s.m1, R * Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.tr = af3::gtForward(l.trans, s.m1, mOut, R, 1);
  s.p1 = af3::cloneR(pair, P * Cz);
  s.opm = qopmForward(l.opm, mOut, s.p1, S, L);
  s.pb = af3::pairformerForward(l.pairBlock, {}, s.p1, nullptr, pairMask, maskT, nullptr, nullptr, pairOut, nullptr, L);
  return s;
}
inline void layerBackward(Layer& l, const LayerSave& s, const real* msaMask, const real* pairMask, const real* maskT,
                          const real* dmOut, const real* dpairOut, real* dm, real* dpair, int S, int L, int Cz) {
  const size_t R = (size_t)S * L, P = (size_t)L * L;
  const int Cm = l.pwa.Cm;
  real* dp = af3::cloneR(dpairOut, P * Cz);
  af3::pairformerBackward(l.pairBlock, {}, s.pb, pairMask, maskT, nullptr, dpairOut, nullptr, dp, nullptr, nullptr);
  real* d = af3::cloneR(dmOut, R * Cm);
  qopmBackward(l.opm, s.opm, dp, d);                                // pair1 = pair + qOPM(mOut)
  { Scratch sc; real* du = af3::cloneR(d, R * Cm); af3::gtBackward(l.trans, s.tr, du, d); }
  { Scratch sc; real* du = af3::cloneR(d, R * Cm); af3::pwaBackward(l.pwa, s.pwa, msaMask, du, d, dp); }
  GK(cudaMemcpyAsync(dm, d, R * Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  GK(cudaMemcpyAsync(dpair, dp, P * Cz * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

// every parameter of layer l under `pre` ("f/msap/blocks/0"), the export's names
inline void forEachParam(Layer& l, const std::string& pre, int Cm, int Cz, int H, int D, int I, double lambdaInit,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  auto ln = [&](const std::string& n, LayerNorm& x, int C, const std::string& s = "Scale", const std::string& o = "Offset") {
    x.C = C; f(n + s, x.g, C); f(n + o, x.b, C);
  };
  auto lin = [&](const std::string& n, Linear& x, int K, int N, const std::string& bias = "") {
    x.K = K; x.N = N; x.hasBias = !bias.empty(); f(n, x.w, (size_t)K * N); if (!bias.empty()) f(bias, x.b, N);
  };
  const std::string Pw = pre + "/pwa/";
  l.pwa.Cm = Cm; l.pwa.Cz = Cz; l.pwa.H = H; l.pwa.D = D; l.pwa.noKeyMask = true;
  ln(Pw + "norm", l.pwa.ln, Cm);
  ln(Pw + "pairNorm", l.pwa.pn, Cz);
  lin(Pw + "pairLogits", l.pwa.logits, Cz, H);
  lin(Pw + "value", l.pwa.v, Cm, H * D);
  lin(Pw + "gate", l.pwa.gate, Cm, H * D);
  lin(Pw + "output", l.pwa.out, H * D, Cm);
  const std::string Tr = pre + "/msaTransition/";
  l.trans.C = Cm; l.trans.F = I;
  ln(Tr + "inputLayerNorm", l.trans.ln, Cm);
  lin(Tr + "transition1", l.trans.t1, Cm, 2 * I);
  lin(Tr + "transition2", l.trans.t2, I, Cm);
  const std::string O = pre + "/opm/";
  QOpm& o = l.opm;
  o.Cm = Cm; o.Cz = Cz; o.lambdaInit = lambdaInit;
  ln(O + "norm", o.ln, Cm);
  lin(O + "q", o.q, Cm, 256);
  lin(O + "k", o.k, Cm, 256);
  ln(O + "qNorm", o.qn, 128);
  ln(O + "kNorm", o.kn, 128);
  f(O + "lambdaQ1", o.lq1, 128); f(O + "lambdaK1", o.lk1, 128); f(O + "lambdaQ2", o.lq2, 128); f(O + "lambdaK2", o.lk2, 128);
  lin(O + "left", o.left, Cm, o.O);
  lin(O + "right", o.right, Cm, o.O);
  lin(O + "toPair", o.toPair, o.O * o.O, 2 * Cz, O + "toPairBias");
  const int pf = 4;
  af3::forEachParam(l.pairBlock, pre, Cz, 0, 0, 0, 0, 0, pf, 0, false, false, f, false, false, "/");
}

}  // namespace msap
}  // namespace grad
