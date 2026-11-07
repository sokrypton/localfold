// ESMFold2's diffusion module for training: one denoiser call D(x, t) into every parameter and the trunk's pair and
// single inputs. cuda/ef2/src/diffusion.cuh's makeDenoiser() + denoise() in their float32 arm (FAST off, ATOM_BF16 off)
// are what the forward is held to:
//
//   pair    = 2 transitions(LN([z_trunk | relpos]) Wz)              ; each token block's bias = LN_b(pair) Wbias_b
//   single  = 2 transitions(LN(s_inputs) Ws + LN(cos(2 pi (t_noise w + b))) Wnoise)
//   q       = c0 + [x / sqrt(t^2 + sigma^2) | 0] Wcoords;  q = encoder SWA stack (c0)
//   a       = pool(relu(q WtoToken)) + LN(single; stepNorm) WsingleToToken
//   12 token blocks: a += sigmoid(single Wog + bog) attention(adaLN(a)), a += sigmoid(...) swiglu(adaLN(a))
//             adaLN(a) = sigmoid(sn Wg + bg) LN(a) + sn Wsh,  sn = LN(single) * singleScale (each adaLN its own)
//   q      += gather(LN(a; tokenNorm) WtokenToAtom) live;  q = decoder SWA stack (c0);  r = LN(q; norm) Wout
//   D       = sigma^2 / (sigma^2 + t^2) x + sigma t / sqrt(sigma^2 + t^2) r
// Every token block's input is kept and the block replayed before its backward; the SWA stacks are ef2atom.cuh's.
#pragma once
#include "ef2atom.cuh"

namespace grad {
namespace ef2 {

// ESMFold2's relative position one-hot (139 columns): offset 66 | token offset 66 | same entity | chain 6
__global__ void relOneHotK(const int* ri, const int* asym, const int* sym, const int* ent, const int* ti, real* out, int T) {
  const int rb = 32, cb = 2, w = 2 * rb + 2, W = 2 * w + 1 + 2 * cb + 2;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * W) return;
  int c = t % W; size_t ij = t / W; int i = ij / T, j = ij % T;
  auto clip = [](int v, int hi) { return v < 0 ? 0 : v > hi ? hi : v; };
  bool sameChain = asym[i] == asym[j], sameRes = ri[i] == ri[j];
  int b0 = sameChain ? clip(ri[i] - ri[j] + rb, 2 * rb) : 2 * rb + 1;
  int b1 = sameChain && sameRes ? clip(ti[i] - ti[j] + rb, 2 * rb) : 2 * rb + 1;
  int b3 = sameChain ? 2 * cb + 1 : clip(sym[i] - sym[j] + cb, 2 * cb);
  out[t] = (c == b0 || c == w + b1 || c == 2 * w + 1 + b3 || (c == 2 * w && ent[i] == ent[j])) ? (real)1 : 0;
}
// [a | b] rows (widths wa, wb), and back
__global__ void catK(const real* a, int wa, const real* b, int wb, real* out, size_t rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int W = wa + wb;
  if (t >= rows * W) return;
  size_t r = t / W; int c = t % W;
  out[t] = c < wa ? a[r * wa + c] : b[r * wb + c - wa];
}
__global__ void splitK(const real* d, int wa, int wb, real* da, real* db, size_t rows) {     // (added; either null)
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int W = wa + wb;
  if (t >= rows * W) return;
  size_t r = t / W; int c = t % W;
  if (c < wa) { if (da) da[r * wa + c] += d[t]; } else if (db) db[r * wb + c - wa] += d[t];
}
__global__ void fourierK(const real* w, const real* b, real tn, real* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cos(2 * (real)3.14159265358979323846 * (tn * w[i] + b[i]));
}
__global__ void fourierBwdK(const real* w, const real* b, real tn, const real* d, real* dw, real* db, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  real g = -d[i] * sin(2 * (real)3.14159265358979323846 * (tn * w[i] + b[i])) * 2 * (real)3.14159265358979323846;
  dw[i] += g * tn; db[i] += g;
}
__global__ void addRowBroadcastK(real* x, const real* v, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] += v[t % C];
}
__global__ void mulVecK(const real* x, const real* v, real* y, size_t rows, int C) {     // y = x * v[c]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) y[t] = x[t] * v[t % C];
}
__global__ void mulVecBwdK(const real* x, const real* v, const real* dy, real* dx, real* dv, size_t rows, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;      // by channel: dv[c] = sum_r dy x; dx = dy v (added)
  if (c >= C) return;
  real s = 0;
  for (size_t r = 0; r < rows; ++r) { s += dy[r * C + c] * x[r * C + c]; dx[r * C + c] += dy[r * C + c] * v[c]; }
  dv[c] += s;
}
// y = sigmoid(g + gb) * x (+ s);  the adaLN combine and the out gate, and their backward
__global__ void sigGateK(const real* g, const real* gb, const real* x, const real* s, real* y, size_t rows, int C, bool add) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  real v = x[t] / (1 + exp(-(g[t] + gb[t % C]))) + (s ? s[t] : (real)0);
  y[t] = add ? y[t] + v : v;
}
__global__ void sigGateBwdK(const real* g, const real* gb, const real* x, const real* dy, real* dg, real* dx, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;      // dg written, dx (added)
  if (t >= rows * C) return;
  real s = 1 / (1 + exp(-(g[t] + gb[t % C])));
  dg[t] = dy[t] * x[t] * s * (1 - s);
  dx[t] += dy[t] * s;
}
// mean over each token's valid atoms (CSR), and back
__global__ void poolK(const real* v, const int* start, const int* atoms, real* out, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int tok = t / C, c = t % C, a0 = start[tok], a1 = start[tok + 1];
  real s = 0;
  for (int k = a0; k < a1; ++k) s += v[(size_t)atoms[k] * C + c];
  out[t] = a1 > a0 ? s / (a1 - a0) : (real)0;
}
__global__ void poolBwdK(const real* d, const int* start, const int* atoms, real* dv, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int tok = t / C, c = t % C, a0 = start[tok], a1 = start[tok + 1];
  for (int k = a0; k < a1; ++k) dv[(size_t)atoms[k] * C + c] = d[t] / (a1 - a0);
}
// q[a] += pt[token(a)] * live, and back (by token: its atoms)
__global__ void gatherTokK(const real* pt, const int* toTok, const real* mask, real* q, int A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * C) return;
  size_t a = t / C;
  q[t] += pt[(size_t)toTok[a] * C + t % C] * mask[a];
}
__global__ void gatherTokBwdK(const real* dq, const int* start, const int* atoms, real* dpt, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int tok = t / C, c = t % C;
  real s = 0;
  for (int k = start[tok]; k < start[tok + 1]; ++k) s += dq[(size_t)atoms[k] * C + c];   // (an invalid atom's is masked)
  dpt[t] = s;
}
__global__ void coordsInK(const real* x, real inv, real* r6, int A) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < A * 6) r6[t] = t % 6 < 3 ? x[(t / 6) * 3 + t % 6] * inv : (real)0;
}
__global__ void coordsInBwdK(const real* dr6, real inv, real* dx, int A) {   // (r6 = [x inv | 0]; dx added)
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < A * 3) dx[t] += dr6[(t / 3) * 6 + t % 3] * inv;
}
__global__ void edmK(const real* x, const real* r, real keep, real take, real* out, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = keep * x[t] + take * r[t];
}
// [P, H] -> [H, T, T], and back
__global__ void toHeadMajorK(const real* pb, real* out, size_t P, int H, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= P * H) return;
  size_t p = t / H; int h = t % H;
  if (back) out[t] = pb[(size_t)h * P + p];     // (pb is [H, P] here, out [P, H])
  else out[(size_t)h * P + p] = pb[t];
}

// ---------------------------------------------------------------- the model
struct Transition { LayerNorm ln; Linear a, b, out; int C = 0, Hd = 0; };
struct AdaLN { Param scale; Linear gate, shift; Param gateBias; };
struct TokenBlock {
  LayerNorm pairLn; Linear pairBias;
  AdaLN attnAda, trAda;
  Linear q, kv, gate, out, swish, trOut, attnOg, trOg;
  Param attnOgB, trOgB;
};
struct Ef2Denoiser {
  int T = 0, A = 0, Cz = 0, Ct = 0, H = 0, Si = 0, nf = 0, Hd = 0, Ca = 128;
  real sigma = 16;
  LayerNorm zInLn, sInLn, noiseLn, stepLn, tokenLn, encLn, decLn;
  Linear zProj, relPos, sProj, noiseProj, singleToToken, encLinear, coords, toToken, tokenToAtom, outLin;
  Transition zT[2], sT[2];
  Param fw, fb;
  std::vector<TokenBlock> blocks;
  SwaStack enc, dec;
};
inline void forEachParam(Ef2Denoiser& m, int nb, int atomBlocks, int atomHidden, int pairHd, int singleHd,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  auto ln = [&](const std::string& n, LayerNorm& l, int C, const std::string& s = "/scale", const std::string& o = "/offset") {
    l.C = C; f(n + s, l.g, C); f(n + o, l.b, C);
  };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; f(n, l.w, (size_t)K * N); };
  auto tr = [&](const std::string& n, Transition& t, int C, int Hd) {
    t.C = C; t.Hd = Hd; ln(n + "norm", t.ln, C); lin(n + "aProjection", t.a, C, Hd); lin(n + "bProjection", t.b, C, Hd);
    lin(n + "outProjection", t.out, Hd, C);
  };
  const std::string D = "f/diffusion/", E = "f/diffusionAtomEncoder", Dd = "f/diffusionAtomDecoder";
  const int Cz = m.Cz, Ct = m.Ct;
  ln(D + "zInputNorm", m.zInLn, 2 * Cz); lin(D + "zProjection", m.zProj, 2 * Cz, Cz);
  lin("f/featuriser/relPos", m.relPos, 139, Cz);
  for (int l = 0; l < 2; ++l) tr(D + "zTransitions/" + std::to_string(l) + "/", m.zT[l], Cz, pairHd);
  ln(D + "sInputNorm", m.sInLn, m.Si); lin(D + "sProjection", m.sProj, m.Si, Ct);
  f(D + "fourier/weights", m.fw, m.nf); f(D + "fourier/offsets", m.fb, m.nf);
  ln(D + "noiseNorm", m.noiseLn, m.nf); lin(D + "noiseProjection", m.noiseProj, m.nf, Ct);
  for (int l = 0; l < 2; ++l) tr(D + "sTransitions/" + std::to_string(l) + "/", m.sT[l], Ct, singleHd);
  ln(D + "stepNorm", m.stepLn, Ct); lin(D + "singleToToken", m.singleToToken, Ct, Ct);
  m.blocks.assign(nb, TokenBlock{});
  for (int b = 0; b < nb; ++b) {
    const std::string B = D + "tokenBlocks/" + std::to_string(b) + "/";
    TokenBlock& k = m.blocks[b];
    ln(B + "attention/pair", k.pairLn, Cz, "NormScale", "NormOffset");
    lin(B + "attention/pairBiasWeights", k.pairBias, Cz, m.H);
    auto ada = [&](const std::string& n, AdaLN& a) {
      f(n + "singleScale", a.scale, Ct); lin(n + "gateWeights", a.gate, Ct, Ct); f(n + "gateBias", a.gateBias, Ct);
      lin(n + "shiftWeights", a.shift, Ct, Ct);
    };
    ada(B + "attention/adaln/", k.attnAda); ada(B + "transition/adaln/", k.trAda);
    lin(B + "attention/queryWeights", k.q, Ct, Ct); k.q.hasBias = true; f(B + "attention/queryBias", k.q.b, Ct);
    lin(B + "attention/kvWeights", k.kv, Ct, 2 * Ct); lin(B + "attention/gateWeights", k.gate, Ct, Ct);
    lin(B + "attention/outWeights", k.out, Ct, Ct);
    lin(B + "attention/outGateWeights", k.attnOg, Ct, Ct); f(B + "attention/outGateBias", k.attnOgB, Ct);
    lin(B + "transition/swishWeights", k.swish, Ct, 2 * m.Hd); lin(B + "transition/outWeights", k.trOut, m.Hd, Ct);
    lin(B + "transition/outGateWeights", k.trOg, Ct, Ct); f(B + "transition/outGateBias", k.trOgB, Ct);
  }
  ln(D + "tokenNorm", m.tokenLn, Ct);
  lin(E + "/linear", m.encLinear, 389, m.Ca); ln(E + "/norm", m.encLn, m.Ca);
  lin(E + "/coordsLinear", m.coords, 6, m.Ca); lin(E + "/toToken", m.toToken, m.Ca, Ct);
  m.enc.C = m.dec.C = m.Ca; m.enc.hidden = m.dec.hidden = atomHidden; m.enc.hw = m.dec.hw = 64;
  forEachParam(m.enc, E, atomBlocks, f);
  lin(Dd + "/tokenToAtom", m.tokenToAtom, Ct, m.Ca);
  forEachParam(m.dec, Dd, atomBlocks, f);
  ln(Dd + "/norm", m.decLn, m.Ca); lin(Dd + "/outputLinear", m.outLin, m.Ca, 3);
}

struct TrSave { LnSave ln; real *xn, *a, *b; };
inline TrSave trForward(Transition& t, const real* x, real* y, size_t rows) {   // y = x + out(silu(a LN x) * b LN x)
  TrSave s{};
  GK(cudaMemcpyAsync(y, x, rows * t.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.xn = ARENA.get(rows * t.C); s.ln = lnForward(t.ln, x, s.xn, rows);
  s.a = ARENA.get(rows * t.Hd); t.a.forward(s.xn, s.a, rows);
  s.b = ARENA.get(rows * t.Hd); t.b.forward(s.xn, s.b, rows);
  real* g = ARENA.get(rows * 2 * t.Hd);         // [silu(a) | b] -> swiglu's layout
  catK<<<blocks(rows * 2 * t.Hd), 256, 0, STREAM>>>(s.a, t.Hd, s.b, t.Hd, g, rows);
  real* h = ARENA.get(rows * t.Hd); swigluK<<<blocks(rows * t.Hd), 256, 0, STREAM>>>(g, h, rows, t.Hd);
  t.out.forward(h, y, rows, 1);
  return s;
}
// dy -> dx (added: the branch; dx may be dy itself, whose residual it then already holds)
inline void trBackward(Transition& t, const TrSave& s, const real* dy, real* dx, size_t rows) {
  Scratch scratch;
  real* g = ARENA.get(rows * 2 * t.Hd);
  catK<<<blocks(rows * 2 * t.Hd), 256, 0, STREAM>>>(s.a, t.Hd, s.b, t.Hd, g, rows);
  real* h = ARENA.get(rows * t.Hd); swigluK<<<blocks(rows * t.Hd), 256, 0, STREAM>>>(g, h, rows, t.Hd);
  real* dh = ARENA.get(rows * t.Hd); t.out.backward(h, dy, dh, rows);
  real* dg = ARENA.get(rows * 2 * t.Hd); swigluBwdK<<<blocks(rows * t.Hd), 256, 0, STREAM>>>(g, dh, dg, rows, t.Hd);
  real *da = ARENA.zeros(rows * t.Hd), *db = ARENA.zeros(rows * t.Hd);
  splitK<<<blocks(rows * 2 * t.Hd), 256, 0, STREAM>>>(dg, t.Hd, t.Hd, da, db, rows);
  real* dxn = ARENA.get(rows * t.C); t.a.backward(s.xn, da, dxn, rows); t.b.backward(s.xn, db, dxn, rows, true);
  lnBackward(t.ln, s.ln, dxn, dx, rows, true);
}
// one token block's saves
struct TokSave {
  real *sna, *snt, *ga, *sha, *gt, *sht, *ogA, *ogT, *an, *x, *q, *kv, *g, *ctx, *cg, *o, *a1, *an2, *x2, *w, *sg, *o2;
  LnSave l1, l2;
  real* lse = nullptr;   // the attention's logit max and log-sum, kept for its backward
};
struct TokenCtx { const real* single; const real* sn; const real* bias; int T; };     // sn: LN(single), affine-free
inline Sdpa tokenSdpa(const Ef2Denoiser& m, const real* bias, real* grad) {
  Sdpa a; a.B = 1; a.H = m.H; a.Q = a.K = m.T; a.D = m.Ct / m.H; a.scale = 1 / sqrt((real)a.D);
  a.bias[0] = Bias{bias, 0, (long long)m.T * m.T, m.T, grad};
  return a;
}
inline TokSave tokenForward(Ef2Denoiser& m, TokenBlock& k, const TokenCtx& c, const real* a, real* out) {
  TokSave s{};
  const int T = c.T, C = m.Ct; const size_t TC = (size_t)T * C;
  LayerNorm plain; plain.C = C;
  auto ada = [&](AdaLN& ad, real*& sn, real*& g, real*& sh) {
    sn = ARENA.get(TC); mulVecK<<<blocks(TC), 256, 0, STREAM>>>(c.sn, ad.scale.w, sn, T, C);
    g = ARENA.get(TC); ad.gate.forward(sn, g, T);
    sh = ARENA.get(TC); ad.shift.forward(sn, sh, T);
  };
  ada(k.attnAda, s.sna, s.ga, s.sha); ada(k.trAda, s.snt, s.gt, s.sht);
  s.ogA = ARENA.get(TC); k.attnOg.forward(c.single, s.ogA, T);
  s.ogT = ARENA.get(TC); k.trOg.forward(c.single, s.ogT, T);
  // attention
  s.an = ARENA.get(TC); s.l1 = lnForward(plain, a, s.an, T);
  s.x = ARENA.get(TC); sigGateK<<<blocks(TC), 256, 0, STREAM>>>(s.ga, k.attnAda.gateBias.w, s.an, s.sha, s.x, T, C, false);
  s.q = ARENA.get(TC); k.q.forward(s.x, s.q, T);
  s.kv = ARENA.get(TC * 2); k.kv.forward(s.x, s.kv, T);
  s.g = ARENA.get(TC); k.gate.forward(s.x, s.g, T);
  s.ctx = ARENA.get(TC);
  { const Sdpa sd_ = tokenSdpa(m, c.bias, nullptr); s.lse = ARENA.get((size_t)2 * sd_.B * sd_.H * sd_.Q); sdpaForward(sd_, s.q, C, s.kv, 2 * C, s.kv + C, 2 * C, s.ctx, C, s.lse); }
  s.cg = ARENA.get(TC); sigGateK<<<blocks(TC), 256, 0, STREAM>>>(s.g, ARENA.zeros(C), s.ctx, nullptr, s.cg, T, C, false);
  s.o = ARENA.get(TC); k.out.forward(s.cg, s.o, T);
  s.a1 = cloneR(a, TC); sigGateK<<<blocks(TC), 256, 0, STREAM>>>(s.ogA, k.attnOgB.w, s.o, nullptr, s.a1, T, C, true);
  // transition
  s.an2 = ARENA.get(TC); s.l2 = lnForward(plain, s.a1, s.an2, T);
  s.x2 = ARENA.get(TC); sigGateK<<<blocks(TC), 256, 0, STREAM>>>(s.gt, k.trAda.gateBias.w, s.an2, s.sht, s.x2, T, C, false);
  s.w = ARENA.get((size_t)T * 2 * m.Hd); k.swish.forward(s.x2, s.w, T);
  s.sg = ARENA.get((size_t)T * m.Hd); swigluK<<<blocks((size_t)T * m.Hd), 256, 0, STREAM>>>(s.w, s.sg, T, m.Hd);
  s.o2 = ARENA.get(TC); k.trOut.forward(s.sg, s.o2, T);
  GK(cudaMemcpyAsync(out, s.a1, TC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  sigGateK<<<blocks(TC), 256, 0, STREAM>>>(s.ogT, k.trOgB.w, s.o2, nullptr, out, T, C, true);
  return s;
}
// dout -> da (written), dsingle, dsn (added), dbias (added: [H, T, T])
inline void tokenBackward(Ef2Denoiser& m, TokenBlock& k, const TokenCtx& c, const real* a, const TokSave& s, const real* dout,
                          real* da, real* dsingle, real* dsn, real* dbias) {
  Scratch scratch;
  const int T = c.T, C = m.Ct; const size_t TC = (size_t)T * C;
  LayerNorm plain; plain.C = C;
  real* zeroB = ARENA.zeros(C);
  // out = a1 + sig(ogT + b) o2
  real* da1 = cloneR(dout, TC);
  real *dogT = ARENA.get(TC), *do2 = ARENA.zeros(TC);
  sigGateBwdK<<<blocks(TC), 256, 0, STREAM>>>(s.ogT, k.trOgB.w, s.o2, dout, dogT, do2, T, C);
  colSum(k.trOgB.g, dogT, T, C);
  k.trOg.backward(c.single, dogT, dsingle, T, true);
  real* dsg = ARENA.get((size_t)T * m.Hd); k.trOut.backward(s.sg, do2, dsg, T);
  real* dw = ARENA.get((size_t)T * 2 * m.Hd); swigluBwdK<<<blocks((size_t)T * m.Hd), 256, 0, STREAM>>>(s.w, dsg, dw, T, m.Hd);
  real* dx2 = ARENA.get(TC); k.swish.backward(s.x2, dw, dx2, T);
  // x2 = sig(gt + bg) an2 + sht
  real *dgt = ARENA.get(TC), *dan2 = ARENA.zeros(TC);
  sigGateBwdK<<<blocks(TC), 256, 0, STREAM>>>(s.gt, k.trAda.gateBias.w, s.an2, dx2, dgt, dan2, T, C);
  colSum(k.trAda.gateBias.g, dgt, T, C);
  real* dsnt = ARENA.get(TC); k.trAda.gate.backward(s.snt, dgt, dsnt, T); k.trAda.shift.backward(s.snt, dx2, dsnt, T, true);
  mulVecBwdK<<<blocks(C, 64), 64, 0, STREAM>>>(c.sn, k.trAda.scale.w, dsnt, dsn, k.trAda.scale.g, T, C);
  lnBackward(plain, s.l2, dan2, da1, T, true);
  // a1 = a + sig(ogA + b) o
  real* dA = cloneR(da1, TC);
  real *dogA = ARENA.get(TC), *dO = ARENA.zeros(TC);
  sigGateBwdK<<<blocks(TC), 256, 0, STREAM>>>(s.ogA, k.attnOgB.w, s.o, da1, dogA, dO, T, C);
  colSum(k.attnOgB.g, dogA, T, C);
  k.attnOg.backward(c.single, dogA, dsingle, T, true);
  real* dcg = ARENA.get(TC); k.out.backward(s.cg, dO, dcg, T);
  real *dg = ARENA.get(TC), *dctx = ARENA.zeros(TC);
  sigGateBwdK<<<blocks(TC), 256, 0, STREAM>>>(s.g, zeroB, s.ctx, dcg, dg, dctx, T, C);
  real *dq = ARENA.get(TC), *dkv = ARENA.get(TC * 2);
  sdpaBackward(tokenSdpa(m, c.bias, dbias), s.q, C, s.kv, 2 * C, s.kv + C, 2 * C, dctx, C, dq, C, dkv, 2 * C, dkv + C, 2 * C, false, false, false, s.ctx, C, s.lse);
  real* dx = ARENA.get(TC);
  k.q.backward(s.x, dq, dx, T); k.kv.backward(s.x, dkv, dx, T, true); k.gate.backward(s.x, dg, dx, T, true);
  real *dga = ARENA.get(TC), *dan = ARENA.zeros(TC);
  sigGateBwdK<<<blocks(TC), 256, 0, STREAM>>>(s.ga, k.attnAda.gateBias.w, s.an, dx, dga, dan, T, C);
  colSum(k.attnAda.gateBias.g, dga, T, C);
  real* dsna = ARENA.get(TC); k.attnAda.gate.backward(s.sna, dga, dsna, T); k.attnAda.shift.backward(s.sna, dx, dsna, T, true);
  mulVecBwdK<<<blocks(C, 64), 64, 0, STREAM>>>(c.sn, k.attnAda.scale.w, dsna, dsn, k.attnAda.scale.g, T, C);
  lnBackward(plain, s.l1, dan, dA, T, true);
  GK(cudaMemcpyAsync(da, dA, TC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

// the batch's atoms and tokens
struct Ef2Batch {
  AtomLayout L;
  const real* feat;                     // [A, 389] atom features
  const int *tokStart, *tokAtoms, *atomToTok;
  const int *ri, *asym, *sym, *ent, *ti;
  int T;
};
struct Ef2Save {
  real *rel, *joined, *jn, *pair0, *pair, *biasPB, *pn;
  std::vector<real*> biases;            // [H, T, T] each block's
  std::vector<LnSave> pairLns; std::vector<real*> pairNs;
  TrSave zt[2], st[2];
  LnSave lzin, lsin, lnoise, lstep, ltok, lenc, ldec, lsn;
  real *sinN, *sBase, *four, *fourN, *noise, *single, *sn, *snStep;
  real *encPre, *c0, *r6, *q0, *q, *tokPre, *tok, *pooled, *a0, *aOut, *aN, *pt, *qd0, *qd, *qN, *r;
  std::vector<real*> as;
  SwaRun encRun, decRun;
  real tn, inv, keep, take, t;
};
inline Ef2Save ef2Forward(Ef2Denoiser& m, const Ef2Batch& b, const real* zTrunk, const real* sInputs, const real* x, real t, real* out) {
  Ef2Save s{};
  const int T = b.T, A = b.L.A, Cz = m.Cz, Ct = m.Ct, Ca = m.Ca, nb = (int)m.blocks.size();
  const size_t P = (size_t)T * T, TC = (size_t)T * Ct, AC = (size_t)A * Ca;
  const real s2 = m.sigma * m.sigma, t2 = t * t;
  s.t = t; s.tn = (real)0.25 * log(fmax(t / m.sigma, (real)1e-20)); s.inv = 1 / sqrt(t2 + s2);
  s.keep = s2 / (s2 + t2); s.take = m.sigma * t / sqrt(s2 + t2);
  // the conditioning pair and every block's bias
  s.rel = ARENA.get(P * 139);
  relOneHotK<<<blocks(P * 139), 256, 0, STREAM>>>(b.ri, b.asym, b.sym, b.ent, b.ti, s.rel, T);
  real* relProj = ARENA.get(P * Cz); m.relPos.forward(s.rel, relProj, P);
  s.joined = ARENA.get(P * 2 * Cz); catK<<<blocks(P * 2 * Cz), 256, 0, STREAM>>>(zTrunk, Cz, relProj, Cz, s.joined, P);
  s.jn = ARENA.get(P * 2 * Cz); s.lzin = lnForward(m.zInLn, s.joined, s.jn, P);
  s.pair0 = ARENA.get(P * Cz); m.zProj.forward(s.jn, s.pair0, P);
  real* p1 = ARENA.get(P * Cz); s.pair = ARENA.get(P * Cz);
  s.zt[0] = trForward(m.zT[0], s.pair0, p1, P); s.zt[1] = trForward(m.zT[1], p1, s.pair, P);
  s.biases.assign(nb, nullptr); s.pairLns.assign(nb, LnSave{}); s.pairNs.assign(nb, nullptr);
  for (int k = 0; k < nb; ++k) {
    s.pairNs[k] = ARENA.get(P * Cz); s.pairLns[k] = lnForward(m.blocks[k].pairLn, s.pair, s.pairNs[k], P);
    real* pb = ARENA.get(P * m.H); m.blocks[k].pairBias.forward(s.pairNs[k], pb, P);
    s.biases[k] = ARENA.get(P * m.H); toHeadMajorK<<<blocks(P * m.H), 256, 0, STREAM>>>(pb, s.biases[k], P, m.H, false);
  }
  // the single
  s.sinN = ARENA.get((size_t)T * m.Si); s.lsin = lnForward(m.sInLn, sInputs, s.sinN, T);
  real* single0 = ARENA.get(TC); m.sProj.forward(s.sinN, single0, T);
  s.four = ARENA.get(m.nf); fourierK<<<blocks(m.nf), 256, 0, STREAM>>>(m.fw.w, m.fb.w, s.tn, s.four, m.nf);
  s.fourN = ARENA.get(m.nf); s.lnoise = lnForward(m.noiseLn, s.four, s.fourN, 1);
  s.noise = ARENA.get(Ct); m.noiseProj.forward(s.fourN, s.noise, 1);
  addRowBroadcastK<<<blocks(TC), 256, 0, STREAM>>>(single0, s.noise, T, Ct);
  real* single1 = ARENA.get(TC); s.single = ARENA.get(TC);
  s.st[0] = trForward(m.sT[0], single0, single1, T); s.st[1] = trForward(m.sT[1], single1, s.single, T);
  LayerNorm plain; plain.C = Ct;
  s.sn = ARENA.get(TC); s.lsn = lnForward(plain, s.single, s.sn, T);
  // the encoder
  s.encPre = ARENA.get(AC); m.encLinear.forward(b.feat, s.encPre, A);
  s.c0 = ARENA.get(AC); s.lenc = lnForward(m.encLn, s.encPre, s.c0, A);
  s.r6 = ARENA.get((size_t)A * 6); coordsInK<<<blocks((size_t)A * 6), 256, 0, STREAM>>>(x, s.inv, s.r6, A);
  s.q0 = cloneR(s.c0, AC); m.coords.forward(s.r6, s.q0, A, 1);
  s.q = ARENA.get(AC); swaStackForward(m.enc, b.L, s.c0, s.q0, s.q, s.encRun);
  s.tokPre = ARENA.get((size_t)A * Ct); m.toToken.forward(s.q, s.tokPre, A);
  s.tok = ARENA.get((size_t)A * Ct); reluK<<<blocks((size_t)A * Ct), 256, 0, STREAM>>>(s.tok, s.tokPre, (size_t)A * Ct);
  s.a0 = ARENA.get(TC); poolK<<<blocks(TC), 256, 0, STREAM>>>(s.tok, b.tokStart, b.tokAtoms, s.a0, T, Ct);
  s.snStep = ARENA.get(TC); s.lstep = lnForward(m.stepLn, s.single, s.snStep, T);
  m.singleToToken.forward(s.snStep, s.a0, T, 1);
  // the token blocks
  s.as.assign(nb + 1, nullptr);
  s.as[0] = s.a0;
  TokenCtx c{s.single, s.sn, nullptr, T};
  for (int k = 0; k < nb; ++k) {
    s.as[k + 1] = ARENA.get(TC);
    Scratch sc;
    c.bias = s.biases[k];
    tokenForward(m, m.blocks[k], c, s.as[k], s.as[k + 1]);
  }
  s.aN = ARENA.get(TC); s.ltok = lnForward(m.tokenLn, s.as[nb], s.aN, T);
  s.pt = ARENA.get((size_t)T * Ca); m.tokenToAtom.forward(s.aN, s.pt, T);
  s.qd0 = cloneR(s.q, AC); gatherTokK<<<blocks(AC), 256, 0, STREAM>>>(s.pt, b.atomToTok, b.L.mask, s.qd0, A, Ca);
  s.qd = ARENA.get(AC); swaStackForward(m.dec, b.L, s.c0, s.qd0, s.qd, s.decRun);
  s.qN = ARENA.get(AC); s.ldec = lnForward(m.decLn, s.qd, s.qN, A);
  s.r = ARENA.get((size_t)A * 3); m.outLin.forward(s.qN, s.r, A);
  edmK<<<blocks((size_t)A * 3), 256, 0, STREAM>>>(x, s.r, s.keep, s.take, out, A * 3);
  return s;
}
// dout [A, 3] -> every parameter; dzTrunk, dsInputs, dx (added; any may be null)
inline void ef2Backward(Ef2Denoiser& m, const Ef2Batch& b, const Ef2Save& s, const real* zTrunk, const real* sInputs,
                        const real* dout, real* dzTrunk, real* dsInputs, real* dx) {
  Scratch scratch;
  const int T = b.T, A = b.L.A, Cz = m.Cz, Ct = m.Ct, Ca = m.Ca, nb = (int)m.blocks.size();
  const size_t P = (size_t)T * T, TC = (size_t)T * Ct, AC = (size_t)A * Ca;
  LayerNorm plain; plain.C = Ct;
  real* dr = ARENA.get((size_t)A * 3);
  GK(cudaMemcpyAsync(dr, dout, (size_t)A * 3 * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  scaleK<<<blocks((size_t)A * 3), 256, 0, STREAM>>>(dr, s.take, (size_t)A * 3);
  if (dx) { real* t = cloneR(dout, (size_t)A * 3); scaleK<<<blocks((size_t)A * 3), 256, 0, STREAM>>>(t, s.keep, (size_t)A * 3); add(dx, t, (size_t)A * 3); }
  real* dqN = ARENA.get(AC); m.outLin.backward(s.qN, dr, dqN, A);
  real* dqd = ARENA.get(AC); lnBackward(m.decLn, s.ldec, dqN, dqd, A, false);
  real *dqd0 = ARENA.get(AC), *dc0 = ARENA.zeros(AC);
  swaStackBackward(m.dec, b.L, s.c0, s.decRun, dqd, dqd0, dc0);
  // qd0 = q + gather(pt) live
  real* dq = cloneR(dqd0, AC);
  real* dqm = ARENA.get(AC); GK(cudaMemcpyAsync(dqm, dqd0, AC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  real* dpt = ARENA.get((size_t)T * Ca);
  gatherTokBwdK<<<blocks((size_t)T * Ca), 256, 0, STREAM>>>(dqm, b.tokStart, b.tokAtoms, dpt, T, Ca);
  real* daN = ARENA.get(TC); m.tokenToAtom.backward(s.aN, dpt, daN, T);
  real* da = ARENA.get(TC); lnBackward(m.tokenLn, s.ltok, daN, da, T, false);
  // the token blocks, last first; their biases' gradients kept to fold back through the pair
  real *dsingle = ARENA.zeros(TC), *dsn = ARENA.zeros(TC);
  std::vector<real*> dbias(nb);
  for (int k = 0; k < nb; ++k) dbias[k] = ARENA.zeros(P * m.H);
  TokenCtx c{s.single, s.sn, nullptr, T};
  for (int k = nb; k-- > 0;) {
    Scratch sc;
    c.bias = s.biases[k];
    real* o = ARENA.get(TC);
    TokSave ts = tokenForward(m, m.blocks[k], c, s.as[k], o);
    real* dIn = ARENA.get(TC);
    tokenBackward(m, m.blocks[k], c, s.as[k], ts, da, dIn, dsingle, dsn, dbias[k]);
    GK(cudaMemcpyAsync(da, dIn, TC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  // a0 = pool(relu(q WtoToken)) + LN(single; step) WsingleToToken
  real* dsnStep = ARENA.get(TC); m.singleToToken.backward(s.snStep, da, dsnStep, T);
  lnBackward(m.stepLn, s.lstep, dsnStep, dsingle, T, true);
  real* dtok = ARENA.zeros((size_t)A * Ct);
  poolBwdK<<<blocks(TC), 256, 0, STREAM>>>(da, b.tokStart, b.tokAtoms, dtok, T, Ct);
  reluBwdK<<<blocks((size_t)A * Ct), 256, 0, STREAM>>>(dtok, s.tokPre, (size_t)A * Ct);
  m.toToken.backward(s.q, dtok, dq, A, true);
  real* dq0 = ARENA.get(AC);
  swaStackBackward(m.enc, b.L, s.c0, s.encRun, dq, dq0, dc0);
  add(dc0, dq0, AC);                                   // q0 = c0 + r6 Wcoords
  real* dr6 = dx ? ARENA.get((size_t)A * 6) : nullptr;
  m.coords.backward(s.r6, dq0, dr6, A);
  if (dx) coordsInBwdK<<<blocks((size_t)A * 3), 256, 0, STREAM>>>(dr6, s.inv, dx, A);
  real* dencPre = ARENA.get(AC); lnBackward(m.encLn, s.lenc, dc0, dencPre, A, false);
  m.encLinear.backward(b.feat, dencPre, nullptr, A);
  // the single: sn = LN(single) (the adaLNs'), the transitions, the noise, the inputs
  lnBackward(plain, s.lsn, dsn, dsingle, T, true);
  for (int l = 2; l-- > 0;) trBackward(m.sT[l], s.st[l], dsingle, dsingle, T);
  real* dnoise = ARENA.zeros(Ct); colSum(dnoise, dsingle, T, Ct);
  real* dfourN = ARENA.get(m.nf); m.noiseProj.backward(s.fourN, dnoise, dfourN, 1);
  real* dfour = ARENA.get(m.nf); lnBackward(m.noiseLn, s.lnoise, dfourN, dfour, 1, false);
  fourierBwdK<<<blocks(m.nf), 256, 0, STREAM>>>(m.fw.w, m.fb.w, s.tn, dfour, m.fw.g, m.fb.g, m.nf);
  real* dsinN = ARENA.get((size_t)T * m.Si); m.sProj.backward(s.sinN, dsingle, dsinN, T);
  if (dsInputs) lnBackward(m.sInLn, s.lsin, dsinN, dsInputs, T, true);
  else { real* t = ARENA.get((size_t)T * m.Si); lnBackward(m.sInLn, s.lsin, dsinN, t, T, false); }
  // the pair: every block's bias, the transitions, the projection, [z | relpos]
  real* dpair = ARENA.zeros(P * Cz);
  for (int k = 0; k < nb; ++k) {
    real* dpb = ARENA.get(P * m.H); toHeadMajorK<<<blocks(P * m.H), 256, 0, STREAM>>>(dbias[k], dpb, P, m.H, true);
    real* dpn = ARENA.get(P * Cz); m.blocks[k].pairBias.backward(s.pairNs[k], dpb, dpn, P);
    lnBackward(m.blocks[k].pairLn, s.pairLns[k], dpn, dpair, P, true);
  }
  for (int l = 2; l-- > 0;) trBackward(m.zT[l], s.zt[l], dpair, dpair, P);
  real* djn = ARENA.get(P * 2 * Cz); m.zProj.backward(s.jn, dpair, djn, P);
  real* dj = ARENA.get(P * 2 * Cz); lnBackward(m.zInLn, s.lzin, djn, dj, P, false);
  real* drel = ARENA.zeros(P * Cz);
  splitK<<<blocks(P * 2 * Cz), 256, 0, STREAM>>>(dj, Cz, Cz, dzTrunk, drel, P);
  m.relPos.backward(s.rel, drel, nullptr, P);
}

}  // namespace ef2
}  // namespace grad
