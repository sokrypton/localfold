// ESMFold2's sliding-window atom transformer with 3D RoPE, for training: the atom embedding c0 and the adaLN-Zero SWA
// block stacks the inputs embedder and the diffusion module's encoder and decoder share. cuda/ef2/src/atoms.cuh's
// prepareAtoms() and swaStack() are what the forward is held to - in its float32 arm (ATOM_BF16 off: the module
// narrows q, k, v to bfloat16, which a training graph does not).
//
//   c0 = LN(atomFeatures W)
//   block: mod = silu(c0) Wadaln -> shift_a scale_a gate_a shift_f scale_f gate_f
//          x += gate_a * (ctx * sigmoid(xm Wgate) * live) Wout,   xm = rms(x) (1 + scale_a) + shift_a
//          ctx = window attention over valid atoms by rank, q and k rms-normed per head (32) then rotated
//          x += gate_f * swiglu(xm2 Wup) Wdown,                    xm2 = rms(x) (1 + scale_f) + shift_f
// Every rms affine-free, eps 1.19e-7. Each block's input is kept and the block replayed before its backward.
#pragma once
#include "ops.cuh"

namespace grad {
namespace ef2 {

constexpr real RMS_EPS = (real)1.1920928955078125e-7;
constexpr int HD = 32;

__global__ void siluFwdK(const real* x, real* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = x[t] / (1 + exp(-x[t]));
}
__global__ void siluBwdK(const real* x, const real* dy, real* dx, size_t n) {     // dx += dy silu'(x)
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  real s = 1 / (1 + exp(-x[t]));
  dx[t] += dy[t] * s * (1 + x[t] * (1 - s));
}
// a warp a row: y = rms(x) (1 + mod[scale]) + mod[shift]; xhat and 1/r saved
__global__ void rmsModK(const real* x, const real* mod, int shift, int scale, real* y, real* xhat, real* inv, int rows, int C) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  const real* xr = x + (size_t)row * C;
  const real* m = mod + (size_t)row * 6 * C;
  real s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c] * xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  real iv = 1 / sqrt(s / C + RMS_EPS);
  if (lane == 0) inv[row] = iv;
  for (int c = lane; c < C; c += 32) {
    real h = xr[c] * iv;
    xhat[(size_t)row * C + c] = h;
    y[(size_t)row * C + c] = h * (1 + m[scale * C + c]) + m[shift * C + c];
  }
}
// dy -> dmod (shift, scale; written), dx (added)
__global__ void rmsModBwdK(const real* xhat, const real* inv, const real* mod, int shift, int scale, const real* dy,
                           real* dmod, real* dx, int rows, int C) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  const real* m = mod + (size_t)row * 6 * C;
  real* dm = dmod + (size_t)row * 6 * C;
  real dot = 0;
  for (int c = lane; c < C; c += 32) {
    size_t k = (size_t)row * C + c;
    real g = dy[k], h = xhat[k];
    dm[shift * C + c] = g; dm[scale * C + c] = g * h;
    dot += g * (1 + m[scale * C + c]) * h;
  }
  for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(~0u, dot, o);
  dot /= C;
  for (int c = lane; c < C; c += 32) {
    size_t k = (size_t)row * C + c;
    dx[k] += (dy[k] * (1 + m[scale * C + c]) - xhat[k] * dot) * inv[row];
  }
}
// x[a, c] += mod[a, which, c] d[a, c]; and back: dmod[which] = dx d, dd = dx mod[which]
__global__ void gatedAddK(real* x, const real* mod, const real* d, int A, int C, int which) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * C) return;
  size_t a = t / C; int c = t % C;
  x[t] += mod[a * 6 * C + which * C + c] * d[t];
}
__global__ void gatedAddBwdK(const real* dx, const real* mod, const real* d, real* dmod, real* dd, int A, int C, int which) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * C) return;
  size_t a = t / C; int c = t % C;
  dmod[a * 6 * C + which * C + c] = dx[t] * d[t];
  dd[t] = dx[t] * mod[a * 6 * C + which * C + c];
}
// q and k (parts 0, 1 of qkv [A, 3C]) rms-normed per head of 32 and rotated: out[d] = n[d] c + sigma n[d ^ 16] s,
// sigma -1 below 16 (cos/sin [A, 16], indexed d & 15); a thread a (atom, part, head); the normed n saved
__global__ void qkPrepK(const real* qkv, const real* cosT, const real* sinT, real* out, real* nrm, real* inv, int A, int C) {
  const int heads = C / HD;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * 2 * heads) return;
  int h = t % heads; int part = (t / heads) % 2; size_t a = t / (2 * heads);
  const real* x = qkv + a * 3 * C + part * C + h * HD;
  real s = 0;
  for (int d = 0; d < HD; ++d) s += x[d] * x[d];
  real iv = 1 / sqrt(s / HD + RMS_EPS);
  inv[t] = iv;
  real n[HD];
  for (int d = 0; d < HD; ++d) { n[d] = x[d] * iv; nrm[a * 2 * C + part * C + h * HD + d] = n[d]; }
  real* o = out + a * 3 * C + part * C + h * HD;
  for (int d = 0; d < HD; ++d) {
    real c = cosT[a * 16 + (d & 15)], sn = sinT[a * 16 + (d & 15)];
    o[d] = n[d] * c + (d < 16 ? -1 : 1) * n[d ^ 16] * sn;
  }
}
__global__ void qkPrepBwdK(const real* dout, const real* nrm, const real* inv, const real* cosT, const real* sinT, real* dqkv,
                           int A, int C) {
  const int heads = C / HD;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * 2 * heads) return;
  int h = t % heads; int part = (t / heads) % 2; size_t a = t / (2 * heads);
  const real* g = dout + a * 3 * C + part * C + h * HD;
  const real* n = nrm + a * 2 * C + part * C + h * HD;
  real dn[HD], dot = 0;
  for (int e = 0; e < HD; ++e) {
    real c = cosT[a * 16 + (e & 15)], sn = sinT[a * 16 + (e & 15)];
    dn[e] = g[e] * c + (e < 16 ? 1 : -1) * g[e ^ 16] * sn;
    dot += dn[e] * n[e];
  }
  dot /= HD;
  real* dx = dqkv + a * 3 * C + part * C + h * HD;
  for (int e = 0; e < HD; ++e) dx[e] = (dn[e] - n[e] * dot) * inv[t];
}
// the window attention by valid rank: P [heads, nValid, W] (W = 2 hw + 1), the softmax of scale q.k over the keys
// r - hw .. r + hw that exist; ctx[valid[r]] = sum P v
__global__ void swaScoresK(const real* qkv, const int* valid, int nValid, int hw, real scale, real* P, int C) {
  const int heads = C / HD, W = 2 * hw + 1;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * nValid) return;
  int h = t / nValid, r = t % nValid;
  const real* q = qkv + (size_t)valid[r] * 3 * C + h * HD;
  real* p = P + t * W;
  real mx = -INFINITY;
  for (int o = 0; o < W; ++o) {
    int k = r - hw + o;
    if (k < 0 || k >= nValid) { p[o] = -INFINITY; continue; }
    const real* kk = qkv + (size_t)valid[k] * 3 * C + C + h * HD;
    real s = 0;
    for (int d = 0; d < HD; ++d) s += q[d] * kk[d];
    p[o] = s * scale; mx = fmax(mx, p[o]);
  }
  real sum = 0;
  for (int o = 0; o < W; ++o) { real e = p[o] == -INFINITY ? 0 : exp(p[o] - mx); p[o] = e; sum += e; }
  for (int o = 0; o < W; ++o) p[o] /= sum;
}
// ctx [A, C]: zero on an invalid atom
__global__ void swaApplyK(const real* qkv, const int* valid, int nValid, int hw, const real* P, real* ctx, int C) {
  const int heads = C / HD, W = 2 * hw + 1;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * nValid * HD) return;
  int d = t % HD; size_t hr = t / HD; int h = hr / nValid, r = hr % nValid;
  const real* p = P + hr * W;
  real s = 0;
  for (int o = 0; o < W; ++o) {
    int k = r - hw + o;
    if (k < 0 || k >= nValid) continue;
    s += p[o] * qkv[(size_t)valid[k] * 3 * C + 2 * C + h * HD + d];
  }
  ctx[(size_t)valid[r] * C + h * HD + d] = s;
}
// dP [heads, nValid, W] = dctx[query] . v[key], then the softmax's backward into dS (scaled), in place
__global__ void swaScoresBwdK(const real* qkv, const int* valid, int nValid, int hw, real scale, const real* P,
                              const real* dctx, real* dS, int C) {
  const int heads = C / HD, W = 2 * hw + 1;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * nValid) return;
  int h = t / nValid, r = t % nValid;
  const real* g = dctx + (size_t)valid[r] * C + h * HD;
  const real* p = P + t * W;
  real* ds = dS + t * W;
  real dot = 0;
  for (int o = 0; o < W; ++o) {
    int k = r - hw + o;
    real dp = 0;
    if (k >= 0 && k < nValid) {
      const real* v = qkv + (size_t)valid[k] * 3 * C + 2 * C + h * HD;
      for (int d = 0; d < HD; ++d) dp += g[d] * v[d];
    }
    ds[o] = dp; dot += p[o] * dp;
  }
  for (int o = 0; o < W; ++o) ds[o] = p[o] * (ds[o] - dot) * scale;
}
// dq (by query), dk and dv (by key: the queries k - o .. k + o whose windows hold it) into dqkv (written)
__global__ void swaQKVBwdK(const real* qkv, const int* valid, int nValid, int hw, const real* P, const real* dS,
                           const real* dctx, real* dqkv, int C) {
  const int heads = C / HD, W = 2 * hw + 1;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * nValid * HD) return;
  int d = t % HD; size_t hr = t / HD; int h = hr / nValid, r = hr % nValid;
  size_t at = (size_t)valid[r] * 3 * C + h * HD + d;
  real dq = 0, dk = 0, dv = 0;
  for (int o = 0; o < W; ++o) {
    int k = r - hw + o;                // r as the query: its key k
    if (k >= 0 && k < nValid) dq += dS[hr * W + o] * qkv[(size_t)valid[k] * 3 * C + C + h * HD + d];
    int q = r + hw - o;                // r as the key of query q (whose offset to r is o)
    if (q >= 0 && q < nValid) {
      size_t qr = (size_t)h * nValid + q;
      dk += dS[qr * W + o] * qkv[(size_t)valid[q] * 3 * C + h * HD + d];
      dv += P[qr * W + o] * dctx[(size_t)valid[q] * C + h * HD + d];
    }
  }
  dqkv[at] = dq; dqkv[at + C] = dk; dqkv[at + 2 * C] = dv;
}
// cg = ctx sigmoid(g) live, and back
__global__ void gateLiveFwdK(const real* ctx, const real* g, const real* mask, real* out, int A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * C) return;
  out[t] = mask[t / C] == 0 ? (real)0 : ctx[t] / (1 + exp(-g[t]));
}
__global__ void gateLiveBwdK(const real* ctx, const real* g, const real* mask, const real* d, real* dctx, real* dg, int A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * C) return;
  if (mask[t / C] == 0) { dctx[t] = 0; dg[t] = 0; return; }
  real s = 1 / (1 + exp(-g[t]));
  dctx[t] = d[t] * s; dg[t] = d[t] * ctx[t] * s * (1 - s);
}

struct SwaBlock { Linear adaln, qkv, gate, out, up, down; };
struct SwaStack { std::vector<SwaBlock> blocks; int C = 128, hidden = 256, hw = 64; };
inline void forEachParam(SwaStack& st, const std::string& prefix, int nb, const std::function<void(const std::string&, Param&, size_t)>& f) {
  const int C = st.C, Hd = st.hidden;
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; f(n, l.w, (size_t)K * N); };
  st.blocks.assign(nb, SwaBlock{});
  for (int b = 0; b < nb; ++b) {
    const std::string B = prefix + "/blocks/" + std::to_string(b) + "/";
    SwaBlock& k = st.blocks[b];
    lin(B + "adaln", k.adaln, C, 6 * C); lin(B + "qkv", k.qkv, C, 3 * C); lin(B + "attnGate", k.gate, C, C);
    lin(B + "attnOut", k.out, C, C); lin(B + "ffnUp", k.up, C, 2 * Hd); lin(B + "ffnDown", k.down, Hd, C);
  }
}
// the atom layout a stack reads: the valid atoms in rank order, the mask, the rotary table
struct AtomLayout { int A = 0, nValid = 0; const int* valid; const real* mask; const real *cosT, *sinT; };

struct SwaSave { real *mod, *xh1, *iv1, *xm1, *qkvRaw, *qkv, *nrm, *ivh, *P, *ctx, *g, *cg, *att, *x1, *xh2, *iv2, *xm2, *h, *sg; };
inline SwaSave swaForward(const SwaStack& st, const SwaBlock& b, const AtomLayout& L, const real* sc0, const real* x, real* out) {
  SwaSave s{};
  const int A = L.A, C = st.C, Hd = st.hidden, heads = C / HD, hw = std::min(st.hw, A), W = 2 * hw + 1;
  const size_t AC = (size_t)A * C;
  s.mod = ARENA.get(AC * 6); b.adaln.forward(sc0, s.mod, A);
  s.xh1 = ARENA.get(AC); s.iv1 = ARENA.get(A); s.xm1 = ARENA.get(AC);
  rmsModK<<<(unsigned)((A + 7) / 8), 256, 0, STREAM>>>(x, s.mod, 0, 1, s.xm1, s.xh1, s.iv1, A, C);
  s.qkvRaw = ARENA.get(AC * 3); b.qkv.forward(s.xm1, s.qkvRaw, A);
  s.qkv = cloneR(s.qkvRaw, AC * 3);
  s.nrm = ARENA.get(AC * 2); s.ivh = ARENA.get((size_t)A * 2 * heads);
  qkPrepK<<<blocks((size_t)A * 2 * heads, 128), 128, 0, STREAM>>>(s.qkvRaw, L.cosT, L.sinT, s.qkv, s.nrm, s.ivh, A, C);
  s.P = ARENA.get((size_t)heads * std::max(L.nValid, 1) * W);
  s.ctx = ARENA.zeros(AC);
  if (L.nValid) {
    swaScoresK<<<blocks((size_t)heads * L.nValid, 128), 128, 0, STREAM>>>(s.qkv, L.valid, L.nValid, hw, 1 / sqrt((real)HD), s.P, C);
    swaApplyK<<<blocks((size_t)heads * L.nValid * HD), 256, 0, STREAM>>>(s.qkv, L.valid, L.nValid, hw, s.P, s.ctx, C);
  }
  s.g = ARENA.get(AC); b.gate.forward(s.xm1, s.g, A);
  s.cg = ARENA.get(AC); gateLiveFwdK<<<blocks(AC), 256, 0, STREAM>>>(s.ctx, s.g, L.mask, s.cg, A, C);
  s.att = ARENA.get(AC); b.out.forward(s.cg, s.att, A);
  s.x1 = cloneR(x, AC); gatedAddK<<<blocks(AC), 256, 0, STREAM>>>(s.x1, s.mod, s.att, A, C, 2);
  s.xh2 = ARENA.get(AC); s.iv2 = ARENA.get(A); s.xm2 = ARENA.get(AC);
  rmsModK<<<(unsigned)((A + 7) / 8), 256, 0, STREAM>>>(s.x1, s.mod, 3, 4, s.xm2, s.xh2, s.iv2, A, C);
  s.h = ARENA.get((size_t)A * 2 * Hd); b.up.forward(s.xm2, s.h, A);
  s.sg = ARENA.get((size_t)A * Hd); swigluK<<<blocks((size_t)A * Hd), 256, 0, STREAM>>>(s.h, s.sg, A, Hd);
  real* f = ARENA.get(AC); b.down.forward(s.sg, f, A);
  GK(cudaMemcpyAsync(out, s.x1, AC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  gatedAddK<<<blocks(AC), 256, 0, STREAM>>>(out, s.mod, f, A, C, 5);
  return s;
}
// dout -> dx (written), dsc0 (added: the silu'd conditioning's gradient)
inline void swaBackward(SwaStack& st, SwaBlock& b, const AtomLayout& L, const real* sc0, const real* x, const SwaSave& s,
                        const real* dout, real* dx, real* dsc0) {
  Scratch scratch;
  const int A = L.A, C = st.C, Hd = st.hidden, heads = C / HD, hw = std::min(st.hw, A), W = 2 * hw + 1;
  const size_t AC = (size_t)A * C;
  real* dmod = ARENA.zeros(AC * 6);
  // x2 = x1 + gate_f f
  real* f = ARENA.get(AC); b.down.forward(s.sg, f, A);
  real* df = ARENA.get(AC);
  gatedAddBwdK<<<blocks(AC), 256, 0, STREAM>>>(dout, s.mod, f, dmod, df, A, C, 5);
  real* dsg = ARENA.get((size_t)A * Hd); b.down.backward(s.sg, df, dsg, A);
  real* dh = ARENA.get((size_t)A * 2 * Hd); swigluBwdK<<<blocks((size_t)A * Hd), 256, 0, STREAM>>>(s.h, dsg, dh, A, Hd);
  real* dxm2 = ARENA.get(AC); b.up.backward(s.xm2, dh, dxm2, A);
  real* dx1 = cloneR(dout, AC);
  rmsModBwdK<<<(unsigned)((A + 7) / 8), 256, 0, STREAM>>>(s.xh2, s.iv2, s.mod, 3, 4, dxm2, dmod, dx1, A, C);
  // x1 = x + gate_a att
  real* datt = ARENA.get(AC);
  gatedAddBwdK<<<blocks(AC), 256, 0, STREAM>>>(dx1, s.mod, s.att, dmod, datt, A, C, 2);
  real* dcg = ARENA.get(AC); b.out.backward(s.cg, datt, dcg, A);
  real *dctx = ARENA.get(AC), *dg = ARENA.get(AC);
  gateLiveBwdK<<<blocks(AC), 256, 0, STREAM>>>(s.ctx, s.g, L.mask, dcg, dctx, dg, A, C);
  real* dxm1 = ARENA.get(AC); b.gate.backward(s.xm1, dg, dxm1, A);
  real* dqkv = ARENA.zeros(AC * 3);
  if (L.nValid) {
    real* dS = ARENA.get((size_t)heads * L.nValid * W);
    swaScoresBwdK<<<blocks((size_t)heads * L.nValid, 128), 128, 0, STREAM>>>(s.qkv, L.valid, L.nValid, hw, 1 / sqrt((real)HD), s.P, dctx, dS, C);
    swaQKVBwdK<<<blocks((size_t)heads * L.nValid * HD), 256, 0, STREAM>>>(s.qkv, L.valid, L.nValid, hw, s.P, dS, dctx, dqkv, C);
  }
  real* dRaw = cloneR(dqkv, AC * 3);               // v passes straight; q, k through their norm and rotation
  qkPrepBwdK<<<blocks((size_t)A * 2 * heads, 128), 128, 0, STREAM>>>(dqkv, s.nrm, s.ivh, L.cosT, L.sinT, dRaw, A, C);
  b.qkv.backward(s.xm1, dRaw, dxm1, A, true);
  GK(cudaMemcpyAsync(dx, dx1, AC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  rmsModBwdK<<<(unsigned)((A + 7) / 8), 256, 0, STREAM>>>(s.xh1, s.iv1, s.mod, 0, 1, dxm1, dmod, dx, A, C);
  b.adaln.backward(sc0, dmod, dsc0, A, true);
}
// The stack: c0 conditions every block (silu'd once); x in place of out allowed.
struct SwaRun { std::vector<real*> acts; real* sc0 = nullptr; };
inline void swaStackForward(SwaStack& st, const AtomLayout& L, const real* c0, const real* x, real* out, SwaRun& r) {
  const size_t AC = (size_t)L.A * st.C;
  r.sc0 = ARENA.get(AC); siluFwdK<<<blocks(AC), 256, 0, STREAM>>>(c0, r.sc0, AC);
  r.acts.assign(st.blocks.size() + 1, nullptr);
  for (auto& p : r.acts) p = ARENA.get(AC);
  GK(cudaMemcpyAsync(r.acts[0], x, AC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  for (size_t b = 0; b < st.blocks.size(); ++b) {
    Scratch sc;
    swaForward(st, st.blocks[b], L, r.sc0, r.acts[b], r.acts[b + 1]);
  }
  GK(cudaMemcpyAsync(out, r.acts.back(), AC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}
// dout -> dx (written), dc0 (added)
inline void swaStackBackward(SwaStack& st, const AtomLayout& L, const real* c0, const SwaRun& r, const real* dout, real* dx, real* dc0) {
  Scratch scratch;
  const size_t AC = (size_t)L.A * st.C;
  real* d = cloneR(dout, AC);
  real* dsc0 = ARENA.zeros(AC);
  for (size_t b = st.blocks.size(); b-- > 0;) {
    Scratch sc;
    real* o = ARENA.get(AC);
    SwaSave s = swaForward(st, st.blocks[b], L, r.sc0, r.acts[b], o);
    real* dIn = ARENA.get(AC);
    swaBackward(st, st.blocks[b], L, r.sc0, r.acts[b], s, d, dIn, dsc0);
    GK(cudaMemcpyAsync(d, dIn, AC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  GK(cudaMemcpyAsync(dx, d, AC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  siluBwdK<<<blocks(AC), 256, 0, STREAM>>>(c0, dsc0, dc0, AC);
}

}  // namespace ef2
}  // namespace grad
