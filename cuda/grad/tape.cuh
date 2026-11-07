// A reverse-mode tape for training, on the device, in `real` (float, or double under -DGRAD_F64): every op's backward
// written beside its forward and replayed in reverse. The weights are Params, whose gradients the backward accumulates
// (Param.g) - the difference from cuda/bioemu2/src/vjp.cuh's tape, which it follows op for op and which holds the
// weights constant (edm_fkc asks a denoiser for d/dx alone). Every node is kept; the tape's memory is the arena's.
//
// Linear weights are torch's [out, in] (y = x W^T + b), embedding tables [rows, C].
#pragma once
#include <deque>
#include <functional>
#include "ops.cuh"

namespace grad {
namespace tp {

struct Node { real* v; real* g; size_t rows, cols; size_t n() const { return rows * cols; } };

__global__ void accK(real* g, const real* d, size_t n) { size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n) g[t] += d[t]; }
__global__ void addBK(const real* a, const real* b, real* y, size_t n, size_t bn) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n) y[t] = a[t] + b[t % bn];
}
__global__ void colSumAccK(real* g, const real* d, size_t rows, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  real s = 0; for (size_t r = 0; r < rows; ++r) s += d[r * C + c];
  g[c] += s;
}
__global__ void mulK2(const real* a, const real* b, real* y, size_t n) { size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n) y[t] = a[t] * b[t]; }
__global__ void mulAccK(real* g, const real* d, const real* o, size_t n) { size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n) g[t] += d[t] * o[t]; }
__global__ void scaleK2(const real* x, real* y, real s, size_t n) { size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n) y[t] = x[t] * s; }
__global__ void axpyK2(real* g, const real* d, real s, size_t n) { size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n) g[t] += d[t] * s; }
__global__ void siluK2(const real* x, real* y, size_t n) { size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n) y[t] = x[t] / (1 + exp(-x[t])); }
__global__ void siluBK(real* g, const real* d, const real* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) { real s = 1 / (1 + exp(-x[t])); g[t] += d[t] * s * (1 + x[t] * (1 - s)); }
}
__global__ void sigK(const real* x, real* y, size_t n) { size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n) y[t] = 1 / (1 + exp(-x[t])); }
__global__ void sigBK(real* g, const real* d, const real* y, size_t n) { size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n) g[t] += d[t] * y[t] * (1 - y[t]); }
__global__ void reluK2(const real* x, real* y, size_t n) { size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n) y[t] = x[t] > 0 ? x[t] : 0; }
__global__ void reluBK(real* g, const real* d, const real* x, size_t n) { size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < n && x[t] > 0) g[t] += d[t]; }
__global__ void maskRowsK2(const real* x, real* y, const int* m, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < rows * C) y[t] = m[t / C] ? x[t] : 0;
}
__global__ void maskRowsBK(real* g, const real* d, const int* m, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < rows * C && m[t / C]) g[t] += d[t];
}
__global__ void gatherK2(const real* x, real* y, const int* idx, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  int i = idx[t / C]; y[t] = i < 0 ? 0 : x[(size_t)i * C + t % C];
}
__global__ void scatterBK(real* g, const real* d, const int* idx, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  int i = idx[t / C]; if (i >= 0) atomicAdd(&g[(size_t)i * C + t % C], d[t]);
}
__global__ void copyColsK2(const real* x, int ldx, int from, real* y, int ldy, int to, size_t rows, int w) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * w) { size_t r = t / w; int c = t % w; y[r * ldy + to + c] = x[r * ldx + from + c]; }
}
__global__ void accColsK2(real* g, int ldg, int to, const real* d, int ldd, int from, size_t rows, int w) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * w) { size_t r = t / w; int c = t % w; g[r * ldg + to + c] += d[r * ldd + from + c]; }
}
// LayerNorm, eps given, affine optional (w and/or b null): xhat and 1/std kept
__global__ void lnFK(const real* x, real* y, real* xhat, real* inv, size_t rows, int C, const real* w, const real* b, real eps) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const real* xr = x + row * C;
  real s = 0; for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  real mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { real d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  real iv = 1 / sqrt(v / C + eps);
  if (lane == 0) inv[row] = iv;
  for (int c = lane; c < C; c += 32) {
    real h = (xr[c] - mean) * iv; xhat[row * C + c] = h;
    y[row * C + c] = h * (w ? w[c] : (real)1) + (b ? b[c] : (real)0);
  }
}
// gx += inv (g - mean g - xhat mean(g xhat)), g = gy w
__global__ void lnBK(real* gx, const real* gy, const real* xhat, const real* inv, size_t rows, int C, const real* w) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  real s1 = 0, s2 = 0;
  for (int c = lane; c < C; c += 32) { real g = gy[row * C + c] * (w ? w[c] : (real)1); s1 += g; s2 += g * xhat[row * C + c]; }
  for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(~0u, s1, o); s2 += __shfl_xor_sync(~0u, s2, o); }
  s1 /= C; s2 /= C;
  for (int c = lane; c < C; c += 32) {
    real g = gy[row * C + c] * (w ? w[c] : (real)1);
    gx[row * C + c] += inv[row] * (g - s1 - xhat[row * C + c] * s2);
  }
}
__global__ void lnParamBK(const real* gy, const real* xhat, real* gw, real* gb, size_t rows, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  real sw = 0, sb = 0;
  for (size_t r = 0; r < rows; ++r) { real g = gy[r * C + c]; sw += g * xhat[r * C + c]; sb += g; }
  if (gw) gw[c] += sw;
  if (gb) gb[c] += sb;
}
// adaLN: y = xhat (1 + cond[oa]) + cond[ob]
__global__ void adaFK(const real* xhat, const real* cond, int ldc, int oa, int ob, real* y, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = t % C;
  y[t] = xhat[t] * (1 + cond[r * ldc + oa + c]) + cond[r * ldc + ob + c];
}
__global__ void adaBK(const real* gy, const real* xhat, const real* cond, int ldc, int oa, int ob, real* gn, real* gc, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = t % C;
  gn[t] = gy[t] * (1 + cond[r * ldc + oa + c]);
  if (gc) { gc[r * ldc + oa + c] += gy[t] * xhat[t]; gc[r * ldc + ob + c] += gy[t]; }
}
// y = x + cond[og] h
__global__ void gateFK(const real* x, const real* h, const real* cond, int ldc, int og, real* y, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = t % C;
  y[t] = x[t] + cond[r * ldc + og + c] * h[t];
}
__global__ void gateBK(const real* gy, const real* h, const real* cond, int ldc, int og, real* gx, real* gh, real* gc, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = t % C;
  gx[t] += gy[t];
  gh[t] += gy[t] * cond[r * ldc + og + c];
  gc[r * ldc + og + c] += gy[t] * h[t];
}

// attention with an additive bias node: full (every query its keys 0..K) or BioEmu2's window (query atom i of window
// w = i / 32: keys w*32 - 48 + j, j < 128, out of range no key). Logits (q + q_bias) . k / sqrt(HD) + bias; a key masked
// (live 0) or out of range: -1e6 (the vendor's). Bias layout: full [H, R, K]; window [NW, H, 32, 128].
struct AttnShape { bool window; int R, H, HD, K; const int* live; };
__device__ inline int keyOf(const AttnShape& a, int q, int j) {
  if (!a.window) return j;
  int k = (q / 32) * 32 - 48 + j;
  return k >= 0 && k < a.R ? k : -1;
}
__device__ inline size_t biasAt(const AttnShape& a, int q, int h, int j) {
  if (!a.window) return ((size_t)h * a.R + q) * a.K + j;
  return (((size_t)(q / 32) * a.H + h) * 32 + q % 32) * a.K + j;
}
// a thread a (query, head): P [R, H, K]
__global__ void attnFK(AttnShape a, const real* qkv, const real* qb, const real* bias, real* P, real* o) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)a.R * a.H) return;
  int q = t / a.H, h = t % a.H;
  const int D = a.H * a.HD;
  const real sc = 1 / sqrt((real)a.HD);
  real* p = P + t * a.K;
  real mx = -INFINITY;
  for (int j = 0; j < a.K; ++j) {
    int k = keyOf(a, q, j);
    real l;
    if (k < 0) l = (real)-1e6;
    else {
      real s = 0;
      for (int d = 0; d < a.HD; ++d) s += (qkv[(size_t)q * 3 * D + h * a.HD + d] + qb[h * a.HD + d]) * qkv[(size_t)k * 3 * D + D + h * a.HD + d];
      l = s * sc + bias[biasAt(a, q, h, j)];
      if (!a.live[k]) l = (real)-1e6;
    }
    p[j] = l; mx = fmax(mx, l);
  }
  real sum = 0;
  for (int j = 0; j < a.K; ++j) { p[j] = exp(p[j] - mx); sum += p[j]; }
  for (int j = 0; j < a.K; ++j) p[j] /= sum;
  for (int d = 0; d < a.HD; ++d) {
    real s = 0;
    for (int j = 0; j < a.K; ++j) { int k = keyOf(a, q, j); if (k >= 0) s += p[j] * qkv[(size_t)k * 3 * D + 2 * D + h * a.HD + d]; }
    o[(size_t)q * D + h * a.HD + d] = s;
  }
}
// dl [R, H, K] = P (dP - P.dP), the live keys' (a -1e6 logit carries no gradient); the query's gradient and the bias's
__global__ void attnBQK(AttnShape a, const real* qkv, const real* P, const real* go, real* dl, real* gqkv, real* gqb, real* gbias) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)a.R * a.H) return;
  int q = t / a.H, h = t % a.H;
  const int D = a.H * a.HD;
  const real sc = 1 / sqrt((real)a.HD);
  const real* p = P + t * a.K;
  real* d = dl + t * a.K;
  real dot = 0;
  for (int j = 0; j < a.K; ++j) {
    int k = keyOf(a, q, j);
    real dp = 0;
    if (k >= 0) for (int e = 0; e < a.HD; ++e) dp += go[(size_t)q * D + h * a.HD + e] * qkv[(size_t)k * 3 * D + 2 * D + h * a.HD + e];
    d[j] = dp; dot += p[j] * dp;
  }
  for (int j = 0; j < a.K; ++j) {
    int k = keyOf(a, q, j);
    d[j] = (k >= 0 && a.live[k]) ? p[j] * (d[j] - dot) : 0;
    if (k >= 0 && a.live[k] && gbias) atomicAdd(&gbias[biasAt(a, q, h, j)], d[j]);
  }
  for (int e = 0; e < a.HD; ++e) {
    real s = 0;
    for (int j = 0; j < a.K; ++j) { int k = keyOf(a, q, j); if (k >= 0) s += d[j] * qkv[(size_t)k * 3 * D + D + h * a.HD + e]; }
    gqkv[(size_t)q * 3 * D + h * a.HD + e] += s * sc;
    if (gqb) atomicAdd(&gqb[h * a.HD + e], s * sc);
  }
}
// the key and value gradients, a thread a (key, head): every query whose window holds it
__global__ void attnBKV(AttnShape a, const real* qkv, const real* qb, const real* P, const real* dl, const real* go, real* gqkv) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)a.R * a.H) return;
  int k = t / a.H, h = t % a.H;
  const int D = a.H * a.HD;
  const real sc = 1 / sqrt((real)a.HD);
  int q0 = 0, q1 = a.R;
  if (a.window) { q0 = max(0, ((k - 79) / 32) * 32 - 32); q1 = min(a.R, ((k + 48) / 32 + 1) * 32 + 32); }
  for (int e = 0; e < a.HD; ++e) {
    real gk = 0, gv = 0;
    for (int q = q0; q < q1; ++q) {
      int j = a.window ? k - ((q / 32) * 32 - 48) : k;
      if (j < 0 || j >= a.K) continue;
      size_t row = ((size_t)q * a.H + h) * a.K + j;
      gk += dl[row] * (qkv[(size_t)q * 3 * D + h * a.HD + e] + qb[h * a.HD + e]) * sc;
      gv += P[row] * go[(size_t)q * D + h * a.HD + e];
    }
    gqkv[(size_t)k * 3 * D + D + h * a.HD + e] += gk;
    gqkv[(size_t)k * 3 * D + 2 * D + h * a.HD + e] += gv;
  }
}

class Tape {
 public:
  std::deque<Node> nodes;
  std::vector<std::function<void()>> backs;

  Node* leaf(const real* data, size_t rows, size_t cols) {
    Node& n = make(rows, cols);
    GK(cudaMemcpyAsync(n.v, data, n.n() * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    return &n;
  }
  Node* zeros(size_t rows, size_t cols) { Node& n = make(rows, cols); GK(cudaMemsetAsync(n.v, 0, n.n() * sizeof(real), STREAM)); return &n; }
  real* grad(Node* x) { return x->g; }
  // the seed into out's gradient, every backward in reverse
  void backward(Node* out, const real* seed) {
    accK<<<blocks(out->n()), 256, 0, STREAM>>>(out->g, seed, out->n());
    for (size_t k = backs.size(); k-- > 0;) backs[k]();
  }
  // y = x W^T (+ b), W [out, in]
  Node* linear(Node* x, Param& W, Param* b, int out) {
    const int in = (int)x->cols; const size_t rows = x->rows;
    Node* y = &make(rows, out);
    mmBT(x->v, W.w, y->v, (int)rows, out, in);
    if (b) biasK<<<blocks(rows * out), 256, 0, STREAM>>>(y->v, b->w, rows, out);
    Param* Wp = &W;
    backs.push_back([=] {
      mmAT(y->g, x->v, Wp->g, (int)rows, out, in, 1);                    // dW [out, in] += dy^T x
      if (b) colSumAccK<<<blocks(out, 64), 64, 0, STREAM>>>(b->g, y->g, rows, out);
      mm(y->g, Wp->w, x->g, (int)rows, out, in, 1);                      // dx += dy W
    });
    return y;
  }
  Node* layerNorm(Node* x, Param* w, Param* b, real eps = (real)1e-5) {
    Node* y = &make(x->rows, x->cols);
    real* xhat = ARENA.get(x->n()); real* inv = ARENA.get(x->rows);
    const int C = (int)x->cols;
    lnFK<<<blocks(x->rows, 8), 256, 0, STREAM>>>(x->v, y->v, xhat, inv, x->rows, C, w ? w->w : nullptr, b ? b->w : nullptr, eps);
    backs.push_back([=] {
      lnBK<<<blocks(x->rows, 8), 256, 0, STREAM>>>(x->g, y->g, xhat, inv, x->rows, C, w ? w->w : nullptr);
      if (w || b) lnParamBK<<<blocks(C, 64), 64, 0, STREAM>>>(y->g, xhat, w ? w->g : nullptr, b ? b->g : nullptr, x->rows, C);
    });
    return y;
  }
  Node* silu(Node* x) { Node* y = &make(x->rows, x->cols); siluK2<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, y->v, x->n());
    backs.push_back([=] { siluBK<<<blocks(x->n()), 256, 0, STREAM>>>(x->g, y->g, x->v, x->n()); }); return y; }
  Node* sigmoid(Node* x) { Node* y = &make(x->rows, x->cols); sigK<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, y->v, x->n());
    backs.push_back([=] { sigBK<<<blocks(x->n()), 256, 0, STREAM>>>(x->g, y->g, y->v, x->n()); }); return y; }
  Node* relu(Node* x) { Node* y = &make(x->rows, x->cols); reluK2<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, y->v, x->n());
    backs.push_back([=] { reluBK<<<blocks(x->n()), 256, 0, STREAM>>>(x->g, y->g, x->v, x->n()); }); return y; }
  // a + b, b of a's shape or one row broadcast
  Node* add(Node* a, Node* b) {
    Node* y = &make(a->rows, a->cols);
    addBK<<<blocks(a->n()), 256, 0, STREAM>>>(a->v, b->v, y->v, a->n(), b->n());
    backs.push_back([=] {
      accK<<<blocks(a->n()), 256, 0, STREAM>>>(a->g, y->g, a->n());
      if (b->n() == a->n()) accK<<<blocks(a->n()), 256, 0, STREAM>>>(b->g, y->g, a->n());
      else colSumAccK<<<blocks(b->n(), 64), 64, 0, STREAM>>>(b->g, y->g, a->rows, (int)b->n());
    });
    return y;
  }
  // x + a parameter row broadcast (a bias as a node)
  Node* addParam(Node* x, Param& p) {
    Node* y = &make(x->rows, x->cols);
    addBK<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, p.w, y->v, x->n(), p.n);
    Param* pp = &p;
    backs.push_back([=] {
      accK<<<blocks(x->n()), 256, 0, STREAM>>>(x->g, y->g, x->n());
      colSumAccK<<<blocks(pp->n, 64), 64, 0, STREAM>>>(pp->g, y->g, x->rows, (int)pp->n);
    });
    return y;
  }
  Node* mul(Node* a, Node* b) {
    Node* y = &make(a->rows, a->cols);
    mulK2<<<blocks(a->n()), 256, 0, STREAM>>>(a->v, b->v, y->v, a->n());
    backs.push_back([=] {
      mulAccK<<<blocks(a->n()), 256, 0, STREAM>>>(a->g, y->g, b->v, a->n());
      mulAccK<<<blocks(a->n()), 256, 0, STREAM>>>(b->g, y->g, a->v, a->n());
    });
    return y;
  }
  Node* scale(Node* x, real s) {
    Node* y = &make(x->rows, x->cols);
    scaleK2<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, y->v, s, x->n());
    backs.push_back([=] { axpyK2<<<blocks(x->n()), 256, 0, STREAM>>>(x->g, y->g, s, x->n()); });
    return y;
  }
  Node* maskRows(Node* x, const int* mask) {
    Node* y = &make(x->rows, x->cols);
    maskRowsK2<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, y->v, mask, x->rows, (int)x->cols);
    backs.push_back([=] { maskRowsBK<<<blocks(x->n()), 256, 0, STREAM>>>(x->g, y->g, mask, x->rows, (int)x->cols); });
    return y;
  }
  // rows index[i] of x (-1: a zero row)
  Node* gather(Node* x, const int* index, size_t rows) {
    Node* y = &make(rows, x->cols);
    gatherK2<<<blocks(y->n()), 256, 0, STREAM>>>(x->v, y->v, index, rows, (int)x->cols);
    backs.push_back([=] { scatterBK<<<blocks(y->n()), 256, 0, STREAM>>>(x->g, y->g, index, rows, (int)x->cols); });
    return y;
  }
  Node* concat(Node* a, Node* b) {
    const int ca = (int)a->cols, cb = (int)b->cols, c = ca + cb;
    Node* y = &make(a->rows, c);
    copyColsK2<<<blocks(a->n()), 256, 0, STREAM>>>(a->v, ca, 0, y->v, c, 0, a->rows, ca);
    copyColsK2<<<blocks(b->n()), 256, 0, STREAM>>>(b->v, cb, 0, y->v, c, ca, b->rows, cb);
    backs.push_back([=] {
      accColsK2<<<blocks(a->n()), 256, 0, STREAM>>>(a->g, ca, 0, y->g, c, 0, a->rows, ca);
      accColsK2<<<blocks(b->n()), 256, 0, STREAM>>>(b->g, cb, 0, y->g, c, ca, b->rows, cb);
    });
    return y;
  }
  Node* adaLn(Node* x, Node* cond, int oa, int ob) {
    const int C = (int)x->cols;
    Node* y = &make(x->rows, C);
    real *xhat = ARENA.get(x->n()), *inv = ARENA.get(x->rows), *tmp = ARENA.get(x->n());
    lnFK<<<blocks(x->rows, 8), 256, 0, STREAM>>>(x->v, tmp, xhat, inv, x->rows, C, nullptr, nullptr, (real)1e-5);
    adaFK<<<blocks(x->n()), 256, 0, STREAM>>>(xhat, cond->v, (int)cond->cols, oa, ob, y->v, x->rows, C);
    backs.push_back([=] {
      real* gn = ARENA.get(x->n());
      adaBK<<<blocks(x->n()), 256, 0, STREAM>>>(y->g, xhat, cond->v, (int)cond->cols, oa, ob, gn, cond->g, x->rows, C);
      lnBK<<<blocks(x->rows, 8), 256, 0, STREAM>>>(x->g, gn, xhat, inv, x->rows, C, nullptr);
    });
    return y;
  }
  Node* gatedAdd(Node* x, Node* h, Node* cond, int og) {
    const int C = (int)x->cols;
    Node* y = &make(x->rows, C);
    gateFK<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, h->v, cond->v, (int)cond->cols, og, y->v, x->rows, C);
    backs.push_back([=] {
      gateBK<<<blocks(x->n()), 256, 0, STREAM>>>(y->g, h->v, cond->v, (int)cond->cols, og, x->g, h->g, cond->g, x->rows, C);
    });
    return y;
  }
  // qkv [R, 3D], the query bias a parameter [D], the bias a node
  Node* attention(Node* qkv, Param& qb, Node* bias, AttnShape a) {
    const int D = a.H * a.HD;
    Node* y = &make(a.R, D);
    real* P = ARENA.get((size_t)a.R * a.H * a.K);
    attnFK<<<blocks((size_t)a.R * a.H, 64), 64, 0, STREAM>>>(a, qkv->v, qb.w, bias->v, P, y->v);
    Param* qp = &qb;
    backs.push_back([=] {
      real* dl = ARENA.get((size_t)a.R * a.H * a.K);
      attnBQK<<<blocks((size_t)a.R * a.H, 64), 64, 0, STREAM>>>(a, qkv->v, P, y->g, dl, qkv->g, qp->g, bias->g);
      attnBKV<<<blocks((size_t)a.R * a.H, 64), 64, 0, STREAM>>>(a, qkv->v, qp->w, P, dl, y->g, qkv->g);
    });
    return y;
  }
  // a custom op: forward run now, its backward (reading y->g, writing its inputs' and params' gradients) on the tape
  Node* custom(size_t rows, size_t cols, const std::function<void(Node*)>& fwd, const std::function<void(Node*)>& bwd) {
    Node* y = &make(rows, cols);
    fwd(y);
    backs.push_back([=] { bwd(y); });
    return y;
  }

 private:
  Node& make(size_t rows, size_t cols) {
    real* v = ARENA.get(rows * cols); real* g = ARENA.zeros(rows * cols);
    nodes.push_back({v, g, rows, cols});
    return nodes.back();
  }
};

}  // namespace tp
}  // namespace grad
