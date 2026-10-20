// BioEmu2's denoiser on a reverse-mode tape, on the device: D(x, sigma) and the vector-Jacobian product
// w . dD/dx, w . dD/dsigma - gradients into ACTIVATIONS only, the weights constants - which is what edm_fkc's
// corrector asks of a denoiser. cpu/bioemu2/denoiser.js on cpu/bioemu2/tape.js, op for op: the same graph, every
// op's backward written beside its forward, in float32 with cuBLAS held to plain float32 (no TF32) while it runs.
// Held to af3-any-model's jax.grad (oracle-dumps/bioemu2-6mrr.json's vjp arms: localfold-bioemu2 --oracle) and, by
// that, to the CPU tape.
//
// What does not depend on sigma or the coordinates comes from the Denoiser (denoise.cuh): q0, a0, the single
// conditioner's static half sc0, every layer's pair biases (wbias, tbias). One particle a tape; every activation is
// kept (a 640-atom input holds ~0.3 GB) - `vjpBytes` reports what a call held.
#pragma once
#include <deque>
#include "denoise_call.cuh"

namespace be2vjp {
using namespace be2;

// ---------------------------------------------------------------- the device arena (reset per tape)
struct Arena {
  std::vector<std::pair<char*, size_t>> chunks;
  size_t chunk = 0, at = 0, high = 0, used = 0;
  float* get(size_t n) {
    size_t bytes = (n * 4 + 255) & ~(size_t)255;
    while (chunk < chunks.size() && at + bytes > chunks[chunk].second) { ++chunk; at = 0; }
    if (chunk == chunks.size()) {
      size_t size = std::max(bytes, (size_t)256 << 20);
      char* p; CK(cudaMalloc(&p, size));
      chunks.push_back({p, size});
      at = 0;
    }
    float* p = (float*)(chunks[chunk].first + at);
    at += bytes; used += bytes; high = std::max(high, used);
    return p;
  }
  void reset() { chunk = 0; at = 0; used = 0; }
  void release() { CK(cudaStreamSynchronize(STREAM)); for (auto& c : chunks) CK(cudaFree(c.first)); chunks.clear(); reset(); }
  size_t held() const { size_t s = 0; for (auto& c : chunks) s += c.second; return s; }
};
inline Arena ARENA;

struct Node {
  float* v = nullptr; float* g = nullptr;
  size_t rows = 0, cols = 0; bool track = false;
  size_t n() const { return rows * cols; }
};
// a function of sigma, on the host: its value, its derivative, and the gradient that reaches it
struct Scalar { double v = 0, dv = 0, g = 0; };

__global__ void addK(const float* a, const float* b, float* y, size_t n, size_t bn) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = a[t] + b[t % bn];
}
__global__ void accK(float* g, const float* d, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) g[t] += d[t];
}
__global__ void colSumAccK(float* g, const float* d, size_t rows, int C) {     // g[c] += sum over rows of d[r, c]
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  double s = 0; for (size_t r = 0; r < rows; ++r) s += d[r * C + c];
  g[c] += (float)s;
}
__global__ void mulK(const float* a, const float* b, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = a[t] * b[t];
}
__global__ void mulAccK(float* g, const float* d, const float* o, size_t n) {   // g += d * o
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) g[t] += d[t] * o[t];
}
__global__ void scaleK(const float* x, float* y, float s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = x[t] * s;
}
__global__ void axpyK(float* g, const float* d, float s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) g[t] += s * d[t];
}
__global__ void siluFwdK(const float* x, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = x[t] / (1.f + expf(-x[t]));
}
__global__ void siluBackK(float* g, const float* d, const float* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) { float s = 1.f / (1.f + expf(-x[t])); g[t] += d[t] * s * (1.f + x[t] * (1.f - s)); }
}
__global__ void sigmoidFwdK(const float* x, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = 1.f / (1.f + expf(-x[t]));
}
__global__ void sigmoidBackK(float* g, const float* d, const float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) g[t] += d[t] * y[t] * (1.f - y[t]);
}
__global__ void maskRowsK(const float* x, float* y, const int* mask, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) y[t] = mask[t / C] ? x[t] : 0.f;
}
__global__ void maskRowsAccK(float* g, const float* d, const int* mask, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C && mask[t / C]) g[t] += d[t];
}
__global__ void gatherK(const float* x, float* y, const int* index, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  int src = index[t / C];
  y[t] = src >= 0 ? x[(size_t)src * C + t % C] : 0.f;
}
__global__ void scatterAccK(float* g, const float* d, const int* index, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  int src = index[t / C];
  if (src >= 0) atomicAdd(g + (size_t)src * C + t % C, d[t]);
}
__global__ void copyColsK(const float* x, int ldx, int from, float* y, int ldy, int to, size_t rows, int w) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * w) y[(t / w) * ldy + to + t % w] = x[(t / w) * ldx + from + t % w];
}
__global__ void accColsK(float* g, int ldg, int to, const float* d, int ldd, int from, size_t rows, int w) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * w) g[(t / w) * ldg + to + t % w] += d[(t / w) * ldd + from + t % w];
}
__global__ void dotK(const float* a, const float* b, size_t n, double* out) {
  __shared__ double s[256];
  double acc = 0;
  for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) acc += (double)a[t] * b[t];
  s[threadIdx.x] = acc; __syncthreads();
  for (int o = 128; o; o >>= 1) { if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(out, s[0]);
}
// LayerNorm (biased variance, eps 1e-5), optional affine; the row's normalised values and 1/std kept for the backward
__global__ void lnFwdK(const float* x, float* y, float* xhat, float* inv, size_t rows, int C, const float* w, const float* b) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * C;
  float s = 0; for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mu = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mu; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float iv = rsqrtf(v / C + 1e-5f);
  if (lane == 0) inv[row] = iv;
  for (int c = lane; c < C; c += 32) {
    float h = (xr[c] - mu) * iv;
    xhat[row * C + c] = h;
    y[row * C + c] = h * (w ? w[c] : 1.f) + (b ? b[c] : 0.f);
  }
}
// gx += inv (g - mean(g) - xhat mean(g xhat)), g = gy * w
__global__ void lnBackK(float* gx, const float* gy, const float* xhat, const float* inv, size_t rows, int C, const float* w) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  float m1 = 0, m2 = 0;
  for (int c = lane; c < C; c += 32) { float g = gy[row * C + c] * (w ? w[c] : 1.f); m1 += g; m2 += g * xhat[row * C + c]; }
  for (int o = 16; o; o >>= 1) { m1 += __shfl_xor_sync(~0u, m1, o); m2 += __shfl_xor_sync(~0u, m2, o); }
  m1 /= C; m2 /= C;
  for (int c = lane; c < C; c += 32) {
    float g = gy[row * C + c] * (w ? w[c] : 1.f);
    gx[row * C + c] += inv[row] * (g - m1 - xhat[row * C + c] * m2);
  }
}
// adaLN: y = xhat (1 + alpha) + beta, alpha and beta columns of a conditioning matrix of x's rows
__global__ void adaFwdK(const float* xhat, const float* cond, int ldc, int oa, int ob, float* y, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = (int)(t % C);
  y[t] = xhat[t] * (1.f + cond[r * ldc + oa + c]) + cond[r * ldc + ob + c];
}
__global__ void adaBackK(const float* gy, const float* xhat, const float* cond, int ldc, int oa, int ob, float* gn,
                         float* gc, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = (int)(t % C);
  gn[t] = gy[t] * (1.f + cond[r * ldc + oa + c]);
  if (gc) { gc[r * ldc + oa + c] += gy[t] * xhat[t]; gc[r * ldc + ob + c] += gy[t]; }
}
// y = x + gate h, gate a column block of the conditioning
__global__ void gateFwdK(const float* x, const float* h, const float* cond, int ldc, int og, float* y, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  y[t] = x[t] + cond[(t / C) * ldc + og + t % C] * h[t];
}
__global__ void gateBackK(const float* gy, const float* h, const float* cond, int ldc, int og, float* gx, float* gh,
                          float* gc, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = (int)(t % C);
  if (gx) gx[t] += gy[t];
  if (gh) gh[t] += gy[t] * cond[r * ldc + og + c];
  if (gc) gc[r * ldc + og + c] += gy[t] * h[t];
}

// ---------------------------------------------------------------- attention with a constant pair bias
// window: query i's K = 128 keys are atoms win*32 - 48 + s (out of range: no key); full: every token. The key mask
// and the out-of-range slot give the vendor's -1e6 logit. q = (qkv[:, :D] + q_bias), scaled inside.
struct AttnShape { bool window; int R, H, HD, K; const float* bias; const int* live; };
__device__ __forceinline__ int keyOf(const AttnShape& a, int i, int s) {
  if (!a.window) return s;
  int k = (i / WQ) * WQ - CTX + s;
  return k >= 0 && k < a.R ? k : -1;
}
__device__ __forceinline__ float biasOf(const AttnShape& a, int h, int i, int s) {
  if (!a.window) return a.bias[((size_t)h * a.R + i) * a.R + s];
  return a.bias[((((size_t)(i / WQ)) * a.H + h) * WQ + i % WQ) * WK + s];
}
// a warp per (query, head): its lanes over the keys for the logits and the softmax, over the head's dims for the output
__global__ void attnFwdK(AttnShape a, const float* qkv, const float* qBias, float* P, float* out) {
  const size_t t = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (t >= (size_t)a.R * a.H) return;
  const int i = (int)(t / a.H), h = (int)(t % a.H), D = a.H * a.HD;
  const float scale = rsqrtf((float)a.HD);
  const float* q = qkv + (size_t)i * 3 * D + h * a.HD;
  const float* qb = qBias + h * a.HD;
  float* p = P + t * a.K;
  float mx = -INFINITY;
  for (int s = lane; s < a.K; s += 32) {
    int k = keyOf(a, i, s);
    float l = NEG_INF;
    if (k >= 0 && a.live[k]) {
      const float* kk = qkv + (size_t)k * 3 * D + D + h * a.HD;
      float dp = 0; for (int e = 0; e < a.HD; ++e) dp += (q[e] + qb[e]) * kk[e];
      l = dp * scale + biasOf(a, h, i, s);
    }
    p[s] = l; mx = fmaxf(mx, l);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  float sum = 0;
  for (int s = lane; s < a.K; s += 32) { float e = expf(p[s] - mx); p[s] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  const float inv = 1.f / sum;
  for (int s = lane; s < a.K; s += 32) p[s] *= inv;
  __syncwarp();
  for (int e = lane; e < a.HD; e += 32) {
    float acc = 0;
    for (int s = 0; s < a.K; ++s) { int k = keyOf(a, i, s); if (k >= 0) acc += p[s] * qkv[(size_t)k * 3 * D + 2 * D + h * a.HD + e]; }
    out[(size_t)i * D + h * a.HD + e] = acc;
  }
}
// a warp per (query, head): dl = P (dP - P.dP) scale on live keys (kept), and the query's gradient
__global__ void attnBackQK(AttnShape a, const float* qkv, const float* P, const float* gy, float* dl, float* gqkv) {
  const size_t t = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (t >= (size_t)a.R * a.H) return;
  const int i = (int)(t / a.H), h = (int)(t % a.H), D = a.H * a.HD;
  const float scale = rsqrtf((float)a.HD);
  const float* p = P + t * a.K;
  float* d = dl + t * a.K;
  const float* g = gy + (size_t)i * D + h * a.HD;
  float dot = 0;
  for (int s = lane; s < a.K; s += 32) {
    int k = keyOf(a, i, s);
    float dp = 0;
    if (k >= 0) { const float* v = qkv + (size_t)k * 3 * D + 2 * D + h * a.HD; for (int e = 0; e < a.HD; ++e) dp += g[e] * v[e]; }
    d[s] = dp; dot += p[s] * dp;
  }
  for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(~0u, dot, o);
  for (int s = lane; s < a.K; s += 32) {
    int k = keyOf(a, i, s);
    d[s] = k >= 0 && a.live[k] ? p[s] * (d[s] - dot) * scale : 0.f;
  }
  __syncwarp();
  for (int e = lane; e < a.HD; e += 32) {
    float gq = 0;
    for (int s = 0; s < a.K; ++s) { int k = keyOf(a, i, s); if (k >= 0 && d[s] != 0) gq += d[s] * qkv[(size_t)k * 3 * D + D + h * a.HD + e]; }
    gqkv[(size_t)i * 3 * D + h * a.HD + e] += gq;
  }
}
// a warp per (key, head), its lanes over the head's dims: the key and value gradients over every query that reads it
__global__ void attnBackKVK(AttnShape a, const float* qkv, const float* qBias, const float* P, const float* dl,
                            const float* gy, float* gqkv) {
  const size_t t = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (t >= (size_t)a.R * a.H) return;
  const int k = (int)(t / a.H), h = (int)(t % a.H), D = a.H * a.HD;
  for (int e = lane; e < a.HD; e += 32) {
    float gk = 0, gv = 0;
    const float qb = qBias[h * a.HD + e];
    auto one = [&](int i, int s) {
      const size_t row = ((size_t)i * a.H + h) * a.K + s;
      gv += P[row] * gy[(size_t)i * D + h * a.HD + e];
      gk += dl[row] * (qkv[(size_t)i * 3 * D + h * a.HD + e] + qb);
    };
    if (!a.window) {
      for (int i = 0; i < a.R; ++i) one(i, k);
    } else {
      // the windows whose 128 keys include k: s = k - (win*32 - 48) in [0, 128)
      const int lo = max(0, (k + CTX - WK + WQ) / WQ), hi = min((k + CTX) / WQ, a.R / WQ - 1);
      for (int win = lo; win <= hi; ++win) {
        const int s = k - (win * WQ - CTX);
        if (s < 0 || s >= WK) continue;
        for (int il = 0; il < WQ; ++il) one(win * WQ + il, s);
      }
    }
    gqkv[(size_t)k * 3 * D + D + h * a.HD + e] += gk;
    gqkv[(size_t)k * 3 * D + 2 * D + h * a.HD + e] += gv;
  }
}

// ---------------------------------------------------------------- the tape
class Tape {
 public:
  std::deque<Node> nodes;
  std::vector<std::function<void()>> backs;
  std::vector<Scalar*> scalars;

  Node* leaf(const float* data, size_t rows, size_t cols, bool track = false) {
    Node& n = make(rows, cols, track);
    CK(cudaMemcpyAsync(n.v, data, n.n() * 4, cudaMemcpyDeviceToDevice, STREAM));
    return &n;
  }
  float* grad(Node* x) {
    if (!x->g) { x->g = ARENA.get(x->n()); CK(cudaMemsetAsync(x->g, 0, x->n() * 4, STREAM)); }
    return x->g;
  }
  void backward(Node* out, const float* seed) {
    CK(cudaMemcpyAsync(grad(out), seed, out->n() * 4, cudaMemcpyDeviceToDevice, STREAM));
    for (size_t k = backs.size(); k-- > 0;) backs[k]();
  }

  // y = x W^T (+ b), W [out, in] torch layout, a constant
  Node* linear(Node* x, const float* W, const float* b, int out) {
    const int in = (int)x->cols, rows = (int)x->rows;
    Node* y = &make(x->rows, out, x->track);
    const float one = 1.f, zero = 0.f;
    CB(cublasSgemm(H, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &one, W, in, x->v, in, &zero, y->v, out));
    if (b) addBiasRows(y->v, b, rows, out);
    if (x->track) backs.push_back([=] {
      if (!y->g) return;
      const float one = 1.f;
      CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_N, in, rows, out, &one, W, in, y->g, out, &one, grad(x), in));
    });
    return y;
  }
  Node* layerNorm(Node* x, const float* w = nullptr, const float* b = nullptr) {
    Node* y = &make(x->rows, x->cols, x->track);
    float* xhat = ARENA.get(x->n()); float* inv = ARENA.get(x->rows);
    lnFwdK<<<(unsigned)((x->rows + 7) / 8), 256, 0, STREAM>>>(x->v, y->v, xhat, inv, x->rows, (int)x->cols, w, b);
    if (x->track) backs.push_back([=] {
      if (!y->g) return;
      lnBackK<<<(unsigned)((x->rows + 7) / 8), 256, 0, STREAM>>>(grad(x), y->g, xhat, inv, x->rows, (int)x->cols, w);
    });
    return y;
  }
  Node* silu(Node* x) {
    Node* y = &make(x->rows, x->cols, x->track);
    siluFwdK<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, y->v, x->n());
    if (x->track) backs.push_back([=] { if (y->g) siluBackK<<<blocks(x->n()), 256, 0, STREAM>>>(grad(x), y->g, x->v, x->n()); });
    return y;
  }
  Node* sigmoid(Node* x) {
    Node* y = &make(x->rows, x->cols, x->track);
    sigmoidFwdK<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, y->v, x->n());
    if (x->track) backs.push_back([=] { if (y->g) sigmoidBackK<<<blocks(x->n()), 256, 0, STREAM>>>(grad(x), y->g, y->v, x->n()); });
    return y;
  }
  // a + b, b of a's shape or one row broadcast over a's rows
  Node* add(Node* a, Node* b) {
    Node* y = &make(a->rows, a->cols, a->track || b->track);
    addK<<<blocks(a->n()), 256, 0, STREAM>>>(a->v, b->v, y->v, a->n(), b->n());
    if (y->track) backs.push_back([=] {
      if (!y->g) return;
      if (a->track) accK<<<blocks(a->n()), 256, 0, STREAM>>>(grad(a), y->g, a->n());
      if (b->track) {
        if (b->n() == a->n()) accK<<<blocks(a->n()), 256, 0, STREAM>>>(grad(b), y->g, a->n());
        else colSumAccK<<<blocks(b->n()), 256, 0, STREAM>>>(grad(b), y->g, a->rows, (int)b->n());
      }
    });
    return y;
  }
  Node* mul(Node* a, Node* b) {
    Node* y = &make(a->rows, a->cols, a->track || b->track);
    mulK<<<blocks(a->n()), 256, 0, STREAM>>>(a->v, b->v, y->v, a->n());
    if (y->track) backs.push_back([=] {
      if (!y->g) return;
      if (a->track) mulAccK<<<blocks(a->n()), 256, 0, STREAM>>>(grad(a), y->g, b->v, a->n());
      if (b->track) mulAccK<<<blocks(a->n()), 256, 0, STREAM>>>(grad(b), y->g, a->v, a->n());
    });
    return y;
  }
  // x * s, s a function of sigma
  Node* scale(Node* x, Scalar* s) {
    Node* y = &make(x->rows, x->cols, true);
    scaleK<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, y->v, (float)s->v, x->n());
    backs.push_back([=] {
      if (!y->g) return;
      if (x->track) axpyK<<<blocks(x->n()), 256, 0, STREAM>>>(grad(x), y->g, (float)s->v, x->n());
      s->g += dot(y->g, x->v, x->n());
    });
    return y;
  }
  Node* maskRows(Node* x, const int* mask) {
    Node* y = &make(x->rows, x->cols, x->track);
    maskRowsK<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, y->v, mask, x->rows, (int)x->cols);
    if (x->track) backs.push_back([=] { if (y->g) maskRowsAccK<<<blocks(x->n()), 256, 0, STREAM>>>(grad(x), y->g, mask, x->rows, (int)x->cols); });
    return y;
  }
  // rows index[i] of x (-1: a zero row); `index` on the device, `rows` of them
  Node* gather(Node* x, const int* index, size_t rows) {
    Node* y = &make(rows, x->cols, x->track);
    gatherK<<<blocks(y->n()), 256, 0, STREAM>>>(x->v, y->v, index, rows, (int)x->cols);
    if (x->track) backs.push_back([=] { if (y->g) scatterAccK<<<blocks(y->n()), 256, 0, STREAM>>>(grad(x), y->g, index, rows, (int)x->cols); });
    return y;
  }
  Node* slice(Node* x, int from, int width) {
    Node* y = &make(x->rows, width, x->track);
    copyColsK<<<blocks(y->n()), 256, 0, STREAM>>>(x->v, (int)x->cols, from, y->v, width, 0, x->rows, width);
    if (x->track) backs.push_back([=] {
      if (y->g) accColsK<<<blocks(y->n()), 256, 0, STREAM>>>(grad(x), (int)x->cols, from, y->g, width, 0, x->rows, width);
    });
    return y;
  }
  Node* concat(Node* a, Node* b) {
    const int ca = (int)a->cols, cb = (int)b->cols, c = ca + cb;
    Node* y = &make(a->rows, c, a->track || b->track);
    copyColsK<<<blocks(a->n()), 256, 0, STREAM>>>(a->v, ca, 0, y->v, c, 0, a->rows, ca);
    copyColsK<<<blocks(b->n()), 256, 0, STREAM>>>(b->v, cb, 0, y->v, c, ca, b->rows, cb);
    if (y->track) backs.push_back([=] {
      if (!y->g) return;
      if (a->track) accColsK<<<blocks(a->n()), 256, 0, STREAM>>>(grad(a), ca, 0, y->g, c, 0, a->rows, ca);
      if (b->track) accColsK<<<blocks(b->n()), 256, 0, STREAM>>>(grad(b), cb, 0, y->g, c, ca, b->rows, cb);
    });
    return y;
  }
  // (1 + alpha) LN(x) + beta, alpha/beta columns [oa, oa + C) / [ob, ob + C) of `cond` (x's rows)
  Node* adaLn(Node* x, Node* cond, int oa, int ob) {
    const int C = (int)x->cols;
    Node* y = &make(x->rows, C, x->track || cond->track);
    float* xhat = ARENA.get(x->n()); float* inv = ARENA.get(x->rows); float* tmp = ARENA.get(x->n());
    lnFwdK<<<(unsigned)((x->rows + 7) / 8), 256, 0, STREAM>>>(x->v, tmp, xhat, inv, x->rows, C, nullptr, nullptr);
    adaFwdK<<<blocks(x->n()), 256, 0, STREAM>>>(xhat, cond->v, (int)cond->cols, oa, ob, y->v, x->rows, C);
    if (y->track) backs.push_back([=] {
      if (!y->g) return;
      float* gn = ARENA.get(x->n());
      adaBackK<<<blocks(x->n()), 256, 0, STREAM>>>(y->g, xhat, cond->v, (int)cond->cols, oa, ob, gn,
                                                   cond->track ? grad(cond) : nullptr, x->rows, C);
      if (x->track) lnBackK<<<(unsigned)((x->rows + 7) / 8), 256, 0, STREAM>>>(grad(x), gn, xhat, inv, x->rows, C, nullptr);
    });
    return y;
  }
  // x + gate h, gate columns [og, og + C) of `cond`
  Node* gatedAdd(Node* x, Node* h, Node* cond, int og) {
    const int C = (int)x->cols;
    Node* y = &make(x->rows, C, x->track || h->track || cond->track);
    gateFwdK<<<blocks(x->n()), 256, 0, STREAM>>>(x->v, h->v, cond->v, (int)cond->cols, og, y->v, x->rows, C);
    if (y->track) backs.push_back([=] {
      if (!y->g) return;
      gateBackK<<<blocks(x->n()), 256, 0, STREAM>>>(y->g, h->v, cond->v, (int)cond->cols, og, x->track ? grad(x) : nullptr,
                                                    h->track ? grad(h) : nullptr, cond->track ? grad(cond) : nullptr, x->rows, C);
    });
    return y;
  }
  Node* attention(Node* qkv, const float* qBias, AttnShape a) {
    const int D = a.H * a.HD;
    Node* y = &make(a.R, D, qkv->track);
    float* P = ARENA.get((size_t)a.R * a.H * a.K);
    attnFwdK<<<blocks((size_t)a.R * a.H * 32, 128), 128, 0, STREAM>>>(a, qkv->v, qBias, P, y->v);
    if (qkv->track) backs.push_back([=] {
      if (!y->g) return;
      float* dl = ARENA.get((size_t)a.R * a.H * a.K);
      attnBackQK<<<blocks((size_t)a.R * a.H * 32, 128), 128, 0, STREAM>>>(a, qkv->v, P, y->g, dl, grad(qkv));
      attnBackKVK<<<blocks((size_t)a.R * a.H * 32, 128), 128, 0, STREAM>>>(a, qkv->v, qBias, P, dl, y->g, grad(qkv));
    });
    return y;
  }
  // cos(2 pi (t w + b)), one row, t a function of sigma
  Node* fourier(Scalar* t, const float* w, const float* b, int F) {
    Node* y = &make(1, F, true);
    std::vector<float> hw = download(w, F), hb = download(b, F), hy(F);
    for (int k = 0; k < F; ++k) hy[k] = cosf(2.f * (float)M_PI * ((float)t->v * hw[k] + hb[k]));
    CK(cudaMemcpyAsync(y->v, hy.data(), F * 4, cudaMemcpyHostToDevice, STREAM));
    CK(cudaStreamSynchronize(STREAM));
    backs.push_back([=] {
      if (!y->g) return;
      std::vector<float> g = download(y->g, F);
      double acc = 0;
      for (int k = 0; k < F; ++k) acc += g[k] * -sin(2 * M_PI * (t->v * hw[k] + hb[k])) * 2 * M_PI * hw[k];
      t->g += acc;
    });
    return y;
  }
  double dot(const float* a, const float* b, size_t n) {
    double* d = (double*)ARENA.get(2);
    CK(cudaMemsetAsync(d, 0, 8, STREAM));
    dotK<<<std::min<unsigned>(blocks(n), 1024), 256, 0, STREAM>>>(a, b, n, d);
    double h; CK(cudaMemcpyAsync(&h, d, 8, cudaMemcpyDeviceToHost, STREAM)); CK(cudaStreamSynchronize(STREAM));
    return h;
  }

 private:
  Node& make(size_t rows, size_t cols, bool track) {
    nodes.push_back({ARENA.get(rows * cols), nullptr, rows, cols, track});
    return nodes.back();
  }
};

// ---------------------------------------------------------------- the denoiser on the tape (denoiser.js)
inline const float* lv(const std::string& n, int l) { return PL(SM + n, l); }
inline const float* lvOpt(const std::string& n, int l) { return PLopt(SM + n, l); }
inline const float* pv(const std::string& n) { return P(SM + n); }
inline const float* pvOpt(const std::string& n) { return Popt(SM + n); }

// fc3(silu(fc1 x) * fc2 x) at layer l (no norm), or the unstacked module with its norm
inline Node* swiglu(Tape& t, Node* x, const std::string& p, int l) {
  auto w = [&](const char* n) { return l < 0 ? pv(p + n) : lv(p + n, l); };
  const int I = l < 0 ? dimP(SM + p + "fc1/weight", 0) : dimP(SM + p + "fc1/weight", 1);
  const int Co = l < 0 ? dimP(SM + p + "fc3/weight", 0) : dimP(SM + p + "fc3/weight", 1);
  return t.linear(t.mul(t.silu(t.linear(x, w("fc1/weight"), nullptr, I)), t.linear(x, w("fc2/weight"), nullptr, I)),
                  w("fc3/weight"), nullptr, Co);
}
// a layer's conditioning on the conditioning rows: cond_2(silu(LN(c)))
inline Node* layerCondition(Tape& t, Node* c, const std::string& p, int l, int width) {
  return t.linear(t.silu(t.layerNorm(c, lv(p + "cond_0/weight", l), lvOpt(p + "cond_0/bias", l))),
                  lv(p + "cond_2/weight", l), lvOpt(p + "cond_2/bias", l), width);
}
// ConditionedTransformerLayer: x + g_a attn(adaLN) ; x + g_m transition(adaLN)
inline Node* conditioned(Tape& t, Node* x, Node* cond, const std::string& p, const std::string& att, int l, AttnShape a) {
  const int D = (int)x->cols;
  Node* h = t.adaLn(x, cond, 0, D);
  Node* qkv = t.linear(h, lv(p + att + "proj_qkv/weight", l), nullptr, 3 * D);
  Node* g = t.sigmoid(t.linear(h, lv(p + att + "proj_g/weight", l), nullptr, D));
  Node* o = t.attention(qkv, lv(p + att + "proj_q_bias", l), a);
  Node* ao = t.linear(t.mul(g, o), lv(p + att + "proj_o/weight", l), nullptr, D);
  x = t.gatedAdd(x, ao, cond, 2 * D);
  Node* m = swiglu(t, t.adaLn(x, cond, 3 * D, 4 * D), p + "transition/", l);
  return t.gatedAdd(x, m, cond, 5 * D);
}

// 🔴 THE VJP READS A FLOAT32 STATIC CONTEXT, EVEN UNDER --fast. The pair biases, q0 and the conditioner's static half
// built on the f16 tensor cores move the forward by ~1e-3 - inside its bound - and the network's part of d/dx by 2%
// at sigma 16, where it nearly cancels c_skip w: measured against jax.grad, 2.0e-2 fast against 1.6e-4 float32.
inline Denoiser exactContext(const float* sTrunk, const float* zTrunk, const float* sInputs, int Si) {
  cublasMath_t mode; CB(cublasGetMathMode(H, &mode)); CB(cublasSetMathMode(H, CUBLAS_PEDANTIC_MATH));
  const bool half = GEMM_HALF; GEMM_HALF = false;
  Denoiser d = staticContext(sTrunk, zTrunk, sInputs, Si, nullptr);
  CK(cudaStreamSynchronize(STREAM));
  GEMM_HALF = half; CB(cublasSetMathMode(H, mode));
  return d;
}

struct VjpResult { std::vector<float> den, gx; double gs = 0; size_t bytes = 0; };

// D(x, sigma) of one particle and, with a cotangent w (device, [N, 3]) - or one made from D by `seedOf` once the
// forward has run (edm_fkc's: the potentials' gradient at D) - w . dD/dx and w . dD/dsigma
using SeedOf = std::function<std::vector<float>(const std::vector<float>&)>;
inline VjpResult denoiseVjp(const Denoiser& d, const float* xDev, double sigma, const float* wDev, const float* condDev = nullptr,
                            const SeedOf& seedOf = nullptr) {
  ARENA.reset();
  cublasMath_t mode; CB(cublasGetMathMode(H, &mode)); CB(cublasSetMathMode(H, CUBLAS_PEDANTIC_MATH));
  const bool half = GEMM_HALF; GEMM_HALF = false;
  const int T = d.T, N = d.N, Ca = 128, Dt = 768, Dc = Dt + Ca;
  const double sd = SIGMA_DATA;
  Tape t;
  Scalar cIn{1 / sqrt(sigma * sigma + sd * sd), -sigma / pow(sigma * sigma + sd * sd, 1.5)};
  Scalar tt{0.25 * log(sigma / sd), 0.25 / sigma};
  Scalar cSkip{sd * sd / (sigma * sigma + sd * sd), -2 * sigma * sd * sd / pow(sigma * sigma + sd * sd, 2)};
  Scalar cOut{sigma * sd / sqrt(sd * sd + sigma * sigma), pow(sd, 3) / pow(sd * sd + sigma * sigma, 1.5)};
  Node* x = t.leaf(xDev, N, 3, true);
  Node* r = t.maskRows(t.scale(x, &cIn), Idev("atom_mask"));
  // the single conditioning
  const std::string sc = "single_conditioner/";
  const int F = dimP(SM + sc + "fourier_embed/proj/bias", 0);
  Node* four = t.layerNorm(t.fourier(&tt, pv(sc + "fourier_embed/proj/weight"), pv(sc + "fourier_embed/proj/bias"), F),
                           pv(sc + "norm_fourier/weight"), pv(sc + "norm_fourier/bias"));
  // (cond, and a zero row after its T: what a padding atom conditions on - its rows of every layer's conditioning)
  float* sc0z = ARENA.get((size_t)(T + 1) * Dt);
  CK(cudaMemcpyAsync(sc0z, d.sc0, (size_t)T * Dt * 4, cudaMemcpyDeviceToDevice, STREAM));
  CK(cudaMemsetAsync(sc0z + (size_t)T * Dt, 0, Dt * 4, STREAM));
  // (the zero row stays zero: its fourier term is added to the T real rows only)
  Node* s0 = t.leaf(sc0z, T + 1, Dt);
  Node* fr = t.linear(four, pv(sc + "fourier_to_single/weight"), nullptr, Dt);
  std::vector<int> realRow(T + 1, 1); realRow[T] = 0;
  int* realRows = (int*)ARENA.get(T + 1);
  CK(cudaMemcpyAsync(realRows, realRow.data(), (T + 1) * 4, cudaMemcpyHostToDevice, STREAM));
  std::vector<int> zeros(T + 1, 0);
  int* bcast = (int*)ARENA.get(T + 1);
  CK(cudaMemcpyAsync(bcast, zeros.data(), (T + 1) * 4, cudaMemcpyHostToDevice, STREAM));
  Node* s = t.add(s0, t.maskRows(t.gather(fr, bcast, T + 1), realRows));
  for (int i = 0; i < 2; ++i) {
    const std::string p = sc + "transitions_" + std::to_string(i) + "/";
    Node* n = t.layerNorm(s, pv(p + "norm/weight"), pv(p + "norm/bias"));
    s = t.add(s, t.maskRows(swiglu(t, n, p, -1), realRows));
  }
  Node* cond = s;                                          // [T + 1, 768]
  // the atom stream's initial state
  const std::string af = "atom_feature_embedder/";
  Node* q = t.add(t.leaf(d.q0, N, Ca), t.linear(r, pv(af + "r_to_q_trans/weight"), nullptr, Ca));
  if (d.dyn) {
    float* cq = ARENA.get((size_t)N * Ca);
    linear(condDev ? condDev : d.coordsCond, pv(af + "cond_to_q_trans/weight"), cq, N, 3, Ca);
    scaleK<<<blocks((size_t)N * Ca), 256, 0, STREAM>>>(cq, cq, d.dynGate, (size_t)N * Ca);
    q = t.add(q, t.leaf(cq, N, Ca));
  }
  Node* a = t.add(t.leaf(d.a0, T, Dt), t.linear(t.maskRows(t.gather(r, Idev("token_center_atom_index"), T), Idev("token_pad_mask")),
                                                         pv("r_token_to_a/weight"), nullptr, Dt));
  // each atom's conditioning row (a padding atom: the zero row T)
  const int* tok = M.i("token_idx_at_atom");
  const int* apad = M.i("atom_pad_mask");
  std::vector<int> atomRow(N), atomTok(N);
  for (int i = 0; i < N; ++i) { atomRow[i] = apad[i] ? tok[i] : T; atomTok[i] = apad[i] ? tok[i] : -1; }
  int* atomRows = (int*)ARENA.get(N); int* atomToks = (int*)ARENA.get(N);
  CK(cudaMemcpyAsync(atomRows, atomRow.data(), N * 4, cudaMemcpyHostToDevice, STREAM));
  CK(cudaMemcpyAsync(atomToks, atomTok.data(), N * 4, cudaMemcpyHostToDevice, STREAM));
  // the token rows of cond (its first T)
  std::vector<int> ident(T); for (int i = 0; i < T; ++i) ident[i] = i;
  int* identRows = (int*)ARENA.get(T);
  CK(cudaMemcpyAsync(identRows, ident.data(), T * 4, cudaMemcpyHostToDevice, STREAM));
  Node* condTok = t.gather(cond, identRows, T);
  const std::string L = "layers/", ab = "attention/attention_layer/pair_bias_attn/", tb = "attention/pair_bias_attn/";
  const size_t Pn = (size_t)T * T, wb = (size_t)N * WK * 4, tbn = 16 * Pn;
  float* tbias = d.tbias ? nullptr : ARENA.get(tbn);
  for (int l = 0; l < d.layers; ++l) {
    const std::string ap = L + "atom_block/transformer_layer/", tp = L + "token_block/";
    Node* condA = t.gather(layerCondition(t, cond, ap, l, 6 * Ca), atomRows, N);
    q = conditioned(t, q, condA, ap, ab, l, AttnShape{true, N, 4, 32, WK, d.wbias + wb * l, Idev("atom_pad_mask")});
    const float* bias = d.tbias ? d.tbias + tbn * l : tbias;
    if (!d.tbias) tokenBias(d.z, PL(SM + tp + tb + "norm_z/weight", l), PL(SM + tp + tb + "proj_z/weight", l), tbias, T, 256, 16);
    Node* cT = layerCondition(t, condTok, tp, l, 6 * Dt);
    a = conditioned(t, a, cT, tp, tb, l, AttnShape{false, T, 16, 48, T, bias, Idev("token_pad_mask")});
    // the coupling
    const std::string cp = L + "coupling/";
    Node* qc = t.gather(t.linear(t.silu(t.layerNorm(cond, lv(cp + "q_coupling_cond_0/weight", l), lvOpt(cp + "q_coupling_cond_0/bias", l))),
                                 lv(cp + "q_coupling_cond_2/weight", l), lvOpt(cp + "q_coupling_cond_2/bias", l), 2 * Dc + Ca), atomRows, N);
    Node* scn = t.linear(t.silu(t.layerNorm(condTok, lv(cp + "s_coupling_cond_0/weight", l), lvOpt(cp + "s_coupling_cond_0/bias", l))),
                         lv(cp + "s_coupling_cond_2/weight", l), lvOpt(cp + "s_coupling_cond_2/bias", l), 2 * Dc + Dt);
    Node* sqIn = t.concat(t.gather(a, atomToks, N), q);
    Node* sToQ = swiglu(t, t.adaLn(sqIn, qc, 0, Dc), cp + "s_to_q_mixer/", l);
    Node* qsIn = t.concat(t.gather(q, Idev("token_center_atom_index"), T), a);
    Node* qToS = swiglu(t, t.adaLn(qsIn, scn, 0, Dc), cp + "q_to_s_mixer/", l);
    a = t.gatedAdd(a, qToS, scn, 2 * Dc);
    q = t.gatedAdd(q, sToQ, qc, 2 * Dc);
  }
  // the readout
  const std::string ro = "readout/";
  const int Hd = dimP(SM + ro + "q_to_hidden/weight", 0);
  Node* h = t.add(t.linear(q, pv(ro + "q_to_hidden/weight"), nullptr, Hd),
                  t.gather(t.linear(a, pv(ro + "s_to_hidden/weight"), nullptr, Hd), atomToks, N));
  Node* out = t.linear(t.layerNorm(h, pv(ro + "to_out_0/weight")), pv(ro + "to_out_1/weight"), nullptr, 3);
  Node* den = t.add(t.scale(x, &cSkip), t.scale(out, &cOut));
  VjpResult res;
  res.den = download(den->v, (size_t)N * 3);
  if (seedOf) {
    std::vector<float> seed = seedOf(res.den);
    float* sd = ARENA.get(seed.size());
    CK(cudaMemcpyAsync(sd, seed.data(), seed.size() * 4, cudaMemcpyHostToDevice, STREAM));
    wDev = sd;
  }
  if (wDev) {
    t.backward(den, wDev);
    res.gx = download(t.grad(x), (size_t)N * 3);
    for (Scalar* s : {&cIn, &tt, &cSkip, &cOut}) res.gs += s->g * s->dv;
  }
  CK(cudaStreamSynchronize(STREAM));
  res.bytes = ARENA.high;
  GEMM_HALF = half;
  CB(cublasSetMathMode(H, mode));
  return res;
}

}  // namespace be2vjp
