// The per-atom conditioning and the atom cross-attention encoder, which builds the
// 384 atom columns of target_feat and runs again inside every denoiser call.
// Transcribed from src/af3/diffusion/{atom-conditioning,atom-encoder}-reference.js. f32.
#pragma once
#include "pairtrack.cuh"

struct Gather { const int* idx; const float* mask; int count; };
inline Gather gatherOf(const std::string& name) {
  return { Idev(name + ".indices"), Fdev(name + ".mask"), (int)M.len(name + ".indices") };
}
// Diffusion samples in flight through the denoiser's per-step path: every per-sample tensor is
// NS copies, sample-major, and what the samples share (the conditioning, masks, the attention
// biases, the adaptive LayerNorms' scales) is read at (row % one sample's rows).
inline int NS = 1;
// out[index] = mask ? source[indices[index]] : 0; over `ns` samples, sample k reading source rows
// offset by k * srcRows
__global__ void convertK(const int* idx, const float* mask, const float* src, float* out, int count, int C,
                         size_t srcRows, int ns) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)ns * count * C) return;
  size_t i = t / C; int c = (int)(t % C);
  int g = (int)(i % count); size_t k = i / count;
  out[t] = mask[g] != 0 ? src[((size_t)idx[g] + k * srcRows) * C + c] : 0.f;
}
inline void convert(const Gather& g, const float* src, float* out, int C, size_t srcRows = 0, int ns = 1) {
  convertK<<<blocks((size_t)ns * g.count * C), 256, 0, STREAM>>>(g.idx, g.mask, src, out, g.count, C, srcRows, ns);
}
// LayerNorm with the TWO-PASS variance (this module's convention), optional scale/offset.
__global__ void layerNormSlowK(const float* in, float* out, size_t rows, int C, const float* scale,
                               const float* offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* x = in + row * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += x[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = x[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32)
    out[row * C + c] = (x[c] - mean) * inv * (scale ? scale[c] : 1.f) + (offset ? offset[c] : 0.f);
}
inline void layerNormSlow(const float* in, float* out, size_t rows, int C, const float* scale,
                          const float* offset) {
  layerNormSlowK<<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(in, out, rows, C, scale, offset);
}
inline const float* Wopt(const std::string& k) { return hasW(k) ? W(k) : nullptr; }

// ---------------------------------------------------------------- per-atom conditioning
__global__ void perAtomK(const float* pos, const float* mask, const int* element, const float* charge,
                         const int* nameChars, const float* Wpos, const float* Wmask, const float* Welem,
                         const float* Wcharge, const float* Wname, const float* bias, float* act,
                         int rows, int C, bool rawCharge) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)rows * C) return;
  int r = (int)(t / C), c = (int)(t % C);
  float p0 = pos[r * 3], p1 = pos[r * 3 + 1], p2 = pos[r * 3 + 2];
  float m = mask[r], ch = rawCharge ? charge[r] : asinhf(charge[r]);
  float v = ((p0 * Wpos[c]) + p1 * Wpos[C + c]) + p2 * Wpos[2 * C + c];
  if (bias) v += bias[c];
  v += m * Wmask[c];
  int z = element[r];
  if (z >= 0 && z < 128) v += Welem[(size_t)z * C + c];
  v += ch * Wcharge[c];
  float name = 0;
  for (int k = 0; k < 4; ++k) {
    int code = nameChars[r * 4 + k];
    if (code >= 0 && code < 64) name += Wname[(size_t)(k * 64 + code) * C + c];
  }
  v += name;
  act[t] = v * m;
}
// The reference embedding's prefix is "targetFeat.reference" for target_feat and
// "atomReference" for the diffusion head; the two are the same tensors.
inline float* perAtomConditioning(const std::string& ref, int rows) {
  int C = (int)M.meta(ref + ".channels");
  float* act = scratch<float>(ref + ".cond", (size_t)rows * C);
  perAtomK<<<blocks((size_t)rows * C), 256, 0, STREAM>>>(
    Fdev("batch.refPos"), Fdev("batch.refMask"), Idev("batch.refElement"), Fdev("batch.refCharge"),
    Idev("batch.refAtomNameChars"), W(ref + ".embedRefPos"), W(ref + ".embedRefMask"),
    W(ref + ".embedRefElement"), W(ref + ".embedRefCharge"), W(ref + ".embedRefAtomName"),
    Wopt(ref + ".embedAtomFeaturesBias"), act, rows, C, M.flag("trunk.dialect.rawRefCharge"));
  return act;
}

// ---------------------------------------------------------------- the encoder's pieces
// x[r] *= m[r % period]
__global__ void scaleByRowK(float* x, const float* m, size_t rows, int C, size_t period) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < rows * C) x[t] *= m[(t / C) % period];
}
__global__ void reluK(const float* x, float* y, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = x[i] > 0 ? x[i] : 0;
}
__global__ void adaLnCombineK(const float* xn, const float* scale, const float* shift, float* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sigm(scale[i]) * xn[i] + shift[i];
}
__global__ void mulSigmoidK(float* x, const float* g, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] *= sigm(g[i]);
}
__global__ void addBiasRowsK(float* y, const float* b, size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < rows * C) y[i] += b[i % C];
}
// sigmoid(cond' W + b) * LN(x) + cond' Wshift, cond' = LN_scale_only(cond)
inline void adaptiveLayerNorm(const float* x, const float* cond, float* out, size_t rows, int C, int condC,
                              const std::string& w) {
  float* xn = scratch<float>("ada.xn", rows * C);
  float* cn = scratch<float>("ada.cn", rows * condC);
  float* sc = scratch<float>("ada.scale", rows * C);
  float* sh = scratch<float>("ada.shift", rows * C);
  layerNormSlow(x, xn, rows, C, nullptr, nullptr);
  layerNormSlow(cond, cn, rows, condC, W(w + "SingleCondLayerNormScale"), nullptr);
  linear<float, float>(cn, sc, rows, condC, C, w + "SingleCondScaleWeights");
  addBiasRowsK<<<blocks(rows * C), 256, 0, STREAM>>>(sc, W(w + "SingleCondScaleBias"), rows, C);
  linear<float, float>(cn, sh, rows, condC, C, w + "SingleCondBias");
  adaLnCombineK<<<blocks(rows * C), 256, 0, STREAM>>>(xn, sc, sh, out, rows * C);
}

// One subset's attention: 32 queries against 128 keys, one warp a query, lanes over keys.
// logits = q k^T / sqrt(d) + mask + pair logits; out = softmax . v
__global__ void atomAttentionK(const float* q, const float* k, const float* v, const float* qMask,
                               const float* kMask, const float* pairLogits, float* out, int subsets,
                               int queries, int keys, int heads, int D, bool keyMasked) {
  int s = blockIdx.x, h = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  int nw = blockDim.x / 32, Wd = heads * D;
  extern __shared__ float sm[];
  float* Ks = sm; float* Vs = sm + keys * (D + 1); float* P = Vs + keys * (D + 1);   // P: nw x keys
  for (int t = threadIdx.x; t < keys * D; t += blockDim.x) {
    int key = t / D, e = t % D; size_t row = (size_t)s * keys + key;
    Ks[key * (D + 1) + e] = k[row * Wd + h * D + e];
    Vs[key * (D + 1) + e] = v[row * Wd + h * D + e];
  }
  __syncthreads();
  float scale = 1.f / sqrtf((float)D);
  for (int qi = warp; qi < queries; qi += nw) {
    size_t qrow = (size_t)s * queries + qi;
    const float* qp = q + qrow * Wd + h * D;
    float* Pw = P + warp * keys;
    float mx = -INFINITY;
    for (int key = lane; key < keys; key += 32) {
      float dot = 0;
      for (int e = 0; e < D; ++e) dot += qp[e] * Ks[key * (D + 1) + e];
      size_t krow = (size_t)s * keys + key;
      float maskBias = keyMasked ? -1e9f * ((1.f - qMask[qrow]) + (1.f - kMask[krow]))
                                 : 1e9f * (qMask[qrow] - 1.f) * (kMask[krow] - 1.f);
      float l = dot * scale + maskBias + pairLogits[(((size_t)s * heads + h) * queries + qi) * keys + key];
      Pw[key] = l; mx = fmaxf(mx, l);
    }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
    float sum = 0;
    for (int key = lane; key < keys; key += 32) { float e = expf(Pw[key] - mx); Pw[key] = e; sum += e; }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    __syncwarp();
    for (int e = lane; e < D; e += 32) {
      float acc = 0;
      for (int key = 0; key < keys; ++key) acc += Pw[key] * Vs[key * (D + 1) + e];
      out[qrow * Wd + h * D + e] = acc / sum;
    }
    __syncwarp();
  }
}
// out = act + attention + projected * sigmoid(gate)
__global__ void atomBlockOutK(const float* act, const float* attention, const float* projected,
                              const float* gate, float* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = act[i] + attention[i] + projected[i] * sigm(gate[i]);
}

struct AtomShape { int tokens, dense, subsets, queries, keys; };
inline AtomShape atomShape() {
  return { (int)M.meta("batch.shape.tokens"), (int)M.meta("batch.shape.dense"), (int)M.meta("batch.shape.subsets"),
           (int)M.meta("batch.shape.queries"), (int)M.meta("batch.shape.keys") };
}

// Everything a cross-attention block derives from the conditioning, which is the same at
// every denoiser step: the adaptive LayerNorms' scales and shifts and both zero-init gates.
struct AtomBlockCache {
  float *qScale, *qShift, *kScale, *kShift, *zg, *ffwScale, *ffwShift, *tg, *pairLogits;
  bool chained = false;     // the keys' adaptive LN reads the normalised queries (OpenDDE, protenix2)
};
// scale = LN_s(cond) W + b and shift = LN_s(cond) W' for one adaptive LayerNorm
inline void adaCond(const float* cond, size_t rows, int C, int condC, const std::string& w, float* scale, float* shift) {
  float* cn = scratch<float>("ada.cn", rows * condC);
  layerNormSlow(cond, cn, rows, condC, W(w + "SingleCondLayerNormScale"), nullptr);
  linear<float, float>(cn, scale, rows, condC, C, w + "SingleCondScaleWeights");
  addBiasRowsK<<<blocks(rows * C), 256, 0, STREAM>>>(scale, W(w + "SingleCondScaleBias"), rows, C);
  linear<float, float>(cn, shift, rows, condC, C, w + "SingleCondBias");
}
// The key side's scale and shift are per ATOM (a key row is a gathered query row), so they are
// computed over the query rows and the block gathers the projected keys and values instead.
inline AtomBlockCache prepareAtomBlock(const std::string& B, const float* qCond, size_t qRows, int C,
                                       float* pairLogits) {
  AtomBlockCache c{};
  auto a = [&](const std::string& tag, size_t n) { return scratch<float>(B + tag, n); };
  c.qScale = a(".qs", qRows * C); c.qShift = a(".qh", qRows * C);
  c.kScale = a(".ks", qRows * C); c.kShift = a(".kh", qRows * C);
  c.ffwScale = a(".fs", qRows * C); c.ffwShift = a(".fh", qRows * C);
  c.zg = a(".zg", qRows * C); c.tg = a(".tg", qRows * C);
  adaCond(qCond, qRows, C, C, B + ".q", c.qScale, c.qShift);
  adaCond(qCond, qRows, C, C, B + ".k", c.kScale, c.kShift);
  adaCond(qCond, qRows, C, C, B + ".ffw", c.ffwScale, c.ffwShift);
  linear<float, float>(qCond, c.zg, qRows, C, C, B + ".AdaptiveZeroCondWeights");
  addBiasRowsK<<<blocks(qRows * C), 256, 0, STREAM>>>(c.zg, W(B + ".AdaptiveZeroCondBias"), qRows, C);
  linear<float, float>(qCond, c.tg, qRows, C, C, B + ".ffwAdaptiveZeroCondWeights");
  addBiasRowsK<<<blocks(qRows * C), 256, 0, STREAM>>>(c.tg, W(B + ".ffwAdaptiveZeroCondBias"), qRows, C);
  c.pairLogits = pairLogits;
  c.chained = M.flag(B + ".chainedAtomLayerNorm");
  return c;
}
// sigmoid(scale) * LN(x) + shift, LN without affine
template <class TO>
__global__ void adaLnK(const float* x, const float* scale, const float* shift, TO* out, size_t rows, int C,
                       size_t period) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) {
    size_t k = (row % period) * C + c;
    out[row * C + c] = fromF<TO>(sigm(scale[k]) * ((xr[c] - mean) * inv) + shift[k]);
  }
}
template <class TO>
inline void adaLn(const float* x, const float* scale, const float* shift, TO* out, size_t rows, int C,
                  size_t period) {
  adaLnK<TO><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(x, scale, shift, out, rows, C, period);
}

template <class TO>
__global__ void adaLn2K(const float* x, const float* s1, const float* h1, const float* s2, const float* h2, TO* o1, TO* o2,
                        size_t rows, int C, size_t period) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * C;
  if (C == 128) {           // a float4 a lane, every load issued at once (few blocks: latency is the cost)
    size_t k = (row % period) * C + lane * 4;
    float4 v = *(const float4*)(xr + lane * 4), a = *(const float4*)(s1 + k), b = *(const float4*)(h1 + k),
           c2 = *(const float4*)(s2 + k), d = *(const float4*)(h2 + k);
    float sum = v.x + v.y + v.z + v.w;
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    float mean = sum / C, q = (v.x - mean) * (v.x - mean) + (v.y - mean) * (v.y - mean) + (v.z - mean) * (v.z - mean) + (v.w - mean) * (v.w - mean);
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
    float inv = 1.f / sqrtf(q / C + 1e-5f);
    float n[4] = { (v.x - mean) * inv, (v.y - mean) * inv, (v.z - mean) * inv, (v.w - mean) * inv };
    float as[4] = { a.x, a.y, a.z, a.w }, bs[4] = { b.x, b.y, b.z, b.w }, cs[4] = { c2.x, c2.y, c2.z, c2.w }, ds[4] = { d.x, d.y, d.z, d.w };
    for (int e = 0; e < 4; ++e) {
      o1[row * C + lane * 4 + e] = fromF<TO>(sigm(as[e]) * n[e] + bs[e]);
      o2[row * C + lane * 4 + e] = fromF<TO>(sigm(cs[e]) * n[e] + ds[e]);
    }
    return;
  }
  float s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) {
    size_t k = (row % period) * C + c; float n = (xr[c] - mean) * inv;
    o1[row * C + c] = fromF<TO>(sigm(s1[k]) * n + h1[k]);
    o2[row * C + c] = fromF<TO>(sigm(s2[k]) * n + h2[k]);
  }
}
struct AtomStep {
  Gather queriesToKeys;
  const float *qMask, *kMask;
  bool keyMasked, noResidual;
};
// [a | b] as one (C, 2W) matrix, both stored (in, out)
inline std::string pairedWeight(const std::string& a, const std::string& b, int C, int Wd) {
  return concatColumns(a + "|" + b + "~", C, {{a, Wd, false}, {b, Wd, false}});
}
// a key row gathered from the queries (zero where masked), then its adaptive LN; warp per row
template <class TO>
__global__ void gatherAdaLnK(const float* act, const int* idx, const float* gmask, const float* scale,
                             const float* shift, TO* out, size_t rows, int C) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  bool live = gmask[row] != 0;
  const float* xr = act + (size_t)idx[row] * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += live ? xr[c] : 0.f;
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = (live ? xr[c] : 0.f) - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) {
    size_t k = row * C + c;
    out[k] = fromF<TO>(sigm(scale[k]) * (((live ? xr[c] : 0.f) - mean) * inv) + shift[k]);
  }
}
// act += y * sigmoid(gate); out = sigmoid(scale) * LN(act) + shift; warp per row
template <class TO>
__global__ void gatedAddAdaLnRowsK(float* act, const float* y, const float* gate, const float* scale,
                                   const float* shift, TO* out, size_t rows, int C, size_t period) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  float* a = act + row * C;
  float s = 0;
  size_t pr = (row % period) * C;
  if (C == 128) {           // a float4 a lane, every load issued at once (few blocks: latency is the cost)
    int c0 = lane * 4;
    float4 x = *(const float4*)(a + c0), yy = *(const float4*)(y + row * C + c0), g = *(const float4*)(gate + pr + c0),
           sc = *(const float4*)(scale + pr + c0), sh = *(const float4*)(shift + pr + c0);
    x.x += yy.x * sigm(g.x); x.y += yy.y * sigm(g.y); x.z += yy.z * sigm(g.z); x.w += yy.w * sigm(g.w);
    *(float4*)(a + c0) = x;
    float sum = x.x + x.y + x.z + x.w;
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    float mean = sum / C, q = (x.x - mean) * (x.x - mean) + (x.y - mean) * (x.y - mean) + (x.z - mean) * (x.z - mean) + (x.w - mean) * (x.w - mean);
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
    float inv = 1.f / sqrtf(q / C + 1e-5f);
    TO* o = out + row * C + c0;
    o[0] = fromF<TO>(sigm(sc.x) * ((x.x - mean) * inv) + sh.x); o[1] = fromF<TO>(sigm(sc.y) * ((x.y - mean) * inv) + sh.y);
    o[2] = fromF<TO>(sigm(sc.z) * ((x.z - mean) * inv) + sh.z); o[3] = fromF<TO>(sigm(sc.w) * ((x.w - mean) * inv) + sh.w);
    return;
  }
  for (int c = lane; c < C; c += 32) { float v = a[c] + y[row * C + c] * sigm(gate[pr + c]); a[c] = v; s += v; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  __syncwarp();
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = a[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) out[row * C + c] = fromF<TO>(sigm(scale[pr + c]) * ((a[c] - mean) * inv) + shift[pr + c]);
}
__global__ void addSigmoidGatedK(float* x, const float* y, const float* gate, size_t n, size_t period) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += y[i] * sigm(gate[i % period]);
}
// One subset's attention, q (+bias) and the gate read out of qg [rows][2W], k and v out of
// kv [rows][2W]; writes the gated attention [rows][W]. One warp a query, lanes over keys.
template <class T>
__global__ void atomAttentionFusedK(const T* qg, const float* qBias, const T* kv, const float* qMask,
                                    const float* kMask, const float* pairLogits, T* out, int queries,
                                    int keys, int heads, int D, bool keyMasked, int subsets) {
  int s = blockIdx.x, ss = s % subsets, h = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  int nw = blockDim.x / 32, Wd = heads * D, W2 = 2 * Wd;
  extern __shared__ float sm[];
  float* Ks = sm; float* Vs = sm + keys * (D + 1); float* P = Vs + keys * (D + 1); float* Q = P + nw * keys;
  for (int t = threadIdx.x; t < keys * D; t += blockDim.x) {
    int key = t / D, e = t % D; size_t row = (size_t)s * keys + key;
    Ks[key * (D + 1) + e] = toF(kv[row * W2 + h * D + e]);
    Vs[key * (D + 1) + e] = toF(kv[row * W2 + Wd + h * D + e]);
  }
  __syncthreads();
  float scale = 1.f / sqrtf((float)D);
  for (int qi = warp; qi < queries; qi += nw) {
    size_t qrow = (size_t)s * queries + qi;
    float* Qw = Q + warp * D;
    for (int e = lane; e < D; e += 32) Qw[e] = toF(qg[qrow * W2 + h * D + e]) + qBias[h * D + e];
    __syncwarp();
    float* Pw = P + warp * keys;
    float mx = -INFINITY;
    for (int key = lane; key < keys; key += 32) {
      float dot = 0;
      for (int e = 0; e < D; ++e) dot += Qw[e] * Ks[key * (D + 1) + e];
      size_t krow = (size_t)ss * keys + key, qm = (size_t)ss * queries + qi;
      float maskBias = keyMasked ? -1e9f * ((1.f - qMask[qm]) + (1.f - kMask[krow]))
                                 : 1e9f * (qMask[qm] - 1.f) * (kMask[krow] - 1.f);
      float l = dot * scale + maskBias + pairLogits[(((size_t)ss * heads + h) * queries + qi) * keys + key];
      Pw[key] = l; mx = fmaxf(mx, l);
    }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
    float sum = 0;
    for (int key = lane; key < keys; key += 32) { float e = expf(Pw[key] - mx); Pw[key] = e; sum += e; }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    __syncwarp();
    for (int e = lane; e < D; e += 32) {
      float acc = 0;
      for (int key = 0; key < keys; ++key) acc += Pw[key] * Vs[key * (D + 1) + e];
      out[qrow * Wd + h * D + e] = fromF<T>(acc / sum * sigm(toF(qg[qrow * W2 + Wd + h * D + e])));
    }
    __syncwarp();
  }
}

// The same attention on the tensor cores, for the f16 path at D = 32, 128 keys, 32 queries:
// two warps of 16 queries a (subset, head); S = Q K^T in registers (mma.sync m16n8k16), the pair
// logits and mask added there, the softmax over the 128 keys in registers, O = P V with P's
// accumulators reused as the A operand, the gate applied on the way out.
template <int D, int KEYS>
__global__ void __launch_bounds__(64) atomAttentionMMA(const half* qg, const float* qBias, const half* kv,
    const float* qMask, const float* kMask, const float* pairLogits, half* out, int queries, int heads,
    bool keyMasked, float scale, int subsets) {
  constexpr int LD = D + 8;
  __shared__ __align__(16) half Ks[KEYS * LD];
  __shared__ __align__(16) half Vs[KEYS * LD];
  int s = blockIdx.x, h = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int Wd = heads * D, W2 = 2 * Wd;
  // every load issued before anything waits (a block is two warps and few blocks run: each round
  // trip was exposed): K and V asynchronously, then Q, the gates, and S's starting value - the pair
  // logits and the mask - while they land
  for (int t = threadIdx.x; t < KEYS * (D / 8) * 2; t += 64) {
    int which = t / (KEYS * (D / 8)), u = t % (KEYS * (D / 8)), key = u / (D / 8), c = (u % (D / 8)) * 8;
    cpAsync16((which ? Vs : Ks) + key * LD + c, kv + ((size_t)s * KEYS + key) * W2 + which * Wd + h * D + c, true);
  }
  asm volatile("cp.async.commit_group;");
  int q0 = warp * 16, r0 = q0 + g, r1 = r0 + 8;
  size_t i0 = (size_t)s * queries + r0, i1 = (size_t)s * queries + r1;
  float qs = scale * LOG2E;
  auto q2 = [&](size_t i, int e) -> uint32_t {
    float2 f = __half22float2(*reinterpret_cast<const half2*>(qg + i * W2 + h * D + e));
    return pack2((f.x + qBias[h * D + e]) * qs, (f.y + qBias[h * D + e + 1]) * qs);
  };
  uint32_t qa[D / 16][4];
  for (int ks = 0; ks < D / 16; ++ks) {
    int e = ks * 16 + tig * 2;
    qa[ks][0] = q2(i0, e); qa[ks][1] = q2(i1, e); qa[ks][2] = q2(i0, e + 8); qa[ks][3] = q2(i1, e + 8);
  }
  half2 ga[D / 8], gb[D / 8];
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    ga[et] = *reinterpret_cast<const half2*>(qg + i0 * W2 + Wd + h * D + e);
    gb[et] = *reinterpret_cast<const half2*>(qg + i1 * W2 + Wd + h * D + e);
  }
  int ss = s % subsets;                     // the subset within its sample: masks and logits are shared
  const float* pl0 = pairLogits + (((size_t)ss * heads + h) * queries + r0) * KEYS;
  const float* pl1 = pairLogits + (((size_t)ss * heads + h) * queries + r1) * KEYS;
  float qm0 = qMask[(size_t)ss * queries + r0], qm1 = qMask[(size_t)ss * queries + r1];
  float sv[KEYS / 8][4];
  for (int nt = 0; nt < KEYS / 8; ++nt) {
    int key = nt * 8 + tig * 2;
    float2 b0 = *reinterpret_cast<const float2*>(pl0 + key), b1 = *reinterpret_cast<const float2*>(pl1 + key);
    float km0 = kMask[(size_t)ss * KEYS + key], km1 = kMask[(size_t)ss * KEYS + key + 1];
    auto mb = [&](float qm, float km) {
      return keyMasked ? -1e9f * ((1.f - qm) + (1.f - km)) : 1e9f * (qm - 1.f) * (km - 1.f);
    };
    sv[nt][0] = (b0.x + mb(qm0, km0)) * LOG2E; sv[nt][1] = (b0.y + mb(qm0, km1)) * LOG2E;
    sv[nt][2] = (b1.x + mb(qm1, km0)) * LOG2E; sv[nt][3] = (b1.y + mb(qm1, km1)) * LOG2E;
  }
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  float m0 = -INFINITY, m1 = -INFINITY;
  for (int nt = 0; nt < KEYS / 8; ++nt) {
    uint32_t kb[4];
    ldsm4(kb, Ks + (nt * 8 + (lane & 7)) * LD + (lane >> 3) * 8);
    mma16816(sv[nt], qa[0], kb[0], kb[1]);
    mma16816(sv[nt], qa[1], kb[2], kb[3]);
    m0 = fmaxf(m0, fmaxf(sv[nt][0], sv[nt][1])); m1 = fmaxf(m1, fmaxf(sv[nt][2], sv[nt][3]));
  }
  m0 = fmaxf(m0, __shfl_xor_sync(~0u, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(~0u, m0, 2));
  m1 = fmaxf(m1, __shfl_xor_sync(~0u, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(~0u, m1, 2));
  float l0 = 0, l1 = 0;
  for (int nt = 0; nt < KEYS / 8; ++nt) {
    sv[nt][0] = exp2f(sv[nt][0] - m0); sv[nt][1] = exp2f(sv[nt][1] - m0);
    sv[nt][2] = exp2f(sv[nt][2] - m1); sv[nt][3] = exp2f(sv[nt][3] - m1);
    l0 += sv[nt][0] + sv[nt][1]; l1 += sv[nt][2] + sv[nt][3];
  }
  l0 += __shfl_xor_sync(~0u, l0, 1); l0 += __shfl_xor_sync(~0u, l0, 2);
  l1 += __shfl_xor_sync(~0u, l1, 1); l1 += __shfl_xor_sync(~0u, l1, 2);
  float o[D / 8][4] = {};
  for (int t = 0; t < KEYS / 16; ++t) {
    uint32_t pa[4] = { pack2(sv[2 * t][0], sv[2 * t][1]), pack2(sv[2 * t][2], sv[2 * t][3]),
                       pack2(sv[2 * t + 1][0], sv[2 * t + 1][1]), pack2(sv[2 * t + 1][2], sv[2 * t + 1][3]) };
    for (int et = 0; et < D / 8; et += 2) {
      uint32_t vb[4];
      ldsm4t(vb, Vs + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LD + (et + (lane >> 4)) * 8);
      mma16816(o[et], pa, vb[0], vb[1]);
      mma16816(o[et + 1], pa, vb[2], vb[3]);
    }
  }
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    float2 fa = __half22float2(ga[et]), fb = __half22float2(gb[et]);
    *reinterpret_cast<half2*>(out + i0 * Wd + h * D + e) = __floats2half2_rn(o[et][0] / l0 * sigm(fa.x), o[et][1] / l0 * sigm(fa.y));
    *reinterpret_cast<half2*>(out + i1 * Wd + h * D + e) = __floats2half2_rn(o[et][2] / l1 * sigm(fb.x), o[et][3] / l1 * sigm(fb.y));
  }
}

// out[r] = mask[g] ? in[idx[g] + k * srcRows] : 0 for r = k * count + g, rows of `chunks` 16-byte pieces
template <class T>
__global__ void gatherRowsK(const T* in, const int* idx, const float* mask, T* out, size_t rows, int chunks,
                            size_t count, size_t srcRows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * chunks) return;
  size_t r = t / chunks; int c = (int)(t % chunks);
  size_t g = r % count, k = r / count;
  uint4 v = mask[g] != 0 ? reinterpret_cast<const uint4*>(in)[((size_t)idx[g] + k * srcRows) * chunks + c]
                         : make_uint4(0, 0, 0, 0);
  reinterpret_cast<uint4*>(out)[t] = v;
}
// rf3's q/k LayerNorm in an atom block: per atom row, q (with its bias, which the attention then
// does not add again) and k each normalised over heads x dimension, scale and offset; a warp a row.
// Keys are normalised per atom before the gather - the same arithmetic on every real key (a padded
// slot, which the reference would normalise too, is masked out of rf3's attention).
template <class T>
__global__ void atomKqNormK(T* qg, T* kv, const float* qBias, const float* qs, const float* qo, const float* ks,
                            const float* ko, size_t rows, int Wd) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  for (int side = 0; side < 2; ++side) {
    T* x = (side ? kv : qg) + row * 2 * Wd;
    const float* sc = side ? ks : qs; const float* of = side ? ko : qo;
    float s = 0;
    for (int c = lane; c < Wd; c += 32) s += toF(x[c]) + (side ? 0.f : qBias[c]);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / Wd, q = 0;
    for (int c = lane; c < Wd; c += 32) { float d = toF(x[c]) + (side ? 0.f : qBias[c]) - mean; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
    float inv = rsqrtf(q / Wd + 1e-5f);
    for (int c = lane; c < Wd; c += 32)
      x[c] = fromF<T>((toF(x[c]) + (side ? 0.f : qBias[c]) - mean) * inv * sc[c] + of[c]);
  }
}
inline const float* zeros(size_t n) {          // a device vector of zeros, at least n long
  static float* z = nullptr; static size_t have = 0;
  if (n > have) { if (z) CK(cudaFree(z)); z = dalloc(n); CK(cudaMemset(z, 0, n * 4)); have = n; }
  return z;
}
// One block of the cross-attention transformer, in place on act [queryRows][C], from the block's cache.
// T is the GEMM inputs' type (f16 on the fast path); the residual stream stays f32.
template <class T>
void crossAttentionBlockT(float* act, const AtomStep& st, const AtomBlockCache& bc, const AtomShape& sh,
                          int C, int heads, int D, const std::string& B) {
  size_t q1 = (size_t)sh.subsets * sh.queries;                  // one sample's query rows
  size_t qRows = q1 * NS, kRows = (size_t)sh.subsets * sh.keys * NS;
  int Wd = heads * D;
  // rf3's wiring: the transition reads the block's INPUT, both terms added to act
  float* pre = nullptr;
  if (st.noResidual) {
    pre = scratch<float>("ab.pre", qRows * C);
    CK(cudaMemcpyAsync(pre, act, qRows * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  T* xq = scratch<T>("ab.xqk", 2 * qRows * C);
  T* xk = xq + qRows * C;
  if (bc.chained) {         // xk = adaLN_k(adaLN_q(x)): the queries normalised in f32 first
    float* xqF = scratch<float>("ab.xqF", qRows * C);
    adaLn<float>(act, bc.qScale, bc.qShift, xqF, qRows, C, q1);
    if constexpr (std::is_same_v<T, half>) toHalfK<<<blocks(qRows * C), 256, 0, STREAM>>>(xqF, xq, qRows * C);
    else CK(cudaMemcpyAsync(xq, xqF, qRows * C * 4, cudaMemcpyDeviceToDevice, STREAM));
    adaLn<T>(xqF, bc.kScale, bc.kShift, xk, qRows, C, q1);
  } else {
    adaLn2K<T><<<(unsigned)((qRows + 7) / 8), 256, 0, STREAM>>>(act, bc.qScale, bc.qShift, bc.kScale, bc.kShift, xq, xk,
                                                               qRows, C, q1);
  }
  T* qg = scratch<T>("ab.qgkv", 2 * qRows * 2 * Wd); T* kvAtom = qg + qRows * 2 * Wd;
  T* kv = scratch<T>("ab.kv", kRows * 2 * Wd);
  {
    std::string w = concatColumns(B + ".qgkv~stack", 1, {{pairedWeight(B + ".qProjection", B + ".gatingQuery", C, Wd), C * 2 * Wd, false},
                                                         {pairedWeight(B + ".kProjection", B + ".vProjection", C, Wd), C * 2 * Wd, false}});
    const void* Wp; if constexpr (std::is_same_v<T, float>) Wp = W(w); else Wp = Wh(w);
    const float one = 1.f, zero = 0.f;
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, 2 * Wd, (int)qRows, C, &one, Wp, cudaType<T>(), 2 * Wd,
       (long long)C * 2 * Wd, xq, cudaType<T>(), C, (long long)qRows * C, &zero, qg, cudaType<T>(), 2 * Wd,
       (long long)qRows * 2 * Wd, 2, CUBLAS_COMPUTE_32F,
       std::is_same_v<T, float> && !F32_TF32 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  const float* qBias = W(B + ".qBias");
  if (hasW(B + ".queryLayerNormScale")) {       // rf3: normalised per atom row, the q bias inside
    atomKqNormK<T><<<(unsigned)((qRows + 7) / 8), 256, 0, STREAM>>>(qg, kvAtom, qBias, W(B + ".queryLayerNormScale"),
      W(B + ".queryLayerNormOffset"), W(B + ".keyLayerNormScale"), W(B + ".keyLayerNormOffset"), qRows, Wd);
    qBias = zeros(Wd);
  }
  int chunks = (int)(2 * Wd * sizeof(T) / 16);
  gatherRowsK<T><<<blocks(kRows * chunks), 256, 0, STREAM>>>(kvAtom, st.queriesToKeys.idx, st.queriesToKeys.mask, kv,
    kRows, chunks, (size_t)st.queriesToKeys.count, q1);
  T* gathered = scratch<T>("ab.gathered", qRows * Wd);
  int warps = 8;
  size_t smem = ((size_t)sh.keys * (D + 1) * 2 + (size_t)warps * sh.keys + (size_t)warps * D) * 4;
  dim3 grid((unsigned)(sh.subsets * NS), heads);
  if constexpr (std::is_same_v<T, half>) {
    if (D == 32 && sh.keys == 128 && sh.queries == 32)
      atomAttentionMMA<32, 128><<<grid, 64, 0, STREAM>>>(
        qg, qBias, kv, st.qMask, st.kMask, bc.pairLogits, gathered, sh.queries, heads, st.keyMasked,
        1.f / sqrtf((float)D), sh.subsets);
    else
      atomAttentionFusedK<T><<<grid, warps * 32, smem, STREAM>>>(
        qg, qBias, kv, st.qMask, st.kMask, bc.pairLogits, gathered, sh.queries, sh.keys, heads, D,
        st.keyMasked, sh.subsets);
  } else {
    atomAttentionFusedK<T><<<grid, warps * 32, smem, STREAM>>>(
      qg, qBias, kv, st.qMask, st.kMask, bc.pairLogits, gathered, sh.queries, sh.keys, heads, D,
      st.keyMasked, sh.subsets);
  }
  float* attention = scratch<float>("ab.attention", qRows * C);
  linear<T, float>(gathered, attention, qRows, Wd, C, B + ".Transition2");
  T* tn = scratch<T>("ab.tn", qRows * C);
  if (st.noResidual) {
    addSigmoidGatedK<<<blocks(qRows * C), 256, 0, STREAM>>>(act, attention, bc.zg, qRows * C, q1 * C);
    adaLn<T>(pre, bc.ffwScale, bc.ffwShift, tn, qRows, C, q1);
  } else {
    gatedAddAdaLnRowsK<T><<<(unsigned)((qRows + 7) / 8), 256, 0, STREAM>>>(act, attention, bc.zg, bc.ffwScale, bc.ffwShift,
                                                                          tn, qRows, C, q1);
  }
  int I = C * 2;
  T* wide = scratch<T>("ab.wide", qRows * 3 * I);
  T* gated = scratch<T>("ab.gated", qRows * I);
  bool up; std::string w1 = upGatedTransition1(B, C, I, up);
  linear<T, T>(tn, wide, qRows, C, up ? 3 * I : 2 * I, w1);
  swiglu<T>(wide, gated, qRows, I, up);
  float* projected = scratch<float>("ab.projected", qRows * C);
  linear<T, float>(gated, projected, qRows, I, C, B + ".ffwTransition2");
  addSigmoidGatedK<<<blocks(qRows * C), 256, 0, STREAM>>>(act, projected, bc.tg, qRows * C, q1 * C);
}
inline bool ATOM_HALF = false;        // the atom blocks' GEMMs in f16 (set by --fast)
inline void crossAttentionBlock(float* act, const AtomStep& st, const AtomBlockCache& bc, const AtomShape& sh,
                                int C, int heads, int D, const std::string& B) {
  if (ATOM_HALF) crossAttentionBlockT<half>(act, st, bc, sh, C, heads, D, B);
  else crossAttentionBlockT<float>(act, st, bc, sh, C, heads, D, B);
}

// pair[q][k] = row[q] + col[k] + valid * (offsets W + dist / (1 + |d|^2) + Wvalid)
//              (+ the trunk pair, gathered per token pair)
__global__ void atomPairK(const float* row, const float* col, const float* qPos, const float* kPos,
                          const float* qUid, const float* kUid, const float* kMask, const float* Woff,
                          const float* Wdist, const float* Wvalid, const float* trunkPair,
                          const int* tqIdx, const float* tqMask, const int* tkIdx, const float* tkMask,
                          float* pair, int subsets, int queries, int keys, int Cp, int tokens, bool maskPadded) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)subsets * queries * keys * Cp;
  if (t >= total) return;
  int c = (int)(t % Cp); size_t rest = t / Cp; int key = (int)(rest % keys); size_t qi = rest / keys;
  int s = (int)(qi / queries);
  size_t ki = (size_t)s * keys + key;
  float v = row[qi * Cp + c] + col[ki * Cp + c];
  bool valid = qUid[qi] == kUid[ki] && (!maskPadded || kMask[ki] != 0);
  float d0 = qPos[qi * 3] - kPos[ki * 3], d1 = qPos[qi * 3 + 1] - kPos[ki * 3 + 1], d2 = qPos[qi * 3 + 2] - kPos[ki * 3 + 2];
  float sq = d0 * d0 + d1 * d1 + d2 * d2;
  float off = d0 * Woff[c] + d1 * Woff[Cp + c] + d2 * Woff[2 * Cp + c];
  if (valid) v += (off + Wdist[c] / (1.f + sq)) + Wvalid[c];
  if (trunkPair && tqMask[qi] != 0 && tkMask[ki] != 0)
    v += trunkPair[((size_t)tqIdx[qi] * tokens + tkIdx[ki]) * Cp + c];
  pair[t] = v;
}
// flat [(s,q,k)][blocks*heads] -> per block [s][h][q][k]
__global__ void atomLogitsLayoutK(const float* flat, float* out, int block, int nblocks, int subsets,
                                  int heads, int queries, int keys) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)subsets * heads * queries * keys;
  if (t >= total) return;
  int key = (int)(t % keys); size_t rest = t / keys; int q = (int)(rest % queries); rest /= queries;
  int h = (int)(rest % heads); int s = (int)(rest / heads);
  out[t] = flat[(((size_t)s * queries + q) * keys + key) * nblocks * heads + block * heads + h];
}
// per token: mean over its real atoms of relu(projected)
// (over `rows` token rows of every sample, the mask read per sample's token: row % tokens)
// aggregateK over convert(q2t, ...)'s output, without materialising it: the same sum in the same
// order (a query the gather masks reads as 0, which the ReLU keeps 0)
// act[q] = qStart[q] + qMask[q] * (positions gathered to q) W, W (3, C)
__global__ void encoderStartK(const float* qStart, const float* pos, const int* idx, const float* gmask, const float* Wp,
                              const float* qMask, float* act, size_t rows, int C, size_t q1, size_t atoms) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t q = t / C; int c = (int)(t % C);
  size_t gq = q % q1, k = q / q1;
  float v = 0.f;
  if (gmask[gq] != 0) {
    const float* p = pos + ((size_t)idx[gq] + k * atoms) * 3;
    v = p[0] * Wp[c] + p[1] * Wp[C + c] + p[2] * Wp[2 * C + c];
  }
  act[t] = qStart[gq * C + c] + v * qMask[gq];
}
__global__ void aggregateGatherK(const float* projected, const int* idx, const float* gmask, const float* atomMask,
                                 float* out, size_t rows, int tokens, int dense, int Cp, size_t q1) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * Cp) return;
  size_t token = t / Cp; int c = (int)(t % Cp);
  size_t k = token / tokens; size_t g0 = (token % tokens) * dense;
  float count = 0, sum = 0;
  for (int a = 0; a < dense; ++a) {
    float m = atomMask[g0 + a];
    count += m;
    if (m != 0) {
      float v = gmask[g0 + a] != 0 ? projected[((size_t)idx[g0 + a] + k * q1) * Cp + c] : 0.f;
      sum += v > 0 ? v : 0;
    }
  }
  out[t] = count > 0 ? sum / count : 0.f;
}
__global__ void aggregateK(const float* tokenAtoms, const float* atomMask, float* out, size_t rows, int tokens,
                           int dense, int Cp) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * Cp) return;
  size_t token = t / Cp; int c = (int)(t % Cp);
  float count = 0, sum = 0;
  for (int a = 0; a < dense; ++a) {
    float m = atomMask[(token % tokens) * dense + a];
    count += m;
    if (m != 0) { float v = tokenAtoms[((size_t)token * dense + a) * Cp + c]; sum += v > 0 ? v : 0; }
  }
  out[t] = count > 0 ? sum / count : 0.f;
}

// Per-block pair logits off a pair conditioning: LN (scale only), one projection to
// blocks * heads, then laid out per block.
inline std::vector<float*> atomPairLogits(const std::string& P, const float* pair, size_t pairRows, int Cp, int nblocks,
                                          int heads, const AtomShape& sh) {
  std::vector<float*> out;
  size_t per = (size_t)sh.subsets * heads * sh.queries * sh.keys;
  float* pn = scratch<float>("apl.pn", pairRows * Cp);
  if (M.flag(P + ".pairNormPerBlock")) {
    // per block its own LayerNorm scale and its own projection to `heads` (OpenDDE, protenix2) -
    // the mean and variance are the pair's either way; the difference is which weights a block reads
    float* flat = scratch<float>("apl.flat", pairRows * heads);
    for (int b = 0; b < nblocks; ++b) {
      std::string k = std::to_string(b);
      layerNormSlow(pair, pn, pairRows, Cp, W(P + ".pairInputLayerNormScales." + k), nullptr);
      linear<float, float>(pn, flat, pairRows, Cp, heads, P + ".pairLogitsProjections." + k);
      float* pl = scratch<float>(P + ".pl" + k, per);
      atomLogitsLayoutK<<<blocks(per), 256, 0, STREAM>>>(flat, pl, 0, 1, sh.subsets, heads, sh.queries, sh.keys);
      out.push_back(pl);
    }
    return out;
  }
  layerNormSlow(pair, pn, pairRows, Cp, W(P + ".pairInputLayerNormScale"), nullptr);
  float* flat = scratch<float>("apl.flat", pairRows * nblocks * heads);
  linear<float, float>(pn, flat, pairRows, Cp, nblocks * heads, P + ".pairLogitsProjection");
  for (int b = 0; b < nblocks; ++b) {
    float* pl = scratch<float>(P + ".pl" + std::to_string(b), per);
    atomLogitsLayoutK<<<blocks(per), 256, 0, STREAM>>>(flat, pl, b, nblocks, sh.subsets, heads, sh.queries, sh.keys);
    out.push_back(pl);
  }
  return out;
}

// The encoder's per-fold part, and the decoder's: everything but the activation.
struct EncoderOut {
  float* tokenAct;        // [tokens][perToken]
  float* skip;            // [queryRows][C]
  float *qMask, *kMask, *qCond, *kCond, *pair;
  float* qStart;            // the activation's start: qCond, or (preTrunkQuery) it before the trunk term
  std::vector<AtomBlockCache> blocks;
  int C, heads, D, perToken;
  bool keyMasked, noResidual;
};

// trunkSingle [tokens][Cs] and trunkPair [tokens^2][Cz] may be null (target_feat).
inline EncoderOut prepareEncoder(const std::string& E, const std::string& refPrefix, const float* trunkSingle,
                                 const float* trunkPair) {
  AtomShape sh = atomShape();
  EncoderOut o{};
  o.C = (int)M.meta(E + ".channels"); int Cp = (int)M.meta(E + ".pairChannels");
  o.heads = (int)M.meta(E + ".heads"); o.D = (int)M.meta(E + ".dimension");
  o.perToken = (int)M.meta(E + ".perTokenChannels");
  int C = o.C;
  size_t atoms = (size_t)sh.tokens * sh.dense;
  size_t qRows = (size_t)sh.subsets * sh.queries, kRows = (size_t)sh.subsets * sh.keys;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2k = gatherOf("batch.queriesToKeys");
  if ((size_t)t2q.count != qRows || (size_t)q2k.count != kRows) {
    fprintf(stderr, "atom gathers %d/%d against %zu/%zu rows\n", t2q.count, q2k.count, qRows, kRows); exit(1);
  }
  float* cond = perAtomConditioning(refPrefix, (int)atoms);
  o.qCond = scratch<float>(E + ".qCond", qRows * C);
  convert(t2q, cond, o.qCond, C);
  o.qMask = scratch<float>(E + ".qMask", qRows);
  convert(t2q, Fdev("batch.refMask"), o.qMask, 1);
  o.qStart = o.qCond;
  if (M.flag("trunk.dialect.preTrunkQuery") && trunkSingle) {
    // the queries start from the per-atom features alone (rf3, boltz2); every adaptive LayerNorm
    // still reads the full conditioning
    o.qStart = scratch<float>(E + ".qStart", qRows * C);
    CK(cudaMemcpyAsync(o.qStart, o.qCond, qRows * C * 4, cudaMemcpyDeviceToDevice, STREAM));
    scaleByRowK<<<blocks(qRows * C), 256, 0, STREAM>>>(o.qStart, o.qMask, qRows, C, qRows);
  }
  if (trunkSingle) {
    int Cs = (int)M.meta(E + ".trunkSingleChannels");
    float* ln = scratch<float>("enc.tsln", (size_t)sh.tokens * Cs);
    layerNormSlow(trunkSingle, ln, sh.tokens, Cs, W(E + ".lnormTrunkSingleCondScale"),
                  Wopt(E + ".lnormTrunkSingleCondOffset"));
    float* proj = scratch<float>("enc.tsproj", (size_t)sh.tokens * C);
    linear<float, float>(ln, proj, sh.tokens, Cs, C, E + ".embedTrunkSingleCond");
    float* perQuery = scratch<float>("enc.perQuery", qRows * C);
    convert(gatherOf("batch.tokensToQueries"), proj, perQuery, C);
    addK<<<blocks(qRows * C), 256, 0, STREAM>>>(o.qCond, perQuery, qRows * C);
  }
  scaleByRowK<<<blocks(qRows * C), 256, 0, STREAM>>>(o.qCond, o.qMask, qRows, C, qRows);
  o.kCond = scratch<float>(E + ".kCond", kRows * C);
  convert(q2k, o.qCond, o.kCond, C);
  o.kMask = scratch<float>(E + ".kMask", kRows);
  convert(q2k, o.qMask, o.kMask, 1);
  // the pair conditioning
  float* rq = scratch<float>("enc.rq", qRows * C); float* rk = scratch<float>("enc.rk", kRows * C);
  reluK<<<blocks(qRows * C), 256, 0, STREAM>>>(o.qCond, rq, qRows * C);
  reluK<<<blocks(kRows * C), 256, 0, STREAM>>>(o.kCond, rk, kRows * C);
  float* row = scratch<float>("enc.row", qRows * Cp); float* col = scratch<float>("enc.col", kRows * Cp);
  linear<float, float>(rq, row, qRows, C, Cp, E + ".singleToPairCondRow");
  linear<float, float>(rk, col, kRows, C, Cp, E + ".singleToPairCondCol");
  float* tp = nullptr;
  if (trunkPair) {
    int Cz = (int)M.meta(E + ".trunkPairChannels");
    size_t pairs = (size_t)sh.tokens * sh.tokens;
    float* ln = scratch<float>("enc.tpln", pairs * Cz);
    layerNormSlow(trunkPair, ln, pairs, Cz, W(E + ".lnormTrunkPairCondScale"), Wopt(E + ".lnormTrunkPairCondOffset"));
    tp = scratch<float>("enc.tp", pairs * Cp);
    linear<float, float>(ln, tp, pairs, Cz, Cp, E + ".embedTrunkPairCond");
  }
  float* qPos = scratch<float>("enc.qPos", qRows * 3); float* kPos = scratch<float>("enc.kPos", kRows * 3);
  float* qUid = scratch<float>("enc.qUid", qRows); float* kUid = scratch<float>("enc.kUid", kRows);
  convert(t2q, Fdev("batch.refPos"), qPos, 3);
  convert(q2k, qPos, kPos, 3);
  // the batch's own uids, every time: a process-wide cache handed OpenDDE's structural-token
  // diffusion the RESIDUE batch's (the target_feat encoder ran first), the wrong length and order
  float* uidF = scratch<float>("enc.uid", M.len("batch.refSpaceUid"));
  {
    std::vector<float> u(M.len("batch.refSpaceUid"));
    for (size_t i = 0; i < u.size(); ++i) u[i] = (float)M.i("batch.refSpaceUid")[i];
    CK(cudaMemcpyAsync(uidF, u.data(), u.size() * 4, cudaMemcpyHostToDevice, STREAM));
    CK(cudaStreamSynchronize(STREAM));     // u is a host temporary
  }
  convert(t2q, uidF, qUid, 1);
  convert(q2k, qUid, kUid, 1);
  size_t pairRows = qRows * sh.keys;
  o.pair = scratch<float>(E + ".pair", pairRows * Cp);
  Gather tq = gatherOf("batch.tokensToQueries"), tk = gatherOf("batch.tokensToKeys");
  atomPairK<<<blocks(pairRows * Cp), 256, 0, STREAM>>>(row, col, qPos, kPos, qUid, kUid, o.kMask,
    W(E + ".embedPairOffsets"), W(E + ".embedPairDistances"), W(E + ".embedPairOffsetsValid"), tp,
    tq.idx, tq.mask, tk.idx, tk.mask, o.pair, sh.subsets, sh.queries, sh.keys, Cp, sh.tokens,
    M.flag("trunk.dialect.maskPaddedKeys"));
  float* h1 = scratch<float>("enc.h1", pairRows * Cp); float* h2 = scratch<float>("enc.h2", pairRows * Cp);
  reluK<<<blocks(pairRows * Cp), 256, 0, STREAM>>>(o.pair, h1, pairRows * Cp);
  linear<float, float>(h1, h2, pairRows, Cp, Cp, E + ".pairMlp1");
  reluK<<<blocks(pairRows * Cp), 256, 0, STREAM>>>(h2, h1, pairRows * Cp);
  linear<float, float>(h1, h2, pairRows, Cp, Cp, E + ".pairMlp2");
  reluK<<<blocks(pairRows * Cp), 256, 0, STREAM>>>(h2, h1, pairRows * Cp);
  linear<float, float>(h1, o.pair, pairRows, Cp, Cp, E + ".pairMlp3", false, 1.f);
  int nblocks = 0; while (M.has(E + ".blocks." + std::to_string(nblocks) + ".qProjection")) ++nblocks;
  std::vector<float*> logits = atomPairLogits(E, o.pair, pairRows, Cp, nblocks, o.heads, sh);
  for (int b = 0; b < nblocks; ++b)
    o.blocks.push_back(prepareAtomBlock(E + ".blocks." + std::to_string(b), o.qCond, qRows, C, logits[b]));
  o.keyMasked = M.flag(E + ".blocks.0.keyMaskedAtomAttention");
  o.noResidual = M.flag(E + ".blocks.0.diffusionNoResidual");
  return o;
}
// rf3's chirality signal: per atom, d/dx of sum over its centres of (improper dihedral - ideal)^2,
// a central difference in double over each of the atom's (centre, corner) entries - an inverted
// index built once, so each atom sums its own terms (deterministic, no atomics). Transcribed from
// src/af3/diffusion/chiral-gradient.js.
__device__ double improperDihedral(const double* a, const double* b, const double* c, const double* d) {
  const double eps = 1e-6;
  double b0[3], b1[3], b2[3];
  for (int k = 0; k < 3; ++k) { b0[k] = a[k] - b[k]; b1[k] = c[k] - b[k]; b2[k] = d[k] - c[k]; }
  double length = sqrt(b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]) + eps;
  double n[3] = {b1[0] / length, b1[1] / length, b1[2] / length};
  double pb0 = b0[0] * n[0] + b0[1] * n[1] + b0[2] * n[2], pb2 = b2[0] * n[0] + b2[1] * n[1] + b2[2] * n[2];
  double v[3], w[3];
  for (int k = 0; k < 3; ++k) { v[k] = b0[k] - pb0 * n[k]; w[k] = b2[k] - pb2 * n[k]; }
  double cr[3] = {n[1] * v[2] - n[2] * v[1], n[2] * v[0] - n[0] * v[2], n[0] * v[1] - n[1] * v[0]};
  return atan2(cr[0] * w[0] + cr[1] * w[1] + cr[2] * w[2] + eps, v[0] * w[0] + v[1] * w[1] + v[2] * w[2] + eps);
}
__global__ void chiralGradK(const float* positions, const int* centers, const float* angles, const int* offsets,
                            const int* entries, float* grads, size_t atoms, int ns) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= atoms * ns) return;
  size_t k = t / atoms, atom = t % atoms;
  const float* x = positions + k * atoms * 3;
  double g[3] = {0, 0, 0};
  const double step = 1e-4;
  for (int e = offsets[atom]; e < offsets[atom + 1]; ++e) {
    int centre = entries[e] >> 2, corner = entries[e] & 3;
    double ideal = angles[centre];
    if (ideal == 0) continue;
    double p[4][3];
    for (int q = 0; q < 4; ++q) for (int a = 0; a < 3; ++a) p[q][a] = x[(size_t)centers[centre * 4 + q] * 3 + a];
    for (int a = 0; a < 3; ++a) {
      double keep = p[corner][a];
      p[corner][a] = keep + step; double up = improperDihedral(p[0], p[1], p[2], p[3]) - ideal;
      p[corner][a] = keep - step; double down = improperDihedral(p[0], p[1], p[2], p[3]) - ideal;
      p[corner][a] = keep;
      double derivative = (up * up - down * down) / (2 * step);
      if (isfinite(derivative)) g[a] += derivative;
    }
  }
  for (int a = 0; a < 3; ++a) grads[t * 3 + a] = (float)g[a];
}
struct Chirality { int *centers, *offsets, *entries; float* angles; size_t atoms; };
inline Chirality CHIRALITY{};                               // reset with each input of a batch
inline const Chirality& chirality(size_t atoms) {           // the inverted index, once per input
  Chirality& c = CHIRALITY;
  if (!c.centers) {
    int count = (int)M.meta("chiral.count");
    const int* centers = M.i("chiral.centers");
    std::vector<int> offsets(atoms + 1, 0), entries((size_t)count * 4);
    for (int i = 0; i < count * 4; ++i) offsets[centers[i] + 1]++;
    for (size_t a = 0; a < atoms; ++a) offsets[a + 1] += offsets[a];
    std::vector<int> fill(offsets.begin(), offsets.end() - 1);
    for (int i = 0; i < count * 4; ++i) entries[fill[centers[i]]++] = i;   // (centre << 2) | corner
    c = { const_cast<int*>(Idev("chiral.centers")), upload(offsets.data(), offsets.size()), upload(entries.data(), entries.size()),
          const_cast<float*>(Fdev("chiral.angles")), atoms };
  }
  return c;
}
// The encoder's per-step part: the activation from (scaled) positions, the blocks, the
// per-token aggregation. Fills o.skip and o.tokenAct.
inline void encoderStep(const std::string& E, EncoderOut& o, const float* atomPositions) {
  AtomShape sh = atomShape();
  int C = o.C;
  size_t atoms = (size_t)sh.tokens * sh.dense, q1 = (size_t)sh.subsets * sh.queries, qRows = q1 * NS;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2t = gatherOf("batch.queriesToTokenAtoms");
  float* act = scratch<float>(E + ".act", qRows * C);
  bool maskPerBlock = M.flag(E + ".blocks.0.maskAtomActPerBlock");   // padded atom rows zeroed before every block
  bool chiral = atomPositions && hasW(E + ".atomChiralToFeatures");
  if (atomPositions && !chiral) {
    // the start, the positions gathered to queries, their three-row projection, the query mask and
    // the add, in one pass (a K = 3 GEMM wrote its whole output at a fraction of the bandwidth)
    if (lenW(E + ".atomPositionsToFeatures") != (size_t)3 * C) { fprintf(stderr, "encoder: positions projection is not 3 x C\n"); exit(1); }
    encoderStartK<<<blocks(qRows * C), 256, 0, STREAM>>>(o.qStart, atomPositions, t2q.idx, t2q.mask,
      W(E + ".atomPositionsToFeatures"), o.qMask, act, qRows, C, q1, atoms);
  } else {
    for (int k = 0; k < NS; ++k)
      CK(cudaMemcpyAsync(act + k * q1 * C, o.qStart, q1 * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  if (chiral) {
    float* gp = scratch<float>("enc.gp", qRows * 3);
    convert(t2q, atomPositions, gp, 3, atoms, NS);
    float* positional = scratch<float>("enc.positional", qRows * C);
    linear<float, float>(gp, positional, qRows, 3, C, E + ".atomPositionsToFeatures");
    {
      // rf3's chirality term: the gradient's projection added beside the positions'
      if (!M.has("chiral.centers")) { fprintf(stderr, "this bundle reads chirality centres the input lacks: export it again\n"); exit(1); }
      const Chirality& ch = chirality(atoms);
      float* grads = scratch<float>("enc.chiralGrads", atoms * NS * 3);
      chiralGradK<<<blocks(atoms * NS), 128, 0, STREAM>>>(atomPositions, ch.centers, ch.angles, ch.offsets, ch.entries,
                                                         grads, atoms, NS);
      float* gc = scratch<float>("enc.gc", qRows * 3);
      convert(t2q, grads, gc, 3, atoms, NS);
      linear<float, float>(gc, positional, qRows, 3, C, E + ".atomChiralToFeatures", false, 1.f);
    }
    scaleByRowK<<<blocks(qRows * C), 256, 0, STREAM>>>(positional, o.qMask, qRows, C, q1);
    addK<<<blocks(qRows * C), 256, 0, STREAM>>>(act, positional, qRows * C);
  }
  AtomStep st{ gatherOf("batch.queriesToKeys"), o.qMask, o.kMask, o.keyMasked, o.noResidual };
  for (size_t b = 0; b < o.blocks.size(); ++b) {
    if (maskPerBlock) scaleByRowK<<<blocks(qRows * C), 256, 0, STREAM>>>(act, o.qMask, qRows, C, q1);
    crossAttentionBlock(act, st, o.blocks[b], sh, C, o.heads, o.D, E + ".blocks." + std::to_string(b));
  }
  scaleByRowK<<<blocks(qRows * C), 256, 0, STREAM>>>(act, o.qMask, qRows, C, q1);
  o.skip = act;
  float* projected = scratch<float>("enc.aggr", qRows * o.perToken);
  linear<float, float>(act, projected, qRows, C, o.perToken, E + ".projectAtomFeaturesForAggr");
  // the gather back to token atoms fused into the mean: the gathered tensor was atoms x 768 floats
  // a sample, written and read again every step (96 MB at 261 tokens and five samples)
  o.tokenAct = scratch<float>(E + ".tokenAct", (size_t)sh.tokens * NS * o.perToken);
  aggregateGatherK<<<blocks((size_t)sh.tokens * NS * o.perToken), 256, 0, STREAM>>>(projected, q2t.idx, q2t.mask,
    Fdev("batch.refMask"), o.tokenAct, (size_t)sh.tokens * NS, sh.tokens, sh.dense, o.perToken, q1);
}
inline EncoderOut atomEncoder(const std::string& E, const std::string& refPrefix, const float* trunkSingle,
                              const float* trunkPair, const float* atomPositions) {
  EncoderOut o = prepareEncoder(E, refPrefix, trunkSingle, trunkPair);
  encoderStep(E, o, atomPositions);
  return o;
}

// target_feat: [aatype one-hot 31 | profile 31 | deletion mean | atom features 384]
__global__ void targetFeatK(const int* aatype, const float* profile, const float* delMean, const float* atom,
                            float* out, int tokens) {
  int width = 31 * 2 + 1 + 384;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)tokens * width) return;
  int token = (int)(t / width), c = (int)(t % width);
  float v;
  if (c < 31) v = aatype[token] == c ? 1.f : 0.f;
  else if (c < 62) v = profile[token * 31 + (c - 31)];
  else if (c == 62) v = delMean[token];
  else v = atom[(size_t)token * 384 + (c - 63)];
  out[t] = v;
}
// boltz2's target_feat: the atom encoder's 384 columns PLUS six bias-free projections (restype,
// [profile | deletion mean], mol type, cyclic, method = x-ray, modified), a sum rather than AF3's
// concatenation
__global__ void targetFeatSumK(float* tf, const int* aatype, const float* profile, const float* delMean,
                               const int* isDna, const int* isRna, const int* isLigand, const int* isModified,
                               const float* wRes, const float* wProf, const float* wMol, const float* wMethod,
                               const float* wMod, int tokens, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)tokens * C) return;
  int token = (int)(t / C), c = (int)(t % C);
  float v = 0;
  int a = aatype[token];
  if (a >= 0 && a < 31) v += wRes[a * C + c];
  for (int k = 0; k < 31; ++k) v += profile[token * 31 + k] * wProf[k * C + c];
  v += delMean[token] * wProf[31 * C + c];
  // (a flag the batch does not carry is false everywhere)
  int mol = isLigand && isLigand[token] ? 3 : isRna && isRna[token] ? 2 : isDna && isDna[token] ? 1 : 0;
  v += wMol[mol * C + c];
  v += wMethod[1 * C + c];                      // x-ray diffraction, the default method
  v += wMod[(isModified && isModified[token] ? 1 : 0) * C + c];
  tf[t] += v;                                   // cyclic: no cyclic chains, so its term is zero
}
inline const int* flagDev(const std::string& k) { return M.has(k) ? Idev(k) : nullptr; }
inline float* buildTargetFeat() {
  int tokens = (int)M.meta("batch.tokens");
  if (M.flag("trunk.dialect.targetFeatAtomOnly")) {
    const std::string S = "targetFeat.encoder.targetFeatSum.";
    if (!hasW(S + "resType")) { fprintf(stderr, "atom-only target_feat without its six summed terms\n"); exit(1); }
    EncoderOut e = atomEncoder("targetFeat.encoder", "targetFeat.reference", nullptr, nullptr, nullptr);
    int C = 384;
    float* tf = scratch<float>("targetFeat", (size_t)tokens * C);
    CK(cudaMemcpyAsync(tf, e.tokenAct, (size_t)tokens * C * 4, cudaMemcpyDeviceToDevice, STREAM));
    targetFeatSumK<<<blocks((size_t)tokens * C), 256, 0, STREAM>>>(tf, Idev("batch.aatype"), Fdev("batch.profile"),
      Fdev("batch.deletionMean"), flagDev("batch.isDna"), flagDev("batch.isRna"), flagDev("batch.isLigand"), flagDev("batch.isModified"),
      W(S + "resType"), W(S + "msaProfile"), W(S + "molType"), W(S + "method"), W(S + "modified"), tokens, C);
    return tf;
  }
  EncoderOut e = atomEncoder("targetFeat.encoder", "targetFeat.reference", nullptr, nullptr, nullptr);
  float* tf = scratch<float>("targetFeat", (size_t)tokens * 447);
  targetFeatK<<<blocks((size_t)tokens * 447), 256, 0, STREAM>>>(Idev("batch.aatype"), Fdev("batch.profile"),
    Fdev("batch.deletionMean"), e.tokenAct, tf, tokens);
  return tf;
}
