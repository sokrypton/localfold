// The confidence head: pLDDT, PAE, PDE. Transcribed from cpu/af3/confidence/confidence.js.
// f32 throughout - the WebGPU head pins f32 too, for accuracy, and it is four blocks.
#pragma once
#include "trunk.cuh"
#include "atom.cuh"

// pair[i][j] += left[j] + right[i] + W_dgram[bin(|b_i - b_j|^2)] * mask
// (Wdist: protenix2's second, unbinned distance term - a bias-free projection of the raw distance)
// the distogram bin of each pair (-1: none) and its squared distance, once a pair rather than once a
// channel (a 39-step double-precision search per element was 24 ms of a 1044-token fold)
__global__ void confidenceBinK(const float* beta, int n, int bins, float dmin, float dmax, bool caBins, int* binOut,
                               float* sqOut, bool chaiBins = false) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)n * n) return;
  int i = (int)(ij / n), j = (int)(ij % n);
  float sq = 0;
  for (int k = 0; k < 3; ++k) { float d = beta[i * 3 + k] - beta[j * 3 + k]; sq += d * d; }
  int bin = -1;
  if (chaiBins) {
    // chai-1's: 16 bins, how many of 15 evenly spaced bounds from 3.375 to 21.375 the distance (+1e-10) is past
    float distance = sqrtf(sq + 1e-10f);
    bin = 0;
    for (int at = 0; at < bins - 1; ++at) bin += distance > 3.375f + at * (18.f / (bins - 2));
  } else if (caBins) {
    // rf3's: the bin is how many of `bins - 1` evenly spaced bounds the (real, +1e-10) distance is past
    double distance = sqrt((double)sq + 1e-10);
    bin = (int)((distance - dmin) * ((bins - 1) / (double)(dmax - dmin)));
    if (bin < 0) bin = 0;
    else if (bin > bins - 1) bin = bins - 1;
    while (bin > 0 && !(distance > dmin + (bin - 1) * ((double)(dmax - dmin) / (bins - 1)))) --bin;
    while (bin < bins - 1 && distance > dmin + bin * ((double)(dmax - dmin) / (bins - 1))) ++bin;
  } else {
    double dist = sqrt((double)sq);
    int b0 = (int)((dist - dmin) * ((bins - 1) / (double)(dmax - dmin)));
    if (b0 < 0) b0 = 0;
    else if (b0 >= bins) b0 = bins - 1;
    for (int d = -1; d <= 1; ++d) {
      int b = b0 + d;
      if (b < 0 || b >= bins) continue;
      double lo = dmin + (double)(dmax - dmin) * b / (bins - 1), hi = dmin + (double)(dmax - dmin) * (b + 1) / (bins - 1);
      double lower = lo * lo, upper = b + 1 < bins ? hi * hi : 1e8;
      if (sq > lower && sq < upper) { bin = b; break; }
    }
  }
  binOut[ij] = bin; sqOut[ij] = sq;
}
template <class PT = float>
__global__ void confidencePairInitK(float* pair, const float* left, const float* right, const int* binOf,
                                    const float* sqOf, const float* pairMask, const float* Wd, int n, int C,
                                    const float* Wdist, bool caBins, bool unmasked = false, size_t t0 = 0, size_t tEnd = 0) {
  size_t end = tEnd ? tEnd : (size_t)n * n * C;
  if ((C & 3) == 0 && (t0 & 3) == 0 && (end & 3) == 0) {
    size_t t4 = (t0 >> 2) + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t4 >= (end >> 2)) return;
    size_t c4 = (size_t)(C >> 2), ij = t4 / c4;
    int c = (int)(t4 % c4) << 2, i = (int)(ij / n), j = (int)(ij % n);
    float4 l = *(const float4*)(left + (size_t)j * C + c);
    float4 r = *(const float4*)(right + (size_t)i * C + c);
    float4 v = make_float4(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
    int bin = binOf[ij];
    if (bin >= 0) {
      float m = unmasked ? 1.f : pairMask[ij];
      float4 wd = *(const float4*)(Wd + (size_t)bin * C + c);
      v.x += wd.x * m; v.y += wd.y * m; v.z += wd.z * m; v.w += wd.w * m;
    }
    if (!caBins && Wdist) {
      float dist = sqrtf(sqOf[ij] + 1e-10f);
      float4 wdi = *(const float4*)(Wdist + c);
      v.x += dist * wdi.x; v.y += dist * wdi.y; v.z += dist * wdi.z; v.w += dist * wdi.w;
    }
    float4 p = pairLd4<PT>(pair, t4 << 2);
    pairSt4<PT>(pair, t4 << 2, make_float4(p.x + v.x, p.y + v.y, p.z + v.z, p.w + v.w));
    return;
  }
  size_t t = t0 + (size_t)blockIdx.x * blockDim.x + threadIdx.x;     // ([t0, tEnd): a sharded pair's held rows)
  if (t >= end) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  float v = left[(size_t)j * C + c] + right[(size_t)i * C + c];
  int bin = binOf[ij];
  if (bin >= 0) v += Wd[(size_t)bin * C + c] * (unmasked ? 1.f : pairMask[ij]);
  if (!caBins && Wdist) v += sqrtf(sqOf[ij] + 1e-10f) * Wdist[c];
  pairSt<PT>(pair, t, pairLd<PT>(pair, t) + v);
}
// expectation over bins of softmax(logits) . centres, optionally symmetrised (logits[ij] + logits[ji])
__global__ void expectationK(const float* logits, float* out, const float* mask, size_t rows, int bins,
                             const float* centres, int symmetricN, float scale,
                             float* out2 = nullptr, const float* centres2 = nullptr) {
  size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  if (bins == 64) {
    const float4* l0 = reinterpret_cast<const float4*>(logits + r * 64);
    const float4* l1 = symmetricN ? reinterpret_cast<const float4*>(logits + ((r % symmetricN) * symmetricN + (r / symmetricN)) * 64) : nullptr;
    float mx = -INFINITY;
    for (int k = 0; k < 16; ++k) {
      float4 a = l0[k];
      if (l1) { float4 b = l1[k]; a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
      mx = fmaxf(mx, a.x); mx = fmaxf(mx, a.y); mx = fmaxf(mx, a.z); mx = fmaxf(mx, a.w);
    }
    float total = 0, weighted = 0, weighted2 = 0;
    const float4* c1 = reinterpret_cast<const float4*>(centres);
    const float4* c2 = reinterpret_cast<const float4*>(centres2);
    for (int k = 0; k < 16; ++k) {
      float4 a = l0[k], c = c1[k];
      if (l1) { float4 b = l1[k]; a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
      float p0 = expf(a.x - mx); total += p0; weighted += p0 * c.x;
      float p1 = expf(a.y - mx); total += p1; weighted += p1 * c.y;
      float p2 = expf(a.z - mx); total += p2; weighted += p2 * c.z;
      float p3 = expf(a.w - mx); total += p3; weighted += p3 * c.w;
      if (out2) {
        float4 d = c2[k];
        weighted2 += p0 * d.x; weighted2 += p1 * d.y; weighted2 += p2 * d.z; weighted2 += p3 * d.w;
      }
    }
    out[r] = weighted / total * scale * (mask ? mask[r] : 1.f);
    if (out2) out2[r] = weighted2 / total * scale;
    return;
  }
  if (bins == 50 && (((uintptr_t)logits | (uintptr_t)centres | (uintptr_t)centres2) & 7) == 0) {
    const float2* l0 = reinterpret_cast<const float2*>(logits + r * 50);
    const float2* l1 = symmetricN ? reinterpret_cast<const float2*>(logits + ((r % symmetricN) * symmetricN + (r / symmetricN)) * 50) : nullptr;
    float mx = -INFINITY;
    for (int k = 0; k < 25; ++k) {
      float2 a = l0[k];
      if (l1) { float2 b = l1[k]; a.x += b.x; a.y += b.y; }
      mx = fmaxf(mx, a.x); mx = fmaxf(mx, a.y);
    }
    float total = 0, weighted = 0, weighted2 = 0;
    const float2* c1 = reinterpret_cast<const float2*>(centres);
    const float2* c2 = reinterpret_cast<const float2*>(centres2);
    for (int k = 0; k < 25; ++k) {
      float2 a = l0[k], c = c1[k];
      if (l1) { float2 b = l1[k]; a.x += b.x; a.y += b.y; }
      float p0 = expf(a.x - mx); total += p0; weighted += p0 * c.x;
      float p1 = expf(a.y - mx); total += p1; weighted += p1 * c.y;
      if (out2) {
        float2 d = c2[k];
        weighted2 += p0 * d.x; weighted2 += p1 * d.y;
      }
    }
    out[r] = weighted / total * scale * (mask ? mask[r] : 1.f);
    if (out2) out2[r] = weighted2 / total * scale;
    return;
  }
  auto at = [&](int b) {
    if (symmetricN == 0) return logits[r * bins + b];
    size_t i = r / symmetricN, j = r % symmetricN;
    return logits[r * bins + b] + logits[(j * symmetricN + i) * bins + b];
  };
  float mx = -INFINITY;
  for (int b = 0; b < bins; ++b) mx = fmaxf(mx, at(b));
  float total = 0, weighted = 0, weighted2 = 0;
  for (int b = 0; b < bins; ++b) {
    float p = expf(at(b) - mx);
    total += p;
    weighted += p * centres[b];
    if (out2) weighted2 += p * centres2[b];
  }
  out[r] = weighted / total * scale * (mask ? mask[r] : 1.f);
  if (out2) out2[r] = weighted2 / total * scale;
}

inline bool CONF_HALF = false;   // the head's four pairformer blocks in f16
// rf3's parameter-free LayerNorm over a WHOLE tensor, real rows only (mask per row), the mean
// and variance over `vendorWidth` columns (wider than C: the extra ones zero, each adding mean^2).
// Deterministic: per-block partials, then one block sums them in order.
// (the pair's element type PT: a bf16 confidence pair, TRUNK_PAIR16; the live rows counted in the same pass - one
// thread counting them over every pair was a serial loop of tokens^2)
template <class PT = float>
__global__ void maskedSumK(const float* x, const float* mask, size_t rows, int C, const double* mean, double* partial,
                           double* livePartial) {
  __shared__ double red[256], liveRed[256];
  double acc = 0, live = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows * C; i += (size_t)gridDim.x * blockDim.x) {
    if (!(mask[i / C] > 0)) continue;
    if (i % C == 0) live += 1;
    double v = pairLd<PT>(x, i);
    if (mean) { v -= *mean; v *= v; }
    acc += v;
  }
  red[threadIdx.x] = acc; liveRed[threadIdx.x] = live; __syncthreads();
  for (int w = blockDim.x / 2; w; w >>= 1) {
    if (threadIdx.x < w) { red[threadIdx.x] += red[threadIdx.x + w]; liveRed[threadIdx.x] += liveRed[threadIdx.x + w]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { partial[blockIdx.x] = red[0]; livePartial[blockIdx.x] = liveRed[0]; }
}
__global__ void globalStatK(const double* partial, const double* livePartial, int parts, int C, int vendorWidth,
                            double* stat, int which) {
  if (threadIdx.x || blockIdx.x) return;
  double total = 0, live = 0; for (int k = 0; k < parts; ++k) { total += partial[k]; live += livePartial[k]; }
  double count = fmax(live * vendorWidth, 1.0);
  if (which == 0) stat[0] = total / count;                      // the mean
  else {                                                        // 1 / std
    double variance = total + (double)(vendorWidth - C) * live * stat[0] * stat[0];
    stat[1] = 1.0 / sqrt(variance / count + 1e-5);
  }
}
template <class PT = float>
__global__ void applyGlobalNormK(float* x, size_t n, const double* stat) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pairSt<PT>(x, i, (float)((pairLd<PT>(x, i) - stat[0]) * stat[1]));
}
template <class PT = float>
inline void maskedGlobalNorm(float* x, const float* rowMask, size_t rows, int C, int vendorWidth) {
  const int parts = 512;
  double* partial = scratch<double>("gn.partial", parts); double* stat = scratch<double>("gn.stat", 2);
  double* live = scratch<double>("gn.live", parts);
  maskedSumK<PT><<<parts, 256, 0, STREAM>>>(x, rowMask, rows, C, nullptr, partial, live);
  globalStatK<<<1, 1, 0, STREAM>>>(partial, live, parts, C, vendorWidth, stat, 0);
  maskedSumK<PT><<<parts, 256, 0, STREAM>>>(x, rowMask, rows, C, stat, partial, live);
  globalStatK<<<1, 1, 0, STREAM>>>(partial, live, parts, C, vendorWidth, stat, 1);
  applyGlobalNormK<PT><<<blocks(rows * C), 256, 0, STREAM>>>(x, rows * C, stat);
}
// rf3: left/right += (-mean / std) * (row 0 + row 1) of each absent projection (confidence_head.py target_feat_absent)
__global__ void absentColumnsK(float* left, float* right, const float* wl, const float* wr, const double* stat, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  const int c = (int)(t % C);
  const float v = (float)(-stat[0] * stat[1]);
  left[t] += v * wl[c] + v * wl[C + c];
  right[t] += v * wr[c] + v * wr[C + c];
}
__global__ void clampK(float* x, float limit, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = fminf(limit, fmaxf(-limit, x[i]));
}
// boltz2's re-embedded pair (after LN_z and the relative encoding): right[i] + left[j] off the
// normalised s_inputs, its own 64-bin distance embedding (bounds evenly over 2..22 A), the bond
// contact and bond-order terms and the contact conditioning's unspecified constant
// each pair's distance bin, once (in double, as it always was): taken per ELEMENT - C times a pair, a double sqrt
// and a 63-step double comparison loop each - it was 252 ms of one launch on a T4 at 510 tokens, whose FP64 runs at
// 1/32 of its f32
__global__ void reembedBinK(const float* beta, int* binOut, int n) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)n * n) return;
  int i = (int)(ij / n), j = (int)(ij % n);
  double sq = 1e-10;
  for (int k = 0; k < 3; ++k) { double d = (double)beta[i * 3 + k] - beta[j * 3 + k]; sq += d * d; }
  double distance = sqrt(sq);
  int bin = (int)((distance - 2.0) * (62.0 / 20.0));
  if (bin < 0) bin = 0;
  else if (bin > 63) bin = 63;
  while (bin > 0 && !(distance > 2.0 + 20.0 * (bin - 1) / 62)) --bin;
  while (bin < 63 && distance > 2.0 + 20.0 * bin / 62) ++bin;
  binOut[ij] = bin;
}
template <class PT = float>
__global__ void reembedPairK(float* pair, const float* left, const float* right, const int* bins, const float* pairMask,
                             const float* Wd, const float* bonds, const float* orders, const float* wBond,
                             const float* wBondType, const float* unspecified, int n, int C, size_t t0 = 0, size_t tEnd = 0) {
  // [t0, tEnd): the elements this launch covers (a sharded pair's held rows); tEnd 0, all of them
  size_t end = tEnd ? tEnd : (size_t)n * n * C;
  if ((C & 3) == 0 && (t0 & 3) == 0 && (end & 3) == 0) {
    size_t t4 = (t0 >> 2) + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t4 >= (end >> 2)) return;
    size_t c4 = (size_t)(C >> 2), ij = t4 / c4;
    int c = (int)(t4 % c4) << 2, i = (int)(ij / n), j = (int)(ij % n);
    int bin = bins[ij];
    int o = orders ? (int)orders[ij] : 0;
    if (o < 0 || o >= 7) o = 0;
    float pm = pairMask[ij], bnd = bonds ? bonds[ij] : 0.f;
    float4 r = *(const float4*)(right + (size_t)i * C + c), l = *(const float4*)(left + (size_t)j * C + c);
    float4 wd = *(const float4*)(Wd + (size_t)bin * C + c);
    float4 wbt = *(const float4*)(wBondType + (size_t)o * C + c), un = *(const float4*)(unspecified + c);
    float4 v = make_float4(r.x + l.x + wd.x * pm + wbt.x + un.x,
                           r.y + l.y + wd.y * pm + wbt.y + un.y,
                           r.z + l.z + wd.z * pm + wbt.z + un.z,
                           r.w + l.w + wd.w * pm + wbt.w + un.w);
    if (bonds) {
      float4 wb = *(const float4*)(wBond + c);
      v.x += bnd * wb.x; v.y += bnd * wb.y; v.z += bnd * wb.z; v.w += bnd * wb.w;
    }
    float4 p = pairLd4<PT>(pair, t4 << 2);
    pairSt4<PT>(pair, t4 << 2, make_float4(p.x + v.x, p.y + v.y, p.z + v.z, p.w + v.w));
    return;
  }
  size_t t = t0 + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= end) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  int bin = bins[ij];
  int o = orders ? (int)orders[ij] : 0;
  if (o < 0 || o >= 7) o = 0;
  float v = right[(size_t)i * C + c] + left[(size_t)j * C + c] + Wd[(size_t)bin * C + c] * pairMask[ij]
          + (bonds ? bonds[ij] * wBond[c] : 0.f) + wBondType[(size_t)o * C + c] + unspecified[c];
  pairSt<PT>(pair, t, pairLd<PT>(pair, t) + v);
}
// prod[i][j][e] = a[i][e] b[j][e], rows i0.. of a chunk
__global__ void outerProductRowsK(const float* a, const float* b, float* out, int i0, int rowsI, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0) {
    size_t c4 = (size_t)(C >> 2);
    if (t >= (size_t)rowsI * n * c4) return;
    int e = (int)(t % c4) << 2; size_t ij = t / c4; int i = i0 + (int)(ij / n), j = (int)(ij % n);
    float4 a4 = *(const float4*)(a + (size_t)i * C + e), b4 = *(const float4*)(b + (size_t)j * C + e);
    *(float4*)(out + (t << 2)) = make_float4(a4.x * b4.x, a4.y * b4.y, a4.z * b4.z, a4.w * b4.w);
    return;
  }
  if (t >= (size_t)rowsI * n * C) return;
  int e = (int)(t % C); size_t ij = t / C; int i = i0 + (int)(ij / n), j = (int)(ij % n);
  out[t] = a[(size_t)i * C + e] * b[(size_t)j * C + e];
}
// boltz2's split pair heads: the inter-chain logits where the two tokens' chains differ
// rows [r0, r0 + cnt) of z + z^T, row-major
template <class PT = float>
__global__ void symmetriseRowsK(const float* z, float* out, size_t r0, size_t cnt, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  int c = (int)(t % C); size_t ij = r0 + t / C; size_t i = ij / T, j = ij % T;
  out[t] = pairLd<PT>(z, ij * C + c) + pairLd<PT>(z, (j * T + i) * C + c);
}
// the transposed pairs of [r0, r0 + cnt): out[(ij - r0) C + c] = z[ji C + c]
template <class PT = float>
__global__ void transposedRowsK(const float* z, float* out, size_t r0, size_t cnt, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  int c = (int)(t % C); size_t ij = r0 + t / C; size_t i = ij / T, j = ij % T;
  out[t] = pairLd<PT>(z, (j * T + i) * C + c);
}
// a chunk of the pair's own rows [r0, r0 + cnt), widened (a bf16 pair's, TRUNK_PAIR16)
template <class PT = float>
__global__ void ownRowsK(const float* z, float* out, size_t r0, size_t cnt, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < cnt * C) out[t] = pairLd<PT>(z, r0 * C + t);
}
// a chunk's own logits from its inter-chain ones (interChainLogitsRowsK, into a chunk-sized buffer)
__global__ void interChainLogitsChunkK(float* chunk, const float* inter, const int* asym, int n, int bins, size_t r0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * bins) return;
  size_t ij = r0 + t / bins; int i = (int)(ij / n), j = (int)(ij % n);
  if (asym[i] != asym[j]) chunk[t] = inter[t];
}
// the same over pairs [r0, r0 + cnt) of logits, from a chunk of inter-chain logits
__global__ void interChainLogitsRowsK(float* logits, const float* inter, const int* asym, int n, int bins, size_t r0,
                                      size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * bins) return;
  size_t ij = r0 + t / bins; int i = (int)(ij / n), j = (int)(ij % n);
  if (asym[i] != asym[j]) logits[r0 * bins + t] = inter[t];
}
__global__ void interChainLogitsK(float* logits, const float* inter, const int* asym, int n, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * bins) return;
  size_t ij = t / bins; int i = (int)(ij / n), j = (int)(ij % n);
  if (asym[i] != asym[j]) logits[t] = inter[t];
}
// p16: the pair and the trunk's pair bf16 (TRUNK_PAIR16) - and `pair` may BE the trunk's pair: its only read is the
// row-wise LayerNorm, which a warp does row by row, reading a row before writing it
inline void boltz2Reembed(float* pair, float* single, const float* trunkPair, const float* trunkSingle,
                          const float* targetFeat, const float* pseudoBeta, const float* pairMask, int n, int C, int Cs, int F,
                          bool p16 = false) {
  const std::string R = "confidence.reembed.";
  size_t pairs = (size_t)n * n;
  float* sIn = scratch<float>("conf.sInputs", (size_t)n * F);
  layerNorm2<float, float>(targetFeat, sIn, n, F, R + "sInputsNormScale", R + "sInputsNormOffset");
  layerNorm2<float, float>(trunkSingle, single, n, Cs, R + "sNormScale", R + "sNormOffset");
  linear<float, float>(sIn, single, n, F, Cs, R + "sInputToS", false, 1.f);
  if (p16) layerNorm2<__nv_bfloat16, __nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(trunkPair),
                                                    reinterpret_cast<__nv_bfloat16*>(pair), pairs, C, R + "zNormScale", R + "zNormOffset");
  else layerNorm2<float, float>(trunkPair, pair, pairs, C, R + "zNormScale", R + "zNormOffset");
  if (lenW(R + "relPosProject") != (size_t)139 * C) { fprintf(stderr, "relPosProject is not 139 x %d\n", C); exit(1); }
  WITH_PT(p16, relativeEncodingK<PT><<<blocks((C & 3) == 0 ? (pairs * (C >> 2)) : (pairs * C)), 256, 0, STREAM>>>(
    Idev("batch.features.residueIndex"), Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"),
    Idev("batch.features.entityId"), Idev("batch.features.symId"), W(R + "relPosProject"), pair, n, C, 32, 2));
  float* lrp = scratch<float>("conf.lrp", 4 * (size_t)n * C);
  float *left = lrp, *right = lrp + (size_t)n * C, *p1 = lrp + 2 * (size_t)n * C, *p2 = lrp + 3 * (size_t)n * C;
  {
    std::string w4 = concatColumns(R + "lrp12~stack", 1, {{R + "leftTargetFeatProject", F * C, false},
                                                          {R + "rightTargetFeatProject", F * C, false},
                                                          {R + "sToZProdIn1", F * C, false},
                                                          {R + "sToZProdIn2", F * C, false}});
    const float one = 1.f, zero = 0.f;
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, C, n, F, &one, W(w4), CUDA_R_32F, C,
       (long long)F * C, sIn, CUDA_R_32F, F, 0, &zero, lrp, CUDA_R_32F, C,
       (long long)n * C, 4, F32_TF32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
       F32_TF32 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
  }
  if (lenW(R + "distogramFeatProject") != (size_t)64 * C) { fprintf(stderr, "the reembed distogram is not 64 bins\n"); exit(1); }
  int* bins = scratch<int>("conf.bin", pairs);
  reembedBinK<<<blocks(pairs), 256, 0, STREAM>>>(pseudoBeta, bins, n);
  WITH_PT(p16, reembedPairK<PT><<<blocks((C & 3) == 0 ? (pairs * (C >> 2)) : (pairs * C)), 256, 0, STREAM>>>(pair, left, right, bins, pairMask, W(R + "distogramFeatProject"),
    M.has("batch.bondMatrix") ? Fdev("batch.bondMatrix") : nullptr,
    M.has("batch.bondOrderMatrix") ? Fdev("batch.bondOrderMatrix") : nullptr,
    W(R + "tokenBondsProject"), W(R + "tokenBondsTypeEmbed"), W(R + "contactEncodingUnspecified"), n, C));
  // the outer product of the two s_inputs projections through sToZProdOut, in row chunks
  int rowsPer = std::max<int>(1, (int)(CHUNK / ((size_t)n * C)));
  float* prod = scratch<float>("conf.prod", (size_t)std::min(rowsPer, n) * n * C);
  for (int i0 = 0; i0 < n; i0 += rowsPer) {
    int r = std::min(rowsPer, n - i0); size_t elems = (size_t)r * n * C;
    outerProductRowsK<<<blocks((C & 3) == 0 ? (elems >> 2) : elems), 256, 0, STREAM>>>(p1, p2, prod, i0, r, n, C);
    if (!p16) { linear<float, float>(prod, pair + (size_t)i0 * n * C, (size_t)r * n, C, C, R + "sToZProdOut", false, 1.f); continue; }
    float* out = scratch<float>("conf.prodOut", (size_t)std::min(rowsPer, n) * n * C);   // (a bf16 pair: added through f32)
    linear<float, float>(prod, out, (size_t)r * n, C, C, R + "sToZProdOut");
    pairAddK<__nv_bfloat16, float><<<blocks((elems & 3) == 0 ? (elems >> 2) : elems), 256, 0, STREAM>>>(
      reinterpret_cast<float*>(reinterpret_cast<__nv_bfloat16*>(pair) + (size_t)i0 * n * C), out, elems);
  }
}
// ...on several GPUs: this rank's rows [lo, lo + rows) of the pair (base: the bf16 slab indexed by global pair), from
// the trunk slab's same rows; the single whole, as above
inline void boltz2ReembedRows(float* base, float* single, const void* trunkRows, const float* trunkSingle,
                              const float* targetFeat, const float* pseudoBeta, const float* pairMask, int n, int C, int Cs,
                              int F, int lo, int rows) {
  using B16 = __nv_bfloat16;
  const std::string R = "confidence.reembed.";
  const size_t hf = (size_t)lo * n, hp = (size_t)rows * n;
  float* sIn = scratch<float>("conf.sInputs", (size_t)n * F);
  layerNorm2<float, float>(targetFeat, sIn, n, F, R + "sInputsNormScale", R + "sInputsNormOffset");
  layerNorm2<float, float>(trunkSingle, single, n, Cs, R + "sNormScale", R + "sNormOffset");
  linear<float, float>(sIn, single, n, F, Cs, R + "sInputToS", false, 1.f);
  if (!hp) return;
  layerNorm2<B16, B16>(reinterpret_cast<const B16*>(trunkRows), reinterpret_cast<B16*>(base) + hf * C, hp, C, R + "zNormScale", R + "zNormOffset");
  if (lenW(R + "relPosProject") != (size_t)139 * C) { fprintf(stderr, "relPosProject is not 139 x %d\n", C); exit(1); }
  relativeEncodingK<B16><<<blocks((C & 3) == 0 ? (hp * (C >> 2)) : (hp * C)), 256, 0, STREAM>>>(
    Idev("batch.features.residueIndex"), Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"),
    Idev("batch.features.entityId"), Idev("batch.features.symId"), W(R + "relPosProject"), base, n, C, 32, 2, hf * C, (hf + hp) * C);
  float* lrp = scratch<float>("conf.lrp", 4 * (size_t)n * C);
  float *left = lrp, *right = lrp + (size_t)n * C, *p1 = lrp + 2 * (size_t)n * C, *p2 = lrp + 3 * (size_t)n * C;
  {
    std::string w4 = concatColumns(R + "lrp12~stack", 1, {{R + "leftTargetFeatProject", F * C, false},
                                                          {R + "rightTargetFeatProject", F * C, false},
                                                          {R + "sToZProdIn1", F * C, false},
                                                          {R + "sToZProdIn2", F * C, false}});
    const float one = 1.f, zero = 0.f;
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, C, n, F, &one, W(w4), CUDA_R_32F, C,
       (long long)F * C, sIn, CUDA_R_32F, F, 0, &zero, lrp, CUDA_R_32F, C,
       (long long)n * C, 4, F32_TF32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
       F32_TF32 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
  }
  if (lenW(R + "distogramFeatProject") != (size_t)64 * C) { fprintf(stderr, "the reembed distogram is not 64 bins\n"); exit(1); }
  int* bins = scratch<int>("conf.bin", (size_t)n * n);
  reembedBinK<<<blocks((size_t)n * n), 256, 0, STREAM>>>(pseudoBeta, bins, n);
  reembedPairK<B16><<<blocks((C & 3) == 0 ? (hp * (C >> 2)) : (hp * C)), 256, 0, STREAM>>>(base, left, right, bins, pairMask, W(R + "distogramFeatProject"),
    M.has("batch.bondMatrix") ? Fdev("batch.bondMatrix") : nullptr,
    M.has("batch.bondOrderMatrix") ? Fdev("batch.bondOrderMatrix") : nullptr,
    W(R + "tokenBondsProject"), W(R + "tokenBondsTypeEmbed"), W(R + "contactEncodingUnspecified"), n, C, hf * C, (hf + hp) * C);
  const int rowsPer = std::max<int>(1, std::min<int>(rows, (int)(CHUNK / ((size_t)n * C))));
  float* prod = scratch<float>("conf.prod", (size_t)rowsPer * n * C);
  float* out = scratch<float>("conf.prodOut", (size_t)rowsPer * n * C);
  for (int i0 = lo; i0 < lo + rows; i0 += rowsPer) {
    int r = std::min(rowsPer, lo + rows - i0); size_t elems = (size_t)r * n * C;
    outerProductRowsK<<<blocks((C & 3) == 0 ? (elems >> 2) : elems), 256, 0, STREAM>>>(p1, p2, prod, i0, r, n, C);
    linear<float, float>(prod, out, (size_t)r * n, C, C, R + "sToZProdOut");
    pairAddK<B16, float><<<blocks((elems & 3) == 0 ? (elems >> 2) : elems), 256, 0, STREAM>>>(
      reinterpret_cast<float*>(reinterpret_cast<B16*>(base) + (size_t)i0 * n * C), out, elems);
  }
}
struct ConfidenceOut { std::vector<float> plddt, pae, pde, tmTerm; double meanPlddt, ptm, iptm; };

// consumeTrunkPair: the trunk's pair is read by nothing after this call, so the head works in it rather
// than in a copy (the caller says so for a card short of room, on its last confidence call)
// chai-1's pLDDT: each (token, dense slot)'s ATOM37 index, by its name (AF3's 4 characters, ASCII - 32)
inline const int* atom37Index(int n, int dense) {
  static const char* ATOM37[37] = { "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2",
    "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH",
    "CZ", "CZ2", "CZ3", "NZ", "OXT" };
  const int* chars = M.i("batch.refAtomNameChars");
  std::vector<int> idx((size_t)n * dense, 0);
  for (size_t a = 0; a < idx.size(); ++a)
    for (int k = 0; k < 37; ++k) {
      bool same = true;
      for (int c = 0; c < 4; ++c) {
        int want = c < (int)strlen(ATOM37[k]) ? ATOM37[k][c] - 32 : 0;
        if (chars[a * 4 + c] != want) { same = false; break; }
      }
      if (same) { idx[a] = k; break; }
    }
  int* d = scratch<int>("conf.atom37", idx.size());
  CK(cudaMemcpy(d, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
  return d;
}
__global__ void gatherPlddt37K(const float* p37, const int* idx, float* out, int n, int dense, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * dense * bins) return;
  int b = (int)(t % bins); size_t slot = t / bins; size_t token = slot / dense;
  out[t] = p37[(token * 37 + idx[slot]) * bins + b];
}
// chai-1's confidence triangle attention: each direction's output projection plus its transposed twin, summed in
// place once (their two applications cancel to one at inference - af3-any-model modules.py, dual_output)
__global__ void addInPlaceK(float* a, const float* b, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] += b[i];
}
inline void foldDualOutputs() {
  static bool done = false;
  if (done) return;
  done = true;
  bool any = false;
  for (int k = 0; M.has("confidence.blocks." + std::to_string(k) + ".singleChannels"); ++k)
    for (int a = 1; a <= 2; ++a) {
      std::string G = "confidence.blocks." + std::to_string(k) + ".pairAttention" + std::to_string(a);
      if (!hasW(G + ".outputProjectionTransposed")) continue;
      addInPlaceK<<<blocks(lenW(G + ".outputProjection")), 256, 0, STREAM>>>(const_cast<float*>(W(G + ".outputProjection")),
        W(G + ".outputProjectionTransposed"), lenW(G + ".outputProjection"));
      any = true;
    }
  if (any) CK(cudaStreamSynchronize(STREAM));
}
inline const float* hostSeqMask(const float* seqMask, int n, std::vector<float>& buf) {
  if (M.has("batch.seqMask") && M.len("batch.seqMask") == (size_t)n && seqMask == Fdev("batch.seqMask"))
    return M.f("batch.seqMask");
  buf = download(seqMask, n);
  return buf.data();
}
// pTM and ipTM off the PAE logits' per-pair expected TM terms: the best anchor's mean over the pairs it selects
// (ipTM: other chains only). shared/heads/tm-score.js. out.tmTerm takes the terms
inline void confidenceTm(ConfidenceOut& out, std::vector<float> term, const float* seqMask, int n) {
  std::vector<float> seqBuf;
  const float* seq = hostSeqMask(seqMask, n, seqBuf);
  const int* asym = M.i("batch.asymId");
  auto reduce = [&](bool interOnly) {
    double best = -1e30; bool any = false;
    for (int i = 0; i < n; ++i) {
      double tot = 0; int cnt = 0;
      for (int j = 0; j < n; ++j) {
        if (!(seq[i] > 0 && seq[j] > 0) || (interOnly && asym[i] == asym[j])) continue;
        tot += term[(size_t)i * n + j]; ++cnt;
      }
      if (cnt && std::isnan(tot / cnt)) return (double)NAN;     // (std::max passes a NaN by: the score must not)
      if (cnt) { any = true; best = std::max(best, tot / cnt); }
    }
    return any ? best : NAN;
  };
  out.ptm = reduce(false); out.iptm = reduce(true);
  out.tmTerm = std::move(term);
}
// pLDDT off the head's single (and its mean over real atoms); headNorm the head's LayerNorm-or-none
template <class HN>
inline void confidencePlddt(ConfidenceOut& out, const float* single, int n, int Cs, HN&& headNorm) {
  const std::string P = "confidence";
  const int dense = (int)M.meta("batch.dense");
  const int PB = 50;
  float pc[50];
  for (int b = 0; b < PB; ++b) pc[b] = 0.5f / PB + (float)b / PB;
  float* dpc = scratch<float>("conf.dpc", PB);
  CK(cudaMemcpyAsync(dpc, pc, PB * sizeof(float), cudaMemcpyHostToDevice, STREAM));
  float* slnBuf = scratch<float>("conf.sln", (size_t)n * Cs);
  const float* sln = headNorm(single, slnBuf, n, Cs, "plddtLn");
  float* pl = scratch<float>("conf.plddtLogits", (size_t)n * dense * PB);
  if (M.flag("trunk.dialect.chaiConfidence")) {
    // chai-1 predicts pLDDT over the 37 ATOM37 slots and gathers each dense slot's by its atom NAME (no match: slot 0)
    float* p37 = scratch<float>("conf.plddt37", (size_t)n * 37 * PB);
    linear<float, float>(sln, p37, n, Cs, 37 * PB, P + ".plddtLogits");
    gatherPlddt37K<<<blocks((size_t)n * dense * PB), 256, 0, STREAM>>>(p37, atom37Index(n, dense), pl, n, dense, PB);
  } else
    linear<float, float>(sln, pl, n, Cs, dense * PB, P + ".plddtLogits");
  float* plddt = scratch<float>("conf.plddt", (size_t)n * dense);
  expectationK<<<blocks((size_t)n * dense), 256, 0, STREAM>>>(pl, plddt, nullptr, (size_t)n * dense, PB, dpc, 0, 100.f);
  out.plddt.resize((size_t)n * dense);
  CK(cudaMemcpyAsync(out.plddt.data(), plddt, (size_t)n * dense * sizeof(float), cudaMemcpyDeviceToHost, STREAM));
}
inline void finishConfidencePlddt(ConfidenceOut& out, int n, int dense) {
  // (rf3, intellifold2, openbind0: each atom's by its rank among the token's real atoms - plddtByRank in
  // cpu/af3/confidence/confidence.js; a softmax a slot, so the expectation gathers as the logits would)
  if (M.flag("trunk.dialect.perAtomHeadsByRank")) {
    const int* chars = M.i("batch.refAtomNameChars");
    std::vector<float> ranked(out.plddt.size());
    for (int t = 0; t < n; ++t) {
      int rank = -1;
      for (int s = 0; s < dense; ++s) {
        size_t flat = (size_t)t * dense + s;
        bool real = chars[flat * 4] || chars[flat * 4 + 1] || chars[flat * 4 + 2] || chars[flat * 4 + 3];
        if (real) ++rank;
        ranked[flat] = out.plddt[(size_t)t * dense + std::min(std::max(rank, 0), dense - 1)];
      }
    }
    out.plddt.swap(ranked);
  }
  const float* mask = M.f("batch.predDenseAtomMask");
  double sum = 0, count = 0;
  for (size_t i = 0; i < out.plddt.size(); ++i) if (mask[i]) { sum += out.plddt[i]; count += 1; }
  out.meanPlddt = sum / std::max(count, 1.0);
}
inline ConfidenceOut confidenceHead(const float* trunkPair, const float* trunkSingle, const float* targetFeat,
                                    const float* pseudoBeta, const float* seqMask, const float* pairMask, int n,
                                    bool consumeTrunkPair = false) {
  const std::string P = "confidence";
  int C = (int)M.meta(P + ".pairChannels"), Cs = (int)M.meta(P + ".singleChannels"), F = (int)M.meta(P + ".targetFeatWidth");
  int dense = (int)M.meta("batch.dense");
  bool caDgram = M.flag("trunk.dialect.confidenceCaDgram");
  size_t pairs = (size_t)n * n;
  bool reembed = M.flag("trunk.dialect.reembedConfidencePair");
  // (boltz2 builds its pair from the trunk's: in f32 into a pair of its own; in bf16 in place when it may, its one
  // read of the trunk's pair being a row-wise LayerNorm)
  bool inPlace = consumeTrunkPair && (!reembed || TRUNK_PAIR16);
  // a bf16 trunk pair (TRUNK_PAIR16, a card short of room): the head's pair is bf16 too, its blocks run under PAIR16
  // and its heads widen a chunk of rows at a time (af3.cu keeps it only for a head that takes it: no re-embedding)
  const bool p16 = TRUNK_PAIR16;
  if (p16 && !shortPair(pairs, C)) {
    fprintf(stderr, "the confidence head was handed a bf16 pair it does not take\n"); exit(1);
  }
  const size_t pairBytes = pairs * C * (p16 ? 2 : 4);
  float* pair = inPlace ? const_cast<float*>(trunkPair) : scratch<float>("conf.pair", pairBytes / 4);
  float* single = scratch<float>("conf.single", (size_t)n * Cs);
  if (reembed) {
    boltz2Reembed(pair, single, trunkPair, trunkSingle, targetFeat, pseudoBeta, pairMask, n, C, Cs, F, p16);
  } else {
  if (!inPlace) CK(cudaMemcpyAsync(pair, trunkPair, pairBytes, cudaMemcpyDeviceToDevice, STREAM));
  CK(cudaMemcpyAsync(single, trunkSingle, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  if (M.flag("trunk.dialect.confidenceGlobalNorm")) {
    // rf3 normalises every detached trunk input over the whole tensor first (target_feat over the
    // vendor's 449 columns, two wider than ours)
    float* tf = scratch<float>("conf.targetFeat", (size_t)n * F);
    CK(cudaMemcpyAsync(tf, targetFeat, (size_t)n * F * 4, cudaMemcpyDeviceToDevice, STREAM));
    if (p16) maskedGlobalNorm<__nv_bfloat16>(pair, pairMask, pairs, C, C);
    else maskedGlobalNorm(pair, pairMask, pairs, C, C);
    maskedGlobalNorm(single, seqMask, n, Cs, Cs);
    maskedGlobalNorm(tf, seqMask, n, F, 449);
    targetFeat = tf;
  }
  float* lr = scratch<float>("conf.lr", 2 * (size_t)n * C); float *left = lr, *right = lr + (size_t)n * C;
  {
    std::string w2 = concatColumns(P + ".lrTargetFeat~stack", 1, {{P + ".leftTargetFeatProject", F * C, false},
                                                                  {P + ".rightTargetFeatProject", F * C, false}});
    const float one = 1.f, zero = 0.f;
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, C, n, F, &one, W(w2), CUDA_R_32F, C,
       (long long)F * C, targetFeat, CUDA_R_32F, F, 0, &zero, lr, CUDA_R_32F, C,
       (long long)n * C, 2, F32_TF32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
       F32_TF32 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
  }
  // (rf3: its two dropped s_inputs columns at their normalised value -mean/std, through their own rows - the same on every
  // token; gn.stat still holds target_feat's statistics)
  if (M.flag("trunk.dialect.confidenceGlobalNorm") && hasW(P + ".leftTargetFeatProjectAbsent")) {
    if (lenW(P + ".leftTargetFeatProjectAbsent") != (size_t)2 * C) { fprintf(stderr, "rf3's absent target_feat rows are not 2 x %d\n", C); exit(1); }
    absentColumnsK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(left, right, W(P + ".leftTargetFeatProjectAbsent"),
      W(P + ".rightTargetFeatProjectAbsent"), scratch<double>("gn.stat", 2), n, C);
  }
  int bins = (int)(lenW(P + ".distogramFeatProject") / C);
  int* binOf = scratch<int>("conf.bin", pairs); float* sqOf = scratch<float>("conf.sq", pairs);
  const bool chai = M.flag("trunk.dialect.chaiConfidence");     // chai-1's 16-bin distance embedding, unmasked
  confidenceBinK<<<blocks(pairs), 256, 0, STREAM>>>(pseudoBeta, n, bins, 3.25f, 50.75f, caDgram, binOf, sqOf, chai);
  size_t initGrid = (C & 3) == 0 ? (pairs * (C >> 2)) : (pairs * C);
  if (p16) confidencePairInitK<__nv_bfloat16><<<blocks(initGrid), 256, 0, STREAM>>>(pair, left, right, binOf, sqOf, pairMask,
    W(P + ".distogramFeatProject"), n, C, Wopt(P + ".distanceFeatProject"), caDgram, chai);
  else confidencePairInitK<<<blocks(initGrid), 256, 0, STREAM>>>(pair, left, right, binOf, sqOf, pairMask,
    W(P + ".distogramFeatProject"), n, C, Wopt(P + ".distanceFeatProject"), caDgram, chai);
  if (bins != (chai ? 16 : caDgram ? 40 : 39)) { fprintf(stderr, "confidence distogram has %d bins\n", bins); exit(1); }
  if (hasW(P + ".inputSingleNormScale")) {
    // the trunk single clamped to +-512 and LayerNormed before any use (protenix2)
    clampK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(single, 512.f, (size_t)n * Cs);
    float* sn = scratch<float>("conf.singleNorm", (size_t)n * Cs);
    layerNorm2<float, float>(single, sn, n, Cs, P + ".inputSingleNormScale", P + ".inputSingleNormOffset");
    CK(cudaMemcpyAsync(single, sn, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  }
  foldDualOutputs();
  memReport("confidence: pair initialised");
  int nb = 0; while (M.has(P + ".blocks." + std::to_string(nb) + ".singleChannels")) ++nb;
  bool swap = M.flag("trunk.dialect.swapTransposedBias"), divide = M.flag("trunk.dialect.triangleMulDivideByLength");
  struct Pair16Scope { bool was; explicit Pair16Scope(bool on) : was(PAIR16) { PAIR16 = on; } ~Pair16Scope() { PAIR16 = was; } };
  {
  Pair16Scope scope(p16);
  if (p16 && !CONF_HALF) { fprintf(stderr, "a bf16 confidence pair wants the f16 head (--fast)\n"); exit(1); }
  for (int k = 0; k < nb; ++k)
    if (CONF_HALF) pairformerBlockAt<half>(pair, single, pairMask, seqMask, n, C, Cs, P + ".blocks." + std::to_string(k), swap, divide);
    else pairformerBlockAt<float>(pair, single, pairMask, seqMask, n, C, Cs, P + ".blocks." + std::to_string(k), swap, divide);
  }
  // on a card short of room the stack's scratch goes before the heads allocate theirs, so the two peak
  // apart rather than together
  if (shortPair(pairs, C)) releaseScratch({ "tri.", "trib.", "grid.", "tr.", "st." });
  memReport("confidence: blocks done");
  // the error bins: 64 of them up to 31 A, the last one step past the second-to-last
  const int NB = 64; double step = 31.0 / (NB - 2);
  float dCentresBuf[128];
  for (int b = 0; b < NB - 1; ++b) dCentresBuf[b] = (float)(b * step + step / 2);
  dCentresBuf[NB - 1] = (float)(dCentresBuf[NB - 2] + step);
  double d0 = 0;
  {
    std::vector<float> seqBuf;
    const float* seq = hostSeqMask(seqMask, n, seqBuf);
    int real = 0; for (int i = 0; i < n; ++i) real += seq[i] > 0;
    d0 = 1.24 * std::cbrt(std::max(real, 19) - 15.0) - 1.8;
    for (int b = 0; b < NB; ++b) dCentresBuf[NB + b] = (float)(1 / (1 + (double)dCentresBuf[b] * dCentresBuf[b] / (d0 * d0)));
  }
  float* dTables = scratch<float>("conf.centres", 2 * NB);
  CK(cudaMemcpyAsync(dTables, dCentresBuf, 2 * NB * sizeof(float), cudaMemcpyHostToDevice, STREAM));
  const float* dCentres = dTables;
  const float* dPerBin = dTables + NB;
  // on a card short of room the heads' LayerNorm (and the symmetrised pair it reads) run in row chunks
  bool chunked = shortPair(pairs, C);
  size_t rowsPer = chunked ? std::max<size_t>(1, std::min(pairs, CHUNK / C)) : pairs;
  float* ln = scratch<float>("conf.ln", rowsPer * C);
  // chunked, each chunk's logits are taken down to their expectations at once, so they are a chunk's and not the
  // whole matrix's - [pairs, 64] f32 is TWICE a 128-channel f32 pair (25.6 GB at 10,000 tokens)
  float* logits = scratch<float>("conf.logits", (chunked ? rowsPer : pairs) * NB);
  ConfidenceOut out;
  float* pde = scratch<float>("conf.pde", pairs); float* pae = scratch<float>("conf.pae", pairs);
  bool preSym = M.flag("trunk.dialect.preSymmetrisedPde");
  // a head LayerNorm the bundle does not carry is no LayerNorm (boltz2 reads z and s directly)
  auto headNorm = [&](const float* x, float* buf, size_t rows, int Cx, const std::string& name) -> const float* {
    if (!hasW(P + "." + name + "Scale")) return x;
    layerNorm2<float, float>(x, buf, rows, Cx, P + "." + name + "Scale", P + "." + name + "Offset");
    return buf;
  };
  // boltz2 splits each pair head into an intra-chain and an inter-chain projection
  const int* asymDev = Idev("batch.asymId");
  float* interLogits = hasW(P + ".interHalfDistanceLogits") || hasW(P + ".paeInterLogits")
    ? scratch<float>("conf.interLogits", (chunked ? rowsPer : pairs) * NB) : nullptr;
  auto project = [&](const float* x, const std::string& w, const std::string& inter) {
    linear<float, float>(x, logits, pairs, C, NB, P + "." + w);
    if (!hasW(P + "." + inter)) return;
    linear<float, float>(x, interLogits, pairs, C, NB, P + "." + inter);
    interChainLogitsK<<<blocks(pairs * NB), 256, 0, STREAM>>>(logits, interLogits, asymDev, n, NB);
  };
  if (hasW(P + ".interHalfDistanceLogits") && !preSym) { fprintf(stderr, "split PDE heads need the pre-symmetrised PDE\n"); exit(1); }
  // a head's logits, LN(x) W (with boltz2's inter-chain half), a chunk of rows at a time when chunked
  float* sym = preSym && chunked ? scratch<float>("conf.sym", rowsPer * C) : nullptr;
  // chunked: a chunk's logits (with boltz2's inter-chain half) and their expectation, pair by pair
  auto chunkLogits = [&](const float* xn, size_t r0, size_t cnt, const std::string& w, const std::string& inter) {
    linear<float, float>(xn, logits, cnt, C, NB, P + "." + w);
    if (!hasW(P + "." + inter)) return;
    linear<float, float>(xn, interLogits, cnt, C, NB, P + "." + inter);
    interChainLogitsChunkK<<<blocks(cnt * NB), 256, 0, STREAM>>>(logits, interLogits, asymDev, n, NB, r0, cnt);
  };
  float* tmTerm = scratch<float>("conf.tmTerm", pairs);
  if (chunked) {
    // the PDE: AF3 symmetrises AFTER the projection (expectation of l_ij + l_ji), protenix2 and boltz2 before it
    // (LN(z + z^T) W). Chunked, AF3's form is taken through linearity, W LN(z_ij) + W LN(z_ji) = W (LN(z_ij) +
    // LN(z_ji)): one projection of the summed norms, its own chunk - the same logits up to rounding
    float* sym2 = !preSym ? scratch<float>("conf.symT", rowsPer * C) : nullptr;
    float* ln2 = !preSym ? scratch<float>("conf.ln2", rowsPer * C) : nullptr;
    float* own = p16 ? scratch<float>("conf.own", rowsPer * C) : nullptr;
    auto ownRows = [&](size_t r0, size_t cnt) -> const float* {     // a chunk of the pair's rows, as f32
      if (!p16) return pair + r0 * C;
      ownRowsK<__nv_bfloat16><<<blocks(cnt * C), 256, 0, STREAM>>>(pair, own, r0, cnt, C);
      return own;
    };
    for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
      size_t cnt = std::min(rowsPer, pairs - r0);
      const float* x = preSym ? nullptr : ownRows(r0, cnt);
      if (preSym) {
        if (p16) symmetriseRowsK<__nv_bfloat16><<<blocks(cnt * C), 256, 0, STREAM>>>(pair, sym, r0, cnt, n, C);
        else symmetriseRowsK<<<blocks(cnt * C), 256, 0, STREAM>>>(pair, sym, r0, cnt, n, C);
        x = sym;
      }
      const float* xn = headNorm(x, ln, cnt, C, "logitsLn");
      if (!preSym) {
        // (no head norm, the rows are the pair's own: copied before the transposed rows are added to them)
        if (xn != ln) { CK(cudaMemcpyAsync(ln, xn, cnt * C * 4, cudaMemcpyDeviceToDevice, STREAM)); xn = ln; }
        if (p16) transposedRowsK<__nv_bfloat16><<<blocks(cnt * C), 256, 0, STREAM>>>(pair, sym2, r0, cnt, n, C);
        else transposedRowsK<<<blocks(cnt * C), 256, 0, STREAM>>>(pair, sym2, r0, cnt, n, C);
        addK<<<blocks(cnt * C), 256, 0, STREAM>>>(ln, headNorm(sym2, ln2, cnt, C, "logitsLn"), cnt * C);
      }
      chunkLogits(xn, r0, cnt, "leftHalfDistanceLogits", "interHalfDistanceLogits");
      expectationK<<<blocks(cnt), 256, 0, STREAM>>>(logits, pde + r0, pairMask + r0, cnt, NB, dCentres, 0, 1.f);
    }
    for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
      size_t cnt = std::min(rowsPer, pairs - r0);
      chunkLogits(headNorm(ownRows(r0, cnt), ln, cnt, C, "paeLogitsLn"), r0, cnt, "paeLogits", "paeInterLogits");
      expectationK<<<blocks(cnt), 256, 0, STREAM>>>(logits, pae + r0, pairMask + r0, cnt, NB, dCentres, 0, 1.f, tmTerm + r0, dPerBin);
    }
  } else {
    const float* src = pair;
    if (preSym) {
      // symmetrised BEFORE the projection: LN(z + z^T) W (protenix2, boltz2); AF3 adds the transpose after
      float* whole = scratch<float>("conf.sym", pairs * C);
      symmetriseK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, whole, n, C);
      src = whole;
    }
    project(headNorm(src, ln, pairs, C, "logitsLn"), "leftHalfDistanceLogits", "interHalfDistanceLogits");
    expectationK<<<blocks(pairs), 256, 0, STREAM>>>(logits, pde, pairMask, pairs, NB, dCentres, preSym ? 0 : n, 1.f);
    project(headNorm(pair, ln, pairs, C, "paeLogitsLn"), "paeLogits", "paeInterLogits");
    expectationK<<<blocks(pairs), 256, 0, STREAM>>>(logits, pae, pairMask, pairs, NB, dCentres, 0, 1.f, tmTerm, dPerBin);
  }
  // pTM and ipTM off the PAE logits: per pair the expected TM term, then the best anchor's
  // mean over the pairs it selects (ipTM: other chains only). shared/heads/tm-score.js.
  confidencePlddt(out, single, n, Cs, headNorm);
  std::vector<float> tmHost(pairs);
  out.pae.resize(pairs);
  out.pde.resize(pairs);
  CK(cudaMemcpyAsync(tmHost.data(), tmTerm, pairs * sizeof(float), cudaMemcpyDeviceToHost, STREAM));
  CK(cudaMemcpyAsync(out.pae.data(), pae, pairs * sizeof(float), cudaMemcpyDeviceToHost, STREAM));
  CK(cudaMemcpyAsync(out.pde.data(), pde, pairs * sizeof(float), cudaMemcpyDeviceToHost, STREAM));
  CK(cudaStreamSynchronize(STREAM));
  finishConfidencePlddt(out, n, dense);
  confidenceTm(out, std::move(tmHost), seqMask, n);
  return out;
}

// Several GPUs on a sharded pair (sharded.cuh): the head on this rank's rows - its pair a copy of the trunk's slab, its
// blocks sh::pairUpdates and the sharded single track, its PAE and PDE heads over held rows (the PDE's transposed
// rows the same rows of z^T, one all-to-all transpose), every rank's rows of PAE, PDE and the TM terms read onto rank 0
// over the peers' buffers. Collective; the result is rank 0's (the others' carry nothing). The AF3 lineage's head
// without re-embedding or a global norm (not boltz2 or rf3 yet), in f16 (--fast)
// whether this model's confidence head has a sharded form: the AF3 lineage's plain head in f16 (af3.cu gathers the pair
// onto rank 0 for the others' - boltz2's re-embedding, rf3's, chai-1's - and runs their one-GPU head there)
inline bool confidenceShardable() {
  const std::string P = "confidence";
  return !(M.flag("trunk.dialect.confidenceGlobalNorm") || M.flag("trunk.dialect.chaiConfidence") || !CONF_HALF ||
           (hasW(P + ".interHalfDistanceLogits") && !M.flag("trunk.dialect.preSymmetrisedPde")));
}
inline ConfidenceOut confidenceSharded(const mg::Shared& trunkZ, int lo, int rowsHere, const float* trunkSingle,
                                       const float* targetFeat, const float* pseudoBeta, const float* seqMask,
                                       const float* pairMask, int n, bool consume = false) {
  // consume: nothing reads the trunk's slab after this call - the head works in it rather than in a copy
  const std::string P = "confidence";
  int C = (int)M.meta(P + ".pairChannels"), Cs = (int)M.meta(P + ".singleChannels"), F = (int)M.meta(P + ".targetFeatWidth");
  const bool caDgram = M.flag("trunk.dialect.confidenceCaDgram"), chai = M.flag("trunk.dialect.chaiConfidence");
  if (!confidenceShardable()) {
    fprintf(stderr, "a sharded confidence head: the AF3 lineage's head and boltz2's in f16 only (not rf3 or chai-1 yet)\n"); exit(1);
  }
  const bool reembed = M.flag("trunk.dialect.reembedConfidencePair");
  const size_t pairs = (size_t)n * n, hp = (size_t)rowsHere * n, hf = (size_t)lo * n;
  const size_t slab = (size_t)sh::maxStored(n) * n * C * 2;
  mg::Shared& cz = consume ? const_cast<mg::Shared&>(trunkZ) : mg::shared("sh.cz", slab);
  mg::Shared* czTp = nullptr;                // (made where a form wants it: sh::pairUpdates, the PDE's rows)
  mg::tick("conf: slabs shared");
  float* pair = (float*)cz.local;
  float* base = (float*)((__nv_bfloat16*)cz.local - hf * C);
  float* single = scratch<float>("conf.single", (size_t)n * Cs);
  if (reembed) boltz2ReembedRows(base, single, trunkZ.local, trunkSingle, targetFeat, pseudoBeta, pairMask, n, C, Cs, F, lo, rowsHere);
  else {
  if (hp && !consume) CK(cudaMemcpyAsync(pair, trunkZ.local, hp * C * 2, cudaMemcpyDeviceToDevice, STREAM));
  CK(cudaMemcpyAsync(single, trunkSingle, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  float* lr = scratch<float>("conf.lr", 2 * (size_t)n * C); float *left = lr, *right = lr + (size_t)n * C;
  {
    std::string w2 = concatColumns(P + ".lrTargetFeat~stack", 1, {{P + ".leftTargetFeatProject", F * C, false},
                                                                  {P + ".rightTargetFeatProject", F * C, false}});
    const float one = 1.f, zero = 0.f;
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, C, n, F, &one, W(w2), CUDA_R_32F, C,
       (long long)F * C, targetFeat, CUDA_R_32F, F, 0, &zero, lr, CUDA_R_32F, C,
       (long long)n * C, 2, F32_TF32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
       F32_TF32 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
  }
  int bins = (int)(lenW(P + ".distogramFeatProject") / C);
  if (bins != (caDgram ? 40 : 39)) { fprintf(stderr, "confidence distogram has %d bins\n", bins); exit(1); }
  int* binOf = scratch<int>("conf.bin", pairs); float* sqOf = scratch<float>("conf.sq", pairs);
  confidenceBinK<<<blocks(pairs), 256, 0, STREAM>>>(pseudoBeta, n, bins, 3.25f, 50.75f, caDgram, binOf, sqOf, false);
  if (hp) confidencePairInitK<__nv_bfloat16><<<blocks((C & 3) == 0 ? (hp * (C >> 2)) : (hp * C)), 256, 0, STREAM>>>(base, left, right, binOf, sqOf, pairMask,
    W(P + ".distogramFeatProject"), n, C, Wopt(P + ".distanceFeatProject"), caDgram, false, hf * C, (hf + hp) * C);
  // (every pair's bin and distance, read by that init alone: 2.4 GB of whole-pair tables a rank at 17,290 tokens on 8)
  if (shortPair(pairs, C)) releaseScratch({ "conf.bin", "conf.sq" });
  if (hasW(P + ".inputSingleNormScale")) {
    clampK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(single, 512.f, (size_t)n * Cs);
    float* sn = scratch<float>("conf.singleNorm", (size_t)n * Cs);
    layerNorm2<float, float>(single, sn, n, Cs, P + ".inputSingleNormScale", P + ".inputSingleNormOffset");
    CK(cudaMemcpyAsync(single, sn, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  }
  foldDualOutputs();
  mg::tick("conf: pair initialised");
  int nb = 0; while (M.has(P + ".blocks." + std::to_string(nb) + ".singleChannels")) ++nb;
  bool swap = M.flag("trunk.dialect.swapTransposedBias"), divide = M.flag("trunk.dialect.triangleMulDivideByLength");
  const bool was16 = PAIR16;
  PAIR16 = true;
  for (int k = 0; k < nb; ++k) {
    const std::string B = P + ".blocks." + std::to_string(k);
    sh::pairUpdates<half>(cz, czTp, "sh.czT", pairMask, n, C, B, swap, divide, 4);
    mg::tick("conf: block pair");
    singleTrack<half>(single, pair, seqMask, n, C, Cs, B, nullptr, lo);
    mg::tick("conf: block single");
  }
  if (!czTp) {        // (a card short of room, the blocks' exchange buffers given back first: the triangle's product share
                      //  idle beside a new z^T share was 9 GB too many a rank at 11,856 tokens on 4)
    releaseScratch({ "tri.", "trib.", "grid.", "tr.", "st." }); trimPool();
    mg::release({ "sh.tp", "sh.tsa", "sh.tsb", "sh.zTc", "sh.bias" }, mg::FREE);
    czTp = &mg::shared("sh.czT", slab);
  }
  const mg::Shared& czT = *czTp;
  sh::transpose(cz, czT, n, C);              // (the PDE's transposed rows)
  mg::tick("conf: transposed");
  PAIR16 = was16;
  releaseScratch({ "tri.", "trib.", "grid.", "tr.", "st." });
  const int NB = 64; double step = 31.0 / (NB - 2);
  std::vector<float> centres(NB);
  for (int b = 0; b < NB - 1; ++b) centres[b] = (float)(b * step + step / 2);
  centres[NB - 1] = (float)(centres[NB - 2] + step);
  float* dCentres = scratch<float>("conf.centres", NB);
  CK(cudaMemcpyAsync(dCentres, centres.data(), NB * sizeof(float), cudaMemcpyHostToDevice, STREAM));
  double d0;
  {
    std::vector<float> seqBuf;
    const float* seq = hostSeqMask(seqMask, n, seqBuf);
    int real = 0; for (int i = 0; i < n; ++i) real += seq[i] > 0;
    d0 = 1.24 * std::cbrt(std::max(real, 19) - 15.0) - 1.8;
  }
  std::vector<float> perBin(NB);
  for (int b = 0; b < NB; ++b) perBin[b] = (float)(1 / (1 + (double)centres[b] * centres[b] / (d0 * d0)));
  float* dPerBin = scratch<float>("conf.perBin", NB);
  CK(cudaMemcpyAsync(dPerBin, perBin.data(), NB * sizeof(float), cudaMemcpyHostToDevice, STREAM));
  auto headNorm = [&](const float* x, float* buf, size_t rows, int Cx, const std::string& name) -> const float* {
    if (!hasW(P + "." + name + "Scale")) return x;
    layerNorm2<float, float>(x, buf, rows, Cx, P + "." + name + "Scale", P + "." + name + "Offset");
    return buf;
  };
  const bool preSym = M.flag("trunk.dialect.preSymmetrisedPde");
  const size_t held = (size_t)sh::maxStored(n) * n;
  mg::Shared& pdeS = mg::shared("sh.c.pde", std::max<size_t>(held, 1) * 4);
  mg::Shared& paeS = mg::shared("sh.c.pae", std::max<size_t>(held, 1) * 4);
  mg::Shared& tmS = mg::shared("sh.c.tm", std::max<size_t>(held, 1) * 4);
  float *pde = (float*)pdeS.local, *pae = (float*)paeS.local, *tm = (float*)tmS.local;
  const size_t rowsPer = std::max<size_t>(1, std::min(std::max<size_t>(hp, 1), CHUNK / C));
  float* ln = scratch<float>("conf.ln", rowsPer * C);
  float* ln2 = scratch<float>("conf.ln2", rowsPer * C);
  float* own = scratch<float>("conf.own", rowsPer * C);
  float* trn = scratch<float>("conf.symT", rowsPer * C);
  float* logits = scratch<float>("conf.logits", rowsPer * NB);
  // boltz2's pair heads: an intra-chain and an inter-chain projection, each pair taking its own
  const int* asymDev = Idev("batch.asymId");
  float* interLogits = hasW(P + ".interHalfDistanceLogits") || hasW(P + ".paeInterLogits") ? scratch<float>("conf.interLogits", rowsPer * NB) : nullptr;
  auto chunkLogits = [&](const float* xn, size_t r0, size_t cnt, const std::string& w, const std::string& inter) {
    linear<float, float>(xn, logits, cnt, C, NB, P + "." + w);
    if (!hasW(P + "." + inter)) return;
    linear<float, float>(xn, interLogits, cnt, C, NB, P + "." + inter);
    interChainLogitsChunkK<<<blocks(cnt * NB), 256, 0, STREAM>>>(logits, interLogits, asymDev, n, NB, hf + r0, cnt);
  };
  for (size_t r0 = 0; r0 < hp; r0 += rowsPer) {
    size_t cnt = std::min(rowsPer, hp - r0);
    ownRowsK<__nv_bfloat16><<<blocks(cnt * C), 256, 0, STREAM>>>(pair, own, r0, cnt, C);
    ownRowsK<__nv_bfloat16><<<blocks(cnt * C), 256, 0, STREAM>>>((const float*)czT.local, trn, r0, cnt, C);
    // the PDE: AF3 symmetrises after the projection (W LN(z_ij) + W LN(z_ji), through linearity one projection of the
    // summed norms), protenix2 before it (LN(z_ij + z_ji) W) - the one-GPU head's chunked forms
    const float* xn;
    if (preSym) {
      addK<<<blocks(cnt * C), 256, 0, STREAM>>>(trn, own, cnt * C);
      xn = headNorm(trn, ln, cnt, C, "logitsLn");
    } else {
      xn = headNorm(own, ln, cnt, C, "logitsLn");
      if (xn != ln) { CK(cudaMemcpyAsync(ln, xn, cnt * C * 4, cudaMemcpyDeviceToDevice, STREAM)); xn = ln; }
      addK<<<blocks(cnt * C), 256, 0, STREAM>>>(ln, headNorm(trn, ln2, cnt, C, "logitsLn"), cnt * C);
    }
    chunkLogits(xn, r0, cnt, "leftHalfDistanceLogits", "interHalfDistanceLogits");
    expectationK<<<blocks(cnt), 256, 0, STREAM>>>(logits, pde + r0, pairMask + hf + r0, cnt, NB, dCentres, 0, 1.f);
    chunkLogits(headNorm(own, ln, cnt, C, "paeLogitsLn"), r0, cnt, "paeLogits", "paeInterLogits");
    expectationK<<<blocks(cnt), 256, 0, STREAM>>>(logits, pae + r0, pairMask + hf + r0, cnt, NB, dCentres, 0, 1.f, tm + r0, dPerBin);
  }
  mg::tick("conf: heads");
  ConfidenceOut out;
  mg::fence(); mg::hostFence();
  if (mg::RANK == 0) {
    std::vector<float> term(pairs);
    out.pae.resize(pairs); out.pde.resize(pairs);
    for (int r = 0; r < mg::WORLD; ++r) {
      int rlo, rhi; sh::rowsOf(n, r, rlo, rhi); const size_t cntR = (size_t)sh::storedRows(n, r) * n;
      if (!cntR) continue;
      CK(cudaMemcpy(out.pae.data() + (size_t)rlo * n, paeS.peer[r], cntR * 4, cudaMemcpyDefault));
      CK(cudaMemcpy(out.pde.data() + (size_t)rlo * n, pdeS.peer[r], cntR * 4, cudaMemcpyDefault));
      CK(cudaMemcpy(term.data() + (size_t)rlo * n, tmS.peer[r], cntR * 4, cudaMemcpyDefault));
    }
    confidenceTm(out, std::move(term), seqMask, n);
    confidencePlddt(out, single, n, Cs, headNorm);
    CK(cudaStreamSynchronize(STREAM));
    finishConfidencePlddt(out, n, (int)M.meta("batch.dense"));
  }
  mg::tick("conf: gathered, pTM, pLDDT");
  mg::release({ "sh.cz", "sh.c." });       // (collective: the next sample's head makes its own)
  mg::tick("conf: released");
  return out;
}
