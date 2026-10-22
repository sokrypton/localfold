// ESMFold2's sliding-window atom transformer with 3D RoPE (cpu/esmfold2/atom-transformer.js
// is the reading): the inputs embedder, and the stacks the diffusion module's atom encoder and
// decoder reuse.
//
//   c0 = LN(atomFeatures @ linear);  q = c0 (+ coords @ coordsLinear in the diffusion's encoder)
//   3 x adaLN-Zero block, conditioned on c0 throughout:
//       mod = silu(c0) @ adaln -> shift_a scale_a gate_a shift_f scale_f gate_f
//       q  += gate_a * SWA(rms(q) (1 + scale_a) + shift_a)
//       q  += gate_f * swiglu(rms(q) (1 + scale_f) + shift_f)
//   SWA: qkv; q, k rms-normed PER HEAD then rotated; q, k, v narrowed to bfloat16 (the module casts
//   whatever the model's dtype); keys within halfWindow in rank among VALID atoms, the diagonal always;
//   out = (ctx * live * sigmoid(input @ attnGate)) @ attnOut
//   Every rms is affine-free with torch's eps (float32 epsilon, 1.19e-7).
#pragma once
#include "ops.cuh"

constexpr int ATOM_FEATURES = 3 + 1 + 1 + 128 + 4 * 64;   // 389
constexpr float RMS_EPS = 1.1920928955078125e-7f;
// the module narrows q, k, v to bfloat16 whatever the model's dtype; false is the control arm, held to
// an oracle written with oracle.py --float32-attention (the rope table stays bfloat16 either way)
inline bool ATOM_BF16 = true;

__device__ __forceinline__ float bf16Round(float v) {      // round to nearest even, back to float
  unsigned w = __float_as_uint(v);
  w = (w + 0x7fffu + ((w >> 16) & 1u)) & 0xffff0000u;
  return __uint_as_float(w);
}

__global__ void atomFeaturesK(const float* pos, const float* charge, const float* mask, const int* element,
                              const int* nameChars, float* out, int A) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * ATOM_FEATURES) return;
  int atom = (int)(t / ATOM_FEATURES), f = (int)(t % ATOM_FEATURES);
  bool live = mask[atom] != 0.f;
  float v = 0.f;
  if (f < 3) v = pos[atom * 3 + f];
  else if (f == 3) v = charge[atom];
  else if (f == 4) v = mask[atom];
  else if (f < 5 + 128) v = live && element[atom] == f - 5 ? 1.f : 0.f;
  else { int i = (f - 133) / 64, ch = (f - 133) % 64; v = live && nameChars[atom * 4 + i] == ch ? 1.f : 0.f; }
  out[t] = v;
}
// the rotary table [A, 16]: x's two pairs (base 20), y's, z's, then the space uid's ten (base 10000);
// spacing 1 / base^(i/n); stored in bfloat16 as the model stores it
__device__ __constant__ double ATOM_INV_FREQ[16] = {
  0x1.0000000000000p+0, 0x1.c9f25c5bfedd9p-3, 0x1.0000000000000p+0, 0x1.c9f25c5bfedd9p-3,
  0x1.0000000000000p+0, 0x1.c9f25c5bfedd9p-3, 0x1.0000000000000p+0, 0x1.97a967f7524b3p-2,
  0x1.44960c576b374p-3, 0x1.0270ac3f8a9fap-4, 0x1.9b8c272fbe6dep-6, 0x1.47ae147ae147bp-7,
  0x1.04e74cc73ee88p-8, 0x1.9f7861b7937a3p-10, 0x1.4acda94717d66p-11, 0x1.0763f01e8e5adp-12
};
__global__ void ropeTableK(const float* pos, const int* uid, float* cosT, float* sinT, int A) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= A * 16) return;
  int atom = t >> 4, at = t & 15;
  double coord = (at < 6) ? (double)pos[atom * 3 + (at >> 1)] : (double)uid[atom];
  float s, c;
  fastSincosD(coord * ATOM_INV_FREQ[at], &s, &c);
  cosT[t] = bf16Round(c);
  sinT[t] = bf16Round(s);
}
__global__ void rankK(const float* __restrict__ mask, int* __restrict__ rank, int A) {      // inclusive scan, 256 threads
  if (blockIdx.x) return;
  __shared__ int warpSums[8];
  __shared__ int base;
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) base = 0;
  __syncthreads();
  for (int a0 = 0; a0 < A; a0 += 256) {
    int a = a0 + tid;
    int v = (a < A && mask[a] != 0.f) ? 1 : 0;
    int s = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int n = __shfl_up_sync(~0u, s, o);
      if (lane >= o) s += n;
    }
    if (lane == 31) warpSums[warp] = s;
    __syncthreads();
    if (warp == 0) {
      int ws = (lane < 8) ? warpSums[lane] : 0;
      #pragma unroll
      for (int o = 1; o < 8; o <<= 1) {
        int n = __shfl_up_sync(~0u, ws, o);
        if (lane >= o) ws += n;
      }
      if (lane < 8) warpSums[lane] = ws;
    }
    __syncthreads();
    int prefix = base + (warp > 0 ? warpSums[warp - 1] : 0) + s;
    if (a < A) rank[a] = prefix - 1;
    if (tid == 255) base += warpSums[7];
    __syncthreads();
  }
}
__global__ void siluK(const float* __restrict__ x, float* __restrict__ y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)x | (uintptr_t)y) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 xv = __ldg(reinterpret_cast<const float4*>(x) + t);
      reinterpret_cast<float4*>(y)[t] = make_float4(siluF(xv.x), siluF(xv.y), siluF(xv.z), siluF(xv.w));
    }
    return;
  }
  if (t < n) y[t] = siluF(x[t]);
}
// rms(x) * (1 + mod[scale]) + mod[shift], a warp a row (C = 128); optionally fuses x += mod[gateWhich] * d first
__global__ void rmsModulateK(float* __restrict__ x, const float* __restrict__ mod, float* __restrict__ y,
                             int rows, int C, int shift, int scale,
                             const float* __restrict__ d = nullptr, int gateWhich = 0) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  float* xr = x + (size_t)row * C;
  float* yr = y + (size_t)row * C;
  const float* m = mod + (size_t)row * 6 * C;
  const float* dr = d ? d + (size_t)row * C : nullptr;
  if (C == 128) {
    float xv[4];
    float s = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int c = k * 32 + lane;
      float v = __ldg(xr + c);
      if (dr) {
        v += __ldg(m + gateWhich * 128 + c) * __ldg(dr + c);
        xr[c] = v;
      }
      xv[k] = v;
      s += v * v;
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float inv = rsqrtf(s * (1.f / 128.f) + RMS_EPS);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int c = k * 32 + lane;
      yr[c] = xv[k] * inv * (1.f + __ldg(m + scale * 128 + c)) + __ldg(m + shift * 128 + c);
    }
    return;
  }
  float s = 0;
  for (int c = lane; c < C; c += 32) {
    float v = xr[c];
    if (dr) { v += m[gateWhich * C + c] * dr[c]; xr[c] = v; }
    s += v * v;
  }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float inv = rsqrtf(s / C + RMS_EPS);
  for (int c = lane; c < C; c += 32) yr[c] = xr[c] * inv * (1.f + m[scale * C + c]) + m[shift * C + c];
}
// q and k (heads of 32, inside the packed [A, 3C] qkv): rms per head, rotate, narrow to bfloat16; v narrowed.
// One warp (32 lanes, d = 0..31) per (atom, head): coalesced reads/writes, cos/sin loaded once for both q and k.
__global__ void qkvPrepareK(float* __restrict__ qkv, const float* __restrict__ cosT, const float* __restrict__ sinT,
                            int A, int C, int heads, bool bf16) {
  int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int d = threadIdx.x & 31;
  if (warpId >= A * heads) return;
  int head = (heads == 4) ? (warpId & 3) : (warpId % heads);
  int atom = (heads == 4) ? (warpId >> 2) : (warpId / heads);
  float c = __ldg(cosT + atom * 16 + (d & 15));
  float sn = __ldg(sinT + atom * 16 + (d & 15));
  float* base = qkv + (size_t)atom * 3 * C + head * 32;
  #pragma unroll
  for (int part = 0; part < 2; ++part) {
    float* p = base + part * C;
    float x = p[d];
    float s = x * x;
    #pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float inv = rsqrtf(s * (1.f / 32.f) + RMS_EPS);
    float v = x * inv;
    float partner = __shfl_xor_sync(~0u, v, 16);
    float out = (d < 16) ? (v * c - partner * sn) : (v * c + partner * sn);
    p[d] = bf16 ? bf16Round(out) : out;
  }
  if (bf16) {
    float* pv = base + 2 * C;
    pv[d] = bf16Round(pv[d]);
  }
}
// the windowed attention itself, f32 throughout (no [heads, A, A] matrix): a valid query's keys are the
// valid atoms within halfWindow in rank, valid[rank - hw .. rank + hw], itself among them. A block takes
// 32 consecutive valid queries of one head and walks the union of their windows in tiles of SWA_KT keys
// staged in shared memory; a warp a query at a time, online softmax, a lane a key for the scores and a
// lane a channel for the output. An invalid atom's row is zero (gateLiveK zeroes it in any case).
constexpr int SWA_KT = 144, SWA_Q = 16;     // 16 queries (4 warps) and their whole +-64 window in one tile
__global__ void __launch_bounds__(128, 4) swaWindowK(const float* __restrict__ qkv, const int* __restrict__ valid,
                                                     int nValid, float* __restrict__ ctx, int C,
                                                     int halfWindow, float scale) {
  __shared__ float Ks[SWA_KT][33];
  __shared__ float Vs[SWA_KT][32];
  __shared__ float4 Qs[SWA_Q / 4][32];
  int head = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  int r0 = blockIdx.x * SWA_Q, r1 = min(nValid, r0 + SWA_Q) - 1;
  int klo = max(0, r0 - halfWindow), khi = min(nValid - 1, r1 + halfWindow);
  // each warp's four consecutive queries, together: one key read serves all four
  int rq = r0 + 4 * warp;
  float m[4], l[4], acc[4];
  float4 q4;
  #pragma unroll
  for (int u = 0; u < 4; ++u) { m[u] = -INFINITY; l[u] = 0.f; acc[u] = 0.f; }
  q4.x = __ldg(qkv + (size_t)valid[min(rq + 0, nValid - 1)] * 3 * C + head * 32 + lane) * scale;
  q4.y = __ldg(qkv + (size_t)valid[min(rq + 1, nValid - 1)] * 3 * C + head * 32 + lane) * scale;
  q4.z = __ldg(qkv + (size_t)valid[min(rq + 2, nValid - 1)] * 3 * C + head * 32 + lane) * scale;
  q4.w = __ldg(qkv + (size_t)valid[min(rq + 3, nValid - 1)] * 3 * C + head * 32 + lane) * scale;
  Qs[warp][lane] = q4;
  for (int t0 = klo; t0 <= khi; t0 += SWA_KT) {
    int n = min(SWA_KT, khi - t0 + 1);
    __syncthreads();
    for (int e = threadIdx.x; e < n * 8; e += blockDim.x) {
      int j = e >> 3, part = e & 7;
      const float* row = qkv + (size_t)valid[t0 + j] * 3 * C + head * 32 + part * 4;
      float4 k = *reinterpret_cast<const float4*>(row + C), v = *reinterpret_cast<const float4*>(row + 2 * C);
      Ks[j][part * 4] = k.x; Ks[j][part * 4 + 1] = k.y; Ks[j][part * 4 + 2] = k.z; Ks[j][part * 4 + 3] = k.w;
      *reinterpret_cast<float4*>(&Vs[j][part * 4]) = v;
    }
    __syncthreads();
    if (rq > r1) continue;
    // the union of the four windows inside this tile
    int lo = max(rq - halfWindow, t0) - t0, hi = min(min(rq + 3, r1) + halfWindow, t0 + n - 1) - t0;
    for (int j0 = lo; j0 <= hi; j0 += 32) {
      int j = j0 + lane, key = t0 + j;
      float s[4] = {0.f, 0.f, 0.f, 0.f};
      if (j <= hi) {
        #pragma unroll
        for (int d = 0; d < 32; ++d) {
          float k = Ks[j][d];
          float4 qv = Qs[warp][d];
          s[0] += qv.x * k;
          s[1] += qv.y * k;
          s[2] += qv.z * k;
          s[3] += qv.w * k;
        }
      }
      float p[4];
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        bool ok = j <= hi && abs(key - (rq + u)) <= halfWindow;
        float v = ok ? s[u] : -INFINITY, cm = v;
        #pragma unroll
        for (int o = 16; o; o >>= 1) cm = fmaxf(cm, __shfl_xor_sync(~0u, cm, o));
        float mn = fmaxf(m[u], cm);
        if (mn == -INFINITY) { p[u] = 0.f; continue; }      // nothing of this query's in the chunk yet
        float corr = (m[u] == -INFINITY) ? 0.f : (m[u] == mn) ? 1.f : expf(m[u] - mn);
        p[u] = ok ? expf(v - mn) : 0.f;
        float ps = p[u];
        #pragma unroll
        for (int o = 16; o; o >>= 1) ps += __shfl_xor_sync(~0u, ps, o);
        l[u] = l[u] * corr + ps; acc[u] *= corr; m[u] = mn;
      }
      if (j0 + 31 <= hi) {
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
          float v = Vs[j0 + k][lane];
          #pragma unroll
          for (int u = 0; u < 4; ++u) acc[u] += __shfl_sync(~0u, p[u], k) * v;
        }
      } else {
        int cnt = hi - j0 + 1;
        for (int k = 0; k < cnt; ++k) {
          float v = Vs[j0 + k][lane];
          #pragma unroll
          for (int u = 0; u < 4; ++u) acc[u] += __shfl_sync(~0u, p[u], k) * v;
        }
      }
    }
  }
  #pragma unroll
  for (int u = 0; u < 4; ++u)
    if (rq + u <= r1) ctx[(size_t)valid[rq + u] * C + head * 32 + lane] = acc[u] / l[u];
}
__global__ void gateLiveK(float* __restrict__ ctx, const float* __restrict__ gate, const float* __restrict__ mask, size_t A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)ctx | (uintptr_t)gate) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = A * (size_t)c4;
    if (t >= n4) return;
    size_t a = (c4 == 32) ? (t >> 5) : (t / (size_t)c4);
    if (__ldg(mask + a) == 0.f) {
      reinterpret_cast<float4*>(ctx)[t] = make_float4(0.f, 0.f, 0.f, 0.f);
      return;
    }
    float4 cv = reinterpret_cast<const float4*>(ctx)[t];
    float4 gv = __ldg(reinterpret_cast<const float4*>(gate) + t);
    reinterpret_cast<float4*>(ctx)[t] = make_float4(
        cv.x / (1.f + expf(-gv.x)),
        cv.y / (1.f + expf(-gv.y)),
        cv.z / (1.f + expf(-gv.z)),
        cv.w / (1.f + expf(-gv.w)));
    return;
  }
  if (t >= A * C) return;
  if (mask[t / C] == 0.f) { ctx[t] = 0.f; return; }
  ctx[t] /= (1.f + expf(-gate[t]));
}
__global__ void gatedAddK(float* __restrict__ x, const float* __restrict__ mod, const float* __restrict__ d, int A, int C, int which) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)x | (uintptr_t)mod | (uintptr_t)d) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = (size_t)A * c4;
    if (t >= n4) return;
    size_t a = (c4 == 32) ? (t >> 5) : (t / (size_t)c4);
    int c = (c4 == 32) ? (int)(t & 31) : (int)(t % (size_t)c4);
    float4 xv = reinterpret_cast<const float4*>(x)[t];
    float4 dv = __ldg(reinterpret_cast<const float4*>(d) + t);
    float4 mv = __ldg(reinterpret_cast<const float4*>(mod + a * 6 * C + which * C) + c);
    reinterpret_cast<float4*>(x)[t] = make_float4(xv.x + mv.x * dv.x, xv.y + mv.y * dv.y, xv.z + mv.z * dv.z, xv.w + mv.w * dv.w);
    return;
  }
  if (t >= (size_t)A * C) return;
  size_t a = t / C; int c = (int)(t % C);
  x[t] += mod[a * 6 * C + which * C + c] * d[t];
}

struct AtomCtx { int A, C, heads, hidden; const float* mask; const int* rank; const float* cosT; const float* sinT;
                 const int* valid; int nValid;                  // the valid atoms in order (rank -> atom)
                 const int* tokenStart; const int* tokenAtoms;  // each token's valid atoms (CSR)
               };

inline void dumpOnce(const char* name, const float* d, size_t n) {
  static std::set<std::string> done;
  const char* dir = getenv("EF2_DUMPDIR");
  if (!dir || done.count(name)) return;
  done.insert(name);
  auto h = download(d, n); FILE* f = fopen((std::string(dir) + "/" + name + ".bin").c_str(), "wb");
  fwrite(h.data(), 4, h.size(), f); fclose(f);
}
inline void swaBlock(const AtomCtx& a, float* x, const float* cond, const std::string& B, int halfWindow,
                     const float* preMod = nullptr, const float* preSilu = nullptr) {
  size_t A = a.A; int C = a.C;
  size_t acVec = (C & 3) == 0 ? A * (size_t)(C >> 2) : A * C;
  const float* mod = preMod;
  if (!mod) {
    const float* sc = preSilu;
    if (!sc) {
      float* scBuf = scratch<float>("atom.silu", A * C);
      siluK<<<blocks(acVec), 256, 0, STREAM>>>(cond, scBuf, A * C);
      sc = scBuf;
    }
    float* modBuf = scratch<float>("atom.mod", A * 6 * C);
    gemm(sc, F(B + "adaln"), modBuf, A, C, 6 * C);
    mod = modBuf;
  }
  float* xm = scratch<float>("atom.xm", A * C);
  rmsModulateK<<<(unsigned)((A + 7) / 8), 256, 0, STREAM>>>(x, mod, xm, (int)A, C, 0, 1);
  float* qkv = scratch<float>("atom.qkv", A * 3 * C);
  gemm(xm, F(B + "qkv"), qkv, A, C, 3 * C);
  dumpOnce("mod", mod, A * 6 * C); dumpOnce("xm", xm, A * C); dumpOnce("qkv_raw", qkv, A * 3 * C);
  qkvPrepareK<<<blocks(A * a.heads * 32, 256), 256, 0, STREAM>>>(qkv, a.cosT, a.sinT, (int)A, C, a.heads, ATOM_BF16);
  dumpOnce("qkv", qkv, A * 3 * C); dumpOnce("cos", a.cosT, A * 16);
  float* ctx = scratch<float>("atom.ctx", A * C);
  if (getenv("EF2_DUMPDIR")) CK(cudaMemsetAsync(ctx, 0, A * C * 4, STREAM));
  if (a.nValid)
    swaWindowK<<<dim3((unsigned)((a.nValid + SWA_Q - 1) / SWA_Q), (unsigned)a.heads), SWA_Q / 4 * 32, 0, STREAM>>>(
        qkv, a.valid, a.nValid, ctx, C, std::min(halfWindow, (int)A), 1.f / sqrtf(32.f));
  float* gate = scratch<float>("atom.gate", A * C);
  gemm(xm, F(B + "attnGate"), gate, A, C, C);
  dumpOnce("ctx", ctx, A * C);
  gateLiveK<<<blocks(acVec), 256, 0, STREAM>>>(ctx, gate, a.mask, A, C);
  float* att = scratch<float>("atom.att", A * C);
  gemm(ctx, F(B + "attnOut"), att, A, C, C);
  rmsModulateK<<<(unsigned)((A + 7) / 8), 256, 0, STREAM>>>(x, mod, xm, (int)A, C, 3, 4, att, 2);
  float* h = scratch<float>("atom.h", A * 2 * a.hidden); float* g = scratch<float>("atom.g", A * a.hidden);
  gemm(xm, F(B + "ffnUp"), h, A, C, 2 * a.hidden);
  swigluK<<<blocks((a.hidden & 3) == 0 ? A * (size_t)(a.hidden >> 2) : A * a.hidden), 256, 0, STREAM>>>(h, g, A, a.hidden);
  gemm(g, F(B + "ffnDown"), att, A, a.hidden, C);
  gatedAddK<<<blocks(acVec), 256, 0, STREAM>>>(x, mod, att, (int)A, C, 5);
}
inline void swaStack(const AtomCtx& a, float* x, const float* cond, const std::string& prefix, int blocksN, int halfWindow,
                     const float* preModStack = nullptr) {
  const float* sc = nullptr;
  if (!preModStack && blocksN > 0) {
    size_t A = a.A; int C = a.C;
    size_t acVec = (C & 3) == 0 ? A * (size_t)(C >> 2) : A * C;
    float* scBuf = scratch<float>("atom.silu", A * C);
    siluK<<<blocks(acVec), 256, 0, STREAM>>>(cond, scBuf, A * C);
    sc = scBuf;
  }
  size_t modStride = (size_t)a.A * 6 * a.C;
  for (int b = 0; b < blocksN; ++b)
    swaBlock(a, x, cond, prefix + "/blocks/" + std::to_string(b) + "/", halfWindow,
             preModStack ? preModStack + b * modStride : nullptr, sc);
}

__global__ void reluK(float* __restrict__ x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && ((uintptr_t)x & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 xv = reinterpret_cast<const float4*>(x)[t];
      reinterpret_cast<float4*>(x)[t] = make_float4(fmaxf(xv.x, 0.f), fmaxf(xv.y, 0.f), fmaxf(xv.z, 0.f), fmaxf(xv.w, 0.f));
    }
    return;
  }
  if (t < n) x[t] = fmaxf(x[t], 0.f);
}
// mean over each token's atoms, weighted by the mask, into s_inputs' first C columns (row stride ld)
__global__ void scatterMeanK(const float* v, const int* tokenStart, const int* tokenAtoms, const float* mask,
                             float* out, int T, int C, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // one thread per (token, channel)
  if (t >= (size_t)T * C) return;
  int token = (int)(t / C), c = (int)(t % C);
  float s = 0, w = 0;
  for (int k = tokenStart[token]; k < tokenStart[token + 1]; ++k) {
    int a = tokenAtoms[k];
    s += v[(size_t)a * C + c] * mask[a]; w += mask[a];
  }
  out[(size_t)token * ld + c] = s / fmaxf(w, 1e-9f);
}
__global__ void tailInputsK(const float* aatype, const float* profile, const float* delMean, float* out, int T, int C,
                            int K, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * (2 * K + 1)) return;
  int token = (int)(t / (2 * K + 1)), f = (int)(t % (2 * K + 1));
  float v = f < K ? aatype[token * K + f] : !profile ? 0.f : f < 2 * K ? profile[token * K + f - K] : delMean[token];
  out[(size_t)token * ld + C + f] = v;
}

// the atom side of an input, built once: features -> c0, the rope table, ranks
struct Atoms { AtomCtx ctx; float* c0; };
inline Atoms prepareAtoms(int A, const std::string& prefix) {
  Atoms at{};
  int C = (int)M.meta("meta/atomChannels");
  at.ctx.A = A; at.ctx.C = C; at.ctx.heads = (int)M.meta("meta/atomHeads");
  at.ctx.hidden = (int)dimOf("f/" + prefix + "/blocks/0/ffnDown", 0);
  at.ctx.mask = W("atom_mask");
  int* rank = dallocT<int>(A); rankK<<<1, 256, 0, STREAM>>>(at.ctx.mask, rank, A); at.ctx.rank = rank;
  {   // the valid atoms in order, and each token's (for the windowed attention and the pooling)
    std::vector<float> mask = download(at.ctx.mask, A);
    std::vector<int> toToken(A), valid;
    CK(cudaMemcpy(toToken.data(), Idev("atom_to_token"), (size_t)A * 4, cudaMemcpyDeviceToHost));
    int T = (int)M.meta("meta/tokens");
    std::vector<std::vector<int>> byToken(T);
    for (int a = 0; a < A; ++a) if (mask[a] != 0.f) { valid.push_back(a); byToken.at(toToken[a]).push_back(a); }
    std::vector<int> start{0}, atoms;
    for (auto& v : byToken) { atoms.insert(atoms.end(), v.begin(), v.end()); start.push_back((int)atoms.size()); }
    at.ctx.nValid = (int)valid.size();
    at.ctx.valid = upload(valid.data(), std::max<size_t>(valid.size(), 1));
    at.ctx.tokenStart = upload(start.data(), start.size());
    at.ctx.tokenAtoms = upload(atoms.data(), std::max<size_t>(atoms.size(), 1));
  }
  float* cosT = dalloc((size_t)A * 16); float* sinT = dalloc((size_t)A * 16);
  ropeTableK<<<blocks((size_t)A * 16), 256, 0, STREAM>>>(W("ref_pos"), Idev("ref_space_uid"), cosT, sinT, A);
  at.ctx.cosT = cosT; at.ctx.sinT = sinT;
  float* feat = scratch<float>("atom.features", (size_t)A * ATOM_FEATURES);
  atomFeaturesK<<<blocks((size_t)A * ATOM_FEATURES), 256, 0, STREAM>>>(W("ref_pos"), W("ref_charge"), at.ctx.mask,
    Idev("ref_element"), Idev("ref_atom_name_chars"), feat, A);
  at.c0 = dalloc((size_t)A * C);
  gemm(feat, F(prefix + "/linear"), at.c0, A, ATOM_FEATURES, C);
  layerNorm(at.c0, at.c0, A, C, F(prefix + "/norm/scale"), F(prefix + "/norm/offset"));
  return at;
}

inline void freeAtoms(Atoms& at) {
  CK(cudaStreamSynchronize(STREAM));
  for (const void* p : {(const void*)at.c0, (const void*)at.ctx.rank, (const void*)at.ctx.cosT, (const void*)at.ctx.sinT,
                        (const void*)at.ctx.valid, (const void*)at.ctx.tokenStart, (const void*)at.ctx.tokenAtoms})
    CK(cudaFree((void*)p));
  at = Atoms{};
}
// the inputs embedder: s_inputs [T, 451] = [pool(relu(stack(c0) @ toToken)) | aatype | profile | deletion mean]
// halfWindow 64 is biohub's esm package - the vendor's code, which windows every atom stack (flash-attn
// window_size=(64, 64) on CUDA, the rank mask on the CPU). Dense is the page's reading, from Synthyra's
// fastplms, which never windows this stage (docs/EF2FAST.md); --inputs-window=0 is that arm. Measured
// natively against the crystals, neither reading wins everywhere: 6MRR 0.92 A windowed against 1.59
// dense (five seeds), 1QYS 0.89 against 0.94, 5CAJ 2.10 against 1.86 (three)
inline int INPUTS_HALF_WINDOW = 64;
inline void inputsEmbedder(int T, int A, float* sInputs, int sWidth, bool check = false) {
  Atoms at = prepareAtoms(A, "atom");
  if (check) checkOracle("atom norm (c0)", at.c0, (size_t)A * at.ctx.C, "o/atom/norm");
  int C = at.ctx.C, Ct = (int)dimOf("f/atom/toToken", 1), K = (int)M.meta("meta/classes");
  float* x = scratch<float>("embed.x", (size_t)A * C);
  CK(cudaMemcpyAsync(x, at.c0, (size_t)A * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  size_t acVec = (C & 3) == 0 ? (size_t)A * (C >> 2) : (size_t)A * C;
  float* sc = scratch<float>("atom.silu", (size_t)A * C);
  siluK<<<blocks(acVec), 256, 0, STREAM>>>(at.c0, sc, (size_t)A * C);
  for (int b = 0; b < (int)M.meta("meta/atomBlocks"); ++b) {
    swaBlock(at.ctx, x, at.c0, "atom/blocks/" + std::to_string(b) + "/", INPUTS_HALF_WINDOW, nullptr, sc);
    if (check) checkOracle(("atom block " + std::to_string(b)).c_str(), x, (size_t)A * C, "o/atom/block" + std::to_string(b));
  }
  float* tok = scratch<float>("embed.tok", (size_t)A * Ct);
  gemm(x, F("atom/toToken"), tok, A, C, Ct);
  reluK<<<blocks((Ct & 3) == 0 ? (size_t)A * (Ct >> 2) : (size_t)A * Ct), 256, 0, STREAM>>>(tok, (size_t)A * Ct);
  scatterMeanK<<<blocks((size_t)T * Ct), 256, 0, STREAM>>>(tok, at.ctx.tokenStart, at.ctx.tokenAtoms, at.ctx.mask, sInputs, T, Ct, sWidth);
  // the alignment's profile and mean deletion only where the checkpoint reads them: the released models do; the
  // experimental tier (disable_msa_features) zeroes both, and its bundles predate the field
  const bool msaFeatures = M.has("meta/msaFeatures") && M.meta("meta/msaFeatures") > 0;
  tailInputsK<<<blocks((size_t)T * (2 * K + 1)), 256, 0, STREAM>>>(W("aatype"), msaFeatures ? W("profile") : nullptr,
                                                                 msaFeatures ? W("deletion_mean") : nullptr, sInputs,
                                                                 T, Ct, K, sWidth);
  freeAtoms(at);
}
