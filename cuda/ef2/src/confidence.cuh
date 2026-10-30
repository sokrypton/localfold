// Synthyra's ESMFold2 confidence head (biohub ships the checkpoint without one; Synthyra trained it on
// the same frozen trunk). cpu/esmfold2/confidence.js and their ConfidenceHead are the reading:
//
//   s = LN(s_inputs);  z = LN(z_trunk) + rel_pos + bonds + s_to_z(s)_i + s_to_z_T(s)_j + prod_out(in1(s)_i * in2(s)_j)
//   z += distance_embed[#(|x_rep_i - x_rep_j| > boundaries)]
//   z = z + trunk_4_blocks(z)                       (the trunk's own pair block, its own weights)
//   single = (softmax_j(z @ attn) z) @ out          (row-attention pooling)
//   pLDDT per atom = E[bin] over [0, 1] of LN(single[token]) @ plddtWeight[slot in token]
//   PAE = E[bin] over [0, 32] of z @ pae;  pTM / ipTM from the PAE logits (d0 from the live token count)
#pragma once
#include "trunk.cuh"

__global__ void confZK(float* z, RelIdx rel, Bonds bonds, const float* wBond, const float* rows,
                       const float* cols, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  z[t] += relPosAt(rel, i, j, c, C) + bondAt(bonds, i, j) * wBond[c] + rows[(size_t)i * C + c] + cols[(size_t)j * C + c];
}
// Fused LayerNorm(zTrunk; zNorm) + confZK / confZRowsK for C == 256, a warp per pair row
__global__ void __launch_bounds__(256, 2)
confZNorm256K(const float* __restrict__ zIn, float* __restrict__ zOut,
              const float* __restrict__ scale, const float* __restrict__ offset,
              RelIdx rel, Bonds bonds, const float* __restrict__ wBond,
              const float* __restrict__ rows, const float* __restrict__ cols,
              size_t p0, size_t n, int T, bool z16 = false) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= n) return;
  int lane = threadIdx.x & 31;
  size_t ij = p0 + row;
  int i = (int)(ij / (size_t)T), j = (int)(ij % (size_t)T);
  const int rb = 32, cb = 2, w = 2 * rb + 2;
  auto clip = [](int v, int hi) { return v < 0 ? 0 : v > hi ? hi : v; };
  int ai = __ldg(rel.asym + i), aj = __ldg(rel.asym + j);
  int ri = __ldg(rel.ri + i), rj = __ldg(rel.ri + j);
  bool sameChain = ai == aj, sameRes = ri == rj, sameEnt = __ldg(rel.ent + i) == __ldg(rel.ent + j);
  int b0 = sameChain ? clip(ri - rj + rb, 2 * rb) : 2 * rb + 1;
  int b1 = sameChain && sameRes ? clip(__ldg(rel.ti + i) - __ldg(rel.ti + j) + rb, 2 * rb) : 2 * rb + 1;
  int b3 = sameChain ? 2 * cb + 1 : clip(__ldg(rel.sym + i) - __ldg(rel.sym + j) + cb, 2 * cb);
  const float* w0 = rel.Wt + (size_t)b0 * 256;
  const float* w1 = rel.Wt + (size_t)(w + b1) * 256;
  const float* w2 = rel.Wt + (size_t)(2 * w) * 256;
  const float* w3 = rel.Wt + (size_t)(2 * w + 1 + b3) * 256;
  float bond = bondAt(bonds, i, j);
  float xv[8];
  float s = 0.f;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    int c = k * 32 + lane;
    float v = z16 ? __bfloat162float(__ldg(reinterpret_cast<const __nv_bfloat16*>(zIn) + row * 256 + c))
                  : __ldg(zIn + row * 256 + c);
    xv[k] = v;
    s += v;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / 256.f, v = 0.f;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    float d = xv[k] - mean;
    v += d * d;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / 256.f + 1e-5f);
  float* yr = zOut + row * 256;
  const float* riRow = rows + (size_t)i * 256;
  const float* cjRow = cols + (size_t)j * 256;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    int c = k * 32 + lane;
    float n = (xv[k] - mean) * inv * __ldg(scale + c) + __ldg(offset + c);
    float rp = __ldg(w0 + c) + __ldg(w1 + c) + __ldg(w3 + c);
    if (sameEnt) rp += __ldg(w2 + c);
    yr[c] = n + (rp + bond * __ldg(wBond + c) + __ldg(riRow + c) + __ldg(cjRow + c));
  }
}
__global__ void outerProductK(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out,
                              size_t p0, size_t n, int T, int C) {   // a_i * b_j, rows p0..
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)a | (uintptr_t)b | (uintptr_t)out) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = n * (size_t)c4;
    if (t >= n4) return;
    size_t q = ((c4 & (c4 - 1)) == 0) ? (t >> (__ffs(c4) - 1)) : (t / c4);
    int c = ((c4 & (c4 - 1)) == 0) ? (int)(t & (size_t)(c4 - 1)) : (int)(t % c4);
    size_t ij = p0 + q;
    int i = (int)(ij / (size_t)T), j = (int)(ij % (size_t)T);
    float4 av = __ldg(reinterpret_cast<const float4*>(a + (size_t)i * C) + c);
    float4 bv = __ldg(reinterpret_cast<const float4*>(b + (size_t)j * C) + c);
    reinterpret_cast<float4*>(out)[t] = make_float4(av.x * bv.x, av.y * bv.y, av.z * bv.z, av.w * bv.w);
    return;
  }
  if (t >= n * C) return;
  int c = (int)(t % C); size_t ij = p0 + t / C; int i = (int)(ij / T), j = (int)(ij % T);
  out[t] = a[(size_t)i * C + c] * b[(size_t)j * C + c];
}
// a warp a pair: its distance and bucket once (a ballot over the edges), then the row 16 bytes a lane - an
// element a thread took the distance and walked every edge for each of the C channels (12 ms of a 1,044-token
// confidence head's 194; C a multiple of 4)
__global__ void distanceEmbedK(float* z, const float* x, const int* rep, const float* edges, int nEdges,
                               const float* table, int T, int C, __nv_bfloat16* z16Out = nullptr) {
  size_t ij = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5; int lane = threadIdx.x & 31;
  if (ij >= (size_t)T * T) return;
  int i = (int)(ij / T), j = (int)(ij % T);
  const float* a = x + (size_t)rep[i] * 3; const float* b = x + (size_t)rep[j] * 3;
  float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  float d = sqrtf(dx * dx + dy * dy + dz * dz);
  int bucket = 0;
  for (int e0 = 0; e0 < nEdges; e0 += 32) bucket += __popc(__ballot_sync(~0u, e0 + lane < nEdges && d > edges[e0 + lane]));
  float4* row = reinterpret_cast<float4*>(z + ij * C);
  const float4* tb = reinterpret_cast<const float4*>(table + (size_t)bucket * C);
  uint2* row16 = z16Out ? reinterpret_cast<uint2*>(z16Out + ij * C) : nullptr;
  for (int c = lane; c < C / 4; c += 32) {
    float4 v = row[c], w = tb[c];
    v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w;
    row[c] = v;
    if (row16) {
      __nv_bfloat162 b0 = __floats2bfloat162_rn(v.x, v.y);
      __nv_bfloat162 b1 = __floats2bfloat162_rn(v.z, v.w);
      row16[c] = make_uint2(*reinterpret_cast<uint32_t*>(&b0), *reinterpret_cast<uint32_t*>(&b1));
    }
  }
}
// ...a chunk of pair positions [p0, p0 + n) of z (EF2_Z16's head, which builds its pair a chunk of rows at a time)
__global__ void confZRowsK(float* z, RelIdx rel, Bonds bonds, const float* wBond, const float* rows,
                           const float* cols, size_t p0, size_t n, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * C) return;
  int c = (int)(t % C); size_t ij = p0 + t / C; int i = (int)(ij / T), j = (int)(ij % T);
  z[t] += relPosAt(rel, i, j, c, C) + bondAt(bonds, i, j) * wBond[c] + rows[(size_t)i * C + c] + cols[(size_t)j * C + c];
}
__global__ void distanceEmbedRowsK(float* z, const float* x, const int* rep, const float* edges, int nEdges,
                                   const float* table, size_t p0, size_t n, int T, int C) {
  size_t q = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5; int lane = threadIdx.x & 31;
  if (q >= n) return;
  size_t ij = p0 + q; int i = (int)(ij / T), j = (int)(ij % T);
  const float* a = x + (size_t)rep[i] * 3; const float* b = x + (size_t)rep[j] * 3;
  float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  float d = sqrtf(dx * dx + dy * dy + dz * dz);
  int bucket = 0;
  for (int e0 = 0; e0 < nEdges; e0 += 32) bucket += __popc(__ballot_sync(~0u, e0 + lane < nEdges && d > edges[e0 + lane]));
  float4* row = reinterpret_cast<float4*>(z + q * C);
  const float4* tb = reinterpret_cast<const float4*>(table + (size_t)bucket * C);
  for (int c = lane; c < C / 4; c += 32) { float4 v = row[c], w = tb[c]; v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w; row[c] = v; }
}
// row-attention pooling: pooled[i] = sum_j softmax_j(score[i, j]) z[i, j]   (a block per row, C <= 1024)
__global__ void rowPoolK(const float* __restrict__ z, const float* __restrict__ score, float* __restrict__ pooled, int T, int C) {
  int i = blockIdx.x;
  __shared__ float m, s;
  __shared__ float sw[2048];
  const float* sc = score + (size_t)i * T;
  if (threadIdx.x == 0) {
    float mx = -INFINITY;
    for (int j = 0; j < T; ++j) mx = fmaxf(mx, sc[j]);
    float sum = 0;
    for (int j = 0; j < T; ++j) sum += expf(sc[j] - mx);
    m = mx; s = sum;
  }
  __syncthreads();
  float mx = m, sum = s;
  if (T <= 2048) {
    for (int j = threadIdx.x; j < T; j += blockDim.x) sw[j] = expf(sc[j] - mx) / sum;
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
      float acc = 0;
      const float* zr = z + (size_t)i * T * C + c;
      for (int j = 0; j < T; ++j) acc += sw[j] * zr[(size_t)j * C];
      pooled[(size_t)i * C + c] = acc;
    }
    return;
  }
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = 0;
    for (int j = 0; j < T; ++j) acc += expf(sc[j] - mx) / sum * z[((size_t)i * T + j) * C + c];
    pooled[(size_t)i * C + c] = acc;
  }
}
__global__ void gatherAtomsK(const float* __restrict__ tok, const int* __restrict__ atomToToken, float* __restrict__ out, int A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)tok | (uintptr_t)out) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = (size_t)A * c4;
    if (t < n4) {
      int a = (int)(t / (size_t)c4), c = (int)(t % (size_t)c4);
      reinterpret_cast<float4*>(out)[t] = __ldg(reinterpret_cast<const float4*>(tok + (size_t)atomToToken[a] * C) + c);
    }
    return;
  }
  if (t < (size_t)A * C) out[t] = tok[(size_t)atomToToken[t / C] * C + t % C];
}
// per atom: logits over bins from its slot's table, then the expectation over [0, 1]
__global__ void plddtAtomK(const float* __restrict__ s, const int* __restrict__ atomToToken,
                           const int* __restrict__ slot, const float* __restrict__ table,
                           float* __restrict__ plddt, int A, int C, int bins) {
  int a = blockIdx.x, tid = threadIdx.x;
  __shared__ float shS[512];
  __shared__ float logit[64];
  __shared__ float mxSh;
  const float* sa = s + (size_t)(atomToToken ? __ldg(atomToToken + a) : a) * C;
  if (C <= 512) {
    for (int c = tid; c < C; c += blockDim.x) shS[c] = __ldg(sa + c);
    __syncthreads();
  }
  const float* tb = table + (size_t)slot[a] * C * bins;
  if (C == 384 && bins == 50) {
    if (tid < 50) {
      float acc = 0;
      const float* tbb = tb + tid;
      #pragma unroll 8
      for (int c = 0; c < 384; ++c) acc += shS[c] * __ldg(tbb + c * 50);
      logit[tid] = acc;
    }
  } else {
    for (int b = tid; b < bins; b += blockDim.x) {
      float acc = 0;
      const float* tbb = tb + b;
      if (C <= 512) {
        #pragma unroll 4
        for (int c = 0; c < C; ++c) acc += shS[c] * __ldg(tbb + (size_t)c * bins);
      } else {
        for (int c = 0; c < C; ++c) acc += sa[c] * tbb[(size_t)c * bins];
      }
      logit[b] = acc;
    }
  }
  __syncthreads();
  if (tid == 0) {
    float mx = -INFINITY;
    for (int b = 0; b < bins; ++b) mx = fmaxf(mx, logit[b]);
    mxSh = mx;
  }
  __syncthreads();
  float mx = mxSh;
  for (int b = tid; b < bins; b += blockDim.x) logit[b] = expf(logit[b] - mx);
  __syncthreads();
  if (tid == 0) {
    float tot = 0, e = 0;
    for (int b = 0; b < bins; ++b) { float w = logit[b]; tot += w; e += w * (b + 0.5f) / bins; }
    plddt[a] = e / tot;
  }
}

// PAE and the pTM terms from the logits, a thread a pair: pae = sum p_b centre_b, tm = sum p_b / (1 + (centre_b / d0)^2);
// then a block a row: the row's mean tm over every column and over the other chains' columns (double)
__global__ void paeK(const float* __restrict__ logits, float* __restrict__ pae, float* __restrict__ tm,
                     size_t P, int bins, double width, double d0) {
  __shared__ double shCentre[64], shDenom[64];
  if (bins <= 64 && threadIdx.x < bins) {
    int b = threadIdx.x;
    double centre = width * (b + 0.5), r = centre / d0;
    shCentre[b] = centre;
    shDenom[b] = 1.0 / (1.0 + r * r);
  }
  __syncthreads();
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= P) return;
  const float* lg = logits + ij * bins;
  if (bins == 64 && ((uintptr_t)lg & 15) == 0) {
    const float4* lg4 = reinterpret_cast<const float4*>(lg);
    float4 v[16];
    float mx = -INFINITY;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
      float4 q = __ldg(lg4 + k);
      v[k] = q;
      mx = fmaxf(mx, fmaxf(fmaxf(q.x, q.y), fmaxf(q.z, q.w)));
    }
    double tot = 0, mean = 0, t = 0;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
      int b0 = k * 4;
      double w0 = exp((double)v[k].x - mx); tot += w0; mean += w0 * shCentre[b0 + 0]; t += w0 * shDenom[b0 + 0];
      double w1 = exp((double)v[k].y - mx); tot += w1; mean += w1 * shCentre[b0 + 1]; t += w1 * shDenom[b0 + 1];
      double w2 = exp((double)v[k].z - mx); tot += w2; mean += w2 * shCentre[b0 + 2]; t += w2 * shDenom[b0 + 2];
      double w3 = exp((double)v[k].w - mx); tot += w3; mean += w3 * shCentre[b0 + 3]; t += w3 * shDenom[b0 + 3];
    }
    pae[ij] = (float)(mean / tot); tm[ij] = (float)(t / tot);
    return;
  }
  float mx = -INFINITY;
  for (int b = 0; b < bins; ++b) mx = fmaxf(mx, lg[b]);
  double tot = 0, mean = 0, t = 0;
  for (int b = 0; b < bins; ++b) {
    double w = exp((double)lg[b] - mx), centre = width * (b + 0.5), r = centre / d0;
    tot += w; mean += w * centre; t += w / (1 + r * r);
  }
  pae[ij] = (float)(mean / tot); tm[ij] = (float)(t / tot);
}
__global__ void tmRowsK(const float* tm, const int* asym, double* rows, int T) {   // rows [T][2]: mean, inter-chain mean
  int i = blockIdx.x;
  __shared__ double s[2][256], c[256];
  double sum = 0, isum = 0, icnt = 0;
  for (int j = threadIdx.x; j < T; j += blockDim.x) {
    double v = tm[(size_t)i * T + j]; sum += v;
    if (asym[i] != asym[j]) { isum += v; icnt += 1; }
  }
  s[0][threadIdx.x] = sum; s[1][threadIdx.x] = isum; c[threadIdx.x] = icnt;
  __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) {
    if (threadIdx.x < o) { s[0][threadIdx.x] += s[0][threadIdx.x + o]; s[1][threadIdx.x] += s[1][threadIdx.x + o]; c[threadIdx.x] += c[threadIdx.x + o]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { rows[2 * i] = s[0][0] / (T + 1e-8); rows[2 * i + 1] = s[1][0] / (c[0] + 1e-8); }
}
// each anchor row's TM terms summed per chain - out [i0 + row][nc] - which is all the chain-level pTM and ipTM read:
// T x nc numbers where the matrix is T x T, so the sharded head can all-gather them at any size (rows: block's rows)
__global__ void tmChainRowsK(const float* tm, const int* chainOf, double* out, int T, int nc, int i0) {
  extern __shared__ double acc[];
  const int r = blockIdx.x;
  for (int k = threadIdx.x; k < nc; k += blockDim.x) acc[k] = 0;
  __syncthreads();
  for (int j = threadIdx.x; j < T; j += blockDim.x) atomicAdd(&acc[chainOf[j]], (double)tm[(size_t)r * T + j]);
  __syncthreads();
  for (int k = threadIdx.x; k < nc; k += blockDim.x) out[(size_t)(i0 + r) * nc + k] = acc[k];
}
// the chains by asym id, sorted, and each token's index among them - the writer's order, and the kernel's
inline std::vector<int> chainOrder(const std::vector<int>& asym, std::vector<int>& chainOf) {
  std::vector<int> ids;
  for (int a : asym) if (std::find(ids.begin(), ids.end(), a) == ids.end()) ids.push_back(a);
  std::sort(ids.begin(), ids.end());
  if (ids.size() * sizeof(double) > 48 * 1024) throw std::runtime_error("the confidence head's per-chain TM sums hold 6144 chains at most");
  chainOf.resize(asym.size());
  for (size_t i = 0; i < asym.size(); ++i) chainOf[i] = (int)(std::find(ids.begin(), ids.end(), asym[i]) - ids.begin());
  return ids;
}

// `tmChainRows`: [T][chains] - each anchor's TM terms summed over each chain (tmChainRowsK), which every chain-level pTM
// and ipTM is a max-of-means over. `atomToken`/`atomMask`: for each chain's mean pLDDT
struct Confidence { std::vector<float> plddtAtom, plddtToken, pae, atomMask; std::vector<double> tmChainRows; std::vector<int> atomToken; double ptm, iptm, meanPlddt; };

// consume: the trunk's z is read by nothing after this call - on a card short of room the head then works in it
inline Confidence confidenceHead(int T, int A, const float* zTrunk, const float* sInputs, int Si, const float* xDevice,
                                 bool check, bool consume = false, __nv_bfloat16* z16Kept = nullptr) {
  // z16Kept: EF2_Z16's trunk pair, kept on the device through the sampler (ef2.cu, where there was the room) - the
  // head builds its own pair over it in place; null where it is parked in PARK_HOST
  int C = (int)dimOf("f/confidence/sToZ", 1), Cs = (int)dimOf("f/confidence/poolingOutput", 1);
  size_t P = (size_t)T * T;
  float* s = scratch<float>("cf.s", (size_t)T * Si);
  layerNorm(sInputs, s, T, Si, F("confidence/sInputsNorm/scale"), F("confidence/sInputsNorm/offset"));
  // on a card short of room, a last call normalises the trunk's z in place (each lane reads its channels before
  // writing them): a 256-channel f32 pair fewer, 4.1 GB at 2000 tokens
  // (tight only where the device lacks the room for the two pair-sized tensors the roomy form holds: parking
  // the residual on the host was 1.17 s of a 1,044-token fold on a 40 GB A100 - a 1.1 GB pinned allocation
  // and a copy each way - with 30 GB free; LOCALFOLD_BIG=1 still takes the parked form)
  bool tight = shortPair(P, C) && !roomFor(2 * P * C * 4), inPlace = consume;
  float* r = scratch<float>("cf.r", (size_t)T * C); float* c = scratch<float>("cf.c", (size_t)T * C);
  float* l = scratch<float>("cf.l", (size_t)T * C); float* rr = scratch<float>("cf.rr", (size_t)T * C);
  gemm(s, F("confidence/sToZ"), r, T, Si, C);
  gemm(s, F("confidence/sToZTranspose"), c, T, Si, C);
  gemm(s, F("confidence/sToZProdIn1"), l, T, Si, C);
  gemm(s, F("confidence/sToZProdIn2"), rr, T, Si, C);
  // EF2_Z16 (zTrunk null: the trunk's bf16 pair parked in PARK_HOST): the head's pair built a chunk of token rows at
  // a time - normalised, its single and distance terms added, the sum kept (bf16) as the residual -
  // into a bf16 stack its blocks run on in place; then a chunk at a time again, the residual added back, the pooling
  // score, the row pooling and the PAE logits taken off the f32 rows. One bf16 pair on the card where the f32 head
  // held one f32 pair beside the blocks' (the head was the next limit: 30 GB at 5,434 tokens)
  float* pooled16 = nullptr; float* paeL16 = nullptr;
  int pb = (int)dimOf("f/confidence/pae", 1);
  if (!zTrunk) {
    if (!consume) { fprintf(stderr, "a parked bf16 trunk pair is read by the last confidence call only\n"); exit(1); }
    using B16 = __nv_bfloat16;
    const int rb = (int)std::max<size_t>(1, ((size_t)64 << 20) / ((size_t)T * C * 4));   // token rows a chunk
    const size_t per = (size_t)rb * T;
    // the trunk's pair on the device (z16Kept) - the stack built over it in place, a chunk read before it is
    // written - or parked; the residual is the stack as its blocks take it (bf16), kept on the device beside a kept
    // pair, else in the parked pair's own pinned buffer once read (an f32 residual pinned of its own was 6.3 s of a
    // 2,964-token fold on 40 GB, 9 GB of pinning)
    const bool kept = z16Kept != nullptr;
    B16* stack = kept ? z16Kept : dallocT<B16>(P * C);
    float* zc = scratch<float>("cf.zc", per * C); float* xn = scratch<float>("cf.xn", per * C);
    float* prodc = scratch<float>("cf.prodc", per * C); B16* stage = scratch<B16>("cf.z16", per * C);
    for (size_t p0 = 0; p0 < P; p0 += per) {
      size_t n = std::min(per, P - p0);
      const B16* src = stack + p0 * C;
      if (!kept) { CK(cudaMemcpyAsync(stage, reinterpret_cast<const B16*>(PARK_HOST) + p0 * C, n * C * 2, cudaMemcpyHostToDevice, STREAM)); src = stage; }
      if (C == 256) {
        confZNorm256K<<<(unsigned)((n + 7) / 8), 256, 0, STREAM>>>(
            reinterpret_cast<const float*>(src), zc, F("confidence/zNorm/scale"), F("confidence/zNorm/offset"),
            relIdx(), BONDS, F("featuriser/tokenBonds"), r, c, p0, n, T, true);
      } else {
        ef2FromBf16K<<<blocks(n * C), 256, 0, STREAM>>>(src, xn, n * C);
        layerNorm(xn, zc, n, C, F("confidence/zNorm/scale"), F("confidence/zNorm/offset"));
        confZRowsK<<<blocks(n * C), 256, 0, STREAM>>>(zc, relIdx(), BONDS, F("featuriser/tokenBonds"), r, c, p0, n, T, C);
      }
      outerProductK<<<blocks((C & 3) == 0 ? n * (size_t)(C >> 2) : n * C), 256, 0, STREAM>>>(l, rr, prodc, p0, n, T, C);
      gemm(prodc, F("confidence/sToZProdOut"), zc, n, C, C, 1.f);
      distanceEmbedRowsK<<<blocks(n * 32), 256, 0, STREAM>>>(zc, xDevice, Idev("distogram_atom_idx"), F("confidence/boundaries"),
        (int)M.len("f/confidence/boundaries"), F("confidence/distanceEmbedding"), p0, n, T, C);
      ef2ToBf16K<<<blocks(n * C), 256, 0, STREAM>>>(zc, stack + p0 * C, n * C);
    }
    releaseScratch({ "cf.zc", "cf.xn", "cf.prodc", "cf.z16" });
    B16* resid = kept ? dallocT<B16>(P * C) : reinterpret_cast<B16*>(PARK_HOST);
    CK(cudaMemcpyAsync(resid, stack, P * C * 2, cudaMemcpyDefault, STREAM));
    float* mask = scratch<float>("trunk.mask", P);
    fillK<<<blocks(P), 256, 0, STREAM>>>(mask, 1.f, P);
    PAIR16 = true;
    for (int b = 0; M.has("f/confidence/blocks/" + std::to_string(b) + "/pairTransition/transition1"); ++b)
      trunkBlock(reinterpret_cast<float*>(stack), mask, T, C, "confidence/blocks", b);
    PAIR16 = false;
    releaseScratch({ "ftri.", "ftr.", "trib." });
    pooled16 = scratch<float>("cf.pooled", (size_t)T * C);
    paeL16 = scratch<float>("cf.pae", P * pb);
    zc = scratch<float>("cf.zc", per * C); xn = scratch<float>("cf.xn", per * C);
    float* score = scratch<float>("cf.scoreRows", per);
    B16* rh = kept ? nullptr : scratch<B16>("cf.residRows", per * C);
    for (size_t p0 = 0; p0 < P; p0 += per) {
      size_t n = std::min(per, P - p0);
      const B16* rs = resid + p0 * C;
      if (rh) { CK(cudaMemcpyAsync(rh, rs, n * C * 2, cudaMemcpyHostToDevice, STREAM)); rs = rh; }
      ef2FromBf16K<<<blocks(n * C), 256, 0, STREAM>>>(rs, xn, n * C);
      ef2FromBf16K<<<blocks(n * C), 256, 0, STREAM>>>(stack + p0 * C, zc, n * C);
      addK<<<blocks(n * C), 256, 0, STREAM>>>(zc, xn, n * C);
      gemm(zc, F("confidence/poolingAttention"), score, n, C, 1);
      rowPoolK<<<(unsigned)(n / T), 256, 0, STREAM>>>(zc, score, pooled16 + (p0 / T) * C, T, C);
      if (M.has("f/confidence/paeNorm/scale")) {
        layerNorm(zc, xn, n, C, F("confidence/paeNorm/scale"), F("confidence/paeNorm/offset"));
        gemm(xn, F("confidence/pae"), paeL16 + p0 * pb, n, C, pb);
      } else gemm(zc, F("confidence/pae"), paeL16 + p0 * pb, n, C, pb);
    }
    CK(cudaStreamSynchronize(STREAM));
    if (kept) CK(cudaFree(resid));              // (a kept pair is the caller's, freed with z)
    else CK(cudaFree(stack));
    releaseScratch({ "cf.zc", "cf.xn", "cf.scoreRows", "cf.residRows" });
  }
  float* z = !zTrunk ? nullptr : inPlace ? const_cast<float*>(zTrunk) : dalloc(P * C);
  if (zTrunk) {
  if (C == 256) {
    confZNorm256K<<<(unsigned)((P + 7) / 8), 256, 0, STREAM>>>(
        zTrunk, z, F("confidence/zNorm/scale"), F("confidence/zNorm/offset"),
        relIdx(), BONDS, F("featuriser/tokenBonds"), r, c, 0, P, T, false);
  } else {
    layerNorm(zTrunk, z, P, C, F("confidence/zNorm/scale"), F("confidence/zNorm/offset"));
    confZK<<<blocks(P * C), 256, 0, STREAM>>>(z, relIdx(), BONDS, F("featuriser/tokenBonds"), r, c, T, C);
  }
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));
  float* prod = scratch<float>("cf.prod", chunk * C);
  for (size_t p0 = 0; p0 < P; p0 += chunk) {
    size_t n = std::min(chunk, P - p0);
    outerProductK<<<blocks((C & 3) == 0 ? n * (size_t)(C >> 2) : n * C), 256, 0, STREAM>>>(l, rr, prod, p0, n, T, C);
    gemm(prod, F("confidence/sToZProdOut"), z + p0 * C, n, C, C, 1.f);
  }
  const bool p16Conf = !tight && !check && ef2Pair16(T, C);
  __nv_bfloat16* p16 = p16Conf ? scratch<__nv_bfloat16>("ef2.p16", P * C) : nullptr;
  distanceEmbedK<<<blocks(P * 32), 256, 0, STREAM>>>(z, xDevice, Idev("distogram_atom_idx"), F("confidence/boundaries"),
    (int)M.len("f/confidence/boundaries"), F("confidence/distanceEmbedding"), T, C, p16);
  // z + trunk(z) - on a card short of room z waits in pinned host memory while the blocks run on it in place,
  // then comes back a chunk at a time and is added (the same sum: one more pair-sized tensor not on the card)
  float* mask = scratch<float>("trunk.mask", P);
  fillK<<<blocks(P), 256, 0, STREAM>>>(mask, 1.f, P);
  if (p16Conf) {
    PAIR16 = true;
    for (int b = 0; M.has("f/confidence/blocks/" + std::to_string(b) + "/pairTransition/transition1"); ++b)
      trunkBlock(reinterpret_cast<float*>(p16), mask, T, C, "confidence/blocks", b);
    PAIR16 = false;
    size_t n = P * C;
    ef2AddBf16K<<<blocks((n & 3) == 0 ? (n >> 2) : n), 256, 0, STREAM>>>(z, p16, n);
  } else {
    float* stack = tight ? z : scratch<float>("trunk.stack", P * C);
    static float* heldBuf = nullptr; static size_t heldHave = 0;     // (pinned, kept for the process, grown as needed)
    if (tight && heldHave < P * C) {
      if (heldBuf) CK(cudaFreeHost(heldBuf));
      if (cudaMallocHost(&heldBuf, P * C * 4) != cudaSuccess) {
        fprintf(stderr, "cannot pin %.2f GB of host memory for the confidence head's residual\n", P * C * 4 / 1e9); exit(1);
      }
      heldHave = P * C;
    }
    float* held = tight ? heldBuf : nullptr;
    if (tight) CK(cudaMemcpyAsync(held, z, P * C * 4, cudaMemcpyDeviceToHost, STREAM));
    else CK(cudaMemcpyAsync(stack, z, P * C * 4, cudaMemcpyDeviceToDevice, STREAM));
    for (int b = 0; M.has("f/confidence/blocks/" + std::to_string(b) + "/pairTransition/transition1"); ++b)
      trunkBlock(stack, mask, T, C, "confidence/blocks", b);
    if (tight) {
      releaseScratch({ "ftri.", "ftr.", "trib." });
      float* back = scratch<float>("cf.back", chunk * C);
      for (size_t p0 = 0; p0 < P; p0 += chunk) {
        size_t n = std::min(chunk, P - p0);
        CK(cudaMemcpyAsync(back, held + p0 * C, n * C * 4, cudaMemcpyHostToDevice, STREAM));
        addK<<<blocks(n * C), 256, 0, STREAM>>>(z + p0 * C, back, n * C);
      }
    } else {
      addK<<<blocks(P * C), 256, 0, STREAM>>>(z, stack, P * C);
    }
  }
  }
  // single by row-attention pooling
  float* pooled = pooled16 ? pooled16 : scratch<float>("cf.pooled", (size_t)T * C);
  float* single = scratch<float>("cf.single", (size_t)T * Cs);
  if (!pooled16) {
    float* score = scratch<float>("cf.score", P);
    gemm(z, F("confidence/poolingAttention"), score, P, C, 1);
    rowPoolK<<<T, 256, 0, STREAM>>>(z, score, pooled, T, C);
  }
  gemm(pooled, F("confidence/poolingOutput"), single, T, C, Cs);
  // pLDDT per atom (a table a slot: the atom's position within its token, clamped)
  int bins = (int)dimOf("f/confidence/plddtWeight", 2), slots = (int)dimOf("f/confidence/plddtWeight", 0);
  std::vector<int> a2t(A), slot(A);
  CK(cudaMemcpy(a2t.data(), Idev("atom_to_token"), A * 4, cudaMemcpyDeviceToHost));
  for (int a = 0, k = 0; a < A; ++a) { if (a > 0 && a2t[a] != a2t[a - 1]) k = 0; slot[a] = std::min(k++, slots - 1); }
  int* dSlot = scratch<int>("cf.slot", A);
  CK(cudaMemcpyAsync(dSlot, slot.data(), (size_t)A * 4, cudaMemcpyHostToDevice, STREAM));
  float* pa = scratch<float>("cf.pa", A);
  layerNorm(single, single, T, Cs, F("confidence/plddtNorm/scale"), F("confidence/plddtNorm/offset"));
  plddtAtomK<<<A, 64, 0, STREAM>>>(single, Idev("atom_to_token"), dSlot, F("confidence/plddtWeight"), pa, A, Cs, bins);
  // PAE logits
  float* paeL = paeL16 ? paeL16 : scratch<float>("cf.pae", P * pb);
  if (paeL16) {}
  else if (M.has("f/confidence/paeNorm/scale")) {
    // the released heads LayerNorm the pair in front of the PAE projection (pae_ln); a chunk of rows at a time
    size_t per = std::max<size_t>(1, std::min(P, ((size_t)64 << 20) / (4 * (size_t)C)));
    float* zn = scratch<float>("cf.paeIn", per * C);
    for (size_t p0 = 0; p0 < P; p0 += per) {
      size_t n = std::min(per, P - p0);
      layerNorm(z + p0 * C, zn, n, C, F("confidence/paeNorm/scale"), F("confidence/paeNorm/offset"));
      gemm(zn, F("confidence/pae"), paeL + p0 * pb, n, C, pb);
    }
  } else {
    gemm(z, F("confidence/pae"), paeL, P, C, pb);
  }
  Confidence out;
  out.plddtAtom = download(pa, A);
  std::vector<float> mask_ = download(W("atom_mask"), A);
  std::vector<int> asym(T); CK(cudaMemcpy(asym.data(), Idev("asym_id"), T * 4, cudaMemcpyDeviceToHost));
  out.plddtToken.assign(T, 0.f);
  std::vector<float> cnt(T, 0.f);
  double wsum = 0, wtot = 0;
  for (int a = 0; a < A; ++a) {
    out.plddtToken[a2t[a]] += out.plddtAtom[a] * mask_[a]; cnt[a2t[a]] += mask_[a];
    wsum += out.plddtAtom[a] * mask_[a]; wtot += mask_[a];
  }
  for (int t = 0; t < T; ++t) out.plddtToken[t] /= std::max(cnt[t], 1e-6f);
  out.meanPlddt = wsum / (wtot + 1e-8);
  // PAE and pTM / ipTM, on the device (the logits were 17 MB to download at 261 tokens and 4.4 M
  // exponentials on the host: 50 of the head's 62 ms)
  double width = 32.0 / pb, d0 = 1.24 * cbrt(std::max(T, 19) - 15.0) - 1.8;
  float* paeD = scratch<float>("cf.paeOut", P); float* tmD = scratch<float>("cf.tm", P);
  double* rowsD = scratch<double>("cf.tmRows", (size_t)2 * T);
  paeK<<<blocks(P), 256, 0, STREAM>>>(paeL, paeD, tmD, P, pb, width, d0);
  tmRowsK<<<T, 256, 0, STREAM>>>(tmD, Idev("asym_id"), rowsD, T);
  out.pae = download(paeD, P);
  {
    std::vector<int> chainOf; const int nc = (int)chainOrder(asym, chainOf).size();
    int* dChainOf = upload(chainOf.data(), T);
    double* dRows = scratch<double>("cf.tmChainRows", (size_t)T * nc);
    tmChainRowsK<<<T, 256, nc * sizeof(double), STREAM>>>(tmD, dChainOf, dRows, T, nc, 0);
    out.tmChainRows.resize((size_t)T * nc);
    CK(cudaMemcpy(out.tmChainRows.data(), dRows, out.tmChainRows.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaFree(dChainOf));
  }
  out.atomToken = a2t; out.atomMask = mask_;
  std::vector<double> rows(2 * T);
  CK(cudaMemcpy(rows.data(), rowsD, rows.size() * 8, cudaMemcpyDeviceToHost));
  double ptm = -1e30, iptm = -1e30;
  for (int i = 0; i < T; ++i) { ptm = std::max(ptm, rows[2 * i]); iptm = std::max(iptm, rows[2 * i + 1]); }
  out.ptm = ptm; out.iptm = iptm;
  if (check) {
    checkOracle("confidence pLDDT per atom", pa, A, "o/conf/plddt_per_atom");
    checkOracle("confidence PAE logits", paeL, P * pb, "o/conf/pae_logits");
    if (M.has("o/conf/ptm")) printf("  pTM %.6f against the oracle's %.6f\n", out.ptm, M.f("o/conf/ptm")[0]);
  }
  if (z && !inPlace) CK(cudaFree(z));
  return out;
}

// AlphaFold 3's confidence files beside the PDB, in cuda/af3's layout (its sampler.cuh) so one reader
// takes all three ports: <stem>_confidences.json - the expected PAE, token_plddts (the head's own pooling
// of its atoms), the token layout (chain letters from asym_id, residue numbers from residue_index) - and
// <stem>_summary_confidences.json (pTM, ipTM for a complex, mean pLDDT). Atom pLDDTs are the PDB's B factors.
inline void writeConfidences(const std::string& pdb, int T, const Confidence& conf, const std::vector<float>& contacts = {}) {
  std::string stem = pdb.size() > 4 && pdb.substr(pdb.size() - 4) == ".pdb" ? pdb.substr(0, pdb.size() - 4) : pdb;
  std::vector<int> asym(T), res(T);
  CK(cudaMemcpy(asym.data(), Idev("asym_id"), T * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(res.data(), Idev("residue_index"), T * 4, cudaMemcpyDeviceToHost));
  auto chainId = [](int a) {
    std::string id; for (a = a - 1; ; a = a / 26 - 1) { id.insert(id.begin(), (char)('A' + a % 26)); if (a < 26) break; }
    return id;
  };
  bool chains = false; for (int t = 1; t < T; ++t) chains |= asym[t] != asym[0];
  int first = *std::min_element(asym.begin(), asym.end());     // (EF2 counts chains and residues from 0;
  FILE* f = fopen((stem + "_confidences.json").c_str(), "w");    //  the PDB from A and 1)
  {
    std::string j = "{\"pae\": ";
    appendMatrix2(j, conf.pae.data(), T);
    if (!contacts.empty()) { j += ",\n \"contact_probs\": "; appendMatrix2(j, contacts.data(), T); }
    fwrite(j.data(), 1, j.size(), f);
  }
  fprintf(f, ",\n \"token_plddts\": [");
  for (int i = 0; i < T; ++i) fprintf(f, "%s%.2f", i ? ", " : "", 100.f * conf.plddtToken[i]);
  fprintf(f, "],\n \"token_chain_ids\": [");
  for (int i = 0; i < T; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", chainId(asym[i] - first + 1).c_str());
  fprintf(f, "],\n \"token_res_ids\": [");
  for (int i = 0; i < T; ++i) fprintf(f, "%s%d", i ? ", " : "", res[i] + 1);
  fprintf(f, "]}\n");
  fclose(f);
  // 🔴 THE CHAIN-LEVEL FIELDS, AS cuda/af3's writeConfidences DEFINES THEM (sampler.cuh) - every chain's pTM and ipTM and
  // every pair's ipTM a max over anchors of the mean TM term over the tokens it selects, the pair's PAE minimum over
  // ordered pairs, the chain's mean atom pLDDT. They were computed (the TM matrix is the pTM's) and reduced to two
  // global numbers, so a binder's own pTM and its interface ipTM were not in the file (2026-10-30)
  std::vector<int> chainOf;
  const int nc = (int)chainOrder(asym, chainOf).size();
  std::vector<int> chainCount(nc, 0);
  for (int i = 0; i < T; ++i) chainCount[chainOf[i]]++;
  if (conf.tmChainRows.size() != (size_t)T * nc) throw std::runtime_error("the confidence head returned no per-chain TM sums");
  const std::vector<double>& rowSum = conf.tmChainRows;
  auto best = [&](int from, bool others, int to, bool notTo) {
    double top = -1e30; bool any = false;
    for (int i = 0; i < T; ++i) {
      if (others ? chainOf[i] == from : chainOf[i] != from) continue;
      const double* r = rowSum.data() + (size_t)i * nc;
      double total = 0; int count = 0;
      if (notTo) { for (int b = 0; b < nc; ++b) if (b != to) { total += r[b]; count += chainCount[b]; } }
      else { total = r[to]; count = chainCount[to]; }
      if (count) { any = true; top = std::max(top, total / count); }
    }
    return any ? top : NAN;
  };
  auto maxNan = [](double x, double y) { return std::isnan(x) ? y : std::isnan(y) ? x : std::max(x, y); };
  auto num = [](FILE* g, double v) { if (std::isfinite(v)) fprintf(g, "%.4f", v); else fprintf(g, "null"); };
  auto list = [&](FILE* g, const char* key, const std::vector<double>& v) {
    fprintf(g, "\"%s\": [", key);
    for (int a = 0; a < (int)v.size(); ++a) { if (a) fprintf(g, ", "); num(g, v[a]); }
    fprintf(g, "], ");
  };
  f = fopen((stem + "_summary_confidences.json").c_str(), "w");
  fprintf(f, "{");
  {
    std::vector<double> chainPtm(nc), chainIptm(nc);
    for (int a = 0; a < nc; ++a) {
      chainPtm[a] = best(a, false, a, false);
      chainIptm[a] = maxNan(best(a, false, a, true), best(a, true, a, false));
    }
    list(f, "chain_ptm", chainPtm);
    if (nc > 1) list(f, "chain_iptm", chainIptm);
    fprintf(f, "\"chain_pair_iptm\": [");
    for (int a = 0; a < nc; ++a) {
      fprintf(f, "%s[", a ? ", " : "");
      for (int b = 0; b < nc; ++b) {
        if (b) fprintf(f, ", ");
        num(f, a == b ? chainPtm[a] : maxNan(best(a, false, b, false), best(b, false, a, false)));
      }
      fprintf(f, "]");
    }
    fprintf(f, "], ");
  }
  {
    std::vector<double> mn((size_t)nc * nc, INFINITY);
    for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j) {
      double& m = mn[(size_t)chainOf[i] * nc + chainOf[j]];
      m = std::min(m, (double)conf.pae[(size_t)i * T + j]);
    }
    fprintf(f, "\"chain_pair_pae_min\": [");
    for (int a = 0; a < nc; ++a) {
      fprintf(f, "%s[", a ? ", " : "");
      for (int b = 0; b < nc; ++b) { if (b) fprintf(f, ", "); num(f, mn[(size_t)a * nc + b]); }
      fprintf(f, "]");
    }
    fprintf(f, "], ");
  }
  if (conf.atomToken.size() == conf.plddtAtom.size()) {
    std::vector<double> sum(nc, 0), cnt(nc, 0), plddt(nc, NAN);
    for (size_t a = 0; a < conf.atomToken.size(); ++a) {
      int c = chainOf[conf.atomToken[a]];
      sum[c] += 100.0 * conf.plddtAtom[a] * conf.atomMask[a]; cnt[c] += conf.atomMask[a];
    }
    for (int c = 0; c < nc; ++c) if (cnt[c] > 0) plddt[c] = sum[c] / cnt[c];
    list(f, "chain_plddt", plddt);
  }
  if (chains) fprintf(f, "\"ptm\": %.4f, \"iptm\": %.4f, \"mean_plddt\": %.2f}\n", conf.ptm, conf.iptm, 100 * conf.meanPlddt);
  else fprintf(f, "\"ptm\": %.4f, \"iptm\": null, \"mean_plddt\": %.2f}\n", conf.ptm, 100 * conf.meanPlddt);
  fclose(f);
}
