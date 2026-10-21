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
__global__ void outerProductK(const float* a, const float* b, float* out, size_t p0, size_t n, int T, int C) {   // a_i * b_j, rows p0..
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * C) return;
  int c = (int)(t % C); size_t ij = p0 + t / C; int i = (int)(ij / T), j = (int)(ij % T);
  out[t] = a[(size_t)i * C + c] * b[(size_t)j * C + c];
}
// a warp a pair: its distance and bucket once (a ballot over the edges), then the row 16 bytes a lane - an
// element a thread took the distance and walked every edge for each of the C channels (12 ms of a 1,044-token
// confidence head's 194; C a multiple of 4)
__global__ void distanceEmbedK(float* z, const float* x, const int* rep, const float* edges, int nEdges,
                               const float* table, int T, int C) {
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
  for (int c = lane; c < C / 4; c += 32) { float4 v = row[c], w = tb[c]; v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w; row[c] = v; }
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
__global__ void plddtAtomK(const float* __restrict__ s, const int* __restrict__ slot, const float* __restrict__ table,
                           float* __restrict__ plddt, int A, int C, int bins) {
  int a = blockIdx.x, tid = threadIdx.x;
  __shared__ float shS[512];
  __shared__ float logit[64];
  __shared__ float mxSh;
  const float* sa = s + (size_t)a * C;
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

struct Confidence { std::vector<float> plddtAtom, plddtToken, pae; double ptm, iptm, meanPlddt; };

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
  bool tight = shortPair(P, C) && !roomFor(2 * P * C * 4), inPlace = consume && tight;
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
      ef2FromBf16K<<<blocks(n * C), 256, 0, STREAM>>>(src, xn, n * C);
      layerNorm(xn, zc, n, C, F("confidence/zNorm/scale"), F("confidence/zNorm/offset"));
      confZRowsK<<<blocks(n * C), 256, 0, STREAM>>>(zc, relIdx(), BONDS, F("featuriser/tokenBonds"), r, c, p0, n, T, C);
      outerProductK<<<blocks(n * C), 256, 0, STREAM>>>(l, rr, prodc, p0, n, T, C);
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
  layerNorm(zTrunk, z, P, C, F("confidence/zNorm/scale"), F("confidence/zNorm/offset"));
  confZK<<<blocks(P * C), 256, 0, STREAM>>>(z, relIdx(), BONDS, F("featuriser/tokenBonds"), r, c, T, C);
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));
  float* prod = scratch<float>("cf.prod", chunk * C);
  for (size_t p0 = 0; p0 < P; p0 += chunk) {
    size_t n = std::min(chunk, P - p0);
    outerProductK<<<blocks(n * C), 256, 0, STREAM>>>(l, rr, prod, p0, n, T, C);
    gemm(prod, F("confidence/sToZProdOut"), z + p0 * C, n, C, C, 1.f);
  }
  distanceEmbedK<<<blocks(P * 32), 256, 0, STREAM>>>(z, xDevice, Idev("distogram_atom_idx"), F("confidence/boundaries"),
    (int)M.len("f/confidence/boundaries"), F("confidence/distanceEmbedding"), T, C);
  // z + trunk(z) - on a card short of room z waits in pinned host memory while the blocks run on it in place,
  // then comes back a chunk at a time and is added (the same sum: one more pair-sized tensor not on the card)
  float* mask = scratch<float>("trunk.mask", P);
  fillK<<<blocks(P), 256, 0, STREAM>>>(mask, 1.f, P);
  float* stack = tight ? z : dalloc(P * C);
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
    CK(cudaFree(stack));
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
  int* dSlot = upload(slot.data(), A);
  float* sa = scratch<float>("cf.sa", (size_t)A * Cs); float* pa = scratch<float>("cf.pa", A);
  gatherAtomsK<<<blocks((size_t)A * Cs), 256, 0, STREAM>>>(single, Idev("atom_to_token"), sa, A, Cs);
  layerNorm(sa, sa, A, Cs, F("confidence/plddtNorm/scale"), F("confidence/plddtNorm/offset"));
  plddtAtomK<<<A, 64, 0, STREAM>>>(sa, dSlot, F("confidence/plddtWeight"), pa, A, Cs, bins);
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
  CK(cudaFree(dSlot));
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
  f = fopen((stem + "_summary_confidences.json").c_str(), "w");
  if (chains) fprintf(f, "{\"ptm\": %.4f, \"iptm\": %.4f, \"mean_plddt\": %.2f}\n", conf.ptm, conf.iptm, 100 * conf.meanPlddt);
  else fprintf(f, "{\"ptm\": %.4f, \"iptm\": null, \"mean_plddt\": %.2f}\n", conf.ptm, 100 * conf.meanPlddt);
  fclose(f);
}
