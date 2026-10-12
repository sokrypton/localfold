#pragma once
// ESMFold2's trunk across several GPUs (--gpus=N): cuda/af3's ranks and shared buffers (multigpu.cuh) and its sharded
// triangle (sharded.cuh's triangleOn, ESMFold2's weight names through sh::WNAME). A trunk block is two triangle
// multiplications and a transition - no attention - so each rank holds its rows of the pair (bf16, as EF2_Z16 holds it
// whole), the recycle and the streamed z_init are row-local, the transition runs on the rank's own rows, and only the
// triangles exchange. After the last pass the pair goes onto rank 0, which folds on alone as one GPU does (the others
// leave): the trunk is the fold's time and, on one card, its memory.
#include "trunk.cuh"
#include "../../af3/src/multigpu.cuh"
#include "../../af3/src/sharded.cuh"

// this rank's rows of the trunk's pair (the shared buffer the trunk leaves them in)
inline mg::Shared* EF2_ZS = nullptr;
struct B16Host { void* p = nullptr; size_t have = 0, bytes = 0; };
inline std::string ef2WeightName(const std::string& pre, const char* leaf) { return "f/" + pre + "/" + leaf; }

// the trunk on this rank's rows: returns them bf16, indexed by the global pair (EF2_ZS the buffer)
inline float* shardedTrunkEf2(int T, int C, int loops) {
  using B16 = __nv_bfloat16;
  sh::WNAME = ef2WeightName;
  const size_t P = (size_t)T * T;
  int lo, hi; sh::rowsOf(T, mg::RANK, lo, hi);
  const int rows = sh::storedRows(T, mg::RANK);
  const size_t held = (size_t)rows * T;                          // this rank's pairs, from pair lo * T
  mg::Shared& zS = mg::shared("sh.z", std::max<size_t>((size_t)sh::maxStored(T) * T * C * 2, 16));
  B16* zl = (B16*)zS.local;
  B16* base = zl - (size_t)lo * T * C;                           // (indexed by the global pair)
  CK(cudaMemsetAsync(zl, 0, std::max<size_t>(held * C * 2, 1), STREAM));
  const float* mask = nullptr;                                   // (every pair real: the triangle reads null as ones)
  const size_t chunk = std::min<size_t>(std::max<size_t>(held, 1), ((size_t)64 << 20) / (4 * (size_t)C));
  float* xn = scratch<float>("trunk.xn", chunk * C); float* zc = scratch<float>("trunk.zc", chunk * C);
  const ZInitStream& zs = ZINIT_STREAM;
  const int bi = lmPairBlock(T, zs.P);
  float* lm = scratch<float>("zi.lmRows", (size_t)bi * T * zs.P);
  const int blocksN = (int)M.meta("meta/blocks");
  for (int loop = 0; loop < loops; ++loop) {
    // the recycle, f32 a chunk of this rank's rows at a time
    for (size_t r0 = 0; r0 < held; r0 += chunk) {
      size_t r = std::min(chunk, held - r0);
      ef2FromBf16K<<<blocks(r * C), 256, 0, STREAM>>>(zl + r0 * C, zc, r * C);
      layerNorm(zc, xn, r, C, F("recycle/norm/scale"), F("recycle/norm/offset"));
      gemm(xn, F("recycle/projection"), zc, r, C, C);
      ef2ToBf16K<<<blocks(r * C), 256, 0, STREAM>>>(zc, zl + r0 * C, r * C);
    }
    // z_init on this rank's token rows (zInitAddStreamed's, the language model's pair made a block of rows at a time)
    for (int i0 = lo; i0 < lo + rows; i0 += bi) {
      int b = std::min(bi, lo + rows - i0); size_t cells = (size_t)b * T;
      lmPairRows(zs.s, T, zs.P, i0, b, lm);
      zInitAddRowsK<<<blocks(cells * C), 256, 0, STREAM>>>(base, zs.rows, zs.cols, relIdx(), BONDS,
                                                           F("featuriser/tokenBonds"), lm, T, C, (size_t)i0 * T, cells);
    }
    PAIR16 = true;
    for (int b = 0; b < blocksN; ++b) {
      const std::string B = "blocks/" + std::to_string(b) + "/";
      sh::triangleSharded<half>(zS, mask, T, C, B + "triangleMultiplicationOutgoing", false, false);
      sh::triangleSharded<half>(zS, mask, T, C, B + "triangleMultiplicationIncoming", false, true);
      if (held) transition256(reinterpret_cast<float*>(zl), held, C, B + "pairTransition/");
      mg::fence();                                   // (the next block's triangle reads every rank's rows)
    }
    PAIR16 = false;
    mg::tick("trunk pass");
  }
  // (the triangle's exchange buffers given back: its product share alone is a share - 12.5 GB a rank at 9,880 tokens on
  // 4, which the contacts' transposed share then could not find)
  mg::release({ "sh.tp", "sh.tsa", "sh.tsb", "sh.bias", "sh.bg", "sh.zTc" }, mg::FREE);
  releaseScratch(); trimPool();
  EF2_ZS = &zS;
  return reinterpret_cast<float*>(base);
}

// rank 0's parked pair: every rank's rows copied into its pinned host memory (PARK_HOST, as EF2_Z16 parks the whole
// pair - the confidence head reads it from there), every shared buffer given back, the others gone
inline void pairOntoRank0HostEf2(int T, int C) {
  const size_t row = (size_t)T * C * 2, whole = (size_t)T * row;
  releaseScratch();
  if (mg::RANK == 0 && PARK_HAVE < whole) {
    if (PARK_HOST) CK(cudaFreeHost(PARK_HOST));
    if (cudaMallocHost(&PARK_HOST, whole) != cudaSuccess) {
      fprintf(stderr, "cannot pin %.2f GB of host memory for the gathered pair: this input needs more host RAM\n", whole / 1e9); exit(1);
    }
    PARK_HAVE = whole;
  }
  mg::fence();
  if (mg::RANK == 0)
    for (int r = 0; r < mg::WORLD; ++r) {
      int a, e; sh::rowsOf(T, r, a, e); const int rn = sh::storedRows(T, r);
      if (rn) CK(cudaMemcpyAsync((char*)PARK_HOST + (size_t)a * row, EF2_ZS->peer[r], (size_t)rn * row, cudaMemcpyDefault, STREAM));
    }
  CK(cudaStreamSynchronize(STREAM));
  mg::fence();
  mg::release({ "" }, mg::FINAL);
  if (mg::RANK != 0) mg::leave();
  mg::finish();
  mg::WORLD = 1; SHORT_PAIR_SHARE = 1; EF2_QLO = EF2_QROWS = 0; EF2_ZS = nullptr;
}

// this rank's share in its own pinned host buffer through the sampler (which never reads it), and back - or kept on the
// device where every rank has the room for it to stay through the sampler and the confidence head beside it: the head's
// residual (a share's size) and the sharded triangle's whole form (its own estimate) past an eighth of the card. Parking was
// pinning alone: 1.1 s for a 1.58 GB share at 4,940 tokens on 8, where a rank holds under 7 GB
inline B16Host EF2_SHARE_HOST{};
// (returns whether it was kept: unparkShareEf2 and confidenceHeadSharded take it)
inline bool parkShareEf2(int T, int C) {
  const size_t bytes = (size_t)sh::storedRows(T, mg::RANK) * T * C * 2;
  size_t f = 0, t = 0; CK(cudaMemGetInfo(&f, &t));
  if (!BIG_FORCED && mg::everyRankHas(bytes + sh::wholeFormBytes(T, C, 2) + t / 8)) return true;
  if (EF2_SHARE_HOST.have < bytes) {
    if (EF2_SHARE_HOST.p) CK(cudaFreeHost(EF2_SHARE_HOST.p));
    if (cudaMallocHost(&EF2_SHARE_HOST.p, std::max<size_t>(bytes, 1)) != cudaSuccess) {
      fprintf(stderr, "cannot pin %.2f GB of host memory to park this GPU's share of the pair\n", bytes / 1e9); exit(1);
    }
    EF2_SHARE_HOST.have = bytes;
  }
  EF2_SHARE_HOST.bytes = EF2_ZS->bytes;
  if (bytes) CK(cudaMemcpyAsync(EF2_SHARE_HOST.p, EF2_ZS->local, bytes, cudaMemcpyDeviceToHost, STREAM));
  mg::release({ "sh.z" }, mg::FREE);
  EF2_ZS = nullptr;
  return false;
}
inline void unparkShareEf2(int T, int C, bool kept) {
  releaseScratch(); trimPool();
  if (kept) return;
  EF2_ZS = &mg::shared("sh.z", EF2_SHARE_HOST.bytes);
  const size_t bytes = (size_t)sh::storedRows(T, mg::RANK) * T * C * 2;
  if (bytes) CK(cudaMemcpyAsync(EF2_ZS->local, EF2_SHARE_HOST.p, bytes, cudaMemcpyHostToDevice, STREAM));
}

// pair positions [p0, p0 + n) of z + z^T off this rank's rows of z and of z^T (both bf16, indexed by the global pair)
__global__ void symmetriseShareK(const __nv_bfloat16* z, const __nv_bfloat16* zT, float* out, size_t p0, size_t n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * C) return;
  out[t] = __bfloat162float(z[p0 * C + t]) + __bfloat162float(zT[p0 * C + t]);
}
// the contact probabilities of every pair (rank 0's; the others return nothing): each rank's rows off its rows of z and
// of z^T (a transposed share made for this and given back), the rows all-gathered
template <class ContactsK>
inline std::vector<float> shardedContactsEf2(int T, int C, int bins, ContactsK contacts) {
  using B16 = __nv_bfloat16;
  const size_t P = (size_t)T * T;
  int lo, hi; sh::rowsOf(T, mg::RANK, lo, hi);
  const size_t p0 = (size_t)lo * T;
  mg::Shared& zTS = mg::shared("sh.zT", EF2_ZS->bytes);
  PAIR16 = true; sh::transpose(*EF2_ZS, zTS, T, C); PAIR16 = false;
  const B16* zb = (const B16*)EF2_ZS->local - p0 * C; const B16* zTb = (const B16*)zTS.local - p0 * C;
  // (rows padded to 4 floats in the gathered buffer: allGather moves 16-byte pieces)
  const int rowsHere = sh::storedRows(T, mg::RANK), Tp = (T + 3) / 4 * 4;
  mg::Shared& probsS = mg::shared("ef2.contacts", (size_t)T * Tp * 4);
  float* probs = (float*)probsS.local;
  const int Bn = (int)dimOf("f/distogram/weights", 1);
  const int rb = (int)std::max<size_t>(1, ((size_t)64 << 20) / ((size_t)T * std::max(C, Bn)));    // token rows a chunk
  float* zs = scratch<float>("dg.symRows", (size_t)rb * T * C); float* dg = scratch<float>("dg.logitRows", (size_t)rb * T * bins);
  float* pr = scratch<float>("dg.probRows", (size_t)rb * T);
  for (int i0 = lo; i0 < lo + rowsHere; i0 += rb) {
    const int b = std::min(rb, lo + rowsHere - i0); const size_t n = (size_t)b * T, q0 = (size_t)i0 * T;
    symmetriseShareK<<<blocks(n * C), 256, 0, STREAM>>>(zb, zTb, zs, q0, n, C);
    gemm(zs, F("distogram/weights"), dg, n, C, Bn);
    addBias(dg, F("distogram/bias"), n, Bn);
    contacts(dg, q0, n, pr);
    CK(cudaMemcpy2DAsync(probs + (size_t)i0 * Tp, (size_t)Tp * 4, pr, (size_t)T * 4, (size_t)T * 4, b, cudaMemcpyDeviceToDevice, STREAM));
  }
  mg::allGather(probsS, (size_t)lo * Tp * 4, (size_t)rowsHere * Tp * 4);
  std::vector<float> out;
  if (mg::RANK == 0) {
    out.resize(P);
    CK(cudaMemcpy2D(out.data(), (size_t)T * 4, probs, (size_t)Tp * 4, (size_t)T * 4, T, cudaMemcpyDeviceToHost));
  }
  mg::release({ "sh.zT", "ef2.contacts" }, mg::FREE);
  releaseScratch({ "dg." });
  return out;
}

// tmRowsK's, rows [i0, i0 + gridDim.x) of a chunk tm holding those rows alone
__global__ void tmRowsAtK(const float* tm, const int* asym, double* rows, int T, int i0) {
  const int il = blockIdx.x, i = i0 + il;
  __shared__ double s[2][256], c[256];
  double sum = 0, isum = 0, icnt = 0;
  for (int j = threadIdx.x; j < T; j += blockDim.x) {
    double v = tm[(size_t)il * T + j]; sum += v;
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

// The confidence head across the ranks (confidence.cuh's, its EF2_Z16 form sharded): each rank builds the head's pair
// on its rows of the trunk's - in place, a chunk of token rows at a time, the result kept as the residual (bf16, where the
// trunk's share waited through the sampler) - the head's blocks run as the trunk's do (the sharded triangle, the transition on own rows), and the
// pooling, the PAE and the pTM rows come off each rank's rows and are all-gathered. Every rank returns the same
// Confidence; rank 0's is written.
// shareKept: the trunk's share stayed on the device through the sampler (parkShareEf2's answer)
inline Confidence confidenceHeadSharded(int T, int A, const float* sInputs, int Si, const float* xDevice, bool shareKept) {
  using B16 = __nv_bfloat16;
  const int C = (int)dimOf("f/confidence/sToZ", 1), Cs = (int)dimOf("f/confidence/poolingOutput", 1);
  const size_t P = (size_t)T * T;
  int lo, hi; sh::rowsOf(T, mg::RANK, lo, hi);
  const int rows = sh::storedRows(T, mg::RANK);
  const size_t held = (size_t)rows * T;
  B16* zl = (B16*)EF2_ZS->local;
  float* s = scratch<float>("cf.s", (size_t)T * Si);
  layerNorm(sInputs, s, T, Si, F("confidence/sInputsNorm/scale"), F("confidence/sInputsNorm/offset"));
  float* r = scratch<float>("cf.r", (size_t)T * C); float* c = scratch<float>("cf.c", (size_t)T * C);
  float* l = scratch<float>("cf.l", (size_t)T * C); float* rr = scratch<float>("cf.rr", (size_t)T * C);
  gemm(s, F("confidence/sToZ"), r, T, Si, C);
  gemm(s, F("confidence/sToZTranspose"), c, T, Si, C);
  gemm(s, F("confidence/sToZProdIn1"), l, T, Si, C);
  gemm(s, F("confidence/sToZProdIn2"), rr, T, Si, C);
  const int rb = (int)std::max<size_t>(1, ((size_t)64 << 20) / ((size_t)T * C * 4));   // token rows a chunk
  const size_t per = (size_t)rb * T;
  // the residual: the head's pair rows as its blocks take them (bf16) - on the device where the trunk's share stayed
  // through the sampler, else in the share's pinned host buffer, free again once the share is back (an f32 copy pinned
  // of its own was 2.1 s of a 3.3 s head at 4,940 tokens on 8, and 35 GB of host memory a rank at 16,549)
  B16* resid = shareKept ? dallocT<B16>(std::max<size_t>(held * C, 1)) : (B16*)EF2_SHARE_HOST.p;
  {
    float* zc = scratch<float>("cf.zc", per * C); float* xn = scratch<float>("cf.xn", per * C);
    float* prodc = scratch<float>("cf.prodc", per * C);
    for (int i0 = lo; i0 < lo + rows; i0 += rb) {
      const int b = std::min(rb, lo + rows - i0); const size_t n = (size_t)b * T, q0 = (size_t)i0 * T, ql = (size_t)(i0 - lo) * T;
      ef2FromBf16K<<<blocks(n * C), 256, 0, STREAM>>>(zl + ql * C, xn, n * C);
      layerNorm(xn, zc, n, C, F("confidence/zNorm/scale"), F("confidence/zNorm/offset"));
      confZRowsK<<<blocks(n * C), 256, 0, STREAM>>>(zc, relIdx(), BONDS, F("featuriser/tokenBonds"), r, c, q0, n, T, C);
      outerProductK<<<blocks(n * C), 256, 0, STREAM>>>(l, rr, prodc, q0, n, T, C);
      gemm(prodc, F("confidence/sToZProdOut"), zc, n, C, C, 1.f);
      distanceEmbedRowsK<<<blocks(n * 32), 256, 0, STREAM>>>(zc, xDevice, Idev("distogram_atom_idx"), F("confidence/boundaries"),
        (int)M.len("f/confidence/boundaries"), F("confidence/distanceEmbedding"), q0, n, T, C);
      ef2ToBf16K<<<blocks(n * C), 256, 0, STREAM>>>(zc, zl + ql * C, n * C);
    }
    releaseScratch({ "cf.zc", "cf.xn", "cf.prodc" });
    if (held) CK(cudaMemcpyAsync(resid, zl, held * C * 2, cudaMemcpyDefault, STREAM));
  }
  const float* mask = nullptr;                                   // (every pair real)
  PAIR16 = true;
  for (int b = 0; M.has("f/confidence/blocks/" + std::to_string(b) + "/pairTransition/transition1"); ++b) {
    const std::string B = "confidence/blocks/" + std::to_string(b) + "/";
    mg::fence();                                       // (every rank's rows of the head's pair written)
    sh::triangleSharded<half>(*EF2_ZS, mask, T, C, B + "triangleMultiplicationOutgoing", false, false);
    sh::triangleSharded<half>(*EF2_ZS, mask, T, C, B + "triangleMultiplicationIncoming", false, true);
    if (held) transition256(reinterpret_cast<float*>(zl), held, C, B + "pairTransition/");
  }
  PAIR16 = false;
  releaseScratch({ "ftri.", "ftr.", "trib." });
  // the head's outputs off this rank's rows: the pooling's rows, the PAE and pTM rows - all-gathered
  const int pb = (int)dimOf("f/confidence/pae", 1), Tp = (T + 3) / 4 * 4;
  mg::Shared& pooledS = mg::shared("ef2.cf.pooled", (size_t)T * C * 4);
  mg::Shared& tmS = mg::shared("ef2.cf.tmrows", (size_t)T * 2 * 8);
  mg::Shared& paeS = mg::shared("ef2.cf.pae", (size_t)T * Tp * 4);
  float* pooled = (float*)pooledS.local; double* rowsD = (double*)tmS.local; float* paeAll = (float*)paeS.local;
  const double width = 32.0 / pb, d0 = 1.24 * cbrt(std::max(T, 19) - 15.0) - 1.8;
  {
    float* zc = scratch<float>("cf.zc", per * C); float* xn = scratch<float>("cf.xn", per * C);
    float* score = scratch<float>("cf.scoreRows", per); float* paeL = scratch<float>("cf.paeRows", per * pb);
    float* paeD = scratch<float>("cf.paeOutRows", per); float* tmD = scratch<float>("cf.tmRows1", per);
    B16* rh = shareKept ? nullptr : scratch<B16>("cf.residRows", per * C);
    for (int i0 = lo; i0 < lo + rows; i0 += rb) {
      const int b = std::min(rb, lo + rows - i0); const size_t n = (size_t)b * T, ql = (size_t)(i0 - lo) * T;
      const B16* rs = resid + ql * C;
      if (rh) { CK(cudaMemcpyAsync(rh, rs, n * C * 2, cudaMemcpyHostToDevice, STREAM)); rs = rh; }
      ef2FromBf16K<<<blocks(n * C), 256, 0, STREAM>>>(rs, xn, n * C);
      ef2FromBf16K<<<blocks(n * C), 256, 0, STREAM>>>(zl + ql * C, zc, n * C);
      addK<<<blocks(n * C), 256, 0, STREAM>>>(zc, xn, n * C);
      gemm(zc, F("confidence/poolingAttention"), score, n, C, 1);
      rowPoolK<<<(unsigned)b, 256, 0, STREAM>>>(zc, score, pooled + (size_t)i0 * C, T, C);
      if (M.has("f/confidence/paeNorm/scale")) {
        layerNorm(zc, xn, n, C, F("confidence/paeNorm/scale"), F("confidence/paeNorm/offset"));
        gemm(xn, F("confidence/pae"), paeL, n, C, pb);
      } else gemm(zc, F("confidence/pae"), paeL, n, C, pb);
      paeK<<<blocks(n), 256, 0, STREAM>>>(paeL, paeD, tmD, n, pb, width, d0);
      tmRowsAtK<<<(unsigned)b, 256, 0, STREAM>>>(tmD, Idev("asym_id"), rowsD, T, i0);
      CK(cudaMemcpy2DAsync(paeAll + (size_t)i0 * Tp, (size_t)Tp * 4, paeD, (size_t)T * 4, (size_t)T * 4, b, cudaMemcpyDeviceToDevice, STREAM));
    }
    releaseScratch({ "cf.zc", "cf.xn", "cf.scoreRows", "cf.paeRows", "cf.paeOutRows", "cf.tmRows1", "cf.residRows" });
    if (shareKept) CK(cudaFree(resid));
  }
  mg::allGather(pooledS, (size_t)lo * C * 4, (size_t)rows * C * 4);
  mg::allGather(tmS, (size_t)lo * 16, (size_t)rows * 16);
  mg::allGather(paeS, (size_t)lo * Tp * 4, (size_t)rows * Tp * 4);
  // the single and the per-atom pLDDT (confidenceHead's, on the gathered pooling) - every rank alike
  float* single = scratch<float>("cf.single", (size_t)T * Cs);
  gemm(pooled, F("confidence/poolingOutput"), single, T, C, Cs);
  int bins = (int)dimOf("f/confidence/plddtWeight", 2), slots = (int)dimOf("f/confidence/plddtWeight", 0);
  std::vector<int> a2t(A), slot(A);
  CK(cudaMemcpy(a2t.data(), Idev("atom_to_token"), A * 4, cudaMemcpyDeviceToHost));
  for (int a = 0, k = 0; a < A; ++a) { if (a > 0 && a2t[a] != a2t[a - 1]) k = 0; slot[a] = std::min(k++, slots - 1); }
  int* dSlot = upload(slot.data(), A);
  float* sa = scratch<float>("cf.sa", (size_t)A * Cs); float* pa = scratch<float>("cf.pa", A);
  gatherAtomsK<<<blocks((size_t)A * Cs), 256, 0, STREAM>>>(single, Idev("atom_to_token"), sa, A, Cs);
  layerNorm(sa, sa, A, Cs, F("confidence/plddtNorm/scale"), F("confidence/plddtNorm/offset"));
  plddtAtomK<<<A, 64, 0, STREAM>>>(sa, dSlot, F("confidence/plddtWeight"), pa, A, Cs, bins);
  Confidence out;
  out.plddtAtom = download(pa, A);
  std::vector<float> mask_ = download(W("atom_mask"), A);
  out.plddtToken.assign(T, 0.f);
  std::vector<float> cnt(T, 0.f);
  double wsum = 0, wtot = 0;
  for (int a = 0; a < A; ++a) {
    out.plddtToken[a2t[a]] += out.plddtAtom[a] * mask_[a]; cnt[a2t[a]] += mask_[a];
    wsum += out.plddtAtom[a] * mask_[a]; wtot += mask_[a];
  }
  for (int t = 0; t < T; ++t) out.plddtToken[t] /= std::max(cnt[t], 1e-6f);
  out.meanPlddt = wsum / (wtot + 1e-8);
  out.pae.resize(P);
  CK(cudaMemcpy2D(out.pae.data(), (size_t)T * 4, paeAll, (size_t)Tp * 4, (size_t)T * 4, T, cudaMemcpyDeviceToHost));
  std::vector<double> rowsH(2 * (size_t)T);
  CK(cudaMemcpy(rowsH.data(), rowsD, rowsH.size() * 8, cudaMemcpyDeviceToHost));
  double ptm = -1e30, iptm = -1e30;
  for (int i = 0; i < T; ++i) { ptm = std::max(ptm, rowsH[2 * i]); iptm = std::max(iptm, rowsH[2 * i + 1]); }
  out.ptm = ptm; out.iptm = iptm;
  CK(cudaFree(dSlot));
  return out;
}
