#pragma once
// One fold across several GPUs, phase 2: the pair updates on a SHARDED pair - rank r holds pair rows [lo_r, hi_r) only
// (its slab, n_r x n x C), shares of the 16-padded length so every slab boundary suits the FP8 GEMM's leading
// dimensions. A kernel that touches only its own rows runs unchanged on a slab through a shifted base pointer
// (base = slab - lo * n * C): they index the pair by global row.
//   triangle outgoing - each rank builds the rows of b it owns, all-gathered into the whole b the blocked form keeps
//     on one GPU too, then this rank's blocks of a against it (triangleBlockedTN's outgoing branch on a slab);
//   triangle incoming - the same on z^T: p[i][j] = sum_k a(z[k][j]) b(z[k][i]) = sum_k a(zT[j][k]) b(zT[i][k]) is the
//     outgoing form on z^T with the incoming weights, its output at zT[j][i] = z[i][j] (the gate read there too);
//   row attention - the bias (every row's, every rank needs it) from this rank's rows then all-gathered; the rows in
//     chunks; column attention is row attention on z^T with the bias flipped (swap -> !swap);
//   the transition - local.
// z^T comes from an all-to-all transpose (transpose()), four a block. multigpu.cuh has the transport.
#include "pairtrack.cuh"

namespace sh {
// (the default with several GPUs; LOCALFOLD_MG_SHARDED=0 keeps phase 1, the pair replicated and its updates' work split)
inline bool ON = !getenv("LOCALFOLD_MG_SHARDED") || atoi(getenv("LOCALFOLD_MG_SHARDED"));
// the diffusion's pair rows on this rank - the trunk's slab carried into it (DLO -1: the whole pair, one GPU)
inline int DLO = -1, DROWS = 0;
inline int padded(int n) { return (n + 15) / 16 * 16; }
// rank r's rows of the pair: its share of the padded length, its stored (real) rows clipped to n
inline void rowsOf(int n, int r, int& lo, int& hi) { mg::shareOf(padded(n), 16, r, lo, hi); }
inline int storedRows(int n, int r) { int lo, hi; rowsOf(n, r, lo, hi); return std::max(0, std::min(hi, n) - lo); }
inline int maxStored(int n) { int m = 0; for (int r = 0; r < mg::WORLD; ++r) m = std::max(m, storedRows(n, r)); return m; }
inline size_t elem() { return PAIR16 ? 2 : 4; }
inline float* baseOf(void* slab, int lo, int n, int C) { return (float*)((char*)slab - (size_t)lo * n * C * elem()); }

// every rank's slab, for a kernel that reads them directly (peer access over NVLink or PCIe peer-to-peer, which
// multigpu.cuh requires): base pointer and first row each
struct PeerSlabs { const uint4* p[mg::MAX_RANKS]; int lo[mg::MAX_RANKS], rows[mg::MAX_RANKS]; int world; };
inline PeerSlabs peerSlabs(const mg::Shared& z, int n) {
  PeerSlabs ps{};
  ps.world = mg::WORLD;
  for (int r = 0; r < mg::WORLD; ++r) { int hi; rowsOf(n, r, ps.lo[r], hi); ps.rows[r] = storedRows(n, r); ps.p[r] = (const uint4*)z.peer[r]; }
  return ps;
}
// zT[j][i] = z[i][lo + j] for this rank's columns j and every row i, read straight from the rank holding row i - one
// kernel over every peer at once (a staged copy a peer, one after another on one stream, ran the 7 peers of 8 x A100
// in series: the transposes were most of what kept the transition at 2.6x on eight GPUs)
__global__ void transposeP2PK(PeerSlabs ps, uint4* __restrict__ zT, int n, int lo, int cols, int vecs) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * cols * vecs) return;
  const int v = (int)(t % vecs); const size_t rest = t / vecs; const int j = (int)(rest % cols), i = (int)(rest / cols);
  int p = 0;
  while (p + 1 < ps.world && i >= ps.lo[p + 1]) ++p;
  zT[((size_t)j * n + i) * vecs + v] = ps.p[p][((size_t)(i - ps.lo[p]) * n + lo + j) * vecs + v];
}
// zT (this rank's rows of z^T: z's columns [lo, hi)) from every rank's slab of z
inline void transpose(const mg::Shared& z, const mg::Shared& zT, int n, int C) {
  const size_t pb = (size_t)C * elem(); const int vecs = (int)(pb / 16);
  int lo, hi; rowsOf(n, mg::RANK, lo, hi); const int cols = storedRows(n, mg::RANK);
  mg::fence();                                    // (every rank's z written)
  if (cols) transposeP2PK<<<blocks((size_t)n * cols * vecs), 256, 0, STREAM>>>(peerSlabs(z, n), (uint4*)zT.local, n, lo, cols, vecs);
  mg::fence();                                    // (no rank rewrites its z while another still reads it)
}

// The triangle's ring: a block of a against every rank's rows of b - this rank's own first, then each peer's staged
// on a copy stream while the GEMM before it runs (two staging buffers where the room allows, else one, in series).
// gemm(m, B, qlo) enqueues the block's product columns [qlo, qlo + m) on STREAM
template <class T, class G>
inline void ringOver(const mg::Shared& bmine, int n, int np, int C, T* const* stages, int nstage, G gemm) {
  static cudaStream_t cs = [] { cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); return s; }();
  static cudaEvent_t ready[2], used[2];
  static bool made = false;
  if (!made) { for (int k = 0; k < 2; ++k) { CK(cudaEventCreateWithFlags(&ready[k], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&used[k], cudaEventDisableTiming)); } made = true; }
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  if (hi > lo) gemm(hi - lo, (const T*)bmine.local, lo);
  std::vector<int> order;
  for (int q = 1; q < mg::WORLD; ++q) { const int r = (mg::RANK + q) % mg::WORLD; int a, b; rowsOf(n, r, a, b); if (b > a) order.push_back(r); }
  if (order.empty()) return;
  for (int k = 0; k < nstage; ++k) CK(cudaEventRecord(used[k], STREAM));    // (a buffer's last reader: an earlier GEMM)
  auto issue = [&](int k) {
    const int r = order[k], slot = k % nstage; int a, b; rowsOf(n, r, a, b);
    CK(cudaStreamWaitEvent(cs, used[slot], 0));
    CK(cudaMemcpyAsync(stages[slot], bmine.peer[r], (size_t)(b - a) * np * C * sizeof(T), cudaMemcpyDefault, cs));
    CK(cudaEventRecord(ready[slot], cs));
  };
  issue(0);
  for (int k = 0; k < (int)order.size(); ++k) {
    if (nstage == 2 && k + 1 < (int)order.size()) issue(k + 1);
    const int slot = k % nstage; int a, b; rowsOf(n, order[k], a, b);
    CK(cudaStreamWaitEvent(STREAM, ready[slot], 0));
    gemm(b - a, (const T*)stages[slot], a);
    CK(cudaEventRecord(used[slot], STREAM));
    if (nstage == 1 && k + 1 < (int)order.size()) issue(k + 1);
  }
}

// The channel-split triangle's exchanges as kernels reading or writing every rank's buffers directly (16 bytes a
// thread): a group's a and b planes pulled - planes [gc][np][np] from rank r's [C][rows_r][np] at channels q0.. - and
// its product's rows pushed into each rank's [C][rows_r][np]. As copies they were 3W a group, ~9,700 a 2,964-token
// trunk pass, each ~17 us of launch the GPU sat idle for.
struct PeerPlanes { uint4* p[mg::MAX_RANKS]; uint4* q[mg::MAX_RANKS]; int lo[mg::MAX_RANKS], rows[mg::MAX_RANKS]; int world; };
inline PeerPlanes peerPlanes(const mg::Shared& a, const mg::Shared* b, int n) {
  PeerPlanes pp{}; pp.world = mg::WORLD;
  for (int r = 0; r < mg::WORLD; ++r) {
    int lo, hi; rowsOf(n, r, lo, hi); pp.lo[r] = lo; pp.rows[r] = hi - lo;
    pp.p[r] = (uint4*)a.peer[r]; pp.q[r] = b ? (uint4*)b->peer[r] : nullptr;
  }
  return pp;
}
__device__ __forceinline__ int ownerOf(const PeerPlanes& pp, int i) { int r = 0; while (r + 1 < pp.world && i >= pp.lo[r + 1]) ++r; return r; }
__global__ void pullPlanesK(PeerPlanes pp, uint4* __restrict__ a, uint4* __restrict__ b, int q0, int gc, int np, int vecs) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)gc * np * vecs) return;
  const int v = (int)(t % vecs); const size_t rest = t / vecs; const int i = (int)(rest % np), c = (int)(rest / np);
  const int r = ownerOf(pp, i);
  const size_t src = ((size_t)(q0 + c) * pp.rows[r] + (i - pp.lo[r])) * vecs + v;
  a[t] = pp.p[r][src]; b[t] = pp.q[r][src];
}
// ...the operands exchanged e4m3 (LOCALFOLD_MG_FP8_EXCHANGE=1: half the bytes of the largest exchange there is) and
// widened to bf16 here for the GEMM - 8 values a thread, vecs8 = np / 8 a row
__device__ __forceinline__ uint4 widen8(uint2 s) {
  const __nv_fp8_e4m3* e = reinterpret_cast<const __nv_fp8_e4m3*>(&s);
  __align__(16) __nv_bfloat16 o[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16(float(e[k]));
  return *reinterpret_cast<const uint4*>(o);
}
__global__ void pullPlanes8K(PeerPlanes pp, uint4* __restrict__ a, uint4* __restrict__ b, int q0, int gc, int np, int vecs8) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)gc * np * vecs8) return;
  const int v = (int)(t % vecs8); const size_t rest = t / vecs8; const int i = (int)(rest % np), c = (int)(rest / np);
  const int r = ownerOf(pp, i);
  const size_t src = ((size_t)(q0 + c) * pp.rows[r] + (i - pp.lo[r])) * vecs8 + v;
  a[t] = widen8(reinterpret_cast<const uint2*>(pp.p[r])[src]);
  b[t] = widen8(reinterpret_cast<const uint2*>(pp.q[r])[src]);
}
__global__ void pushRowsK(PeerPlanes pp, const uint4* __restrict__ prod, int q0, int gc, int np, int vecs) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)gc * np * vecs) return;
  const int v = (int)(t % vecs); const size_t rest = t / vecs; const int i = (int)(rest % np), c = (int)(rest / np);
  const int r = ownerOf(pp, i);
  pp.p[r][((size_t)(q0 + c) * pp.rows[r] + (i - pp.lo[r])) * vecs + v] = prod[t];
}

// every other rank's rows of the attention's bias (columns where swapped: the rows of z^T), read straight from it - the
// planes [heads][n][stride] f16; ownership turns on 16-row boundaries, so no 16-byte piece straddles two ranks
__global__ void gatherBiasK(PeerPlanes pp, uint4* __restrict__ bias, int heads, int n, int stride, bool swap, int rank) {
  const int vecs = stride / 8;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * n * vecs) return;
  const int v = (int)(t % vecs), i = (int)((t / vecs) % n);
  const int r = swap ? (v * 8 < n ? ownerOf(pp, v * 8) : rank) : ownerOf(pp, i);
  if (r == rank) return;
  bias[t] = pp.p[r][t];
}

// a sharded op's scratch given back after it only where the card is short: giving back drains the device
// (releaseScratch's synchronize), four times a block, each a bubble the size of the host's re-enqueue - with a rank's
// kernels an eighth as long, the GPU sat 14% idle; held, the next block reuses the same buffers
inline void releaseIfShort(int n, int C, std::initializer_list<const char*> names) {
  if (!roomFor((size_t)4 * maxStored(n) * n * C * 2)) releaseScratch(names);
}
// the triangle multiplication's outgoing form on a slab (z, or z^T with the incoming weights), its contraction split by
// CHANNEL: p[c][i][j] = sum_k a[c][i][k] b[c][j][k] is independent in c, so rank s takes its share of the channels over
// EVERY row - each rank builds a, b and the gate for its own rows (one pass of the input kernel), rank s pulls its
// channels' planes of every rank's a and b (whole np x np planes), contracts them in one full-size batched GEMM, and
// pushes each rank its rows of the product; each rank then runs the output kernel on its rows. A rank receives
// 2 (W-1)/W n^2 C/W of operands and (W-1)/W of its rows' product where the ring it replaces received (W-1)/W of the
// whole b - 1.97 GB a triangle against 0.74 at 2,964 tokens on 8 - and the GEMMs are the one-GPU shape, where the
// ring's 371-row stages filled cuBLAS's 256-row tiles to 72%. Channels go a group at a time where the planes do not
// all fit (a rank's own choice: nothing meets between groups).
template <class TQ>
// INCOMING (bf16): the incoming form on z itself, p[c][i][j] = sum_k a[c][k][j] b[c][k][i] - the planes a rank gathers
// are whole, so only the GEMM's operands turn (no z^T: two of a block's four transposes gone)
inline void triangleOn(const mg::Shared& z, const float* mask, int n, int C, const std::string& pre, bool divideByLength,
                       bool incoming = false) {
  using B16 = __nv_bfloat16;
  constexpr bool F8 = sizeof(TQ) == 1;
  const int np = padded(n); const size_t cs = (size_t)np * np, plane = (size_t)np * np;
  const float alpha = divideByLength ? 1.f / n : 1.f;
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  const int rows = hi - lo;
  float* base = baseOf(z.local, lo, n, C);
  std::string pg = projectionGate(pre, C);
  half* wt = scratch<half>("trib.wt", triInTileHalves(C));
  tileTriIn(Wh(pg), Wh(pre + ".gatingLinear"), C, 16, wt);
  int maxShare = 0; for (int r = 0; r < mg::WORLD; ++r) { int a0, a1; rowsOf(n, r, a0, a1); maxShare = std::max(maxShare, a1 - a0); }
  const size_t mine = (size_t)maxShare * np * C;
  // (LOCALFOLD_MG_FP8_EXCHANGE=1: a and b written e4m3 for the exchange, widened back to bf16 as they are pulled - the
  // FP8 path's own operand precision, which folds within 0.03 A of bf16's; off by default until a box measures it)
  static const bool X8 = getenv("LOCALFOLD_MG_FP8_EXCHANGE") && atoi(getenv("LOCALFOLD_MG_FP8_EXCHANGE"));
  const bool x8 = X8 && !F8;
  const std::string tag = std::to_string(C) + (F8 ? "f8" : x8 ? "x8" : "");
  // this rank's rows' a and b ([C][rows][np], read by every rank) and its rows' product ([C][rows][np], written by
  // every rank)
  mg::Shared& aS = mg::shared("sh.ta" + tag, mine * (x8 ? 1 : sizeof(TQ)));
  mg::Shared& bS = mg::shared("sh.tb" + tag, mine * (x8 ? 1 : sizeof(TQ)));
  mg::Shared& pS = mg::shared("sh.tp" + std::to_string(C), mine * sizeof(B16));
  half* t2 = scratch<half>("trib.t2", (size_t)std::max(rows, 1) * np * C);
  if (F8 && incoming) { fprintf(stderr, "sharded triangle: the FP8 GEMM takes the outgoing (TN) form only\n"); exit(1); }
  int c0, c1; mg::shareOf(C, 1, mg::RANK, c0, c1);
  const int cg = c1 - c0;
  // channels a group, two groups' planes (a, b and the product) held: group q + 1's pulls (a copy stream) and group
  // q - 1's pushes (another) run beside group q's GEMM - a quarter of the channels a group where there are enough,
  // fewer where the room is short
  int g = std::max(1, (cg + 3) / 4);
  while (g > 1 && !roomFor((size_t)2 * g * plane * (2 * sizeof(TQ) + sizeof(B16)))) g = (g + 1) / 2;
  TQ* aAll[2] = {}; TQ* bAll[2] = {}; B16* pAll[2] = {};
  for (int k = 0; k < 2 && cg; ++k) {
    aAll[k] = scratch<TQ>("trib.aall" + std::to_string(k), (size_t)g * plane);
    bAll[k] = scratch<TQ>("trib.ball" + std::to_string(k), (size_t)g * plane);
    pAll[k] = scratch<B16>("trib.pall" + std::to_string(k), (size_t)g * plane);
  }
  static cudaStream_t pullS = nullptr, pushS = nullptr;
  static cudaEvent_t fenced, pulled[2], gemmed[2], pushed[2];
  if (!pullS) {
    CK(cudaStreamCreateWithFlags(&pullS, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&pushS, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&fenced, cudaEventDisableTiming));
    for (int k = 0; k < 2; ++k)
      for (cudaEvent_t* e : { &pulled[k], &gemmed[k], &pushed[k] }) CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
  }
  const float* lnS = W(pre + ".leftNormInputScale"); const float* lnO = W(pre + ".leftNormInputOffset");
  const RectMap mineRm{lo, np, 0, (size_t)rows * np};
  const PeerPlanes abPlanes = peerPlanes(aS, &bS, n), pPlanes = peerPlanes(pS, nullptr, n);
  const int vecsA = (int)((size_t)np * sizeof(TQ) / 16), vecsP = (int)((size_t)np * sizeof(B16) / 16);
  // (at function scope: an if constexpr inside the generic lambdas below does not discard its other branch)
  auto gemmAll = [&](int gc, const TQ* A, const TQ* Bm, B16* P) {
    if constexpr (F8) fp8GemmTN(np, np, np, alpha, Bm, np, (long long)plane, A, np, (long long)plane, P, np, (long long)plane, gc);
    else if (incoming) bf16Gemms(CUBLAS_OP_N, CUBLAS_OP_T, np, np, np, alpha, A, np, (long long)plane, Bm, np, (long long)plane, P, np, (long long)plane, gc);
    else bf16Gemms(CUBLAS_OP_T, CUBLAS_OP_N, np, np, np, alpha, Bm, np, (long long)plane, A, np, (long long)plane, P, np, (long long)plane, gc);
  };
  wideWidth(C, [&](auto cw) {
    constexpr int CC = decltype(cw)::value, WO = 4;
    wideWarps(C, [&](auto warps) {
      constexpr int WI = decltype(warps)::value;
      WITH_PAIR_T(
        static bool attr = false, attr8 = false;
        constexpr auto kern = triIn256For<CC, WI, TQ, PT>();
        constexpr auto kern8 = triIn256For<CC, WI, __nv_fp8_e4m3, PT>();
        const unsigned grid = (unsigned)(((size_t)rows * np + triInRowsOf(WI) - 1) / triInRowsOf(WI));
        if (rows > 0 && !x8) {
          if (!attr) { smemAttr(kern, (int)wideTriInSmemW(CC, WI)); attr = true; }
          kern<<<grid, 32 * triInWarpsOf(WI), wideTriInSmemW(CC, WI), STREAM>>>(
            base, mask, lnS, lnO, wt, (TQ*)aS.local, (TQ*)bS.local, t2, n, np, cs, nullptr, mineRm);
        } else if (rows > 0) {
          if (!attr8) { smemAttr(kern8, (int)wideTriInSmemW(CC, WI)); attr8 = true; }
          kern8<<<grid, 32 * triInWarpsOf(WI), wideTriInSmemW(CC, WI), STREAM>>>(
            base, mask, lnS, lnO, wt, (__nv_fp8_e4m3*)aS.local, (__nv_fp8_e4m3*)bS.local, t2, n, np, cs, nullptr, mineRm);
        }
        mg::fence();                                  // (every rank's a and b written before any is read)
        CK(cudaEventRecord(fenced, STREAM));
        CK(cudaStreamWaitEvent(pullS, fenced, 0)); CK(cudaStreamWaitEvent(pushS, fenced, 0));
        int q = 0;
        for (int q0 = c0; q0 < c1; q0 += g, ++q) {
          const int gc = std::min(g, c1 - q0), sl = q % 2;
          if (q >= 2) CK(cudaStreamWaitEvent(pullS, gemmed[sl], 0));     // (group q - 2's GEMM done with these planes)
          if (x8) pullPlanes8K<<<blocks((size_t)gc * np * (np / 8)), 256, 0, pullS>>>(abPlanes, (uint4*)aAll[sl], (uint4*)bAll[sl], q0, gc, np, np / 8);
          else pullPlanesK<<<blocks((size_t)gc * np * vecsA), 256, 0, pullS>>>(abPlanes, (uint4*)aAll[sl], (uint4*)bAll[sl], q0, gc, np, vecsA);
          CK(cudaEventRecord(pulled[sl], pullS));
          CK(cudaStreamWaitEvent(STREAM, pulled[sl], 0));
          if (q >= 2) CK(cudaStreamWaitEvent(STREAM, pushed[sl], 0));    // (group q - 2's pushes done reading this product)
          gemmAll(gc, aAll[sl], bAll[sl], pAll[sl]);
          CK(cudaEventRecord(gemmed[sl], STREAM));
          CK(cudaStreamWaitEvent(pushS, gemmed[sl], 0));
          pushRowsK<<<blocks((size_t)gc * np * vecsP), 256, 0, pushS>>>(pPlanes, (const uint4*)pAll[sl], q0, gc, np, vecsP);
          CK(cudaEventRecord(pushed[sl], pushS));
        }
        if (q) { CK(cudaStreamWaitEvent(STREAM, pushed[(q - 1) % 2], 0)); CK(cudaStreamWaitEvent(STREAM, pulled[(q - 1) % 2], 0)); }
        if (q > 1) CK(cudaStreamWaitEvent(STREAM, pushed[q % 2], 0));
        mg::fence();                                  // (every rank's product rows here; no rank reads a or b any more)
        if (rows > 0)
          triangleOutRun<CC, WO, B16>((const B16*)pS.local, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"),
                                      Wh(pre + ".outputProjection"), t2, into(base), n, np, nullptr, mineRm);
      );
    });
  });
  releaseIfShort(n, C, { "trib." });
}
template <class T>
inline void triangleSharded(const mg::Shared& z, const float* mask, int n, int C, const std::string& pre, bool divide,
                            bool incoming = false) {
  if (C > 256) { fprintf(stderr, "sharded pair: the triangle takes 128 or 256 channels, not %d\n", C); exit(1); }
  if (fp8Tensor()) triangleOn<__nv_fp8_e4m3>(z, mask, n, C, pre, divide, incoming);
  else triangleOn<__nv_bfloat16>(z, mask, n, C, pre, divide, incoming);
}

// row attention on a slab (z; or z^T, with the bias's swap flipped - column attention)
inline void rowAttention(const mg::Shared& z, const float* mask, int n, int C, int heads, int D, const std::string& pre,
                         bool swap) {
  const int Wd = heads * D;
  if (!(C == 128 && Wd == 128 && heads <= 16 && D == 32) || hasW(pre + ".gatingQueryBias") || hasW(pre + ".outputProjectionBias")) {
    fprintf(stderr, "sharded pair: the grid attention's fused 128-channel form only (%s)\n", pre.c_str()); exit(1);
  }
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  const int rowsHere = storedRows(n, mg::RANK);
  float* base = baseOf(z.local, lo, n, C);
  const int stride = (n + 7) / 8 * 8;
  mg::Shared& biasS = mg::shared("sh.bias" + std::to_string(heads), (size_t)heads * n * stride * 2);
  half* bias = (half*)biasS.local;
  std::string qkvg = qkvgWeight(pre, C, Wd, true);
  std::string wb = paddedColumns(pre + ".pairBiasProjection", C, heads, 16);
  const float scale = 1.f / sqrtf((float)D);
  CK(cudaMemsetAsync(bias, 0, (size_t)heads * n * stride * 2, STREAM));
  const size_t pairsHere = (size_t)rowsHere * n;
  const bool f8 = (MASK_ALL_ONES || !mask) && fp8Attn();
  // this rank's rows' bias: where every held row's q, k, v and gate fit, ONE input pass writes them and the bias
  // together (the one-GPU form's fusion - the separate LayerNorm-and-project pass and its layout kernel were 65 ms of a
  // simulated rank's 3.0 s trunk at 2,964 tokens on 8); else the bias alone, in chunks, and the projections per chunk
  const bool whole = !f8 && rowsHere && roomFor((pairsHere + 128) * 4 * Wd * 2);
  half* qkvgAll = whole ? scratch<half>("grid.qkvg", (pairsHere + 128) * 4 * Wd) : nullptr;
  if (whole) gridIn128(base, pre, qkvg, qkvgAll, n, (size_t)lo * n, pairsHere, false, Wh(wb), bias, heads, stride, swap);
  else {
    size_t per = std::max<size_t>(1, std::min(pairsHere, CHUNK / 16));
    float* raw = scratch<float>("grid.raw16", per * 16);
    for (size_t q0 = 0; q0 < pairsHere; q0 += per) {
      size_t r = std::min(per, pairsHere - q0);
      lnHeads128<16>(pairRow((float*)z.local, q0, C), pre + ".actNormScale", pre + ".actNormOffset", wb, raw, r);
      biasFromRawRowsK<half><<<blocks((size_t)heads * r), 256, 0, STREAM>>>(raw, bias, (size_t)lo * n + q0, r, n, stride,
                                                                          heads, swap, LOG2E);
    }
  }
  // every rank's: rows of each head's plane, or columns when swapped
  mg::fence();
  gatherBiasK<<<blocks((size_t)heads * n * (stride / 8)), 256, 0, STREAM>>>(peerPlanes(biasS, nullptr, n), (uint4*)bias, heads, n,
                                                                          stride, swap, mg::RANK);
  mg::fence();
  // this rank's rows, in chunks
  size_t R = std::max<size_t>(1, std::min<size_t>(std::max(rowsHere, 1), CHUNK / ((size_t)n * 4 * Wd)));
  {
    size_t f, t; deviceMemInfo(&f, &t);
    size_t perRow = (size_t)n * 5 * Wd * 2, spare = f > t / 16 ? f - t / 16 : 0;
    R = std::max<size_t>(R, std::min<size_t>(rowsHere, std::min<size_t>(spare / perRow, 256)));
  }
  for (size_t r0 = lo; r0 < (size_t)(lo + rowsHere); r0 += R) {
    size_t rows = std::min(R, (size_t)(lo + rowsHere) - r0), prs = rows * n;
    half* qkvgOut = whole ? qkvgAll + (r0 - lo) * n * 4 * Wd : scratch<half>("grid.qkvg", (std::min<size_t>(R, rowsHere) * n + 128) * 4 * Wd);
    uint8_t* kv8 = f8 ? scratch<uint8_t>("grid.kv8", std::min<size_t>(R, rowsHere) * n * 2 * Wd) : nullptr;
    if (!whole) gridIn128(base, pre, qkvg, qkvgOut, n, r0 * n, prs, false, nullptr, nullptr, 0, 0, false, kv8);
    half* gathered = scratch<half>("grid.gathered", std::min<size_t>(R, rowsHere) * n * Wd);
    if (f8) flash8Run(qkvgOut, kv8, kv8 + prs * Wd, bias, stride, gathered, n, heads, rows, scale, nullptr);
    else flashGrid<half>(qkvgOut, bias, stride, MASK_ALL_ONES ? nullptr : mask, gathered, n, heads, D, r0, rows, false, scale);
    gridOut128(gathered, pre + ".outputProjection", into(base), n, r0 * n, prs, false);
  }
  releaseIfShort(n, C, { "grid.qkvg", "grid.gathered", "grid.kv8", "attn.vt8" });
}


// ---- the generic forms (any width, an f32 or bf16 pair: the template stack's 64 channels, rf3's biased attention,
// the 256-channel models), after triangleBlocked's and gridAttention's own generic paths
template <class T>
inline void triangleGenericOn(const mg::Shared& z, const float* mask, int n, int C, const std::string& pre, bool divideByLength) {
  const int np = padded(n); const size_t cs = (size_t)np * np;
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  float* pair = baseOf(z.local, lo, n, C);
  std::string pgOf[2] = { operandWeight(pre, C, 0), operandWeight(pre, C, 1) };
  float alpha = divideByLength ? 1.f / n : 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  size_t per = std::max<size_t>(32, CHUNK / (4 * C));
  float* m = scratch<float>("trib.mask", per);
  T* ln = scratch<T>("trib.ln", per * C); T* pgOut = scratch<T>("trib.pg", per * 2 * C);
  auto operands = [&](const TriRect& r, T* out, int side) {
    for (size_t q0 = 0; q0 < r.size(); q0 += per) {
      size_t cnt = std::min(per, r.size() - q0);
      WITH_PAIR_T(rectLayerNormK<T, PT><<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(pair, mask, ln, m, r, q0, cnt, n, C,
        W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset")));
      linear<T, T>(ln, pgOut, cnt, C, 2 * C, pgOf[side]);
      rectGateK<T><<<dim3((unsigned)((cnt + 31) / 32), (C + 31) / 32), dim3(32, 8), 0, STREAM>>>(pgOut, m, out, q0, cnt, C, r.size());
    }
  };
  int maxShare = 0; for (int r = 0; r < mg::WORLD; ++r) { int a0, a1; rowsOf(n, r, a0, a1); maxShare = std::max(maxShare, a1 - a0); }
  mg::Shared& bmine = mg::shared(std::string("sh.bg") + (sizeof(T) == 2 ? "16." : "32.") + std::to_string(C), (size_t)maxShare * np * C * sizeof(T));
  if (hi > lo) operands({lo, hi - lo, 0, np}, (T*)bmine.local, 1);
  // (a ring, as triangleOn: one rank's b staged at a time against each block of a)
  const size_t stageElems = (size_t)maxShare * np * C;
  T* stages[2] = { mg::WORLD > 1 ? scratch<T>("trib.bstage", stageElems) : nullptr, nullptr };
  if (mg::WORLD > 2 && roomFor(stageElems * sizeof(T) * 2)) stages[1] = scratch<T>("trib.bstage2", stageElems);
  const int nstage = stages[1] ? 2 : 1;
  mg::fence();
  int width = (int)std::max<size_t>(8, std::min<size_t>(np, (CHUNK / C) / np / 8 * 8));
  {
    size_t f, t; deviceMemInfo(&f, &t);
    size_t perRow = (size_t)np * C * (sizeof(T) + 4), spare = f > t / 16 ? f - t / 16 : 0;
    width = (int)std::max<size_t>(width, std::min<size_t>(np, std::min<size_t>(spare / perRow, 1024) / 8 * 8));
  }
  width = std::min(width, std::max(8, hi - lo));
  T* a = scratch<T>("trib.a", (size_t)width * np * C);
  float* prod = scratch<float>("trib.prod", (size_t)width * np * C);
  T* t1 = scratch<T>("trib.t1", per * C); T* t2 = scratch<T>("trib.t2", per * C);
  for (int k0 = lo; k0 < std::min(hi, n); k0 += width) {
    int w = std::min(width, hi - k0);
    TriRect r{k0, w, 0, np};
    operands(r, a, 0);
    ringOver<T>(bmine, n, np, C, stages, nstage, [&](int m, const T* B, int qlo) {
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, m, w, np, &alpha, B, cudaType<T>(), np, (long long)m * np, a,
        cudaType<T>(), np, r.size(), &zero, prod + qlo, CUDA_R_32F, np, r.size(), C, CUBLAS_COMPUTE_32F, algo));
    });
    for (size_t q0 = 0; q0 < r.size(); q0 += per) {
      size_t cnt = std::min(per, r.size() - q0);
      rectCenterNormK<T><<<(unsigned)((cnt + 31) / 32), dim3(32, 8), 0, STREAM>>>(prod, ln, q0, cnt, C, r.size(),
        W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"));
      linear<T, T>(ln, t1, cnt, C, C, pre + ".outputProjection");
      WITH_PAIR_T(rectLayerNormK<T, PT><<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(pair, mask, ln, nullptr, r, q0, cnt, n, C,
        W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset")));
      linear<T, T>(ln, t2, cnt, C, C, pre + ".gatingLinear");
      WITH_PAIR_T(rectGatedAddK<T, PT><<<blocks(cnt * C), 256, 0, STREAM>>>(into(pair), t1, t2, r, q0, cnt, n, C));
    }
  }
  mg::fence();                                    // (no rank rewrites its b rows while another may still copy them)
  releaseIfShort(n, C, { "trib." });
}
// bias[h][i][j] (or [h][j][i] when swapped) = scale * raw[q][h] for the pairs p0 + q, raw pair-major (a linear's)
template <class TB>
__global__ void biasFromFlatRowsK(const float* raw, TB* bias, size_t p0, size_t cnt, int n, int stride, int heads, bool swap,
                                  float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * cnt) return;
  size_t q = t / heads; int h = (int)(t % heads);
  size_t p = p0 + q, a = p / n, b = p % n, i = swap ? b : a, j = swap ? a : b;
  bias[((size_t)h * n + i) * stride + j] = fromF<TB>(scale * raw[t]);
}
template <class T>
inline void rowAttentionGeneric(const mg::Shared& z, const float* mask, int n, int C, int heads, int D, const std::string& pre,
                                bool swap) {
  const int Wd = heads * D;
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  const int rowsHere = storedRows(n, mg::RANK);
  float* pair = baseOf(z.local, lo, n, C);
  constexpr bool fast = std::is_same_v<T, half>;
  const int stride = (n + 7) / 8 * 8;
  mg::Shared& biasS = mg::shared(std::string("sh.biasg") + (fast ? "16." : "32.") + std::to_string(heads), (size_t)heads * n * stride * sizeof(T));
  T* bias = (T*)biasS.local;
  CK(cudaMemsetAsync(bias, 0, (size_t)heads * n * stride * sizeof(T), STREAM));
  const size_t pairsHere = (size_t)rowsHere * n;
  size_t per = std::max<size_t>(1, std::min(std::max<size_t>(pairsHere, 1), CHUNK / C));
  T* lnc = scratch<T>("grid.normChunk", per * C);
  float* raw = scratch<float>("grid.rawbias", per * heads);
  for (size_t q0 = 0; q0 < pairsHere; q0 += per) {
    size_t r = std::min(per, pairsHere - q0);
    lnPairRows<T>((float*)z.local, q0, lnc, r, C, pre + ".actNormScale", pre + ".actNormOffset");
    linear<T, float>(lnc, raw, r, C, heads, pre + ".pairBiasProjection");
    biasFromFlatRowsK<T><<<blocks((size_t)heads * r), 256, 0, STREAM>>>(raw, bias, (size_t)lo * n + q0, r, n, stride, heads, swap,
                                                                        fast ? LOG2E : 1.f);
  }
  mg::fence();
  const size_t e = sizeof(T);
  for (int r = 0; r < mg::WORLD; ++r) {
    if (r == mg::RANK) continue;
    int rlo, rhi; rowsOf(n, r, rlo, rhi); rhi = std::min(rhi, n);
    if (rhi <= rlo) continue;
    for (int h = 0; h < heads; ++h) {
      const size_t plane = (size_t)h * n * stride * e;
      if (!swap) CK(cudaMemcpyAsync((char*)bias + plane + (size_t)rlo * stride * e, (const char*)biasS.peer[r] + plane +
                                    (size_t)rlo * stride * e, (size_t)(rhi - rlo) * stride * e, cudaMemcpyDefault, STREAM));
      else CK(cudaMemcpy2DAsync((char*)bias + plane + (size_t)rlo * e, (size_t)stride * e, (const char*)biasS.peer[r] + plane +
                                (size_t)rlo * e, (size_t)stride * e, (size_t)(rhi - rlo) * e, n, cudaMemcpyDefault, STREAM));
    }
  }
  mg::fence();
  std::string qkvg = qkvgWeight(pre, C, Wd, true);
  const float* gateBias = hasW(pre + ".gatingQueryBias") ? W(pre + ".gatingQueryBias") : nullptr;
  const float* outBias = hasW(pre + ".outputProjectionBias") ? W(pre + ".outputProjectionBias") : nullptr;
  const float scale = 1.f / sqrtf((float)D);
  size_t R = std::max<size_t>(1, std::min<size_t>(std::max(rowsHere, 1), CHUNK / ((size_t)n * 4 * Wd)));
  for (size_t r0 = lo; r0 < (size_t)(lo + rowsHere); r0 += R) {
    size_t rows = std::min(R, (size_t)(lo + rowsHere) - r0), prs = rows * n;
    T* act = scratch<T>("grid.act", std::min<size_t>(R, rowsHere) * n * C);
    lnPairRows<T>((float*)z.local, (r0 - lo) * n, act, prs, C, pre + ".actNormScale", pre + ".actNormOffset");
    T* qkvgOut = scratch<T>("grid.qkvg", (std::min<size_t>(R, rowsHere) * n + 128) * 4 * Wd);
    linear<T, T>(act, qkvgOut, prs, C, 4 * Wd, qkvg);
    if (gateBias) addGateBiasK<T><<<blocks(prs * Wd), 256, 0, STREAM>>>(qkvgOut, gateBias, prs, Wd);
    T* gathered = scratch<T>("grid.gathered", std::min<size_t>(R, rowsHere) * n * Wd);
    flashGrid<T>(qkvgOut, bias, stride, MASK_ALL_ONES && fast ? nullptr : mask, gathered, n, heads, D, r0, rows, false, scale);
    float* o = scratch<float>("grid.o", std::min<size_t>(R, rowsHere) * n * C);
    linear<T, float>(gathered, o, prs, Wd, C, pre + ".outputProjection");
    if (outBias) addBiasK<<<blocks(prs * C), 256, 0, STREAM>>>(o, outBias, prs, C);
    WITH_PAIR_T(addGridK<PT, float><<<blocks(prs * C / 4), 256, 0, STREAM>>>(into(pair), o, n, C, r0, rows, false));
  }
  releaseIfShort(n, C, { "grid." });
}
// A pairformer block's pair updates on the sharded pair z (this rank's slab), zT a slab-sized buffer for z^T
template <class T>
inline void pairUpdates(const mg::Shared& z, const mg::Shared& zT, const float* mask, int n, int C, const std::string& pre,
                        bool swap, bool divide, int transitionFactor) {
  // the fused forms where they apply (the 128/256-channel streaming triangle on a bf16 pair, the 128-channel 4 x 32
  // grid attention without biases), the generic ones otherwise
  auto tri = [&](const mg::Shared& s, const std::string& p) {
    if constexpr (std::is_same_v<T, half>) {
      if (PAIR16 && (C == 128 || C == 256) && blockedFused(n, C)) { triangleSharded<T>(s, mask, n, C, p, divide); return; }
    }
    triangleGenericOn<T>(s, mask, n, C, p, divide);
  };
  int heads = (int)M.meta(pre + ".pairAttention1.heads"), D = (int)M.meta(pre + ".pairAttention1.dimension");
  auto att = [&](const mg::Shared& s, const std::string& p, bool sw) {
    if constexpr (std::is_same_v<T, half>) {
      if (FUSED_GRID && !skipFused("grid") && gridFusedFits() && !RESIDUAL_UNTRANSPOSED && C == 128 && heads * D == 128 && D == 32 &&
          !hasW(p + ".gatingQueryBias") && !hasW(p + ".outputProjectionBias")) {
        rowAttention(s, mask, n, C, heads, D, p, sw); return;
      }
    }
    rowAttentionGeneric<T>(s, mask, n, C, heads, D, p, sw);
  };
  tri(z, pre + ".triangleMultiplicationOutgoing"); stage("tri.out");
  // the incoming form straight on z where the fused bf16 triangle runs (no transposes); else as the outgoing form on z^T
  bool direct = false;
  if constexpr (std::is_same_v<T, half>) direct = PAIR16 && (C == 128 || C == 256) && blockedFused(n, C) && !fp8Tensor();
  if (direct) triangleSharded<T>(z, mask, n, C, pre + ".triangleMultiplicationIncoming", divide, true);
  else {
    transpose(z, zT, n, C);
    tri(zT, pre + ".triangleMultiplicationIncoming");
    transpose(zT, z, n, C);
  }
  stage("tri.in");
  att(z, pre + ".pairAttention1", false); stage("grid.row");
  transpose(z, zT, n, C);
  att(zT, pre + ".pairAttention2", !swap); stage("grid.col");
  transpose(zT, z, n, C);
  const int rowsHere = storedRows(n, mg::RANK);
  if (rowsHere) transition<T>((float*)z.local, (size_t)rowsHere * n, C, transitionFactor, pre + ".pairTransition");
  stage("transition");
}

}  // namespace sh
