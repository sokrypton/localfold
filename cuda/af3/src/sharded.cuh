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
// (every exchange kernel walks the rows from this rank's own first row, wrapping - `rot`: walked from row 0 by every
// rank at once, all eight GPUs read one peer, then the next, and an all-to-all of 268 MB a rank on 8 x A100 ran at
// 56 GB/s a GPU against 120 rotated; the copy engines' 64 against 233)
struct PeerSlabs { const uint4* p[mg::MAX_RANKS]; int lo[mg::MAX_RANKS], rows[mg::MAX_RANKS]; int world, rot; };
inline PeerSlabs peerSlabs(const mg::Shared& z, int n) {
  PeerSlabs ps{};
  ps.world = mg::WORLD;
  { int hi; rowsOf(n, mg::RANK, ps.rot, hi); }
  for (int r = 0; r < mg::WORLD; ++r) { int hi; rowsOf(n, r, ps.lo[r], hi); ps.rows[r] = storedRows(n, r); ps.p[r] = (const uint4*)z.peer[r]; }
  return ps;
}
// a transpose, pushed: this rank's rows [j0, j0 + cols) of a slab (src, from row j0) written into column j0.. of every row i
// of the other slab, in the part of the rank holding row i (the writes contiguous along j) - z^T home into z, or z out
// into z^T (src z's rows, ps z^T's slabs, j0 this rank's first row)
__global__ void pushTransposeK(PeerSlabs ps, const uint4* __restrict__ zT, int n, int j0, int cols, int vecs) {
  const size_t total_ = (size_t)n * cols * vecs;
  for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total_; t += (size_t)gridDim.x * blockDim.x) {
    const int v = (int)(t % vecs); const size_t rest = t / vecs; const int j = (int)(rest % cols), i = (int)((rest / cols + ps.rot) % n);
    int p = 0;
    while (p + 1 < ps.world && i >= ps.lo[p + 1]) ++p;
    ((uint4*)ps.p[p])[((size_t)(i - ps.lo[p]) * n + j0 + j) * vecs + v] = zT[((size_t)j * n + i) * vecs + v];
  }
}

// ...a chunk at a time: this rank's rows [lo, lo + rows) of z (src) at every rank's chunk k of its columns, into that
// rank's z^T - rank p's chunk its columns [lo_p + k P_p, lo_p + min(rows_p, (k + 1) P_p)), P_p = ceil(rows_p / chunks)
// (slot: each destination's chunk written from its slot's row 0 - a buffer of one chunk, not the share)
__global__ void pushTransposeChunkK(PeerSlabs ps, const uint4* __restrict__ src, int n, int lo, int rows, int vecs, int k,
                                    int chunks, int Pmax, int rank, bool slot = false) {
  const size_t total_ = (size_t)ps.world * Pmax * rows * vecs;
  for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total_; t += (size_t)gridDim.x * blockDim.x) {
    const int v = (int)(t % vecs); size_t rest = t / vecs;
    const int i = (int)(rest % rows); rest /= rows;
    const int jj = (int)(rest % Pmax), p = (int)((rest / Pmax + rank) % ps.world);
    const int P = (ps.rows[p] + chunks - 1) / chunks, jl = k * P + jj;
    if (jj >= P || jl >= ps.rows[p]) continue;
    const int j = ps.lo[p] + jl;
    ((uint4*)ps.p[p])[((size_t)(slot ? jj : jl) * n + lo + i) * vecs + v] = src[((size_t)i * n + j) * vecs + v];
  }
}
// A push BESIDE work (on a side stream, overlapping a GEMM or an attention): a few SMs' worth of blocks striding over it,
// so the work it overlaps keeps the rest of the card - with a block a 16-byte piece every SM held a slot for as long as
// the transfer took, which over a real link is ~10x a simulated rank's local writes (NVLink fills from ~16 SMs: NCCL's
// channels). 64 blocks of 256 copy 381 GB/s locally (32: 214, the whole card 684), over the ~250-300 GB/s an A100's links
// take outbound; the exposed pushes, on the stream, keep the whole card
constexpr unsigned SIDE_PUSH_BLOCKS = 64;
inline unsigned sideBlocks(size_t threads) { return std::min<unsigned>(SIDE_PUSH_BLOCKS, (unsigned)blocks(threads)); }
// zT (this rank's rows of z^T: z's columns [lo, hi)) from every rank's slab of z
inline void transpose(const mg::Shared& z, const mg::Shared& zT, int n, int C) {
  const size_t pb = (size_t)C * elem(); const int vecs = (int)(pb / 16);
  int lo, hi; rowsOf(n, mg::RANK, lo, hi); const int cols = storedRows(n, mg::RANK);
  // (pushed: this rank's rows of z into every rank's z^T, as pushTransposeK takes z^T home - a read of another GPU's
  // memory through a PCIe host bridge ran at a thirtieth of a write)
  mg::fence();                                    // (every rank's z written; its z^T free)
  if (cols) pushTransposeK<<<blocks((size_t)n * cols * vecs), 256, 0, STREAM>>>(peerSlabs(zT, n), (const uint4*)z.local, n, lo, cols, vecs);
  mg::sent("transposes", (size_t)n * cols * pb * (mg::WORLD - 1) / mg::WORLD, false);
  mg::fence();                                    // (every rank's columns here)
}

// The triangle's ring: a block of a against every rank's rows of b - this rank's own first, then each peer's staged
// on a copy stream while the GEMM before it runs (two staging buffers where the room allows, else one, in series).
// gemm(m, B, qlo) enqueues the block's product columns [qlo, qlo + m) on STREAM
template <class T, class G>
// (channels [c0, c0 + cw) of each share alone - the shares are channel-major [C][rows][np], so a window is one run of
// them; gemm sees the window's planes)
inline void ringOver(const mg::Shared& bmine, int n, int np, int c0, int cw, T* const* stages, int nstage, G gemm) {
  static cudaStream_t cs = [] { cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); return s; }();
  static cudaEvent_t ready[2], used[2];
  static bool made = false;
  if (!made) { for (int k = 0; k < 2; ++k) { CK(cudaEventCreateWithFlags(&ready[k], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&used[k], cudaEventDisableTiming)); } made = true; }
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  if (hi > lo) gemm(hi - lo, (const T*)bmine.local + (size_t)c0 * (hi - lo) * np, lo);
  std::vector<int> order;
  for (int q = 1; q < mg::WORLD; ++q) { const int r = (mg::RANK + q) % mg::WORLD; int a, b; rowsOf(n, r, a, b); if (b > a) order.push_back(r); }
  if (order.empty()) return;
  for (int k = 0; k < nstage; ++k) CK(cudaEventRecord(used[k], STREAM));    // (a buffer's last reader: an earlier GEMM)
  auto issue = [&](int k) {
    const int r = order[k], slot = k % nstage; int a, b; rowsOf(n, r, a, b);
    CK(cudaStreamWaitEvent(cs, used[slot], 0));
    CK(cudaMemcpyAsync(stages[slot], (const T*)bmine.peer[r] + (size_t)c0 * (b - a) * np, (size_t)(b - a) * np * cw * sizeof(T),
                       cudaMemcpyDefault, cs));
    mg::sent("triangle ring", (size_t)(b - a) * np * cw * sizeof(T), true);
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
// (rows walked from this rank's own, as PeerSlabs's)
struct PeerPlanes { uint4* p[mg::MAX_RANKS]; uint4* q[mg::MAX_RANKS]; int lo[mg::MAX_RANKS], rows[mg::MAX_RANKS]; int world, rot, rank, maxRows; };
inline PeerPlanes peerPlanes(const mg::Shared& a, const mg::Shared* b, int n) {
  PeerPlanes pp{}; pp.world = mg::WORLD; pp.rank = mg::RANK;
  { int hi; rowsOf(n, mg::RANK, pp.rot, hi); }
  for (int r = 0; r < mg::WORLD; ++r) {
    int lo, hi; rowsOf(n, r, lo, hi); pp.lo[r] = lo; pp.rows[r] = hi - lo;
    pp.p[r] = (uint4*)a.peer[r]; pp.q[r] = b ? (uint4*)b->peer[r] : nullptr; pp.maxRows = std::max(pp.maxRows, hi - lo);
  }
  return pp;
}
__device__ __forceinline__ int ownerOf(const PeerPlanes& pp, int i) { int r = 0; while (r + 1 < pp.world && i >= pp.lo[r + 1]) ++r; return r; }
// ...the operands exchanged e4m3 (LOCALFOLD_MG_FP8_EXCHANGE=1: half the bytes of the largest exchange there is) and
// widened to bf16 here for the GEMM - 8 values a thread, vecs8 = np / 8 a row
__device__ __forceinline__ uint4 widen8(uint2 s) {
  const __nv_fp8_e4m3* e = reinterpret_cast<const __nv_fp8_e4m3*>(&s);
  __align__(16) __nv_bfloat16 o[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16(float(e[k]));
  return *reinterpret_cast<const uint4*>(o);
}
// each rank's slot of a step: this rank's rows [lo, lo + rows) of its channels [ch, ch + cnt) of this rank's a and b
// ([C][rows][np] from channel 0, as sent: bf16, or e4m3 under LOCALFOLD_MG_FP8_EXCHANGE), written into its planes
// [g][np][np]; the ranks walked from this one (rot)
struct PushSlots { uint4* a[mg::MAX_RANKS]; uint4* b[mg::MAX_RANKS]; int ch[mg::MAX_RANKS], cnt[mg::MAX_RANKS]; int world, rank; };
// ...and the same moves by COPY ENGINES: each (rank, channel) piece is one run at both ends - a channel's plane's rows
// [lo, lo + rows), np wide - so cudaMemcpyAsync between devices takes it with no SM held, where the push kernels held
// theirs for as long as the link took (a simulated rank's local writes hide that; a box's NVLink is ~10x slower). The
// destinations from the next rank on, as the kernels go
inline void copyPlanes(const PushSlots& ps, const void* srcA, const void* srcB, int rows, int lo, int np, int vecs, int g, cudaStream_t st) {
  const size_t piece = (size_t)rows * vecs * 16;
  for (int k = 0; k < ps.world; ++k) {
    const int r = (ps.rank + k) % ps.world;
    for (int c = 0; c < ps.cnt[r] && c < g; ++c) {
      const size_t src = (size_t)(ps.ch[r] + c) * piece, dst = ((size_t)c * np + lo) * vecs * 16;
      CK(cudaMemcpyAsync((char*)ps.a[r] + dst, (const char*)srcA + src, piece, cudaMemcpyDefault, st));
      CK(cudaMemcpyAsync((char*)ps.b[r] + dst, (const char*)srcB + src, piece, cudaMemcpyDefault, st));
    }
  }
}
inline void copyRows(const PeerPlanes& pp, const void* prod, int q0, int gc, int np, int vecs, cudaStream_t st) {
  for (int k = 0; k < pp.world; ++k) {
    const int r = (pp.rank + k) % pp.world;
    if (pp.rows[r] <= 0) continue;
    const size_t piece = (size_t)pp.rows[r] * vecs * 16;
    for (int c = 0; c < gc; ++c)
      CK(cudaMemcpyAsync((char*)pp.p[r] + (size_t)(q0 + c) * piece, (const char*)prod + ((size_t)c * np + pp.lo[r]) * vecs * 16,
                         piece, cudaMemcpyDefault, st));
  }
}
// e4m3 planes widened to bf16, 8 values a thread
__global__ void widenPlanesK(const uint2* __restrict__ in, uint4* __restrict__ out, size_t n8) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n8) out[t] = widen8(in[t]);
}
// (ONE DESTINATION AT A TIME, each rank starting at its own: through a PCIe host bridge 4 ranks' writes ran at 17 GB/s
// a GPU so, at 6 where each GPU's consecutive writes went to every owner in turn - row by row within each channel)

// this rank's rows of the attention's bias (columns where swapped: the rows of z^T) written into every other rank's -
// the planes [heads][n][stride] f16; ownership turns on 16-row boundaries, so no 16-byte piece straddles two ranks
__global__ void pushBiasK(PeerPlanes pp, const uint4* __restrict__ bias, int heads, int n, int stride, bool swap, int rank) {
  // (one destination at a time, each rank from the next: pushRowsK's note; a piece is this rank's when its row is, or
  // its 8 columns' first - a column past n the last rank's)
  const int vecs = stride / 8, lo = pp.lo[rank], hi = rank + 1 < pp.world ? pp.lo[rank + 1] : (swap ? stride : n);
  const int own = swap ? (min(hi, stride) - lo) / 8 : min(hi, n) - lo;       // (vec columns, or rows)
  const size_t per = swap ? (size_t)heads * n * own : (size_t)heads * own * vecs;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (own <= 0 || t >= per * (pp.world - 1)) return;
  const int p = (rank + 1 + (int)(t / per)) % pp.world;
  const size_t e = t % per;
  size_t at;
  if (swap) { const int v = lo / 8 + (int)(e % own); const size_t hi_ = e / own; at = hi_ * vecs + v; }   // (h, i) whole rows
  else { const int v = (int)(e % vecs), i = lo + (int)(e / vecs % own), h = (int)(e / vecs / own); at = ((size_t)h * n + i) * vecs + v; }
  pp.p[p][at] = bias[at];
}

// a sharded op's scratch given back after it only where the card is short: giving back drains the device
// (releaseScratch's synchronize), four times a block, each a bubble the size of the host's re-enqueue - with a rank's
// kernels an eighth as long, the GPU sat 14% idle; held, the next block reuses the same buffers
inline void releaseIfShort(int n, int C, std::initializer_list<const char*> names) {
  cudaStreamCaptureStatus cap; CK(cudaStreamIsCapturing(STREAM, &cap));
  if (cap == cudaStreamCaptureStatusNone && !roomFor((size_t)4 * maxStored(n) * n * C * 2)) releaseScratch(names);
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
// a weight's name from its module's prefix: AlphaFold 3's "pre.leaf"; cuda/ef2 sets its own ("f/pre/leaf") to run
// ESMFold2's trunk through triangleOn
inline std::string (*WNAME)(const std::string&, const char*) = [](const std::string& pre, const char* leaf) {
  return pre + "." + leaf;
};
// what the triangle's whole form holds on this rank beyond z (an operand of TQBYTES a channel): a, b and the gate a slab
// each, two steps' slots, two products, the product share, and 1 GB spare - what its rounds decision asks every rank for
inline size_t wholeFormBytes(int n, int C, size_t tqBytes) {
  const int np = padded(n); const size_t plane = (size_t)np * np;
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  int maxShare = 0; for (int r = 0; r < mg::WORLD; ++r) { int a0, a1; rowsOf(n, r, a0, a1); maxShare = std::max(maxShare, a1 - a0); }
  const size_t mine = (size_t)maxShare * np * C;
  const int gs = getenv("LOCALFOLD_MG_TRI_GROUPS") ? atoi(getenv("LOCALFOLD_MG_TRI_GROUPS")) : 0;
  const int grp = gs > 0 ? gs : np >= 4096 ? 8 : np >= 2048 ? 4 : 1, cg = (C + mg::WORLD - 1) / mg::WORLD;
  const size_t gw = (size_t)((cg + grp - 1) / grp), slab = (size_t)std::max(hi - lo, 1) * np * C;
  return 2 * slab * tqBytes + slab * 2 + 4 * gw * plane * tqBytes + 2 * gw * plane * 2 + mine * 2 + ((size_t)1 << 30);
}
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
  std::string pg = concatColumns(WNAME(pre, "projectionGate~"), C, {{WNAME(pre, "projection"), 2 * C, false},
                                                                      {WNAME(pre, "gate"), 2 * C, false}});
  half* wt = scratch<half>("trib.wt", triInTileHalves(C));
  tileTriIn(Wh(pg), Wh(WNAME(pre, "gatingLinear")), C, 16, wt);
  int maxShare = 0; for (int r = 0; r < mg::WORLD; ++r) { int a0, a1; rowsOf(n, r, a0, a1); maxShare = std::max(maxShare, a1 - a0); }
  const size_t mine = (size_t)maxShare * np * C;
  // (LOCALFOLD_MG_FP8_EXCHANGE=1: a and b written e4m3 for the exchange, widened back to bf16 as they are pulled - the
  // FP8 path's own operand precision, which folds within 0.03 A of bf16's; off by default until a box measures it)
  static const bool X8 = getenv("LOCALFOLD_MG_FP8_EXCHANGE") && atoi(getenv("LOCALFOLD_MG_FP8_EXCHANGE"));
  const bool x8 = X8 && !F8;
  const std::string tag = std::to_string(C) + (F8 ? "f8" : x8 ? "x8" : "");
  // this rank's rows' product ([C][rows][np]) is written by every rank
  // ROUNDS (a card short of room; LOCALFOLD_MG_TRI_ROUNDS=1/0 forces either): a and b made a window of K channels at a
  // time, every rank its rows' for the window, each rank contracting its slice of it - this rank's a and b a window
  // (K/C of a slab each) where they were two whole slabs: 9.4 -> 2.4 GB a rank at 12,103 tokens on 8. The input
  // pass's LayerNorm repeated a window
  static const int R_SET = getenv("LOCALFOLD_MG_TRI_ROUNDS") ? atoi(getenv("LOCALFOLD_MG_TRI_ROUNDS")) : -1;
  // (asked of every rank, once a shape: the whole form's own buffers - a, b and the gate a slab each, two steps' slots,
  // two products, the product share - against what each has free. shortPair's rule, 18x an f32 pair a GPU, put a
  // simulated rank of 8 in rounds at 4,940 tokens with 30 GB to spare, where the input pass run a window at a time was
  // 41% of its GPU time)
  static std::map<std::pair<int, int>, bool> roundsOf;
  bool rounds = false;
  if (!F8 && !x8) {
    if (R_SET >= 0) rounds = R_SET > 0;
    else if (BIG_FORCED) rounds = true;
    else {
      auto it = roundsOf.find({ n, C });
      if (it == roundsOf.end()) {
        it = roundsOf.emplace(std::make_pair(n, C), !mg::everyRankHas(wholeFormBytes(n, C, sizeof(TQ)))).first;
      }
      rounds = it->second;
    }
  }
  // (a 256-channel pair - ESMFold2's - in windows of C / 16: the windows, their slots and the gate's chunk at C / 8 left a
  // simulated rank of 8 0.6 GB short at 14,820 tokens beside its two 14 GB shares)
  mg::Shared& pS = mg::shared("sh.tp" + std::to_string(C), mine * sizeof(B16));
  // ...and on a card shorter still (every rank asked, once a shape), windows and gate chunks of half that: C / 32 - at
  // 16,549 tokens on 8 x 40 GB the two 17.5 GB shares left 5 GB, where C / 16's windows, slots, product and gate are 5.5
  static std::map<std::pair<int, int>, int> narrowOf;
  int div = C >= 256 ? 16 : 8;
  if (rounds && C >= 256) {
    auto it = narrowOf.find({ n, C });
    if (it == narrowOf.end()) {
      const size_t k16 = C / 16, wins = 2 * (size_t)std::max(rows, 1) * np * k16 * sizeof(TQ) + 3 * ((k16 + mg::WORLD - 1) / mg::WORLD) * plane * 2
                       + (size_t)((rows + 15) / 16) * np * C * 2 + ((size_t)1 << 30);
      it = narrowOf.emplace(std::make_pair(n, C), mg::everyRankHas(wins) ? 16 : 32).first;
      if (it->second == 32 && mg::RANK == 0 && getenv("LOCALFOLD_MEM")) fprintf(stderr, "  sharded triangle at %d tokens: windows of C / 32\n", n);
    }
    div = it->second;
  }
  const int K = std::max(8, C / div), gR = (K + mg::WORLD - 1) / mg::WORLD;    // (K: a multiple of the input pass's 8)
  // (the gate: made in the input pass for every row and held to the output - a slab of it; in ROUNDS, a card short of
  // room, made a chunk of rows at a time just before the output pass instead: a slab less, 7.6 GB a rank at 10,868
  // tokens on 4, for one more read of the rows and the gating linear again)
  const int gateRows = rounds ? std::max(16, (rows + div - 1) / div) : std::max(rows, 1);
  half* t2 = scratch<half>("trib.t2", (size_t)gateRows * np * C);
  if (F8 && incoming) { fprintf(stderr, "sharded triangle: the FP8 GEMM takes the outgoing (TN) form only\n"); exit(1); }
  // every rank's channels (the plain form: a share of C each, a group at a time)
  int cLo[mg::MAX_RANKS], cHi[mg::MAX_RANKS], maxCg = 0;
  for (int r = 0; r < mg::WORLD; ++r) { mg::shareOf(C, 1, r, cLo[r], cHi[r]); maxCg = std::max(maxCg, cHi[r] - cLo[r]); }
  // channels a group, two groups' planes held: group q - 1's product pushes (a side stream) run beside group q's GEMM
  // (small folds one group: a quarter of a rank's channels is a GEMM too small to fill the card - at 988 tokens four
  // groups of four planes ran the contraction at 70% of one group's rate. Past 2,048 the exchange and the GEMM are of a
  // size and the exposed first exchange and last push shrink as 1/groups; LOCALFOLD_MG_TRI_GROUPS=k sets it)
  static const int G_SET = getenv("LOCALFOLD_MG_TRI_GROUPS") ? atoi(getenv("LOCALFOLD_MG_TRI_GROUPS")) : 0;
  const int groups = G_SET > 0 ? G_SET : np >= 4096 ? 8 : np >= 2048 ? 4 : 1;
  // (alike on every rank: the slots below are shared, and every rank meets the same fences)
  const int g = rounds ? gR : std::max(1, (maxCg + groups - 1) / groups);
  const int steps = rounds ? (C + K - 1) / K : (maxCg + g - 1) / g;
  // EVERY EXCHANGE A PUSH: the rank holding a row writes it into the rank that wants it. Reading another GPU's memory
  // through a PCIe host bridge ran at 0.4-0.9 GB/s a GPU on 4 x A6000 where writing it ran at 15-20 (a write is posted;
  // a read waits for its answer) - a 988-token fold took 52.7 s on the four against 3.0 on one, 39 of its trunk's 47 s
  // the triangle's pulls; over NVLink the two are alike (118 GB/s pushed, 120-147 pulled on 8 x A100).
  // This rank's rows of a and b (all channels; or a window of K at a time, in rounds), local, and each rank's slots for
  // the planes it contracts ([g][np][np], two steps' worth), written by every rank
  const size_t aw = (size_t)std::max(rows, 1) * np * (rounds ? K : C);
  void* aL = x8 ? (void*)scratch<__nv_fp8_e4m3>("trib.a8", aw) : (void*)scratch<TQ>("trib.a", aw);
  void* bL = x8 ? (void*)scratch<__nv_fp8_e4m3>("trib.b8", aw) : (void*)scratch<TQ>("trib.b", aw);
  // (in rounds one step's slots, not two: a step's fill does not overlap the GEMM before it there, and a fence ahead of
  // each fill keeps a peer's GEMM off a slot being refilled - 2.4 GB a rank at 12,350 tokens on 4)
  mg::Shared* slotA[2]; mg::Shared* slotB[2];
  for (int k = 0; k < (rounds ? 1 : 2); ++k) {
    slotA[k] = &mg::shared("sh.tsa" + tag + "." + std::to_string(k), (size_t)g * plane * (x8 ? 1 : sizeof(TQ)));
    slotB[k] = &mg::shared("sh.tsb" + tag + "." + std::to_string(k), (size_t)g * plane * (x8 ? 1 : sizeof(TQ)));
  }
  if (rounds) { slotA[1] = slotA[0]; slotB[1] = slotB[0]; }
  TQ* aW = x8 ? scratch<TQ>("trib.aw", (size_t)g * plane) : nullptr;   // (x8: a step's planes widened from the wire's e4m3)
  TQ* bW = x8 ? scratch<TQ>("trib.bw", (size_t)g * plane) : nullptr;
  B16* pAll[2] = {};
  pAll[0] = scratch<B16>("trib.pall0", (size_t)g * plane);
  // (a second product - one step's push beside the next one's GEMM - where it fits)
  if (!rounds || roomFor((size_t)g * plane * sizeof(B16))) pAll[1] = scratch<B16>("trib.pall1", (size_t)g * plane);
  static cudaStream_t pushS = nullptr, xS = nullptr;
  static cudaEvent_t gemmed[2], pushed[2], fenced, filled;
  if (!pushS) {
    CK(cudaStreamCreateWithFlags(&pushS, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&xS, cudaStreamNonBlocking));
    for (int k = 0; k < 2; ++k) for (cudaEvent_t* e : { &gemmed[k], &pushed[k] }) CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&fenced, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&filled, cudaEventDisableTiming));
  }
  const float* lnS = W(WNAME(pre, "leftNormInputScale")); const float* lnO = W(WNAME(pre, "leftNormInputOffset"));
  const RectMap mineRm{lo, np, 0, (size_t)rows * np};
  const PeerPlanes pPlanes = peerPlanes(pS, nullptr, n);
  const int vecsP = (int)((size_t)np * sizeof(B16) / 16);
  const int vecsW = (int)((size_t)np * (x8 ? 1 : sizeof(TQ)) / 16);   // (16-byte pieces of a slot's row, as sent)
  // (at function scope: an if constexpr inside the generic lambdas below does not discard its other branch)
  auto gemmAll = [&](int gc, const TQ* A, const TQ* Bm, B16* P) {
    if constexpr (F8) fp8GemmTN(np, np, np, alpha, Bm, np, (long long)plane, A, np, (long long)plane, P, np, (long long)plane, gc);
    else if (incoming) bf16Gemms(CUBLAS_OP_N, CUBLAS_OP_T, np, np, np, alpha, A, np, (long long)plane, Bm, np, (long long)plane, P, np, (long long)plane, gc);
    else bf16Gemms(CUBLAS_OP_T, CUBLAS_OP_N, np, np, np, alpha, Bm, np, (long long)plane, A, np, (long long)plane, P, np, (long long)plane, gc);
  };
  // step q's channels for rank r: [from, to), and where they start in this rank's a and b
  auto range = [&](int q, int r, int& from, int& to) {
    if (rounds) { const int k0 = q * K, k1 = std::min(C, k0 + K); from = std::min(k1, k0 + r * gR); to = std::min(k1, from + gR); }
    else { from = std::min(cHi[r], cLo[r] + q * g); to = std::min(cHi[r], from + g); }
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
        // this rank's rows of a and b, channels [k0, k1) (the gate's too where t2 is handed), at aL's channel 0
        // (a window - rounds' a few channels, one or two of the kernel's 8-channel steps a block - on the plain 8-warp form
        // where the whole pass takes the two-tile one: two tiles a warp pay off over 32 steps' weight fragments, and a
        // window is its rows' loads - 2.26 -> 1.60 ms a C / 32 window at 4,940 tokens on a simulated rank of 8, with
        // lnRowsToShared's 16-byte rows)
        constexpr auto kernW = triIn256For<CC, 8, TQ, PT>();
        static bool attrW = false;
        auto produce = [&](int k0, int k1, half* gate) {
          if (rows <= 0) return;
          RectMap win = mineRm; win.ab0 = k0 / 8; win.ab1 = k1 / 8;
          const size_t back = (size_t)k0 * mineRm.size;
          if (!x8 && rounds && WI == TRIIN_TWO_TILES) {
            if (!attrW) { smemAttr(kernW, (int)wideTriInSmemW(CC, 8)); attrW = true; }
            kernW<<<(unsigned)(((size_t)rows * np + triInRowsOf(8) - 1) / triInRowsOf(8)), 32 * 8, wideTriInSmemW(CC, 8), STREAM>>>(
              base, mask, lnS, lnO, wt, (TQ*)aL - back, (TQ*)bL - back, gate, n, np, cs, nullptr, win);
          } else if (!x8) {
            if (!attr) { smemAttr(kern, (int)wideTriInSmemW(CC, WI)); attr = true; }
            kern<<<grid, 32 * triInWarpsOf(WI), wideTriInSmemW(CC, WI), STREAM>>>(
              base, mask, lnS, lnO, wt, (TQ*)aL - back, (TQ*)bL - back, gate, n, np, cs, nullptr, win);
          } else {
            if (!attr8) { smemAttr(kern8, (int)wideTriInSmemW(CC, WI)); attr8 = true; }
            kern8<<<grid, 32 * triInWarpsOf(WI), wideTriInSmemW(CC, WI), STREAM>>>(
              base, mask, lnS, lnO, wt, (__nv_fp8_e4m3*)aL - back, (__nv_fp8_e4m3*)bL - back, gate, n, np, cs, nullptr, win);
          }
        };
        // (DIAG LOCALFOLD_MG_TRI_TIMES=1: the phases' GPU time on rank 0's stream, printed every 96 triangles)
        static const bool tt = getenv("LOCALFOLD_MG_TRI_TIMES") != nullptr;
        static double acc[4] = {}; static int calls = 0;
        std::vector<cudaEvent_t> te;
        auto mark = [&]() { if (!tt) return; cudaEvent_t e; CK(cudaEventCreate(&e)); CK(cudaEventRecord(e, STREAM)); te.push_back(e); };
        mark();
        if (!rounds) produce(0, C, t2);
        mark();
        // step q's slots filled: this rank's rows of every rank's channels for the step, into that rank's slot (on `st`)
        auto fill = [&](int q, cudaStream_t st) {
          const int sl = q % 2, k0 = rounds ? q * K : 0;
          PushSlots ps{}; ps.world = mg::WORLD; ps.rank = mg::RANK;
          for (int r = 0; r < mg::WORLD; ++r) {
            int f, t; range(q, r, f, t);
            ps.a[r] = (uint4*)slotA[sl]->peer[r]; ps.b[r] = (uint4*)slotB[sl]->peer[r]; ps.ch[r] = f - k0; ps.cnt[r] = t - f;
          }
          if (rows > 0) copyPlanes(ps, aL, bL, rows, lo, np, vecsW, g, st);
          if (rows > 0) mg::sent("triangle operands", (size_t)2 * (mg::WORLD - 1) * g * rows * vecsW * 16, st != STREAM);
        };
        // OVERLAPPED (not in rounds): step q + 1's slots pushed on a side stream while step q's GEMM runs - its slots were
        // last read by step q - 1's GEMM, which every rank finished before passing step q's fence. Through a PCIe host
        // bridge the pushes were 107 of a triangle's 135 ms at 2,964 tokens on 4 x A6000, beside a 12 ms GEMM, in series
        bool used[2] = {};
        if (rounds) produce(0, std::min(C, K), nullptr);
        fill(0, STREAM);
        mg::fence();
        for (int q = 0; q < steps; ++q) {
          const int sl = q % 2, sp = pAll[1] ? sl : 0;
          const bool next = q + 1 < steps;
          if (next && !rounds) { CK(cudaEventRecord(fenced, STREAM)); CK(cudaStreamWaitEvent(xS, fenced, 0)); fill(q + 1, xS); CK(cudaEventRecord(filled, xS)); }
          int m0, m1; range(q, mg::RANK, m0, m1);
          const int gc = m1 - m0;
          if (gc > 0) {
            if (used[sp]) CK(cudaStreamWaitEvent(STREAM, pushed[sp], 0));    // (the last pushes from this product done)
            const TQ *A = (const TQ*)slotA[sl]->local, *Bm = (const TQ*)slotB[sl]->local;
            if (x8) {        // (e4m3 on the wire, widened here)
              widenPlanesK<<<blocks((size_t)gc * plane / 8), 256, 0, STREAM>>>((const uint2*)A, (uint4*)aW, (size_t)gc * plane / 8);
              widenPlanesK<<<blocks((size_t)gc * plane / 8), 256, 0, STREAM>>>((const uint2*)Bm, (uint4*)bW, (size_t)gc * plane / 8);
              A = aW; Bm = bW;
            }
            gemmAll(gc, A, Bm, pAll[sp]);
            CK(cudaEventRecord(gemmed[sp], STREAM));
            CK(cudaStreamWaitEvent(pushS, gemmed[sp], 0));
            copyRows(pPlanes, pAll[sp], m0, gc, np, vecsP, pushS);
            mg::sent("triangle products", (size_t)(mg::WORLD - 1) * gc * pPlanes.maxRows * vecsP * 16, true);
            CK(cudaEventRecord(pushed[sp], pushS));
            used[sp] = true;
          }
          if (!next) break;
          if (rounds) { const int k1 = (q + 1) * K; produce(k1, std::min(C, k1 + K), nullptr); mg::fence(); fill(q + 1, STREAM); }
          else CK(cudaStreamWaitEvent(STREAM, filled, 0));
          mg::fence();                                // (every rank's rows in this rank's next slot)
        }
        mark();
        for (int k = 0; k < 2; ++k) if (used[k]) CK(cudaStreamWaitEvent(STREAM, pushed[k], 0));
        mg::fence();                                  // (every rank's product rows here)
        mark();
        if (rows > 0 && !rounds)
          triangleOutRun<CC, WO, B16>((const B16*)pS.local, W(WNAME(pre, "centerNormScale")), W(WNAME(pre, "centerNormOffset")),
                                      Wh(WNAME(pre, "outputProjection")), t2, into(base), n, np, nullptr, mineRm);
        for (int r0 = 0; rounds && r0 < rows; r0 += gateRows) {      // (rows are independent: a chunk's gate reads its
          const int cr = std::min(gateRows, rows - r0);                //  own rows, unwritten until its output)
          RectMap rm{lo + r0, np, 0, (size_t)cr * np};
          RectMap gm = rm; gm.ab0 = gm.ab1 = 0;                        // (the gating linear alone)
          if (!attr) { smemAttr(kern, (int)wideTriInSmemW(CC, WI)); attr = true; }
          kern<<<(unsigned)((rm.size + triInRowsOf(WI) - 1) / triInRowsOf(WI)), 32 * triInWarpsOf(WI), wideTriInSmemW(CC, WI), STREAM>>>(
            base, mask, lnS, lnO, wt, (TQ*)nullptr, (TQ*)nullptr, t2, n, np, cs, nullptr, gm);
          rm.stride = mineRm.size;
          triangleOutRun<CC, WO, B16>((const B16*)pS.local + (size_t)r0 * np, W(WNAME(pre, "centerNormScale")), W(WNAME(pre, "centerNormOffset")),
                                      Wh(WNAME(pre, "outputProjection")), t2, into(base), n, np, nullptr, rm);
        }
        if (tt) {
          mark(); CK(cudaEventSynchronize(te.back()));
          auto el = [&](size_t a, size_t b) { float m = 0; CK(cudaEventElapsedTime(&m, te[a], te[b])); return (double)m; };
          acc[0] += el(0, 1); acc[1] += el(1, 2); acc[2] += el(2, 3); acc[3] += el(3, 4);
          for (auto e : te) CK(cudaEventDestroy(e));
          if (++calls % 96 == 0 && mg::RANK == 0)
            fprintf(stderr, "  tri %d calls: input pass %.1f, exchange + GEMMs %.1f, product tail + fence %.1f, output %.1f ms\n", calls,
                    acc[0], acc[1], acc[2], acc[3]);
        }
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
// With `from` (column attention: z is the z^T slab, `from` the pair z), the transposes are folded in: each rank pushes
// its rows of `from` into every rank's z^T, then each chunk of rows, once attended and through the pair transition
// `transitionPre` (pointwise, so on z^T as well as z), is pushed back into `from`'s rows on a side stream while the next
// chunk attends. (Where the input pass does not fit whole: the same pushes, unoverlapped, so every rank meets the same
// fences.)
inline void rowAttention(const mg::Shared& z, const float* mask, int n, int C, int heads, int D, const std::string& pre,
                         bool swap, const mg::Shared* from = nullptr, const std::string& transitionPre = "",
                         int transitionFactor = 0) {
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
  const bool folded = from && whole;
  // (a chunk a quarter-million pairs or more, up to four: at 988 tokens on 8 a rank's 124 rows in four chunks left
  // the attention and the transition a quarter-filled card, +14 ms of a 218 ms simulated pass; at 2,964 four cost nothing)
  const int chunks = (int)std::clamp<size_t>(pairsHere / 262144, 1, 4);
  static cudaStream_t pushS = nullptr;
  static std::vector<cudaEvent_t> evs;
  auto ev = [&](size_t k) {
    while (evs.size() <= k) { cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); evs.push_back(e); }
    return evs[k];
  };
  const int vecs = (int)((size_t)C * elem() / 16);
  // z^T pushed a chunk of every rank's columns at a time, a fence each (the count alike on every rank: off the largest
  // share), chunk k + 1's pushes on a side stream beside chunk k's input pass where the form is folded
  static cudaStream_t xS = nullptr;
  static cudaEvent_t xFenced, xFilled;
  if (from && !xS) {
    CK(cudaStreamCreateWithFlags(&xS, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&xFenced, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&xFilled, cudaEventDisableTiming));
  }
  const int XC = (int)std::clamp<size_t>((size_t)maxStored(n) * n / 262144, 1, 4), Pmax = (maxStored(n) + XC - 1) / XC;
  auto zTChunk = [&](int k, cudaStream_t st) {
    if (rowsHere) pushTransposeChunkK<<<st != STREAM ? sideBlocks((size_t)mg::WORLD * Pmax * rowsHere * vecs)
                                                     : blocks((size_t)mg::WORLD * Pmax * rowsHere * vecs), 256, 0, st>>>(
      peerSlabs(z, n), (const uint4*)from->local, n, lo, rowsHere, vecs, k, XC, Pmax, mg::RANK);
    if (rowsHere) mg::sent("transposes", (size_t)(mg::WORLD - 1) * Pmax * rowsHere * vecs * 16, st != STREAM);
  };
  if (from) mg::fence();                          // (every rank's z written; its z^T free)
  if (from && !folded)       // (the same fences as the folded form: a rank's choice is its own)
    for (int k = 0; k < XC; ++k) { zTChunk(k, STREAM); mg::fence(); }
  if (folded) {
    if (!pushS) CK(cudaStreamCreateWithFlags(&pushS, cudaStreamNonBlocking));
    zTChunk(0, STREAM);
    mg::fence();                                  // (every rank's columns of chunk 0 here)
    const int P = (rowsHere + XC - 1) / XC;
    for (int k = 0; k < XC; ++k) {
      if (k + 1 < XC) { CK(cudaEventRecord(xFenced, STREAM)); CK(cudaStreamWaitEvent(xS, xFenced, 0)); zTChunk(k + 1, xS); CK(cudaEventRecord(xFilled, xS)); }
      const int j0 = k * P, cols = std::min(P, rowsHere - j0);
      if (cols > 0)
        gridIn128(base, pre, qkvg, qkvgAll + (size_t)j0 * n * 4 * Wd, n, (size_t)(lo + j0) * n, (size_t)cols * n, false, Wh(wb),
                  bias, heads, stride, swap);
      if (k + 1 < XC) { CK(cudaStreamWaitEvent(STREAM, xFilled, 0)); mg::fence(); }
    }
  } else if (whole) gridIn128(base, pre, qkvg, qkvgAll, n, (size_t)lo * n, pairsHere, false, Wh(wb), bias, heads, stride, swap);
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
  mg::fence();                                    // (every rank's own written and zeroed)
  const PeerPlanes bp = peerPlanes(biasS, nullptr, n);
  const size_t bw = (size_t)heads * std::max((size_t)bp.maxRows * (stride / 8), (size_t)n * (bp.maxRows / 8 + 2));   // (one destination's)
  if (mg::WORLD > 1) mg::sent("attention bias", bw * (mg::WORLD - 1) * 16, false);
  if (mg::WORLD > 1) pushBiasK<<<blocks(bw * (mg::WORLD - 1)), 256, 0, STREAM>>>(bp, (const uint4*)bias, heads, n,
                                                                        stride, swap, mg::RANK);
  mg::fence();
  // this rank's rows, in chunks
  size_t R = std::max<size_t>(1, std::min<size_t>(std::max(rowsHere, 1), CHUNK / ((size_t)n * 4 * Wd)));
  {
    size_t f, t; deviceMemInfo(&f, &t);
    size_t perRow = (size_t)n * 5 * Wd * 2, spare = f > t / 16 ? f - t / 16 : 0;
    R = std::max<size_t>(R, std::min<size_t>(rowsHere, std::min<size_t>(spare / perRow, 256)));
  }
  if (folded) R = std::min<size_t>(R, (rowsHere + chunks - 1) / chunks);    // (chunks enough that most pushes hide)
  const PeerSlabs back = folded ? peerSlabs(*from, n) : PeerSlabs{};   // (lo: rows of z, as the pulls')
  size_t pushes = 0;
  for (size_t r0 = lo; r0 < (size_t)(lo + rowsHere); r0 += R) {
    size_t rows = std::min(R, (size_t)(lo + rowsHere) - r0), prs = rows * n;
    half* qkvgOut = whole ? qkvgAll + (r0 - lo) * n * 4 * Wd : scratch<half>("grid.qkvg", (std::min<size_t>(R, rowsHere) * n + 128) * 4 * Wd);
    uint8_t* kv8 = f8 ? scratch<uint8_t>("grid.kv8", std::min<size_t>(R, rowsHere) * n * 2 * Wd) : nullptr;
    if (!whole) gridIn128(base, pre, qkvg, qkvgOut, n, r0 * n, prs, false, nullptr, nullptr, 0, 0, false, kv8);
    half* gathered = scratch<half>("grid.gathered", std::min<size_t>(R, rowsHere) * n * Wd);
    if (f8) flash8Run(qkvgOut, kv8, kv8 + prs * Wd, bias, stride, gathered, n, heads, rows, scale, nullptr);
    else flashGrid<half>(qkvgOut, bias, stride, MASK_ALL_ONES ? nullptr : mask, gathered, n, heads, D, r0, rows, false, scale);
    gridOut128(gathered, pre + ".outputProjection", into(base), n, r0 * n, prs, false);
    if (folded) {
      transition<half>(pairRow((float*)z.local, (r0 - lo) * n, C), prs, C, transitionFactor, transitionPre);
      cudaEvent_t e = ev(64 + pushes++);
      CK(cudaEventRecord(e, STREAM)); CK(cudaStreamWaitEvent(pushS, e, 0));
      pushTransposeK<<<sideBlocks((size_t)n * rows * vecs), 256, 0, pushS>>>(back, (const uint4*)z.local + (r0 - lo) * n * vecs, n,
                                                                       (int)r0, (int)rows, vecs);
      mg::sent("transposes", (size_t)n * rows * vecs * 16 * (mg::WORLD - 1) / mg::WORLD, true);
    }
  }
  if (folded) {
    CK(cudaEventRecord(ev(63), pushS)); CK(cudaStreamWaitEvent(STREAM, ev(63), 0));
    mg::fence();                                  // (every rank's columns home)
  } else if (from) {
    if (rowsHere) {
      transition<half>((float*)z.local, (size_t)rowsHere * n, C, transitionFactor, transitionPre);
      pushTransposeK<<<blocks((size_t)n * rowsHere * vecs), 256, 0, STREAM>>>(peerSlabs(*from, n), (const uint4*)z.local, n, lo, rowsHere, vecs);
      mg::sent("transposes", (size_t)n * rowsHere * vecs * 16 * (mg::WORLD - 1) / mg::WORLD, false);
    }
    mg::fence();
  }
  releaseIfShort(n, C, { "grid.qkvg", "grid.gathered", "grid.kv8", "attn.vt8" });
}


// COLUMN attention on a card short of room, with no z^T share: the bias straight off this rank's rows of z (a pair's
// bias is its own LayerNorm and projection, so z[k][j] gives z^T[j][k]'s - written at the transposed place), then
// z^T a chunk of every rank's columns at a time into a ring of two chunk-sized slots, each chunk attended, through the
// pair transition and pushed home before the next - a quarter of a share held where z^T was a whole one (9 GB a rank at
// 11,856 tokens on 4)
inline void columnAttentionRing(const mg::Shared& z, const float* mask, int n, int C, int heads, int D, const std::string& pre,
                                bool swap, const std::string& transitionPre, int transitionFactor) {
  const int Wd = heads * D;
  int lo, hi; rowsOf(n, mg::RANK, lo, hi);
  const int rowsHere = storedRows(n, mg::RANK);
  const int stride = (n + 7) / 8 * 8;
  mg::Shared& biasS = mg::shared("sh.bias" + std::to_string(heads), (size_t)heads * n * stride * 2);
  half* bias = (half*)biasS.local;
  std::string qkvg = qkvgWeight(pre, C, Wd, true);
  std::string wb = paddedColumns(pre + ".pairBiasProjection", C, heads, 16);
  const float scale = 1.f / sqrtf((float)D);
  CK(cudaMemsetAsync(bias, 0, (size_t)heads * n * stride * 2, STREAM));
  const size_t pairsHere = (size_t)rowsHere * n;
  const bool f8 = (MASK_ALL_ONES || !mask) && fp8Attn();
  {
    size_t per = std::max<size_t>(1, std::min(pairsHere, CHUNK / 16));
    float* raw = scratch<float>("grid.raw16", per * 16);
    for (size_t q0 = 0; q0 < pairsHere; q0 += per) {
      size_t r = std::min(per, pairsHere - q0);
      lnHeads128<16>(pairRow((float*)z.local, q0, C), pre + ".actNormScale", pre + ".actNormOffset", wb, raw, r);
      biasFromRawRowsK<half><<<blocks((size_t)heads * r), 256, 0, STREAM>>>(raw, bias, (size_t)lo * n + q0, r, n, stride,
                                                                          heads, !swap, LOG2E);
    }
  }
  mg::fence();                                    // (every rank's own written and zeroed)
  const PeerPlanes bp = peerPlanes(biasS, nullptr, n);
  const size_t bw = (size_t)heads * std::max((size_t)bp.maxRows * (stride / 8), (size_t)n * (bp.maxRows / 8 + 2));
  if (mg::WORLD > 1) mg::sent("attention bias", bw * (mg::WORLD - 1) * 16, false);
  if (mg::WORLD > 1) pushBiasK<<<blocks(bw * (mg::WORLD - 1)), 256, 0, STREAM>>>(bp, (const uint4*)bias, heads, n, stride, !swap,
                                                                        mg::RANK);
  const int vecs = (int)((size_t)C * elem() / 16);
  const int XC = 8, Pmax = (maxStored(n) + XC - 1) / XC, P = (rowsHere + XC - 1) / XC;
  mg::Shared* slot[2] = { &mg::shared("sh.zTc0", std::max<size_t>((size_t)Pmax * n * C * elem(), 1)),
                          &mg::shared("sh.zTc1", std::max<size_t>((size_t)Pmax * n * C * elem(), 1)) };
  const PeerSlabs back = peerSlabs(z, n);
  size_t R = std::max<size_t>(1, std::min<size_t>(std::max(P, 1), CHUNK / ((size_t)n * 4 * Wd)));
  for (int k = 0; k < XC; ++k) {
    const mg::Shared& sl = *slot[k % 2];
    mg::fence();     // (k = 0: every rank's bias pushed; then every rank done with this slot's last chunk, k - 2)
    if (rowsHere) pushTransposeChunkK<<<blocks((size_t)mg::WORLD * Pmax * rowsHere * vecs), 256, 0, STREAM>>>(
      peerSlabs(sl, n), (const uint4*)z.local, n, lo, rowsHere, vecs, k, XC, Pmax, mg::RANK, true);
    if (rowsHere) mg::sent("transposes", (size_t)(mg::WORLD - 1) * Pmax * rowsHere * vecs * 16, false);
    mg::fence();                                  // (every rank's columns of chunk k in this rank's slot)
    const int j0 = k * P, cols = std::min(P, rowsHere - j0);
    float* sbase = baseOf(sl.local, lo + j0, n, C);  // (indexed by the global pair, z^T's rows)
    for (int c0 = 0; c0 < cols; c0 += (int)R) {
      const size_t r0 = (size_t)(lo + j0 + c0), rows = std::min<size_t>(R, cols - c0), prs = rows * n;
      half* qkvgOut = scratch<half>("grid.qkvg", (R * n + 128) * 4 * Wd);
      uint8_t* kv8 = f8 ? scratch<uint8_t>("grid.kv8", R * n * 2 * Wd) : nullptr;
      gridIn128(sbase, pre, qkvg, qkvgOut, n, r0 * n, prs, false, nullptr, nullptr, 0, 0, false, kv8);
      half* gathered = scratch<half>("grid.gathered", R * n * Wd);
      if (f8) flash8Run(qkvgOut, kv8, kv8 + prs * Wd, bias, stride, gathered, n, heads, rows, scale, nullptr);
      else flashGrid<half>(qkvgOut, bias, stride, MASK_ALL_ONES ? nullptr : mask, gathered, n, heads, D, r0, rows, false, scale);
      gridOut128(gathered, pre + ".outputProjection", into(sbase), n, r0 * n, prs, false);
      transition<half>(pairRow((float*)sl.local, (size_t)c0 * n, C), prs, C, transitionFactor, transitionPre);
      pushTransposeK<<<blocks((size_t)n * rows * vecs), 256, 0, STREAM>>>(back, (const uint4*)sl.local + (size_t)c0 * n * vecs, n,
                                                                        (int)r0, (int)rows, vecs);
      mg::sent("transposes", (size_t)n * rows * vecs * 16 * (mg::WORLD - 1) / mg::WORLD, false);
    }
  }
  mg::fence();                                    // (every rank's columns home)
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
  // (a ring, as triangleOn: one rank's b staged at a time against each block of a - on a card short of room a window of
  // an eighth of its channels at a time: a whole share staged was 4.5 GB of a simulated rank of 4's out-of-memory at
  // 11,856 tokens, in the template stack)
  const int cw = shortPair((size_t)n * n, C) ? std::max(1, C / 8) : C;
  const size_t stageElems = (size_t)maxShare * np * cw;
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
    for (int c0 = 0; c0 < C; c0 += cw)
      ringOver<T>(bmine, n, np, c0, std::min(cw, C - c0), stages, nstage, [&](int m, const T* B, int qlo) {
        CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, m, w, np, &alpha, B, cudaType<T>(), np, (long long)m * np,
          a + (size_t)c0 * r.size(), cudaType<T>(), np, r.size(), &zero, prod + (size_t)c0 * r.size() + qlo, CUDA_R_32F, np, r.size(),
          std::min(cw, C - c0), CUBLAS_COMPUTE_32F, algo));
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
  for (int k = 1; k < mg::WORLD; ++k) {          // (from the next rank on: rot)
    const int r = (mg::RANK + k) % mg::WORLD;
    int rlo, rhi; rowsOf(n, r, rlo, rhi); rhi = std::min(rhi, n);
    if (rhi <= rlo) continue;
    for (int h = 0; h < heads; ++h) {
      const size_t plane = (size_t)h * n * stride * e;
      mg::sent("attention bias", (size_t)(rhi - rlo) * stride * e, false);
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
// A pairformer block's pair updates on the sharded pair z (this rank's slab); zT a slab-sized buffer for z^T, made
// (shared, as zTName) the first time a form here wants it - the fused forms on a card short of room never do
template <class T>
inline void pairUpdates(const mg::Shared& z, mg::Shared*& zTp, const std::string& zTName, const float* mask, int n, int C,
                        const std::string& pre, bool swap, bool divide, int transitionFactor) {
  auto zTof = [&]() -> const mg::Shared& { if (!zTp) zTp = &mg::shared(zTName, z.bytes); return *zTp; };
  // the fused forms where they apply (the 128/256-channel streaming triangle on a bf16 pair, the 128-channel 4 x 32
  // grid attention without biases), the generic ones otherwise
  auto tri = [&](const mg::Shared& s, const std::string& p) {
    if constexpr (std::is_same_v<T, half>) {
      if (PAIR16 && (C == 128 || C == 256) && blockedFused(n, C)) { triangleSharded<T>(s, mask, n, C, p, divide); return; }
    }
    triangleGenericOn<T>(s, mask, n, C, p, divide);
  };
  int heads = (int)M.meta(pre + ".pairAttention1.heads"), D = (int)M.meta(pre + ".pairAttention1.dimension");
  auto fusedAtt = [&](const std::string& p) {
    if constexpr (std::is_same_v<T, half>)
      return FUSED_GRID && !skipFused("grid") && gridFusedFits() && !RESIDUAL_UNTRANSPOSED && C == 128 && heads * D == 128 &&
             D == 32 && !hasW(p + ".gatingQueryBias") && !hasW(p + ".outputProjectionBias");
    return false;
  };
  auto att = [&](const mg::Shared& s, const std::string& p, bool sw) {
    if constexpr (std::is_same_v<T, half>) if (fusedAtt(p)) { rowAttention(s, mask, n, C, heads, D, p, sw); return; }
    rowAttentionGeneric<T>(s, mask, n, C, heads, D, p, sw);
  };
  tri(z, pre + ".triangleMultiplicationOutgoing"); stage("tri.out");
  // the incoming form straight on z where the fused bf16 triangle runs (no transposes); else as the outgoing form on z^T
  bool direct = false;
  if constexpr (std::is_same_v<T, half>) direct = PAIR16 && (C == 128 || C == 256) && blockedFused(n, C) && !fp8Tensor();
  if (direct) triangleSharded<T>(z, mask, n, C, pre + ".triangleMultiplicationIncoming", divide, true);
  else {
    const mg::Shared& zT = zTof();
    transpose(z, zT, n, C);
    tri(zT, pre + ".triangleMultiplicationIncoming");
    transpose(zT, z, n, C);
  }
  stage("tri.in");
  att(z, pre + ".pairAttention1", false); stage("grid.row");
  // (the transposes and the transition inside: rowAttention's `from`; LOCALFOLD_MG_FOLD_TRANSPOSES=0 the plain
  // transposes, to measure what the overlap is worth over a real interconnect)
  static const bool FOLD = !getenv("LOCALFOLD_MG_FOLD_TRANSPOSES") || atoi(getenv("LOCALFOLD_MG_FOLD_TRANSPOSES"));
  if (fusedAtt(pre + ".pairAttention2") && shortPair((size_t)n * n, C)) {
    columnAttentionRing(z, mask, n, C, heads, D, pre + ".pairAttention2", !swap, pre + ".pairTransition", transitionFactor);
    stage("grid.col+transition");
    return;
  }
  const mg::Shared& zT = zTof();
  if (FOLD && fusedAtt(pre + ".pairAttention2")) {
    rowAttention(zT, mask, n, C, heads, D, pre + ".pairAttention2", !swap, &z, pre + ".pairTransition", transitionFactor);
    stage("grid.col+transition");
    return;
  }
  transpose(z, zT, n, C);
  att(zT, pre + ".pairAttention2", !swap); stage("grid.col");
  transpose(zT, z, n, C);
  const int rowsHere = storedRows(n, mg::RANK);
  if (rowsHere) transition<T>((float*)z.local, (size_t)rowsHere * n, C, transitionFactor, pre + ".pairTransition");
  stage("transition");
}

}  // namespace sh
