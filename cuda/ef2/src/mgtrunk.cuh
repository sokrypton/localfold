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
  float* mask = scratch<float>("trunk.mask", P);
  fillK<<<blocks(P), 256, 0, STREAM>>>(mask, 1.f, P);
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
      zInitAddRowsK<<<blocks(cells * C), 256, 0, STREAM>>>(base, zs.rows, zs.cols, relIdx(), W("token_bonds"),
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

// this rank's share in its own pinned host buffer through the sampler (which never reads it), and back
inline B16Host EF2_SHARE_HOST{};
inline void parkShareEf2(int T, int C) {
  const size_t bytes = (size_t)sh::storedRows(T, mg::RANK) * T * C * 2;
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
}
inline void unparkShareEf2(int T, int C) {
  releaseScratch(); trimPool();
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
