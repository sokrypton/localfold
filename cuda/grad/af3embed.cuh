// The AF3 lineage's input embedder and distogram head for training: target_feat (its atom encoder, af3denoise.cuh's
// DiffEncoder under targetFeat.encoder / targetFeat.reference, and the concatenation), z_init and s_init (the left and
// right single projections' outer sum, the recycled pair and single through their LayerNorms and projections, the
// relative encoding, the bond terms) and the MSA's embedding; the distogram head. cuda/af3/src/atom.cuh's
// buildTargetFeat() and trunk.cuh's embed() (up to the template term) and distogram() are what the forward is held to.
//
// The generic path: AF3, boltz2's bond-type table, OpenDDE's pair from s_init (pairInitFromSingle), the paired-query
// MSA column. Not here: chai-1's token embedding and relative encoding, boltz2's atom-only target_feat sum, MC dropout,
// constraint tables. The recycled pair and single are inputs whose gradients are returned (a caller that stops them,
// as AF3's training does between cycles, drops them).
#pragma once
#include "af3denoise.cuh"

namespace grad {
namespace af3 {

// [aatype one-hot 31 | profile 31 | deletion mean | atom features P]
__global__ void targetFeatCatK(const int* aatype, const real* profile, const real* delMean, const real* atom, real* out,
                               int n, int P) {
  const int W = 63 + P;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * W) return;
  int i = t / W, c = t % W;
  out[t] = c < 31 ? (aatype[i] == c ? (real)1 : 0) : c < 62 ? profile[i * 31 + c - 31] : c == 62 ? delMean[i] : atom[(size_t)i * P + c - 63];
}
__global__ void targetFeatAtomBwdK(const real* d, real* datom, int n, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * P) return;
  datom[t] = d[(t / P) * (63 + P) + 63 + t % P];
}
struct TargetFeat { DiffEncoder enc; };
struct TargetFeatSave { EncSave es; real* atom; };
inline TargetFeatSave targetFeatForward(TargetFeat& m, const AtomBatch& b, const int* aatype, const real* profile,
                                        const real* delMean, real* out) {
  TargetFeatSave s{};
  const int n = b.tokens, P = m.enc.perToken;
  s.atom = ARENA.get((size_t)n * P);
  s.es = encForward(m.enc, b, nullptr, nullptr, nullptr, nullptr, s.atom);
  targetFeatCatK<<<blocks((size_t)n * (63 + P)), 256, 0, STREAM>>>(aatype, profile, delMean, s.atom, out, n, P);
  return s;
}
inline void targetFeatBackward(TargetFeat& m, const AtomBatch& b, const TargetFeatSave& s, const real* dout) {
  Scratch scratch;
  const int n = b.tokens, P = m.enc.perToken, C = m.enc.C, Cp = m.enc.Cp;
  const size_t Rq = (size_t)b.subsets * b.Q, rows = Rq * b.K;
  real* datom = ARENA.get((size_t)n * P);
  targetFeatAtomBwdK<<<blocks((size_t)n * P), 256, 0, STREAM>>>(dout, datom, n, P);
  encBackward(m.enc, b, s.es, nullptr, nullptr, datom, ARENA.zeros(Rq * C), ARENA.zeros(rows * Cp), ARENA.zeros(Rq * C),
              nullptr, nullptr, nullptr);
}

// ---------------------------------------------------------------- z_init, s_init, the MSA
struct TrunkEmbedder {
  int C = 0, Cs = 0, Cm = 0, F = 0, msaWidth = 34, rel = 139;
  bool fromSingle = false, hasBond = false, hasBondType = false, pairedQuery = false;
  Linear left, right, single, prevPair, prevSingle, msaFromTarget, relpos;
  LayerNorm prevPairLn, prevSingleLn;
  Param msaW, bondW, bondTypeTable, unspecified;
};
inline void forEachParam(TrunkEmbedder& e, const std::function<bool(const std::string&)>& has,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string E = "trunk.embedder.";
  auto ln = [&](const std::string& n, LayerNorm& l, int C) {
    l.C = C; f(n + "Scale", l.g, C); if (has(n + "Offset")) f(n + "Offset", l.b, C);
  };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; f(n, l.w, (size_t)K * N); };
  const int src = e.fromSingle ? e.Cs : e.F;
  lin(E + "singleActivations", e.single, e.F, e.Cs);
  lin(E + "leftSingle", e.left, src, e.C); lin(E + "rightSingle", e.right, src, e.C);
  ln(E + "prevEmbeddingNorm", e.prevPairLn, e.C); lin(E + "prevEmbedding", e.prevPair, e.C, e.C);
  lin(E + "positionActivations", e.relpos, e.rel, e.C);
  e.hasBond = has(E + "bondEmbedding"); if (e.hasBond) f(E + "bondEmbedding", e.bondW, e.C);
  e.hasBondType = has(E + "tokenBondsTypeEmbed");
  if (e.hasBondType) { f(E + "tokenBondsTypeEmbed", e.bondTypeTable, (size_t)7 * e.C); f(E + "contactEncodingUnspecified", e.unspecified, e.C); }
  ln(E + "prevSingleEmbeddingNorm", e.prevSingleLn, e.Cs); lin(E + "prevSingleEmbedding", e.prevSingle, e.Cs, e.Cs);
  lin(E + "extraMsaTargetFeat", e.msaFromTarget, e.F, e.Cm);
  f(E + "msaActivations", e.msaW, (size_t)e.msaWidth * e.Cm);
}
struct EmbedBatch {
  const int *ri, *ti, *asym, *ent, *sym;      // [n]
  const real *bonds, *bondOrders;              // [n, n] (either may be null)
  const int* msaRows; const real* deletion;    // [S, n]
  int n, S;
};
// pair = left[i] + right[j] (+ the bond terms)
__global__ void outerSumBondK(const real* l, const real* r, const real* bonds, const real* bw, const real* orders,
                              const real* table, const real* unspec, real* pair, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / n, j = ij % n;
  real v = l[(size_t)i * C + c] + r[(size_t)j * C + c];
  if (bw) v += bonds[ij] * bw[c];
  if (table) { int o = orders ? (int)orders[ij] : 0; if (o < 0 || o >= 7) o = 0; v += table[o * C + c] + unspec[c]; }
  pair[t] = v;
}
// d pair -> dleft, dright (written), the bond weights (added); a thread a (token, channel)
__global__ void outerSumBondBwdK(const real* d, const real* bonds, const real* orders, real* dl, real* dr, real* dbw,
                                 real* dtable, real* dunspec, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int c = t % C, i = t / C;
  real sl = 0, sr = 0, sb = 0;
  for (int j = 0; j < n; ++j) {
    real a = d[((size_t)i * n + j) * C + c];
    sl += a; sr += d[((size_t)j * n + i) * C + c];
    if (dbw) sb += a * bonds[(size_t)i * n + j];
    if (dtable) { int o = orders ? (int)orders[(size_t)i * n + j] : 0; if (o < 0 || o >= 7) o = 0; atomicAdd(&dtable[o * C + c], a); }
  }
  dl[t] = sl; dr[t] = sr;
  if (dbw) atomicAdd(&dbw[c], sb);
  if (dunspec) atomicAdd(&dunspec[c], sl);
}
// msa = W[code] + clip(d, 0, 1) W[32] + atan(d / 3) 2/pi W[33] (+ W[34] on the query row) + fromTarget[token]
__global__ void msaEmbedK(const int* rows, const real* del, const real* W, const real* ft, real* msa, size_t count, int n,
                          int C, int width, bool pairedQuery) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count * C) return;
  int c = t % C; size_t row = t / C; int tok = row % n;
  int code = rows[row]; real d = del[row];
  real v = (code >= 0 && code < 32) ? W[(size_t)code * C + c] : 0;
  v += fmin(fmax(d, (real)0), (real)1) * W[32 * C + c] + atan(d / 3) * (real)(2 / 3.14159265358979) * W[33 * C + c];
  if (width > 34 && pairedQuery && row < (size_t)n) v += W[34 * C + c];
  msa[t] = v + ft[(size_t)tok * C + c];
}
// by (token, channel): dW rows, and d fromTarget summed over the rows
__global__ void msaEmbedBwdK(const int* rows, const real* del, const real* d, real* dW, real* dft, size_t count, int n,
                             int C, int width, bool pairedQuery) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int c = t % C, tok = t / C;
  real s = 0;
  for (size_t row = tok; row < count; row += n) {
    real g = d[row * C + c]; s += g;
    int code = rows[row]; real dl = del[row];
    if (code >= 0 && code < 32) atomicAdd(&dW[(size_t)code * C + c], g);
    atomicAdd(&dW[32 * C + c], fmin(fmax(dl, (real)0), (real)1) * g);
    atomicAdd(&dW[33 * C + c], atan(dl / 3) * (real)(2 / 3.14159265358979) * g);
    if (width > 34 && pairedQuery && row < (size_t)n) atomicAdd(&dW[34 * C + c], g);
  }
  dft[t] = s;
}
struct EmbedSave { real *src, *sInit, *l, *r, *ppn, *rel, *psn, *ft; LnSave lpp, lps; };
// targetFeat [n, F], prevPair [n*n, C], prevSingle [n, Cs] -> pair (z_init, before the template term), single, msa
inline EmbedSave trunkEmbedForward(TrunkEmbedder& e, const EmbedBatch& b, const real* tf, const real* prevPair,
                                   const real* prevSingle, real* pair, real* single, real* msa) {
  EmbedSave s{};
  const int n = b.n; const size_t nn = (size_t)n * n;
  s.src = const_cast<real*>(tf);
  if (e.fromSingle) { s.sInit = ARENA.get((size_t)n * e.Cs); e.single.forward(tf, s.sInit, n); s.src = s.sInit; }
  s.l = ARENA.get((size_t)n * e.C); e.left.forward(s.src, s.l, n);
  s.r = ARENA.get((size_t)n * e.C); e.right.forward(s.src, s.r, n);
  // (the bond term only where the batch has a bond matrix: a plain polymer has none, as cuda/af3's embed() asks)
  outerSumBondK<<<blocks(nn * e.C), 256, 0, STREAM>>>(s.l, s.r, b.bonds, e.hasBond && b.bonds ? e.bondW.w : nullptr, b.bondOrders,
      e.hasBondType ? e.bondTypeTable.w : nullptr, e.hasBondType ? e.unspecified.w : nullptr, pair, n, e.C);
  s.ppn = ARENA.get(nn * e.C); s.lpp = lnForward(e.prevPairLn, prevPair, s.ppn, nn);
  e.prevPair.forward(s.ppn, pair, nn, 1);
  s.rel = ARENA.get(nn * e.rel);
  relposK<<<blocks(nn * e.rel), 256, 0, STREAM>>>(b.ri, b.ti, b.asym, b.ent, b.sym, s.rel, n);
  e.relpos.forward(s.rel, pair, nn, 1);
  // single
  e.single.forward(tf, single, n);
  s.psn = ARENA.get((size_t)n * e.Cs); s.lps = lnForward(e.prevSingleLn, prevSingle, s.psn, n);
  e.prevSingle.forward(s.psn, single, n, 1);
  // msa
  s.ft = ARENA.get((size_t)n * e.Cm); e.msaFromTarget.forward(tf, s.ft, n);
  msaEmbedK<<<blocks((size_t)b.S * n * e.Cm), 256, 0, STREAM>>>(b.msaRows, b.deletion, e.msaW.w, s.ft, msa, (size_t)b.S * n, n,
                                                                e.Cm, e.msaWidth, e.pairedQuery);
  return s;
}
// dpair, dsingle, dmsa -> every parameter; dtf, dprevPair, dprevSingle (added; any may be null)
inline void trunkEmbedBackward(TrunkEmbedder& e, const EmbedBatch& b, const EmbedSave& s, const real* tf, const real* dpair,
                               const real* dsingle, const real* dmsa, real* dtf, real* dprevPair, real* dprevSingle) {
  Scratch scratch;
  const int n = b.n; const size_t nn = (size_t)n * n;
  real* dtfl = ARENA.zeros((size_t)n * e.F);
  if (dmsa) {
    real* dft = ARENA.get((size_t)n * e.Cm);
    msaEmbedBwdK<<<blocks((size_t)n * e.Cm), 256, 0, STREAM>>>(b.msaRows, b.deletion, dmsa, e.msaW.g, dft, (size_t)b.S * n, n,
                                                                e.Cm, e.msaWidth, e.pairedQuery);
    e.msaFromTarget.backward(tf, dft, dtfl, n, true);
  }
  real* dsInit = e.fromSingle ? ARENA.zeros((size_t)n * e.Cs) : nullptr;
  if (dsingle) {
    real* dpsn = ARENA.get((size_t)n * e.Cs); e.prevSingle.backward(s.psn, dsingle, dpsn, n);
    if (dprevSingle) lnBackward(e.prevSingleLn, s.lps, dpsn, dprevSingle, n, true);
    else { real* t = ARENA.get((size_t)n * e.Cs); lnBackward(e.prevSingleLn, s.lps, dpsn, t, n, false); }
    e.single.backward(tf, dsingle, dtfl, n, true);
  }
  if (dpair) {
    e.relpos.backward(s.rel, dpair, nullptr, nn);
    real* dppn = ARENA.get(nn * e.C); e.prevPair.backward(s.ppn, dpair, dppn, nn);
    if (dprevPair) lnBackward(e.prevPairLn, s.lpp, dppn, dprevPair, nn, true);
    else { real* t = ARENA.get(nn * e.C); lnBackward(e.prevPairLn, s.lpp, dppn, t, nn, false); }
    real *dl = ARENA.get((size_t)n * e.C), *dr = ARENA.get((size_t)n * e.C);
    outerSumBondBwdK<<<blocks((size_t)n * e.C), 256, 0, STREAM>>>(dpair, b.bonds, b.bondOrders, dl, dr, e.hasBond && b.bonds ? e.bondW.g : nullptr,
        e.hasBondType ? e.bondTypeTable.g : nullptr, e.hasBondType ? e.unspecified.g : nullptr, n, e.C);
    real* dsrc = e.fromSingle ? dsInit : dtfl;
    e.left.backward(s.src, dl, dsrc, n, true);
    e.right.backward(s.src, dr, dsrc, n, true);
  }
  if (e.fromSingle) e.single.backward(tf, dsInit, dtfl, n, true);     // (s_init's own: the pair's source)
  if (dtf) add(dtf, dtfl, (size_t)n * e.F);
}

// ---------------------------------------------------------------- the distogram head
// logits = scale (h[i, j] + h[j, i]), h = pair W + b (scale 1/2 under mlpDistogram)
struct DistogramHead { Linear half; int bins = 64, C = 0; real scale = 1; };
inline void forEachParam(DistogramHead& d, const std::function<bool(const std::string&)>& has,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string D = "trunk.distogram.";
  d.half.K = d.C; d.half.N = d.bins; d.half.hasBias = has(D + "halfLogitsBias");
  f(D + "halfLogits", d.half.w, (size_t)d.C * d.bins);
  if (d.half.hasBias) f(D + "halfLogitsBias", d.half.b, d.bins);
}
__global__ void symScaleK(const real* h, real* out, int n, int B, real s) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * B) return;
  int c = t % B; size_t ij = t / B; int i = ij / n, j = ij % n;
  out[t] = s * (h[t] + h[((size_t)j * n + i) * B + c]);
}
inline real* distogramForward(DistogramHead& d, const real* pair, int n, real* logits) {
  const size_t nn = (size_t)n * n;
  real* h = ARENA.get(nn * d.bins); d.half.forward(pair, h, nn);
  symScaleK<<<blocks(nn * d.bins), 256, 0, STREAM>>>(h, logits, n, d.bins, d.scale);
  return h;
}
inline void distogramBackward(DistogramHead& d, const real* pair, int n, const real* dlogits, real* dpair) {
  Scratch scratch;
  const size_t nn = (size_t)n * n;
  real* dh = ARENA.get(nn * d.bins);
  symScaleK<<<blocks(nn * d.bins), 256, 0, STREAM>>>(dlogits, dh, n, d.bins, d.scale);
  d.half.backward(pair, dh, dpair, nn, true);
}

}  // namespace af3
}  // namespace grad
