// The AF3 lineage's whole denoiser for training: D(x, sigma) from the trunk's single and pair, target_feat and the
// noisy atoms, forward with saves and backward into every parameter and into the trunk's outputs - what a structure
// loss needs. cuda/af3/src/diffusion.cuh's denoise() is what the forward is held to; its pieces, in order:
//
//   conditioning  pair  = T2(T1(LN([trunk pair | relpos one-hot]) W))   (or the split / projected-relpos forms)
//                 single = T2(T1(LN([trunk single | target_feat]) W + LN(fourier(sigma)) Wn)) (+ a final norm)
//   encoder       per-atom conditioning (reference features) + the trunk single's projection, masked; the atom pair
//                 (row/col of relu(cond), reference offsets, the conditioning pair's projection), a 3-layer MLP; queries
//                 = cond (+ positions * in(sigma)) through the atom stack; aggregated to tokens: mean of relu(W act)
//   transformer   act = aggregated + W LN(single); the 24-block token transformer (af3diff.cuh); LN
//   decoder       broadcast W act to atoms + the encoder's skip; the atom stack; LN; W -> a position update
//   out           skip(sigma) x + out(sigma) update, masked; skip = 256 / (sigma^2 + 256), out = 16 sigma / sqrt(...)
//
// Not here: chai-1's conditioning and atom-pair forms (its own featurisation of both) - its atom stack and transformer
// are af3diff.cuh's. rf3's chirality term enters as a constant input feature (its finite-difference gradient of the
// positions is not differentiated; the term's weight is).
#pragma once
#include <optional>
#include "af3diff.cuh"

namespace grad {
namespace af3 {

// the relative position one-hot (maxIdx 32, maxChain 2: 139 columns), cuda/af3's pairFeaturesK
__global__ void relposK(const int* ri, const int* ti, const int* asym, const int* ent, const int* sym, real* out, int n) {
  const int maxIdx = 32, maxChain = 2, positionBins = 2 * maxIdx + 2, rel = positionBins * 2 + 1 + 2 * maxChain + 2;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * rel) return;
  int k = t % rel; size_t ij = t / rel; int i = ij / n, j = ij % n;
  auto clamp = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
  bool sameChain = asym[i] == asym[j], sameEntity = ent[i] == ent[j];
  int c0 = sameChain ? clamp(ri[i] - ri[j] + maxIdx, 2 * maxIdx) : 2 * maxIdx + 1;
  bool sameRes = sameChain && ri[i] == ri[j];
  int c1 = positionBins + (sameRes ? clamp(ti[i] - ti[j] + maxIdx, 2 * maxIdx) : 2 * maxIdx + 1);
  int c2 = positionBins * 2;
  int c3 = positionBins * 2 + 1 + (sameEntity ? clamp(sym[i] - sym[j] + maxChain, 2 * maxChain) : 2 * maxChain + 1);
  out[t] = (k == c0 || k == c1 || k == c3 || (k == c2 && sameEntity)) ? 1 : 0;
}
// out[r] = [a[r] | b[r]] with zero columns at p0, p1 (-1: none); back: split the gradient
__global__ void concat2K(const real* a, int wa, const real* b, int wb, real* out, size_t rows, int p0, int p1, real* da, real* db) {
  int w = wa + wb + (p0 >= 0) + (p1 >= 0);
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * w) return;
  size_t r = t / w; int c = t % w;
  bool pad = c == p0 || c == p1;
  int src = c - (p0 >= 0 && c > p0) - (p1 >= 0 && c > p1);
  if (!da) { out[t] = pad ? (real)0 : src < wa ? a[r * wa + src] : b[r * wb + (src - wa)]; return; }
  if (pad) return;
  if (src < wa) da[r * wa + src] += out[t];
  else if (db) db[r * wb + (src - wa)] += out[t];
}
// (trD: the level read off the device instead - a captured sampler step replayed at each level)
__global__ void fourierFwdK(const real* w, const real* b, real tr, real* out, int nc, const real* trD = nullptr) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (trD) tr = *trD;
  if (k < nc) out[k] = cos((real)6.283185307179586 * (tr * w[k] + b[k]));
}
__global__ void fourierBwdK(const real* w, const real* b, real tr, const real* d, real* dw, real* db, int nc) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nc) return;
  real s = -sin((real)6.283185307179586 * (tr * w[k] + b[k])) * (real)6.283185307179586 * d[k];
  dw[k] += s * tr; db[k] += s;
}
// x[r, c] += v[c] (forward); dv[c] += sum_r d[r, c] (backward is colSum)
__global__ void addRowVecK(real* x, const real* v, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] += v[t % C];
}

// ---------------------------------------------------------------- the conditioning
struct DiffCond {
  int Cz = 0, Cs = 0, Czt = 0, Cst = 0, F = 0, rel = 139, nc = 0;
  bool split = false, relpe = false, pad = false;
  LayerNorm pairLn, singleLn, noiseLn, zTrunkLn, singleFinal;
  bool hasSingleFinal = false;
  Linear pairProj, singleProj, noiseProj, relpeProj, zTrunkProj;
  GatedTransition pairT[2], singleT[2];
  Param fw, fb;
  // chai-1's (trunk.dialect.chaiDiffusionConditioning): its 163 structure token-pair columns through a biased projection
  // beside the trunk pair (and the declared bonds), the pair track closed by an affine LayerNorm; set chai before
  // forEachParam and the ranks and bonds before condForward
  bool chai = false, hasStructBond = false;
  Linear structW;
  Param structBond;
  LayerNorm pairFinal;
  const int *entRank = nullptr, *symRank = nullptr;     // [n]: the dense ranks of the entity and sym ids
  const real* declBonds = nullptr;                       // [n, n] (null: none)
  int width() const { return chai ? Czt + Cz : split ? 2 * Cz : relpe ? Czt + Cz : Czt + rel; }
  int swidth() const { return Cst + F + (pad ? 2 : 0); }
};
struct CondSave {
  real *rel, *relProj, *tn, *tproj, *f2, *f2n, *pair0, *pair1, *pairOut, *f1, *f1n, *base, *e, *en, *nproj, *s0, *s1, *sOut, *sFinal, *pairPre;
  LnSave lp, lz, ls, le, lf, lpf;
  GtSave pt0, pt1, st0, st1;
  real tr;
  int n;
};
// chai-1's structure token-pair columns (cuda/af3's chaiStructurePairK): docking (5: none), the relative chain (6..11),
// entity (12..14), residue (15..81) and token (82..148) separations, and the two masked restraints (155, 162)
__global__ void chaiStructOneHotK(const int* ri, const int* ti, const int* asym, const int* entRank, const int* symRank, real* out, int n) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)n * n) return;
  int i = ij / n, j = ij % n;
  auto clip = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
  bool same = asym[i] == asym[j];
  int rss = same ? clip(ri[i] - ri[j] + 33, 0, 65) : 66;
  int rts = same && ri[i] == ri[j] ? clip(ti[i] - ti[j] + 32, 0, 65) : 66;
  int relE = entRank[i] - entRank[j];
  int rchain = relE != 0 ? 5 : clip(symRank[i] - symRank[j] + 2, 0, 4);
  int rent = clip(relE + 1, 0, 2);
  real* o = out + ij * 163;
  for (int k = 0; k < 163; ++k) o[k] = 0;
  o[5] = o[155] = o[162] = 1; o[6 + rchain] = 1; o[12 + rent] = 1; o[15 + rss] = 1; o[82 + rts] = 1;
}
inline CondSave condForward(DiffCond& c, const real* trunkSingle, const real* trunkPair, const real* targetFeat,
                            const int* ri, const int* ti, const int* asym, const int* ent, const int* sym, real sigma, int n) {
  const size_t nn = (size_t)n * n; const int W = c.width(), SW = c.swidth();
  CondSave s{}; s.n = n;
  s.f2 = ARENA.get(nn * W);
  if (c.chai) {
    s.rel = ARENA.get(nn * 163);
    chaiStructOneHotK<<<blocks(nn), 256, 0, STREAM>>>(ri, ti, asym, c.entRank, c.symRank, s.rel, n);
    s.relProj = ARENA.get(nn * c.Cz); c.structW.forward(s.rel, s.relProj, nn);
    if (c.hasStructBond && c.declBonds) mm(c.declBonds, c.structBond.w, s.relProj, (int)nn, 1, c.Cz, 1);
    concat2K<<<blocks(nn * W), 256, 0, STREAM>>>(trunkPair, c.Czt, s.relProj, c.Cz, s.f2, nn, -1, -1, nullptr, nullptr);
  } else {
  s.rel = ARENA.get(nn * c.rel);
  relposK<<<blocks(nn * c.rel), 256, 0, STREAM>>>(ri, ti, asym, ent, sym, s.rel, n);
  if (c.split || c.relpe) {
    s.relProj = ARENA.get(nn * c.Cz); c.relpeProj.forward(s.rel, s.relProj, nn);
    const real* first = trunkPair; int fw = c.Czt;
    if (c.split) {
      s.tn = ARENA.get(nn * c.Czt); s.lz = lnForward(c.zTrunkLn, trunkPair, s.tn, nn);
      s.tproj = ARENA.get(nn * c.Cz); c.zTrunkProj.forward(s.tn, s.tproj, nn);
      first = s.tproj; fw = c.Cz;
    }
    concat2K<<<blocks(nn * W), 256, 0, STREAM>>>(first, fw, s.relProj, c.Cz, s.f2, nn, -1, -1, nullptr, nullptr);
  } else concat2K<<<blocks(nn * W), 256, 0, STREAM>>>(trunkPair, c.Czt, s.rel, c.rel, s.f2, nn, -1, -1, nullptr, nullptr);
  }
  s.f2n = ARENA.get(nn * W); s.lp = lnForward(c.pairLn, s.f2, s.f2n, nn);
  s.pair0 = ARENA.get(nn * c.Cz); c.pairProj.forward(s.f2n, s.pair0, nn);
  s.pair1 = cloneR(s.pair0, nn * c.Cz); s.pt0 = gtForward(c.pairT[0], s.pair0, s.pair1, nn, 1);
  s.pairOut = cloneR(s.pair1, nn * c.Cz); s.pt1 = gtForward(c.pairT[1], s.pair1, s.pairOut, nn, 1);
  if (c.chai) { s.pairPre = s.pairOut; s.pairOut = ARENA.get(nn * c.Cz); s.lpf = lnForward(c.pairFinal, s.pairPre, s.pairOut, nn); }
  s.f1 = ARENA.get((size_t)n * SW);
  concat2K<<<blocks((size_t)n * SW), 256, 0, STREAM>>>(trunkSingle, c.Cst, targetFeat, c.F, s.f1, n,
                                                       c.pad ? c.Cst + 31 : -1, c.pad ? c.Cst + 63 : -1, nullptr, nullptr);
  s.f1n = ARENA.get((size_t)n * SW); s.ls = lnForward(c.singleLn, s.f1, s.f1n, n);
  s.base = ARENA.get((size_t)n * c.Cs); c.singleProj.forward(s.f1n, s.base, n);
  s.tr = (real)0.25 * log(sigma / (real)16);
  s.e = ARENA.get(c.nc); fourierFwdK<<<blocks(c.nc), 256, 0, STREAM>>>(c.fw.w, c.fb.w, s.tr, s.e, c.nc);
  s.en = ARENA.get(c.nc); s.le = lnForward(c.noiseLn, s.e, s.en, 1);
  s.nproj = ARENA.get(c.Cs); c.noiseProj.forward(s.en, s.nproj, 1);
  s.s0 = cloneR(s.base, (size_t)n * c.Cs);
  addRowVecK<<<blocks((size_t)n * c.Cs), 256, 0, STREAM>>>(s.s0, s.nproj, n, c.Cs);
  s.s1 = cloneR(s.s0, (size_t)n * c.Cs); s.st0 = gtForward(c.singleT[0], s.s0, s.s1, n, 1);
  s.sOut = cloneR(s.s1, (size_t)n * c.Cs); s.st1 = gtForward(c.singleT[1], s.s1, s.sOut, n, 1);
  s.sFinal = s.sOut;
  if (c.hasSingleFinal) { s.sFinal = ARENA.get((size_t)n * c.Cs); s.lf = lnForward(c.singleFinal, s.sOut, s.sFinal, n); }
  return s;
}
// the conditioning at another noise level: `base`'s pair half and its noise-free single projection shared, the noise
// embedding and the single transitions recomputed for `sigma` (a batch of samples: the pair half computed once)
// (trD: the embedding's input read off the device - a sampler step's graph, s.tr then not this level's)
inline CondSave condSingleForward(DiffCond& c, const CondSave& base, real sigma, const real* trD = nullptr) {
  CondSave s = base;
  const int n = base.n;
  s.tr = (real)0.25 * log(sigma / (real)16);
  s.e = ARENA.get(c.nc); fourierFwdK<<<blocks(c.nc), 256, 0, STREAM>>>(c.fw.w, c.fb.w, s.tr, s.e, c.nc, trD);
  s.en = ARENA.get(c.nc); s.le = lnForward(c.noiseLn, s.e, s.en, 1);
  s.nproj = ARENA.get(c.Cs); c.noiseProj.forward(s.en, s.nproj, 1);
  s.s0 = cloneR(s.base, (size_t)n * c.Cs);
  addRowVecK<<<blocks((size_t)n * c.Cs), 256, 0, STREAM>>>(s.s0, s.nproj, n, c.Cs);
  s.s1 = cloneR(s.s0, (size_t)n * c.Cs); s.st0 = gtForward(c.singleT[0], s.s0, s.s1, n, 1);
  s.sOut = cloneR(s.s1, (size_t)n * c.Cs); s.st1 = gtForward(c.singleT[1], s.s1, s.sOut, n, 1);
  s.sFinal = s.sOut;
  if (c.hasSingleFinal) { s.sFinal = ARENA.get((size_t)n * c.Cs); s.lf = lnForward(c.singleFinal, s.sOut, s.sFinal, n); }
  return s;
}
// dpair [nn, Cz], dsingle [n, Cs] -> dtrunkSingle, dtrunkPair, dtargetFeat (added; any may be null). Either input may be
// null, its half skipped: a batch backs each sample's single half and then the samples' summed pair once
inline void condBackward(DiffCond& c, const CondSave& s, const real* dpairIn, const real* dsingleIn, real* dtrunkSingle,
                         real* dtrunkPair, real* dtargetFeat) {
  Scratch scratch;
  const int n = s.n; const size_t nn = (size_t)n * n; const int W = c.width(), SW = c.swidth();
  // single
  if (dsingleIn) {
  real* ds = cloneR(dsingleIn, (size_t)n * c.Cs);
  if (c.hasSingleFinal) { real* t = ARENA.get((size_t)n * c.Cs); lnBackward(c.singleFinal, s.lf, ds, t, n, false); ds = t; }
  { real* du = cloneR(ds, (size_t)n * c.Cs); gtBackward(c.singleT[1], s.st1, du, ds); }
  { real* du = cloneR(ds, (size_t)n * c.Cs); gtBackward(c.singleT[0], s.st0, du, ds); }
  real* dnp = ARENA.zeros(c.Cs);
  colSum(dnp, ds, n, c.Cs);
  real* den = ARENA.get(c.nc); c.noiseProj.backward(s.en, dnp, den, 1);
  real* de = ARENA.get(c.nc); lnBackward(c.noiseLn, s.le, den, de, 1, false);
  fourierBwdK<<<blocks(c.nc), 256, 0, STREAM>>>(c.fw.w, c.fb.w, s.tr, de, c.fw.g, c.fb.g, c.nc);
  real* df1n = ARENA.get((size_t)n * SW); c.singleProj.backward(s.f1n, ds, df1n, n);
  real* df1 = ARENA.get((size_t)n * SW); lnBackward(c.singleLn, s.ls, df1n, df1, n, false);
  real* tsd = dtrunkSingle ? dtrunkSingle : ARENA.zeros((size_t)n * c.Cst);
  concat2K<<<blocks((size_t)n * SW), 256, 0, STREAM>>>(nullptr, c.Cst, nullptr, c.F, df1, n, c.pad ? c.Cst + 31 : -1,
                                                       c.pad ? c.Cst + 63 : -1, tsd, dtargetFeat);
  }
  // pair
  if (!dpairIn) return;
  real* dp = cloneR(dpairIn, nn * c.Cz);
  if (c.chai) { real* t = ARENA.get(nn * c.Cz); lnBackward(c.pairFinal, s.lpf, dp, t, nn, false); dp = t; }
  { real* du = cloneR(dp, nn * c.Cz); gtBackward(c.pairT[1], s.pt1, du, dp); }
  { real* du = cloneR(dp, nn * c.Cz); gtBackward(c.pairT[0], s.pt0, du, dp); }
  real* df2n = ARENA.get(nn * W); c.pairProj.backward(s.f2n, dp, df2n, nn);
  real* df2 = ARENA.get(nn * W); lnBackward(c.pairLn, s.lp, df2n, df2, nn, false);
  real* tpd = dtrunkPair ? dtrunkPair : ARENA.zeros(nn * c.Czt);
  if (c.chai) {
    real* ds2 = ARENA.zeros(nn * c.Cz);
    concat2K<<<blocks(nn * W), 256, 0, STREAM>>>(nullptr, c.Czt, nullptr, c.Cz, df2, nn, -1, -1, tpd, ds2);
    c.structW.backward(s.rel, ds2, nullptr, nn);
    if (c.hasStructBond && c.declBonds) mmAT(c.declBonds, ds2, c.structBond.g, (int)nn, 1, c.Cz);
  } else if (c.split || c.relpe) {
    real* drp = ARENA.zeros(nn * c.Cz);
    real* dfirst = c.split ? ARENA.zeros(nn * c.Cz) : tpd;
    concat2K<<<blocks(nn * W), 256, 0, STREAM>>>(nullptr, c.split ? c.Cz : c.Czt, nullptr, c.Cz, df2, nn, -1, -1, dfirst, drp);
    c.relpeProj.backward(s.rel, drp, nullptr, nn);
    if (c.split) {
      real* dtn = ARENA.get(nn * c.Czt); c.zTrunkProj.backward(s.tn, dfirst, dtn, nn);
      lnBackward(c.zTrunkLn, s.lz, dtn, tpd, nn, true);
    }
  } else concat2K<<<blocks(nn * W), 256, 0, STREAM>>>(nullptr, c.Czt, nullptr, c.rel, df2, nn, -1, -1, tpd, nullptr);
}

// ---------------------------------------------------------------- the per-atom conditioning and the encoder
struct DiffEncoder {
  int C = 0, Cp = 0, H = 0, D = 0, perToken = 0, Cst = 0, Cz = 0;
  // the reference embedding (atomReference)
  Param wPos, wMask, wElem, wCharge, wName, bias; bool hasBias = false, rawCharge = false;
  LayerNorm tsLn, tpLn; Linear tsProj, tpProj;
  Linear row, col, mlp1, mlp2, mlp3, posProj, aggr, chiralProj;
  Param wOff, wDist, wValid;
  bool hasChiral = false, hasTrunkSingle = true, hasTrunkPair = true, hasPositions = true;
  AtomStack stack;
  bool preTrunkQuery = false, keysUidFromQueries = false, maskPaddedKeys = false;
  // chai-1's encoder (trunk.dialect.chaiAtomStack): its conditioning LayerNormed (affine-free), its atom-pair term a
  // biased projection of 14 features, a two-layer pair MLP with no ReLU on its input, the attention within one
  // reference space; set before forEachParam
  bool chai = false;
  Param wPairFeat, pairFeatBias;      // [14, Cp], [Cp]
};
// the batch's atom layout, on the device
struct AtomBatch {
  int tokens = 0, dense = 0, subsets = 0, Q = 32, K = 128;
  Gather t2q, q2k, q2t, tq, tk;            // tokenAtomsToQueries, queriesToKeys, queriesToTokenAtoms, tokensToQueries, tokensToKeys
  const real *refPos, *refMask, *refCharge, *atomMask, *uid;   // [atoms ...]; uid as real
  const int *refElement, *nameChars;
  const real* chiral = nullptr;            // [atoms, 3] rf3's chirality term, constant
};
__global__ void perAtomFwdK(const real* pos, const real* mask, const int* element, const real* charge, const int* name,
                            const real* Wpos, const real* Wmask, const real* Welem, const real* Wcharge, const real* Wname,
                            const real* bias, real* act, int rows, int C, bool rawCharge) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)rows * C) return;
  int r = t / C, c = t % C;
  real m = mask[r], ch = rawCharge ? charge[r] : asinh(charge[r]);
  real v = pos[r * 3] * Wpos[c] + pos[r * 3 + 1] * Wpos[C + c] + pos[r * 3 + 2] * Wpos[2 * C + c];
  if (bias) v += bias[c];
  v += m * Wmask[c];
  int z = element[r]; if (z >= 0 && z < 128) v += Welem[(size_t)z * C + c];
  v += ch * Wcharge[c];
  for (int k = 0; k < 4; ++k) { int code = name[r * 4 + k]; if (code >= 0 && code < 64) v += Wname[(size_t)(k * 64 + code) * C + c]; }
  act[t] = v * m;
}
__global__ void perAtomBwdK(const real* pos, const real* mask, const int* element, const real* charge, const int* name,
                            const real* d, real* dWpos, real* dWmask, real* dWelem, real* dWcharge, real* dWname, real* dbias,
                            int rows, int C, bool rawCharge) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)rows * C) return;
  int r = t / C, c = t % C;
  real m = mask[r], g = d[t] * m;
  if (g == 0) return;
  real ch = rawCharge ? charge[r] : asinh(charge[r]);
  for (int a = 0; a < 3; ++a) atomicAdd(&dWpos[a * C + c], g * pos[r * 3 + a]);
  if (dbias) atomicAdd(&dbias[c], g);
  atomicAdd(&dWmask[c], g * m);
  int z = element[r]; if (z >= 0 && z < 128) atomicAdd(&dWelem[(size_t)z * C + c], g);
  atomicAdd(&dWcharge[c], g * ch);
  for (int k = 0; k < 4; ++k) { int code = name[r * 4 + k]; if (code >= 0 && code < 64) atomicAdd(&dWname[(size_t)(k * 64 + code) * C + c], g); }
}
// pair[q, k] = row[q] + col[k] + valid (offsets W + dist / (1 + |d|^2) + Wvalid) (+ the token pair, gathered)
__global__ void atomPairFwdK(const real* row, const real* col, const real* qPos, const real* kPos, const real* qUid,
                             const real* kUid, const real* kMask, const real* Woff, const real* Wdist, const real* Wvalid,
                             const real* tp, const int* tqI, const real* tqM, const int* tkI, const real* tkM, real* pair,
                             int subsets, int Q, int K, int Cp, int tokens, bool maskPadded) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)subsets * Q * K * Cp) return;
  int c = t % Cp; size_t rest = t / Cp; int key = rest % K; size_t qi = rest / K; int s = qi / Q;
  size_t ki = (size_t)s * K + key;
  real v = row[qi * Cp + c] + col[ki * Cp + c];
  bool valid = qUid[qi] == kUid[ki] && (!maskPadded || kMask[ki] != 0);
  if (valid) {
    real d0 = qPos[qi * 3] - kPos[ki * 3], d1 = qPos[qi * 3 + 1] - kPos[ki * 3 + 1], d2 = qPos[qi * 3 + 2] - kPos[ki * 3 + 2];
    v += (d0 * Woff[c] + d1 * Woff[Cp + c] + d2 * Woff[2 * Cp + c]) + Wdist[c] / (1 + d0 * d0 + d1 * d1 + d2 * d2) + Wvalid[c];
  }
  if (tp && tqM[qi] != 0 && tkM[ki] != 0) v += tp[((size_t)tqI[qi] * tokens + tkI[ki]) * Cp + c];
  pair[t] = v;
}
__global__ void atomPairBwdK(const real* d, const real* qPos, const real* kPos, const real* qUid, const real* kUid,
                             const real* kMask, real* drow, real* dcol, real* dWoff, real* dWdist, real* dWvalid, real* dtp,
                             const int* tqI, const real* tqM, const int* tkI, const real* tkM, int subsets, int Q, int K,
                             int Cp, int tokens, bool maskPadded) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)subsets * Q * K * Cp) return;
  int c = t % Cp; size_t rest = t / Cp; int key = rest % K; size_t qi = rest / K; int s = qi / Q;
  size_t ki = (size_t)s * K + key;
  real g = d[t];
  atomicAdd(&drow[qi * Cp + c], g);
  atomicAdd(&dcol[ki * Cp + c], g);
  bool valid = qUid[qi] == kUid[ki] && (!maskPadded || kMask[ki] != 0);
  if (valid) {
    real d0 = qPos[qi * 3] - kPos[ki * 3], d1 = qPos[qi * 3 + 1] - kPos[ki * 3 + 1], d2 = qPos[qi * 3 + 2] - kPos[ki * 3 + 2];
    atomicAdd(&dWoff[c], g * d0); atomicAdd(&dWoff[Cp + c], g * d1); atomicAdd(&dWoff[2 * Cp + c], g * d2);
    atomicAdd(&dWdist[c], g / (1 + d0 * d0 + d1 * d1 + d2 * d2));
    atomicAdd(&dWvalid[c], g);
  }
  if (dtp && tqM[qi] != 0 && tkM[ki] != 0) atomicAdd(&dtp[((size_t)tqI[qi] * tokens + tkI[ki]) * Cp + c], g);
}
// tokenAct[t] = mean over t's real atoms of relu(projected[query of the atom]) + snProj[t]
__global__ void aggregateK(const real* proj, const int* q2tI, const real* q2tM, const real* atomMask, const real* snProj,
                           real* out, int tokens, int dense, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)tokens * P) return;
  int tok = t / P, c = t % P;
  real s = 0, cnt = 0;
  for (int a = 0; a < dense; ++a) {
    size_t atom = (size_t)tok * dense + a;
    if (atomMask[atom] == 0) continue;
    cnt += atomMask[atom];
    if (q2tM[atom] == 0) continue;
    real v = proj[(size_t)q2tI[atom] * P + c];
    s += v > 0 ? v : 0;
  }
  out[t] = (cnt > 0 ? s / cnt : (real)0) + (snProj ? snProj[t] : (real)0);
}
__global__ void aggregateBwdK(const real* proj, const int* q2tI, const real* q2tM, const real* atomMask, const real* d,
                              real* dproj, int tokens, int dense, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)tokens * P) return;
  int tok = t / P, c = t % P;
  real cnt = 0;
  for (int a = 0; a < dense; ++a) cnt += atomMask[(size_t)tok * dense + a];
  if (cnt <= 0) return;
  for (int a = 0; a < dense; ++a) {
    size_t atom = (size_t)tok * dense + a;
    if (atomMask[atom] == 0 || q2tM[atom] == 0) continue;
    size_t q = (size_t)q2tI[atom] * P + c;
    if (proj[q] > 0) atomicAdd(&dproj[q], d[t] / cnt);
  }
}


// the decoder's broadcast: act[q] = (proj[token of q's atom] + skip[q]) * qMask[q]
__global__ void broadcastK(const real* proj, const int* t2qI, const real* t2qM, const real* skip, const real* qMask, real* act,
                           size_t Rq, int C, int dense) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= Rq * C) return;
  size_t q = t / C; int c = t % C;
  real v = t2qM[q] != 0 ? proj[(size_t)(t2qI[q] / dense) * C + c] : (real)0;
  act[t] = (v + skip[t]) * qMask[q];
}
__global__ void broadcastBwdK(const real* d, const int* t2qI, const real* t2qM, const real* qMask, real* dproj, real* dskip,
                              size_t Rq, int C, int dense) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= Rq * C) return;
  size_t q = t / C; int c = t % C;
  real g = d[t] * qMask[q];
  dskip[t] += g;
  if (t2qM[q] != 0) atomicAdd(&dproj[(size_t)(t2qI[q] / dense) * C + c], g);
}
// out = (skip x + out upd) mask, and the scaled input x mask in
__global__ void edmOutK(const real* x, const real* upd, const real* mask, real* o, size_t atoms, real sk, real ou,
                        const real* scal = nullptr) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (scal) { sk = scal[2]; ou = scal[3]; }
  if (t < atoms * 3) o[t] = (sk * x[t] + ou * upd[t]) * mask[t / 3];
}
__global__ void scaleMaskK(const real* x, const real* mask, real* y, size_t atoms, real a, const real* aD = nullptr) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (aD) a = *aD;
  if (t < atoms * 3) y[t] = x[t] * mask[t / 3] * a;
}
// the uid gather stock AF3 makes off the wrong layout (keysUidFromQueriesLayout off): the index clamped to the array
__global__ void gatherClampK(const int* idx, const real* m, const real* src, size_t len, real* out, int count) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count) return;
  size_t i = (size_t)idx[t] < len ? (size_t)idx[t] : len - 1;
  out[t] = m[t] != 0 ? src[i] : (real)0;
}

// chai-1's atom-pair term: [one_hot(#(|d|^2 > e^2, e in 0 1 2 3 4 5 6 8 12 16), or 11 across reference spaces; 12) |
// 1 / (1 + |d|^2) | valid] W + b, plus row, col and the trunk's pair (cuda/af3's chaiAtomPairK); and back
__device__ inline int chaiPairBin(real sq, bool valid) {
  const real edges[10] = {0, 1, 4, 9, 16, 25, 36, 64, 144, 256};
  int idx = 0;
  for (int e = 0; e < 10; ++e) idx += sq > edges[e];
  return valid ? idx : 11;
}
__global__ void chaiPairFwdK(const real* row, const real* col, const real* qPos, const real* kPos, const real* qUid, const real* kUid,
                             const real* Wf, const real* bf, const real* tp, const int* tqI, const real* tqM, const int* tkI,
                             const real* tkM, real* pair, int subsets, int Q, int K, int Cp, int tokens) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)subsets * Q * K * Cp) return;
  int c = t % Cp; size_t rest = t / Cp; int key = rest % K; size_t qi = rest / K; int s = qi / Q;
  size_t ki = (size_t)s * K + key;
  real v = row[qi * Cp + c] + col[ki * Cp + c];
  if (tp && tqM[qi] != 0 && tkM[ki] != 0) v += tp[((size_t)tqI[qi] * tokens + tkI[ki]) * Cp + c];
  bool valid = qUid[qi] == kUid[ki];
  real d0 = qPos[qi * 3] - kPos[ki * 3], d1 = qPos[qi * 3 + 1] - kPos[ki * 3 + 1], d2 = qPos[qi * 3 + 2] - kPos[ki * 3 + 2];
  real sq = d0 * d0 + d1 * d1 + d2 * d2;
  v += bf[c] + Wf[(size_t)chaiPairBin(sq, valid) * Cp + c] + Wf[(size_t)12 * Cp + c] / (1 + sq) + (valid ? Wf[(size_t)13 * Cp + c] : 0);
  pair[t] = v;
}
__global__ void chaiPairBwdK(const real* d, const real* qPos, const real* kPos, const real* qUid, const real* kUid, real* drow,
                             real* dcol, real* dWf, real* dbf, real* dtp, const int* tqI, const real* tqM, const int* tkI,
                             const real* tkM, int subsets, int Q, int K, int Cp, int tokens) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)subsets * Q * K * Cp) return;
  int c = t % Cp; size_t rest = t / Cp; int key = rest % K; size_t qi = rest / K; int s = qi / Q;
  size_t ki = (size_t)s * K + key;
  real g = d[t];
  atomicAdd(&drow[qi * Cp + c], g); atomicAdd(&dcol[ki * Cp + c], g); atomicAdd(&dbf[c], g);
  bool valid = qUid[qi] == kUid[ki];
  real d0 = qPos[qi * 3] - kPos[ki * 3], d1 = qPos[qi * 3 + 1] - kPos[ki * 3 + 1], d2 = qPos[qi * 3 + 2] - kPos[ki * 3 + 2];
  real sq = d0 * d0 + d1 * d1 + d2 * d2;
  atomicAdd(&dWf[(size_t)chaiPairBin(sq, valid) * Cp + c], g);
  atomicAdd(&dWf[(size_t)12 * Cp + c], g / (1 + sq));
  if (valid) atomicAdd(&dWf[(size_t)13 * Cp + c], g);
  if (dtp && tqM[qi] != 0 && tkM[ki] != 0) atomicAdd(&dtp[((size_t)tqI[qi] * tokens + tkI[ki]) * Cp + c], g);
}
// chai-1's attention mask: both atoms real and in one reference space (cuda/af3's sameRefSpaceMaskK), on the bias
__global__ void chaiRefMaskK(real* mb, const real* qUid, const real* tqM, const real* kUid, const real* tkM, int subsets, int Q, int K) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)subsets * Q * K) return;
  int k = t % K; size_t r = t / K; int q = r % Q; int s = (int)(r / Q);
  size_t qi = (size_t)s * Q + q, ki = (size_t)s * K + k;
  if (!(tqM[qi] != 0 && tkM[ki] != 0 && qUid[qi] == kUid[ki])) mb[t] = (real)-1e9;
}

struct EncSave {
  real *cond, *qCond0, *tsn, *tsProj, *qCond, *kCond, *qMask, *kMask, *rq, *rk, *row, *col, *tpn, *tp, *qPos, *kPos, *qUid, *kUid,
       *pair0, *h1, *h2, *pair, *pos, *act0, *skip, *proj, *mb, *qPre;
  LnSave lts, ltp, lc;
  StackRun run;
  AtomCtx ctx;
  StackPre spre;     // the atom stack's terms every sample shares (its pair logits, its blocks' conditioning)
  int n;
};
struct DecSave { real *proj, *act0, *act, *xm, *xn, *upd, *qc, *kc; LnSave ln, lc; StackRun run; AtomCtx ctx; };
// the decoder's part every sample of a batch shares (decShared): its context (the attention's mask bias; chai's
// conditioning norm) and its stack's terms
struct DecPre { AtomCtx ctx; real *qc = nullptr, *kc = nullptr; LnSave lc; StackPre pre; };
struct Denoiser {
  DiffCond cond;
  DiffEncoder enc;
  DiffTransformer tr;
  LayerNorm snLn, outLn; bool hasSnLn = true;
  Linear snProj;
  Linear decBroadcast, decUpdate;
  AtomStack decStack;
  LayerNorm decLn;
  bool decChai = false;        // chai-1: the decoder conditioned on an affine LayerNorm of the encoder's, within one reference space
  LayerNorm decCondLn;
  Params params;
  DiffRun run;          // the transformer's kept block inputs: cudaMalloc'd once, reused by every call at this size
};
struct DenoiseSave {
  CondSave cs; EncSave es; DecSave ds;
  LnSave lsn, lout;
  real *sn, *snP, *tok, *trOut, *actn, *scaled, *updAtoms, *out;
  real sigma, sk, ou, in;
  const real* scal = nullptr;   // (a sampler step's on the device instead: [tr, in, sk, ou], denoiseScalars' order)
  int n;
};

// The encoder in two parts: what every sample of a diffusion batch shares - the reference embedding, the trunk's single
// and pair terms, the atom pair and its MLP, the masks; nothing of it reads the noisy positions - once a batch
// (encShared), and each sample's own queries, atom stack and aggregation (encSample). encForward is the two in turn.
inline EncSave encShared(DiffEncoder& e, const AtomBatch& b, const real* trunkSingle, const real* condPair) {
  const int C = e.C, Cp = e.Cp, n = b.tokens;
  const size_t atoms = (size_t)b.tokens * b.dense, Rq = (size_t)b.subsets * b.Q, Rk = (size_t)b.subsets * b.K, rows = Rq * b.K;
  EncSave s{}; s.n = n;
  s.cond = ARENA.get(atoms * C);
  perAtomFwdK<<<blocks(atoms * C), 256, 0, STREAM>>>(b.refPos, b.refMask, b.refElement, b.refCharge, b.nameChars, e.wPos.w,
      e.wMask.w, e.wElem.w, e.wCharge.w, e.wName.w, e.hasBias ? e.bias.w : nullptr, s.cond, (int)atoms, C, e.rawCharge);
  s.qMask = ARENA.get(Rq); gather(b.t2q, b.refMask, s.qMask, 1);
  s.qCond0 = ARENA.get(Rq * C); gather(b.t2q, s.cond, s.qCond0, C);
  s.qCond = cloneR(s.qCond0, Rq * C);
  if (trunkSingle) {
    s.tsn = ARENA.get((size_t)n * e.Cst); s.lts = lnForward(e.tsLn, trunkSingle, s.tsn, n);
    s.tsProj = ARENA.get((size_t)n * C); e.tsProj.forward(s.tsn, s.tsProj, n);
    real* pq = ARENA.get(Rq * C); gather(b.tq, s.tsProj, pq, C); add(s.qCond, pq, Rq * C);
  }
  if (e.chai) {     // chai: the conditioning LayerNormed, affine-free, before the mask
    s.qPre = s.qCond; s.qCond = ARENA.get(Rq * C);
    LayerNorm plain; plain.C = C;
    s.lc = lnForward(plain, s.qPre, s.qCond, Rq);
  }
  rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(s.qCond, s.qMask, Rq, C);
  s.kCond = ARENA.get(Rk * C); gather(b.q2k, s.qCond, s.kCond, C);
  s.kMask = ARENA.get(Rk); gather(b.q2k, s.qMask, s.kMask, 1);
  s.rq = ARENA.get(Rq * C); reluK<<<blocks(Rq * C), 256, 0, STREAM>>>(s.rq, s.qCond, Rq * C);
  s.rk = ARENA.get(Rk * C); reluK<<<blocks(Rk * C), 256, 0, STREAM>>>(s.rk, s.kCond, Rk * C);
  s.row = ARENA.get(Rq * Cp); e.row.forward(s.rq, s.row, Rq);
  s.col = ARENA.get(Rk * Cp); e.col.forward(s.rk, s.col, Rk);
  if (condPair) {
    const size_t nn = (size_t)n * n;
    s.tpn = ARENA.get(nn * e.Cz); s.ltp = lnForward(e.tpLn, condPair, s.tpn, nn);
    s.tp = ARENA.get(nn * Cp); e.tpProj.forward(s.tpn, s.tp, nn);
  }
  s.qPos = ARENA.get(Rq * 3); gather(b.t2q, b.refPos, s.qPos, 3);
  s.kPos = ARENA.get(Rk * 3); gather(b.q2k, s.qPos, s.kPos, 3);
  s.qUid = ARENA.get(Rq); gather(b.t2q, b.uid, s.qUid, 1);
  s.kUid = ARENA.get(Rk);
  if (e.keysUidFromQueries) gather(b.q2k, s.qUid, s.kUid, 1);
  else gatherClampK<<<blocks(Rk), 256, 0, STREAM>>>(b.q2k.idx, b.q2k.mask, b.uid, atoms, s.kUid, (int)Rk);
  s.pair0 = ARENA.get(rows * Cp);
  if (e.chai)
    chaiPairFwdK<<<blocks(rows * Cp), 256, 0, STREAM>>>(s.row, s.col, s.qPos, s.kPos, s.qUid, s.kUid, e.wPairFeat.w, e.pairFeatBias.w,
        s.tp, b.tq.idx, b.tq.mask, b.tk.idx, b.tk.mask, s.pair0, b.subsets, b.Q, b.K, Cp, n);
  else
  atomPairFwdK<<<blocks(rows * Cp), 256, 0, STREAM>>>(s.row, s.col, s.qPos, s.kPos, s.qUid, s.kUid, s.kMask, e.wOff.w, e.wDist.w,
      e.wValid.w, s.tp, b.tq.idx, b.tq.mask, b.tk.idx, b.tk.mask, s.pair0, b.subsets, b.Q, b.K, Cp, n, e.maskPaddedKeys);
  s.h1 = ARENA.get(rows * Cp); s.h2 = ARENA.get(rows * Cp);
  if (e.chai) {     // chai's pair MLP: pair = pair0 + W2 relu(W1 pair0)
    e.mlp1.forward(s.pair0, s.h1, rows);
    s.pair = cloneR(s.pair0, rows * Cp);
    Scratch sc;
    real* r = ARENA.get(rows * Cp); reluK<<<blocks(rows * Cp), 256, 0, STREAM>>>(r, s.h1, rows * Cp);
    e.mlp2.forward(r, s.pair, rows, 1);
  } else {
  {
    Scratch sc;
    real* r = ARENA.get(rows * Cp);
    reluK<<<blocks(rows * Cp), 256, 0, STREAM>>>(r, s.pair0, rows * Cp); e.mlp1.forward(r, s.h1, rows);
    reluK<<<blocks(rows * Cp), 256, 0, STREAM>>>(r, s.h1, rows * Cp); e.mlp2.forward(r, s.h2, rows);
  }
  s.pair = cloneR(s.pair0, rows * Cp);
  {
    Scratch sc;
    real* r = ARENA.get(rows * Cp);
    reluK<<<blocks(rows * Cp), 256, 0, STREAM>>>(r, s.h2, rows * Cp); e.mlp3.forward(r, s.pair, rows, 1);
  }
  }
  s.mb = ARENA.get(rows);
  atomMaskBiasK<<<blocks(rows), 256, 0, STREAM>>>(s.qMask, s.kMask, s.mb, b.subsets, b.Q, b.K, e.stack.blocks[0].keyMasked);
  if (e.chai) chaiRefMaskK<<<blocks(rows), 256, 0, STREAM>>>(s.mb, s.qUid, b.tq.mask, s.kUid, b.tk.mask, b.subsets, b.Q, b.K);
  s.ctx = AtomCtx{b.q2k, s.qMask, s.kMask, s.qCond, s.kCond, s.mb, b.subsets, b.Q, b.K};
  s.spre = stackPre(e.stack, s.ctx, s.pair);
  return s;
}
inline EncSave encSample(DiffEncoder& e, const AtomBatch& b, EncSave s, const real* positions, const real* snProj, real* tokenAct) {
  const int C = e.C, n = b.tokens;
  const size_t Rq = (size_t)b.subsets * b.Q;
  // the queries: the start (cond, or its masked pre-trunk copy - chai's always), plus the positions' projection, masked
  const bool preStart = e.preTrunkQuery && (s.tsn || e.chai);   // (s.tsn: a trunk single was given)
  s.act0 = cloneR(preStart ? s.qCond0 : s.qCond, Rq * C);
  if (preStart) rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(s.act0, s.qMask, Rq, C);
  if (positions) {
    s.pos = ARENA.get(Rq * 3); gather(b.t2q, positions, s.pos, 3);
    real* pp = ARENA.get(Rq * C); e.posProj.forward(s.pos, pp, Rq);
    if (e.hasChiral && b.chiral) { real* cg = ARENA.get(Rq * 3); gather(b.t2q, b.chiral, cg, 3); e.chiralProj.forward(cg, pp, Rq, 1); }
    rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(pp, s.qMask, Rq, C);
    add(s.act0, pp, Rq * C);
  }
  s.skip = ARENA.get(Rq * C);
  stackForward(e.stack, s.ctx, s.pair, s.act0, s.skip, s.run, &s.spre);
  rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(s.skip, s.qMask, Rq, C);
  s.proj = ARENA.get(Rq * e.perToken); e.aggr.forward(s.skip, s.proj, Rq);
  aggregateK<<<blocks((size_t)n * e.perToken), 256, 0, STREAM>>>(s.proj, b.q2t.idx, b.q2t.mask, b.refMask, snProj, tokenAct, n,
                                                                  b.dense, e.perToken);
  return s;
}
inline EncSave encForward(DiffEncoder& e, const AtomBatch& b, const real* trunkSingle, const real* condPair,
                          const real* positions, const real* snProj, real* tokenAct) {
  return encSample(e, b, encShared(e, b, trunkSingle, condPair), positions, snProj, tokenAct);
}
// The backward in the forward's two parts. The shared tensors' gradients (EncGrads: the conditioning queries and keys,
// the start's, the atom pair) are summed over a batch's samples by encSampleBackward and backed once by
// encSharedBackward - the shared part is linear in them, so once is the sum of each sample's.
struct EncGrads { real *dqCond, *dqCond0, *dkCond, *dpair; StackGrads sg; };   // (sg: the stack's shared terms')
inline AtomCtx atomShape(const AtomBatch& b) { AtomCtx c; c.subsets = b.subsets; c.Q = b.Q; c.K = b.K; return c; }
inline EncGrads encGrads(const DiffEncoder& e, const AtomBatch& b) {
  const size_t Rq = (size_t)b.subsets * b.Q, Rk = (size_t)b.subsets * b.K, rows = Rq * b.K;
  return EncGrads{ARENA.zeros(Rq * e.C), ARENA.zeros(Rq * e.C), ARENA.zeros(Rk * e.C), ARENA.zeros(rows * e.Cp), stackGrads(e.stack, atomShape(b))};
}
// one sample: dtokenAct, dskip (from the decoder), dpair (from the decoder's pair logits), dqCondDec (the decoder's
// conditioning) -> dpositions (added; may be null), and the shared tensors' gradients into g (added)
inline void encSampleBackward(DiffEncoder& e, const AtomBatch& b, const EncSave& s, const real* trunkSingle, const real* positions,
                              const real* dtokenAct, real* dskipIn, real* dpairIn, real* dqCondIn, real* dpositions, EncGrads& g) {
  Scratch scratch;
  const int C = e.C, Cp = e.Cp, n = s.n;
  const size_t Rq = (size_t)b.subsets * b.Q, rows = Rq * b.K;
  real* dproj = ARENA.zeros(Rq * e.perToken);
  aggregateBwdK<<<blocks((size_t)n * e.perToken), 256, 0, STREAM>>>(s.proj, b.q2t.idx, b.q2t.mask, b.refMask, dtokenAct, dproj,
                                                                     n, b.dense, e.perToken);
  real* dskip = cloneR(dskipIn, Rq * C);
  e.aggr.backward(s.skip, dproj, dskip, Rq, true);
  rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(dskip, s.qMask, Rq, C);
  real* dact0 = ARENA.get(Rq * C);
  if (dqCondIn) add(g.dqCond, dqCondIn, Rq * C);
  if (dpairIn) add(g.dpair, dpairIn, rows * Cp);
  stackBackward(e.stack, s.ctx, s.pair, s.run, dskip, dact0, g.dqCond, g.dkCond, g.dpair, &s.spre, &g.sg);
  if (positions) {
    real* dpp = cloneR(dact0, Rq * C);
    rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(dpp, s.qMask, Rq, C);
    real* dpos = ARENA.get(Rq * 3);
    e.posProj.backward(s.pos, dpp, dpos, Rq);
    if (e.hasChiral && b.chiral) { real* cg = ARENA.get(Rq * 3); gather(b.t2q, b.chiral, cg, 3); e.chiralProj.backward(cg, dpp, nullptr, Rq); }
    if (dpositions) scatter(b.t2q, dpos, dpositions, 3);
  }
  if (e.preTrunkQuery && (trunkSingle || e.chai)) {
    real* t = cloneR(dact0, Rq * C); rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(t, s.qMask, Rq, C); add(g.dqCond0, t, Rq * C);
  } else add(g.dqCond, dact0, Rq * C);
}
// ...and the shared part's, once: g -> dtrunkSingle, dcondPair (added; may be null) and the encoder's parameters
inline void encSharedBackward(DiffEncoder& e, const AtomBatch& b, const EncSave& s, const real* trunkSingle, EncGrads& g,
                              real* dtrunkSingle, real* dcondPair) {
  Scratch scratch;
  const int C = e.C, Cp = e.Cp, n = s.n;
  const size_t atoms = (size_t)b.tokens * b.dense, Rq = (size_t)b.subsets * b.Q, Rk = (size_t)b.subsets * b.K, rows = Rq * b.K;
  real *dqCond = g.dqCond, *dqCond0 = g.dqCond0, *dkCond = g.dkCond, *dpair = g.dpair;
  stackPreBackward(e.stack, s.ctx, s.pair, s.spre, g.sg, dqCond, dkCond, dpair);   // (the samples' summed: once)
  // the pair: MLP residual, then its features
  if (e.chai) {
    real *r = ARENA.get(rows * Cp), *dr = ARENA.get(rows * Cp), *dh = ARENA.get(rows * Cp);
    reluK<<<blocks(rows * Cp), 256, 0, STREAM>>>(r, s.h1, rows * Cp);
    e.mlp2.backward(r, dpair, dr, rows);
    reluBwdK<<<blocks(rows * Cp), 256, 0, STREAM>>>(dr, s.h1, rows * Cp);
    e.mlp1.backward(s.pair0, dr, dh, rows);
    add(dpair, dh, rows * Cp);
  } else {
    real *r = ARENA.get(rows * Cp), *dr = ARENA.get(rows * Cp), *dh = ARENA.get(rows * Cp);
    reluK<<<blocks(rows * Cp), 256, 0, STREAM>>>(r, s.h2, rows * Cp);
    e.mlp3.backward(r, dpair, dr, rows);
    reluBwdK<<<blocks(rows * Cp), 256, 0, STREAM>>>(dr, s.h2, rows * Cp);
    reluK<<<blocks(rows * Cp), 256, 0, STREAM>>>(r, s.h1, rows * Cp);
    e.mlp2.backward(r, dr, dh, rows);
    reluBwdK<<<blocks(rows * Cp), 256, 0, STREAM>>>(dh, s.h1, rows * Cp);
    reluK<<<blocks(rows * Cp), 256, 0, STREAM>>>(r, s.pair0, rows * Cp);
    e.mlp1.backward(r, dh, dr, rows);
    reluBwdK<<<blocks(rows * Cp), 256, 0, STREAM>>>(dr, s.pair0, rows * Cp);
    add(dpair, dr, rows * Cp);                       // d pair0 = d pair (the residual) + the MLP's
  }
  real *drow = ARENA.zeros(Rq * Cp), *dcol = ARENA.zeros(Rk * Cp), *dtp = s.tp ? ARENA.zeros((size_t)n * n * Cp) : nullptr;
  if (e.chai)
    chaiPairBwdK<<<blocks(rows * Cp), 256, 0, STREAM>>>(dpair, s.qPos, s.kPos, s.qUid, s.kUid, drow, dcol, e.wPairFeat.g, e.pairFeatBias.g,
        dtp, b.tq.idx, b.tq.mask, b.tk.idx, b.tk.mask, b.subsets, b.Q, b.K, Cp, n);
  else
  atomPairBwdK<<<blocks(rows * Cp), 256, 0, STREAM>>>(dpair, s.qPos, s.kPos, s.qUid, s.kUid, s.kMask, drow, dcol, e.wOff.g,
      e.wDist.g, e.wValid.g, dtp, b.tq.idx, b.tq.mask, b.tk.idx, b.tk.mask, b.subsets, b.Q, b.K, Cp, n, e.maskPaddedKeys);
  if (s.tp) {
    const size_t nn = (size_t)n * n;
    real* dtpn = ARENA.get(nn * e.Cz); e.tpProj.backward(s.tpn, dtp, dtpn, nn);
    if (dcondPair) lnBackward(e.tpLn, s.ltp, dtpn, dcondPair, nn, true);
    else { real* t = ARENA.get(nn * e.Cz); lnBackward(e.tpLn, s.ltp, dtpn, t, nn, false); }
  }
  real* drq = ARENA.get(Rq * C); e.row.backward(s.rq, drow, drq, Rq);
  reluBwdK<<<blocks(Rq * C), 256, 0, STREAM>>>(drq, s.qCond, Rq * C); add(dqCond, drq, Rq * C);
  real* drk = ARENA.get(Rk * C); e.col.backward(s.rk, dcol, drk, Rk);
  reluBwdK<<<blocks(Rk * C), 256, 0, STREAM>>>(drk, s.kCond, Rk * C); add(dkCond, drk, Rk * C);
  scatter(b.q2k, dkCond, dqCond, C);                 // kCond = gather(qCond)
  rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(dqCond, s.qMask, Rq, C);   // qCond = mask (qCond0 + trunk term)
  if (e.chai) {     // ... its LayerNorm (affine-free, so its weights take nothing)
    real* d = ARENA.get(Rq * C);
    LayerNorm plain; plain.C = C;
    lnBackward(plain, s.lc, dqCond, d, Rq, false);
    dqCond = d;
  }
  add(dqCond0, dqCond, Rq * C);
  if (trunkSingle) {
    real* dts = ARENA.zeros((size_t)n * C); scatter(b.tq, dqCond, dts, C);
    real* dtsn = ARENA.get((size_t)n * e.Cst); e.tsProj.backward(s.tsn, dts, dtsn, n);
    if (dtrunkSingle) lnBackward(e.tsLn, s.lts, dtsn, dtrunkSingle, n, true);
    else { real* t = ARENA.get((size_t)n * e.Cst); lnBackward(e.tsLn, s.lts, dtsn, t, n, false); }
  }
  real* dcond = ARENA.zeros(atoms * C); scatter(b.t2q, dqCond0, dcond, C);
  perAtomBwdK<<<blocks(atoms * C), 256, 0, STREAM>>>(b.refPos, b.refMask, b.refElement, b.refCharge, b.nameChars, dcond, e.wPos.g,
      e.wMask.g, e.wElem.g, e.wCharge.g, e.wName.g, e.hasBias ? e.bias.g : nullptr, (int)atoms, C, e.rawCharge);
}
// dtokenAct, dskip (from the decoder), dpair (from the decoder's pair logits), dqCondDec (the decoder's conditioning)
//   -> dtrunkSingle, dcondPair, dpositions (added; may be null)
inline void encBackward(DiffEncoder& e, const AtomBatch& b, const EncSave& s, const real* trunkSingle, const real* positions,
                        const real* dtokenAct, real* dskipIn, real* dpairIn, real* dqCondIn, real* dtrunkSingle,
                        real* dcondPair, real* dpositions) {
  Scratch scratch;
  EncGrads g = encGrads(e, b);
  encSampleBackward(e, b, s, trunkSingle, positions, dtokenAct, dskipIn, dpairIn, dqCondIn, dpositions, g);
  encSharedBackward(e, b, s, trunkSingle, g, dtrunkSingle, dcondPair);
}

// the whole denoiser: trunk single [n, Cst], trunk pair [nn, Czt], target_feat [n, F], noisy positions [atoms, 3] -> out
// The denoiser in two per-sample halves around the transformer (denoisePre: the scaled positions, the single's
// projection, the encoder; denoisePost: the decoder and the EDM output), so that S samples can share one batched
// transformer pass and one pair conditioning (denoiseBatchForward / denoiseBatchBackward) - the same function sample by
// sample, as OpenFold3's diffusion module batches its 48.
inline void denoiseScales(DenoiseSave& s, real sigma) {
  s.sigma = sigma;
  const real d = sigma * sigma + 256;
  s.sk = 256 / d; s.ou = sigma * 16 / sqrt(d); s.in = 1 / sqrt(d);
}
// the four scalars a denoiser pass takes from its noise level, as denoiseScales and condSingleForward make them:
// [the Fourier embedding's input, in, skip, out]
inline void denoiseScalars(real sigma, real* o) {
  DenoiseSave s; denoiseScales(s, sigma);
  o[0] = (real)0.25 * log(sigma / (real)16); o[1] = s.in; o[2] = s.sk; o[3] = s.ou;
}
// (shared: the encoder's batch-shared part, made once by the caller - encForward's whole otherwise)
inline void denoisePre(Denoiser& m, const AtomBatch& b, DenoiseSave& s, const real* trunkSingle, const real* x, real* tok,
                       const EncSave* shared = nullptr) {
  const int n = b.tokens; const size_t atoms = (size_t)n * b.dense;
  s.scaled = ARENA.get(atoms * 3);
  scaleMaskK<<<blocks(atoms * 3), 256, 0, STREAM>>>(x, b.atomMask, s.scaled, atoms, s.in, s.scal ? s.scal + 1 : nullptr);
  const int Cs = m.cond.Cs, P = m.enc.perToken;
  s.sn = s.cs.sFinal;
  if (m.hasSnLn) { s.sn = ARENA.get((size_t)n * Cs); s.lsn = lnForward(m.snLn, s.cs.sFinal, s.sn, n); }
  s.snP = ARENA.get((size_t)n * P); m.snProj.forward(s.sn, s.snP, n);
  s.tok = tok;
  s.es = shared ? encSample(m.enc, b, *shared, s.scaled, s.snP, s.tok) : encForward(m.enc, b, trunkSingle, s.cs.pairOut, s.scaled, s.snP, s.tok);
}
// the decoder's shared part, made once a batch from the encoder's
inline DecPre decShared(Denoiser& m, const AtomBatch& b, const EncSave& enc) {
  const size_t Rq = (size_t)b.subsets * b.Q, Rk = (size_t)b.subsets * b.K;
  const int C = m.enc.C;
  DecPre d;
  d.ctx = enc.ctx;
  d.ctx.maskBias = ARENA.get(Rq * b.K);
  atomMaskBiasK<<<blocks(Rq * b.K), 256, 0, STREAM>>>(enc.qMask, enc.kMask, const_cast<real*>(d.ctx.maskBias), b.subsets, b.Q,
                                                      b.K, m.decStack.blocks[0].keyMasked);
  if (m.decChai) {
    chaiRefMaskK<<<blocks(Rq * b.K), 256, 0, STREAM>>>(const_cast<real*>(d.ctx.maskBias), enc.qUid, b.tq.mask, enc.kUid, b.tk.mask,
                                                       b.subsets, b.Q, b.K);
    d.qc = ARENA.get(Rq * C); d.lc = lnForward(m.decCondLn, enc.qCond, d.qc, Rq);
    d.kc = ARENA.get(Rk * C); gather(b.q2k, d.qc, d.kc, C);
    d.ctx.qCond = d.qc; d.ctx.kCond = d.kc;
  }
  d.pre = stackPre(m.decStack, d.ctx, enc.pair);
  return d;
}
// ...and its backward once: the samples' summed gradients (sg) -> the encoder's shared conditioning and atom pair (added)
inline void decSharedBackward(Denoiser& m, const AtomBatch& b, const EncSave& enc, const DecPre& d, const StackGrads& sg, EncGrads& eg) {
  Scratch scratch;
  const size_t Rq = (size_t)b.subsets * b.Q, Rk = (size_t)b.subsets * b.K;
  const int C = m.enc.C;
  real *dq = ARENA.zeros(Rq * C), *dk = ARENA.zeros(Rk * C);
  stackPreBackward(m.decStack, d.ctx, enc.pair, d.pre, sg, dq, dk, eg.dpair);
  scatter(b.q2k, dk, dq, C);
  if (m.decChai) { real* t = ARENA.get(Rq * C); lnBackward(m.decCondLn, d.lc, dq, t, Rq, false); dq = t; }
  add(eg.dqCond, dq, Rq * C);
}
// (dec: the decoder's shared part made already)
inline void denoisePost(Denoiser& m, const AtomBatch& b, DenoiseSave& s, const real* x, const real* trOut, real* out,
                        const DecPre* dec = nullptr) {
  const int n = b.tokens; const size_t atoms = (size_t)n * b.dense, Rq = (size_t)b.subsets * b.Q;
  const int P = m.enc.perToken;
  s.trOut = const_cast<real*>(trOut);
  s.actn = ARENA.get((size_t)n * P); s.lout = lnForward(m.outLn, s.trOut, s.actn, n);
  // the decoder
  const int C = m.enc.C;
  s.ds.proj = ARENA.get((size_t)n * C); m.decBroadcast.forward(s.actn, s.ds.proj, n);
  s.ds.act0 = ARENA.get(Rq * C);
  broadcastK<<<blocks(Rq * C), 256, 0, STREAM>>>(s.ds.proj, b.t2q.idx, b.t2q.mask, s.es.skip, s.es.qMask, s.ds.act0, Rq, C, b.dense);
  if (dec) s.ds.ctx = dec->ctx;
  else {
  s.ds.ctx = s.es.ctx;
  s.ds.ctx.maskBias = ARENA.get(Rq * b.K);
  atomMaskBiasK<<<blocks(Rq * b.K), 256, 0, STREAM>>>(s.es.qMask, s.es.kMask, const_cast<real*>(s.ds.ctx.maskBias), b.subsets, b.Q,
                                                      b.K, m.decStack.blocks[0].keyMasked);
  if (m.decChai) {
    chaiRefMaskK<<<blocks(Rq * b.K), 256, 0, STREAM>>>(const_cast<real*>(s.ds.ctx.maskBias), s.es.qUid, b.tq.mask, s.es.kUid, b.tk.mask,
                                                       b.subsets, b.Q, b.K);
    const size_t Rk = (size_t)b.subsets * b.K;
    s.ds.qc = ARENA.get(Rq * C); s.ds.lc = lnForward(m.decCondLn, s.es.qCond, s.ds.qc, Rq);
    s.ds.kc = ARENA.get(Rk * C); gather(b.q2k, s.ds.qc, s.ds.kc, C);
    s.ds.ctx.qCond = s.ds.qc; s.ds.ctx.kCond = s.ds.kc;
  }
  }
  s.ds.act = ARENA.get(Rq * C);
  stackForward(m.decStack, s.ds.ctx, s.es.pair, s.ds.act0, s.ds.act, s.ds.run, dec ? &dec->pre : nullptr);
  s.ds.xm = cloneR(s.ds.act, Rq * C); rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(s.ds.xm, s.es.qMask, Rq, C);
  s.ds.xn = ARENA.get(Rq * C); s.ds.ln = lnForward(m.decLn, s.ds.xm, s.ds.xn, Rq);
  s.ds.upd = ARENA.get(Rq * 3); m.decUpdate.forward(s.ds.xn, s.ds.upd, Rq);
  s.updAtoms = ARENA.get(atoms * 3); gather(b.q2t, s.ds.upd, s.updAtoms, 3);
  edmOutK<<<blocks(atoms * 3), 256, 0, STREAM>>>(x, s.updAtoms, b.atomMask, out, atoms, s.sk, s.ou, s.scal);
}
inline DenoiseSave denoiseForward(Denoiser& m, const AtomBatch& b, const real* trunkSingle, const real* trunkPair,
                                  const real* targetFeat, const real* seqMask, const real* x, real sigma, const int* ri,
                                  const int* ti, const int* asym, const int* ent, const int* sym, real* out) {
  const int n = b.tokens, P = m.enc.perToken;
  DenoiseSave s{}; s.n = n;
  denoiseScales(s, sigma);
  s.cs = condForward(m.cond, trunkSingle, trunkPair, targetFeat, ri, ti, asym, ent, sym, sigma, n);
  denoisePre(m, b, s, trunkSingle, x, ARENA.get((size_t)n * P));
  real* trOut = ARENA.get((size_t)n * P);
  {
    // the transformer keeps its own block inputs (cudaMalloc'd); its arena use is scoped inside
    Scratch sc;
    transformerForward(m.tr, m.run, s.tok, s.cs.sFinal, seqMask, s.cs.pairOut, trOut, n);
  }
  denoisePost(m, b, s, x, trOut, out);
  return s;
}
// a sampler's step (no gradient): D(x, sigma) on a conditioning's pair half and an encoder's shared part made once for
// every step (pair: condForward's, enc: encShared's) - only the noise level's single half and the sample's own parts
// made again, as denoiseBatchForward's sample k is; after the first step (first false) the transformer's pair logits too.
// scal: the level's scalars on the device (denoiseScalars'), read there instead of sigma's - so that one captured
// step replays at every level
inline void denoiseStepForward(Denoiser& m, const AtomBatch& b, const CondSave& pair, const EncSave& enc, const DecPre& dec,
                               const real* trunkSingle, const real* seqMask, const real* x, real sigma, real* out, bool first,
                               const real* scal = nullptr) {
  Scratch sc;
  const int n = b.tokens, P = m.enc.perToken;
  DenoiseSave s{}; s.n = n;
  denoiseScales(s, sigma); s.scal = scal;
  s.cs = condSingleForward(m.cond, pair, sigma, scal);
  denoisePre(m, b, s, trunkSingle, x, ARENA.get((size_t)n * P), &enc);
  real* trOut = ARENA.get((size_t)n * P);
  {
    Scratch st;
    transformerForward(m.tr, m.run, s.tok, s.cs.sFinal, seqMask, s.cs.pairOut, trOut, n, 1, !first);
  }
  denoisePost(m, b, s, x, trOut, out, &dec);
}
// the backward's halves: after the transformer (dout -> the transformer output's gradient, and what the encoder's
// backward reads), then before it (the transformer input's gradient -> the encoder, the single's projection)
struct DenoiseGrads { real *dtrOut, *dskip, *dpairDec, *dqCondDec; };
// (dec, dsg: the decoder's shared part, and its gradients added into dsg - g.dqCondDec and g.dpairDec then null)
inline DenoiseGrads denoiseBackPost(Denoiser& m, const AtomBatch& b, const DenoiseSave& s, const real* dout, real* dx,
                                    const DecPre* dec = nullptr, StackGrads* dsg = nullptr) {
  const int n = s.n, C = m.enc.C, P = m.enc.perToken, Cp = m.enc.Cp;
  const size_t atoms = (size_t)n * b.dense, Rq = (size_t)b.subsets * b.Q, Rk = (size_t)b.subsets * b.K, rows = Rq * b.K;
  // out = (sk x + ou upd) mask
  real* dupdA = ARENA.get(atoms * 3);
  scaleMaskK<<<blocks(atoms * 3), 256, 0, STREAM>>>(dout, b.atomMask, dupdA, atoms, s.ou);
  if (dx) { real* t = ARENA.get(atoms * 3); scaleMaskK<<<blocks(atoms * 3), 256, 0, STREAM>>>(dout, b.atomMask, t, atoms, s.sk); add(dx, t, atoms * 3); }
  real* dupd = ARENA.zeros(Rq * 3); scatter(b.q2t, dupdA, dupd, 3);
  real* dxn = ARENA.get(Rq * C); m.decUpdate.backward(s.ds.xn, dupd, dxn, Rq);
  real* dact = ARENA.get(Rq * C); lnBackward(m.decLn, s.ds.ln, dxn, dact, Rq, false);
  rowMaskK<<<blocks(Rq * C), 256, 0, STREAM>>>(dact, s.es.qMask, Rq, C);
  DenoiseGrads g{};
  real* dact0 = ARENA.get(Rq * C);
  if (dsg) stackBackward(m.decStack, s.ds.ctx, s.es.pair, s.ds.run, dact, dact0, nullptr, nullptr, nullptr, &dec->pre, dsg);
  else {
    real* dkCondDec = ARENA.zeros(Rk * C);
    g.dqCondDec = ARENA.zeros(Rq * C); g.dpairDec = ARENA.zeros(rows * Cp);
    stackBackward(m.decStack, s.ds.ctx, s.es.pair, s.ds.run, dact, dact0, g.dqCondDec, dkCondDec, g.dpairDec);
    scatter(b.q2k, dkCondDec, g.dqCondDec, C);
    if (m.decChai) { real* t = ARENA.get(Rq * C); lnBackward(m.decCondLn, s.ds.lc, g.dqCondDec, t, Rq, false); g.dqCondDec = t; }
  }
  real* ddproj = ARENA.zeros((size_t)n * C); g.dskip = ARENA.zeros(Rq * C);
  broadcastBwdK<<<blocks(Rq * C), 256, 0, STREAM>>>(dact0, b.t2q.idx, b.t2q.mask, s.es.qMask, ddproj, g.dskip, Rq, C, b.dense);
  real* dactn = ARENA.get((size_t)n * P); m.decBroadcast.backward(s.actn, ddproj, dactn, n);
  g.dtrOut = ARENA.get((size_t)n * P); lnBackward(m.outLn, s.lout, dactn, g.dtrOut, n, false);
  return g;
}
// dtok, the transformer input's gradient; dsF [n, Cs] the conditioning single's (added to); dpc the pair conditioning's
// (shared: the encoder's batch-shared gradients, summed here and backed by the caller once - dpc and dtrunkSingle then
// untouched by this sample)
inline void denoiseBackPre(Denoiser& m, const AtomBatch& b, const DenoiseSave& s, const DenoiseGrads& g, const real* trunkSingle,
                           const real* dtok, real* dsF, real* dpc, real* dtrunkSingle, real* dx, EncGrads* shared = nullptr) {
  const int n = s.n, Cs = m.cond.Cs, P = m.enc.perToken;
  const size_t atoms = (size_t)n * b.dense;
  // the encoder (its tokenAct = aggregated + snProj)
  real* dsnP = cloneR(dtok, (size_t)n * P);
  real* dscaled = dx ? ARENA.zeros(atoms * 3) : nullptr;
  if (shared) encSampleBackward(m.enc, b, s.es, trunkSingle, s.scaled, dtok, g.dskip, g.dpairDec, g.dqCondDec, dscaled, *shared);
  else encBackward(m.enc, b, s.es, trunkSingle, s.scaled, dtok, g.dskip, g.dpairDec, g.dqCondDec, dtrunkSingle, dpc, dscaled);
  if (dx) {                                          // scaled = x mask in
    real* t = ARENA.get(atoms * 3);
    scaleMaskK<<<blocks(atoms * 3), 256, 0, STREAM>>>(dscaled, b.atomMask, t, atoms, s.in);
    add(dx, t, atoms * 3);
  }
  real* dsn = ARENA.get((size_t)n * Cs); m.snProj.backward(s.sn, dsnP, dsn, n);
  if (m.hasSnLn) lnBackward(m.snLn, s.lsn, dsn, dsF, n, true); else add(dsF, dsn, (size_t)n * Cs);
}
// dout [atoms, 3] -> every parameter; dtrunkSingle, dtrunkPair, dtargetFeat, dx (added; any may be null)
inline void denoiseBackward(Denoiser& m, const AtomBatch& b, const DenoiseSave& s, const real* trunkSingle, const real* seqMask,
                            const real* dout, real* dtrunkSingle, real* dtrunkPair, real* dtargetFeat, real* dx) {
  Scratch scratch;
  const int n = s.n, Cs = m.cond.Cs, P = m.enc.perToken;
  const size_t nn = (size_t)n * n;
  DenoiseGrads g = denoiseBackPost(m, b, s, dout, dx);
  // the transformer
  real *dtok = ARENA.get((size_t)n * P), *dsF = ARENA.zeros((size_t)n * Cs), *dpc = ARENA.zeros(nn * m.cond.Cz);
  transformerBackward(m.tr, m.run, s.cs.sFinal, seqMask, s.cs.pairOut, g.dtrOut, dtok, dsF, dpc);
  denoiseBackPre(m, b, s, g, trunkSingle, dtok, dsF, dpc, dtrunkSingle, dx);
  condBackward(m.cond, s.cs, dpc, dsF, dtrunkSingle, dtrunkPair, dtargetFeat);
}

// S samples at once (OpenFold3's diffusion batch): one pair conditioning, each sample's own noise level in its single
// conditioning, its encoder and decoder, and ONE transformer pass over the S samples' tokens [S n] - the pair logits
// read by all. outs[k] = D(xs[k], sigmas[k]); the saves kept in the arena for denoiseBatchBackward.
struct DenoiseBatch {
  CondSave pair;                          // (sample 0's conditioning: its pair half everyone's)
  EncSave enc;                            // (the encoder's shared part: everyone's)
  DecPre dec;                             // (the decoder's)
  std::vector<DenoiseSave> per;
  real *tokS = nullptr, *condS = nullptr, *maskS = nullptr, *trOutS = nullptr;
  int S = 0, n = 0;
};
inline DenoiseBatch denoiseBatchForward(Denoiser& m, const AtomBatch& b, const real* trunkSingle, const real* trunkPair,
                                        const real* targetFeat, const real* seqMask, const std::vector<const real*>& xs,
                                        const std::vector<real>& sigmas, const int* ri, const int* ti, const int* asym,
                                        const int* ent, const int* sym, const std::vector<real*>& outs) {
  DenoiseBatch B; B.S = (int)xs.size(); B.n = b.tokens;
  const int n = b.tokens, P = m.enc.perToken, Cs = m.cond.Cs, S = B.S;
  std::optional<Range> r_; r_.emplace("dn.cond");
  B.pair = condForward(m.cond, trunkSingle, trunkPair, targetFeat, ri, ti, asym, ent, sym, sigmas[0], n);
  r_.reset(); r_.emplace("dn.encShared");
  B.enc = encShared(m.enc, b, trunkSingle, B.pair.pairOut);
  B.dec = decShared(m, b, B.enc);
  r_.reset();
  B.tokS = ARENA.get((size_t)S * n * P); B.condS = ARENA.get((size_t)S * n * Cs); B.maskS = ARENA.get((size_t)S * n);
  B.trOutS = ARENA.get((size_t)S * n * P);
  B.per.resize(S);
  for (int k = 0; k < S; ++k) {
    DenoiseSave& s = B.per[k];
    s.n = n; denoiseScales(s, sigmas[k]);
    { Range rc_("dn.condSingle"); s.cs = k == 0 ? B.pair : condSingleForward(m.cond, B.pair, sigmas[k]); }
    { Range rp_("dn.pre"); denoisePre(m, b, s, trunkSingle, xs[k], B.tokS + (size_t)k * n * P, &B.enc); }
    GK(cudaMemcpyAsync(B.condS + (size_t)k * n * Cs, s.cs.sFinal, (size_t)n * Cs * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(B.maskS + (size_t)k * n, seqMask, (size_t)n * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  {
    std::optional<Scratch> sc;   // (kept saves stay in the caller's scope, for the backward)
    if (!m.run.keep) sc.emplace();
    Range rt_("dn.transformer");
    transformerForward(m.tr, m.run, B.tokS, B.condS, B.maskS, B.pair.pairOut, B.trOutS, n, S);
  }
  for (int k = 0; k < S; ++k) { Range rq_("dn.post"); denoisePost(m, b, B.per[k], xs[k], B.trOutS + (size_t)k * n * P, outs[k], &B.dec); }
  return B;
}
// douts[k] -> every parameter; dtrunkSingle, dtrunkPair, dtargetFeat and dxs[k] (added; any may be null, dxs empty)
inline void denoiseBatchBackward(Denoiser& m, const AtomBatch& b, const DenoiseBatch& B, const real* trunkSingle,
                                 const std::vector<const real*>& douts, real* dtrunkSingle, real* dtrunkPair,
                                 real* dtargetFeat, const std::vector<real*>& dxs) {
  Scratch scratch;
  const int n = B.n, S = B.S, P = m.enc.perToken, Cs = m.cond.Cs;
  const size_t nn = (size_t)n * n;
  std::vector<DenoiseGrads> g(S);
  EncGrads eg = encGrads(m.enc, b);
  StackGrads dsg = stackGrads(m.decStack, atomShape(b));
  real* dtrOutS = ARENA.get((size_t)S * n * P);
  for (int k = 0; k < S; ++k) {
    { Range rb_("dnb.post"); g[k] = denoiseBackPost(m, b, B.per[k], douts[k], dxs.empty() ? nullptr : dxs[k], &B.dec, &dsg); }
    GK(cudaMemcpyAsync(dtrOutS + (size_t)k * n * P, g[k].dtrOut, (size_t)n * P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  real *dtokS = ARENA.get((size_t)S * n * P), *dsFS = ARENA.zeros((size_t)S * n * Cs), *dpc = ARENA.zeros(nn * m.cond.Cz);
  { Range rt_("dnb.transformer"); transformerBackward(m.tr, m.run, B.condS, B.maskS, B.pair.pairOut, dtrOutS, dtokS, dsFS, dpc); }
  for (int k = 0; k < S; ++k) {
    real* dsF = dsFS + (size_t)k * n * Cs;
    { Range rb_("dnb.pre"); denoiseBackPre(m, b, B.per[k], g[k], trunkSingle, dtokS + (size_t)k * n * P, dsF, dpc, dtrunkSingle, dxs.empty() ? nullptr : dxs[k], &eg); }
    { Range rc_("dnb.condSingle"); condBackward(m.cond, B.per[k].cs, nullptr, dsF, dtrunkSingle, nullptr, dtargetFeat); }     // (each sample's single half)
  }
  { Range re_("dnb.encShared"); decSharedBackward(m, b, B.enc, B.dec, dsg, eg); encSharedBackward(m.enc, b, B.enc, trunkSingle, eg, dtrunkSingle, dpc); }
  { Range rc_("dnb.cond"); condBackward(m.cond, B.pair, dpc, nullptr, nullptr, dtrunkPair, nullptr); }                  // (the one pair half)
}


// Every parameter of an atom encoder: the reference embedding under R ("atomReference", "targetFeat.reference"), the
// encoder under E. The trunk terms (a trunk single, a token pair), the positions' projection and rf3's chirality are
// each taken where the bundle has them - target_feat's encoder has none of them.
inline void forEachParam(DiffEncoder& e, const std::string& E, const std::string& R, int nb, bool perBlock, bool upGate,
                         bool kqNorm, bool noTr2, const std::function<bool(const std::string&)>& has,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  auto ln = [&](const std::string& n, LayerNorm& l, int C) {
    l.C = C; f(n + "Scale", l.g, C); if (has(n + "Offset")) f(n + "Offset", l.b, C);
  };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; l.w.fanIn = K; f(n, l.w, (size_t)K * N); };
  const int C = e.C, Cp = e.Cp;
  f(R + ".embedRefPos", e.wPos, (size_t)3 * C); f(R + ".embedRefMask", e.wMask, C);
  f(R + ".embedRefElement", e.wElem, (size_t)128 * C); f(R + ".embedRefCharge", e.wCharge, C);
  f(R + ".embedRefAtomName", e.wName, (size_t)256 * C);
  e.hasBias = has(R + ".embedAtomFeaturesBias"); if (e.hasBias) f(R + ".embedAtomFeaturesBias", e.bias, C);
  e.hasTrunkSingle = has(E + ".embedTrunkSingleCond");
  if (e.hasTrunkSingle) { ln(E + ".lnormTrunkSingleCond", e.tsLn, e.Cst); lin(E + ".embedTrunkSingleCond", e.tsProj, e.Cst, C); }
  lin(E + ".singleToPairCondRow", e.row, C, Cp); lin(E + ".singleToPairCondCol", e.col, C, Cp);
  e.hasTrunkPair = has(E + ".embedTrunkPairCond");
  if (e.hasTrunkPair) { ln(E + ".lnormTrunkPairCond", e.tpLn, e.Cz); lin(E + ".embedTrunkPairCond", e.tpProj, e.Cz, Cp); }
  if (e.chai) {
    f(E + ".embedAtomPairFeat", e.wPairFeat, (size_t)14 * Cp); f(E + ".embedAtomPairFeatBias", e.pairFeatBias, Cp);
    lin(E + ".pairMlp1", e.mlp1, Cp, Cp); lin(E + ".pairMlp2", e.mlp2, Cp, Cp);
  } else {
    f(E + ".embedPairOffsets", e.wOff, (size_t)3 * Cp); f(E + ".embedPairDistances", e.wDist, Cp);
    f(E + ".embedPairOffsetsValid", e.wValid, Cp);
    lin(E + ".pairMlp1", e.mlp1, Cp, Cp); lin(E + ".pairMlp2", e.mlp2, Cp, Cp); lin(E + ".pairMlp3", e.mlp3, Cp, Cp);
  }
  e.hasPositions = has(E + ".atomPositionsToFeatures");
  if (e.hasPositions) lin(E + ".atomPositionsToFeatures", e.posProj, 3, C);
  e.hasChiral = has(E + ".atomChiralToFeatures"); if (e.hasChiral) lin(E + ".atomChiralToFeatures", e.chiralProj, 3, C);
  lin(E + ".projectAtomFeaturesForAggr", e.aggr, C, e.perToken);
  e.stack.C = C; e.stack.Cp = Cp; e.stack.H = e.H; e.stack.D = e.D;
  forEachParam(e.stack, E, nb, perBlock, e.chai, upGate, f, kqNorm, noTr2);
}

// Every parameter of the denoiser under its bundle names. `has` answers for the optional tensors (LayerNorm offsets
// some families drop, a projection bias, the openfold3 lineage's split/relpe forms); the dims are the Denoiser's,
// set by the caller (cond.{Cz,Cs,Czt,Cst,F,nc}, enc.{C,Cp,H,D,perToken,Cst,Cz}, tr.*, decStack.*), and nbEnc/nbDec the
// two atom stacks' depths.
inline void forEachParam(Denoiser& m, int nbEnc, int nbDec, int nSuper, bool perBlock, bool upGate, bool kqNorm, bool noTr2,
                         bool trUpGate, bool trKqNorm, const std::function<bool(const std::string&)>& has,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  auto ln = [&](const std::string& n, LayerNorm& l, int C) {
    l.C = C; f(n + "Scale", l.g, C); if (has(n + "Offset")) f(n + "Offset", l.b, C);
  };
  auto lin = [&](const std::string& n, Linear& l, int K, int N, const std::string& bias = "") {
    l.K = K; l.N = N; l.hasBias = !bias.empty() && has(bias); l.w.fanIn = K; f(n, l.w, (size_t)K * N); if (l.hasBias) f(bias, l.b, N);
  };
  auto gt = [&](const std::string& n, GatedTransition& t, int C) {
    t.C = C; t.F = 2 * C; ln(n + ".ffwLayerNorm", t.ln, C);
    lin(n + ".ffwTransition1", t.t1, C, 4 * C); lin(n + ".ffwTransition2", t.t2, 2 * C, C);
  };
  const std::string P = "diffusion.conditioning", E = "diffusion.encoder", R = "atomReference", Dd = "diffusion.decoder";
  DiffCond& c = m.cond;
  if (c.chai) {
    c.structW.K = 163; c.structW.N = c.Cz; c.structW.hasBias = true;
    f(P + ".structurePairWeights", c.structW.w, (size_t)163 * c.Cz); f(P + ".structurePairBias", c.structW.b, c.Cz);
    c.hasStructBond = has(P + ".structureBondWeights"); if (c.hasStructBond) f(P + ".structureBondWeights", c.structBond, c.Cz);
    ln(P + ".pairCondFinalNorm", c.pairFinal, c.Cz);
  }
  c.split = !c.chai && has(P + ".zTrunkProjection"); c.relpe = !c.chai && !c.split && has(P + ".relpeProjection");
  if (c.split) { ln(P + ".zTrunkNorm", c.zTrunkLn, c.Czt); lin(P + ".zTrunkProjection", c.zTrunkProj, c.Czt, c.Cz); }
  if (c.split || c.relpe) lin(P + ".relpeProjection", c.relpeProj, c.rel, c.Cz);
  ln(P + ".pairCondInitialNorm", c.pairLn, c.width());
  lin(P + ".pairCondInitialProjection", c.pairProj, c.width(), c.Cz);
  for (int k = 0; k < 2; ++k) gt(P + ".pairTransitions." + std::to_string(k), c.pairT[k], c.Cz);
  ln(P + ".singleCondInitialNorm", c.singleLn, c.swidth());
  lin(P + ".singleCondInitialProjection", c.singleProj, c.swidth(), c.Cs, P + ".singleCondInitialProjectionBias");
  f(P + ".fourierWeight", c.fw, c.nc); f(P + ".fourierBias", c.fb, c.nc);
  ln(P + ".noiseEmbeddingInitialNorm", c.noiseLn, c.nc);
  lin(P + ".noiseEmbeddingInitialProjection", c.noiseProj, c.nc, c.Cs);
  for (int k = 0; k < 2; ++k) gt(P + ".singleTransitions." + std::to_string(k), c.singleT[k], c.Cs);
  c.hasSingleFinal = has(P + ".singleCondFinalNormScale");
  if (c.hasSingleFinal) ln(P + ".singleCondFinalNorm", c.singleFinal, c.Cs);
  // the encoder
  DiffEncoder& e = m.enc;
  forEachParam(e, E, R, nbEnc, perBlock, upGate, kqNorm, noTr2, has, f);
  const int C = e.C;
  // the transformer and its two norms
  if (has("diffusion.singleCondEmbeddingNormScale")) { m.hasSnLn = true; ln("diffusion.singleCondEmbeddingNorm", m.snLn, c.Cs); }
  else m.hasSnLn = false;
  lin("diffusion.singleCondEmbeddingProjection", m.snProj, c.Cs, e.perToken);
  forEachParam(m.tr, "diffusion.transformer", nSuper, trUpGate, trKqNorm, f);
  ln("diffusion.outputNorm", m.outLn, e.perToken);
  // the decoder
  lin(Dd + ".projectTokenFeaturesForBroadcast", m.decBroadcast, e.perToken, C);
  forEachParam(m.decStack, Dd, nbDec, perBlock, m.decChai, upGate, f, kqNorm, noTr2);
  if (m.decChai) ln(Dd + ".postAtomCondLayerNorm", m.decCondLn, C);
  ln(Dd + ".atomFeaturesLayerNorm", m.decLn, C);
  lin(Dd + ".atomFeaturesToPositionUpdate", m.decUpdate, C, 3);
}
}  // namespace af3
}  // namespace grad
