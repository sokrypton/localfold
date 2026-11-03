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
// boltz2's atom-only target_feat: the encoder's 384 columns plus six summed token terms (cuda/af3's targetFeatSumK) -
// the residue type, the profile and mean deletion, the molecule type, the method (x-ray), the modified flag, and Boltz's
// DNA unknown (DN) rows where the bundle has them
struct AtomSum {
  Param res, prof, mol, method, mod, dnRes, dnProf;     // [31, C], [32, C], [4, C], [M, C], [2, C], [C], [C]
  bool dn = false;
};
struct TokenFlags { const int *isDna, *isRna, *isLigand, *isModified; };   // (each may be null: false everywhere)
__global__ void atomSumK(real* tf, const int* aatype, const real* profile, const real* delMean, TokenFlags fl,
                         const real* wRes, const real* wProf, const real* wMol, const real* wMethod, const real* wMod,
                         const real* dnRes, const real* dnProf, int tokens, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)tokens * C) return;
  int token = (int)(t / C), c = (int)(t % C);
  real v = 0;
  int a = aatype[token];
  if (a >= 0 && a < 31) v += wRes[a * C + c];
  for (int k = 0; k < 31; ++k) v += profile[token * 31 + k] * wProf[k * C + c];
  v += delMean[token] * wProf[31 * C + c];
  bool dna = fl.isDna && fl.isDna[token];
  if (dnRes && dna) v += (a == 30 ? dnRes[c] : 0) + profile[token * 31 + 30] * dnProf[c];
  int mol = fl.isLigand && fl.isLigand[token] ? 3 : fl.isRna && fl.isRna[token] ? 2 : dna ? 1 : 0;
  v += wMol[mol * C + c] + wMethod[1 * C + c] + wMod[(fl.isModified && fl.isModified[token] ? 1 : 0) * C + c];
  tf[t] += v;
}
// one thread a (table row, channel) over the tokens: the residue, molecule, method, modified and DN tables
__global__ void atomSumBwdK(const real* d, const int* aatype, const real* profile, TokenFlags fl, real* gRes, real* gMol,
                           real* gMethod, real* gMod, real* gDnRes, real* gDnProf, int tokens, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int rows = 31 + 4 + 1 + 2 + 2;
  if (t >= (size_t)rows * C) return;
  int k = (int)(t / C), c = (int)(t % C);
  real acc = 0;
  for (int token = 0; token < tokens; ++token) {
    real g = d[(size_t)token * C + c];
    int a = aatype[token];
    bool dna = fl.isDna && fl.isDna[token];
    int mol = fl.isLigand && fl.isLigand[token] ? 3 : fl.isRna && fl.isRna[token] ? 2 : dna ? 1 : 0;
    int mod = fl.isModified && fl.isModified[token] ? 1 : 0;
    if (k < 31) { if (a == k) acc += g; }
    else if (k < 35) { if (mol == k - 31) acc += g; }
    else if (k == 35) acc += g;
    else if (k < 38) { if (mod == k - 36) acc += g; }
    else if (k == 38) { if (dna && a == 30) acc += g; }
    else if (dna) acc += profile[token * 31 + 30] * g;
  }
  if (k < 31) gRes[(size_t)k * C + c] += acc;
  else if (k < 35) gMol[(size_t)(k - 31) * C + c] += acc;
  else if (k == 35) gMethod[(size_t)1 * C + c] += acc;
  else if (k < 38) gMod[(size_t)(k - 36) * C + c] += acc;
  else if (gDnRes) (k == 38 ? gDnRes : gDnProf)[c] += acc;
}
// chai-1's token embedding: feat = one_hot(aatype) W + b + [profile | deletion mean] W' (+ DX on a DNA unknown) + esm W_e;
// target_feat = [the encoder's columns | feat] W_trunk, and the diffusion module's own [..] W_structure
struct ChaiToken {
  Linear tok, esm, trunk, structure;     // [31, C] + b, [E, C], [P + C, out] each
  Param prof, dx;                         // [32, C], [C]
  bool hasDx = false;
  int C = 0, E = 0;
};
inline void forEachParam(ChaiToken& t, int P, int C, int E, int out, int outS, bool dx,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string S = "targetFeat.encoder.chaiToken.";
  t.C = C; t.E = E; t.hasDx = dx;
  t.tok.K = 31; t.tok.N = C; f(S + "tokenFeatureWeights", t.tok.w, (size_t)31 * C); f(S + "tokenFeatureBias", t.tok.b, C);
  f(S + "msaProfileWeights", t.prof, (size_t)32 * C);
  if (dx) f(S + "dx", t.dx, C);
  t.esm.K = E; t.esm.N = C; t.esm.hasBias = false; f(S + "esmWeights", t.esm.w, (size_t)E * C);
  t.trunk.K = P + C; t.trunk.N = out; t.trunk.hasBias = false; f(S + "singleProjInTrunk", t.trunk.w, (size_t)(P + C) * out);
  t.structure.K = P + C; t.structure.N = outS; t.structure.hasBias = false; f(S + "singleProjInStructure", t.structure.w, (size_t)(P + C) * outS);
}
__global__ void chaiTokFeatK(const int* aatype, const real* profile, const real* delMean, const real* Wp, const real* dx,
                             const int* isDna, real* feat, int tokens, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)tokens * C) return;
  int token = (int)(t / C), c = (int)(t % C);
  int a = aatype[token];
  real v = 0;
  if (dx && isDna && isDna[token] && a == 30) v += dx[c];
  for (int k = 0; k < 31; ++k) v += profile[(size_t)token * 31 + k] * Wp[(size_t)k * C + c];
  feat[t] += v + delMean[token] * Wp[(size_t)31 * C + c];
}
__global__ void oneHot31K(const int* aatype, real* oh, int n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)n * 31) oh[t] = aatype[t / 31] == (int)(t % 31) ? (real)1 : 0;
}
__global__ void catColsK(const real* a, int wa, const real* b, int wb, real* out, size_t rows, bool back, real* da, real* db) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int W = wa + wb;
  if (t >= rows * W) return;
  size_t r = t / W; int c = t % W;
  if (!back) out[t] = c < wa ? a[r * wa + c] : b[r * wb + c - wa];
  else if (c < wa) da[r * wa + c] = out[t]; else db[r * wb + c - wa] = out[t];
}
__global__ void dnaUnknownSumK(const real* d, const int* aatype, const int* isDna, real* g, int tokens, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  real acc = 0;
  for (int t = 0; t < tokens; ++t) if (isDna && isDna[t] && aatype[t] == 30) acc += d[(size_t)t * C + c];
  g[c] += acc;
}
struct TargetFeat { DiffEncoder enc; bool atomOnly = false, chai = false; AtomSum sum; ChaiToken ct; TokenFlags flags{}; };
struct TargetFeatSave { EncSave es; real *atom, *oh, *cat; const int* aatype; const real *profile, *delMean, *esm; };
// boltz2's six tables, under targetFeat.encoder.targetFeatSum.
inline void forEachParam(AtomSum& a, int C, int methods, bool dn, const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string S = "targetFeat.encoder.targetFeatSum.";
  f(S + "resType", a.res, (size_t)31 * C); f(S + "msaProfile", a.prof, (size_t)32 * C); f(S + "molType", a.mol, (size_t)4 * C);
  f(S + "method", a.method, (size_t)methods * C); f(S + "modified", a.mod, (size_t)2 * C);
  a.dn = dn;
  if (dn) { f(S + "dnResType", a.dnRes, C); f(S + "dnMsaProfile", a.dnProf, C); }
}
// (chai-1: esm [n, E] its ESM2 embeddings, outStructure the diffusion module's target_feat - null: not wanted)
inline TargetFeatSave targetFeatForward(TargetFeat& m, const AtomBatch& b, const int* aatype, const real* profile,
                                        const real* delMean, real* out, const real* esm = nullptr, real* outStructure = nullptr) {
  TargetFeatSave s{};
  s.aatype = aatype; s.profile = profile; s.delMean = delMean; s.esm = esm;
  const int n = b.tokens, P = m.enc.perToken;
  s.atom = ARENA.get((size_t)n * P);
  s.es = encForward(m.enc, b, nullptr, nullptr, nullptr, nullptr, s.atom);
  if (m.chai) {
    const int C = m.ct.C;
    s.oh = ARENA.get((size_t)n * 31); oneHot31K<<<blocks((size_t)n * 31), 256, 0, STREAM>>>(aatype, s.oh, n);
    real* feat = ARENA.get((size_t)n * C);
    m.ct.tok.forward(s.oh, feat, n);
    chaiTokFeatK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(aatype, profile, delMean, m.ct.prof.w, m.ct.hasDx ? m.ct.dx.w : nullptr,
                                                           m.flags.isDna, feat, n, C);
    m.ct.esm.forward(esm, feat, n, 1);
    s.cat = ARENA.get((size_t)n * (P + C));
    catColsK<<<blocks((size_t)n * (P + C)), 256, 0, STREAM>>>(s.atom, P, feat, C, s.cat, n, false, nullptr, nullptr);
    m.ct.trunk.forward(s.cat, out, n);
    if (outStructure) m.ct.structure.forward(s.cat, outStructure, n);
    return s;
  }
  if (m.atomOnly) {
    GK(cudaMemcpyAsync(out, s.atom, (size_t)n * P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    atomSumK<<<blocks((size_t)n * P), 256, 0, STREAM>>>(out, aatype, profile, delMean, m.flags, m.sum.res.w, m.sum.prof.w, m.sum.mol.w,
      m.sum.method.w, m.sum.mod.w, m.sum.dn ? m.sum.dnRes.w : nullptr, m.sum.dn ? m.sum.dnProf.w : nullptr, n, P);
    return s;
  }
  targetFeatCatK<<<blocks((size_t)n * (63 + P)), 256, 0, STREAM>>>(aatype, profile, delMean, s.atom, out, n, P);
  return s;
}
inline void targetFeatBackward(TargetFeat& m, const AtomBatch& b, const TargetFeatSave& s, const real* dout,
                               const real* doutStructure = nullptr, real* desm = nullptr) {
  Scratch scratch;
  const int n = b.tokens, P = m.enc.perToken, C = m.enc.C, Cp = m.enc.Cp;
  const size_t Rq = (size_t)b.subsets * b.Q, rows = Rq * b.K;
  real* datom = ARENA.get((size_t)n * P);
  if (m.chai) {
    const int Ct = m.ct.C;
    real* dcat = ARENA.get((size_t)n * (P + Ct));
    m.ct.trunk.backward(s.cat, dout, dcat, n);
    if (doutStructure) m.ct.structure.backward(s.cat, doutStructure, dcat, n, true);
    real* dfeat = ARENA.get((size_t)n * Ct);
    catColsK<<<blocks((size_t)n * (P + Ct)), 256, 0, STREAM>>>(nullptr, P, nullptr, Ct, dcat, n, true, datom, dfeat);
    m.ct.tok.backward(s.oh, dfeat, nullptr, n);
    mmAT(s.profile, dfeat, m.ct.prof.g, n, 31, Ct);
    mmAT(s.delMean, dfeat, m.ct.prof.g + (size_t)31 * Ct, n, 1, Ct);
    if (m.ct.hasDx) dnaUnknownSumK<<<blocks(Ct, 64), 64, 0, STREAM>>>(dfeat, s.aatype, m.flags.isDna, m.ct.dx.g, n, Ct);
    m.ct.esm.backward(s.esm, dfeat, desm, n, desm != nullptr);
  } else if (m.atomOnly) {
    GK(cudaMemcpyAsync(datom, dout, (size_t)n * P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    atomSumBwdK<<<blocks((size_t)40 * P), 256, 0, STREAM>>>(dout, s.aatype, s.profile, m.flags, m.sum.res.g, m.sum.mol.g, m.sum.method.g,
      m.sum.mod.g, m.sum.dn ? m.sum.dnRes.g : nullptr, m.sum.dn ? m.sum.dnProf.g : nullptr, n, P);
    mmAT(s.profile, dout, m.sum.prof.g, n, 31, P);                       // (rows 0..30: the profile)
    mmAT(s.delMean, dout, m.sum.prof.g + (size_t)31 * P, n, 1, P);      // (row 31: the mean deletion)
  } else targetFeatAtomBwdK<<<blocks((size_t)n * P), 256, 0, STREAM>>>(dout, datom, n, P);
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
  // chai-1's (trunk.dialect.recycleFromInit / chaiMsaFeatures): its relative encoding (134 position columns and a bias,
  // the chain-entity table, the declared bonds), its first pass recycling z_init and s_init themselves, its 41 MSA
  // columns with a bias and the single's projection; set chai before forEachParam
  bool chai = false, hasCe = false, hasChaiBond = false;
  Param posBias, ce, chaiBond, msaBias;
};
inline void forEachParam(TrunkEmbedder& e, const std::function<bool(const std::string&)>& has,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string E = "trunk.embedder.";
  auto ln = [&](const std::string& n, LayerNorm& l, int C) {
    l.C = C; f(n + "Scale", l.g, C); if (has(n + "Offset")) f(n + "Offset", l.b, C);
  };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; f(n, l.w, (size_t)K * N); };
  if (e.chai) e.rel = 134;
  const int src = e.fromSingle ? e.Cs : e.F;
  lin(E + "singleActivations", e.single, e.F, e.Cs);
  lin(E + "leftSingle", e.left, src, e.C); lin(E + "rightSingle", e.right, src, e.C);
  ln(E + "prevEmbeddingNorm", e.prevPairLn, e.C); lin(E + "prevEmbedding", e.prevPair, e.C, e.C);
  lin(E + "positionActivations", e.relpos, e.rel, e.C);
  e.hasBond = has(E + "bondEmbedding"); if (e.hasBond) f(E + "bondEmbedding", e.bondW, e.C);
  e.hasBondType = has(E + "tokenBondsTypeEmbed");
  if (e.hasBondType) { f(E + "tokenBondsTypeEmbed", e.bondTypeTable, (size_t)7 * e.C); f(E + "contactEncodingUnspecified", e.unspecified, e.C); }
  ln(E + "prevSingleEmbeddingNorm", e.prevSingleLn, e.Cs); lin(E + "prevSingleEmbedding", e.prevSingle, e.Cs, e.Cs);
  if (e.chai) {
    f(E + "positionActivationsBias", e.posBias, e.C);
    e.hasCe = has(E + "chaiChainEntity"); if (e.hasCe) f(E + "chaiChainEntity", e.ce, (size_t)9 * e.C);
    e.hasChaiBond = has(E + "chaiBondTrunk"); if (e.hasChaiBond) f(E + "chaiBondTrunk", e.chaiBond, e.C);
    e.msaWidth = 41;
    lin(E + "extraMsaTargetFeat", e.msaFromTarget, e.Cs, e.Cm);
    f(E + "msaActivations", e.msaW, (size_t)41 * e.Cm);
    f(E + "msaActivationsBias", e.msaBias, e.Cm);
    return;
  }
  lin(E + "extraMsaTargetFeat", e.msaFromTarget, e.F, e.Cm);
  f(E + "msaActivations", e.msaW, (size_t)e.msaWidth * e.Cm);
}
struct EmbedBatch {
  const int *ri, *ti, *asym, *ent, *sym;      // [n]
  const real *bonds, *bondOrders;              // [n, n] (either may be null)
  const int* msaRows; const real* deletion;    // [S, n]
  int n, S;
  // chai-1's: the declared bonds [n, n], its MSA's 41 feature columns [S n, 41] (chaiMsaFeatures, inputs), and whether
  // this is the first pass (z_init and s_init recycled through themselves)
  const real* declaredBonds = nullptr;
  const real* chaiMsa = nullptr;
  bool firstPass = true;
};
// chai-1's relative encoding one-hots: [n n, 134] (the residue offset 0..66, the token offset 67..133) and [n n, 9]
// (the chain 0..5, the entity 6..8) - cuda/af3's chaiRelativeEncodingK
__global__ void chaiRelOneHotK(const int* ri, const int* ti, const int* asym, const int* ent, const int* sym, real* pos,
                               real* ce, int n) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)n * n) return;
  int i = ij / n, j = ij % n;
  auto clip = [](int v) { return v < 0 ? 0 : (v > 65 ? 65 : v); };
  bool same = asym[i] == asym[j];
  int rss = same ? clip(ri[i] - ri[j] + 33) : 66;
  int rts = same && ri[i] == ri[j] ? clip(ti[i] - ti[j] + 32) : 66;
  for (int k = 0; k < 134; ++k) pos[ij * 134 + k] = (k == rss || k == 67 + rts) ? (real)1 : 0;
  if (!ce) return;
  int relE = ent[i] - ent[j], relS = sym[i] - sym[j] + 2;
  int rchain = relE != 0 ? 5 : (relS < 0 ? 0 : relS > 4 ? 4 : relS);
  int rent = relE + 1 < 0 ? 0 : relE + 1 > 2 ? 2 : relE + 1;
  for (int k = 0; k < 9; ++k) ce[ij * 9 + k] = (k == rchain || k == 6 + rent) ? (real)1 : 0;
}
// chai-1's MSA's 41 columns: paired | source one-hot (5) | atan(d/3) 2/pi | clip(d) | the residue class one-hot (32)
// (cuda/af3's chaiMsaEmbedK, its weights' rows in order) - an input, built once from the alignment
__global__ void chaiMsaFeatK(const int* rows, const float* deletion, const float* msaMask, const int* asymId, const int* chainHit,
                             int chains, int pairedRows, const int* isLigand, const int* source, real* F, size_t count, int n) {
  size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count) return;
  int token = (int)(row % n); size_t s = row / n;
  real* f = F + row * 41;
  for (int k = 0; k < 41; ++k) f[k] = 0;
  bool paired = s > 0 && s < (size_t)pairedRows && msaMask[row] != 0 && chainHit[s * chains + asymId[token]] && chainHit[s * chains + asymId[0]];
  float d = deletion[row];
  int code = rows[row];
  if (isLigand && isLigand[token]) code = s == 0 ? 20 : 31;
  int src = s == 0 ? 4 : source ? source[s] : 2;
  f[0] = paired ? 1 : 0;
  f[1 + src] = 1;
  f[7] = atanf(d / 3.f) * (2.f / 3.14159265358979f);
  f[8] = fminf(fmaxf(d, 0.f), 1.f);
  if (code >= 0 && code < 32) f[9 + code] = 1;
}
// msa = F W + b + fromSingle[token], and back (dfs [n, C] summed over the rows)
__global__ void chaiMsaAddK(real* msa, const real* bias, const real* fs, size_t count, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count * C) return;
  int c = t % C; size_t row = t / C;
  msa[t] += bias[c] + fs[(size_t)(row % n) * C + c];
}
__global__ void rowsSumK(const real* d, real* out, size_t count, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int c = t % C, tok = t / C;
  real s = 0;
  for (size_t row = tok; row < count; row += n) s += d[row * C + c];
  out[t] = s;
}
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
struct EmbedSave { real *src, *sInit, *l, *r, *ppn, *rel, *psn, *ft, *ceOh, *z0, *s0; LnSave lpp, lps; };
// targetFeat [n, F], prevPair [n*n, C], prevSingle [n, Cs] -> pair (z_init, before the template term), single, msa
inline EmbedSave trunkEmbedForward(TrunkEmbedder& e, const EmbedBatch& b, const real* tf, const real* prevPair,
                                   const real* prevSingle, real* pair, real* single, real* msa) {
  EmbedSave s{};
  const int n = b.n; const size_t nn = (size_t)n * n;
  if (e.chai) {
    s.src = const_cast<real*>(tf);
    s.l = ARENA.get((size_t)n * e.C); e.left.forward(tf, s.l, n);
    s.r = ARENA.get((size_t)n * e.C); e.right.forward(tf, s.r, n);
    outerSumBondK<<<blocks(nn * e.C), 256, 0, STREAM>>>(s.l, s.r, nullptr, nullptr, nullptr, nullptr, nullptr, pair, n, e.C);
    s.rel = ARENA.get(nn * 134); s.ceOh = e.hasCe ? ARENA.get(nn * 9) : nullptr;
    chaiRelOneHotK<<<blocks(nn), 256, 0, STREAM>>>(b.ri, b.ti, b.asym, b.ent, b.sym, s.rel, s.ceOh, n);
    e.relpos.forward(s.rel, pair, nn, 1);
    biasK<<<blocks(nn * e.C), 256, 0, STREAM>>>(pair, e.posBias.w, nn, e.C);
    if (e.hasCe) mm(s.ceOh, e.ce.w, pair, (int)nn, 9, e.C, 1);
    if (e.hasChaiBond && b.declaredBonds) mm(b.declaredBonds, e.chaiBond.w, pair, (int)nn, 1, e.C, 1);
    // the recycled pair: z_init itself on the first pass (before the bond terms, which the port adds after it)
    const real* pp = prevPair;
    if (b.firstPass) { s.z0 = cloneR(pair, nn * e.C); pp = s.z0; }
    s.ppn = ARENA.get(nn * e.C); s.lpp = lnForward(e.prevPairLn, pp, s.ppn, nn);
    e.prevPair.forward(s.ppn, pair, nn, 1);
    {
      Scratch sc;
      real* t = ARENA.get(nn * e.C);
      outerSumBondK<<<blocks(nn * e.C), 256, 0, STREAM>>>(ARENA.zeros((size_t)n * e.C), ARENA.zeros((size_t)n * e.C), b.bonds,
          e.hasBond && b.bonds ? e.bondW.w : nullptr, b.bondOrders, e.hasBondType ? e.bondTypeTable.w : nullptr,
          e.hasBondType ? e.unspecified.w : nullptr, t, n, e.C);
      add(pair, t, nn * e.C);
    }
    // the single: s_init recycled through itself on the first pass
    e.single.forward(tf, single, n);
    const real* ps = prevSingle;
    if (b.firstPass) { s.s0 = cloneR(single, (size_t)n * e.Cs); ps = s.s0; }
    s.psn = ARENA.get((size_t)n * e.Cs); s.lps = lnForward(e.prevSingleLn, ps, s.psn, n);
    e.prevSingle.forward(s.psn, single, n, 1);
    // the MSA, from the single
    s.ft = ARENA.get((size_t)n * e.Cm); e.msaFromTarget.forward(single, s.ft, n);
    const size_t rows = (size_t)b.S * n;
    mm(b.chaiMsa, e.msaW.w, msa, (int)rows, 41, e.Cm);
    chaiMsaAddK<<<blocks(rows * e.Cm), 256, 0, STREAM>>>(msa, e.msaBias.w, s.ft, rows, n, e.Cm);
    return s;
  }
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
  if (e.chai) {
    real* ds = ARENA.zeros((size_t)n * e.Cs);
    if (dsingle) add(ds, dsingle, (size_t)n * e.Cs);
    if (dmsa) {
      const size_t rows = (size_t)b.S * n;
      mmAT(b.chaiMsa, dmsa, e.msaW.g, (int)rows, 41, e.Cm);
      colSum(e.msaBias.g, dmsa, rows, e.Cm);
      real* dft = ARENA.get((size_t)n * e.Cm);
      rowsSumK<<<blocks((size_t)n * e.Cm), 256, 0, STREAM>>>(dmsa, dft, rows, n, e.Cm);
      // the single the MSA read is the finished one: its projection's input
      real* sFin = ARENA.get((size_t)n * e.Cs);
      e.single.forward(tf, sFin, n); e.prevSingle.forward(s.psn, sFin, n, 1);
      e.msaFromTarget.backward(sFin, dft, ds, n, true);
    }
    {   // single = s_init + W LN(s_init | prevSingle)
      real* dpsn = ARENA.get((size_t)n * e.Cs); e.prevSingle.backward(s.psn, ds, dpsn, n);
      real* dsInit = cloneR(ds, (size_t)n * e.Cs);
      if (b.firstPass) lnBackward(e.prevSingleLn, s.lps, dpsn, dsInit, n, true);
      else if (dprevSingle) lnBackward(e.prevSingleLn, s.lps, dpsn, dprevSingle, n, true);
      else { real* t = ARENA.get((size_t)n * e.Cs); lnBackward(e.prevSingleLn, s.lps, dpsn, t, n, false); }
      e.single.backward(tf, dsInit, dtfl, n, true);
    }
    if (dpair) {
      // the bond terms read the finished pair's gradient; z_init (outer, relative) also the recycle's through it
      real *dl = ARENA.get((size_t)n * e.C), *dr = ARENA.get((size_t)n * e.C);
      outerSumBondBwdK<<<blocks((size_t)n * e.C), 256, 0, STREAM>>>(dpair, b.bonds, b.bondOrders, dl, dr, e.hasBond && b.bonds ? e.bondW.g : nullptr,
          e.hasBondType ? e.bondTypeTable.g : nullptr, e.hasBondType ? e.unspecified.g : nullptr, n, e.C);
      real* dppn = ARENA.get(nn * e.C); e.prevPair.backward(s.ppn, dpair, dppn, nn);
      real* dz0 = cloneR(dpair, nn * e.C);
      if (b.firstPass) lnBackward(e.prevPairLn, s.lpp, dppn, dz0, nn, true);
      else if (dprevPair) lnBackward(e.prevPairLn, s.lpp, dppn, dprevPair, nn, true);
      else { real* t = ARENA.get(nn * e.C); lnBackward(e.prevPairLn, s.lpp, dppn, t, nn, false); }
      e.relpos.backward(s.rel, dz0, nullptr, nn);
      colSum(e.posBias.g, dz0, nn, e.C);
      if (e.hasCe) mmAT(s.ceOh, dz0, e.ce.g, (int)nn, 9, e.C);
      if (e.hasChaiBond && b.declaredBonds) mmAT(b.declaredBonds, dz0, e.chaiBond.g, (int)nn, 1, e.C);
      outerSumBondBwdK<<<blocks((size_t)n * e.C), 256, 0, STREAM>>>(dz0, nullptr, nullptr, dl, dr, nullptr, nullptr, nullptr, n, e.C);
      e.left.backward(tf, dl, dtfl, n, true);
      e.right.backward(tf, dr, dtfl, n, true);
    }
    if (dtf) add(dtf, dtfl, (size_t)n * e.F);
    return;
  }
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
// (chai-1's MLP distogram: h = half(gelu(hidden(LN(pair)) + b)))
struct DistogramHead { Linear half, hidden; LayerNorm ln; bool mlp = false; int bins = 64, C = 0, Hd = 0; real scale = 1; };
inline void forEachParam(DistogramHead& d, const std::function<bool(const std::string&)>& has,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string D = "trunk.distogram.";
  d.mlp = has(D + "hidden");
  int in = d.C;
  if (d.mlp) {
    d.ln.C = d.C; f(D + "inputLayerNormScale", d.ln.g, d.C); f(D + "inputLayerNormOffset", d.ln.b, d.C);
    d.hidden.K = d.C; d.hidden.N = d.Hd; f(D + "hidden", d.hidden.w, (size_t)d.C * d.Hd); f(D + "hiddenBias", d.hidden.b, d.Hd);
    in = d.Hd;
  }
  d.half.K = in; d.half.N = d.bins; d.half.hasBias = has(D + "halfLogitsBias");
  f(D + "halfLogits", d.half.w, (size_t)in * d.bins);
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
  real* h = ARENA.get(nn * d.bins);
  if (d.mlp) {
    Scratch sc;
    real *xn = ARENA.get(nn * d.C), *pre = ARENA.get(nn * d.Hd), *a = ARENA.get(nn * d.Hd);
    lnForward(d.ln, pair, xn, nn); d.hidden.forward(xn, pre, nn);
    geluK<<<blocks(nn * d.Hd), 256, 0, STREAM>>>(a, pre, nn * d.Hd);
    d.half.forward(a, h, nn);
  } else d.half.forward(pair, h, nn);
  symScaleK<<<blocks(nn * d.bins), 256, 0, STREAM>>>(h, logits, n, d.bins, d.scale);
  return h;
}
inline void distogramBackward(DistogramHead& d, const real* pair, int n, const real* dlogits, real* dpair) {
  Scratch scratch;
  const size_t nn = (size_t)n * n;
  real* dh = ARENA.get(nn * d.bins);
  symScaleK<<<blocks(nn * d.bins), 256, 0, STREAM>>>(dlogits, dh, n, d.bins, d.scale);
  if (!d.mlp) { d.half.backward(pair, dh, dpair, nn, true); return; }
  real *xn = ARENA.get(nn * d.C), *pre = ARENA.get(nn * d.Hd), *a = ARENA.get(nn * d.Hd);
  LnSave ls = lnForward(d.ln, pair, xn, nn); d.hidden.forward(xn, pre, nn);
  geluK<<<blocks(nn * d.Hd), 256, 0, STREAM>>>(a, pre, nn * d.Hd);
  real* da = ARENA.get(nn * d.Hd); d.half.backward(a, dh, da, nn);
  geluBwdK<<<blocks(nn * d.Hd), 256, 0, STREAM>>>(da, pre, nn * d.Hd);
  real* dxn = ARENA.get(nn * d.C); d.hidden.backward(xn, da, dxn, nn);
  lnBackward(d.ln, ls, dxn, dpair, nn, true);
}

}  // namespace af3
}  // namespace grad
