// OpenFold3's training losses (openfold3/core/loss: diffusion.py, distogram.py, confidence.py), on the host in float64:
// each one's value and its gradient - into the atom positions for the diffusion terms (the alignment and every
// target detached, as there), into the logits for the cross entropies. The layout is the trainer's dense one,
// [tokens][dense] slots, and an atom is a slot whose `exists` is set (the batch's predDenseAtomMask).
//
// What each follows, by OpenFold3 function:
//   mse_loss            AF3 eq. 3: the truth weighted-rigid-aligned onto the prediction (Algorithm 28, weights
//                       1 + 5 DNA + 5 RNA + 10 ligand), 1/3 sum w |x - x_gt|^2 over the resolved atoms / their COUNT
//   bond_loss           AF3 eq. 5: every atom pair of a token pair (ligand i, polymer j) the token bonds join, the
//                       squared difference of their distances, a mean over the pairs
//   smooth_lddt_loss    Algorithm 27: atom pairs l != m both resolved, cutoff 30 A where l is a nucleotide's, else 15
//   diffusion_loss      (t^2 + sd^2) / (t sd)^2 (w_mse mse + w_bond bond) + w_lddt smooth_lddt
//   distogram           64 bins over [2, 22] by nearest centre, representative atoms (CB, CA for glycine, C4 for a
//                       purine, C2 a pyrimidine, an atomised token's own atom), every pair of resolved ones
//   all_atom_plddt_loss each resolved atom's LDDT against the resolved CA / C1' of standard polymer tokens, l != m,
//                       cutoff by m's type (15 protein, 30 nucleotide); an atom with none reads 0; 50 bins on [0, 1]
//   pde_loss            |d - d_gt| of the representative atoms, pairs both resolved in the truth, 64 bins on [0, 32]
//   pae_loss            Algorithm 30 over the token frames (N CA C; C3' C1' C4'; an atomised token's two nearest
//                       same-chain atoms, the angle between 25 and 155 degrees), valid in both structures
//   experimentally_resolved  2 bins a slot, the truth's resolved flag, every existing atom
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace of3 {

constexpr double EPS = 1e-8;
// a count OpenFold3 sums in float32 (`torch.sum(mask, dtype=torch.float32) + eps`): there the eps is below the
// count's precision and vanishes - the bond loss over one bonded pair shows it at 1e-8
inline double den32(double count) { return (double)((float)count + 1e-8f); }

struct Layout {
  int n = 0, dense = 0;
  std::vector<char> exists;                         // [n * dense]
  std::vector<char> protein, dna, rna, ligand, atomized;   // [n]
  std::vector<int> asym;                            // [n]
  std::vector<int> rep, lddtRep;                    // [n]: the representative and the LDDT reference slot, -1 none
  std::vector<std::array<int, 3>> frame;            // [n]: a polymer token's (a, b, c) slots; atomised: chosen per structure
  std::vector<int> start;                           // [n]: an atomised token's atom (its first existing slot)
  std::vector<double> bondTok;                      // [n * n]: the token bonds (token_bonds), 0/1
  size_t atoms() const { return (size_t)n * dense; }
  bool nucleotide(int t) const { return dna[t] || rna[t]; }
};

// the layout from the batch's per-slot atom names (four chars each, 32 subtracted, 0 for none) and its token types
inline Layout makeLayout(int n, int dense, const float* exists, const int* names, const int* isDna, const int* isRna,
                         const int* isLigand, const int* isModified, const int* asym, const float* bonds) {
  Layout L; L.n = n; L.dense = dense;
  L.exists.assign((size_t)n * dense, 0);
  for (size_t i = 0; i < L.exists.size(); ++i) L.exists[i] = exists[i] > 0;
  L.protein.assign(n, 0); L.dna.assign(n, 0); L.rna.assign(n, 0); L.ligand.assign(n, 0); L.atomized.assign(n, 0);
  L.asym.assign(asym, asym + n); L.rep.assign(n, -1); L.lddtRep.assign(n, -1); L.start.assign(n, -1);
  L.frame.assign(n, {-1, -1, -1});
  L.bondTok.assign((size_t)n * n, 0);
  if (bonds) for (size_t i = 0; i < (size_t)n * n; ++i) L.bondTok[i] = bonds[i] > 0;
  for (int t = 0; t < n; ++t) {
    L.dna[t] = isDna && isDna[t] > 0; L.rna[t] = isRna && isRna[t] > 0; L.ligand[t] = isLigand && isLigand[t] > 0;
    L.protein[t] = !L.dna[t] && !L.rna[t] && !L.ligand[t];
    // an atomised token: a ligand's atom or a modified residue's (one atom a token)
    int count = 0;
    std::vector<std::pair<std::string, int>> at;
    for (int k = 0; k < dense; ++k) {
      const size_t i = (size_t)t * dense + k;
      if (!L.exists[i]) continue;
      if (L.start[t] < 0) L.start[t] = (int)i;
      ++count;
      std::string nm; for (int c = 0; c < 4; ++c) { int ch = names[i * 4 + c]; if (ch > 0) nm += (char)(ch + 32); }
      at.push_back({nm, (int)i});
    }
    L.atomized[t] = L.ligand[t] || (isModified && isModified[t] > 0) || count == 1;
    auto slot = [&](const char* nm) { for (auto& [a, i] : at) if (a == nm) return i; return -1; };
    if (L.atomized[t]) { L.rep[t] = L.start[t]; continue; }
    if (L.protein[t]) {
      const int cb = slot("CB");
      L.rep[t] = cb >= 0 ? cb : slot("CA");            // (glycine: CA)
      L.lddtRep[t] = slot("CA");
      L.frame[t] = {slot("N"), slot("CA"), slot("C")};
    } else {
      const bool purine = slot("N9") >= 0;
      L.rep[t] = purine ? slot("C4") : slot("C2");
      L.lddtRep[t] = slot("C1'");
      L.frame[t] = {slot("C3'"), slot("C1'"), slot("C4'")};
    }
  }
  return L;
}

inline double dist(const std::vector<double>& x, size_t a, size_t b) {
  double d2 = 0; for (int p = 0; p < 3; ++p) { double e = x[a * 3 + p] - x[b * 3 + p]; d2 += e * e; } return std::sqrt(EPS + d2);
}

// Algorithm 28: the rotation R and translation t that map `a` onto `b` (weights w), by Horn's quaternion method
inline void weightedAlign(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& w,
                          double R[3][3], double t[3]) {
  double ca[3] = {0, 0, 0}, cb[3] = {0, 0, 0}, sw = 0;
  const size_t n = w.size();
  for (size_t i = 0; i < n; ++i) { sw += w[i]; for (int c = 0; c < 3; ++c) { ca[c] += w[i] * a[i * 3 + c]; cb[c] += w[i] * b[i * 3 + c]; } }
  for (int c = 0; c < 3; ++c) { ca[c] /= sw + EPS; cb[c] /= sw + EPS; }
  double S[3][3] = {{0}};
  for (size_t i = 0; i < n; ++i)
    for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q) S[p][q] += w[i] * (a[i * 3 + p] - ca[p]) * (b[i * 3 + q] - cb[q]);
  const double N[4][4] = {{S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
                          {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
                          {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
                          {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]}};
  // the largest eigenvector of N (Jacobi)
  double A[4][4], V[4][4];
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) { A[i][j] = N[i][j]; V[i][j] = i == j; }
  for (int sweep = 0; sweep < 60; ++sweep) {
    double off = 0; for (int i = 0; i < 4; ++i) for (int j = i + 1; j < 4; ++j) off += A[i][j] * A[i][j];
    if (off < 1e-30) break;
    for (int p = 0; p < 4; ++p) for (int q = p + 1; q < 4; ++q) {
      if (std::fabs(A[p][q]) < 1e-300) continue;
      const double th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const double tt = (th >= 0 ? 1 : -1) / (std::fabs(th) + std::sqrt(th * th + 1)), c = 1 / std::sqrt(tt * tt + 1), s = tt * c;
      for (int k = 0; k < 4; ++k) { const double x = A[k][p], y = A[k][q]; A[k][p] = c * x - s * y; A[k][q] = s * x + c * y; }
      for (int k = 0; k < 4; ++k) { const double x = A[p][k], y = A[q][k]; A[p][k] = c * x - s * y; A[q][k] = s * x + c * y; }
      for (int k = 0; k < 4; ++k) { const double x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
    }
  }
  int best = 0; for (int i = 1; i < 4; ++i) if (A[i][i] > A[best][best]) best = i;
  const double q0 = V[0][best], q1 = V[1][best], q2 = V[2][best], q3 = V[3][best];
  const double Rm[3][3] = {{q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)},
                           {2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)},
                           {2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3}};
  for (int p = 0; p < 3; ++p) { for (int q = 0; q < 3; ++q) R[p][q] = Rm[p][q]; }
  for (int p = 0; p < 3; ++p) { t[p] = cb[p]; for (int q = 0; q < 3; ++q) t[p] -= R[p][q] * ca[q]; }
}

struct Diffusion { double mse = 0, bond = 0, lddt = 0, total = 0; };
struct DiffusionWeights { double mse = 4, bond = 0, lddt = 4, dna = 5, rna = 5, ligand = 10, sigmaData = 16; };

// one diffusion sample: x the denoised positions, xt/gm the truth and its resolved mask; dx += d total / d x * scale
// `smoothLddt` (null: the host's below) computes 1 - lddt and adds its scaled gradient - of3dev::SmoothLddt on the device
inline Diffusion diffusionLoss(const Layout& L, const std::vector<double>& x, const std::vector<double>& xt,
                               const std::vector<double>& gm, double t, const DiffusionWeights& W, std::vector<double>* dx,
                               double scale = 1,
                               const std::function<double(double, std::vector<double>*)>& smoothLddt = {},
                               const std::vector<double>* lossAtom = nullptr) {
  // (lossAtom: OpenFold3's loss_token_mask on atoms - the disordered set's protein-only diffusion terms; the alignment
  // still weighs every resolved atom)
  Diffusion D;
  const size_t A = L.atoms();
  const double wt = (t * t + W.sigmaData * W.sigmaData) / ((t * W.sigmaData) * (t * W.sigmaData));
  std::vector<double> w(A, 0), m(A, 0);
  std::vector<size_t> live;
  for (size_t i = 0; i < A; ++i) {
    if (!L.exists[i]) continue;
    const int tk = (int)(i / L.dense);
    w[i] = 1 + W.dna * L.dna[tk] + W.rna * L.rna[tk] + W.ligand * L.ligand[tk];
    m[i] = gm[i] > 0;
    if (m[i] && (!lossAtom || (*lossAtom)[i] > 0)) live.push_back(i);
  }
  // mse: the truth aligned onto the prediction (detached)
  {
    std::vector<double> wa(A); for (size_t i = 0; i < A; ++i) wa[i] = w[i] * m[i];
    double R[3][3], tr[3];
    weightedAlign(xt, x, wa, R, tr);
    double cnt = 0; for (size_t i : live) cnt += 1;
    for (size_t i : live) {
      for (int p = 0; p < 3; ++p) {
        double al = tr[p]; for (int c = 0; c < 3; ++c) al += R[p][c] * xt[i * 3 + c];
        const double e = x[i * 3 + p] - al;
        D.mse += e * e * w[i] / 3.0 / (cnt + EPS);
        if (dx) (*dx)[i * 3 + p] += scale * wt * W.mse * 2 * e * w[i] / 3.0 / (cnt + EPS);
      }
    }
  }
  // bond: atom pairs (l of a ligand token i, m of a polymer token j bonded to it), both resolved
  if (W.bond != 0) {
    std::vector<std::pair<size_t, size_t>> pairs;
    for (int i = 0; i < L.n; ++i) {
      if (!L.ligand[i]) continue;
      for (int j = 0; j < L.n; ++j) {
        if (L.ligand[j] || !L.bondTok[(size_t)i * L.n + j]) continue;
        for (int a = 0; a < L.dense; ++a) for (int b = 0; b < L.dense; ++b) {
          const size_t l = (size_t)i * L.dense + a, mm = (size_t)j * L.dense + b;
          if (L.exists[l] && L.exists[mm] && m[l] && m[mm]) pairs.push_back({l, mm});
        }
      }
    }
    for (auto [l, mm] : pairs) {
      const double dp = dist(x, l, mm), dg = dist(xt, l, mm), e = dp - dg;
      D.bond += e * e / den32((double)pairs.size());
      if (dx) for (int p = 0; p < 3; ++p) {
        const double g = scale * wt * W.bond * 2 * e / den32((double)pairs.size()) * (x[l * 3 + p] - x[mm * 3 + p]) / dp;
        (*dx)[l * 3 + p] += g; (*dx)[mm * 3 + p] -= g;
      }
    }
  }
  // smooth LDDT
  if (smoothLddt) D.lddt = smoothLddt(scale * W.lddt, dx);
  else {
    double num = 0, den = 0;
    struct T { size_t l, m; double g; };
    std::vector<T> terms;
    for (size_t l : live) {
      const bool nuc = L.nucleotide((int)(l / L.dense));
      for (size_t mm : live) {
        if (l == mm) continue;
        const double dg = dist(xt, l, mm);
        if (!(dg < (nuc ? 30 : 15))) continue;
        const double dp = dist(x, l, mm), del = std::fabs(dg - dp);
        double e = 0, de = 0;
        for (double c : {0.5, 1.0, 2.0, 4.0}) { const double s = 1 / (1 + std::exp(-(c - del))); e += s / 4; de -= s * (1 - s) / 4; }
        num += e; den += 1;
        terms.push_back({l, mm, de * (dp > dg ? 1 : -1) / dp});
      }
    }
    D.lddt = 1 - num / (den + EPS);
    if (dx) for (auto& T_ : terms)
      for (int p = 0; p < 3; ++p) {
        const double g = -scale * W.lddt * T_.g * (x[T_.l * 3 + p] - x[T_.m * 3 + p]) / (den + EPS);
        (*dx)[T_.l * 3 + p] += g; (*dx)[T_.m * 3 + p] -= g;
      }
  }
  D.total = wt * (W.mse * D.mse + W.bond * D.bond) + W.lddt * D.lddt;
  return D;
}

// lDDT (all atoms, as a validation metric): over resolved atom pairs l != m within 15 A in the truth (30 where m is a
// nucleotide's), the mean over thresholds 0.5, 1, 2, 4 of |d - d_true| below it, averaged over the pairs
inline double lddt(const Layout& L, const std::vector<double>& x, const std::vector<double>& xt, const std::vector<double>& gm) {
  std::vector<size_t> live; for (size_t i = 0; i < L.atoms(); ++i) if (L.exists[i] && gm[i] > 0) live.push_back(i);
  double num = 0, den = 0;
  for (size_t l : live) for (size_t m : live) {
    if (l == m) continue;
    const double dg = dist(xt, l, m);
    if (!(dg < (L.nucleotide((int)(m / L.dense)) ? 30 : 15))) continue;
    const double d = std::fabs(dist(x, l, m) - dg);
    num += 0.25 * ((d < 0.5) + (d < 1) + (d < 2) + (d < 4)); den += 1;
  }
  return den > 0 ? num / den : 0;
}

// cross entropy against a one-hot bin; its gradient (softmax - onehot) * scale added into g
template <class R> inline double crossEntropy(const R* lg, int K, int bin, double scale, double* g) {
  double mx = -1e300; for (int k = 0; k < K; ++k) mx = std::max(mx, (double)lg[k]);
  double z = 0; for (int k = 0; k < K; ++k) z += std::exp(lg[k] - mx);
  if (g) for (int k = 0; k < K; ++k) g[k] += scale * (std::exp(lg[k] - mx) / z - (k == bin));
  return -(lg[bin] - mx - std::log(z));
}
// the nearest of K bin centres lo + (k + 1/2) (hi - lo) / K (binned_one_hot)
inline int nearestBin(double v, double lo, double hi, int K) {
  const double w = (hi - lo) / K;
  int best = 0; double bd = 1e300;
  for (int k = 0; k < K; ++k) { const double d = std::fabs(v - (lo + w / 2 + k * w)); if (d < bd) { bd = d; best = k; } }
  return best;
}

// the representative atoms' positions and masks under a structure's atom mask
inline void reps(const Layout& L, const std::vector<double>& am, std::vector<int>& idx, std::vector<char>& ok) {
  idx.assign(L.n, -1); ok.assign(L.n, 0);
  for (int t = 0; t < L.n; ++t) { idx[t] = L.rep[t]; ok[t] = idx[t] >= 0 && am[idx[t]] > 0; }
}

// distogram: logits [n * n * K] (K 64), gradient * scale into g [n * n * K]
template <class R> inline double distogramLoss(const Layout& L, const std::vector<double>& xt, const std::vector<double>& gm,
                                                const R* logits, int K, double scale, double* g) {
  std::vector<int> ri; std::vector<char> ok; reps(L, gm, ri, ok);
  double sum = 0, cnt = 0;
  for (int i = 0; i < L.n; ++i) for (int j = 0; j < L.n; ++j) cnt += ok[i] && ok[j];
  for (int i = 0; i < L.n; ++i) for (int j = 0; j < L.n; ++j) {
    if (!ok[i] || !ok[j]) continue;
    double d2 = 0; for (int p = 0; p < 3; ++p) { double e = xt[ri[i] * 3 + p] - xt[ri[j] * 3 + p]; d2 += e * e; }
    const size_t ij = (size_t)i * L.n + j;
    sum += crossEntropy(logits + ij * K, K, nearestBin(std::sqrt(d2), 2, 22, K), scale / den32(cnt), g ? g + ij * K : nullptr);
  }
  return sum / den32(cnt);
}

// pLDDT: logits [atoms * K] (dense slots, K 50)
template <class R> inline double plddtLoss(const Layout& L, const std::vector<double>& x, const std::vector<double>& xt,
                                            const std::vector<double>& gm, const R* logits, int K, double scale, double* g) {
  const size_t A = L.atoms();
  std::vector<size_t> live; for (size_t i = 0; i < A; ++i) if (L.exists[i] && gm[i] > 0) live.push_back(i);
  std::vector<std::pair<size_t, bool>> refs;    // the reference atoms m, and whether a nucleotide's
  for (int t = 0; t < L.n; ++t)
    if (!L.atomized[t] && L.lddtRep[t] >= 0 && gm[L.lddtRep[t]] > 0) refs.push_back({(size_t)L.lddtRep[t], L.nucleotide(t)});
  double sum = 0;
  for (size_t l : live) {
    double acc = 0, cnt = 0;
    for (auto [mm, nuc] : refs) {
      if (mm == l) continue;
      const double dg = dist(xt, l, mm);
      if (!(dg < (nuc ? 30 : 15))) continue;
      const double d = std::fabs(dg - dist(x, l, mm));
      acc += 0.25 * ((d < 0.5) + (d < 1) + (d < 2) + (d < 4)); cnt += 1;
    }
    const double lddt = acc / den32(cnt);
    sum += crossEntropy(logits + l * K, K, nearestBin(lddt, 0, 1, K), scale / den32((double)live.size()), g ? g + l * K : nullptr);
  }
  return sum / den32((double)live.size());
}

// PDE: logits [n * n * K]; the prediction's representatives under every existing atom
template <class R> inline double pdeLoss(const Layout& L, const std::vector<double>& x, const std::vector<double>& xt,
                                          const std::vector<double>& gm, const R* logits, int K, double scale, double* g) {
  std::vector<int> ri; std::vector<char> ok; reps(L, gm, ri, ok);
  double sum = 0, cnt = 0;
  for (int i = 0; i < L.n; ++i) for (int j = 0; j < L.n; ++j) cnt += ok[i] && ok[j];
  for (int i = 0; i < L.n; ++i) for (int j = 0; j < L.n; ++j) {
    if (!ok[i] || !ok[j]) continue;
    const double e = std::fabs(dist(x, ri[i], ri[j]) - dist(xt, ri[i], ri[j]));
    const size_t ij = (size_t)i * L.n + j;
    sum += crossEntropy(logits + ij * K, K, nearestBin(e, 0, 32, K), scale / den32(cnt), g ? g + ij * K : nullptr);
  }
  return sum / den32(cnt);
}

// a structure's token frames (get_token_frame_atoms): (a, b, c) slots and validity
inline void frames(const Layout& L, const std::vector<double>& x, const std::vector<double>& am,
                   std::vector<std::array<int, 3>>& f, std::vector<char>& valid, double angle = 25) {
  f.assign(L.n, {-1, -1, -1}); valid.assign(L.n, 0);
  const double cmax = std::cos(angle * M_PI / 180), cmin = std::cos((180 - angle) * M_PI / 180);
  for (int t = 0; t < L.n; ++t) {
    std::array<int, 3> fr = L.frame[t];
    if (L.atomized[t]) {
      // the start atom's two nearest same-chain atoms (the start itself the nearest, at distance sqrt(3 eps))
      const int s = L.start[t];
      if (s < 0 || !(am[s] > 0)) continue;
      std::vector<std::pair<double, int>> d;
      for (size_t i = 0; i < L.atoms(); ++i) {
        if (!L.exists[i] || !(am[i] > 0) || L.asym[i / L.dense] != L.asym[t]) continue;
        double s2 = 0; for (int p = 0; p < 3; ++p) { double e = x[s * 3 + p] - x[i * 3 + p]; s2 += EPS + e * e; }
        d.push_back({std::sqrt(s2), (int)i});
      }
      if (d.size() < 3) continue;
      std::partial_sort(d.begin(), d.begin() + 3, d.end());
      fr = {d[1].second, s, d[2].second};
    }
    if (fr[0] < 0 || fr[1] < 0 || fr[2] < 0) continue;
    if (!(am[fr[0]] > 0 && am[fr[1]] > 0 && am[fr[2]] > 0)) continue;
    if (L.atomized[t]) {
      double u[3], v[3], uv = 0, uu = EPS, vv = EPS;
      for (int p = 0; p < 3; ++p) { u[p] = x[fr[0] * 3 + p] - x[fr[1] * 3 + p]; v[p] = x[fr[2] * 3 + p] - x[fr[1] * 3 + p]; }
      for (int p = 0; p < 3; ++p) { uv += u[p] * v[p]; uu += u[p] * u[p]; vv += v[p] * v[p]; }
      const double c = uv / (std::sqrt(uu) * std::sqrt(vv));
      if (!(c < cmax && c > cmin)) continue;
    }
    f[t] = fr; valid[t] = 1;
  }
}
// Algorithm 29: point x_j expressed in frame (a, b, c)
inline void inFrame(const std::vector<double>& x, const std::array<int, 3>& f, const double* pt, double out[3]) {
  double w1[3], w2[3], n1 = EPS, n2 = EPS;
  for (int p = 0; p < 3; ++p) { w1[p] = x[f[0] * 3 + p] - x[f[1] * 3 + p]; w2[p] = x[f[2] * 3 + p] - x[f[1] * 3 + p]; n1 += w1[p] * w1[p]; n2 += w2[p] * w2[p]; }
  n1 = std::sqrt(n1); n2 = std::sqrt(n2);
  double e1[3], e2[3], a = EPS, b = EPS;
  for (int p = 0; p < 3; ++p) { w1[p] /= n1; w2[p] /= n2; e1[p] = w1[p] + w2[p]; e2[p] = w2[p] - w1[p]; a += e1[p] * e1[p]; b += e2[p] * e2[p]; }
  a = std::sqrt(a); b = std::sqrt(b);
  for (int p = 0; p < 3; ++p) { e1[p] /= a; e2[p] /= b; }
  const double e3[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
  double d[3]; for (int p = 0; p < 3; ++p) d[p] = pt[p] - x[f[1] * 3 + p];
  out[0] = e1[0] * d[0] + e1[1] * d[1] + e1[2] * d[2];
  out[1] = e2[0] * d[0] + e2[1] * d[1] + e2[2] * d[2];
  out[2] = e3[0] * d[0] + e3[1] * d[1] + e3[2] * d[2];
}
// PAE: logits [n * n * K], i the frame token
template <class R> inline double paeLoss(const Layout& L, const std::vector<double>& x, const std::vector<double>& xt,
                                          const std::vector<double>& gm, const R* logits, int K, double scale, double* g) {
  std::vector<double> all(L.atoms()); for (size_t i = 0; i < all.size(); ++i) all[i] = L.exists[i];
  std::vector<std::array<int, 3>> fp, fg; std::vector<char> vp, vg;
  frames(L, x, all, fp, vp); frames(L, xt, gm, fg, vg);
  std::vector<int> rp, rg; std::vector<char> op, og; reps(L, all, rp, op); reps(L, gm, rg, og);
  double sum = 0, cnt = 0;
  for (int i = 0; i < L.n; ++i) for (int j = 0; j < L.n; ++j) cnt += vp[i] && op[j] && vg[i] && og[j];
  for (int i = 0; i < L.n; ++i) {
    if (!vp[i] || !vg[i]) continue;
    for (int j = 0; j < L.n; ++j) {
      if (!op[j] || !og[j]) continue;
      double a[3], b[3];
      inFrame(x, fp[i], &x[(size_t)rp[j] * 3], a); inFrame(xt, fg[i], &xt[(size_t)rg[j] * 3], b);
      const double e = std::sqrt(EPS + (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
      const size_t ij = (size_t)i * L.n + j;
      sum += crossEntropy(logits + ij * K, K, nearestBin(e, 0, 32, K), scale / (cnt + EPS), g ? g + ij * K : nullptr);
    }
  }
  return sum / (cnt + EPS);
}

// experimentally resolved: logits [atoms * 2]
template <class R> inline double resolvedLoss(const Layout& L, const std::vector<double>& gm, const R* logits, double scale,
                                               double* g) {
  double sum = 0, cnt = 0;
  for (size_t i = 0; i < L.atoms(); ++i) cnt += L.exists[i];
  for (size_t i = 0; i < L.atoms(); ++i)
    if (L.exists[i]) sum += crossEntropy(logits + i * 2, 2, gm[i] > 0, scale / den32(cnt), g ? g + i * 2 : nullptr);
  return sum / den32(cnt);
}

}  // namespace of3

#ifdef __CUDACC__
// ---- smooth LDDT on the device (diffusionLoss's quadratic part: at a 384-token crop ~3,000 atoms, 9 M pairs a
// sample, 48 samples a step). The same arithmetic in float64; held to the host's by test-of3loss.
namespace of3dev {
// pass 1: num = sum c e, den = sum c over pairs l != m, both resolved (`live`), the cutoff by l's type
__global__ void smoothLddtK(const double* __restrict__ x, const double* __restrict__ xt, const int* __restrict__ live,
                            const char* __restrict__ nuc, int L, double* __restrict__ sums) {
  double num = 0, den = 0;
  for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < (size_t)L * L; t += (size_t)gridDim.x * blockDim.x) {
    const int a = (int)(t / L), b = (int)(t % L);
    if (a == b) continue;
    const int l = live[a], m = live[b];
    double g2 = of3::EPS, p2 = of3::EPS;
    for (int q = 0; q < 3; ++q) { const double e = xt[l * 3 + q] - xt[m * 3 + q], f = x[l * 3 + q] - x[m * 3 + q]; g2 += e * e; p2 += f * f; }
    const double dg = sqrt(g2);
    if (!(dg < (nuc[a] ? 30.0 : 15.0))) continue;
    const double del = fabs(dg - sqrt(p2));
    double e = 0;
    for (double c : {0.5, 1.0, 2.0, 4.0}) e += 0.25 / (1 + exp(-(c - del)));
    num += e; den += 1;
  }
  for (int o = 16; o; o >>= 1) { num += __shfl_xor_sync(~0u, num, o); den += __shfl_xor_sync(~0u, den, o); }
  if ((threadIdx.x & 31) == 0) { atomicAdd(sums, num); atomicAdd(sums + 1, den); }
}
// pass 2: dx_l += -scale * de/dd (x_l - x_m) / d / den over m, a thread a row l (no atomics: l's own sum)
__global__ void smoothLddtGradK(const double* __restrict__ x, const double* __restrict__ xt, const int* __restrict__ live,
                                const char* __restrict__ nuc, int L, const double* __restrict__ sums, double scale,
                                double* __restrict__ dx) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= L) return;
  const int l = live[a];
  const double inv = scale / (sums[1] + of3::EPS);
  double g[3] = {0, 0, 0};
  for (int b = 0; b < L; ++b) {
    if (a == b) continue;
    const int m = live[b];
    // (l, m) and (m, l) are both terms, each with its own cutoff by its first atom's type
    double dvec[3], g2 = of3::EPS, p2 = of3::EPS;
    for (int q = 0; q < 3; ++q) { const double e = xt[l * 3 + q] - xt[m * 3 + q]; dvec[q] = x[l * 3 + q] - x[m * 3 + q]; g2 += e * e; p2 += dvec[q] * dvec[q]; }
    const double dg = sqrt(g2), dp = sqrt(p2), del = fabs(dg - dp);
    double de = 0;
    for (double c : {0.5, 1.0, 2.0, 4.0}) { const double s = 1 / (1 + exp(-(c - del))); de -= 0.25 * s * (1 - s); }
    const double k = de * (dp > dg ? 1 : -1) / dp;
    const int terms = (dg < (nuc[a] ? 30.0 : 15.0)) + (dg < (nuc[b] ? 30.0 : 15.0));
    for (int q = 0; q < 3; ++q) g[q] -= terms * k * dvec[q];
  }
  for (int q = 0; q < 3; ++q) dx[l * 3 + q] += inv * g[q];
}
struct SmoothLddt {
  int L = 0; int* live = nullptr; char* nuc = nullptr; double *x = nullptr, *xt = nullptr, *dx = nullptr, *sums = nullptr; size_t A = 0;
  // once an example: its resolved atoms and their types, and the truth
  void init(const of3::Layout& Lay, const std::vector<double>& xtH, const std::vector<double>& gm) {
    release();
    A = Lay.atoms();
    std::vector<int> lv; std::vector<char> nc;
    for (size_t i = 0; i < A; ++i) if (Lay.exists[i] && gm[i] > 0) { lv.push_back((int)i); nc.push_back(Lay.nucleotide((int)(i / Lay.dense))); }
    L = (int)lv.size();
    cudaMalloc(&live, std::max(1, L) * sizeof(int)); cudaMalloc(&nuc, std::max(1, L));
    cudaMalloc(&x, A * 3 * 8); cudaMalloc(&xt, A * 3 * 8); cudaMalloc(&dx, A * 3 * 8); cudaMalloc(&sums, 16);
    cudaMemcpy(live, lv.data(), L * sizeof(int), cudaMemcpyHostToDevice); cudaMemcpy(nuc, nc.data(), L, cudaMemcpyHostToDevice);
    cudaMemcpy(xt, xtH.data(), A * 3 * 8, cudaMemcpyHostToDevice);
  }
  // 1 - lddt, and scale * d(1 - lddt)/dx added into dxH
  double run(const std::vector<double>& xH, double scale, std::vector<double>* dxH, cudaStream_t s) {
    cudaMemcpyAsync(x, xH.data(), A * 3 * 8, cudaMemcpyHostToDevice, s);
    cudaMemsetAsync(sums, 0, 16, s);
    smoothLddtK<<<std::min<size_t>(((size_t)L * L + 255) / 256, 4096), 256, 0, s>>>(x, xt, live, nuc, L, sums);
    if (dxH) {
      cudaMemsetAsync(dx, 0, A * 3 * 8, s);
      smoothLddtGradK<<<(L + 127) / 128, 128, 0, s>>>(x, xt, live, nuc, L, sums, scale, dx);
      std::vector<double> h(A * 3);
      cudaMemcpyAsync(h.data(), dx, A * 3 * 8, cudaMemcpyDeviceToHost, s);
      double sh[2]; cudaMemcpyAsync(sh, sums, 16, cudaMemcpyDeviceToHost, s); cudaStreamSynchronize(s);
      for (size_t i = 0; i < A * 3; ++i) (*dxH)[i] += h[i];
      return 1 - sh[0] / (sh[1] + of3::EPS);
    }
    double sh[2]; cudaMemcpyAsync(sh, sums, 16, cudaMemcpyDeviceToHost, s); cudaStreamSynchronize(s);
    return 1 - sh[0] / (sh[1] + of3::EPS);
  }
  void release() { for (void* p : {(void*)live, (void*)nuc, (void*)x, (void*)xt, (void*)dx, (void*)sums}) if (p) cudaFree(p); live = nullptr; nuc = nullptr; x = xt = dx = sums = nullptr; }
};
}  // namespace of3dev
#endif
