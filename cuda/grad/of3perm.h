// OpenFold3's ground-truth permutation (openfold3/core/utils/permutation_alignment.py, AF3 SI 4.2): which copy of a
// repeated molecule in the truth each predicted molecule is scored against, and which naming of each symmetric
// component (a Val's CG1/CG2, a Phe's ring, a symmetric ligand) - chosen on the mini-rollout's prediction and then
// used by every loss of the step. On the host, in float64.
//
//   anchor        the truth's least ambiguous molecule: of the entities with the fewest copies, the copy with the
//                 most resolved token centres (more than 3), a polymer before a ligand (get_gt_anchor_mask)
//   alignments    the anchor's resolved centres onto each predicted copy of its entity, one candidate each
//                 (get_anchor_transformations)
//   matching      for each candidate, every predicted molecule greedily to the nearest unused truth copy of its entity
//                 by centroid (the predicted molecules with the most resolved truth first); the candidate whose
//                 matched centroids are nearest wins (find_greedy_optimal_mol_permutation)
//   atoms         the prediction aligned onto the permuted truth's centres, then within each component the RDKit
//                 automorphism (cuda/grad/make_of3_example.py's perms.txt) nearest over the resolved atoms
//                 (get_final_atom_permutation_index)
//
// The prediction is the crop and the truth the whole entry, as there: a cropped copy of chain A may be matched to
// chain B's same residues though B is not in the crop.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "of3loss.h"

namespace of3 {

struct PermInput {
  // the whole entry: per token its entity, copy (sym), chain (asym), position in its molecule, residue index, whether
  // a ligand's, its centre slot (-1 none); per slot its name; the truth and its resolved mask
  int n = 0, dense = 0;
  std::vector<int> ent, sym, asym, molTok, ri, centre;
  std::vector<char> ligand;
  std::vector<std::string> name;           // [n * dense], "" where no atom
  std::vector<double> xt, gm;
  // a component's symmetric namings (perms.txt): (asym, residue) -> namings, the first the identity's order
  std::map<std::pair<int, int>, std::vector<std::vector<std::string>>> perms;
};

// R, t minimising sum w |R a_i + t - b_i|^2 (weightedAlign's quaternion method), applied to points
struct Rigid { double R[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, t[3] = {0, 0, 0}; };
inline Rigid fit(const std::vector<std::array<double, 3>>& a, const std::vector<std::array<double, 3>>& b, const std::vector<double>& w) {
  std::vector<double> A(a.size() * 3), B(b.size() * 3);
  for (size_t i = 0; i < a.size(); ++i) for (int q = 0; q < 3; ++q) { A[i * 3 + q] = a[i][q]; B[i * 3 + q] = b[i][q]; }
  Rigid r; weightedAlign(A, B, w, r.R, r.t); return r;
}
inline std::array<double, 3> applyRigid(const Rigid& r, const std::array<double, 3>& p) {
  std::array<double, 3> o;
  for (int q = 0; q < 3; ++q) { o[q] = r.t[q]; for (int c = 0; c < 3; ++c) o[q] += r.R[q][c] * p[c]; }
  return o;
}

// The permuted truth for a prediction over tokens `keep` (indices into the whole entry, ascending) with positions
// `xp` in the crop's dense layout: xtOut/gmOut in that layout. `report` (optional) gets the chain matching.
inline void permuteTruth(const PermInput& P, const std::vector<int>& keep, const std::vector<double>& xp,
                         std::vector<double>& xtOut, std::vector<double>& gmOut, std::string* report = nullptr) {
  const int m = (int)keep.size(), D = P.dense;
  auto gtCentre = [&](int t) { const int a = P.centre[t]; return std::array<double, 3>{P.xt[a * 3], P.xt[a * 3 + 1], P.xt[a * 3 + 2]}; };
  auto gtRes = [&](int t) { return P.centre[t] >= 0 && P.gm[P.centre[t]] > 0; };
  auto pdCentre = [&](int k) { const int a = P.centre[keep[k]] - keep[k] * D + k * D; return std::array<double, 3>{xp[a * 3], xp[a * 3 + 1], xp[a * 3 + 2]}; };
  // ---- the anchor
  std::set<std::pair<int, int>> combos;
  for (int t = 0; t < P.n; ++t) if (gtRes(t)) combos.insert({P.ent[t], P.sym[t]});
  auto isLigandEntity = [&](int e) { for (int t = 0; t < P.n; ++t) if (P.ent[t] == e && !P.ligand[t]) return false; return true; };
  struct Cand { int ent = -1, sym = -1, resolved = -1; };
  auto leastAmbiguous = [&](bool ligands) {
    std::map<int, int> copies;   // entity -> copies with a resolved centre
    for (auto& [e, s] : combos) if (isLigandEntity(e) == ligands) ++copies[e];
    std::map<int, std::vector<int>> byCount; for (auto& [e, c] : copies) byCount[c].push_back(e);
    Cand best; bool have = false;
    for (auto& [count, ents] : byCount) {
      Cand c;
      for (int e : ents) {
        std::map<int, int> res; for (int t = 0; t < P.n; ++t) if (P.ent[t] == e && gtRes(t)) ++res[P.sym[t]];
        Cand ce; for (auto& [s, r] : res) if (r > ce.resolved) ce = {e, s, r};   // (the first sym with the most)
        if (ce.resolved > c.resolved) c = ce;
      }
      if (c.resolved > 3) return c;
      if (!have) { best = c; have = true; }
    }
    return best;
  };
  bool polymers = false, ligands = false;
  for (auto& [e, s] : combos) (isLigandEntity(e) ? ligands : polymers) = true;
  Cand anchor;
  if (polymers) anchor = leastAmbiguous(false);
  if (!polymers || anchor.resolved < 3) {
    if (ligands) { Cand l = leastAmbiguous(true); if (l.resolved > 3 || !polymers) anchor = l; }
  }
  // ---- the candidate alignments: the anchor's centres onto each predicted copy of its entity
  std::map<int, std::vector<int>> pdChains;   // sym -> crop tokens, of the anchor entity
  for (int k = 0; k < m; ++k) if (P.ent[keep[k]] == anchor.ent) pdChains[P.sym[keep[k]]].push_back(k);
  std::vector<Rigid> cands;
  for (auto& [s, toks] : pdChains) {
    std::set<int> idx; for (int k : toks) idx.insert(P.molTok[keep[k]]);
    std::vector<std::array<double, 3>> a, b; std::vector<double> w;
    // (the anchor's tokens at the copy's positions, in order: both sides ascend by position)
    std::map<int, int> gtAt; for (int t = 0; t < P.n; ++t) if (P.ent[t] == anchor.ent && P.sym[t] == anchor.sym && idx.count(P.molTok[t])) gtAt[P.molTok[t]] = t;
    for (int k : toks) {
      auto it = gtAt.find(P.molTok[keep[k]]);
      if (it == gtAt.end()) continue;
      a.push_back(gtCentre(it->second)); b.push_back(pdCentre(k)); w.push_back(gtRes(it->second) ? 1 : 0);
    }
    cands.push_back(fit(a, b, w));
  }
  if (cands.empty()) cands.push_back(Rigid{});
  // ---- the greedy matching under each candidate
  std::set<int> pdEnts; for (int k = 0; k < m; ++k) pdEnts.insert(P.ent[keep[k]]);
  const double BIG = std::numeric_limits<double>::max();
  double bestRmsd = BIG; std::map<std::pair<int, int>, int> best;   // (entity, predicted sym) -> truth sym
  for (const Rigid& T : cands) {
    std::vector<double> dists; std::map<std::pair<int, int>, int> perm;
    for (int e : pdEnts) {
      std::map<int, std::vector<int>> gtSyms, pdSyms;
      for (int t = 0; t < P.n; ++t) if (P.ent[t] == e) gtSyms[P.sym[t]].push_back(t);
      for (int k = 0; k < m; ++k) if (P.ent[keep[k]] == e) pdSyms[P.sym[keep[k]]].push_back(k);
      // the predicted copies with the most resolved truth (over every truth copy, at their positions) first
      std::vector<std::pair<int, int>> order;
      for (auto& [s, ks] : pdSyms) {
        std::set<int> idx; for (int k : ks) idx.insert(P.molTok[keep[k]]);
        int r = 0; for (auto& [gs, ts] : gtSyms) for (int t : ts) r += idx.count(P.molTok[t]) && gtRes(t);
        order.push_back({r, s});
      }
      std::stable_sort(order.begin(), order.end(), [](auto& x, auto& y) { return x.first > y.first; });
      std::set<int> used;
      for (auto& [r_, s] : order) {
        const auto& ks = pdSyms[s];
        std::set<int> idx; for (int k : ks) idx.insert(P.molTok[keep[k]]);
        std::map<int, std::array<double, 3>> pdAt; for (int k : ks) pdAt[P.molTok[keep[k]]] = pdCentre(k);
        double bd = BIG; int bs = -1; bool first = true;
        for (auto& [gs, ts] : gtSyms) {
          // the truth copy's centroid over its resolved centres at these positions, and the prediction's over the same
          double gc[3] = {0, 0, 0}, pc[3] = {0, 0, 0}, cnt = 0;
          for (int t : ts) {
            if (!idx.count(P.molTok[t]) || !gtRes(t)) continue;
            auto g = applyRigid(T, gtCentre(t)); auto p = pdAt[P.molTok[t]];
            for (int q = 0; q < 3; ++q) { gc[q] += g[q]; pc[q] += p[q]; }
            cnt += 1;
          }
          double d2 = BIG;
          if (cnt > 0) { d2 = 0; for (int q = 0; q < 3; ++q) { const double v = pc[q] / (cnt + 1e-4) - gc[q] / (cnt + 1e-4); d2 += v * v; } }
          if (used.count(gs)) d2 = std::numeric_limits<double>::infinity();
          if (first || d2 < bd) { bd = d2; bs = gs; first = false; }
        }
        used.insert(bs); perm[{e, s}] = bs; dists.push_back(bd);
      }
    }
    double maxRes = 0; bool anyRes = false;
    for (double v : dists) if (v != BIG && std::isfinite(v)) { maxRes = std::max(maxRes, v); anyRes = true; }
    double sum = 0; for (double v : dists) sum += (v == BIG || !std::isfinite(v)) ? (anyRes ? maxRes + 1 : 1) : v;
    const double rmsd = std::sqrt(sum / std::max<size_t>(1, dists.size()));
    if (rmsd < bestRmsd) { bestRmsd = rmsd; best = perm; }
  }
  // ---- the crop's tokens in the matched truth copies, and the prediction aligned onto them
  std::map<std::tuple<int, int, int>, int> gtTok;   // (entity, sym, position) -> token
  for (int t = 0; t < P.n; ++t) gtTok[{P.ent[t], P.sym[t], P.molTok[t]}] = t;
  std::vector<int> toGt(m);
  for (int k = 0; k < m; ++k) {
    const int t = keep[k];
    toGt[k] = gtTok.at({P.ent[t], best.at({P.ent[t], P.sym[t]}), P.molTok[t]});
  }
  Rigid A;
  {
    std::vector<std::array<double, 3>> a, b; std::vector<double> w;
    for (int k = 0; k < m; ++k) if (P.centre[keep[k]] >= 0) { a.push_back(pdCentre(k)); b.push_back(gtCentre(toGt[k])); w.push_back(gtRes(toGt[k]) ? 1 : 0); }
    A = fit(a, b, w);
  }
  // ---- each component's naming: the automorphism nearest the aligned prediction over the resolved atoms
  xtOut.assign((size_t)m * D * 3, 0); gmOut.assign((size_t)m * D, 0);
  std::map<std::pair<int, int>, std::vector<int>> comps;   // a predicted component (asym, residue) -> crop tokens
  for (int k = 0; k < m; ++k) comps[{P.asym[keep[k]], P.ri[keep[k]]}].push_back(k);
  for (auto& [key, ks] : comps) {
    // the predicted atoms by name, and the matched truth component's atoms by name
    std::map<std::string, int> pdSlot, gtSlot;
    for (int k : ks) for (int s = 0; s < D; ++s) { const auto& nm = P.name[(size_t)keep[k] * D + s]; if (!nm.empty()) pdSlot[nm] = k * D + s; }
    std::set<int> gtToks; for (int k : ks) gtToks.insert(toGt[k]);
    // (an atomised component spans tokens the crop may have cut: the truth's whole component, by its asym and residue)
    const int gAsym = P.asym[toGt[ks[0]]], gRi = P.ri[toGt[ks[0]]];
    for (int t = 0; t < P.n; ++t) if (P.asym[t] == gAsym && P.ri[t] == gRi)
      for (int s = 0; s < D; ++s) { const auto& nm = P.name[(size_t)t * D + s]; if (!nm.empty()) gtSlot[nm] = t * D + s; }
    // the candidate namings: the identity, and each automorphism defined on every predicted atom and onto truth atoms
    std::vector<std::map<std::string, std::string>> namings(1);
    for (auto& [nm, sl] : pdSlot) namings[0][nm] = nm;
    auto pit = P.perms.find({gAsym, gRi});
    if (pit != P.perms.end()) {
      const auto& base = pit->second[0];
      for (size_t q = 1; q < pit->second.size(); ++q) {
        std::map<std::string, std::string> nmap; bool ok = true;
        for (size_t i = 0; i < base.size(); ++i) {
          if (!pdSlot.count(base[i])) continue;
          if (!gtSlot.count(pit->second[q][i])) { ok = false; break; }
          nmap[base[i]] = pit->second[q][i];
        }
        if (ok && nmap.size() == pdSlot.size()) namings.push_back(nmap);
      }
    }
    size_t pick = 0; double bestD = std::numeric_limits<double>::infinity();
    for (size_t q = 0; q < namings.size(); ++q) {
      double s2 = 0, cnt = 0;
      for (auto& [nm, ps] : pdSlot) {
        auto it = gtSlot.find(namings[q].at(nm));
        if (it == gtSlot.end() || !(P.gm[it->second] > 0)) continue;
        auto p = applyRigid(A, std::array<double, 3>{xp[ps * 3], xp[ps * 3 + 1], xp[ps * 3 + 2]});
        for (int c = 0; c < 3; ++c) { const double v = p[c] - P.xt[it->second * 3 + c]; s2 += v * v; }
        cnt += 1;
      }
      const double rmsd = cnt > 0 ? std::sqrt(s2 / cnt) : std::numeric_limits<double>::quiet_NaN();
      if (rmsd < bestD) { bestD = rmsd; pick = q; }
    }
    for (auto& [nm, ps] : pdSlot) {
      auto it = gtSlot.find(namings[pick].at(nm));
      if (it == gtSlot.end()) continue;
      for (int c = 0; c < 3; ++c) xtOut[ps * 3 + c] = P.xt[it->second * 3 + c];
      gmOut[ps] = P.gm[it->second];
    }
  }
  if (report) {
    *report = "";
    for (auto& [es, gs] : best) if (es.second != gs) *report += " entity " + std::to_string(es.first) + " copy " + std::to_string(es.second) + "->" + std::to_string(gs);
  }
}

}  // namespace of3
