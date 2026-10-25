// A structure's stereochemistry: which atoms are chiral centres and which double bonds are
// stereogenic. Ported from `shared/rfd4p/pipeline/chirality.js`'s `ensureStereochemistryAnnotations`.
//
// This is the FIRST half of the chirality work - the per-atom annotation. The second is
// `computeChiralityFeatures`, which turns these into the four tensors the model reads and needs
// the Kabsch fit and the stereoisomer templates (both generated into `tables.inc` already).
//
// 🔴 A STANDARD RESIDUE DOES NOT GET DETECTED, IT GETS LOOKED UP. An alanine's CA is a chiral
// centre by definition and `STD_POLYMER_CHIRAL_CENTERS` says so; running the detector over it
// would ask whether its four neighbours are topologically distinct, which for a backbone they
// are not always, and would drop centres the model expects. The detector is for everything else.
#pragma once
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "cif.h"

namespace rfd4p {
namespace chirality {

using std::string;
using std::vector;

struct Edge {
  int to = 0, type = 0;
};

/**
 * `computeWlClasses`: a six-round Weisfeiler-Lehman colouring over one residue's graph, with one
 * atom optionally removed.
 *
 * 🔴 REMOVING THE CENTRE IS THE POINT. A tetrahedral atom is a stereocentre only if its
 * neighbours are topologically DISTINCT, and they are only distinct in the graph that does not
 * contain the centre joining them - with it, every neighbour is one bond from the same atom.
 */
inline std::map<int, string> wlClasses(const cif::AtomSite& a, const vector<int>& subset,
                                       const std::map<int, vector<Edge>>& adj, int exclude) {
  std::map<int, string> labels;
  for (int u : subset) {
    if (u == exclude) continue;
    int degree = 0;
    auto it = adj.find(u);
    if (it != adj.end()) {
      for (const Edge& e : it->second) if (e.to != exclude) ++degree;
    }
    labels[u] = std::to_string(a.atomicNumber[u]) + ":" + std::to_string(a.charge[u]) + ":" +
                std::to_string((int)a.isAromatic[u]) + ":" + std::to_string(degree);
  }
  for (int iter = 0; iter < 6; ++iter) {
    std::map<int, string> next;
    for (int u : subset) {
      if (u == exclude) continue;
      vector<string> around;
      auto it = adj.find(u);
      if (it != adj.end()) {
        for (const Edge& e : it->second) {
          if (e.to == exclude || !labels.count(e.to)) continue;
          around.push_back(std::to_string(e.type) + ":" + labels[e.to]);
        }
      }
      std::sort(around.begin(), around.end());
      string joined;
      for (size_t k = 0; k < around.size(); ++k) joined += (k ? "," : "") + around[k];
      next[u] = labels[u] + "(" + joined + ")";
    }
    labels = next;
  }
  return labels;
}

/**
 * `resolveStereoCoords`: the conditioned coordinates where every atom of the set has them, then
 * the internal ones where they share a coordinate group, then the structure's own.
 *
 * 🔴 ALL-OR-NOTHING OVER THE SET, not per atom. A handedness read half from a conditioned frame
 * and half from the raw one is a handedness measured in no frame at all.
 */
struct Coords {
  const vector<float>* xyz = nullptr;   // [n, 3], or the structure's three separate columns
  const cif::AtomSite* atoms = nullptr;
  double at(int i, int k) const {
    if (xyz) return (*xyz)[(size_t)i * 3 + k];
    return k == 0 ? atoms->x[i] : (k == 1 ? atoms->y[i] : atoms->z[i]);
  }
};

inline Coords resolveStereoCoords(const vector<int>& indices, const cif::Annotations& ann,
                                  const cif::AtomSite& a) {
  auto finite3 = [](const vector<float>& v, int i) {
    return std::isfinite(v[(size_t)i * 3]) && std::isfinite(v[(size_t)i * 3 + 1]) &&
           std::isfinite(v[(size_t)i * 3 + 2]);
  };
  auto crd = ann.floats.find("condition_coordinate_1_atom");
  if (crd != ann.floats.end()) {
    bool all = true;
    for (int i : indices) all = all && finite3(crd->second, i);
    if (all) return {&crd->second, &a};
  }
  auto intcrd = ann.floats.find("condition_InternalCoordinate_1_atom");
  auto groups = ann.ints.find("condition_CoordinateGroup_1_atom");
  if (intcrd != ann.floats.end() && groups != ann.ints.end() && !indices.empty()) {
    bool all = true;
    for (int i : indices) {
      all = all && finite3(intcrd->second, i) &&
            groups->second[i] == groups->second[indices[0]];
    }
    if (all) return {&intcrd->second, &a};
  }
  return {nullptr, &a};
}

struct Stereo {
  vector<int> chiralType;
  vector<uint8_t> stereogenicBondAtoms;
};

/** `ensureStereochemistryAnnotations`. `atomize` may be empty, meaning "not a polymer". */
inline Stereo annotate(const cif::AtomSite& a, const vector<cif::Bond>& bonds,
                       const cif::Annotations& ann, const vector<uint8_t>& atomize) {
  const int n = (int)a.n;
  Stereo out;
  out.chiralType.assign(n, 0);
  out.stereogenicBondAtoms.assign(n, 0);

  std::map<int, vector<Edge>> adj;
  for (int i = 0; i < n; ++i) adj[i];
  for (const cif::Bond& b : bonds) {
    if (b.type == tables::BT_COORDINATION) continue;
    adj[b.u].push_back({b.v, b.type});
    adj[b.v].push_back({b.u, b.type});
  }

  const vector<int> starts = cif::residueSegments(a);
  for (size_t r = 0; r + 1 < starts.size(); ++r) {
    const int s = starts[r], e = starts[r + 1];
    const string& resName = a.resName[s];
    // A masked residue and an expandable segment's sentinel have no chemistry to read.
    bool range = resName.find('-') != string::npos;
    for (char ch : resName) range = range && (isdigit((unsigned char)ch) || ch == '-');
    if (resName == "<M>" || range) continue;

    const bool atomized = atomize.empty() ? !a.isPolymer[s] : atomize[s] != 0;
    if (!atomized) {
      bool any = false;
      for (int i = s; i < e; ++i) {
        auto it = tables::STD_POLYMER_CHIRAL_CENTERS.find(resName + ":" + a.atomName[i]);
        if (it != tables::STD_POLYMER_CHIRAL_CENTERS.end()) { out.chiralType[i] = it->second; any = true; }
      }
      // The reference `continue`s whenever the residue IS in the table, found centres or not.
      bool known = false;
      for (const auto& [key, type] : tables::STD_POLYMER_CHIRAL_CENTERS) {
        if (key.rfind(resName + ":", 0) == 0) { known = true; break; }
      }
      (void)any;
      if (known) continue;
    }

    vector<int> residue;
    for (int i = s; i < e; ++i) residue.push_back(i);

    for (int c : residue) {
      if (a.isAromatic[c]) continue;
      const vector<Edge>& nbs = adj[c];
      if (nbs.size() != 3 && nbs.size() != 4) continue;
      bool allSingle = true;
      for (const Edge& edge : nbs) allSingle = allSingle && edge.type == tables::BT_SINGLE;
      if (!allSingle) continue;
      const int z = a.atomicNumber[c], q = a.charge[c];
      if (nbs.size() == 3) {
        // Three heavy neighbours is sp3 only for a neutral carbon (one implicit hydrogen) or a
        // trivalent sulfur or phosphorus (a lone pair).
        if (!((z == 6 && q == 0) || z == 16 || z == 15)) continue;
      } else {
        if (z != 6 && z != 7 && z != 14 && z != 15 && z != 16) continue;
      }
      // A phosphate or sulfate centre is not a stereocentre however it looks.
      if (z == 15 || z == 16) {
        int oxygens = 0;
        for (const Edge& edge : nbs) oxygens += a.atomicNumber[edge.to] == 8 ? 1 : 0;
        if (oxygens >= 2) continue;
      }
      if (a.isAromatic[c]) {
        bool metal = false;
        for (const Edge& edge : nbs) {
          string el = a.element[edge.to];
          for (char& ch : el) ch = (char)toupper((unsigned char)ch);
          metal = metal || tables::METAL_ELEMENTS.count(el) > 0;
        }
        if (metal) continue;
      }
      const std::map<int, string> wl = wlClasses(a, residue, adj, c);
      std::set<string> distinct;
      for (const Edge& edge : nbs) {
        auto it = wl.find(edge.to);
        distinct.insert(it == wl.end() ? string() : it->second);
      }
      if (distinct.size() != nbs.size()) continue;

      vector<int> all{c};
      for (const Edge& edge : nbs) all.push_back(edge.to);
      const Coords src = resolveStereoCoords(all, ann, a);
      if (!std::isfinite(src.at(c, 0))) continue;
      double v[3][3];
      bool finite = true;
      for (int k = 0; k < 3; ++k) {
        for (int ax = 0; ax < 3; ++ax) v[k][ax] = src.at(nbs[k].to, ax) - src.at(c, ax);
        finite = finite && std::isfinite(v[k][0]);
      }
      if (!finite) continue;
      const double cr[3] = {v[1][1] * v[2][2] - v[1][2] * v[2][1],
                            v[1][2] * v[2][0] - v[1][0] * v[2][2],
                            v[1][0] * v[2][1] - v[1][1] * v[2][0]};
      const double signedTp = v[0][0] * cr[0] + v[0][1] * cr[1] + v[0][2] * cr[2];
      const double norms = lf::jsmath::hypot(v[0][0], v[0][1], v[0][2]) *
                           lf::jsmath::hypot(v[1][0], v[1][1], v[1][2]) *
                           lf::jsmath::hypot(v[2][0], v[2][1], v[2][2]);
      // 🔴 A FLAT CENTRE IS NOT A CENTRE. Four neighbours in a plane have no handedness, and the
      // normalised triple product is what says so - without this an aromatic-adjacent carbon
      // picks up a sign from rounding.
      if (std::fabs(signedTp) / std::max(norms, 1e-8) < 0.05) continue;
      out.chiralType[c] = signedTp >= 0 ? 1 : 2;   // CW, CCW
    }

    // A double bond is stereogenic when each end has one or two neighbours and, where it has
    // two, they are topologically different - otherwise there is nothing for the sides to be.
    const std::map<int, string> wlFull = wlClasses(a, residue, adj, -1);
    for (int u : residue) {
      for (const Edge& edge : adj[u]) {
        const int v = edge.to;
        if (v <= u || v < s || v >= e || edge.type != tables::BT_DOUBLE) continue;
        if (a.isAromatic[u] || a.isAromatic[v]) continue;
        vector<int> left, right;
        for (const Edge& e2 : adj[u]) if (e2.to != v) left.push_back(e2.to);
        for (const Edge& e2 : adj[v]) if (e2.to != u) right.push_back(e2.to);
        if (left.empty() || left.size() > 2 || right.empty() || right.size() > 2) continue;
        auto label = [&](int i) {
          auto it = wlFull.find(i);
          return it == wlFull.end() ? string() : it->second;
        };
        if (left.size() == 2 && label(left[0]) == label(left[1])) continue;
        if (right.size() == 2 && label(right[0]) == label(right[1])) continue;
        out.stereogenicBondAtoms[u] = 1;
        out.stereogenicBondAtoms[v] = 1;
      }
    }
  }
  return out;
}

}  // namespace chirality
}  // namespace rfd4p
