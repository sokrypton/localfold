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
#include <array>
#include <vector>
#include "../../featurise/jsmath.h"
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
  const vector<double>* xyz = nullptr;   // [n, 3], or the structure's three separate columns
  const cif::AtomSite* atoms = nullptr;
  double at(int i, int k) const {
    if (xyz) return (*xyz)[(size_t)i * 3 + k];
    return k == 0 ? atoms->x[i] : (k == 1 ? atoms->y[i] : atoms->z[i]);
  }
};

inline Coords resolveStereoCoords(const vector<int>& indices, const cif::Annotations& ann,
                                  const cif::AtomSite& a) {
  auto finite3 = [](const vector<double>& v, int i) {
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

// ---------------------------------------------------------------- the Kabsch fit
//
// `alignVectors`: the proper rotation carrying `b` onto `a`, through the eigendecomposition of
// B^T B rather than a general SVD. Ported literally, including the sort below.
using Mat3 = std::array<std::array<double, 3>, 3>;
using Vec3 = std::array<double, 3>;

inline Mat3 matMul3(const Mat3& A, const Mat3& B) {
  Mat3 C{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return C;
}
inline Mat3 transpose3(const Mat3& A) {
  Mat3 T{};
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) T[i][j] = A[j][i];
  return T;
}
inline double det3(const Mat3& m) {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
inline Vec3 cross3(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline Vec3 normalize3(const Vec3& v) {
  const double n = lf::jsmath::hypot(v[0], v[1], v[2]);
  return n > 1e-12 ? Vec3{v[0] / n, v[1] / n, v[2] / n} : Vec3{0, 0, 0};
}

inline void jacobiEigen3(const Mat3& A, Vec3& eigVals, Mat3& eigVecs) {
  Mat3 S = A;
  Mat3 V{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  for (int iter = 0; iter < 50; ++iter) {
    int p = 0, q = 1;
    double maxVal = std::fabs(S[0][1]);
    if (std::fabs(S[0][2]) > maxVal) { maxVal = std::fabs(S[0][2]); p = 0; q = 2; }
    if (std::fabs(S[1][2]) > maxVal) { maxVal = std::fabs(S[1][2]); p = 1; q = 2; }
    if (maxVal < 1e-15) break;
    const double app = S[p][p], aqq = S[q][q], apq = S[p][q];
    const double tau = (aqq - app) / (2 * apq);
    const double t = tau >= 0 ? 1 / (tau + std::sqrt(1 + tau * tau))
                              : -1 / (-tau + std::sqrt(1 + tau * tau));
    const double c = 1 / std::sqrt(1 + t * t), sn = t * c;
    S[p][p] = app - t * apq;
    S[q][q] = aqq + t * apq;
    S[p][q] = S[q][p] = 0;
    for (int r = 0; r < 3; ++r) {
      if (r != p && r != q) {
        const double srp = S[r][p], srq = S[r][q];
        S[r][p] = S[p][r] = c * srp - sn * srq;
        S[r][q] = S[q][r] = sn * srp + c * srq;
      }
      const double vrp = V[r][p], vrq = V[r][q];
      V[r][p] = c * vrp - sn * vrq;
      V[r][q] = sn * vrp + c * vrq;
    }
  }
  // 🔴 THE COMPARATOR IS `S[j][j] - S[i][j]`, AND THAT SECOND INDEX IS THE REFERENCE'S. It reads
  // an OFF-diagonal where it plainly means `S[i][i]`, so the order is not a sort by eigenvalue at
  // all - after convergence the off-diagonals are ~0 and the comparator is just `S[j][j]`. Ported
  // as written, with V8's own binary insertion sort, because the eigenvector ORDER decides the
  // rotation and "fixing" it here would disagree with every number the page produces.
  int order[3] = {0, 1, 2};
  auto cmp = [&](int i, int j) { return S[j][j] - S[i][j]; };
  for (int i = 1; i < 3; ++i) {
    const int pivot = order[i];
    int left = 0, right = i;
    while (left < right) {
      const int mid = (left + right) >> 1;
      if (cmp(pivot, order[mid]) < 0) right = mid; else left = mid + 1;
    }
    for (int k = i; k > left; --k) order[k] = order[k - 1];
    order[left] = pivot;
  }
  for (int i = 0; i < 3; ++i) eigVals[i] = std::max(0.0, S[order[i]][order[i]]);
  for (int r = 0; r < 3; ++r) for (int i = 0; i < 3; ++i) eigVecs[r][i] = V[r][order[i]];
  if (det3(eigVecs) < 0) for (int r = 0; r < 3; ++r) eigVecs[r][2] = -eigVecs[r][2];
}

struct Alignment {
  Mat3 rotation{};
  double rssd = 0;
};

inline Alignment alignVectors(const vector<Vec3>& a, const vector<Vec3>& b) {
  Mat3 B{};
  double normSqSum = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    for (int k = 0; k < 3; ++k) normSqSum += a[i][k] * a[i][k] + b[i][k] * b[i][k];
    for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q) B[p][q] += a[i][p] * b[i][q];
  }
  Vec3 eigVals{};
  Mat3 V{};
  jacobiEigen3(matMul3(transpose3(B), B), eigVals, V);
  Vec3 sv{std::sqrt(std::max(0.0, eigVals[0])), std::sqrt(std::max(0.0, eigVals[1])),
          std::sqrt(std::max(0.0, eigVals[2]))};
  const Mat3 BV = matMul3(B, V);
  Vec3 u[3]{};
  u[0] = sv[0] > 1e-10 ? normalize3({BV[0][0] / sv[0], BV[1][0] / sv[0], BV[2][0] / sv[0]})
                       : Vec3{1, 0, 0};
  if (sv[1] > 1e-10) {
    Vec3 v1{BV[0][1] / sv[1], BV[1][1] / sv[1], BV[2][1] / sv[1]};
    const double d = v1[0] * u[0][0] + v1[1] * u[0][1] + v1[2] * u[0][2];
    u[1] = normalize3({v1[0] - d * u[0][0], v1[1] - d * u[0][1], v1[2] - d * u[0][2]});
  } else {
    const Vec3 cand = std::fabs(u[0][0]) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    const double d = cand[0] * u[0][0] + cand[1] * u[0][1] + cand[2] * u[0][2];
    u[1] = normalize3({cand[0] - d * u[0][0], cand[1] - d * u[0][1], cand[2] - d * u[0][2]});
  }
  u[2] = cross3(u[0], u[1]);
  const double dot2 = BV[0][2] * u[2][0] + BV[1][2] * u[2][1] + BV[2][2] * u[2][2];
  const double sign2 = (sv[2] > 1e-10 && dot2 < 0) ? -1 : 1;
  Mat3 U{};
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) U[r][c] = u[c][r];
  Alignment out;
  out.rotation = matMul3(U, transpose3(V));
  out.rssd = std::sqrt(std::max(0.0, normSqSum - 2 * (sv[0] + sv[1] + sign2 * sv[2])));
  return out;
}

inline vector<Vec3> applyRotation(const Mat3& R, const vector<Vec3>& pts) {
  vector<Vec3> out(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    for (int r = 0; r < 3; ++r) {
      out[i][r] = R[r][0] * pts[i][0] + R[r][1] * pts[i][1] + R[r][2] * pts[i][2];
    }
  }
  return out;
}

inline vector<Vec3> unitVectors(const vector<Vec3>& centered) {
  vector<Vec3> unit(centered.size());
  for (size_t i = 0; i < centered.size(); ++i) {
    const double n = std::max(lf::jsmath::hypot(centered[i][0], centered[i][1], centered[i][2]), 1e-8);
    for (int k = 0; k < 3; ++k) unit[i][k] = centered[i][k] / n;
  }
  return unit;
}

/** `idealizeCoords`: the template orientation closest to the observed one, rotated onto it. */
inline vector<Vec3> idealizeCoords(const vector<Vec3>& centered, const string& geometry) {
  auto it = tables::STEREOISOMER_TEMPLATES.find(geometry + ":" + std::to_string(centered.size()));
  bool finite = true;
  for (const Vec3& p : centered) {
    finite = finite && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
  }
  if (it == tables::STEREOISOMER_TEMPLATES.end() || !finite) {
    vector<Vec3> out(centered.size());
    for (size_t i = 0; i < centered.size(); ++i) {
      for (int k = 0; k < 3; ++k) out[i][k] = (float)centered[i][k];
    }
    return out;
  }
  const vector<Vec3> unit = unitVectors(centered);
  size_t best = 0;
  double bestRmsd = INFINITY;
  const double sqrtN = std::sqrt((double)centered.size());
  for (size_t i = 0; i < it->second.size(); ++i) {
    const double rmsd = alignVectors(unit, it->second[i]).rssd / sqrtN;
    if (rmsd < bestRmsd) { bestRmsd = rmsd; best = i; }
  }
  const vector<Vec3> rotated = applyRotation(alignVectors(unit, it->second[best]).rotation,
                                             it->second[best]);
  vector<Vec3> out(rotated.size());
  for (size_t i = 0; i < rotated.size(); ++i) {
    for (int k = 0; k < 3; ++k) out[i][k] = (float)rotated[i][k];
  }
  return out;
}

/** `classifyDoubleBondStereoisomer`: +1 for the substituents on one side, -1 for the other. */
inline int classifyDoubleBond(const vector<Vec3>& c, bool& ok) {
  ok = false;
  for (const Vec3& p : c) {
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) return 0;
  }
  Vec3 axis{c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]};
  const double len = lf::jsmath::hypot(axis[0], axis[1], axis[2]);
  if (len < 1e-6) return 0;
  for (int k = 0; k < 3; ++k) axis[k] /= len;
  Vec3 sub0{c[2][0] - c[0][0], c[2][1] - c[0][1], c[2][2] - c[0][2]};
  Vec3 sub1{c[3][0] - c[1][0], c[3][1] - c[1][1], c[3][2] - c[1][2]};
  const double d0 = sub0[0] * axis[0] + sub0[1] * axis[1] + sub0[2] * axis[2];
  const double d1 = sub1[0] * axis[0] + sub1[1] * axis[1] + sub1[2] * axis[2];
  Vec3 p0{}, p1{};
  for (int k = 0; k < 3; ++k) { p0[k] = sub0[k] - d0 * axis[k]; p1[k] = sub1[k] - d1 * axis[k]; }
  const double l0 = lf::jsmath::hypot(p0[0], p0[1], p0[2]);
  const double l1 = lf::jsmath::hypot(p1[0], p1[1], p1[2]);
  if (l0 < 1e-6 || l1 < 1e-6) return 0;
  const double cosine = (p0[0] * p1[0] + p0[1] * p1[1] + p0[2] * p1[2]) / (l0 * l1);
  if (std::fabs(cosine) < 0.5) return 0;
  ok = true;
  return cosine > 0 ? 1 : -1;
}

// ---------------------------------------------------------------- the four tensors
//
// `computeChiralityFeatures`. `tokenIds` and `atomize` come from the TRANSFORMS, which the native
// side does not have yet; passed empty, the frame and owner tensors stay at -1 exactly as the
// reference leaves them when a raw atom array has no `token_id` - which is the configuration this
// is gated in.
struct Features {
  vector<float> chirality;        // [n, 6 * 3]
  vector<int> frameAtoms;         // [n, 6, 8]
  vector<int> slotOwner;          // [n, 6]
  vector<int> tetrahedralVertices;// [n, 4]
  vector<uint8_t> isChiralCenter; // [n]
};

struct Center {
  int idx = 0;
  vector<int> bonded;
  string geometry;
};

inline Features features(const cif::AtomSite& a, const vector<cif::Bond>& bonds,
                         const cif::Annotations& ann, const Stereo& stereo,
                         const vector<int>& tokenIds, const vector<uint8_t>& atomize) {
  const int n = (int)a.n;
  const int nBlocks = tables::MAX_OVERLAPPING_CENTERS;
  const int embedDim = nBlocks * 3;
  Features out;
  out.chirality.assign((size_t)n * embedDim, 0.f);
  out.frameAtoms.assign((size_t)n * nBlocks * 8, -1);
  out.slotOwner.assign((size_t)n * nBlocks, -1);
  out.tetrahedralVertices.assign((size_t)n * 4, -1);
  out.isChiralCenter.assign(n, 0);

  vector<vector<int>> adj(n);
  for (const cif::Bond& b : bonds) { adj[b.u].push_back(b.v); adj[b.v].push_back(b.u); }

  // The centres, by their annotated chiral type. Only the geometries with a template count.
  vector<Center> centers;
  for (int i = 0; i < n; ++i) {
    const int ct = stereo.chiralType[i];
    string geometry;
    if (ct == 1 || ct == 2 || ct == 3) geometry = "tetrahedral";      // CW, CCW, unspecified-hand
    else if (ct == 4) geometry = "square_planar";
    else if (ct == 5) geometry = "trigonal_bipyramidal";
    else if (ct == 6) geometry = "octahedral";
    if (geometry.empty()) continue;
    vector<int> bonded = adj[i];
    std::sort(bonded.begin(), bonded.end());
    if (!tables::STEREOISOMER_TEMPLATES.count(geometry + ":" + std::to_string(bonded.size()))) {
      continue;
    }
    centers.push_back({i, bonded, geometry});
  }

  // 🔴 `tetrahedral_vertices` IS FILLED ONLY WHERE THE CALLER HAS TOKEN IDS, and until the batch
  // was built nobody passed any - so this stayed at -1 and the gate, passing `{}` on both sides,
  // agreed with it. It is the four TOKENS around a tetrahedral centre, and it exists only where
  // the centre and all its neighbours are atomised ligand atoms: a polymer stereocentre is
  // described by the frames above instead.
  //
  // 🔴 AND A THREE-COORDINATE CENTRE USES ITSELF AS THE FOURTH VERTEX. The lone pair has no
  // atom, so the centre stands in for it - dropping to three entries would change the shape and
  // padding with -1 would name a vertex that is not there.
  if (!tokenIds.empty()) {
    for (const Center& c : centers) {
      if (c.geometry != "tetrahedral") continue;
      if (c.bonded.size() != 3 && c.bonded.size() != 4) continue;
      vector<int> support{c.idx};
      support.insert(support.end(), c.bonded.begin(), c.bonded.end());
      bool ok = true;
      for (const int idx : support) {
        const bool ligand = !a.isPolymer[idx];
        if (!ligand || atomize.empty() || !atomize[idx]) { ok = false; break; }
      }
      if (!ok) continue;
      vector<int> v = c.bonded;
      if (v.size() == 3) v.push_back(c.idx);
      for (int k = 0; k < 4; ++k) out.tetrahedralVertices[(size_t)c.idx * 4 + k] = tokenIds[v[k]];
    }
  }

  struct Frame { bool isCenter; int cidx; vector<int> bonded; vector<Vec3> rotated; };
  vector<Frame> frames;
  for (const Center& c : centers) {
    if (!atomize.empty() && !atomize[c.idx]) continue;
    vector<int> all{c.idx};
    for (int b : c.bonded) all.push_back(b);
    const Coords src = resolveStereoCoords(all, ann, a);
    vector<Vec3> centered;
    for (int b : c.bonded) {
      centered.push_back({src.at(b, 0) - src.at(c.idx, 0), src.at(b, 1) - src.at(c.idx, 1),
                          src.at(b, 2) - src.at(c.idx, 2)});
    }
    frames.push_back({true, c.idx, c.bonded, idealizeCoords(centered, c.geometry)});
  }

  // The stereogenic double bonds, each aligned onto its planar template.
  for (const cif::Bond& b : bonds) {
    if (b.type != tables::BT_DOUBLE) continue;
    const int left = b.u, right = b.v;
    if (!stereo.stereogenicBondAtoms[left] || !stereo.stereogenicBondAtoms[right]) continue;
    if (!atomize.empty() && (!atomize[left] || !atomize[right])) continue;
    vector<int> ln, rn;
    for (int x : adj[left]) if (x != right) ln.push_back(x);
    for (int x : adj[right]) if (x != left) rn.push_back(x);
    std::sort(ln.begin(), ln.end());
    std::sort(rn.begin(), rn.end());
    if (ln.empty() || ln.size() > 2 || rn.empty() || rn.size() > 2) continue;
    vector<int> indices{left, right};
    for (int x : ln) indices.push_back(x);
    for (int x : rn) indices.push_back(x);
    const Coords src = resolveStereoCoords(indices, ann, a);
    vector<Vec3> coords;
    bool finite = true;
    for (int idx : indices) {
      coords.push_back({src.at(idx, 0), src.at(idx, 1), src.at(idx, 2)});
      finite = finite && std::isfinite(coords.back()[0]) && std::isfinite(coords.back()[1]) &&
               std::isfinite(coords.back()[2]);
    }
    if (!finite) continue;
    const vector<int> refIdx{left, right, ln[0], rn[0]};
    vector<Vec3> ref;
    for (int idx : refIdx) ref.push_back({src.at(idx, 0), src.at(idx, 1), src.at(idx, 2)});
    bool ok = false;
    const int side = classifyDoubleBond(ref, ok);
    if (!ok) continue;
    auto tpl = tables::DOUBLE_BOND_TEMPLATES.find(
        std::to_string(side) + ":" + std::to_string(ln.size()) + ":" + std::to_string(rn.size()));
    if (tpl == tables::DOUBLE_BOND_TEMPLATES.end()) continue;
    Vec3 mid{(src.at(left, 0) + src.at(right, 0)) / 2, (src.at(left, 1) + src.at(right, 1)) / 2,
             (src.at(left, 2) + src.at(right, 2)) / 2};
    vector<Vec3> centered;
    for (const Vec3& c : coords) centered.push_back({c[0] - mid[0], c[1] - mid[1], c[2] - mid[2]});
    const vector<Vec3> rotated = applyRotation(alignVectors(centered, tpl->second).rotation,
                                               tpl->second);
    vector<Vec3> rounded(rotated.size());
    for (size_t i = 0; i < rotated.size(); ++i) {
      for (int k = 0; k < 3; ++k) rounded[i][k] = (float)rotated[i][k];
    }
    frames.push_back({false, left, indices, rounded});
  }

  // 🔴 EACH FRAME TAKES THE FIRST BLOCK NONE OF ITS ATOMS ALREADY OCCUPIES, which is what makes
  // six blocks enough for overlapping centres - two centres sharing an atom land in different
  // blocks, and a block is only free if EVERY atom of the frame is free in it.
  std::map<int, vector<int>> atomBlocks;
  for (const Frame& f : frames) {
    for (int block = 0; block < nBlocks; ++block) {
      const int offset = block * 3;
      bool occupied = false;
      for (int b : f.bonded) {
        const size_t base = (size_t)b * embedDim + offset;
        if (out.chirality[base] != 0 || out.chirality[base + 1] != 0 ||
            out.chirality[base + 2] != 0) {
          occupied = true;
          break;
        }
      }
      if (occupied) continue;
      if (!tokenIds.empty()) {
        vector<int> support{f.isCenter ? f.cidx : f.bonded[0], f.isCenter ? f.cidx : f.bonded[1]};
        for (int b : f.bonded) support.push_back(b);
        bool all = true;
        for (int idx : support) all = all && (atomize.empty() || atomize[idx]);
        if (all) {
          const size_t frameBase = ((size_t)f.cidx * nBlocks + block) * 8;
          for (size_t k = 0; k < support.size() && k < 8; ++k) {
            out.frameAtoms[frameBase + k] = tokenIds[support[k]];
          }
          for (int b : f.bonded) {
            out.slotOwner[(size_t)b * nBlocks + block] = tokenIds[f.cidx] * nBlocks + block;
          }
        }
      }
      for (size_t k = 0; k < f.bonded.size(); ++k) {
        const size_t base = (size_t)f.bonded[k] * embedDim + offset;
        for (int ax = 0; ax < 3; ++ax) out.chirality[base + ax] = (float)f.rotated[k][ax];
        atomBlocks[f.bonded[k]].push_back(block);
      }
      break;
    }
  }

  // The unoccupied blocks of an atom repeat its occupied ones, cycling - so every block carries
  // a direction rather than a zero the model would read as "no chirality here".
  for (auto& [atomIdx, blocks] : atomBlocks) {
    vector<int> occupied = blocks;
    std::sort(occupied.begin(), occupied.end());
    vector<int> empty;
    for (int b = 0; b < nBlocks; ++b) {
      if (std::find(occupied.begin(), occupied.end(), b) == occupied.end()) empty.push_back(b);
    }
    for (size_t i = 0; i < empty.size(); ++i) {
      const int target = empty[i], source = occupied[i % occupied.size()];
      const size_t dst = (size_t)atomIdx * embedDim + target * 3;
      const size_t src = (size_t)atomIdx * embedDim + source * 3;
      for (int ax = 0; ax < 3; ++ax) out.chirality[dst + ax] = out.chirality[src + ax];
      if (!tokenIds.empty()) {
        out.slotOwner[(size_t)atomIdx * nBlocks + target] =
            out.slotOwner[(size_t)atomIdx * nBlocks + source];
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    if (stereo.chiralType[i] != 0 && (atomize.empty() || atomize[i])) out.isChiralCenter[i] = 1;
  }
  return out;
}

}  // namespace chirality
}  // namespace rfd4p
