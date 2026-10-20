// BioEmu2's steering, natively: shared/bioemu2/potentials.js (the potentials' energies with their hand-written
// gradients, their parameter schedules, potentialSpecs and hostTables) and sampler.js's steeredSample (the edm_gd and
// edm_fk presets: gradient guidance on the denoised estimate, Feynman-Kac weights, systematic resampling), transcribed
// line for line and held to them (tools/check-native-bioemu2-steering.mjs: the energies and gradients to the last bit
// or 1e-12, the sampler on a toy denoiser to the JavaScript and to af3-any-model's own recorded run). Host code, no
// CUDA: the denoiser is the caller's (bioemu2.cu's on the device, the probe's toy), and so is every random draw.
//
// The tables are shared/bioemu2/steering-tables.js's, written into the input by the featuriser (`--steering`:
// export_input.mjs, cuda/featurise's bioemu2-featurise) and read here; hostTables is applied for the potentials in use.
// edm_fkc (the corrector, which needs the denoiser's vector-Jacobian product) is not here.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../featurise/jsmath.h"

namespace be2steer {

constexpr double SIGMA_DATA = 16.0, SIGMA_MIN = 0.0004, SIGMA_MAX = 160.0, RHO = 7;
constexpr double INF = std::numeric_limits<double>::infinity();
inline double fr(double v) { return (double)(float)v; }        // Math.fround

// ---- schedules and specs ------------------------------------------------------------------------------------------
// a parameter: a constant, or one of BioEmu2's three schedules of normalised time t
struct Param {
  enum Kind { CONST, EXPONENTIAL, PIECEWISE, LINEAR } kind = CONST;
  double value = 0, start = 0, end = 0, alpha = 0, startT = 0, endT = 0, startVal = 0, endVal = 0;
  std::vector<double> thresholds, values;
  static Param constant(double v) { Param p; p.value = v; return p; }
  static Param exponential(double s, double e, double a) { Param p; p.kind = EXPONENTIAL; p.start = s; p.end = e; p.alpha = a; return p; }
  static Param piecewise(std::vector<double> t, std::vector<double> v) { Param p; p.kind = PIECEWISE; p.thresholds = t; p.values = v; return p; }
  double at(double t) const {
    switch (kind) {
      case CONST: return value;
      // (Math.exp is V8's fdlibm and std::exp glibc's: they can differ in the last place, which the fround after
      // every resolve - stepParameters - all but always removes)
      case EXPONENTIAL: return alpha != 0 ? start + ((end - start) * (std::exp(alpha * t) - 1)) / (std::exp(alpha) - 1)
                                          : start + (end - start) * t;
      case PIECEWISE: { size_t i = 0; while (i < thresholds.size() && t > thresholds[i]) ++i; return values.at(i); }
      case LINEAR: return t > startT || t < endT ? 0.0 : startVal - ((startVal - endVal) * (startT - t)) / (startT - endT);
    }
    return 0;
  }
};
struct Spec {
  std::string kind;
  std::map<std::string, Param> params;
  double sigmaTolerance = std::numeric_limits<double>::quiet_NaN();
  bool has(const std::string& k) const { return params.count(k) > 0; }
};

inline const std::vector<std::string>& types() {
  static const std::vector<std::string> t = {
      "SymmetricChainCOMPotential", "InterChainVDWOverlapPotential", "PoseBustersPotential", "ChiralAtomPotential",
      "StereoBondPotential", "PlanarBondPotential", "IntraChainVDWOverlapPotential", "PolymerLocalClashPotential",
      "BondLengthPotential", "BondAnglePotential"};
  return t;
}
inline const std::vector<std::string>& paramNames() {
  static const std::vector<std::string> p = {"buffer", "bond_buffer", "angle_buffer", "clash_buffer", "minimum_distance"};
  return p;
}

// get_default_potentials (potentials.js DEFAULT_SPECS)
inline std::vector<Spec> defaultSpecs() {
  using P = Param;
  auto spec = [](std::string kind, std::map<std::string, Param> p, double tol = std::numeric_limits<double>::quiet_NaN()) {
    Spec s; s.kind = kind; s.params = p; s.sigmaTolerance = tol; return s;
  };
  const P step125 = P::piecewise({0.4}, {0.125, 0.0}), step01 = P::piecewise({0.6}, {0.01, 0.0});
  return {
      spec("SymmetricChainCOMPotential", {{"guidance_interval", P::constant(4)}, {"guidance_weight", P::constant(0.5)},
                                          {"resampling_weight", P::constant(0.5)}, {"buffer", P::exponential(1.0, 5.0, -2.0)}}),
      spec("InterChainVDWOverlapPotential", {{"guidance_interval", P::constant(5)}, {"guidance_weight", step125},
                                             {"resampling_weight", step01}, {"buffer", P::constant(0.225)}}),
      spec("PoseBustersPotential", {{"guidance_interval", P::constant(1)}, {"guidance_weight", P::constant(0.10)},
                                    {"resampling_weight", P::constant(0.1)}, {"bond_buffer", P::constant(0.20)},
                                    {"angle_buffer", P::constant(0.20)}, {"clash_buffer", P::constant(0.15)}}),
      spec("ChiralAtomPotential", {{"guidance_interval", P::constant(1)}, {"guidance_weight", P::constant(0.20)},
                                   {"resampling_weight", P::constant(1.0)}, {"buffer", P::constant(0.52360)}}),
      spec("StereoBondPotential", {{"guidance_interval", P::constant(1)}, {"guidance_weight", P::constant(0.10)},
                                   {"resampling_weight", P::constant(1.0)}, {"buffer", P::constant(0.52360)}}),
      spec("PlanarBondPotential", {{"guidance_interval", P::constant(1)}, {"guidance_weight", P::constant(0.10)},
                                   {"resampling_weight", P::constant(1.0)}, {"buffer", P::constant(0.26180)}}),
      spec("IntraChainVDWOverlapPotential", {{"guidance_interval", P::constant(5)}, {"guidance_weight", step125},
                                             {"resampling_weight", step01}, {"buffer", P::constant(0.225)}}),
      spec("PolymerLocalClashPotential", {{"guidance_interval", P::constant(1)}, {"guidance_weight", step125},
                                          {"resampling_weight", P::constant(0.1)}, {"minimum_distance", P::constant(2.0)}}),
      spec("BondLengthPotential", {{"guidance_interval", P::constant(1)}, {"guidance_weight", P::constant(0.05)},
                                   {"resampling_weight", P::constant(0.1)}}, 10.0),
      spec("BondAnglePotential", {{"guidance_interval", P::constant(1)}, {"guidance_weight", P::constant(0.05)},
                                  {"resampling_weight", P::constant(0.1)}}, 10.0),
  };
}

// stepParameters: each spec's resolved values at t, rounded to float32 (absent ones NaN)
struct StepParams {
  std::vector<double> guidance, resampling;
  std::map<std::string, std::vector<double>> buffers;
};
inline StepParams stepParameters(double t, const std::vector<Spec>& specs) {
  StepParams out;
  for (auto& s : specs) {
    out.guidance.push_back(fr(s.has("guidance_weight") ? s.params.at("guidance_weight").at(t) : 0.0));
    out.resampling.push_back(fr(s.has("resampling_weight") ? s.params.at("resampling_weight").at(t) : 1.0));
  }
  for (auto& k : paramNames())
    for (auto& s : specs) out.buffers[k].push_back(s.has(k) ? fr(s.params.at(k).at(t)) : std::numeric_limits<double>::quiet_NaN());
  return out;
}
inline std::vector<int> guidanceInterval(double t, const std::vector<Spec>& specs) {
  std::vector<int> out;
  for (auto& s : specs) out.push_back((int)std::trunc(s.has("guidance_interval") ? s.params.at("guidance_interval").at(t) : 1));
  return out;
}

// ---- the tables -----------------------------------------------------------------------------------------------------
struct Tables {
  int nPad = 0;
  std::vector<int> pad, chain, chainSizes;
  std::vector<double> radius;                                  // (float32 values)
  std::vector<int> symA, symB;                                 // symmetric_chain_index (all), then the filtered pairs
  std::vector<int> pairA, pairB;
  std::vector<int> interMask, polyMask;
  std::unordered_set<long long> interKeys, polyKeys;
  std::vector<std::vector<int>> interEx, polyEx;               // the same keys as each atom's sorted partners (exclusions())
  std::vector<int> clashI, clashJ, bondI, bondJ, angleI, angleJ, angleC;
  std::vector<int> rbI, rbJ, rbBond, rbAngle;
  std::vector<double> rbUpper, rbLower;                        // (float32 values)
  std::vector<std::array<int, 4>> chiral, stereo, planarImp;
  std::vector<int> chiralOri, stereoOri;
  std::vector<std::array<int, 6>> planar;
  std::vector<double> bondRef, angleRef;                       // (lo, hi, sigma) triplets, doubles
  std::map<int, std::vector<double>> bondLower, bondUpper, angleLower, angleUpper;   // per spec index (float32 values)
};

// compute_geometry_bounds (steering-tables.js geometryBounds): in float32 as the vendor's torch arithmetic
inline std::pair<double, double> geometryBounds(const double* ref, double tolerance, bool angle) {
  double lo = ref[0], hi = ref[1], sigma = ref[2];
  if (angle) { lo = (lo * M_PI) / 180; hi = (hi * M_PI) / 180; sigma = (sigma * M_PI) / 180; }
  const double width = fr(fr(sigma) * fr(tolerance));
  const double lower = fr(fr(lo) - width), upper = fr(fr(hi) + width);
  return {std::max(0.0, lower), std::min(angle ? M_PI : INF, upper)};
}

// the bonded-pair keys a * nPad + b as each a's sorted b's, so the overlap loop walks them beside its sorted pairs
inline void exclusionsOf(const std::unordered_set<long long>& keys, int nPad, std::vector<std::vector<int>>& out) {
  out.assign(nPad, {});
  for (long long k : keys) out[k / nPad].push_back((int)(k % nPad));
  for (auto& v : out) std::sort(v.begin(), v.end());
}
inline void exclusions(Tables& t) {
  exclusionsOf(t.interKeys, t.nPad, t.interEx);
  exclusionsOf(t.polyKeys, t.nPad, t.polyEx);
}

// hostTables: the planar impropers, the chain-pair filter, each Bond/Angle spec's bounds at its own sigma_tolerance
inline void hostTables(Tables& t, const std::vector<Spec>& specs) {
  exclusions(t);
  static const int improper[4][2] = {{1, 4}, {2, 5}, {3, 0}, {0, 3}};
  t.planarImp.clear();
  for (size_t k = 0; k < t.planar.size(); ++k)
    for (int m = 0; m < 2; ++m)
      t.planarImp.push_back({t.planar[k][improper[0][m]], t.planar[k][improper[1][m]], t.planar[k][improper[2][m]],
                             t.planar[k][improper[3][m]]});
  t.pairA.clear(); t.pairB.clear();
  for (size_t k = 0; k < t.symA.size(); ++k)
    if (t.chainSizes[t.symA[k]] > 1 && t.chainSizes[t.symB[k]] > 1) { t.pairA.push_back(t.symA[k]); t.pairB.push_back(t.symB[k]); }
  for (size_t i = 0; i < specs.size(); ++i) {
    if (specs[i].kind == "BondLengthPotential") {
      auto& lo = t.bondLower[(int)i]; auto& hi = t.bondUpper[(int)i]; lo.clear(); hi.clear();
      for (size_t k = 0; k * 3 < t.bondRef.size(); ++k) {
        auto b = geometryBounds(&t.bondRef[k * 3], specs[i].sigmaTolerance, false);
        lo.push_back(fr(b.first)); hi.push_back(fr(b.second));
      }
    } else if (specs[i].kind == "BondAnglePotential") {
      auto& lo = t.angleLower[(int)i]; auto& hi = t.angleUpper[(int)i]; lo.clear(); hi.clear();
      for (size_t k = 0; k * 3 < t.angleRef.size(); ++k) {
        auto b = geometryBounds(&t.angleRef[k * 3], specs[i].sigmaTolerance, true);
        lo.push_back(fr(b.first)); hi.push_back(fr(b.second));
      }
    }
  }
}

// ---- geometry with gradients (potentials.js, the same operations in the same order) ---------------------------------
using V = std::array<double, 3>;
inline V sub(const double* c, int i, int j) { return {c[i * 3] - c[j * 3], c[i * 3 + 1] - c[j * 3 + 1], c[i * 3 + 2] - c[j * 3 + 2]}; }
inline double dot(const V& a, const V& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline V cross(const V& a, const V& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
inline V scale(const V& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline void addTo(double* g, int i, const V& v, double s = 1) {
  if (!g) return;
  g[i * 3] += v[0] * s; g[i * 3 + 1] += v[1] * s; g[i * 3 + 2] += v[2] * s;
}
inline std::pair<double, double> flatBottom(double v, double lo, double hi) {
  return {std::max(lo - v, 0.0) + std::max(v - hi, 0.0), v < lo ? -1.0 : v > hi ? 1.0 : 0.0};
}
inline double jsSign(double v) { return v > 0 ? 1.0 : v < 0 ? -1.0 : v; }

inline double distance(const double* c, int i, int j, double* grad, double g) {
  V r = sub(c, i, j);
  double s = dot(r, r);
  double d = s > 0 ? std::sqrt(s) : 0;
  if (grad && g != 0 && d > 0) { addTo(grad, i, r, g / d); addTo(grad, j, r, -g / d); }
  return d;
}

template <class G>
inline double dihedral(const double* c, int i, int j, int k, int l, double* grad, G gFn) {
  V rij = sub(c, i, j), rkj = sub(c, k, j), rkl = sub(c, k, l);
  V n1 = cross(rij, rkj), n2 = cross(rkj, rkl), m = cross(n1, n2);
  double L = std::sqrt(dot(rkj, rkj));
  double sin = dot(rkj, m) / L, cos = dot(n1, n2);
  double theta = lf::jsmath::atan2(sin, cos);
  double g = gFn(theta);
  if (grad && g != 0) {
    double denom = sin * sin + cos * cos;
    double gs = (g * cos) / denom, gc = (-g * sin) / denom;
    V gn1 = scale(n2, gc), gn2 = scale(n1, gc);
    V grkj = {m[0] * gs / L, m[1] * gs / L, m[2] * gs / L};
    double proj = (gs * dot(rkj, m)) / (L * L * L);
    grkj = {grkj[0] - proj * rkj[0], grkj[1] - proj * rkj[1], grkj[2] - proj * rkj[2]};
    V gm = scale(rkj, gs / L);
    V a = cross(n2, gm), b = cross(gm, n1);
    gn1 = {gn1[0] + a[0], gn1[1] + a[1], gn1[2] + a[2]};
    gn2 = {gn2[0] + b[0], gn2[1] + b[1], gn2[2] + b[2]};
    V grij = cross(rkj, gn1);
    V t1 = cross(gn1, rij), t2 = cross(rkl, gn2);
    grkj = {grkj[0] + t1[0] + t2[0], grkj[1] + t1[1] + t2[1], grkj[2] + t1[2] + t2[2]};
    V grkl = cross(gn2, rkj);
    addTo(grad, i, grij); addTo(grad, j, grij, -1);
    addTo(grad, k, grkj); addTo(grad, j, grkj, -1);
    addTo(grad, k, grkl); addTo(grad, l, grkl, -1);
  }
  return theta;
}

template <class G>
inline double angle(const double* c, int i, int j, int centre, double* grad, G gFn, double eps = 1e-6) {
  V a = sub(c, i, centre), b = sub(c, j, centre);
  double la = std::sqrt(dot(a, a)), lb = std::sqrt(dot(b, b));
  double na = std::max(la, eps), nb = std::max(lb, eps);
  V ua = scale(a, 1 / na), ub = scale(b, 1 / nb);
  double raw = dot(ua, ub);
  double clipped = std::min(std::max(raw, -1 + eps), 1 - eps);
  double theta = lf::jsmath::acos(clipped);
  double g = gFn(theta);
  if (grad && g != 0 && raw > -1 + eps && raw < 1 - eps) {
    double gd = -g / std::sqrt(1 - clipped * clipped);
    auto back = [&](const V& u, const V& other, double len, double n) {
      V gu = scale(other, gd);
      if (len > eps) { double p = dot(u, gu); return scale(V{gu[0] - u[0] * p, gu[1] - u[1] * p, gu[2] - u[2] * p}, 1 / n); }
      return scale(gu, 1 / n);
    };
    V ga = back(ua, ub, la, na), gb = back(ub, ua, lb, nb);
    addTo(grad, i, ga); addTo(grad, centre, ga, -1);
    addTo(grad, j, gb); addTo(grad, centre, gb, -1);
  }
  return theta;
}

inline double overlapPair(const double* c, int i, int j, double minimum, double* grad, double g) {
  V r = sub(c, i, j);
  double d = std::sqrt(dot(r, r) + 1e-12);
  double e = minimum - d;
  if (e <= 0) return 0;
  if (grad && g != 0) { addTo(grad, i, r, -g / d); addTo(grad, j, r, g / d); }
  return e;
}

// the VDW overlap's candidate pairs: every (a, b > a) the JavaScript's double loop would score, in its order, but only
// those within `reach` of each other (a cell list) - the pairs further apart score exactly 0 and add nothing, so the
// sum is the JavaScript's to the bit
inline void nearPairs(const double* c, const std::vector<int>& eligible, int nPad, double reach,
                      std::vector<std::pair<int, int>>& out) {
  out.clear();
  static thread_local std::vector<int> atoms, cell, start, order, nb;
  atoms.clear();
  for (int a = 0; a < nPad; ++a) if (eligible[a]) atoms.push_back(a);
  if (atoms.empty() || !(reach > 0) || !std::isfinite(reach)) return;
  double lo[3] = {INF, INF, INF}, hi[3] = {-INF, -INF, -INF};
  for (int a : atoms) for (int x = 0; x < 3; ++x) { lo[x] = std::min(lo[x], c[a * 3 + x]); hi[x] = std::max(hi[x], c[a * 3 + x]); }
  long long dim[3];
  for (int x = 0; x < 3; ++x) dim[x] = (long long)std::floor((hi[x] - lo[x]) / reach) + 1;
  const double reach2 = reach * reach;
  auto close = [&](int a, int b) {
    const double d0 = c[a * 3] - c[b * 3], d1 = c[a * 3 + 1] - c[b * 3 + 1], d2 = c[a * 3 + 2] - c[b * 3 + 2];
    return d0 * d0 + d1 * d1 + d2 * d2 < reach2;            // (further apart: an overlap of exactly 0)
  };
  const long long cells = dim[0] * dim[1] * dim[2];
  if (!std::isfinite(hi[0] - lo[0]) || cells > 8ll * (long long)atoms.size() + 4096) {
    // (spread out - an early sample - the grid would be mostly empty: every pair, in order)
    for (size_t i = 0; i < atoms.size(); ++i)
      for (size_t j = i + 1; j < atoms.size(); ++j) if (close(atoms[i], atoms[j])) out.push_back({atoms[i], atoms[j]});
    return;
  }
  // a counting sort of the atoms by cell, ascending within each (the order the double loop meets them)
  cell.resize(atoms.size());
  start.assign(cells + 1, 0);
  for (size_t i = 0; i < atoms.size(); ++i) {
    const int a = atoms[i];
    long long k[3];
    for (int x = 0; x < 3; ++x) k[x] = std::min<long long>(dim[x] - 1, (long long)std::floor((c[a * 3 + x] - lo[x]) / reach));
    cell[i] = (int)((k[0] * dim[1] + k[1]) * dim[2] + k[2]);
    start[cell[i] + 1] += 1;
  }
  for (long long k = 0; k < cells; ++k) start[k + 1] += start[k];
  order.resize(atoms.size());
  {
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (size_t i = 0; i < atoms.size(); ++i) order[fill[cell[i]]++] = atoms[i];
  }
  for (size_t i = 0; i < atoms.size(); ++i) {
    const int a = atoms[i];
    nb.clear();
    const long long ck = cell[i] % dim[2], cj = (cell[i] / dim[2]) % dim[1], ci = cell[i] / (dim[1] * dim[2]);
    for (long long x = std::max(0ll, ci - 1); x <= std::min(dim[0] - 1, ci + 1); ++x)
      for (long long y = std::max(0ll, cj - 1); y <= std::min(dim[1] - 1, cj + 1); ++y)
        for (long long z = std::max(0ll, ck - 1); z <= std::min(dim[2] - 1, ck + 1); ++z) {
          const long long k = (x * dim[1] + y) * dim[2] + z;
          for (int q = start[k]; q < start[k + 1]; ++q) if (order[q] > a && close(a, order[q])) nb.push_back(order[q]);
        }
    std::sort(nb.begin(), nb.end());
    for (int b : nb) out.push_back({a, b});
  }
}

// one spec's energy of one particle at its resolved parameters, its gradient times `weight` added into grad if given
inline double energy(const std::string& kind, int i, const double* c, const Tables& t, const std::map<std::string, double>& p,
                     double* grad = nullptr, double weight = 1) {
  const int nPad = t.nPad;
  const auto& radius = t.radius;
  const auto& chain = t.chain;
  double total = 0;
  auto P = [&](const char* k) { return p.at(k); };
  if (kind == "SymmetricChainCOMPotential") {
    if (t.pairA.empty()) return 0;
    const int nChain = (int)t.chainSizes.size();
    std::vector<double> com(nChain * 3, 0.0), count(nChain, 0.0), gcom(nChain * 3, 0.0);
    for (int a = 0; a < nPad; ++a) {
      if (!t.pad[a]) continue;
      count[chain[a]] += 1;
      for (int x = 0; x < 3; ++x) com[chain[a] * 3 + x] += c[a * 3 + x];
    }
    for (int k = 0; k < nChain; ++k) for (int x = 0; x < 3; ++x) com[k * 3 + x] /= std::max(count[k], 1.0);
    const double buffer = P("buffer");
    for (size_t k = 0; k < t.pairA.size(); ++k) {
      double d = distance(com.data(), t.pairA[k], t.pairB[k], gcom.data(), 0);
      if (buffer - d > 0) {
        total += buffer - d;
        if (grad && d > 0) distance(com.data(), t.pairA[k], t.pairB[k], gcom.data(), -weight);
      }
    }
    if (grad)
      for (int a = 0; a < nPad; ++a) {
        if (!t.pad[a]) continue;
        int k = chain[a];
        for (int x = 0; x < 3; ++x) grad[a * 3 + x] += gcom[k * 3 + x] / std::max(count[k], 1.0);
      }
    return total;
  }
  if (kind == "InterChainVDWOverlapPotential" || kind == "IntraChainVDWOverlapPotential") {
    const bool same = kind == "IntraChainVDWOverlapPotential";
    const auto& eligible = same ? t.polyMask : t.interMask;
    const auto& ex = same ? t.polyEx : t.interEx;
    const double s = 1.0 - P("buffer");
    double maxR = 0;
    for (int a = 0; a < nPad; ++a) if (eligible[a]) maxR = std::max(maxR, radius[a]);
    static thread_local std::vector<std::pair<int, int>> pairs;
    nearPairs(c, eligible, nPad, (maxR + maxR) * s + 1e-6, pairs);
    // (pairs come a ascending, then b ascending: each a's exclusions walked once beside them)
    int lastA = -1;
    size_t at = 0;
    for (auto& [a, b] : pairs) {
      if (a != lastA) { lastA = a; at = 0; }
      if ((chain[a] == chain[b]) != same) continue;
      const auto& mine = ex[a];
      while (at < mine.size() && mine[at] < b) ++at;
      if (at < mine.size() && mine[at] == b) continue;
      total += overlapPair(c, a, b, (radius[a] + radius[b]) * s, grad, weight);
    }
    return total;
  }
  if (kind == "PoseBustersPotential") {
    if (t.rbI.empty()) return 0;
    const double bb = fr(P("bond_buffer")), ab = fr(P("angle_buffer")), cb = fr(P("clash_buffer"));
    const double mb = std::min(bb, ab);
    for (size_t k = 0; k < t.rbI.size(); ++k) {
      const bool bond = t.rbBond[k] != 0, ang = t.rbAngle[k] != 0;
      const double lo0 = fr(t.rbLower[k]), hi0 = fr(t.rbUpper[k]);
      const double buffer = bond && !ang ? bb : !bond && ang ? ab : bond && ang ? mb : cb;
      const double lo = fr(lo0 * fr(1 - buffer));
      const double hi = bond || ang ? fr(hi0 * fr(1 + buffer)) : INF;
      double d = distance(c, t.rbI[k], t.rbJ[k], nullptr, 0);
      auto [e, de] = flatBottom(d, lo, hi);
      total += e;
      if (grad && de != 0) distance(c, t.rbI[k], t.rbJ[k], grad, de * weight);
    }
    return total;
  }
  if (kind == "ChiralAtomPotential" || kind == "StereoBondPotential" || kind == "PlanarBondPotential") {
    const auto& idx = kind == "ChiralAtomPotential" ? t.chiral : kind == "StereoBondPotential" ? t.stereo : t.planarImp;
    if (idx.empty()) return 0;
    const double b = fr(P("buffer"));
    for (size_t k = 0; k < idx.size(); ++k) {
      double lo, hi;
      bool absolute = false;
      if (kind == "ChiralAtomPotential") {
        const bool o = t.chiralOri[k] != 0;
        lo = o ? b : -INF; hi = o ? INF : -b;
      } else if (kind == "StereoBondPotential") {
        const bool o = t.stereoOri[k] != 0;
        lo = o ? fr(fr(M_PI) - b) : -INF; hi = o ? INF : b;
        absolute = true;
      } else {
        lo = -INF; hi = b; absolute = true;
      }
      double here = 0;
      dihedral(c, idx[k][0], idx[k][1], idx[k][2], idx[k][3], grad, [&](double theta) {
        double v = absolute ? std::fabs(theta) : theta;
        auto [e, de] = flatBottom(v, lo, hi);
        here = e;
        return de * weight * (absolute ? jsSign(theta) : 1);
      });
      total += here;
    }
    return total;
  }
  if (kind == "PolymerLocalClashPotential") {
    if (t.clashI.empty()) return 0;
    const double minimum = P("minimum_distance");
    for (size_t k = 0; k < t.clashI.size(); ++k) total += overlapPair(c, t.clashI[k], t.clashJ[k], minimum, grad, weight);
    return total;
  }
  if (kind == "BondLengthPotential") {
    const auto& lo = t.bondLower.at(i);
    const auto& hi = t.bondUpper.at(i);
    for (size_t k = 0; k < t.bondI.size(); ++k) {
      double d = distance(c, t.bondI[k], t.bondJ[k], nullptr, 0);
      auto [e, de] = flatBottom(d, lo[k], hi[k]);
      total += e;
      if (grad && de != 0) distance(c, t.bondI[k], t.bondJ[k], grad, de * weight);
    }
    return total;
  }
  if (kind == "BondAnglePotential") {
    const auto& lo = t.angleLower.at(i);
    const auto& hi = t.angleUpper.at(i);
    for (size_t k = 0; k < t.angleI.size(); ++k) {
      double e = 0;
      angle(c, t.angleI[k], t.angleJ[k], t.angleC[k], grad, [&](double theta) {
        auto [v, de] = flatBottom(theta, lo[k], hi[k]);
        e = v;
        return de * weight;
      });
      total += e;
    }
    return total;
  }
  throw std::runtime_error(kind);
}

// every spec's energy of one particle; `weights` (per spec) scale the gradient added into grad
// (`gradientOnly`: a spec whose weight is 0 is skipped outright - its energy is computed by the JavaScript and read by
// nothing, and it adds no gradient - so the guidance loop's result is the JavaScript's to the bit, at a fraction of it)
inline std::vector<double> energies(const double* c, const Tables& t, const StepParams& p, const std::vector<Spec>& specs,
                                    double* grad = nullptr, const std::vector<double>* weights = nullptr,
                                    bool gradientOnly = false) {
  std::vector<double> out;
  for (size_t i = 0; i < specs.size(); ++i) {
    const double w = weights ? (*weights)[i] : 1;
    if (gradientOnly && w == 0) { out.push_back(0); continue; }
    std::map<std::string, double> at;
    for (auto& k : paramNames()) at[k] = p.buffers.at(k)[i];
    out.push_back(energy(specs[i].kind, (int)i, c, t, at, grad && w != 0 ? grad : nullptr, w));
  }
  return out;
}

// ---- the weighted rigid alignment (cpu/af3/diffusion/rigid-align.js weightedRigidAlign), its float32 result ---------
inline void symmetricEigen(const double* m, double values[3], double v[9]) {
  double a[9];
  std::memcpy(a, m, sizeof a);
  const double I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::memcpy(v, I, sizeof I);
  static const int PQ[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  for (int sweep = 0; sweep < 24; ++sweep) {
    double off = 0;
    for (auto& pq : PQ) off += a[pq[0] * 3 + pq[1]] * a[pq[0] * 3 + pq[1]];
    if (off < 1e-30) break;
    for (auto& pq : PQ) {
      const int p = pq[0], q = pq[1];
      const double apq = a[p * 3 + q];
      if (std::fabs(apq) < 1e-300) continue;
      const double theta = (a[q * 3 + q] - a[p * 3 + p]) / (2 * apq);
      const double t = jsSign(theta != 0 && !std::isnan(theta) ? theta : 1) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
      const double c = 1 / std::sqrt(t * t + 1), s = t * c;
      for (int k = 0; k < 3; ++k) { double akp = a[k * 3 + p], akq = a[k * 3 + q]; a[k * 3 + p] = c * akp - s * akq; a[k * 3 + q] = s * akp + c * akq; }
      for (int k = 0; k < 3; ++k) { double apk = a[p * 3 + k], aqk = a[q * 3 + k]; a[p * 3 + k] = c * apk - s * aqk; a[q * 3 + k] = s * apk + c * aqk; }
      for (int k = 0; k < 3; ++k) { double vkp = v[k * 3 + p], vkq = v[k * 3 + q]; v[k * 3 + p] = c * vkp - s * vkq; v[k * 3 + q] = s * vkp + c * vkq; }
    }
  }
  values[0] = a[0]; values[1] = a[4]; values[2] = a[8];
}

inline std::vector<double> rigidAlign(const std::vector<double>& x, const std::vector<double>& target, const std::vector<double>& weights) {
  const int atoms = (int)weights.size();
  double total = 0, centre[3] = {0, 0, 0}, centreTarget[3] = {0, 0, 0};
  for (int atom = 0; atom < atoms; ++atom) {
    const double w = weights[atom];
    total += w;
    for (int axis = 0; axis < 3; ++axis) { centre[axis] += w * x[atom * 3 + axis]; centreTarget[axis] += w * target[atom * 3 + axis]; }
  }
  const double by = 1 / std::max(total, 1e-8);
  for (int axis = 0; axis < 3; ++axis) { centre[axis] *= by; centreTarget[axis] *= by; }
  double h[9] = {0};
  for (int atom = 0; atom < atoms; ++atom) {
    const double w = weights[atom];
    if (w == 0) continue;
    for (int i = 0; i < 3; ++i) {
      const double a = target[atom * 3 + i] - centreTarget[i];
      for (int j = 0; j < 3; ++j) h[i * 3 + j] += w * a * (x[atom * 3 + j] - centre[j]);
    }
  }
  double hth[9];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    double sum = 0;
    for (int k = 0; k < 3; ++k) sum += h[k * 3 + i] * h[k * 3 + j];
    hth[i * 3 + j] = sum;
  }
  double values[3], vectors[9];
  symmetricEigen(hth, values, vectors);
  int order[3] = {0, 1, 2};
  std::stable_sort(order, order + 3, [&](int a, int b) { return values[b] - values[a] < 0; });
  double vv[9] = {0}, uu[9] = {0}, sigmas[3];
  for (int column = 0; column < 3; ++column) {
    const int from = order[column];
    for (int row = 0; row < 3; ++row) vv[row * 3 + column] = vectors[row * 3 + from];
    sigmas[column] = std::sqrt(std::max(values[from], 0.0));
  }
  const double floor = 1e-12 * std::fabs(values[order[0]] != 0 ? values[order[0]] : 1);
  std::vector<int> good;
  for (int column = 0; column < 3; ++column) {
    if (values[order[column]] <= floor) { sigmas[column] = 0; continue; }
    for (int row = 0; row < 3; ++row) {
      double sum = 0;
      for (int k = 0; k < 3; ++k) sum += h[row * 3 + k] * vv[k * 3 + column];
      uu[row * 3 + column] = sum / sigmas[column];
    }
    const double length = lf::jsmath::hypot({uu[column], uu[3 + column], uu[6 + column]});
    if (std::fabs(length - 1) > 1e-6) { uu[column] = uu[3 + column] = uu[6 + column] = 0; sigmas[column] = 0; }
    else good.push_back(column);
  }
  for (int column = 0; column < 3; ++column) {
    if (std::find(good.begin(), good.end(), column) != good.end()) continue;
    double best[3] = {0, 0, 0}, bestLength = -1;
    const double seeds[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (auto& seed : seeds) {
      double w[3] = {seed[0], seed[1], seed[2]};
      for (int k : good) {
        const double d = w[0] * uu[k] + w[1] * uu[3 + k] + w[2] * uu[6 + k];
        w[0] -= d * uu[k]; w[1] -= d * uu[3 + k]; w[2] -= d * uu[6 + k];
      }
      const double length = lf::jsmath::hypot({w[0], w[1], w[2]});
      if (length > bestLength) { bestLength = length; std::memcpy(best, w, sizeof w); }
    }
    for (int row = 0; row < 3; ++row) uu[row * 3 + column] = best[row] / bestLength;
    good.push_back(column);
  }
  double uvt[9];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    double sum = 0;
    for (int k = 0; k < 3; ++k) sum += uu[i * 3 + k] * vv[j * 3 + k];
    uvt[i * 3 + j] = sum;
  }
  const double determinant = uvt[0] * (uvt[4] * uvt[8] - uvt[5] * uvt[7]) - uvt[1] * (uvt[3] * uvt[8] - uvt[5] * uvt[6])
                             + uvt[2] * (uvt[3] * uvt[7] - uvt[4] * uvt[6]);
  const double sign = determinant < 0 ? -1 : 1;
  double r[9];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    double sum = 0;
    for (int k = 0; k < 3; ++k) sum += uu[i * 3 + k] * (k == 2 ? sign : 1) * vv[j * 3 + k];
    r[i * 3 + j] = sum;
  }
  std::vector<double> out(x.size(), 0.0);         // (a Float32Array: its untouched tail is 0)
  for (int atom = 0; atom < atoms; ++atom)
    for (int i = 0; i < 3; ++i) {
      double sum = 0;
      for (int j = 0; j < 3; ++j) sum += (x[atom * 3 + j] - centre[j]) * r[i * 3 + j];
      out[atom * 3 + i] = fr(sum + centreTarget[i]);
    }
  return out;
}

// ---- the steered sampler (sampler.js steeredSample) -----------------------------------------------------------------
using Particles = std::vector<std::vector<double>>;
struct Draws {
  std::function<Particles()> x0;
  // step k: each particle's rotation (row-major 3x3, applied as x R), translation and N(0, 1) noise
  std::function<void(int, std::vector<std::array<double, 9>>&, std::vector<std::array<double, 3>>&, Particles&)> step;
  std::function<std::vector<double>(int, int)> uniform;     // (resampling event, groups) -> one uniform a group
};
struct SteerSettings {
  int steps = 200, gdSteps = 20;
  double stepScale = 1.0, noiseScale = 1.0, gamma0 = 0.8, gammaMin = 1.0, fkLambda = 4.0;
  int resamplingInterval = 3;          // edm_fk; 0 = none (edm_gd)
  int groupSize = 0;                   // 0: every particle one group
};
struct TraceRow { int k; bool resample; std::vector<double> w; std::vector<int> idx; std::vector<double> energy; };

// edm_sample_schedule: num_steps Karras levels (x sigma_data), then 0, in float32 as the vendor's
inline std::vector<double> schedule(int numSteps) {
  if (numSteps < 2) throw std::runtime_error("BioEmu2 samples with at least 2 steps");
  const double inv = 1.0 / RHO;
  const double hi = fr(std::pow(fr(SIGMA_MAX), inv)), lo = fr(std::pow(fr(SIGMA_MIN), inv));
  std::vector<double> out(numSteps + 1, 0.0);
  for (int k = 0; k < numSteps; ++k) {
    const double s = fr(std::pow(fr(hi + fr(fr((double)k / fr(numSteps - 1)) * fr(lo - hi))), RHO));
    out[k] = fr(s * fr(SIGMA_DATA));
  }
  return out;
}

inline std::vector<double> centreAndRotate(const std::vector<double>& x, const std::vector<double>& mask,
                                           const std::array<double, 9>& rot, const std::array<double, 3>& trans) {
  const size_t n = mask.size();
  double w = 0, centre[3] = {0, 0, 0};
  for (size_t a = 0; a < n; ++a) if (mask[a] != 0) { w += 1; for (int d = 0; d < 3; ++d) centre[d] += x[a * 3 + d]; }
  for (int d = 0; d < 3; ++d) centre[d] /= w;
  std::vector<double> out(x.size(), 0.0);
  for (size_t a = 0; a < n; ++a) {
    const double v[3] = {x[a * 3] - centre[0], x[a * 3 + 1] - centre[1], x[a * 3 + 2] - centre[2]};
    for (int s = 0; s < 3; ++s) out[a * 3 + s] = v[0] * rot[s] + v[1] * rot[3 + s] + v[2] * rot[6 + s] + trans[s];
  }
  return out;
}
inline std::vector<double> rotateOnly(const std::vector<double>& x, const std::array<double, 9>& rot) {
  std::vector<double> out(x.size());
  for (size_t a = 0; a < x.size() / 3; ++a)
    for (int s = 0; s < 3; ++s) out[a * 3 + s] = x[a * 3] * rot[s] + x[a * 3 + 1] * rot[3 + s] + x[a * 3 + 2] * rot[6 + s];
  return out;
}
inline std::vector<double> groupedSoftmax(const std::vector<double>& logw, int group) {
  std::vector<double> out(logw.size());
  for (size_t g = 0; g < logw.size(); g += group) {
    double max = -INF;
    for (size_t i = g; i < g + group; ++i) max = std::max(max, logw[i]);
    double sum = 0;
    for (size_t i = g; i < g + group; ++i) { out[i] = std::exp(logw[i] - max); sum += out[i]; }
    for (size_t i = g; i < g + group; ++i) out[i] /= sum;
  }
  return out;
}
inline std::vector<int> systematicResample(const std::vector<double>& w, int group, const std::vector<double>& u) {
  std::vector<int> out(w.size());
  for (size_t g = 0; g < w.size() / group; ++g) {
    std::vector<double> cum;
    double s = 0;
    for (int i = 0; i < group; ++i) { s += w[g * group + i]; cum.push_back(s); }
    for (int i = 0; i < group; ++i) {
      const double pos = (i + u[g]) / group;
      int idx = 0;
      while (idx < group && cum[idx] < pos) ++idx;              // searchsorted, side='left'
      out[g * group + i] = std::min(idx, group - 1) + (int)g * group;
    }
  }
  return out;
}

using Denoise = std::function<Particles(const Particles&, double, const Particles*)>;
// where a steered run's host time went (ms): the FK energies, the guidance descent, the realignment
struct SteerTimes { double energies = 0, guidance = 0, align = 0; };
inline SteerTimes STEER_TIMES;
inline double since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// fn(p) for every particle, on up to hardware_concurrency threads
inline void forParticles(int P, const std::function<void(int)>& fn) {
  const int threads = std::max(1, std::min<int>(P, (int)std::thread::hardware_concurrency()));
  if (threads == 1) { for (int p = 0; p < P; ++p) fn(p); return; }
  std::vector<std::thread> pool;
  std::atomic<int> next{0};
  for (int k = 0; k < threads; ++k)
    pool.emplace_back([&] { for (int p; (p = next++) < P;) fn(p); });
  for (auto& t : pool) t.join();
}

// edm_steered_sampler: presets edm_gd (resamplingInterval 0) and edm_fk (3). The particles after the last step.
inline Particles steeredSample(const Denoise& denoise, const Tables& tables, const std::vector<double>& mask, int P,
                               const Draws& draws, const SteerSettings& o, const std::vector<Spec>& specs,
                               Particles* companion = nullptr, std::vector<TraceRow>* trace = nullptr,
                               const std::function<void(int, int)>& progress = nullptr) {
  const int group = o.groupSize > 0 ? o.groupSize : P;
  if (P % group) throw std::runtime_error("the particles must divide into resampling groups");
  const std::vector<double> sig = schedule(o.steps);
  std::vector<double> gam(sig.size());
  for (size_t k = 0; k < sig.size(); ++k) gam[k] = sig[k] > o.gammaMin ? fr(o.gamma0) : 0;
  const int last = o.steps - 1;
  const size_t n = mask.size();
  Particles x = draws.x0();
  for (auto& v : x) for (auto& a : v) a = fr(sig[0] * a);
  Particles sgu(P, std::vector<double>(n * 3, 0.0));
  std::vector<double> ePrev(P, 0.0);
  bool hasPrev = false;
  Particles comp = companion ? *companion : Particles();
  const bool withComp = companion != nullptr;
  int uniformIndex = -1;
  std::vector<std::array<double, 9>> rot;
  std::vector<std::array<double, 3>> trans;
  Particles eps;
  for (int k = 0; k < o.steps; ++k) {
    const double sTm = sig[k], sT = sig[k + 1], g = gam[k + 1];
    const double tHat = fr(sTm * (1 + g));
    const double noiseVar = o.noiseScale * o.noiseScale * std::max(tHat * tHat - sTm * sTm, 0.0);
    const double noiseStd = fr(std::sqrt(noiseVar));
    const bool resample = o.resamplingInterval > 0 && (k % o.resamplingInterval == 0 || k == last);
    if (resample) uniformIndex += 1;
    const double tNorm = 1.0 - (double)(k + 1) / o.steps;
    const StepParams prm = stepParameters(tNorm, specs);
    const std::vector<int> interval = guidanceInterval(tNorm, specs);
    draws.step(k, rot, trans, eps);
    for (int p = 0; p < P; ++p) x[p] = centreAndRotate(x[p], mask, rot[p], trans[p]);
    if (withComp) for (int p = 0; p < P; ++p) comp[p] = centreAndRotate(comp[p], mask, rot[p], trans[p]);
    for (int p = 0; p < P; ++p) sgu[p] = rotateOnly(sgu[p], rot[p]);
    Particles noise(P), noisy(P);
    for (int p = 0; p < P; ++p) {
      noise[p].resize(eps[p].size());
      for (size_t i = 0; i < eps[p].size(); ++i) noise[p][i] = noiseStd * eps[p][i];
      noisy[p].resize(x[p].size());
      for (size_t i = 0; i < x[p].size(); ++i) noisy[p][i] = x[p][i] + noise[p][i];
    }
    Particles x0 = denoise(noisy, tHat, withComp ? &comp : nullptr);
    std::vector<double> w(P, 1.0 / group);
    std::vector<double> energy = ePrev;
    auto phase = std::chrono::steady_clock::now();
    if (resample) {
      energy.assign(P, 0.0);
      std::vector<double> un(P);
      std::vector<std::vector<double>> each(P);
      forParticles(P, [&](int p) { each[p] = energies(x0[p].data(), tables, prm, specs); });
      for (int p = 0; p < P; ++p) {
        const auto& e = each[p];
        double sum = 0;
        for (size_t i = 0; i < e.size(); ++i) sum = sum + prm.resampling[i] * e[i];
        energy[p] = sum;
        const double logG = hasPrev ? ePrev[p] - energy[p] : -energy[p];
        double ll = 0;
        if (o.gdSteps > 0 && noiseVar > 0) {
          for (size_t i = 0; i < noise[p].size(); ++i) {
            const double a = noise[p][i], b = noise[p][i] + sgu[p][i];
            ll += a * a - b * b;
          }
          ll /= 2 * std::max(noiseVar, 1e-30);
        }
        const double value = ll + o.fkLambda * logG;
        un[p] = std::isfinite(value) ? value : -1e4;
      }
      w = groupedSoftmax(un, group);
    }
    STEER_TIMES.energies += since(phase);
    phase = std::chrono::steady_clock::now();
    if (k < last && o.gdSteps > 0) {
      Particles upd(P, std::vector<double>(n * 3, 0.0));
      std::vector<double> coef(specs.size()), at(n * 3), grad(n * 3);
      for (int gs = 0; gs < o.gdSteps; ++gs) {
        bool any = false;
        for (size_t i = 0; i < specs.size(); ++i) {
          coef[i] = prm.guidance[i] != 0 && gs % interval[i] == 0 ? prm.guidance[i] : 0;
          any = any || coef[i] != 0;
        }
        if (!any) continue;
        // (the particles are independent here: a thread each, every one's arithmetic unchanged)
        forParticles(P, [&](int p) {
          std::vector<double> at(n * 3), grad(n * 3, 0.0);
          for (size_t i = 0; i < at.size(); ++i) at[i] = x0[p][i] + upd[p][i];
          energies(at.data(), tables, prm, specs, grad.data(), &coef, true);
          for (size_t i = 0; i < grad.size(); ++i) upd[p][i] -= grad[i];
        });
      }
      const double factor = (-o.stepScale * (sT - tHat)) / std::max(tHat, 1e-12);
      for (int p = 0; p < P; ++p) {
        for (size_t i = 0; i < x0[p].size(); ++i) x0[p][i] = x0[p][i] + upd[p][i];
        for (size_t i = 0; i < upd[p].size(); ++i) sgu[p][i] = upd[p][i] * factor;
      }
    }
    STEER_TIMES.guidance += since(phase);
    phase = std::chrono::steady_clock::now();
    std::vector<int> idx(P);
    for (int p = 0; p < P; ++p) idx[p] = p;
    if (resample) idx = systematicResample(w, group, draws.uniform(uniformIndex, P / group));
    if (trace) trace->push_back({k, resample, w, idx, energy});
    auto pick = [&](Particles& v) { Particles out(P); for (int p = 0; p < P; ++p) out[p] = v[idx[p]]; v.swap(out); };
    pick(x); pick(noisy); pick(x0); pick(sgu);
    { std::vector<double> e(P); for (int p = 0; p < P; ++p) e[p] = energy[idx[p]]; energy.swap(e); }
    if (withComp) pick(comp);
    hasPrev = hasPrev || resample;
    ePrev = energy;
    const double step = (o.stepScale * (sT - tHat)) / std::max(tHat, 1e-12);
    for (int p = 0; p < P; ++p) {
      std::vector<double> aligned = rigidAlign(noisy[p], x0[p], mask);
      for (size_t i = 0; i < aligned.size(); ++i) x[p][i] = aligned[i] + step * (aligned[i] - x0[p][i]);
    }
    STEER_TIMES.align += since(phase);
    if (progress) progress(k + 1, o.steps);
  }
  if (companion) *companion = comp;
  return x;
}

}  // namespace be2steer
