// Pocket and contact constraints: the native twin of shared/af3/featurise/constraints.js and of the pocket and
// contact rows in web/entities.js. boltz2's contact conditioning and chai-1's two distance restraints, built on the
// finished batch from each token's chain, residue and atom names (af3-any-model model_features.py _boltz2_constraints,
// _chai1_constraints); and the per-pair term as a table of a few rows and an index into it, which the CUDA and Metal
// trunks add - see the JavaScript for the reasoning, which this follows line for line.
#pragma once
#include <cmath>
#include <cstdlib>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lf {

// ---------------------------------------------------------------- web/entities.js parsePocket / parseContact
struct ConstraintEnd { std::string chain; int residue; bool hasAtom; std::string atom; };
struct PocketRow { std::string binder; std::vector<ConstraintEnd> contacts; double maxDistance; };
struct ContactRow { ConstraintEnd token1, token2; double maxDistance; };
inline constexpr double DEFAULT_CONSTRAINT_DISTANCE = 6;

inline std::string constraintTrim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n\f\v"), b = s.find_last_not_of(" \t\r\n\f\v");
  return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}
inline std::string constraintUpper(std::string s) { for (auto& c : s) c = (char)std::toupper((unsigned char)c); return s; }
inline std::optional<ConstraintEnd> constraintEnd(const std::string& text) {
  static const std::regex end(R"(^\s*([A-Za-z]+)\s*(\d+)\s*(?::\s*([A-Za-z0-9']+)\s*)?$)");
  std::smatch m;
  if (!std::regex_match(text, m, end)) return std::nullopt;
  return ConstraintEnd{constraintUpper(m[1]), std::stoi(m[2]), m[3].matched, m[3].matched ? constraintUpper(m[3]) : ""};
}
// the `@ 6` ending a row: (the text before it, the distance), 6 when absent, nullopt when it is not a positive number
inline std::optional<std::pair<std::string, double>> constraintDistance(const std::string& text) {
  size_t at = text.rfind('@');
  if (at == std::string::npos) return std::make_pair(text, DEFAULT_CONSTRAINT_DISTANCE);
  std::string tail = constraintTrim(text.substr(at + 1));
  char* stop = nullptr;
  double v = tail.empty() ? 0 : std::strtod(tail.c_str(), &stop);
  if (tail.empty() || *stop != '\0' || !std::isfinite(v) || !(v > 0)) return std::nullopt;
  return std::make_pair(text.substr(0, at), v);
}
inline std::optional<PocketRow> parsePocket(const std::string& value) {
  auto parts = constraintDistance(constraintTrim(value));
  if (!parts) return std::nullopt;
  size_t colon = parts->first.find(':');
  if (colon == std::string::npos) return std::nullopt;
  std::string binder = constraintUpper(constraintTrim(parts->first.substr(0, colon)));
  if (binder.empty()) return std::nullopt;
  for (char c : binder) if (c < 'A' || c > 'Z') return std::nullopt;
  PocketRow row{binder, {}, parts->second};
  std::string rest = parts->first.substr(colon + 1);
  size_t s = 0, d;
  std::vector<std::string> pieces;
  while ((d = rest.find(',', s)) != std::string::npos) { pieces.push_back(rest.substr(s, d - s)); s = d + 1; }
  pieces.push_back(rest.substr(s));
  for (auto& piece : pieces) {
    std::string t = constraintTrim(piece);
    if (t.empty()) continue;
    auto e = constraintEnd(t);
    if (!e) return std::nullopt;
    row.contacts.push_back(*e);
  }
  if (row.contacts.empty()) return std::nullopt;
  return row;
}
inline std::optional<ContactRow> parseContact(const std::string& value) {
  auto parts = constraintDistance(constraintTrim(value));
  if (!parts) return std::nullopt;
  std::vector<std::string> sides;
  size_t s = 0, d;
  while ((d = parts->first.find('-', s)) != std::string::npos) { sides.push_back(parts->first.substr(s, d - s)); s = d + 1; }
  sides.push_back(parts->first.substr(s));
  if (sides.size() != 2) return std::nullopt;
  auto a = constraintEnd(sides[0]), b = constraintEnd(sides[1]);
  if (!a || !b) return std::nullopt;
  return ContactRow{*a, *b, parts->second};
}

// ---------------------------------------------------------------- resolved (asym numbers), as expandEntities emits them
struct ResolvedEnd { int asym; int residue; bool hasAtom; std::string atom; };
struct Constraint {
  enum Kind { Pocket, Contact } kind = Pocket;
  int binder = -1; std::vector<ResolvedEnd> contacts;       // a pocket
  ResolvedEnd token1{}, token2{};                           // a contact
  double maxDistance = DEFAULT_CONSTRAINT_DISTANCE;
};

// ---------------------------------------------------------------- shared/af3/featurise/constraints.js
inline const int CC_UNSPECIFIED = 0, CC_UNSELECTED = 1, CC_POCKET_BINDER = 2, CC_BINDER_POCKET = 3, CC_CONTACT = 4;

// what the feature builders read off a finished batch
struct ConstraintTokens {
  int tokens = 0, polymerChains = 0;
  std::vector<int> asym, residue;                 // asym from 0, -1 for a token with no atom
  std::vector<std::set<std::string>> names;
  std::vector<std::string> chains, chainKinds;    // the polymer chains' sequences and kinds
};
inline ConstraintTokens constraintTokens(int tokens, int dense, const std::vector<int>& asymId,
                                         const std::vector<int>& residueIndex, const std::vector<float>& refMask,
                                         const std::vector<int>& refAtomNameChars, const std::vector<std::string>& chains,
                                         const std::vector<std::string>& chainKinds) {
  ConstraintTokens t;
  t.tokens = tokens; t.polymerChains = (int)chains.size(); t.chains = chains; t.chainKinds = chainKinds;
  t.asym.resize(tokens); t.residue.resize(tokens); t.names.resize(tokens);
  for (int i = 0; i < tokens; ++i) {
    for (int slot = 0; slot < dense; ++slot) {
      size_t flat = (size_t)i * dense + slot;
      if (refMask[flat] == 0) continue;
      std::string name;
      for (int c = 0; c < 4; ++c) {
        int v = refAtomNameChars[flat * 4 + c];
        if (v > 0 && v < 64) name += (char)(v + 32);
      }
      t.names[i].insert(constraintTrim(name));
    }
    t.asym[i] = t.names[i].empty() ? -1 : asymId[i] - 1;
    t.residue[i] = residueIndex[i];
  }
  return t;
}
inline std::string describeEnd(const ResolvedEnd& e) {
  return "[" + std::to_string(e.asym) + ", " + std::to_string(e.residue) + (e.hasAtom ? ", " + e.atom : "") + "]";
}
inline std::vector<int> boltzTokens(const ConstraintTokens& t, const ResolvedEnd& end, const std::string& where) {
  bool ligand = end.asym >= t.polymerChains, hasAtom = end.hasAtom && !end.atom.empty();
  if (ligand && !hasAtom) throw std::runtime_error(where + " " + describeEnd(end) + ": a ligand token is named by its atom");
  if (!ligand && hasAtom)
    throw std::runtime_error(where + " " + describeEnd(end) + ": boltz2 conditions polymer residues, not atoms");
  std::vector<int> hit;
  for (int i = 0; i < t.tokens; ++i)
    if (t.asym[i] == end.asym && t.residue[i] == end.residue && (!ligand || t.names[i].count(end.atom))) hit.push_back(i);
  if (hit.empty()) throw std::runtime_error(where + " " + describeEnd(end) + " names no token of the input");
  return hit;
}
inline std::vector<int> binderTokens(const ConstraintTokens& t, int binder, const std::string& where) {
  std::vector<int> out;
  for (int i = 0; i < t.tokens; ++i) if (t.asym[i] == binder) out.push_back(i);
  if (out.empty()) throw std::runtime_error(where + ": binder chain " + std::to_string(binder) + " has no token");
  return out;
}
// boltz2's contact_conditioning (classes) and contact_threshold, n x n
inline void boltz2ConstraintFeatures(const ConstraintTokens& t, const std::vector<Constraint>& cs,
                                     std::vector<int>& classes, std::vector<float>& threshold) {
  int n = t.tokens;
  classes.assign((size_t)n * n, CC_UNSELECTED);
  threshold.assign((size_t)n * n, 0.f);
  auto set = [&](int i, int j, int c, double d) { classes[(size_t)i * n + j] = c; threshold[(size_t)i * n + j] = (float)d; };
  for (size_t k = 0; k < cs.size(); ++k) {
    if (cs[k].kind != Constraint::Pocket) continue;
    std::string where = "constraints[" + std::to_string(k) + "] pocket";
    auto binder = binderTokens(t, cs[k].binder, where);
    for (auto& end : cs[k].contacts)
      for (int p : boltzTokens(t, end, where))
        for (int b : binder) { set(b, p, CC_BINDER_POCKET, cs[k].maxDistance); set(p, b, CC_POCKET_BINDER, cs[k].maxDistance); }
  }
  for (size_t k = 0; k < cs.size(); ++k) {
    if (cs[k].kind != Constraint::Contact) continue;
    int i = boltzTokens(t, cs[k].token1, "constraints[" + std::to_string(k) + "] token1")[0];
    int j = boltzTokens(t, cs[k].token2, "constraints[" + std::to_string(k) + "] token2")[0];
    set(i, j, CC_CONTACT, cs[k].maxDistance); set(j, i, CC_CONTACT, cs[k].maxDistance);
  }
  bool all = true;
  for (int c : classes) if (c != CC_UNSELECTED) { all = false; break; }
  if (all) std::fill(classes.begin(), classes.end(), CC_UNSPECIFIED);
}
// chai-1's TokenDistanceRestraint and TokenPairPocketRestraint, n x n, -1 where none
inline void chai1ConstraintFeatures(const ConstraintTokens& t, const std::vector<Constraint>& cs,
                                    std::vector<float>& contact, std::vector<float>& pocket) {
  int n = t.tokens;
  static const std::string letters = "ARNDCQEGHILKMFPSTWYVX";
  auto residue = [&](const ResolvedEnd& end, const std::string& where) {
    if (end.hasAtom && !end.atom.empty())
      throw std::runtime_error(where + " " + describeEnd(end) + ": chai1 conditions residues, not atoms");
    if (end.asym >= t.polymerChains || (end.asym < (int)t.chainKinds.size() && t.chainKinds[end.asym] != "protein"))
      throw std::runtime_error(where + " " + describeEnd(end) + ": chai1 restrains protein residues only (a ligand only as a binder)");
    const std::string& seq = t.chains[end.asym];
    if (end.residue < 1 || end.residue > (int)seq.size() || letters.find(seq[end.residue - 1]) == std::string::npos)
      throw std::runtime_error(where + " " + describeEnd(end) + ": chai1 restrains standard residues only");
    std::vector<int> hit;
    for (int i = 0; i < n; ++i) if (t.asym[i] == end.asym && t.residue[i] == end.residue) hit.push_back(i);
    if (hit.size() != 1)
      throw std::runtime_error(where + " " + describeEnd(end) + ": chai1 wants a residue of exactly one token, found "
                               + std::to_string(hit.size()));
    return hit[0];
  };
  contact.assign((size_t)n * n, -1.f);
  pocket.assign((size_t)n * n, -1.f);
  for (size_t k = 0; k < cs.size(); ++k) {
    std::string where = "constraints[" + std::to_string(k) + "]";
    if (cs[k].kind == Constraint::Contact) {
      int i = residue(cs[k].token1, where + " token1"), j = residue(cs[k].token2, where + " token2");
      contact[(size_t)i * n + j] = (float)cs[k].maxDistance;
    } else {
      auto binder = binderTokens(t, cs[k].binder, where + " pocket");
      for (auto& end : cs[k].contacts) {
        int p = residue(end, where + " pocket");
        for (int b : binder) pocket[(size_t)p * n + b] = (float)cs[k].maxDistance;
      }
    }
  }
}
// applyConstraints' checks, for a family that reads them; a family that reads neither refuses the job
inline void checkConstraints(const std::vector<Constraint>& cs, const std::string& family, const std::string& model) {
  if (cs.empty()) return;
  if (family != "boltz2" && family != "chai1")
    throw std::runtime_error(model + " was not trained with pocket or contact conditioning and cannot honour the job's"
                             " constraints; boltz2 and chai1 take pocket and contact constraints");
  for (size_t k = 0; k < cs.size(); ++k) {
    if (!(cs[k].maxDistance > 0))
      throw std::runtime_error("constraints[" + std::to_string(k) + "]: maxDistance must be a positive number of Angstrom");
    if (cs[k].kind == Constraint::Pocket && cs[k].contacts.empty())
      throw std::runtime_error("constraints[" + std::to_string(k) + "]: a pocket needs at least one contact residue");
  }
}

// ---------------------------------------------------------------- the per-pair term as a table (constraints.js)
struct ContactTable { std::vector<float> index, table; int entries = 0; };
// boltz2: the WHOLE contact term per (class, distance) - it replaces the unspecified constant. Weights row-major:
// fourier (1, C) and (C), encoder (3 + 1 + C, C) [in][out] and (C), the two constants (C).
inline ContactTable boltz2ContactTable(const std::vector<float>& classes, const std::vector<float>& threshold,
                                       const float* fourierW, const float* fourierB, const float* encoderW,
                                       const float* encoderB, const float* unspecified, const float* unselected, int C) {
  ContactTable out;
  std::map<std::pair<int, float>, int> keys;
  std::vector<std::pair<int, float>> rows;
  out.index.resize(classes.size());
  for (size_t p = 0; p < classes.size(); ++p) {
    int cls = (int)classes[p];
    float d = cls >= 2 ? threshold[p] : 0.f;
    auto key = std::make_pair(cls, d);
    auto it = keys.find(key);
    int at;
    if (it == keys.end()) { at = (int)rows.size(); keys[key] = at; rows.push_back(key); } else at = it->second;
    out.index[p] = (float)at;
  }
  out.entries = (int)rows.size();
  out.table.assign((size_t)rows.size() * C, 0.f);
  for (size_t r = 0; r < rows.size(); ++r) {
    auto [cls, d] = rows[r];
    float* o = out.table.data() + r * C;
    if (cls < 0) continue;
    if (cls == CC_UNSPECIFIED) { for (int c = 0; c < C; ++c) o[c] = unspecified[c]; continue; }
    if (cls == CC_UNSELECTED) { for (int c = 0; c < C; ++c) o[c] = unselected[c]; continue; }
    double tn = ((double)d - 4) / 16;
    std::vector<double> x(4 + C, 0.0);
    x[cls - 2] = 1; x[3] = tn;
    for (int k = 0; k < C; ++k) x[4 + k] = std::cos(2 * M_PI * (tn * fourierW[k] + fourierB[k]));
    for (int c = 0; c < C; ++c) {
      double sum = encoderB[c];
      for (int k = 0; k < 4 + C; ++k) if (x[k] != 0) sum += x[k] * encoderW[(size_t)k * C + c];
      o[c] = (float)sum;
    }
  }
  return out;
}
// chai-1: the restraint columns minus their no-restraint value, through chai1_restraint_trunk (14, C); row 0 zero
inline ContactTable chai1RestraintTable(const std::vector<float>& contact, const std::vector<float>& pocket,
                                        const float* weights, int C) {
  ContactTable out;
  auto rbf = [](double d, double lo, double hi, double* f) {
    for (int k = 0; k < 7; ++k) f[k] = 0;
    if (d == -1) return;
    double sigma = (hi - lo) / 5;
    for (int k = 0; k < 6; ++k) {
      double x = std::min(std::pow((lo + k * sigma - d) / sigma, 2.0), 16.0);
      f[k] = x == 16 ? 0 : std::exp(-x);
    }
    f[6] = -1;
  };
  std::map<std::pair<float, float>, int> keys{{{-1.f, -1.f}, 0}};
  std::vector<std::pair<float, float>> rows{{-1.f, -1.f}};
  out.index.resize(contact.size());
  for (size_t p = 0; p < contact.size(); ++p) {
    auto key = std::make_pair(contact[p], pocket[p]);
    auto it = keys.find(key);
    int at;
    if (it == keys.end()) { at = (int)rows.size(); keys[key] = at; rows.push_back(key); } else at = it->second;
    out.index[p] = (float)at;
  }
  out.entries = (int)rows.size();
  out.table.assign((size_t)rows.size() * C, 0.f);
  for (size_t r = 1; r < rows.size(); ++r) {
    double f[14];
    rbf(rows[r].first, 6, 30, f);
    rbf(rows[r].second, 6, 20, f + 7);
    for (int c = 0; c < C; ++c) {
      double sum = 0;
      for (int k = 0; k < 14; ++k) sum += f[k] * weights[(size_t)k * C + c];
      out.table[r * C + c] = (float)sum;
    }
  }
  return out;
}

}  // namespace lf
