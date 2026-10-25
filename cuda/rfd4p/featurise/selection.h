// `evalPathSelection`: a condition's `sel('chain/res_name/res_id/atom_name')` over real atoms.
//
// 🔴 THE CONTIG SUBSET COULD NOT EXPRESS WHAT THE SHIPPED PRESETS SAY. `featurise.cpp`'s
// selection reader takes a union of `sel('*/<segment>/*/*')`, which is everything a pure design
// job can mean - there are no chains, no residue numbers and no atoms. Every structure preset
// says more than that: `ppi_pdl1` locks a chain with `sel('A/*/*/*')` and names four hotspot
// residues with `sel('A/*/[54,56,66,115]/*')`.
//
// 🔴 AND A FIELD IS SPLIT AT THE TOP LEVEL ONLY. `[54,56]` contains a comma and `A/*/[1-3,7]/*`
// contains both separators, so splitting on `,` or `/` naively tears a list in half and the
// selection then matches a residue nobody named.
#pragma once
#include "atoms.h"

namespace rfd4p {
namespace selection {

using std::string;
using std::vector;

/** `splitTopLevel`: split on `sep`, ignoring one inside `[...]` or quotes. */
inline vector<string> splitTopLevel(const string& s, char sep) {
  vector<string> parts;
  string buf;
  int depth = 0;
  bool inSingle = false, inDouble = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char ch = s[i];
    if (ch == '\'' && !inDouble) {
      if (inSingle && i + 1 < s.size() && s[i + 1] == '\'') { buf += "''"; ++i; continue; }
      inSingle = !inSingle;
      buf += ch;
      continue;
    }
    if (ch == '"' && !inSingle) {
      if (i == 0 || s[i - 1] != '\\') inDouble = !inDouble;
      buf += ch;
      continue;
    }
    if (!inSingle && !inDouble) {
      if (ch == '[' || ch == '(') ++depth;
      else if (ch == ']' || ch == ')') --depth;
      else if (ch == sep && depth == 0) { parts.push_back(buf); buf.clear(); continue; }
    }
    buf += ch;
  }
  parts.push_back(buf);
  return parts;
}

inline string trim(string v) {
  while (!v.empty() && isspace((unsigned char)v.front())) v.erase(v.begin());
  while (!v.empty() && isspace((unsigned char)v.back())) v.pop_back();
  return v;
}

/**
 * `expandPathField`: one field of a path, as the set of values it names.
 *
 * 🔴 A RANGE IS ONLY A RANGE INSIDE A LIST AND ONLY FOR RESIDUE NUMBERS. `[1-3]` is 1, 2, 3;
 * a bare `1-3` is the single value `parseInt("1-3")` = 1, and an ATOM NAME list never expands
 * a hyphen at all - which matters, because plenty of atom names contain one.
 *
 * 🔴 AND A RANGE MAY COUNT DOWN. `[7-3]` is 7, 6, 5, 4, 3; the reference picks the step from
 * the endpoints rather than requiring them in order.
 */
inline vector<string> expandPathField(const string& rawField, bool isResId) {
  const string v = trim(rawField.empty() ? "*" : rawField);
  if (v.empty() || v == "*") return {"*"};
  if (v.size() >= 2 && v.front() == '[' && v.back() == ']') {
    vector<string> out;
    for (const string& raw : splitTopLevel(v.substr(1, v.size() - 2), ',')) {
      const string item = trim(raw);
      if (item.empty()) continue;
      if (item == "*") { out.push_back("*"); continue; }
      if (isResId) {
        // `-?\d+--?\d+`: find the hyphen that is not a leading sign.
        size_t at = item[0] == '-' ? 1 : 0;
        size_t dash = string::npos;
        while (at < item.size()) {
          if (item[at] == '-') { dash = at; break; }
          if (!isdigit((unsigned char)item[at])) { dash = string::npos; at = item.size(); break; }
          ++at;
        }
        bool range = dash != string::npos && dash + 1 < item.size();
        if (range) {
          const string lo = item.substr(0, dash), hi = item.substr(dash + 1);
          auto numeric = [](const string& x) {
            if (x.empty()) return false;
            for (size_t k = (x[0] == '-' ? 1u : 0u); k < x.size(); ++k) {
              if (!isdigit((unsigned char)x[k])) return false;
            }
            return x != "-";
          };
          if (numeric(lo) && numeric(hi)) {
            const int s = atoi(lo.c_str()), e = atoi(hi.c_str());
            const int step = s <= e ? 1 : -1;
            for (int k = s; step > 0 ? k <= e : k >= e; k += step) out.push_back(std::to_string(k));
            continue;
          }
        }
        // 🔴 NORMALISED THROUGH THE INTEGER, so `[0054]` and `[54]` name the same residue and
        // `[ 54 ]` is not a third thing. Inert on every selection the presets write, which all
        // use clean numbers - which is why it is stated rather than left to be discovered.
        out.push_back(std::to_string(atoi(item.c_str())));
      } else {
        out.push_back(item);
      }
    }
    return out;
  }
  return {isResId ? std::to_string(atoi(v.c_str())) : v};
}

/** `evalPathSelection`: the atoms a comma-separated set of paths names. */
inline vector<uint8_t> evalPathSelection(const atoms::AtomArray& a, const string& selectionStr) {
  const size_t n = a.n();
  vector<uint8_t> out(n, 0);
  for (const string& rawToken : splitTopLevel(selectionStr, ',')) {
    const string token = trim(rawToken);
    if (token.empty()) continue;
    const vector<string> parts = splitTopLevel(token, '/');
    auto field = [&](size_t k, bool isResId) {
      return expandPathField(k < parts.size() ? parts[k] : "*", isResId);
    };
    const vector<string> chains = field(0, false);
    const vector<string> resNames = field(1, false);
    const vector<string> resIds = field(2, true);
    const vector<string> atomNames = field(3, false);
    const vector<string> transIds = field(4, false);
    auto asSet = [](const vector<string>& v) -> std::set<string> {
      for (const string& x : v) if (x == "*") return {};
      return std::set<string>(v.begin(), v.end());
    };
    const std::set<string> mChain = asSet(chains), mResName = asSet(resNames),
                           mResId = asSet(resIds), mAtom = asSet(atomNames),
                           mTrans = asSet(transIds);
    for (size_t i = 0; i < n; ++i) {
      // The reference's own skip. It cannot change the answer - this loop only ever SETS a
      // bit - and removing it leaves all seventeen cases green; kept because it is theirs.
      if (out[i]) continue;
      if (!mChain.empty() && !mChain.count(a.site.chainId[i])) continue;
      if (!mResName.empty() && !mResName.count(a.site.resName[i])) continue;
      if (!mResId.empty() && !mResId.count(std::to_string(a.site.resId[i]))) continue;
      if (!mAtom.empty() && !mAtom.count(a.site.atomName[i])) continue;
      // 🔴 `transformation_id` IS "1" WHERE THE READER DID NOT SET ONE, not absent - the
      // reference defaults it the same way (`cif.js` fills the column with "1" outright), and a
      // selection that names it must match that. Checked both ways on all eighteen structures:
      // `.../1` is every atom and `.../2` is none.
      //
      // 🔴 AND IT IS A CONSTANT HERE RATHER THAN A COLUMN, WHICH IS ONLY SAFE BECAUSE THIS PORT
      // DOES NOT EXPAND AN ASSEMBLY. `prepare.js` writes a copy's operation into
      // `transformation_id` and then renames every chain `<chain>_<id>`; the day that path is
      // ported, this line is a silent wrong answer on every selection that names a copy, and
      // the corpus above cannot see it because no file it reads has a second copy.
      if (!mTrans.empty() && !mTrans.count("1")) continue;
      out[i] = 1;
    }
  }
  return out;
}

/**
 * The expression layer, as far as it is covered: a union of `sel(...)` terms.
 *
 * 🔴 WHAT IS NOT COVERED IS REFUSED BY NAME. The reference has a whole tokenizer and parser -
 * `and`, `or`, `not`, comparisons, lists, `has_bonds`, `has_nan_coord`, `sa_mask` and bare
 * attribute names. Every shipped preset is one `sel(...)`, so that is what this takes; a
 * selection it does not understand must NOT quietly match nothing, because a condition the
 * model is told is absent is a different job from the one that was asked for.
 */
inline vector<uint8_t> evalSelection(const atoms::AtomArray& a, const string& expr) {
  const size_t n = a.n();
  vector<uint8_t> out(n, 0);
  size_t at = 0, lastEnd = 0;
  bool sawTerm = false;
  while (at < expr.size()) {
    const size_t open = expr.find("sel('", at);
    if (open == string::npos) break;
    const size_t close = expr.find("')", open);
    if (close == string::npos) {
      fprintf(stderr, "selection '%s' has an unterminated sel(...)\n", expr.c_str());
      exit(1);
    }
    // 🔴 WHAT SEPARATES TWO TERMS DECIDES WHETHER THERE ARE TWO. The reference parses an
    // EXPRESSION, so `sel(a)|sel(b)` is a union and `sel(a),sel(b)` is `sel(a)` - the parser
    // returns at the comma and the second term is silently dropped. Measured: unioning the
    // comma form put 8 extra atoms in the mask on three structures. Neither behaviour is safe
    // to guess at, so a comma between terms is REFUSED: matching the reference would mean
    // silently discarding a condition somebody wrote, and unioning means applying one they
    // did not get.
    if (sawTerm) {
      const string gap = trim(expr.substr(lastEnd, open - lastEnd));
      if (gap != "|" && gap != "or") {
        fprintf(stderr, "selection '%s' joins terms with '%s'; only '|' is covered, and the "
                        "reference DROPS everything after a comma\n", expr.c_str(), gap.c_str());
        exit(1);
      }
    }
    const vector<uint8_t> m = evalPathSelection(a, expr.substr(open + 5, close - open - 5));
    for (size_t i = 0; i < n; ++i) out[i] |= m[i];
    sawTerm = true;
    at = close + 2;
    lastEnd = at;
  }
  if (!sawTerm) {
    fprintf(stderr, "selection '%s' is not a union of sel(...) terms, which is all this "
                    "featuriser covers\n", expr.c_str());
    exit(1);
  }
  if (expr.find('&') != string::npos || expr.find('~') != string::npos ||
      expr.find(" and ") != string::npos || expr.find(" not ") != string::npos) {
    fprintf(stderr, "selection '%s' uses and/not; only a union of sel(...) is covered\n",
            expr.c_str());
    exit(1);
  }
  return out;
}

}  // namespace selection
}  // namespace rfd4p
