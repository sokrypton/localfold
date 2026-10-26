// `applyConditions`: a spec's `conditions:` written onto a real atom array.
//
// 🔴 THE SELECTIONS ARE EVALUATED AGAINST THE INPUT'S OWN RESIDUE NUMBERS, NOT THE ASSEMBLED
// ONES. `applyContig` renumbers every preserved segment from 1, so by the time conditions are
// applied the array no longer carries the numbers the job was written against. The reference
// swaps `res_id` for `originalResId` for the whole of this function and puts it back in a
// `finally` - so a condition naming residue 54 of the input means residue 54 OF THE INPUT, not
// whatever is 54th in the output. Getting this wrong silently conditions the wrong residues.
#pragma once
#include "atoms.h"
#include "selection.h"

namespace rfd4p {
namespace conditions {

using std::string;
using std::vector;

struct Entry {
  string select;
  string group;
  bool hasGroup = false;
  size_t keys = 0;   // how many fields the entry named, which `C_CRDGRP` holds to `select` alone
};
// 🔴 INSERTION ORDER, NOT SORTED. Two arms of one condition can touch the same atoms - a
// `C_CRD` with a `true` arm over a chain and a `false` arm over two of its residues is exactly
// that - and the LAST one applied wins. The reference walks the JSON object, which is insertion
// order; a `std::map` sorts, so "false" came before "true" and the false arm was overwritten
// instead of overwriting. Measured: 48 coordinates and 16 groups wrong on every structure.
using Arms = vector<std::pair<string, vector<Entry>>>;   // condition value -> entries, in order
using Spec = vector<std::pair<string, Arms>>;            // annotation name -> arms, in order

inline bool coerceBool(const string& v, const char* what) {
  string low = v;
  for (char& c : low) c = (char)tolower((unsigned char)c);
  if (low == "true") return true;
  if (low == "false") return false;
  fprintf(stderr, "%s: expected a boolean condition value (true/false), got '%s'\n", what,
          v.c_str());
  exit(1);
}

/** `getDefaultCoordinateGroups`: 0 where the conditioned coordinate is finite, -1 where not. */
inline vector<int> defaultCoordinateGroups(const atoms::AtomArray& a) {
  const size_t n = a.n();
  vector<int> groups(n, -1);
  auto it = a.ann.floats.find("condition_coordinate_1_atom");
  if (it == a.ann.floats.end() || it->second.size() != n * 3) return groups;
  for (size_t i = 0; i < n; ++i) {
    if (std::isfinite(it->second[i * 3]) && std::isfinite(it->second[i * 3 + 1]) &&
        std::isfinite(it->second[i * 3 + 2])) {
      groups[i] = 0;
    }
  }
  return groups;
}

/**
 * `separate_coordinate_groups` (spec.js): each input's internal-coordinate groups renumbered into
 * one namespace, so two files' group 1 do not become one rigid group. In place.
 */
inline void separateCoordinateGroups(vector<atoms::AtomArray>& arrays) {
  int next = 1;
  for (atoms::AtomArray& a : arrays) {
    const size_t n = a.n();
    auto have = a.ann.ints.find("condition_CoordinateGroup_1_atom");
    const vector<int> old = have != a.ann.ints.end() ? have->second : defaultCoordinateGroups(a);
    vector<int> groups = old;
    auto intcrd = a.ann.floats.find("condition_InternalCoordinate_1_atom");
    if (intcrd != a.ann.floats.end() && intcrd->second.size() == n * 3) {
      for (size_t i = 0; i < n; ++i) {
        if (old[i] > 0) continue;
        for (int c = 0; c < 3; ++c) {
          if (std::isfinite(intcrd->second[i * 3 + c])) {
            fprintf(stderr, "Internal coordinate targets exist outside internal coordinate groups.\n");
            exit(1);
          }
        }
      }
    }
    std::set<int> ids;
    for (const int g : old) if (g > 0) ids.insert(g);
    for (const int gid : ids) {
      for (size_t i = 0; i < n; ++i) if (old[i] == gid) groups[i] = next;
      ++next;
    }
    if (have != a.ann.ints.end()) have->second = groups;
  }
}

// `set_cath_condition` (shared/rfd4p/conditions.js `normalizeCatLabel`): a C.A.T label from the
// model's own vocabulary - `x.x.x`, `C.x.x`, `C.A.x` or `C.A.T` - with a fourth `.x` accepted and
// dropped, stored with a trailing `.x`. This appended `.x` to anything, so `3.40.50.20` became a
// five-field code and a typo became a label nothing maps.
inline string normalizeCatLabel(const string& value) {
  string label = value;
  vector<string> parts;
  {
    size_t at = 0;
    while (true) {
      const size_t dot = label.find('.', at);
      parts.push_back(label.substr(at, dot == string::npos ? string::npos : dot - at));
      if (dot == string::npos) break;
      at = dot + 1;
    }
  }
  if (parts.size() == 4 && parts[3] == "x") {
    label = label.substr(0, label.size() - 2);
    parts.pop_back();
  }
  const bool valid =
      label == "x.x.x" ||
      (parts.size() == 3 && parts[1] == "x" && parts[2] == "x" && tables::CATH_C.count(parts[0])) ||
      (parts.size() == 3 && parts[2] == "x" && tables::CATH_A.count(parts[0] + "." + parts[1])) ||
      (parts.size() == 3 && tables::CATH_T.count(label));
  if (!valid) {
    fprintf(stderr, "Unknown CAT label '%s'; choose a C/A/T label from the model vocabulary, or "
                    "'x.x.x' to clear.\n", value.c_str());
    exit(1);
  }
  return label + ".x";
}

inline bool isBooleanText(const string& v) {
  string low = v;
  for (char& c : low) c = (char)tolower((unsigned char)c);
  return low == "true" || low == "false";
}

inline void applyConditions(atoms::AtomArray& a, const Spec& conditions,
                            const vector<int>& originalResId) {
  const size_t n = a.n();
  // 🔴 UPSTREAM'S CHECKS BEFORE ANYTHING IS WRITTEN (`condition_spec._apply_conditions`; spec.js
  // has the same), in its words: a direct `C_INTCRD`, a two-body name, a pipeline annotation, and
  // a `C_CRDGRP` value that is not a positive group id or entries naming more than `select`.
  for (const auto& [annotationName, values] : conditions) {
    auto it = tables::CONDITION_REGISTRY.find(annotationName);
    if (it == tables::CONDITION_REGISTRY.end()) continue;   // refused by name below
    const string dtype = it->second.dtype;
    if (dtype == "intcrd") {
      fprintf(stderr, "C_INTCRD is a saved payload; edit C_CRDGRP instead.\n");
      exit(1);
    }
    if (dtype == "twobody") {
      fprintf(stderr, "'%s' resolves to 'basepair', a 2-body condition -- condition_spec has no "
                      "selection-pair grammar, so only 1-body conditions (per-atom or per-residue) "
                      "can be set here.\n", annotationName.c_str());
      exit(1);
    }
    if (dtype == "internal") {
      fprintf(stderr, "'%s' names %s, which the pipeline writes itself; this port does not take "
                      "it from a job.\n", annotationName.c_str(), it->second.fullName);
      exit(1);
    }
    if (dtype == "crdgrp") {
      for (const auto& [value, entries] : values) {
        if (value != "null" && !value.empty()) {
          bool digits = true;
          for (char c : value) digits = digits && isdigit((unsigned char)c);
          const long v = digits && value.size() < 9 ? atol(value.c_str()) : -1;
          if (!digits || !(v > 0 && v <= 32767)) {
            fprintf(stderr, "C_CRDGRP requires an integer ID from 1 to 32767; reserved IDs are not "
                            "editable.\n");
            exit(1);
          }
        }
        for (const Entry& e : entries) {
          if (e.keys != 1 || e.select.empty()) {
            fprintf(stderr, "C_CRDGRP entries must contain only select; the value is the group "
                            "ID.\n");
            exit(1);
          }
        }
      }
    }
  }
  const vector<int> realResId = a.site.resId;
  if (originalResId.size() == n) a.site.resId = originalResId;
  // 🔴 ORDERED, because a later group's id depends on how many came before it.
  vector<std::pair<string, vector<int>>> pendingIndexGroups;

  for (const auto& [annotationName, values] : conditions) {
    // The registry, then the open-ended `C_SEQ_PROB_<token>` family (`resolveConditionSpec`):
    // upstream registers one sequence-probability condition per standard residue, under names
    // a fixed table does not hold.
    string full, dtype;
    double resolvedDefault = 0;
    try {
      const selection::query::Resolved r = selection::query::resolve(annotationName);
      full = r.fullName;
      dtype = r.dtype;
      if (r.def.t == selection::query::Q::NUM) resolvedDefault = r.def.d;
    } catch (const std::exception&) {
      fprintf(stderr, "condition '%s' is not in the registry\n", annotationName.c_str());
      exit(1);
    }
    const auto specIt = tables::CONDITION_REGISTRY.find(annotationName);
    const string enumName = specIt == tables::CONDITION_REGISTRY.end() ? "" : specIt->second.enumName;
    auto annSpec = tables::ANNOTATIONS_1BODY.find(full);
    // A column the table does not describe takes the condition's own default: NaN for a
    // sequence probability, which is "unspecified" - a 0 there would constrain every residue.
    const double fallback =
        annSpec == tables::ANNOTATIONS_1BODY.end() ? resolvedDefault : annSpec->second.fallback;
    const string fallbackText =
        annSpec == tables::ANNOTATIONS_1BODY.end() ? string() : string(annSpec->second.text);

    if (dtype == "crdgrp") {
      // 🔴 A COORDINATE GROUP IS ASSIGNED OVER THE WHOLE CONDITION AT ONCE and `null` means
      // "back to the default", which is 0 where the coordinate is conditioned and -1 where it
      // is not - not the literal -1, and not zero.
      if (!a.ann.ints.count(full)) a.ann.ints[full] = defaultCoordinateGroups(a);
      vector<int> groups = a.ann.ints[full];
      const vector<int> defaults = defaultCoordinateGroups(a);
      for (const auto& [value, entries] : values) {
        for (const Entry& e : entries) {
          const vector<uint8_t> mask = selection::evalSelection(a, e.select);
          for (size_t i = 0; i < n; ++i) {
            if (!mask[i]) continue;
            groups[i] = (value.empty() || value == "null") ? defaults[i] : atoi(value.c_str());
          }
        }
      }
      a.ann.ints[full] = groups;
      continue;
    }

    for (const auto& [value, entries] : values) {
      vector<std::pair<string, vector<int>>> groupedIndices;
      for (const Entry& e : entries) {
        const vector<uint8_t> mask = selection::evalSelection(a, e.select);
        vector<int> indices;
        for (size_t i = 0; i < n; ++i) if (mask[i]) indices.push_back((int)i);

        const bool unindexedGroup =
            full == "condition_index_1_residue" && !coerceBool(value, annotationName.c_str());
        if (!unindexedGroup && e.hasGroup) {
          fprintf(stderr, "%s: `group:` is only meaningful for C_IDX (value false), not \"%s\".\n",
                  annotationName.c_str(), value.c_str());
          exit(1);
        }
        if (unindexedGroup) {
          if (!e.hasGroup) {
            fprintf(stderr, "%s: value %s requires a 'group:' field alongside 'select:'\n",
                    annotationName.c_str(), value.c_str());
            exit(1);
          }
          auto at = std::find_if(groupedIndices.begin(), groupedIndices.end(),
                                 [&](const auto& kv) { return kv.first == e.group; });
          if (at == groupedIndices.end()) { groupedIndices.push_back({e.group, {}}); at = groupedIndices.end() - 1; }
          at->second.insert(at->second.end(), indices.begin(), indices.end());
          continue;
        }
        if (dtype == "coord") {
          const bool active = coerceBool(value, annotationName.c_str());
          if (!a.ann.floats.count(full)) a.ann.floats[full].assign(n * 3, NAN);
          vector<double>& crd = a.ann.floats[full];
          for (const int idx : indices) {
            crd[idx * 3] = active ? a.site.x[idx] : NAN;
            crd[idx * 3 + 1] = active ? a.site.y[idx] : NAN;
            crd[idx * 3 + 2] = active ? a.site.z[idx] : NAN;
          }
          // 🔴 AND THE COORDINATE GROUP IS RE-SYNCED, because `condition_coordinate` having
          // changed moves what the DEFAULT group is - but only where the condition was turned
          // ON, or where the group was not already a positive one somebody chose.
          vector<int> crdGrp = a.ann.ints.count("condition_CoordinateGroup_1_atom")
                                   ? a.ann.ints["condition_CoordinateGroup_1_atom"]
                                   : defaultCoordinateGroups(a);
          const vector<int> defGrp = defaultCoordinateGroups(a);
          for (const int idx : indices) {
            if (active || crdGrp[idx] <= 0) crdGrp[idx] = defGrp[idx];
          }
          a.ann.ints["condition_CoordinateGroup_1_atom"] = crdGrp;
        } else if (dtype == "seq") {
          // 🔴 TRUE OR FALSE AND NOTHING ELSE: a sequence condition copies the residue's own
          // identity (upstream's `ground_truth_value`) or masks it. This also took a literal
          // residue name - `C_SEQ: {ALA: ...}` - which upstream refuses as not a boolean.
          if (!a.ann.strings.count(full)) a.ann.strings[full].assign(n, "<M>");
          vector<string>& seq = a.ann.strings[full];
          const bool on = coerceBool(value, annotationName.c_str());
          for (const int idx : indices) seq[idx] = on ? a.site.resName[idx] : string("<M>");
        } else if (dtype == "bool") {
          if (!a.ann.ints.count(full)) a.ann.ints[full].assign(n, (int)fallback);
          const bool b = coerceBool(value, annotationName.c_str());
          for (const int idx : indices) a.ann.ints[full][idx] = b ? 1 : 0;
        } else if (dtype == "enum") {
          if (!a.ann.ints.count(full)) a.ann.ints[full].assign(n, (int)fallback);
          int enumVal = 0;
          bool numeric = !value.empty();
          for (size_t k = (value[0] == '-' ? 1u : 0u); k < value.size(); ++k) {
            if (!isdigit((unsigned char)value[k])) { numeric = false; break; }
          }
          if (numeric) {
            // `coerce_enum_value`: an integer must BE a member's value.
            enumVal = atoi(value.c_str());
            auto members = tables::CONDITION_ENUMS.find(enumName);
            bool member = false;
            if (members != tables::CONDITION_ENUMS.end()) {
              for (const auto& [name, v] : members->second) member = member || v == enumVal;
            }
            if (!member) {
              fprintf(stderr, "Invalid enum value %s for %s\n", value.c_str(),
                      annotationName.c_str());
              exit(1);
            }
          } else {
            auto members = tables::CONDITION_ENUMS.find(enumName);
            if (members == tables::CONDITION_ENUMS.end()) {
              fprintf(stderr, "condition '%s' names enum '%s', which is not a table here\n",
                      annotationName.c_str(), enumName.c_str());
              exit(1);
            }
            auto hit = members->second.find(value);
            if (hit == members->second.end()) {
              fprintf(stderr, "Invalid enum value %s for %s\n", value.c_str(),
                      annotationName.c_str());
              exit(1);
            }
            enumVal = hit->second;
          }
          for (const int idx : indices) a.ann.ints[full][idx] = enumVal;
        } else if (dtype == "centroid") {
          if (!a.ann.floats.count(full)) a.ann.floats[full].assign(n * 3, NAN);
          vector<double> parts;
          for (const string& p : selection::splitTopLevel(value, ',')) {
            parts.push_back(atof(selection::trim(p).c_str()));
          }
          // 🔴 EXACTLY THREE, AS UPSTREAM'S `parse_centroid_value` REQUIRES. Fewer was refused
          // here as a deliberate divergence from the page, which stored `slice(0, 3)` - two
          // numbers as a two-element "point" - and more were truncated on both sides. Upstream
          // refuses anything but three, and now so do both ports.
          bool finite = true;
          for (const double v : parts) finite = finite && std::isfinite(v);
          if (parts.size() != 3 || !finite) {
            fprintf(stderr, "ChainCentroid value must have exactly 3 components (x, y, z), got "
                            "'%s'.\n", value.c_str());
            exit(1);
          }
          for (const int idx : indices) {
            for (int k = 0; k < 3; ++k) a.ann.floats[full][idx * 3 + k] = parts[k];
          }
        } else if (dtype == "cath") {
          // `set_cath_condition`: a label from the model's vocabulary, see `normalizeCatLabel`.
          // (This appended `.x` to whatever it was given, as the page did - so `3.40.50.20`
          // became a five-field code and a one-level `3` became `3.x`, both labels upstream
          // refuses and nothing maps.)
          if (!a.ann.strings.count(full)) a.ann.strings[full].assign(n, "x.x.x.x");
          const string cath = normalizeCatLabel(value);
          for (const int idx : indices) a.ann.strings[full][idx] = cath;
        } else if (dtype == "float") {
          // A number, which a boolean is not and NaN is not either (`Number('abc')` wrote NaN,
          // which for a sequence probability means "unspecified", silently).
          const string num = selection::trim(value);   // `Number()` ignores surrounding space
          char* end = nullptr;
          const double f = strtod(num.c_str(), &end);
          if (isBooleanText(num) || num.empty() || (end && *end) || !std::isfinite(f)) {
            fprintf(stderr, "%s writes a number, so it cannot take \"%s\".\n",
                    annotationName.c_str(), value.c_str());
            exit(1);
          }
          if (!a.ann.floats.count(full)) a.ann.floats[full].assign(n, fallback);
          for (const int idx : indices) a.ann.floats[full][idx] = f;
        } else {
          if (isBooleanText(value)) {
            fprintf(stderr, "%s writes a value, so it cannot take \"%s\".\n", full.c_str(),
                    value.c_str());
            exit(1);
          }
          if (!a.ann.strings.count(full)) a.ann.strings[full].assign(n, fallbackText);
          for (const int idx : indices) a.ann.strings[full][idx] = value;
        }
      }

      if (!groupedIndices.empty()) {
        if (!a.ann.ints.count("condition_index_1_residue")) {
          a.ann.ints["condition_index_1_residue"].assign(n, 1);
        }
        for (const auto& [label, idxs] : groupedIndices) {
          for (const int idx : idxs) a.ann.ints["condition_index_1_residue"][idx] = 0;
          auto at = std::find_if(pendingIndexGroups.begin(), pendingIndexGroups.end(),
                                 [&](const auto& kv) { return kv.first == label; });
          if (at == pendingIndexGroups.end()) { pendingIndexGroups.push_back({label, {}}); at = pendingIndexGroups.end() - 1; }
          at->second.insert(at->second.end(), idxs.begin(), idxs.end());
        }
      }
    }
  }

  // 🔴 A GROUP IS NUMBERED AFTER EVERY DEFAULT GROUP, AND AN EMPTY ONE TAKES NO NUMBER. The ids
  // start past the highest the default grouping produced, and a label whose atoms all turned
  // out to be indexed anyway is SKIPPED rather than given an id nothing carries - which would
  // shift every later group by one.
  if (!pendingIndexGroups.empty()) {
    vector<int> groupIds = atoms::defaultIndexGroups(a);
    const vector<int>& isIndexed = a.ann.ints["condition_index_1_residue"];
    int nextId = -1;
    for (const int v : groupIds) nextId = std::max(nextId, v);
    ++nextId;
    for (const auto& [label, idxs] : pendingIndexGroups) {
      vector<int> unindexed;
      for (const int idx : idxs) if (!isIndexed[idx]) unindexed.push_back(idx);
      if (unindexed.empty()) continue;
      for (const int idx : unindexed) groupIds[idx] = nextId;
      ++nextId;
    }
    a.ann.ints["condition_index-group_1_token"] = groupIds;
  }

  a.site.resId = realResId;
  a.ids = cif::computeIds(a.site, a.bonds);
}

}  // namespace conditions
}  // namespace rfd4p
