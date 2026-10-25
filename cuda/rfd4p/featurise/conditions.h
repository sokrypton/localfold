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

inline void applyConditions(atoms::AtomArray& a, const Spec& conditions,
                            const vector<int>& originalResId) {
  const size_t n = a.n();
  const vector<int> realResId = a.site.resId;
  if (originalResId.size() == n) a.site.resId = originalResId;
  // 🔴 ORDERED, because a later group's id depends on how many came before it.
  vector<std::pair<string, vector<int>>> pendingIndexGroups;

  for (const auto& [annotationName, values] : conditions) {
    auto specIt = tables::CONDITION_REGISTRY.find(annotationName);
    if (specIt == tables::CONDITION_REGISTRY.end()) {
      fprintf(stderr, "condition '%s' is not in the registry\n", annotationName.c_str());
      exit(1);
    }
    const string full = specIt->second.fullName, dtype = specIt->second.dtype;
    auto annSpec = tables::ANNOTATIONS_1BODY.find(full);
    const double fallback = annSpec == tables::ANNOTATIONS_1BODY.end() ? 0 : annSpec->second.fallback;
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
          // 🔴 `true` MEANS "THE RESIDUE'S OWN NAME", not the string "true". A sequence
          // condition is either a boolean - lock this residue to what it is, or mask it - or a
          // literal residue name to force.
          if (!a.ann.strings.count(full)) a.ann.strings[full].assign(n, "<M>");
          vector<string>& seq = a.ann.strings[full];
          string low = value;
          for (char& c : low) c = (char)tolower((unsigned char)c);
          const bool isBool = low == "true" || low == "false";
          for (const int idx : indices) {
            seq[idx] = isBool ? (low == "true" ? a.site.resName[idx] : string("<M>")) : value;
          }
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
            enumVal = atoi(value.c_str());
          } else {
            auto members = tables::CONDITION_ENUMS.find(specIt->second.enumName);
            if (members == tables::CONDITION_ENUMS.end()) {
              fprintf(stderr, "condition '%s' names enum '%s', which is not a table here\n",
                      annotationName.c_str(), specIt->second.enumName);
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
          parts.resize(3, NAN);
          for (const int idx : indices) {
            for (int k = 0; k < 3; ++k) a.ann.floats[full][idx * 3 + k] = parts[k];
          }
        } else if (dtype == "cath") {
          // 🔴 A CATH CODE GAINS A TRAILING `.x` IF IT HAS NONE, so `3.40.50` is stored as
          // `3.40.50.x` - the fourth level is the homologous superfamily and a spec that does
          // not name it is saying "any", not "level four is missing".
          if (!a.ann.strings.count(full)) a.ann.strings[full].assign(n, "x.x.x.x");
          string cath = value;
          if (cath.size() < 2 || cath.compare(cath.size() - 2, 2, ".x") != 0) cath += ".x";
          for (const int idx : indices) a.ann.strings[full][idx] = cath;
        } else if (dtype == "float") {
          if (!a.ann.floats.count(full)) a.ann.floats[full].assign(n, fallback);
          const double f = atof(value.c_str());
          for (const int idx : indices) a.ann.floats[full][idx] = f;
        } else {
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
