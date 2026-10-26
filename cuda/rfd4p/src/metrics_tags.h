#pragma once
// A job's `metrics_tags` as bits, in shared/rfd4p/metrics.js's `METRICS_TAGS` order: every tag a
// ported design metric or pre-oracle filter gates on. A tag outside the list gates nothing here,
// so dropping it loses nothing.
#include <algorithm>
#include <string>
#include <vector>

namespace rfd4p {

inline const std::vector<std::string>& metricsTagNames() {
  static const std::vector<std::string> names = {
      "design", "structure_prediction", "small_molecule_only", "hotspot_conditioning", "unindexed",
      "partial_diffusion", "insertion_rmsd_filter", "motif_rmsd_filter", "ligand_clash_filter",
      "no_ligand"};
  return names;
}

inline int metricsTagBits(const std::vector<std::string>& tags) {
  int bits = 0;
  const auto& names = metricsTagNames();
  for (size_t k = 0; k < names.size(); ++k) {
    if (std::find(tags.begin(), tags.end(), names[k]) != tags.end()) bits |= 1 << k;
  }
  return bits;
}

inline std::vector<std::string> metricsTagsFromBits(int bits) {
  std::vector<std::string> out;
  const auto& names = metricsTagNames();
  for (size_t k = 0; k < names.size(); ++k) if (bits & (1 << k)) out.push_back(names[k]);
  return out;
}

}  // namespace rfd4p
