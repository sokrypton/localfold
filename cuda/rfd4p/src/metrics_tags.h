#pragma once
// A job's `metrics_tags` as bits, in shared/rfd4p/metrics.js's `METRICS_TAGS` order: every tag a
// ported design metric, pre-oracle filter or refolding oracle gates on. A tag outside the list gates nothing here,
// so dropping it loses nothing.
#include <algorithm>
#include <string>
#include <vector>

namespace rfd4p {

inline const std::vector<std::string>& metricsTagNames() {
  static const std::vector<std::string> names = {
      "design", "structure_prediction", "small_molecule_only", "hotspot_conditioning", "unindexed",
      "partial_diffusion", "insertion_rmsd_filter", "motif_rmsd_filter", "ligand_clash_filter",
      "no_ligand", "3d_coordinate_conditioning", "internal_coordinate_conditioning",
      // the refolding oracle's, its refolding metrics' and its pass criteria's (oracle.h)
      "unconditional_refolding", "af3_refolding", "motif_scaffold_refolding", "internal_coordinate_motif_scaffold_refolding",
      "sm_binder_refolding", "ame_refolding", "ame_no_ligand_refolding", "ame_original_benchmark_refolding",
      "disco_refolding", "ppi_refolding", "antibody", "dna_refolding",
      "apo_refolding", "developability", "af2_refolding"};
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
