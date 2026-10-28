// Crosslinks (af3x's `crosslinks`): the native twin of the crosslink row in web/entities.js and of
// shared/chem/crosslinkers.js. A crosslinker is its reacted bridge, a component of its own (XL-<name>), bonded to the
// two residues it joined: an NHS-ester end to K NZ / S OG / T OG1 / Y OH or residue 1's N, a diazirine end to the
// residue's last side-chain atom. The data is generated (crosslinkers.inc, tools/gen-native-featuriser-tables.mjs).
#pragma once
#include <cctype>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace lf {

struct CrosslinkerEntry {
  std::string name, code, end1, end2, attach1, attach2;
  std::string component;    // its mmCIF block
};
#include "crosslinkers.inc"

inline std::string crosslinkUpper(std::string s) { for (auto& c : s) c = (char)std::toupper((unsigned char)c); return s; }

// a crosslinker by name, as af3x and the suppliers write them (sulfo-SDA, BS(PEG)5, ...) - or null
inline const CrosslinkerEntry* crosslinker(const std::string& name) {
  std::string key = crosslinkUpper(name);
  size_t a = key.find_first_not_of(" \t\r\n"), b = key.find_last_not_of(" \t\r\n");
  key = a == std::string::npos ? std::string() : key.substr(a, b - a + 1);
  for (auto& [alias, to] : crosslinkerAliases()) if (alias == key) { key = to; break; }
  for (auto& e : crosslinkerTable()) if (e.name == key) return &e;
  return nullptr;
}
// a built-in bridge component's mmCIF block by its code, or null
inline const std::string* crosslinkerComponent(const std::string& code) {
  std::string key = crosslinkUpper(code);
  for (auto& e : crosslinkerTable()) if (e.code == key) return &e.component;
  return nullptr;
}
inline const std::string* crosslinkLookup(const std::vector<std::pair<std::string, std::string>>& table,
                                          const std::string& key) {
  for (auto& [k, v] : table) if (k == key) return &v;
  return nullptr;
}

struct CrosslinkRowEnd { std::string chain; int residue; };
struct CrosslinkRow { std::string name; CrosslinkRowEnd from, to; };
// web/entities.js parseCrosslink: "DSSO: A2 - B30"
inline std::optional<CrosslinkRow> parseCrosslink(const std::string& value) {
  static const std::regex row(R"(^\s*([^:]+?)\s*:\s*([A-Za-z]+)\s*(\d+)\s*-\s*([A-Za-z]+)\s*(\d+)\s*$)");
  std::smatch m;
  if (!std::regex_match(value, m, row)) return std::nullopt;
  return CrosslinkRow{m[1], {crosslinkUpper(m[2]), std::stoi(m[3])}, {crosslinkUpper(m[4]), std::stoi(m[5])}};
}

}  // namespace lf
