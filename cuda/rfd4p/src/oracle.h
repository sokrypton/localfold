#pragma once
/**
 * Upstream's refolding oracle (rfproteina/metrics/oracles), with LocalFold's own AlphaFold 3
 * (cuda/af3/localfold-af3) where upstream runs DeepMind's in a container: each design that passes
 * the pre-oracle filters is refolded from its sequence alone and scored - the oracle's confidences,
 * the design-against-refold RMSDs (`all_refolding_metrics.yaml`) and the pass criteria
 * (`all_pass_criteria.yaml`), under the job's `metrics_tags`.
 *
 * Upstream builds a design's AF3 input by writing it as mmCIF and reading that back with AF3's own
 * `Input.from_mmcif` (af3_oracle.py `_build_manifest`, af3_runtime.py `build_fold_input_from_cif`);
 * this writes the job that round trip produces, directly. tools/check-rfd4p-oracle.py holds it to
 * that round trip by featurising both with AlphaFold 3's own featuriser, and the scores to
 * upstream's oracle, refolding and pass-criteria managers run over the same refold.
 *
 * `oracle_apo` folds the design's chain of interest alone (`ApoOnlyTransform`), filed and scored
 * under its save_id `apo`. What is not ported is refused by name rather than approximated: a
 * covalently attached modifier (`_rename_covalent_modifiers_for_af3`), a non-standard polymer
 * residue and a chain id two chain instances share.
 */
#include "metrics.h"
#include "../featurise/cif.h"
#include "../../featurise/json.h"
#include <charconv>
#include <optional>
#include <sys/stat.h>
#include <tuple>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace rfd4p {
namespace oracle {
using metrics::Array;
using metrics::Gate;
using metrics::Values;

/**
 * One entry of `refolding_oracles/af3.yaml` or `af2.yaml`: its config key, `oracle_name`, template
 * query and tags, for AlphaFold 2 the `model_name` it folds with, and the save_id of its
 * pre-oracle transform where it has one (`apo`: `ApoOnlyTransform`, the chain of interest alone).
 */
struct Spec { std::string key, name, templateQuery; Gate gate; std::string model, saveId; };
inline const std::vector<Spec>& af3Specs() {
  static const std::vector<Spec> specs = {
      {"oracle", "af3", "", {{"af3_refolding", "unconditional_refolding", "motif_scaffold_refolding",
                               "internal_coordinate_motif_scaffold_refolding", "sm_binder_refolding", "ame_refolding",
                               "ame_no_ligand_refolding", "ame_original_benchmark_refolding", "disco_refolding"}, {}, {}}},
      {"oracle_ppi", "af3_ppi", "(sa_mask('C_CRD') | sa_mask('C_INTCRD')) & is_protein", {{"ppi_refolding", "antibody"}, {}, {}}},
      {"oracle_dna", "af3_dna", "sa_mask('C_CRD') & ~is_nucleic_acid", {{"dna_refolding"}, {}, {}}},
      {"oracle_apo", "af3_apo", "", {{"apo_refolding"}, {}, {}}, "", "apo"},
  };
  return specs;
}
inline const std::vector<Spec>& af2Specs() {
  static const std::vector<Spec> specs = {
      {"oracle", "af2", "", {{"af2_refolding", "unconditional_refolding", "motif_scaffold_refolding",
                               "internal_coordinate_motif_scaffold_refolding", "ame_no_ligand_refolding"}, {}, {}},
       "model_1_ptm"},
      {"oracle_ppi", "af2_ppi", "sa_mask('C_CRD')", {{"ppi_refolding", "antibody"}, {}, {}}, "model_1_multimer_v3"},
  };
  return specs;
}

[[noreturn]] inline void refuse(const std::string& why) { throw std::runtime_error(why); }

// ------------------------------------------------------------------------------------------------
// The design as the oracle reads it.

inline bool isNucleicResidue(const Array& a, int i) { return metrics::nucleicLink(metrics::linkType(a, i)); }
inline bool isDnaResidue(const Array& a, int i) { return metrics::linkType(a, i) == "DNA LINKING"; }
inline bool isStandardPolymerResidue(const std::string& r) {
  return metrics::isStandardAa(r) || r == "UNK" || r == "A" || r == "C" || r == "G" || r == "U" || r == "N" ||
         r == "DA" || r == "DC" || r == "DG" || r == "DT" || r == "DN";
}

/** `get_chain_of_interest`: the designed chain's instance id and whether it is unambiguous. */
inline std::pair<std::string, bool> chainOfInterestOf(const Array& a) {
  std::set<std::string> set(a.chainIid.begin(), a.chainIid.end());
  std::vector<std::string> chains(set.begin(), set.end());
  if (chains.size() < 2) return {chains.empty() ? std::string() : chains[0], true};
  std::vector<std::string> protein;
  for (const auto& ch : chains) {
    bool all = true;
    for (int i = 0; i < a.n; ++i) if (a.chainIid[i] == ch && !a.isProtein[i]) all = false;
    if (all) protein.push_back(ch);
  }
  if (protein.size() == 1) return {protein[0], true};
  if (protein.empty()) return {chains[0], false};
  auto mean = [&](const std::string& ch, auto test) {
    int total = 0, hit = 0;
    for (int i = 0; i < a.n; ++i) if (a.chainIid[i] == ch) { ++total; hit += test(i); }
    return (double)hit / total;
  };
  std::vector<double> seq;
  for (const auto& ch : protein) seq.push_back(mean(ch, [&](int i) { return a.cSeq[i] != metrics::MASKED; }));
  const double m = *std::min_element(seq.begin(), seq.end());
  std::vector<std::string> least;
  for (size_t k = 0; k < protein.size(); ++k) if (seq[k] == m) least.push_back(protein[k]);
  if (least.size() == 1) return {least[0], true};
  std::vector<double> crd;
  for (const auto& ch : least) {
    crd.push_back(mean(ch, [&](int i) { return (bool)a.crd[i]; }) + mean(ch, [&](int i) { return (bool)a.internal[i]; }));
  }
  const double mc = *std::min_element(crd.begin(), crd.end());
  std::vector<std::string> fewest;
  for (size_t k = 0; k < least.size(); ++k) if (crd[k] == mc) fewest.push_back(least[k]);
  return {fewest[0], fewest.size() == 1};
}

/**
 * The few selection strings upstream's oracle and refolding configs use, over the design's own
 * annotations (metrics.js names them: `crd` is C_CRD's mask, `internal` C_INTCRD's, `crdsrc`
 * SA_CRDSRC). A query this does not know is refused rather than guessed at.
 */
inline std::vector<uint8_t> mask(const Array& a, const std::string& query, const std::string& coi) {
  std::vector<uint8_t> m(a.n, 0);
  auto backbone = [&](int i) {
    const std::string& n = a.atomName[i];
    return a.isProtein[i] && (n == "N" || n == "CA" || n == "C" || n == "O");
  };
  auto ligand = [&](int i) { return !a.isPolymer[i] && a.resName[i] != "HOH"; };
  auto chainOfInterest = [&](int i) {
    if (coi.empty()) refuse("Cannot unambiguously determine chain of interest; use an explicit chain query.");
    return a.chainIid[i] == coi;
  };
  for (int i = 0; i < a.n; ++i) {
    bool v;
    if (query == "(sa_mask('C_CRD') | sa_mask('C_INTCRD')) & is_protein") v = (a.crd[i] || a.internal[i]) && a.isProtein[i];
    else if (query == "sa_mask('C_CRD') & ~is_nucleic_acid") v = a.crd[i] && !isNucleicResidue(a, i);
    else if (query == "sa_mask('C_CRD')") v = a.crd[i];
    else if (query == "is_protein_backbone") v = backbone(i);
    else if (query == "~is_chain_of_interest & (sa_mask('C_CRD') | is_dna)") v = !chainOfInterest(i) && (a.crd[i] || isDnaResidue(a, i));
    else if (query == "is_chain_of_interest & is_protein") v = chainOfInterest(i) && a.isProtein[i];
    else if (query == "~is_chain_of_interest & sa_mask('C_CRD') & is_protein_backbone") v = !chainOfInterest(i) && a.crd[i] && backbone(i);
    else if (query == "sa_mask('SA_CRDSRC') & is_protein") v = a.crdsrc[i] && a.isProtein[i];
    else if (query == "sa_mask('SA_CRDSRC') & is_protein & (atom_name == 'CA')") v = a.crdsrc[i] && a.isProtein[i] && a.atomName[i] == "CA";
    else if (query == "is_ligand") v = ligand(i);
    else if (query == "is_protein_backbone & ~sa_mask('C_CRD')") v = backbone(i) && !a.crd[i];
    else refuse("the oracle reads no selection '" + query + "'");
    m[i] = v;
  }
  return m;
}

/** An Array cut to `keep` (bonds among the kept atoms). */
inline Array subset(const Array& a, const std::vector<int>& keep) {
  Array o;
  o.n = (int)keep.size();
  std::map<int, int> at;
  for (size_t k = 0; k < keep.size(); ++k) at[keep[k]] = (int)k;
  auto pick = [&](auto& dst, const auto& src) {
    if (src.size() != (size_t)a.n) return;
    for (const int i : keep) dst.push_back(src[i]);
  };
  for (const int i : keep) for (int c = 0; c < 3; ++c) o.coord.push_back(a.coord[(size_t)i * 3 + c]);
  pick(o.resName, a.resName); pick(o.atomName, a.atomName); pick(o.element, a.element); pick(o.chainId, a.chainId);
  pick(o.insCode, a.insCode); pick(o.chainIid, a.chainIid); pick(o.compType, a.compType); pick(o.cSeq, a.cSeq);
  pick(o.resId, a.resId); pick(o.tokenId, a.tokenId); pick(o.cSec, a.cSec); pick(o.isPolymer, a.isPolymer);
  pick(o.isProtein, a.isProtein); pick(o.atomize, a.atomize); pick(o.crd, a.crd); pick(o.internal, a.internal);
  pick(o.hot, a.hot); pick(o.uii, a.uii); pick(o.hasInput, a.hasInput); pick(o.seqProb, a.seqProb);
  pick(o.charge, a.charge); pick(o.coordGroup, a.coordGroup); pick(o.molIid, a.molIid); pick(o.molEntity, a.molEntity);
  pick(o.condAt, a.condAt); pick(o.crdsrc, a.crdsrc); pick(o.grafted, a.grafted); pick(o.pnUnit, a.pnUnit);
  pick(o.crdValue, a.crdValue); pick(o.intCrd, a.intCrd);
  for (size_t k = 0; k < a.bonds.size(); ++k) {
    const auto p = at.find(a.bonds[k].first), q = at.find(a.bonds[k].second);
    if (p == at.end() || q == at.end()) continue;
    o.bonds.push_back({std::min(p->second, q->second), std::max(p->second, q->second)});
    o.bondType.push_back(a.bondType[k]);
  }
  return o;
}

/**
 * `remove_terminal_oxygen_atoms`: a non-atomized OXT of a multi-residue protein chain and OP3 of a
 * multi-residue nucleic chain go (the indices kept). Chains are instances (`chain_iid`).
 */
inline std::vector<int> withoutTerminalOxygens(const Array& a) {
  std::map<std::string, std::set<std::pair<int, std::string>>> residues;
  for (int i = 0; i < a.n; ++i) residues[a.chainIid[i]].insert({a.resId[i], a.insCode[i]});
  std::vector<int> keep;
  for (int i = 0; i < a.n; ++i) {
    const bool multi = residues[a.chainIid[i]].size() > 1;
    const bool oxt = a.atomName[i] == "OXT" && a.isProtein[i];
    const bool op3 = a.atomName[i] == "OP3" && isNucleicResidue(a, i);
    if (multi && (oxt || op3) && !a.atomize[i]) continue;
    keep.push_back(i);
  }
  return keep;
}

// ------------------------------------------------------------------------------------------------
// The AF3 job.

inline std::string jsonString(const std::string& s) {
  std::string o = "\"";
  for (const char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      case '\r': o += "\\r"; break;
      default:
        if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
  }
  return o + "\"";
}
/** A float32 as the shortest double that reads back as it (Python's repr of `float(np.float32)`). */
inline std::string reprFloat(float v) {
  char b[64];
  const auto r = std::to_chars(b, b + sizeof b, (double)v);
  return std::string(b, r.ptr);
}
inline std::string fixed3(double v) { char b[64]; snprintf(b, sizeof b, "%.3f", v); return b; }

inline bool finiteAt(const Array& a, int i) {
  return std::isfinite(a.coord[(size_t)i * 3]) && std::isfinite(a.coord[(size_t)i * 3 + 1]) &&
         std::isfinite(a.coord[(size_t)i * 3 + 2]);
}

/** Residue spans `[start, end)` of `atoms` (indices into `a`), by chain, number, insertion code and name. */
inline std::vector<std::pair<int, int>> residueSpans(const Array& a, const std::vector<int>& atoms) {
  std::vector<std::pair<int, int>> out;
  for (size_t k = 0; k < atoms.size(); ++k) {
    const int i = atoms[k];
    if (k == 0) { out.push_back({0, 1}); continue; }
    const int p = atoms[k - 1];
    if (a.chainId[i] != a.chainId[p] || a.resId[i] != a.resId[p] || a.insCode[i] != a.insCode[p] ||
        a.resName[i] != a.resName[p]) {
      out.push_back({(int)k, (int)k + 1});
    } else {
      out.back().second = (int)k + 1;
    }
  }
  return out;
}

/** A user CCD component for one residue's atoms, as `_build_user_ccd_string` writes it from the design. */
inline std::string userCcdBlock(const Array& a, const std::vector<int>& atoms) {
  const std::string& code = a.resName[atoms[0]];
  std::string type = a.compType[atoms[0]];
  if (type.empty() || type == ".") type = "NON-POLYMER";
  auto quote = [](const std::string& v) {
    return v.find_first_of(" '\"") == std::string::npos ? v : "'" + v + "'";
  };
  std::string s = "data_" + code + "\n#\n_chem_comp.id " + code + "\n_chem_comp.name " + code + "\n_chem_comp.type " +
                  quote(type) + "\n_chem_comp.formula ?\n_chem_comp.mon_nstd_parent_comp_id ?\n_chem_comp.pdbx_synonyms ?\n"
                  "_chem_comp.formula_weight ?\n#\nloop_\n_chem_comp_atom.comp_id\n_chem_comp_atom.atom_id\n"
                  "_chem_comp_atom.type_symbol\n_chem_comp_atom.charge\n_chem_comp_atom.pdbx_aromatic_flag\n"
                  "_chem_comp_atom.pdbx_leaving_atom_flag\n_chem_comp_atom.pdbx_model_Cartn_x_ideal\n"
                  "_chem_comp_atom.pdbx_model_Cartn_y_ideal\n_chem_comp_atom.pdbx_model_Cartn_z_ideal\n";
  std::map<int, int> at;
  for (size_t k = 0; k < atoms.size(); ++k) at[atoms[k]] = (int)k;
  for (const int i : atoms) {
    s += code + " " + quote(a.atomName[i]) + " " + a.element[i] + " " + std::to_string(a.charge[i]) + " N N " +
         reprFloat(a.coord[(size_t)i * 3]) + " " + reprFloat(a.coord[(size_t)i * 3 + 1]) + " " +
         reprFloat(a.coord[(size_t)i * 3 + 2]) + "\n";
  }
  s += "#\nloop_\n_chem_comp_bond.comp_id\n_chem_comp_bond.atom_id_1\n_chem_comp_bond.atom_id_2\n"
       "_chem_comp_bond.value_order\n_chem_comp_bond.pdbx_aromatic_flag\n";
  for (size_t k = 0; k < a.bonds.size(); ++k) {
    if (!at.count(a.bonds[k].first) || !at.count(a.bonds[k].second)) continue;
    const int t = a.bondType[k];
    static const char* order[] = {"SING", "SING", "DOUB", "TRIP", "QUAD", "SING", "DOUB", "TRIP", "SING", "AROM"};
    const bool aromatic = t >= 5 && t <= 7;
    s += code + " " + quote(a.atomName[a.bonds[k].first]) + " " + quote(a.atomName[a.bonds[k].second]) + " " +
         order[t >= 0 && t <= 9 ? t : 1] + " " + (aromatic || t == 9 ? "Y" : "N") + "\n";
  }
  return s + "#\n";
}

/** A template chain as mmCIF: the residues' atoms, numbered in the chain's own order. */
inline std::string templateCif(const Array& a, const std::vector<int>& chainAtoms, const std::string& chain,
                               const std::vector<std::pair<int, int>>& spans, const std::vector<double>& coord) {
  std::string s = "data_template_chain_" + chain + "\n#\n_pdbx_audit_revision_history.revision_date 1900-01-01\n#\n"
                  "loop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n"
                  "_entity_poly_seq.hetero\n";
  for (size_t r = 0; r < spans.size(); ++r) {
    s += "1 " + std::to_string(r + 1) + " " + a.resName[chainAtoms[spans[r].first]] + " n\n";
  }
  s += "#\n_entity_poly.entity_id 1\n_entity_poly.type 'polypeptide(L)'\n#\n_entity.id 1\n_entity.type polymer\n#\n"
       "_struct_asym.id " + chain + "\n_struct_asym.entity_id 1\n#\nloop_\n_atom_site.group_PDB\n_atom_site.id\n"
       "_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n"
       "_atom_site.label_asym_id\n_atom_site.label_entity_id\n_atom_site.label_seq_id\n_atom_site.pdbx_PDB_ins_code\n"
       "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n"
       "_atom_site.auth_seq_id\n_atom_site.auth_asym_id\n_atom_site.pdbx_PDB_model_num\n";
  int id = 1;
  for (size_t r = 0; r < spans.size(); ++r) {
    for (int k = spans[r].first; k < spans[r].second; ++k) {
      const int i = chainAtoms[k];
      const std::string name = a.atomName[i].find('\'') != std::string::npos ? "\"" + a.atomName[i] + "\"" : a.atomName[i];
      s += "ATOM " + std::to_string(id++) + " " + a.element[i] + " " + name + " . " + a.resName[i] + " " + chain + " 1 " +
           std::to_string(r + 1) + " ? " + fixed3(coord[(size_t)i * 3]) + " " + fixed3(coord[(size_t)i * 3 + 1]) +
           " " + fixed3(coord[(size_t)i * 3 + 2]) + " 1.00 0.00 " + std::to_string(r + 1) + " " + chain + " 1\n";
    }
  }
  return s + "#\n";
}

/** A code the Chemical Component Dictionary could name: 1-5 upper-case letters or digits. */
inline bool ccdShaped(const std::string& c) {
  if (c.empty() || c.size() > 5) return false;
  for (const char ch : c) if (!(std::isdigit((unsigned char)ch) || (ch >= 'A' && ch <= 'Z'))) return false;
  return true;
}

/** `input-components.cif` (featurise.cpp `componentDefinitions`) as `{code: its data_ block}`. */
inline std::map<std::string, std::string> readComponents(const std::string& path) {
  std::map<std::string, std::string> out;
  std::ifstream f(path);
  if (!f) return out;
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  std::vector<size_t> starts;
  for (size_t at = 0; at < text.size(); at = text.find('\n', at), at = at == std::string::npos ? text.size() : at + 1) {
    if (text.compare(at, 5, "data_") == 0) starts.push_back(at);
  }
  for (size_t k = 0; k < starts.size(); ++k) {
    const std::string block = text.substr(starts[k], (k + 1 < starts.size() ? starts[k + 1] : text.size()) - starts[k]);
    const size_t eol = block.find('\n');
    out.emplace(block.substr(5, eol - 5), block);
  }
  return out;
}

/** What the scoring needs again once the oracle has run: the design as it was handed over, and how its chains were named. */
struct Prepared {
  Array design;                                   // the design, terminal oxygens removed
  std::vector<std::string> foldChain;             // per atom of `design`, the chain the oracle is handed
  std::map<std::string, std::string> foldToIid;   // the oracle's chain -> the design's chain instance
  std::string job;                                // the job (alphafold3 dialect)
  // The job's chains in the order the oracle takes them (AF3's by id, AF2's by its assembly); our
  // folds name their output's chains by that position, where AlphaFold keeps the job's ids.
  std::vector<std::string> foldOrder;
  std::vector<int> foldLengths;                   // residues a chain, in that order (AF2)
};

/** `str_id_to_int_id`: an mmCIF chain id's place in AF3's chain order (A..Z, then AA, BA, ...). */
inline long chainOrderKey(const std::string& id) {
  long out = 0, scale = 1;
  for (const char c : id) {
    if (c < 'A' || c > 'Z') refuse("String ID must be upper case letters, got " + id + ".");
    out += (c - 'A' + 1) * scale;
    scale *= 26;
  }
  return out;
}
/** The name our AF3 (cuda/af3/src/sampler.cuh `chainId`) gives the k-th chain of its output. */
inline std::string positionalChain(int k) {
  std::string id;
  for (int a = k; ; a = a / 26 - 1) { id.insert(id.begin(), (char)('A' + a % 26)); if (a < 26) break; }
  return id;
}

/**
 * `_build_manifest` + `Input.from_mmcif`: the AF3 job for one design under one spec. `coi` is the
 * chain of interest (an instance id) or "" where ambiguous; a template query names its residues.
 */
inline Prepared prepare(const Array& full, const Spec& spec, const std::string& name, uint32_t seed,
                        const std::map<std::string, std::string>& components) {
  Prepared p;
  p.design = subset(full, withoutTerminalOxygens(full));
  const Array& d = p.design;
  for (int i = 0; i < d.n; ++i) {
    if (d.resName[i] == "UNL") {
      refuse("Input structure for '" + name + "' contains residues named 'UNL'. AF3 treats 'UNL' as a reserved token "
             "and silently drops most ligand atoms.");
    }
  }
  const auto [coi, unambiguous] = chainOfInterestOf(d);
  const std::vector<uint8_t> templ = spec.templateQuery.empty() ? std::vector<uint8_t>(d.n, 0)
                                                                : mask(d, spec.templateQuery, unambiguous ? coi : "");
  // `_disambiguate_chain_ids`: a chain id two instances share would be renamed; every design
  // here names its chains uniquely, and the renaming is not ported.
  std::map<std::string, std::set<std::string>> iids;
  for (int i = 0; i < d.n; ++i) iids[d.chainId[i]].insert(d.chainIid[i]);
  for (const auto& [chain, set] : iids) {
    if (set.size() > 1) refuse("chain " + chain + " has several instances, whose renaming for AF3 is not ported");
    p.foldToIid[chain] = *set.begin();
  }
  p.foldChain = d.chainId;
  // `_rename_covalent_modifiers_for_af3`: a non-polymer bonded to a standard amino acid.
  for (size_t k = 0; k < d.bonds.size(); ++k) {
    const int i = d.bonds[k].first, j = d.bonds[k].second;
    for (const auto [poly, mod] : {std::make_pair(i, j), std::make_pair(j, i)}) {
      if (d.isPolymer[poly] && metrics::isStandardAa(d.resName[poly]) && !d.isPolymer[mod] &&
          !(d.chainId[poly] == d.chainId[mod] && d.resId[poly] == d.resId[mod])) {
        refuse("a covalently attached modifier (" + d.resName[mod] + " on " + d.resName[poly] +
               "), which the oracle would rename for AF3 - that renaming is not ported");
      }
    }
  }
  // NaN atoms are not handed over (AF3's reader refuses them).
  std::vector<int> valid;
  for (int i = 0; i < d.n; ++i) if (finiteAt(d, i)) valid.push_back(i);

  // 🔴 CHAINS IN AlphaFold 3's ORDER, BY ID, NOT THE ARRAY'S: AF3 sorts a job's chains
  // (`str_id_to_int_id`) and featurises them so, whatever order the job lists them in - a binder
  // designed as chain B beside a target A is A then B to the model. Our AF3 keeps the job's order
  // and puts every polymer before every ligand, so the job is written sorted, and a design whose
  // ligand sorts before a polymer - which our AF3 would reorder - is refused.
  std::vector<std::string> order;
  std::map<std::string, std::vector<int>> chainAtoms;
  for (const int i : valid) {
    if (!chainAtoms.count(d.chainId[i])) order.push_back(d.chainId[i]);
    chainAtoms[d.chainId[i]].push_back(i);
  }
  std::stable_sort(order.begin(), order.end(),
                   [](const std::string& x, const std::string& y) { return chainOrderKey(x) < chainOrderKey(y); });
  for (size_t k = 1; k < order.size(); ++k) {
    if (!d.isPolymer[chainAtoms[order[k - 1]][0]] && d.isPolymer[chainAtoms[order[k]][0]]) {
      refuse("ligand chain " + order[k - 1] + " sorts before polymer chain " + order[k] + ", which AlphaFold 3 keeps and "
             "our AF3 (polymers first) would reorder");
    }
  }
  p.foldOrder = order;
  std::string seqs;
  std::map<std::string, std::vector<int>> userComponents;   // first occurrence of each non-standard residue
  std::vector<std::string> userOrder;
  for (const std::string& chain : order) {
    const std::vector<int>& atoms = chainAtoms[chain];
    const auto spans = residueSpans(d, atoms);
    for (const auto& [s, e] : spans) {
      const std::string& r = d.resName[atoms[s]];
      if (!isStandardPolymerResidue(r) && !userComponents.count(r)) {
        userComponents[r] = std::vector<int>(atoms.begin() + s, atoms.begin() + e);
        userOrder.push_back(r);
      }
    }
    const int first = atoms[0];
    std::string entry;
    if (!d.isPolymer[first]) {
      std::string codes;
      for (const auto& [s, e] : spans) codes += (codes.empty() ? "" : ", ") + jsonString(d.resName[atoms[s]]);
      entry = "{\"ligand\": {\"id\": " + jsonString(chain) + ", \"ccdCodes\": [" + codes + "]}}";
    } else {
      const bool protein = d.isProtein[first];
      const bool dna = !protein && isDnaResidue(d, first);
      std::string sequence;
      for (const auto& [s, e] : spans) {
        const std::string& r = d.resName[atoms[s]];
        if (!isStandardPolymerResidue(r)) {
          refuse("a non-standard polymer residue (" + r + "), which AF3 would take as a modification - not ported");
        }
        if (protein) {
          static const char* letters = "ARNDCQEGHILKMFPSTWYV";
          char c = 'X';
          for (int k = 0; k < 20; ++k) if (r == metrics::STANDARD_AA[k]) c = letters[k];
          sequence += c;
        } else {
          sequence += r.size() == 2 ? r[1] : r[0];   // DA -> A, A -> A
        }
      }
      std::string templates = "[]";
      if (protein) {
        std::vector<int> query;
        for (size_t r = 0; r < spans.size(); ++r) {
          bool any = false;
          for (int k = spans[r].first; k < spans[r].second; ++k) any = any || templ[atoms[k]];
          if (any) query.push_back((int)r);
        }
        if (!query.empty()) {
          std::string idx;
          for (size_t k = 0; k < query.size(); ++k) idx += (k ? ", " : "") + std::to_string(query[k]);
          templates = "[{\"mmcif\": " + jsonString(templateCif(d, atoms, chain, spans, std::vector<double>(d.coord.begin(), d.coord.end()))) + ", \"queryIndices\": [" + idx +
                      "], \"templateIndices\": [" + idx + "]}]";
        }
      }
      if (protein) {
        entry = "{\"protein\": {\"id\": " + jsonString(chain) + ", \"sequence\": " + jsonString(sequence) +
                ", \"modifications\": [], \"unpairedMsa\": \"\", \"pairedMsa\": \"\", \"templates\": " + templates + "}}";
      } else if (dna) {
        entry = "{\"dna\": {\"id\": " + jsonString(chain) + ", \"sequence\": " + jsonString(sequence) +
                ", \"modifications\": []}}";
      } else {
        entry = "{\"rna\": {\"id\": " + jsonString(chain) + ", \"sequence\": " + jsonString(sequence) +
                ", \"modifications\": [], \"unpairedMsa\": \"\"}}";
      }
    }
    seqs += (seqs.empty() ? "" : ", ") + entry;
  }
  // `bondedAtomPairs`: the bonds between residues the mmCIF's struct_conn carries - every
  // inter-residue bond but a polymer's own backbone link.
  std::string bonds;
  std::set<int> isValid(valid.begin(), valid.end());
  for (size_t k = 0; k < d.bonds.size(); ++k) {
    const int i = d.bonds[k].first, j = d.bonds[k].second;
    if (!isValid.count(i) || !isValid.count(j)) continue;
    if (d.chainId[i] == d.chainId[j] && d.resId[i] == d.resId[j] && d.insCode[i] == d.insCode[j]) continue;
    const bool backbone = d.isPolymer[i] && d.isPolymer[j] && d.chainId[i] == d.chainId[j] &&
                          (std::set<std::string>{d.atomName[i], d.atomName[j]} == std::set<std::string>{"C", "N"} ||
                           std::set<std::string>{d.atomName[i], d.atomName[j]} == std::set<std::string>{"O3'", "P"});
    if (backbone) continue;
    auto end = [&](int x) {
      return "[" + jsonString(d.chainId[x]) + ", " + std::to_string(d.resId[x]) + ", " + jsonString(d.atomName[x]) + "]";
    };
    bonds += (bonds.empty() ? "" : ", ") + std::string("[") + end(i) + ", " + end(j) + "]";
  }
  // `_build_user_ccd_string`: every non-standard residue's definition as atomworks resolves it -
  // the input's own (its registry: an annotated input's `_chem_comp_atom`, hydrogens and all),
  // else the dictionary's, else the design's own atoms. A code shaped like a dictionary one is left
  // to AF3's own dictionary, which is the same entry.
  std::string ccd;
  for (const std::string& r : userOrder) {
    const auto it = components.find(r);
    if (it != components.end()) ccd += it->second;
    else if (!ccdShaped(r)) ccd += userCcdBlock(d, userComponents[r]);
  }
  p.job = "{\"dialect\": \"alphafold3\", \"version\": 4, \"name\": " + jsonString(name) + ", \"sequences\": [" + seqs +
          "], \"modelSeeds\": [" + std::to_string(seed) + "], \"bondedAtomPairs\": " +
          (bonds.empty() ? std::string("null") : "[" + bonds + "]") + ", \"userCCD\": " +
          (ccd.empty() ? std::string("null") : jsonString(ccd)) + "}";
  return p;
}

// ------------------------------------------------------------------------------------------------
// The fold, and what it says.

/**
 * numpy's float32 mean (`np.mean` of a float32 array): the pairwise sum numpy reduces with - eight
 * accumulators below its 128-element block, halves above - in float32, divided in float32.
 */
template <class T> inline T pairwiseSum(const T* a, size_t n) {
  if (n < 8) {
    T r = 0;   // numpy starts the short sum at -0.0, which adds as 0
    for (size_t i = 0; i < n; ++i) r += a[i];
    return r;
  }
  if (n <= 128) {
    T r[8];
    for (int k = 0; k < 8; ++k) r[k] = a[k];
    size_t i = 8;
    for (; i < n - (n % 8); i += 8) for (int k = 0; k < 8; ++k) r[k] += a[i + k];
    T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2);
}
inline float pairwiseSum32(const float* a, size_t n) { return pairwiseSum(a, n); }
inline float mean32(const std::vector<float>& v) { return pairwiseSum32(v.data(), v.size()) / (float)v.size(); }
/**
 * `np.mean` of the top-left `rows x rows` of a float32 array `stride` wide, held row-major in `m`:
 * contiguous (rows == stride) it is the one pairwise sum; a strided view numpy reduces through its
 * 8192-element buffer, a whole number of rows at a time, each buffer's pairwise sum added in turn.
 */
inline float meanCorner32(const std::vector<float>& m, size_t rows, size_t stride) {
  if (rows == stride) return mean32(m);
  const size_t step = (8192 / rows) * rows;
  if (step == 0) refuse("a " + std::to_string(rows) + "-residue PAE is wider than numpy's reduction buffer, which is not ported");
  float acc = 0.f;
  for (size_t s = 0; s < m.size(); s += step) acc += pairwiseSum32(m.data() + s, std::min(step, m.size() - s));
  return acc / (float)m.size();
}

/** Python's `round(x, 4)` (correctly rounded, as printf is). */
inline double round4(double v) { char b[64]; snprintf(b, sizeof b, "%.4f", v); return strtod(b, nullptr); }

inline std::string readText(const std::string& path) {
  std::ifstream f(path);
  if (!f) refuse("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/** One atom of the refold as AF3 wrote it. */
struct RefoldAtom { std::string chain, resName, atomName; int resId; double x, y, z, b; };

/** The refold's `_atom_site`, as atomworks' minimal reader takes it: label chain, label (else auth) number. */
inline std::vector<RefoldAtom> readRefold(const std::string& path) {
  const cif::Block block = cif::parseBlock(readText(path));
  const auto site = block.find("atom_site");
  if (site == block.end()) refuse(path + " has no _atom_site");
  auto column = [&](const char* k) -> const std::vector<std::string>& {
    const auto it = site->second.find(k);
    if (it == site->second.end()) refuse(path + " has no _atom_site." + k);
    return it->second;
  };
  const auto& chain = column("label_asym_id");
  const auto& comp = column("label_comp_id");
  const auto& atom = column("label_atom_id");
  const auto& seq = column("label_seq_id");
  const auto& auth = column("auth_seq_id");
  const auto& x = column("Cartn_x");
  const auto& y = column("Cartn_y");
  const auto& z = column("Cartn_z");
  const auto& b = column("B_iso_or_equiv");
  std::vector<RefoldAtom> out;
  for (size_t r = 0; r < chain.size(); ++r) {
    const std::string& n = seq[r] == "." || seq[r] == "?" ? auth[r] : seq[r];
    out.push_back({chain[r], comp[r], atom[r], (int)strtol(n.c_str(), nullptr, 10), strtod(x[r].c_str(), nullptr),
                   strtod(y[r].c_str(), nullptr), strtod(z[r].c_str(), nullptr), strtod(b[r].c_str(), nullptr)});
  }
  return out;
}

/**
 * An oracle's confidences in the AF3 shape upstream's `extract_confidence_metrics` reads: per chain
 * (in the fold's order, named as the job names them) and per chain pair, and the scalars.
 */
struct Summary {
  std::vector<std::string> chains;
  std::vector<double> chainPlddt, chainPae, chainPtm;
  std::vector<std::vector<double>> pairPae, pairPaeMin, pairIptm;
  std::vector<std::pair<std::string, double>> scalars;   // overall_plddt, overall_pae, ptm, iptm, ranking_score, ...
};

/**
 * `_augment_summary_confidences`: AF3's summary, its per-chain pLDDT and PAE recomputed from the full
 * confidences as upstream's worker does. `named` reads our AF3's chain names back as the job's.
 */
inline Summary af3Summary(const std::string& summaryPath, const std::string& confidencesPath,
                          const std::map<std::string, std::string>& named) {
  const lf::Json summary = lf::parseJson(readText(summaryPath));
  const lf::Json full = lf::parseJson(readText(confidencesPath));
  auto numbers = [](const lf::Json* j) {
    std::vector<double> v;
    if (j && j->isArray()) for (const auto& e : j->a) v.push_back(e.isNumber() ? e.n : NAN);
    return v;
  };
  auto strings = [&](const lf::Json* j) {
    std::vector<std::string> v;
    if (j && j->isArray()) for (const auto& e : j->a) v.push_back(named.count(e.s) ? named.at(e.s) : e.s);
    return v;
  };
  const std::vector<double> atomPlddt = numbers(full.get("atom_plddts"));
  const std::vector<std::string> atomChain = strings(full.get("atom_chain_ids"));
  const std::vector<std::string> tokenChain = strings(full.get("token_chain_ids"));
  const lf::Json* paeJ = full.get("pae");
  if (!paeJ || !paeJ->isArray()) refuse(confidencesPath + " has no pae");
  const size_t nt = tokenChain.size();
  std::vector<double> pae;
  for (const auto& row : paeJ->a) for (const auto& v : row.a) pae.push_back(v.n);
  if (pae.size() != nt * nt) refuse(confidencesPath + ": the pae is not tokens x tokens");
  Summary s;
  for (const auto& c : tokenChain) if (std::find(s.chains.begin(), s.chains.end(), c) == s.chains.end()) s.chains.push_back(c);
  const size_t nc = s.chains.size();
  s.pairPae.assign(nc, std::vector<double>(nc));
  auto tokens = [&](const std::string& c) { std::vector<size_t> t; for (size_t i = 0; i < nt; ++i) if (tokenChain[i] == c) t.push_back(i); return t; };
  for (size_t a = 0; a < nc; ++a) {
    std::vector<float> own;
    for (size_t k = 0; k < atomChain.size(); ++k) if (atomChain[k] == s.chains[a]) own.push_back((float)atomPlddt[k]);
    s.chainPlddt.push_back(own.empty() ? 0.0 : round4((double)mean32(own) / 100.0));
    const auto ti = tokens(s.chains[a]);
    for (size_t b = 0; b < nc; ++b) {
      const auto tj = tokens(s.chains[b]);
      std::vector<float> block;   // pae[np.ix_(ti, tj)], row-major
      for (const size_t i : ti) for (const size_t j : tj) block.push_back((float)pae[i * nt + j]);
      s.pairPae[a][b] = round4((double)mean32(block));
    }
    s.chainPae.push_back(s.pairPae[a][a]);
  }
  std::vector<float> plddt32(atomPlddt.begin(), atomPlddt.end()), pae32(pae.begin(), pae.end());
  s.scalars.push_back({"overall_plddt", round4((double)mean32(plddt32) / 100.0)});
  s.scalars.push_back({"overall_pae", round4((double)mean32(pae32))});
  for (const char* key : {"ptm", "iptm", "ranking_score", "fraction_disordered", "has_clash"}) {
    const lf::Json* v = summary.get(key);
    if (lf::Json::absent(v)) continue;
    s.scalars.push_back({key, v->kind == lf::Json::Bool ? (v->b ? 1.0 : 0.0) : v->n});
  }
  auto matrix = [&](const char* key) {
    std::vector<std::vector<double>> out;
    const lf::Json* v = summary.get(key);
    if (lf::Json::absent(v)) return out;
    for (const auto& row : v->a) { std::vector<double> r; for (const auto& e : row.a) r.push_back(e.isNumber() ? e.n : NAN); out.push_back(r); }
    return out;
  };
  s.pairPaeMin = matrix("chain_pair_pae_min");
  s.pairIptm = matrix("chain_pair_iptm");
  s.chainPtm = numbers(summary.get("chain_ptm"));
  return s;
}

/**
 * `extract_confidence_metrics`: a summary translated into the flat keys the pass criteria read, the
 * chains named by their instance (`toIid`), `coi` the chain of interest's or "".
 */
inline Values extractConfidenceMetrics(const Summary& s, const std::map<std::string, std::string>& toIid,
                                       const std::string& coi) {
  const size_t nc = s.chains.size();
  std::vector<std::string> inOrder;
  for (const auto& c : s.chains) {
    const auto it = toIid.find(c);
    inOrder.push_back(it == toIid.end() ? std::string() : it->second);
  }
  Values m;
  auto scalar = [&](const char* key) -> std::optional<double> {
    for (const auto& [k, v] : s.scalars) if (k == key) return v;
    return std::nullopt;
  };
  for (const auto& [src, dst, scale] : std::vector<std::tuple<const char*, const char*, double>>{
           {"overall_plddt", "overall_plddt", 100.0}, {"overall_pae", "overall_pae", 1.0}, {"ptm", "overall_ptm", 100.0},
           {"iptm", "overall_iptm", 100.0}, {"ranking_score", "ranking_score", 1.0},
           {"fraction_disordered", "fraction_disordered", 1.0}, {"has_clash", "has_clash", 1.0}}) {
    if (const auto v = scalar(src)) m.push_back({dst, *v * scale});
  }
  for (const auto& [mat, prefix, scale] : std::vector<std::tuple<const std::vector<std::vector<double>>*, const char*, double>>{
           {&s.pairPaeMin, "min_ipae", 1.0}, {&s.pairIptm, "iptm", 100.0}}) {
    if (mat->empty()) continue;
    for (size_t i = 0; i < nc; ++i) for (size_t j = i + 1; j < nc; ++j) {
      if (inOrder[i].empty() || inOrder[j].empty()) continue;
      std::vector<double> vals;
      for (const double v : {(*mat)[i][j], (*mat)[j][i]}) if (!std::isnan(v)) vals.push_back(v);
      if (vals.empty()) continue;
      double mean = 0;
      for (const double v : vals) mean += v;
      m.push_back({std::string(prefix) + "_" + inOrder[i] + "_" + inOrder[j], mean / vals.size() * scale});
    }
  }
  const auto at = std::find(inOrder.begin(), inOrder.end(), coi);
  if (coi.empty() || at == inOrder.end()) return m;
  const size_t c = (size_t)(at - inOrder.begin());
  m.push_back({"plddt_coi", s.chainPlddt[c] * 100.0});
  if (c < s.chainPtm.size() && !std::isnan(s.chainPtm[c])) m.push_back({"ptm_coi", s.chainPtm[c] * 100.0});
  m.push_back({"pae_coi", s.chainPae[c]});
  // `_coi_interface_vals`: forward and reverse averaged, either where the other is absent.
  auto interface = [&](const std::vector<std::vector<double>>& mat) {
    std::vector<double> out;
    for (size_t j = 0; j < nc; ++j) {
      if (j == c) continue;
      const double f = mat[c][j], r = mat[j][c];
      if (!std::isnan(f) && !std::isnan(r)) out.push_back((f + r) / 2.0);
      else if (!std::isnan(f)) out.push_back(f);
      else if (!std::isnan(r)) out.push_back(r);
    }
    return out;
  };
  const std::vector<std::tuple<const std::vector<std::vector<double>>*, const char*, bool, double>> reductions = {
      {&s.pairPae, "ipae_coi", false, 1.0}, {&s.pairPaeMin, "min_ipae_coi", true, 1.0}, {&s.pairIptm, "iptm_coi", false, 100.0}};
  for (const auto& [mat, name, minimum, scale] : reductions) {
    if (mat->empty()) continue;
    const std::vector<double> vals = interface(*mat);
    if (vals.empty()) continue;
    double r = 0;
    if (minimum) r = *std::min_element(vals.begin(), vals.end());
    else { for (const double v : vals) r += v; r /= vals.size(); }
    m.push_back({name, r * scale});
  }
  return m;
}

// ------------------------------------------------------------------------------------------------
// AlphaFold 2 (af2_oracle.py, af2/af2_runtime.py).

/** `nearest_bucket`: the residue count AF2's features are padded to. */
inline size_t af2Bucket(size_t residues) {
  for (const size_t b : {64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072}) if (residues <= b) return b;
  refuse("Sequence has " + std::to_string(residues) + " residues, exceeding the largest AF2 bucket (3072)");
}

/**
 * `AF2Oracle._build_manifest` + `build_raw_features_from_payload`: the job our AF2 folds for one
 * design under one spec - protein chains only, each its single sequence, in the order AF2's
 * multimer assembly takes them (`add_assembly_features`: chains grouped by sequence, the groups in
 * order of first appearance) - and a template of the spec's residues at their coordinates-with-
 * insertion (`_build_af2_template_dict`), the rest of the chain at the design's. `cwi` is per atom
 * of the design with terminal oxygens removed, or empty.
 */
inline Prepared prepareAf2(const Array& full, const Spec& spec, const std::string& name, const std::vector<double>& cwi) {
  Prepared p;
  p.design = subset(full, withoutTerminalOxygens(full));
  const Array& d = p.design;
  for (int i = 0; i < d.n; ++i) {
    if (d.resName[i] == "UNL") refuse("Input structure for '" + name + "' contains unsupported residue 'UNL'.");
  }
  std::set<std::string> other;
  for (int i = 0; i < d.n; ++i) if (!d.isProtein[i]) other.insert(d.resName[i]);
  if (!other.empty()) {
    std::string list;
    for (const std::string& r : other) list += (list.empty() ? "'" : " '") + r + "'";
    refuse("AF2Oracle only supports proteins; '" + name + "' contains non-protein residues: [" + list + "].");
  }
  const auto [coi, unambiguous] = chainOfInterestOf(d);
  const std::vector<uint8_t> templ = spec.templateQuery.empty() ? std::vector<uint8_t>(d.n, 0)
                                                                : mask(d, spec.templateQuery, unambiguous ? coi : "");
  std::map<std::string, std::set<std::string>> iids;
  for (int i = 0; i < d.n; ++i) iids[d.chainId[i]].insert(d.chainIid[i]);
  for (const auto& [chain, set] : iids) {
    if (set.size() > 1) refuse("chain " + chain + " has several instances, whose renaming for AF2 is not ported");
    p.foldToIid[chain] = *set.begin();
  }
  p.foldChain = d.chainId;
  std::vector<std::string> order;
  std::map<std::string, std::vector<int>> chainAtoms;
  for (int i = 0; i < d.n; ++i) {
    if (!finiteAt(d, i)) continue;   // what is handed over: the atoms with coordinates
    if (!chainAtoms.count(d.chainId[i])) order.push_back(d.chainId[i]);
    chainAtoms[d.chainId[i]].push_back(i);
  }
  if (order.empty()) refuse("AF2 input '" + name + "' has no atoms with finite coordinates.");
  const bool multimer = spec.model.find("multimer") != std::string::npos;
  if (!multimer && order.size() != 1) {
    refuse("AF2 model '" + spec.model + "' requires a monomer input, got " + std::to_string(order.size()) + " chains.");
  }
  if (multimer && order.size() < 2) {
    refuse("AF2 model '" + spec.model + "' requires a multimer input, got " + std::to_string(order.size()) + " chain.");
  }
  std::map<std::string, std::string> sequence;
  for (const std::string& chain : order) {
    for (const auto& [s, e] : residueSpans(d, chainAtoms[chain])) {
      const std::string& r = d.resName[chainAtoms[chain][s]];
      int k = 0;
      while (k < 20 && r != metrics::STANDARD_AA[k]) ++k;
      // `parse_protein_chains` would leave the residue out of the sequence, and the refold would no
      // longer line up with the design
      if (k == 20) refuse("a non-standard residue (" + r + "), which AF2's reader leaves out of the sequence");
      sequence[chain] += "ARNDCQEGHILKMFPSTWYV"[k];
    }
  }
  // `add_assembly_features`: identical sequences one entity, the entities in order of first appearance.
  std::vector<std::string> entities;
  for (const std::string& chain : order) {
    if (std::find(entities.begin(), entities.end(), sequence[chain]) == entities.end()) entities.push_back(sequence[chain]);
  }
  for (const std::string& e : entities) for (const std::string& chain : order) if (sequence[chain] == e) p.foldOrder.push_back(chain);
  std::string seqs;
  for (const std::string& chain : p.foldOrder) {
    const std::vector<int>& atoms = chainAtoms[chain];
    const auto spans = residueSpans(d, atoms);
    p.foldLengths.push_back((int)spans.size());
    std::vector<int> query;
    for (size_t r = 0; r < spans.size(); ++r) {
      bool any = false;
      for (int k = spans[r].first; k < spans[r].second; ++k) any = any || templ[atoms[k]];
      if (any) query.push_back((int)r);
    }
    std::string templates = "[]";
    if (!query.empty()) {
      std::vector<double> coord(d.coord.begin(), d.coord.end());
      for (const int i : atoms) {
        if (!templ[i] || cwi.empty()) continue;
        const double* c = &cwi[(size_t)i * 3];
        if (std::isfinite(c[0]) && std::isfinite(c[1]) && std::isfinite(c[2])) std::copy(c, c + 3, coord.begin() + (size_t)i * 3);
      }
      std::string idx;
      for (size_t k = 0; k < query.size(); ++k) idx += (k ? ", " : "") + std::to_string(query[k]);
      templates = "[{\"mmcif\": " + jsonString(templateCif(d, atoms, chain, spans, coord)) + ", \"queryIndices\": [" + idx +
                  "], \"templateIndices\": [" + idx + "]}]";
    }
    seqs += (seqs.empty() ? "" : ", ") + std::string("{\"protein\": {\"id\": ") + jsonString(chain) + ", \"sequence\": " +
            jsonString(sequence[chain]) + ", \"modifications\": [], \"unpairedMsa\": \"\", \"pairedMsa\": \"\", \"templates\": " +
            templates + "}}";
  }
  p.job = "{\"dialect\": \"alphafold3\", \"version\": 4, \"name\": " + jsonString(name) + ", \"sequences\": [" + seqs +
          "], \"modelSeeds\": [0]}";
  return p;
}

/** numpy's `np.round(x, 4)`: scaled, rounded half to even, scaled back - in the array's own precision. */
inline double npRound4(double v) { return std::nearbyint(v * 10000.0) / 10000.0; }
inline double npRound4f(float v) { return (double)(std::nearbyintf(v * 10000.0f) / 10000.0f); }

/**
 * DeepMind's `confidence.predicted_tm_score` over the residues `idx` of an `n x n x 64` PAE logit
 * array, as numpy computes it: the softmax in float32, the TM terms and their weighting in float64,
 * the alignment with the best score. `interface` weighs only the pairs across `asym`.
 */
inline double predictedTmScore(const std::vector<float>& logits, size_t n, const std::vector<int>& idx,
                               const std::vector<int>* asym) {
  const size_t m = idx.size();
  const int clipped = std::max((int)m, 19);
  const double d0 = 1.24 * std::pow((double)(clipped - 15), 1.0 / 3) - 1.8;
  double tmPerBin[64];
  for (int k = 0; k < 64; ++k) {
    const float c = k < 63 ? 0.25f + 0.5f * (float)k : 31.75f;   // `_calculate_bin_centers` of linspace(0, 31, 63)
    tmPerBin[k] = 1.0 / (1 + (double)(c * c) / (d0 * d0));
  }
  std::vector<double> term(m), weighted(m);
  double best = -1;
  size_t at = 0;
  std::vector<double> perAlignment(m);
  for (size_t r = 0; r < m; ++r) {
    size_t count = 0;
    for (size_t c = 0; c < m; ++c) count += !asym || (*asym)[r] != (*asym)[c];
    const double normed = 1.0 / (1e-8 + (double)count);
    for (size_t c = 0; c < m; ++c) {
      const bool pair = !asym || (*asym)[r] != (*asym)[c];
      const float* x = &logits[((size_t)idx[r] * n + (size_t)idx[c]) * 64];
      float top = x[0];
      for (int k = 1; k < 64; ++k) top = std::max(top, x[k]);
      float e[64];
      for (int k = 0; k < 64; ++k) e[k] = std::exp(x[k] - top);
      const float sum = pairwiseSum(e, 64);
      double t[64];
      for (int k = 0; k < 64; ++k) t[k] = (double)(e[k] / sum) * tmPerBin[k];
      term[c] = pairwiseSum(t, 64) * (pair ? 1.0 : 0.0);
      weighted[c] = term[c] * (pair ? normed : 0.0);
    }
    perAlignment[r] = pairwiseSum(weighted.data(), m);
    if (perAlignment[r] > best) { best = perAlignment[r]; at = r; }
  }
  return perAlignment[at];
}

/**
 * `_chain_summary`: our AF2's raw confidences (`--raw-confidences`: per-residue pLDDT, the PAE, pTM
 * and ipTM, and the PAE head's logits) summarised as upstream's AF2 worker summarises the model's,
 * written beside them as its `<stem>_summary_confidences.json`. `chains` and `lengths` are the
 * fold's order.
 */
inline Summary af2Summary(const std::string& base, const std::vector<std::string>& chains, const std::vector<int>& lengths,
                          const std::string& model) {
  const lf::Json raw = lf::parseJson(readText(base + "_model_raw_confidences.json"));
  std::vector<float> plddt, pae;
  for (const auto& v : raw.get("plddt")->a) plddt.push_back((float)v.n);
  for (const auto& v : raw.get("pae")->a) pae.push_back((float)v.n);
  const size_t n = plddt.size();
  if (pae.size() != n * n) refuse(base + "_model_raw_confidences.json: the pae is not residues x residues");
  size_t total = 0;
  for (const int l : lengths) total += (size_t)l;
  if (total != n) refuse("AF2 folded " + std::to_string(n) + " residues of the job's " + std::to_string(total));
  std::vector<float> logits(n * n * 64);
  {
    FILE* f = fopen((base + "_model_pae_logits.f32").c_str(), "rb");
    if (!f || fread(logits.data(), 4, logits.size(), f) != logits.size()) refuse("cannot read " + base + "_model_pae_logits.f32");
    fclose(f);
  }
  const double ptm = raw.get("ptm")->n;
  const lf::Json* iptmJ = raw.get("iptm");
  const double iptm = lf::Json::absent(iptmJ) ? 0.0 : iptmJ->n;   // `prediction_result.get("iptm", 0.0)`
  std::vector<int> chainIndex;
  for (size_t c = 0; c < lengths.size(); ++c) chainIndex.insert(chainIndex.end(), (size_t)lengths[c], (int)c);
  auto residues = [&](size_t c) { std::vector<int> r; for (size_t i = 0; i < n; ++i) if (chainIndex[i] == (int)c) r.push_back((int)i); return r; };
  Summary s;
  s.chains = chains;
  const size_t nc = chains.size();
  s.pairPae.assign(nc, std::vector<double>(nc));
  s.pairPaeMin.assign(nc, std::vector<double>(nc));
  s.pairIptm.assign(nc, std::vector<double>(nc));
  std::vector<float> pairIptm32(nc * nc, 0.f);   // `chain_pair_iptm` is a float32 array
  for (size_t a = 0; a < nc; ++a) {
    const auto ra = residues(a);
    std::vector<float> own;
    for (const int i : ra) own.push_back(plddt[(size_t)i]);
    s.chainPlddt.push_back(npRound4((double)mean32(own) / 100.0));
    for (size_t b = 0; b < nc; ++b) {
      const auto rb = residues(b);
      std::vector<float> block;
      for (const int i : ra) for (const int j : rb) block.push_back(pae[(size_t)i * n + (size_t)j]);
      s.pairPae[a][b] = npRound4((double)mean32(block));
      s.pairPaeMin[a][b] = npRound4((double)*std::min_element(block.begin(), block.end()));
    }
    s.chainPae.push_back(s.pairPae[a][a]);
    const double own_ptm = predictedTmScore(logits, n, ra, nullptr);
    s.chainPtm.push_back(npRound4(own_ptm));
    for (size_t b = a; b < nc; ++b) {
      std::vector<int> both, asym;
      for (size_t i = 0; i < n; ++i) {
        if (chainIndex[i] == (int)a || chainIndex[i] == (int)b) { both.push_back((int)i); asym.push_back(chainIndex[i]); }
      }
      const double score = predictedTmScore(logits, n, both, a != b ? &asym : nullptr);
      pairIptm32[a * nc + b] = pairIptm32[b * nc + a] = (float)score;
    }
  }
  for (size_t a = 0; a < nc; ++a) for (size_t b = 0; b < nc; ++b) s.pairIptm[a][b] = npRound4f(pairIptm32[a * nc + b]);
  const size_t bucket = af2Bucket(n);
  const double overallPlddt = (double)mean32(plddt) / 100.0;
  const double ranking = nc > 1 ? 0.8 * iptm + 0.2 * ptm : overallPlddt;
  s.scalars = {{"overall_plddt", round4(overallPlddt)}, {"overall_pae", round4((double)meanCorner32(pae, n, bucket))},
               {"ptm", round4(ptm)}, {"iptm", round4(iptm)}, {"ranking_score", round4(ranking)}};
  // as upstream's raw summary artifact holds it
  auto list = [](const std::vector<double>& v) {
    std::string o = "[";
    for (size_t k = 0; k < v.size(); ++k) o += (k ? ", " : "") + jsNumber(v[k]);
    return o + "]";
  };
  auto matrix = [&](const std::vector<std::vector<double>>& m) {
    std::string o = "[";
    for (size_t k = 0; k < m.size(); ++k) o += (k ? ", " : "") + list(m[k]);
    return o + "]";
  };
  std::string j = "{";
  for (const auto& [k, v] : s.scalars) j += jsonString(k) + ": " + jsNumber(v) + ", ";
  std::string ids;
  for (size_t k = 0; k < nc; ++k) ids += (k ? ", " : "") + jsonString(chains[k]);
  j += "\"chain_ids\": [" + ids + "], \"chain_plddt\": " + list(s.chainPlddt) + ", \"chain_ptm\": " + list(s.chainPtm) +
       ", \"chain_pae\": " + list(s.chainPae) + ", \"chain_pair_pae\": " + matrix(s.pairPae) + ", \"chain_pair_pae_min\": " +
       matrix(s.pairPaeMin) + ", \"chain_pair_iptm\": " + matrix(s.pairIptm) + ", \"af2_model_name\": " + jsonString(model) + "}";
  std::ofstream(base + "_summary_confidences.json") << j << "\n";
  return s;
}

/** Our AF2's PDB as atoms; its chains, named by position, are the fold's `order` in turn. */
inline std::vector<RefoldAtom> readAf2Pdb(const std::string& path, const std::vector<std::string>& order) {
  std::ifstream f(path);
  if (!f) refuse("cannot read " + path);
  std::vector<RefoldAtom> atoms;
  std::vector<char> seen;
  std::string line;
  auto trim = [](std::string t) {
    t.erase(0, t.find_first_not_of(' '));
    t.erase(t.find_last_not_of(' ') + 1);
    return t;
  };
  while (std::getline(f, line)) {
    if (line.compare(0, 4, "ATOM") != 0) continue;
    const char chain = line[21];
    if (std::find(seen.begin(), seen.end(), chain) == seen.end()) seen.push_back(chain);
    const size_t k = (size_t)(std::find(seen.begin(), seen.end(), chain) - seen.begin());
    if (k >= order.size()) refuse(path + " has more chains than the job");
    atoms.push_back({order[k], trim(line.substr(17, 3)), trim(line.substr(12, 4)), atoi(line.substr(22, 4).c_str()),
                     strtod(line.substr(30, 8).c_str(), nullptr), strtod(line.substr(38, 8).c_str(), nullptr),
                     strtod(line.substr(46, 8).c_str(), nullptr), strtod(line.substr(60, 6).c_str(), nullptr)});
  }
  if (seen.size() != order.size()) refuse(path + " has " + std::to_string(seen.size()) + " chains, the job " + std::to_string(order.size()));
  return atoms;
}

/**
 * `_finalize_af3_result`'s coordinate transfer: the refold's atoms onto the design's, residue by
 * residue and name by name, after AF3's added OXTs and filled-in side chains are trimmed away. A
 * residue the design has no coordinates for stays NaN. Throws where upstream's transfer fails.
 */
inline std::vector<double> transfer(const Prepared& p, const std::vector<RefoldAtom>& refold, std::vector<double>& plddt) {
  const Array& d = p.design;
  // Residues of the design all of whose atoms are NaN are not handed over and stay NaN.
  std::vector<uint8_t> missing(d.n, 0);
  {
    std::vector<int> all(d.n);
    for (int i = 0; i < d.n; ++i) all[i] = i;
    for (const auto& [s, e] : residueSpans(d, all)) {
      bool every = true;
      for (int k = s; k < e; ++k) every = every && !finiteAt(d, all[k]);
      if (every) for (int k = s; k < e; ++k) missing[all[k]] = 1;
    }
  }
  std::vector<double> coord(d.coord.begin(), d.coord.end());
  plddt.assign(d.n, 0.0);
  std::vector<std::string> chains;
  for (int i = 0; i < d.n; ++i) if (!missing[i] && std::find(chains.begin(), chains.end(), p.foldChain[i]) == chains.end()) chains.push_back(p.foldChain[i]);
  for (const std::string& chain : chains) {
    std::vector<int> inp;
    for (int i = 0; i < d.n; ++i) if (!missing[i] && p.foldChain[i] == chain) inp.push_back(i);
    std::vector<int> out;
    for (size_t k = 0; k < refold.size(); ++k) if (refold[k].chain == chain) out.push_back((int)k);
    // Residues of each: the design's by its own grouping, AF3's by chain, number and name.
    const auto inSpans = residueSpans(d, inp);
    std::vector<std::pair<int, int>> outSpans;
    for (size_t k = 0; k < out.size(); ++k) {
      const RefoldAtom& a = refold[out[k]];
      if (k && a.resId == refold[out[k - 1]].resId && a.resName == refold[out[k - 1]].resName) outSpans.back().second = (int)k + 1;
      else outSpans.push_back({(int)k, (int)k + 1});
    }
    if (inSpans.size() != outSpans.size()) {
      refuse("Chain '" + chain + "': " + std::to_string(inSpans.size()) + " residues in input vs " +
             std::to_string(outSpans.size()) + " in AF3 output");
    }
    for (size_t r = 0; r < inSpans.size(); ++r) {
      const std::string& resName = d.resName[inp[inSpans[r].first]];
      if (refold[out[outSpans[r].first]].resName != resName) refuse("Chain '" + chain + "': residue sequence mismatch");
      std::set<std::string> inNames;
      bool inOxt = false;
      for (int k = inSpans[r].first; k < inSpans[r].second; ++k) {
        inNames.insert(d.atomName[inp[k]]);
        inOxt = inOxt || d.atomName[inp[k]] == "OXT";
      }
      // An OXT AF3 added where the design has none goes; then, for a standard polymer residue,
      // the atoms AF3 filled in that the design never had.
      std::vector<int> kept;
      for (int k = outSpans[r].first; k < outSpans[r].second; ++k) {
        if (refold[out[k]].atomName == "OXT" && !inOxt) continue;
        kept.push_back(out[k]);
      }
      const int nIn = inSpans[r].second - inSpans[r].first;
      if ((int)kept.size() > nIn && isStandardPolymerResidue(resName)) {
        std::vector<int> trimmed;
        for (const int k : kept) if (inNames.count(refold[k].atomName)) trimmed.push_back(k);
        if ((int)(kept.size() - trimmed.size()) == (int)kept.size() - nIn) kept = trimmed;
      }
      if ((int)kept.size() != nIn) {
        refuse("Chain '" + chain + "', residue " + std::to_string(r) + " (" + resName + "): " + std::to_string(nIn) +
               " atoms in input vs " + std::to_string(kept.size()) + " in AF3");
      }
      std::map<std::string, int> byName;
      for (const int k : kept) {
        if (byName.count(refold[k].atomName)) refuse("Chain '" + chain + "', residue " + std::to_string(r) + ": duplicate atom name '" + refold[k].atomName + "' in AF3 output");
        byName[refold[k].atomName] = k;
      }
      for (int k = inSpans[r].first; k < inSpans[r].second; ++k) {
        const int i = inp[k];
        const auto it = byName.find(d.atomName[i]);
        if (it == byName.end()) refuse("Chain '" + chain + "', residue " + std::to_string(r) + ": atom name '" + d.atomName[i] + "' not in AF3 output");
        const RefoldAtom& a = refold[it->second];
        coord[(size_t)i * 3] = a.x;
        coord[(size_t)i * 3 + 1] = a.y;
        coord[(size_t)i * 3 + 2] = a.z;
        plddt[i] = a.b;
      }
    }
  }
  return coord;
}

/**
 * `align_atom_array_to_input_frame`: the refold moved so its anchor atoms - the absolute
 * coordinate group, else the largest internal one - sit at their coordinates-with-insertion.
 * Without staged coordinates-with-insertion, or under three anchors, it stays where AF3 put it.
 */
inline void alignToInputFrame(const Array& d, const std::vector<double>& cwi, std::vector<double>& coord) {
  if (cwi.empty()) return;
  auto valid = [&](int i) {
    return std::isfinite(cwi[(size_t)i * 3]) && std::isfinite(cwi[(size_t)i * 3 + 1]) && std::isfinite(cwi[(size_t)i * 3 + 2]);
  };
  std::vector<int> anchor;
  for (int i = 0; i < d.n; ++i) if (d.coordGroup[i] == 0 && valid(i)) anchor.push_back(i);
  if (anchor.empty()) {
    std::map<int, int> sizes;
    for (int i = 0; i < d.n; ++i) if (metrics::internalGroup(d.coordGroup[i]) && valid(i)) ++sizes[d.coordGroup[i]];
    if (sizes.empty()) return;
    int best = sizes.begin()->first;
    for (const auto& [g, n] : sizes) if (n > sizes[best]) best = g;
    for (int i = 0; i < d.n; ++i) if (d.coordGroup[i] == best && valid(i)) anchor.push_back(i);
  }
  if (anchor.size() < 3) return;
  const metrics::FitMap fit = metrics::fitOnto(cwi, coord, anchor, anchor);
  for (int i = 0; i < d.n; ++i) {
    const auto v = fit(coord[(size_t)i * 3], coord[(size_t)i * 3 + 1], coord[(size_t)i * 3 + 2]);
    for (int c = 0; c < 3; ++c) coord[(size_t)i * 3 + c] = v[c];
  }
}

/** A refolded design as written (upstream's `{example_id}_{oracle_name}_refolded.cif`): the design's atoms, pLDDT as B. */
inline std::string refoldCif(const Array& d, const std::vector<double>& coord, const std::vector<double>& plddt) {
  std::string s = "data_refold\n#\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n"
                  "_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
                  "_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n"
                  "_atom_site.B_iso_or_equiv\n_atom_site.auth_seq_id\n_atom_site.auth_asym_id\n_atom_site.pdbx_PDB_model_num\n";
  for (int i = 0; i < d.n; ++i) {
    if (!std::isfinite(coord[(size_t)i * 3])) continue;
    const std::string name = d.atomName[i].find('\'') != std::string::npos ? "\"" + d.atomName[i] + "\"" : d.atomName[i];
    char b[32];
    snprintf(b, sizeof b, "%.2f", plddt[i]);
    s += std::string(d.isPolymer[i] ? "ATOM" : "HETATM") + " " + std::to_string(i + 1) + " " + d.element[i] + " " + name +
         " . " + d.resName[i] + " " + d.chainId[i] + " " + (d.isPolymer[i] ? std::to_string(d.resId[i]) : ".") + " ? " +
         fixed3(coord[(size_t)i * 3]) + " " + fixed3(coord[(size_t)i * 3 + 1]) + " " + fixed3(coord[(size_t)i * 3 + 2]) +
         " 1.00 " + b + " " + std::to_string(d.resId[i]) + " " + d.chainId[i] + " 1\n";
  }
  return s + "#\n";
}

/** A `StructureComparisonMetric` over the whole structure: symmetry resolved, then aligned and scored. */
inline Values comparison(const Array& d, const std::vector<double>& gtCoord, const std::vector<double>& predCoord,
                         const std::vector<uint8_t>& align, const std::vector<uint8_t>& subset) {
  std::vector<int> all(d.n);
  for (int i = 0; i < d.n; ++i) all[i] = i;
  const symmetry::View view = metrics::sliceView(d, all);
  std::vector<double> gt = symmetry::permuteGtSubunits(predCoord, gtCoord, view, align, subset, true);
  const std::vector<int> classes = metrics::classesOf(subset);
  gt = symmetry::permuteGtAtoms(predCoord, gt, view, classes.empty() ? nullptr : &classes, &align);
  const metrics::Comparison r = metrics::structureComparison(gt, predCoord, align, align, subset, subset);
  if (!r.ok) return {};
  return {{"rmsd", r.rmsd}};
}

/** One oracle's refold of one design (or of its transform's output): what the metrics read. */
struct Refold {
  const Spec* spec;
  Array design;                       // what was handed over: the design, or the chain of interest alone
  std::vector<double> cwi;            // its coordinates-with-insertion, where it has them
  Values confidences;                 // the oracle's own keys, unprefixed
  std::string chainOfInterest;
  std::vector<double> coord, plddt;   // per atom of `design`
  /** Its address by origin, `refold.{oracle_key}` or `refold.{oracle_key}.{save_id}`. */
  std::string label() const { return spec->saveId.empty() ? spec->key : spec->key + "." + spec->saveId; }
  /** Its kind, the alias `refold.untransformed` or `refold.{save_id}` where only it claims it. */
  std::string kind() const { return spec->saveId.empty() ? "untransformed" : spec->saveId; }
};

/**
 * `all_refolding_metrics.yaml` over one refold: what it was handed is the ground truth, the refold
 * the prediction; `tags` are the job's plus the refold addresses (`is_refold:...`,
 * `exists_refold:...`) and `oracle:<name>` run_oracle_and_refolding_metrics injects, and `refolds`
 * every refold of the design by those addresses, for a metric that compares two of them.
 */
inline Values refoldingMetrics(const Refold& r, const std::vector<std::string>& tags,
                               const std::map<std::string, const Refold*>& refolds) {
  const Array& d = r.design;
  const std::vector<double>& cwi = r.cwi;
  const auto [coi, unambiguous] = chainOfInterestOf(d);
  const std::string c = unambiguous ? coi : "";
  const std::vector<double> gt(d.coord.begin(), d.coord.end());
  Array refold = d;
  for (size_t k = 0; k < r.coord.size(); ++k) refold.coord[k] = (float)r.coord[k];
  const std::vector<double> pred(refold.coord.begin(), refold.coord.end());
  const std::vector<std::string> refoldingAny = {
      "unconditional_refolding", "ppi_refolding", "antibody", "dna_refolding", "motif_scaffold_refolding",
      "internal_coordinate_motif_scaffold_refolding", "sm_binder_refolding", "ame_refolding", "ame_no_ligand_refolding",
      "ame_original_benchmark_refolding", "disco_refolding"};
  struct Entry { const char* name; Gate gate; std::function<Values()> fn; };
  auto masked = [&](const char* align, const char* sub) {
    return [&, align, sub] { return comparison(d, gt, pred, mask(d, align, c), mask(d, sub, c)); };
  };
  const std::vector<Entry> entries = {
      {"refolded_backbone_rmsd", {refoldingAny, {"is_refold:untransformed"}, {}}, masked("is_protein_backbone", "is_protein_backbone")},
      {"refolded_target_aligned_binder_rmsd", {{"ppi_refolding", "dna_refolding", "antibody"}, {"is_refold:untransformed"}, {}},
       masked("~is_chain_of_interest & (sa_mask('C_CRD') | is_dna)", "is_chain_of_interest & is_protein")},
      {"refolded_target_backbone_aligned_binder_rmsd", {{"ppi_refolding"}, {"is_refold:untransformed"}, {}},
       masked("~is_chain_of_interest & sa_mask('C_CRD') & is_protein_backbone", "is_chain_of_interest & is_protein")},
      {"refolded_motif_allatom_rmsd",
       {{"motif_scaffold_refolding", "internal_coordinate_motif_scaffold_refolding", "ame_refolding",
         "ame_no_ligand_refolding", "ame_original_benchmark_refolding"}, {"is_refold:untransformed"}, {}},
       [&] {
         if (cwi.empty()) refuse("refolded_motif_allatom_rmsd reads the design's coordinates-with-insertion, which this design has none of");
         const auto sel = mask(d, "sa_mask('SA_CRDSRC') & is_protein", c);
         return metrics::motifRmsd(refold, cwi, [&](int i) { return sel[i] != 0; });
       }},
      {"refolded_motif_ca_rmsd", {{"motif_scaffold_refolding", "internal_coordinate_motif_scaffold_refolding"}, {"is_refold:untransformed"}, {}},
       [&] {
         if (cwi.empty()) refuse("refolded_motif_ca_rmsd reads the design's coordinates-with-insertion, which this design has none of");
         const auto sel = mask(d, "sa_mask('SA_CRDSRC') & is_protein & (atom_name == 'CA')", c);
         return metrics::motifRmsd(refold, cwi, [&](int i) { return sel[i] != 0; });
       }},
      {"refolded_backbone_aligned_ligand_rmsd", {{"sm_binder_refolding"}, {"is_refold:untransformed"}, {}},
       masked("is_protein_backbone", "is_ligand")},
      // Two refolds against each other: the apo one as the ground truth, the untransformed one as
      // the prediction, each selected over its own atoms. Where the apo refold is the whole design
      // the two have one atom set and symmetry is resolved; otherwise the selections are matched
      // in order, and a count that differs is no metric.
      {"refolded_apo_holo_rmsd", {{}, {"is_refold:apo", "exists_refold:untransformed"}, {}},
       [&]() -> Values {
         const Refold& apo = *refolds.at("apo");
         const Refold& holo = *refolds.at("untransformed");
         const char* q = "is_protein_backbone & ~sa_mask('C_CRD')";
         const std::vector<uint8_t> ga = mask(apo.design, q, ""), pa = mask(holo.design, q, "");
         if (apo.design.n == holo.design.n) return comparison(holo.design, apo.coord, holo.coord, ga, ga);
         const metrics::Comparison c = metrics::structureComparison(apo.coord, holo.coord, ga, pa, ga, pa);
         if (!c.ok) return {};
         return {{"rmsd", c.rmsd}};
       }},
  };
  if (std::find(tags.begin(), tags.end(), "disco_refolding") != tags.end()) {
    refuse("max_ligand_centroid_distance (disco_refolding) is not ported");
  }
  Values out;
  for (const Entry& e : entries) {
    if (!metrics::applies(e.gate, tags)) continue;
    for (const auto& [k, v] : e.fn()) out.push_back({std::string(e.name) + "." + k, v});
  }
  return out;
}

/** `all_pass_criteria.yaml` over the merged design, oracle and refolding metrics: `<name>.pass` / `.all_metrics_present`. */
inline Values passCriteria(const std::map<std::string, double>& m, const std::vector<std::string>& tags) {
  struct Bound { std::string key; double min = NAN, max = NAN, maxExclusive = NAN; };
  struct Criterion { const char* name; Gate gate; std::vector<Bound> bounds; };
  const std::vector<Criterion> criteria = {
      {"unconditional", {{"unconditional_refolding"}, {}, {}}, {{"refolded_backbone_rmsd.rmsd", NAN, 1.5}}},
      {"ppi", {{"ppi_refolding"}, {}, {}},
       {{"oracle_ppi.ptm_coi", 80.0}, {"oracle_ppi.min_ipae_coi", NAN, 1.5}, {"refolded_target_backbone_aligned_binder_rmsd.rmsd", NAN, 2.5}}},
      {"antibody", {{"antibody"}, {}, {}}, {{"oracle_ppi.min_ipae_coi", NAN, 2.0}, {"refolded_target_aligned_binder_rmsd.rmsd", NAN, 2.0}}},
      {"dna", {{"dna_refolding"}, {}, {}}, {{"oracle_dna.min_ipae_coi", NAN, 1.5}, {"refolded_target_aligned_binder_rmsd.rmsd", NAN, 8.0}}},
      {"motif_scaffold", {{"motif_scaffold_refolding"}, {}, {}},
       {{"refolded_backbone_rmsd.rmsd", NAN, 2.0}, {"refolded_motif_allatom_rmsd.rmsd", NAN, 2.0}, {"refolded_motif_ca_rmsd.rmsd", NAN, 1.0}}},
      {"internal_coords", {{"internal_coordinate_motif_scaffold_refolding"}, {}, {}},
       {{"refolded_backbone_rmsd.rmsd", NAN, 2.0}, {"refolded_motif_allatom_rmsd.rmsd", NAN, 2.0}, {"refolded_motif_ca_rmsd.rmsd", NAN, 1.0}}},
      {"sm_binder", {{"sm_binder_refolding"}, {}, {}},
       {{"oracle.min_ipae_coi", NAN, 1.5}, {"oracle.iptm_coi", 80.0}, {"refolded_backbone_rmsd.rmsd", NAN, 1.5},
        {"refolded_backbone_aligned_ligand_rmsd.rmsd", NAN, 5.0}}},
      {"ame", {{"ame_refolding"}, {}, {"no_ligand"}},
       {{"oracle.min_ipae_coi", NAN, NAN, 1.5}, {"refolded_backbone_rmsd.rmsd", NAN, 2.0}, {"refolded_motif_allatom_rmsd.rmsd", NAN, 1.5},
        {"clash_metrics.frac_atoms_clashing_inter_residue_polymer_ligand_1p5A", NAN, 0}}},
      {"ame_no_ligand", {{"ame_refolding"}, {"no_ligand"}, {}}, {{"refolded_backbone_rmsd.rmsd", NAN, 2.0}, {"refolded_motif_allatom_rmsd.rmsd", NAN, 1.5}}},
  };
  if (std::find(tags.begin(), tags.end(), "disco_refolding") != tags.end() ||
      std::find(tags.begin(), tags.end(), "developability") != tags.end()) {
    refuse("the disco and developability pass criteria read metrics that are not ported");
  }
  Values out;
  for (const Criterion& c : criteria) {
    if (!metrics::applies(c.gate, tags)) continue;
    bool passed = true, present = true;
    for (const Bound& b : c.bounds) {
      const auto it = m.find(b.key);
      if (it == m.end()) { present = false; passed = false; continue; }
      const double v = it->second;
      if (!std::isnan(b.min) && !(v >= b.min)) passed = false;
      if (!std::isnan(b.max) && !(v <= b.max)) passed = false;
      if (!std::isnan(b.maxExclusive) && !(v < b.maxExclusive)) passed = false;
    }
    out.push_back({std::string(c.name) + ".pass", passed ? 1.0 : 0.0});
    out.push_back({std::string(c.name) + ".all_metrics_present", present ? 1.0 : 0.0});
  }
  return out;
}

/** Where the oracle's files go, and the folding binary. */
struct Run {
  std::string oracle;    // "af3" (refolding_oracles/af3.yaml) or "af2" (af2.yaml)
  std::string binary;    // cuda/af3/localfold-af3 or cuda/af2/localfold-af2
  std::string dir;       // the design's oracle directory
  std::string name;      // the design's example id
  uint32_t seed = 0;
  std::map<std::string, std::string> components;   // the inputs' own component definitions
  bool keepRaw = false;  // keep AF2's PAE logits (residues^2 x 64 floats), which a checker reads again
};

/** The stem upstream files a refold under (`output_stem`): `{example_id}[_{save_id}]_{oracle_name}`. */
inline std::string stem(const Run& run, const Spec& spec) {
  return run.name + (spec.saveId.empty() ? "" : "_" + spec.saveId) + "_" + spec.name;
}

/** What a design's refolding scored: the metrics, the oracles' labels (`<key>.af2_model_name`), the chain of interest. */
struct Scores {
  Values values;
  std::vector<std::pair<std::string, std::string>> labels;
  std::string chainOfInterest;
};

/**
 * Refold one design with every oracle its tags admit and score it: the oracles' confidences
 * (`<key>.<metric>`), the refolding RMSDs and the pass criteria, merged as
 * `run_oracle_refolding_and_pass_criteria` returns them. `prior` is the design's own metrics.
 */
inline Scores refoldDesign(const Array& full, const std::vector<double>& fullCwi, const Run& run,
                           const std::vector<std::string>& tags, const Values& prior) {
  const bool af2 = run.oracle == "af2";
  if (!af2 && run.oracle != "af3") refuse("no refolding oracle '" + run.oracle + "' (af3 or af2)");
  std::vector<const Spec*> active;
  for (const Spec& s : af2 ? af2Specs() : af3Specs()) if (metrics::applies(s.gate, tags)) active.push_back(&s);
  Scores scores;
  Values& out = scores.values;
  if (active.empty()) return scores;
  mkdir(run.dir.c_str(), 0755);
  // The design as the oracle is handed it (`remove_terminal_oxygen_atoms`), its coordinates-with-insertion beside it.
  const std::vector<int> keep = withoutTerminalOxygens(full);
  const Array design = subset(full, keep);
  std::vector<double> cwi;
  if (!fullCwi.empty()) for (const int i : keep) for (int c = 0; c < 3; ++c) cwi.push_back(fullCwi[(size_t)i * 3 + c]);
  const auto [coi, unambiguous] = chainOfInterestOf(design);
  scores.chainOfInterest = unambiguous ? coi : "";
  std::vector<Refold> refolds;
  for (const Spec* spec : active) {
    // `ApoOnlyTransform`: the chain of interest alone, which the transform names (even when ambiguous).
    Array input = design;
    std::vector<double> inputCwi = cwi;
    std::string chainOfInterest = scores.chainOfInterest;
    if (spec->saveId == "apo") {
      if (coi.empty()) refuse("Could not determine a chain of interest for the input structure (no protein chains).");
      std::vector<int> kept;
      for (int i = 0; i < design.n; ++i) if (design.chainIid[i] == coi) kept.push_back(i);
      input = subset(design, kept);
      inputCwi.clear();
      if (!cwi.empty()) for (const int i : kept) for (int c = 0; c < 3; ++c) inputCwi.push_back(cwi[(size_t)i * 3 + c]);
      chainOfInterest = coi;
    } else if (!spec->saveId.empty()) {
      refuse("no pre-oracle transform with save_id '" + spec->saveId + "'");
    }
    const Prepared p = af2 ? prepareAf2(input, *spec, stem(run, *spec), inputCwi)
                           : prepare(input, *spec, stem(run, *spec), run.seed, run.components);
    const std::string base = run.dir + "/" + stem(run, *spec);
    std::ofstream(base + "_job.json") << p.job << "\n";
    if (spec->saveId.empty()) {
      const auto [c, u] = chainOfInterestOf(p.design);
      chainOfInterest = u ? c : "";
    }
    Refold r{spec, p.design, inputCwi, {}, chainOfInterest, {}, {}};
    std::vector<RefoldAtom> atoms;
    if (af2) {
      // `load_model_runners` + `predict_and_write`: three recycles, the multimer's early stop
      // (its config's 0.5 A), no BERT masking (`pct_seq_mask` 0), seed 0 (the payload's default),
      // and a chain without a template padded as DeepMind's pipeline pads it (`_pad_templates`).
      const bool multimer = spec->model.find("multimer") != std::string::npos;
      const std::string cmd = "'" + run.binary + "' --job='" + base + "_job.json' --model=" + spec->model + " --out='" + base +
                              "_model.pdb' --recycles=3 --tolerance=" + (multimer ? "0.5" : "0") +
                              " --msa-mask=0 --template-pad=zero --seed=0 --fast --raw-confidences > '" + base + "_af2.log' 2>&1";
      if (std::system(cmd.c_str()) != 0) refuse("AF2 failed on " + stem(run, *spec) + " - see " + base + "_af2.log");
      r.confidences = extractConfidenceMetrics(af2Summary(base, p.foldOrder, p.foldLengths, spec->model), p.foldToIid,
                                               r.chainOfInterest);
      if (!run.keepRaw) std::remove((base + "_model_pae_logits.f32").c_str());   // read, and no artifact
      atoms = readAf2Pdb(base + "_model.pdb", p.foldOrder);
      scores.labels.push_back({r.label() + ".af2_model_name", spec->model});
    } else {
      const std::string cmd = "'" + run.binary + "' --job='" + base + "_job.json' --out='" + base +
                              "_model.cif' --recycles=5 --steps=200 --seed=" + std::to_string(run.seed) + " > '" + base +
                              "_af3.log' 2>&1";
      if (std::system(cmd.c_str()) != 0) refuse("AF3 failed on " + stem(run, *spec) + " - see " + base + "_af3.log");
      // Our AF3 names its chains by position; the job's ids, in the order the job lists them, are what they are.
      std::map<std::string, std::string> named;
      for (size_t k = 0; k < p.foldOrder.size(); ++k) named[positionalChain((int)k)] = p.foldOrder[k];
      r.confidences = extractConfidenceMetrics(
          af3Summary(base + "_model_summary_confidences.json", base + "_model_confidences.json", named), p.foldToIid,
          r.chainOfInterest);
      atoms = readRefold(base + "_model.cif");
      for (RefoldAtom& a : atoms) {
        const auto it = named.find(a.chain);
        if (it == named.end()) refuse("AF3's output names a chain " + a.chain + " the job does not have");
        a.chain = it->second;
      }
    }
    r.coord = transfer(p, atoms, r.plddt);
    alignToInputFrame(p.design, inputCwi, r.coord);
    // As the refolded mmCIF stores it, which is what upstream's refolding metrics read back.
    for (double& v : r.coord) if (std::isfinite(v)) v = strtod(fixed3(v).c_str(), nullptr);
    std::ofstream(base + "_refolded.cif") << refoldCif(p.design, r.coord, r.plddt);
    for (const auto& [k, v] : r.confidences) out.push_back({r.label() + "." + k, v});
    refolds.push_back(std::move(r));
  }
  // Every refold by origin, and by kind where exactly one refold is of that kind.
  std::map<std::string, const Refold*> byAddress;
  std::map<std::string, int> ofKind;
  for (const Refold& r : refolds) { byAddress[r.label()] = &r; ++ofKind[r.kind()]; }
  for (const Refold& r : refolds) if (ofKind[r.kind()] == 1 && !byAddress.count(r.kind())) byAddress[r.kind()] = &r;
  std::map<std::string, double> merged;
  for (const auto& [k, v] : prior) merged[k] = v;
  for (const auto& [k, v] : out) merged[k] = v;
  for (const Refold& r : refolds) {
    std::vector<std::string> t = tags;
    for (const auto& [address, other] : byAddress) t.push_back("exists_refold:" + address);
    for (const auto& [address, other] : byAddress) if (other == &r) t.push_back("is_refold:" + address);
    t.push_back("oracle:" + r.spec->name);
    // A transformed refold's metrics carry its save_id after the metric's name.
    for (const auto& [metric, v] : refoldingMetrics(r, t, byAddress)) {
      std::string k = metric;
      if (!r.spec->saveId.empty()) { const size_t dot = k.find('.'); k = k.substr(0, dot) + "." + r.spec->saveId + k.substr(dot); }
      auto it = std::find_if(out.begin(), out.end(), [&](const auto& o) { return o.first == k; });
      if (it != out.end()) it->second = v; else out.push_back({k, v});
      merged[k] = v;
    }
  }
  for (const auto& kv : passCriteria(merged, tags)) out.push_back(kv);
  return scores;
}

/** The scores as `<design>.refold.json` holds them: NaN as the string "NaN", as the metrics' file does. */
inline std::string toJson(const Scores& scores) {
  const Values& v = scores.values;
  std::string s = "{\"metrics\":{";
  for (size_t k = 0; k < v.size(); ++k) {
    s += (k ? "," : "") + jsonString(v[k].first) + ":" + (std::isnan(v[k].second) ? std::string("\"NaN\"") : jsNumber(v[k].second));
  }
  size_t written = v.size();
  for (const auto& [k, label] : scores.labels) s += (written++ ? "," : "") + jsonString(k) + ":" + jsonString(label);
  const std::string& coi = scores.chainOfInterest;
  return s + "},\"chainOfInterest\":" + (coi.empty() ? std::string("null") : jsonString(coi)) + "}";
}

}  // namespace oracle
}  // namespace rfd4p
