// cuda/bioemu2/export_input.mjs, natively: BioEmu2's input for the CUDA port - shared/bioemu2/featurise.js's features
// (bioemu2_features.h, byte for byte the JavaScript's) in cuda/af3's model.idx/model.bin format, with pdb.template and
// confidence.template.json beside them. Every convention is documented in the JavaScript; this transcribes.
//
//   bioemu2-featurise <out dir> (--job=<AF3 job.json> | --sequence=<SEQ>[:<SEQ>...]) [--mode=structure|equilibrium]
//                     [--contacts=<contacts.json>] [--propagate-from=<pdb|cif>] [--lag-ns=1] [--propagation-steps=1]
//                     [--a3m=<merged.a3m> | --a3m=<a.a3m>,<b.a3m>,...] [--paired-a3m=...] [--ccd=<components.cif>]
//
// (--ccd is this side's alone: the worker's offline dictionary. The JavaScript reads the fixtures, then the cache.)
#pragma once
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "bioemu2_features.h"
#include "entries.h"
#include "fetch.h"
#include "jsmath.h"

namespace lf {

struct Bioemu2Export { Entries entries; std::string pdb, confidence, said; };

namespace bioemu2x {

// splitA3mByChains (shared/bioemu2/inputs.js)
inline std::vector<std::string> splitA3mByChains(const std::string& a3m, const std::vector<int>& lengths) {
  std::vector<std::vector<std::string>> out(lengths.size());
  std::string header;
  bool hasHeader = false;
  for (auto& raw : splitLines(a3m)) {
    std::string line = trimWs(raw);
    if (line.empty()) continue;
    if (line[0] == '>') { header = line; hasHeader = true; continue; }
    if (line[0] == '#') continue;
    std::vector<std::string> pieces(lengths.size());
    size_t chain = 0;
    int column = 0;
    for (char c : line) {
      bool insertion = c >= 'a' && c <= 'z';
      if (!insertion) {
        while (chain < lengths.size() - 1 && column >= lengths[chain]) { chain += 1; column = 0; }
        column += 1;
      }
      pieces[chain] += c;
    }
    for (size_t at = 0; at < pieces.size(); ++at) out[at].push_back((hasHeader ? header : ">row") + "\n" + pieces[at]);
  }
  std::vector<std::string> texts;
  for (auto& rows : out) {
    std::string t;
    for (size_t i = 0; i < rows.size(); ++i) t += (i ? "\n" : "") + rows[i];
    texts.push_back(t + "\n");
  }
  return texts;
}

// smilesTemplates (shared/bioemu2/inputs.js): this port's conformer, atoms named element-and-counter, centred
inline bioemu2::Template smilesTemplate(const std::string& smiles) {
  Component c = chem::smilesComponent(smiles, "LIG");
  static const std::vector<std::string> SYMBOLS = {"H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL", "SI", "P", "S",
    "CL", "AR", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR", "RB",
    "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I"};
  bioemu2::Template t;
  std::map<std::string, int> counts;
  for (auto& a : c.atoms) {
    std::string symbol = a.element >= 1 && a.element <= (int)SYMBOLS.size() ? SYMBOLS[a.element - 1] : "X";
    int k = ++counts[symbol];
    t.names.push_back(symbol + std::to_string(k));
    t.element.push_back(a.element);
    t.charge.push_back(a.charge);
  }
  double n = (double)c.atoms.size(), mean[3] = {0, 0, 0};
  for (auto& a : c.atoms) { mean[0] += a.x / n; mean[1] += a.y / n; mean[2] += a.z / n; }
  for (auto& a : c.atoms) {
    t.coords.push_back((float)(a.x - mean[0]));
    t.coords.push_back((float)(a.y - mean[1]));
    t.coords.push_back((float)(a.z - mean[2]));
  }
  for (auto& b : c.bonds) t.bonds.push_back({b.from, b.to});
  return t;
}

// Number(text) for a trimmed slice of a PDB column: "" is 0, anything not a number NaN
inline double jsNumberOf(const std::string& raw) {
  std::string t = trimWs(raw);
  if (t.empty()) return 0;
  char* end = nullptr;
  double v = std::strtod(t.c_str(), &end);
  return end && *end == '\0' ? v : NAN;
}

inline std::string keyOf(const std::string& chain, double residue, const std::string& name) {
  return chain + "|" + jsNumber(residue) + "|" + name;
}

// the starting frame, every atom by (chain, residue, name) from a PDB or an mmCIF, as export_input.mjs reads it
inline std::vector<float> propagationFrame(const std::string& path, const bioemu2::Features& F) {
  std::string text = readFile(path);
  std::map<std::string, std::array<double, 3>> where;
  bool cif = false;
  {
    bool data = false, site = false;
    for (auto& line : splitLines(text)) {
      if (line.compare(0, 5, "data_") == 0) data = true;
      if (line.find("_atom_site.") != std::string::npos) site = true;
    }
    cif = data && site;
  }
  if (cif) {
    std::vector<std::string> columns;
    for (auto& line : splitLines(text))
      if (line.compare(0, 11, "_atom_site.") == 0) {
        std::string c = line.substr(11);
        size_t sp = c.find_first_of(" \t");
        columns.push_back(sp == std::string::npos ? c : c.substr(0, sp));
      }
    auto col = [&](const std::string& name) {
      auto it = std::find(columns.begin(), columns.end(), name);
      return it == columns.end() ? -1 : (int)(it - columns.begin());
    };
    for (auto& line : splitLines(text)) {
      if (!(line.compare(0, 5, "ATOM ") == 0 || line.compare(0, 5, "ATOM\t") == 0 || line.compare(0, 7, "HETATM ") == 0
            || line.compare(0, 7, "HETATM\t") == 0)) continue;
      std::vector<std::string> v;
      std::string cur;
      for (char ch : trimWs(line)) {
        if (ch == ' ' || ch == '\t') { if (!cur.empty()) { v.push_back(cur); cur.clear(); } }
        else cur += ch;
      }
      if (!cur.empty()) v.push_back(cur);
      auto at = [&](int i) { return i >= 0 && i < (int)v.size() ? v[i] : std::string(); };
      std::string chain = at(col("auth_asym_id") >= 0 ? col("auth_asym_id") : col("label_asym_id"));
      std::string seq = col("label_seq_id") >= 0 && at(col("label_seq_id")) != "." ? at(col("label_seq_id")) : at(col("auth_seq_id"));
      std::string name = at(col("label_atom_id"));
      if (name.size() >= 2 && name.front() == '"') name = name.substr(1);
      if (!name.empty() && name.back() == '"') name.pop_back();
      where[keyOf(chain, jsNumberOf(seq), name)] = {jsNumberOf(at(col("Cartn_x"))), jsNumberOf(at(col("Cartn_y"))), jsNumberOf(at(col("Cartn_z")))};
    }
  } else {
    for (auto& line : splitLines(text)) {
      if (line.compare(0, 6, "ATOM  ") != 0 && line.compare(0, 6, "HETATM") != 0) continue;
      auto slice = [&](size_t a, size_t b) { return a >= line.size() ? std::string() : line.substr(a, b - a); };
      std::string chain = line.size() > 21 ? std::string(1, line[21]) : "undefined";
      where[keyOf(chain, jsNumberOf(slice(22, 26)), trimWs(slice(12, 16)))] =
          {jsNumberOf(slice(30, 38)), jsNumberOf(slice(38, 46)), jsNumberOf(slice(46, 54))};
    }
  }
  std::vector<float> flat;
  int missing = 0;
  std::string examples;
  for (auto& [c, r, n] : F.atomKeys) {
    auto it = where.find(keyOf(c, r, n));
    if (it == where.end()) {
      if (missing < 3) examples += std::string(examples.empty() ? "" : ",") + "[\"" + c + "\"," + std::to_string(r) + ",\"" + n + "\"]";
      ++missing;
      continue;
    }
    for (double v : it->second) flat.push_back((float)v);
  }
  if (missing)
    throw std::runtime_error(path + " lacks " + std::to_string(missing) + " atom(s) of this input, e.g. [" + examples + "]");
  return flat;
}

inline std::string jsonQuote(const std::string& s) {      // JSON.stringify of a string
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
    else if (c == '\n') o += "\\n";
    else if (c == '\r') o += "\\r";
    else if (c == '\t') o += "\\t";
    else if (c == '\b') o += "\\b";
    else if (c == '\f') o += "\\f";
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += (char)c;
  }
  return o + "\"";
}

inline std::string fixed3(double v) { char b[64]; snprintf(b, sizeof b, "%.3f", v); return b; }
inline std::string bxPadStart(const std::string& s, size_t n) { return s.size() >= n ? s : std::string(n - s.size(), ' ') + s; }
inline std::string bxPadEnd(const std::string& s, size_t n) { return s.size() >= n ? s : s + std::string(n - s.size(), ' '); }

}  // namespace bioemu2x

inline Bioemu2Export exportBioemu2(const Args& args) {
  using namespace bioemu2x;
  Bioemu2Export out;
  std::string jobPath = args.option("job"), sequence = args.option("sequence");
  if (args.positional.empty() || (jobPath.empty() && sequence.empty()))
    throw std::runtime_error("usage: bioemu2-featurise <out dir> --job=<job.json> | --sequence=<SEQ>[:<SEQ>...]"
                             " [--mode=structure|equilibrium] [--contacts=<json>] [--propagate-from=<pdb|cif>] [--lag-ns=1]"
                             " [--propagation-steps=1] [--a3m=...] [--paired-a3m=...]");
  Json json;
  if (!jobPath.empty()) json = parseJson(readFile(jobPath));
  else {
    std::string text = "{\"sequences\":[";
    size_t i = 0, at = 0;
    while (true) {
      size_t colon = sequence.find(':', at);
      std::string s = upper(sequence.substr(at, colon == std::string::npos ? std::string::npos : colon - at));
      text += std::string(i ? "," : "") + "{\"protein\":{\"id\":\"" + std::string(1, (char)(65 + i)) + "\",\"sequence\":" + jsonQuote(s) + "}}";
      ++i;
      if (colon == std::string::npos) break;
      at = colon + 1;
    }
    json = parseJson(text + "]}");
  }
  bioemu2::Job job = bioemu2::jobFromAf3Json(json);
  std::string mode = args.option("mode", "structure");
  if (mode != "structure" && mode != "equilibrium") throw std::runtime_error("--mode=" + mode + ": structure or equilibrium");

  // the components: the job's userCCD, the fixtures, a local dictionary (--ccd), else the per-component cache / RCSB
  std::map<std::string, std::string> user;
  if (const Json* u = json.get("userCCD"); u && u->isString()) {
    const std::string& text = u->s;
    std::vector<size_t> starts;
    for (size_t at = 0; at < text.size(); at = text.find('\n', at), at = at == std::string::npos ? text.size() : at + 1)
      if (text.compare(at, 5, "data_") == 0) starts.push_back(at);
    for (size_t i = 0; i < starts.size(); ++i) {
      std::string block = text.substr(starts[i], (i + 1 < starts.size() ? starts[i + 1] : text.size()) - starts[i]);
      std::string code = upper(stripQuotes(cifValue(block, "_chem_comp.id")));
      if (!code.empty()) user[code] = block;
    }
  }
  std::string fixtures = fetch::home() + "/tools/fixtures/ccd", dictionary = args.option("ccd");
  std::map<std::string, std::string> cache;
  bioemu2::CcdReader ccd = [user, fixtures, dictionary, cache](const std::string& raw) mutable {
    std::string code = upper(raw);
    auto u = user.find(code);
    if (u != user.end()) return u->second;
    auto c = cache.find(code);
    if (c != cache.end()) return c->second;
    std::string text, path = fixtures + "/" + code + ".cif";
    if (fetch::exists(path)) text = readFile(path);
    else text = dictionary.empty() ? ccdText(code) : ccdFileText(dictionary, code);
    cache[code] = text;
    return text;
  };

  std::map<std::string, bioemu2::Template> smiles;
  for (auto& chain : job.chains)
    if (chain.hasSmiles) smiles[chain.id] = smilesTemplate(chain.smiles);

  // the alignments, as the page hands them over
  std::vector<bioemu2::Chain*> proteins;
  for (auto& c : job.chains) if (c.kind == "protein") proteins.push_back(&c);
  auto alignmentsOf = [&](const std::string& arg, std::vector<std::string>& texts) {
    if (arg.empty()) return false;
    std::vector<std::string> paths = splitNonEmpty(arg, ',');
    if (paths.size() == 1) {
      std::vector<int> lengths;
      for (auto* c : proteins) lengths.push_back((int)c->sequence.size());
      texts = splitA3mByChains(readFile(paths[0]), lengths);
    } else {
      if (paths.size() != proteins.size())
        throw std::runtime_error(std::to_string(paths.size()) + " alignments for " + std::to_string(proteins.size())
                                 + " protein chain(s): name one a chain, or one merged");
      for (auto& p : paths) texts.push_back(readFile(p));
    }
    return true;
  };
  std::vector<std::string> unpaired, paired;
  bool haveUnpaired = alignmentsOf(args.option("a3m"), unpaired);
  bool havePaired = alignmentsOf(args.option("paired-a3m"), paired);
  if (haveUnpaired) {
    bool merged = splitNonEmpty(args.option("a3m"), ',').size() == 1;
    std::map<std::string, size_t> bySequence;
    for (size_t at = 0; at < proteins.size(); ++at) bySequence[upper(proteins[at]->sequence)] = at;
    for (size_t at = 0; at < proteins.size(); ++at) {
      size_t from = merged ? bySequence[upper(proteins[at]->sequence)] : at;
      proteins[at]->hasUnpaired = true;
      proteins[at]->unpairedMsa = unpaired[from];
      proteins[at]->hasPaired = havePaired;
      proteins[at]->pairedMsa = havePaired ? paired[from] : "";
    }
  }

  std::vector<bioemu2::Contact> contacts;
  std::string contactsPath = args.option("contacts");
  if (!contactsPath.empty()) contacts = bioemu2::parseContacts(parseJson(readFile(contactsPath)));
  bioemu2::Features F = bioemu2::featurise(job, ccd, mode, contactsPath.empty() ? nullptr : &contacts, smiles);
  const int T = F.tokens, N = F.atomsPadded;

  std::string propagateFrom = args.option("propagate-from");
  double propagationSteps = std::strtod(args.option("propagation-steps", "1").c_str(), nullptr);
  std::vector<float> x0;
  if (!propagateFrom.empty()) {
    std::vector<float> flat = propagationFrame(propagateFrom, F);
    std::vector<double> asDoubles(flat.begin(), flat.end());
    bioemu2::addPropagation(F, asDoubles, std::strtod(args.option("lag-ns", "1").c_str(), nullptr));
    x0.assign((size_t)N * 3, 0.f);
    std::copy(flat.begin(), flat.end(), x0.begin());
  }

  // the entries, in export_input.mjs's order
  Entries& E = out.entries;
  for (const char* name : {"res_type", "residue_index", "asym_id", "entity_id", "sym_id", "token_center_atom_index",
                           "token_pad_mask", "method_feature", "pocket_feature", "msa", "msa_paired", "has_deletion", "msa_mask",
                           "element", "name_encoding", "token_idx_at_atom", "ref_space_uid", "atom_pad_mask", "atom_mask"})
    E.i(name, F.ints.at(name));
  for (const char* name : {"deletion_mean", "profile", "token_bonds", "contact_threshold", "deletion_value",
                           "template_coords", "charge", "coords_cond"})
    E.t(name, F.floats.at(name));
  const auto& cc = F.ints.at("contact_conditioning");
  std::vector<int> cls((size_t)T * T, 0);
  for (size_t p = 0; p < (size_t)T * T; ++p) for (int k = 0; k < 5; ++k) if (cc[p * 5 + k]) cls[p] = k;
  E.i("contact_class", std::move(cls));
  if (!x0.empty()) E.t("x0", x0);
  const int S = F.msaRows;
  const auto& mask = F.ints.at("msa_mask");
  int nValid = 0;
  for (int s = 0; s < S; ++s) {
    bool any = false;
    for (int t = 0; t < T; ++t) any = any || mask[(size_t)s * T + t] != 0;
    nValid += any ? 1 : 0;
  }
  E.m("meta/tokens", T);
  E.m("meta/atoms", N);
  E.m("meta/real_atoms", F.atoms);
  E.m("meta/msa_rows", S);
  E.m("meta/n_valid_rows", std::max(nValid, 1));
  E.m("meta/mode", mode == "equilibrium" ? 1 : 0);
  E.m("meta/dt", F.dt);
  E.m("meta/is_dynamics_conditioned", F.isDynamicsConditioned ? 1 : 0);
  E.m("meta/propagation_steps", x0.empty() ? 0 : propagationSteps);

  // the structure's records and the output's tokens, in the job's chain order
  std::vector<std::string> chainOrder;
  for (auto& c : job.chains) chainOrder.push_back(c.id);
  std::map<std::string, std::vector<int>> atomsOfChain;
  for (size_t atom = 0; atom < F.atomKeys.size(); ++atom) atomsOfChain[std::get<0>(F.atomKeys[atom])].push_back((int)atom);
  static const std::string CHAIN_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const auto& symbols = elementSymbols();
  std::string pdb;
  std::vector<int> outTokens;
  std::vector<std::string> tokenChainIds;
  std::vector<int> tokenResIds;
  int serial = 1;
  for (size_t c = 0; c < chainOrder.size(); ++c) {
    const std::string& id = chainOrder[c];
    const bioemu2::Chain& chain = job.chains[c];
    bool polymer = chain.kind == "protein" || chain.kind == "rna" || chain.kind == "dna";
    std::set<int> modified;
    for (auto& m : chain.modifications) modified.insert(m.position);
    std::string letter = id.size() == 1 ? id : std::string(1, CHAIN_CHARACTERS[c % CHAIN_CHARACTERS.size()]);
    int lastToken = -1;
    for (int atom : atomsOfChain[id]) {
      int residue = std::get<1>(F.atomKeys[atom]);
      const std::string& name = std::get<2>(F.atomKeys[atom]);
      int token = F.tokenIdxAtAtom[atom];
      if (token != lastToken) { outTokens.push_back(token); tokenChainIds.push_back(id); tokenResIds.push_back(residue); lastToken = token; }
      bool het = !polymer || modified.count(residue);
      std::string resName = F.labels[atom];
      if (!het && (chain.kind == "dna" || chain.kind == "rna") && resName != "UNK") resName = bxPadStart(resName, 3);
      int z = F.element[atom];
      std::string element = z == 0 ? "X" : (z - 1 < (int)symbols.size() ? symbols[z - 1] : "X");
      std::string symbol = element.size() == 1 ? element : element.substr(0, 1) + [&] {
        std::string r = element.substr(1); for (auto& ch : r) ch = (char)std::tolower((unsigned char)ch); return r; }();
      std::string field = name.size() < 4 && symbol.size() == 1 ? bxPadEnd(" " + name, 4) : bxPadEnd(name, 4).substr(0, 4);
      pdb += std::string(het ? "HETATM" : "ATOM  ") + bxPadStart(std::to_string(serial % 100000), 5) + " " + field + " "
           + bxPadStart(resName.substr(0, 3), 3) + " " + letter + bxPadStart(std::to_string(residue), 4) + "    "
           + bxPadStart(fixed3(std::floor(atom / 1000.0)), 8) + bxPadStart(fixed3(atom % 1000), 8) + bxPadStart(fixed3(0), 8)
           + "  1.00  0.00          " + bxPadStart(upper(symbol), 2) + "\n";
      serial += 1;
    }
    pdb += "TER\n";
  }
  out.pdb = pdb + "END\n";
  std::string conf = "{\"tokens\":[";
  for (size_t i = 0; i < outTokens.size(); ++i) conf += (i ? "," : "") + std::to_string(outTokens[i]);
  conf += "],\"token_chain_ids\":[";
  for (size_t i = 0; i < tokenChainIds.size(); ++i) conf += (i ? "," : "") + jsonQuote(tokenChainIds[i]);
  conf += "],\"token_res_ids\":[";
  for (size_t i = 0; i < tokenResIds.size(); ++i) conf += (i ? "," : "") + std::to_string(tokenResIds[i]);
  out.confidence = conf + "]}";
  out.said = std::to_string(T) + " tokens, " + std::to_string(F.atoms) + " atoms (" + std::to_string(N) + " padded), "
           + std::to_string(S) + " MSA rows, " + mode
           + (x0.empty() ? "" : ", propagating " + jsNumber(propagationSteps) + " step(s)") + " -> " + args.positional;
  return out;
}

}  // namespace lf
