// cuda/esmfold2/export_input.mjs, natively: ESMFold2's input for the CUDA port - the page's own features
// (shared/esmfold2/featurise.js featuriseForEsmfold2 over AF3's featuriser with ESMFold2's conventions, its
// language-model input, representative-atoms.js, contacts.js' bin counts), the alignment block as biohub's esm builds
// it, and pdb.template - the page's PDB records with each atom's index where its coordinates go.
//
//   esmfold2-featurise <out dir> (--job=<AF3 job.json> | --sequence=<A:B> [--kinds= --ligands= --smiles= --modify=])
//                      [--a3m=<one a protein chain>] [--fold-bundle=<dir>]
#pragma once
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "af3_export.h"
#include "jsmath.h"
#include "pdb.h"

namespace lf {

struct Esmfold2Export { Entries entries; std::string pdb; std::vector<std::string> said; };

inline Esmfold2Export exportEsmfold2(const Args& args) {
  Esmfold2Export out;
  std::string sequence = upper(trimWs(args.option("sequence")));
  if (args.positional.empty() || (sequence.empty() && args.option("job").empty()))
    throw std::runtime_error("usage: esmfold2-featurise <out dir> --sequence=<SEQ>[:<SEQ>...] [--kinds=protein,dna,...] [--ligands=GOL,ATP]"
                             " [--smiles=OCC(O)CO|...] [--modify=SEP@3[@chain]] | --job=<AF3 job.json>");
  std::string kinds = args.option("kinds");
  ComponentSource components;
  Expanded request;
  bool haveJob = false;
  if (!args.option("job").empty()) {
    if (!sequence.empty()) throw std::runtime_error("--job and --sequence both name the input");
    Job job = jobFromJson(readFile(args.option("job")));
    for (auto& note : job.notes) out.said.push_back("job: " + note);
    request = expandEntities(job.entities);
    haveJob = true;
    sequence = request.sequence;
    kinds.clear();
    for (size_t i = 0; i < request.chainKinds.size(); ++i) kinds += (i ? "," : "") + request.chainKinds[i];
    if (!job.userCcd.empty()) components.addUserCcd(job.userCcd);
  }
  FeaturiseOptions o;
  if (haveJob) {
    for (auto& l : request.ligands) {
      if (l.kind == LigandRef::Code) o.ligands.push_back(components.get(l.code));
      else if (l.kind == LigandRef::Chain) {
        std::vector<Component> parts;
        for (auto& c : l.codes) parts.push_back(components.get(c));
        o.ligands.push_back(ligandChain(parts));
      } else o.ligands.push_back(chem::smilesComponent(l.smiles, l.code.empty() ? "LIG" : l.code));
    }
    for (auto& m : request.modifications) o.modifications.push_back({m.chain, (int)m.position, components.get(m.code)});
  }
  for (auto& code : splitNonEmpty(args.option("ligands"), ',')) o.ligands.push_back(components.get(code));
  {
    std::vector<std::string> seen;
    for (auto& s : splitNonEmpty(args.option("smiles"), '|')) {
      size_t at = std::find(seen.begin(), seen.end(), s) - seen.begin();
      if (at == seen.size()) seen.push_back(s);
      o.ligands.push_back(chem::smilesComponent(s, ligandName(at)));
    }
  }
  for (auto& spec : splitNonEmpty(args.option("modify"), ',')) {
    auto parts = splitOn(spec, '@');
    o.modifications.push_back({parts.size() > 2 ? (int)jsNumberOf(parts[2]) : 0, (int)jsNumberOf(parts.size() > 1 ? parts[1] : ""),
                               components.get(parts[0])});
  }
  if (!kinds.empty()) o.chainKinds = splitOn(kinds, ',');
  if (haveJob) o.bonds = request.bonds;
  // featuriseForEsmfold2: AF3's featuriser with no terminal atoms, symmetric bonds and an atomised residue bonded in
  o.terminalAtoms = 1;
  o.symmetriseBonds = true;
  o.atomizedBackboneBonds = true;
  o.hasUnpairedFrom = false;
  Batch b = featuriseProtein(sequence, o);
  const int T = b.tokens, dense = b.dense, CLASSES = 33, UNKNOWN = 22;
  const int MOL_PROTEIN = 0, MOL_DNA = 1, MOL_RNA = 2, MOL_NONPOLYMER = 3;
  std::vector<int> molType(T, 0);
  std::vector<char> ligandToken(T, 0), atomised(T, 0);
  for (auto& s : b.ligandSpans) for (int t = s.from; t < s.from + s.count; ++t) ligandToken[t] = 1;
  for (int t = 0; t < T; ++t) {
    if (ligandToken[t]) { molType[t] = MOL_NONPOLYMER; continue; }
    const std::string& kind = b.chainKinds[(int)b.chainOfResidue[b.residueOfToken[t]]];
    molType[t] = kind == "dna" ? MOL_DNA : kind == "rna" ? MOL_RNA : MOL_PROTEIN;
  }
  int live = 0;
  for (float m : b.refMask) live += m != 0;
  int atoms = (live + 31) / 32 * 32;
  std::vector<float> refPos((size_t)atoms * 3, 0), refCharge(atoms, 0), mask(atoms, 0);
  std::vector<int> refElement(atoms, 0), refNames((size_t)atoms * 4, 0), refSpaceUid(atoms, 0), atomToToken(atoms, 0), denseSlot(atoms, -1);
  {
    int at = 0;
    for (int t = 0; t < T; ++t)
      for (int slot = 0; slot < dense; ++slot) {
        int from = t * dense + slot;
        if (b.refMask[from] == 0) continue;
        for (int x = 0; x < 3; ++x) refPos[at * 3 + x] = b.refPos[from * 3 + x];
        refCharge[at] = b.refCharge[from]; refElement[at] = b.refElement[from];
        for (int i = 0; i < 4; ++i) refNames[at * 4 + i] = b.refAtomNameChars[from * 4 + i];
        refSpaceUid[at] = b.refSpaceUid[from]; atomToToken[at] = t; denseSlot[at] = from; mask[at] = 1;
        ++at;
      }
  }
  for (auto& s : b.modifiedSpans) if (!s.oneToken) for (int t = s.from; t < s.from + s.count; ++t) atomised[t] = 1;
  // AF3_TO_ESMFOLD2_AATYPE and AATYPE_TO_ESM_ID
  auto esmfold2Aatype = [&](int af3) {
    if (af3 >= 0 && af3 <= 19) return af3 + 2;
    if (af3 >= 22 && af3 <= 25) return af3 + 1;
    if (af3 >= 26 && af3 <= 29) return af3 + 2;
    return UNKNOWN;
  };
  static const int ESM_ID[22] = {0, 0, 5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7};   // aatype 2..21
  std::vector<float> aatype((size_t)T * CLASSES, 0);
  std::vector<int> residueType(T), inputIds(T);
  for (int t = 0; t < T; ++t) {
    int code = molType[t] == MOL_NONPOLYMER || atomised[t] ? UNKNOWN : esmfold2Aatype(b.aatype[t]);
    residueType[t] = code;
    aatype[(size_t)t * CLASSES + code] = 1;
    int id = code >= 2 && code <= 21 ? ESM_ID[code] : code == UNKNOWN ? 3 : 24;
    inputIds[t] = molType[t] == MOL_PROTEIN && !atomised[t] ? id : 24;
  }
  auto zeroBase = [](const std::vector<int>& v) { std::vector<int> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = v[i] - 1; return o; };
  std::vector<int> residueIndex = zeroBase(b.residueIndex), tokenIndex = zeroBase(b.tokenIndex), asymId = zeroBase(b.asymId),
                   entityId = zeroBase(b.entityId), symId = zeroBase(b.symId);
  Entries& E = out.entries;
  E.i("residue_index", residueIndex); E.i("token_index", tokenIndex); E.i("asym_id", asymId);
  E.i("entity_id", entityId); E.i("sym_id", symId); E.i("mol_type", molType);
  E.i("res_type", residueType); E.i("input_ids", inputIds);
  // the contact rule (contacts.js): per pair, how many distogram bins lie under its threshold
  {
    int bins = 128;
    std::string bundle = args.option("fold-bundle");
    if (!bundle.empty()) {
      Json m = parseJson(readFile(bundle + "/manifest.json"));
      const Json* trunk = m.get("trunk");
      const Json* d = trunk ? trunk->get("distogramBins") : nullptr;
      bins = d && !d->isNull() ? (int)d->n : 128;
    }
    if (bins != 128 && bins != 64) throw std::runtime_error("a " + std::to_string(bins) + "-bin distogram: only 128 (2-52) and 64 (AF3's) are known");
    auto klass = [&](int t) {
      if (molType[t] == MOL_NONPOLYMER) return 1;
      if (molType[t] != MOL_PROTEIN) return 0;
      int at = residueType[t] - 2;
      return at >= 0 && at < 20 ? 2 + at : 22;
    };
    static const int LIGAND_PROTEIN[20] = {5, 8, 7, 7, 6, 7, 7, 5, 8, 6, 7, 7, 8, 8, 6, 6, 6, 7, 8, 6};
    auto angstroms = [&](int a, int c) -> double {
      auto kindOf = [](int k) { return k == 0 ? std::string("nucleic") : k == 1 ? std::string("ligand") : std::string("protein"); };
      std::string ka = kindOf(a), kb = kindOf(c);
      if (ka == "ligand" && kb == "protein" && c >= 2 && c < 22) return LIGAND_PROTEIN[c - 2];
      if (kb == "ligand" && ka == "protein" && a >= 2 && a < 22) return LIGAND_PROTEIN[a - 2];
      std::string key = ka < kb ? ka + "-" + kb : kb + "-" + ka;
      static const std::map<std::string, double> BY = {{"protein-protein", 8}, {"nucleic-protein", 10}, {"ligand-protein", 7},
                                                       {"nucleic-nucleic", 9}, {"ligand-nucleic", 7}, {"ligand-ligand", 5}};
      auto it = BY.find(key);
      return it == BY.end() ? 8 : it->second;
    };
    auto under = [&](double threshold) {
      int count = 0;
      if (bins == 128) {
        double width = (52.0 - 2.0) / 128;
        for (int bin = 0; bin < 128; ++bin) if (2 + (bin + 0.5) * width < threshold) ++count;
      } else {
        std::vector<double> breaks;
        for (int k = 0; k < 63; ++k) breaks.push_back(2.3125 + (k * (21.6875 - 2.3125)) / 62);
        double spacing = breaks[62] - breaks[61];
        auto top = [&](int bin) { return bin < 63 ? breaks[bin] : breaks[62] + spacing; };
        while (count <= 63 && top(count) <= threshold + 1e-3) ++count;
      }
      return count;
    };
    std::vector<int> classes(T);
    for (int t = 0; t < T; ++t) classes[t] = klass(t);
    std::vector<int> counts((size_t)T * T);
    std::map<int, int> cache;
    for (int i = 0; i < T; ++i)
      for (int j = 0; j < T; ++j) {
        int key = classes[i] * 23 + classes[j];
        auto it = cache.find(key);
        if (it == cache.end()) it = cache.emplace(key, under(angstroms(classes[i], classes[j]))).first;
        counts[(size_t)i * T + j] = it->second;
      }
    E.i("contact_bins", counts);
    E.m("meta/contactBinsFor", bins);
  }
  // representativeAtoms: CB (else CA) for a residue, C4 / C2 for a base, else the token's first atom
  {
    auto named = [&](int atom, const std::string& want) {
      for (int i = 0; i < 4; ++i) {
        int wanted = i < (int)want.size() ? (unsigned char)want[i] - 32 : 0;
        if (refNames[atom * 4 + i] != wanted) return false;
      }
      return true;
    };
    std::vector<int> alpha(T, -1), beta(T, -1), baseAtom(T, -1), first(T, -1), rep(T);
    for (int atom = 0; atom < atoms; ++atom) {
      if (mask[atom] == 0) continue;
      int t = atomToToken[atom];
      if (first[t] < 0) first[t] = atom;
      if (named(atom, "CA")) alpha[t] = atom;
      if (named(atom, "CB")) beta[t] = atom;
      int rt = residueType[t];
      if (named(atom, (rt == 23 || rt == 24 || rt == 28 || rt == 29) ? "C4" : "C2")) baseAtom[t] = atom;
    }
    for (int t = 0; t < T; ++t) {
      bool nucleic = molType[t] == MOL_DNA || molType[t] == MOL_RNA;
      int pick = nucleic ? baseAtom[t] : (beta[t] >= 0 ? beta[t] : alpha[t]);
      rep[t] = pick >= 0 ? pick : std::max(0, first[t]);
    }
    E.i("distogram_atom_idx", rep);
  }
  // the alignment, as biohub's esm builds it
  {
    const int GAP = 1, UNK = 22;
    static const std::string LETTERS = "ARNDCQEGHILKMFPSTWYV";
    auto code = [&](char ch) { if (ch == '-') return GAP; size_t at = LETTERS.find(ch); return at != std::string::npos && ch ? (int)at + 2 : UNK; };
    struct Msa { std::vector<std::vector<int>> rows; std::vector<std::vector<double>> dels; };
    auto parse = [&](const std::string& text) {
      Msa m;
      // text.split(/^>/m).slice(1): each record from a line-initial ">"
      std::vector<std::string> records;
      size_t at = 0;
      bool started = false;
      std::string current;
      while (at <= text.size()) {
        size_t end = text.find('\n', at);
        std::string line = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
        if (!line.empty() && line[0] == '>') { if (started) records.push_back(current); current = line.substr(1); started = true; }
        else if (started) current += "\n" + line;
        if (end == std::string::npos) break;
        at = end + 1;
      }
      if (started) records.push_back(current);
      for (auto& record : records) {
        size_t nl = record.find('\n');
        std::string seq;
        if (nl != std::string::npos) for (char c : record.substr(nl + 1)) if (!std::isspace((unsigned char)c)) seq += c;
        std::vector<int> r;
        std::vector<double> d;
        int ins = 0;
        for (char ch : seq) {
          if (ch == '.' || (ch >= 'a' && ch <= 'z')) { ++ins; continue; }
          r.push_back(code((char)std::toupper((unsigned char)ch)));
          d.push_back(ins);
          ins = 0;
        }
        m.rows.push_back(r);
        m.dels.push_back(d);
      }
      return m;
    };
    std::vector<std::string> chainSeqs = splitOn(sequence, ':');
    std::vector<std::string> chainKinds = kinds.empty() ? std::vector<std::string>(chainSeqs.size(), "protein") : splitOn(kinds, ',');
    std::vector<std::string> a3ms = splitOn(args.option("a3m"), ',');
    std::vector<std::pair<int, int>> firstToken;      // asym -> its first token, in token order
    for (int t = 0; t < T; ++t)
      if (std::find_if(firstToken.begin(), firstToken.end(), [&](auto& p) { return p.first == asymId[t]; }) == firstToken.end())
        firstToken.push_back({asymId[t], t});
    std::map<int, Msa> chainMsa;
    size_t proteinAt = 0;
    for (size_t c = 0; c < chainSeqs.size(); ++c) {
      if (c >= chainKinds.size() || chainKinds[c] != "protein") continue;
      std::string path = proteinAt < a3ms.size() ? a3ms[proteinAt] : "";
      ++proteinAt;
      Msa m;
      if (!path.empty()) m = parse(readFile(path));
      else {
        std::vector<int> r;
        for (char ch : chainSeqs[c]) r.push_back(code(ch));
        m.rows.push_back(r);
        m.dels.push_back(std::vector<double>(chainSeqs[c].size(), 0));
      }
      if (c < firstToken.size()) chainMsa[firstToken[c].first] = m;
    }
    size_t deepest = 1;
    for (auto& [k, m] : chainMsa) deepest = std::max(deepest, m.rows.size());
    int depth = (int)std::min<size_t>(16384, deepest);
    std::vector<int> rows((size_t)depth * T, GAP);
    std::vector<float> dels((size_t)depth * T, 0.0f);
    for (int t = 0; t < T; ++t) {
      auto it = chainMsa.find(asymId[t]);
      if (it == chainMsa.end()) { rows[t] = residueType[t]; continue; }
      const Msa& m = it->second;
      int width = (int)m.rows[0].size();
      int firstOf = 0;
      for (auto& p : firstToken) if (p.first == asymId[t]) firstOf = p.second;
      int col = std::min(residueIndex[t] - residueIndex[firstOf], width - 1);
      for (int r = 0; r < std::min(depth, (int)m.rows.size()); ++r) {
        bool inside = col >= 0 && col < (int)m.rows[r].size();
        rows[(size_t)r * T + t] = inside ? m.rows[r][col] : 0;
        dels[(size_t)r * T + t] = inside ? (float)m.dels[r][col] : NAN;
      }
    }
    std::vector<float> profile((size_t)T * 33, 0.0f), deletionMean(T, 0.0f);
    for (int r = 0; r < depth; ++r)
      for (int t = 0; t < T; ++t) {
        float& p = profile[(size_t)t * 33 + rows[(size_t)r * T + t]];
        p = (float)((double)p + 1.0 / depth);
        deletionMean[t] = (float)((double)deletionMean[t] + (M_PI / 2) * jsmath::atan((double)dels[(size_t)r * T + t] / 3) / depth);
      }
    E.i("msa/rows", rows); E.t("msa/deletion", dels);
    E.m("meta/msa_depth", depth);
    E.t("profile", profile); E.t("deletion_mean", deletionMean);
  }
  E.t("aatype", aatype);
  E.t("token_bonds", b.hasBonds ? b.bondMatrix : std::vector<float>((size_t)T * T, 0.0f));
  E.t("ref_pos", refPos); E.t("ref_charge", refCharge); E.t("atom_mask", mask);
  E.i("ref_element", refElement); E.i("ref_atom_name_chars", refNames);
  E.i("ref_space_uid", refSpaceUid); E.i("atom_to_token", atomToToken);
  // languageModelInput: one row a protein residue, each chain [BOS, residues, EOS] in chain order
  {
    struct Row { int id, chain; };
    std::vector<Row> rows;
    std::map<std::pair<int, int>, int> rowOfKey;
    std::vector<int> tokenToRow(T, -1);
    for (int t = 0; t < T; ++t) {
      if (molType[t] != MOL_PROTEIN) continue;
      auto key = std::make_pair(asymId[t], residueIndex[t]);
      auto it = rowOfKey.find(key);
      int row;
      if (it == rowOfKey.end()) { row = (int)rows.size(); rowOfKey[key] = row; rows.push_back({inputIds[t], asymId[t]}); }
      else row = it->second;
      tokenToRow[t] = row;
    }
    std::vector<int> ids, sequenceId;
    if (!rows.empty()) {
      std::set<int> chains;
      for (auto& r : rows) chains.insert(r.chain);
      std::vector<int> positionOfRow(rows.size(), 0);
      int index = 0;
      for (int chain : chains) {
        ids.push_back(0); sequenceId.push_back(index);
        for (size_t r = 0; r < rows.size(); ++r) {
          if (rows[r].chain != chain) continue;
          positionOfRow[r] = (int)ids.size();
          ids.push_back(rows[r].id); sequenceId.push_back(index);
        }
        ids.push_back(2); sequenceId.push_back(index);
        ++index;
      }
      for (int t = 0; t < T; ++t) if (tokenToRow[t] >= 0) tokenToRow[t] = positionOfRow[tokenToRow[t]];
    }
    E.i("lm/ids", ids); E.i("lm/sequence_id", sequenceId); E.i("lm/token_to_row", tokenToRow);
    E.m("meta/tokens", T); E.m("meta/atoms", atoms); E.m("meta/lm_rows", (double)ids.size()); E.m("meta/classes", CLASSES);
    out.said.push_back(std::to_string(T) + " tokens, " + std::to_string(atoms) + " atoms (" + std::to_string(live) + " live), "
                       + std::to_string(ids.size()) + " tower rows -> " + args.positional);
  }
  // pdb.template: each atom's index as its coordinates (x = index / 1000, y = index % 1000)
  {
    std::vector<float> positions((size_t)T * dense * 3, 0.0f);
    for (int atom = 0; atom < atoms; ++atom) {
      int slot = denseSlot[atom];
      if (slot < 0) continue;
      positions[(size_t)slot * 3] = (float)(atom / 1000);
      positions[(size_t)slot * 3 + 1] = (float)(atom % 1000);
    }
    out.pdb = pdbText(b, positions);
  }
  return out;
}

}  // namespace lf
