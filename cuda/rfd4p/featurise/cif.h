// mmCIF, read natively: the tokeniser, the block reader and the `_atom_site` loop.
//
// Ported from `shared/rfd4p/pipeline/cif.js`'s `tokenizeCif`, `parseCifBlock` and the
// `_atom_site` half of `parseCifToAtomArray`. This is the FIRST half of the structure path - what
// comes straight out of the file. The second half is the chemistry the reference derives from it
// (`is_polymer`, `chem_comp_type`, `atomic_number`, the leaving-atom flags, the id and entity
// annotations and the bonds, which need the CCD tables), and it is not here yet.
//
// 🔴 ITS BAR IS THE REFERENCE'S OWN COLUMNS, FIELD BY FIELD - `gate.py`'s `cif` arm dumps what
// `parseCifToAtomArray` produces and compares every column this reads. A reader that is merely
// plausible is the worst kind: a mis-parsed occupancy or a dropped annotation reaches the model as
// a number rather than as an error.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "tables.inc"

namespace rfd4p {
namespace cif {

using std::string;
using std::vector;

struct Token {
  string value;
  bool quoted = false;
};

/**
 * mmCIF's lexical rules, which are not a shell's:
 *   - a `;` in COLUMN ZERO opens a multi-line value, closed by the next `;` in column zero;
 *   - a quote closes only when the character after it is whitespace, so `O5'` is one token and
 *     `'a b'` is another - this is why a naive split on quotes mangles nucleic atom names;
 *   - `#` starts a comment, but only outside a quoted value.
 */
inline vector<Token> tokenize(const string& text) {
  vector<Token> out;
  const size_t n = text.size();
  size_t i = 0;
  bool atLineStart = true;
  while (i < n) {
    if (atLineStart && text[i] == ';') {
      size_t end = i + 1;
      bool closed = false;
      while (end < n) {
        if ((text[end] == '\n' || text[end] == '\r') && end + 1 < n) {
          size_t next = end + 1;
          if (text[end] == '\r' && next < n && text[next] == '\n') ++next;
          if (next < n && text[next] == ';') {
            out.push_back({text.substr(i + 1, end - i - 1), true});
            i = next + 1;
            atLineStart = false;
            closed = true;
            break;
          }
        }
        ++end;
      }
      if (!closed) break;
      continue;
    }
    const char ch = text[i];
    if (ch == '\n' || ch == '\r') { atLineStart = true; ++i; continue; }
    if (ch == ' ' || ch == '\t') { ++i; continue; }
    atLineStart = false;
    if (ch == '#') {
      while (i < n && text[i] != '\n' && text[i] != '\r') ++i;
      continue;
    }
    if (ch == '\'' || ch == '"') {
      const char quote = ch;
      size_t j = i + 1;
      while (j < n) {
        if (text[j] == quote) {
          const char next = j + 1 < n ? text[j + 1] : ' ';
          if (next == ' ' || next == '\t' || next == '\n' || next == '\r') break;
        }
        ++j;
      }
      out.push_back({text.substr(i + 1, j - i - 1), true});
      i = j + 1;
      continue;
    }
    size_t j = i;
    while (j < n) {
      const char c = text[j];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '#') break;
      ++j;
    }
    out.push_back({text.substr(i, j - i), false});
    i = j;
  }
  return out;
}

/** `block[category][column]` -> the column's values, a single-valued item being a column of one. */
using Block = std::map<string, std::map<string, vector<string>>>;

inline Block parseBlock(const string& text) {
  const vector<Token> tokens = tokenize(text);
  Block block;
  const size_t n = tokens.size();
  size_t i = 0;
  auto split = [](const string& name, string& cat, string& col) {
    const size_t dot = name.find('.');
    cat = name.substr(1, dot == string::npos ? string::npos : dot - 1);
    col = dot == string::npos ? "" : name.substr(dot + 1);
  };
  while (i < n) {
    const Token& tok = tokens[i];
    if (!tok.quoted && tok.value.rfind("data_", 0) == 0) { ++i; continue; }
    if (!tok.quoted && tok.value == "loop_") {
      ++i;
      vector<std::pair<string, string>> headers;
      while (i < n && !tokens[i].quoted && !tokens[i].value.empty() && tokens[i].value[0] == '_') {
        string cat, col;
        split(tokens[i].value, cat, col);
        headers.push_back({cat, col});
        block[cat][col];
        ++i;
      }
      if (headers.empty()) continue;
      while (i < n) {
        const Token& t = tokens[i];
        if (!t.quoted && (t.value == "loop_" || (!t.value.empty() && t.value[0] == '_') ||
                          t.value.rfind("data_", 0) == 0)) {
          break;
        }
        for (size_t c = 0; c < headers.size() && i < n; ++c, ++i) {
          block[headers[c].first][headers[c].second].push_back(tokens[i].value);
        }
      }
      continue;
    }
    if (!tok.quoted && !tok.value.empty() && tok.value[0] == '_') {
      string cat, col;
      split(tok.value, cat, col);
      block[cat][col] = {i + 1 < n ? tokens[i + 1].value : string()};
      i += 2;
      continue;
    }
    ++i;
  }
  return block;
}

// `.`, `?` and the empty string are mmCIF's "not given", and each kind has its own fallback.
inline bool unset(const string& s) { return s.empty() || s == "." || s == "?"; }
inline int safeInt(const string& s, int fallback = 0) {
  if (unset(s)) return fallback;
  char* end = nullptr;
  const long v = strtol(s.c_str(), &end, 10);
  return end == s.c_str() ? fallback : (int)v;
}
inline float safeFloat(const string& s, float fallback = NAN) {
  if (unset(s) || s == "nan" || s == "NaN") return fallback;
  char* end = nullptr;
  const double v = strtod(s.c_str(), &end);
  return end == s.c_str() ? fallback : (float)v;
}
inline bool safeBool(const string& s, bool fallback = false) {
  if (unset(s)) return fallback;
  if (s == "True" || s == "true" || s == "1" || s == "Y") return true;
  if (s == "False" || s == "false" || s == "0" || s == "N") return false;
  return fallback;
}

/**
 * The `_atom_site` loop as the reference reads it - the columns that come STRAIGHT FROM THE FILE,
 * and the `condition_*` / `annotation_*` columns beside them, which is how an annotated structure
 * carries its job.
 */
struct AtomSite {
  size_t n = 0;
  vector<string> element, atomName, resName, chainId, insCode, labelAltId;
  vector<int> labelEntityId;
  vector<int> resId, authSeqId, atomId, charge;
  vector<float> x, y, z, bFactor, occupancy;
  vector<uint8_t> hetero;
  // Every `_atom_site.<name>` this reader does not have a field for, kept by name: the annotation
  // columns are exactly these, and a job's conditioning lives in them.
  std::map<string, vector<string>> extra;

  // Derived, not read: the chemistry `parseCifToAtomArray` works out from the columns above and
  // from the file's own `_chem_comp` / `_chem_comp_atom` / `_entity_poly` tables.
  vector<int> atomicNumber, chainType;
  vector<uint8_t> isPolymer, isAromatic, isLeavingAtom;
  vector<string> chemCompType;
};

/**
 * `parseCifToAtomArray`'s derived half: the atomic number, the component type, the aromatic and
 * leaving-atom flags, the charge fallbacks and the per-chain type.
 *
 * 🔴 THE FILE'S OWN TABLES WIN OVER THE STANDARD ONES. `_chem_comp_atom` carries a charge, an
 * aromatic flag and a leaving flag per component atom, and a structure that has them means them -
 * the built-in tables are the fallback for a component the file says nothing about, not the other
 * way round.
 *
 * 🔴 AND A NON-POLYMER CHAIN TAKES `auth_seq_id` AS ITS RESIDUE NUMBER (`update_nonpoly_seq_ids`),
 * which is not cosmetic: a ligand's `label_seq_id` is `.` and so -1 for every one of its atoms,
 * which would collapse a whole het group into one residue.
 */
inline void deriveChemistry(AtomSite& a, const Block& block) {
  auto table = [&](const char* cat, const char* col) -> const vector<string>* {
    auto c = block.find(cat);
    if (c == block.end()) return nullptr;
    auto it = c->second.find(col);
    return it == c->second.end() ? nullptr : &it->second;
  };
  // `_chem_comp.type`, by component id.
  std::map<string, string> compType;
  if (const vector<string>* ids = table("chem_comp", "id")) {
    const vector<string>* types = table("chem_comp", "type");
    if (!types) types = table("chem_comp", "pdbx_type");
    for (size_t i = 0; types && i < ids->size() && i < types->size(); ++i) {
      if (unset((*types)[i])) continue;
      string up = (*types)[i];
      for (char& ch : up) ch = (char)toupper((unsigned char)ch);
      compType[(*ids)[i]] = up;
    }
  }
  // `_chem_comp_atom`, by `<comp>:<atom>`.
  std::map<string, int> ccaCharge;
  std::map<string, bool> ccaArom, ccaLeaving;
  if (const vector<string>* cids = table("chem_comp_atom", "comp_id")) {
    const vector<string>* aids = table("chem_comp_atom", "atom_id");
    const vector<string>* chgs = table("chem_comp_atom", "charge");
    const vector<string>* aroms = table("chem_comp_atom", "pdbx_aromatic_flag");
    const vector<string>* leavs = table("chem_comp_atom", "pdbx_leaving_atom_flag");
    for (size_t i = 0; aids && i < cids->size() && i < aids->size(); ++i) {
      const string key = (*cids)[i] + ":" + (*aids)[i];
      if (chgs && i < chgs->size()) ccaCharge[key] = safeInt((*chgs)[i], 0);
      if (aroms && i < aroms->size()) ccaArom[key] = (*aroms)[i] == "Y";
      if (leavs && i < leavs->size()) ccaLeaving[key] = (*leavs)[i] == "Y";
    }
  }
  // `_entity.type` and `_entity_poly`, which say what KIND of chain each one is.
  std::map<string, string> entityType, entityPolyType, chainToPolyType;
  if (const vector<string>* ids = table("entity", "id")) {
    if (const vector<string>* types = table("entity", "type")) {
      for (size_t i = 0; i < ids->size() && i < types->size(); ++i) {
        string low = (*types)[i];
        for (char& ch : low) ch = (char)tolower((unsigned char)ch);
        entityType[(*ids)[i]] = low;
      }
    }
  }
  if (const vector<string>* eids = table("entity_poly", "entity_id")) {
    const vector<string>* ptypes = table("entity_poly", "type");
    const vector<string>* strands = table("entity_poly", "pdbx_strand_id");
    for (size_t i = 0; i < eids->size(); ++i) {
      string pt = ptypes && i < ptypes->size() ? (*ptypes)[i] : string();
      for (char& ch : pt) ch = (char)tolower((unsigned char)ch);
      entityPolyType[(*eids)[i]] = pt;
      if (strands && i < strands->size()) {
        string list = (*strands)[i];
        size_t from = 0;
        while (from <= list.size()) {
          const size_t comma = list.find(',', from);
          string one = list.substr(from, comma == string::npos ? string::npos : comma - from);
          while (!one.empty() && isspace((unsigned char)one.front())) one.erase(one.begin());
          while (!one.empty() && isspace((unsigned char)one.back())) one.pop_back();
          if (!one.empty()) chainToPolyType[one] = pt;
          if (comma == string::npos) break;
          from = comma + 1;
        }
      }
    }
  }

  a.atomicNumber.assign(a.n, 0);
  a.chemCompType.assign(a.n, "");
  a.isAromatic.assign(a.n, 0);
  a.isLeavingAtom.assign(a.n, 0);
  a.chainType.assign(a.n, 0);
  a.isPolymer.assign(a.n, 0);
  for (size_t k = 0; k < a.n; ++k) {
    const string key = a.resName[k] + ":" + a.atomName[k];
    if (a.charge[k] == 0) {
      auto cc = ccaCharge.find(key);
      if (cc != ccaCharge.end()) a.charge[k] = cc->second;
      else {
        auto sc = tables::STD_RESIDUE_CHARGES.find(key);
        if (sc != tables::STD_RESIDUE_CHARGES.end()) a.charge[k] = sc->second;
      }
    }
    auto ar = ccaArom.find(key);
    a.isAromatic[k] = ar != ccaArom.end() ? (ar->second ? 1 : 0)
                                          : (tables::STD_RESIDUE_AROMATIC.count(key) ? 1 : 0);
    auto lv = ccaLeaving.find(key);
    if (lv != ccaLeaving.end()) {
      a.isLeavingAtom[k] = lv->second ? 1 : 0;
    } else {
      const bool aa = tables::AA_RESIDUES.count(a.resName[k]) && a.atomName[k] == "OXT";
      const bool na = (tables::RNA_RESIDUES.count(a.resName[k]) ||
                       tables::DNA_RESIDUES.count(a.resName[k])) && a.atomName[k] == "OP3";
      a.isLeavingAtom[k] = (aa || na) ? 1 : 0;
    }
    string up = a.element[k];
    for (char& ch : up) ch = (char)toupper((unsigned char)ch);
    auto z = tables::ELEMENT_TO_ATOMIC_NUMBER.find(up);
    a.atomicNumber[k] = z == tables::ELEMENT_TO_ATOMIC_NUMBER.end() ? 0 : z->second;
    auto ct = compType.find(a.resName[k]);
    if (ct != compType.end()) a.chemCompType[k] = ct->second;
    else if (a.resName[k] == "GLY") a.chemCompType[k] = "PEPTIDE LINKING";
    else if (tables::AA_RESIDUES.count(a.resName[k])) a.chemCompType[k] = "L-PEPTIDE LINKING";
    else if (tables::DNA_RESIDUES.count(a.resName[k])) a.chemCompType[k] = "DNA LINKING";
    else if (tables::RNA_RESIDUES.count(a.resName[k])) a.chemCompType[k] = "RNA LINKING";
    else a.chemCompType[k] = "NON-POLYMER";
  }

  // The chain type is decided ONCE a chain, from its FIRST atom, and applied to all of them.
  std::map<string, int> byChain;
  for (size_t k = 0; k < a.n; ++k) {
    if (byChain.count(a.chainId[k])) continue;
    const string eid = std::to_string(a.labelEntityId[k]);
    string poly;
    auto c2p = chainToPolyType.find(a.chainId[k]);
    if (c2p != chainToPolyType.end()) poly = c2p->second;
    if (poly.empty()) { auto e = entityPolyType.find(eid); if (e != entityPolyType.end()) poly = e->second; }
    if (poly.empty()) { auto e = entityType.find(eid); if (e != entityType.end()) poly = e->second; }
    int type = -1;
    if (!poly.empty()) {
      auto m = tables::MMCIF_CHAIN_TYPE_MAP.find(poly);
      if (m != tables::MMCIF_CHAIN_TYPE_MAP.end()) type = m->second;
    }
    if (type < 0) {
      const string& res = a.resName[k];
      // 🔴 `5-20` AND `<M>` ARE POLYPEPTIDE, which is not a curiosity: an expandable segment's
      // sentinel is named by its length range and a masked residue by `<M>`, and both must come
      // out as protein or a contig's own chain is typed NON-POLYMER.
      bool range = !res.empty() && res.find('-') != string::npos;
      for (char ch : res) range = range && (isdigit((unsigned char)ch) || ch == '-');
      if (tables::DNA_RESIDUES.count(res)) type = tables::CT_DNA;
      else if (tables::RNA_RESIDUES.count(res)) type = tables::CT_RNA;
      else if (tables::AA_RESIDUES.count(res) || range || res == "<M>")
        type = tables::CT_POLYPEPTIDE_L;
      else type = tables::CT_NON_POLYMER;
    }
    byChain[a.chainId[k]] = type;
  }
  for (size_t k = 0; k < a.n; ++k) {
    a.chainType[k] = byChain[a.chainId[k]];
    a.isPolymer[k] = a.chainType[k] <= tables::CT_RNA ? 1 : 0;
    if (!a.isPolymer[k]) a.resId[k] = a.authSeqId[k];
  }
}



inline AtomSite readAtomSite(const Block& block) {
  auto cat = block.find("atom_site");
  if (cat == block.end()) {
    fprintf(stderr, "this mmCIF has no _atom_site loop\n");
    exit(1);
  }
  const auto& cols = cat->second;
  auto column = [&](const char* name) -> const vector<string>* {
    auto it = cols.find(name);
    return it == cols.end() ? nullptr : &it->second;
  };
  auto at = [](const vector<string>* c, size_t i) -> string {
    return c && i < c->size() ? (*c)[i] : string();
  };

  const vector<string>* labelAtom = column("label_atom_id");
  const vector<string>* authAtom = column("auth_atom_id");
  const size_t rows = labelAtom ? labelAtom->size() : (authAtom ? authAtom->size() : 0);

  // 🔴 MODEL ONE AND THE FIRST ALTLOC, which is a FILTER and not a formality: a deposited entry
  // with twenty NMR models would otherwise come through twenty times over, and a residue with two
  // conformations twice. The altloc key is the reference's - `label_asym`, `label_seq`,
  // `auth_seq`, `label_comp`, `label_atom` - and the FIRST row with a given key wins.
  const vector<string>* modelNums = column("pdbx_PDB_model_num");
  const vector<string>* altIds = column("label_alt_id");
  const string firstModel = modelNums && !modelNums->empty() ? (*modelNums)[0] : string();
  std::vector<size_t> selected;
  std::map<string, bool> seenAlt;
  for (size_t i = 0; i < rows; ++i) {
    if (modelNums && !modelNums->empty() && at(modelNums, i) != firstModel) continue;
    const string alt = altIds ? at(altIds, i) : ".";
    if (!unset(alt)) {
      const string key = at(column("label_asym_id"), i) + "_" + at(column("label_seq_id"), i) +
                         "_" + at(column("auth_seq_id"), i) + "_" +
                         at(column("label_comp_id"), i) + "_" + at(labelAtom, i);
      if (seenAlt.count(key)) continue;
      seenAlt[key] = true;
    }
    selected.push_back(i);
  }

  // 🔴 `label_*` WINS, NOT `auth_*`. The reference reads `label_atom_id || auth_atom_id` and
  // `safeInt(label_seq_id, -1)` for the residue number - writing it the other way round is the
  // obvious guess and it is wrong, and on a file where the two numberings differ it renumbers
  // every residue a contig names.
  AtomSite a;
  a.n = selected.size();
  for (size_t k = 0; k < a.n; ++k) {
    const size_t i = selected[k];
    auto pick = [&](const char* first, const char* second, const char* fallback) {
      const string v = at(column(first), i);
      if (!v.empty()) return v;
      const string w = at(column(second), i);
      return w.empty() ? string(fallback) : w;
    };
    a.hetero.push_back(at(column("group_PDB"), i) == "HETATM" ? 1 : 0);
    const string el = at(column("type_symbol"), i);
    a.element.push_back(el.empty() ? "C" : el);
    a.atomName.push_back(pick("label_atom_id", "auth_atom_id", ""));
    a.resName.push_back(pick("label_comp_id", "auth_comp_id", ""));
    a.chainId.push_back(pick("label_asym_id", "auth_asym_id", "A"));
    a.labelEntityId.push_back(safeInt(at(column("label_entity_id"), i), 0));
    const int resId = safeInt(at(column("label_seq_id"), i), -1);
    a.resId.push_back(resId);
    const string ins = at(column("pdbx_PDB_ins_code"), i);
    a.insCode.push_back(unset(ins) ? "" : ins);
    a.authSeqId.push_back(safeInt(at(column("auth_seq_id"), i), resId));
    a.atomId.push_back(safeInt(at(column("id"), i), (int)k + 1));
    a.bFactor.push_back(safeFloat(at(column("B_iso_or_equiv"), i)));
    a.occupancy.push_back(safeFloat(at(column("occupancy"), i), 1.0f));
    const string alt = at(column("label_alt_id"), i);
    a.labelAltId.push_back(alt.empty() ? "." : alt);
    a.x.push_back(safeFloat(at(column("Cartn_x"), i)));
    a.y.push_back(safeFloat(at(column("Cartn_y"), i)));
    a.z.push_back(safeFloat(at(column("Cartn_z"), i)));
    // 🔴 THE FORMAL CHARGE ALONE. Where it is zero the reference falls back to the file's own
    // `_chem_comp_atom` table and then to `STD_RESIDUE_CHARGES`, which is CHEMISTRY and belongs
    // with the bonds - not here, and this reader does not pretend otherwise.
    a.charge.push_back(safeInt(at(column("pdbx_formal_charge"), i), 0));
  }
  static const char* KNOWN[] = {
      "group_PDB", "type_symbol", "label_atom_id", "auth_atom_id", "label_comp_id",
      "auth_comp_id", "label_asym_id", "auth_asym_id", "label_seq_id", "auth_seq_id",
      "label_alt_id", "label_entity_id", "pdbx_PDB_ins_code", "id", "pdbx_formal_charge",
      "Cartn_x", "Cartn_y", "Cartn_z", "B_iso_or_equiv", "occupancy", "pdbx_PDB_model_num"};
  for (const auto& [name, values] : cols) {
    bool known = false;
    for (const char* k : KNOWN) known = known || name == k;
    if (!known) {
      vector<string>& dst = a.extra[name];
      for (size_t k : selected) dst.push_back(k < values.size() ? values[k] : string());
    }
  }
  return a;
}

}  // namespace cif
}  // namespace rfd4p
