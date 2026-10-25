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
#include <algorithm>
#include <map>
#include <set>
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
inline double safeFloat(const string& s, double fallback = NAN) {
  if (unset(s) || s == "nan" || s == "NaN") return fallback;
  char* end = nullptr;
  const double v = strtod(s.c_str(), &end);
  return end == s.c_str() ? fallback : v;
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
  // 🔴 DOUBLE, NOT FLOAT. The reference parses these with `parseFloat` and keeps them as
  // JavaScript numbers all the way into the chirality fit, which is sensitive enough that
  // narrowing them here moves the fitted rotation in the sixth figure. Nothing about an mmCIF's
  // three decimals needs double - what needs it is agreeing with the reference bit for bit.
  vector<double> x, y, z, bFactor, occupancy;
  vector<uint8_t> hetero;
  // Every `_atom_site.<name>` this reader does not have a field for, kept by name: the annotation
  // columns are exactly these, and a job's conditioning lives in them.
  std::map<string, vector<string>> extra;
  // The rows of the file this reader kept, which an annotation column must be indexed by.
  vector<size_t> selected;

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
  //
  // 🔴 AND NEITHER FILTER COULD FIRE ON ANYTHING THIS GATE READ. Scanned over the seven shipped
  // structures: every one is model 1 alone and every `label_alt_id` is `.`, so both were ported
  // from the reference and gated by nothing. `sm_sam+twomodels` and `motif_idx_1prw+altloc`
  // build a file that trips them, and each is watched failing: without the model filter the
  // first is **132 atoms against 66**, without the altloc skip the second is **330 against
  // 306**. The altloc case writes the DISPLACED conformation first, because a port that kept
  // the last row rather than the first agrees whenever they happen to be in written order.
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
  //
  // 🔴 AND THE TWO NUMBERINGS ARE EQUAL IN EVERY ROW OF EVERY SHIPPED FILE, so this said
  // "renumbers every residue a contig names" with nothing able to show it.
  // `motif_idx_1prw+authnum` puts them 500 apart, and taking `auth_seq_id` there is watched
  // failing on `res_id` in all 306 atoms, on eight selections, on every condition - and on
  // SIX CONTIGS, each refused outright with "Preserved segment 'A1-6' matched no atoms". That
  // last is the shape worth knowing: the wrong column does not produce a wrong structure, it
  // produces a REFUSAL, which reads as a malformed job rather than as a reader bug.
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
    a.occupancy.push_back(safeFloat(at(column("occupancy"), i), 1.0));
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
  a.selected = selected;
  return a;
}

/**
 * The annotation columns, typed. A structure that carries its job carries it HERE - the
 * conditioning the page would otherwise build from a contig.
 *
 * 🔴 EVERY VALUE HAS ITS OWN DEFAULT AND A `.` MEANS IT. An int column defaults to -1 or 0, a
 * bool to true or false, `condition_sequence_1_residue` to `<M>`; filling a missing entry with
 * zero instead would be conditioning the model on something the file did not say.
 *
 * 🔴 AND THE LEGACY `mask_*` COLUMNS REWRITE THE `condition_*` ONES. `mask_index_1_residue`
 * BECOMES `condition_index_1_residue`, and `mask_sequence_1_residue` turns the residue's own name
 * into its sequence condition where the mask is set and `<M>` where it is not - so a file written
 * the old way means the opposite of what reading the columns separately would give.
 *
 * 🔴 AND NO SHIPPED STRUCTURE CARRIES ONE, so all three rewrites were ported from the reference
 * and gated by nothing. `motif_idx_1prw+legacymask` writes the three columns in place of the
 * conditions, and skipping the index rewrite alone is watched failing on the ANNOTATION KEY SET
 * (`mask_index_1_residue` left where `condition_index_1_residue` should be) and on five contigs'
 * `condition_index`. A port that merely ignored a column it did not know would be green on
 * every file here and would condition the model on nothing where the file conditions it on
 * everything.
 */
struct Annotations {
  std::map<string, vector<int>> ints;        // int and bool columns
  // 🔴 DOUBLE, for the coordinates' reason: `condition_coordinate_1_atom` feeds the chirality
  // fit, and a float there moves the fitted rotation in the sixth figure on any input that uses
  // conditioned coordinates. None of the shipped examples does, which is exactly why it is
  // written down rather than left to be found.
  std::map<string, vector<double>> floats;
  std::map<string, vector<string>> strings;
};

inline Annotations readAnnotations(const AtomSite& a, const Block& block,
                                   const vector<size_t>& selected) {
  Annotations out;
  auto cat = block.find("atom_site");
  const auto& cols = cat->second;
  for (const auto& [name, values] : cols) {
    const tables::AnnotationSpec* spec = nullptr;
    tables::AnnotationSpec derived{"", 1, 0, ""};
    auto known = tables::ANNOTATIONS_1BODY.find(name);
    if (known != tables::ANNOTATIONS_1BODY.end()) {
      spec = &known->second;
    } else if (name.rfind("condition_sequence-probability-", 0) == 0) {
      derived = {"float", 1, NAN, ""};
      spec = &derived;
    } else if (name.rfind("mask_sequence-probability-", 0) == 0) {
      derived = {"bool", 1, 0, ""};
      spec = &derived;
    } else if (name == "condition_cath-label_1_residue" && !block.count(name)) {
      derived = {"str", 1, 0, "x.x.x.x"};
      spec = &derived;
    }
    if (!spec || spec->cols != 1) continue;
    const string type = spec->type;
    for (size_t k = 0; k < selected.size(); ++k) {
      const string raw = selected[k] < values.size() ? values[selected[k]] : string();
      if (type == "int") out.ints[name].push_back(safeInt(raw, (int)spec->fallback));
      else if (type == "bool") out.ints[name].push_back(safeBool(raw, spec->fallback != 0) ? 1 : 0);
      else if (type == "float") out.floats[name].push_back(safeFloat(raw, spec->fallback));
      else out.strings[name].push_back(unset(raw) ? string(spec->text) : raw);
    }
  }
  // The three-wide categories, each its own `loop_` keyed by `idx0`.
  for (const auto& [name, spec] : tables::ANNOTATIONS_1BODY) {
    if (spec.cols == 1) continue;
    auto c = block.find(name);
    if (c == block.end()) continue;
    vector<double>& arr = out.floats[name];
    arr.assign(a.n * spec.cols, spec.fallback);
    auto col = [&](const string& key) -> const vector<string>* {
      auto it = c->second.find(key);
      return it == c->second.end() ? nullptr : &it->second;
    };
    const vector<string>* idx0 = col("idx0");
    const vector<string>* val0 = col("val0");
    const size_t rows = idx0 ? idx0->size() : (val0 ? val0->size() : 0);
    for (size_t r = 0; r < rows; ++r) {
      const int atomIdx = idx0 ? safeInt((*idx0)[r], -1) : (int)r;
      if (atomIdx < 0 || atomIdx >= (int)a.n) continue;
      for (int ch = 0; ch < spec.cols; ++ch) {
        const vector<string>* v = col("val" + std::to_string(ch));
        const string raw = v && r < v->size() ? (*v)[r] : string();
        arr[(size_t)atomIdx * spec.cols + ch] =
            string(spec.type) == "int" ? (double)safeInt(raw, (int)spec.fallback)
                                       : safeFloat(raw, spec.fallback);
      }
    }
  }
  // The legacy masks, which REPLACE the condition columns they shadow.
  if (out.ints.count("mask_index_1_residue")) {
    out.ints["condition_index_1_residue"] = out.ints["mask_index_1_residue"];
    out.ints.erase("mask_index_1_residue");
  }
  if (out.ints.count("mask_sequence_1_residue")) {
    const vector<int> mask = out.ints["mask_sequence_1_residue"];
    vector<string> seq(a.n);
    auto existing = out.strings.find("condition_sequence_1_residue");
    for (size_t i = 0; i < a.n; ++i) {
      if (!mask[i]) { seq[i] = "<M>"; continue; }
      const bool have = existing != out.strings.end() && existing->second[i] != "<M>";
      seq[i] = have ? existing->second[i] : a.resName[i];
    }
    out.strings["condition_sequence_1_residue"] = seq;
    out.ints.erase("mask_sequence_1_residue");
  }
  if (out.ints.count("mask_coordinate_1_atom")) {
    const vector<int> mask = out.ints["mask_coordinate_1_atom"];
    vector<double>& crd = out.floats["condition_coordinate_1_atom"];
    const vector<double> existing = crd;
    crd.assign(a.n * 3, NAN);
    for (size_t i = 0; i < a.n; ++i) {
      if (!mask[i]) continue;
      if (existing.size() == a.n * 3) {
        for (int ch = 0; ch < 3; ++ch) crd[i * 3 + ch] = existing[i * 3 + ch];
      } else {
        crd[i * 3] = a.x[i];
        crd[i * 3 + 1] = a.y[i];
        crd[i * 3 + 2] = a.z[i];
      }
    }
    out.ints.erase("mask_coordinate_1_atom");
  }
  return out;
}

/** `getResidueSegments`: the index where each residue starts, plus `n`. */
inline vector<int> residueSegments(const AtomSite& a) {
  if (a.n == 0) return {0};
  vector<int> starts{0};
  for (size_t i = 1; i < a.n; ++i) {
    if (a.chainId[i] != a.chainId[i - 1] || a.resId[i] != a.resId[i - 1] ||
        a.resName[i] != a.resName[i - 1] || a.insCode[i] != a.insCode[i - 1]) {
      starts.push_back((int)i);
    }
  }
  starts.push_back((int)a.n);
  return starts;
}

/** `getPolymerizationAtoms`: the atom that leaves one residue and the one that enters the next. */
inline std::pair<string, string> polymerizationAtoms(const string& resName, int chainType) {
  if (chainType == tables::CT_POLYPEPTIDE_L || chainType == tables::CT_POLYPEPTIDE_D ||
      chainType == tables::CT_CYCLIC_PSEUDO_PEPTIDE || tables::AA_RESIDUES.count(resName)) {
    return {"C", "N"};
  }
  if (chainType == tables::CT_DNA || chainType == tables::CT_RNA ||
      chainType == tables::CT_DNA_RNA_HYBRID || tables::RNA_RESIDUES.count(resName) ||
      tables::DNA_RESIDUES.count(resName)) {
    return {"O3'", "P"};
  }
  return {"", ""};
}

struct Bond {
  int u = 0, v = 0, type = 0;
};

/**
 * `buildBondsForAtomArray`: the declared links, then each residue's own, then the polymer
 * backbone, then a retype of everything incident on a metal.
 *
 * 🔴 THE ORDER OF THE RESULT IS INSERTION ORDER, NOT SORTED. The reference accumulates into a
 * JavaScript `Map`, which preserves it, and the bond LIST is what later stages index - so a
 * `std::map` here would reorder every bond and move every index that names one.
 *
 * 🔴 AND A FIRST BOND FOR A PAIR WINS. `addBond` only inserts where the key is absent, so a
 * `_struct_conn` covalent link is NOT overwritten by the residue dictionary's idea of the same
 * pair - which is the whole point of reading `struct_conn` first.
 */
inline vector<Bond> buildBonds(const AtomSite& a, const Block& block) {
  vector<Bond> bonds;
  std::map<long long, size_t> seen;
  const int n = (int)a.n;
  auto addBond = [&](int u, int v, int type) {
    if (u == v || u < 0 || v < 0 || u >= n || v >= n) return;
    const int lo = std::min(u, v), hi = std::max(u, v);
    const long long key = (long long)lo * n + hi;
    if (seen.count(key)) return;
    seen[key] = bonds.size();
    bonds.push_back({lo, hi, type});
  };
  auto table = [&](const char* cat, const char* col) -> const vector<string>* {
    auto c = block.find(cat);
    if (c == block.end()) return nullptr;
    auto it = c->second.find(col);
    return it == c->second.end() ? nullptr : &it->second;
  };

  // The residue dictionary: the built-in tables, overlaid by the file's own `_chem_comp_bond`.
  std::map<string, std::map<string, int>> resBonds;
  for (const auto& [res, pairs] : tables::STD_RESIDUE_BONDS) {
    for (const auto& [key, bt] : pairs) resBonds[res][key] = bt;
  }
  if (const vector<string>* comps = table("chem_comp_bond", "comp_id")) {
    const vector<string>* a1s = table("chem_comp_bond", "atom_id_1");
    const vector<string>* a2s = table("chem_comp_bond", "atom_id_2");
    const vector<string>* orders = table("chem_comp_bond", "value_order");
    const vector<string>* aroms = table("chem_comp_bond", "pdbx_aromatic_flag");
    for (size_t i = 0; a1s && a2s && i < comps->size() && i < a1s->size() && i < a2s->size(); ++i) {
      const string& x = (*a1s)[i];
      const string& y = (*a2s)[i];
      string order = orders && i < orders->size() && !(*orders)[i].empty() ? (*orders)[i] : "SING";
      string arom = aroms && i < aroms->size() && !(*aroms)[i].empty() ? (*aroms)[i] : "N";
      for (char& ch : order) ch = (char)toupper((unsigned char)ch);
      for (char& ch : arom) ch = (char)toupper((unsigned char)ch);
      auto it = tables::COMP_BOND_ORDER_TO_TYPE.find(order + ":" + arom);
      resBonds[(*comps)[i]][x < y ? x + ":" + y : y + ":" + x] =
          it == tables::COMP_BOND_ORDER_TO_TYPE.end() ? tables::BT_ANY : it->second;
    }
  }

  const vector<int> starts = residueSegments(a);
  vector<int> atomToRes(a.n, 0);
  for (size_t r = 0; r + 1 < starts.size(); ++r) {
    for (int i = starts[r]; i < starts[r + 1]; ++i) atomToRes[i] = (int)r;
  }

  // 1. `_struct_conn`: the covalent links the file declares, at the identity symmetry only.
  if (const vector<string>* types = table("struct_conn", "conn_type_id")) {
    for (size_t row = 0; row < types->size(); ++row) {
      if ((*types)[row] != "covale") continue;
      const vector<string>* s1 = table("struct_conn", "ptnr1_symmetry");
      const vector<string>* s2 = table("struct_conn", "ptnr2_symmetry");
      if (s1 && row < s1->size() && (*s1)[row] != "1_555") continue;
      if (s2 && row < s2->size() && (*s2)[row] != "1_555") continue;
      int found[2] = {-1, -1};
      for (int p = 0; p < 2; ++p) {
        const string tag = "ptnr" + std::to_string(p + 1);
        auto col = [&](const char* suffix) -> string {
          const vector<string>* c = table("struct_conn", (tag + suffix).c_str());
          return c && row < c->size() ? (*c)[row] : string();
        };
        const string asym = col("_label_asym_id"), comp = col("_label_comp_id");
        string seq = col("_label_seq_id");
        if (seq == "." && !col("_auth_seq_id").empty()) seq = col("_auth_seq_id");
        const int seqInt = safeInt(seq, -999999);
        const string atomName = col("_label_atom_id");
        // 🔴 THE LAST MATCH WINS, which is the reference's loop and not a typo: it keeps
        // assigning rather than breaking out.
        for (int i = 0; i < n; ++i) {
          if (a.chainId[i] == asym && a.resName[i] == comp && a.resId[i] == seqInt &&
              a.atomName[i] == atomName) {
            found[p] = i;
          }
        }
      }
      if (found[0] >= 0 && found[1] >= 0 && found[0] != found[1]) {
        const vector<string>* ord = table("struct_conn", "pdbx_value_order");
        string o = ord && row < ord->size() && !(*ord)[row].empty() ? (*ord)[row] : "sing";
        for (char& ch : o) ch = (char)tolower((unsigned char)ch);
        auto it = tables::STRUCT_CONN_ORDER_TO_TYPE.find(o);
        addBond(found[0], found[1],
                it == tables::STRUCT_CONN_ORDER_TO_TYPE.end() ? tables::BT_SINGLE : it->second);
      }
    }
  }

  // 2. Each residue's own bonds, from the dictionary.
  for (size_t r = 0; r + 1 < starts.size(); ++r) {
    const int s = starts[r], e = starts[r + 1];
    auto m = resBonds.find(a.resName[s]);
    if (m == resBonds.end()) continue;
    for (int i = s; i < e; ++i) {
      for (int j = i + 1; j < e; ++j) {
        const string& x = a.atomName[i];
        const string& y = a.atomName[j];
        auto bt = m->second.find(x < y ? x + ":" + y : y + ":" + x);
        if (bt != m->second.end()) addBond(i, j, bt->second);
      }
    }
  }

  // 3. The polymer backbone, between consecutive residues of one chain - unless one of the two
  //    atoms is already bonded to the other residue, which a declared link would have made.
  for (size_t r = 0; r + 2 < starts.size(); ++r) {
    const int s1 = starts[r], e1 = starts[r + 1], s2 = starts[r + 1], e2 = starts[r + 2];
    if (a.chainId[s1] != a.chainId[s2]) continue;
    if (a.resId[s2] - a.resId[s1] != 1) continue;
    const string outName = polymerizationAtoms(a.resName[s1], a.chainType[s1]).first;
    const string inName = polymerizationAtoms(a.resName[s2], a.chainType[s2]).second;
    if (outName.empty() || inName.empty()) continue;
    int idxOut = -1, idxIn = -1;
    for (int i = s1; i < e1; ++i) if (a.atomName[i] == outName) { idxOut = i; break; }
    for (int i = s2; i < e2; ++i) if (a.atomName[i] == inName) { idxIn = i; break; }
    if (idxOut < 0 || idxIn < 0) continue;
    bool linked = false;
    for (const Bond& b : bonds) {
      if (b.type == tables::BT_COORDINATION) continue;
      if ((b.u == idxOut && atomToRes[b.v] == (int)r + 1) ||
          (b.v == idxOut && atomToRes[b.u] == (int)r + 1) ||
          (b.u == idxIn && atomToRes[b.v] == (int)r) ||
          (b.v == idxIn && atomToRes[b.u] == (int)r)) {
        linked = true;
        break;
      }
    }
    if (!linked) addBond(idxOut, idxIn, tables::BT_SINGLE);
  }

  // 4. Anything touching a metal is a COORDINATION bond, whatever it was.
  for (Bond& b : bonds) {
    string eu = a.element[b.u], ev = a.element[b.v];
    for (char& ch : eu) ch = (char)toupper((unsigned char)ch);
    for (char& ch : ev) ch = (char)toupper((unsigned char)ch);
    if (tables::METAL_ELEMENTS.count(eu) || tables::METAL_ELEMENTS.count(ev)) {
      b.type = tables::BT_COORDINATION;
    }
  }
  return bonds;
}

/**
 * `computeIdAndEntityAnnotations`: the pn-unit, molecule and entity ids, and the instance ids.
 *
 * Three levels, each a connected-component pass over the bonds plus a Weisfeiler-Lehman hash that
 * decides which instances are the SAME entity - two chains with the same residues in the same
 * arrangement get one id, which is what makes a homodimer a homodimer.
 *
 * 🔴 COORDINATION BONDS ARE EXCLUDED EVERYWHERE HERE. A metal touching two chains does not make
 * them one molecule, and leaving them in merges a protein with whatever its zinc also touches.
 *
 * 🔴 AND THE SORT IS NUMERIC WHERE THE COLUMN IS. `molecule_id` and `res_id` are numbers and
 * `chain_id` and `pn_unit_id` are strings; sorting the numeric ones as text puts 10 before 9 and
 * renumbers every entity past the ninth.
 */
struct Ids {
  vector<string> pnUnitId, chainIid, pnUnitIid;
  vector<int> moleculeId, chainEntity, pnUnitEntity, moleculeEntity, moleculeIid;
};

inline Ids computeIds(const AtomSite& a, const vector<Bond>& bonds) {
  const int n = (int)a.n;
  Ids out;
  if (n == 0) return out;
  vector<std::pair<int, int>> links;
  for (const Bond& b : bonds) {
    if (b.type != tables::BT_COORDINATION) links.push_back({b.u, b.v});
  }

  // 1. pn_unit_id: the chain, with connected NON-POLYMER chains merged into one comma-joined name.
  out.pnUnitId = a.chainId;
  {
    vector<string> nonPoly;
    for (int i = 0; i < n; ++i) {
      if (!a.isPolymer[i] &&
          std::find(nonPoly.begin(), nonPoly.end(), a.chainId[i]) == nonPoly.end()) {
        nonPoly.push_back(a.chainId[i]);
      }
    }
    std::sort(nonPoly.begin(), nonPoly.end());
    if (!nonPoly.empty()) {
      std::map<string, int> idx;
      for (size_t k = 0; k < nonPoly.size(); ++k) idx[nonPoly[k]] = (int)k;
      vector<std::set<int>> adj(nonPoly.size());
      for (size_t k = 0; k < nonPoly.size(); ++k) adj[k].insert((int)k);
      for (const auto& [u, v] : links) {
        if (a.isPolymer[u] || a.isPolymer[v]) continue;
        auto cu = idx.find(a.chainId[u]), cv = idx.find(a.chainId[v]);
        if (cu == idx.end() || cv == idx.end() || cu->second == cv->second) continue;
        adj[cu->second].insert(cv->second);
        adj[cv->second].insert(cu->second);
      }
      vector<uint8_t> seen(nonPoly.size(), 0);
      for (size_t i = 0; i < nonPoly.size(); ++i) {
        if (seen[i]) continue;
        vector<string> comp;
        vector<int> queue{(int)i};
        seen[i] = 1;
        for (size_t q = 0; q < queue.size(); ++q) {
          comp.push_back(nonPoly[queue[q]]);
          for (int nb : adj[queue[q]]) {
            if (!seen[nb]) { seen[nb] = 1; queue.push_back(nb); }
          }
        }
        vector<string> sorted = comp;
        std::sort(sorted.begin(), sorted.end());
        string joined;
        for (size_t k = 0; k < sorted.size(); ++k) joined += (k ? "," : "") + sorted[k];
        std::set<string> members(comp.begin(), comp.end());
        for (int k = 0; k < n; ++k) if (members.count(a.chainId[k])) out.pnUnitId[k] = joined;
      }
    }
  }

  // 2. molecule_id: the connected components of the pn-units.
  {
    vector<string> units;
    for (int i = 0; i < n; ++i) {
      if (std::find(units.begin(), units.end(), out.pnUnitId[i]) == units.end()) {
        units.push_back(out.pnUnitId[i]);
      }
    }
    std::sort(units.begin(), units.end());
    std::map<string, int> idx;
    for (size_t k = 0; k < units.size(); ++k) idx[units[k]] = (int)k;
    vector<std::set<int>> adj(units.size());
    for (size_t k = 0; k < units.size(); ++k) adj[k].insert((int)k);
    for (const auto& [u, v] : links) {
      const int pu = idx[out.pnUnitId[u]], pv = idx[out.pnUnitId[v]];
      if (pu == pv) continue;
      adj[pu].insert(pv);
      adj[pv].insert(pu);
    }
    std::map<string, int> unitToMol;
    vector<uint8_t> seen(units.size(), 0);
    int counter = 0;
    for (size_t i = 0; i < units.size(); ++i) {
      if (seen[i]) continue;
      vector<int> queue{(int)i};
      seen[i] = 1;
      for (size_t q = 0; q < queue.size(); ++q) {
        unitToMol[units[queue[q]]] = counter;
        for (int nb : adj[queue[q]]) {
          if (!seen[nb]) { seen[nb] = 1; queue.push_back(nb); }
        }
      }
      ++counter;
    }
    out.moleculeId.assign(n, 0);
    for (int i = 0; i < n; ++i) out.moleculeId[i] = unitToMol[out.pnUnitId[i]];
  }

  // 3. The entity at each level: a Weisfeiler-Lehman hash over the level below.
  auto levelEntities = [&](const vector<string>& level, bool levelNumeric,
                           const vector<string>& lowerId, bool lowerNumeric,
                           const vector<string>& lowerEntity, bool includeInterBonds) {
    auto less = [](bool numeric) {
      return [numeric](const string& x, const string& y) {
        return numeric ? atof(x.c_str()) < atof(y.c_str()) : x < y;
      };
    };
    vector<string> instances;
    for (int i = 0; i < n; ++i) {
      if (std::find(instances.begin(), instances.end(), level[i]) == instances.end()) {
        instances.push_back(level[i]);
      }
    }
    std::sort(instances.begin(), instances.end(), less(levelNumeric));
    std::map<string, int> hashToEntity;
    int next = 0;
    vector<int> result(n, 0);
    for (const string& inst : instances) {
      vector<int> members;
      for (int i = 0; i < n; ++i) if (level[i] == inst) members.push_back(i);
      std::map<int, int> toLocal;
      for (size_t l = 0; l < members.size(); ++l) toLocal[members[l]] = (int)l;
      vector<string> nodes;
      for (int g : members) {
        if (std::find(nodes.begin(), nodes.end(), lowerId[g]) == nodes.end()) {
          nodes.push_back(lowerId[g]);
        }
      }
      std::sort(nodes.begin(), nodes.end(), less(lowerNumeric));
      std::map<string, int> nodeIdx;
      for (size_t k = 0; k < nodes.size(); ++k) nodeIdx[nodes[k]] = (int)k;
      vector<string> labels(nodes.size());
      for (size_t k = 0; k < nodes.size(); ++k) {
        for (int g : members) {
          if (lowerId[g] == nodes[k]) { labels[k] = lowerEntity[g]; break; }
        }
      }
      vector<std::set<int>> adj(nodes.size());
      for (size_t k = 0; k < nodes.size(); ++k) adj[k].insert((int)k);
      vector<string> inter;
      for (const auto& [u, v] : links) {
        if (!toLocal.count(u) || !toLocal.count(v)) continue;
        const int nu = nodeIdx[lowerId[u]], nv = nodeIdx[lowerId[v]];
        if (nu == nv) continue;
        adj[nu].insert(nv);
        adj[nv].insert(nu);
        if (includeInterBonds) {
          const string t1 = lowerEntity[u] + ":" + std::to_string(a.resId[u]) + ":" +
                            a.resName[u] + ":" + a.atomName[u];
          const string t2 = lowerEntity[v] + ":" + std::to_string(a.resId[v]) + ":" +
                            a.resName[v] + ":" + a.atomName[v];
          inter.push_back(t1 < t2 ? t1 + "|" + t2 : t2 + "|" + t1);
        }
      }
      vector<string> cur = labels;
      for (int iter = 0; iter < 3; ++iter) {
        vector<string> nextLabels(nodes.size());
        for (size_t k = 0; k < nodes.size(); ++k) {
          vector<string> nb;
          for (int j : adj[k]) nb.push_back(cur[j]);
          std::sort(nb.begin(), nb.end());
          string joined;
          for (size_t q = 0; q < nb.size(); ++q) joined += (q ? "," : "") + nb[q];
          nextLabels[k] = cur[k] + "(" + joined + ")";
        }
        cur = nextLabels;
      }
      vector<string> sortedLabels = cur;
      std::sort(sortedLabels.begin(), sortedLabels.end());
      string wl;
      for (size_t k = 0; k < sortedLabels.size(); ++k) wl += (k ? ";" : "") + sortedLabels[k];
      std::map<string, int> counts;
      for (const string& l : labels) counts[l] += 1;
      string countPart;
      bool first = true;
      for (const auto& [label, c] : counts) {
        countPart += (first ? "" : ",") + label + ":" + std::to_string(c);
        first = false;
      }
      std::sort(inter.begin(), inter.end());
      string interPart;
      for (size_t k = 0; k < inter.size(); ++k) interPart += (k ? "&" : "") + inter[k];
      const string hash =
          wl + "_" + std::to_string(nodes.size()) + "_" + countPart + "_" + interPart;
      auto it = hashToEntity.find(hash);
      int eid;
      if (it == hashToEntity.end()) { eid = next++; hashToEntity[hash] = eid; }
      else eid = it->second;
      for (int g : members) result[g] = eid;
    }
    return result;
  };

  auto asText = [](const vector<int>& v) {
    vector<string> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = std::to_string(v[i]);
    return out;
  };
  const vector<string> resIdText = asText(a.resId);
  const vector<string> moleculeIdText = asText(out.moleculeId);
  out.chainEntity = levelEntities(a.chainId, false, resIdText, true, a.resName, false);
  out.pnUnitEntity =
      levelEntities(out.pnUnitId, false, a.chainId, false, asText(out.chainEntity), true);
  out.moleculeEntity =
      levelEntities(moleculeIdText, true, out.pnUnitId, false, asText(out.pnUnitEntity), true);

  // The instance ids. `transformation_id` is "1" throughout for every file this reads.
  out.chainIid.resize(n);
  out.pnUnitIid.resize(n);
  vector<string> molIid(n);
  for (int i = 0; i < n; ++i) {
    out.chainIid[i] = a.chainId[i] + "_1";
    string joined;
    size_t from = 0;
    const string& p = out.pnUnitId[i];
    while (from <= p.size()) {
      const size_t comma = p.find(',', from);
      joined += (from ? "," : "") + p.substr(from, comma == string::npos ? string::npos : comma - from) + "_1";
      if (comma == string::npos) break;
      from = comma + 1;
    }
    out.pnUnitIid[i] = joined;
    molIid[i] = std::to_string(out.moleculeId[i]) + "1";
  }
  vector<string> uniqueMol;
  for (const string& m : molIid) {
    if (std::find(uniqueMol.begin(), uniqueMol.end(), m) == uniqueMol.end()) uniqueMol.push_back(m);
  }
  std::sort(uniqueMol.begin(), uniqueMol.end());
  std::map<string, int> molMap;
  for (size_t k = 0; k < uniqueMol.size(); ++k) molMap[uniqueMol[k]] = (int)k;
  out.moleculeIid.assign(n, 0);
  for (int i = 0; i < n; ++i) out.moleculeIid[i] = molMap[molIid[i]];
  return out;
}

}  // namespace cif
}  // namespace rfd4p
