// An AlphaFold 3 job, read as the page reads one: web/job-json.js jobFromJson (both dialects, every refusal in
// its own words), web/entities.js entitiesProblem / expandEntities / parseContact, web/sequence.js cleanSequence.
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd.h"
#include "chem.h"
#include "json.h"
#include "jsnum.h"

namespace lf {

struct Refusal : std::runtime_error { using std::runtime_error::runtime_error; };
[[noreturn]] inline void refuse(const std::string& m) { throw Refusal(m); }

// JavaScript's String(value) and Number(value) for a JSON value
inline std::string jsString(const Json* v) {
  if (v == nullptr) return "undefined";
  switch (v->kind) {
    case Json::Null: return "null";
    case Json::Bool: return v->b ? "true" : "false";
    case Json::Number: return jsNumber(v->n);
    case Json::String: return v->s;
    case Json::Array: {
      std::string out;
      for (size_t i = 0; i < v->a.size(); ++i) out += (i ? "," : "") + (v->a[i].isNull() ? std::string() : jsString(&v->a[i]));
      return out;
    }
    default: return "[object Object]";
  }
}
inline double jsToNumber(const Json* v) {
  if (v == nullptr) return NAN;
  switch (v->kind) {
    case Json::Null: return 0;
    case Json::Bool: return v->b ? 1 : 0;
    case Json::Number: return v->n;
    case Json::String: {
      std::string s = trimWs(v->s);
      if (s.empty()) return 0;
      char* end = nullptr;
      double d = std::strtod(s.c_str(), &end);
      return (end && *end == 0) ? d : NAN;
    }
    case Json::Array: return v->a.empty() ? 0 : (v->a.size() == 1 ? jsToNumber(&v->a[0]) : NAN);
    default: return NAN;
  }
}
inline bool jsIsInteger(double d) { return std::isfinite(d) && std::floor(d) == d; }

// ---------------------------------------------------------------- web/sequence.js
inline bool startsTrimmed(const std::string& line, char c) {
  size_t i = 0;
  while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
  return i < line.size() && line[i] == c;
}
inline std::string cleanSequence(const std::string& text) {
  std::vector<std::string> lines;
  for (auto& l : splitLines(text)) if (!startsTrimmed(l, ';')) lines.push_back(l);
  size_t start = lines.size();
  for (size_t i = 0; i < lines.size(); ++i) if (startsTrimmed(lines[i], '>')) { start = i; break; }
  std::vector<std::string> body(start == lines.size() ? lines.begin() : lines.begin() + start + 1, lines.end());
  size_t end = body.size();
  for (size_t i = 0; i < body.size(); ++i) if (startsTrimmed(body[i], '>')) { end = i; break; }
  std::string out;
  for (size_t i = 0; i < end; ++i)
    for (char c : body[i])
      if (!(std::isspace((unsigned char)c) || std::isdigit((unsigned char)c) || c == '.' || c == '-' || c == '*'))
        out += (char)std::toupper((unsigned char)c);
  return out;
}
inline std::string joinSet(const std::vector<char>& bad, const std::string& sep) {
  std::string out;
  for (size_t i = 0; i < bad.size(); ++i) out += (i ? sep : "") + std::string(1, bad[i]);
  return out;
}
inline std::optional<std::string> sequenceProblem(const std::string& s) {
  if (s.empty()) return "Enter a protein sequence";
  const std::string ok = "ARNDCQEGHILKMFPSTWYVX";
  std::vector<char> bad;
  for (char c : s) if (ok.find(c) == std::string::npos && std::find(bad.begin(), bad.end(), c) == bad.end()) bad.push_back(c);
  if (bad.empty()) return std::nullopt;
  return bad.size() == 1 ? std::string(1, bad[0]) + " is not one of the twenty amino acids"
                         : joinSet(bad, ", ") + " are not among the twenty amino acids";
}
inline std::optional<std::string> nucleicProblem(const std::string& s, const std::string& kind) {
  std::string bases = kind == "dna" ? "ACGT" : "ACGU";
  std::string label = upper(kind);
  if (s.empty()) return "Enter a " + label + " sequence";
  std::vector<char> bad;
  for (char c : s) if (bases.find(c) == std::string::npos && std::find(bad.begin(), bad.end(), c) == bad.end()) bad.push_back(c);
  if (bad.empty()) return std::nullopt;
  std::string listed = joinSet(std::vector<char>(bases.begin(), bases.end()), ", ");
  char swapped = kind == "rna" ? 'T' : 'U';
  if (bad.size() == 1 && bad[0] == swapped) {
    std::string instead = kind == "rna" ? "U (uracil)" : "T (thymine)";
    return std::string(1, swapped) + " is not " + (kind == "rna" ? "an" : "a") + " " + label + " base - " + label + " uses "
           + instead + ". Change the entity type if this is " + (kind == "rna" ? "DNA" : "RNA");
  }
  return bad.size() == 1 ? std::string(1, bad[0]) + " is not one of " + listed : joinSet(bad, ", ") + " are not among " + listed;
}

// ---------------------------------------------------------------- web/entities.js
struct Modification { std::string code; double position; };
struct OptText { bool present = false; std::string text; };     // null / a string ("" is the asked-for none)
// a row's template: "search", or "upload" - an inline mmCIF and, where the job gives one, its residue mapping
struct EntityTemplate { std::string kind; std::string text; bool hasMapping = false; std::vector<std::pair<int, int>> mapping; };
struct Entity {
  std::string type, value;
  double copies = 1;
  std::vector<Modification> modifications;
  std::vector<std::string> ids;
  bool hasInlineMsa = false;
  OptText unpairedMsa, pairedMsa;
  std::optional<EntityTemplate> templ;
};

struct ContactEnd { std::string chain; int residue; bool hasAtom; std::string atom; };
inline std::optional<std::pair<ContactEnd, ContactEnd>> parseContact(const std::string& value) {
  std::string text = trimWs(value);
  if (text.empty()) return std::nullopt;
  std::vector<std::string> sides;
  size_t s = 0, d;
  while ((d = text.find('-', s)) != std::string::npos) { sides.push_back(text.substr(s, d - s)); s = d + 1; }
  sides.push_back(text.substr(s));
  if (sides.size() != 2) return std::nullopt;
  static const std::regex end(R"(^\s*([A-Za-z]+)\s*(\d+)\s*(?::\s*([A-Za-z0-9']+)\s*)?$)");
  auto one = [&](const std::string& side) -> std::optional<ContactEnd> {
    std::smatch m;
    if (!std::regex_match(side, m, end)) return std::nullopt;
    ContactEnd e{upper(m[1]), std::stoi(m[2]), m[3].matched, m[3].matched ? upper(m[3]) : ""};
    return e;
  };
  auto from = one(sides[0]), to = one(sides[1]);
  if (!from || !to) return std::nullopt;
  return std::make_pair(*from, *to);
}

struct CommonModification { const char* code; char parent; };
inline const std::vector<CommonModification>& commonModifications() {
  static const std::vector<CommonModification> m = {{"SEP", 'S'}, {"TPO", 'T'}, {"PTR", 'Y'}, {"HYP", 'P'}, {"MLY", 'K'},
    {"M3L", 'K'}, {"ALY", 'K'}, {"KCX", 'K'}, {"CSO", 'C'}, {"CME", 'C'}, {"OCS", 'C'}, {"SNC", 'C'}, {"NEP", 'H'},
    {"AGM", 'R'}, {"PCA", 'E'}};
  return m;
}
inline bool ccdShaped(const std::string& code) {
  if (code.empty() || code.size() > 5) return false;
  for (char c : code) if (!std::isalnum((unsigned char)c) || std::islower((unsigned char)c)) return false;
  return true;
}
inline std::optional<std::string> modificationProblem(const Modification& m, const std::string& sequence) {
  std::string code = upper(trimWs(m.code));
  if (code.empty()) return "Choose a modification";
  if (!ccdShaped(code)) return "A CCD code is 1-5 letters or digits, like SEP or PTR";
  if (code == "MSE") return "MSE is not supported yet: AF3 treats it as methionine rather than as a modified residue";
  if (!jsIsInteger(m.position) || m.position < 1) return "A position is a whole number, counting from 1";
  if (m.position > (double)sequence.size())
    return "Position " + jsNumber(m.position) + " is past the end of a " + std::to_string(sequence.size()) + "-residue sequence";
  char parent = sequence[(size_t)m.position - 1];
  for (auto& known : commonModifications())
    if (code == known.code && parent != known.parent)
      return code + " modifies " + std::string(1, known.parent) + ", but position " + jsNumber(m.position) + " is " + std::string(1, parent);
  return std::nullopt;
}
// a SMILES ligand's heavy-atom count, or a refusal in the parser's words (set by the SMILES module)
inline int SMILES_ATOM_COUNT(const std::string& smiles) { return (int)chem::parseSmiles(smiles).atoms.size(); }
inline std::optional<std::string> entityProblem(const Entity& e) {
  static const std::vector<std::string> types = {"protein", "dna", "rna", "ligand", "smiles", "contact"};
  if (std::find(types.begin(), types.end(), e.type) == types.end()) return "Unknown entity type " + e.type;
  if (e.type == "contact") {
    auto parsed = parseContact(e.value);
    if (!parsed) return "A contact is two residues, as A12:SG - B1:C25 (the atoms optional)";
    if (parsed->first.chain == parsed->second.chain && parsed->first.residue == parsed->second.residue)
      return "A contact joins two different residues";
    return std::nullopt;
  }
  if (!jsIsInteger(e.copies) || e.copies < 1) return "Copies must be a whole number, at least 1";
  if (e.copies > 20) return "At most 20 copies";
  std::string value = trimWs(e.value);
  if (value.empty()) {
    if (e.type == "ligand") return "Enter a CCD code";
    if (e.type == "smiles") return "Enter a SMILES string";
    return e.type == "protein" ? std::string("Enter a protein sequence") : "Enter a " + upper(e.type) + " sequence";
  }
  if (e.type == "ligand") {
    size_t s = 0, d;
    std::vector<std::string> codes;
    while ((d = value.find(',', s)) != std::string::npos) { codes.push_back(value.substr(s, d - s)); s = d + 1; }
    codes.push_back(value.substr(s));
    for (auto& c : codes) {
      std::string t = trimWs(c);
      bool ok = !t.empty() && t.size() <= 5;
      for (char ch : t) if (!std::isalnum((unsigned char)ch)) ok = false;
      if (!ok) return "A CCD code is 1-5 letters or digits, like HEM or ATP";
    }
    return std::nullopt;
  }
  if (e.type == "smiles") {
    try {
      int atoms = SMILES_ATOM_COUNT(value);
      if (atoms > 150) return "That is " + std::to_string(atoms) + " heavy atoms; at most 150 here";
    } catch (const std::exception& error) {
      return std::string(error.what());
    }
    return std::nullopt;
  }
  if (value.find(':') != std::string::npos) return "One sequence per entity - use Add entity for another chain";
  std::string cleaned = cleanSequence(value);
  bool nucleic = e.type == "dna" || e.type == "rna";
  auto fault = nucleic ? nucleicProblem(cleaned, e.type) : sequenceProblem(cleaned);
  if (fault) return fault;
  std::set<double> seen;
  for (auto& m : e.modifications) {
    std::string code = upper(trimWs(m.code));
    if (nucleic)
      for (auto& known : commonModifications())
        if (code == known.code) return code + " is a modified amino acid, not a base of a " + upper(e.type) + " chain";
    auto f = modificationProblem(m, cleaned);
    if (f) return f;
    if (seen.count(m.position)) return "Two modifications on residue " + jsNumber(m.position);
    seen.insert(m.position);
  }
  return std::nullopt;
}
inline std::optional<std::string> templateProblem(const Entity& e) {
  if (!e.templ) return std::nullopt;
  if (e.type != "protein") return "Only a protein chain can take a template";
  return std::nullopt;
}
inline std::optional<std::string> entitiesProblem(const std::vector<Entity>& entities) {
  if (entities.empty()) return "Add an entity to fold";
  for (size_t i = 0; i < entities.size(); ++i) {
    auto p = entityProblem(entities[i]);
    if (!p) p = templateProblem(entities[i]);
    if (!p) continue;
    return entities.size() == 1 ? *p : "Entity " + std::to_string(i + 1) + ": " + *p;
  }
  return std::nullopt;
}

inline std::string ligandName(size_t index) {
  if (index == 0) return "LIG";
  if (index < 9) return "LG" + std::to_string(index + 1);
  if (index < 99) return "L" + std::to_string(index + 1);
  throw std::runtime_error("at most 99 distinct SMILES ligands in one job");
}

struct LigandRef {                 // expandEntities' ligandCodes entry
  enum Kind { Code, Chain, Smiles } kind = Code;
  std::string code;                // a CCD code, or a SMILES ligand's given name
  std::vector<std::string> codes;  // a chain of components
  std::string smiles;
};
struct ChainModification { int chain; double position; std::string code; };
struct Expanded {
  std::vector<std::string> chains, chainKinds;
  std::vector<LigandRef> ligands;
  std::vector<ChainModification> modifications;
  std::vector<std::pair<int, EntityTemplate>> templates;
  std::vector<DeclaredBond> bonds;
  std::string sequence;
};

inline Expanded expandEntities(const std::vector<Entity>& entities) {
  if (auto p = entitiesProblem(entities)) throw Refusal(*p);
  Expanded x;
  std::vector<std::pair<std::string, std::string>> smilesCodes;
  for (auto& e : entities) {
    if (e.type == "contact") continue;
    for (int copy = 0; copy < (int)e.copies; ++copy) {
      if (e.type == "protein" || e.type == "dna" || e.type == "rna") {
        for (auto& m : e.modifications) x.modifications.push_back({(int)x.chains.size(), m.position, upper(trimWs(m.code))});
        x.chainKinds.push_back(e.type);
        if (e.templ && e.type == "protein") x.templates.push_back({(int)x.chains.size(), *e.templ});
        x.chains.push_back(cleanSequence(e.value));
      } else if (e.type == "smiles") {
        std::string text = trimWs(e.value);
        std::string code;
        for (auto& [t, c] : smilesCodes) if (t == text) code = c;
        if (code.empty()) { code = ligandName(smilesCodes.size()); smilesCodes.push_back({text, code}); }
        LigandRef r; r.kind = LigandRef::Smiles; r.smiles = text; r.code = code;
        x.ligands.push_back(r);
      } else {
        std::string v = upper(trimWs(e.value));
        std::vector<std::string> codes;
        size_t s = 0, d;
        while ((d = v.find(',', s)) != std::string::npos) { codes.push_back(trimWs(v.substr(s, d - s))); s = d + 1; }
        codes.push_back(trimWs(v.substr(s)));
        LigandRef r;
        if (codes.size() == 1) { r.kind = LigandRef::Code; r.code = codes[0]; }
        else { r.kind = LigandRef::Chain; r.codes = codes; }
        x.ligands.push_back(r);
      }
    }
  }
  for (auto& e : entities) {
    if (e.type != "contact") continue;
    auto parsed = parseContact(e.value);
    if (!parsed) continue;
    auto asymOf = [](const std::string& letters) {
      int total = 0;
      for (char c : letters) total = total * 26 + (c - 64);
      return total - 1;
    };
    auto end = [&](const ContactEnd& side) { return BondEnd{asymOf(side.chain), side.residue, side.hasAtom, side.atom}; };
    x.bonds.push_back({end(parsed->first), end(parsed->second)});
  }
  for (size_t i = 0; i < x.chains.size(); ++i) x.sequence += (i ? ":" : "") + x.chains[i];
  return x;
}

// ---------------------------------------------------------------- web/job-json.js
struct Job {
  std::string name;
  bool hasSeed = false;
  double seed = 0;
  std::vector<Entity> entities;
  std::vector<std::string> notes;
  bool hasAlignments = false;
  std::vector<OptText> unpaired, paired;      // one a polymer chain copy
  std::string userCcd;
};

inline std::string ptmCode(const Json* type, const std::string& where) {
  std::string raw = upper(trimWs(Json::absent(type) ? std::string() : jsString(type)));
  std::string code = raw.compare(0, 4, "CCD_") == 0 ? raw.substr(4) : raw;
  if (!ccdShaped(code)) refuse(where + ": \"" + (Json::absent(type) ? std::string() : jsString(type)) + "\" is not a CCD code");
  return code;
}

// A template the job carries inline (AF3's `templates` of a protein chain): the exporter lifts every one out
// before the page's reader sees the job - up to four a chain, chain i's k-th in slot k - and builds them itself
struct JobTemplate { std::string text; int chain = 0; std::string label; bool hasMapping = false; std::vector<std::pair<int, int>> mapping; };

inline Job jobFromJson(const std::string& text, std::vector<std::vector<JobTemplate>>* lifted = nullptr) {
  Json parsed;
  try { parsed = parseJson(text); } catch (const std::exception& e) { throw Refusal(std::string("that is not JSON: ") + e.what()); }
  if (lifted != nullptr) {          // cuda/af3/export-model.mjs: jobTemplates, then `delete body.templates`
    Json* first = parsed.isArray() ? (parsed.a.empty() ? nullptr : &parsed.a[0]) : &parsed;
    Json* seqs = nullptr;
    if (first != nullptr && first->isObject()) for (auto& [k, v] : first->o) if (k == "sequences") seqs = &v;
    int polymerCopy = 0;
    if (seqs != nullptr && seqs->isArray()) for (auto& entry : seqs->a) {
      if (!entry.isObject() || entry.o.empty()) continue;
      auto& [kind, body] = entry.o[0];
      if ((kind != "protein" && kind != "rna" && kind != "dna") || !body.isObject()) continue;
      const Json* id = body.get("id");
      int copies = id != nullptr && id->isArray() ? (int)id->a.size() : 1;
      const Json* list = body.get("templates");
      for (int c = 0; c < copies; ++c) {
        if (list != nullptr && list->isArray())
          for (size_t k = 0; k < std::min<size_t>(4, list->a.size()); ++k) {
            const Json& t = list->a[k];
            const Json* mmcif = t.get("mmcif");
            if (mmcif == nullptr || !mmcif->isString())
              throw std::runtime_error("template " + std::to_string(k) + " of chain " + std::to_string(polymerCopy) + ": give the mmCIF inline");
            const Json* q = t.get("queryIndices");
            const Json* ti = t.get("templateIndices");
            if ((q == nullptr) != (ti == nullptr) || (q && (!q->isArray() || !ti->isArray() || q->a.size() != ti->a.size())))
              throw std::runtime_error("template " + std::to_string(k) + " of chain " + std::to_string(polymerCopy)
                                       + ": queryIndices and templateIndices are two lists of one length");
            JobTemplate jt;
            jt.text = mmcif->s; jt.chain = polymerCopy; jt.label = "job template " + std::to_string(k);
            if (q) {
              jt.hasMapping = true;
              for (size_t i = 0; i < q->a.size(); ++i) jt.mapping.push_back({(int)jsToNumber(&q->a[i]), (int)jsToNumber(&ti->a[i])});
            }
            if (lifted->size() <= k) lifted->resize(k + 1);
            (*lifted)[k].push_back(std::move(jt));
          }
        ++polymerCopy;
      }
      body.o.erase(std::remove_if(body.o.begin(), body.o.end(), [](auto& kv) { return kv.first == "templates"; }), body.o.end());
    }
  }
  std::vector<const Json*> jobs;
  if (parsed.isArray()) for (auto& j : parsed.a) jobs.push_back(&j);
  else jobs.push_back(&parsed);
  if (jobs.empty()) refuse("that file holds no jobs");
  Job out;
  if (jobs.size() > 1) out.notes.push_back(std::to_string(jobs.size()) + " jobs in the file; loaded the first");
  static const Json emptyObject = [] { Json j; j.kind = Json::Object; return j; }();
  const Json& job = jobs[0]->isObject() ? *jobs[0] : emptyObject;
  if (!Json::absent(job.get("userCCDPath")))
    refuse("userCCDPath points at a file beside the JSON, which this reader cannot open - inline it as userCCD");
  if (!Json::absent(job.get("userCCD")) && !job.get("userCCD")->isString()) refuse("userCCD is mmCIF text");
  const Json* dialectValue = job.get("dialect");
  std::string dialectText = Json::absent(dialectValue) ? "" : jsString(dialectValue);
  if (!Json::absent(dialectValue) && dialectText != "alphafoldserver" && dialectText != "alphafold3")
    refuse("dialect \"" + dialectText + "\" is not one this page reads (alphafoldserver, alphafold3)");
  std::string dialect = dialectText == "alphafold3" ? "alphafold3" : "alphafoldserver";
  {
    bool hasDialect = !Json::absent(dialectValue), hasVersion = !Json::absent(job.get("version"));
    if (hasDialect != hasVersion)
      refuse(std::string("a job carries both `dialect` and `version` or neither, and this one has only `")
             + (hasDialect ? "dialect" : "version") + "`");
    if (hasVersion) {
      double v = jsToNumber(job.get("version"));
      std::vector<double> known = dialect == "alphafoldserver" ? std::vector<double>{1, 3} : std::vector<double>{1, 2, 3, 4};
      if (std::find(known.begin(), known.end(), v) == known.end()) {
        std::string list;
        for (size_t i = 0; i < known.size(); ++i) list += (i ? ", " : "") + jsNumber(known[i]);
        refuse("version " + jsString(job.get("version")) + " of the " + dialect + " dialect is not one this page reads (" + list + ")");
      }
    }
  }
  const Json* entries = job.get("sequences");
  if (entries == nullptr || !entries->isArray() || entries->a.empty()) refuse("no `sequences` in that job");
  bool sawEmpty = false, singleSequence = false;

  static const std::map<std::string, std::pair<std::string, std::string>> ENTRY_TYPES = {
    {"proteinChain", {"protein", "alphafoldserver"}}, {"dnaSequence", {"dna", "alphafoldserver"}},
    {"rnaSequence", {"rna", "alphafoldserver"}}, {"ion", {"ligand", "alphafoldserver"}}, {"ligand", {"ligand", ""}},
    {"protein", {"protein", "alphafold3"}}, {"dna", {"dna", "alphafold3"}}, {"rna", {"rna", "alphafold3"}}};
  static const std::map<std::string, std::map<std::string, std::vector<std::string>>> ALLOWED = {
    {"alphafoldserver", {{"protein", {"sequence", "glycans", "modifications", "count", "maxTemplateDate", "useStructureTemplate"}},
                         {"dna", {"sequence", "modifications", "count"}}, {"rna", {"sequence", "modifications", "count"}},
                         {"ligand", {"ligand", "ion", "count"}}}},
    {"alphafold3", {{"protein", {"id", "sequence", "modifications", "description", "unpairedMsa", "unpairedMsaPath", "pairedMsa",
                                 "pairedMsaPath", "templates"}},
                    {"dna", {"id", "sequence", "modifications", "description"}},
                    {"rna", {"id", "sequence", "modifications", "description", "unpairedMsa", "unpairedMsaPath"}},
                    {"ligand", {"id", "ccdCodes", "smiles", "description"}}}}};
  auto checkKeys = [](const Json& body, const std::vector<std::string>& allowed, const std::string& where) {
    std::vector<std::string> unknown;
    for (auto& [k, v] : body.o) if (std::find(allowed.begin(), allowed.end(), k) == allowed.end()) unknown.push_back(k);
    if (!unknown.empty()) {
      std::string u, a;
      for (size_t i = 0; i < unknown.size(); ++i) u += (i ? ", " : "") + unknown[i];
      for (size_t i = 0; i < allowed.size(); ++i) a += (i ? ", " : "") + allowed[i];
      refuse(where + ": " + u + " is not a field of this entry - AlphaFold 3 takes " + a);
    }
  };
  auto idsOf = [](const Json& body) {
    std::vector<std::string> ids;
    const Json* id = body.get("id");
    if (id != nullptr && id->isArray()) for (auto& v : id->a) ids.push_back(jsString(&v));
    else if (id != nullptr) ids.push_back(jsString(id));
    return ids;
  };
  auto copiesOf = [](const Json& body, const std::string& where) {
    const Json* count = body.get("count");
    if (count != nullptr) {
      double n = jsToNumber(count);
      if (!jsIsInteger(n) || n < 1) refuse(where + ": count " + jsString(count));
      return n;
    }
    const Json* id = body.get("id");
    if (id != nullptr && id->isArray()) return (double)std::max<size_t>(1, id->a.size());
    return 1.0;
  };

  for (size_t index = 0; index < entries->a.size(); ++index) {
    const Json& entry = entries->a[index];
    size_t keys = entry.isObject() ? entry.o.size() : 0;
    if (keys != 1) refuse("sequences[" + std::to_string(index) + "] has " + std::to_string(keys) + " keys, and an entry names one kind of chain");
    const std::string& key = entry.o[0].first;
    auto type = ENTRY_TYPES.find(key);
    if (type == ENTRY_TYPES.end()) refuse("sequences[" + std::to_string(index) + "]: \"" + key + "\" is not a chain kind this page reads");
    static const Json emptyBody = [] { Json j; j.kind = Json::Object; return j; }();
    const Json& body = entry.o[0].second.isObject() ? entry.o[0].second : emptyBody;
    std::string where = "sequences[" + std::to_string(index) + "] (" + key + ")";
    std::string entryDialect = type->second.second.empty()
      ? ((body.get("ligand") != nullptr || body.get("ion") != nullptr) ? "alphafoldserver" : "alphafold3") : type->second.second;
    checkKeys(body, ALLOWED.at(entryDialect).at(type->second.first), where);
    if (!Json::absent(body.get("glycans"))) refuse(where + ": `glycans` is not supported in this dialect, upstream included");
    if (!Json::absent(body.get("maxTemplateDate")))
      refuse(where + ": `maxTemplateDate` chooses which template is found, and this page has no such control");
    Entity e;
    e.copies = copiesOf(body, where);
    e.ids = idsOf(body);
    if (type->second.first == "ligand") {
      if (!Json::absent(body.get("smiles"))) {
        if (!Json::absent(body.get("ccdCodes")))
          refuse(where + ": a ligand with both `smiles` and `ccdCodes` names itself twice, and this page cannot tell which was meant");
        std::string smiles = trimWs(jsString(body.get("smiles")));
        if (smiles.empty()) refuse(where + ": an empty `smiles`");
        try { SMILES_ATOM_COUNT(smiles); } catch (const std::exception& error) { refuse(where + ": `smiles` " + error.what()); }
        e.type = "smiles"; e.value = smiles;
        out.entities.push_back(e);
        continue;
      }
      const Json* named = body.get("ligand") != nullptr ? body.get("ligand") : body.get("ion");
      std::vector<std::string> list;
      const Json* codes = body.get("ccdCodes");
      if (!Json::absent(codes)) {
        if (codes->isArray()) for (auto& c : codes->a) list.push_back(jsString(&c));
        else list.push_back(jsString(codes));
      } else if (named != nullptr) {
        list.push_back(jsString(named));
      }
      if (list.empty()) refuse(where + ": a ligand with no code");
      std::string code;
      for (size_t i = 0; i < list.size(); ++i) {
        std::string c = upper(trimWs(list[i]));
        if (c.compare(0, 4, "CCD_") == 0) c = c.substr(4);
        code += (i ? "," : "") + c;
      }
      e.type = "ligand"; e.value = code;
      out.entities.push_back(e);
      continue;
    }
    const Json* sequence = body.get("sequence");
    if (sequence == nullptr || !sequence->isString() || trimWs(sequence->s).empty()) refuse(where + ": no sequence");
    if (const Json* mods = body.get("modifications"); !Json::absent(mods)) {
      if (mods->isArray()) {
        for (auto& m : mods->a) {
          const Json* t = Json::absent(m.get("ptmType")) ? m.get("modificationType") : m.get("ptmType");
          std::string code = ptmCode(t, where);
          const Json* p = Json::absent(m.get("ptmPosition")) ? m.get("basePosition") : m.get("ptmPosition");
          double position = jsToNumber(Json::absent(p) ? nullptr : p);
          if (Json::absent(p)) position = NAN;
          if (!jsIsInteger(position)) refuse(where + ": modification " + code + " has no whole-number position");
          e.modifications.push_back({code, position});
        }
      }
    }
    for (const char* field : {"unpairedMsaPath", "pairedMsaPath"})
      if (!Json::absent(body.get(field)))
        refuse(where + ": " + field + " points at a file beside the JSON, which a page cannot read - paste the alignment or use the upload box");
    auto read = [&](const char* field) {
      OptText t;
      const Json* v = body.get(field);
      if (Json::absent(v)) return t;
      if (!v->isString()) refuse(where + ": " + field + " is A3M text");
      t.present = true;
      t.text = v->s;
      if (trimWs(v->s).empty()) { sawEmpty = true; t.text = ""; }
      return t;
    };
    e.hasInlineMsa = true;
    e.unpairedMsa = read("unpairedMsa");
    e.pairedMsa = read("pairedMsa");
    if (type->second.first == "protein") {
      const Json* ust = body.get("useStructureTemplate");
      if (ust != nullptr && ust->kind == Json::Bool && ust->b) e.templ = EntityTemplate{"search"};
      else if (const Json* t = body.get("templates"); !Json::absent(t)) {
        // readTemplates
        if (!t->isArray()) refuse(where + ": templates is not a list");
        if (t->a.size() > 1) refuse(where + ": " + std::to_string(t->a.size()) + " templates, and this page takes one per chain");
        if (t->a.size() == 1) {
          const Json& tm = t->a[0];
          if (!Json::absent(tm.get("mmcifPath")))
            refuse(where + ": mmcifPath points at a file beside the JSON, which a page cannot read - inline the mmCIF or pick a template on the row");
          checkKeys(tm, {"mmcif", "mmcifPath", "queryIndices", "templateIndices"}, where + " template");
          EntityTemplate et{"upload"};
          const Json *q = tm.get("queryIndices"), *ti = tm.get("templateIndices");
          if (!Json::absent(q) || !Json::absent(ti)) {
            if (!q || !ti || !q->isArray() || !ti->isArray() || q->a.size() != ti->a.size())
              refuse(where + ": queryIndices and templateIndices are two lists of one length");
            for (auto* list : {q, ti})
              for (auto& v : list->a)
                if (!v.isNumber() || !jsIsInteger(v.n) || v.n < 0) refuse(where + ": queryIndices and templateIndices hold whole numbers from 0");
            et.hasMapping = true;
            for (size_t k = 0; k < q->a.size(); ++k) et.mapping.push_back({(int)q->a[k].n, (int)ti->a[k].n});
          }
          const Json* mm = tm.get("mmcif");
          if (!mm || !mm->isString() || trimWs(mm->s).empty()) refuse(where + ": a template with no mmcif in it");
          et.text = mm->s;
          e.templ = et;
        }
      }
    }
    e.type = type->second.first;
    e.value = upper(trimWs(sequence->s));
    out.entities.push_back(e);
  }
  // ligands after polymers, the order the featuriser builds chains in (Array.prototype.sort is stable)
  std::stable_sort(out.entities.begin(), out.entities.end(), [](const Entity& a, const Entity& b) {
    auto lig = [](const Entity& e) { return e.type == "ligand" || e.type == "smiles"; };
    return !lig(a) && lig(b);
  });
  std::vector<std::pair<std::string, int>> asymOfId;
  auto setAsym = [&](const std::string& id, int asym) {
    for (auto& [k, v] : asymOfId) if (k == id) { v = asym; return; }
    asymOfId.push_back({id, asym});
  };
  int polymerChains = 0;
  for (auto& e : out.entities) {
    if (e.type == "ligand" || e.type == "smiles") continue;
    for (auto& id : e.ids) setAsym(id, polymerChains++);
    if (e.ids.empty()) polymerChains += (int)e.copies;
  }
  int ligandAt = polymerChains;
  for (auto& e : out.entities) {
    if (e.type != "ligand" && e.type != "smiles") continue;
    for (auto& id : e.ids) setAsym(id, ligandAt++);
    if (e.ids.empty()) ligandAt += (int)e.copies;
  }
  struct Pair { int fromAsym, fromResidue; std::string fromAtom; int toAsym, toResidue; std::string toAtom; };
  std::vector<Pair> bonds;
  if (const Json* pairs = job.get("bondedAtomPairs"); !Json::absent(pairs) && pairs->isArray()) {
    for (size_t index = 0; index < pairs->a.size(); ++index) {
      std::string where = "bondedAtomPairs[" + std::to_string(index) + "]";
      const Json& pair = pairs->a[index];
      if (!pair.isArray() || pair.a.size() != 2) refuse(where + ": a bond is two atoms");
      auto end = [&](const Json& side, const char* which, int& asym, int& residue, std::string& atom) {
        if (!side.isArray() || side.a.size() != 3) refuse(where + " " + which + ": an atom is [chain, residue, atom]");
        std::string chain = jsString(&side.a[0]);
        int found = -1;
        for (auto& [k, v] : asymOfId) if (k == chain) found = v;
        if (found < 0) {
          std::string have;
          for (size_t i = 0; i < asymOfId.size(); ++i) have += (i ? ", " : "") + asymOfId[i].first;
          refuse(where + " " + which + ": no chain \"" + chain + "\" in this job (it has " + (have.empty() ? "none named" : have) + ")");
        }
        const Json& r = side.a[1];
        if (!r.isNumber() || !jsIsInteger(r.n) || r.n < 1) refuse(where + " " + which + ": residue " + jsString(&r) + " is not a position");
        asym = found; residue = (int)r.n; atom = jsString(&side.a[2]);
      };
      Pair p;
      end(pair.a[0], "from", p.fromAsym, p.fromResidue, p.fromAtom);
      end(pair.a[1], "to", p.toAsym, p.toResidue, p.toAtom);
      bonds.push_back(p);
    }
  }
  std::set<int> ligandAsyms;
  for (auto& [k, v] : asymOfId) if (v >= polymerChains) ligandAsyms.insert(v);
  std::vector<Pair> reaching;
  for (auto& b : bonds) if (ligandAsyms.count(b.fromAsym) || ligandAsyms.count(b.toAsym)) reaching.push_back(b);
  if (reaching.size() != bonds.size())
    out.notes.push_back(std::to_string(bonds.size() - reaching.size()) + " of " + std::to_string(bonds.size())
                        + " bonded pairs join two polymer residues, which AlphaFold 3 does not put in `token_bonds` either - they are not sent to the model");
  if (const Json* seeds = job.get("modelSeeds"); seeds != nullptr) {
    std::vector<const Json*> list;
    if (seeds->isArray()) for (auto& s : seeds->a) list.push_back(&s);
    else list.push_back(seeds);
    if (list.size() > 1) out.notes.push_back(std::to_string(list.size()) + " seeds in the file; folding the first");
    if (!list.empty()) {
      double first = jsToNumber(list[0]);
      if (!std::isfinite(first) || first < 0) refuse("seed " + jsString(list[0]));
      out.hasSeed = true;
      out.seed = std::floor(first);
    }
  }
  bool carried = false;
  for (auto& e : out.entities) {
    if (!e.hasInlineMsa) continue;
    for (int c = 0; c < (int)e.copies; ++c) {
      out.unpaired.push_back(e.unpairedMsa);
      out.paired.push_back(e.pairedMsa);
      if ((e.unpairedMsa.present && !e.unpairedMsa.text.empty()) || (e.pairedMsa.present && !e.pairedMsa.text.empty())) carried = true;
    }
  }
  if (carried) {
    out.hasAlignments = true;
    size_t bare = 0;
    for (size_t i = 0; i < out.unpaired.size(); ++i)
      if ((!out.unpaired[i].present || out.unpaired[i].text.empty()) && (!out.paired[i].present || out.paired[i].text.empty())) ++bare;
    out.notes.push_back("the file's own alignments for " + std::to_string(out.unpaired.size() - bare) + " of " + std::to_string(out.unpaired.size())
                        + " chains" + (bare == 0 ? std::string() : " - the other " + std::to_string(bare) + " fold" + (bare == 1 ? "s" : "")
                        + " from " + (bare == 1 ? "its" : "their") + " own sequence"));
  } else if (sawEmpty) {
    singleSequence = true;
  }
  if (singleSequence) out.notes.push_back("the file asks for no alignment, so the MSA dial is set to none");
  if (auto p = entitiesProblem(out.entities)) refuse(*p);
  auto pageLabel = [](int asym) {
    std::string label;
    for (int at = asym; at >= 0; at = (int)std::floor(at / 26.0) - 1) label = std::string(1, (char)(65 + at % 26)) + label;
    return label;
  };
  for (auto& b : reaching) {
    auto side = [&](int asym, int residue, const std::string& atom) {
      return pageLabel(asym) + std::to_string(residue) + (atom.empty() ? std::string() : ":" + atom);
    };
    Entity c;
    c.type = "contact";
    c.value = side(b.fromAsym, b.fromResidue, b.fromAtom) + " - " + side(b.toAsym, b.toResidue, b.toAtom);
    out.entities.push_back(c);
  }
  if (const Json* n = job.get("name"); n != nullptr && n->isString()) out.name = n->s;
  if (const Json* u = job.get("userCCD"); u != nullptr && u->isString() && !trimWs(u->s).empty()) out.userCcd = u->s;
  return out;
}

}  // namespace lf
