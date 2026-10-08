// A chemical component from the CCD, as shared/af3/featurise/ccd-component.js reads one: its heavy atoms with
// ideal coordinates, its bonds, its parent - and the two reshapings the featuriser makes of one (a residue's
// leaving atoms dropped, several bonded components as one ligand chain). Fetched from the RCSB, as the page
// fetches it, into a cache (LOCALFOLD_CCD_DIR, default ~/.cache/localfold/ccd) - or, given a local CCD file
// (--ccd=<components.cif[.gz]>, wwPDB's whole dictionary; `fetch-weights ccd` downloads it), read from that alone.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include "jsnum.h"
#include "tables.h"

namespace lf {

struct CompAtom {
  std::string name;
  int element = 0;
  double charge = 0;
  bool leaving = false;
  double x = 0, y = 0, z = 0;
  int componentSlot = -1;          // polymerResidue's: the atom's index in its component
};
struct CompBond { int from, to, order; };
// a bond the job declares (bondedAtomPairs), by chain (asym, from 0), residue (from 1) and atom name
struct BondEnd { int asym; int residue; bool hasAtom; std::string atom; };
struct DeclaredBond { BondEnd from, to; };
struct CompPart { std::string code; int from, count; };
struct Component {
  std::string code;
  std::vector<CompAtom> atoms;
  std::vector<CompBond> bonds;
  std::string parent;              // "" is null
  std::vector<CompPart> residues;  // ligandChain's; empty for one component
  bool smiles = false;             // built from a SMILES (shared/chem), not read from the dictionary
};

inline std::string trimWs(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) ++a;
  while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}
inline std::string upper(std::string s) { for (auto& c : s) c = (char)std::toupper((unsigned char)c); return s; }
inline std::vector<std::string> splitLines(const std::string& text) {   // text.split(/\r?\n/)
  std::vector<std::string> out;
  size_t start = 0;
  for (size_t i = 0; i <= text.size(); ++i) {
    if (i == text.size() || text[i] == '\n') {
      size_t end = i;
      if (end > start && text[end - 1] == '\r') --end;
      out.push_back(text.substr(start, end - start));
      start = i + 1;
    }
  }
  return out;
}
// value.replace(/^['"]|['"]$/g, "")
inline std::string stripQuotes(std::string v) {
  if (!v.empty() && (v[0] == '\'' || v[0] == '"')) v.erase(0, 1);
  if (!v.empty() && (v.back() == '\'' || v.back() == '"')) v.pop_back();
  return v;
}
// trimmed.match(/'[^']*'|"[^"]*"|\S+/g)
inline std::vector<std::string> cifFields(const std::string& line) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < line.size()) {
    char c = line[i];
    if (c == '\'' || c == '"') {
      size_t close = line.find(c, i + 1);
      if (close != std::string::npos) { out.push_back(line.substr(i, close - i + 1)); i = close + 1; continue; }
    }
    if (std::isspace((unsigned char)c)) { ++i; continue; }
    size_t j = i;
    while (j < line.size() && !std::isspace((unsigned char)line[j])) ++j;
    out.push_back(line.substr(i, j - i));
    i = j;
  }
  return out;
}
struct CifLoop { std::vector<std::string> columns; std::vector<std::vector<std::string>> rows; };
inline bool allHashOrSpace(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s) if (c != '#' && !std::isspace((unsigned char)c)) return false;
  return true;
}
inline CifLoop loopOf(const std::string& text, const std::string& prefix) {
  CifLoop loop;
  std::vector<std::pair<std::string, std::string>> single;
  bool inHeader = false, inBody = false;
  for (const auto& line : splitLines(text)) {
    std::string trimmed = trimWs(line);
    if (trimmed.compare(0, prefix.size() + 1, prefix + ".") == 0) {
      if (inBody) break;
      std::string rest = trimmed.substr(prefix.size() + 1);
      std::vector<std::string> parts;                       // .split(/\s+/)
      {
        size_t i = 0, start = 0;
        for (; i <= rest.size(); ++i) {
          if (i == rest.size() || std::isspace((unsigned char)rest[i])) {
            parts.push_back(rest.substr(start, i - start));
            while (i < rest.size() && std::isspace((unsigned char)rest[i])) ++i;
            start = i;
            if (i == rest.size()) break;
            --i;
          }
        }
      }
      loop.columns.push_back(parts[0]);
      if (parts.size() > 1) {
        std::string joined;
        for (size_t k = 1; k < parts.size(); ++k) joined += (k > 1 ? " " : "") + parts[k];
        single.emplace_back(parts[0], joined);
      }
      inHeader = true;
      continue;
    }
    if (!inHeader) continue;
    if (trimmed.empty() || trimmed == "loop_" || allHashOrSpace(trimmed)) {
      if (inBody) break;
      continue;
    }
    if (trimmed[0] == '_') break;
    std::vector<std::string> fields;
    for (auto& f : cifFields(trimmed)) fields.push_back(stripQuotes(f));
    if (fields.size() != loop.columns.size()) {
      if (inBody) break;
      continue;
    }
    inBody = true;
    loop.rows.push_back(fields);
  }
  if (loop.rows.empty() && !single.empty()) {
    std::vector<std::string> row;
    for (auto& item : loop.columns) {
      std::string v;
      for (auto& [k, val] : single) if (k == item) { v = val; break; }
      row.push_back(stripQuotes(v));
    }
    loop.rows.push_back(row);
  }
  return loop;
}
// text.match(/<key>\s+(\S+)/)?.[1]?.replace(/['"]/g, ""); "" when absent
inline std::string cifValue(const std::string& text, const std::string& key) {
  size_t at = 0;
  while ((at = text.find(key, at)) != std::string::npos) {
    size_t i = at + key.size();
    size_t ws = i;
    while (i < text.size() && std::isspace((unsigned char)text[i])) ++i;
    if (i > ws && i < text.size()) {
      size_t j = i;
      while (j < text.size() && !std::isspace((unsigned char)text[j])) ++j;
      std::string v;
      for (char c : text.substr(i, j - i)) if (c != '\'' && c != '"') v += c;
      return v;
    }
    at += 1;
  }
  return "";
}

inline int elementNumber(const std::string& symbol) {      // ELEMENT_SYMBOLS.indexOf(symbol) + 1
  const auto& s = elementSymbols();
  for (size_t i = 0; i < s.size(); ++i) if (s[i] == symbol) return (int)i + 1;
  return 0;
}

inline Component parseCcdComponent(const std::string& text) {
  Component c;
  c.code = cifValue(text, "_chem_comp.id");
  if (c.code.empty()) throw std::runtime_error("this mmCIF has no _chem_comp.id");
  CifLoop atomLoop = loopOf(text, "_chem_comp_atom");
  auto col = [&](const std::string& name) {
    for (size_t i = 0; i < atomLoop.columns.size(); ++i) if (atomLoop.columns[i] == name) return (int)i;
    return -1;
  };
  auto at = [&](const std::vector<std::string>& row, const std::string& name, bool& present) -> std::string {
    int i = col(name);
    present = i >= 0;
    return i < 0 ? std::string() : row[i];
  };
  auto atOr = [&](const std::vector<std::string>& row, const std::string& name, const std::string& fallback) {
    bool p; std::string v = at(row, name, p); return p ? v : fallback;
  };
  // parseFloat of a column, NaN where the column is absent (Number.parseFloat(undefined))
  auto num = [&](const std::vector<std::string>& row, const std::string& name) {
    bool p; std::string v = at(row, name, p); return p ? jsParseFloat(v) : NAN;
  };
  auto isHydrogen = [&](const std::vector<std::string>& row) {
    std::string s = upper(atOr(row, "type_symbol", ""));
    return s == "H" || s == "D";
  };
  std::vector<const std::vector<std::string>*> heavy;
  for (auto& row : atomLoop.rows) if (!isHydrogen(row)) heavy.push_back(&row);
  bool monatomic = heavy.size() == 1;
  bool ideal = true;
  for (auto* row : heavy)
    for (const char* axis : {"x", "y", "z"})
      if (!std::isfinite(num(*row, std::string("pdbx_model_Cartn_") + axis + "_ideal"))) ideal = false;
  double centre[3] = {0, 0, 0};
  if (!ideal && !monatomic) {
    const char* axes[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; ++a) {
      double sum = 0;
      for (auto* row : heavy) sum += num(*row, std::string("model_Cartn_") + axes[a]);
      centre[a] = sum / (double)heavy.size();
    }
  }
  auto coordinate = [&](const std::vector<std::string>& row, int axis) {
    const char* axes[3] = {"x", "y", "z"};
    std::string column = ideal ? std::string("pdbx_model_Cartn_") + axes[axis] + "_ideal" : std::string("model_Cartn_") + axes[axis];
    double parsed = num(row, column);
    if (std::isfinite(parsed)) return ideal ? parsed : jsRound((parsed - centre[axis]) * 1000) / 1000;
    if (monatomic) return 0.0;
    throw std::runtime_error(c.code + " has no usable " + axes[axis] + " coordinate");
  };
  std::map<std::string, int> byName;
  for (auto& row : atomLoop.rows) {
    if (isHydrogen(row)) continue;
    CompAtom a;
    a.name = atOr(row, "atom_id", "");
    byName[a.name] = (int)c.atoms.size();
    a.element = elementNumber(upper(atOr(row, "type_symbol", "")));
    double q = jsParseFloat(atOr(row, "charge", "0"));
    a.charge = (std::isnan(q) || q == 0) ? 0 : q;
    a.leaving = upper(atOr(row, "pdbx_leaving_atom_flag", "N")) == "Y";
    a.x = coordinate(row, 0); a.y = coordinate(row, 1); a.z = coordinate(row, 2);
    c.atoms.push_back(a);
  }
  if (c.atoms.empty()) throw std::runtime_error(c.code + " has no heavy atoms");
  CifLoop bondLoop = loopOf(text, "_chem_comp_bond");
  auto bcol = [&](const std::string& name) {
    for (size_t i = 0; i < bondLoop.columns.size(); ++i) if (bondLoop.columns[i] == name) return (int)i;
    return -1;
  };
  for (auto& row : bondLoop.rows) {
    int i1 = bcol("atom_id_1"), i2 = bcol("atom_id_2");
    if (i1 < 0 || i2 < 0) continue;
    auto f = byName.find(row[i1]), t = byName.find(row[i2]);
    if (f == byName.end() || t == byName.end()) continue;
    int ia = bcol("pdbx_aromatic_flag"), io = bcol("value_order");
    bool aromatic = upper(ia < 0 ? "N" : row[ia]) == "Y";
    std::string order = io < 0 ? "SING" : row[io];
    int o = order == "SING" ? 1 : order == "DOUB" ? 2 : order == "TRIP" ? 3 : order == "QUAD" ? 4 : 1;   // AROM and the rest: 1
    c.bonds.push_back({f->second, t->second, aromatic ? 4 : o});
  }
  std::string named = cifValue(text, "_chem_comp.mon_nstd_parent_comp_id");
  if (!named.empty() && named != "?" && named != ".") c.parent = upper(named.substr(0, named.find(',')));
  return c;
}

// several bonded components as ONE ligand chain (a glycan), one residue each
inline Component ligandChain(const std::vector<Component>& components) {
  if (components.size() == 1) return components[0];
  Component out;
  for (auto& comp : components) {
    int from = (int)out.atoms.size();
    out.residues.push_back({comp.code, from, (int)comp.atoms.size()});
    out.atoms.insert(out.atoms.end(), comp.atoms.begin(), comp.atoms.end());
    for (auto& b : comp.bonds) out.bonds.push_back({b.from + from, b.to + from, b.order});
    out.code += (out.code.empty() ? "" : "-") + comp.code;
  }
  return out;
}

// a modified residue's parent as its chain's letter, or 0
inline char parentLetter(const std::string& parent, const std::string& kind) {
  if (parent.empty()) return 0;
  if (kind == "protein") {
    static const std::map<std::string, char> amino = {{"ALA", 'A'}, {"ARG", 'R'}, {"ASN", 'N'}, {"ASP", 'D'},
      {"CYS", 'C'}, {"GLN", 'Q'}, {"GLU", 'E'}, {"GLY", 'G'}, {"HIS", 'H'}, {"ILE", 'I'}, {"LEU", 'L'}, {"LYS", 'K'},
      {"MET", 'M'}, {"PHE", 'F'}, {"PRO", 'P'}, {"SER", 'S'}, {"THR", 'T'}, {"TRP", 'W'}, {"TYR", 'Y'}, {"VAL", 'V'}};
    auto it = amino.find(parent);
    return it == amino.end() ? 0 : it->second;
  }
  if (kind == "dna") {
    static const std::map<std::string, char> deoxy = {{"DA", 'A'}, {"DC", 'C'}, {"DG", 'G'}, {"DT", 'T'}};
    auto it = deoxy.find(parent);
    return it == deoxy.end() ? 0 : it->second;
  }
  return parent == "A" || parent == "C" || parent == "G" || parent == "U" ? parent[0] : 0;
}

// a component as a residue of a chain: its leaving atoms gone but the terminal one at the chain's end
inline Component polymerResidue(const Component& comp, bool isTerminal, const std::string& terminalAtom) {
  Component out;
  out.code = comp.code;
  out.parent = comp.parent;
  std::vector<int> renumbered(comp.atoms.size(), -1);
  int next = 0;
  for (size_t i = 0; i < comp.atoms.size(); ++i) {
    bool keep = !comp.atoms[i].leaving || (isTerminal && comp.atoms[i].name == terminalAtom);
    if (!keep) continue;
    renumbered[i] = next++;
    CompAtom a = comp.atoms[i];
    a.componentSlot = (int)i;
    out.atoms.push_back(a);
  }
  for (auto& b : comp.bonds)
    if (renumbered[b.from] >= 0 && renumbered[b.to] >= 0) out.bonds.push_back({renumbered[b.from], renumbered[b.to], b.order});
  return out;
}

// ccdUrl's check, then the dictionary entry: the cache, else the RCSB
inline std::string ccdCode(const std::string& code) {
  std::string u = upper(trimWs(code));
  bool ok = !u.empty() && u.size() <= 5;
  for (char ch : u) if (!std::isalnum((unsigned char)ch)) ok = false;
  if (!ok) throw std::runtime_error(code + " is not a CCD code: expected 1-5 letters or digits");
  return u;
}
inline std::string readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}
inline std::string ccdText(const std::string& code) {
  std::string u = ccdCode(code);
  const char* env = std::getenv("LOCALFOLD_CCD_DIR");
  std::string dir = env ? env : std::string(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") + "/.cache/localfold/ccd";
  std::string path = dir + "/" + u + ".cif";
  struct stat st;
  if (stat(path.c_str(), &st) == 0 && st.st_size > 0) return readFile(path);
  std::string mk = "mkdir -p '" + dir + "'";
  if (std::system(mk.c_str()) != 0) throw std::runtime_error("cannot create " + dir);
  std::string part = path + ".part" + std::to_string(::getpid());
  std::string cmd = "curl -sS -L -o '" + part + "' -w '%{http_code}' https://files.rcsb.org/ligands/download/" + u + ".cif";
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) throw std::runtime_error("could not fetch " + u + ": curl did not start");
  char status[16] = {0};
  size_t got = fread(status, 1, sizeof status - 1, p);
  int rc = pclose(p);
  std::string code3(status, got);
  if (rc != 0 || code3 != "200") {
    std::remove(part.c_str());
    throw std::runtime_error("could not fetch " + u + ": " + (code3.empty() ? "no answer" : code3));
  }
  std::rename(part.c_str(), path.c_str());
  return readFile(path);
}

// A local CCD - wwPDB's components.cif, plain (mapped) or gzipped (decompressed once) - indexed by its data_ lines on
// first use and kept for the process (a resident server reads it once). A code it lacks is refused: a run that names
// a dictionary folds from that dictionary, never from the network behind it, and its blocks never enter the cache
// above, so a custom dictionary cannot leak into a later run without it.
struct CcdFile {
  const char* data = nullptr; size_t size = 0;
  std::string inflated;                              // (a .gz's text)
  std::map<std::string, std::pair<size_t, size_t>> at;   // code -> [start, end) of its block
};
inline const CcdFile& ccdFile(const std::string& path) {
  static std::map<std::string, CcdFile> open;
  auto it = open.find(path);
  if (it != open.end()) return it->second;
  CcdFile f;
  if (path.size() > 3 && path.compare(path.size() - 3, 3, ".gz") == 0) {
    FILE* p = popen(("gzip -dc '" + path + "'").c_str(), "r");
    if (!p) throw std::runtime_error("cannot read " + path);
    char buf[1 << 16]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, p)) > 0) f.inflated.append(buf, n);
    if (pclose(p) != 0) throw std::runtime_error(path + " did not decompress");
    f.data = f.inflated.data(); f.size = f.inflated.size();
  } else {
    int fd = ::open(path.c_str(), O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) throw std::runtime_error("cannot read the CCD " + path);
    f.size = (size_t)st.st_size;
    void* m = f.size ? mmap(nullptr, f.size, PROT_READ, MAP_PRIVATE, fd, 0) : nullptr;
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("cannot map the CCD " + path);
    f.data = (const char*)m;
  }
  std::string last; size_t lastAt = 0;
  for (size_t i = 0; i + 5 <= f.size;) {
    if ((i == 0 || f.data[i - 1] == '\n') && std::memcmp(f.data + i, "data_", 5) == 0) {
      size_t e = i + 5; while (e < f.size && f.data[e] != '\n' && f.data[e] != '\r' && f.data[e] != ' ') ++e;
      if (!last.empty()) f.at[last] = {lastAt, i};
      last = std::string(f.data + i + 5, e - i - 5); lastAt = i;
      i = e;
    } else {
      const void* nl = std::memchr(f.data + i, '\n', f.size - i);
      if (!nl) break;
      i = (const char*)nl - f.data + 1;
    }
  }
  if (!last.empty()) f.at[last] = {lastAt, f.size};
  if (f.at.empty()) throw std::runtime_error(path + " holds no data_ blocks: not a CCD (wwPDB's components.cif)");
  return open.emplace(path, std::move(f)).first->second;
}
inline std::string ccdFileText(const std::string& path, const std::string& code) {
  std::string u = ccdCode(code);
  const CcdFile& f = ccdFile(path);
  auto it = f.at.find(u);
  if (it == f.at.end()) throw std::runtime_error(u + " is not in the CCD " + path);
  return std::string(f.data + it->second.first, it->second.second - it->second.first);
}

}  // namespace lf
