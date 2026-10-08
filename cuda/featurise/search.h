// The alignment search, as the page runs it: shared/input/mmseqs2-api.js against the ColabFold MMseqs2 server -
// one chain's search (env: UniRef plus the environmental databases), a complex's (each distinct chain searched, the
// distinct chains PAIRED, the per-chain blocks merged the way the WEIGHTS read them), the template hits it reports
// and the structures it serves - and shared/input/chains.js's merges. It sends the sequences to api.colabfold.com,
// so a caller asks for it; nothing here searches unasked.
//
// 🔴 NO LIBRARY: curl and gzip, the two programs every Linux box and every Colab runtime has, run as children - so a
// native fold depends on nothing a fresh runtime has to install.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <regex>
#include <set>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "json.h"
#include "msa.h"

namespace lf::search {

inline std::string apiUrl() {
  const char* v = std::getenv("LOCALFOLD_MMSEQS_API");
  std::string u = v && *v ? v : "https://api.colabfold.com";
  if (u.back() != '/') u += '/';
  return u;
}

inline std::string shellQuote(const std::string& s) {
  std::string out = "'";
  for (char c : s) { if (c == '\'') out += "'\\''"; else out += c; }
  return out + "'";
}

inline std::string runCapture(const std::string& command, int& status) {
  FILE* p = popen(command.c_str(), "r");
  if (!p) throw std::runtime_error("cannot run: " + command);
  std::string out;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
  status = pclose(p);
  return out;
}

struct Response { int status = 0; std::string body; };

// one HTTP request through curl: the body, and the status it came back with (0 when it never connected)
inline Response httpOnce(const std::string& url, const std::vector<std::pair<std::string, std::string>>* form) {
  char tmpl[] = "/tmp/localfold-http-XXXXXX";
  int fd = mkstemp(tmpl);
  if (fd < 0) throw std::runtime_error("cannot make a temporary file");
  close(fd);
  std::string command = "curl -s -L --max-time 600 -o " + shellQuote(tmpl) + " -w '%{http_code}'";
  if (form) {
    command += " -H 'Content-Type: application/x-www-form-urlencoded;charset=UTF-8'";
    for (auto& [k, v] : *form) command += " --data-urlencode " + shellQuote(k + "=" + v);
  }
  command += " " + shellQuote(url);
  int status = 0;
  std::string code = runCapture(command, status);
  Response r;
  r.status = std::atoi(code.c_str());
  r.body = readFile(tmpl);
  std::remove(tmpl);
  return r;
}

// request(): five attempts, retrying 429 and 5xx (and a connection that failed) with 0.5, 1, 2, 4, 8 s between
inline std::string request(const std::string& url, const std::vector<std::pair<std::string, std::string>>* form, const std::string& label) {
  std::string last;
  for (int attempt = 0; attempt < 5; ++attempt) {
    Response r = httpOnce(url, form);
    if (r.status >= 200 && r.status < 300) return r.body;
    if (r.status != 0 && r.status != 429 && r.status != 500 && r.status != 502 && r.status != 503 && r.status != 504) {
      last = label + " failed: HTTP " + std::to_string(r.status);
      break;
    }
    last = r.status == 0 ? label + " failed: could not reach " + url : label + " failed: HTTP " + std::to_string(r.status);
    std::this_thread::sleep_for(std::chrono::milliseconds(500 << attempt));
  }
  throw std::runtime_error(last.empty() ? label + " failed" : last);
}

inline std::string gunzip(const std::string& bytes) {
  char tmpl[] = "/tmp/localfold-gz-XXXXXX";
  int fd = mkstemp(tmpl);
  if (fd < 0) throw std::runtime_error("cannot make a temporary file");
  if (write(fd, bytes.data(), bytes.size()) != (ssize_t)bytes.size()) { close(fd); throw std::runtime_error("short write"); }
  close(fd);
  int status = 0;
  std::string out = runCapture("gzip -dc < " + shellQuote(tmpl), status);
  std::remove(tmpl);
  if (status != 0) throw std::runtime_error("the MMseqs2 result did not decompress");
  return out;
}

// readTarFiles: a ustar archive's regular files, by path ("./" stripped), in archive order
inline std::vector<std::pair<std::string, std::string>> readTarFiles(const std::string& bytes) {
  std::vector<std::pair<std::string, std::string>> files;
  auto field = [&](size_t start, size_t length) {
    size_t end = start;
    while (end < start + length && bytes[end] != 0) ++end;
    return bytes.substr(start, end - start);
  };
  for (size_t offset = 0; offset + 512 <= bytes.size();) {
    bool zero = true;
    for (size_t k = 0; k < 512; ++k) if (bytes[offset + k] != 0) { zero = false; break; }
    if (zero) break;
    std::string name = field(offset, 100), prefix = field(offset + 345, 155);
    std::string path = prefix.empty() ? name : prefix + "/" + name;
    std::string size8 = trimWs(field(offset + 124, 12));
    double size = size8.empty() ? 0 : (double)std::strtoull(size8.c_str(), nullptr, 8);
    size_t dataStart = offset + 512, dataEnd = dataStart + (size_t)size;
    if (path.empty() || dataEnd > bytes.size()) throw std::runtime_error("MMseqs2 result contains a truncated tar entry");
    unsigned char type = (unsigned char)bytes[offset + 156];
    if (type == 0 || type == '0') {
      std::string clean = path.compare(0, 2, "./") == 0 ? path.substr(2) : path;
      bool replaced = false;
      for (auto& f : files) if (f.first == clean) { f.second = bytes.substr(dataStart, dataEnd - dataStart); replaced = true; }
      if (!replaced) files.push_back({clean, bytes.substr(dataStart, dataEnd - dataStart)});
    }
    offset = dataStart + (size_t)std::ceil(size / 512) * 512;
  }
  return files;
}

inline const std::string* findFile(const std::vector<std::pair<std::string, std::string>>& files, const std::string& suffix) {
  for (auto& [path, bytes] : files)
    if (path == suffix || (path.size() > suffix.size() && path.compare(path.size() - suffix.size() - 1, std::string::npos, "/" + suffix) == 0))
      return &bytes;
  return nullptr;
}

// queryBlock: the \0-separated block whose first line is >id
inline std::string queryBlock(const std::string& contents, int queryId) {
  std::string text;
  for (char c : contents) if (c != '\r') text += c;
  size_t start = 0;
  std::string head = ">" + std::to_string(queryId);
  while (start <= text.size()) {
    size_t end = text.find('\0', start);
    if (end == std::string::npos) end = text.size();
    std::string trimmed = trimWs(text.substr(start, end - start));
    if (trimmed.compare(0, head.size(), head) == 0 && (trimmed.size() == head.size() || std::isspace((unsigned char)trimmed[head.size()])))
      return trimmed;
    if (end == text.size()) break;
    start = end + 1;
    while (start < text.size() && text[start] == '\0') ++start;
  }
  throw std::runtime_error("MMseqs2 result does not contain query " + std::to_string(queryId));
}

struct Hit { std::string id, chain, target; double identity, evalue, bits, queryStart, templateStart; std::string cigar; };
using Hits = std::map<int, std::vector<Hit>>;

inline Hits extractTemplateHits(const std::vector<std::pair<std::string, std::string>>& files) {
  Hits hits;
  const std::string* m8 = findFile(files, "pdb70.m8");
  if (!m8) return hits;
  for (auto& raw : splitLines(*m8)) {
    std::vector<std::string> parts;
    std::string line = trimWs(raw), cur;
    for (char c : line) { if (std::isspace((unsigned char)c)) { if (!cur.empty()) parts.push_back(cur), cur.clear(); } else cur += c; }
    if (!cur.empty()) parts.push_back(cur);
    if (parts.size() < 12) continue;
    double chainIndex = jsNumberOf(parts[0]) - 101;
    if (!(std::isfinite(chainIndex) && std::floor(chainIndex) == chainIndex) || chainIndex < 0) continue;
    std::string target = parts[1];
    size_t us = target.find('_');
    Hit h;
    std::string id = target.substr(0, us);
    for (auto& c : id) c = (char)std::tolower((unsigned char)c);
    h.id = id;
    h.chain = us == std::string::npos ? "A" : target.substr(us + 1, target.find('_', us + 1) - us - 1);
    h.target = target;
    h.identity = jsNumberOf(parts[2]); h.evalue = jsNumberOf(parts[10]); h.bits = jsNumberOf(parts[11]);
    h.queryStart = jsNumberOf(parts[6]); h.templateStart = jsNumberOf(parts[8]);
    h.cigar = parts.size() > 12 ? parts[12] : "";
    hits[(int)chainIndex].push_back(h);
  }
  return hits;
}

inline std::string normalizedSequence(const std::string& s) {
  std::string v;
  for (char c : s) if (!std::isspace((unsigned char)c)) v += (char)std::toupper((unsigned char)c);
  if (v.empty() || v.find_first_not_of("ARNDCQEGHILKMFPSTWYVX") != std::string::npos)
    throw std::runtime_error("Sequence must contain only standard amino-acid letters or X");
  return v;
}

// runMmseqs2Job / generateMmseqs2Msa's loop: submit, wait while the server is busy, download, decompress
inline std::string runJob(const std::string& query, const std::string& endpoint, const std::string& mode,
                          const std::function<void(const std::string&)>& say) {
  std::string base = apiUrl();
  std::vector<std::pair<std::string, std::string>> form = {{"q", query}, {"mode", mode}};
  std::mt19937 rng(std::random_device{}());
  auto pause = [&]() { std::this_thread::sleep_for(std::chrono::milliseconds(5000 + (int)(rng() % 5000))); };
  auto parseTicket = [](const std::string& body, std::string& id) {
    Json j = parseJson(body);
    const Json* s = j.get("status");
    const Json* i = j.get("id");
    id = i && i->isString() ? i->s : "";
    return s && s->isString() ? upper(s->s) : std::string("ERROR");
  };
  std::string ticket, status = "UNKNOWN";
  while (ticket.empty()) {
    say("submitting");
    status = parseTicket(request(base + "ticket/" + endpoint, &form, "MMseqs2 submission"), ticket);
    if (status == "UNKNOWN" || status == "RATELIMIT") { ticket.clear(); say("retrying (" + status + ")"); pause(); continue; }
    if (status == "ERROR") throw std::runtime_error("MMseqs2 rejected the sequence or is temporarily unavailable");
    if (status == "MAINTENANCE") throw std::runtime_error("The MMseqs2 API is undergoing maintenance; try again later");
    if (ticket.empty()) throw std::runtime_error("MMseqs2 returned " + status + " without a ticket");
  }
  while (status == "UNKNOWN" || status == "PENDING" || status == "RUNNING" || status == "RATELIMIT") {
    say(status == "RUNNING" ? "running" : "queued");
    pause();
    std::string ignored;
    status = parseTicket(request(base + "ticket/" + ticket, nullptr, "MMseqs2 status"), ignored);
  }
  if (status != "COMPLETE") throw std::runtime_error("MMseqs2 search ended with status " + status);
  say("downloading");
  return gunzip(request(base + "result/download/" + ticket, nullptr, "MMseqs2 result download"));
}

struct Searched { std::string a3m; Hits hits; };

// generateMmseqs2Msa: one chain - UniRef's block and the environmental one, each opening with its own query
inline Searched searchOne(const std::string& sequenceValue, const std::function<void(const std::string&)>& say) {
  std::string sequence = normalizedSequence(sequenceValue);
  auto files = readTarFiles(runJob(">101\n" + sequence + "\n", "msa", "env", say));
  const std::string* uniref = findFile(files, "uniref.a3m");
  if (!uniref) throw std::runtime_error("MMseqs2 result is missing uniref.a3m");
  const std::string* env = findFile(files, "bfd.mgnify30.metaeuk30.smag30.a3m");
  if (!env) throw std::runtime_error("MMseqs2 result is missing the environmental A3M");
  Searched s;
  s.a3m = queryBlock(*uniref, 101) + "\n" + queryBlock(*env, 101) + "\n";
  if (parseA3m(s.a3m).sequences[0] != sequence) throw std::runtime_error("MMseqs2 returned an A3M for a different query sequence");
  s.hits = extractTemplateHits(files);
  return s;
}

// generateMmseqs2PairedMsa: the distinct chains' paired rows, one A3M each
inline std::vector<std::string> searchPaired(const std::vector<std::string>& unique, const std::function<void(const std::string&)>& say) {
  std::string query;
  for (size_t i = 0; i < unique.size(); ++i) query += ">" + std::to_string(101 + i) + "\n" + unique[i] + "\n";
  auto files = readTarFiles(runJob(query, "pair", "pairgreedy", say));
  const std::string* pair = findFile(files, "pair.a3m");
  if (!pair) throw std::runtime_error("MMseqs2 pairing result is missing pair.a3m");
  std::vector<std::string> out;
  size_t depth = 0;
  for (size_t i = 0; i < unique.size(); ++i) {
    out.push_back(queryBlock(*pair, 101 + (int)i) + "\n");
    A3m a = parseA3m(out.back());
    if (a.sequences[0] != unique[i]) throw std::runtime_error("MMseqs2 returned a paired A3M for a different query sequence");
    if (i == 0) depth = a.sequences.size();
    else if (a.sequences.size() != depth) throw std::runtime_error("MMseqs2 paired A3Ms do not have aligned row counts");
  }
  return out;
}

// fetchMmseqs2Templates: the server's mmCIF for each "<pdb>_<chain>" target, by lower-case name
inline std::map<std::string, std::string> fetchTemplates(const std::vector<std::string>& targets) {
  std::vector<std::string> wanted;
  static const std::regex SHAPE(R"(^[A-Za-z0-9]{4}_[A-Za-z0-9]+$)");
  for (auto& t : targets) if (std::regex_match(t, SHAPE) && std::find(wanted.begin(), wanted.end(), t) == wanted.end()) wanted.push_back(t);
  std::map<std::string, std::string> structures;
  if (wanted.empty()) return structures;
  std::string list;
  for (size_t i = 0; i < wanted.size(); ++i) list += (i ? "," : "") + wanted[i];
  for (auto& [path, bytes] : readTarFiles(gunzip(request(apiUrl() + "template/" + list, nullptr, "MMseqs2 template download")))) {
    std::string name = path.substr(path.rfind('/') == std::string::npos ? 0 : path.rfind('/') + 1);
    if (name.size() < 4 || name.compare(name.size() - 4, 4, ".cif") != 0) continue;
    std::string key = name.substr(0, name.size() - 4);
    for (auto& c : key) c = (char)std::tolower((unsigned char)c);
    structures[key] = bytes;
  }
  return structures;
}

// ---------------------------------------------------------------- shared/input/chains.js
inline std::string mergeUnpairedChainA3ms(const std::vector<std::string>& texts) {
  if (texts.empty()) throw std::runtime_error("at least one chain A3M is required");
  std::vector<A3m> al;
  for (auto& t : texts) al.push_back(parseA3m(t));
  std::string query;
  for (auto& a : al) query += a.sequences[0];
  std::string out = ">query\n" + query;
  for (size_t c = 0; c < al.size(); ++c) {
    int left = 0, right = 0;
    for (size_t k = 0; k < al.size(); ++k) { if (k < c) left += al[k].length; else if (k > c) right += al[k].length; }
    for (size_t row = 1; row < al[c].raw.size(); ++row)
      out += "\n>chain_" + std::to_string(c + 1) + "|" + al[c].descriptions[row] + "\n" + std::string(left, '-') + al[c].raw[row] + std::string(right, '-');
  }
  return out + "\n";
}
inline std::string mergeChainA3ms(const std::vector<std::string>& texts) {
  if (texts.empty()) throw std::runtime_error("at least one chain A3M is required");
  std::vector<A3m> al;
  for (auto& t : texts) al.push_back(parseA3m(t));
  std::string query;
  for (auto& a : al) query += a.sequences[0];
  std::string out = ">query\n" + query;
  std::vector<std::pair<std::string, std::vector<int>>> groups;
  for (size_t c = 0; c < al.size(); ++c) {
    auto g = std::find_if(groups.begin(), groups.end(), [&](auto& e) { return e.first == al[c].sequences[0]; });
    if (g == groups.end()) groups.push_back({al[c].sequences[0], {(int)c}});
    else g->second.push_back((int)c);
  }
  for (auto& [q, chains] : groups) {
    const A3m& a = al[chains[0]];
    std::string label;
    for (size_t i = 0; i < chains.size(); ++i) label += (i ? "+" : "") + std::to_string(chains[i] + 1);
    for (size_t row = 1; row < a.raw.size(); ++row) {
      std::string parts;
      for (size_t c = 0; c < al.size(); ++c)
        parts += std::find(chains.begin(), chains.end(), (int)c) != chains.end() ? a.raw[row] : std::string(al[c].length, '-');
      out += "\n>chain_" + label + "|" + a.descriptions[row] + "\n" + parts;
    }
  }
  return out + "\n";
}
inline std::string concatenateA3mBlocks(const std::string& pairedText, const std::string& unpairedText) {
  A3m p = parseA3m(pairedText), u = parseA3m(unpairedText);
  if (p.sequences[0] != u.sequences[0]) throw std::runtime_error("the paired and unpaired A3M blocks describe different queries");
  std::string out;
  for (size_t r = 0; r < p.raw.size(); ++r) out += (out.empty() ? "" : "\n") + (">" + p.descriptions[r]) + "\n" + p.raw[r];
  for (size_t r = 1; r < u.raw.size(); ++r) out += "\n>" + u.descriptions[r] + "\n" + u.raw[r];
  return out + "\n";
}
inline std::string deduplicateUnpairedAgainstPaired(const std::string& unpairedText, const std::string& pairedText) {
  if (pairedText.empty()) return unpairedText;
  A3m p = parseA3m(pairedText), u = parseA3m(unpairedText);
  std::set<std::string> seen(p.sequences.begin(), p.sequences.end());
  std::string out;
  for (size_t r = 0; r < u.raw.size(); ++r) {
    if (r > 0 && seen.count(u.sequences[r])) continue;
    out += (out.empty() ? "" : "\n") + (">" + u.descriptions[r]) + "\n" + u.raw[r];
  }
  return out + "\n";
}

// "monomer" (block-diagonal), "multimer" (dense within an entity), or an AF3 family (row by row)
inline std::string mergeFor(const std::string& model, const std::vector<std::string>& texts) {
  if (model == "monomer") return mergeUnpairedChainA3ms(texts);
  if (model == "multimer") return mergeChainA3ms(texts);
  return mergeRowAlignedChainA3ms(texts);
}

struct Merged { std::string a3m; bool hasPaired = false; std::string paired, unpaired, unpairedProfile; };
// mergeSearchedChains
inline Merged mergeSearchedChains(const std::vector<std::string>& sequences, const std::vector<std::string>& chainA3ms,
                                  const std::map<std::string, std::string>& pairedA3ms, const std::string& model) {
  Merged m;
  bool hasPairing = !pairedA3ms.empty() && model != "monomer";
  if (hasPairing) {
    std::vector<std::string> list;
    for (auto& s : sequences) list.push_back(pairedA3ms.at(s));
    m.paired = mergeRowAlignedChainA3ms(list);
    m.hasPaired = true;
  }
  m.unpairedProfile = mergeFor(model, chainA3ms);
  std::vector<std::string> dedup = chainA3ms;
  if (hasPairing) for (size_t c = 0; c < sequences.size(); ++c) dedup[c] = deduplicateUnpairedAgainstPaired(chainA3ms[c], pairedA3ms.at(sequences[c]));
  m.unpaired = mergeFor(model, dedup);
  m.a3m = m.hasPaired ? concatenateA3mBlocks(m.paired, m.unpaired) : m.unpaired;
  return m;
}

struct ComplexSearched { Merged merged; Hits hits; std::vector<std::string> chainA3ms; };
// generateMmseqs2ComplexMsa: each distinct chain searched once, paired when the model reads pairs
inline ComplexSearched searchComplex(const std::vector<std::string>& values, const std::string& model,
                                     const std::function<void(const std::string&)>& say) {
  if (values.size() < 2) throw std::runtime_error("a complex MSA requires at least two chain sequences");
  std::vector<std::string> sequences, unique;
  for (auto& v : values) sequences.push_back(normalizedSequence(v));
  for (auto& s : sequences) if (std::find(unique.begin(), unique.end(), s) == unique.end()) unique.push_back(s);
  std::map<std::string, Searched> bySequence;
  for (size_t i = 0; i < unique.size(); ++i)
    bySequence[unique[i]] = searchOne(unique[i], [&](const std::string& m) { say("chain search " + std::to_string(i + 1) + " of " + std::to_string(unique.size()) + ": " + m); });
  ComplexSearched out;
  for (auto& s : sequences) out.chainA3ms.push_back(bySequence[s].a3m);
  std::map<std::string, std::string> paired;
  if (model != "monomer" && unique.size() > 1) {
    auto list = searchPaired(unique, [&](const std::string& m) { say("paired search: " + m); });
    for (size_t i = 0; i < unique.size(); ++i) paired[unique[i]] = list[i];
  }
  out.merged = mergeSearchedChains(sequences, out.chainA3ms, paired, model);
  for (size_t i = 0; i < sequences.size(); ++i) {
    auto& h = bySequence[sequences[i]].hits;
    out.hits[(int)i] = h.count(0) ? h.at(0) : std::vector<Hit>{};
  }
  return out;
}

}  // namespace lf::search
