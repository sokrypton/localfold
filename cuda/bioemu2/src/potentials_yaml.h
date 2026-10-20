// A BioEmu2 `--potentials` file, natively: shared/bioemu2/potentials-yaml.js's parseSimpleYaml (the YAML subset the
// vendor's preset files use, anything else refused by name) into an lf::Json tree, then potentials.js's
// potentialSpecs. Both transcribed; the refusals say the same things.
#pragma once
#include <regex>
#include <string>
#include <vector>

#include "../../featurise/json.h"
#include "steer.h"

namespace be2steer {

inline std::string yamlTrim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
inline lf::Json yamlNumber(double v) { lf::Json j; j.kind = lf::Json::Number; j.n = v; return j; }
inline lf::Json yamlString(const std::string& v) { lf::Json j; j.kind = lf::Json::String; j.s = v; return j; }

inline lf::Json yamlScalar(const std::string& text, const std::string& where) {
  const std::string t = yamlTrim(text);
  lf::Json j;
  if (t == "" || t == "~" || t == "null" || t == "Null" || t == "NULL") return j;
  if (t == "true" || t == "True" || t == "TRUE") { j.kind = lf::Json::Bool; j.b = true; return j; }
  if (t == "false" || t == "False" || t == "FALSE") { j.kind = lf::Json::Bool; j.b = false; return j; }
  static const std::regex number(R"(^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$)"), inf(R"(^[-+]?\.(inf|Inf|INF)$)");
  if (std::regex_match(t, number)) return yamlNumber(std::strtod(t.c_str(), nullptr));
  if (std::regex_match(t, inf)) return yamlNumber(t[0] == '-' ? -INF : INF);
  if (t.size() >= 2 && ((t.front() == '"' && t.back() == '"') || (t.front() == '\'' && t.back() == '\''))) return yamlString(t.substr(1, t.size() - 2));
  if (t[0] == '[') {
    if (t.back() != ']') throw std::runtime_error(where + ": an unclosed flow list");
    const std::string inner = yamlTrim(t.substr(1, t.size() - 2));
    j.kind = lf::Json::Array;
    if (inner.empty()) return j;
    size_t at = 0;
    while (true) {
      size_t comma = inner.find(',', at);
      j.a.push_back(yamlScalar(inner.substr(at, comma == std::string::npos ? std::string::npos : comma - at), where));
      if (comma == std::string::npos) break;
      at = comma + 1;
    }
    return j;
  }
  if (t[0] == '{') {
    if (t.back() != '}') throw std::runtime_error(where + ": an unclosed flow mapping");
    const std::string inner = yamlTrim(t.substr(1, t.size() - 2));
    if (inner.find_first_of("{[") != std::string::npos) throw std::runtime_error(where + ": a nested flow collection; write it as a block");
    j.kind = lf::Json::Object;
    if (inner.empty()) return j;
    size_t at = 0;
    while (true) {
      size_t comma = inner.find(',', at);
      std::string part = inner.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
      size_t colon = part.find(':');
      if (colon == std::string::npos) throw std::runtime_error(where + ": \"" + yamlTrim(part) + "\" in a flow mapping is not \"key: value\"");
      std::string key = yamlTrim(part.substr(0, colon));
      if (!key.empty() && (key.front() == '\'' || key.front() == '"')) key = key.substr(1);
      if (!key.empty() && (key.back() == '\'' || key.back() == '"')) key.pop_back();
      j.o.push_back({key, yamlScalar(part.substr(colon + 1), where)});
      if (comma == std::string::npos) break;
      at = comma + 1;
    }
    return j;
  }
  if (std::string("&*!|>").find(t[0]) != std::string::npos)
    throw std::runtime_error(where + ": \"" + t + "\" is a YAML construct a potentials file does not use");
  return yamlString(t);
}

inline std::string yamlUncomment(const std::string& line) {
  char quote = 0;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quote) { if (c == quote) quote = 0; continue; }
    if (c == '"' || c == '\'') quote = c;
    else if (c == '#' && (i == 0 || isspace((unsigned char)line[i - 1]))) return line.substr(0, i);
  }
  return line;
}

inline lf::Json parseSimpleYaml(const std::string& text) {
  struct Line { int indent; std::string text; int number; };
  std::vector<Line> lines;
  int number = 0;
  size_t at0 = 0;
  while (at0 <= text.size()) {
    size_t end = text.find('\n', at0);
    std::string raw = text.substr(at0, end == std::string::npos ? std::string::npos : end - at0);
    if (!raw.empty() && raw.back() == '\r') raw.pop_back();
    ++number;
    if (raw.find('\t') != std::string::npos) throw std::runtime_error("line " + std::to_string(number) + ": a tab; YAML indents with spaces");
    std::string line = yamlUncomment(raw);
    while (!line.empty() && isspace((unsigned char)line.back())) line.pop_back();
    const std::string trimmed = yamlTrim(line);
    if (!(trimmed.empty() || trimmed == "---")) {
      int indent = (int)(line.size() - line.substr(line.find_first_not_of(' ')).size());
      lines.push_back({indent, trimmed, number});
    }
    if (end == std::string::npos) break;
    at0 = end + 1;
  }
  size_t at = 0;
  auto where = [](const Line& l) { return "line " + std::to_string(l.number); };
  static const std::regex colonRe(R"(:(\s|$))");
  auto startsDash = [](const std::string& s) { return s.rfind("- ", 0) == 0 || s == "-"; };
  std::function<lf::Json(int)> block, mapping, sequence;
  auto keyValue = [&](const Line& line, const std::string& t) {
    std::smatch m;
    if (!std::regex_search(t, m, colonRe)) throw std::runtime_error(where(line) + ": expected \"key: value\", got \"" + t + "\"");
    const size_t colon = m.position(0);
    std::string key = yamlTrim(t.substr(0, colon));
    if (key.size() >= 2 && ((key.front() == '"' && key.back() == '"') || (key.front() == '\'' && key.back() == '\''))) key = key.substr(1, key.size() - 2);
    return std::pair<std::string, std::string>{key, yamlTrim(t.substr(colon + 1))};
  };
  block = [&](int indent) -> lf::Json {
    if (at >= lines.size()) return lf::Json();
    const Line& first = lines[at];
    if (first.indent < indent) return lf::Json();
    if (startsDash(first.text)) return sequence(first.indent);
    return mapping(first.indent);
  };
  mapping = [&](int indent) -> lf::Json {
    lf::Json out; out.kind = lf::Json::Object;
    while (at < lines.size() && lines[at].indent == indent && lines[at].text.rfind("- ", 0) != 0) {
      const Line line = lines[at];
      auto [key, rest] = keyValue(line, line.text);
      ++at;
      if (out.get(key)) throw std::runtime_error(where(line) + ": " + key + " given twice");
      if (rest.empty()) out.o.push_back({key, at < lines.size() && lines[at].indent > indent ? block(lines[at].indent) : lf::Json()});
      else out.o.push_back({key, yamlScalar(rest, where(line))});
    }
    if (at < lines.size() && lines[at].indent > indent) throw std::runtime_error(where(lines[at]) + ": unexpected indentation");
    return out;
  };
  sequence = [&](int indent) -> lf::Json {
    lf::Json out; out.kind = lf::Json::Array;
    while (at < lines.size() && lines[at].indent == indent && startsDash(lines[at].text)) {
      const Line line = lines[at];
      const std::string rest = line.text == "-" ? "" : yamlTrim(line.text.substr(2));
      if (rest.empty()) {
        ++at;
        out.a.push_back(at < lines.size() && lines[at].indent > indent ? block(lines[at].indent) : lf::Json());
      } else if (std::regex_search(rest, colonRe) && rest[0] != '[' && rest[0] != '"' && rest[0] != '\'') {
        const std::string after = line.text.substr(2);
        const int childIndent = indent + 2 + (int)(after.size() - after.substr(after.find_first_not_of(' ')).size());
        lines[at] = {childIndent, rest, line.number};
        out.a.push_back(mapping(childIndent));
      } else {
        ++at;
        out.a.push_back(yamlScalar(rest, where(line)));
      }
    }
    return out;
  };
  lf::Json result = block(0);
  if (at < lines.size()) throw std::runtime_error(where(lines[at]) + ": unexpected \"" + lines[at].text + "\"");
  if (result.kind == lf::Json::Null) { result.kind = lf::Json::Object; }
  return result;
}

// potentials.js potentialSpecs over the parsed file
inline std::vector<Spec> potentialSpecs(const lf::Json& raw) {
  const lf::Json* list = raw.get("potentials");
  if (raw.kind != lf::Json::Object || !list) throw std::runtime_error("no potentials: list");
  if (list->kind != lf::Json::Array || list->a.empty()) throw std::runtime_error("the potentials list is empty; use the edm sampler");
  static const std::map<std::string, std::vector<std::string>> required = {
      {"SymmetricChainCOMPotential", {"buffer"}}, {"InterChainVDWOverlapPotential", {"buffer"}},
      {"IntraChainVDWOverlapPotential", {"buffer"}}, {"PoseBustersPotential", {"bond_buffer", "angle_buffer", "clash_buffer"}},
      {"ChiralAtomPotential", {"buffer"}}, {"StereoBondPotential", {"buffer"}}, {"PlanarBondPotential", {"buffer"}},
      {"PolymerLocalClashPotential", {"minimum_distance"}}};
  auto numberOf = [](const lf::Json& v, const std::string& what) {
    if (v.kind == lf::Json::Number) return v.n;
    if (v.kind == lf::Json::Bool) return v.b ? 1.0 : 0.0;
    throw std::runtime_error(what + " is not a number");
  };
  std::vector<Spec> out;
  for (auto& entry : list->a) {
    const lf::Json* target = entry.get("_target_");
    if (!target || target->kind != lf::Json::String) throw std::runtime_error("a potential without _target_");
    const std::string kind = target->s.substr(target->s.rfind('.') + 1);
    if (std::find(types().begin(), types().end(), kind) == types().end()) {
      std::string all;
      for (auto& t : types()) all += (all.empty() ? "" : ",") + t;
      throw std::runtime_error(kind + " is not one of BioEmu2's ported potentials " + all);
    }
    Spec s; s.kind = kind;
    const lf::Json* params = entry.get("parameters");
    if (params && params->kind == lf::Json::Object)
      for (auto& kv : params->o) {
        const std::string& k = kv.first;
        const lf::Json& v = kv.second;
        if (v.kind == lf::Json::Object) {
          const lf::Json* sched = v.get("_target_");
          const std::string name = sched && sched->kind == lf::Json::String ? sched->s.substr(sched->s.rfind('.') + 1) : "";
          auto arg = [&](const char* a) {
            const lf::Json* x = v.get(a);
            if (!x) throw std::runtime_error(name + " needs " + a);
            return numberOf(*x, std::string(a));
          };
          auto list = [&](const char* a) {
            const lf::Json* x = v.get(a);
            if (!x || x->kind != lf::Json::Array) throw std::runtime_error(name + " needs a list " + a);
            std::vector<double> o; for (auto& y : x->a) o.push_back(numberOf(y, a)); return o;
          };
          if (name == "ExponentialInterpolation") s.params[k] = Param::exponential(arg("start"), arg("end"), arg("alpha"));
          else if (name == "PiecewiseStepFunction") s.params[k] = Param::piecewise(list("thresholds"), list("values"));
          else if (name == "LinearInterpolation") {
            Param p; p.kind = Param::LINEAR; p.startT = arg("start_t"); p.endT = arg("end_t");
            p.startVal = arg("start_val"); p.endVal = arg("end_val"); s.params[k] = p;
          } else throw std::runtime_error("unknown schedule " + name);
        } else {
          s.params[k] = Param::constant(numberOf(v, kind + "." + k));
        }
      }
    auto need = required.find(kind);
    if (need != required.end())
      for (auto& r : need->second) if (!s.has(r)) throw std::runtime_error(kind + " needs parameters[" + r + "]");
    const lf::Json* tol = entry.get("sigma_tolerance");
    if ((kind == "BondLengthPotential" || kind == "BondAnglePotential") && !tol) throw std::runtime_error(kind + " needs sigma_tolerance");
    if (tol) s.sigmaTolerance = numberOf(*tol, "sigma_tolerance");
    for (auto& [k, v] : entry.o)
      if (k != "_target_" && k != "parameters" && k != "sigma_tolerance") throw std::runtime_error(kind + " takes no " + k);
    out.push_back(s);
  }
  return out;
}

}  // namespace be2steer
