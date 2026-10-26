// `parseConditionSpecYaml` (shared/rfd4p/pipeline/spec.js), natively: the YAML a condition spec is
// written in - block mappings and lists, flow `{...}` / `[...]`, quoted and plain scalars - read
// into the same `lf::Json` a JSON job is.
//
// 🔴 THE PAGE'S SUBSET, LINE FOR LINE, NOT A YAML LIBRARY. Upstream reads its specs with PyYAML,
// and the page with this subset; a spec both ports read must mean one thing on both, so this
// follows the page's reader exactly - including where the subset stops - and gate.py holds the
// two to the same featurised bytes on a YAML job.
#pragma once
#include "../../featurise/json.h"
#include <string>
#include <vector>

namespace rfd4p {
namespace yaml {

using lf::Json;
using std::string;
using std::vector;

inline Json str(const string& s) { Json v; v.kind = Json::String; v.s = s; return v; }
inline Json boolean(bool b) { Json v; v.kind = Json::Bool; v.b = b; return v; }
inline Json number(double n) { Json v; v.kind = Json::Number; v.n = n; return v; }
inline Json object() { Json v; v.kind = Json::Object; return v; }
inline Json array() { Json v; v.kind = Json::Array; return v; }
inline void put(Json& obj, const string& k, Json v) {
  for (auto& kv : obj.o) if (kv.first == k) { kv.second = std::move(v); return; }
  obj.o.push_back({k, std::move(v)});
}
// A key as `String(parseYamlScalar(...))` spells it.
inline string keyText(const Json& v) {
  if (v.kind == Json::String) return v.s;
  if (v.kind == Json::Bool) return v.b ? "true" : "false";
  if (v.kind == Json::Null) return "null";
  char buf[64];
  if (v.n == (long long)v.n) snprintf(buf, sizeof buf, "%lld", (long long)v.n);
  else snprintf(buf, sizeof buf, "%.17g", v.n);
  return buf;
}
inline string trimmed(const string& s) {
  size_t a = 0, b = s.size();
  while (a < b && isspace((unsigned char)s[a])) ++a;
  while (b > a && isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}

inline string stripComment(const string& line) {
  bool inSingle = false, inDouble = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '\'' && !inDouble) {
      if (inSingle && i + 1 < line.size() && line[i + 1] == '\'') { ++i; continue; }
      inSingle = !inSingle;
    } else if (ch == '"' && !inSingle) {
      inDouble = !inDouble;
    } else if (ch == '#' && !inSingle && !inDouble) {
      if (i == 0 || isspace((unsigned char)line[i - 1])) return line.substr(0, i);
    }
  }
  return line;
}

inline Json scalar(const string& raw) {
  const string s = trimmed(raw);
  if (s.empty()) return Json{};
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
    const string inner = s.substr(1, s.size() - 2);
    string out;
    if (s.front() == '\'') {
      for (size_t i = 0; i < inner.size(); ++i) {
        out += inner[i];
        if (inner[i] == '\'' && i + 1 < inner.size() && inner[i + 1] == '\'') ++i;
      }
    } else {
      for (size_t i = 0; i < inner.size(); ++i) {
        if (inner[i] == '\\' && i + 1 < inner.size()) { out += inner[++i]; continue; }
        out += inner[i];
      }
    }
    return str(out);
  }
  if (s == "true" || s == "True" || s == "TRUE") return boolean(true);
  if (s == "false" || s == "False" || s == "FALSE") return boolean(false);
  if (s == "null" || s == "Null" || s == "NULL" || s == "~") return Json{};
  // /^-?\d+$/ and /^-?\d+\.\d*(?:[eE][+-]?\d+)?$/
  size_t i = s[0] == '-' ? 1 : 0;
  const size_t d0 = i;
  while (i < s.size() && isdigit((unsigned char)s[i])) ++i;
  if (i > d0 && i == s.size()) return number((double)strtoll(s.c_str(), nullptr, 10));
  if (i > d0 && i < s.size() && s[i] == '.') {
    size_t j = i + 1;
    while (j < s.size() && isdigit((unsigned char)s[j])) ++j;
    if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
      size_t k = j + 1;
      if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
      const size_t k0 = k;
      while (k < s.size() && isdigit((unsigned char)s[k])) ++k;
      if (k > k0 && k == s.size()) return number(strtod(s.c_str(), nullptr));
    } else if (j == s.size()) {
      return number(strtod(s.c_str(), nullptr));
    }
  }
  return str(s);
}

inline vector<string> splitTopLevel(const string& s, char sep) {
  vector<string> parts;
  string buf;
  int depth = 0;
  bool inSingle = false, inDouble = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char ch = s[i];
    if (ch == '\'' && !inDouble) {
      if (inSingle && i + 1 < s.size() && s[i + 1] == '\'') { buf += "''"; ++i; continue; }
      inSingle = !inSingle;
      buf += ch;
      continue;
    }
    if (ch == '"' && !inSingle) {
      if (i == 0 || s[i - 1] != '\\') inDouble = !inDouble;
      buf += ch;
      continue;
    }
    if (!inSingle && !inDouble) {
      if (ch == '[' || ch == '{' || ch == '(') ++depth;
      else if (ch == ']' || ch == '}' || ch == ')') --depth;
      else if (ch == sep && depth == 0) { parts.push_back(buf); buf.clear(); continue; }
    }
    buf += ch;
  }
  parts.push_back(buf);
  return parts;
}

inline long topLevelColon(const string& s) {
  int depth = 0;
  bool inSingle = false, inDouble = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char ch = s[i];
    if (ch == '\'' && !inDouble) {
      if (inSingle && i + 1 < s.size() && s[i + 1] == '\'') { ++i; continue; }
      inSingle = !inSingle;
      continue;
    }
    if (ch == '"' && !inSingle) {
      if (i == 0 || s[i - 1] != '\\') inDouble = !inDouble;
      continue;
    }
    if (!inSingle && !inDouble) {
      if (ch == '[' || ch == '{' || ch == '(') ++depth;
      else if (ch == ']' || ch == '}' || ch == ')') --depth;
      else if (ch == ':' && depth == 0 && (i + 1 == s.size() || isspace((unsigned char)s[i + 1]))) return (long)i;
    }
  }
  return -1;
}

inline Json flow(const string& text) {
  const string s = trimmed(text);
  if (s.size() >= 2 && s.front() == '{' && s.back() == '}') {
    Json obj = object();
    const string inner = trimmed(s.substr(1, s.size() - 2));
    if (inner.empty()) return obj;
    for (const string& part : splitTopLevel(inner, ',')) {
      const string p = trimmed(part);
      if (p.empty()) continue;
      const long c = topLevelColon(p);
      if (c < 0) throw std::runtime_error("Invalid YAML flow mapping entry: " + p);
      put(obj, keyText(scalar(p.substr(0, (size_t)c))), flow(p.substr((size_t)c + 1)));
    }
    return obj;
  }
  if (s.size() >= 2 && s.front() == '[' && s.back() == ']') {
    Json arr = array();
    const string inner = trimmed(s.substr(1, s.size() - 2));
    if (inner.empty()) return arr;
    for (const string& part : splitTopLevel(inner, ',')) {
      const string p = trimmed(part);
      if (!p.empty()) arr.a.push_back(flow(p));
    }
    return arr;
  }
  return scalar(s);
}

class Reader {
 public:
  explicit Reader(const string& text) {
    vector<string> raw;
    {
      size_t from = 0;
      while (from <= text.size()) {
        size_t nl = text.find('\n', from);
        string line = text.substr(from, nl == string::npos ? string::npos : nl - from);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        raw.push_back(line);
        if (nl == string::npos) break;
        from = nl + 1;
      }
    }
    auto unclosed = [](const string& s) {
      int d = 0;
      bool inS = false, inD = false;
      for (size_t j = 0; j < s.size(); ++j) {
        const char c = s[j];
        if (c == '\'' && !inD) inS = !inS;
        else if (c == '"' && !inS && (j == 0 || s[j - 1] != '\\')) inD = !inD;
        else if (!inS && !inD) {
          if (c == '{' || c == '[') ++d;
          else if (c == '}' || c == ']') --d;
        }
      }
      return d;
    };
    for (size_t i = 0; i < raw.size(); ++i) {
      string cleaned = stripComment(raw[i]);
      while (!cleaned.empty() && isspace((unsigned char)cleaned.back())) cleaned.pop_back();
      if (trimmed(cleaned).empty()) continue;
      size_t indent = 0;
      while (indent < cleaned.size() && cleaned[indent] == ' ') ++indent;
      string content = cleaned.substr(indent);
      while (unclosed(content) > 0 && i + 1 < raw.size()) {
        ++i;
        const string next = trimmed(stripComment(raw[i]));
        if (!next.empty()) content += " " + next;
      }
      lines_.push_back({indent, content});
    }
  }
  Json parse() {
    Json v = block(0);
    return v.kind == Json::Null && lines_.empty() ? object() : (v.kind == Json::Null ? object() : v);
  }

 private:
  struct Line { size_t indent; string content; };
  vector<Line> lines_;
  size_t pos_ = 0;
  static bool isItem(const string& c) { return c.rfind("- ", 0) == 0 || c == "-"; }
  static Json valueOf(const string& v) { return v[0] == '{' || v[0] == '[' ? flow(v) : scalar(v); }

  Json block(size_t minIndent) {
    if (pos_ >= lines_.size() || lines_[pos_].indent < minIndent) return Json{};
    if (isItem(lines_[pos_].content)) {
      Json arr = array();
      const size_t listIndent = lines_[pos_].indent;
      while (pos_ < lines_.size() && lines_[pos_].indent == listIndent && isItem(lines_[pos_].content)) {
        const string rest = trimmed(lines_[pos_].content.substr(1));
        ++pos_;
        if (rest.empty()) { arr.a.push_back(block(listIndent + 1)); continue; }
        if (rest[0] == '{' || rest[0] == '[') { arr.a.push_back(flow(rest)); continue; }
        const long c = topLevelColon(rest);
        if (c < 0) { arr.a.push_back(scalar(rest)); continue; }
        Json item = object();
        const string k = keyText(scalar(rest.substr(0, (size_t)c)));
        const string v = trimmed(rest.substr((size_t)c + 1));
        put(item, k, v.empty() ? block(listIndent + 1) : valueOf(v));
        while (pos_ < lines_.size() && lines_[pos_].indent > listIndent) {
          const string sub = lines_[pos_].content;
          const long sc = topLevelColon(sub);
          if (sc < 0) break;
          const string sk = keyText(scalar(sub.substr(0, (size_t)sc)));
          const string sv = trimmed(sub.substr((size_t)sc + 1));
          const size_t subIndent = lines_[pos_].indent;
          ++pos_;
          put(item, sk, sv.empty() ? block(subIndent + 1) : valueOf(sv));
        }
        arr.a.push_back(item);
      }
      return arr;
    }
    Json obj = object();
    const size_t mapIndent = lines_[pos_].indent;
    while (pos_ < lines_.size() && lines_[pos_].indent == mapIndent && lines_[pos_].content[0] != '-') {
      const string line = lines_[pos_].content;
      const long c = topLevelColon(line);
      if (c < 0) throw std::runtime_error("Invalid YAML mapping line: " + line);
      const string k = keyText(scalar(line.substr(0, (size_t)c)));
      const string v = trimmed(line.substr((size_t)c + 1));
      ++pos_;
      put(obj, k, v.empty() ? block(mapIndent + 1) : valueOf(v));
    }
    return obj;
  }
};

/** A spec file's text as JSON: JSON when it starts with `{`, else the YAML subset. */
inline Json parseSpec(const string& text) {
  const string t = trimmed(text);
  if (!t.empty() && t[0] == '{') {
    try {
      return lf::JsonReader(t).parse();
    } catch (const std::exception&) {
      return flow(t);
    }
  }
  return Reader(text).parse();
}

}  // namespace yaml
}  // namespace rfd4p
