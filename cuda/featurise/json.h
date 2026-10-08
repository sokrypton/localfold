// A JSON value and reader for the featuriser - only what an AlphaFold 3 job needs: objects keep their keys
// in the order the text gives them (JavaScript's Object.keys order for string keys, which web/job-json.js
// relies on), numbers are doubles (JSON.parse's), strings are UTF-8.
#pragma once
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lf {

struct Json {
  enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Json> a;
  std::vector<std::pair<std::string, Json>> o;

  bool isNull() const { return kind == Null; }
  bool isString() const { return kind == String; }
  bool isNumber() const { return kind == Number; }
  bool isArray() const { return kind == Array; }
  bool isObject() const { return kind == Object; }
  // a key's value, or nullptr: JavaScript's `undefined`
  const Json* get(const std::string& key) const {
    if (kind != Object) return nullptr;
    for (auto& [k, v] : o)
      if (k == key) return &v;
    return nullptr;
  }
  // `undefined` or `null`, which `x ?? y` treats alike
  static bool absent(const Json* v) { return v == nullptr || v->kind == Null; }
};

class JsonReader {
 public:
  explicit JsonReader(const std::string& text) : t(text) {}
  Json parse() {
    Json v = value();
    space();
    if (i != t.size()) fail("Unexpected non-whitespace character after JSON");
    return v;
  }

 private:
  const std::string& t;
  size_t i = 0;
  [[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error(what + " at position " + std::to_string(i));
  }
  void space() {
    while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\r' || t[i] == '\t')) ++i;
  }
  Json value() {
    space();
    if (i >= t.size()) fail("Unexpected end of JSON input");
    char c = t[i];
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') { Json v; v.kind = Json::String; v.s = string(); return v; }
    if (c == 't' && t.compare(i, 4, "true") == 0) { i += 4; Json v; v.kind = Json::Bool; v.b = true; return v; }
    if (c == 'f' && t.compare(i, 5, "false") == 0) { i += 5; Json v; v.kind = Json::Bool; return v; }
    if (c == 'n' && t.compare(i, 4, "null") == 0) { i += 4; return Json{}; }
    if (c == '-' || (c >= '0' && c <= '9')) return number();
    fail(std::string("Unexpected token ") + c);
  }
  Json object() {
    Json v; v.kind = Json::Object; ++i;
    space();
    if (i < t.size() && t[i] == '}') { ++i; return v; }
    for (;;) {
      space();
      if (i >= t.size() || t[i] != '"') fail("Expected property name");
      std::string key = string();
      space();
      if (i >= t.size() || t[i] != ':') fail("Expected ':'");
      ++i;
      Json item = value();
      // a repeated key keeps its FIRST position and its LAST value, as JSON.parse does
      bool replaced = false;
      for (auto& [k, existing] : v.o)
        if (k == key) { existing = std::move(item); replaced = true; break; }
      if (!replaced) v.o.emplace_back(std::move(key), std::move(item));
      space();
      if (i < t.size() && t[i] == ',') { ++i; continue; }
      if (i < t.size() && t[i] == '}') { ++i; return v; }
      fail("Expected ',' or '}'");
    }
  }
  Json array() {
    Json v; v.kind = Json::Array; ++i;
    space();
    if (i < t.size() && t[i] == ']') { ++i; return v; }
    for (;;) {
      v.a.push_back(value());
      space();
      if (i < t.size() && t[i] == ',') { ++i; continue; }
      if (i < t.size() && t[i] == ']') { ++i; return v; }
      fail("Expected ',' or ']'");
    }
  }
  static void utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
      out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
    } else {
      out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
      out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
    }
  }
  unsigned hex4() {
    if (i + 4 > t.size()) fail("Bad unicode escape");
    unsigned v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = t[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("Bad unicode escape");
    }
    return v;
  }
  std::string string() {
    ++i;
    std::string out;
    while (i < t.size()) {
      char c = t[i++];
      if (c == '"') return out;
      if ((unsigned char)c < 0x20) fail("Bad control character in string literal");
      if (c != '\\') { out += c; continue; }
      if (i >= t.size()) break;
      char e = t[i++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < t.size() && t[i] == '\\' && t[i + 1] == 'u') {
            size_t save = i;
            i += 2;
            unsigned low = hex4();
            if (low >= 0xDC00 && low < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            else { i = save; }
          }
          utf8(out, cp);
          break;
        }
        default: fail("Bad escaped character");
      }
    }
    fail("Unterminated string in JSON");
  }
  Json number() {
    size_t start = i;
    if (t[i] == '-') ++i;
    if (i < t.size() && t[i] == '0') ++i;
    else if (i < t.size() && t[i] >= '1' && t[i] <= '9') while (i < t.size() && isdigit((unsigned char)t[i])) ++i;
    else fail("No number after minus sign");
    if (i < t.size() && t[i] == '.') {
      ++i;
      if (i >= t.size() || !isdigit((unsigned char)t[i])) fail("Unterminated fractional number");
      while (i < t.size() && isdigit((unsigned char)t[i])) ++i;
    }
    if (i < t.size() && (t[i] == 'e' || t[i] == 'E')) {
      ++i;
      if (i < t.size() && (t[i] == '+' || t[i] == '-')) ++i;
      if (i >= t.size() || !isdigit((unsigned char)t[i])) fail("Exponent part is missing a number");
      while (i < t.size() && isdigit((unsigned char)t[i])) ++i;
    }
    Json v; v.kind = Json::Number; v.n = std::strtod(t.substr(start, i - start).c_str(), nullptr);
    return v;
  }
};

inline Json parseJson(const std::string& text) { return JsonReader(text).parse(); }

}  // namespace lf
