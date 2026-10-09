// A small JSON reader: a bundle's manifest.json, a job's summary. Numbers are doubles.
#pragma once
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mt {
struct Json {
  enum Type { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false; double num = 0; std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;      // (in the file's order)
  const Json* get(const std::string& k) const {
    for (auto& [key, v] : obj) if (key == k) return &v;
    return nullptr;
  }
  static Json parse(const std::string& s) {
    size_t i = 0;
    Json j = value(s, i);
    return j;
  }
 private:
  static void ws(const std::string& s, size_t& i) { while (i < s.size() && isspace((unsigned char)s[i])) ++i; }
  static std::string string(const std::string& s, size_t& i) {
    std::string out; ++i;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\') {
        ++i;
        char c = s[i];
        if (c == 'n') out += '\n'; else if (c == 't') out += '\t'; else if (c == 'r') out += '\r';
        else if (c == 'b') out += '\b'; else if (c == 'f') out += '\f';
        else if (c == 'u') {
          unsigned code = (unsigned)strtoul(s.substr(i + 1, 4).c_str(), nullptr, 16); i += 4;
          if (code < 0x80) out += (char)code;
          else if (code < 0x800) { out += (char)(0xc0 | (code >> 6)); out += (char)(0x80 | (code & 0x3f)); }
          else { out += (char)(0xe0 | (code >> 12)); out += (char)(0x80 | ((code >> 6) & 0x3f)); out += (char)(0x80 | (code & 0x3f)); }
        } else out += c;
        ++i;
      } else out += s[i++];
    }
    ++i;
    return out;
  }
  static Json value(const std::string& s, size_t& i) {
    ws(s, i);
    Json j;
    if (i >= s.size()) throw std::runtime_error("JSON ends early");
    char c = s[i];
    if (c == '{') {
      j.t = OBJ; ++i; ws(s, i);
      if (s[i] == '}') { ++i; return j; }
      while (true) {
        ws(s, i);
        std::string k = string(s, i);
        ws(s, i); ++i;                    // ':'
        j.obj.emplace_back(k, value(s, i));
        ws(s, i);
        if (s[i] == ',') { ++i; continue; }
        ++i; break;                       // '}'
      }
    } else if (c == '[') {
      j.t = ARR; ++i; ws(s, i);
      if (s[i] == ']') { ++i; return j; }
      while (true) {
        j.arr.push_back(value(s, i));
        ws(s, i);
        if (s[i] == ',') { ++i; continue; }
        ++i; break;
      }
    } else if (c == '"') { j.t = STR; j.str = string(s, i); }
    else if (!s.compare(i, 4, "true")) { j.t = BOOL; j.b = true; i += 4; }
    else if (!s.compare(i, 5, "false")) { j.t = BOOL; i += 5; }
    else if (!s.compare(i, 4, "null")) { i += 4; }
    else { j.t = NUM; char* end; j.num = strtod(s.c_str() + i, &end); i = end - s.c_str(); }
    return j;
  }
};
}  // namespace mt
