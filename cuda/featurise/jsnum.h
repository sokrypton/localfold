// JavaScript's number semantics where the featuriser's output depends on them: String(number) (the
// exporter writes every scalar entry with it), Number.parseFloat (the CCD reader), Math.round.
#pragma once
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace lf {

// ECMAScript Number::toString(10): the shortest digits that round-trip, laid out by its rules
inline std::string jsNumber(double v) {
  if (std::isnan(v)) return "NaN";
  if (v == 0) return "0";                         // -0 too
  if (std::isinf(v)) return v < 0 ? "-Infinity" : "Infinity";
  std::string sign = v < 0 ? "-" : "";
  double a = std::fabs(v);
  char buf[64];
#ifdef __APPLE__
  // (std::to_chars for a double needs macOS 13.3's libc++, and the Metal port runs on 13.0: the fewest
  // significant digits whose correctly rounded decimal reads back as the same double - which is what
  // to_chars' shortest form is, since printf rounds to the nearest)
  for (int p = 0; p < 17; ++p) {
    snprintf(buf, sizeof buf, "%.*e", p, a);
    if (strtod(buf, nullptr) == a) break;
  }
  std::string sci(buf);                           // d[.ddd]e[+-]XX
#else
  auto r = std::to_chars(buf, buf + sizeof buf, a, std::chars_format::scientific);
  std::string sci(buf, r.ptr);                    // d[.ddd]e[+-]XX
#endif
  size_t e = sci.find('e');
  std::string mant = sci.substr(0, e);
  int exp10 = std::atoi(sci.c_str() + e + 1);
  std::string digits;
  for (char c : mant) if (c != '.') digits += c;
  int k = (int)digits.size(), n = exp10 + 1;      // value = 0.digits * 10^n
  std::string out;
  if (k <= n && n <= 21) {
    out = digits + std::string(n - k, '0');
  } else if (0 < n && n <= 21) {
    out = digits.substr(0, n) + "." + digits.substr(n);
  } else if (-6 < n && n <= 0) {
    out = "0." + std::string(-n, '0') + digits;
  } else {
    int shown = n - 1;
    std::string ex = (shown < 0 ? "-" : "+") + std::to_string(std::abs(shown));
    out = k == 1 ? digits + "e" + ex : digits.substr(0, 1) + "." + digits.substr(1) + "e" + ex;
  }
  return sign + out;
}

// Number.parseFloat: leading whitespace, then the longest prefix that is a decimal literal (or Infinity);
// NaN where there is none
inline double jsParseFloat(const std::string& text) {
  size_t i = 0;
  while (i < text.size() && std::isspace((unsigned char)text[i])) ++i;
  size_t start = i;
  if (i < text.size() && (text[i] == '+' || text[i] == '-')) ++i;
  if (text.compare(i, 8, "Infinity") == 0) return text[start] == '-' ? -INFINITY : INFINITY;
  size_t digitsStart = i;
  while (i < text.size() && std::isdigit((unsigned char)text[i])) ++i;
  bool any = i > digitsStart;
  if (i < text.size() && text[i] == '.') {
    ++i;
    size_t f = i;
    while (i < text.size() && std::isdigit((unsigned char)text[i])) ++i;
    any = any || i > f;
  }
  if (!any) return NAN;
  if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
    size_t save = i++;
    if (i < text.size() && (text[i] == '+' || text[i] == '-')) ++i;
    size_t ed = i;
    while (i < text.size() && std::isdigit((unsigned char)text[i])) ++i;
    if (i == ed) i = save;
  }
  return std::strtod(text.substr(start, i - start).c_str(), nullptr);
}

// Number.parseInt(text, 10): leading whitespace, a sign, the longest run of digits; NaN where there is none
inline double jsParseInt(const std::string& text) {
  size_t i = 0;
  while (i < text.size() && std::isspace((unsigned char)text[i])) ++i;
  bool negative = false;
  if (i < text.size() && (text[i] == '+' || text[i] == '-')) negative = text[i++] == '-';
  size_t start = i;
  while (i < text.size() && std::isdigit((unsigned char)text[i])) ++i;
  if (i == start) return NAN;
  double v = std::strtod(text.substr(start, i - start).c_str(), nullptr);
  return negative ? -v : v;
}

// Number(text): the whole string, trimmed, must be a number - "" and blanks are 0, anything else NaN
inline double jsNumberOf(const std::string& text) {
  size_t a = 0, b = text.size();
  auto ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; };
  while (a < b && ws(text[a])) ++a;
  while (b > a && ws(text[b - 1])) --b;
  std::string t = text.substr(a, b - a);
  if (t.empty()) return 0;
  if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
    for (size_t i = 2; i < t.size(); ++i) if (!std::isxdigit((unsigned char)t[i])) return NAN;
    return (double)std::stoull(t.substr(2), nullptr, 16);
  }
  size_t i = 0;
  if (t[i] == '+' || t[i] == '-') ++i;
  if (t.compare(i, std::string::npos, "Infinity") == 0) return t[0] == '-' ? -INFINITY : INFINITY;
  size_t d = i;
  while (i < t.size() && std::isdigit((unsigned char)t[i])) ++i;
  bool any = i > d;
  if (i < t.size() && t[i] == '.') { ++i; size_t f = i; while (i < t.size() && std::isdigit((unsigned char)t[i])) ++i; any = any || i > f; }
  if (!any) return NAN;
  if (i < t.size() && (t[i] == 'e' || t[i] == 'E')) {
    ++i;
    if (i < t.size() && (t[i] == '+' || t[i] == '-')) ++i;
    size_t e = i;
    while (i < t.size() && std::isdigit((unsigned char)t[i])) ++i;
    if (i == e) return NAN;
  }
  if (i != t.size()) return NAN;
  return std::strtod(t.c_str(), nullptr);
}

// Math.round: halves toward +Infinity
inline double jsRound(double v) { double c = std::ceil(v); return c - 0.5 <= v ? c : c - 1.0; }   // Math.round, as V8 rounds

}  // namespace lf
