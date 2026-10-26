// Upstream's selection language, natively: `atomworks.io.utils.query.QueryExpression` and its
// `sel('...')` path DSL, as RFD4-Proteina's `resolve_mask` evaluates a condition's `select:`.
//
// 🔴 A PORT OF shared/rfd4p/pipeline/query.js, NOT A SECOND READING OF THE LANGUAGE. The query
// is a PYTHON expression evaluated with numpy's semantics, and every way that differs from what
// it reads like - `&` binding tighter than a comparison, chained comparisons, `np.isin` typing a
// mixed list as strings, a NameError for a bare chain id - is decided by atomworks, not here.
// This took a union of `sel(...)` terms and refused everything else; tools/check-rfd4p-query.py
// holds it, the page and atomworks to the same mask on every atom of every shipped structure.
//
// 🔴 AND A FIELD IS SPLIT AT THE TOP LEVEL ONLY. `[54,56]` contains a comma and `A/*/[1-3,7]/*`
// contains both separators, so splitting on `,` or `/` naively tears a list in half.
#pragma once
#include "atoms.h"
#include <charconv>
#include <functional>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace rfd4p {
namespace selection {

using std::string;
using std::vector;

/** `splitTopLevel`: split on `sep`, ignoring one inside `[...]` or quotes. */
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
      if (ch == '[' || ch == '(') ++depth;
      else if (ch == ']' || ch == ')') --depth;
      else if (ch == sep && depth == 0) { parts.push_back(buf); buf.clear(); continue; }
    }
    buf += ch;
  }
  parts.push_back(buf);
  return parts;
}

inline string trim(string v) {
  while (!v.empty() && isspace((unsigned char)v.front())) v.erase(v.begin());
  while (!v.empty() && isspace((unsigned char)v.back())) v.pop_back();
  return v;
}

namespace query {

struct SyntaxError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------- the tokenizer

struct Tok {
  enum T { STR, NUM, NAME, KW, OP, END } t;
  string v;
  double d = 0;
};

inline bool isKeyword(const string& w) {
  static const std::set<string> K = {"and", "or", "not", "in", "is", "True", "False", "None",
                                     "if", "else", "lambda", "for", "await", "yield"};
  return K.count(w) > 0;
}

inline string decodeEscapes(const string& body) {
  string out;
  for (size_t i = 0; i < body.size(); ++i) {
    const char c = body[i];
    if (c != '\\' || i + 1 >= body.size()) { out += c; continue; }
    const char n = body[++i];
    switch (n) {
      case 'n': out += '\n'; break;
      case 't': out += '\t'; break;
      case 'r': out += '\r'; break;
      case '\\': out += '\\'; break;
      case '\'': out += '\''; break;
      case '"': out += '"'; break;
      case 'a': out += '\x07'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'v': out += '\v'; break;
      case '0': out += '\0'; break;
      case '\n': break;
      default: out += '\\'; out += n;
    }
  }
  return out;
}

/** A Python number literal at `i` (query.js's one regex, by hand): its length, or 0. */
inline size_t numberLength(const string& s, size_t i, bool* complex) {
  auto in = [&](size_t k, const char* set) { return k < s.size() && strchr(set, s[k]) && s[k]; };
  auto dig = [&](size_t k) { return k < s.size() && isdigit((unsigned char)s[k]); };
  *complex = false;
  size_t j = i;
  const char* radix[3][2] = {{"xX", "0123456789abcdefABCDEF_"}, {"oO", "01234567_"}, {"bB", "01_"}};
  bool matched = false;
  if (i < s.size() && s[i] == '0') {
    for (auto& r : radix) {
      if (in(i + 1, r[0]) && in(i + 2, r[1])) {
        j = i + 2;
        while (in(j, r[1])) ++j;
        matched = true;
        break;
      }
    }
  }
  if (!matched) {
    if (dig(i)) {
      j = i;
      while (in(j, "0123456789_")) ++j;
      if (j < s.size() && s[j] == '.') ++j;
      while (in(j, "0123456789_")) ++j;
    } else if (i < s.size() && s[i] == '.' && dig(i + 1)) {
      j = i + 1;
      while (in(j, "0123456789_")) ++j;
    } else {
      return 0;
    }
    if (in(j, "eE")) {
      size_t e = j + 1;
      if (in(e, "+-")) ++e;
      if (dig(e)) {
        ++e;
        while (in(e, "0123456789_")) ++e;
        j = e;
      }
    }
  }
  if (in(j, "jJ")) { *complex = true; ++j; }
  return j - i;
}

inline vector<Tok> tokenize(const string& src) {
  vector<Tok> toks;
  static const char* OPS[] = {"**", "//", "<<", ">>", "<=", ">=", "==", "!=", "(", ")", "[", "]",
                              ",", "<", ">", "&", "|", "^", "~", "+", "-", "*", "/", "%", "@",
                              ".", ":"};
  size_t i = 0;
  while (i < src.size()) {
    const char c = src[i];
    if (isspace((unsigned char)c)) { ++i; continue; }
    // A string, with an optional prefix of one or two of r/u/b/f.
    size_t pl = 0;
    while (pl < 2 && i + pl < src.size() && strchr("rRuUbBfF", src[i + pl]) && src[i + pl]) ++pl;
    if (i + pl < src.size() && (src[i + pl] == '\'' || src[i + pl] == '"')) {
      string prefix = src.substr(i, pl);
      for (char& ch : prefix) ch = (char)tolower((unsigned char)ch);
      const char quote = src[i + pl];
      size_t j = i + pl + 1;
      const string q3(3, quote);
      const bool triple = src.compare(j - 1, 3, q3) == 0;
      if (triple) j += 2;
      string body;
      bool closed = false;
      while (j < src.size()) {
        if (src[j] == '\\' && j + 1 < src.size()) { body += src[j]; body += src[j + 1]; j += 2; continue; }
        if (triple ? src.compare(j, 3, q3) == 0 : src[j] == quote) { closed = true; j += triple ? 3 : 1; break; }
        if (!triple && src[j] == '\n') break;
        body += src[j];
        ++j;
      }
      if (!closed) throw SyntaxError("unterminated string literal");
      if (prefix.find('b') != string::npos || prefix.find('f') != string::npos) {
        throw Error(string("Unsupported node type: ") +
                    (prefix.find('b') != string::npos ? "bytes" : "JoinedStr"));
      }
      toks.push_back({Tok::STR, prefix.find('r') != string::npos ? body : decodeEscapes(body)});
      i = j;
      continue;
    }
    bool complex = false;
    if (const size_t len = numberLength(src, i, &complex)) {
      if (complex) throw Error("Unsupported node type: complex");
      string text;
      for (size_t k = i; k < i + len; ++k) if (src[k] != '_') text += src[k];
      double v = 0;
      if (text.size() > 1 && text[0] == '0' && strchr("xXoObB", text[1])) {
        const int base = strchr("xX", text[1]) ? 16 : strchr("oO", text[1]) ? 8 : 2;
        for (size_t k = 2; k < text.size(); ++k) {
          const char ch = (char)tolower((unsigned char)text[k]);
          v = v * base + (isdigit((unsigned char)ch) ? ch - '0' : ch - 'a' + 10);
        }
      } else {
        v = strtod(text.c_str(), nullptr);
      }
      Tok t{Tok::NUM, text};
      t.d = v;
      toks.push_back(t);
      i += len;
      continue;
    }
    if (isalpha((unsigned char)c) || c == '_') {
      size_t j = i;
      while (j < src.size() && (isalnum((unsigned char)src[j]) || src[j] == '_')) ++j;
      const string w = src.substr(i, j - i);
      toks.push_back({isKeyword(w) ? Tok::KW : Tok::NAME, w});
      i = j;
      continue;
    }
    const char* op = nullptr;
    for (const char* o : OPS) {
      if (src.compare(i, strlen(o), o) == 0) { op = o; break; }
    }
    if (!op) throw SyntaxError(string("invalid character '") + c + "'");
    toks.push_back({Tok::OP, op});
    i += strlen(op);
  }
  toks.push_back({Tok::END, ""});
  return toks;
}

// ---------------------------------------------------------------- the parser (Python's grammar)

/** One scalar: a number (a boolean is 0/1, as numpy compares it), a string, None, or an object. */
struct Q {
  enum T { NUL, NUM, STR, OBJ } t = NUL;
  double d = 0;
  string s;
  static Q num(double v) { Q q; q.t = NUM; q.d = v; return q; }
  static Q str(string v) { Q q; q.t = STR; q.s = std::move(v); return q; }
  static Q obj() { Q q; q.t = OBJ; return q; }
};

struct Node {
  enum K { CONST, LIST, TUPLE, NAME, COMPARE, BOOL, BINOP, UNARY, CALL } k;
  Q v;
  string id, op;
  vector<string> ops;
  vector<Node> kids;   // elts / values / comparators (after `left`) / args
  bool keywords = false;
};

class Parser {
 public:
  explicit Parser(const string& src) : toks_(tokenize(src)) {}
  Node parse() {
    Node tree = expression();
    // 🔴 THE WHOLE STRING IS ONE EXPRESSION: `ast.parse(..., mode="eval")` refuses anything left.
    if (peek().t != Tok::END) throw SyntaxError("invalid syntax at '" + peek().v + "'");
    return tree;
  }

 private:
  vector<Tok> toks_;
  size_t p_ = 0;
  const Tok& peek(size_t k = 0) const { return toks_[std::min(p_ + k, toks_.size() - 1)]; }
  bool isOp(const char* v, size_t k = 0) const { return peek(k).t == Tok::OP && peek(k).v == v; }
  bool isKw(const char* v, size_t k = 0) const { return peek(k).t == Tok::KW && peek(k).v == v; }
  void expectOp(const char* v) {
    if (!isOp(v)) throw SyntaxError(string("expected '") + v + "'");
    ++p_;
  }

  Node expression() {
    Node e = orTest();
    if (isKw("if")) throw Error("Unsupported node type: IfExp");
    return e;
  }
  Node boolOf(const char* kw, Node (Parser::*next)()) {
    vector<Node> values{(this->*next)()};
    while (isKw(kw)) { ++p_; values.push_back((this->*next)()); }
    if (values.size() == 1) return values[0];
    Node n{Node::BOOL};
    n.op = kw;
    n.kids = std::move(values);
    return n;
  }
  Node orTest() { return boolOf("or", &Parser::andTest); }
  Node andTest() { return boolOf("and", &Parser::notTest); }
  Node notTest() {
    if (isKw("not")) {
      ++p_;
      Node n{Node::UNARY};
      n.op = "not";
      n.kids.push_back(notTest());
      return n;
    }
    return comparison();
  }
  bool compOp(string* op) {
    const Tok& t = peek();
    if (t.t == Tok::OP && (t.v == "<" || t.v == ">" || t.v == "==" || t.v == ">=" || t.v == "<=" ||
                           t.v == "!=")) {
      *op = t.v;
      ++p_;
      return true;
    }
    if (isKw("in")) { ++p_; *op = "in"; return true; }
    if (isKw("not") && isKw("in", 1)) { p_ += 2; *op = "not in"; return true; }
    if (isKw("is")) {
      ++p_;
      if (isKw("not")) { ++p_; *op = "is not"; return true; }
      *op = "is";
      return true;
    }
    return false;
  }
  Node comparison() {
    Node left = bitOr();
    Node n{Node::COMPARE};
    string op;
    while (compOp(&op)) {
      n.ops.push_back(op);
      n.kids.push_back(bitOr());
    }
    if (n.ops.empty()) return left;
    n.kids.insert(n.kids.begin(), left);
    return n;
  }
  Node binary(Node (Parser::*next)(), std::initializer_list<const char*> ops) {
    Node left = (this->*next)();
    for (;;) {
      const char* hit = nullptr;
      for (const char* o : ops) if (isOp(o)) hit = o;
      if (!hit) return left;
      ++p_;
      Node n{Node::BINOP};
      n.op = hit;
      n.kids.push_back(left);
      n.kids.push_back((this->*next)());
      left = n;
    }
  }
  Node factor() {
    if (isOp("+") || isOp("-") || isOp("~")) {
      Node n{Node::UNARY};
      n.op = peek().v;
      ++p_;
      n.kids.push_back(factor());
      return n;
    }
    return power();
  }
  Node power() {
    Node base = primary();
    if (isOp("**")) {
      ++p_;
      Node n{Node::BINOP};
      n.op = "**";
      n.kids.push_back(base);
      n.kids.push_back(factor());
      return n;
    }
    return base;
  }
  Node term() { return binary(&Parser::factor, {"*", "/", "//", "%", "@"}); }
  Node arith() { return binary(&Parser::term, {"+", "-"}); }
  Node shift() { return binary(&Parser::arith, {"<<", ">>"}); }
  Node bitAnd() { return binary(&Parser::shift, {"&"}); }
  Node bitXor() { return binary(&Parser::bitAnd, {"^"}); }
  Node bitOr() { return binary(&Parser::bitXor, {"|"}); }

  Node primary() {
    Node node = atom();
    for (;;) {
      if (isOp("(")) {
        ++p_;
        Node call{Node::CALL};
        call.kids.push_back(node);   // the function, then the arguments
        while (!isOp(")")) {
          if (peek().t == Tok::NAME && isOp("=", 1)) call.keywords = true;
          if (isOp("*") || isOp("**")) throw Error("Unsupported node type: Starred");
          call.kids.push_back(expression());
          if (isOp(",")) ++p_;
          else break;
        }
        expectOp(")");
        node = call;
      } else if (isOp("[")) {
        throw Error("Unsupported node type: Subscript");
      } else if (isOp(".")) {
        throw Error("Unsupported node type: Attribute");
      } else {
        return node;
      }
    }
  }
  Node atom() {
    const Tok t = peek();
    if (t.t == Tok::NUM) { ++p_; Node n{Node::CONST}; n.v = Q::num(t.d); return n; }
    if (t.t == Tok::STR) {
      string v;
      while (peek().t == Tok::STR) { v += peek().v; ++p_; }
      Node n{Node::CONST};
      n.v = Q::str(v);
      return n;
    }
    if (t.t == Tok::KW && (t.v == "True" || t.v == "False" || t.v == "None")) {
      ++p_;
      Node n{Node::CONST};
      if (t.v != "None") n.v = Q::num(t.v == "True" ? 1 : 0);
      return n;
    }
    if (t.t == Tok::NAME) { ++p_; Node n{Node::NAME}; n.id = t.v; return n; }
    if (isOp("(")) {
      ++p_;
      if (isOp(")")) { ++p_; return Node{Node::TUPLE}; }
      Node first = expression();
      if (isOp(",")) {
        Node n{Node::TUPLE};
        n.kids.push_back(first);
        while (isOp(",")) {
          ++p_;
          if (isOp(")")) break;
          n.kids.push_back(expression());
        }
        expectOp(")");
        return n;
      }
      expectOp(")");
      return first;
    }
    if (isOp("[")) {
      ++p_;
      Node n{Node::LIST};
      while (!isOp("]")) {
        n.kids.push_back(expression());
        if (isOp(",")) ++p_;
        else break;
      }
      expectOp("]");
      return n;
    }
    if (t.t == Tok::KW && t.v == "lambda") throw Error("Unsupported node type: Lambda");
    throw SyntaxError(t.t == Tok::END ? "invalid syntax at end of query"
                                      : "invalid syntax at '" + t.v + "'");
  }
};

// ---------------------------------------------------------------- numpy's semantics

struct Val {
  enum K { SC, ARR, LIST } k = SC;
  Q sc;
  vector<Q> arr;
  vector<Val> list;
};
inline Val scv(Q q) { Val v; v.sc = std::move(q); return v; }
inline Val arrv(vector<Q> a) { Val v; v.k = Val::ARR; v.arr = std::move(a); return v; }
inline Q qb(bool b) { return Q::num(b ? 1 : 0); }

inline bool truthy(const Q& x) {
  switch (x.t) {
    case Q::NUL: return false;
    case Q::NUM: return x.d != 0;   // NaN is truthy, as `bool(nan)` is
    case Q::STR: return !x.s.empty();
    default: return true;
  }
}

inline bool cmp(const string& op, const Q& x, const Q& y) {
  if (op == "==" || op == "!=") {
    bool eq;
    if (x.t == Q::NUL || y.t == Q::NUL) eq = x.t == y.t;
    else if (x.t != y.t) eq = false;
    else if (x.t == Q::NUM) eq = x.d == y.d;
    else if (x.t == Q::STR) eq = x.s == y.s;
    else eq = false;
    return op == "==" ? eq : !eq;
  }
  if (x.t == Q::NUL || y.t == Q::NUL || x.t != y.t || x.t == Q::OBJ) {
    throw Error("'" + op + "' not supported between these operand types");
  }
  if (x.t == Q::NUM) {
    return op == "<" ? x.d < y.d : op == "<=" ? x.d <= y.d : op == ">" ? x.d > y.d : x.d >= y.d;
  }
  return op == "<" ? x.s < y.s : op == "<=" ? x.s <= y.s : op == ">" ? x.s > y.s : x.s >= y.s;
}

inline vector<Q> scalars(const Val& v) {
  if (v.k == Val::ARR) return v.arr;
  if (v.k == Val::LIST) {
    vector<Q> out;
    for (const Val& e : v.list) out.push_back(e.k == Val::SC ? e.sc : Q::obj());
    return out;
  }
  throw Error("Argument is not iterable");
}

inline Val elementwise(const Val& a, const Val& b, const std::function<Q(const Q&, const Q&)>& f) {
  if (a.k == Val::SC && b.k == Val::SC) return scv(f(a.sc, b.sc));
  if (a.k == Val::ARR && b.k == Val::ARR) {
    if (a.arr.size() != b.arr.size()) throw Error("operands could not be broadcast together");
    vector<Q> out(a.arr.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = f(a.arr[i], b.arr[i]);
    return arrv(std::move(out));
  }
  if (a.k == Val::ARR && b.k == Val::SC) {
    vector<Q> out(a.arr.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = f(a.arr[i], b.sc);
    return arrv(std::move(out));
  }
  if (a.k == Val::SC && b.k == Val::ARR) {
    vector<Q> out(b.arr.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = f(a.sc, b.arr[i]);
    return arrv(std::move(out));
  }
  // A Python list on one side: numpy makes it an array, so it broadcasts by position.
  return elementwise(a.k == Val::LIST ? arrv(scalars(a)) : a, b.k == Val::LIST ? arrv(scalars(b)) : b, f);
}

inline vector<uint8_t> ensureBool(const Val& v, size_t n) {
  vector<uint8_t> out;
  if (v.k == Val::SC) return vector<uint8_t>(n, truthy(v.sc));
  for (const Q& q : scalars(v)) out.push_back(truthy(q));
  if (out.size() != n) {
    throw Error("Query resulted in mask of length " + std::to_string(out.size()) +
                ", but AtomArray has length " + std::to_string(n));
  }
  return out;
}

/** `String(x)` of a number, as `np.isin` sees a list's numbers once a string has made it text. */
inline string jsString(const Q& q) {
  if (q.t == Q::STR) return q.s;
  if (q.t == Q::NUL) return "null";
  if (q.t == Q::OBJ) return "[object]";
  if (std::isnan(q.d)) return "NaN";
  if (std::isinf(q.d)) return q.d > 0 ? "Infinity" : "-Infinity";
  if (q.d == std::floor(q.d) && std::fabs(q.d) < 1e21) {
    char buf[32];
    snprintf(buf, sizeof buf, "%.0f", q.d == 0 ? 0.0 : q.d);
    return buf;
  }
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof buf, q.d);
  return string(buf, r.ptr);
}

// ---------------------------------------------------------------- the atoms

inline vector<int> residueStarts(const atoms::AtomArray& a) {
  vector<int> s;
  for (size_t i = 0; i < a.n(); ++i) {
    if (i == 0 || a.site.chainId[i] != a.site.chainId[i - 1] || a.site.resId[i] != a.site.resId[i - 1] ||
        a.site.insCode[i] != a.site.insCode[i - 1]) {
      s.push_back((int)i);
    }
  }
  return s;
}

inline vector<int> chainStarts(const atoms::AtomArray& a) {
  vector<int> s;
  for (size_t i = 0; i < a.n(); ++i) {
    if (i == 0 || a.site.chainId[i] != a.site.chainId[i - 1] || a.site.resId[i] < a.site.resId[i - 1]) {
      s.push_back((int)i);
    }
  }
  return s;
}

inline vector<int> tokenStartsNoStop(const atoms::AtomArray& a) {
  if (a.n() == 0) return {};
  vector<int> s = atoms::tokenStarts(a);
  s.pop_back();
  return s;
}

using Reduction = std::function<Q(const vector<Q>&)>;

inline double toNumber(const Q& q) {
  if (q.t == Q::NUM) return q.d;
  if (q.t == Q::NUL) return 0;
  if (q.t == Q::STR) {
    const string t = trim(q.s);
    if (t.empty()) return 0;
    char* end = nullptr;
    const double v = strtod(t.c_str(), &end);
    return end && *end == 0 ? v : NAN;
  }
  return NAN;
}

inline const Reduction* reduction(const string& name) {
  static const std::map<string, Reduction> R = {
      {"all", [](const vector<Q>& xs) { for (const Q& x : xs) if (!truthy(x)) return qb(false); return qb(true); }},
      {"any", [](const vector<Q>& xs) { for (const Q& x : xs) if (truthy(x)) return qb(true); return qb(false); }},
      {"max", [](const vector<Q>& xs) { double m = -INFINITY; for (const Q& x : xs) if (toNumber(x) > m) m = toNumber(x); return Q::num(m); }},
      {"min", [](const vector<Q>& xs) { double m = INFINITY; for (const Q& x : xs) if (toNumber(x) < m) m = toNumber(x); return Q::num(m); }},
      {"sum", [](const vector<Q>& xs) { double s = 0; for (const Q& x : xs) s += toNumber(x); return Q::num(s); }},
      {"mean", [](const vector<Q>& xs) { double s = 0; for (const Q& x : xs) s += toNumber(x); return Q::num(s / xs.size()); }},
  };
  auto it = R.find(name);
  return it == R.end() ? nullptr : &it->second;
}

inline vector<Q> spreadSegments(const vector<int>& starts, size_t n, const vector<Q>& values,
                                const Reduction& reduce) {
  vector<Q> out(n);
  for (size_t s = 0; s < starts.size(); ++s) {
    const size_t a = starts[s], b = s + 1 < starts.size() ? starts[s + 1] : n;
    const Q r = reduce(vector<Q>(values.begin() + a, values.begin() + b));
    for (size_t i = a; i < b; ++i) out[i] = r;
  }
  return out;
}

template <class T>
inline vector<Q> nums(const vector<T>& v) {
  vector<Q> out;
  out.reserve(v.size());
  for (const T& x : v) out.push_back(Q::num((double)x));
  return out;
}
inline vector<Q> strs(const vector<string>& v) {
  vector<Q> out;
  out.reserve(v.size());
  for (const string& x : v) out.push_back(Q::str(x));
  return out;
}

/** The annotations upstream derives on demand (`ANNOTATOR_REGISTRY`), each as atomworks does. */
inline std::optional<vector<Q>> derived(const atoms::AtomArray& a, const string& name) {
  using namespace tables;
  const size_t n = a.n();
  const vector<int>& ct = a.site.chainType;
  const vector<string>& rn = a.site.resName;
  const vector<string>& an = a.site.atomName;
  auto typeIn = [&](const std::unordered_set<int>& types) {
    vector<Q> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = qb(types.count(ct[i]));
    return out;
  };
  auto named = [&](std::initializer_list<const std::unordered_set<string>*> sets,
                   std::initializer_list<const string*> extra, int maskedType) {
    vector<Q> out(n);
    for (size_t i = 0; i < n; ++i) {
      bool hit = false;
      for (auto* s : sets) hit = hit || s->count(rn[i]);
      for (auto* e : extra) hit = hit || rn[i] == *e;
      if (maskedType >= 0) hit = hit || (rn[i] == Q_MASKED && ct[i] == maskedType);
      out[i] = qb(hit);
    }
    return out;
  };
  auto backbone = [&](const char* polymer, const std::unordered_set<string>& bb, bool inside) {
    const vector<Q> p = *derived(a, polymer);
    vector<Q> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = qb(truthy(p[i]) && (bb.count(an[i]) > 0) == inside);
    return out;
  };
  auto startsMask = [&](const vector<int>& starts) {
    vector<Q> out(n, qb(false));
    for (const int s : starts) out[s] = qb(true);
    return out;
  };
  if (name == "is_protein") return typeIn(Q_PROTEIN_CHAIN_TYPES);
  if (name == "is_nucleic_acid") return typeIn(Q_NUCLEIC_CHAIN_TYPES);
  if (name == "is_dna") return typeIn({Q_CHAIN_DNA});
  if (name == "is_rna") return typeIn({Q_CHAIN_RNA});
  if (name == "is_polymer") return typeIn(Q_POLYMER_CHAIN_TYPES);
  if (name == "is_ligand") return typeIn(Q_NON_POLYMER_CHAIN_TYPES);
  if (name == "is_standard_aa") return named({&Q_STANDARD_AA}, {}, -1);
  if (name == "is_standard_or_unknown_aa") return named({&Q_STANDARD_AA}, {&Q_UNKNOWN_AA}, -1);
  if (name == "is_standard_unknown_or_masked_aa") return named({&Q_STANDARD_AA}, {&Q_UNKNOWN_AA}, Q_CHAIN_POLYPEPTIDE_L);
  if (name == "is_standard_rna") return named({&Q_STANDARD_RNA}, {}, -1);
  if (name == "is_standard_dna") return named({&Q_STANDARD_DNA}, {}, -1);
  if (name == "is_standard_or_unknown_rna") return named({&Q_STANDARD_RNA}, {&Q_UNKNOWN_RNA}, -1);
  if (name == "is_standard_or_unknown_dna") return named({&Q_STANDARD_DNA}, {&Q_UNKNOWN_DNA}, -1);
  if (name == "is_standard_unknown_or_masked_rna") return named({&Q_STANDARD_RNA}, {&Q_UNKNOWN_RNA}, Q_CHAIN_RNA);
  if (name == "is_standard_unknown_or_masked_dna") return named({&Q_STANDARD_DNA}, {&Q_UNKNOWN_DNA}, Q_CHAIN_DNA);
  if (name == "is_protein_backbone") return backbone("is_protein", Q_PROTEIN_BACKBONE_ATOM_NAMES, true);
  if (name == "is_protein_sidechain") return backbone("is_protein", Q_PROTEIN_BACKBONE_ATOM_NAMES, false);
  if (name == "is_rna_backbone") return backbone("is_rna", Q_RNA_BACKBONE_ATOM_NAMES, true);
  if (name == "is_dna_backbone") return backbone("is_dna", Q_DNA_BACKBONE_ATOM_NAMES, true);
  if (name == "is_nucleic_acid_backbone") return backbone("is_nucleic_acid", Q_NUCLEIC_ACID_BACKBONE_ATOM_NAMES, true);
  if (name == "is_pyrimidine") return named({&Q_PYRIMIDINE_RESIDUES}, {}, -1);
  if (name == "is_purine") return named({&Q_PURINE_RESIDUES}, {}, -1);
  if (name == "is_metal") return named({&Q_METAL_ELEMENTS}, {}, -1);
  if (name == "is_carbohydrate") return named({&Q_CARBOHYDRATE_NAMES}, {}, -1);
  if (name == "is_tip_atom") {
    vector<Q> out(n);
    for (size_t i = 0; i < n; ++i) {
      auto it = Q_TIP_ATOM_NAMES.find(rn[i]);
      out[i] = qb(it != Q_TIP_ATOM_NAMES.end() && it->second.count(an[i]));
    }
    return out;
  }
  if (name == "is_res_start") return startsMask(residueStarts(a));
  if (name == "is_chain_start") return startsMask(chainStarts(a));
  if (name == "is_token_start") return startsMask(tokenStartsNoStop(a));
  if (name == "token_id") {
    const vector<int> starts = tokenStartsNoStop(a);
    vector<Q> out(n);
    for (size_t t = 0; t < starts.size(); ++t) {
      const size_t b = t + 1 < starts.size() ? starts[t + 1] : n;
      for (size_t i = starts[t]; i < b; ++i) out[i] = Q::num((double)t);
    }
    return out;
  }
  if (name == "res_min_occupancy") return spreadSegments(residueStarts(a), n, nums(a.site.occupancy), *reduction("min"));
  if (name == "token_min_occupancy") return spreadSegments(tokenStartsNoStop(a), n, nums(a.site.occupancy), *reduction("min"));
  if (name == "res_has_tip_atom") return spreadSegments(residueStarts(a), n, *derived(a, "is_tip_atom"), *reduction("any"));
  if (name == "within_chain_res_idx") {
    // `get_within_group_res_idx`: a residue's ordinal within its chain (by `chain_iid` where
    // there is one), counted over residue starts.
    const vector<string>& group = a.ids.chainIid.size() == n ? a.ids.chainIid : a.site.chainId;
    vector<Q> out(n);
    std::map<string, int> seen;
    const vector<int> starts = residueStarts(a);
    for (size_t s = 0; s < starts.size(); ++s) {
      const size_t b = s + 1 < starts.size() ? starts[s + 1] : n;
      const int k = seen[group[starts[s]]]++;
      for (size_t i = starts[s]; i < b; ++i) out[i] = Q::num(k);
    }
    return out;
  }
  return std::nullopt;
}

/**
 * A name in the query's namespace: the atom array's own columns as the page's reader names them
 * (cif.js), then its annotations, then the derived ones.
 *
 * 🔴 THE NAMESPACE IS UPSTREAM'S: atomworks parses no aromatic or leaving-atom flag onto an atom,
 * names the alternate location `altloc_id` as well, and has no `b_factor` where the file had no
 * B column - each a NameError there, and one here.
 */
inline std::optional<vector<Q>> attribute(const atoms::AtomArray& a, const string& name) {
  const size_t n = a.n();
  const cif::AtomSite& s = a.site;
  if (name == "x") return nums(s.x);
  if (name == "y") return nums(s.y);
  if (name == "z") return nums(s.z);
  if (name == "b_factor") return s.noBFactorColumn ? std::nullopt : std::optional<vector<Q>>(nums(s.bFactor));
  if (name == "altloc_id" || name == "label_alt_id") return strs(s.labelAltId);
  if (name == "transformation_id") return vector<Q>(n, Q::str("1"));
  static const std::unordered_map<string, vector<string> cif::AtomSite::*> STR = {
      {"element", &cif::AtomSite::element}, {"atom_name", &cif::AtomSite::atomName},
      {"res_name", &cif::AtomSite::resName}, {"chain_id", &cif::AtomSite::chainId},
      {"ins_code", &cif::AtomSite::insCode}, {"chem_comp_type", &cif::AtomSite::chemCompType}};
  static const std::unordered_map<string, vector<int> cif::AtomSite::*> INT = {
      {"label_entity_id", &cif::AtomSite::labelEntityId}, {"res_id", &cif::AtomSite::resId},
      {"auth_seq_id", &cif::AtomSite::authSeqId}, {"charge", &cif::AtomSite::charge},
      {"chain_type", &cif::AtomSite::chainType}, {"atomic_number", &cif::AtomSite::atomicNumber}};
  if (auto it = STR.find(name); it != STR.end() && (s.*(it->second)).size() == n) return strs(s.*(it->second));
  if (auto it = INT.find(name); it != INT.end() && (s.*(it->second)).size() == n) return nums(s.*(it->second));
  if (name == "occupancy") return nums(s.occupancy);
  if (name == "hetero") return nums(s.hetero);
  if (name == "is_polymer" && s.isPolymer.size() == n) return nums(s.isPolymer);
  // `addGlobalAtomIdAnnotation` replaces the file's ids with 0..n-1 once it has run.
  if (name == "atom_id") return a.atomId.size() == n ? nums(a.atomId) : nums(s.atomId);
  static const std::unordered_map<string, vector<string> cif::Ids::*> ISTR = {
      {"pn_unit_id", &cif::Ids::pnUnitId}, {"chain_iid", &cif::Ids::chainIid},
      {"pn_unit_iid", &cif::Ids::pnUnitIid}};
  static const std::unordered_map<string, vector<int> cif::Ids::*> IINT = {
      {"molecule_id", &cif::Ids::moleculeId}, {"chain_entity", &cif::Ids::chainEntity},
      {"pn_unit_entity", &cif::Ids::pnUnitEntity}, {"molecule_entity", &cif::Ids::moleculeEntity},
      {"molecule_iid", &cif::Ids::moleculeIid}};
  if (auto it = ISTR.find(name); it != ISTR.end() && (a.ids.*(it->second)).size() == n) return strs(a.ids.*(it->second));
  if (auto it = IINT.find(name); it != IINT.end() && (a.ids.*(it->second)).size() == n) return nums(a.ids.*(it->second));
  // The columns the pipeline's own stages attach, once they have.
  static const std::unordered_map<string, vector<uint8_t> atoms::AtomArray::*> FLAGS = {
      {"is_protein", &atoms::AtomArray::isProtein}, {"is_rna", &atoms::AtomArray::isRna},
      {"is_dna", &atoms::AtomArray::isDna}, {"is_nucleic_acid", &atoms::AtomArray::isNucleicAcid},
      {"is_protein_backbone", &atoms::AtomArray::isProteinBackbone},
      {"is_rna_backbone", &atoms::AtomArray::isRnaBackbone},
      {"is_dna_backbone", &atoms::AtomArray::isDnaBackbone},
      {"is_nucleic_acid_backbone", &atoms::AtomArray::isNucleicAcidBackbone},
      {"is_standard_aa", &atoms::AtomArray::isStandardAa},
      {"is_standard_or_unknown_aa", &atoms::AtomArray::isStandardOrUnknownAa},
      {"is_standard_unknown_or_masked_aa", &atoms::AtomArray::isStandardUnknownOrMaskedAa},
      {"is_standard_rna", &atoms::AtomArray::isStandardRna},
      {"is_standard_or_unknown_rna", &atoms::AtomArray::isStandardOrUnknownRna},
      {"is_standard_unknown_or_masked_rna", &atoms::AtomArray::isStandardUnknownOrMaskedRna},
      {"is_standard_dna", &atoms::AtomArray::isStandardDna},
      {"is_standard_or_unknown_dna", &atoms::AtomArray::isStandardOrUnknownDna},
      {"is_standard_unknown_or_masked_dna", &atoms::AtomArray::isStandardUnknownOrMaskedDna},
      {"atomize", &atoms::AtomArray::atomize},
      {"is_covalent_modification", &atoms::AtomArray::isCovalentModification},
      {"is_unindexed_source", &atoms::AtomArray::isUnindexedSource},
      {"supports_local_latent", &atoms::AtomArray::supportsLocalLatent}};
  static const std::unordered_map<string, vector<int> atoms::AtomArray::*> STAGE = {
      {"token_id", &atoms::AtomArray::tokenId},
      {"within_chain_source_res_idx", &atoms::AtomArray::withinChainSourceResIdx},
      {"insertion_source_idx", &atoms::AtomArray::insertionSourceIdx}};
  if (auto it = FLAGS.find(name); it != FLAGS.end() && (a.*(it->second)).size() == n && n) return nums(a.*(it->second));
  if (auto it = STAGE.find(name); it != STAGE.end() && (a.*(it->second)).size() == n && n) return nums(a.*(it->second));
  if (auto it = a.ann.ints.find(name); it != a.ann.ints.end() && it->second.size() == n) return nums(it->second);
  if (auto it = a.ann.strings.find(name); it != a.ann.strings.end() && it->second.size() == n) return strs(it->second);
  if (auto it = a.ann.floats.find(name); it != a.ann.floats.end()) {
    if (it->second.size() == n) return nums(it->second);
    if (n && it->second.size() == n * 3) return vector<Q>(n, Q::obj());   // a coordinate triple
  }
  return derived(a, name);
}

// ---------------------------------------------------------------- `sa_mask(...)`

struct Resolved {
  string fullName, dtype;
  Q def;
};

/** `resolveConditionSpec`: the registry, then the open-ended sequence-probability names. */
inline Resolved resolve(const string& name) {
  auto it = tables::CONDITION_REGISTRY.find(name);
  if (it != tables::CONDITION_REGISTRY.end()) {
    const auto& c = it->second;
    Resolved r{c.fullName, c.dtype, Q::obj()};
    if (c.defaultIsNum) r.def = Q::num(c.defaultNum);
    else if (c.defaultIsStr) r.def = Q::str(c.defaultStr);
    return r;
  }
  for (const char* prefix : {"C_SEQ_PROB_", "sequence-probability-"}) {
    if (name.rfind(prefix, 0) == 0) {
      return {"condition_sequence-probability-" + name.substr(strlen(prefix)) + "_1_residue", "float",
              Q::num(NAN)};
    }
  }
  if (name.rfind("condition_sequence-probability-", 0) == 0) return {name, "float", Q::num(NAN)};
  throw Error("Unrecognized Condition/StandardAnnotation name: " + name);
}

/** Each condition's presence mask, as its own `mask_from_annotation` defines it upstream. */
inline vector<Q> saMask(const atoms::AtomArray& a, const string& name) {
  const Resolved spec = resolve(name);
  if (spec.dtype == "twobody") {
    throw Error("sa_mask('" + name + "') requires a one-body standard annotation, but `" +
                spec.fullName + "` is 2-body and has no per-atom mask.");
  }
  const size_t n = a.n();
  const string& f = spec.fullName;
  vector<Q> out(n);
  if (f == "condition_coordinate_1_atom" || f == "condition_InternalCoordinate_1_atom" ||
      f == "condition_ChainCentroid_1_atom") {
    auto it = a.ann.floats.find(f);
    const bool have = it != a.ann.floats.end() && it->second.size() == n * 3;
    for (size_t i = 0; i < n; ++i) {
      out[i] = qb(have && (std::isfinite(it->second[i * 3]) || std::isfinite(it->second[i * 3 + 1]) ||
                           std::isfinite(it->second[i * 3 + 2])));
    }
    return out;
  }
  std::optional<vector<Q>> values;
  if (auto it = a.ann.ints.find(f); it != a.ann.ints.end() && it->second.size() == n) values = nums(it->second);
  else if (auto it = a.ann.strings.find(f); it != a.ann.strings.end() && it->second.size() == n) values = strs(it->second);
  else if (auto it = a.ann.floats.find(f); it != a.ann.floats.end() && it->second.size() == n) values = nums(it->second);
  Q def = spec.def;
  std::function<bool(const Q&)> mask;
  auto differs = [](const Q& d) { return [d](const Q& v) { return !cmp("==", v, d); }; };
  if (f == "condition_CoordinateGroup_1_atom" || f == "condition_index-group_1_token") {
    mask = [](const Q&) { return true; };
  } else if (f == "condition_sequence_1_residue") {
    def = Q::str(tables::Q_MASKED);
    mask = differs(def);
  } else if (f == "condition_cath-label_1_residue") {
    def = Q::str("x.x.x.x");
    mask = differs(def);
  } else if (f == "annotation_src-component_1_atom") {
    def = Q::str(".");
    mask = differs(def);
  } else if (f == "annotation_expsegmin_1_residue" || f == "annotation_expsegmax_1_residue") {
    def = Q::num(-1);
    mask = differs(def);
  } else if (f.rfind("condition_sequence-probability-", 0) == 0) {
    def = Q::num(NAN);
    mask = [](const Q& v) { return !(v.t == Q::NUM && std::isnan(v.d)); };
  } else if (spec.dtype == "bool") {
    mask = [](const Q& v) { return truthy(v); };
  } else {
    mask = differs(def);
  }
  for (size_t i = 0; i < n; ++i) out[i] = qb(mask(values ? (*values)[i] : def));
  return out;
}

// ---------------------------------------------------------------- `sel('...')`

struct Field {
  bool star = false;
  string s;
  long long id = 0;
};

inline Field scalarField(const string& value, bool isResId) {
  Field f;
  if (!isResId) { f.s = value; return f; }
  // Python's `int()`: optional sign, digits (with underscores between them), surrounding space.
  string t = trim(value), digits;
  size_t k = 0;
  if (k < t.size() && (t[k] == '+' || t[k] == '-')) digits += t[k++];
  bool ok = k < t.size() && isdigit((unsigned char)t[k]);
  for (; ok && k < t.size(); ++k) {
    if (isdigit((unsigned char)t[k])) digits += t[k];
    else if (t[k] == '_' && k + 1 < t.size() && isdigit((unsigned char)t[k + 1]) && isdigit((unsigned char)t[k - 1])) continue;
    else ok = false;
  }
  if (!ok) {
    throw Error("Invalid res_id '" + value + "': ranges must be wrapped in brackets, e.g. '[5-10]'.");
  }
  f.id = strtoll(digits.c_str(), nullptr, 10);
  return f;
}

inline vector<Field> parseField(const string& raw, bool isResId) {
  const string v = trim(raw);
  if (v == "*" || v.empty()) return {Field{true}};
  if (v.find('[') != string::npos || v.find(']') != string::npos) {
    if (!(v.front() == '[' && v.back() == ']')) throw Error("Malformed bracket list in selection field: '" + v + "'");
    vector<string> items;
    {
      string buf;
      for (const char c : v.substr(1, v.size() - 2)) {
        if (c == ',') { items.push_back(trim(buf)); buf.clear(); } else { buf += c; }
      }
      items.push_back(trim(buf));
    }
    items.erase(std::remove(items.begin(), items.end(), string()), items.end());
    if (items.empty()) throw Error("Empty bracket list in selection field: '" + v + "'");
    vector<Field> out;
    for (const string& item : items) {
      if (item == "*") { out.push_back(Field{true}); continue; }
      if (isResId) {
        // `^(-?\d+)-(-?\d+)$`: a range, which may count down.
        size_t at = item[0] == '-' ? 1 : 0;
        const size_t lo0 = at;
        while (at < item.size() && isdigit((unsigned char)item[at])) ++at;
        if (at > lo0 && at < item.size() && item[at] == '-') {
          size_t hi = at + 1;
          if (hi < item.size() && item[hi] == '-') ++hi;
          const size_t hi0 = hi;
          while (hi < item.size() && isdigit((unsigned char)item[hi])) ++hi;
          if (hi > hi0 && hi == item.size()) {
            const long long s = strtoll(item.substr(0, at).c_str(), nullptr, 10);
            const long long e = strtoll(item.substr(at + 1).c_str(), nullptr, 10);
            const long long step = s <= e ? 1 : -1;
            for (long long k = s; step > 0 ? k <= e : k >= e; k += step) {
              Field f;
              f.id = k;
              out.push_back(f);
            }
            continue;
          }
        }
      }
      out.push_back(scalarField(item, isResId));
    }
    return out;
  }
  return {scalarField(v, isResId)};
}

/** `AtomSelectionStack.from_query(...).get_mask(raise_on_empty=False)`. */
inline vector<Q> pathMask(const atoms::AtomArray& a, const string& query) {
  const size_t n = a.n();
  vector<uint8_t> out(n, 0);
  vector<string> tokens;
  {
    string buf;
    int depth = 0;
    for (const char ch : query) {
      if (ch == '[') ++depth;
      else if (ch == ']') --depth;
      if (ch == ',' && depth == 0) { tokens.push_back(buf); buf.clear(); } else { buf += ch; }
    }
    tokens.push_back(buf);
  }
  for (string token : tokens) {
    token = trim(token);
    if (token.empty()) continue;
    vector<string> parts;
    {
      size_t from = 0;
      while (true) {
        const size_t slash = token.find('/', from);
        parts.push_back(token.substr(from, slash == string::npos ? string::npos : slash - from));
        if (slash == string::npos) break;
        from = slash + 1;
      }
    }
    while (parts.size() < 5) parts.push_back("*");
    const vector<Field> chains = parseField(parts[0], false), resNames = parseField(parts[1], false),
                        resIds = parseField(parts[2], true), atomNames = parseField(parts[3], false),
                        trans = parseField(parts[4], false);
    auto any = [](const vector<Field>& fs, auto match) {
      for (const Field& f : fs) if (f.star || match(f)) return true;
      return false;
    };
    for (size_t i = 0; i < n; ++i) {
      if (out[i]) continue;
      if (!any(chains, [&](const Field& f) { return a.site.chainId[i] == f.s; })) continue;
      if (!any(resNames, [&](const Field& f) { return a.site.resName[i] == f.s; })) continue;
      if (!any(resIds, [&](const Field& f) { return a.site.resId[i] == f.id; })) continue;
      if (!any(atomNames, [&](const Field& f) { return a.site.atomName[i] == f.s; })) continue;
      // `transformation_id` is "1" throughout: this port expands no assembly.
      if (!any(trans, [&](const Field& f) { return f.s == "1"; })) continue;
      out[i] = 1;
    }
  }
  vector<Q> mask(n);
  for (size_t i = 0; i < n; ++i) mask[i] = qb(out[i]);
  return mask;
}

// ---------------------------------------------------------------- the evaluator

inline Val evaluate(const Node& node, const atoms::AtomArray& a) {
  const size_t n = a.n();
  auto ev = [&](const Node& x) { return evaluate(x, a); };
  switch (node.k) {
    case Node::CONST: return scv(node.v);
    case Node::LIST:
    case Node::TUPLE: {
      Val v;
      v.k = Val::LIST;
      for (const Node& e : node.kids) v.list.push_back(ev(e));
      return v;
    }
    case Node::NAME: {
      auto v = attribute(a, node.id);
      if (!v) throw Error("Name '" + node.id + "' is not defined");
      return arrv(std::move(*v));
    }
    case Node::COMPARE: {
      Val left = ev(node.kids[0]);
      std::optional<Val> result;
      for (size_t j = 0; j < node.ops.size(); ++j) {
        const string& op = node.ops[j];
        Val right = ev(node.kids[j + 1]);
        Val r;
        if (op == "in" || op == "not in") {
          if (right.k == Val::SC) throw Error("Argument is not iterable");
          const bool invert = op == "not in";
          vector<Q> pool = scalars(right);
          auto member = [](const Q& x, const vector<Q>& list) {
            for (const Q& y : list) if (cmp("==", x, y)) return true;
            return false;
          };
          if (left.k == Val::ARR) {
            // `np.isin` makes the pool ONE array first, and numpy types a list holding any
            // string as strings - so `res_id in [5, 6, '7']` matches nothing.
            bool anyStr = false;
            for (const Q& y : pool) anyStr = anyStr || y.t == Q::STR;
            if (anyStr) for (Q& y : pool) y = Q::str(jsString(y));
            vector<Q> out(left.arr.size());
            for (size_t i = 0; i < out.size(); ++i) out[i] = qb(member(left.arr[i], pool) != invert);
            r = arrv(std::move(out));
          } else {
            // A scalar on the left is Python's own `in`, which does not coerce.
            r = scv(qb(member(left.k == Val::SC ? left.sc : Q::obj(), pool) != invert));
          }
        } else if (op == "is" || op == "is not") {
          throw Error("Unsupported comparison operator: " + op);
        } else {
          r = elementwise(left, right, [&](const Q& x, const Q& y) { return qb(cmp(op, x, y)); });
        }
        result = result ? elementwise(*result, r, [](const Q& x, const Q& y) { return qb(truthy(x) && truthy(y)); })
                        : r;
        left = right;
      }
      return *result;
    }
    case Node::BOOL: {
      vector<uint8_t> acc = ensureBool(ev(node.kids[0]), n);
      for (size_t k = 1; k < node.kids.size(); ++k) {
        const vector<uint8_t> v = ensureBool(ev(node.kids[k]), n);
        for (size_t i = 0; i < n; ++i) acc[i] = node.op == "and" ? (acc[i] && v[i]) : (acc[i] || v[i]);
      }
      return arrv(nums(acc));
    }
    case Node::BINOP: {
      if (node.op != "&" && node.op != "|") throw Error("Unsupported binary operation: " + node.op);
      const vector<uint8_t> x = ensureBool(ev(node.kids[0]), n), y = ensureBool(ev(node.kids[1]), n);
      vector<Q> out(n);
      for (size_t i = 0; i < n; ++i) out[i] = qb(node.op == "&" ? (x[i] && y[i]) : (x[i] || y[i]));
      return arrv(std::move(out));
    }
    case Node::UNARY: {
      const Val v = ev(node.kids[0]);
      if (node.op == "not" || node.op == "~") {
        vector<uint8_t> b = ensureBool(v, n);
        for (auto& x : b) x = !x;
        return arrv(nums(b));
      }
      auto check = [&](const Q& x) {
        if (x.t == Q::STR) throw Error("bad operand type for unary " + node.op + ": 'str'");
        if (node.op == "+") return x;
        return Q::num(-toNumber(x));
      };
      if (v.k == Val::SC) return scv(check(v.sc));
      vector<Q> out = scalars(v);
      for (Q& x : out) x = check(x);
      return arrv(std::move(out));
    }
    case Node::CALL: {
      const Node& func = node.kids[0];
      const vector<Node> args(node.kids.begin() + 1, node.kids.end());
      if (func.k != Node::NAME) throw Error("Complex function calls not supported");
      const string& fn = func.id;
      if (fn != "apply_and_spread" && fn != "has_nan_coord" && fn != "has_bonds" && fn != "sa_mask" &&
          fn != "sel") {
        throw Error("Function '" + fn + "' is not defined");
      }
      if (fn == "apply_and_spread") {
        if (args.size() != 3 || node.keywords) {
          throw Error("Function 'apply_and_spread' expects level, data, and reduction arguments");
        }
        const Val level = ev(args[0]), data = ev(args[1]), red = ev(args[2]);
        if (level.k != Val::SC || level.sc.t != Q::STR) throw Error("apply_and_spread() level must be a string");
        if (red.k != Val::SC || red.sc.t != Q::STR) throw Error("apply_and_spread() reduction must be a string");
        const Reduction* reduce = reduction(red.sc.s);
        if (!reduce) throw Error("Unknown apply_and_spread() reduction '" + red.sc.s + "'");
        if (data.k == Val::SC || scalars(data).size() != n) {
          throw Error("apply_and_spread() data must have one value per atom");
        }
        const vector<Q> values = scalars(data);
        const string& lv = level.sc.s;
        if (lv == "atom") throw Error("Apply and spread does not make sense for atom level.");
        if (lv == "residue") return arrv(spreadSegments(residueStarts(a), n, values, *reduce));
        if (lv == "token") return arrv(spreadSegments(tokenStartsNoStop(a), n, values, *reduce));
        if (lv == "chain" || lv == "molecule" || lv == "system") {
          vector<string> key(n, "1");
          if (lv == "chain") key = a.ids.chainIid.size() == n ? a.ids.chainIid : a.site.chainId;
          if (lv == "molecule") {
            if (a.ids.moleculeId.size() != n) throw Error("no molecule_id annotation");
            for (size_t i = 0; i < n; ++i) key[i] = std::to_string(a.ids.moleculeId[i]);
          }
          std::map<string, vector<size_t>> groups;
          for (size_t i = 0; i < n; ++i) groups[key[i]].push_back(i);
          vector<Q> spread(n);
          for (const auto& [g, idx] : groups) {
            vector<Q> xs;
            for (const size_t i : idx) xs.push_back(values[i]);
            const Q r = (*reduce)(xs);
            for (const size_t i : idx) spread[i] = r;
          }
          return arrv(std::move(spread));
        }
        throw Error("'" + lv + "' is not a valid Level");
      }
      if (fn == "sel" || fn == "sa_mask") {
        if (args.size() != 1 || node.keywords) throw Error("Function '" + fn + "' expects a single argument");
        const Val arg = ev(args[0]);
        if (arg.k != Val::SC || arg.sc.t != Q::STR) throw Error("Function '" + fn + "' argument must evaluate to a string");
        return arrv(fn == "sel" ? pathMask(a, arg.sc.s) : saMask(a, arg.sc.s));
      }
      if (!args.empty() || node.keywords) throw Error("Function '" + fn + "' does not accept arguments");
      vector<Q> out(n, qb(false));
      if (fn == "has_nan_coord") {
        for (size_t i = 0; i < n; ++i) {
          out[i] = qb(std::isnan(a.site.x[i]) || std::isnan(a.site.y[i]) || std::isnan(a.site.z[i]));
        }
      } else {
        for (const cif::Bond& b : a.bonds) { out[b.u] = qb(true); out[b.v] = qb(true); }
      }
      return arrv(std::move(out));
    }
  }
  throw Error("Unsupported node type");
}

}  // namespace query

/**
 * `resolve_mask(atom_array, query)`: a condition's `select:` as the mask it names, or a refusal
 * naming the query - never a silent empty mask, because a condition the model is told is absent
 * is a different job from the one that was asked for.
 */
inline vector<uint8_t> evalSelection(const atoms::AtomArray& a, const string& expr) {
  try {
    const query::Node tree = query::Parser(expr).parse();
    return query::ensureBool(query::evaluate(tree, a), a.n());
  } catch (const query::SyntaxError& e) {
    fprintf(stderr, "invalid selection '%s': %s\n", expr.c_str(), e.what());
  } catch (const std::exception& e) {
    fprintf(stderr, "selection '%s': %s\n", expr.c_str(), e.what());
  }
  exit(1);
}

}  // namespace selection
}  // namespace rfd4p
