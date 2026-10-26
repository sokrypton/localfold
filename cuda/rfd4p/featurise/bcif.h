// BinaryCIF, read into the same block `cif::parseBlock` returns for an mmCIF - a port of
// shared/rfd4p/pipeline/bcif.js, which says why it exists and where the format is specified.
// Every value comes back as TEXT (`.` and `?` where the mask says so), a number as the shortest
// text that reads back as the same value, so everything after the reader is the text path's.
#pragma once
#include "cif.h"
#include "../../featurise/zstd.h"
#include <charconv>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rfd4p {
namespace bcif {

using std::string;
using std::vector;

/** A MessagePack value: the types BinaryCIF uses. */
struct Msg {
  enum K { Nil, Bool, Int, Float, Str, Bin, Arr, Map } k = Nil;
  bool b = false;
  long long i = 0;
  double f = 0;
  string s;                                  // Str and Bin
  vector<Msg> a;
  vector<std::pair<string, Msg>> m;
  const Msg* get(const string& key) const {
    for (const auto& [k2, v] : m) if (k2 == key) return &v;
    return nullptr;
  }
  double num() const { return k == Float ? f : (double)i; }
};

class Unpacker {
 public:
  Unpacker(const string& bytes) : d_((const uint8_t*)bytes.data()), n_(bytes.size()) {}
  Msg value() {
    const uint8_t t = byte();
    Msg v;
    if (t < 0x80) { v.k = Msg::Int; v.i = t; return v; }
    if (t < 0x90) return map(t & 0x0f);
    if (t < 0xa0) return arr(t & 0x0f);
    if (t < 0xc0) return str(t & 0x1f);
    if (t >= 0xe0) { v.k = Msg::Int; v.i = (int8_t)t; return v; }
    switch (t) {
      case 0xc0: return v;
      case 0xc2: v.k = Msg::Bool; v.b = false; return v;
      case 0xc3: v.k = Msg::Bool; v.b = true; return v;
      case 0xc4: return bin(be(1));
      case 0xc5: return bin(be(2));
      case 0xc6: return bin(be(4));
      case 0xca: { uint32_t u = (uint32_t)be(4); float x; std::memcpy(&x, &u, 4); v.k = Msg::Float; v.f = x; return v; }
      case 0xcb: { uint64_t u = be(8); double x; std::memcpy(&x, &u, 8); v.k = Msg::Float; v.f = x; return v; }
      case 0xcc: v.k = Msg::Int; v.i = (long long)be(1); return v;
      case 0xcd: v.k = Msg::Int; v.i = (long long)be(2); return v;
      case 0xce: v.k = Msg::Int; v.i = (long long)be(4); return v;
      case 0xcf: v.k = Msg::Int; v.i = (long long)be(8); return v;
      case 0xd0: v.k = Msg::Int; v.i = (int8_t)be(1); return v;
      case 0xd1: v.k = Msg::Int; v.i = (int16_t)be(2); return v;
      case 0xd2: v.k = Msg::Int; v.i = (int32_t)be(4); return v;
      case 0xd3: v.k = Msg::Int; v.i = (int64_t)be(8); return v;
      case 0xd9: return str(be(1));
      case 0xda: return str(be(2));
      case 0xdb: return str(be(4));
      case 0xdc: return arr(be(2));
      case 0xdd: return arr(be(4));
      case 0xde: return map(be(2));
      case 0xdf: return map(be(4));
      default: throw std::runtime_error("BinaryCIF: unsupported MessagePack type");
    }
  }

 private:
  const uint8_t* d_;
  size_t n_, p_ = 0;
  uint8_t byte() {
    if (p_ >= n_) throw std::runtime_error("BinaryCIF: truncated");
    return d_[p_++];
  }
  uint64_t be(int count) {
    uint64_t v = 0;
    for (int c = 0; c < count; ++c) v = (v << 8) | byte();
    return v;
  }
  Msg str(uint64_t len) {
    if (p_ + len > n_) throw std::runtime_error("BinaryCIF: truncated");
    Msg v; v.k = Msg::Str; v.s.assign((const char*)d_ + p_, len); p_ += len; return v;
  }
  Msg bin(uint64_t len) { Msg v = str(len); v.k = Msg::Bin; return v; }
  Msg arr(uint64_t len) { Msg v; v.k = Msg::Arr; for (uint64_t i = 0; i < len; ++i) v.a.push_back(value()); return v; }
  Msg map(uint64_t len) {
    Msg v; v.k = Msg::Map;
    for (uint64_t i = 0; i < len; ++i) {
      Msg key = value();
      Msg val = value();
      v.m.push_back({key.s, std::move(val)});
    }
    return v;
  }
};

/** Whether these bytes are BinaryCIF: a MessagePack map, where an mmCIF starts with text. */
inline bool isBinaryCif(const string& bytes) {
  if (bytes.empty()) return false;
  const uint8_t t = (uint8_t)bytes[0];
  return (t >= 0x80 && t <= 0x8f) || t == 0xde || t == 0xdf;
}

/** A decoded column: numbers or strings. */
struct Column {
  bool strings = false;
  vector<double> num;
  vector<string> str;
};

inline Column byteArray(const string& data, int type) {
  Column c;
  auto get = [&](size_t at, size_t size) { uint64_t u = 0; for (size_t k = 0; k < size; ++k) u |= (uint64_t)(uint8_t)data[at + k] << (8 * k); return u; };
  size_t size = 0;
  switch (type) { case 1: case 4: size = 1; break; case 2: case 5: size = 2; break; case 3: case 6: case 32: size = 4; break; case 33: size = 8; break;
    default: throw std::runtime_error("BinaryCIF: unknown ByteArray type"); }
  const size_t n = data.size() / size;
  c.num.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const uint64_t u = get(i * size, size);
    switch (type) {
      case 1: c.num[i] = (int8_t)u; break;
      case 2: c.num[i] = (int16_t)u; break;
      case 3: c.num[i] = (int32_t)u; break;
      case 4: c.num[i] = (uint8_t)u; break;
      case 5: c.num[i] = (uint16_t)u; break;
      case 6: c.num[i] = (uint32_t)u; break;
      case 32: { uint32_t w = (uint32_t)u; float x; std::memcpy(&x, &w, 4); c.num[i] = x; break; }
      default: { double x; std::memcpy(&x, &u, 8); c.num[i] = x; }
    }
  }
  return c;
}

/** Undo a column's encodings, last applied first (bcif.js `decode`). */
inline Column decode(const Msg& data, const Msg& encodings) {
  Column v;
  bool raw = true;   // `data` still the packed bytes
  const string bytes = data.s;
  for (size_t k = encodings.a.size(); k-- > 0;) {
    const Msg& e = encodings.a[k];
    const string kind = e.get("kind")->s;
    auto num = [&](const char* key) { return e.get(key)->num(); };
    if (kind == "ByteArray") {
      v = byteArray(bytes, (int)num("type"));
    } else if (kind == "FixedPoint") {
      for (double& x : v.num) x = x / num("factor");
    } else if (kind == "IntervalQuantization") {
      const double lo = num("min"), step = (num("max") - lo) / (num("numSteps") - 1);
      for (double& x : v.num) x = lo + step * x;
    } else if (kind == "RunLength") {
      vector<double> out;
      for (size_t i = 0; i + 1 < v.num.size(); i += 2) for (long long r = 0; r < (long long)v.num[i + 1]; ++r) out.push_back(v.num[i]);
      v.num = std::move(out);
    } else if (kind == "Delta") {
      double acc = num("origin");
      for (double& x : v.num) { acc += x; x = acc; }
    } else if (kind == "IntegerPacking") {
      const bool unsign = e.get("isUnsigned")->b;
      const int bc = (int)num("byteCount");
      // The upper limit continues a value, and for a signed packing the lower too.
      const double upper = unsign ? (bc == 1 ? 0xff : 0xffff) : (bc == 1 ? 0x7f : 0x7fff);
      const double lower = -upper - 1;
      vector<double> out;
      double acc = 0;
      for (const double x : v.num) {
        acc += x;
        if (x != upper && (unsign || x != lower)) { out.push_back(acc); acc = 0; }
      }
      v.num = std::move(out);
    } else if (kind == "StringArray") {
      Msg offData; offData.k = Msg::Bin; offData.s = e.get("offsets")->s;
      Msg idxData; idxData.k = Msg::Bin; idxData.s = bytes;
      const Column offsets = decode(offData, *e.get("offsetEncoding"));
      const Column indices = decode(idxData, *e.get("dataEncoding"));
      const string& text = e.get("stringData")->s;
      vector<string> strings;
      for (size_t i = 0; i + 1 < offsets.num.size(); ++i) {
        strings.push_back(text.substr((size_t)offsets.num[i], (size_t)(offsets.num[i + 1] - offsets.num[i])));
      }
      v.strings = true;
      v.str.clear();
      for (const double ix : indices.num) v.str.push_back(ix < 0 ? string() : strings[(size_t)ix]);
    } else {
      throw std::runtime_error("BinaryCIF: unknown encoding " + kind);
    }
    raw = false;
  }
  (void)raw;
  return v;
}

/** `String(x)`: an integral value as its digits, any other as the shortest text that reads back. */
inline string text(double x) {
  if (x == (double)(long long)x && std::fabs(x) < 1e15) return std::to_string((long long)x);
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof buf, x);
  return string(buf, r.ptr);
}

/** A BinaryCIF file's first data block, as `cif::parseBlock` shapes an mmCIF's. */
inline cif::Block parseBinaryBlock(const string& bytes) {
  Unpacker u(bytes);
  const Msg file = u.value();
  const Msg* blocks = file.get("dataBlocks");
  if (!blocks || blocks->a.empty()) throw std::runtime_error("BinaryCIF holds no data block");
  cif::Block out;
  for (const Msg& category : blocks->a[0].get("categories")->a) {
    string name = category.get("name")->s;
    if (!name.empty() && name[0] == '_') name.erase(0, 1);
    auto& cat = out[name];
    for (const Msg& column : category.get("columns")->a) {
      const Msg& data = *column.get("data");
      const Column values = decode(*data.get("data"), *data.get("encoding"));
      const Msg* maskMsg = column.get("mask");
      Column mask;
      const bool masked = maskMsg && maskMsg->k == Msg::Map;
      if (masked) mask = decode(*maskMsg->get("data"), *maskMsg->get("encoding"));
      vector<string>& dst = cat[column.get("name")->s];
      const size_t n = values.strings ? values.str.size() : values.num.size();
      dst.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        if (masked && i < mask.num.size() && mask.num[i] == 1) { dst.push_back("."); continue; }
        if (masked && i < mask.num.size() && mask.num[i] == 2) { dst.push_back("?"); continue; }
        dst.push_back(values.strings ? values.str[i] : text(values.num[i]));
      }
    }
  }
  return out;
}

/** A structure's block: BinaryCIF or mmCIF, from its (already unwrapped) bytes. */
inline cif::Block parseStructure(const string& bytes) {
  return isBinaryCif(bytes) ? parseBinaryBlock(bytes) : cif::parseBlock(bytes);
}

/**
 * A structure file's bytes, plain or unwrapped from Zstandard or gzip - upstream takes all three.
 * gzip through a `gzip -dc` child, as cuda/featurise does (no library).
 */
inline string readStructureFile(const string& path, bool* ok) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { *ok = false; return string(); }
  string bytes;
  char buf[65536];
  size_t got;
  while ((got = fread(buf, 1, sizeof buf, f)) > 0) bytes.append(buf, got);
  fclose(f);
  *ok = true;
  if (bytes.size() > 2 && (uint8_t)bytes[0] == 0x1f && (uint8_t)bytes[1] == 0x8b) {
    string quoted = "'";
    for (const char c : path) { if (c == '\'') quoted += "'\\''"; else quoted += c; }
    quoted += "'";
    FILE* p = popen(("gzip -dc " + quoted).c_str(), "r");
    if (!p) { fprintf(stderr, "%s: gzip did not start\n", path.c_str()); exit(1); }
    string out;
    while ((got = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, got);
    if (pclose(p) != 0) { fprintf(stderr, "%s: did not decompress as gzip\n", path.c_str()); exit(1); }
    return out;
  }
  return lf::zstd::isCompressed(bytes) ? lf::zstd::decompress(bytes, path.c_str()) : bytes;
}

}  // namespace bcif
}  // namespace rfd4p
