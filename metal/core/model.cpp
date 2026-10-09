// metal/core/model.h: bundles decoded on the device straight from their mapped shards, an input directory read into
// one buffer.
#include "model.h"
#include "json.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iterator>

namespace mt {
Model M;

const Tensor& Model::at(const std::string& name) const {
  auto it = t.find(name);
  if (it == t.end()) die("no tensor %s", name.c_str());
  return it->second;
}
double Model::meta(const std::string& name) const {
  auto it = metaV.find(name);
  if (it == metaV.end()) die("no metadata %s", name.c_str());
  return it->second;
}
int64_t Model::dim(const std::string& name, int k) const {
  const Tensor& x = at(name);
  if (k < 0) k += (int)x.shape.size();
  if (k < 0 || k >= (int)x.shape.size()) die("%s has no dimension %d", name.c_str(), k);
  return x.shape[k];
}
const float* Model::f(const std::string& name) {
  Tensor& x = const_cast<Tensor&>(at(name));
  if (x.f32) return x.f32;
  if (!x.f16) die("%s is not a float tensor", name.c_str());
  x.f32 = allocT<float>(x.n);
  x.madeF32 = true;
  toFloat(x.f16, x.f32, x.n);
  return x.f32;
}
const half* Model::h(const std::string& name) {
  Tensor& x = const_cast<Tensor&>(at(name));
  if (x.f16) return x.f16;
  if (!x.f32) die("%s is not a float tensor", name.c_str());
  x.f16 = allocT<half>(x.n);
  toHalf(x.f32, x.f16, x.n);
  return x.f16;
}
const int* Model::i(const std::string& name) {
  const Tensor& x = at(name);
  if (!x.i32) die("%s is not an integer tensor", name.c_str());
  return x.i32;
}
const float* Model::hostF(const std::string& name) { const float* p = f(name); sync(); return (const float*)host(p); }
const int* Model::hostI(const std::string& name) { const int* p = i(name); sync(); return (const int*)host(p); }
void Model::dropFloatCopies(const std::string& prefix) {
  for (auto& [name, x] : t)
    if (!name.rfind(prefix, 0) && x.madeF32) { release(x.f32); x.f32 = nullptr; x.madeF32 = false; }
}

void Model::loadBundle(const std::string& dir, const std::string& prefix,
                       const std::function<bool(const std::string&, size_t)>& asHalf) {
  std::ifstream in(dir + "/manifest.json");
  if (!in) die("no %s/manifest.json", dir.c_str());
  Json m = Json::parse(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
  for (const char* block : {"trunk", "languageModel"})
    if (const Json* b = m.get(block))
      for (auto& [k, v] : b->obj) if (v.t == Json::NUM) metaV["meta/" + k] = v.num;
  const Json* tensors = m.get("tensors");
  if (!tensors) die("%s/manifest.json has no tensors", dir.c_str());
  // every tensor's destination: one float32 and one float16 allocation for the bundle, each tensor 16-byte aligned
  struct Rec { std::string name; DecodeEntry e; std::string file; bool out16; };
  std::vector<Rec> recs;
  size_t bytes32 = 0, bytes16 = 0;
  for (auto& [name, r] : tensors->obj) {
    Rec rec; rec.name = prefix + "/" + name; rec.e = {};
    rec.file = r.get("file")->str;
    const std::string& dtype = r.get("dtype")->str;
    size_t n = 1; std::vector<int64_t> shape;
    for (auto& d : r.get("shape")->arr) { n *= (size_t)d.num; shape.push_back((int64_t)d.num); }
    rec.e.n = n;
    rec.e.src = r.get("byteOffset") ? (u64)r.get("byteOffset")->num : 0;
    if (dtype == "float32") rec.e.kind = 0;
    else if (dtype == "float16") rec.e.kind = 1;
    else if (dtype.size() == 4 && !dtype.compare(0, 3, "int") && dtype[3] >= '1' && dtype[3] <= '7') {
      rec.e.kind = 3; rec.e.bits = dtype[3] - '0';
      rec.e.block = (uint)r.get("block")->num;
      rec.e.scale = (u64)r.get("scaleOffset")->num;
      const Json* z = r.get("zeroOffset");
      if (!z) die("%s: %s with no zero offset (a symmetric code this port does not read)", rec.name.c_str(), dtype.c_str());
      rec.e.zero = (u64)z->num;
    } else die("%s: unsupported dtype %s", rec.name.c_str(), dtype.c_str());
    rec.out16 = asHalf && asHalf(rec.name, n);
    rec.e.out16 = rec.out16;
    size_t& at = rec.out16 ? bytes16 : bytes32;
    at = (at + 15) / 16 * 16;
    rec.e.dst = at;
    at += n * (rec.out16 ? 2 : 4);
    Tensor x; x.shape = shape; x.n = n; x.segment = -1;
    if (t.count(rec.name)) die("%s is in two bundles", rec.name.c_str());
    t[rec.name] = x;
    recs.push_back(rec);
  }
  char* base32 = bytes32 ? (char*)alloc(bytes32) : nullptr;
  char* base16 = bytes16 ? (char*)alloc(bytes16) : nullptr;
  weightHeld += bytes32 + bytes16;
  for (auto& rec : recs) {
    Tensor& x = t[rec.name];
    if (rec.out16) { x.f16 = (half*)(base16 + rec.e.dst); rec.e.dst = (u64)x.f16; }
    else { x.f32 = (float*)(base32 + rec.e.dst); rec.e.dst = (u64)x.f32; }
  }
  // each shard mapped (no copy: the GPU reads it where the file is) and its tensors decoded in one dispatch
  std::map<std::string, std::vector<DecodeEntry>> byFile;
  for (auto& rec : recs) byFile[rec.file].push_back(rec.e);
  std::vector<void*> maps;
  for (auto& [file, entries] : byFile) {
    size_t n;
    void* raw = mapFile(dir + "/" + file, &n);
    maps.push_back(raw);
    for (auto& e : entries) {
      size_t end = e.kind == 0 ? e.src + 4 * e.n : e.kind == 1 ? e.src + 2 * e.n : e.src + (e.n * e.bits + 7) / 8;
      if (end > n) die("%s/%s is shorter than its manifest says", dir.c_str(), file.c_str());
    }
    for (size_t t0 = 0; t0 < entries.size(); t0 += 65535) {
      size_t cnt = std::min<size_t>(65535, entries.size() - t0);
      DecodeEntry* table = uploadNew(entries.data() + t0, cnt);
      DecodeArgs a{(const uchar*)raw, table};
      dispatch("lf_decode", &a, sizeof a, Grid{64, (uint32_t)cnt, 1}, 256, 0, "bundle decode");
      release(table);
    }
  }
  for (void* p : maps) unmapFile(p);
}

// ---------------------------------------------------------------- the weight walk
namespace {
struct Src { std::string file; DecodeEntry e; std::vector<long long> shape; bool hasDelta = false; std::string dfile; DecodeEntry de; };
Json readJson(const std::string& path) {
  std::ifstream in(path);
  if (!in) die("no %s", path.c_str());
  return Json::parse(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
}
Src record(const std::string& where, const std::string& name, const Json& r) {
  Src x; x.e = {};
  x.file = where + "/" + r.get("file")->str;
  const std::string& dtype = r.get("dtype")->str;
  size_t n = 1;
  for (auto& d : r.get("shape")->arr) { n *= (size_t)d.num; x.shape.push_back((long long)d.num); }
  x.e.n = n;
  x.e.src = r.get("byteOffset") ? (u64)r.get("byteOffset")->num : 0;
  if (dtype == "float32") x.e.kind = 0;
  else if (dtype == "float16") x.e.kind = 1;
  else if (dtype.size() == 4 && !dtype.compare(0, 3, "int") && dtype[3] >= '1' && dtype[3] <= '7') {
    x.e.kind = 3; x.e.bits = dtype[3] - '0'; x.e.block = (uint)r.get("block")->num;
    x.e.scale = (u64)r.get("scaleOffset")->num;
    const Json* z = r.get("zeroOffset");
    if (!z) die("%s: %s with no zero offset", name.c_str(), dtype.c_str());
    x.e.zero = (u64)z->num;
  } else die("%s: unsupported dtype %s", name.c_str(), dtype.c_str());
  return x;
}
std::map<std::string, Src> sources(const std::string& dir, const std::string& deltaDir, std::map<std::string, double>* meta) {
  Json m = readJson(dir + "/manifest.json");
  if (meta)
    for (const char* block : {"trunk", "languageModel"})
      if (const Json* b = m.get(block))
        for (auto& [k, v] : b->obj) if (v.t == Json::NUM) (*meta)["meta/" + k] = v.num;
  std::map<std::string, Src> out;
  for (auto& [name, r] : m.get("tensors")->obj) out[name] = record(dir, name, r);
  if (deltaDir.empty()) return out;
  Json dm = readJson(deltaDir + "/manifest.json");
  const Json* header = dm.get("delta"); const Json* dt = dm.get("tensors");
  if (!header || !dt) die("%s carries no delta header - it is not a delta", deltaDir.c_str());
  auto names = [&](const char* key) { std::vector<std::string> v; if (const Json* l = header->get(key)) for (auto& x : l->arr) v.push_back(x.str); return v; };
  for (auto& n : names("absent")) out.erase(n);
  for (auto& n : names("whole")) {
    const Json* r = dt->get(n);
    if (!r) die("%s: the delta names %s whole and holds no such tensor", deltaDir.c_str(), n.c_str());
    out[n] = record(deltaDir, n, *r);
  }
  for (auto& n : names("addTo")) {
    const Json* r = dt->get(n);
    auto it = out.find(n);
    if (!r || it == out.end()) die("%s: %s is addTo but missing from the delta or the base", deltaDir.c_str(), n.c_str());
    Src d = record(deltaDir, n, *r);
    if (d.e.n != it->second.e.n) die("%s: %s differs in length from its base", deltaDir.c_str(), n.c_str());
    it->second.hasDelta = true; it->second.dfile = d.file; it->second.de = d.e;
  }
  return out;
}
// decode entries file by file: each shard mapped, its plain entries first and the accumulating ones after
void decodeAll(std::vector<std::pair<std::string, DecodeEntry>>& entries) {
  std::map<std::string, std::vector<DecodeEntry>> byFile[2];
  for (auto& [f, e] : entries) byFile[(e.flags & 2) ? 1 : 0][f].push_back(e);
  for (int pass = 0; pass < 2; ++pass)
    for (auto& [file, list] : byFile[pass]) {
      size_t n;
      void* raw = mapFile(file, &n);
      for (size_t t0 = 0; t0 < list.size(); t0 += 65535) {
        size_t cnt = std::min<size_t>(65535, list.size() - t0);
        DecodeEntry* table = uploadNew(list.data() + t0, cnt);
        DecodeArgs a{(const uchar*)raw, table};
        dispatch("lf_decode", &a, sizeof a, Grid{64, (uint32_t)cnt, 1}, 256, 0, "bundle decode");
        release(table);
      }
      unmapFile(raw);
    }
}
}  // namespace

std::map<std::string, std::vector<long long>> Model::bundleShapes(const std::string& dir, const std::string& deltaDir) {
  std::map<std::string, std::vector<long long>> out;
  for (auto& [n, s] : sources(dir, deltaDir, nullptr)) out[n] = s.shape;
  return out;
}

void Model::loadBundleWalk(const std::string& dir, const std::vector<std::string>& lines, const std::string& deltaDir,
                           const std::function<bool(const std::string&, size_t)>& asHalf) {
  std::map<std::string, Src> src = sources(dir, deltaDir, &metaV);
  struct Part { char op; int rank; i64 dims[6], ds[6], s[2][6]; i64 dst, off[2]; std::string from[2]; };
  struct Target { std::string name; size_t n = 0; char kind = 0; std::string from; size_t first = 0; std::vector<float> consts;
                  std::vector<Part> parts; bool isInt = false, half = false; size_t at = 0; };
  std::vector<Target> targets;
  std::map<std::string, size_t> targetOf;
  auto target = [&](const std::string& name, size_t n) -> Target& {
    auto it = targetOf.find(name);
    if (it != targetOf.end()) {
      if (targets[it->second].n != n) die("the weight walk gives %s two lengths", name.c_str());
      return targets[it->second];
    }
    targetOf[name] = targets.size();
    targets.push_back({}); targets.back().name = name; targets.back().n = n;
    return targets.back();
  };
  for (const std::string& line : lines) {
    std::istringstream in(line); char kind; std::string name; in >> kind >> name;
    if (kind == 'm') { double v; in >> v; metaV[name] = v; continue; }
    if (kind == 'D') continue;
    if (kind == 'b') {
      std::string from; size_t first, n; in >> from >> first >> n;
      auto it = src.find(from);
      if (it == src.end() || first + n > it->second.e.n) die("the weight walk: %s does not hold [%zu, %zu) of %s", dir.c_str(), first, first + n, from.c_str());
      Target& t = target(name, n); t.kind = 'b'; t.from = from; t.first = first;
    } else if (kind == 'z') { size_t n; in >> n; target(name, n).kind = 'z'; }
    else if (kind == 'c') {
      size_t n; in >> n; Target& t = target(name, n); t.kind = 'c'; t.consts.resize(n);
      for (size_t k = 0; k < n; ++k) in >> t.consts[k];
    } else if (kind == 'p') {
      size_t n; Part p{}; in >> n >> p.op >> p.rank;
      if (p.rank < 1 || p.rank > 6) die("the weight walk: a malformed p line for %s", name.c_str());
      for (int k = 0; k < p.rank; ++k) in >> p.dims[k];
      in >> p.dst;
      for (int k = 0; k < p.rank; ++k) in >> p.ds[k];
      int nsrc = p.op == 'o' ? 0 : p.op == 'x' ? 2 : 1;
      for (int q = 0; q < nsrc; ++q) {
        in >> p.from[q] >> p.off[q];
        for (int k = 0; k < p.rank; ++k) in >> p.s[q][k];
        if (!src.count(p.from[q])) die("the weight walk: %s has no %s (another export of the bundle?)", dir.c_str(), p.from[q].c_str());
      }
      if (in.fail()) die("the weight walk: a malformed p line for %s", name.c_str());
      Target& t = target(name, n); t.kind = 'p'; t.parts.push_back(p);
      if (p.op == 'i') t.isInt = true;
    } else die("the weight walk: a line of kind %c", kind);
  }
  // the destinations: float32 tensors in one allocation; float16 ones gathered in float32 scratch at the same element
  // offsets, then converted into theirs in one pass
  size_t n32 = 0, n16 = 0;
  for (auto& t : targets) {
    t.half = asHalf && !t.isInt && asHalf(t.name, t.n);
    size_t& at = t.half ? n16 : n32;
    at = (at + 7) / 8 * 8; t.at = at; at += t.n;
  }
  float* base32 = n32 ? (float*)alloc(n32 * 4) : nullptr;
  half* base16 = n16 ? (half*)alloc(n16 * 2) : nullptr;
  float* temp16 = n16 ? (float*)alloc(n16 * 4) : nullptr;
  weightHeld += n32 * 4 + n16 * 2;
  auto dstOf = [&](const Target& t) { return t.half ? temp16 + t.at : base32 + t.at; };
  // the sources the parts gather from, whole, into scratch
  std::map<std::string, float*> scratchOf;
  size_t nScratch = 0;
  std::map<std::string, size_t> scratchAt;
  for (auto& t : targets) for (auto& p : t.parts) for (int q = 0; q < 2; ++q)
    if (!p.from[q].empty() && !scratchAt.count(p.from[q])) { nScratch = (nScratch + 7) / 8 * 8; scratchAt[p.from[q]] = nScratch; nScratch += src[p.from[q]].e.n; }
  float* scratchBase = nScratch ? (float*)alloc(nScratch * 4) : nullptr;
  std::vector<std::pair<std::string, DecodeEntry>> entries;
  auto decodeInto = [&](const Src& s, float* dst, size_t first, size_t n) {
    DecodeEntry e = s.e; e.dst = (u64)dst; e.first = first; e.n = n; e.out16 = 0; e.flags = s.hasDelta ? 1 : 0;
    entries.push_back({s.file, e});
    if (s.hasDelta) { DecodeEntry d = s.de; d.dst = (u64)dst; d.first = first; d.n = n; d.out16 = 0; d.flags = 2; entries.push_back({s.dfile, d}); }
  };
  for (auto& [name, at] : scratchAt) decodeInto(src[name], scratchBase + at, 0, src[name].e.n);
  for (auto& t : targets) if (t.kind == 'b') decodeInto(src[t.from], dstOf(t), t.first, t.n);
  decodeAll(entries);
  // zeros (the allocations are zeroed), literal values, and the parts
  for (auto& t : targets) if (t.kind == 'c') upload(dstOf(t), t.consts.data(), t.n * 4);
  std::vector<GatherPart> parts;
  for (auto& t : targets)
    for (auto& p : t.parts) {
      GatherPart g{}; g.dst = (u64)(dstOf(t) + p.dst); g.rank = p.rank; g.op = p.op; g.n = 1;
      for (int k = 0; k < p.rank; ++k) { g.dims[k] = p.dims[k]; g.n *= p.dims[k]; g.ds[k] = p.ds[k]; g.s0[k] = p.s[0][k]; g.s1[k] = p.s[1][k]; }
      if (!p.from[0].empty()) g.src0 = (u64)(scratchBase + scratchAt[p.from[0]] + p.off[0]);
      if (!p.from[1].empty()) g.src1 = (u64)(scratchBase + scratchAt[p.from[1]] + p.off[1]);
      parts.push_back(g);
    }
  for (size_t t0 = 0; t0 < parts.size(); t0 += 65535) {
    size_t cnt = std::min<size_t>(65535, parts.size() - t0);
    GatherPart* table = uploadNew(parts.data() + t0, cnt);
    GatherArgs a{table};
    dispatch("lf_gather", &a, sizeof a, Grid{64, (uint32_t)cnt, 1}, 256, 0, "weight gather");
    release(table);
  }
  if (n16) toHalf(temp16, base16, n16);
  // the tensors, with their shapes from the walk's own metadata
  for (auto& t : targets) {
    Tensor x; x.n = t.n; x.segment = -1;
    if (t.half) x.f16 = base16 + t.at;
    else { x.f32 = base32 + t.at; if (t.isInt) x.i32 = (int*)x.f32; }
    auto r = metaV.find(t.name + "#r");
    if (r != metaV.end()) for (int k = 0; k < (int)r->second; ++k) x.shape.push_back((int64_t)metaV[t.name + "#" + std::to_string(k)]);
    else x.shape = {(int64_t)t.n};
    if (t.isInt) x.f32 = nullptr;
    if (this->t.count(t.name)) die("%s is in two bundles", t.name.c_str());
    this->t[t.name] = x;
  }
  sync();
  if (scratchBase) release(scratchBase);
  if (temp16) release(temp16);
}

void Model::loadInput(const std::string& dir) {
  std::ifstream idx(dir + "/model.idx");
  if (!idx) die("no %s/model.idx", dir.c_str());
  int fd = open((dir + "/model.bin").c_str(), O_RDONLY);
  if (fd < 0) die("no %s/model.bin", dir.c_str());
  struct stat st; fstat(fd, &st);
  size_t bytes = (size_t)st.st_size;
  char* buf = (char*)alloc(std::max<size_t>(bytes, 16));
  inputBuffers.push_back(buf);
  sync();                                   // (the buffer is written on the host)
  char* h = (char*)host(buf);
  for (size_t got = 0; got < bytes;) {
    ssize_t r = pread(fd, h + got, bytes - got, (off_t)got);
    if (r <= 0) die("cannot read %s/model.bin", dir.c_str());
    got += (size_t)r;
  }
  close(fd);
  std::string line;
  while (std::getline(idx, line)) {
    std::istringstream ls(line); char kind; std::string name; ls >> kind >> name;
    if (kind == 'm') { double v; ls >> v; metaV[name] = v; inputNames.push_back(name); continue; }
    size_t off, n; ls >> off >> n;
    if ((off + n) * 4 > bytes) die("%s/model.bin is shorter than its index", dir.c_str());
    Tensor x; x.n = n; x.shape = {(int64_t)n}; x.segment = 0;
    if (kind == 'i') x.i32 = (int*)(buf + off * 4);
    else x.f32 = (float*)(buf + off * 4);
    if (t.count(name)) die("%s is in two places", name.c_str());
    t[name] = x;
    inputNames.push_back(name);
  }
}
void Model::unloadInput() {
  for (auto& n : inputNames) {
    auto it = t.find(n);
    if (it != t.end()) {
      if (it->second.f16) release(it->second.f16);     // (made on demand from the input's float32)
      t.erase(it);
    }
    metaV.erase(n);
  }
  inputNames.clear();
  for (void* b : inputBuffers) release(b);
  inputBuffers.clear();
}
}  // namespace mt
