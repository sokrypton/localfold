// metal/core/model.h: bundles decoded on the device straight from their mapped shards, an input directory read into
// one buffer.
#include "model.h"
#include "json.h"
#include <dirent.h>
#include <dlfcn.h>
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
void Model::retire(const std::string& name) {
  auto it = t.find(name);
  if (it == t.end() || retired.count(name)) return;
  retired.insert(name);
  Tensor& x = it->second;
  if (!x.owned) return;
  for (const void* p : {(const void*)x.f16, (const void*)x.f32}) if (p) release(p);   // (its own, and any conversion)
  weightHeld -= x.n * (x.f16 && !x.madeF32 ? 2 : 4);
  x.f16 = nullptr; x.f32 = nullptr; x.i32 = nullptr; x.madeF32 = false;
}
// The walk's allocations rebuilt without the retired tensors (the live ones copied over in stream order, the old
// allocations given back once the work issued so far is done), and every retired tensor's own copies released
void Model::compact() {
  if (retired.empty()) return;
  auto inBlock = [&](const void* p, const Block& b) { return (const char*)p >= b.base && (const char*)p < b.base + b.bytes; };
  // a retired tensor's own copies (an on-demand conversion: its own allocation), before the blocks are rebuilt
  for (auto& name : retired) {
    auto it = t.find(name);
    if (it == t.end()) continue;
    Tensor& x = it->second;
    if (x.owned) continue;
    for (const void* p : {(const void*)x.f16, (const void*)x.f32}) {
      if (!p) continue;
      bool owned = false;
      for (auto& b : blocks) owned |= inBlock(p, b);
      if (!owned) release(p);       // (an on-demand conversion: its own allocation)
    }
  }
  for (auto& b : blocks) {
    bool dead = false;      // (a block holding nothing retired is left where it is)
    for (auto& name : retired) {
      const Tensor& x = t.at(name);
      for (const void* p : {(const void*)x.f16, (const void*)x.f32, (const void*)x.i32}) dead |= p && inBlock(p, b);
    }
    if (!dead) continue;
    struct Live { Tensor* x; const char* from; size_t bytes; };
    std::vector<Live> live;
    size_t bytes = 0;
    for (auto& [name, x] : t) {
      const void* p = x.f16 ? (const void*)x.f16 : x.f32 ? (const void*)x.f32 : (const void*)x.i32;
      if (x.f16 && x.f32) p = inBlock(x.f16, b) ? (const void*)x.f16 : (const void*)x.f32;
      if (!p || !inBlock(p, b) || retired.count(name)) continue;
      size_t n = x.n * (p == (const void*)x.f16 ? 2 : 4);
      bytes = (bytes + 31) / 32 * 32;
      live.push_back({&x, (const char*)p, n});
      bytes += n;
    }
    char* base = bytes ? (char*)alloc(bytes) : nullptr;
    size_t at = 0;
    for (auto& l : live) {
      at = (at + 31) / 32 * 32;
      copy(base + at, l.from, l.bytes);
      if ((const void*)l.from == (const void*)l.x->f16) l.x->f16 = (half*)(base + at);
      else { if (l.x->i32 == (const int*)l.from) l.x->i32 = (int*)(base + at); if (l.x->f32 == (const float*)l.from) l.x->f32 = (float*)(base + at); }
      at += l.bytes;
    }
    weightHeld -= b.bytes; weightHeld += bytes;
    release(b.base);
    b = {base, bytes};
  }
  for (auto& name : retired) {
    auto it = t.find(name);
    if (it == t.end()) continue;
    Tensor& x = it->second;
    x.f16 = nullptr; x.f32 = nullptr; x.i32 = nullptr; x.madeF32 = false;      // (its name, length and shape kept)
  }
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
static void dieRetired(const std::string& name) { die("%s was retired once its derived form was built, and is read again", name.c_str()); }
const float* Model::f(const std::string& name) {
  Tensor& x = const_cast<Tensor&>(at(name));
  if (!x.f32 && !x.f16 && !x.i32 && retired.count(name)) dieRetired(name);
  if (x.f32) return x.f32;
  if (!x.f16) die("%s is not a float tensor", name.c_str());
  x.f32 = allocT<float>(x.n);
  x.madeF32 = true;
  toFloat(x.f16, x.f32, x.n);
  return x.f32;
}
const half* Model::h(const std::string& name) {
  Tensor& x = const_cast<Tensor&>(at(name));
  if (!x.f32 && !x.f16 && !x.i32 && retired.count(name)) dieRetired(name);
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
void Model::unloadBundle(const std::string& prefix) {
  auto it = bundleAllocs.find(prefix);
  if (it == bundleAllocs.end()) die("no bundle %s to unload", prefix.c_str());
  const std::string p = prefix + "/";
  for (auto x = t.begin(); x != t.end();) {
    if (x->first.compare(0, p.size(), p)) { ++x; continue; }
    Tensor& v = x->second;
    // a tensor holding both forms holds one conversion made on demand: the float32 one where madeF32 says so, else
    // the float16 one
    if (v.f32 && v.f16) release(v.madeF32 ? (const void*)v.f32 : (const void*)v.f16);
    weightHeld -= v.n * (v.madeF32 || !v.f32 ? 2 : 4);
    x = t.erase(x);
  }
  for (void* b : it->second) release(b);
  bundleAllocs.erase(it);
  for (auto d = derivedW.begin(); d != derivedW.end();) {
    size_t colon = d->first.find(':');      // (a derived weight's key is "<what>:<the tensor it was made from>")
    if (colon != std::string::npos && !d->first.compare(colon + 1, p.size(), p)) { release(d->second); d = derivedW.erase(d); }
    else ++d;
  }
}
void Model::swapPrefix(const std::string& to, const std::string& from) {
  auto swapIn = [&](auto& m) {
    std::vector<std::string> keys;
    for (auto& [name, v] : m) if (!name.compare(0, from.size(), from)) keys.push_back(name.substr(from.size()));
    for (auto& k : keys) {
      auto a = m.find(to + k), b = m.find(from + k);
      if (a == m.end()) { m[to + k] = b->second; m.erase(b); }
      else std::swap(a->second, b->second);
    }
  };
  swapIn(t); swapIn(metaV);
}
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
    else if (dtype == "int8") {   // symmetric: a code a byte and a float16 scale a block
      rec.e.kind = 2; rec.e.bits = 8; rec.e.block = (uint)r.get("block")->num; rec.e.scale = (u64)r.get("scaleOffset")->num;
    }
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
  for (char* b : {base32, base16}) if (b) bundleAllocs[prefix].push_back(b);
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
  else if (dtype == "int8") {   // symmetric: a code a byte and a float16 scale a block
    x.e.kind = 2; x.e.bits = 8; x.e.block = (uint)r.get("block")->num; x.e.scale = (u64)r.get("scaleOffset")->num;
  }
  else if (dtype.size() == 4 && !dtype.compare(0, 3, "int") && dtype[3] >= '1' && dtype[3] <= '7') {
    x.e.kind = 3; x.e.bits = dtype[3] - '0'; x.e.block = (uint)r.get("block")->num;
    x.e.scale = (u64)r.get("scaleOffset")->num;
    const Json* z = r.get("zeroOffset");
    if (!z) die("%s: %s with no zero offset", name.c_str(), dtype.c_str());
    x.e.zero = (u64)z->num;
  } else die("%s: unsupported dtype %s", name.c_str(), dtype.c_str());
  return x;
}
// ---------------------------------------------------------------- af3-any-model blobs
// A blob (<model>.bin.zst) is a stream of records - [scope, name, dtype, shape, bytes] - decompressed once into
// `<blob>.raw/NNN` shards of at most 256 MB, cut at record boundaries (never before an int8 tensor's `__q_scale`), with
// `done` listing them: the cache cuda/af3 keeps too. zstd is linked into the port binaries
// (metal/build.sh): its functions are found in the process itself.
struct ZIn { const void* src; size_t size, pos; };
struct ZOut { void* dst; size_t size, pos; };
std::vector<std::string> blobShards(const std::string& blob) {
  std::string dir = blob + ".raw";
  auto listed = [&]() {
    std::vector<std::string> out; std::ifstream done(dir + "/done"); std::string f;
    while (done >> f) out.push_back(dir + "/" + f);
    return out;
  };
  if (auto have = listed(); !have.empty()) return have;
  void* self = dlopen(nullptr, RTLD_NOW);
  auto sym = [&](const char* n) { void* f = dlsym(self, n); if (!f) die("reading %s needs zstd, and this binary has no %s", blob.c_str(), n); return f; };
  auto create = (void* (*)())sym("ZSTD_createDStream");
  auto init = (size_t (*)(void*))sym("ZSTD_initDStream");
  auto step = (size_t (*)(void*, ZOut*, ZIn*))sym("ZSTD_decompressStream");
  auto isError = (unsigned (*)(size_t))sym("ZSTD_isError");
  auto errName = (const char* (*)(size_t))sym("ZSTD_getErrorName");
  auto freeStream = (size_t (*)(void*))sym("ZSTD_freeDStream");
  FILE* in = fopen(blob.c_str(), "rb");
  if (!in) die("cannot read %s", blob.c_str());
  std::string tmp = dir + ".part." + std::to_string(getpid());
  mkdir(tmp.c_str(), 0755);
  const size_t SHARD = (size_t)256 << 20;
  std::vector<char> pending; std::vector<std::string> names; FILE* out = nullptr; size_t inShard = 0;
  auto emit = [&](bool all) {
    size_t at = 0;
    for (;;) {
      if (pending.size() - at < 20) break;
      int32_t h[5]; memcpy(h, pending.data() + at, 20);
      size_t len = 20 + (size_t)h[0] + h[1] + h[2] + 4 * (size_t)h[3] + (size_t)(uint32_t)h[4];
      if (pending.size() - at < len) break;
      std::string nm(pending.data() + at + 20 + h[0], (size_t)h[1]);
      bool scale = nm.size() > 9 && !nm.compare(nm.size() - 9, 9, "__q_scale");
      if (!out || (inShard > 0 && inShard + len > SHARD && !scale)) {
        if (out) fclose(out);
        char file[16]; snprintf(file, sizeof file, "%03zu", names.size()); names.push_back(file);
        out = fopen((tmp + "/" + file).c_str(), "wb"); inShard = 0;
        if (!out) die("cannot write %s/%s", tmp.c_str(), file);
      }
      fwrite(pending.data() + at, 1, len, out); inShard += len; at += len;
    }
    pending.erase(pending.begin(), pending.begin() + at);
    if (all && !pending.empty()) die("%s: a truncated record", blob.c_str());
  };
  void* ds = create(); init(ds);
  std::vector<char> ib(1 << 20), ob(1 << 22);
  size_t got, last = 0;
  auto take = [&](ZIn& zi) {
    ZOut zo{ob.data(), ob.size(), 0};
    last = step(ds, &zo, &zi);
    if (isError(last)) die("%s: %s", blob.c_str(), errName(last));
    pending.insert(pending.end(), ob.data(), ob.data() + zo.pos);
    if (pending.size() > ((size_t)64 << 20)) emit(false);
    return zo.pos;
  };
  while ((got = fread(ib.data(), 1, ib.size(), in)) > 0) { ZIn zi{ib.data(), got, 0}; while (zi.pos < zi.size) take(zi); }
  while (last != 0) { ZIn zi{nullptr, 0, 0}; if (take(zi) == 0) break; }
  freeStream(ds); fclose(in);
  if (last != 0) die("%s: a truncated stream", blob.c_str());
  emit(true);
  if (out) fclose(out);
  { std::ofstream done(tmp + "/done"); for (auto& n : names) done << n << "\n"; }
  if (rename(tmp.c_str(), dir.c_str())) die("cannot write %s", dir.c_str());
  return listed();
}
// the one *.bin.zst a directory holds, or "" when the directory is a bundle (a manifest.json)
std::string findBlob(const std::string& dir) {
  struct stat st;
  if (!stat((dir + "/manifest.json").c_str(), &st)) return "";
  std::vector<std::string> found;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) { std::string n = e->d_name; if (n.size() > 8 && n.compare(n.size() - 8, 8, ".bin.zst") == 0) found.push_back(n); }
    closedir(d);
  }
  if (found.size() != 1) die("%s holds %s and no manifest.json", dir.c_str(), found.empty() ? "no *.bin.zst" : "more than one *.bin.zst");
  return dir + "/" + found[0];
}
// every tensor of a decompressed blob as a source: float32, float16, bfloat16, or int8 with its float32 scales
std::map<std::string, Src> blobSources(const std::string& blob) {
  struct Rec { std::string dtype; std::vector<long long> shape; size_t offset, bytes; std::string file; };
  std::map<std::string, Rec> recs;
  for (const std::string& shard : blobShards(blob)) {
    FILE* f = fopen(shard.c_str(), "rb");
    if (!f) die("cannot read %s", shard.c_str());
    size_t at = 0;
    for (;;) {
      int32_t h[5];
      if (fread(h, 4, 5, f) != 5) break;
      std::string scope(h[0], 0), name(h[1], 0), dtype(h[2], 0);
      std::vector<int32_t> shape(h[3]);
      if (fread(scope.data(), 1, h[0], f) != (size_t)h[0] || fread(name.data(), 1, h[1], f) != (size_t)h[1] ||
          fread(dtype.data(), 1, h[2], f) != (size_t)h[2] || fread(shape.data(), 4, h[3], f) != (size_t)h[3])
        die("%s: a truncated record", shard.c_str());
      at += 20 + h[0] + h[1] + h[2] + 4 * (size_t)h[3];
      recs[scope + "/" + name] = {dtype, std::vector<long long>(shape.begin(), shape.end()), at, (size_t)(uint32_t)h[4], shard};
      at += (size_t)(uint32_t)h[4];
      fseek(f, (long)at, SEEK_SET);
    }
    fclose(f);
  }
  std::map<std::string, Src> out;
  for (auto& [name, r] : recs) {
    if (!name.compare(0, 9, "__meta__/")) continue;
    if (name.size() > 9 && !name.compare(name.size() - 9, 9, "__q_scale")) continue;
    Src x; x.e = {}; x.file = r.file; x.shape = r.shape;
    size_t n = 1; for (long long d : r.shape) n *= (size_t)d;
    x.e.n = n; x.e.src = r.offset;
    if (r.dtype == "float32") x.e.kind = 0;
    else if (r.dtype == "float16") x.e.kind = 1;
    else if (r.dtype == "uint16") x.e.kind = 7;
    else if (r.dtype == "int8") {
      auto sc = recs.find(name + "__q_scale");
      if (sc == recs.end() || sc->second.dtype != "float32" || r.shape.empty() || sc->second.file != r.file)
        die("%s: int8 with no float32 __q_scale beside it", name.c_str());
      const auto& ss = sc->second.shape;
      if (ss.back() != r.shape.back() || (ss.size() != 1 && ss.size() != 2)) die("%s: a __q_scale of an unknown layout", name.c_str());
      x.e.kind = 6; x.e.scale = sc->second.offset; x.e.block = (uint)r.shape.back();
      x.e.zero = n / x.e.block; x.e.bits = ss.size() == 1 ? 1 : (uint)ss[0];
    } else die("%s: unsupported blob dtype %s", name.c_str(), r.dtype.c_str());
    out[name] = x;
  }
  return out;
}

std::map<std::string, Src> sources(const std::string& dir, const std::string& deltaDir, std::map<std::string, double>* meta) {
  std::string blob = findBlob(dir);
  if (!blob.empty()) {
    if (!deltaDir.empty()) die("a delta over an af3-any-model blob");
    return blobSources(blob);
  }
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
                           const std::function<bool(const std::string&, size_t)>& asHalf,
                           const std::function<bool(const std::string&)>& own) {
  std::map<std::string, Src> src = sources(dir, deltaDir, &metaV);
  struct Part { char op; int rank; i64 dims[6], ds[6], s[2][6]; i64 dst, off[2]; std::string from[2]; };
  struct Target { std::string name; size_t n = 0; char kind = 0; std::string from; size_t first = 0; std::vector<float> consts;
                  std::vector<Part> parts; bool isInt = false, half = false, owned = false; void* own = nullptr; size_t at = 0; };
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
  // the destinations: float32 tensors in one allocation, float16 ones in another - a slice decoded straight into its
  // half (no float32 copy of it is ever made), the half tensors built from parts or literals first and contiguous, so
  // they alone are assembled in float32 scratch and converted in one pass
  size_t n32 = 0, n16 = 0, nStaged = 0;
  for (auto& t : targets) {
    t.half = asHalf && !t.isInt && asHalf(t.name, t.n);
    t.owned = own && t.kind == 'b' && !t.isInt && own(t.name);       // (a slice decoded straight into its own allocation)
    if (t.owned) t.own = alloc(t.n * (t.half ? 2 : 4));
  }
  for (int pass = 0; pass < 3; ++pass)
    for (auto& t : targets) {
      if (t.owned) continue;
      bool staged = t.half && (t.kind == 'p' || t.kind == 'c');
      if (pass != (t.half ? (staged ? 0 : 1) : 2)) continue;
      size_t& at = t.half ? n16 : n32;
      at = (at + 7) / 8 * 8; t.at = at; at += t.n;
      if (staged) nStaged = n16;
    }
  float* base32 = n32 ? (float*)alloc(n32 * 4) : nullptr;
  half* base16 = n16 ? (half*)alloc(n16 * 2) : nullptr;
  float* temp16 = nStaged ? (float*)alloc(nStaged * 4) : nullptr;
  if (base32) blocks.push_back({(char*)base32, n32 * 4});
  if (base16) blocks.push_back({(char*)base16, n16 * 2});
  weightHeld += n32 * 4 + n16 * 2;
  for (auto& t : targets) if (t.owned) weightHeld += t.n * (t.half ? 2 : 4);
  auto dstOf = [&](const Target& t) { return t.half ? temp16 + t.at : base32 + t.at; };    // (a staged half's float32 home)
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
  for (auto& t : targets) {
    if (t.kind != 'b') continue;
    if (!t.half) { decodeInto(src[t.from], t.owned ? (float*)t.own : base32 + t.at, t.first, t.n); continue; }
    // straight into float16: the slice rounded once (a delta's base rounded, then the delta added and rounded)
    const Src& sr = src[t.from];
    DecodeEntry e = sr.e; e.dst = (u64)(t.owned ? (half*)t.own : base16 + t.at); e.first = t.first; e.n = t.n; e.out16 = 1; e.flags = 0;
    entries.push_back({sr.file, e});
    if (sr.hasDelta) { DecodeEntry d = sr.de; d.dst = e.dst; d.first = t.first; d.n = t.n; d.out16 = 1; d.flags = 2; entries.push_back({sr.dfile, d}); }
  }
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
  if (nStaged) toHalf(temp16, base16, nStaged);
  // the tensors, with their shapes from the walk's own metadata
  for (auto& t : targets) {
    Tensor x; x.n = t.n; x.segment = -1; x.owned = t.owned;
    if (t.half) x.f16 = t.owned ? (half*)t.own : base16 + t.at;
    else { x.f32 = t.owned ? (float*)t.own : base32 + t.at; if (t.isInt) x.i32 = (int*)x.f32; }
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
