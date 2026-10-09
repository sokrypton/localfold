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
