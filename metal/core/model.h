// A model's weights and an input, by name: page bundles (manifest.json and its shards) decoded on the device, and a
// featurised input directory (model.idx / model.bin, cuda/featurise's). One namespace for both - a bundle's tensors
// under its prefix ("f/...", "c/..."), the input's as they are named, metadata as "meta/<key>".
#pragma once
#include "core.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace mt {
struct Tensor {
  float* f32 = nullptr; half* f16 = nullptr; int* i32 = nullptr;
  std::vector<int64_t> shape;
  size_t n = 0;
  int segment = -1;
  bool madeF32 = false;      // (a float32 copy made on demand from the float16 tensor)
};
class Model {
 public:
  // a bundle under `prefix/`: every tensor decoded on the device, as float16 where asHalf(name, elements) says (no
  // float32 copy is made of it), float32 otherwise; the manifest's trunk / languageModel numbers as meta/<key>
  void loadBundle(const std::string& dir, const std::string& prefix,
                  const std::function<bool(const std::string&, size_t)>& asHalf = nullptr);
  // a bundle through a WEIGHT WALK (cuda/featurise's af2_weights.h / af3_weights.h): the port's own tensors as `lines`
  // describe them - `b` a slice of a bundle tensor, `z` zeros, `c` literal values, `p` gathered parts (a strided view,
  // a product, integers, ones), `m` metadata - and with `deltaDir` the bundle read as a delta model's base (its
  // `addTo` tensors the base rounded to float16 plus the delta, `whole` the delta's, `absent` gone)
  void loadBundleWalk(const std::string& dir, const std::vector<std::string>& lines, const std::string& deltaDir = "",
                      const std::function<bool(const std::string&, size_t)>& asHalf = nullptr);
  // the tensors a bundle holds, by name and shape (a delta's absent ones gone): what a walk is worked out from
  static std::map<std::string, std::vector<long long>> bundleShapes(const std::string& dir, const std::string& deltaDir = "");
  // an input directory; unloadInput() forgets it
  void loadInput(const std::string& dir);
  void unloadInput();

  bool has(const std::string& name) const { return t.count(name) || metaV.count(name); }
  double meta(const std::string& name) const;
  double meta(const std::string& name, double fallback) const { auto it = metaV.find(name); return it == metaV.end() ? fallback : it->second; }
  // a tensor as float32 / float16 (made from the other once and kept), an input's integers
  const float* f(const std::string& name);
  const half* h(const std::string& name);
  const int* i(const std::string& name);
  size_t len(const std::string& name) const { return at(name).n; }
  int64_t dim(const std::string& name, int k) const;
  int rank(const std::string& name) const { return (int)at(name).shape.size(); }
  // the bytes on the host (after the work that wrote them): a float32 tensor, an input's integers
  const float* hostF(const std::string& name);
  const int* hostI(const std::string& name);
  // a weight derived from others (an interleaving, a concatenation): built once by `make` into n elements
  template <class T> T* derived(const std::string& key, size_t n, const std::function<void(T*)>& make) {
    auto it = derivedW.find(key);
    if (it != derivedW.end()) return (T*)it->second;
    T* p = allocT<T>(n);
    make(p);
    derivedW[key] = p;
    return p;
  }
  // the float32 copies of half tensors made on demand: dropped (a stage that read them is done)
  void dropFloatCopies(const std::string& prefix);
  size_t weightBytes() const { return weightHeld; }

 private:
  const Tensor& at(const std::string& name) const;
  std::map<std::string, Tensor> t;
  std::map<std::string, double> metaV;
  std::map<std::string, void*> derivedW;
  std::vector<void*> inputBuffers;
  std::vector<std::string> inputNames;
  size_t weightHeld = 0;
};
extern Model M;

// a file mapped into a no-copy shared buffer the GPU reads (core.mm): its device address, and its release
void* mapFile(const std::string& path, size_t* bytes);
void unmapFile(void* p);
}  // namespace mt
