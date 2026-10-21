// af3-any-model's int8 blob (bioemu2.int8.bin.zst - what cuda/featurise/fetch.h fetches for `bioemu2`, and what
// cuda/bioemu2 and the worker fold with) read AS the page's bundle: a weight walk (metal/core's loadBundleWalk) that
// writes every bundle tensor this port reads from the blob's, so the port has one set of names and layouts whichever
// weights it is given. tools/export_bioemu2_model.py is the reading - this is that export, done on the device:
//
//   * the blob's three stacks (the MSA module's 4 layers, the PairMixer's 48, the denoiser's 16) UNSTACKED into
//     <stack>/<l>/<rest>: a slice of the stacked tensor, its leading dimension dropped;
//   * the PAIR TRACK - every tri_mul_out, tri_mul_in and pair transition of the MSA module and the PairMixer - in AF3's
//     layout under msa/<l>/ and pairmixer/<l>/: (in, out) where the blob is torch's (out, in); the triangle's a and b
//     INTERLEAVED by channel where Boltz blocks them [a | b], and the INCOMING triangle's halves swapped (so AF3's
//     incoming einsum computes Boltz's); the transition's fc1 and fc2 blocked into one [gate | value] transition1;
//   * everything else as it is, under its own name.
// Checked by name and shape against the int5 bundle's manifest: 1908 tensors either way, none missing, none extra.
#include "bioemu2.h"
#include <map>
#include <sstream>

namespace {
const std::string BLOB = "diffuser/~/bioemu2/";
std::string dims(const std::vector<long long>& s) {
  std::string out;
  for (long long d : s) out += " " + std::to_string(d);
  return out;
}
void shapeLines(std::vector<std::string>& L, const std::string& target, const std::vector<long long>& shape) {
  L.push_back("m " + target + "#r " + std::to_string(shape.size()));
  for (size_t k = 0; k < shape.size(); ++k) L.push_back("m " + target + "#" + std::to_string(k) + " " + std::to_string(shape[k]));
}
// a strided view of one source into a target: dims d, the target's strides ds from offset dst, the source's s from off
std::string view(const std::string& target, size_t n, const std::vector<long long>& d, long long dst, const std::vector<long long>& ds,
                 const std::string& from, long long off, const std::vector<long long>& s) {
  std::ostringstream o;
  o << "p " << target << " " << n << " v " << d.size() << dims(d) << " " << dst << dims(ds) << " " << from << " " << off << dims(s);
  return o.str();
}
}  // namespace

void loadBlobAsBundle(const std::string& dir, const std::function<bool(const std::string&, size_t)>& asHalf) {
  const std::map<std::string, std::vector<long long>> shapes = Model::bundleShapes(dir);
  auto shapeOf = [&](const std::string& n) -> const std::vector<long long>& {
    auto it = shapes.find(BLOB + n);
    if (it == shapes.end()) die("BioEmu2: %s is not af3-any-model's bioemu2 blob (no %s)", dir.c_str(), n.c_str());
    return it->second;
  };
  auto count = [](const std::vector<long long>& s) { size_t n = 1; for (long long d : s) n *= (size_t)d; return n; };
  const std::vector<std::string> stacks = {"sequence_embedder/msa_module/layers/", "sequence_embedder/pairmixer_module/layers/",
                                           "structure_module/layers/"};
  struct PairStack { std::string stack, tag, transition; };
  const std::vector<PairStack> pairStacks = {{"sequence_embedder/msa_module/layers/", "msa", "z_transition"},
                                             {"sequence_embedder/pairmixer_module/layers/", "pairmixer", "transition_z"}};
  std::vector<std::string> L;
  for (auto& [full, shape] : shapes) {
    if (full.compare(0, BLOB.size(), BLOB)) continue;
    const std::string r = full.substr(BLOB.size());
    std::string stack;
    for (auto& s : stacks) if (!r.compare(0, s.size(), s)) stack = s;
    if (stack.empty()) {                                   // (as it is)
      L.push_back("b w/" + r + " " + full + " 0 " + std::to_string(count(shape)));
      shapeLines(L, "w/" + r, shape);
      continue;
    }
    const std::string rest = r.substr(stack.size()), module = rest.substr(0, rest.find('/'));
    bool pair = false;
    for (auto& ps : pairStacks)
      if (ps.stack == stack && (module == "tri_mul_out" || module == "tri_mul_in" || module == ps.transition)) pair = true;
    if (pair) continue;                                    // (below)
    const std::vector<long long> inner(shape.begin() + 1, shape.end());
    const size_t per = count(inner);
    for (long long l = 0; l < shape[0]; ++l) {             // (unstacked: a layer is a slice)
      const std::string t = "w/" + stack + std::to_string(l) + "/" + rest;
      L.push_back("b " + t + " " + full + " " + std::to_string(l * per) + " " + std::to_string(per));
      shapeLines(L, t, inner);
    }
  }
  for (auto& ps : pairStacks) {
    const std::string S = BLOB + ps.stack;
    const std::vector<long long>& pin = shapeOf(ps.stack + "tri_mul_out/p_in/weight");    // [layers][2C][C]
    const long long layers = pin[0], C = pin[2];
    const long long I = shapeOf(ps.stack + ps.transition + "/fc1/weight")[1];              // [layers][I][C]
    for (long long l = 0; l < layers; ++l) {
      const std::string B = "w/" + ps.tag + "/" + std::to_string(l) + "/";
      auto slice = [&](const std::string& t, const std::string& from, long long n, const std::vector<long long>& shape) {
        L.push_back("b " + t + " " + S + from + " " + std::to_string(l * n) + " " + std::to_string(n));
        shapeLines(L, t, shape);
      };
      for (const char* m : {"tri_mul_out", "tri_mul_in"}) {
        const bool incoming = std::string(m) == "tri_mul_in";
        const std::string T = B + (incoming ? "triangleMultiplicationIncoming/" : "triangleMultiplicationOutgoing/");
        const std::string M_ = std::string(m) + "/";
        slice(T + "leftNormInputScale", M_ + "norm_in/weight", C, {C});
        slice(T + "leftNormInputOffset", M_ + "norm_in/bias", C, {C});
        slice(T + "centerNormScale", M_ + "norm_out/weight", C, {C});
        slice(T + "centerNormOffset", M_ + "norm_out/bias", C, {C});
        // (in, out) from torch's (out, in): t[k][o] = w[o][k]
        for (auto [to, from] : {std::pair<const char*, const char*>{"outputProjection", "p_out/weight"}, {"gatingLinear", "g_out/weight"}}) {
          L.push_back(view(T + to, (size_t)(C * C), {C, C}, 0, {C, 1}, S + M_ + from, l * C * C, {1, C}));
          shapeLines(L, T + to, {C, C});
        }
        // [C][2C], a and b interleaved by channel: column 2c a's channel c, 2c + 1 b's - from torch's [2C][C], [a | b]
        // blocked (the incoming triangle's halves swapped: its "a" is Boltz's b)
        const long long first = incoming ? C : 0, second = incoming ? 0 : C;
        for (auto [to, from] : {std::pair<const char*, const char*>{"projection", "p_in/weight"}, {"gate", "g_in/weight"}}) {
          const long long base = l * 2 * C * C;
          L.push_back(view(T + to, (size_t)(2 * C * C), {C, C}, 0, {2 * C, 2}, S + M_ + from, base + first * C, {1, C}));
          L.push_back(view(T + to, (size_t)(2 * C * C), {C, C}, 1, {2 * C, 2}, S + M_ + from, base + second * C, {1, C}));
          shapeLines(L, T + to, {C, 2 * C});
        }
      }
      const std::string T = B + "pairTransition/", M_ = ps.transition + "/";
      slice(T + "inputLayerNormScale", M_ + "norm/weight", C, {C});
      slice(T + "inputLayerNormOffset", M_ + "norm/bias", C, {C});
      // transition1 [C][2I] = [fc1 | fc2]^T (blocked: the gate, then the value); transition2 [I][C] = fc3^T
      L.push_back(view(T + "transition1", (size_t)(2 * I * C), {C, I}, 0, {2 * I, 1}, S + M_ + "fc1/weight", l * I * C, {1, C}));
      L.push_back(view(T + "transition1", (size_t)(2 * I * C), {C, I}, I, {2 * I, 1}, S + M_ + "fc2/weight", l * I * C, {1, C}));
      shapeLines(L, T + "transition1", {C, 2 * I});
      L.push_back(view(T + "transition2", (size_t)(I * C), {I, C}, 0, {C, 1}, S + M_ + "fc3/weight", l * C * I, {1, I}));
      shapeLines(L, T + "transition2", {I, C});
    }
  }
  M.loadBundleWalk(dir, L, "", asHalf);
}
