// AlphaFold 2's weights as cuda/af2 reads them, worked out from the page's bundle itself: every parameter of
// DeepMind's multimer graph (the names cuda/af2 reads, "w/<module>/<parameter>" - a monomer's converted onto that
// graph as alphafold3/af2/convert.py converts it) as a strided view of the bundle tensor the manifest's own parameter
// tables name for it. The bundle keeps DeepMind's monomer layout (tools/export_monomer_model.py,
// export_multimer_model.py), so a view is the identity but for a fixed few:
//   - the fused names (flat_params_to_haiku fuse=True): a triangle multiplication's layer_norm_input ->
//     left_norm_input and center_layer_norm -> center_norm, pair_activiations -> ~_relative_encoding/position_activations, affine_update -> quat_rigid/rigid;
//   - a triangle multiplication's projection and gate: the bundle's left and right halves, side by side;
//   - left_single, right_single and preprocess_1d: the 22-class target feature's last 21 rows;
//   - invariant point attention: q/kv scalar and point projections split per head (and kv into k and v) - and the
//     multimer checkpoint has no scalar biases, so those are zeros;
// then the structure module's residue tables (the bundle's geometry*, two of them integer) and three regime flags.
//
// 🔴 THIS REPLACED cuda/af2/maps/*.map, which were these lines written to a file by cuda/af2/make_map.py - which
// traced element ids through the reference converter and the page's exporter in JAX and Node. The format is the
// same (common.cuh's loadBundle reads it): `p <name> <length> v|i <rank> <dims> <dst> <dst strides> <bundle tensor>
// <offset> <strides>`, `z <name> <length>`, `m <name> <value>`, and `D <model>` first for a delta model.
#pragma once
#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "af3_weights.h"
#include "json.h"

namespace lf::weights {

inline std::string readText(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

struct Af2Part { std::vector<long long> dims, dstStride, srcStride; long long dst = 0; std::string src; long long off = 0; };
struct Af2Tensor { std::string module, param; std::vector<long long> shape; std::vector<Af2Part> parts; bool zeros = false; };

inline std::vector<long long> rowMajor(const std::vector<long long>& shape) {
  std::vector<long long> s(shape.size(), 1);
  for (int k = (int)shape.size() - 2; k >= 0; --k) s[k] = s[k + 1] * shape[k + 1];
  return s;
}

inline std::string replaceAll(std::string s, const std::string& from, const std::string& to) {
  for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) s.replace(at, from.size(), to);
  return s;
}

// every AF2 weight line for a page bundle: `manifest` its manifest.json, `shapes` the tensors it holds (a delta's
// `absent` ones already gone), `deltaModel` the delta's model name (a D line) or ""
inline std::vector<std::string> af2WeightLines(const Json& manifest, const Shapes& shapes, const std::string& deltaModel) {
  const Json* model = manifest.get("model");
  const Json* modelName = model ? model->get("name") : nullptr;
  bool multimer = modelName && modelName->isString() && modelName->s.find("multimer") != std::string::npos;
  std::vector<Af2Tensor> out;
  auto identity = [&](const std::string& key, const std::string& src) {
    Af2Tensor t;
    size_t cut = key.rfind('/');
    t.module = key.substr(0, cut); t.param = key.substr(cut + 1);
    t.shape = shapes.dims(src);
    Af2Part p;
    p.dims = t.shape; p.dstStride = rowMajor(t.shape); p.srcStride = p.dstStride; p.src = src;
    for (size_t k = 0; k < p.dims.size(); ++k) if (p.dims[k] == 1) p.srcStride[k] = 0;    // (make_map.py: a unit axis steps nowhere)
    t.parts.push_back(p);
    return t;
  };
  // the parameter tables: section -> {module path: {param: bundle tensor}} (nested for the confidence heads)
  const std::map<std::string, std::string> PREFIX = {{"evoformerStack", "evoformer/evoformer_iteration/"},
    {"extraMsaStack", "evoformer/extra_msa_stack/"}, {"embedding", "evoformer/"},
    {"templateEmbedding", multimer ? "evoformer/" : "evoformer/template_embedding/"}, {"templateSingle", "evoformer/"},
    {"structureModule", "structure_module/"}, {"confidenceHeads", ""}};
  struct Leaf { std::string section, path, tensor; };
  std::vector<Leaf> leaves;
  for (auto& entry : manifest.o) {
    const std::string section = entry.first;          // (named, not a structured binding: a lambda below captures it)
    const Json& body = entry.second;
    if (!body.isObject()) continue;
    const Json* params = body.get("parameters");
    if (!params || !params->isObject()) continue;
    if (!PREFIX.count(section)) throw std::runtime_error("an AF2 bundle section this port does not know: " + section);
    std::function<void(const Json&, const std::string&)> walk = [&](const Json& node, const std::string& path) {
      for (auto& [k, v] : node.o) {
        std::string at = path.empty() ? k : path + "/" + k;
        if (v.isString()) leaves.push_back({section, at, v.s});
        else if (v.isObject()) walk(v, at);
      }
    };
    walk(*params, "");
  }
  // the structure module's invariant point attention, split and reshaped (read off its shapes)
  std::map<std::string, std::string> ipa;          // leaf name -> bundle tensor
  for (auto& L : leaves)
    if (L.section == "structureModule" && L.path.find("invariant_point_attention/") != std::string::npos) ipa[L.path.substr(L.path.rfind("invariant_point_attention/") + 26)] = L.tensor;
  for (auto& L : leaves) {
    if (!shapes.has(L.tensor)) continue;            // (a delta model's absent tensors: a template-free model's template embedder)
    std::string path = L.path;
    if (L.section == "confidenceHeads") {
      path = replaceAll(replaceAll(path, "predictedLddt/", "predicted_lddt_head/"), "predictedAlignedError/", "predicted_aligned_error_head/");
    }
    std::string key = PREFIX.at(L.section) + path;
    if (key.find("triangle_multiplication_") != std::string::npos) {     // (the triangle multiplications' own names)
      key = replaceAll(key, "/layer_norm_input/", "/left_norm_input/");
      key = replaceAll(key, "/center_layer_norm/", "/center_norm/");
    }
    if (L.section == "embedding") key = replaceAll(key, "evoformer/pair_activiations/", "evoformer/~_relative_encoding/position_activations/");
    key = replaceAll(key, "/affine_update/", "/quat_rigid/rigid/");
    // the triangle multiplications' fused projection and gate: left half, then right
    bool left = key.find("/left_projection/") != std::string::npos || key.find("/left_gate/") != std::string::npos;
    bool right = key.find("/right_projection/") != std::string::npos || key.find("/right_gate/") != std::string::npos;
    if (right && key.find("triangle_multiplication_") != std::string::npos) continue;     // (placed with its left half)
    if (left && key.find("triangle_multiplication_") != std::string::npos) {
      std::string fused = replaceAll(replaceAll(key, "/left_projection/", "/projection/"), "/left_gate/", "/gate/");
      std::string rightPath = replaceAll(replaceAll(L.path, "/left_projection/", "/right_projection/"), "/left_gate/", "/right_gate/");
      std::string rightTensor;
      for (auto& R : leaves) if (R.section == L.section && R.path == rightPath) rightTensor = R.tensor;
      if (rightTensor.empty()) throw std::runtime_error(L.path + " has no right half in the bundle");
      Af2Tensor t;
      size_t cut = fused.rfind('/');
      t.module = "w/" + fused.substr(0, cut); t.param = fused.substr(cut + 1);
      auto half = shapes.dims(L.tensor);
      t.shape = half; t.shape.back() *= 2;
      for (int side = 0; side < 2; ++side) {
        Af2Part p;
        p.dims = half; p.dstStride = rowMajor(t.shape); p.srcStride = rowMajor(half); p.dst = side ? half.back() : 0;
        p.src = side ? rightTensor : L.tensor;
        t.parts.push_back(p);
      }
      out.push_back(t);
      continue;
    }
    if (L.path.find("invariant_point_attention/") != std::string::npos) {
      std::string leaf = L.path.substr(L.path.rfind("invariant_point_attention/") + 26);
      std::string base = PREFIX.at("structureModule") + L.path.substr(0, L.path.rfind("invariant_point_attention/") + 26);
      long long H = shapes.dims(ipa.at("trainable_point_weights"))[0];
      long long Pq = shapes.size(ipa.at("q_point_local/bias")) / (3 * H), Pkv = shapes.size(ipa.at("kv_point_local/bias")) / (3 * H);
      long long S = shapes.size(ipa.at("q_scalar/bias")) / H, C = shapes.dims(ipa.at("q_scalar/weights"))[0];
      // (`dims` the view's axes; the tensor's own shape folds a point projection's last two, [3, points], into one)
      auto view = [&](const std::string& name, std::vector<long long> dims, long long off, std::vector<long long> srcStride) {
        Af2Tensor t;
        std::string k = "w/" + base + name;
        size_t cut = k.rfind('/');
        t.module = k.substr(0, cut); t.param = k.substr(cut + 1); t.shape = dims;
        if (name.find("point_projection") != std::string::npos) { t.shape.pop_back(); t.shape.back() = 3 * dims.back(); }
        Af2Part p; p.dims = dims; p.dstStride = rowMajor(dims); p.src = L.tensor; p.off = off; p.srcStride = srcStride;
        t.parts.push_back(p);
        out.push_back(t);
      };
      auto zeros = [&](const std::string& name, std::vector<long long> shape) {
        Af2Tensor t;
        std::string k = "w/" + base + name;
        size_t cut = k.rfind('/');
        t.module = k.substr(0, cut); t.param = k.substr(cut + 1); t.shape = shape; t.zeros = true;
        out.push_back(t);
      };
      if (leaf == "kv_point_local/bias") {
        view("k_point_projection/point_projection/bias", {H, 3, Pq}, 0, {Pkv, H * Pkv, 1});
        view("v_point_projection/point_projection/bias", {H, 3, Pkv - Pq}, Pq, {Pkv, H * Pkv, 1});
      } else if (leaf == "kv_point_local/weights") {
        view("k_point_projection/point_projection/weights", {C, H, 3, Pq}, 0, {3 * H * Pkv, Pkv, H * Pkv, 1});
        view("v_point_projection/point_projection/weights", {C, H, 3, Pkv - Pq}, Pq, {3 * H * Pkv, Pkv, H * Pkv, 1});
      } else if (leaf == "q_point_local/bias") {
        view("q_point_projection/point_projection/bias", {H, 3, Pq}, 0, {Pq, H * Pq, 1});
      } else if (leaf == "q_point_local/weights") {
        view("q_point_projection/point_projection/weights", {C, H, 3, Pq}, 0, {3 * H * Pq, Pq, H * Pq, 1});
      } else if (leaf == "kv_scalar/bias") {
        if (multimer) { zeros("k_scalar_projection/bias", {H, S}); zeros("v_scalar_projection/bias", {H, S}); }
        else { view("k_scalar_projection/bias", {H, S}, 0, {2 * S, 1}); view("v_scalar_projection/bias", {H, S}, S, {2 * S, 1}); }
      } else if (leaf == "kv_scalar/weights") {
        view("k_scalar_projection/weights", {C, H, S}, 0, {2 * H * S, 2 * S, 1});
        view("v_scalar_projection/weights", {C, H, S}, S, {2 * H * S, 2 * S, 1});
      } else if (leaf == "q_scalar/bias") {
        if (multimer) zeros("q_scalar_projection/bias", {H, S});
        else view("q_scalar_projection/bias", {H, S}, 0, {S, 1});
      } else if (leaf == "q_scalar/weights") {
        view("q_scalar_projection/weights", {C, H, S}, 0, {H * S, S, 1});
      } else {
        out.push_back(identity("w/" + key, L.tensor));
      }
      continue;
    }
    // the 22-class target feature's projections read its last 21 rows
    if (L.section == "embedding" && (path == "left_single/weights" || path == "right_single/weights" || path == "preprocess_1d/weights")
        && shapes.dims(L.tensor)[0] == 22) {
      Af2Tensor t = identity("w/" + key, L.tensor);
      long long width = t.shape[1];
      t.shape[0] = 21;
      t.parts[0].dims = t.shape; t.parts[0].off = width;
      out.push_back(t);
      continue;
    }
    // a monomer's relative-position projection onto the multimer graph's 73 features (2 * 32 + 2 residue offsets,
    // same-entity, 2 * 2 + 2 chain offsets): its 65 residue-offset rows, the rest zero (alphafold3/af2/convert.py)
    if (!multimer && L.section == "embedding" && path == "pair_activiations/weights") {
      Af2Tensor t = identity("w/" + key, L.tensor);
      t.shape[0] = 73;
      out.push_back(t);
      continue;
    }
    out.push_back(identity("w/" + key, L.tensor));
  }
  // the distogram head (cuda/af2 reads it for an oracle; only the monomer's bundle carries it)
  if (shapes.has("af2DistogramHalfLogitsWeights")) {
    out.push_back(identity("w/distogram_head/half_logits/weights", "af2DistogramHalfLogitsWeights"));
    out.push_back(identity("w/distogram_head/half_logits/bias", "af2DistogramHalfLogitsBias"));
  }
  // export_weights.py's order: by module, then parameter
  std::stable_sort(out.begin(), out.end(), [](const Af2Tensor& a, const Af2Tensor& b) {
    return std::tie(a.module, a.param) < std::tie(b.module, b.param);
  });
  std::vector<std::string> lines;
  if (!deltaModel.empty()) lines.push_back("D " + deltaModel);
  auto join = [](const std::vector<long long>& v) { std::string s; for (size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]); return s; };
  for (auto& t : out) {
    std::string key = t.module + "/" + t.param;
    long long n = 1;
    for (long long d : t.shape) n *= d;
    if (t.zeros) lines.push_back("z " + key + " " + std::to_string(n));
    for (auto& p : t.parts)
      lines.push_back("p " + key + " " + std::to_string(n) + " v " + std::to_string(p.dims.size()) + " " + join(p.dims) + " " + std::to_string(p.dst) + " "
                      + join(p.dstStride) + " " + p.src + " " + std::to_string(p.off) + " " + join(p.srcStride));
    lines.push_back("m " + key + "#r " + std::to_string(t.shape.size()));
    for (size_t k = 0; k < t.shape.size(); ++k) lines.push_back("m " + key + "#" + std::to_string(k) + " " + std::to_string(t.shape[k]));
  }
  // the structure module's residue tables: the bundle's own (float32 there), two of them integer
  for (auto& [name, tensor, op] : std::vector<std::tuple<const char*, const char*, const char*>>{
         {"rigid_group_default_frame", "geometryDefaultFrames", "v"}, {"atom14_to_rigid_group", "geometryAtom14ToGroup", "i"},
         {"atom14_rigid_group_positions", "geometryAtom14Positions", "v"}, {"atom14_mask", "geometryAtom14Mask", "v"},
         {"atom37_to_atom14", "geometryAtom37ToAtom14", "i"}, {"atom37_mask", "geometryAtom37Mask", "v"}}) {
    std::string n = std::to_string(shapes.size(tensor));
    lines.push_back(std::string("p c/") + name + " " + n + " " + op + " 1 " + n + " 0 1 " + tensor + " 0 1");
  }
  lines.push_back(std::string("m meta/multimer ") + (multimer ? "1" : "0"));
  lines.push_back(std::string("m meta/position_scale ") + (multimer ? "20.0" : "10.0"));
  lines.push_back(std::string("m meta/opm_first ") + (multimer ? "1" : "0"));
  return lines;
}

}  // namespace lf::weights
