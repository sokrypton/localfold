// The AF3-lineage weights as cuda/af3 reads them, worked out from the bundle itself: shared/af3/weights/weights.js
// and diffusion-weights.js (trunkWeights, diffusionWeights, confidenceWeights, targetFeatureWeights, atomReference,
// OpenDDE's expander, refiner and confidence head), walked in the order cuda/af3/export-model.mjs walks their
// objects - so each native tensor gets the bundle tensor (or the slice of a stacked one) the page's loader reads
// for it, the zeros and ones the loader makes, the per-block LayerNorm scales it folds into their projections,
// every width it reads off a shape, and the family's dialect (dialects.inc, generated from shared/af3/dialect.js).
//
// 🔴 THIS REPLACED cuda/af3/maps/*.map, which were these lines written to a file by cuda/make_map.mjs from a
// float32 export - Node at every re-export, and a file per model to keep in step. The lines are the same format
// (common.cuh's loadBundle reads them): `m <name> <value>`, `b <name> <bundle tensor> <first element> <length>`,
// `z <name> <length>`, `p <name> <length> o|x ...` (ones; a scale folded into a projection) and `c <name> <values>`.
// Only names and shapes are read here - an af3-any-model blob and a LocalFold bundle name their tensors alike.
#pragma once
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "jsnum.h"
#include "tables.h"

namespace lf::weights {

struct DialectTable {
  const char* name;
  std::vector<std::pair<const char*, const char*>> walk;     // the `trunk.dialect.*` metadata lines, in order
  std::vector<std::pair<const char*, const char*>> values;   // every field: "true", "false", "null", a number, ...
};
#include "dialects.inc"

struct Dialect {
  const DialectTable* t = nullptr;
  std::string value(const std::string& key) const {
    for (auto& [k, v] : t->values) if (key == k) return v;
    return "";                                                // undefined
  }
  bool is(const std::string& key) const { return value(key) == "true"; }   // dialect.key === true
  bool defined(const std::string& key) const { std::string v = value(key); return !v.empty() && v != "null"; }
};
inline Dialect dialectNamed(std::string model) {
  for (auto& [alias, name] : familyAliases()) if (alias == model) { model = name; break; }
  for (auto& d : dialectTables()) if (model == d.name) return Dialect{&d};
  std::string known;
  for (auto& d : dialectTables()) known += (known.empty() ? "" : ", ") + std::string(d.name);
  throw std::runtime_error("no AF3 dialect for model \"" + model + "\"; known: " + known);
}

// a bundle's tensors: name -> shape
struct Shapes {
  std::map<std::string, std::vector<long long>> shape;
  bool has(const std::string& n) const { return shape.count(n) > 0; }
  const std::vector<long long>& dims(const std::string& n) const {
    auto it = shape.find(n);
    if (it == shape.end() || it->second.empty()) throw std::runtime_error("no shape for " + n + "; a width cannot be derived from it");
    return it->second;
  }
  long long size(const std::string& n) const {           // (a scalar's shape is empty: one element)
    auto it = shape.find(n);
    if (it == shape.end()) throw std::runtime_error("this bundle has no " + n);
    long long s = 1;
    for (long long d : it->second) s *= d;
    return s;
  }
};

// the lines, in the walk's order
struct Walk {
  const Shapes& S;
  std::vector<std::string> lines;
  explicit Walk(const Shapes& s) : S(s) {}
  void num(const std::string& p, double v) { lines.push_back("m " + p + " " + jsNumber(v)); }
  void flag(const std::string& p, bool v) { lines.push_back("m " + p + (v ? " 1" : " 0")); }
  void whole(const std::string& p, const std::string& src) {        // store.tensor(src)
    lines.push_back("b " + p + " " + src + " 0 " + std::to_string(S.size(src)));
  }
  void stacked(const std::string& p, const std::string& src, long long index, int dims = 1) {   // stacked(store, src, index)
    const auto& shape = S.dims(src);
    long long count = dims == 2 ? shape[0] * shape[1] : shape[0];
    if (index >= count) throw std::runtime_error(src + " has " + std::to_string(count) + " blocks; asked for " + std::to_string(index));
    long long stride = 1;
    for (size_t k = dims; k < shape.size(); ++k) stride *= shape[k];
    lines.push_back("b " + p + " " + src + " " + std::to_string(index * stride) + " " + std::to_string(stride));
  }
  void stackedIfPresent(const std::string& p, const std::string& src, long long index, int dims = 1) {
    if (S.has(src)) stacked(p, src, index, dims);
  }
  void wholeIfPresent(const std::string& p, const std::string& src) { if (S.has(src)) whole(p, src); }
  void zeros(const std::string& p, long long n) { lines.push_back("z " + p + " " + std::to_string(n)); }
  void ones(const std::string& p, long long n) { lines.push_back("p " + p + " " + std::to_string(n) + " o 1 " + std::to_string(n) + " 0 1"); }
  // a per-block LayerNorm scale folded into its projection, one block's [C, H]: proj[c, h] * scale[c]
  void fold2(const std::string& p, long long C, long long H, const std::string& scale, long long sOff, const std::string& proj, long long pOff) {
    lines.push_back("p " + p + " " + std::to_string(C * H) + " x 2 " + std::to_string(C) + " " + std::to_string(H) + " 0 " + std::to_string(H)
                    + " 1 " + scale + " " + std::to_string(sOff) + " 1 0 " + proj + " " + std::to_string(pOff) + " " + std::to_string(H) + " 1");
  }
  // ...and B blocks of it packed [C, B, H]
  void fold3(const std::string& p, long long C, long long B, long long H, const std::string& scale, long long sOff, const std::string& proj,
             long long pOff) {
    lines.push_back("p " + p + " " + std::to_string(C * B * H) + " x 3 " + std::to_string(C) + " " + std::to_string(B) + " " + std::to_string(H)
                    + " 0 " + std::to_string(B * H) + " " + std::to_string(H) + " 1 " + scale + " " + std::to_string(sOff) + " 1 "
                    + std::to_string(C) + " 0 " + proj + " " + std::to_string(pOff) + " " + std::to_string(H) + " " + std::to_string(C * H) + " 1");
  }
  void raw(const std::string& line) { lines.push_back(line); }
};

const std::string EVO = "diffuser/evoformer";
const std::string MSA_STACK = EVO + "/__layer_stack_no_per_layer/msa_stack";
const std::string PAIRFORMER = EVO + "/__layer_stack_no_per_layer_1/trunk_pairformer";
const std::string TEMPLATE = EVO + "/template_embedding";
const std::string TEMPLATE_SINGLE = TEMPLATE + "/single_template_embedding";
const std::string CONFIDENCE = "diffuser/confidence_head";
const std::string CONFIDENCE_STACK = CONFIDENCE + "/__layer_stack_no_per_layer/confidence_pairformer";
const std::string TEMPLATE_STACK = TEMPLATE_SINGLE + "/__layer_stack_no_per_layer/template_embedding_iteration";
const std::string TEMPLATE_FUSED_STACK = TEMPLATE + "/__layer_stack_no_per_layer/tmpl_pairformer";
const std::string HEAD = "diffuser/~/diffusion_head";
const std::string ENCODER = HEAD + "/diffusion_atom_transformer_encoder";
const std::string DECODER = HEAD + "/diffusion_atom_transformer_decoder";
const std::string TX = HEAD + "/transformer";
const std::string STRUCTURAL_EXPANDER = "diffuser/structural_token_expander";
const std::string STRUCTURAL_REFINER = "diffuser/structural_token_refiner/trunk_pairformer";
const std::string OPENDDE_CONFIDENCE_STACK = CONFIDENCE + "/pairformer_stack/trunk_pairformer";

// stock AlphaFold 3's Fourier noise embedding: a constant of its source, frozen from a fixed seed, which DeepMind's
// af3.bin.zst does not carry - as the 9 significant digits that round-trip each float32 (cuda/make_map.mjs wrote it so)
#include "af3_fourier.inc"

// ---------------------------------------------------------------- weights.js
inline void pairTrack(Walk& w, const std::string& p, const std::string& root, long long index) {
  auto at = [&](const std::string& path, const std::string& leaf) { w.stacked(p + path, root + "/" + leaf, index); };
  const auto& q = w.S.dims(root + "/pair_attention1/q_projection/weights");
  long long gridHeads = q[1], gridDimension = q[2];
  for (std::string direction : {"outgoing", "incoming"}) {
    std::string t = std::string(".triangleMultiplication") + (direction == "outgoing" ? "Outgoing" : "Incoming") + ".";
    std::string s = "triangle_multiplication_" + direction + "/";
    at(t + "leftNormInputScale", s + "left_norm_input/scale");
    at(t + "leftNormInputOffset", s + "left_norm_input/offset");
    at(t + "projection", s + "projection/weights");
    at(t + "gate", s + "gate/weights");
    at(t + "centerNormScale", s + "center_norm/scale");
    at(t + "centerNormOffset", s + "center_norm/offset");
    at(t + "outputProjection", s + "output_projection/weights");
    at(t + "gatingLinear", s + "gating_linear/weights");
  }
  for (std::string which : {"1", "2"}) {
    std::string g = ".pairAttention" + which + ".", s = "pair_attention" + which + "/";
    w.num(p + g + "heads", gridHeads);
    w.num(p + g + "dimension", gridDimension);
    at(g + "actNormScale", s + "act_norm/scale");
    at(g + "actNormOffset", s + "act_norm/offset");
    at(g + "pairBiasProjection", s + "pair_bias_projection/weights");
    at(g + "qProjection", s + "q_projection/weights");
    at(g + "kProjection", s + "k_projection/weights");
    at(g + "vProjection", s + "v_projection/weights");
    at(g + "gatingQuery", s + "gating_query/weights");
    at(g + "outputProjection", s + "output_projection/weights");
    if (w.S.has(root + "/" + s + "gating_query/bias")) at(g + "gatingQueryBias", s + "gating_query/bias");
    if (w.S.has(root + "/" + s + "output_projection/bias")) at(g + "outputProjectionBias", s + "output_projection/bias");
    if (w.S.has(root + "/" + s + "output_projection_transposed/weights")) at(g + "outputProjectionTransposed", s + "output_projection_transposed/weights");
  }
  at(".pairTransition.inputLayerNormScale", "pair_transition/input_layer_norm/scale");
  at(".pairTransition.inputLayerNormOffset", "pair_transition/input_layer_norm/offset");
  at(".pairTransition.transition1", "pair_transition/transition1/weights");
  at(".pairTransition.transition2", "pair_transition/transition2/weights");
}

inline void embedderWeights(Walk& w, const std::string& p) {
  auto T = [&](const std::string& path, const std::string& name) { w.whole(p + path, EVO + "/" + name); };
  const auto& rel = w.S.dims(EVO + "/~_relative_encoding/position_activations/weights");
  const auto& single = w.S.dims(EVO + "/single_activations/weights");
  const auto& tf = w.S.dims(EVO + "/extra_msa_target_feat/weights");
  w.num(p + ".pairChannels", rel[1]); w.num(p + ".singleChannels", single[1]); w.num(p + ".msaChannels", tf[1]);
  w.num(p + ".targetFeatWidth", tf[0]); w.num(p + ".relativeWidth", rel[0]);
  T(".leftSingle", "left_single/weights"); T(".rightSingle", "right_single/weights");
  T(".prevEmbeddingNormScale", "prev_embedding_layer_norm/scale"); T(".prevEmbeddingNormOffset", "prev_embedding_layer_norm/offset");
  T(".prevEmbedding", "prev_embedding/weights"); T(".positionActivations", "~_relative_encoding/position_activations/weights");
  if (w.S.has(EVO + "/bond_embedding/weights")) T(".bondEmbedding", "bond_embedding/weights");
  if (w.S.has(EVO + "/~_relative_encoding/position_activations/bias")) {
    T(".positionActivationsBias", "~_relative_encoding/position_activations/bias");
    T(".msaActivationsBias", "msa_activations/bias");
  }
  if (w.S.has(EVO + "/token_bonds_type_embed/weights")) {
    T(".tokenBondsTypeEmbed", "token_bonds_type_embed/weights");
    T(".contactEncodingUnspecified", "contact_encoding_unspecified");
    T(".contactEncodingUnselected", "contact_encoding_unselected");
  }
  T(".msaActivations", "msa_activations/weights"); T(".extraMsaTargetFeat", "extra_msa_target_feat/weights");
  T(".singleActivations", "single_activations/weights");
  T(".prevSingleEmbeddingNormScale", "prev_single_embedding_layer_norm/scale");
  T(".prevSingleEmbeddingNormOffset", "prev_single_embedding_layer_norm/offset");
  T(".prevSingleEmbedding", "prev_single_embedding/weights");
}

inline void templateWeights(Walk& w, const std::string& p, const Dialect& d) {
  bool hasFused = w.S.has(TEMPLATE + "/a_proj/weights");
  bool fused = d.defined("fusedTemplateEmbedder") ? d.is("fusedTemplateEmbedder") : hasFused;
  if (d.defined("fusedTemplateEmbedder") && hasFused != fused)
    throw std::runtime_error(std::string("this bundle ") + (hasFused ? "carries" : "does not carry") + " template a_proj and its dialect says otherwise");
  auto T = [&](const std::string& path, const std::string& name) { w.whole(p + path, name); };
  if (fused) {
    w.flag(p + ".fused", true);
    w.num(p + ".queryChannels", w.S.dims(TEMPLATE + "/z_norm/scale")[0]);
    w.num(p + ".featureWidth", w.S.dims(TEMPLATE + "/a_proj/weights")[0]);
    w.num(p + ".channels", w.S.dims(TEMPLATE + "/v_norm/scale")[0]);
    for (int b = 0; b < 2; ++b) pairTrack(w, p + ".blocks." + std::to_string(b), TEMPLATE_FUSED_STACK, b);
    T(".queryEmbeddingNormScale", TEMPLATE + "/z_norm/scale"); T(".queryEmbeddingNormOffset", TEMPLATE + "/z_norm/offset");
    T(".zProjection", TEMPLATE + "/z_proj/weights"); T(".aProjection", TEMPLATE + "/a_proj/weights");
    T(".outputLayerNormScale", TEMPLATE + "/v_norm/scale"); T(".outputLayerNormOffset", TEMPLATE + "/v_norm/offset");
    T(".outputLinear", TEMPLATE + "/u_proj/weights");
    return;
  }
  w.flag(p + ".fused", false);
  w.num(p + ".queryChannels", w.S.dims(TEMPLATE_SINGLE + "/query_embedding_norm/scale")[0]);
  w.num(p + ".channels", w.S.dims(TEMPLATE_SINGLE + "/output_layer_norm/scale")[0]);
  for (int b = 0; b < 2; ++b) pairTrack(w, p + ".blocks." + std::to_string(b), TEMPLATE_STACK, b);
  T(".queryEmbeddingNormScale", TEMPLATE_SINGLE + "/query_embedding_norm/scale");
  T(".queryEmbeddingNormOffset", TEMPLATE_SINGLE + "/query_embedding_norm/offset");
  for (int k : {8, 2, 3, 0, 1, 4, 5, 6, 7})
    T(".templatePairEmbedding" + std::to_string(k), TEMPLATE_SINGLE + "/template_pair_embedding_" + std::to_string(k) + "/weights");
  T(".outputLayerNormScale", TEMPLATE_SINGLE + "/output_layer_norm/scale");
  T(".outputLayerNormOffset", TEMPLATE_SINGLE + "/output_layer_norm/offset");
  T(".outputLinear", TEMPLATE + "/output_linear/weights");
  if (w.S.has(TEMPLATE_SINGLE + "/template_feature_bias")) T(".templateFeatureBias", TEMPLATE_SINGLE + "/template_feature_bias");
}

inline void msaBlockWeights(Walk& w, const std::string& p, long long index) {
  auto at = [&](const std::string& path, const std::string& leaf) { w.stacked(p + path, MSA_STACK + "/" + leaf, index); };
  auto has = [&](const std::string& leaf) { return w.S.has(MSA_STACK + "/" + leaf); };
  const auto& v = w.S.dims(MSA_STACK + "/msa_attention1/v_projection/weights");
  long long outerChannels = w.S.dims(MSA_STACK + "/outer_product_mean/left_projection/weights")[2];
  w.num(p + ".pairChannels", w.S.dims(MSA_STACK + "/msa_attention1/pair_logits/weights")[1]);
  w.num(p + ".msaChannels", v[1]);
  pairTrack(w, p, MSA_STACK, index);
  std::string o = ".outerProductMean.";
  w.num(p + o + "outerChannels", outerChannels);
  at(o + "layerNormInputScale", "outer_product_mean/layer_norm_input/scale");
  at(o + "layerNormInputOffset", "outer_product_mean/layer_norm_input/offset");
  at(o + "leftProjection", "outer_product_mean/left_projection/weights");
  at(o + "rightProjection", "outer_product_mean/right_projection/weights");
  if (has("outer_product_mean/left_projection/bias")) at(o + "leftProjectionBias", "outer_product_mean/left_projection/bias");
  if (has("outer_product_mean/right_projection/bias")) at(o + "rightProjectionBias", "outer_product_mean/right_projection/bias");
  at(o + "outputW", "outer_product_mean/output_w");
  at(o + "outputB", "outer_product_mean/output_b");
  if (has("outer_product_mean/product_norm/scale")) {
    at(o + "productNormScale", "outer_product_mean/product_norm/scale");
    at(o + "productNormOffset", "outer_product_mean/product_norm/offset");
    w.num(p + o + "groups", (double)(outerChannels * outerChannels) / w.S.dims(MSA_STACK + "/outer_product_mean/product_norm/scale")[1]);
  }
  std::string a = ".msaAttention1.";
  w.num(p + a + "heads", v[2]); w.num(p + a + "dimension", v[3]);
  at(a + "actNormScale", "msa_attention1/act_norm/scale"); at(a + "actNormOffset", "msa_attention1/act_norm/offset");
  at(a + "pairNormScale", "msa_attention1/pair_norm/scale"); at(a + "pairNormOffset", "msa_attention1/pair_norm/offset");
  at(a + "pairLogits", "msa_attention1/pair_logits/weights"); at(a + "vProjection", "msa_attention1/v_projection/weights");
  at(a + "gatingQuery", "msa_attention1/gating_query/weights"); at(a + "outputProjection", "msa_attention1/output_projection/weights");
  std::string t = ".msaTransition.";
  at(t + "inputLayerNormScale", "msa_transition/input_layer_norm/scale"); at(t + "inputLayerNormOffset", "msa_transition/input_layer_norm/offset");
  at(t + "transition1", "msa_transition/transition1/weights"); at(t + "transition2", "msa_transition/transition2/weights");
}

// pairformerBlockWeights (the trunk's, the confidence head's, OpenDDE's refiner and confidence stacks)
inline void pairformerBlockWeights(Walk& w, const std::string& p, long long index, const std::string& root) {
  auto at = [&](const std::string& path, const std::string& leaf) { w.stacked(p + path, root + "/" + leaf, index); };
  const auto& q = w.S.dims(root + "/single_attention_q_projection/weights");
  long long pairChannels = w.S.dims(root + "/single_pair_logits_norm/scale")[1];
  w.num(p + ".pairChannels", pairChannels); w.num(p + ".singleChannels", q[1]);
  pairTrack(w, p, root, index);
  at(".singlePairLogitsNormScale", "single_pair_logits_norm/scale");
  at(".singlePairLogitsNormOffset", "single_pair_logits_norm/offset");
  at(".singlePairLogitsProjection", "single_pair_logits_projection/weights");
  std::string a = ".singleAttention.";
  w.num(p + a + "heads", q[2]); w.num(p + a + "dimension", q[3]);
  at(a + "layerNormScale", "single_attention_layer_norm/scale"); at(a + "layerNormOffset", "single_attention_layer_norm/offset");
  at(a + "qProjection", "single_attention_q_projection/weights"); at(a + "qBias", "single_attention_q_projection/bias");
  at(a + "kProjection", "single_attention_k_projection/weights"); at(a + "vProjection", "single_attention_v_projection/weights");
  at(a + "gatingQuery", "single_attention_gating_query/weights"); at(a + "outputProjection", "single_attention_transition2/weights");
  std::string t = ".singleTransition.";
  at(t + "inputLayerNormScale", "single_transition/input_layer_norm/scale");
  at(t + "inputLayerNormOffset", "single_transition/input_layer_norm/offset");
  at(t + "transition1", "single_transition/transition1/weights"); at(t + "transition2", "single_transition/transition2/weights");
}

inline void distogramWeights(Walk& w, const std::string& p, const Dialect& d) {
  std::string name = "diffuser/distogram_head/half_logits/weights", biasName = "diffuser/distogram_head/half_logits/bias";
  const auto& shape = w.S.dims(name);
  bool present = w.S.has(biasName);
  if (!d.defined("distogramBias"))
    throw std::runtime_error("dialect.distogramBias has no default: stock AF3 trains no bias on half_logits and OpenDDE does");
  if (present != d.is("distogramBias"))
    throw std::runtime_error(std::string("this bundle ") + (present ? "carries" : "does not carry") + " " + biasName + ", and its dialect says "
                             + d.value("distogramBias") + "; one of the two is wrong and the fold would be silently different either way");
  bool mlp = d.is("mlpDistogram");
  w.whole(p + ".halfLogits", name);
  w.num(p + ".pairChannels", mlp ? w.S.dims("diffuser/distogram_head/hidden/weights")[0] : shape[0]);
  w.num(p + ".bins", shape[1]);
  if (present) w.whole(p + ".halfLogitsBias", biasName);
  if (mlp) {
    w.whole(p + ".inputLayerNormScale", "diffuser/distogram_head/input_layer_norm/scale");
    w.whole(p + ".inputLayerNormOffset", "diffuser/distogram_head/input_layer_norm/offset");
    w.whole(p + ".hidden", "diffuser/distogram_head/hidden/weights");
    w.whole(p + ".hiddenBias", "diffuser/distogram_head/hidden/bias");
    w.num(p + ".hiddenWidth", shape[0]);
  }
}

inline void trunkWeights(Walk& w, const Dialect& d, bool& dialectWritten) {
  long long pairformerBlocks = w.S.dims(PAIRFORMER + "/single_attention_q_projection/bias")[0];
  long long msaBlocks = w.S.dims(MSA_STACK + "/outer_product_mean/output_b")[0];
  for (auto& [k, v] : d.t->walk) w.raw(std::string("m trunk.dialect.") + k + " " + v);
  dialectWritten = true;
  embedderWeights(w, "trunk.embedder");
  templateWeights(w, "trunk.template", d);
  for (long long i = 0; i < msaBlocks; ++i) msaBlockWeights(w, "trunk.msaBlocks." + std::to_string(i), i);
  for (long long i = 0; i < pairformerBlocks; ++i) pairformerBlockWeights(w, "trunk.pairformerBlocks." + std::to_string(i), i, PAIRFORMER);
  distogramWeights(w, "trunk.distogram", d);
}

inline void confidenceWeights(Walk& w, const std::string& p) {
  auto T = [&](const std::string& path, const std::string& name) { w.whole(p + path, CONFIDENCE + "/" + name); };
  auto has = [&](const std::string& name) { return w.S.has(CONFIDENCE + "/" + name); };
  bool reembedScope = has("~_boltz2_reembed/left_target_feat_project/weights");
  const auto& tf = w.S.dims(CONFIDENCE + (reembedScope ? "/~_boltz2_reembed/left_target_feat_project/weights" : "/~_embed_features/left_target_feat_project/weights"));
  long long singleChannels = w.S.dims(CONFIDENCE + (reembedScope ? "/~_boltz2_reembed/s_norm/scale" : "/plddt_logits_ln/scale"))[0];
  long long stackBlocks = w.S.dims(CONFIDENCE_STACK + "/single_attention_q_projection/bias")[0];
  w.num(p + ".pairChannels", tf[1]); w.num(p + ".singleChannels", singleChannels); w.num(p + ".targetFeatWidth", tf[0]);
  for (long long i = 0; i < stackBlocks; ++i) pairformerBlockWeights(w, p + ".blocks." + std::to_string(i), i, CONFIDENCE_STACK);
  if (has("~_boltz2_reembed/z_norm/scale")) {
    std::string r = ".reembed.";
    for (auto& [path, name] : std::vector<std::pair<std::string, std::string>>{
           {"sInputsNormScale", "s_inputs_norm/scale"}, {"sInputsNormOffset", "s_inputs_norm/offset"}, {"sNormScale", "s_norm/scale"},
           {"sNormOffset", "s_norm/offset"}, {"sInputToS", "s_input_to_s/weights"}, {"zNormScale", "z_norm/scale"}, {"zNormOffset", "z_norm/offset"},
           {"relPosProject", "rel_pos_project/weights"}, {"tokenBondsProject", "token_bonds_project/weights"},
           {"tokenBondsTypeEmbed", "token_bonds_type_embed/weights"}})
      T(r + path, "~_boltz2_reembed/" + name);
    T(r + "contactEncodingUnspecified", "contact_encoding_unspecified");
    T(r + "contactEncodingUnselected", "contact_encoding_unselected");
    for (auto& [path, name] : std::vector<std::pair<std::string, std::string>>{
           {"leftTargetFeatProject", "left_target_feat_project/weights"}, {"rightTargetFeatProject", "right_target_feat_project/weights"},
           {"sToZProdIn1", "s_to_z_prod_in1/weights"}, {"sToZProdIn2", "s_to_z_prod_in2/weights"}, {"sToZProdOut", "s_to_z_prod_out/weights"},
           {"distogramFeatProject", "distogram_feat_project/weights"}})
      T(r + path, "~_boltz2_reembed/" + name);
  } else {
    T(".leftTargetFeatProject", "~_embed_features/left_target_feat_project/weights");
    T(".rightTargetFeatProject", "~_embed_features/right_target_feat_project/weights");
    T(".distogramFeatProject", "~_embed_features/distogram_feat_project/weights");
    if (has("~_embed_features/distance_feat_project/weights")) T(".distanceFeatProject", "~_embed_features/distance_feat_project/weights");
  }
  if (has("input_single_norm/scale")) { T(".inputSingleNormScale", "input_single_norm/scale"); T(".inputSingleNormOffset", "input_single_norm/offset"); }
  if (has("logits_ln/scale")) {
    T(".logitsLnScale", "logits_ln/scale"); T(".logitsLnOffset", "logits_ln/offset");
    T(".paeLogitsLnScale", "pae_logits_ln/scale"); T(".paeLogitsLnOffset", "pae_logits_ln/offset");
    T(".plddtLnScale", "plddt_logits_ln/scale"); T(".plddtLnOffset", "plddt_logits_ln/offset");
    if (has("experimentally_resolved_ln/scale")) {
      T(".resolvedLnScale", "experimentally_resolved_ln/scale"); T(".resolvedLnOffset", "experimentally_resolved_ln/offset");
    }
  }
  T(".leftHalfDistanceLogits", "left_half_distance_logits/weights");
  T(".paeLogits", "pae_logits/weights");
  if (has("inter_half_distance_logits/weights")) { T(".interHalfDistanceLogits", "inter_half_distance_logits/weights"); T(".paeInterLogits", "pae_inter_logits/weights"); }
  T(".plddtLogits", "plddt_logits/weights");
  if (has("experimentally_resolved_logits/weights")) T(".experimentallyResolvedLogits", "experimentally_resolved_logits/weights");
}

inline void structuralExpanderWeights(Walk& w, const std::string& p) {
  const std::string E = STRUCTURAL_EXPANDER + "/";
  const auto& role = w.S.dims(E + "single_input_role_embedding");
  w.num(p + ".roles", role[0]); w.num(p + ".singleInputChannels", role[1]);
  w.num(p + ".singleChannels", w.S.dims(E + "single_role_embedding")[1]); w.num(p + ".pairChannels", w.S.dims(E + "same_parent_embedding")[1]);
  for (auto& [path, name] : std::vector<std::pair<std::string, std::string>>{
         {"singleInputRoleEmbedding", "single_input_role_embedding"}, {"singleRoleEmbedding", "single_role_embedding"},
         {"singleSplitNormScale", "single_split_norm/scale"}, {"singleSplitNormOffset", "single_split_norm/offset"},
         {"singleSplit1", "single_split_1/weights"}, {"singleSplit2", "single_split_2/weights"}, {"pairBlockProj", "pair_block_proj"},
         {"sameParentEmbedding", "same_parent_embedding"}, {"sameResidueTwinEmbedding", "same_residue_twin_embedding"},
         {"prevBbChainEmbedding", "prev_bb_chain_embedding"}, {"nextBbChainEmbedding", "next_bb_chain_embedding"},
         {"rolePairTypeEmbedding", "role_pair_type_embedding"}, {"attnBiasSameParent", "attn_bias_same_parent"},
         {"attnBiasSameResidueTwin", "attn_bias_same_residue_twin"}, {"attnBiasPrevBbChain", "attn_bias_prev_bb_chain"},
         {"attnBiasNextBbChain", "attn_bias_next_bb_chain"}, {"attnBiasRolePairType", "attn_bias_role_pair_type"}})
    w.whole(p + "." + path, E + name);
}

inline void openddeConfidenceWeights(Walk& w, const std::string& p) {
  const std::string C = CONFIDENCE + "/";
  const auto& dd = w.S.dims(C + "linear_no_bias_d/weights");
  const auto& pl = w.S.dims(C + "plddt_weight");
  w.num(p + ".pairChannels", dd[1]); w.num(p + ".singleChannels", pl[1]);
  w.num(p + ".singleInputChannels", w.S.dims(C + "linear_no_bias_s1/weights")[0]);
  w.num(p + ".distanceBins", dd[0]); w.num(p + ".paeBins", w.S.dims(C + "linear_no_bias_pae/weights")[1]);
  w.num(p + ".pdeBins", w.S.dims(C + "linear_no_bias_pde/weights")[1]); w.num(p + ".plddtBins", pl[2]);
  w.num(p + ".resolvedBins", w.S.dims(C + "resolved_weight")[2]); w.num(p + ".denseSlots", pl[0]);
  for (auto& [path, name] : std::vector<std::pair<std::string, std::string>>{
         {"inputStrunkLnScale", "input_strunk_ln/scale"}, {"inputStrunkLnOffset", "input_strunk_ln/offset"},
         {"s1", "linear_no_bias_s1/weights"}, {"s2", "linear_no_bias_s2/weights"}, {"distance", "linear_no_bias_d/weights"},
         {"distanceRaw", "linear_no_bias_d_wo_onehot/weights"}, {"paeLnScale", "pae_ln/scale"}, {"paeLnOffset", "pae_ln/offset"},
         {"pae", "linear_no_bias_pae/weights"}, {"pdeLnScale", "pde_ln/scale"}, {"pdeLnOffset", "pde_ln/offset"},
         {"pde", "linear_no_bias_pde/weights"}, {"plddtLnScale", "plddt_ln/scale"}, {"plddtLnOffset", "plddt_ln/offset"},
         {"plddtWeight", "plddt_weight"}, {"resolvedLnScale", "resolved_ln/scale"}, {"resolvedLnOffset", "resolved_ln/offset"},
         {"resolvedWeight", "resolved_weight"}})
    w.whole(p + "." + path, C + name);
  for (int i = 0; i < 4; ++i) pairformerBlockWeights(w, p + ".blocks." + std::to_string(i), i, OPENDDE_CONFIDENCE_STACK);
}

// ---------------------------------------------------------------- diffusion-weights.js
inline std::string stackName(bool perBlock) { return perBlock ? "__layer_stack_no_per_layer" : "__layer_stack_with_per_layer"; }
inline std::string txStackFor(bool perBlock) { std::string n = stackName(perBlock); return TX + "/" + n + "/" + n + "/transformer"; }

// atomPairNorm: an atom stack's pair LayerNorm and its logits projection - once for the stack, or once a block
// (folded into each block's projection, the scales then ones) - and the fields it fills, in the encoder's order
struct PairNorm { bool perBlock; std::string scaleName, projName; long long C = 0, H = 0, blocks = 3; };
inline PairNorm atomPairNorm(const Walk& w, const std::string& stackRoot, bool perBlock) {
  PairNorm n{perBlock, stackRoot + "/pair_input_layer_norm/scale", stackRoot + "/pair_logits_projection/weights"};
  if (perBlock) { n.C = w.S.dims(n.scaleName).back(); n.H = w.S.size(n.projName) / w.S.dims(n.projName)[0] / n.C; }
  return n;
}
inline void pairNormScale0(Walk& w, const std::string& p, const PairNorm& n) {          // scale[0]
  if (n.perBlock) w.ones(p, n.C); else w.whole(p, n.scaleName);
}
inline void pairNormProjection(Walk& w, const std::string& p, const PairNorm& n) {     // packedProjection ?? projection[0]
  if (n.perBlock) w.fold3(p, n.C, n.blocks, n.H, n.scaleName, 0, n.projName, 0); else w.whole(p, n.projName);
}
inline void pairNormScales(Walk& w, const std::string& p, const PairNorm& n) {
  for (int b = 0; b < n.blocks; ++b) { if (n.perBlock) w.ones(p + "." + std::to_string(b), n.C); else w.whole(p + "." + std::to_string(b), n.scaleName); }
}
inline void pairNormProjections(Walk& w, const std::string& p, const PairNorm& n) {
  for (int b = 0; b < n.blocks; ++b) {
    std::string q = p + "." + std::to_string(b);
    if (n.perBlock) w.fold2(q, n.C, n.H, n.scaleName, b * n.C, n.projName, b * n.C * n.H); else w.whole(q, n.projName);
  }
}

inline void atomBlockWith(Walk& w, const std::string& p, const std::string& root, long long index, const Dialect& d) {
  bool chai = d.is("chaiAtomStack");
  auto at = [&](const std::string& path, const std::string& leaf) { w.stacked(p + "." + path, root + leaf, index); };
  auto maybe = [&](const std::string& path, const std::string& leaf) { w.stackedIfPresent(p + "." + path, root + leaf, index); };
  auto opt = [&](const std::string& path, const std::string& leaf) { if (chai) maybe(path, leaf); else at(path, leaf); };
  opt("qSingleCondLayerNormScale", "qsingle_cond_layer_norm/scale"); at("qSingleCondScaleWeights", "qsingle_cond_scale/weights");
  opt("qSingleCondScaleBias", "qsingle_cond_scale/bias"); at("qSingleCondBias", "qsingle_cond_bias/weights");
  opt("kSingleCondLayerNormScale", "ksingle_cond_layer_norm/scale"); at("kSingleCondScaleWeights", "ksingle_cond_scale/weights");
  opt("kSingleCondScaleBias", "ksingle_cond_scale/bias"); at("kSingleCondBias", "ksingle_cond_bias/weights");
  at("qProjection", "q_projection/weights"); at("qBias", "q_projection/bias"); at("kProjection", "k_projection/weights");
  at("vProjection", "v_projection/weights");
  if (chai) {
    const auto& q = w.S.dims(root + "q_projection/weights");
    long long n = 1;
    for (size_t k = 1; k < q.size(); ++k) n *= q[k];
    w.zeros(p + ".gatingQuery", n);
  } else at("gatingQuery", "gating_query/weights");
  maybe("queryLayerNormScale", "query_layer_norm/scale"); maybe("queryLayerNormOffset", "query_layer_norm/offset");
  maybe("keyLayerNormScale", "key_layer_norm/scale"); maybe("keyLayerNormOffset", "key_layer_norm/offset");
  opt("Transition2", "transition2/weights"); at("AdaptiveZeroCondWeights", "adaptive_zero_cond/weights");
  at("AdaptiveZeroCondBias", "adaptive_zero_cond/bias");
  opt("ffwSingleCondLayerNormScale", "ffw_single_cond_layer_norm/scale"); at("ffwSingleCondScaleWeights", "ffw_single_cond_scale/weights");
  opt("ffwSingleCondScaleBias", "ffw_single_cond_scale/bias"); at("ffwSingleCondBias", "ffw_single_cond_bias/weights");
  at("ffwTransition1", "ffw_transition1/weights"); maybe("ffwAToB", "ffw_a_to_b/weights"); at("ffwTransition2", "ffw_transition2/weights");
  at("ffwAdaptiveZeroCondWeights", "ffw_adaptive_zero_cond/weights"); at("ffwAdaptiveZeroCondBias", "ffw_adaptive_zero_cond/bias");
  // atomBlockDialect, then chaiAtomStack (an undefined flag writes nothing)
  for (const char* k : {"chainedAtomLayerNorm", "keyMaskedAtomAttention", "maskAtomActPerBlock", "diffusionNoResidual", "chaiAtomStack"}) {
    std::string v = d.value(k);
    if (v == "true" || v == "false") w.flag(p + "." + k, v == "true");
    else if (!v.empty() && v != "null") throw std::runtime_error(std::string("an atom block's flag ") + k + " is " + v);
    else if (std::string(k) != "chaiAtomStack") throw std::runtime_error("an atom block's dialect flags have no defaults");
  }
}

inline void constantAtomBias(Walk& w, const std::string& p, const std::string& a, const std::string& b) {
  bool ha = w.S.has(a), hb = w.S.has(b);
  if (ha && hb) throw std::runtime_error("this bundle carries " + a + " and " + b + "; they are the same term and a checkpoint with both is a converter bug, not a sum");
  if (ha) w.whole(p, a);
  else if (hb) w.whole(p, b);
}

inline void conditioningWeights(Walk& w, const std::string& p, const Dialect& d, bool stockAf3) {
  auto T = [&](const std::string& path, const std::string& name) { w.whole(p + path, HEAD + "/" + name); };
  auto O = [&](const std::string& path, const std::string& name) { w.wholeIfPresent(p + path, HEAD + "/" + name); };
  if (!d.defined("splitPairConditioning")) throw std::runtime_error("dialect.splitPairConditioning has no default");
  bool splitPair = d.is("splitPairConditioning");
  bool hasSplit = w.S.has(HEAD + "/z_trunk_projection/weights");
  if (hasSplit != splitPair) throw std::runtime_error(std::string("this bundle ") + (hasSplit ? "carries" : "does not carry") + " z_trunk_projection and its dialect says otherwise");
  if (!d.defined("projectedRelpos"))
    throw std::runtime_error("dialect.projectedRelpos has no default: AF3 concatenates the RAW 139 relative-position features and protenix2 projects them to the pair width first");
  bool projectedRelpos = d.is("projectedRelpos");
  bool hasRelpe = w.S.has(HEAD + "/relpe_projection/weights");
  if (hasRelpe != (splitPair || projectedRelpos))
    throw std::runtime_error(std::string("this bundle ") + (hasRelpe ? "carries" : "does not carry") + " relpe_projection and its dialect says otherwise");
  bool chaiCond = d.is("chaiDiffusionConditioning");
  const auto& pci = w.S.dims(HEAD + "/pair_cond_initial_projection/weights");
  const auto& sa = w.S.dims("diffuser/evoformer/single_activations/weights");
  w.num(p + ".pairChannels", pci[1]);
  w.num(p + ".seqChannels", w.S.dims(HEAD + "/single_cond_initial_projection/weights")[1]);
  w.num(p + ".trunkSingleChannels", sa[1]); w.num(p + ".targetFeatWidth", sa[0]); w.num(p + ".relativeWidth", 139);
  w.num(p + ".trunkPairChannels", chaiCond ? pci[0] / 2.0 : splitPair ? (double)w.S.dims(HEAD + "/z_trunk_projection/weights")[0]
                                  : projectedRelpos ? (double)(pci[0] - w.S.dims(HEAD + "/relpe_projection/weights")[1]) : (double)(pci[0] - 139));
  T(".pairCondInitialNormScale", "pair_cond_initial_norm/scale"); O(".pairCondInitialNormOffset", "pair_cond_initial_norm/offset");
  T(".pairCondInitialProjection", "pair_cond_initial_projection/weights");
  if (splitPair) {
    T(".zTrunkNormScale", "z_trunk_norm/scale"); O(".zTrunkNormOffset", "z_trunk_norm/offset");
    T(".zTrunkProjection", "z_trunk_projection/weights"); T(".relpeProjection", "relpe_projection/weights");
  } else if (projectedRelpos) T(".relpeProjection", "relpe_projection/weights");
  auto transition = [&](const std::string& q, const std::string& prefix) {
    T(q + ".ffwLayerNormScale", prefix + "ffw_layer_norm/scale"); T(q + ".ffwLayerNormOffset", prefix + "ffw_layer_norm/offset");
    T(q + ".ffwTransition1", prefix + "ffw_transition1/weights"); T(q + ".ffwTransition2", prefix + "ffw_transition2/weights");
  };
  transition(".pairTransitions.0", "pair_transition_0"); transition(".pairTransitions.1", "pair_transition_1");
  if (chaiCond) {
    T(".pairCondFinalNormScale", "pair_cond_final_norm/scale"); T(".pairCondFinalNormOffset", "pair_cond_final_norm/offset");
    T(".singleCondFinalNormScale", "single_cond_final_norm/scale"); T(".singleCondFinalNormOffset", "single_cond_final_norm/offset");
    w.whole(p + ".structurePairWeights", "diffuser/chai1_structure_token_pair/weights");
    w.whole(p + ".structurePairBias", "diffuser/chai1_structure_token_pair/bias");
    w.whole(p + ".structureBondWeights", "diffuser/chai1_structure_bond/weights");
  }
  T(".singleCondInitialNormScale", "single_cond_initial_norm/scale"); O(".singleCondInitialNormOffset", "single_cond_initial_norm/offset");
  T(".singleCondInitialProjection", "single_cond_initial_projection/weights");
  O(".singleCondInitialProjectionBias", "single_cond_initial_projection/bias");
  transition(".singleTransitions.0", "single_transition_0"); transition(".singleTransitions.1", "single_transition_1");
  if (stockAf3) { w.raw("c " + p + ".fourierWeight " + AF3_FOURIER_WEIGHT); w.raw("c " + p + ".fourierBias " + AF3_FOURIER_BIAS); }
  else { T(".fourierWeight", "fourier_embedding_weight"); T(".fourierBias", "fourier_embedding_bias"); }
  T(".noiseEmbeddingInitialNormScale", "noise_embedding_initial_norm/scale");
  O(".noiseEmbeddingInitialNormOffset", "noise_embedding_initial_norm/offset");
  T(".noiseEmbeddingInitialProjection", "noise_embedding_initial_projection/weights");
}

inline void diffusionWeights(Walk& w, const std::string& p, const Dialect& d, bool stockAf3) {
  auto T = [&](const std::string& path, const std::string& name) { w.whole(p + path, HEAD + "/" + name); };
  auto O = [&](const std::string& path, const std::string& name) { w.wholeIfPresent(p + path, HEAD + "/" + name); };
  if (!d.defined("perBlockPairLayerNorm"))
    throw std::runtime_error("dialect.perBlockPairLayerNorm has no default: AF3 normalises the token transformer's pair conditioning once for the stack, OpenDDE once per block");
  bool perBlockPair = d.is("perBlockPairLayerNorm");
  std::string sn = stackName(perBlockPair);
  if (!d.defined("perBlockAtomPairLayerNorm")) throw std::runtime_error("dialect.perBlockAtomPairLayerNorm has no default");
  bool atomPerBlock = d.is("perBlockAtomPairLayerNorm");
  std::string atomStack = atomPerBlock ? "/__layer_stack_no_per_layer" : "";
  PairNorm enc = atomPairNorm(w, ENCODER + atomStack, atomPerBlock), dec = atomPairNorm(w, DECODER + atomStack, atomPerBlock);
  std::string projectionName = perBlockPair ? TX + "/" + sn + "/" + sn + "/pair_logits_projection/weights" : TX + "/" + sn + "/pair_logits_projection/weights";
  const auto& projShape = w.S.dims(projectionName);
  long long superBlocks = projShape[0], projectionStride = w.S.size(projectionName) / superBlocks;
  bool chai = d.is("chaiAtomStack");
  std::string tx = txStackFor(perBlockPair);
  // txShape
  const auto& q = w.S.dims(tx + "q_projection/weights");
  if (q.size() != 5) throw std::runtime_error("the token transformer's q_projection is rank " + std::to_string(q.size()) + ", not 5");
  long long channels = q[2], heads = q[3], dimension = q[4];
  if (heads * dimension != channels) throw std::runtime_error("the token transformer's heads do not make its channels");
  long long hidden = w.S.dims(tx + "ffw_transition1/weights").back();
  if (hidden % (channels * 2) != 0) throw std::runtime_error("the token transformer's transition hidden is not a whole SwiGLU factor");
  long long seqChannels = w.S.dims(HEAD + "/single_cond_initial_projection/weights")[1];
  w.num(p + ".seqChannels", seqChannels); w.num(p + ".perTokenChannels", 768);
  if (!chai) T(".singleCondEmbeddingNormScale", "single_cond_embedding_norm/scale");
  O(".singleCondEmbeddingNormOffset", "single_cond_embedding_norm/offset");
  T(".singleCondEmbeddingProjection", "single_cond_embedding_projection/weights");
  T(".outputNormScale", "output_norm/scale"); O(".outputNormOffset", "output_norm/offset");
  conditioningWeights(w, p + ".conditioning", d, stockAf3);
  std::string t = p + ".transformer";
  w.num(t + ".channels", channels); w.num(t + ".heads", heads); w.num(t + ".dimension", dimension);
  w.num(t + ".blocksPerSuperBlock", q[1]); w.num(t + ".transitionFactor", (double)hidden / (channels * 2));
  w.num(t + ".condChannels", seqChannels);
  w.num(t + ".pairChannels", perBlockPair ? w.S.dims(TX + "/" + sn + "/" + sn + "/pair_logits_projection/weights")[2]
                                          : w.S.dims(TX + "/" + sn + "/pair_logits_projection/weights")[1]);
  w.flag(t + ".pairNormPerBlock", perBlockPair);
  w.flag(t + ".noResidual", d.is("diffusionNoResidual"));
  w.flag(t + ".chaiAdaLn", chai);
  std::string scaleName = TX + "/" + sn + "/" + sn + "/pair_input_layer_norm/scale";
  long long foldChannels = perBlockPair ? w.S.dims(scaleName)[2] : 0;
  if (perBlockPair) w.ones(t + ".pairInputLayerNormScale", foldChannels);
  else w.whole(t + ".pairInputLayerNormScale", TX + "/pair_input_layer_norm/scale");
  for (long long s = 0; s < superBlocks; ++s) {
    std::string g = t + ".superBlocks." + std::to_string(s);
    if (perBlockPair) {
      long long B = w.S.dims(scaleName)[1], C = foldChannels, H = projShape[3];
      w.fold3(g + ".pairLogitsProjection", C, B, H, scaleName, s * B * C, projectionName, s * B * C * H);
    } else {
      w.raw("b " + g + ".pairLogitsProjection " + projectionName + " " + std::to_string(s * projectionStride) + " " + std::to_string(projectionStride));
    }
    for (int inner = 0; inner < 4; ++inner) {
      std::string bp = g + ".blocks." + std::to_string(inner);
      long long index = s * 4 + inner;
      auto at = [&](const std::string& path, const std::string& leaf) { w.stacked(bp + "." + path, tx + leaf, index, 2); };
      auto maybeTx = [&](const std::string& path, const std::string& leaf) { w.stackedIfPresent(bp + "." + path, tx + leaf, index, 2); };
      auto opt = [&](const std::string& path, const std::string& leaf) { if (chai) maybeTx(path, leaf); else at(path, leaf); };
      opt("SingleCondLayerNormScale", "single_cond_layer_norm/scale"); at("SingleCondScaleWeights", "single_cond_scale/weights");
      opt("SingleCondScaleBias", "single_cond_scale/bias"); at("SingleCondBias", "single_cond_bias/weights");
      at("qProjection", "q_projection/weights"); at("qBias", "q_projection/bias"); at("kProjection", "k_projection/weights");
      at("vProjection", "v_projection/weights");
      if (chai) {
        long long n = 1;
        for (size_t k = 2; k < q.size(); ++k) n *= q[k];
        w.zeros(bp + ".gatingQuery", n);
      } else at("gatingQuery", "gating_query/weights");
      maybeTx("queryLayerNormScale", "query_layer_norm/scale"); maybeTx("queryLayerNormOffset", "query_layer_norm/offset");
      maybeTx("keyLayerNormScale", "key_layer_norm/scale"); maybeTx("keyLayerNormOffset", "key_layer_norm/offset");
      at("Transition2", "transition2/weights"); at("AdaptiveZeroCondWeights", "adaptive_zero_cond/weights");
      at("AdaptiveZeroCondBias", "adaptive_zero_cond/bias");
      opt("ffwSingleCondLayerNormScale", "ffw_single_cond_layer_norm/scale"); at("ffwSingleCondScaleWeights", "ffw_single_cond_scale/weights");
      opt("ffwSingleCondScaleBias", "ffw_single_cond_scale/bias"); at("ffwSingleCondBias", "ffw_single_cond_bias/weights");
      at("ffwTransition1", "ffw_transition1/weights"); maybeTx("ffwAToB", "ffw_a_to_b/weights"); at("ffwTransition2", "ffw_transition2/weights");
      at("ffwAdaptiveZeroCondWeights", "ffw_adaptive_zero_cond/weights"); at("ffwAdaptiveZeroCondBias", "ffw_adaptive_zero_cond/bias");
    }
  }
  std::string e = p + ".encoder";
  w.num(e + ".channels", 128); w.num(e + ".pairChannels", 16); w.num(e + ".heads", 4); w.num(e + ".dimension", 32);
  w.num(e + ".perTokenChannels", 768); w.num(e + ".trunkSingleChannels", 384);
  w.num(e + ".trunkPairChannels", w.S.dims(HEAD + "/diffusion_embed_trunk_pair_cond/weights")[0]);
  auto TE = [&](const std::string& path, const std::string& name) { w.whole(e + path, HEAD + "/" + name); };
  TE(".singleToPairCondRow", "diffusion_single_to_pair_cond_row_1/weights");
  TE(".singleToPairCondCol", "diffusion_single_to_pair_cond_col_1/weights");
  if (!chai) {
    TE(".embedPairOffsets", "diffusion_embed_pair_offsets_1/weights"); TE(".embedPairDistances", "diffusion_embed_pair_distances_1/weights");
    TE(".embedPairOffsetsValid", "diffusion_embed_pair_offsets_valid/weights");
  } else {
    TE(".embedAtomPairFeat", "diffusion_embed_atom_pair_feat/weights"); TE(".embedAtomPairFeatBias", "diffusion_embed_atom_pair_feat/bias");
  }
  TE(".pairMlp1", "diffusion_pair_mlp_1/weights"); TE(".pairMlp2", "diffusion_pair_mlp_2/weights");
  if (!chai) TE(".pairMlp3", "diffusion_pair_mlp_3/weights");
  pairNormScale0(w, e + ".pairInputLayerNormScale", enc); pairNormProjection(w, e + ".pairLogitsProjection", enc);
  pairNormScales(w, e + ".pairInputLayerNormScales", enc); pairNormProjections(w, e + ".pairLogitsProjections", enc);
  w.flag(e + ".pairNormPerBlock", enc.perBlock);
  TE(".lnormTrunkSingleCondScale", "diffusion_lnorm_trunk_single_cond/scale");
  w.wholeIfPresent(e + ".lnormTrunkSingleCondOffset", HEAD + "/diffusion_lnorm_trunk_single_cond/offset");
  TE(".embedTrunkSingleCond", "diffusion_embed_trunk_single_cond/weights");
  TE(".lnormTrunkPairCondScale", "diffusion_lnorm_trunk_pair_cond/scale");
  w.wholeIfPresent(e + ".lnormTrunkPairCondOffset", HEAD + "/diffusion_lnorm_trunk_pair_cond/offset");
  TE(".embedTrunkPairCond", "diffusion_embed_trunk_pair_cond/weights");
  TE(".atomPositionsToFeatures", "diffusion_atom_positions_to_features/weights");
  w.wholeIfPresent(e + ".atomChiralToFeatures", HEAD + "/diffusion_atom_chiral_to_features/weights");
  TE(".projectAtomFeaturesForAggr", "diffusion_project_atom_features_for_aggr/weights");
  std::string encStack = ENCODER + "/" + stackName(atomPerBlock) + "/diffusion_atom_transformer_encoder";
  for (int b = 0; b < 3; ++b) atomBlockWith(w, e + ".blocks." + std::to_string(b), encStack, b, d);
  std::string c = p + ".decoder";
  w.num(c + ".channels", 128); w.num(c + ".pairChannels", 16); w.num(c + ".heads", 4); w.num(c + ".dimension", 32);
  w.num(c + ".perTokenChannels", 768);
  pairNormScale0(w, c + ".pairInputLayerNormScale", dec); pairNormProjection(w, c + ".pairLogitsProjection", dec);
  pairNormScales(w, c + ".pairInputLayerNormScales", dec); pairNormProjections(w, c + ".pairLogitsProjections", dec);
  w.flag(c + ".pairNormPerBlock", dec.perBlock);
  auto TD = [&](const std::string& path, const std::string& name) { w.whole(c + path, HEAD + "/" + name); };
  TD(".projectTokenFeaturesForBroadcast", "diffusion_project_token_features_for_broadcast/weights");
  TD(".atomFeaturesLayerNormScale", "diffusion_atom_features_layer_norm/scale");
  w.wholeIfPresent(c + ".atomFeaturesLayerNormOffset", HEAD + "/diffusion_atom_features_layer_norm/offset");
  TD(".atomFeaturesToPositionUpdate", "diffusion_atom_features_to_position_update/weights");
  if (chai) {
    TD(".postAtomCondLayerNormScale", "diffusion_post_atom_cond_layer_norm/scale");
    TD(".postAtomCondLayerNormOffset", "diffusion_post_atom_cond_layer_norm/offset");
  }
  std::string decStack = DECODER + "/" + stackName(atomPerBlock) + "/diffusion_atom_transformer_decoder";
  for (int b = 0; b < 3; ++b) atomBlockWith(w, c + ".blocks." + std::to_string(b), decStack, b, d);
}

inline void targetFeatureWeights(Walk& w, const std::string& p, const Dialect& d) {
  const std::string root = "diffuser/evoformer_conditioning", encoder = root + "_atom_transformer_encoder";
  if (!d.defined("perBlockAtomPairLayerNorm"))
    throw std::runtime_error("dialect.perBlockAtomPairLayerNorm has no default: AF3 normalises the atom-pair conditioning once for the stack, OpenDDE once per block");
  bool perBlockPair = d.is("perBlockAtomPairLayerNorm"), chai = d.is("chaiAtomStack");
  std::string stack = encoder + "/" + stackName(perBlockPair) + "/evoformer_conditioning_atom_transformer_encoder";
  std::string stackRoot = perBlockPair ? encoder + "/__layer_stack_no_per_layer" : encoder;
  auto W = [&](const std::string& path, const std::string& leaf) { w.whole(p + path, root + "_" + leaf + "/weights"); };
  PairNorm pn = atomPairNorm(w, stackRoot, perBlockPair);
  std::string r = ".reference";
  w.num(p + r + ".channels", 128);
  W(r + ".embedRefPos", "embed_ref_pos"); W(r + ".embedRefMask", "embed_ref_mask"); W(r + ".embedRefElement", "embed_ref_element");
  W(r + ".embedRefCharge", "embed_ref_charge"); W(r + ".embedRefAtomName", "embed_ref_atom_name");
  constantAtomBias(w, p + r + ".embedAtomFeaturesBias", root + "_embed_atom_features_bias", root + "_conformer_embedding_bias");
  std::string e = p + ".encoder";
  w.num(e + ".channels", 128); w.num(e + ".pairChannels", 16); w.num(e + ".heads", 4); w.num(e + ".dimension", 32);
  w.num(e + ".perTokenChannels", 384);
  W(".encoder.singleToPairCondRow", "single_to_pair_cond_row_1"); W(".encoder.singleToPairCondCol", "single_to_pair_cond_col_1");
  if (!chai) {
    W(".encoder.embedPairOffsets", "embed_pair_offsets_1"); W(".encoder.embedPairDistances", "embed_pair_distances_1");
    W(".encoder.embedPairOffsetsValid", "embed_pair_offsets_valid");
  } else {
    W(".encoder.embedAtomPairFeat", "embed_atom_pair_feat"); w.whole(e + ".embedAtomPairFeatBias", root + "_embed_atom_pair_feat/bias");
  }
  W(".encoder.pairMlp1", "pair_mlp_1"); W(".encoder.pairMlp2", "pair_mlp_2");
  if (!chai) W(".encoder.pairMlp3", "pair_mlp_3");
  pairNormScale0(w, e + ".pairInputLayerNormScale", pn); pairNormProjection(w, e + ".pairLogitsProjection", pn);
  w.flag(e + ".pairNormPerBlock", pn.perBlock);
  pairNormScales(w, e + ".pairInputLayerNormScales", pn); pairNormProjections(w, e + ".pairLogitsProjections", pn);
  W(".encoder.projectAtomFeaturesForAggr", "project_atom_features_for_aggr");
  for (int b = 0; b < 3; ++b) atomBlockWith(w, e + ".blocks." + std::to_string(b), stack, b, d);
  if (w.S.has("diffuser/boltz2_res_type_encoding/weights")) {
    for (auto& [path, name] : std::vector<std::pair<std::string, std::string>>{
           {"resType", "boltz2_res_type_encoding"}, {"msaProfile", "boltz2_msa_profile_encoding"}, {"molType", "boltz2_mol_type_conditioning"},
           {"cyclic", "boltz2_cyclic_conditioning"}, {"method", "boltz2_method_conditioning"}, {"modified", "boltz2_modified_conditioning"}})
      w.whole(e + ".targetFeatSum." + path, "diffuser/" + name + "/weights");
  }
  if (d.is("chaiTokenEmbedding")) {
    for (auto& [path, name] : std::vector<std::pair<std::string, std::string>>{
           {"tokenFeatureWeights", "token_feature_embedding/weights"}, {"tokenFeatureBias", "token_feature_embedding/bias"},
           {"msaProfileWeights", "msa_profile_embedding/weights"}, {"esmWeights", "esm_embedding/weights"},
           {"singleProjInTrunk", "single_proj_in_trunk/weights"}, {"singleProjInStructure", "single_proj_in_structure/weights"}})
      w.whole(e + ".chaiToken." + path, "diffuser/chai1_" + name);
  }
  w.num(e + ".trunkSingleChannels", 384); w.num(e + ".trunkPairChannels", 128);
  w.zeros(e + ".lnormTrunkSingleCondScale", 384); w.zeros(e + ".embedTrunkSingleCond", 384 * 128);
  w.zeros(e + ".lnormTrunkPairCondScale", 128); w.zeros(e + ".embedTrunkPairCond", 128 * 16);
  w.zeros(e + ".atomPositionsToFeatures", 3 * 128);
}

inline void atomReference(Walk& w, const std::string& p) {
  w.num(p + ".channels", 128);
  for (auto& [path, name] : std::vector<std::pair<std::string, std::string>>{
         {"embedRefPos", "diffusion_embed_ref_pos/weights"}, {"embedRefMask", "diffusion_embed_ref_mask/weights"},
         {"embedRefElement", "diffusion_embed_ref_element/weights"}, {"embedRefCharge", "diffusion_embed_ref_charge/weights"},
         {"embedRefAtomName", "diffusion_embed_ref_atom_name/weights"}})
    w.whole(p + "." + path, HEAD + "/" + name);
  constantAtomBias(w, p + ".embedAtomFeaturesBias", HEAD + "/diffusion_embed_atom_features_bias", HEAD + "/diffusion_conformer_embedding_bias");
}

// every weight line of one family's bundle, in cuda/af3/export-model.mjs's order
inline std::vector<std::string> af3WeightLines(const std::string& family, const Shapes& shapes) {
  Dialect d = dialectNamed(family);
  std::string canonical = d.t->name;
  Walk w(shapes);
  bool dialectWritten = false;
  trunkWeights(w, d, dialectWritten);
  diffusionWeights(w, "diffusion", d, canonical == "alphafold3");
  if (d.is("structuralTokens")) {
    structuralExpanderWeights(w, "expander");
    for (int i = 0; i < 4; ++i) pairformerBlockWeights(w, "refiner.blocks." + std::to_string(i), i, STRUCTURAL_REFINER);
    openddeConfidenceWeights(w, "ddeConfidence");
  } else {
    confidenceWeights(w, "confidence");
  }
  targetFeatureWeights(w, "targetFeat", d);
  atomReference(w, "atomReference");
  return w.lines;
}

}  // namespace lf::weights
