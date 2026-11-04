// The page's int8 bundle, read as this port's weights - and the widths checked against it.
//
// 🔴 NO WEIGHT WALK, WHICH IS THE POINT OF READING THE PAGE'S BUNDLE. metal/bioemu2 needs one because it is handed
// af3-any-model's blob and has to rewrite it into the page's names and layouts; here the bundle IS the page's, so
// every tensor arrives under the checkpoint's own name and `cpu/rfd4p/reference.js` can be read as the spec directly.
#include "rfd4p.h"

Dims D_;
// 🔴 HALF BY DEFAULT, AND THE MEASUREMENT IS WHY. Over 54 tokens every trunk stage is within 1e-4 of the CPU
// reference on the reference's own input and the chained trunk within 1.6e-4, the designs come out with the same
// geometry (CA-CA median 3.81 A), and a generation is 3.5x faster: 12.58 s -> 3.63 s for a 60-step design, and
// 3.36 -> 1.00 s for a 183-token one. `--f32` is the exact arm and the one the oracle gate's tight bounds
// belong to; it is one flag away and the recipe in main.mm runs both.
bool HALF = true;
bool NO_INSERTION = false;

void loadWeights(const std::string& dir) {
  // 🔴 EVERY TENSOR FLOAT32 UNTIL A STAGE ASKS OTHERWISE. The trunk's GEMMs are held to a float64 CPU reference at
  // 1e-7 to start with, and a half weight is a 1e-3 arm: it goes in per stage, measured, not as a global default.
  M.loadBundle(dir, "w", HALF ? [](const std::string& name, size_t n) {
    return n >= 4096 && name.find("_norm") == std::string::npos && name.find("ln_out") == std::string::npos;
  } : std::function<bool(const std::string&, size_t)>(nullptr));
}

void checkDims() {
  auto dim = [](const char* name, int k) { return (int)M.dim(std::string("w/") + name, k); };
  const Dims want = D_;
  D_.D = dim("nn.init_repr_factory.linear_out.weight", 0);
  D_.seqDim = dim("nn.init_repr_factory.linear_out.weight", 1);
  D_.C = dim("nn.cond_factory.linear_out.weight", 0);
  D_.condDim = dim("nn.cond_factory.linear_out.weight", 1);
  D_.P = dim("nn.pair_repr_builder.init_repr_factory.linear_out.weight", 0);
  D_.pairDim = dim("nn.pair_repr_builder.init_repr_factory.linear_out.weight", 1);
  D_.pairCondDim = dim("nn.pair_repr_builder.cond_factory.linear_out.weight", 1);
  D_.H = dim("nn.transformer_layers.0.mhba.mha.to_bias.weight", 0);
  D_.T = dim("nn.pair_update_layers.0.tri_mult_out.linear_a_p.weight", 0);
  int layers = 0;
  while (M.has("w/nn.transformer_layers." + std::to_string(layers) + ".mhba.mha.to_bias.weight")) layers += 1;
  D_.layers = layers;
  // 🔴 REFUSED BY NAME RATHER THAN FOLDED: a checkpoint of other widths is a different network, and the kernels
  // here are built for these. webgpu/rfd4p/trunk.js's `checkConfig` says the same thing from the config's side.
  if (D_.D != want.D || D_.P != want.P || D_.C != want.C || D_.H != want.H || D_.T != want.T || D_.layers != want.layers)
    die("rfd4p: this bundle is D=%d P=%d C=%d H=%d T=%d over %d layers; these kernels are built for %d/%d/%d/%d/%d over %d",
        D_.D, D_.P, D_.C, D_.H, D_.T, D_.layers, want.D, want.P, want.C, want.H, want.T, want.layers);
}
