// The autoencoder's encoder: a structure's atoms in, each token's 32 local latents out.
//
// `cpu/rfd4p/encoder.js` is the spec. The transformer is the decoder's - `aeLayer` under this prefix - so what is
// here is the 1587-wide feature assembly and the head:
//   s_init = init_repr_factory(token type, the 37 atoms absolute and CA-relative with their masks, the chain
//            index, the incident bond-type counts, and each atom's element and charge embedded)
//   12 x   the same LayerNorm layers, over the ATTENDED rows
//   head   LayerNorm then latent_decoder_mean_n_log_scale -> [na][64], split into the mean and the log scale
//
// 🔴 THE GATE'S ARM IS THE DETERMINISTIC ONE AND THAT IS NOT A SIMPLIFICATION. The sampled latent is
// `mean + eps * exp(logScale)` with eps drawn from the SAMPLER's stream, so comparing a sampled z would compare
// two Box-Muller implementations rather than two encoders; the reference's own export takes `deterministic: true`
// for exactly that reason, and the draw is covered where it belongs - in the generation loop's replayed noise.
#include "rfd4p.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

EncDims ED_;

void checkEncoderDims() {
  auto dim = [](const char* name, int k) { return (int)M.dim(std::string("w/autoencoder.encoder.") + name, k); };
  const EncDims want;
  ED_.D = dim("init_repr_factory.linear_out.weight", 0);
  ED_.feat = dim("init_repr_factory.linear_out.weight", 1);
  ED_.latent = dim("latent_decoder_mean_n_log_scale.1.weight", 0) / 2;
  ED_.C = dim("transformer_layers.0.mhba.adaln.to_gamma.0.weight", 1);
  int layers = 0;
  while (M.has("w/autoencoder.encoder.transformer_layers." + std::to_string(layers) + ".mhba.mha.to_qkv.weight")) layers += 1;
  ED_.layers = layers;
  if (ED_.D != want.D || ED_.C != want.C || ED_.feat != want.feat || ED_.latent != want.latent || ED_.layers != want.layers)
    die("rfd4p encoder: this bundle is D=%d C=%d feat=%d latent=%d over %d layers; these kernels are built for "
        "%d/%d/%d/%d over %d", ED_.D, ED_.C, ED_.feat, ED_.latent, ED_.layers, want.D, want.C, want.feat,
        want.latent, want.layers);
}

void encoderFeatures(int n, const float* coorsNm, const float* atomMask, const float* chainIid,
                     const float* bondCounts, const int* element, const int* charge, float* feats) {
  const int fdim = ED_.feat;
  // The block offsets, read off the input's metadata because they are DERIVED FROM `feats_seq` and not a layout
  // anybody chose: reordering that list moves every block, and the exporter writes the offsets it used.
  const int ttDim = (int)M.meta("meta/encTtDim", 32), chDim = (int)M.meta("meta/encChargeDim", 16);
  const int oAbs = (int)M.meta("meta/encOffAbs", 32), oRel = (int)M.meta("meta/encOffRel", 180);
  const int oChain = (int)M.meta("meta/encOffChain", 328), oBonds = (int)M.meta("meta/encOffBonds", 329);
  const int oElCh = (int)M.meta("meta/encOffElCh", 699);
  const int ttCreator = (int)M.meta("meta/encTtCreator", 0), elChCreator = (int)M.meta("meta/encElChCreator", 5);
  const int slot = 8 + chDim;
  const std::string P = "autoencoder.encoder.init_repr_factory.feat_creators.";
  const float* ttTab = M.hostF("w/" + P + std::to_string(ttCreator) + ".embedding.weight");
  const float* elTab = M.hostF("w/" + P + std::to_string(elChCreator) + ".element_embedding.weight");
  const float* chTab = M.hostF("w/" + P + std::to_string(elChCreator) + ".charge_embedding.weight");
  const int* tokenType = M.hostI("in/tokenType");

  std::vector<float> out((size_t)n * fdim, 0.f);
  for (int i = 0; i < n; ++i) {
    float* o = out.data() + (size_t)i * fdim;
    for (int k = 0; k < ttDim; ++k) o[k] = ttTab[(size_t)tokenType[i] * ttDim + k];
    // the alpha carbon, if it is modelled: the relative block is measured from it
    const bool caPlaced = atomMask[i * 37 + 1] != 0;
    const float ca[3] = {caPlaced ? coorsNm[(i * 37 + 1) * 3] : 0.f, caPlaced ? coorsNm[(i * 37 + 1) * 3 + 1] : 0.f,
                         caPlaced ? coorsNm[(i * 37 + 1) * 3 + 2] : 0.f};
    for (int a = 0; a < 37; ++a) {
      const size_t at = (size_t)i * 37 + a;
      const float m = atomMask[at] != 0 ? 1.f : 0.f;
      o[oAbs + 111 + a] = m;
      o[oRel + 111 + a] = m;
      if (m != 0.f) {
        for (int k = 0; k < 3; ++k) {
          const float v = coorsNm[at * 3 + k];
          o[oAbs + a * 3 + k] = v;
          o[oRel + a * 3 + k] = (float)(v - ca[k]);
        }
        const int el = element[at];
        const int ch = std::min(10, std::max(0, charge[at] + 4));
        for (int k = 0; k < 8; ++k) o[oElCh + a * slot + k] = elTab[(size_t)el * 8 + k];
        for (int k = 0; k < chDim; ++k) o[oElCh + a * slot + 8 + k] = chTab[(size_t)ch * chDim + k];
      }
    }
    o[oChain] = chainIid[i];
    for (int k = 0; k < 370; ++k) o[oBonds + k] = bondCounts[(size_t)i * 370 + k];
  }
  // 🔴 UPLOADED, NOT WRITTEN THROUGH `host()`. `host()` hands back the CPU side of a shared buffer and says
  // "read them after a sync"; `upload` is in stream order, which is what a buffer a queued kernel will read needs.
  upload(feats, out.data(), out.size() * 4);
}

void encode(const float* feats, const int* attnIdx, int na, float* mean, float* logScale) {
  const int D = ED_.D, C = ED_.C, fdim = ED_.feat, L = ED_.latent;
  const std::string P = "autoencoder.encoder.";
  float* fg = scratch<float>("enc.fg", (size_t)na * fdim);
  gatherRows(feats, fg, attnIdx, na, fdim);
  float* c = scratch<float>("enc.c", (size_t)na * C);
  fill(c, 0, (size_t)na * C * 4);   // identically zero, as in the decoder

  float* s = scratch<float>("enc.s", (size_t)na * D);
  lin(fg, P + "init_repr_factory.linear_out.weight", s, na, fdim, D);
  for (int l = 0; l < ED_.layers; ++l) aeLayer(P, l, s, c, na);

  float* hn = scratch<float>("enc.hn", (size_t)na * D);
  layerNorm(s, hn, na, D, M.f("w/" + P + "latent_decoder_mean_n_log_scale.0.weight"),
            M.f("w/" + P + "latent_decoder_mean_n_log_scale.0.bias"));
  float* flat = scratch<float>("enc.flat", (size_t)na * 2 * L);
  lin(hn, P + "latent_decoder_mean_n_log_scale.1.weight", flat, na, D, 2 * L);
  // the head's two halves, column blocks of one row: a strided copy rather than a kernel
  copy2d(mean, L * 4, flat, 2 * L * 4, L * 4, na);
  copy2d(logScale, L * 4, flat + L, 2 * L * 4, L * 4, na);
}
