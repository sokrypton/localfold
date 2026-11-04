// The autoencoder's decoder: a token's latents and its alpha carbon in, its 37 atoms out.
//
// `cpu/rfd4p/decoder.js` is the spec, and `webgpu/rfd4p/decoder.js` reads the same way:
//   c       is IDENTICALLY ZERO - the conditioning factory has no features, and two bias-free SwiGLU transitions
//           of zero are zero, so this port allocates it and never computes it
//   s_init  = init_repr_factory(the 214 features: ca_coors_nm, z_latent_seq, the token-type embedding, C_SEQ's)
//   12 x    [pair-biasless gated attention + SwiGLU transition], both under adaLNs - LayerNorm throughout, where
//           the trunk is RMSNorm throughout
//   heads   logit_linear (147), struct_linear (111 = 37 x 3 relative to the alpha carbon), atom_presence_linear (37)
//
// 🔴 THE ATTENDED ROWS ARE GATHERED AND THE ATTENTION RUNS UNMASKED OVER THEM, WHICH IS EXACT AND NOT AN
// APPROXIMATION. `attn = padding_mask & ~is_in_context_conditioning`, and the reference gives a masked key a score
// of -1e4 and zeroes a masked row at every stage; na rows with no mask at all compute the same numbers, as
// webgpu/rfd4p/decoder.js says in its own header. It also makes the decoder's cost the DESIGNED length rather than
// the input's, which for a binder on a big target is most of the work.
#include "rfd4p.h"
#include <cmath>
#include "rfd4p/featurise/tables.inc"

DecDims DD_;
float* DEC_TAP_ADA = nullptr;
float* DEC_TAP_ATTN = nullptr;

void checkDecoderDims() {
  auto dim = [](const char* name, int k) { return (int)M.dim(std::string("w/autoencoder.decoder.") + name, k); };
  const DecDims want;
  DD_.D = dim("init_repr_factory.linear_out.weight", 0);
  DD_.feat = dim("init_repr_factory.linear_out.weight", 1);
  DD_.types = dim("logit_linear.1.weight", 0);
  DD_.C = dim("transformer_layers.0.mhba.adaln.to_gamma.0.weight", 1);
  int layers = 0;
  while (M.has("w/autoencoder.decoder.transformer_layers." + std::to_string(layers) + ".mhba.mha.to_qkv.weight")) layers += 1;
  DD_.layers = layers;
  if (DD_.D != want.D || DD_.C != want.C || DD_.feat != want.feat || DD_.types != want.types || DD_.layers != want.layers)
    die("rfd4p decoder: this bundle is D=%d C=%d feat=%d types=%d over %d layers; these kernels are built for "
        "%d/%d/%d/%d over %d", DD_.D, DD_.C, DD_.feat, DD_.types, DD_.layers, want.D, want.C, want.feat, want.types,
        want.layers);
}

void decoderFeatures(const float* caNm, const float* zLatent, const int* tokenType, const int* seq, const float* pad,
                     float* feats) {
  const int n = D_.n, fdim = DD_.feat, types = DD_.types, latent = DD_.latent;
  // 🔴 ON THE HOST, AS EVERY OTHER PORT BUILDS ITS FEATURES. Two embedding lookups and a copy over n rows is
  // nothing beside the stack, and keeping the layout in one language is what lets `cpu/rfd4p/decoder.js` be read
  // as the spec line by line.
  const float* ttTab = M.hostF("w/autoencoder.decoder.init_repr_factory.feat_creators.2.embedding.weight");
  const float* seqTab = M.hostF("w/autoencoder.decoder.init_repr_factory.feat_creators.3.embedding.weight");
  float* out = (float*)host(feats);
  memset(out, 0, (size_t)n * fdim * 4);
  for (int i = 0; i < n; ++i) {
    if (pad && !pad[i]) continue;
    float* o = out + (size_t)i * fdim;
    for (int k = 0; k < 3; ++k) o[k] = caNm[i * 3 + k];
    for (int k = 0; k < latent; ++k) o[3 + k] = zLatent[(size_t)i * latent + k];
    for (int k = 0; k < 32; ++k) o[3 + latent + k] = ttTab[(size_t)tokenType[i] * 32 + k];
    // 🔴 A MASK TOKEN CONTRIBUTES NOTHING, and that is not the same as embedding token 0: `<M>` is residue type 0
    // and its row of the table is a real row, so taking it would condition the decoder on "this is a mask".
    if (seq[i] != 0)
      for (int k = 0; k < types; ++k) o[3 + latent + 32 + k] = seqTab[(size_t)seq[i] * types + k];
  }
}

// one decoder layer, in place on `s` over `rows` gathered rows. The trunk's twin is `tokenLayer` in trunk.mm; the
// differences are all here: LayerNorm rather than RMSNorm, no pair bias, and the 128-wide conditioning.
void decoderLayer(int layer, float* s, const float* c, size_t rows) {
  const int D = DD_.D, H = DD_.H, dimHead = DD_.D / DD_.H, C = DD_.C;
  const std::string L = "autoencoder.decoder.transformer_layers." + std::to_string(layer) + ".";
  const std::string MHA = L + "mhba.mha.";
  if (rows > (size_t)LF_ATTN_MAX_N) die("rfd4p decoder: %zu rows is past this port's float32 attention (%d)", rows, LF_ATTN_MAX_N);

  float* xa = scratch<float>("dl.xa", rows * D);
  adalnRows(s, c, L + "mhba.adaln", xa, rows, D, C, true);
  if (layer == 0 && DEC_TAP_ADA) copy(DEC_TAP_ADA, xa, rows * D * 4);
  float* nodeN = scratch<float>("dl.nodeN", rows * D);
  layerNorm(xa, nodeN, rows, D, M.f("w/" + MHA + "node_norm.weight"), opt(MHA + "node_norm.bias"));

  float* qkv = scratch<float>("dl.qkv", rows * 3 * D);
  lin(nodeN, MHA + "to_qkv.weight", qkv, rows, D, 3 * D, M.f("w/" + MHA + "to_qkv.bias"));
  float* q = scratch<float>("dl.q", rows * D);
  float* k = scratch<float>("dl.k", rows * D);
  layerNorm(qkv, q, rows, D, M.f("w/" + MHA + "q_layer_norm.weight"), opt(MHA + "q_layer_norm.bias"), 1e-5f, 3 * D, D);
  layerNorm(qkv + D, k, rows, D, M.f("w/" + MHA + "k_layer_norm.weight"), opt(MHA + "k_layer_norm.bias"), 1e-5f, 3 * D, D);

  float* g = scratch<float>("dl.g", rows * D);
  lin(nodeN, MHA + "to_g.weight", g, rows, D, D, M.f("w/" + MHA + "to_g.bias"));

  float* ctx = scratch<float>("dl.ctx", rows * D);
  RfdAttnArgs at{};
  at.q = q; at.k = k; at.v = qkv; at.bias = nullptr; at.g = g; at.out = ctx;
  at.n = (uint)rows; at.heads = (uint)H; at.dimHead = (uint)dimHead; at.ldv = (uint)(3 * D); at.vOff = 2 * D;
  at.scale = 1.f / sqrtf((float)dimHead);
  dispatch("rfd4p_attention", &at, sizeof at, Grid{(uint32_t)rows, (uint32_t)H, 1}, 256, rows * 4, "dec_attn");

  float* attnOut = scratch<float>("dl.attnOut", rows * D);
  lin(ctx, MHA + "to_out_node.weight", attnOut, rows, D, D, M.f("w/" + MHA + "to_out_node.bias"));
  if (layer == 0 && DEC_TAP_ATTN) copy(DEC_TAP_ATTN, attnOut, rows * D * 4);
  scaleAdd(s, attnOut, c, L + "mhba.scale_output", rows, D, C);

  adalnRows(s, c, L + "transition.adaln", xa, rows, D, C, true);
  float* hid = scratch<float>("dl.hid", rows * 8 * D);
  float* gated = scratch<float>("dl.gated", rows * 4 * D);
  float* tr = scratch<float>("dl.tr", rows * D);
  transition(xa, tr, rows, D, 4, L + "transition.transition", hid, gated);
  scaleAdd(s, tr, c, L + "transition.scale_output", rows, D, C);
}

void decode(const float* feats, const int* attnIdx, int na, float* sOut, float* logits, float* coors, float* presence,
            float* sInitOut, float* perLayer) {
  const int D = DD_.D, C = DD_.C, fdim = DD_.feat;
  const std::string P = "autoencoder.decoder.";
  float* fg = scratch<float>("dec.fg", (size_t)na * fdim);
  gatherRows(feats, fg, attnIdx, na, fdim);
  // c is zero and stays zero; allocated once rather than computed (see the header)
  float* c = scratch<float>("dec.c", (size_t)na * C);
  fill(c, 0, (size_t)na * C * 4);

  float* s = sOut;
  lin(fg, P + "init_repr_factory.linear_out.weight", s, na, fdim, D);
  if (sInitOut) copy(sInitOut, s, (size_t)na * D * 4);
  for (int l = 0; l < DD_.layers; ++l) {
    decoderLayer(l, s, c, na);
    if (perLayer) copy(perLayer + (size_t)l * na * D, s, (size_t)na * D * 4);
  }

  float* hn = scratch<float>("dec.hn", (size_t)na * D);
  struct Head { const char* name; float* out; int width; };
  const Head hs[] = {{"logit_linear", logits, DD_.types}, {"struct_linear", coors, 111},
                     {"atom_presence_linear", presence, 37}};
  for (const Head& h : hs) {
    layerNorm(s, hn, na, D, M.f("w/" + P + h.name + ".0.weight"), M.f("w/" + P + h.name + ".0.bias"));
    lin(hn, P + std::string(h.name) + ".1.weight", h.out, na, D, h.width);
  }
}

void decoderLayerOne(int layer, float* s, const float* c, size_t rows) { decoderLayer(layer, s, c, rows); }
