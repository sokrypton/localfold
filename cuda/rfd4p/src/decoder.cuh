// RFD4-Proteina's autoencoder DECODER on CUDA: local latents and CA positions in, all-atom
// coordinates out. Held to `cpu/rfd4p/decoder.js` (`decodeLatentsCpu`) stage by stage.
//
// It is a plainer transformer than the trunk and differs from it in three ways that each change
// every number if missed:
//
// 🔴 EVERY NORM IS LayerNorm, NOT RMSNorm. The trunk's `norm_type` is `rmsnorm_torch`; the decoder is
//    built with `layer_norm` and every one of its norms carries a bias.
// 🔴 THERE IS NO PAIR BIAS. `use_attn_pair_bias` is false and the decoder has no pair track at all,
//    so the attention is 12 plain heads - flash still wants a bias pointer, so it gets a zeroed one.
// 🔴 AND ITS CONDITIONING IS A CONSTANT. The cond factory has no features (`feats_cond_seq: null`),
//    so `c` is zero, and the two bias-free SwiGLU transitions leave it zero. The adaLN's `norm_cond`
//    DOES have a bias, so the normed conditioning is that bias broadcast over every token, and each
//    layer's gamma and beta are one vector each - computed once, not per token and not per layer.
#pragma once
#include "trunk.cuh"

namespace rfd4p {

struct DecDims {
  int n = 0, D = 0, C = 0, H = 0, nlayers = 0;
  int nTypes = 147, latent = 32, fdim = 0;
  int dimHead() const { return D / H; }
};

inline std::string dwn(const std::string& name) { return "w/autoencoder.decoder." + name; }

// The decoder's input features, `[n, 3 + 32 + 32 + nTypes]` in `feats_seq` order: the CA position in
// nanometres, the latent, the token type's embedding, and the conditioned residue type's - the last
// left at zero for a token whose sequence is still `<M>`.
__global__ void decoderFeatsK(const float* __restrict__ caNm, const float* __restrict__ z,
                              const int* __restrict__ tokenType, const int* __restrict__ seq,
                              const float* __restrict__ typeEmb, const float* __restrict__ seqEmb,
                              const float* __restrict__ pad, float* __restrict__ feats, int n,
                              int latent, int nTypes, int fdim) {
  const int i = blockIdx.x;
  if (i >= n) return;
  float* row = feats + (size_t)i * fdim;
  const bool live = pad[i] != 0.f;
  for (int c = threadIdx.x; c < fdim; c += blockDim.x) {
    float v = 0.f;
    if (live) {
      if (c < 3) {
        v = caNm[i * 3 + c];
      } else if (c < 3 + latent) {
        v = z[(size_t)i * latent + (c - 3)];
      } else if (c < 3 + latent + 32) {
        v = typeEmb[(size_t)tokenType[i] * 32 + (c - 3 - latent)];
      } else if (seq[i] != 0) {                     // 0 is `<M>`, the unconditioned token
        v = seqEmb[(size_t)seq[i] * nTypes + (c - 3 - latent - 32)];
      }
    }
    row[c] = v;
  }
}

// One decoder layer. The same shape as the trunk's, with LayerNorm throughout and no pair bias.
inline void decoderLayer(float* s, const float* gammaMha, const float* betaMha, const float* gammaTr,
                         const float* betaTr, const float* scaleMha, const float* scaleTr,
                         const float* mask, const float* zeroBias, int stride, const DecDims& d,
                         const std::string& pre,
                         const std::function<void(const char*, const float*, size_t)>& tap = nullptr) {
  const size_t n = d.n;
  const int W_ = d.D;
  float* ada = scratch<float>("rfd4p.dec.ada", n * W_);
  float* attn = scratch<float>("rfd4p.dec.attn", n * W_);

  // adaLN: LayerNorm(s) without affine, scaled by sigmoid(gamma) and shifted by beta. gamma and beta
  // are one row each (the conditioning is constant), so `adaLnApplyK` broadcasts them.
  float* normed = scratch<float>("rfd4p.dec.norm", n * W_);
  layerNorm(s, normed, n, W_, nullptr, nullptr);
  adaLnApplyK<<<blocks(n * W_), 256, 0, STREAM>>>(normed, gammaMha, nullptr, betaMha, mask, ada, n,
                                                  W_, true);
  if (tap) tap("d/ada0", ada, n * W_);

  // Attention: one fused [4W, W] projection, q and k LayerNormed into the packed buffer.
  const std::string qkvgW = stacked(dwn(pre + ".mhba.mha.qkvg~"),
                                    {dwn(pre + ".mhba.mha.to_qkv.weight"),
                                     dwn(pre + ".mhba.mha.to_g.weight")});
  const std::string qkvgB = stacked(dwn(pre + ".mhba.mha.qkvgb~"),
                                    {dwn(pre + ".mhba.mha.to_qkv.bias"),
                                     dwn(pre + ".mhba.mha.to_g.bias")});
  float* nodes = scratch<float>("rfd4p.dec.nodes", n * W_);
  layerNorm(ada, nodes, n, W_, W(dwn(pre + ".mhba.mha.node_norm.weight")),
            W(dwn(pre + ".mhba.mha.node_norm.bias")));
  float* qkv = scratch<float>("rfd4p.dec.qkv", n * (size_t)W_ * 4);
  linear<float, float>(nodes, qkv, n, W_, W_ * 4, qkvgW, true);
  addBias(qkv, W(qkvgB), n, W_ * 4);

  const size_t packed = n * (size_t)W_ * 4;
  if (F32_TF32) {
    half* qkvg = scratch<half>("rfd4p.dec.qkvg16", packed);
    half* out = scratch<half>("rfd4p.dec.out16", n * W_);
    layerNorm<half>(qkv, qkvg, n, W_, W(dwn(pre + ".mhba.mha.q_layer_norm.weight")),
                    W(dwn(pre + ".mhba.mha.q_layer_norm.bias")), 1e-5f, W_ * 4, W_ * 4);
    layerNorm<half>(qkv + W_, qkvg + W_, n, W_, W(dwn(pre + ".mhba.mha.k_layer_norm.weight")),
                    W(dwn(pre + ".mhba.mha.k_layer_norm.bias")), 1e-5f, W_ * 4, W_ * 4);
    packVGK<half><<<blocks(n * W_), 256, 0, STREAM>>>(qkv, qkvg, n, W_);
    flashGrid<half>(qkvg, (const half*)zeroBias, stride, mask, out, d.n, d.H, d.dimHead(), 0, 1,
                    false, 1.f / sqrtf((float)d.dimHead()), nullptr);
    linear<half, float>(out, attn, n, W_, W_, dwn(pre + ".mhba.mha.to_out_node.weight"), true);
  } else {
    float* qkvg = scratch<float>("rfd4p.dec.qkvg", packed);
    float* out = scratch<float>("rfd4p.dec.out", n * W_);
    layerNorm(qkv, qkvg, n, W_, W(dwn(pre + ".mhba.mha.q_layer_norm.weight")),
              W(dwn(pre + ".mhba.mha.q_layer_norm.bias")), 1e-5f, W_ * 4, W_ * 4);
    layerNorm(qkv + W_, qkvg + W_, n, W_, W(dwn(pre + ".mhba.mha.k_layer_norm.weight")),
              W(dwn(pre + ".mhba.mha.k_layer_norm.bias")), 1e-5f, W_ * 4, W_ * 4);
    packVGK<float><<<blocks(n * W_), 256, 0, STREAM>>>(qkv, qkvg, n, W_);
    flashGrid<float>(qkvg, (const float*)zeroBias, stride, mask, out, d.n, d.H, d.dimHead(), 0, 1,
                     false, 1.f / sqrtf((float)d.dimHead()), nullptr);
    linear<float, float>(out, attn, n, W_, W_, dwn(pre + ".mhba.mha.to_out_node.weight"), true);
  }
  addBias(attn, W(dwn(pre + ".mhba.mha.to_out_node.bias")), n, W_);
  if (tap) tap("d/attn0", attn, n * W_);
  adaScaleAddBroadcastK<<<blocks(n * W_), 256, 0, STREAM>>>(attn, scaleMha, mask, s, n, W_);

  layerNorm(s, normed, n, W_, nullptr, nullptr);
  adaLnApplyK<<<blocks(n * W_), 256, 0, STREAM>>>(normed, gammaTr, nullptr, betaTr, mask, ada, n, W_,
                                                  true);
  swiGluTransition(ada, attn, n, W_, 4, mask, "autoencoder.decoder." + pre + ".transition.transition");
  adaScaleAddBroadcastK<<<blocks(n * W_), 256, 0, STREAM>>>(attn, scaleTr, mask, s, n, W_);
}

/**
 * Decode. `logits [n, nTypes]`, `coors [n, 111]` (37 atoms, CA-relative, nanometres) and
 * `presence [n, 37]` come back on the device; the postprocess that turns them into a structure is
 * per token and runs on the host.
 */
inline void decode(const DecDims& d, const float* caNm, const float* z, const int* tokenType,
                   const int* seq, const float* attnMask, float* logits, float* coors,
                   float* presence,
                   const std::function<void(const char*, const float*, size_t)>& tap = nullptr) {
  // 🔴 THE DECODER RUNS IN f32 EVEN UNDER --fast, AND IT IS FREE. It is ONE pass at the end of a
  // generation against the trunk's nlayers x recycles x nsteps, so its share of the wall clock is
  // a fraction of a percent - and reduced precision costs it far more than it costs the trunk,
  // because its softmax is near-uniform over every key where the trunk's is sharpened by a pair
  // bias. In f16 its stages read 1.3e-02 against 3.1e-05 in f32, for no time worth having.
  const bool wasTf32 = F32_TF32;
  F32_TF32 = false;
  CB(cublasSetMathMode(H, CUBLAS_PEDANTIC_MATH));
  struct Restore {
    bool was;
    ~Restore() {
      F32_TF32 = was;
      cublasSetMathMode(H, was ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH);
    }
  } restore{wasTf32};

  const size_t n = d.n;
  const int D = d.D, C = d.C;
  auto emit = [&](const char* name, const float* p, size_t count) { if (tap) tap(name, p, count); };

  // The conditioning: `c` is zero, so each layer's gamma and beta are one row, from `norm_cond`'s
  // bias alone. Computed once for the whole decode.
  float* zeroC = scratch<float>("rfd4p.dec.c", C);
  CK(cudaMemsetAsync(zeroC, 0, C * sizeof(float), STREAM));
  float* cond = scratch<float>("rfd4p.dec.cond", (size_t)d.nlayers * 6 * D);
  auto condAt = [&](int layer, int slot) { return cond + ((size_t)layer * 6 + slot) * D; };
  float* normedC = scratch<float>("rfd4p.dec.nc", C);
  for (int i = 0; i < d.nlayers; ++i) {
    const std::string L = "transformer_layers." + std::to_string(i);
    const char* adalns[2] = {".mhba.adaln", ".transition.adaln"};
    for (int k = 0; k < 2; ++k) {
      const std::string pre = L + adalns[k];
      layerNorm(zeroC, normedC, 1, C, W(dwn(pre + ".norm_cond.weight")), W(dwn(pre + ".norm_cond.bias")));
      linear<float, float>(normedC, condAt(i, 2 * k), 1, C, D, dwn(pre + ".to_gamma.0.weight"), true);
      addBias(condAt(i, 2 * k), W(dwn(pre + ".to_gamma.0.bias")), 1, D);
      linear<float, float>(normedC, condAt(i, 2 * k + 1), 1, C, D, dwn(pre + ".to_beta.weight"), true);
    }
    const char* scales[2] = {".mhba.scale_output", ".transition.scale_output"};
    for (int k = 0; k < 2; ++k) {
      linear<float, float>(zeroC, condAt(i, 4 + k), 1, C, D,
                           dwn(L + scales[k] + ".to_adaln_zero_gamma.0.weight"), true);
      addBias(condAt(i, 4 + k), W(dwn(L + scales[k] + ".to_adaln_zero_gamma.0.bias")), 1, D);
    }
  }

  float* feats = scratch<float>("rfd4p.dec.feats", n * (size_t)d.fdim);
  decoderFeatsK<<<(unsigned)n, 256, 0, STREAM>>>(
      caNm, z, tokenType, seq, W(dwn("init_repr_factory.feat_creators.2.embedding.weight")),
      W(dwn("init_repr_factory.feat_creators.3.embedding.weight")), attnMask, feats, d.n, d.latent,
      d.nTypes, d.fdim);
  emit("d/feats", feats, n * (size_t)d.fdim);

  float* s = scratch<float>("rfd4p.dec.s", n * D);
  linear<float, float>(feats, s, n, d.fdim, D, dwn("init_repr_factory.linear_out.weight"), true);
  maskRowsK<<<blocks(n * D), 256, 0, STREAM>>>(s, attnMask, n, D);
  emit("d/sInit", s, n * D);

  // flash needs a bias pointer even where the model has none; one zeroed plane serves every layer.
  const int stride = (d.n + 7) / 8 * 8;
  const size_t biasFloats = (size_t)d.H * d.n * stride;
  void* zeroBias;
  if (F32_TF32) {
    zeroBias = scratch<half>("rfd4p.dec.bias16", biasFloats);
    CK(cudaMemsetAsync(zeroBias, 0, biasFloats * sizeof(half), STREAM));
  } else {
    zeroBias = scratch<float>("rfd4p.dec.bias", biasFloats);
    CK(cudaMemsetAsync(zeroBias, 0, biasFloats * sizeof(float), STREAM));
  }

  for (int i = 0; i < d.nlayers; ++i) {
    decoderLayer(s, condAt(i, 0), condAt(i, 1), condAt(i, 2), condAt(i, 3), condAt(i, 4),
                 condAt(i, 5), attnMask, (const float*)zeroBias, stride, d,
                 "transformer_layers." + std::to_string(i), i == 0 ? tap : nullptr);
    emit(("d/s" + std::to_string(i)).c_str(), s, n * D);
  }

  // Three heads, each a LayerNorm with a bias and a projection.
  float* h = scratch<float>("rfd4p.dec.h", n * D);
  const std::pair<const char*, std::pair<float*, int>> heads[3] = {
      {"logit_linear", {logits, d.nTypes}}, {"struct_linear", {coors, 111}},
      {"atom_presence_linear", {presence, 37}}};
  for (const auto& [name, out] : heads) {
    layerNorm(s, h, n, D, W(dwn(std::string(name) + ".0.weight")), W(dwn(std::string(name) + ".0.bias")));
    linear<float, float>(h, out.first, n, D, out.second, dwn(std::string(name) + ".1.weight"), true);
  }
  emit("d/logits", logits, n * (size_t)d.nTypes);
  emit("d/coors", coors, n * 111);
  emit("d/presence", presence, n * 37);
}

}  // namespace rfd4p
