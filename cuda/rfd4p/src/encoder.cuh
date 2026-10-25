// RFD4-Proteina's autoencoder ENCODER on CUDA: all-atom coordinates in, token-level local latents
// out. Held to `cpu/rfd4p/encoder.js` (`encodeLatentsCpu`).
//
// It is the DECODER's transformer run the other way round - LayerNorm throughout, no pair bias, a
// conditioning that is identically zero - so the layer and the conditioning are `decoder.cuh`'s,
// with the weight prefix as the only difference. What is its own is the two ends:
//
// 🔴 THE FEATURES ARE PER ATOM, NOT PER TOKEN. 1,587 columns a token, six blocks, and every one of
//    them is gated on `atom37_mask`: a slot that is not there contributes ZERO rather than an
//    embedding of nothing. The relative block subtracts the token's own CA, and subtracts it only
//    where the CA itself is present.
// 🔴 AND THE HEAD IS A REPARAMETERISATION, which is where the sampler's randomness enters the
//    partial-diffusion path: `z = mean + eps * exp(logScale)`, masked to the emitting tokens. The
//    draw is the SAMPLER's stream - `encodeLatentsCpu` is handed the generator's own rng and
//    consumes `n * 32` from it - so the port replays it from the same recorded stream rather than
//    redrawing, for the reason `sampler.cuh` gives.
#pragma once
#include "decoder.cuh"

namespace rfd4p {

struct EncDims {
  int n = 0, D = 0, C = 0, H = 0, nlayers = 0;
  int latent = 32, fdim = 0, ttDim = 32, chargeDim = 16;
  int dimHead() const { return D / H; }
  int elChargeSlot() const { return 8 + chargeDim; }
};

/**
 * `encoderSeqFeatures`: `[n, fdim]` in `feats_seq` order - the token type's embedding, the 37 atom
 * positions and their mask, the same again relative to the token's CA, the chain index, the
 * incident bond-type counts, and an element-plus-charge embedding a slot.
 *
 * One block a token, because every column of a row reads the same 37-slot window.
 */
__global__ void encoderFeatsK(const float* __restrict__ coorsNm, const float* __restrict__ atomMask,
                              const int* __restrict__ tokenType, const float* __restrict__ chainIid,
                              const float* __restrict__ bondCounts,
                              const int* __restrict__ element, const int* __restrict__ charge,
                              const float* __restrict__ pad, const float* __restrict__ ttEmb,
                              const float* __restrict__ elEmb, const float* __restrict__ chEmb,
                              float* __restrict__ feats, int n, int fdim, int ttDim, int chargeDim,
                              int offAbs, int offRel, int offChain, int offBonds, int offElCh) {
  const int i = blockIdx.x;
  if (i >= n) return;
  float* row = feats + (size_t)i * fdim;
  for (int c = threadIdx.x; c < fdim; c += blockDim.x) row[c] = 0.f;
  __syncthreads();
  if (pad[i] == 0.f) return;

  const int slotDim = 8 + chargeDim;
  // The CA is subtracted only where it is present, which is what makes the relative block zero
  // rather than a position measured from the origin on a token with no backbone.
  const bool caLive = atomMask[(size_t)i * 37 + 1] != 0.f;
  const float ca0 = caLive ? coorsNm[((size_t)i * 37 + 1) * 3] : 0.f;
  const float ca1 = caLive ? coorsNm[((size_t)i * 37 + 1) * 3 + 1] : 0.f;
  const float ca2 = caLive ? coorsNm[((size_t)i * 37 + 1) * 3 + 2] : 0.f;

  for (int c = threadIdx.x; c < ttDim; c += blockDim.x) {
    row[c] = ttEmb[(size_t)tokenType[i] * ttDim + c];
  }
  if (threadIdx.x == 0 && offChain >= 0) row[offChain] = chainIid[i];

  for (int a = threadIdx.x; a < 37; a += blockDim.x) {
    const size_t slot = (size_t)i * 37 + a;
    const bool live = atomMask[slot] != 0.f;
    if (offAbs >= 0) {
      if (live) for (int k = 0; k < 3; ++k) row[offAbs + a * 3 + k] = coorsNm[slot * 3 + k];
      row[offAbs + 111 + a] = live ? 1.f : 0.f;
    }
    if (offRel >= 0) {
      if (live) {
        row[offRel + a * 3] = coorsNm[slot * 3] - ca0;
        row[offRel + a * 3 + 1] = coorsNm[slot * 3 + 1] - ca1;
        row[offRel + a * 3 + 2] = coorsNm[slot * 3 + 2] - ca2;
      }
      row[offRel + 111 + a] = live ? 1.f : 0.f;
    }
    if (!live) continue;
    if (offBonds >= 0) {
      for (int k = 0; k < 10; ++k) row[offBonds + a * 10 + k] = bondCounts[slot * 10 + k];
    }
    if (offElCh >= 0) {
      const int ch = min(10, max(0, charge[slot] + 4));
      float* dst = row + offElCh + a * slotDim;
      for (int k = 0; k < 8; ++k) dst[k] = elEmb[(size_t)element[slot] * 8 + k];
      for (int k = 0; k < chargeDim; ++k) dst[8 + k] = chEmb[(size_t)ch * chargeDim + k];
    }
  }
}

// `encoderPostprocess`: split the head's `[n, 2 * latent]` and reparameterise. `eps` null is the
// deterministic arm, where z is the mean.
__global__ void encoderSampleK(const float* __restrict__ flat, const float* __restrict__ eps,
                               const float* __restrict__ emit, float* __restrict__ mean,
                               float* __restrict__ logScale, float* __restrict__ z, int n,
                               int latent) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n * latent) return;
  const int tok = (int)(i / latent), k = (int)(i % latent);
  if (emit[tok] == 0.f) { mean[i] = logScale[i] = z[i] = 0.f; return; }
  const float m = flat[(size_t)tok * 2 * latent + k];
  const float ls = flat[(size_t)tok * 2 * latent + latent + k];
  mean[i] = m;
  logScale[i] = ls;
  z[i] = eps ? m + eps[i] * expf(ls) : m;
}

/**
 * Encode. `mean`, `logScale` and `zLatent`, each `[n, latent]`, come back on the device.
 * `eps` may be null for the deterministic arm.
 *
 * 🔴 IT RUNS IN f32 EVEN UNDER --fast, for the DECODER's reason: it is one pass at the start of a
 * partial-diffusion run against the trunk's nlayers x recycles x nsteps, and its attention has no
 * pair bias to sharpen its softmax, so reduced precision costs it far more than it saves.
 */
inline void encode(const EncDims& d, const float* coorsNm, const float* atomMask,
                   const int* tokenType, const float* chainIid, const float* bondCounts,
                   const int* element, const int* charge, const float* pad, const float* attnMask,
                   const float* emit, const float* eps, float* mean, float* logScale, float* z,
                   const int* offsets,
                   const std::function<void(const char*, const float*, size_t)>& tap = nullptr) {
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

  const std::string root = "autoencoder.encoder";
  auto ewn = [&](const std::string& name) { return aew(root, name); };
  const size_t n = d.n;
  const int D = d.D, C = d.C;
  auto emitTap = [&](const char* name, const float* p, size_t count) { if (tap) tap(name, p, count); };

  float* cond = aeConditioning(root, d.nlayers, C, D, "enc");
  auto condAt = [&](int layer, int slot) { return cond + ((size_t)layer * 6 + slot) * D; };

  float* feats = scratch<float>("rfd4p.enc.feats", n * (size_t)d.fdim);
  encoderFeatsK<<<(unsigned)n, 256, 0, STREAM>>>(
      coorsNm, atomMask, tokenType, chainIid, bondCounts, element, charge, pad,
      W(ewn("init_repr_factory.feat_creators." + std::to_string(offsets[5]) + ".embedding.weight")),
      W(ewn("init_repr_factory.feat_creators." + std::to_string(offsets[6]) +
            ".element_embedding.weight")),
      W(ewn("init_repr_factory.feat_creators." + std::to_string(offsets[6]) +
            ".charge_embedding.weight")),
      feats, d.n, d.fdim, d.ttDim, d.chargeDim, offsets[0], offsets[1], offsets[2], offsets[3],
      offsets[4]);
  emitTap("e/feats", feats, n * (size_t)d.fdim);

  float* s = scratch<float>("rfd4p.enc.s", n * D);
  linear<float, float>(feats, s, n, d.fdim, D, ewn("init_repr_factory.linear_out.weight"), true);
  maskRowsK<<<blocks(n * D), 256, 0, STREAM>>>(s, attnMask, n, D);
  emitTap("e/sInit", s, n * D);

  const int stride = (d.n + 7) / 8 * 8;
  const size_t biasFloats = (size_t)d.H * d.n * stride;
  float* zeroBias = scratch<float>("rfd4p.enc.bias", biasFloats);
  CK(cudaMemsetAsync(zeroBias, 0, biasFloats * sizeof(float), STREAM));

  DecDims ld;
  ld.n = d.n; ld.D = d.D; ld.C = d.C; ld.H = d.H; ld.nlayers = d.nlayers;
  for (int i = 0; i < d.nlayers; ++i) {
    aeLayer(root, s, condAt(i, 0), condAt(i, 1), condAt(i, 2), condAt(i, 3), condAt(i, 4),
            condAt(i, 5), attnMask, zeroBias, stride, ld,
            "transformer_layers." + std::to_string(i));
    emitTap(("e/s" + std::to_string(i)).c_str(), s, n * D);
  }

  float* h = scratch<float>("rfd4p.enc.h", n * D);
  float* flat = scratch<float>("rfd4p.enc.flat", n * (size_t)d.latent * 2);
  layerNorm(s, h, n, D, W(ewn("latent_decoder_mean_n_log_scale.0.weight")),
            W(ewn("latent_decoder_mean_n_log_scale.0.bias")));
  linear<float, float>(h, flat, n, D, d.latent * 2,
                       ewn("latent_decoder_mean_n_log_scale.1.weight"), true);
  emitTap("e/flat", flat, n * (size_t)d.latent * 2);

  encoderSampleK<<<blocks(n * (size_t)d.latent), 256, 0, STREAM>>>(flat, eps, emit, mean, logScale,
                                                                   z, d.n, d.latent);
  emitTap("e/mean", mean, n * (size_t)d.latent);
  emitTap("e/zLatent", z, n * (size_t)d.latent);
}

}  // namespace rfd4p
