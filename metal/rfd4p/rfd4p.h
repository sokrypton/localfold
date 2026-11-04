// RFD4-Proteina natively on Metal: the design model `webgpu/rfd4p/` runs in the page and `cuda/rfd4p/` runs on
// NVIDIA, written for Apple's GPUs against metal/core. One port's worth of names and layouts, read from the PAGE's
// published int8 bundle (`model-rfd4p-int8`: manifest.json and its shards), which metal/core's `loadBundle` already
// decodes - its `dtype: int8` with `block: 64` and a float16 scale a block is the same codec the fold bundles use.
//
// 🔴 THE REFERENCE IS THE BUNDLE'S OWN ARITHMETIC, NOT THE VENDOR CHECKPOINT. `--oracle=` reads what
// `cuda/rfd4p/export_input.mjs --oracle=` wrote from `cpu/rfd4p/reference.js` over these same weights
// (tools/rfd4p-bundle-to-safetensors.mjs dequantises the shards for it), so the quantisation is on both sides of
// every comparison and cancels: a residual is this port's, not the weights'. It is the argument metal/bioemu2 makes
// for holding itself to the page's int5 bundle, and it is what makes a Mac with no 1.7 GB checkpoint able to gate.
#pragma once
#include "core.h"
#include "kernels.h"
#include "model.h"
#include <string>

using namespace mt;

// The trunk's widths. Read off the weights rather than from config.json, and checked: a bundle whose shapes disagree
// with what these kernels are built for is refused by name (`checkDims`).
struct Dims {
  int n = 0;            // tokens (the input's meta/n)
  int nRecycle = 1;
  int D = 768;          // token_dim
  int P = 384;          // pair_repr_dim
  int C = 256;          // dim_cond
  int H = 12;           // nheads
  int T = 256;          // tri_mult_c
  int layers = 14;
  int seqDim = 0;       // the sequence features' width (897 at this checkpoint)
  int condDim = 0;      // 609
  int pairDim = 0;      // 107
  int pairCondDim = 0;  // 512
};
extern Dims D_;

// ---------------------------------------------------------------- weights.mm
// the bundle under "w/", every tensor float32 (float16 where `--f16` asks and a GEMM can take it)
void loadWeights(const std::string& dir);
void checkDims();
extern bool HALF;

// ---------------------------------------------------------------- ops.mm
// Y[rows, out] = X[rows, in] W[out, in]^T (+ bias) - torch's layout, which every tensor in this checkpoint uses
void lin(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, const float* bias = nullptr,
         bool relu = false);
// a tensor that may not be in this checkpoint (`tri_mult_bias: false` leaves the triangle's biases out, and the
// CPU reference reads every one of them optionally): its float32 bytes, or null
const float* opt(const std::string& name);
// y = x / sqrt(mean(x^2) + eps) * w, w optional (`nn.*_norm.weight`); rmsNormH also writes the halves
void rmsNorm(const float* x, float* y, size_t rows, int C, const float* w, float eps = 1e-5f);
// g[r][i] = silu(h[r][I + i]) * h[r][i], from the one [rows][2I] product `swish_linear.0` makes
void swiglu(const float* h, float* g, size_t rows, int I);
// one SwiGLU transition (`transition_c_1`, `transition_c_2`): dimInner = dim * expansion, no bias, no mask
void transition(const float* x, float* y, size_t rows, int dim, int expansion, const std::string& prefix, float* hid, float* gated);
// AdaptiveLayerNorm against a conditioning row per row: y = RMS(x) * sigmoid(gamma) + beta, where gamma and beta
// are that layer's projections of RMSNorm(cond, norm_cond.weight) - `prefix` is the adaLN's (e.g. "....mhba.adaln")
void adalnRows(const float* x, const float* cond, const std::string& prefix, float* y, size_t rows, int dim);
// x += y * sigmoid(scale_output's gamma of `cond`) - AdaptiveOutputScale and the block's residual
void scaleAdd(float* x, const float* y, const float* cond, const std::string& prefix, size_t rows, int dim);
// the largest n this port's float32 attention takes (its scores live in threadgroup memory)
constexpr int LF_ATTN_MAX_N = 6144;

// ---------------------------------------------------------------- trunk.mm
// c = transition_c_2(transition_c_1(cond_factory(features))) over `rows` conditioning rows
void condC(const float* condFeats, float* c, size_t rows);
// s_init = init_repr_factory(seq features)
void initSingle(const float* seqFeats, float* sInit);
// z_init = AdaLN(RMSNorm(pair_repr_builder(pair features)), RMSNorm(pair cond)) - the pair conditioning is one row
void initPair(const float* pairFeats, const float* pairCondFeats, float* zInit);
// one `MultiheadAttnAndTransition` block, in place on `s`: the pair-biased gated attention under its adaLN and
// output scale, then the SwiGLU transition under its own, each with a residual
void tokenLayer(int layer, float* s, const float* z, const float* c);
// one `PairReprUpdate`, in place on `z`: the hadamard pair write from the single track, the outgoing and incoming
// triangle multiplications, and the pair transition - each with a residual
void pairUpdate(int layer, const float* s, float* z);
// the finished trunk's outputs: the alpha carbons (3), the local latents (32), the sequence logits (147), and the
// insertion logits (n x n, or null for none)
void heads(const float* s, float* bbCa, float* localLatents, float* logits, float* insertion);
// one cycle's input: s = s_init + Lin(RMSNorm(s)), z = z_init + Lin(RMSNorm(z))
void recycleIn(const float* sInit, const float* zInit, float* s, float* z);
