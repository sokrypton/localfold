// RFD4-Proteina's trunk (`local_latents_transformer`) on CUDA, held to `cpu/rfd4p/reference.js`.
//
// Per cycle, over `nlayers` layers:
//   s = s + adaScale(PairBiasAttention(adaLN(s, c)), c)         `MultiheadAttnAndTransition`
//   s = s + adaScale(SwiGLU(adaLN(s, c)), c)
//   every `update_pair_repr_every_n` layers, and never on the last:
//   z = PairReprUpdate(s, z)   = hadamard write, triangle out, triangle in, pair transition
//
// 🔴 EVERY NORM HERE IS RMSNorm EXCEPT THE PAIR TRACK'S, WHICH ARE LayerNorm. The trunk's `norm_type`
// is `rmsnorm_torch` and that is what the single track, the conditioning, the QK norm and z_init use;
// the triangle's `layer_norm_in`/`layer_norm_out` and the pair transition's `layer_norm` are genuine
// mean-and-variance LayerNorms with a bias. Mixing them up changes nothing structural and every number.
//
// 🔴 AND THE ATTENTION RUNS ON THE f32 FLASH KERNEL, DELIBERATELY. `flashGrid<float>` is the precise
// reference path (one thread a query, no log2 domain, no f16 rounding), which is what a first parity
// pass wants; the f16 kernels need the bias pre-multiplied by LOG2E and land ~1e-3 from the CPU, which
// is a speed decision to take against a gate that already passes, not before one exists.
#pragma once
#include "ops.cuh"
#include "../../af3/src/flash.cuh"

namespace rfd4p {

struct Dims {
  int n = 0, D = 0, P = 0, C = 0, PC = 0, TRI = 0, H = 0, nlayers = 0, everyN = 2;
  int seqF = 0, condF = 0, pairF = 0, pairCondF = 0, classes = 147, latent = 32;
  bool useTriMult = true, updatePairRepr = true;
  size_t pairs() const { return (size_t)n * n; }
  int dimHead() const { return D / H; }
};

struct Trunk {
  Dims d;
  const float* mask = nullptr;      // [n]
  const float* pairMask = nullptr;  // [n * n]
  // The finished trunk.
  float* bbCa = nullptr;            // [n, 3]
  float* localLatents = nullptr;    // [n, latent]
  float* logits = nullptr;          // [n, classes]
  float* insertion = nullptr;       // [n, n], only when asked
  float* s = nullptr;               // [n, D]
  void* z = nullptr;                // [n * n, P], f32 or half (see `forward`)
};

inline std::string wn(const std::string& name) { return "w/" + name; }
// A weight the checkpoint may not carry: the triangle's LayerNorms have no bias at all under this
// config (`tri_mult_bias: false`), and the reference reads them with `getOptionalTensorData`.
inline const float* optW(const std::string& name) { return hasW(name) ? W(name) : nullptr; }

// Several `[out, in]` weights stacked along `out`, built once and cached under `key`. A row-major
// `[out, in]` matrix stacks by plain concatenation, so this is a copy and not a layout change - which
// is the whole reason the exporter leaves the checkpoint's orientation alone.
inline std::string stacked(const std::string& key, const std::vector<std::string>& parts) {
  static std::set<std::string> built;
  if (!built.count(key)) {
    size_t total = 0;
    for (const auto& part : parts) total += lenW(part);
    float* w = dalloc(total);
    size_t at = 0;
    for (const auto& part : parts) {
      CK(cudaMemcpyAsync(w + at, W(part), lenW(part) * sizeof(float), cudaMemcpyDeviceToDevice, STREAM));
      at += lenW(part);
    }
    CK(cudaStreamSynchronize(STREAM));
    deviceWeight(key, w, total);
    built.insert(key);
  }
  return key;
}

// ---------------------------------------------------------------- the two conditioned wrappers
//
// `out = RMSNorm(x) * sigmoid(gamma) + beta`, gamma and beta two linears of `RMSNorm(cond; norm_cond)`.
// `condRows` is `n` for the token track and 1 for the pair track's single broadcast row.
inline void adaLn(const float* x, const float* cond, float* out, size_t rows, int C, size_t condRows,
                  int condC, const float* rowMask, const std::string& pre) {
  float* normedX = scratch<float>("rfd4p.ada.x", rows * C);
  float* normedC = scratch<float>("rfd4p.ada.c", condRows * condC);
  float* gamma = scratch<float>("rfd4p.ada.g", condRows * C);
  float* beta = scratch<float>("rfd4p.ada.b", condRows * C);
  rmsNorm(x, normedX, rows, C, nullptr);
  rmsNorm(cond, normedC, condRows, condC, W(wn(pre + ".norm_cond.weight")));
  linear<float, float>(normedC, gamma, condRows, condC, C, wn(pre + ".to_gamma.0.weight"), true);
  linear<float, float>(normedC, beta, condRows, condC, C, wn(pre + ".to_beta.weight"), true);
  adaLnApplyK<<<blocks(rows * C), 256, 0, STREAM>>>(normedX, gamma, W(wn(pre + ".to_gamma.0.bias")),
                                                    beta, rowMask, out, rows, C, condRows == 1);
}

// `out = x * sigmoid(cond W + b)` - the residual's gate, zero-initialised at training time.
// ---------------------------------------------------------------- the conditioning, hoisted
//
// 🔴 EVERY ONE OF THESE PROJECTIONS IS A FUNCTION OF `c` ALONE, AND `c` DOES NOT CHANGE INSIDE THE
// RECYCLE LOOP. Cycle 0 uses c0 and every later cycle uses c1, so at six cycles the per-layer gammas
// and betas were being recomputed six times to the same numbers - 504 GEMMs a pass where 168 will do.
// They are computed once per conditioning and read by every cycle.
//
// Six tensors a layer: the two adaLNs' gamma and beta, and the two output scales' gamma. The adaLNs
// project a NORMED c (each site has its own `norm_cond.weight`); the output scales project c raw.
struct CondPack {
  float* buf = nullptr;
  size_t n = 0;
  int D = 0;
  float* at(int layer, int slot) const { return buf + ((size_t)layer * 6 + slot) * n * D; }
};

inline void buildCondPack(const float* c, const Dims& d, const char* slot, CondPack& pack) {
  const size_t n = d.n;
  const int D = d.D, C = d.C;
  pack.n = n;
  pack.D = D;
  pack.buf = scratch<float>(slot, (size_t)d.nlayers * 6 * n * D);
  float* normedC = scratch<float>("rfd4p.cp.nc", n * C);
  for (int i = 0; i < d.nlayers; ++i) {
    const std::string L = "nn.transformer_layers." + std::to_string(i);
    const char* adalns[2] = {".mhba.adaln", ".transition.adaln"};
    for (int k = 0; k < 2; ++k) {
      const std::string pre = L + adalns[k];
      rmsNorm(c, normedC, n, C, W(wn(pre + ".norm_cond.weight")));
      linear<float, float>(normedC, pack.at(i, 2 * k), n, C, D, wn(pre + ".to_gamma.0.weight"), true);
      linear<float, float>(normedC, pack.at(i, 2 * k + 1), n, C, D, wn(pre + ".to_beta.weight"), true);
    }
    const char* scales[2] = {".mhba.scale_output", ".transition.scale_output"};
    for (int k = 0; k < 2; ++k) {
      linear<float, float>(c, pack.at(i, 4 + k), n, C, D,
                           wn(L + scales[k] + ".to_adaln_zero_gamma.0.weight"), true);
    }
  }
}

// adaLN with its gamma and beta already projected.
inline void adaLnPacked(const float* x, const float* gamma, const float* beta, const float* gammaBias,
                        float* out, size_t rows, int C, const float* rowMask) {
  float* normedX = scratch<float>("rfd4p.ada.x", rows * C);
  rmsNorm(x, normedX, rows, C, nullptr);
  adaLnApplyK<<<blocks(rows * C), 256, 0, STREAM>>>(normedX, gamma, gammaBias, beta, rowMask, out,
                                                    rows, C, false);
}

// `silu(gates) * xPart` through `[dim -> 2 * inner]` and back, the masked rows zeroed. Replaces its
// input (the conditioning's two transitions) or feeds a residual (the token transition) - the caller
// decides; this writes `out` and adds nothing.
inline void swiGluTransition(const float* x, float* out, size_t rows, int dim, int expansion,
                             const float* rowMask, const std::string& pre) {
  const int inner = dim * expansion;
  float* wide = scratch<float>("rfd4p.sw.wide", rows * (size_t)inner * 2);
  float* gated = scratch<float>("rfd4p.sw.gated", rows * (size_t)inner);
  linear<float, float>(x, wide, rows, dim, inner * 2, wn(pre + ".swish_linear.0.weight"), true);
  swiGluSecondK<<<blocks(rows * inner), 256, 0, STREAM>>>(wide, gated, rows, inner);
  linear<float, float>(gated, out, rows, inner, dim, wn(pre + ".linear_out.weight"), true);
  if (rowMask) maskRowsK<<<blocks(rows * dim), 256, 0, STREAM>>>(out, rowMask, rows, dim);
}

// ---------------------------------------------------------------- pair-biased attention
//
// 🔴 THE QK NORM IS OVER THE WHOLE 768, NOT PER HEAD. `pair_bias_attn.py` norms the concatenated
// projection before it is split into twelve heads of 64, so a per-head RMSNorm - the obvious reading,
// and what most ports do - is a different function.
template <class TZ>
inline void pairBiasAttention(const float* nodes, const TZ* pair, const float* pairMask,
                              const float* tokenMask, float* out, const Dims& d,
                              const std::string& pre) {
  const size_t n = d.n;
  const int W_ = d.D;
  float* normed = scratch<float>("rfd4p.at.norm", n * W_);
  rmsNorm(nodes, normed, n, W_, W(wn(pre + ".node_norm.weight")));

  // q, k, v and the gate are four projections of one normed input, so they are one [4W, W] GEMM and
  // one bias - and the result is already [q | k | v | g] per token, which is flash's own layout.
  const std::string qkvgW = stacked(wn(pre + ".qkvg~"),
                                     {wn(pre + ".to_qkv.weight"), wn(pre + ".to_g.weight")});
  const std::string qkvgB = stacked(wn(pre + ".qkvgb~"),
                                     {wn(pre + ".to_qkv.bias"), wn(pre + ".to_g.bias")});
  float* qkv = scratch<float>("rfd4p.at.qkv", n * (size_t)W_ * 4);
  linear<float, float>(normed, qkv, n, W_, W_ * 4, qkvgW, true);
  addBias(qkv, W(qkvgB), n, W_ * 4);

  // The pair bias: RMSNorm, project to `heads`, and lay out for flash, in one kernel.
  const int stride = (d.n + 7) / 8 * 8;

  // 🔴 THE f32 KERNEL DEREFERENCES `mask` WITHOUT CHECKING IT (flash.cuh:1061), where the f16 ones
  // take a MASKED template bool - so a null here is an illegal access, not an unmasked attention. It
  // is a KEY mask, `mask[row * n + j]`, and with one grid row that is the token mask itself; the
  // reference's per-PAIR masking rides in the bias above.
  //
  // The f16 arm is the whole point of --fast here: the f32 kernel is one thread a query and was
  // 11.9% of the pass, where this one is on the tensor cores. It is the same call with half operands
  // and the bias in the log2 domain.
  const int biasFloats = (int)((size_t)d.H * d.n * stride);
  const size_t packed = n * (size_t)W_ * 4;
  if (F32_TF32) {
    half* qkvg = scratch<half>("rfd4p.at.qkvg16", packed);
    half* bias = scratch<half>("rfd4p.at.bias16", biasFloats);
    half* attn = scratch<half>("rfd4p.at.out16", n * W_);
    rmsNorm<half>(qkv, qkvg, n, W_, W(wn(pre + ".q_layer_norm.weight")), 1e-5f, W_ * 4, W_ * 4);
    rmsNorm<half>(qkv + W_, qkvg + W_, n, W_, W(wn(pre + ".k_layer_norm.weight")), 1e-5f, W_ * 4, W_ * 4);
    packVGK<half><<<blocks(n * W_), 256, 0, STREAM>>>(qkv, qkvg, n, W_);
    pairBias<TZ, half>(pair, wn(pre + ".pair_norm.weight"), wn(pre + ".to_bias.weight"), pairMask,
                       bias, d.n, stride, d.H, d.P, LOG2E);
    flashGrid<half>(qkvg, bias, stride, tokenMask, attn, d.n, d.H, d.dimHead(), 0, 1, false,
                    1.f / sqrtf((float)d.dimHead()), nullptr);
    linear<half, float>(attn, out, n, W_, W_, wn(pre + ".to_out_node.weight"), true);
  } else {
    float* qkvg = scratch<float>("rfd4p.at.qkvg", packed);
    float* bias = scratch<float>("rfd4p.at.bias", biasFloats);
    float* attn = scratch<float>("rfd4p.at.out", n * W_);
    rmsNorm(qkv, qkvg, n, W_, W(wn(pre + ".q_layer_norm.weight")), 1e-5f, W_ * 4, W_ * 4);
    rmsNorm(qkv + W_, qkvg + W_, n, W_, W(wn(pre + ".k_layer_norm.weight")), 1e-5f, W_ * 4, W_ * 4);
    packVGK<float><<<blocks(n * W_), 256, 0, STREAM>>>(qkv, qkvg, n, W_);
    pairBias<TZ, float>(pair, wn(pre + ".pair_norm.weight"), wn(pre + ".to_bias.weight"), pairMask,
                        bias, d.n, stride, d.H, d.P, 1.f);
    flashGrid<float>(qkvg, bias, stride, tokenMask, attn, d.n, d.H, d.dimHead(), 0, 1, false,
                     1.f / sqrtf((float)d.dimHead()), nullptr);
    linear<float, float>(attn, out, n, W_, W_, wn(pre + ".to_out_node.weight"), true);
  }
  addBias(out, W(wn(pre + ".to_out_node.bias")), n, W_);
}

// ---------------------------------------------------------------- the triangle
//
// 🔴 ITS HIDDEN WIDTH IS NOT THE PAIR WIDTH, WHICH IS WHY NEITHER ef2's NOR af3's TRIANGLE FITS.
// Both assume `projection` is `(C, 2C)` and contract at the pair width; RFD4 projects 384 down to 256
// and comes back up, so the contraction is at 256 and the gate at 384.
//
// The contraction without materialising O(n^3): one `n x n` GEMM a channel, the channel axis being the
// batch. `a` is `[i][k][TRI]`, so channel `c`'s matrix is column-major with `lda = TRI` and the batch
// stride is 1 - the channels interleave, which is legal and saves transposing into planes the way AF3
// does. Outgoing is `x[i][j] = sum_k a[i][k] b[j][k]`, incoming `sum_k a[k][i] b[k][j]`, and the only
// difference is which operand is transposed.
// 🔴 THE PLANES ARE HALF UNDER --fast, AND THEY ARE THE BIGGEST INTERMEDIATE IN THE PORT: five
// projections of [pairs, 384] at n = 100 is 1408 x 10,000 floats, 56 MB written and read back once a
// triangle. The GEMMs around them are at 144 TFLOP/s against this card's 156 TF32 ceiling, so what is
// left to win is bytes, not arithmetic. The contraction still accumulates in f32 - a sum over n of
// products is where half overflows, which is the one place AF3's pair track keeps f32 too.
template <class TZ, class TP, bool LONG>
inline void triangleT(TZ* z, const float* pairMask, const Dims& d, bool outgoing,
                      const std::string& pre) {
  // On a long chain the contraction's output is narrowed too, and the gate is transposed before it
  // is read. Both invert with length - see the table below - so they are one decision.
  using TX = std::conditional_t<LONG, TP, float>;
  const size_t rows = d.pairs();
  const int cZ = d.P, cH = d.TRI, n = d.n;
  TP* zNorm = scratch<TP>("rfd4p.tri.zn", rows * cZ);
  layerNormIn<TZ, TP>(z, zNorm, rows, cZ, W(wn(pre + ".layer_norm_in.weight")),
                      optW(wn(pre + ".layer_norm_in.bias")));

  // 🔴 ONE PROJECTION, NOT FOUR, AND IT LANDS IN THE LAYOUT THE CONTRACTION WANTS. a_p, a_g, b_p and
  // b_g all read the same `zNorm`, so their weights stack into one `[4 * cH, cZ]` matrix - a plain
  // concatenation, because `[out, in]` row-major stacks along `out`. `linearPlanes` then writes the
  // result channel-major, which is what `toPlanesK` used to produce: that transpose was the largest
  // single kernel in the profile at 14.8%, and it is now nothing at all.
  // a_p, a_g, b_p, b_g AND the output gate all read this same zNorm, so all five stack into one
  // [4 * cH + cZ, cZ] projection - one GEMM a triangle instead of five.
  // 🔴 THE OUTPUT GATE JOINS THE FUSED PROJECTION ONLY ON A SHORT CHAIN. Channel-major it costs a
  // strided read, and dodging that costs a transpose; on a long chain both are dearer than simply
  // projecting it row-major on its own, which is also 384 fewer channels for the fused GEMM to
  // write. Short: one GEMM, strided read. Long: two GEMMs, dense read, no transpose.
  const int fusedOut = LONG ? 4 * cH : 4 * cH + cZ;
  const std::string fused =
      LONG ? stacked(wn(pre + ".abpg~"),
                     {wn(pre + ".linear_a_p.weight"), wn(pre + ".linear_a_g.weight"),
                      wn(pre + ".linear_b_p.weight"), wn(pre + ".linear_b_g.weight")})
           : stacked(wn(pre + ".abpgz~"),
                     {wn(pre + ".linear_a_p.weight"), wn(pre + ".linear_a_g.weight"),
                      wn(pre + ".linear_b_p.weight"), wn(pre + ".linear_b_g.weight"),
                      wn(pre + ".linear_g.weight")});
  TP* planes = scratch<TP>("rfd4p.tri.planes", (size_t)fusedOut * rows);
  linearPlanes<TP>(zNorm, planes, rows, cZ, fusedOut, fused);
  gatePlanes<TP>(planes, pairMask, rows, cH);
  const TP* aT = planes;
  const TP* bT = planes + 2 * rows * (size_t)cH;
  const TP* gPlanes = LONG ? nullptr : planes + 4 * rows * (size_t)cH;

  // One n x n plane a channel, contracted as a batch of GEMMs, then back. Outgoing and incoming
  // differ only in which operand is transposed.
  TX* xT = scratch<TX>("rfd4p.tri.xt", rows * cH);
  const float one = 1.f, zero = 0.f;
  constexpr cudaDataType_t TPT = std::is_same_v<TP, half> ? CUDA_R_16F : CUDA_R_32F;
  constexpr cudaDataType_t TXT = std::is_same_v<TX, half> ? CUDA_R_16F : CUDA_R_32F;
  CB(cublasGemmStridedBatchedEx(
      H, outgoing ? CUBLAS_OP_T : CUBLAS_OP_N, outgoing ? CUBLAS_OP_N : CUBLAS_OP_T, n, n, n, &one,
      bT, TPT, n, (long long)n * n, aT, TPT, n, (long long)n * n, &zero,
      xT, TXT, n, (long long)n * n, cH,
      std::is_same_v<TP, half> ? CUBLAS_COMPUTE_32F : computeType(), CUBLAS_GEMM_DEFAULT));
  // The contraction's output stays in TX all the way to its norm: `layerNormIn` takes a half input
  // where ef2's `layerNorm` cannot, which is what forced a widening transpose here before.
  TX* x = scratch<TX>("rfd4p.tri.x", rows * cH);
  fromPlanes<TX, TX>(xT, x, rows, cH);
  TP* xNorm = scratch<TP>("rfd4p.tri.xn", rows * cH);
  layerNormIn<TX, TP>(x, xNorm, rows, cH, W(wn(pre + ".layer_norm_out.weight")),
                      optW(wn(pre + ".layer_norm_out.bias")));
  // 🔴 `xProj` IS THE BIGGEST THING THE GATED ADD READS - f32 it is 245 MB at n=400 against the
  // gate's 123 and z's 123 - so it is narrowed too where the projection has no bias to add. This
  // checkpoint's `linear_z` has none; one that does keeps the f32 path rather than growing a
  // half-precision bias add nothing has measured.
  const bool zBias = hasW(wn(pre + ".linear_z.bias"));
  float* xProj32 = zBias ? scratch<float>("rfd4p.tri.xp", rows * cZ) : nullptr;
  TP* xProj16 = zBias ? nullptr : scratch<TP>("rfd4p.tri.xp16", rows * cZ);
  if (zBias) {
    linear<TP, float>(xNorm, xProj32, rows, cH, cZ, wn(pre + ".linear_z.weight"), true);
    addBias(xProj32, W(wn(pre + ".linear_z.bias")), rows, cZ);
  } else {
    linear<TP, TP>(xNorm, xProj16, rows, cH, cZ, wn(pre + ".linear_z.weight"), true);
  }
  // Straight into z: the gate and the residual it always feeds, in one pass. Safe because `zNorm`
  // was taken at the top, so nothing downstream of here reads z again. The gate arrives channel-major
  // out of the fused projection, which costs the kernel an index and saves a whole GEMM.
  // 🔴 WHETHER TO TRANSPOSE THE GATE FIRST DEPENDS ON THE LENGTH, AND IT INVERTS. The gate comes out
  // of the fused projection channel-major, so reading it in place costs a transaction an element;
  // that is cheap on a short chain and ruinous on a long one, where this kernel was 23.6% of the
  // pass and 2.7x off its own bandwidth. Measured both ways, `--fast`, 6 recycles:
  //
  //        n=100          n=200          n=400
  //   in place   49.8      147.8          555.9
  //   transposed 54.8      149.2          513.3
  //
  // so the crossover is about 40,000 pairs and the rule is on `rows`, which is what sets the stride.
  if constexpr (LONG) {
    TP* gate = scratch<TP>("rfd4p.tri.gate", rows * cZ);
    linear<TP, TP>(zNorm, gate, rows, cZ, cZ, wn(pre + ".linear_g.weight"), true);
    if (zBias) mulSigmoidAddGate<float, TP, TZ>(xProj32, gate, pairMask, z, rows, cZ);
    else mulSigmoidAddGate<TP, TP, TZ>(xProj16, gate, pairMask, z, rows, cZ);
  } else {
    if (zBias) mulSigmoidAddPlaneGate<float, TP, TZ>(xProj32, gPlanes, pairMask, z, rows, cZ);
    else mulSigmoidAddPlaneGate<TP, TP, TZ>(xProj16, gPlanes, pairMask, z, rows, cZ);
  }
}

// 🔴 TWO CHOICES IN THE TRIANGLE INVERT WITH LENGTH, AND THE SHORT ANSWER WAS RECORDED FIRST. The
// gate's strided read and the contraction's f32 output are both cheaper on a short chain and dearer
// on a long one. Measured `--fast`, 6 recycles:
//
//                 n=100     n=200     n=400
//   short form     49.9     149.2     555.9
//   long form      51.2     148.0     507.6
//
// The crossover is about 40,000 pairs. 🔴 AND THE LONG FORM IS UNREACHABLE BY THE ORACLE, whose only
// case is 54 tokens - so `LOCALFOLD_BIG` forces it at any size, exactly as cuda/af3's big-input
// paths are checked. An untested branch that only runs on inputs nobody gates is how a port grows a
// second, wrong implementation.
inline bool longForm(size_t rows) {
  static const bool forced = getenv("LOCALFOLD_BIG") != nullptr;
  return forced || rows >= 40000;
}

template <class TZ>
inline void triangle(TZ* z, const float* pairMask, const Dims& d, bool outgoing,
                     const std::string& pre) {
  using TP = std::conditional_t<std::is_same_v<TZ, half>, half, float>;
  if (longForm(d.pairs())) triangleT<TZ, TP, true>(z, pairMask, d, outgoing, pre);
  else triangleT<TZ, TP, false>(z, pairMask, d, outgoing, pre);
}

// ---------------------------------------------------------------- the pair update
//
// 🔴 THE TWO HALVES OF `linear_x` ARE INDEXED BY DIFFERENT TOKENS, AND THE OBVIOUS READING IS WRONG.
// `pair_update.py` writes `pi[:, None, :, :] * pj[:, :, None, :]`: the FIRST half is broadcast along
// the i axis, so it is indexed by **j**, and the second half by **i**. Swapping them transposes the
// write and still produces a plausible pair tensor.
template <class TZ>
__global__ void hadamardWriteK(const float* __restrict__ pij, const float* __restrict__ pairMask,
                               TZ* __restrict__ z, int n, int P) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n * n * P) return;
  size_t pair = i / P;
  int c = (int)(i - pair * P);
  if (pairMask && pairMask[pair] == 0.f) return;
  int row = (int)(pair / n), col = (int)(pair % n);
  z[i] = (TZ)((float)z[i] + pij[(size_t)col * 2 * P + c] * pij[(size_t)row * 2 * P + P + c]);
}
__global__ void hadamardWrite4K(const float4* __restrict__ pij, const float* __restrict__ pairMask,
                                float4* __restrict__ z, int n, int P4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n * n * P4) return;
  size_t pair = i / P4;
  int c = (int)(i - pair * P4);
  if (pairMask && pairMask[pair] == 0.f) return;
  int row = (int)(pair / n), col = (int)(pair % n);
  const float4 pi = pij[(size_t)col * 2 * P4 + c];
  const float4 pj = pij[(size_t)row * 2 * P4 + P4 + c];
  float4 zz = z[i];
  zz.x += pi.x * pj.x; zz.y += pi.y * pj.y; zz.z += pi.z * pj.z; zz.w += pi.w * pj.w;
  z[i] = zz;
}
__global__ void hadamardWrite4HK(const float4* __restrict__ pij, const float* __restrict__ pairMask,
                                 __half2* __restrict__ z, int n, int P4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n * n * P4) return;
  const size_t pair = i / P4;
  const int c = (int)(i - pair * P4);
  if (pairMask && pairMask[pair] == 0.f) return;
  const int row = (int)(pair / n), col = (int)(pair % n);
  const float4 pi = pij[(size_t)col * 2 * P4 + c];
  const float4 pj = pij[(size_t)row * 2 * P4 + P4 + c];
  float2 lo = __half22float2(z[2 * i]), hi = __half22float2(z[2 * i + 1]);
  lo.x += pi.x * pj.x; lo.y += pi.y * pj.y; hi.x += pi.z * pj.z; hi.y += pi.w * pj.w;
  z[2 * i] = __floats2half2_rn(lo.x, lo.y);
  z[2 * i + 1] = __floats2half2_rn(hi.x, hi.y);
}

template <class TZ>
inline void hadamardWrite(const float* pij, const float* pairMask, TZ* z, int n, int P) {
  if constexpr (std::is_same_v<TZ, half>) {
    if ((P & 3) == 0 && ((uintptr_t)pij & 15) == 0 && ((uintptr_t)z & 7) == 0) {
      hadamardWrite4HK<<<blocks((size_t)n * n * (P >> 2)), 256, 0, STREAM>>>(
          (const float4*)pij, pairMask, (__half2*)z, n, P >> 2);
      return;
    }
  }
  if constexpr (std::is_same_v<TZ, float>) {
    if ((P & 3) == 0 && ((((uintptr_t)pij | (uintptr_t)z) & 15) == 0)) {
      hadamardWrite4K<<<blocks((size_t)n * n * (P >> 2)), 256, 0, STREAM>>>(
          (const float4*)pij, pairMask, (float4*)z, n, P >> 2);
      return;
    }
  }
  hadamardWriteK<TZ><<<blocks((size_t)n * n * P), 256, 0, STREAM>>>(pij, pairMask, z, n, P);
}

template <class TZ>
inline void pairReprUpdate(const float* s, TZ* z, const float* mask, const float* pairMask,
                           const Dims& d, const std::string& pre) {
  const size_t rows = d.pairs();
  const int n = d.n, P = d.P, D = d.D;
  float* xNorm = scratch<float>("rfd4p.pu.xn", (size_t)n * D);
  layerNorm(s, xNorm, n, D, W(wn(pre + ".layer_norm_in.weight")), W(wn(pre + ".layer_norm_in.bias")));
  float* pij = scratch<float>("rfd4p.pu.pij", (size_t)n * P * 2);
  linear<float, float>(xNorm, pij, n, D, P * 2, wn(pre + ".linear_x.weight"), true);
  hadamardWrite<TZ>(pij, pairMask, z, n, P);

  if (d.useTriMult) {
    triangle<TZ>(z, pairMask, d, true, pre + ".tri_mult_out");
    triangle<TZ>(z, pairMask, d, false, pre + ".tri_mult_in");
  }

  // PairTransition: LayerNorm, up, ReLU, down, residual.
  const std::string tr = pre + ".transition_out";
  const int hidden = (int)lenW(wn(tr + ".linear_1.bias"));
  float* o = scratch<float>("rfd4p.pu.o", rows * P);
  // The widened hidden is [pairs, 768] - the largest buffer in the update after the planes - so under
  // --fast it is half, and with it the normed input the up projection reads.
  if (F32_TF32) {
    half* zn = scratch<half>("rfd4p.pu.zn16", rows * P);
    half* h = scratch<half>("rfd4p.pu.h16", rows * (size_t)hidden);
    layerNormIn<TZ, half>(z, zn, rows, P, W(wn(tr + ".layer_norm.weight")),
                          W(wn(tr + ".layer_norm.bias")));
    linear<half, half>(zn, h, rows, P, hidden, wn(tr + ".linear_1.weight"), true);
    biasRelu<half>(h, W(wn(tr + ".linear_1.bias")), rows, hidden);
    linear<half, float>(h, o, rows, hidden, P, wn(tr + ".linear_2.weight"), true);
  } else {
    float* zn = scratch<float>("rfd4p.pu.zn", rows * P);
    float* h = scratch<float>("rfd4p.pu.h", rows * (size_t)hidden);
    layerNormIn<TZ, float>(z, zn, rows, P, W(wn(tr + ".layer_norm.weight")),
                           W(wn(tr + ".layer_norm.bias")));
    linear<float, float>(zn, h, rows, P, hidden, wn(tr + ".linear_1.weight"), true);
    biasRelu<float>(h, W(wn(tr + ".linear_1.bias")), rows, hidden);
    linear<float, float>(h, o, rows, hidden, P, wn(tr + ".linear_2.weight"), true);
  }
  biasAdd<TZ>(o, W(wn(tr + ".linear_2.bias")), pairMask, z, rows, P);
}

// ---------------------------------------------------------------- one token layer
template <class TZ>
inline void tokenLayer(float* s, const TZ* z, const CondPack& cond, int layer, const float* mask,
                       const float* pairMask, const Dims& d, const std::string& pre) {
  const size_t n = d.n;
  const int D = d.D;
  float* ada = scratch<float>("rfd4p.l.ada", n * D);
  float* attn = scratch<float>("rfd4p.l.attn", n * D);

  adaLnPacked(s, cond.at(layer, 0), cond.at(layer, 1),
              W(wn(pre + ".mhba.adaln.to_gamma.0.bias")), ada, n, D, mask);
  pairBiasAttention<TZ>(ada, z, pairMask, mask, attn, d, pre + ".mhba.mha");
  adaScaleAddK<<<blocks(n * D), 256, 0, STREAM>>>(
      attn, cond.at(layer, 4), W(wn(pre + ".mhba.scale_output.to_adaln_zero_gamma.0.bias")), mask, s,
      n, D);

  adaLnPacked(s, cond.at(layer, 2), cond.at(layer, 3),
              W(wn(pre + ".transition.adaln.to_gamma.0.bias")), ada, n, D, mask);
  swiGluTransition(ada, attn, n, D, 4, mask, pre + ".transition.transition");
  adaScaleAddK<<<blocks(n * D), 256, 0, STREAM>>>(
      attn, cond.at(layer, 5),
      W(wn(pre + ".transition.scale_output.to_adaln_zero_gamma.0.bias")), mask, s, n, D);
}

// ---------------------------------------------------------------- the whole trunk
//
// `tap` is given every named stage as it is produced, which is what the --oracle arm reads; it is null
// in a fold.
// 🔴 THE PAIR TENSOR IS half UNDER --fast, AND IT IS THE MODEL'S STATE, NOT AN INTERMEDIATE. About a
// third of the pass reads or writes z, so its width sets the floor for `layerNormIn`, `pairBiasK`,
// the hadamard write, the transition's residual and both triangles' gated adds - every one of which
// is at its bandwidth already. The f32 arm keeps f32, the oracle compares in f32 either way, and the
// recycle projection and the heads are unchanged: only the storage narrows.
template <class TZ>
inline void forwardT(Trunk& t, const float* seq, const float* cond0, const float* cond1,
                     const float* pairF, const float* pairCondF, int nRecycle, bool wantInsertion,
                     const std::function<void(const char*, const float*, size_t)>& tap = nullptr) {
  const Dims& d = t.d;
  const size_t n = d.n, pairs = d.pairs();
  const int D = d.D, P = d.P, C = d.C, PC = d.PC;
  const float* mask = t.mask;
  const float* pairMask = t.pairMask;
  TZ* z = (TZ*)t.z;
  // A stage comparison is always in f32; a half pair tensor is widened into this on the way out.
  auto emit = [&](const char* name, const float* p, size_t count) { if (tap) tap(name, p, count); };
  auto emitPair = [&](const std::string& name, const TZ* p, size_t count) {
    if (!tap) return;
    if constexpr (std::is_same_v<TZ, float>) {
      tap(name.c_str(), (const float*)p, count);
    } else {
      float* wide = scratch<float>("rfd4p.emit.z", count);
      widenK<<<blocks(count), 256, 0, STREAM>>>(p, wide, count);
      tap(name.c_str(), wide, count);
    }
  };

  // c = two SwiGLU transitions of the conditioning features, with no residual between them.
  auto buildCond = [&](const float* feats, const char* slot) {
    float* c = scratch<float>(slot, n * C);
    float* tmp = scratch<float>("rfd4p.c.tmp", n * C);
    linear<float, float>(feats, c, n, d.condF, C, wn("nn.cond_factory.linear_out.weight"), true);
    swiGluTransition(c, tmp, n, C, 2, mask, "nn.transition_c_1");
    swiGluTransition(tmp, c, n, C, 2, mask, "nn.transition_c_2");
    return c;
  };
  const float* c0 = buildCond(cond0, "rfd4p.c0");
  const float* c1 = nRecycle > 1 ? buildCond(cond1, "rfd4p.c1") : c0;
  emit("o/c", c0, n * C);
  CondPack pack0, pack1;
  buildCondPack(c0, d, "rfd4p.pack0", pack0);
  if (nRecycle > 1) buildCondPack(c1, d, "rfd4p.pack1", pack1);

  float* sInit = scratch<float>("rfd4p.sInit", n * D);
  linear<float, float>(seq, sInit, n, d.seqF, D, wn("nn.init_repr_factory.linear_out.weight"), true);
  maskRowsK<<<blocks(n * D), 256, 0, STREAM>>>(sInit, mask, n, D);
  emit("o/sInit", sInit, n * D);

  float* zInit = scratch<float>("rfd4p.zInit", pairs * P);
  {
    float* raw = scratch<float>("rfd4p.zi.raw", pairs * P);
    linear<float, float>(pairF, raw, pairs, d.pairF, P,
                         wn("nn.pair_repr_builder.init_repr_factory.linear_out.weight"), true);
    rmsNorm(raw, zInit, pairs, P, W(wn("nn.pair_repr_builder.init_repr_factory.ln_out.weight")));
    maskRowsK<<<blocks(pairs * P), 256, 0, STREAM>>>(zInit, pairMask, pairs, P);
    float* zCondRaw = scratch<float>("rfd4p.zi.craw", PC);
    float* zCond = scratch<float>("rfd4p.zi.cond", PC);
    linear<float, float>(pairCondF, zCondRaw, 1, d.pairCondF, PC,
                         wn("nn.pair_repr_builder.cond_factory.linear_out.weight"), true);
    rmsNorm(zCondRaw, zCond, 1, PC, W(wn("nn.pair_repr_builder.cond_factory.ln_out.weight")));
    float* adaOut = scratch<float>("rfd4p.zi.ada", pairs * P);
    adaLn(zInit, zCond, adaOut, pairs, P, 1, PC, pairMask, "nn.pair_repr_builder.adaln");
    CK(cudaMemcpyAsync(zInit, adaOut, pairs * P * sizeof(float), cudaMemcpyDeviceToDevice, STREAM));
  }
  emit("o/zInit", zInit, pairs * P);
  TZ* zInitZ = (TZ*)zInit;
  if constexpr (std::is_same_v<TZ, half>) {
    zInitZ = scratch<half>("rfd4p.zInit16", pairs * P);
    toHalfK<<<blocks(pairs * P), 256, 0, STREAM>>>(zInit, zInitZ, pairs * P);
  }

  CK(cudaMemsetAsync(t.s, 0, n * D * sizeof(float), STREAM));
  CK(cudaMemsetAsync(z, 0, pairs * P * sizeof(TZ), STREAM));
  const int cycles = std::max(1, nRecycle);
  for (int cycle = 0; cycle < cycles; ++cycle) {
    // s = s_init + Lin(RMS(s)), z = z_init + Lin(RMS(z)); both are zero on the first cycle, so the
    // projection of a zero row is a zero row and this is s_init / z_init exactly.
    float* norm = scratch<float>("rfd4p.cy.norm", std::max(n * D, pairs * P));
    rmsNorm(t.s, norm, n, D, W(wn("nn.s_cycle_norm.weight")));
    float* delta = scratch<float>("rfd4p.cy.delta", std::max(n * D, pairs * P));
    linear<float, float>(norm, delta, n, D, D, wn("nn.s_cycle_proj.weight"), true);
    CK(cudaMemcpyAsync(t.s, sInit, n * D * sizeof(float), cudaMemcpyDeviceToDevice, STREAM));
    addIntoK<<<blocks(n * D), 256, 0, STREAM>>>(t.s, delta, n * D);

    rmsNormIn<TZ, float>(z, norm, pairs, P, W(wn("nn.z_cycle_norm.weight")), nullptr);
    linear<float, float>(norm, delta, pairs, P, P, wn("nn.z_cycle_proj.weight"), true);
    CK(cudaMemcpyAsync(z, zInitZ, pairs * P * sizeof(TZ), cudaMemcpyDeviceToDevice, STREAM));
    addIntoTK<TZ><<<blocks(pairs * P), 256, 0, STREAM>>>(z, delta, pairs * P);

    const CondPack& pack = cycle == 0 ? pack0 : pack1;
    const bool last = cycle == cycles - 1;
    for (int i = 0; i < d.nlayers; ++i) {
      tokenLayer<TZ>(t.s, z, pack, i, mask, pairMask, d,
                     "nn.transformer_layers." + std::to_string(i));
      if (d.updatePairRepr && i < d.nlayers - 1 && i % d.everyN == 0) {
        pairReprUpdate<TZ>(t.s, z, mask, pairMask, d,
                           "nn.pair_update_layers." + std::to_string(i));
      }
      if (last) {
        emit(("o/s" + std::to_string(i)).c_str(), t.s, n * D);
        emitPair("o/z" + std::to_string(i), z, pairs * P);
      }
    }
  }

  // The three heads, each a norm and a projection off the finished single track.
  float* normed = scratch<float>("rfd4p.head.n", n * D);
  rmsNorm(t.s, normed, n, D, W(wn("nn.ca_linear.0.weight")));
  linear<float, float>(normed, t.bbCa, n, D, 3, wn("nn.ca_linear.1.weight"), true);
  rmsNorm(t.s, normed, n, D, W(wn("nn.local_latents_linear.0.weight")));
  linear<float, float>(normed, t.localLatents, n, D, d.latent, wn("nn.local_latents_linear.1.weight"), true);
  rmsNorm(t.s, normed, n, D, W(wn("nn.sequence_readout.0.weight")));
  linear<float, float>(normed, t.logits, n, D, d.classes, wn("nn.sequence_readout.1.weight"), true);

  emitPair("o/z", z, pairs * P);

  // The insertion head: plain projections of s, no norm, logits q k^T / sqrt(D). See the 🔴 in
  // cpu/rfd4p/reference.js - this was skipped there for a wrong tensor name and nothing noticed.
  if (wantInsertion && t.insertion && hasW(wn("nn.insertion_q_proj.weight"))) {
    float* q = scratch<float>("rfd4p.ins.q", n * D);
    float* k = scratch<float>("rfd4p.ins.k", n * D);
    linear<float, float>(t.s, q, n, D, D, wn("nn.insertion_q_proj.weight"), true);
    linear<float, float>(t.s, k, n, D, D, wn("nn.insertion_k_proj.weight"), true);
    const float alpha = 1.f / sqrtf((float)D), beta = 0.f;
    CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)n, D, &alpha, k, CUDA_R_32F, D,
                    q, CUDA_R_32F, D, &beta, t.insertion, CUDA_R_32F, (int)n,
                    computeType(), CUBLAS_GEMM_DEFAULT));
  }
}

inline void forward(Trunk& t, const float* seq, const float* cond0, const float* cond1,
                    const float* pairF, const float* pairCondF, int nRecycle, bool wantInsertion,
                    const std::function<void(const char*, const float*, size_t)>& tap = nullptr) {
  if (F32_TF32) {
    forwardT<half>(t, seq, cond0, cond1, pairF, pairCondF, nRecycle, wantInsertion, tap);
  } else {
    forwardT<float>(t, seq, cond0, cond1, pairF, pairCondF, nRecycle, wantInsertion, tap);
  }
}

}  // namespace rfd4p
