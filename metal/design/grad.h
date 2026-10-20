// metal/design: the backward of what the folding network is built from, written against metal/core.
//
// Design is gradient descent on a sequence through the network, so it needs a backward every other backend here
// lacks. The convention throughout: a cotangent is named d<forward name>, every array is a device pointer from
// metal/core's allocator, and a parameter's gradient is ACCUMULATED only where a function says so - otherwise it is
// written, so a caller zeroes nothing.
#pragma once
#include "core.h"

namespace mt {

// ---------------------------------------------------------------- scratch
// Declare this tree's families of scratch to metal/core, so the large buffers of modules that run one after another
// ALIAS instead of each holding its own (core.h shareScratch; metal/af2/main.mm does the same for the forward).
// A driver must call it once before folding, or every module's scratch is its own and a block holds the sum rather
// than the maximum - at a 192-residue pair that is 830 MB against 324.
//
// The nesting matters as much as the families: `attentionModuleBwd` runs INSIDE `biasedSelfAttentionBwd` and
// `attentionBwd` inside that, and layerNormBwd's statistics are live inside every module, so those are declared as
// living around each other's turn and are never aliased together.
void designShareScratch();

// ---------------------------------------------------------------- LayerNorm
// The forward is metal/core's layerNorm: y[c] = ((x[c] - mean) * inv) * scale[c] + offset[c]. This computes the row
// statistics itself (the forward saves none), then dx, and optionally the parameters' gradients.
//
//   x, dy    the forward's input and the incoming cotangent, [rows][C] (ldx / ldy if padded)
//   dx       [rows][C], written
//   dscale / doffset   [C], written; either may be null
// `scale` must be the forward's, or dx is the gradient of a DIFFERENT function - a null scale means the forward had
// none, not "ignore it".
void layerNormBwd(const float* x, const float* dy, float* dx, size_t rows, int C, const float* scale,
                  float* dscale = nullptr, float* doffset = nullptr, float eps = 1e-5f,
                  int ldx = 0, int ldy = 0, int lddx = 0);

// ---------------------------------------------------------------- a linear layer
// The forward is Y[rows][out] = X[rows][in] W[in][out] (+ bias). Both products below are metal/core's GEMM with one
// operand transposed, which it already serves, so there is no kernel here:
//   dX[rows][in] = dY[rows][out] W^T        (W read as stored [in][out], i.e. transposed for this product)
//   dW[in][out]  = X^T dY                   (X read as stored [rows][in], i.e. transposed for this product)
//   dbias[out]   = sum over rows of dY
// Any of dX, dW, dbias may be null. Nothing is accumulated: each is written.
// `accumulateX` adds into dX instead of writing it - the outer product's two projections share one input, so its
// cotangent is the sum of both paths. dW and dbias are always written.
void linearBwd(const float* X, const float* W, const float* dY, float* dX, float* dW, float* dbias,
               size_t rows, int in, int out, bool accumulateX = false);

// The column sums on their own - a bias's gradient, and what doffset is.
void columnSum(const float* dy, float* out, size_t rows, int C, int ldy = 0);

// ---------------------------------------------------------------- elementwise
// dx = dy * (ref > 0). `ref` may be relu's input or its output; the sign is the same.
void reluBwd(const float* dy, const float* ref, float* dx, size_t n);

// out = y * sigmoid(g), the gate AF2 closes attention and the triangle with. `g` is pre-sigmoid. Either output may
// be null, and either may alias its own input.
void gateBwd(const float* dout, const float* y, const float* g, float* dy, float* dg, size_t n);

// ---------------------------------------------------------------- Transition (Alg. 9 MSATransition / 15 PairTransition)
// The forward, as AF2 writes it:  out = relu(LayerNorm(act) W1 + b1) W2 + b2.
// (AF2's Transition builds a mask and then never applies it - modules.py:277 against :297. Not a simplification here.)
//
// This RECOMPUTES the forward it needs rather than taking a save set, which is what AF2 runs under anyway
// (`use_remat=True`, bindcraft/af2.py:362). Only the post-relu activation and the normalised input are kept, both in
// scratch; the pre-relu value is never needed, because relu(h) > 0 exactly where h > 0.
//
//   act      [rows][C], the module's input
//   dOut     [rows][C], the cotangent of its output
//   dAct     [rows][C], written. Any parameter gradient may be null.
// H is the intermediate width (AF2: C * num_intermediate_factor, 4 for the pair track).
void transitionBwd(const float* act, const float* dOut, size_t rows, int C, int H,
                   const float* lnScale, const float* lnOffset,
                   const float* W1, const float* b1, const float* W2, const float* b2,
                   float* dAct,
                   float* dLnScale = nullptr, float* dLnOffset = nullptr,
                   float* dW1 = nullptr, float* db1 = nullptr,
                   float* dW2 = nullptr, float* db2 = nullptr);

// ---------------------------------------------------------------- OuterProductMean (Alg. 10)
// See kernels.h for the forward this is the backward of, and for why it is tiled over i.
//
//   msa      [S][N][Cm], the module's input;  mask [S][N]
//   Wl, Wr   [Cm][O]  with biases [O];  Wo [O][O][F], bo [F]
//   dOut     [N][N][F], the cotangent of the module's output
//   dMsa     [S][N][Cm], written
// Any parameter gradient may be null. `tile` is how many rows of i a pass holds - it trades the [T][N][O][O]
// scratch against the number of dispatches, and changes nothing about the result.
void outerProductMeanBwd(const float* msa, const float* mask, size_t S, int N, int Cm, int O, int F,
                         const float* lnScale, const float* lnOffset,
                         const float* Wl, const float* bl, const float* Wr, const float* br,
                         const float* Wo, const float* dOut,
                         float* dMsa,
                         float* dLnScale = nullptr, float* dLnOffset = nullptr,
                         float* dWl = nullptr, float* dbl = nullptr,
                         float* dWr = nullptr, float* dbr = nullptr,
                         float* dWo = nullptr, float* dbo = nullptr,
                         float eps = 1e-3f, int tile = 8);

// ---------------------------------------------------------------- attention
// See kernels.h for the forward, the layouts, and why this recomputes scores rather than storing them.
//
// The reference forward, which also saves the per-row logsumexp the backward needs. metal/core's tuned `attention()`
// does NOT return it, so using that one here would need a core change; this is the self-contained version.
//   q (already scaled), k, v   [B][pos][H][D];  out [B][Q][H][D];  L [B][H][Q]
void attentionFwdSaved(const float* q, const float* k, const float* v,
                       const float* maskBias, const float* pairBias,
                       float* out, float* L, int B, int Q, int K, int H, int D);

// out, L from the forward above. dq, dk, dv and dPairBias may each be null. Nothing is accumulated.
void attentionBwd(const float* q, const float* k, const float* v,
                  const float* maskBias, const float* pairBias,
                  const float* out, const float* L, const float* dOut,
                  float* dq, float* dk, float* dv, float* dPairBias,
                  int B, int Q, int K, int H, int D);

// ---------------------------------------------------------------- the Attention MODULE (modules.py:336-399)
// The forward around the core above:
//   q = (qData Wq) * key_dim^-0.5     k = mData Wk     v = mData Wv        [B][pos][H][D]
//   wa = attend(q, k, v, maskBias, pairBias)
//   if gating:  wa *= sigmoid(qData Wg + bg)
//   out = wa Wo + bo                                                       [B][Q][O]
// Wq/Wk/Wv/Wg are AF2's [C][H][D] parameters, which flatten to [C][H*D] and so are plain linears; only the gate and
// the output projection carry a bias, as the reference has it.
//
// SELF-ATTENTION: pass the SAME pointer for dQData and dMData (AF2 does this everywhere - q_data and m_data are both
// the normalised activation) and all five paths accumulate into it. Then Q must equal K and Cq equal Cm.
struct AttnModuleWeights {
  const float* Wq; const float* Wk; const float* Wv;
  const float* Wg = nullptr; const float* bg = nullptr;    // Wg null: no gating
  const float* Wo; const float* bo;
};
struct AttnModuleGrads {
  float* Wq = nullptr; float* Wk = nullptr; float* Wv = nullptr;
  float* Wg = nullptr; float* bg = nullptr;
  float* Wo = nullptr; float* bo = nullptr;
};
void attentionModuleBwd(const float* qData, const float* mData,
                        const float* maskBias, const float* pairBias,
                        const AttnModuleWeights& w, const float* dOut,
                        float* dQData, float* dMData, float* dPairBias,
                        int B, int Q, int K, int Cq, int Cm, int H, int D, int O,
                        const AttnModuleGrads& g = {});

// ------------------------------------------- biased self-attention: MSARowAttentionWithPairBias AND TriangleAttention
// The two are the same shape, so one function serves both (modules.py:527-561 and :698-729):
//   maskBias = 1e9 * (mask - 1)                                   [B][K]
//   actN     = LayerNorm_query_norm(act)                          [B * Q][Cq]
//   biasN    = (bias source is act) ? actN : LayerNorm_feat_2d_norm(biasSrc)      [Q * K][Cb]
//   nb[h][q][k] = sum_c biasN[(q, k)][c] feat2dW[c][h]            ('qkc,ch->hqk', so a linear then a permute)
//   out      = Attention(actN, actN, maskBias, nb)                [B][Q][O]
//
// MSARowAttentionWithPairBias has TWO LayerNorms and takes its bias from a DIFFERENT tensor (the pair) than the one
// it attends over (the MSA): pass `biasSrc` = the pair, with bScale/bOffset.
// TriangleAttention has ONE: the bias comes from the same normalised pair it attends over. Pass `biasSrc == act`
// and leave bScale/bOffset null, and `dBiasSrc` is then ignored - that path accumulates into dAct, which is right,
// because it IS the same tensor.
//
// The bias source must have exactly Q * K rows. `per_column` is NOT handled here: AF2 swaps the pair's axes before
// and after (`modules.py:703` and `:731`), and so should a caller - it is a transpose, not a different gradient.
struct BiasedAttnWeights {
  const float* qScale; const float* qOffset;
  const float* bScale = nullptr; const float* bOffset = nullptr;
  const float* feat2dW;
  AttnModuleWeights attn;
};
struct BiasedAttnGrads {
  float* qScale = nullptr; float* qOffset = nullptr;
  float* bScale = nullptr; float* bOffset = nullptr;
  float* feat2dW = nullptr;
  AttnModuleGrads attn;
};
void biasedSelfAttentionBwd(const float* act, const float* biasSrc, const float* mask,
                            int B, int Q, int Cq, int Cb, int H, int D, int O,
                            const BiasedAttnWeights& w, const float* dOut,
                            float* dAct, float* dBiasSrc, const BiasedAttnGrads& g = {});

// ---------------------------------------------------------------- MSAColumnAttention (Alg. 8)
// The same module as MSARowAttentionWithPairBias with the first two axes swapped and NO pair bias
// (modules.py:589-614): the MSA is attended down its columns, so each residue is a batch and the sequences are the
// queries. `w.feat2dW` is left null and the bias path disappears.
//   msa      [S][N][Cm];  msaMask [S][N];  dOut [S][N][O];  dMsa [S][N][Cm], written
void msaColumnAttentionBwd(const float* msa, const float* msaMask, int S, int N, int Cm, int H, int D, int O,
                           const BiasedAttnWeights& w, const float* dOut, float* dMsa,
                           const BiasedAttnGrads& g = {});

// ---------------------------------------------------------------- MSAColumnGlobalAttention (the extra-MSA stack's)
// See kernels.h for the forward and for why this one stores its attention weights where every other attention here
// refuses to. Wk and Wv have NO head axis; Wq, Wg and Wo do. Gating is required.
//   msa [S][N][Cm];  msaMask [S][N];  dOut [S][N][O];  dMsa [S][N][Cm], written
struct GlobalAttnWeights {
  const float* qScale; const float* qOffset;   // query_norm
  const float* Wq;                             // [Cm][H * D]
  const float* Wk; const float* Wv;            // [Cm][D]   - shared across heads
  const float* Wg; const float* bg;            // [Cm][H * D], [H * D]
  const float* Wo; const float* bo;            // [H * D][O], [O]
};
struct GlobalAttnGrads {
  float* qScale = nullptr; float* qOffset = nullptr;
  float* Wq = nullptr; float* Wk = nullptr; float* Wv = nullptr;
  float* Wg = nullptr; float* bg = nullptr;
  float* Wo = nullptr; float* bo = nullptr;
};
void msaColumnGlobalAttentionBwd(const float* msa, const float* msaMask, int S, int N, int Cm, int H, int D, int O,
                                 const GlobalAttnWeights& w, const float* dOut, float* dMsa,
                                 const GlobalAttnGrads& g = {});

// ---------------------------------------------------------------- TriangleMultiplication (Alg. 11 / 12)
// The forward, as AF2 writes it (modules.py:947-1012). With n2 = N * N pair rows, C the pair width and I the
// intermediate width (num_intermediate_channel):
//   act0 = LayerNorm_input(pair)                       [n2][C]
//   aM   = mask * (act0 Wl  + bl)     a = aM * sigmoid(act0 Wlg + blg)       [n2][I]
//   bM   = mask * (act0 Wr  + br)     b = bM * sigmoid(act0 Wrg + brg)       [n2][I]
//   p    = outgoing ? sum_k a[i][k][c] b[j][k][c] : sum_k a[k][j][c] b[k][i][c]
//   q    = LayerNorm_center(p)
//   out  = (q Wo + bo) * sigmoid(act0 Wg + bg)         [n2][C]
// Note the ORDER: the mask multiplies the projection, then the gate multiplies that - not the other way round.
//
// This is where the Evoformer's O(N^3) lives, so the two contractions of the backward are BATCHED GEMMS over the
// channel, which means the operands are packed channel-major ([n2][I] -> [I][n2]) exactly as metal/core's forward
// keeps them. See `grad.mm` for the four products, which differ between the two equations.
//
//   pair     [n2][C], the module's input;  mask [n2]
//   dOut     [n2][C], the cotangent of its output;  dPair [n2][C], written
// Any parameter gradient may be null.
struct TriMulWeights {
  const float* lnScale; const float* lnOffset;            // the input LayerNorm
  const float* Wl; const float* bl;                       // left_projection   [C][I]
  const float* Wr; const float* br;                       // right_projection  [C][I]
  const float* Wlg; const float* blg;                     // left_gate         [C][I]
  const float* Wrg; const float* brg;                     // right_gate        [C][I]
  const float* centerScale; const float* centerOffset;    // the centre LayerNorm
  const float* Wo; const float* bo;                       // output_projection [I][C]
  const float* Wg; const float* bg;                       // gating_linear     [C][C]
};
struct TriMulGrads {
  float* lnScale = nullptr; float* lnOffset = nullptr;
  float* Wl = nullptr; float* bl = nullptr;
  float* Wr = nullptr; float* br = nullptr;
  float* Wlg = nullptr; float* blg = nullptr;
  float* Wrg = nullptr; float* brg = nullptr;
  float* centerScale = nullptr; float* centerOffset = nullptr;
  float* Wo = nullptr; float* bo = nullptr;
  float* Wg = nullptr; float* bg = nullptr;
};
void triangleMultiplicationBwd(const float* pair, const float* mask, int N, int C, int I, bool outgoing,
                               const TriMulWeights& w, const float* dOut, float* dPair,
                               const TriMulGrads& g = {});

// ---------------------------------------------------------------- EvoformerIteration (Alg. 6, lines 2-10)
// A whole block's backward, composed of everything above. AF2 wraps each module in `dropout_wrapper`
// (modules.py:46-76), which is `new_act = output_act + dropout(module(input_act, mask))` - and with dropout off,
// which is what a design trajectory runs, a plain RESIDUAL. So a module's cotangent BOTH passes straight through
// and goes through the module, summed; nothing here is a chain of overwrites.
//
// The forward order (modules.py:1300-1388), with `outerProductMeanFirst` as the config's
// `outer_product_mean.first`:
//    [opm first]  pair += OPM(msa)
//    msa  += MSARowAttentionWithPairBias(msa, pair)        <- reads BOTH tracks, so it feeds both cotangents
//    msa  += MSAColumnAttention(msa)                        (or the global one, in the extra-MSA stack)
//    msa  += MSATransition(msa)
//    [else]       pair += OPM(msa)
//    pair += TriangleMultiplication(pair, outgoing)
//    pair += TriangleMultiplication(pair, incoming)
//    pair += TriangleAttention(pair, starting_node)         per_row
//    pair += TriangleAttention(pair, ending_node)           per_column, so the pair is transposed around it
//    pair += PairTransition(pair)
//
//   msa  [S][N][Cm], msaMask [S][N];  pair [N][N][Cz], pairMask [N][N]
//   dMsaOut / dPairOut   the cotangents of the block's two outputs
//   dMsa / dPair         written; may be the same buffers as dMsaOut / dPairOut
struct TransitionWeights {
  const float* lnScale; const float* lnOffset;
  const float* W1; const float* b1; const float* W2; const float* b2;
};
struct TransitionGrads {
  float* lnScale = nullptr; float* lnOffset = nullptr;
  float* W1 = nullptr; float* b1 = nullptr; float* W2 = nullptr; float* b2 = nullptr;
};
struct OpmWeights {
  const float* lnScale; const float* lnOffset;
  const float* Wl; const float* bl; const float* Wr; const float* br;
  const float* Wo;
};
struct OpmGrads {
  float* lnScale = nullptr; float* lnOffset = nullptr;
  float* Wl = nullptr; float* bl = nullptr; float* Wr = nullptr; float* br = nullptr;
  float* Wo = nullptr; float* bo = nullptr;
};
struct EvoBlockWeights {
  OpmWeights opm;
  BiasedAttnWeights msaRow;                    // two LayerNorms, bias from the pair
  BiasedAttnWeights msaCol;                    // one, no bias (feat2dW null)
  TransitionWeights msaTransition;
  TriMulWeights triOutgoing, triIncoming;
  BiasedAttnWeights triAttnStart, triAttnEnd;
  TransitionWeights pairTransition;
};
struct EvoBlockGrads {
  OpmGrads opm;
  BiasedAttnGrads msaRow, msaCol;
  TransitionGrads msaTransition;
  TriMulGrads triOutgoing, triIncoming;
  BiasedAttnGrads triAttnStart, triAttnEnd;
  TransitionGrads pairTransition;
};
struct EvoBlockShape {
  int S, N, Cm, Cz;                            // sequences, residues, MSA and pair widths
  int msaHeads, msaDim;                        // the MSA attentions' heads and head width
  int triHeads, triDim;                        // the triangle attentions'
  int triIntermediate;                         // the triangle multiplications' num_intermediate_channel
  int msaIntermediate, pairIntermediate;       // the transitions' widths (num_intermediate_factor * channel)
  int opmOuter;                                // num_outer_channel (32)
  bool outerProductMeanFirst;
};
// 🔴 THE SAVE SET, AND WHY IT IS AN ARGUMENT RATHER THAN RECOMPUTED HERE. Each module's backward recomputes its
// OWN forward, but it needs the activation it was given, and in a block that is the output of every module before
// it. So a block backward needs the chain's intermediate activations, and there are only two honest ways to get
// them: duplicate AF2's whole forward in this tree, or have the forward hand them over. The forward already exists
// (metal/af2), so it hands them over - which is also what a training step does, and what BindCraft2's own
// GLUE_PORT.md means by carrying the trunk's save set across the boundary.
//
// Each field is the activation that module was CALLED with. `msaIn` serves both the block's input and the MSA row
// attention, because an outer-product-mean that runs first writes the pair and leaves the MSA alone.
struct EvoBlockSaved {
  const float* msaIn;                 // the block's MSA input; OPM's input when it runs first
  const float* pairForRow;            // the pair as MSARowAttentionWithPairBias saw it
  const float* msaForCol;
  const float* msaForTransition;
  const float* msaForOpm;             // the MSA after its transition; OPM's input when it does NOT run first
  const float* pairForTriOut;
  const float* pairForTriIn;
  const float* pairForAttnStart;
  const float* pairForAttnEnd;
  const float* pairForTransition;
};
void evoformerIterationBwd(const float* msaMask, const float* pairMask, const EvoBlockSaved& saved,
                           const EvoBlockShape& sh, const EvoBlockWeights& w,
                           const float* dMsaOut, const float* dPairOut,
                           float* dMsa, float* dPair, const EvoBlockGrads& g = {});

}  // namespace mt
