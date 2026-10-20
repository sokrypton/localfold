// metal/design's host half: the dispatches behind grad.h.
#include "grad.h"
#include "kernels.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <utility>

namespace mt {

// ---------------------------------------------------------------- the heads, and the metrics
void distogramHeadBwd(const float* pair, const float* W, const float* dLogits, int N, int Cz, int bins,
                      float* dPair, float* dW, float* dbias) {
  if (!N) return;
  const size_t pr = (size_t)N * N;
  float* dHalf = (float*)scratchBytes("design.head.dhalf", pr * bins * sizeof(float));
  // logits = half + half^T, which is self-adjoint: each half entry gets its own cotangent and its transpose's
  run1d("lf_sym_add", pr * bins, SymAddArgs{dLogits, dHalf, pr * (size_t)bins, (uint)N, (uint)bins, 0, 0});
  linearBwd(pair, W, dHalf, dPair, dW, dbias, pr, Cz, bins);
}

void paeHeadBwd(const float* pair, const float* W, const float* dLogits, int N, int Cz, int bins,
                float* dPair, float* dW, float* dbias) {
  if (!N) return;
  linearBwd(pair, W, dLogits, dPair, dW, dbias, (size_t)N * N, Cz, bins);
}

void paeMetricsBwd(const float* logits, const float* breaks, const float* residueWeights, const float* asymId,
                   int N, int bins, const PaeMetricCotangents& d, float* dLogits) {
  if (!N) return;
  const size_t pr = (size_t)N * N;
  auto buf = [&](const char* name, size_t n) {
    return (float*)scratchBytes(std::string("design.pae.") + name, n * sizeof(float));
  };
  float* centers = buf("centers", bins);
  float* tmPerBin = buf("tmPerBin", bins);
  float* tmP = buf("tmP", pr);
  float* normedP = buf("normedP", pr);
  float* perAlignP = buf("perAlignP", N);
  float* dPerAlignP = buf("dPerAlignP", N);
  float* tmI = buf("tmI", pr);
  float* normedI = buf("normedI", pr);
  float* perAlignI = buf("perAlignI", N);
  float* dPerAlignI = buf("dPerAlignI", N);

  PaeArgs base{};
  base.logits = logits; base.breaks = breaks; base.weights = residueWeights; base.asym = asymId;
  base.centers = centers; base.tmPerBin = tmPerBin;
  base.N = (uint)N; base.bins = (uint)bins;

  { PaeArgs a = base; a.n = 1; run1d("lf_pae_consts", 1, a); }

  // the whole-complex set (pTM), then the interface set (ipTM) - the same kernels with the other mask
  const bool wantP = d.ptm != nullptr;
  const bool wantI = asymId && (d.iptm || d.iptmPerResidue);
  if (wantP) {
    PaeArgs a = base; a.tm = tmP; a.normed = normedP; a.perAlign = perAlignP; a.interface = 0;
    a.n = pr; run1d("lf_pae_tm", pr, a);
    a.n = (size_t)N; run1d("lf_pae_rows", (size_t)N, a);
    a.dPtm = d.ptm; a.dPerAlign = dPerAlignP; a.n = 1;
    run1d("lf_pae_dperalign", 1, a);
  }
  if (wantI) {
    PaeArgs a = base; a.tm = tmI; a.normed = normedI; a.perAlign = perAlignI; a.interface = 1;
    a.n = pr; run1d("lf_pae_tm", pr, a);
    a.n = (size_t)N; run1d("lf_pae_rows", (size_t)N, a);
    a.dIptm = d.iptm; a.dIptmPer = d.iptmPerResidue; a.dPerAlign = dPerAlignI; a.n = 1;
    run1d("lf_pae_dperalign", 1, a);
  }
  {
    PaeArgs a = base;
    a.dPae = d.pae;
    a.normed = wantI ? normedI : nullptr;
    a.dPerAlign = wantI ? dPerAlignI : nullptr;
    a.asym = wantI ? asymId : nullptr;
    a.normedP = wantP ? normedP : nullptr;
    a.dPerAlignP = wantP ? dPerAlignP : nullptr;
    a.dLogits = dLogits;
    a.n = pr;
    run1d("lf_pae_dlogits", pr, a);
  }
}

// ---------------------------------------------------------------- EvoformerIteration
void evoformerIterationBwd(const float* msaMask, const float* pairMask, const EvoBlockSaved& sv,
                           const EvoBlockShape& sh, const EvoBlockWeights& w,
                           const float* dMsaOut, const float* dPairOut,
                           float* dMsa, float* dPair, const EvoBlockGrads& g) {
  const int S = sh.S, N = sh.N, Cm = sh.Cm, Cz = sh.Cz;
  if (!S || !N) return;
  const size_t msaRows = (size_t)S * N, pairRows = (size_t)N * N;
  auto buf = [&](const char* name, size_t n) {
    return (float*)scratchBytes(std::string("design.evo.") + name, n * sizeof(float));
  };
  float* tmpMsa = buf("tmpMsa", msaRows * Cm);
  float* tmpPair = buf("tmpPair", pairRows * Cz);
  float* tmpPair2 = buf("tmpPair2", pairRows * Cz);

  // every module's contribution is added to the running cotangent, because every module is a residual
  auto addMsa = [&]() { run1d("lf_accum", msaRows * Cm, AccumArgs{dMsa, tmpMsa, msaRows * (size_t)Cm, 0}); };
  auto addPair = [&](const float* src) {
    run1d("lf_accum", pairRows * Cz, AccumArgs{dPair, src, pairRows * (size_t)Cz, 0});
  };
  if (dMsa != dMsaOut) copy(dMsa, dMsaOut, msaRows * (size_t)Cm * sizeof(float));
  if (dPair != dPairOut) copy(dPair, dPairOut, pairRows * (size_t)Cz * sizeof(float));

  auto transition = [&](const float* act, size_t rows, int C, int H, const TransitionWeights& tw,
                        const TransitionGrads& tg, float* out) {
    transitionBwd(act, rows == msaRows ? dMsa : dPair, rows, C, H, tw.lnScale, tw.lnOffset,
                  tw.W1, tw.b1, tw.W2, tw.b2, out, tg.lnScale, tg.lnOffset, tg.W1, tg.b1, tg.W2, tg.b2);
  };

  // -------- in reverse. 10: the pair transition
  transition(sv.pairForTransition, pairRows, Cz, sh.pairIntermediate, w.pairTransition, g.pairTransition, tmpPair);
  addPair(tmpPair);

  // -------- 9: the ending node, which AF2 runs on the TRANSPOSED pair (modules.py:703, :731)
  {
    float* pairT = buf("pairT", pairRows * Cz);
    float* maskT = buf("maskT", pairRows);
    float* dT = buf("dT", pairRows * Cz);
    float* outT = buf("outT", pairRows * Cz);
    auto swap = [&](const float* x, float* y, int C) {
      run1d("lf_swap01", pairRows * C, Swap01Args{x, y, pairRows * (size_t)C, (uint)N, (uint)N, (uint)C, 0});
    };
    swap(sv.pairForAttnEnd, pairT, Cz);
    swap(pairMask, maskT, 1);
    swap(dPair, dT, Cz);
    biasedSelfAttentionBwd(pairT, pairT, maskT, N, N, Cz, Cz, sh.triHeads, sh.triDim, Cz,
                           w.triAttnEnd, dT, outT, outT, g.triAttnEnd);
    swap(outT, tmpPair, Cz);                 // the swap is its own inverse at N x N
    addPair(tmpPair);
  }

  // -------- 8: the starting node, per_row
  biasedSelfAttentionBwd(sv.pairForAttnStart, sv.pairForAttnStart, pairMask, N, N, Cz, Cz,
                         sh.triHeads, sh.triDim, Cz, w.triAttnStart, dPair, tmpPair, tmpPair, g.triAttnStart);
  addPair(tmpPair);

  // -------- 7 and 6: the two triangle multiplications
  triangleMultiplicationBwd(sv.pairForTriIn, pairMask, N, Cz, sh.triIntermediate, false,
                            w.triIncoming, dPair, tmpPair, g.triIncoming);
  addPair(tmpPair);
  triangleMultiplicationBwd(sv.pairForTriOut, pairMask, N, Cz, sh.triIntermediate, true,
                            w.triOutgoing, dPair, tmpPair, g.triOutgoing);
  addPair(tmpPair);

  // -------- 5: the outer product, if it runs LAST in the forward. It writes the PAIR from the MSA, so the pair's
  // cotangent passes through untouched and only the MSA's grows.
  auto opm = [&](const float* msaAct) {
    outerProductMeanBwd(msaAct, msaMask, S, N, Cm, sh.opmOuter, Cz, w.opm.lnScale, w.opm.lnOffset,
                        w.opm.Wl, w.opm.bl, w.opm.Wr, w.opm.br, w.opm.Wo, dPair, tmpMsa,
                        g.opm.lnScale, g.opm.lnOffset, g.opm.Wl, g.opm.bl, g.opm.Wr, g.opm.br,
                        g.opm.Wo, g.opm.bo);
    addMsa();
  };
  if (!sh.outerProductMeanFirst) opm(sv.msaForOpm);

  // -------- 4: the MSA transition
  transition(sv.msaForTransition, msaRows, Cm, sh.msaIntermediate, w.msaTransition, g.msaTransition, tmpMsa);
  addMsa();

  // -------- 3: the MSA column attention
  msaColumnAttentionBwd(sv.msaForCol, msaMask, S, N, Cm, sh.msaHeads, sh.msaDim, Cm,
                        w.msaCol, dMsa, tmpMsa, g.msaCol);
  addMsa();

  // -------- 2: the MSA row attention, the one module that reads BOTH tracks, so it feeds both cotangents
  biasedSelfAttentionBwd(sv.msaIn, sv.pairForRow, msaMask, S, N, Cm, Cz, sh.msaHeads, sh.msaDim, Cm,
                         w.msaRow, dMsa, tmpMsa, tmpPair2, g.msaRow);
  addMsa();
  addPair(tmpPair2);

  // -------- 1: the outer product, if it runs FIRST
  if (sh.outerProductMeanFirst) opm(sv.msaIn);
}

// The families (core.h shareScratch): a block's modules - the transitions, the outer product, the triangle
// multiplications and the attentions - run one after another, so their large buffers alias. What must NOT alias is
// anything live around another's turn: the attention module runs inside the biased-attention wrapper and the core
// attention inside that, and a LayerNorm's row statistics are live inside every one of them.
void designShareScratch() {
  shareScratch({{"design.tr.", "dtr"}, {"design.opm.", "dopm"}, {"design.tri.", "dtri"},
                {"design.battn.", "dbattn"}, {"design.attnm.", "dattnm"}, {"design.attn.", "dattn"},
                {"design.mcol.", "dmcol"}, {"design.gattn.", "dgattn"}, {"design.evo.", "devo"},
                {"design.head.", "dhead"}, {"design.pae.", "dpae"},
                {"design.ln.", "dln"}},
               {{"dln", "dtr"}, {"dln", "dopm"}, {"dln", "dtri"}, {"dln", "dbattn"}, {"dln", "dattnm"},
                {"dln", "dattn"}, {"dln", "dmcol"}, {"dattnm", "dbattn"}, {"dattn", "dattnm"},
                {"dattn", "dbattn"},
                // the column attention transposes into the per-column orientation and calls the biased attention
                // on it, so everything that one holds is live inside its turn
                {"dmcol", "dbattn"}, {"dmcol", "dattnm"}, {"dmcol", "dattn"}, {"dln", "dgattn"},
                // the block holds its running cotangents and temporaries across every module's turn
                {"devo", "dtr"}, {"devo", "dopm"}, {"devo", "dtri"}, {"devo", "dbattn"}, {"devo", "dattnm"},
                {"devo", "dattn"}, {"devo", "dmcol"}, {"devo", "dgattn"}, {"devo", "dln"}});
}

// a simdgroup a row, eight a threadgroup - metal/core's layerNormRun grid, so the rows map the same way
static Grid rowGroups(size_t rows) {
  size_t groups = (rows + 7) / 8;
  return groups <= 65535 ? Grid{(uint32_t)groups, 1, 1}
                         : Grid{65535, (uint32_t)((groups + 65534) / 65535), 1};
}

void layerNormBwd(const float* x, const float* dy, float* dx, size_t rows, int C, const float* scale,
                  float* dscale, float* doffset, float eps, int ldx, int ldy, int lddx) {
  if (!rows) return;
  ldx = ldx ? ldx : C;
  ldy = ldy ? ldy : C;
  lddx = lddx ? lddx : C;

  // the row statistics, as the forward computed them (two floats a row; see kernels.h for why they are not recomputed
  // inside the column sums)
  float* mean = (float*)scratchBytes("design.ln.mean", rows * sizeof(float));
  float* inv = (float*)scratchBytes("design.ln.inv", rows * sizeof(float));
  LnStatsArgs sa{x, nullptr, mean, inv, rows, C, ldx, eps, 0};
  dispatch("lf_ln_stats", &sa, sizeof sa, rowGroups(rows), 256, 0, "design.ln.stats");

  if (dx) {
    LnBwdArgs a{x, nullptr, dy, nullptr, dx, nullptr, scale, mean, inv, rows, C, ldx, ldy, lddx, 0};
    dispatch("lf_ln_bwd", &a, sizeof a, rowGroups(rows), 256, 0, "design.ln.bwd");
  }
  if (dscale || doffset) {
    // dscale needs n, so it needs x and the statistics; doffset alone does not
    ColSumArgs a{dy, nullptr, dscale ? x : nullptr, nullptr, doffset, dscale,
                 dscale ? mean : nullptr, dscale ? inv : nullptr, rows, C, ldx, ldy, 0, 0};
    dispatch("lf_col_sums", &a, sizeof a, Grid{(uint32_t)C, 1, 1}, 256, 0, "design.ln.params");
  }
}

void columnSum(const float* dy, float* out, size_t rows, int C, int ldy) {
  if (!rows || !C) return;
  ColSumArgs a{dy, nullptr, nullptr, nullptr, out, nullptr, nullptr, nullptr, rows, C, C, ldy ? ldy : C, 0, 0};
  dispatch("lf_col_sums", &a, sizeof a, Grid{(uint32_t)C, 1, 1}, 256, 0, "design.colsum");
}

void reluBwd(const float* dy, const float* ref, float* dx, size_t n) {
  if (!n) return;
  run1d("lf_relu_bwd", n, ReluBwdArgs{dy, ref, dx, n});
}

void gateBwd(const float* dout, const float* y, const float* g, float* dy, float* dg, size_t n) {
  if (!n) return;
  run1d("lf_gate_bwd", n, GateBwdArgs{dout, y, g, dy, dg, n});
}

void linearBwd(const float* X, const float* W, const float* dY, float* dX, float* dW, float* dbias,
               size_t rows, int in, int out, bool accumulateX) {
  if (!rows) return;
  if (dX) {
    // dX[rows][in] = dY[rows][out] * W^T. In Gemm's terms the contraction is over `out`, the result is `in` wide, and
    // the weight wanted is [out][in] - which is W, stored [in][out], read transposed.
    Gemm g{};
    g.X = (void*)dY; g.tx = F32; g.transX = false;
    g.W = (void*)W; g.tw = F32; g.transW = true;
    g.Y = dX; g.ty = F32;
    g.rows = rows; g.in = out; g.out = in;
    g.beta = accumulateX ? 1.f : 0.f;
    g.sx = (int64_t)rows * out; g.sw = (int64_t)in * out; g.sy = (int64_t)rows * in;
    gemm(g);
  }
  if (dW) {
    // dW[in][out] = X^T dY: `in` rows, the contraction over `rows`, and the left operand wanted is [in][rows] - which
    // is X, stored [rows][in], read transposed.
    Gemm g{};
    g.X = (void*)X; g.tx = F32; g.transX = true;
    g.W = (void*)dY; g.tw = F32; g.transW = false;
    g.Y = dW; g.ty = F32;
    g.rows = (size_t)in; g.in = (int)rows; g.out = out;
    g.sx = (int64_t)rows * in; g.sw = (int64_t)rows * out; g.sy = (int64_t)in * out;
    gemm(g);
  }
  if (dbias) columnSum(dY, dbias, rows, out);
}

void transitionBwd(const float* act, const float* dOut, size_t rows, int C, int H,
                   const float* lnScale, const float* lnOffset,
                   const float* W1, const float* b1, const float* W2, const float* b2,
                   float* dAct,
                   float* dLnScale, float* dLnOffset, float* dW1, float* db1, float* dW2, float* db2) {
  if (!rows) return;
  float* ln = (float*)scratchBytes("design.tr.ln", rows * (size_t)C * sizeof(float));
  float* a1 = (float*)scratchBytes("design.tr.a1", rows * (size_t)H * sizeof(float));
  float* dh = (float*)scratchBytes("design.tr.dh", rows * (size_t)H * sizeof(float));
  float* dln = (float*)scratchBytes("design.tr.dln", rows * (size_t)C * sizeof(float));

  // -------- the forward, recomputed: the normalised input and the post-relu activation
  layerNorm(act, ln, rows, C, lnScale, lnOffset);
  {
    Gemm g{};
    g.X = ln; g.W = W1; g.Y = a1;
    g.rows = rows; g.in = C; g.out = H;
    g.bias = b1; g.relu = true;
    gemm(g);
  }

  // -------- the second linear: dW2, db2, and the cotangent of the activation (into dh, then masked in place)
  linearBwd(a1, W2, dOut, dh, dW2, db2, rows, H, C);
  reluBwd(dh, a1, dh, rows * (size_t)H);

  // -------- the first linear: dW1, db1, and the cotangent of the normalised input
  linearBwd(ln, W1, dh, dln, dW1, db1, rows, C, H);

  // -------- and through the LayerNorm, back to the module's input
  layerNormBwd(act, dln, dAct, rows, C, lnScale, dLnScale, dLnOffset);
}

void outerProductMeanBwd(const float* msa, const float* mask, size_t S, int N, int Cm, int O, int F,
                         const float* lnScale, const float* lnOffset,
                         const float* Wl, const float* bl, const float* Wr, const float* br,
                         const float* Wo, const float* dOut,
                         float* dMsa,
                         float* dLnScale, float* dLnOffset, float* dWl, float* dbl, float* dWr, float* dbr,
                         float* dWo, float* dbo, float eps, int tile) {
  if (!S || !N) return;
  const size_t rows = S * (size_t)N;        // the msa as a flat [S * N][Cm]
  const size_t pairs = (size_t)N * N;
  const size_t proj = rows * O;
  const uint OO = (uint)O * O;

  float* act = (float*)scratchBytes("design.opm.act", rows * Cm * sizeof(float));
  float* left = (float*)scratchBytes("design.opm.left", proj * sizeof(float));
  float* right = (float*)scratchBytes("design.opm.right", proj * sizeof(float));
  float* dleft = (float*)scratchBytes("design.opm.dleft", proj * sizeof(float));
  float* dright = (float*)scratchBytes("design.opm.dright", proj * sizeof(float));
  float* dact = (float*)scratchBytes("design.opm.dact", rows * Cm * sizeof(float));
  float* dPre = (float*)scratchBytes("design.opm.dpre", pairs * F * sizeof(float));
  float* P = (float*)scratchBytes("design.opm.P", (size_t)tile * N * OO * sizeof(float));
  float* dP = (float*)scratchBytes("design.opm.dP", (size_t)tile * N * OO * sizeof(float));

  // -------- the forward, recomputed: the normalised msa and the two masked projections
  layerNorm(msa, act, rows, Cm, lnScale, lnOffset);
  {
    Gemm g{}; g.X = act; g.W = Wl; g.Y = left; g.rows = rows; g.in = Cm; g.out = O; g.bias = bl; gemm(g);
  }
  {
    Gemm g{}; g.X = act; g.W = Wr; g.Y = right; g.rows = rows; g.in = Cm; g.out = O; g.bias = br; gemm(g);
  }
  run1d("lf_opm_mask", proj, OpmMaskArgs{left, mask, proj, (uint)N, (uint)O, 0, 0});
  run1d("lf_opm_mask", proj, OpmMaskArgs{right, mask, proj, (uint)N, (uint)O, 0, 0});

  // -------- the division by (eps + norm) comes off first, and dbo is its column sum
  run1d("lf_opm_dpre", pairs, OpmDpreArgs{dOut, mask, dPre, pairs, (uint)S, (uint)N, (uint)F, 0, eps, 0});
  if (dbo) columnSum(dPre, dbo, pairs, F);

  // -------- the tiles. dWo, dleft and dright accumulate, so they start at zero
  if (dWo) fill(dWo, 0, (size_t)OO * F * sizeof(float));
  fill(dleft, 0, proj * sizeof(float));
  fill(dright, 0, proj * sizeof(float));
  for (int i0 = 0; i0 < N; i0 += tile) {
    const uint T = (uint)std::min(tile, N - i0);
    const size_t pn = (size_t)T * N * OO;
    run1d("lf_opm_p", pn, OpmPArgs{left, right, P, pn, (uint)S, (uint)N, (uint)O, (uint)i0, T, 0});
    if (dWo)
      run1d("lf_opm_dwo", (size_t)OO * F,
            OpmDwoArgs{P, dPre, dWo, (size_t)OO * F, (uint)N, (uint)O, (uint)F, (uint)i0, T, 0});
    run1d("lf_opm_dp", pn, OpmDpArgs{dPre, Wo, dP, pn, (uint)N, (uint)O, (uint)F, (uint)i0, T, 0});
    run1d("lf_opm_dleft", S * T * O,
          OpmDleftArgs{dP, right, dleft, S * T * O, (uint)S, (uint)N, (uint)O, (uint)i0, T, 0});
    run1d("lf_opm_dright", proj,
          OpmDrightArgs{dP, left, dright, proj, (uint)S, (uint)N, (uint)O, (uint)i0, T, 0});
  }

  // -------- back through the masks, the two linears (sharing one input), and the LayerNorm
  run1d("lf_opm_mask", proj, OpmMaskArgs{dleft, mask, proj, (uint)N, (uint)O, 0, 0});
  run1d("lf_opm_mask", proj, OpmMaskArgs{dright, mask, proj, (uint)N, (uint)O, 0, 0});
  linearBwd(act, Wl, dleft, dact, dWl, dbl, rows, Cm, O, false);
  linearBwd(act, Wr, dright, dact, dWr, dbr, rows, Cm, O, true);     // accumulate: one input, two paths
  layerNormBwd(msa, dact, dMsa, rows, Cm, lnScale, dLnScale, dLnOffset);
}

// ---------------------------------------------------------------- attention
static Grid attnRows(size_t rows) {
  size_t groups = (rows + 7) / 8;
  return groups <= 65535 ? Grid{(uint32_t)groups, 1, 1}
                         : Grid{65535, (uint32_t)((groups + 65534) / 65535), 1};
}

void attentionFwdSaved(const float* q, const float* k, const float* v, const float* maskBias,
                       const float* pairBias, float* out, float* L, int B, int Q, int K, int H, int D) {
  if (!B || !Q) return;
  if (D > 64) { fprintf(stderr, "design: attention D=%d, this path holds D <= 64\n", D); exit(1); }
  const size_t rows = (size_t)B * H * Q;
  AttnBwdArgs a{};
  a.q = q; a.k = k; a.v = v; a.maskBias = maskBias; a.pairBias = pairBias;
  a.wout = out; a.wL = L;
  a.n = rows; a.B = B; a.Q = Q; a.K = K; a.H = H; a.D = D;
  dispatch("lf_attn_ref_fwd", &a, sizeof a, attnRows(rows), 256, 0, "design.attn.fwd");
}

void attentionBwd(const float* q, const float* k, const float* v, const float* maskBias, const float* pairBias,
                  const float* out, const float* L, const float* dOut,
                  float* dq, float* dk, float* dv, float* dPairBias, int B, int Q, int K, int H, int D) {
  if (!B || !Q) return;
  if (D > 64) { fprintf(stderr, "design: attention D=%d, this path holds D <= 64\n", D); exit(1); }
  const size_t qrows = (size_t)B * H * Q, krows = (size_t)B * H * K;
  float* Drow = (float*)scratchBytes("design.attn.drow", qrows * sizeof(float));

  AttnBwdArgs a{};
  a.q = q; a.k = k; a.v = v; a.maskBias = maskBias; a.pairBias = pairBias;
  a.out = out; a.L = L; a.dOut = dOut;
  a.B = B; a.Q = Q; a.K = K; a.H = H; a.D = D;

  // softmax's own term, from the forward's output
  AttnBwdArgs d = a;
  d.dq = Drow;                                   // (the Drow kernel writes through dq)
  d.n = qrows;
  dispatch("lf_attn_drow", &d, sizeof d, attnRows(qrows), 256, 0, "design.attn.drow");
  a.Drow = Drow;

  if (dq) {
    AttnBwdArgs p = a; p.dq = dq; p.n = qrows;
    dispatch("lf_attn_bwd_dq", &p, sizeof p, attnRows(qrows), 256, 0, "design.attn.dq");
  }
  if (dk || dv) {
    // both come off the same pass; a caller wanting one still pays for the other
    float* sk = dk ? dk : (float*)scratchBytes("design.attn.sk", krows * (size_t)D * sizeof(float));
    float* sv = dv ? dv : (float*)scratchBytes("design.attn.sv", krows * (size_t)D * sizeof(float));
    AttnBwdArgs p = a; p.dk = sk; p.dv = sv; p.n = krows;
    dispatch("lf_attn_bwd_dkv", &p, sizeof p, attnRows(krows), 256, 0, "design.attn.dkv");
  }
  if (dPairBias) {
    const size_t n = (size_t)H * Q * K;
    AttnBwdArgs p = a; p.dPairBias = dPairBias; p.n = n;
    run1d("lf_attn_bwd_dbias", n, p);
  }
}

void attentionModuleBwd(const float* qData, const float* mData, const float* maskBias, const float* pairBias,
                        const AttnModuleWeights& w, const float* dOut,
                        float* dQData, float* dMData, float* dPairBias,
                        int B, int Q, int K, int Cq, int Cm, int H, int D, int O,
                        const AttnModuleGrads& g) {
  if (!B || !Q) return;
  const bool same = dQData == dMData;
  if (same && (Q != K || Cq != Cm)) {
    fprintf(stderr, "design: attentionModuleBwd got one cotangent for q and m, but Q!=K or Cq!=Cm\n");
    exit(1);
  }
  const size_t qrows = (size_t)B * Q, krows = (size_t)B * K;
  const int W = H * D;
  const float scale = 1.f / std::sqrt((float)D);

  auto buf = [&](const char* name, size_t n) {
    return (float*)scratchBytes(std::string("design.attnm.") + name, n * sizeof(float));
  };
  float* qs = buf("qs", qrows * W);
  float* kk = buf("k", krows * W);
  float* vv = buf("v", krows * W);
  float* wa = buf("wa", qrows * W);
  float* L = buf("L", (size_t)B * H * Q);
  float* gpre = buf("gpre", qrows * W);
  float* dwa = buf("dwa", qrows * W);
  float* dwa2 = buf("dwa2", qrows * W);
  float* dgpre = buf("dgpre", qrows * W);
  // REUSED, not allocated. By the time the core backward runs, three of the buffers above are dead whether or not
  // this attention gates: `gpre` and `dgpre` are read for the last time by gateBwd and by the gate's own linear -
  // or, with no gate, are never touched at all - and ONE of dwa/dwa2 holds the gate stage's spent cotangent while
  // the other carries the core's. 54 MB at a 192-residue triangle attention.
  //
  // 🔴 WHICH of dwa/dwa2 is dead depends on the gate, so it is bound AFTER that branch, below. An earlier version
  // bound it here, before a std::swap, so `dqs` ended up aliasing the very cotangent attentionBwd READS: the dq
  // pass overwrote its own input and the dk/dv and dBias passes then read rubbish. 21 cases failed and said so.
  //
  // Q == K wherever this reuse matters: cross-attention with K > Q would want the k-sized buffers bigger, so they
  // are taken from the q-sized ones only when that holds, and allocated when it does not.
  float* dkk = krows <= qrows ? gpre : buf("dk", krows * W);
  float* dvv = krows <= qrows ? dgpre : buf("dv", krows * W);

  auto lin = [&](const float* X, const float* Wm, const float* b, float* Y, size_t rows, int in, int out) {
    Gemm gm{}; gm.X = X; gm.W = Wm; gm.Y = Y; gm.rows = rows; gm.in = in; gm.out = out; gm.bias = b; gemm(gm);
  };

  // -------- the forward, recomputed
  lin(qData, w.Wq, nullptr, qs, qrows, Cq, W);
  run1d("lf_scale", qrows * W, ScaleArgs{qs, qrows * W, scale, 0});
  lin(mData, w.Wk, nullptr, kk, krows, Cm, W);
  lin(mData, w.Wv, nullptr, vv, krows, Cm, W);
  attentionFwdSaved(qs, kk, vv, maskBias, pairBias, wa, L, B, Q, K, H, D);
  if (w.Wg) lin(qData, w.Wg, w.bg, gpre, qrows, Cq, W);

  // -------- the output projection, then the gate
  const float* waOut = wa;
  if (w.Wg) {
    run1d("lf_mul_sigmoid", qrows * W, MulSigmoidArgs{wa, gpre, dwa2, qrows * (size_t)W});
    waOut = dwa2;                             // the gated activation the output projection actually consumed
  }
  linearBwd(waOut, w.Wo, dOut, dwa, g.Wo, g.bo, qrows, W, O);

  bool wrote = false;                          // whether dQData has been written yet (the rest accumulate)
  float* dAttnOut = dwa;                       // the cotangent of the UNgated attention output
  float* dqs = dwa2;                           // and the spent buffer, free for the core's dq
  if (w.Wg) {
    gateBwd(dwa, wa, gpre, dwa2, dgpre, qrows * (size_t)W);
    linearBwd(qData, w.Wg, dgpre, dQData, g.Wg, g.bg, qrows, Cq, W, false);
    wrote = true;
    dAttnOut = dwa2;                           // gateBwd put d/d(wa) here, and dwa is now spent
    dqs = dwa;
  }

  // -------- the core
  attentionBwd(qs, kk, vv, maskBias, pairBias, wa, L, dAttnOut, dqs, dkk, dvv, dPairBias, B, Q, K, H, D);

  // -------- and back through the three projections. q's carries AF2's scale.
  run1d("lf_scale", qrows * W, ScaleArgs{dqs, qrows * W, scale, 0});
  linearBwd(qData, w.Wq, dqs, dQData, g.Wq, nullptr, qrows, Cq, W, wrote);
  wrote = true;
  linearBwd(mData, w.Wk, dkk, dMData, g.Wk, nullptr, krows, Cm, W, same ? true : false);
  linearBwd(mData, w.Wv, dvv, dMData, g.Wv, nullptr, krows, Cm, W, true);
}

void biasedSelfAttentionBwd(const float* act, const float* biasSrc, const float* mask,
                            int B, int Q, int Cq, int Cb, int H, int D, int O,
                            const BiasedAttnWeights& w, const float* dOut,
                            float* dAct, float* dBiasSrc, const BiasedAttnGrads& g) {
  if (!B || !Q) return;
  const int K = Q;                            // self-attention
  const bool shared = biasSrc == act;         // TriangleAttention: the bias source IS the attended tensor
  const size_t rows = (size_t)B * Q;
  const size_t nbRows = (size_t)Q * K;
  auto buf = [&](const char* name, size_t n) {
    return (float*)scratchBytes(std::string("design.battn.") + name, n * sizeof(float));
  };
  float* actN = buf("actN", rows * Cq);
  float* biasN = (!w.feat2dW || shared) ? actN : buf("biasN", nbRows * Cb);
  const bool hasBias = w.feat2dW != nullptr;   // MSAColumnAttention has none: it attends with the mask alone
  float* nbRow = hasBias ? buf("nbRow", nbRows * H) : nullptr;
  float* nb = hasBias ? buf("nb", nbRows * H) : nullptr;
  float* maskBias = buf("maskBias", (size_t)B * K);
  float* dActN = buf("dActN", rows * Cq);
  float* dNb = hasBias ? buf("dNb", nbRows * H) : nullptr;
  float* dNbRow = hasBias ? buf("dNbRow", nbRows * H) : nullptr;

  // -------- the forward, recomputed
  run1d("lf_affine", (size_t)B * K, AffineArgs{mask, maskBias, (size_t)B * K, 1e9f, -1e9f, 0});
  layerNorm(act, actN, rows, Cq, w.qScale, w.qOffset);
  if (hasBias && !shared) layerNorm(biasSrc, biasN, nbRows, Cb, w.bScale, w.bOffset);
  if (hasBias) {
    Gemm gm{}; gm.X = biasN; gm.W = w.feat2dW; gm.Y = nbRow;
    gm.rows = nbRows; gm.in = Cb; gm.out = H; gemm(gm);          // [Q * K][H]
    run1d("lf_tri_pack", nbRows * H, TriPackArgs{nbRow, nb, nbRows * H, nbRows, (uint)H, 0});   // -> [H][Q * K]
  }

  // -------- the attention module. Self-attention, so one cotangent for q_data and m_data.
  attentionModuleBwd(actN, actN, maskBias, nb, w.attn, dOut, dActN, dActN, dNb,
                     B, Q, K, Cq, Cq, H, D, O, g.attn);

  // -------- the bias's linear: [H][Q * K] back to [Q * K][H], then through feat2dW
  if (!hasBias) {
    layerNormBwd(act, dActN, dAct, rows, Cq, w.qScale, g.qScale, g.qOffset);
    return;
  }
  run1d("lf_tri_unpack", nbRows * H, TriPackArgs{dNb, dNbRow, nbRows * H, nbRows, (uint)H, 0});
  if (shared) {
    // the same tensor feeds the attention and the bias, so this accumulates into the attention's own cotangent
    linearBwd(biasN, w.feat2dW, dNbRow, dActN, g.feat2dW, nullptr, nbRows, Cb, H, true);
  } else {
    float* dBiasN = buf("dBiasN", nbRows * Cb);
    linearBwd(biasN, w.feat2dW, dNbRow, dBiasN, g.feat2dW, nullptr, nbRows, Cb, H, false);
    layerNormBwd(biasSrc, dBiasN, dBiasSrc, nbRows, Cb, w.bScale, g.bScale, g.bOffset);
  }

  // -------- and the attended tensor's own LayerNorm
  layerNormBwd(act, dActN, dAct, rows, Cq, w.qScale, g.qScale, g.qOffset);
}

void msaColumnAttentionBwd(const float* msa, const float* msaMask, int S, int N, int Cm, int H, int D, int O,
                           const BiasedAttnWeights& w, const float* dOut, float* dMsa,
                           const BiasedAttnGrads& g) {
  if (!S || !N) return;
  if (w.feat2dW) { fprintf(stderr, "design: MSAColumnAttention takes no pair bias\n"); exit(1); }
  auto buf = [&](const char* name, size_t n) {
    return (float*)scratchBytes(std::string("design.mcol.") + name, n * sizeof(float));
  };
  const size_t rows = (size_t)S * N;
  float* actT = buf("actT", rows * Cm);        // [N][S][Cm]
  float* maskT = buf("maskT", rows);
  float* dOutT = buf("dOutT", rows * O);
  float* dActT = buf("dActT", rows * Cm);
  auto swap = [&](const float* x, float* y, int A, int Bx, int C) {
    run1d("lf_swap01", (size_t)A * Bx * C,
          Swap01Args{x, y, (size_t)A * Bx * C, (uint)A, (uint)Bx, (uint)C, 0});
  };
  // into the per-column orientation, as AF2 does before the module and after it
  swap(msa, actT, S, N, Cm);
  swap(msaMask, maskT, S, N, 1);
  swap(dOut, dOutT, S, N, O);
  // a residue is a batch, the sequences are the queries
  biasedSelfAttentionBwd(actT, actT, maskT, N, S, Cm, Cm, H, D, O, w, dOutT, dActT, dActT, g);
  // and back, which is the swap read the other way round
  run1d("lf_swap01", rows * Cm, Swap01Args{dActT, dMsa, rows * Cm, (uint)N, (uint)S, (uint)Cm, 0});
}

void msaColumnGlobalAttentionBwd(const float* msa, const float* msaMask, int S, int N, int Cm, int H, int D,
                                 int O, const GlobalAttnWeights& w, const float* dOut, float* dMsa,
                                 const GlobalAttnGrads& g) {
  if (!S || !N) return;
  if (!w.Wg) { fprintf(stderr, "design: MSAColumnGlobalAttention needs its gate\n"); exit(1); }
  const int B = N, K = S, W = H * D;           // after the swap: a residue a batch, the sequences the keys
  const size_t rows = (size_t)B * K;
  const float scale = 1.f / std::sqrt((float)D);
  auto buf = [&](const char* name, size_t n) {
    return (float*)scratchBytes(std::string("design.gattn.") + name, n * sizeof(float));
  };
  float* actT = buf("actT", rows * Cm);
  float* maskT = buf("maskT", rows);
  float* dOutT = buf("dOutT", rows * O);
  float* actN = buf("actN", rows * Cm);
  float* qavg = buf("qavg", (size_t)B * Cm);
  float* minv = buf("minv", (size_t)B);
  float* q = buf("q", (size_t)B * W);
  float* kk = buf("k", rows * D);
  float* vv = buf("v", rows * D);
  float* weights = buf("weights", (size_t)B * H * K);
  float* wa = buf("wa", (size_t)B * W);
  float* gpre = buf("gpre", rows * W);
  float* wag = buf("wag", rows * W);
  float* dwag = buf("dwag", rows * W);
  float* dgpre = buf("dgpre", rows * W);
  float* dwa = buf("dwa", (size_t)B * W);
  float* dlogits = buf("dlogits", (size_t)B * H * K);
  float* dq = buf("dq", (size_t)B * W);
  float* dkk = buf("dk", rows * D);
  float* dvv = buf("dv", rows * D);
  float* dqavg = buf("dqavg", (size_t)B * Cm);
  float* dActN = buf("dActN", rows * Cm);
  float* dActT = buf("dActT", rows * Cm);

  auto swap = [&](const float* x, float* y, int A, int Bx, int C) {
    run1d("lf_swap01", (size_t)A * Bx * C, Swap01Args{x, y, (size_t)A * Bx * C, (uint)A, (uint)Bx, (uint)C, 0});
  };
  auto lin = [&](const float* X, const float* Wm, const float* b, float* Y, size_t r, int in, int out) {
    Gemm gm{}; gm.X = X; gm.W = Wm; gm.Y = Y; gm.rows = r; gm.in = in; gm.out = out; gm.bias = b; gemm(gm);
  };
  GattnArgs base{};
  base.B = (uint)B; base.K = (uint)K; base.C = (uint)Cm; base.H = (uint)H; base.D = (uint)D;

  // -------- into the per-column orientation, then the forward, recomputed
  swap(msa, actT, S, N, Cm);
  swap(msaMask, maskT, S, N, 1);
  swap(dOut, dOutT, S, N, O);
  layerNorm(actT, actN, rows, Cm, w.qScale, w.qOffset);
  {
    GattnArgs a = base; a.qData = actN; a.mask = maskT; a.wqavg = qavg; a.wminv = minv;
    a.n = (size_t)B * Cm;
    run1d("lf_gattn_qavg", a.n, a);
  }
  lin(qavg, w.Wq, nullptr, q, (size_t)B, Cm, W);
  run1d("lf_scale", (size_t)B * W, ScaleArgs{q, (size_t)B * W, scale, 0});
  lin(actN, w.Wk, nullptr, kk, rows, Cm, D);
  lin(actN, w.Wv, nullptr, vv, rows, Cm, D);
  {
    GattnArgs a = base; a.q = q; a.k = kk; a.v = vv; a.mask = maskT;
    a.wweights = weights; a.wwa = wa; a.n = (size_t)B * H;
    size_t groups = (a.n + 7) / 8;
    Grid gr = groups <= 65535 ? Grid{(uint32_t)groups, 1, 1}
                              : Grid{65535, (uint32_t)((groups + 65534) / 65535), 1};
    dispatch("lf_gattn_fwd", &a, sizeof a, gr, 256, 0, "design.gattn.fwd");
  }
  lin(actN, w.Wg, w.bg, gpre, rows, Cm, W);
  {
    GattnArgs a = base; a.wa = wa; a.gpre = gpre; a.dgpre = wag; a.n = rows * W;
    run1d("lf_gattn_gate_fwd", a.n, a);        // (dgpre is the write target here)
  }

  // -------- the output projection, then the gate that broadcast the single query
  linearBwd(wag, w.Wo, dOutT, dwag, g.Wo, g.bo, rows, W, O);
  {
    GattnArgs a = base; a.wa = wa; a.gpre = gpre; a.dwag = dwag; a.dgpre = dgpre; a.dwa = dwa;
    a.n = (size_t)B * W;
    run1d("lf_gattn_gate_bwd", a.n, a);
  }
  linearBwd(actN, w.Wg, dgpre, dActN, g.Wg, g.bg, rows, Cm, W, false);   // the first contributor to dActN

  // -------- the softmax, then the three projections
  {
    GattnArgs a = base; a.weights = weights; a.v = vv; a.dwa = dwa; a.dlogits = dlogits; a.n = (size_t)B * H;
    run1d("lf_gattn_dlogits", a.n, a);
  }
  {
    GattnArgs a = base; a.dlogits = dlogits; a.k = kk; a.dq = dq; a.n = (size_t)B * W;
    run1d("lf_gattn_dq", a.n, a);
  }
  run1d("lf_scale", (size_t)B * W, ScaleArgs{dq, (size_t)B * W, scale, 0});
  linearBwd(qavg, w.Wq, dq, dqavg, g.Wq, nullptr, (size_t)B, Cm, W);
  {
    GattnArgs a = base; a.weights = weights; a.dlogits = dlogits; a.dwa = dwa; a.q = q;
    a.dv = dvv; a.dk = dkk; a.n = rows * D;
    run1d("lf_gattn_dv", a.n, a);
  }
  linearBwd(actN, w.Wk, dkk, dActN, g.Wk, nullptr, rows, Cm, D, true);
  linearBwd(actN, w.Wv, dvv, dActN, g.Wv, nullptr, rows, Cm, D, true);
  {   // and the masked mean's own path, which accumulates
    GattnArgs a = base; a.dq = dqavg; a.mask = maskT; a.wminv = minv; a.dqData = dActN; a.n = rows * Cm;
    run1d("lf_gattn_dqavg", a.n, a);
  }

  // -------- the LayerNorm, and back to the row orientation
  layerNormBwd(actT, dActN, dActT, rows, Cm, w.qScale, g.qScale, g.qOffset);
  run1d("lf_swap01", rows * Cm, Swap01Args{dActT, dMsa, rows * Cm, (uint)B, (uint)K, (uint)Cm, 0});
}

// a batched GEMM over the channel: every operand is [batch][rows * in] with one plane a channel
static void batched(const float* X, bool transX, const float* W, bool transW, float* Y,
                    size_t rows, int in, int out, int batch, size_t plane, float beta = 0.f) {
  Gemm g{};
  g.X = X; g.transX = transX;
  g.W = W; g.transW = transW;
  g.Y = Y; g.beta = beta;
  g.rows = rows; g.in = in; g.out = out;
  g.batch = batch;
  g.sx = (int64_t)plane; g.sw = (int64_t)plane; g.sy = (int64_t)plane;
  gemm(g);
}

void triangleMultiplicationBwd(const float* pair, const float* mask, int N, int C, int I, bool outgoing,
                               const TriMulWeights& w, const float* dOut, float* dPair, const TriMulGrads& gr) {
  if (!N) return;
  const size_t n2 = (size_t)N * N;
  const size_t plane = n2;                     // one channel's [N][N] matrix
  auto rowBuf = [&](const char* name, int width) {
    return (float*)scratchBytes(std::string("design.tri.") + name, n2 * (size_t)width * sizeof(float));
  };
  float* act0 = rowBuf("act0", C);
  float* aM = rowBuf("aM", I);
  float* bM = rowBuf("bM", I);
  float* gaPre = rowBuf("gaPre", I);
  float* gbPre = rowBuf("gbPre", I);
  float* gPre = rowBuf("gPre", C);
  float* arow = rowBuf("arow", I);
  float* brow = rowBuf("brow", I);
  float* prow = rowBuf("prow", I);
  float* q = rowBuf("q", I);
  float* o = rowBuf("o", C);
  float* tmpI = rowBuf("tmpI", I);
  float* tmpI2 = rowBuf("tmpI2", I);
  float* tmpC = rowBuf("tmpC", C);
  float* A = rowBuf("A", I);                   // the packed [I][n2] planes
  float* B = rowBuf("B", I);
  float* P = rowBuf("P", I);
  // REUSED, not allocated: three buffers are provably dead by the time these are wanted, and a pair-sized buffer is
  // 18 MB at N=192, so naming them here saves 54. If a lifetime below is wrong the test suite fails - the gradients
  // are what these feed.
  //   q    the output projection consumed it, before the product's cotangent exists
  //   prow the centre LayerNorm consumed it, likewise
  //   gPre the output gate consumed it, at the very start
  float* dA = q;
  float* dB = prow;
  float* dln = gPre;

  auto linFwd = [&](const float* X, const float* W, const float* b, float* Y, int in, int out) {
    Gemm g{}; g.X = X; g.W = W; g.Y = Y; g.rows = n2; g.in = in; g.out = out; g.bias = b; gemm(g);
  };
  auto pack = [&](const float* src, float* dst, int width) {
    run1d("lf_tri_pack", n2 * width, TriPackArgs{src, dst, n2 * width, n2, (uint)width, 0});
  };
  auto unpack = [&](const float* src, float* dst, int width) {
    run1d("lf_tri_unpack", n2 * width, TriPackArgs{src, dst, n2 * width, n2, (uint)width, 0});
  };
  auto maskMul = [&](float* y, int width) {
    run1d("lf_opm_mask", n2 * width, OpmMaskArgs{y, mask, n2 * width, (uint)N, (uint)width, 0, 0});
  };

  // ---------------- the forward, recomputed
  layerNorm(pair, act0, n2, C, w.lnScale, w.lnOffset);
  linFwd(act0, w.Wl, w.bl, aM, C, I);
  maskMul(aM, I);
  linFwd(act0, w.Wr, w.br, bM, C, I);
  maskMul(bM, I);
  linFwd(act0, w.Wlg, w.blg, gaPre, C, I);
  linFwd(act0, w.Wrg, w.brg, gbPre, C, I);
  run1d("lf_mul_sigmoid", n2 * I, MulSigmoidArgs{aM, gaPre, arow, n2 * (size_t)I});
  run1d("lf_mul_sigmoid", n2 * I, MulSigmoidArgs{bM, gbPre, brow, n2 * (size_t)I});
  pack(arow, A, I);
  pack(brow, B, I);
  // outgoing p_c = A_c B_c^T (B_c stored [j][k], read transposed);  incoming p_c = B_c^T A_c
  if (outgoing) batched(A, false, B, true, P, N, N, N, I, plane);
  else          batched(B, true, A, false, P, N, N, N, I, plane);
  unpack(P, prow, I);
  layerNorm(prow, q, n2, I, w.centerScale, w.centerOffset);
  linFwd(q, w.Wo, w.bo, o, I, C);
  linFwd(act0, w.Wg, w.bg, gPre, C, C);

  // ---------------- the output gate, and the two paths out of it
  // out = o * sigmoid(gPre): tmpC gets d/do, and the gate's own cotangent goes back through its linear
  float* dgPre = rowBuf("dgPre", C);
  gateBwd(dOut, o, gPre, tmpC, dgPre, n2 * (size_t)C);   // (gPre is read here and reused as dln at the end)
  linearBwd(act0, w.Wg, dgPre, dPair, gr.Wg, gr.bg, n2, C, C, false);   // the first contributor to dact0
  float* dact0 = dPair;                                                 // accumulated in place from here on

  // ---------------- the output projection and the centre LayerNorm
  linearBwd(q, w.Wo, tmpC, tmpI, gr.Wo, gr.bo, n2, I, C);
  layerNormBwd(prow, tmpI, tmpI2, n2, I, w.centerScale, gr.centerScale, gr.centerOffset);

  // ---------------- the product. dP is packed in place of P, which is finished with
  float* dP = P;
  pack(tmpI2, dP, I);
  if (outgoing) {
    // p[i][j] = sum_k a[i][k] b[j][k]:  da = dP B,  db = dP^T A
    batched(dP, false, B, false, dA, N, N, N, I, plane);
    batched(dP, true, A, false, dB, N, N, N, I, plane);
  } else {
    // p[i][j] = sum_k a[k][j] b[k][i]:  da = B dP,  db = A dP^T
    batched(B, false, dP, false, dA, N, N, N, I, plane);
    batched(A, false, dP, true, dB, N, N, N, I, plane);
  }
  unpack(dA, arow, I);     // (arow and brow are finished with: the packed copies were the operands)
  unpack(dB, brow, I);

  // ---------------- the two gated projections, then the input LayerNorm
  // a = aM * sigmoid(gaPre), and aM = mask * linear, so the mask multiplies the cotangent too
  gateBwd(arow, aM, gaPre, tmpI, tmpI2, n2 * (size_t)I);
  maskMul(tmpI, I);
  linearBwd(act0, w.Wl, tmpI, dact0, gr.Wl, gr.bl, n2, C, I, true);
  linearBwd(act0, w.Wlg, tmpI2, dact0, gr.Wlg, gr.blg, n2, C, I, true);

  gateBwd(brow, bM, gbPre, tmpI, tmpI2, n2 * (size_t)I);
  maskMul(tmpI, I);
  linearBwd(act0, w.Wr, tmpI, dact0, gr.Wr, gr.br, n2, C, I, true);
  linearBwd(act0, w.Wrg, tmpI2, dact0, gr.Wrg, gr.brg, n2, C, I, true);

  // dact0 is dPair's buffer; the LayerNorm reads it and overwrites it with the module's own cotangent
  copy(dln, dact0, n2 * (size_t)C * sizeof(float));
  layerNormBwd(pair, dln, dPair, n2, C, w.lnScale, gr.lnScale, gr.lnOffset);
}

}  // namespace mt
