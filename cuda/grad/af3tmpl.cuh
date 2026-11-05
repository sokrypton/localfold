// The AF3 lineage's template embedder for training (cuda/af3/src/trunk.cuh's templateEmbedding is the forward):
//
//   query = LN(pair) @ (templatePairEmbedding8 | zProjection)                         the same for every slot
//   per slot k with a weight:  act = query + features_k
//       nine-projection embedder (af3, intellifold2, openbind0, opendde, mol1):
//         features = onehot(aatype)[j] W2 + onehot(aatype)[i] W3 + W0[bin] + pb W1 + uv . (W4, W5, W6) + bb W7
//       fused (protenix2, boltz2, rf3, chai-1):  features = dense feature columns @ aProjection  (+ chai's bias)
//       act = blocks(act) (+ act, boltz2's outer residual);  y_k = LN(act) (* chai's per-template coverage)
//   pair += relu(sum_k repeat_k y_k / (1e-7 + templates)) @ outputLinear
//
// The blocks are the pairformer's without a single track (af3.cuh), chai-1's parallel. Saved for the backward: the
// query's LayerNorm, and each slot's stack input and output; the stack is replayed block by block.
#pragma once
#include "af3.cuh"

namespace grad {
namespace af3 {

struct TemplateEmbedder {
  LayerNorm qln, outLn;
  Linear query, aProj, w2, w3, outLin;
  Param w0, w1, w4, w5, w6, w7, featBias;
  std::vector<Pairformer> blocks;
  Dialect d;
  bool fused = false, outer = false, chaiMask = false, hasFeatBias = false;
  int Cq = 0, Ct = 0, bins = 0, width = 0;
};
// every parameter under "trunk.template.", the port's names; has(name) asks the bundle
inline void forEachParam(TemplateEmbedder& m, int Cq, int Ct, int width, int bins, int nb, int Hp, int Dp, int pairFactor,
                         bool gridBiases, bool parallel, const std::function<bool(const std::string&)>& has,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string P = "trunk.template.";
  m.Cq = Cq; m.Ct = Ct; m.width = width; m.bins = bins;
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; l.w.fanIn = K; f(P + n, l.w, (size_t)K * N); };
  m.qln.C = Cq; f(P + "queryEmbeddingNormScale", m.qln.g, Cq); f(P + "queryEmbeddingNormOffset", m.qln.b, Cq);
  if (m.fused) {
    lin("zProjection", m.query, Cq, Ct);
    lin("aProjection", m.aProj, width, Ct);
  } else {
    lin("templatePairEmbedding8", m.query, Cq, Ct);
    lin("templatePairEmbedding2", m.w2, 31, Ct);
    lin("templatePairEmbedding3", m.w3, 31, Ct);
    f(P + "templatePairEmbedding0", m.w0, (size_t)bins * Ct);
    f(P + "templatePairEmbedding1", m.w1, Ct); f(P + "templatePairEmbedding4", m.w4, Ct);
    f(P + "templatePairEmbedding5", m.w5, Ct); f(P + "templatePairEmbedding6", m.w6, Ct);
    f(P + "templatePairEmbedding7", m.w7, Ct);
  }
  m.hasFeatBias = has(P + "templateFeatureBias");          // (chai-1's, once on the stack's input, in either form)
  if (m.hasFeatBias) f(P + "templateFeatureBias", m.featBias, Ct);
  m.blocks.resize(nb);
  for (int b = 0; b < nb; ++b)
    forEachParam(m.blocks[b], P + "blocks." + std::to_string(b), Ct, 0, Hp, Dp, 0, 0, pairFactor, 0, gridBiases, false, f, true,
                 parallel, ".");
  m.outLn.C = Ct; f(P + "outputLayerNormScale", m.outLn.g, Ct); f(P + "outputLayerNormOffset", m.outLn.b, Ct);
  lin("outputLinear", m.outLin, Ct, Cq);
}
// one slot's inputs (the exporter's "template.<k>." tensors)
struct TemplateSlot {
  real repeat = 0;
  const int* aatype = nullptr;                  // [n]
  const int* bin = nullptr;                     // [n, n], -1 none (null: no geometry, a padded slot)
  const real *pb = nullptr, *uv = nullptr, *bb = nullptr;   // [n, n], [n, n, 3], [n, n]
  const real* dense = nullptr;                  // fused: [n, n, width]
  const real* coverage = nullptr;               // chai-1: [n, n], the slot's output mask
};
struct TemplateSave {
  real *qn = nullptr, *query = nullptr, *summed = nullptr;
  LnSave lq;
  std::vector<real*> ins, outs;                 // per slot: the stack's input, its output (the output norm's input)
  std::vector<LnSave> lo;
  int n = 0, templates = 0;
};
__global__ void tmplOneHotK(const int* aatype, real* oh, int n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * 31) return;
  oh[t] = aatype[t / 31] == (int)(t % 31) ? (real)1 : 0;
}
// act[i][j] += row[j] + col[i]
__global__ void tmplRowColK(real* act, const real* row, const real* col, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  act[t] += row[(size_t)j * C + c] + col[(size_t)i * C + c];
}
// drow[j] = sum_i d[i][j], dcol[i] = sum_j d[i][j]
__global__ void tmplRowColBwdK(const real* d, real* drow, real* dcol, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int a = (int)(t / C), c = (int)(t % C);
  real r = 0, k = 0;
  for (int b = 0; b < n; ++b) { r += d[((size_t)b * n + a) * C + c]; k += d[((size_t)a * n + b) * C + c]; }
  drow[t] = r; dcol[t] = k;
}
__global__ void tmplGeometryK(real* act, const int* bin, const real* pb, const real* uv, const real* bb, const real* W0,
                              const real* W1, const real* W4, const real* W5, const real* W6, const real* W7, size_t pairs, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * C) return;
  int c = (int)(t % C); size_t p = t / C;
  real v = 0;
  int b = bin[p];
  if (b >= 0) v += W0[(size_t)b * C + c];
  v += pb[p] * W1[c] + uv[p * 3] * W4[c] + uv[p * 3 + 1] * W5[c] + uv[p * 3 + 2] * W6[c] + bb[p] * W7[c];
  act[t] += v;
}
// the geometry's weights: one thread per (row k of W0, or one of the five scalar features, channel c), over every pair
__global__ void tmplGeometryBwdK(const real* d, const int* bin, const real* pb, const real* uv, const real* bb, real* dW0,
                                 real* dW1, real* dW4, real* dW5, real* dW6, real* dW7, size_t pairs, int C, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)(bins + 5) * C) return;
  int k = (int)(t / C), c = (int)(t % C);
  real acc = 0;
  for (size_t p = 0; p < pairs; ++p) {
    real g = d[p * C + c];
    if (k < bins) { if (bin[p] == k) acc += g; }
    else if (k == bins) acc += pb[p] * g;
    else if (k < bins + 4) acc += uv[p * 3 + (k - bins - 1)] * g;
    else acc += bb[p] * g;
  }
  if (k < bins) dW0[(size_t)k * C + c] += acc;
  else (k == bins ? dW1 : k == bins + 1 ? dW4 : k == bins + 2 ? dW5 : k == bins + 3 ? dW6 : dW7)[c] += acc;
}
__global__ void tmplReluScaleK(const real* x, real* y, real s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = fmax((real)0, x[t] * s);
}
__global__ void tmplReluScaleBwdK(const real* x, real* d, real s, size_t n) {    // d = d * s where x s > 0
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) d[t] = x[t] * s > 0 ? d[t] * s : (real)0;
}
__global__ void tmplAddScaledK(real* y, const real* x, real s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] += s * x[t];
}
// act = query + slot k's features (the stack's input)
inline void templateInput(TemplateEmbedder& m, const TemplateSlot& k, const real* query, real* act, int n) {
  const size_t pairs = (size_t)n * n, PC = pairs * m.Ct;
  GK(cudaMemcpyAsync(act, query, PC * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  Scratch sc;
  if (m.hasFeatBias) biasK<<<blocks(PC), 256, 0, STREAM>>>(act, m.featBias.w, pairs, m.Ct);
  if (m.fused) { m.aProj.forward(k.dense, act, pairs, 1); return; }
  real* oh = ARENA.get((size_t)n * 31); tmplOneHotK<<<blocks((size_t)n * 31), 256, 0, STREAM>>>(k.aatype, oh, n);
  real *row = ARENA.get((size_t)n * m.Ct), *col = ARENA.get((size_t)n * m.Ct);
  m.w2.forward(oh, row, n); m.w3.forward(oh, col, n);
  tmplRowColK<<<blocks(PC), 256, 0, STREAM>>>(act, row, col, n, m.Ct);
  if (k.bin) tmplGeometryK<<<blocks(PC), 256, 0, STREAM>>>(act, k.bin, k.pb, k.uv, k.bb, m.w0.w, m.w1.w, m.w4.w, m.w5.w, m.w6.w,
                                                         m.w7.w, pairs, m.Ct);
}
inline void tmplStackForward(TemplateEmbedder& m, real* z, const real* pairMask, int n) {
  for (auto& b : m.blocks) {
    Scratch sc;
    pairformerForward(b, m.d, z, nullptr, pairMask, pairMask, nullptr, nullptr, z, nullptr, n);
  }
}
// pair [n, n, Cq] -> out (+)= the template term (out may be pair); templates the divisor (every slot, the exporter's)
inline TemplateSave templateForward(TemplateEmbedder& m, const std::vector<TemplateSlot>& slots, int templates,
                                    const real* pair, const real* pairMask, int n, real* out) {
  TemplateSave s{}; s.n = n; s.templates = templates;
  const size_t pairs = (size_t)n * n, PC = pairs * m.Ct;
  s.qn = ARENA.get(pairs * m.Cq); s.lq = lnForward(m.qln, pair, s.qn, pairs);
  s.query = ARENA.get(PC); m.query.forward(s.qn, s.query, pairs);
  s.summed = ARENA.zeros(PC);
  s.ins.assign(slots.size(), nullptr); s.outs.assign(slots.size(), nullptr); s.lo.assign(slots.size(), LnSave{});
  real* y = ARENA.get(PC);                       // (each slot's normalised output, summed and not kept)
  for (size_t k = 0; k < slots.size(); ++k) {
    if (slots[k].repeat == 0) continue;
    s.ins[k] = ARENA.get(PC);
    templateInput(m, slots[k], s.query, s.ins[k], n);
    s.outs[k] = cloneR(s.ins[k], PC);
    tmplStackForward(m, s.outs[k], pairMask, n);
    if (m.outer) add(s.outs[k], s.ins[k], PC);
    s.lo[k] = lnForward(m.outLn, s.outs[k], y, pairs);
    if (m.chaiMask && slots[k].coverage) rowMaskK<<<blocks(PC), 256, 0, STREAM>>>(y, slots[k].coverage, pairs, m.Ct);
    tmplAddScaledK<<<blocks(PC), 256, 0, STREAM>>>(s.summed, y, slots[k].repeat, PC);
  }
  Scratch sc;
  real* h = ARENA.get(PC);
  tmplReluScaleK<<<blocks(PC), 256, 0, STREAM>>>(s.summed, h, (real)(1 / (1e-7 + templates)), PC);
  m.outLin.forward(h, out, pairs, 1);
  return s;
}
// dout -> every parameter; dpair added (the query's path only: the residual into the pair is the caller's)
inline void templateBackward(TemplateEmbedder& m, const std::vector<TemplateSlot>& slots, const TemplateSave& s,
                             const real* pairMask, const real* dout, real* dpair) {
  Scratch scratch;
  const int n = s.n; const size_t pairs = (size_t)n * n, PC = pairs * m.Ct;
  const real scale = (real)(1 / (1e-7 + s.templates));
  real* h = ARENA.get(PC);
  tmplReluScaleK<<<blocks(PC), 256, 0, STREAM>>>(s.summed, h, scale, PC);
  real* dsum = ARENA.get(PC); m.outLin.backward(h, dout, dsum, pairs);
  tmplReluScaleBwdK<<<blocks(PC), 256, 0, STREAM>>>(s.summed, dsum, scale, PC);
  real* dq = ARENA.zeros(PC);
  for (size_t k = 0; k < slots.size(); ++k) {
    if (slots[k].repeat == 0) continue;
    Scratch sc;
    real* dy = ARENA.zeros(PC);
    tmplAddScaledK<<<blocks(PC), 256, 0, STREAM>>>(dy, dsum, slots[k].repeat, PC);
    if (m.chaiMask && slots[k].coverage) rowMaskK<<<blocks(PC), 256, 0, STREAM>>>(dy, slots[k].coverage, pairs, m.Ct);
    real* da = ARENA.get(PC); lnBackward(m.outLn, s.lo[k], dy, da, pairs, false);
    real* dIn = m.outer ? cloneR(da, PC) : nullptr;              // (boltz2: the stack's input reads the residual too)
    // the stack, replayed block by block from its input
    std::vector<real*> ins(m.blocks.size());
    for (size_t b = 0; b < m.blocks.size(); ++b) {
      ins[b] = b == 0 ? s.ins[k] : cloneR(ins[b - 1], PC);
      if (b > 0) { Scratch t; pairformerForward(m.blocks[b - 1], m.d, ins[b], nullptr, pairMask, pairMask, nullptr, nullptr, ins[b], nullptr, n); }
    }
    for (size_t b = m.blocks.size(); b-- > 0;) {
      Scratch t;
      real* po = ARENA.get(PC);
      auto bs = pairformerForward(m.blocks[b], m.d, ins[b], nullptr, pairMask, pairMask, nullptr, nullptr, po, nullptr, n);
      pairformerBackward(m.blocks[b], m.d, bs, pairMask, pairMask, nullptr, da, nullptr, da, nullptr, nullptr);
    }
    if (dIn) add(da, dIn, PC);
    // the slot's features
    add(dq, da, PC);
    if (m.hasFeatBias) colSum(m.featBias.g, da, pairs, m.Ct);
    if (m.fused) {
      m.aProj.backward(slots[k].dense, da, nullptr, pairs);
    } else {
      real* oh = ARENA.get((size_t)n * 31); tmplOneHotK<<<blocks((size_t)n * 31), 256, 0, STREAM>>>(slots[k].aatype, oh, n);
      real *drow = ARENA.get((size_t)n * m.Ct), *dcol = ARENA.get((size_t)n * m.Ct);
      tmplRowColBwdK<<<blocks((size_t)n * m.Ct), 256, 0, STREAM>>>(da, drow, dcol, n, m.Ct);
      m.w2.backward(oh, drow, nullptr, n); m.w3.backward(oh, dcol, nullptr, n);
      if (slots[k].bin)
        tmplGeometryBwdK<<<blocks((size_t)(m.bins + 5) * m.Ct), 256, 0, STREAM>>>(da, slots[k].bin, slots[k].pb, slots[k].uv,
          slots[k].bb, m.w0.g, m.w1.g, m.w4.g, m.w5.g, m.w6.g, m.w7.g, pairs, m.Ct, m.bins);
    }
  }
  real* dqn = ARENA.get(pairs * m.Cq); m.query.backward(s.qn, dq, dqn, pairs);
  lnBackward(m.qln, s.lq, dqn, dpair, pairs, true);
}

}  // namespace af3
}  // namespace grad
