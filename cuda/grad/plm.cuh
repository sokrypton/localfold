// The protein language-model towers for training: ESM-C (cuda/plm/esmc.cuh) and ESM2 (cuda/plm/esm2.cuh) as one
// pre-LayerNorm transformer block with flags, forward with saves and backward into every parameter.
//
//   ESM-C:  h = LN(x; attn_norm) qkv;  q, k = LN(q; q_norm), LN(k; k_norm) over the full width, scale only;
//           RoPE (split halves, head 64, position = row); attention within a chain; x += h_attn Wout / r;
//           x += swiglu(LN(x; ffn_norm) W1) W2 / r   (r = sqrt(layers / 36)); out = LN(x; final_norm), scale only
//   ESM2:   biases on every Linear, no q/k norms, x = embed * 0.88, exact GELU, final LN with offset
//
// The sequence head (ESM-C's: Linear, GELU, LayerNorm, Linear to the vocabulary) and a masked-LM cross entropy
// close it into a loss. Rematerialised per block like cuda/grad/af2.cuh's stack: each block's input kept, its
// forward replayed before its backward.
#pragma once
#include "pair.cuh"

namespace grad {
namespace plm {

struct Config {
  int C = 1152, H = 18, D = 64, F = 3072, layers = 36, vocab = 64;
  bool swiglu = true, qkNorm = true, bias = false, finalOffset = false;
  real residual = 1;      // the residual branch's multiplier, 1 / sqrt(layers / 36) for ESM-C
  real embedScale = 1;    // ESM2: 1 - 0.15 * 0.8
  double ropeBase = 10000;
};

struct Block {
  LayerNorm ln1, qn, kn, ln2;
  Linear qkv, out, fc1, fc2;
};
struct Head {             // ESM-C's sequence head
  Linear dense, out;
  LayerNorm ln;
  bool present = false;
};
struct Tower {
  Config cfg;
  Param embed;
  std::vector<Block> blocks;
  LayerNorm final;
  Head head;
  Params params;
};

// every parameter, by the bundle's name (cuda/plm's), with its size
inline void forEachParam(Tower& t, const std::function<void(const std::string&, Param&, size_t)>& f) {
  const Config& c = t.cfg;
  auto ln = [&](const std::string& n, LayerNorm& l, int C, bool offset) {
    l.C = C; f(n + "/scale", l.g, C); if (offset) f(n + "/offset", l.b, C);
  };
  auto lin = [&](const std::string& n, Linear& l, int K, int N, bool bias) {
    l.K = K; l.N = N; l.hasBias = bias; f(n + "/weights", l.w, (size_t)K * N); if (bias) f(n + "/bias", l.b, N);
  };
  f("embed/weights", t.embed, (size_t)c.vocab * c.C);
  t.blocks.resize(c.layers);
  for (int l = 0; l < c.layers; ++l) {
    Block& b = t.blocks[l];
    std::string B = "blocks/" + std::to_string(l) + "/";
    ln(B + "attn_norm", b.ln1, c.C, true);
    lin(B + "qkv", b.qkv, c.C, 3 * c.C, c.bias);
    if (c.qkNorm) { ln(B + "q_norm", b.qn, c.C, false); ln(B + "k_norm", b.kn, c.C, false); }
    lin(B + "attn_out", b.out, c.C, c.C, c.bias);
    ln(B + "ffn_norm", b.ln2, c.C, true);
    lin(B + "fc1", b.fc1, c.C, c.swiglu ? 2 * c.F : c.F, c.bias);
    lin(B + "fc2", b.fc2, c.F, c.C, c.bias);
  }
  ln("final_norm", t.final, c.C, c.finalOffset);
  if (t.head.present) {
    lin("head/dense", t.head.dense, c.C, c.C, true);
    ln("head/norm", t.head.ln, c.C, true);
    lin("head/out", t.head.out, c.C, c.vocab, true);
  }
}
inline void allocTower(Tower& t) {
  forEachParam(t, [&](const std::string& n, Param& p, size_t len) { allocParam(p, n, len); t.params.add(p); });
}

inline void copyCols(const real* x, int ldx, real* y, int ldy, size_t rows, int cols, bool accumulate = false) {
  copyColsK<<<blocks(rows * cols), 256, 0, STREAM>>>(x, ldx, y, ldy, rows, cols, accumulate);
}
// y = x + a * u
__global__ void addScaledK(real* y, const real* x, const real* u, real a, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = x[t] + a * u[t];
}

struct BlockSave {
  const real* x0;
  LnSave l1, lq, lk, l2;
  real *xn1, *qkv, *qraw, *kraw, *q, *k, *ctx, *x1, *xn2, *h;
};
inline Sdpa towerSdpa(const Config& c, int R, const int* seq) {
  Sdpa a;
  a.B = 1; a.H = c.H; a.Q = R; a.K = R; a.D = c.D; a.scale = 1 / sqrt((real)c.D);
  a.seqQ = seq; a.seqK = seq;
  return a;
}

// x0 [R, C] -> x2 [R, C]
inline BlockSave blockForward(const Config& c, Block& b, const real* x0, real* x2, size_t R, const int* seq,
                              const int* pos) {
  const int C = c.C, F = c.F, F1 = c.swiglu ? 2 * F : F;
  BlockSave s{};
  s.x0 = x0;
  s.xn1 = ARENA.get(R * C);
  s.l1 = lnForward(b.ln1, x0, s.xn1, R);
  s.qkv = ARENA.get(R * 3 * C);
  b.qkv.forward(s.xn1, s.qkv, R);
  s.q = ARENA.get(R * C); s.k = ARENA.get(R * C);
  if (c.qkNorm) {
    s.qraw = ARENA.get(R * C); s.kraw = ARENA.get(R * C);
    copyCols(s.qkv, 3 * C, s.qraw, C, R, C);
    copyCols(s.qkv + C, 3 * C, s.kraw, C, R, C);
    s.lq = lnForward(b.qn, s.qraw, s.q, R);
    s.lk = lnForward(b.kn, s.kraw, s.k, R);
  } else {
    copyCols(s.qkv, 3 * C, s.q, C, R, C);
    copyCols(s.qkv + C, 3 * C, s.k, C, R, C);
  }
  rope(s.q, (int)R, c.H, c.D, C, pos, c.ropeBase);
  rope(s.k, (int)R, c.H, c.D, C, pos, c.ropeBase);
  s.ctx = ARENA.get(R * C);
  sdpaForward(towerSdpa(c, (int)R, seq), s.q, C, s.k, C, s.qkv + 2 * C, 3 * C, s.ctx, C);
  s.x1 = ARENA.get(R * C);
  {
    Scratch scratch;
    real* u = ARENA.get(R * C);
    b.out.forward(s.ctx, u, R);
    addScaledK<<<blocks(R * C), 256, 0, STREAM>>>(s.x1, x0, u, c.residual, R * C);
  }
  s.xn2 = ARENA.get(R * C);
  s.l2 = lnForward(b.ln2, s.x1, s.xn2, R);
  s.h = ARENA.get(R * F1);
  b.fc1.forward(s.xn2, s.h, R);
  {
    Scratch scratch;
    real* g = ARENA.get(R * F);
    if (c.swiglu) swigluK<<<blocks(R * F), 256, 0, STREAM>>>(s.h, g, R, F);
    else geluK<<<blocks(R * F), 256, 0, STREAM>>>(g, s.h, R * F);
    real* u = ARENA.get(R * C);
    b.fc2.forward(g, u, R);
    addScaledK<<<blocks(R * C), 256, 0, STREAM>>>(x2, s.x1, u, c.residual, R * C);
  }
  return s;
}

// dx2 -> dx0 (written; may alias dx2)
inline void blockBackward(const Config& c, Block& b, const BlockSave& s, const real* dx2, real* dx0, size_t R,
                          const int* seq, const int* pos) {
  Scratch scratch;
  const int C = c.C, F = c.F, F1 = c.swiglu ? 2 * F : F;
  real* dx1 = ARENA.get(R * C);
  GK(cudaMemcpyAsync(dx1, dx2, R * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  real* du = ARENA.get(R * C);
  GK(cudaMemcpyAsync(du, dx2, R * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  scaleK<<<blocks(R * C), 256, 0, STREAM>>>(du, c.residual, R * C);
  // the feed-forward: u = act(h) W2 + b2
  {
    real* g = ARENA.get(R * F);
    if (c.swiglu) swigluK<<<blocks(R * F), 256, 0, STREAM>>>(s.h, g, R, F);
    else geluK<<<blocks(R * F), 256, 0, STREAM>>>(g, s.h, R * F);
    real* dg = ARENA.get(R * F);
    b.fc2.backward(g, du, dg, R);
    real* dh = ARENA.get(R * F1);
    if (c.swiglu) swigluBwdK<<<blocks(R * F), 256, 0, STREAM>>>(s.h, dg, dh, R, F);
    else { GK(cudaMemcpyAsync(dh, dg, R * F * sizeof(real), cudaMemcpyDeviceToDevice, STREAM)); geluBwdK<<<blocks(R * F), 256, 0, STREAM>>>(dh, s.h, R * F); }
    real* dxn2 = ARENA.get(R * C);
    b.fc1.backward(s.xn2, dh, dxn2, R);
    lnBackward(b.ln2, s.l2, dxn2, dx1, R, true);
  }
  // the attention: u = ctx Wout + b
  GK(cudaMemcpyAsync(du, dx1, R * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  scaleK<<<blocks(R * C), 256, 0, STREAM>>>(du, c.residual, R * C);
  real* dctx = ARENA.get(R * C);
  b.out.backward(s.ctx, du, dctx, R);
  real* dqkv = ARENA.get(R * 3 * C);
  real *dq = ARENA.get(R * C), *dk = ARENA.get(R * C);
  sdpaBackward(towerSdpa(c, (int)R, seq), s.q, C, s.k, C, s.qkv + 2 * C, 3 * C, dctx, C, dq, C, dk, C, dqkv + 2 * C, 3 * C);
  rope(dq, (int)R, c.H, c.D, C, pos, c.ropeBase, true);
  rope(dk, (int)R, c.H, c.D, C, pos, c.ropeBase, true);
  if (c.qkNorm) {
    real *dqr = ARENA.get(R * C), *dkr = ARENA.get(R * C);
    lnBackward(b.qn, s.lq, dq, dqr, R, false);
    lnBackward(b.kn, s.lk, dk, dkr, R, false);
    dq = dqr; dk = dkr;
  }
  copyCols(dq, C, dqkv, 3 * C, R, C);
  copyCols(dk, C, dqkv + C, 3 * C, R, C);
  real* dxn1 = ARENA.get(R * C);
  b.qkv.backward(s.xn1, dqkv, dxn1, R);
  lnBackward(b.ln1, s.l1, dxn1, dx1, R, true);
  GK(cudaMemcpyAsync(dx0, dx1, R * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

// ---------------------------------------------------------------- the whole tower, rematerialised
struct Run {
  std::vector<real*> xs;      // [layers + 1]: each block's input, the last block's output
  real *out = nullptr;        // the final LayerNorm's output [R, C]
  LnSave finalSave;
  size_t R = 0;
  void ensure(const Config& c, size_t R_) {
    if (R_ == R && !xs.empty()) return;
    for (real* p : xs) GK(cudaFree(p));
    if (out) GK(cudaFree(out));
    R = R_;
    xs.assign(c.layers + 1, nullptr);
    for (auto& p : xs) GK(cudaMalloc(&p, R * c.C * sizeof(real)));
    GK(cudaMalloc(&out, R * c.C * sizeof(real)));
  }
};
// ids, seq (chain id a row), pos (RoPE position a row; null = the row) -> run.out
inline void towerForward(Tower& t, Run& run, const int* ids, const int* seq, const int* pos, size_t R,
                         int upToLayers = -1) {
  const Config& c = t.cfg;
  int L = upToLayers < 0 ? c.layers : upToLayers;
  run.ensure(c, R);
  embedK<<<blocks(R * c.C), 256, 0, STREAM>>>(ids, t.embed.w, run.xs[0], R, c.C, c.embedScale);
  for (int l = 0; l < L; ++l) {
    ARENA.reset();
    blockForward(c, t.blocks[l], run.xs[l], run.xs[l + 1], R, seq, pos);
  }
  ARENA.reset();
  run.finalSave = lnForward(t.final, run.xs[L], run.out, R);   // (its stats live in the arena until the next reset)
}
// dout [R, C] (the final LayerNorm's output's gradient) -> every parameter's
inline void towerBackward(Tower& t, Run& run, const real* dout, const int* ids, const int* seq, const int* pos,
                          int upToLayers = -1) {
  const Config& c = t.cfg;
  int L = upToLayers < 0 ? c.layers : upToLayers;
  size_t R = run.R;
  real* dx; GK(cudaMallocAsync(&dx, R * c.C * sizeof(real), STREAM));
  {
    // the final norm: its stats were in the arena, so recompute them
    ARENA.reset();
    real* y = ARENA.get(R * c.C);
    LnSave fs = lnForward(t.final, run.xs[L], y, R);
    lnBackward(t.final, fs, dout, dx, R, false);
  }
  real* scratchOut; GK(cudaMallocAsync(&scratchOut, R * c.C * sizeof(real), STREAM));
  for (int l = L; l-- > 0;) {
    ARENA.reset();
    BlockSave s = blockForward(c, t.blocks[l], run.xs[l], scratchOut, R, seq, pos);
    blockBackward(c, t.blocks[l], s, dx, dx, R, seq, pos);
  }
  ARENA.reset();
  embedBwdK<<<blocks(R * c.C), 256, 0, STREAM>>>(ids, dx, t.embed.g, R, c.C, c.embedScale);
  GK(cudaFreeAsync(dx, STREAM)); GK(cudaFreeAsync(scratchOut, STREAM));
}

// the sequence head and a masked-LM cross entropy over the rows whose target >= 0: returns the mean loss, and
// leaves the gradient of the tower's output in dout
inline double headLoss(Tower& t, const real* x, const int* target, int count, real* dout, size_t R, bool grads = true) {
  Scratch scratch;
  const Config& c = t.cfg;
  real *h = ARENA.get(R * c.C), *g = ARENA.get(R * c.C), *gn = ARENA.get(R * c.C), *z = ARENA.get(R * c.vocab),
       *dz = ARENA.get(R * c.vocab);
  t.head.dense.forward(x, h, R);
  geluK<<<blocks(R * c.C), 256, 0, STREAM>>>(g, h, R * c.C);
  LnSave ls = lnForward(t.head.ln, g, gn, R);
  t.head.out.forward(gn, z, R);
  static double* acc = nullptr;
  static int* cnt = nullptr;
  if (!acc) { GK(cudaMalloc(&acc, 8)); GK(cudaMalloc(&cnt, 4)); }
  GK(cudaMemsetAsync(acc, 0, 8, STREAM));
  GK(cudaMemcpyAsync(cnt, &count, 4, cudaMemcpyHostToDevice, STREAM));
  crossEntropyK<<<blocks(R, 8), 256, 0, STREAM>>>(z, target, dz, (int)R, c.vocab, acc, cnt);
  if (!grads) {
    double loss; GK(cudaMemcpyAsync(&loss, acc, 8, cudaMemcpyDeviceToHost, STREAM));
    GK(cudaStreamSynchronize(STREAM));
    return loss;
  }
  real* dgn = ARENA.get(R * c.C);
  t.head.out.backward(gn, dz, dgn, R);
  real* dg = ARENA.get(R * c.C);
  lnBackward(t.head.ln, ls, dgn, dg, R, false);
  geluBwdK<<<blocks(R * c.C), 256, 0, STREAM>>>(dg, h, R * c.C);
  t.head.dense.backward(x, dg, dout, R);
  double loss; GK(cudaMemcpyAsync(&loss, acc, 8, cudaMemcpyDeviceToHost, STREAM));
  GK(cudaStreamSynchronize(STREAM));
  return loss;
}

}  // namespace plm
}  // namespace grad
