// AlphaFold 2's Evoformer block for training: forward with saves, backward to the inputs and to every parameter.
// af3-any-model's EvoformerIteration (alphafold3/af2/model/modules.py) - the multimer graph every AF2 checkpoint
// runs on - with the main stack's column attention or the extra-MSA stack's global column attention, and the outer
// product mean before the MSA track (multimer) or after it (a converted monomer). Fused triangle multiplication
// projections, as both regimes use.
//
// Held to cuda/grad/oracle_af2_block.py (jax.vjp in float64) by cuda/grad/test_af2_block.cu.
#pragma once
#include <functional>
#include <map>
#include "pair.cuh"

namespace grad {
namespace af2 {

// ---------------------------------------------------------------- global column attention (the extra stack)
// per batch b (a residue column) over S rows: qa = masked mean of x; q = qa Wq D^-1/2 [H, D]; k = x Wk, v = x Wv
// [S, D] (one head's worth, shared); w = softmax_s(q k^T + mask); wa = w v [H, D]; out = (wa * sigmoid(x Wg + bg))
// Wo + bo.
struct GlobalAttention {
  Param wq, wk, wv, wg, bg, wo, bo;
  int C = 64, H = 8, D = 8;
};
struct GaSave { const real* x; real *k, *v, *g, *w, *wa, *q; int B, S; };
// one block a column b; S rows, H heads of D (D <= 32, H D <= 256)
__global__ void gaFwdK(const real* __restrict__ x, const real* __restrict__ mask, const real* __restrict__ wq,
                       const real* __restrict__ k, const real* __restrict__ v, real* __restrict__ q,
                       real* __restrict__ w, real* __restrict__ wa, int B, int S, int C, int H, int D) {
  int b = blockIdx.x;
  extern __shared__ real sh[];
  real* qa = sh;               // [C]
  real* qs = sh + C;           // [H D]
  const real* xb = x + (size_t)b * S * C;
  const real* mb = mask + (size_t)b * S;
  real msum = 0;
  for (int s = 0; s < S; ++s) msum += mb[s];
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    real a = 0;
    for (int s = 0; s < S; ++s) a += mb[s] * xb[(size_t)s * C + c];
    qa[c] = a / (msum + 1e-10f);
  }
  __syncthreads();
  real scale = rsqrt((real)D);
  for (int t = threadIdx.x; t < H * D; t += blockDim.x) {
    real a = 0;
    for (int c = 0; c < C; ++c) a += qa[c] * wq[(size_t)c * H * D + t];
    qs[t] = a * scale;
    q[(size_t)b * H * D + t] = a * scale;
  }
  __syncthreads();
  // logits and softmax: one warp a head
  int warp = threadIdx.x / 32, lane = threadIdx.x & 31, warps = blockDim.x / 32;
  for (int h = warp; h < H; h += warps) {
    real* wr = w + ((size_t)b * H + h) * S;
    real mx = -INFINITY;
    for (int s = lane; s < S; s += 32) {
      real a = 0;
      for (int d = 0; d < D; ++d) a += qs[h * D + d] * k[((size_t)b * S + s) * D + d];
      a += 1e9f * (mb[s] - 1.f);
      wr[s] = a; mx = fmax(mx, a);
    }
    for (int o = 16; o; o >>= 1) mx = fmax(mx, __shfl_xor_sync(~0u, mx, o));
    real sum = 0;
    for (int s = lane; s < S; s += 32) { real e = exp(wr[s] - mx); wr[s] = e; sum += e; }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    for (int s = lane; s < S; s += 32) wr[s] /= sum;
    __syncwarp();
    for (int d = 0; d < D; ++d) {
      real a = 0;
      for (int s = lane; s < S; s += 32) a += wr[s] * v[((size_t)b * S + s) * D + d];
      for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(~0u, a, o);
      if (lane == 0) wa[((size_t)b * H + h) * D + d] = a;
    }
  }
}
// y[b, s, (h d)] = wa[b, (h d)] * g[b, s, (h d)]
__global__ void gaMixK(const real* wa, const real* g, real* y, int B, int S, int HD) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)B * S * HD) return;
  int hd = t % HD; size_t bs = t / HD; int b = (int)(bs / S);
  y[t] = wa[(size_t)b * HD + hd] * g[t];
}
// dwa[b, hd] = sum_s dy[b, s, hd] g[b, s, hd] ; dgp = dy wa g (1 - g)
__global__ void gaMixBwdK(const real* dy, const real* wa, const real* g, real* dwa, real* dgp, int B, int S,
                          int HD) {
  int b = blockIdx.x;
  for (int hd = threadIdx.x; hd < HD; hd += blockDim.x) {
    real a = 0, w = wa[(size_t)b * HD + hd];
    for (int s = 0; s < S; ++s) {
      size_t t = ((size_t)b * S + s) * HD + hd;
      real gv = g[t];
      a += dy[t] * gv;
      dgp[t] = dy[t] * w * gv * (1.f - gv);
    }
    dwa[(size_t)b * HD + hd] = a;
  }
}
// the softmax/attention backward for one column: dwa [H, D] -> dq [H D] (scaled q), dk, dv [S, D]
__global__ void gaBwdK(const real* __restrict__ mask, const real* __restrict__ q, const real* __restrict__ k,
                       const real* __restrict__ v, const real* __restrict__ w, const real* __restrict__ dwa,
                       real* __restrict__ dq, real* __restrict__ dk, real* __restrict__ dv, int B, int S, int H,
                       int D, real* __restrict__ dsAll) {
  int b = blockIdx.x;
  // ds [H][S] a column, in global memory: in shared it was H S reals, past a launch's 48 KB at AF2's 5,120 extra rows
  // (float32) and at 1,000 (float64) - the launch then failed and the gradient was whatever the buffers held
  real* ds = dsAll + (size_t)b * H * S;
  for (int t = threadIdx.x; t < H * S; t += blockDim.x) {
    int h = t / S, s = t % S;
    real dp = 0;
    for (int d = 0; d < D; ++d) dp += dwa[((size_t)b * H + h) * D + d] * v[((size_t)b * S + s) * D + d];
    ds[t] = dp;                       // dP
  }
  __syncthreads();
  int warp = threadIdx.x / 32, lane = threadIdx.x & 31, warps = blockDim.x / 32;
  for (int h = warp; h < H; h += warps) {
    const real* wr = w + ((size_t)b * H + h) * S;
    real a = 0;
    for (int s = lane; s < S; s += 32) a += ds[h * S + s] * wr[s];
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(~0u, a, o);
    for (int s = lane; s < S; s += 32) ds[h * S + s] = wr[s] * (ds[h * S + s] - a);
  }
  __syncthreads();
  for (int t = threadIdx.x; t < H * D; t += blockDim.x) {
    int h = t / D, d = t % D;
    real a = 0;
    for (int s = 0; s < S; ++s) a += ds[h * S + s] * k[((size_t)b * S + s) * D + d];
    dq[(size_t)b * H * D + t] = a;
  }
  for (int t = threadIdx.x; t < S * D; t += blockDim.x) {
    int s = t / D, d = t % D;
    real ak = 0, av = 0;
    for (int h = 0; h < H; ++h) {
      ak += ds[h * S + s] * q[((size_t)b * H + h) * D + d];
      av += w[((size_t)b * H + h) * S + s] * dwa[((size_t)b * H + h) * D + d];
    }
    dk[((size_t)b * S + s) * D + d] = ak;
    dv[((size_t)b * S + s) * D + d] = av;
  }
}
// dx[b, s, c] += mask[b, s] / (msum + eps) * dqa[b, c]
// msum[b] = sum_s mask[b, s] (a warp a row)
__global__ void maskSumK(const real* mask, real* msum, int B, int S) {
  int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (b >= B) return;
  real m = 0;
  for (int s = lane; s < S; s += 32) m += mask[(size_t)b * S + s];
  for (int o = 16; o; o >>= 1) m += __shfl_xor_sync(~0u, m, o);
  if (!lane) msum[b] = m;
}
// (the mask sum once a column: summed per element it was an S-long loop for every one of B S C elements, 4.4 ms
// of an extra-MSA block's 34)
__global__ void gaMeanBwdK(const real* mask, const real* msum, const real* dqa, real* dx, int B, int S, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)B * S * C) return;
  int c = t % C; size_t bs = t / C; int b = (int)(bs / S);
  dx[t] += mask[bs] / (msum[b] + 1e-10f) * dqa[(size_t)b * C + c];
}
__global__ void gaMeanK(const real* x, const real* mask, real* qa, int B, int S, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)B * C) return;
  int c = t % C, b = (int)(t / C);
  real a = 0, m = 0;
  for (int s = 0; s < S; ++s) { real mv = mask[(size_t)b * S + s]; a += mv * x[((size_t)b * S + s) * C + c]; m += mv; }
  qa[t] = a / (m + 1e-10f);
}

inline GaSave gaForward(const GlobalAttention& a, const real* x, const real* mask, real* out, int B, int S) {
  size_t rows = (size_t)B * S;
  const int HD = a.H * a.D;
  GaSave s{x, ARENA.get(rows * a.D), ARENA.get(rows * a.D), ARENA.get(rows * HD), ARENA.get((size_t)B * a.H * S),
           ARENA.get((size_t)B * HD), ARENA.get((size_t)B * HD), B, S};
  mm(x, a.wk.w, s.k, (int)rows, a.C, a.D);
  mm(x, a.wv.w, s.v, (int)rows, a.C, a.D);
  mm(x, a.wg.w, s.g, (int)rows, a.C, HD);
  biasK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.g, a.bg.w, rows, HD);
  sigmoidK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.g, rows * HD);
  gaFwdK<<<B, 256, (a.C + HD) * sizeof(real), STREAM>>>(x, mask, a.wq.w, s.k, s.v, s.q, s.w, s.wa, B, S, a.C, a.H, a.D);
  Scratch scratch;
  real* y = ARENA.get(rows * HD);
  gaMixK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.wa, s.g, y, B, S, HD);
  mm(y, a.wo.w, out, (int)rows, HD, a.C);
  biasK<<<blocks(rows * a.C), 256, 0, STREAM>>>(out, a.bo.w, rows, a.C);
  return s;
}
inline void gaBackward(GlobalAttention& a, const GaSave& s, const real* mask, const real* dout, real* dx,
                       bool accumulate) {
  Scratch scratch;
  const int B = s.B, S = s.S, HD = a.H * a.D;
  size_t rows = (size_t)B * S;
  colSum(a.bo.g, dout, rows, a.C);
  real* y = ARENA.get(rows * HD);
  gaMixK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.wa, s.g, y, B, S, HD);
  mmAT(y, dout, a.wo.g, (int)rows, HD, a.C);
  real* dy = y;
  mmBT(dout, a.wo.w, dy, (int)rows, HD, a.C);
  real *dwa = ARENA.get((size_t)B * HD), *dgp = ARENA.get(rows * HD);
  gaMixBwdK<<<B, 128, 0, STREAM>>>(dy, s.wa, s.g, dwa, dgp, B, S, HD);
  real *dq = ARENA.get((size_t)B * HD), *dk = ARENA.get(rows * a.D), *dv = ARENA.get(rows * a.D);
  real* dsAll = ARENA.get((size_t)B * a.H * S);
  gaBwdK<<<B, 256, 0, STREAM>>>(mask, s.q, s.k, s.v, s.w, dwa, dq, dk, dv, B, S, a.H, a.D, dsAll);
  GK(cudaGetLastError());
  // q = qa Wq * scale: dWq += qa^T dq scale, dqa = dq Wq^T scale
  real* qa = ARENA.get((size_t)B * a.C);
  gaMeanK<<<blocks((size_t)B * a.C), 256, 0, STREAM>>>(s.x, mask, qa, B, S, a.C);
  const real scale = 1.f / sqrt((real)a.D);
  mmAT(qa, dq, a.wq.g, B, a.C, HD, 1.f, scale);
  real* dqa = ARENA.get((size_t)B * a.C);
  mmBT(dq, a.wq.w, dqa, B, a.C, HD, 0.f, scale);
  mmAT(s.x, dk, a.wk.g, (int)rows, a.C, a.D);
  mmAT(s.x, dv, a.wv.g, (int)rows, a.C, a.D);
  mmAT(s.x, dgp, a.wg.g, (int)rows, a.C, HD);
  colSum(a.bg.g, dgp, rows, HD);
  mmBT(dk, a.wk.w, dx, (int)rows, a.C, a.D, accumulate ? 1.f : 0.f);
  mmBT(dv, a.wv.w, dx, (int)rows, a.C, a.D, 1.f);
  mmBT(dgp, a.wg.w, dx, (int)rows, a.C, HD, 1.f);
  real* msum = ARENA.get(B);
  maskSumK<<<blocks((size_t)B * 32), 256, 0, STREAM>>>(mask, msum, B, S);
  gaMeanBwdK<<<blocks(rows * a.C), 256, 0, STREAM>>>(mask, msum, dqa, dx, B, S, a.C);
}

// ---------------------------------------------------------------- attention modules with their norms
struct RowAttn {           // MSA row attention with pair bias
  LayerNorm qn, pn;
  Param feat2d;            // [Cz, H]
  Attention at;
};
struct TriAttn {           // triangle attention (starting node: per_row; ending node: per_column)
  LayerNorm qn;
  Param feat2d;            // [Cz, H]
  Attention at;
  bool perColumn = false;
};
struct ColAttn {           // the main stack's MSA column attention
  LayerNorm qn;
  Attention at;
};
struct ColGlobal {
  LayerNorm qn;
  GlobalAttention at;
};

// nb[h, q, k] = sum_c pn[q, k, c] w[c, h] (+ 1e9 (pairMask - 1) for the row attention): pn [L L, Cz] -> [H, L L]
inline void pairBias(const real* pn, const real* w, real* nb, int L, int Cz, int H) {
  Scratch scratch;
  real* t = ARENA.get((size_t)L * L * H);
  mm(pn, w, t, L * L, Cz, H);
  transpose(t, nb, L * L, H);
}
__global__ void addMaskBiasK(real* nb, const real* pairMask, int H, size_t LL) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)H * LL) nb[t] += 1e9f * (pairMask[t % LL] - 1.f);
}

// ---------------------------------------------------------------- the block
struct Block {
  bool extra = false, opmFirst = true;
  // training's shared dropout (AF2's dropout_wrapper): off unless set; the masks come from a counter hash of
  // (dropSeed, module, position), so the remat replay and the backward regenerate them and none is stored
  bool dropout = false;
  unsigned long long dropSeed = 0;
  int Cm = 256, Cz = 128;
  Opm opm;
  RowAttn row;
  ColAttn col;
  ColGlobal colg;
  Transition msaTrans, pairTrans;
  TriMul triOut, triIn;
  TriAttn triStart, triEnd;
  Params params;
};

// every parameter, by haiku's name under the block
inline void forEachParam(Block& b, const std::function<void(const std::string&, Param&, size_t)>& f) {
  auto ln = [&](const std::string& n, LayerNorm& l, int C) { l.C = C; f(n + "/scale", l.g, C); f(n + "/offset", l.b, C); };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; f(n + "/weights", l.w, (size_t)K * N); f(n + "/bias", l.b, N); };
  auto att = [&](const std::string& n, Attention& a, int C, int H, int D, int O) {
    a.C = C; a.H = H; a.D = D; a.O = O; a.gating = true;
    size_t HD = (size_t)H * D;
    f(n + "/query_w", a.wq, C * HD); f(n + "/key_w", a.wk, C * HD); f(n + "/value_w", a.wv, C * HD);
    f(n + "/gating_w", a.wg, C * HD); f(n + "/gating_b", a.bg, HD);
    f(n + "/output_w", a.wo, HD * O); f(n + "/output_b", a.bo, O);
  };
  const int Cm = b.Cm, Cz = b.Cz;
  ln("outer_product_mean/layer_norm_input", b.opm.ln, Cm);
  b.opm.C = Cm;
  lin("outer_product_mean/left_projection", b.opm.left, Cm, 32);
  lin("outer_product_mean/right_projection", b.opm.right, Cm, 32);
  f("outer_product_mean/output_w", b.opm.wo, (size_t)32 * 32 * Cz); f("outer_product_mean/output_b", b.opm.bo, Cz);
  ln("msa_row_attention_with_pair_bias/query_norm", b.row.qn, Cm);
  ln("msa_row_attention_with_pair_bias/feat_2d_norm", b.row.pn, Cz);
  f("msa_row_attention_with_pair_bias/feat_2d_weights", b.row.feat2d, (size_t)Cz * 8);
  att("msa_row_attention_with_pair_bias/attention", b.row.at, Cm, 8, Cm / 8, Cm);
  if (!b.extra) {
    ln("msa_column_attention/query_norm", b.col.qn, Cm);
    att("msa_column_attention/attention", b.col.at, Cm, 8, Cm / 8, Cm);
  } else {
    auto& g = b.colg.at;
    ln("msa_column_global_attention/query_norm", b.colg.qn, Cm);
    g.C = Cm; g.H = 8; g.D = Cm / 8;
    const std::string n = "msa_column_global_attention/attention";
    size_t HD = (size_t)g.H * g.D;
    f(n + "/query_w", g.wq, Cm * HD); f(n + "/key_w", g.wk, (size_t)Cm * g.D); f(n + "/value_w", g.wv, (size_t)Cm * g.D);
    f(n + "/gating_w", g.wg, Cm * HD); f(n + "/gating_b", g.bg, HD);
    f(n + "/output_w", g.wo, HD * Cm); f(n + "/output_b", g.bo, Cm);
  }
  ln("msa_transition/input_layer_norm", b.msaTrans.ln, Cm);
  lin("msa_transition/transition1", b.msaTrans.l1, Cm, 4 * Cm);
  lin("msa_transition/transition2", b.msaTrans.l2, 4 * Cm, Cm);
  for (auto* t : {&b.triOut, &b.triIn}) {
    std::string n = t == &b.triOut ? "triangle_multiplication_outgoing" : "triangle_multiplication_incoming";
    t->C = Cz; t->I = Cz; t->outgoing = t == &b.triOut;
    ln(n + "/left_norm_input", t->lnIn, Cz);
    ln(n + "/center_norm", t->lnCenter, Cz);
    lin(n + "/projection", t->proj, Cz, 2 * Cz);
    lin(n + "/gate", t->gate, Cz, 2 * Cz);
    lin(n + "/output_projection", t->out, Cz, Cz);
    lin(n + "/gating_linear", t->gating, Cz, Cz);
  }
  for (auto* t : {&b.triStart, &b.triEnd}) {
    std::string n = t == &b.triStart ? "triangle_attention_starting_node" : "triangle_attention_ending_node";
    t->perColumn = t == &b.triEnd;
    ln(n + "/query_norm", t->qn, Cz);
    f(n + "/feat_2d_weights", t->feat2d, (size_t)Cz * 4);
    att(n + "/attention", t->at, Cz, 4, Cz / 4, Cz);
  }
  ln("pair_transition/input_layer_norm", b.pairTrans.ln, Cz);
  lin("pair_transition/transition1", b.pairTrans.l1, Cz, 4 * Cz);
  lin("pair_transition/transition2", b.pairTrans.l2, 4 * Cz, Cz);
}

inline void allocBlock(Block& b) {
  forEachParam(b, [&](const std::string& n, Param& p, size_t len) { allocParam(p, n, len); b.params.add(p); });
}

// AF2's apply_dropout on an update [A, B, C] in place: keep / (1 - rate), one draw per position with the broadcast
// axis (per_row: A, per_column: B) dropped. trans: the update is held [B, A, C] (an ending-node or column
// update before it is swapped back), its draws still those of [A, B, C]. The forward scales the update, the backward
// the update's gradient, with the same draws.
__host__ __device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
template <class T>
__global__ void dropoutK(T* x, int A, int B, int C, bool perRow, bool trans, float rate, unsigned long long seed) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * B * C) return;
  const int c = (int)(t % C);
  const size_t r = t / C;
  int a, b;
  if (trans) { a = (int)(r % A); b = (int)(r / A); } else { a = (int)(r / B); b = (int)(r % B); }
  const unsigned long long key = perRow ? (unsigned long long)b * C + c : ((unsigned long long)a * C + c) | (1ull << 62);
  const float u = (float)(mix64(seed ^ mix64(key)) >> 40) * (1.f / 16777216.f);
  if (u < rate) str(x + t, 0); else str(x + t, ldr(x + t) / (1 - (real)rate));   // (real arithmetic: the f64 gate's)
}
// rates by module (AF2's config): row attention 0.15 per row, the four pair modules 0.25 (the ending node per
// column), every other 0
enum DropMod { DROP_ROW = 1, DROP_TRI_OUT, DROP_TRI_IN, DROP_TRI_START, DROP_TRI_END };
template <class T>
inline void dropout(const Block& b, DropMod m, T* x, int A, int Bd, int C, bool trans) {
  if (!b.dropout) return;
  const float rate = m == DROP_ROW ? 0.15f : 0.25f;
  dropoutK<<<blocks((size_t)A * Bd * C), 256, 0, STREAM>>>(x, A, Bd, C, m != DROP_TRI_END, trans, rate,
                                                          mix64(b.dropSeed * 8 + m));
}

// What the backward needs from the forward, per module.
struct BlockSave {
  int S, L;
  const real *msa0, *pair0;      // the block's inputs (owned by the caller)
  real *msa1, *msa2, *msa3;      // the MSA track after row attention, column attention (transposed back), transition
  real *pair1, *pair2, *pair3, *pair4, *pair5;   // after OPM (or the input), triOut, triIn, triStart, triEnd
  OpmSave opm;
  LnSave rowQ, rowP; real *rowXn, *rowPn, *rowNb; AttnSave rowA;
  LnSave colQ; real *colX, *colXn, *colMask; AttnSave colA; GaSave colG;
  TransSave msaT, pairT;
  TriMulSave tOut, tIn;
  LnSave stQ, enQ; real *stXn, *enX, *enXn, *stNb, *enNb, *pairMaskT; AttnSave stA, enA;
  real *pairOpm;                 // the pair after OPM when opm runs last
};

__global__ void copyK(real* d, const real* s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) d[t] = s[t];
}
inline real* clone(const real* x, size_t n) {
  real* y = ARENA.get(n);
  GK(cudaMemcpyAsync(y, x, n * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  return y;
}

// msa [S, L, Cm], pair [L, L, Cz] -> msaOut, pairOut (may alias nothing). msaMask [S, L], pairMask [L, L].
inline BlockSave blockForward(Block& b, const real* msa, const real* pair, const real* msaMask, const real* pairMask,
                              real* msaOut, real* pairOut, int S, int L) {
  BlockSave s{};
  s.S = S; s.L = L; s.msa0 = msa; s.pair0 = pair;
  const int Cm = b.Cm, Cz = b.Cz;
  size_t mr = (size_t)S * L, pr = (size_t)L * L;
  real* pcur = clone(pair, pr * Cz);
  if (b.opmFirst) s.opm = opmForward(b.opm, msa, msaMask, pcur, S, L);
  s.pair1 = pcur;
  // MSA row attention with pair bias
  s.rowXn = ARENA.get(mr * Cm);
  s.rowQ = lnForward(b.row.qn, msa, s.rowXn, mr);
  s.rowPn = ARENA.get(pr * Cz);
  s.rowP = lnForward(b.row.pn, s.pair1, s.rowPn, pr);
  s.rowNb = ARENA.get(pr * 8);
  pairBias(s.rowPn, b.row.feat2d.w, s.rowNb, L, Cz, 8);
  addMaskBiasK<<<blocks(pr * 8), 256, 0, STREAM>>>(s.rowNb, pairMask, 8, pr);
  real* upd = ARENA.get(mr * Cm);
  s.rowA = attnForward(b.row.at, s.rowXn, msaMask, s.rowNb, upd, S, L);
  dropout(b, DROP_ROW, upd, S, L, Cm, false);
  s.msa1 = clone(msa, mr * Cm);
  add(s.msa1, upd, mr * Cm);
  // MSA column attention, on the transposed MSA [L, S, C]
  s.colX = ARENA.get(mr * Cm);
  swap01(s.msa1, s.colX, S, L, Cm);
  s.colMask = ARENA.get(mr);
  transpose(msaMask, s.colMask, S, L);
  s.colXn = ARENA.get(mr * Cm);
  real* updT = ARENA.get(mr * Cm);
  if (!b.extra) {
    s.colQ = lnForward(b.col.qn, s.colX, s.colXn, mr);
    s.colA = attnForward(b.col.at, s.colXn, s.colMask, nullptr, updT, L, S);
  } else {
    s.colQ = lnForward(b.colg.qn, s.colX, s.colXn, mr);
    s.colG = gaForward(b.colg.at, s.colXn, s.colMask, updT, L, S);
  }
  s.msa2 = clone(s.msa1, mr * Cm);
  swap01(updT, s.msa2, L, S, Cm, true);
  // MSA transition
  s.msaT = transForward(b.msaTrans, s.msa2, upd, mr);
  s.msa3 = clone(s.msa2, mr * Cm);
  add(s.msa3, upd, mr * Cm);
  real* pc = s.pair1;
  if (!b.opmFirst) {
    pc = clone(s.pair1, pr * Cz);
    s.opm = opmForward(b.opm, s.msa3, msaMask, pc, S, L);
    s.pairOpm = pc;
  }
  // triangle multiplications
  real* pu = ARENA.get(pr * Cz);
  s.tOut = triMulForward(b.triOut, pc, pairMask, pu, L);
  dropout(b, DROP_TRI_OUT, pu, L, L, Cz, false);
  s.pair2 = clone(pc, pr * Cz); add(s.pair2, pu, pr * Cz);
  s.tIn = triMulForward(b.triIn, s.pair2, pairMask, pu, L);
  dropout(b, DROP_TRI_IN, pu, L, L, Cz, false);
  s.pair3 = clone(s.pair2, pr * Cz); add(s.pair3, pu, pr * Cz);
  // triangle attention, starting node (per row)
  s.stXn = ARENA.get(pr * Cz);
  s.stQ = lnForward(b.triStart.qn, s.pair3, s.stXn, pr);
  s.stNb = ARENA.get(pr * 4);
  pairBias(s.stXn, b.triStart.feat2d.w, s.stNb, L, Cz, 4);
  s.stA = attnForward(b.triStart.at, s.stXn, pairMask, s.stNb, pu, L, L);
  dropout(b, DROP_TRI_START, pu, L, L, Cz, false);
  s.pair4 = clone(s.pair3, pr * Cz); add(s.pair4, pu, pr * Cz);
  // ending node: on the transposed pair, with the transposed mask
  s.enX = ARENA.get(pr * Cz);
  swap01(s.pair4, s.enX, L, L, Cz);
  s.pairMaskT = ARENA.get(pr);
  transpose(pairMask, s.pairMaskT, L, L);
  s.enXn = ARENA.get(pr * Cz);
  s.enQ = lnForward(b.triEnd.qn, s.enX, s.enXn, pr);
  s.enNb = ARENA.get(pr * 4);
  pairBias(s.enXn, b.triEnd.feat2d.w, s.enNb, L, Cz, 4);
  s.enA = attnForward(b.triEnd.at, s.enXn, s.pairMaskT, s.enNb, pu, L, L);
  s.pair5 = clone(s.pair4, pr * Cz);
  dropout(b, DROP_TRI_END, pu, L, L, Cz, true);
  swap01(pu, s.pair5, L, L, Cz, true);
  // pair transition
  s.pairT = transForward(b.pairTrans, s.pair5, pu, pr);
  GK(cudaMemcpyAsync(pairOut, s.pair5, pr * Cz * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  add(pairOut, pu, pr * Cz);
  GK(cudaMemcpyAsync(msaOut, s.msa3, mr * Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  return s;
}

// the gradient of a pair bias nb = transpose(pn W) to pn and W: dnb [H, L L]
inline void pairBiasBackward(const real* pn, Param& w, const real* dnb, real* dpn, int L, int Cz, int H,
                             bool accumulate) {
  Scratch scratch;
  real* t = ARENA.get((size_t)L * L * H);
  transpose(dnb, t, H, L * L);                // [L L, H]
  mmAT(pn, t, w.g, L * L, Cz, H);
  mmBT(t, w.w, dpn, L * L, Cz, H, accumulate ? 1.f : 0.f);
}

// the pair gradient's buffers either way (blockBackwardT)
#ifndef GRAD_F64
inline bf* bfGetT(size_t n) { return bfGet(n); }
inline void toP(const real* x, bf* y, size_t n) { toBf(x, y, n); }
inline void fromP(const bf* x, real* y, size_t n) { bfToFK<<<blocks(n), 256, 0, STREAM>>>(x, y, n); }
#else
inline real* bfGetT(size_t n) { return ARENA.get(n); }
#endif
inline void toP(const real* x, real* y, size_t n) { GK(cudaMemcpyAsync(y, x, n * sizeof(real), cudaMemcpyDeviceToDevice, STREAM)); }
inline void fromP(const real* x, real* y, size_t n) { GK(cudaMemcpyAsync(y, x, n * sizeof(real), cudaMemcpyDeviceToDevice, STREAM)); }
// PD: the running gradients' type - real, or bf16 in BF16 mode (the pair's and the MSA's: bindcraft2 keeps its MSA
// cotangent float; test-bf16 holds this one to autocast's level)
template <class PD>
inline void blockBackwardT(Block& b, const BlockSave& s, const real* msaMask, const real* pairMask,
                           const real* dmsaOut, const real* dpairOut, real* dmsa, real* dpair) {
  const int S = s.S, L = s.L, Cm = b.Cm, Cz = b.Cz;
  size_t mr = (size_t)S * L, pr = (size_t)L * L;
  constexpr bool BP = sizeof(PD) == 2;
  auto getP = [&](size_t n) { return reinterpret_cast<PD*>(BP ? (real*)bfGetT(n) : ARENA.get(n)); };
  auto cloneP = [&](const PD* x, size_t n) { PD* y = getP(n); GK(cudaMemcpyAsync(y, x, n * sizeof(PD), cudaMemcpyDeviceToDevice, STREAM)); return y; };
  PD* dp = getP(pr * Cz);                     // the running gradient of the pair
  toP(dpairOut, dp, pr * Cz);
  PD* dm = getP(mr * Cm);                     // and of the MSA (bf16 as well in BF16 mode)
  toP(dmsaOut, dm, mr * Cm);
  // pair transition
  { Range r("bw.pairTrans"); transBackward(b.pairTrans, s.pairT, (const PD*)dp, dp, true); }
  // ending node: pair5 = pair4 + swap(attn(enXn))
  {
    Range r("bw.triEnd");
    Scratch scratch;
    PD* dupdT = getP(pr * Cz);
    swap01(dp, dupdT, L, L, Cz);
    dropout(b, DROP_TRI_END, dupdT, L, L, Cz, true);
    real* dnb = ARENA.zeros(pr * 4);
    real* dxn = ARENA.get(pr * Cz);
    attnBackward(b.triEnd.at, s.enA, s.pairMaskT, s.enNb, dupdT, dxn, dnb, false);
    pairBiasBackward(s.enXn, b.triEnd.feat2d, dnb, dxn, L, Cz, 4, true);
    PD* dx = getP(pr * Cz);
    lnBackward(b.triEnd.qn, s.enQ, dxn, dx, pr, false);
    swap01(dx, dp, L, L, Cz, true);
  }
  // starting node
  {
    Range r("bw.triStart");
    Scratch scratch;
    real* dnb = ARENA.zeros(pr * 4);
    real* dxn = ARENA.get(pr * Cz);
    const PD* dst = dp;
    if (b.dropout) { PD* t = cloneP(dp, pr * Cz); dropout(b, DROP_TRI_START, t, L, L, Cz, false); dst = t; }
    attnBackward(b.triStart.at, s.stA, pairMask, s.stNb, dst, dxn, dnb, false);
    pairBiasBackward(s.stXn, b.triStart.feat2d, dnb, dxn, L, Cz, 4, true);
    lnBackward(b.triStart.qn, s.stQ, dxn, dp, pr, true);
  }
  // triangle multiplications: pair3 = pair2 + tIn(pair2); pair2 = pc + tOut(pc)
  {
    Scratch scratch;
    // (BF16: in place - triMulBackwardBf reads dout whole before it adds into dpair)
    // (with dropout the module's gradient is the masked copy, so nothing is in place)
    const bool inPlace = BF16 && !b.dropout;
    PD* dpu = inPlace ? dp : cloneP(dp, pr * Cz);
    dropout(b, DROP_TRI_IN, dpu, L, L, Cz, false);
    { Range r("bw.triIn"); triMulBackward(b.triIn, s.tIn, pairMask, dpu, dp); }
    Range r("bw.triOut");
    if (!inPlace) dpu = cloneP(dp, pr * Cz);
    dropout(b, DROP_TRI_OUT, dpu, L, L, Cz, false);
    triMulBackward(b.triOut, s.tOut, pairMask, dpu, dp);
  }
  if (!b.opmFirst) opmBackward(b.opm, s.opm, msaMask, dp, dm);    // pc = pair1 + opm(msa3)
  // MSA transition: msa3 = msa2 + T(msa2)
  { Range r("bw.msaTrans"); transBackward(b.msaTrans, s.msaT, dm, dm, true); }
  // column attention: msa2 = msa1 + swap(col(swap(msa1)))
  {
    Scratch scratch;
    Range r("bw.col");
    PD* dupdT = getP(mr * Cm);
    swap01(dm, dupdT, S, L, Cm);
    real* dxn = ARENA.get(mr * Cm);
    PD* dx = getP(mr * Cm);
    if (!b.extra) {
      attnBackward(b.col.at, s.colA, s.colMask, nullptr, (const PD*)dupdT, dxn, nullptr, false);
      lnBackward(b.col.qn, s.colQ, dxn, dx, mr, false);
    } else {   // (the global column attention is float throughout)
      const real* du;
      if constexpr (BP) { real* t = ARENA.get(mr * Cm); fromP(dupdT, t, mr * Cm); du = t; } else du = dupdT;
      gaBackward(b.colg.at, s.colG, s.colMask, du, dxn, false);
      lnBackward(b.colg.qn, s.colQ, dxn, dx, mr, false);
    }
    swap01(dx, dm, L, S, Cm, true);
  }
  // row attention: msa1 = msa + row(LN(msa), bias(LN(pair1)))
  {
    Scratch scratch;
    Range r("bw.row");
    real* dnb = ARENA.zeros(pr * 8);
    real* dxn = ARENA.get(mr * Cm);
    const PD* drow = dm;
    if (b.dropout) { PD* t = cloneP(dm, mr * Cm); dropout(b, DROP_ROW, t, S, L, Cm, false); drow = t; }
    attnBackward(b.row.at, s.rowA, msaMask, s.rowNb, drow, dxn, dnb, false);
    lnBackward(b.row.qn, s.rowQ, dxn, dm, mr, true);
    real* dpn = ARENA.get(pr * Cz);
    pairBiasBackward(s.rowPn, b.row.feat2d, dnb, dpn, L, Cz, 8, false);
    lnBackward(b.row.pn, s.rowP, dpn, dp, pr, true);
  }
  if (b.opmFirst) { Range r("bw.opm"); opmBackward(b.opm, s.opm, msaMask, dp, dm); }    // pair1 = pair + opm(msa)
  fromP(dm, dmsa, mr * Cm);
  fromP(dp, dpair, pr * Cz);
}
// dmsaOut, dpairOut -> dmsa, dpair (written). Every parameter gradient accumulates.
inline void blockBackward(Block& b, const BlockSave& s, const real* msaMask, const real* pairMask,
                          const real* dmsaOut, const real* dpairOut, real* dmsa, real* dpair) {
#ifndef GRAD_F64
  if (BF16) { blockBackwardT<bf>(b, s, msaMask, pairMask, dmsaOut, dpairOut, dmsa, dpair); return; }
#endif
  blockBackwardT<real>(b, s, msaMask, pairMask, dmsaOut, dpairOut, dmsa, dpair);
}

}  // namespace af2
}  // namespace grad

namespace grad {
namespace af2 {

// ---------------------------------------------------------------- a stack of blocks, rematerialised
// The forward keeps every block's INPUT (msa and pair, fp32) and nothing else; the backward replays block i's
// forward with its saves straight before block i's backward, and resets the arena after. Held: n block inputs plus
// ONE block's activations, for the price of one extra forward - AF2's own use_remat.
struct Stack {
  std::vector<Block> blocks;
  std::vector<real*> msaIn, pairIn;     // [n + 1]: block i's input, and the stack's output last
  int S = 0, L = 0;
  size_t inputBytes() const { return (msaIn.size() * (size_t)S * L * blocks[0].Cm + pairIn.size() * (size_t)L * L * 128) * sizeof(real); }
  void ensure(int S_, int L_) {
    if (S_ == S && L_ == L && !msaIn.empty()) return;
    for (real* p : msaIn) GK(cudaFree(p));
    for (real* p : pairIn) GK(cudaFree(p));
    msaIn.assign(blocks.size() + 1, nullptr); pairIn.assign(blocks.size() + 1, nullptr);
    S = S_; L = L_;
    for (size_t i = 0; i <= blocks.size(); ++i) {
      GK(cudaMalloc(&msaIn[i], (size_t)S * L * blocks[0].Cm * sizeof(real)));
      GK(cudaMalloc(&pairIn[i], (size_t)L * L * 128 * sizeof(real)));
    }
  }
  // msa/pair -> msaIn.back()/pairIn.back()
  void forward(const real* msa, const real* pair, const real* msaMask, const real* pairMask, int S_, int L_) {
    ensure(S_, L_);
    GK(cudaMemcpyAsync(msaIn[0], msa, (size_t)S * L * blocks[0].Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(pairIn[0], pair, (size_t)L * L * 128 * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    for (size_t i = 0; i < blocks.size(); ++i) {
      Scratch sc;          // (a scope, not a reset: a caller's saves in the arena outlive the stack)
      blockForward(blocks[i], msaIn[i], pairIn[i], msaMask, pairMask, msaIn[i + 1], pairIn[i + 1], S, L);
    }
  }
  // gradients of the stack's output -> of its input (dmsa, dpair: in place)
  void backward(const real* msaMask, const real* pairMask, real* dmsa, real* dpair) {
    real *om, *op;
    GK(cudaMallocAsync(&om, (size_t)S * L * blocks[0].Cm * sizeof(real), STREAM));
    GK(cudaMallocAsync(&op, (size_t)L * L * 128 * sizeof(real), STREAM));
    for (size_t i = blocks.size(); i-- > 0;) {
      Scratch sc;
      BlockSave s = blockForward(blocks[i], msaIn[i], pairIn[i], msaMask, pairMask, om, op, S, L);
      blockBackward(blocks[i], s, msaMask, pairMask, dmsa, dpair, dmsa, dpair);
    }
    GK(cudaFreeAsync(om, STREAM)); GK(cudaFreeAsync(op, STREAM));
  }
};

}  // namespace af2
}  // namespace grad
