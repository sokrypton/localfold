// ESM-C's tower and ESMFold2's language-model shim (cuda/plm/esmc.cuh and cuda/ef2/src/shim.cuh are the reading):
//
//   tower block: h = LN(x) @ qkv; q, k = LN(q), LN(k) over the full width (no offset), rotated (head 64, base 10000);
//                x += attn(q, k, v; within a chain) @ attn_out / s;  x += swiglu(LN(x) @ fc1) @ fc2 / s
//   shim: single = (sum_k softmax(combine)_k LN(h_k) @ projection) @ downproject + bias over the 37 hidden states;
//         pair [a * b | a - b] -> 512 -> gelu -> 256 -> 256 -> LN
// The tower's four matrices a block and the shim's are float16; activations float32 between them.
#include "ef2.h"
#include <cmath>

// a block matrix as float16: its own copy, or - a RESIDENT tower (ESM-C 6B, its codes kept in the mapped shards:
// Model::loadBundle's `resident`) - expanded into scratch for this one GEMM
static const half* towerW(const std::string& name) {
  if (!M.isResident(name)) return M.h(name);
  half* w = scratch<half>("tower.w16", M.len(name));
  M.expand(name, w);
  return w;
}
static void towerLin(const half* X, const std::string& name, float* Y, size_t rows, int in, int out, float alpha = 1.f, float beta = 0.f) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = towerW(name); g.tw = F16; g.Y = Y; g.ty = F32; g.rows = rows; g.in = in; g.out = out;
  g.alpha = alpha; g.beta = beta; g.label = name.c_str();
  gemm(g);
}
// fc1 with its gate and up columns interleaved for gemmSwiglu: in place once for a decoded tower, into scratch for a
// resident one
static const half* towerFc1(const std::string& name, int C, int ffn) {
  if (!M.isResident(name)) return swigluPairsInPlace(name, C, ffn);
  const half* w = towerW(name);
  half* out = scratch<half>("tower.w16s", (size_t)C * 2 * ffn);
  run1d("ef2_interleave8", (size_t)C * 2 * ffn, Interleave8Args{w, w + ffn, out, (uint)C, (uint)ffn, (uint)(2 * ffn), 0});
  return out;
}

static void esmcBlock(const Esmc& e, float* x, const int* seq, int layer) {
  std::string B = "c/blocks/" + std::to_string(layer) + "/";
  size_t R = e.rows; int C = e.model, H = e.heads;
  half* xn = scratch<half>("esmc.xn", R * C);
  float* qkv = scratch<float>("esmc.qkv", R * 3 * C);
  layerNorm(x, xn, R, C, M.f(B + "attn_norm/scale"), M.f(B + "attn_norm/offset"));
  towerLin(xn, B + "qkv/weights", qkv, R, C, 3 * C);
  float* q = scratch<float>("esmc.q", R * C); float* k = scratch<float>("esmc.k", R * C);
  layerNorm(qkv, q, R, C, M.f(B + "q_norm/scale"), nullptr, 1e-5f, 3 * C, C);
  layerNorm(qkv + C, k, R, C, M.f(B + "k_norm/scale"), nullptr, 1e-5f, 3 * C, C);
  run1d("ef2_rope", R * H * 32, RopeArgs{q, (uint)R, (uint)H, (uint)C, 0});
  run1d("ef2_rope", R * H * 32, RopeArgs{k, (uint)R, (uint)H, (uint)C, 0});
  // scores [H, R, R] = q_h k_h^T, softmax within a chain, ctx_h = S_h v_h (v in the packed qkv, stride 3C)
  float* S = scratch<float>("esmc.scores", (size_t)H * R * R);
  float* ctx = scratch<float>("esmc.ctx", R * C);
  {
    Gemm g{}; g.X = q; g.ldx = C; g.sx = 64; g.W = k; g.transW = true; g.ldw = C; g.sw = 64;
    g.Y = S; g.ldy = (int)R; g.sy = (int64_t)R * R; g.rows = R; g.in = 64; g.out = (int)R; g.batch = H; g.label = "esmc.scores";
    gemm(g);
  }
  run("ef2_softmax_seq", grid1d((size_t)H * R, 1), 256, SoftmaxSeqArgs{S, seq, (uint)R, 0.125f});
  {
    Gemm g{}; g.X = S; g.ldx = (int)R; g.sx = (int64_t)R * R; g.W = qkv + 2 * C; g.ldw = 3 * C; g.sw = 64;
    g.Y = ctx; g.ldy = C; g.sy = 64; g.rows = R; g.in = (int)R; g.out = 64; g.batch = H; g.label = "esmc.context";
    gemm(g);
  }
  if (!M.isResident(B + "attn_out/weights")) lin(ctx, B + "attn_out/weights", x, R, C, C, 1.f, nullptr, 1.f / e.residualScale);
  else {
    Gemm g{}; g.X = ctx; g.tx = F32; g.W = towerW(B + "attn_out/weights"); g.tw = F16; g.half = true; g.Y = x; g.ty = F32;
    g.rows = R; g.in = C; g.out = C; g.beta = 1.f; g.alpha = 1.f / e.residualScale; g.label = "attn_out";
    gemm(g);
  }
  layerNorm(x, xn, R, C, M.f(B + "ffn_norm/scale"), M.f(B + "ffn_norm/offset"));
  half* g = scratch<half>("esmc.g", R * e.ffn);
  gemmSwiglu(xn, towerFc1(B + "fc1/weights", C, e.ffn), g, R, C, e.ffn);
  towerLin(g, B + "fc2/weights", x, R, e.ffn, C, 1.f / e.residualScale, 1.f);
}

void languageModel(const Esmc& e, const int* ids, const int* seq, const int* tokenToRow, int T, float* lmZ) {
  size_t R = e.rows; int C = e.model, P = e.pair;
  float* single = scratch<float>("shim.single", std::max<size_t>(R, 1) * P);
  if (R > 0) {
    // the mix weights, a softmax of a constant
    std::vector<float> mix(M.len("c/lm/combine"));
    { const float* cmb = M.hostF("c/lm/combine"); float mx = -INFINITY, s = 0;
      for (size_t i = 0; i < mix.size(); ++i) mx = std::max(mx, cmb[i]);
      for (size_t i = 0; i < mix.size(); ++i) { mix[i] = expf(cmb[i] - mx); s += mix[i]; }
      for (auto& m : mix) m /= s; }
    float* acc = scratch<float>("shim.acc", R * P);
    half* xn = scratch<half>("shim.xn", R * C);
    // each hidden state's LayerNorm and projection, added into the mix by the GEMM itself (alpha mix_k, beta 1)
    auto mixIn = [&](int k, const float* state) {
      layerNorm(state, xn, R, C, M.f("c/lm/norm/scale"), M.f("c/lm/norm/offset"));
      lin(xn, "c/lm/projection/weights", acc, R, C, P, k ? 1.f : 0.f, nullptr, mix[k]);
    };
    float* x = scratch<float>("esmc.x", R * C);
    run1d("ef2_embed", R * C, EmbedArgs{ids, M.f("c/embed/weights"), x, (uint)R, (uint)C});
    mixIn(0, x);
    for (int l = 0; l < e.layers; ++l) {
      esmcBlock(e, x, seq, l);
      if (l + 1 < e.layers) mixIn(l + 1, x);
    }
    float* last = scratch<float>("esmc.last", R * C);
    layerNorm(x, last, R, C, M.f("c/final_norm/scale"), nullptr);
    mixIn(e.layers, last);
    lin(acc, "c/lm/downproject/weights", single, R, P, P, 0.f, M.f("c/lm/downproject/bias"));
  }
  // a non-protein token's state is zero: LN(0) is the offset, the mix sums to one
  float* zero = scratch<float>("shim.zero", P);
  float* z1 = scratch<float>("shim.z1", P);
  lin(M.f("c/lm/norm/offset"), "c/lm/projection/weights", z1, 1, C, P);
  lin(z1, "c/lm/downproject/weights", zero, 1, P, P, 0.f, M.f("c/lm/downproject/bias"));
  float* s = scratch<float>("shim.tokens", (size_t)T * P);
  run1d("ef2_scatter_rows", (size_t)T * P, ScatterRowsArgs{single, tokenToRow, zero, s, (uint)T, (uint)P});
  if (!lmZ) return;
  // the pair, a block of rows at a time: [a * b | a - b] -> mlp1 + bias, gelu -> mlp2 + bias -> LN. The join in half
  // where the states are small; the join and both MLP GEMMs in float where they are not. ESM-C 6B's states reach |a|
  // 163-189 on 6MRR: a * a on the diagonal is 27,000-36,000, and either the join or mlp1's output passed half's 65504 -
  // inf, then a NaN pair. 600M's and 300M's reach 29-50 (6MRR, 5CAJ, 1TIM). Decided from the states themselves (T x P
  // floats read back) with the line at 90, between the two, so the small towers keep their half path byte for byte
  const bool wide = [&] {
    std::vector<float> h = download(s, (size_t)T * P);
    float m = 0; for (float v : h) m = std::max(m, std::fabs(v));
    if (getenv("EF2_STARTUP")) printf("language model: per-token states up to %.1f\n", m);
    return m > 90.f;
  }();
  int bi = std::max(1, std::min(T, (int)(((size_t)32 << 20) / ((size_t)T * 2 * P * (wide ? 2 : 1)))));
  half* join = wide ? nullptr : scratch<half>("shim.join", (size_t)bi * T * 2 * P);
  float* joinF = wide ? scratch<float>("shim.joinf", (size_t)bi * T * 2 * P) : nullptr;
  half* hid = wide ? nullptr : scratch<half>("shim.hid", (size_t)bi * T * P);
  float* hidF = wide ? scratch<float>("shim.hidf", (size_t)bi * T * P) : nullptr;
  for (int i0 = 0; i0 < T; i0 += bi) {
    int b = std::min(bi, T - i0); size_t cells = (size_t)b * T;
    float* o = lmZ + (size_t)i0 * T * P;
    if (wide) {
      run1d("ef2_pair_join_f", cells * P, PairJoinFArgs{s, joinF, (uint)T, (uint)i0, (uint)b, (uint)P});
      Gemm g{}; g.X = joinF; g.tx = F32; g.W = M.f("c/lm/pair_mlp_1/weights"); g.tw = F32; g.Y = hidF; g.ty = F32;
      g.rows = cells; g.in = 2 * P; g.out = P; g.bias = M.f("c/lm/pair_mlp_1/bias"); g.gelu = true; g.label = "lm.pair_mlp_1";
      gemm(g);
      Gemm g2{}; g2.X = hidF; g2.tx = F32; g2.W = M.f("c/lm/pair_mlp_2/weights"); g2.tw = F32; g2.Y = o; g2.ty = F32;
      g2.rows = cells; g2.in = P; g2.out = P; g2.bias = M.f("c/lm/pair_mlp_2/bias"); g2.label = "lm.pair_mlp_2";
      gemm(g2);
    } else {
      run1d("ef2_pair_join", cells * P, PairJoinArgs{s, join, (uint)T, (uint)i0, (uint)b, (uint)P});
      linH(join, "c/lm/pair_mlp_1/weights", hid, cells, 2 * P, P, M.f("c/lm/pair_mlp_1/bias"), true);
      lin(hid, "c/lm/pair_mlp_2/weights", o, cells, P, P, 0.f, M.f("c/lm/pair_mlp_2/bias"));
    }
    layerNorm(o, o, cells, P, M.f("c/lm/pair_norm/scale"), M.f("c/lm/pair_norm/offset"));
  }
}
