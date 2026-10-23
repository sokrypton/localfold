// ESM-C's tower and ESMFold2's language-model shim (cuda/plm/esmc.cuh and cuda/ef2/src/shim.cuh are the reading):
//
//   tower block: h = LN(x) @ qkv; q, k = LN(q), LN(k) over the full width (no offset), rotated (head 64, base 10000);
//                x += attn(q, k, v; within a chain) @ attn_out / s;  x += swiglu(LN(x) @ fc1) @ fc2 / s
//   shim: single = (sum_k softmax(combine)_k LN(h_k) @ projection) @ downproject + bias over the 37 hidden states;
//         pair [a * b | a - b] -> 512 -> gelu -> 256 -> 256 -> LN
// The tower's four matrices a block and the shim's are float16; activations float32 between them.
#include "ef2.h"
#include <cmath>

// the language-model shim's tensor: the FOLDING bundle's own where it carries one (the released models' exports do -
// tools/export_esmfold2_trunk.py - and the shim is per model, so one ESM-C 6B tower serves ESMFold2-Fast and ESMFold2),
// else the tower bundle's (the 600M and 300M towers ship the shim beside the tower and their folding bundles have none)
static std::string shim(const std::string& leaf) {
  return (M.has("f/lm/combine") ? "f/lm/" : "c/lm/") + leaf;
}
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
  if (M.isSwiglu(name)) return towerW(name);
  if (!M.isResident(name)) return swigluPairsInPlace(name, C, ffn);
  half* w = const_cast<half*>(towerW(name));
  if (ffn % 8 == 0 && ffn <= 8192) {
    run(ffn <= 4096 ? "ef2_interleave8_inplace" : "ef2_interleave8_inplace_wide",
        Grid{(uint32_t)C, 1, 1}, 256, Interleave8Args{w, w + ffn, w, (uint)C, (uint)ffn, (uint)(2 * ffn), 0});
    return w;
  }
  half* out = scratch<half>("tower.w16s", (size_t)C * 2 * ffn);
  run1d("ef2_interleave8", (size_t)C * (ffn >> 2), Interleave8Args{w, w + ffn, out, (uint)C, (uint)ffn, (uint)(2 * ffn), 0});
  return out;
}

static void esmcBlock(const Esmc& e, float* x, half* xn, const int* seq, int layer) {
  std::string B = "c/blocks/" + std::to_string(layer) + "/";
  size_t R = e.rows; int C = e.model, H = e.heads;
  float* qkv = scratch<float>("esmc.qkv", R * 3 * C);
  towerLin(xn, B + "qkv/weights", qkv, R, C, 3 * C);
  float* q = scratch<float>("esmc.q", R * C); float* k = scratch<float>("esmc.k", R * C);
  if (C % 64 == 0 && C <= 1152) {
    run("ef2_qk_norm_rope", Grid{(uint32_t)((R + 7) / 8), 2, 1}, 256,
        QkNormRopeArgs{qkv, M.f(B + "q_norm/scale"), M.f(B + "k_norm/scale"), q, k, (uint)R, (uint)C});
  } else {
    layerNorm(qkv, q, R, C, M.f(B + "q_norm/scale"), nullptr, 1e-5f, 3 * C, C);
    layerNorm(qkv + C, k, R, C, M.f(B + "k_norm/scale"), nullptr, 1e-5f, 3 * C, C);
    run1d("ef2_rope", R * 32, RopeArgs{q, k, (uint)R, (uint)H, (uint)C, 0});
  }
  half* ctx = scratch<half>("esmc.ctx", R * C);
  run("ef2_esmc_attend", Grid{(uint32_t)((R + 31) / 32), (uint32_t)H, 1}, 128,
      EsmcAttendArgs{q, k, qkv + 2 * C, seq, ctx, (uint)R, (uint)C, (uint)(3 * C), 0});
  towerLin(ctx, B + "attn_out/weights", x, R, C, C, 1.f / e.residualScale, 1.f);
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
    std::vector<float> mix(M.len(shim("combine")));
    { const float* cmb = M.hostF(shim("combine")); float mx = -INFINITY, s = 0;
      for (size_t i = 0; i < mix.size(); ++i) mx = std::max(mx, cmb[i]);
      for (size_t i = 0; i < mix.size(); ++i) { mix[i] = expf(cmb[i] - mx); s += mix[i]; }
      for (auto& m : mix) m /= s; }
    float* acc = scratch<float>("shim.acc", R * P);
    half* attnXn = scratch<half>("esmc.xn", R * C);
    float* x = scratch<float>("esmc.x", R * C);
    run1d("ef2_embed", R * C, EmbedArgs{ids, M.f("c/embed/weights"), x, (uint)R, (uint)C});
    if (C <= 1152 && (C & 3) == 0) {
      float* mixXn = scratch<float>("shim.mixXn", R * C);
      const float* ss = M.f(shim("norm/scale"));
      const float* so = M.f(shim("norm/offset"));
      size_t sgs = R <= 256 ? 2 : 8;
      Grid gRows = grid1d((R + sgs - 1) / sgs, 1);
      for (int l = 0; l < e.layers; ++l) {
        std::string B = "c/blocks/" + std::to_string(l) + "/";
        run("ef2_shim_norm_acc", gRows, (int)(sgs * 32),
            ShimNormAccArgs{x, ss, so, M.f(B + "attn_norm/scale"), M.f(B + "attn_norm/offset"), nullptr,
                            mixXn, attnXn, (uint)R, (uint)C, l == 0 ? 1u : 0u, mix[l]});
        esmcBlock(e, x, attnXn, seq, l);
      }
      run("ef2_shim_norm_acc", gRows, (int)(sgs * 32),
          ShimNormAccArgs{x, ss, so, nullptr, nullptr, M.f("c/final_norm/scale"),
                          mixXn, attnXn, (uint)R, (uint)C, 0u, mix[e.layers]});
      lin(attnXn, shim("projection/weights"), acc, R, C, P);
    } else {
      half* xn = scratch<half>("shim.xn", R * C);
      // each hidden state's LayerNorm and projection, added into the mix by the GEMM itself (alpha mix_k, beta 1)
      auto mixIn = [&](int k, const float* state) {
        layerNorm(state, xn, R, C, M.f(shim("norm/scale")), M.f(shim("norm/offset")));
        lin(xn, shim("projection/weights"), acc, R, C, P, k ? 1.f : 0.f, nullptr, mix[k]);
      };
      for (int l = 0; l < e.layers; ++l) {
        std::string B = "c/blocks/" + std::to_string(l) + "/";
        layerNormDual(x, xn, M.f(shim("norm/scale")), M.f(shim("norm/offset")),
                      attnXn, M.f(B + "attn_norm/scale"), M.f(B + "attn_norm/offset"), R, C);
        lin(xn, shim("projection/weights"), acc, R, C, P, l ? 1.f : 0.f, nullptr, mix[l]);
        esmcBlock(e, x, attnXn, seq, l);
      }
      float* last = scratch<float>("esmc.last", R * C);
      layerNorm(x, last, R, C, M.f("c/final_norm/scale"), nullptr);
      mixIn(e.layers, last);
    }
    lin(acc, shim("downproject/weights"), single, R, P, P, 0.f, M.f(shim("downproject/bias")));
  }
  // a non-protein token's state is zero: LN(0) is the offset, the mix sums to one
  float* zero = scratch<float>("shim.zero", P);
  float* z1 = scratch<float>("shim.z1", P);
  lin(M.f(shim("norm/offset")), shim("projection/weights"), z1, 1, C, P);
  lin(z1, shim("downproject/weights"), zero, 1, P, P, 0.f, M.f(shim("downproject/bias")));
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
      run1d("ef2_pair_join_f", (cells * P) >> 2, PairJoinFArgs{s, joinF, (uint)T, (uint)i0, (uint)b, (uint)P});
      Gemm g{}; g.X = joinF; g.tx = F32; g.W = M.f(shim("pair_mlp_1/weights")); g.tw = F32; g.Y = hidF; g.ty = F32;
      g.rows = cells; g.in = 2 * P; g.out = P; g.bias = M.f(shim("pair_mlp_1/bias")); g.gelu = true; g.label = "lm.pair_mlp_1";
      gemm(g);
      Gemm g2{}; g2.X = hidF; g2.tx = F32; g2.W = M.f(shim("pair_mlp_2/weights")); g2.tw = F32; g2.Y = o; g2.ty = F32;
      g2.rows = cells; g2.in = P; g2.out = P; g2.bias = M.f(shim("pair_mlp_2/bias")); g2.label = "lm.pair_mlp_2";
      gemm(g2);
    } else {
      run1d("ef2_pair_join", (cells * P) >> 2, PairJoinArgs{s, join, (uint)T, (uint)i0, (uint)b, (uint)P});
      linH(join, shim("pair_mlp_1/weights"), hid, cells, 2 * P, P, M.f(shim("pair_mlp_1/bias")), true);
      lin(hid, shim("pair_mlp_2/weights"), o, cells, P, P, 0.f, M.f(shim("pair_mlp_2/bias")));
    }
    layerNorm(o, o, cells, P, M.f(shim("pair_norm/scale")), M.f(shim("pair_norm/offset")));
  }
}
