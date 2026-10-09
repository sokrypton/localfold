// ESM2 3B, the language model chai-1's tokens read (cuda/plm/esm2.cuh is the reading; alphafold3/model/esm.py):
//
//   x = embed(ids) * (1 - 0.15 * 0.8)                       token dropout's inference constant
//   per block:  h = LN(x);  q, k, v = h W + b;  q, k = RoPE(q, k) (head 64, base 10000, split halves, position = row)
//               x += attn(q, k, v) W_out + b;  x += gelu(LN(x) fc1 + b) fc2 + b
//   out = LN(x; final_norm), the last state only, one chain at a time as [BOS, residues, EOS], BOS and EOS dropped
//
// Its matrices stay resident as int8 codes in the mapped blob (Model::loadBlobResident), each expanded to half for its
// GEMM; biases and norms float32. The attention is the core's flash kernel, its gate held open.
#include "af3.h"
#include <cmath>

static std::string blockName(int layer, const std::string& module, const std::string& leaf) {
  return "e/esm2/blocks/" + module + "/" + num(layer) + "/" + leaf;
}
// Y = X W (+ beta Y) (+ bias) (gelu), W the named resident matrices side by side
static void esmGemm(const half* X, const std::vector<std::string>& names, void* Y, bool yHalf, int rows, int in, int out, float beta,
                    const float* bias = nullptr, bool gelu = false) {
  half* w = scratch<half>("esm2.w16", (size_t)in * out);
  int col0 = 0;
  for (const std::string& name : names) {
    const Model::Int8Matrix& m = M.int8(name);
    if (m.rows != (size_t)in || col0 + m.cols > out) die("%s is not a [%d, *] int8 matrix of the blob's", name.c_str(), in);
    int g = (int)((m.rows + m.rowBlocks - 1) / m.rowBlocks);
    run1d("af3_expand8", m.rows * m.cols, Expand8Args{m.codes, m.scales, w, (uint)m.rows, (uint)m.cols, (uint)g, (uint)out, (uint)col0, 0});
    col0 += m.cols;
  }
  if (col0 != out) die("%s...: %d columns, not %d", names[0].c_str(), col0, out);
  Gemm gm{}; gm.X = X; gm.tx = F16; gm.W = w; gm.tw = F16; gm.Y = Y; gm.ty = yHalf ? F16 : F32; gm.rows = rows; gm.in = in; gm.out = out;
  gm.beta = beta; gm.bias = bias; gm.gelu = gelu; gm.accFloat = true; gm.label = "esm2";
  gemm(gm);
}

// one chain: ids [n + 2] (BOS, its residues, EOS) -> out [n][C], the last state after the final LayerNorm
static void embedChain(const int* ids, int n, float* out) {
  int layers = 0; while (M.hasInt8(blockName(layers, "q", "weights"))) ++layers;
  int C = (int)M.meta("e/esm2/embed/weights#1"), ffn = M.int8(blockName(0, "fc1", "weights")).cols, heads = C / 64;
  if (layers == 0 || C % 64) die("e/: not ESM2's blob (%d blocks, width %d)", layers, C);
  int R = n + 2;
  float* x = scratch<float>("esm2.x", (size_t)R * C);
  half* xn = scratch<half>("esm2.xn", (size_t)R * C);
  float* qkv = scratch<float>("esm2.qkv", (size_t)R * 3 * C);
  half* qkvg = scratch<half>("esm2.qkvg", (size_t)R * 4 * C);
  half* ctx = scratch<half>("esm2.ctx", (size_t)R * C);
  half* h = scratch<half>("esm2.h", (size_t)R * ffn);
  // the rotation table in double, as the reference's sincos: position r, frequency 10000^(-2d/64)
  std::vector<float> cosT((size_t)R * 32), sinT((size_t)R * 32);
  for (int r = 0; r < R; ++r)
    for (int d = 0; d < 32; ++d) {
      double angle = (double)r * std::pow(10000.0, -(double)(2 * d) / 64.0);
      cosT[(size_t)r * 32 + d] = (float)std::cos(angle); sinT[(size_t)r * 32 + d] = (float)std::sin(angle);
    }
  float* dCos = scratch<float>("esm2.cos", cosT.size()); upload(dCos, cosT.data(), cosT.size() * 4);
  float* dSin = scratch<float>("esm2.sin", sinT.size()); upload(dSin, sinT.data(), sinT.size() * 4);
  run1d("af3_esm_embed", (size_t)R * C, EsmEmbedArgs{ids, W("e/esm2/embed/weights"), x, (uint)R, (uint)C, 1.f - 0.15f * 0.8f, 0});
  for (int l = 0; l < layers; ++l) {
    auto N = [&](const char* module, const char* leaf) { return blockName(l, module, leaf); };
    layerNorm(x, xn, R, C, W(N("attn_norm", "scale")), W(N("attn_norm", "offset")));
    const float* qkvBias = M.derived<float>("esm2.qkvBias:" + num(l), (size_t)3 * C, [&](float* b) {
      for (int m = 0; m < 3; ++m) copy(b + m * C, W(N(m == 0 ? "q" : m == 1 ? "k" : "v", "bias")), (size_t)C * 4);
    });
    esmGemm(xn, {N("q", "weights"), N("k", "weights"), N("v", "weights")}, qkv, false, R, C, 3 * C, 0.f, qkvBias);
    run1d("af3_esm_pack", (size_t)R * C, EsmPackArgs{qkv, dCos, dSin, qkvg, (uint)R, (uint)C});
    Attention at{}; at.qkvg = qkvg; at.out = ctx; at.n = R; at.heads = heads; at.D = 64; at.rows = 1; at.scale = 0.125f;
    attention(at);
    esmGemm(ctx, {N("attn_out", "weights")}, x, false, R, C, C, 1.f, W(N("attn_out", "bias")));
    layerNorm(x, xn, R, C, W(N("ffn_norm", "scale")), W(N("ffn_norm", "offset")));
    esmGemm(xn, {N("fc1", "weights")}, h, true, R, C, ffn, 0.f, W(N("fc1", "bias")), true);
    esmGemm(h, {N("fc2", "weights")}, x, false, R, ffn, C, 1.f, W(N("fc2", "bias")));
  }
  layerNorm(x + C, out, n, C, W("e/esm2/final_norm/scale"), W("e/esm2/final_norm/offset"));
}

// every protein chain alone through the tower, its rows gathered onto the tokens (zeros elsewhere): [tokens][E]
float* esmEmbeddings(int tokens, int& E) {
  E = (int)M.meta("e/esm2/embed/weights#1");
  const int* ids = M.hostI("esm.ids"); const int* lens = M.hostI("esm.chainLengths");
  size_t chains = M.len("esm.chainLengths"), total = 0;
  for (size_t c = 0; c < chains; ++c) total += lens[c];
  float* rows = scratch<float>("esm.rows", std::max<size_t>(1, total) * E);
  size_t row = 0, at = 0;
  for (size_t c = 0; c < chains; ++c) {
    int L = lens[c];
    int* d = scratch<int>("esm.ids", L + 2);
    upload(d, ids + at, (size_t)(L + 2) * 4);
    embedChain(d, L, rows + row * E);
    row += L; at += L + 2;
  }
  float* out = scratch<float>("esm.emb", (size_t)tokens * E);
  run1d("af3_gather_esm", (size_t)tokens * E, GatherEsmArgs{rows, M.i("esm.tokenRow"), out, (uint)tokens, (uint)E});
  releaseScratch({"esm2.", "esm.rows", "esm.ids"});
  return out;
}
