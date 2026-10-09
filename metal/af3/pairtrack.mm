// The pair track - triangle multiplication, grid ("triangle") attention, transition - and the pairformer's single
// track, shared by the trunk's pairformer, its MSA and template stacks and the confidence head (cuda/af3/src/
// pairtrack.cuh and trunk.cuh's singleTrack are the reading). Every attention runs on the core's flash kernel: the
// column direction ACROSS the pair's leading axis (its strides), so nothing is transposed.
#include "af3.h"
#include <cmath>

static int round8(int n) { return (n + 7) / 8 * 8; }

half* biasLayout(const float* raw, const std::string& name, int n, int heads, int stride, bool swap) {
  half* bias = scratch<half>(name, (size_t)heads * n * stride);
  run1d("af3_bias_layout", (size_t)heads * n * stride,
        BiasLayoutArgs{raw, bias, (uint)n, (uint)stride, (uint)heads, swap ? 1u : 0u, (uint)heads, 0, (float)M_LOG2E, 0});
  return bias;
}

// ---------------------------------------------------------------- triangle multiplication
// LN -> one GEMM gating a and b into channel-major padded planes (gemmTriGate; AF3 interleaves a and b by channel) ->
// the planes' batched product (/ n under triangleMulDivideByLength) -> the centre LayerNorm back to rows -> the
// output projection -> the gating linear's GEMM adding the gated product into the pair (gemmGatedAdd)
void triangle(float* pair, const Masks& m, int n, int C, const std::string& pre, bool outgoing, bool divide) {
  size_t P = (size_t)n * n;
  int np = round8(n); size_t plane = (size_t)np * np;
  half* xn = scratch<half>("tri.xn", P * C);
  ln(pair, xn, P, C, pre + ".leftNormInputScale", pre + ".leftNormInputOffset");
  half* a = scratch<half>("tri.a", plane * C); half* b = scratch<half>("tri.b", plane * C);
  static half *zeroedA = nullptr, *zeroedB = nullptr; static size_t zeroedSize = 0;
  if (np != n && (a != zeroedA || b != zeroedB || plane * C != zeroedSize)) {      // the padding is written by nothing
    fill(a, 0, plane * C * 2); fill(b, 0, plane * C * 2);
    zeroedA = a; zeroedB = b; zeroedSize = plane * C;
  }
  const half* w4 = M.derived<half>("trigate:" + pre, (size_t)C * 4 * C, [&](half* out) {
    run1d("af3_trigate_weight", (size_t)C * 4 * C, TriGateWArgs{Wh(pre + ".projection"), Wh(pre + ".gate"), out, (uint)C, (uint)C});
  });
  gemmTriGate(xn, w4, m.pair, a, b, 0, P, C, plane, n, np);
  float* prod = scratch<float>("tri.prod", plane * C);
  {   // per channel: outgoing prod[i][j] = sum_k a[i][k] b[j][k]; incoming sum_k b[k][i] a[k][j]
    Gemm g{}; g.tx = F16; g.tw = F16; g.ty = F32; g.rows = np; g.in = np; g.out = np; g.ldx = g.ldw = g.ldy = np;
    g.sx = g.sw = g.sy = (int64_t)plane; g.batch = C; g.Y = prod; g.alpha = divide ? 1.f / n : 1.f;
    g.label = outgoing ? "triangle out contraction" : "triangle in contraction";
    if (outgoing) { g.X = a; g.W = b; g.transW = true; }
    else { g.X = b; g.transX = true; g.W = a; }
    gemm(g);
  }
  half* cn = scratch<half>("tri.cn", P * C);
  centerNorm(prod, cn, n, np, C, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"));
  half* outH = scratch<half>("tri.outh", P * C);
  linH(cn, pre + ".outputProjection", outH, P, C, C);
  gemmGatedAdd(xn, Wh(pre + ".gatingLinear"), outH, pair, P, C, C);
}

// ---------------------------------------------------------------- grid attention
// rows of the pair (tr: its columns) attend along themselves, biased by LN(pair) projected per head - the column
// direction's bias read transposed where the dialect swaps it (swapTransposedBias)
void gridAttention(float* pair, const Masks& m, int n, int C, const std::string& pre, bool tr, bool swap) {
  int heads = metaI(pre + ".heads"), D = metaI(pre + ".dimension"), Wd = heads * D;
  size_t P = (size_t)n * n;
  half* xn = scratch<half>("grid.xn", P * C);
  ln(pair, xn, P, C, pre + ".actNormScale", pre + ".actNormOffset");
  float* raw = scratch<float>("grid.raw", P * heads);
  lin(xn, pre + ".pairBiasProjection", raw, P, C, heads);
  int stride = round8(n);
  half* bias = biasLayout(raw, "grid.bias", n, heads, stride, tr && swap);
  half* qkvg = scratch<half>("grid.qkvg", P * 4 * Wd);
  linW(xn, qkvgWeight(pre, C, Wd, true), qkvg, P, C, 4 * Wd, qkvgBias(pre, Wd), "grid qkvg");
  half* o = scratch<half>("grid.o", P * Wd);
  Attention at{}; at.qkvg = qkvg; at.out = o; at.n = n; at.heads = heads; at.D = D; at.rows = n; at.scale = 1.f / sqrtf((float)D);
  at.bias = bias; at.biasStride = stride;
  if (!m.ones) { at.mask = m.pair; at.maskB = tr ? 1 : n; at.maskK = tr ? n : 1; }
  if (tr) { at.rowStride = 4 * Wd; at.posStride = (int64_t)n * 4 * Wd; at.outRowStride = Wd; at.outPosStride = (int64_t)n * Wd; }
  attention(at);
  lin(o, pre + ".outputProjection", pair, P, Wd, C, 1.f, Wopt(pre + ".outputProjectionBias"));
}

// ---------------------------------------------------------------- transition
// LN -> SwiGLU in the first GEMM's epilogue -> the second GEMM adding into x; in row chunks
void transition(float* x, size_t rows, int C, const std::string& pre) {
  size_t w1 = lenW(pre + ".transition1");
  if (w1 % (2 * (size_t)C)) die("%s.transition1 has %zu elements, not %d x 2I", pre.c_str(), w1, C);
  int I = (int)(w1 / (2 * (size_t)C));
  size_t chunk = std::min(rows, std::max<size_t>(64, ((size_t)128 << 20) / (3 * (size_t)I)));
  half* xn = scratch<half>("tr.xn", chunk * C);
  half* g = scratch<half>("tr.g", chunk * I);
  const half* wp = swigluPairs(pre + ".transition1", Wh(pre + ".transition1"), C, I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    ln(x + r0 * C, xn, r, C, pre + ".inputLayerNormScale", pre + ".inputLayerNormOffset");
    gemmSwiglu(xn, wp, g, r, C, I);
    lin(g, pre + ".transition2", x + r0 * C, r, I, C, 1.f);
  }
}

// the five pair updates of a pairformer / MSA / template block, in AF3's order
void pairUpdates(float* pair, const Masks& m, int n, int C, const std::string& pre) {
  bool swap = flag("trunk.dialect.swapTransposedBias"), divide = flag("trunk.dialect.triangleMulDivideByLength");
  triangle(pair, m, n, C, pre + ".triangleMultiplicationOutgoing", true, divide);
  triangle(pair, m, n, C, pre + ".triangleMultiplicationIncoming", false, divide);
  gridAttention(pair, m, n, C, pre + ".pairAttention1", false, swap);
  gridAttention(pair, m, n, C, pre + ".pairAttention2", true, swap);
  transition(pair, (size_t)n * n, C, pre + ".pairTransition");
}

// ---------------------------------------------------------------- the single track
// attention over the tokens biased by the pair (LN, a projection to the heads), then the transition
void singleTrack(float* single, const float* pair, const Masks& m, int n, int C, int Cs, const std::string& B,
                 const float* extraBias) {
  const std::string A = B + ".singleAttention";
  int heads = metaI(A + ".heads"), d = metaI(A + ".dimension"), Wd = heads * d;
  size_t P = (size_t)n * n;
  half* pln = scratch<half>("st.pln", P * C);
  ln(pair, pln, P, C, B + ".singlePairLogitsNormScale", B + ".singlePairLogitsNormOffset");
  float* raw = scratch<float>("st.raw", P * heads);
  lin(pln, B + ".singlePairLogitsProjection", raw, P, C, heads);
  if (extraBias) run1d("af3_add_bias_heads", P * heads, AddBiasHeadsArgs{raw, extraBias, P, (uint)heads, 0});   // (OpenDDE)
  int stride = round8(n);
  half* bias = biasLayout(raw, "st.bias", n, heads, stride, false);
  half* nrm = scratch<half>("st.nrm", (size_t)n * Cs);
  ln(single, nrm, n, Cs, A + ".layerNormScale", A + ".layerNormOffset");
  half* qkvg = scratch<half>("st.qkvg", (size_t)n * 4 * Wd);
  linW(nrm, qkvgWeight(A, Cs, Wd, false), qkvg, n, Cs, 4 * Wd, nullptr, "single qkvg");
  half* o = scratch<half>("st.o", (size_t)n * Wd);
  Attention at{}; at.qkvg = qkvg; at.out = o; at.n = n; at.heads = heads; at.D = d; at.rows = 1; at.scale = 1.f / sqrtf((float)d);
  at.bias = bias; at.biasStride = stride; at.qBias = W(A + ".qBias");
  if (!m.ones) { at.mask = m.seq; at.maskB = 0; at.maskK = 1; }
  attention(at);
  lin(o, A + ".outputProjection", single, n, Wd, Cs, 1.f);
  transition(single, n, Cs, B + ".singleTransition");
}
void pairformerBlock(float* pair, float* single, const Masks& m, int n, int C, int Cs, const std::string& B, const float* extraBias) {
  if (flag("trunk.dialect.parallelPairformer")) die("chai-1's parallel pairformer is not in the native port yet");
  pairUpdates(pair, m, n, C, B);
  singleTrack(single, pair, m, n, C, Cs, B, extraBias);
}
