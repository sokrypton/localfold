// The pair track - triangle multiplication, grid ("triangle") attention, transition - and the pairformer's single
// track, shared by the trunk's pairformer, its MSA and template stacks and the confidence head (cuda/af3/src/
// pairtrack.cuh and trunk.cuh's singleTrack are the reading). Every attention runs on the core's flash kernel: the
// column direction ACROSS the pair's leading axis (its strides), so nothing is transposed.
#include "af3.h"
#include <cmath>

static int round8(int n) { return (n + 7) / 8 * 8; }
// The pair track's working tensors are shared by name across its stages, which run one after another: a pair-sized half
// buffer for each stage's normalised input (pr.x) and two for its intermediates (pr.h1, pr.h2), and one large buffer
// that is the grid attention's q|k|v|g or the triangle's two planes and their product (pr.big). Held per stage they
// were twice the room: 4.0 against 2.0 GB at 1044 tokens.
static void* bigBuffer(size_t bytes) { return scratchBytes("pr.big", bytes); }

half* biasLayout(const float* raw, const std::string& name, int n, int heads, int stride, bool swap) {
  half* bias = scratch<half>(name, (size_t)heads * n * stride);
  const BiasLayoutArgs a{raw, bias, (uint)n, (uint)stride, (uint)heads, swap ? 1u : 0u, (uint)heads, 0, (float)M_LOG2E, 0};
  run("af3_bias_layout", Grid{(uint32_t)((stride + 31) / 32), (uint32_t)((n + 31) / 32), 1}, 256, a);
  return bias;
}

// ---------------------------------------------------------------- triangle multiplication
const half* triGateWeight(const std::string& pre, int C) {
  if (triQuartersApply(C))      // (the matrix units' quarters, built once through a blocks-of-8 temporary)
    return M.derived<half>("trigateQ:" + pre, (size_t)C * 4 * C, [&](half* out) {
      half* w8 = allocT<half>((size_t)C * 4 * C);
      run1d("af3_trigate_weight", (size_t)C * 4 * C, TriGateWArgs{Wh(pre + ".projection"), Wh(pre + ".gate"), w8, (uint)C, (uint)C});
      triQuarters(w8, out, nullptr, nullptr, C);
      release(w8);
    });
  return M.derived<half>("trigate:" + pre, (size_t)C * 4 * C, [&](half* out) {
    run1d("af3_trigate_weight", (size_t)C * 4 * C, TriGateWArgs{Wh(pre + ".projection"), Wh(pre + ".gate"), out, (uint)C, (uint)C});
  });
}
// LN -> one GEMM gating a and b into channel-major padded planes (gemmTriGate; AF3 interleaves a and b by channel) ->
// the planes' batched product (/ n under triangleMulDivideByLength) -> the centre LayerNorm back to rows -> the
// output projection -> the gating linear's GEMM adding the gated product into the pair (gemmGatedAdd)
// the pair's next LayerNorms where the previous update's GEMM already wrote them (its epilogue's lnOut, lnOut2): the
// pair normalised, the norm's name and the buffer. pairLn of the same pair and norm takes its entry; a miss drops both
// (whatever came between may have moved the pair). LOCALFOLD_LN_EMIT=0: none emitted, every norm its own pass
struct Emitted { const float* x = nullptr; std::string norm; half* buf = nullptr; };
static Emitted emitted[2];
static void emit(int k, const float* x, const std::string& norm, half* buf) { emitted[k] = {x, norm, buf}; }
static half* takeEmitted(const float* x, const std::string& norm) {
  for (auto& e : emitted)
    if (e.x == x && e.norm == norm) { e.x = nullptr; return e.buf; }
  emitted[0].x = emitted[1].x = nullptr;
  return nullptr;
}
static half* pairLn(const float* pair, size_t P, int C, const std::string& norm) {
  if (half* b = takeEmitted(pair, norm)) return b;
  half* xn = scratch<half>("pr.x", P * C);
  ln(pair, xn, P, C, norm + "Scale", norm + "Offset");
  return xn;
}
// chai-1's parallel updates all read one input, so its affine-free LayerNorm is taken once a block (parallelPairUpdates)
// and each update's own scale and offset are folded into the weights that read it (foldNorm): the input and, during the
// first (in-place) update, the pair itself. LOCALFOLD_FOLD_NORM=0 is the control
static struct { const float* src[2] = {nullptr, nullptr}; half* xhat = nullptr; } shared;
static const half* sharedNorm(const float* x) {
  return shared.xhat && x && (x == shared.src[0] || x == shared.src[1]) ? shared.xhat : nullptr;
}
// where an update's GEMM emits the next norm: whichever of pr.x and pr.x2 its own input is not
static half* lnTarget(size_t P, int C, const half* inUse) {
  half* x1 = scratch<half>("pr.x", P * C);
  return inUse == x1 ? scratch<half>("pr.x2", P * C) : x1;
}
void triangle(float* pair, const Masks& m, int n, int C, const std::string& pre, bool outgoing, bool divide, float* into,
              const std::string& nextNorm) {
  size_t P = (size_t)n * n;
  int np = round8(n); size_t plane = (size_t)np * np;
  const half* xs = sharedNorm(pair);
  half* xn = xs ? (half*)xs : pairLn(pair, P, C, pre + ".leftNormInput");
  Folded tg, gl;
  if (xs) {
    tg = foldNorm("tg:" + pre, triGateWeight(pre, C), C, 4 * C, pre + ".leftNormInput", nullptr);
    gl = foldNorm("gl:" + pre, Wh(pre + ".gatingLinear"), C, C, pre + ".leftNormInput", nullptr);
  }
  char* big = (char*)bigBuffer(plane * C * 8);         // a and b (half) and their product (float)
  half* a = (half*)big; half* b = a + plane * C;
  padZero(a, 2 * (size_t)C, n, np);                   // (the padding is written by nothing here, and pr.big is shared)
  gemmTriGate(xn, xs ? tg.w : triGateWeight(pre, C), m.pair, a, b, 0, P, C, plane, n, np, xs ? tg.b : nullptr, triQuartersApply(C));
  float* prod = (float*)(big + plane * C * 4);
  {   // per channel: outgoing prod[i][j] = sum_k a[i][k] b[j][k]; incoming sum_k b[k][i] a[k][j]
    Gemm g{}; g.tx = F16; g.tw = F16; g.ty = F32; g.rows = np; g.in = np; g.out = np; g.ldx = g.ldw = g.ldy = np;
    g.sx = g.sw = g.sy = (int64_t)plane; g.batch = C; g.Y = prod; g.alpha = divide ? 1.f / n : 1.f;
    g.label = outgoing ? "triangle out contraction" : "triangle in contraction";
    if (outgoing) { g.X = a; g.W = b; g.transW = true; }
    else { g.X = b; g.transX = true; g.W = a; }
    gemm(g);
  }
  half* cn = scratch<half>("pr.h1", P * C);
  centerNorm(prod, cn, n, np, C, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"));
  half* outH = scratch<half>("pr.h2", P * C);
  half* lnOut = !nextNorm.empty() && !into && hasW(nextNorm + "Scale") ? lnTarget(P, C, xn) : nullptr;
  if (gemmGatedAddDual(xn, xs ? gl.w : Wh(pre + ".gatingLinear"), cn, Wh(pre + ".outputProjection"), into ? into : pair, P, C, C,
                       xs ? gl.b : nullptr, nullptr, outH,
                       "triangle output and gated add", lnOut, lnOut ? W(nextNorm + "Scale") : nullptr,
                       lnOut ? Wopt(nextNorm + "Offset") : nullptr))
    emit(0, pair, nextNorm, lnOut);
}

// ---------------------------------------------------------------- grid attention
// rows of the pair (tr: its columns) attend along themselves, biased by LN(pair) projected per head - the column
// direction's bias read transposed where the dialect swaps it (swapTransposedBias)
void gridAttention(float* pair, const Masks& m, int n, int C, const std::string& pre, bool tr, bool swap, float* into,
                   bool untransposed, const std::string& nextNorm) {
  int heads = metaI(pre + ".heads"), D = metaI(pre + ".dimension"), Wd = heads * D;
  size_t P = (size_t)n * n;
  const half* xs = sharedNorm(pair);
  half* xn = xs ? (half*)xs : pairLn(pair, P, C, pre + ".actNorm");
  float* raw = scratch<float>("grid.raw", P * heads);
  Folded fb, fq;
  if (xs) {
    fb = foldNorm("gb:" + pre, Wh(pre + ".pairBiasProjection"), C, heads, pre + ".actNorm", nullptr);
    fq = foldNorm("gq:" + pre, qkvgWeight(pre, C, Wd, true), C, 4 * Wd, pre + ".actNorm", qkvgBias(pre, Wd));
    linW(xn, fb.w, raw, P, C, heads, 0.f, fb.b, 1.f, "grid bias projection");
  } else lin(xn, pre + ".pairBiasProjection", raw, P, C, heads);
  int stride = round8(n);
  half* bias = biasLayout(raw, "grid.bias", n, heads, stride, tr && swap);
  half* qkvg = (half*)bigBuffer(P * 4 * Wd * 2);
  linW(xn, xs ? fq.w : qkvgWeight(pre, C, Wd, true), qkvg, P, C, 4 * Wd, xs ? fq.b : qkvgBias(pre, Wd), "grid qkvg");
  half* o = scratch<half>("pr.h1", P * Wd);
  Attention at{}; at.qkvg = qkvg; at.out = o; at.n = n; at.heads = heads; at.D = D; at.rows = n; at.scale = 1.f / sqrtf((float)D);
  at.bias = bias; at.biasStride = stride;
  if (!m.ones) { at.mask = m.pair; at.maskB = tr ? 1 : n; at.maskK = tr ? n : 1; }
  if (tr) { at.rowStride = 4 * Wd; at.posStride = (int64_t)n * 4 * Wd; at.outRowStride = Wd; at.outPosStride = (int64_t)n * Wd; }
  // (untransposed: the column direction's output kept at (r, j), chai's - its ending-node update at (i, j) is AF3's at (j, i))
  if (tr && untransposed) { at.outRowStride = 0; at.outPosStride = 0; }
  attention(at);
  // chai-1's confidence blocks: each direction's output projection plus its transposed twin, summed once (their two
  // applications cancel to one at inference - af3-any-model's dual_output)
  if (!pre.compare(0, 11, "confidence.") && hasW(pre + ".outputProjectionTransposed")) {
    const half* w = M.derived<half>("dual:" + pre, (size_t)Wd * C, [&](half* out) {
      float* sum = allocT<float>((size_t)Wd * C);
      copy(sum, W(pre + ".outputProjection"), (size_t)Wd * C * 4);
      add(sum, W(pre + ".outputProjectionTransposed"), (size_t)Wd * C);
      toHalf(sum, out, (size_t)Wd * C);
      release(sum);
    });
    linW(o, w, into ? into : pair, P, Wd, C, 1.f, Wopt(pre + ".outputProjectionBias"), 1.f, "grid output");
  } else if (!nextNorm.empty() && !into && hasW(nextNorm + "Scale") && lenW(pre + ".outputProjection") == (size_t)Wd * C) {
    half* lnOut = lnTarget(P, C, xn);
    Gemm g{}; g.X = o; g.tx = F16; g.W = Wh(pre + ".outputProjection"); g.tw = F16; g.Y = pair; g.rows = P; g.in = Wd; g.out = C;
    g.beta = 1.f; g.bias = Wopt(pre + ".outputProjectionBias"); g.label = "grid output, next norm";
    g.lnOut = lnOut; g.lnScale = W(nextNorm + "Scale"); g.lnOffset = Wopt(nextNorm + "Offset");
    if (gemm(g)) emit(0, pair, nextNorm, lnOut);
  } else lin(o, pre + ".outputProjection", into ? into : pair, P, Wd, C, 1.f, Wopt(pre + ".outputProjectionBias"));
}

// ---------------------------------------------------------------- transition
// LN -> SwiGLU in the first GEMM's epilogue -> the second GEMM adding into x; in row chunks
void transition(float* x, size_t rows, int C, const std::string& pre, float* into, const std::string& next1,
                const std::string& next2) {
  size_t w1 = lenW(pre + ".transition1");
  if (w1 % (2 * (size_t)C)) die("%s.transition1 has %zu elements, not %d x 2I", pre.c_str(), w1, C);
  int I = (int)(w1 / (2 * (size_t)C));
  size_t chunk = std::min(rows, std::max<size_t>(64, ((size_t)128 << 20) / (3 * (size_t)I)));
  half* xn = scratch<half>("tr.xn", chunk * C);
  half* g = scratch<half>("tr.g", chunk * I);
  const half* wp = swigluPairs(pre + ".transition1", C, I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    half* in = r0 == 0 && r == rows ? takeEmitted(x, pre + ".inputLayerNorm") : nullptr;
    if (const half* xs = in ? nullptr : sharedNorm(x)) {    // (chai's shared input norm: this norm's affine on its rows)
      run1d("af3_affine_h", r * C, AffineHArgs{xs + r0 * C, xn, W(pre + ".inputLayerNormScale"), Wopt(pre + ".inputLayerNormOffset"),
                                              (u64)(r * C), (uint)C, 0});
      in = xn;
    }
    if (!in) { ln(x + r0 * C, xn, r, C, pre + ".inputLayerNormScale", pre + ".inputLayerNormOffset"); in = xn; }
    gemmSwiglu(in, wp, g, r, C, I);
    // the pair's next norms (next1 into pr.x, next2 into pr.x2 - both free once the SwiGLU has read its input)
    if (!next1.empty() && !into && r == rows && C == 128 && hasW(next1 + "Scale") && (next2.empty() || hasW(next2 + "Scale"))) {
      Gemm G{}; G.X = g; G.tx = F16; G.W = Wh(pre + ".transition2"); G.tw = F16; G.Y = x; G.rows = r; G.in = I; G.out = C;
      G.beta = 1.f; G.label = "transition2, next norms";
      G.lnOut = scratch<half>("pr.x", rows * C); G.lnScale = W(next1 + "Scale"); G.lnOffset = Wopt(next1 + "Offset");
      if (!next2.empty()) { G.lnOut2 = scratch<half>("pr.x2", rows * C); G.lnScale2 = W(next2 + "Scale"); G.lnOffset2 = Wopt(next2 + "Offset"); }
      if (lenW(pre + ".transition2") != (size_t)I * C) die("%s.transition2 has %zu elements", pre.c_str(), lenW(pre + ".transition2"));
      if (gemm(G)) { emit(0, x, next1, G.lnOut); if (G.lnOut2) emit(1, x, next2, G.lnOut2); }
      continue;
    }
    lin(g, pre + ".transition2", (into ? into : x) + r0 * C, r, I, C, 1.f);
  }
}

// the five pair updates of a pairformer / MSA / template block, in AF3's order
void pairUpdates(float* pair, const Masks& m, int n, int C, const std::string& pre, const std::string& next1,
                 const std::string& next2) {
  bool swap = flag("trunk.dialect.swapTransposedBias"), divide = flag("trunk.dialect.triangleMulDivideByLength");
  triangle(pair, m, n, C, pre + ".triangleMultiplicationOutgoing", true, divide, nullptr, pre + ".triangleMultiplicationIncoming.leftNormInput");
  triangle(pair, m, n, C, pre + ".triangleMultiplicationIncoming", false, divide, nullptr, pre + ".pairAttention1.actNorm");
  gridAttention(pair, m, n, C, pre + ".pairAttention1", false, swap, nullptr, false, pre + ".pairAttention2.actNorm");
  gridAttention(pair, m, n, C, pre + ".pairAttention2", true, swap, nullptr, false, pre + ".pairTransition.inputLayerNorm");
  transition(pair, (size_t)n * n, C, pre + ".pairTransition", nullptr, next1, next2);
}

// chai-1's parallel updates: the stage's input kept (par.base); the first update runs on the pair itself (it IS the input
// then), every later one reads the kept input and adds its residual into the pair; the ending-node attention's residual
// untransposed
void parallelPairUpdates(float* pair, const Masks& m, int n, int C, const std::string& pre, const char* which) {
  bool swap = flag("trunk.dialect.swapTransposedBias"), divide = flag("trunk.dialect.triangleMulDivideByLength");
  size_t P = (size_t)n * n;
  static const bool fold = !getenv("LOCALFOLD_FOLD_NORM") || atoi(getenv("LOCALFOLD_FOLD_NORM")) != 0;
  shared = {};
  // (folded, every update reads the input only through its norm - so no kept copy: they all take the pair, whose
  // xhat is the block's input whatever the updates before have added)
  float* base = fold ? pair : scratch<float>("par.base", P * C);
  if (fold) {
    shared.xhat = scratch<half>("par.xhat", P * C);
    layerNorm(pair, shared.xhat, P, C, nullptr, nullptr);
    shared.src[0] = pair;
  } else copy(base, pair, P * C * 4);
  bool first = true;
  for (const char* u = which; *u; ++u) {
    bool inPlace = first && *u != 'c';
    first = false;
    float* in = inPlace ? pair : base;
    float* into = inPlace ? nullptr : pair;
    switch (*u) {
      case 'o': triangle(in, m, n, C, pre + ".triangleMultiplicationOutgoing", true, divide, into); break;
      case 'i': triangle(in, m, n, C, pre + ".triangleMultiplicationIncoming", false, divide, into); break;
      case 'r': gridAttention(in, m, n, C, pre + ".pairAttention1", false, swap, into); break;
      case 'c': gridAttention(in, m, n, C, pre + ".pairAttention2", true, swap, into, true); break;
      case 't': transition(in, P, C, pre + ".pairTransition", into); break;
    }
  }
}

// ---------------------------------------------------------------- the single track
// attention over the tokens biased by the pair (LN, a projection to the heads), then the transition
void singleTrack(float* single, const float* pair, const Masks& m, int n, int C, int Cs, const std::string& B,
                 const float* extraBias) {
  const std::string A = B + ".singleAttention";
  // chai-1's parallel block: the gate is sigmoid(g + 1) (its gating linear's bias, a constant) and the transition reads
  // the block's INPUT single: s = s0 + attention(s0) + transition(s0)
  const bool parallel = flag("trunk.dialect.parallelPairformer");
  float* s0 = nullptr;
  if (parallel) { s0 = scratch<float>("st.s0", (size_t)n * Cs); copy(s0, single, (size_t)n * Cs * 4); }
  int heads = metaI(A + ".heads"), d = metaI(A + ".dimension"), Wd = heads * d;
  size_t P = (size_t)n * n;
  const half* xs = sharedNorm(pair);
  half* pln = xs ? (half*)xs : pairLn(pair, P, C, B + ".singlePairLogitsNorm");
  float* raw = scratch<float>("st.raw", P * heads);
  if (xs) {
    Folded f = foldNorm("sp:" + B, Wh(B + ".singlePairLogitsProjection"), C, heads, B + ".singlePairLogitsNorm", nullptr);
    linW(pln, f.w, raw, P, C, heads, 0.f, f.b, 1.f, "pair logits");
  } else lin(pln, B + ".singlePairLogitsProjection", raw, P, C, heads);
  if (extraBias) run1d("af3_add_bias_heads", P * heads, AddBiasHeadsArgs{raw, extraBias, P, (uint)heads, 0});   // (OpenDDE)
  int stride = round8(n);
  half* bias = biasLayout(raw, "st.bias", n, heads, stride, false);
  half* nrm = scratch<half>("st.nrm", (size_t)n * Cs);
  ln(single, nrm, n, Cs, A + ".layerNormScale", A + ".layerNormOffset");
  half* qkvg = scratch<half>("st.qkvg", (size_t)n * 4 * Wd);
  const float* gateBias = !parallel ? nullptr : M.derived<float>("gateOne:" + num(Wd), (size_t)4 * Wd, [&](float* b) {
    std::vector<float> v((size_t)4 * Wd, 0.f);
    for (int k = 3 * Wd; k < 4 * Wd; ++k) v[k] = 1.f;
    upload(b, v.data(), v.size() * 4);
  });
  linW(nrm, qkvgWeight(A, Cs, Wd, false), qkvg, n, Cs, 4 * Wd, gateBias, "single qkvg");
  half* o = scratch<half>("st.o", (size_t)n * Wd);
  Attention at{}; at.qkvg = qkvg; at.out = o; at.n = n; at.heads = heads; at.D = d; at.rows = 1; at.scale = 1.f / sqrtf((float)d);
  at.bias = bias; at.biasStride = stride; at.qBias = W(A + ".qBias");
  if (!m.ones) { at.mask = m.seq; at.maskB = 0; at.maskK = 1; }
  attention(at);
  lin(o, A + ".outputProjection", single, n, Wd, Cs, 1.f);
  if (parallel) transition(s0, n, Cs, B + ".singleTransition", single);
  else transition(single, n, Cs, B + ".singleTransition");
}
void pairformerBlock(float* pair, float* single, const Masks& m, int n, int C, int Cs, const std::string& B, const float* extraBias,
                     const std::string& nextB) {
  if (flag("trunk.dialect.parallelPairformer")) {
    // the single track reads the pair ENTERING the block, kept by the parallel updates
    parallelPairUpdates(pair, m, n, C, B, "oirct");
    // (folded, the pair's xhat is the block input's; otherwise the kept copy)
    singleTrack(single, shared.xhat ? pair : scratch<float>("par.base", (size_t)n * n * C), m, n, C, Cs, B, extraBias);
    shared = {};
    return;
  }
  // (the transition emits the single track's pair norm and the next block's first, where the caller names that block)
  pairUpdates(pair, m, n, C, B, B + ".singlePairLogitsNorm", nextB.empty() ? "" : nextB + ".triangleMultiplicationOutgoing.leftNormInput");
  singleTrack(single, pair, m, n, C, Cs, B, extraBias);
}
