// The pair track - triangle multiplication, grid ("triangle") attention, transition - and the pairformer's single
// track, shared by the trunk's pairformer, its MSA and template stacks and the confidence head (cuda/af3/src/
// pairtrack.cuh and trunk.cuh's singleTrack are the reading). Every attention runs on the core's flash kernel: the
// column direction ACROSS the pair's leading axis (its strides), so nothing is transposed.
#include "af3.h"
#include <cmath>
#include <type_traits>

static int round8(int n) { return (n + 7) / 8 * 8; }
// The pair track's working tensors are shared by name across its stages, which run one after another: a pair-sized half
// buffer for each stage's normalised input (pr.x, and pr.x2 for a norm emitted beside it), one for the grid attention's
// output (pr.h1; the triangle's centre norm too where its product is float), and one large buffer that is a block of
// the grid attention's q|k|v|g or the triangle's planes, product, centre norm and value (pr.big). Held per stage they
// were twice the room: 4.0 against 2.0 GB at 1044 tokens.
static void* bigBuffer(size_t bytes) { return scratchBytes("pr.big", bytes); }

half* biasLayout(const float* raw, const std::string& name, int n, int heads, int stride, bool swap) {
  half* bias = scratch<half>(name, (size_t)heads * n * stride);
  const BiasLayoutArgs a{raw, bias, (uint)n, (uint)stride, (uint)heads, swap ? 1u : 0u, (uint)heads, 0, (float)M_LOG2E, 0};
  run("af3_bias_layout", Grid{(uint32_t)((stride + 31) / 32), (uint32_t)((n + 31) / 32), (uint32_t)heads}, 256, a);
  return bias;
}

// ---------------------------------------------------------------- triangle multiplication
const half* triGateWeight(const std::string& pre, int C) {
  const bool q = triQuartersApply(C);
  return M.derived<half>((q ? "trigateQ:" : "trigate:") + pre, (size_t)C * 4 * C, [&](half* out) {
    run1d("af3_trigate_weight", (C & 1) ? (size_t)C * 4 * C : (size_t)C * C,
          TriGateWArgs{Wh(pre + ".projection"), Wh(pre + ".gate"), out, 0, (uint)C, (uint)C, q ? 1u : 0u, 0});
  });
}
// LN -> one GEMM gating a and b into channel-major padded planes (gemmTriGate; AF3 interleaves a and b by channel) ->
// the planes' batched product (/ n under triangleMulDivideByLength) -> the centre LayerNorm back to rows -> the
// output projection -> the gating linear's GEMM adding the gated product into the pair (gemmGatedAdd)
// the pair's next LayerNorms where the previous update's GEMM already wrote them (its epilogue's lnOut, lnOut2): the
// pair normalised, the norm's name and the buffer. pairLn of the same pair and norm takes its entry; a miss drops both
// (whatever came between may have moved the pair). LOCALFOLD_LN_EMIT=0: none emitted, every norm its own pass
struct Emitted { const void* x = nullptr; std::string norm; half* buf = nullptr; };
static Emitted emitted[6];
static void emit(int k, const void* x, const std::string& norm, half* buf) { emitted[k] = {x, norm, buf}; }
void emitPairLn(int k, const void* x, const std::string& norm, half* buf) { emit(k, x, norm, buf); }
static bool canEmitLn(int C, size_t rows = 1024) {
  return C == 64 || C == 128 || C == 256 || ((C == 384 || C == 512) && rows > 128 && matrixUnits());
}
bool canEmitPairLn(int C, size_t rows) { return canEmitLn(C, rows); }
static half* takeEmitted(const void* x, const std::string& norm) {
  for (auto& e : emitted)
    if (e.x == x && e.norm == norm) { e.x = nullptr; return e.buf; }
  for (auto& e : emitted)
    if (e.x == x) e.x = nullptr;
  return nullptr;
}
half* takePairLn(const void* x, const std::string& norm) { return takeEmitted(x, norm); }
template <class T> static half* pairLnT(const T* pair, size_t P, int C, const std::string& norm, const std::string& nextNorm = "") {
  if (half* b = takeEmitted(pair, norm)) return b;
  const bool msaPair = norm.find(".msaAttention1.pairNorm") != std::string::npos;
  half* xn = scratch<half>(msaPair ? "mnorm.p1" : "pr.x", P * C);
  if (!nextNorm.empty() && (nextNorm == "__xhat__" || hasW(nextNorm + "Scale"))) {
    const bool xh = nextNorm == "__xhat__";
    half* xn2 = scratch<half>(msaPair ? "mnorm.p2" : (xh ? "par.xhat" : "pr.x2"), P * C);
    layerNormDual(pair, xn, W(norm + "Scale"), Wopt(norm + "Offset"),
                  xn2, xh ? nullptr : W(nextNorm + "Scale"), xh ? nullptr : Wopt(nextNorm + "Offset"), P, C);
    emit(0, pair, nextNorm, xn2);
    return xn;
  }
  ln(pair, xn, P, C, norm + "Scale", norm + "Offset");
  return xn;
}
half* pairLn(const float* pair, size_t P, int C, const std::string& norm, const std::string& nextNorm) { return pairLnT(pair, P, C, norm, nextNorm); }
half* pairLn(const half* pair, size_t P, int C, const std::string& norm, const std::string& nextNorm) { return pairLnT(pair, P, C, norm, nextNorm); }
// Y += X W (+ bias): the residual into the pair (or the single) in its own type
static void addLinW(const half* X, const half* w, float* Y, size_t rows, int in, int out, const float* bias, const char* label) {
  linW(X, w, Y, rows, in, out, 1.f, bias, 1.f, label);
}
static void addLinW(const half* X, const half* w, half* Y, size_t rows, int in, int out, const float* bias, const char* label) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = w; g.tw = F16; g.Y = Y; g.ty = F16; g.rows = rows; g.in = in; g.out = out;
  g.beta = 1.f; g.bias = bias; g.label = label;
  gemm(g);
}
template <class T> static void addLin(const half* X, const std::string& w, T* Y, size_t rows, int in, int out, const float* bias = nullptr, const char* label = nullptr) {
  if (lenW(w) != (size_t)in * out) die("%s has %zu elements, not %d x %d", w.c_str(), lenW(w), in, out);
  addLinW(X, Wh(w), Y, rows, in, out, bias, label ? label : w.c_str());
}
template <class T> constexpr DT dtOf() { return std::is_same<T, half>::value ? F16 : F32; }
// chai-1's parallel updates all read one input, so its affine-free LayerNorm is taken once a block (parallelPairUpdates)
// and each update's own scale and offset are folded into the weights that read it (foldNorm): the input and, during the
// first (in-place) update, the pair itself. LOCALFOLD_FOLD_NORM=0 is the control
static struct { const void* src[2] = {nullptr, nullptr}; half* xhat = nullptr; } shared;
static const half* sharedNorm(const void* x) {
  return shared.xhat && x && (x == shared.src[0] || x == shared.src[1]) ? shared.xhat : nullptr;
}
// where an update's GEMM emits the next norm: whichever of pr.x and pr.x2 its own input is not
static half* lnTarget(size_t P, int C, const half* inUse) {
  half* x1 = scratch<half>("pr.x", P * C);
  return inUse == x1 ? scratch<half>("pr.x2", P * C) : x1;
}
template <class T>
static void triangleT(T* pair, const Masks& m, int n, int C, const std::string& pre, bool outgoing, bool divide, T* into,
                      const std::string& nextNorm) {
  size_t P = (size_t)n * n;
  int np = round8(n); size_t plane = (size_t)np * np;
  const half* xs = sharedNorm(pair);
  half* xn = xs ? (half*)xs : pairLn(pair, P, C, pre + ".leftNormInput");
  Folded tg, gl;
  if (xs) {
    tg = foldNorm("tg:" + pre, triGateWeight(pre, C), C, 4 * C, pre + ".leftNormInput", nullptr, true);
    gl = foldNorm("gl:" + pre, Wh(pre + ".gatingLinear"), C, C, pre + ".leftNormInput", nullptr);
  }
  // the planes, their contraction, the centre norm and the gated add's value in pr.big (core's triContract: 0.63 GB at
  // 1044 tokens, where the planes with a float-sized product and pr.h1, pr.h2 beside them were 1.7)
  void* big = bigBuffer(triBytes(C, np));
  half* a = triPlanes(big, n, C, np);
  gemmTriGate(xn, xs ? tg.w : triGateWeight(pre, C), m.pair, a, a + plane, 0, P, C, 2 * plane, n, np, xs ? tg.b : nullptr,
              triQuartersApply(C));
  TriMid mid = triContract(big, n, C, outgoing, divide ? 1.f / n : 1.f, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"),
                           halfProductScale() ? nullptr : scratch<half>("pr.h1", P * C),
                           outgoing ? "triangle out contraction" : "triangle in contraction");
  half* cn = mid.cn; half* outH = mid.outH;
  half* lnOut = !nextNorm.empty() && !into && canEmitLn(C) && hasW(nextNorm + "Scale") ? lnTarget(P, C, xn) : nullptr;
  if (gemmGatedAddDual(xn, xs ? gl.w : Wh(pre + ".gatingLinear"), cn, Wh(pre + ".outputProjection"), into ? into : pair, P, C, C,
                       xs ? gl.b : nullptr, nullptr, outH,
                       "triangle output and gated add", lnOut, lnOut ? W(nextNorm + "Scale") : nullptr,
                       lnOut ? Wopt(nextNorm + "Offset") : nullptr))
    emit(0, pair, nextNorm, lnOut);
}
void triangle(float* pair, const Masks& m, int n, int C, const std::string& pre, bool outgoing, bool divide, float* into,
              const std::string& nextNorm) {
  triangleT(pair, m, n, C, pre, outgoing, divide, into, nextNorm);
}
void triangle(half* pair, const Masks& m, int n, int C, const std::string& pre, bool outgoing, bool divide, half* into,
              const std::string& nextNorm) {
  triangleT(pair, m, n, C, pre, outgoing, divide, into, nextNorm);
}

// ---------------------------------------------------------------- grid attention
// rows of the pair (tr: its columns) attend along themselves, biased by LN(pair) projected per head - the column
// direction's bias read transposed where the dialect swaps it (swapTransposedBias)
template <class T>
static void gridAttentionT(T* pair, const Masks& m, int n, int C, const std::string& pre, bool tr, bool swap, T* into,
                           bool untransposed, const std::string& nextNorm) {
  int heads = metaI(pre + ".heads"), D = metaI(pre + ".dimension"), Wd = heads * D;
  size_t P = (size_t)n * n;
  const half* xs = sharedNorm(pair);
  half* xn = xs ? (half*)xs : pairLn(pair, P, C, pre + ".actNorm");
  Folded fb, fq;
  if (xs) {
    fb = foldNorm("gb:" + pre, Wh(pre + ".pairBiasProjection"), C, heads, pre + ".actNorm", nullptr);
    fq = foldNorm("gq:" + pre, qkvgWeight(pre, C, Wd, true), C, 4 * Wd, pre + ".actNorm", qkvgBias(pre, Wd), true);
  }
  // (rows padded to a whole 64-key tile with -inf, so the attention's last key tile needs no per-element mask)
  int stride = (n + 63) / 64 * 64;
  // (the pair bias projected and laid out per head in one kernel; LOCALFOLD_PAIR_BIAS=0 the control - a GEMM, then the layout)
  static const bool fusedPairBias = !getenv("LOCALFOLD_PAIR_BIAS") || atoi(getenv("LOCALFOLD_PAIR_BIAS")) != 0;
  half* bias = nullptr;
  if (fusedPairBias && (C == 64 || (C % 128 == 0 && C <= 512)) && (heads == 4 || heads == 8 || heads == 12 || heads == 16)) {
    bias = scratch<half>("grid.bias", (size_t)heads * n * stride);
    run("af3_pair_bias", Grid{(uint32_t)((n + 7) / 8), (uint32_t)((n + 7) / 8), 1}, 256,
        PairBiasArgs{xn, xs ? fb.w : Wh(pre + ".pairBiasProjection"), xs ? fb.b : nullptr, nullptr, bias,
                     (uint)n, (uint)C, (uint)heads, (tr && swap) ? 1u : 0u, (uint)stride, 0});
  } else {
    float* raw = scratch<float>("grid.raw", P * heads);
    if (xs) linW(xn, fb.w, raw, P, C, heads, 0.f, fb.b, 1.f, "grid bias projection");
    else lin(xn, pre + ".pairBiasProjection", raw, P, C, heads);
    bias = biasLayout(raw, "grid.bias", n, heads, stride, tr && swap);
  }
  // q|k|v|g a block of rows at a time (tr: of columns), 256 MB of them: the whole pair's was 1.1 GB at 1044 tokens, the
  // largest buffer of a fold. A row's attention reads its own n positions and nothing else, so the blocks are the
  // same arithmetic; below ~500 tokens one block is the whole pair
  const size_t perRow = (size_t)n * 4 * Wd * 2;
  const int rb = qkvgBlockRows(perRow, n);
  half* qkvg = (half*)bigBuffer((size_t)rb * perRow);
  const half* wq = xs ? fq.w : qkvgWeight(pre, C, Wd, true);
  const float* bq = xs ? fq.b : qkvgBias(pre, Wd);
  half* o = scratch<half>("pr.h1", P * Wd);
  for (int r0 = 0; r0 < n; r0 += rb) {
    const int r = std::min(rb, n - r0);
    // rows r0.. of xn are contiguous (and every column, the whole pair); columns r0.. are r rows at each of the n
    // positions, a GEMM batched over them
    Gemm g{}; g.tx = F16; g.W = wq; g.tw = F16; g.Y = qkvg; g.ty = F16; g.in = C; g.out = 4 * Wd; g.bias = bq; g.accFloat = true;
    g.label = "grid qkvg";
    if (!tr || r == n) { g.X = xn + (size_t)r0 * n * C; g.rows = (size_t)r * n; }
    else { g.X = xn + (size_t)r0 * C; g.rows = r; g.ldx = C; g.sx = (int64_t)n * C; g.sy = (int64_t)r * 4 * Wd; g.batch = n; }
    gemm(g);
    Attention at{}; at.qkvg = qkvg; at.n = n; at.heads = heads; at.D = D; at.rows = r; at.scale = 1.f / sqrtf((float)D);
    at.bias = bias; at.biasStride = stride; at.biasPadInf = true; at.r0 = r0;
    if (!m.ones) { at.mask = m.pair; at.maskB = tr ? 1 : n; at.maskK = tr ? n : 1; }
    if (!tr) at.out = o + (size_t)r0 * n * Wd;
    else {
      at.rowStride = 4 * Wd; at.posStride = (int64_t)r * 4 * Wd;
      // (untransposed: the column direction's output kept at (r, j), chai's - its ending-node update at (i, j) is AF3's at (j, i))
      if (untransposed) at.out = o + (size_t)r0 * n * Wd;
      else { at.out = o + (size_t)r0 * Wd; at.outRowStride = Wd; at.outPosStride = (int64_t)n * Wd; }
    }
    attention(at);
  }
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
    addLinW(o, w, into ? into : pair, P, Wd, C, Wopt(pre + ".outputProjectionBias"), "grid output");
  } else if (!nextNorm.empty() && !into && canEmitLn(C) && hasW(nextNorm + "Scale") && lenW(pre + ".outputProjection") == (size_t)Wd * C) {
    half* lnOut = lnTarget(P, C, xn);
    Gemm g{}; g.X = o; g.tx = F16; g.W = Wh(pre + ".outputProjection"); g.tw = F16; g.Y = pair; g.ty = dtOf<T>(); g.rows = P; g.in = Wd; g.out = C;
    g.beta = 1.f; g.bias = Wopt(pre + ".outputProjectionBias"); g.label = "grid output, next norm";
    g.lnOut = lnOut; g.lnScale = W(nextNorm + "Scale"); g.lnOffset = Wopt(nextNorm + "Offset");
    if (gemm(g)) emit(0, pair, nextNorm, lnOut);
  } else addLin(o, pre + ".outputProjection", into ? into : pair, P, Wd, C, Wopt(pre + ".outputProjectionBias"), "grid output");
}
void gridAttention(float* pair, const Masks& m, int n, int C, const std::string& pre, bool tr, bool swap, float* into,
                   bool untransposed, const std::string& nextNorm) {
  gridAttentionT(pair, m, n, C, pre, tr, swap, into, untransposed, nextNorm);
}
void gridAttention(half* pair, const Masks& m, int n, int C, const std::string& pre, bool tr, bool swap, half* into,
                   bool untransposed, const std::string& nextNorm) {
  gridAttentionT(pair, m, n, C, pre, tr, swap, into, untransposed, nextNorm);
}

// ---------------------------------------------------------------- transition
// chai's transitions on the shared norm: the affine folded into the weights (LOCALFOLD_FOLD_TRANSITION=0, the control: an
// affine pass over the rows first)
static bool foldTransition() {
  static const bool on = !getenv("LOCALFOLD_FOLD_TRANSITION") || atoi(getenv("LOCALFOLD_FOLD_TRANSITION")) != 0;
  return on;
}
// LN -> SwiGLU in the first GEMM's epilogue -> the second GEMM adding into x; in row chunks
template <class T>
static void transitionT(T* x, size_t rows, int C, const std::string& pre, T* into, const std::string& next1,
                        const std::string& next2, const char* scratchPrefix) {
  size_t w1 = lenW(pre + ".transition1");
  if (w1 % (2 * (size_t)C)) die("%s.transition1 has %zu elements, not %d x 2I", pre.c_str(), w1, C);
  int I = (int)(w1 / (2 * (size_t)C));
  const bool padSt = rows <= 256 && (rows & (rows <= 128 ? 15 : 31)) && !strcmp(scratchPrefix, "st.") && !sharedNorm(x);
  size_t rp = padSt ? (rows <= 128 ? ((rows + 15) & ~(size_t)15) : ((rows + 31) & ~(size_t)31)) : rows;
  size_t chunk = std::min(rows, std::max<size_t>(64, ((size_t)128 << 20) / (3 * (size_t)I)));
  half* xn = scratch<half>(std::string(scratchPrefix) + "xn", std::max(chunk, rp) * C);
  half* g = scratch<half>(std::string(scratchPrefix) + "g", std::max(chunk, rp) * I);
  if (padSt) {
    static const half* zeroedXn = nullptr; static size_t zeroedRows = 0;
    if (xn != zeroedXn || rows != zeroedRows) {
      fill(xn + rows * C, 0, (rp - rows) * C * sizeof(half));
      zeroedXn = xn; zeroedRows = rows;
    }
  }
  const half* wp = swigluPairs(pre + ".transition1", C, I);
  // the input norm the previous update emitted, whole, read a chunk at a time - and the next norms emitted a chunk at a
  // time, each chunk's rows of pr.x / pr.x2 written once the chunk's SwiGLU has read them (a long pair runs in chunks:
  // both used to be a LayerNorm pass of their own past ~300 tokens)
  half* inAll = takeEmitted(x, pre + ".inputLayerNorm");
  const bool isSt = !strcmp(scratchPrefix, "st.");
  const bool isMsa = !strcmp(scratchPrefix, "msatr.");
  const bool msaNext = !isSt && !isMsa && next1.find(".msaAttention1.pairNorm") != std::string::npos;
  const bool xh1 = next1 == "__xhat__";
  const bool emitNext = !next1.empty() && !into && canEmitLn(C, rows) && (xh1 || hasW(next1 + "Scale")) && (next2.empty() || hasW(next2 + "Scale"));
  half* ln1Buf = emitNext ? (isSt ? scratch<half>("st.nrm", std::max(rows, rp) * C)
                                  : isMsa ? scratch<half>("mnorm.m1", rows * C)
                                          : msaNext ? scratch<half>("mnorm.p1", rows * C)
                                                    : ((xh1 && shared.xhat != scratch<half>("par.xhat", rows * C)) ? scratch<half>("par.xhat", rows * C)
                                                                                                                   : scratch<half>("pr.x", rows * C))) : nullptr;
  half* ln2Buf = (emitNext && !next2.empty()) ? (isSt ? scratch<half>("st.xn", std::max(rows, rp) * C)
                                                      : isMsa ? scratch<half>("mnorm.m2", rows * C)
                                                              : scratch<half>("pr.x2", rows * C)) : nullptr;
  bool emittedAll = emitNext;
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    size_t rg = (padSt && r == rows) ? rp : r;
    half* in = inAll ? inAll + r0 * C : nullptr;
    const half* xs = in ? nullptr : sharedNorm(x);
    if (xs && foldTransition()) {     // (chai's shared input norm: this norm's affine folded into the SwiGLU's weights)
      Folded f = foldNorm("tr:" + pre, wp, C, 2 * I, pre + ".inputLayerNorm", nullptr, true);
      gemmSwiglu(xs + r0 * C, f.w, g, r, C, I, f.b);
    } else {
      if (xs) {      // (the control: the affine on its rows)
        run1d("af3_affine_h", r * C, AffineHArgs{xs + r0 * C, xn, W(pre + ".inputLayerNormScale"), Wopt(pre + ".inputLayerNormOffset"),
                                                (u64)(r * C), (uint)C, 0});
        in = xn;
      }
      if (!in) { ln(x + r0 * C, xn, r, C, pre + ".inputLayerNormScale", pre + ".inputLayerNormOffset"); in = xn; }
      gemmSwiglu(in, wp, g, rg, C, I);
    }
    // the pair's next norms (next1 into pr.x, next2 into pr.x2 - both free once the SwiGLU has read its input)
    if (emittedAll) {
      Gemm G{}; G.X = g; G.tx = F16; G.W = Wh(pre + ".transition2"); G.tw = F16; G.Y = x + r0 * C; G.ty = dtOf<T>(); G.rows = r;
      G.in = I; G.out = C; G.beta = 1.f; G.label = "transition2, next norms";
      G.lnOut = ln1Buf + r0 * C; G.lnScale = xh1 ? nullptr : W(next1 + "Scale"); G.lnOffset = xh1 ? nullptr : Wopt(next1 + "Offset");
      if (ln2Buf) {
        G.lnOut2 = ln2Buf + r0 * C; G.lnScale2 = W(next2 + "Scale"); G.lnOffset2 = Wopt(next2 + "Offset");
      }
      if (lenW(pre + ".transition2") != (size_t)I * C) die("%s.transition2 has %zu elements", pre.c_str(), lenW(pre + ".transition2"));
      if (gemm(G)) continue;
      if (r0) die("%s: the next norms emitted for some chunks and not others", pre.c_str());
      emittedAll = false;       // (the GEMM took no emission - every chunk falls back alike)
      continue;
    }
    addLin(g, pre + ".transition2", (into ? into : x) + r0 * C, rg, I, C, nullptr, "transition2");
  }
  if (emittedAll) {
    int s0 = isSt ? 2 : isMsa ? 4 : 0;
    emit(s0, x, next1, ln1Buf);
    if (ln2Buf) emit(s0 + 1, x, next2, ln2Buf);
  }
}
void transition(float* x, size_t rows, int C, const std::string& pre, float* into, const std::string& next1,
                const std::string& next2, const char* scratchPrefix) {
  transitionT(x, rows, C, pre, into, next1, next2, scratchPrefix);
}
void transition(half* x, size_t rows, int C, const std::string& pre, half* into, const std::string& next1,
                const std::string& next2, const char* scratchPrefix) {
  transitionT(x, rows, C, pre, into, next1, next2, scratchPrefix);
}

// Mol-1's single transition in float (dialect singleTransitionF32): its single reaches 3e5 on 6MRR and the gated
// activation 6.6e4, past half's 65504 - gemmSwiglu stores it in half, an infinity on the second pass and NaN after.
// The LN'd input stays half; the product, the gate and the second GEMM are float
static void transitionF32(float* x, size_t rows, int C, const std::string& pre) {
  int I = (int)(lenW(pre + ".transition1") / (2 * (size_t)C));
  if (I % 8) die("%s: a hidden width of %d is not a multiple of 8", pre.c_str(), I);
  half* xn = scratch<half>("st32.xn", rows * C);
  float* ab = scratch<float>("st32.ab", rows * 2 * I);
  float* g = scratch<float>("st32.g", rows * I);
  ln(x, xn, rows, C, pre + ".inputLayerNormScale", pre + ".inputLayerNormOffset");
  linW(xn, swigluPairs(pre + ".transition1", C, I), ab, rows, C, 2 * I, 0.f, nullptr, 1.f, "single transition1, float");
  run1d("af3_swiglu_f32", rows * I, SwigluF32Args{ab, g, (uint)(rows * I), (uint)I});
  if (lenW(pre + ".transition2") != (size_t)I * C) die("%s.transition2 has %zu elements", pre.c_str(), lenW(pre + ".transition2"));
  Gemm G{}; G.X = g; G.tx = F32; G.W = Wh(pre + ".transition2"); G.tw = F16; G.Y = x; G.rows = rows; G.in = I; G.out = C;
  G.beta = 1.f; G.label = "single transition2, float";
  gemm(G);
}

// the five pair updates of a pairformer / MSA / template block, in AF3's order
template <class T>
static void pairUpdatesT(T* pair, const Masks& m, int n, int C, const std::string& pre, const std::string& next1,
                         const std::string& next2) {
  bool swap = flag("trunk.dialect.swapTransposedBias"), divide = flag("trunk.dialect.triangleMulDivideByLength");
  triangle(pair, m, n, C, pre + ".triangleMultiplicationOutgoing", true, divide, nullptr, pre + ".triangleMultiplicationIncoming.leftNormInput");
  // (Mol-1's PairMixer blocks carry no triangle attention: the incoming triangle emits the transition's norm)
  const bool grid = M.has(pre + ".pairAttention1.heads");
  triangle(pair, m, n, C, pre + ".triangleMultiplicationIncoming", false, divide, nullptr,
           grid ? pre + ".pairAttention1.actNorm" : pre + ".pairTransition.inputLayerNorm");
  if (grid) {
    gridAttention(pair, m, n, C, pre + ".pairAttention1", false, swap, nullptr, false, pre + ".pairAttention2.actNorm");
    gridAttention(pair, m, n, C, pre + ".pairAttention2", true, swap, nullptr, false, pre + ".pairTransition.inputLayerNorm");
  }
  transition(pair, (size_t)n * n, C, pre + ".pairTransition", nullptr, next1, next2);
}
void pairUpdates(float* pair, const Masks& m, int n, int C, const std::string& pre, const std::string& next1,
                 const std::string& next2) {
  pairUpdatesT(pair, m, n, C, pre, next1, next2);
}
void pairUpdates(half* pair, const Masks& m, int n, int C, const std::string& pre, const std::string& next1,
                 const std::string& next2) {
  pairUpdatesT(pair, m, n, C, pre, next1, next2);
}

// chai-1's parallel updates: the stage's input kept (par.base); the first update runs on the pair itself (it IS the input
// then), every later one reads the kept input and adds its residual into the pair; the ending-node attention's residual
// untransposed
template <class T>
static void parallelPairUpdatesT(T* pair, const Masks& m, int n, int C, const std::string& pre, const char* which,
                                 bool emitNextXhat) {
  bool swap = flag("trunk.dialect.swapTransposedBias"), divide = flag("trunk.dialect.triangleMulDivideByLength");
  size_t P = (size_t)n * n;
  static const bool fold = !getenv("LOCALFOLD_FOLD_NORM") || atoi(getenv("LOCALFOLD_FOLD_NORM")) != 0;
  shared = {};
  // (folded, every update reads the input only through its norm - so no kept copy: they all take the pair, whose
  // xhat is the block's input whatever the updates before have added)
  T* base = fold ? pair : scratch<T>("par.base", P * C);
  if (fold) {
    if (half* em = takeEmitted(pair, "__xhat__")) shared.xhat = em;
    else {
      shared.xhat = scratch<half>("par.xhat", P * C);
      layerNorm(pair, shared.xhat, P, C, nullptr, nullptr);
    }
    shared.src[0] = pair;
  } else copy(base, pair, P * C * sizeof(T));
  bool first = true;
  for (const char* u = which; *u; ++u) {
    bool inPlace = fold || (first && *u != 'c');
    first = false;
    T* in = inPlace ? pair : base;
    T* into = inPlace ? nullptr : pair;
    const bool last = u[1] == '\0';
    switch (*u) {
      case 'o': triangle(in, m, n, C, pre + ".triangleMultiplicationOutgoing", true, divide, into); break;
      case 'i': triangle(in, m, n, C, pre + ".triangleMultiplicationIncoming", false, divide, into); break;
      case 'r': gridAttention(in, m, n, C, pre + ".pairAttention1", false, swap, into); break;
      case 'c': gridAttention(in, m, n, C, pre + ".pairAttention2", true, swap, into, true); break;
      case 't': transition(in, P, C, pre + ".pairTransition", into, (fold && last && emitNextXhat) ? "__xhat__" : ""); break;
    }
  }
}
void parallelPairUpdates(float* pair, const Masks& m, int n, int C, const std::string& pre, const char* which,
                         bool emitNextXhat) {
  parallelPairUpdatesT(pair, m, n, C, pre, which, emitNextXhat);
}
void parallelPairUpdates(half* pair, const Masks& m, int n, int C, const std::string& pre, const char* which,
                         bool emitNextXhat) {
  parallelPairUpdatesT(pair, m, n, C, pre, which, emitNextXhat);
}

// ---------------------------------------------------------------- the single track
// attention over the tokens biased by the pair (LN, a projection to the heads), then the transition
template <class T>
static void singleTrackT(float* single, const T* pair, const Masks& m, int n, int C, int Cs, const std::string& B,
                         const float* extraBias, const std::string& nextPairNorm = "", const std::string& nextB = "") {
  const std::string A = B + ".singleAttention";
  // chai-1's parallel block: the gate is sigmoid(g + 1) (its gating linear's bias, a constant) and the transition reads
  // the block's INPUT single: s = s0 + attention(s0) + transition(s0)
  const bool parallel = flag("trunk.dialect.parallelPairformer");
  const size_t np = n <= 128 ? (size_t)((n + 15) & ~15) : (n <= 256 ? (size_t)((n + 31) & ~31) : (size_t)n);
  int heads = metaI(A + ".heads"), d = metaI(A + ".dimension"), Wd = heads * d;
  size_t P = (size_t)n * n;
  const half* xs = sharedNorm(pair);
  half* pln = xs ? (half*)xs : pairLn(pair, P, C, B + ".singlePairLogitsNorm", nextPairNorm);
  int stride = (n + 63) / 64 * 64;
  // (the pair bias projected and laid out per head in one kernel; LOCALFOLD_PAIR_BIAS=0 the control - a GEMM, then the layout)
  static const bool fusedPairBias = !getenv("LOCALFOLD_PAIR_BIAS") || atoi(getenv("LOCALFOLD_PAIR_BIAS")) != 0;
  half* bias = nullptr;
  if (fusedPairBias && (C == 64 || (C % 128 == 0 && C <= 512)) && (heads == 4 || heads == 8 || heads == 12 || heads == 16)) {
    Folded f;
    if (xs) f = foldNorm("sp:" + B, Wh(B + ".singlePairLogitsProjection"), C, heads, B + ".singlePairLogitsNorm", nullptr);
    bias = scratch<half>("st.bias", (size_t)heads * n * stride);
    run("af3_pair_bias", Grid{(uint32_t)((n + 7) / 8), (uint32_t)((n + 7) / 8), 1}, 256,
        PairBiasArgs{pln, xs ? f.w : Wh(B + ".singlePairLogitsProjection"), xs ? f.b : nullptr, extraBias, bias,
                     (uint)n, (uint)C, (uint)heads, 0u, (uint)stride, 0});
  } else {
    float* raw = scratch<float>("st.raw", P * heads);
    if (xs) {
      Folded f = foldNorm("sp:" + B, Wh(B + ".singlePairLogitsProjection"), C, heads, B + ".singlePairLogitsNorm", nullptr);
      linW(pln, f.w, raw, P, C, heads, 0.f, f.b, 1.f, "pair logits");
    } else lin(pln, B + ".singlePairLogitsProjection", raw, P, C, heads);
    if (extraBias) run1d("af3_add_bias_heads", P * heads, AddBiasHeadsArgs{raw, extraBias, P, (uint)heads, 0});   // (OpenDDE)
    bias = biasLayout(raw, "st.bias", n, heads, stride, false);
  }
  half* nrm = scratch<half>("st.nrm", np * Cs);
  half* qkvg = scratch<half>("st.qkvg", np * 4 * Wd);
  half* o = scratch<half>("st.o", np * Wd);
  if (np > (size_t)n) {
    static const half *zeroedNrm = nullptr, *zeroedO = nullptr; static int zeroedN = 0;
    if (nrm != zeroedNrm || o != zeroedO || n != zeroedN) {
      fill(nrm + (size_t)n * Cs, 0, (np - n) * Cs * sizeof(half));
      fill(o + (size_t)n * Wd, 0, (np - n) * Wd * sizeof(half));
      zeroedNrm = nrm; zeroedO = o; zeroedN = n;
    }
  }
  const std::string tn = B + ".singleTransition.inputLayerNorm";
  if (half* b = takeEmitted(single, A + ".layerNorm")) {
    nrm = b;
  } else if (parallel) {
    half* txn = scratch<half>("st.xn", np * Cs);
    layerNormDual(single, nrm, W(A + ".layerNormScale"), Wopt(A + ".layerNormOffset"),
                  txn, W(tn + "Scale"), Wopt(tn + "Offset"), n, Cs);
    emit(3, single, tn, txn);
  } else {
    ln(single, nrm, n, Cs, A + ".layerNormScale", A + ".layerNormOffset");
  }
  const float* gateBias = !parallel ? nullptr : M.derived<float>("gateOne:" + num(Wd), (size_t)4 * Wd, [&](float* b) {
    std::vector<float> v((size_t)4 * Wd, 0.f);
    for (int k = 3 * Wd; k < 4 * Wd; ++k) v[k] = 1.f;
    upload(b, v.data(), v.size() * 4);
  });
  linW(nrm, qkvgWeight(A, Cs, Wd, false), qkvg, np, Cs, 4 * Wd, gateBias, "single qkvg");
  Attention at{}; at.qkvg = qkvg; at.out = o; at.n = n; at.heads = heads; at.D = d; at.rows = 1; at.scale = 1.f / sqrtf((float)d);
  at.bias = bias; at.biasStride = stride; at.biasPadInf = true; at.qBias = W(A + ".qBias");
  if (!m.ones) { at.mask = m.seq; at.maskB = 0; at.maskK = 1; }
  attention(at);
  if (!parallel && canEmitLn(Cs, n) && hasW(tn + "Scale") && lenW(A + ".outputProjection") == (size_t)Wd * Cs) {
    half* txn = scratch<half>("st.xn", np * Cs);
    Gemm g{}; g.X = o; g.tx = F16; g.W = Wh(A + ".outputProjection"); g.tw = F16; g.Y = single; g.rows = n;
    g.in = Wd; g.out = Cs; g.beta = 1.f; g.label = "single output, next norm";
    g.lnOut = txn; g.lnScale = W(tn + "Scale"); g.lnOffset = Wopt(tn + "Offset");
    if (gemm(g)) emit(2, single, tn, txn);
  } else lin(o, A + ".outputProjection", single, np, Wd, Cs, 1.f);
  const std::string nextSn1 = nextB.empty() ? "" : nextB + ".singleAttention.layerNorm";
  const std::string nextSn2 = (parallel && !nextB.empty()) ? nextB + ".singleTransition.inputLayerNorm" : "";
  if (flag("trunk.dialect.singleTransitionF32")) transitionF32(single, n, Cs, B + ".singleTransition");
  else transition(single, n, Cs, B + ".singleTransition", nullptr, nextSn1, nextSn2, "st.");
}
void singleTrack(float* single, const float* pair, const Masks& m, int n, int C, int Cs, const std::string& B,
                 const float* extraBias) {
  singleTrackT(single, pair, m, n, C, Cs, B, extraBias);
}
void singleTrack(float* single, const half* pair, const Masks& m, int n, int C, int Cs, const std::string& B,
                 const float* extraBias) {
  singleTrackT(single, pair, m, n, C, Cs, B, extraBias);
}
template <class T>
static void pairformerBlockT(T* pair, float* single, const Masks& m, int n, int C, int Cs, const std::string& B, const float* extraBias,
                             const std::string& nextB) {
  if (flag("trunk.dialect.parallelPairformer")) {
    // the single track reads the pair ENTERING the block, kept by the parallel updates
    parallelPairUpdates(pair, m, n, C, B, "oirct", !nextB.empty());
    // (folded, the pair's xhat is the block input's; otherwise the kept copy)
    singleTrackT(single, shared.xhat ? pair : scratch<T>("par.base", (size_t)n * n * C), m, n, C, Cs, B, extraBias, "", nextB);
    shared = {};
    return;
  }
  // (the transition emits the single track's pair norm and the next block's first, where the caller names that block)
  const std::string nextNorm = nextB.empty() ? "" : nextB + ".triangleMultiplicationOutgoing.leftNormInput";
  pairUpdates(pair, m, n, C, B, B + ".singlePairLogitsNorm", nextNorm);
  singleTrackT(single, pair, m, n, C, Cs, B, extraBias, nextNorm, nextB);
}
void pairformerBlock(float* pair, float* single, const Masks& m, int n, int C, int Cs, const std::string& B, const float* extraBias,
                     const std::string& nextB) {
  pairformerBlockT(pair, single, m, n, C, Cs, B, extraBias, nextB);
}
void pairformerBlock(half* pair, float* single, const Masks& m, int n, int C, int Cs, const std::string& B, const float* extraBias,
                     const std::string& nextB) {
  pairformerBlockT(pair, single, m, n, C, Cs, B, extraBias, nextB);
}
