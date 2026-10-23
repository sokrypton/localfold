// z_init, the recycle and the 24-block pair trunk, and the contact map (cuda/ef2/src/trunk.cuh is the reading):
//
//   z_init = z_init_1(s)[i] + z_init_2(s)[j] + rel_pos + token_bonds + lm_z
//   z = 0;  num_loops + 1 times:  z = z_init + Linear(LN(z));  z = trunk(z)
//   block: z += triangle out(z); z += triangle in(z); z += transition(z)       (no biases)
// A triangle: LN -> one GEMM gating a and b into channel-major planes (gemmTriGate) -> the planes' product, a
// batched GEMM -> the centre LayerNorm back to rows -> the output projection -> the gate's GEMM adding its gated
// product into the pair (gemmGatedAdd). A transition: LN -> SwiGLU in the first GEMM's epilogue -> the second GEMM
// adding into the pair. The pair is float32 (a half copy through the blocks where the matrix units run); everything
// between is float16.
#include "ef2.h"
#include <algorithm>
#include <random>
#include <type_traits>

void zInit(int T, int C, const float* sInputs, int Si, const float* lmZ, float* z) {
  float* rows = scratch<float>("zi.rows", (size_t)T * C); float* cols = scratch<float>("zi.cols", (size_t)T * C);
  lin(sInputs, "f/featuriser/zInit1", rows, T, Si, C);
  lin(sInputs, "f/featuriser/zInit2", cols, T, Si, C);
  run("ef2_zinit", grid1d((size_t)T * T, 1), 256,
      ZInitArgs{rows, cols, relIdx(), In("token_bonds"), F("featuriser/tokenBonds"), lmZ, z, (uint)T, (uint)C});
}

// the triangle's projection and gate as one weight for gemmTriGate: channel c's (pa ga pb gb) in blocks of 8 - ESMFold2
// interleaves a and b by channel (column 2c is a's, 2c + 1 b's)
static const half* triGate(const std::string& Tn, int C) {
  const bool q = triQuartersApply(C);
  return M.derived<half>((q ? "trigateQ:" : "trigate:") + Tn, (size_t)C * 4 * C, [&](half* out) {
    run1d("ef2_trigate_weight", (size_t)C * C, TriGateWArgs{Fh(Tn + "projection"), Fh(Tn + "gate"), out, (uint)C, (uint)C, q ? 1u : 0u, 0});
  });
}
// the pair's next LayerNorm where the previous update's GEMM already wrote it (Gemm::lnOut, gemmGatedAdd's): the pair,
// the norm's name (its Scale / Offset prefix) and the buffer, taken by the next update of the same; a miss drops it.
// LOCALFOLD_LN_EMIT=0: none emitted
static struct { const void* x = nullptr; std::string norm; half* buf = nullptr; } emitted;
void emitTrunkLn(const void* x, const std::string& norm, half* buf) { emitted = {x, norm, buf}; }
static half* takeEmitted(const void* x, const std::string& norm) {
  const bool hit = emitted.x == x && emitted.norm == norm;
  emitted.x = nullptr;
  return hit ? emitted.buf : nullptr;
}
// where an update's GEMM emits the next norm: over its own input norm, in place. An emitting GEMM covers every output
// column of a row block in one tile (the emission's own condition, core.mm's lnEmit), so the block's input rows are
// read by its matmul before the epilogue - behind a threadgroup barrier - writes the norm over them, and no other tile
// reads those rows; the transition's chunks overwrite rows their SwiGLU has already read. A second pair-sized buffer
// (eln.b) beside the triangle's planes was the peak: 558 MB at 1044 residues
static half* lnTarget(size_t P, int C, const half* inUse) {
  return inUse ? (half*)inUse : scratch<half>("ftri.xn", P * C);
}
// (the pair float, or half where foldingTrunk keeps a half copy for the blocks)
template <class T>
static void triangle(T* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing, const std::string& next) {
  size_t P = (size_t)L * L;
  int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
  half* xn = takeEmitted(pair, Tn + "leftNormInput");
  if (!xn) {
    xn = scratch<half>("ftri.xn", P * C);
    layerNorm(pair, xn, P, C, F(Tn + "leftNormInputScale"), F(Tn + "leftNormInputOffset"));
  }
  // the planes, their contraction, the centre norm and the gated add's value in one buffer (core's triContract: 1.26 GB
  // at 1044 residues, against 1.69 for the planes and the product apart)
  void* big = scratchBytes("ftri.big", triBytes(C, Lp));
  half* a = triPlanes(big, L, C, Lp);
  gemmTriGate(xn, triGate(Tn, C), mask, a, a + plane, 0, P, C, 2 * plane, L, Lp, nullptr, triQuartersApply(C));
  TriMid mid = triContract(big, L, C, outgoing, 1.f, F(Tn + "centerNormScale"), F(Tn + "centerNormOffset"),
                           halfProductScale() ? nullptr : scratch<half>("ftri.cn", P * C),
                           outgoing ? "triangle out contraction" : "triangle in contraction");
  half* cn = mid.cn; half* outH = mid.outH;
  half* lnOut = next.empty() ? nullptr : lnTarget(P, C, xn);
  if (gemmGatedAddDual(xn, Fh(Tn + "gatingLinear"), cn, Fh(Tn + "outputProjection"), pair, P, C, C, nullptr, nullptr, outH,
                       "triangle output and gated add", lnOut, lnOut ? F(next + "Scale") : nullptr, lnOut ? F(next + "Offset") : nullptr))
    emitted = {pair, next, lnOut};
}
template <class T>
static void transition(T* pair, size_t P, int C, const std::string& Tn, const std::string& next) {
  int I = dimOf("f/" + Tn + "transition2", 0);
  size_t chunk = std::max<size_t>(64, ((size_t)128 << 20) / (3 * (size_t)I));
  size_t rows = std::min(P, chunk);
  // (a row's norm is its own: the emitted input and the emitted output both go a chunk of rows at a time)
  half* in = takeEmitted(pair, Tn + "inputLayerNorm");
  half* lnOut = next.empty() ? nullptr : lnTarget(P, C, in);
  half* xn = in ? nullptr : scratch<half>("ftr.xn", rows * C);
  half* g = scratch<half>("ftr.g", rows * I);
  const half* w1 = swigluPairsInPlace("f/" + Tn + "transition1", C, I);
  bool all = lnOut != nullptr;
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    const half* x = in ? in + r0 * C : xn;
    if (!in) layerNorm(pair + r0 * C, xn, r, C, F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"));
    gemmSwiglu(x, w1, g, r, C, I);
    Gemm G{}; G.X = g; G.tx = F16; G.W = Fh(Tn + "transition2"); G.tw = F16; G.Y = pair + r0 * C;
    G.ty = std::is_same<T, half>::value ? F16 : F32; G.rows = r; G.in = I; G.out = C;
    G.beta = 1.f; G.label = lnOut ? "transition2, next norm" : "transition2";
    if (lnOut) {
      const bool slash = next.back() == '/';
      G.lnOut = lnOut + r0 * C;
      G.lnScale = F(next + (slash ? "scale" : "Scale"));
      G.lnOffset = F(next + (slash ? "offset" : "Offset"));
    }
    all = gemm(G) && all;
  }
  if (all) emitted = {pair, next, lnOut};
}
template <class T>
static void trunkBlockT(T* pair, const float* mask, int L, int C, const std::string& prefix, int b, bool nextFollows,
                        const std::string& lastNext = "") {
  std::string B = prefix + "/" + std::to_string(b) + "/";
  // (each update's last GEMM emitting the next one's norm - the next block's first where it follows directly)
  triangle(pair, mask, L, C, B + "triangleMultiplicationOutgoing/", true, B + "triangleMultiplicationIncoming/leftNormInput");
  triangle(pair, mask, L, C, B + "triangleMultiplicationIncoming/", false, B + "pairTransition/inputLayerNorm");
  transition(pair, (size_t)L * L, C, B + "pairTransition/",
             nextFollows ? prefix + "/" + std::to_string(b + 1) + "/triangleMultiplicationOutgoing/leftNormInput" : lastNext);
}
void trunkBlock(float* pair, const float* mask, int L, int C, const std::string& prefix, int b, bool nextFollows,
                const std::string& lastNext) {
  trunkBlockT(pair, mask, L, C, prefix, b, nextFollows, lastNext);
}
void trunkBlock(half* pair, const float* mask, int L, int C, const std::string& prefix, int b, bool nextFollows,
                const std::string& lastNext) {
  trunkBlockT(pair, mask, L, C, prefix, b, nextFollows, lastNext);
}
void prepareTrunkWeights() {
  const int C = (int)M.meta("meta/pairChannels", 0);
  if (!C) return;
  for (const char* pfx : {"blocks", "lmEncoder/blocks", "coda/blocks", "confidence/blocks", "msaEncoder/blocks"}) {
    for (int b = 0; M.has(std::string("f/") + pfx + "/" + std::to_string(b) + "/triangleMultiplicationOutgoing/projection"); ++b) {
      std::string B = std::string(pfx) + "/" + std::to_string(b) + "/";
      triGate(B + "triangleMultiplicationOutgoing/", C);
      triGate(B + "triangleMultiplicationIncoming/", C);
    }
  }
}
bool halfPairBlocks(int C) {
  static const bool halfWanted = !getenv("LOCALFOLD_HALF_PAIR") || atoi(getenv("LOCALFOLD_HALF_PAIR")) != 0;
  return halfWanted && C % 8 == 0;
}

void foldingTrunk(int T, int C, const float* zi, float*& z, int loops, const float* lmZ, uint64_t seed) {
  size_t P = (size_t)T * T;
  float* mask = scratch<float>("trunk.mask", P);
  run1d("ef2_fill", P, FillFArgs{mask, P, 1.f, 0});          // every token is real
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));
  if (M.has("f/recycle/decay")) {
    half* xn = scratch<half>("trunk.xn", chunk * C);
    // the released models (ESMFold2, ESMFold2-Fast), cuda/ef2's trunk.cuh: z starts as noise; each pass refines a
    // dropped-out copy of the language model's pair through its own pair-only blocks, injects z_init plus that,
    // LayerNorm'd and projected, into a per-channel decay of z, and runs the trunk; after the last pass the readout and
    // the coda (biohub's modeling_esmfold2.py _run_one_loop and forward). The float pair throughout.
    // EF2_DETERMINISTIC=1: no dropout and a zero initial state (af3-any-model's reading, for a tensor comparison)
    if (!lmZ) die("the parcae recycle needs the language model's pair");
    static const bool deterministic = getenv("EF2_DETERMINISTIC") != nullptr;
    const float p = deterministic ? 0.f : (float)M.meta("meta/lmDropout");
    const int blocksN = (int)M.meta("meta/blocks"), lmBlocks = (int)M.meta("meta/lmEncoderBlocks"),
              codaBlocks = (int)M.meta("meta/codaBlocks");
    if (deterministic) fill(z, 0, P * C * 4);
    else run1d("ef2_trunc_normal", P * C, TruncNormalArgs{z, P * C, seed ^ 0x5eedull, sqrtf(2.f / (5.f * C)), 0});
    float* inject = scratch<float>("trunk.inject", P * C);
    float* y = scratch<float>("trunk.y", chunk * C);
    for (int loop = 0; loop < loops; ++loop) {
      if (p > 0.f) run1d("ef2_dropout", P * C, DropoutArgs{lmZ, inject, P * C, seed, (uint64_t)(1000 + loop), p, 0});
      else copy(inject, lmZ, P * C * 4);
      for (int b = 0; b < lmBlocks; ++b) trunkBlock(inject, mask, T, C, "lmEncoder/blocks", b, b + 1 < lmBlocks);
      add(inject, zi, P * C);
      for (size_t r0 = 0; r0 < P; r0 += chunk) {
        size_t r = std::min(chunk, P - r0);
        layerNorm(inject + r0 * C, xn, r, C, F("recycle/norm/scale"), F("recycle/norm/offset"));
        Gemm g{}; g.X = xn; g.tx = F16; g.W = Fh("recycle/projection"); g.tw = F16; g.Y = y; g.rows = r; g.in = C; g.out = C;
        g.label = "recycle";
        gemm(g);
        run1d("ef2_decay_update", r * C, DecayArgs{z + r0 * C, y, F("recycle/decay"), r * C, (uint)C, 0});
      }
      for (int b = 0; b < blocksN; ++b) trunkBlock(z, mask, T, C, "blocks", b, b + 1 < blocksN);
    }
    for (size_t r0 = 0; r0 < P; r0 += chunk) {          // the readout, a chunk of rows at a time, in place
      size_t r = std::min(chunk, P - r0);
      lin(z + r0 * C, "f/readout", y, r, C, C);
      copy(z + r0 * C, y, r * C * 4);
    }
    for (int b = 0; b < codaBlocks; ++b) trunkBlock(z, mask, T, C, "coda/blocks", b, b + 1 < codaBlocks);
    return;
  }
  int blocks = (int)M.meta("meta/blocks");
  // The blocks over a half pair: their updates end in bandwidth-bound GEMMs that read and write it, and half moves half
  // the bytes. Where the matrix units run (their kernels take a half residual); every residual is summed in float and
  // rounded once, every norm reads the half values. LOCALFOLD_HALF_PAIR=0 the control. And no float pair while they do:
  // the recycle (z = LN(z) @ projection + z_init) reads the half one, which is what the float one held (it was the half
  // one widened), and writes it back a chunk at a time - bit-identical, and z given back for the whole trunk (1.1 GB at
  // 1044 residues, without a reallocation a loop)
  if (halfPairBlocks(C)) {
    release(z);
    z = nullptr;
    half* zh = scratch<half>("trunk.zh", P * C);
    fill(zh, 0, P * C * 2);
    const std::string firstLn = "blocks/0/triangleMultiplicationOutgoing/leftNormInput";
    for (int loop = 0; loop < loops; ++loop) {
      half* in = takeEmitted(zh, "recycle/norm/");
      half* lnOut = lnTarget(P, C, in);
      half* xn = in ? nullptr : scratch<half>("frec.xn", chunk * C);
      bool all = true;
      for (size_t r0 = 0; r0 < P; r0 += chunk) {
        size_t r = std::min(chunk, P - r0);
        const half* x = in ? in + r0 * C : xn;
        if (!in) layerNorm(zh + r0 * C, xn, r, C, F("recycle/norm/scale"), F("recycle/norm/offset"));
        Gemm g{}; g.X = x; g.tx = F16; g.W = Fh("recycle/projection"); g.tw = F16;
        g.Y = zh + r0 * C; g.ty = F16; g.Yin = zi + r0 * C; g.yinFloat = true;
        g.beta = 1.f; g.rows = r; g.in = C; g.out = C; g.label = "recycle";
        g.lnOut = lnOut + r0 * C; g.lnScale = F(firstLn + "Scale"); g.lnOffset = F(firstLn + "Offset");
        all = gemm(g) && all;
      }
      if (all) emitted = {zh, firstLn, lnOut};
      for (int b = 0; b < blocks; ++b)
        trunkBlockT(zh, mask, T, C, "blocks", b, b + 1 < blocks, loop + 1 < loops ? "recycle/norm/" : "");
    }
    releaseScratch({"ftri.", "ftr.", "frec.", "eln."});
    z = allocT<float>(P * C);
    toFloat(zh, z, P * C);
    return;
  }
  fill(z, 0, P * C * 4);
  const std::string firstLn = "blocks/0/triangleMultiplicationOutgoing/leftNormInput";
  for (int loop = 0; loop < loops; ++loop) {
    // z = LN(z) @ projection + z_init, a chunk of rows at a time
    half* in = takeEmitted(z, "recycle/norm/");
    half* lnOut = lnTarget(P, C, in);
    half* xn = in ? nullptr : scratch<half>("frec.xn", chunk * C);
    bool all = true;
    for (size_t r0 = 0; r0 < P; r0 += chunk) {
      size_t r = std::min(chunk, P - r0);
      const half* x = in ? in + r0 * C : xn;
      if (!in) layerNorm(z + r0 * C, xn, r, C, F("recycle/norm/scale"), F("recycle/norm/offset"));
      Gemm g{}; g.X = x; g.tx = F16; g.W = Fh("recycle/projection"); g.tw = F16; g.Y = z + r0 * C; g.Yin = zi + r0 * C;
      g.beta = 1.f; g.rows = r; g.in = C; g.out = C; g.label = "recycle";
      g.lnOut = lnOut + r0 * C; g.lnScale = F(firstLn + "Scale"); g.lnOffset = F(firstLn + "Offset");
      all = gemm(g) && all;
    }
    if (all) emitted = {z, firstLn, lnOut};
    for (int b = 0; b < blocks; ++b)
      trunkBlock(z, mask, T, C, "blocks", b, b + 1 < blocks, loop + 1 < loops ? "recycle/norm/" : "");
  }
}

// ---------------------------------------------------------------- the full ESMFold2's MSA encoder (cuda/ef2's msa.cuh)
// biohub's MSAEncoder: m = embed([one-hot(33) * mask | has_deletion | deletion_value]) + project_inputs(s_inputs)[token];
// per block pair += OPM(m), then (not the last) m += PWA(m, pair) and a SwiGLU transition, then the pair-only block. Its
// answer replaces z_init (msa_encoder_overwrite), and neither the alignment nor z_init changes between passes, so it runs
// once a fold, in place over z_init. Float throughout; the GEMMs as the rest of the port's (lin).
static void msaOuterProductMean(float* pair, const float* m, const float* mask, int S, int T, int Cm, int C, const std::string& Bn) {
  const int Hh = (int)M.meta("meta/opmHidden");
  size_t rows = (size_t)S * T;
  float* ln = scratch<float>("msa.ln", rows * Cm);
  layerNorm(m, ln, rows, Cm, F(Bn + "normScale"), F(Bn + "normOffset"));
  float* x = scratch<float>("msa.opmx", rows * 2 * Hh);
  lin(ln, "f/" + Bn + "projection", x, rows, Cm, 2 * Hh);
  float* a = scratch<float>("msa.opma", rows * Hh); float* b = scratch<float>("msa.opmb", rows * Hh);
  run1d("ef2_opm_split", rows * Hh, OpmSplitArgs{x, mask, a, b, rows, (uint)Hh, 0});
  float* count = scratch<float>("msa.count", (size_t)T * T);      // count[i][j] = sum_s mask[s][i] mask[s][j]
  { Gemm g{}; g.X = mask; g.transX = true; g.W = mask; g.Y = count; g.rows = T; g.in = S; g.out = T; g.label = "msa.count"; gemm(g); }
  size_t per = (size_t)T * (2 * Hh * Hh + C) * 4;                  // a query row's outer products and its update
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(T, ((size_t)256 << 20) / per));
  float* P = scratch<float>("msa.opmP", (size_t)Bi * T * Hh * Hh);
  float* Pp = scratch<float>("msa.opmPp", (size_t)Bi * T * Hh * Hh);
  float* X = scratch<float>("msa.opmX", (size_t)Bi * T * C);
  for (int i0 = 0; i0 < T; i0 += Bi) {
    int bi = std::min(Bi, T - i0);
    // P [(i, c)][(j, e)] = sum_s a[s, i, c] b[s, j, e]: a's block of rows read transposed, [S][T H]
    Gemm g{}; g.X = a + (size_t)i0 * Hh; g.transX = true; g.ldx = T * Hh; g.W = b; g.Y = P;
    g.rows = (size_t)bi * Hh; g.in = S; g.out = T * Hh; g.label = "msa.opm";
    gemm(g);
    run1d("ef2_opm_permute", (size_t)bi * T * Hh * Hh, OpmPermuteArgs{P, Pp, (uint)bi, (uint)T, (uint)Hh, 0});
    // (float operands: the outer products are sums over every row of the alignment - 1,024 of them - and pass half's
    // 65504, which lin's half rounding of X turned inf at 59 tokens)
    Gemm go{}; go.X = Pp; go.W = M.f("f/" + Bn + "output"); go.Y = X; go.rows = (size_t)bi * T; go.in = Hh * Hh; go.out = C;
    go.label = "msa.opm.output";
    gemm(go);
    run1d("ef2_opm_add", (size_t)bi * T * C, OpmAddArgs{pair, X, F(Bn + "outputBias"), count, (uint)i0, (uint)bi, (uint)T, (uint)C});
  }
}
static void msaPairWeightedAveraging(float* m, const float* pair, int S, int T, int Cm, int C, const std::string& Bn) {
  const int heads = (int)M.meta("meta/msaHeads"), d = (int)M.meta("meta/msaHeadWidth"), Wd = heads * d;
  size_t rows = (size_t)S * T, P = (size_t)T * T;
  float* ln = scratch<float>("msa.ln", rows * Cm);
  layerNorm(m, ln, rows, Cm, F(Bn + "normScale"), F(Bn + "normOffset"));
  float* flat = scratch<float>("msa.flat", P * heads);
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));
  float* pln = scratch<float>("msa.pln", chunk * C);
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    layerNorm(pair + r0 * C, pln, r, C, F(Bn + "pairNormScale"), F(Bn + "pairNormOffset"));
    lin(pln, "f/" + Bn + "pairLogits", flat + r0 * heads, r, C, heads);
  }
  float* w = scratch<float>("msa.w", (size_t)heads * P);
  run("ef2_pwa_softmax", grid1d((size_t)heads * T, 1), 256, PwaSoftmaxArgs{flat, w, (uint)T, (uint)heads});
  // each row's own values, weighted sums, gate and output: a block of rows at a time
  int Sc = (int)std::max<size_t>(1, std::min<size_t>(S, ((size_t)128 << 20) / ((size_t)T * Wd * 4)));
  float* v = scratch<float>("msa.v", (size_t)Sc * T * Wd); float* vh = scratch<float>("msa.vh", (size_t)Sc * T * Wd);
  float* oh = scratch<float>("msa.oh", (size_t)Sc * T * Wd); float* g = scratch<float>("msa.g", (size_t)Sc * T * Wd);
  for (int s0 = 0; s0 < S; s0 += Sc) {
    int sc = std::min(Sc, S - s0); size_t cr = (size_t)sc * T;
    const float* lnc = ln + (size_t)s0 * T * Cm;
    lin(lnc, "f/" + Bn + "value", v, cr, Cm, Wd);
    run1d("ef2_pwa_to_heads", cr * Wd, PwaHeadsArgs{v, vh, nullptr, (uint)sc, (uint)T, (uint)heads, (uint)d});
    // per head: O_h [i][(s, e)] = sum_j w_h[i][j] V_h[j][(s, e)]
    Gemm gm{}; gm.X = w; gm.W = vh; gm.Y = oh; gm.rows = T; gm.in = T; gm.out = sc * d;
    gm.batch = heads; gm.sx = (int64_t)P; gm.sw = (int64_t)T * sc * d; gm.sy = (int64_t)T * sc * d; gm.label = "msa.pwa";
    gemm(gm);
    lin(lnc, "f/" + Bn + "gate", g, cr, Cm, Wd);
    run1d("ef2_pwa_from_heads", cr * Wd, PwaHeadsArgs{oh, v, g, (uint)sc, (uint)T, (uint)heads, (uint)d});
    lin(v, "f/" + Bn + "output", m + (size_t)s0 * T * Cm, cr, Wd, Cm, 1.f);
  }
}
// z_init (in, out): the encoder's pair. The alignment's rows past msaMaxDepth (1024) subsampled keeping the query and the
// a3m's order, and 10% of the columns masked below the query - both drawn from the seed, as biohub's processor does at
// inference; EF2_DETERMINISTIC=1 takes the first rows and masks nothing
void msaEncode(float* zi, const float* sInputs, int T, int C, int Si, uint64_t seed) {
  static const bool deterministic = getenv("EF2_DETERMINISTIC") != nullptr;
  const int depth = (int)M.meta("meta/msa_depth"), Cm = (int)M.meta("meta/msaChannels");
  const int maxDepth = (int)M.meta("meta/msaMaxDepth");
  const float rate = deterministic ? 0.f : (float)M.meta("meta/msaColumnMaskRate");
  std::mt19937_64 rng(seed ^ 0x4d5341ull);
  std::vector<int> keep(depth);
  for (int s = 0; s < depth; ++s) keep[s] = s;
  if (depth > maxDepth) {
    if (!deterministic) std::shuffle(keep.begin() + 1, keep.end(), rng);
    keep.resize(maxDepth);
    std::sort(keep.begin(), keep.end());
  }
  const int S = (int)keep.size();
  const int* rowsH = M.hostI("msa/rows"); const float* delH = M.hostF("msa/deletion");
  std::vector<int> rows((size_t)S * T); std::vector<float> del((size_t)S * T), mask((size_t)S * T, 1.f);
  for (int s = 0; s < S; ++s)
    for (int t = 0; t < T; ++t) { rows[(size_t)s * T + t] = rowsH[(size_t)keep[s] * T + t]; del[(size_t)s * T + t] = delH[(size_t)keep[s] * T + t]; }
  if (S > 1 && rate > 0.f) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    for (int t = 0; t < T; ++t)
      if (u(rng) < rate) for (int s = 1; s < S; ++s) mask[(size_t)s * T + t] = 0.f;
  }
  if (getenv("EF2_SAVE_MSA")) {                 // (a comparison aid: the rows, deletions and mask the encoder reads)
    FILE* f = fopen(getenv("EF2_SAVE_MSA"), "wb");
    int dims[2] = {S, T}; fwrite(dims, 4, 2, f);
    fwrite(rows.data(), 4, rows.size(), f); fwrite(del.data(), 4, del.size(), f); fwrite(mask.data(), 4, mask.size(), f);
    fclose(f);
  }
  size_t R = (size_t)S * T, P = (size_t)T * T;
  int* rowsD = scratch<int>("msa.rows", R); float* delD = scratch<float>("msa.del", R); float* maskD = scratch<float>("msa.mask", R);
  upload(rowsD, rows.data(), R * 4); upload(delD, del.data(), R * 4); upload(maskD, mask.data(), R * 4);
  float* fromInputs = scratch<float>("msa.fromInputs", (size_t)T * Cm);
  lin(sInputs, "f/msaEncoder/projectInputs", fromInputs, T, Si, Cm);
  float* m = scratch<float>("msa.m", R * Cm);
  run1d("ef2_msa_embed", R * Cm, MsaEmbedArgs{rowsD, delD, maskD, F("msaEncoder/embed"), fromInputs, m, R, (uint)T, (uint)Cm});
  float* pmask = scratch<float>("msa.pmask", P);
  run1d("ef2_fill", P, FillFArgs{pmask, P, 1.f, 0});
  const int blocksN = (int)M.meta("meta/msaBlocks");
  for (int b = 0; b < blocksN; ++b) {
    std::string Bn = "msaEncoder/blocks/" + std::to_string(b) + "/";
    msaOuterProductMean(zi, m, maskD, S, T, Cm, C, Bn + "outerProductMean/");
    if (M.has("f/" + Bn + "pairWeightedAveraging/value")) {
      msaPairWeightedAveraging(m, zi, S, T, Cm, C, Bn + "pairWeightedAveraging/");
      transition(m, R, Cm, Bn + "msaTransition/", "");
    }
    trunkBlock(zi, pmask, T, C, "msaEncoder/blocks", b, false);
  }
  releaseScratch({"msa."});
}

std::vector<float> contactMap(const float* z, int T, int C) {
  if (!M.has("contact_bins")) return {};
  int bins = (int)M.meta("meta/distogramBins");
  if ((int)M.meta("meta/contactBinsFor") != bins)
    die("the input's contact bins were counted for a %d-bin distogram and this one has %d", (int)M.meta("meta/contactBinsFor"), bins);
  size_t P = (size_t)T * T;
  int Bn = dimOf("f/distogram/weights", 1);
  float* probs = scratch<float>("dg.contacts", P);
  // a block of pair positions at a time: z + z^T, the projection and its bias, the contact mass
  size_t per = std::max<size_t>(1, std::min(P, ((size_t)64 << 20) / std::max(C, Bn) / 4));
  float* zs = scratch<float>("dg.sym", per * C); float* dg = scratch<float>("dg.logits", per * Bn);
  for (size_t p0 = 0; p0 < P; p0 += per) {
    size_t n = std::min(per, P - p0);
    run1d("ef2_sym_rows", (n * C) >> 2, SymRowsArgs{z, zs, p0, n, (uint)T, (uint)C});
    lin(zs, "f/distogram/weights", dg, n, C, Bn, 0.f, F("distogram/bias"));
    run1d("ef2_contacts", n, ContactsArgs{dg, Ii("contact_bins") + p0, probs + p0, n, (uint)bins, 0});
  }
  return download(probs, P);
}
