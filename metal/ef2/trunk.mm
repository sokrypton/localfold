// z_init, the recycle and the 24-block pair trunk, and the contact map (cuda/ef2/src/trunk.cuh is the reading):
//
//   z_init = z_init_1(s)[i] + z_init_2(s)[j] + rel_pos + token_bonds + lm_z
//   z = 0;  num_loops + 1 times:  z = z_init + Linear(LN(z));  z = trunk(z)
//   block: z += triangle out(z); z += triangle in(z); z += transition(z)       (no biases)
// A triangle: LN -> one GEMM gating a and b into channel-major planes (gemmTriGate) -> the planes' product, a
// batched GEMM -> the centre LayerNorm back to rows -> the output projection -> the gate's GEMM adding its gated
// product into the pair (gemmGatedAdd). A transition: LN -> SwiGLU in the first GEMM's epilogue -> the second GEMM
// adding into the pair. The pair is float32; everything between is float16.
#include "ef2.h"

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
  return M.derived<half>("trigate:" + Tn, (size_t)C * 4 * C, [&](half* out) {
    run1d("ef2_trigate_weight", (size_t)C * 4 * C, TriGateWArgs{Fh(Tn + "projection"), Fh(Tn + "gate"), out, (uint)C, (uint)C});
  });
}
static void triangle(float* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing) {
  size_t P = (size_t)L * L;
  int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
  half* xn = scratch<half>("ftri.xn", P * C);
  layerNorm(pair, xn, P, C, F(Tn + "leftNormInputScale"), F(Tn + "leftNormInputOffset"));
  half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
  // the pad is written by nothing: zeroed once for a buffer and a size
  static half *zeroedA = nullptr, *zeroedB = nullptr; static int zeroedLp = 0;
  if (Lp != L && (a != zeroedA || b != zeroedB || Lp != zeroedLp)) {
    fill(a, 0, plane * C * 2); fill(b, 0, plane * C * 2);
    zeroedA = a; zeroedB = b; zeroedLp = Lp;
  }
  gemmTriGate(xn, triGate(Tn, C), mask, a, b, 0, P, C, plane, L, Lp);
  // per channel: outgoing prod[j][i] = sum_k a[j][k] b[i][k]; incoming sum_k b[k][j] a[k][i]
  float* prod = scratch<float>("ftri.prod", plane * C);
  {
    Gemm g{}; g.tx = F16; g.tw = F16; g.ty = F32; g.rows = Lp; g.in = Lp; g.out = Lp; g.ldx = g.ldw = g.ldy = Lp;
    g.sx = g.sw = g.sy = (int64_t)plane; g.batch = C; g.Y = prod; g.label = outgoing ? "triangle out contraction" : "triangle in contraction";
    if (outgoing) { g.X = a; g.W = b; g.transW = true; }
    else { g.X = b; g.transX = true; g.W = a; }
    gemm(g);
  }
  half* cn = scratch<half>("ftri.cn", P * C);
  run("ef2_center_norm", grid1d((P + 31) / 32, 1), 256,
      CenterNormArgs{prod, cn, P, (uint)C, (uint)L, (uint)Lp, 0, F(Tn + "centerNormScale"), F(Tn + "centerNormOffset")});
  half* outH = scratch<half>("ftri.outh", P * C);
  linH(cn, "f/" + Tn + "outputProjection", outH, P, C, C);
  gemmGatedAdd(xn, Fh(Tn + "gatingLinear"), outH, pair, P, C, C);
}
static void transition(float* pair, size_t P, int C, const std::string& Tn) {
  int I = dimOf("f/" + Tn + "transition2", 0);
  size_t chunk = std::max<size_t>(64, ((size_t)128 << 20) / (3 * (size_t)I));
  size_t rows = std::min(P, chunk);
  half* xn = scratch<half>("ftr.xn", rows * C);
  half* g = scratch<half>("ftr.g", rows * I);
  const half* w1 = swigluPairs(Tn + "transition1", Fh(Tn + "transition1"), C, I);
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    layerNorm(pair + r0 * C, xn, r, C, F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"));
    gemmSwiglu(xn, w1, g, r, C, I);
    lin(g, "f/" + Tn + "transition2", pair + r0 * C, r, I, C, 1.f);
  }
}
void trunkBlock(float* pair, const float* mask, int L, int C, const std::string& prefix, int b) {
  std::string B = prefix + "/" + std::to_string(b) + "/";
  triangle(pair, mask, L, C, B + "triangleMultiplicationOutgoing/", true);
  triangle(pair, mask, L, C, B + "triangleMultiplicationIncoming/", false);
  transition(pair, (size_t)L * L, C, B + "pairTransition/");
}

void foldingTrunk(int T, int C, const float* zi, float* z, int loops) {
  size_t P = (size_t)T * T;
  float* mask = scratch<float>("trunk.mask", P);
  run1d("ef2_fill", P, FillFArgs{mask, P, 1.f, 0});          // every token is real
  if (M.has("f/recycle/decay")) die("this model's recycle (the released ESMFold2's parcae loop) is not in the native port yet");
  fill(z, 0, P * C * 4);
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));
  half* xn = scratch<half>("trunk.xn", chunk * C);
  int blocks = (int)M.meta("meta/blocks");
  for (int loop = 0; loop < loops; ++loop) {
    // z = LN(z) @ projection + z_init, a chunk of rows at a time
    for (size_t r0 = 0; r0 < P; r0 += chunk) {
      size_t r = std::min(chunk, P - r0);
      layerNorm(z + r0 * C, xn, r, C, F("recycle/norm/scale"), F("recycle/norm/offset"));
      Gemm g{}; g.X = xn; g.tx = F16; g.W = Fh("recycle/projection"); g.tw = F16; g.Y = z + r0 * C; g.Yin = zi + r0 * C;
      g.beta = 1.f; g.rows = r; g.in = C; g.out = C; g.label = "recycle";
      gemm(g);
    }
    for (int b = 0; b < blocks; ++b) trunkBlock(z, mask, T, C, "blocks", b);
  }
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
    run1d("ef2_sym_rows", n * C, SymRowsArgs{z, zs, p0, n, (uint)T, (uint)C});
    lin(zs, "f/distogram/weights", dg, n, C, Bn, 0.f, F("distogram/bias"));
    run1d("ef2_contacts", n, ContactsArgs{dg, Ii("contact_bins") + p0, probs + p0, n, (uint)bins, 0});
  }
  return download(probs, P);
}
