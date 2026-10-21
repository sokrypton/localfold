// metal/bioemu2's shared pieces: the bundle's weights, the GEMM on either layout, the vendor's SwiGLU transition, and
// the pair track's two updates in AF3's layout (cuda/bioemu2/src/net.cuh and trunk.cuh's trimul are the reading).
#include "bioemu2.h"
#include <cmath>
#include <map>

Dims D_;
bool HALF = true;

static std::string wn(const std::string& n) { return "w/" + n; }
bool hasW(const std::string& n) { return M.has(wn(n)); }
const float* Wf(const std::string& n) {
  if (!M.has(wn(n))) die("BioEmu2: the bundle has no %s", n.c_str());
  return M.f(wn(n));
}
const float* WfOpt(const std::string& n) { return M.has(wn(n)) ? M.f(wn(n)) : nullptr; }
int dimW(const std::string& n, int k) {
  if (!M.has(wn(n))) die("BioEmu2: the bundle has no %s", n.c_str());
  const std::string r = wn(n) + "#r";          // (a weight walk's tensor - the blob's - carries its shape as metadata)
  if (M.has(r)) return k < (int)M.meta(r) ? (int)M.meta(wn(n) + "#" + std::to_string(k)) : 1;
  return k < M.rank(wn(n)) ? (int)M.dim(wn(n), k) : 1;
}
size_t lenW(const std::string& n) { return M.len(wn(n)); }

// HALF: the weight's float16 copy, X rounded to half as it is staged and the product accumulated in float - as CUDA's
// GEMM_HALF, and only where both sides are at least 16 wide (r_to_q's 3-wide coordinates stay float). `wide`: an input
// that may pass float16's range, multiplied in float against the half weight
static void linear(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta, int ldx, int ldy,
                   bool torch, bool wide, const float* bias = nullptr) {
  const int d0 = dimW(w, 0), d1 = dimW(w, 1);
  if (torch ? (d0 != out || d1 != in) : (d0 != in || d1 != out))
    die("BioEmu2: %s is [%d, %d], read as %s [%d, %d]", w.c_str(), d0, d1, torch ? "(out, in)" : "(in, out)",
        torch ? out : in, torch ? in : out);
  Gemm g{}; g.X = X; g.tx = F32; g.ldx = ldx; g.transW = torch; g.Y = Y; g.ty = F32; g.ldy = ldy;
  g.rows = rows; g.in = in; g.out = out; g.beta = beta; g.label = w.c_str(); g.bias = bias;
  if (HALF && in >= 16 && out >= 16) { g.W = M.h("w/" + w); g.tw = F16; g.half = !wide; }
  else { g.W = Wf(w); g.tw = F32; }
  gemm(g);
}
void linT(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta, int ldx, int ldy, bool wide,
          const float* bias) {
  linear(X, w, Y, rows, in, out, beta, ldx, ldy, true, wide, bias);
}
void linA(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta, int ldx, int ldy, bool wide) {
  linear(X, w, Y, rows, in, out, beta, ldx, ldy, false, wide);
}
void relu(const float* x, float* y, size_t n) { run1d("be2_unary", n, Be2UnaryArgs{x, y, n, 0, 0}); }
void silu(const float* x, float* y, size_t n) { run1d("be2_unary", n, Be2UnaryArgs{x, y, n, 1, 0}); }

// ---------------------------------------------------------------- the SwiGLU transition
// 🔴 EVERY SwiGLU PRODUCT IS STORED AS g / 64 ON THE FAST ARM, because the products are not bounded by anything float16
// can hold. Measured over three jobs (6MRR, 6MRR + GOL + SEP, a 156-token heteromer), the largest silu(fc1 x) fc2 x per
// module: the coupling's s_to_q_mixer 268,272 - four times float16's 65,504 - the atom transition 84,467 (cuda/bioemu2
// met this one: 87,750), the MSA transition 25,943, the pair transitions 9,845; the others under 2,000. Nearly the same
// on every job: they are properties of the weights. /64 puts the ceiling at 4.2 M, 15x past the worst, and is exact -
// a power of two, folded into fc2's rows of the derived weight (fc2 / 64 in float16: a weight under 3.9e-3 goes
// subnormal, its error then 3.8e-6, a thousandth of the int5 step it already carries) and undone by fc3's alpha
static const float SWIGLU_SCALE = getenv("BE2_SWIGLU_SCALE") ? (float)atof(getenv("BE2_SWIGLU_SCALE")) : 64.f;   // (an arm, BE2_SWIGLU_SCALE)
static const half* swigluWeight(const std::string& p, int C, int I) {
  return M.derived<half>("swiglu:" + p, (size_t)C * 2 * I, [&](half* out) {
    be2Run1d("be2_swiglu_weight", (size_t)C * 2 * I, Be2SwigluWArgs{M.h("w/" + p + "fc1/weight"), M.h("w/" + p + "fc2/weight"), out,
                                                                     (uint)C, (uint)I, 1.f / SWIGLU_SCALE, 0});
  });
}
const half* swigluPairs(const std::string& p, int C, int I) { return swigluWeight(p, C, I); }
float swigluScale() { return SWIGLU_SCALE; }
void transitionFc3(const half* g, float* out, size_t rows, int I, const std::string& p, bool residual) {
  const int Co = dimW(p + "fc3/weight", 0);
  Gemm G{}; G.X = g; G.tx = F16; G.W = M.h("w/" + p + "fc3/weight"); G.tw = F16; G.transW = true; G.Y = out; G.ty = F32;
  G.rows = rows; G.in = I; G.out = Co; G.alpha = SWIGLU_SCALE; G.beta = residual ? 1.f : 0.f; G.label = "transition fc3";
  gemm(G);
}
// the fast arm's transition on an input already in half: fc1 | fc2 and the SwiGLU as one GEMM (gemmSwiglu: the product
// in its epilogue, into half), then fc3 scaled back; a chunk of rows at a time
void transitionHalf(const half* xh, float* out, size_t rows, int C, const std::string& p, bool residual) {
  const int I = dimW(p + "fc1/weight", 0), Co = dimW(p + "fc3/weight", 0);
  if (dimW(p + "fc1/weight", 1) != C || dimW(p + "fc3/weight", 1) != I) die("BioEmu2: %s's shapes do not chain", p.c_str());
  const half* w12 = swigluWeight(p, C, I);
  const size_t chunk = std::max<size_t>(64, ((size_t)128 << 20) / (2 * (size_t)I));
  half* g = scratch<half>("be2.trh.g", std::min(rows, chunk) * I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    const size_t r = std::min(chunk, rows - r0);
    gemmSwiglu(xh + r0 * C, w12, g, r, C, I);
    Gemm G{}; G.X = g; G.tx = F16; G.W = M.h("w/" + p + "fc3/weight"); G.tw = F16; G.transW = true; G.Y = out + r0 * Co; G.ty = F32;
    G.rows = r; G.in = I; G.out = Co; G.alpha = SWIGLU_SCALE; G.beta = residual ? 1.f : 0.f; G.label = "transition fc3";
    gemm(G);
  }
}

// out = fc3(silu(fc1 x') fc2 x'), a chunk of rows at a time (the widened rows are 2I a row)
void transition(const float* x, float* out, size_t rows, int C, const std::string& p, bool residual, bool norm, int ldx) {
  if (HALF) {
    half* xh = scratch<half>("be2.tr.xh", rows * C);
    if (norm) layerNorm(x, xh, rows, C, Wf(p + "norm/weight"), Wf(p + "norm/bias"), 1e-5f, ldx ? ldx : C, C);
    else if (!ldx || ldx == C) toHalf(x, xh, rows * C);
    else die("BioEmu2: a strided transition input without a norm");
    transitionHalf(xh, out, rows, C, p, residual);
    return;
  }
  const int I = dimW(p + "fc1/weight", 0), Co = dimW(p + "fc3/weight", 0);    // (a mixer narrows: 896 -> 128)
  const float* xin = x; int ld = ldx ? ldx : C;
  if (norm) {
    float* xn = scratch<float>("be2.tr.xn", rows * C);
    layerNorm(x, xn, rows, C, Wf(p + "norm/weight"), Wf(p + "norm/bias"), 1e-5f, ld, C);
    xin = xn; ld = C;
  }
  const size_t chunk = std::max<size_t>(1, ((size_t)96 << 20) / (4 * (size_t)I * 3));
  float* h1 = scratch<float>("be2.tr.h1", std::min(rows, chunk) * I);
  float* h2 = scratch<float>("be2.tr.h2", std::min(rows, chunk) * I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    const size_t r = std::min(chunk, rows - r0);
    linT(xin + r0 * ld, p + "fc1/weight", h1, r, C, I, 0.f, ld);
    linT(xin + r0 * ld, p + "fc2/weight", h2, r, C, I, 0.f, ld);
    run1d("be2_silu_mul", r * I, Be2SiluMulArgs{h1, h2, h1, r * I});
    // (a measurement: the largest SwiGLU product a module class reaches, printed at exit - with --f32 only, the one arm
    // that holds the product in float; the fast arm never materialises it, so it prints nothing there)
    if (getenv("BE2_SWIGLU_MAX")) {
      static std::map<std::string, float> worst;
      static bool hooked = false;
      if (!hooked) { hooked = true; atexit([] { for (auto& [k, v] : worst) fprintf(stderr, "swiglu max %-40s %.6g\n", k.c_str(), v); }); }
      const std::vector<float> v = download(h1, r * I);
      float m = 0; for (float x : v) m = std::max(m, fabsf(x));
      std::string k = p; for (char& ch : k) if (isdigit((unsigned char)ch)) ch = '#';
      worst[k] = std::max(worst[k], m);
    }
    // (🔴 the SwiGLU product is unbounded: 87,750 in the atom stream's transition at 6MRR - cuda/bioemu2's note - past
    // float16's 65,504, so fc3 multiplies it in float against the half weight rather than rounding it to inf)
    linT(h1, p + "fc3/weight", out + r0 * Co, r, I, Co, residual ? 1.f : 0.f, 0, 0, true);
  }
}

// ---------------------------------------------------------------- the pair track's fast arm (metal/ef2's)
// The pair stays FLOAT32 - its RMS is ~4.6e3 at 6MRR, where float16's step is 4 - and everything between is half: the
// triangle a LayerNorm into half, one GEMM gating a and b into channel-major planes (gemmTriGate), their contraction and
// centre norm (triContract), the output projection and the gate's GEMM adding into the pair (gemmGatedAddDual); the
// transition a LayerNorm, the SwiGLU in the first GEMM's epilogue (stored / 64, as every SwiGLU here) and the second
// GEMM adding into the pair. The bundle's pair track is AF3's layout, which is what these kernels were written for.
static float* ONES = nullptr; static size_t ONES_N = 0;
static const float* onesMask(size_t n) {
  if (ONES_N < n) {
    if (ONES) release(ONES);
    std::vector<float> v(n, 1.f);
    ONES = uploadNew(v.data(), n); ONES_N = n;       // (every token is real here: no token padding)
  }
  return ONES;
}
static const half* triGateWeight(const std::string& p, int C) {
  if (triQuartersApply(C))      // (the matrix units' quarters, built once through a blocks-of-8 temporary)
    return M.derived<half>("trigateQ:" + p, (size_t)C * 4 * C, [&](half* out) {
      half* w8 = allocT<half>((size_t)C * 4 * C);
      be2Run1d("be2_trigate_weight", (size_t)C * 4 * C, Be2TriGateWArgs{M.h("w/" + p + "projection"), M.h("w/" + p + "gate"), w8,
                                                                         (uint)C, (uint)C});
      triQuarters(w8, out, nullptr, nullptr, C);
      release(w8);
    });
  return M.derived<half>("trigate:" + p, (size_t)C * 4 * C, [&](half* out) {
    be2Run1d("be2_trigate_weight", (size_t)C * 4 * C, Be2TriGateWArgs{M.h("w/" + p + "projection"), M.h("w/" + p + "gate"), out,
                                                                       (uint)C, (uint)C});
  });
}
static void triangleFast(float* z, int T, int C, const std::string& p, bool outgoing) {
  const size_t P = (size_t)T * T;
  const int Tp = (T + 7) / 8 * 8; const size_t plane = (size_t)Tp * Tp;
  half* xn = scratch<half>("be2.ftri.xn", P * C);
  layerNorm(z, xn, P, C, Wf(p + "leftNormInputScale"), Wf(p + "leftNormInputOffset"));
  void* big = scratchBytes("be2.ftri.big", triBytes(C, Tp));
  half* a = triPlanes(big, T, C, Tp);
  gemmTriGate(xn, triGateWeight(p, C), onesMask(P), a, a + plane, 0, P, C, 2 * plane, T, Tp, nullptr, triQuartersApply(C));
  TriMid mid = triContract(big, T, C, outgoing, 1.f, Wf(p + "centerNormScale"), Wf(p + "centerNormOffset"),
                           halfProductScale() ? nullptr : scratch<half>("be2.ftri.cn", P * C),
                           outgoing ? "triangle out contraction" : "triangle in contraction");
  gemmGatedAddDual(xn, M.h("w/" + p + "gatingLinear"), mid.cn, M.h("w/" + p + "outputProjection"), z, P, C, C, nullptr, nullptr,
                   mid.outH, "triangle output and gated add");
}
static void pairTransitionFast(float* z, size_t rows, int C, const std::string& p) {
  const int I = dimW(p + "transition1", 1) / 2;
  const half* w1 = M.derived<half>("swiglu8:" + p, (size_t)C * 2 * I, [&](half* out) {
    const half* t1 = M.h("w/" + p + "transition1");
    be2Run1d("be2_interleave8", (size_t)C * 2 * I, Be2Interleave8Args{t1, t1 + I, out, (uint)C, (uint)I, (uint)(2 * I),
                                                                        1.f / SWIGLU_SCALE});
  });
  const size_t chunk = std::max<size_t>(64, ((size_t)128 << 20) / (3 * (size_t)I));
  const size_t r1 = std::min(rows, chunk);
  half* xn = scratch<half>("be2.ftr.xn", r1 * C);
  half* g = scratch<half>("be2.ftr.g", r1 * I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    const size_t r = std::min(chunk, rows - r0);
    layerNorm(z + r0 * C, xn, r, C, Wf(p + "inputLayerNormScale"), Wf(p + "inputLayerNormOffset"));
    gemmSwiglu(xn, w1, g, r, C, I);
    Gemm G{}; G.X = g; G.tx = F16; G.W = M.h("w/" + p + "transition2"); G.tw = F16; G.Y = z + r0 * C; G.ty = F32;
    G.rows = r; G.in = I; G.out = C; G.alpha = SWIGLU_SCALE; G.beta = 1.f; G.label = "pair transition2";
    gemm(G);
  }
}

void pairTransition(float* z, size_t rows, int C, const std::string& p) {
  if (HALF) { pairTransitionFast(z, rows, C, p); return; }
  const int I = dimW(p + "transition1", 1) / 2;
  const size_t chunk = std::max<size_t>(1, ((size_t)96 << 20) / (4 * (size_t)I * 3));
  const size_t r1 = std::min(rows, chunk);
  float* xn = scratch<float>("be2.ptr.xn", r1 * C);
  float* h = scratch<float>("be2.ptr.h", r1 * 2 * I);
  float* g = scratch<float>("be2.ptr.g", r1 * I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    const size_t r = std::min(chunk, rows - r0);
    layerNorm(z + r0 * C, xn, r, C, Wf(p + "inputLayerNormScale"), Wf(p + "inputLayerNormOffset"));
    linA(xn, p + "transition1", h, r, C, 2 * I);
    run1d("be2_swiglu_blocked", r * I, Be2SwigluBlockedArgs{h, g, r, (uint)I, 0});
    if (getenv("BE2_SWIGLU_MAX")) {
      static float worst = 0;
      static bool hooked = false;
      if (!hooked) { hooked = true; atexit([] { fprintf(stderr, "swiglu max %-40s %.6g\n", "pair transitions (msa/, pairmixer/)", worst); }); }
      for (float x : download(g, r * I)) worst = std::max(worst, fabsf(x));
    }
    linA(g, p + "transition2", z + r0 * C, r, I, C, 1.f);
  }
}

// TriangleMultiplication in AF3's form: zn = LN(z); a | b = projection(zn) sigmoid(gate(zn)), interleaved by channel;
// outgoing out[i][j] = sum_k a[i][k] b[j][k], incoming out[i][j] = sum_k a[k][j] b[k][i] (the bundle's incoming halves
// are swapped at export, so this is Boltz's sum_k a[k][i] b[k][j]); z += outputProjection(LN(out)) sigmoid(gatingLinear(zn))
void triangle(float* z, int T, int C, const std::string& p, bool outgoing) {
  if (HALF) { triangleFast(z, T, C, p, outgoing); return; }
  const size_t P = (size_t)T * T;
  float* zn = scratch<float>("be2.tri.zn", P * C);
  layerNorm(z, zn, P, C, Wf(p + "leftNormInputScale"), Wf(p + "leftNormInputOffset"));
  float* proj = scratch<float>("be2.tri.proj", P * 2 * C); float* gate = scratch<float>("be2.tri.gate", P * 2 * C);
  linA(zn, p + "projection", proj, P, C, 2 * C);
  linA(zn, p + "gate", gate, P, C, 2 * C);
  float* a = scratch<float>("be2.tri.a", P * C); float* b = scratch<float>("be2.tri.b", P * C);
  run1d("be2_tri_split", P * 2 * C, Be2TriSplitArgs{proj, gate, a, b, P, (uint)C, 0});
  float* prod = proj;      // (the projection is spent: its first half holds the product, channel-major)
  Gemm g{}; g.tx = F32; g.tw = F32; g.ty = F32; g.Y = prod; g.rows = T; g.in = T; g.out = T;
  g.batch = C; g.sx = (int64_t)P; g.sw = (int64_t)P; g.sy = (int64_t)P;
  g.label = outgoing ? "triangle out contraction" : "triangle in contraction";
  if (outgoing) { g.X = a; g.W = b; g.transW = true; }        // a [i][k] . b [j][k]^T
  else { g.X = b; g.transX = true; g.W = a; }                 // b [k][i]^T . a [k][j]
  gemm(g);
  float* rows = gate;
  run1d("be2_channel_to_rows", P * C, Be2ChannelToRowsArgs{prod, rows, P, (uint)C, 0});
  layerNorm(rows, rows, P, C, Wf(p + "centerNormScale"), Wf(p + "centerNormOffset"));
  float* out = a; float* gl = b;
  linA(rows, p + "outputProjection", out, P, C, C);
  linA(zn, p + "gatingLinear", gl, P, C, C);
  run1d("be2_gate_mul_add", P * C, Be2GateMulAddArgs{z, out, gl, P * C});
}

void relPos(const std::string& p, float* out, int T, int C, bool add, int ldo) {
  run1d("be2_rel_pos", (size_t)T * T * C,
        Be2RelPosArgs{Wf(p + "embed_rel_pos/weight"), Wf(p + "embed_rel_token/weight"), Wf(p + "embed_same_entity/weight"),
                      Wf(p + "embed_rel_chain/weight"), Iin("asym_id"), Iin("entity_id"), Iin("residue_index"), Iin("sym_id"),
                      out, (uint)T, (uint)C, add ? 1u : 0u, (uint)(ldo ? ldo : C)});
}
