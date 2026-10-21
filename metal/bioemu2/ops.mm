// metal/bioemu2's shared pieces: the bundle's weights, the GEMM on either layout, the vendor's SwiGLU transition, and
// the pair track's two updates in AF3's layout (cuda/bioemu2/src/net.cuh and trunk.cuh's trimul are the reading).
#include "bioemu2.h"

Dims D_;

static std::string wn(const std::string& n) { return "w/" + n; }
bool hasW(const std::string& n) { return M.has(wn(n)); }
const float* Wf(const std::string& n) {
  if (!M.has(wn(n))) die("BioEmu2: the bundle has no %s", n.c_str());
  return M.f(wn(n));
}
const float* WfOpt(const std::string& n) { return M.has(wn(n)) ? M.f(wn(n)) : nullptr; }
int dimW(const std::string& n, int k) {
  if (!M.has(wn(n))) die("BioEmu2: the bundle has no %s", n.c_str());
  return k < M.rank(wn(n)) ? (int)M.dim(wn(n), k) : 1;
}
size_t lenW(const std::string& n) { return M.len(wn(n)); }

static void linear(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta, int ldx, int ldy,
                   bool torch) {
  const int d0 = dimW(w, 0), d1 = dimW(w, 1);
  if (torch ? (d0 != out || d1 != in) : (d0 != in || d1 != out))
    die("BioEmu2: %s is [%d, %d], read as %s [%d, %d]", w.c_str(), d0, d1, torch ? "(out, in)" : "(in, out)",
        torch ? out : in, torch ? in : out);
  Gemm g{}; g.X = X; g.tx = F32; g.ldx = ldx; g.W = Wf(w); g.tw = F32; g.transW = torch; g.Y = Y; g.ty = F32; g.ldy = ldy;
  g.rows = rows; g.in = in; g.out = out; g.beta = beta; g.label = w.c_str();
  gemm(g);
}
void linT(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta, int ldx, int ldy) {
  linear(X, w, Y, rows, in, out, beta, ldx, ldy, true);
}
void linA(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta, int ldx, int ldy) {
  linear(X, w, Y, rows, in, out, beta, ldx, ldy, false);
}
void relu(const float* x, float* y, size_t n) { run1d("be2_unary", n, Be2UnaryArgs{x, y, n, 0, 0}); }
void silu(const float* x, float* y, size_t n) { run1d("be2_unary", n, Be2UnaryArgs{x, y, n, 1, 0}); }

// out = fc3(silu(fc1 x') fc2 x'), a chunk of rows at a time (the widened rows are 2I a row)
void transition(const float* x, float* out, size_t rows, int C, const std::string& p, bool residual, bool norm, int ldx) {
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
    linT(h1, p + "fc3/weight", out + r0 * Co, r, I, Co, residual ? 1.f : 0.f);
  }
}

void pairTransition(float* z, size_t rows, int C, const std::string& p) {
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
    linA(g, p + "transition2", z + r0 * C, r, I, C, 1.f);
  }
}

// TriangleMultiplication in AF3's form: zn = LN(z); a | b = projection(zn) sigmoid(gate(zn)), interleaved by channel;
// outgoing out[i][j] = sum_k a[i][k] b[j][k], incoming out[i][j] = sum_k a[k][j] b[k][i] (the bundle's incoming halves
// are swapped at export, so this is Boltz's sum_k a[k][i] b[k][j]); z += outputProjection(LN(out)) sigmoid(gatingLinear(zn))
void triangle(float* z, int T, int C, const std::string& p, bool outgoing) {
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
