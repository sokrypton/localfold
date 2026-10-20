// BioEmu2's building blocks: the weights read straight out of af3-any-model's blob (one scope,
// diffuser/~/bioemu2, the vendor's PyTorch names and (out, in) layouts - converters/bioemu2.py is a rename), and
// the kernels every stage shares. float32 throughout on the reference path (cuda/bioemu2/oracle.py); --fast
// takes TF32 in cuBLAS. Shared infrastructure - the model file, scratch, cuBLAS - is cuda/af3's, and the
// LayerNorm and SwiGLU kernels are cuda/ef2's.
#pragma once
#include "../../ef2/src/ops.cuh"

namespace be2 {

constexpr int WQ = 32, WK = 128, CTX = (WK - WQ) / 2;   // atom query / key windows; keys centred on queries
constexpr float SIGMA_DATA = 16.f;
constexpr float NEG_INF = -1e6f;                         // the vendor's -INF (1e6): a masked logit, not -inf

// ---------------------------------------------------------------- weights
inline const std::string SCOPE = "w/diffuser/~/bioemu2/";
inline bool hasP(const std::string& n) { return M.has(SCOPE + n); }
inline const float* P(const std::string& n) { return W(SCOPE + n); }
inline int dimP(const std::string& n, int k) { return (int)M.meta(SCOPE + n + "#" + std::to_string(k)); }
inline size_t lenP(const std::string& n) { return M.len(SCOPE + n); }
// layer l of a stacked tensor ([layers, ...])
inline const float* PL(const std::string& n, int l) {
  return P(n) + (size_t)l * (lenP(n) / (size_t)dimP(n, 0));
}
inline const float* PLopt(const std::string& n, int l) { return hasP(n) ? PL(n, l) : nullptr; }
inline const float* Popt(const std::string& n) { return hasP(n) ? P(n) : nullptr; }

// ---------------------------------------------------------------- GEMM, row-major, torch weights
// Y[rows, out] = alpha X[rows, in] W[out, in]^T + beta Y   (ldx/ldy: row strides, for a slice of a wider row)
inline void linear(const float* X, const float* Wt, float* Y, size_t rows, int in, int out, float beta = 0.f,
                   int ldx = 0, int ldy = 0, float alpha = 1.f) {
  CB(cublasSgemm(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha, Wt, in, X, ldx ? ldx : in, &beta, Y,
                 ldy ? ldy : out));
}

// ---------------------------------------------------------------- elementwise
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__global__ void biasRowsK(float* y, const float* b, size_t rows, int C, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  y[(t / C) * ld + t % C] += b[t % C];
}
inline void addBiasRows(float* y, const float* b, size_t rows, int C, int ld = 0) {
  biasRowsK<<<blocks(rows * C), 256, 0, STREAM>>>(y, b, rows, C, ld ? ld : C);
}
__global__ void siluK(const float* x, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = siluF(x[t]);
}
__global__ void reluK(const float* x, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = fmaxf(x[t], 0.f);
}
// g = silu(h1) * h2 (two separate projections, fc1 and fc2)
__global__ void siluMulK(const float* h1, const float* h2, float* g, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) g[t] = siluF(h1[t]) * h2[t];
}
// x[r, c] += gate[r * ldg + c] * h[r, c]
__global__ void gatedAddK(float* x, const float* h, const float* gate, size_t rows, int C, int ldg) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = (int)(t % C);
  x[t] += gate[r * ldg + c] * h[t];
}
// y = h * sigmoid(g), elementwise
__global__ void sigmoidMulK(const float* h, const float* g, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = h[t] * sigm(g[t]);
}
// adaLN: y = LN(x) (1 + alpha) + beta, alpha and beta rows of a wider conditioning tensor (stride ldc), x's own
// stride ldx; a warp a row
__global__ void adaLnK(const float* x, float* y, size_t rows, int C, int ldx, const float* alpha, const float* beta,
                       int ldc, float scale) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * ldx;
  float s = 0; for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32)
    y[row * C + c] = scale * ((xr[c] - mean) * inv * (1.f + alpha[row * ldc + c]) + beta[row * ldc + c]);
}
inline void adaLn(const float* x, float* y, size_t rows, int C, const float* alpha, const float* beta, int ldc,
                  int ldx = 0, float scale = 1.f) {
  adaLnK<<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(x, y, rows, C, ldx ? ldx : C, alpha, beta, ldc, scale);
}
// rows of a token tensor broadcast to atoms (zero on padding atoms); and each token's centre atom (zero on
// padding tokens)
__global__ void tokensToAtomsK(const float* tok, const int* atomTok, const int* atomPad, float* out, size_t N, int C,
                               int ldo, float beta) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N * C) return;
  size_t a = t / C; int c = (int)(t % C);
  float v = atomPad[a] ? tok[(size_t)atomTok[a] * C + c] : 0.f;
  out[a * ldo + c] = beta == 0.f ? v : beta * out[a * ldo + c] + v;
}
inline void tokensToAtoms(const float* tok, float* out, size_t N, int C, int ldo = 0, float beta = 0.f) {
  tokensToAtomsK<<<blocks(N * C), 256, 0, STREAM>>>(tok, Idev("token_idx_at_atom"), Idev("atom_pad_mask"), out, N, C,
                                                    ldo ? ldo : C, beta);
}
__global__ void centreAtomsK(const float* atom, const int* centre, float* out, int T, int C, int ldo) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  size_t i = t / C; int c = (int)(t % C);
  out[i * ldo + c] = atom[(size_t)centre[i] * C + c];   // (every token here is real: no token padding)
}
inline void centreAtoms(const float* atom, float* out, int T, int C, int ldo = 0) {
  centreAtomsK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(atom, Idev("token_center_atom_index"), out, T, C, ldo ? ldo : C);
}

// ---------------------------------------------------------------- the SwiGLU transition
// out = fc3(silu(fc1 x') * fc2 x'), x' = LN(x) when the module has a norm; with `residual` the output is added to
// `out` (beta 1). A chunk of rows at a time: the widened rows are 2I a row.
inline void transition(const float* x, float* out, size_t rows, int C, const std::string& p, int layer = -1,
                       bool residual = true, bool norm = true, int ldx = 0) {
  auto w = [&](const std::string& n) { return layer < 0 ? P(p + n) : PL(p + n, layer); };
  const int I = layer < 0 ? dimP(p + "fc1/weight", 0) : dimP(p + "fc1/weight", 1);
  const int Co = layer < 0 ? dimP(p + "fc3/weight", 0) : dimP(p + "fc3/weight", 1);   // (a mixer narrows: 896 -> 128)
  const float* xin = x; int ld = ldx ? ldx : C;
  if (norm) {
    float* xn = scratch<float>("be2.tr.xn", rows * C);
    layerNorm(x, xn, rows, C, w("norm/weight"), w("norm/bias"), 1e-5f, ld, C);
    xin = xn; ld = C;
  }
  size_t chunk = std::max<size_t>(1, ((size_t)96 << 20) / (4 * (size_t)I * 3));
  float* h1 = scratch<float>("be2.tr.h1", std::min(rows, chunk) * I);
  float* h2 = scratch<float>("be2.tr.h2", std::min(rows, chunk) * I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    linear(xin + r0 * ld, w("fc1/weight"), h1, r, C, I, 0.f, ld);
    linear(xin + r0 * ld, w("fc2/weight"), h2, r, C, I, 0.f, ld);
    siluMulK<<<blocks(r * I), 256, 0, STREAM>>>(h1, h2, h1, r * I);
    linear(h1, w("fc3/weight"), out + r0 * Co, r, I, Co, residual ? 1.f : 0.f);
  }
}

// ---------------------------------------------------------------- the relative position encoding
// FusedRelativePositionEncoder over the token index (arange), into `out` (+= when add)
struct RelTables { const float *pos, *tok, *ent, *chain; };
inline RelTables relTables(const std::string& p) {
  return {P(p + "embed_rel_pos/weight"), P(p + "embed_rel_token/weight"), P(p + "embed_same_entity/weight"),
          P(p + "embed_rel_chain/weight")};
}
__global__ void relPosK(RelTables w, const int* asym, const int* ent, const int* res, const int* sym, float* out,
                        int T, int C, bool add, int ldo) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  const int rmax = 32, smax = 2;
  auto clip = [](int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; };
  bool sameChain = asym[i] == asym[j], sameRes = res[i] == res[j], sameEnt = ent[i] == ent[j];
  int dRes = sameChain ? clip(res[i] - res[j] + rmax, 0, 2 * rmax) : 2 * rmax + 1;
  int dTok = sameChain && sameRes ? clip(i - j + rmax, 0, 2 * rmax) : 2 * rmax + 1;
  int dChain = sameEnt ? clip(sym[i] - sym[j] + smax, 0, 2 * smax) : 2 * smax + 1;
  float v = w.pos[(size_t)dRes * C + c] + w.tok[(size_t)dTok * C + c] + (sameEnt ? w.ent[c] : 0.f)
            + w.chain[(size_t)dChain * C + c];
  float* o = out + ij * ldo + c;
  *o = add ? *o + v : v;
}
inline void relPos(const std::string& p, float* out, int T, int C, bool add, int ldo = 0) {
  relPosK<<<blocks((size_t)T * T * C), 256, 0, STREAM>>>(relTables(p), Idev("asym_id"), Idev("entity_id"),
      Idev("residue_index"), Idev("sym_id"), out, T, C, add, ldo ? ldo : C);
}

}  // namespace be2
