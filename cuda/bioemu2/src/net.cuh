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
// GEMM_HALF (the sampler under --fast): X rounded to f16 and the weight read from its f16 mirror, f32 accumulation
// and output - the tensor cores' f16 rate, twice TF32's. Every input it takes is a normalised activation (LN or adaLN)
// or an attention/SwiGLU product of them, far inside f16's range.
// 🔴 EXCEPT A SwiGLU PRODUCT, WHICH IS UNBOUNDED: silu(fc1 x) fc2 x reaches 87,750 in the atom stream's transition at
// 6MRR (f16 stops at 65,504, and the fold came back NaN), so fc3 takes bf16 - f32's range, 8 bits of mantissa
// (`wide`).
inline bool GEMM_HALF = false;
template <class TH> inline std::map<const float*, TH*>& mirrors() { static std::map<const float*, TH*> m; return m; }
__global__ void toBf16K(const float* x, __nv_bfloat16* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = __float2bfloat16(x[t]);
}
template <class TH> inline const TH* halfMirror(const float* w, size_t n) {
  auto& m = mirrors<TH>();
  auto it = m.find(w);
  if (it != m.end()) return it->second;
  TH* h = dallocT<TH>(n);
  if constexpr (std::is_same_v<TH, half>) toHalfK<<<blocks(n), 256, 0, STREAM>>>(w, h, n);
  else toBf16K<<<blocks(n), 256, 0, STREAM>>>(w, h, n);
  return m[w] = h;
}
template <class TH> __global__ void toHalfRowsK(const float* x, TH* y, size_t rows, int in, int ldx) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * in) return;
  const float v = x[(t / in) * ldx + t % in];
  // (saturated: a padding atom's state, which nothing reads, grows past f16 and would be inf)
  if constexpr (std::is_same_v<TH, half>) y[t] = __float2half(fminf(fmaxf(v, -65504.f), 65504.f)); else y[t] = __float2bfloat16(v);
}
template <class TH> inline void linearHalf(const float* X, const float* Wt, float* Y, size_t rows, int in, int out,
                                           float beta, int ldx, int ldy, float alpha) {
  const TH* wh = halfMirror<TH>(Wt, (size_t)in * out);
  TH* xh = scratch<TH>(std::is_same_v<TH, half> ? "be2.gemm.xh" : "be2.gemm.xb", rows * in);
  toHalfRowsK<TH><<<blocks(rows * in), 256, 0, STREAM>>>(X, xh, rows, in, ldx ? ldx : in);
  const cudaDataType t = std::is_same_v<TH, half> ? CUDA_R_16F : CUDA_R_16BF;
  CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha, wh, t, in, xh, t, in,
                  &beta, Y, CUDA_R_32F, ldy ? ldy : out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
inline void linear(const float* X, const float* Wt, float* Y, size_t rows, int in, int out, float beta = 0.f,
                   int ldx = 0, int ldy = 0, float alpha = 1.f, bool wide = false) {
  if (GEMM_HALF && in >= 16 && out >= 16) {
    if (wide) linearHalf<__nv_bfloat16>(X, Wt, Y, rows, in, out, beta, ldx, ldy, alpha);
    else linearHalf<half>(X, Wt, Y, rows, in, out, beta, ldx, ldy, alpha);
    return;
  }
  CB(cublasSgemm(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha, Wt, in, X, ldx ? ldx : in, &beta, Y,
                 ldy ? ldy : out));
}
// ...from an input already in f16 / bf16 (written so by the kernel that made it: halfLn, adaLnTo, sigmoidMulTo,
// swigluTo), so no conversion pass runs between
template <class TH> inline void linearPre(const TH* Xh, const TH* Wh, float* Y, size_t rows, int in, int out,
                                          float beta = 0.f, int ldy = 0) {
  const float alpha = 1.f;
  const cudaDataType t = std::is_same_v<TH, half> ? CUDA_R_16F : CUDA_R_16BF;
  CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha, Wh, t, in, Xh, t, in,
                  &beta, Y, CUDA_R_32F, ldy ? ldy : out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
// two torch-layout weights stacked as one: [W1; W2] (rows), its half mirror - fc1 and fc2 as one GEMM
template <class TH> inline const TH* stackedMirror(const float* w1, const float* w2, size_t n) {
  static std::map<const float*, TH*> m;
  auto it = m.find(w1);
  if (it != m.end()) return it->second;
  TH* h = dallocT<TH>(2 * n);
  if constexpr (std::is_same_v<TH, half>) { toHalfK<<<blocks(n), 256, 0, STREAM>>>(w1, h, n); toHalfK<<<blocks(n), 256, 0, STREAM>>>(w2, h + n, n); }
  else { toBf16K<<<blocks(n), 256, 0, STREAM>>>(w1, h, n); toBf16K<<<blocks(n), 256, 0, STREAM>>>(w2, h + n, n); }
  return m[w1] = h;
}
template <class TH> __device__ __forceinline__ TH toH(float v) {
  if constexpr (std::is_same_v<TH, half>) return __float2half(fminf(fmaxf(v, -65504.f), 65504.f));
  else return __float2bfloat16(v);
}

// ---------------------------------------------------------------- elementwise
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
// LayerNorm into f16 (scale/offset may be null), a warp a row
template <class TH> __global__ void lnToK(const float* x, TH* y, size_t rows, int C, const float* scale, const float* offset,
                                          int ldx) {
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
  for (int c = lane; c < C; c += 32) {
    float n = (xr[c] - mean) * inv;
    if (scale) n *= scale[c];
    if (offset) n += offset[c];
    y[row * C + c] = toH<TH>(n);
  }
}
template <class TH> inline void lnTo(const float* x, TH* y, size_t rows, int C, const float* scale, const float* offset, int ldx = 0) {
  lnToK<TH><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(x, y, rows, C, scale, offset, ldx ? ldx : C);
}
// the SwiGLU's product from fc1|fc2's [rows, 2I] f32 output, into fc3's bf16 input
__global__ void swigluToK(const half* h12, __nv_bfloat16* g, size_t rows, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int c = (int)(t % I);
  g[t] = __float2bfloat16(siluF(__half2float(h12[r * 2 * I + c])) * __half2float(h12[r * 2 * I + I + c]));
}
// y = h sigmoid(g), into f16
__global__ void sigmoidMulToK(const float* h, const float* g, half* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = toH<half>(h[t] / (1.f + expf(-g[t])));
}
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
// x[r, c] += gate[map(r) * ldg + c] * h[r, c]   (map null: the identity)
__global__ void gatedAddK(float* x, const float* h, const float* gate, size_t rows, int C, int ldg, const int* map = nullptr) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = (int)(t % C);
  size_t g = map ? (size_t)map[r] : r;
  x[t] += gate[g * ldg + c] * h[t];
}
inline void gatedAdd(float* x, const float* h, const float* gate, size_t rows, int C, int ldg, const int* map) {
  gatedAddK<<<blocks(rows * C), 256, 0, STREAM>>>(x, h, gate, rows, C, ldg, map);
}
// y = h * sigmoid(g), elementwise
__global__ void sigmoidMulK(const float* h, const float* g, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = h[t] * sigm(g[t]);
}
// adaLN: y = LN(x) (1 + alpha) + beta, alpha and beta rows of a wider conditioning tensor (stride ldc), x's own
// stride ldx; a warp a row
template <class TY>
__global__ void adaLnK(const float* x, TY* y, size_t rows, int C, int ldx, const float* alpha, const float* beta,
                       int ldc, float scale, const int* map) {
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
  const size_t cr = map ? (size_t)map[row] : row;
  for (int c = lane; c < C; c += 32) {
    const float v2 = scale * ((xr[c] - mean) * inv * (1.f + alpha[cr * ldc + c]) + beta[cr * ldc + c]);
    if constexpr (std::is_same_v<TY, float>) y[row * C + c] = v2; else y[row * C + c] = toH<TY>(v2);
  }
}
template <class TY = float>
inline void adaLn(const float* x, TY* y, size_t rows, int C, const float* alpha, const float* beta, int ldc,
                  int ldx = 0, float scale = 1.f, const int* map = nullptr) {
  adaLnK<TY><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(x, y, rows, C, ldx ? ldx : C, alpha, beta, ldc, scale, map);
}
// rows of a token tensor broadcast to atoms (zero on padding atoms); and each token's centre atom (zero on
// padding tokens)
// (several samples stacked: `samples` blocks of N atoms and of T tokens)
__global__ void tokensToAtomsK(const float* tok, const int* atomTok, const int* atomPad, float* out, size_t N, int T,
                               int C, int ldo, float beta, int samples) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)samples * N * C) return;
  size_t a = t / C, s = a / N, al = a % N; int c = (int)(t % C);
  float v = atomPad[al] ? tok[(s * T + atomTok[al]) * C + c] : 0.f;
  out[a * ldo + c] = beta == 0.f ? v : beta * out[a * ldo + c] + v;
}
inline void tokensToAtoms(const float* tok, float* out, size_t N, int C, int ldo = 0, float beta = 0.f, int samples = 1,
                          int T = 0) {
  tokensToAtomsK<<<blocks((size_t)samples * N * C), 256, 0, STREAM>>>(tok, Idev("token_idx_at_atom"),
      Idev("atom_pad_mask"), out, N, T, C, ldo ? ldo : C, beta, samples);
}
__global__ void centreAtomsK(const float* atom, const int* centre, float* out, int T, int N, int C, int ldo, int samples) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)samples * T * C) return;
  size_t i = t / C, s = i / T, il = i % T; int c = (int)(t % C);
  out[i * ldo + c] = atom[(s * N + centre[il]) * C + c];   // (every token here is real: no token padding)
}
inline void centreAtoms(const float* atom, float* out, int T, int C, int ldo = 0, int samples = 1, int N = 0) {
  centreAtomsK<<<blocks((size_t)samples * T * C), 256, 0, STREAM>>>(atom, Idev("token_center_atom_index"), out, T, N, C,
                                                                    ldo ? ldo : C, samples);
}

// ---------------------------------------------------------------- the SwiGLU transition
// out = fc3(silu(fc1 x') * fc2 x'), x' = LN(x) when the module has a norm; with `residual` the output is added to
// `out` (beta 1). A chunk of rows at a time: the widened rows are 2I a row.
// GEMM_HALF: the input in f16 (from the caller's kernel, `xh`, or normed/converted here), fc1 | fc2 as ONE GEMM over
// their stacked mirror, the SwiGLU into fc3's bf16 input
inline void transitionHalf(const float* x, const half* xh, float* out, size_t rows, int C, const std::string& p, int layer,
                           bool residual, bool norm, int ldx) {
  auto w = [&](const std::string& n) { return layer < 0 ? P(p + n) : PL(p + n, layer); };
  const int I = layer < 0 ? dimP(p + "fc1/weight", 0) : dimP(p + "fc1/weight", 1);
  const int Co = layer < 0 ? dimP(p + "fc3/weight", 0) : dimP(p + "fc3/weight", 1);
  const half* w12 = stackedMirror<half>(w("fc1/weight"), w("fc2/weight"), (size_t)I * C);
  const __nv_bfloat16* w3 = halfMirror<__nv_bfloat16>(w("fc3/weight"), (size_t)Co * I);
  // (a chunk sized for the L2 - 16 MB of widened rows - measured 8% SLOWER at 494 tokens than this: the GEMMs lose more
  // to their smaller M than the SwiGLU gains from the cache)
  size_t chunk = std::max<size_t>(1, ((size_t)128 << 20) / (4 * (size_t)I * 2));
  if (!xh) {
    half* c = scratch<half>("be2.tr.xh", rows * C);
    if (norm) lnTo<half>(x, c, rows, C, w("norm/weight"), w("norm/bias"), ldx);
    else toHalfRowsK<half><<<blocks(rows * C), 256, 0, STREAM>>>(x, c, rows, C, ldx ? ldx : C);
    xh = c;
  }
  // (fc1 | fc2's output in bf16: the SwiGLU reads it once and it is the widest tensor of the layer - written f32 it was
  // 9 ms of an 82 ms step at 494 tokens)
  half* h12 = scratch<half>("be2.tr.h12", std::min(rows, chunk) * 2 * I);
  __nv_bfloat16* g = scratch<__nv_bfloat16>("be2.tr.g", std::min(rows, chunk) * I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    {
      const float one = 1.f, zero = 0.f;
      CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, 2 * I, (int)r, C, &one, w12, CUDA_R_16F, C, xh + r0 * C, CUDA_R_16F, C,
                      &zero, h12, CUDA_R_16F, 2 * I, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      if (getenv("BE2_DEBUG_HALF")) {
        CK(cudaStreamSynchronize(STREAM));
        std::vector<half> hh(r * 2 * I); CK(cudaMemcpy(hh.data(), h12, hh.size() * 2, cudaMemcpyDeviceToHost));
        float mx = 0; for (auto v : hh) mx = fmaxf(mx, fabsf(__half2float(v)));
        static float worst = 0;
        if (mx > worst) { worst = mx; printf("fc1|fc2 max %g (%s)\n", mx, p.c_str()); }
      }
    }
    swigluToK<<<blocks(r * I), 256, 0, STREAM>>>(h12, g, r, I);
    linearPre<__nv_bfloat16>(g, w3, out + r0 * Co, r, I, Co, residual ? 1.f : 0.f);
  }
}
inline void transition(const float* x, float* out, size_t rows, int C, const std::string& p, int layer = -1,
                       bool residual = true, bool norm = true, int ldx = 0) {
  if (GEMM_HALF) { transitionHalf(x, nullptr, out, rows, C, p, layer, residual, norm, ldx); return; }
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
    linear(h1, w("fc3/weight"), out + r0 * Co, r, I, Co, residual ? 1.f : 0.f, 0, 0, 1.f, true);
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
