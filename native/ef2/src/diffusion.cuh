// ESMFold2's diffusion module (src/esmfold2/diffusion-reference.js and biohub's DiffusionModule are the
// reading), one denoiser call:
//
//   conditioning: pair = [z_trunk | rel_pos] -> LN -> zProjection, + 2 transitions   (once a fold)
//                 single = LN(s_inputs) @ sProjection + noise(t), + 2 transitions     (every step)
//                 noise(t) = LN(cos(2 pi (t_noise w + b))) @ noiseProjection, t_noise = log(t / sigma) / 4
//   r = x_noisy / sqrt(t^2 + sigma^2);  q = c0 + [r | 0] @ coordsLinear;  q = 3 windowed SWA blocks (c0)
//   a = pool(relu(q @ toToken)) + LN(single; stepNorm) @ singleToToken
//   12 token blocks:  a += attention(adaLN(a, single), pair bias) * sigmoid(single @ outGate + b)
//                     a += swiglu(adaLN(a, single)) * sigmoid(single @ outGate + b)
//   a = LN(a; tokenNorm);  q = skip + gather(a @ tokenToAtom);  q = 3 windowed SWA blocks (c0)
//   x_denoised = sigma^2 / (sigma^2 + t^2) x_noisy + sigma t / sqrt(sigma^2 + t^2) * (LN(q) @ outputLinear)
#pragma once
#include "atoms.cuh"

constexpr int DIFFUSION_HALF_WINDOW = 64;     // the diffusion's atom stacks are windowed (128) in both references

// rows [p0, p0 + n) of [z | rel_pos]
__global__ void joinPairRelK(const float* z, RelIdx rel, float* out, size_t p0, size_t n, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * 2 * C) return;
  size_t p = p0 + t / (2 * C); int c = (int)(t % (2 * C));
  out[t] = c < C ? z[p * C + c] : relPosAt(rel, (int)(p / T), (int)(p % T), c - C, C);
}
__global__ void siluMulK(const float* a, const float* b, float* out, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = siluF(a[t]) * b[t];
}
// TransitionLayer: x += out(silu(LN(x) @ a) * (LN(x) @ b)), over rows in chunks
inline void transitionLayer(float* x, size_t rows, int C, const std::string& B) {
  int Hd = (int)dimOf("f/" + B + "aProjection", 1);
  size_t chunk = std::max<size_t>(1, ((size_t)32 << 20) / (3 * (size_t)Hd));
  float* xn = scratch<float>("dtr.xn", std::min(rows, chunk) * C);
  float* a = scratch<float>("dtr.a", std::min(rows, chunk) * Hd); float* b = scratch<float>("dtr.b", std::min(rows, chunk) * Hd);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    layerNorm(x + r0 * C, xn, r, C, F(B + "norm/scale"), F(B + "norm/offset"));
    gemm(xn, F(B + "aProjection"), a, r, C, Hd);
    gemm(xn, F(B + "bProjection"), b, r, C, Hd);
    siluMulK<<<blocks(r * Hd), 256, 0, STREAM>>>(a, b, a, r * Hd);
    gemm(a, F(B + "outProjection"), x + r0 * C, r, Hd, C, 1.f);
  }
}
// the noise level's scalars, on the device so one captured step serves every level:
// [0] t_noise = log(t / sigma) / 4, [1] 1 / sqrt(t^2 + sigma^2), [2] sigma^2 / (sigma^2 + t^2) (keep),
// [3] sigma t / sqrt(sigma^2 + t^2) (take)
struct NoiseLevel { float v[4]; };
inline NoiseLevel noiseLevel(float t, float sigma) {
  float s2 = sigma * sigma, t2 = t * t;
  return {{0.25f * logf(fmaxf(t / sigma, 1e-20f)), 1.f / sqrtf(t2 + s2), s2 / (s2 + t2), sigma * t / sqrtf(s2 + t2)}};
}
__global__ void fourierK(const float* w, const float* b, const float* level, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cosf(2.f * 3.14159265358979323846f * (level[0] * w[i] + b[i]));
}
__global__ void addRowK(float* x, const float* row, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] += row[t % C];
}
__global__ void coordsInputK(const float* x, const float* level, float* out, int A) {   // [A, 6] = [x / denom | 0]
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= A * 6) return;
  int a = t / 6, k = t % 6;
  out[t] = k < 3 ? x[a * 3 + k] * level[1] : 0.f;
}
// adaLN: sigmoid(LN(s; scale) @ gate + gateBias) * LN(a) + LN(s; scale) @ shift
// (g and sh rows ld apart)
__global__ void adaCombineK(const float* an, const float* g, const float* gb, const float* sh, float* out, size_t T, int C,
                            int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T * C) return;
  size_t gi = (t / C) * ld + t % C;
  out[t] = an[t] / (1.f + expf(-(g[gi] + gb[t % C]))) + sh[gi];
}
__global__ void sigmoidMulK(float* x, const float* g, const float* gb, size_t T, int C, int ld) {   // x *= sigmoid(g (+ gb))
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T * C) return;
  x[t] *= 1.f / (1.f + expf(-(g[(t / C) * ld + t % C] + (gb ? gb[t % C] : 0.f))));
}
// out[j] = x * scale[j] for each of n scales, [n, rows, C]
__global__ void scaleCopiesK(const float* x, const float* scales, float* out, size_t rows, int C, int n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * rows * C) return;
  size_t per = rows * C; int j = (int)(t / per); size_t r = t % per;
  out[t] = x[r] * scales[(size_t)j * C + r % C];
}
// scores [H, T, T] + bias [H, T, T], softmax per row
template <class B>
__global__ void biasSoftmaxK(float* S, const B* bias, int T, float scale) {
  size_t row = blockIdx.x;
  float* s = S + row * T; const B* b = bias + row * T;
  __shared__ float red[32];
  float m = -INFINITY;
  for (int j = threadIdx.x; j < T; j += blockDim.x) { float v = s[j] * scale + (float)b[j]; s[j] = v; m = fmaxf(m, v); }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  m = red[0];
  __syncthreads();
  float sum = 0;
  for (int j = threadIdx.x; j < T; j += blockDim.x) { float e = expf(s[j] - m); s[j] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < T; j += blockDim.x) s[j] *= inv;
}
template <class B>
__global__ void pairToHeadsK(const float* pb, B* out, size_t P, int Hh) {   // [P, H] -> [H, P]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < P * Hh) out[(t % Hh) * P + t / Hh] = (B)pb[t];
}
__global__ void gatherTokensK(const float* perToken, const int* atomToToken, const float* mask, float* q, int A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * C) return;
  int a = (int)(t / C), c = (int)(t % C);
  int token = mask[a] != 0.f ? atomToToken[a] : 0;
  q[t] += perToken[(size_t)token * C + c];
}
__global__ void edmCombineK(const float* xNoisy, const float* r, float* out, const float* level, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = level[2] * xNoisy[t] + level[3] * r[t];
}

struct Denoiser {
  int T, A, Cz, Ct, heads, tokenBlocks, Si, atomBlocks;
  float sigma;
  Atoms atoms;                    // the diffusion encoder's c0, rope table, ranks
  // per token block, [H, T, T] (the pair itself is not kept): f16 under --fast, f32 otherwise
  // (f16 costs the float32 path 5e-5 on the denoiser)
  std::vector<void*> biases; bool biasHalf;
  const float* sInputs;
  // every token block's projections of the single alone, one batched GEMM a step: entry e of G [T, 72 Ct]
  // (columns e Ct..): 4b+0..3 block b's attention gate, shift, transition gate, shift (from LN(single)
  // scaled by that adaLN's singleScale: snScaled [2 blocks, T, Ct]); 48+2b, 48+2b+1 its two out gates
  float* single; float* snScaled; float* G; float* scales; const void** ptrs; int entries;
  float* level;                   // NoiseLevel on the device
};

inline Denoiser makeDenoiser(int T, int A, const float* zTrunk, const float* sInputs, bool check) {
  Denoiser d{};
  d.T = T; d.A = A; d.Cz = (int)M.meta("meta/pairChannels"); d.Ct = (int)M.meta("meta/tokenChannels2");
  d.heads = (int)M.meta("meta/tokenHeads"); d.tokenBlocks = (int)M.meta("meta/tokenBlocks");
  d.Si = (int)M.meta("meta/singleInputs"); d.atomBlocks = (int)M.meta("meta/atomBlocks"); d.sigma = (float)M.meta("meta/sigmaData");
  d.sInputs = sInputs;
  size_t P = (size_t)T * T; int Cz = d.Cz;
  // row chunks of 64 MB: neither [z | rel_pos] nor a normalised copy of the pair is ever whole
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (8 * (size_t)Cz));
  float* joined = scratch<float>("dc.joined", chunk * 2 * Cz);
  float* pair = scratch<float>("dc.pair", P * Cz);
  for (size_t p0 = 0; p0 < P; p0 += chunk) {
    size_t n = std::min(chunk, P - p0);
    joinPairRelK<<<blocks(n * 2 * Cz), 256, 0, STREAM>>>(zTrunk, relIdx(), joined, p0, n, T, Cz);
    layerNorm(joined, joined, n, 2 * Cz, F("diffusion/zInputNorm/scale"), F("diffusion/zInputNorm/offset"));
    gemm(joined, F("diffusion/zProjection"), pair + p0 * Cz, n, 2 * Cz, Cz);
  }
  for (int l = 0; l < 2; ++l) transitionLayer(pair, P, Cz, "diffusion/zTransitions/" + std::to_string(l) + "/");
  if (check) checkOracle("diffusion conditioning pair", pair, P * Cz, "o/cond/pair");
  d.biasHalf = FAST;
  float* pn = scratch<float>("dc.pn", chunk * Cz); float* pb = scratch<float>("dc.pb", P * d.heads);
  for (int b = 0; b < d.tokenBlocks; ++b) {
    std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/attention/";
    for (size_t p0 = 0; p0 < P; p0 += chunk) {
      size_t n = std::min(chunk, P - p0);
      layerNorm(pair + p0 * Cz, pn, n, Cz, F(B + "pairNormScale"), F(B + "pairNormOffset"));
      gemm(pn, F(B + "pairBiasWeights"), pb + p0 * d.heads, n, Cz, d.heads);
    }
    void* bias;
    if (d.biasHalf) { half* h = dallocT<half>(P * d.heads); pairToHeadsK<<<blocks(P * d.heads), 256, 0, STREAM>>>(pb, h, P, d.heads); bias = h; }
    else { float* f = dalloc(P * d.heads); pairToHeadsK<<<blocks(P * d.heads), 256, 0, STREAM>>>(pb, f, P, d.heads); bias = f; }
    d.biases.push_back(bias);
  }
  if ((size_t)P * Cz * 4 > ((size_t)128 << 20)) releaseScratch();     // the joined and projected pairs (a large input)
  d.atoms = prepareAtoms(A, "diffusionAtomEncoder");
  int Ct = d.Ct, nb = d.tokenBlocks;
  d.entries = 6 * nb;
  d.single = dalloc((size_t)T * Ct); d.snScaled = dalloc((size_t)2 * nb * T * Ct);
  d.G = dalloc((size_t)T * d.entries * Ct); d.scales = dalloc((size_t)2 * nb * Ct);
  std::vector<const void*> w(d.entries), x(d.entries), y(d.entries);
  for (int b = 0; b < nb; ++b) {
    std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/";
    const char* part[2] = {"attention/", "transition/"};
    for (int k = 0; k < 2; ++k) {
      std::string P = B + part[k];
      CK(cudaMemcpyAsync(d.scales + (size_t)(2 * b + k) * Ct, F(P + "adaln/singleScale"), Ct * 4, cudaMemcpyDeviceToDevice, STREAM));
      const float* in = d.snScaled + (size_t)(2 * b + k) * T * Ct;
      w[4 * b + 2 * k] = F(P + "adaln/gateWeights"); x[4 * b + 2 * k] = in;
      w[4 * b + 2 * k + 1] = F(P + "adaln/shiftWeights"); x[4 * b + 2 * k + 1] = in;
      w[4 * nb + 2 * b + k] = F(P + "outGateWeights"); x[4 * nb + 2 * b + k] = d.single;
    }
  }
  for (int e = 0; e < d.entries; ++e) y[e] = d.G + (size_t)e * Ct;
  std::vector<const void*> all(w); all.insert(all.end(), x.begin(), x.end()); all.insert(all.end(), y.begin(), y.end());
  d.ptrs = (const void**)upload((const uintptr_t*)all.data(), all.size());
  d.level = dalloc(4);
  return d;
}

// single [T, Ct] at noise level t
inline void conditioningSingle(const Denoiser& d, float* single) {
  int T = d.T, Ct = d.Ct;
  float* sn = scratch<float>("dc.sn", (size_t)T * d.Si);
  layerNorm(d.sInputs, sn, T, d.Si, F("diffusion/sInputNorm/scale"), F("diffusion/sInputNorm/offset"));
  gemm(sn, F("diffusion/sProjection"), single, T, d.Si, Ct);
  int nf = (int)M.len("f/diffusion/fourier/weights");
  float* four = scratch<float>("dc.fourier", nf); float* noise = scratch<float>("dc.noise", Ct);
  fourierK<<<blocks(nf), 256, 0, STREAM>>>(F("diffusion/fourier/weights"), F("diffusion/fourier/offsets"), d.level, four, nf);
  layerNorm(four, four, 1, nf, F("diffusion/noiseNorm/scale"), F("diffusion/noiseNorm/offset"));
  gemm(four, F("diffusion/noiseProjection"), noise, 1, nf, Ct);
  addRowK<<<blocks((size_t)T * Ct), 256, 0, STREAM>>>(single, noise, T, Ct);
  for (int l = 0; l < 2; ++l) transitionLayer(single, T, Ct, "diffusion/sTransitions/" + std::to_string(l) + "/");
}

// the single's projections for every token block, one batched GEMM
inline void singleProjections(const Denoiser& d) {
  int T = d.T, C = d.Ct, n = 2 * d.tokenBlocks;
  float* sn = scratch<float>("dn.sn0", (size_t)T * C);
  layerNorm(d.single, sn, T, C, nullptr, nullptr);
  scaleCopiesK<<<blocks((size_t)n * T * C), 256, 0, STREAM>>>(sn, d.scales, d.snScaled, T, C, n);
  const float one = 1.f, zero = 0.f;
  cublasComputeType_t ct = GEMM16 ? CUBLAS_COMPUTE_32F_FAST_16F : FAST ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  CB(cublasGemmBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, C, T, C, &one, d.ptrs, CUDA_R_32F, C, d.ptrs + d.entries,
                         CUDA_R_32F, C, &zero, (void* const*)(d.ptrs + 2 * d.entries), CUDA_R_32F, d.entries * C,
                         d.entries, ct, FAST ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
}
// adaLN: sigmoid(LN(s; scale) @ gate + gateBias) * LN(a) + LN(s; scale) @ shift, the projections from G
inline void adaLN(const Denoiser& d, const float* a, float* out, int e, const std::string& B) {
  int T = d.T, C = d.Ct, ld = d.entries * C;
  float* an = scratch<float>("ada.an", (size_t)T * C);
  layerNorm(a, an, T, C, nullptr, nullptr);
  adaCombineK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(an, d.G + (size_t)e * C, F(B + "gateBias"),
                                                        d.G + (size_t)(e + 1) * C, out, T, C, ld);
}

inline void tokenBlock(const Denoiser& d, float* a, int b) {
  int T = d.T, C = d.Ct, Hh = d.heads, D = C / Hh, ld = d.entries * C, nb = d.tokenBlocks;
  std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/";
  float* x = scratch<float>("tb.x", (size_t)T * C);
  adaLN(d, a, x, 4 * b, B + "attention/adaln/");
  float* q = scratch<float>("tb.q", (size_t)T * C); float* kv = scratch<float>("tb.kv", (size_t)T * 2 * C);
  float* gt = scratch<float>("tb.gate", (size_t)T * C);
  gemm(x, F(B + "attention/queryWeights"), q, T, C, C);
  addBias(q, F(B + "attention/queryBias"), T, C);
  gemm(x, F(B + "attention/kvWeights"), kv, T, C, 2 * C);
  gemm(x, F(B + "attention/gateWeights"), gt, T, C, C);
  float* S = scratch<float>("tb.scores", (size_t)Hh * T * T); float* ctx = scratch<float>("tb.ctx", (size_t)T * C);
  const float one = 1.f, zero = 0.f;
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, T, T, D, &one, kv, 2 * C, D, q, C, D, &zero, S, T,
                               (long long)T * T, Hh));
  if (d.biasHalf) biasSoftmaxK<<<(unsigned)(Hh * T), 256, 0, STREAM>>>(S, (const half*)d.biases[b], T, 1.f / sqrtf((float)D));
  else biasSoftmaxK<<<(unsigned)(Hh * T), 256, 0, STREAM>>>(S, (const float*)d.biases[b], T, 1.f / sqrtf((float)D));
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, D, T, T, &one, kv + C, 2 * C, D, S, T, (long long)T * T,
                               &zero, ctx, C, D, Hh));
  sigmoidMulK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(ctx, gt, nullptr, T, C, C);
  float* o = scratch<float>("tb.o", (size_t)T * C);
  gemm(ctx, F(B + "attention/outWeights"), o, T, C, C);
  sigmoidMulK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(o, d.G + (size_t)(4 * nb + 2 * b) * C, F(B + "attention/outGateBias"), T, C, ld);
  addK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(a, o, (size_t)T * C);
  // the conditioned transition
  adaLN(d, a, x, 4 * b + 2, B + "transition/adaln/");
  int Hd = (int)dimOf("f/" + B + "transition/outWeights", 0);
  float* w = scratch<float>("tb.wide", (size_t)T * 2 * Hd); float* g = scratch<float>("tb.gated", (size_t)T * Hd);
  gemm(x, F(B + "transition/swishWeights"), w, T, C, 2 * Hd);
  swigluK<<<blocks((size_t)T * Hd), 256, 0, STREAM>>>(w, g, T, Hd);
  gemm(g, F(B + "transition/outWeights"), o, T, Hd, C);
  sigmoidMulK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(o, d.G + (size_t)(4 * nb + 2 * b + 1) * C, F(B + "transition/outGateBias"), T, C, ld);
  addK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(a, o, (size_t)T * C);
}

// one denoiser call: x_noisy [A, 3] at the noise level in d.level -> x_denoised [A, 3]; nothing here
// reads the host, so the sampler captures it once and replays it at every level
inline bool SAMPLER16 = true;     // --fast: the denoiser's GEMMs on f16 tensor cores (--no-sampler16: TF32)
inline void denoiseAtLevel(const Denoiser& d, const float* xNoisy, float* xDenoised, bool check = false) {
  struct Restore { bool was = GEMM16; ~Restore() { GEMM16 = was; } } restore;
  GEMM16 = FAST && SAMPLER16;
  int T = d.T, A = d.A, Ct = d.Ct; const AtomCtx& ac = d.atoms.ctx; int Ca = ac.C;
  float* single = d.single;
  conditioningSingle(d, single);
  singleProjections(d);
  if (check) checkOracle("diffusion conditioning single", single, (size_t)T * Ct, "o/cond/single");
  float* r6 = scratch<float>("dn.r6", (size_t)A * 6);
  coordsInputK<<<blocks((size_t)A * 6), 256, 0, STREAM>>>(xNoisy, d.level, r6, A);
  float* q = scratch<float>("dn.q", (size_t)A * Ca);
  gemm(r6, F("diffusionAtomEncoder/coordsLinear"), q, A, 6, Ca);
  addK<<<blocks((size_t)A * Ca), 256, 0, STREAM>>>(q, d.atoms.c0, (size_t)A * Ca);
  swaStack(ac, q, d.atoms.c0, "diffusionAtomEncoder", d.atomBlocks, DIFFUSION_HALF_WINDOW);
  float* tok = scratch<float>("dn.tok", (size_t)A * Ct);
  gemm(q, F("diffusionAtomEncoder/toToken"), tok, A, Ca, Ct);
  reluK<<<blocks((size_t)A * Ct), 256, 0, STREAM>>>(tok, (size_t)A * Ct);
  float* a = scratch<float>("dn.a", (size_t)T * Ct);
  scatterMeanK<<<blocks((size_t)T * Ct), 256, 0, STREAM>>>(tok, ac.tokenStart, ac.tokenAtoms, ac.mask, a, T, Ct, Ct);
  float* sn = scratch<float>("dn.sn", (size_t)T * Ct); float* st = scratch<float>("dn.st", (size_t)T * Ct);
  layerNorm(single, sn, T, Ct, F("diffusion/stepNorm/scale"), F("diffusion/stepNorm/offset"));
  gemm(sn, F("diffusion/singleToToken"), st, T, Ct, Ct);
  addK<<<blocks((size_t)T * Ct), 256, 0, STREAM>>>(a, st, (size_t)T * Ct);
  for (int b = 0; b < d.tokenBlocks; ++b) tokenBlock(d, a, b);
  layerNorm(a, a, T, Ct, F("diffusion/tokenNorm/scale"), F("diffusion/tokenNorm/offset"));
  // the decoder: q (the encoder's output, the skip) + the tokens gathered back
  float* pt = scratch<float>("dn.pt", (size_t)T * Ca);
  gemm(a, F("diffusionAtomDecoder/tokenToAtom"), pt, T, Ct, Ca);
  gatherTokensK<<<blocks((size_t)A * Ca), 256, 0, STREAM>>>(pt, Idev("atom_to_token"), ac.mask, q, A, Ca);
  swaStack(ac, q, d.atoms.c0, "diffusionAtomDecoder", d.atomBlocks, DIFFUSION_HALF_WINDOW);
  layerNorm(q, q, A, Ca, F("diffusionAtomDecoder/norm/scale"), F("diffusionAtomDecoder/norm/offset"));
  float* r = scratch<float>("dn.r", (size_t)A * 3);
  gemm(q, F("diffusionAtomDecoder/outputLinear"), r, A, Ca, 3);
  edmCombineK<<<blocks((size_t)A * 3), 256, 0, STREAM>>>(xNoisy, r, xDenoised, d.level, A * 3);
}
inline void freeDenoiser(Denoiser& d) {
  CK(cudaStreamSynchronize(STREAM));
  for (void* b : d.biases) CK(cudaFree(b));
  for (void* p : {(void*)d.single, (void*)d.snScaled, (void*)d.G, (void*)d.scales, (void*)d.ptrs, (void*)d.level})
    CK(cudaFree(p));
  freeAtoms(d.atoms);
  d.biases.clear();
}
inline void setLevel(const Denoiser& d, float t) {
  NoiseLevel lv = noiseLevel(t, d.sigma);
  CK(cudaMemcpyAsync(d.level, lv.v, sizeof lv.v, cudaMemcpyHostToDevice, STREAM));
}
inline void denoise(const Denoiser& d, const float* xNoisy, float t, float* xDenoised, bool check = false) {
  setLevel(d, t);
  denoiseAtLevel(d, xNoisy, xDenoised, check);
}
