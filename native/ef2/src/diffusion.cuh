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

__global__ void joinPairRelK(const float* z, const float* rel, float* out, size_t P, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= P * 2 * C) return;
  size_t p = t / (2 * C); int c = (int)(t % (2 * C));
  out[t] = c < C ? z[p * C + c] : rel[p * C + c - C];
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
__global__ void fourierK(const float* w, const float* b, float t, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cosf(2.f * 3.14159265358979323846f * (t * w[i] + b[i]));
}
__global__ void addRowK(float* x, const float* row, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] += row[t % C];
}
__global__ void coordsInputK(const float* x, float scale, float* out, int A) {   // [A, 6] = [x / denom | 0]
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= A * 6) return;
  int a = t / 6, k = t % 6;
  out[t] = k < 3 ? x[a * 3 + k] * scale : 0.f;
}
// adaLN: sigmoid(LN(s; scale) @ gate + gateBias) * LN(a) + LN(s; scale) @ shift
__global__ void adaCombineK(const float* an, const float* g, const float* gb, const float* sh, float* out, size_t T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T * C) return;
  out[t] = an[t] / (1.f + expf(-(g[t] + gb[t % C]))) + sh[t];
}
__global__ void sigmoidMulK(float* x, const float* g, const float* gb, size_t T, int C) {   // x *= sigmoid(g (+ gb))
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T * C) return;
  x[t] *= 1.f / (1.f + expf(-(g[t] + (gb ? gb[t % C] : 0.f))));
}
// scores [H, T, T] + bias [H, T, T], softmax per row
__global__ void biasSoftmaxK(float* S, const float* bias, int T, float scale) {
  size_t row = blockIdx.x;
  float* s = S + row * T; const float* b = bias + row * T;
  __shared__ float red[32];
  float m = -INFINITY;
  for (int j = threadIdx.x; j < T; j += blockDim.x) { float v = s[j] * scale + b[j]; s[j] = v; m = fmaxf(m, v); }
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
__global__ void pairToHeadsK(const float* pb, float* out, size_t P, int Hh) {   // [P, H] -> [H, P]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < P * Hh) out[(t % Hh) * P + t / Hh] = pb[t];
}
__global__ void gatherTokensK(const float* perToken, const int* atomToToken, const float* mask, float* q, int A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * C) return;
  int a = (int)(t / C), c = (int)(t % C);
  int token = mask[a] != 0.f ? atomToToken[a] : 0;
  q[t] += perToken[(size_t)token * C + c];
}
__global__ void edmCombineK(const float* xNoisy, const float* r, float* out, float keep, float take, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = keep * xNoisy[t] + take * r[t];
}

struct Denoiser {
  int T, A, Cz, Ct, heads, tokenBlocks, Si, atomBlocks;
  float sigma;
  Atoms atoms;                    // the diffusion encoder's c0, rope table, ranks
  float* pair;                    // the conditioning's pair, [P, Cz]
  std::vector<float*> biases;     // per token block, [H, T, T]
  const float* sInputs;
};

inline Denoiser makeDenoiser(int T, int A, const float* zTrunk, const float* relPos, const float* sInputs, bool check) {
  Denoiser d{};
  d.T = T; d.A = A; d.Cz = (int)M.meta("meta/pairChannels"); d.Ct = (int)M.meta("meta/tokenChannels2");
  d.heads = (int)M.meta("meta/tokenHeads"); d.tokenBlocks = (int)M.meta("meta/tokenBlocks");
  d.Si = (int)M.meta("meta/singleInputs"); d.atomBlocks = (int)M.meta("meta/atomBlocks"); d.sigma = (float)M.meta("meta/sigmaData");
  d.sInputs = sInputs;
  size_t P = (size_t)T * T; int Cz = d.Cz;
  float* joined = scratch<float>("dc.joined", P * 2 * Cz);
  joinPairRelK<<<blocks(P * 2 * Cz), 256, 0, STREAM>>>(zTrunk, relPos, joined, P, Cz);
  layerNorm(joined, joined, P, 2 * Cz, F("diffusion/zInputNorm/scale"), F("diffusion/zInputNorm/offset"));
  d.pair = dalloc(P * Cz);
  gemm(joined, F("diffusion/zProjection"), d.pair, P, 2 * Cz, Cz);
  for (int l = 0; l < 2; ++l) transitionLayer(d.pair, P, Cz, "diffusion/zTransitions/" + std::to_string(l) + "/");
  if (check) checkOracle("diffusion conditioning pair", d.pair, P * Cz, "o/cond/pair");
  float* pn = scratch<float>("dc.pn", P * Cz); float* pb = scratch<float>("dc.pb", P * d.heads);
  for (int b = 0; b < d.tokenBlocks; ++b) {
    std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/attention/";
    layerNorm(d.pair, pn, P, Cz, F(B + "pairNormScale"), F(B + "pairNormOffset"));
    gemm(pn, F(B + "pairBiasWeights"), pb, P, Cz, d.heads);
    float* bias = dalloc(P * d.heads);
    pairToHeadsK<<<blocks(P * d.heads), 256, 0, STREAM>>>(pb, bias, P, d.heads);
    d.biases.push_back(bias);
  }
  d.atoms = prepareAtoms(A, "diffusionAtomEncoder");
  return d;
}

// single [T, Ct] at noise level t
inline void conditioningSingle(const Denoiser& d, float t, float* single) {
  int T = d.T, Ct = d.Ct;
  float* sn = scratch<float>("dc.sn", (size_t)T * d.Si);
  layerNorm(d.sInputs, sn, T, d.Si, F("diffusion/sInputNorm/scale"), F("diffusion/sInputNorm/offset"));
  gemm(sn, F("diffusion/sProjection"), single, T, d.Si, Ct);
  int nf = (int)M.len("f/diffusion/fourier/weights");
  float* four = scratch<float>("dc.fourier", nf); float* noise = scratch<float>("dc.noise", Ct);
  float tNoise = 0.25f * logf(fmaxf(t / d.sigma, 1e-20f));
  fourierK<<<blocks(nf), 256, 0, STREAM>>>(F("diffusion/fourier/weights"), F("diffusion/fourier/offsets"), tNoise, four, nf);
  layerNorm(four, four, 1, nf, F("diffusion/noiseNorm/scale"), F("diffusion/noiseNorm/offset"));
  gemm(four, F("diffusion/noiseProjection"), noise, 1, nf, Ct);
  addRowK<<<blocks((size_t)T * Ct), 256, 0, STREAM>>>(single, noise, T, Ct);
  for (int l = 0; l < 2; ++l) transitionLayer(single, T, Ct, "diffusion/sTransitions/" + std::to_string(l) + "/");
}

inline void adaLN(const float* a, const float* single, float* out, int T, int C, const std::string& B) {
  float* an = scratch<float>("ada.an", (size_t)T * C); float* sn = scratch<float>("ada.sn", (size_t)T * C);
  float* g = scratch<float>("ada.g", (size_t)T * C); float* sh = scratch<float>("ada.sh", (size_t)T * C);
  layerNorm(a, an, T, C, nullptr, nullptr);
  layerNorm(single, sn, T, C, F(B + "singleScale"), nullptr);
  gemm(sn, F(B + "gateWeights"), g, T, C, C);
  gemm(sn, F(B + "shiftWeights"), sh, T, C, C);
  adaCombineK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(an, g, F(B + "gateBias"), sh, out, T, C);
}

inline void tokenBlock(const Denoiser& d, float* a, const float* single, int b) {
  int T = d.T, C = d.Ct, Hh = d.heads, D = C / Hh;
  std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/";
  float* x = scratch<float>("tb.x", (size_t)T * C);
  adaLN(a, single, x, T, C, B + "attention/adaln/");
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
  biasSoftmaxK<<<(unsigned)(Hh * T), 256, 0, STREAM>>>(S, d.biases[b], T, 1.f / sqrtf((float)D));
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, D, T, T, &one, kv + C, 2 * C, D, S, T, (long long)T * T,
                               &zero, ctx, C, D, Hh));
  sigmoidMulK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(ctx, gt, nullptr, T, C);
  float* o = scratch<float>("tb.o", (size_t)T * C); float* og = scratch<float>("tb.og", (size_t)T * C);
  gemm(ctx, F(B + "attention/outWeights"), o, T, C, C);
  gemm(single, F(B + "attention/outGateWeights"), og, T, C, C);
  sigmoidMulK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(o, og, F(B + "attention/outGateBias"), T, C);
  addK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(a, o, (size_t)T * C);
  // the conditioned transition
  adaLN(a, single, x, T, C, B + "transition/adaln/");
  int Hd = (int)dimOf("f/" + B + "transition/outWeights", 0);
  float* w = scratch<float>("tb.wide", (size_t)T * 2 * Hd); float* g = scratch<float>("tb.gated", (size_t)T * Hd);
  gemm(x, F(B + "transition/swishWeights"), w, T, C, 2 * Hd);
  swigluK<<<blocks((size_t)T * Hd), 256, 0, STREAM>>>(w, g, T, Hd);
  gemm(g, F(B + "transition/outWeights"), o, T, Hd, C);
  gemm(single, F(B + "transition/outGateWeights"), og, T, C, C);
  sigmoidMulK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(o, og, F(B + "transition/outGateBias"), T, C);
  addK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(a, o, (size_t)T * C);
}

// one denoiser call: x_noisy [A, 3] at t -> x_denoised [A, 3]
inline void denoise(const Denoiser& d, const float* xNoisy, float t, float* xDenoised, bool check = false) {
  int T = d.T, A = d.A, Ct = d.Ct; const AtomCtx& ac = d.atoms.ctx; int Ca = ac.C;
  float* single = scratch<float>("dn.single", (size_t)T * Ct);
  conditioningSingle(d, t, single);
  if (check) checkOracle("diffusion conditioning single", single, (size_t)T * Ct, "o/cond/single");
  float denom = sqrtf(t * t + d.sigma * d.sigma);
  float* r6 = scratch<float>("dn.r6", (size_t)A * 6);
  coordsInputK<<<blocks((size_t)A * 6), 256, 0, STREAM>>>(xNoisy, 1.f / denom, r6, A);
  float* q = scratch<float>("dn.q", (size_t)A * Ca);
  gemm(r6, F("diffusionAtomEncoder/coordsLinear"), q, A, 6, Ca);
  addK<<<blocks((size_t)A * Ca), 256, 0, STREAM>>>(q, d.atoms.c0, (size_t)A * Ca);
  swaStack(ac, q, d.atoms.c0, "diffusionAtomEncoder", d.atomBlocks, DIFFUSION_HALF_WINDOW);
  float* tok = scratch<float>("dn.tok", (size_t)A * Ct);
  gemm(q, F("diffusionAtomEncoder/toToken"), tok, A, Ca, Ct);
  reluK<<<blocks((size_t)A * Ct), 256, 0, STREAM>>>(tok, (size_t)A * Ct);
  float* a = scratch<float>("dn.a", (size_t)T * Ct);
  scatterMeanK<<<blocks((size_t)T * Ct), 256, 0, STREAM>>>(tok, Idev("atom_to_token"), ac.mask, a, A, T, Ct, Ct);
  float* sn = scratch<float>("dn.sn", (size_t)T * Ct); float* st = scratch<float>("dn.st", (size_t)T * Ct);
  layerNorm(single, sn, T, Ct, F("diffusion/stepNorm/scale"), F("diffusion/stepNorm/offset"));
  gemm(sn, F("diffusion/singleToToken"), st, T, Ct, Ct);
  addK<<<blocks((size_t)T * Ct), 256, 0, STREAM>>>(a, st, (size_t)T * Ct);
  for (int b = 0; b < d.tokenBlocks; ++b) tokenBlock(d, a, single, b);
  layerNorm(a, a, T, Ct, F("diffusion/tokenNorm/scale"), F("diffusion/tokenNorm/offset"));
  // the decoder: q (the encoder's output, the skip) + the tokens gathered back
  float* pt = scratch<float>("dn.pt", (size_t)T * Ca);
  gemm(a, F("diffusionAtomDecoder/tokenToAtom"), pt, T, Ct, Ca);
  gatherTokensK<<<blocks((size_t)A * Ca), 256, 0, STREAM>>>(pt, Idev("atom_to_token"), ac.mask, q, A, Ca);
  swaStack(ac, q, d.atoms.c0, "diffusionAtomDecoder", d.atomBlocks, DIFFUSION_HALF_WINDOW);
  layerNorm(q, q, A, Ca, F("diffusionAtomDecoder/norm/scale"), F("diffusionAtomDecoder/norm/offset"));
  float* r = scratch<float>("dn.r", (size_t)A * 3);
  gemm(q, F("diffusionAtomDecoder/outputLinear"), r, A, Ca, 3);
  float s2 = d.sigma * d.sigma, t2 = t * t;
  edmCombineK<<<blocks((size_t)A * 3), 256, 0, STREAM>>>(xNoisy, r, xDenoised, s2 / (s2 + t2), d.sigma * t / sqrtf(s2 + t2), A * 3);
}
