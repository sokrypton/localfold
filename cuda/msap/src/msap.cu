// MSA Pairformer (Akiyama et al., Cell 2026; yoakiyama/MSA_Pairformer) in CUDA: an alignment in, residue contacts
// (Cb and ConFind), the query's language-model logits and the embeddings out. 22 layers of
//
//   msa  += PWA(msa, pair)            the MSA's pair-weighted averaging (AF3's Algorithm 10)
//   msa  += transition(msa)           SwiGLU, 464 channels
//   pair += query-biased OPM(msa)     the outer product mean over sequences weighted by a differential attention
//                                     against the query row (this file: qOpm)
//   pair += tri out, tri in, transition (256 channels)
//
// then a final PWA and transition over the MSA. Everything but the query-biased outer product, the input
// embeddings and the heads is cuda/ef2's (ESMFold2's MSA encoder is the same module set): its kernels read the
// weights tools/export_msap.py writes in ESMFold2's names and layouts.
//
//   msap --weights=<dir> --oracle=<dir>           the oracle's own subsampled alignment (tools/oracle/dump_msap.py),
//                                                 every stage compared with it
#include "../../ef2/src/shim.cuh"   // (trunk.cuh's streamed z_init reads it; unused here)
#include "../../ef2/src/msa.cuh"
#include <chrono>

// m[s][l][:] = W[token][:]: the one-hot's projection is a row gather
__global__ void msaEmbedK(const int* tok, const float* W, float* m, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  m[t] = W[(size_t)tok[t / C] * C + (int)(t % C)];
}
// The pair's initial value: RelativePositionEncoding's one-hot over [d_res (66) | d_token (66) | same entity (1) |
// d_chain (6)] times its weight, plus the bond feature. Only d_token varies (d_res is r_max everywhere, d_chain 2 s_max
// + 1 and same entity 1 - the release's single-chain form), the token index shifted 100 past each chain break so a
// cross-chain pair clips to the edge bin; the bond (|i - j| = 1) is over the UNSHIFTED index, so it spans a chain break
__global__ void pairInitK(const int* shifted, const float* W, const float* bondW, float* z, int L, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / L), j = (int)(ij % L);
  int d = min(max(shifted[i] - shifted[j] + 32, 0), 64);
  float v = W[32 * C + c] + W[(66 + d) * C + c] + W[132 * C + c] + W[138 * C + c];
  if (abs(i - j) == 1) v += bondW[c];
  z[t] = v;
}

// ---------------------------------------------------------------- the query-biased outer product
// LayerNorm of each 128-wide half of a [rows][256] q or k (both halves take the same scale and offset)
__global__ void halvesNormK(float* x, size_t rows, const float* sc, const float* of) {
  size_t r = (size_t)blockIdx.x;                     // a block a (row, half), 128 threads
  if (r >= rows * 2) return;
  float* v = x + r * 128;
  float a = v[threadIdx.x];
  __shared__ float red[4];
  float s = a; for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  float mean = (red[0] + red[1] + red[2] + red[3]) / 128.f;
  __syncthreads();
  float d = a - mean, q = d * d; for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = q;
  __syncthreads();
  float var = (red[0] + red[1] + red[2] + red[3]) / 128.f;
  v[threadIdx.x] = d * rsqrtf(var + 1e-5f) * sc[threadIdx.x] + of[threadIdx.x];
}
// raw[t][s] = scale * mean over l of q_t[l] . k_t[s][l] (t the half): a block a (s, t)
__global__ void seqScoresK(const float* q, const float* k, float* raw, int S, int L, float scale) {
  int s = blockIdx.x, t = blockIdx.y;
  float acc = 0.f;
  for (int l = threadIdx.y; l < L; l += blockDim.y) {
    const float* qv = q + (size_t)l * 256 + t * 128;
    const float* kv = k + ((size_t)s * L + l) * 256 + t * 128;
    float d = 0.f;
    for (int c = threadIdx.x; c < 128; c += 32) d += qv[c] * kv[c];
    acc += d;
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(~0u, acc, o);
  __shared__ float red[32];
  if (threadIdx.x == 0) red[threadIdx.y] = acc;
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    float sum = 0.f; for (int y = 0; y < (int)blockDim.y; ++y) sum += red[y];
    raw[(size_t)t * S + s] = sum / L * scale;
  }
}
// w = softmax_s(raw0 - lambda raw1), and its sum (the outer product's denominator) - one block
__global__ void seqWeightsK(const float* raw, float lambda, float* w, float* sumOut, int S) {
  __shared__ float red[1024];
  float mx = -INFINITY;
  for (int s = threadIdx.x; s < S; s += blockDim.x) mx = fmaxf(mx, raw[s] - lambda * raw[S + s]);
  red[threadIdx.x] = mx; __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) { if ((int)threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
  mx = red[0]; __syncthreads();
  float sum = 0.f;
  for (int s = threadIdx.x; s < S; s += blockDim.x) sum += expf(raw[s] - lambda * raw[S + s] - mx);
  red[threadIdx.x] = sum; __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) { if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
  sum = red[0]; __syncthreads();
  for (int s = threadIdx.x; s < S; s += blockDim.x) w[s] = expf(raw[s] - lambda * raw[S + s] - mx) / sum;
  __syncthreads();
  if (threadIdx.x == 0) { float t = 0.f; for (int s = 0; s < S; ++s) t += w[s]; *sumOut = t; }
}
// a [S][L][16] scaled by sqrt(w_s + eps), laid out [L*16][S] for the contraction (column (l, c) contiguous over s)
__global__ void opmScaleK(const float* x, const float* w, float* out, int S, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * L * 16) return;
  int c = (int)(t % 16); size_t r = t / 16; int l = (int)(r % L), s = (int)(r / L);
  out[((size_t)l * 16 + c) * S + s] = x[t] * sqrtf(w[s] + 1e-32f);
}
// prod [(i, c)][(j, e)] -> [i][j][c * 16 + e] / denominator
__global__ void opmPermuteK(const float* prod, float* out, int L, const float* denom) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * 256) return;
  int ce = (int)(t % 256); size_t ij = t / 256; int i = (int)(ij / L), j = (int)(ij % L);
  int c = ce / 16, e = ce % 16;
  out[t] = prod[((size_t)i * 16 + c) * ((size_t)L * 16) + (size_t)j * 16 + e] / (*denom + 1e-32f);
}
// pair += SwiGLU(h) - h [rows][512] with the bias added (the gate is the first half: the export swapped them)
__global__ void opmGateAddK(const float* h, const float* bias, float* pair, size_t rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * 256) return;
  size_t r = t / 256; int c = (int)(t % 256);
  float g = h[r * 512 + c] + bias[c], x = h[r * 512 + 256 + c] + bias[256 + c];
  pair[t] += g / (1.f + expf(-g)) * x;
}
// one layer's query-biased outer product: pair += SwiGLU(Linear(outer(a sqrt w, b sqrt w) / sum w))
inline void qOpm(float* pair, const float* m, int S, int L, int l, float* wOut) {
  const std::string O = "msap/blocks/" + std::to_string(l) + "/opm/";
  const int Cm = 464; size_t R = (size_t)S * L, P = (size_t)L * L;
  float* nm = scratch<float>("opm.nm", R * Cm);
  layerNorm(m, nm, R, Cm, F(O + "normScale"), F(O + "normOffset"));
  float* q = scratch<float>("opm.q", (size_t)L * 256); float* k = scratch<float>("opm.k", R * 256);
  gemm(nm, F(O + "q"), q, L, Cm, 256);                     // (the query is row 0: its L rows lead nm)
  gemm(nm, F(O + "k"), k, R, Cm, 256);
  halvesNormK<<<(unsigned)(L * 2), 128, 0, STREAM>>>(q, L, F(O + "qNormScale"), F(O + "qNormOffset"));
  halvesNormK<<<(unsigned)(R * 2), 128, 0, STREAM>>>(k, R, F(O + "kNormScale"), F(O + "kNormOffset"));
  float* raw = scratch<float>("opm.raw", (size_t)2 * S);
  seqScoresK<<<dim3(S, 2), dim3(32, 16), 0, STREAM>>>(q, k, raw, S, L, 1.f / sqrtf(128.f));
  // lambda = exp(sum lq1 lk1) - exp(sum lq2 lk2) + lambda_init (the host's: four 128-vectors)
  auto dot = [&](const char* a, const char* b) {
    const float *x = M.f("f/" + O + a), *y = M.f("f/" + O + b); double d = 0; for (int c = 0; c < 128; ++c) d += (double)x[c] * y[c];
    return (float)d;
  };
  const float lambda = expf(dot("lambdaQ1", "lambdaK1")) - expf(dot("lambdaQ2", "lambdaK2")) + (float)M.meta("meta/opmLambdaInit/" + std::to_string(l));
  float* w = wOut; float* wsum = scratch<float>("opm.wsum", 1);
  seqWeightsK<<<1, 1024, 0, STREAM>>>(raw, lambda, w, wsum, S);
  float* a = scratch<float>("opm.a", R * 16); float* b = scratch<float>("opm.b", R * 16);
  gemm(nm, F(O + "left"), a, R, Cm, 16); gemm(nm, F(O + "right"), b, R, Cm, 16);
  float* as = scratch<float>("opm.as", R * 16); float* bs = scratch<float>("opm.bs", R * 16);
  opmScaleK<<<blocks(R * 16), 256, 0, STREAM>>>(a, w, as, S, L);
  opmScaleK<<<blocks(R * 16), 256, 0, STREAM>>>(b, w, bs, S, L);
  // prod (col-major [(j, e)][(i, c)], i.e. row-major [(i, c)][(j, e)]) = sum_s a[s][(i, c)] b[s][(j, e)]
  const size_t LC = (size_t)L * 16;
  float* prod = scratch<float>("opm.prod", LC * LC);
  const float one = 1.f, zero = 0.f;
  CB(cublasSgemm(H, CUBLAS_OP_T, CUBLAS_OP_N, (int)LC, (int)LC, S, &one, bs, S, as, S, &zero, prod, (int)LC));
  float* outer = scratch<float>("opm.outer", P * 256);
  opmPermuteK<<<blocks(P * 256), 256, 0, STREAM>>>(prod, outer, L, wsum);
  float* h = scratch<float>("opm.h", P * 512);
  gemm(outer, F(O + "toPair"), h, P, 256, 512);
  opmGateAddK<<<blocks(P * 256), 256, 0, STREAM>>>(h, F(O + "toPairBias"), pair, P);
}

// ---------------------------------------------------------------- the heads
// contacts[i][j] = (sigmoid(LN(z_ij) . w + b) + sigmoid(LN(z_ji) . w + b)) / 2: one warp a pair
__global__ void contactK(const float* z, const float* sc, const float* of, const float* w, const float* b, float* logit, size_t P) {
  size_t warp = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32; int lane = threadIdx.x & 31;
  if (warp >= P) return;
  const float* v = z + warp * 256;
  float x[8], s = 0.f;
  for (int k = 0; k < 8; ++k) { x[k] = v[lane + 32 * k]; s += x[k]; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / 256.f, q = 0.f;
  for (int k = 0; k < 8; ++k) { float d = x[k] - mean; q += d * d; }
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
  float inv = rsqrtf(q / 256.f + 1e-5f), dot = 0.f;
  for (int k = 0; k < 8; ++k) { int c = lane + 32 * k; dot += ((x[k] - mean) * inv * sc[c] + of[c]) * w[c]; }
  for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(~0u, dot, o);
  if (lane == 0) logit[warp] = 1.f / (1.f + expf(-(dot + b[0])));
}
__global__ void symmetriseK(const float* p, float* out, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L) return;
  int i = (int)(t / L), j = (int)(t % L);
  out[t] = 0.5f * (p[t] + p[(size_t)j * L + i]);
}
inline void contacts(const float* z, int L, const std::string& head, float* out) {
  size_t P = (size_t)L * L;
  float* p = scratch<float>("head.p", P);
  contactK<<<blocks(P * 32), 256, 0, STREAM>>>(z, F(head + "normScale"), F(head + "normOffset"), F(head + "weight"), F(head + "bias"), p, P);
  symmetriseK<<<blocks(P), 256, 0, STREAM>>>(p, out, L);
}
__global__ void lmGateK(const float* h, const float* bias, float* out, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * 464) return;
  size_t r = t / 464; int c = (int)(t % 464);
  float g = h[r * 928 + c] + bias[c], x = h[r * 928 + 464 + c] + bias[464 + c];
  out[t] = g / (1.f + expf(-g)) * x;
}
__global__ void addRowBiasK(float* x, const float* b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] += b[t % C];
}
// the LM head over the query row: logits [L][26]
inline void lmHead(const float* mq, int L, float* logits) {
  float* x = scratch<float>("lm.x", (size_t)L * 464); float* h = scratch<float>("lm.h", (size_t)L * 928);
  layerNorm(mq, x, L, 464, F("msap/lm/initNormScale"), F("msap/lm/initNormOffset"));
  gemm(x, F("msap/lm/dense"), h, L, 464, 928);
  lmGateK<<<blocks((size_t)L * 464), 256, 0, STREAM>>>(h, F("msap/lm/denseBias"), x, L);
  layerNorm(x, x, L, 464, F("msap/lm/preLogitNormScale"), F("msap/lm/preLogitNormOffset"));
  gemm(x, F("msap/lm/logits"), logits, L, 464, 26);
  addRowBiasK<<<blocks((size_t)L * 26), 256, 0, STREAM>>>(logits, F("msap/lm/logitsBias"), L, 26);
}

// every stage against the oracle's (--check), or none - a comparison downloads the whole tensor
inline bool checking = false;
#define CHECK(...) do { if (checking) check(__VA_ARGS__); } while (0)

int main(int argc, char** argv) {
  std::string weights, oracle; int repeats = 1;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--oracle=", 9)) oracle = argv[i] + 9;
    else if (!strcmp(argv[i], "--fast")) FAST = true;
    else if (!strcmp(argv[i], "--check")) checking = true;
    else if (!strncmp(argv[i], "--repeat=", 9)) repeats = atoi(argv[i] + 9);
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
  }
  if (weights.empty() || oracle.empty()) { fprintf(stderr, "msap --weights=<dir> --oracle=<dir> [--fast]\n"); return 1; }
  M.load(weights); M.load(oracle);
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  CB(cublasSetMathMode(H, FAST ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
  M.upload(0);
  const int S = (int)M.meta("o/meta/depth"), L = (int)M.meta("o/meta/length"), C = 256, Cm = 464;
  const int depth = (int)M.meta("meta/depth"), cbLayer = (int)M.meta("meta/contactLayer"), cfLayer = (int)M.meta("meta/confindContactLayer");
  size_t R = (size_t)S * L, P = (size_t)L * L;
  printf("msap: %d rows x %d columns, %d layers%s\n", S, L, depth, FAST ? " (fast)" : "");
  // the token index, shifted 100 past each chain break
  std::vector<int> shifted(L);
  const int* br = M.i("o/chainBreaks");
  for (int l = 0; l < L; ++l) { shifted[l] = l; for (size_t k = 0; k < M.len("o/chainBreaks"); ++k) if (br[k] >= 0 && l >= br[k]) shifted[l] += 100; }
  int* shiftedD = upload(shifted.data(), L);
  int* tokD = scratch<int>("in.tok", R);
  CK(cudaMemcpyAsync(tokD, M.i("o/tokens"), R * 4, cudaMemcpyHostToDevice, STREAM));
  float* m = dalloc(R * Cm); float* z = dalloc(P * C);
  float* pmask = dalloc(P); fillK<<<blocks(P), 256, 0, STREAM>>>(pmask, 1.f, P);
  float* w = dalloc(S); float* cb = dalloc(P); float* cf = dalloc(P); float* logits = dalloc((size_t)L * 26);
  for (int rep = 0; rep < repeats; ++rep) {
  CK(cudaStreamSynchronize(STREAM));
  auto t0 = std::chrono::steady_clock::now();
  msaEmbedK<<<blocks(R * Cm), 256, 0, STREAM>>>(tokD, F("msap/msaInit"), m, R, Cm);
  pairInitK<<<blocks(P * C), 256, 0, STREAM>>>(shiftedD, F("msap/relpos"), F("msap/bond"), z, L, C);
  CHECK("msa init", m, R * Cm, "o/msaInit");
  CHECK("pair init", z, P * C, "o/pairInit");
  for (int l = 0; l < depth; ++l) {
    const std::string B = "msap/blocks/" + std::to_string(l) + "/", Lo = "o/layer" + std::to_string(l) + "/";
    msaPairWeightedAveraging(m, z, S, L, Cm, C, B + "pwa/");
    if (l == 0) CHECK("l0 after pwa", m, R * Cm, "o/layer0/afterPwa");
    rowsTransition(m, R, Cm, B + "msaTransition/");
    if (l == 0) CHECK("l0 after msa transition", m, R * Cm, "o/layer0/afterMsaTransition");
    qOpm(z, m, S, L, l, w);
    CHECK(("l" + std::to_string(l) + " seq weights").c_str(), w, S, Lo + "seqWeights");
    if (l == 0 && checking) {
      CHECK("l0 after opm", z, P * C, "o/layer0/afterOpm");
      // the pair block's three modules one at a time, as trunkBlock runs them
      const std::string T = "msap/blocks/0/";
      if (FAST) triangleFast(z, pmask, L, C, T + "triangleMultiplicationOutgoing/", true);
      else triangle(z, pmask, L, C, T + "triangleMultiplicationOutgoing/", true);
      CHECK("l0 after tri out", z, P * C, "o/layer0/afterTriOut");
      if (FAST) triangleFast(z, pmask, L, C, T + "triangleMultiplicationIncoming/", false);
      else triangle(z, pmask, L, C, T + "triangleMultiplicationIncoming/", false);
      CHECK("l0 after tri in", z, P * C, "o/layer0/afterTriIn");
      if (FAST) transitionFast(z, P, C, T + "pairTransition/"); else pairTransition(z, L, C, T + "pairTransition/");
      CHECK("l0 after pair transition", z, P * C, "o/layer0/afterPairTransition");
    } else trunkBlock(z, pmask, L, C, "msap/blocks", l);
    if (M.has(Lo + "pair")) { CHECK(("l" + std::to_string(l) + " pair").c_str(), z, P * C, Lo + "pair"); CHECK(("l" + std::to_string(l) + " msa").c_str(), m, R * Cm, Lo + "msa"); }
    if (l == cbLayer) { contacts(z, L, "msap/cbContact/", cb); CHECK("cb contacts", cb, P, "o/cbContacts"); }
    if (l == cfLayer) { contacts(z, L, "msap/confindContact/", cf); CHECK("confind contacts", cf, P, "o/confindContacts"); }
  }
  msaPairWeightedAveraging(m, z, S, L, Cm, C, "msap/finalPwa/");
  rowsTransition(m, R, Cm, "msap/finalMsaTransition/");
  CHECK("final msa query", m, (size_t)L * Cm, "o/final/msaQuery");
  lmHead(m, L, logits);
  CHECK("logits", logits, (size_t)L * 26, "o/logits");
  CK(cudaStreamSynchronize(STREAM));
  printf("msap: %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  return 0;
}
