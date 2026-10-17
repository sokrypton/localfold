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
//   localfold-msap --a3m=<alignment> --out=<dir> [--fast] ...   contacts, logits, embeddings (USAGE below)
//   localfold-msap --oracle=<dir> --check                       tools/oracle/dump_msap.py's own subsample, every stage
//                                                               compared with the reference's
//
// --fast (f16 activations into tensor-core GEMMs, f32 accumulation and residuals) is 7.8e-4 from the float32
// reference on 5CAJ's contacts, where the reference's own bfloat16 is 5.3e-3 from it (1B70); without it every
// stage is within 2.6e-6.
#include "../../ef2/src/shim.cuh"   // (trunk.cuh's streamed z_init reads it; unused here)
#include "../../ef2/src/msa.cuh"
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>
#include "select.cuh"

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
// --fast: the same in f16 - the scaled operands, the contraction's product (f32 accumulation), the permuted outer
// product and the projection's 512 columns (its bias in cuBLASLt's epilogue)
__global__ void opmScaleHK(const float* x, const float* w, half* out, int S, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * L * 16) return;
  int c = (int)(t % 16); size_t r = t / 16; int l = (int)(r % L), s = (int)(r / L);
  out[((size_t)l * 16 + c) * S + s] = __float2half(x[t] * sqrtf(w[s] + 1e-32f));
}
// a thread a (i, j, c): the 16 e of one product row, 32 contiguous bytes in and out
__global__ void opmPermuteHK(const half* prod, half* out, int L, const float* denom) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * 16) return;
  int c = (int)(t % 16); size_t ij = t / 16; int i = (int)(ij / L), j = (int)(ij % L);
  const float inv = 1.f / (*denom + 1e-32f);
  const uint4* src = reinterpret_cast<const uint4*>(prod + ((size_t)i * 16 + c) * ((size_t)L * 16) + (size_t)j * 16);
  uint4* dst = reinterpret_cast<uint4*>(out + ij * 256 + c * 16);
#pragma unroll
  for (int v = 0; v < 2; ++v) {
    uint4 q = src[v]; half2* h = reinterpret_cast<half2*>(&q);
#pragma unroll
    for (int e = 0; e < 4; ++e) { float2 f = __half22float2(h[e]); h[e] = __floats2half2_rn(f.x * inv, f.y * inv); }
    dst[v] = q;
  }
}
__global__ void opmGateAddHK(const half* h, float* pair, size_t rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * 256) return;
  size_t r = t / 256; int c = (int)(t % 256);
  float g = __half2float(h[r * 512 + c]), x = __half2float(h[r * 512 + 256 + c]);
  pair[t] += g / (1.f + expf(-g)) * x;
}
// one layer's query-biased outer product: pair += SwiGLU(Linear(outer(a sqrt w, b sqrt w) / sum w))
inline void qOpm(float* pair, const float* m, int S, int L, int l, float* wOut) {
  const std::string O = "msap/blocks/" + std::to_string(l) + "/opm/";
  const int Cm = 464; size_t R = (size_t)S * L, P = (size_t)L * L;
  float* q = scratch<float>("opm.q", (size_t)L * 256); float* k = scratch<float>("opm.k", R * 256);
  float* a = scratch<float>("opm.a", R * 16); float* b = scratch<float>("opm.b", R * 16);
  // (the query is row 0: its L rows lead the normed rows)
  if (FAST) {
    half* nm = scratch<half>("opm.nmh", R * Cm);
    layerNormH(m, nm, R, Cm, F(O + "normScale"), F(O + "normOffset"));
    ltGemm(nm, Fh(O + "q"), q, false, L, Cm, 256, nullptr, false, 0.f);
    ltGemm(nm, Fh(O + "k"), k, false, R, Cm, 256, nullptr, false, 0.f);
    ltGemm(nm, Fh(O + "left"), a, false, R, Cm, 16, nullptr, false, 0.f);
    ltGemm(nm, Fh(O + "right"), b, false, R, Cm, 16, nullptr, false, 0.f);
  } else {
    float* nm = scratch<float>("opm.nm", R * Cm);
    layerNorm(m, nm, R, Cm, F(O + "normScale"), F(O + "normOffset"));
    gemm(nm, F(O + "q"), q, L, Cm, 256);
    gemm(nm, F(O + "k"), k, R, Cm, 256);
    gemm(nm, F(O + "left"), a, R, Cm, 16); gemm(nm, F(O + "right"), b, R, Cm, 16);
  }
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
  const size_t LC = (size_t)L * 16;
  const float one = 1.f, zero = 0.f;
  if (FAST) {
    half* as = scratch<half>("opm.ash", R * 16); half* bs = scratch<half>("opm.bsh", R * 16);
    opmScaleHK<<<blocks(R * 16), 256, 0, STREAM>>>(a, w, as, S, L);
    opmScaleHK<<<blocks(R * 16), 256, 0, STREAM>>>(b, w, bs, S, L);
    half* prod = scratch<half>("opm.prodh", LC * LC);
    CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, (int)LC, (int)LC, S, &one, bs, CUDA_R_16F, S, as, CUDA_R_16F, S, &zero, prod,
                    CUDA_R_16F, (int)LC, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    half* outer = scratch<half>("opm.outerh", P * 256);
    opmPermuteHK<<<blocks(P * 16), 256, 0, STREAM>>>(prod, outer, L, wsum);
    const size_t chunk = std::min<size_t>(P, (size_t)64 << 20 >> 9);       // (the projection's rows: 64 M halves)
    half* h = scratch<half>("opm.hh", chunk * 512);
    for (size_t r0 = 0; r0 < P; r0 += chunk) {
      size_t r = std::min(chunk, P - r0);
      ltGemm(outer + r0 * 256, Fh(O + "toPair"), h, true, r, 256, 512, F(O + "toPairBias"), false, 0.f);
      opmGateAddHK<<<blocks(r * 256), 256, 0, STREAM>>>(h, pair + r0 * 256, r);
    }
    return;
  }
  float* as = scratch<float>("opm.as", R * 16); float* bs = scratch<float>("opm.bs", R * 16);
  opmScaleK<<<blocks(R * 16), 256, 0, STREAM>>>(a, w, as, S, L);
  opmScaleK<<<blocks(R * 16), 256, 0, STREAM>>>(b, w, bs, S, L);
  // prod (col-major [(j, e)][(i, c)], i.e. row-major [(i, c)][(j, e)]) = sum_s a[s][(i, c)] b[s][(j, e)]
  float* prod = scratch<float>("opm.prod", LC * LC);
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

// ---------------------------------------------------------------- the subsample
// the reference's greedy_select on the device: from the query, repeatedly the row with the largest mean Hamming
// distance (every column, a gap a token like any other) to the rows chosen so far - summed here as integers, so a
// tie is a tie (the first row wins, numpy's argmax) - over rows [n][L] of tokens
__global__ void hammingAddK(const unsigned char* tok, int n, int L, const int* last, int* sum, const char* chosen) {
  int k = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (k >= n || chosen[k]) return;
  const unsigned char *a = tok + (size_t)k * L, *b = tok + (size_t)*last * L;
  int d = 0;
  for (int i = lane; i < L; i += 32) d += a[i] != b[i];
  for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(~0u, d, o);
  if (lane == 0) sum[k] += d;
}
__global__ void argmaxPickK(const int* sum, int n, char* chosen, int* last, int* picks, int step) {
  __shared__ int bv[1024], bi[1024];
  int v = -1, at = n;
  for (int k = threadIdx.x; k < n; k += blockDim.x) if (!chosen[k] && sum[k] > v) { v = sum[k]; at = k; }
  bv[threadIdx.x] = v; bi[threadIdx.x] = at; __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) {
    if ((int)threadIdx.x < o) {
      int ov = bv[threadIdx.x + o], oi = bi[threadIdx.x + o];
      if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) { bv[threadIdx.x] = ov; bi[threadIdx.x] = oi; }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { chosen[bi[0]] = 1; *last = bi[0]; picks[step] = bi[0]; }
}
inline std::vector<int> greedyRows(const std::vector<std::string>& rows, const std::vector<int>& from, int depth) {
  const int n = (int)from.size(), L = (int)rows[0].size();
  if (n <= depth) return from;
  std::vector<unsigned char> h((size_t)n * L);
  for (int k = 0; k < n; ++k) for (int i = 0; i < L; ++i) h[(size_t)k * L + i] = (unsigned char)msapToken(rows[from[k]][i]);
  unsigned char* tok = upload(h.data(), h.size());
  int* sum = dallocT<int>(n); char* chosen = dallocT<char>(n); int* last = dallocT<int>(1); int* picks = dallocT<int>(depth);
  CK(cudaMemsetAsync(sum, 0, n * 4, STREAM)); CK(cudaMemsetAsync(chosen, 0, n, STREAM)); CK(cudaMemsetAsync(last, 0, 4, STREAM));
  CK(cudaMemsetAsync(chosen, 1, 1, STREAM)); CK(cudaMemsetAsync(picks, 0, 4, STREAM));
  for (int step = 1; step < depth; ++step) {
    hammingAddK<<<(n + 7) / 8, 256, 0, STREAM>>>(tok, n, L, last, sum, chosen);
    argmaxPickK<<<1, 1024, 0, STREAM>>>(sum, n, chosen, last, picks, step);
  }
  std::vector<int> p(depth);
  CK(cudaMemcpyAsync(p.data(), picks, depth * 4, cudaMemcpyDeviceToHost, STREAM)); CK(cudaStreamSynchronize(STREAM));
  for (void* x : { (void*)tok, (void*)sum, (void*)chosen, (void*)last, (void*)picks }) CK(cudaFree(x));
  std::sort(p.begin(), p.end());
  std::vector<int> out; for (int k : p) out.push_back(from[k]);
  return out;
}

// every stage against the oracle's (--check), or none - a comparison downloads the whole tensor
inline bool checking = false;
#define CHECK(...) do { if (checking) check(__VA_ARGS__); } while (0)

static const char* USAGE =
  "localfold-msap --a3m=<alignment> --out=<dir> [--fast] [--depth=512] [--chain-breaks=i,j] [--weights=<dir>]\n"
  "               [--embeddings | --contacts-only]\n"
  "  writes <dir>/cb_contacts.npy and confind_contacts.npy [L][L], logits.npy [L][26] (the query's 26 tokens,\n"
  "  ARNDCEQGHILKMFPSTWYVXBZUO-), subsample.a3m (the rows the model read), and with --embeddings query.npy [L][464]\n"
  "  (the final MSA's query row) and pair.npy [L][L][256] (the pair after the last layer)\n"
  "  --chain-breaks: the residue each chain after the first starts at (a ColabFold '#' header gives them)\n"
  "  --contacts-only: stop at the last contact head's layer (19 of 22) - no logits, no embeddings\n"
  "localfold-msap --oracle=<dump dir> [--check] [--repeat=N]    tools/oracle/dump_msap.py's subsample, its stages compared\n";

int main(int argc, char** argv) {
  std::string weights, oracle, a3m, out, breaksArg; int repeats = 1, maxDepth = 512; bool embeddings = false, contactsOnly = false;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--oracle=", 9)) oracle = argv[i] + 9;
    else if (!strncmp(argv[i], "--a3m=", 6)) a3m = argv[i] + 6;
    else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
    else if (!strncmp(argv[i], "--depth=", 8)) maxDepth = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--chain-breaks=", 15)) breaksArg = argv[i] + 15;
    else if (!strcmp(argv[i], "--embeddings")) embeddings = true;
    else if (!strcmp(argv[i], "--contacts-only")) contactsOnly = true;
    else if (!strcmp(argv[i], "--fast")) FAST = true;
    else if (!strcmp(argv[i], "--check")) checking = true;
    else if (!strncmp(argv[i], "--repeat=", 9)) repeats = atoi(argv[i] + 9);
    else { fprintf(stderr, "unknown flag %s\n%s", argv[i], USAGE); return 1; }
  }
  if (contactsOnly && (embeddings || checking)) { fprintf(stderr, "--contacts-only writes neither embeddings nor every stage\n"); return 1; }
  if (a3m.empty() == oracle.empty() || (!a3m.empty() && out.empty()) || maxDepth < 1) { fputs(USAGE, stderr); return 1; }
  if (weights.empty()) {              // beside the binary, as cuda/<port>/weights
    char self[4096]; ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    std::string dir = n > 0 ? std::string(self, n) : "."; dir = dir.substr(0, dir.rfind('/'));
    weights = dir + "/weights";
  }
  M.load(weights);
  if (!oracle.empty()) M.load(oracle);
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  CB(cublasSetMathMode(H, FAST ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
  M.upload(0);
  // the alignment: the oracle's subsample as it is, or an A3M filtered and subsampled here
  std::vector<int> tokens, breaks; std::vector<std::string> subsample; int S, L;
  auto t0 = std::chrono::steady_clock::now();
  if (!oracle.empty()) {
    S = (int)M.meta("o/meta/depth"); L = (int)M.meta("o/meta/length");
    tokens.assign(M.i("o/tokens"), M.i("o/tokens") + (size_t)S * L);
    for (size_t k = 0; k < M.len("o/chainBreaks"); ++k) if (M.i("o/chainBreaks")[k] >= 0) breaks.push_back(M.i("o/chainBreaks")[k]);
  } else {
    Alignment al = readA3m(a3m);
    breaks = al.chainBreaks;
    if (!breaksArg.empty()) { breaks.clear(); for (const char* p = breaksArg.c_str(); *p;) { breaks.push_back((int)strtol(p, (char**)&p, 10)); if (*p == ',') ++p; else break; } }
    L = (int)al.rows[0].size();
    std::vector<int> kept = filterRows(al.rows, 0.7f, 0.3f);
    std::vector<int> rows = greedyRows(al.rows, kept, maxDepth);
    S = (int)rows.size();
    for (int k : rows) { subsample.push_back(al.rows[k]); for (char c : al.rows[k]) tokens.push_back(msapToken(c)); }
    printf("msap: %zu rows, %zu past coverage 70%% and query identity 30%%, %d chosen\n", al.rows.size(), kept.size(), S);
  }
  for (int b : breaks) if (b <= 0 || b >= L) { fprintf(stderr, "a chain break at %d is outside 1..%d\n", b, L - 1); return 1; }
  const int C = 256, Cm = 464;
  const int depth = (int)M.meta("meta/depth"), cbLayer = (int)M.meta("meta/contactLayer"), cfLayer = (int)M.meta("meta/confindContactLayer");
  size_t R = (size_t)S * L, P = (size_t)L * L;
  printf("msap: %d rows x %d columns, %zu chains, %d layers%s (input %.0f ms)\n", S, L, breaks.size() + 1, depth, FAST ? " (fast)" : "",
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  // the token index, shifted 100 past each chain break
  std::vector<int> shifted(L);
  for (int l = 0; l < L; ++l) { shifted[l] = l; for (int b : breaks) if (l >= b) shifted[l] += 100; }
  int* shiftedD = upload(shifted.data(), L);
  int* tokD = upload(tokens.data(), R);
  float* m = dalloc(R * Cm); float* z = dalloc(P * C);
  float* pmask = dalloc(P); fillK<<<blocks(P), 256, 0, STREAM>>>(pmask, 1.f, P);
  float* w = dalloc(S); float* cb = dalloc(P); float* cf = dalloc(P); float* logits = dalloc((size_t)L * 26);
  for (int rep = 0; rep < repeats; ++rep) {
  CK(cudaStreamSynchronize(STREAM));
  t0 = std::chrono::steady_clock::now();
  msaEmbedK<<<blocks(R * Cm), 256, 0, STREAM>>>(tokD, F("msap/msaInit"), m, R, Cm);
  pairInitK<<<blocks(P * C), 256, 0, STREAM>>>(shiftedD, F("msap/relpos"), F("msap/bond"), z, L, C);
  CHECK("msa init", m, R * Cm, "o/msaInit");
  CHECK("pair init", z, P * C, "o/pairInit");
  const int layers = contactsOnly ? std::max(cbLayer, cfLayer) + 1 : depth;
  for (int l = 0; l < layers; ++l) {
    const std::string B = "msap/blocks/" + std::to_string(l) + "/", Lo = "o/layer" + std::to_string(l) + "/";
    if (FAST) msaPairWeightedAveragingFast(m, z, S, L, Cm, C, B + "pwa/"); else msaPairWeightedAveraging(m, z, S, L, Cm, C, B + "pwa/");
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
  if (!contactsOnly) {
    if (FAST) msaPairWeightedAveragingFast(m, z, S, L, Cm, C, "msap/finalPwa/"); else msaPairWeightedAveraging(m, z, S, L, Cm, C, "msap/finalPwa/");
    rowsTransition(m, R, Cm, "msap/finalMsaTransition/");
    CHECK("final msa query", m, (size_t)L * Cm, "o/final/msaQuery");
    lmHead(m, L, logits);
    CHECK("logits", logits, (size_t)L * 26, "o/logits");
  }
  CK(cudaStreamSynchronize(STREAM));
  printf("msap: %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  if (!out.empty()) {
    mkdir(out.c_str(), 0755);
    writeNpy(out + "/cb_contacts.npy", download(cb, P), { (size_t)L, (size_t)L });
    writeNpy(out + "/confind_contacts.npy", download(cf, P), { (size_t)L, (size_t)L });
    if (!contactsOnly) writeNpy(out + "/logits.npy", download(logits, (size_t)L * 26), { (size_t)L, 26 });
    if (embeddings) {
      writeNpy(out + "/query.npy", download(m, (size_t)L * Cm), { (size_t)L, (size_t)Cm });
      writeNpy(out + "/pair.npy", download(z, P * C), { (size_t)L, (size_t)L, (size_t)C });
    }
    if (!subsample.empty()) {
      FILE* f = fopen((out + "/subsample.a3m").c_str(), "w");
      if (!f) { fprintf(stderr, "cannot write %s/subsample.a3m\n", out.c_str()); return 1; }
      for (size_t k = 0; k < subsample.size(); ++k) fprintf(f, ">%zu\n%s\n", k, subsample[k].c_str());
      fclose(f);
    }
    printf("msap: wrote %s\n", out.c_str());
  }
  return 0;
}
