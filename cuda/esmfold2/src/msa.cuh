// The full ESMFold2's MSA encoder (ESMFold2-Fast has none), float32. biohub's modeling_esmfold2.py MSAEncoder,
// MSAEncoderBlock, OuterProductMean and MSAPairWeightedAveraging (modeling_esmfold2_common.py) are the reading:
//
//   m = embed([one-hot(33) * mask | has_deletion | deletion_value]) + project_inputs(s_inputs)[token]
//   per block:  pair += OPM(m)                     Wout(outer(a, b)) / max(pair count, 1), the bias divided too
//               (not the last)  m += PWA(m, pair)  softmax over j of a head's LN(pair) logits, weighting
//                                                  each row's values, gated; m += SwiGLU transition(m)
//               pair += tri out, tri in, pair transition      (the trunk's pair-only block)
//
// Its input pair is z_init and its answer REPLACES the injection every pass (msa_encoder_overwrite) - and
// since neither the alignment nor z_init changes between passes (the subsample and the column mask are drawn
// once a fold here, where biohub draws the column mask once a fold and the subsample every pass), it runs ONCE
// a fold, in place over z_init. Layout [row][token][channel] (cuda/af3's), which no einsum here minds.
#pragma once
#include "trunk.cuh"
#include <random>

inline bool hasMsaEncoder() { return M.has("meta/msaBlocks") && M.meta("meta/msaBlocks") > 0; }

// one-hot rows of the embedding where the column is live, the two deletion features always (biohub masks only
// the one-hot: "bias-free MSAEncoder.embed requires zeroed padding"), plus the projected s_inputs
__global__ void msaEmbedEf2K(const int* rows, const float* del, const float* mask, const float* E, const float* fromInputs,
                             float* m, size_t count, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count * C) return;
  int c = (int)(t % C); size_t row = t / C; int token = (int)(row % T);
  float d = del[row];
  float v = mask[row] > 0.f ? E[(size_t)rows[row] * C + c] : 0.f;
  if (d > 0.f) v += E[(size_t)33 * C + c];
  v += 1.57079632679489662f * atanf(d / 3.f) * E[(size_t)34 * C + c];
  m[t] = v + fromInputs[(size_t)token * C + c];
}
// x [rows][a | b] (2H) times the row's mask, split into a [rows][H] and b [rows][H]
__global__ void opmSplitK(const float* x, const float* mask, float* a, float* b, size_t rows, int Hh) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * Hh) return;
  size_t r = t / Hh; int c = (int)(t % Hh); float k = mask[r];
  a[t] = x[r * 2 * Hh + c] * k; b[t] = x[r * 2 * Hh + Hh + c] * k;
}
// P [(i, c)][(j, e)] -> Pp [(i, j)][(c, e)]
__global__ void opmPermuteEf2K(const float* P, float* Pp, int bi, int T, int Hh) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)bi * T * Hh * Hh) return;
  int e = (int)(t % Hh); size_t r = t / Hh; int c = (int)(r % Hh); r /= Hh; int j = (int)(r % T); int i = (int)(r / T);
  Pp[t] = P[(((size_t)i * Hh + c) * T + j) * Hh + e];
}
// pair[(i0 + i) j] += (X + b) / max(count, 1)
__global__ void opmAddClampedK(float* pair, const float* X, const float* b, const float* count, int i0, int bi, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)bi * T * C) return;
  int c = (int)(t % C); size_t ij = t / C;
  size_t at = (size_t)i0 * T + ij;
  pair[at * C + c] += (X[t] + b[c]) / fmaxf(count[at], 1.f);
}
inline void msaOuterProductMean(float* pair, const float* m, const float* mask, int S, int T, int Cm, int C,
                                const std::string& Bn) {
  const int Hh = (int)M.meta("meta/opmHidden");
  size_t rows = (size_t)S * T;
  float* ln = scratch<float>("msa.ln", rows * Cm);
  layerNorm(m, ln, rows, Cm, F(Bn + "normScale"), F(Bn + "normOffset"));
  float* x = scratch<float>("msa.opmx", rows * 2 * Hh);
  gemm(ln, F(Bn + "projection"), x, rows, Cm, 2 * Hh);
  float* a = scratch<float>("msa.opma", rows * Hh); float* b = scratch<float>("msa.opmb", rows * Hh);
  opmSplitK<<<blocks(rows * Hh), 256, 0, STREAM>>>(x, mask, a, b, rows, Hh);
  const float one = 1.f, zero = 0.f;
  float* count = scratch<float>("msa.count", (size_t)T * T);      // count[i][j] = sum_s mask[s][i] mask[s][j]
  CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, T, T, S, &one, mask, T, mask, T, &zero, count, T));
  size_t per = (size_t)T * (2 * Hh * Hh + C) * 4;                  // a query row's outer products and its update
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(T, ((size_t)256 << 20) / per));
  float* P = scratch<float>("msa.opmP", (size_t)Bi * T * Hh * Hh);
  float* Pp = scratch<float>("msa.opmPp", (size_t)Bi * T * Hh * Hh);
  float* X = scratch<float>("msa.opmX", (size_t)Bi * T * C);
  for (int i0 = 0; i0 < T; i0 += Bi) {
    int bi = std::min(Bi, T - i0);
    // row-major P [(i, c)][(j, e)] = sum_s a[s, i, c] b[s, j, e]; col-major P^T = b (T*H x S) a_blk^T
    CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, T * Hh, bi * Hh, S, &one, b, T * Hh, a + (size_t)i0 * Hh, T * Hh, &zero,
                   P, T * Hh));
    opmPermuteEf2K<<<blocks((size_t)bi * T * Hh * Hh), 256, 0, STREAM>>>(P, Pp, bi, T, Hh);
    gemm(Pp, F(Bn + "output"), X, (size_t)bi * T, Hh * Hh, C);
    opmAddClampedK<<<blocks((size_t)bi * T * C), 256, 0, STREAM>>>(pair, X, F(Bn + "outputBias"), count, i0, bi, T, C);
  }
}
// w[h][i][j] = softmax over j of flat[(i j)][h] (every token is real: no mask)
__global__ void pwaSoftmaxK(const float* flat, float* w, int T, int heads) {
  size_t rowId = blockIdx.x; int h = (int)(rowId / T), i = (int)(rowId % T);
  float* out = w + rowId * T;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < T; j += blockDim.x) { float v = flat[((size_t)i * T + j) * heads + h]; out[j] = v; mx = fmaxf(mx, v); }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); mx = red[0]; __syncthreads();
  float s = 0;
  for (int j = threadIdx.x; j < T; j += blockDim.x) { float e = expf(out[j] - mx); out[j] = e; s += e; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < T; j += blockDim.x) out[j] *= inv;
}
// v [s][j][h d + e] -> [h][j][s][e]
__global__ void pwaToHeadsK(const float* v, float* out, int S, int T, int heads, int d) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * T * heads * d) return;
  int e = (int)(t % d); size_t r = t / d; int s = (int)(r % S); r /= S; int j = (int)(r % T); int h = (int)(r / T);
  out[t] = v[((size_t)s * T + j) * heads * d + h * d + e];
}
// o [h][i][s][e] -> [s][i][h d + e], times sigmoid(gate)
__global__ void pwaFromHeadsK(const float* o, const float* gate, float* out, int S, int T, int heads, int d) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * T * heads * d) return;
  int c = (int)(t % (heads * d)); size_t si = t / (heads * d); int i = (int)(si % T), s = (int)(si / T);
  int h = c / d, e = c % d;
  out[t] = o[(((size_t)h * T + i) * S + s) * d + e] / (1.f + expf(-gate[t]));
}
inline void msaPairWeightedAveraging(float* m, const float* pair, int S, int T, int Cm, int C, const std::string& Bn) {
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
    gemm(pln, F(Bn + "pairLogits"), flat + r0 * heads, r, C, heads);
  }
  float* w = scratch<float>("msa.w", (size_t)heads * P);
  pwaSoftmaxK<<<(unsigned)(heads * T), 128, 0, STREAM>>>(flat, w, T, heads);
  // each row's own values, weighted sums, gate and output: a block of rows at a time
  int Sc = (int)std::max<size_t>(1, std::min<size_t>(S, ((size_t)128 << 20) / ((size_t)T * Wd * 4)));
  float* v = scratch<float>("msa.v", (size_t)Sc * T * Wd); float* vh = scratch<float>("msa.vh", (size_t)Sc * T * Wd);
  float* oh = scratch<float>("msa.oh", (size_t)Sc * T * Wd); float* g = scratch<float>("msa.g", (size_t)Sc * T * Wd);
  const float one = 1.f, zero = 0.f;
  for (int s0 = 0; s0 < S; s0 += Sc) {
    int sc = std::min(Sc, S - s0); size_t cr = (size_t)sc * T;
    const float* lnc = ln + (size_t)s0 * T * Cm;
    gemm(lnc, F(Bn + "value"), v, cr, Cm, Wd);
    pwaToHeadsK<<<blocks(cr * Wd), 256, 0, STREAM>>>(v, vh, sc, T, heads, d);
    // per head: O_h (T x sc d) = W_h (T x T) V_h (T x sc d); col-major O^T = V^T W^T
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, sc * d, T, T, &one, vh, sc * d, (long long)T * sc * d, w, T,
                                 (long long)P, &zero, oh, sc * d, (long long)T * sc * d, heads));
    gemm(lnc, F(Bn + "gate"), g, cr, Cm, Wd);
    pwaFromHeadsK<<<blocks(cr * Wd), 256, 0, STREAM>>>(oh, g, v, sc, T, heads, d);
    gemm(v, F(Bn + "output"), m + (size_t)s0 * T * Cm, cr, Wd, Cm, 1.f);
  }
}

// z_init (in, out): the encoder's pair. The alignment's rows past 1024 are subsampled keeping the query and
// the a3m's order, and 10% of the non-query rows' columns masked - both drawn from the seed, as biohub's
// processor does at inference (msa_max_depth 1024, msa_column_mask_rate 0.1); EF2_DETERMINISTIC=1 takes the
// first rows and masks nothing (af3-any-model's reading, for a tensor comparison)
inline void msaEncode(float* zi, const float* sInputs, int T, int C, int Si, uint64_t seed) {
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
  const int* rowsH = M.i("msa/rows"); const float* delH = M.f("msa/deletion");
  std::vector<int> rows((size_t)S * T); std::vector<float> del((size_t)S * T), mask((size_t)S * T, 1.f);
  for (int s = 0; s < S; ++s)
    for (int t = 0; t < T; ++t) { rows[(size_t)s * T + t] = rowsH[(size_t)keep[s] * T + t]; del[(size_t)s * T + t] = delH[(size_t)keep[s] * T + t]; }
  if (S > 1 && rate > 0.f) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    for (int t = 0; t < T; ++t)
      if (u(rng) < rate) for (int s = 1; s < S; ++s) mask[(size_t)s * T + t] = 0.f;
  }
  size_t R = (size_t)S * T, P = (size_t)T * T;
  int* rowsD = scratch<int>("msa.rows", R); float* delD = scratch<float>("msa.del", R); float* maskD = scratch<float>("msa.mask", R);
  CK(cudaMemcpyAsync(rowsD, rows.data(), R * 4, cudaMemcpyHostToDevice, STREAM));
  CK(cudaMemcpyAsync(delD, del.data(), R * 4, cudaMemcpyHostToDevice, STREAM));
  CK(cudaMemcpyAsync(maskD, mask.data(), R * 4, cudaMemcpyHostToDevice, STREAM));
  float* fromInputs = scratch<float>("msa.fromInputs", (size_t)T * Cm);
  gemm(sInputs, F("msaEncoder/projectInputs"), fromInputs, T, Si, Cm);
  float* m = scratch<float>("msa.m", R * Cm);
  msaEmbedEf2K<<<blocks(R * Cm), 256, 0, STREAM>>>(rowsD, delD, maskD, F("msaEncoder/embed"), fromInputs, m, R, T, Cm);
  float* pmask = scratch<float>("msa.pmask", P);
  fillK<<<blocks(P), 256, 0, STREAM>>>(pmask, 1.f, P);
  const int blocksN = (int)M.meta("meta/msaBlocks");
  for (int b = 0; b < blocksN; ++b) {
    std::string Bn = "msaEncoder/blocks/" + std::to_string(b) + "/";
    msaOuterProductMean(zi, m, maskD, S, T, Cm, C, Bn + "outerProductMean/");
    if (M.has("f/" + Bn + "pairWeightedAveraging/value")) {
      msaPairWeightedAveraging(m, zi, S, T, Cm, C, Bn + "pairWeightedAveraging/");
      rowsTransition(m, R, Cm, Bn + "msaTransition/");
    }
    trunkBlock(zi, pmask, T, C, "msaEncoder/blocks", b);
  }
  releaseScratch({ "msa." });
}
