// z_init, the recycle and the 24-block pair trunk, and the distogram (src/esmfold2/fold.js and
// src/esmfold2/pair-features-reference.js are the reading):
//
//   z_init = z_init_1(s)[i] + z_init_2(s)[j] + rel_pos + token_bonds + lm_z
//   z = 0;  num_loops + 1 times:  z = z_init + Linear(LN(z));  z = trunk(z)
//   trunk block: z += triangle out(z); z += triangle in(z); z += transition(z)      (no biases anywhere)
//   distogram = (z + z^T) @ W + b
// The trunk is AF3's pairformer block with its grid attentions and single track removed, and the
// bundle carries it in AF3's shapes: a triangle's projection and gate interleave a and b per channel,
// the transition's widening is [gate | value].
#pragma once
#include "fused256.cuh"

// relative position bins (residue 66, token 66, same entity 1, chain 6 = 139), one-hot rows summed;
// computed where it is read (z_init, the diffusion's conditioning, the confidence head), never stored
struct RelIdx { const int *ri, *asym, *sym, *ent, *ti; const float* Wt; };
inline RelIdx relIdx() {
  return {Idev("residue_index"), Idev("asym_id"), Idev("sym_id"), Idev("entity_id"), Idev("token_index"), F("featuriser/relPos")};
}
__device__ __forceinline__ float relPosAt(const RelIdx& r, int i, int j, int c, int C) {
  const int rb = 32, cb = 2;
  auto clip = [](int v, int hi) { return v < 0 ? 0 : v > hi ? hi : v; };
  bool sameChain = r.asym[i] == r.asym[j], sameRes = r.ri[i] == r.ri[j];
  int b0 = sameChain ? clip(r.ri[i] - r.ri[j] + rb, 2 * rb) : 2 * rb + 1;
  int b1 = sameChain && sameRes ? clip(r.ti[i] - r.ti[j] + rb, 2 * rb) : 2 * rb + 1;
  int b3 = sameChain ? 2 * cb + 1 : clip(r.sym[i] - r.sym[j] + cb, 2 * cb);
  const int w = 2 * rb + 2;
  float v = r.Wt[(size_t)b0 * C + c] + r.Wt[(size_t)(w + b1) * C + c] + r.Wt[(size_t)(2 * w + 1 + b3) * C + c];
  if (r.ent[i] == r.ent[j]) v += r.Wt[(size_t)(2 * w) * C + c];
  return v;
}
__global__ void relPosK(RelIdx r, float* out, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // one thread per (pair, channel)
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C;
  out[t] = relPosAt(r, (int)(ij / T), (int)(ij % T), c, C);
}
// z_init = rows[i] + cols[j] + relpos + bonds * w_bond + lm_z
__global__ void zInitK(const float* rows, const float* cols, RelIdx rel, const float* bonds, const float* wBond,
                       const float* lmZ, float* z, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  z[t] = rows[(size_t)i * C + c] + cols[(size_t)j * C + c] + relPosAt(rel, i, j, c, C) + bonds[ij] * wBond[c] + lmZ[t];
}

inline void zInit(int T, int C, const float* sInputs, int Si, const float* lmZ, float* z, bool check) {
  size_t P = (size_t)T * T;
  float* rows = scratch<float>("zi.rows", (size_t)T * C); float* cols = scratch<float>("zi.cols", (size_t)T * C);
  gemm(sInputs, F("featuriser/zInit1"), rows, T, Si, C);
  gemm(sInputs, F("featuriser/zInit2"), cols, T, Si, C);
  if (check) {
    checkOracle("z_init_1", rows, (size_t)T * C, "o/z_init_1");
    checkOracle("z_init_2", cols, (size_t)T * C, "o/z_init_2");
    float* rel = scratch<float>("zi.rel", P * C);
    relPosK<<<blocks(P * C), 256, 0, STREAM>>>(relIdx(), rel, T, C);
    checkOracle("rel_pos", rel, P * C, "o/rel_pos");
  }
  zInitK<<<blocks(P * C), 256, 0, STREAM>>>(rows, cols, relIdx(), W("token_bonds"), F("featuriser/tokenBonds"), lmZ, z, T, C);
}

// ---------------------------------------------------------------- the trunk, float32
__global__ void triSplitInterleavedK(const float* proj, const float* gate, const float* mask, float* a, float* b,
                                     size_t pairs, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * 2 * C) return;
  int c2 = (int)(t % (2 * C)); size_t ij = t / (2 * C);
  float v = proj[t] * mask[ij] / (1.f + expf(-gate[t]));
  (c2 & 1 ? b : a)[(size_t)(c2 >> 1) * pairs + ij] = v;
}
__global__ void fillK(float* out, float v, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = v;
}

inline void triangle(float* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing) {
  size_t P = (size_t)L * L;
  float* xn = scratch<float>("tri.xn", P * C);
  layerNorm(pair, xn, P, C, F(Tn + "leftNormInputScale"), F(Tn + "leftNormInputOffset"));
  float* proj = scratch<float>("tri.proj", P * 2 * C); float* gate = scratch<float>("tri.gate", P * 2 * C);
  gemm(xn, F(Tn + "projection"), proj, P, C, 2 * C);
  gemm(xn, F(Tn + "gate"), gate, P, C, 2 * C);
  float* a = scratch<float>("tri.a", P * C); float* b = scratch<float>("tri.b", P * C);
  triSplitInterleavedK<<<blocks(P * 2 * C), 256, 0, STREAM>>>(proj, gate, mask, a, b, P, C);
  float* prod = scratch<float>("tri.prod", P * C);
  const float one = 1.f, zero = 0.f;
  // per channel (row-major [i][k] planes): outgoing out[i,j] = sum_k a[i,k] b[j,k]; incoming sum_k a[k,j] b[k,i]
  if (outgoing)
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, L, L, L, &one, b, L, P, a, L, P, &zero, prod, L, P, C));
  else
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_T, L, L, L, &one, a, L, P, b, L, P, &zero, prod, L, P, C));
  float* rowsP = scratch<float>("tri.rows", P * C);
  channelMajorToRowsK<<<blocks(P * C), 256, 0, STREAM>>>(prod, rowsP, P, C);
  layerNorm(rowsP, rowsP, P, C, F(Tn + "centerNormScale"), F(Tn + "centerNormOffset"));
  float* out = scratch<float>("tri.out", P * C);
  gemm(rowsP, F(Tn + "outputProjection"), out, P, C, C);
  float* g = scratch<float>("tri.g", P * C);
  gemm(xn, F(Tn + "gatingLinear"), g, P, C, C);
  gateMulAddK<<<blocks(P * C), 256, 0, STREAM>>>(pair, out, g, P * C);
}
inline void pairTransition(float* pair, int L, int C, const std::string& Tn) {
  size_t P = (size_t)L * L; int I = (int)dimOf("f/" + Tn + "transition2", 0);
  float* xn = scratch<float>("tr.xn", P * C);
  layerNorm(pair, xn, P, C, F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"));
  // a chunk of rows at a time: the widened rows are 2I per pair
  size_t chunk = std::max<size_t>(1, ((size_t)64 << 20) / (2 * (size_t)I));
  float* h = scratch<float>("tr.h", std::min(P, chunk) * 2 * I); float* g = scratch<float>("tr.g", std::min(P, chunk) * I);
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    gemm(xn + r0 * C, F(Tn + "transition1"), h, r, C, 2 * I);
    swigluK<<<blocks(r * I), 256, 0, STREAM>>>(h, g, r, I);
    gemm(g, F(Tn + "transition2"), pair + r0 * C, r, I, C, 1.f);
  }
}
// the 256-channel block on the fused kernels (fused256.cuh): triInK -> the f16 contraction -> triangleOutK,
// and transitionUpK -> the second GEMM (cuBLASLt, the residual as beta)
inline void triangle256(float* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing) {
  int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
  half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
  half* t2 = scratch<half>("ftri.t2", plane * C);
  triIn256<8>(pair, mask, Tn, a, b, t2, L, Lp, plane);
  float* prod = scratch<float>("ftri.prod", plane * C);
  const float one = 1.f, zero = 0.f;
  if (outgoing)
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16F, Lp, plane, a, CUDA_R_16F, Lp,
                                  plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  else
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16F, Lp, plane, b, CUDA_R_16F, Lp,
                                  plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  triangleOut<4>(prod, F(Tn + "centerNormScale"), F(Tn + "centerNormOffset"), Fh(Tn + "outputProjection"), t2, pair, L, Lp);
}
inline void transition256(float* pair, size_t P, int C, const std::string& Tn) {
  int I = (int)dimOf("f/" + Tn + "transition2", 0);
  // whole waves of transitionUpK within the same ~64 MB of widened rows (see transitionUpWaveRows)
  size_t wave = transitionUpWaveRows<8>(), budget = ((size_t)64 << 20) / (2 * (size_t)I);
  size_t chunk = std::max<size_t>(1, budget / wave) * wave;
  half* g = scratch<half>("ftr.g", std::min(P, chunk) * I);
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    transitionUp<8>(pair + r0 * C, F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"), Fh(Tn + "transition1"), g, r, I);
    ltGemm(g, Fh(Tn + "transition2"), pair + r0 * C, false, r, I, C, nullptr, false, 1.f);
  }
}
inline void trunkBlock(float* pair, const float* mask, int L, int C, const std::string& prefix, int b) {
  std::string B = prefix + "/" + std::to_string(b) + "/";
  // the fused kernels' 64-128-pair tiles leave a small pair track's device idle: measured, warm, trunk of
  // 4 passes - 68 tokens 43.2 ms unfused against 52.7 fused; 92: 71.5 / 58.3; 195: 201 / 190;
  // 261: 368 / 320; 476: 1131 / 953
  if (FAST && FUSED256 && C == 256 && L >= FUSED256_MIN_TOKENS && fused256Fits()) {
    triangle256(pair, mask, L, C, B + "triangleMultiplicationOutgoing/", true);
    triangle256(pair, mask, L, C, B + "triangleMultiplicationIncoming/", false);
    transition256(pair, (size_t)L * L, C, B + "pairTransition/");
    return;
  }
  if (FAST) {
    triangleFast(pair, mask, L, C, B + "triangleMultiplicationOutgoing/", true);
    triangleFast(pair, mask, L, C, B + "triangleMultiplicationIncoming/", false);
    transitionFast(pair, (size_t)L * L, C, B + "pairTransition/");
    return;
  }
  triangle(pair, mask, L, C, B + "triangleMultiplicationOutgoing/", true);
  triangle(pair, mask, L, C, B + "triangleMultiplicationIncoming/", false);
  pairTransition(pair, L, C, B + "pairTransition/");
}

// the recycle loop: z (out) = the trunk's last pass
inline void foldingTrunk(int T, int C, const float* zInitP, float* z, int loops, bool check) {
  size_t P = (size_t)T * T;
  float* mask = scratch<float>("trunk.mask", P);
  fillK<<<blocks(P), 256, 0, STREAM>>>(mask, 1.f, P);      // every token is real (the token mask is all ones)
  CK(cudaMemsetAsync(z, 0, P * C * 4, STREAM));
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));   // the recycle row by row, 64 MB at a time
  float* xn = scratch<float>("trunk.xn", chunk * C);
  int blocksN = (int)M.meta("meta/blocks");
  for (int loop = 0; loop < loops; ++loop) {
    for (size_t r0 = 0; r0 < P; r0 += chunk) {
      size_t r = std::min(chunk, P - r0);
      layerNorm(z + r0 * C, xn, r, C, F("recycle/norm/scale"), F("recycle/norm/offset"));
      gemm(xn, F("recycle/projection"), z + r0 * C, r, C, C);
    }
    addK<<<blocks(P * C), 256, 0, STREAM>>>(z, zInitP, P * C);
    if (check) checkOracle(("trunk pass " + std::to_string(loop) + " in").c_str(), z, P * C, "o/loop" + std::to_string(loop) + "/in");
    // (a CUDA graph of the 24 blocks, replayed for passes after the first, measured no faster: 453
    // against 455 ms of trunk at 261 tokens; the GPU is busy, the launches are not the cost)
    for (int b = 0; b < blocksN; ++b) trunkBlock(z, mask, T, C, "blocks", b);
    if (check) checkOracle(("trunk pass " + std::to_string(loop) + " out").c_str(), z, P * C, "o/loop" + std::to_string(loop) + "/out");
    if (getenv("EF2_PASS_TIMES")) {
      static auto t0 = std::chrono::steady_clock::now();
      CK(cudaStreamSynchronize(STREAM));
      printf("    pass %d done at %.1f ms\n", loop, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
  }
}

inline void distogram(const float* z, int T, int C, float* logits) {
  size_t P = (size_t)T * T; int Bn = (int)dimOf("f/distogram/weights", 1);
  float* zs = scratch<float>("dg.sym", P * C);
  symmetriseK<<<blocks(P * C), 256, 0, STREAM>>>(z, zs, T, C);
  gemm(zs, F("distogram/weights"), logits, P, C, Bn);
  addBias(logits, F("distogram/bias"), P, Bn);
}
