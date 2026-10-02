// Synthyra's ESMFold2 confidence head (biohub ships the checkpoint without one; Synthyra trained it on
// the same frozen trunk). src/esmfold2/confidence-reference.js and their ConfidenceHead are the reading:
//
//   s = LN(s_inputs);  z = LN(z_trunk) + rel_pos + bonds + s_to_z(s)_i + s_to_z_T(s)_j + prod_out(in1(s)_i * in2(s)_j)
//   z += distance_embed[#(|x_rep_i - x_rep_j| > boundaries)]
//   z = z + trunk_4_blocks(z)                       (the trunk's own pair block, its own weights)
//   single = (softmax_j(z @ attn) z) @ out          (row-attention pooling)
//   pLDDT per atom = E[bin] over [0, 1] of LN(single[token]) @ plddtWeight[slot in token]
//   PAE = E[bin] over [0, 32] of z @ pae;  pTM / ipTM from the PAE logits (d0 from the live token count)
#pragma once
#include "trunk.cuh"

__global__ void confZK(float* z, RelIdx rel, const float* bonds, const float* wBond, const float* rows,
                       const float* cols, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  z[t] += relPosAt(rel, i, j, c, C) + bonds[ij] * wBond[c] + rows[(size_t)i * C + c] + cols[(size_t)j * C + c];
}
__global__ void outerProductK(const float* a, const float* b, float* out, size_t p0, size_t n, int T, int C) {   // a_i * b_j, rows p0..
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * C) return;
  int c = (int)(t % C); size_t ij = p0 + t / C; int i = (int)(ij / T), j = (int)(ij % T);
  out[t] = a[(size_t)i * C + c] * b[(size_t)j * C + c];
}
__global__ void distanceEmbedK(float* z, const float* x, const int* rep, const float* edges, int nEdges,
                               const float* table, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  const float* a = x + (size_t)rep[i] * 3; const float* b = x + (size_t)rep[j] * 3;
  float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  float d = sqrtf(dx * dx + dy * dy + dz * dz);
  int bucket = 0;
  for (int e = 0; e < nEdges; ++e) bucket += d > edges[e];
  z[t] += table[(size_t)bucket * C + c];
}
// row-attention pooling: pooled[i] = sum_j softmax_j(score[i, j]) z[i, j]   (a block per row, C <= 1024)
__global__ void rowPoolK(const float* z, const float* score, float* pooled, int T, int C) {
  int i = blockIdx.x;
  __shared__ float m, s;
  if (threadIdx.x == 0) {
    float mx = -INFINITY;
    for (int j = 0; j < T; ++j) mx = fmaxf(mx, score[(size_t)i * T + j]);
    float sum = 0;
    for (int j = 0; j < T; ++j) sum += expf(score[(size_t)i * T + j] - mx);
    m = mx; s = sum;
  }
  __syncthreads();
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = 0;
    for (int j = 0; j < T; ++j) acc += expf(score[(size_t)i * T + j] - m) / s * z[((size_t)i * T + j) * C + c];
    pooled[(size_t)i * C + c] = acc;
  }
}
__global__ void gatherAtomsK(const float* tok, const int* atomToToken, float* out, int A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)A * C) out[t] = tok[(size_t)atomToToken[t / C] * C + t % C];
}
// per atom: logits over bins from its slot's table, then the expectation over [0, 1]
__global__ void plddtAtomK(const float* s, const int* slot, const float* table, float* plddt, int A, int C, int bins) {
  int a = blockIdx.x;
  __shared__ float logit[64];
  const float* tb = table + (size_t)slot[a] * C * bins;
  for (int b = threadIdx.x; b < bins; b += blockDim.x) {
    float acc = 0;
    for (int c = 0; c < C; ++c) acc += s[(size_t)a * C + c] * tb[(size_t)c * bins + b];
    logit[b] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float mx = -INFINITY; for (int b = 0; b < bins; ++b) mx = fmaxf(mx, logit[b]);
    float tot = 0, e = 0;
    for (int b = 0; b < bins; ++b) { float w = expf(logit[b] - mx); tot += w; e += w * (b + 0.5f) / bins; }
    plddt[a] = e / tot;
  }
}

struct Confidence { std::vector<float> plddtAtom, plddtToken, pae; double ptm, iptm, meanPlddt; };

inline Confidence confidenceHead(int T, int A, const float* zTrunk, const float* sInputs, int Si, const float* xDevice,
                                 bool check) {
  int C = (int)dimOf("f/confidence/sToZ", 1), Cs = (int)dimOf("f/confidence/poolingOutput", 1);
  size_t P = (size_t)T * T;
  float* s = scratch<float>("cf.s", (size_t)T * Si);
  layerNorm(sInputs, s, T, Si, F("confidence/sInputsNorm/scale"), F("confidence/sInputsNorm/offset"));
  float* z = dalloc(P * C);
  layerNorm(zTrunk, z, P, C, F("confidence/zNorm/scale"), F("confidence/zNorm/offset"));
  float* r = scratch<float>("cf.r", (size_t)T * C); float* c = scratch<float>("cf.c", (size_t)T * C);
  float* l = scratch<float>("cf.l", (size_t)T * C); float* rr = scratch<float>("cf.rr", (size_t)T * C);
  gemm(s, F("confidence/sToZ"), r, T, Si, C);
  gemm(s, F("confidence/sToZTranspose"), c, T, Si, C);
  gemm(s, F("confidence/sToZProdIn1"), l, T, Si, C);
  gemm(s, F("confidence/sToZProdIn2"), rr, T, Si, C);
  confZK<<<blocks(P * C), 256, 0, STREAM>>>(z, relIdx(), W("token_bonds"), F("featuriser/tokenBonds"), r, c, T, C);
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));
  float* prod = scratch<float>("cf.prod", chunk * C);
  for (size_t p0 = 0; p0 < P; p0 += chunk) {
    size_t n = std::min(chunk, P - p0);
    outerProductK<<<blocks(n * C), 256, 0, STREAM>>>(l, rr, prod, p0, n, T, C);
    gemm(prod, F("confidence/sToZProdOut"), z + p0 * C, n, C, C, 1.f);
  }
  distanceEmbedK<<<blocks(P * C), 256, 0, STREAM>>>(z, xDevice, Idev("distogram_atom_idx"), F("confidence/boundaries"),
    (int)M.len("f/confidence/boundaries"), F("confidence/distanceEmbedding"), T, C);
  // z + trunk(z)
  float* stack = dalloc(P * C);
  CK(cudaMemcpyAsync(stack, z, P * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  float* mask = scratch<float>("trunk.mask", P);
  fillK<<<blocks(P), 256, 0, STREAM>>>(mask, 1.f, P);
  for (int b = 0; M.has("f/confidence/blocks/" + std::to_string(b) + "/pairTransition/transition1"); ++b)
    trunkBlock(stack, mask, T, C, "confidence/blocks", b);
  addK<<<blocks(P * C), 256, 0, STREAM>>>(z, stack, P * C);
  CK(cudaFree(stack));
  // single by row-attention pooling
  float* score = scratch<float>("cf.score", P);
  gemm(z, F("confidence/poolingAttention"), score, P, C, 1);
  float* pooled = scratch<float>("cf.pooled", (size_t)T * C); float* single = scratch<float>("cf.single", (size_t)T * Cs);
  rowPoolK<<<T, 256, 0, STREAM>>>(z, score, pooled, T, C);
  gemm(pooled, F("confidence/poolingOutput"), single, T, C, Cs);
  // pLDDT per atom (a table a slot: the atom's position within its token, clamped)
  int bins = (int)dimOf("f/confidence/plddtWeight", 2), slots = (int)dimOf("f/confidence/plddtWeight", 0);
  std::vector<int> a2t(A), slot(A);
  CK(cudaMemcpy(a2t.data(), Idev("atom_to_token"), A * 4, cudaMemcpyDeviceToHost));
  for (int a = 0, k = 0; a < A; ++a) { if (a > 0 && a2t[a] != a2t[a - 1]) k = 0; slot[a] = std::min(k++, slots - 1); }
  int* dSlot = upload(slot.data(), A);
  float* sa = scratch<float>("cf.sa", (size_t)A * Cs); float* pa = scratch<float>("cf.pa", A);
  gatherAtomsK<<<blocks((size_t)A * Cs), 256, 0, STREAM>>>(single, Idev("atom_to_token"), sa, A, Cs);
  layerNorm(sa, sa, A, Cs, F("confidence/plddtNorm/scale"), F("confidence/plddtNorm/offset"));
  plddtAtomK<<<A, 64, 0, STREAM>>>(sa, dSlot, F("confidence/plddtWeight"), pa, A, Cs, bins);
  // PAE logits
  int pb = (int)dimOf("f/confidence/pae", 1);
  float* paeL = scratch<float>("cf.pae", P * pb);
  gemm(z, F("confidence/pae"), paeL, P, C, pb);
  Confidence out;
  out.plddtAtom = download(pa, A);
  std::vector<float> mask_ = download(W("atom_mask"), A), logits = download(paeL, P * pb);
  std::vector<int> asym(T); CK(cudaMemcpy(asym.data(), Idev("asym_id"), T * 4, cudaMemcpyDeviceToHost));
  out.plddtToken.assign(T, 0.f);
  std::vector<float> cnt(T, 0.f);
  double wsum = 0, wtot = 0;
  for (int a = 0; a < A; ++a) {
    out.plddtToken[a2t[a]] += out.plddtAtom[a] * mask_[a]; cnt[a2t[a]] += mask_[a];
    wsum += out.plddtAtom[a] * mask_[a]; wtot += mask_[a];
  }
  for (int t = 0; t < T; ++t) out.plddtToken[t] /= std::max(cnt[t], 1e-6f);
  out.meanPlddt = wsum / (wtot + 1e-8);
  // PAE and pTM / ipTM
  double width = 32.0 / pb, d0 = 1.24 * cbrt(std::max(T, 19) - 15.0) - 1.8;
  out.pae.assign(P, 0.f);
  double ptm = -1e30, iptm = -1e30;
  for (int i = 0; i < T; ++i) {
    double sum = 0, count = 0, isum = 0, icount = 0;
    for (int j = 0; j < T; ++j) {
      const float* lg = &logits[((size_t)i * T + j) * pb];
      double mx = -1e30; for (int b = 0; b < pb; ++b) mx = std::max(mx, (double)lg[b]);
      double tot = 0, mean = 0, tm = 0;
      for (int b = 0; b < pb; ++b) {
        double w = exp(lg[b] - mx), centre = width * (b + 0.5);
        tot += w; mean += w * centre; tm += w / (1 + (centre / d0) * (centre / d0));
      }
      out.pae[(size_t)i * T + j] = (float)(mean / tot);
      sum += tm / tot; count += 1;
      if (asym[i] != asym[j]) { isum += tm / tot; icount += 1; }
    }
    ptm = std::max(ptm, sum / (count + 1e-8));
    iptm = std::max(iptm, isum / (icount + 1e-8));
  }
  out.ptm = ptm; out.iptm = iptm;
  if (check) {
    checkOracle("confidence pLDDT per atom", pa, A, "o/conf/plddt_per_atom");
    checkOracle("confidence PAE logits", paeL, P * pb, "o/conf/pae_logits");
    if (M.has("o/conf/ptm")) printf("  pTM %.6f against the oracle's %.6f\n", out.ptm, M.f("o/conf/ptm")[0]);
  }
  CK(cudaFree(z)); CK(cudaFree(dSlot));
  return out;
}
