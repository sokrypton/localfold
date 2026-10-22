// One call of BioEmu2's denoiser, several samples at once (see denoise.cuh for the model). Everything that depends on
// sigma alone - the single conditioning and every layer's conditioning projections, on the token side AND the atom
// side - is computed once a call and shared by every sample; each atom row reads its token's (CondLayer's map).
#pragma once
#include "denoise.cuh"

namespace be2 {

// the row maps of `samples` stacked samples: each atom to its token's conditioning row (padding atoms to 0), each
// token to its own
struct RowMaps { int samples = 0; int* atomTok = nullptr; int* tok = nullptr; };
inline RowMaps ROW_MAPS;
inline const RowMaps& rowMaps(int samples, int T, int N) {
  if (ROW_MAPS.samples == samples && ROW_MAPS.atomTok) return ROW_MAPS;
  if (ROW_MAPS.atomTok) { CK(cudaFree(ROW_MAPS.atomTok)); CK(cudaFree(ROW_MAPS.tok)); }
  const int* tok = M.i("token_idx_at_atom");
  std::vector<int> a((size_t)samples * N), t((size_t)samples * T);
  for (size_t r = 0; r < a.size(); ++r) a[r] = tok[r % N] >= 0 ? tok[r % N] : T;   // (a padding atom: the zero row T)
  for (size_t r = 0; r < t.size(); ++r) t[r] = (int)(r % T);
  ROW_MAPS = {samples, upload(a.data(), a.size()), upload(t.data(), t.size())};
  return ROW_MAPS;
}
inline void forgetRowMaps() {
  if (ROW_MAPS.atomTok) { CK(cudaStreamSynchronize(STREAM)); CK(cudaFree(ROW_MAPS.atomTok)); CK(cudaFree(ROW_MAPS.tok)); }
  ROW_MAPS = {};
}

__global__ void scaleMaskedBatchK(const float* x, const int* mask, float* r, size_t N, size_t n, float s) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n * 3) r[t] = mask[(t / 3) % N] ? x[t] * s : 0.f;
}
__global__ void repeatRowsK(const float* __restrict__ src, float* __restrict__ dst, size_t elems, size_t total) {   // dst = src, src, ...
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((elems & 3) == 0 && (total & 3) == 0 && (((uintptr_t)src | (uintptr_t)dst) & 15) == 0) {
    size_t e4 = elems >> 2, tot4 = total >> 2;
    if (t < tot4) {
      reinterpret_cast<float4*>(dst)[t] = __ldg(reinterpret_cast<const float4*>(src) + (t % e4));
    }
    return;
  }
  if (t < total) dst[t] = src[t % elems];
}
inline void repeatRows(const float* src, float* dst, size_t elems, size_t total) {
  size_t nt = ((elems & 3) == 0 && (total & 3) == 0 && (((uintptr_t)src | (uintptr_t)dst) & 15) == 0) ? (total >> 2) : total;
  repeatRowsK<<<blocks(nt), 256, 0, STREAM>>>(src, dst, elems, total);
}

__global__ void initCondK(const float* __restrict__ sc0, const float* __restrict__ f768,
                          float* __restrict__ cnd, int T, int Dt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int d4 = Dt >> 2;
  size_t n4 = (size_t)(T + 1) * d4;
  if (t >= n4) return;
  size_t r = t / (size_t)d4;
  int c4 = (int)(t - r * (size_t)d4);
  if (r < (size_t)T) {
    float4 sv = __ldg(reinterpret_cast<const float4*>(sc0) + t);
    float4 fv = __ldg(reinterpret_cast<const float4*>(f768) + c4);
    reinterpret_cast<float4*>(cnd)[t] = make_float4(sv.x + fv.x, sv.y + fv.y, sv.z + fv.z, sv.w + fv.w);
  } else {
    reinterpret_cast<float4*>(cnd)[t] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
}

__global__ void cosEmbedBatchK(const float* __restrict__ w, const float* __restrict__ b,
                               float* __restrict__ out, const float* __restrict__ tArr, int steps, int F) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= steps * F) return;
  int s = idx / F, k = idx - s * F;
  float t = __ldg(tArr + s);
  out[idx] = fastCosF(2.f * (float)M_PI * (t * __ldg(w + k) + __ldg(b + k)));
}

__global__ void initCondPackedBatchK(const float* __restrict__ sc0, const float* __restrict__ f768,
                                     float* __restrict__ cndPacked, int steps, int T, int Dt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int d4 = Dt >> 2;
  size_t td4 = (size_t)T * d4;
  size_t n4 = (size_t)steps * td4;
  if (t >= n4) return;
  size_t s = t / td4, rem = t - s * td4;
  int c4 = (int)(rem % (size_t)d4);
  float4 sv = __ldg(reinterpret_cast<const float4*>(sc0) + rem);
  float4 fv = __ldg(reinterpret_cast<const float4*>(f768) + s * (size_t)d4 + c4);
  reinterpret_cast<float4*>(cndPacked)[t] = make_float4(sv.x + fv.x, sv.y + fv.y, sv.z + fv.z, sv.w + fv.w);
}

__global__ void padCondBatchK(const float* __restrict__ cndPacked, float* __restrict__ cndAll,
                              int steps, int T, int Dt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int d4 = Dt >> 2;
  size_t stride4 = (size_t)(T + 1) * d4;
  size_t n4 = (size_t)steps * stride4;
  if (t >= n4) return;
  size_t s = t / stride4, rem = t - s * stride4;
  size_t r = rem / (size_t)d4;
  if (r < (size_t)T) {
    reinterpret_cast<float4*>(cndAll)[t] = __ldg(reinterpret_cast<const float4*>(cndPacked) + s * (size_t)T * d4 + rem);
  } else {
    reinterpret_cast<float4*>(cndAll)[t] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
}

struct StepCond {
  const float* condAAll = nullptr;
  const float* condTAll = nullptr;
  const float* cqcAll   = nullptr;
  const float* cscAll   = nullptr;
  int stepBatch = 1;
  int stepIdx = 0;
};

inline float* precomputeAllSingleCond(const Denoiser& d, const std::vector<float>& hTEmb) {
  const int steps = (int)hTEmb.size(), T = d.T, Dt = 768;
  const std::string sc = SM + "single_conditioner/";
  const int F = dimP(sc + "fourier_embed/proj/bias", 0);
  float* tDev = scratch<float>("be2.sc.tDev", steps);
  CK(cudaMemcpyAsync(tDev, hTEmb.data(), (size_t)steps * 4, cudaMemcpyHostToDevice, STREAM));
  float* four = scratch<float>("be2.sc.fourB", (size_t)steps * F);
  cosEmbedBatchK<<<blocks((size_t)steps * F), 256, 0, STREAM>>>(P(sc + "fourier_embed/proj/weight"),
                                                                P(sc + "fourier_embed/proj/bias"),
                                                                four, tDev, steps, F);
  layerNorm(four, four, steps, F, P(sc + "norm_fourier/weight"), P(sc + "norm_fourier/bias"));
  float* f768 = scratch<float>("be2.sc.f768B", (size_t)steps * Dt);
  linear(four, P(sc + "fourier_to_single/weight"), f768, steps, F, Dt);
  float* cndPacked = scratch<float>("be2.sc.cndPacked", (size_t)steps * T * Dt);
  initCondPackedBatchK<<<blocks(((size_t)steps * T * Dt) >> 2), 256, 0, STREAM>>>(d.sc0, f768, cndPacked, steps, T, Dt);
  for (int i = 0; i < 2; ++i)
    transition(cndPacked, cndPacked, (size_t)steps * T, Dt, sc + "transitions_" + std::to_string(i) + "/");
  float* cndAll = scratch<float>("be2.sc.cndAll", (size_t)steps * (T + 1) * Dt);
  padCondBatchK<<<blocks(((size_t)steps * (T + 1) * Dt) >> 2), 256, 0, STREAM>>>(cndPacked, cndAll, steps, T, Dt);
  return cndAll;
}

inline void computeStepCondBatch(const Denoiser& d, const float* cndBatch, int B,
                                 float* condAAll, float* condTAll, float* cqcAll, float* cscAll) {
  const int T = d.T, Ca = 128, Dt = 768, Dc = Dt + Ca;
  const std::string L = SM + "layers/";
  const std::string cp = L + "coupling/";
  const std::string ap = L + "atom_block/transformer_layer/", tp = L + "token_block/";
  Cond0Tables c0Tab{
    {P(ap + "cond_0/weight"),
     P(tp + "cond_0/weight"),
     P(cp + "q_coupling_cond_0/weight"),
     P(cp + "s_coupling_cond_0/weight")},
    {Popt(ap + "cond_0/bias"),
     Popt(tp + "cond_0/bias"),
     Popt(cp + "q_coupling_cond_0/bias"),
     Popt(cp + "s_coupling_cond_0/bias")}
  };
  const int rows = B * (T + 1);
  const int rowsT = B == 1 ? T : rows;
  const size_t condAStride = (size_t)rows * (6 * Ca);
  const size_t condTStride = (size_t)rowsT * (6 * Dt);
  const size_t cqcStride   = (size_t)rows * (2 * Dc + Ca);
  const size_t cscStride   = (size_t)rowsT * (2 * Dc + Dt);
  const size_t cnhStride   = (size_t)rows * Dt;
  const size_t streamStride = (size_t)d.layers * cnhStride;
  dim3 c0Grid((unsigned)((rows + 7) / 8), (unsigned)d.layers);
  if (GEMM_HALF) {
    half* cnhAll = scratch<half>("be2.dn.cnhAll", 4 * streamStride);
    lnToCond4K<half><<<c0Grid, 256, 0, STREAM>>>(cndBatch, cnhAll, T, rows, c0Tab);
    linearBatchPre<half>(cnhAll + 0 * streamStride, (long long)cnhStride,
                         halfMirror<half>(P(ap + "cond_2/weight"), (size_t)d.layers * (6 * Ca) * Dt),
                         (long long)(6 * Ca) * Dt, condAAll, (long long)condAStride, rows, Dt, 6 * Ca, d.layers);
    linearBatchPre<half>(cnhAll + 1 * streamStride, (long long)cnhStride,
                         halfMirror<half>(P(tp + "cond_2/weight"), (size_t)d.layers * (6 * Dt) * Dt),
                         (long long)(6 * Dt) * Dt, condTAll, (long long)condTStride, rowsT, Dt, 6 * Dt, d.layers);
    linearBatchPre<half>(cnhAll + 2 * streamStride, (long long)cnhStride,
                         halfMirror<half>(P(cp + "q_coupling_cond_2/weight"), (size_t)d.layers * (2 * Dc + Ca) * Dt),
                         (long long)(2 * Dc + Ca) * Dt, cqcAll, (long long)cqcStride, rows, Dt, 2 * Dc + Ca, d.layers);
    linearBatchPre<half>(cnhAll + 3 * streamStride, (long long)cnhStride,
                         halfMirror<half>(P(cp + "s_coupling_cond_2/weight"), (size_t)d.layers * (2 * Dc + Dt) * Dt),
                         (long long)(2 * Dc + Dt) * Dt, cscAll, (long long)cscStride, rowsT, Dt, 2 * Dc + Dt, d.layers);
  } else {
    float* cnAll = scratch<float>("be2.dn.cnAll", 4 * streamStride);
    lnToCond4K<float><<<c0Grid, 256, 0, STREAM>>>(cndBatch, cnAll, T, rows, c0Tab);
    linearBatch(cnAll + 0 * streamStride, (long long)cnhStride, P(ap + "cond_2/weight"),
                (long long)(6 * Ca) * Dt, condAAll, (long long)condAStride, rows, Dt, 6 * Ca, d.layers);
    linearBatch(cnAll + 1 * streamStride, (long long)cnhStride, P(tp + "cond_2/weight"),
                (long long)(6 * Dt) * Dt, condTAll, (long long)condTStride, rowsT, Dt, 6 * Dt, d.layers);
    linearBatch(cnAll + 2 * streamStride, (long long)cnhStride, P(cp + "q_coupling_cond_2/weight"),
                (long long)(2 * Dc + Ca) * Dt, cqcAll, (long long)cqcStride, rows, Dt, 2 * Dc + Ca, d.layers);
    linearBatch(cnAll + 3 * streamStride, (long long)cnhStride, P(cp + "s_coupling_cond_2/weight"),
                (long long)(2 * Dc + Dt) * Dt, cscAll, (long long)cscStride, rowsT, Dt, 2 * Dc + Dt, d.layers);
  }
  addBiasBatchRows(condAAll, Popt(ap + "cond_2/bias"), rows, 6 * Ca, d.layers);
  addBiasBatchRows(condTAll, Popt(tp + "cond_2/bias"), rowsT, 6 * Dt, d.layers);
  addBiasBatchRows(cqcAll, Popt(cp + "q_coupling_cond_2/bias"), rows, 2 * Dc + Ca, d.layers);
  addBiasBatchRows(cscAll, Popt(cp + "s_coupling_cond_2/bias"), rowsT, 2 * Dc + Dt, d.layers);
}

// x [samples, N, 3] (noisy, angstrom) at sigma -> den [samples, N, 3]. `cond` (optional, [samples, N, 3]) is each
// sample's conditioning frame (propagation's companion, augmented with the state); null takes the featurised
// coords_cond for every sample
inline void denoise(const Denoiser& d, const float* x, float sigma, float* den, const float* cond = nullptr,
                    int samples = 1, const StepCond* preCond = nullptr) {
  const int T = d.T, N = d.N, Ca = 128, Dt = 768;
  const size_t RN = (size_t)samples * N, RT = (size_t)samples * T;
  const RowMaps& maps = rowMaps(samples, T, N);
  const std::string sc = SM + "single_conditioner/", af = SM + "atom_feature_embedder/";
  const float cIn = 1.f / sqrtf(sigma * sigma + SIGMA_DATA * SIGMA_DATA), t = 0.25f * logf(sigma / SIGMA_DATA);
  float* r = scratch<float>("be2.dn.r", RN * 3);
  scaleMaskedBatchK<<<blocks(RN * 3), 256, 0, STREAM>>>(x, Idev("atom_mask"), r, N, RN, cIn);
  float* cnd = nullptr;
  if (!preCond) {
    // the single conditioning (shared)
    const int F = dimP(sc + "fourier_embed/proj/bias", 0);
    float* four = scratch<float>("be2.dn.four", F);
    cosEmbedK<<<blocks(F), 256, 0, STREAM>>>(P(sc + "fourier_embed/proj/weight"), P(sc + "fourier_embed/proj/bias"), four, t, F);
    layerNorm(four, four, 1, F, P(sc + "norm_fourier/weight"), P(sc + "norm_fourier/bias"));
    float* f768 = scratch<float>("be2.dn.f768", Dt);
    linear(four, P(sc + "fourier_to_single/weight"), f768, 1, F, Dt);
    // cond, and a ZERO row after its T: what a padding atom of the reference conditions on (c_to_q is 0 there)
    cnd = scratch<float>("be2.dn.cond", (size_t)(T + 1) * Dt);
    initCondK<<<blocks(((size_t)(T + 1) * Dt) >> 2), 256, 0, STREAM>>>(d.sc0, f768, cnd, T, Dt);
    for (int i = 0; i < 2; ++i) transition(cnd, cnd, T, Dt, sc + "transitions_" + std::to_string(i) + "/");
  }
  // the atom stream's initial state
  float* q = scratch<float>("be2.dn.q", RN * Ca);
  repeatRows(d.q0, q, (size_t)N * Ca, RN * Ca);
  linear(r, P(af + "r_to_q_trans/weight"), q, RN, 3, Ca, 1.f);
  if (d.dyn) {
    if (cond) {
      linear(cond, P(af + "cond_to_q_trans/weight"), q, RN, 3, Ca, 1.f, 0, 0, d.dynGate);
    } else {
      float* cq = scratch<float>("be2.dn.cq", RN * Ca);
      linear(d.coordsCond, P(af + "cond_to_q_trans/weight"), cq, N, 3, Ca);
      repeatRows(cq, cq, (size_t)N * Ca, RN * Ca);
      scaledAddK<<<blocks((RN * Ca) >> 2), 256, 0, STREAM>>>(q, cq, d.dynGate, RN * Ca);
    }
  }
  float* a = scratch<float>("be2.dn.a", RT * Dt);
  repeatRows(d.a0, a, (size_t)T * Dt, RT * Dt);
  float* rc = scratch<float>("be2.dn.rc", RT * 3);
  centreAtoms(r, rc, T, 3, 0, samples, N);
  linear(rc, P(SM + "r_token_to_a/weight"), a, RT, 3, Dt, 1.f);
  // the layers
  const std::string L = SM + "layers/";
  const std::string ab = "attention/attention_layer/pair_bias_attn/", tb = "attention/pair_bias_attn/";
  const size_t Pn = (size_t)T * T, wb = (size_t)N * WK * 4, tbn = 16 * Pn;
  const int Dc = Dt + Ca;                                  // the coupling's width, 896
  float* tbias = d.tbias ? nullptr : scratch<float>("be2.dn.tbias", tbn);
  StepCond localCond;
  if (!preCond) {
    const size_t condAStride1 = (size_t)(T + 1) * (6 * Ca);
    const size_t condTStride1 = (size_t)T * (6 * Dt);
    const size_t cqcStride1   = (size_t)(T + 1) * (2 * Dc + Ca);
    const size_t cscStride1   = (size_t)T * (2 * Dc + Dt);
    float* condAAll = scratch<float>("be2.dn.condA", (size_t)d.layers * condAStride1);
    float* condTAll = scratch<float>("be2.dn.condT", (size_t)d.layers * condTStride1);
    float* cqcAll   = scratch<float>("be2.dn.cqc",   (size_t)d.layers * cqcStride1);
    float* cscAll   = scratch<float>("be2.dn.csc",   (size_t)d.layers * cscStride1);
    computeStepCondBatch(d, cnd, 1, condAAll, condTAll, cqcAll, cscAll);
    localCond = {condAAll, condTAll, cqcAll, cscAll, 1, 0};
    preCond = &localCond;
  }
  const int rowsT1 = preCond->stepBatch == 1 ? T : (T + 1);
  const size_t condAStride = (size_t)(T + 1) * (6 * Ca);
  const size_t condTStride = (size_t)rowsT1 * (6 * Dt);
  const size_t cqcStride   = (size_t)(T + 1) * (2 * Dc + Ca);
  const size_t cscStride   = (size_t)rowsT1 * (2 * Dc + Dt);
  float* dq = scratch<float>("be2.dn.dq", RN * Ca);
  float* da = scratch<float>("be2.dn.da", RT * Dt);
  float* tAtom = scratch<float>("be2.dn.tAtom", RN * Ca);
  float* tTok  = scratch<float>("be2.dn.tTok",  RT * Dt);
  const int* atomTokDev = Idev("token_idx_at_atom");
  const int* atomPadDev = Idev("atom_pad_mask");
  const int* centreDev = Idev("token_center_atom_index");
  const std::string cp = L + "coupling/";
  const std::string ap = L + "atom_block/transformer_layer/", tp = L + "token_block/";
  for (int l = 0; l < d.layers; ++l) {
    const size_t lb = (size_t)l * preCond->stepBatch + preCond->stepIdx;
    const size_t lbPrev = l > 0 ? ((size_t)(l - 1) * preCond->stepBatch + preCond->stepIdx) : 0;
    const float* condA = preCond->condAAll + lb * condAStride;
    const float* condT = preCond->condTAll + lb * condTStride;
    const float* cqc   = preCond->cqcAll   + lb * cqcStride;
    const float* csc   = preCond->cscAll   + lb * cscStride;
    const float* preGateQ = l > 0 ? (preCond->cqcAll + lbPrev * cqcStride + 2 * Dc) : nullptr;
    const float* preGateA = l > 0 ? (preCond->cscAll + lbPrev * cscStride + 2 * Dc) : nullptr;
    const CondLayer atomL{ap, ab, l, true, 4};
    condLayer(atomL, q, (int)RN, Ca, condA, maps.atomTok, d.wbias + wb * l, N, tAtom,
              l > 0 ? dq : nullptr, preGateQ, 2 * Dc + Ca);
    const float* bias = d.tbias ? d.tbias + tbn * l : tbias;
    if (!d.tbias) tokenBias(d.z, PL(tp + tb + "norm_z/weight", l), PL(tp + tb + "proj_z/weight", l), tbias, T, 256, 16);
    const CondLayer tokL{tp, tb, l, false, 16};
    condLayer(tokL, a, (int)RT, Dt, condT, maps.tok, bias, T, tTok,
              l > 0 ? da : nullptr, preGateA, 2 * Dc + Dt);
    if (GEMM_HALF) {
      // s_to_q = [a broadcast | q]; q_to_s = [q centre | a], fused with post-transition gatedAdd directly into adaLN
      half* sqh = scratch<half>("be2.dn.sqh", RN * Dc); half* qsh = scratch<half>("be2.dn.qsh", RT * Dc);
      adaLnQ2SK<half><<<(unsigned)((RT + 7) / 8), 256, 0, STREAM>>>(q, tAtom, condA + 5 * Ca, 6 * Ca, atomTokDev,
                                                                    a, tTok, condT + 5 * Dt, 6 * Dt, centreDev,
                                                                    qsh, RT, N, T, csc, csc + Dc, 2 * Dc + Dt, maps.tok);
      adaLnS2QK<half><<<(unsigned)((RN + 7) / 8), 256, 0, STREAM>>>(a, q, tAtom, condA + 5 * Ca, 6 * Ca,
                                                                    atomTokDev, atomPadDev,
                                                                    sqh, RN, N, T, cqc, cqc + Dc, 2 * Dc + Ca, maps.atomTok);
      transitionHalf(nullptr, sqh, dq, RN, Dc, cp + "s_to_q_mixer/", l, false, false, 0);
      transitionHalf(nullptr, qsh, da, RT, Dc, cp + "q_to_s_mixer/", l, false, false, 0);
    } else {
      float* sqn = scratch<float>("be2.dn.sqn", RN * Dc); float* qsn = scratch<float>("be2.dn.qsn", RT * Dc);
      adaLnQ2SK<float><<<(unsigned)((RT + 7) / 8), 256, 0, STREAM>>>(q, tAtom, condA + 5 * Ca, 6 * Ca, atomTokDev,
                                                                     a, tTok, condT + 5 * Dt, 6 * Dt, centreDev,
                                                                     qsn, RT, N, T, csc, csc + Dc, 2 * Dc + Dt, maps.tok);
      adaLnS2QK<float><<<(unsigned)((RN + 7) / 8), 256, 0, STREAM>>>(a, q, tAtom, condA + 5 * Ca, 6 * Ca,
                                                                     atomTokDev, atomPadDev,
                                                                     sqn, RN, N, T, cqc, cqc + Dc, 2 * Dc + Ca, maps.atomTok);
      transition(sqn, dq, RN, Dc, cp + "s_to_q_mixer/", l, false, false);
      transition(qsn, da, RT, Dc, cp + "q_to_s_mixer/", l, false, false);
    }
    if (l + 1 == d.layers) {
      gatedAdd(q, dq, cqc + 2 * Dc, RN, Ca, 2 * Dc + Ca, maps.atomTok);
      gatedAdd(a, da, csc + 2 * Dc, RT, Dt, 2 * Dc + Dt, maps.tok);
    }
  }
  // the readout
  const std::string ro = SM + "readout/";
  const int Hd = dimP(ro + "q_to_hidden/weight", 0);
  float* h = scratch<float>("be2.dn.h", RN * Hd);
  float* ah = scratch<float>("be2.dn.ah", RT * Hd);
  linear(q, P(ro + "q_to_hidden/weight"), h, RN, Ca, Hd);
  linear(a, P(ro + "s_to_hidden/weight"), ah, RT, Dt, Hd);
  tokensToAtoms(ah, h, N, Hd, 0, 1.f, samples, T);
  layerNorm(h, h, RN, Hd, P(ro + "to_out_0/weight"), nullptr);
  float* out = scratch<float>("be2.dn.out", RN * 3);
  linear(h, P(ro + "to_out_1/weight"), out, RN, Hd, 3);
  const float s2 = sigma * sigma, sd2 = SIGMA_DATA * SIGMA_DATA;
  edmOutK<<<blocks(RN * 3), 256, 0, STREAM>>>(x, out, den, RN * 3, sd2 / (s2 + sd2), sigma * SIGMA_DATA / sqrtf(sd2 + s2));
  if (getenv("BE2_DEBUG_HALF")) {
    auto bad = [&](const char* n, const float* p, size_t k) {
      auto h = download(p, k); size_t b = 0; for (float v : h) b += !std::isfinite(v);
      printf("  %s: %zu of %zu nonfinite\n", n, b, k);
    };
    bad("q", q, RN * Ca); bad("a", a, RT * Dt); bad("h", h, RN * Hd); bad("out", out, RN * 3); bad("x", x, RN * 3);
  }
}

}  // namespace be2
