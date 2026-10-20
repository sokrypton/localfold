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
__global__ void repeatRowsK(const float* src, float* dst, size_t elems, size_t total) {   // dst = src, src, ...
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < total) dst[t] = src[t % elems];
}

// x [samples, N, 3] (noisy, angstrom) at sigma -> den [samples, N, 3]. `cond` (optional, [samples, N, 3]) is each
// sample's conditioning frame (propagation's companion, augmented with the state); null takes the featurised
// coords_cond for every sample
inline void denoise(const Denoiser& d, const float* x, float sigma, float* den, const float* cond = nullptr, int samples = 1) {
  const int T = d.T, N = d.N, Ca = 128, Dt = 768;
  const size_t RN = (size_t)samples * N, RT = (size_t)samples * T;
  const RowMaps& maps = rowMaps(samples, T, N);
  const std::string sc = SM + "single_conditioner/", af = SM + "atom_feature_embedder/";
  const float cIn = 1.f / sqrtf(sigma * sigma + SIGMA_DATA * SIGMA_DATA), t = 0.25f * logf(sigma / SIGMA_DATA);
  float* r = scratch<float>("be2.dn.r", RN * 3);
  scaleMaskedBatchK<<<blocks(RN * 3), 256, 0, STREAM>>>(x, Idev("atom_mask"), r, N, RN, cIn);
  // the single conditioning (shared)
  const int F = dimP(sc + "fourier_embed/proj/bias", 0);
  float* four = scratch<float>("be2.dn.four", F);
  cosEmbedK<<<blocks(F), 256, 0, STREAM>>>(P(sc + "fourier_embed/proj/weight"), P(sc + "fourier_embed/proj/bias"), four, t, F);
  layerNorm(four, four, 1, F, P(sc + "norm_fourier/weight"), P(sc + "norm_fourier/bias"));
  float* f768 = scratch<float>("be2.dn.f768", Dt);
  linear(four, P(sc + "fourier_to_single/weight"), f768, 1, F, Dt);
  // cond, and a ZERO row after its T: what a padding atom of the reference conditions on (c_to_q is 0 there)
  float* cnd = scratch<float>("be2.dn.cond", (size_t)(T + 1) * Dt);
  CK(cudaMemcpyAsync(cnd, d.sc0, (size_t)T * Dt * 4, cudaMemcpyDeviceToDevice, STREAM));
  CK(cudaMemsetAsync(cnd + (size_t)T * Dt, 0, Dt * 4, STREAM));

  broadcastAddK<<<blocks((size_t)T * Dt), 256, 0, STREAM>>>(cnd, f768, T, Dt);
  for (int i = 0; i < 2; ++i) transition(cnd, cnd, T, Dt, sc + "transitions_" + std::to_string(i) + "/");
  // the atom stream's initial state
  float* q = scratch<float>("be2.dn.q", RN * Ca);
  repeatRowsK<<<blocks(RN * Ca), 256, 0, STREAM>>>(d.q0, q, (size_t)N * Ca, RN * Ca);
  linear(r, P(af + "r_to_q_trans/weight"), q, RN, 3, Ca, 1.f);
  if (d.dyn) {
    float* cq = scratch<float>("be2.dn.cq", RN * Ca);
    if (cond) linear(cond, P(af + "cond_to_q_trans/weight"), cq, RN, 3, Ca);
    else {
      linear(d.coordsCond, P(af + "cond_to_q_trans/weight"), cq, N, 3, Ca);
      repeatRowsK<<<blocks(RN * Ca), 256, 0, STREAM>>>(cq, cq, (size_t)N * Ca, RN * Ca);
    }
    scaledAddK<<<blocks(RN * Ca), 256, 0, STREAM>>>(q, cq, d.dynGate, RN * Ca);
  }
  float* a = scratch<float>("be2.dn.a", RT * Dt);
  repeatRowsK<<<blocks(RT * Dt), 256, 0, STREAM>>>(d.a0, a, (size_t)T * Dt, RT * Dt);
  float* rc = scratch<float>("be2.dn.rc", RT * 3);
  centreAtoms(r, rc, T, 3, 0, samples, N);
  linear(rc, P(SM + "r_token_to_a/weight"), a, RT, 3, Dt, 1.f);
  // the layers
  const std::string L = SM + "layers/";
  const std::string ab = "attention/attention_layer/pair_bias_attn/", tb = "attention/pair_bias_attn/";
  const size_t Pn = (size_t)T * T, wb = (size_t)N * WK * 4, tbn = 16 * Pn;
  const int Dc = Dt + Ca;                                  // the coupling's width, 896
  float* tbias = d.tbias ? nullptr : scratch<float>("be2.dn.tbias", tbn);
  float* cqc = scratch<float>("be2.dn.cqc", (size_t)(T + 1) * (2 * Dc + Ca));
  float* csc = scratch<float>("be2.dn.csc", (size_t)T * (2 * Dc + Dt));
  float* sq = scratch<float>("be2.dn.sq", RN * Dc);
  float* qs = scratch<float>("be2.dn.qs", RT * Dc);
  float* dq = scratch<float>("be2.dn.dq", RN * Ca);
  float* da = scratch<float>("be2.dn.da", RT * Dt);
  float* cn = scratch<float>("be2.dn.cn", (size_t)(T + 1) * Dt);
  float* sqn = scratch<float>("be2.dn.sqn", RN * Dc);
  float* qsn = scratch<float>("be2.dn.qsn", RT * Dc);
  for (int l = 0; l < d.layers; ++l) {
    const CondLayer atomL{L + "atom_block/transformer_layer/", ab, l, true, 4};
    // (the atom stream's conditioning is its token's row of cond: computed on the T token rows)
    float* condA = layerCond(atomL, cnd, T + 1, Dt, Ca, "be2.dn.condA");
    condLayer(atomL, q, (int)RN, Ca, condA, maps.atomTok, d.wbias + wb * l, N);
    const float* bias = d.tbias ? d.tbias + tbn * l : tbias;
    if (!d.tbias) tokenBias(d.z, PL(L + "token_block/" + tb + "norm_z/weight", l), PL(L + "token_block/" + tb + "proj_z/weight", l), tbias, T, 256, 16);
    const CondLayer tokL{L + "token_block/", tb, l, false, 16};
    float* condT = layerCond(tokL, cnd, T, Dt, Dt, "be2.dn.condT");
    condLayer(tokL, a, (int)RT, Dt, condT, maps.tok, bias, T);
    // the coupling's conditioning, both on the token rows (the atom side's input is cond broadcast to atoms)
    const std::string cp = L + "coupling/";
    layerNorm(cnd, cn, T + 1, Dt, PL(cp + "q_coupling_cond_0/weight", l), PLopt(cp + "q_coupling_cond_0/bias", l));
    siluK<<<blocks((size_t)(T + 1) * Dt), 256, 0, STREAM>>>(cn, cn, (size_t)(T + 1) * Dt);
    linear(cn, PL(cp + "q_coupling_cond_2/weight", l), cqc, T + 1, Dt, 2 * Dc + Ca);
    if (const float* b = PLopt(cp + "q_coupling_cond_2/bias", l)) addBiasRows(cqc, b, T + 1, 2 * Dc + Ca);
    layerNorm(cnd, cn, T, Dt, PL(cp + "s_coupling_cond_0/weight", l), PLopt(cp + "s_coupling_cond_0/bias", l));
    siluK<<<blocks((size_t)T * Dt), 256, 0, STREAM>>>(cn, cn, (size_t)T * Dt);
    linear(cn, PL(cp + "s_coupling_cond_2/weight", l), csc, T, Dt, 2 * Dc + Dt);
    if (const float* b = PLopt(cp + "s_coupling_cond_2/bias", l)) addBiasRows(csc, b, T, 2 * Dc + Dt);
    // s_to_q = [a broadcast | q]; q_to_s = [q centre | a]
    tokensToAtoms(a, sq, N, Dt, Dc, 0.f, samples, T);
    CK(cudaMemcpy2DAsync(sq + Dt, Dc * 4, q, Ca * 4, Ca * 4, RN, cudaMemcpyDeviceToDevice, STREAM));
    centreAtoms(q, qs, T, Ca, Dc, samples, N);
    CK(cudaMemcpy2DAsync(qs + Ca, Dc * 4, a, Dt * 4, Dt * 4, RT, cudaMemcpyDeviceToDevice, STREAM));
    if (GEMM_HALF) {
      half* sqh = scratch<half>("be2.dn.sqh", RN * Dc); half* qsh = scratch<half>("be2.dn.qsh", RT * Dc);
      adaLn<half>(sq, sqh, RN, Dc, cqc, cqc + Dc, 2 * Dc + Ca, 0, 1.f, maps.atomTok);
      transitionHalf(nullptr, sqh, dq, RN, Dc, cp + "s_to_q_mixer/", l, false, false, 0);
      adaLn<half>(qs, qsh, RT, Dc, csc, csc + Dc, 2 * Dc + Dt, 0, 1.f, maps.tok);
      transitionHalf(nullptr, qsh, da, RT, Dc, cp + "q_to_s_mixer/", l, false, false, 0);
    } else {
      adaLn(sq, sqn, RN, Dc, cqc, cqc + Dc, 2 * Dc + Ca, 0, 1.f, maps.atomTok);
      transition(sqn, dq, RN, Dc, cp + "s_to_q_mixer/", l, false, false);
      adaLn(qs, qsn, RT, Dc, csc, csc + Dc, 2 * Dc + Dt, 0, 1.f, maps.tok);
      transition(qsn, da, RT, Dc, cp + "q_to_s_mixer/", l, false, false);
    }
    gatedAdd(q, dq, cqc + 2 * Dc, RN, Ca, 2 * Dc + Ca, maps.atomTok);
    gatedAdd(a, da, csc + 2 * Dc, RT, Dt, 2 * Dc + Dt, maps.tok);
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
