// BioEmu2's dual-stream denoiser (network/bioemu2.py's static_context and denoise):
//
//   static, once a fold: z = transitions(W LN([z_trunk | relpos]));  q0 = c (the per-atom features);
//     c' = c + s_to_c(LN s_trunk) broadcast;  p = pair geometry + z_to_p(LN z_trunk) windowed, through the tail with c'
//   a call at sigma: r = c_in x;  s = single_embed(LN [s_trunk | s_inputs]) + fourier(t) (+ the lag's time embedding),
//     two transitions -> cond;  q = q0 + W r (+ the conditioning frame's term);  a = W LN(s_trunk) + W r[centre]
//     16 layers: q through a windowed conditioned layer (cond broadcast to atoms), a through a full one (cond),
//     then the coupling: q += gamma_q T([a | q] adaLN), a += gamma_s T([q_centre | a] adaLN) - both from the
//     layer's outputs, added together
//     readout: out = W LN(W q + W a broadcast);  D = c_skip x + c_out out   (EDM, sigma_data 16)
// Everything that does not depend on sigma or the coordinates - the per-layer pair biases of both attentions, the
// single conditioner's embedding, a's trunk half - is computed once a fold (Denoiser).
#pragma once
#include "trunk.cuh"

namespace be2 {

inline const std::string SM = "structure_module/";

struct Denoiser {
  int T = 0, N = 0, Si = 0, layers = 0;
  float* z = nullptr;          // [P, 256] the pairwise conditioner's pair
  float* q0 = nullptr;         // [N, 128]
  float* p = nullptr;          // [N, 128, 16] the windowed pair
  float* wbias = nullptr;      // [layers][NW, 4, 32, 128]
  float* tbias = nullptr;      // [layers][16, T, T], or null: made per call (a card short of room)
  float* sc0 = nullptr;        // [T, 768] single_embed(LN [s_trunk | s_inputs]) + the lag's time embedding
  float* a0 = nullptr;         // [T, 768] s_trunk_to_a(LN s_trunk)
  float dynGate = 0.f;         // the conditioning frame's gate (zero unconditioned)
  bool dyn = false;
  const float* coordsCond = nullptr;
};

// DampedHarmonicTimeEmbedding of a scalar lag: [envelope (n), sin (n * Q), cos (n * Q)], timescale-major
inline std::vector<float> dampedHarmonic(float dt, const std::vector<float>& taus, const std::vector<float>& q) {
  const size_t n = taus.size(), Q = q.size();
  std::vector<float> out(n * (1 + 2 * Q));
  for (size_t i = 0; i < n; ++i) {
    float x = dt / taus[i], env = expf(-x);
    out[i] = env;
    for (size_t k = 0; k < Q; ++k) {
      float ph = 2.f * (float)M_PI * q[k] * x;
      out[n + i * Q + k] = env * sinf(ph);
      out[n + n * Q + i * Q + k] = env * cosf(ph);
    }
  }
  return out;
}
inline std::vector<float> hostP(const std::string& n) { return download(P(n), lenP(n)); }

// propagation: the lag's time embedding added to the single conditioner's static half, and the conditioning
// frame's gate, sum(exp(-dt / taus) softplus(log_w)); coordsCond [N, 3] the (centred) frame on the device
inline void setDynamics(Denoiser& d, float dt, const float* coordsCond) {
  const std::string sc = SM + "single_conditioner/", af = SM + "atom_feature_embedder/";
  const int T = d.T, Dt = dimP(sc + "single_embed/weight", 0);
  std::vector<float> emb = dampedHarmonic(dt, hostP(sc + "dt_embed/taus"), hostP(sc + "dt_embed/quality_factors"));
  float* de = upload(emb.data(), emb.size());
  float* dv = scratch<float>("be2.sc.dt", Dt);
  linear(de, P(sc + "dt_embed_to_single/weight"), dv, 1, (int)emb.size(), Dt);
  const float one = 1.f;
  for (int i = 0; i < T; ++i) CB(cublasSaxpy(H, Dt, &one, dv, 1, d.sc0 + (size_t)i * Dt, 1));
  CK(cudaStreamSynchronize(STREAM)); CK(cudaFree(de));
  std::vector<float> taus = hostP(af + "dt_gate_embed/taus"), lw = hostP(af + "dt_gate_log_w");
  double g = 0;
  for (size_t k = 0; k < taus.size(); ++k) g += expf(-dt / taus[k]) * log1pf(expf(lw[k]));
  d.dynGate = (float)g; d.dyn = true; d.coordsCond = coordsCond;
}

inline Denoiser staticContext(const float* sTrunk, const float* zTrunk, const float* sInputs, int Si,
                              const std::function<void(const char*, const float*, size_t)>& tap) {
  Denoiser d; d.T = D_.T; d.N = D_.N; d.Si = Si;
  const int T = d.T, N = d.N, C = 256, Cs = 384; size_t Pn = (size_t)T * T;
  // the pairwise conditioner
  const std::string pc = SM + "pairwise_conditioner/";
  float* cat = scratch<float>("be2.sc.cat", Pn * 2 * C);
  CK(cudaMemcpy2DAsync(cat, 2 * C * 4, zTrunk, C * 4, C * 4, Pn, cudaMemcpyDeviceToDevice, STREAM));
  relPos("rel_pos/", cat + C, T, C, false, 2 * C);
  layerNorm(cat, cat, Pn, 2 * C, P(pc + "dim_pairwise_init_proj_0/weight"), P(pc + "dim_pairwise_init_proj_0/bias"));
  d.z = dalloc(Pn * C);
  linear(cat, P(pc + "dim_pairwise_init_proj_1/weight"), d.z, Pn, 2 * C, C);
  for (int i = 0; i < 2; ++i) transition(d.z, d.z, Pn, C, pc + "transitions_" + std::to_string(i) + "/");
  // the atoms
  const std::string af = SM + "atom_feature_embedder/";
  const int Ca = dimP(af + "proj_atom_continuous/weight", 0);
  float* c = scratch<float>("be2.sc.c", (size_t)N * Ca);
  atomFeatures(af, c, N);
  d.q0 = dalloc((size_t)N * Ca);
  CK(cudaMemcpyAsync(d.q0, c, (size_t)N * Ca * 4, cudaMemcpyDeviceToDevice, STREAM));
  d.p = dalloc((size_t)N * WK * 16);
  pairGeometry(af, d.p, N);
  float* sn = scratch<float>("be2.sc.sn", (size_t)T * Cs);
  layerNorm(sTrunk, sn, T, Cs, P(af + "s_to_c_trans_0/weight"), P(af + "s_to_c_trans_0/bias"));
  float* s2c = scratch<float>("be2.sc.s2c", (size_t)T * Ca);
  linear(sn, P(af + "s_to_c_trans_1/weight"), s2c, T, Cs, Ca);
  tokensToAtoms(s2c, c, N, Ca, 0, 1.f);
  float* zn = scratch<float>("be2.sc.zn", Pn * C);
  layerNorm(zTrunk, zn, Pn, C, P(af + "z_to_p_trans_0/weight"), P(af + "z_to_p_trans_0/bias"));
  float* z2p = scratch<float>("be2.sc.z2p", Pn * 16);
  linear(zn, P(af + "z_to_p_trans_1/weight"), z2p, Pn, C, 16);
  pairTokenAddK<<<blocks((size_t)N * WK), 256, 0, STREAM>>>(z2p, Idev("token_idx_at_atom"), Idev("atom_pad_mask"), d.p, N, T, 16);
  pairTail(af, d.p, c, N);
  if (tap) { tap("ctx.z", d.z, Pn * C); tap("ctx.q0", d.q0, (size_t)N * Ca); tap("ctx.p", d.p, (size_t)N * WK * 16); }
  // the per-layer biases
  const std::string ab = SM + "layers/atom_block/transformer_layer/attention/attention_layer/pair_bias_attn/";
  const std::string tb = SM + "layers/token_block/attention/pair_bias_attn/";
  d.layers = dimP(ab + "proj_z/weight", 0);
  const int Ha = dimP(ab + "proj_z/weight", 1), Ht = dimP(tb + "proj_z/weight", 1);
  const size_t wb = (size_t)N * WK * Ha, tbn = (size_t)Ht * Pn;
  d.wbias = dalloc(wb * d.layers);
  for (int l = 0; l < d.layers; ++l) windowBias(d.p, PL(ab + "norm_z/weight", l), PL(ab + "proj_z/weight", l), d.wbias + wb * l, N, Ha);
  if (roomFor(tbn * d.layers * 4 + Pn * C * 4 * 2)) {
    d.tbias = dalloc(tbn * d.layers);
    for (int l = 0; l < d.layers; ++l) tokenBias(d.z, PL(tb + "norm_z/weight", l), PL(tb + "proj_z/weight", l), d.tbias + tbn * l, T, C, Ht);
  }
  // the single conditioner's sigma-free half
  const std::string sc = SM + "single_conditioner/";
  const int Dt = dimP(sc + "single_embed/weight", 0), Wi = Cs + Si;
  float* si = scratch<float>("be2.sc.si", (size_t)T * Wi);
  CK(cudaMemcpy2DAsync(si, Wi * 4, sTrunk, Cs * 4, Cs * 4, T, cudaMemcpyDeviceToDevice, STREAM));
  CK(cudaMemcpy2DAsync(si + Cs, Wi * 4, sInputs, Si * 4, Si * 4, T, cudaMemcpyDeviceToDevice, STREAM));
  layerNorm(si, si, T, Wi, P(sc + "norm_single/weight"), P(sc + "norm_single/bias"));
  d.sc0 = dalloc((size_t)T * Dt);
  linear(si, P(sc + "single_embed/weight"), d.sc0, T, Wi, Dt);
  addBiasRows(d.sc0, P(sc + "single_embed/bias"), T, Dt);
  if (M.flag("meta/is_dynamics_conditioned")) setDynamics(d, (float)M.meta("meta/dt"), Fdev("coords_cond"));
  float* sn2 = scratch<float>("be2.sc.sn2", (size_t)T * Cs);
  layerNorm(sTrunk, sn2, T, Cs, P(SM + "s_trunk_to_a_0/weight"), P(SM + "s_trunk_to_a_0/bias"));
  d.a0 = dalloc((size_t)T * Dt);
  linear(sn2, P(SM + "s_trunk_to_a_1/weight"), d.a0, T, Cs, Dt);
  return d;
}
inline void freeDenoiser(Denoiser& d) {
  CK(cudaStreamSynchronize(STREAM));
  for (float* p : {d.z, d.q0, d.p, d.wbias, d.tbias, d.sc0, d.a0}) if (p) CK(cudaFree(p));
  d = Denoiser{};
}

// r = x c_in on real atoms
__global__ void scaleMaskedK(const float* x, const int* mask, float* r, size_t N, float s) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < N * 3) r[t] = mask[t / 3] ? x[t] * s : 0.f;
}
// out = [tokens_to_atoms(a) | q] (s_to_q's input) or [centre(q) | a] (q_to_s's)
__global__ void edmOutK(const float* x, const float* out, float* d, size_t n, float skip, float cout) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) d[t] = skip * x[t] + cout * out[t];
}
__global__ void cosEmbedK(const float* w, const float* b, float* out, float t, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n) out[k] = cosf(2.f * (float)M_PI * (t * w[k] + b[k]));
}
__global__ void broadcastAddK(float* y, const float* v, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) y[t] += v[t % C];
}
__global__ void scaledAddK(float* y, const float* x, float s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] += s * x[t];
}


}  // namespace be2
