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
  float* buf = nullptr;        // single contiguous allocation backing z, q0, p, wbias, tbias, sc0, a0
  float* z = nullptr;          // [P, 256] the pairwise conditioner's pair
  float* q0 = nullptr;         // [N, 128]
  float* p = nullptr;          // [N, 128, 16] the windowed pair
  float* wbias = nullptr;      // [layers][NW, 4, 32, 128]
  float* tbias = nullptr;      // [layers][16, T, T], or null: made per call (a card short of room)
  float* sc0 = nullptr;        // [T, 768] single_embed(LN [s_trunk | s_inputs]) + the lag's time embedding
  float* a0 = nullptr;         // [T, 768] s_trunk_to_a(LN s_trunk)
  float dynGate = 0.f;         // the conditioning frame's gate (zero unconditioned)
  bool dyn = false;
  bool ownZ = false;
  const float* coordsCond = nullptr;
};

__global__ void broadcastAddK(float* __restrict__ y, const float* __restrict__ v, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)v) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t < n4) {
      int col4 = (int)(t % (size_t)c4);
      float4 yv = reinterpret_cast<const float4*>(y)[t];
      float4 vv = __ldg(reinterpret_cast<const float4*>(v) + col4);
      reinterpret_cast<float4*>(y)[t] = make_float4(yv.x + vv.x, yv.y + vv.y, yv.z + vv.z, yv.w + vv.w);
    }
    return;
  }
  if (t < rows * C) y[t] += v[t % C];
}

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
  size_t nt = ((Dt & 3) == 0 && (((uintptr_t)d.sc0 | (uintptr_t)dv) & 15) == 0) ? (size_t)T * (Dt >> 2) : (size_t)T * Dt;
  broadcastAddK<<<blocks(nt), 256, 0, STREAM>>>(d.sc0, dv, T, Dt);
  CK(cudaStreamSynchronize(STREAM)); CK(cudaFree(de));
  std::vector<float> taus = hostP(af + "dt_gate_embed/taus"), lw = hostP(af + "dt_gate_log_w");
  double g = 0;
  for (size_t k = 0; k < taus.size(); ++k) g += expf(-dt / taus[k]) * log1pf(expf(lw[k]));
  d.dynGate = (float)g; d.dyn = true; d.coordsCond = coordsCond;
}

// Fused [zTrunk | relPos] concatenation + LayerNorm(512) into f32 or f16, one warp per pair row.
// Eliminates materializing and reading/writing the [Pn, 512] intermediate cat buffer (1.5 GB traffic at T=494).
template <class TH>
__global__ void __launch_bounds__(256, 2)
pairCondInitLnK(const float* __restrict__ zTrunk, RelTables w,
                const int* __restrict__ asym, const int* __restrict__ ent,
                const int* __restrict__ res, const int* __restrict__ sym,
                const float* __restrict__ scale, const float* __restrict__ offset,
                TH* __restrict__ y, size_t rows, int T, size_t r0 = 0) {
  size_t r = (size_t)blockIdx.x * 8 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (r >= rows) return;
  size_t row = r0 + r;
  constexpr int C = 256, W = 512;
  int i = (int)(row / (size_t)T), j = (int)(row % (size_t)T);
  const int rmax = 32, smax = 2;
  auto clip = [](int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; };
  bool sameChain = asym[i] == asym[j], sameRes = res[i] == res[j], sameEnt = ent[i] == ent[j];
  int dRes = sameChain ? clip(res[i] - res[j] + rmax, 0, 2 * rmax) : 2 * rmax + 1;
  int dTok = sameChain && sameRes ? clip(i - j + rmax, 0, 2 * rmax) : 2 * rmax + 1;
  int dChain = sameEnt ? clip(sym[i] - sym[j] + smax, 0, 2 * smax) : 2 * smax + 1;
  const float* zr = zTrunk + row * C;
  const float* wPos = w.pos + (size_t)dRes * C;
  const float* wTok = w.tok + (size_t)dTok * C;
  const float* wChn = w.chain + (size_t)dChain * C;
  float xv[16];
  float s = 0.f;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    float v = __ldg(zr + k * 32 + lane);
    xv[k] = v;
    s += v;
  }
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    int cr = k * 32 + lane;
    float v = __ldg(wPos + cr) + __ldg(wTok + cr) + (sameEnt ? __ldg(w.ent + cr) : 0.f) + __ldg(wChn + cr);
    xv[8 + k] = v;
    s += v;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s * (1.f / 512.f), var = 0.f;
  #pragma unroll
  for (int k = 0; k < 16; ++k) {
    float d = xv[k] - mean;
    var += d * d;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) var += __shfl_xor_sync(~0u, var, o);
  float inv = rsqrtf(var * (1.f / 512.f) + 1e-5f);
  TH* yr = y + r * W;
  #pragma unroll
  for (int k = 0; k < 16; ++k) {
    int c = k * 32 + lane;
    float n = (xv[k] - mean) * inv * __ldg(scale + c) + __ldg(offset + c);
    if constexpr (std::is_same_v<TH, float>) yr[c] = n;
    else yr[c] = toH<TH>(n);
  }
}

// Fused [sTrunk (Cs=384) | sInputs (Si)] concatenation + LayerNorm(Cs + Si), one warp per token row.
__global__ void __launch_bounds__(256, 4)
singleCondCatLnK(const float* __restrict__ sTrunk, const float* __restrict__ sInputs,
                 const float* __restrict__ scale, const float* __restrict__ offset,
                 float* __restrict__ si, int T, int Cs, int Si) {
  int row = (int)blockIdx.x * 8 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (row >= T) return;
  const int Wi = Cs + Si;
  const float* st = sTrunk + (size_t)row * Cs;
  const float* sin = sInputs + (size_t)row * Si;
  float* yr = si + (size_t)row * Wi;
  float sum = 0.f;
  for (int c = lane; c < Wi; c += 32) {
    float v = (c < Cs) ? __ldg(st + c) : __ldg(sin + (c - Cs));
    sum += v;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  float mean = sum / (float)Wi, var = 0.f;
  for (int c = lane; c < Wi; c += 32) {
    float v = (c < Cs) ? __ldg(st + c) : __ldg(sin + (c - Cs));
    float d = v - mean;
    var += d * d;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) var += __shfl_xor_sync(~0u, var, o);
  float inv = rsqrtf(var / (float)Wi + 1e-5f);
  for (int c = lane; c < Wi; c += 32) {
    float v = (c < Cs) ? __ldg(st + c) : __ldg(sin + (c - Cs));
    yr[c] = (v - mean) * inv * __ldg(scale + c) + __ldg(offset + c);
  }
}

inline Denoiser staticContext(const float* sTrunk, const float* zTrunk, const float* sInputs, int Si,
                              const std::function<void(const char*, const float*, size_t)>& tap, bool ownZ = false) {
  Denoiser d; d.T = D_.T; d.N = D_.N; d.Si = Si; d.ownZ = ownZ;
  const int T = d.T, N = d.N, C = 256, Cs = 384; size_t Pn = (size_t)T * T;
  const std::string pc = SM + "pairwise_conditioner/";
  const std::string af = SM + "atom_feature_embedder/";
  const std::string ab = SM + "layers/atom_block/transformer_layer/attention/attention_layer/pair_bias_attn/";
  const std::string tb = SM + "layers/token_block/attention/pair_bias_attn/";
  const std::string sc = SM + "single_conditioner/";
  const int Ca = dimP(af + "proj_atom_continuous/weight", 0);
  d.layers = dimP(ab + "proj_z/weight", 0);
  const int Ha = dimP(ab + "proj_z/weight", 1), Ht = dimP(tb + "proj_z/weight", 1);
  const int Dt = dimP(sc + "single_embed/weight", 0), Wi = Cs + Si;
  const size_t wb = (size_t)N * WK * Ha, tbn = (size_t)Ht * Pn;
  auto align64 = [](size_t n) { return (n + 63) & ~size_t(63); };
  const bool keepTbias = Pn <= ((size_t)1 << 20) && roomFor(tbn * d.layers * 4 + Pn * C * 4 * 2);
  const size_t offZ     = 0;
  const size_t offQ0    = ownZ ? 0 : (offZ + align64(Pn * C));
  const size_t offP     = offQ0    + align64((size_t)N * Ca);
  const size_t offWbias = offP     + align64((size_t)N * WK * 16);
  const size_t offSc0   = offWbias + align64(wb * d.layers);
  const size_t offA0    = offSc0   + align64((size_t)T * Dt);
  const size_t offTbias = offA0    + align64((size_t)T * Dt);
  d.buf   = dalloc(offTbias + (keepTbias ? align64(tbn * d.layers) : 0));
  d.z     = ownZ ? const_cast<float*>(zTrunk) : (d.buf + offZ);
  d.q0    = d.buf + offQ0;
  d.p     = d.buf + offP;
  d.wbias = d.buf + offWbias;
  d.sc0   = d.buf + offSc0;
  d.a0    = d.buf + offA0;
  d.tbias = keepTbias ? (d.buf + offTbias) : nullptr;
  size_t pChunk = std::min<size_t>(Pn, (size_t)1 << 20);
  // the atoms (run before the pairwise conditioner when d.z reuses zTrunk in place)
  float* c = scratch<float>("be2.sc.c", (size_t)N * Ca);
  atomFeatures(af, c, N);
  CK(cudaMemcpyAsync(d.q0, c, (size_t)N * Ca * 4, cudaMemcpyDeviceToDevice, STREAM));
  pairGeometry(af, d.p, N);
  float* s2c = scratch<float>("be2.sc.s2c", (size_t)T * Ca);
  float* z2p = scratch<float>("be2.sc.z2p", Pn * 16);
  if (GEMM_HALF) {
    half* snh = scratch<half>("be2.sc.snh", (size_t)T * Cs);
    lnTo<half>(sTrunk, snh, T, Cs, P(af + "s_to_c_trans_0/weight"), P(af + "s_to_c_trans_0/bias"));
    linearPre<half>(snh, halfMirror<half>(P(af + "s_to_c_trans_1/weight"), (size_t)Ca * Cs), s2c, T, Cs, Ca);
    tokensToAtoms(s2c, c, N, Ca, 0, 1.f);
    half* znh = scratch<half>("be2.sc.znh", pChunk * C);
    const half* wZ2P = halfMirror<half>(P(af + "z_to_p_trans_1/weight"), (size_t)16 * C);
    for (size_t r0 = 0; r0 < Pn; r0 += pChunk) {
      size_t r = std::min(pChunk, Pn - r0);
      lnTo<half>(zTrunk + r0 * C, znh, r, C, P(af + "z_to_p_trans_0/weight"), P(af + "z_to_p_trans_0/bias"));
      linearPre<half>(znh, wZ2P, z2p + r0 * 16, r, C, 16);
    }
  } else {
    float* sn = scratch<float>("be2.sc.sn", (size_t)T * Cs);
    layerNorm(sTrunk, sn, T, Cs, P(af + "s_to_c_trans_0/weight"), P(af + "s_to_c_trans_0/bias"));
    linear(sn, P(af + "s_to_c_trans_1/weight"), s2c, T, Cs, Ca);
    tokensToAtoms(s2c, c, N, Ca, 0, 1.f);
    float* zn = scratch<float>("be2.sc.zn", pChunk * C);
    for (size_t r0 = 0; r0 < Pn; r0 += pChunk) {
      size_t r = std::min(pChunk, Pn - r0);
      layerNorm(zTrunk + r0 * C, zn, r, C, P(af + "z_to_p_trans_0/weight"), P(af + "z_to_p_trans_0/bias"));
      linear(zn, P(af + "z_to_p_trans_1/weight"), z2p + r0 * 16, r, C, 16);
    }
  }
  pairTokenAddK<<<blocks((size_t)N * WK), 256, 0, STREAM>>>(z2p, Idev("token_idx_at_atom"), Idev("atom_pad_mask"), d.p, N, T, 16);
  pairTail(af, d.p, c, N);
  // the pairwise conditioner
  if (GEMM_HALF) {
    half* cath = scratch<half>("be2.sc.cath", pChunk * 2 * C);
    const half* wProj1 = halfMirror<half>(P(pc + "dim_pairwise_init_proj_1/weight"), (size_t)C * 2 * C);
    for (size_t r0 = 0; r0 < Pn; r0 += pChunk) {
      size_t r = std::min(pChunk, Pn - r0);
      pairCondInitLnK<half><<<(unsigned)((r + 7) / 8), 256, 0, STREAM>>>(
          zTrunk, relTables("rel_pos/"), Idev("asym_id"), Idev("entity_id"), Idev("residue_index"), Idev("sym_id"),
          P(pc + "dim_pairwise_init_proj_0/weight"), P(pc + "dim_pairwise_init_proj_0/bias"), cath, r, T, r0);
      linearPre<half>(cath, wProj1, d.z + r0 * C, r, 2 * C, C);
    }
  } else {
    float* cat = scratch<float>("be2.sc.cat", pChunk * 2 * C);
    for (size_t r0 = 0; r0 < Pn; r0 += pChunk) {
      size_t r = std::min(pChunk, Pn - r0);
      pairCondInitLnK<float><<<(unsigned)((r + 7) / 8), 256, 0, STREAM>>>(
          zTrunk, relTables("rel_pos/"), Idev("asym_id"), Idev("entity_id"), Idev("residue_index"), Idev("sym_id"),
          P(pc + "dim_pairwise_init_proj_0/weight"), P(pc + "dim_pairwise_init_proj_0/bias"), cat, r, T, r0);
      linear(cat, P(pc + "dim_pairwise_init_proj_1/weight"), d.z + r0 * C, r, 2 * C, C);
    }
  }
  for (int i = 0; i < 2; ++i) transition(d.z, d.z, Pn, C, pc + "transitions_" + std::to_string(i) + "/");
  if (tap) { tap("ctx.z", d.z, Pn * C); tap("ctx.q0", d.q0, (size_t)N * Ca); tap("ctx.p", d.p, (size_t)N * WK * 16); }
  // the per-layer biases
  WinBiasPtrs wbPtrs{};
  for (int l = 0; l < d.layers; ++l) {
    wbPtrs.normW[l] = PL(ab + "norm_z/weight", l);
    wbPtrs.projZ[l] = PL(ab + "proj_z/weight", l);
  }
  windowBiasBatch(d.p, wbPtrs, d.wbias, N, Ha, d.layers);
  if (d.tbias) {
    tokenBiasAll(d.z, P(tb + "norm_z/weight"), P(tb + "proj_z/weight"), d.tbias, T, C, Ht, d.layers);
  }
  // the single conditioner's sigma-free half
  float* si = scratch<float>("be2.sc.si", (size_t)T * Wi);
  singleCondCatLnK<<<(unsigned)((T + 7) / 8), 256, 0, STREAM>>>(
      sTrunk, sInputs, P(sc + "norm_single/weight"), P(sc + "norm_single/bias"), si, T, Cs, Si);
  linear(si, P(sc + "single_embed/weight"), d.sc0, T, Wi, Dt);
  addBiasRows(d.sc0, P(sc + "single_embed/bias"), T, Dt);
  if (M.flag("meta/is_dynamics_conditioned")) setDynamics(d, (float)M.meta("meta/dt"), Fdev("coords_cond"));
  if (GEMM_HALF) {
    half* sn2h = scratch<half>("be2.sc.sn2h", (size_t)T * Cs);
    lnTo<half>(sTrunk, sn2h, T, Cs, P(SM + "s_trunk_to_a_0/weight"), P(SM + "s_trunk_to_a_0/bias"));
    linearPre<half>(sn2h, halfMirror<half>(P(SM + "s_trunk_to_a_1/weight"), (size_t)Dt * Cs), d.a0, T, Cs, Dt);
  } else {
    float* sn2 = scratch<float>("be2.sc.sn2", (size_t)T * Cs);
    layerNorm(sTrunk, sn2, T, Cs, P(SM + "s_trunk_to_a_0/weight"), P(SM + "s_trunk_to_a_0/bias"));
    linear(sn2, P(SM + "s_trunk_to_a_1/weight"), d.a0, T, Cs, Dt);
  }
  return d;
}
inline void freeDenoiser(Denoiser& d) {
  CK(cudaStreamSynchronize(STREAM));
  if (d.buf) { CK(cudaFree(d.buf)); if (d.ownZ && d.z) CK(cudaFree(d.z)); }
  else for (float* p : {d.z, d.q0, d.p, d.wbias, d.tbias, d.sc0, d.a0}) if (p) CK(cudaFree(p));
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
  if (k < n) out[k] = fastCosF(2.f * (float)M_PI * (t * w[k] + b[k]));
}
__global__ void scaledAddK(float* __restrict__ y, const float* __restrict__ x, float s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)y | (uintptr_t)x) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 yv = reinterpret_cast<const float4*>(y)[t];
      float4 xv = __ldg(reinterpret_cast<const float4*>(x) + t);
      reinterpret_cast<float4*>(y)[t] = make_float4(yv.x + s * xv.x, yv.y + s * xv.y, yv.z + s * xv.z, yv.w + s * xv.w);
    }
    return;
  }
  if (t < n) y[t] += s * x[t];
}


}  // namespace be2
