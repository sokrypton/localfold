// BioEmu2's dual-stream denoiser (cuda/bioemu2/src/denoise.cuh and denoise_call.cuh are the reading):
//
//   static, once a fold: z = transitions(W LN([z_trunk | relpos]));  q0 = c (the per-atom features);
//     c' = c + s_to_c(LN s_trunk) broadcast;  p = pair geometry + z_to_p(LN z_trunk) windowed, through the tail with c'
//   a call at sigma: r = c_in x;  s = single_embed(LN [s_trunk | s_inputs]) + fourier(t) (+ the lag's time embedding),
//     two transitions -> cond;  q = q0 + W r (+ the conditioning frame's term);  a = W LN(s_trunk) + W r[centre]
//     16 layers: q through a windowed conditioned layer (cond broadcast to atoms), a through a full one (cond),
//     then the coupling: q += gamma_q T([a | q] adaLN), a += gamma_s T([q_centre | a] adaLN) - both from the
//     layer's outputs, added together
//     readout: out = W LN(W q + W a broadcast);  D = c_skip x + c_out out   (EDM, sigma_data 16)
// Everything that does not depend on sigma or the coordinates - every layer's pair biases, the single conditioner's
// embedding, a's trunk half - is computed once a fold (Denoiser). Everything that depends on sigma alone - the single
// conditioning and every layer's conditioning projections, on the token side AND the atom side - once a call, shared
// by every sample: an atom row reads its token's row (CondLayer's map), exact, because a broadcast row is the row.
#include "bioemu2.h"
#include <cmath>

static const std::string SM = "structure_module/";

// ---------------------------------------------------------------- the static context
// DampedHarmonicTimeEmbedding of a scalar lag: [envelope (n), sin (n * Q), cos (n * Q)], timescale-major
static std::vector<float> dampedHarmonic(float dt, const float* taus, size_t n, const float* q, size_t Q) {
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
// propagation: the lag's time embedding added to the single conditioner's static half, and the conditioning frame's
// gate, sum(exp(-dt / taus) softplus(log_w)); coordsCond [N][3] the (centred) frame on the device
void setDynamics(Denoiser& d, float dt, const float* coordsCond) {
  const std::string sc = SM + "single_conditioner/", af = SM + "atom_feature_embedder/";
  const int T = d.T, Dt = dimW(sc + "single_embed/weight", 0);
  const std::string taus = "w/" + sc + "dt_embed/taus", qf = "w/" + sc + "dt_embed/quality_factors";
  std::vector<float> emb = dampedHarmonic(dt, M.hostF(taus), M.len(taus), M.hostF(qf), M.len(qf));
  float* de = uploadNew(emb.data(), emb.size());
  float* dv = scratch<float>("be2.sc.dt", Dt);
  linT(de, sc + "dt_embed_to_single/weight", dv, 1, (int)emb.size(), Dt);
  be2Run1d("be2_broadcast_add", (size_t)T * Dt, Be2BroadcastAddArgs{d.sc0, dv, (u64)T, (uint)Dt, 0});
  release(de);
  const std::string gt = "w/" + af + "dt_gate_embed/taus", lw = "w/" + af + "dt_gate_log_w";
  const float* gtau = M.hostF(gt); const float* glw = M.hostF(lw);
  double g = 0;
  for (size_t k = 0; k < M.len(gt); ++k) g += expf(-dt / gtau[k]) * log1pf(expf(glw[k]));
  d.dynGate = (float)g; d.dyn = true; d.coordsCond = coordsCond;
}

Denoiser staticContext(const float* sTrunk, const float* zTrunk, const float* sInputs, int Si, const Tap& tap) {
  Denoiser d; d.T = D_.T; d.N = D_.N; d.Si = Si;
  const int T = d.T, N = d.N;
  const size_t P = (size_t)T * T;
  // the pairwise conditioner
  const std::string pc = SM + "pairwise_conditioner/";
  const int C = dimW(pc + "dim_pairwise_init_proj_1/weight", 0), Cs = dimW(SM + "s_trunk_to_a_1/weight", 1);
  float* cat = scratch<float>("be2.sc.cat", P * 2 * C);
  copy2d(cat, (size_t)2 * C * 4, zTrunk, (size_t)C * 4, (size_t)C * 4, P);
  relPos("rel_pos/", cat + C, T, C, false, 2 * C);
  layerNorm(cat, cat, P, 2 * C, Wf(pc + "dim_pairwise_init_proj_0/weight"), Wf(pc + "dim_pairwise_init_proj_0/bias"));
  d.z = allocT<float>(P * C);
  linT(cat, pc + "dim_pairwise_init_proj_1/weight", d.z, P, 2 * C, C);
  for (int i = 0; i < 2; ++i) transition(d.z, d.z, P, C, pc + "transitions_" + num(i) + "/");
  // the atoms
  const std::string af = SM + "atom_feature_embedder/";
  const int Ca = dimW(af + "proj_atom_continuous/weight", 0);
  float* c = scratch<float>("be2.sc.c", (size_t)N * Ca);
  atomFeatures(af, c, N);
  d.q0 = allocT<float>((size_t)N * Ca);
  copy(d.q0, c, (size_t)N * Ca * 4);
  d.p = allocT<float>((size_t)N * WK * 16);
  pairGeometry(af, d.p, N);
  float* sn = scratch<float>("be2.sc.sn", (size_t)T * Cs);
  layerNorm(sTrunk, sn, T, Cs, Wf(af + "s_to_c_trans_0/weight"), Wf(af + "s_to_c_trans_0/bias"));
  float* s2c = scratch<float>("be2.sc.s2c", (size_t)T * Ca);
  linT(sn, af + "s_to_c_trans_1/weight", s2c, T, Cs, Ca);
  be2Run1d("be2_tokens_to_atoms", (size_t)N * Ca, Be2TokensToAtomsArgs{s2c, Iin("token_idx_at_atom"), Iin("atom_pad_mask"), c, (u64)N,
                                                                    (uint)T, (uint)Ca, (uint)Ca, 1, 1.f, 0});
  float* zn = scratch<float>("be2.sc.zn", P * C);
  layerNorm(zTrunk, zn, P, C, Wf(af + "z_to_p_trans_0/weight"), Wf(af + "z_to_p_trans_0/bias"));
  float* z2p = scratch<float>("be2.sc.z2p", P * 16);
  linT(zn, af + "z_to_p_trans_1/weight", z2p, P, C, 16);
  run1d("be2_pair_token_add", (size_t)N * WK, Be2PairTokenAddArgs{z2p, Iin("token_idx_at_atom"), Iin("atom_pad_mask"), d.p,
                                                                   (uint)N, (uint)T, 16, 0});
  pairTail(af, d.p, c, N);
  if (tap) { tap("ctx.z", d.z, P * C); tap("ctx.q0", d.q0, (size_t)N * Ca); tap("ctx.p", d.p, (size_t)N * WK * 16); }
  // the per-layer biases
  auto ab = [](int l) { return SM + "layers/" + num(l) + "/atom_block/transformer_layer/attention/attention_layer/pair_bias_attn/"; };
  auto tb = [](int l) { return SM + "layers/" + num(l) + "/token_block/attention/pair_bias_attn/"; };
  while (hasW(ab(d.layers) + "proj_z/weight")) ++d.layers;
  const int Ha = dimW(ab(0) + "proj_z/weight", 0), Ht = dimW(tb(0) + "proj_z/weight", 0);
  const size_t wb = (size_t)N * WK * Ha, tbn = (size_t)Ht * P;
  d.wbias = allocT<float>(wb * d.layers);
  for (int l = 0; l < d.layers; ++l) windowBias(d.p, Wf(ab(l) + "norm_z/weight"), Wf(ab(l) + "proj_z/weight"), d.wbias + wb * l, N, Ha);
  if (HALF) d.tbiasH = allocT<half>(tbn * d.layers); else d.tbias = allocT<float>(tbn * d.layers);
  float* proj = scratch<float>("be2.tb.proj", P * Ht);
  for (int l = 0; l < d.layers; ++l) {
    layerNorm(d.z, zn, P, C, Wf(tb(l) + "norm_z/weight"), nullptr);
    linT(zn, tb(l) + "proj_z/weight", proj, P, C, Ht);
    run1d("be2_token_bias", P * Ht, Be2TokenBiasArgs{proj, HALF ? nullptr : d.tbias + tbn * l, (uint)T, (uint)Ht,
                                                      HALF ? d.tbiasH + tbn * l : nullptr, (float)M_LOG2E, 0});
  }
  // the single conditioner's sigma-free half
  const std::string sc = SM + "single_conditioner/";
  const int Dt = dimW(sc + "single_embed/weight", 0), Wi = Cs + Si;
  float* si = scratch<float>("be2.sc.si", (size_t)T * Wi);
  copy2d(si, (size_t)Wi * 4, sTrunk, (size_t)Cs * 4, (size_t)Cs * 4, T);
  copy2d(si + Cs, (size_t)Wi * 4, sInputs, (size_t)Si * 4, (size_t)Si * 4, T);
  layerNorm(si, si, T, Wi, Wf(sc + "norm_single/weight"), Wf(sc + "norm_single/bias"));
  d.sc0 = allocT<float>((size_t)T * Dt);
  linT(si, sc + "single_embed/weight", d.sc0, T, Wi, Dt);
  addBias(d.sc0, Wf(sc + "single_embed/bias"), T, Dt);
  if (M.meta("meta/is_dynamics_conditioned", 0) != 0) setDynamics(d, (float)M.meta("meta/dt"), Fin("coords_cond"));
  float* sn2 = scratch<float>("be2.sc.sn2", (size_t)T * Cs);
  layerNorm(sTrunk, sn2, T, Cs, Wf(SM + "s_trunk_to_a_0/weight"), Wf(SM + "s_trunk_to_a_0/bias"));
  d.a0 = allocT<float>((size_t)T * Dt);
  linT(sn2, SM + "s_trunk_to_a_1/weight", d.a0, T, Cs, Dt);
  return d;
}
void freeDenoiser(Denoiser& d) {
  for (float* p : {d.z, d.q0, d.p, d.wbias, d.tbias, d.sc0, d.a0}) if (p) release(p);
  if (d.tbiasH) release(d.tbiasH);
  d = Denoiser{};
}

// ---------------------------------------------------------------- the coupling split
// The coupling's atom-side mixer (s_to_q, fc1 | fc2 over [a_token (768) | q_atom (128)] adaLN'd) is a third of the
// sampler unsplit. Exactly, W y = inv (sa P[token] + sq Z[atom] + da Qa + dq Qq) + B: P = W_a Xa once a sample-TOKEN row,
// Z = W_q Xq the atom's own 128 columns, Qa = W_a (1 + alpha_a), Qq = W_q (1 + alpha_q), B = W beta once a conditioning
// row, the row's LayerNorm statistics combined from its halves' (parallel variance) - webgpu/bioemu2's couplingSplit,
// 9.25 -> 2.7 G multiply-adds a layer at 68 tokens x 5 samples; the sampler 14.5 -> 12.9 s at 68 tokens x 5 samples x
// 200 steps on an M5. 🔴 Z NEVER LEAVES THE FLOAT TILE: the combine and the SwiGLU are its GEMM's epilogue
// (gemmCouplingCombine). Built first with Z stored, it lost precision in half and its speed in float. Every product here is on the blocks-of-8 weight, so u's b half is fc2 / 64 as
// gemmSwiglu's, and fc3 scales it back. Off the matrix units the unsplit mixer is the only path
// (BE2_COUPLING_SPLIT: 0 never, 1 always, unset - the rule below)
static const int COUPLING_SPLIT = getenv("BE2_COUPLING_SPLIT") ? atoi(getenv("BE2_COUPLING_SPLIT")) : -1;
// 🔴 THE SPLIT'S Qa | Qq | B IS A COST A LAYER THAT THE SAMPLES DO NOT SHARE OUT: (T + 1) x 2I x (Dt + Ca + Dc), twice
// (hi | lo), against the unsplit mixer's samples N x 2I x Dc. Taken where its multiply-adds, times SPLIT_MARGIN (its
// kernels' lower efficiency, measured), are fewer than the unsplit's
static bool splitPays(int samples, int T, int N, int Dt, int Ca) {
  if (COUPLING_SPLIT >= 0) return COUPLING_SPLIT != 0;
  const double Dc = Dt + Ca, RN = (double)samples * N, RT = (double)samples * T, R = T + 1.0;
  const double unsplit = RN * Dc, split = 2 * R * (Dt + Ca + Dc) + RT * Dt + RN * 2 * Ca;
  // (measured, 68 tokens x 100 steps on an M5, unsplit / split: 1 sample 2.36 / 2.64 s, 2 3.56 / 3.52 - with the
  // combine in registers; before it 2 was 3.67 / 3.70 - the ratio below 1.16 and 1.60 there: the line between 1 and 2)
  static const double margin = getenv("BE2_SPLIT_MARGIN") ? atof(getenv("BE2_SPLIT_MARGIN")) : 1.5;
  return split * margin < unsplit;
}
static void couplingSplit(const float* a, float* q, const float* cqc, const int* atomTok, int samples, int T, int N,
                   int Dt, int Ca, const std::string& p) {
  const int Dc = Dt + Ca, I = dimW(p + "fc1/weight", 0), ldc = 2 * Dc + Ca;
  const size_t RN = (size_t)samples * N, RT = (size_t)samples * T, R = (size_t)T + 1;
  const half* w12 = swigluPairs(p, Dc, I);             // [Dc][2I]: rows 0..Dt W_a, Dt.. W_q
  half* gR = scratch<half>("be2.cs.gR", 2 * R * Dc); half* beta = scratch<half>("be2.cs.beta", 2 * R * Dc);
  float* QB = scratch<float>("be2.cs.QB", R * 4 * I);
  // 🔴 Xq IS TWO HALVES, [hi | lo], against W_q stacked twice (K 128 -> 256, free here): in one half the split read
  // 9.7e-3 at sigma 1 and 2.7e-4 at 0.05 against webgpu/bioemu2's float32 arm, where unsplit reads 3.3e-3 and 4.0e-5;
  // hi | lo gives 3.1e-3 and 3.5e-5. Xa in one half costs nothing measurable (hi | lo there: 9.6e-3, 2.6e-4)
  // (in blocks of 2 - be2_cs_wq2 - so the combine runs in the vector epilogue's registers; BE2_CS_STAGED=1 the control,
  // the blocks-of-8 rows and a staged float tile)
  static const bool staged = getenv("BE2_CS_STAGED") && atoi(getenv("BE2_CS_STAGED")) != 0;
  const size_t wq = (size_t)Ca * 2 * I;
  const half* wQ = staged ? M.derived<half>("csQ2:" + p, 2 * wq, [&](half* out) {
    copy2d(out, wq * 2, w12 + (size_t)Dt * 2 * I, wq * 2, wq * 2, 1);
    copy2d(out + wq, wq * 2, w12 + (size_t)Dt * 2 * I, wq * 2, wq * 2, 1);
  }) : M.derived<half>("csQ2b:" + p, 2 * wq, [&](half* out) {
    be2Run1d("be2_cs_wq2", 2 * wq, Be2CsWqArgs{M.h("w/" + p + "fc1/weight"), M.h("w/" + p + "fc2/weight"), out, (uint)Dc, (uint)I,
                                             (uint)Dt, (uint)Ca, 1.f / swigluScale(), 0});
  });
  float* tstats = scratch<float>("be2.cs.tstats", RT * 3); half* xa = scratch<half>("be2.cs.xa", RT * Dt);
  float* P = scratch<float>("be2.cs.P", RT * 2 * I);
  float* stats = scratch<float>("be2.cs.stats", RN * 8); half* xq = scratch<half>("be2.cs.xq", RN * Ca * 2);
  half* g = scratch<half>("be2.cs.g", RN * I);
  be2Run1d("be2_cs_gamma", R * Dc, Be2CsGammaArgs{cqc, gR, beta, (u64)(R * Dc), (uint)Dt, (uint)Ca, (uint)ldc, (uint)R});
  // R | B [R][2][2I] in float - R = Qq - Qa / 6 (da = -dq / 6 exactly, as 768 da + 128 dq = 0: one vector where Qa and
  // Qq were two, and the epilogue reads a quarter less) - each two GEMMs on the matrix units, the rows' hi then their
  // remainder added (float's precision: as one float GEMM off the matrix units, Qa | Qq | B were 12% of the sampler)
  auto qb = [&](const half* X, int col, const char* label) {
    for (int part = 0; part < 2; ++part) {
      Gemm G{}; G.X = X + (size_t)part * R * Dc; G.tx = F16; G.W = w12; G.tw = F16; G.Y = QB + col; G.ty = F32; G.ldy = 4 * I;
      G.rows = R; G.in = Dc; G.out = 2 * I; G.beta = part ? 1.f : 0.f; G.label = label; gemm(G);
    }
  };
  qb(gR, 0, "coupling R");
  qb(beta, 2 * I, "coupling B");
  auto rowsGrid = [](size_t rows) {
    const size_t groups = (rows + 7) / 8;
    return Grid{(uint32_t)std::min<size_t>(groups, 65535), (uint32_t)((groups + 65534) / 65535), 1};
  };
  run("be2_cs_tok", rowsGrid(RT), 256, Be2CsTokArgs{a, cqc, tstats, xa, (u64)RT, (uint)T, (uint)Dt, (uint)ldc, 0});
  { Gemm G{}; G.X = xa; G.tx = F16; G.W = w12; G.tw = F16; G.Y = P; G.ty = F32; G.rows = RT; G.in = Dt; G.out = 2 * I;
    G.label = "coupling P"; gemm(G); }
  run("be2_cs_atom", rowsGrid(RN), 256, Be2CsAtomArgs{q, tstats, cqc, Iin("token_idx_at_atom"), Iin("atom_pad_mask"), atomTok,
                                                    stats, xq, (u64)RN, (uint)N, (uint)T, (uint)Ca, (uint)Dt, (uint)ldc, 0});
  if (!gemmCouplingCombine(xq, wQ, g, RN, 2 * Ca, I, stats, P, QB, !staged)) die("BioEmu2: the coupling split off the matrix units");
  transitionFc3(g, q, RN, I, p, false, cqc + 2 * Dc, ldc, atomTok);      // (q += gamma_q s_to_q, in fc3's epilogue)
}

// ---------------------------------------------------------------- one call
// the row maps of `samples` stacked samples: each atom to its token's conditioning row (padding atoms to the zero row
// T), each token to its own
struct RowMaps { int samples = 0; int* atomTok = nullptr; int* tok = nullptr; };
static RowMaps ROW_MAPS;
// 🔴 FORGOTTEN EVERY FOLD: the maps are this input's atoms' tokens, and a resident server's next job is another input -
// keyed on the sample count alone, a second fold of the same count read the first one's (cuda/bioemu2's resident server
// met this class of bug: its triangle mask remembered by address, docs/BIOEMU2.md)
void forgetRowMaps() {
  if (ROW_MAPS.atomTok) { release(ROW_MAPS.atomTok); release(ROW_MAPS.tok); }
  ROW_MAPS = {};
}
static const RowMaps& rowMaps(int samples, int T, int N) {
  if (ROW_MAPS.samples == samples && ROW_MAPS.atomTok) return ROW_MAPS;
  if (ROW_MAPS.atomTok) { release(ROW_MAPS.atomTok); release(ROW_MAPS.tok); }
  const int* tok = M.hostI("token_idx_at_atom");
  std::vector<int> a((size_t)samples * N), t((size_t)samples * T);
  for (size_t r = 0; r < a.size(); ++r) a[r] = tok[r % N] >= 0 ? tok[r % N] : T;
  for (size_t r = 0; r < t.size(); ++r) t[r] = (int)(r % T);
  ROW_MAPS = {samples, uploadNew(a.data(), a.size()), uploadNew(t.data(), t.size())};
  return ROW_MAPS;
}

void denoise(const Denoiser& d, const float* x, float sigma, float* den, const float* cond, int samples) {
  const int T = d.T, N = d.N;
  const size_t RN = (size_t)samples * N, RT = (size_t)samples * T;
  const RowMaps& maps = rowMaps(samples, T, N);
  const std::string sc = SM + "single_conditioner/", af = SM + "atom_feature_embedder/";
  const int Ca = dimW(af + "r_to_q_trans/weight", 0), Dt = dimW(sc + "single_embed/weight", 0);
  const float cIn = 1.f / sqrtf(sigma * sigma + SIGMA_DATA * SIGMA_DATA), t = 0.25f * logf(sigma / SIGMA_DATA);
  float* r = scratch<float>("be2.dn.r", RN * 3);
  be2Run1d("be2_scale_masked", RN * 3, Be2ScaleMaskedArgs{x, Iin("atom_mask"), r, (u64)N, RN, cIn, 0});
  // the single conditioning (shared)
  const int F = dimW(sc + "fourier_embed/proj/bias", 0);
  float* four = scratch<float>("be2.dn.four", F);
  run1d("be2_cos_embed", F, Be2CosEmbedArgs{Wf(sc + "fourier_embed/proj/weight"), Wf(sc + "fourier_embed/proj/bias"), four, t, (uint)F});
  layerNorm(four, four, 1, F, Wf(sc + "norm_fourier/weight"), Wf(sc + "norm_fourier/bias"));
  float* f768 = scratch<float>("be2.dn.f768", Dt);
  linT(four, sc + "fourier_to_single/weight", f768, 1, F, Dt);
  // cond, and a ZERO row after its T: what a padding atom of the reference conditions on (c_to_q is 0 there)
  float* cnd = scratch<float>("be2.dn.cond", (size_t)(T + 1) * Dt);
  copy(cnd, d.sc0, (size_t)T * Dt * 4);
  fill(cnd + (size_t)T * Dt, 0, (size_t)Dt * 4);
  be2Run1d("be2_broadcast_add", (size_t)T * Dt, Be2BroadcastAddArgs{cnd, f768, (u64)T, (uint)Dt, 0});
  for (int i = 0; i < 2; ++i) transition(cnd, cnd, T, Dt, sc + "transitions_" + num(i) + "/");
  // the atom stream's initial state
  float* q = scratch<float>("be2.dn.q", RN * Ca);
  be2Run1d("be2_repeat", RN * Ca, Be2RepeatArgs{d.q0, q, (u64)N * Ca, RN * Ca});
  linT(r, af + "r_to_q_trans/weight", q, RN, 3, Ca, 1.f);
  if (d.dyn) {
    float* cq = scratch<float>("be2.dn.cq", RN * Ca);
    if (cond) linT(cond, af + "cond_to_q_trans/weight", cq, RN, 3, Ca);
    else {
      float* one = scratch<float>("be2.dn.cq1", (size_t)N * Ca);
      linT(d.coordsCond, af + "cond_to_q_trans/weight", one, N, 3, Ca);
      be2Run1d("be2_repeat", RN * Ca, Be2RepeatArgs{one, cq, (u64)N * Ca, RN * Ca});
    }
    run1d("be2_scaled_add", RN * Ca, Be2ScaledAddArgs{q, cq, RN * Ca, d.dynGate, 0});
  }
  float* a = scratch<float>("be2.dn.a", RT * Dt);
  be2Run1d("be2_repeat", RT * Dt, Be2RepeatArgs{d.a0, a, (u64)T * Dt, RT * Dt});
  float* rc = scratch<float>("be2.dn.rc", RT * 3);
  be2Run1d("be2_centre_atoms", RT * 3, Be2CentreAtomsArgs{r, Iin("token_center_atom_index"), rc, (uint)T, (uint)N, 3, 3, (uint)samples, 0});
  linT(rc, SM + "r_token_to_a/weight", a, RT, 3, Dt, 1.f);
  // the layers
  const std::string ab = "attention/attention_layer/pair_bias_attn/", tb = "attention/pair_bias_attn/";
  const int Ha = 4, Ht = 16;
  const size_t P = (size_t)T * T, wb = (size_t)N * WK * Ha, tbn = (size_t)Ht * P;
  const int Dc = Dt + Ca;                                  // the coupling's width, 896
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
    const std::string L = SM + "layers/" + num(l) + "/";
    const CondLayer atomL{L + "atom_block/transformer_layer/", ab, true, Ha};
    // (the atom stream's conditioning is its token's row of cond: computed on the T + 1 token rows)
    float* condA = layerCond(atomL, cnd, T + 1, Dt, Ca, "be2.dn.condA");
    condLayer(atomL, q, (int)RN, Ca, condA, maps.atomTok, d.wbias + wb * l, N);
    const CondLayer tokL{L + "token_block/", tb, false, Ht};
    float* condT = layerCond(tokL, cnd, T, Dt, Dt, "be2.dn.condT");
    condLayer(tokL, a, (int)RT, Dt, condT, maps.tok, d.tbias ? d.tbias + tbn * l : nullptr, T,
              d.tbiasH ? d.tbiasH + tbn * l : nullptr);
    // the coupling's conditioning, both on the token rows (the atom side's input is cond broadcast to atoms)
    const std::string cp = L + "coupling/";
    layerNorm(cnd, cn, T + 1, Dt, Wf(cp + "q_coupling_cond_0/weight"), WfOpt(cp + "q_coupling_cond_0/bias"));
    silu(cn, cn, (size_t)(T + 1) * Dt);
    linT(cn, cp + "q_coupling_cond_2/weight", cqc, T + 1, Dt, 2 * Dc + Ca, 0.f, 0, 0, false, WfOpt(cp + "q_coupling_cond_2/bias"));
    layerNorm(cnd, cn, T, Dt, Wf(cp + "s_coupling_cond_0/weight"), WfOpt(cp + "s_coupling_cond_0/bias"));
    silu(cn, cn, (size_t)T * Dt);
    linT(cn, cp + "s_coupling_cond_2/weight", csc, T, Dt, 2 * Dc + Dt, 0.f, 0, 0, false, WfOpt(cp + "s_coupling_cond_2/bias"));
    // s_to_q = [a broadcast | q]; q_to_s = [q centre | a]
    // (the fast arm on the matrix units: s_to_q SPLIT - couplingSplit - and its row never built)
    const bool csplit = HALF && gemmTensorOps() && splitPays(samples, T, N, Dt, Ca);
    if (!csplit) {
      be2Run1d("be2_tokens_to_atoms", RN * Dt, Be2TokensToAtomsArgs{a, Iin("token_idx_at_atom"), Iin("atom_pad_mask"), sq, (u64)N,
                                                                 (uint)T, (uint)Dt, (uint)Dc, (uint)samples, 0.f, 0});
      copy2d(sq + Dt, (size_t)Dc * 4, q, (size_t)Ca * 4, (size_t)Ca * 4, RN);
    }
    be2Run1d("be2_centre_atoms", RT * Ca, Be2CentreAtomsArgs{q, Iin("token_center_atom_index"), qs, (uint)T, (uint)N, (uint)Ca,
                                                          (uint)Dc, (uint)samples, 0});
    copy2d(qs + Ca, (size_t)Dc * 4, a, (size_t)Dt * 4, (size_t)Dt * 4, RT);
    if (HALF) {      // (the adaLN straight into the fused transition's half input)
      half* sqh = scratch<half>("be2.dn.sqh", RN * Dc); half* qsh = scratch<half>("be2.dn.qsh", RT * Dc);
      // (q += gamma_q s_to_q and a += gamma_s q_to_s in the fc3s' epilogues: both mixers' inputs - sq and qs, or the
      // split's reads of a and q - are taken before either update)
      if (csplit) couplingSplit(a, q, cqc, maps.atomTok, samples, T, N, Dt, Ca, cp + "s_to_q_mixer/");
      else {
        adaLn(sq, nullptr, RN, Dc, cqc, cqc + Dc, 2 * Dc + Ca, maps.atomTok, sqh);
        transitionHalf(sqh, q, RN, Dc, cp + "s_to_q_mixer/", false, cqc + 2 * Dc, 2 * Dc + Ca, maps.atomTok);
      }
      adaLn(qs, nullptr, RT, Dc, csc, csc + Dc, 2 * Dc + Dt, maps.tok, qsh);
      transitionHalf(qsh, a, RT, Dc, cp + "q_to_s_mixer/", false, csc + 2 * Dc, 2 * Dc + Dt, maps.tok);
    } else {
      adaLn(sq, sqn, RN, Dc, cqc, cqc + Dc, 2 * Dc + Ca, maps.atomTok);
      transition(sqn, dq, RN, Dc, cp + "s_to_q_mixer/", false, false);
      adaLn(qs, qsn, RT, Dc, csc, csc + Dc, 2 * Dc + Dt, maps.tok);
      transition(qsn, da, RT, Dc, cp + "q_to_s_mixer/", false, false);
      be2Run1d("be2_gated_add", RN * Ca, Be2GatedAddArgs{q, dq, cqc + 2 * Dc, maps.atomTok, RN, (uint)Ca, (uint)(2 * Dc + Ca)});
      be2Run1d("be2_gated_add", RT * Dt, Be2GatedAddArgs{a, da, csc + 2 * Dc, maps.tok, RT, (uint)Dt, (uint)(2 * Dc + Dt)});
    }
  }
  // the readout
  const std::string ro = SM + "readout/";
  const int Hd = dimW(ro + "q_to_hidden/weight", 0);
  float* h = scratch<float>("be2.dn.h", RN * Hd);
  float* ah = scratch<float>("be2.dn.ah", RT * Hd);
  // (wide: the two streams RAW, never normalised. 🔴 A PADDING atom's row grows past float16 - cuda/bioemu2 survives it
  // because its half staging saturates; metal/core's rounds to inf, and an inf value meeting an attention weight of
  // exactly 0 is NaN. Kept in float, every row stays finite)
  linT(q, ro + "q_to_hidden/weight", h, RN, Ca, Hd, 0.f, 0, 0, true);
  linT(a, ro + "s_to_hidden/weight", ah, RT, Dt, Hd, 0.f, 0, 0, true);
  be2Run1d("be2_tokens_to_atoms", RN * Hd, Be2TokensToAtomsArgs{ah, Iin("token_idx_at_atom"), Iin("atom_pad_mask"), h, (u64)N,
                                                             (uint)T, (uint)Hd, (uint)Hd, (uint)samples, 1.f, 0});
  layerNorm(h, h, RN, Hd, Wf(ro + "to_out_0/weight"), nullptr);
  float* out = scratch<float>("be2.dn.out", RN * 3);
  linT(h, ro + "to_out_1/weight", out, RN, Hd, 3);
  const float s2 = sigma * sigma, sd2 = SIGMA_DATA * SIGMA_DATA;
  run1d("be2_edm_out", RN * 3, Be2EdmOutArgs{x, out, den, RN * 3, sd2 / (s2 + sd2), sigma * SIGMA_DATA / sqrtf(sd2 + s2)});
}
