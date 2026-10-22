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
  const float* wbNorm = M.derived<float>("sm.wb.norm", (size_t)d.layers * 16, [&](float* out) {
    for (int l = 0; l < d.layers; ++l) copy(out + (size_t)l * 16, Wf(ab(l) + "norm_z/weight"), 64);
  });
  const float* wbProj = M.derived<float>("sm.wb.proj", (size_t)d.layers * Ha * 16, [&](float* out) {
    for (int l = 0; l < d.layers; ++l) copy(out + (size_t)l * Ha * 16, Wf(ab(l) + "proj_z/weight"), (size_t)Ha * 64);
  });
  windowBias(d.p, wbNorm, wbProj, d.wbias, N, Ha, d.layers);
  if (HALF) d.tbiasH = allocT<half>(tbn * d.layers); else d.tbias = allocT<float>(tbn * d.layers);
  if (HALF && C == 128) {
    const float* tbNorm = M.derived<float>("sm.tb.norm", (size_t)d.layers * C, [&](float* out) {
      for (int l = 0; l < d.layers; ++l) copy(out + (size_t)l * C, Wf(tb(l) + "norm_z/weight"), (size_t)C * 4);
    });
    const half* tbProj = M.derived<half>("sm.tb.proj", (size_t)d.layers * Ht * C, [&](half* out) {
      for (int l = 0; l < d.layers; ++l) copy(out + (size_t)l * Ht * C, M.h("w/" + tb(l) + "proj_z/weight"), (size_t)Ht * C * 2);
    });
    half* znh = scratch<half>("be2.tb.znh", (size_t)d.layers * P * C);
    float* proj = scratch<float>("be2.tb.proj", (size_t)d.layers * P * Ht);
    const size_t groups = (P + 7) / 8;
    run("be2_cond_in", Grid{(uint32_t)std::min<size_t>(groups, 65535), (uint32_t)((groups + 65534) / 65535), 1}, 256,
        Be2CondInArgs{d.z, tbNorm, nullptr, znh, (u64)P, (uint)P, (uint)C, 0, (uint)d.layers});
    Gemm G{}; G.X = znh; G.tx = F16; G.sx = (int64_t)P * C; G.W = tbProj; G.tw = F16; G.transW = true; G.sw = (int64_t)Ht * C;
    G.Y = proj; G.ty = F32; G.sy = (int64_t)P * Ht; G.rows = P; G.in = C; G.out = Ht; G.batch = d.layers; G.label = "tb proj_z";
    gemm(G);
    be2Run1d("be2_token_bias", (size_t)d.layers * P * Ht,
             Be2TokenBiasArgs{proj, nullptr, (uint)T, (uint)Ht, d.tbiasH, (float)M_LOG2E, (uint)d.layers});
  } else {
    float* proj = scratch<float>("be2.tb.proj", P * Ht);
    for (int l = 0; l < d.layers; ++l) {
      layerNorm(d.z, zn, P, C, Wf(tb(l) + "norm_z/weight"), nullptr);
      linT(zn, tb(l) + "proj_z/weight", proj, P, C, Ht);
      run1d("be2_token_bias", P * Ht, Be2TokenBiasArgs{proj, HALF ? nullptr : d.tbias + tbn * l, (uint)T, (uint)Ht,
                                                        HALF ? d.tbiasH + tbn * l : nullptr, (float)M_LOG2E, 1});
    }
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
                          int Dt, int Ca, const std::string& p, const float* preQB = nullptr, bool preTok = false) {
  const int Dc = Dt + Ca, I = dimW(p + "fc1/weight", 0), ldc = 2 * Dc + Ca;
  const size_t RN = (size_t)samples * N, RT = (size_t)samples * T, R = (size_t)T + 1;
  const half* w12 = swigluPairs(p, Dc, I);             // [Dc][2I]: rows 0..Dt W_a, Dt.. W_q
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
    run("be2_cs_wq2", Grid{(uint32_t)((I + 31) / 32), (uint32_t)((Ca + 31) / 32), 1}, 256,
        Be2CsWqArgs{M.h("w/" + p + "fc1/weight"), M.h("w/" + p + "fc2/weight"), out, (uint)Dc, (uint)I,
                    (uint)Dt, (uint)Ca, 1.f / swigluScale(), 0});
  });
  float* tstats = scratch<float>("be2.cs.tstats", RT * 3); half* xa = scratch<half>("be2.cs.xa", RT * Dt);
  float* P = scratch<float>("be2.cs.P", RT * 2 * I);
  float* stats = scratch<float>("be2.cs.stats", RN * 8); half* xq = scratch<half>("be2.cs.xq", RN * Ca * 2);
  half* g = scratch<half>("be2.cs.g", RN * I);
  const float* QB = preQB;
  if (!QB) {
    half* gR = scratch<half>("be2.cs.gR", 4 * R * Dc);
    float* qbb = scratch<float>("be2.cs.QB", R * 4 * I);
    be2Run1d("be2_cs_gamma", R * Dc, Be2CsGammaArgs{cqc, gR, nullptr, (u64)(R * Dc), (uint)Dt, (uint)Ca, (uint)ldc, (uint)R});
    // R | B [R][2][2I] in float - R = Qq - Qa / 6 (da = -dq / 6 exactly, as 768 da + 128 dq = 0: one vector where Qa and
    // Qq were two, and the epilogue reads a quarter less) - even rows gR, odd rows beta, stacked into [2R][Dc] so R and B
    // share a single 2R-row GEMM per hi/lo part (64x128 vector epilogue instead of two 80x32 staged GEMMs)
    for (int part = 0; part < 2; ++part) {
      Gemm G{}; G.X = gR + (size_t)part * 2 * R * Dc; G.tx = F16; G.W = w12; G.tw = F16; G.Y = qbb; G.ty = F32; G.ldy = 2 * I;
      G.rows = 2 * R; G.in = Dc; G.out = 2 * I; G.beta = part ? 1.f : 0.f; G.label = "coupling RB"; gemm(G);
    }
    QB = qbb;
  }
  auto rowsGrid = [](size_t rows) {
    const size_t groups = (rows + 7) / 8;
    return Grid{(uint32_t)std::min<size_t>(groups, 65535), (uint32_t)((groups + 65534) / 65535), 1};
  };
  if (!preTok)
    run("be2_cs_tok", rowsGrid(RT), 256, Be2CsTokArgs{a, cqc, tstats, xa, (u64)RT, (uint)T, (uint)Dt, (uint)ldc, 0});
  { Gemm G{}; G.X = xa; G.tx = F16; G.W = w12; G.tw = F16; G.Y = P; G.ty = F32; G.rows = RT; G.in = Dt; G.out = 2 * I;
    G.label = "coupling P"; gemm(G); }
  run("be2_cs_atom", rowsGrid(RN), 256, Be2CsAtomArgs{q, tstats, cqc, Iin("token_idx_at_atom"), Iin("atom_pad_mask"), atomTok,
                                                    stats, xq, (u64)RN, (uint)N, (uint)T, (uint)Ca, (uint)Dt, (uint)ldc, 0});
  if (!gemmCouplingCombine(xq, wQ, g, RN, 2 * Ca, I, stats, P, QB, !staged)) die("BioEmu2: the coupling split off the matrix units");
  transitionFc3(g, q, RN, I, p, false, cqc + 2 * Dc, ldc, atomTok);      // (q += gamma_q s_to_q, in fc3's epilogue)
}

// ---------------------------------------------------------------- the conditioning, layers batched
// Every layer's four adaLN MLPs - silu(LN(c) w + b) . cond_2 + bias - read the SAME rows (the call's cond), and only
// LN's affine and the two linears are the layer's: so a group of layers' are one batched GEMM a kind (metal/core's
// pointer-array batch, each layer's own weight - nothing stacked but the small vectors), where each was a 68-row GEMM
// of its own (7% of the sampler in the profile). The fast arm's; BE2_BATCH_COND=0 the control
static const bool BATCH_COND = !getenv("BE2_BATCH_COND") || atoi(getenv("BE2_BATCH_COND")) != 0;
struct CondKind { const char* key; std::string (*prefix)(int); int rowsExtra; const char *l0, *l2; };
static std::string atomPrefix(int l) { return SM + "layers/" + num(l) + "/atom_block/transformer_layer/"; }
static std::string tokPrefix(int l) { return SM + "layers/" + num(l) + "/token_block/"; }
static std::string couplingPrefix(int l) { return SM + "layers/" + num(l) + "/coupling/"; }
static const CondKind COND_KINDS[4] = {{"atom", atomPrefix, 1, "cond_0", "cond_2"}, {"tok", tokPrefix, 0, "cond_0", "cond_2"},
                                       {"q", couplingPrefix, 1, "q_coupling_cond_0", "q_coupling_cond_2"},
                                       {"s", couplingPrefix, 0, "s_coupling_cond_0", "s_coupling_cond_2"}};
// [layers][n] of one tensor per layer, as a float buffer (made once)
static const float* stackedVectors(const std::string& key, const CondKind& k, const char* tensor, int layers, size_t n) {
  return M.derived<float>("cgstack:" + key + tensor, (size_t)layers * n, [&](float* out) {
    for (int l = 0; l < layers; ++l) copy(out + (size_t)l * n, Wf(k.prefix(l) + tensor), n * 4);
  });
}
// a device array of pointers, rewritten only when one changes (after a sync: an earlier call may still read it)
static u64* pointerArray(const std::string& key, const std::vector<u64>& v) {
  struct Arr { u64* dev = nullptr; size_t cap = 0; std::vector<u64> host; };
  static std::map<std::string, Arr> arrays;
  Arr& r = arrays[key];
  if (r.host != v) {
    mt::sync();
    if (r.cap < v.size()) { if (r.dev) release(r.dev); r.dev = allocT<u64>(v.size()); r.cap = v.size(); }
    upload(r.dev, v.data(), v.size() * sizeof(u64));
    r.host = v;
  }
  return r.dev;
}
// layers l0 .. l0 + n - 1's conditioning of every kind into out[kind] + j (Rc x outC); cndP has K * (T + 1) rows (each
// step's last row zero: the padding atoms'), cndT has K * T rows
static void condGroup(const float* cndP, const float* cndT, int T, int K, int Dt, int layers, int l0, int n, const int outC[4], float* out[4]) {
  half* cin = scratch<half>("be2.cg.cin", (size_t)n * K * (T + 1) * Dt);
  for (int k = 0; k < 4; ++k) {
    const CondKind& CK = COND_KINDS[k];
    const int Rc = K * (T + CK.rowsExtra), O = outC[k];
    const float* cnd = CK.rowsExtra ? cndP : cndT;
    const std::string l0n = std::string(CK.l0), l2n = std::string(CK.l2);
    const float* w = stackedVectors(CK.key, CK, (l0n + "/weight").c_str(), layers, Dt);
    const float* b = hasW(CK.prefix(0) + l0n + "/bias") ? stackedVectors(CK.key, CK, (l0n + "/bias").c_str(), layers, Dt) : nullptr;
    const bool loopLayers = Dt == 768 && Rc >= 256;
    const size_t rows = loopLayers ? (size_t)Rc : (size_t)n * Rc, groups = (rows + 7) / 8;
    run("be2_cond_in", Grid{(uint32_t)std::min<size_t>(groups, 65535), (uint32_t)((groups + 65534) / 65535), 1}, 256,
        Be2CondInArgs{cnd, w, b, cin, (u64)rows, (uint)Rc, (uint)Dt, (uint)l0, loopLayers ? (uint)n : 0});
    std::vector<u64> X, Wp, Y;
    for (int j = 0; j < n; ++j) {
      X.push_back((u64)(cin + (size_t)j * Rc * Dt));
      Wp.push_back((u64)M.h("w/" + CK.prefix(l0 + j) + l2n + "/weight"));
      Y.push_back((u64)(out[k] + (size_t)j * Rc * O));
    }
    const std::string pk = std::string("cg.") + CK.key + "." + num(n) + "." + num(K);
    Gemm G{}; G.ptrs = true; G.batch = n;
    G.X = pointerArray(pk + ".x", X); G.tx = F16; G.W = pointerArray(pk + ".w", Wp); G.tw = F16; G.transW = true;
    G.Y = pointerArray(pk + ".y", Y); G.ty = F32; G.rows = Rc; G.in = Dt; G.out = O; G.label = "conditioning, layers batched";
    if (hasW(CK.prefix(0) + l2n + "/bias"))
      G.bias = stackedVectors(CK.key, CK, (l2n + "/bias").c_str(), layers, O) + (size_t)l0 * O;
    gemm(G);
  }
}

// ---------------------------------------------------------------- one call
// the row maps of `samples` stacked samples: each atom to its token's conditioning row (padding atoms to the zero row
// T), each token to its own
struct RowMaps { int samples = 0; int* atomTok = nullptr; int* tok = nullptr; };
static RowMaps ROW_MAPS;
static std::vector<float> SIGMA_SCHED;
static int SIGMA_NEXT = 0, SIGMA_BATCH_I0 = -1, SIGMA_BATCH_K = 0, SIGMA_MAX_K = 1;
static const float* SIGMA_SC0 = nullptr;
void setSigmaSchedule(const float* sigmas, int n) {
  if (sigmas && n > 0) SIGMA_SCHED.assign(sigmas, sigmas + n);
  else SIGMA_SCHED.clear();
  SIGMA_NEXT = 0; SIGMA_BATCH_I0 = -1; SIGMA_BATCH_K = 0; SIGMA_MAX_K = 1; SIGMA_SC0 = nullptr;
}
// 🔴 FORGOTTEN EVERY FOLD: the maps are this input's atoms' tokens, and a resident server's next job is another input -
// keyed on the sample count alone, a second fold of the same count read the first one's (cuda/bioemu2's resident server
// met this class of bug: its triangle mask remembered by address, docs/BIOEMU2.md)
void forgetRowMaps() {
  if (ROW_MAPS.atomTok) { release(ROW_MAPS.atomTok); release(ROW_MAPS.tok); }
  ROW_MAPS = {};
  setSigmaSchedule(nullptr, 0);
  recordEmitted(nullptr, nullptr, nullptr, nullptr);
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
  const bool csplit = HALF && gemmTensorOps() && splitPays(samples, T, N, Dt, Ca);
  float* cqc = scratch<float>("be2.dn.cqc", (size_t)(T + 1) * (2 * Dc + Ca));
  float* csc = scratch<float>("be2.dn.csc", (size_t)T * (2 * Dc + Dt));
  float* sq = HALF ? nullptr : scratch<float>("be2.dn.sq", RN * Dc);
  float* qs = HALF ? nullptr : scratch<float>("be2.dn.qs", RT * Dc);
  float* dq = HALF ? nullptr : scratch<float>("be2.dn.dq", RN * Ca);
  float* da = HALF ? nullptr : scratch<float>("be2.dn.da", RT * Dt);
  float* cn = scratch<float>("be2.dn.cn", (size_t)(T + 1) * Dt);
  float* sqn = HALF ? nullptr : scratch<float>("be2.dn.sqn", RN * Dc);
  float* qsn = HALF ? nullptr : scratch<float>("be2.dn.qsn", RT * Dc);
  // the conditioning in groups of layers (condGroup) - as many as fit 64 MiB, all 16 at 68 tokens
  const int outC[4] = {6 * Ca, 6 * Dt, 2 * Dc + Ca, 2 * Dc + Dt};
  const bool batchCond = HALF && BATCH_COND;
  const size_t perLayer = ((size_t)(T + 1) * (outC[0] + outC[2]) + (size_t)T * (outC[1] + outC[3])) * 4;
  const int G = batchCond ? (int)std::max<size_t>(1, std::min<size_t>(d.layers, ((size_t)64 << 20) / perLayer)) : 1;
  const int I = dimW(SM + "layers/0/coupling/s_to_q_mixer/fc1/weight", 0), ldc = 2 * Dc + Ca;
  const size_t R = (size_t)T + 1;
  // when all layers fit in one group and sample() gave its sigma schedule, batch K steps of sigma conditioning and
  // coupling RB ahead so their GEMMs run on K * T and 2 * K * (T + 1) rows instead of a single step's
  int curStep = -1, curK = 1, bi = 0;
  if (batchCond && G == d.layers && !SIGMA_SCHED.empty()) {
    if (SIGMA_SC0 != d.sc0) { SIGMA_SC0 = d.sc0; SIGMA_BATCH_I0 = -1; SIGMA_NEXT = 0; }
    if (SIGMA_NEXT > 0 && SIGMA_SCHED[SIGMA_NEXT - 1] == sigma) curStep = SIGMA_NEXT - 1;
    else if (SIGMA_NEXT < (int)SIGMA_SCHED.size() && SIGMA_SCHED[SIGMA_NEXT] == sigma) curStep = SIGMA_NEXT++;
    else setSigmaSchedule(nullptr, 0);
  }
  const int F = dimW(sc + "fourier_embed/proj/bias", 0);
  float* cnd = nullptr;
  float* cg[4] = {nullptr, nullptr, nullptr, nullptr};
  float* cgQB = nullptr;
  if (curStep >= 0) {
    const int steps = (int)SIGMA_SCHED.size();
    const size_t stepBytes = (size_t)d.layers * (perLayer + (csplit ? R * 4 * I * 4 : 0));
    const int envK = getenv("BE2_COND_STEPS") ? std::max(1, atoi(getenv("BE2_COND_STEPS")))
                                              : std::max(1, std::min(steps, (int)(((size_t)448 << 20) / std::max<size_t>(stepBytes, 1))));
    const int batches = std::max(1, (steps + envK - 1) / envK);
    SIGMA_MAX_K = std::max(1, (steps + batches - 1) / batches);
    for (int k = 0; k < 4; ++k)
      cg[k] = scratch<float>(std::string("be2.cg.") + COND_KINDS[k].key, (size_t)G * SIGMA_MAX_K * (T + COND_KINDS[k].rowsExtra) * outC[k]);
    if (csplit) cgQB = scratch<float>("be2.cg.QB", (size_t)G * SIGMA_MAX_K * R * 4 * I);
    if (SIGMA_BATCH_I0 < 0 || curStep < SIGMA_BATCH_I0 || curStep >= SIGMA_BATCH_I0 + SIGMA_BATCH_K) {
      SIGMA_BATCH_I0 = curStep;
      SIGMA_BATCH_K = std::min(SIGMA_MAX_K, steps - curStep);
      const int K = SIGMA_BATCH_K;
      float* four = scratch<float>("be2.dn.four", (size_t)K * F);
      half* fourh = scratch<half>("be2.dn.fourh", (size_t)K * F);
      float* f768 = scratch<float>("be2.dn.f768", (size_t)K * Dt);
      std::vector<float> hostTs(K);
      for (int s = 0; s < K; ++s) hostTs[s] = 0.25f * logf(SIGMA_SCHED[curStep + s] / SIGMA_DATA);
      float* tsDev = scratch<float>("be2.dn.ts", K);
      upload(tsDev, hostTs.data(), (size_t)K * 4);
      run1d("be2_cos_embed", (size_t)K * F,
            Be2CosEmbedArgs{Wf(sc + "fourier_embed/proj/weight"), Wf(sc + "fourier_embed/proj/bias"), four, 0.f, (uint)F, tsDev, (uint)K, 0});
      layerNorm(four, fourh, K, F, Wf(sc + "norm_fourier/weight"), Wf(sc + "norm_fourier/bias"));
      { Gemm Gf{}; Gf.X = fourh; Gf.tx = F16; Gf.W = M.h("w/" + sc + "fourier_to_single/weight"); Gf.tw = F16; Gf.transW = true;
        Gf.Y = f768; Gf.ty = F32; Gf.rows = K; Gf.in = F; Gf.out = Dt; Gf.label = "fourier_to_single"; gemm(Gf); }
      float* cndT = scratch<float>("be2.dn.condT_k", (size_t)K * T * Dt);
      be2Run1d("be2_repeat", (size_t)K * T * Dt, Be2RepeatArgs{d.sc0, cndT, (u64)T * Dt, (u64)K * T * Dt});
      be2Run1d("be2_broadcast_add", (size_t)K * T * Dt, Be2BroadcastAddArgs{cndT, f768, (u64)K * T, (uint)Dt, (uint)T});
      for (int i = 0; i < 2; ++i) transition(cndT, cndT, (size_t)K * T, Dt, sc + "transitions_" + num(i) + "/");
      float* cndP = scratch<float>("be2.dn.condP_k", (size_t)K * (T + 1) * Dt);
      fill(cndP, 0, (size_t)K * (T + 1) * Dt * 4);
      copy2d(cndP, (size_t)(T + 1) * Dt * 4, cndT, (size_t)T * Dt * 4, (size_t)T * Dt * 4, K);
      condGroup(cndP, cndT, T, K, Dt, d.layers, 0, G, outC, cg);
      if (cgQB) {
        const size_t RK = (size_t)K * R;
        half* gR = scratch<half>("be2.cs.gR", (size_t)G * 4 * RK * Dc);
        be2Run1d("be2_cs_gamma", (size_t)G * RK * Dc,
                 Be2CsGammaArgs{cg[2], gR, nullptr, (u64)(G * RK * Dc), (uint)Dt, (uint)Ca, (uint)ldc, (uint)RK});
        for (int part = 0; part < 2; ++part) {
          std::vector<u64> X, Wp, Y;
          for (int j = 0; j < G; ++j) {
            const std::string p = SM + "layers/" + num(j) + "/coupling/s_to_q_mixer/";
            X.push_back((u64)(gR + ((size_t)j * 2 + part) * 2 * RK * Dc));
            Wp.push_back((u64)swigluPairs(p, Dc, I));
            Y.push_back((u64)(cgQB + (size_t)j * RK * 4 * I));
          }
          const std::string pk = "cg.qb." + num(G) + "." + num(K) + "." + num(part);
          Gemm Gq{}; Gq.ptrs = true; Gq.batch = G;
          Gq.X = pointerArray(pk + ".x", X); Gq.tx = F16; Gq.W = pointerArray(pk + ".w", Wp); Gq.tw = F16;
          Gq.Y = pointerArray(pk + ".y", Y); Gq.ty = F32; Gq.rows = 2 * RK; Gq.in = Dc; Gq.out = 2 * I;
          Gq.beta = part ? 1.f : 0.f; Gq.label = "coupling RB, layers batched";
          gemm(Gq);
        }
      }
    }
    curK = SIGMA_BATCH_K;
    bi = curStep - SIGMA_BATCH_I0;
  } else {
    float* four = scratch<float>("be2.dn.four", F);
    run1d("be2_cos_embed", F, Be2CosEmbedArgs{Wf(sc + "fourier_embed/proj/weight"), Wf(sc + "fourier_embed/proj/bias"), four, t, (uint)F, nullptr, 1, 0});
    float* f768 = scratch<float>("be2.dn.f768", Dt);
    if (HALF) {
      half* fourh = scratch<half>("be2.dn.fourh", F);
      layerNorm(four, fourh, 1, F, Wf(sc + "norm_fourier/weight"), Wf(sc + "norm_fourier/bias"));
      Gemm Gf{}; Gf.X = fourh; Gf.tx = F16; Gf.W = M.h("w/" + sc + "fourier_to_single/weight"); Gf.tw = F16; Gf.transW = true; Gf.Y = f768; Gf.ty = F32;
      Gf.rows = 1; Gf.in = F; Gf.out = Dt; Gf.label = "fourier_to_single"; gemm(Gf);
    } else {
      layerNorm(four, four, 1, F, Wf(sc + "norm_fourier/weight"), Wf(sc + "norm_fourier/bias"));
      linT(four, sc + "fourier_to_single/weight", f768, 1, F, Dt);
    }
    // cond, and a ZERO row after its T: what a padding atom of the reference conditions on (c_to_q is 0 there)
    cnd = scratch<float>("be2.dn.cond", (size_t)(T + 1) * Dt);
    copy(cnd, d.sc0, (size_t)T * Dt * 4);
    fill(cnd + (size_t)T * Dt, 0, (size_t)Dt * 4);
    be2Run1d("be2_broadcast_add", (size_t)T * Dt, Be2BroadcastAddArgs{cnd, f768, (u64)T, (uint)Dt, 0});
    for (int i = 0; i < 2; ++i) transition(cnd, cnd, T, Dt, sc + "transitions_" + num(i) + "/");
    if (batchCond)
      for (int k = 0; k < 4; ++k)
        cg[k] = scratch<float>(std::string("be2.cg.") + COND_KINDS[k].key, (size_t)G * (T + COND_KINDS[k].rowsExtra) * outC[k]);
    cgQB = (batchCond && csplit) ? scratch<float>("be2.cg.QB", (size_t)G * R * 4 * I) : nullptr;
  }
  const float* pendTokH = nullptr; const float* pendTokGate = nullptr; int pendTokLdg = 0;
  for (int l = 0; l < d.layers; ++l) {
    const std::string L = SM + "layers/" + num(l) + "/";
    if (batchCond && curStep < 0 && l % G == 0) {
      const int gn = std::min(G, d.layers - l);
      condGroup(cnd, cnd, T, 1, Dt, d.layers, l, gn, outC, cg);
      if (cgQB) {
        half* gR = scratch<half>("be2.cs.gR", (size_t)gn * 4 * R * Dc);
        be2Run1d("be2_cs_gamma", (size_t)gn * R * Dc,
                 Be2CsGammaArgs{cg[2], gR, nullptr, (u64)(gn * R * Dc), (uint)Dt, (uint)Ca, (uint)ldc, (uint)R});
        for (int part = 0; part < 2; ++part) {
          std::vector<u64> X, Wp, Y;
          for (int j = 0; j < gn; ++j) {
            const std::string p = SM + "layers/" + num(l + j) + "/coupling/s_to_q_mixer/";
            X.push_back((u64)(gR + ((size_t)j * 2 + part) * 2 * R * Dc));
            Wp.push_back((u64)swigluPairs(p, Dc, I));
            Y.push_back((u64)(cgQB + (size_t)j * R * 4 * I));
          }
          const std::string pk = "cg.qb." + num(gn) + "." + num(part);
          Gemm G{}; G.ptrs = true; G.batch = gn;
          G.X = pointerArray(pk + ".x", X); G.tx = F16; G.W = pointerArray(pk + ".w", Wp); G.tw = F16;
          G.Y = pointerArray(pk + ".y", Y); G.ty = F32; G.rows = 2 * R; G.in = Dc; G.out = 2 * I;
          G.beta = part ? 1.f : 0.f; G.label = "coupling RB, layers batched";
          gemm(G);
        }
      }
    }
    const int lj = l % G;
    const size_t slot = (size_t)lj * curK + bi;
    const CondLayer atomL{L + "atom_block/transformer_layer/", ab, true, Ha};
    // (the atom stream's conditioning is its token's row of cond: computed on the T + 1 token rows)
    float* condA = batchCond ? cg[0] + slot * (T + 1) * outC[0] : layerCond(atomL, cnd, T + 1, Dt, Ca, "be2.dn.condA");
    condLayer(atomL, q, (int)RN, Ca, condA, maps.atomTok, d.wbias + wb * l, N);
    const CondLayer tokL{L + "token_block/", tb, false, Ht};
    float* condT = batchCond ? cg[1] + slot * T * outC[1] : layerCond(tokL, cnd, T, Dt, Dt, "be2.dn.condT");
    const float* tokTrH = nullptr;
    condLayer(tokL, a, (int)RT, Dt, condT, maps.tok, d.tbias ? d.tbias + tbn * l : nullptr, T,
              d.tbiasH ? d.tbiasH + tbn * l : nullptr, pendTokH, pendTokGate, pendTokLdg, HALF ? &tokTrH : nullptr);
    pendTokH = nullptr; pendTokGate = nullptr;
    // the coupling's conditioning, both on the token rows (the atom side's input is cond broadcast to atoms)
    const std::string cp = L + "coupling/";
    if (batchCond) { cqc = cg[2] + slot * (T + 1) * outC[2]; csc = cg[3] + slot * T * outC[3]; }
    else {
      layerNorm(cnd, cn, T + 1, Dt, Wf(cp + "q_coupling_cond_0/weight"), WfOpt(cp + "q_coupling_cond_0/bias"));
      silu(cn, cn, (size_t)(T + 1) * Dt);
      linT(cn, cp + "q_coupling_cond_2/weight", cqc, T + 1, Dt, 2 * Dc + Ca, 0.f, 0, 0, false, WfOpt(cp + "q_coupling_cond_2/bias"));
      layerNorm(cnd, cn, T, Dt, Wf(cp + "s_coupling_cond_0/weight"), WfOpt(cp + "s_coupling_cond_0/bias"));
      silu(cn, cn, (size_t)T * Dt);
      linT(cn, cp + "s_coupling_cond_2/weight", csc, T, Dt, 2 * Dc + Dt, 0.f, 0, 0, false, WfOpt(cp + "s_coupling_cond_2/bias"));
    }
    // s_to_q = [a broadcast | q]; q_to_s = [q centre | a]
    // (the fast arm: both inputs normalised directly into half from q and a, or s_to_q SPLIT via couplingSplit)
    if (HALF) {
      const bool csplit = gemmTensorOps() && splitPays(samples, T, N, Dt, Ca);
      auto rowsGrid = [](size_t rows) {
        const size_t groups = (rows + 7) / 8;
        return Grid{(uint32_t)std::min<size_t>(groups, 65535), (uint32_t)((groups + 65534) / 65535), 1};
      };
      half* qsh = scratch<half>("be2.dn.qsh", RT * Dc);
      float* tstats = csplit ? scratch<float>("be2.cs.tstats", RT * 3) : nullptr;
      half* xa = csplit ? scratch<half>("be2.cs.xa", RT * Dt) : nullptr;
      run("be2_ada_ln_q2s", rowsGrid(RT), 256,
          Be2AdaLnQ2SArgs{q, a, Iin("token_center_atom_index"), csc, csc + Dc, maps.tok, qsh, (u64)RT,
                          (uint)T, (uint)N, (uint)Ca, (uint)Dt, (uint)(2 * Dc + Dt), (uint)(6 * Dt),
                          tokTrH, tokTrH ? condT + 5 * Dt : nullptr,
                          csplit ? cqc : nullptr, tstats, xa, (uint)(2 * Dc + Ca), 0});
      // (q += gamma_q s_to_q and a += gamma_s q_to_s in the fc3s' epilogues: qsh is already formed from untouched q and
      // a before s_to_q updates q, and s_to_q reads untouched a before q_to_s updates a)
      if (csplit) couplingSplit(a, q, cqc, maps.atomTok, samples, T, N, Dt, Ca, cp + "s_to_q_mixer/",
                                cgQB ? cgQB + slot * R * 4 * I : nullptr, true);
      else {
        half* sqh = scratch<half>("be2.dn.sqh", RN * Dc);
        run("be2_ada_ln_s2q", rowsGrid(RN), 256,
            Be2AdaLnS2QArgs{a, q, Iin("token_idx_at_atom"), Iin("atom_pad_mask"), cqc, cqc + Dc, maps.atomTok, sqh, (u64)RN,
                            (uint)T, (uint)N, (uint)Dt, (uint)Ca, (uint)(2 * Dc + Ca), 0});
        transitionHalf(sqh, q, RN, Dc, cp + "s_to_q_mixer/", false, cqc + 2 * Dc, 2 * Dc + Ca, maps.atomTok);
      }
      transitionHalf(qsh, a, RT, Dc, cp + "q_to_s_mixer/", false, csc + 2 * Dc, 2 * Dc + Dt, maps.tok,
                     batchCond && l + 1 < d.layers && (l + 1) % G != 0 ? &pendTokH : nullptr);
      if (pendTokH) { pendTokGate = csc + 2 * Dc; pendTokLdg = 2 * Dc + Dt; }
    } else {
      be2Run1d("be2_tokens_to_atoms", RN * Dt, Be2TokensToAtomsArgs{a, Iin("token_idx_at_atom"), Iin("atom_pad_mask"), sq, (u64)N,
                                                                 (uint)T, (uint)Dt, (uint)Dc, (uint)samples, 0.f, 0});
      copy2d(sq + Dt, (size_t)Dc * 4, q, (size_t)Ca * 4, (size_t)Ca * 4, RN);
      be2Run1d("be2_centre_atoms", RT * Ca, Be2CentreAtomsArgs{q, Iin("token_center_atom_index"), qs, (uint)T, (uint)N, (uint)Ca,
                                                            (uint)Dc, (uint)samples, 0});
      copy2d(qs + Ca, (size_t)Dc * 4, a, (size_t)Dt * 4, (size_t)Dt * 4, RT);
      adaLn(sq, sqn, RN, Dc, cqc, cqc + Dc, 2 * Dc + Ca, maps.atomTok);
      transition(sqn, dq, RN, Dc, cp + "s_to_q_mixer/", false, false);
      adaLn(qs, qsn, RT, Dc, csc, csc + Dc, 2 * Dc + Dt, maps.tok);
      transition(qsn, da, RT, Dc, cp + "q_to_s_mixer/", false, false);
      be2Run1d("be2_gated_add", (RN * Ca) >> 2, Be2GatedAddArgs{q, dq, cqc + 2 * Dc, maps.atomTok, RN, (uint)Ca, (uint)(2 * Dc + Ca)});
      be2Run1d("be2_gated_add", (RT * Dt) >> 2, Be2GatedAddArgs{a, da, csc + 2 * Dc, maps.tok, RT, (uint)Dt, (uint)(2 * Dc + Dt)});
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
