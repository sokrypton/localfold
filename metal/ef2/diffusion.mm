// ESMFold2's diffusion module and its EDM sampler (cuda/ef2/src/diffusion.cuh and sampler.cuh are the reading):
//
//   conditioning: pair = [z_trunk | rel_pos] -> LN -> zProjection, + 2 transitions   (once a fold)
//                 single = LN(s_inputs) @ sProjection + noise(t), + 2 transitions     (every step)
//   r = x_noisy / sqrt(t^2 + sigma^2);  q = c0 + [r | 0] @ coordsLinear;  3 windowed SWA blocks (c0)
//   a = pool(relu(q @ toToken)) + LN(single) @ singleToToken
//   12 token blocks: a += attention(adaLN(a, single), pair bias) * sigmoid(single @ outGate + b)
//                    a += swiglu(adaLN(a, single)) * sigmoid(single @ outGate + b)
//   a = LN(a);  q = skip + gather(a @ tokenToAtom);  3 windowed SWA blocks;  x_d = EDM(x_noisy, LN(q) @ outputLinear)
#include "ef2.h"
#include <cmath>
#include <random>

static constexpr int DIFFUSION_HALF_WINDOW = 64;

// TransitionLayer: x += out(silu(LN(x) @ a) * (LN(x) @ b)), the SwiGLU in the first GEMM's epilogue
static void transitionLayer(float* x, size_t rows, int C, const std::string& B,
                            const half* inLn = nullptr, half* outLn = nullptr,
                            const float* nextScale = nullptr, const float* nextOffset = nullptr) {
  int Hd = dimOf("f/" + B + "aProjection", 1);
  const half* w = swigluPairs2(B, Fh(B + "aProjection"), Fh(B + "bProjection"), C, Hd);
  size_t chunk = std::max<size_t>(64, ((size_t)32 << 20) / (3 * (size_t)Hd));
  size_t n = std::min(rows, chunk);
  half* xn = inLn ? nullptr : scratch<half>("dtr.xn", n * C); half* g = scratch<half>("dtr.g", n * Hd);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    const half* xh = inLn ? inLn + r0 * C : xn;
    if (!inLn) layerNorm(x + r0 * C, xn, r, C, F(B + "norm/scale"), F(B + "norm/offset"));
    gemmSwiglu(xh, w, g, r, C, Hd);
    if (outLn && HALF_GEMM) {
      Gemm G{}; G.X = g; G.tx = F16; G.W = M.h("f/" + B + "outProjection"); G.tw = F16; G.Y = x + r0 * C; G.ty = F32;
      G.rows = r; G.in = Hd; G.out = C; G.beta = 1.f; G.label = "transition outProjection";
      G.lnOut = outLn + r0 * C; G.lnScale = nextScale; G.lnOffset = nextOffset;
      if (!gemm(G)) layerNorm(x + r0 * C, outLn + r0 * C, r, C, nextScale, nextOffset);
    } else {
      lin(g, "f/" + B + "outProjection", x + r0 * C, r, Hd, C, 1.f);
    }
  }
}

static const float* pairBiasScales(int nb, int Cz) {
  return M.derived<float>("ef2.pb.scales", (size_t)nb * Cz, [&](float* out) {
    for (int b = 0; b < nb; ++b)
      copy(out + (size_t)b * Cz, F("diffusion/tokenBlocks/" + std::to_string(b) + "/attention/pairNormScale"), (size_t)Cz * 4);
  });
}
static const float* pairBiasOffsets(int nb, int Cz) {
  return M.derived<float>("ef2.pb.offsets", (size_t)nb * Cz, [&](float* out) {
    for (int b = 0; b < nb; ++b)
      copy(out + (size_t)b * Cz, F("diffusion/tokenBlocks/" + std::to_string(b) + "/attention/pairNormOffset"), (size_t)Cz * 4);
  });
}
static const half* pairBiasWeights(int nb, int Cz, int H) {
  return M.derived<half>("ef2.pb.weights", (size_t)nb * Cz * H, [&](half* out) {
    for (int b = 0; b < nb; ++b)
      copy(out + (size_t)b * Cz * H, Fh("diffusion/tokenBlocks/" + std::to_string(b) + "/attention/pairBiasWeights"), (size_t)Cz * H * 2);
  });
}

Denoiser makeDenoiser(int T, int A, const float* zTrunk, const float* sInputs) {
  Denoiser d{};
  d.T = T; d.A = A; d.Cz = (int)M.meta("meta/pairChannels"); d.Ct = (int)M.meta("meta/tokenChannels2");
  d.heads = (int)M.meta("meta/tokenHeads"); d.tokenBlocks = (int)M.meta("meta/tokenBlocks");
  d.Si = (int)M.meta("meta/singleInputs"); d.atomBlocks = (int)M.meta("meta/atomBlocks"); d.sigma = (float)M.meta("meta/sigmaData");
  d.sInputs = sInputs;
  size_t P = (size_t)T * T; int Cz = d.Cz;
  // the conditioning pair, row chunks of 64 MB: [z | rel_pos] -> LN -> zProjection, then two transitions
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (8 * (size_t)Cz));
  half* jn = scratch<half>("dc.jn", chunk * 2 * Cz);
  float* pair = scratch<float>("dc.pair", P * Cz);
  half* trLn = (HALF_GEMM && chunk == P && Cz == 256) ? scratch<half>("dc.trln", P * Cz) : nullptr;
  bool haveTrLn = false;
  for (size_t p0 = 0; p0 < P; p0 += chunk) {
    size_t n = std::min(chunk, P - p0);
    if (Cz == 256) {
      run("ef2_join_pair_rel", grid1d((n + 7) / 8, 1), 256,
          JoinPairRelArgs{zTrunk, relIdx(), nullptr, p0, n, (uint)T, (uint)Cz,
                          F("diffusion/zInputNorm/scale"), F("diffusion/zInputNorm/offset"), jn});
    } else {
      float* joined = scratch<float>("dc.joined", chunk * 2 * Cz);
      run1d("ef2_join_pair_rel", (n * 2 * Cz) >> 2,
            JoinPairRelArgs{zTrunk, relIdx(), joined, p0, n, (uint)T, (uint)Cz, nullptr, nullptr, nullptr});
      layerNorm(joined, jn, n, 2 * Cz, F("diffusion/zInputNorm/scale"), F("diffusion/zInputNorm/offset"));
    }
    if (trLn) {
      Gemm G{}; G.X = jn; G.tx = F16; G.W = M.h("f/diffusion/zProjection"); G.tw = F16; G.Y = pair + p0 * Cz; G.ty = F32;
      G.rows = n; G.in = 2 * Cz; G.out = Cz; G.label = "f/diffusion/zProjection";
      G.lnOut = trLn; G.lnScale = F("diffusion/zTransitions/0/norm/scale"); G.lnOffset = F("diffusion/zTransitions/0/norm/offset");
      haveTrLn = gemm(G);
    } else {
      lin(jn, "f/diffusion/zProjection", pair + p0 * Cz, n, 2 * Cz, Cz);
    }
  }
  if (trLn && haveTrLn) {
    transitionLayer(pair, P, Cz, "diffusion/zTransitions/0/", trLn, trLn,
                    F("diffusion/zTransitions/1/norm/scale"), F("diffusion/zTransitions/1/norm/offset"));
    transitionLayer(pair, P, Cz, "diffusion/zTransitions/1/", trLn);
  } else {
    for (int l = 0; l < 2; ++l) transitionLayer(pair, P, Cz, "diffusion/zTransitions/" + std::to_string(l) + "/");
  }
  // each token block's pair bias, [H, T, T] in half (the pair itself is not kept)
  d.stride = (T + 7) / 8 * 8;
  const size_t perBias = (size_t)d.heads * T * d.stride;
  d.biasBuf = allocT<half>((size_t)d.tokenBlocks * perBias);
  for (int b = 0; b < d.tokenBlocks; ++b) d.biases.push_back(d.biasBuf + (size_t)b * perBias);
  if (HALF_GEMM && Cz == 256) {
    const int nb = d.tokenBlocks, H = d.heads;
    half* pn = scratch<half>("dc.pn", (size_t)nb * chunk * Cz);
    float* pb = scratch<float>("dc.pb", (size_t)nb * P * H);
    const float* sc = pairBiasScales(nb, Cz);
    const float* of = pairBiasOffsets(nb, Cz);
    const half* pw = pairBiasWeights(nb, Cz, H);
    for (size_t p0 = 0; p0 < P; p0 += chunk) {
      size_t n = std::min(chunk, P - p0);
      run("ef2_scale_copies", grid1d((n + 7) / 8, 1), 256,
          ScaleCopiesArgs{pair + p0 * Cz, sc, nullptr, pn, (uint)n, (uint)Cz, (uint)nb, 1, of});
      Gemm G{}; G.X = pn; G.tx = F16; G.sx = (int64_t)n * Cz; G.W = pw; G.tw = F16; G.sw = (int64_t)Cz * H;
      G.Y = pb + p0 * H; G.ty = F32; G.sy = (int64_t)P * H; G.rows = n; G.in = Cz; G.out = H; G.batch = nb;
      G.label = "pairBiasWeights";
      gemm(G);
    }
    run1d("ef2_pair_to_heads", (size_t)nb * (H == 16 ? P * 4 : P * H),
          PairToHeadsArgs{pb, d.biasBuf, P, (uint)H, (uint)T, (uint)d.stride, (uint)nb});
  } else {
    half* pn = scratch<half>("dc.pn", chunk * Cz); float* pb = scratch<float>("dc.pb", P * d.heads);
    for (int b = 0; b < d.tokenBlocks; ++b) {
      std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/attention/";
      for (size_t p0 = 0; p0 < P; p0 += chunk) {
        size_t n = std::min(chunk, P - p0);
        layerNorm(pair + p0 * Cz, pn, n, Cz, F(B + "pairNormScale"), F(B + "pairNormOffset"));
        lin(pn, "f/" + B + "pairBiasWeights", pb + p0 * d.heads, n, Cz, d.heads);
      }
      run1d("ef2_pair_to_heads", d.heads == 16 ? P * 4 : P * d.heads,
            PairToHeadsArgs{pb, d.biases[b], P, (uint)d.heads, (uint)T, (uint)d.stride, 1});
    }
  }
  releaseScratch({"dc.joined", "dc.jn", "dc.trln", "dc.pair", "dc.pn", "dc.pb", "dtr."});
  d.atoms = prepareAtoms(A, "diffusionAtomEncoder");
  {
    int Ca = d.atoms.ctx.C;
    size_t n = (size_t)A * Ca;
    d.encMod = allocT<float>((size_t)d.atomBlocks * A * 6 * Ca);
    d.decMod = allocT<float>((size_t)d.atomBlocks * A * 6 * Ca);
    float* sc = HALF_GEMM ? nullptr : scratch<float>("atom.silu", n);
    half* sch = HALF_GEMM ? scratch<half>("atom.siluh", n) : nullptr;
    run1d("ef2_silu", n, SiluArgs{d.atoms.c0, sc, sch, n});
    if (HALF_GEMM) {
      int nb = d.atomBlocks, k = 2 * nb;
      uint64_t* ptrs = scratch<uint64_t>("atom.ptrs", 3 * k);
      std::vector<uint64_t> hp(3 * k);
      for (int b = 0; b < nb; ++b) {
        hp[2 * b] = (uint64_t)M.h("f/diffusionAtomEncoder/blocks/" + std::to_string(b) + "/adaln");
        hp[2 * b + 1] = (uint64_t)M.h("f/diffusionAtomDecoder/blocks/" + std::to_string(b) + "/adaln");
        hp[k + 2 * b] = hp[k + 2 * b + 1] = (uint64_t)sch;
        hp[2 * k + 2 * b] = (uint64_t)(d.encMod + (size_t)b * A * 6 * Ca);
        hp[2 * k + 2 * b + 1] = (uint64_t)(d.decMod + (size_t)b * A * 6 * Ca);
      }
      upload(ptrs, hp.data(), 3 * k * 8);
      Gemm G{}; G.X = ptrs + k; G.tx = F16; G.W = ptrs; G.tw = F16; G.Y = ptrs + 2 * k; G.ty = F32;
      G.rows = A; G.in = Ca; G.out = 6 * Ca; G.batch = k; G.ptrs = true; G.label = "atom adaln";
      gemm(G);
    } else {
      for (int b = 0; b < d.atomBlocks; ++b) {
        lin(sc, "f/diffusionAtomEncoder/blocks/" + std::to_string(b) + "/adaln", d.encMod + (size_t)b * A * 6 * Ca, A, Ca, 6 * Ca);
        lin(sc, "f/diffusionAtomDecoder/blocks/" + std::to_string(b) + "/adaln", d.decMod + (size_t)b * A * 6 * Ca, A, Ca, 6 * Ca);
      }
    }
  }
  int Ct = d.Ct, nb = d.tokenBlocks;
  d.entries = 6 * nb;
  d.sBase = allocT<float>((size_t)T * Ct);
  {
    half* sn = scratch<half>("dc.sn", (size_t)T * d.Si);
    layerNorm(d.sInputs, sn, T, d.Si, F("diffusion/sInputNorm/scale"), F("diffusion/sInputNorm/offset"));
    lin(sn, "f/diffusion/sProjection", d.sBase, T, d.Si, Ct);
    releaseScratch({"dc.sn"});
  }
  d.single = allocT<float>((size_t)T * Ct); d.snScaled = allocT<float>((size_t)(2 * nb + 1) * T * Ct);
  d.G = allocT<float>((size_t)T * d.entries * Ct); d.scales = allocT<float>((size_t)(2 * nb + 1) * Ct);
  for (int b = 0; b < nb; ++b)
    for (int k = 0; k < 2; ++k)
      copy(d.scales + (size_t)(2 * b + k) * Ct, F("diffusion/tokenBlocks/" + std::to_string(b) + (k ? "/transition/" : "/attention/") +
                                                   "adaln/singleScale"), Ct * 4);
  copy(d.scales + (size_t)(2 * nb) * Ct, F("diffusion/stepNorm/scale"), Ct * 4);
  d.level = allocT<float>(4);
  return d;
}
void freeDenoiser(Denoiser& d) {
  if (d.biasBuf) release(d.biasBuf);
  else for (half* b : d.biases) release(b);
  for (void* p : {(void*)d.sBase, (void*)d.single, (void*)d.snScaled, (void*)d.G, (void*)d.scales, (void*)d.level, (void*)d.encMod, (void*)d.decMod}) release(p);
  freeAtoms(d.atoms);
  d.biases.clear();
  d.biasBuf = nullptr;
}

// every token block's adaLN gate and shift projections (one batched GEMM over the 2nb scaled copies of LN(single), each
// against its [gate | shift] weight) and their two out gates (one GEMM over single): G [T, 6nb Ct] - entry 4b + 2k and
// 4b + 2k + 1 block b's k-th adaLN gate and shift, entry 4nb + 2b + k its k-th out gate
static const half* adaWeights(int nb, int Ct) {
  return M.derived<half>("ef2.ada", (size_t)2 * nb * Ct * 2 * Ct, [&](half* w) {
    for (int b = 0; b < nb; ++b)
      for (int k = 0; k < 2; ++k) {
        std::string P = "diffusion/tokenBlocks/" + std::to_string(b) + (k ? "/transition/" : "/attention/");
        half* o = w + (size_t)(2 * b + k) * Ct * 2 * Ct;
        run1d("ef2_stack_cols", (size_t)Ct * (Ct >> 1),
              StackColsArgs{Fh(P + "adaln/gateWeights"), Fh(P + "adaln/shiftWeights"), nullptr, o, (uint)Ct, (uint)Ct, (uint)Ct, 0});
      }
  });
}
static const half* outGateWeights(int nb, int Ct) {
  return M.derived<half>("ef2.outgates", (size_t)Ct * 2 * nb * Ct, [&](half* w) {
    for (int b = 0; b < nb; ++b)
      for (int k = 0; k < 2; ++k)
        copy2d(w + (size_t)(2 * b + k) * Ct, (size_t)2 * nb * Ct * 2,
               Fh("diffusion/tokenBlocks/" + std::to_string(b) + (k ? "/transition/" : "/attention/") + "outGateWeights"), Ct * 2, Ct * 2, Ct);
  });
}
static const float* outGateBiases(int nb, int Ct) {
  return M.derived<float>("ef2.outgateb", (size_t)2 * nb * Ct, [&](float* w) {
    for (int b = 0; b < nb; ++b)
      for (int k = 0; k < 2; ++k)
        copy(w + (size_t)(2 * b + k) * Ct,
             F("diffusion/tokenBlocks/" + std::to_string(b) + (k ? "/transition/" : "/attention/") + "outGateBias"), Ct * 4);
  });
}
// a token block's q | k | v | gate projection as one [Ct, 4Ct] weight, and its bias (the query's; zero elsewhere)
static const half* qkvgWeight(int b, int C) {
  std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/attention/";
  return M.derived<half>("ef2.qkvg." + std::to_string(b), (size_t)C * 4 * C, [&](half* w) {
    run1d("ef2_stack_cols", (size_t)C * C,
          StackColsArgs{Fh(B + "queryWeights"), Fh(B + "kvWeights"), Fh(B + "gateWeights"), w, (uint)C, (uint)C, (uint)(2 * C), (uint)C});
  });
}

static void conditioningSteps(const Denoiser& d, const float* levels, int K, float* single, float* G, float* sTok) {
  int T = d.T, Ct = d.Ct, nb = d.tokenBlocks, n = 2 * nb, ld = d.entries * Ct;
  size_t KT = (size_t)K * T;
  int nf = (int)M.len("f/diffusion/fourier/weights");
  float* four = scratch<float>("dc.fourier", (size_t)K * nf); float* noise = scratch<float>("dc.noise", (size_t)K * Ct);
  run1d("ef2_fourier", (size_t)K * nf, FourierArgs{F("diffusion/fourier/weights"), F("diffusion/fourier/offsets"), levels, four, (uint)nf, (uint)K});
  if (HALF_GEMM) {
    half* fourH = scratch<half>("dc.fourierH", (size_t)K * nf);
    layerNorm(four, fourH, K, nf, F("diffusion/noiseNorm/scale"), F("diffusion/noiseNorm/offset"));
    lin(fourH, "f/diffusion/noiseProjection", noise, K, nf, Ct);
  } else {
    layerNorm(four, four, K, nf, F("diffusion/noiseNorm/scale"), F("diffusion/noiseNorm/offset"));
    lin(four, "f/diffusion/noiseProjection", noise, K, nf, Ct);
  }
  run1d("ef2_add_row", (KT * Ct) >> 2, AddRowArgs{d.sBase, noise, single, (u64)KT, (uint)Ct, (uint)T});
  for (int l = 0; l < 2; ++l) transitionLayer(single, KT, Ct, "diffusion/sTransitions/" + std::to_string(l) + "/");
  half* snScaledH = nullptr;
  float* snScaledF = nullptr;
  half* snStep = nullptr;
  if (HALF_GEMM && Ct == 768) {
    snScaledH = K == 1 ? (half*)(void*)d.snScaled : scratch<half>("dn.snsH", (size_t)(n + 1) * KT * Ct);
    snStep = snScaledH + (size_t)n * KT * Ct;
    run("ef2_scale_copies", grid1d((KT + 7) / 8, 1), 256,
        ScaleCopiesArgs{single, d.scales, nullptr, snScaledH, (uint)KT, (uint)Ct, (uint)n, 1, F("diffusion/stepNorm/offset")});
  } else {
    float* sn = scratch<float>("dn.sn0", KT * Ct);
    layerNorm(single, sn, KT, Ct, nullptr, nullptr);
    snScaledH = K == 1 ? (half*)(void*)d.snScaled : (HALF_GEMM ? scratch<half>("dn.snsH", (size_t)n * KT * Ct) : nullptr);
    snScaledF = K == 1 ? d.snScaled : (HALF_GEMM ? nullptr : scratch<float>("dn.snsF", (size_t)n * KT * Ct));
    run1d("ef2_scale_copies", ((size_t)n * KT * Ct) >> 2,
          ScaleCopiesArgs{sn, d.scales, HALF_GEMM ? nullptr : snScaledF, HALF_GEMM ? snScaledH : nullptr, (uint)KT, (uint)Ct, (uint)n, 0, nullptr});
    snStep = scratch<half>("dn.sn", KT * Ct);
    layerNorm(single, snStep, KT, Ct, F("diffusion/stepNorm/scale"), F("diffusion/stepNorm/offset"));
  }
  Gemm g{};
  if (HALF_GEMM) { g.X = snScaledH; g.tx = F16; } else { g.X = snScaledF; g.half = true; }
  g.sx = (int64_t)KT * Ct; g.W = adaWeights(nb, Ct); g.tw = F16; g.sw = (int64_t)Ct * 2 * Ct;
  g.Y = G; g.ldy = ld; g.sy = 2 * Ct; g.rows = KT; g.in = Ct; g.out = 2 * Ct; g.batch = n; g.label = "adaLN projections";
  gemm(g);
  Gemm o{}; o.X = single; o.W = outGateWeights(nb, Ct); o.tw = F16; o.half = true; o.Y = G + (size_t)4 * nb * Ct; o.ldy = ld;
  o.bias = outGateBiases(nb, Ct); o.rows = KT; o.in = Ct; o.out = 2 * nb * Ct; o.label = "out gates";
  gemm(o);
  lin(snStep, "f/diffusion/singleToToken", sTok, KT, Ct, Ct);
}
// adaLN into half: sigmoid(gate + gateBias) * LN(a) + shift, the projections from G (optionally folding a preceding gated residual into a)
static void adaLN(const Denoiser& d, const float* G, float* a, half* out, int e, const std::string& B,
                  const float* resY = nullptr, const float* resGate = nullptr) {
  int T = d.T, C = d.Ct, ld = d.entries * C;
  run("ef2_ada_ln", grid1d((size_t)(T + 7) / 8, 1), 256,
      AdaCombineArgs{a, G + (size_t)e * C, F(B + "gateBias"), G + (size_t)(e + 1) * C, out, resY, resGate, (uint)T, (uint)C, (uint)ld, 0});
}
static void tokenBlock(const Denoiser& d, const float* G, float* a, int b, const float*& pendingY, const float*& pendingGate) {
  int T = d.T, C = d.Ct, Hh = d.heads, D = C / Hh, ld = d.entries * C, nb = d.tokenBlocks;
  std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/";
  half* x = scratch<half>("tb.x", (size_t)T * C);
  adaLN(d, G, a, x, 4 * b, B + "attention/adaln/", pendingY, pendingGate);
  pendingY = pendingGate = nullptr;
  // q | k | v | gate in one GEMM (half, the attention's layout), then the flash kernel: the query bias, QK^T, the pair
  // bias, the softmax, PV and the gate - no [H, T, T] scores
  half* qkvg = scratch<half>("tb.qkvg", (size_t)T * 4 * C);
  {
    Gemm g{}; g.X = x; g.tx = F16; g.W = qkvgWeight(b, C); g.tw = F16; g.Y = qkvg; g.ty = F16; g.rows = T; g.in = C; g.out = 4 * C;
    g.accFloat = true; g.label = "token qkvg";
    gemm(g);
  }
  half* ctx = scratch<half>("tb.ctx", (size_t)T * C);
  Attention at{}; at.qkvg = qkvg; at.out = ctx; at.n = T; at.heads = Hh; at.D = D; at.rows = 1; at.scale = 1.f / sqrtf((float)D);
  at.bias = d.biases[b]; at.biasStride = d.stride; at.qBias = F(B + "attention/queryBias");
  attention(at);
  float* o = scratch<float>("tb.o", (size_t)T * C);
  const float* gAtt = G + (size_t)(4 * nb + 2 * b) * C;
  if (!HALF_GEMM || !gemmGateResidual(ctx, Fh(B + "attention/outWeights"), a, T, T, C, C, gAtt, ld, T, 1.f, "token outWeights")) {
    lin(ctx, "f/" + B + "attention/outWeights", o, T, C, C);
    pendingY = o; pendingGate = gAtt;
  }
  // the conditioned transition
  adaLN(d, G, a, x, 4 * b + 2, B + "transition/adaln/", pendingY, pendingGate);
  pendingY = pendingGate = nullptr;
  int Hd = dimOf("f/" + B + "transition/outWeights", 0);
  half* g = scratch<half>("tb.gated", (size_t)T * Hd);
  gemmSwiglu(x, swigluPairsInPlace("f/" + B + "transition/swishWeights", C, Hd), g, T, C, Hd);
  const float* gTr = G + (size_t)(4 * nb + 2 * b + 1) * C;
  if (!HALF_GEMM || !gemmGateResidual(g, Fh(B + "transition/outWeights"), a, T, T, Hd, C, gTr, ld, T, 1.f, "transition outWeights")) {
    lin(g, "f/" + B + "transition/outWeights", o, T, Hd, C);
    if (b + 1 < nb) { pendingY = o; pendingGate = gTr; }
    else run1d("ef2_sigmoid_mul", (size_t)T * C, SigmoidMulArgs{o, gTr, nullptr, a, (uint)T, (uint)C, (uint)ld, 0});
  }
}

// one denoiser call: x_noisy [A, 3] at the noise level in `level` -> x_denoised [A, 3]
static void denoiseStep(const Denoiser& d, const float* xNoisy, float* xDenoised, const float* level,
                        const float* G, const float* sTok) {
  int T = d.T, A = d.A, Ct = d.Ct; const AtomCtx& ac = d.atoms.ctx; int Ca = ac.C;
  float* r6 = scratch<float>("dn.r6", (size_t)A * 6);
  run1d("ef2_coords_input", (size_t)A * 6, CoordsInputArgs{xNoisy, level, r6, (uint)A, 0});
  float* q = scratch<float>("dn.q", (size_t)A * Ca);
  lin(r6, "f/diffusionAtomEncoder/coordsLinear", q, A, 6, Ca, 1.f, nullptr, 1.f, d.atoms.c0);     // + c0
  swaStack(ac, q, d.atoms.c0, "diffusionAtomEncoder", d.atomBlocks, DIFFUSION_HALF_WINDOW, d.encMod);
  float* tok = scratch<float>("dn.tok", (size_t)A * Ct);
  {
    Gemm g{}; g.X = q; g.W = M.h("f/diffusionAtomEncoder/toToken"); g.tw = F16; g.half = true; g.Y = tok; g.rows = A; g.in = Ca;
    g.out = Ct; g.relu = true; g.label = "toToken";
    gemm(g);
  }
  float* a = scratch<float>("dn.a", (size_t)T * Ct);
  run1d("ef2_scatter_mean", ((size_t)T * Ct) >> 2, ScatterMeanArgs{tok, ac.tokenStart, ac.tokenAtoms, ac.mask, a, (uint)T, (uint)Ct, (uint)Ct, 0, sTok});
  const float* pendingY = nullptr; const float* pendingGate = nullptr;
  for (int b = 0; b < d.tokenBlocks; ++b) tokenBlock(d, G, a, b, pendingY, pendingGate);
  half* an = scratch<half>("dn.an", (size_t)T * Ct);
  layerNorm(a, an, T, Ct, F("diffusion/tokenNorm/scale"), F("diffusion/tokenNorm/offset"));
  // the decoder: q (the encoder's output, the skip) + the tokens gathered back
  float* pt = scratch<float>("dn.pt", (size_t)T * Ca);
  lin(an, "f/diffusionAtomDecoder/tokenToAtom", pt, T, Ct, Ca);
  run1d("ef2_gather_tokens", ((size_t)A * Ca) >> 2, GatherTokensArgs{pt, Ii("atom_to_token"), ac.mask, q, (uint)A, (uint)Ca});
  swaStack(ac, q, d.atoms.c0, "diffusionAtomDecoder", d.atomBlocks, DIFFUSION_HALF_WINDOW, d.decMod);
  float* r = scratch<float>("dn.r", (size_t)A * 3);
  if (HALF_GEMM) {
    half* qh = scratch<half>("dn.qh", (size_t)A * Ca);
    layerNorm(q, qh, A, Ca, F("diffusionAtomDecoder/norm/scale"), F("diffusionAtomDecoder/norm/offset"));
    lin(qh, "f/diffusionAtomDecoder/outputLinear", r, A, Ca, 3);
  } else {
    layerNorm(q, q, A, Ca, F("diffusionAtomDecoder/norm/scale"), F("diffusionAtomDecoder/norm/offset"));
    lin(q, "f/diffusionAtomDecoder/outputLinear", r, A, Ca, 3);
  }
  run1d("ef2_edm_combine", (size_t)A * 3, EdmArgs{xNoisy, r, xDenoised, level, (uint)(A * 3), 0});
}
static void setLevel(const Denoiser& d, float t) {
  float s2 = d.sigma * d.sigma, t2 = t * t;
  float v[4] = {0.25f * logf(fmaxf(t / d.sigma, 1e-20f)), 1.f / sqrtf(t2 + s2), s2 / (s2 + t2), d.sigma * t / sqrtf(s2 + t2)};
  upload(d.level, v, sizeof v);
}
void denoise(const Denoiser& d, const float* xNoisy, float t, float* xDenoised) {
  setLevel(d, t);
  float* sTok = scratch<float>("dn.sTok", (size_t)d.T * d.Ct);
  conditioningSteps(d, d.level, 1, d.single, d.G, sTok);
  denoiseStep(d, xNoisy, xDenoised, d.level, d.G, sTok);
}

// ---------------------------------------------------------------- the sampler
bool samplerFlag(const char* f, SamplerSettings& s, bool* bad) {
  auto value = [&](const char* name, const char* a, const char* b, bool& set) {
    const size_t n = strlen(name);
    if (strncmp(f, name, n)) return false;
    if (!strcmp(f + n, a)) set = false; else if (!strcmp(f + n, b)) set = true;
    else { fprintf(stderr, "%s takes %s or %s, not %s\n", name, a, b, f + n); if (!bad) exit(1); *bad = true; }
    return true;
  };
  if (value("--schedule=", "karras", "geometric", s.geometric) || value("--update=", "euler", "exponential", s.exponential)) return true;
  if (!strcmp(f, "--no-churn")) { s.churn = false; return true; }
  return false;
}
static std::vector<double> noiseSchedule(const SamplerSettings& s, double sigmaData) {
  std::vector<double> v;
  if (s.steps == 1) v = {s.sMax * sigmaData, 0};
  else {
    double hi = pow(s.sMax, 1 / s.p), lo = pow(s.sMin, 1 / s.p);
    for (int k = 0; k < s.steps; ++k)
      v.push_back(s.geometric ? sigmaData * exp(log(s.sMax) + (double)k / (s.steps - 1) * (log(s.sMin) - log(s.sMax)))
                              : sigmaData * pow(hi + (double)k / (s.steps - 1) * (lo - hi), s.p));
    v.push_back(0);
  }
  if (s.maxSigma <= 0) return v;
  std::vector<double> out{s.maxSigma};
  for (double x : v) if (x <= s.maxSigma) out.push_back(x);
  return out;
}
// x (rows, 3) aligned onto target by the weighted Kabsch rotation (double; a reflection refused through the determinant)
static void rigidAlign(std::vector<float>& x, const std::vector<float>& target, const std::vector<float>& w, int n) {
  double tot = 0, cx[3] = {0, 0, 0}, ct[3] = {0, 0, 0};
  for (int a = 0; a < n; ++a) { tot += w[a]; for (int k = 0; k < 3; ++k) { cx[k] += w[a] * x[a * 3 + k]; ct[k] += w[a] * target[a * 3 + k]; } }
  tot = std::max(tot, 1e-8);
  for (int k = 0; k < 3; ++k) { cx[k] /= tot; ct[k] /= tot; }
  double h[9] = {0};
  for (int a = 0; a < n; ++a) {
    if (w[a] == 0) continue;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
      h[i * 3 + j] += w[a] * (target[a * 3 + i] - ct[i]) * (x[a * 3 + j] - cx[j]);
  }
  double m[9], v[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { double s = 0; for (int k = 0; k < 3; ++k) s += h[k * 3 + i] * h[k * 3 + j]; m[i * 3 + j] = s; }
  for (int sweep = 0; sweep < 32; ++sweep) {
    double off = m[1] * m[1] + m[2] * m[2] + m[5] * m[5];
    if (off < 1e-30) break;
    const int P[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (auto& pq : P) {
      int p = pq[0], q = pq[1];
      double apq = m[p * 3 + q];
      if (fabs(apq) < 1e-300) continue;
      double theta = (m[q * 3 + q] - m[p * 3 + p]) / (2 * apq);
      double t = (theta >= 0 ? 1 : -1) / (fabs(theta) + sqrt(theta * theta + 1));
      double c = 1 / sqrt(t * t + 1), s = t * c;
      for (int k = 0; k < 3; ++k) { double akp = m[k * 3 + p], akq = m[k * 3 + q]; m[k * 3 + p] = c * akp - s * akq; m[k * 3 + q] = s * akp + c * akq; }
      for (int k = 0; k < 3; ++k) { double apk = m[p * 3 + k], aqk = m[q * 3 + k]; m[p * 3 + k] = c * apk - s * aqk; m[q * 3 + k] = s * apk + c * aqk; }
      for (int k = 0; k < 3; ++k) { double vkp = v[k * 3 + p], vkq = v[k * 3 + q]; v[k * 3 + p] = c * vkp - s * vkq; v[k * 3 + q] = s * vkp + c * vkq; }
    }
  }
  double ev[3] = {m[0], m[4], m[8]};
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int a, int b) { return ev[a] > ev[b]; });
  double vv[9], uu[9] = {0};
  bool good[3] = {false, false, false};
  double floor = 1e-12 * fabs(ev[order[0]] ? ev[order[0]] : 1);
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) vv[r * 3 + c] = v[r * 3 + order[c]];
    double sig = sqrt(std::max(ev[order[c]], 0.0));
    if (ev[order[c]] <= floor) continue;
    for (int r = 0; r < 3; ++r) { double s = 0; for (int k = 0; k < 3; ++k) s += h[r * 3 + k] * vv[k * 3 + c]; uu[r * 3 + c] = s / sig; }
    double len = sqrt(uu[c] * uu[c] + uu[3 + c] * uu[3 + c] + uu[6 + c] * uu[6 + c]);
    if (fabs(len - 1) > 1e-6) { uu[c] = uu[3 + c] = uu[6 + c] = 0; } else good[c] = true;
  }
  for (int c = 0; c < 3; ++c) {
    if (good[c]) continue;
    double best[3] = {0, 0, 0}, bestLen = -1;
    for (int sd = 0; sd < 3; ++sd) {
      double wv[3] = {0, 0, 0}; wv[sd] = 1;
      for (int k = 0; k < 3; ++k) if (good[k]) { double dot = wv[0] * uu[k] + wv[1] * uu[3 + k] + wv[2] * uu[6 + k]; for (int r = 0; r < 3; ++r) wv[r] -= dot * uu[r * 3 + k]; }
      double len = sqrt(wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2]);
      if (len > bestLen) { bestLen = len; for (int r = 0; r < 3; ++r) best[r] = wv[r]; }
    }
    for (int r = 0; r < 3; ++r) uu[r * 3 + c] = best[r] / bestLen;
    good[c] = true;
  }
  double uvt[9];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { double s = 0; for (int k = 0; k < 3; ++k) s += uu[i * 3 + k] * vv[j * 3 + k]; uvt[i * 3 + j] = s; }
  double det = uvt[0] * (uvt[4] * uvt[8] - uvt[5] * uvt[7]) - uvt[1] * (uvt[3] * uvt[8] - uvt[5] * uvt[6]) + uvt[2] * (uvt[3] * uvt[7] - uvt[4] * uvt[6]);
  double sign = det < 0 ? -1 : 1, r[9];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { double s = 0; for (int k = 0; k < 3; ++k) s += uu[i * 3 + k] * (k == 2 ? sign : 1) * vv[j * 3 + k]; r[i * 3 + j] = s; }
  for (int a = 0; a < n; ++a) {
    double dd[3] = {x[a * 3] - cx[0], x[a * 3 + 1] - cx[1], x[a * 3 + 2] - cx[2]};
    for (int i = 0; i < 3; ++i) x[a * 3 + i] = (float)(dd[0] * r[i * 3] + dd[1] * r[i * 3 + 1] + dd[2] * r[i * 3 + 2] + ct[i]);
  }
}
// the draws are cuda/ef2's own (a seeded mt19937_64): the same seed, the same sample
std::vector<float> sample(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int* stepsRun,
                          const std::function<void(const float*, int, int)>& onStep) {
  int A = d.A;
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> N(0, 1);
  std::vector<double> sched = noiseSchedule(s, d.sigma);
  std::vector<float> mask = download(d.atoms.ctx.mask, A);
  std::vector<float> x(A * 3), xd(A * 3);
  for (auto& v : x) v = (float)(sched[0] * N(rng));
  float* dx = allocT<float>((size_t)A * 3); float* dd = allocT<float>((size_t)A * 3);
  int steps = (int)sched.size() - 1;
  std::vector<float> hostLevels(std::max(steps, 1) * 4);
  float s2 = d.sigma * d.sigma;
  for (int i = 0; i < steps; ++i) {
    double sigma = sched[i], next = sched[i + 1], gamma = s.churn && next > s.gammaMin ? s.gamma0 : 0;
    float tf = (float)(sigma * (1 + gamma)), t2 = tf * tf;
    hostLevels[i * 4 + 0] = 0.25f * logf(fmaxf(tf / d.sigma, 1e-20f));
    hostLevels[i * 4 + 1] = 1.f / sqrtf(t2 + s2);
    hostLevels[i * 4 + 2] = s2 / (s2 + t2);
    hostLevels[i * 4 + 3] = d.sigma * tf / sqrtf(s2 + t2);
  }
  float* levels = scratch<float>("dn.levels", (size_t)std::max(steps, 1) * 4);
  if (steps > 0) upload(levels, hostLevels.data(), (size_t)steps * 16);
  size_t stepBytes = (size_t)d.T * (d.entries + d.tokenBlocks + 4) * d.Ct * 4;
  int maxK = getenv("EF2_COND_BATCH") ? std::max(1, atoi(getenv("EF2_COND_BATCH")))
                                      : std::max(1, std::min(steps, (int)(((size_t)320 << 20) / std::max<size_t>(stepBytes, 1))));
  int batches = std::max(1, (steps + maxK - 1) / maxK);
  int batchK = std::max(1, (steps + batches - 1) / batches);
  float* bSingle = batchK == 1 ? d.single : scratch<float>("dn.bsingle", (size_t)batchK * d.T * d.Ct);
  float* bG = batchK == 1 ? d.G : scratch<float>("dn.bG", (size_t)batchK * d.T * d.entries * d.Ct);
  float* bSTok = scratch<float>("dn.bsTok", (size_t)batchK * d.T * d.Ct);
  for (int i = 0; i < steps; ++i) {
    if (i % batchK == 0) {
      int k = std::min(batchK, steps - i);
      conditioningSteps(d, levels + i * 4, k, bSingle, bG, bSTok);
    }
    int bi = i % batchK;
    double sigma = sched[i], next = sched[i + 1], gamma = s.churn && next > s.gammaMin ? s.gamma0 : 0;
    double tot = 0, c[3] = {0, 0, 0};
    for (int a = 0; a < A; ++a) { tot += mask[a]; for (int k = 0; k < 3; ++k) c[k] += mask[a] * x[a * 3 + k]; }
    for (int k = 0; k < 3; ++k) c[k] /= std::max(tot, 1.0);
    double q[4]; for (auto& v : q) v = N(rng);
    double qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    double qr = q[0] / qn, qi = q[1] / qn, qj = q[2] / qn, qk = q[3] / qn;
    double R[9] = {1 - 2 * (qj * qj + qk * qk), 2 * (qi * qj - qk * qr), 2 * (qi * qk + qj * qr),
                   2 * (qi * qj + qk * qr), 1 - 2 * (qi * qi + qk * qk), 2 * (qj * qk - qi * qr),
                   2 * (qi * qk - qj * qr), 2 * (qj * qk + qi * qr), 1 - 2 * (qi * qi + qj * qj)};
    double sh[3] = {N(rng), N(rng), N(rng)};
    for (int a = 0; a < A; ++a) {
      double p[3] = {x[a * 3] - c[0], x[a * 3 + 1] - c[1], x[a * 3 + 2] - c[2]};
      for (int k = 0; k < 3; ++k) x[a * 3 + k] = (float)(p[0] * R[0 * 3 + k] + p[1] * R[1 * 3 + k] + p[2] * R[2 * 3 + k] + sh[k]);
    }
    double t = sigma * (1 + gamma);
    double eps = s.noiseScale * sqrt(std::max(t * t - sigma * sigma, 0.0));
    for (auto& v : x) v += (float)(eps * N(rng));
    upload(dx, x.data(), (size_t)A * 12);
    denoiseStep(d, dx, dd, levels + i * 4,
                bG + (size_t)bi * d.T * d.entries * d.Ct,
                bSTok + (size_t)bi * d.T * d.Ct);
    if (onStep) onStep(dd, i + 1, steps);
    download(xd.data(), dd, (size_t)A * 12);
    rigidAlign(x, xd, mask, A);
    // (Euler's x + f (x - d) is the exponential step's r^S x + (1 - r^S) d at f = r^S - 1)
    double f = s.exponential ? pow(next / t, s.stepScale) - 1 : s.stepScale * (next - t) / t;
    for (int k = 0; k < A * 3; ++k) x[k] = (float)(x[k] + f * (x[k] - xd[k]));
  }
  release(dx); release(dd);
  if (stepsRun) *stepsRun = steps;
  return x;
}
