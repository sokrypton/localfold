// ESMFold2's sliding-window atom transformer with 3D RoPE (cuda/ef2/src/atoms.cuh is the reading): the inputs
// embedder, and the stacks the diffusion module's atom encoder and decoder reuse.
//
//   c0 = LN(atomFeatures @ linear);  3 adaLN-Zero blocks conditioned on c0:
//     mod = silu(c0) @ adaln -> shift_a scale_a gate_a shift_f scale_f gate_f
//     q += gate_a * SWA(rms(q) (1 + scale_a) + shift_a);  q += gate_f * swiglu(rms(q) (1 + scale_f) + shift_f)
//   SWA: qkv; q, k rms-normed per head then rotated; q, k, v narrowed to bfloat16; keys within halfWindow in rank
//   among the valid atoms; out = (ctx * live * sigmoid(input @ attnGate)) @ attnOut
#include "ef2.h"

bool ATOM_BF16 = true;
int INPUTS_HALF_WINDOW = 64;
static constexpr int ATOM_FEATURES = 389;

Atoms prepareAtoms(int A, const std::string& prefix) {
  Atoms at{};
  int C = (int)M.meta("meta/atomChannels");
  at.ctx.A = A; at.ctx.C = C; at.ctx.heads = (int)M.meta("meta/atomHeads");
  at.ctx.hidden = dimOf("f/" + prefix + "/blocks/0/ffnDown", 0);
  at.ctx.mask = In("atom_mask");
  {   // the valid atoms in order, and each token's (the windowed attention and the pooling)
    const float* mask = M.hostF("atom_mask");
    const int* toToken = M.hostI("atom_to_token");
    int T = (int)M.meta("meta/tokens");
    std::vector<std::vector<int>> byToken(T);
    std::vector<int> valid;
    for (int a = 0; a < A; ++a) if (mask[a] != 0.f) { valid.push_back(a); byToken.at(toToken[a]).push_back(a); }
    std::vector<int> start{0}, atoms;
    for (auto& v : byToken) { atoms.insert(atoms.end(), v.begin(), v.end()); start.push_back((int)atoms.size()); }
    at.ctx.nValid = (int)valid.size();
    at.ctx.valid = uploadNew(valid.data(), valid.size());
    at.ctx.tokenStart = uploadNew(start.data(), start.size());
    at.ctx.tokenAtoms = uploadNew(atoms.data(), atoms.size());
  }
  float* cosT = allocT<float>((size_t)A * 16); float* sinT = allocT<float>((size_t)A * 16);
  run1d("ef2_rope_table", (size_t)A * 16, RopeTableArgs{In("ref_pos"), Ii("ref_space_uid"), cosT, sinT, (uint)A, 0});
  at.ctx.cosT = cosT; at.ctx.sinT = sinT;
  float* feat = scratch<float>("atom.features", (size_t)A * ATOM_FEATURES);
  run1d("ef2_atom_features", (size_t)A * ATOM_FEATURES,
        AtomFeaturesArgs{In("ref_pos"), In("ref_charge"), at.ctx.mask, Ii("ref_element"), Ii("ref_atom_name_chars"), feat, (uint)A, 0});
  at.c0 = allocT<float>((size_t)A * C);
  lin(feat, "f/" + prefix + "/linear", at.c0, A, ATOM_FEATURES, C);
  layerNorm(at.c0, at.c0, A, C, F(prefix + "/norm/scale"), F(prefix + "/norm/offset"));
  return at;
}
void freeAtoms(Atoms& at) {
  for (const void* p : {(const void*)at.c0, (const void*)at.ctx.cosT, (const void*)at.ctx.sinT, (const void*)at.ctx.valid,
                        (const void*)at.ctx.tokenStart, (const void*)at.ctx.tokenAtoms})
    release(p);
  at = Atoms{};
}

static Grid rowsOf8(size_t rows) { return grid1d((rows + 7) / 8, 1); }
static void swaBlock(const AtomCtx& a, float* x, const float* sc, const half* sch, const std::string& B, int halfWindow) {
  size_t A = a.A; int C = a.C;
  float* mod = scratch<float>("atom.mod", A * 6 * C);
  float* qkv = scratch<float>("atom.qkv", A * 3 * C);
  float* gate = scratch<float>("atom.gate", A * C);
  float* ctx = scratch<float>("atom.ctx", A * C);
  float* att = scratch<float>("atom.att", A * C);
  if (HALF_GEMM) {
    lin(sch, B + "adaln", mod, A, C, 6 * C);
    half* xmh = scratch<half>("atom.xmh", A * C);
    run("ef2_rms_modulate", rowsOf8(A), 256, RmsModArgs{x, mod, nullptr, xmh, (uint)A, (uint)C, 0, 1});
    lin(xmh, B + "qkv", qkv, A, C, 3 * C);
    lin(xmh, B + "attnGate", gate, A, C, C);
    run("ef2_qkv_prepare", rowsOf8(A * a.heads), 256, QkvPrepArgs{qkv, a.cosT, a.sinT, (uint)A, (uint)C, (uint)a.heads, ATOM_BF16 ? 1u : 0u});
    fill(ctx, 0, A * C * 4);
    if (a.nValid)
      run("ef2_swa", Grid{(uint32_t)((a.nValid + 15) / 16), (uint32_t)a.heads, 1}, 128,
          SwaArgs{qkv, a.valid, ctx, (uint)a.nValid, (uint)C, (uint)std::min(halfWindow, (int)A), 1.f / sqrtf(32.f)});
    half* ctxh = scratch<half>("atom.ctxh", A * C);
    run1d("ef2_gate_live", A * C, GateLiveArgs{ctx, gate, a.mask, ctxh, (uint)A, (uint)C});
    {
      Gemm G{}; G.X = ctxh; G.tx = F16; G.W = M.h(B + "attnOut"); G.tw = F16; G.Y = x; G.ty = F32;
      G.rows = A; G.in = C; G.out = C; G.label = "atom attnOut";
      if (!gemmGatedRows(G, mod + 2 * C, 6 * C, nullptr)) {
        G.Y = att; gemm(G);
        run1d("ef2_gated_add", (A * C) >> 2, GatedAddArgs{x, mod, att, (uint)A, (uint)C, 2, 0});
      }
    }
    run("ef2_rms_modulate", rowsOf8(A), 256, RmsModArgs{x, mod, nullptr, xmh, (uint)A, (uint)C, 3, 4});
    half* gh = scratch<half>("atom.gh", A * a.hidden);
    gemmSwiglu(xmh, swigluPairs(B + "ffnUp", M.h(B + "ffnUp"), C, a.hidden), gh, A, C, a.hidden);
    {
      Gemm G{}; G.X = gh; G.tx = F16; G.W = M.h(B + "ffnDown"); G.tw = F16; G.Y = x; G.ty = F32;
      G.rows = A; G.in = a.hidden; G.out = C; G.label = "atom ffnDown";
      if (!gemmGatedRows(G, mod + 5 * C, 6 * C, nullptr)) {
        G.Y = att; gemm(G);
        run1d("ef2_gated_add", (A * C) >> 2, GatedAddArgs{x, mod, att, (uint)A, (uint)C, 5, 0});
      }
    }
    return;
  }
  lin(sc, B + "adaln", mod, A, C, 6 * C);
  float* xm = scratch<float>("atom.xm", A * C);
  run("ef2_rms_modulate", rowsOf8(A), 256, RmsModArgs{x, mod, xm, nullptr, (uint)A, (uint)C, 0, 1});
  lin(xm, B + "qkv", qkv, A, C, 3 * C);
  run("ef2_qkv_prepare", rowsOf8(A * a.heads), 256, QkvPrepArgs{qkv, a.cosT, a.sinT, (uint)A, (uint)C, (uint)a.heads, ATOM_BF16 ? 1u : 0u});
  fill(ctx, 0, A * C * 4);
  if (a.nValid)
    run("ef2_swa", Grid{(uint32_t)((a.nValid + 15) / 16), (uint32_t)a.heads, 1}, 128,
        SwaArgs{qkv, a.valid, ctx, (uint)a.nValid, (uint)C, (uint)std::min(halfWindow, (int)A), 1.f / sqrtf(32.f)});
  lin(xm, B + "attnGate", gate, A, C, C);
  run1d("ef2_gate_live", A * C, GateLiveArgs{ctx, gate, a.mask, nullptr, (uint)A, (uint)C});
  lin(ctx, B + "attnOut", att, A, C, C);
  run1d("ef2_gated_add", (A * C) >> 2, GatedAddArgs{x, mod, att, (uint)A, (uint)C, 2, 0});
  run("ef2_rms_modulate", rowsOf8(A), 256, RmsModArgs{x, mod, xm, nullptr, (uint)A, (uint)C, 3, 4});
  float* h = scratch<float>("atom.h", A * 2 * a.hidden); float* g = scratch<float>("atom.g", A * a.hidden);
  lin(xm, B + "ffnUp", h, A, C, 2 * a.hidden);
  run1d("ef2_swiglu", A * a.hidden, SwigluArgs{h, g, (uint)A, (uint)a.hidden});
  lin(g, B + "ffnDown", att, A, a.hidden, C);
  run1d("ef2_gated_add", (A * C) >> 2, GatedAddArgs{x, mod, att, (uint)A, (uint)C, 5, 0});
}
void swaStack(const AtomCtx& a, float* x, const float* cond, const std::string& prefix, int blocks, int halfWindow) {
  size_t n = (size_t)a.A * a.C;
  float* sc = HALF_GEMM ? nullptr : scratch<float>("atom.silu", n);
  half* sch = HALF_GEMM ? scratch<half>("atom.siluh", n) : nullptr;
  run1d("ef2_silu", n, SiluArgs{cond, sc, sch, n});
  for (int b = 0; b < blocks; ++b) swaBlock(a, x, sc, sch, "f/" + prefix + "/blocks/" + std::to_string(b) + "/", halfWindow);
}

// the inputs embedder: s_inputs [T, 451] = [pool(relu(stack(c0) @ toToken)) | aatype | profile | deletion mean]
void inputsEmbedder(int T, int A, float* sInputs, int sWidth) {
  Atoms at = prepareAtoms(A, "atom");
  int C = at.ctx.C, Ct = dimOf("f/atom/toToken", 1), K = (int)M.meta("meta/classes");
  float* x = scratch<float>("embed.x", (size_t)A * C);
  copy(x, at.c0, (size_t)A * C * 4);
  swaStack(at.ctx, x, at.c0, "atom", (int)M.meta("meta/atomBlocks"), INPUTS_HALF_WINDOW);
  float* tok = scratch<float>("embed.tok", (size_t)A * Ct);
  Gemm g{}; g.X = x; g.W = M.h("f/atom/toToken"); g.tw = F16; g.half = true; g.Y = tok; g.rows = A; g.in = C; g.out = Ct; g.relu = true;
  gemm(g);
  run1d("ef2_scatter_mean", (size_t)T * Ct,
        ScatterMeanArgs{tok, at.ctx.tokenStart, at.ctx.tokenAtoms, at.ctx.mask, sInputs, (uint)T, (uint)Ct, (uint)sWidth, 0});
  // the alignment's profile and mean deletion only where the checkpoint reads them (the experimental tier zeroes both)
  const bool msaFeatures = M.meta("meta/msaFeatures", 0) > 0;
  run1d("ef2_tail_inputs", (size_t)T * (2 * K + 1),
        TailInputsArgs{In("aatype"), msaFeatures ? In("profile") : nullptr, msaFeatures ? In("deletion_mean") : nullptr, sInputs,
                       (uint)T, (uint)Ct, (uint)K, (uint)sWidth});
  sync();
  freeAtoms(at);
}
