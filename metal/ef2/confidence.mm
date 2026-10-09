// Synthyra's ESMFold2 confidence head (cuda/ef2/src/confidence.cuh is the reading):
//
//   s = LN(s_inputs);  z = LN(z_trunk) + rel_pos + bonds + s_to_z(s)_i + s_to_z_T(s)_j + prod_out(in1(s)_i * in2(s)_j)
//   z += distance_embed[#(|x_rep_i - x_rep_j| > boundaries)];  z = z + trunk_4_blocks(z)
//   single = (softmax_j(z @ attn) z) @ out;  pLDDT per atom = E[bin] of LN(single[token]) @ plddtWeight[slot]
//   PAE = E[bin] over [0, 32] of z @ pae;  pTM / ipTM from the PAE logits
// Its own projections stay float32 and multiply in float (their f16 rounding cost 5CAJ's pLDDT 16 points on the
// translated port); its four pair blocks are the trunk's, in half.
#include "ef2.h"
#include <cmath>

Confidence confidenceHead(int T, int A, const float* zTrunk, const float* sInputs, int Si, const float* xDevice) {
  struct Restore { bool was = HALF_GEMM; ~Restore() { HALF_GEMM = was; } } restore;
  HALF_GEMM = false;
  int C = dimOf("f/confidence/sToZ", 1), Cs = dimOf("f/confidence/poolingOutput", 1);
  size_t P = (size_t)T * T;
  float* s = scratch<float>("cf.s", (size_t)T * Si);
  layerNorm(sInputs, s, T, Si, F("confidence/sInputsNorm/scale"), F("confidence/sInputsNorm/offset"));
  float* z = allocT<float>(P * C);
  layerNorm(zTrunk, z, P, C, F("confidence/zNorm/scale"), F("confidence/zNorm/offset"));
  float* r = scratch<float>("cf.r", (size_t)T * C); float* c = scratch<float>("cf.c", (size_t)T * C);
  float* l = scratch<float>("cf.l", (size_t)T * C); float* rr = scratch<float>("cf.rr", (size_t)T * C);
  lin(s, "f/confidence/sToZ", r, T, Si, C);
  lin(s, "f/confidence/sToZTranspose", c, T, Si, C);
  lin(s, "f/confidence/sToZProdIn1", l, T, Si, C);
  lin(s, "f/confidence/sToZProdIn2", rr, T, Si, C);
  run("ef2_conf_z", grid1d(P, 1), 256, ConfZArgs{z, relIdx(), In("token_bonds"), F("featuriser/tokenBonds"), r, c, (uint)T, (uint)C});
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));
  half* prod = scratch<half>("cf.prod", chunk * C);
  for (size_t p0 = 0; p0 < P; p0 += chunk) {
    size_t n = std::min(chunk, P - p0);
    run1d("ef2_outer", n * C, OuterArgs{l, rr, prod, p0, n, (uint)T, (uint)C});
    Gemm g{}; g.X = prod; g.tx = F16; g.W = F("confidence/sToZProdOut"); g.tw = F32; g.Y = z + p0 * C; g.beta = 1.f;
    g.rows = n; g.in = C; g.out = C; g.label = "confidence prodOut";
    gemm(g);
  }
  run("ef2_distance_embed", grid1d((P + 7) / 8, 1), 256,
      DistEmbedArgs{z, xDevice, Ii("distogram_atom_idx"), F("confidence/boundaries"), F("confidence/distanceEmbedding"),
                    (uint)M.len("f/confidence/boundaries"), (uint)T, (uint)C, 0});
  // z + trunk(z)
  float* mask = scratch<float>("trunk.mask", P);
  run1d("ef2_fill", P, FillFArgs{mask, P, 1.f, 0});
  float* stack = allocT<float>(P * C);
  copy(stack, z, P * C * 4);
  for (int b = 0; M.has("f/confidence/blocks/" + std::to_string(b) + "/pairTransition/transition1"); ++b)
    trunkBlock(stack, mask, T, C, "confidence/blocks", b);
  add(z, stack, P * C);
  release(stack);
  releaseScratch({"ftri.", "ftr."});
  // single by row-attention pooling
  float* score = scratch<float>("cf.score", P);
  lin(z, "f/confidence/poolingAttention", score, P, C, 1);
  float* pooled = scratch<float>("cf.pooled", (size_t)T * C); float* single = scratch<float>("cf.single", (size_t)T * Cs);
  run("ef2_row_pool", Grid{(uint32_t)T, 1, 1}, 256, RowPoolArgs{z, score, pooled, (uint)T, (uint)C});
  lin(pooled, "f/confidence/poolingOutput", single, T, C, Cs);
  // pLDDT per atom (a table a slot: the atom's position within its token, clamped)
  int bins = dimOf("f/confidence/plddtWeight", 2), slots = dimOf("f/confidence/plddtWeight", 0);
  std::vector<int> a2t(M.hostI("atom_to_token"), M.hostI("atom_to_token") + A), slot(A);
  for (int a = 0, k = 0; a < A; ++a) { if (a > 0 && a2t[a] != a2t[a - 1]) k = 0; slot[a] = std::min(k++, slots - 1); }
  int* dSlot = uploadNew(slot.data(), A);
  float* sa = scratch<float>("cf.sa", (size_t)A * Cs); float* pa = scratch<float>("cf.pa", A);
  run1d("ef2_gather_atoms", (size_t)A * Cs, GatherAtomsArgs{single, Ii("atom_to_token"), sa, (uint)A, (uint)Cs});
  layerNorm(sa, sa, A, Cs, F("confidence/plddtNorm/scale"), F("confidence/plddtNorm/offset"));
  run("ef2_plddt_atom", Grid{(uint32_t)A, 1, 1}, 64, PlddtArgs{sa, dSlot, F("confidence/plddtWeight"), pa, (uint)A, (uint)Cs, (uint)bins, 0});
  // PAE logits (the released heads LayerNorm the pair in front of the projection)
  int pb = dimOf("f/confidence/pae", 1);
  float* paeL = scratch<float>("cf.pae", P * pb);
  if (M.has("f/confidence/paeNorm/scale")) {
    size_t per = std::max<size_t>(1, std::min(P, ((size_t)64 << 20) / (4 * (size_t)C)));
    float* zn = scratch<float>("cf.paeIn", per * C);
    for (size_t p0 = 0; p0 < P; p0 += per) {
      size_t n = std::min(per, P - p0);
      layerNorm(z + p0 * C, zn, n, C, F("confidence/paeNorm/scale"), F("confidence/paeNorm/offset"));
      lin(zn, "f/confidence/pae", paeL + p0 * pb, n, C, pb);
    }
  } else lin(z, "f/confidence/pae", paeL, P, C, pb);
  Confidence out;
  out.plddtAtom = download(pa, A);
  const float* mask_ = M.hostF("atom_mask");
  std::vector<int> asym(M.hostI("asym_id"), M.hostI("asym_id") + T);
  out.plddtToken.assign(T, 0.f);
  std::vector<float> cnt(T, 0.f);
  double wsum = 0, wtot = 0;
  for (int a = 0; a < A; ++a) {
    out.plddtToken[a2t[a]] += out.plddtAtom[a] * mask_[a]; cnt[a2t[a]] += mask_[a];
    wsum += out.plddtAtom[a] * mask_[a]; wtot += mask_[a];
  }
  for (int t = 0; t < T; ++t) out.plddtToken[t] /= std::max(cnt[t], 1e-6f);
  out.meanPlddt = wsum / (wtot + 1e-8);
  // PAE and pTM / ipTM on the device
  double width = 32.0 / pb, d0 = 1.24 * cbrt(std::max(T, 19) - 15.0) - 1.8;
  float* paeD = scratch<float>("cf.paeOut", P); float* tmD = scratch<float>("cf.tm", P);
  float* rowsD = scratch<float>("cf.tmRows", (size_t)2 * T);
  run1d("ef2_pae", P, PaeArgs{paeL, paeD, tmD, P, (uint)pb, (float)width, (float)d0, 0});
  run("ef2_tm_rows", Grid{(uint32_t)T, 1, 1}, 256, TmRowsArgs{tmD, Ii("asym_id"), rowsD, (uint)T, 0});
  out.pae = download(paeD, P);
  std::vector<float> rows = download(rowsD, (size_t)2 * T);
  double ptm = -1e30, iptm = -1e30;
  for (int i = 0; i < T; ++i) { ptm = std::max(ptm, (double)rows[2 * i]); iptm = std::max(iptm, (double)rows[2 * i + 1]); }
  out.ptm = ptm; out.iptm = iptm;
  release(z); release(dSlot);
  return out;
}
