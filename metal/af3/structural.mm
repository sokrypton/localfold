// OpenDDE's structural tokens (cuda/af3/src/structural.cuh is the reading). The trunk runs on residues; after it each
// standard residue becomes a backbone and a sidechain token (a ligand atom stays one), the expander maps the trunk's
// single and pair onto those, a four-block refiner runs on them, and the diffusion and OpenDDE's own confidence head run
// in that token space. The layout, the structural batch (sbatch.*) and the pair features come from the featuriser.
#include "af3.h"
#include <cmath>

static const int ROLES = 7;

// The expander and the refiner. Reads the residue batch (call before the structural batch is swapped in).
Structural expandStructural(const Trunk& t) {
  const std::string E = "expander.";
  Structural s{};
  s.n = (int)M.meta("sbatch.tokens");
  int n = s.n, nRes = t.n;
  int F = metaI(E + "singleInputChannels"), Cs = metaI(E + "singleChannels"), C = metaI(E + "pairChannels");
  if (metaI(E + "roles") != ROLES) die("the expander has %d roles, not 7", metaI(E + "roles"));
  size_t pairs = (size_t)n * n;
  const int* parent = M.i("structural.parent"); const int* role = M.i("structural.role");
  // s_inputs and the single
  s.targetFeat = allocT<float>((size_t)n * F);
  run1d("af3_gather_parent", (size_t)n * F, GatherParentArgs{t.targetFeat, parent, role, W(E + "singleInputRoleEmbedding"), s.targetFeat,
                                                            (uint)n, (uint)F});
  float* ps = scratch<float>("sx.parentSingle", (size_t)n * Cs);
  run1d("af3_gather_parent", (size_t)n * Cs, GatherParentArgs{t.single, parent, role, nullptr, ps, (uint)n, (uint)Cs});
  half* lnS = scratch<half>("sx.ln", (size_t)n * Cs);
  ln(ps, lnS, n, Cs, E + "singleSplitNormScale", E + "singleSplitNormOffset");
  float* hidden = scratch<float>("sx.hidden", (size_t)n * 2 * Cs);
  lin(lnS, E + "singleSplit1", hidden, n, Cs, 2 * Cs);
  run1d("af3_silu", (size_t)n * 2 * Cs, SiluInPlaceArgs{hidden, (u64)n * 2 * Cs});
  float* proj = scratch<float>("sx.proj", (size_t)n * Cs);
  lin(hidden, E + "singleSplit2", proj, n, 2 * Cs, Cs);
  s.single = allocT<float>((size_t)n * Cs);
  run1d("af3_single_struct", (size_t)n * Cs, SingleStructArgs{s.single, ps, proj, role, W(E + "singleRoleEmbedding"), (uint)n, (uint)Cs});
  // the pair: each structural pair through the matrix its two roles name (49 of them), one GEMM per matrix over the
  // pairs sorted by it
  const int* hr = M.hostI("structural.role");
  std::vector<int> order(pairs), start(ROLES * ROLES + 1, 0);
  for (size_t ij = 0; ij < pairs; ++ij) ++start[hr[ij / n] * ROLES + hr[ij % n] + 1];
  for (int k = 0; k < ROLES * ROLES; ++k) start[k + 1] += start[k];
  { std::vector<int> at(start.begin(), start.end() - 1);
    for (size_t ij = 0; ij < pairs; ++ij) order[at[hr[ij / n] * ROLES + hr[ij % n]]++] = (int)ij; }
  int* dOrder = uploadNew(order.data(), pairs);
  if (lenW(E + "pairBlockProj") != (size_t)ROLES * ROLES * C * C) die("pairBlockProj is not 49 x C x C");
  s.pair = allocT<float>(pairs * C);
  half* gathered = scratch<half>("sx.gathered", pairs * C);
  float* projected = scratch<float>("sx.projected", pairs * C);
  run1d("af3_gather_pair_sorted", pairs * C, GatherPairSortedArgs{t.pair, dOrder, parent, gathered, pairs, (uint)n, (uint)nRes, (uint)C, 0});
  const half* Wp = Wh(E + "pairBlockProj");
  for (int k = 0; k < ROLES * ROLES; ++k) {
    size_t lo = start[k], hi = start[k + 1];
    if (lo < hi) linW(gathered + lo * C, Wp + (size_t)k * C * C, projected + lo * C, hi - lo, C, C, 0.f, nullptr, 1.f, "expander pair");
  }
  const int *sp = M.i("structural.sameParent"), *tw = M.i("structural.twin"), *pv = M.i("structural.prevBackbone"),
            *nx = M.i("structural.nextBackbone"), *ty = M.i("structural.rolePairType");
  run1d("af3_scatter_pair", pairs * C,
        ScatterPairArgs{s.pair, t.pair, parent, projected, dOrder, sp, tw, pv, nx, ty, W(E + "sameParentEmbedding"),
                        W(E + "sameResidueTwinEmbedding"), W(E + "prevBbChainEmbedding"), W(E + "nextBbChainEmbedding"),
                        W(E + "rolePairTypeEmbedding"), pairs, (uint)n, (uint)nRes, (uint)C, 0});
  s.bias = allocT<float>(pairs);
  run1d("af3_attn_bias", pairs, AttnBiasArgs{s.bias, sp, tw, pv, nx, ty, W(E + "attnBiasSameParent"), W(E + "attnBiasSameResidueTwin"),
                                             W(E + "attnBiasPrevBbChain"), W(E + "attnBiasNextBbChain"), W(E + "attnBiasRolePairType"), pairs});
  release(dOrder);
  releaseScratch({"sx."});
  // the masks, from the structural batch's own seq mask
  std::vector<float> seq(M.hostF("sbatch.seqMask"), M.hostF("sbatch.seqMask") + n), pm(pairs);
  bool ones = true;
  for (int i = 0; i < n; ++i) { ones &= seq[i] > 0; for (int j = 0; j < n; ++j) pm[(size_t)i * n + j] = seq[i] * seq[j]; }
  s.seqMask = uploadNew(seq.data(), n); s.pairMask = uploadNew(pm.data(), pairs);
  s.masks = {s.pairMask, s.seqMask, ones};
  // the refiner: four pairformer blocks, each adding the expander's attention bias
  for (int k = 0; M.has("refiner.blocks." + num(k) + ".singleChannels"); ++k)
    pairformerBlock(s.pair, s.single, s.masks, n, C, Cs, "refiner.blocks." + num(k), s.bias);
  return s;
}
void freeStructural(Structural& s) {
  for (const void* p : {(const void*)s.single, (const void*)s.pair, (const void*)s.targetFeat, (const void*)s.bias, (const void*)s.seqMask,
                        (const void*)s.pairMask})
    if (p) release(p);
  s = Structural{};
}

// ---------------------------------------------------------------- OpenDDE's confidence head
// the structural tokens' coordinates, refined pair and single, s_inputs: pLDDT per structural atom slot, PAE, PDE and
// the TM term per structural pair (d0 from `tmTokens`, the residue count pTM is reported over)
DdeConfidence ddeConfidence(const Structural& st, const float* coords, int dense, int tmTokens) {
  const std::string P = "ddeConfidence.";
  int n = st.n, C = metaI(P + "pairChannels"), Cs = metaI(P + "singleChannels"), F = metaI(P + "singleInputChannels");
  int bins = metaI(P + "distanceBins"), paeBins = metaI(P + "paeBins"), pdeBins = metaI(P + "pdeBins");
  int slots = metaI(P + "denseSlots"), PB = metaI(P + "plddtBins");
  if (slots != dense) die("plddt_weight has %d slots, the batch %d", slots, dense);
  size_t pairs = (size_t)n * n;
  float* single = scratch<float>("ddc.single", (size_t)n * Cs);
  copy(single, st.single, (size_t)n * Cs * 4);
  run1d("af3_clamp", (size_t)n * Cs, ClampArgs{single, (u64)n * Cs, 512.f, 0});
  float* sn = scratch<float>("ddc.singleNorm", (size_t)n * Cs);
  ln(single, sn, n, Cs, P + "inputStrunkLnScale", P + "inputStrunkLnOffset");
  float* pair = scratch<float>("ddc.pair", pairs * C);
  copy(pair, st.pair, pairs * C * 4);
  float* s1 = scratch<float>("ddc.s1", (size_t)n * C); float* s2 = scratch<float>("ddc.s2", (size_t)n * C);
  lin(st.targetFeat, P + "s1", s1, n, F, C);
  lin(st.targetFeat, P + "s2", s2, n, F, C);
  run1d("af3_dde_pair_init", pairs * C, DdePairInitArgs{pair, s1, s2, coords, W(P + "distance"), W(P + "distanceRaw"), (uint)n, (uint)C,
                                                        (uint)bins, 0});
  for (int k = 0; M.has(P + "blocks." + num(k) + ".singleChannels"); ++k)
    pairformerBlock(pair, sn, st.masks, n, C, Cs, P + "blocks." + num(k), st.bias);
  // PAE from the pair, PDE from the symmetrised pair; 64 bins over [0, 32]
  int NB = std::max(paeBins, pdeBins);
  half* lnP = scratch<half>("ddc.ln", pairs * C);
  float* sym = scratch<float>("ddc.sym", pairs * C);
  float* logits = scratch<float>("ddc.logits", pairs * NB);
  float* pde = scratch<float>("ddc.pde", pairs); float* pae = scratch<float>("ddc.pae", pairs); float* tm = scratch<float>("ddc.tm", pairs);
  auto centresFor = [](int nb) { std::vector<float> c(nb); for (int b = 0; b < nb; ++b) c[b] = 32.f / nb * (b + 0.5f); return c; };
  std::vector<float> pdeC = centresFor(pdeBins), paeC = centresFor(paeBins);
  double d0 = 1.24 * std::cbrt(std::max(tmTokens, 19) - 15.0) - 1.8;
  std::vector<float> perBin(paeBins);
  for (int b = 0; b < paeBins; ++b) perBin[b] = (float)(1 / (1 + (double)paeC[b] * paeC[b] / (d0 * d0)));
  float* dPde = scratch<float>("ddc.pdeC", pdeBins); upload(dPde, pdeC.data(), pdeBins * 4);
  float* dPae = scratch<float>("ddc.paeC", paeBins); upload(dPae, paeC.data(), paeBins * 4);
  float* dPer = scratch<float>("ddc.perBin", paeBins); upload(dPer, perBin.data(), paeBins * 4);
  run1d("af3_symmetrise", pairs * C, SymmetriseArgs{pair, sym, (uint)n, (uint)C, 1.f, 0});
  ln(sym, lnP, pairs, C, P + "pdeLnScale", P + "pdeLnOffset");
  lin(lnP, P + "pde", logits, pairs, C, pdeBins);
  run1d("af3_expectation", pairs, ExpectationArgs{logits, pde, nullptr, dPde, pairs, (uint)pdeBins, 0, 1.f, 0});
  ln(pair, lnP, pairs, C, P + "paeLnScale", P + "paeLnOffset");
  lin(lnP, P + "pae", logits, pairs, C, paeBins);
  run1d("af3_expectation", pairs, ExpectationArgs{logits, pae, nullptr, dPae, pairs, (uint)paeBins, 0, 1.f, 0});
  run1d("af3_expectation", pairs, ExpectationArgs{logits, tm, nullptr, dPer, pairs, (uint)paeBins, 0, 1.f, 0});
  DdeConfidence out;
  out.pde = download(pde, pairs); out.pae = download(pae, pairs); out.tmTerm = download(tm, pairs);
  // pLDDT per atom: the token's normalised single against the matrix of the atom's dense slot
  const half* wk = M.derived<half>("dde.plddtSlotMajor", (size_t)slots * Cs * PB, [&](half* o) {
    run1d("af3_slot_major", (size_t)slots * Cs * PB, SlotMajorArgs{W(P + "plddtWeight"), o, (uint)slots, (uint)Cs, (uint)PB, 0});
  });
  half* sln = scratch<half>("ddc.sln", (size_t)n * Cs);
  ln(sn, sln, n, Cs, P + "plddtLnScale", P + "plddtLnOffset");
  float* pl = scratch<float>("ddc.plddtLogits", (size_t)n * slots * PB);
  linW(sln, wk, pl, n, Cs, slots * PB, 0.f, nullptr, 1.f, "dde plddt");
  std::vector<float> pc(PB);
  for (int b = 0; b < PB; ++b) pc[b] = (b + 0.5f) / PB;
  float* dpc = scratch<float>("ddc.plddtC", PB); upload(dpc, pc.data(), PB * 4);
  float* plddt = scratch<float>("ddc.plddt", (size_t)n * slots);
  run1d("af3_expectation", (size_t)n * slots, ExpectationArgs{pl, plddt, nullptr, dpc, (u64)n * slots, (uint)PB, 0, 100.f, 0});
  out.plddt = download(plddt, (size_t)n * slots);
  releaseScratch({"ddc."});
  return out;
}
