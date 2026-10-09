// The confidence head: the trunk's pair and single re-embedded with the sample's pseudo-beta distances, four
// pairformer blocks, then pLDDT, PAE, PDE and the TM terms (cuda/af3/src/confidence.cuh is the reading).
#include "af3.h"
#include <cmath>

ConfidenceOut confidenceHead(const Trunk& t, const float* pseudoBeta) {
  const std::string P = "confidence";
  int n = t.n, C = metaI(P + ".pairChannels"), Cs = metaI(P + ".singleChannels"), F = metaI(P + ".targetFeatWidth");
  int dense = metaI("batch.dense");
  size_t pairs = (size_t)n * n;
  if (flag("trunk.dialect.reembedConfidencePair")) die("boltz2's re-embedded confidence pair is not in the native port yet");
  if (flag("trunk.dialect.confidenceGlobalNorm")) die("rf3's confidence normalisation is not in the native port yet");
  if (flag("trunk.dialect.chaiConfidence")) die("chai-1's confidence head is not in the native port yet");
  bool caDgram = flag("trunk.dialect.confidenceCaDgram");
  float* pair = scratch<float>("conf.pair", pairs * C);
  float* single = scratch<float>("conf.single", (size_t)n * Cs);
  copy(pair, t.pair, pairs * C * 4);
  copy(single, t.single, (size_t)n * Cs * 4);
  float* left = scratch<float>("conf.left", (size_t)n * C); float* right = scratch<float>("conf.right", (size_t)n * C);
  lin(t.targetFeat, P + ".leftTargetFeatProject", left, n, F, C);
  lin(t.targetFeat, P + ".rightTargetFeatProject", right, n, F, C);
  int bins = (int)(lenW(P + ".distogramFeatProject") / C);
  if (bins != (caDgram ? 40 : 39)) die("the confidence distogram has %d bins", bins);
  int* binOf = scratch<int>("conf.bin", pairs); float* sqOf = scratch<float>("conf.sq", pairs);
  run1d("af3_conf_bin", pairs, ConfBinArgs{pseudoBeta, binOf, sqOf, (uint)n, (uint)bins, 3.25f, 50.75f, caDgram ? 1u : 0u, 0});
  run1d("af3_conf_pair_init", pairs * C, ConfPairInitArgs{pair, left, right, binOf, sqOf, t.pairMask, W(P + ".distogramFeatProject"),
                                                          Wopt(P + ".distanceFeatProject"), (uint)n, (uint)C, caDgram ? 1u : 0u, 0});
  if (hasW(P + ".inputSingleNormScale")) {     // the trunk single clamped to +-512 and LayerNormed first (protenix2)
    run1d("af3_clamp", (size_t)n * Cs, ClampArgs{single, (u64)n * Cs, 512.f, 0});
    float* sn = scratch<float>("conf.singleNorm", (size_t)n * Cs);
    ln(single, sn, n, Cs, P + ".inputSingleNormScale", P + ".inputSingleNormOffset");
    copy(single, sn, (size_t)n * Cs * 4);
  }
  int nb = 0; while (M.has(P + ".blocks." + num(nb) + ".singleChannels")) ++nb;
  for (int k = 0; k < nb; ++k) pairformerBlock(pair, single, t.masks, n, C, Cs, P + ".blocks." + num(k));
  // the error bins: 64 up to 31 A, the last one step past the second-to-last
  const int NB = 64; double step = 31.0 / (NB - 2);
  std::vector<float> centres(NB);
  for (int b = 0; b < NB - 1; ++b) centres[b] = (float)(b * step + step / 2);
  centres[NB - 1] = (float)(centres[NB - 2] + step);
  float* dCentres = scratch<float>("conf.centres", NB);
  upload(dCentres, centres.data(), NB * 4);
  half* lnPair = scratch<half>("conf.ln", pairs * C);
  float* logits = scratch<float>("conf.logits", pairs * NB);
  float* inter = hasW(P + ".interHalfDistanceLogits") || hasW(P + ".paeInterLogits") ? scratch<float>("conf.inter", pairs * NB) : nullptr;
  const int* asym = M.i("batch.asymId");
  // a head's logits: LN(x) W (a head LayerNorm the bundle does not carry is none), with boltz2's inter-chain half
  auto head = [&](const float* x, const std::string& norm, const std::string& w, const std::string& interW) {
    if (hasW(P + "." + norm + "Scale")) ln(x, lnPair, pairs, C, P + "." + norm + "Scale", P + "." + norm + "Offset");
    else toHalf(x, lnPair, pairs * C);
    lin(lnPair, P + "." + w, logits, pairs, C, NB);
    if (!hasW(P + "." + interW)) return;
    lin(lnPair, P + "." + interW, inter, pairs, C, NB);
    run1d("af3_inter_chain", pairs * NB, InterChainArgs{logits, inter, asym, (uint)n, (uint)NB});
  };
  float* pde = scratch<float>("conf.pde", pairs); float* pae = scratch<float>("conf.pae", pairs);
  bool preSym = flag("trunk.dialect.preSymmetrisedPde");
  if (hasW(P + ".interHalfDistanceLogits") && !preSym) die("split PDE heads need the pre-symmetrised PDE");
  if (preSym) {      // symmetrised BEFORE the projection: LN(z + z^T) W (protenix2, boltz2); AF3 adds the transpose after
    float* sym = scratch<float>("conf.sym", pairs * C);
    run1d("af3_symmetrise", pairs * C, SymmetriseArgs{pair, sym, (uint)n, (uint)C, 1.f, 0});
    head(sym, "logitsLn", "leftHalfDistanceLogits", "interHalfDistanceLogits");
  } else head(pair, "logitsLn", "leftHalfDistanceLogits", "interHalfDistanceLogits");
  run1d("af3_expectation", pairs, ExpectationArgs{logits, pde, t.pairMask, dCentres, pairs, (uint)NB, preSym ? 0u : (uint)n, 1.f, 0});
  head(pair, "paeLogitsLn", "paeLogits", "paeInterLogits");
  run1d("af3_expectation", pairs, ExpectationArgs{logits, pae, t.pairMask, dCentres, pairs, (uint)NB, 0, 1.f, 0});
  ConfidenceOut out;
  {   // pTM and ipTM off the PAE logits: per pair the expected TM term, then the best anchor's mean over what it selects
    std::vector<float> seq = download(t.seqMask, n);
    const int* asymH = M.hostI("batch.asymId");
    int real = 0; for (float v : seq) real += v > 0;
    double d0 = 1.24 * std::cbrt(std::max(real, 19) - 15.0) - 1.8;
    std::vector<float> perBin(NB);
    for (int b = 0; b < NB; ++b) perBin[b] = (float)(1 / (1 + (double)centres[b] * centres[b] / (d0 * d0)));
    float* dPerBin = scratch<float>("conf.perBin", NB);
    upload(dPerBin, perBin.data(), NB * 4);
    float* dTerm = scratch<float>("conf.tmTerm", pairs);
    run1d("af3_expectation", pairs, ExpectationArgs{logits, dTerm, nullptr, dPerBin, pairs, (uint)NB, 0, 1.f, 0});
    std::vector<float> term = download(dTerm, pairs);
    auto reduce = [&](bool interOnly) {
      double best = -1e30; bool any = false;
      for (int i = 0; i < n; ++i) {
        double tot = 0; int cnt = 0;
        for (int j = 0; j < n; ++j) {
          if (!(seq[i] > 0 && seq[j] > 0) || (interOnly && asymH[i] == asymH[j])) continue;
          tot += term[(size_t)i * n + j]; ++cnt;
        }
        if (cnt) { any = true; best = std::max(best, tot / cnt); }
      }
      return any ? best : NAN;
    };
    out.ptm = reduce(false); out.iptm = reduce(true);
    out.tmTerm = std::move(term);
  }
  const int PB = 50;
  std::vector<float> pc(PB);
  for (int b = 0; b < PB; ++b) pc[b] = 0.5f / PB + (float)b / PB;
  float* dpc = scratch<float>("conf.plddtCentres", PB);
  upload(dpc, pc.data(), PB * 4);
  half* sln = scratch<half>("conf.sln", (size_t)n * Cs);
  if (hasW(P + ".plddtLnScale")) ln(single, sln, n, Cs, P + ".plddtLnScale", P + ".plddtLnOffset");
  else toHalf(single, sln, (size_t)n * Cs);
  float* pl = scratch<float>("conf.plddtLogits", (size_t)n * dense * PB);
  lin(sln, P + ".plddtLogits", pl, n, Cs, dense * PB);
  float* plddt = scratch<float>("conf.plddt", (size_t)n * dense);
  run1d("af3_expectation", (size_t)n * dense, ExpectationArgs{pl, plddt, nullptr, dpc, (u64)n * dense, (uint)PB, 0, 100.f, 0});
  out.plddt = download(plddt, (size_t)n * dense);
  out.pae = download(pae, pairs);
  out.pde = download(pde, pairs);
  const float* mask = M.hostF("batch.refMask");
  double sum = 0, count = 0;
  for (size_t i = 0; i < out.plddt.size(); ++i) if (mask[i]) { sum += out.plddt[i]; count += 1; }
  out.meanPlddt = sum / std::max(count, 1.0);
  releaseScratch({"conf."});
  return out;
}
