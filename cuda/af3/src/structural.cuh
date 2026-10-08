// OpenDDE's structural tokens. The trunk runs on residues; after it each standard residue becomes a
// backbone and a sidechain token (a ligand atom stays one), the expander maps the trunk's single
// and pair onto those, a four-block refiner runs on them, and the diffusion and OpenDDE's own
// confidence head run in that token space. Transcribed from webgpu/af3/fold.js
// (expandToStructuralTokens), cpu/af3/structure/structural-expander.js and
// webgpu/af3/confidence/opendde-confidence.js. The layout, the structural batch (sbatch.*) and the
// pair features come from the exporter, which runs the page's own structural-tokens.js.
#pragma once
#include "confidence.cuh"

constexpr int DDE_ROLES = 7;

// The structural batch replaces the residue batch for everything after the trunk: every
// batch.<x> the structural batch carries is swapped with sbatch.<x> (and back, to write the
// structure on residues). The device copies of the swapped fields are dropped with them.
inline void swapBatch() {
  std::vector<std::string> keys;
  for (auto& [name, e] : M.index) if (name.rfind("sbatch.", 0) == 0) keys.push_back(name.substr(7));
  for (auto& k : keys) {
    std::string a = "batch." + k, b = "sbatch." + k;
    if (!M.index.count(a)) { M.index[a] = M.index[b]; M.index.erase(b); }
    else std::swap(M.index[a], M.index[b]);
    for (auto* key : {&a, &b}) { WF.erase(*key); WH.erase(*key); WLEN.erase(*key); IDEV.erase(*key); }
  }
}

// a structural-stage seam against oracle.structural.stages.<name>, whose token axis the reference
// pads (160 for 6MRR's 130): the first `rows` rows (and, for a [N][N] matrix, columns) compared
inline void structuralTap(const char* name, const float* d, int rows, int cols, bool square = false) {
  std::string k = std::string("oracle.structural.stages.") + name;
  if (!M.has(k)) return;
  int N = (int)M.meta(k + ".shape0");
  std::vector<float> mine = download(d, (size_t)rows * (square ? rows : cols)), theirs;
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < (square ? rows : cols); ++j) theirs.push_back(M.f(k)[(size_t)i * (square ? N : cols) + j]);
  printf("  %-24s relRMS %.3e  (%d of the oracle's %d tokens)\n", name, relRms(mine.data(), theirs.data(), mine.size()), rows, N);
}
struct Structural {
  int n = 0;
  float *single = nullptr, *pair = nullptr, *targetFeat = nullptr, *bias = nullptr;
  float *seqMask = nullptr, *pairMask = nullptr;
};

// targetFeat[parent] + role embedding (the expander's s_inputs), or the parent's single
__global__ void gatherParentK(const float* src, const int* parent, const int* role, const float* roleEmb, float* out,
                              int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int i = (int)(t / C), c = (int)(t % C);
  out[t] = src[(size_t)parent[i] * C + c] + (roleEmb ? roleEmb[(size_t)role[i] * C + c] : 0.f);
}
__global__ void siluK(float* x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; x[i] = v / (1.f + expf(-v)); }
}
// out = a + b + roleEmb[role]
__global__ void singleStructK(float* out, const float* a, const float* b, const int* role, const float* roleEmb,
                              int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int i = (int)(t / C), c = (int)(t % C);
  out[t] = a[t] + b[t] + roleEmb[(size_t)role[i] * C + c];
}
// the residue pair under each structural pair, in matrix-group order
__global__ void gatherPairSortedK(const float* pair, const int* order, const int* parent, float* out, int n, int nRes,
                                  int C, size_t rows) {       // (rows: of `order`, from its pointer - a chunk of it)
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = (int)(t % C);
  int ij = order[r], i = ij / n, j = ij % n;
  out[t] = pair[((size_t)parent[i] * nRes + parent[j]) * C + c];
}
// pair[ij] = gathered + projected + the five boolean-feature embeddings (the false row counts)
__global__ void scatterPairK(float* pair, const float* gathered, const float* projected, const int* order,
                             const int* sameParent, const int* twin, const int* prev, const int* next, const int* type,
                             const float* eSame, const float* eTwin, const float* ePrev, const float* eNext,
                             const float* eType, int n, int C, size_t rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = (int)(t % C);
  int ij = order[r];
  pair[(size_t)ij * C + c] = gathered[t] + projected[t] + eSame[sameParent[ij] * C + c] + eTwin[twin[ij] * C + c]
                           + ePrev[prev[ij] * C + c] + eNext[next[ij] * C + c] + eType[type[ij] * C + c];
}
__global__ void attentionBiasK(float* bias, const int* sameParent, const int* twin, const int* prev, const int* next,
                               const int* type, const float* bSame, const float* bTwin, const float* bPrev,
                               const float* bNext, const float* bType, size_t pairs) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= pairs) return;
  bias[ij] = bSame[0] * sameParent[ij] + bTwin[0] * twin[ij] + bPrev[0] * prev[ij] + bNext[0] * next[ij]
           + bType[type[ij]];
}

// The expander and the refiner. Reads the residue batch (call before swapBatch).
inline Structural expandStructural(const float* trunkSingle, const float* trunkPair, const float* targetFeat, int nRes,
                                   bool fast) {
  const std::string E = "expander.";
  Structural s;
  s.n = (int)M.meta("sbatch.tokens");
  int n = s.n;
  int F = (int)M.meta(E + "singleInputChannels"), Cs = (int)M.meta(E + "singleChannels"), C = (int)M.meta(E + "pairChannels");
  if ((int)M.meta(E + "roles") != DDE_ROLES) { fprintf(stderr, "the expander has %d roles, not 7\n", (int)M.meta(E + "roles")); exit(1); }
  size_t pairs = (size_t)n * n;
  const int* parent = Idev("structural.parent"); const int* role = Idev("structural.role");
  // s_inputs and the single
  s.targetFeat = dalloc((size_t)n * F);
  gatherParentK<<<blocks((size_t)n * F), 256, 0, STREAM>>>(targetFeat, parent, role, W(E + "singleInputRoleEmbedding"),
                                                          s.targetFeat, n, F);
  float* ps = scratch<float>("st.parentSingle", (size_t)n * Cs);
  gatherParentK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(trunkSingle, parent, role, nullptr, ps, n, Cs);
  float* ln = scratch<float>("st.ln", (size_t)n * Cs);
  layerNorm2<float, float>(ps, ln, n, Cs, E + "singleSplitNormScale", E + "singleSplitNormOffset");
  float* hidden = scratch<float>("st.hidden", (size_t)n * 2 * Cs);
  linear<float, float>(ln, hidden, n, Cs, 2 * Cs, E + "singleSplit1");
  siluK<<<blocks((size_t)n * 2 * Cs), 256, 0, STREAM>>>(hidden, (size_t)n * 2 * Cs);
  float* proj = scratch<float>("st.proj", (size_t)n * Cs);
  linear<float, float>(hidden, proj, n, 2 * Cs, Cs, E + "singleSplit2");
  s.single = dalloc((size_t)n * Cs);
  singleStructK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(s.single, ps, proj, role, W(E + "singleRoleEmbedding"), n, Cs);
  structuralTap("expander.targetFeat", s.targetFeat, n, F);
  structuralTap("expander.single", s.single, n, Cs);
  // the pair: each structural pair through the matrix its two roles name (49 of them), as one
  // GEMM per matrix over the pairs sorted by it
  const int* hp = M.i("structural.parent"); const int* hr = M.i("structural.role");
  std::vector<int> order(pairs), start(DDE_ROLES * DDE_ROLES + 1, 0);
  for (size_t ij = 0; ij < pairs; ++ij) ++start[hr[ij / n] * DDE_ROLES + hr[ij % n] + 1];
  for (int k = 0; k < DDE_ROLES * DDE_ROLES; ++k) start[k + 1] += start[k];
  { std::vector<int> fill(start.begin(), start.end() - 1);
    for (size_t ij = 0; ij < pairs; ++ij) order[fill[hr[ij / n] * DDE_ROLES + hr[ij % n]]++] = (int)ij; }
  (void)hp;
  int* dOrder = upload(order.data(), pairs);
  if (lenW(E + "pairBlockProj") != (size_t)DDE_ROLES * DDE_ROLES * C * C) { fprintf(stderr, "pairBlockProj is not 49 x C x C\n"); exit(1); }
  s.pair = dalloc(pairs * C);
  const int *sp = Idev("structural.sameParent"), *tw = Idev("structural.twin"), *pv = Idev("structural.prevBackbone"),
            *nx = Idev("structural.nextBackbone"), *ty = Idev("structural.rolePairType");
  // the sorted pairs whole, or - where two f32 [pairs, C] work buffers do not fit beside the pair they build (12.2 GB
  // each at 1450 residues) - in chunks of the sorted order, each role's GEMM over its part of the chunk (a GEMM of
  // other rows may round differently, so only there)
  const size_t per = roomFor(2 * pairs * C * 4) ? pairs : std::max<size_t>(1, CHUNK / C);
  float* gathered = scratch<float>("st.gathered", per * C);
  float* projected = scratch<float>("st.projected", per * C);
  for (size_t a = 0; a < pairs; a += per) {
    size_t cnt = std::min(per, pairs - a);
    gatherPairSortedK<<<blocks(cnt * C), 256, 0, STREAM>>>(trunkPair, dOrder + a, parent, gathered, n, nRes, C, cnt);
    for (int k = 0; k < DDE_ROLES * DDE_ROLES; ++k) {
      size_t lo = std::max<size_t>(start[k], a), hi = std::min<size_t>(start[k + 1], a + cnt);
      if (lo >= hi) continue;
      std::string w = E + "pairBlockProj#" + std::to_string(k);
      if (!WF.count(w)) deviceWeight(w, (float*)W(E + "pairBlockProj") + (size_t)k * C * C, (size_t)C * C);
      linear<float, float>(gathered + (lo - a) * C, projected + (lo - a) * C, hi - lo, C, C, w);
    }
    scatterPairK<<<blocks(cnt * C), 256, 0, STREAM>>>(s.pair, gathered, projected, dOrder + a, sp, tw, pv, nx, ty,
      W(E + "sameParentEmbedding"), W(E + "sameResidueTwinEmbedding"), W(E + "prevBbChainEmbedding"),
      W(E + "nextBbChainEmbedding"), W(E + "rolePairTypeEmbedding"), n, C, cnt);
  }
  s.bias = dalloc(pairs);
  attentionBiasK<<<blocks(pairs), 256, 0, STREAM>>>(s.bias, sp, tw, pv, nx, ty, W(E + "attnBiasSameParent"),
    W(E + "attnBiasSameResidueTwin"), W(E + "attnBiasPrevBbChain"), W(E + "attnBiasNextBbChain"),
    W(E + "attnBiasRolePairType"), pairs);
  structuralTap("expander.attnBias", s.bias, n, n, true);
  CK(cudaStreamSynchronize(STREAM)); CK(cudaFree(dOrder));
  // the two f32 [pairs, C] work buffers read by nothing past the scatter: given back before the refiner, whose
  // pairformer blocks over the same structural pair need the room (9.8 GB each at 1300 residues, 2530 subtokens,
  // which is what the refiner's triangle ran out beside) - cheap through the scratch pool
  releaseScratch({ "st.gathered", "st.projected" });
  // the masks, from the structural batch's own seq mask
  std::vector<float> seq(M.f("sbatch.seqMask"), M.f("sbatch.seqMask") + n), pm(pairs);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) pm[(size_t)i * n + j] = seq[i] * seq[j];
  s.seqMask = upload(seq.data(), n); s.pairMask = upload(pm.data(), pairs);
  stage("structural.expand");
  // the refiner: four pairformer blocks, each adding the expander's attention bias
  bool swap = M.flag("trunk.dialect.swapTransposedBias"), divide = M.flag("trunk.dialect.triangleMulDivideByLength");
  for (int k = 0; M.has("refiner.blocks." + std::to_string(k) + ".singleChannels"); ++k) {
    std::string B = "refiner.blocks." + std::to_string(k);
    if (fast) pairformerBlockAt<half>(s.pair, s.single, s.pairMask, s.seqMask, n, C, Cs, B, swap, divide, s.bias);
    else pairformerBlockAt<float>(s.pair, s.single, s.pairMask, s.seqMask, n, C, Cs, B, swap, divide, s.bias);
    structuralTap(("refiner.block" + std::to_string(k) + ".single").c_str(), s.single, n, Cs);
  }
  stage("structural.refine");
  return s;
}

// ---------------------------------------------------------------- OpenDDE's confidence head
// pair + s1[j] + s2[i] + the distance's bin embedding (3.25 A up, 1.25 A bins, the last open) +
// the raw distance's projection, off the per-token coordinates the sampler produced
__global__ void ddePairInitK(float* pair, const float* s1, const float* s2, const float* coords, const float* Wd,
                             const float* Wraw, int n, int C, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  float sq = 0;
  for (int k = 0; k < 3; ++k) { float d = coords[i * 3 + k] - coords[j * 3 + k]; sq += d * d; }
  float distance = sqrtf(fmaxf(1e-10f, sq));
  int bin = (int)floorf((distance - 3.25f) / 1.25f);
  if (distance < 3.25f) bin = -1;
  if (bin >= bins) bin = bins - 1;
  pair[t] += s1[(size_t)j * C + c] + s2[(size_t)i * C + c] + (bin >= 0 ? Wd[(size_t)bin * C + c] : 0.f)
           + distance * Wraw[c];
}
// plddt_weight [slot][c][bin] -> [c][slot * bins + bin], so one GEMM gives every slot's logits
__global__ void slotMajorK(const float* w, float* out, int slots, int C, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)slots * C * bins) return;
  int b = (int)(t % bins); size_t r = t / bins; int c = (int)(r % C), slot = (int)(r / C);
  out[(size_t)c * slots * bins + slot * bins + b] = w[t];
}
// tokens' coordinates, refined pair and single, s_inputs: all on the structural tokens.
// Returns pLDDT per structural atom slot, PAE/PDE per structural pair, and the per-pair TM term
// (d0 from `tmTokens`, the residue count pTM is reported over).
struct DdeConfidence { std::vector<float> plddt, pae, pde, tmTerm; };
inline DdeConfidence ddeConfidence(const float* refinedPair, const float* refinedSingle, const float* sInputs,
                                   const float* coords, const float* seqMask, const float* pairMask, const float* bias,
                                   int n, int dense, int tmTokens, bool consume = false) {
  const std::string P = "ddeConfidence.";
  int C = (int)M.meta(P + "pairChannels"), Cs = (int)M.meta(P + "singleChannels"), F = (int)M.meta(P + "singleInputChannels");
  int bins = (int)M.meta(P + "distanceBins"), paeBins = (int)M.meta(P + "paeBins"), pdeBins = (int)M.meta(P + "pdeBins");
  int slots = (int)M.meta(P + "denseSlots"), PB = (int)M.meta(P + "plddtBins");
  if (slots != dense) { fprintf(stderr, "plddt_weight has %d slots, the batch %d\n", slots, dense); exit(1); }
  size_t pairs = (size_t)n * n;
  float* single = scratch<float>("dc.single", (size_t)n * Cs);
  CK(cudaMemcpyAsync(single, refinedSingle, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  clampK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(single, 512.f, (size_t)n * Cs);
  float* sn = scratch<float>("dc.singleNorm", (size_t)n * Cs);
  layerNorm2<float, float>(single, sn, n, Cs, P + "inputStrunkLnScale", P + "inputStrunkLnOffset");
  // consume: the refined pair worked on in place - the fold's last head, nothing reads it after (the copy is a whole
  // f32 pair, 7.7 GB at 1150 residues)
  float* pair = consume ? const_cast<float*>(refinedPair) : scratch<float>("dc.pair", pairs * C);
  if (!consume) CK(cudaMemcpyAsync(pair, refinedPair, pairs * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  float* s1 = scratch<float>("dc.s1", (size_t)n * C); float* s2 = scratch<float>("dc.s2", (size_t)n * C);
  linear<float, float>(sInputs, s1, n, F, C, P + "s1");
  linear<float, float>(sInputs, s2, n, F, C, P + "s2");
  ddePairInitK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, s1, s2, coords, W(P + "distance"), W(P + "distanceRaw"), n, C, bins);
  bool swap = M.flag("trunk.dialect.swapTransposedBias"), divide = M.flag("trunk.dialect.triangleMulDivideByLength");
  for (int k = 0; M.has(P + "blocks." + std::to_string(k) + ".singleChannels"); ++k) {
    std::string B = P + "blocks." + std::to_string(k);
    if (CONF_HALF) pairformerBlockAt<half>(pair, sn, pairMask, seqMask, n, C, Cs, B, swap, divide, bias);
    else pairformerBlockAt<float>(pair, sn, pairMask, seqMask, n, C, Cs, B, swap, divide, bias);
  }
  // PAE from the pair, PDE from the symmetrised pair; 64 bins over [0, 32], softmax against centres - in blocks of
  // rows (each pair's LayerNorm, projection and expectation are its own, so byte-identical): the symmetrised and the
  // LayerNorm'd pair whole were two more f32 copies of the pair beside it and the refined one, 7.7 GB each at 1150
  // residues (2236 structural tokens), which is what ran out of a 40 GB card
  DdeConfidence out;
  int NB = std::max(paeBins, pdeBins);
  const size_t R = std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)n * std::max(C, NB))));
  float* ln = scratch<float>("dc.ln", R * n * C);
  float* sym = scratch<float>("dc.sym", R * n * C);
  float* logits = scratch<float>("dc.logits", R * n * NB);
  float* pde = scratch<float>("dc.pde", pairs); float* pae = scratch<float>("dc.pae", pairs);
  float* tm = scratch<float>("dc.tm", pairs);
  auto centresFor = [](int nb) {
    std::vector<float> c(nb); for (int b = 0; b < nb; ++b) c[b] = 32.f / nb * (b + 0.5f); return c;
  };
  std::vector<float> pdeC = centresFor(pdeBins), paeC = centresFor(paeBins);
  double d0 = 1.24 * std::cbrt(std::max(tmTokens, 19) - 15.0) - 1.8;     // the TM term per pair off the PAE distribution
  std::vector<float> perBin(paeBins);
  for (int b = 0; b < paeBins; ++b) perBin[b] = (float)(1 / (1 + (double)paeC[b] * paeC[b] / (d0 * d0)));
  float* dPde = upload(pdeC.data(), pdeBins); float* dPae = upload(paeC.data(), paeBins);
  float* dPer = upload(perBin.data(), paeBins);
  for (size_t r0 = 0; r0 < (size_t)n; r0 += R) {
    size_t r = std::min(R, (size_t)n - r0), rows = r * n, p0 = r0 * n;
    symmetriseRowsK<<<blocks(rows * C), 256, 0, STREAM>>>(pair, sym, n, C, r0, r);
    layerNorm2<float, float>(sym, ln, rows, C, P + "pdeLnScale", P + "pdeLnOffset");
    linear<float, float>(ln, logits, rows, C, pdeBins, P + "pde");
    expectationK<<<blocks(rows), 256, 0, STREAM>>>(logits, pde + p0, nullptr, rows, pdeBins, dPde, 0, 1.f);
    layerNorm2<float, float>(pair + p0 * C, ln, rows, C, P + "paeLnScale", P + "paeLnOffset");
    linear<float, float>(ln, logits, rows, C, paeBins, P + "pae");
    expectationK<<<blocks(rows), 256, 0, STREAM>>>(logits, pae + p0, nullptr, rows, paeBins, dPae, 0, 1.f);
    expectationK<<<blocks(rows), 256, 0, STREAM>>>(logits, tm + p0, nullptr, rows, paeBins, dPer, 0, 1.f);
  }
  out.pde = download(pde, pairs); out.pae = download(pae, pairs); out.tmTerm = download(tm, pairs);
  CK(cudaFree(dPde)); CK(cudaFree(dPae)); CK(cudaFree(dPer));
  // pLDDT per atom: the token's normalised single against the matrix of the atom's dense slot
  std::string wk = P + "plddtWeight~slotMajor";
  if (!WF.count(wk)) {
    float* w = dalloc((size_t)slots * Cs * PB);
    slotMajorK<<<blocks((size_t)slots * Cs * PB), 256, 0, STREAM>>>(W(P + "plddtWeight"), w, slots, Cs, PB);
    deviceWeight(wk, w, (size_t)slots * Cs * PB);
  }
  float* sln = scratch<float>("dc.sln", (size_t)n * Cs);
  layerNorm2<float, float>(sn, sln, n, Cs, P + "plddtLnScale", P + "plddtLnOffset");
  float* pl = scratch<float>("dc.plddtLogits", (size_t)n * slots * PB);
  linear<float, float>(sln, pl, n, Cs, slots * PB, wk);
  std::vector<float> pc(PB);
  for (int b = 0; b < PB; ++b) pc[b] = (b + 0.5f) / PB;
  float* dpc = upload(pc.data(), PB);
  float* plddt = scratch<float>("dc.plddt", (size_t)n * slots);
  expectationK<<<blocks((size_t)n * slots), 256, 0, STREAM>>>(pl, plddt, nullptr, (size_t)n * slots, PB, dpc, 0, 100.f);
  out.plddt = download(plddt, (size_t)n * slots);
  CK(cudaFree(dpc));
  return out;
}
