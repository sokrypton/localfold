// AlphaFold 2's whole model for training, composed (cuda/grad/af2io.cuh, af2.cuh, af2sm.cuh): the embedder, the extra
// MSA stack, the main stack, the single, the structure module and every head - one pass, the recycled inputs zero. Two
// ways:
//   1. float32, every block: against af3-any-model's JAX itself (cuda/af2/oracle.py's dump of data-test59, pass 0) -
//      the pair after the extra stack, the main stack's msa, pair and single, the structure module's trajectory, act and
//      atom14 positions, the pLDDT, PAE, distogram and masked-MSA logits;
//   2. -DGRAD_F64: the backward against central differences through the whole composition, each stack one block.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-af2model cuda/grad/test_af2model.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-af2model [--weights=cuda/af2/weights-model_1_ptm] [--input=cuda/af2/data-test59]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-af2model-f64 cuda/grad/test_af2model.cu -lcublas -lcublasLt -ldl -lcurand
#include "../af2/src/evoformer.cuh"
#include "af2.cuh"
#include "af2io.cuh"
#include "af2sm.cuh"
#include "gradcheck.cuh"
#include <chrono>

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
  return d;
}
static g::real* devReal(const float* dev, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), dev, n * 4, cudaMemcpyDeviceToHost));
  return devReal(std::vector<double>(h.begin(), h.end()));
}
static std::vector<double> gauss(size_t n, unsigned seed, double sd) {
  std::vector<double> v(n);
  unsigned long long x = seed * 2654435761ull + 1;
  for (auto& e : v) {
    double u1, u2;
    x = x * 6364136223846793005ull + 1442695040888963407ull; u1 = ((x >> 11) + 0.5) / 9007199254740992.0;
    x = x * 6364136223846793005ull + 1442695040888963407ull; u2 = ((x >> 11) + 0.5) / 9007199254740992.0;
    e = sd * std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
  return v;
}

int main(int argc, char** argv) {
  int failed = 0;
  std::string weights = "cuda/af2/weights-model_1_ptm", input = "cuda/af2/data-test59";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.load(weights);
  M.load(input);
  M.load(input + "/oracle");
  const bool multimer = M.flag("meta/multimer");
  const int L = (int)M.len("aatype"), N = (int)(M.len("f0/msa_mask") / L);
  int E = (int)(M.len("f0/extra_msa_mask") / L);
  if (getenv("GRAD_EXTRA_ROWS")) E = std::min(E, atoi(getenv("GRAD_EXTRA_ROWS")));     // (the alignment's first rows only)
  const size_t pairs = (size_t)L * L;
  g::Params ps;
  auto load = [&](const std::string& nm, g::Param& p, size_t len, int block) {
    g::allocParam(p, nm + (block >= 0 ? "@" + std::to_string(block) : std::string()), len); ps.add(p);
    const size_t have = M.len("w/" + nm) / (block >= 0 ? (size_t)dimW(nm, 0) : 1);
    if (have != len) { fprintf(stderr, "%s: %zu, %zu expected\n", nm.c_str(), have, len); exit(1); }
    std::vector<float> h(len); GK(cudaMemcpy(h.data(), P(nm, block), len * 4, cudaMemcpyDeviceToHost));
    std::vector<g::real> r(h.begin(), h.end());
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  };
  auto plain = [&](const std::string& nm, g::Param& p, size_t len) { load(nm, p, len, -1); };
  g::af2::Embedder emb; emb.multimer = multimer;
  g::af2::forEachParam(emb, plain);
  // the stacks: each block's slice of the stacked tensors
  int nExtra = (int)dimW("evoformer/extra_msa_stack/outer_product_mean/left_projection/weights", 0);
  int nMain = (int)dimW("evoformer/evoformer_iteration/outer_product_mean/left_projection/weights", 0);
#ifdef GRAD_F64
  nExtra = 1; nMain = 1;
#endif
  g::af2::Stack ext, mainS;
  ext.blocks.resize(nExtra); mainS.blocks.resize(nMain);
  for (int k = 0; k < nExtra; ++k) {
    g::af2::Block& b = ext.blocks[k]; b.extra = true; b.Cm = emb.Ce; b.opmFirst = multimer;
    g::af2::forEachParam(b, [&](const std::string& nm, g::Param& p, size_t len) { load("evoformer/extra_msa_stack/" + nm, p, len, k); });
  }
  for (int k = 0; k < nMain; ++k) {
    g::af2::Block& b = mainS.blocks[k]; b.opmFirst = multimer;
    g::af2::forEachParam(b, [&](const std::string& nm, g::Param& p, size_t len) { load("evoformer/evoformer_iteration/" + nm, p, len, k); });
  }
  g::af2::StructureModule sm;
  g::af2::forEachParam(sm, plain);
  g::af2::Heads hd; hd.hasPae = M.has("w/predicted_aligned_error_head/logits/weights");
  g::af2::forEachParam(hd, plain);
  // the inputs: pass 0's features, the recycled ones zero
  g::af2::EmbedInputs in{};
  in.L = L; in.N = N; in.E = E;
  in.msaFeat = devReal(W("f0/msa_feat"), M.len("f0/msa_feat")); in.aatype = Idev("aatype");
  in.ri = Idev("residue_index"); in.asym = Idev("asym_id"); in.entity = Idev("entity_id"); in.sym = Idev("sym_id");
  in.extraMsa = Idev("f0/extra_msa"); in.hasDel = devReal(W("f0/extra_has_deletion"), (size_t)E * L);
  in.delVal = devReal(W("f0/extra_deletion_value"), (size_t)E * L);
  in.prevRow = devReal(std::vector<double>((size_t)L * emb.C, 0)); in.prevPair = devReal(std::vector<double>(pairs * emb.Cz, 0));
  in.prevPos = devReal(std::vector<double>((size_t)L * 111, 0));
  g::real *msaMask = devReal(W("f0/msa_mask"), (size_t)N * L), *extraMask = devReal(W("f0/extra_msa_mask"), (size_t)E * L);
  g::real* seqMask = devReal(W("seq_mask"), L);
  std::vector<double> pmH(pairs); { auto sq = std::vector<float>(M.f("seq_mask"), M.f("seq_mask") + L);
    for (int i = 0; i < L; ++i) for (int j = 0; j < L; ++j) pmH[(size_t)i * L + j] = sq[i] * sq[j]; }
  g::real* pairMask = devReal(pmH);
  g::af2::ResidueTables T;
  T.defaultFrames = devReal(W("c/rigid_group_default_frame"), M.len("c/rigid_group_default_frame"));
  T.litPos = devReal(W("c/atom14_rigid_group_positions"), M.len("c/atom14_rigid_group_positions"));
  T.atom14Mask = devReal(W("c/atom14_mask"), M.len("c/atom14_mask"));
  T.atom37Mask = devReal(W("c/atom37_mask"), M.len("c/atom37_mask"));
  T.atom14Group = Idev("c/atom14_to_rigid_group"); T.atom37To14 = Idev("c/atom37_to_atom14");
  T.aatype = Idev("aatype");
  printf("AF2 whole model (%s, %s): %d residues, %d MSA rows, %d extra rows; extra stack %d blocks, main %d, structure module %d "
         "iterations, %s\n", weights.c_str(), input.c_str(), L, N, E, nExtra, nMain, sm.iterations, multimer ? "multimer" : "monomer");
  g::real *msa, *pair, *extra;
  GK(cudaMalloc(&msa, (size_t)N * L * emb.C * sizeof(g::real))); GK(cudaMalloc(&pair, pairs * emb.Cz * sizeof(g::real)));
  GK(cudaMalloc(&extra, (size_t)E * L * emb.Ce * sizeof(g::real)));
  struct Run { g::af2::EmbedSave es; g::af2::SmRun sr; g::af2::HeadSave hs; g::af2::HeadOut ho; g::real* single; };
  // the forward: the stacks keep their block inputs (their own buffers); the rest in the arena
  auto forward = [&](Run& r) {
    r.es = g::af2::embedForward(emb, in, msa, pair, extra);
    ext.forward(extra, pair, extraMask, pairMask, E, L);
    mainS.forward(msa, ext.pairIn.back(), msaMask, pairMask, N, L);
    const g::real *m = mainS.msaIn.back(), *z = mainS.pairIn.back();
    r.single = g::ARENA.get((size_t)L * hd.Cs); hd.single.forward(m, r.single, L);
    g::af2::smForward(sm, r.sr, r.single, z, seqMask, T, L);
    r.hs = g::af2::headsForward(hd, m, z, r.sr.acts[sm.iterations], L, N, r.ho);
  };
#ifndef GRAD_F64
  g::ARENA.reset();
  Run r; forward(r);
  GK(cudaStreamSynchronize(STREAM));
  auto cmp = [&](const char* what, const g::real* mine, const std::string& key, size_t n, double bar) {
    auto a = g::host(mine, n); const float* o = M.f(key);
    double e = g::relRms(std::vector<double>(a.begin(), a.end()), std::vector<double>(o, o + n));
    printf("  %-44s %.3e%s\n", what, e, e < bar ? "" : "  <-- ABOVE BAR");
    if (!(e < bar)) ++failed;
  };
  cmp("the pair after the extra stack", ext.pairIn.back(), "o/extra/pair", pairs * emb.Cz, 1e-4);
  cmp("the msa after the main stack", mainS.msaIn.back(), "o/full/msa", (size_t)N * L * emb.C, 1e-4);
  cmp("the pair after the main stack", mainS.pairIn.back(), "o/full/pair", pairs * emb.Cz, 1e-4);
  cmp("the single", r.single, "o/full/single", (size_t)L * hd.Cs, 1e-4);
  {   // (the dump's frames are 3 x 4 [R | t] row by row, these R (9) then t (3))
    auto a = g::host(r.sr.traj, (size_t)sm.iterations * L * 12); const float* o = M.f("o/full/traj");
    std::vector<double> x, y;
    for (size_t q = 0; q < (size_t)sm.iterations * L; ++q) for (int row = 0; row < 3; ++row) for (int k = 0; k < 4; ++k) {
      x.push_back(a[q * 12 + (k < 3 ? row * 3 + k : 9 + row)]); y.push_back(o[q * 12 + row * 4 + k]);
    }
    double e = g::relRms(x, y);
    printf("  %-44s %.3e%s\n", "the structure module's trajectory", e, e < 1e-4 ? "" : "  <-- ABOVE BAR");
    if (!(e < 1e-4)) ++failed;
  }
  cmp("the structure module's act", r.sr.acts[sm.iterations], "o/full/structure_act", (size_t)L * sm.C, 1e-4);
  cmp("atom14", r.sr.pos14, "o/full/final_atom14_positions", (size_t)L * 42, 1e-4);
  cmp("pLDDT logits", r.ho.plddt, "o/full/plddt_logits", (size_t)L * hd.plddtBins, 1e-4);
  if (hd.hasPae) cmp("PAE logits", r.ho.pae, "o/full/pae_logits", pairs * hd.paeBins, 1e-4);
  cmp("distogram logits", r.ho.distogram, "o/full/distogram_logits", pairs * hd.dgBins, 1e-4);
  cmp("masked-MSA logits", r.ho.maskedMsa, "o/full/masked_msa_logits", (size_t)N * L * hd.msaClasses, 1e-4);
  // and a whole training step, timed
  {
    auto ctD = devReal(gauss(pairs * hd.dgBins, 3, 1)), ctP = devReal(gauss((size_t)L * hd.plddtBins, 4, 1)),
         ctT = devReal(gauss((size_t)sm.iterations * L * 12, 5, 1));
    for (int rep = 0; rep < 2; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      g::ARENA.reset(); ps.zeroGrad();
      Run q; forward(q);
      GK(cudaStreamSynchronize(STREAM));
      auto t1 = std::chrono::steady_clock::now();
      g::real *dm = g::ARENA.zeros((size_t)N * L * emb.C), *dz = g::ARENA.zeros(pairs * emb.Cz), *dact = g::ARENA.zeros((size_t)L * sm.C),
              *dsg = g::ARENA.zeros((size_t)L * hd.Cs);
      g::af2::HeadOut d{}; d.distogram = ctD; d.plddt = ctP;
      g::af2::headsBackward(hd, q.hs, mainS.msaIn.back(), mainS.pairIn.back(), q.sr.acts[sm.iterations], L, N, d, nullptr, dm, dz, dact);
      g::af2::smBackward(sm, q.sr, seqMask, T, ctT, dact, nullptr, nullptr, nullptr, nullptr, dsg, dz);
      hd.single.backward(mainS.msaIn.back(), dsg, dm, L, true);
      mainS.backward(msaMask, pairMask, dm, dz);
      g::real* de = g::ARENA.zeros((size_t)E * L * emb.Ce);
      ext.backward(extraMask, pairMask, de, dz);
      g::af2::embedBackward(emb, in, q.es, dm, dz, de);
      GK(cudaStreamSynchronize(STREAM));
      auto t2 = std::chrono::steady_clock::now();
      if (rep) printf("  a step (%d + %d blocks, %d residues): forward %.0f ms, backward %.0f ms; arena high %.2f GB, stack inputs %.2f GB\n",
                      nExtra, nMain, L, std::chrono::duration<double, std::milli>(t1 - t0).count(),
                      std::chrono::duration<double, std::milli>(t2 - t1).count(), g::ARENA.high / 1e9,
                      (ext.inputBytes() + mainS.inputBytes()) / 1e9);
    }
  }
#else
  // every head and the structure module's outputs a cotangent
  const size_t nTraj = (size_t)sm.iterations * L * 12;
  g::real *ctT = devReal(gauss(nTraj, 3, 1)), *ctP = devReal(gauss((size_t)L * hd.plddtBins, 4, 1)),
          *ctA = hd.hasPae ? devReal(gauss(pairs * hd.paeBins, 5, 1)) : nullptr, *ctD = devReal(gauss(pairs * hd.dgBins, 6, 1)),
          *ctM = devReal(gauss((size_t)N * L * hd.msaClasses, 7, 1)), *ctR = devReal(gauss((size_t)L * 37, 8, 1)),
          *ct14 = devReal(gauss((size_t)L * 42, 9, 1));
  g::real *oT, *oP, *oA = nullptr, *oD, *oM, *oR, *o14;
  for (auto [p, n] : std::vector<std::pair<g::real**, size_t>>{{&oT, nTraj}, {&oP, (size_t)L * hd.plddtBins}, {&oD, pairs * hd.dgBins},
                                                               {&oM, (size_t)N * L * hd.msaClasses}, {&oR, (size_t)L * 37}, {&o14, (size_t)L * 42}})
    GK(cudaMalloc(p, n * sizeof(g::real)));
  if (hd.hasPae) GK(cudaMalloc(&oA, pairs * hd.paeBins * sizeof(g::real)));
  auto copyOut = [&](const Run& r) {
    auto cp = [](g::real* d, const g::real* s, size_t n) { GK(cudaMemcpyAsync(d, s, n * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM)); };
    cp(oT, r.sr.traj, nTraj); cp(oP, r.ho.plddt, (size_t)L * hd.plddtBins); if (oA) cp(oA, r.ho.pae, pairs * hd.paeBins);
    cp(oD, r.ho.distogram, pairs * hd.dgBins); cp(oM, r.ho.maskedMsa, (size_t)N * L * hd.msaClasses); cp(oR, r.ho.resolved, (size_t)L * 37);
    cp(o14, r.sr.pos14, (size_t)L * 42);
  };
  auto fwd = [&] { g::ARENA.reset(); Run r; forward(r); copyOut(r); };
  ps.zeroGrad();
  g::ARENA.reset();
  Run r; forward(r);
  g::real *dm = g::ARENA.zeros((size_t)N * L * emb.C), *dz = g::ARENA.zeros(pairs * emb.Cz), *dact = g::ARENA.zeros((size_t)L * sm.C),
          *dsg = g::ARENA.zeros((size_t)L * hd.Cs), *de = g::ARENA.zeros((size_t)E * L * emb.Ce);
  g::af2::HeadOut d{}; d.plddt = ctP; d.pae = ctA; d.distogram = ctD; d.maskedMsa = ctM; d.resolved = ctR;
  // the heads first (their pLDDT reads the structure module's act), then the structure module (its single and pair)
  g::af2::headsBackward(hd, r.hs, mainS.msaIn.back(), mainS.pairIn.back(), r.sr.acts[sm.iterations], L, N, d, nullptr, dm, dz, dact);
  g::af2::smBackward(sm, r.sr, seqMask, T, ctT, dact, nullptr, nullptr, ct14, nullptr, dsg, dz);
  hd.single.backward(mainS.msaIn.back(), dsg, dm, L, true);          // (the single is the msa's first row's projection)
  mainS.backward(msaMask, pairMask, dm, dz);
  ext.backward(extraMask, pairMask, de, dz);
  g::af2::embedBackward(emb, in, r.es, dm, dz, de);
  GK(cudaStreamSynchronize(STREAM));
  g::GradCheck gc;
  gc.verbose = getenv("GRAD_VERBOSE") != nullptr;
  gc.rel = getenv("GRAD_REL") ? atof(getenv("GRAD_REL")) : 1e-8;     // (the transitions' and side chain's ReLUs)
  for (auto* p : ps.all) gc.add(*p);
  gc.output(oT, ctT, nTraj); gc.output(oP, ctP, (size_t)L * hd.plddtBins); if (oA) gc.output(oA, ctA, pairs * hd.paeBins);
  gc.output(oD, ctD, pairs * hd.dgBins); gc.output(oM, ctM, (size_t)N * L * hd.msaClasses); gc.output(oR, ctR, (size_t)L * 37);
  gc.output(o14, ct14, (size_t)L * 42);
  failed += gc.run(fwd);
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
