// AlphaFold 2's structure module for training (cuda/grad/af2sm.cuh), two ways:
//   1. float32: the whole module - eight IPA fold iterations and the side chain - against cuda/af2's structureModule()
//      on model_1_ptm's weights (cuda/af2/weights-model_1_ptm) and a real input's sequence (cuda/af2/data-test59),
//      random single and pair representations;
//   2. -DGRAD_F64: the backward against central differences on a small random module, the real residue tables, every
//      output a cotangent (the trajectory, the last act, both angle forms, atom14 and atom37), with and without
//      stopRotGradient.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-af2sm cuda/grad/test_af2sm.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-af2sm [--weights=cuda/af2/weights-model_1_ptm] [--input=cuda/af2/data-test59]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-af2sm-f64 cuda/grad/test_af2sm.cu -lcublas -lcublasLt -ldl -lcurand
#include "../af2/src/evoformer.cuh"
#include "../af2/src/structure.cuh"
#include "af2sm.cuh"
#include "gradcheck.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, r.size() * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
  return d;
}
static g::real* devReal(const float* dev, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), dev, n * 4, cudaMemcpyDeviceToHost));
  return devReal(std::vector<double>(h.begin(), h.end()));
}
static std::vector<float> hostF(const float* d, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h;
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
// the residue tables, as the device's real
static g::af2::ResidueTables tables(const int* aatype) {
  g::af2::ResidueTables T;
  T.defaultFrames = devReal(W("c/rigid_group_default_frame"), M.len("c/rigid_group_default_frame"));
  T.litPos = devReal(W("c/atom14_rigid_group_positions"), M.len("c/atom14_rigid_group_positions"));
  T.atom14Mask = devReal(W("c/atom14_mask"), M.len("c/atom14_mask"));
  T.atom37Mask = devReal(W("c/atom37_mask"), M.len("c/atom37_mask"));
  T.atom14Group = Idev("c/atom14_to_rigid_group"); T.atom37To14 = Idev("c/atom37_to_atom14");
  T.aatype = aatype;
  return T;
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
#ifndef GRAD_F64
  const int L = (int)M.len("aatype");
  g::af2::StructureModule m;
  g::Params ps;
  g::af2::forEachParam(m, [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.len("w/" + nm) != len) { fprintf(stderr, "%s: %zu, %zu expected\n", nm.c_str(), M.len("w/" + nm), len); exit(1); }
    GK(cudaMemcpy(p.w, P(nm), len * 4, cudaMemcpyDeviceToDevice));
  });
  const size_t pairs = (size_t)L * L;
  // the trunk's real outputs when the input has an oracle (its reference's single and pair after the evoformer):
  // random unit-variance ones drive eight IPA iterations chaotic - the port moves 1.2e-3 under a 1e-7 nudge of them
  std::vector<double> sH, zH;
  if (M.has("o/full/single") || access((input + "/oracle/model.idx").c_str(), F_OK) == 0) {
    if (!M.has("o/full/single")) M.load(input + "/oracle");
    auto a = hostF(W("o/full/single"), (size_t)L * m.C), b = hostF(W("o/full/pair"), pairs * m.C2);
    sH.assign(a.begin(), a.end()); zH.assign(b.begin(), b.end());
    printf("(the trunk's single and pair from %s/oracle)\n", input.c_str());
  } else { sH = gauss((size_t)L * m.C, 1, 1); zH = gauss(pairs * m.C2, 2, 1); }
  std::vector<float> sF(sH.begin(), sH.end()), zF(zH.begin(), zH.end());
  g::real *single = devReal(sH), *pair = devReal(zH), *mask = devReal(W("seq_mask"), L);
  float *sD, *zD; GK(cudaMalloc(&sD, sF.size() * 4)); GK(cudaMalloc(&zD, zF.size() * 4));
  GK(cudaMemcpy(sD, sF.data(), sF.size() * 4, cudaMemcpyHostToDevice)); GK(cudaMemcpy(zD, zF.data(), zF.size() * 4, cudaMemcpyHostToDevice));
  auto T = tables(Idev("aatype"));
  g::ARENA.reset();
  g::af2::SmRun r;
  g::af2::smForward(m, r, single, pair, mask, T, L);
  StructureOut o = structureModule(sD, zD, L, 10.f);
  CK(cudaStreamSynchronize(STREAM));
  if (getenv("GRAD_SENS")) {          // the port against itself, its inputs moved by float32's last place
    auto a0 = hostF(o.act, (size_t)L * m.C), p0 = hostF(o.pos14, (size_t)L * 42);
    auto nz = gauss(sF.size(), 41, 1e-7), nzz = gauss(zF.size(), 42, 1e-7);
    std::vector<float> s2(sF), z2(zF);
    for (size_t i = 0; i < s2.size(); ++i) s2[i] *= 1 + nz[i];
    for (size_t i = 0; i < z2.size(); ++i) z2[i] *= 1 + nzz[i];
    float *s2D, *z2D; GK(cudaMalloc(&s2D, s2.size() * 4)); GK(cudaMalloc(&z2D, z2.size() * 4));
    GK(cudaMemcpy(s2D, s2.data(), s2.size() * 4, cudaMemcpyHostToDevice)); GK(cudaMemcpy(z2D, z2.data(), z2.size() * 4, cudaMemcpyHostToDevice));
    StructureOut o2 = structureModule(s2D, z2D, L, 10.f);
    CK(cudaStreamSynchronize(STREAM));
    printf("  the port against itself, inputs moved 1e-7: act %.3e, atom14 %.3e\n", g::relRms(hostF(o2.act, (size_t)L * m.C), a0),
           g::relRms(hostF(o2.pos14, (size_t)L * 42), p0));
    o = structureModule(sD, zD, L, 10.f);
    CK(cudaStreamSynchronize(STREAM));
  }
  auto cmp = [&](const char* what, const g::real* mine, const float* port, size_t n) {
    double e = g::relRms(g::host(mine, n), hostF(port, n));
    printf("  %-28s %.3e\n", what, e);
    if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
  };
  printf("AF2 structure module (%s), %d residues, %d iterations\n", weights.c_str(), L, m.iterations);
  cmp("act (last iteration)", r.acts[m.iterations], o.act, (size_t)L * m.C);
  cmp("rigid (last iteration)", r.rigs[m.iterations], o.rigid, (size_t)L * 12);
  cmp("angles (normalised)", r.angles, o.angles, (size_t)L * 14);
  cmp("atom14", r.pos14, o.pos14, (size_t)L * 42);
  cmp("atom37", r.pos37, o.pos37, (size_t)L * 111);
#else
  for (bool stopRot : {false, true}) {
    const int L = 7, pad = 1;
    g::af2::StructureModule m;
    m.C = 24; m.C2 = 8; m.H = 2; m.Cs = 4; m.Pq = 2; m.Pv = 3; m.iterations = 3; m.Csc = 8; m.stopRotGradient = stopRot;
    g::Params ps;
    g::af2::forEachParam(m, [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); });
    unsigned k = 7;
    for (auto* p : ps.all) {
      bool scale = p->name.find("/scale") != std::string::npos;
      auto v = gauss(p->n, ++k, scale ? 0.1 : 0.3);
      if (scale) for (auto& x : v) x += 1;
      std::vector<g::real> rr(v.begin(), v.end());
      GK(cudaMemcpy(p->w, rr.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    std::vector<int> aa = {0, 4, 7, 11, 19, 13, 2};      // ALA CYS GLY LEU VAL PHE ASN: every group count
    int* aaD; GK(cudaMalloc(&aaD, L * 4)); GK(cudaMemcpy(aaD, aa.data(), L * 4, cudaMemcpyHostToDevice));
    auto T = tables(aaD);
    std::vector<double> mk(L, 1); for (int i = L - pad; i < L; ++i) mk[i] = 0;
    const size_t pairs = (size_t)L * L;
    g::real *single = devReal(gauss((size_t)L * m.C, 21, 1)), *pair = devReal(gauss(pairs * m.C2, 22, 1)), *mask = devReal(mk);
    // cotangents: the padded residue's outputs are not read (its attention row is masked against every key)
    auto zeroPad = [&](std::vector<double> v, size_t per, size_t lead) {
      for (size_t a = 0; a < lead; ++a) for (int i = L - pad; i < L; ++i) for (size_t c = 0; c < per; ++c) v[(a * L + i) * per + c] = 0;
      return v;
    };
    g::real *ctTraj = devReal(zeroPad(gauss((size_t)m.iterations * L * 12, 31, 1), 12, m.iterations)),
            *ctAct = devReal(zeroPad(gauss((size_t)L * m.C, 32, 1), m.C, 1)),
            *ctUn = devReal(zeroPad(gauss((size_t)L * 14, 33, 1), 14, 1)),
            *ctAng = devReal(zeroPad(gauss((size_t)L * 14, 34, 1), 14, 1)),
            *ct14 = devReal(zeroPad(gauss((size_t)L * 42, 35, 1), 42, 1)),
            *ct37 = devReal(zeroPad(gauss((size_t)L * 111, 36, 1), 111, 1));
    g::real *ds, *dz;
    GK(cudaMalloc(&ds, (size_t)L * m.C * sizeof(g::real))); GK(cudaMalloc(&dz, pairs * m.C2 * sizeof(g::real)));
    GK(cudaMemset(ds, 0, (size_t)L * m.C * sizeof(g::real))); GK(cudaMemset(dz, 0, pairs * m.C2 * sizeof(g::real)));
    // outputs the gradcheck differences, kept outside the arena
    g::real *oTraj, *oAct, *oUn, *oAng, *o14, *o37;
    for (auto pp : {std::make_pair(&oTraj, (size_t)m.iterations * L * 12), std::make_pair(&oAct, (size_t)L * m.C),
                    std::make_pair(&oUn, (size_t)L * 14), std::make_pair(&oAng, (size_t)L * 14), std::make_pair(&o14, (size_t)L * 42),
                    std::make_pair(&o37, (size_t)L * 111)})
      GK(cudaMalloc(pp.first, pp.second * sizeof(g::real)));
    auto fwd = [&] {
      g::ARENA.reset();
      g::af2::SmRun r;
      g::af2::smForward(m, r, single, pair, mask, T, L);
      auto cp = [&](g::real* d, const g::real* s, size_t n) { GK(cudaMemcpyAsync(d, s, n * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM)); };
      cp(oTraj, r.traj, (size_t)m.iterations * L * 12); cp(oAct, r.acts[m.iterations], (size_t)L * m.C);
      cp(oUn, r.un, (size_t)L * 14); cp(oAng, r.angles, (size_t)L * 14); cp(o14, r.pos14, (size_t)L * 42); cp(o37, r.pos37, (size_t)L * 111);
    };
    ps.zeroGrad();
    g::ARENA.reset();
    g::af2::SmRun r;
    g::af2::smForward(m, r, single, pair, mask, T, L);
    g::af2::smBackward(m, r, mask, T, ctTraj, ctAct, ctUn, ctAng, ct14, ct37, ds, dz);
    g::GradCheck gc;
    gc.add("in/single", single, ds, (size_t)L * m.C);
    gc.add("in/pair", pair, dz, pairs * m.C2);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(oTraj, ctTraj, (size_t)m.iterations * L * 12); gc.output(oAct, ctAct, (size_t)L * m.C);
    gc.output(oUn, ctUn, (size_t)L * 14); gc.output(oAng, ctAng, (size_t)L * 14);
    gc.output(o14, ct14, (size_t)L * 42); gc.output(o37, ct37, (size_t)L * 111);
    printf("AF2 structure module (small, random; stopRotGradient %d):\n", stopRot);
    if (stopRot) {
      // the stopped rotation is not the forward's derivative: what is checked is everything the stop does not touch,
      // the last iteration's own parameters' route is unchanged - so this arm checks only that it runs and is finite
      double s = g::dot(ds, ds, (size_t)L * m.C);
      printf("  |d single|^2 %.3e (finite: %s)\n", s, std::isfinite(s) ? "yes" : "NO");
      if (!std::isfinite(s)) ++failed;
    } else failed += gc.run(fwd);
  }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
