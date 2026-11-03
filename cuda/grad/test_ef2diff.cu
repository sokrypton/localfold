// ESMFold2's diffusion module for training (cuda/grad/ef2diff.cuh), two ways:
//   1. float32: one denoiser call against cuda/ef2's makeDenoiser + denoise in their float32 arm (FAST and ATOM_BF16
//      off) on the exported weights and a real input's atoms and tokens, random trunk pair, s_inputs and noisy atoms;
//   2. -DGRAD_F64: the backward against central differences on the same module and input.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-ef2diff cuda/grad/test_ef2diff.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-ef2diff [--weights=cuda/ef2/weights] [--input=cuda/ef2/data-6mrr] [--t=5]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-ef2diff-f64 cuda/grad/test_ef2diff.cu -lcublas -lcublasLt -ldl -lcurand
#include "../ef2/src/shim.cuh"
#include "../ef2/src/atoms.cuh"
#include "../ef2/src/trunk.cuh"
#include "../ef2/src/diffusion.cuh"
#include "ef2diff.cuh"
#include "gradcheck.cuh"

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
static float* devF(const std::vector<double>& h) {
  std::vector<float> r(h.begin(), h.end());
  float* d; GK(cudaMalloc(&d, r.size() * 4)); GK(cudaMemcpy(d, r.data(), r.size() * 4, cudaMemcpyHostToDevice));
  return d;
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
  std::string weights = "cuda/ef2/weights", input = "cuda/ef2/data-6mrr";
  double t = 5;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
    else if (!strncmp(argv[i], "--t=", 4)) t = atof(argv[i] + 4);
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.load(weights);
  M.load(input);
  ATOM_BF16 = false; FAST = false; GEMM16 = false;
  const int T = (int)M.meta("meta/tokens"), A = (int)M.len("atom_mask");
  g::ef2::Ef2Denoiser m;
  m.T = T; m.A = A; m.Cz = (int)M.meta("meta/pairChannels"); m.Ct = (int)M.meta("meta/tokenChannels2");
  m.H = (int)M.meta("meta/tokenHeads"); m.Si = (int)M.meta("meta/singleInputs"); m.sigma = (g::real)M.meta("meta/sigmaData");
  m.nf = (int)M.len("f/diffusion/fourier/weights"); m.Ca = (int)M.meta("meta/atomChannels");
  m.Hd = (int)dimOf("f/diffusion/tokenBlocks/0/transition/outWeights", 0);
  const int nb = (int)M.meta("meta/tokenBlocks"), ab = (int)M.meta("meta/atomBlocks");
  const int pairHd = (int)dimOf("f/diffusion/zTransitions/0/aProjection", 1), singleHd = (int)dimOf("f/diffusion/sTransitions/0/aProjection", 1);
  const int atomHd = (int)dimOf("f/diffusionAtomEncoder/blocks/0/ffnDown", 0);
  g::Params ps;
  g::ef2::forEachParam(m, nb, ab, atomHd, pairHd, singleHd, [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.len(nm) != len) { fprintf(stderr, "%s: %zu, %zu expected\n", nm.c_str(), M.len(nm), len); exit(1); }
    std::vector<float> h(len); GK(cudaMemcpy(h.data(), W(nm), len * 4, cudaMemcpyDeviceToHost));
    std::vector<g::real> r(h.begin(), h.end());
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  });
  Atoms at = prepareAtoms(A, "diffusionAtomEncoder");
  float* featF = scratch<float>("test.features", (size_t)A * ATOM_FEATURES);
  atomFeaturesK<<<blocks((size_t)A * ATOM_FEATURES), 256, 0, STREAM>>>(W("ref_pos"), W("ref_charge"), at.ctx.mask,
    Idev("ref_element"), Idev("ref_atom_name_chars"), featF, A);
  g::ef2::Ef2Batch b;
  b.L = g::ef2::AtomLayout{A, at.ctx.nValid, at.ctx.valid, devReal(at.ctx.mask, A), devReal(at.ctx.cosT, (size_t)A * 16),
                           devReal(at.ctx.sinT, (size_t)A * 16)};
  b.feat = devReal(featF, (size_t)A * ATOM_FEATURES);
  b.tokStart = at.ctx.tokenStart; b.tokAtoms = at.ctx.tokenAtoms; b.atomToTok = Idev("atom_to_token");
  b.ri = Idev("residue_index"); b.asym = Idev("asym_id"); b.sym = Idev("sym_id"); b.ent = Idev("entity_id"); b.ti = Idev("token_index");
  b.T = T;
  const size_t P = (size_t)T * T;
  auto zH = gauss(P * m.Cz, 1, 1), sH = gauss((size_t)T * m.Si, 2, 1), xH = gauss((size_t)A * 3, 3, t);
  g::real *z = devReal(zH), *si = devReal(sH), *x = devReal(xH), *out;
  GK(cudaMalloc(&out, (size_t)A * 3 * sizeof(g::real)));
  printf("ESMFold2 diffusion module (%s, %s), %d tokens, %d atoms, t %g: %d token blocks (C %d, %d heads), %d atom blocks, pair %d\n",
         weights.c_str(), input.c_str(), T, A, t, nb, m.Ct, m.H, ab, m.Cz);
#ifndef GRAD_F64
  g::ARENA.reset();
  g::ef2::ef2Forward(m, b, z, si, x, (g::real)t, out);
  Denoiser d = makeDenoiser(T, A, devF(zH), devF(sH), false);
  float* ref = dalloc((size_t)A * 3);
  denoise(d, devF(xH), (float)t, ref, false);
  CK(cudaStreamSynchronize(STREAM));
  auto mine = g::host(out, (size_t)A * 3); std::vector<float> port((size_t)A * 3);
  GK(cudaMemcpy(port.data(), ref, port.size() * 4, cudaMemcpyDeviceToHost));
  auto mk = download(at.ctx.mask, A);
  // the update alone, D - keep x (the skip term dominates at a small t)
  const double keep = (double)m.sigma * m.sigma / ((double)m.sigma * m.sigma + t * t);
  std::vector<double> a, r;
  for (int i = 0; i < A; ++i) if (mk[i] != 0) for (int k = 0; k < 3; ++k) { a.push_back(mine[i * 3 + k] - keep * xH[i * 3 + k]); r.push_back(port[i * 3 + k] - keep * xH[i * 3 + k]); }
  double e = g::relRms(a, r);
  printf("  against cuda/ef2's denoise() (float32 arm): its update %.3e over %zu valid atoms\n", e, a.size() / 3);
  if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
#else
  auto ctH = gauss((size_t)A * 3, 9, 1);
  auto mk = download(at.ctx.mask, A);
  for (int i = 0; i < A; ++i) if (mk[i] == 0) for (int k = 0; k < 3; ++k) ctH[i * 3 + k] = 0;
  g::real* ct = devReal(ctH);
  g::real *dz, *ds, *dx;
  GK(cudaMalloc(&dz, P * m.Cz * sizeof(g::real))); GK(cudaMalloc(&ds, (size_t)T * m.Si * sizeof(g::real)));
  GK(cudaMalloc(&dx, (size_t)A * 3 * sizeof(g::real)));
  GK(cudaMemset(dz, 0, P * m.Cz * sizeof(g::real))); GK(cudaMemset(ds, 0, (size_t)T * m.Si * sizeof(g::real)));
  GK(cudaMemset(dx, 0, (size_t)A * 3 * sizeof(g::real)));
  auto fwd = [&] { g::ARENA.reset(); g::ef2::ef2Forward(m, b, z, si, x, (g::real)t, out); };
  ps.zeroGrad();
  g::ARENA.reset();
  auto s = g::ef2::ef2Forward(m, b, z, si, x, (g::real)t, out);
  g::ef2::ef2Backward(m, b, s, z, si, ct, dz, ds, dx);
  g::GradCheck gc;
  gc.add("in/zTrunk", z, dz, P * m.Cz);
  gc.add("in/sInputs", si, ds, (size_t)T * m.Si);
  gc.add("in/x", x, dx, (size_t)A * 3);
  for (auto* p : ps.all) gc.add(*p);
  gc.output(out, ct, (size_t)A * 3);
  failed += gc.run(fwd);
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
