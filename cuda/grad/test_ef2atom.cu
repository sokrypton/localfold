// ESMFold2's sliding-window atom stack for training (cuda/grad/ef2atom.cuh), two ways:
//   1. float32: c0 and every block of the inputs embedder's stack against cuda/ef2's prepareAtoms + swaBlock in their
//      float32 arm (ATOM_BF16 off), and against the vendor's own dump in that arm (oracle.py --float32-attention:
//      data-*/oracle-f32att's o/atom/norm, o/atom/block<b>), on the exported weights and a real input;
//   2. -DGRAD_F64: the backward against central differences on the same stack, input and layout (the stack's real
//      window, the real valid atoms and rotary table).
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-ef2atom cuda/grad/test_ef2atom.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-ef2atom [--weights=cuda/ef2/weights] [--input=cuda/ef2/data-6mrr]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-ef2atom-f64 cuda/grad/test_ef2atom.cu -lcublas -lcublasLt -ldl -lcurand
#include "../ef2/src/atoms.cuh"
#include "ef2atom.cuh"
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

int main(int argc, char** argv) {
  int failed = 0;
  std::string weights = "cuda/ef2/weights", input = "cuda/ef2/data-6mrr";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.load(weights);
  M.load(input);
  const bool oracle = access((input + "/oracle-f32att/model.idx").c_str(), F_OK) == 0;
  if (oracle) M.load(input + "/oracle-f32att");
  ATOM_BF16 = false;
  const int A = (int)M.len("atom_mask"), nb = (int)M.meta("meta/atomBlocks");
  Atoms at = prepareAtoms(A, "atom");          // the layout and the port's c0
  const int C = at.ctx.C;
  const size_t AC = (size_t)A * C;
  g::ef2::SwaStack st; st.C = C; st.hidden = at.ctx.hidden; st.hw = INPUTS_HALF_WINDOW;
  if (C / g::ef2::HD != at.ctx.heads) { fprintf(stderr, "%d heads of %d, not of 32\n", at.ctx.heads, C / at.ctx.heads); return 1; }
  g::Params ps;
  g::LayerNorm ln; ln.C = C;
  g::Linear lin; lin.K = ATOM_FEATURES; lin.N = C; lin.hasBias = false;
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.len(nm) != len) { fprintf(stderr, "%s: %zu, %zu expected\n", nm.c_str(), M.len(nm), len); exit(1); }
    std::vector<float> h(len); GK(cudaMemcpy(h.data(), W(nm), len * 4, cudaMemcpyDeviceToHost));
    std::vector<g::real> r(h.begin(), h.end());
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  };
  load("f/atom/linear", lin.w, (size_t)ATOM_FEATURES * C);
  load("f/atom/norm/scale", ln.g, C); load("f/atom/norm/offset", ln.b, C);
  g::ef2::forEachParam(st, "f/atom", nb, load);
  g::ef2::AtomLayout L{A, at.ctx.nValid, at.ctx.valid, devReal(at.ctx.mask, A), devReal(at.ctx.cosT, (size_t)A * 16),
                       devReal(at.ctx.sinT, (size_t)A * 16)};
  float* featF = scratch<float>("test.features", (size_t)A * ATOM_FEATURES);
  atomFeaturesK<<<blocks((size_t)A * ATOM_FEATURES), 256, 0, STREAM>>>(W("ref_pos"), W("ref_charge"), at.ctx.mask,
    Idev("ref_element"), Idev("ref_atom_name_chars"), featF, A);
  g::real* feat = devReal(featF, (size_t)A * ATOM_FEATURES);
  g::real *c0, *out;
  GK(cudaMalloc(&c0, AC * sizeof(g::real))); GK(cudaMalloc(&out, AC * sizeof(g::real)));
  printf("ESMFold2 atom stack (%s, %s): %d atoms (%d valid), %d blocks, C %d, %d heads, hidden %d, window +-%d\n",
         weights.c_str(), input.c_str(), A, L.nValid, nb, C, at.ctx.heads, st.hidden, st.hw);
#ifndef GRAD_F64
  g::ARENA.reset();
  g::real* pre = g::ARENA.get(AC); lin.forward(feat, pre, A);
  g::lnForward(ln, pre, c0, A);
  g::ef2::SwaRun r;
  g::ef2::swaStackForward(st, L, c0, c0, out, r);
  // the port, block by block
  float* x = scratch<float>("test.x", AC);
  CK(cudaMemcpyAsync(x, at.c0, AC * 4, cudaMemcpyDeviceToDevice, STREAM));
  float* sc = scratch<float>("test.silu", AC);
  siluK<<<blocks(AC), 256, 0, STREAM>>>(at.c0, sc, AC);
  std::vector<std::vector<float>> portBlocks;
  for (int b = 0; b < nb; ++b) {
    swaBlock(at.ctx, x, at.c0, "atom/blocks/" + std::to_string(b) + "/", INPUTS_HALF_WINDOW, nullptr, sc);
    CK(cudaStreamSynchronize(STREAM));
    portBlocks.push_back(hostF(x, AC));
  }
  auto cmp = [&](const std::string& what, const std::vector<double>& mine, const std::vector<float>& ref) {
    double e = g::relRms(mine, ref);
    printf("  %-34s %.3e\n", what.c_str(), e);
    if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
  };
  auto hv = [&](const g::real* d) { auto v = g::host(d, AC); return std::vector<double>(v.begin(), v.end()); };
  cmp("c0 against the port", hv(c0), hostF(at.c0, AC));
  if (oracle) cmp("c0 against the vendor (f32 attention)", hv(c0), std::vector<float>(M.f("o/atom/norm"), M.f("o/atom/norm") + AC));
  for (int b = 0; b < nb; ++b) {
    cmp("block " + std::to_string(b) + " against the port", hv(r.acts[b + 1]), portBlocks[b]);
    const std::string k = "o/atom/block" + std::to_string(b);
    if (oracle && M.has(k)) cmp("block " + std::to_string(b) + " against the vendor", hv(r.acts[b + 1]), std::vector<float>(M.f(k), M.f(k) + AC));
  }
#else
  auto ctH = gauss(AC, 9, 1);
  g::real* ct = devReal(ctH);
  g::real *dlin_unused = nullptr; (void)dlin_unused;
  auto fwd = [&] {
    g::ARENA.reset();
    g::real* pre = g::ARENA.get(AC); lin.forward(feat, pre, A);
    g::lnForward(ln, pre, c0, A);
    g::ef2::SwaRun r;
    g::ef2::swaStackForward(st, L, c0, c0, out, r);
  };
  ps.zeroGrad();
  g::ARENA.reset();
  g::real* pre = g::ARENA.get(AC); lin.forward(feat, pre, A);
  auto ls = g::lnForward(ln, pre, c0, A);
  g::ef2::SwaRun r;
  g::ef2::swaStackForward(st, L, c0, c0, out, r);
  g::real *dx = g::ARENA.get(AC), *dc0 = g::ARENA.zeros(AC);
  g::ef2::swaStackBackward(st, L, c0, r, ct, dx, dc0);
  g::add(dc0, dx, AC);                         // the stack's input is c0 itself
  g::real* dpre = g::ARENA.get(AC); g::lnBackward(ln, ls, dc0, dpre, A, false);
  lin.backward(feat, dpre, nullptr, A);
  g::GradCheck gc;
  for (auto* p : ps.all) gc.add(*p);
  gc.output(out, ct, AC);
  failed += gc.run(fwd);
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
