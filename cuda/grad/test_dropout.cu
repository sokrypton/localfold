// AF2's training dropout (af2::Block::dropout): the backward regenerates the forward's masks, so the block with
// dropout on is still a function its gradient is the gradient of. Float64: central differences (gradcheck.cuh) on
// every parameter and both inputs, on random weights. Any build: dropout moves the output, and the same seed is the
// same block (the remat replay depends on it).
//   cuda/grad/test-dropout(-f64)
#include <cstring>
#include <random>
#include "af2.cuh"
#include "gradcheck.cuh"
using namespace grad;

static real* upload(const std::vector<real>& h) {
  real* p; GK(cudaMalloc(&p, h.size() * sizeof(real)));
  GK(cudaMemcpy(p, h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice));
  return p;
}
static std::vector<real> hostOf(const real* d, size_t n) {
  std::vector<real> h(n); GK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost)); return h;
}

int main() {
  init();
  std::mt19937 rng(11);
  std::normal_distribution<double> nd(0, 1);
  auto randv = [&](size_t n, double sd) { std::vector<real> v(n); for (auto& x : v) x = (real)(nd(rng) * sd); return v; };
  const int L = 20, S = 5;
  af2::Block b; b.Cm = 256;
  af2::forEachParam(b, [&](const std::string& n, Param& p, size_t len) {
    allocParam(p, n, len);
    auto v = randv(len, n.find("/scale") != std::string::npos ? 0.0 : 0.05);
    if (n.find("/scale") != std::string::npos) for (auto& x : v) x = 1;
    GK(cudaMemcpy(p.w, v.data(), len * sizeof(real), cudaMemcpyHostToDevice));
    b.params.add(p);
  });
  const size_t mr = (size_t)S * L * b.Cm, pr = (size_t)L * L * 128;
  real *msa = upload(randv(mr, 1)), *pair = upload(randv(pr, 1)), *ctm = upload(randv(mr, 1)), *ctp = upload(randv(pr, 1));
  std::vector<real> ones((size_t)L * L, 1);
  real *mm = upload(std::vector<real>(ones.begin(), ones.begin() + (size_t)S * L)), *pm = upload(ones);
  real *mo, *po, *dm, *dp;
  GK(cudaMalloc(&mo, mr * sizeof(real))); GK(cudaMalloc(&po, pr * sizeof(real)));
  GK(cudaMalloc(&dm, mr * sizeof(real))); GK(cudaMalloc(&dp, pr * sizeof(real)));
  int failed = 0;
  auto fwd = [&] { ARENA.reset(); af2::blockForward(b, msa, pair, mm, pm, mo, po, S, L); };
  // dropout moves the output; the same seed gives the same output
  b.dropout = false; fwd(); auto off = hostOf(po, pr);
  b.dropout = true; b.dropSeed = 5; fwd(); auto on1 = hostOf(po, pr);
  fwd(); auto on2 = hostOf(po, pr);
  double moved = 0, norm = 0;
  for (size_t i = 0; i < pr; ++i) { moved += (double)(on1[i] - off[i]) * (on1[i] - off[i]); norm += (double)off[i] * off[i]; }
  const bool same = !memcmp(on1.data(), on2.data(), pr * sizeof(real));
  printf("  dropout moves the pair output by %.2e (relative); the same seed twice: %s\n", sqrt(moved / norm),
         same ? "identical" : "DIFFERENT  <-- ABOVE");
  failed += !same || !(sqrt(moved / norm) > 1e-2);
  // its gradient
  b.params.zeroGrad(); ARENA.reset();
  af2::BlockSave s = af2::blockForward(b, msa, pair, mm, pm, mo, po, S, L);
  af2::blockBackward(b, s, mm, pm, ctm, ctp, dm, dp);
  GK(cudaStreamSynchronize(STREAM)); GK(cudaGetLastError());
  GradCheck gc;
  gc.add("in/msa", msa, dm, mr);
  gc.add("in/pair", pair, dp, pr);
  for (auto* p : b.params.all) gc.add(*p);
  gc.output(mo, ctm, mr);
  gc.output(po, ctp, pr);
  failed += gc.run(fwd);
  printf(failed ? "FAIL\n" : "PASS\n");
  return failed ? 1 : 0;
}
