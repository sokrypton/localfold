// The language-model towers' training graph (cuda/grad/plm.cuh), two ways:
//   1. its float32 forward against cuda/plm's own esmcTower on the same bundle (the inference port the page and
//      ESMFold2 run), every layer: the conventions - norms, RoPE, the chain mask, the residual scale - are the port's;
//   2. (-DGRAD_F64) its backward against central differences of its own forward (gradcheck.cuh): ESM-C's blocks
//      and a randomly initialised sequence head with the masked-LM cross entropy, and an ESM2-shaped tower (biases,
//      GELU, no q/k norms, embedding scale).
//
//   nvcc -O2 -std=c++17 -arch=sm_80 -o cuda/grad/test-plm cuda/grad/test_plm.cu -lcublas -lcurand
//   nvcc ... -DGRAD_F64 -o cuda/grad/test-plm-f64 ...
//   cuda/grad/test-plm [--bundle=model-esmc-600m-f32] [--rows=67]
#include "../plm/esmc.cuh"
#include "gradcheck.cuh"
#include "plm.cuh"

namespace g = grad;

__global__ void toRealK(const float* x, g::real* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = (g::real)x[t];
}
static int* upInts(const std::vector<int>& v) {
  int* d; CK(cudaMalloc(&d, v.size() * 4)); CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return d;
}
static void randomise(g::Params& ps, unsigned long long seed, double sd) {
  curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, seed);
  for (auto* p : ps.all) {
    size_t n2 = (p->n + 1) & ~(size_t)1;
    g::real* t; CK(cudaMalloc(&t, n2 * sizeof(g::real)));
    g::randn(gen, t, n2);
    bool scale = p->name.find("scale") != std::string::npos;
    std::vector<g::real> h(p->n);
    CK(cudaMemcpy(h.data(), t, p->n * sizeof(g::real), cudaMemcpyDeviceToHost));
    for (auto& x : h) x = scale ? 1 + 0.1 * x : sd * x;
    CK(cudaMemcpy(p->w, h.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    CK(cudaFree(t));
  }
  curandDestroyGenerator(gen);
}

// the tower and its head over R rows into a loss: sum(out * ct) + the head's cross entropy; gradcheck over everything
static int gradcheckTower(g::plm::Tower& t, int R, const std::vector<int>& ids, const std::vector<int>& seq, const char* what) {
  int* dIds = upInts(ids); int* dSeq = upInts(seq);
  std::vector<int> target(R, -1);
  int count = 0;
  for (int r = 0; r < R; r += 3) { target[r] = (ids[r] * 7 + 3) % t.cfg.vocab; ++count; }
  int* dTarget = upInts(target);
  g::plm::Run run;
  size_t n = (size_t)R * t.cfg.C;
  g::real *ct, *dout, *dhead;
  CK(cudaMalloc(&ct, n * sizeof(g::real))); CK(cudaMalloc(&dout, n * sizeof(g::real))); CK(cudaMalloc(&dhead, n * sizeof(g::real)));
  { curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 5);
    g::randn(gen, ct, (n + 1) & ~(size_t)1); curandDestroyGenerator(gen); }
  // the loss's output tensors for the difference: the tower's output (weighted by ct) and the head's logits
  // (weighted by dlogits is not linear in them), so the head is checked through its own scalar loss: one extra
  // "output" of one element holding the cross entropy
  g::real* ce; CK(cudaMalloc(&ce, sizeof(g::real)));
  g::real one = 1; g::real* dOne; CK(cudaMalloc(&dOne, sizeof(g::real))); CK(cudaMemcpy(dOne, &one, sizeof(g::real), cudaMemcpyHostToDevice));
  auto forward = [&] {
    g::ARENA.reset();
    g::plm::towerForward(t, run, dIds, dSeq, nullptr, R);
    double l = g::plm::headLoss(t, run.out, dTarget, count, dhead, R, false);
    g::real lr = (g::real)l;
    CK(cudaMemcpyAsync(ce, &lr, sizeof(g::real), cudaMemcpyHostToDevice, g::STREAM));
    CK(cudaStreamSynchronize(g::STREAM));
  };
  // and every block input a cotangent of its own, through towerBackward's dstate (ESMFold2's shim reads them all)
  std::vector<g::real*> cs(t.cfg.layers);
  for (int l = 0; l < t.cfg.layers; ++l) {
    CK(cudaMalloc(&cs[l], n * sizeof(g::real)));
    curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 11 + l);
    g::randn(gen, cs[l], (n + 1) & ~(size_t)1); curandDestroyGenerator(gen);
  }
  t.params.zeroGrad();
  g::ARENA.reset();
  g::plm::towerForward(t, run, dIds, dSeq, nullptr, R);
  g::plm::headLoss(t, run.out, dTarget, count, dhead, R);
  g::add(dhead, ct, n);                                      // d(sum(out ct) + ce)/d out
  g::plm::towerBackward(t, run, dhead, dIds, dSeq, nullptr, -1, [&](int l, g::real* dx) { g::add(dx, cs[l], n); });
  g::GradCheck gc;
  for (auto* p : t.params.all) gc.add(*p);
  gc.output(run.out, ct, n);
  gc.output(ce, dOne, 1);
  for (int l = 0; l < t.cfg.layers; ++l) gc.output(run.xs[l], cs[l], n);
  printf("%s: %d layers, C %d, %d rows, %zu tensors, %zu parameters\n", what, t.cfg.layers, t.cfg.C, R,
         t.params.all.size(), t.params.count());
  return gc.run(forward);
}

int main(int argc, char** argv) {
  std::string bundle = "model-esmc-600m-f32";
  int R = 67;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--rows=", 7)) R = atoi(argv[i] + 7);
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  int failed = 0;
  // two chains, so the chain mask is exercised: rows [0, 40) chain 0, the rest chain 1
  std::vector<int> ids(R), seq(R);
  for (int r = 0; r < R; ++r) { ids[r] = 4 + (r * 13 + 5) % 20; seq[r] = r < 40 ? 0 : 1; }
  ids[0] = 0; ids[39] = 2; ids[40] = 0; ids[R - 1] = 2;

#ifndef GRAD_F64
  // ---- 1. the forward against cuda/plm's esmcTower
  M.loadBundle(bundle, "c");
  int layers = 0;
  while (M.has("c/blocks/" + std::to_string(layers) + "/qkv/weights")) ++layers;
  g::plm::Tower t;
  t.cfg.C = (int)dimOf("c/final_norm/scale", 0);
  t.cfg.F = (int)dimOf("c/blocks/0/fc2/weights", 0);
  t.cfg.layers = layers;
  t.cfg.H = t.cfg.C / 64;
  t.cfg.vocab = (int)dimOf("c/embed/weights", 0);
  t.cfg.residual = 1 / sqrt(layers / 36.0);
  g::plm::allocTower(t);
  for (auto* p : t.params.all) toRealK<<<g::blocks(p->n), 256, 0, STREAM>>>(W("c/" + p->name), p->w, p->n);
  int *dIds = upInts(ids), *dSeq = upInts(seq);
  Esmc e{R, t.cfg.C, t.cfg.H, t.cfg.F, layers, 0, (float)sqrt(layers / 36.0)};
  std::vector<float> ref((size_t)R * t.cfg.C);
  esmcTower(e, dIds, dSeq, [&](int k, const float* x) {
    if (k == layers) CK(cudaMemcpyAsync(ref.data(), x, ref.size() * 4, cudaMemcpyDeviceToHost, STREAM));
  });
  CK(cudaStreamSynchronize(STREAM));
  g::plm::Run run;
  g::plm::towerForward(t, run, dIds, dSeq, nullptr, R);
  auto ours = g::host(run.out, ref.size());
  double e1 = g::relRms(ours, ref);
  printf("ESM-C (%s, %d layers, %d rows, 2 chains): training forward against cuda/plm's esmcTower %.3e\n",
         bundle.c_str(), layers, R, e1);
  if (!(e1 < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
#else
  // ---- 2. the backward, in float64, against differences of the same forward
  {
    g::plm::Tower t;
    t.cfg = g::plm::Config{};
    t.cfg.C = 128; t.cfg.H = 2; t.cfg.F = 192; t.cfg.layers = 2; t.cfg.vocab = 33; t.cfg.residual = 0.9;
    t.head.present = true;
    g::plm::allocTower(t);
    randomise(t.params, 1, 0.08);
    for (auto& x : ids) x %= t.cfg.vocab;
    failed += gradcheckTower(t, R, ids, seq, "ESM-C-shaped (swiglu, q/k norms, no biases)");
  }
  {
    g::plm::Tower t;
    t.cfg.C = 128; t.cfg.H = 2; t.cfg.F = 256; t.cfg.layers = 2; t.cfg.vocab = 33;
    t.cfg.swiglu = false; t.cfg.qkNorm = false; t.cfg.bias = true; t.cfg.finalOffset = true; t.cfg.embedScale = 0.88;
    t.head.present = true;
    g::plm::allocTower(t);
    randomise(t.params, 2, 0.08);
    failed += gradcheckTower(t, R, ids, seq, "ESM2-shaped (gelu, biases, no q/k norms)");
  }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
