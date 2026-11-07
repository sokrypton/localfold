// BF16 mode (bf16 storage and bf16 tensor cores where a module takes them, float accumulation) against float32 on
// the same block, weights and inputs: the AF2 Evoformer block (main and extra stacks) and the AF3 pairformer block,
// with masks that zero some rows and columns. Held by relative rms - the output, the input gradients, and every
// parameter gradient (all of them as one vector, and the worst single tensor) - because bf16's rounding is ~0.4% of
// an element and a 1% defect anywhere is 1% of something.
//   cuda/grad/test-bf16
#include <cstring>
#include <curand.h>
#include "af2.cuh"
#include "af3.cuh"
using namespace grad;

static curandGenerator_t gen;
static float* randn(size_t n, float sd) {
  float* p; GK(cudaMalloc(&p, ((n + 1) & ~1) * 4)); curandGenerateNormal(gen, p, (n + 1) & ~1, 0.f, sd); return p;
}
static std::vector<float> hostv(const float* d, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h;
}
static float* mask(size_t rows, size_t cols, unsigned seed) {   // ~1 in 9 zero, a whole row and column zero
  std::vector<float> h(rows * cols);
  for (size_t i = 0; i < h.size(); ++i) h[i] = ((i * 2654435761u + seed) % 9) && i / cols != 1 && i % cols != 2 ? 1.f : 0.f;
  float* p; GK(cudaMalloc(&p, h.size() * 4)); GK(cudaMemcpy(p, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); return p;
}
struct Acc { double e = 0, y = 0; };
static double rms(const std::vector<float>& a, const std::vector<float>& b, Acc* acc = nullptr) {
  double e = 0, y = 0;
  for (size_t i = 0; i < a.size(); ++i) { e += (double)(a[i] - b[i]) * (a[i] - b[i]); y += (double)b[i] * b[i]; }
  if (acc) { acc->e += e; acc->y += y; }
  return sqrt(e / (y + 1e-30));
}
static int fails = 0;
static void report(const char* what, double r, double bar) {
  printf("  %-60s %.2e%s\n", what, r, r > bar ? "  <-- ABOVE" : "");
  fails += !(r <= bar);
}
// the arms' outputs and gradients, float then bf16, compared. Bars: PyTorch's own bf16 autocast against float32 on
// the same blocks (bench_torch_*_block.py, these shapes) is 2.7e-2 / 1.9e-2 on AF2's input gradients and 2.7e-2 on
// its parameters', 4.3e-3 / 4.9e-3 and 7.4e-3 on AF3's - that much is bf16, not a defect.
struct Run { std::vector<std::vector<float>> out, grads; };
static void compare(const char* name, Run& f, Run& b, const std::vector<Param*>& ps, const char* const* outNames,
                    double outBar, double inBar, double tensorBar) {
  printf("%s\n", name);
  for (size_t i = 0; i < f.out.size(); ++i) report(outNames[i], rms(b.out[i], f.out[i]), i < 2 ? outBar : inBar);
  Acc all; size_t count = 0;
  std::vector<Acc> per(ps.size());
  for (size_t i = 0; i < ps.size(); ++i) { rms(b.grads[i], f.grads[i], &per[i]); all.e += per[i].e; all.y += per[i].y; count += ps[i]->n; }
  report("every parameter gradient", sqrt(all.e / all.y), 1e-2);
  // a tensor's own: floored at 10% of the average gradient element, since some are exactly zero in exact arithmetic
  // (a LayerNorm offset feeding a softmax's logits only through a per-row constant) and rounding is all they hold.
  // AF2's transitions read ~5% here: bf16's rounding moves ReLU inputs across the kink (autocast's do the same)
  double worst = 0; std::string wn;
  for (size_t i = 0; i < ps.size(); ++i) {
    double r = sqrt(per[i].e / std::max(per[i].y, 1e-2 * ps[i]->n * all.y / count));
    if (r > worst) { worst = r; wn = ps[i]->name; }
  }
  char what[160]; snprintf(what, sizeof what, "worst tensor (%s)", wn.c_str());
  report(what, worst, tensorBar);
}

int main(int argc, char** argv) {
  init();
  if (argc > 1 && !strcmp(argv[1], "--no-flash")) FLASH = false;   // (which half of BF16 a difference comes from)
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  // main, extra, main with training dropout (the same masks both arms), and main with its residual states in bf16
  // (Block::bf16Res, opt-in: held to twice the bars - bindcraft2 measured its bf16 chain at twice the error)
  for (int arm = 0; arm < 4; ++arm) {
    const bool extra = arm == 1;
    curandSetPseudoRandomGeneratorSeed(gen, 3);
    const int L = 72, S = 40;
    af2::Block b; b.extra = extra; b.Cm = extra ? 64 : 256;
    b.dropout = arm == 2; b.dropSeed = 9; b.bf16Res = arm == 3;
    af2::forEachParam(b, [&](const std::string& n, Param& p, size_t len) {
      p.name = n; p.n = len; p.w = randn(len, 0.05f); GK(cudaMalloc(&p.g, len * 4)); b.params.add(p); });
    for (auto* p : b.params.all)
      if (p->name.find("/scale") != std::string::npos) { std::vector<float> one(p->n, 1.f); GK(cudaMemcpy(p->w, one.data(), p->n * 4, cudaMemcpyHostToDevice)); }
    size_t mr = (size_t)S * L * b.Cm, pr = (size_t)L * L * 128;
    float *msa = randn(mr, 1.f), *pair = randn(pr, 1.f), *ctm = randn(mr, 1.f), *ctp = randn(pr, 1.f);
    float *mm = mask(S, L, 1), *pm = mask(L, L, 5);
    float *mo, *po, *dm, *dp;
    GK(cudaMalloc(&mo, mr * 4)); GK(cudaMalloc(&po, pr * 4)); GK(cudaMalloc(&dm, mr * 4)); GK(cudaMalloc(&dp, pr * 4));
    Run r[2];
    for (int bfa = 0; bfa < 2; ++bfa) {
      setBf16(bfa == 1);
      b.params.zeroGrad(); ARENA.reset();
      af2::BlockSave s = af2::blockForward(b, msa, pair, mm, pm, mo, po, S, L);
      af2::blockBackward(b, s, mm, pm, ctm, ctp, dm, dp);
      GK(cudaStreamSynchronize(STREAM)); GK(cudaGetLastError());
      r[bfa].out = {hostv(mo, mr), hostv(po, pr), hostv(dm, mr), hostv(dp, pr)};
      for (auto* p : b.params.all) r[bfa].grads.push_back(hostv(p->g, p->n));
    }
    const char* names[] = {"msa out", "pair out", "d msa", "d pair"};
    const double k = arm == 3 ? 2 : 1;
    compare(arm == 1 ? "AF2 extra block (L 72, S 40, Cm 64)" : arm == 2 ? "AF2 block with dropout"
            : arm == 3 ? "AF2 block, bf16 residual states (2x the bars)" : "AF2 block (L 72, S 40, Cm 256)",
            r[0], r[1], b.params.all, names,
            k * 1e-2, k * 3.5e-2, k * 8e-2);
  }
  {
    // A STACK (af2::Stack, 3 blocks). Its BF16 backward keeps the running gradients bf16 from block to block, which a
    // block's own blockBackward would round them to on entry anyway; its keep tiers run the last blocks' backward on
    // their forward's saves, with no replay; and its bf16 carry rounds the residual stream to bf16 between blocks.
    // The first two compute what they replace - not bitwise, the attention's dQ and the LayerNorms' parameter
    // gradients being float atomics, so they are held to the stack's own run-to-run difference - and the carry is
    // held to float32 at the bars the blocks above are.
    curandSetPseudoRandomGeneratorSeed(gen, 5);
    const int L = 40, S = 24, nb = 3;
    af2::Stack st; st.blocks.resize(nb);
    std::vector<Param*> all;
    for (int i = 0; i < nb; ++i) {
      af2::Block& b = st.blocks[i];
      af2::forEachParam(b, [&](const std::string& n, Param& p, size_t len) {
        p.name = n; p.n = len; p.w = randn(len, 0.05f); GK(cudaMalloc(&p.g, len * 4)); b.params.add(p); all.push_back(&p); });
    }
    size_t mr = (size_t)S * L * 256, pr = (size_t)L * L * 128;
    float *msa = randn(mr, 1.f), *pair = randn(pr, 1.f), *ctm = randn(mr, 1.f), *ctp = randn(pr, 1.f);
    float *mm = mask(S, L, 1), *pm = mask(L, L, 5);
    float *dm, *dp, *mo, *po;
    GK(cudaMalloc(&dm, mr * 4)); GK(cudaMalloc(&dp, pr * 4)); GK(cudaMalloc(&mo, mr * 4)); GK(cudaMalloc(&po, pr * 4));
    // arms: 0 1 2 no carry, no keep: the stack, blockBackward chained by hand, the stack again; 3 no carry, keep;
    // 4 5 carry, no keep and keep; 6 float32; 7 on a stream of its own, so the blocks' backwards are CUDA graphs;
    // 8 9 a budget that buys no whole block, only the graded keep (blocks 1 and 2's attention outputs), launched and
    // as graphs
    struct Arm { bool bf, carry, manual; size_t keep; bool stream = false; };
    const Arm arms[10] = {{true, false, false, 0}, {true, false, true, 0}, {true, false, false, 0}, {true, false, false, SIZE_MAX},
                          {true, true, false, 0}, {true, true, false, SIZE_MAX}, {false, false, false, 0},
                          {true, false, false, 0, true}, {true, false, false, 16u << 20}, {true, false, false, 16u << 20, true}};
    Run got[10];
    for (int a = 0; a < 10; ++a) {
      if (arms[a].stream) { GK(cudaStreamCreateWithFlags(&STREAM, cudaStreamNonBlocking)); GB(cublasSetStream(BLAS, STREAM)); }
      setBf16(arms[a].bf);
      st.bf16Carry = arms[a].carry; st.keepBytes = arms[a].keep;
      for (auto& b : st.blocks) b.params.zeroGrad();
      ARENA.reset();
      st.forward(msa, pair, mm, pm, S, L);
      if (arms[a].keep == SIZE_MAX && st.keptBlocks() != nb - 1) { printf("  keep tiers kept %zu of %d blocks\n", st.keptBlocks(), nb - 1); fails++; }
      if (arms[a].keep && arms[a].keep != SIZE_MAX && (st.keptBlocks() != 0 || st.graded != (size_t)nb - 1)) {
        printf("  graded keep: %zu whole, %zu graded (want 0, %d)\n", st.keptBlocks(), st.graded, nb - 1); fails++;
      }
      GK(cudaStreamSynchronize(STREAM));   // (hostv copies on the legacy stream, which does not wait for a non-blocking one)
      got[a].out = {hostv(st.msaIn.back(), mr), hostv(st.pairIn.back(), pr)};
      // (on STREAM: a device-to-device cudaMemcpy returns before it completes, on the legacy stream, which a
      // non-blocking STREAM does not wait for - the backward read the previous arm's gradients one run in four)
      GK(cudaMemcpyAsync(dm, ctm, mr * 4, cudaMemcpyDeviceToDevice, STREAM));
      GK(cudaMemcpyAsync(dp, ctp, pr * 4, cudaMemcpyDeviceToDevice, STREAM));
      if (!arms[a].manual) st.backward(mm, pm, dm, dp);
      else
        for (int i = nb; i-- > 0;) {
          Scratch sc;
          af2::BlockSave s = af2::blockForward(st.blocks[i], st.msaIn[i], st.pairIn[i], mm, pm, mo, po, S, L);
          af2::blockBackward(st.blocks[i], s, mm, pm, dm, dp, dm, dp);
        }
      GK(cudaStreamSynchronize(STREAM)); GK(cudaGetLastError());
      got[a].out.push_back(hostv(dm, mr)); got[a].out.push_back(hostv(dp, pr));
      for (auto* p : all) got[a].grads.push_back(hostv(p->g, p->n));
      if (arms[a].stream) { GK(cudaStreamSynchronize(STREAM)); STREAM = 0; GB(cublasSetStream(BLAS, STREAM)); }
    }
    auto worstRel = [&](const Run& x, const Run& y) {   // the worst output's or tensor's relative rms
      double w = 0;
      for (size_t t = 0; t < x.out.size(); ++t) w = std::max(w, rms(x.out[t], y.out[t]));
      for (size_t t = 0; t < x.grads.size(); ++t) w = std::max(w, rms(x.grads[t], y.grads[t]));
      return w;
    };
    const double self = worstRel(got[0], got[2]);
    printf("AF2 BF16 stack (3 blocks), run to run %.2e; against it, by the same bar (2x + 1e-6):\n", self);
    report("bf16 between blocks, against blockBackward's float", worstRel(got[0], got[1]), 2 * self + 1e-6);
    report("keep tiers, against replaying every block", worstRel(got[0], got[3]), 2 * self + 1e-6);
    report("keep tiers with the bf16 carry, against replaying", worstRel(got[4], got[5]), 2 * self + 1e-6);
    report("each block's backward a CUDA graph, against launched", worstRel(got[0], got[7]), 2 * self + 1e-6);
    report("graded keep (attention outputs only), against replaying", worstRel(got[0], got[8]), 2 * self + 1e-6);
    report("graded keep, each backward a CUDA graph", worstRel(got[0], got[9]), 2 * self + 1e-6);
    const char* names[] = {"msa out", "pair out", "d msa", "d pair"};
    compare("AF2 stack (3 blocks, L 40, S 24) with the bf16 carry", got[6], got[4], all, names, 1e-2, 3.5e-2, 8e-2);
  }
  {
    curandSetPseudoRandomGeneratorSeed(gen, 4);
    const int n = 80, Cz = 128, Cs = 384, Hp = 4, Dp = 32, Hs = 16, Ds = 24;
    af3::Pairformer b; af3::Dialect d; Params ps;
    af3::forEachParam(b, "blk", Cz, Cs, Hp, Dp, Hs, Ds, 4, 4, false, true, [&](const std::string& nm, Param& p, size_t len) {
      p.name = nm; p.n = len; p.w = randn(len, 0.05f); GK(cudaMalloc(&p.g, len * 4)); ps.add(p); });
    for (auto* p : ps.all)
      if (p->name.find("Scale") != std::string::npos) { std::vector<float> one(p->n, 1.f); GK(cudaMemcpy(p->w, one.data(), p->n * 4, cudaMemcpyHostToDevice)); }
    size_t nn = (size_t)n * n;
    float *pair = randn(nn * Cz, 1.f), *sing = randn((size_t)n * Cs, 1.f), *ctp = randn(nn * Cz, 1.f), *cts = randn((size_t)n * Cs, 1.f);
    float *pm = mask(n, n, 7), *seq = mask(1, n, 9);
    float *po, *so, *dp, *ds;
    GK(cudaMalloc(&po, nn * Cz * 4)); GK(cudaMalloc(&so, (size_t)n * Cs * 4)); GK(cudaMalloc(&dp, nn * Cz * 4)); GK(cudaMalloc(&ds, (size_t)n * Cs * 4));
    Run r[2];
    for (int arm = 0; arm < 2; ++arm) {
      setBf16(arm == 1);
      ps.zeroGrad(); ARENA.reset();
      auto s = af3::pairformerForward(b, d, pair, sing, pm, pm, seq, nullptr, po, so, n);
      af3::pairformerBackward(b, d, s, pm, pm, seq, ctp, cts, dp, ds, nullptr);
      GK(cudaStreamSynchronize(STREAM)); GK(cudaGetLastError());
      r[arm].out = {hostv(po, nn * Cz), hostv(so, (size_t)n * Cs), hostv(dp, nn * Cz), hostv(ds, (size_t)n * Cs)};
      for (auto* p : ps.all) r[arm].grads.push_back(hostv(p->g, p->n));
    }
    const char* names[] = {"pair out", "single out", "d pair", "d single"};
    compare("AF3 pairformer block (n 80)", r[0], r[1], ps.all, names, 1e-2, 1e-2, 3e-2);
  }
  printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
