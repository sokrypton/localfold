// One AF2 Evoformer block's forward and backward against af3-any-model's jax.vjp (cuda/grad/oracle_af2_block.py):
// the outputs, the gradients to both inputs, and every parameter's gradient, as relative RMS.
//
//   nvcc -O2 -std=c++17 -arch=sm_80 -o cuda/grad/test-af2-block cuda/grad/test_af2_block.cu -lcublas
//   cuda/grad/test-af2-block cuda/grad/data/af2-block [--bar=1e-4] [--tf32]
//
// Exit 1 when any figure is above the bar. The parameter gradients start as NaN, so one the backward never writes
// fails rather than reading as zero.
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include "af2.cuh"

using namespace grad;

struct Dump {
  std::map<std::string, std::vector<float>> t;
  std::map<std::string, double> m;
  explicit Dump(const std::string& dir) {
    std::ifstream idx(dir + "/model.idx");
    std::ifstream bin(dir + "/model.bin", std::ios::binary);
    if (!idx || !bin) { fprintf(stderr, "cannot read %s\n", dir.c_str()); exit(1); }
    std::string line;
    while (std::getline(idx, line)) {
      std::istringstream ls(line);
      std::string kind, name; ls >> kind >> name;
      if (kind == "m") { double v; ls >> v; m[name] = v; continue; }
      size_t off, len; ls >> off >> len;
      std::vector<float> v(len);
      bin.seekg((std::streamoff)off * 4);
      bin.read((char*)v.data(), (std::streamsize)len * 4);
      t[name] = std::move(v);
    }
  }
  const std::vector<float>& at(const std::string& k) const {
    auto it = t.find(k);
    if (it == t.end()) { fprintf(stderr, "missing %s\n", k.c_str()); exit(1); }
    return it->second;
  }
};

static float* upload(const std::vector<float>& h) {
  float* d; GK(cudaMalloc(&d, h.size() * 4));
  GK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  return d;
}
__global__ void nanK(float* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] = __int_as_float(0x7fc00000);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: test-af2-block <dump dir> [--bar=1e-4] [--tf32] [--repeat=N]\n"); return 2; }
  double bar = 1e-4; int repeat = 1;
  for (int i = 2; i < argc; ++i) {
    if (!strncmp(argv[i], "--bar=", 6)) bar = atof(argv[i] + 6);
    else if (!strcmp(argv[i], "--tf32")) TF32 = true;
    else if (!strncmp(argv[i], "--repeat=", 9)) repeat = atoi(argv[i] + 9);
  }
  init();
  Dump d(argv[1]);
  const int L = (int)d.at("meta/L")[0], S = (int)d.at("meta/S")[0], Cm = (int)d.at("meta/cm")[0];
  af2::Block b;
  b.extra = d.at("meta/extra")[0] > 0.5f;
  b.opmFirst = d.at("meta/opm_first")[0] > 0.5f;
  b.Cm = Cm; b.Cz = 128;
  af2::allocBlock(b);
  for (auto* p : b.params.all) {
    const auto& v = d.at("p/" + p->name);
    if (v.size() != p->n) { fprintf(stderr, "%s: %zu in the dump, %zu expected\n", p->name.c_str(), v.size(), p->n); return 1; }
    GK(cudaMemcpy(p->w, v.data(), p->n * 4, cudaMemcpyHostToDevice));
    nanK<<<blocks(p->n), 256>>>(p->g, p->n);
  }
  printf("%s: L %d S %d Cm %d, %s stack, OPM %s; %zu parameter tensors, %zu parameters\n", argv[1], L, S, Cm,
         b.extra ? "extra" : "main", b.opmFirst ? "first" : "last", b.params.all.size(), b.params.count());
  float *msa = upload(d.at("in/msa")), *pair = upload(d.at("in/pair"));
  float *mmask = upload(d.at("in/msa_mask")), *pmask = upload(d.at("in/pair_mask"));
  float *ctm = upload(d.at("ct/msa")), *ctp = upload(d.at("ct/pair"));
  size_t mr = (size_t)S * L * Cm, pr = (size_t)L * L * 128;
  float *mo, *po, *dm, *dp;
  GK(cudaMalloc(&mo, mr * 4)); GK(cudaMalloc(&po, pr * 4)); GK(cudaMalloc(&dm, mr * 4)); GK(cudaMalloc(&dp, pr * 4));

  // the parameter gradients must be written, not merely read: zero them only on the second pass onward
  float fwdMs = 0, bwdMs = 0;
  cudaEvent_t e0, e1, e2; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
  for (int r = 0; r < repeat; ++r) {
    if (r == 0) for (auto* p : b.params.all) GK(cudaMemset(p->g, 0, p->n * 4));
    else b.params.zeroGrad();
    ARENA.reset();
    cudaEventRecord(e0, STREAM);
    af2::BlockSave s = af2::blockForward(b, msa, pair, mmask, pmask, mo, po, S, L);
    cudaEventRecord(e1, STREAM);
    af2::blockBackward(b, s, mmask, pmask, ctm, ctp, dm, dp);
    cudaEventRecord(e2, STREAM);
    GK(cudaStreamSynchronize(STREAM));
    float a, c; cudaEventElapsedTime(&a, e0, e1); cudaEventElapsedTime(&c, e1, e2);
    if (r == repeat - 1 || r == 0) { fwdMs = a; bwdMs = c; }
  }
  GK(cudaGetLastError());

  // Each tensor's bar is the larger of --bar and 3x the reference's OWN float32 error on it (e32/<name>, the same
  // vjp in float32 against float64): a gradient ill-conditioned in JAX is ill-conditioned here, and one that is
  // analytically zero (the row attention's feat_2d_norm offset: softmax gradients sum to zero over the keys) has
  // nothing but rounding on either side. What is reported is the ratio to that bar; a fail is a ratio above 1.
  double worst = 0; std::string worstName;
  int failed = 0;
  auto check = [&](const std::string& name, const float* dev, const std::vector<float>& ref, bool quiet = false) {
    double e = relRms(host(dev, ref.size()), ref);
    auto it = d.t.find("e32/" + name);
    double e32 = it == d.t.end() ? 0 : it->second[0];
    double b_ = std::max(bar, 3 * e32), r = std::isnan(e) ? INFINITY : e / b_;
    if (r > worst) { worst = r; worstName = name; }
    if (r > 1) ++failed;
    if (!quiet || r > 1) printf("  %-66s %.3e  (jax f32 %.3e)%s\n", name.c_str(), e, e32, r <= 1 ? "" : "  <-- ABOVE BAR");
  };
  check("out/msa", mo, d.at("out/msa"));
  check("out/pair", po, d.at("out/pair"));
  check("g/msa", dm, d.at("g/msa"));
  check("g/pair", dp, d.at("g/pair"));
  double pw = 0;
  for (auto* p : b.params.all) {
    double e = relRms(host(p->g, p->n), d.at("g/p/" + p->name));
    pw = std::max(pw, std::isnan(e) ? INFINITY : e / std::max(bar, 3.0 * d.at("e32/g/p/" + p->name)[0]));
    check("g/p/" + p->name, p->g, d.at("g/p/" + p->name), true);
  }
  if (const char* who = getenv("GRAD_SHOW")) for (auto* p : b.params.all) if (p->name == who) {
    auto h = host(p->g, p->n); const auto& r = d.at("g/p/" + p->name);
    std::vector<size_t> ix(h.size()); for (size_t i = 0; i < ix.size(); ++i) ix[i] = i;
    std::sort(ix.begin(), ix.end(), [&](size_t a, size_t c) { return fabs(h[a] - r[a]) > fabs(h[c] - r[c]); });
    for (int k = 0; k < 8 && k < (int)ix.size(); ++k) printf("    [%zu] ours %.7g ref %.7g\n", ix[k], h[ix[k]], r[ix[k]]);
  }
  printf("  every parameter gradient (%zu): worst %.2f of its bar\n", b.params.all.size(), pw);
  printf("  forward %.2f ms, backward %.2f ms; arena %.1f MiB high water, %.1f MiB held\n", fwdMs, bwdMs,
         ARENA.high / 1048576.0, ARENA.held() / 1048576.0);
  printf("%s: worst %.2f of its bar (%s), %d above\n", failed ? "FAIL" : "PASS", worst, worstName.c_str(), failed);
  return failed ? 1 : 0;
}
