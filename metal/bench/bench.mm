// metal/bench: GEMM arms timed against each other, interleaved (this laptop's clock drifts over minutes).
//   metal/bench/localfold-bench <out> <rows> <in> [arms: EP bits, comma-separated, default 0,64] [types: hhh|hhf|fhf]
#include "core.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern const char* PORT_SOURCE;
using namespace mt;

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: localfold-bench <out> <rows> <in> [arms] [hhh|hhf|fhf] [reps]\n"); return 1; }
  setSource("bench", PORT_SOURCE);
  int out = atoi(argv[1]); size_t rows = strtoull(argv[2], nullptr, 10); int in = atoi(argv[3]);
  std::vector<int> arms;
  { std::string a = argc > 4 ? argv[4] : "0,64"; size_t p = 0;
    while (p <= a.size()) { size_t q = a.find(',', p); if (q == std::string::npos) q = a.size(); arms.push_back(atoi(a.substr(p, q - p).c_str())); p = q + 1; } }
  std::string types = argc > 5 ? argv[5] : "hhf";
  int reps = argc > 6 ? atoi(argv[6]) : 10;
  DT tx = types[0] == 'h' ? F16 : F32, tw = types[1] == 'h' ? F16 : F32, ty = types[2] == 'h' ? F16 : F32;
  void* X = alloc(rows * in * (tx == F16 ? 2 : 4)); void* W = alloc((size_t)in * out * (tw == F16 ? 2 : 4));
  void* Y = alloc(rows * out * (ty == F16 ? 2 : 4));
  Gemm g{}; g.X = X; g.tx = tx; g.W = W; g.tw = tw; g.Y = Y; g.ty = ty; g.rows = rows; g.in = in; g.out = out; g.half = true;
  double flops = 2.0 * rows * in * out;
  {   // inputs: values, not whatever the allocation held
    std::vector<float> xs(rows * in), ws((size_t)in * out);
    for (size_t i = 0; i < xs.size(); ++i) xs[i] = (float)((i * 2654435761u) % 1000) / 1000.f - 0.5f;
    for (size_t i = 0; i < ws.size(); ++i) ws[i] = ((float)((i * 40503u) % 1000) / 1000.f - 0.5f) * 0.1f;
    if (tx == F16) { std::vector<half> h(xs.begin(), xs.end()); upload(X, h.data(), h.size() * 2); } else upload(X, xs.data(), xs.size() * 4);
    if (tw == F16) { std::vector<half> h(ws.begin(), ws.end()); upload(W, h.data(), h.size() * 2); } else upload(W, ws.data(), ws.size() * 4);
  }
  // every arm's answer against the first arm's: a fast arm that computes something else is not a result
  std::vector<float> first;
  auto answer = [&] {
    std::vector<float> v(rows * out);
    if (ty == F32) download(v.data(), Y, v.size() * 4);
    else { std::vector<half> h(v.size()); download(h.data(), Y, h.size() * 2); for (size_t i = 0; i < v.size(); ++i) v[i] = (float)h[i]; }
    return v;
  };
  std::vector<double> err(arms.size(), 0);
  std::vector<std::vector<double>> times(arms.size());
  for (int round = 0; round < 7; ++round)
    for (size_t k = 0; k < arms.size(); ++k) {
      GEMM_EXTRA_EP = arms[k] < 0 ? 0 : arms[k];
      fill(Y, 0, rows * out * (ty == F16 ? 2 : 4));
      auto runOnce = [&] {
        if (arms[k] < 0) {        // a negative arm: lf_gemm_x, its shape from LF_X=TR,TC,BK,SGR,SGC,HACC
          int tr = 64, tc = 64, bk = 16, sgr = 2, sgc = 2, hacc = 1;
          if (const char* e = getenv("LF_X")) sscanf(e, "%d,%d,%d,%d,%d,%d", &tr, &tc, &bk, &sgr, &sgc, &hacc);
          char name[96], decl[256];
          snprintf(name, sizeof name, "gemmx_%d_%d_%d_%d_%d_%d", tr, tc, bk, sgr, sgc, hacc);
          snprintf(decl, sizeof decl, "template [[host_name(\"%s\")]] kernel void lf_gemm_x<%d, %d, %d, %d, %d, %s>(constant GemmArgs&, uint3, uint, uint);",
                   name, tr, tc, bk, sgr, sgc, hacc ? "true" : "false");
          GemmArgs a{}; a.A = (uint64_t)W; a.B = (uint64_t)X; a.D = (uint64_t)Y; a.m = out; a.n = (int)rows; a.k = in; a.lda = out; a.ldb = in; a.ldd = out;
          if (rows % tr || out % tc || in % bk) { fprintf(stderr, "LF_X: the shape does not divide\n"); exit(1); }
          dispatchInstance(name, decl, &a, sizeof a, Grid{(uint32_t)(out / tc), (uint32_t)(rows / tr), 1}, 32 * sgr * sgc);
        } else gemm(g);
      };
      runOnce(); sync();        // (warm: compiles)
      if (round == 0) {
        std::vector<float> v = answer();
        if (k == 0) first = v;
        double num = 0, den = 0;
        for (size_t i = 0; i < v.size(); ++i) { double d = v[i] - first[i]; num += d * d; den += (double)first[i] * first[i]; }
        err[k] = sqrt(num / std::max(den, 1e-30));
      }
      double t0 = now();
      for (int r = 0; r < reps; ++r) runOnce();
      sync();
      if (round) times[k].push_back((now() - t0) / reps);
    }
  for (size_t k = 0; k < arms.size(); ++k) {
    std::sort(times[k].begin(), times[k].end());
    double t = times[k][times[k].size() / 2];
    printf("  EP+%-4d %8.3f ms  %5.2f TFLOP/s  (relRMS against the first arm %.1e)%s\n", arms[k], t * 1e3, flops / t / 1e12, err[k],
           err[k] > 1e-2 ? "  WRONG" : "");
  }
  return 0;
}
