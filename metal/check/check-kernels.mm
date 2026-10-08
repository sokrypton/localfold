// Holds each Metal kernel to a CPU reference and times it.
//
//     bash metal/build.sh && metal/check/check-kernels [--tokens=128,256,400] [--rounds=7]
//
// For each token count it runs grid_attend and pair_transition on synthetic
// inputs, compares a sample of rows against a float reference written here
// from the kernels' own specification (not from the kernels), and prints the
// median GPU time of `rounds` command buffers of ten dispatches each. It exits
// 1 if any kernel misses its bound, so it is a gate and not a report.
//
// The bounds follow the PRECISION, not the kernel: q, k, v, the probabilities
// and the transition's normalised rows and intermediate are half, so the
// attention is held to 1e-3 and the transition's UPDATE (out - residual, the
// part the kernel computes) to 1e-4. Both pass at ~3e-4 and ~1e-5.
//
// WebGPU numbers to read these against, same M2, stock flags:
//   grid.attend      tools/gpu/bench-grid-attend-passes.js   1.27 / 12.21 / 35.92 ms at 128 / 256 / 400
//   pair-transition  tools/gpu/bench-transition.js            24.7 ms at 256 (19.2 a call inside a trunk pass)
// They come from another process on a machine that drifts, so a ratio is good
// to about 15%.
#include "../runtime.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

static const char* KERNELS =
#include "../kernels.inc"
    ;

using lfmetal::Runtime;

namespace {

std::mt19937 rng(42);
float uniform(float a) { return std::uniform_real_distribution<float>(-a, a)(rng); }
float roundHalf(float x) { return (float)(__fp16)x; }

std::string option(int argc, char** argv, const std::string& name, const std::string& fallback) {
  std::string prefix = "--" + name + "=";
  for (int i = 1; i < argc; ++i) if (std::string(argv[i]).rfind(prefix, 0) == 0) return argv[i] + prefix.size();
  return fallback;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

bool report(const char* kernel, int tokens, double ms, double flops, double relRms, double bound) {
  bool ok = std::isfinite(relRms) && relRms <= bound;   // a NaN must fail, not slip past a `>`
  std::printf("%-16s N=%4d  %8.3f ms  %5.2f TFLOP/s  relRMS %.2e (bound %.0e)  %s\n", kernel, tokens, ms,
              flops / (ms * 1e-3) / 1e12, relRms, bound, ok ? "ok" : "FAILED");
  return ok;
}

bool checkGridAttend(Runtime& rt, int N, int rounds) {
  const int H = 4, D = 32, size = N * N * H * D;
  std::vector<__fp16> q(size), k(size), v(size);
  for (int x = 0; x < size; ++x) { q[x] = uniform(1); k[x] = uniform(1); v[x] = uniform(1); }
  std::vector<float> bias(H * N * N), mask(N * N, 1.0f);
  for (float& b : bias) b = uniform(1);
  for (int x = 3; x < N * N; x += 17) mask[x] = 0;   // some masked keys in every row
  struct { uint32_t N, H; float scale; } shape = {(uint32_t)N, (uint32_t)H, 1.0f / std::sqrt((float)D)};

  id<MTLBuffer> bq = rt.buffer(size * 2, q.data()), bk = rt.buffer(size * 2, k.data()), bv = rt.buffer(size * 2, v.data());
  id<MTLBuffer> bb = rt.buffer(bias.size() * 4, bias.data()), bm = rt.buffer(mask.size() * 4, mask.data());
  id<MTLBuffer> bo = rt.buffer(size * 2);
  id<MTLComputePipelineState> pso = rt.pipeline("grid_attend_d32", 128);
  auto dispatch = [&](int reps) {
    return rt.run([&](id<MTLComputeCommandEncoder> e) {
      [e setComputePipelineState:pso];
      id<MTLBuffer> buffers[] = {bq, bk, bv, bb, bm, bo};
      for (int x = 0; x < 6; ++x) [e setBuffer:buffers[x] offset:0 atIndex:x];
      [e setBytes:&shape length:sizeof shape atIndex:6];
      for (int r = 0; r < reps; ++r)
        [e dispatchThreadgroups:MTLSizeMake((N + 31) / 32, N, H) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }) / reps;
  };
  dispatch(1);

  const __fp16* out = (const __fp16*)bo.contents;
  double err = 0, norm = 0;
  std::vector<float> logits(N), o(D);
  for (int row : {0, N / 2, N - 1}) for (int h = 0; h < H; ++h) for (int i = 0; i < N; i += std::max(1, N / 37)) {
    for (int j = 0; j < N; ++j) {
      float s = 0;
      for (int d = 0; d < D; ++d) s += (float)q[((row * N + i) * H + h) * D + d] * (float)k[((row * N + j) * H + h) * D + d];
      logits[j] = s * shape.scale + bias[(h * N + i) * N + j] - (mask[row * N + j] <= 0 ? 1e9f : 0.0f);
    }
    float m = *std::max_element(logits.begin(), logits.end()), l = 0;
    std::fill(o.begin(), o.end(), 0.0f);
    for (int j = 0; j < N; ++j) {
      float p = std::exp(logits[j] - m);
      l += p;
      for (int d = 0; d < D; ++d) o[d] += p * (float)v[((row * N + j) * H + h) * D + d];
    }
    for (int d = 0; d < D; ++d) {
      double want = o[d] / l, got = out[((row * N + i) * H + h) * D + d];
      err += (got - want) * (got - want);
      norm += want * want;
    }
  }
  std::vector<double> times;
  for (int r = 0; r < rounds; ++r) times.push_back(dispatch(10));
  double flops = 4.0 * N * (double)N * N * H * D;
  return report("grid_attend", N, median(times), flops, std::sqrt(err / norm), 1e-3);
}

bool checkPairTransition(Runtime& rt, int N, int rounds) {
  const int C = 128, F = 512, rows = N * N;
  const float eps = 1e-5f;
  std::vector<float> pair(rows * C), gamma(C), beta(C);
  for (float& x : pair) x = uniform(2);
  for (int c = 0; c < C; ++c) { gamma[c] = 1 + uniform(0.3f); beta[c] = uniform(0.3f); }
  std::vector<__fp16> w1(C * 2 * F), w2(F * C);
  for (auto& w : w1) w = uniform(0.15f);
  for (auto& w : w2) w = uniform(0.08f);
  struct { uint32_t rows; float eps; } shape = {(uint32_t)rows, eps};

  id<MTLBuffer> bp = rt.buffer(pair.size() * 4), bg = rt.buffer(C * 4, gamma.data()), bb = rt.buffer(C * 4, beta.data());
  id<MTLBuffer> b1 = rt.buffer(w1.size() * 2, w1.data()), b2 = rt.buffer(w2.size() * 2, w2.data());
  id<MTLComputePipelineState> pso = rt.pipeline("pair_transition_c128_f512", 64);
  auto dispatch = [&](int reps) {
    return rt.run([&](id<MTLComputeCommandEncoder> e) {
      [e setComputePipelineState:pso];
      id<MTLBuffer> buffers[] = {bp, bg, bb, b1, b2};
      for (int x = 0; x < 5; ++x) [e setBuffer:buffers[x] offset:0 atIndex:x];
      [e setBytes:&shape length:sizeof shape atIndex:5];
      for (int r = 0; r < reps; ++r)
        [e dispatchThreadgroups:MTLSizeMake((rows + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }) / reps;
  };
  // In place, so the check runs on a fresh copy and the timing on whatever is left.
  std::memcpy(bp.contents, pair.data(), pair.size() * 4);
  dispatch(1);

  const float* out = (const float*)bp.contents;
  double err = 0, norm = 0;
  std::vector<float> y(C), g(F);
  for (int row = 0; row < rows; row += rows / 61) {
    const float* x = &pair[row * C];
    float mean = 0, var = 0;
    for (int c = 0; c < C; ++c) mean += x[c];
    mean /= C;
    for (int c = 0; c < C; ++c) var += (x[c] - mean) * (x[c] - mean);
    var /= C;
    for (int c = 0; c < C; ++c) y[c] = roundHalf((x[c] - mean) / std::sqrt(var + eps) * gamma[c] + beta[c]);
    for (int f = 0; f < F; ++f) {
      float a = 0, b = 0;
      for (int c = 0; c < C; ++c) { a += y[c] * (float)w1[c * 2 * F + f]; b += y[c] * (float)w1[c * 2 * F + F + f]; }
      g[f] = roundHalf(a / (1 + std::exp(-a)) * b);
    }
    for (int c = 0; c < C; ++c) {
      float update = 0;
      for (int f = 0; f < F; ++f) update += g[f] * (float)w2[f * C + c];
      double got = out[row * C + c] - x[c];
      err += (got - update) * (got - update);
      norm += (double)update * update;
    }
  }
  std::vector<double> times;
  for (int r = 0; r < rounds; ++r) times.push_back(dispatch(10));
  double flops = 2.0 * rows * (double)(C * 2 * F + F * C);
  return report("pair_transition", N, median(times), flops, std::sqrt(err / norm), 1e-4);
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    Runtime rt(KERNELS);
    std::printf("%s - kernels compiled from source in %.0f ms\n", rt.name().c_str(), rt.compileMs);
    int rounds = std::stoi(option(argc, argv, "rounds", "7"));
    std::string tokens = option(argc, argv, "tokens", "128,256,400");
    bool ok = true;
    for (size_t at = 0; at < tokens.size();) {
      size_t comma = tokens.find(',', at);
      int N = std::stoi(tokens.substr(at, comma - at));
      ok = checkGridAttend(rt, N, rounds) && ok;
      ok = checkPairTransition(rt, N, rounds) && ok;
      at = comma == std::string::npos ? tokens.size() : comma + 1;
    }
    return ok ? 0 : 1;
  }
}
