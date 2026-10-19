// metal/design's differential test: every backward here against TWO independent references, in double.
//
//   bash metal/build.sh design && metal/design/localfold-design
//
// WHY TWO. A double-precision analytic reference catches a kernel that does not compute what was derived. It cannot
// catch a DERIVATION that is wrong, because the reference repeats it. So each case is also checked by central
// difference against the FORWARD alone: with S(x) = sum(dy * forward(x)), dS/dx must be the cotangent the kernel
// produced. The forward is the thing metal/core's own selftest already holds to the GPU, so an agreement here means
// the chain rule was applied correctly, not just consistently.
#include "grad.h"
#include "kernels.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

extern const char* PORT_SOURCE;
using namespace mt;

static std::mt19937 rng(11);
static std::vector<double> randv(size_t n, double s = 1.0) {
  std::normal_distribution<double> N(0, s);
  std::vector<double> v(n);
  for (auto& x : v) x = N(rng);
  return v;
}
static std::vector<float> toF(const std::vector<double>& v) {
  std::vector<float> f(v.size());
  for (size_t i = 0; i < v.size(); ++i) f[i] = (float)v[i];
  return f;
}
static int failures = 0;
static void report(const char* what, double rel, double bound) {
  bool ok = rel <= bound && std::isfinite(rel);
  if (!ok) ++failures;
  printf("  %-56s relRMS %.2e %s\n", what, rel, ok ? "ok" : "FAIL");
}
static double relRms(const std::vector<double>& ref, const std::vector<float>& got) {
  double num = 0, den = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = ref[i] - got[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  return sqrt(num / std::max(den, 1e-30));
}
static double relRmsD(const std::vector<double>& ref, const std::vector<double>& got) {
  std::vector<float> f(got.size());
  for (size_t i = 0; i < got.size(); ++i) f[i] = (float)got[i];
  return relRms(ref, f);
}

// ---------------------------------------------------------------- LayerNorm, in double, exactly as the forward is
// y[c] = ((x[c] - mean) * inv) * scale[c] + offset[c], the variance biased, inv = 1/sqrt(var + eps)
static void lnForward(const std::vector<double>& x, size_t rows, int C, const std::vector<double>& scale,
                      const std::vector<double>& offset, double eps, std::vector<double>& y) {
  y.assign(rows * C, 0.0);
  for (size_t r = 0; r < rows; ++r) {
    const double* xr = x.data() + r * C;
    double s = 0;
    for (int c = 0; c < C; ++c) s += xr[c];
    double mean = s / C, v = 0;
    for (int c = 0; c < C; ++c) { double d = xr[c] - mean; v += d * d; }
    double inv = 1.0 / sqrt(v / C + eps);
    for (int c = 0; c < C; ++c) y[r * C + c] = (xr[c] - mean) * inv * scale[c] + offset[c];
  }
}

static void lnBackward(const std::vector<double>& x, const std::vector<double>& dy, size_t rows, int C,
                       const std::vector<double>& scale, double eps, std::vector<double>& dx,
                       std::vector<double>& dscale, std::vector<double>& doffset) {
  dx.assign(rows * C, 0.0);
  dscale.assign(C, 0.0);
  doffset.assign(C, 0.0);
  for (size_t r = 0; r < rows; ++r) {
    const double* xr = x.data() + r * C;
    const double* dyr = dy.data() + r * C;
    double s = 0;
    for (int c = 0; c < C; ++c) s += xr[c];
    double mean = s / C, v = 0;
    for (int c = 0; c < C; ++c) { double d = xr[c] - mean; v += d * d; }
    double inv = 1.0 / sqrt(v / C + eps);
    double mg = 0, mgn = 0;
    for (int c = 0; c < C; ++c) {
      double g = dyr[c] * scale[c], n = (xr[c] - mean) * inv;
      mg += g;
      mgn += g * n;
    }
    mg /= C;
    mgn /= C;
    for (int c = 0; c < C; ++c) {
      double g = dyr[c] * scale[c], n = (xr[c] - mean) * inv;
      dx[r * C + c] = inv * (g - mg - n * mgn);
      dscale[c] += dyr[c] * n;
      doffset[c] += dyr[c];
    }
  }
}

// S(x) = sum(dy * forward(x)), differentiated by central difference - the derivation's own check
static std::vector<double> lnFiniteDiff(std::vector<double> x, const std::vector<double>& dy, size_t rows, int C,
                                        const std::vector<double>& scale, const std::vector<double>& offset,
                                        double eps, const std::vector<size_t>& probes) {
  std::vector<double> out, y;
  auto S = [&](const std::vector<double>& xx) {
    lnForward(xx, rows, C, scale, offset, eps, y);
    double s = 0;
    for (size_t i = 0; i < y.size(); ++i) s += dy[i] * y[i];
    return s;
  };
  for (size_t j : probes) {
    double h = 1e-5 * std::max(1.0, fabs(x[j]));
    double save = x[j];
    x[j] = save + h;
    double a = S(x);
    x[j] = save - h;
    double b = S(x);
    x[j] = save;
    out.push_back((a - b) / (2 * h));
  }
  return out;
}

static void lnCase(size_t rows, int C, double eps) {
  std::vector<double> x = randv(rows * C), dy = randv(rows * C, 0.5),
                      scale = randv(C, 0.7), offset = randv(C, 0.3);
  // a scale near 1, as a trained LayerNorm's is, so the test is not dominated by a pathological weight
  for (auto& s : scale) s += 1.0;

  std::vector<double> rdx, rdscale, rdoffset;
  lnBackward(x, dy, rows, C, scale, eps, rdx, rdscale, rdoffset);

  auto fx = toF(x), fdy = toF(dy), fscale = toF(scale);
  void* dX = uploadNew(fx.data(), fx.size());
  void* dDy = uploadNew(fdy.data(), fdy.size());
  void* dScaleW = uploadNew(fscale.data(), fscale.size());
  void* dDx = alloc(rows * C * 4);
  void* dDscale = alloc((size_t)C * 4);
  void* dDoffset = alloc((size_t)C * 4);

  layerNormBwd((const float*)dX, (const float*)dDy, (float*)dDx, rows, C, (const float*)dScaleW,
               (float*)dDscale, (float*)dDoffset, (float)eps);

  char label[160];
  snprintf(label, sizeof label, "ln bwd dx %zux%d", rows, C);
  report(label, relRms(rdx, download((const float*)dDx, rows * C)), 2e-5);
  snprintf(label, sizeof label, "ln bwd dscale %zux%d", rows, C);
  report(label, relRms(rdscale, download((const float*)dDscale, C)), 2e-5);
  snprintf(label, sizeof label, "ln bwd doffset %zux%d", rows, C);
  report(label, relRms(rdoffset, download((const float*)dDoffset, C)), 2e-5);

  // and the derivation itself, against the forward
  std::vector<size_t> probes;
  std::uniform_int_distribution<size_t> U(0, rows * C - 1);
  for (int i = 0; i < 12; ++i) probes.push_back(U(rng));
  auto fd = lnFiniteDiff(x, dy, rows, C, scale, offset, eps, probes);
  std::vector<double> picked;
  for (size_t j : probes) picked.push_back(rdx[j]);
  snprintf(label, sizeof label, "ln bwd dx vs central difference %zux%d", rows, C);
  report(label, relRmsD(fd, picked), 1e-6);

  release(dX); release(dDy); release(dScaleW); release(dDx); release(dDscale); release(dDoffset);
}

// ---------------------------------------------------------------- a linear layer
static void linCase(size_t rows, int in, int out) {
  std::vector<double> X = randv(rows * in), W = randv((size_t)in * out, 0.1), dY = randv(rows * out, 0.5);

  std::vector<double> rdX(rows * in, 0), rdW((size_t)in * out, 0), rdbias(out, 0);
  for (size_t r = 0; r < rows; ++r)
    for (int i = 0; i < in; ++i) {
      double s = 0;
      for (int o = 0; o < out; ++o) s += dY[r * out + o] * W[(size_t)i * out + o];
      rdX[r * in + i] = s;
    }
  for (int i = 0; i < in; ++i)
    for (int o = 0; o < out; ++o) {
      double s = 0;
      for (size_t r = 0; r < rows; ++r) s += X[r * in + i] * dY[r * out + o];
      rdW[(size_t)i * out + o] = s;
    }
  for (int o = 0; o < out; ++o) {
    double s = 0;
    for (size_t r = 0; r < rows; ++r) s += dY[r * out + o];
    rdbias[o] = s;
  }

  auto fX = toF(X), fW = toF(W), fdY = toF(dY);
  void* dXb = uploadNew(fX.data(), fX.size());
  void* dWb = uploadNew(fW.data(), fW.size());
  void* dYb = uploadNew(fdY.data(), fdY.size());
  void* ddX = alloc(rows * in * 4);
  void* ddW = alloc((size_t)in * out * 4);
  void* ddB = alloc((size_t)out * 4);

  linearBwd((const float*)dXb, (const float*)dWb, (const float*)dYb, (float*)ddX, (float*)ddW, (float*)ddB,
            rows, in, out);

  char label[160];
  snprintf(label, sizeof label, "linear bwd dX %zux%dx%d", rows, in, out);
  report(label, relRms(rdX, download((const float*)ddX, rows * in)), 1e-5);
  snprintf(label, sizeof label, "linear bwd dW %zux%dx%d", rows, in, out);
  report(label, relRms(rdW, download((const float*)ddW, (size_t)in * out)), 1e-5);
  snprintf(label, sizeof label, "linear bwd dbias %zux%dx%d", rows, in, out);
  report(label, relRms(rdbias, download((const float*)ddB, out)), 1e-5);

  release(dXb); release(dWb); release(dYb); release(ddX); release(ddW); release(ddB);
}

int main() {
  setSource("design", PORT_SOURCE);
  printf("LayerNorm backward\n");
  for (auto [rows, C] : std::vector<std::pair<size_t, int>>{{8, 128}, {64, 256}, {333, 128}, {1024, 384}, {7, 64}})
    lnCase(rows, C, 1e-5);
  printf("linear backward\n");
  for (auto [rows, in, out] : std::vector<std::tuple<size_t, int, int>>{
           {64, 128, 128}, {333, 256, 64}, {1024, 128, 256}, {37, 64, 100}})
    linCase(rows, in, out);
  printf(failures ? "\n%d FAILED\n" : "\nall ok\n", failures);
  return failures ? 1 : 0;
}
