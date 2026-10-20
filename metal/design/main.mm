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
#include "structgrad.h"
#include "kernels.h"
#include <cmath>
#include <cstdio>
#include <cstring>
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
// A gradient that is ANALYTICALLY ZERO cannot be judged by a relative measure: both sides are rounding noise and
// the ratio is meaningless. feat_2d_norm's OFFSET is exactly such a parameter - it adds a constant to every logit of
// a softmax row, which cancels - so this passes on an absolute floor instead, and says so.
static void reportGrad(const char* what, const std::vector<double>& fd, const std::vector<double>& mine,
                       double bound) {
  double num = 0, den = 0;
  for (size_t i = 0; i < fd.size(); ++i) {
    double d = fd[i] - mine[i];
    num += d * d;
    den += fd[i] * fd[i];
  }
  const double rel = sqrt(num / std::max(den, 1e-30));
  const double absRms = sqrt(num / std::max<size_t>(fd.size(), 1));
  const bool zero = absRms <= 1e-6 && sqrt(den / std::max<size_t>(fd.size(), 1)) <= 1e-6;
  const bool ok = (rel <= bound && std::isfinite(rel)) || zero;
  if (!ok) ++failures;
  if (zero) printf("  %-56s both ~0  %.1e %s\n", what, absRms, ok ? "ok (zero gradient)" : "FAIL");
  else printf("  %-56s relRMS %.2e %s\n", what, rel, ok ? "ok" : "FAIL");
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

// ---------------------------------------------------------------- the elementwise backwards
static void ewiseCases(size_t n) {
  std::vector<double> dy = randv(n), ref = randv(n), y = randv(n), g = randv(n);
  char label[160];

  // relu
  std::vector<double> rdx(n);
  for (size_t i = 0; i < n; ++i) rdx[i] = ref[i] > 0 ? dy[i] : 0.0;
  auto fdy = toF(dy), fref = toF(ref);
  void* dDy = uploadNew(fdy.data(), n);
  void* dRef = uploadNew(fref.data(), n);
  void* dDx = alloc(n * 4);
  reluBwd((const float*)dDy, (const float*)dRef, (float*)dDx, n);
  snprintf(label, sizeof label, "relu bwd n=%zu", n);
  report(label, relRms(rdx, download((const float*)dDx, n)), 1e-6);

  // the gate: out = y * sigmoid(g)
  std::vector<double> rdyv(n), rdgv(n);
  for (size_t i = 0; i < n; ++i) {
    double s = 1.0 / (1.0 + exp(-g[i]));
    rdyv[i] = dy[i] * s;
    rdgv[i] = dy[i] * y[i] * s * (1 - s);
  }
  auto fy = toF(y), fg = toF(g);
  void* dY = uploadNew(fy.data(), n);
  void* dG = uploadNew(fg.data(), n);
  void* dDyO = alloc(n * 4);
  void* dDgO = alloc(n * 4);
  gateBwd((const float*)dDy, (const float*)dY, (const float*)dG, (float*)dDyO, (float*)dDgO, n);
  snprintf(label, sizeof label, "gate bwd dy n=%zu", n);
  report(label, relRms(rdyv, download((const float*)dDyO, n)), 1e-6);
  snprintf(label, sizeof label, "gate bwd dg n=%zu", n);
  report(label, relRms(rdgv, download((const float*)dDgO, n)), 1e-6);

  release(dDy); release(dRef); release(dDx); release(dY); release(dG); release(dDyO); release(dDgO);
}

// ---------------------------------------------------------------- Transition, the whole module
// out = relu(LayerNorm(act) W1 + b1) W2 + b2
struct TrParams {
  std::vector<double> lnScale, lnOffset, W1, b1, W2, b2;
};

static void trForward(const std::vector<double>& act, size_t rows, int C, int H, const TrParams& p,
                      std::vector<double>& ln, std::vector<double>& a1, std::vector<double>& out) {
  lnForward(act, rows, C, p.lnScale, p.lnOffset, 1e-5, ln);
  a1.assign(rows * H, 0.0);
  for (size_t r = 0; r < rows; ++r)
    for (int h = 0; h < H; ++h) {
      double s = p.b1[h];
      for (int c = 0; c < C; ++c) s += ln[r * C + c] * p.W1[(size_t)c * H + h];
      a1[r * H + h] = s > 0 ? s : 0.0;
    }
  out.assign(rows * C, 0.0);
  for (size_t r = 0; r < rows; ++r)
    for (int c = 0; c < C; ++c) {
      double s = p.b2[c];
      for (int h = 0; h < H; ++h) s += a1[r * H + h] * p.W2[(size_t)h * C + c];
      out[r * C + c] = s;
    }
}

static void trCase(size_t rows, int C, int H) {
  std::vector<double> act = randv(rows * C), dOut = randv(rows * C, 0.5);
  TrParams p;
  p.lnScale = randv(C, 0.7);
  for (auto& s : p.lnScale) s += 1.0;
  p.lnOffset = randv(C, 0.3);
  p.W1 = randv((size_t)C * H, 0.15);
  p.b1 = randv(H, 0.1);
  p.W2 = randv((size_t)H * C, 0.15);
  p.b2 = randv(C, 0.1);

  std::vector<double> ln, a1, out;
  trForward(act, rows, C, H, p, ln, a1, out);

  // the analytic backward, in double
  std::vector<double> dh(rows * H, 0), dln(rows * C, 0), rdW2((size_t)H * C, 0), rdb2(C, 0),
      rdW1((size_t)C * H, 0), rdb1(H, 0);
  for (int h = 0; h < H; ++h)
    for (int c = 0; c < C; ++c) {
      double s = 0;
      for (size_t r = 0; r < rows; ++r) s += a1[r * H + h] * dOut[r * C + c];
      rdW2[(size_t)h * C + c] = s;
    }
  for (int c = 0; c < C; ++c) {
    double s = 0;
    for (size_t r = 0; r < rows; ++r) s += dOut[r * C + c];
    rdb2[c] = s;
  }
  for (size_t r = 0; r < rows; ++r)
    for (int h = 0; h < H; ++h) {
      double s = 0;
      for (int c = 0; c < C; ++c) s += dOut[r * C + c] * p.W2[(size_t)h * C + c];
      dh[r * H + h] = a1[r * H + h] > 0 ? s : 0.0;    // relu'(h) == (relu(h) > 0)
    }
  for (int c = 0; c < C; ++c)
    for (int h = 0; h < H; ++h) {
      double s = 0;
      for (size_t r = 0; r < rows; ++r) s += ln[r * C + c] * dh[r * H + h];
      rdW1[(size_t)c * H + h] = s;
    }
  for (int h = 0; h < H; ++h) {
    double s = 0;
    for (size_t r = 0; r < rows; ++r) s += dh[r * H + h];
    rdb1[h] = s;
  }
  for (size_t r = 0; r < rows; ++r)
    for (int c = 0; c < C; ++c) {
      double s = 0;
      for (int h = 0; h < H; ++h) s += dh[r * H + h] * p.W1[(size_t)c * H + h];
      dln[r * C + c] = s;
    }
  std::vector<double> rdAct, rdLnScale, rdLnOffset;
  lnBackward(act, dln, rows, C, p.lnScale, 1e-5, rdAct, rdLnScale, rdLnOffset);

  // the GPU
  auto fact = toF(act), fdOut = toF(dOut), fls = toF(p.lnScale), flo = toF(p.lnOffset),
       fw1 = toF(p.W1), fb1 = toF(p.b1), fw2 = toF(p.W2), fb2 = toF(p.b2);
  void* dAct_ = uploadNew(fact.data(), fact.size());
  void* dOut_ = uploadNew(fdOut.data(), fdOut.size());
  void* dLs = uploadNew(fls.data(), fls.size());
  void* dLo = uploadNew(flo.data(), flo.size());
  void* dW1_ = uploadNew(fw1.data(), fw1.size());
  void* dB1_ = uploadNew(fb1.data(), fb1.size());
  void* dW2_ = uploadNew(fw2.data(), fw2.size());
  void* dB2_ = uploadNew(fb2.data(), fb2.size());
  void* gdAct = alloc(rows * C * 4);
  void* gdLs = alloc((size_t)C * 4);
  void* gdLo = alloc((size_t)C * 4);
  void* gdW1 = alloc((size_t)C * H * 4);
  void* gdB1 = alloc((size_t)H * 4);
  void* gdW2 = alloc((size_t)H * C * 4);
  void* gdB2 = alloc((size_t)C * 4);

  transitionBwd((const float*)dAct_, (const float*)dOut_, rows, C, H,
                (const float*)dLs, (const float*)dLo, (const float*)dW1_, (const float*)dB1_,
                (const float*)dW2_, (const float*)dB2_,
                (float*)gdAct, (float*)gdLs, (float*)gdLo, (float*)gdW1, (float*)gdB1,
                (float*)gdW2, (float*)gdB2);

  char label[160];
  const char* shape = "";
  char sbuf[64];
  snprintf(sbuf, sizeof sbuf, "%zux%dx%d", rows, C, H);
  shape = sbuf;
  snprintf(label, sizeof label, "transition bwd dAct %s", shape);
  report(label, relRms(rdAct, download((const float*)gdAct, rows * C)), 5e-5);
  snprintf(label, sizeof label, "transition bwd dLnScale %s", shape);
  report(label, relRms(rdLnScale, download((const float*)gdLs, C)), 5e-5);
  snprintf(label, sizeof label, "transition bwd dLnOffset %s", shape);
  report(label, relRms(rdLnOffset, download((const float*)gdLo, C)), 5e-5);
  snprintf(label, sizeof label, "transition bwd dW1 %s", shape);
  report(label, relRms(rdW1, download((const float*)gdW1, (size_t)C * H)), 5e-5);
  snprintf(label, sizeof label, "transition bwd db1 %s", shape);
  report(label, relRms(rdb1, download((const float*)gdB1, H)), 5e-5);
  snprintf(label, sizeof label, "transition bwd dW2 %s", shape);
  report(label, relRms(rdW2, download((const float*)gdW2, (size_t)H * C)), 5e-5);
  snprintf(label, sizeof label, "transition bwd db2 %s", shape);
  report(label, relRms(rdb2, download((const float*)gdB2, C)), 5e-5);

  // ---- and the derivation, against the forward alone: S = sum(dOut * out), by central difference.
  // Over the INPUT and over every parameter, because a wrong derivation in one of the seven would otherwise be
  // invisible - the analytic reference above repeats whatever was derived.
  auto S = [&](const std::vector<double>& a, const TrParams& q) {
    std::vector<double> l, h, o;
    trForward(a, rows, C, H, q, l, h, o);
    double s = 0;
    for (size_t i = 0; i < o.size(); ++i) s += dOut[i] * o[i];
    return s;
  };
  auto fdCheck = [&](const char* what, std::vector<double>& target, const std::vector<double>& analytic,
                     const std::vector<double>& base) {
    std::vector<double> fd, got;
    std::uniform_int_distribution<size_t> U(0, target.size() - 1);
    for (int i = 0; i < 8; ++i) {
      size_t j = U(rng);
      double save = target[j], h = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + h;
      double aS = S(act, p);
      target[j] = save - h;
      double bS = S(act, p);
      target[j] = save;
      fd.push_back((aS - bS) / (2 * h));
      got.push_back(analytic[j]);
    }
    (void)base;
    char lab[160];
    snprintf(lab, sizeof lab, "transition %s vs central difference %s", what, shape);
    reportGrad(lab, fd, got, 1e-6);
  };
  fdCheck("dAct", act, rdAct, act);
  fdCheck("dLnScale", p.lnScale, rdLnScale, p.lnScale);
  fdCheck("dLnOffset", p.lnOffset, rdLnOffset, p.lnOffset);
  fdCheck("dW1", p.W1, rdW1, p.W1);
  fdCheck("db1", p.b1, rdb1, p.b1);
  fdCheck("dW2", p.W2, rdW2, p.W2);
  fdCheck("db2", p.b2, rdb2, p.b2);

  release(dAct_); release(dOut_); release(dLs); release(dLo); release(dW1_); release(dB1_);
  release(dW2_); release(dB2_); release(gdAct); release(gdLs); release(gdLo); release(gdW1);
  release(gdB1); release(gdW2); release(gdB2);
}

// ---------------------------------------------------------------- OuterProductMean, the whole module
struct OpmParams {
  std::vector<double> lnScale, lnOffset, Wl, bl, Wr, br, Wo, bo;
};

// out[i][j][f], exactly as modules.py:1157-1212 builds it
static void opmForward(const std::vector<double>& msa, const std::vector<double>& mask, size_t S, int N, int Cm,
                       int O, int F, const OpmParams& p, double eps, std::vector<double>& out) {
  const size_t rows = S * (size_t)N;
  std::vector<double> act;
  lnForward(msa, rows, Cm, p.lnScale, p.lnOffset, 1e-5, act);
  std::vector<double> left(rows * O), right(rows * O);
  for (size_t r = 0; r < rows; ++r)
    for (int c = 0; c < O; ++c) {
      double l = p.bl[c], rr = p.br[c];
      for (int k = 0; k < Cm; ++k) {
        l += act[r * Cm + k] * p.Wl[(size_t)k * O + c];
        rr += act[r * Cm + k] * p.Wr[(size_t)k * O + c];
      }
      left[r * O + c] = l * mask[r];
      right[r * O + c] = rr * mask[r];
    }
  out.assign((size_t)N * N * F, 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      double norm = 0;
      for (size_t s = 0; s < S; ++s) norm += mask[s * N + i] * mask[s * N + j];
      const double inv = 1.0 / (eps + norm);
      for (int f = 0; f < F; ++f) {
        double acc = p.bo[f];
        for (int c = 0; c < O; ++c)
          for (int e = 0; e < O; ++e) {
            double P = 0;
            for (size_t s = 0; s < S; ++s) P += left[(s * N + i) * O + c] * right[(s * N + j) * O + e];
            acc += P * p.Wo[((size_t)c * O + e) * F + f];
          }
        out[(((size_t)i * N) + j) * F + f] = acc * inv;
      }
    }
}

static void opmCase(size_t S, int N, int Cm, int O, int F, int tile) {
  const size_t rows = S * (size_t)N;
  const double eps = 1e-3;
  std::vector<double> msa = randv(rows * Cm), dOut = randv((size_t)N * N * F, 0.5), mask(rows, 1.0);
  // a ragged mask, so the norm and the masking are exercised rather than being all ones
  std::uniform_real_distribution<double> U01(0, 1);
  for (size_t r = 0; r < rows; ++r) mask[r] = U01(rng) < 0.25 ? 0.0 : 1.0;
  for (int i = 0; i < N; ++i) mask[i] = 1.0;          // row 0 all present, so no pair has a zero norm

  OpmParams p;
  p.lnScale = randv(Cm, 0.5);
  for (auto& s : p.lnScale) s += 1.0;
  p.lnOffset = randv(Cm, 0.2);
  p.Wl = randv((size_t)Cm * O, 0.2);
  p.bl = randv(O, 0.1);
  p.Wr = randv((size_t)Cm * O, 0.2);
  p.br = randv(O, 0.1);
  p.Wo = randv((size_t)O * O * F, 0.1);
  p.bo = randv(F, 0.1);

  auto fmsa = toF(msa), fmask = toF(mask), fdOut = toF(dOut), fls = toF(p.lnScale), flo = toF(p.lnOffset),
       fwl = toF(p.Wl), fbl = toF(p.bl), fwr = toF(p.Wr), fbr = toF(p.br), fwo = toF(p.Wo);
  void* dMsa_ = uploadNew(fmsa.data(), fmsa.size());
  void* dMask = uploadNew(fmask.data(), fmask.size());
  void* dOut_ = uploadNew(fdOut.data(), fdOut.size());
  void* dLs = uploadNew(fls.data(), fls.size());
  void* dLo = uploadNew(flo.data(), flo.size());
  void* dWl_ = uploadNew(fwl.data(), fwl.size());
  void* dBl_ = uploadNew(fbl.data(), fbl.size());
  void* dWr_ = uploadNew(fwr.data(), fwr.size());
  void* dBr_ = uploadNew(fbr.data(), fbr.size());
  void* dWo_ = uploadNew(fwo.data(), fwo.size());
  void* gdMsa = alloc(rows * Cm * 4);
  void* gdLs = alloc((size_t)Cm * 4);
  void* gdLo = alloc((size_t)Cm * 4);
  void* gdWl = alloc((size_t)Cm * O * 4);
  void* gdBl = alloc((size_t)O * 4);
  void* gdWr = alloc((size_t)Cm * O * 4);
  void* gdBr = alloc((size_t)O * 4);
  void* gdWo = alloc((size_t)O * O * F * 4);
  void* gdBo = alloc((size_t)F * 4);

  outerProductMeanBwd((const float*)dMsa_, (const float*)dMask, S, N, Cm, O, F,
                      (const float*)dLs, (const float*)dLo, (const float*)dWl_, (const float*)dBl_,
                      (const float*)dWr_, (const float*)dBr_, (const float*)dWo_, (const float*)dOut_,
                      (float*)gdMsa, (float*)gdLs, (float*)gdLo, (float*)gdWl, (float*)gdBl,
                      (float*)gdWr, (float*)gdBr, (float*)gdWo, (float*)gdBo, (float)eps, tile);

  char shape[64];
  snprintf(shape, sizeof shape, "S%zu N%d Cm%d O%d F%d t%d", S, N, Cm, O, F, tile);

  // Only the central difference here: the analytic reference for a four-index contraction is as likely to be wrong
  // as the kernel, and the forward above is the thing worth trusting. Every gradient the module produces is probed.
  auto S_ = [&](const std::vector<double>& m, const OpmParams& q) {
    std::vector<double> o;
    opmForward(m, mask, S, N, Cm, O, F, q, eps, o);
    double acc = 0;
    for (size_t i = 0; i < o.size(); ++i) acc += dOut[i] * o[i];
    return acc;
  };
  auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
    std::vector<double> fd, mine;
    std::uniform_int_distribution<size_t> U(0, target.size() - 1);
    for (int t = 0; t < count; ++t) {
      size_t j = U(rng);
      double save = target[j], h = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + h;
      double aS = S_(msa, p);
      target[j] = save - h;
      double bS = S_(msa, p);
      target[j] = save;
      fd.push_back((aS - bS) / (2 * h));
      mine.push_back(got[j]);
    }
    char lab[200];
    snprintf(lab, sizeof lab, "opm %s vs central difference %s", what, shape);
    reportGrad(lab, fd, mine, 2e-5);
  };

  probe("dMsa", msa, download((const float*)gdMsa, rows * Cm), 6);
  probe("dLnScale", p.lnScale, download((const float*)gdLs, Cm), 4);
  probe("dLnOffset", p.lnOffset, download((const float*)gdLo, Cm), 4);
  probe("dWl", p.Wl, download((const float*)gdWl, (size_t)Cm * O), 4);
  probe("dbl", p.bl, download((const float*)gdBl, O), 3);
  probe("dWr", p.Wr, download((const float*)gdWr, (size_t)Cm * O), 4);
  probe("dbr", p.br, download((const float*)gdBr, O), 3);
  probe("dWo", p.Wo, download((const float*)gdWo, (size_t)O * O * F), 4);
  probe("dbo", p.bo, download((const float*)gdBo, F), 3);

  release(dMsa_); release(dMask); release(dOut_); release(dLs); release(dLo); release(dWl_); release(dBl_);
  release(dWr_); release(dBr_); release(dWo_); release(gdMsa); release(gdLs); release(gdLo); release(gdWl);
  release(gdBl); release(gdWr); release(gdBr); release(gdWo); release(gdBo);
}

// ---------------------------------------------------------------- attention
// out[b][q][h][c] = sum_k softmax_k(dot(q,k) + maskBias[b][k] + pairBias[h][q][k]) * v[b][k][h][c]
static void attnForward(const std::vector<double>& q, const std::vector<double>& k, const std::vector<double>& v,
                        const std::vector<double>& mb, const std::vector<double>& pb,
                        int B, int Q, int K, int H, int D, std::vector<double>& out) {
  auto at = [&](const std::vector<double>& t, int b, int pos, int h, int c, int P) {
    return t[(((size_t)b * P + pos) * H + h) * D + c];
  };
  out.assign((size_t)B * Q * H * D, 0.0);
  std::vector<double> s(K);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int qq = 0; qq < Q; ++qq) {
        double mx = -1e300;
        for (int kk = 0; kk < K; ++kk) {
          double d = 0;
          for (int c = 0; c < D; ++c) d += at(q, b, qq, h, c, Q) * at(k, b, kk, h, c, K);
          d += mb[(size_t)b * K + kk] + pb[((size_t)h * Q + qq) * K + kk];
          s[kk] = d;
          mx = std::max(mx, d);
        }
        double tot = 0;
        for (int kk = 0; kk < K; ++kk) { s[kk] = exp(s[kk] - mx); tot += s[kk]; }
        for (int c = 0; c < D; ++c) {
          double acc = 0;
          for (int kk = 0; kk < K; ++kk) acc += s[kk] * at(v, b, kk, h, c, K);
          out[(((size_t)b * Q + qq) * H + h) * D + c] = acc / tot;
        }
      }
}

static void attnCase(int B, int Q, int K, int H, int D) {
  const size_t qn = (size_t)B * Q * H * D, kn = (size_t)B * K * H * D;
  std::vector<double> q = randv(qn, 0.5), k = randv(kn, 0.5), v = randv(kn), dOut = randv(qn, 0.5),
                      mb = randv((size_t)B * K, 0.5), pb = randv((size_t)H * Q * K, 0.5);
  // a few keys fully masked out, as AF2's mask bias does with -1e9
  for (int b = 0; b < B; ++b)
    for (int kk = 0; kk < K; ++kk)
      if ((b + kk) % 7 == 0) mb[(size_t)b * K + kk] = -1e4;

  auto up = [&](const std::vector<double>& x) { auto f = toF(x); return uploadNew(f.data(), f.size()); };
  void* dQ = up(q); void* dK = up(k); void* dV = up(v); void* dMb = up(mb); void* dPb = up(pb);
  void* dDo = up(dOut);
  void* gOut = alloc(qn * 4);
  void* gL = alloc((size_t)B * H * Q * 4);
  void* gdq = alloc(qn * 4);
  void* gdk = alloc(kn * 4);
  void* gdv = alloc(kn * 4);
  void* gdpb = alloc((size_t)H * Q * K * 4);

  attentionFwdSaved((const float*)dQ, (const float*)dK, (const float*)dV, (const float*)dMb,
                    (const float*)dPb, (float*)gOut, (float*)gL, B, Q, K, H, D);
  // the reference forward must agree first, or the backward is differentiating something else
  std::vector<double> refOut;
  attnForward(q, k, v, mb, pb, B, Q, K, H, D, refOut);
  char label[200];
  snprintf(label, sizeof label, "attn forward B%d Q%d K%d H%d D%d", B, Q, K, H, D);
  report(label, relRms(refOut, download((const float*)gOut, qn)), 2e-6);

  attentionBwd((const float*)dQ, (const float*)dK, (const float*)dV, (const float*)dMb, (const float*)dPb,
               (const float*)gOut, (const float*)gL, (const float*)dDo,
               (float*)gdq, (float*)gdk, (float*)gdv, (float*)gdpb, B, Q, K, H, D);

  char shape[80];
  snprintf(shape, sizeof shape, "B%d Q%d K%d H%d D%d", B, Q, K, H, D);
  auto S_ = [&](const std::vector<double>& qq, const std::vector<double>& kk, const std::vector<double>& vv,
                const std::vector<double>& pbb) {
    std::vector<double> o;
    attnForward(qq, kk, vv, mb, pbb, B, Q, K, H, D, o);
    double acc = 0;
    for (size_t i = 0; i < o.size(); ++i) acc += dOut[i] * o[i];
    return acc;
  };
  auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
    std::vector<double> fd, mine;
    std::uniform_int_distribution<size_t> U(0, target.size() - 1);
    for (int t = 0; t < count; ++t) {
      size_t j = U(rng);
      double save = target[j], hh = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + hh;
      double aS = S_(q, k, v, pb);
      target[j] = save - hh;
      double bS = S_(q, k, v, pb);
      target[j] = save;
      fd.push_back((aS - bS) / (2 * hh));
      mine.push_back(got[j]);
    }
    char lab[220];
    snprintf(lab, sizeof lab, "attn %s vs central difference %s", what, shape);
    reportGrad(lab, fd, mine, 3e-5);
  };
  probe("dq", q, download((const float*)gdq, qn), 6);
  probe("dk", k, download((const float*)gdk, kn), 6);
  probe("dv", v, download((const float*)gdv, kn), 6);
  probe("dPairBias", pb, download((const float*)gdpb, (size_t)H * Q * K), 6);

  release(dQ); release(dK); release(dV); release(dMb); release(dPb); release(dDo);
  release(gOut); release(gL); release(gdq); release(gdk); release(gdv); release(gdpb);
}

// ---------------------------------------------------------------- the Attention module
struct AttnMP { std::vector<double> Wq, Wk, Wv, Wg, bg, Wo, bo; };
static double sigm(double x);

static void attnModForward(const std::vector<double>& qData, const std::vector<double>& mData,
                           const std::vector<double>& mb, const std::vector<double>& pb, const AttnMP& p,
                           bool gating, int B, int Q, int K, int Cq, int Cm, int H, int D, int O,
                           std::vector<double>& out) {
  const int W = H * D;
  const double scale = 1.0 / sqrt((double)D);
  auto proj = [&](const std::vector<double>& X, const std::vector<double>& Wm, const std::vector<double>* b,
                  size_t rows, int in, std::vector<double>& Y) {
    Y.assign(rows * W, 0.0);
    for (size_t r = 0; r < rows; ++r)
      for (int o = 0; o < W; ++o) {
        double s = b ? (*b)[o] : 0.0;
        for (int c = 0; c < in; ++c) s += X[r * in + c] * Wm[(size_t)c * W + o];
        Y[r * W + o] = s;
      }
  };
  std::vector<double> q, k, v, gp;
  proj(qData, p.Wq, nullptr, (size_t)B * Q, Cq, q);
  for (auto& x : q) x *= scale;
  proj(mData, p.Wk, nullptr, (size_t)B * K, Cm, k);
  proj(mData, p.Wv, nullptr, (size_t)B * K, Cm, v);
  std::vector<double> wa;
  attnForward(q, k, v, mb, pb, B, Q, K, H, D, wa);
  if (gating) {
    proj(qData, p.Wg, &p.bg, (size_t)B * Q, Cq, gp);
    for (size_t i = 0; i < wa.size(); ++i) wa[i] *= sigm(gp[i]);
  }
  out.assign((size_t)B * Q * O, 0.0);
  for (size_t r = 0; r < (size_t)B * Q; ++r)
    for (int o = 0; o < O; ++o) {
      double s = p.bo[o];
      for (int c = 0; c < W; ++c) s += wa[r * W + c] * p.Wo[(size_t)c * O + o];
      out[r * O + o] = s;
    }
}

static void attnModCase(int B, int Q, int K, int Cq, int Cm, int H, int D, int O, bool gating, bool self_) {
  const int W = H * D;
  if (self_) { K = Q; Cm = Cq; }
  std::vector<double> qData = randv((size_t)B * Q * Cq), mData;
  if (self_) mData = qData;
  else mData = randv((size_t)B * K * Cm);
  std::vector<double> dOut = randv((size_t)B * Q * O, 0.5),
                      mb((size_t)B * K, 0.0), pb = randv((size_t)H * Q * K, 0.5);
  for (int b = 0; b < B; ++b)
    for (int kk = 0; kk < K; ++kk)
      if ((b + kk) % 5 == 0) mb[(size_t)b * K + kk] = -1e4;

  AttnMP p;
  p.Wq = randv((size_t)Cq * W, 0.2);
  p.Wk = randv((size_t)Cm * W, 0.2);
  p.Wv = randv((size_t)Cm * W, 0.2);
  p.Wg = randv((size_t)Cq * W, 0.2);
  p.bg = randv(W, 0.1);
  p.Wo = randv((size_t)W * O, 0.2);
  p.bo = randv(O, 0.1);

  auto up = [&](const std::vector<double>& x) { auto f = toF(x); return uploadNew(f.data(), f.size()); };
  void* dQd = up(qData);
  void* dMd = self_ ? dQd : up(mData);
  void* dMb = up(mb); void* dPb = up(pb); void* dDo = up(dOut);
  void* hWq = up(p.Wq); void* hWk = up(p.Wk); void* hWv = up(p.Wv);
  void* hWg = up(p.Wg); void* hbg = up(p.bg); void* hWo = up(p.Wo); void* hbo = up(p.bo);

  AttnModuleWeights w{};
  w.Wq = (const float*)hWq; w.Wk = (const float*)hWk; w.Wv = (const float*)hWv;
  if (gating) { w.Wg = (const float*)hWg; w.bg = (const float*)hbg; }
  w.Wo = (const float*)hWo; w.bo = (const float*)hbo;

  void* gQd = alloc((size_t)B * Q * Cq * 4);
  void* gMd = self_ ? gQd : alloc((size_t)B * K * Cm * 4);
  void* gPb = alloc((size_t)H * Q * K * 4);
  AttnModuleGrads gg{};
  void* gWq = alloc((size_t)Cq * W * 4); void* gWk = alloc((size_t)Cm * W * 4);
  void* gWv = alloc((size_t)Cm * W * 4); void* gWg = alloc((size_t)Cq * W * 4);
  void* gbg = alloc((size_t)W * 4); void* gWo = alloc((size_t)W * O * 4); void* gbo = alloc((size_t)O * 4);
  gg.Wq = (float*)gWq; gg.Wk = (float*)gWk; gg.Wv = (float*)gWv;
  if (gating) { gg.Wg = (float*)gWg; gg.bg = (float*)gbg; }
  gg.Wo = (float*)gWo; gg.bo = (float*)gbo;

  attentionModuleBwd((const float*)dQd, (const float*)dMd, (const float*)dMb, (const float*)dPb, w,
                     (const float*)dDo, (float*)gQd, (float*)gMd, (float*)gPb,
                     B, Q, K, Cq, Cm, H, D, O, gg);

  char shape[110];
  snprintf(shape, sizeof shape, "%s%s B%d Q%d K%d H%d D%d O%d", self_ ? "self" : "cross",
           gating ? "+gate" : "", B, Q, K, H, D, O);

  auto S_ = [&]() {
    std::vector<double> o;
    attnModForward(qData, self_ ? qData : mData, mb, pb, p, gating, B, Q, K, Cq, Cm, H, D, O, o);
    double acc = 0;
    for (size_t i = 0; i < o.size(); ++i) acc += dOut[i] * o[i];
    return acc;
  };
  auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
    std::vector<double> fd, mine;
    std::uniform_int_distribution<size_t> U(0, target.size() - 1);
    for (int t = 0; t < count; ++t) {
      size_t j = U(rng);
      double save = target[j], hh = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + hh;
      double aS = S_();
      target[j] = save - hh;
      double bS = S_();
      target[j] = save;
      fd.push_back((aS - bS) / (2 * hh));
      mine.push_back(got[j]);
    }
    char lab[240];
    snprintf(lab, sizeof lab, "attnmod %s vs central difference %s", what, shape);
    reportGrad(lab, fd, mine, 3e-5);
  };

  probe("dQData", qData, download((const float*)gQd, (size_t)B * Q * Cq), 6);
  if (!self_) probe("dMData", mData, download((const float*)gMd, (size_t)B * K * Cm), 5);
  probe("dPairBias", pb, download((const float*)gPb, (size_t)H * Q * K), 5);
  probe("dWq", p.Wq, download((const float*)gWq, (size_t)Cq * W), 4);
  probe("dWk", p.Wk, download((const float*)gWk, (size_t)Cm * W), 4);
  probe("dWv", p.Wv, download((const float*)gWv, (size_t)Cm * W), 4);
  if (gating) {
    probe("dWg", p.Wg, download((const float*)gWg, (size_t)Cq * W), 4);
    probe("dbg", p.bg, download((const float*)gbg, W), 3);
  }
  probe("dWo", p.Wo, download((const float*)gWo, (size_t)W * O), 4);
  probe("dbo", p.bo, download((const float*)gbo, O), 3);

  release(dQd); if (!self_) release(dMd);
  release(dMb); release(dPb); release(dDo);
  release(hWq); release(hWk); release(hWv); release(hWg); release(hbg); release(hWo); release(hbo);
  release(gQd); if (!self_) release(gMd);
  release(gPb); release(gWq); release(gWk); release(gWv); release(gWg); release(gbg); release(gWo); release(gbo);
}

// ------------------------------------------- biased self-attention (MSARowAttentionWithPairBias / TriangleAttention)
static void biasedForward(const std::vector<double>& act, const std::vector<double>& biasSrc,
                          const std::vector<double>& mask, int B, int Q, int Cq, int Cb, int H, int D, int O,
                          bool shared, const std::vector<double>& qScale, const std::vector<double>& qOffset,
                          const std::vector<double>& bScale, const std::vector<double>& bOffset,
                          const std::vector<double>& feat2dW, const AttnMP& ap, bool gating,
                          std::vector<double>& out) {
  const int K = Q;
  const size_t rows = (size_t)B * Q, nbRows = (size_t)Q * K;
  std::vector<double> mb((size_t)B * K);
  for (size_t i = 0; i < mb.size(); ++i) mb[i] = 1e9 * (mask[i] - 1.0);
  std::vector<double> actN, biasN;
  lnForward(act, rows, Cq, qScale, qOffset, 1e-5, actN);
  if (shared || feat2dW.empty()) biasN = actN;
  else lnForward(biasSrc, nbRows, Cb, bScale, bOffset, 1e-5, biasN);
  // an empty feat2dW means the module has no pair bias at all (MSAColumnAttention), and the bias is then zero
  std::vector<double> nb((size_t)H * nbRows, 0.0);
  if (!feat2dW.empty())
    for (size_t r = 0; r < nbRows; ++r)
      for (int h = 0; h < H; ++h) {
        double s = 0;
        for (int c = 0; c < Cb; ++c) s += biasN[r * Cb + c] * feat2dW[(size_t)c * H + h];
        nb[(size_t)h * nbRows + r] = s;        // [h][q][k]
      }
  attnModForward(actN, actN, mb, nb, ap, gating, B, Q, K, Cq, Cq, H, D, O, out);
}

static void biasedCase(int B, int Q, int Cq, int Cb, int H, int D, int O, bool shared) {
  const int K = Q, W = H * D;
  const size_t rows = (size_t)B * Q, nbRows = (size_t)Q * K;
  if (shared) { Cb = Cq; }
  std::vector<double> act = randv(rows * Cq), biasSrc;
  if (shared) { if (rows != nbRows) { printf("  (shared needs B == Q)\n"); return; } biasSrc = act; }
  else biasSrc = randv(nbRows * Cb);
  std::vector<double> dOut = randv(rows * O, 0.5), mask((size_t)B * K, 1.0);
  for (int b = 0; b < B; ++b)
    for (int kk = 1; kk < K; ++kk)
      if ((b * 3 + kk) % 6 == 0) mask[(size_t)b * K + kk] = 0.0;    // key 0 always present

  std::vector<double> qScale = randv(Cq, 0.3), qOffset = randv(Cq, 0.2),
                      bScale = randv(Cb, 0.3), bOffset = randv(Cb, 0.2),
                      feat2dW = randv((size_t)Cb * H, 0.3);
  for (auto& s : qScale) s += 1.0;
  for (auto& s : bScale) s += 1.0;
  AttnMP ap;
  ap.Wq = randv((size_t)Cq * W, 0.2); ap.Wk = randv((size_t)Cq * W, 0.2); ap.Wv = randv((size_t)Cq * W, 0.2);
  ap.Wg = randv((size_t)Cq * W, 0.2); ap.bg = randv(W, 0.1);
  ap.Wo = randv((size_t)W * O, 0.2);  ap.bo = randv(O, 0.1);

  auto up = [&](const std::vector<double>& x) { auto f = toF(x); return uploadNew(f.data(), f.size()); };
  void* dAct = up(act);
  void* dBs = shared ? dAct : up(biasSrc);
  void* dMask = up(mask); void* dDo = up(dOut);
  void* hqs = up(qScale); void* hqo = up(qOffset); void* hbs = up(bScale); void* hbo_ = up(bOffset);
  void* hf2 = up(feat2dW);
  void* hWq = up(ap.Wq); void* hWk = up(ap.Wk); void* hWv = up(ap.Wv);
  void* hWg = up(ap.Wg); void* hbg = up(ap.bg); void* hWo = up(ap.Wo); void* hbo = up(ap.bo);

  BiasedAttnWeights w{};
  w.qScale = (const float*)hqs; w.qOffset = (const float*)hqo;
  if (!shared) { w.bScale = (const float*)hbs; w.bOffset = (const float*)hbo_; }
  w.feat2dW = (const float*)hf2;
  w.attn.Wq = (const float*)hWq; w.attn.Wk = (const float*)hWk; w.attn.Wv = (const float*)hWv;
  w.attn.Wg = (const float*)hWg; w.attn.bg = (const float*)hbg;
  w.attn.Wo = (const float*)hWo; w.attn.bo = (const float*)hbo;

  void* gAct = alloc(rows * Cq * 4);
  void* gBs = shared ? gAct : alloc(nbRows * Cb * 4);
  void* gqs = alloc((size_t)Cq * 4); void* gqo = alloc((size_t)Cq * 4);
  void* gbs = alloc((size_t)Cb * 4); void* gbo_ = alloc((size_t)Cb * 4);
  void* gf2 = alloc((size_t)Cb * H * 4);
  void* gWq = alloc((size_t)Cq * W * 4); void* gWk = alloc((size_t)Cq * W * 4);
  void* gWv = alloc((size_t)Cq * W * 4); void* gWg = alloc((size_t)Cq * W * 4);
  void* gbg = alloc((size_t)W * 4); void* gWo = alloc((size_t)W * O * 4); void* gbo = alloc((size_t)O * 4);
  BiasedAttnGrads gg{};
  gg.qScale = (float*)gqs; gg.qOffset = (float*)gqo;
  if (!shared) { gg.bScale = (float*)gbs; gg.bOffset = (float*)gbo_; }
  gg.feat2dW = (float*)gf2;
  gg.attn.Wq = (float*)gWq; gg.attn.Wk = (float*)gWk; gg.attn.Wv = (float*)gWv;
  gg.attn.Wg = (float*)gWg; gg.attn.bg = (float*)gbg;
  gg.attn.Wo = (float*)gWo; gg.attn.bo = (float*)gbo;

  biasedSelfAttentionBwd((const float*)dAct, (const float*)dBs, (const float*)dMask,
                         B, Q, Cq, Cb, H, D, O, w, (const float*)dDo,
                         (float*)gAct, (float*)gBs, gg);

  char shape[120];
  snprintf(shape, sizeof shape, "%s B%d Q%d Cq%d Cb%d H%d D%d O%d",
           shared ? "triangle-like" : "msarow-like", B, Q, Cq, Cb, H, D, O);
  auto S_ = [&]() {
    std::vector<double> o;
    biasedForward(act, shared ? act : biasSrc, mask, B, Q, Cq, Cb, H, D, O, shared,
                  qScale, qOffset, bScale, bOffset, feat2dW, ap, true, o);
    double acc = 0;
    for (size_t i = 0; i < o.size(); ++i) acc += dOut[i] * o[i];
    return acc;
  };
  auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
    std::vector<double> fd, mine;
    std::uniform_int_distribution<size_t> U(0, target.size() - 1);
    for (int t = 0; t < count; ++t) {
      size_t j = U(rng);
      double save = target[j], hh = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + hh;
      double aS = S_();
      target[j] = save - hh;
      double bS = S_();
      target[j] = save;
      fd.push_back((aS - bS) / (2 * hh));
      mine.push_back(got[j]);
    }
    char lab[250];
    snprintf(lab, sizeof lab, "biasedattn %s vs central difference %s", what, shape);
    reportGrad(lab, fd, mine, 3e-5);
  };

  probe("dAct", act, download((const float*)gAct, rows * Cq), 6);
  if (!shared) probe("dBiasSrc", biasSrc, download((const float*)gBs, nbRows * Cb), 5);
  probe("dqScale", qScale, download((const float*)gqs, Cq), 3);
  probe("dqOffset", qOffset, download((const float*)gqo, Cq), 3);
  if (!shared) {
    probe("dbScale", bScale, download((const float*)gbs, Cb), 3);
    probe("dbOffset", bOffset, download((const float*)gbo_, Cb), 3);
  }
  probe("dfeat2dW", feat2dW, download((const float*)gf2, (size_t)Cb * H), 4);
  probe("dWq", ap.Wq, download((const float*)gWq, (size_t)Cq * W), 3);
  probe("dWv", ap.Wv, download((const float*)gWv, (size_t)Cq * W), 3);
  probe("dWg", ap.Wg, download((const float*)gWg, (size_t)Cq * W), 3);
  probe("dWo", ap.Wo, download((const float*)gWo, (size_t)W * O), 3);

  release(dAct); if (!shared) release(dBs);
  release(dMask); release(dDo); release(hqs); release(hqo); release(hbs); release(hbo_); release(hf2);
  release(hWq); release(hWk); release(hWv); release(hWg); release(hbg); release(hWo); release(hbo);
  release(gAct); if (!shared) release(gBs);
  release(gqs); release(gqo); release(gbs); release(gbo_); release(gf2);
  release(gWq); release(gWk); release(gWv); release(gWg); release(gbg); release(gWo); release(gbo);
}

// ---------------------------------------------------------------- MSAColumnAttention
static void mcolCase(int S, int N, int Cm, int H, int D, int O) {
  const int W = H * D;
  const size_t rows = (size_t)S * N;
  std::vector<double> msa = randv(rows * Cm), dOut = randv(rows * O, 0.5), mask(rows, 1.0);
  for (int s = 0; s < S; ++s)
    for (int n = 1; n < N; ++n)
      if ((s * 2 + n) % 5 == 0) mask[(size_t)s * N + n] = 0.0;      // sequence 0 always present
  std::vector<double> qScale = randv(Cm, 0.3), qOffset = randv(Cm, 0.2);
  for (auto& x : qScale) x += 1.0;
  AttnMP ap;
  ap.Wq = randv((size_t)Cm * W, 0.2); ap.Wk = randv((size_t)Cm * W, 0.2); ap.Wv = randv((size_t)Cm * W, 0.2);
  ap.Wg = randv((size_t)Cm * W, 0.2); ap.bg = randv(W, 0.1);
  ap.Wo = randv((size_t)W * O, 0.2);  ap.bo = randv(O, 0.1);

  auto up = [&](const std::vector<double>& x) { auto f = toF(x); return uploadNew(f.data(), f.size()); };
  void* dMsaIn = up(msa); void* dMask = up(mask); void* dDo = up(dOut);
  void* hqs = up(qScale); void* hqo = up(qOffset);
  void* hWq = up(ap.Wq); void* hWk = up(ap.Wk); void* hWv = up(ap.Wv);
  void* hWg = up(ap.Wg); void* hbg = up(ap.bg); void* hWo = up(ap.Wo); void* hbo = up(ap.bo);
  BiasedAttnWeights w{};
  w.qScale = (const float*)hqs; w.qOffset = (const float*)hqo; w.feat2dW = nullptr;
  w.attn.Wq = (const float*)hWq; w.attn.Wk = (const float*)hWk; w.attn.Wv = (const float*)hWv;
  w.attn.Wg = (const float*)hWg; w.attn.bg = (const float*)hbg;
  w.attn.Wo = (const float*)hWo; w.attn.bo = (const float*)hbo;
  void* gMsa = alloc(rows * Cm * 4);
  void* gqs = alloc((size_t)Cm * 4); void* gqo = alloc((size_t)Cm * 4);
  void* gWq = alloc((size_t)Cm * W * 4); void* gWv = alloc((size_t)Cm * W * 4);
  void* gWo = alloc((size_t)W * O * 4);
  BiasedAttnGrads gg{};
  gg.qScale = (float*)gqs; gg.qOffset = (float*)gqo;
  gg.attn.Wq = (float*)gWq; gg.attn.Wv = (float*)gWv; gg.attn.Wo = (float*)gWo;

  msaColumnAttentionBwd((const float*)dMsaIn, (const float*)dMask, S, N, Cm, H, D, O, w,
                        (const float*)dDo, (float*)gMsa, gg);

  // the host forward: the same module with the first two axes swapped, which is what AF2 does
  auto fwd = [&]() {
    std::vector<double> actT(rows * Cm), maskT(rows), out;
    for (int s = 0; s < S; ++s)
      for (int n = 0; n < N; ++n) {
        maskT[(size_t)n * S + s] = mask[(size_t)s * N + n];
        for (int c = 0; c < Cm; ++c) actT[((size_t)n * S + s) * Cm + c] = msa[((size_t)s * N + n) * Cm + c];
      }
    std::vector<double> none;
    biasedForward(actT, actT, maskT, N, S, Cm, Cm, H, D, O, true, qScale, qOffset, none, none,
                  none, ap, true, out);
    double acc = 0;                           // the cotangent is in the ORIGINAL orientation
    for (int s = 0; s < S; ++s)
      for (int n = 0; n < N; ++n)
        for (int o = 0; o < O; ++o)
          acc += dOut[((size_t)s * N + n) * O + o] * out[((size_t)n * S + s) * O + o];
    return acc;
  };
  char shape[80];
  snprintf(shape, sizeof shape, "S%d N%d Cm%d H%d D%d O%d", S, N, Cm, H, D, O);
  auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
    std::vector<double> fd, mine;
    std::uniform_int_distribution<size_t> U(0, target.size() - 1);
    for (int t = 0; t < count; ++t) {
      size_t j = U(rng);
      double save = target[j], hh = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + hh;
      double aS = fwd();
      target[j] = save - hh;
      double bS = fwd();
      target[j] = save;
      fd.push_back((aS - bS) / (2 * hh));
      mine.push_back(got[j]);
    }
    char lab[220];
    snprintf(lab, sizeof lab, "msacol %s vs central difference %s", what, shape);
    reportGrad(lab, fd, mine, 3e-5);
  };
  probe("dMsa", msa, download((const float*)gMsa, rows * Cm), 6);
  probe("dqScale", qScale, download((const float*)gqs, Cm), 3);
  probe("dqOffset", qOffset, download((const float*)gqo, Cm), 3);
  probe("dWq", ap.Wq, download((const float*)gWq, (size_t)Cm * W), 3);
  probe("dWv", ap.Wv, download((const float*)gWv, (size_t)Cm * W), 3);
  probe("dWo", ap.Wo, download((const float*)gWo, (size_t)W * O), 3);

  release(dMsaIn); release(dMask); release(dDo); release(hqs); release(hqo);
  release(hWq); release(hWk); release(hWv); release(hWg); release(hbg); release(hWo); release(hbo);
  release(gMsa); release(gqs); release(gqo); release(gWq); release(gWv); release(gWo);
}

// ---------------------------------------------------------------- MSAColumnGlobalAttention
struct GattnP { std::vector<double> qScale, qOffset, Wq, Wk, Wv, Wg, bg, Wo, bo; };

// modules.py:432-497 and :640-669, in double. msa is [S][N][Cm]; the module swaps to [N][S][Cm] first.
static void gattnForward(const std::vector<double>& msa, const std::vector<double>& mask,
                         int S, int N, int Cm, int H, int D, int O, const GattnP& p,
                         std::vector<double>& out) {
  const int B = N, K = S, W = H * D;
  const size_t rows = (size_t)B * K;
  std::vector<double> actT(rows * Cm), maskT(rows);
  for (int s = 0; s < S; ++s)
    for (int n = 0; n < N; ++n) {
      maskT[(size_t)n * S + s] = mask[(size_t)s * N + n];
      for (int c = 0; c < Cm; ++c) actT[((size_t)n * S + s) * Cm + c] = msa[((size_t)s * N + n) * Cm + c];
    }
  std::vector<double> actN;
  lnForward(actT, rows, Cm, p.qScale, p.qOffset, 1e-5, actN);

  std::vector<double> qavg((size_t)B * Cm, 0.0);
  for (int b = 0; b < B; ++b) {
    double den = 0;
    for (int k = 0; k < K; ++k) den += maskT[(size_t)b * K + k];
    for (int c = 0; c < Cm; ++c) {
      double num = 0;
      for (int k = 0; k < K; ++k) num += maskT[(size_t)b * K + k] * actN[((size_t)b * K + k) * Cm + c];
      qavg[(size_t)b * Cm + c] = num / (den + 1e-10);
    }
  }
  const double scale = 1.0 / sqrt((double)D);
  std::vector<double> q((size_t)B * W, 0.0), kk(rows * D, 0.0), vv(rows * D, 0.0);
  for (int b = 0; b < B; ++b)
    for (int o = 0; o < W; ++o) {
      double s2 = 0;
      for (int c = 0; c < Cm; ++c) s2 += qavg[(size_t)b * Cm + c] * p.Wq[(size_t)c * W + o];
      q[(size_t)b * W + o] = s2 * scale;
    }
  for (size_t r = 0; r < rows; ++r)
    for (int c = 0; c < D; ++c) {
      double a1 = 0, a2 = 0;
      for (int e = 0; e < Cm; ++e) {
        a1 += actN[r * Cm + e] * p.Wk[(size_t)e * D + c];
        a2 += actN[r * Cm + e] * p.Wv[(size_t)e * D + c];
      }
      kk[r * D + c] = a1;
      vv[r * D + c] = a2;
    }
  std::vector<double> wa((size_t)B * W, 0.0), logit(K);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      double mx = -1e300;
      for (int k = 0; k < K; ++k) {
        double d = 1e9 * (maskT[(size_t)b * K + k] - 1.0);
        for (int c = 0; c < D; ++c) d += q[(size_t)b * W + h * D + c] * kk[((size_t)b * K + k) * D + c];
        logit[k] = d;
        mx = std::max(mx, d);
      }
      double tot = 0;
      for (int k = 0; k < K; ++k) { logit[k] = exp(logit[k] - mx); tot += logit[k]; }
      for (int c = 0; c < D; ++c) {
        double acc = 0;
        for (int k = 0; k < K; ++k) acc += logit[k] * vv[((size_t)b * K + k) * D + c];
        wa[(size_t)b * W + h * D + c] = acc / tot;
      }
    }
  // the gate broadcasts that single query back over every position
  std::vector<double> wag(rows * W, 0.0);
  for (int b = 0; b < B; ++b)
    for (int k = 0; k < K; ++k)
      for (int o = 0; o < W; ++o) {
        double gp = p.bg[o];
        for (int c = 0; c < Cm; ++c) gp += actN[((size_t)b * K + k) * Cm + c] * p.Wg[(size_t)c * W + o];
        wag[((size_t)b * K + k) * W + o] = wa[(size_t)b * W + o] * sigm(gp);
      }
  out.assign(rows * O, 0.0);
  for (size_t r = 0; r < rows; ++r)
    for (int o = 0; o < O; ++o) {
      double s2 = p.bo[o];
      for (int c = 0; c < W; ++c) s2 += wag[r * W + c] * p.Wo[(size_t)c * O + o];
      out[r * O + o] = s2;
    }
}

static void gattnCase(int S, int N, int Cm, int H, int D, int O) {
  const int W = H * D;
  const size_t rows = (size_t)S * N;
  std::vector<double> msa = randv(rows * Cm), dOut = randv(rows * O, 0.5), mask(rows, 1.0);
  for (int s = 1; s < S; ++s)
    for (int n = 0; n < N; ++n)
      if ((s + n) % 4 == 0) mask[(size_t)s * N + n] = 0.0;      // sequence 0 always present
  GattnP p;
  p.qScale = randv(Cm, 0.3); for (auto& x : p.qScale) x += 1.0;
  p.qOffset = randv(Cm, 0.2);
  p.Wq = randv((size_t)Cm * W, 0.2);
  p.Wk = randv((size_t)Cm * D, 0.2);
  p.Wv = randv((size_t)Cm * D, 0.2);
  p.Wg = randv((size_t)Cm * W, 0.2); p.bg = randv(W, 0.1);
  p.Wo = randv((size_t)W * O, 0.2);  p.bo = randv(O, 0.1);

  auto up = [&](const std::vector<double>& x) { auto f = toF(x); return uploadNew(f.data(), f.size()); };
  void* dMsaIn = up(msa); void* dMask = up(mask); void* dDo = up(dOut);
  void* hqs = up(p.qScale); void* hqo = up(p.qOffset); void* hWq = up(p.Wq);
  void* hWk = up(p.Wk); void* hWv = up(p.Wv); void* hWg = up(p.Wg); void* hbg = up(p.bg);
  void* hWo = up(p.Wo); void* hbo = up(p.bo);
  GlobalAttnWeights w{};
  w.qScale = (const float*)hqs; w.qOffset = (const float*)hqo; w.Wq = (const float*)hWq;
  w.Wk = (const float*)hWk; w.Wv = (const float*)hWv;
  w.Wg = (const float*)hWg; w.bg = (const float*)hbg;
  w.Wo = (const float*)hWo; w.bo = (const float*)hbo;
  void* gMsa = alloc(rows * Cm * 4);
  void* gqs = alloc((size_t)Cm * 4); void* gqo = alloc((size_t)Cm * 4);
  void* gWq = alloc((size_t)Cm * W * 4);
  void* gWk = alloc((size_t)Cm * D * 4); void* gWv = alloc((size_t)Cm * D * 4);
  void* gWg = alloc((size_t)Cm * W * 4); void* gbg = alloc((size_t)W * 4);
  void* gWo = alloc((size_t)W * O * 4); void* gbo = alloc((size_t)O * 4);
  GlobalAttnGrads gg{};
  gg.qScale = (float*)gqs; gg.qOffset = (float*)gqo;
  gg.Wq = (float*)gWq; gg.Wk = (float*)gWk; gg.Wv = (float*)gWv;
  gg.Wg = (float*)gWg; gg.bg = (float*)gbg; gg.Wo = (float*)gWo; gg.bo = (float*)gbo;

  msaColumnGlobalAttentionBwd((const float*)dMsaIn, (const float*)dMask, S, N, Cm, H, D, O, w,
                              (const float*)dDo, (float*)gMsa, gg);

  char shape[90];
  snprintf(shape, sizeof shape, "S%d N%d Cm%d H%d D%d O%d", S, N, Cm, H, D, O);
  auto S_ = [&]() {
    std::vector<double> o;
    gattnForward(msa, mask, S, N, Cm, H, D, O, p, o);
    double acc = 0;                           // o is [N][S][O]; the cotangent is [S][N][O]
    for (int s = 0; s < S; ++s)
      for (int n = 0; n < N; ++n)
        for (int oo = 0; oo < O; ++oo)
          acc += dOut[((size_t)s * N + n) * O + oo] * o[((size_t)n * S + s) * O + oo];
    return acc;
  };
  auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
    std::vector<double> fd, mine;
    std::uniform_int_distribution<size_t> U(0, target.size() - 1);
    for (int t = 0; t < count; ++t) {
      size_t j = U(rng);
      double save = target[j], hh = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + hh;
      double aS = S_();
      target[j] = save - hh;
      double bS = S_();
      target[j] = save;
      fd.push_back((aS - bS) / (2 * hh));
      mine.push_back(got[j]);
    }
    char lab[230];
    snprintf(lab, sizeof lab, "gattn %s vs central difference %s", what, shape);
    reportGrad(lab, fd, mine, 3e-5);
  };
  probe("dMsa", msa, download((const float*)gMsa, rows * Cm), 6);
  probe("dqScale", p.qScale, download((const float*)gqs, Cm), 3);
  probe("dqOffset", p.qOffset, download((const float*)gqo, Cm), 3);
  probe("dWq", p.Wq, download((const float*)gWq, (size_t)Cm * W), 4);
  probe("dWk", p.Wk, download((const float*)gWk, (size_t)Cm * D), 4);
  probe("dWv", p.Wv, download((const float*)gWv, (size_t)Cm * D), 4);
  probe("dWg", p.Wg, download((const float*)gWg, (size_t)Cm * W), 4);
  probe("dbg", p.bg, download((const float*)gbg, W), 3);
  probe("dWo", p.Wo, download((const float*)gWo, (size_t)W * O), 4);
  probe("dbo", p.bo, download((const float*)gbo, O), 3);

  release(dMsaIn); release(dMask); release(dDo); release(hqs); release(hqo); release(hWq);
  release(hWk); release(hWv); release(hWg); release(hbg); release(hWo); release(hbo);
  release(gMsa); release(gqs); release(gqo); release(gWq); release(gWk); release(gWv); release(gWg);
  release(gbg); release(gWo); release(gbo);
}

// ---------------------------------------------------------------- TriangleMultiplication, the whole module
struct TriP {
  std::vector<double> lnScale, lnOffset, Wl, bl, Wr, br, Wlg, blg, Wrg, brg,
      centerScale, centerOffset, Wo, bo, Wg, bg;
};
static double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }

// modules.py:947-1012, in double
static void triForward(const std::vector<double>& pair, const std::vector<double>& mask, int N, int C, int I,
                       bool outgoing, const TriP& p, std::vector<double>& out) {
  const size_t n2 = (size_t)N * N;
  std::vector<double> act0;
  lnForward(pair, n2, C, p.lnScale, p.lnOffset, 1e-5, act0);
  std::vector<double> a(n2 * I), b(n2 * I);
  for (size_t r = 0; r < n2; ++r)
    for (int c = 0; c < I; ++c) {
      double l = p.bl[c], rr = p.br[c], lg = p.blg[c], rg = p.brg[c];
      for (int k = 0; k < C; ++k) {
        double v = act0[r * C + k];
        l += v * p.Wl[(size_t)k * I + c];
        rr += v * p.Wr[(size_t)k * I + c];
        lg += v * p.Wlg[(size_t)k * I + c];
        rg += v * p.Wrg[(size_t)k * I + c];
      }
      a[r * I + c] = (l * mask[r]) * sigm(lg);      // the mask first, then the gate
      b[r * I + c] = (rr * mask[r]) * sigm(rg);
    }
  std::vector<double> prod(n2 * I, 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int c = 0; c < I; ++c) {
        double s = 0;
        for (int k = 0; k < N; ++k)
          s += outgoing ? a[((size_t)i * N + k) * I + c] * b[((size_t)j * N + k) * I + c]
                        : a[((size_t)k * N + j) * I + c] * b[((size_t)k * N + i) * I + c];
        prod[((size_t)i * N + j) * I + c] = s;
      }
  std::vector<double> q;
  lnForward(prod, n2, I, p.centerScale, p.centerOffset, 1e-5, q);
  out.assign(n2 * C, 0.0);
  for (size_t r = 0; r < n2; ++r)
    for (int c = 0; c < C; ++c) {
      double o = p.bo[c];
      for (int k = 0; k < I; ++k) o += q[r * I + k] * p.Wo[(size_t)k * C + c];
      double g = p.bg[c];
      for (int k = 0; k < C; ++k) g += act0[r * C + k] * p.Wg[(size_t)k * C + c];
      out[r * C + c] = o * sigm(g);
    }
}

static void triCase(int N, int C, int I, bool outgoing) {
  const size_t n2 = (size_t)N * N;
  std::vector<double> pair = randv(n2 * C), dOut = randv(n2 * C, 0.5), mask(n2, 1.0);
  std::uniform_real_distribution<double> U01(0, 1);
  for (size_t r = 0; r < n2; ++r) mask[r] = U01(rng) < 0.2 ? 0.0 : 1.0;

  TriP p;
  p.lnScale = randv(C, 0.4);
  for (auto& s : p.lnScale) s += 1.0;
  p.lnOffset = randv(C, 0.2);
  p.Wl = randv((size_t)C * I, 0.2);   p.bl = randv(I, 0.1);
  p.Wr = randv((size_t)C * I, 0.2);   p.br = randv(I, 0.1);
  p.Wlg = randv((size_t)C * I, 0.2);  p.blg = randv(I, 0.1);
  p.Wrg = randv((size_t)C * I, 0.2);  p.brg = randv(I, 0.1);
  p.centerScale = randv(I, 0.4);
  for (auto& s : p.centerScale) s += 1.0;
  p.centerOffset = randv(I, 0.2);
  p.Wo = randv((size_t)I * C, 0.2);   p.bo = randv(C, 0.1);
  p.Wg = randv((size_t)C * C, 0.2);   p.bg = randv(C, 0.1);

  // upload
  auto up = [&](const std::vector<double>& v) {
    auto f = toF(v);
    return uploadNew(f.data(), f.size());
  };
  void* dPairIn = up(pair);
  void* dMask = up(mask);
  void* dOutB = up(dOut);
  TriMulWeights w{};
  void* h[16];
  const std::vector<double>* srcs[16] = {&p.lnScale, &p.lnOffset, &p.Wl, &p.bl, &p.Wr, &p.br, &p.Wlg, &p.blg,
                                         &p.Wrg, &p.brg, &p.centerScale, &p.centerOffset, &p.Wo, &p.bo,
                                         &p.Wg, &p.bg};
  for (int i = 0; i < 16; ++i) h[i] = up(*srcs[i]);
  w.lnScale = (const float*)h[0]; w.lnOffset = (const float*)h[1];
  w.Wl = (const float*)h[2]; w.bl = (const float*)h[3];
  w.Wr = (const float*)h[4]; w.br = (const float*)h[5];
  w.Wlg = (const float*)h[6]; w.blg = (const float*)h[7];
  w.Wrg = (const float*)h[8]; w.brg = (const float*)h[9];
  w.centerScale = (const float*)h[10]; w.centerOffset = (const float*)h[11];
  w.Wo = (const float*)h[12]; w.bo = (const float*)h[13];
  w.Wg = (const float*)h[14]; w.bg = (const float*)h[15];

  void* gPair = alloc(n2 * C * 4);
  void* gp[16];
  for (int i = 0; i < 16; ++i) gp[i] = alloc(srcs[i]->size() * 4);
  TriMulGrads gg{};
  gg.lnScale = (float*)gp[0]; gg.lnOffset = (float*)gp[1];
  gg.Wl = (float*)gp[2]; gg.bl = (float*)gp[3];
  gg.Wr = (float*)gp[4]; gg.br = (float*)gp[5];
  gg.Wlg = (float*)gp[6]; gg.blg = (float*)gp[7];
  gg.Wrg = (float*)gp[8]; gg.brg = (float*)gp[9];
  gg.centerScale = (float*)gp[10]; gg.centerOffset = (float*)gp[11];
  gg.Wo = (float*)gp[12]; gg.bo = (float*)gp[13];
  gg.Wg = (float*)gp[14]; gg.bg = (float*)gp[15];

  triangleMultiplicationBwd((const float*)dPairIn, (const float*)dMask, N, C, I, outgoing,
                            w, (const float*)dOutB, (float*)gPair, gg);

  char shape[80];
  snprintf(shape, sizeof shape, "%s N%d C%d I%d", outgoing ? "out" : "in", N, C, I);

  auto S_ = [&](const std::vector<double>& pr, const TriP& q) {
    std::vector<double> o;
    triForward(pr, mask, N, C, I, outgoing, q, o);
    double acc = 0;
    for (size_t i = 0; i < o.size(); ++i) acc += dOut[i] * o[i];
    return acc;
  };
  auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
    std::vector<double> fd, mine;
    std::uniform_int_distribution<size_t> U(0, target.size() - 1);
    for (int t = 0; t < count; ++t) {
      size_t j = U(rng);
      double save = target[j], hh = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + hh;
      double aS = S_(pair, p);
      target[j] = save - hh;
      double bS = S_(pair, p);
      target[j] = save;
      fd.push_back((aS - bS) / (2 * hh));
      mine.push_back(got[j]);
    }
    char lab[200];
    snprintf(lab, sizeof lab, "trimul %s %s", what, shape);
    reportGrad(lab, fd, mine, 3e-5);
  };

  probe("dPair", pair, download((const float*)gPair, n2 * C), 5);
  const char* names[16] = {"dLnScale", "dLnOffset", "dWl", "dbl", "dWr", "dbr", "dWlg", "dblg",
                           "dWrg", "dbrg", "dCenterScale", "dCenterOffset", "dWo", "dbo", "dWg", "dbg"};
  std::vector<double>* tgt[16] = {&p.lnScale, &p.lnOffset, &p.Wl, &p.bl, &p.Wr, &p.br, &p.Wlg, &p.blg,
                                  &p.Wrg, &p.brg, &p.centerScale, &p.centerOffset, &p.Wo, &p.bo,
                                  &p.Wg, &p.bg};
  for (int i = 0; i < 16; ++i)
    probe(names[i], *tgt[i], download((const float*)gp[i], srcs[i]->size()), 3);

  release(dPairIn); release(dMask); release(dOutB); release(gPair);
  for (int i = 0; i < 16; ++i) { release(h[i]); release(gp[i]); }
}

// ---------------------------------------------------------------- the heads, and the metrics a loss reads
static void headCase(int N, int Cz, int bins) {
  const size_t pr = (size_t)N * N;
  std::vector<double> pair = randv(pr * Cz), W = randv((size_t)Cz * bins, 0.2), b0 = randv(bins, 0.1),
                      dLogits = randv(pr * bins, 0.5);
  auto half = [&](const std::vector<double>& p, const std::vector<double>& w, const std::vector<double>& bb) {
    std::vector<double> h(pr * bins);
    for (size_t r = 0; r < pr; ++r)
      for (int t = 0; t < bins; ++t) {
        double s2 = bb[t];
        for (int c = 0; c < Cz; ++c) s2 += p[r * Cz + c] * w[(size_t)c * bins + t];
        h[r * bins + t] = s2;
      }
    return h;
  };
  auto up = [&](const std::vector<double>& x) { auto f = toF(x); return uploadNew(f.data(), f.size()); };
  void* dP = up(pair); void* dW_ = up(W); void* dB = up(b0); void* dL = up(dLogits);
  void* gPair = alloc(pr * Cz * 4); void* gW = alloc((size_t)Cz * bins * 4); void* gB = alloc((size_t)bins * 4);

  for (int sym = 0; sym < 2; ++sym) {
    if (sym) distogramHeadBwd((const float*)dP, (const float*)dW_, (const float*)dL, N, Cz, bins,
                              (float*)gPair, (float*)gW, (float*)gB);
    else     paeHeadBwd((const float*)dP, (const float*)dW_, (const float*)dL, N, Cz, bins,
                        (float*)gPair, (float*)gW, (float*)gB);
    auto S_ = [&]() {
      auto h = half(pair, W, b0);
      double acc = 0;
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          for (int t = 0; t < bins; ++t) {
            const double v = sym ? h[((size_t)i * N + j) * bins + t] + h[((size_t)j * N + i) * bins + t]
                                 : h[((size_t)i * N + j) * bins + t];
            acc += dLogits[((size_t)i * N + j) * bins + t] * v;
          }
      return acc;
    };
    auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
      std::vector<double> fd, mine;
      std::uniform_int_distribution<size_t> Ud(0, target.size() - 1);
      for (int t = 0; t < count; ++t) {
        size_t j = Ud(rng);
        double save = target[j], hh = 1e-5 * std::max(1.0, fabs(save));
        target[j] = save + hh; double aS = S_();
        target[j] = save - hh; double bS = S_();
        target[j] = save;
        fd.push_back((aS - bS) / (2 * hh));
        mine.push_back(got[j]);
      }
      char lab[200];
      snprintf(lab, sizeof lab, "%s %s vs central difference N%d Cz%d bins%d",
               sym ? "distogram" : "paehead", what, N, Cz, bins);
      reportGrad(lab, fd, mine, 3e-5);
    };
    probe("dPair", pair, download((const float*)gPair, pr * Cz), 5);
    probe("dW", W, download((const float*)gW, (size_t)Cz * bins), 4);
    probe("dbias", b0, download((const float*)gB, bins), 3);
  }
  release(dP); release(dW_); release(dB); release(dL); release(gPair); release(gW); release(gB);
}

// bindcraft/af2.py:152-163 and confidence.py:113-180, in double
struct PaeOut { std::vector<double> pae, perAlignP, perAlignI; double ptm = 0, iptm = 0; };
static PaeOut paeForward(const std::vector<double>& logits, const std::vector<double>& breaks,
                         const std::vector<double>& wts, const std::vector<double>& asym, int N, int bins) {
  PaeOut o;
  const size_t pr = (size_t)N * N;
  const double step = breaks[1] - breaks[0];
  std::vector<double> ctr(bins);
  for (int b = 0; b + 1 < bins; ++b) ctr[b] = breaks[b] + step / 2;
  ctr[bins - 1] = ctr[bins - 2] + step;
  double sum = 0;
  for (int i = 0; i < N; ++i) sum += wts[i];
  const double d0 = 1.24 * cbrt(std::max(sum, 19.0) - 15.0) - 1.8;
  std::vector<double> tmb(bins);
  for (int b = 0; b < bins; ++b) tmb[b] = 1.0 / (1.0 + ctr[b] * ctr[b] / (d0 * d0));

  std::vector<double> raw(pr), tmP(pr), tmI(pr);
  for (size_t r = 0; r < pr; ++r) {
    double mx = -1e300;
    for (int b = 0; b < bins; ++b) mx = std::max(mx, logits[r * bins + b]);
    double tot = 0, a1 = 0, a2 = 0;
    for (int b = 0; b < bins; ++b) {
      const double e = exp(logits[r * bins + b] - mx);
      tot += e; a1 += e * ctr[b]; a2 += e * tmb[b];
    }
    raw[r] = a1 / tot;
    tmP[r] = a2 / tot;
  }
  o.pae.assign(pr, 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      o.pae[(size_t)i * N + j] = (raw[(size_t)i * N + j] + raw[(size_t)j * N + i]) / 2;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      const bool keep = asym.empty() ? false : (asym[i] != asym[j]);
      tmI[(size_t)i * N + j] = keep ? tmP[(size_t)i * N + j] : 0.0;
    }
  auto align = [&](const std::vector<double>& tm, bool iface) {
    std::vector<double> pa(N, 0.0);
    for (int i = 0; i < N; ++i) {
      double rs = 0;
      for (int j = 0; j < N; ++j) {
        const bool keep = iface ? (!asym.empty() && asym[i] != asym[j]) : true;
        rs += keep ? wts[i] * wts[j] : 0.0;
      }
      double p = 0;
      for (int j = 0; j < N; ++j) {
        const bool keep = iface ? (!asym.empty() && asym[i] != asym[j]) : true;
        const double nm = (keep ? wts[i] * wts[j] : 0.0) / (1e-8 + rs);
        p += tm[(size_t)i * N + j] * nm;
      }
      pa[i] = p;
    }
    return pa;
  };
  o.perAlignP = align(tmP, false);
  o.perAlignI = align(tmI, true);
  double bp = -1e300, bi = -1e300;
  for (int i = 0; i < N; ++i) {
    bp = std::max(bp, o.perAlignP[i] * wts[i]);
    bi = std::max(bi, o.perAlignI[i] * wts[i]);
  }
  o.ptm = bp;
  o.iptm = bi;
  return o;
}

static void paeMetricCase(int N, int bins) {
  const size_t pr = (size_t)N * N;
  std::vector<double> logits = randv(pr * bins, 1.0), breaks(bins - 1), wts(N, 1.0), asym(N, 0.0);
  for (int b = 0; b + 1 < bins; ++b) breaks[b] = 31.0 * b / (bins - 2);     // jnp.linspace(0, max_error_bin, ...)
  std::uniform_real_distribution<double> U01(0, 1);
  for (int i = 0; i < N; ++i) wts[i] = 0.5 + 0.5 * U01(rng);                // a soft seq_mask, as a loss may pass
  for (int i = 0; i < N; ++i) asym[i] = i < N / 2 ? 0.0 : 1.0;             // two chains, so ipTM is non-trivial
  std::vector<double> dPae = randv(pr, 0.5), dIptmPer = randv(N, 0.5);
  std::vector<double> dPtm(1, 0.7), dIptm(1, -0.4);

  auto up = [&](const std::vector<double>& x) { auto f = toF(x); return uploadNew(f.data(), f.size()); };
  void* dL = up(logits); void* dBr = up(breaks); void* dW_ = up(wts); void* dAs = up(asym);
  void* dPa = up(dPae); void* dPt = up(dPtm); void* dIp = up(dIptm); void* dIpr = up(dIptmPer);
  void* gL = alloc(pr * bins * 4);
  PaeMetricCotangents dc{};
  dc.pae = (const float*)dPa; dc.ptm = (const float*)dPt;
  dc.iptm = (const float*)dIp; dc.iptmPerResidue = (const float*)dIpr;
  paeMetricsBwd((const float*)dL, (const float*)dBr, (const float*)dW_, (const float*)dAs,
                N, bins, dc, (float*)gL);

  auto S_ = [&]() {
    PaeOut o = paeForward(logits, breaks, wts, asym, N, bins);
    double acc = dPtm[0] * o.ptm + dIptm[0] * o.iptm;
    for (size_t i = 0; i < pr; ++i) acc += dPae[i] * o.pae[i];
    for (int i = 0; i < N; ++i) acc += dIptmPer[i] * o.perAlignI[i];
    return acc;
  };
  std::vector<double> fd, mine;
  auto got = download((const float*)gL, pr * bins);
  std::uniform_int_distribution<size_t> Ud(0, logits.size() - 1);
  for (int t = 0; t < 10; ++t) {
    size_t j = Ud(rng);
    double save = logits[j], hh = 1e-5;
    logits[j] = save + hh; double aS = S_();
    logits[j] = save - hh; double bS = S_();
    logits[j] = save;
    fd.push_back((aS - bS) / (2 * hh));
    mine.push_back(got[j]);
  }
  char lab[200];
  snprintf(lab, sizeof lab, "pae metrics (pae+ptm+iptm+per-residue) vs central difference N%d bins%d", N, bins);
  reportGrad(lab, fd, mine, 5e-5);
  release(dL); release(dBr); release(dW_); release(dAs); release(dPa); release(dPt);
  release(dIp); release(dIpr); release(gL);
}

// ---------------------------------------------------------------- EvoformerIteration, the whole block
struct EvoP {
  OpmParams opm;
  std::vector<double> rowQS, rowQO, rowBS, rowBO, rowF2;  AttnMP rowAttn;
  std::vector<double> colQS, colQO;                       AttnMP colAttn;
  TrParams msaTr;
  TriP triOut, triIn;
  std::vector<double> saQS, saQO, saF2;                   AttnMP saAttn;
  std::vector<double> eaQS, eaQO, eaF2;                   AttnMP eaAttn;
  TrParams pairTr;
};
struct EvoDims { int S, N, Cm, Cz, mH, mD, tH, tD, tI, mI, pI, oO; bool opmFirst; };

// the N x N transpose, which is its own inverse
static std::vector<double> swapNN(const std::vector<double>& x, int N, int C) {
  std::vector<double> y(x.size());
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int c = 0; c < C; ++c) y[((size_t)i * N + j) * C + c] = x[((size_t)j * N + i) * C + c];
  return y;
}

// modules.py:1300-1388, with dropout off so every wrapper is a plain residual. `sv` collects each module's input,
// which is what the Metal backward is handed.
static void evoForward(const std::vector<double>& msa0, const std::vector<double>& msaMask,
                       const std::vector<double>& pair0, const std::vector<double>& pairMask,
                       const EvoDims& d, const EvoP& p,
                       std::vector<double>& msaOut, std::vector<double>& pairOut,
                       std::vector<std::vector<double>>* sv = nullptr) {
  const size_t mr = (size_t)d.S * d.N, pr = (size_t)d.N * d.N;
  std::vector<double> msa = msa0, pair = pair0, out, ln, a1;
  auto keep = [&](const std::vector<double>& v) { if (sv) sv->push_back(v); };
  auto addTo = [](std::vector<double>& dst, const std::vector<double>& src) {
    for (size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
  };
  auto doOpm = [&](const std::vector<double>& m) {
    opmForward(m, msaMask, d.S, d.N, d.Cm, d.oO, d.Cz, p.opm, 1e-3, out);
    addTo(pair, out);
  };

  keep(msa);                                                   // 0: msaIn
  if (d.opmFirst) doOpm(msa);
  keep(pair);                                                  // 1: pairForRow
  biasedForward(msa, pair, msaMask, d.S, d.N, d.Cm, d.Cz, d.mH, d.mD, d.Cm, false,
                p.rowQS, p.rowQO, p.rowBS, p.rowBO, p.rowF2, p.rowAttn, true, out);
  addTo(msa, out);
  keep(msa);                                                   // 2: msaForCol
  {   // the column attention, which AF2 runs on the transposed MSA
    std::vector<double> actT(mr * d.Cm), maskT(mr), outT, none;
    for (int s = 0; s < d.S; ++s)
      for (int n = 0; n < d.N; ++n) {
        maskT[(size_t)n * d.S + s] = msaMask[(size_t)s * d.N + n];
        for (int c = 0; c < d.Cm; ++c)
          actT[((size_t)n * d.S + s) * d.Cm + c] = msa[((size_t)s * d.N + n) * d.Cm + c];
      }
    biasedForward(actT, actT, maskT, d.N, d.S, d.Cm, d.Cm, d.mH, d.mD, d.Cm, true,
                  p.colQS, p.colQO, none, none, none, p.colAttn, true, outT);
    for (int s = 0; s < d.S; ++s)
      for (int n = 0; n < d.N; ++n)
        for (int c = 0; c < d.Cm; ++c)
          msa[((size_t)s * d.N + n) * d.Cm + c] += outT[((size_t)n * d.S + s) * d.Cm + c];
  }
  keep(msa);                                                   // 3: msaForTransition
  trForward(msa, mr, d.Cm, d.mI, p.msaTr, ln, a1, out);
  addTo(msa, out);
  keep(msa);                                                   // 4: msaForOpm
  if (!d.opmFirst) doOpm(msa);

  keep(pair);                                                  // 5: pairForTriOut
  triForward(pair, pairMask, d.N, d.Cz, d.tI, true, p.triOut, out);
  addTo(pair, out);
  keep(pair);                                                  // 6: pairForTriIn
  triForward(pair, pairMask, d.N, d.Cz, d.tI, false, p.triIn, out);
  addTo(pair, out);
  keep(pair);                                                  // 7: pairForAttnStart
  biasedForward(pair, pair, pairMask, d.N, d.N, d.Cz, d.Cz, d.tH, d.tD, d.Cz, true,
                p.saQS, p.saQO, p.saQS, p.saQO, p.saF2, p.saAttn, true, out);
  addTo(pair, out);
  keep(pair);                                                  // 8: pairForAttnEnd
  {   // the ending node, on the transposed pair
    std::vector<double> pT = swapNN(pair, d.N, d.Cz), mT = swapNN(pairMask, d.N, 1), oT;
    biasedForward(pT, pT, mT, d.N, d.N, d.Cz, d.Cz, d.tH, d.tD, d.Cz, true,
                  p.eaQS, p.eaQO, p.eaQS, p.eaQO, p.eaF2, p.eaAttn, true, oT);
    addTo(pair, swapNN(oT, d.N, d.Cz));
  }
  keep(pair);                                                  // 9: pairForTransition
  trForward(pair, pr, d.Cz, d.pI, p.pairTr, ln, a1, out);
  addTo(pair, out);
  msaOut = msa;
  pairOut = pair;
}

static void evoCase(EvoDims d) {
  const int mW = d.mH * d.mD, tW = d.tH * d.tD;
  const size_t mr = (size_t)d.S * d.N, pr = (size_t)d.N * d.N;
  std::vector<double> msa = randv(mr * d.Cm), pair = randv(pr * d.Cz),
                      dMsaOut = randv(mr * d.Cm, 0.5), dPairOut = randv(pr * d.Cz, 0.5),
                      msaMask(mr, 1.0), pairMask(pr, 1.0);
  for (size_t i = 0; i < mr; ++i) if (i % 7 == 3) msaMask[i] = 0.0;
  for (int i = 0; i < d.N; ++i) msaMask[i] = 1.0;              // sequence 0 all present
  for (size_t i = 0; i < pr; ++i) if (i % 11 == 5) pairMask[i] = 0.0;
  for (int i = 0; i < d.N; ++i) { pairMask[(size_t)i * d.N + i] = 1.0; pairMask[i] = 1.0; pairMask[(size_t)i * d.N] = 1.0; }

  auto norm = [&](int C) { auto v = randv(C, 0.3); for (auto& x : v) x += 1.0; return v; };
  auto attn = [&](int C, int W, int O) {
    AttnMP a;
    a.Wq = randv((size_t)C * W, 0.2); a.Wk = randv((size_t)C * W, 0.2); a.Wv = randv((size_t)C * W, 0.2);
    a.Wg = randv((size_t)C * W, 0.2); a.bg = randv(W, 0.1);
    a.Wo = randv((size_t)W * O, 0.2); a.bo = randv(O, 0.1);
    return a;
  };
  auto tri = [&](int C, int I) {
    TriP t;
    t.lnScale = norm(C); t.lnOffset = randv(C, 0.2);
    t.Wl = randv((size_t)C * I, 0.2);  t.bl = randv(I, 0.1);
    t.Wr = randv((size_t)C * I, 0.2);  t.br = randv(I, 0.1);
    t.Wlg = randv((size_t)C * I, 0.2); t.blg = randv(I, 0.1);
    t.Wrg = randv((size_t)C * I, 0.2); t.brg = randv(I, 0.1);
    t.centerScale = norm(I); t.centerOffset = randv(I, 0.2);
    t.Wo = randv((size_t)I * C, 0.2);  t.bo = randv(C, 0.1);
    t.Wg = randv((size_t)C * C, 0.2);  t.bg = randv(C, 0.1);
    return t;
  };
  auto trp = [&](int C, int H) {
    TrParams t;
    t.lnScale = norm(C); t.lnOffset = randv(C, 0.2);
    t.W1 = randv((size_t)C * H, 0.15); t.b1 = randv(H, 0.1);
    t.W2 = randv((size_t)H * C, 0.15); t.b2 = randv(C, 0.1);
    return t;
  };
  EvoP p;
  p.opm.lnScale = norm(d.Cm); p.opm.lnOffset = randv(d.Cm, 0.2);
  p.opm.Wl = randv((size_t)d.Cm * d.oO, 0.2); p.opm.bl = randv(d.oO, 0.1);
  p.opm.Wr = randv((size_t)d.Cm * d.oO, 0.2); p.opm.br = randv(d.oO, 0.1);
  p.opm.Wo = randv((size_t)d.oO * d.oO * d.Cz, 0.1); p.opm.bo = randv(d.Cz, 0.1);
  p.rowQS = norm(d.Cm); p.rowQO = randv(d.Cm, 0.2);
  p.rowBS = norm(d.Cz); p.rowBO = randv(d.Cz, 0.2);
  p.rowF2 = randv((size_t)d.Cz * d.mH, 0.3); p.rowAttn = attn(d.Cm, mW, d.Cm);
  p.colQS = norm(d.Cm); p.colQO = randv(d.Cm, 0.2); p.colAttn = attn(d.Cm, mW, d.Cm);
  p.msaTr = trp(d.Cm, d.mI);
  p.triOut = tri(d.Cz, d.tI); p.triIn = tri(d.Cz, d.tI);
  p.saQS = norm(d.Cz); p.saQO = randv(d.Cz, 0.2);
  p.saF2 = randv((size_t)d.Cz * d.tH, 0.3); p.saAttn = attn(d.Cz, tW, d.Cz);
  p.eaQS = norm(d.Cz); p.eaQO = randv(d.Cz, 0.2);
  p.eaF2 = randv((size_t)d.Cz * d.tH, 0.3); p.eaAttn = attn(d.Cz, tW, d.Cz);
  p.pairTr = trp(d.Cz, d.pI);

  // the forward, in double, which also yields the save set the backward is handed
  std::vector<std::vector<double>> sv;
  std::vector<double> msaOut, pairOut;
  evoForward(msa, msaMask, pair, pairMask, d, p, msaOut, pairOut, &sv);

  auto up = [&](const std::vector<double>& x) { auto f = toF(x); return uploadNew(f.data(), f.size()); };
  std::vector<void*> owned;
  auto U = [&](const std::vector<double>& x) { void* q = up(x); owned.push_back(q); return (const float*)q; };

  EvoBlockSaved S_{};
  S_.msaIn = U(sv[0]); S_.pairForRow = U(sv[1]); S_.msaForCol = U(sv[2]);
  S_.msaForTransition = U(sv[3]); S_.msaForOpm = U(sv[4]);
  S_.pairForTriOut = U(sv[5]); S_.pairForTriIn = U(sv[6]);
  S_.pairForAttnStart = U(sv[7]); S_.pairForAttnEnd = U(sv[8]); S_.pairForTransition = U(sv[9]);

  EvoBlockWeights w{};
  w.opm = {U(p.opm.lnScale), U(p.opm.lnOffset), U(p.opm.Wl), U(p.opm.bl), U(p.opm.Wr), U(p.opm.br),
           U(p.opm.Wo)};
  auto battn = [&](const std::vector<double>& qs, const std::vector<double>& qo,
                   const std::vector<double>* bs, const std::vector<double>* bo,
                   const std::vector<double>* f2, const AttnMP& a) {
    BiasedAttnWeights b{};
    b.qScale = U(qs); b.qOffset = U(qo);
    if (bs) { b.bScale = U(*bs); b.bOffset = U(*bo); }
    b.feat2dW = f2 ? U(*f2) : nullptr;
    b.attn.Wq = U(a.Wq); b.attn.Wk = U(a.Wk); b.attn.Wv = U(a.Wv);
    b.attn.Wg = U(a.Wg); b.attn.bg = U(a.bg); b.attn.Wo = U(a.Wo); b.attn.bo = U(a.bo);
    return b;
  };
  w.msaRow = battn(p.rowQS, p.rowQO, &p.rowBS, &p.rowBO, &p.rowF2, p.rowAttn);
  w.msaCol = battn(p.colQS, p.colQO, nullptr, nullptr, nullptr, p.colAttn);
  w.msaTransition = {U(p.msaTr.lnScale), U(p.msaTr.lnOffset), U(p.msaTr.W1), U(p.msaTr.b1),
                     U(p.msaTr.W2), U(p.msaTr.b2)};
  auto trimul = [&](const TriP& t) {
    TriMulWeights m{};
    m.lnScale = U(t.lnScale); m.lnOffset = U(t.lnOffset);
    m.Wl = U(t.Wl); m.bl = U(t.bl); m.Wr = U(t.Wr); m.br = U(t.br);
    m.Wlg = U(t.Wlg); m.blg = U(t.blg); m.Wrg = U(t.Wrg); m.brg = U(t.brg);
    m.centerScale = U(t.centerScale); m.centerOffset = U(t.centerOffset);
    m.Wo = U(t.Wo); m.bo = U(t.bo); m.Wg = U(t.Wg); m.bg = U(t.bg);
    return m;
  };
  w.triOutgoing = trimul(p.triOut);
  w.triIncoming = trimul(p.triIn);
  w.triAttnStart = battn(p.saQS, p.saQO, nullptr, nullptr, &p.saF2, p.saAttn);
  w.triAttnEnd = battn(p.eaQS, p.eaQO, nullptr, nullptr, &p.eaF2, p.eaAttn);
  w.pairTransition = {U(p.pairTr.lnScale), U(p.pairTr.lnOffset), U(p.pairTr.W1), U(p.pairTr.b1),
                      U(p.pairTr.W2), U(p.pairTr.b2)};

  const float* mMask = U(msaMask);
  const float* pMask = U(pairMask);
  const float* dMo = U(dMsaOut);
  const float* dPo = U(dPairOut);
  void* gMsa = alloc(mr * d.Cm * 4);
  void* gPair = alloc(pr * d.Cz * 4);
  // a sample of parameter gradients from four different modules, to catch a weight wired to the wrong module
  void* gRowWv = alloc((size_t)d.Cm * mW * 4);
  void* gTriOutWl = alloc((size_t)d.Cz * d.tI * 4);
  void* gPairTrW1 = alloc((size_t)d.Cz * d.pI * 4);
  void* gOpmWl = alloc((size_t)d.Cm * d.oO * 4);
  EvoBlockGrads g{};
  g.msaRow.attn.Wv = (float*)gRowWv;
  g.triOutgoing.Wl = (float*)gTriOutWl;
  g.pairTransition.W1 = (float*)gPairTrW1;
  g.opm.Wl = (float*)gOpmWl;

  EvoBlockShape sh{d.S, d.N, d.Cm, d.Cz, d.mH, d.mD, d.tH, d.tD, d.tI, d.mI, d.pI, d.oO, d.opmFirst};
  evoformerIterationBwd(mMask, pMask, S_, sh, w, dMo, dPo, (float*)gMsa, (float*)gPair, g);

  char shape[120];
  snprintf(shape, sizeof shape, "S%d N%d Cm%d Cz%d%s", d.S, d.N, d.Cm, d.Cz, d.opmFirst ? " opm-first" : "");
  auto total = [&]() {
    std::vector<double> mo, po;
    evoForward(msa, msaMask, pair, pairMask, d, p, mo, po);
    double acc = 0;
    for (size_t i = 0; i < mo.size(); ++i) acc += dMsaOut[i] * mo[i];
    for (size_t i = 0; i < po.size(); ++i) acc += dPairOut[i] * po[i];
    return acc;
  };
  auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
    std::vector<double> fd, mine;
    std::uniform_int_distribution<size_t> Ud(0, target.size() - 1);
    for (int t = 0; t < count; ++t) {
      size_t j = Ud(rng);
      double save = target[j], hh = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + hh;
      double aS = total();
      target[j] = save - hh;
      double bS = total();
      target[j] = save;
      fd.push_back((aS - bS) / (2 * hh));
      mine.push_back(got[j]);
    }
    char lab[240];
    snprintf(lab, sizeof lab, "evoblock %s vs central difference %s", what, shape);
    reportGrad(lab, fd, mine, 5e-5);
  };
  probe("dMsa", msa, download((const float*)gMsa, mr * d.Cm), 6);
  probe("dPair", pair, download((const float*)gPair, pr * d.Cz), 6);
  probe("dMsaRow.Wv", p.rowAttn.Wv, download((const float*)gRowWv, (size_t)d.Cm * mW), 3);
  probe("dTriOutgoing.Wl", p.triOut.Wl, download((const float*)gTriOutWl, (size_t)d.Cz * d.tI), 3);
  probe("dPairTransition.W1", p.pairTr.W1, download((const float*)gPairTrW1, (size_t)d.Cz * d.pI), 3);
  probe("dOpm.Wl", p.opm.Wl, download((const float*)gOpmWl, (size_t)d.Cm * d.oO), 3);

  for (void* q : owned) release(q);
  release(gMsa); release(gPair); release(gRowWv); release(gTriOutWl); release(gPairTrW1); release(gOpmWl);
}

// ---------------------------------------------------------------- the structure module's rigids
// A rigid is 12 doubles, the rotation row-major then the translation - the forward's layout. The operands below are
// RANDOM MATRICES rather than rotations on purpose: these kernels treat the rotation as a general matrix, and an
// orthogonal or symmetric operand hides a transposed index.
static void rigCompose(const double* A, const double* B, double* C) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  for (int k = 0; k < 3; ++k)
    C[9 + k] = A[k * 3] * B[9] + A[k * 3 + 1] * B[10] + A[k * 3 + 2] * B[11] + A[9 + k];
}
static void quatRigid(const double* u, double* B) {
  double w = 1, x = u[0], y = u[1], z = u[2];
  double inv = 1.0 / sqrt(w * w + x * x + y * y + z * z);
  w *= inv; x *= inv; y *= inv; z *= inv;
  B[0] = 1 - 2 * (y * y + z * z); B[1] = 2 * (x * y - w * z);     B[2] = 2 * (x * z + w * y);
  B[3] = 2 * (x * y + w * z);     B[4] = 1 - 2 * (x * x + z * z); B[5] = 2 * (y * z - w * x);
  B[6] = 2 * (x * z - w * y);     B[7] = 2 * (y * z + w * x);     B[8] = 1 - 2 * (x * x + y * y);
  B[9] = u[3]; B[10] = u[4]; B[11] = u[5];
}
static double rigUpdS(const std::vector<double>& A, const std::vector<double>& u, const std::vector<double>& dC,
                      int L) {
  double S = 0;
  for (int i = 0; i < L; ++i) {
    double B[12], C[12];
    quatRigid(&u[(size_t)i * 6], B);
    rigCompose(&A[(size_t)i * 12], B, C);
    for (int k = 0; k < 12; ++k) S += dC[(size_t)i * 12 + k] * C[k];
  }
  return S;
}

static void rigidUpdCase(int L) {
  std::vector<double> A = randv((size_t)L * 12), u = randv((size_t)L * 6, 0.5),
                      dC = randv((size_t)L * 12, 0.5);

  // reference one: the analytic backward, in double
  std::vector<double> rdA((size_t)L * 12, 0), rdu((size_t)L * 6, 0);
  for (int i = 0; i < L; ++i) {
    const double* a = &A[(size_t)i * 12];
    const double* g = &dC[(size_t)i * 12];
    double B[12];
    quatRigid(&u[(size_t)i * 6], B);
    for (int k = 0; k < 3; ++k) {
      for (int j = 0; j < 3; ++j) {
        double s = g[9 + k] * B[9 + j];
        for (int m = 0; m < 3; ++m) s += g[k * 3 + m] * B[j * 3 + m];
        rdA[(size_t)i * 12 + k * 3 + j] = s;
      }
      rdA[(size_t)i * 12 + 9 + k] = g[9 + k];
    }
    double G[9];
    for (int j = 0; j < 3; ++j)
      for (int m = 0; m < 3; ++m) {
        double s = 0;
        for (int k = 0; k < 3; ++k) s += g[k * 3 + m] * a[k * 3 + j];
        G[j * 3 + m] = s;
      }
    double w = 1, x = u[(size_t)i * 6], y = u[(size_t)i * 6 + 1], z = u[(size_t)i * 6 + 2];
    double inv = 1.0 / sqrt(w * w + x * x + y * y + z * z);
    w *= inv; x *= inv; y *= inv; z *= inv;
    double dw = 2 * (-G[1] * z + G[2] * y + G[3] * z - G[5] * x - G[6] * y + G[7] * x);
    double dx = 2 * (G[1] * y + G[2] * z + G[3] * y - 2 * x * G[4] - G[5] * w + G[6] * z + G[7] * w - 2 * x * G[8]);
    double dy = 2 * (-2 * y * G[0] + G[1] * x + G[2] * w + G[3] * x + G[5] * z - G[6] * w + G[7] * z - 2 * y * G[8]);
    double dz = 2 * (-2 * z * G[0] - G[1] * w + G[2] * x + G[3] * w - 2 * z * G[4] + G[5] * y + G[6] * x + G[7] * y);
    double dot = dw * w + dx * x + dy * y + dz * z;
    rdu[(size_t)i * 6] = (dx - x * dot) * inv;
    rdu[(size_t)i * 6 + 1] = (dy - y * dot) * inv;
    rdu[(size_t)i * 6 + 2] = (dz - z * dot) * inv;
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += g[9 + k] * a[k * 3 + j];
      rdu[(size_t)i * 6 + 3 + j] = s;
    }
  }

  auto fA = toF(A), fu = toF(u), fdC = toF(dC);
  void* dAb = uploadNew(fA.data(), fA.size());
  void* dub = uploadNew(fu.data(), fu.size());
  void* dCb = uploadNew(fdC.data(), fdC.size());
  void* ddA = alloc((size_t)L * 12 * 4);
  void* ddu = alloc((size_t)L * 6 * 4);
  rigidUpdateBwd((const float*)dAb, (const float*)dub, (const float*)dCb, (float*)ddA, (float*)ddu, L);

  char label[160];
  auto gotA = download((const float*)ddA, (size_t)L * 12), gotu = download((const float*)ddu, (size_t)L * 6);
  snprintf(label, sizeof label, "rigid update bwd dRigOld L=%d", L);
  report(label, relRms(rdA, gotA), 1e-5);
  snprintf(label, sizeof label, "rigid update bwd dUpd L=%d", L);
  report(label, relRms(rdu, gotu), 1e-5);

  // reference two: the central difference against the FORWARD alone, which is what catches a wrong derivation
  const double h = 1e-6;
  std::vector<double> fdA((size_t)L * 12), fdu((size_t)L * 6), mineA(gotA.begin(), gotA.end()),
      mineU(gotu.begin(), gotu.end());
  for (size_t t = 0; t < A.size(); ++t) {
    double keep = A[t];
    A[t] = keep + h; double p = rigUpdS(A, u, dC, L);
    A[t] = keep - h; double m = rigUpdS(A, u, dC, L);
    A[t] = keep;
    fdA[t] = (p - m) / (2 * h);
  }
  for (size_t t = 0; t < u.size(); ++t) {
    double keep = u[t];
    u[t] = keep + h; double p = rigUpdS(A, u, dC, L);
    u[t] = keep - h; double m = rigUpdS(A, u, dC, L);
    u[t] = keep;
    fdu[t] = (p - m) / (2 * h);
  }
  snprintf(label, sizeof label, "rigid update fd dRigOld L=%d", L);
  reportGrad(label, fdA, mineA, 1e-5);
  snprintf(label, sizeof label, "rigid update fd dUpd L=%d", L);
  reportGrad(label, fdu, mineU, 1e-5);

  // and that `accumulateRigOld` adds rather than writes - an iteration needs it, because the same rigid also feeds
  // that iteration's point projections and its side chains
  rigidUpdateBwd((const float*)dAb, (const float*)dub, (const float*)dCb, (float*)ddA, (float*)ddu, L, true);
  std::vector<double> twice(rdA.size());
  for (size_t t = 0; t < twice.size(); ++t) twice[t] = 2 * rdA[t];
  snprintf(label, sizeof label, "rigid update bwd dRigOld accumulated L=%d", L);
  report(label, relRms(twice, download((const float*)ddA, (size_t)L * 12)), 1e-5);

  release(dAb); release(dub); release(dCb); release(ddA); release(ddu);
}

// ---------------------------------------------------------------- the point projections
static double pointsS(const std::vector<double>& proj, const std::vector<double>& rig,
                      const std::vector<double>& dOut, int L, int H, int P) {
  double S = 0;
  for (int i = 0; i < L; ++i)
    for (int h = 0; h < H; ++h) {
      const double* s = &proj[((size_t)i * H + h) * 3 * P];
      const double* r = &rig[(size_t)i * 12];
      for (int p = 0; p < P; ++p) {
        double loc[3] = {s[p], s[P + p], s[2 * P + p]};
        for (int k = 0; k < 3; ++k) {
          double g = r[k * 3] * loc[0] + r[k * 3 + 1] * loc[1] + r[k * 3 + 2] * loc[2] + r[9 + k];
          S += dOut[(((size_t)i * H + h) * P + p) * 3 + k] * g;
        }
      }
    }
  return S;
}

static void pointsGlobalCase(int L, int H, int P) {
  std::vector<double> proj = randv((size_t)L * H * 3 * P), rig = randv((size_t)L * 12),
                      dOut = randv((size_t)L * H * P * 3, 0.5);

  std::vector<double> rdProj((size_t)L * H * 3 * P, 0), rdRig((size_t)L * 12, 0);
  for (int i = 0; i < L; ++i) {
    const double* r = &rig[(size_t)i * 12];
    for (int h = 0; h < H; ++h) {
      const double* s = &proj[((size_t)i * H + h) * 3 * P];
      double* ds = &rdProj[((size_t)i * H + h) * 3 * P];
      for (int p = 0; p < P; ++p) {
        const double* dg = &dOut[(((size_t)i * H + h) * P + p) * 3];
        double loc[3] = {s[p], s[P + p], s[2 * P + p]};
        for (int j = 0; j < 3; ++j) ds[j * P + p] = r[j] * dg[0] + r[3 + j] * dg[1] + r[6 + j] * dg[2];
        for (int k = 0; k < 3; ++k) {
          for (int j = 0; j < 3; ++j) rdRig[(size_t)i * 12 + k * 3 + j] += dg[k] * loc[j];
          rdRig[(size_t)i * 12 + 9 + k] += dg[k];
        }
      }
    }
  }

  auto fp = toF(proj), fr = toF(rig), fd = toF(dOut);
  void* dp = uploadNew(fp.data(), fp.size());
  void* dr = uploadNew(fr.data(), fr.size());
  void* dd = uploadNew(fd.data(), fd.size());
  void* ddp = alloc((size_t)L * H * 3 * P * 4);
  void* ddr = alloc((size_t)L * 12 * 4);
  pointsGlobalBwd((const float*)dp, (const float*)dr, (const float*)dd, (float*)ddp, (float*)ddr, L, H, P);

  char label[160];
  auto gotP = download((const float*)ddp, (size_t)L * H * 3 * P), gotR = download((const float*)ddr, (size_t)L * 12);
  snprintf(label, sizeof label, "points global bwd dProj L=%d H=%d P=%d", L, H, P);
  report(label, relRms(rdProj, gotP), 1e-5);
  snprintf(label, sizeof label, "points global bwd dRig L=%d H=%d P=%d", L, H, P);
  report(label, relRms(rdRig, gotR), 1e-5);

  const double h = 1e-6;
  std::vector<double> fdP(proj.size()), fdR(rig.size()), mineP(gotP.begin(), gotP.end()),
      mineR(gotR.begin(), gotR.end());
  for (size_t t = 0; t < proj.size(); ++t) {
    double keep = proj[t];
    proj[t] = keep + h; double a = pointsS(proj, rig, dOut, L, H, P);
    proj[t] = keep - h; double b = pointsS(proj, rig, dOut, L, H, P);
    proj[t] = keep;
    fdP[t] = (a - b) / (2 * h);
  }
  for (size_t t = 0; t < rig.size(); ++t) {
    double keep = rig[t];
    rig[t] = keep + h; double a = pointsS(proj, rig, dOut, L, H, P);
    rig[t] = keep - h; double b = pointsS(proj, rig, dOut, L, H, P);
    rig[t] = keep;
    fdR[t] = (a - b) / (2 * h);
  }
  snprintf(label, sizeof label, "points global fd dProj L=%d H=%d P=%d", L, H, P);
  reportGrad(label, fdP, mineP, 1e-5);
  snprintf(label, sizeof label, "points global fd dRig L=%d H=%d P=%d", L, H, P);
  reportGrad(label, fdR, mineR, 1e-5);

  release(dp); release(dr); release(dd); release(ddp); release(ddr);
}

// ---------------------------------------------------------------- InvariantPointAttention: the attention weights
// The forward in double, exactly as af2_ipa_weights computes it. `attn` is produced HERE and handed to the kernel,
// so both references and the kernel are differentiating one consistent softmax.
static void ipaWForward(const std::vector<double>& qs, const std::vector<double>& ks,
                        const std::vector<double>& qp, const std::vector<double>& kp,
                        const std::vector<double>& b2d, const std::vector<double>& pwRaw,
                        const std::vector<double>& mask, int L, int H, int Cs, int Pq, double base,
                        std::vector<double>& attn) {
  attn.assign((size_t)L * H * L, 0.0);
  const double sc = sqrt(1.0 / 3.0);
  for (int q = 0; q < L; ++q)
    for (int h = 0; h < H; ++h) {
      const double w = base * log(1.0 + exp(pwRaw[h]));
      std::vector<double> s(L);
      for (int k = 0; k < L; ++k) {
        double d2 = 0;
        for (int p = 0; p < Pq; ++p)
          for (int x = 0; x < 3; ++x) {
            double d = qp[(((size_t)q * H + h) * Pq + p) * 3 + x] - kp[(((size_t)k * H + h) * Pq + p) * 3 + x];
            d2 += d * d;
          }
        double dot = 0;
        for (int c = 0; c < Cs; ++c)
          dot += qs[((size_t)q * H + h) * Cs + c] * ks[((size_t)k * H + h) * Cs + c];
        s[k] = (dot - 0.5 * w * d2 + b2d[((size_t)q * L + k) * H + h] - 1e5 * (1 - mask[q] * mask[k])) * sc;
      }
      double mx = s[0];
      for (int k = 1; k < L; ++k) mx = std::max(mx, s[k]);
      double sum = 0;
      for (int k = 0; k < L; ++k) { s[k] = exp(s[k] - mx); sum += s[k]; }
      for (int k = 0; k < L; ++k) attn[((size_t)q * H + h) * L + k] = s[k] / sum;
    }
}

static void ipaWeightsCase(int L, int H, int Cs, int Pq) {
  const size_t lhc = (size_t)L * H * Cs, lhp = (size_t)L * H * Pq * 3, pb = (size_t)L * L * H;
  const double base = sqrt(1.0 / (Pq * 9.0 / 2.0));
  std::vector<double> qs = randv(lhc, 0.5), ks = randv(lhc, 0.5), qp = randv(lhp, 0.5), kp = randv(lhp, 0.5),
                      b2d = randv(pb, 0.5), pwRaw = randv(H, 0.5), dAttn = randv((size_t)L * H * L, 0.5),
                      mask(L, 1.0);
  for (int i = 0; i < L; ++i) if (i % 5 == 3) mask[i] = 0.0;   // AF2's -1e5 penalty on a masked pair

  std::vector<double> attn;
  ipaWForward(qs, ks, qp, kp, b2d, pwRaw, mask, L, H, Cs, Pq, base, attn);

  // reference one: the analytic backward, in double
  std::vector<double> rqs(lhc, 0), rks(lhc, 0), rqp(lhp, 0), rkp(lhp, 0), rb2d(pb, 0), rpw(H, 0),
      dLogit((size_t)L * H * L, 0);
  const double sc = sqrt(1.0 / 3.0);
  for (int q = 0; q < L; ++q)
    for (int h = 0; h < H; ++h) {
      double corr = 0;
      for (int k = 0; k < L; ++k) corr += attn[((size_t)q * H + h) * L + k] * dAttn[((size_t)q * H + h) * L + k];
      for (int k = 0; k < L; ++k) {
        size_t t = ((size_t)q * H + h) * L + k;
        dLogit[t] = attn[t] * (dAttn[t] - corr) * sc;
      }
    }
  for (int q = 0; q < L; ++q)
    for (int h = 0; h < H; ++h) {
      const double w = base * log(1.0 + exp(pwRaw[h]));
      for (int k = 0; k < L; ++k) {
        double g = dLogit[((size_t)q * H + h) * L + k];
        rb2d[((size_t)q * L + k) * H + h] = g;
        for (int c = 0; c < Cs; ++c) {
          rqs[((size_t)q * H + h) * Cs + c] += g * ks[((size_t)k * H + h) * Cs + c];
          rks[((size_t)k * H + h) * Cs + c] += g * qs[((size_t)q * H + h) * Cs + c];
        }
        for (int p = 0; p < Pq; ++p)
          for (int x = 0; x < 3; ++x) {
            size_t qi = (((size_t)q * H + h) * Pq + p) * 3 + x, ki = (((size_t)k * H + h) * Pq + p) * 3 + x;
            double d = qp[qi] - kp[ki];
            rqp[qi] += -w * g * d;
            rkp[ki] += w * g * d;
            rpw[h] += g * (-0.5) * d * d;
          }
      }
    }
  for (int h = 0; h < H; ++h) rpw[h] *= base / (1.0 + exp(-pwRaw[h]));

  auto up = [&](const std::vector<double>& x) { auto f = toF(x); return uploadNew(f.data(), f.size()); };
  void* bAttn = up(attn); void* bQs = up(qs); void* bKs = up(ks); void* bQp = up(qp); void* bKp = up(kp);
  std::vector<double> pwv(H);
  for (int h = 0; h < H; ++h) pwv[h] = base * log(1.0 + exp(pwRaw[h]));
  void* bPw = up(pwv); void* bRaw = up(pwRaw); void* bDa = up(dAttn);
  void* gqs = alloc(lhc * 4); void* gks = alloc(lhc * 4); void* gqp = alloc(lhp * 4); void* gkp = alloc(lhp * 4);
  void* gb2d = alloc(pb * 4); void* gpw = alloc((size_t)H * 4);

  IpaWeightsSaved saved{(const float*)bAttn, (const float*)bQs, (const float*)bKs, (const float*)bQp,
                        (const float*)bKp, (const float*)bPw, (const float*)bRaw};
  IpaWeightsGrads grads{(float*)gqs, (float*)gks, (float*)gqp, (float*)gkp, (float*)gb2d, (float*)gpw};
  ipaWeightsBwd(saved, (const float*)bDa, L, H, Cs, Pq, (float)base, grads);

  char label[200], shape[80];
  snprintf(shape, sizeof shape, "L%d H%d Cs%d Pq%d", L, H, Cs, Pq);
  auto chk = [&](const char* what, const std::vector<double>& ref, void* got, size_t n, double bound) {
    snprintf(label, sizeof label, "IPA weights bwd %s %s", what, shape);
    report(label, relRms(ref, download((const float*)got, n)), bound);
  };
  chk("dqs", rqs, gqs, lhc, 1e-5);
  chk("dks", rks, gks, lhc, 1e-5);
  chk("dqp", rqp, gqp, lhp, 1e-5);
  chk("dkp", rkp, gkp, lhp, 1e-5);
  chk("db2d", rb2d, gb2d, pb, 1e-5);
  chk("dPointWeightRaw", rpw, gpw, H, 1e-5);

  // reference two: sampled central differences against the FORWARD alone
  auto S_ = [&]() {
    std::vector<double> a;
    ipaWForward(qs, ks, qp, kp, b2d, pwRaw, mask, L, H, Cs, Pq, base, a);
    double acc = 0;
    for (size_t i = 0; i < a.size(); ++i) acc += dAttn[i] * a[i];
    return acc;
  };
  auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
    std::vector<double> fd, mine;
    std::uniform_int_distribution<size_t> U(0, target.size() - 1);
    for (int t = 0; t < count; ++t) {
      size_t j = U(rng);
      double save = target[j], hh = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + hh; double aS = S_();
      target[j] = save - hh; double bS = S_();
      target[j] = save;
      fd.push_back((aS - bS) / (2 * hh));
      mine.push_back(got[j]);
    }
    snprintf(label, sizeof label, "IPA weights %s vs central difference %s", what, shape);
    reportGrad(label, fd, mine, 3e-5);
  };
  probe("dqs", qs, download((const float*)gqs, lhc), 5);
  probe("dks", ks, download((const float*)gks, lhc), 5);
  probe("dqp", qp, download((const float*)gqp, lhp), 5);
  probe("dkp", kp, download((const float*)gkp, lhp), 5);
  probe("db2d", b2d, download((const float*)gb2d, pb), 5);
  probe("dPointWeightRaw", pwRaw, download((const float*)gpw, H), H < 3 ? H : 3);

  release(bAttn); release(bQs); release(bKs); release(bQp); release(bKp); release(bPw); release(bRaw);
  release(bDa); release(gqs); release(gks); release(gqp); release(gkp); release(gb2d); release(gpw);
}

// ---------------------------------------------------------------- InvariantPointAttention: the outputs
static void ipaOForward(const std::vector<double>& attn, const std::vector<double>& vs,
                        const std::vector<double>& vp, const std::vector<double>& act2d,
                        const std::vector<double>& rig, int L, int H, int Cs, int Pv, int C2,
                        std::vector<double>& fin) {
  const int Fw = H * Cs + 4 * H * Pv + H * C2, base = H * Cs;
  fin.assign((size_t)L * Fw, 0.0);
  for (int q = 0; q < L; ++q) {
    const double* r = &rig[(size_t)q * 12];
    for (int h = 0; h < H; ++h) {
      const double* at = &attn[((size_t)q * H + h) * L];
      double* f = &fin[(size_t)q * Fw];
      for (int c = 0; c < Cs; ++c) {
        double s = 0;
        for (int k = 0; k < L; ++k) s += at[k] * vs[((size_t)k * H + h) * Cs + c];
        f[h * Cs + c] = s;
      }
      for (int p = 0; p < Pv; ++p) {
        double g[3] = {0, 0, 0};
        for (int k = 0; k < L; ++k)
          for (int x = 0; x < 3; ++x) g[x] += at[k] * vp[(((size_t)k * H + h) * Pv + p) * 3 + x];
        double d[3] = {g[0] - r[9], g[1] - r[10], g[2] - r[11]}, loc[3];
        for (int j = 0; j < 3; ++j) loc[j] = r[j] * d[0] + r[3 + j] * d[1] + r[6 + j] * d[2];
        int idx = h * Pv + p;
        for (int x = 0; x < 3; ++x) f[base + x * H * Pv + idx] = loc[x];
        f[base + 3 * H * Pv + idx] = sqrt(std::max(loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2], 1e-16));
      }
      for (int c = 0; c < C2; ++c) {
        double s = 0;
        for (int k = 0; k < L; ++k) s += at[k] * act2d[((size_t)q * L + k) * C2 + c];
        f[base + 4 * H * Pv + h * C2 + c] = s;
      }
    }
  }
}

static void ipaOutputsCase(int L, int H, int Cs, int Pv, int C2) {
  const int Fw = H * Cs + 4 * H * Pv + H * C2, base = H * Cs;
  const size_t an = (size_t)L * H * L, vsn = (size_t)L * H * Cs, vpn = (size_t)L * H * Pv * 3,
               a2n = (size_t)L * L * C2;
  // a genuine softmax for attn, so the magnitudes are the forward's
  std::vector<double> attn(an), vs = randv(vsn), vp = randv(vpn), act2d = randv(a2n), rig = randv((size_t)L * 12),
                      df = randv((size_t)L * Fw, 0.5);
  {
    std::vector<double> raw = randv(an);
    for (size_t r = 0; r < an / L; ++r) {
      double mx = raw[r * L], sum = 0;
      for (int k = 1; k < L; ++k) mx = std::max(mx, raw[r * L + k]);
      for (int k = 0; k < L; ++k) { attn[r * L + k] = exp(raw[r * L + k] - mx); sum += attn[r * L + k]; }
      for (int k = 0; k < L; ++k) attn[r * L + k] /= sum;
    }
  }

  // reference one: the analytic backward, in double
  std::vector<double> rda(an, 0), rdvs(vsn, 0), rdvp(vpn, 0), rda2(a2n, 0), rdr((size_t)L * 12, 0),
      dgAll((size_t)L * H * Pv * 3, 0);
  for (int q = 0; q < L; ++q) {
    const double* r = &rig[(size_t)q * 12];
    for (int h = 0; h < H; ++h) {
      const double* at = &attn[((size_t)q * H + h) * L];
      const double* f = &df[(size_t)q * Fw];
      for (int p = 0; p < Pv; ++p) {
        double g[3] = {0, 0, 0};
        for (int k = 0; k < L; ++k)
          for (int x = 0; x < 3; ++x) g[x] += at[k] * vp[(((size_t)k * H + h) * Pv + p) * 3 + x];
        double d[3] = {g[0] - r[9], g[1] - r[10], g[2] - r[11]}, loc[3];
        for (int j = 0; j < 3; ++j) loc[j] = r[j] * d[0] + r[3 + j] * d[1] + r[6 + j] * d[2];
        int idx = h * Pv + p;
        double n2 = loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2], nrm = sqrt(std::max(n2, 1e-16));
        double dn = f[base + 3 * H * Pv + idx], dloc[3];
        for (int x = 0; x < 3; ++x)
          dloc[x] = f[base + x * H * Pv + idx] + (n2 > 1e-16 ? dn * loc[x] / nrm : 0.0);
        double dd[3];
        for (int m = 0; m < 3; ++m) {
          dd[m] = r[m * 3] * dloc[0] + r[m * 3 + 1] * dloc[1] + r[m * 3 + 2] * dloc[2];
          for (int j = 0; j < 3; ++j) rdr[(size_t)q * 12 + m * 3 + j] += dloc[j] * d[m];
          rdr[(size_t)q * 12 + 9 + m] -= dd[m];
          dgAll[((((size_t)q * H + h) * Pv) + p) * 3 + m] = dd[m];
        }
      }
      for (int k = 0; k < L; ++k) {
        double s = 0;
        for (int c = 0; c < Cs; ++c) {
          s += f[h * Cs + c] * vs[((size_t)k * H + h) * Cs + c];
          rdvs[((size_t)k * H + h) * Cs + c] += at[k] * f[h * Cs + c];
        }
        for (int p = 0; p < Pv; ++p)
          for (int x = 0; x < 3; ++x) {
            double dgv = dgAll[((((size_t)q * H + h) * Pv) + p) * 3 + x];
            s += dgv * vp[(((size_t)k * H + h) * Pv + p) * 3 + x];
            rdvp[(((size_t)k * H + h) * Pv + p) * 3 + x] += at[k] * dgv;
          }
        for (int c = 0; c < C2; ++c) {
          s += f[base + 4 * H * Pv + h * C2 + c] * act2d[((size_t)q * L + k) * C2 + c];
          rda2[((size_t)q * L + k) * C2 + c] += at[k] * f[base + 4 * H * Pv + h * C2 + c];
        }
        rda[((size_t)q * H + h) * L + k] = s;
      }
    }
  }

  auto up = [&](const std::vector<double>& x) { auto fv = toF(x); return uploadNew(fv.data(), fv.size()); };
  void* bAt = up(attn); void* bVs = up(vs); void* bVp = up(vp); void* bA2 = up(act2d); void* bR = up(rig);
  void* bDf = up(df);
  void* gda = alloc(an * 4); void* gdvs = alloc(vsn * 4); void* gdvp = alloc(vpn * 4);
  void* gda2 = alloc(a2n * 4); void* gdr = alloc((size_t)L * 12 * 4);

  IpaOutputsSaved saved{(const float*)bAt, (const float*)bVs, (const float*)bVp, (const float*)bA2,
                        (const float*)bR};
  IpaOutputsGrads grads{(float*)gda, (float*)gdvs, (float*)gdvp, (float*)gda2, (float*)gdr};
  ipaOutputsBwd(saved, (const float*)bDf, L, H, Cs, Pv, C2, grads);

  char label[200], shape[80];
  snprintf(shape, sizeof shape, "L%d H%d Cs%d Pv%d C2%d", L, H, Cs, Pv, C2);
  auto chk = [&](const char* what, const std::vector<double>& ref, void* got, size_t n, double bound) {
    snprintf(label, sizeof label, "IPA outputs bwd %s %s", what, shape);
    report(label, relRms(ref, download((const float*)got, n)), bound);
  };
  chk("dAttn", rda, gda, an, 1e-5);
  chk("dvs", rdvs, gdvs, vsn, 1e-5);
  chk("dvp", rdvp, gdvp, vpn, 1e-5);
  chk("dAct2d", rda2, gda2, a2n, 1e-5);
  chk("dRig", rdr, gdr, (size_t)L * 12, 1e-5);

  auto S_ = [&]() {
    std::vector<double> fin;
    ipaOForward(attn, vs, vp, act2d, rig, L, H, Cs, Pv, C2, fin);
    double acc = 0;
    for (size_t i = 0; i < fin.size(); ++i) acc += df[i] * fin[i];
    return acc;
  };
  auto probe = [&](const char* what, std::vector<double>& target, const std::vector<float>& got, int count) {
    std::vector<double> fd, mine;
    std::uniform_int_distribution<size_t> U(0, target.size() - 1);
    for (int t = 0; t < count; ++t) {
      size_t j = U(rng);
      double save = target[j], hh = 1e-5 * std::max(1.0, fabs(save));
      target[j] = save + hh; double aS = S_();
      target[j] = save - hh; double bS = S_();
      target[j] = save;
      fd.push_back((aS - bS) / (2 * hh));
      mine.push_back(got[j]);
    }
    snprintf(label, sizeof label, "IPA outputs %s vs central difference %s", what, shape);
    reportGrad(label, fd, mine, 3e-5);
  };
  probe("dAttn", attn, download((const float*)gda, an), 5);
  probe("dvs", vs, download((const float*)gdvs, vsn), 5);
  probe("dvp", vp, download((const float*)gdvp, vpn), 5);
  probe("dAct2d", act2d, download((const float*)gda2, a2n), 5);
  probe("dRig", rig, download((const float*)gdr, (size_t)L * 12), 6);

  release(bAt); release(bVs); release(bVp); release(bA2); release(bR); release(bDf);
  release(gda); release(gdvs); release(gdvp); release(gda2); release(gdr);
}

// ---------------------------------------------------------------- --mem: what each backward HOLDS, at real shapes
// The test cases above are deliberately tiny, so they say nothing about memory. These are AF2's own widths at a
// design-sized complex, which is what has to fit in 16 GB of unified memory.
static void memReport() {
  designShareScratch();
  const int N = 192, Cz = 128, I = 128, Cm = 256, O = 32, F = 128, S = 8, H = 4, D = 32;
  const size_t n2 = (size_t)N * N;
  printf("AF2 widths at N=%d (pair %d, msa %d x %d, triangle intermediate %d)\n", N, Cz, S, Cm, I);

  auto z = [](size_t n) { return alloc(n * sizeof(float)); };
  bool isolate = true;
  auto show = [&](const char* what) {
    sync();
    if (!isolate) return;
    printf("  %-40s scratch %8.1f MB   peak live %8.1f MB\n", what,
           scratchHeld() / 1048576.0, peakLive() / 1048576.0);
    releaseScratch();
    resetPeak();
  };

  {   // Transition, on the pair track
    void* act = z(n2 * Cz); void* dOut = z(n2 * Cz); void* dAct = z(n2 * Cz);
    void* ls = z(Cz); void* lo = z(Cz); void* W1 = z((size_t)Cz * Cz * 4); void* b1 = z((size_t)Cz * 4);
    void* W2 = z((size_t)Cz * 4 * Cz); void* b2 = z(Cz);
    transitionBwd((const float*)act, (const float*)dOut, n2, Cz, Cz * 4, (const float*)ls, (const float*)lo,
                  (const float*)W1, (const float*)b1, (const float*)W2, (const float*)b2, (float*)dAct);
    show("transitionBwd (pair track)");
    for (void* p : {act, dOut, dAct, ls, lo, W1, b1, W2, b2}) release(p);
  }
  {   // OuterProductMean
    void* msa = z((size_t)S * N * Cm); void* mask = z((size_t)S * N); void* dOut = z(n2 * F);
    void* dMsa = z((size_t)S * N * Cm);
    void* ls = z(Cm); void* lo = z(Cm); void* Wl = z((size_t)Cm * O); void* bl = z(O);
    void* Wr = z((size_t)Cm * O); void* br = z(O); void* Wo = z((size_t)O * O * F);
    outerProductMeanBwd((const float*)msa, (const float*)mask, S, N, Cm, O, F, (const float*)ls, (const float*)lo,
                        (const float*)Wl, (const float*)bl, (const float*)Wr, (const float*)br,
                        (const float*)Wo, (const float*)dOut, (float*)dMsa);
    show("outerProductMeanBwd");
    for (void* p : {msa, mask, dOut, dMsa, ls, lo, Wl, bl, Wr, br, Wo}) release(p);
  }
  {   // TriangleMultiplication - the one flagged as the ceiling
    void* pair = z(n2 * Cz); void* mask = z(n2); void* dOut = z(n2 * Cz); void* dPair = z(n2 * Cz);
    void* v1 = z((size_t)Cz * I); void* v2 = z((size_t)I * Cz); void* s1 = z(Cz); void* s2 = z(I);
    void* bI = z(I); void* bC = z(Cz); void* wgg = z((size_t)Cz * Cz);
    TriMulWeights w{};
    w.lnScale = (const float*)s1; w.lnOffset = (const float*)s1;
    w.Wl = (const float*)v1; w.bl = (const float*)bI;
    w.Wr = (const float*)v1; w.br = (const float*)bI;
    w.Wlg = (const float*)v1; w.blg = (const float*)bI;
    w.Wrg = (const float*)v1; w.brg = (const float*)bI;
    w.centerScale = (const float*)s2; w.centerOffset = (const float*)s2;
    w.Wo = (const float*)v2; w.bo = (const float*)bC;
    w.Wg = (const float*)wgg; w.bg = (const float*)bC;
    triangleMultiplicationBwd((const float*)pair, (const float*)mask, N, Cz, I, true, w,
                              (const float*)dOut, (float*)dPair);
    show("triangleMultiplicationBwd");
    for (void* p : {pair, mask, dOut, dPair, v1, v2, s1, s2, bI, bC, wgg}) release(p);
  }
  {   // TriangleAttention, through biasedSelfAttentionBwd
    const int W = H * D;
    void* pair = z(n2 * Cz); void* mask = z(n2); void* dOut = z(n2 * Cz); void* dPair = z(n2 * Cz);
    void* sc = z(Cz); void* f2 = z((size_t)Cz * H); void* wp = z((size_t)Cz * W); void* wo = z((size_t)W * Cz);
    void* bw = z(W); void* bc = z(Cz);
    BiasedAttnWeights w{};
    w.qScale = (const float*)sc; w.qOffset = (const float*)sc; w.feat2dW = (const float*)f2;
    w.attn.Wq = (const float*)wp; w.attn.Wk = (const float*)wp; w.attn.Wv = (const float*)wp;
    w.attn.Wg = (const float*)wp; w.attn.bg = (const float*)bw;
    w.attn.Wo = (const float*)wo; w.attn.bo = (const float*)bc;
    biasedSelfAttentionBwd((const float*)pair, (const float*)pair, (const float*)mask,
                           N, N, Cz, Cz, H, D, Cz, w, (const float*)dOut, (float*)dPair, (float*)dPair);
    show("biasedSelfAttentionBwd (triangle attn)");
    for (void* p : {pair, mask, dOut, dPair, sc, f2, wp, wo, bw, bc}) release(p);
  }
  printf("\n  a pair tensor at this size is %.1f MB; the Evoformer holds several per block\n",
         n2 * (size_t)Cz * 4 / 1048576.0);
}

// What a BLOCK holds: the same four modules one after another with nothing released between them, which is what an
// Evoformer block does. Without shareScratch this is the sum of the four; with it, the maximum.
// Run it twice to see the difference: LOCALFOLD_SHARED_SCRATCH=0 is the control.
static void memBlock(int N) {
  designShareScratch();
  const int Cz = 128, I = 128, Cm = 256, O = 32, F = 128, S = 8, H = 4, D = 32;
  const size_t n2 = (size_t)N * N;
  auto z = [](size_t n) { return alloc(n * sizeof(float)); };
  releaseScratch();
  resetPeak();

  void* pair = z(n2 * Cz); void* pmask = z(n2); void* dOutP = z(n2 * Cz); void* dPair = z(n2 * Cz);
  void* msa = z((size_t)S * N * Cm); void* mmask = z((size_t)S * N); void* dMsa = z((size_t)S * N * Cm);
  void* dOutZ = z(n2 * F);
  void* sC = z(Cz); void* sI = z(I); void* sM = z(Cm);
  void* bI = z(I); void* bC = z(Cz); void* bO = z(O);
  void* wCI = z((size_t)Cz * I); void* wIC = z((size_t)I * Cz); void* wCC = z((size_t)Cz * Cz);
  void* wC4 = z((size_t)Cz * Cz * 4); void* b4 = z((size_t)Cz * 4);
  void* wMO = z((size_t)Cm * O); void* wOOF = z((size_t)O * O * F);
  const int W = H * D;
  void* w2d = z((size_t)Cz * H); void* wqkv = z((size_t)Cz * W); void* wout = z((size_t)W * Cz); void* bW = z(W);

  // the pair transition
  transitionBwd((const float*)pair, (const float*)dOutP, n2, Cz, Cz * 4, (const float*)sC, (const float*)sC,
                (const float*)wC4, (const float*)b4, (const float*)wC4, (const float*)bC, (float*)dPair);
  // the outer product
  outerProductMeanBwd((const float*)msa, (const float*)mmask, S, N, Cm, O, F, (const float*)sM, (const float*)sM,
                      (const float*)wMO, (const float*)bO, (const float*)wMO, (const float*)bO,
                      (const float*)wOOF, (const float*)dOutZ, (float*)dMsa);
  // both triangle multiplications
  TriMulWeights tw{};
  tw.lnScale = (const float*)sC; tw.lnOffset = (const float*)sC;
  tw.Wl = (const float*)wCI; tw.bl = (const float*)bI;
  tw.Wr = (const float*)wCI; tw.br = (const float*)bI;
  tw.Wlg = (const float*)wCI; tw.blg = (const float*)bI;
  tw.Wrg = (const float*)wCI; tw.brg = (const float*)bI;
  tw.centerScale = (const float*)sI; tw.centerOffset = (const float*)sI;
  tw.Wo = (const float*)wIC; tw.bo = (const float*)bC;
  tw.Wg = (const float*)wCC; tw.bg = (const float*)bC;
  for (bool outgoing : {true, false})
    triangleMultiplicationBwd((const float*)pair, (const float*)pmask, N, Cz, I, outgoing, tw,
                              (const float*)dOutP, (float*)dPair);
  // both triangle attentions
  BiasedAttnWeights bw{};
  bw.qScale = (const float*)sC; bw.qOffset = (const float*)sC; bw.feat2dW = (const float*)w2d;
  bw.attn.Wq = (const float*)wqkv; bw.attn.Wk = (const float*)wqkv; bw.attn.Wv = (const float*)wqkv;
  bw.attn.Wg = (const float*)wqkv; bw.attn.bg = (const float*)bW;
  bw.attn.Wo = (const float*)wout; bw.attn.bo = (const float*)bC;
  for (int t = 0; t < 2; ++t)
    biasedSelfAttentionBwd((const float*)pair, (const float*)pair, (const float*)pmask,
                           N, N, Cz, Cz, H, D, Cz, bw, (const float*)dOutP, (float*)dPair, (float*)dPair);
  sync();
  const bool off = getenv("LOCALFOLD_SHARED_SCRATCH") && !atoi(getenv("LOCALFOLD_SHARED_SCRATCH"));
  printf("one Evoformer block's backward at N=%d, %s (a pair buffer is %.1f MB, the sharing floor %.0f MB):\n",
         N, off ? "sharing OFF (the control)" : "sharing on", n2 * (size_t)Cz * 4 / 1048576.0,
         (getenv("LOCALFOLD_SHARED_MIN") ? atoll(getenv("LOCALFOLD_SHARED_MIN")) : (32LL << 20)) / 1048576.0);
  printf("  transition + outer product + 2 triangle multiplications + 2 triangle attentions\n");
  printf("  scratch held %8.1f MB     peak live %8.1f MB\n", scratchHeld() / 1048576.0, peakLive() / 1048576.0);
  for (void* p : {pair, pmask, dOutP, dPair, msa, mmask, dMsa, dOutZ, sC, sI, sM, bI, bC, bO, wCI, wIC, wCC,
                  wC4, b4, wMO, wOOF, w2d, wqkv, wout, bW}) release(p);
}

int main(int argc, char** argv) {
  setSource("design", PORT_SOURCE);
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--mem")) { memReport(); return 0; }
    else if (!strcmp(argv[i], "--mem-block")) {
      memBlock(i + 1 < argc ? atoi(argv[i + 1]) : 192);
      return 0;
    }
  // the sharing is on for the test too, so the differential cases run against the aliased buffers a fold would use
  // (LOCALFOLD_SHARED_SCRATCH=0 is the control; LOCALFOLD_SCRIBBLE=1 NaN-fills every slot at a change of family and
  // the results must not move - see docs/BINDCRAFT2.md)
  designShareScratch();
  printf("LayerNorm backward\n");
  for (auto [rows, C] : std::vector<std::pair<size_t, int>>{{8, 128}, {64, 256}, {333, 128}, {1024, 384}, {7, 64}})
    lnCase(rows, C, 1e-5);
  printf("linear backward\n");
  for (auto [rows, in, out] : std::vector<std::tuple<size_t, int, int>>{
           {64, 128, 128}, {333, 256, 64}, {1024, 128, 256}, {37, 64, 100}})
    linCase(rows, in, out);
  printf("elementwise backward\n");
  for (size_t n : {1024u, 70001u}) ewiseCases(n);
  printf("Transition backward (the whole module)\n");
  // AF2's shapes: the pair track is C=128 with factor 4, the MSA track C=256 with factor 4
  for (auto [rows, C, H] : std::vector<std::tuple<size_t, int, int>>{
           {64, 128, 512}, {333, 128, 512}, {96, 256, 1024}, {17, 64, 128}})
    trCase(rows, C, H);
  printf("OuterProductMean backward (the whole module)\n");
  // the reference's own shapes are O=32 and F=c_z=128; the small cases keep the double forward affordable
  opmCase(5, 24, 48, 8, 16, 8);
  opmCase(8, 20, 32, 32, 16, 8);
  opmCase(4, 17, 24, 8, 12, 5);      // a tile that does not divide N
  opmCase(3, 16, 32, 32, 128, 4);    // the real O and F
  printf("attention backward (flash-shaped: no score stored)\n");
  attnCase(6, 12, 12, 4, 32);      // the triangle's shape: D = 128 / 4 heads
  attnCase(5, 10, 14, 3, 8);       // Q != K, as MSA attention has
  attnCase(4, 9, 9, 2, 64);        // the widest head this path holds
  attnCase(3, 7, 7, 8, 16);
  printf("Attention module backward (projections, gate, output)\n");
  attnModCase(5, 9, 9, 16, 16, 4, 8, 16, true, true);    // self-attention with gating: what AF2 runs
  attnModCase(4, 8, 8, 12, 12, 2, 16, 12, false, true);  // ungated, as the triangle's config can be
  attnModCase(4, 7, 10, 12, 14, 3, 8, 10, true, false);  // cross: separate q_data and m_data
  printf("biased self-attention (MSARowAttentionWithPairBias / TriangleAttention)\n");
  biasedCase(7, 7, 12, 12, 3, 8, 12, true);    // TriangleAttention: one LayerNorm, bias from the attended tensor
  biasedCase(5, 6, 10, 14, 2, 8, 10, false);   // MSARowAttentionWithPairBias: two, and the bias from the pair
  printf("MSAColumnAttention (the per-column orientation, no pair bias)\n");
  mcolCase(6, 5, 12, 3, 8, 12);
  mcolCase(4, 7, 16, 2, 8, 16);
  printf("MSAColumnGlobalAttention (one masked-mean query, k/v with no head)\n");
  gattnCase(6, 5, 16, 4, 4, 16);
  gattnCase(9, 4, 12, 2, 6, 12);
  printf("TriangleMultiplication backward (the whole module)\n");
  triCase(10, 16, 12, true);
  triCase(10, 16, 12, false);
  triCase(7, 8, 8, true);
  triCase(7, 8, 8, false);
  printf("the heads: distogram and predicted aligned error\n");
  headCase(6, 10, 8);
  headCase(5, 8, 16);
  printf("the metrics a loss reads off the PAE head\n");
  paeMetricCase(8, 12);
  paeMetricCase(11, 9);
  printf("EvoformerIteration backward (the WHOLE block, eight residual modules)\n");
  evoCase({3, 5, 8, 6, 2, 4, 2, 3, 6, 8, 6, 4, false});
  evoCase({2, 4, 8, 6, 2, 4, 2, 3, 6, 8, 6, 4, true});     // outer product first, as the multimer config has it
  printf("the structure module: the backbone update (Alg. 23)\n");
  rigidUpdCase(5);
  rigidUpdCase(17);
  printf("the structure module: the point projections into the global frame\n");
  pointsGlobalCase(6, 3, 4);        // AF2's IPA: 12 heads of 4 query points and 8 value points
  pointsGlobalCase(4, 12, 8);
  pointsGlobalCase(9, 1, 1);        // the degenerate shape, where a stride bug has nowhere to hide and must not
  printf("the structure module: InvariantPointAttention's attention weights (Alg. 22)\n");
  ipaWeightsCase(7, 3, 6, 2);
  ipaWeightsCase(5, 2, 4, 4);       // AF2's own num_point_qk
  ipaWeightsCase(6, 1, 16, 4);      // AF2's num_scalar_qk, one head
  printf("the structure module: InvariantPointAttention's outputs\n");
  ipaOutputsCase(6, 3, 4, 2, 5);
  ipaOutputsCase(5, 2, 6, 4, 8);
  ipaOutputsCase(4, 1, 16, 8, 12);  // AF2's num_scalar_v and num_point_v
  printf(failures ? "\n%d FAILED\n" : "\nall ok\n", failures);
  return failures ? 1 : 0;
}
