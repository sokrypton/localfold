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
  if (shared) biasN = actN;
  else lnForward(biasSrc, nbRows, Cb, bScale, bOffset, 1e-5, biasN);
  std::vector<double> nb((size_t)H * nbRows, 0.0);
  for (size_t r = 0; r < nbRows; ++r)
    for (int h = 0; h < H; ++h) {
      double s = 0;
      for (int c = 0; c < Cb; ++c) s += biasN[r * Cb + c] * feat2dW[(size_t)c * H + h];
      nb[(size_t)h * nbRows + r] = s;          // [h][q][k]
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

int main() {
  setSource("design", PORT_SOURCE);
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
  printf("TriangleMultiplication backward (the whole module)\n");
  triCase(10, 16, 12, true);
  triCase(10, 16, 12, false);
  triCase(7, 8, 8, true);
  triCase(7, 8, 8, false);
  printf(failures ? "\n%d FAILED\n" : "\nall ok\n", failures);
  return failures ? 1 : 0;
}
