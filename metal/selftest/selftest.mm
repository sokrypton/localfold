// metal/core's self-test: the GEMM (every operand layout, the precisions, batches, the fused epilogues) and the
// LayerNorm against the host, in double. `bash metal/build.sh selftest && metal/selftest/localfold-selftest`
#include "core.h"
#include <cmath>
#include <random>

extern const char* PORT_SOURCE;
using namespace mt;

static std::mt19937 rng(7);
static std::vector<float> randv(size_t n, float s = 1.f) {
  std::normal_distribution<float> N(0, s);
  std::vector<float> v(n);
  for (auto& x : v) x = N(rng);
  return v;
}
static std::vector<half> toH(const std::vector<float>& v) { std::vector<half> h(v.size()); for (size_t i = 0; i < v.size(); ++i) h[i] = (half)v[i]; return h; }
static int failures = 0;
static void report(const char* what, double rel, double bound) {
  bool ok = rel <= bound && std::isfinite(rel);
  if (!ok) ++failures;
  printf("  %-58s relRMS %.2e %s\n", what, rel, ok ? "ok" : "FAIL");
}
static double relRms(const std::vector<double>& ref, const std::vector<float>& got) {
  double num = 0, den = 0;
  for (size_t i = 0; i < ref.size(); ++i) { double d = ref[i] - got[i]; num += d * d; den += ref[i] * ref[i]; }
  return sqrt(num / std::max(den, 1e-30));
}

// Y[r][o] = sum_k X(r, k) W(k, o): the layouts as core.h's Gemm names them
static void gemmCase(const char* what, size_t R, int K, int O, bool tx, bool tw, DT dx, DT dw, DT dy, bool half_, int batch, double bound) {
  std::vector<float> X = randv((size_t)batch * R * K), W = randv((size_t)batch * K * O, 0.1f);
  std::vector<double> ref((size_t)batch * R * O, 0);
  for (int b = 0; b < batch; ++b)
    for (size_t r = 0; r < R; ++r)
      for (int o = 0; o < O; ++o) {
        double s = 0;
        for (int k = 0; k < K; ++k) {
          float x = tx ? X[(size_t)b * R * K + (size_t)k * R + r] : X[(size_t)b * R * K + r * K + k];
          float w = tw ? W[(size_t)b * K * O + (size_t)o * K + k] : W[(size_t)b * K * O + (size_t)k * O + o];
          s += (double)x * w;
        }
        ref[(size_t)b * R * O + r * O + o] = s;
      }
  void* dX = dx == F32 ? (void*)uploadNew(X.data(), X.size()) : (void*)uploadNew(toH(X).data(), X.size());
  void* dW = dw == F32 ? (void*)uploadNew(W.data(), W.size()) : (void*)uploadNew(toH(W).data(), W.size());
  void* dY = alloc(ref.size() * 4);
  Gemm g{}; g.X = dX; g.tx = dx; g.transX = tx; g.W = dW; g.tw = dw; g.transW = tw; g.Y = dY; g.ty = dy;
  g.rows = R; g.in = K; g.out = O; g.half = half_; g.batch = batch; g.sx = (int64_t)R * K; g.sw = (int64_t)K * O; g.sy = (int64_t)R * O;
  gemm(g);
  std::vector<float> got(ref.size());
  if (dy == F32) got = download((const float*)dY, ref.size());
  else { auto h = download((const half*)dY, ref.size()); for (size_t i = 0; i < h.size(); ++i) got[i] = (float)h[i]; }
  char label[160]; snprintf(label, sizeof label, "%s %zux%dx%d%s%s b%d", what, R, K, O, tx ? " tX" : "", tw ? " tW" : "", batch);
  report(label, relRms(ref, got), bound);
  release(dX); release(dW); release(dY);
}

int main() {
  setenv("LOCALFOLD_LN_EMIT", "2", 0);
  setSource("selftest", PORT_SOURCE);
  printf("GEMM\n");
  for (auto [R, K, O] : std::vector<std::tuple<size_t, int, int>>{{37, 100, 70}, {300, 256, 512}, {1000, 128, 48}, {64, 1152, 3456}, {5, 6, 128}}) {
    gemmCase("f32", R, K, O, false, false, F32, F32, F32, false, 1, 1e-5);
    gemmCase("f16 -> f32", R, K, O, false, false, F16, F16, F32, false, 1, 2e-3);
    gemmCase("f16 -> f16", R, K, O, false, false, F16, F16, F16, false, 1, 5e-3);
    gemmCase("f32 x f16, half", R, K, O, false, false, F32, F16, F32, true, 1, 2e-3);
  }
  gemmCase("f32", 200, 64, 200, false, true, F32, F32, F32, false, 3, 1e-5);
  gemmCase("f32", 200, 200, 64, true, false, F32, F32, F32, false, 3, 1e-5);
  gemmCase("f16 -> f32", 264, 264, 264, false, true, F16, F16, F32, false, 4, 2e-3);
  gemmCase("f16 -> f32", 264, 264, 264, true, false, F16, F16, F32, false, 4, 2e-3);

  printf("epilogues\n");
  {   // bias, relu, beta
    size_t R = 150; int K = 96, O = 80;
    auto X = randv(R * K), W = randv((size_t)K * O, 0.1f), b = randv(O), Y0 = randv(R * O);
    std::vector<double> ref(R * O);
    for (size_t r = 0; r < R; ++r) for (int o = 0; o < O; ++o) {
      double s = 0; for (int k = 0; k < K; ++k) s += (double)X[r * K + k] * W[(size_t)k * O + o];
      ref[r * O + o] = std::max(0.0, 0.5 * s + 2.0 * Y0[r * O + o] + b[o]);
    }
    float* dX = uploadNew(X.data(), X.size()); float* dW = uploadNew(W.data(), W.size()); float* db = uploadNew(b.data(), b.size());
    float* dY = uploadNew(Y0.data(), Y0.size());
    Gemm g{}; g.X = dX; g.W = dW; g.Y = dY; g.rows = R; g.in = K; g.out = O; g.alpha = 0.5f; g.beta = 2.f; g.bias = db; g.relu = true;
    gemm(g);
    report("alpha, beta, bias, relu", relRms(ref, download(dY, R * O)), 1e-5);
  }
  for (size_t R : {size_t(333), size_t(68), size_t(16)}) {   // SwiGLU in blocks of 8 (tall tile + short tiles)
    int K = 128, Hd = 256;
    auto X = randv(R * K), W = randv((size_t)K * 2 * Hd, 0.1f);       // W [K][2Hd]: a then b
    std::vector<float> Wp((size_t)K * 2 * Hd);
    for (int k = 0; k < K; ++k) for (int c = 0; c < 2 * Hd; ++c) {
      int m = c / 16, q = c % 16;
      Wp[(size_t)k * 2 * Hd + c] = W[(size_t)k * 2 * Hd + (q < 8 ? 8 * m + q : Hd + 8 * m + q - 8)];
    }
    std::vector<double> ref(R * Hd);
    for (size_t r = 0; r < R; ++r) for (int c = 0; c < Hd; ++c) {
      double a = 0, b = 0;
      for (int k = 0; k < K; ++k) { a += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * 2 * Hd + c]; b += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * 2 * Hd + Hd + c]; }
      ref[r * Hd + c] = a / (1 + exp(-a)) * b;
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(Wp).data(), Wp.size()); half* dG = allocT<half>(R * Hd);
    gemmSwiglu(dX, dW, dG, R, K, Hd);
    auto h = download(dG, R * Hd); std::vector<float> got(h.begin(), h.end());
    report("swiglu", relRms(ref, got), 5e-3);
    release(dX); release(dW); release(dG);
  }
  for (bool pairs2 : {false, true}) {   // coupling combine (BioEmu2 couplingSplit): blocks of 8 and blocks of 2
    size_t R = 213, T = 36, Tc = 18; int K = 128, Hd = 192, H2 = 2 * Hd;
    auto X = randv(R * K), W = randv((size_t)K * H2, 0.1f), P = randv(T * H2, 0.3f), QB = randv(Tc * 2 * H2, 0.3f);
    std::vector<float> Wp((size_t)K * H2), P_p(T * H2), QB_p(Tc * 2 * H2), stats(R * 8, 0.f);
    const int blk = pairs2 ? 2 : 8;
    auto pack = [&](const float* src, float* dst, size_t rows) {
      for (size_t r = 0; r < rows; ++r) for (int c = 0; c < H2; ++c) {
        int m = c / (2 * blk), q = c % (2 * blk);
        dst[r * H2 + c] = src[r * H2 + (q < blk ? blk * m + q : Hd + blk * m + q - blk)];
      }
    };
    pack(W.data(), Wp.data(), K); pack(P.data(), P_p.data(), T); pack(QB.data(), QB_p.data(), Tc * 2);
    for (size_t r = 0; r < R; ++r) {
      stats[r * 8 + 0] = 0.8f; stats[r * 8 + 2] = -0.2f; stats[r * 8 + 3] = 1.1f; stats[r * 8 + 5] = 0.9f;
      stats[r * 8 + 6] = (r % 9 == 0) ? -1.f : (float)(r / 6);
      stats[r * 8 + 7] = (float)(r / 12);
    }
    std::vector<double> ref(R * Hd);
    for (size_t r = 0; r < R; ++r) {
      double inv = stats[r * 8 + 0], dq = stats[r * 8 + 2], sq = stats[r * 8 + 3], sa = stats[r * 8 + 5];
      int st = (int)stats[r * 8 + 6]; uint tq = (uint)stats[r * 8 + 7];
      for (int c = 0; c < Hd; ++c) {
        double za = 0, zb = 0;
        for (int k = 0; k < K; ++k) {
          za += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * H2 + c];
          zb += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * H2 + Hd + c];
        }
        double pa = st >= 0 ? sa * P[(size_t)st * H2 + c] : 0.0;
        double pb = st >= 0 ? sa * P[(size_t)st * H2 + Hd + c] : 0.0;
        double ua = inv * (pa + sq * za + dq * QB[(size_t)tq * 2 * H2 + c]) + QB[((size_t)tq * 2 + 1) * H2 + c];
        double ub = inv * (pb + sq * zb + dq * QB[(size_t)tq * 2 * H2 + Hd + c]) + QB[((size_t)tq * 2 + 1) * H2 + Hd + c];
        ref[r * Hd + c] = ua / (1.0 + exp(-ua)) * ub;
      }
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(Wp).data(), Wp.size()); half* dG = allocT<half>(R * Hd);
    float *dS = uploadNew(stats.data(), stats.size()), *dP = uploadNew(P_p.data(), P_p.size()), *dQB = uploadNew(QB_p.data(), QB_p.size());
    if (gemmCouplingCombine(dX, dW, dG, R, K, Hd, dS, dP, dQB, pairs2)) {
      auto h = download(dG, R * Hd);
      report(pairs2 ? "coupling combine (blocks of 2)" : "coupling combine (blocks of 8)", relRms(ref, std::vector<float>(h.begin(), h.end())), 5e-3);
    } else { ++failures; printf("  coupling combine: FAIL (not taken)\n"); }
    release(dX); release(dW); release(dG); release(dS); release(dP); release(dQB);
  }
  {   // gated add
    size_t R = 300; int K = 128, O = 128;
    auto X = randv(R * K), W = randv((size_t)K * O, 0.1f), aux = randv(R * O), Y0 = randv(R * O);
    std::vector<double> ref(R * O);
    for (size_t r = 0; r < R; ++r) for (int o = 0; o < O; ++o) {
      double s = 0; for (int k = 0; k < K; ++k) s += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * O + o];
      ref[r * O + o] = Y0[r * O + o] + (double)(float)(half)aux[r * O + o] / (1 + exp(-s));
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(W).data(), W.size()); half* dA = uploadNew(toH(aux).data(), aux.size());
    float* dY = uploadNew(Y0.data(), Y0.size());
    gemmGatedAdd(dX, dW, dA, dY, R, K, O);
    report("gated add", relRms(ref, download(dY, R * O)), 1e-3);
  }
  for (int O : {128, 64, 256, 384, 512}) {   // the next LayerNorm emitted by the epilogue (two norms of the same rows): Y += X W, then LN(Y) twice
    if (O > 256 && !gemmTensorOps()) continue;
    size_t R = 300; int K = 256;
    auto X = randv(R * K), W = randv((size_t)K * O, 0.1f), Y0 = randv(R * O), s1 = randv(O), o1 = randv(O), s2 = randv(O), o2 = randv(O);
    std::vector<double> y(R * O), ref1(R * O), ref2(R * O);
    for (size_t r = 0; r < R; ++r) {
      double m = 0, v = 0;
      for (int o = 0; o < O; ++o) {
        double t = 0; for (int k = 0; k < K; ++k) t += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * O + o];
        y[r * O + o] = Y0[r * O + o] + t; m += y[r * O + o];
      }
      m /= O;
      for (int o = 0; o < O; ++o) v += (y[r * O + o] - m) * (y[r * O + o] - m);
      double inv = 1 / sqrt(v / O + 1e-5);
      for (int o = 0; o < O; ++o) {
        ref1[r * O + o] = (y[r * O + o] - m) * inv * s1[o] + o1[o];
        ref2[r * O + o] = (y[r * O + o] - m) * inv * s2[o] + o2[o];
      }
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(W).data(), W.size());
    float* dY = uploadNew(Y0.data(), Y0.size());
    float *ds1 = uploadNew(s1.data(), O), *do1 = uploadNew(o1.data(), O), *ds2 = uploadNew(s2.data(), O), *do2 = uploadNew(o2.data(), O);
    half* dN1 = allocT<half>(R * O); half* dN2 = allocT<half>(R * O);
    Gemm g{}; g.X = dX; g.tx = F16; g.W = dW; g.tw = F16; g.Y = dY; g.rows = R; g.in = K; g.out = O; g.beta = 1.f;
    g.lnOut = dN1; g.lnScale = ds1; g.lnOffset = do1; g.lnOut2 = dN2; g.lnScale2 = ds2; g.lnOffset2 = do2;
    if (gemm(g)) {
      char l0[64], l1[64], l2[64];
      snprintf(l0, sizeof l0, "emitted LayerNorm (C %d): the output", O);
      snprintf(l1, sizeof l1, "emitted LayerNorm (C %d): the first norm", O);
      snprintf(l2, sizeof l2, "emitted LayerNorm (C %d): the second norm", O);
      report(l0, relRms(y, download(dY, R * O)), 1e-3);
      auto h1 = download(dN1, R * O), h2 = download(dN2, R * O);
      report(l1, relRms(ref1, std::vector<float>(h1.begin(), h1.end())), 3e-3);
      report(l2, relRms(ref2, std::vector<float>(h2.begin(), h2.end())), 3e-3);
    } else { ++failures; printf("  emitted LayerNorm (C %d): FAIL (not taken)\n", O); }
  }
  for (int O : {128, 64, 256, 384, 512}) {   // ...and by the triangle's fused tail: pair += Xv Wv sigmoid(Xg Wg), then LN(pair)
    if (O > 256 && !gemmTensorOps()) continue;
    size_t R = 300; int K = O;
    auto Xg = randv(R * K), Wg = randv((size_t)K * O, 0.1f), Xv = randv(R * K), Wv = randv((size_t)K * O, 0.1f), Y0 = randv(R * O);
    auto s1 = randv(O), o1 = randv(O);
    std::vector<double> y(R * O), ref(R * O);
    for (size_t r = 0; r < R; ++r) {
      double m = 0, v = 0;
      for (int o = 0; o < O; ++o) {
        double a = 0, b = 0;
        for (int k = 0; k < K; ++k) {
          a += (double)(float)(half)Xg[r * K + k] * (float)(half)Wg[(size_t)k * O + o];
          b += (double)(float)(half)Xv[r * K + k] * (float)(half)Wv[(size_t)k * O + o];
        }
        y[r * O + o] = Y0[r * O + o] + (double)(float)(half)b / (1 + exp(-a)); m += y[r * O + o];
      }
      m /= O;
      for (int o = 0; o < O; ++o) v += (y[r * O + o] - m) * (y[r * O + o] - m);
      double inv = 1 / sqrt(v / O + 1e-5);
      for (int o = 0; o < O; ++o) ref[r * O + o] = (y[r * O + o] - m) * inv * s1[o] + o1[o];
    }
    half* dXg = uploadNew(toH(Xg).data(), Xg.size()); half* dWg = uploadNew(toH(Wg).data(), Wg.size());
    half* dXv = uploadNew(toH(Xv).data(), Xv.size()); half* dWv = uploadNew(toH(Wv).data(), Wv.size());
    float* dY = uploadNew(Y0.data(), Y0.size()); half* tmp = allocT<half>(R * O); half* dN = allocT<half>(R * O);
    float *ds1 = uploadNew(s1.data(), O), *do1 = uploadNew(o1.data(), O);
    if (gemmGatedAddDual(dXg, dWg, dXv, dWv, dY, R, K, O, nullptr, nullptr, tmp, nullptr, dN, ds1, do1)) {
      char l0[64], l1[64];
      snprintf(l0, sizeof l0, "emitted LayerNorm (C %d), the triangle's tail: the pair", O);
      snprintf(l1, sizeof l1, "emitted LayerNorm (C %d), the triangle's tail: the norm", O);
      report(l0, relRms(y, download(dY, R * O)), 2e-3);
      auto h = download(dN, R * O);
      report(l1, relRms(ref, std::vector<float>(h.begin(), h.end())), 3e-3);
    } else { ++failures; printf("  emitted LayerNorm, the triangle's tail (C %d): FAIL (not taken)\n", O); }
  }
  for (int O : {384, 512}) {   // ...and the triangle's fused tail without emitted LayerNorm at C = 384 (OpenDDE) and C = 512 (IntelliFold-2)
    size_t R = 160; int K = O;
    auto Xg = randv(R * K), Wg = randv((size_t)K * O, 0.1f), Xv = randv(R * K), Wv = randv((size_t)K * O, 0.1f), Y0 = randv(R * O);
    std::vector<double> y(R * O);
    for (size_t r = 0; r < R; ++r) {
      for (int o = 0; o < O; ++o) {
        double a = 0, b = 0;
        for (int k = 0; k < K; ++k) {
          a += (double)(float)(half)Xg[r * K + k] * (float)(half)Wg[(size_t)k * O + o];
          b += (double)(float)(half)Xv[r * K + k] * (float)(half)Wv[(size_t)k * O + o];
        }
        y[r * O + o] = Y0[r * O + o] + (double)(float)(half)b / (1 + exp(-a));
      }
    }
    half* dXg = uploadNew(toH(Xg).data(), Xg.size()); half* dWg = uploadNew(toH(Wg).data(), Wg.size());
    half* dXv = uploadNew(toH(Xv).data(), Xv.size()); half* dWv = uploadNew(toH(Wv).data(), Wv.size());
    float* dY = uploadNew(Y0.data(), Y0.size()); half* tmp = allocT<half>(R * O);
    gemmGatedAddDual(dXg, dWg, dXv, dWv, dY, R, K, O, nullptr, nullptr, tmp);
    char l0[64];
    snprintf(l0, sizeof l0, "the triangle's fused tail (C %d): the pair", O);
    report(l0, relRms(y, download(dY, R * O)), 2e-3);
    release(dXg); release(dWg); release(dXv); release(dWv); release(dY); release(tmp);
  }
  {   // the same two over a half pair (AF3's pairformer stack): the residual summed in float, rounded once on its store,
      // the norm taken from the stored half values
    size_t R = 300; int K = 128, O = 128;
    auto X = randv(R * K), W = randv((size_t)K * O, 0.1f), Y0 = randv(R * O), s1 = randv(O), o1 = randv(O);
    auto Xv = randv(R * K), Wv = randv((size_t)K * O, 0.1f);
    std::vector<double> y(R * O), ref(R * O), y2(R * O), ref2(R * O);
    auto norm = [&](const std::vector<double>& v, std::vector<double>& out) {
      for (size_t r = 0; r < R; ++r) {
        double m = 0, var = 0;
        for (int o = 0; o < O; ++o) m += (float)(half)v[r * O + o];
        m /= O;
        for (int o = 0; o < O; ++o) { double d = (float)(half)v[r * O + o] - m; var += d * d; }
        double inv = 1 / sqrt(var / O + 1e-5);
        for (int o = 0; o < O; ++o) out[r * O + o] = ((float)(half)v[r * O + o] - m) * inv * s1[o] + o1[o];
      }
    };
    for (size_t r = 0; r < R; ++r) for (int o = 0; o < O; ++o) {
      double a = 0, b = 0;
      for (int k = 0; k < K; ++k) {
        a += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * O + o];
        b += (double)(float)(half)Xv[r * K + k] * (float)(half)Wv[(size_t)k * O + o];
      }
      y[r * O + o] = (float)(half)Y0[r * O + o] + a;
      y2[r * O + o] = (float)(half)Y0[r * O + o] + (double)(float)(half)b / (1 + exp(-a));
    }
    norm(y, ref); norm(y2, ref2);
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(W).data(), W.size());
    half* dXv = uploadNew(toH(Xv).data(), Xv.size()); half* dWv = uploadNew(toH(Wv).data(), Wv.size());
    half* dY = uploadNew(toH(Y0).data(), Y0.size()); half* dN = allocT<half>(R * O); half* tmp = allocT<half>(R * O);
    float *ds1 = uploadNew(s1.data(), O), *do1 = uploadNew(o1.data(), O);
    auto f32 = [](const std::vector<half>& h) { return std::vector<float>(h.begin(), h.end()); };
    Gemm g{}; g.X = dX; g.tx = F16; g.W = dW; g.tw = F16; g.Y = dY; g.ty = F16; g.rows = R; g.in = K; g.out = O; g.beta = 1.f;
    g.lnOut = dN; g.lnScale = ds1; g.lnOffset = do1;
    bool em1 = gemm(g);
    report("half pair, residual: the output", relRms(y, f32(download(dY, R * O))), 1e-3);
    if (em1) report("half pair, emitted LayerNorm: the norm", relRms(ref, f32(download(dN, R * O))), 3e-3);
    upload(dY, toH(Y0).data(), R * O * 2);
    bool em2 = gemmGatedAddDual(dX, dW, dXv, dWv, dY, R, K, O, nullptr, nullptr, tmp, nullptr, dN, ds1, do1);
    report("half pair, the triangle's tail: the pair", relRms(y2, f32(download(dY, R * O))), 2e-3);
    if (em2) report("half pair, the triangle's tail: the norm", relRms(ref2, f32(download(dN, R * O))), 3e-3);
  }
  if (halfProductScale()) {   // the triangle's product in half, scaled (core.mm's halfProductScale): the contraction's
      // scaled saturating store, then the centre LayerNorm from it, against the norm of the exact product
    const float hs = halfProductScale();
    int n = 37, np = 40, K = 256; size_t plane = (size_t)np * np;
    for (int C : {128, 256, 384, 512, 100}) {
      auto A = randv((size_t)C * plane, 30.f), B = randv((size_t)C * plane, 30.f), sc = randv(C), of = randv(C);
      for (int c = 0; c < C; ++c) for (int i = n; i < np; ++i) for (int k = 0; k < np; ++k) A[(size_t)c * plane + (size_t)i * np + k] = B[(size_t)c * plane + (size_t)i * np + k] = 0;
      std::vector<double> prod((size_t)C * plane, 0), ref((size_t)n * n * C);
      for (int c = 0; c < C; ++c) for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        double t = 0;
        for (int k = 0; k < np && k < K; ++k) t += (double)(float)(half)A[(size_t)c * plane + (size_t)i * np + k] * (float)(half)B[(size_t)c * plane + (size_t)j * np + k];
        prod[(size_t)c * plane + (size_t)i * np + j] = t;
      }
      for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        double m = 0, var = 0;
        for (int c = 0; c < C; ++c) m += prod[(size_t)c * plane + (size_t)i * np + j];
        m /= C;
        for (int c = 0; c < C; ++c) { double d = prod[(size_t)c * plane + (size_t)i * np + j] - m; var += d * d; }
        const double inv = 1 / sqrt(var / C + 1e-5);
        for (int c = 0; c < C; ++c) ref[((size_t)i * n + j) * C + c] = (prod[(size_t)c * plane + (size_t)i * np + j] - m) * inv * sc[c] + of[c];
      }
      half* dA = uploadNew(toH(A).data(), A.size()); half* dB = uploadNew(toH(B).data(), B.size());
      half* dP = allocT<half>((size_t)C * plane); half* dO = allocT<half>((size_t)n * n * C);
      float *dsc = uploadNew(sc.data(), C), *dof = uploadNew(of.data(), C);
      Gemm g{}; g.tx = F16; g.tw = F16; g.ty = F16; g.rows = np; g.in = np; g.out = np; g.ldx = g.ldw = g.ldy = np;
      g.sx = g.sw = g.sy = (int64_t)plane; g.batch = C; g.Y = dP; g.alpha = hs; g.accFloat = true; g.X = dA; g.W = dB; g.transW = true;
      gemm(g);
      std::vector<double> pref((size_t)C * plane); for (size_t i = 0; i < pref.size(); ++i) pref[i] = prod[i] * hs;
      char l1[96], l2[96];
      snprintf(l1, sizeof l1, "half product (x %g) %d channels: the contraction", hs, C);
      auto ph = download(dP, (size_t)C * plane);
      report(l1, relRms(pref, std::vector<float>(ph.begin(), ph.end())), 1e-3);
      centerNorm(dP, hs, dO, n, np, C, dsc, dof);
      auto oh = download(dO, (size_t)n * n * C);
      snprintf(l2, sizeof l2, "half product %d channels: the centre LayerNorm", C);
      report(l2, relRms(ref, std::vector<float>(oh.begin(), oh.end())), 3e-3);
      release(dA); release(dB); release(dP); release(dO); release(dsc); release(dof);
    }
  } else printf("  half product: not taken here (no matrix units)\n");
  {   // the triangle's gate: n x n pairs, C channels, planes padded to np
    int n = 37, np = 40, C = 64; size_t P = (size_t)n * n, plane = (size_t)np * np;
    auto X = randv(P * C), proj = randv((size_t)C * 2 * C, 0.1f), gate = randv((size_t)C * 2 * C, 0.1f), mask = randv(P);
    std::vector<float> W4((size_t)C * 4 * C);      // channel c's (pa ga pb gb) in blocks of 8
    for (int r = 0; r < C; ++r) for (int col = 0; col < 4 * C; ++col) {
      int kind = (col % 32) / 8, c = 8 * (col / 32) + col % 8, k = kind == 1 ? 2 : kind == 2 ? 1 : kind;
      W4[(size_t)r * 4 * C + col] = k < 2 ? proj[(size_t)r * 2 * C + 2 * c + k] : gate[(size_t)r * 2 * C + 2 * c + k - 2];
    }
    std::vector<double> refA(plane * C, 0), refB(plane * C, 0);
    for (size_t p = 0; p < P; ++p) for (int c = 0; c < C; ++c) {
      double pa = 0, pb = 0, ga = 0, gb = 0;
      for (int k = 0; k < C; ++k) {
        double x = (float)(half)X[p * C + k];
        pa += x * (float)(half)proj[(size_t)k * 2 * C + 2 * c]; pb += x * (float)(half)proj[(size_t)k * 2 * C + 2 * c + 1];
        ga += x * (float)(half)gate[(size_t)k * 2 * C + 2 * c]; gb += x * (float)(half)gate[(size_t)k * 2 * C + 2 * c + 1];
      }
      size_t q = (p / n) * np + p % n;
      refA[(size_t)c * plane + q] = pa * mask[p] / (1 + exp(-ga)); refB[(size_t)c * plane + q] = pb * mask[p] / (1 + exp(-gb));
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(W4).data(), W4.size()); float* dm = uploadNew(mask.data(), P);
    half* dA = allocT<half>(plane * C); half* dB = allocT<half>(plane * C);
    gemmTriGate(dX, dW, dm, dA, dB, 0, P, C, plane, n, np);
    auto a = download(dA, plane * C), b = download(dB, plane * C);
    report("triangle gate a", relRms(refA, std::vector<float>(a.begin(), a.end())), 5e-3);
    report("triangle gate b", relRms(refB, std::vector<float>(b.begin(), b.end())), 5e-3);
  }  {   // the triangle's gate with a bias (AF2's), C 128: two 128-column tiles
    int n = 37, np = 40, C = 128; size_t P = (size_t)n * n, plane = (size_t)np * np;
    auto X = randv(P * C), proj = randv((size_t)C * 2 * C, 0.1f), gate = randv((size_t)C * 2 * C, 0.1f), mask = randv(P), bias4 = randv((size_t)4 * C);
    std::vector<float> W4((size_t)C * 4 * C);      // channel c's (pa ga pb gb) in blocks of 8
    for (int r = 0; r < C; ++r) for (int col = 0; col < 4 * C; ++col) {
      int kind = (col % 32) / 8, c = 8 * (col / 32) + col % 8, k = kind == 1 ? 2 : kind == 2 ? 1 : kind;
      W4[(size_t)r * 4 * C + col] = k < 2 ? proj[(size_t)r * 2 * C + 2 * c + k] : gate[(size_t)r * 2 * C + 2 * c + k - 2];
    }
    std::vector<double> refA(plane * C, 0), refB(plane * C, 0);
    for (size_t p = 0; p < P; ++p) for (int c = 0; c < C; ++c) {
      double pa = 0, pb = 0, ga = 0, gb = 0;
      { int base = (c / 8) * 32 + c % 8; pa = bias4[base]; ga = bias4[base + 8]; pb = bias4[base + 16]; gb = bias4[base + 24]; }
      for (int k = 0; k < C; ++k) {
        double x = (float)(half)X[p * C + k];
        pa += x * (float)(half)proj[(size_t)k * 2 * C + 2 * c]; pb += x * (float)(half)proj[(size_t)k * 2 * C + 2 * c + 1];
        ga += x * (float)(half)gate[(size_t)k * 2 * C + 2 * c]; gb += x * (float)(half)gate[(size_t)k * 2 * C + 2 * c + 1];
      }
      size_t q = (p / n) * np + p % n;
      refA[(size_t)c * plane + q] = pa * mask[p] / (1 + exp(-ga)); refB[(size_t)c * plane + q] = pb * mask[p] / (1 + exp(-gb));
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(W4).data(), W4.size()); float* dm = uploadNew(mask.data(), P);
    half* dA = allocT<half>(plane * C); half* dB = allocT<half>(plane * C);
    float* dbias = uploadNew(bias4.data(), bias4.size());
    gemmTriGate(dX, dW, dm, dA, dB, 0, P, C, plane, n, np, dbias);
    auto a = download(dA, plane * C), b = download(dB, plane * C);
    report("triangle gate a, bias", relRms(refA, std::vector<float>(a.begin(), a.end())), 5e-3);
    report("triangle gate b, bias", relRms(refB, std::vector<float>(b.begin(), b.end())), 5e-3);
  }
  {   // gated rows (BioEmu2 adaLN residual): Y[r][o] += gate[map ? map[r] : r][o] * (alpha X W)[r][o]
    size_t R = 213, G = 48; int K = 128, O = 256, ldg = 512;
    auto X = randv(R * K), W = randv((size_t)K * O, 0.1f), gate = randv(G * ldg), Y0 = randv(R * O);
    std::vector<int> mp(R);
    for (size_t r = 0; r < R; ++r) mp[r] = (int)((r * 13 + 5) % G);
    std::vector<double> ref(R * O);
    for (size_t r = 0; r < R; ++r) for (int o = 0; o < O; ++o) {
      double s = 0; for (int k = 0; k < K; ++k) s += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * O + o];
      ref[r * O + o] = Y0[r * O + o] + (double)gate[(size_t)mp[r] * ldg + o] * (0.75 * s);
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(W).data(), W.size());
    float* dg = uploadNew(gate.data(), gate.size()); float* dY = uploadNew(Y0.data(), Y0.size());
    int* dm = uploadNew(mp.data(), mp.size());
    Gemm g{}; g.X = dX; g.tx = F16; g.W = dW; g.tw = F16; g.Y = dY; g.ty = F32; g.rows = R; g.in = K; g.out = O; g.alpha = 0.75f;
    if (gemmGatedRows(g, dg, ldg, dm)) report("gated rows (mapped)", relRms(ref, download(dY, R * O)), 1e-3);
    else { ++failures; printf("  gated rows (mapped): FAIL (not taken)\n"); }
    release(dX); release(dW); release(dg); release(dY); release(dm);
  }
  for (bool pre : {false, true}) {   // gated residual (AF3 diffusion transformer): act[r][o] += (alpha X W)[r][o] * (pre ? g : sigmoid(g))
    size_t rows = 195, prows = 208, period = 65; int K = 128, O = 256, ld = 512;
    auto X = randv(prows * K), W = randv((size_t)K * O, 0.1f), gate = randv(period * ld), A0 = randv(prows * O);
    std::vector<double> ref(A0.begin(), A0.end());
    for (size_t r = 0; r < rows; ++r) for (int o = 0; o < O; ++o) {
      double s = 0; for (int k = 0; k < K; ++k) s += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * O + o];
      double gp = gate[(r % period) * ld + o];
      double gm = pre ? gp : 1.0 / (1.0 + exp(-gp));
      ref[r * O + o] += (0.5 * s) * gm;
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(W).data(), W.size());
    float* dg = uploadNew(gate.data(), gate.size()); float* dA = uploadNew(A0.data(), A0.size());
    if (gemmGateResidual(dX, dW, dA, prows, rows, K, O, dg, ld, period, 0.5f, "gate res", pre))
      report(pre ? "gate residual (pre-sigmoid)" : "gate residual (sigmoid)", relRms(ref, download(dA, prows * O)), 1e-3);
    else { ++failures; printf("  gate residual: FAIL (not taken)\n"); }
    release(dX); release(dW); release(dg); release(dA);
  }
  for (bool chained : {false, true}) {   // gated residual + emitted periodic adaLN (128 wide, unchained & chained, + rowMask & out3)
    size_t rows = 195, period = 65; int K = 128, O = 128, ld = 256;
    auto X = randv(rows * K), W = randv((size_t)K * O, 0.1f), gate = randv(period * ld), A0 = randv(rows * O);
    auto s6 = randv(6 * period * O), rmask = randv(period);
    std::vector<double> ref(A0.begin(), A0.end()), refLn1(rows * O), refLn2(rows * O), refLn3(rows * O), refMasked(A0.begin(), A0.end());
    for (size_t r = 0; r < rows; ++r) {
      double m = 0, var = 0;
      double rm = rmask[r % period];
      for (int o = 0; o < O; ++o) {
        double s = 0; for (int k = 0; k < K; ++k) s += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * O + o];
        double val = (ref[r * O + o] + (0.5 * s) * gate[(r % period) * ld + o]) * rm;
        ref[r * O + o] = val;
        refMasked[r * O + o] = (double)(float)(half)val;
        m += val;
      }
      m /= O;
      for (int o = 0; o < O; ++o) { double d = ref[r * O + o] - m; var += d * d; }
      double inv = 1.0 / sqrt(var / O + 1e-5);
      size_t pr = (r % period) * O, span = period * O;
      double m2 = 0, var2 = 0;
      for (int o = 0; o < O; ++o) {
        double nrm = (ref[r * O + o] - m) * inv;
        double y1 = nrm * s6[pr + o] + s6[span + pr + o];
        refLn1[r * O + o] = y1;
        refLn3[r * O + o] = nrm * s6[4 * span + pr + o] + s6[5 * span + pr + o];
        if (!chained) refLn2[r * O + o] = nrm * s6[2 * span + pr + o] + s6[3 * span + pr + o];
        else m2 += y1;
      }
      if (chained) {
        m2 /= O;
        for (int o = 0; o < O; ++o) { double d = refLn1[r * O + o] - m2; var2 += d * d; }
        double inv2 = 1.0 / sqrt(var2 / O + 1e-5);
        for (int o = 0; o < O; ++o) refLn2[r * O + o] = (refLn1[r * O + o] - m2) * inv2 * s6[2 * span + pr + o] + s6[3 * span + pr + o];
      }
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(W).data(), W.size());
    float* dg = uploadNew(gate.data(), gate.size()); float* dA = uploadNew(A0.data(), A0.size());
    float *ds6 = uploadNew(s6.data(), s6.size()), *drm = uploadNew(rmask.data(), rmask.size());
    size_t span = period * O;
    half *dLn1 = allocT<half>(rows * O), *dLn2 = allocT<half>(rows * O), *dLn3 = allocT<half>(rows * O);
    bool emitted = false;
    gemmGateResidual(dX, dW, dA, rows, rows, K, O, dg, ld, period, 0.5f, "gate res ln", true,
                     dLn1, ds6, ds6 + span, 1e-5f, dLn2, ds6 + 2 * span, ds6 + 3 * span, chained, &emitted, drm, dLn3);
    if (emitted) {
      report(chained ? "gate residual + adaLN (chained): act" : "gate residual + adaLN: act", relRms(ref, download(dA, rows * O)), 1e-3);
      auto h1 = download(dLn1, rows * O), h2 = download(dLn2, rows * O), h3 = download(dLn3, rows * O);
      report(chained ? "gate residual + adaLN (chained): ln1" : "gate residual + adaLN: ln1", relRms(refLn1, std::vector<float>(h1.begin(), h1.end())), 3e-3);
      report(chained ? "gate residual + adaLN (chained): ln2" : "gate residual + adaLN: ln2", relRms(refLn2, std::vector<float>(h2.begin(), h2.end())), 3e-3);
      report(chained ? "gate residual + adaLN (chained): ln3" : "gate residual + adaLN: ln3", relRms(refLn3, std::vector<float>(h3.begin(), h3.end())), 3e-3);
    }
    if (!chained) {
      upload(dA, A0.data(), A0.size() * 4);
      gemmGateResidual(dX, dW, dA, rows, rows, K, O, dg, ld, period, 0.5f, "gate res mask+h", true,
                       nullptr, nullptr, nullptr, 1e-5f, nullptr, nullptr, nullptr, false, nullptr, drm, dLn3);
      report("gate residual + rowMask: act", relRms(ref, download(dA, rows * O)), 1e-3);
      auto hm = download(dLn3, rows * O);
      report("gate residual + rowMask: actH", relRms(refMasked, std::vector<float>(hm.begin(), hm.end())), 3e-3);
    }
    release(dX); release(dW); release(dg); release(dA); release(ds6); release(drm); release(dLn1); release(dLn2); release(dLn3);
  }
  {   // conditioning permuted store (AF3 diffusion conditioning)
    size_t valid = 135, prows = 144; int period = 45, K = 128, W = 128, BS = 3, O = BS * W, sigCols = 64;
    auto X = randv(prows * K), Wgt = randv((size_t)K * O, 0.1f), bias = randv(O);
    std::vector<double> ref(valid * O, 0);
    for (size_t j = 0; j < valid; ++j) for (int i = 0; i < O; ++i) {
      double s = bias[i];
      for (int k = 0; k < K; ++k) s += (double)(float)(half)X[j * K + k] * (float)(half)Wgt[(size_t)k * O + i];
      size_t kk = j / period, r = j % period, bs = i / W, col = i % W;
      if ((int)col < sigCols) s = 1.0 / (1.0 + exp(-s));
      ref[((kk * BS + bs) * period + r) * W + col] = s;
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(Wgt).data(), Wgt.size());
    float* db = uploadNew(bias.data(), bias.size()); float* dY = allocT<float>(valid * O);
    if (gemmCondPermuted(dX, dW, dY, prows, valid, K, O, db, period, W, sigCols))
      report("conditioning permuted store", relRms(ref, download(dY, valid * O)), 1e-3);
    else { ++failures; printf("  conditioning permuted store: FAIL (not taken)\n"); }
    release(dX); release(dW); release(db); release(dY);
  }
  {   // in-place half multiply: Y[r][o] *= (X W)[r][o]
    size_t R = 195; int K = 128, O = 128;
    auto X = randv(R * K), W = randv((size_t)K * O, 0.1f), Y0 = randv(R * O);
    std::vector<double> ref(R * O);
    for (size_t r = 0; r < R; ++r) for (int o = 0; o < O; ++o) {
      double s = 0; for (int k = 0; k < K; ++k) s += (double)(float)(half)X[r * K + k] * (float)(half)W[(size_t)k * O + o];
      ref[r * O + o] = (double)(float)(half)Y0[r * O + o] * s;
    }
    half* dX = uploadNew(toH(X).data(), X.size()); half* dW = uploadNew(toH(W).data(), W.size());
    half* dY = uploadNew(toH(Y0).data(), Y0.size());
    if (gemmMulH(dX, dW, dY, R, K, O)) {
      auto h = download(dY, R * O);
      report("in-place half multiply (gemmMulH)", relRms(ref, std::vector<float>(h.begin(), h.end())), 3e-3);
    } else { ++failures; printf("  gemmMulH: FAIL (not taken)\n"); }
    release(dX); release(dW); release(dY);
  }
  {   // outer product permuted + normalised output add (float and half pair)
    int bi = 12, L = 19, O = 16, S = 32, C = 128;
    int ldl = L * O, ldr = L * O;
    auto lt = randv((size_t)S * ldl, 0.3f), rt = randv((size_t)S * ldr, 0.3f);
    std::vector<double> refP((size_t)bi * L * O * O);
    for (int i = 0; i < bi; ++i) for (int j = 0; j < L; ++j) for (int c = 0; c < O; ++c) for (int e = 0; e < O; ++e) {
      double s = 0;
      for (int k = 0; k < S; ++k) s += (double)(float)(half)lt[(size_t)k * ldl + i * O + c] * (float)(half)rt[(size_t)k * ldr + j * O + e];
      refP[(((size_t)i * L + j) * O + c) * O + e] = 0.5 * s;
    }
    half* dlt = uploadNew(toH(lt).data(), lt.size()); half* drt = uploadNew(toH(rt).data(), rt.size());
    half* dX = allocT<half>((size_t)bi * L * O * O);
    if (gemmOpmPermuted(dlt, ldl, drt, ldr, dX, bi, L, O, S, 0.5f)) {
      auto h = download(dX, (size_t)bi * L * O * O);
      report("outer product permuted (gemmOpmPermuted)", relRms(refP, std::vector<float>(h.begin(), h.end())), 2e-3);
    } else { ++failures; printf("  gemmOpmPermuted: FAIL (not taken)\n"); }
    size_t rows = (size_t)bi * L; int in = O * O;
    auto W = randv((size_t)in * C, 0.1f), bias = randv(128), norm = randv(rows, 0.5f), P0 = randv(rows * C);
    for (auto& v : norm) v = std::abs(v) + 1.f;
    std::vector<double> refF(rows * C), refH(rows * C), refAfter(rows * C);
    auto xh = download(dX, rows * in);
    for (size_t r = 0; r < rows; ++r) for (int o = 0; o < C; ++o) {
      double s = 0;
      for (int k = 0; k < in; ++k) s += (double)(float)xh[r * in + k] * (float)(half)W[(size_t)k * C + o];
      double add = (bias[o & 127] + 2.0 * s) / (1e-3 + norm[r]);
      double addAfter = (2.0 * s) / std::max((double)norm[r], 1.0) + bias[o & 127];
      refF[r * C + o] = P0[r * C + o] + add;
      refH[r * C + o] = (double)(float)(half)P0[r * C + o] + add;
      refAfter[r * C + o] = P0[r * C + o] + addAfter;
    }
    half* dW = uploadNew(toH(W).data(), W.size()); float* db = uploadNew(bias.data(), bias.size());
    float* dn = uploadNew(norm.data(), norm.size());
    float* dPf = uploadNew(P0.data(), P0.size()); half* dPh = uploadNew(toH(P0).data(), P0.size());
    if (gemmOpmOut(dX, dW, dPf, db, dn, rows, in, C, 2.f))
      report("outer product output add (float pair)", relRms(refF, download(dPf, rows * C)), 1e-3);
    else { ++failures; printf("  gemmOpmOut float: FAIL (not taken)\n"); }
    upload(dPf, P0.data(), P0.size() * 4);
    if (gemmOpmOut(dX, dW, dPf, db, dn, rows, in, C, 2.f, nullptr, nullptr, nullptr, 1e-5f, nullptr, nullptr, nullptr, true))
      report("outer product output add, bias after norm", relRms(refAfter, download(dPf, rows * C)), 1e-3);
    else { ++failures; printf("  gemmOpmOut after: FAIL (not taken)\n"); }
    if (gemmOpmOut(dX, dW, dPh, db, dn, rows, in, C, 2.f)) {
      auto h = download(dPh, rows * C);
      report("outer product output add (half pair)", relRms(refH, std::vector<float>(h.begin(), h.end())), 3e-3);
    } else { ++failures; printf("  gemmOpmOut half: FAIL (not taken)\n"); }
    auto sc = randv(C), of = randv(C), sc2 = randv(C), of2 = randv(C);
    std::vector<double> refLn1(rows * C), refLn2(rows * C);
    for (size_t r = 0; r < rows; ++r) {
      double m = 0, var = 0;
      for (int o = 0; o < C; ++o) m += (float)(half)refH[r * C + o];
      m /= C;
      for (int o = 0; o < C; ++o) { double d = (float)(half)refH[r * C + o] - m; var += d * d; }
      double inv = 1 / sqrt(var / C + 1e-5);
      for (int o = 0; o < C; ++o) {
        refLn1[r * C + o] = ((float)(half)refH[r * C + o] - m) * inv * sc[o] + of[o];
        refLn2[r * C + o] = ((float)(half)refH[r * C + o] - m) * inv * sc2[o] + of2[o];
      }
    }
    upload(dPh, toH(P0).data(), P0.size() * 2);
    half* dLn1 = allocT<half>(rows * C); half* dLn2 = allocT<half>(rows * C);
    float *dsc = uploadNew(sc.data(), C), *dof = uploadNew(of.data(), C);
    float *dsc2 = uploadNew(sc2.data(), C), *dof2 = uploadNew(of2.data(), C);
    if (gemmOpmOut(dX, dW, dPh, db, dn, rows, in, C, 2.f, dLn1, dsc, dof, 1e-5f, dLn2, dsc2, dof2)) {
      auto h1 = download(dLn1, rows * C), h2 = download(dLn2, rows * C);
      report("outer product output add, emitted LN 1", relRms(refLn1, std::vector<float>(h1.begin(), h1.end())), 3e-3);
      report("outer product output add, emitted LN 2", relRms(refLn2, std::vector<float>(h2.begin(), h2.end())), 3e-3);
    } else { ++failures; printf("  gemmOpmOut emitted LN: FAIL (not taken)\n"); }
    release(dLn1); release(dLn2); release(dsc); release(dof); release(dsc2); release(dof2);
    release(dlt); release(drt); release(dX); release(dW); release(db); release(dn); release(dPf); release(dPh);
    for (int opmC : {256, 384}) {   // shallow OPM output add at C = 256 (with emitted LN) and C = 384
      int bi2 = 5, L2 = 7, in2 = 64, out2 = L2 * opmC;
      auto X2 = randv((size_t)bi2 * in2), W2 = randv((size_t)in2 * out2, 0.1f), b2 = randv(opmC), n2 = randv((size_t)bi2 * L2, 0.5f), P2 = randv((size_t)bi2 * out2);
      auto sc1 = randv(opmC), of1 = randv(opmC), sc2 = randv(opmC), of2 = randv(opmC);
      for (auto& v : n2) v = std::abs(v) + 1.f;
      std::vector<double> ref2((size_t)bi2 * out2), ref2Ln1((size_t)bi2 * out2), ref2Ln2((size_t)bi2 * out2);
      for (int i = 0; i < bi2; ++i) for (int j = 0; j < L2; ++j) {
        double m = 0, var = 0;
        for (int c = 0; c < opmC; ++c) {
          int o = j * opmC + c;
          double s = 0;
          for (int k = 0; k < in2; ++k) s += (double)(float)(half)X2[i * in2 + k] * (float)(half)W2[(size_t)k * out2 + o];
          double val = P2[i * out2 + o] + (b2[c] + s) / (1e-3 + n2[i * L2 + j]);
          ref2[i * out2 + o] = val;
          m += val;
        }
        m /= opmC;
        for (int c = 0; c < opmC; ++c) { double d = ref2[i * out2 + j * opmC + c] - m; var += d * d; }
        double inv = 1.0 / sqrt(var / opmC + 1e-5);
        for (int c = 0; c < opmC; ++c) {
          int o = j * opmC + c;
          ref2Ln1[i * out2 + o] = (ref2[i * out2 + o] - m) * inv * sc1[c] + of1[c];
          ref2Ln2[i * out2 + o] = (ref2[i * out2 + o] - m) * inv * sc2[c] + of2[c];
        }
      }
      half* dX2 = uploadNew(toH(X2).data(), X2.size()); half* dW2 = uploadNew(toH(W2).data(), W2.size());
      float *db2 = uploadNew(b2.data(), b2.size()), *dn2 = uploadNew(n2.data(), n2.size()), *dP2 = uploadNew(P2.data(), P2.size());
      float *dsc1 = uploadNew(sc1.data(), opmC), *dof1 = uploadNew(of1.data(), opmC), *dsc2 = uploadNew(sc2.data(), opmC), *dof2 = uploadNew(of2.data(), opmC);
      half *dL1 = allocT<half>((size_t)bi2 * out2), *dL2 = allocT<half>((size_t)bi2 * out2);
      const bool wantLn = opmC == 256 && matrixUnits();
      if (gemmOpmOut(dX2, dW2, dP2, db2, dn2, bi2, in2, out2, 1.f, wantLn ? dL1 : nullptr, dsc1, dof1, 1e-5f, wantLn ? dL2 : nullptr, dsc2, dof2, false, opmC)) {
        char l0[64], l1[64], l2[64];
        snprintf(l0, sizeof l0, "outer product output add (C %d)", opmC);
        report(l0, relRms(ref2, download(dP2, (size_t)bi2 * out2)), 1e-3);
        if (wantLn) {
          auto h1 = download(dL1, (size_t)bi2 * out2), h2 = download(dL2, (size_t)bi2 * out2);
          snprintf(l1, sizeof l1, "outer product output add (C %d), emitted LN 1", opmC);
          snprintf(l2, sizeof l2, "outer product output add (C %d), emitted LN 2", opmC);
          report(l1, relRms(ref2Ln1, std::vector<float>(h1.begin(), h1.end())), 3e-3);
          report(l2, relRms(ref2Ln2, std::vector<float>(h2.begin(), h2.end())), 3e-3);
        }
      } else { ++failures; printf("  gemmOpmOut C %d: FAIL (not taken)\n", opmC); }
      release(dX2); release(dW2); release(db2); release(dn2); release(dP2); release(dsc1); release(dof1); release(dsc2); release(dof2); release(dL1); release(dL2);
    }
  }

  printf("attention\n");
  for (auto cfg : std::vector<std::tuple<int, int, int, int, bool, bool, bool>>{
         {3, 37, 4, 32, true, false, false}, {2, 70, 8, 8, false, true, false}, {2, 130, 2, 16, true, true, false},
         {3, 37, 4, 32, true, true, true}, {1, 261, 16, 48, true, false, false}, {2, 64, 4, 64, false, false, false},
         {2, 100, 4, 32, true, true, true}}) {
    int B = std::get<0>(cfg), n = std::get<1>(cfg), H = std::get<2>(cfg), D = std::get<3>(cfg);
    bool withBias = std::get<4>(cfg), withMask = std::get<5>(cfg), strided = std::get<6>(cfg);
    int W = H * D;
    // qkvg [B][n][4W] dense, or strided: [n][B][4W] (an attention across the leading axis)
    auto qkvg = randv((size_t)B * n * 4 * W, 0.7f);
    auto bias = randv((size_t)H * n * n, 2.f);
    std::vector<float> mask((size_t)B * n, 1.f);
    if (withMask) for (size_t i = 0; i < mask.size(); ++i) mask[i] = (i * 7 + 3) % 5 ? 1.f : 0.f;
    std::vector<float> qb = randv(W, 0.3f);
    float scale = 1.f / sqrtf((float)D);
    auto at = [&](int b, int p, int role, int c) -> float {
      size_t row = strided ? (size_t)p * B + b : (size_t)b * n + p;
      return (float)(half)qkvg[row * 4 * W + role * W + c];
    };
    std::vector<double> ref((size_t)B * n * W);
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) for (int q = 0; q < n; ++q) {
      std::vector<double> sc(n); double mx = -1e300;
      for (int k = 0; k < n; ++k) {
        double dot = 0;
        for (int d = 0; d < D; ++d) dot += ((double)at(b, q, 0, h * D + d) + qb[h * D + d]) * at(b, k, 1, h * D + d);
        double v = dot * scale + (withBias ? (double)(float)(half)(bias[((size_t)h * n + q) * n + k] * 1.4426950408889634f) / 1.4426950408889634 : 0.0);
        if (withMask && mask[(size_t)b * n + k] == 0.f) v -= 1e9 / 1.4426950408889634;
        sc[k] = v; mx = std::max(mx, v);
      }
      double sum = 0; for (auto& v : sc) { v = exp(v - mx); sum += v; }
      for (int d = 0; d < D; ++d) {
        double o = 0; for (int k = 0; k < n; ++k) o += sc[k] * at(b, k, 2, h * D + d);
        size_t row = strided ? (size_t)q * B + b : (size_t)b * n + q;
        ref[row * W + h * D + d] = o / sum / (1 + exp(-(double)at(b, q, 3, h * D + d)));
      }
    }
    std::vector<half> bh((size_t)H * n * n);
    for (size_t i = 0; i < bh.size(); ++i) bh[i] = (half)(bias[i] * 1.4426950408889634f);
    half* dq = uploadNew(toH(qkvg).data(), qkvg.size()); half* db = uploadNew(bh.data(), bh.size());
    float* dm = uploadNew(mask.data(), mask.size()); float* dqb = uploadNew(qb.data(), qb.size());
    half* dout = allocT<half>((size_t)B * n * W);
    Attention A{}; A.qkvg = dq; A.out = dout; A.n = n; A.heads = H; A.D = D; A.rows = B; A.scale = scale; A.qBias = dqb;
    if (withBias) { A.bias = db; A.biasStride = n; }
    if (withMask) A.mask = dm;
    if (strided) { A.rowStride = 4 * W; A.posStride = (int64_t)B * 4 * W; A.outRowStride = W; A.outPosStride = (int64_t)B * W; }
    attention(A);
    auto h = download(dout, (size_t)B * n * W);
    char l[96]; snprintf(l, sizeof l, "B %d n %d H %d D %d%s%s%s", B, n, H, D, withBias ? " bias" : "", withMask ? " mask" : "", strided ? " strided" : "");
    report(l, relRms(ref, std::vector<float>(h.begin(), h.end())), 5e-3);
    if (withBias && !withMask) {
      int bs = (n + 63) / 64 * 64;
      std::vector<half> bhp((size_t)H * n * bs, (half)-INFINITY);
      for (int h0 = 0; h0 < H; ++h0) for (int q = 0; q < n; ++q) for (int k = 0; k < n; ++k)
        bhp[((size_t)h0 * n + q) * bs + k] = bh[((size_t)h0 * n + q) * n + k];
      half* dbp = uploadNew(bhp.data(), bhp.size());
      Attention AN = A; AN.bias = dbp; AN.biasStride = bs; AN.biasPadInf = true;
      attention(AN);
      auto hn = download(dout, (size_t)B * n * W);
      snprintf(l, sizeof l, "B %d n %d H %d D %d biasPadInf (qBias)", B, n, H, D);
      report(l, relRms(ref, std::vector<float>(hn.begin(), hn.end())), 5e-3);
      release(dbp);
    }
  }

  printf("LayerNorm\n");
  for (int C : {256, 1152, 389, 128, 1536}) {
    size_t R = 77;
    auto X = randv(R * C, 3.f), s = randv(C), o = randv(C);
    std::vector<double> ref(R * C);
    for (size_t r = 0; r < R; ++r) {
      double m = 0, v = 0;
      for (int c = 0; c < C; ++c) m += X[r * C + c];
      m /= C;
      for (int c = 0; c < C; ++c) v += (X[r * C + c] - m) * (X[r * C + c] - m);
      double inv = 1 / sqrt(v / C + 1e-5);
      for (int c = 0; c < C; ++c) ref[r * C + c] = (X[r * C + c] - m) * inv * s[c] + o[c];
    }
    float* dX = uploadNew(X.data(), X.size()); float* ds = uploadNew(s.data(), C); float* dO = uploadNew(o.data(), C);
    float* dY = allocT<float>(R * C); half* dH = allocT<half>(R * C);
    layerNorm(dX, dY, R, C, ds, dO);
    layerNorm(dX, dH, R, C, ds, dO);
    char l[64]; snprintf(l, sizeof l, "C %d -> f32", C);
    report(l, relRms(ref, download(dY, R * C)), 1e-6);
    auto h = download(dH, R * C);
    snprintf(l, sizeof l, "C %d -> f16", C);
    report(l, relRms(ref, std::vector<float>(h.begin(), h.end())), 1e-3);
  }
  printStats();
  printf("%s\n", failures ? "FAILED" : "all passed");
  return failures ? 1 : 0;
}
