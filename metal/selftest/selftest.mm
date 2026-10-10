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
  {   // SwiGLU in blocks of 8
    size_t R = 333; int K = 128, Hd = 256;
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

  printf("attention\n");
  for (auto cfg : std::vector<std::tuple<int, int, int, int, bool, bool, bool>>{
         {3, 37, 4, 32, true, false, false}, {2, 70, 8, 8, false, true, false}, {2, 130, 2, 16, true, true, false},
         {3, 37, 4, 32, true, true, true}, {1, 261, 16, 48, true, false, false}, {2, 64, 4, 64, false, false, false}}) {
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
