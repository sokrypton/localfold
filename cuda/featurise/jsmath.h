// JavaScript's Math, as V8 computes it - so a conformer built here is the page's to the last bit.
//
// 🔴 glibc IS NOT V8. V8's Math.sin/cos/asin/acos are fdlibm 5.3 (src/base/ieee754.cc) and differ from glibc's in
// the last place on 3-8% of arguments (measured on 200,000), and a distance-geometry embedding amplifies one ulp into
// a different conformer. Math.hypot is V8's own: normalise by the largest, Kahan-sum the squares, sqrt times the
// largest (builtins/math.tq). sqrt, abs, min, max, floor are exact everywhere. Built with -ffp-contract=off, since
// a fused multiply-add is a different rounding.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

namespace lf::jsmath {

inline int32_t hi(double x) { uint64_t b; std::memcpy(&b, &x, 8); return (int32_t)(b >> 32); }
inline uint32_t lo(double x) { uint64_t b; std::memcpy(&b, &x, 8); return (uint32_t)b; }
inline double words(uint32_t h, uint32_t l) { uint64_t b = ((uint64_t)h << 32) | l; double x; std::memcpy(&x, &b, 8); return x; }
inline double withLow(double x, uint32_t l) { return words((uint32_t)hi(x), l); }

inline double kernelSin(double x, double y, int iy) {
  const double half = 5.00000000000000000000e-01, S1 = words(0xBFC55555, 0x55555549), S2 = words(0x3F811111, 0x1110F8A6),
               S3 = words(0xBF2A01A0, 0x19C161D5), S4 = words(0x3EC71DE3, 0x57B1FE7D), S5 = words(0xBE5AE5E6, 0x8A2B9CEB),
               S6 = words(0x3DE5D93A, 0x5ACFD57C);
  int32_t ix = hi(x) & 0x7FFFFFFF;
  if (ix < 0x3E400000 && (int)x == 0) return x;
  double z = x * x, v = z * x, r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
  if (iy == 0) return x + v * (S1 + z * r);
  return x - ((z * (half * y - v * r) - y) - v * S1);
}

inline double kernelCos(double x, double y) {
  const double one = 1.0, C1 = words(0x3FA55555, 0x5555554C), C2 = words(0xBF56C16C, 0x16C15177), C3 = words(0x3EFA01A0, 0x19CB1590),
               C4 = words(0xBE927E4F, 0x809C52AD), C5 = words(0x3E21EE9E, 0xBDB4B1C4), C6 = words(0xBDA8FAE9, 0xBE8838D4);
  int32_t ix = hi(x) & 0x7FFFFFFF;
  if (ix < 0x3E400000 && (int)x == 0) return one;
  double z = x * x, r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
  if (ix < 0x3FD33333) return one - (0.5 * z - (z * r - x * y));
  double qx = ix > 0x3FE90000 ? 0.28125 : words((uint32_t)(ix - 0x00200000), 0);
  double hz = 0.5 * z - qx, a = one - qx;
  return a - (hz - (z * r - x * y));
}

// __ieee754_rem_pio2 for |x| up to 2^19 (pi/2) - a conformer's angles are nowhere near; beyond that it refuses
inline int remPio2(double x, double* y) {
  static const int32_t npio2_hw[] = {
    0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB, 0x402C463A, 0x402F6A7A,
    0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A, 0x403DD85A, 0x403F6A7A,
    0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C, 0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB, 0x4046C6CB, 0x40478FDB,
    0x404858EB, 0x404921FB};
  const double half = 0.5, invpio2 = words(0x3FE45F30, 0x6DC9C883), pio2_1 = words(0x3FF921FB, 0x54400000),
               pio2_1t = words(0x3DD0B461, 0x1A626331), pio2_2 = words(0x3DD0B461, 0x1A600000), pio2_2t = words(0x3BA3198A, 0x2E037073),
               pio2_3 = words(0x3BA3198A, 0x2E000000), pio2_3t = words(0x397B839A, 0x252049C1);
  int32_t hx = hi(x), ix = hx & 0x7FFFFFFF;
  if (ix <= 0x3FE921FB) { y[0] = x; y[1] = 0; return 0; }
  if (ix < 0x4002D97C) {
    if (hx > 0) {
      double z = x - pio2_1;
      if (ix != 0x3FF921FB) { y[0] = z - pio2_1t; y[1] = (z - y[0]) - pio2_1t; }
      else { z -= pio2_2; y[0] = z - pio2_2t; y[1] = (z - y[0]) - pio2_2t; }
      return 1;
    }
    double z = x + pio2_1;
    if (ix != 0x3FF921FB) { y[0] = z + pio2_1t; y[1] = (z - y[0]) + pio2_1t; }
    else { z += pio2_2; y[0] = z + pio2_2t; y[1] = (z - y[0]) + pio2_2t; }
    return -1;
  }
  if (ix <= 0x413921FB) {
    double t = std::fabs(x);
    int n = (int)(t * invpio2 + half);
    double fn = (double)n, r = t - fn * pio2_1, w = fn * pio2_1t;
    if (n < 32 && ix != npio2_hw[n - 1]) {
      y[0] = r - w;
    } else {
      int32_t j = ix >> 20;
      y[0] = r - w;
      int32_t i = j - ((hi(y[0]) >> 20) & 0x7FF);
      if (i > 16) {
        t = r; w = fn * pio2_2; r = t - w; w = fn * pio2_2t - ((t - r) - w); y[0] = r - w;
        i = j - ((hi(y[0]) >> 20) & 0x7FF);
        if (i > 49) { t = r; w = fn * pio2_3; r = t - w; w = fn * pio2_3t - ((t - r) - w); y[0] = r - w; }
      }
    }
    y[1] = (r - y[0]) - w;
    if (hx < 0) { y[0] = -y[0]; y[1] = -y[1]; return -n; }
    return n;
  }
  throw std::domain_error("jsmath: an angle past 2^19 pi/2, which no conformer reaches");
}

inline double sin(double x) {
  int32_t ix = hi(x) & 0x7FFFFFFF;
  if (ix <= 0x3FE921FB) return kernelSin(x, 0.0, 0);
  if (ix >= 0x7FF00000) return x - x;
  double y[2];
  switch (remPio2(x, y) & 3) {
    case 0: return kernelSin(y[0], y[1], 1);
    case 1: return kernelCos(y[0], y[1]);
    case 2: return -kernelSin(y[0], y[1], 1);
    default: return -kernelCos(y[0], y[1]);
  }
}

inline double cos(double x) {
  int32_t ix = hi(x) & 0x7FFFFFFF;
  if (ix <= 0x3FE921FB) return kernelCos(x, 0.0);
  if (ix >= 0x7FF00000) return x - x;
  double y[2];
  switch (remPio2(x, y) & 3) {
    case 0: return kernelCos(y[0], y[1]);
    case 1: return -kernelSin(y[0], y[1], 1);
    case 2: return -kernelCos(y[0], y[1]);
    default: return kernelSin(y[0], y[1], 1);
  }
}

namespace detail {
inline const double pS0 = words(0x3FC55555, 0x55555555), pS1 = words(0xBFD4D612, 0x03EB6F7D), pS2 = words(0x3FC9C155, 0x0E884455),
                    pS3 = words(0xBFA48228, 0xB5688F3B), pS4 = words(0x3F49EFE0, 0x7501B288), pS5 = words(0x3F023DE1, 0x0DFDF709),
                    qS1 = words(0xC0033A27, 0x1C8A2D4B), qS2 = words(0x40002AE5, 0x9C598AC8), qS3 = words(0xBFE6066C, 0x1B8D0159),
                    qS4 = words(0x3FB3B8C5, 0xB12E9282), pio2_hi = words(0x3FF921FB, 0x54442D18),
                    pio2_lo = words(0x3C91A626, 0x33145C07), pio4_hi = words(0x3FE921FB, 0x54442D18), pi = words(0x400921FB, 0x54442D18);
inline double P(double t) { return t * (pS0 + t * (pS1 + t * (pS2 + t * (pS3 + t * (pS4 + t * pS5))))); }
inline double Q(double t) { return 1.0 + t * (qS1 + t * (qS2 + t * (qS3 + t * qS4))); }
}  // namespace detail

inline double asin(double x) {
  using namespace detail;
  int32_t hx = hi(x), ix = hx & 0x7FFFFFFF;
  if (ix >= 0x3FF00000) {
    if (((ix - 0x3FF00000) | lo(x)) == 0) return x * pio2_hi + x * pio2_lo;
    return (x - x) / (x - x);
  }
  if (ix < 0x3FE00000) {
    double t = 0.0;
    if (ix < 0x3E400000) { if (1.0e300 + x > 1.0) return x; }
    else t = x * x;
    double w = P(t) / Q(t);
    return x + x * w;
  }
  double w = 1.0 - std::fabs(x), t = w * 0.5, p = P(t), q = Q(t), s = std::sqrt(t);
  if (ix >= 0x3FEF3333) {
    w = p / q;
    t = pio2_hi - (2.0 * (s + s * w) - pio2_lo);
  } else {
    w = withLow(s, 0);
    double c = (t - w * w) / (s + w), r = p / q;
    p = 2.0 * s * r - (pio2_lo - 2.0 * c);
    q = pio4_hi - 2.0 * w;
    t = pio4_hi - (p - q);
  }
  return hx > 0 ? t : -t;
}

inline double acos(double x) {
  using namespace detail;
  int32_t hx = hi(x), ix = hx & 0x7FFFFFFF;
  if (ix >= 0x3FF00000) {
    if (((ix - 0x3FF00000) | lo(x)) == 0) return hx > 0 ? 0.0 : pi + 2.0 * pio2_lo;
    return (x - x) / (x - x);
  }
  if (ix < 0x3FE00000) {
    if (ix <= 0x3C600000) return pio2_hi + pio2_lo;
    double z = x * x, r = P(z) / Q(z);
    return pio2_hi - (x - (pio2_lo - x * r));
  }
  if (hx < 0) {
    double z = (1.0 + x) * 0.5, s = std::sqrt(z), r = P(z) / Q(z), w = r * s - pio2_lo;
    return pi - 2.0 * (s + w);
  }
  double z = (1.0 - x) * 0.5, s = std::sqrt(z), df = withLow(s, 0), c = (z - df * df) / (s + df);
  double r = P(z) / Q(z), w = r * s + c;
  return 2.0 * (df + w);
}

// Math.atan (fdlibm s_atan.c)
inline double atan(double x) {
  static const double atanhi[] = {words(0x3FDDAC67, 0x0561BB4F), words(0x3FE921FB, 0x54442D18), words(0x3FEF730B, 0xD281F69B),
                                  words(0x3FF921FB, 0x54442D18)};
  static const double atanlo[] = {words(0x3C7A2B7F, 0x222F65E2), words(0x3C81A626, 0x33145C07), words(0x3C700788, 0x7AF0CBBD),
                                  words(0x3C91A626, 0x33145C07)};
  static const double aT[] = {words(0x3FD55555, 0x5555550D), words(0xBFC99999, 0x9998EBC4), words(0x3FC24924, 0x920083FF),
                              words(0xBFBC71C6, 0xFE231671), words(0x3FB745CD, 0xC54C206E), words(0xBFB3B0F2, 0xAF749A6D),
                              words(0x3FB10D66, 0xA0D03D51), words(0xBFADDE2D, 0x52DEFD9A), words(0x3FA97B4B, 0x24760DEB),
                              words(0xBFA2B444, 0x2C6A6C2F), words(0x3F90AD3A, 0xE322DA11)};
  int32_t hx = hi(x), ix = hx & 0x7FFFFFFF;
  int id;
  if (ix >= 0x44100000) {
    if (ix > 0x7FF00000 || (ix == 0x7FF00000 && lo(x) != 0)) return x + x;
    return hx > 0 ? atanhi[3] + atanlo[3] : -atanhi[3] - atanlo[3];
  }
  if (ix < 0x3FDC0000) {
    if (ix < 0x3E200000 && 1.0e300 + x > 1.0) return x;
    id = -1;
  } else {
    x = std::fabs(x);
    if (ix < 0x3FF30000) {
      if (ix < 0x3FE60000) { id = 0; x = (2.0 * x - 1.0) / (2.0 + x); }
      else { id = 1; x = (x - 1.0) / (x + 1.0); }
    } else if (ix < 0x40038000) { id = 2; x = (x - 1.5) / (1.0 + 1.5 * x); }
    else { id = 3; x = -1.0 / x; }
  }
  double z = x * x, w = z * z;
  double s1 = z * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))));
  double s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))));
  if (id < 0) return x - x * (s1 + s2);
  z = atanhi[id] - ((x * (s1 + s2) - atanlo[id]) - x);
  return hx < 0 ? -z : z;
}

// Math.hypot (builtins/math.tq MathHypot)
inline double hypot(std::initializer_list<double> values) {
  if (values.size() == 0) return 0;
  double max = 0;
  bool nan = false;
  for (double v : values) {
    if (std::isnan(v)) nan = true;
    else if (std::fabs(v) > max) max = std::fabs(v);
  }
  if (max == INFINITY) return INFINITY;
  if (nan) return NAN;
  if (max == 0) return 0;
  double sum = 0, compensation = 0;
  for (double v : values) {
    double n = std::fabs(v) / max, summand = n * n - compensation, preliminary = sum + summand;
    compensation = (preliminary - sum) - summand;
    sum = preliminary;
  }
  return std::sqrt(sum) * max;
}
template <typename... T> inline double hypot(T... v) { return hypot({(double)v...}); }

// Math.round (CodeStubAssembler::Float64Round): ceil, less one unless ceil - 1/2 <= x - so -0.4 is -0
inline double round(double v) {
  double c = std::ceil(v);
  return c - 0.5 <= v ? c : c - 1.0;
}

}  // namespace lf::jsmath
