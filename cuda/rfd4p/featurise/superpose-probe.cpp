// A deterministic point pair superposed by `src/superpose.h`, for `gate.py` to diff against
// `output.js`'s own `superimposeCoords3D`. Both sides generate the points, for the reason the
// assignment probe does: a double through text is a different number in the last place.
//
//   superpose-probe <seed> <n> <noise>
#include "../src/superpose.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: superpose-probe seed n noise\n"); return 2; }
  const int seed = atoi(argv[1]), n = atoi(argv[2]);
  const double noise = atof(argv[3]);
  // A reflected set, which is the only way det(H) goes negative - where a Kabsch that can
  // return an improper rotation does.
  const double flip = (argc > 4 && argv[4][0] == '1') ? -1.0 : 1.0;
  uint32_t s = (uint32_t)seed;
  auto next = [&]() { s = s * 1664525u + 1013904223u; return (double)(s >> 8) / 16777216.0; };
  std::vector<double> fixed(n * 3), mobile(n * 3);
  for (int i = 0; i < n * 3; ++i) fixed[i] = next() * 40.0 - 20.0;
  // mobile = fixed rotated by a fixed angle, translated, plus noise
  const double c = 0.8253356149, sn = 0.5646424734;   // cos(0.6), sin(0.6)
  for (int i = 0; i < n; ++i) {
    const double x = fixed[i * 3], y = fixed[i * 3 + 1], z = fixed[i * 3 + 2] * flip;
    mobile[i * 3] = c * x - sn * y + 3.5 + (next() - 0.5) * noise;
    mobile[i * 3 + 1] = sn * x + c * y - 7.25 + (next() - 0.5) * noise;
    mobile[i * 3 + 2] = z + 11.0 + (next() - 0.5) * noise;
  }
  const rfd4p::superpose::Fit f = rfd4p::superpose::superimpose(fixed, mobile);
  for (int i = 0; i < 9; ++i) printf("%.12f ", f.R[i]);
  for (int k = 0; k < 3; ++k) printf("%.12f ", f.fixedCenter[k]);
  for (int k = 0; k < 3; ++k) printf("%.12f ", f.mobileCenter[k]);
  printf("\n");
  return 0;
}
