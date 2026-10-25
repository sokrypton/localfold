// A deterministic matrix solved by `src/assign.h`, for `gate.py` to diff against `output.js`'s
// own `linearSumAssignmentMinimize`/`Maximize`.
//
// 🔴 THE MATRIX IS GENERATED THE SAME WAY ON BOTH SIDES rather than passed between them: a
// Float64 written and re-read as text is a different matrix in the last place, and an assignment
// is a DISCRETE answer that can flip on exactly that. The generator is a plain LCG so both
// languages can spell it.
//
//   assign-probe <seed> <rows> <cols> <n|x> [infEvery]
#include "../src/assign.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: assign-probe seed rows cols n|x [infEvery]\n"); return 2; }
  const int seed = atoi(argv[1]), nR = atoi(argv[2]), nC = atoi(argv[3]);
  const bool maxi = argv[4][0] == 'x';
  const int infEvery = argc > 5 ? atoi(argv[5]) : 0;
  uint32_t s = (uint32_t)seed;
  auto next = [&]() { s = s * 1664525u + 1013904223u; return (double)(s >> 8) / 16777216.0; };
  std::vector<double> m((size_t)nR * nC);
  for (size_t i = 0; i < m.size(); ++i) {
    const double v = next() * 20.0 - 10.0;
    m[i] = (infEvery && (i % (size_t)infEvery) == 0) ? INFINITY : v;
  }
  const auto a = maxi ? rfd4p::assign::maximize(m, nR, nC) : rfd4p::assign::minimize(m, nR, nC);
  for (size_t k = 0; k < a.row.size(); ++k) printf("%d:%d ", a.row[k], a.col[k]);
  printf("\n");
  return 0;
}
