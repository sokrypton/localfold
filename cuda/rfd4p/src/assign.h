// The linear sum assignment (`scipy.optimize.linear_sum_assignment`), as `output.js` has it.
//
// 🔴 IT DECIDES WHERE AN UNINDEXED MOTIF GOES, so a wrong answer is a plausible structure with
// the motif in the wrong place - not a crash and not a number anybody reads. That is why it is
// ported exactly rather than replaced by something greedy that agrees on easy inputs.
//
// 🔴 AND IT IS JONKER-VOLGENANT WITH POTENTIALS, NOT THE O(n^4) TEXTBOOK HUNGARIAN. The two
// agree on the optimum where it is unique and may differ where it is NOT - a tie is broken by
// the order the algorithm happens to reach it - so a different solver is a different answer on
// exactly the inputs where it matters least and is hardest to notice.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace rfd4p {
namespace assign {

using std::vector;

struct Assignment {
  vector<int> row, col;
};

/**
 * The minimum-cost assignment of `nRows` rows to `nCols` columns, row-major cost.
 *
 * 🔴 A NON-FINITE COST BECOMES A LARGE FINITE PENALTY rather than an infinity, because the dual
 * potentials have to stay finite for the algorithm to terminate - and the caller then checks
 * whether the chosen cell was one of them. Leaving infinities in makes `delta` infinite and the
 * potentials NaN, which fails silently as an arbitrary assignment.
 */
inline Assignment minimize(const vector<double>& cost, int nRows, int nCols) {
  Assignment out;
  if (nRows <= 0 || nCols <= 0) return out;
  // 🔴 MORE ROWS THAN COLUMNS IS SOLVED TRANSPOSED, AND THAT HALF IS LOAD-BEARING: the loop
  // below assumes every row can be matched, and without the transpose it does not TERMINATE -
  // the augmenting walk never reaches an unmatched column. (The sort afterwards keeps the pairs
  // in row order; measured, removing it changed nothing on any shape tried, so it is the
  // reference's shape rather than a rule.)
  if (nRows > nCols) {
    vector<double> t((size_t)nCols * nRows);
    for (int r = 0; r < nRows; ++r) {
      for (int c = 0; c < nCols; ++c) t[(size_t)c * nRows + r] = cost[(size_t)r * nCols + c];
    }
    const Assignment flipped = minimize(t, nCols, nRows);
    vector<std::pair<int, int>> pairs;
    for (size_t k = 0; k < flipped.row.size(); ++k) {
      pairs.push_back({flipped.col[k], flipped.row[k]});
    }
    std::sort(pairs.begin(), pairs.end());
    for (const auto& [r, c] : pairs) { out.row.push_back(r); out.col.push_back(c); }
    return out;
  }

  double maxFinite = 0;
  for (const double v : cost) {
    if (std::isfinite(v) && std::fabs(v) > maxFinite) maxFinite = std::fabs(v);
  }
  const double infPenalty = (maxFinite + 1.0) * (nRows + 1) * 1e6;
  const double INF = std::numeric_limits<double>::infinity();

  vector<double> u(nRows + 1, 0.0), v(nCols + 1, 0.0), minv(nCols + 1, 0.0);
  vector<int> p(nCols + 1, 0), way(nCols + 1, 0);
  vector<uint8_t> used(nCols + 1, 0);

  for (int i = 1; i <= nRows; ++i) {
    p[0] = i;
    int j0 = 0;
    std::fill(minv.begin(), minv.end(), INF);
    std::fill(used.begin(), used.end(), 0);
    do {
      used[j0] = 1;
      const int i0 = p[j0];
      double delta = INF;
      int j1 = 0;
      const size_t rowOffset = (size_t)(i0 - 1) * nCols;
      for (int j = 1; j <= nCols; ++j) {
        if (used[j]) continue;
        const double raw = cost[rowOffset + (j - 1)];
        const double c = std::isfinite(raw) ? raw : infPenalty;
        const double cur = c - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (int j = 0; j <= nCols; ++j) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  out.row.resize(nRows);
  out.col.assign(nRows, 0);
  for (int r = 0; r < nRows; ++r) out.row[r] = r;
  for (int j = 1; j <= nCols; ++j) {
    if (p[j] > 0) out.col[p[j] - 1] = j - 1;
  }
  return out;
}

/**
 * The maximum-score assignment.
 *
 * It subtracts from the maximum rather than negating, which is the reference's own
 * transformation. Measured: negating instead is INERT - adding a constant to every cost cannot
 * move the optimum - so this is the reference's shape and not a rule, and it is written down
 * that way rather than as a 🔴 nobody can falsify. What does matter is that a non-finite score
 * stays non-finite here instead of becoming a very negative number the solver would choose.
 */
inline Assignment maximize(const vector<double>& score, int nRows, int nCols) {
  if (nRows <= 0 || nCols <= 0) return {};
  double maxVal = -std::numeric_limits<double>::infinity();
  for (const double v : score) if (std::isfinite(v) && v > maxVal) maxVal = v;
  if (!std::isfinite(maxVal)) maxVal = 0;
  vector<double> cost(score.size());
  for (size_t i = 0; i < score.size(); ++i) {
    cost[i] = std::isfinite(score[i]) ? maxVal - score[i]
                                      : std::numeric_limits<double>::infinity();
  }
  return minimize(cost, nRows, nCols);
}

}  // namespace assign
}  // namespace rfd4p
