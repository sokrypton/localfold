// A job's pocket and contact constraints on Metal, as cuda/af3/src/constraints.cuh does them on CUDA: each site's
// per-pair term as a table of a few rows and each pair's row in it (cuda/featurise/constraints.h, built on the host from
// the site's own weights and the batch the featuriser exported), ADDED to a pair after the kernels that build it - so a
// fold without constraints is the one it was. boltz2's two sites already add the unspecified constant to every pair;
// their table is the term MINUS it.
#pragma once
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "af3.h"
#include "constraints.h"   // (cuda/featurise/constraints.h: on the include path, beside af3_weights.h)

struct ConstraintTableHost { const float* key = nullptr; lf::ContactTable t; };

// the site's table, rebuilt when the batch changes (a new input: its host array moves), uploaded into scratch
inline bool constraintTableFor(const std::string& site, const std::string& batchKey,
                               const std::function<lf::ContactTable()>& build, const float** index, const float** table) {
  static std::map<std::string, ConstraintTableHost> cache;
  if (!M.has(batchKey)) return false;
  const float* key = M.hostF(batchKey);
  ConstraintTableHost& c = cache[site];
  if (c.key != key) { c.t = build(); c.key = key; }
  float* i = scratch<float>("ct.index." + site, c.t.index.size());
  float* tb = scratch<float>("ct.table." + site, c.t.table.size());
  upload(i, c.t.index.data(), c.t.index.size() * 4);
  upload(tb, c.t.table.data(), c.t.table.size() * 4);
  *index = i; *table = tb;
  return true;
}
inline std::vector<float> constraintHostCopy(const std::string& k) {
  const float* p = M.hostF(k);
  return std::vector<float>(p, p + M.len(k));
}
// out[row][offset + c] += table[index[p0 + row]][c], rows [0, rows)
inline void addConstraintRows(float* out, const float* index, const float* table, size_t rows, size_t p0, int C,
                              int stride, int offset) {
  run1d("af3_add_table", rows * C, AddTableArgs{out, index, table, (u64)rows, (u64)p0, (uint)C, (uint)stride, (uint)offset, 0});
}
// boltz2's contact conditioning at the site whose weights are `pre` + contact*: added to `pair` (pairs x C). Its weights
// are read either way, so an unconstrained fold does not list them as never read.
inline void addBoltz2Constraints(float* pair, size_t pairs, int C, const std::string& pre) {
  const float *fw = M.f(pre + "contactFourierWeights"), *fb = M.f(pre + "contactFourierBias");
  const float *ew = M.f(pre + "contactEncoderWeights"), *eb = M.f(pre + "contactEncoderBias");
  const float *un = M.f(pre + "contactEncodingUnspecified"), *us = M.f(pre + "contactEncodingUnselected");
  const float *index = nullptr, *table = nullptr;
  if (!constraintTableFor(pre, "batch.contactConditioning", [&] {
        lf::ContactTable t = lf::boltz2ContactTable(constraintHostCopy("batch.contactConditioning"),
                                                    constraintHostCopy("batch.contactThreshold"), fw, fb, ew, eb, un, us, C);
        for (int r = 0; r < t.entries; ++r) for (int c = 0; c < C; ++c) t.table[(size_t)r * C + c] -= un[c];
        return t;
      }, &index, &table)) return;
  addConstraintRows(pair, index, table, pairs, 0, C, C, 0);
}
// chai-1's two restraints through the (14, C) rows of `key` from row `first`: added to rows [0, rows) of `out`, each
// `stride` wide from column `offset`, which are pairs [p0, p0 + rows)
inline void addChai1Constraints(float* out, size_t rows, size_t p0, int C, int stride, int offset, const std::string& site,
                                const std::string& key, int first) {
  if (M.has("batch.tokenDistanceRestraint") && !hasW(key))
    die("the job has pocket or contact constraints and these chai-1 weights carry no %s: delete the cached blob to fetch"
        " the current one", key.c_str());
  const float *index = nullptr, *table = nullptr;
  if (!constraintTableFor(site, "batch.tokenDistanceRestraint", [&] {
        return lf::chai1RestraintTable(constraintHostCopy("batch.tokenDistanceRestraint"),
                                       constraintHostCopy("batch.tokenPairPocketRestraint"), M.f(key) + (size_t)first * C, C);
      }, &index, &table)) return;
  addConstraintRows(out, index, table, rows, p0, C, stride, offset);
}
