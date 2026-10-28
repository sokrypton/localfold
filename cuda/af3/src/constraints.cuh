// A job's pocket and contact constraints on the device: each site's per-pair term as a table of a few rows and each
// pair's row in it (cuda/featurise/constraints.h, built on the host from the site's own weights and the batch the
// featuriser exported), ADDED to a pair - so no existing kernel changes and a fold without constraints is the one it
// was. boltz2's two sites already add the unspecified constant to every pair; their table is the term MINUS it.
//   boltz2: trunk.embedder.contact* (the z-init) and confidence.reembed.contact* (the head's rebuilt z)
//   chai-1: trunk.embedder.chaiRestraintTrunk (the z-init) and diffusion.conditioning.structurePairWeights rows
//           149..162 (its structure pair, beside the trunk's)
#pragma once
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "../../featurise/constraints.h"

struct ConstraintTable { const float* index = nullptr; const float* table = nullptr; int entries = 0; };

// pair[p][c] += table[index[p]][c] over pairs [first, end), the pair stored float or bf16 (PT)
template <class PT = float>
__global__ void addConstraintTableK(float* pair, const float* __restrict__ index, const float* __restrict__ table,
                                    size_t first, size_t end, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x, p = first + t / C;
  if (p >= end) return;
  int c = (int)(t % C);
  size_t at = p * C + c;
  pairSt<PT>(pair, at, pairLd<PT>(pair, at) + __ldg(table + (size_t)__ldg(index + p) * C + c));
}
// LOCALFOLD_SAVE_SEAMS=<dir>: a float buffer as <dir>/<name>.npy (a site's rows before and after its term: the term)
inline void constraintSeam(const std::string& name, const float* d, size_t n) {
  if (const char* dir = getenv("LOCALFOLD_SAVE_SEAMS")) writeNpy(std::string(dir) + "/" + name + ".npy", download(d, n), { n });
}
template <class PT = float>
inline void addConstraintTable(float* pair, const ConstraintTable& t, size_t first, size_t end, int C,
                               const char* seam = nullptr) {
  if (t.table == nullptr || end <= first) return;
  const bool save = seam != nullptr && std::is_same_v<PT, float> && first == 0;
  if (save) constraintSeam(std::string(seam) + "_before", pair, end * C);
  addConstraintTableK<PT><<<blocks((end - first) * C), 256, 0, STREAM>>>(pair, t.index, t.table, first, end, C);
  if (save) constraintSeam(std::string(seam) + "_after", pair, end * C);
}
// ...into rows of a wider buffer: out[row][offset + c] += table[index[p0 + row]][c] (chai-1's structure pair rows)
__global__ void addConstraintRowsK(float* out, const float* __restrict__ index, const float* __restrict__ table,
                                   size_t p0, size_t rows, int stride, int offset, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t row = t / C; int c = (int)(t % C);
  out[row * stride + offset + c] += __ldg(table + (size_t)__ldg(index + p0 + row) * C + c);
}

// the table for one site, rebuilt when the batch changes (a new job: its host array moves, as denseRank's does)
inline ConstraintTable constraintTableFor(const std::string& site, const std::string& batchKey,
                                          const std::function<lf::ContactTable()>& build) {
  static LF_TLS std::map<std::string, std::pair<const float*, std::pair<ConstraintTable, std::vector<float*>>>> cache;
  if (!M.has(batchKey)) return {};
  const float* host = M.f(batchKey);
  auto it = cache.find(site);
  if (it != cache.end() && it->second.first == host) return it->second.second.first;
  if (it != cache.end()) for (float* d : it->second.second.second) CK(cudaFree(d));
  lf::ContactTable t = build();
  float* index = upload(t.index.data(), t.index.size());
  float* table = upload(t.table.data(), t.table.size());
  ConstraintTable out{index, table, t.entries};
  cache[site] = {host, {out, {index, table}}};
  return out;
}
inline std::vector<float> hostCopy(const std::string& k) { const float* p = M.f(k); return std::vector<float>(p, p + M.len(k)); }

// boltz2's contact conditioning at the site whose weights are `pre` + contact* (trunk.embedder., confidence.reembed.):
// each row the whole term minus the unspecified constant every pair already carries. Its weights are read either way,
// so an unconstrained fold does not list them as never read.
inline ConstraintTable boltz2ConstraintTable(const std::string& pre, int C) {
  const float *fw = M.f(pre + "contactFourierWeights"), *fb = M.f(pre + "contactFourierBias");
  const float *ew = M.f(pre + "contactEncoderWeights"), *eb = M.f(pre + "contactEncoderBias");
  const float *un = M.f(pre + "contactEncodingUnspecified"), *us = M.f(pre + "contactEncodingUnselected");
  return constraintTableFor(pre, "batch.contactConditioning", [&] {
    lf::ContactTable t = lf::boltz2ContactTable(hostCopy("batch.contactConditioning"), hostCopy("batch.contactThreshold"),
                                                fw, fb, ew, eb, un, us, C);
    for (int r = 0; r < t.entries; ++r) for (int c = 0; c < C; ++c) t.table[(size_t)r * C + c] -= un[c];
    return t;
  });
}
// chai-1's two restraints through `rows` (14, C), the weight `key` from its row `first`
inline ConstraintTable chai1ConstraintTable(const std::string& site, const std::string& key, int first, int C) {
  if (M.has("batch.tokenDistanceRestraint") && !hasW(key)) {
    fprintf(stderr, "the job has pocket or contact constraints and these chai-1 weights carry no %s: delete the cached"
                    " blob to fetch the current one\n", key.c_str());
    exit(1);
  }
  return constraintTableFor(site, "batch.tokenDistanceRestraint", [&] {
    const float* w = M.f(key) + (size_t)first * C;
    return lf::chai1RestraintTable(hostCopy("batch.tokenDistanceRestraint"), hostCopy("batch.tokenPairPocketRestraint"), w, C);
  });
}
