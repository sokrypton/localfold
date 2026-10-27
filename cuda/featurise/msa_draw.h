// Which alignment rows a trunk pass reads: AlphaFold 3's per-pass MSA draw. The C++ twin of shared/af3/msa-draw.js,
// integer for integer - read its header for why (AF3 shuffles and truncates to num_msa inside every trunk pass, and
// this port drew once for the whole fold) and for the three modes. test/msa-draw.test.js pins the values; cuda/af3
// (trunk.cuh drawMsa) and metal/af3 (trunk.mm drawMsa) gather these rows before each pass's MSA embedding.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace lf {

inline uint32_t msaMix32(uint32_t x) {      // lowbias32
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}

// a pass's rows: `count` distinct indices of `rows`, ascending (every index when rows <= count)
inline std::vector<int> msaDraw(uint64_t seed, int pass, int rows, int count) {
  std::vector<int> order(rows);
  for (int r = 0; r < rows; ++r) order[r] = r;
  if (rows <= count) return order;
  uint32_t state = msaMix32(msaMix32((uint32_t)seed ^ msaMix32((uint32_t)(seed >> 32))) + (uint32_t)(pass + 1) * 0x9e3779b9u);
  for (int at = 0; at < count; ++at) {
    state += 0x9e3779b9u;
    uint32_t span = (uint32_t)(rows - at);
    int swap = at + (int)(((uint64_t)msaMix32(state) * span) >> 32);
    std::swap(order[at], order[swap]);
  }
  order.resize(count);
  std::sort(order.begin(), order.end());
  return order;
}

inline uint32_t msaSeedKey(uint64_t seed) { return msaMix32((uint32_t)seed ^ msaMix32((uint32_t)(seed >> 32))); }

// protenix's rows for a pass (msa-draw.js msaNativeRows): a count uniform in [1, rows], then that many rows, ascending
inline std::vector<int> msaNativeRows(uint64_t seed, int pass, int rows) {
  uint32_t state = msaMix32(msaSeedKey(seed) + (uint32_t)(pass + 1) * 0x9e3779b9u + 0x632be5abu);
  auto next = [&]() { state += 0x9e3779b9u; return msaMix32(state); };
  const int count = 1 + (int)(((uint64_t)next() * (uint32_t)rows) >> 32);
  std::vector<int> order(rows);
  for (int r = 0; r < rows; ++r) order[r] = r;
  for (int at = 0; at < count; ++at) {
    int swap = at + (int)(((uint64_t)next() * (uint32_t)(rows - at)) >> 32);
    std::swap(order[at], order[swap]);
  }
  order.resize(count);
  std::sort(order.begin(), order.end());
  return order;
}

// protenix's MC dropout on the recycled pair's projection (msa-draw.js): whether a seed applies it, a pass's key, the
// keep threshold at a rate, and the element hash a kernel compares with it - (pair row, channel), so nothing overflows
inline bool mcDropoutApplies(uint64_t seed, double odds) {
  return odds > 0 && msaMix32(msaSeedKey(seed) ^ 0x2545f491u) < (uint32_t)std::floor(odds * 4294967296.0);
}
inline uint32_t mcDropoutKey(uint64_t seed, int pass) {
  return msaMix32(msaSeedKey(seed) + (uint32_t)(pass + 1) * 0x9e3779b9u + 0x6d2b79f5u);
}
inline uint32_t mcDropoutThreshold(double rate) { return (uint32_t)std::floor(rate * 4294967296.0); }

// mode: "pass" (a fresh draw every pass, AF3's), "once" (pass 0's for every pass), "prefix" (the first rows: chai-1),
// "native" (protenix's: msaNativeRows)
inline std::vector<int> msaDrawRows(const std::string& mode, uint64_t seed, int pass, int rows, int count) {
  if (mode == "native") return msaNativeRows(seed, pass, rows);
  if (mode == "prefix" || rows <= count) {
    std::vector<int> v(std::min(rows, count));
    for (int r = 0; r < (int)v.size(); ++r) v[r] = r;
    return v;
  }
  if (mode != "pass" && mode != "once") throw std::runtime_error("--msa-draw=" + mode + ": not pass, once, prefix or native");
  return msaDraw(seed, mode == "once" ? 0 : pass, rows, count);
}

// The pool as an input carries it: `batch.msaPacked`, an int32 a cell (code | deletion << 8 | mask << 16 -
// shared/af3/msa-draw.js packMsa), or an older input's three entries. A trunk widens only the rows a pass reads.
struct MsaPoolView {
  const int32_t* packed = nullptr;                             // batch.msaPacked, or
  const int32_t* codes = nullptr; const float* deletion = nullptr; const float* mask = nullptr;   // the three entries
};
inline void unpackMsaRow(const MsaPoolView& pool, size_t row, size_t n, int* codes, float* deletion, float* mask) {
  const size_t at = row * n;
  if (pool.packed) {
    for (size_t k = 0; k < n; ++k) {
      const int32_t v = pool.packed[at + k];
      codes[k] = (int8_t)(v & 0xff); deletion[k] = (float)((v >> 8) & 0xff); mask[k] = (float)((v >> 16) & 1);
    }
    return;
  }
  for (size_t k = 0; k < n; ++k) { codes[k] = pool.codes[at + k]; deletion[k] = pool.deletion[at + k]; mask[k] = pool.mask[at + k]; }
}

}  // namespace lf
