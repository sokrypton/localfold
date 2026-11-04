// A design's metrics and its pre-oracle verdict: upstream's own evaluation of what this port just wrote.
//
// 🔴 THE ARITHMETIC IS cuda/rfd4p/src/metrics.h, DERIVED AT BUILD TIME AND NOT REIMPLEMENTED. That file is
// already CUDA-free - it is `shared/rfd4p/metrics.js` in C++, which `tools/check-rfd4p-metrics.py` holds to
// upstream's own metric manager and `gate.py` holds value for value to `design-metrics.mjs` - so the one thing
// stopping this port from simply including it was that it pulls in `generate.cuh` for `AtomRecord` and the
// insertion mapping, and that file carries three `__global__` kernels clang cannot compile. `derive.sh`
// extracts the host half and repoints those includes; see it for why that beats copying or moving the code.
//
// What is this file's own work is the other half of the call: upstream's metrics read the INPUT through tables
// the featuriser writes beside the tensors (`g/m*`, `outputTables`), and those have to be handed over in the
// layout `metrics::Inputs` declares. cuda/rfd4p/src/rfd4p.cu's `--metrics=` block is the reading.
#include "rfd4p.h"
#include "design.h"
#include "rfd4p-metrics.h"
#include "rfd4p/src/metrics_tags.h"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

bool writeMetrics(const std::string& path, const Design& d, int n, bool* pass) {
  // 🔴 A WORK DIRECTORY FEATURISED WITHOUT THE METRIC TABLES IS REFUSED BY NAME, not quietly scored on
  // whatever happens to be there: every one of these is an input the metrics read the design against, and a
  // missing table would read as a design that conditions on nothing.
  if (!M.has("g/mTok")) {
    fprintf(stderr, "--metrics needs a work directory featurised with the metric tables\n");
    return false;
  }
  rfd4p::metrics::Inputs in;
  in.nTokens = n;
  auto ints = [&](const char* key, size_t count) {
    return std::vector<int>(M.hostI(key), M.hostI(key) + count);
  };
  in.tok = ints("g/mTok", (size_t)n * 3);
  in.chainIid = ints("g/mChainIid", (size_t)n * 4);
  in.compType = ints("g/mCompType", (size_t)n * 8);
  in.seq = ints("g/mSeq", (size_t)n * 2);
  in.hot = ints("g/mHot", (size_t)n * 37);
  in.pnUnit = ints("g/mPnUnit", (size_t)n * 4);
  in.tokMol = ints("g/mTokMol", (size_t)n * 2);
  in.inAtom = ints("g/mInAtom", M.len("g/mInAtom"));
  if (M.has("g/mInBonds") && M.len("g/mInBonds") > 0) in.inBonds = ints("g/mInBonds", M.len("g/mInBonds"));
  in.condCoord = M.has("g/condCoord") ? M.hostF("g/condCoord") : nullptr;
  in.condIntCrd = M.has("g/condIntCrd") ? M.hostF("g/condIntCrd") : nullptr;
  in.coordGroup = M.has("g/coordGroup") ? M.hostI("g/coordGroup") : nullptr;
  in.resName = ints("g/outResName", (size_t)n * 2);
  in.condIndex = M.has("g/condIndex") ? M.hostI("g/condIndex") : nullptr;
  // 🔴 `g/mSeqProb` IS FLOAT32 CARRIED IN THE INTEGER TABLE, bit for bit: the exporter writes the sequence
  // probability conditions through the same int32 channel as everything else, and reading them as integers
  // gives 28 columns of enormous nonsense that the adherence metrics score a design against.
  {
    const std::vector<int> bits = ints("g/mSeqProb", (size_t)n * 28);
    in.seqProb.resize(bits.size());
    memcpy(in.seqProb.data(), bits.data(), bits.size() * 4);
  }
  rfd4p::metrics::Result r;
  try {
    r = rfd4p::metrics::designMetrics(d.records, in, d.proposed,
                                      rfd4p::metricsTagsFromBits((int)M.meta("meta/metricsTags", 0)));
  } catch (const std::exception& e) {
    fprintf(stderr, "--metrics: %s\n", e.what());
    return false;
  }
  std::ofstream(path) << rfd4p::metrics::toJson(r) << "\n";
  if (pass) *pass = r.pass;
  return true;
}
