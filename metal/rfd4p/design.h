// The design a sampled trajectory becomes: the atom records, and what the writer and the metrics both read.
//
// 🔴 THE RECORDS ARE cuda/rfd4p's OWN, DERIVED AT BUILD TIME AND NOT COPIED. `metal/rfd4p/derive.sh` extracts
// the host half of `cuda/rfd4p/src/generate.cuh` into `rfd4p-gen-host.h` on every build - `AtomRecord`, the
// record builder, the two writers, the insertion mapping and the input-frame alignment - because all of it is
// plain C++ that both ports must agree on to the BYTE: `cuda/rfd4p/check-writer.mjs` holds that writer to the
// page's text, and `gate.py` holds the design metrics to `design-metrics.mjs` value for value. A second
// implementation here would have to pass the same two gates and would drift between them.
//
// 🔴 AND THE PORT HAD ONE, WHICH DIVERGED. This port wrote its own 180-line records-and-writers pair and
// nothing ever compared it with the other port's: it named every atom by its atom37 SLOT rather than the
// residue's own atom order, left out the PDB's `HEADER`/`TITLE` and every `TER`, dropped the two mmCIF
// annotation columns (`is_unindexed_insertion` and the coordinate-conditioned region), printed coordinates
// through `snprintf("%8.3f")` where the reference rounds with `toFixed`, and carried no source component at
// all. Every geometry gate passed, because none of that is geometry - check-writer.mjs's own complaint about
// the NVIDIA writer, one port along.
#pragma once
#include "rfd4p-gen-host.h"
#include <string>
#include <vector>

// What the writer and the design metrics read off the input: the featuriser's `outputTables` and the job's
// conditioning, as the work directory holds them. A de-novo design leaves the naming tables null - the token
// index IS the residue number there - and `inName` is what says so.
struct OutInput {
  int n = 0;
  const int* outResId = nullptr;
  const int* outChain = nullptr;
  const int* chainIid = nullptr;
  // the coordinate conditioning, and the insertion head's logits from the LAST step (the only pass that asks)
  const int* condIndex = nullptr;
  const int* resType = nullptr;
  const float* coords = nullptr;
  const float* condCoord = nullptr;
  const float* condIntCrd = nullptr;
  const int* coordGroup = nullptr;
  const float* insLogits = nullptr;
  // the per-token and per-slot naming an input structure supplies
  const int* atomize = nullptr; const int* hetero = nullptr; const int* entity = nullptr;
  const int* insCode = nullptr; const int* resName = nullptr; const int* srcComp = nullptr;
  const int* inName = nullptr; const int* inElem = nullptr; const int* inCharge = nullptr;
  const int* inB = nullptr; const int* inOcc = nullptr;
};

// A finished design: the records, the insertion head's mapping AS PROPOSED (which the metrics count the
// unindexed motif against, and which is not the same as the one that could be applied), and the geometry check.
struct Design {
  std::vector<rfd4p::AtomRecord> records;
  rfd4p::Insertion proposed;
  double caSpacing[3] = {0, 0, 0};   // consecutive CA-CA within a chain: median, min, max
  int grafted = 0;                   // unindexed tokens the mapping placed
  bool unindexedUnplaced = false;    // ...and whether there were any it could not
  bool shifted = false;              // whether the centring offset was available and added back
};

// The decoder's heads into a structure (`formatSampleLocalLatents`), over the decoder's ATTENDED rows.
rfd4p::GenOut decoderPostprocess(int n, const float* caNm, const int* seq, const float* logits,
                                 const float* coorsFlat, const float* presence, const float* emit,
                                 const int* coordGroup, const int* attnIdx, int na);
// The records, the unindexed motif grafted onto the tokens the insertion head names, and the whole design put
// back into the INPUT's frame. `g` is rewritten in place, as the grafting and the alignment both do.
Design buildDesign(rfd4p::GenOut& g, const OutInput& in);
// The design written - mmCIF where the path ends `.cif` (RFD4-Proteina's own format), PDB otherwise.
int writeStructure(const std::string& path, const Design& d);
// What to write about a design: the metrics, and optionally the refolding oracle over what passes them.
struct MetricsRun {
  std::string path;         // the metrics JSON
  std::string refold;       // "" for none, else "af3" or "af2" - upstream's refolding oracles
  std::string designPath;   // the structure just written, which names the refold's files
  std::string genDir;       // where `input-components.cif` is
  uint32_t seed = 0;
  bool keepRaw = false;     // keep AF2's PAE logits, which a checker reads again
};
// Upstream's design metrics and pre-oracle verdict over those records, as `design-metrics.mjs` writes them, and
// then the refolding oracle on what survives the filters - as upstream runs it, and in that order.
// Returns false and says why when the work directory was featurised without the metric tables.
bool writeMetrics(const MetricsRun& run, const Design& d, int n, bool* pass);
