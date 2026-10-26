// The whole generation: noise in, a structure out. Drives `trunk.cuh` over the flow schedule in
// `sampler.cuh`, decodes the last step's clean estimate with `decoder.cuh`, and writes a PDB.
//
// 🔴 ONLY 35 OF THE 897 SEQUENCE COLUMNS MOVE BETWEEN STEPS - `xt_bb_ca` at offset 0 and
// `xt_local_latents` at offset 3 - so the features are exported once and those columns are
// overwritten on the device. The pair track's first 30 columns are its CA distance bins, refilled
// the same way. Everything else about a step is the conditioning, which depends on t alone and is
// exported for every step.
#pragma once
#include "decoder.cuh"
#include "sampler.cuh"
#include "superpose.h"
#include "assign.h"
#include "../featurise/tables.inc"
#include <array>
#include <charconv>
#include <cstring>
#include <limits>
#include <string>
#include <map>

namespace rfd4p {

// x_t into the sequence features: the three CA columns and the thirty-two latent columns.
__global__ void writeStateK(float* __restrict__ feats, const float* __restrict__ bb,
                            const float* __restrict__ ll, int n, int fdim, int latent) {
  const int i = blockIdx.x;
  if (i >= n) return;
  float* row = feats + (size_t)i * fdim;
  for (int c = threadIdx.x; c < 3 + latent; c += blockDim.x) {
    row[c] = c < 3 ? bb[i * 3 + c] : ll[(size_t)i * latent + (c - 3)];
  }
}

// `xt_bb_ca_pair_dists`: a one-hot over `dim` buckets of the CA-CA distance, bucketed right-open
// against `limits[k] = min + (max - min) * k / (dim - 2)`, which is `torch.bucketize(right=False)`.
//
// 🔴 A PADDED TOKEN'S PAIR IS LEFT ALONE, NOT BINNED. The reference skips a row or a column whose
// `padding_mask` is clear, so those thirty columns stay zero where this kernel would have written
// a one-hot of a distance between two positions that are not there. It is inert on an input with
// no padding - which is every case measured here - and that is exactly why it is worth writing
// down rather than discovering on the first padded batch.
__global__ void pairDistBinsK(float* __restrict__ pair, const float* __restrict__ caNm,
                              const float* __restrict__ mask, int n, int pdim, int dim, float mn,
                              float mx) {
  size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)n * n) return;
  const int i = (int)(p / n), j = (int)(p % n);
  if (mask[i] == 0.f || mask[j] == 0.f) return;
  const float dx = caNm[i * 3] - caNm[j * 3];
  const float dy = caNm[i * 3 + 1] - caNm[j * 3 + 1];
  const float dz = caNm[i * 3 + 2] - caNm[j * 3 + 2];
  const float r = sqrtf(dx * dx + dy * dy + dz * dz);
  int bin = 0;
  while (bin < dim - 1 && mn + (mx - mn) * bin / (float)(dim - 2) < r) ++bin;
  float* row = pair + p * pdim;
  for (int c = 0; c < dim; ++c) row[c] = 0.f;
  row[bin] = 1.f;
}

// The emit masks applied to the trunk's two predictions, as `flowEmitMasks` does.
__global__ void maskEmitK(float* __restrict__ x, const float* __restrict__ emit, size_t rows,
                          int dim) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * (size_t)dim && emit[i / dim] == 0.f) x[i] = 0.f;
}

struct GenOut {
  std::vector<float> coors;      // [n, 37, 3] in Angstrom
  std::vector<uint8_t> atomMask; // [n, 37]
  std::vector<int> residueType;  // [n], -1 where nothing was emitted
};

/**
 * `_format_sample_local_latents`: the decoder's three heads into a structure. Per token, so it runs
 * on the host - the arrays are [n, 147] and [n, 111] and the work is an argmax and a table lookup.
 */
inline GenOut postprocess(int n, int nTypes, const std::vector<float>& logits,
                          const std::vector<float>& coorsFlat, const std::vector<float>& presence,
                          const std::vector<float>& caNm, const std::vector<int>& seq,
                          const std::vector<int>& tokenType, const std::vector<int>& coordGroup,
                          const std::vector<int>& allowed, const std::vector<int>& legal,
                          const std::vector<float>& emit, const std::vector<float>& pad) {
  GenOut out;
  out.coors.assign((size_t)n * 111, 0.f);
  out.atomMask.assign((size_t)n * 37, 0);
  out.residueType.assign(n, -1);
  for (int i = 0; i < n; ++i) {
    if (emit[i] == 0.f) {
      // Not emitted: no residue, but a padded token still carries the sampled CA in slot 1.
      if (pad[i] != 0.f) {
        out.atomMask[(size_t)i * 37 + 1] = 1;
        for (int k = 0; k < 3; ++k) out.coors[(size_t)i * 111 + 3 + k] = caNm[i * 3 + k] * 10.f;
      }
      continue;
    }
    const int tt = std::max(0, std::min(6, tokenType[i]));
    int best = 0;
    float bestV = -3.4028234663852886e38f;
    for (int r = 0; r < nTypes; ++r) {
      const float v = allowed[(size_t)tt * nTypes + r] ? logits[(size_t)i * nTypes + r]
                                                       : -3.4028234663852886e38f;
      if (v > bestV) { bestV = v; best = r; }
    }
    const int rt = seq[i] != 0 ? seq[i] : best;    // 0 is `<M>`
    out.residueType[i] = rt;
    for (int a = 0; a < 37; ++a) {
      const bool required = coordGroup[(size_t)i * 37 + a] >= 0;
      const bool present = presence[(size_t)i * 37 + a] > 0.f || a == 1 || required;
      if (!(legal[(size_t)rt * 37 + a] && present)) continue;
      out.atomMask[(size_t)i * 37 + a] = 1;
      for (int k = 0; k < 3; ++k) {
        const float rel = a == 1 ? 0.f : coorsFlat[(size_t)i * 111 + a * 3 + k];
        out.coors[(size_t)i * 111 + a * 3 + k] = (rel + caNm[i * 3 + k]) * 10.f;
      }
    }
  }
  return out;
}

/**
 * The per-token naming an INPUT structure supplies: its own residue numbers and chain letters.
 *
 * 🔴 EMPTY FOR A CONTIG, AND THAT IS NOT A MISSING CASE. A pure design job has no input, so the
 * token index IS the residue number and the chain letters are the ones this port assigned -
 * which is what `chainName` spells. A job with an input must keep what the file said, or a
 * motif comes out renumbered and no longer superposes on the target it was built against.
 */
// What the page's writer reads off the INPUT atoms (`processSampleToAtoms`), as the featuriser's
// `outputTables` packs it: per token from its first atom, and per input atom keyed by its slot in
// the token's OWN atom37 layout. Empty for a job with no input structure.
struct OutTables {
  std::vector<int> atomize, hetero, entity, insCode, resName, srcComp;   // per token
  std::vector<int> inName, inElem, inCharge, inB, inOcc;                 // per token x 37 (B, occ x2)
  bool present() const { return !inName.empty(); }
};

/** `packString`'s inverse: `words` little-endian ints back to the string, stopping at a zero. */
inline std::string unpackString(const int* w, int words) {
  std::string out;
  for (int k = 0; k < words * 4; ++k) {
    const char c = (char)(((uint32_t)w[k / 4] >> (8 * (k % 4))) & 0xffu);
    if (!c) break;
    out += c;
  }
  return out;
}
inline double unpackDouble(const int* w) {
  const uint64_t bits = (uint64_t)(uint32_t)w[0] | ((uint64_t)(uint32_t)w[1] << 32);
  double v;
  memcpy(&v, &bits, 8);
  return v;
}

/** `inferElementSymbol` (shared/rfd4p/encoding.js), on its own element table. */
inline std::string inferElement(const std::string& atomName, const std::string& resName = "") {
  auto upper = [](std::string v) {
    while (!v.empty() && isspace((unsigned char)v.front())) v.erase(v.begin());
    while (!v.empty() && isspace((unsigned char)v.back())) v.pop_back();
    for (char& c : v) c = (char)toupper((unsigned char)c);
    return v;
  };
  const std::string clean = upper(atomName), res = upper(resName);
  if (clean.empty() || clean == "X") return "C";
  if (clean == res && tables::INFER_ELEMENTS.count(clean)) return clean;
  size_t k = 0;
  while (k < clean.size() && isdigit((unsigned char)clean[k])) ++k;
  const std::string stripped = clean.substr(k);
  auto starts = [&](const char* p) { return stripped.compare(0, strlen(p), p) == 0; };
  if (starts("SE") && (res == "SEC" || res == "MSE" || res == "SEB")) return "SE";
  if (starts("CL")) return "CL";
  if (starts("BR")) return "BR";
  if (starts("AS") && (res == "CAS" || res == "CAF")) return "AS";
  if (!stripped.empty() && tables::INFER_ELEMENTS.count(stripped.substr(0, 1))) return stripped.substr(0, 1);
  return "C";
}

/** A residue token's own atom37 slot name (`token_atoms`), which is NOT one table for all. */
inline std::string tokenAtomName(int rt, int slot) {
  if (rt < 0 || rt >= tables::N_TOKENS || slot < 0 || slot >= 37) return std::string();
  return tables::TOKEN_ATOMS[rt][slot];
}

struct OutNaming {
  OutTables in;
  std::vector<int> resId;    // per token, the input's own
  std::vector<int> chain;    // per token, up to two characters packed little-endian
  // 🔴 AN UNINDEXED TOKEN IS NOT WRITTEN AT ALL. It is in-context conditioning that the
  // sampler never generated, and the reference emits only the INDEXED tokens - the unindexed
  // motif reaches the output by being grafted onto one of them. Writing them produced 40
  // spurious `UNK` residues on `motif_unidx_1prw`, which the fold's own count did not report
  // because it counts tokens with a known residue type and these have none.
  std::vector<uint8_t> emit;
  bool skip(int i) const { return !emit.empty() && i < (int)emit.size() && !emit[i]; }
  bool empty() const { return resId.empty(); }
};

/** A packed chain code back to its letters; `chainName(idx)` where there is none. */
inline std::string chainFromCode(int code) {
  std::string out;
  for (int k = 0; k < 2; ++k) {
    const char c = (char)((code >> (8 * k)) & 0xff);
    if (c) out += c;
  }
  return out;
}

/**
 * `annotate_unindexed_insertion`: where each unindexed motif token is grafted.
 *
 * 🔴 AN UNINDEXED MOTIF IS NEVER EMITTED, SO IT HAS TO BE PUT SOMEWHERE. It is in-context
 * conditioning - the sampler does not write it - so the design comes out without it, and the
 * motif's conditioned coordinates reach the output only by being TRANSFERRED onto a generated
 * token. Which token is the insertion head's Hungarian assignment, and a wrong one is a
 * plausible structure with the motif in the wrong place.
 */
struct Insertion {
  std::vector<int> target;   // per unindexed token, the generated token it maps onto, or -1
  std::vector<int> source;   // the unindexed tokens, in order
};

/**
 * `_get_target_indices_corresponding_to_inserted`: the motif's atoms onto the target's.
 *
 * 🔴 BY NAME FIRST, THEN BY ELEMENT AND DISTANCE - and it REFUSES rather than guessing. A name
 * that the target does not have falls through to a Hungarian over the unmatched pairs, where a
 * different element is an infinite cost; if any source atom is left unmatched, or the chosen
 * cell was one of those infinities, the whole mapping is rejected and the caller falls back.
 * Returning a partial match would graft half a residue.
 */
inline bool matchInsertedAtoms(const std::vector<std::string>& srcName,
                               const std::vector<std::string>& srcElem,
                               const std::vector<std::array<double, 3>>& srcPos,
                               const std::vector<std::string>& tgtName,
                               const std::vector<std::string>& tgtElem,
                               const std::vector<std::array<double, 3>>& tgtPos,
                               std::vector<int>& outIdx) {
  const size_t nSrc = srcName.size(), nTgt = tgtName.size();
  outIdx.assign(nSrc, -1);
  std::map<std::string, int> byName;
  // The LAST target of a name, as the reference's `Map.set` keeps.
  for (size_t t = 0; t < nTgt; ++t) byName[tgtName[t]] = (int)t;
  std::vector<uint8_t> used(nTgt, 0);
  std::vector<int> unmatchedSrc;
  for (size_t s = 0; s < nSrc; ++s) {
    auto it = byName.find(srcName[s]);
    if (it != byName.end() && !used[it->second]) { outIdx[s] = it->second; used[it->second] = 1; }
    else unmatchedSrc.push_back((int)s);
  }
  if (unmatchedSrc.empty()) return true;
  std::vector<int> unmatchedTgt;
  for (size_t t = 0; t < nTgt; ++t) if (!used[t]) unmatchedTgt.push_back((int)t);
  if (unmatchedSrc.size() > unmatchedTgt.size()) return false;
  std::vector<double> dist(unmatchedSrc.size() * unmatchedTgt.size());
  for (size_t r = 0; r < unmatchedSrc.size(); ++r) {
    for (size_t c = 0; c < unmatchedTgt.size(); ++c) {
      const int s = unmatchedSrc[r], tt = unmatchedTgt[c];
      if (srcElem[s] != tgtElem[tt]) {
        dist[r * unmatchedTgt.size() + c] = std::numeric_limits<double>::infinity();
      } else {
        // `Math.hypot`, V8's own (jsmath), as the reference measures it.
        dist[r * unmatchedTgt.size() + c] =
            lf::jsmath::hypot({srcPos[s][0] - tgtPos[tt][0], srcPos[s][1] - tgtPos[tt][1],
                               srcPos[s][2] - tgtPos[tt][2]});
      }
    }
  }
  const assign::Assignment a =
      assign::minimize(dist, (int)unmatchedSrc.size(), (int)unmatchedTgt.size());
  for (size_t k = 0; k < a.row.size(); ++k) {
    const size_t cell = (size_t)a.row[k] * unmatchedTgt.size() + a.col[k];
    if (!std::isfinite(dist[cell])) return false;
    outIdx[unmatchedSrc[a.row[k]]] = unmatchedTgt[a.col[k]];
  }
  for (size_t s = 0; s < nSrc; ++s) if (outIdx[s] < 0) return false;
  return true;
}

/**
 * `align_atom_array_to_input_frame`, with `insert_internal_coord_conditioned_groups` before it.
 *
 * 🔴 AND IT IS A 0.12 A REFINEMENT, NOT A RESCUE - measured, against the guess. The obvious
 * reading is that the sampler works in a centred frame and the output therefore arrives
 * somewhere arbitrary; it does not. The conditioning pins the coordinates DURING sampling, so
 * the generated structure is already in the conditioned frame and this corrects a median of
 * 0.118 A (max 0.133) on `ppi_pdl1` with its target conditioned. It is here because it is the
 * reference's and because the residual is real, not because it saves a misplaced design.
 *
 * 🔴 AND THE FRAME IT ALIGNS ONTO IS THE CENTRED ONE, NOT THE DEPOSITED ONE. `condition_
 * coordinate_1_atom` is shifted by `centerAndFeaturizeCentroid` along with the structure, so a
 * job whose input sits at (100, -50, 25) still comes out near the origin. Reading "input frame"
 * as "where the file put it" is what made the first test of this look like a 229 A failure.
 *
 * 🔴 AND AN INTERNAL-COORDINATE GROUP IS PLACED ONTO WHERE THE MODEL PUT IT, NOT THE OTHER WAY
 * ROUND: the group's own rigid geometry is the MOBILE set and the generated coordinates are
 * FIXED, so the ligand keeps its shape and takes the position the model chose. It writes into
 * the reference coordinates and NOT into the anchors - a group that is also coordinate
 * conditioned contributes its superposed geometry to the frame, and a group that is not
 * contributes nothing to it.
 */
/**
 * `buildPredictedInsertionMapping`: the model's own answer, from the insertion head.
 *
 * 🔴 THE HEAD IS ASKED ONLY ON THE LAST STEP, and that is the one whose logits mean anything:
 * a mapping captured at t = 0 is noise. `trunk.cuh` already passes `s == steps - 1`.
 *
 * 🔴 AND THE SCORE IS READ AT THE TOKEN'S OWN INDEX IN AN n x n MATRIX, not at its position in
 * the unindexed list: row `uPos`, column `iPos`, both absolute. Compacting either axis first
 * gives a matrix of the right shape and the wrong entries.
 */
inline Insertion predictedInsertionMapping(int n, const float* logits, const int* condIndex,
                                           const std::vector<uint8_t>& emitted,
                                           const std::vector<int>& chain) {
  Insertion out;
  if (!logits || !condIndex) return out;
  std::vector<int> unidx, idx;
  for (int tk = 0; tk < n; ++tk) {
    const bool indexed = condIndex[(size_t)tk * 37 + 1] != 0;
    if (!indexed) unidx.push_back(tk);
    else if (emitted.empty() || emitted[tk]) idx.push_back(tk);
  }
  if (unidx.empty() || idx.empty()) return out;
  out.source = unidx;
  out.target.assign(unidx.size(), -1);
  // 🔴 ONE ASSIGNMENT PER CHAIN, as upstream solves it: an unindexed token maps onto a distinct
  // generated token of its OWN chain, never across chains - chains in sorted order, and one
  // with no generated token left unmapped.
  auto chainOf = [&](int tk) {
    return tk < (int)chain.size() ? chainFromCode(chain[tk]) : std::string();
  };
  std::vector<std::string> chains;
  for (int tk : unidx) chains.push_back(chainOf(tk));
  std::sort(chains.begin(), chains.end());
  chains.erase(std::unique(chains.begin(), chains.end()), chains.end());
  for (const std::string& c : chains) {
    std::vector<int> rows, cols;   // rows index `unidx`, cols are tokens
    for (size_t r = 0; r < unidx.size(); ++r) if (chainOf(unidx[r]) == c) rows.push_back((int)r);
    for (int tk : idx) if (chainOf(tk) == c) cols.push_back(tk);
    if (cols.empty()) continue;
    std::vector<double> score(rows.size() * cols.size());
    for (size_t r = 0; r < rows.size(); ++r) {
      for (size_t k = 0; k < cols.size(); ++k) {
        score[r * cols.size() + k] = (double)logits[(size_t)unidx[rows[r]] * n + cols[k]];
      }
    }
    const assign::Assignment a = assign::maximize(score, (int)rows.size(), (int)cols.size());
    for (size_t k = 0; k < a.row.size(); ++k) out.target[rows[a.row[k]]] = cols[a.col[k]];
  }
  return out;
}

/**
 * The transfer: a mapped motif's conditioned coordinates onto the token it was grafted to.
 *
 * 🔴 IT WRITES INTO THE ANCHOR SET, WHICH IS THE WHOLE POINT. An unindexed token's conditioned
 * coordinates never reach the output otherwise - the token is not emitted - so without this the
 * alignment has no anchors from the motif at all and the design is placed by the rest of the
 * conditioning alone.
 */
inline void applyInsertion(const GenOut& g, int n, const Insertion& ins, const OutNaming& naming,
                           const int* resType, const float* coords, float* condCoord,
                           float* condIntCrd, int* coordGroup,
                           std::vector<uint8_t>& isInsertion, std::vector<int>* crdSource = nullptr,
                           bool dryRun = false, bool* allMatched = nullptr,
                           std::vector<int>* srcAt = nullptr) {
  if (!dryRun) isInsertion.assign((size_t)n * 37, 0);
  if (crdSource && !dryRun) crdSource->assign((size_t)n * 37, -1);
  if (srcAt && !dryRun) srcAt->assign((size_t)n * 37, -1);
  if (allMatched) *allMatched = true;
  if (ins.source.empty()) {
    if (allMatched) *allMatched = false;
    return;
  }
  const OutTables& in = naming.in;
  auto finite3 = [](const float* p, size_t at) {
    return p && std::isfinite(p[at * 3]) && std::isfinite(p[at * 3 + 1]) && std::isfinite(p[at * 3 + 2]);
  };
  // An input atom of token `tk` by NAME, as the page's `inByName` finds it: its slot in the
  // token's own layout, or -1.
  auto inputSlot = [&](int tk, const std::string& name) {
    if (!in.present() || name == "UNK" || name == "OXT" || name == "OP3") return -1;
    for (int s = 0; s < 37; ++s) {
      const int w = in.inName[(size_t)tk * 37 + s];
      if (w && unpackString(&w, 1) == name) return s;
    }
    return -1;
  };
  for (size_t k = 0; k < ins.source.size(); ++k) {
    const int u = ins.source[k], tgt = ins.target[k];
    if (tgt < 0) continue;
    std::vector<std::string> sn, se, tn, te;
    std::vector<std::array<double, 3>> sp, tp;
    std::vector<int> sSlot, tSlot;
    // 🔴 THE SOURCE IS THE MOTIF AS THE INPUT HAD IT - its own atoms, names and elements, where
    // this took the GLOBAL atom37 slot name for every residue and the first letter as the
    // element. For a protein motif the two agree; for a nucleotide C1' was matched as `CA`.
    for (int s = 0; s < 37; ++s) {
      const size_t at = (size_t)u * 37 + s;
      std::string nm, el;
      if (in.present()) {
        if (!in.inName[at]) continue;
        nm = unpackString(&in.inName[at], 1);
        el = in.inElem[at] ? unpackString(&in.inElem[at], 1) : std::string();
        if (el.empty()) el = inferElement(nm);
      } else {
        if (!g.atomMask[at] && !finite3(condCoord, at)) continue;
        nm = tokenAtomName(resType ? resType[u] : -1, s);
        el = inferElement(nm);
      }
      // 🔴 `OXT` AND `OP3` ARE DROPPED FROM THE SOURCE, as the reference drops them: a leaving
      // atom has no counterpart in a generated residue and would make every match fail.
      if (nm.empty() || nm == "OXT" || nm == "OP3") continue;
      sn.push_back(nm);
      se.push_back(el);
      std::array<double, 3> pos{0, 0, 0};
      const float* src = finite3(coords, at) ? coords : finite3(condCoord, at) ? condCoord : nullptr;
      if (src) for (int c = 0; c < 3; ++c) pos[c] = (double)src[at * 3 + c];
      sp.push_back(pos);
      sSlot.push_back(s);
    }
    // The target in its records' order, named by ITS OWN residue's layout.
    const int rt = g.residueType[tgt];
    if (rt >= 0 && rt < tables::N_TOKENS) {
      for (const int s : tables::TOKEN_ATOM_ORDER()[rt]) {
        const size_t at = (size_t)tgt * 37 + s;
        if (!g.atomMask[at]) continue;
        const std::string nm = tokenAtomName(rt, s);
        if (nm.empty()) continue;
        const int is = inputSlot(tgt, nm);
        std::string el = is >= 0 && in.inElem[(size_t)tgt * 37 + is]
                             ? unpackString(&in.inElem[(size_t)tgt * 37 + is], 1) : std::string();
        if (el.empty() || el == "X") el = inferElement(nm);
        tn.push_back(nm);
        te.push_back(el);
        tp.push_back({(double)g.coors[(size_t)tgt * 111 + s * 3],
                      (double)g.coors[(size_t)tgt * 111 + s * 3 + 1],
                      (double)g.coors[(size_t)tgt * 111 + s * 3 + 2]});
        tSlot.push_back(s);
      }
    }
    std::vector<int> matched;
    if (sn.empty() || !matchInsertedAtoms(sn, se, sp, tn, te, tp, matched)) {
      // 🔴 ONE PAIR THAT CANNOT BE MATCHED REJECTS THE WHOLE MAPPING, which is what makes the
      // fallback a fallback: the reference validates every pair before applying any, and a
      // half-applied mapping would graft some motifs and silently drop others.
      if (allMatched) *allMatched = false;
      continue;
    }
    if (dryRun) continue;
    for (int s : tSlot) isInsertion[(size_t)tgt * 37 + s] = 1;
    for (size_t m = 0; m < matched.size(); ++m) {
      const size_t from = (size_t)u * 37 + sSlot[m];
      const size_t to = (size_t)tgt * 37 + tSlot[matched[m]];
      if (coordGroup && coordGroup[from] != -1) coordGroup[to] = coordGroup[from];
      const bool crd = finite3(condCoord, from), intCrd = finite3(condIntCrd, from);
      if (crd) for (int c = 0; c < 3; ++c) condCoord[to * 3 + c] = condCoord[from * 3 + c];
      if (intCrd) for (int c = 0; c < 3; ++c) condIntCrd[to * 3 + c] = condIntCrd[from * 3 + c];
      if (crdSource && (crd || intCrd)) (*crdSource)[to] = u;
      if (srcAt) (*srcAt)[to] = (int)from;
    }
  }
}

/**
 * The model's mapping, if every pair of it can be grafted; otherwise none.
 *
 * 🔴 ALL OR NOTHING, AND NOTHING IN ITS PLACE. Upstream builds the head's mapping, and
 * `annotate_unindexed_insertion` either applies all of it or - one pair whose atoms cannot be
 * matched - none, warning and writing the design without the motif's annotations. Both ports
 * used to substitute a GEOMETRIC mapping of their own at that point, which upstream has never
 * had: a structure upstream writes without a grafted motif came out here with one, from a rule
 * nobody had checked against anything. Validated before anything is written, so a motif is
 * never grafted half by one pass and half by another.
 */
inline Insertion resolveInsertion(const GenOut& g, int n, const float* insertionLogits,
                                  const int* condIndex, const int* resType, const float* coords,
                                  const float* condCoord, const float* condIntCrd,
                                  const std::vector<uint8_t>& emitted, const OutNaming& naming) {
  std::vector<uint8_t> scratchFlags;
  const Insertion predicted =
      predictedInsertionMapping(n, insertionLogits, condIndex, emitted, naming.chain);
  if (predicted.source.empty()) return predicted;
  bool ok = false;
  applyInsertion(g, n, predicted, naming, resType, coords, const_cast<float*>(condCoord),
                 const_cast<float*>(condIntCrd), nullptr, scratchFlags, nullptr, true, &ok);
  return ok ? predicted : Insertion{};
}

inline void alignToInputFrame(GenOut& g, int n, const float* condCoord, const float* condIntCrd,
                              const int* coordGroup, const OutNaming& naming) {
  if (!condCoord) return;
  const size_t slots = (size_t)n * 37;
  // 🔴 ONLY WHAT IS WRITTEN TAKES PART. The reference builds its records first and aligns those,
  // so an UNINDEXED motif token - never written, its coordinates grafted elsewhere - is no anchor.
  // Its conditioned coordinates were anchors here, on every unindexed job, and moved the whole
  // design by a few tenths of an angstrom against the page's.
  auto written = [&](size_t s) { return g.atomMask[s] && !naming.skip((int)(s / 37)); };
  // The reference coordinates: the conditioned position where there is one, else the generated.
  std::vector<double> cwi(slots * 3, 0.0);
  std::vector<int> anchors;
  for (size_t s = 0; s < slots; ++s) {
    if (!written(s)) continue;
    const bool has = std::isfinite(condCoord[s * 3]) && std::isfinite(condCoord[s * 3 + 1]) &&
                     std::isfinite(condCoord[s * 3 + 2]);
    for (int k = 0; k < 3; ++k) {
      cwi[s * 3 + k] = has ? (double)condCoord[s * 3 + k]
                           : (double)g.coors[(s / 37) * 111 + (s % 37) * 3 + k];
    }
    if (has) anchors.push_back((int)s);
  }
  if (condIntCrd && coordGroup) {
    std::map<int, std::vector<int>> byGroup;
    for (size_t s = 0; s < slots; ++s) {
      if (!written(s) || coordGroup[s] <= 0) continue;
      if (!std::isfinite(condIntCrd[s * 3]) || !std::isfinite(condIntCrd[s * 3 + 1]) ||
          !std::isfinite(condIntCrd[s * 3 + 2])) {
        continue;
      }
      byGroup[coordGroup[s]].push_back((int)s);
    }
    for (const auto& [gid, members] : byGroup) {
      std::vector<double> fixed, mobile;
      for (const int s : members) {
        for (int k = 0; k < 3; ++k) {
          fixed.push_back((double)g.coors[(size_t)(s / 37) * 111 + (s % 37) * 3 + k]);
          mobile.push_back((double)condIntCrd[(size_t)s * 3 + k]);
        }
      }
      const superpose::Fit f = superpose::superimpose(fixed, mobile);
      if (!f.ok) continue;
      for (size_t k = 0; k < members.size(); ++k) {
        const double m[3] = {mobile[k * 3] - f.mobileCenter[0], mobile[k * 3 + 1] - f.mobileCenter[1],
                             mobile[k * 3 + 2] - f.mobileCenter[2]};
        for (int r = 0; r < 3; ++r) {
          cwi[(size_t)members[k] * 3 + r] = (float)(m[0] * f.R[r * 3] + m[1] * f.R[r * 3 + 1] +
                                                    m[2] * f.R[r * 3 + 2] + f.fixedCenter[r]);
        }
      }
    }
  }
  // Three anchors or none, as upstream's `anchor_mask.sum() < 3` (output.js has the note).
  if (anchors.size() < 3) return;
  std::vector<double> fixed, mobile;
  for (const int s : anchors) {
    for (int k = 0; k < 3; ++k) {
      fixed.push_back(cwi[(size_t)s * 3 + k]);
      mobile.push_back((double)g.coors[(size_t)(s / 37) * 111 + (s % 37) * 3 + k]);
    }
  }
  const superpose::Fit f = superpose::superimpose(fixed, mobile);
  if (!f.ok) return;
  // 🔴 ROUNDED TO FLOAT AT EACH COMPONENT, as the reference's `Math.fround` is - the writer
  // prints three decimals and a double carried to the print rounds differently in the last place.
  for (size_t s = 0; s < slots; ++s) {
    if (!g.atomMask[s]) continue;
    float* c = &g.coors[(size_t)(s / 37) * 111 + (s % 37) * 3];
    const double m[3] = {(double)c[0] - f.mobileCenter[0], (double)c[1] - f.mobileCenter[1],
                         (double)c[2] - f.mobileCenter[2]};
    for (int r = 0; r < 3; ++r) {
      c[r] = (float)(m[0] * f.R[r * 3] + m[1] * f.R[r * 3 + 1] + m[2] * f.R[r * 3 + 2] +
                     f.fixedCenter[r]);
    }
  }
}

/** A chain index as mmCIF/PDB names them: A..Z, then AA, BB, ... as the reference's own does. */
inline std::string chainName(int idx) {
  if (idx < 26) return std::string(1, (char)('A' + idx));
  const int k = idx - 26;
  return std::string(k / 26 + 2, (char)('A' + k % 26));
}

// ---------------------------------------------------------------- the records
//
// `processSampleToAtoms` (shared/rfd4p/output.js) to the letter, so the two writers agree as TEXT
// - cuda/rfd4p/check-writer.mjs holds this file to the page's on the same sample.

struct AtomRecord {
  bool het = false, unindexedInsertion = false;
  std::string atomName, resName, chain, insCode, element, srcComp = ".";
  int resId = 1, entity = 0, charge = 0;
  double b = NAN, occ = 1.0;
  float x = 0, y = 0, z = 0;
  // What the design metrics read the input through (metrics.h; `inputIndex`, `condSource` and
  // `tokenInputIndex` in shared/rfd4p/output.js): the token, the input atom this one is (token *
  // 37 + its input slot, or -1 where the generated residue added it), the input atom its 1-body
  // conditions come from (the motif atom a grafted one was grafted from, else `inAt`), the token
  // its residue-level ones come from, and its coordinate conditioning.
  int token = -1, inAt = -1, condAt = -1, resTok = -1;
  bool atomized = false, crd = false, internal = false;
  // ...and for the motif RMSDs: whether the atom was grafted, its live coordinate group, the
  // input atoms its C_CRD and C_INTCRD values come from (-1 for none), and its own residue's
  // source component (`srcComp` above is the coordinate-conditioned region's, a grafted motif's).
  bool grafted = false;
  int group = -1, crdAt = -1, intAt = -1;
  std::string ownSrc = ".";
};

inline std::vector<AtomRecord> buildRecords(const GenOut& g, int n, const std::vector<int>& chainOf,
                                            const OutNaming& naming, const float* condCoord,
                                            const float* condIntCrd,
                                            const std::vector<uint8_t>& isInsertion,
                                            const std::vector<int>& crdSource,
                                            const std::vector<int>& srcAt = {},
                                            const int* coordGroup = nullptr) {
  const OutTables& in = naming.in;
  auto group = [&](int at) { return coordGroup && at >= 0 ? coordGroup[at] : -1; };
  std::vector<AtomRecord> out;
  auto finite3 = [](const float* p, size_t at) {
    return p && std::isfinite(p[at * 3]) && std::isfinite(p[at * 3 + 1]) && std::isfinite(p[at * 3 + 2]);
  };
  auto srcCompOf = [&](int tk) {
    return in.present() ? unpackString(&in.srcComp[(size_t)tk * 4], 4) : std::string(".");
  };
  for (int i = 0; i < n; ++i) {
    if (naming.skip(i)) continue;
    AtomRecord base;
    base.chain = i < (int)naming.chain.size() ? chainFromCode(naming.chain[i])
                                              : chainName(i < (int)chainOf.size() ? chainOf[i] : 0);
    if (base.chain.empty()) base.chain = "A";
    base.resId = i < (int)naming.resId.size() ? naming.resId[i] : i + 1;
    if (in.present()) {
      base.insCode = unpackString(&in.insCode[i], 1);
      base.entity = in.entity[i];
    }
    auto place = [&](AtomRecord& r, int slot) {
      r.x = g.coors[(size_t)i * 111 + slot * 3];
      r.y = g.coors[(size_t)i * 111 + slot * 3 + 1];
      r.z = g.coors[(size_t)i * 111 + slot * 3 + 2];
      if (!isInsertion.empty() && isInsertion[(size_t)i * 37 + slot]) r.unindexedInsertion = true;
    };
    // 🔴 AN ATOMISED TOKEN IS ONE ATOM, AND IT IS THE INPUT'S: its own name, residue, element and
    // HETATM flag, at the token's sampled position. This wrote `UNK CA`, element C, `ATOM`.
    if (in.present() && in.atomize[i]) {
      const size_t at = (size_t)i * 37 + 1;
      AtomRecord r = base;
      r.atomName = unpackString(&in.inName[at], 1);
      r.resName = unpackString(&in.resName[(size_t)i * 2], 2);
      const std::string el = in.inElem[at] ? unpackString(&in.inElem[at], 1) : std::string();
      r.element = el.empty() ? inferElement(r.atomName) : el;
      r.het = in.hetero[i] != 0;
      r.b = unpackDouble(&in.inB[at * 2]);
      r.occ = 1.0;
      r.charge = in.inCharge[at];
      if (finite3(condCoord, at) || finite3(condIntCrd, at)) r.srcComp = srcCompOf(i);
      r.token = r.resTok = i;
      r.inAt = r.condAt = (int)at;
      r.atomized = true;
      r.crd = finite3(condCoord, at);
      r.internal = group((int)at) != -1 && group((int)at) != 0;
      r.group = group((int)at);
      r.crdAt = finite3(condCoord, at) ? (int)at : -1;
      r.intAt = finite3(condIntCrd, at) ? (int)at : -1;
      r.ownSrc = srcCompOf(i);
      place(r, 1);
      out.push_back(r);
      continue;
    }
    const int rt = g.residueType[i];
    if (rt < 0 || rt >= tables::N_TOKENS) continue;
    const std::string inRes = in.present() ? unpackString(&in.resName[(size_t)i * 2], 2) : "<M>";
    const bool wasMasked = inRes == "<M>" || inRes == "UNK";
    const std::string srcResidue = wasMasked ? "." : srcCompOf(i);
    for (const int slot : tables::TOKEN_ATOM_ORDER()[rt]) {
      const size_t at = (size_t)i * 37 + slot;
      if (!g.atomMask[at]) continue;
      AtomRecord r = base;
      r.atomName = tokenAtomName(rt, slot);
      r.resName = tables::TOKENS[rt];
      int is = -1;
      if (in.present() && r.atomName != "UNK" && r.atomName != "OXT" && r.atomName != "OP3") {
        for (int s2 = 0; s2 < 37; ++s2) {
          const int w = in.inName[(size_t)i * 37 + s2];
          if (w && unpackString(&w, 1) == r.atomName) { is = s2; break; }
        }
      }
      const size_t ia = (size_t)i * 37 + (is < 0 ? 0 : is);
      const bool crd = is >= 0 && (finite3(condCoord, ia) || finite3(condIntCrd, ia));
      r.het = wasMasked ? false : (in.present() && in.hetero[i] != 0);
      if (is >= 0) {
        r.b = unpackDouble(&in.inB[ia * 2]);
        const double occ = unpackDouble(&in.inOcc[ia * 2]);
        r.occ = !wasMasked && occ > 0 ? occ : 1.0;
        r.charge = in.inCharge[ia];
      }
      const std::string el = is >= 0 && in.inElem[ia] ? unpackString(&in.inElem[ia], 1) : std::string();
      r.element = !el.empty() && el != "X" ? el : inferElement(r.atomName);
      if (crd) r.srcComp = srcResidue;
      // A grafted motif atom names the motif it came from.
      if (!crdSource.empty() && crdSource[at] >= 0) r.srcComp = srcCompOf(crdSource[at]);
      const int from = srcAt.empty() ? -1 : srcAt[at];
      r.token = i;
      r.inAt = is >= 0 ? (int)ia : -1;
      r.condAt = from >= 0 ? from : r.inAt;
      r.resTok = from >= 0 ? from / 37 : i;
      r.crd = (is >= 0 && finite3(condCoord, ia)) || (from >= 0 && finite3(condCoord, from));
      const int cg = from >= 0 && group(from) != -1 ? group(from) : group(r.inAt);
      r.internal = cg != -1 && cg != 0;
      r.grafted = from >= 0;
      r.group = cg;
      r.crdAt = from >= 0 && finite3(condCoord, from) ? from : (is >= 0 && finite3(condCoord, ia) ? (int)ia : -1);
      r.intAt = from >= 0 && finite3(condIntCrd, from) ? from : (is >= 0 && finite3(condIntCrd, ia) ? (int)ia : -1);
      r.ownSrc = srcResidue;
      place(r, slot);
      out.push_back(r);
    }
  }
  return out;
}

/** JavaScript's `Number.prototype.toFixed`: the nearest decimal, a tie going UP, on the exact value. */
inline std::string jsToFixed(double x, int digits) {
  if (!std::isfinite(x)) return x != x ? "NaN" : (x > 0 ? "Infinity" : "-Infinity");
  char buf[512];
  snprintf(buf, sizeof buf, "%.60f", std::fabs(x));   // the binary value, exactly
  std::string s = buf;
  const size_t dot = s.find('.');
  std::string intPart = s.substr(0, dot), frac = s.substr(dot + 1);
  const bool up = frac.size() > (size_t)digits && frac[digits] >= '5';
  std::string kept = intPart + frac.substr(0, digits);
  if (up) {
    int k = (int)kept.size() - 1;
    while (k >= 0 && kept[k] == '9') kept[k--] = '0';
    if (k < 0) kept.insert(kept.begin(), '1');
    else ++kept[k];
  }
  const std::string ip = kept.substr(0, kept.size() - digits), fp = kept.substr(kept.size() - digits);
  // The sign of a NEGATIVE value survives rounding to zero - `(-0.0003).toFixed(3)` is "-0.000" -
  // where negative zero itself prints "0.000".
  std::string out = (x < 0 ? "-" : "") + ip;
  if (digits > 0) out += "." + fp;
  return out;
}

/** JavaScript's `String(number)` for a finite double: the shortest form that round-trips. */
inline std::string jsNumber(double x) {
  if (x == 0) return "0";
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof buf, x);
  return std::string(buf, r.ptr);
}

/** `_atom_site.label_atom_id`: a name with a quote or a space in it is quoted. */
inline std::string cifAtomName(const std::string& name) {
  if (name.empty()) return ".";
  if (name.find('\'') != std::string::npos || name.find_first_of(" \t") != std::string::npos) {
    return "\"" + name + "\"";
  }
  return name;
}

/**
 * `Cartn_x` and its two neighbours: the float, scaled, rounded half to even on an exact half
 * (`formatCifCoord`, which is NumPy's and Biotite's) and printed with `toFixed(3)`.
 */
inline std::string cifCoord(float v) {
  const double scaled = (double)v * 1000.0;
  const double fl = std::floor(scaled);
  const double diff = scaled - fl;
  double rounded;
  if (std::fabs(diff - 0.5) < 1e-6) rounded = std::fmod(fl, 2.0) == 0 ? fl : fl + 1;
  else rounded = std::floor(scaled + 0.5);
  return jsToFixed(rounded / 1000.0, 3);
}

/** The records as mmCIF (`toCifString`), which is the format RFD4-Proteina itself writes. */
inline std::string toCif(const std::vector<AtomRecord>& atoms) {
  static const char* COLUMNS[] = {
      "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
      "label_asym_id", "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "auth_seq_id",
      "auth_comp_id", "auth_asym_id", "auth_atom_id", "B_iso_or_equiv", "occupancy",
      "pdbx_formal_charge", "is_unindexed_insertion",
      "annotation_src-coordinate-conditioned-region_1_atom", "Cartn_x", "Cartn_y", "Cartn_z",
      "pdbx_PDB_model_num"};
  std::string out = "data_rfd4p_design\n#\nloop_\n";
  for (const char* c : COLUMNS) out += std::string("_atom_site.") + c + "\n";
  for (size_t i = 0; i < atoms.size(); ++i) {
    const AtomRecord& a = atoms[i];
    const std::string atomId = cifAtomName(a.atomName);
    const std::string comp = a.resName.empty() ? "UNK" : a.resName;
    const std::string asym = a.chain.empty() ? "A" : a.chain;
    const std::string charge =
        a.charge != 0 ? (a.charge > 0 ? "+" + std::to_string(a.charge) : std::to_string(a.charge)) : "?";
    out += (a.het ? "HETATM " : "ATOM ") + std::to_string(i + 1) + " " +
           (a.element.empty() ? inferElement(a.atomName) : a.element) + " " + atomId + " . " +
           comp + " " + asym + " " + std::to_string(a.entity) + " " + std::to_string(a.resId) +
           " " + (a.insCode.empty() ? "." : a.insCode) + " " + std::to_string(a.resId) + " " +
           comp + " " + asym + " " + atomId + " " + (std::isfinite(a.b) ? jsNumber(a.b) : "nan") +
           " " + (std::isfinite(a.occ) ? jsToFixed(a.occ, 1) : "1.0") + " " + charge + " " +
           (a.unindexedInsertion ? "True" : "False") + " " + (a.srcComp.empty() ? "." : a.srcComp) +
           " " + cifCoord(a.x) + " " + cifCoord(a.y) + " " + cifCoord(a.z) + " 1\n";
  }
  out += "#\n";
  return out;
}

/** `formatPdbAtomField`: columns 13-16, a one-letter element's name starting one column in. */
inline std::string pdbAtomField(const std::string& rawName, const std::string& element) {
  std::string name = rawName.empty() ? "CA" : rawName;
  if (name.size() >= 4) return name.substr(0, 4);
  std::string el = element.empty() ? inferElement(name) : element;
  for (char& c : el) c = (char)toupper((unsigned char)c);
  auto pad = [](std::string v, size_t w) { while (v.size() < w) v += ' '; return v; };
  if (el.size() == 1 && name.compare(0, 1, el) == 0) return " " + pad(name, 3);
  return pad(name, 4);
}

/** The records as PDB (`toPdbString`), TER between chains and after the last. */
inline std::string toPdb(const std::vector<AtomRecord>& atoms) {
  auto padStart = [](std::string v, size_t w) { while (v.size() < w) v = " " + v; return v; };
  std::string out = "HEADER    DE NOVO DESIGN                          01-JAN-26   RFD4\n"
                    "TITLE     RFD4-Proteina Design\n";
  std::string prevChain;
  bool havePrev = false, prevHet = false;
  for (size_t i = 0; i < atoms.size(); ++i) {
    const AtomRecord& a = atoms[i];
    const std::string chain = (a.chain.empty() ? std::string("A") : a.chain).substr(0, 1);
    if (havePrev && prevChain != chain && !prevHet) out += "TER\n";
    prevChain = chain;
    prevHet = a.het;
    havePrev = true;
    std::string el = a.element.empty() ? inferElement(a.atomName) : a.element;
    for (char& c : el) c = (char)toupper((unsigned char)c);
    std::string res = (a.resName.empty() ? std::string("UNK") : a.resName).substr(0, 3);
    std::string seq = std::to_string(a.resId);
    if (seq.size() > 4) seq = seq.substr(seq.size() - 4);
    const std::string ins = a.insCode.empty() ? " " : a.insCode.substr(0, 1);
    out += std::string(a.het ? "HETATM" : "ATOM  ") + padStart(std::to_string((i + 1) % 100000), 5) +
           " " + pdbAtomField(a.atomName, el) + " " + padStart(res, 3) + " " + chain +
           padStart(seq, 4) + ins + "   " + padStart(jsToFixed(a.x, 3), 8) +
           padStart(jsToFixed(a.y, 3), 8) + padStart(jsToFixed(a.z, 3), 8) +
           padStart(jsToFixed(std::isfinite(a.occ) ? a.occ : 1.0, 2), 6) +
           padStart(jsToFixed(std::isfinite(a.b) ? a.b : 0.0, 2), 6) + "          " +
           padStart(el.substr(0, 2), 2) + "  \n";
  }
  out += "TER\nEND\n";
  return out;
}

}  // namespace rfd4p
