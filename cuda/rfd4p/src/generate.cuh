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
#include <array>
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
struct OutNaming {
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
 * token. Which token is a Hungarian assignment, and a wrong one is a plausible structure with
 * the motif in the wrong place.
 *
 * 🔴 AND THE SCORE IS NEGATIVE SQUARED DISTANCE WITH A 1e4 PENALTY FOR A RESIDUE MISMATCH, so
 * an identity disagreement costs more than any distance a structure can produce - the mapping
 * prefers a far ALA to a near GLY, which is the point: it is placing a known motif, not finding
 * the nearest atom.
 */
struct Insertion {
  std::vector<int> target;   // per unindexed token, the generated token it maps onto, or -1
  std::vector<int> source;   // the unindexed tokens, in order
};

inline Insertion geometricInsertionMapping(const GenOut& g, int n, const int* condIndex,
                                           const int* resType, const float* coords,
                                           const float* condCoord, const float* condIntCrd,
                                           const std::vector<uint8_t>& emitted) {
  Insertion out;
  std::vector<int> unidx, idx;
  for (int tk = 0; tk < n; ++tk) {
    // A token is unindexed when its live slots are; `condition_index` is per atom and uniform
    // over a token, so slot 1 (the CA, always present) decides.
    const bool indexed = condIndex ? condIndex[(size_t)tk * 37 + 1] != 0 : true;
    if (!indexed) unidx.push_back(tk);
    else if (emitted.empty() || emitted[tk]) idx.push_back(tk);
  }
  if (unidx.empty() || idx.empty()) return out;

  std::vector<double> score((size_t)unidx.size() * idx.size());
  for (size_t r = 0; r < unidx.size(); ++r) {
    const int u = unidx[r];
    const size_t ca = (size_t)u * 37 + 1;
    // 🔴 THE REFERENCE COORDINATE FALLS BACK IN A FIXED ORDER: `coord_to_be_noised`, then the
    // conditioned coordinate, then the internal one, then the origin. Each is a different
    // frame, and taking whichever is finite FIRST is what the reference does.
    double ref[3] = {0, 0, 0};
    const float* src = nullptr;
    if (coords && std::isfinite(coords[ca * 3])) src = coords;
    else if (condCoord && std::isfinite(condCoord[ca * 3])) src = condCoord;
    else if (condIntCrd && std::isfinite(condIntCrd[ca * 3])) src = condIntCrd;
    if (src) for (int k = 0; k < 3; ++k) ref[k] = (double)src[ca * 3 + k];
    const int uType = resType ? resType[u] : -1;
    for (size_t c = 0; c < idx.size(); ++c) {
      const int i = idx[c];
      const float* gca = &g.coors[(size_t)i * 111 + 1 * 3];
      double d2 = 0;
      for (int k = 0; k < 3; ++k) { const double dd = ref[k] - (double)gca[k]; d2 += dd * dd; }
      const int iType = g.residueType[i];
      const double penalty = (uType >= 0 && uType == iType) ? 0.0 : 1e4;
      score[r * idx.size() + c] = -(d2 + penalty);
    }
  }
  const assign::Assignment a = assign::maximize(score, (int)unidx.size(), (int)idx.size());
  out.source = unidx;
  out.target.assign(unidx.size(), -1);
  for (size_t k = 0; k < a.row.size(); ++k) out.target[a.row[k]] = idx[a.col[k]];
  return out;
}

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
  for (size_t t = 0; t < nTgt; ++t) byName.emplace(tgtName[t], (int)t);
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
        double d = 0;
        for (int k = 0; k < 3; ++k) {
          const double dd = srcPos[s][k] - tgtPos[tt][k];
          d += dd * dd;
        }
        dist[r * unmatchedTgt.size() + c] = std::sqrt(d);
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
 * The transfer: a mapped motif's conditioned coordinates onto the token it was grafted to.
 *
 * 🔴 IT WRITES INTO THE ANCHOR SET, WHICH IS THE WHOLE POINT. An unindexed token's conditioned
 * coordinates never reach the output otherwise - the token is not emitted - so without this the
 * alignment has no anchors from the motif at all and the design is placed by the rest of the
 * conditioning alone.
 */
inline void applyInsertion(const GenOut& g, int n, const Insertion& ins,
                           const std::vector<std::string>& atomNames, float* condCoord,
                           float* condIntCrd, int* coordGroup,
                           std::vector<uint8_t>& isInsertion) {
  isInsertion.assign((size_t)n * 37, 0);
  if (ins.source.empty()) return;
  // 🔴 SLOT TO NAME IS RESIDUE-INDEPENDENT IN THIS PORT, which is what the writer above already
  // assumes: `atomNames` is one 37-long list read from the model directory and every residue
  // indexes it the same way. The reference carries a per-residue slot map; if the two ever
  // disagree this is where it shows, and the writer would be wrong in the same place.
  auto nameOf = [&](int, int slot) -> std::string {
    return slot < (int)atomNames.size() ? atomNames[slot] : std::string();
  };
  for (size_t k = 0; k < ins.source.size(); ++k) {
    const int u = ins.source[k], tgt = ins.target[k];
    if (tgt < 0) continue;
    std::vector<std::string> sn, se, tn, te;
    std::vector<std::array<double, 3>> sp, tp;
    std::vector<int> sSlot, tSlot;
    for (int s = 0; s < 37; ++s) {
      const size_t at = (size_t)u * 37 + s;
      const std::string nm = nameOf(0, s);
      // 🔴 `OXT` AND `OP3` ARE DROPPED FROM THE SOURCE, as the reference drops them: a leaving
      // atom has no counterpart in a generated residue and would make every match fail.
      if (nm.empty() || nm == "OXT" || nm == "OP3") continue;
      if (!g.atomMask[at] && !(condCoord && std::isfinite(condCoord[at * 3]))) continue;
      sn.push_back(nm);
      se.push_back(nm.substr(0, 1));
      std::array<double, 3> pos{0, 0, 0};
      const float* src = condCoord && std::isfinite(condCoord[at * 3]) ? condCoord : nullptr;
      if (src) for (int c = 0; c < 3; ++c) pos[c] = (double)src[at * 3 + c];
      sp.push_back(pos);
      sSlot.push_back(s);
    }
    for (int s = 0; s < 37; ++s) {
      const size_t at = (size_t)tgt * 37 + s;
      if (!g.atomMask[at]) continue;
      const std::string nm = nameOf(0, s);
      if (nm.empty()) continue;
      tn.push_back(nm);
      te.push_back(nm.substr(0, 1));
      tp.push_back({(double)g.coors[(size_t)tgt * 111 + s * 3],
                    (double)g.coors[(size_t)tgt * 111 + s * 3 + 1],
                    (double)g.coors[(size_t)tgt * 111 + s * 3 + 2]});
      tSlot.push_back(s);
    }
    std::vector<int> matched;
    if (sn.empty() || !matchInsertedAtoms(sn, se, sp, tn, te, tp, matched)) continue;
    for (int s : tSlot) isInsertion[(size_t)tgt * 37 + s] = 1;
    for (size_t m = 0; m < matched.size(); ++m) {
      const size_t from = (size_t)u * 37 + sSlot[m];
      const size_t to = (size_t)tgt * 37 + tSlot[matched[m]];
      if (coordGroup && coordGroup[from] != -1) coordGroup[to] = coordGroup[from];
      if (condCoord && std::isfinite(condCoord[from * 3])) {
        for (int c = 0; c < 3; ++c) condCoord[to * 3 + c] = condCoord[from * 3 + c];
      }
      if (condIntCrd && std::isfinite(condIntCrd[from * 3])) {
        for (int c = 0; c < 3; ++c) condIntCrd[to * 3 + c] = condIntCrd[from * 3 + c];
      }
    }
  }
}

inline void alignToInputFrame(GenOut& g, int n, const float* condCoord, const float* condIntCrd,
                              const int* coordGroup) {
  if (!condCoord) return;
  const size_t slots = (size_t)n * 37;
  // The reference coordinates: the conditioned position where there is one, else the generated.
  std::vector<double> cwi(slots * 3, 0.0);
  std::vector<int> anchors;
  for (size_t s = 0; s < slots; ++s) {
    if (!g.atomMask[s]) continue;
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
      if (!g.atomMask[s] || coordGroup[s] <= 0) continue;
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
  if (anchors.empty()) return;
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

/** `_atom_site.label_atom_id`: a name with a quote in it (`O5'`) is quoted. */
inline std::string cifAtomName(const std::string& name) {
  if (name.empty()) return ".";
  if (name.find('\'') != std::string::npos || name.find(' ') != std::string::npos) {
    return "\"" + name + "\"";
  }
  return name;
}

/**
 * `Cartn_x` and its two neighbours, to three decimals ROUND-HALF-TO-EVEN - which is NumPy's and
 * Biotite's, and so the reference's. `%.3f` is round-half-away-from-zero and differs on an exact
 * half, which a coordinate hits often enough to notice when two files are diffed.
 */
inline std::string cifCoord(float v) {
  const double scaled = (double)v * 1000.0;
  const double fl = std::floor(scaled);
  const double diff = scaled - fl;
  double rounded;
  if (std::fabs(diff - 0.5) < 1e-6) rounded = std::fmod(fl, 2.0) == 0 ? fl : fl + 1;
  else rounded = std::floor(scaled + 0.5);
  char buf[32];
  snprintf(buf, sizeof buf, "%.3f", rounded / 1000.0);
  return buf;
}

/**
 * The structure as mmCIF (`toCifString`), which is the format RFD4-Proteina itself writes.
 * `atomNames` is 37 long, `resNames` 147, `chainOf` one chain index a token.
 */
inline std::string toCif(const GenOut& g, int n, const std::vector<std::string>& atomNames,
                         const std::vector<std::string>& resNames,
                         const std::vector<int>& chainOf, const OutNaming& naming = {}) {
  static const char* COLUMNS[] = {
      "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
      "label_asym_id", "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "auth_seq_id",
      "auth_comp_id", "auth_asym_id", "auth_atom_id", "B_iso_or_equiv", "occupancy",
      "pdbx_formal_charge", "is_unindexed_insertion",
      "annotation_src-coordinate-conditioned-region_1_atom", "Cartn_x", "Cartn_y", "Cartn_z",
      "pdbx_PDB_model_num"};
  std::string out = "data_rfd4p_design\n#\nloop_\n";
  for (const char* c : COLUMNS) out += std::string("_atom_site.") + c + "\n";
  int serial = 1;
  char line[512];
  for (int i = 0; i < n; ++i) {
    if (naming.skip(i)) continue;
    const int rt = g.residueType[i];
    const std::string res = rt >= 0 && rt < (int)resNames.size() ? resNames[rt] : "UNK";
    const std::string chain = i < (int)naming.chain.size()
                                  ? chainFromCode(naming.chain[i])
                                  : chainName(i < (int)chainOf.size() ? chainOf[i] : 0);
    const int resId = i < (int)naming.resId.size() ? naming.resId[i] : i + 1;
    for (int a = 0; a < 37; ++a) {
      if (!g.atomMask[(size_t)i * 37 + a]) continue;
      const std::string name = cifAtomName(atomNames[a]);
      snprintf(line, sizeof line, "ATOM %d %s %s . %s %s 0 %d . %d %s %s %s nan 1.0 ? False . %s %s %s 1\n",
               serial++, atomNames[a].substr(0, 1).c_str(), name.c_str(), res.c_str(),
               chain.c_str(), resId, resId, res.c_str(), chain.c_str(), name.c_str(),
               cifCoord(g.coors[(size_t)i * 111 + a * 3]).c_str(),
               cifCoord(g.coors[(size_t)i * 111 + a * 3 + 1]).c_str(),
               cifCoord(g.coors[(size_t)i * 111 + a * 3 + 2]).c_str());
      out += line;
    }
  }
  out += "#\n";
  return out;
}

/** The structure as PDB records. `atomNames` is 37 long, `resNames` 147. */
inline std::string toPdb(const GenOut& g, int n, const std::vector<std::string>& atomNames,
                         const std::vector<std::string>& resNames,
                         const std::vector<int>& chainOf, const OutNaming& naming = {}) {
  std::string out;
  char line[96];
  int serial = 1;
  for (int i = 0; i < n; ++i) {
    if (naming.skip(i)) continue;
    const int rt = g.residueType[i];
    const std::string res = rt >= 0 && rt < (int)resNames.size() ? resNames[rt] : "UNK";
    // 🔴 THE CHAIN, NOT A LITERAL 'A'. Every atom used to go into chain A, which for a binder or
    // a complex is a structure no viewer and no scorer can read correctly - the chains are there
    // in the batch and were simply not carried to the writer.
    const std::string chainStr = i < (int)naming.chain.size()
                                     ? chainFromCode(naming.chain[i])
                                     : chainName(i < (int)chainOf.size() ? chainOf[i] : 0);
    const char chain = chainStr.empty() ? 'A' : chainStr[0];
    const int resId = i < (int)naming.resId.size() ? naming.resId[i] : i + 1;
    for (int a = 0; a < 37; ++a) {
      if (!g.atomMask[(size_t)i * 37 + a]) continue;
      const std::string& name = atomNames[a];
      // PDB columns: a one- or two-letter element name starts at 14, not 13.
      char field[5];
      snprintf(field, sizeof field, name.size() >= 4 ? "%-4s" : " %-3s", name.c_str());
      snprintf(line, sizeof line,
               "ATOM  %5d %4s %3s %c%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n", serial++,
               field, res.c_str(), chain, resId, g.coors[(size_t)i * 111 + a * 3],
               g.coors[(size_t)i * 111 + a * 3 + 1], g.coors[(size_t)i * 111 + a * 3 + 2],
               name.substr(0, 1).c_str());
      out += line;
    }
  }
  out += "TER\nEND\n";
  return out;
}

}  // namespace rfd4p
