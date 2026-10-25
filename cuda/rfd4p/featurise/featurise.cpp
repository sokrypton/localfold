// The RFD4-Proteina featuriser in C++: a condition spec in, the port's input directory out, with
// no JavaScript anywhere on the path.
//
//     cuda/rfd4p/featurise/rfd4p-featurise <out dir> --spec=<job.json> --weights=<dir>
//                                          [--seed=N] [--step=N] [--length=N] [--generate]
//
// 🔴 IT IS HELD TO THE SAME BYTES AS `export_input.mjs`, NOT TO A TOLERANCE. The JavaScript
// exporter drives the page's own `shared/rfd4p/` and stays as the reference; this writes the same
// `model.idx` and `model.bin`, entry by entry, and `gate.py`'s `featurise` arm fails on a single
// differing float. That bar is `cuda/featurise`'s and it is the only one worth having: a
// featuriser that merely agrees closely is a second, slightly different model.
//
// 🔴 AND THAT BAR IS WHY `jsmath.h` CARRIES V8's log AND exp. `get_time_embedding` is
// `exp(-log(2000) / (half - 1) * i)` a channel, and glibc's own log and exp differ from V8's on
// 2.6% and 7.2% of arguments - so a featuriser built on the system libm cannot be held to bytes,
// however right its structure.
//
// WHAT IS COVERED SO FAR: a pure condition spec - segments of protein, DNA or RNA, fixed or
// ranged in length, in one chain or several, with secondary-structure and CATH conditioning. An
// input STRUCTURE (the seven `web/examples/*.cif.zst` presets) still goes through the exporter,
// because it needs the mmCIF reader, the CCD bond tables and the chirality templates.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../featurise/json.h"
#include "../../featurise/jsmath.h"
#include "../../featurise/zstd.h"
#include "contig.h"
#include "selection.h"
#include "conditions.h"
#include "../src/schedule.h"
#include "atoms.h"
#include "chirality.h"
#include "cif.h"
#include "tables.inc"

namespace rfd4p {
namespace featurise {

using std::string;
using std::vector;

// ---------------------------------------------------------------- the bundle's own numbers
//
// Every width, every feature list and every embedding table comes from the bundle, never from a
// literal here: a dimension typed into this file is a dimension that stops agreeing the day the
// checkpoint changes, and the port's whole design is that the weights decide.
struct Config {
  vector<string> featsSeq, featsCondSeq, featsPairRepr, featsPairCond;
  int seqSepDim = 64, xtPairDistDim = 30, tEmbDim = 256, recycleDim = 1;
  double xtPairDistMin = 0.1, xtPairDistMax = 3.0;
  bool noiseAlignChirality = true;
};

// ---------------------------------------------------------------- the batch
//
// 🔴 ONLY WHAT THE FEATURES READ. The reference's `convertToProteinaBatch` emits eighty keys and
// the four feature builders read sixty-five of them; the rest reach the model through nothing on
// this path. Each field here carries the reference's own INITIALISATION value, because for a spec
// input most of them are never written and the init value IS the feature - `element` is -1 and
// `condition_index` is 1 by that route, not by anybody's choice.
struct Batch {
  int n = 0;
  // 🔴 THESE ELEVEN ARE THE STRUCTURE PATH'S, AND A CONTIG LEAVES THEM AT THEIR FILL. On the
  // spec path `convertToProteinaBatch` is a no-op over all of them, which is why they were
  // absent until a real structure had to go through: `residue_type` is `<M>` for every token,
  // `coords` is zero because there are none, and `is_resolved_mask` is false because nothing
  // was observed. A field a contig never writes is still a field the model reads.
  vector<int> residueType, tokIdx, insertionSourceIdx;  // [n]
  vector<uint8_t> isResolvedMask;                       // [n]
  vector<float> lossWeight;                             // [n]
  vector<float> coords, coordsNm;                       // [n, 37, 3]
  vector<int> groundTruthElement;                       // [n, 37]
  vector<int> incidentBondTypeCounts;                   // [n, 37, 10]
  vector<uint8_t> conditionIndex;                       // [n, 37]  fill 1
  vector<int> tetrahedralVertices;                      // [n, 4]
  float centroidWithoutCond[3] = {0, 0, 0};
  vector<uint8_t> paddingMask;              // [n]  all 1
  vector<int> tokenType;                    // [n]
  vector<int> moleculeIid, moleculeEntityId, chainIid, indexGroup;
  vector<uint8_t> supportsLocalLatent, isInContextConditioning, coordCondMask;
  vector<int> conditionSequence;            // [n]  residue_type of the conditioned sequence
  vector<int> secondaryStructure;           // [n]
  vector<int> cathLabel;                    // [n, 3], -1 where unset
  vector<float> seqProb;                    // [28, n], NaN where unknown

  vector<uint8_t> atom37Mask;               // [n, 37]
  vector<int> element, charge, rasaBinned, hbond, pistack, pication, saltbridge, hydrophobic;
  vector<int> metalCoord, hotspot, coordGroup;
  vector<float> condCoord, condInternalCoord;   // [n, 37, 3]  NaN
  vector<float> chainCentroid;                  // [n, 3]      NaN
  vector<float> chirality;                      // [n, 18]
  vector<int> chiralFrameAtoms, chiralSlotOwner;
  vector<int> naSsState, naSsStateMask;

  vector<uint8_t> bondMask, isAromatic;         // [n, n]
  vector<int> bondOrder;                        // [n, n]
  vector<int8_t> canonicalPolymerOrdering;      // [n, n]
  vector<int> tokenBondPathLength;              // [n, n]
  vector<int> naSsPair, naSsPairMask;           // [n, n]

  void resize(int tokens) {
    n = tokens;
    const size_t t = tokens, a = t * 37, p = t * t;
    residueType.assign(t, 0);          // 0 is `<M>`
    tokIdx.assign(t, 0);
    insertionSourceIdx.assign(t, -1);
    isResolvedMask.assign(t, 0);
    lossWeight.assign(t, 1.0f);
    coords.assign(a * 3, 0.f);
    coordsNm.assign(a * 3, 0.f);
    groundTruthElement.assign(a, 0);
    incidentBondTypeCounts.assign(a * 10, 0);
    conditionIndex.assign(a, 1);
    tetrahedralVertices.assign(t * 4, 0);
    centroidWithoutCond[0] = centroidWithoutCond[1] = centroidWithoutCond[2] = std::nanf("");
    paddingMask.assign(t, 1);
    tokenType.assign(t, tables::TT_PROTEIN);
    moleculeIid.assign(t, 0);
    moleculeEntityId.assign(t, 1);    // the inference remap, not a count
    chainIid.assign(t, 0);
    indexGroup.assign(t, 0);
    supportsLocalLatent.assign(t, 1);
    isInContextConditioning.assign(t, 0);
    coordCondMask.assign(t, 0);
    conditionSequence.assign(t, 0);   // 0 is `<M>`
    secondaryStructure.assign(t, 0);
    cathLabel.assign(t * 3, -1);
    seqProb.assign(28 * t, std::nanf(""));
    atom37Mask.assign(a, 0);
    element.assign(a, -1);
    charge.assign(a, 0);
    rasaBinned.assign(a, 0);
    hbond.assign(a, 0);
    pistack.assign(a, 0);
    pication.assign(a, 0);
    saltbridge.assign(a, 0);
    hydrophobic.assign(a, 0);
    metalCoord.assign(a, 0);
    hotspot.assign(a, 0);
    coordGroup.assign(a, -1);
    condCoord.assign(a * 3, std::nanf(""));
    condInternalCoord.assign(a * 3, std::nanf(""));
    chainCentroid.assign(t * 3, std::nanf(""));
    chirality.assign(t * 18, 0.f);
    chiralFrameAtoms.assign(t * tables::N_TOKEN_TYPES, -1);   // resized by the caller that has F
    chiralSlotOwner.assign(t * 6, -1);
    naSsState.assign(t, 0);
    naSsStateMask.assign(t, 0);
    bondMask.assign(p, 0);
    isAromatic.assign(p, 0);
    bondOrder.assign(p, 0);
    canonicalPolymerOrdering.assign(p, 0);
    tokenBondPathLength.assign(p, 0);
    naSsPair.assign(p, 0);
    naSsPairMask.assign(p, 0);
  }
};

// ---------------------------------------------------------------- the features
//
// `get_time_embedding` (max_positions 2000): `[sin, cos]` over `dim / 2` geometric frequencies.
// Every rounding is where the reference's is - `Math.fround` at the product and at the frequency,
// doubles everywhere else - because this is compared as bytes.
inline vector<float> timeEmbedding(double t, int dim) {
  const int half = dim >> 1;
  vector<float> out(dim);
  const double scale = lf::jsmath::log(2000.0) / (half - 1);
  const float tt = (float)(t * 2000.0);
  for (int i = 0; i < half; ++i) {
    const float freq = (float)lf::jsmath::exp((float)(-scale * i));
    const float a = (float)((double)tt * freq);
    out[i] = (float)lf::jsmath::sin(a);
    out[half + i] = (float)lf::jsmath::cos(a);
  }
  return out;
}


// ---------------------------------------------------------------- the weights this needs
//
// Four embedding tables and three CATH ones, read out of the SAME `model.idx`/`model.bin` the port
// reads. There is no second weight format and no second reader: the exporter writes one and both
// the featuriser and the fold read it.
struct Weights {
  std::unordered_map<string, std::pair<size_t, size_t>> index;   // name -> (offset, count)
  vector<float> blob;
  const float* operator()(const string& name) const {
    auto it = index.find(name);
    if (it == index.end()) {
      fprintf(stderr, "the weights have no %s\n", name.c_str());
      exit(1);
    }
    return blob.data() + it->second.first;
  }
  bool has(const string& name) const { return index.count(name) > 0; }
};

/**
 * The PUBLISHED int8 bundle's float32 tensors, by name.
 *
 * 🔴 EVERY TABLE THIS NEEDS IS float32 IN THAT BUNDLE, which is not luck: only the matrices are
 * coded, and an embedding table is not a matrix anybody would quantise - "quantising a LayerNorm's
 * scale saves a few kilobytes and moves every number downstream of it" (src/bundle.cuh). So the
 * featuriser reads the bundle directly and a whole design needs no float checkpoint, which is the
 * difference between a run anybody can make and one that needs NVIDIA's weight terms accepted.
 */
inline Weights readBundleWeights(const string& dir, const std::set<string>& wanted) {
  Weights w;
  string text;
  {
    FILE* f = fopen((dir + "/manifest.json").c_str(), "rb");
    if (!f) { fprintf(stderr, "no %s/manifest.json\n", dir.c_str()); exit(1); }
    char buf[65536];
    size_t got;
    while ((got = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, got);
    fclose(f);
  }
  const lf::Json manifest = lf::parseJson(text);
  const lf::Json* tensors = manifest.get("tensors");
  if (!tensors || !tensors->isObject()) {
    fprintf(stderr, "%s/manifest.json has no tensors\n", dir.c_str());
    exit(1);
  }
  size_t at = 0;
  for (const auto& [name, rec] : tensors->o) {
    if (!wanted.count(name)) continue;
    const lf::Json* dtype = rec.get("dtype");
    if (dtype && dtype->isString() && dtype->s != "float32") {
      fprintf(stderr, "%s is %s in the bundle; this featuriser reads only its float32 tensors\n",
              name.c_str(), dtype->s.c_str());
      exit(1);
    }
    size_t elements = 1;
    const lf::Json* shape = rec.get("shape");
    if (shape && shape->isArray()) for (const lf::Json& d : shape->a) elements *= (size_t)d.n;
    const lf::Json* file = rec.get("file");
    const lf::Json* off = rec.get("byteOffset");
    FILE* bin = fopen((dir + "/shards/" + (file ? file->s : "")).c_str(), "rb");
    if (!bin) {
      fprintf(stderr, "no %s/shards/%s\n", dir.c_str(), file ? file->s.c_str() : "?");
      exit(1);
    }
    w.blob.resize(at + elements);
    if (fseek(bin, (long)(off ? off->n : 0), SEEK_SET) != 0 ||
        fread(w.blob.data() + at, 4, elements, bin) != elements) {
      fprintf(stderr, "%s runs past its shard\n", name.c_str());
      exit(1);
    }
    fclose(bin);
    w.index["w/" + name] = {at, elements};
    at += elements;
  }
  return w;
}

inline Weights readWeights(const string& dir, const std::set<string>& wanted) {
  Weights w;
  FILE* idx = fopen((dir + "/model.idx").c_str(), "r");
  if (!idx) { fprintf(stderr, "no %s/model.idx\n", dir.c_str()); exit(1); }
  // 🔴 LINE BY LINE, NOT `fscanf` OVER THE WHOLE FILE. A `m meta/<key> <value>` line has THREE
  // fields where a tensor line has four, so a four-field scan swallows the next line's `t` as a
  // length, fails, and stops - with the real tensors still ahead of it. The symptom was "the
  // weights have no <name>" for a name that is plainly in the index.
  char line[1024], kind[8], name[512];
  long long off, len;
  vector<std::pair<string, std::pair<long long, long long>>> keep;
  size_t total = 0;
  while (fgets(line, sizeof line, idx)) {
    if (sscanf(line, "%7s %511s %lld %lld", kind, name, &off, &len) != 4) continue;
    if (kind[0] != 't' || !wanted.count(name)) continue;
    keep.push_back({name, {off, len}});
    total += (size_t)len;
  }
  fclose(idx);
  w.blob.resize(total);
  FILE* bin = fopen((dir + "/model.bin").c_str(), "rb");
  if (!bin) { fprintf(stderr, "no %s/model.bin\n", dir.c_str()); exit(1); }
  size_t at = 0;
  for (const auto& [n, where] : keep) {
    if (fseek(bin, where.first * 4, SEEK_SET) != 0 ||
        fread(w.blob.data() + at, 4, (size_t)where.second, bin) != (size_t)where.second) {
      fprintf(stderr, "%s runs past %s/model.bin\n", n.c_str(), dir.c_str());
      exit(1);
    }
    w.index[n] = {at, (size_t)where.second};
    at += (size_t)where.second;
  }
  fclose(bin);
  return w;
}

// ---------------------------------------------------------------- the spec
//
// A contig is comma-separated tokens; the shape this covers is `name:min-maxP` - a NAMED
// expandable polymer segment, which is what `custom_cath_monomer` and `--length=N` are made of.
// A preserved segment (`A16-35`) names residues of an input structure and needs the mmCIF reader.
struct Segment {
  string name;
  int minLen = 0, maxLen = 0;
  bool chainBreak = false;      // the `/0` token: what follows is a new chain
  int tokenType = tables::TT_PROTEIN;
};

inline vector<Segment> parseContig(const string& contig) {
  vector<Segment> out;
  size_t at = 0;
  while (at <= contig.size()) {
    const size_t comma = contig.find(',', at);
    string tok = contig.substr(at, comma == string::npos ? string::npos : comma - at);
    at = comma == string::npos ? contig.size() + 1 : comma + 1;
    // trim
    while (!tok.empty() && isspace((unsigned char)tok.front())) tok.erase(tok.begin());
    while (!tok.empty() && isspace((unsigned char)tok.back())) tok.pop_back();
    if (tok.empty()) continue;
    if (tok == "/0") { out.push_back(Segment{"", 0, 0, true}); continue; }
    const size_t colon = tok.find(':');
    if (colon == string::npos) {
      fprintf(stderr, "contig token '%s' is not a named expandable segment; this featuriser does "
                      "not yet take a preserved segment, which needs the input structure\n",
              tok.c_str());
      exit(1);
    }
    Segment seg;
    seg.name = tok.substr(0, colon);
    string rest = tok.substr(colon + 1);
    // The chain-type suffix. For a contig every token is a `<M>` stub whatever the chain is, so
    // the suffix reaches the batch as `token_type` and nothing else - but that is what the model
    // reads to know it is folding a nucleic chain.
    if (!rest.empty() && isalpha((unsigned char)rest.back())) {
      switch (rest.back()) {
        case 'P': seg.tokenType = tables::TT_PROTEIN; break;
        case 'R': seg.tokenType = tables::TT_RNA; break;
        case 'D': seg.tokenType = tables::TT_DNA; break;
        default:
          fprintf(stderr, "contig segment '%s' has chain type '%c', which is not P, R or D\n",
                  tok.c_str(), rest.back());
          exit(1);
      }
      rest.pop_back();
    }
    const size_t dash = rest.find('-');
    seg.minLen = atoi(rest.substr(0, dash).c_str());
    seg.maxLen = dash == string::npos ? seg.minLen : atoi(rest.substr(dash + 1).c_str());
    out.push_back(seg);
  }
  return out;
}


// ---------------------------------------------------------------- the four feature blocks
//
// Each is laid out from the bundle's own `feats_*` list, in that order, so a reordered list moves
// the columns rather than silently mixing them. The widths are the reference's.

inline void embedRow(const float* table, int dim, int idx, float* dst) {
  for (int k = 0; k < dim; ++k) dst[k] = table[(size_t)idx * dim + k];
}

constexpr int TOKEN_TYPE_ATOMIZED = 3;

/** `trunkSeqFeatures` (`nn.init_repr_factory`): `[n, 897]` for the shipped checkpoint. */
inline vector<float> seqFeatures(const Batch& b, const Config& cfg, const Weights& w,
                                 const vector<float>& xBb, const vector<float>& xLl, int& dimOut) {
  const int n = b.n;
  // Pass one sizes the row; pass two fills it. The reference builds a block list the same way.
  vector<std::pair<string, int>> blocks;
  for (size_t idx = 0; idx < cfg.featsSeq.size(); ++idx) {
    const string& f = cfg.featsSeq[idx];
    int d = -1;
    if (f == "xt_bb_ca") d = 3;
    else if (f == "xt_local_latents") d = 32;
    else if (f == "token_type") d = 32;
    else if (f == "sequence_probabilities") d = 2 * 28;
    else if (f == "cond_a37coors_nm" || f == "cond_a37coors_internal_nm") d = 148;
    else if (f == "rasa_binned") d = 37;
    else if (f == "secondary_structure") d = 16;
    else if (f == "hbond_modernized" || f == "pication") d = 74;
    else if (f == "pistack" || f == "saltbridge" || f == "hydrophobic" || f == "metal_coord" ||
             f == "hotspot" || f == "coord_group_seq") d = 37;
    else if (f == "na_ss_state") d = 2;
    else if (f == "atom_charge" || f == "atomic_element") d = 16;
    else if (f == "chirality") d = 18;
    else if (f == "chain_centroid_xyz") d = 3;
    else { fprintf(stderr, "sequence feature '%s' not implemented\n", f.c_str()); exit(1); }
    blocks.push_back({f, d});
  }
  int dim = 0;
  for (const auto& [f, d] : blocks) dim += d;
  dimOut = dim;
  vector<float> data((size_t)n * dim, 0.f);

  const string pre = "w/nn.init_repr_factory.feat_creators.";
  for (int i = 0; i < n; ++i) {
    if (!b.paddingMask[i]) continue;
    size_t off = (size_t)i * dim;
    for (size_t idx = 0; idx < blocks.size(); ++idx) {
      const string& f = blocks[idx].first;
      float* d = data.data() + off;
      const string tag = std::to_string(idx);
      if (f == "xt_bb_ca") {
        for (int k = 0; k < 3; ++k) d[k] = xBb[(size_t)i * 3 + k];
      } else if (f == "xt_local_latents") {
        for (int k = 0; k < 32; ++k) d[k] = xLl[(size_t)i * 32 + k];
      } else if (f == "token_type") {
        embedRow(w(pre + tag + ".embedding.weight"), 32, b.tokenType[i], d);
      } else if (f == "sequence_probabilities") {
        for (int c = 0; c < 28; ++c) {
          const float p = b.seqProb[(size_t)c * n + i];
          const bool known = !std::isnan(p);
          d[c] = known ? p : 0.f;
          d[28 + c] = known ? 1.f : 0.f;
        }
      } else if (f == "cond_a37coors_nm") {
        for (int a = 0; a < 37; ++a) {
          const size_t base = ((size_t)i * 37 + a) * 3;
          const bool known = std::isfinite(b.condCoord[base]) ||
                             std::isfinite(b.condCoord[base + 1]) ||
                             std::isfinite(b.condCoord[base + 2]);
          if (!known) continue;
          for (int k = 0; k < 3; ++k) {
            const float v = b.condCoord[base + k];
            d[a * 3 + k] = std::isnan(v) ? 0.f : v / 10.f;
          }
          d[111 + a] = 1.f;
        }
      } else if (f == "cond_a37coors_internal_nm") {
        for (int a = 0; a < 37; ++a) {
          const int g = b.coordGroup[(size_t)i * 37 + a];
          if (g == -1 || g == 0) continue;            // UNKNOWN, ABSOLUTE
          const size_t base = ((size_t)i * 37 + a) * 3;
          for (int k = 0; k < 3; ++k) {
            const float v = b.condInternalCoord[base + k];
            d[a * 3 + k] = std::isnan(v) ? 0.f : v / 10.f;
          }
          d[111 + a] = 1.f;
        }
      } else if (f == "rasa_binned") {
        for (int a = 0; a < 37; ++a) {
          d[a] = b.atom37Mask[(size_t)i * 37 + a] ? (float)b.rasaBinned[(size_t)i * 37 + a] : 0.f;
        }
      } else if (f == "secondary_structure") {
        embedRow(w(pre + tag + ".ss_embedding.weight"), 16, b.secondaryStructure[i], d);
      } else if (f == "hbond_modernized" || f == "pication") {
        const vector<int>& ann = f == "hbond_modernized" ? b.hbond : b.pication;
        for (int k = 0; k < 37; ++k) {
          if (!b.atom37Mask[(size_t)i * 37 + k]) continue;
          const int v = ann[(size_t)i * 37 + k];
          d[k] = (v == 1 || v == 3) ? 1.f : 0.f;
          d[37 + k] = (v == 2 || v == 3) ? 1.f : 0.f;
        }
      } else if (f == "pistack" || f == "saltbridge" || f == "hydrophobic" || f == "metal_coord" ||
                 f == "hotspot") {
        const vector<int>& ann = f == "pistack" ? b.pistack
                               : f == "saltbridge" ? b.saltbridge
                               : f == "hydrophobic" ? b.hydrophobic
                               : f == "metal_coord" ? b.metalCoord : b.hotspot;
        for (int k = 0; k < 37; ++k) {
          if (!b.atom37Mask[(size_t)i * 37 + k]) continue;
          d[k] = ann[(size_t)i * 37 + k] != 0 ? 1.f : 0.f;
        }
      } else if (f == "na_ss_state") {
        d[0] = (float)b.naSsState[i];
        d[1] = (float)b.naSsStateMask[i];
      } else if (f == "atom_charge") {
        if (b.tokenType[i] == TOKEN_TYPE_ATOMIZED) {
          const int cls = std::min(10, std::max(0, b.charge[(size_t)i * 37 + 1] + 4));
          embedRow(w(pre + tag + ".embedding.weight"), 16, cls, d);
        }
      } else if (f == "atomic_element") {
        if (b.tokenType[i] == TOKEN_TYPE_ATOMIZED) {
          const int e = std::min(127, std::max(0, b.element[(size_t)i * 37 + 1]));
          embedRow(w(pre + tag + ".embedding.weight"), 16, e, d);
        }
      } else if (f == "coord_group_seq") {
        for (int k = 0; k < 37; ++k) d[k] = (float)b.coordGroup[(size_t)i * 37 + k];
      } else if (f == "chirality") {
        for (int k = 0; k < 18; ++k) d[k] = b.chirality[(size_t)i * 18 + k];
      } else if (f == "chain_centroid_xyz") {
        bool ok = true;
        for (int k = 0; k < 3; ++k) ok = ok && std::isfinite(b.chainCentroid[(size_t)i * 3 + k]);
        if (ok) for (int k = 0; k < 3; ++k) d[k] = b.chainCentroid[(size_t)i * 3 + k] / 10.f;
      }
      off += blocks[idx].second;
    }
  }
  return data;
}

/** `trunkCondFeatures` (`nn.cond_factory`): `[n, 609]`, the same row on every token. */
inline vector<float> condFeatures(const Batch& b, const Config& cfg, const Weights& w, double tBb,
                                  double tLl, int recycleRemaining, int& dimOut) {
  const int n = b.n;
  vector<std::pair<string, int>> blocks;
  for (const string& f : cfg.featsCondSeq) {
    int d = -1;
    if (f == "time_emb_bb_ca" || f == "time_emb_local_latents") d = cfg.tEmbDim;
    else if (f == "cath_code") d = 96;
    else if (f == "recycle_remaining") d = cfg.recycleDim;
    else { fprintf(stderr, "conditioning feature '%s' not implemented\n", f.c_str()); exit(1); }
    blocks.push_back({f, d});
  }
  int dim = 0;
  for (const auto& [f, d] : blocks) dim += d;
  dimOut = dim;
  vector<float> data((size_t)n * dim, 0.f);
  const vector<float> eBb = timeEmbedding(tBb, cfg.tEmbDim);
  const vector<float> eLl = timeEmbedding(tLl, cfg.tEmbDim);
  const string pre = "w/nn.cond_factory.feat_creators.";
  for (int i = 0; i < n; ++i) {
    if (!b.paddingMask[i]) continue;
    size_t off = (size_t)i * dim;
    for (size_t idx = 0; idx < blocks.size(); ++idx) {
      const string& f = blocks[idx].first;
      float* d = data.data() + off;
      if (f == "time_emb_bb_ca") std::copy(eBb.begin(), eBb.end(), d);
      else if (f == "time_emb_local_latents") std::copy(eLl.begin(), eLl.end(), d);
      else if (f == "cath_code") {
        const char* which[3] = {".embedding_C.weight", ".embedding_A.weight", ".embedding_T.weight"};
        for (int k = 0; k < 3; ++k) {
          embedRow(w(pre + std::to_string(idx) + which[k]), 32,
                   b.cathLabel[(size_t)i * 3 + k] + 1, d + 32 * k);
        }
      } else if (f == "recycle_remaining") {
        if (recycleRemaining >= 0) {
          d[std::min(recycleRemaining, cfg.recycleDim - 1)] = 1.f;
        }
      }
      off += blocks[idx].second;
    }
  }
  return data;
}

/** `pairFeatures` (`nn.pair_repr_builder.init_repr_factory`): `[n * n, 107]`. */
inline vector<float> pairFeatures(const Batch& b, const Config& cfg, const vector<float>& caNm,
                                  bool skipDists, int& dimOut) {
  const int n = b.n;
  vector<float> limits(cfg.xtPairDistDim - 1);
  for (int k = 0; k < cfg.xtPairDistDim - 1; ++k) {
    limits[k] = (float)(cfg.xtPairDistMin +
                        (cfg.xtPairDistMax - cfg.xtPairDistMin) * k / (cfg.xtPairDistDim - 2));
  }
  vector<std::pair<string, int>> blocks;
  for (const string& f : cfg.featsPairRepr) {
    int d = -1;
    if (f == "xt_bb_ca_pair_dists" || f == "ca_coors_nm_pair_dists") d = cfg.xtPairDistDim;
    else if (f == "bond_mask" || f == "is_aromatic" || f == "canonical_polymer_ordering" ||
             f == "index_group_pair" || f == "molecule_idx_pair" || f == "molecule_entity_pair")
      d = 1;
    else if (f == "bond_order") d = 5;
    else if (f == "token_bond_path_length") d = cfg.seqSepDim;
    else if (f == "na_ss_pair") d = 2;
    else { fprintf(stderr, "pair feature '%s' not implemented\n", f.c_str()); exit(1); }
    blocks.push_back({f, d});
  }
  int dim = 0;
  for (const auto& [f, d] : blocks) dim += d;
  dimOut = dim;
  vector<float> data((size_t)n * n * dim, 0.f);
  for (int i = 0; i < n; ++i) {
    if (!b.paddingMask[i]) continue;
    for (int j = 0; j < n; ++j) {
      if (!b.paddingMask[j]) continue;
      const size_t ij = (size_t)i * n + j;
      size_t off = ij * dim;
      for (size_t idx = 0; idx < blocks.size(); ++idx) {
        const string& f = blocks[idx].first;
        float* d = data.data() + off;
        if (f == "xt_bb_ca_pair_dists" || f == "ca_coors_nm_pair_dists") {
          if (!skipDists) {
            const float dx = caNm[(size_t)i * 3] - caNm[(size_t)j * 3];
            const float dy = caNm[(size_t)i * 3 + 1] - caNm[(size_t)j * 3 + 1];
            const float dz = caNm[(size_t)i * 3 + 2] - caNm[(size_t)j * 3 + 2];
            const float r = (float)std::sqrt((float)(dx * dx + dy * dy + dz * dz));
            int bin = 0;
            while (bin < (int)limits.size() && limits[bin] < r) ++bin;
            d[bin] = 1.f;
          }
        } else if (f == "bond_mask") d[0] = (float)b.bondMask[ij];
        else if (f == "is_aromatic") d[0] = (float)b.isAromatic[ij];
        else if (f == "canonical_polymer_ordering") d[0] = (float)b.canonicalPolymerOrdering[ij];
        else if (f == "bond_order") d[b.bondOrder[ij]] = 1.f;
        else if (f == "index_group_pair") d[0] = b.indexGroup[i] == b.indexGroup[j] ? 1.f : 0.f;
        else if (f == "molecule_idx_pair") d[0] = b.moleculeIid[i] == b.moleculeIid[j] ? 1.f : 0.f;
        else if (f == "molecule_entity_pair")
          d[0] = b.moleculeEntityId[i] == b.moleculeEntityId[j] ? 1.f : 0.f;
        else if (f == "token_bond_path_length")
          d[std::min(b.tokenBondPathLength[ij], cfg.seqSepDim - 1)] = 1.f;
        else if (f == "na_ss_pair") {
          d[0] = (float)b.naSsPair[ij];
          d[1] = (float)b.naSsPairMask[ij];
        }
        off += blocks[idx].second;
      }
    }
  }
  return data;
}

/** `pairCondFeatures`: `[feats * tEmbDim]`, per example rather than per token. */
inline vector<float> pairCondFeatures(const Config& cfg, double tBb, double tLl) {
  vector<float> out(cfg.featsPairCond.size() * cfg.tEmbDim, 0.f);
  for (size_t k = 0; k < cfg.featsPairCond.size(); ++k) {
    const string& f = cfg.featsPairCond[k];
    vector<float> e;
    if (f == "time_emb_bb_ca") e = timeEmbedding(tBb, cfg.tEmbDim);
    else if (f == "time_emb_local_latents") e = timeEmbedding(tLl, cfg.tEmbDim);
    else { fprintf(stderr, "pair conditioning feature '%s' not implemented\n", f.c_str()); exit(1); }
    std::copy(e.begin(), e.end(), out.begin() + k * cfg.tEmbDim);
  }
  return out;
}

// ---------------------------------------------------------------- the selection language
//
// 🔴 A SUBSET, AND IT SAYS SO RATHER THAN GUESSING. The reference's `evalSelection` is a small
// expression language over `chain/res_name/res_id/atom_name`; a pure contig has no residue numbers
// and no atoms, so what a spec job can actually say is a union of `sel('*/<segment>/*/*')` and
// `sel('*')`. Anything else is REFUSED by name, because a selection this does not understand must
// not quietly match nothing - that is a condition the model is told is absent.
inline vector<uint8_t> evalSelection(const string& expr, const vector<string>& owner) {
  const int n = (int)owner.size();
  vector<uint8_t> out(n, 0);
  size_t at = 0;
  while (at < expr.size()) {
    const size_t open = expr.find("sel('", at);
    if (open == string::npos) break;
    const size_t close = expr.find("')", open);
    if (close == string::npos) {
      fprintf(stderr, "selection '%s' has an unterminated sel(...)\n", expr.c_str());
      exit(1);
    }
    const string path = expr.substr(open + 5, close - open - 5);
    at = close + 2;
    // `chain/res_name/res_id/atom_name`, with a missing tail meaning `*`.
    vector<string> parts;
    size_t from = 0;
    while (true) {
      const size_t slash = path.find('/', from);
      parts.push_back(path.substr(from, slash == string::npos ? string::npos : slash - from));
      if (slash == string::npos) break;
      from = slash + 1;
    }
    while (parts.size() < 4) parts.push_back("*");
    for (size_t k = 0; k < 4; ++k) {
      if (k == 1) continue;
      if (parts[k] != "*") {
        fprintf(stderr, "selection '%s' constrains %s, which needs an input structure; this "
                        "featuriser takes a union of sel('*/<segment>/*/*')\n",
                path.c_str(), k == 0 ? "a chain" : (k == 2 ? "a residue number" : "an atom name"));
        exit(1);
      }
    }
    for (int i = 0; i < n; ++i) {
      if (parts[1] == "*" || parts[1] == owner[i]) out[i] = 1;
    }
  }
  // Whatever separates the terms, a `sel(...)` not matched above has already been taken as a union
  // member; `&` and `~` would change that, so they are refused.
  if (expr.find('&') != string::npos || expr.find('~') != string::npos) {
    fprintf(stderr, "selection '%s' uses & or ~; only | is covered\n", expr.c_str());
    exit(1);
  }
  return out;
}

// `applyConditions` for the two a pure spec can carry: an enum over tokens (secondary structure)
// and a CATH label. Anything else is refused rather than dropped.
inline void applyConditions(Batch& b,
                            const std::map<string, std::map<string, vector<string>>>& conditions,
                            const vector<string>& owner) {
  for (const auto& [key, arms] : conditions) {
    auto spec = tables::CONDITION_REGISTRY.find(key);
    if (spec == tables::CONDITION_REGISTRY.end()) {
      fprintf(stderr, "condition '%s' is not in the registry\n", key.c_str());
      exit(1);
    }
    const string full = spec->second.fullName, dtype = spec->second.dtype;
    for (const auto& [value, selections] : arms) {
      vector<uint8_t> hit(b.n, 0);
      for (const string& sel : selections) {
        const vector<uint8_t> m = evalSelection(sel, owner);
        for (int i = 0; i < b.n; ++i) hit[i] |= m[i];
      }
      if (dtype == "enum") {
        const auto& members = tables::CONDITION_ENUMS.at(spec->second.enumName);
        auto it = members.find(value);
        if (it == members.end()) {
          fprintf(stderr, "'%s' is not a member of %s\n", value.c_str(), spec->second.enumName);
          exit(1);
        }
        if (full != "condition_secondary-structure_1_residue") {
          fprintf(stderr, "condition '%s' needs a per-ATOM write, which a contig has no atoms "
                          "for\n", full.c_str());
          exit(1);
        }
        for (int i = 0; i < b.n; ++i) if (hit[i]) b.secondaryStructure[i] = it->second;
      } else if (dtype == "cath") {
        auto code = tables::CATH_CODES.find(value);
        if (code == tables::CATH_CODES.end()) {
          fprintf(stderr, "CATH code '%s' is not one the maps can spell\n", value.c_str());
          exit(1);
        }
        for (int i = 0; i < b.n; ++i) {
          if (!hit[i]) continue;
          for (int k = 0; k < 3; ++k) b.cathLabel[(size_t)i * 3 + k] = code->second[k];
        }
      } else {
        // 🔴 REFUSED, WHERE THE REFERENCE SILENTLY IGNORES IT. `C_SEQ` on a pure contig reaches
        // `finalizeAndCanonicalizeSequenceConditioning`, which rewrites every stub back to `<M>`,
        // so the page folds such a job with the condition having done NOTHING - measured. A job
        // whose conditioning is quietly dropped is worse than one that will not start, so this
        // says so; what it costs is a job the page accepts and this does not.
        fprintf(stderr, "condition '%s' has dtype %s, which needs an input structure - on a pure "
                        "contig the reference drops it silently, so this refuses rather than "
                        "fold as though it had been applied\n",
                key.c_str(), dtype.c_str());
        exit(1);
      }
    }
  }
}

// ---------------------------------------------------------------- the pipeline's own RNG
//
// 🔴 TWO SEPARATE RANDOM SOURCES, AND THEY ARE NOT THE SAME ONE. The SAMPLER's is xorshift128+
// with Box-Muller (`seededRng`, flow.js); the PIPELINE's is mulberry32 (`createSeededRng`,
// transforms.js) and it decides a ranged segment's realised length. Both are seeded from the job's
// `seed` and neither consumes the other's stream.
//
// 🔴 AND IT IS `atoms.h`'s, NOT A SECOND COPY. This file carried its own identical mulberry32 with
// `permutation` on it while `atoms.h`'s carried `randomSO3`, so the spec path and the structure
// path drew from two objects that merely happened to agree - and the structure path needs BOTH
// methods. One generator, one stream; a second copy is a divergence waiting for someone to edit
// one of them.
using PipelineRng = atoms::PipelineRng;

// ---------------------------------------------------------------- spec -> batch
//
// 🔴 FOR A SPEC INPUT ALMOST EVERY FIELD IS ITS INITIALISATION VALUE, and that is not laziness in
// the reference: every token realised from a contig is a `<M>` stub with no atoms, so
// `buildAtom37Slots` returns -1 for all 37 slots and the whole per-atom loop of
// `convertToProteinaBatch` is a no-op. `element` is -1 and `condition_index` is 1 because nothing
// overwrites the fill, and a port that "helpfully" computed them would disagree with the reference.
//
// What is actually computed is the chain: `token_bond_path_length[i][j] = |i - j|` over the single
// bonded run, and `canonical_polymer_ordering` is -1 at the next residue and +1 at the previous.
/**
 * `convertToProteinaBatch` with real atoms - the general case `buildSpecBatch` below is the
 * contig-shaped specialisation of.
 *
 * 🔴 IT OPENS WITH TWO DRAWS AND THEY ARE THE FIRST THING TO GET RIGHT: the chain permutation
 * then the molecule permutation, from the SAME stream the segment realisation and the internal
 * coordinates have already been drawing from. At one chain each is `permutation(1)`, whose
 * Fisher-Yates loop runs zero times and draws nothing, which is exactly why a one-chain job
 * hides a mistake here and three chains do not.
 *
 * 🔴 AND AT INFERENCE A CHAIN WHOSE SEQUENCE IS ENTIRELY `<M>` TAKES A FRESH ENTITY, in sorted
 * chain order, numbered after every entity the input had. Two masked chains are therefore NOT
 * the same entity however identical they look - which is the opposite of the rule everywhere
 * else in this featuriser, where identical contents mean one entity.
 *
 * 🔴 AND `element` IS THE GROUND TRUTH ONLY WHERE THE SEQUENCE IS GIVEN. `ground_truth_element`
 * always carries the atomic number; `element` carries -1 wherever the residue is masked, so the
 * model is not handed the composition of a residue it is being asked to design.
 */
/**
 * `runValidationPipeline`: the fourteen stages in the reference's own order, then the batch.
 *
 * 🔴 `addGlobalTokenIdAnnotation` RUNS TWICE. It is called after the atomisation decides which
 * residues are one token an atom, and AGAIN after
 * `finalizeAndCanonicalizeSequenceConditioning`. Measured, the second call is INERT - that
 * stage changes neither `atomize` nor the residue boundaries, so removing it leaves all
 * seventeen gate cases green - and it is kept because it is the reference's.
 *
 * 🔴 THE REASON IT MATTERS IS HOW IT WAS FOUND: the dump listed these stages by hand and so did
 * the gate's JavaScript side, so BOTH were missing the second call and agreed perfectly. A gate
 * that re-implements the thing it is checking is checking its own reimplementation. The order
 * lives HERE, once; the gate drives the reference's own `runValidationPipeline` against it, so
 * the next stage added to either is in the comparison without anyone remembering to add it.
 *
 * 🔴 AND `realizeExpandableSegments` IS INFERENCE-ONLY where everything after it is not.
 */
inline Batch buildStructureBatch(atoms::AtomArray& a, const atoms::Centroid& centroid,
                                 atoms::PipelineRng& prng, int pathLengthCutoff,
                                 bool isInference);

inline Batch runValidationPipeline(atoms::AtomArray& a, atoms::PipelineRng& prng,
                                   int pathLengthCutoff = 64, bool isInference = true) {
  atoms::addSrcComponentAnnotation(a);
  if (isInference) atoms::realizeExpandableSegments(a, prng);
  atoms::filterUnwantedAtoms(a);
  atoms::addGlobalAtomIdAnnotation(a);
  atoms::annotateCovalentModificationsAndAtomize(a);
  atoms::addGlobalTokenIdAnnotation(a);
  atoms::finalizeAndCanonicalizeSequenceConditioning(a);
  atoms::addGlobalTokenIdAnnotation(a);
  atoms::featurizeExpandedSegments(a, isInference);
  atoms::materializeInternalCoords(a, prng);
  atoms::unindexFlaggedTokens(a, isInference);
  const atoms::Centroid centroid = atoms::centerAndFeaturizeCentroid(a);
  atoms::addSupportsLocalLatentAnnotation(a);
  return buildStructureBatch(a, centroid, prng, pathLengthCutoff, isInference);
}

inline Batch buildStructureBatch(atoms::AtomArray& a, const atoms::Centroid& centroid,
                                 atoms::PipelineRng& prng, int pathLengthCutoff = 64,
                                 bool isInference = true) {
  const size_t nAtoms = a.n();
  const vector<int> tokStarts = atoms::tokenStarts(a);
  const int nTokens = tokStarts.empty() ? 0 : (int)tokStarts.size() - 1;
  Batch b;
  b.resize(nTokens);
  b.chiralFrameAtoms.assign((size_t)nTokens * 48, -1);
  for (int k = 0; k < 3; ++k) b.centroidWithoutCond[k] = centroid.v[k];
  if (nTokens == 0) return b;

  vector<string> uniqueChains;
  for (const string& c : a.ids.chainIid) {
    if (std::find(uniqueChains.begin(), uniqueChains.end(), c) == uniqueChains.end()) {
      uniqueChains.push_back(c);
    }
  }
  std::sort(uniqueChains.begin(), uniqueChains.end());
  std::map<string, int> chainToIdx;
  for (size_t i = 0; i < uniqueChains.size(); ++i) chainToIdx[uniqueChains[i]] = (int)i;
  std::set<int> molSet(a.ids.moleculeIid.begin(), a.ids.moleculeIid.end());
  const vector<int> uniqueMols(molSet.begin(), molSet.end());
  std::map<int, int> molToIdx;
  for (size_t i = 0; i < uniqueMols.size(); ++i) molToIdx[uniqueMols[i]] = (int)i;
  const vector<int> chainPerm = prng.permutation((int)uniqueChains.size());
  const vector<int> molPerm = prng.permutation((int)uniqueMols.size());

  std::set<int> entSet(a.ids.moleculeEntity.begin(), a.ids.moleculeEntity.end());
  const vector<int> uniqueEntities(entSet.begin(), entSet.end());
  std::map<int, int> entityToIdx;
  for (size_t i = 0; i < uniqueEntities.size(); ++i) entityToIdx[uniqueEntities[i]] = (int)i;
  vector<int> entityIds(nAtoms);
  for (size_t i = 0; i < nAtoms; ++i) entityIds[i] = entityToIdx[a.ids.moleculeEntity[i]];

  vector<string> cSeqNames(nAtoms, "<M>");
  auto cSeqIt = a.ann.strings.find("condition_sequence_1_residue");
  if (cSeqIt != a.ann.strings.end() && cSeqIt->second.size() == nAtoms) cSeqNames = cSeqIt->second;
  if (isInference) {
    int nextEntity = (int)uniqueEntities.size();
    vector<string> maskedChains;
    for (size_t i = 0; i < nAtoms; ++i) {
      if (cSeqNames[i] != "<M>") continue;
      const string& cid = a.ids.chainIid[i];
      if (std::find(maskedChains.begin(), maskedChains.end(), cid) == maskedChains.end()) {
        maskedChains.push_back(cid);
      }
    }
    std::sort(maskedChains.begin(), maskedChains.end());
    for (const string& cid : maskedChains) {
      for (size_t i = 0; i < nAtoms; ++i) if (a.ids.chainIid[i] == cid) entityIds[i] = nextEntity;
      ++nextEntity;
    }
  }

  vector<int> atomTokenType(nAtoms, tables::TT_UNKNOWN);
  for (size_t i = 0; i < nAtoms; ++i) {
    if (a.isStandardUnknownOrMaskedDna[i]) atomTokenType[i] = tables::TT_DNA;
    if (a.isStandardUnknownOrMaskedRna[i]) atomTokenType[i] = tables::TT_RNA;
    if (a.isStandardUnknownOrMaskedAa[i]) atomTokenType[i] = tables::TT_PROTEIN;
    auto it = tables::TOKEN_TO_IDX.find(a.site.resName[i]);
    if (it != tables::TOKEN_TO_IDX.end()) {
      const int tid = tables::TOKEN_GROUP_TYPE[it->second];
      if (tid != tables::TT_UNKNOWN && tid >= 0) atomTokenType[i] = tid;
    }
    if (a.atomize[i]) atomTokenType[i] = tables::TT_ATOMIZED;
  }

  // `is_resolved_mask` spreads over a token with ALL, not ANY - one unobserved atom makes the
  // whole token unresolved, where the context and sequence masks spread with ANY.
  vector<uint8_t> atomResolved(nAtoms);
  for (size_t i = 0; i < nAtoms; ++i) {
    atomResolved[i] = a.site.occupancy[i] > 0.0 && !std::isnan(a.site.x[i]) &&
                      !std::isnan(a.site.y[i]) && !std::isnan(a.site.z[i]);
  }
  auto idxIt = a.ann.ints.find("condition_index_1_residue");
  vector<uint8_t> unindexed(nAtoms, 0);
  if (idxIt != a.ann.ints.end() && idxIt->second.size() == nAtoms) {
    for (size_t i = 0; i < nAtoms; ++i) unindexed[i] = idxIt->second[i] ? 0 : 1;
  }
  vector<uint8_t> resolvedSpread(nAtoms, 0), inContextSpread(nAtoms, 0);
  for (int t = 0; t < nTokens; ++t) {
    const int s = tokStarts[t], e = tokStarts[t + 1];
    bool all = true, any = false;
    for (int i = s; i < e; ++i) { all = all && atomResolved[i]; any = any || unindexed[i]; }
    for (int i = s; i < e; ++i) { resolvedSpread[i] = all; inContextSpread[i] = any; }
  }

  // `encodeResNames(..., '<A>')`: an atomised atom encodes as `<A>` whatever it is called, and
  // a masked non-atomised residue stays `<M>` even where the residue name would resolve.
  const int idxAtomized = tables::TOKEN_TO_IDX.count("<A>") ? tables::TOKEN_TO_IDX.at("<A>") : 0;
  const int idxMasked = tables::TOKEN_TO_IDX.count("<M>") ? tables::TOKEN_TO_IDX.at("<M>") : 0;
  auto encode = [&](const string& name, size_t i) {
    if (a.atomize[i]) return idxAtomized;
    auto it = tables::TOKEN_TO_IDX.find(name);
    return it == tables::TOKEN_TO_IDX.end() ? idxAtomized : it->second;
  };
  vector<int> atomResidueType(nAtoms), atomCSeq(nAtoms);
  for (size_t i = 0; i < nAtoms; ++i) {
    atomResidueType[i] = encode(a.site.resName[i], i);
    atomCSeq[i] = encode(cSeqNames[i], i);
    if (cSeqNames[i] == "<M>" && !a.atomize[i]) atomCSeq[i] = idxMasked;
  }

  // 🔴 RECOMPUTED HERE, ON THE CENTRED ATOMS, because that is where the reference calls it -
  // the chirality fit reads coordinates, and the gate's own earlier dump ran it before the
  // centring. Same function, different input, different sixth figure.
  const chirality::Features chir =
      chirality::features(a.site, a.bonds, a.ann, a.stereo, a.tokenId, a.atomize);

  auto crdIt = a.ann.floats.find("condition_coordinate_1_atom");
  const vector<double>* cCrd =
      crdIt != a.ann.floats.end() && crdIt->second.size() == nAtoms * 3 ? &crdIt->second : nullptr;
  auto intIt = a.ann.floats.find("condition_InternalCoordinate_1_atom");
  const vector<double>* cInt =
      intIt != a.ann.floats.end() && intIt->second.size() == nAtoms * 3 ? &intIt->second : nullptr;
  auto cenIt = a.ann.floats.find("condition_ChainCentroid_1_atom");
  const vector<double>* cCen =
      cenIt != a.ann.floats.end() && cenIt->second.size() == nAtoms * 3 ? &cenIt->second : nullptr;
  auto intCol = [&](const char* key) -> const vector<int>* {
    auto it = a.ann.ints.find(key);
    return it != a.ann.ints.end() && it->second.size() == nAtoms ? &it->second : nullptr;
  };
  const vector<int>* cSec = intCol("condition_secondary-structure_1_residue");
  const vector<int>* cGrp = intCol("condition_index-group_1_token");
  vector<int> derivedGrp;
  if (!cGrp) { derivedGrp = atoms::defaultIndexGroups(a); cGrp = &derivedGrp; }
  const vector<int>* cCrdGrp = intCol("condition_CoordinateGroup_1_atom");
  const vector<int>* cHot = intCol("condition_hotspot_1_atom");
  const vector<int>* cRas = intCol("condition_rasa-binned_1_atom");
  const vector<int>* cHb = intCol("condition_hbond_1_atom");
  const vector<int>* cPiStack = intCol("condition_pistack_1_atom");
  const vector<int>* cPiCation = intCol("condition_pication_1_atom");
  const vector<int>* cSalt = intCol("condition_saltbridge_1_atom");
  const vector<int>* cHphob = intCol("condition_hydrophobic_1_atom");
  const vector<int>* cMcoord = intCol("condition_metal-coord_1_atom");
  auto cathIt = a.ann.strings.find("condition_cath-label_1_residue");
  const vector<string>* cCath =
      cathIt != a.ann.strings.end() && cathIt->second.size() == nAtoms ? &cathIt->second : nullptr;

  for (int t = 0; t < nTokens; ++t) {
    const int i = tokStarts[t];
    b.residueType[t] = atomResidueType[i];
    b.conditionSequence[t] = atomCSeq[i];
    b.tokenType[t] = atomTokenType[i];
    b.tokIdx[t] = a.tokenId[i];
    b.chainIid[t] = chainPerm[chainToIdx[a.ids.chainIid[i]]];
    b.moleculeIid[t] = molPerm[molToIdx[a.ids.moleculeIid[i]]];
    b.moleculeEntityId[t] = entityIds[i];
    b.supportsLocalLatent[t] = a.supportsLocalLatent[i];
    b.isInContextConditioning[t] = inContextSpread[i];
    b.isResolvedMask[t] = resolvedSpread[i];
    b.insertionSourceIdx[t] = a.insertionSourceIdx.size() == nAtoms ? a.insertionSourceIdx[i] : -1;
    b.coordCondMask[t] = cCrd && std::isfinite((*cCrd)[i * 3]) &&
                         std::isfinite((*cCrd)[i * 3 + 1]) && std::isfinite((*cCrd)[i * 3 + 2]);
    b.secondaryStructure[t] = cSec ? (*cSec)[i] : 0;
    b.indexGroup[t] = (*cGrp)[i];
    if (cCath) {
      // `parseCathCode`: the LONGEST progressive key the maps can spell wins, and a part
      // that is `x` stops the walk - so `3.x.50` is the class alone, not the class and the
      // topology. `CATH_CODES` already holds each key's whole triple.
      const string& code = (*cCath)[i];
      if (!code.empty() && code != "x.x.x.x" && code != "null") {
        string key;
        size_t pos = 0;
        for (int part = 0; part < 3; ++part) {
          const size_t dot = code.find('.', pos);
          const string piece = code.substr(pos, dot == string::npos ? string::npos : dot - pos);
          if (piece.empty() || piece == "x") break;
          const string next = key.empty() ? piece : key + "." + piece;
          auto hit = tables::CATH_CODES.find(next);
          if (hit == tables::CATH_CODES.end()) break;
          key = next;
          for (int k = 0; k < 3; ++k) b.cathLabel[(size_t)t * 3 + k] = hit->second[k];
          if (dot == string::npos) break;
          pos = dot + 1;
        }
      }
    }
    if (cCen) {
      for (int k = 0; k < 3; ++k) b.chainCentroid[(size_t)t * 3 + k] = (float)(*cCen)[i * 3 + k];
    }
    for (int k = 0; k < 18; ++k) b.chirality[(size_t)t * 18 + k] = chir.chirality[(size_t)i * 18 + k];
    for (int k = 0; k < 48; ++k) {
      b.chiralFrameAtoms[(size_t)t * 48 + k] = chir.frameAtoms[(size_t)i * 48 + k];
    }
    for (int k = 0; k < 6; ++k) b.chiralSlotOwner[(size_t)t * 6 + k] = chir.slotOwner[(size_t)i * 6 + k];
    for (int k = 0; k < 4; ++k) {
      b.tetrahedralVertices[(size_t)t * 4 + k] = chir.tetrahedralVertices[(size_t)i * 4 + k];
    }
  }

  const vector<int> slots = atoms::buildAtom37Slots(a);
  for (size_t i = 0; i < nAtoms; ++i) {
    const int s = slots[i];
    if (s < 0) continue;
    const size_t ts = (size_t)a.tokenId[i] * 37 + s;
    b.atom37Mask[ts] = 1;
    b.coords[ts * 3] = (float)a.coordToBeNoisedX[i];
    b.coords[ts * 3 + 1] = (float)a.coordToBeNoisedY[i];
    b.coords[ts * 3 + 2] = (float)a.coordToBeNoisedZ[i];
    const int z = a.site.atomicNumber[i];
    b.groundTruthElement[ts] = z;
    b.element[ts] = cSeqNames[i] != "<M>" ? z : -1;
    b.charge[ts] = a.site.charge[i];
    if (cCrd) for (int k = 0; k < 3; ++k) b.condCoord[ts * 3 + k] = (float)(*cCrd)[i * 3 + k];
    if (cInt) for (int k = 0; k < 3; ++k) b.condInternalCoord[ts * 3 + k] = (float)(*cInt)[i * 3 + k];
    b.conditionIndex[ts] = idxIt != a.ann.ints.end() ? (idxIt->second[i] ? 1 : 0) : 1;
    b.coordGroup[ts] = cCrdGrp ? (*cCrdGrp)[i] : -1;
    b.hotspot[ts] = cHot && (*cHot)[i] ? 1 : 0;
    b.rasaBinned[ts] = cRas ? (*cRas)[i] : 0;
    b.hbond[ts] = cHb ? (*cHb)[i] : 0;
    b.pistack[ts] = cPiStack && (*cPiStack)[i] ? 1 : 0;
    b.pication[ts] = cPiCation ? (*cPiCation)[i] : 0;
    b.saltbridge[ts] = cSalt ? (*cSalt)[i] : 0;
    b.hydrophobic[ts] = cHphob && (*cHphob)[i] ? 1 : 0;
    b.metalCoord[ts] = cMcoord ? (*cMcoord)[i] : 0;
  }
  // 🔴 A BOND COUNTS AT BOTH ENDS AND IS NOT DEDUPLICATED. Each end increments its own slot's
  // counter for that bond type, so a bond within one slot's atom counts twice there.
  for (const cif::Bond& bond : a.bonds) {
    for (const int end : {bond.u, bond.v}) {
      const int s = slots[end];
      if (s < 0) continue;
      b.incidentBondTypeCounts[((size_t)a.tokenId[end] * 37 + s) * 10 + bond.type] += 1;
    }
  }
  for (size_t i = 0; i < b.coords.size(); ++i) b.coordsNm[i] = b.coords[i] / 10.0f;

  const atoms::TokenBondFeatures tb = atoms::computeTokenBondFeatures(a, pathLengthCutoff,
                                                                      isInference);
  b.bondMask = tb.bondMask;
  b.isAromatic = tb.isAromatic;
  b.canonicalPolymerOrdering = tb.canonical;
  b.bondOrder.assign(tb.bondOrder.begin(), tb.bondOrder.end());
  b.tokenBondPathLength.assign(tb.pathLength.begin(), tb.pathLength.end());

  for (int k = 0; k < 28; ++k) {
    const string key =
        string("condition_sequence-probability-") + tables::STANDARD_POLYMER_RESIDUES[k] +
        "_1_residue";
    auto it = a.ann.floats.find(key);
    for (int t = 0; t < nTokens; ++t) {
      b.seqProb[(size_t)k * nTokens + t] =
          it != a.ann.floats.end() && it->second.size() == nAtoms
              ? (float)it->second[tokStarts[t]]
              : std::nanf("");
    }
  }
  return b;
}

inline Batch buildSpecBatch(const vector<Segment>& segments,
                            const std::map<string, std::map<string, vector<string>>>& conditions,
                            int seed) {
  // 🔴 A RANGED SEGMENT'S LENGTH IS A DRAW, AND THE ORDER IS THE SPECIFICATION.
  // `realizeExpandableSegments` takes `nextInt(min, max + 1)` once per segment in CONTIG order and
  // nothing else on this path touches that stream, so the whole of the pipeline's randomness is
  // these few integers. (`convertToProteinaBatch` asks for `permutation(1)` twice, whose
  // Fisher-Yates loop runs zero times and draws nothing.)
  PipelineRng prng((uint32_t)seed);
  vector<int> lengths, chainOf, typeOf;
  int n = 0, chains = 0;
  for (const Segment& s : segments) {
    if (s.chainBreak) { ++chains; lengths.push_back(0); continue; }
    // 🔴 THE DRAW HAPPENS EVEN WHEN min == max, and skipping it is not an optimisation: it moves
    // every later draw up one, so the CHAIN and MOLECULE permutations come out different. A
    // one-chain job hides it (those permutations are over one element and draw nothing) and a
    // two-chain one hid it by coincidence; three chains is where it showed.
    const int len = prng.nextInt(s.minLen, s.maxLen + 1);
    lengths.push_back(len);
    for (int k = 0; k < len; ++k) { chainOf.push_back(chains); typeOf.push_back(s.tokenType); }
    n += len;
  }
  ++chains;
  Batch b;
  b.resize(n);
  // 🔴 A CHAIN BREAK BREAKS THE BONDED RUN, so the path length across it is ZERO rather than a
  // large number - there is no path, and the reference leaves the entry at its fill. The
  // canonical ordering stops at the boundary for the same reason.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (chainOf[i] != chainOf[j]) continue;
      b.tokenBondPathLength[(size_t)i * n + j] = std::abs(i - j);
      b.canonicalPolymerOrdering[(size_t)i * n + j] = j == i + 1 ? -1 : (j == i - 1 ? 1 : 0);
    }
  }
  // 🔴 THE CHAIN AND MOLECULE IDS ARE PERMUTED, AND THE DRAWS COME AFTER THE SEGMENT LENGTHS.
  // `convertToProteinaBatch` asks the pipeline's rng for a permutation of the chains and then one
  // of the molecules; with a single chain each is `permutation(1)`, whose Fisher-Yates loop runs
  // zero times and draws nothing, which is why a one-chain job is seed-independent past its
  // lengths. With two it is two draws, and getting their ORDER wrong relabels the chains.
  const vector<int> chainPerm = prng.permutation(chains);
  const vector<int> molPerm = prng.permutation(chains);   // one molecule a chain for a pure contig
  for (int i = 0; i < n; ++i) {
    b.tokenType[i] = typeOf[i];
    b.chainIid[i] = chainPerm[chainOf[i]];
    b.moleculeIid[i] = molPerm[chainOf[i]];
    b.indexGroup[i] = chainOf[i];
    // `molecule_entity_id` at inference: every chain whose sequence is all `<M>` - which is every
    // chain of a contig - takes a FRESH entity after the ones the input had, in chain order.
    b.moleculeEntityId[i] = chains + chainOf[i];
  }
  // Which segment each token came from, so a `sel('*/<name>/*/*')` can select it.
  vector<string> owner(n);
  {
    int at = 0;
    for (size_t si = 0; si < segments.size(); ++si) {
      for (int k = 0; k < lengths[si]; ++k) owner[at++] = segments[si].name;
    }
  }
  applyConditions(b, conditions, owner);
  return b;
}

// ---------------------------------------------------------------- the writer
//
// The exporter's own `model.idx` / `model.bin`: `t <name> <offset> <count>` for floats, `i` for
// int32, `m meta/<key> <value>` for a scalar, offsets counted in ELEMENTS and the index written
// last by rename so a half-written directory is never readable.
struct Dir {
  string path;
  vector<string> lines;
  FILE* bin = nullptr;
  size_t offset = 0;
  explicit Dir(const string& p) : path(p) {
    string cmd = "mkdir -p '" + p + "'";
    if (system(cmd.c_str()) != 0) { fprintf(stderr, "cannot make %s\n", p.c_str()); exit(1); }
    bin = fopen((p + "/model.bin").c_str(), "wb");
    if (!bin) { fprintf(stderr, "cannot write %s/model.bin\n", p.c_str()); exit(1); }
  }
  void meta(const string& key, double value) {
    char buf[256];
    // The exporter writes JavaScript's own number formatting, which prints an integral double
    // without a fractional part; `%g` agrees for every value these keys take.
    snprintf(buf, sizeof buf, "m meta/%s %.17g", key.c_str(), value);
    lines.push_back(buf);
  }
  void tensor(const string& name, const vector<float>& v) {
    char buf[512];
    snprintf(buf, sizeof buf, "t %s %zu %zu", name.c_str(), offset, v.size());
    lines.push_back(buf);
    fwrite(v.data(), 4, v.size(), bin);
    offset += v.size();
  }
  void ints(const string& name, const vector<int>& v) {
    char buf[512];
    snprintf(buf, sizeof buf, "i %s %zu %zu", name.c_str(), offset, v.size());
    lines.push_back(buf);
    fwrite(v.data(), 4, v.size(), bin);
    offset += v.size();
  }
  void close() {
    fclose(bin);
    FILE* idx = fopen((path + "/model.idx.tmp").c_str(), "w");
    for (const string& l : lines) fprintf(idx, "%s\n", l.c_str());
    fclose(idx);
    rename((path + "/model.idx.tmp").c_str(), (path + "/model.idx").c_str());
  }
};

// ---------------------------------------------------------------- the sampler's starting state
//
// `flowEmitMasks` and `sample_noise`, as flow.js has them. Both halves of the noise come out of
// the port's own `Rng`, which is bit-identical to `seededRng` now that jsmath carries V8's log.
struct Emit {
  vector<uint8_t> bb, ll;
};
inline Emit emitMasks(const Batch& b) {
  Emit e;
  e.bb.assign(b.n, 0);
  e.ll.assign(b.n, 0);
  for (int i = 0; i < b.n; ++i) {
    const bool live = b.paddingMask[i] && !b.isInContextConditioning[i];
    e.bb[i] = live ? 1 : 0;
    e.ll[i] = (live && b.supportsLocalLatent[i]) ? 1 : 0;
  }
  return e;
}

// ---------------------------------------------------------------- the entry point
//
// 🔴 A FUNCTION, NOT A `main`, SO THE FOLD BINARY CAN CALL IT. `localfold-rfd4p --job=<spec>`
// links this object and featurises in its OWN process before the fold, which is what every other
// port here does (`lf::standalone::main`) - the alternative is a second process and a directory
// on disk between them.
int run(int argc, char** argv) {
  using lf::Json;
  using lf::parseJson;
  using rfd4p::Schedule;
  using rfd4p::Rng;
  using rfd4p::bbCaSchedule;
  using rfd4p::latentSchedule;
  using rfd4p::CENTERING_SCALE;
  using rfd4p::recycleDepths;
  string out, specPath, weightsDir, configPath, int8Dir, dumpCif;
  int splitAt = 0;
  bool realize = false, atomize = false, unannotated = false;
  string contigStr, selectStr, condJson;
  int seed = 0, step = 0, length = 0;
  bool generate = false;
  string recycleSchedule = "none";
  for (int i = 1; i < argc; ++i) {
    const string a = argv[i];
    if (a.rfind("--", 0) != 0) { out = a; continue; }
    if (!a.compare(0, 7, "--spec=")) specPath = a.substr(7);
    else if (!a.compare(0, 10, "--weights=")) weightsDir = a.substr(10);
    else if (!a.compare(0, 7, "--int8=")) int8Dir = a.substr(7);
    else if (!a.compare(0, 11, "--dump-cif=")) dumpCif = a.substr(11);
    else if (!a.compare(0, 8, "--split=")) splitAt = atoi(a.c_str() + 8);
    else if (a == "--realize") realize = true;
    else if (a == "--atomize") { realize = true; atomize = true; }
    else if (a == "--unannotated") unannotated = true;
    else if (!a.compare(0, 9, "--contig=")) contigStr = a.substr(9);
    else if (!a.compare(0, 9, "--select=")) selectStr = a.substr(9);
    else if (!a.compare(0, 13, "--conditions=")) condJson = a.substr(13);
    else if (!a.compare(0, 9, "--config=")) configPath = a.substr(9);
    else if (!a.compare(0, 7, "--seed=")) seed = atoi(a.c_str() + 7);
    else if (!a.compare(0, 7, "--step=")) step = atoi(a.c_str() + 7);
    else if (!a.compare(0, 9, "--length=")) length = atoi(a.c_str() + 9);
    else if (a == "--generate") generate = true;
    else if (!a.compare(0, 10, "--recycle=")) recycleSchedule = a.substr(10);
    else { fprintf(stderr, "unknown flag %s\n", a.c_str()); return 1; }
  }
  // 🔴 THE mmCIF READER, ON ITS OWN. The structure path is not finished - what this reads is
  // everything that comes STRAIGHT FROM THE FILE, and the chemistry the reference derives from it
  // (bonds, leaving atoms, chain types, the id and entity annotations) is still the exporter's.
  // Dumping what it does read is how that half gets a gate before the other half exists, rather
  // than landing unmeasured and being trusted because it compiles.
  if (!dumpCif.empty()) {
    bool opened = false;
    const string text = lf::zstd::readTextOrCompressed(dumpCif, &opened);
    if (!opened) { fprintf(stderr, "no %s\n", dumpCif.c_str()); return 1; }
    // `--contig=` runs `applyContig` over the structure and dumps what it assembled, which is
    // the one stage between the reader and the transforms.
    const rfd4p::cif::Block block = rfd4p::cif::parseBlock(text);
    rfd4p::cif::AtomSite a = rfd4p::cif::readAtomSite(block);
    rfd4p::cif::deriveChemistry(a, block);
    if (unannotated) {
      // `--unannotated` is the load a JOB does: `input_as_annotated_cif` defaults to false, so
      // the file's own conditions are discarded and the arginines are re-labelled.
      const rfd4p::atoms::AtomArray src = rfd4p::atoms::fromCif(text, false);
      printf("unann\t%zu\t%zu\t%zu\t%zu\n", src.n(), src.ann.ints.size(),
             src.ann.floats.size(), src.ann.strings.size());
      for (size_t i = 0; i < src.n(); ++i) {
        printf("ua\t%s\t%s\t%.17g\t%.17g\t%.17g\n", src.site.resName[i].c_str(),
               src.site.atomName[i].c_str(), src.site.x[i], src.site.y[i], src.site.z[i]);
      }
      return 0;
    }
    if (!selectStr.empty()) {
      // `--select=` evaluates one selection against the structure and dumps the mask, which is
      // the only way to hold the path language to the reference's without a whole job.
      rfd4p::atoms::AtomArray src = rfd4p::atoms::fromCif(text);
      const vector<uint8_t> m = rfd4p::selection::evalSelection(src, selectStr);
      printf("select\t%zu", m.size());
      for (const uint8_t v : m) printf("\t%d", (int)v);
      printf("\n");
      return 0;
    }
    if (!condJson.empty()) {
      // `--conditions=<json>` with `--contig=` runs the spec path over the structure: contig,
      // then conditions against the INPUT's residue numbers, and dumps every annotation they
      // wrote.
      rfd4p::atoms::AtomArray src = rfd4p::atoms::fromCif(text);
      vector<int> originalResId = src.site.resId;
      if (!contigStr.empty()) {
        const rfd4p::contig::Applied applied = rfd4p::contig::applyContig(src, contigStr);
        src = applied.atomArray;
        originalResId = applied.originalResId;
      }
      const Json cj = lf::parseJson(condJson);
      rfd4p::conditions::Spec spec;
      if (cj.isObject()) {
        for (const auto& [key, arms] : cj.o) {
          for (const auto& [value, sels] : arms.o) {
            vector<rfd4p::conditions::Entry> list;
            if (sels.isArray()) {
              for (const Json& e : sels.a) {
                rfd4p::conditions::Entry entry;
                const Json* sel = e.get("select");
                if (sel && sel->isString()) entry.select = sel->s;
                const Json* grp = e.get("group");
                if (grp && !Json::absent(grp)) {
                  entry.hasGroup = true;
                  entry.group = grp->isString() ? grp->s : std::to_string((long long)grp->n);
                }
                list.push_back(entry);
              }
            }
            auto sAt = std::find_if(spec.begin(), spec.end(),
                                    [&](const auto& kv) { return kv.first == key; });
            if (sAt == spec.end()) { spec.push_back({key, {}}); sAt = spec.end() - 1; }
            sAt->second.push_back({value, list});
          }
        }
      }
      rfd4p::conditions::applyConditions(src, spec, originalResId);
      printf("cond\t%zu\n", src.n());
      // 🔴 `res_id` TOO, because the whole function runs with the INPUT's numbering swapped in
      // and puts the assembled one back at the end. Dumping only the annotations leaves that
      // restore unobserved - removing it left every case green.
      printf("condresid");
      for (const int v : src.site.resId) printf("\t%d", v);
      printf("\n");
      for (const auto& [key, v] : src.ann.ints) {
        printf("condi\t%s", key.c_str());
        for (const int x : v) printf("\t%d", x);
        printf("\n");
      }
      for (const auto& [key, v] : src.ann.floats) {
        printf("condf\t%s", key.c_str());
        for (const double x : v) printf("\t%.17g", x);
        printf("\n");
      }
      for (const auto& [key, v] : src.ann.strings) {
        printf("conds\t%s", key.c_str());
        for (const string& x : v) printf("\t%s", x.c_str());
        printf("\n");
      }
      return 0;
    }
    if (!contigStr.empty()) {
      rfd4p::atoms::AtomArray src = rfd4p::atoms::fromCif(text);
      const rfd4p::contig::Applied applied = rfd4p::contig::applyContig(src, contigStr);
      const rfd4p::atoms::AtomArray& r = applied.atomArray;
      printf("contig\t%zu\n", r.n());
      auto seg = [&](const char* key) -> const vector<int>* {
        auto it = r.ann.ints.find(key);
        return it == r.ann.ints.end() ? nullptr : &it->second;
      };
      const vector<int>* mn = seg("annotation_expsegmin_1_residue");
      const vector<int>* mx = seg("annotation_expsegmax_1_residue");
      const vector<int>* uid = seg("_expseg_uid");
      // 🔴 TWO OF THE SENTINEL'S OTHER ANNOTATIONS, because it is SLICED from a real atom and
      // every annotation it inherits has to be reset to its default - without a column here
      // that reset is unobserved and a stub arrives conditioned like its neighbour.
      const vector<int>* cidx = seg("condition_index_1_residue");
      auto cseqIt = r.ann.strings.find("condition_sequence_1_residue");
      const vector<string>* cseq = cseqIt == r.ann.strings.end() ? nullptr : &cseqIt->second;
      for (size_t i = 0; i < r.n(); ++i) {
        printf("cg\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%.17g\t%.17g\t%.17g"
               "\t%d\t%s\n",
               r.site.resName[i].c_str(), r.site.atomName[i].c_str(), r.site.element[i].c_str(),
               r.site.chainId[i].c_str(), r.site.resId[i], r.site.chainType[i],
               mn && i < mn->size() ? (*mn)[i] : -1, mx && i < mx->size() ? (*mx)[i] : -1,
               uid && i < uid->size() ? (*uid)[i] : -1, applied.originalResId[i],
               r.ids.chainIid[i].c_str(), r.site.x[i], r.site.y[i], r.site.z[i],
               cidx && i < cidx->size() ? (*cidx)[i] : -99,
               cseq && i < cseq->size() ? (*cseq)[i].c_str() : "?");
      }
      for (const rfd4p::cif::Bond& b : r.bonds) printf("cgbond\t%d\t%d\t%d\n", b.u, b.v, b.type);
      return 0;
    }
    printf("n %zu\n", a.n);
    for (size_t i = 0; i < a.n; ++i) {
      printf("%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g"
             "\t%d\t%d\t%d\t%d\t%d\t%s\n",
             a.element[i].c_str(), a.atomName[i].c_str(), a.resName[i].c_str(),
             a.chainId[i].c_str(), a.insCode[i].c_str(), a.labelAltId[i].c_str(), a.resId[i],
             a.authSeqId[i], a.atomId[i], a.labelEntityId[i], (int)a.hetero[i], a.x[i],
             a.y[i], a.z[i], a.bFactor[i], a.occupancy[i],
             a.charge[i], a.atomicNumber[i], (int)a.isAromatic[i], (int)a.isLeavingAtom[i],
             a.chainType[i], a.chemCompType[i].c_str());
    }
    const vector<rfd4p::cif::Bond> bonds = rfd4p::cif::buildBonds(a, block);
    for (const rfd4p::cif::Bond& b : bonds) printf("bond\t%d\t%d\t%d\n", b.u, b.v, b.type);
    const rfd4p::cif::Ids ids = rfd4p::cif::computeIds(a, bonds);
    for (size_t i = 0; i < a.n; ++i) {
      printf("id\t%s\t%d\t%d\t%d\t%d\t%s\t%s\t%d\n", ids.pnUnitId[i].c_str(),
             ids.moleculeId[i], ids.chainEntity[i], ids.pnUnitEntity[i], ids.moleculeEntity[i],
             ids.chainIid[i].c_str(), ids.pnUnitIid[i].c_str(), ids.moleculeIid[i]);
    }
    const rfd4p::cif::Annotations ann = rfd4p::cif::readAnnotations(a, block, a.selected);
    for (const auto& [name, v] : ann.ints) {
      printf("ann\ti\t%s", name.c_str());
      for (int x : v) printf("\t%d", x);
      printf("\n");
    }
    for (const auto& [name, v] : ann.floats) {
      printf("ann\tf\t%s", name.c_str());
      for (double x : v) printf("\t%.17g", x);
      printf("\n");
    }
    for (const auto& [name, v] : ann.strings) {
      printf("ann\ts\t%s", name.c_str());
      for (const auto& x : v) printf("\t%s", x.c_str());
      printf("\n");
    }
    const rfd4p::chirality::Stereo st = rfd4p::chirality::annotate(a, bonds, ann, {});
    printf("chiral");
    for (int t : st.chiralType) printf("\t%d", t);
    printf("\nstereobond");
    for (uint8_t b : st.stereogenicBondAtoms) printf("\t%d", (int)b);
    printf("\n");
    const rfd4p::chirality::Features cf = rfd4p::chirality::features(a, bonds, ann, st, {}, {});
    printf("chirfeat");
    for (float v : cf.chirality) printf("\t%.9g", (double)v);
    printf("\nchircenter");
    for (uint8_t v : cf.isChiralCenter) printf("\t%d", (int)v);
    printf("\nchirframe");
    for (int v : cf.frameAtoms) printf("\t%d", v);
    printf("\nchirowner");
    for (int v : cf.slotOwner) printf("\t%d", v);
    printf("\nchirtetra");
    for (int v : cf.tetrahedralVertices) printf("\t%d", v);
    printf("\n");
    // The first transform stages, which a reader can run without the rest of the pipeline.
    rfd4p::atoms::AtomArray arr;
    arr.site = a;
    arr.ann = ann;
    arr.bonds = bonds;
    arr.ids = ids;
    arr.stereo = st;
    // 🔴 THE ROUND TRIP IS THE SLICE/CONCAT GATE. Those two are the only primitives the rest of
    // the pipeline has for inserting and dropping atoms, and a column they forget keeps its OLD
    // values through an insertion - a wrong answer in a field nothing else looks at. Cutting the
    // array in two and putting it back must reproduce it exactly, which `--split=K` asks for and
    // the checker diffs against the un-split dump.
    if (splitAt > 0 && splitAt < (int)arr.n()) {
      arr = rfd4p::atoms::concat({rfd4p::atoms::slice(arr, 0, splitAt),
                                  rfd4p::atoms::slice(arr, splitAt, (int)arr.n())});
    }
    rfd4p::atoms::ensureChemicalFlags(arr);
    rfd4p::atoms::addSrcComponentAnnotation(arr);
    rfd4p::atoms::computeWithinChainSourceResIdx(arr);
    if (realize) {
      rfd4p::atoms::PipelineRng prng((uint32_t)seed);
      rfd4p::atoms::realizeExpandableSegments(arr, prng);
      if (atomize) {
        // 🔴 THE WHOLE DRIVER, NOT A HAND-WRITTEN SEQUENCE. `runValidationPipeline` owns the
        // order - including the SECOND `addGlobalTokenIdAnnotation`, which a hand-written
        // sequence here dropped and the gate could not see, because the gate's JavaScript side
        // was sequenced by the same hand. Both sides drive their own driver now.
        const rfd4p::featurise::Batch sb = rfd4p::featurise::runValidationPipeline(arr, prng);
        const rfd4p::atoms::Centroid centroid{{sb.centroidWithoutCond[0],
                                               sb.centroidWithoutCond[1],
                                               sb.centroidWithoutCond[2]}};
        const vector<int> slots = rfd4p::atoms::buildAtom37Slots(arr);
        printf("atomized\t%zu\n", arr.n());
        printf("azcentroid\t%.17g\t%.17g\t%.17g\n", (double)centroid.v[0], (double)centroid.v[1],
               (double)centroid.v[2]);
        {
          // The batch `runValidationPipeline` returned - not a second one built beside it,
          // which would draw its two permutations again from a stream the first had advanced.
          printf("azbatch\t%d\n", sb.n);
          const auto ints = [&](const char* name, const auto& v) {
            long long sum = 0, nz = 0, weighted = 0;
            for (size_t i = 0; i < v.size(); ++i) {
              const long long x = (long long)v[i];
              sum += x;
              if (x) { ++nz; weighted += x * (long long)(i % 1000003); }
            }
            printf("azbatch\t%s\t%zu\t%lld\t%lld\t%lld\n", name, v.size(), sum, nz, weighted);
          };
          const auto reals = [&](const char* name, const vector<float>& v) {
            // NaN is COUNTED rather than summed: these columns are mostly NaN by design, and a
            // sum over them says only "NaN" however wrong the finite entries are.
            double sum = 0;
            long long nan = 0, nz = 0;
            for (size_t i = 0; i < v.size(); ++i) {
              if (std::isnan(v[i])) { ++nan; continue; }
              sum += (double)v[i] * (double)((i % 1000003) + 1);
              if (v[i] != 0.f) ++nz;
            }
            printf("azbatch\t%s\t%zu\t%.9g\t%lld\t%lld\n", name, v.size(), sum, nan, nz);
          };
          ints("residue_type", sb.residueType);
          ints("condition_sequence", sb.conditionSequence);
          ints("token_type", sb.tokenType);
          ints("tok_idx", sb.tokIdx);
          ints("chain_iid", sb.chainIid);
          ints("molecule_iid", sb.moleculeIid);
          ints("molecule_entity_id", sb.moleculeEntityId);
          ints("supports_local_latent", sb.supportsLocalLatent);
          ints("is_in_context_conditioning", sb.isInContextConditioning);
          ints("is_resolved_mask", sb.isResolvedMask);
          ints("insertion_source_idx", sb.insertionSourceIdx);
          ints("coord_cond_mask", sb.coordCondMask);
          ints("condition_secondary_structure", sb.secondaryStructure);
          ints("condition_index_group", sb.indexGroup);
          ints("cath_label", sb.cathLabel);
          ints("atom37_mask", sb.atom37Mask);
          ints("element", sb.element);
          ints("ground_truth_element", sb.groundTruthElement);
          ints("charge", sb.charge);
          ints("incident_bond_type_counts", sb.incidentBondTypeCounts);
          ints("condition_index", sb.conditionIndex);
          ints("condition_coordinate_group", sb.coordGroup);
          ints("condition_hotspot", sb.hotspot);
          ints("condition_rasa", sb.rasaBinned);
          ints("condition_hbond", sb.hbond);
          ints("condition_pistack", sb.pistack);
          ints("condition_pication", sb.pication);
          ints("condition_saltbridge", sb.saltbridge);
          ints("condition_hydrophobic", sb.hydrophobic);
          ints("condition_metal_coord", sb.metalCoord);
          ints("chiral_frame_atoms", sb.chiralFrameAtoms);
          ints("chiral_slot_owner", sb.chiralSlotOwner);
          ints("tetrahedral_vertices", sb.tetrahedralVertices);
          ints("bond_mask", sb.bondMask);
          ints("bond_order", sb.bondOrder);
          ints("is_aromatic", sb.isAromatic);
          ints("token_bond_path_length", sb.tokenBondPathLength);
          ints("canonical_polymer_ordering", sb.canonicalPolymerOrdering);
          reals("coords", sb.coords);
          reals("coords_nm", sb.coordsNm);
          reals("condition_coordinate", sb.condCoord);
          reals("condition_internal_coordinate", sb.condInternalCoord);
          reals("condition_chain_centroid", sb.chainCentroid);
          reals("chirality", sb.chirality);
          reals("loss_weight", sb.lossWeight);
          reals("seq_prob", sb.seqProb);
          printf("azbatch\tcentroid\t3\t%.17g\t%.17g\t%.17g\n", (double)sb.centroidWithoutCond[0],
                 (double)sb.centroidWithoutCond[1], (double)sb.centroidWithoutCond[2]);
        }
        {
          const rfd4p::atoms::TokenBondFeatures tb = rfd4p::atoms::computeTokenBondFeatures(arr);
          printf("aztb\t%zu\n", tb.n);
          // 🔴 THE PLANES AS CHECKSUMS AND AS NON-ZERO ENTRIES, not element by element: at 924
          // atoms these are five 924x924 matrices and printing them whole is 4 million numbers
          // a structure. A checksum that weights by POSITION catches a transpose, which a sum
          // would not.
          const auto plane = [&](const char* name, auto get) {
            long long sum = 0, nz = 0, weighted = 0;
            for (size_t i = 0; i < tb.n * tb.n; ++i) {
              const long long v = (long long)get(i);
              sum += v;
              if (v) { ++nz; weighted += v * (long long)(i % 1000003); }
            }
            printf("aztb\t%s\t%lld\t%lld\t%lld\n", name, sum, nz, weighted);
          };
          plane("bond_mask", [&](size_t i) { return tb.bondMask[i]; });
          plane("bond_order", [&](size_t i) { return tb.bondOrder[i]; });
          plane("is_aromatic", [&](size_t i) { return tb.isAromatic[i]; });
          plane("path_length", [&](size_t i) { return tb.pathLength[i]; });
          plane("canonical", [&](size_t i) { return tb.canonical[i]; });
        }
        for (const char* key : {"condition_coordinate_1_atom", "condition_ChainCentroid_1_atom"}) {
          auto it = arr.ann.floats.find(key);
          if (it == arr.ann.floats.end()) continue;
          printf("azshift\t%s", key);
          for (double v : it->second) printf("\t%.17g", v);
          printf("\n");
        }
        for (size_t i = 0; i < arr.n(); ++i) {
          printf("azfin\t%.17g\t%.17g\t%.17g\t%d\t%d\n", arr.coordToBeNoisedX[i],
                 arr.coordToBeNoisedY[i], arr.coordToBeNoisedZ[i],
                 (int)arr.supportsLocalLatent[i], slots[i]);
        }
        for (size_t i = 0; i < arr.n(); ++i) {
          printf("azun\t%d\t%d\t%s\n", arr.insertionSourceIdx[i], (int)arr.isUnindexedSource[i],
                 arr.ids.chainIid[i].c_str());
        }
        {
          const vector<int>& g = arr.ann.ints["condition_index-group_1_token"];
          printf("azgrp");
          for (int v : g) printf("\t%d", v);
          printf("\n");
        }
        {
          const vector<int>& g = arr.ann.ints["condition_CoordinateGroup_1_atom"];
          const vector<double>& ic = arr.ann.floats["condition_InternalCoordinate_1_atom"];
          for (size_t i = 0; i < arr.n(); ++i) {
            printf("azic\t%d\t%.17g\t%.17g\t%.17g\n", i < g.size() ? g[i] : -1,
                   ic.size() == arr.n() * 3 ? ic[i * 3] : NAN,
                   ic.size() == arr.n() * 3 ? ic[i * 3 + 1] : NAN,
                   ic.size() == arr.n() * 3 ? ic[i * 3 + 2] : NAN);
          }
        }
        for (size_t i = 0; i < arr.n(); ++i) {
          printf("azxyz\t%.17g\t%.17g\t%.17g\n", arr.site.x[i], arr.site.y[i], arr.site.z[i]);
        }
        for (const string& s : arr.ann.strings["condition_sequence_1_residue"]) {
          printf("azseq\t%s\n", s.c_str());
        }
        // 🔴 THE COLUMNS THAT EXIST, NAMED - because the stage DELETES one that ends up entirely
        // NaN, and a comparison over a fixed list of 28 cannot tell an absent column from a
        // present one full of NaN.
        for (const char* tok : rfd4p::tables::STANDARD_POLYMER_RESIDUES) {
          const string key = string("condition_sequence-probability-") + tok + "_1_residue";
          auto it = arr.ann.floats.find(key);
          if (it == arr.ann.floats.end()) continue;
          printf("azprob\t%s", tok);
          for (double v : it->second) printf("\t%.17g", v);
          printf("\n");
        }
        for (size_t i = 0; i < arr.n(); ++i) {
          printf("az\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\n", arr.site.resName[i].c_str(),
                 arr.site.atomName[i].c_str(), arr.site.element[i].c_str(), arr.site.resId[i],
                 arr.atomId[i], arr.tokenId[i], (int)arr.atomize[i],
                 (int)arr.isCovalentModification[i]);
        }
        for (const rfd4p::cif::Bond& b : arr.bonds) {
          printf("azbond\t%d\t%d\t%d\n", b.u, b.v, b.type);
        }
        return 0;
      }
      printf("realized\t%zu\n", arr.n());
      // 🔴 `_expseg_uid` IS ABSENT WHERE THERE WAS NOTHING TO EXPAND, because the stage returns
      // before creating it - which is the reference's behaviour and which `operator[]` on a map
      // turns into an empty vector and a read past its end. Asking the map first is the fix; a
      // segfault on the one shipped structure with no expandable segment was the symptom.
      auto uidIt = arr.ann.ints.find("_expseg_uid");
      const vector<int>* uid = uidIt == arr.ann.ints.end() ? nullptr : &uidIt->second;
      for (size_t i = 0; i < arr.n(); ++i) {
        printf("rz\t%s\t%s\t%s\t%d\t%d\t%d\t%.17g\t%.17g\t%.17g\t%.17g\n",
               arr.site.resName[i].c_str(), arr.site.atomName[i].c_str(),
               arr.site.element[i].c_str(), arr.site.resId[i], arr.site.authSeqId[i],
               uid && i < uid->size() ? (*uid)[i] : -1, arr.site.x[i], arr.site.y[i],
               arr.site.z[i],
               arr.site.occupancy[i]);
      }
      for (const rfd4p::cif::Bond& b : arr.bonds) {
        printf("rzbond\t%d\t%d\t%d\n", b.u, b.v, b.type);
      }
      return 0;
    }
    const std::pair<const char*, const vector<uint8_t>*> flags[] = {
        {"is_protein", &arr.isProtein}, {"is_rna", &arr.isRna}, {"is_dna", &arr.isDna},
        {"is_nucleic_acid", &arr.isNucleicAcid},
        {"is_protein_backbone", &arr.isProteinBackbone},
        {"is_rna_backbone", &arr.isRnaBackbone}, {"is_dna_backbone", &arr.isDnaBackbone},
        {"is_nucleic_acid_backbone", &arr.isNucleicAcidBackbone},
        {"is_standard_aa", &arr.isStandardAa},
        {"is_standard_or_unknown_aa", &arr.isStandardOrUnknownAa},
        {"is_standard_unknown_or_masked_aa", &arr.isStandardUnknownOrMaskedAa},
        {"is_standard_rna", &arr.isStandardRna},
        {"is_standard_or_unknown_rna", &arr.isStandardOrUnknownRna},
        {"is_standard_unknown_or_masked_rna", &arr.isStandardUnknownOrMaskedRna},
        {"is_standard_dna", &arr.isStandardDna},
        {"is_standard_or_unknown_dna", &arr.isStandardOrUnknownDna},
        {"is_standard_unknown_or_masked_dna", &arr.isStandardUnknownOrMaskedDna}};
    for (const auto& [name, v] : flags) {
      printf("flag\t%s", name);
      for (uint8_t x : *v) printf("\t%d", (int)x);
      printf("\n");
    }
    printf("srccomp");
    for (const auto& v : arr.ann.strings["annotation_src-component_1_atom"]) {
      printf("\t%s", v.c_str());
    }
    printf("\nwithinchain");
    for (int v : arr.withinChainSourceResIdx) printf("\t%d", v);
    printf("\n");
    for (const auto& [name, values] : a.extra) printf("extra\t%s\n", name.c_str());
    return 0;
  }

  if (out.empty() || (weightsDir.empty() && int8Dir.empty()) || (specPath.empty() && !length)) {
    fprintf(stderr, "usage: rfd4p-featurise <out dir> --weights=<dir> "
                    "(--spec=<job.json> | --length=N) [--config=<config.json>] [--seed=N] "
                    "[--step=N]\n");
    return 1;
  }
  // The bundle's own feature lists and widths, from the same config.json the port reads.
  if (configPath.empty()) {
    configPath = int8Dir.empty() ? "model-rfd4p-int8/config.json" : int8Dir + "/config.json";
  }
  string text;
  {
    FILE* f = fopen(configPath.c_str(), "rb");
    if (!f) { fprintf(stderr, "no %s\n", configPath.c_str()); return 1; }
    char buf[65536];
    size_t got;
    while ((got = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, got);
    fclose(f);
  }
  const Json root = parseJson(text);
  const Json* nn = root.get("config") ? root.get("config")->get("nn") : nullptr;
  if (!nn) { fprintf(stderr, "%s has no config.nn\n", configPath.c_str()); return 1; }
  auto strings = [](const Json* v) {
    vector<string> out;
    if (v && v->isArray()) for (const Json& e : v->a) out.push_back(e.s);
    return out;
  };
  auto num = [&](const char* k, double fallback) {
    const Json* v = nn->get(k);
    return v && v->isNumber() ? v->n : fallback;
  };
  Config cfg;
  cfg.featsSeq = strings(nn->get("feats_seq"));
  cfg.featsCondSeq = strings(nn->get("feats_cond_seq"));
  cfg.featsPairRepr = strings(nn->get("feats_pair_repr"));
  cfg.featsPairCond = strings(nn->get("feats_pair_cond"));
  cfg.seqSepDim = (int)num("seq_sep_dim", 64);
  cfg.xtPairDistDim = (int)num("xt_pair_dist_dim", 30);
  cfg.xtPairDistMin = num("xt_pair_dist_min", 0.1);
  cfg.xtPairDistMax = num("xt_pair_dist_max", 3.0);

  // The job. `--length=N` is one unconditioned segment, the shape the exporter's own flag makes.
  string contig;
  std::map<string, std::map<string, vector<string>>> conditions;
  // The structure path's conditions keep their ORDER and their `group:` field, which the spec
  // path's map-of-maps has nowhere to put.
  rfd4p::conditions::Spec structureConditions;
  string inputPath;
  bool asAnnotatedCif = false, specMode = false;
  int nsteps = 60;
  if (length) {
    contig = "a1:" + std::to_string(length) + "-" + std::to_string(length) + "P";
  } else {
    string job;
    FILE* f = fopen(specPath.c_str(), "rb");
    if (!f) { fprintf(stderr, "no %s\n", specPath.c_str()); return 1; }
    char buf[65536];
    size_t got;
    while ((got = fread(buf, 1, sizeof buf, f)) > 0) job.append(buf, got);
    fclose(f);
    const Json spec = parseJson(job);
    const Json* inputKey = spec.get("input");
    const bool hasInput = inputKey && inputKey->isString();
    // 🔴 A CONTIG IS REQUIRED ONLY WITHOUT AN INPUT. A spec that names a structure and no
    // contig means "take the whole structure"; the reference leaves `originalResId` as the
    // input's own numbering and carries on. Demanding one here refused a job the reference
    // accepts.
    if (!hasInput && Json::absent(spec.get("contig"))) {
      fprintf(stderr, "%s has no contig\n", specPath.c_str());
      return 1;
    }
    if (!Json::absent(spec.get("contig"))) contig = spec.get("contig")->s;
    if (!Json::absent(spec.get("nsteps"))) nsteps = (int)spec.get("nsteps")->n;
    for (const char* k : {"recycleSchedule", "recycle_schedule"}) {
      if (!Json::absent(spec.get(k))) recycleSchedule = spec.get(k)->s;
    }
    for (const char* k : {"input", "inputCif", "input_cif"}) {
      const Json* v = spec.get(k);
      if (v && v->isString()) inputPath = v->s;
    }
    {
      const Json* sm = spec.get("specMode");
      specMode = sm && !Json::absent(sm) ? (sm->kind == Json::Bool ? sm->b : sm->n != 0)
                                         : !hasInput;
    }
    for (const char* k : {"input_as_annotated_cif", "inputAsAnnotatedCif"}) {
      const Json* v = spec.get(k);
      if (v && !Json::absent(v)) asAnnotatedCif = v->kind == Json::Bool ? v->b : v->n != 0;
    }
    const Json* cond = spec.get("conditions");
    if (cond && cond->isObject()) {
      for (const auto& [key, arms] : cond->o) {
        rfd4p::conditions::Arms ordered;
        for (const auto& [value, sels] : arms.o) {
          vector<string> list;
          vector<rfd4p::conditions::Entry> entries;
          if (sels.isArray()) {
            for (const Json& e : sels.a) {
              rfd4p::conditions::Entry entry;
              const Json* sel = e.get("select");
              if (sel && sel->isString()) { entry.select = sel->s; list.push_back(sel->s); }
              const Json* grp = e.get("group");
              if (grp && !Json::absent(grp)) {
                entry.hasGroup = true;
                entry.group = grp->isString() ? grp->s : std::to_string((long long)grp->n);
              }
              entries.push_back(entry);
            }
          }
          conditions[key][value] = list;
          ordered.push_back({value, entries});
        }
        structureConditions.push_back({key, ordered});
      }
    }
  }

  // 🔴 TWO ROUTES TO ONE BATCH, AND THE SECOND IS THE WHOLE STRUCTURE PIPELINE.
  // `buildSpecBatch` is the contig-only specialisation: every token is a `<M>` stub, so most of
  // `convertToProteinaBatch` is a no-op and the batch is its initialisation values. A job that
  // names an input runs the real thing - read, contig, conditions, then the fourteen stages -
  // and they must not be allowed to drift apart, which is why the structure route calls
  // `runValidationPipeline` rather than assembling a batch of its own.
  Batch b;
  if (inputPath.empty()) {
    b = buildSpecBatch(parseContig(contig), conditions, seed);
  } else {
    bool opened = false;
    const string cifText = lf::zstd::readTextOrCompressed(inputPath, &opened);
    if (!opened) { fprintf(stderr, "no %s\n", inputPath.c_str()); return 1; }
    // 🔴 A STRUCTURE IS THE JOB UNLESS THE SPEC SAYS OTHERWISE, and the two routes disagree by
    // 67 tokens on the same file. Without `specMode`, `export_input.mjs` hands the CIF straight
    // to `buildBatch` - annotations KEPT, contig and conditions IGNORED - because a preset's
    // contig is written in the original entry's numbering and names residues the cropped
    // example does not contain. With `specMode: true` it is a spec: the annotations are
    // discarded, the arginines re-labelled, and the contig and conditions applied. Reading the
    // first as the second gave 116 tokens where the exporter gives 183.
    rfd4p::atoms::AtomArray src =
        rfd4p::atoms::fromCif(cifText, specMode ? asAnnotatedCif : true);
    vector<int> originalResId = src.site.resId;
    if (specMode) {
      if (!contig.empty()) {
        const rfd4p::contig::Applied applied = rfd4p::contig::applyContig(src, contig);
        src = applied.atomArray;
        originalResId = applied.originalResId;
      }
      if (!structureConditions.empty()) {
        rfd4p::conditions::applyConditions(src, structureConditions, originalResId);
      }
    }
    // 🔴 ONE GENERATOR FOR THE WHOLE PIPELINE, SEEDED FROM THE JOB. Neither the contig nor the
    // conditions draw; `runValidationPipeline` draws the segment lengths, the internal-coordinate
    // rotations and the two permutations, in that order, from this one stream.
    rfd4p::atoms::PipelineRng prng((uint32_t)seed);
    b = runValidationPipeline(src, prng);
  }
  const int n = b.n;

  // The sampler's first step, exactly as the exporter reaches it.
  const Schedule schedBb = bbCaSchedule(nsteps), schedLl = latentSchedule(nsteps);
  const double tBb = (float)schedBb.ts[step], tLl = (float)schedLl.ts[step];
  Rng rng((uint64_t)seed);
  const Emit emit = emitMasks(b);
  vector<float> xBb((size_t)n * 3), xLl((size_t)n * 32);
  rng.randn(xBb.data(), xBb.size());
  {
    double mean[3] = {0, 0, 0};
    int live = 0;
    for (int i = 0; i < n; ++i) {
      if (!emit.bb[i]) { xBb[i * 3] = xBb[i * 3 + 1] = xBb[i * 3 + 2] = 0.f; continue; }
      ++live;
      for (int k = 0; k < 3; ++k) mean[k] += xBb[i * 3 + k];
    }
    for (int k = 0; k < 3; ++k) mean[k] = (float)(mean[k] / std::max(live, 1));
    for (int i = 0; i < n; ++i) {
      if (!emit.bb[i]) continue;
      for (int k = 0; k < 3; ++k) xBb[i * 3 + k] = (float)(xBb[i * 3 + k] - (float)mean[k]);
    }
    vector<int> groups;
    for (int i = 0; i < n; ++i) {
      if (emit.bb[i] && !b.coordCondMask[i] &&
          std::find(groups.begin(), groups.end(), b.moleculeIid[i]) == groups.end()) {
        groups.push_back(b.moleculeIid[i]);
      }
    }
    std::sort(groups.begin(), groups.end());
    for (int g : groups) {
      float off[4];
      rng.randn(off, 3);
      for (int i = 0; i < n; ++i) {
        if (!emit.bb[i] || b.coordCondMask[i] || b.moleculeIid[i] != g) continue;
        for (int k = 0; k < 3; ++k) {
          // 🔴 THE SCALE IS A DOUBLE, AND WRITING IT `0.2f` IS A DIFFERENT MULTIPLIER. The
          // reference's `centeringScale` is JavaScript's 0.2 - the double - times a float drawn
          // from a Float32Array, rounded once at the store. As `0.2f` the product rounds twice
          // and lands one ulp out on about a fifth of tokens; at 53 and 100 residues it happened
          // to agree everywhere, and at 1, 2, 7 and 257 it did not. A sweep of lengths is what
          // separates "exact" from "exact on the length I tried".
          xBb[i * 3 + k] = (float)(xBb[i * 3 + k] + (float)(CENTERING_SCALE * (double)off[k]));
        }
      }
    }
    rng.randn(xLl.data(), xLl.size());
    for (int i = 0; i < n; ++i) {
      if (!emit.ll[i]) for (int k = 0; k < 32; ++k) xLl[(size_t)i * 32 + k] = 0.f;
    }
  }

  const std::set<string> wantedWeights = {
      "w/nn.init_repr_factory.feat_creators.2.embedding.weight",
      "w/nn.init_repr_factory.feat_creators.7.ss_embedding.weight",
      "w/nn.init_repr_factory.feat_creators.16.embedding.weight",
      "w/nn.init_repr_factory.feat_creators.17.embedding.weight",
      "w/nn.cond_factory.feat_creators.2.embedding_C.weight",
      "w/nn.cond_factory.feat_creators.2.embedding_A.weight",
      "w/nn.cond_factory.feat_creators.2.embedding_T.weight",
  };
  Weights w;
  if (!int8Dir.empty()) {
    std::set<string> bare;
    for (const string& k : wantedWeights) bare.insert(k.substr(2));   // drop the `w/`
    w = readBundleWeights(int8Dir, bare);
  } else {
    w = readWeights(weightsDir, wantedWeights);
  }

  int seqDim = 0, condDim = 0, pairDim = 0;
  const vector<float> seq = seqFeatures(b, cfg, w, xBb, xLl, seqDim);
  const vector<float> cond0 = condFeatures(b, cfg, w, tBb, tLl, -1, condDim);
  const vector<float> cond1 = condFeatures(b, cfg, w, tBb, tLl, 1, condDim);
  const vector<float> pair = pairFeatures(b, cfg, xBb, false, pairDim);
  const vector<float> pairCond = pairCondFeatures(cfg, tBb, tLl);

  // The decoder's shape, which the port sizes itself from; it lives beside the trunk's in the
  // bundle's config and is written here because the input directory is where the port looks.
  const Json* ae = root.get("config")->get("autoencoder_config");
  const Json* nnAe = ae ? ae->get("nn_ae") : nullptr;
  const Json* dec = nnAe ? nnAe->get("decoder") : nullptr;
  if (!dec) { fprintf(stderr, "%s has no decoder config\n", configPath.c_str()); return 1; }
  auto decNum = [&](const Json* o, const char* k, double fallback) {
    const Json* v = o ? o->get(k) : nullptr;
    return v && v->isNumber() ? v->n : fallback;
  };

  Dir dir(out);
  dir.meta("n", n);
  dir.meta("nRecycle", 1);
  dir.meta("seqFeatDim", seqDim);
  dir.meta("condFeatDim", condDim);
  dir.meta("pairFeatDim", pairDim);
  dir.meta("pairCondFeatDim", (double)pairCond.size());
  dir.meta("decLayers", decNum(dec, "nlayers", 12));
  dir.meta("decTokenDim", decNum(dec, "token_dim", 768));
  dir.meta("decCondDim", decNum(dec, "dim_cond", 128));
  dir.meta("decHeads", decNum(dec, "nheads", 12));
  dir.meta("decFeatDim", 3 + decNum(nnAe, "latent_z_dim", 32) + 32 +
                             decNum(dec, "n_residue_types", 147));
  dir.tensor("in/seq", seq);
  dir.tensor("in/cond0", cond0);
  dir.tensor("in/cond1", cond1);
  dir.tensor("in/pair", pair);
  dir.tensor("in/pairCond", pairCond);
  {
    vector<float> m(n), attn(n);
    for (int i = 0; i < n; ++i) {
      m[i] = b.paddingMask[i] ? 1.f : 0.f;
      attn[i] = (b.paddingMask[i] && !b.isInContextConditioning[i]) ? 1.f : 0.f;
    }
    dir.tensor("in/mask", m);
    dir.tensor("in/caNm", xBb);
    dir.tensor("in/attnMask", attn);
  }
  dir.ints("in/tokenType", b.tokenType);
  dir.close();
  printf("%s: %d tokens, %.1f MB\n", out.c_str(), n, dir.offset * 4 / 1e6);

  // ---- and everything a whole GENERATION needs, when it is asked for -------------------------
  //
  // 🔴 ONLY THIRTY-FIVE OF THE SEQUENCE COLUMNS MOVE BETWEEN STEPS - `xt_bb_ca` at 0 and
  // `xt_local_latents` at 3 - so the features go out once and the port overwrites those columns on
  // the device. The pair track's distance bins are refilled the same way, which is why they are
  // written WITHOUT them here. Everything else about a step is the conditioning, which depends on
  // t alone and is written for every step.
  if (!generate) return 0;
  const string gen = out + "-gen";
  Dir g(gen);
  const int steps = nsteps;
  const vector<int> depths = recycleDepths(recycleSchedule, steps);

  int pairNoDistDim = 0;
  const vector<float> pairNoDist = pairFeatures(b, cfg, xBb, true, pairNoDistDim);
  vector<float> condAll, cond1All, pairCondAll;
  vector<int> stochastic;
  for (int s2 = 0; s2 < steps; ++s2) {
    const double t2Bb = (float)schedBb.ts[s2], t2Ll = (float)schedLl.ts[s2];
    int d0 = 0, d1 = 0;
    const vector<float> c0 = condFeatures(b, cfg, w, t2Bb, t2Ll, -1, d0);
    const vector<float> c1 = condFeatures(b, cfg, w, t2Bb, t2Ll, 1, d1);
    const vector<float> pc = pairCondFeatures(cfg, t2Bb, t2Ll);
    condAll.insert(condAll.end(), c0.begin(), c0.end());
    cond1All.insert(cond1All.end(), c1.begin(), c1.end());
    pairCondAll.insert(pairCondAll.end(), pc.begin(), pc.end());
    for (int dm = 0; dm < 2; ++dm) {
      const Schedule& sc = dm ? schedLl : schedBb;
      const double t2 = dm ? t2Ll : t2Bb;
      stochastic.push_back(
          (t2 != 0 && !(t2 > sc.step.tLimOde || t2 < sc.step.tLimOdeBelow)) ? 1 : 0);
    }
  }

  // The noise stream, drawn rather than replayed: jsmath carries V8's log, so Box-Muller here is
  // the page's to the bit. The ORDER is `sample_noise`'s - the backbone, its per-molecule offsets,
  // then the latents - and then a draw a step wherever the step is stochastic.
  vector<float> noise;
  {
    Rng g2((uint64_t)seed);
    auto take = [&](size_t count) {
      const size_t at = noise.size();
      noise.resize(at + count);
      g2.randn(noise.data() + at, count);
    };
    take((size_t)n * 3);
    vector<int> groups;
    for (int i = 0; i < n; ++i) {
      if (emit.bb[i] && !b.coordCondMask[i] &&
          std::find(groups.begin(), groups.end(), b.moleculeIid[i]) == groups.end()) {
        groups.push_back(b.moleculeIid[i]);
      }
    }
    std::sort(groups.begin(), groups.end());
    for (size_t k = 0; k < groups.size(); ++k) take(3);
    take((size_t)n * 32);
    for (int s2 = 0; s2 < steps; ++s2) {
      for (int dm = 0; dm < 2; ++dm) {
        if (stochastic[s2 * 2 + dm]) take((size_t)n * (dm ? 32 : 3));
      }
    }
    g.meta("nGroups", (double)groups.size());
  }

  g.meta("steps", steps);
  g.meta("centeringScale", CENTERING_SCALE);
  g.meta("distDim", cfg.xtPairDistDim);
  g.meta("distMin", cfg.xtPairDistMin);
  g.meta("distMax", cfg.xtPairDistMax);
  g.meta("alignChirality", 0);       // a contig has no chiral frames; a structure input will
  g.meta("chirOff", -1);
  g.meta("chiralFrameWidth", 0);
  g.meta("partial", 0);
  g.meta("inverseFolding", 0);
  g.meta("partialTBb", 0);
  g.meta("partialTLl", 0);
  g.meta("baseNsteps", steps);
  g.meta("fullNsteps", 0);
  g.tensor("g/pairNoDist", pairNoDist);
  g.tensor("g/cond", condAll);
  g.tensor("g/cond1", cond1All);
  g.tensor("g/pairCond", pairCondAll);
  g.tensor("g/noise", noise);
  {
    vector<float> eb(n), el(n);
    for (int i = 0; i < n; ++i) { eb[i] = emit.bb[i]; el[i] = emit.ll[i]; }
    g.tensor("g/emitBb", eb);
    g.tensor("g/emitLl", el);
  }
  g.ints("g/depths", depths);
  g.ints("g/stochastic", stochastic);
  {
    vector<int> cc(n), mi(n), cs(n);
    for (int i = 0; i < n; ++i) {
      cc[i] = b.coordCondMask[i] ? 1 : 0;
      mi[i] = b.moleculeIid[i];
      cs[i] = b.conditionSequence[i];
    }
    g.ints("g/coordCond", cc);
    g.ints("g/moleculeIid", mi);
    g.ints("g/chainIid", b.chainIid);
    g.ints("g/condSeq", cs);
    vector<int> allowed(tables::ALLOWED_RESIDUE_TYPES.begin(), tables::ALLOWED_RESIDUE_TYPES.end());
    vector<int> legal(tables::RESTYPE_ATOM37_MASK.begin(), tables::RESTYPE_ATOM37_MASK.end());
    g.ints("g/allowedTypes", allowed);
    g.ints("g/atom37Legal", legal);
    g.ints("g/coordGroup", b.coordGroup);
  }
  g.close();
  {
    FILE* f = fopen((gen + "/names.txt").c_str(), "w");
    for (int a = 0; a < 37; ++a) fprintf(f, a ? " %s" : "%s", tables::ATOM37_NAMES[a]);
    fprintf(f, "\n");
    for (int t = 0; t < tables::N_TOKENS; ++t) fprintf(f, t ? " %s" : "%s", tables::TOKENS[t]);
    fprintf(f, "\n");
    fclose(f);
  }
  printf("%s: %d steps, %.1f MB\n", gen.c_str(), steps, g.offset * 4 / 1e6);
  return 0;
}

}  // namespace featurise
}  // namespace rfd4p

#ifndef RFD4P_FEATURISE_NO_MAIN
int main(int argc, char** argv) { return rfd4p::featurise::run(argc, argv); }
#endif
