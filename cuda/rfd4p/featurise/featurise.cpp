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
#include "../src/schedule.h"
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
struct PipelineRng {
  uint32_t a;
  explicit PipelineRng(uint32_t seed) : a(seed) {}
  double nextFloat() {
    a = a + 0x6d2b79f5u;
    uint32_t t = (a ^ (a >> 15)) * (1u | a);
    t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
    return (double)((t ^ (t >> 14))) / 4294967296.0;
  }
  int nextInt(int lo, int hiExclusive) {
    return lo + (int)std::floor(nextFloat() * (hiExclusive - lo));
  }
  vector<int> permutation(int n) {
    vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    for (int i = n - 1; i > 0; --i) {
      const int j = (int)std::floor(nextFloat() * (i + 1));
      std::swap(p[i], p[j]);
    }
    return p;
  }
};

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
    string text;
    FILE* f = fopen(dumpCif.c_str(), "rb");
    if (!f) { fprintf(stderr, "no %s\n", dumpCif.c_str()); return 1; }
    char buf[65536];
    size_t got;
    while ((got = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, got);
    fclose(f);
    if (text.size() > 3 && (unsigned char)text[0] == 0x28 && (unsigned char)text[1] == 0xb5) {
      fprintf(stderr, "%s is zstd; this reader takes a plain .cif and the page's examples are "
                      "compressed - decompress it first\n", dumpCif.c_str());
      return 1;
    }
    const rfd4p::cif::Block block = rfd4p::cif::parseBlock(text);
    rfd4p::cif::AtomSite a = rfd4p::cif::readAtomSite(block);
    rfd4p::cif::deriveChemistry(a, block);
    printf("n %zu\n", a.n);
    for (size_t i = 0; i < a.n; ++i) {
      printf("%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g"
             "\t%d\t%d\t%d\t%d\t%d\t%s\n",
             a.element[i].c_str(), a.atomName[i].c_str(), a.resName[i].c_str(),
             a.chainId[i].c_str(), a.insCode[i].c_str(), a.labelAltId[i].c_str(), a.resId[i],
             a.authSeqId[i], a.atomId[i], a.labelEntityId[i], (int)a.hetero[i], (double)a.x[i],
             (double)a.y[i], (double)a.z[i], (double)a.bFactor[i], (double)a.occupancy[i],
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
      for (float x : v) printf("\t%.9g", (double)x);
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
    if (Json::absent(spec.get("contig"))) {
      fprintf(stderr, "%s has no contig\n", specPath.c_str());
      return 1;
    }
    contig = spec.get("contig")->s;
    if (!Json::absent(spec.get("nsteps"))) nsteps = (int)spec.get("nsteps")->n;
    for (const char* k : {"recycleSchedule", "recycle_schedule"}) {
      if (!Json::absent(spec.get(k))) recycleSchedule = spec.get(k)->s;
    }
    const Json* cond = spec.get("conditions");
    if (cond && cond->isObject()) {
      for (const auto& [key, arms] : cond->o) {
        for (const auto& [value, sels] : arms.o) {
          vector<string> list;
          if (sels.isArray()) {
            for (const Json& e : sels.a) {
              const Json* sel = e.get("select");
              if (sel && sel->isString()) list.push_back(sel->s);
            }
          }
          conditions[key][value] = list;
        }
      }
    }
  }

  const Batch b = buildSpecBatch(parseContig(contig), conditions, seed);
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
