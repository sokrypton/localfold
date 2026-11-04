// A sampled trajectory turned into a structure, and the structure written out.
//
// `cpu/rfd4p/decoder.js`'s tail is the spec: `sequenceForDecoder`, `decoderPostprocess` and
// `formatSampleLocalLatents`, then the records. All host code - the arrays are [n, 147] and [n, 111] and the work
// is an argmax and a table lookup, which `cuda/rfd4p/src/generate.cuh` says of its own copy for the same reason.
//
// 🔴 THE RECORDS ARE BUILT ONCE AND RENDERED TWICE. `toPdbString` and `toCifString` in the reference are two
// renderings of ONE list, and a design saved as a PDB and as an mmCIF has to be the same design - two formats of
// one structure that disagree are worse than one format. The naming, the HETATM flag, the element, the b-factor
// and the occupancy are decided in `records`; the writers only lay them out. mmCIF is what RFD4-Proteina itself
// writes, so `--out=design.cif` is the native format and `--out=design.pdb` the convenient one.
//
// Not here, and in cuda/rfd4p: a grafted motif atom naming the component it came from, and the
// unindexed-insertion mapping - both bookkeeping over the same coordinates, and both needing tables this port
// does not read.
#include "rfd4p.h"
#include "rfd4p/featurise/tables.inc"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdint>

namespace {

std::string trimName(const char* s) { return s ? std::string(s) : std::string(); }

// a string the exporter packed into `words` ints, four characters each, little-endian and zero-padded
std::string unpackString(const int* w, int words) {
  std::string out;
  for (int i = 0; i < words; ++i)
    for (int k = 0; k < 4; ++k) {
      const char c = (char)((w[i] >> (8 * k)) & 0xff);
      if (c) out += c;
    }
  return out;
}

}  // namespace

// a double the exporter packed as two little-endian int words (`packDouble`)
double unpackDouble(const int* w) {
  uint32_t lo = (uint32_t)w[0], hi = (uint32_t)w[1];
  uint64_t bits = ((uint64_t)hi << 32) | lo;
  double v;
  memcpy(&v, &bits, 8);
  return v;
}

int sequenceForDecoder(int n, const float* logits, const int* condSeq, const int* tokenType, const float* emit,
                       int* seq) {
  int filled = 0;
  for (int i = 0; i < n; ++i) {
    seq[i] = condSeq[i];
    // 🔴 ONLY AN EMITTED TOKEN WITH NO CONDITIONED RESIDUE TAKES THE ARGMAX, and the argmax is over the token
    // TYPE's allowed residues - a protein token cannot come out a nucleotide because its logit happened to win.
    if (emit[i] == 0 || seq[i] != 0) continue;
    const int tt = std::max(0, std::min(rfd4p::tables::N_TOKEN_TYPES - 2, tokenType[i]));
    int best = 0;
    float bestV = -INFINITY;
    for (int r = 0; r < rfd4p::tables::N_TOKENS; ++r) {
      if (!rfd4p::tables::ALLOWED_RESIDUE_TYPES[(size_t)tt * rfd4p::tables::N_TOKENS + r]) continue;
      const float v = logits[(size_t)i * rfd4p::tables::N_TOKENS + r];
      if (v > bestV) { bestV = v; best = r; }
    }
    seq[i] = best;
    filled += 1;
  }
  return filled;
}

void decoderPostprocess(int n, const float* caNm, const int* seq, const float* logits, const float* coorsFlat,
                        const float* presence, const float* emit, const int* coordGroup, const int* attnIdx, int na,
                        float* coors, uint8_t* atomMask, int* residueType) {
  // the decoder's rows are the ATTENDED ones; its outputs are read here against the token they came from
  std::vector<int> rowOf((size_t)n, -1);
  for (int k = 0; k < na; ++k) rowOf[(size_t)attnIdx[k]] = k;
  for (int i = 0; i < n; ++i) {
    residueType[i] = -1;
    if (emit[i] == 0 || rowOf[(size_t)i] < 0) {
      // a padded-away token keeps its sampled alpha carbon and nothing else (`formatSampleLocalLatents`)
      atomMask[(size_t)i * 37 + 1] = 1;
      for (int k = 0; k < 3; ++k) coors[(size_t)i * 111 + 3 + k] = (float)(caNm[i * 3 + k] * 10);
      continue;
    }
    const int row = rowOf[(size_t)i];
    const int tt = std::max(0, std::min(rfd4p::tables::N_TOKEN_TYPES - 2, M.hostI("in/tokenType")[i]));
    int best = 0;
    float bestV = -INFINITY;
    for (int r = 0; r < rfd4p::tables::N_TOKENS; ++r) {
      if (!rfd4p::tables::ALLOWED_RESIDUE_TYPES[(size_t)tt * rfd4p::tables::N_TOKENS + r]) continue;
      const float v = logits[(size_t)row * rfd4p::tables::N_TOKENS + r];
      if (v > bestV) { bestV = v; best = r; }
    }
    const int rt = seq[i] != 0 ? seq[i] : best;
    residueType[i] = rt;
    for (int a = 0; a < 37; ++a) {
      const bool legal = rfd4p::tables::RESTYPE_ATOM37_MASK[(size_t)rt * 37 + a] != 0;
      // 🔴 SLOT 1 IS ALWAYS PRESENT AND A COORDINATE-GROUP ATOM IS REQUIRED, whatever the head says: the alpha
      // carbon is the token's position and a conditioned group's atoms were asked for by the job.
      const bool required = coordGroup && coordGroup[(size_t)i * 37 + a] >= 0;
      const bool present = presence[(size_t)row * 37 + a] > 0 || a == 1 || required;
      if (!(legal && present)) continue;
      atomMask[(size_t)i * 37 + a] = 1;
      for (int k = 0; k < 3; ++k) {
        const float rel = a == 1 ? 0.f : coorsFlat[(size_t)row * 111 + a * 3 + k];
        coors[(size_t)i * 111 + a * 3 + k] = (float)((double)(rel + caNm[i * 3 + k]) * 10);
      }
    }
  }
}

namespace {

struct Rec {
  std::string atomName, resName, chain, element, insCode;
  int resId = 0, entity = 0, charge = 0;
  double b = NAN, occ = 1.0;
  bool het = false, ca = false;
  float x = 0, y = 0, z = 0;
};

// an mmCIF atom id is quoted where it carries a prime (the nucleotides' O5', C1')
std::string cifAtomName(const std::string& name) {
  return name.find('\'') == std::string::npos ? name : "\"" + name + "\"";
}

// `jsToFixed(round-half-to-even(x * 1000) / 1000, 3)`: the reference's own coordinate text, so two ports' files
// of one design are the same BYTES and not merely the same numbers
std::string cifCoord(double v) {
  const double scaled = v * 1000.0;
  const double fl = std::floor(scaled);
  const double diff = scaled - fl;
  double rounded;
  if (std::abs(diff - 0.5) < 1e-6) rounded = std::fmod(fl, 2.0) == 0 ? fl : fl + 1;
  else rounded = std::floor(scaled + 0.5);
  char buf[32];
  snprintf(buf, sizeof buf, "%.3f", rounded / 1000.0);
  return buf;
}

std::string num(double v, int places) {
  if (!std::isfinite(v)) return "nan";
  char buf[48];
  snprintf(buf, sizeof buf, "%.*f", places, v);
  return buf;
}

// the chain a token is written under: the input's own letters where it had them, else A, B, C by instance
std::string chainOf(int i, const int* outChain, const int* chainIid) {
  if (outChain) {
    std::string c;
    for (int k = 0; k < 4; ++k) {
      const char ch = (char)((outChain[i] >> (8 * k)) & 0xff);
      if (ch) c += ch;
    }
    if (!c.empty()) return c;
  }
  if (chainIid) return std::string(1, (char)('A' + (chainIid[i] % 26)));
  return "A";
}

std::string elementOf(const std::string& atomName) {
  // the vocabulary's names are element-plus-counter, so the first character is the element - and atom37 carries
  // no two-letter element
  return atomName.empty() ? "C" : std::string(1, atomName[0]);
}

std::vector<Rec> records(int n, const float* coors, const uint8_t* atomMask, const int* residueType,
                         const int* outResId, const int* outChain, const int* chainIid, const OutTables& in) {
  std::vector<Rec> out;
  for (int i = 0; i < n; ++i) {
    const int rt = residueType[i];
    Rec base;
    base.chain = chainOf(i, outChain, chainIid);
    base.resId = outResId ? outResId[i] : i + 1;
    if (in.present) {
      base.insCode = unpackString(&in.insCode[i], 1);
      base.entity = in.entity[i];
    }
    // 🔴 AN ATOMISED TOKEN IS ONE ATOM AND IT IS THE INPUT'S. A ligand's atom or an atomised residue's is
    // carried over whole - its own name, residue, element and HETATM flag - at the token's sampled position.
    // Written from the vocabulary instead it comes out `UNK CA`, element C, `ATOM`: a glycerol as six alanines.
    if (in.present && in.atomize[i]) {
      const size_t at = (size_t)i * 37 + 1;
      Rec r = base;
      r.atomName = unpackString(&in.inName[at], 1);
      r.resName = unpackString(&in.resName[(size_t)i * 2], 2);
      const std::string el = in.inElem[at] ? unpackString(&in.inElem[at], 1) : std::string();
      r.element = el.empty() || el == "X" ? elementOf(r.atomName) : el;
      r.het = in.hetero[i] != 0;
      r.b = in.inB ? unpackDouble(&in.inB[at * 2]) : NAN;
      r.charge = in.inCharge ? in.inCharge[at] : 0;
      r.x = coors[(size_t)i * 111 + 3];
      r.y = coors[(size_t)i * 111 + 4];
      r.z = coors[(size_t)i * 111 + 5];
      out.push_back(r);
      continue;
    }
    if (rt < 0) continue;
    const std::string inRes = in.present ? unpackString(&in.resName[(size_t)i * 2], 2) : std::string("<M>");
    // a token the job masked is a DESIGNED residue: it is not the input's hetero component whatever the input
    // said, and it carries none of its b-factors
    const bool wasMasked = inRes == "<M>" || inRes == "UNK";
    for (int a = 0; a < 37; ++a) {
      if (!atomMask[(size_t)i * 37 + a]) continue;
      Rec r = base;
      r.atomName = trimName(rfd4p::tables::TOKEN_ATOMS[(size_t)rt][(size_t)a]);
      if (r.atomName.empty()) continue;
      r.resName = trimName(rfd4p::tables::TOKENS[(size_t)rt]);
      r.het = !wasMasked && in.present && in.hetero[i] != 0;
      // 🔴 THE INPUT'S OWN b-FACTOR AND OCCUPANCY, FOUND BY ATOM NAME. The slots of a designed residue are not
      // the input's slots - a motif's ASP is the input's ASP but this token's type may differ - so the carry-over
      // is by NAME, and `OXT` and `OP3` are excluded because the vocabulary's terminal atoms are not the input's.
      int is = -1;
      if (in.present && in.inName && !wasMasked && r.atomName != "UNK" && r.atomName != "OXT" && r.atomName != "OP3") {
        for (int s2 = 0; s2 < 37; ++s2) {
          const int w = in.inName[(size_t)i * 37 + s2];
          if (w && unpackString(&w, 1) == r.atomName) { is = s2; break; }
        }
      }
      if (is >= 0) {
        const size_t ia = (size_t)i * 37 + (size_t)is;
        r.b = in.inB ? unpackDouble(&in.inB[ia * 2]) : NAN;
        const double occ = in.inOcc ? unpackDouble(&in.inOcc[ia * 2]) : 1.0;
        r.occ = occ > 0 ? occ : 1.0;
        r.charge = in.inCharge ? in.inCharge[ia] : 0;
        const std::string el = in.inElem && in.inElem[ia] ? unpackString(&in.inElem[ia], 1) : std::string();
        r.element = el.empty() || el == "X" ? elementOf(r.atomName) : el;
      } else {
        r.element = elementOf(r.atomName);
      }
      r.x = coors[(size_t)i * 111 + a * 3];
      r.y = coors[(size_t)i * 111 + a * 3 + 1];
      r.z = coors[(size_t)i * 111 + a * 3 + 2];
      r.ca = a == 1;
      out.push_back(r);
    }
  }
  return out;
}

// a PDB atom name's column rule: a one-letter element starts in column 14, anything else in 13
std::string pdbAtomField(const std::string& name) {
  if (name.size() >= 4) return name.substr(0, 4);
  return " " + name + std::string(3 - name.size(), ' ');
}

std::string padLeft(const std::string& s, int w) {
  return s.size() >= (size_t)w ? s : std::string((size_t)(w - s.size()), ' ') + s;
}

std::string toPdb(const std::vector<Rec>& recs) {
  std::string out;
  for (size_t i = 0; i < recs.size(); ++i) {
    const Rec& a = recs[i];
    char line[128];
    snprintf(line, sizeof line, "%-6s%5d %s %-3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n",
             a.het ? "HETATM" : "ATOM", (int)((i + 1) % 100000), pdbAtomField(a.atomName).c_str(),
             a.resName.c_str(), a.chain.substr(0, 1).c_str(), a.resId,
             a.insCode.empty() ? " " : a.insCode.substr(0, 1).c_str(), a.x, a.y, a.z,
             std::isfinite(a.occ) ? a.occ : 1.0, std::isfinite(a.b) ? a.b : 0.0,
             padLeft(a.element, 2).c_str());
    out += line;
  }
  out += "END\n";
  return out;
}

std::string toCif(const std::vector<Rec>& recs) {
  static const char* COLUMNS[] = {
      "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id",
      "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "auth_seq_id", "auth_comp_id", "auth_asym_id",
      "auth_atom_id", "B_iso_or_equiv", "occupancy", "pdbx_formal_charge", "Cartn_x", "Cartn_y", "Cartn_z",
      "pdbx_PDB_model_num"};
  std::string out = "data_rfd4p_design\n#\nloop_\n";
  for (const char* c : COLUMNS) out += std::string("_atom_site.") + c + "\n";
  for (size_t i = 0; i < recs.size(); ++i) {
    const Rec& a = recs[i];
    const std::string id = cifAtomName(a.atomName);
    const std::string comp = a.resName.empty() ? "UNK" : a.resName;
    const std::string charge = a.charge != 0
        ? (a.charge > 0 ? "+" + std::to_string(a.charge) : std::to_string(a.charge))
        : std::string("?");
    out += (a.het ? "HETATM " : "ATOM ") + std::to_string(i + 1) + " " + a.element + " " + id + " . " + comp +
           " " + a.chain + " " + std::to_string(a.entity) + " " + std::to_string(a.resId) + " " +
           (a.insCode.empty() ? "." : a.insCode) + " " + std::to_string(a.resId) + " " + comp + " " + a.chain +
           " " + id + " " + num(a.b, 2) + " " + num(std::isfinite(a.occ) ? a.occ : 1.0, 1) + " " + charge + " " +
           cifCoord(a.x) + " " + cifCoord(a.y) + " " + cifCoord(a.z) + " 1\n";
  }
  out += "#\n";
  return out;
}

}  // namespace

int writeStructure(const std::string& path, int n, const float* coors, const uint8_t* atomMask,
                   const int* residueType, const int* outResId, const int* outChain, const int* chainIid,
                   double* caSpacing, const OutTables& in) {
  const std::vector<Rec> recs = records(n, coors, atomMask, residueType, outResId, outChain, chainIid, in);
  // 🔴 THE FORMAT FOLLOWS THE NAME, and mmCIF is the native one: RFD4-Proteina writes `.cif`, and a design saved
  // as a PDB loses a four-character chain id and anything past 99,999 atoms.
  const bool cif = path.size() > 4 && path.compare(path.size() - 4, 4, ".cif") == 0;
  const std::string text = cif ? toCif(recs) : toPdb(recs);
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("cannot write %s", path.c_str());
  fwrite(text.data(), 1, text.size(), f);
  fclose(f);

  // the consecutive alpha-carbon spacing WITHIN A CHAIN: measured across every token in order, a two-chain
  // design reports the jump from one chain to the next as a 60 A "bond", which reads exactly like a broken
  // backbone and is the measurement being wrong rather than the structure
  if (caSpacing) {
    std::vector<double> gaps;
    const Rec* prev = nullptr;
    for (const Rec& r : recs) {
      if (!r.ca) continue;
      if (prev && prev->chain == r.chain)
        gaps.push_back(std::sqrt((double)(r.x - prev->x) * (r.x - prev->x) + (double)(r.y - prev->y) * (r.y - prev->y) +
                                 (double)(r.z - prev->z) * (r.z - prev->z)));
      prev = &r;
    }
    if (!gaps.empty()) {
      std::sort(gaps.begin(), gaps.end());
      caSpacing[0] = gaps[gaps.size() / 2];
      caSpacing[1] = gaps.front();
      caSpacing[2] = gaps.back();
    }
  }
  return (int)recs.size();
}
