// A sampled trajectory turned into a structure, and the structure written out.
//
// `cpu/rfd4p/decoder.js`'s tail is the spec: `sequenceForDecoder` and `decoderPostprocess`, then the records.
// All host code - the arrays are [n, 147] and [n, 111] and the work is an argmax and a table lookup, which
// `cuda/rfd4p/src/generate.cuh` says of its own copy for the same reason.
//
// 🔴 AND THE RECORDS AND THE WRITERS ARE THAT FILE'S, DERIVED AT BUILD TIME. See design.h for why this port
// stopped keeping its own pair and what the pair it kept had got wrong. What is left here is the part that is
// genuinely this port's: the decoder's outputs are computed over the ATTENDED rows and have to be read back
// against the token they came from, where the NVIDIA port scatters them first.
#include "rfd4p.h"
#include "design.h"
#include "rfd4p/featurise/tables.inc"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

rfd4p::GenOut decoderPostprocess(int n, const float* caNm, const int* seq, const float* logits,
                                 const float* coorsFlat, const float* presence, const float* emit,
                                 const int* coordGroup, const int* attnIdx, int na) {
  rfd4p::GenOut out;
  out.coors.assign((size_t)n * 111, 0.f);
  out.atomMask.assign((size_t)n * 37, 0);
  out.residueType.assign((size_t)n, -1);
  // the decoder's rows are the ATTENDED ones; its outputs are read here against the token they came from
  std::vector<int> rowOf((size_t)n, -1);
  for (int k = 0; k < na; ++k) rowOf[(size_t)attnIdx[k]] = k;
  const float* pad = M.hostF("in/mask");
  for (int i = 0; i < n; ++i) {
    if (rowOf[(size_t)i] < 0) {
      // 🔴 A TOKEN THAT IS PADDED AWAY GETS NO ATOM AT ALL, where one merely not emitted keeps its sampled
      // alpha carbon (`formatSampleLocalLatents`). This marked the CA for both, and the mask is what
      // `alignToInputFrame` reads to decide which atoms ANCHOR the fit onto the input - so a padded token with
      // a conditioned coordinate became an anchor the page does not have, and the whole design came out in a
      // frame a thousandth of an angstrom off the page's, visible in the third decimal of every coordinate.
      // Inert until this port aligned anything: the old records skipped a token with no residue type.
      if (pad[i] != 0) {
        out.atomMask[(size_t)i * 37 + 1] = 1;
        for (int k = 0; k < 3; ++k) out.coors[(size_t)i * 111 + 3 + k] = (float)(caNm[i * 3 + k] * 10);
      }
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
    out.residueType[i] = rt;
    for (int a = 0; a < 37; ++a) {
      const bool legal = rfd4p::tables::RESTYPE_ATOM37_MASK[(size_t)rt * 37 + a] != 0;
      // 🔴 SLOT 1 IS ALWAYS PRESENT AND A COORDINATE-GROUP ATOM IS REQUIRED, whatever the head says: the alpha
      // carbon is the token's position and a conditioned group's atoms were asked for by the job.
      const bool required = coordGroup && coordGroup[(size_t)i * 37 + a] >= 0;
      const bool present = presence[(size_t)row * 37 + a] > 0 || a == 1 || required;
      if (!(legal && present)) continue;
      out.atomMask[(size_t)i * 37 + a] = 1;
      for (int k = 0; k < 3; ++k) {
        const float rel = a == 1 ? 0.f : coorsFlat[(size_t)row * 111 + a * 3 + k];
        out.coors[(size_t)i * 111 + a * 3 + k] = (float)((double)(rel + caNm[i * 3 + k]) * 10);
      }
    }
  }
  return out;
}

Design buildDesign(rfd4p::GenOut& g, const OutInput& in) {
  const int n = in.n;
  // The input's own numbering and chain letters where a job named a structure; absent for a contig, where the
  // token index IS the residue number. And what the writer reads off the input atoms (`outputTables`).
  rfd4p::OutNaming naming;
  if (in.outResId && in.outChain) {
    naming.resId.assign(in.outResId, in.outResId + n);
    naming.chain.assign(in.outChain, in.outChain + n);
  }
  if (in.inName) {
    const size_t slots = (size_t)n * 37;
    auto per = [&](const int* src, size_t count, std::vector<int>& into) { into.assign(src, src + count); };
    per(in.atomize, n, naming.in.atomize);
    per(in.hetero, n, naming.in.hetero);
    per(in.entity, n, naming.in.entity);
    per(in.insCode, n, naming.in.insCode);
    per(in.resName, (size_t)n * 2, naming.in.resName);
    per(in.srcComp, (size_t)n * 4, naming.in.srcComp);
    per(in.inName, slots, naming.in.inName);
    per(in.inElem, slots, naming.in.inElem);
    per(in.inCharge, slots, naming.in.inCharge);
    per(in.inB, slots * 2, naming.in.inB);
    per(in.inOcc, slots * 2, naming.in.inOcc);
  }
  // Only the INDEXED tokens are written; an unindexed motif was grafted onto one of them.
  if (in.condIndex) {
    naming.emit.assign((size_t)n, 1);
    for (int i = 0; i < n; ++i) naming.emit[i] = in.condIndex[(size_t)i * 37 + 1] != 0;
  }

  Design out;
  std::vector<int> crdSource, srcAt;
  std::vector<uint8_t> isInsertion;
  // 🔴 THE CONDITIONING IS COPIED BECAUSE THE GRAFTING REWRITES IT. `applyInsertion` moves an unindexed
  // motif's conditioned coordinates onto the token it was grafted to, and the alignment then rotates them -
  // neither may touch the work directory's own mapped bytes, which a second design reads again.
  std::vector<float> condCoord, condIntCrd;
  std::vector<int> coordGroup;
  if (in.condCoord) {
    const size_t slots = (size_t)n * 37;
    condCoord.assign(in.condCoord, in.condCoord + slots * 3);
    if (in.condIntCrd) condIntCrd.assign(in.condIntCrd, in.condIntCrd + slots * 3);
    if (in.coordGroup) coordGroup.assign(in.coordGroup, in.coordGroup + slots);
    // 🔴 THE UNINDEXED MOTIFS FIRST, BECAUSE THEY SUPPLY ANCHORS THE ALIGNMENT THEN USES. An unindexed token
    // is in-context conditioning and is never emitted, so its conditioned coordinates reach the output only by
    // being grafted onto a token that was.
    if (in.condIndex) {
      std::vector<uint8_t> emitted((size_t)n, 1);
      out.proposed = rfd4p::predictedInsertionMapping(n, in.insLogits, in.condIndex, emitted, naming.chain);
      const rfd4p::Insertion ins = rfd4p::resolveInsertion(
          g, n, in.insLogits, in.condIndex, in.resType, in.coords, condCoord.data(),
          condIntCrd.empty() ? nullptr : condIntCrd.data(), emitted, naming);
      out.grafted = (int)ins.source.size();
      if (ins.source.empty()) {
        for (int i = 0; i < n; ++i) out.unindexedUnplaced |= in.condIndex[(size_t)i * 37 + 1] == 0;
      }
      rfd4p::applyInsertion(g, n, ins, naming, in.resType, in.coords, condCoord.data(),
                            condIntCrd.empty() ? nullptr : condIntCrd.data(),
                            coordGroup.empty() ? nullptr : coordGroup.data(), isInsertion, &crdSource, false,
                            nullptr, &srcAt);
    }
  }
  // 🔴 INTO THE INPUT'S FRAME BEFORE ANYTHING IS WRITTEN. A contig has no conditioned coordinates, so this is
  // a no-op there; a structure job's output is otherwise in the SAMPLER's centred frame and superposes on
  // nothing - which is what this port did, so a motif design could not be scored against the target it was
  // built against.
  //
  // 🔴 AND IT STILL LANDS A CENTRING OFFSET AWAY FROM THE PAGE'S, WHICH IS NOT THIS PORT'S TO FIX.
  // `shared/rfd4p/output.js` adds `inputAtoms.center_shift` back after aligning - the centroid
  // `centerAndFeaturizeCentroid` subtracted when it featurised the conditioning - and `alignToInputFrame` has no
  // such step because the featuriser does not export that offset: both `g/coords` and `g/condCoord` come out
  // already centred, measured centroids 1e-7, so there is nothing here to derive it from. It is sub-milliangstrom
  // on an input that was already centred and ANGSTROMS on one that was not (`sm_sam` is over 3 A off the origin,
  // `ppi_pdl1` 168), and it is the whole of what separates this writer from the page's: shifted by hand, the
  // files are byte-identical. The fix is one tensor out of cuda/rfd4p's featuriser and three adds in
  // `alignToInputFrame`, both of which are cuda/'s to make.
  rfd4p::alignToInputFrame(g, n, condCoord.empty() ? nullptr : condCoord.data(),
                           condIntCrd.empty() ? nullptr : condIntCrd.data(),
                           coordGroup.empty() ? nullptr : coordGroup.data(), naming);
  // ...and back out of the centred frame, WHERE THE EXPORT CARRIES THE OFFSET TO DO IT WITH. `meta/centerShift*`
  // is `centerAndFeaturizeCentroid`'s own return value - the centroid it subtracted - and this is
  // `processSampleToAtoms`' last step (output.js), on the same condition: any anchor at all, and a shift that is
  // not zero. One anchor does not fix a frame so the alignment above declines it, and the translation still
  // applies, which is why the count and not the fit decides. Rounded to float per component, as the reference's
  // `Math.fround` is. A work directory written without the offset leaves this a no-op and the design in the
  // centred frame, which is what every native port did before it was exported.
  {
    size_t anchors = 0;
    for (size_t sl = 0; sl < (size_t)n * 37 && !condCoord.empty(); ++sl) {
      if (g.atomMask[sl] && !naming.skip((int)(sl / 37)) && std::isfinite(condCoord[sl * 3])) ++anchors;
    }
    const double sx = M.meta("meta/centerShift0", 0.0), sy = M.meta("meta/centerShift1", 0.0),
                 sz = M.meta("meta/centerShift2", 0.0);
    if (anchors && (sx != 0.0 || sy != 0.0 || sz != 0.0)) {
      for (size_t sl = 0; sl < (size_t)n * 37; ++sl) {
        if (!g.atomMask[sl]) continue;
        float* c = &g.coors[(sl / 37) * 111 + (sl % 37) * 3];
        c[0] = (float)((double)c[0] + sx);
        c[1] = (float)((double)c[1] + sy);
        c[2] = (float)((double)c[2] + sz);
      }
      out.shifted = true;
    }
  }
  // The records the page's `processSampleToAtoms` builds, from the INPUT's conditioning as it was before any
  // grafting - which atom carried a conditioned coordinate is a fact about the input, and the grafted ones are
  // named by `crdSource`.
  const std::vector<int> chainIid =
      in.chainIid ? std::vector<int>(in.chainIid, in.chainIid + n) : std::vector<int>((size_t)n, 0);
  out.records = rfd4p::buildRecords(g, n, chainIid, naming, in.condCoord, in.condIntCrd, isInsertion, crdSource,
                                    srcAt, in.coordGroup);

  // the consecutive alpha-carbon spacing WITHIN A CHAIN: measured across every token in order, a two-chain
  // design reports the jump from one chain to the next as a 60 A "bond", which reads exactly like a broken
  // backbone and is the measurement being wrong rather than the structure
  std::vector<double> gaps;
  const rfd4p::AtomRecord* prev = nullptr;
  for (const rfd4p::AtomRecord& r : out.records) {
    if (r.atomName != "CA") continue;
    if (prev && prev->chain == r.chain) {
      gaps.push_back(std::sqrt((double)(r.x - prev->x) * (r.x - prev->x) +
                               (double)(r.y - prev->y) * (r.y - prev->y) +
                               (double)(r.z - prev->z) * (r.z - prev->z)));
    }
    prev = &r;
  }
  if (!gaps.empty()) {
    std::sort(gaps.begin(), gaps.end());
    out.caSpacing[0] = gaps[gaps.size() / 2];
    out.caSpacing[1] = gaps.front();
    out.caSpacing[2] = gaps.back();
  }
  return out;
}

int writeStructure(const std::string& path, const Design& d) {
  // 🔴 THE FORMAT FOLLOWS THE NAME, and mmCIF is the native one: RFD4-Proteina writes `.cif`, and a design
  // saved as a PDB loses a four-character chain id and anything past 99,999 atoms.
  const bool cif = path.size() > 4 && path.compare(path.size() - 4, 4, ".cif") == 0;
  const std::string text = cif ? rfd4p::toCif(d.records) : rfd4p::toPdb(d.records);
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("cannot write %s", path.c_str());
  fwrite(text.data(), 1, text.size(), f);
  fclose(f);
  return (int)d.records.size();
}
