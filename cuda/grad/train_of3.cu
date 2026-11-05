// OpenFold3's training, in CUDA: the AF3 lineage's whole training network assembled from a bundle (af3embed.cuh,
// af3tmpl.cuh, af3.cuh, af3denoise.cuh, af3conf.cuh) and trained the way openfold3/projects/of3_all_atom trains it -
// OpenBind-0 is OpenFold3 v0.5.0, so the recipe is its own.
//
// A step, an example at a time (model.py OpenFold3.forward, runner.py _training_step_manual_clip):
//   the example   one of --examples (a line each: `input truth [resolution=X] [weight=W] [preferred=i[,j]]`, as
//                 cuda/grad/make_of3_example.py writes them), drawn by weight; cropped to --crop tokens (384) by OpenFold3's
//                 three strategies (contiguous 0.2, spatial 0.4, spatial-interface 0.4; whole when it fits)
//   dropout       pair 0.25 (the triangle updates rowwise, the ending attention columnwise) in every pairformer-style
//                 stack and MSA 0.15 on the pair-weighted averaging, fresh masks a step (--no-dropout: off)
//   the trunk     1 + U{0..3} cycles, the gradient through the last; each cycle's MSA a fresh 1024 rows (the valid
//                 rows in a random order, then invalid ones: MSAModuleEmbedder._subsample_all_msa)
//   the rollout   AF3's sampler, 20 steps, one sample, no gradient; on it the truth's permutation (of3perm.h: which
//                 copy of a repeated molecule, which naming of a symmetric component), used by every loss after; the
//                 confidence head on it with every input detached (pLDDT, PDE, PAE, experimentally resolved)
//   diffusion     --samples (48) noised copies of the augmented truth, sigma = 16 exp(-1.2 + 1.5 N(0, 1))
//   the losses    of3loss.h, held to a transcription of openfold3/core/loss (of3_loss_oracle.py, 1e-14):
//                 4 (w(t) mse + smooth LDDT) + bond, 0.03 distogram, 1e-4 each confidence term - the confidence terms
//                 zero for a structure outside 0.1-4.0 A (or with no resolution), its head then left out of the step
//   the update    of3train.cuh: the example's gradient clipped to norm 10 and accumulated, Adam (0.9, 0.95) after
//                 --batch examples with OpenFold3's warmup and decay, the EMA (0.999); --save / --resume
//
// --check-perm=k disguises example k (copies relabelled, symmetric components renamed) and the permutation must undo it.
// Before training it holds its trunk to cuda/af3's runTrunk on the first example (pass 0 and three recycled passes)
// - the composition this file writes, against the port's - and every example's recomputed atom windows to the
// featuriser's own (the windows a crop rebuilds are the same function).
//
//   nvcc -O2 -std=c++17 -arch=sm_80 --default-stream per-thread -o cuda/grad/train-of3 cuda/grad/train_of3.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/train-of3 --bundle=model-openbind0-f32 --family=openbind0 --examples=<file> [--steps=N] [--batch=1]
//       [--crop=384] [--samples=48] [--lr=1.8e-3] [--warmup=1000] [--bond=0] [--save=<ckpt>] [--save-every=N]
//       [--resume=<ckpt>] [--seed=1] [--init=bundle|of3] [--tf32] [--bf16] [--no-dropout] [--val=<file>] [--smoke]
// --check-ckpt=k / --check-perm=k / --check-bf16=k: the recomputation, the truth's permutation and the bf16 trunk held
// to their references on example k.
// --init=of3: OpenFold3's from-scratch initialisation instead of the bundle's weights.
// --smoke: the gate's short run - before and after, on the first example whole and on fixed draws, the loss above its
// floor (smooth LDDT's 0.196 a weight) must halve.
#include <array>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include "../af3/src/trunk.cuh"
#include "../af3/src/diffusion.cuh"
#include "../af3/src/sampler.cuh"
#include "../featurise/af3_weights.h"
#include "../featurise/featurise.h"
#include "../featurise/tables.h"
#include "adam.cuh"
#include "af3conf.cuh"
#include "af3denoise.cuh"
#include "af3embed.cuh"
#include "af3tmpl.cuh"
#include "of3loss.h"
#include "of3perm.h"
#include "of3train.cuh"

namespace g = grad;

static g::real* dzeros(size_t n) { g::real* p; GK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(g::real))); GK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(g::real))); return p; }

// ---- an example, on the host, at its full size
struct Example {
  std::string input, truthPath;
  double resolution = -1, weight = 1;          // (no resolution: the confidence losses off)
  std::vector<int> preferred;                  // OpenFold3's preferred chain or interface (asym ids), empty for none
  int n = 0, dense = 0, profW = 0, P = 0;
  std::vector<int> aatype, ri, ti, asym, ent, sym, isDna, isRna, isLigand, isMod, refElement, names, uid, pbSlot;
  std::vector<float> profile, delMean, seqMask, refPos, refMask, refCharge, predMask, bonds, orders;
  std::vector<int> msaCode; std::vector<float> msaDel, msaMask;      // the MSA pool [P][n]
  struct Slot { double repeat = 0; std::vector<int> aatype, bin; std::vector<float> pb, uv, bb; bool geo = false; };
  std::vector<Slot> slots; int templates = 0;
  std::vector<double> xt, gm;                  // the truth [atoms * 3] and its resolved mask [atoms]
  std::string permsPath, set = "pdb";        // OpenFold3's dataset: pdb, monomer, rna (distillation), disordered
  of3::PermInput perm;                         // the whole entry, for the truth's permutation (of3perm.h)
  size_t atoms() const { return (size_t)n * dense; }
};

template <class T> static std::vector<T> copyOf(const T* p, size_t n) { return std::vector<T>(p, p + n); }

// the atom windows of a token layout (featurise.h's own atomGathers: what the featuriser built, rebuilt for a crop)
static lf::Batch windows(const Example& e, bool padded, bool qblock) {
  lf::Batch b; b.tokens = e.n; b.dense = e.dense;
  std::vector<int> real;
  for (size_t i = 0; i < e.atoms(); ++i) if (e.predMask[i] > 0) real.push_back((int)i);
  lf::atomGathers(b, real, e.pbSlot, padded, qblock);
  return b;
}

// one line of --examples into M and out of it: the input's keys replace the last example's whole (an input's
// `batch.`, `template.` and `job.` keys - a key one input has and the next lacks must not survive), then its arrays
// a line's own fields (resolution, weight, preferred, perms, set) onto an example
static void applyFields(Example& e, const std::string& line) {
  std::istringstream ls(line);
  std::string skip, kv;
  ls >> skip >> skip;
  e.preferred.clear();
  while (ls >> kv) {
    const size_t eq = kv.find('=');
    const std::string k = kv.substr(0, eq), v = eq == std::string::npos ? "" : kv.substr(eq + 1);
    if (k == "resolution") e.resolution = atof(v.c_str());
    else if (k == "weight") e.weight = atof(v.c_str());
    else if (k == "preferred") { std::stringstream ss(v); std::string x; while (std::getline(ss, x, ',')) e.preferred.push_back(atoi(x.c_str())); }
    else if (k == "perms") e.permsPath = v;
    else if (k == "set") {
      e.set = v;
      if (v != "pdb" && v != "monomer" && v != "rna" && v != "disordered") { fprintf(stderr, "example %s: set=%s is not pdb, monomer, rna or disordered\n", e.input.c_str(), v.c_str()); exit(1); }
    }
    else { fprintf(stderr, "example %s: unknown field %s\n", e.input.c_str(), kv.c_str()); exit(1); }
  }
}
static Example loadExample(const std::string& line, bool padded, bool qblock) {
  Example e;
  { std::istringstream ls(line); ls >> e.input >> e.truthPath; }
  applyFields(e, line);
  for (auto it = M.index.begin(); it != M.index.end();) {
    const std::string& k = it->first;
    if (!k.compare(0, 6, "batch.") || !k.compare(0, 9, "template.") || !k.compare(0, 4, "job.") || !k.compare(0, 7, "oracle."))
      it = M.index.erase(it);
    else ++it;
  }
  M.load(e.input);
  e.n = (int)M.meta("batch.tokens"); e.dense = (int)M.meta("batch.dense");
  const int n = e.n; const size_t A = e.atoms(), nn = (size_t)n * n;
  e.profW = (int)(M.len("batch.profile") / n);
  e.aatype = copyOf(M.i("batch.aatype"), n); e.profile = copyOf(M.f("batch.profile"), (size_t)n * e.profW);
  e.delMean = copyOf(M.f("batch.deletionMean"), n); e.seqMask = copyOf(M.f("batch.seqMask"), n);
  e.ri = copyOf(M.i("batch.features.residueIndex"), n); e.ti = copyOf(M.i("batch.features.tokenIndex"), n);
  e.asym = copyOf(M.i("batch.features.asymId"), n); e.ent = copyOf(M.i("batch.features.entityId"), n);
  e.sym = copyOf(M.i("batch.features.symId"), n);
  e.isDna = copyOf(M.i("batch.isDna"), n); e.isRna = copyOf(M.i("batch.isRna"), n);
  e.isLigand = copyOf(M.i("batch.isLigand"), n); e.isMod = copyOf(M.i("batch.isModified"), n);
  e.refPos = copyOf(M.f("batch.refPos"), A * 3); e.refMask = copyOf(M.f("batch.refMask"), A);
  e.refCharge = copyOf(M.f("batch.refCharge"), A); e.predMask = copyOf(M.f("batch.predDenseAtomMask"), A);
  e.refElement = copyOf(M.i("batch.refElement"), A); e.names = copyOf(M.i("batch.refAtomNameChars"), A * 4);
  e.uid = copyOf(M.i("batch.refSpaceUid"), A);
  if (M.has("batch.bondMatrix")) e.bonds = copyOf(M.f("batch.bondMatrix"), nn);
  if (M.has("batch.bondOrderMatrix")) e.orders = copyOf(M.f("batch.bondOrderMatrix"), nn);
  {
    const int* pi = M.i("batch.tokenAtomsToPseudoBeta.indices"); const float* pmk = M.f("batch.tokenAtomsToPseudoBeta.mask");
    e.pbSlot.assign(n, -1);
    for (int t = 0; t < n; ++t) if (pmk[t] > 0) e.pbSlot[t] = pi[t] - t * e.dense;
  }
  // the MSA pool, unpacked
  {
    lf::MsaPoolView pool;
    if (M.has("batch.msaPacked")) { pool.packed = M.i("batch.msaPacked"); e.P = (int)(M.len("batch.msaPacked") / n); }
    else { pool.codes = M.i("batch.msa"); pool.deletion = M.f("batch.deletionMatrix"); pool.mask = M.f("batch.msaMask"); e.P = (int)(M.len("batch.msa") / n); }
    e.msaCode.resize((size_t)e.P * n); e.msaDel.resize((size_t)e.P * n); e.msaMask.resize((size_t)e.P * n);
    for (int r = 0; r < e.P; ++r)
      lf::unpackMsaRow(pool, r, n, &e.msaCode[(size_t)r * n], &e.msaDel[(size_t)r * n], &e.msaMask[(size_t)r * n]);
  }
  // the template slots
  {
    const int passes = (int)M.meta("template.passes");
    e.templates = (int)M.meta("template.templates");
    e.slots.resize(passes);
    for (int k = 0; k < passes; ++k) {
      const std::string S_ = "template." + std::to_string(k) + ".";
      auto& s = e.slots[k];
      s.repeat = M.meta(S_ + "repeat");
      s.aatype = copyOf(M.i(S_ + "aatype"), n);
      if (M.has(S_ + "distogramBin")) {
        s.geo = true;
        s.bin = copyOf(M.i(S_ + "distogramBin"), nn); s.pb = copyOf(M.f(S_ + "pseudoBetaMask2d"), nn);
        s.uv = copyOf(M.f(S_ + "unitVector"), nn * 3); s.bb = copyOf(M.f(S_ + "backboneMask2d"), nn);
      }
    }
  }
  // the truth: `chain residue name x y z`, chain the asymId, residue the residueIndex, name the slot's
  {
    std::ifstream f(e.truthPath);
    if (!f) { fprintf(stderr, "cannot read %s\n", e.truthPath.c_str()); exit(1); }
    std::map<std::tuple<int, int, std::string>, std::array<double, 3>> at;
    int c, r; std::string nm; double x, y, z;
    while (f >> c >> r >> nm >> x >> y >> z) at[{c, r, nm}] = {x, y, z};
    e.xt.assign(A * 3, 0); e.gm.assign(A, 0);
    int found = 0, wanted = 0;
    for (int t = 0; t < n; ++t)
      for (int k = 0; k < e.dense; ++k) {
        const size_t i = (size_t)t * e.dense + k;
        if (!(e.predMask[i] > 0)) continue;
        ++wanted;
        std::string name; for (int q = 0; q < 4; ++q) { int ch = e.names[i * 4 + q]; if (ch > 0) name += (char)(ch + 32); }
        auto it = at.find({e.asym[t], e.ri[t], name});
        if (it == at.end()) continue;
        for (int q = 0; q < 3; ++q) e.xt[i * 3 + q] = it->second[q];
        e.gm[i] = 1; ++found;
      }
    printf("example %s: %d tokens, %d of %d atoms resolved, %d MSA rows, %zu template slot(s)%s\n", e.input.c_str(), n, found,
           wanted, e.P, e.slots.size(), e.resolution > 0 ? (", resolution " + std::to_string(e.resolution).substr(0, 4) + " A").c_str() : ", no resolution");
    if (found < wanted / 4) { fprintf(stderr, "%s: the truth names %d of the batch's %d atoms - not this input's structure?\n", e.truthPath.c_str(), found, wanted); exit(1); }
  }
  // what the truth's permutation reads (of3perm.h): the whole entry's molecules, centres, names and symmetric namings
  {
    of3::PermInput& P = e.perm;
    P.n = n; P.dense = e.dense; P.ent = e.ent; P.sym = e.sym; P.asym = e.asym; P.ri = e.ri; P.xt = e.xt; P.gm = e.gm;
    const of3::Layout L = of3::makeLayout(n, e.dense, e.predMask.data(), e.names.data(), e.isDna.data(), e.isRna.data(),
                                          e.isLigand.data(), e.isMod.data(), e.asym.data(), nullptr);
    P.molTok.assign(n, 0); P.centre.assign(n, -1); P.ligand.assign(n, 0);
    std::map<int, int> at;
    for (int t = 0; t < n; ++t) {
      P.molTok[t] = at[e.asym[t]]++;
      P.centre[t] = L.atomized[t] ? L.start[t] : L.lddtRep[t];
      P.ligand[t] = e.isLigand[t] != 0;
    }
    P.name.assign(e.atoms(), "");
    for (size_t i = 0; i < e.atoms(); ++i) {
      if (!(e.predMask[i] > 0)) continue;
      for (int q = 0; q < 4; ++q) { int ch = e.names[i * 4 + q]; if (ch > 0) P.name[i] += (char)(ch + 32); }
    }
    if (!e.permsPath.empty()) {
      std::ifstream f(e.permsPath);
      if (!f) { fprintf(stderr, "cannot read %s\n", e.permsPath.c_str()); exit(1); }
      std::string line;
      while (std::getline(f, line)) {
        std::istringstream ls2(line); int c, r; ls2 >> c >> r;
        std::string rest; std::getline(ls2, rest);
        std::vector<std::vector<std::string>> namings;
        std::stringstream ps(rest); std::string part;
        while (std::getline(ps, part, '|')) {
          std::vector<std::string> names; std::stringstream ns(part); std::string x;
          while (std::getline(ns, x, ',')) { x.erase(0, x.find_first_not_of(' ')); x.erase(x.find_last_not_of(' ') + 1); names.push_back(x); }
          namings.push_back(names);
        }
        P.perms[{c, r}] = namings;
      }
    }
  }
  // its atom windows rebuilt here against the featuriser's own (the function a crop calls)
  {
    lf::Batch w = windows(e, padded, qblock);
    auto same = [&](const char* nm, const lf::Gather& g) {
      const std::string key = std::string("batch.") + nm;
      const size_t c = M.len(key + ".indices");
      bool ok = c == (size_t)g.count;
      for (size_t i = 0; ok && i < c; ++i) ok = M.i(key + ".indices")[i] == g.indices[i] && M.f(key + ".mask")[i] == g.mask[i];
      if (!ok) { fprintf(stderr, "%s: the rebuilt %s is not the featuriser's\n", e.input.c_str(), nm); exit(1); }
    };
    same("tokenAtomsToQueries", w.tokenAtomsToQueries); same("queriesToKeys", w.queriesToKeys);
    same("queriesToTokenAtoms", w.queriesToTokenAtoms); same("tokensToQueries", w.tokensToQueries);
    same("tokensToKeys", w.tokensToKeys); same("tokenAtomsToPseudoBeta", w.tokenAtomsToPseudoBeta);
  }
  return e;
}

// the example restricted to tokens `keep` (ascending): every per-token, per-atom and per-pair array and the MSA
// pool's columns cut to them, the identifiers (residue and token index, asym, entity) kept as they were - OpenFold3
// crops the structure and featurises the crop, and these are the arrays that featurisation reads
static Example subset(const Example& e, const std::vector<int>& keep, bool whole = true) {
  // (the crop's arrays only: a copy of the whole entry first would move its pair arrays, gigabytes on a large one)
  Example o;
  o.input = e.input; o.truthPath = e.truthPath; o.resolution = e.resolution; o.weight = e.weight; o.preferred = e.preferred;
  o.dense = e.dense; o.profW = e.profW; o.P = e.P; o.templates = e.templates; o.permsPath = e.permsPath; o.set = e.set;
  o.slots.resize(e.slots.size());
  for (size_t k = 0; k < e.slots.size(); ++k) { o.slots[k].repeat = e.slots[k].repeat; o.slots[k].geo = e.slots[k].geo; }
  const int m = (int)keep.size(), D = e.dense;
  o.n = m;
  auto tok = [&](const auto& v, size_t w) { std::remove_cv_t<std::remove_reference_t<decltype(v)>> r((size_t)m * w);
    for (int a = 0; a < m; ++a) for (size_t q = 0; q < w; ++q) r[(size_t)a * w + q] = v[(size_t)keep[a] * w + q]; return r; };
  auto pair = [&](const auto& v, size_t w) { std::remove_cv_t<std::remove_reference_t<decltype(v)>> r((size_t)m * m * w);
    for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) for (size_t q = 0; q < w; ++q)
      r[((size_t)a * m + b) * w + q] = v[((size_t)keep[a] * e.n + keep[b]) * w + q]; return r; };
  o.aatype = tok(e.aatype, 1); o.profile = tok(e.profile, e.profW); o.delMean = tok(e.delMean, 1); o.seqMask = tok(e.seqMask, 1);
  o.ri = tok(e.ri, 1); o.ti = tok(e.ti, 1); o.asym = tok(e.asym, 1); o.ent = tok(e.ent, 1); o.sym = tok(e.sym, 1);
  o.isDna = tok(e.isDna, 1); o.isRna = tok(e.isRna, 1); o.isLigand = tok(e.isLigand, 1); o.isMod = tok(e.isMod, 1);
  o.pbSlot = tok(e.pbSlot, 1);
  o.refPos = tok(e.refPos, (size_t)D * 3); o.refMask = tok(e.refMask, D); o.refCharge = tok(e.refCharge, D);
  o.predMask = tok(e.predMask, D); o.refElement = tok(e.refElement, D); o.names = tok(e.names, (size_t)D * 4); o.uid = tok(e.uid, D);
  o.xt = tok(e.xt, (size_t)D * 3); o.gm = tok(e.gm, D);
  if (!whole) return o;     // (per token and per atom alone: what a crop reads to choose)
  if (!e.bonds.empty()) o.bonds = pair(e.bonds, 1);
  if (!e.orders.empty()) o.orders = pair(e.orders, 1);
  o.msaCode.assign((size_t)e.P * m, 0); o.msaDel.assign((size_t)e.P * m, 0); o.msaMask.assign((size_t)e.P * m, 0);
  for (int r = 0; r < e.P; ++r) for (int a = 0; a < m; ++a) {
    const size_t s = (size_t)r * e.n + keep[a], d = (size_t)r * m + a;
    o.msaCode[d] = e.msaCode[s]; o.msaDel[d] = e.msaDel[s]; o.msaMask[d] = e.msaMask[s];
  }
  for (size_t k = 0; k < e.slots.size(); ++k) {
    auto& s = o.slots[k]; const auto& f = e.slots[k];
    s.aatype = tok(f.aatype, 1);
    if (f.geo) { s.bin = pair(f.bin, 1); s.pb = pair(f.pb, 1); s.uv = pair(f.uv, 3); s.bb = pair(f.bb, 1); }
  }
  return o;
}

// ---- OpenFold3's training templates (sample_templates, AF3 SI 2.4): for each chain, k = min(U{0..l}, n) of the l
// templates it has, drawn at random (not the top k inference takes) - here over the slots the input carries. Each new
// slot is assembled a chain at a time from the slot that chain drew (its intra-chain block; pairs across chains masked,
// as the features are), the rest padding (the gap restype, no geometry); all-padding slots folded into one pass.
static void sampleTemplates(Example& e, std::mt19937_64& rng) {
  const int n = e.n; const size_t nn = (size_t)n * n;
  std::vector<int> concrete;                 // each of the e.templates slots -> its pass
  for (size_t k = 0; k < e.slots.size(); ++k) for (int r = 0; r < (int)std::lround(e.slots[k].repeat); ++r) concrete.push_back((int)k);
  const int total = (int)concrete.size();
  if (total == 0) return;
  std::vector<int> chains; for (int t = 0; t < n; ++t) if (std::find(chains.begin(), chains.end(), e.asym[t]) == chains.end()) chains.push_back(e.asym[t]);
  const int GAP = 21;
  // per chain: the concrete slots that carry it (a pass with geometry and a residue of it not a gap)
  std::map<int, std::vector<int>> drawn;
  bool any = false;
  for (int c : chains) {
    std::vector<int> avail;
    for (int j = 0; j < total; ++j) {
      const auto& sl = e.slots[concrete[j]];
      if (!sl.geo) continue;
      bool has = false; for (int t = 0; t < n && !has; ++t) has = e.asym[t] == c && sl.aatype[t] != GAP;
      if (has) avail.push_back(j);
    }
    const int l = (int)avail.size();
    const int k = std::min(std::uniform_int_distribution<int>(0, l)(rng), total);
    std::shuffle(avail.begin(), avail.end(), rng);
    avail.resize(k);
    drawn[c] = avail;
    any |= l > 0;
  }
  if (!any) return;                          // (no chain has a template: the input's padding stands)
  std::vector<Example::Slot> real; int padding = 0;
  for (int j = 0; j < total; ++j) {
    Example::Slot o; o.aatype.assign(n, GAP);
    bool filled = false;
    for (int c : chains) filled |= (int)drawn[c].size() > j;
    if (!filled) { ++padding; continue; }
    o.geo = true; o.repeat = 1;
    o.bin.assign(nn, -1); o.pb.assign(nn, 0); o.uv.assign(nn * 3, 0); o.bb.assign(nn, 0);
    for (int c : chains) {
      if ((int)drawn[c].size() <= j) continue;
      const auto& src = e.slots[concrete[drawn[c][j]]];
      for (int a = 0; a < n; ++a) {
        if (e.asym[a] != c) continue;
        o.aatype[a] = src.aatype[a];
        for (int b = 0; b < n; ++b) {
          if (e.asym[b] != c) continue;
          const size_t ab = (size_t)a * n + b;
          o.bin[ab] = src.bin[ab]; o.pb[ab] = src.pb[ab]; o.bb[ab] = src.bb[ab];
          for (int q = 0; q < 3; ++q) o.uv[ab * 3 + q] = src.uv[ab * 3 + q];
        }
      }
    }
    real.push_back(std::move(o));
  }
  if (padding) { Example::Slot pad; pad.aatype.assign(n, GAP); pad.repeat = padding; real.push_back(std::move(pad)); }
  e.slots = std::move(real);
}

// OpenFold3's conformer augmentation (featurization/conformer.py: centre_random_augmentation on each conformer's
// ref_pos): every reference space centred over its live atoms, then a uniform random rotation and an N(0, 1)
// translation - every featurisation, so a fresh pose each step
static void augmentConformers(Example& e, std::mt19937_64& rng) {
  std::normal_distribution<double> nd(0, 1);
  std::map<int, std::vector<size_t>> spaces;
  for (size_t i = 0; i < e.atoms(); ++i) if (e.predMask[i] > 0) spaces[e.uid[i]].push_back(i);
  for (auto& [uid, atoms] : spaces) {
    double c[3] = {0, 0, 0}, live = 0;
    for (size_t i : atoms) if (e.refMask[i] > 0) { for (int q = 0; q < 3; ++q) c[q] += e.refPos[i * 3 + q]; live += 1; }
    if (live == 0) continue;
    for (double& v : c) v /= live;
    double q4[4], qn = 0; for (double& v : q4) { v = nd(rng); qn += v * v; } qn = std::sqrt(qn); for (double& v : q4) v /= qn;
    const double R[3][3] = {{1 - 2 * (q4[2] * q4[2] + q4[3] * q4[3]), 2 * (q4[1] * q4[2] - q4[0] * q4[3]), 2 * (q4[1] * q4[3] + q4[0] * q4[2])},
                            {2 * (q4[1] * q4[2] + q4[0] * q4[3]), 1 - 2 * (q4[1] * q4[1] + q4[3] * q4[3]), 2 * (q4[2] * q4[3] - q4[0] * q4[1])},
                            {2 * (q4[1] * q4[3] - q4[0] * q4[2]), 2 * (q4[2] * q4[3] + q4[0] * q4[1]), 1 - 2 * (q4[1] * q4[1] + q4[2] * q4[2])}};
    const double t[3] = {nd(rng), nd(rng), nd(rng)};
    for (size_t i : atoms) {
      if (!(e.refMask[i] > 0)) { for (int q = 0; q < 3; ++q) e.refPos[i * 3 + q] = 0; continue; }   // (masked: 0, as there)
      double p[3]; for (int q = 0; q < 3; ++q) p[q] = e.refPos[i * 3 + q] - c[q];
      for (int q = 0; q < 3; ++q) { double v = t[q]; for (int u = 0; u < 3; ++u) v += R[q][u] * p[u]; e.refPos[i * 3 + q] = (float)v; }
    }
  }
}

// ---- OpenFold3's crops (core/data/primitives/structure/cropping.py), over the token centre atoms (CA, C1', an
// atomised token's atom) the truth resolves
struct CropWeights { double contiguous = 0.2, spatial = 0.4, interface = 0.4; };
static std::vector<int> cropTokens(const Example& e, const of3::Layout& L, int budget, const CropWeights& cw, std::mt19937_64& rng,
                                   std::string& strategy) {
  const int n = e.n;
  std::vector<int> all(n); for (int t = 0; t < n; ++t) all[t] = t;
  if (n <= budget) { strategy = "whole"; return all; }
  std::uniform_real_distribution<double> U(0, 1);
  const double r = U(rng) * (cw.contiguous + cw.spatial + cw.interface);
  strategy = r < cw.contiguous ? "contiguous" : r < cw.contiguous + cw.spatial ? "spatial" : "spatial_interface";
  std::vector<char> in(n, 0);
  if (strategy == "contiguous") {
    // AF-Multimer's Algorithm 1 as OpenFold3 fixed it: chains in a random order, each a random-length run
    std::vector<int> chains; for (int t = 0; t < n; ++t) if (chains.empty() || chains.back() != e.asym[t]) chains.push_back(e.asym[t]);
    std::sort(chains.begin(), chains.end()); chains.erase(std::unique(chains.begin(), chains.end()), chains.end());
    std::shuffle(chains.begin(), chains.end(), rng);
    int added = 0, remaining = n;
    for (int c : chains) {
      std::vector<int> toks; for (int t = 0; t < n; ++t) if (e.asym[t] == c) toks.push_back(t);
      const int len = (int)toks.size();
      remaining -= len;
      const int hi = std::min(budget - added, len), lo = std::min(len, std::max(0, budget - added - remaining));
      const int size = std::uniform_int_distribution<int>(lo, std::max(lo, hi))(rng);
      added += size;
      const int start = std::uniform_int_distribution<int>(0, len - size)(rng);
      for (int k = start; k < start + size; ++k) in[toks[k]] = 1;
    }
  } else {
    auto centre = [&](int t) { return L.atomized[t] ? L.start[t] : L.lddtRep[t]; };
    std::vector<int> centres;
    for (int t = 0; t < n; ++t) { const int a = centre(t); if (a >= 0 && e.gm[a] > 0) centres.push_back(t); }
    if (centres.empty()) { fprintf(stderr, "%s: no resolved token centre atoms\n", e.input.c_str()); exit(1); }
    auto pos = [&](int t) { const int a = centre(t); return std::array<double, 3>{e.xt[a * 3], e.xt[a * 3 + 1], e.xt[a * 3 + 2]}; };
    auto d2 = [&](int s, int t) { auto p = pos(s), q = pos(t); double v = 0; for (int k = 0; k < 3; ++k) v += (p[k] - q[k]) * (p[k] - q[k]); return v; };
    std::vector<int> pref;
    for (int t : centres) if (e.preferred.empty() || std::count(e.preferred.begin(), e.preferred.end(), e.asym[t])) pref.push_back(t);
    if (pref.empty()) pref = centres;
    bool multi = false; for (int t : centres) multi |= e.asym[t] != e.asym[centres[0]];
    const bool pairPreferred = e.preferred.size() == 2;
    if (pairPreferred || (strategy == "spatial_interface" && multi)) {
      // the preferred centres within 15 A of another chain's centre (between the pair, when a pair is preferred)
      const auto& pool = pairPreferred ? pref : centres;
      std::vector<int> face;
      for (int s : pref) for (int t : pool) if (e.asym[s] != e.asym[t] && d2(s, t) < 15.0 * 15.0) { face.push_back(s); break; }
      if (!face.empty()) pref = face;
    }
    const int ref = pref[std::uniform_int_distribution<size_t>(0, pref.size() - 1)(rng)];
    std::vector<std::pair<double, int>> by; for (int t : centres) by.push_back({d2(ref, t), t});
    std::stable_sort(by.begin(), by.end());
    for (int k = 0; k < std::min<int>(budget, (int)by.size()); ++k) in[by[k].second] = 1;
  }
  std::vector<int> keep; for (int t = 0; t < n; ++t) if (in[t]) keep.push_back(t);
  return keep;
}

// the permutation's input restricted to tokens `keep` - each token's position in its molecule kept as the whole
// entry numbers it, so a chain-cropped truth's copies still align position by position
static of3::PermInput subsetPerm(const of3::PermInput& P, const std::vector<int>& keep) {
  of3::PermInput o; o.n = (int)keep.size(); o.dense = P.dense; o.perms = P.perms;
  for (int t : keep) {
    o.ent.push_back(P.ent[t]); o.sym.push_back(P.sym[t]); o.asym.push_back(P.asym[t]); o.molTok.push_back(P.molTok[t]);
    o.ri.push_back(P.ri[t]); o.ligand.push_back(P.ligand[t]);
    o.centre.push_back(P.centre[t] < 0 ? -1 : P.centre[t] - t * P.dense + (int)(o.ent.size() - 1) * P.dense);
    for (int s = 0; s < P.dense; ++s) {
      const size_t i = (size_t)t * P.dense + s;
      o.name.push_back(P.name[i]); o.gm.push_back(P.gm[i]);
      for (int q = 0; q < 3; ++q) o.xt.push_back(P.xt[i * 3 + q]);
    }
  }
  return o;
}
// OpenFold3's chain crop (crop_chainwise, AF3 SI 2.5.4): past 20 polymer chains, the 20 nearest a random preferred
// (interface) centre - chains by the first of their centres in distance order, ligands held out - plus the preferred
// chains and every ligand chain within 5 A of a kept one (any atoms); all tokens when there are no more than 20
static std::vector<int> chainCrop(const Example& e, const of3::Layout& L, std::mt19937_64& rng, int nChains = 20) {
  std::vector<int> all(e.n); for (int t = 0; t < e.n; ++t) all[t] = t;
  std::set<int> poly, lig;
  for (int t = 0; t < e.n; ++t) (e.isLigand[t] ? lig : poly).insert(e.asym[t]);
  for (int c : lig) poly.erase(c);
  if ((int)poly.size() <= nChains) return all;
  auto centre = [&](int t) { return L.atomized[t] ? L.start[t] : L.lddtRep[t]; };
  std::vector<int> centres;
  for (int t = 0; t < e.n; ++t) { const int a = centre(t); if (a >= 0 && e.gm[a] > 0) centres.push_back(t); }
  auto d2 = [&](int s, int t) { double v = 0; for (int q = 0; q < 3; ++q) { double x = e.xt[centre(s) * 3 + q] - e.xt[centre(t) * 3 + q]; v += x * x; } return v; };
  std::vector<int> pref;
  for (int t : centres) if (e.preferred.empty() || std::count(e.preferred.begin(), e.preferred.end(), e.asym[t])) pref.push_back(t);
  if (pref.empty()) pref = centres;
  if (e.preferred.size() == 2) {
    std::vector<int> face;
    for (int s : pref) for (int t : pref) if (e.asym[s] != e.asym[t] && d2(s, t) < 15.0 * 15.0) { face.push_back(s); break; }
    if (!face.empty()) pref = face;
  }
  const int ref = pref[std::uniform_int_distribution<size_t>(0, pref.size() - 1)(rng)];
  std::vector<std::pair<double, int>> by;
  for (int t : centres) if (!e.isLigand[t]) by.push_back({d2(ref, t), t});
  std::stable_sort(by.begin(), by.end());
  std::vector<int> chosen; std::set<int> seen;
  for (auto& [d, t] : by) if (seen.insert(e.asym[t]).second) { chosen.push_back(e.asym[t]); if ((int)chosen.size() == nChains) break; }
  std::set<int> keepChains(chosen.begin(), chosen.end());
  for (int c : e.preferred) keepChains.insert(c);
  for (int c : lig) {          // a ligand within 5 A of a kept polymer chain (resolved atoms)
    bool near = false;
    for (int t = 0; t < e.n && !near; ++t) {
      if (e.asym[t] != c) continue;
      for (int s = 0; s < e.n && !near; ++s) {
        if (!std::count(chosen.begin(), chosen.end(), e.asym[s])) continue;
        for (int a = 0; a < e.dense && !near; ++a) {
          const size_t i = (size_t)t * e.dense + a; if (!(e.gm[i] > 0)) continue;
          for (int b = 0; b < e.dense && !near; ++b) {
            const size_t j = (size_t)s * e.dense + b; if (!(e.gm[j] > 0)) continue;
            double v = 0; for (int q = 0; q < 3; ++q) { double x = e.xt[i * 3 + q] - e.xt[j * 3 + q]; v += x * x; }
            near = v <= 25.0;
          }
        }
      }
    }
    if (near) keepChains.insert(c);
  }
  std::vector<int> keep; for (int t = 0; t < e.n; ++t) if (keepChains.count(e.asym[t])) keep.push_back(t);
  return keep;
}

// ---- an example on the device: everything the modules read, and what was allocated for it
struct Dev {
  int n = 0, dense = 0, S = 0;
  g::af3::AtomBatch b;
  int *ri, *ti, *asym, *ent, *sym, *aatype, *msaRows;
  g::real *profile, *delMean, *seqMask, *pairMask, *pairMaskT, *bonds = nullptr, *orders = nullptr, *deletion, *msaMask;
  std::vector<g::af3::TemplateSlot> slots; int templates = 0;
  std::vector<void*> owned;
  template <class T> T* up(const std::vector<T>& h) {
    T* d; GK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
    if (!h.empty()) GK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    owned.push_back(d); return d;
  }
  g::real* upR(const std::vector<float>& h) { return up(std::vector<g::real>(h.begin(), h.end())); }
  g::real* upR(const std::vector<double>& h) { return up(std::vector<g::real>(h.begin(), h.end())); }
  g::real* zeros(size_t k) { g::real* d = dzeros(k); owned.push_back(d); return d; }
  void release() { for (void* p : owned) GK(cudaFree(p)); owned.clear(); }
};
static g::af3::Gather devGather(Dev& d, const lf::Gather& q) {
  return g::af3::Gather{d.up(q.indices), d.upR(q.mask), q.count};
}
static void upload(Dev& d, const Example& e, bool padded, bool qblock, int S) {
  d.n = e.n; d.dense = e.dense; d.S = S;
  const int n = e.n; const size_t nn = (size_t)n * n;
  lf::Batch w = windows(e, padded, qblock);
  g::af3::AtomBatch& b = d.b;
  b.tokens = n; b.dense = e.dense; b.subsets = w.subsets; b.Q = lf::QUERIES; b.K = w.keys;
  b.t2q = devGather(d, w.tokenAtomsToQueries); b.q2k = devGather(d, w.queriesToKeys); b.q2t = devGather(d, w.queriesToTokenAtoms);
  b.tq = devGather(d, w.tokensToQueries); b.tk = devGather(d, w.tokensToKeys);
  b.refPos = d.upR(e.refPos); b.refMask = d.upR(e.refMask); b.refCharge = d.upR(e.refCharge); b.atomMask = d.upR(e.predMask);
  b.uid = d.upR(std::vector<double>(e.uid.begin(), e.uid.end()));
  b.refElement = d.up(e.refElement); b.nameChars = d.up(e.names);
  d.ri = d.up(e.ri); d.ti = d.up(e.ti); d.asym = d.up(e.asym); d.ent = d.up(e.ent); d.sym = d.up(e.sym); d.aatype = d.up(e.aatype);
  d.profile = d.upR(e.profile); d.delMean = d.upR(e.delMean); d.seqMask = d.upR(e.seqMask);
  std::vector<double> pm(nn), pt(nn);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) pm[(size_t)i * n + j] = pt[(size_t)j * n + i] = e.seqMask[i] * e.seqMask[j];
  d.pairMask = d.upR(pm); d.pairMaskT = d.upR(pt);
  if (!e.bonds.empty()) d.bonds = d.upR(e.bonds);
  if (!e.orders.empty()) d.orders = d.upR(e.orders);
  d.msaRows = d.up(std::vector<int>((size_t)S * n, 0)); d.deletion = d.zeros((size_t)S * n); d.msaMask = d.zeros((size_t)S * n);
  d.templates = e.templates;
  d.slots.resize(e.slots.size());
  for (size_t k = 0; k < e.slots.size(); ++k) {
    auto& s = d.slots[k]; const auto& f = e.slots[k];
    s.repeat = (g::real)f.repeat; s.aatype = d.up(f.aatype);
    if (f.geo) { s.bin = d.up(f.bin); s.pb = d.upR(f.pb); s.uv = d.upR(f.uv); s.bb = d.upR(f.bb); }
  }
}
// a cycle's MSA rows (MSAModuleEmbedder._subsample_all_msa): the valid rows (any unmasked cell) in a random order,
// then invalid ones, S of them; into d's rows
static void drawMsa(Dev& d, const Example& e, std::mt19937_64& rng) {
  const int n = e.n;
  std::vector<int> valid, invalid;
  for (int r = 0; r < e.P; ++r) {
    bool any = false; for (int t = 0; t < n && !any; ++t) any = e.msaMask[(size_t)r * n + t] > 0;
    (any ? valid : invalid).push_back(r);
  }
  std::shuffle(valid.begin(), valid.end(), rng); std::shuffle(invalid.begin(), invalid.end(), rng);
  std::vector<int> chosen(valid.begin(), valid.begin() + std::min<size_t>(valid.size(), d.S));
  for (size_t k = 0; (int)chosen.size() < d.S && k < invalid.size(); ++k) chosen.push_back(invalid[k]);
  std::vector<int> codes((size_t)d.S * n); std::vector<g::real> del((size_t)d.S * n), mask((size_t)d.S * n);
  for (int k = 0; k < d.S; ++k) for (int t = 0; t < n; ++t) {
    const size_t s = (size_t)chosen[k] * n + t, o = (size_t)k * n + t;
    codes[o] = e.msaCode[s]; del[o] = e.msaDel[s]; mask[o] = e.msaMask[s];
  }
  GK(cudaMemcpy(d.msaRows, codes.data(), codes.size() * 4, cudaMemcpyHostToDevice));
  GK(cudaMemcpy(d.deletion, del.data(), del.size() * sizeof(g::real), cudaMemcpyHostToDevice));
  GK(cudaMemcpy(d.msaMask, mask.data(), mask.size() * sizeof(g::real), cudaMemcpyHostToDevice));
}
static of3::Layout layoutOf(const Example& e) {
  return of3::makeLayout(e.n, e.dense, e.predMask.data(), e.names.data(), e.isDna.data(), e.isRna.data(), e.isLigand.data(),
                         e.isMod.data(), e.asym.data(), e.bonds.empty() ? nullptr : e.bonds.data());
}

int main(int argc, char** argv) {
  std::string bundle = "model-openbind0-f32", family = "openbind0", examplesPath, savePath, resumePath;
  int steps = 30, samples = 48, batch = 1, crop = 384, saveEvery = 0, checkCkpt = -1, checkPerm = -1, checkBf16 = -1; std::string init = "bundle";
  bool dropout = true, dropInFixed = false, bf16Trunk = false;
  int world = 1, rank = 0; std::string dp = "nccl", rendezvous;      // data parallel (of3dist.cuh)
  bool sameDraws = false;   // --same-draws: every rank rank 0's examples and draws (the averaging's check: one process's step)   // (dropInFixed: --check-ckpt's fixed draws with the masks on)
  std::string valPath; int valEvery = 0, valSamples = 5, valSteps = 200, valCrop = 1536; unsigned seed = 1; bool smoke = false, ckpt = true;
  double lrMax = 1.8e-3, bondWeight = 0; long warmup = 1000;
  for (int i = 1; i < argc; ++i) {
    auto opt = [&](const char* k) { const size_t l = strlen(k); return !strncmp(argv[i], k, l) ? argv[i] + l : nullptr; };
    if (auto v = opt("--bundle=")) bundle = v;
    else if (auto v = opt("--family=")) family = v;
    else if (auto v = opt("--examples=")) examplesPath = v;
    else if (auto v = opt("--steps=")) steps = atoi(v);
    else if (auto v = opt("--samples=")) samples = atoi(v);
    else if (auto v = opt("--batch=")) batch = std::max(1, atoi(v));
    else if (auto v = opt("--crop=")) crop = atoi(v);
    else if (auto v = opt("--seed=")) seed = (unsigned)atoi(v);
    else if (auto v = opt("--lr=")) lrMax = atof(v);
    else if (auto v = opt("--warmup=")) warmup = atol(v);
    else if (auto v = opt("--bond=")) bondWeight = atof(v);
    else if (auto v = opt("--save=")) savePath = v;
    else if (auto v = opt("--save-every=")) saveEvery = atoi(v);
    else if (auto v = opt("--resume=")) resumePath = v;
    else if (!strcmp(argv[i], "--smoke")) smoke = true;
    else if (!strcmp(argv[i], "--no-ckpt")) ckpt = false;
    else if (!strcmp(argv[i], "--no-dropout")) dropout = false;
    else if (!strcmp(argv[i], "--bf16")) bf16Trunk = true;
    else if (auto v = opt("--world=")) world = atoi(v);
    else if (auto v = opt("--rank=")) rank = atoi(v);
    else if (auto v = opt("--dp=")) dp = v;
    else if (auto v = opt("--rendezvous=")) rendezvous = v;
    else if (!strcmp(argv[i], "--same-draws")) sameDraws = true;
    else if (auto v = opt("--check-ckpt=")) checkCkpt = atoi(v);
    else if (auto v = opt("--check-perm=")) checkPerm = atoi(v);
    else if (auto v = opt("--check-bf16=")) checkBf16 = atoi(v);
    else if (auto v = opt("--init=")) init = v;
    else if (auto v = opt("--val=")) valPath = v;
    else if (auto v = opt("--val-every=")) valEvery = atoi(v);
    else if (auto v = opt("--val-samples=")) valSamples = atoi(v);
    else if (auto v = opt("--val-steps=")) valSteps = atoi(v);
    else if (auto v = opt("--val-crop=")) valCrop = atoi(v);
    else if (!strcmp(argv[i], "--tf32")) g::TF32 = true;
    else { fprintf(stderr, "train-of3: unknown argument %s\n", argv[i]); return 1; }
  }
  if (examplesPath.empty()) { fprintf(stderr, "train-of3: --examples=<file> is required\n"); return 1; }
  std::vector<std::string> lines;
  {
    std::ifstream f(examplesPath); std::string l;
    if (!f) { fprintf(stderr, "cannot read %s\n", examplesPath.c_str()); return 1; }
    while (std::getline(f, l)) if (!l.empty() && l[0] != '#') lines.push_back(l);
    if (lines.empty()) { fprintf(stderr, "%s names no example\n", examplesPath.c_str()); return 1; }
  }
  // data parallel: a rank's GPU (NCCL: one each), its own draws, rank 0 alone writing the log and the checkpoint
  if (world > 1) {
    if (rank < 0 || rank >= world || rendezvous.empty() || (dp != "nccl" && dp != "files")) {
      fprintf(stderr, "train-of3: --world=%d needs --rank in [0, world), --rendezvous=<dir> and --dp=nccl|files\n", world); return 1;
    }
    int devices = 1; GK(cudaGetDeviceCount(&devices));
    if (dp == "nccl") GK(cudaSetDevice(rank % devices));
    if (rank > 0) { if (!freopen("/dev/null", "w", stdout)) return 1; }
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  std::unique_ptr<g::Comm> comm;
  if (world > 1) {
    if (dp == "nccl") comm.reset(new g::NcclComm(rank, world, rendezvous));
    else comm.reset(new g::FileComm(rank, world, rendezvous));
  }
  M.loadBundle(bundle, "", [&](const std::map<std::string, std::vector<long long>>& shapes) {
    lf::weights::Shapes S; S.shape = shapes; return lf::weights::af3WeightLines(family, S); });
  const lf::FamilyFlags& FF = lf::familyFor(family);
  const bool padded = FF.paddedAtomKeys, qblock = FF.qblockAtomKeys;
  std::vector<Example> examples;
  examples.push_back(loadExample(lines[0], padded, qblock));        // (the first in M while the modules read its metas)
  for (const char* f : {"trunk.dialect.chaiTokenEmbedding", "trunk.dialect.targetFeatAtomOnly", "trunk.dialect.structuralTokens",
                        "trunk.dialect.chaiAtomStack", "trunk.dialect.recycleFromInit"})
    if (M.flag(f)) { fprintf(stderr, "train-of3: %s is not composed here yet\n", f); return 1; }
  setAdaMode(false);
  const int dense0 = examples[0].dense;
  auto has = [&](const std::string& k) { return M.has(k); };
  g::Params ps;
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the bundle, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    std::vector<float> h(len); GK(cudaMemcpy(h.data(), W(nm), len * 4, cudaMemcpyDeviceToHost));
    std::vector<g::real> r(h.begin(), h.end());
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  };
  auto stackFlags = [&](g::af3::AtomStack& st, const std::string& B0) {
    for (auto& b : st.blocks) {
      b.keyMasked = M.flag(B0 + ".keyMaskedAtomAttention"); b.chained = M.flag(B0 + ".chainedAtomLayerNorm");
      b.noResidual = M.flag(B0 + ".diffusionNoResidual"); b.maskActPerBlock = M.flag(B0 + ".maskAtomActPerBlock");
    }
  };
  // ---- target_feat
  g::af3::TargetFeat tfm;
  {
    const std::string E = "targetFeat.encoder";
    g::af3::DiffEncoder& e = tfm.enc;
    e.C = (int)M.meta(E + ".channels"); e.Cp = (int)M.meta(E + ".pairChannels"); e.H = (int)M.meta(E + ".heads");
    e.D = (int)M.meta(E + ".dimension"); e.perToken = (int)M.meta(E + ".perTokenChannels");
    e.Cst = has(E + ".embedTrunkSingleCond") ? (int)(M.at(E + ".embedTrunkSingleCond").length / e.C) : 0;
    e.Cz = has(E + ".embedTrunkPairCond") ? (int)(M.at(E + ".embedTrunkPairCond").length / e.Cp) : 0;
    e.rawCharge = M.flag("trunk.dialect.rawRefCharge"); e.preTrunkQuery = M.flag("trunk.dialect.preTrunkQuery");
    e.keysUidFromQueries = M.flag("trunk.dialect.keysUidFromQueriesLayout"); e.maskPaddedKeys = M.flag("trunk.dialect.maskPaddedKeys");
    int nb = 0; while (M.has(E + ".blocks." + std::to_string(nb) + ".qProjection")) ++nb;
    g::af3::forEachParam(e, E, "targetFeat.reference", nb, M.flag(E + ".pairNormPerBlock"), M.has(E + ".blocks.0.ffwAToB"),
                         M.has(E + ".blocks.0.queryLayerNormScale"), !M.has(E + ".blocks.0.Transition2"), has, load);
    stackFlags(e.stack, E + ".blocks.0");
  }
  // ---- the trunk embedder and the distogram head
  const std::string TE = "trunk.embedder.";
  g::af3::TrunkEmbedder te;
  te.C = (int)M.meta(TE + "pairChannels"); te.Cs = (int)M.meta(TE + "singleChannels"); te.Cm = (int)M.meta(TE + "msaChannels");
  te.F = (int)M.meta(TE + "targetFeatWidth"); te.fromSingle = M.flag("trunk.dialect.pairInitFromSingle");
  te.msaWidth = (int)(M.at(TE + "msaActivations").length / te.Cm);
  g::af3::forEachParam(te, has, load);
  g::af3::DistogramHead dh; dh.C = te.C; dh.bins = (int)M.meta("trunk.distogram.bins");
  dh.scale = M.flag("trunk.dialect.mlpDistogram") ? 0.5f : 1.f;
  if (M.has("trunk.distogram.hidden")) dh.Hd = (int)M.at("trunk.distogram.hiddenBias").length;
  g::af3::forEachParam(dh, has, load);
  // ---- the template embedder (test_af3tmpl.cu's setup)
  g::af3::TemplateEmbedder tm;
  {
    const std::string P = "trunk.template.";
    tm.fused = M.flag(P + "fused"); tm.outer = M.flag("template.outerResidual"); tm.chaiMask = M.flag("trunk.dialect.chaiTemplates");
    tm.d = g::af3::Dialect{M.flag("trunk.dialect.swapTransposedBias"), M.flag("trunk.dialect.triangleMulDivideByLength")};
    const int Ct = (int)M.meta(P + "channels"), Cq = (int)(M.at(P + "queryEmbeddingNormScale").length);
    const int width = (int)M.meta("template.featureWidth");
    const int tb = tm.fused ? 0 : (int)(M.at(P + "templatePairEmbedding0").length / Ct);
    int nb = 0; while (M.has(P + "blocks." + std::to_string(nb) + ".pairTransition.transition1")) ++nb;
    const std::string B0 = P + "blocks.0";
    const int Hp = (int)(M.at(B0 + ".pairAttention1.pairBiasProjection").length / Ct);
    const int Dp = (int)(M.at(B0 + ".pairAttention1.qProjection").length / Ct / Hp);
    const int tf_ = (int)(M.at(B0 + ".pairTransition.transition1").length / (2 * (size_t)Ct * Ct));
    g::af3::forEachParam(tm, Cq, Ct, width, tb, nb, Hp, Dp, tf_, M.has(B0 + ".pairAttention1.gatingQueryBias"),
                         M.flag("trunk.dialect.parallelPairformer"), has, load);
    if (tm.fused) { fprintf(stderr, "train-of3: the fused template features are not composed here yet\n"); return 1; }
  }
  // ---- the MSA stack and the pairformer
  const g::af3::Dialect dl{M.flag("trunk.dialect.swapTransposedBias"), M.flag("trunk.dialect.triangleMulDivideByLength")};
  const bool doubleAdd = M.flag("trunk.dialect.msaDoubleAddPair");
  int nMsa = 0; while (M.has("trunk.msaBlocks." + std::to_string(nMsa) + ".pairChannels")) ++nMsa;
  std::vector<g::af3::MsaBlock> msaB(nMsa);
  for (int k = 0; k < nMsa; ++k) {
    const std::string pre = "trunk.msaBlocks." + std::to_string(k);
    g::af3::MsaBlock& b = msaB[k];
    b.updateFirst = M.flag("trunk.dialect.msaUpdateBeforeOuterProduct"); b.chai = M.flag("trunk.dialect.groupedOuterProduct");
    if (b.chai) { fprintf(stderr, "train-of3: chai-1's grouped MSA is not composed here yet\n"); return 1; }
    g::af3::forEachParam(b, pre, (int)M.meta(pre + ".pairChannels"), (int)M.meta(pre + ".msaChannels"),
                         (int)M.meta(pre + ".pairAttention1.heads"), (int)M.meta(pre + ".pairAttention1.dimension"),
                         (int)M.meta(pre + ".msaAttention1.heads"), (int)M.meta(pre + ".msaAttention1.dimension"),
                         (int)M.meta(pre + ".outerProductMean.outerChannels"), 1,
                         M.has(pre + ".outerProductMean.leftProjectionBias"), M.has(pre + ".pairAttention1.gatingQueryBias"), load);
    b.opm.biasAfterNorm = M.flag("trunk.dialect.opmBiasAfterNorm");
    b.att.chaiMask = M.flag("trunk.dialect.chaiMsaFeatures");
  }
  int nPf = 0; while (M.has("trunk.pairformerBlocks." + std::to_string(nPf) + ".singleChannels")) ++nPf;
  if (M.has("trunk.pairMixerBlocks.0.singleChannels")) { fprintf(stderr, "train-of3: Mol-1's PairMixer is not composed here yet\n"); return 1; }
  std::vector<g::af3::Pairformer> pf(nPf);
  for (int k = 0; k < nPf; ++k) {
    const std::string pre = "trunk.pairformerBlocks." + std::to_string(k);
    const int Cz = (int)M.meta(pre + ".pairChannels"), Cs = (int)M.meta(pre + ".singleChannels");
    g::af3::forEachParam(pf[k], pre, Cz, Cs, (int)M.meta(pre + ".pairAttention1.heads"), (int)M.meta(pre + ".pairAttention1.dimension"),
                         (int)M.meta(pre + ".singleAttention.heads"), (int)M.meta(pre + ".singleAttention.dimension"),
                         (int)(M.at(pre + ".pairTransition.transition1").length / (2 * (size_t)Cz * Cz)),
                         (int)(M.at(pre + ".singleTransition.transition1").length / (2 * (size_t)Cs * Cs)),
                         M.has(pre + ".pairAttention1.gatingQueryBias"), true, load, true, M.flag("trunk.dialect.parallelPairformer"));
  }
  // ---- the denoiser (test_af3denoise.cu's setup)
  g::af3::Denoiser dn;
  {
    const std::string P = "diffusion.conditioning", E = "diffusion.encoder", Dd = "diffusion.decoder", Tn = "diffusion.transformer";
    g::af3::DiffCond& c = dn.cond;
    c.Cz = (int)M.meta(P + ".pairChannels"); c.Cs = (int)M.meta(P + ".seqChannels");
    c.Czt = (int)M.meta(P + ".trunkPairChannels"); c.Cst = (int)M.meta(P + ".trunkSingleChannels");
    c.F = (int)M.meta(P + ".targetFeatWidth"); c.rel = (int)M.meta(P + ".relativeWidth");
    c.nc = (int)M.at(P + ".fourierWeight").length; c.pad = M.flag("trunk.dialect.padSingleCondUnknownDna");
    c.split = has(P + ".zTrunkProjection"); c.relpe = !c.split && has(P + ".relpeProjection");
    g::af3::DiffEncoder& e = dn.enc;
    e.C = (int)M.meta(E + ".channels"); e.Cp = (int)M.meta(E + ".pairChannels"); e.H = (int)M.meta(E + ".heads");
    e.D = (int)M.meta(E + ".dimension"); e.perToken = (int)M.meta(E + ".perTokenChannels");
    e.Cst = (int)M.meta(E + ".trunkSingleChannels"); e.Cz = (int)M.meta(E + ".trunkPairChannels");
    e.rawCharge = M.flag("trunk.dialect.rawRefCharge"); e.preTrunkQuery = M.flag("trunk.dialect.preTrunkQuery");
    e.keysUidFromQueries = M.flag("trunk.dialect.keysUidFromQueriesLayout"); e.maskPaddedKeys = M.flag("trunk.dialect.maskPaddedKeys");
    int nbE = 0; while (M.has(E + ".blocks." + std::to_string(nbE) + ".qProjection")) ++nbE;
    int nbD = 0; while (M.has(Dd + ".blocks." + std::to_string(nbD) + ".qProjection")) ++nbD;
    dn.decStack.C = (int)M.meta(Dd + ".channels"); dn.decStack.Cp = (int)M.meta(Dd + ".pairChannels");
    dn.decStack.H = (int)M.meta(Dd + ".heads"); dn.decStack.D = (int)M.meta(Dd + ".dimension");
    g::af3::DiffTransformer& t = dn.tr;
    t.C = (int)M.meta(Tn + ".channels"); t.Cc = (int)M.meta(Tn + ".condChannels"); t.Cz = (int)M.meta(Tn + ".pairChannels");
    t.H = (int)M.meta(Tn + ".heads"); t.D = (int)M.meta(Tn + ".dimension"); t.perSuper = (int)M.meta(Tn + ".blocksPerSuperBlock");
    t.factor = (int)M.meta(Tn + ".transitionFactor"); t.noResidual = M.flag(Tn + ".noResidual");
    int nSuper = 0; while (M.has(Tn + ".superBlocks." + std::to_string(nSuper) + ".pairLogitsProjection")) ++nSuper;
    g::af3::forEachParam(dn, nbE, nbD, nSuper, M.flag(E + ".pairNormPerBlock"), M.has(E + ".blocks.0.ffwAToB"),
                         M.has(E + ".blocks.0.queryLayerNormScale"), !M.has(E + ".blocks.0.Transition2"),
                         M.has(Tn + ".superBlocks.0.blocks.0.ffwAToB"), M.has(Tn + ".superBlocks.0.blocks.0.queryLayerNormScale"), has, load);
    stackFlags(e.stack, E + ".blocks.0");
    stackFlags(dn.decStack, Dd + ".blocks.0");
  }
  // ---- the confidence head (test_af3conf.cu's setup)
  g::af3::ConfidenceHead ch;
  {
    const std::string P = "confidence";
    ch.C = (int)M.meta(P + ".pairChannels"); ch.Cs = (int)M.meta(P + ".singleChannels"); ch.F = (int)M.meta(P + ".targetFeatWidth");
    ch.reembed = M.flag("trunk.dialect.reembedConfidencePair"); ch.globalNorm = M.flag("trunk.dialect.confidenceGlobalNorm");
    ch.binMode = M.flag("trunk.dialect.confidenceCaDgram") ? g::af3::BIN_CA : g::af3::BIN_AF3;
    ch.bins = ch.reembed ? 64 : (int)(M.at(P + ".distogramFeatProject").length / ch.C); ch.dense = dense0;
    ch.preSym = M.flag("trunk.dialect.preSymmetrisedPde");
    ch.d = g::af3::Dialect{M.flag("trunk.dialect.swapTransposedBias"), M.flag("trunk.dialect.triangleMulDivideByLength")};
    int nb = 0; while (M.has(P + ".blocks." + std::to_string(nb) + ".singleChannels")) ++nb;
    const std::string B0 = P + ".blocks.0";
    const bool hasGrid = M.has(B0 + ".pairAttention1.heads");
    g::af3::forEachParam(ch, nb, hasGrid ? (int)M.meta(B0 + ".pairAttention1.heads") : 0,
                         hasGrid ? (int)M.meta(B0 + ".pairAttention1.dimension") : 0, (int)M.meta(B0 + ".singleAttention.heads"),
                         (int)M.meta(B0 + ".singleAttention.dimension"),
                         (int)(M.at(B0 + ".pairTransition.transition1").length / (2 * (size_t)ch.C * ch.C)),
                         (int)(M.at(B0 + ".singleTransition.transition1").length / (2 * (size_t)ch.Cs * ch.Cs)),
                         M.has(B0 + ".pairAttention1.gatingQueryBias"), hasGrid, M.flag("trunk.dialect.parallelPairformer"), has, load);
  }
  // the confidence head's parameters: left out of an example's clip and count when its confidence losses weigh zero
  // (runner.py _get_sample_disabled_param_names); they were the last added
  const size_t confFrom = [&] { size_t k = 0; for (auto* p : ps.all) { if (!p->name.compare(0, 11, "confidence.")) break; ++k; } return k; }();
  size_t nparams = 0; for (auto* p : ps.all) nparams += p->n;
  printf("%s (%s): %d MSA blocks, %d pairformer blocks; %zu tensors, %.1f M parameters (%zu in the confidence head)\n",
         family.c_str(), bundle.c_str(), nMsa, nPf, ps.all.size(), nparams / 1e6, ps.all.size() - confFrom);
  const int Cz = te.C, Cs = te.Cs, Cm = te.Cm;

  // (lines naming the same input and truth - an entry's chains and interfaces - share one load, their own fields apart)
  for (size_t k = 1; k < lines.size(); ++k) {
    std::string in, tr; { std::istringstream ls(lines[k]); ls >> in >> tr; }
    size_t same = examples.size();
    for (size_t j = 0; j < examples.size(); ++j) if (examples[j].input == in && examples[j].truthPath == tr) { same = j; break; }
    if (same == examples.size()) { examples.push_back(loadExample(lines[k], padded, qblock)); continue; }
    Example e = examples[same];
    const std::string oldPerms = e.permsPath;
    applyFields(e, lines[k]);
    if (e.permsPath != oldPerms) { fprintf(stderr, "%s: one input, two perms files\n", in.c_str()); return 1; }
    examples.push_back(std::move(e));
  }
  printf("every example's atom windows rebuilt as the featuriser built them\n");
  loadExample(lines[0], padded, qblock);                                   // (M is the first example's again: the port reads it)

  // ---- the activations, sized per step
  struct Acts {
    int n = 0, S = 0;
    g::real *tf, *zEmb, *prevPair, *prevSingle, *logits, *zIn;
    std::vector<g::real*> pairs, msas, singles;
    std::vector<void*> owned;
    g::real* z(size_t k) { g::real* p = dzeros(k); owned.push_back(p); return p; }
    void release() { for (void* p : owned) GK(cudaFree(p)); owned.clear(); }
  };
  auto makeActs = [&](int n, int S) {
    Acts a; a.n = n; a.S = S;
    const size_t nn = (size_t)n * n;
    a.tf = a.z((size_t)n * te.F); a.zEmb = a.z(nn * Cz); a.prevPair = a.z(nn * Cz); a.prevSingle = a.z((size_t)n * Cs);
    a.logits = a.z(nn * dh.bins); a.zIn = doubleAdd ? a.z(nn * Cz) : nullptr;
    for (int k = 0; k <= nMsa + nPf; ++k) a.pairs.push_back(a.z(nn * Cz));
    for (int k = 0; k <= nMsa; ++k) a.msas.push_back(a.z((size_t)S * n * Cm));
    for (int k = 0; k <= nPf; ++k) a.singles.push_back(a.z((size_t)n * Cs));
    return a;
  };
  // the trunk forward over d's current MSA rows, its saves in the arena
  g::af3::TargetFeatSave tfS; g::af3::EmbedSave emS; g::af3::TemplateSave tmS;
  std::vector<g::af3::MsaSave> msaS(nMsa);
  std::vector<g::af3::PairformerSave> pfS(nPf);
  auto embedBatch = [&](Dev& d) {
    return g::af3::EmbedBatch{d.ri, d.ti, d.asym, d.ent, d.sym, d.bonds, d.orders, d.msaRows, d.deletion, d.n, d.S};
  };
  auto trunkForward = [&](Dev& d, Acts& a) {
    const int n = d.n, S = d.S; const size_t nn = (size_t)n * n;
    auto eb = embedBatch(d);
    tfS = g::af3::targetFeatForward(tfm, d.b, d.aatype, d.profile, d.delMean, a.tf);
    emS = g::af3::trunkEmbedForward(te, eb, a.tf, a.prevPair, a.prevSingle, a.zEmb, a.singles[0], a.msas[0]);
    GK(cudaMemcpyAsync(a.pairs[0], a.zEmb, nn * Cz * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
    tmS = g::af3::templateForward(tm, d.slots, d.templates, a.zEmb, d.pairMask, n, a.pairs[0]);
    if (doubleAdd) GK(cudaMemcpyAsync(a.zIn, a.pairs[0], nn * Cz * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
    // each block's saves discarded as it finishes (OpenFold3's blocks_per_ckpt 1): the backward recomputes them from
    // the block's input, which pairs/msas/singles keep
    // (--no-ckpt keeps them, for --check-ckpt to hold the recomputation to)
    for (int k = 0; k < nMsa; ++k) {
      const g::Arena::Mark mk = g::ARENA.mark();
      msaS[k] = g::af3::msaForward(msaB[k], dl, a.msas[k], a.pairs[k], d.msaMask, d.pairMask, d.pairMaskT, d.seqMask, a.msas[k + 1], a.pairs[k + 1], S, n);
      if (ckpt) g::ARENA.rewind(mk);
    }
    if (doubleAdd) g::add(a.pairs[nMsa], a.zIn, nn * Cz);
    // (--bf16: the pairformer's bf16 kernels, OpenFold3's bf16-mixed for the bulk of the trunk; its pair and single
    // between blocks stay float32)
    const bool prevBf = g::BF16; g::BF16 = bf16Trunk;
    for (int k = 0; k < nPf; ++k) {
      const g::Arena::Mark mk = g::ARENA.mark();
      pfS[k] = g::af3::pairformerForward(pf[k], dl, a.pairs[nMsa + k], a.singles[k], d.pairMask, d.pairMaskT, d.seqMask, nullptr,
                                         a.pairs[nMsa + k + 1], a.singles[k + 1], n);
      if (ckpt) g::ARENA.rewind(mk);
    }
    g::BF16 = prevBf;
    g::af3::distogramForward(dh, a.pairs[nMsa + nPf], n, a.logits);
  };
  // `c` cycles (model.py run_trunk): the first c - 1 without a gradient, each handing its pair and single on; `draw`
  // fills the cycle's MSA rows first
  auto cycles = [&](Dev& d, Acts& a, int c, const std::function<void()>& draw) {
    const size_t nn = (size_t)d.n * d.n;
    GK(cudaMemsetAsync(a.prevPair, 0, nn * Cz * sizeof(g::real), STREAM));
    GK(cudaMemsetAsync(a.prevSingle, 0, (size_t)d.n * Cs * sizeof(g::real), STREAM));
    for (int pass = 0; pass < c; ++pass) {
      draw();
      g::ARENA.reset();
      trunkForward(d, a);
      if (pass + 1 == c) break;
      GK(cudaMemcpyAsync(a.prevPair, a.pairs[nMsa + nPf], nn * Cz * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
      GK(cudaMemcpyAsync(a.prevSingle, a.singles[nPf], (size_t)d.n * Cs * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
    }
  };

  // ---- the composition, against the port: the first example whole, the port's own (prefix) MSA rows - when it is no
  // larger than a crop (both trunks side by side, uncropped, are what this check costs)
  if (examples[0].n > crop) printf("the composition check skipped: the first example (%d tokens) is larger than a crop\n", examples[0].n);
  else {
    const bool bfWas = bf16Trunk; bf16Trunk = false;     // (the composition is float32's, against the float32 port)
    MSA_DRAW = "prefix";
    const float* refTf = buildTargetFeat();
    Trunk t = makeTrunk(nullptr, 64, false, refTf);
    te.pairedQuery = M.flag("trunk.dialect.msaPairedQueryRow") || (M.flag("trunk.dialect.msaPairedQueryRowAligned") && t.S > 1);
    std::map<std::string, std::vector<float>> seams;
    runTrunk<float>(t, [&](const char* nm, const float* p, size_t len) {
      if (strcmp(nm, "z_init_generic") && strcmp(nm, "z_after_template") && strcmp(nm, "z_after_msa")) return;
      std::vector<float> h(len); CK(cudaMemcpyAsync(h.data(), p, len * 4, cudaMemcpyDeviceToHost, STREAM));
      CK(cudaStreamSynchronize(STREAM)); seams[nm] = std::move(h);
    });
    CK(cudaStreamSynchronize(STREAM));
    const Example& e = examples[0];
    const int n = e.n; const size_t nn = (size_t)n * n;
    Dev d; upload(d, e, padded, qblock, t.S);
    {
      // the port's rows in place of a draw
      const size_t rows = (size_t)t.S * n;
      std::vector<float> h(rows);
      GK(cudaMemcpy(d.msaRows, t.msaRows, rows * 4, cudaMemcpyDeviceToDevice));
      GK(cudaMemcpy(h.data(), t.deletion, rows * 4, cudaMemcpyDeviceToHost));
      std::vector<g::real> r(h.begin(), h.end()); GK(cudaMemcpy(d.deletion, r.data(), rows * sizeof(g::real), cudaMemcpyHostToDevice));
      GK(cudaMemcpy(h.data(), t.msaMask, rows * 4, cudaMemcpyDeviceToHost));
      r.assign(h.begin(), h.end()); GK(cudaMemcpy(d.msaMask, r.data(), rows * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    Acts a = makeActs(n, t.S);
    auto hostF = [&](const float* p, size_t len) { std::vector<float> h(len); GK(cudaMemcpy(h.data(), p, len * 4, cudaMemcpyDeviceToHost)); return h; };
    bool ok = true;
    auto cmp = [&](const char* what, const g::real* mine, const float* port, size_t len) {
      const double e_ = g::relRms(g::host(mine, len), hostF(port, len));
      printf("  trunk %-26s against cuda/af3's runTrunk: %.3e%s\n", what, e_, e_ < 1e-3 ? "" : "  <-- ABOVE 1e-3");
      ok &= e_ < 1e-3;
    };
    cycles(d, a, 1, [] {});
    CK(cudaStreamSynchronize(STREAM));
    for (auto& [nm, h] : seams) {
      const g::real* mine = nm == "z_after_msa" ? a.pairs[nMsa] : nm == "z_init_generic" ? a.zEmb : a.pairs[0];
      printf("  seam %-18s against the port's: %.3e\n", nm.c_str(), g::relRms(g::host(mine, nn * Cz), h));
    }
    cmp("pair (pass 0)", a.pairs[nMsa + nPf], t.pair, nn * Cz); cmp("single (pass 0)", a.singles[nPf], t.single, (size_t)n * Cs);
    for (int pass = 1; pass < 3; ++pass) {
      if (!t.inPlaceRecycle) CK(cudaMemcpyAsync(t.prevPair, t.pair, nn * t.C * (t.p16 ? 2 : 4), cudaMemcpyDeviceToDevice, STREAM));
      CK(cudaMemcpyAsync(t.prevSingle, t.single, (size_t)t.n * t.Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
      runTrunk<float>(t, [](const char*, const float*, size_t) {});
    }
    cycles(d, a, 3, [] {});
    CK(cudaStreamSynchronize(STREAM));
    cmp("pair (three passes)", a.pairs[nMsa + nPf], t.pair, nn * Cz); cmp("single (three passes)", a.singles[nPf], t.single, (size_t)n * Cs);
    if (!ok) { printf("FAIL: the composed trunk is not the port's\n"); return 1; }
    a.release(); d.release();
    bf16Trunk = bfWas;
  }

  // ---- the sampler (the mini-rollout) and the confidence and diffusion steps
  double gamma0 = 0.8, gammaMin = 1.0, noiseScale = 1.003, stepScale = 1.5, sMin = 4e-4, sMax = 160, rho = 7;
  {
    const std::string S = "trunk.dialect.sampler.";
    if (M.has(S + "gamma0")) {
      gamma0 = M.meta(S + "gamma0"); gammaMin = M.meta(S + "gammaMin"); noiseScale = M.meta(S + "noiseScale");
      stepScale = M.meta(S + "stepScale"); rho = M.meta(S + "rho"); sMin = M.meta(S + "sigmaMin"); sMax = M.meta(S + "sigmaMax");
    }
  }
  auto rotation = [](std::normal_distribution<double>& nd, std::mt19937_64& gen, double R[3][3]) {
    double q[4], qn = 0; for (double& v : q) { v = nd(gen); qn += v * v; } qn = sqrt(qn); for (double& v : q) v /= qn;
    const double Rm[3][3] = {{1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])},
                             {2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[0] * q[1])},
                             {2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])}};
    for (int p = 0; p < 3; ++p) for (int c = 0; c < 3; ++c) R[p][c] = Rm[p][c];
  };
  // AF3's Algorithm 18 (OpenFold3's sample_diffusion), one sample, no gradient: the rolled-out positions
  auto rollout = [&](Dev& d, Acts& a, const Example& e, std::mt19937_64& gen, int nsteps) {
    std::normal_distribution<double> nd(0, 1);
    const size_t A = e.atoms();
    auto level = [&](int k) { return noiseSchedule((double)k / nsteps, 16, sMin, sMax, rho); };
    std::vector<double> xs(A * 3, 0);
    for (size_t i = 0; i < A; ++i) if (e.predMask[i] > 0) for (int p = 0; p < 3; ++p) xs[i * 3 + p] = level(0) * nd(gen);
    g::real *xd = g::ARENA.get(A * 3), *od = g::ARENA.get(A * 3);
    size_t live = 0; for (size_t i = 0; i < A; ++i) live += e.predMask[i] > 0;
    for (int k = 1; k <= nsteps; ++k) {
      const double prev = level(k - 1), lv = level(k), tHat = prev * (1 + (lv > gammaMin ? gamma0 : 0));
      double R[3][3]; rotation(nd, gen, R);
      double cen[3] = {0, 0, 0}; for (size_t i = 0; i < A; ++i) if (e.predMask[i] > 0) for (int c = 0; c < 3; ++c) cen[c] += xs[i * 3 + c] / live;
      const double tr[3] = {nd(gen), nd(gen), nd(gen)};
      const double inj = noiseScale * sqrt(std::max(0.0, tHat * tHat - prev * prev));
      std::vector<double> noisy(A * 3, 0);
      for (size_t i = 0; i < A; ++i) {
        if (!(e.predMask[i] > 0)) continue;
        for (int p = 0; p < 3; ++p) {
          double v = tr[p]; for (int c = 0; c < 3; ++c) v += R[p][c] * (xs[i * 3 + c] - cen[c]);
          noisy[i * 3 + p] = v + inj * nd(gen);
        }
      }
      { std::vector<g::real> rr(noisy.begin(), noisy.end()); GK(cudaMemcpy(xd, rr.data(), rr.size() * sizeof(g::real), cudaMemcpyHostToDevice)); }
      {
        g::Scratch sc;
        g::af3::denoiseForward(dn, d.b, a.singles[nPf], a.pairs[nMsa + nPf], a.tf, d.seqMask, xd, (g::real)tHat, d.ri, d.ti, d.asym, d.ent, d.sym, od);
      }
      auto den = g::host(od, A * 3);
      const double coef = stepScale * (lv - tHat) / tHat;
      for (size_t i = 0; i < A * 3; ++i) xs[i] = noisy[i] + coef * (noisy[i] - den[i]);
    }
    return xs;
  };

  struct Losses { double mse = 0, bond = 0, lddt = 0, diffusion = 0, distogram = 0, plddt = 0, pde = 0, pae = 0, resolved = 0, total = 0; };
  of3::DiffusionWeights DW; DW.bond = bondWeight;
  const double distWeight = 3e-2, confWeight = 1e-4;
  // --init=of3: OpenFold3's initialisation in place of the bundle's weights (linear_init_config.py and Linear's
  // schemes): default LeCun and relu He fan-in truncated normals, gating and final zeros (a gating bias 1, an
  // adaptive-zero gate's bias -2), LayerNorms 1 and 0, the Fourier embedding N(0, 1) and U(0, 1). Each weight's kind
  // from its name, as OpenFold3's config assigns it to the module it is.
  if (init == "of3") {
    std::mt19937_64 ig(seed * 7919 + 17);
    auto ends = [](const std::string& s, const std::string& t) { return s.size() >= t.size() && !s.compare(s.size() - t.size(), t.size(), t); };
    auto has = [](const std::string& s, const std::string& t) { return s.find(t) != std::string::npos; };
    // the direct weights' input widths (registered without a Linear): what they embed
    auto directFanIn = [&](const std::string& nm, size_t n) -> int {
      for (auto& [suffix, in] : std::vector<std::pair<std::string, int>>{
             {".embedRefPos", 3}, {".embedRefMask", 1}, {".embedRefElement", 128}, {".embedRefCharge", 1}, {".embedRefAtomName", 256},
             {".embedPairOffsets", 3}, {".embedPairDistances", 1}, {".embedPairOffsetsValid", 1}, {"bondEmbedding", 1},
             {"templatePairEmbedding1", 1}, {"templatePairEmbedding4", 1}, {"templatePairEmbedding5", 1}, {"templatePairEmbedding6", 1},
             {"templatePairEmbedding7", 1}, {"msaActivations", te.msaWidth}}) if (ends(nm, suffix)) return in;
      if (ends(nm, "templatePairEmbedding0")) return (int)(n / M.meta("trunk.template.channels"));
      if (ends(nm, "distogramFeatProject")) return (int)(n / ch.C);
      return 0;
    };
    std::normal_distribution<double> tn(0, 1); std::uniform_real_distribution<double> uni(0, 1);
    std::map<std::string, int> kinds;
    for (size_t i = 0; i < ps.all.size(); ++i) {
      g::Param* p = ps.all[i];
      const std::string& nm = p->name;
      const std::string leaf = nm.substr(nm.rfind('.') + 1);
      std::string kind;
      if (has(nm, "fourierWeight")) kind = "fourier-normal";
      else if (has(nm, "fourierBias")) kind = "fourier-uniform";
      else if (ends(leaf, "Scale")) kind = "one";
      else if (ends(leaf, "Offset") || leaf == "qBias" || leaf == "outputB" || ends(leaf, "SingleCondScaleBias") || leaf == "SingleCondScaleBias") kind = "zero";
      else if (ends(leaf, "AdaptiveZeroCondBias")) kind = "minus-two";                      // gating_ada_zero's bias
      else if (ends(leaf, "AdaptiveZeroCondWeights")) kind = "zero";                        // ...its weights
      else if (ends(leaf, "SingleCondBias") || ends(leaf, "SingleCondScaleWeights")) kind = "zero";   // ada_ln: linear_s, linear_g final
      else if (leaf == "gatingQuery" || leaf == "gate" || leaf == "gatingLinear") kind = "zero";      // gating
      else if (leaf == "transition2" || leaf == "outputW" || leaf == "pairMlp3" || leaf == "atomFeaturesToPositionUpdate" ||
               leaf == "embedTrunkSingleCond" || leaf == "embedTrunkPairCond" || leaf == "singleCondEmbeddingProjection" ||
               leaf == "halfLogits" || leaf == "paeLogits" || leaf == "leftHalfDistanceLogits" || leaf == "plddtLogits" ||
               leaf == "experimentallyResolvedLogits" || leaf == "prevEmbedding" || leaf == "prevSingleEmbedding" ||
               (leaf == "outputProjection") || (leaf == "ffwTransition2" && has(nm, "diffusion.conditioning")))
        kind = "zero";                                                                         // final
      else if (leaf == "transition1" || leaf == "ffwTransition1" || leaf == "pairMlp1" || leaf == "pairMlp2" ||
               leaf == "templatePairEmbedding8")
        kind = "relu";
      else kind = "default";                                                                   // LeCun (Transition2 / ffwTransition2 of an ada block included)
      ++kinds[kind];
      std::vector<double> h(p->n);
      if (kind == "one") std::fill(h.begin(), h.end(), 1.0);
      else if (kind == "zero") std::fill(h.begin(), h.end(), 0.0);
      else if (kind == "minus-two") std::fill(h.begin(), h.end(), -2.0);
      else if (kind == "fourier-normal") for (double& v : h) v = tn(ig);
      else if (kind == "fourier-uniform") for (double& v : h) v = uni(ig);
      else {
        const int fan = p->fanIn > 0 ? p->fanIn : directFanIn(nm, p->n);
        if (fan <= 0) { fprintf(stderr, "--init=of3: no input width for %s\n", nm.c_str()); return 1; }
        // trunc_normal_init_: N(0, std) truncated to [-2, 2] standard deviations, std = sqrt(scale / fan) / 0.87962566
        const double sd = std::sqrt((kind == "relu" ? 2.0 : 1.0) / std::max(1, fan)) / 0.87962566103423978;
        for (double& v : h) { double z; do z = tn(ig); while (z < -2 || z > 2); v = z * sd; }
      }
      std::vector<g::real> r(h.begin(), h.end());
      GK(cudaMemcpy(p->w, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    printf("OpenFold3's initialisation:"); for (auto& [k, c] : kinds) printf(" %s %d", k.c_str(), c); printf("\n");
  }
  g::Of3Optimizer opt; opt.maxLr = lrMax; opt.warmup = warmup; opt.init(ps);
  // the Fourier noise embedding is a buffer in OpenFold3: held, never trained
  for (size_t i = 0; i < ps.all.size(); ++i) if (ps.all[i]->name.find("fourier") != std::string::npos) opt.frozen[i] = 1;
  if (!resumePath.empty()) { opt.load(ps, resumePath); printf("resumed %s at step %ld\n", resumePath.c_str(), opt.step); }
  std::mt19937_64 rng(seed + 7919 * (uint64_t)opt.step + 104729ull * (sameDraws ? 0 : rank));   // (each rank its own examples and draws)
  std::normal_distribution<double> N01(0, 1);
  const double sd = 16;

  // one example through the whole network: its losses, and (train) its gradients in the parameters. `fixed` draws
  // everything from a seeded generator and the noise levels from a fixed ladder - a measurement, comparable run to run.
  // training's dropout (OpenFold3: pair 0.25 in every pairformer-style stack - trunk, MSA module, template, confidence -
  // MSA 0.15), fresh masks a step; off (eval) for a measurement and validation
  auto setDropout = [&](bool on, uint64_t base) {
    for (size_t k = 0; k < pf.size(); ++k) { pf[k].dropRate = on ? 0.25f : 0; pf[k].dropSeed = base * 1000 + k; }
    for (size_t k = 0; k < msaB.size(); ++k) {
      msaB[k].dropRate = on ? 0.15f : 0; msaB[k].dropSeed = base * 1000 + 100 + k;
      msaB[k].pairHalf.dropRate = on ? 0.25f : 0; msaB[k].pairHalf.dropSeed = base * 1000 + 200 + k;
    }
    for (size_t k = 0; k < tm.blocks.size(); ++k) { tm.blocks[k].dropRate = on ? 0.25f : 0; tm.blocks[k].dropSeed = base * 1000 + 300 + k; }
    for (size_t k = 0; k < ch.blocks.size(); ++k) { ch.blocks[k].dropRate = on ? 0.25f : 0; ch.blocks[k].dropSeed = base * 1000 + 400 + k; }
  };
  long stepIndex = 0;   // (an example's index in the run, all ranks alike: the synced recycle draw's key)
  auto runExample = [&](const Example& full, bool train, bool fixed, int nSamples, std::mt19937_64& gen) {
    Losses L;
    setDropout(train && (!fixed || dropInFixed) && dropout, gen());
    std::string strategy = "whole";
    auto now = [] { CK(cudaStreamSynchronize(STREAM)); return std::chrono::steady_clock::now(); };
    auto msOf = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    double tLoss = 0;
    const auto t0 = now();
    const of3::Layout Lfull = layoutOf(full);
    // the dataset's settings (dataset_configs.py): the distillation sets crop 0.25 contiguous / 0.75 spatial with no
    // chain crop and no confidence losses; the disordered set's diffusion terms over protein alone
    const bool distil = full.set == "monomer" || full.set == "rna";
    CropWeights cw; if (distil) cw = CropWeights{0.25, 0.75, 0.0};
    // the chain crop first (the truth becomes the chain-cropped entry, as there), then the token crop
    std::vector<int> keepC(full.n); for (int t = 0; t < full.n; ++t) keepC[t] = t;
    if (!fixed && !distil) keepC = chainCrop(full, Lfull, gen);
    const Example base = keepC.size() == (size_t)full.n ? full : subset(full, keepC, false);
    const of3::PermInput gtPerm = keepC.size() == (size_t)full.n ? full.perm : subsetPerm(full.perm, keepC);
    std::vector<int> keep(base.n); for (int t = 0; t < base.n; ++t) keep[t] = t;
    if (!fixed) keep = cropTokens(base, keepC.size() == (size_t)full.n ? Lfull : layoutOf(base), crop, cw, gen, strategy);
    if (keepC.size() != (size_t)full.n) strategy = "chains " + std::to_string(base.n) + "/" + std::to_string(full.n) + ", " + strategy;
    // (the crop from the whole entry, its tokens mapped through the chain crop)
    std::vector<int> keepFull(keep.size()); for (size_t k = 0; k < keep.size(); ++k) keepFull[k] = keepC[keep[k]];
    Example e = subset(full, keepFull);
    if (!fixed) { sampleTemplates(e, gen); augmentConformers(e, gen); }
    const of3::Layout Lay = layoutOf(e);
    const int n = e.n; const size_t nn = (size_t)n * n, A = e.atoms();
    const int S = std::min(1024, e.P);
    Dev d; upload(d, e, padded, qblock, S);
    Acts a = makeActs(n, S);
    const bool confOn = full.resolution >= 0.1 && full.resolution <= 4.0 && (full.set == "pdb");
    // the trunk
    // (model.py: the recycle count from a generator every rank shares - the same number on every rank a step)
    std::mt19937_64 synced(seed * 1000003ull + (uint64_t)stepIndex);
    const int c = fixed ? 4 : 1 + (int)(synced() % 4);
    cycles(d, a, c, [&] { drawMsa(d, e, gen); });
    const auto t1 = now();
    // the mini-rollout, and on it the truth's permutation (model.py: before every loss of the step)
    std::vector<double> xp;
    {
      g::Scratch sc;
      xp = rollout(d, a, e, gen, 20);
    }
    of3::permuteTruth(gtPerm, keep, xp, e.xt, e.gm, nullptr);
    // the distogram: its loss on the host, its cotangent kept for the backward
    g::real* dLog = nullptr;
    {
      auto lg = g::host(a.logits, nn * dh.bins);
      std::vector<double> gd(nn * dh.bins, 0);
      L.distogram = of3::distogramLoss(Lay, e.xt, e.gm, lg.data(), dh.bins, distWeight, gd.data());
      std::vector<g::real> r(gd.begin(), gd.end());
      dLog = d.zeros(nn * dh.bins);
      GK(cudaMemcpy(dLog, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    // the confidence head on the rollout, its inputs detached
    {
      g::Scratch sc;
      std::vector<double> beta(n * 3, 0);
      for (int t = 0; t < n; ++t) if (e.pbSlot[t] >= 0) for (int p = 0; p < 3; ++p) beta[t * 3 + p] = xp[((size_t)t * e.dense + e.pbSlot[t]) * 3 + p];
      g::real* betaD = g::ARENA.get(n * 3);
      { std::vector<g::real> rr(beta.begin(), beta.end()); GK(cudaMemcpy(betaD, rr.data(), rr.size() * sizeof(g::real), cudaMemcpyHostToDevice)); }
      g::af3::ConfidenceBatch cb{d.ri, d.ti, d.asym, d.ent, d.sym, d.bonds, d.orders};
      g::af3::ConfidenceLogits o{};
      auto cs = g::af3::confidenceForward(ch, a.pairs[nMsa + nPf], a.singles[nPf], a.tf, betaD, d.seqMask, d.pairMask, d.pairMaskT, n, o, cb);
      const double w = confOn ? confWeight : 0;
      auto lPl = g::host(o.plddt, A * ch.PB), lPde = g::host(o.pde, nn * ch.NB), lPae = g::host(o.pae, nn * ch.NB);
      std::vector<double> gPl(lPl.size(), 0), gPde(lPde.size(), 0), gPae(lPae.size(), 0), gRs;
      L.plddt = of3::plddtLoss(Lay, xp, e.xt, e.gm, lPl.data(), ch.PB, w, gPl.data());
      L.pde = of3::pdeLoss(Lay, xp, e.xt, e.gm, lPde.data(), ch.NB, w, gPde.data());
      L.pae = of3::paeLoss(Lay, xp, e.xt, e.gm, lPae.data(), ch.NB, w, gPae.data());
      if (ch.hasResolved) {
        auto lRs = g::host(o.resolved, A * 2); gRs.assign(lRs.size(), 0);
        L.resolved = of3::resolvedLoss(Lay, e.gm, lRs.data(), w, gRs.data());
      }
      if (train && confOn) {
        auto put = [&](const std::vector<double>& h) {
          g::real* p = g::ARENA.get(h.size()); std::vector<g::real> r(h.begin(), h.end());
          GK(cudaMemcpy(p, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice)); return p;
        };
        g::af3::confidenceBackward(ch, cs, a.pairs[nMsa + nPf], a.singles[nPf], a.tf, d.seqMask, d.pairMask, d.pairMaskT,
                                   g::af3::ConfidenceLogits{put(gPde), put(gPae), put(gPl), gRs.empty() ? nullptr : put(gRs)},
                                   nullptr, nullptr, nullptr, cb);
      }
    }
    const auto t2 = now();
    // the diffusion samples (model.py _train_diffusion), each its own denoiser pass
    g::real *dTs = d.zeros((size_t)n * Cs), *dTp = d.zeros(nn * Cz), *dTf = d.zeros((size_t)n * te.F);
    g::real *xD = d.zeros(A * 3), *xo = d.zeros(A * 3), *dxo = d.zeros(A * 3);
    size_t nRes = 0; for (size_t i = 0; i < A; ++i) nRes += e.gm[i] > 0;
    // (the disordered set: OpenFold3's loss_token_mask, protein tokens alone in the mse and smooth LDDT)
    std::vector<double> lossAtom;
    if (full.set == "disordered") {
      lossAtom.assign(A, 0);
      for (int t = 0; t < n; ++t) if (!e.isDna[t] && !e.isRna[t] && !e.isLigand[t]) for (int k = 0; k < e.dense; ++k) lossAtom[(size_t)t * e.dense + k] = 1;
    }
    std::vector<double> gmLoss(e.gm); if (!lossAtom.empty()) for (size_t i = 0; i < A; ++i) gmLoss[i] *= lossAtom[i];
    of3dev::SmoothLddt sl; sl.init(Lay, e.xt, gmLoss);   // (smooth LDDT, quadratic in atoms, on the device)
    for (int smp = 0; smp < nSamples; ++smp) {
      g::Scratch sc;
      const double sigma = fixed ? sd * exp(-1.2 + 1.5 * (-1.0 + 2.0 * smp / std::max(1, nSamples - 1))) : sd * exp(-1.2 + 1.5 * N01(gen));
      // centre_random_augmentation over the resolved atoms (the unresolved at the origin), then the noise
      double R[3][3]; std::normal_distribution<double> nd(0, 1); rotation(nd, gen, R);
      double cen[3] = {0, 0, 0}; for (size_t i = 0; i < A; ++i) if (e.gm[i] > 0) for (int q = 0; q < 3; ++q) cen[q] += e.xt[i * 3 + q] / std::max<size_t>(nRes, 1);
      const double tr[3] = {nd(gen), nd(gen), nd(gen)};
      std::vector<double> xn(A * 3, 0);
      for (size_t i = 0; i < A; ++i) {
        if (!(e.predMask[i] > 0)) continue;
        for (int p = 0; p < 3; ++p) {
          double v = 0;
          if (e.gm[i] > 0) { v = tr[p]; for (int q = 0; q < 3; ++q) v += R[p][q] * (e.xt[i * 3 + q] - cen[q]); }
          xn[i * 3 + p] = v + sigma * nd(gen);
        }
      }
      { std::vector<g::real> r(xn.begin(), xn.end()); GK(cudaMemcpy(xD, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice)); }
      auto ds = g::af3::denoiseForward(dn, d.b, a.singles[nPf], a.pairs[nMsa + nPf], a.tf, d.seqMask, xD, (g::real)sigma, d.ri, d.ti, d.asym, d.ent, d.sym, xo);
      auto xh = g::host(xo, A * 3);
      std::vector<double> xp(xh.begin(), xh.end()), dx(A * 3, 0);
      const auto tl = std::chrono::steady_clock::now();
      auto D = of3::diffusionLoss(Lay, xp, e.xt, e.gm, sigma, DW, &dx, 1.0 / nSamples,
                                  [&](double sc, std::vector<double>* g) { return sl.run(xp, sc, train ? g : nullptr, STREAM); },
                                  lossAtom.empty() ? nullptr : &lossAtom);
      tLoss += msOf(tl, std::chrono::steady_clock::now());
      L.mse += D.mse / nSamples; L.bond += D.bond / nSamples; L.lddt += D.lddt / nSamples; L.diffusion += D.total / nSamples;
      if (train) {
        std::vector<g::real> r(dx.begin(), dx.end());
        GK(cudaMemcpy(dxo, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
        g::real *ts = g::ARENA.zeros((size_t)n * Cs), *tp = g::ARENA.zeros(nn * Cz), *tfd = g::ARENA.zeros((size_t)n * te.F);
        g::af3::denoiseBackward(dn, d.b, ds, a.singles[nPf], d.seqMask, dxo, ts, tp, tfd, nullptr);
        g::add(dTs, ts, (size_t)n * Cs); g::add(dTp, tp, nn * Cz); g::add(dTf, tfd, (size_t)n * te.F);
      }
    }
    L.total = L.diffusion + distWeight * L.distogram + (confOn ? confWeight * (L.plddt + L.pde + L.pae + L.resolved) : 0);
    sl.release();
    const auto t3 = now();
    // the backward: the distogram's and the denoiser's gradients into the trunk, block by block
    if (train) {
      auto eb = embedBatch(d);
      g::real *dp = g::ARENA.zeros(nn * Cz), *dsg = g::ARENA.zeros((size_t)n * Cs);
      g::af3::distogramBackward(dh, a.pairs[nMsa + nPf], n, dLog, dp);
      g::add(dp, dTp, nn * Cz); g::add(dsg, dTs, (size_t)n * Cs);
      for (int k = nPf; k-- > 0;) {
        g::Scratch sc;
        g::real *op = g::ARENA.get(nn * Cz), *os = g::ARENA.get((size_t)n * Cs);   // (the recomputed block's outputs)
        const bool prevBf = g::BF16; g::BF16 = bf16Trunk;
        auto save = ckpt ? g::af3::pairformerForward(pf[k], dl, a.pairs[nMsa + k], a.singles[k], d.pairMask, d.pairMaskT, d.seqMask, nullptr, op, os, n)
                         : pfS[k];
        g::real *dpi = g::ARENA.zeros(nn * Cz), *dsi = g::ARENA.zeros((size_t)n * Cs);
        g::af3::pairformerBackward(pf[k], dl, save, d.pairMask, d.pairMaskT, d.seqMask, dp, dsg, dpi, dsi, nullptr);
        g::BF16 = prevBf;
        GK(cudaMemcpyAsync(dp, dpi, nn * Cz * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
        GK(cudaMemcpyAsync(dsg, dsi, (size_t)n * Cs * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
      }
      g::real* dzIn = doubleAdd ? g::cloneR(dp, nn * Cz) : nullptr;
      g::real* dm = g::ARENA.zeros((size_t)S * n * Cm);
      for (int k = nMsa; k-- > 0;) {
        g::Scratch sc;
        g::real *om = g::ARENA.get((size_t)S * n * Cm), *op = g::ARENA.get(nn * Cz);
        auto save = ckpt ? g::af3::msaForward(msaB[k], dl, a.msas[k], a.pairs[k], d.msaMask, d.pairMask, d.pairMaskT, d.seqMask, om, op, S, n)
                         : msaS[k];
        g::real *dpi = g::ARENA.zeros(nn * Cz), *dmi = g::ARENA.zeros((size_t)S * n * Cm);
        g::af3::msaBackward(msaB[k], dl, save, d.msaMask, d.pairMask, d.pairMaskT, d.seqMask, dm, dp, dmi, dpi);
        GK(cudaMemcpyAsync(dp, dpi, nn * Cz * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
        GK(cudaMemcpyAsync(dm, dmi, (size_t)S * n * Cm * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
      }
      if (dzIn) g::add(dp, dzIn, nn * Cz);
      g::real* dz = g::cloneR(dp, nn * Cz);
      g::af3::templateBackward(tm, d.slots, tmS, d.pairMask, dp, dz);
      g::real* dtf = g::ARENA.zeros((size_t)n * te.F);
      g::af3::trunkEmbedBackward(te, eb, emS, a.tf, dz, dsg, dm, dtf, g::ARENA.zeros(nn * Cz), g::ARENA.zeros((size_t)n * Cs));
      g::add(dtf, dTf, (size_t)n * te.F);
      g::af3::targetFeatBackward(tfm, d.b, tfS, dtf);
    }
    const auto t4 = now();
    int withTemplates = 0; for (auto& sl : e.slots) if (sl.geo) withTemplates += (int)std::lround(sl.repeat);
    if (!fixed) printf("  %s: %d of %d tokens (%s), %d cycles, %d MSA rows, %d templates; ms: trunk %.0f, rollout+confidence %.0f, diffusion %.0f (host losses %.0f), trunk backward %.0f\n",
                       full.input.c_str(), n, full.n, strategy.c_str(), c, S, withTemplates, msOf(t0, t1), msOf(t1, t2), msOf(t2, t3), tLoss, msOf(t3, t4));
    a.release(); d.release();
    return std::make_pair(L, confOn);
  };

  // --check-ckpt=k: example k (whole, fixed draws) backward twice - recomputing each block's saves, and keeping them -
  // and every parameter's gradient compared: the recomputation must change nothing
  if (checkCkpt >= 0) {
    dropInFixed = true;     // (the recomputation must regenerate the dropout masks too)
    auto grads = [&](bool on) {
      ckpt = on; ps.zeroGrad();
      std::mt19937_64 gen(99);
      runExample(examples.at(checkCkpt), true, true, 2, gen);
      std::vector<double> all;
      for (auto* p : ps.all) { auto h = g::host(p->g, p->n); all.insert(all.end(), h.begin(), h.end()); }
      return all;
    };
    // (the backward's atomics make two identical runs differ in the last places: the bar is that floor, measured)
    const auto a = grads(true), b = grads(false), c = grads(true);
    auto rel = [](const std::vector<double>& x, const std::vector<double>& y) {
      double num = 0, den = 0; for (size_t i = 0; i < x.size(); ++i) { num += (x[i] - y[i]) * (x[i] - y[i]); den += y[i] * y[i]; }
      return std::sqrt(num / std::max(den, 1e-300));
    };
    const double e = rel(a, b), floor = rel(a, c);
    const bool ok = e < std::max(4 * floor, 1e-6);
    printf("activation checkpointing: every gradient recomputed against kept saves, relRMS %.3e (two identical runs: %.3e)  %s\n",
           e, floor, ok ? "PASS" : "FAIL");
    ckpt = true; dropInFixed = false;
    if (!ok) return 1;
  }
  // --check-bf16=k: example k (whole, fixed draws, dropout on) with the trunk's pairformer in bf16 and in float32 -
  // every gradient compared: the bf16 kernels' error (bf16 keeps 8 bits), against the float runs' own floor
  if (checkBf16 >= 0) {
    dropInFixed = true;
    auto grads = [&](bool on) {
      bf16Trunk = on; ps.zeroGrad();
      std::mt19937_64 gen(99);
      runExample(examples.at(checkBf16), true, true, 2, gen);
      std::vector<double> all;
      for (auto* p : ps.all) { auto h = g::host(p->g, p->n); all.insert(all.end(), h.begin(), h.end()); }
      return all;
    };
    const auto a = grads(true), b = grads(false), c = grads(false);
    auto rel = [](const std::vector<double>& x, const std::vector<double>& y) {
      double num = 0, den = 0; for (size_t i = 0; i < x.size(); ++i) { num += (x[i] - y[i]) * (x[i] - y[i]); den += y[i] * y[i]; }
      return std::sqrt(num / std::max(den, 1e-300));
    };
    const double e = rel(a, b), floor = rel(c, b);
    const bool ok = e < 0.05;
    printf("bf16 trunk: every gradient against float32's, relRMS %.3e (two float runs: %.3e)  %s\n", e, floor, ok ? "PASS" : "FAIL");
    bf16Trunk = false; dropInFixed = false;
    if (!ok) return 1;
  }
  // --check-perm=k: example k whole, its truth disguised - each repeated molecule's copies relabelled cyclically, every
  // component with a symmetric naming given its first alternative, all rotated, moved and perturbed 0.1 A - must come
  // back: the permuted truth within the noise of the disguise, where the truth as labelled is far
  if (checkPerm >= 0) {
    const Example& E = examples.at(checkPerm);
    const of3::PermInput& P = E.perm;
    std::mt19937_64 gen(5); std::normal_distribution<double> nd(0, 1);
    std::map<int, std::vector<int>> copies; for (int t = 0; t < P.n; ++t) copies[P.ent[t]].push_back(P.sym[t]);
    std::map<std::pair<int, int>, int> shift;   // (entity, sym) -> the sym whose coordinates it shows
    for (auto& [en, ss] : copies) {
      std::vector<int> u(ss); std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end()), u.end());
      for (size_t q = 0; q < u.size(); ++q) shift[{en, u[q]}] = u[(q + 1) % u.size()];
    }
    std::map<std::tuple<int, int, int>, int> tokOf; for (int t = 0; t < P.n; ++t) tokOf[{P.ent[t], P.sym[t], P.molTok[t]}] = t;
    std::vector<double> xp(E.atoms() * 3, 0);
    double R[3][3]; rotation(nd, gen, R);
    int relabelled = 0, renamed = 0;
    for (int t = 0; t < P.n; ++t) {
      const int src = tokOf.at({P.ent[t], shift.at({P.ent[t], P.sym[t]}), P.molTok[t]});
      relabelled += src != t;
      auto pit = P.perms.find({P.asym[src], P.ri[src]});
      std::map<std::string, std::string> rename;
      if (pit != P.perms.end() && pit->second.size() > 1) for (size_t i = 0; i < pit->second[0].size(); ++i) rename[pit->second[0][i]] = pit->second[1][i];
      for (int s2 = 0; s2 < P.dense; ++s2) {
        const size_t i = (size_t)t * P.dense + s2;
        if (P.name[i].empty()) continue;
        std::string nm = rename.count(P.name[i]) ? rename[P.name[i]] : P.name[i];
        renamed += nm != P.name[i];
        int from = -1; for (int u = 0; u < P.dense; ++u) if (P.name[(size_t)src * P.dense + u] == nm) from = src * P.dense + u;
        if (from < 0) for (int u = 0; u < P.dense; ++u) if (P.name[(size_t)src * P.dense + u] == P.name[i]) from = src * P.dense + u;
        for (int c = 0; c < 3; ++c) {
          double v = 3.0 * (c + 1); for (int q = 0; q < 3; ++q) v += R[c][q] * P.xt[from * 3 + q];
          xp[i * 3 + c] = v + 0.1 * nd(gen);
        }
      }
    }
    std::vector<int> all(P.n); for (int t = 0; t < P.n; ++t) all[t] = t;
    std::vector<double> xo, go; std::string rep;
    of3::permuteTruth(P, all, xp, xo, go, &rep);
    auto rmsdTo = [&](const std::vector<double>& x, const std::vector<double>& m) {
      std::vector<double> w(m); double Rr[3][3], tr[3]; of3::weightedAlign(xp, x, w, Rr, tr);
      double s2 = 0, c = 0;
      for (size_t i = 0; i < m.size(); ++i) if (m[i] > 0) { for (int q = 0; q < 3; ++q) { double v = tr[q] - x[i * 3 + q]; for (int u = 0; u < 3; ++u) v += Rr[q][u] * xp[i * 3 + u]; s2 += v * v; } c += 1; }
      return std::sqrt(s2 / c);
    };
    const double asLabelled = rmsdTo(E.xt, E.gm), permuted = rmsdTo(xo, go);
    const bool ok = permuted < 0.25 && (relabelled + renamed == 0 || asLabelled > 2 * permuted);
    printf("truth permutation: %d tokens relabelled, %d atoms renamed; RMSD to the truth as labelled %.3f A, permuted %.3f A (matched:%s)  %s\n",
           relabelled, renamed, asLabelled, permuted, rep.empty() ? " identity" : rep.c_str(), ok ? "PASS" : "FAIL");
    if (!ok) return 1;
  }
  // ---- validation (runner.py's validation_step): the EMA's weights, 1 + 3 cycles, a full rollout of --val-samples
  // (5) x --val-steps (200), each sample's truth permuted and scored by all-atom lDDT; the sample the confidence head
  // ranks first (its mean pLDDT) reported beside the mean and the best. An entry larger than --val-crop is cropped
  // spatially from a seeded draw.
  std::vector<Example> valExamples;
  if (!valPath.empty()) {
    std::ifstream f(valPath); std::string l;
    if (!f) { fprintf(stderr, "cannot read %s\n", valPath.c_str()); return 1; }
    while (std::getline(f, l)) if (!l.empty() && l[0] != '#') valExamples.push_back(loadExample(l, padded, qblock));
    loadExample(lines[0], padded, qblock);
  }
  auto validate = [&](long atStep) {
    opt.swapEma(ps);
    setDropout(false, 0);
    double sumMean = 0, sumTop = 0;
    for (const Example& full : valExamples) {
      std::mt19937_64 gen(777);
      std::string strategy = "whole";
      std::vector<int> keep(full.n); for (int t = 0; t < full.n; ++t) keep[t] = t;
      if (full.n > valCrop) keep = cropTokens(full, layoutOf(full), valCrop, CropWeights{0, 1, 0}, gen, strategy);
      Example e = subset(full, keep);
      augmentConformers(e, gen);
      const of3::Layout Lay = layoutOf(e);
      const int n = e.n; const size_t nn = (size_t)n * n, A = e.atoms();
      Dev d; upload(d, e, padded, qblock, std::min(1024, e.P));
      Acts a = makeActs(n, d.S);
      cycles(d, a, 4, [&] { drawMsa(d, e, gen); });
      std::vector<double> scores, plddts;
      for (int s = 0; s < valSamples; ++s) {
        g::Scratch sc;
        auto xp = rollout(d, a, e, gen, valSteps);
        std::vector<double> xt, gm;
        of3::permuteTruth(full.perm, keep, xp, xt, gm, nullptr);
        scores.push_back(of3::lddt(Lay, xp, xt, gm));
        // the head's mean pLDDT on this sample (its ranking)
        std::vector<double> beta(n * 3, 0);
        for (int t = 0; t < n; ++t) if (e.pbSlot[t] >= 0) for (int p = 0; p < 3; ++p) beta[t * 3 + p] = xp[((size_t)t * e.dense + e.pbSlot[t]) * 3 + p];
        g::real* betaD = g::ARENA.get(n * 3);
        { std::vector<g::real> rr(beta.begin(), beta.end()); GK(cudaMemcpy(betaD, rr.data(), rr.size() * sizeof(g::real), cudaMemcpyHostToDevice)); }
        g::af3::ConfidenceBatch cb{d.ri, d.ti, d.asym, d.ent, d.sym, d.bonds, d.orders};
        g::af3::ConfidenceLogits o{};
        g::af3::confidenceForward(ch, a.pairs[nMsa + nPf], a.singles[nPf], a.tf, betaD, d.seqMask, d.pairMask, d.pairMaskT, n, o, cb);
        auto lg = g::host(o.plddt, A * ch.PB);
        double pl = 0, cnt = 0;
        for (size_t i = 0; i < A; ++i) {
          if (!(e.predMask[i] > 0)) continue;
          double mx = -1e300; for (int k = 0; k < ch.PB; ++k) mx = std::max(mx, (double)lg[i * ch.PB + k]);
          double z = 0, ex = 0; for (int k = 0; k < ch.PB; ++k) { const double p = exp(lg[i * ch.PB + k] - mx); z += p; ex += p * (k + 0.5) / ch.PB; }
          pl += ex / z; cnt += 1;
        }
        plddts.push_back(100 * pl / std::max(1.0, cnt));
      }
      (void)nn;
      const size_t top = std::max_element(plddts.begin(), plddts.end()) - plddts.begin();
      double mean = 0; for (double v : scores) mean += v / scores.size();
      const double best = *std::max_element(scores.begin(), scores.end());
      printf("  val %s (%d tokens, %s): lDDT mean %.4f, best %.4f, top-ranked %.4f (pLDDT %.1f)\n", full.input.c_str(), n,
             strategy.c_str(), mean, best, scores[top], plddts[top]);
      sumMean += mean; sumTop += scores[top];
      a.release(); d.release();
    }
    opt.swapEma(ps);
    printf("validation at step %ld (EMA weights): lDDT mean %.4f, top-ranked %.4f over %zu entries\n", atStep,
           sumMean / std::max<size_t>(1, valExamples.size()), sumTop / std::max<size_t>(1, valExamples.size()), valExamples.size());
  };

  // the smoke test's measurement: the first example whole, every draw seeded, the noise levels on a ladder
  const double lddtFloor = [] { double f = 0; for (double c : {0.5, 1.0, 2.0, 4.0}) f += (1 - 1 / (1 + exp(-c))) / 4; return f; }();
  auto measure = [&](const char* tag) {
    std::mt19937_64 fixedGen(12345);
    auto [L, confOn] = runExample(examples[0], false, true, 4, fixedGen);
    const double above = L.total - DW.lddt * lddtFloor;
    printf("%s loss %.4f (diffusion %.4f: mse %.4f, smooth lddt %.4f, bond %.4f; distogram %.4f; pLDDT %.3f PDE %.3f PAE %.3f resolved %.3f)\n",
           tag, L.total, L.diffusion, L.mse, L.lddt, L.bond, L.distogram, L.plddt, L.pde, L.pae, L.resolved);
    return above;
  };
  const double before = smoke ? measure("before ") : 0;
  // ---- training
  std::vector<double> cum; double totalW = 0;
  for (auto& e : examples) { totalW += e.weight; cum.push_back(totalW); }
  int inBatch = 0;
  for (int step = 0; step < steps; ++step) {
    stepIndex = opt.step * batch + inBatch;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, STREAM);
    const double u = std::uniform_real_distribution<double>(0, totalW)(rng);
    const size_t pick = std::lower_bound(cum.begin(), cum.end(), u) - cum.begin();
    ps.zeroGrad();
    const auto run = runExample(examples[std::min(pick, examples.size() - 1)], true, false, samples, rng);
    const Losses& L = run.first; const bool confOn = run.second;
    const double norm = opt.clipAndAccumulate(ps, [&](size_t i) { return !confOn && i >= confFrom; });
    bool stepped = false;
    if (++inBatch == batch) { opt.apply(ps, comm.get()); inBatch = 0; stepped = true; }
    cudaEventRecord(e1, STREAM); CK(cudaStreamSynchronize(STREAM));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    printf("step %4d  loss %.4f (diffusion %.4f, distogram %.4f, confidence %.3f%s)  |grad| %.3e%s  %.0f ms\n", step, L.total, L.diffusion,
           L.distogram, L.plddt + L.pde + L.pae + L.resolved, confOn ? "" : " off", norm,
           stepped ? (std::string("  lr ") + std::to_string(opt.lastLr).substr(0, 8)).c_str() : "", ms);
    if (rank == 0 && !savePath.empty() && saveEvery > 0 && stepped && opt.step % saveEvery == 0) { opt.save(ps, savePath); printf("saved %s (step %ld)\n", savePath.c_str(), opt.step); }
    if (rank == 0 && !valExamples.empty() && valEvery > 0 && stepped && opt.step % valEvery == 0) validate(opt.step);
  }
  if (rank == 0 && !savePath.empty()) { opt.save(ps, savePath); printf("saved %s (step %ld)\n", savePath.c_str(), opt.step); }
  if (rank == 0 && !valExamples.empty() && (valEvery == 0 || opt.step % valEvery != 0)) validate(opt.step);
  if (world > 1) {
    // the ranks' weights must be one set: a checksum of every weight, every rank's, compared through the same exchange
    double sum = 0, abs = 0;
    for (auto* p : ps.all) { auto h = g::host(p->w, p->n); for (double v : h) { sum += v; abs += std::fabs(v); } }
    std::vector<double> mine{sum, abs};
    g::real* dv; GK(cudaMalloc(&dv, 2 * sizeof(g::real)));
    std::vector<g::real> hv(mine.begin(), mine.end()), all0(hv);
    GK(cudaMemcpy(dv, hv.data(), 2 * sizeof(g::real), cudaMemcpyHostToDevice));
    comm->sumReals(dv, 2);
    GK(cudaMemcpy(hv.data(), dv, 2 * sizeof(g::real), cudaMemcpyDeviceToHost));
    const bool same = std::fabs(hv[0] - world * all0[0]) <= 1e-6 * std::fabs(hv[0]) + 1e-6 && std::fabs(hv[1] - world * all0[1]) <= 1e-6 * hv[1];
    fprintf(stderr, "rank %d: weights checksum %.9e (|w| %.9e), the ranks' mean %.9e  %s\n", rank, all0[0], all0[1], hv[0] / world,
            same ? "in sync" : "OUT OF SYNC");
    if (!same) return 1;
  } else if (sameDraws) {
    double sum = 0, abs = 0;
    for (auto* p : ps.all) { auto h = g::host(p->w, p->n); for (double v : h) { sum += v; abs += std::fabs(v); } }
    fprintf(stderr, "one process: weights checksum %.9e (|w| %.9e)\n", sum, abs);
  }
  if (smoke) {
    const double after = measure("after  ");
    printf("above the floor: %.4f -> %.4f\n", before, after);
    const bool ok = after < 0.5 * before;
    printf("%s\n", ok ? "PASS" : "FAIL: the loss above its floor did not halve");
    return ok ? 0 : 1;
  }
  return 0;
}
