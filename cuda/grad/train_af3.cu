// The AF3 lineage's training loop, first form: one family's whole training network assembled from its bundle - target_feat,
// the trunk embedder, the MSA stack, the pairformer, the distogram head and the denoiser (af3embed.cuh, af3.cuh,
// af3denoise.cuh) - trained on one featurised target against its crystal, every parameter updated by AdamW (adam.cuh).
//
//   loss = 4 L_diffusion + 0.03 L_distogram        (AF3's weights; supplement section 4)
//   L_diffusion: per step `--samples` noise levels, sigma = 16 exp(-1.2 + 1.5 N(0, 1)); the crystal centred, randomly
//     rotated and translated (1 A), noised, denoised; the crystal rigidly aligned onto the prediction (no gradient) and
//     (sigma^2 + 16^2) / (16 sigma)^2 * mean over atoms of |x - x_aligned|^2 / 3
//   L_distogram: cross entropy of the pseudo-beta distance's bin, every pair of resolved tokens
//
// The template term runs whatever the input: a target with no template still has padded slots, and in this lineage they
// weigh (a padded slot's restype is zero, and a one-hot of zero is a one) - 0.54 of the pair here when it was left out.
//
// Recycling: a step runs 1..--cycles passes (uniform), the gradient through the last only. Smooth LDDT (Algorithm 27).
// Not yet: the bond loss (ligand-polymer bonds only: zero on a protein), the confidence head and its losses, mixed precision, more than one target.
//
// Before training it holds its own trunk to cuda/af3's runTrunk on the same input (pass 0): the composition this file
// writes, against the port's.
//
//   nvcc -O2 -std=c++17 -arch=sm_80 --default-stream per-thread -o cuda/grad/train-af3 cuda/grad/train_af3.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/train-af3 --bundle=model-openbind0-f32 --family=openbind0 --input=cuda/grad/data/af3-in-openbind0 \
//       --truth=tools/fixtures/6mrr-crystal.pdb:A [--steps=30] [--lr=1e-4] [--samples=4] [--cycles=4] [--seed=1]
#include <array>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include "../af3/src/trunk.cuh"
#include "../af3/src/diffusion.cuh"
#include "../featurise/af3_weights.h"
#include "adam.cuh"
#include "af3denoise.cuh"
#include "af3embed.cuh"
#include "af3tmpl.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, r.size() * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
  return d;
}
static g::real* devReal(const float* dev, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), dev, n * 4, cudaMemcpyDeviceToHost));
  return devReal(std::vector<double>(h.begin(), h.end()));
}
static g::af3::Gather gatherG(const ::Gather& q) { return g::af3::Gather{q.idx, devReal(q.mask, q.count), q.count}; }
static g::real* dzeros(size_t n) { g::real* p; GK(cudaMalloc(&p, n * sizeof(g::real))); GK(cudaMemset(p, 0, n * sizeof(g::real))); return p; }

// the crystal's atoms in the batch's dense layout: token t's slot k is the atom of chain `chain`, residue
// residueIndex[t], named by refAtomNameChars (each code + 32); mask 1 where the batch has the slot and the file the atom
static void readTruth(const std::string& path, char chain, int n, int dense, std::vector<double>& x, std::vector<double>& m) {
  std::map<std::pair<int, std::string>, std::array<double, 3>> at;
  std::ifstream f(path);
  if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 4, "ATOM") && line.compare(0, 6, "HETATM")) continue;
    if (line.size() < 54 || line[21] != chain) continue;
    if (line[16] != ' ' && line[16] != 'A') continue;     // (the first altloc)
    std::string name = line.substr(12, 4);
    name.erase(0, name.find_first_not_of(' ')); name.erase(name.find_last_not_of(' ') + 1);
    const int res = std::stoi(line.substr(22, 4));
    at[{res, name}] = {std::stod(line.substr(30, 8)), std::stod(line.substr(38, 8)), std::stod(line.substr(46, 8))};
  }
  const int* ri = M.i("batch.features.residueIndex"); const int* names = M.i("batch.refAtomNameChars");
  const float* pm = M.f("batch.predDenseAtomMask");
  x.assign((size_t)n * dense * 3, 0); m.assign((size_t)n * dense, 0);
  int found = 0, wanted = 0;
  for (int t = 0; t < n; ++t)
    for (int k = 0; k < dense; ++k) {
      const size_t i = (size_t)t * dense + k;
      if (!pm[i]) continue;
      ++wanted;
      std::string nm;
      for (int c = 0; c < 4; ++c) { int ch = names[i * 4 + c]; if (ch > 0) nm += (char)(ch + 32); }
      auto it = at.find({ri[t], nm});
      if (it == at.end()) continue;
      for (int c = 0; c < 3; ++c) x[i * 3 + c] = it->second[c];
      m[i] = 1; ++found;
    }
  printf("truth: %s chain %c, %d of the batch's %d atoms\n", path.c_str(), chain, found, wanted);
}

// the rotation that best maps a (weighted, centred) onto b: R minimising sum w |R a + t - b|^2 (Kabsch), and t
static void kabsch(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& w, double R[3][3], double t[3]) {
  double ca[3] = {0, 0, 0}, cb[3] = {0, 0, 0}, sw = 0;
  const size_t n = w.size();
  for (size_t i = 0; i < n; ++i) { sw += w[i]; for (int c = 0; c < 3; ++c) { ca[c] += w[i] * a[i * 3 + c]; cb[c] += w[i] * b[i * 3 + c]; } }
  for (int c = 0; c < 3; ++c) { ca[c] /= sw; cb[c] /= sw; }
  double H[3][3] = {{0}};
  for (size_t i = 0; i < n; ++i)
    for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q) H[p][q] += w[i] * (a[i * 3 + p] - ca[p]) * (b[i * 3 + q] - cb[q]);
  // the quaternion form (Horn): the largest eigenvector of the 4 x 4 symmetric matrix, by power iteration on a shift
  double N[4][4] = {
    {H[0][0] + H[1][1] + H[2][2], H[1][2] - H[2][1], H[2][0] - H[0][2], H[0][1] - H[1][0]},
    {H[1][2] - H[2][1], H[0][0] - H[1][1] - H[2][2], H[0][1] + H[1][0], H[2][0] + H[0][2]},
    {H[2][0] - H[0][2], H[0][1] + H[1][0], -H[0][0] + H[1][1] - H[2][2], H[1][2] + H[2][1]},
    {H[0][1] - H[1][0], H[2][0] + H[0][2], H[1][2] + H[2][1], -H[0][0] - H[1][1] + H[2][2]}};
  double shift = 0; for (int p = 0; p < 4; ++p) for (int q = 0; q < 4; ++q) shift += fabs(N[p][q]);
  for (int p = 0; p < 4; ++p) N[p][p] += shift;
  double qv[4] = {1, 0.01, 0.02, 0.03};
  for (int it = 0; it < 200; ++it) {
    double r[4] = {0, 0, 0, 0}, nr = 0;
    for (int p = 0; p < 4; ++p) for (int q = 0; q < 4; ++q) r[p] += N[p][q] * qv[q];
    for (int p = 0; p < 4; ++p) nr += r[p] * r[p];
    nr = sqrt(nr); for (int p = 0; p < 4; ++p) qv[p] = r[p] / nr;
  }
  const double w0 = qv[0], x = qv[1], y = qv[2], z = qv[3];
  double Rr[3][3] = {{w0 * w0 + x * x - y * y - z * z, 2 * (x * y - w0 * z), 2 * (x * z + w0 * y)},
                     {2 * (x * y + w0 * z), w0 * w0 - x * x + y * y - z * z, 2 * (y * z - w0 * x)},
                     {2 * (x * z - w0 * y), 2 * (y * z + w0 * x), w0 * w0 - x * x - y * y + z * z}};
  for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q) R[p][q] = Rr[p][q];
  for (int p = 0; p < 3; ++p) { t[p] = cb[p]; for (int q = 0; q < 3; ++q) t[p] -= R[p][q] * ca[q]; }
}

int main(int argc, char** argv) {
  std::string bundle = "model-openbind0-f32", family = "openbind0", input = "cuda/grad/data/af3-in-openbind0",
              truth = "tools/fixtures/6mrr-crystal.pdb:A";
  int steps = 30, samples = 4, cycles = 4; unsigned seed = 1; double lr = 1e-4;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--family=", 9)) family = argv[i] + 9;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
    else if (!strncmp(argv[i], "--truth=", 8)) truth = argv[i] + 8;
    else if (!strncmp(argv[i], "--steps=", 8)) steps = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--samples=", 10)) samples = atoi(argv[i] + 10);
    else if (!strncmp(argv[i], "--cycles=", 9)) cycles = std::max(1, atoi(argv[i] + 9));
    else if (!strncmp(argv[i], "--seed=", 7)) seed = (unsigned)atoi(argv[i] + 7);
    else if (!strncmp(argv[i], "--lr=", 5)) lr = atof(argv[i] + 5);
    else if (!strcmp(argv[i], "--tf32")) g::TF32 = true;
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.loadBundle(bundle, "", [&](const std::map<std::string, std::vector<long long>>& shapes) {
    lf::weights::Shapes S; S.shape = shapes; return lf::weights::af3WeightLines(family, S); });
  M.load(input);
  for (const char* f : {"trunk.dialect.chaiTokenEmbedding", "trunk.dialect.targetFeatAtomOnly", "trunk.dialect.structuralTokens",
                        "trunk.dialect.chaiAtomStack", "trunk.dialect.recycleFromInit"})
    if (M.flag(f)) { fprintf(stderr, "train-af3: %s is not composed here yet\n", f); return 1; }
  setAdaMode(false);
  AtomShape sh = atomShape();
  const int n = sh.tokens; const size_t nn = (size_t)n * n, atoms = (size_t)n * sh.dense;
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
  std::vector<g::af3::TemplateSlot> slots;
  int templates = 0;
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
    if (tm.fused) { fprintf(stderr, "train-af3: the fused template features are not composed here yet\n"); return 1; }
    const int passes = (int)M.meta("template.passes");
    templates = (int)M.meta("template.templates");
    slots.resize(passes);
    for (int k = 0; k < passes; ++k) {
      const std::string S_ = "template." + std::to_string(k) + ".";
      g::af3::TemplateSlot& sl = slots[k];
      sl.repeat = (g::real)M.meta(S_ + "repeat");
      sl.aatype = Idev(S_ + "aatype");
      if (M.has(S_ + "distogramBin")) {
        const size_t nn_ = (size_t)atomShape().tokens * atomShape().tokens;
        sl.bin = Idev(S_ + "distogramBin");
        sl.pb = devReal(Fdev(S_ + "pseudoBetaMask2d"), nn_); sl.uv = devReal(Fdev(S_ + "unitVector"), nn_ * 3);
        sl.bb = devReal(Fdev(S_ + "backboneMask2d"), nn_);
      }
    }
  }
  // ---- the MSA stack and the pairformer
  const g::af3::Dialect d{M.flag("trunk.dialect.swapTransposedBias"), M.flag("trunk.dialect.triangleMulDivideByLength")};
  const bool doubleAdd = M.flag("trunk.dialect.msaDoubleAddPair");
  int nMsa = 0; while (M.has("trunk.msaBlocks." + std::to_string(nMsa) + ".pairChannels")) ++nMsa;
  std::vector<g::af3::MsaBlock> msaB(nMsa);
  for (int k = 0; k < nMsa; ++k) {
    const std::string pre = "trunk.msaBlocks." + std::to_string(k);
    g::af3::MsaBlock& b = msaB[k];
    b.updateFirst = M.flag("trunk.dialect.msaUpdateBeforeOuterProduct"); b.chai = M.flag("trunk.dialect.groupedOuterProduct");
    if (b.chai) { fprintf(stderr, "train-af3: chai-1's grouped MSA is not composed here yet\n"); return 1; }
    g::af3::forEachParam(b, pre, (int)M.meta(pre + ".pairChannels"), (int)M.meta(pre + ".msaChannels"),
                         (int)M.meta(pre + ".pairAttention1.heads"), (int)M.meta(pre + ".pairAttention1.dimension"),
                         (int)M.meta(pre + ".msaAttention1.heads"), (int)M.meta(pre + ".msaAttention1.dimension"),
                         (int)M.meta(pre + ".outerProductMean.outerChannels"), 1,
                         M.has(pre + ".outerProductMean.leftProjectionBias"), M.has(pre + ".pairAttention1.gatingQueryBias"), load);
    b.opm.biasAfterNorm = M.flag("trunk.dialect.opmBiasAfterNorm");
    b.att.chaiMask = M.flag("trunk.dialect.chaiMsaFeatures");
  }
  int nPf = 0; while (M.has("trunk.pairformerBlocks." + std::to_string(nPf) + ".singleChannels")) ++nPf;
  if (M.has("trunk.pairMixerBlocks.0.singleChannels")) { fprintf(stderr, "train-af3: Mol-1's PairMixer is not composed here yet\n"); return 1; }
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
  size_t nparams = 0; for (auto* p : ps.all) nparams += p->n;
  printf("%s (%s): %d tokens, %zu atom slots; %d MSA blocks, %d pairformer blocks, distogram %d bins; %zu tensors, %.1f M parameters\n",
         family.c_str(), bundle.c_str(), n, atoms, nMsa, nPf, dh.bins, ps.all.size(), nparams / 1e6);
  // ---- the batch
  g::af3::AtomBatch b;
  b.tokens = n; b.dense = sh.dense; b.subsets = sh.subsets; b.Q = sh.queries; b.K = sh.keys;
  b.t2q = gatherG(gatherOf("batch.tokenAtomsToQueries")); b.q2k = gatherG(gatherOf("batch.queriesToKeys"));
  b.q2t = gatherG(gatherOf("batch.queriesToTokenAtoms")); b.tq = gatherG(gatherOf("batch.tokensToQueries"));
  b.tk = gatherG(gatherOf("batch.tokensToKeys"));
  b.refPos = devReal(Fdev("batch.refPos"), atoms * 3); b.refMask = devReal(Fdev("batch.refMask"), atoms);
  b.refCharge = devReal(Fdev("batch.refCharge"), atoms); b.atomMask = devReal(Fdev("batch.predDenseAtomMask"), atoms);
  {
    std::vector<int> u(M.len("batch.refSpaceUid")); GK(cudaMemcpy(u.data(), Idev("batch.refSpaceUid"), u.size() * 4, cudaMemcpyDeviceToHost));
    b.uid = devReal(std::vector<double>(u.begin(), u.end()));
  }
  b.refElement = Idev("batch.refElement"); b.nameChars = Idev("batch.refAtomNameChars");
  const int *ri = Idev("batch.features.residueIndex"), *ti = Idev("batch.features.tokenIndex"), *asym = Idev("batch.features.asymId"),
            *ent = Idev("batch.features.entityId"), *sym = Idev("batch.features.symId");
  // the port's trunk (pass 0): its MSA rows, and its pair and single - the composition's reference
  MSA_DRAW = "prefix";
  const float* refTf = buildTargetFeat();
  Trunk t = makeTrunk(nullptr, 64, false, refTf);
  te.pairedQuery = M.flag("trunk.dialect.msaPairedQueryRow") || (M.flag("trunk.dialect.msaPairedQueryRowAligned") && t.S > 1);
  std::map<std::string, std::vector<float>> seams;   // (the port's seams, to place a disagreement)
  runTrunk<float>(t, [&](const char* nm, const float* p, size_t len) {
    if (strcmp(nm, "z_init_generic") && strcmp(nm, "z_after_template") && strcmp(nm, "z_after_msa")) return;
    std::vector<float> h(len); CK(cudaMemcpyAsync(h.data(), p, len * 4, cudaMemcpyDeviceToHost, STREAM));
    CK(cudaStreamSynchronize(STREAM)); seams[nm] = std::move(h);
  });
  CK(cudaStreamSynchronize(STREAM));
  const int S = t.S;
  const size_t rows = (size_t)S * n;
  g::af3::EmbedBatch eb{ri, ti, asym, ent, sym,
                        M.has("batch.bondMatrix") ? devReal(Fdev("batch.bondMatrix"), nn) : nullptr,
                        M.has("batch.bondOrderMatrix") ? devReal(Fdev("batch.bondOrderMatrix"), nn) : nullptr,
                        t.msaRows, devReal(t.deletion, rows), n, S};
  const int* aatype = Idev("batch.aatype");
  g::real *profile = devReal(Fdev("batch.profile"), (size_t)n * 31), *delMean = devReal(Fdev("batch.deletionMean"), n);
  g::real *seqMask = devReal(t.seqMask, n), *pairMask = devReal(t.pairMask, nn), *msaMask = devReal(t.msaMask, rows);
  g::real* pairMaskT;
  {
    std::vector<double> pm(nn), pt(nn);
    auto h = g::host(pairMask, nn);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) pt[(size_t)j * n + i] = h[(size_t)i * n + j];
    pairMaskT = devReal(pt);
  }
  // the truth, and the distogram's targets from its pseudo-beta atoms
  std::vector<double> xt, xm;
  {
    const size_t colon = truth.rfind(':');
    readTruth(truth.substr(0, colon), colon == std::string::npos ? 'A' : truth[colon + 1], n, sh.dense, xt, xm);
  }
  const int Cz = te.C, Cs = te.Cs, Cm = te.Cm, bins = dh.bins;
  std::vector<int> binOf(nn, -1);
  {
    const ::Gather pbG = gatherOf("batch.tokenAtomsToPseudoBeta");
    std::vector<int> pbI(n); GK(cudaMemcpy(pbI.data(), pbG.idx, n * 4, cudaMemcpyDeviceToHost));
    std::vector<float> pbM(n); GK(cudaMemcpy(pbM.data(), pbG.mask, n * 4, cudaMemcpyDeviceToHost));
    std::vector<double> bounds(bins - 1);
    for (int k = 0; k < bins - 1; ++k) bounds[k] = 2.3125 + (21.6875 - 2.3125) * k / (bins - 2);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
      const size_t a = pbI[i], c = pbI[j];
      if (!pbM[i] || !pbM[j] || !xm[a] || !xm[c]) continue;
      double d2 = 0; for (int q = 0; q < 3; ++q) { double e = xt[a * 3 + q] - xt[c * 3 + q]; d2 += e * e; }
      const double dd = sqrt(d2);
      int bin = 0; while (bin < bins - 1 && dd > bounds[bin]) ++bin;
      binOf[(size_t)i * n + j] = bin;
    }
  }
  int* binDev; GK(cudaMalloc(&binDev, nn * 4)); GK(cudaMemcpy(binDev, binOf.data(), nn * 4, cudaMemcpyHostToDevice));
  // ---- activations: target_feat, each MSA block's and pairformer block's input and output
  g::real* tf = dzeros((size_t)n * te.F);
  std::vector<g::real*> pairs(nMsa + nPf + 1), msas(nMsa + 1), singles(nPf + 1);
  for (auto& p : pairs) p = dzeros(nn * Cz);
  for (auto& p : msas) p = dzeros(rows * Cm);
  for (auto& p : singles) p = dzeros((size_t)n * Cs);
  g::real* zEmb = dzeros(nn * Cz);     // z_init, before the template term
  g::real *prevPair = dzeros(nn * Cz), *prevSingle = dzeros((size_t)n * Cs), *logits = dzeros(nn * bins), *zIn = doubleAdd ? dzeros(nn * Cz) : nullptr;
  // the trunk forward, its saves in the arena from `mark` on
  std::vector<g::af3::MsaSave> msaS(nMsa);
  std::vector<g::af3::PairformerSave> pfS(nPf);
  g::af3::TargetFeatSave tfS; g::af3::EmbedSave emS; g::af3::TemplateSave tmS;
  auto trunkForward = [&]() {
    tfS = g::af3::targetFeatForward(tfm, b, aatype, profile, delMean, tf);
    emS = g::af3::trunkEmbedForward(te, eb, tf, prevPair, prevSingle, zEmb, singles[0], msas[0]);
    GK(cudaMemcpyAsync(pairs[0], zEmb, nn * Cz * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
    tmS = g::af3::templateForward(tm, slots, templates, zEmb, pairMask, n, pairs[0]);
    if (doubleAdd) GK(cudaMemcpyAsync(zIn, pairs[0], nn * Cz * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
    for (int k = 0; k < nMsa; ++k)
      msaS[k] = g::af3::msaForward(msaB[k], d, msas[k], pairs[k], msaMask, pairMask, pairMaskT, seqMask, msas[k + 1], pairs[k + 1], S, n);
    if (doubleAdd) g::add(pairs[nMsa], zIn, nn * Cz);
    for (int k = 0; k < nPf; ++k)
      pfS[k] = g::af3::pairformerForward(pf[k], d, pairs[nMsa + k], singles[k], pairMask, pairMaskT, seqMask, nullptr,
                                         pairs[nMsa + k + 1], singles[k + 1], n);
    g::af3::distogramForward(dh, pairs[nMsa + nPf], n, logits);
  };
  // `c` passes of the trunk: the first c - 1 without a gradient, each handing its pair and single to the next as the
  // recycled state (AF3's training: the gradient through the last pass only); the last pass's saves in the arena
  auto recycledTrunk = [&](int c) {
    GK(cudaMemsetAsync(prevPair, 0, nn * Cz * sizeof(g::real), STREAM));
    GK(cudaMemsetAsync(prevSingle, 0, (size_t)n * Cs * sizeof(g::real), STREAM));
    for (int pass = 0; pass + 1 < c; ++pass) {
      g::ARENA.reset();
      trunkForward();
      GK(cudaMemcpyAsync(prevPair, pairs[nMsa + nPf], nn * Cz * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
      GK(cudaMemcpyAsync(prevSingle, singles[nPf], (size_t)n * Cs * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
    }
    g::ARENA.reset();
    trunkForward();
  };
  recycledTrunk(1);
  CK(cudaStreamSynchronize(STREAM));
  {
    auto cmp = [&](const char* what, const g::real* mine, const float* port, size_t len) {
      std::vector<float> h(len); GK(cudaMemcpy(h.data(), port, len * 4, cudaMemcpyDeviceToHost));
      const double e = g::relRms(g::host(mine, len), h);
      printf("  trunk %-8s against cuda/af3's runTrunk (pass 0): %.3e%s\n", what, e, e < 1e-3 ? "" : "  <-- ABOVE 1e-3");
      return e < 1e-3;
    };
    for (auto& [nm, h] : seams) {
      const g::real* mine = nm == "z_after_msa" ? pairs[nMsa] : nm == "z_init_generic" ? zEmb : pairs[0];
      printf("  seam %-18s against the port's: %.3e\n", nm.c_str(), g::relRms(g::host(mine, nn * Cz), h));
    }
    const bool ok = cmp("pair", pairs[nMsa + nPf], t.pair, nn * Cz) & cmp("single", singles[nPf], t.single, (size_t)n * Cs);
    if (!ok) { printf("FAIL: the composed trunk is not the port's\n"); return 1; }
    // and recycled: three passes against the port's own recycle (af3.cu's recyclePass)
    const size_t pairsN = nn;
    for (int pass = 1; pass < 3; ++pass) {
      if (!t.inPlaceRecycle)
        CK(cudaMemcpyAsync(t.prevPair, t.pair, pairsN * t.C * (t.p16 ? 2 : 4), cudaMemcpyDeviceToDevice, STREAM));
      CK(cudaMemcpyAsync(t.prevSingle, t.single, (size_t)t.n * t.Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
      runTrunk<float>(t, [](const char*, const float*, size_t) {});
    }
    recycledTrunk(3);
    CK(cudaStreamSynchronize(STREAM));
    const double ep = g::relRms(g::host(pairs[nMsa + nPf], nn * Cz), [&] { std::vector<float> h(nn * Cz);
      GK(cudaMemcpy(h.data(), t.pair, h.size() * 4, cudaMemcpyDeviceToHost)); return h; }());
    const double es = g::relRms(g::host(singles[nPf], (size_t)n * Cs), [&] { std::vector<float> h((size_t)n * Cs);
      GK(cudaMemcpy(h.data(), t.single, h.size() * 4, cudaMemcpyDeviceToHost)); return h; }());
    printf("  trunk pair     against the port's, 3 passes: %.3e\n  trunk single   against the port's, 3 passes: %.3e\n", ep, es);
    if (!(ep < 1e-3 && es < 1e-3)) { printf("FAIL: the recycled trunk is not the port's\n"); return 1; }
  }
  // ---- the training loop
  g::AdamW opt; opt.lr = (g::real)lr; opt.clip = 10.f; opt.init(ps);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> N01(0, 1);
  const double sd = 16;
  g::real *x = dzeros(atoms * 3), *xo = dzeros(atoms * 3), *dxo = dzeros(atoms * 3);
  g::real *dTs = dzeros((size_t)n * Cs), *dTp = dzeros(nn * Cz), *dTf = dzeros((size_t)n * te.F);
  g::real *dLog = dzeros(nn * bins);
  double before = 0;
  std::vector<double> w(atoms);
  for (size_t i = 0; i < atoms; ++i) w[i] = xm[i];
  size_t nAt = 0; for (double v : w) nAt += v > 0;
  std::vector<size_t> live; for (size_t i = 0; i < atoms; ++i) if (w[i] > 0) live.push_back(i);
  // pass -1 and the last only measure, on a fixed sigma ladder, so the two are comparable
  for (int step = -1; step <= steps; ++step) {
    const bool train = step >= 0 && step < steps;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, STREAM);
    ps.zeroGrad();
    // a training step's cycle count uniform in [1, cycles]; the two measuring passes all of them
    const int c = train ? 1 + (int)(rng() % (unsigned)cycles) : cycles;
    recycledTrunk(c);
    const g::Arena::Mark trunkMark = g::ARENA.mark();
    // the distogram loss and its gradient
    double lDist = 0;
    {
      auto L = g::host(logits, nn * bins);
      std::vector<double> dl(nn * bins, 0);
      size_t count = 0; for (int v : binOf) count += v >= 0;
      for (size_t ij = 0; ij < nn; ++ij) {
        if (binOf[ij] < 0) continue;
        double mx = -1e30; for (int k = 0; k < bins; ++k) mx = std::max(mx, (double)L[ij * bins + k]);
        double z = 0; for (int k = 0; k < bins; ++k) z += exp(L[ij * bins + k] - mx);
        lDist += -(L[ij * bins + binOf[ij]] - mx - log(z)) / count;
        for (int k = 0; k < bins; ++k) dl[ij * bins + k] = 0.03 * (exp(L[ij * bins + k] - mx) / z - (k == binOf[ij])) / count;
      }
      std::vector<g::real> r(dl.begin(), dl.end());
      GK(cudaMemcpy(dLog, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    // the diffusion loss, a sample at a time (each sample's denoiser saves freed before the next)
    GK(cudaMemset(dTs, 0, (size_t)n * Cs * sizeof(g::real))); GK(cudaMemset(dTp, 0, nn * Cz * sizeof(g::real)));
    GK(cudaMemset(dTf, 0, (size_t)n * te.F * sizeof(g::real)));
    double lDiff = 0, lLddt = 0, rmsdLow = 0; int nLow = 0;
    for (int smp = 0; smp < samples; ++smp) {
      g::Scratch sc;
      // sigma (the last pass: a fixed ladder, so a step's number is comparable), the augmented truth, the noise
      const double sigma = train ? sd * exp(-1.2 + 1.5 * N01(rng)) : sd * exp(-1.2 + 1.5 * (-1.0 + 2.0 * smp / std::max(1, samples - 1)));
      double q[4]; double qn = 0; for (double& v : q) { v = N01(rng); qn += v * v; } qn = sqrt(qn);
      for (double& v : q) v /= qn;
      const double R[3][3] = {{1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])},
                              {2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[0] * q[1])},
                              {2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])}};
      double cen[3] = {0, 0, 0}; for (size_t i = 0; i < atoms; ++i) for (int c = 0; c < 3; ++c) cen[c] += w[i] * xt[i * 3 + c] / nAt;
      const double sh3[3] = {N01(rng), N01(rng), N01(rng)};
      std::vector<double> xa(atoms * 3, 0), xn(atoms * 3, 0);
      for (size_t i = 0; i < atoms; ++i) {
        if (!w[i]) continue;
        for (int p = 0; p < 3; ++p) {
          double v = sh3[p]; for (int c = 0; c < 3; ++c) v += R[p][c] * (xt[i * 3 + c] - cen[c]);
          xa[i * 3 + p] = v; xn[i * 3 + p] = v + sigma * N01(rng);
        }
      }
      { std::vector<g::real> r(xn.begin(), xn.end()); GK(cudaMemcpy(x, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice)); }
      auto ds = g::af3::denoiseForward(dn, b, singles[nPf], pairs[nMsa + nPf], tf, seqMask, x, (g::real)sigma, ri, ti, asym, ent, sym, xo);
      auto xh = g::host(xo, atoms * 3);
      std::vector<double> xp(xh.begin(), xh.end());
      // the truth aligned onto the prediction (no gradient through the alignment)
      double Ra[3][3], ta[3];
      kabsch(xa, xp, w, Ra, ta);
      const double lam = (sigma * sigma + sd * sd) / ((sigma * sd) * (sigma * sd));
      double mse = 0;
      std::vector<double> dg(atoms * 3, 0);
      for (size_t i = 0; i < atoms; ++i) {
        if (!w[i]) continue;
        for (int p = 0; p < 3; ++p) {
          double al = ta[p]; for (int c = 0; c < 3; ++c) al += Ra[p][c] * xa[i * 3 + c];
          const double e = xp[i * 3 + p] - al;
          mse += e * e / (3.0 * nAt);
          dg[i * 3 + p] = 4.0 * lam * 2 * e / (3.0 * nAt) / samples;
        }
      }
      // AF3's smooth LDDT (its Algorithm 27): over atom pairs whose true distance is under 15 A (30 for a nucleotide:
      // none here), eps = mean over c in {0.5, 1, 2, 4} of sigmoid(c - |d - d_true|), the loss 1 - its mean - frame
      // free, so on the prediction and the unaugmented truth, and its gradient into the denoised atoms by hand
      double num = 0, den = 0;
      std::vector<std::pair<std::pair<size_t, size_t>, double>> terms;   // (l, m) and d eps / d d
      for (size_t l : live) for (size_t m : live) {
        if (l == m) continue;
        double dt2 = 0, dp2 = 0;
        for (int p = 0; p < 3; ++p) {
          const double et = xt[l * 3 + p] - xt[m * 3 + p], ep = xp[l * 3 + p] - xp[m * 3 + p];
          dt2 += et * et; dp2 += ep * ep;
        }
        const double dtr = sqrt(dt2);
        if (dtr >= 15) continue;
        const double dpr = sqrt(dp2), del = fabs(dtr - dpr);
        double eps = 0, deps = 0;
        for (double c : {0.5, 1.0, 2.0, 4.0}) {
          const double sg = 1 / (1 + exp(-(c - del)));
          eps += sg / 4; deps -= sg * (1 - sg) / 4;          // d eps / d del
        }
        num += eps; den += 1;
        terms.push_back({{l, m}, deps * (dpr > dtr ? 1 : -1) / std::max(dpr, 1e-8)});   // d eps / d dpr, over dpr
      }
      const double lddt = den ? num / den : 1;
      for (auto& [lm, gd] : terms) {
        const size_t l = lm.first, m = lm.second;
        for (int p = 0; p < 3; ++p) {          // d(1 - lddt)/dx_l = -(1/den) d eps/d d (x_l - x_m)/d, both ends
          const double v = -4.0 * gd * (xp[l * 3 + p] - xp[m * 3 + p]) / den / samples;
          dg[l * 3 + p] += v; dg[m * 3 + p] -= v;
        }
      }
      lLddt += (1 - lddt) / samples;
      lDiff += lam * mse / samples;
      if (sigma < 4) { rmsdLow += sqrt(mse * 3); ++nLow; }
      if (train) {
        std::vector<g::real> r(dg.begin(), dg.end());
        GK(cudaMemcpy(dxo, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
        g::real *ts = g::ARENA.zeros((size_t)n * Cs), *tp = g::ARENA.zeros(nn * Cz), *tfd = g::ARENA.zeros((size_t)n * te.F);
        g::af3::denoiseBackward(dn, b, ds, singles[nPf], seqMask, dxo, ts, tp, tfd, nullptr);
        g::add(dTs, ts, (size_t)n * Cs); g::add(dTp, tp, nn * Cz); g::add(dTf, tfd, (size_t)n * te.F);
      }
    }
    const double loss = 4 * (lDiff + lLddt) + 0.03 * lDist;
    if (!train) {
      printf("%s  loss %.4f  (diffusion %.4f, smooth lddt %.4f, distogram %.4f; the fixed sigma ladder, rmsd at sigma < 4: %.3f A)\n",
             step < 0 ? "before  " : "after   ", loss, lDiff, lLddt, lDist, nLow ? rmsdLow / nLow : -1.0);
      // the smoke test's verdict: on the same sigmas, the loss above its floor must have come down by half - smooth
      // LDDT cannot reach zero (a perfect pair's eps is the mean of sigmoid(c), 0.804, so its loss floor is 0.196)
      double floor = 0; for (double cc : {0.5, 1.0, 2.0, 4.0}) floor += 4 * (1 - 1 / (1 + exp(-cc))) / 4;
      const double above = loss - floor;
      if (step < 0) { before = above; continue; }
      printf("above the floor (%.4f): %.4f -> %.4f\n", floor, before, above);
      const bool ok = above < 0.5 * before;
      printf("%s\n", ok ? "PASS" : "FAIL: the loss did not halve");
      return ok ? 0 : 1;
    }
    // ---- the backward: the distogram's and the denoiser's gradients into the trunk, block by block
    {
      g::real *dp = g::ARENA.zeros(nn * Cz), *dsg = g::ARENA.zeros((size_t)n * Cs);
      g::af3::distogramBackward(dh, pairs[nMsa + nPf], n, dLog, dp);
      g::add(dp, dTp, nn * Cz); g::add(dsg, dTs, (size_t)n * Cs);
      for (int k = nPf; k-- > 0;) {
        g::Scratch sc;
        g::real *dpi = g::ARENA.zeros(nn * Cz), *dsi = g::ARENA.zeros((size_t)n * Cs);
        g::af3::pairformerBackward(pf[k], d, pfS[k], pairMask, pairMaskT, seqMask, dp, dsg, dpi, dsi, nullptr);
        GK(cudaMemcpyAsync(dp, dpi, nn * Cz * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
        GK(cudaMemcpyAsync(dsg, dsi, (size_t)n * Cs * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
      }
      g::real* dzIn = doubleAdd ? g::cloneR(dp, nn * Cz) : nullptr;     // (boltz2's pre-MSA pair, added back)
      g::real* dm = g::ARENA.zeros(rows * Cm);
      for (int k = nMsa; k-- > 0;) {
        g::Scratch sc;
        g::real *dpi = g::ARENA.zeros(nn * Cz), *dmi = g::ARENA.zeros(rows * Cm);
        g::af3::msaBackward(msaB[k], d, msaS[k], msaMask, pairMask, pairMaskT, seqMask, dm, dp, dmi, dpi);
        GK(cudaMemcpyAsync(dp, dpi, nn * Cz * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
        GK(cudaMemcpyAsync(dm, dmi, rows * Cm * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
      }
      if (dzIn) g::add(dp, dzIn, nn * Cz);
      g::real* dz = g::cloneR(dp, nn * Cz);                // (the template term's residual, and its query path added)
      g::af3::templateBackward(tm, slots, tmS, pairMask, dp, dz);
      g::real* dtf = g::ARENA.zeros((size_t)n * te.F);
      g::af3::trunkEmbedBackward(te, eb, emS, tf, dz, dsg, dm, dtf, g::ARENA.zeros(nn * Cz), g::ARENA.zeros((size_t)n * Cs));
      g::add(dtf, dTf, (size_t)n * te.F);
      g::af3::targetFeatBackward(tfm, b, tfS, dtf);
    }
    opt.update(ps);
    const double gn = opt.lastNorm;
    cudaEventRecord(e1, STREAM); CK(cudaStreamSynchronize(STREAM));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    printf("step %3d  loss %.4f  (diffusion %.4f, smooth lddt %.4f, distogram %.4f)  cycles %d  |grad| %.3e  %.0f ms\n", step, loss, lDiff, lLddt, lDist, c, gn, ms);
    (void)trunkMark;
  }
  return 0;
}
