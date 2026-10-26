// BioEmu2 (microsoft/bioemu2, checkpoint bioemu2-preview) in CUDA: the windowed atom encoder, the PairMixer trunk
// (4 MSA layers, 48 triangle-multiplication blocks, 4 passes), the distogram, and the dual-stream denoiser under its
// EDM sampler - each stage held to af3-any-model's port (network/bioemu2.py, itself held to the vendor at 0.000 A)
// on the same input (cuda/bioemu2/oracle.py).
//
//   localfold-bioemu2 <input dir> --weights=<dir with bioemu2*.bin.zst> [--oracle=<dir>] [--out=fold.pdb]
//       [--samples=5] [--steps=200] [--seed=N] [--fast] [--serve=<dir>] [--sampler=edm|edm_gd|edm_fk] [--potentials=<yaml>]
// <input dir> is cuda/bioemu2/export_input.mjs's; its mode (structure, equilibrium, propagation) is the input's.
// --sampler=edm_gd / edm_fk: BioEmu2's steered samplers (src/steer.h, held to shared/bioemu2/sampler.js to the bit on a
// toy denoiser), every sample a particle through this denoiser, the potentials on the host; --sampler=edm_fkc the
// Feynman-Kac corrector, which takes the potentials' gradient back THROUGH the denoiser (src/vjp.cuh); the input needs its
// steering tables (featurised with --steering). --potentials replaces the default potentials, as the vendor's flag.
//   localfold-bioemu2 --job=<job.json> --out=<pdb> [...]  (any first argument that is a flag): the weights fetched, the
// input featurised in this process while the device starts, the fold (cuda/featurise/standalone.h)
#include "fasttrunk.cuh"
#include "sampler.cuh"
#include "potentials_yaml.h"
#include "vjp.cuh"
#include "../../featurise/standalone_api.h"
#include <thread>
#include <chrono>
#include <unistd.h>

using namespace be2;

static inline double ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
static std::string readText(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

struct Opts { std::string dir, oracle, out = "fold.pdb"; uint64_t seed = 0; int samples = 5; int steps = 200;
              bool stepsGiven = false; double stepScale = -1; int passes = 4; std::string frames;
              std::string sampler = "edm", potentials; };

// ---- steering: the input's tables (steer/*, steering-tables.js's, the featuriser's --steering) as steer.h reads them
static be2steer::Tables steerTables() {
  if (!M.has("steer/atom_pad_mask"))
    throw std::runtime_error("this input has no steering tables: featurise it with --steering for a steered sampler");
  auto I = [](const char* k) {
    const std::string n = std::string("steer/") + k;
    const size_t L = M.len(n);
    const int* p = L ? M.i(n) : nullptr;
    return std::vector<int>(p, p + L);
  };
  auto F = [](const char* k) {
    const std::string n = std::string("steer/") + k;
    const size_t L = M.len(n);
    const float* p = L ? M.f(n) : nullptr;
    return std::vector<double>(p, p + L);
  };
  auto rows = [](const std::vector<int>& v, size_t k) {
    std::vector<std::vector<int>> out(k);
    const size_t n = k ? v.size() / k : 0;
    for (size_t r = 0; r < k; ++r) out[r].assign(v.begin() + r * n, v.begin() + (r + 1) * n);
    return out;
  };
  be2steer::Tables t;
  t.pad = I("atom_pad_mask");
  t.nPad = (int)t.pad.size();
  t.chain = I("atom_chain_id");
  t.radius = F("atom_vdw_radius");
  t.chainSizes = I("chain_sizes");
  auto sym = rows(I("symmetric_chain_index"), 2); t.symA = sym[0]; t.symB = sym[1];
  t.interMask = I("interchain_steric_atom_mask");
  t.polyMask = I("polymer_steric_atom_mask");
  for (int k : I("interchain_bonded_pair_key")) t.interKeys.insert(k);
  for (int k : I("polymer_bonded_pair_key")) t.polyKeys.insert(k);
  auto cl = rows(I("polymer_local_clash_index"), 2); t.clashI = cl[0]; t.clashJ = cl[1];
  auto bi = rows(I("bond_index"), 2); t.bondI = bi[0]; t.bondJ = bi[1];
  auto ai = rows(I("angle_index"), 3); t.angleI = ai[0]; t.angleJ = ai[1]; t.angleC = ai[2];
  auto rb = rows(I("rdkit_bounds_index"), 2); t.rbI = rb[0]; t.rbJ = rb[1];
  t.rbBond = I("rdkit_bounds_bond_mask"); t.rbAngle = I("rdkit_bounds_angle_mask");
  t.rbUpper = F("rdkit_upper_bounds"); t.rbLower = F("rdkit_lower_bounds");
  auto quads = [&](const char* k) {
    auto r = rows(I(k), 4);
    std::vector<std::array<int, 4>> out;
    for (size_t c = 0; c < r[0].size(); ++c) out.push_back({r[0][c], r[1][c], r[2][c], r[3][c]});
    return out;
  };
  t.chiral = quads("chiral_atom_index"); t.chiralOri = I("chiral_atom_orientations");
  t.stereo = quads("stereo_bond_index"); t.stereoOri = I("stereo_bond_orientations");
  auto pl = rows(I("planar_bond_index"), 6);
  for (size_t c = 0; c < pl[0].size(); ++c) t.planar.push_back({pl[0][c], pl[1][c], pl[2][c], pl[3][c], pl[4][c], pl[5][c]});
  for (const char* k : {"bond_ref", "angle_ref"}) {
    std::vector<int> w = I((std::string(k) + ".bits").c_str());
    std::vector<double> d(w.size() / 2);
    if (!d.empty()) std::memcpy(d.data(), w.data(), d.size() * 8);
    (std::string(k) == "bond_ref" ? t.bondRef : t.angleRef) = d;
  }
  return t;
}

// the steered samplers with this port's denoiser: every particle through one call (BATCH_SAMPLES at a time), the
// draws this port's own (a seeded mt19937 per particle, as the plain sampler's; one more for the resampling)
static std::vector<std::vector<float>> steeredFold(const Denoiser& d, const SamplerSettings& ss, const Opts& o, int n,
                                                   const std::function<Denoiser()>& exact = nullptr) {
  using namespace be2steer;
  const int N = d.N;
  const std::vector<Spec> specs = o.potentials.empty() ? defaultSpecs() : potentialSpecs(parseSimpleYaml(readText(o.potentials)));
  Tables t = steerTables();
  if (t.nPad != N) throw std::runtime_error("the steering tables are for " + std::to_string(t.nPad) + " atoms, the input " + std::to_string(N));
  hostTables(t, specs);
  std::vector<double> mask(t.pad.begin(), t.pad.end());
  std::vector<std::mt19937_64> rng;
  for (int p = 0; p < n; ++p) rng.emplace_back(o.seed * 1000003ull + (uint64_t)p);
  std::mt19937_64 resampling(o.seed * 1000003ull + 7919ull);
  std::normal_distribution<double> Nd(0, 1);
  Draws draws;
  draws.x0 = [&] {
    Particles x(n, std::vector<double>((size_t)N * 3));
    for (int p = 0; p < n; ++p) for (auto& v : x[p]) v = Nd(rng[p]);
    return x;
  };
  draws.step = [&](int, std::vector<std::array<double, 9>>& rot, std::vector<std::array<double, 3>>& trans, Particles& eps) {
    rot.resize(n); trans.resize(n); eps.assign(n, std::vector<double>((size_t)N * 3));
    for (int p = 0; p < n; ++p) {
      double R[9], sh[3];
      randomRigid(rng[p], R, sh);
      std::copy(R, R + 9, rot[p].begin()); std::copy(sh, sh + 3, trans[p].begin());
      for (auto& v : eps[p]) v = Nd(rng[p]);
    }
  };
  draws.uniform = [&](int, int groups) {
    std::uniform_real_distribution<double> U(0, 1);
    std::vector<double> u(groups);
    for (auto& v : u) v = U(resampling);
    return u;
  };
  const size_t per = (size_t)N * 3;
  const int batch = std::max(1, std::min(n, BATCH_SAMPLES));
  float* dx = dalloc(per * batch); float* dd = dalloc(per * batch);
  std::vector<float> hx(per * batch), hd(per * batch);
  double denoiseMs = 0;
  Denoise denoiser = [&](const Particles& xs, double sigma, const Particles*) {
    auto t0 = std::chrono::steady_clock::now();
    Particles out(xs.size(), std::vector<double>(per));
    for (size_t p0 = 0; p0 < xs.size(); p0 += batch) {
      const int b = (int)std::min<size_t>(batch, xs.size() - p0);
      for (int k = 0; k < b; ++k) for (size_t i = 0; i < per; ++i) hx[per * k + i] = (float)xs[p0 + k][i];
      CK(cudaMemcpyAsync(dx, hx.data(), per * b * 4, cudaMemcpyHostToDevice, STREAM));
      denoise(d, dx, (float)sigma, dd, nullptr, b);
      CK(cudaMemcpyAsync(hd.data(), dd, per * b * 4, cudaMemcpyDeviceToHost, STREAM));
      CK(cudaStreamSynchronize(STREAM));
      for (int k = 0; k < b; ++k) for (size_t i = 0; i < per; ++i) out[p0 + k][i] = hd[per * k + i];
    }
    denoiseMs += ms(t0);
    return out;
  };
  if (o.sampler == "edm_fkc") {
    // edm_fkc: every particle's D and the potentials' gradient back through the denoiser (src/vjp.cuh), one particle a
    // tape; U_hat at the step's t_hat parameters, every spec weighted 1, as sampler.js's fkcSample asks
    double vjpMs = 0;
    size_t held = 0;
    Denoiser de = exact ? exact() : Denoiser{};
    const Denoiser& dv = exact ? de : d;
    DenoiseGrad withGrad = [&](const Particles& xs, double sigma, const Particles*, double tHat) {
      auto t0 = std::chrono::steady_clock::now();
      DenoisedGrad r;
      const StepParams prm = stepParameters(tHat, specs);
      const std::vector<double> ones(specs.size(), 1.0);
      float* xd = dalloc(per);
      std::vector<float> hxf(per);
      for (size_t p = 0; p < xs.size(); ++p) {
        for (size_t i = 0; i < per; ++i) hxf[i] = (float)xs[p][i];
        CK(cudaMemcpyAsync(xd, hxf.data(), per * 4, cudaMemcpyHostToDevice, STREAM));
        double up = 0;
        be2vjp::VjpResult v = be2vjp::denoiseVjp(dv, xd, sigma, nullptr, nullptr, [&](const std::vector<float>& den) {
          std::vector<double> dd(den.begin(), den.end()), gD(per, 0.0);
          auto e = energies(dd.data(), t, prm, specs, gD.data(), &ones);
          for (double v : e) up = up + v;
          return std::vector<float>(gD.begin(), gD.end());
        });
        r.u += up;
        r.den.emplace_back(v.den.begin(), v.den.end());
        r.gx.emplace_back(v.gx.begin(), v.gx.end());
        r.gs.push_back(v.gs);
        held = std::max(held, v.bytes);
      }
      CK(cudaFree(xd));
      vjpMs += ms(t0);
      return r;
    };
    FkcSettings fo; fo.steps = ss.steps;
    auto t0 = std::chrono::steady_clock::now();
    int nonFinite = -1;
    Particles x = fkcSample(withGrad, t, mask, n, draws, fo, specs, nullptr, nullptr, &nonFinite);
    printf("steered (edm_fkc, %zu potentials): the denoiser and its VJP %.1f ms (%.2f GB a tape), the host %.1f ms%s\n",
           specs.size(), vjpMs, held / 1e9, ms(t0) - vjpMs,
           nonFinite >= 0 ? (", a NON-FINITE weight from step " + std::to_string(nonFinite)).c_str() : "");
    be2vjp::ARENA.release();
    if (exact) freeDenoiser(de);
    CK(cudaFree(dx)); CK(cudaFree(dd));
    std::vector<float> fmask(mask.begin(), mask.end());
    std::vector<std::vector<float>> out;
    for (auto& v : x) { std::vector<float> f(v.begin(), v.end()); centreMasked(f, fmask); out.push_back(f); }
    return out;
  }
  SteerSettings so;
  so.steps = ss.steps; so.stepScale = ss.stepScale; so.noiseScale = ss.noiseScale; so.gamma0 = ss.gamma0; so.gammaMin = ss.gammaMin;
  so.resamplingInterval = o.sampler == "edm_fk" ? 3 : 0;
  auto t0 = std::chrono::steady_clock::now();
  Particles x = steeredSample(denoiser, t, mask, n, draws, so, specs);
  const double total = ms(t0);
  printf("steered (%s, %zu potentials): the denoiser %.1f ms, the host %.1f ms (FK energies %.1f, guidance %.1f,"
         " realignment %.1f)\n", o.sampler.c_str(), specs.size(), denoiseMs, total - denoiseMs, STEER_TIMES.energies,
         STEER_TIMES.guidance, STEER_TIMES.align);
  STEER_TIMES = {};
  CK(cudaFree(dx)); CK(cudaFree(dd));
  std::vector<float> fmask(mask.begin(), mask.end());
  std::vector<std::vector<float>> out;
  for (auto& v : x) {
    std::vector<float> f(v.begin(), v.end());
    centreMasked(f, fmask);
    out.push_back(f);
  }
  return out;
}


static std::string stemOf(const std::string& pdb) {
  return pdb.size() > 4 && pdb.substr(pdb.size() - 4) == ".pdb" ? pdb.substr(0, pdb.size() - 4) : pdb;
}

// <stem>_confidences.json and <stem>_summary_confidences.json, AlphaFold 3's layout as the other ports write it:
// the contact probabilities (the distogram's P(d < 8 A)), the token ids, in the job's token order - and NO pLDDT,
// PAE or pTM, because the checkpoint has no confidence head; they are null, not invented
static void writeConfidences(const std::string& dir, const std::string& pdb, int T, const std::vector<float>& contacts) {
  std::ifstream in(dir + "/confidence.template.json");
  if (!in) { fprintf(stderr, "no %s/confidence.template.json (export_input.mjs writes it)\n", dir.c_str()); exit(1); }
  Json j = Json::parse(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
  std::vector<int> order; for (auto& v : j.get("tokens")->arr) order.push_back((int)v.num);
  const int n = (int)order.size();
  if (n != T) { fprintf(stderr, "confidence.template.json names %d tokens of %d\n", n, T); exit(1); }
  std::vector<float> c((size_t)n * n);
  for (int i = 0; i < n; ++i) for (int k = 0; k < n; ++k) c[(size_t)i * n + k] = contacts[(size_t)order[i] * T + order[k]];
  const std::string stem = stemOf(pdb);
  FILE* f = fopen((stem + "_confidences.json").c_str(), "w");
  std::string s = "{\"contact_probs\": ";
  appendMatrix2(s, c.data(), n);
  fwrite(s.data(), 1, s.size(), f);
  fprintf(f, ",\n \"pae\": null,\n \"token_plddts\": null,\n \"atom_plddts\": null,\n \"token_chain_ids\": [");
  auto& ids = j.get("token_chain_ids")->arr; auto& res = j.get("token_res_ids")->arr;
  for (int i = 0; i < n; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", ids[i].str.c_str());
  fprintf(f, "],\n \"token_res_ids\": [");
  for (int i = 0; i < n; ++i) fprintf(f, "%s%d", i ? ", " : "", (int)res[i].num);
  fprintf(f, "],\n \"note\": \"BioEmu2 has no confidence head: pLDDT, PAE and pTM are not predicted\"}\n");
  fclose(f);
  f = fopen((stem + "_summary_confidences.json").c_str(), "w");
  fprintf(f, "{\"ptm\": null, \"iptm\": null, \"mean_plddt\": null, \"ranking_score\": null,"
             " \"note\": \"BioEmu2 has no confidence head\"}\n");
  fclose(f);
}

// the contact map: the softmax mass of the first bins of each pair's 64 (Boltz's, over 2-22 A), up to 8 A
__global__ void contactProbsK(const float* logits, float* out, size_t pairs, int bins, int near) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= pairs) return;
  const float* l = logits + ij * bins;
  float mx = -INFINITY; for (int b = 0; b < bins; ++b) mx = fmaxf(mx, l[b]);
  float tot = 0, c = 0;
  for (int b = 0; b < bins; ++b) { float e = expf(l[b] - mx); tot += e; if (b < near) c += e; }
  out[ij] = c / tot;
}

static bool FAST_PATH = false;
// a denoiser output against the oracle's over the REAL atoms: a padding atom's coordinates are undefined on both sides
static double checkRealAtoms(const char* label, const float* d, const std::string& oracle) {
  const size_t n = (size_t)M.meta("meta/real_atoms") * 3;
  std::vector<float> h = download(d, n);
  double r = relRms(h.data(), M.f(oracle), n);
  printf("  %-30s relRMS %.3e\n", label, r);
  return r;
}
static int foldInput(const Opts& o) {
  SAMPLER_COND.forget();       // (a fold's conditioning is its own: sampler.cuh)
  GEMM_HALF = false;
  auto t0 = std::chrono::steady_clock::now();
  const int T = (int)M.meta("meta/tokens"), N = (int)M.meta("meta/atoms"), S = (int)M.meta("meta/msa_rows");
  D_ = {T, N, S, (int)M.meta("meta/n_valid_rows")};
  const bool check = !o.oracle.empty();
  const bool equilibrium = M.meta("meta/mode") == 1;
  const int chain = (int)M.meta("meta/propagation_steps", 0);
  printf("BioEmu2: %d tokens, %d atoms (%d padded), %d MSA rows, %s%s\n", T, (int)M.meta("meta/real_atoms"), N, S,
         equilibrium || chain ? "equilibrium" : "structure",
         chain ? (", propagating " + std::to_string(chain) + " step(s)").c_str() : "");
  buildTokenStarts(T, N);
  forgetRowMaps();
  std::map<std::string, double> worst;
  auto tap = [&](const char* name, const float* d, size_t n) {
    if (!check) return;
    double r = checkOracle(name, d, n, std::string("o/") + name);
    if (r >= 0) worst[name] = r;
  };
  const int Si = 455, Cs = 384, C = 256;
  size_t Pn = (size_t)T * T;
  const bool reuseZ = !check && o.sampler == "edm";
  float* sInputs = dalloc((size_t)T * Si);
  inputEmbedder(sInputs, Si, tap);
  float* sInit = dalloc((size_t)T * Cs);
  float* zInit = (reuseZ && Pn > ((size_t)1 << 20)) ? nullptr : dalloc(Pn * C);
  trunkInit(sInputs, Si, sInit, zInit, tap);
  float* s = dalloc((size_t)T * Cs); float* z = dalloc(Pn * C);
  CK(cudaMemsetAsync(s, 0, (size_t)T * Cs * 4, STREAM)); CK(cudaMemsetAsync(z, 0, Pn * C * 4, STREAM));
  CK(cudaStreamSynchronize(STREAM));
  printf("inputs %.1f ms\n", ms(t0)); t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < o.passes; ++k) {
    trunkPass(sInputs, Si, sInit, zInit, s, z);
    std::string ks = "pass" + std::to_string(k + 1);
    tap((ks + ".single").c_str(), s, (size_t)T * Cs);
    tap((ks + ".pair").c_str(), z, Pn * C);
  }
  CK(cudaStreamSynchronize(STREAM));
  printf("trunk %.1f ms (%d passes)\n", ms(t0), o.passes); t0 = std::chrono::steady_clock::now();
  CK(cudaFree(sInit)); if (zInit) CK(cudaFree(zInit));
  releaseScratch();      // (the trunk's buffers - a gigabyte of triangle operands at 500 tokens - are the denoiser's room)
  const int B = dimP("distogram_module/distogram/weight", 0);
  std::vector<float> contacts;
  {
    // Boltz's 64 bins over 2-22 A: the bins whose lower edges (linspace(2, 22, 63)) are at most 8 A
    int near = 0; for (int e = 0; e < B - 1; ++e) if (2.0 + 20.0 * e / (B - 2) <= 8.0) ++near;
    float* probs = scratch<float>("be2.dg.contacts", Pn);
    if (check) {
      float* logits = scratch<float>("be2.dg.logits", Pn * B);
      distogram(z, logits);
      tap("distogram", logits, Pn * B);
      contactProbsK<<<blocks(Pn), 256, 0, STREAM>>>(logits, probs, Pn, B, near);
    } else {
      size_t R = std::max<size_t>(1, std::min<size_t>(T, ((size_t)256 << 10) / std::max(T, 1)));
      float* zs = scratch<float>("be2.dg.zs", R * T * C);
      float* logits = scratch<float>("be2.dg.logits", R * T * B);
      for (size_t r0 = 0; r0 < (size_t)T; r0 += R) {
        size_t r = std::min<size_t>(R, T - r0), n = r * T;
        symmetriseRowsK<<<blocks(n * C), 256, 0, STREAM>>>(z, zs, T, C, r0, r);
        linear(zs, P("distogram_module/distogram/weight"), logits, n, C, B);
        addBiasRows(logits, P("distogram_module/distogram/bias"), n, B);
        contactProbsK<<<blocks(n), 256, 0, STREAM>>>(logits, probs + r0 * T, n, B, near);
      }
    }
    contacts = download(probs, Pn);
  }
  // (the denoiser side - its static context and every call - on the f16 tensor cores under --fast; the trunk keeps TF32)
  GEMM_HALF = FAST_PATH;
  releaseScratch();
  Denoiser d = staticContext(s, z, sInputs, Si, tap, reuseZ);
  if (reuseZ) {
    z = nullptr;
    for (float* p : {sInputs, s}) CK(cudaFree(p));
    sInputs = s = nullptr;
    releaseScratch();
  }
  CK(cudaStreamSynchronize(STREAM));
  printf("static context %.1f ms\n", ms(t0));
  if (check && M.has("o/denoise0.x")) {
    // each recorded level, from OUR trunk and context, then from the reference's own (the denoiser alone)
    float* den = dalloc((size_t)N * 3);
    auto levels = [&](const Denoiser& dn, const char* tag) {
      for (int k = 0; M.has("o/denoise" + std::to_string(k) + ".x"); ++k) {
        std::string key = "o/denoise" + std::to_string(k);
        float* x = upload(M.f(key + ".x"), (size_t)N * 3);
        denoise(dn, x, (float)M.meta(key + ".sigma"), den);
        char label[64]; snprintf(label, sizeof label, "denoise sigma %g (%s)", M.meta(key + ".sigma"), tag);
        double r = checkRealAtoms(label, den, key + ".out");
        worst[label] = r;
        CK(cudaFree(x));
      }
    };
    levels(d, "own trunk");
    float* so = upload(M.f("o/pass" + std::to_string(o.passes) + ".single"), (size_t)T * Cs);
    float* zo = upload(M.f("o/pass" + std::to_string(o.passes) + ".pair"), Pn * C);
    Denoiser dr = staticContext(so, zo, sInputs, Si, nullptr);
    levels(dr, "reference trunk");
    // the denoiser's vector-Jacobian product (src/vjp.cuh) against jax.grad of w . D(x, sigma), from the reference's
    // trunk: the forward first (the tape's D against this port's denoise), then d/dx - its NETWORK part, gx - c_skip w,
    // because at sigma 16 the network's part nearly cancels c_skip w (docs/BIOEMU2.md) - and d/dsigma
    Denoiser dre = FAST_PATH ? be2vjp::exactContext(so, zo, sInputs, Si) : Denoiser{};
    const Denoiser& drv = FAST_PATH ? dre : dr;
    for (int k = 0; M.has("o/vjp" + std::to_string(k) + ".x"); ++k) {
      const std::string key = "o/vjp" + std::to_string(k);
      const double sg = M.meta(key + ".sigma");
      float* x = upload(M.f(key + ".x"), (size_t)N * 3);
      float* w = upload(M.f(key + ".w"), (size_t)N * 3);
      auto t0 = std::chrono::steady_clock::now();
      be2vjp::VjpResult v = be2vjp::denoiseVjp(drv, x, sg, w);
      const double took = ms(t0);
      denoise(drv, x, (float)sg, den);
      std::vector<float> plain = download(den, (size_t)N * 3);
      const size_t real = (size_t)M.meta("meta/real_atoms") * 3;
      const double cs = 256.0 / (sg * sg + 256.0);
      std::vector<double> mine(real), want(real);
      const float* wh = M.f(key + ".w"); const float* gh = M.f(key + ".gx");
      double num = 0, den2 = 0;
      for (size_t i = 0; i < real; ++i) {
        mine[i] = v.gx[i] - cs * wh[i]; want[i] = gh[i] - cs * wh[i];
        num += (mine[i] - want[i]) * (mine[i] - want[i]); den2 += want[i] * want[i];
      }
      const double rx = sqrt(num / den2);
      const double rf = relRms(v.den.data(), plain.data(), real);
      const double gsWant = M.meta(key + ".gs"), rs = fabs(v.gs - gsWant) / std::max(fabs(gsWant), 1e-12);
      char label[96];
      snprintf(label, sizeof label, "vjp sigma %g, d/dx (network)", sg);
      printf("  %-30s relRMS %.3e\n", label, rx);
      worst[label] = rx;
      snprintf(label, sizeof label, "vjp sigma %g, d/dsigma", sg);
      printf("  %-30s relRMS %.3e   (%.6e against %.6e)\n", label, rs, v.gs, gsWant);
      worst[label] = rs;
      printf("    (the tape's D against this port's denoise: relRMS %.3e; the forward and backward %.1f ms, %.2f GB held)\n",
             rf, took, v.bytes / 1e9);
      CK(cudaFree(x)); CK(cudaFree(w));
    }
    if (FAST_PATH) freeDenoiser(dre);
    be2vjp::ARENA.release();
    if (M.has("o/denoise_dyn.x")) {
      float* cc = upload(M.f("o/denoise_dyn.coords_cond"), (size_t)N * 3);
      setDynamics(dr, (float)M.meta("o/denoise_dyn.dt"), cc);
      float* x = upload(M.f("o/denoise_dyn.x"), (size_t)N * 3);
      denoise(dr, x, (float)M.meta("o/denoise_dyn.sigma"), den);
      double r = checkRealAtoms("denoise, propagation (reference trunk)", den, "o/denoise_dyn.out");
      worst["denoise_dyn"] = r;
      CK(cudaFree(x)); CK(cudaFree(cc));
    }
    freeDenoiser(dr); CK(cudaFree(so)); CK(cudaFree(zo)); CK(cudaFree(den));
    double w = 0; for (auto& [k, v] : worst) w = std::max(w, v);
    printf("worst stage relRMS %.3e\n", w);
  }
  if (o.samples > 0) {
    t0 = std::chrono::steady_clock::now();
    SamplerSettings ss; ss.steps = o.steps;
    ss.stepScale = o.stepScale > 0 ? o.stepScale : equilibrium || chain ? 1.0 : 1.5;
    std::vector<std::vector<float>> out;
    if (o.sampler != "edm") {
      if (chain) { fprintf(stderr, "Error: BioEmu2 propagation runs unsteered here; use the edm sampler to propagate\n"); return 1; }
      try {
        out = steeredFold(d, ss, o, o.samples, FAST_PATH ? std::function<Denoiser()>([&] {
          return be2vjp::exactContext(s, z, sInputs, Si); }) : nullptr);
      } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        freeDenoiser(d);
        return 1;
      }
    } else if (chain) {
      std::vector<float> x0(M.f("x0"), M.f("x0") + (size_t)N * 3);
      auto frames = propagate(d, ss, o.seed, o.samples, chain, x0);
      for (auto& tr : frames) for (auto& f : tr) out.push_back(f);    // (trajectory i, step k: sample i * chain + k)
    } else {
      out = sample(d, ss, o.seed, o.samples);
    }
    CK(cudaStreamSynchronize(STREAM));
    printf("sampler %.1f ms (%d sample(s), %d steps)\n", ms(t0), (int)out.size(), ss.steps);
    const std::string stem = stemOf(o.out);
    for (size_t k = 0; k < out.size(); ++k) {
      std::string path = k == 0 ? o.out : stem + "_sample-" + std::to_string(k) + ".pdb";
      writePdb(o.dir + "/pdb.template", path, out[k]);
    }
    writeConfidences(o.dir, o.out, T, contacts);
    printf("-> %s%s\n", o.out.c_str(), out.size() > 1 ? (" (+" + std::to_string(out.size() - 1) + " more samples)").c_str() : "");
  }
  freeDenoiser(d);
  for (float* p : {sInputs, s, z}) if (p) CK(cudaFree(p));
  releaseScratch();
  return 0;
}

static bool flagInto(const std::string& f, Opts& o) {
  if (!f.compare(0, 6, "--out=")) o.out = f.substr(6);
  else if (!f.compare(0, 7, "--seed=")) o.seed = strtoull(f.c_str() + 7, nullptr, 10);
  else if (!f.compare(0, 8, "--steps=")) { o.steps = atoi(f.c_str() + 8); o.stepsGiven = true; }
  else if (!f.compare(0, 10, "--samples=")) o.samples = atoi(f.c_str() + 10);
  else if (!f.compare(0, 13, "--step-scale=")) o.stepScale = atof(f.c_str() + 13);
  else if (!f.compare(0, 9, "--passes=")) o.passes = atoi(f.c_str() + 9);
  else if (!f.compare(0, 10, "--sampler=")) o.sampler = f.substr(10);
  else if (!f.compare(0, 13, "--potentials=")) o.potentials = f.substr(13);
  else return false;
  return true;
}

static int foldMain(int argc, char** argv) {
  if (argc < 2 || !strncmp(argv[1], "--", 2)) {
    fprintf(stderr, "usage: localfold-bioemu2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--samples=5]"
                    " [--steps=200] [--seed=N] [--step-scale=S] [--fast] [--serve=<dir>]\n");
    return 1;
  }
  Opts o; o.dir = argv[1];
  std::string weights, serveDir, compareWith;
  bool fast = false, waitInput = false, waitForever = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (!a.compare(0, 10, "--weights=")) weights = a.substr(10);
    else if (a == "--wait-input") waitInput = true;              // start up while the input is still being exported
    else if (a == "--wait-input=0") waitInput = waitForever = true;   // ...with no timeout (the standalone mode)
    else if (!a.compare(0, 9, "--oracle=")) o.oracle = a.substr(9);
    else if (a == "--fast") fast = true;
    else if (!a.compare(0, 18, "--compare-weights=")) compareWith = a.substr(18);
    else if (!a.compare(0, 8, "--serve=")) serveDir = a.substr(8);
    else if (!flagInto(a, o)) { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
  }
  if (weights.empty()) { fprintf(stderr, "--weights=<dir holding bioemu2*.bin.zst>\n"); return 1; }
  if (o.steps < 2) { fprintf(stderr, "BioEmu2 samples with at least 2 steps, got %d\n", o.steps); return 1; }
  if (o.sampler != "edm" && o.sampler != "edm_gd" && o.sampler != "edm_fk" && o.sampler != "edm_fkc") {
    fprintf(stderr, "Error: BioEmu2 sampler %s: edm, edm_gd, edm_fk or edm_fkc\n", o.sampler.c_str());
    return 1;
  }
  auto tStart = std::chrono::steady_clock::now();
  M.loadBundle(weights, "w");
  for (int sgi = 0; sgi < (int)M.segs.size(); ++sgi) M.upload(sgi);
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  CB(cublasSetMathMode(H, fast ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
  FAST_PATH = fast;
  if (fast) { FAST = true; TRIMUL = trimulFast; PAIR_TRANSITION = pairTransitionFast; }   // (ef2's FAST: its fused forms)
  printf("weights up in %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count());
  if (!compareWith.empty()) {
    // every tensor of these weights against another blob's (the int8 decode against float32): relRMS per tensor
    M.loadBundle(compareWith, "f");
    double worstR = 0; std::string worstName; size_t n = 0;
    for (auto& [k, e] : M.index) {
      if (e.kind == 'm' || k.compare(0, 2, "w/")) continue;
      std::string f = "f/" + k.substr(2);
      if (!M.has(f)) continue;
      auto a = download(W(k), M.len(k)); auto b = download(W(f), M.len(f));
      double r = relRms(a.data(), b.data(), a.size());
      printf("  %-110s %.3e\n", k.c_str() + SCOPE.size(), r);
      if (r > worstR) { worstR = r; worstName = k; }
      ++n;
    }
    printf("%zu tensors, worst %.3e (%s)\n", n, worstR, worstName.c_str());
    finish(0);
  }
  const int weightSegs = (int)M.segs.size();
  if (!serveDir.empty()) {
    serveJobs(serveDir, "bioemu2", [&](const std::string& input, const std::vector<std::string>& flags) {
      Opts j = o; j.dir = input; j.out = "fold.pdb"; j.oracle.clear();
      for (auto& f : flags) flagInto(f, j);
      int seg = (int)M.segs.size();
      M.load(input);
      int code = foldInput(j);
      CK(cudaStreamSynchronize(STREAM));
      forgetEntries(M.unload(seg));
      return code;
    });
    finish(0);
  }
  if (waitInput) {          // the exporter writes model.idx last, by a rename
    std::string idx = o.dir + "/model.idx", failed = o.dir + "/model.failed";
    for (int k = 0; access(idx.c_str(), R_OK) != 0; ++k) {
      if (access(failed.c_str(), F_OK) == 0) { fprintf(stderr, "bioemu2: the input's export failed\n"); finish(1); }
      if (k > 600000 && !waitForever) { fprintf(stderr, "no %s after ten minutes\n", idx.c_str()); finish(1); }
      usleep(1000);
    }
  }
  M.load(o.dir);
  if (!o.oracle.empty()) M.load(o.oracle);
  for (int sgi = weightSegs; sgi < (int)M.segs.size(); ++sgi) M.upload(sgi);
  finish(foldInput(o));
}

int main(int argc, char** argv) {
  if (argc < 2 || !strncmp(argv[1], "--", 2) || !strcmp(argv[1], "-h")) return lf::standalone::main("bioemu2", argc, argv, foldMain);
  return foldMain(argc, argv);
}
