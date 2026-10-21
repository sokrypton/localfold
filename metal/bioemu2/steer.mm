// BioEmu2's steered samplers on Metal: edm_gd (gradient guidance on the potentials' energy of each denoised estimate)
// and edm_fk (that, plus Feynman-Kac resampling every third step). The steering is cuda/bioemu2's own host code -
// src/steer.h and src/potentials_yaml.h, held to shared/bioemu2/sampler.js to the bit on a toy denoiser - included, not
// copied; only the denoiser under it is this port's (bioemu2.cu's steeredFold is the reading, its draws the same seeded
// mt19937s). edm_fkc is refused by name: it takes the potentials' gradient back THROUGH the denoiser (cuda/bioemu2's
// vjp.cuh, the denoiser's whole backward), which this port does not have.
#include "bioemu2.h"
#include "../../cuda/bioemu2/src/potentials_yaml.h"
#include <chrono>
#include <fstream>
#include <random>

namespace {
using namespace be2steer;

std::string readText(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
// the input's steering tables (steer/*: steering-tables.js's, the featuriser's --steering) as steer.h reads them
Tables steerTables() {
  if (!M.has("steer/atom_pad_mask"))
    throw std::runtime_error("this input has no steering tables: featurise it with --steering for a steered sampler");
  auto I = [](const char* k) {
    const std::string n = std::string("steer/") + k;
    const size_t L = M.has(n) ? M.len(n) : 0;
    const int* p = L ? M.hostI(n) : nullptr;
    return std::vector<int>(p, p + L);
  };
  auto F = [](const char* k) {
    const std::string n = std::string("steer/") + k;
    const size_t L = M.has(n) ? M.len(n) : 0;
    const float* p = L ? M.hostF(n) : nullptr;
    return std::vector<double>(p, p + L);
  };
  auto rows = [](const std::vector<int>& v, size_t k) {
    std::vector<std::vector<int>> out(k);
    const size_t n = k ? v.size() / k : 0;
    for (size_t r = 0; r < k; ++r) out[r].assign(v.begin() + r * n, v.begin() + (r + 1) * n);
    return out;
  };
  Tables t;
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
    std::vector<int> w = I((std::string(k) + ".bits").c_str());      // (float64s carried as pairs of int32s)
    std::vector<double> d(w.size() / 2);
    if (!d.empty()) std::memcpy(d.data(), w.data(), d.size() * 8);
    (std::string(k) == "bond_ref" ? t.bondRef : t.angleRef) = d;
  }
  return t;
}
}  // namespace

std::vector<std::vector<float>> steeredFold(const Denoiser& d, const SamplerSettings& ss, const std::string& sampler,
                                            const std::string& potentials, uint64_t seed, int n) {
  if (sampler == "edm_fkc")
    throw std::runtime_error("BioEmu2's edm_fkc takes the potentials' gradient back through the denoiser, which the Metal"
                             " port does not have (cuda/bioemu2's vjp.cuh): edm, edm_gd or edm_fk here");
  if (sampler != "edm_gd" && sampler != "edm_fk") throw std::runtime_error("BioEmu2 sampler " + sampler + ": edm, edm_gd or edm_fk");
  const int N = d.N;
  const std::vector<Spec> specs = potentials.empty() ? defaultSpecs() : potentialSpecs(parseSimpleYaml(readText(potentials)));
  Tables t = steerTables();
  if (t.nPad != N) throw std::runtime_error("the steering tables are for " + std::to_string(t.nPad) + " atoms, the input " + std::to_string(N));
  hostTables(t, specs);
  std::vector<double> mask(t.pad.begin(), t.pad.end());
  std::vector<std::mt19937_64> rng;
  for (int p = 0; p < n; ++p) rng.emplace_back(seed * 1000003ull + (uint64_t)p);
  std::mt19937_64 resampling(seed * 1000003ull + 7919ull);
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
  // every particle through this port's denoiser, BATCH_SAMPLES at a time
  const size_t per = (size_t)N * 3;
  const int batch = std::max(1, std::min(n, BATCH_SAMPLES));
  float* dx = allocT<float>(per * batch); float* dd = allocT<float>(per * batch);
  double denoiseMs = 0;
  Denoise denoiser = [&](const Particles& xs, double sigma, const Particles*) {
    const double t0 = now();
    Particles out(xs.size(), std::vector<double>(per));
    for (size_t p0 = 0; p0 < xs.size(); p0 += batch) {
      const int b = (int)std::min<size_t>(batch, xs.size() - p0);
      float* hx = (float*)host(dx);
      for (int k = 0; k < b; ++k) for (size_t i = 0; i < per; ++i) hx[per * k + i] = (float)xs[p0 + k][i];
      denoise(d, dx, (float)sigma, dd, nullptr, b);
      mt::sync();
      const float* hd = (const float*)host(dd);
      for (int k = 0; k < b; ++k) for (size_t i = 0; i < per; ++i) out[p0 + k][i] = hd[per * k + i];
    }
    denoiseMs += (now() - t0) * 1e3;
    return out;
  };
  SteerSettings so;
  so.steps = ss.steps; so.stepScale = ss.stepScale; so.noiseScale = ss.noiseScale; so.gamma0 = ss.gamma0; so.gammaMin = ss.gammaMin;
  so.resamplingInterval = sampler == "edm_fk" ? 3 : 0;
  const double t0 = now();
  Particles x = steeredSample(denoiser, t, mask, n, draws, so, specs);
  const double total = (now() - t0) * 1e3;
  printf("steered (%s, %zu potentials): the denoiser %.1f ms, the host %.1f ms (FK energies %.1f, guidance %.1f,"
         " realignment %.1f)\n", sampler.c_str(), specs.size(), denoiseMs, total - denoiseMs, STEER_TIMES.energies,
         STEER_TIMES.guidance, STEER_TIMES.align);
  STEER_TIMES = {};
  release(dx); release(dd);
  std::vector<float> fmask(mask.begin(), mask.end());
  std::vector<std::vector<float>> out;
  for (auto& v : x) {
    std::vector<float> f(v.begin(), v.end());
    centreMasked(f, fmask);
    out.push_back(f);
  }
  return out;
}
