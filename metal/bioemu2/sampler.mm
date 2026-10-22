// BioEmu2's EDM sampler and the output (cuda/bioemu2/src/sampler.cuh and bioemu2.cu are the reading - the host side
// is the same code, its draws the same seeded mt19937s, so a fold's samples are cuda/bioemu2's up to the denoiser):
//   levels: sigma_data (sMax^(1/rho) + k/(n-1) (sMin^(1/rho) - sMax^(1/rho)))^rho, k < n, then a 0 appended
//   x = levels[0] N(0, 1)
//   each step (sigma -> sigma'): centre on the real atoms, rotate at random, translate by N(0, 1), zero the padding;
//     gamma = gamma0 where the NEXT level exceeds gammaMin;  t = sigma (1 + gamma)
//     x_noisy = x + noiseScale sqrt(t^2 - sigma^2) N(0, 1);  d = D(x_noisy, t)
//     x_noisy = Kabsch(x_noisy onto d);  x = x_noisy + stepScale (sigma' - t) (x_noisy - d) / t
//   each sample centred on its real atoms at the end
// With a COMPANION (propagation's conditioning frame) the frame takes the state's rotation and translation every step,
// each centred on its own mean, and is the denoiser's coords_cond.
#include "bioemu2.h"
#include "json.h"
#include "../../cuda/af3/src/rigid.cuh"     // (host code: the realigning samplers' weighted Kabsch, shared to the bit)
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

int BATCH_SAMPLES = 64;

static std::vector<float> schedule(const SamplerSettings& s) {
  std::vector<float> v;
  const float hi = powf((float)s.sMax, 1.f / (float)s.rho), lo = powf((float)s.sMin, 1.f / (float)s.rho);
  for (int k = 0; k < s.steps; ++k) {
    float t = s.steps == 1 ? 0.f : (float)k / (float)(s.steps - 1);
    v.push_back(SIGMA_DATA * powf(hi + t * (lo - hi), (float)s.rho));
  }
  v.push_back(0.f);
  return v;
}
// a uniform random rotation (a normalised Gaussian quaternion) and a N(0, 1) translation, applied to x centred on its
// masked mean; padding zeroed
static void augment(std::vector<float>& x, const std::vector<float>& mask, const double R[9], const double sh[3]) {
  const size_t N = mask.size();
  double tot = 0, c[3] = {0, 0, 0};
  for (size_t a = 0; a < N; ++a) { tot += mask[a]; for (int k = 0; k < 3; ++k) c[k] += mask[a] * x[a * 3 + k]; }
  for (int k = 0; k < 3; ++k) c[k] /= tot + 1e-6;
  for (size_t a = 0; a < N; ++a) {
    double p[3] = {x[a * 3] - c[0], x[a * 3 + 1] - c[1], x[a * 3 + 2] - c[2]};
    for (int k = 0; k < 3; ++k)
      x[a * 3 + k] = mask[a] ? (float)(p[0] * R[0 * 3 + k] + p[1] * R[1 * 3 + k] + p[2] * R[2 * 3 + k] + sh[k]) : 0.f;
  }
}
void randomRigid(std::mt19937_64& rng, double R[9], double sh[3]) {
  std::normal_distribution<double> Nd(0, 1);
  double q[4]; for (auto& v : q) v = Nd(rng);
  double qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  double w = q[0] / qn, i = q[1] / qn, j = q[2] / qn, k = q[3] / qn;
  double r[9] = {1 - 2 * (j * j + k * k), 2 * (i * j - k * w), 2 * (i * k + j * w),
                 2 * (i * j + k * w), 1 - 2 * (i * i + k * k), 2 * (j * k - i * w),
                 2 * (i * k - j * w), 2 * (j * k + i * w), 1 - 2 * (i * i + j * j)};
  for (int a = 0; a < 9; ++a) R[a] = r[a];
  for (int a = 0; a < 3; ++a) sh[a] = Nd(rng);
}
void centreMasked(std::vector<float>& x, const std::vector<float>& mask) {
  const size_t N = mask.size(); double tot = 0, c[3] = {0, 0, 0};
  for (size_t a = 0; a < N; ++a) { tot += mask[a]; for (int k = 0; k < 3; ++k) c[k] += mask[a] * x[a * 3 + k]; }
  for (int k = 0; k < 3; ++k) c[k] /= std::max(tot, 1e-6);
  for (size_t a = 0; a < N; ++a) for (int k = 0; k < 3; ++k) x[a * 3 + k] = mask[a] ? (float)(x[a * 3 + k] - c[k]) : 0.f;
}
static std::vector<float> atomMask(int N) {
  std::vector<float> mask(N);
  const int* m = M.hostI("atom_mask");
  for (int a = 0; a < N; ++a) mask[a] = m[a] ? 1.f : 0.f;
  return mask;
}

std::vector<std::vector<float>> sample(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int n,
                                       std::vector<std::vector<float>> companions, bool centreEnd, const StepHook& onStep) {
  const int N = d.N;
  const std::vector<float> mask = atomMask(N);
  const std::vector<float> levels = schedule(s);
  std::vector<std::mt19937_64> rng;
  for (int p = 0; p < n; ++p) rng.emplace_back(seed * 1000003ull + (uint64_t)p);
  std::normal_distribution<double> Nd(0, 1);
  std::vector<std::vector<float>> x(n, std::vector<float>((size_t)N * 3)), den(n, std::vector<float>((size_t)N * 3));
  for (int p = 0; p < n; ++p) for (auto& v : x[p]) v = (float)(levels[0] * Nd(rng[p]));
  const bool comp = !companions.empty();
  const size_t per = (size_t)N * 3;
  // the particles go through the denoiser together, `batch` at a time
  const int batch = std::max(1, std::min(n, BATCH_SAMPLES));
  float* dx = allocT<float>(per * batch); float* dd = allocT<float>(per * batch);
  float* dc = comp ? allocT<float>(per * batch) : nullptr;
  const int steps = (int)levels.size() - 1;
  std::vector<float> stepSigmas(steps);
  for (int i = 0; i < steps; ++i) {
    const double sigma = levels[i], next = levels[i + 1];
    const double gamma = next > s.gammaMin ? s.gamma0 : 0;
    stepSigmas[i] = (float)(sigma * (1 + gamma));
  }
  setSigmaSchedule(stepSigmas.data(), steps);
  for (int i = 0; i < steps; ++i) {
    const double sigma = levels[i], next = levels[i + 1];
    const double gamma = next > s.gammaMin ? s.gamma0 : 0;
    const double t = sigma * (1 + gamma);
    const double eps = s.noiseScale * sqrt(std::max(t * t - sigma * sigma, 0.0));
    for (int p = 0; p < n; ++p) {
      double R[9], sh[3];
      randomRigid(rng[p], R, sh);
      augment(x[p], mask, R, sh);
      if (comp) augment(companions[p], mask, R, sh);
      for (int a = 0; a < N * 3; ++a) x[p][a] += (float)(eps * Nd(rng[p]));
    }
    for (int p0 = 0; p0 < n; p0 += batch) {
      const int b = std::min(batch, n - p0);
      float* hx = (float*)host(dx);
      for (int k = 0; k < b; ++k) std::copy(x[p0 + k].begin(), x[p0 + k].end(), hx + per * k);
      if (comp) { float* hc = (float*)host(dc); for (int k = 0; k < b; ++k) std::copy(companions[p0 + k].begin(), companions[p0 + k].end(), hc + per * k); }
      denoise(d, dx, (float)t, dd, dc, b);
      mt::sync();
      const float* hd = (const float*)host(dd);
      for (int k = 0; k < b; ++k) std::copy(hd + per * k, hd + per * (k + 1), den[p0 + k].begin());
    }
    if (onStep) onStep(i + 1, steps, den[0]);
    for (int p = 0; p < n; ++p) {
      rigidAlign(x[p], den[p], mask, N);
      const double f = s.stepScale * (next - t) / t;
      for (int k = 0; k < N * 3; ++k) x[p][k] = (float)(x[p][k] + f * (x[p][k] - den[p][k]));
    }
  }
  setSigmaSchedule(nullptr, 0);
  release(dx); release(dd); if (dc) release(dc);
  if (centreEnd) for (auto& v : x) centreMasked(v, mask);
  else for (auto& v : x) for (int a = 0; a < N; ++a) if (!mask[a]) v[a * 3] = v[a * 3 + 1] = v[a * 3 + 2] = 0.f;
  return x;
}

// propagation: each particle a trajectory of `chain` lag-time steps from x0, each step conditioned on the previous frame
// (centred, the companion) and its endpoint superposed on that frame. [particle][step] frames
std::vector<std::vector<std::vector<float>>> propagate(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int n,
                                                       int chain, const std::vector<float>& x0) {
  const int N = d.N;
  const std::vector<float> mask = atomMask(N);
  std::vector<std::vector<float>> prev(n, x0);
  std::vector<std::vector<std::vector<float>>> frames(n);
  for (int k = 0; k < chain; ++k) {
    std::vector<std::vector<float>> comps = prev;
    for (auto& c : comps) centreMasked(c, mask);
    std::vector<std::vector<float>> end = sample(d, s, seed * 7919ull + (uint64_t)k, n, comps, false);
    for (int p = 0; p < n; ++p) {
      rigidAlign(end[p], prev[p], mask, N);
      for (int a = 0; a < N; ++a) if (!mask[a]) end[p][a * 3] = end[p][a * 3 + 1] = end[p][a * 3 + 2] = 0.f;
      frames[p].push_back(end[p]);
      prev[p] = end[p];
    }
  }
  return frames;
}

// the records of pdb.template (the featuriser's: an atom's index as x * 1000 + y) with these coordinates
void writePdb(const std::string& templatePath, const std::string& out, const std::vector<float>& x) {
  std::ifstream in(templatePath);
  if (!in) die("no %s (the featuriser writes it)", templatePath.c_str());
  std::string text = "REMARK   1 BioEmu2 (bioemu2-preview): no confidence head - the B-factor column is zero, not a pLDDT\n";
  std::string line;
  while (std::getline(in, line)) {
    if ((line.rfind("ATOM", 0) == 0 || line.rfind("HETATM", 0) == 0) && line.size() >= 66) {
      int atom = (int)lround(atof(line.substr(30, 8).c_str())) * 1000 + (int)lround(atof(line.substr(38, 8).c_str()));
      char coords[64];
      snprintf(coords, sizeof coords, "%8.3f%8.3f%8.3f", x[atom * 3], x[atom * 3 + 1], x[atom * 3 + 2]);
      line = line.substr(0, 30) + coords + line.substr(54);
    }
    text += line + "\n";
  }
  writeWhole(out, text.data(), text.size());
}

static std::string stemOf(const std::string& pdb) {
  return pdb.size() > 4 && pdb.substr(pdb.size() - 4) == ".pdb" ? pdb.substr(0, pdb.size() - 4) : pdb;
}
// <stem>_confidences.json and <stem>_summary_confidences.json in AlphaFold 3's layout: the contact probabilities and the
// token ids in the job's token order - and NO pLDDT, PAE or pTM, which the checkpoint has no head for: null, not invented
void writeConfidences(const std::string& dir, const std::string& pdb, int T, const std::vector<float>& contacts) {
  std::ifstream in(dir + "/confidence.template.json");
  if (!in) die("no %s/confidence.template.json (the featuriser writes it)", dir.c_str());
  Json j = Json::parse(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
  std::vector<int> order; for (auto& v : j.get("tokens")->arr) order.push_back((int)v.num);
  const int n = (int)order.size();
  if (n != T) die("confidence.template.json names %d tokens of %d", n, T);
  std::string s = "{\"contact_probs\": [";
  char b[32];
  for (int i = 0; i < n; ++i) {
    s += i ? ",\n  [" : "[";
    for (int k = 0; k < n; ++k) {
      snprintf(b, sizeof b, "%s%.2f", k ? ", " : "", contacts[(size_t)order[i] * T + order[k]]);
      s += b;
    }
    s += "]";
  }
  s += "],\n \"pae\": null,\n \"token_plddts\": null,\n \"atom_plddts\": null,\n \"token_chain_ids\": [";
  auto& ids = j.get("token_chain_ids")->arr; auto& res = j.get("token_res_ids")->arr;
  for (int i = 0; i < n; ++i) s += std::string(i ? ", " : "") + "\"" + ids[i].str + "\"";
  s += "],\n \"token_res_ids\": [";
  for (int i = 0; i < n; ++i) s += (i ? ", " : "") + std::to_string((int)res[i].num);
  s += "],\n \"note\": \"BioEmu2 has no confidence head: pLDDT, PAE and pTM are not predicted\"}\n";
  const std::string stem = stemOf(pdb);
  writeWhole(stem + "_confidences.json", s.data(), s.size());
  const std::string sum = "{\"ptm\": null, \"iptm\": null, \"mean_plddt\": null, \"ranking_score\": null,"
                          " \"note\": \"BioEmu2 has no confidence head\"}\n";
  writeWhole(stem + "_summary_confidences.json", sum.data(), sum.size());
}
