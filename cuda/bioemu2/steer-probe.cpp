// cuda/bioemu2/src/steer.h on the host, as JSON, for tools/check-native-bioemu2-steering.mjs to hold to
// shared/bioemu2/potentials.js and sampler.js (and to af3-any-model's recorded run):
//
//   steer-probe potentials <topology dump>                 every default potential's energy and gradient, one-hot
//                                                          weights, on the dump's frames at its recorded times
//   steer-probe sample <steering dump> <topology dump>     edm_gd and edm_fk on the dump's toy denoiser and draws
//   steer-probe specs <potentials.yaml>                    the file's potentials and their parameters over t
//
// The tables are the dump's own host tables (tools/oracle/dump_bioemu2_topology.py), as the JavaScript checks read
// them, so a difference is in the potentials or the sampler and never in a table. Built with the featuriser's flags
// (g++ -std=c++17 -O2 -ffp-contract=off).
#include <cstdio>
#include <chrono>
#include <iostream>
#include <random>

#include "../featurise/json.h"
#include "src/potentials_yaml.h"

using namespace be2steer;

static std::string readAll(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot read " + path);
  std::string s;
  char buf[1 << 16];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
  fclose(f);
  return s;
}
// (Python's json writes NaN and Infinity, which JSON does not have: null here, as the JavaScript checks read them)
static lf::Json load(const std::string& path) {
  std::string t = readAll(path), out;
  out.reserve(t.size());
  for (size_t i = 0; i < t.size();) {
    if (!t.compare(i, 3, "NaN")) { out += "null"; i += 3; }
    else if (!t.compare(i, 9, "-Infinity")) { out += "null"; i += 9; }
    else if (!t.compare(i, 8, "Infinity")) { out += "null"; i += 8; }
    else out += t[i++];
  }
  return lf::JsonReader(out).parse();
}
static double num(const lf::Json& v) {
  if (v.kind == lf::Json::Bool) return v.b ? 1 : 0;
  if (v.kind == lf::Json::Null) return std::numeric_limits<double>::quiet_NaN();
  return v.n;
}
static std::vector<double> flat(const lf::Json& v) {
  std::vector<double> out;
  std::function<void(const lf::Json&)> walk = [&](const lf::Json& x) {
    if (x.kind == lf::Json::Array) for (auto& y : x.a) walk(y); else out.push_back(num(x));
  };
  walk(v);
  return out;
}
static std::vector<int> ints(const std::vector<double>& v) { return std::vector<int>(v.begin(), v.end()); }

// the dump's host tables (hostTables applied for the default potentials, as the dump records them)
static Tables hostFromDump(const lf::Json& host) {
  auto F = [&](const char* k) { const lf::Json* v = host.get(k); if (!v) throw std::runtime_error(std::string("no ") + k); return flat(*v); };
  auto rowsOf = [&](const char* k) {
    std::vector<std::vector<int>> out;
    for (auto& r : host.get(k)->a) out.push_back(ints(flat(r)));
    return out;
  };
  Tables t;
  t.pad = ints(F("atom_pad_mask"));
  t.nPad = (int)t.pad.size();
  t.chain = ints(F("atom_chain_id"));
  t.radius = F("atom_vdw_radius");
  t.chainSizes = ints(F("chain_sizes"));
  auto sp = rowsOf("symmetric_chain_pairs");
  if (sp.size() == 2) { t.pairA = sp[0]; t.pairB = sp[1]; }
  t.interMask = ints(F("interchain_steric_atom_mask"));
  t.polyMask = ints(F("polymer_steric_atom_mask"));
  for (double k : F("interchain_bonded_pair_key")) t.interKeys.insert((long long)k);
  for (double k : F("polymer_bonded_pair_key")) t.polyKeys.insert((long long)k);
  auto cl = rowsOf("polymer_local_clash_index"); t.clashI = cl[0]; t.clashJ = cl[1];
  auto bi = rowsOf("bond_index"); t.bondI = bi[0]; t.bondJ = bi[1];
  auto ai = rowsOf("angle_index"); t.angleI = ai[0]; t.angleJ = ai[1]; t.angleC = ai[2];
  auto rb = rowsOf("rdkit_bounds_index");
  if (rb.size() == 2) { t.rbI = rb[0]; t.rbJ = rb[1]; }
  t.rbBond = ints(F("rdkit_bounds_bond_mask")); t.rbAngle = ints(F("rdkit_bounds_angle_mask"));
  t.rbUpper = F("rdkit_upper_bounds"); t.rbLower = F("rdkit_lower_bounds");
  auto quads = [&](const char* k) {
    std::vector<std::array<int, 4>> out;
    auto r = rowsOf(k);
    if (r.size() == 4) for (size_t c = 0; c < r[0].size(); ++c) out.push_back({r[0][c], r[1][c], r[2][c], r[3][c]});
    return out;
  };
  t.chiral = quads("chiral_atom_index"); t.chiralOri = ints(F("chiral_atom_orientations"));
  t.stereo = quads("stereo_bond_index"); t.stereoOri = ints(F("stereo_bond_orientations"));
  t.planarImp = quads("planar_improper_index");
  t.bondLower[8] = F("bond_lower.8"); t.bondUpper[8] = F("bond_upper.8");
  t.angleLower[9] = F("angle_lower.9"); t.angleUpper[9] = F("angle_upper.9");
  exclusions(t);
  return t;
}

static void printArray(const std::vector<double>& v) {
  char buf[40];
  std::cout << "[";
  for (size_t i = 0; i < v.size(); ++i) { snprintf(buf, sizeof buf, "%.17g", v[i]); std::cout << (i ? "," : "") << buf; }
  std::cout << "]";
}

int main(int argc, char** argv) {
  try {
    std::string what = argc > 1 ? argv[1] : "";
    if (what == "potentials" && argc == 3) {
      lf::Json dump = load(argv[2]);
      const lf::Json& steer = *dump.get("steer");
      Tables t = hostFromDump(*steer.get("host"));
      const auto specs = defaultSpecs();
      std::vector<std::vector<double>> frames;
      for (auto& f : dump.get("coords")->a) frames.push_back(flat(f));
      std::cout << "{\"times\":[";
      bool first = true;
      for (auto& at : steer.get("times")->a) {
        const double tt = at.get("t")->n;
        StepParams p = stepParameters(tt, specs);
        std::cout << (first ? "" : ",") << "{\"t\":" << tt << ",\"interval\":[";
        auto iv = guidanceInterval(tt, specs);
        for (size_t i = 0; i < iv.size(); ++i) std::cout << (i ? "," : "") << iv[i];
        std::cout << "],\"energies\":[";
        std::string grads = "";
        for (size_t i = 0; i < specs.size(); ++i) {
          std::vector<double> e;
          std::cout << (i ? "," : "") << "[";
          for (size_t f = 0; f < frames.size(); ++f) {
            std::vector<double> grad(frames[f].size(), 0.0), w(specs.size(), 0.0);
            w[i] = 1;
            double v = energies(frames[f].data(), t, p, specs, grad.data(), &w)[i];
            char buf[40]; snprintf(buf, sizeof buf, "%.17g", v);
            std::cout << (f ? "," : "") << buf;
            grads += (f || i ? "," : "");
            std::string g = "[";
            for (size_t k = 0; k < grad.size(); ++k) { snprintf(buf, sizeof buf, "%.17g", grad[k]); g += (k ? "," : ""); g += buf; }
            grads += g + "]";
          }
          std::cout << "]";
        }
        std::cout << "],\"grads\":[" << grads << "]}";
        first = false;
      }
      std::cout << "]}\n";
      return 0;
    }
    if (what == "sample" && argc == 4) {
      lf::Json dump = load(argv[2]);
      lf::Json topo = load(argv[3]);
      Tables t = hostFromDump(*topo.get("steer")->get("host"));
      std::vector<double> mask(t.pad.begin(), t.pad.end());
      const int P = (int)dump.get("particles")->n, steps = (int)dump.get("steps")->n;
      const lf::Json* gd = dump.get("gd_steps");
      const std::vector<double> ref = flat(*dump.get("ref"));
      const lf::Json& D = *dump.get("draws");
      Draws draws;
      draws.x0 = [&] { Particles x; for (auto& p : D.get("x0")->a) x.push_back(flat(p)); return x; };
      draws.step = [&](int k, std::vector<std::array<double, 9>>& rot, std::vector<std::array<double, 3>>& trans, Particles& eps) {
        rot.clear(); trans.clear(); eps.clear();
        for (auto& r : D.get("rot")->a[k].a) { auto v = flat(r); std::array<double, 9> a; std::copy(v.begin(), v.end(), a.begin()); rot.push_back(a); }
        for (auto& r : D.get("trans")->a[k].a) { auto v = flat(r); trans.push_back({v[0], v[1], v[2]}); }
        for (auto& e : D.get("eps")->a[k].a) eps.push_back(flat(e));
      };
      draws.uniform = [&](int k, int) { return flat(D.get("uniform")->a[k]); };
      const double SD = 16;
      Denoise toy = [&](const Particles& x, double sigma, const Particles*) {
        const double c = (SD * SD) / (sigma * sigma + SD * SD);
        Particles out(x.size());
        for (size_t p = 0; p < x.size(); ++p) {
          out[p].resize(x[p].size());
          for (size_t i = 0; i < x[p].size(); ++i) out[p][i] = c * x[p][i] + (1 - c) * ref[i];
        }
        return out;
      };
      std::cout << "{";
      bool first = true;
      for (auto [name, interval] : {std::pair<const char*, int>{"edm_gd", 0}, {"edm_fk", 3}}) {
        SteerSettings o;
        o.steps = steps; o.stepScale = 1.5; o.resamplingInterval = interval; o.gdSteps = gd ? (int)gd->n : 20;
        std::vector<TraceRow> trace;
        Particles x = steeredSample(toy, t, mask, P, draws, o, defaultSpecs(), nullptr, &trace);
        std::cout << (first ? "" : ",") << "\"" << name << "\":{\"x\":[";
        for (int p = 0; p < P; ++p) { std::cout << (p ? "," : ""); printArray(x[p]); }
        std::cout << "],\"idx\":[";
        for (size_t k = 0; k < trace.size(); ++k) {
          std::cout << (k ? "," : "") << "[";
          for (size_t i = 0; i < trace[k].idx.size(); ++i) std::cout << (i ? "," : "") << trace[k].idx[i];
          std::cout << "]";
        }
        std::cout << "]}";
        first = false;
      }
      std::cout << "}\n";
      return 0;
    }
    if (what == "bench" && argc == 3) {          // (each potential's gradient, timed: where the host's time goes)
      lf::Json dump = load(argv[2]);
      Tables t = hostFromDump(*dump.get("steer")->get("host"));
      const auto specs = defaultSpecs();
      std::vector<double> x = flat(dump.get("coords")->a[0]);
      // (a denoised estimate early in sampling is a COLLAPSED molecule: the sample shrunk toward its centre, which is
      // where the overlap sums are dense)
      if (const char* f = getenv("SHRINK")) for (auto& v : x) v *= atof(f);
      StepParams p = stepParameters(0.9, specs);
      for (size_t i = 0; i < specs.size(); ++i) {
        std::vector<double> w(specs.size(), 0.0), grad(x.size());
        w[i] = 1;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 200; ++r) energies(x.data(), t, p, specs, grad.data(), &w, true);
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
        printf("%-32s %8.1f us a gradient\n", specs[i].kind.c_str(), us);
      }
      return 0;
    }
    if (what == "toy" && argc == 4) {            // (a whole steered run on the toy denoiser, random draws: its cost)
      lf::Json dump = load(argv[2]);
      Tables t = hostFromDump(*dump.get("steer")->get("host"));
      std::vector<double> mask(t.pad.begin(), t.pad.end());
      const std::vector<double> ref = flat(dump.get("coords")->a[0]);
      const int P = 5, n = (int)mask.size();
      std::mt19937_64 g(1);
      std::normal_distribution<double> Nd(0, 1);
      Draws draws;
      draws.x0 = [&] { Particles x(P, std::vector<double>(n * 3)); for (auto& v : x) for (auto& a : v) a = Nd(g); return x; };
      draws.step = [&](int, std::vector<std::array<double, 9>>& rot, std::vector<std::array<double, 3>>& trans, Particles& eps) {
        rot.assign(P, {1, 0, 0, 0, 1, 0, 0, 0, 1}); trans.assign(P, {0, 0, 0});
        eps.assign(P, std::vector<double>(n * 3)); for (auto& v : eps) for (auto& a : v) a = Nd(g);
      };
      draws.uniform = [&](int, int groups) { return std::vector<double>(groups, 0.5); };
      Denoise toy = [&](const Particles& x, double sigma, const Particles*) {
        const double c = getenv("COMPACT") ? 0.0 : 256 / (sigma * sigma + 256);
        Particles out(x.size(), std::vector<double>(n * 3));
        for (size_t p = 0; p < x.size(); ++p) for (int i = 0; i < n * 3; ++i) out[p][i] = c * x[p][i] + (1 - c) * ref[i];
        return out;
      };
      SteerSettings o; o.steps = atoi(argv[3]); o.stepScale = 1.5;
      auto t0 = std::chrono::steady_clock::now();
      steeredSample(toy, t, mask, P, draws, o, defaultSpecs());
      printf("%d steps, %d particles: %.1f s\n", o.steps, P, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      return 0;
    }
    if (what == "specs" && argc == 3) {
      std::vector<Spec> specs;
      try {
        specs = potentialSpecs(parseSimpleYaml(readAll(argv[2])));
      } catch (const std::exception& e) {
        std::string m;
        for (char c : std::string(e.what())) { if (c == '"' || c == '\\') m += '\\'; m += c; }
        std::cout << "{\"error\":\"" << m << "\"}\n";
        return 0;
      }
      std::cout << "{\"kinds\":[";
      for (size_t i = 0; i < specs.size(); ++i) std::cout << (i ? "," : "") << "\"" << specs[i].kind << "\"";
      std::cout << "],\"tolerance\":[";
      for (size_t i = 0; i < specs.size(); ++i) {
        char buf[40]; snprintf(buf, sizeof buf, "%.17g", specs[i].sigmaTolerance);
        std::cout << (i ? "," : "") << (std::isnan(specs[i].sigmaTolerance) ? "null" : buf);
      }
      std::cout << "],\"at\":[";
      for (int k = 0; k <= 20; ++k) {
        const double t = k / 20.0;
        StepParams p = stepParameters(t, specs);
        std::cout << (k ? "," : "") << "{\"guidance_weight\":"; printArray(p.guidance);
        std::cout << ",\"resampling_weight\":"; printArray(p.resampling);
        for (auto& name : paramNames()) {
          std::cout << ",\"" << name << "\":[";
          for (size_t i = 0; i < specs.size(); ++i) {
            char buf[40]; snprintf(buf, sizeof buf, "%.17g", p.buffers[name][i]);
            std::cout << (i ? "," : "") << (std::isnan(p.buffers[name][i]) ? "null" : buf);
          }
          std::cout << "]";
        }
        auto iv = guidanceInterval(t, specs);
        std::cout << ",\"interval\":[";
        for (size_t i = 0; i < iv.size(); ++i) std::cout << (i ? "," : "") << iv[i];
        std::cout << "]}";
      }
      std::cout << "]}\n";
      return 0;
    }
    std::cerr << "usage: steer-probe potentials <topology dump> | sample <steering dump> <topology dump> | specs <potentials.yaml>\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
