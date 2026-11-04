// The flow sampler: RFD4-Proteina's trajectory, step by step, over a generation `cuda/rfd4p/export_input.mjs
// --generate` wrote. `shared/rfd4p/flow.js` is the spec for the schedules and the step.
//
// 🔴 THE NOISE IS REPLAYED, NOT REDRAWN, and that is the export's own reasoning rather than a convenience here:
// Box-Muller runs through V8's `Math.log`, which no port has a bit-exact copy of, and one ulp in one draw
// diverges a sixty-step trajectory. The draw ORDER is a function of the schedule alone, so the stream can be
// written out without running the model - and a port that replays it is comparing its SAMPLER against the
// reference's rather than its random number generator.
//
// 🔴 AND ONLY 35 OF THE 897 SEQUENCE COLUMNS MOVE: `xt_bb_ca` at 0 and `xt_local_latents` at 3. Everything else
// is exported once. The pair track's first 30 columns - the alpha carbons' distance bins - are rebuilt on the
// device each step from the current coordinates, which is the one feature that cannot be frozen or replayed.
#include "rfd4p.h"
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

namespace {

std::vector<double> linspace(double a, double b, int n) {
  std::vector<double> v((size_t)n);
  for (int i = 0; i < n; ++i) v[(size_t)i] = n == 1 ? a : a + (b - a) * i / (n - 1);
  return v;
}

// `get_schedule_tsr_safe` (eps 1e-3, where the score is defined) and `get_schedule` (from zero)
std::vector<double> baseSchedule(int mode, int nsteps, double p, bool tsrSafe) {
  const double eps = tsrSafe ? 1e-3 : 0.0;
  if (mode == 0) return linspace(eps, 1, nsteps + 1);                       // uniform
  if (mode == 1) {                                                          // power
    std::vector<double> v = linspace(tsrSafe ? std::pow(eps, 1 / p) : 0.0, 1, nsteps + 1);
    for (double& x : v) x = std::pow(x, p);
    return v;
  }
  if (mode == 2) {                                                          // log
    if (!(p > 0)) die("rfd4p: p must be > 0 for the log schedule, got %g", p);
    const std::vector<double> e = linspace(-p, tsrSafe ? eps : 0.0, nsteps + 1);
    std::vector<double> t((size_t)nsteps + 1);
    for (int i = 0; i <= nsteps; ++i) t[(size_t)i] = 1 - std::pow(10.0, e[(size_t)(nsteps - i)]);
    double mn = INFINITY;
    for (double v : t) mn = std::min(mn, v);
    for (double& v : t) v -= mn - (tsrSafe ? eps : 0.0);
    double mx = -INFINITY;
    for (double v : t) mx = std::max(mx, v);
    for (double& v : t) v /= mx;
    return t;
  }
  die("rfd4p: schedule mode %d is not one this port knows", mode);
}

// `_schedule_with_tail_multiplier`: the last `tail_steps` of the base schedule at `multiplier` resolution
std::vector<double> withTail(int mode, double p, int tailSteps, int mult, int nsteps, bool tsrSafe) {
  const int ts = nsteps >= tailSteps * mult ? tailSteps : nsteps / (2 * std::max(1, mult));
  if (!ts) return baseSchedule(mode, nsteps, p, tsrSafe);
  const int baseSteps = nsteps - (mult - 1) * ts;
  const int fine = baseSteps * mult;
  const int tailStart = (baseSteps - ts) * mult;
  const std::vector<double> full = baseSchedule(mode, fine, p, tsrSafe);
  std::vector<double> out;
  for (int i = 0; i < tailStart; i += mult) out.push_back(full[(size_t)i]);
  for (int i = tailStart; i <= fine; ++i) out.push_back(full[(size_t)i]);
  return out;
}

}  // namespace

bool stochasticStep(double t, const Modality& m) {
  if (m.mode == 1) return false;                                   // vf: the flow's ODE, never
  if (m.mode == 2) return !(t > m.odeLim);                         // sc: stochastic up to its ODE tail
  return t != 0 && !(t > m.odeLim || t < m.odeBelow);              // vf_tsr: between the two limits
}

Modality modalitySchedule(const char* tag, int nsteps) {
  auto meta = [&](const char* field, double fallback) {
    return M.meta(std::string("meta/samp") + tag + field, fallback);
  };
  Modality m;
  m.mode = (int)meta("Mode", 0);
  m.tsrK = meta("TsrK", 1);
  m.tsrSigma = meta("TsrSigma", 1);
  m.scn = meta("Noise", 1);
  m.odeLim = meta("Ode", 1);
  m.odeBelow = meta("OdeBelow", 0);
  m.center = meta("Center", 0) != 0;
  m.ts = withTail((int)meta("Sched", 0), meta("P", 1), (int)meta("Tail", 0), (int)meta("TailMult", 1), nsteps,
                  m.mode == 0);
  // `get_gt`, then `transform_gt`'s power, then the clamp - and for vf_tsr divided by the tsr ratio squared
  const int gtMode = (int)meta("Gt", 0);
  const double gtP = meta("GtP", 1), clamp = meta("Clamp", 0);
  const bool hasClamp = meta("HasClamp", 0) != 0;
  m.gt.assign(m.ts.size() - 1, 0.0);
  if (gtMode != 3) {                                               // 3: ode, no noise at all
    for (size_t i = 0; i < m.gt.size(); ++i) {
      const double t = std::min(std::max(m.ts[i], 0.0), 1 - 1e-5);
      if (gtMode == 0) m.gt[i] = 1 / (t + 1e-2);
      else if (gtMode == 1) m.gt[i] = (1 - t) / (t + 1e-2);
      else if (gtMode == 2) m.gt[i] = (M_PI / 2) * std::sin((1 - t) * M_PI / 2) / (std::cos((1 - t) * M_PI / 2) + 1e-2);
      else die("rfd4p: gt mode %d is not one this port knows", gtMode);
    }
    if (gtP != 1) {
      double mean = 0;
      for (double g : m.gt) mean += std::log(g);
      mean /= (double)m.gt.size();
      for (double& g : m.gt) {
        const double sig = 1 / (1 + std::exp(-(std::log(g) - mean)));
        const double q = std::min(std::max(std::pow(sig, gtP), 1e-6), 1 - 1e-6);
        g = std::exp(std::log(q / (1 - q)) + mean);
      }
    }
    for (size_t i = 0; i < m.gt.size(); ++i) {
      double v = std::max(m.gt[i], 0.0);
      if (hasClamp) v = std::min(v, clamp);
      if (m.mode == 0) {
        const double snr = (m.ts[i] * m.ts[i]) / ((1 - m.ts[i]) * (1 - m.ts[i]));
        const double r = (snr * m.tsrSigma * m.tsrSigma + 1) / ((snr * m.tsrSigma * m.tsrSigma) / m.tsrK + 1);
        v /= r * r;
      }
      m.gt[i] = v;
    }
  }
  return m;
}

namespace {

// `sample_noise` for both modalities, out of the replayed stream: the backbone zero-centred over its emitted
// tokens, then one N(0, scale^2) offset a molecule instance for tokens with no absolute coordinate conditioning,
// then the latents. 🔴 THE DRAW ORDER IS THE SPECIFICATION - backbone, the offsets, the latents - and one
// transposition shifts every later draw of the whole run.
size_t initialNoise(const float* noise, float* bb, float* ll, int n, int latent, const float* emitBb,
                    const float* emitLl, const int* coordCond, const int* moleculeIid, float scale) {
  size_t at = 0;
  int cnt = 0;
  float mean[3] = {0, 0, 0};
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) bb[i * 3 + k] = emitBb[i] != 0 ? noise[at + (size_t)i * 3 + k] : 0.f;
    if (emitBb[i] == 0) continue;
    cnt += 1;
    for (int k = 0; k < 3; ++k) mean[k] += bb[i * 3 + k];
  }
  at += (size_t)n * 3;
  for (int k = 0; k < 3; ++k) mean[k] = (float)(mean[k] / std::max(cnt, 1));
  for (int i = 0; i < n; ++i)
    if (emitBb[i] != 0)
      for (int k = 0; k < 3; ++k) bb[i * 3 + k] = (float)(bb[i * 3 + k] - mean[k]);
  std::set<int> groups;
  for (int i = 0; i < n; ++i) if (emitBb[i] != 0 && !coordCond[i]) groups.insert(moleculeIid[i]);
  for (int g : groups) {
    const float off[3] = {noise[at], noise[at + 1], noise[at + 2]};
    at += 3;
    for (int i = 0; i < n; ++i) {
      if (!(emitBb[i] != 0 && !coordCond[i]) || moleculeIid[i] != g) continue;
      for (int k = 0; k < 3; ++k) bb[i * 3 + k] = (float)(bb[i * 3 + k] + (float)(scale * off[k]));
    }
  }
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < latent; ++k)
      ll[(size_t)i * latent + k] = emitLl[i] != 0 ? noise[at + (size_t)i * latent + k] : 0.f;
  at += (size_t)n * latent;
  return at;
}

}  // namespace

double (*GATE_CHECK)(const char*, const std::string&, const float*, size_t, double) = nullptr;
double GATE_TRAJ_BOUND = 1e-4;

int generate(const std::string& genDir, const std::string& outPath) {
  (void)genDir;
  const int n = D_.n, latent = DD_.latent;
  const size_t nn = (size_t)n * n;
  const int steps = (int)M.meta("meta/steps");
  if (M.meta("meta/partial", 0) != 0 || M.meta("meta/inverseFolding", 0) != 0)
    die("rfd4p: partial diffusion and inverse folding are not in this port yet (the export says this job is one)");
  Modality bbM = modalitySchedule("Bb", steps), llM = modalitySchedule("Ll", steps);
  if (bbM.center || llM.center)
    die("rfd4p: center_every_step is not implemented here (the reference and cuda/rfd4p run it)");

  const float* noise = M.hostF("g/noise");
  const float* emitBb = M.hostF("g/emitBb");
  const float* emitLl = M.hostF("g/emitLl");
  const int* coordCond = M.hostI("g/coordCond");
  const int* moleculeIid = M.hostI("g/moleculeIid");
  const int* depths = M.hostI("g/depths");
  const int* stoch = M.hostI("g/stochastic");

  std::vector<float> bb0((size_t)n * 3), ll0((size_t)n * latent);
  size_t noiseAt = initialNoise(noise, bb0.data(), ll0.data(), n, latent, emitBb, emitLl, coordCond, moleculeIid,
                               (float)M.meta("meta/centeringScale", 0.2));
  float* xBb = uploadNew(bb0.data(), bb0.size());
  float* xLl = uploadNew(ll0.data(), ll0.size());
  float* eBb = uploadNew(emitBb, (size_t)n);
  float* eLl = uploadNew(emitLl, (size_t)n);

  // the features: the sequence block with its 35 moving columns, and the pair block with its 30
  const int seqDim = D_.seqDim, condDim = D_.condDim, pairDim = D_.pairDim;
  float* seqF = allocT<float>((size_t)n * seqDim);
  copy(seqF, M.f("in/seq"), (size_t)n * seqDim * 4);
  float* pairF = allocT<float>(nn * pairDim);
  copy(pairF, M.f("g/pairNoDist"), nn * pairDim * 4);

  if (GATE_CHECK) {
    // 🔴 THE LOOP'S INPUT BEFORE ITS OUTPUT: the replayed noise against the draw the reference made, so that
    // "the port's noise is wrong", "the port's features are wrong" and "the port's trunk is wrong" are three
    // answers rather than one number.
    GATE_CHECK("initial noise bb_ca", "gen/x0Bb", xBb, (size_t)n * 3, GATE_TRAJ_BOUND);
    GATE_CHECK("initial noise latents", "gen/x0Ll", xLl, (size_t)n * latent, GATE_TRAJ_BOUND);
  }

  float* s = allocT<float>((size_t)n * D_.D);
  float* z = allocT<float>(nn * D_.P);
  float* vBb = allocT<float>((size_t)n * 3);
  float* vLl = allocT<float>((size_t)n * latent);
  float* logits = allocT<float>((size_t)n * 147);
  float* insertion = allocT<float>(nn);
  float* x1Bb = allocT<float>((size_t)n * 3);
  float* x1Ll = allocT<float>((size_t)n * latent);

  for (int st = 0; st < steps; ++st) {
    // the step's own noised state into the sequence features - three columns at 0 and thirty-two at 3
    copy2d(seqF, (size_t)seqDim * 4, xBb, 3 * 4, 3 * 4, (size_t)n);
    copy2d(seqF + 3, (size_t)seqDim * 4, xLl, (size_t)latent * 4, (size_t)latent * 4, (size_t)n);
    // and the pair track's distance bins from the same coordinates
    RfdPairDistArgs pd{};
    pd.caNm = xBb; pd.pair = pairF; pd.rows = nn; pd.n = (uint)n;
    pd.dim = (uint)M.meta("meta/distDim", 30); pd.off = 0; pd.stride = (uint)pairDim;
    pd.dmin = (float)M.meta("meta/distMin", 0.1); pd.dmax = (float)M.meta("meta/distMax", 3);
    run1d("rfd4p_pair_dists", nn, pd);

    if (GATE_CHECK && st == 0) {
      GATE_CHECK("step 0 seq features", "gen/f0Seq", seqF, (size_t)n * seqDim, GATE_TRAJ_BOUND);
      GATE_CHECK("step 0 pair features", "gen/f0Pair", pairF, nn * pairDim, GATE_TRAJ_BOUND);
    }

    // 🔴 THE STATE GOES IN ZEROED EVERY STEP, which is the recycle's contract and not a reset for tidiness:
    // RMSNorm(0) is 0, so the first cycle's input is the init product. A step that inherited the step before
    // it would be a different network.
    fill(s, 0, (size_t)n * D_.D * 4);
    fill(z, 0, nn * D_.P * 4);
    sync();
    trunkForward(seqF, M.f("g/cond") + (size_t)st * n * condDim, M.f("g/cond1") + (size_t)st * n * condDim,
                 pairF, M.f("g/pairCond") + (size_t)st * 512, depths[st], s, z, vBb, vLl, logits,
                 st == steps - 1 ? insertion : nullptr);

    // one Euler(-Maruyama) step a modality, the clean estimate out beside it
    for (int dm = 0; dm < 2; ++dm) {
      const Modality& m = dm == 0 ? bbM : llM;
      const int dim = dm == 0 ? 3 : latent;
      const bool stochastic = stoch[st * 2 + dm] != 0;
      const double t = (float)m.ts[(size_t)st], dt = m.ts[(size_t)st + 1] - m.ts[(size_t)st];
      // r, and (1 - r) / t without ever forming 1 - r by subtraction: with A = snr sigma^2,
      // r = (A + 1) / (A / k + 1), so 1 - r = A (1 - k) / k / (A / k + 1). At t = 1e-3 that is 1e-6-ish and
      // exact, where 1 - r computed from r is 1 minus a number that rounds to 1.
      const double A = (t * t) / ((1 - t) * (1 - t)) * m.tsrSigma * m.tsrSigma;
      const double r = m.mode == 0 ? (A + 1) / (A / m.tsrK + 1) : 1.5;
      const double oneMinusR = m.mode == 0 ? A * (1 - m.tsrK) / m.tsrK / (A / m.tsrK + 1) : 1 - 1.5;
      const double vsCoef = t != 0 ? oneMinusR / t : 0.0;
      RfdStepArgs a{};
      a.x = dm == 0 ? xBb : xLl;
      a.x1 = dm == 0 ? x1Bb : x1Ll;
      a.v = dm == 0 ? vBb : vLl;
      a.emit = dm == 0 ? eBb : eLl;
      a.n = (size_t)n * dim;
      a.dim = (uint)dim;
      a.mode = (uint)m.mode;
      a.t = (float)t;
      a.dt = (float)dt;
      a.gt = (float)m.gt[(size_t)st];
      a.std = stochastic ? (float)std::sqrt(2 * m.gt[(size_t)st] * m.scn * dt) : 0.f;
      a.r = (float)r;
      a.vsCoef = (float)vsCoef;
      a.stochastic = stochastic ? 1u : 0u;
      if (stochastic) {
        a.eps = uploadNew(noise + noiseAt, (size_t)n * dim);
        noiseAt += (size_t)n * dim;
      }
      run1d("rfd4p_sim_step", a.n, a);
    }
    // 🔴 EVERY STEP'S xt AND x1, NOT THE LAST STATE. A trajectory comparison that sees the end alone cannot say
    // which step went wrong - and the backbone's first step divides by t = 1e-3, so a difference the trunk makes
    // in the fifth digit arrives at the end a thousand times larger. The per-step record is what separates "the
    // loop is wired wrong" from "the loop amplifies", which is the export's own reasoning for writing it.
    if (GATE_CHECK && M.has("gen/s" + std::to_string(st) + "Bb")) {
      const std::string tag = std::to_string(st);
      const float* ours[4] = {xBb, xLl, x1Bb, x1Ll};
      const char* keys[4] = {"Bb", "Ll", "X1Bb", "X1Ll"};
      const char* names[4] = {"xt bb_ca", "xt latents", "x1 bb_ca", "x1 latents"};
      const size_t counts[4] = {(size_t)n * 3, (size_t)n * latent, (size_t)n * 3, (size_t)n * latent};
      for (int k = 0; k < 4; ++k)
        GATE_CHECK(("step " + tag + " " + names[k]).c_str(), "gen/s" + tag + keys[k], ours[k], counts[k],
                   GATE_TRAJ_BOUND);
    }
  }
  if (GATE_CHECK) {
    GATE_CHECK("final xt bb_ca", "gen/xBb", xBb, (size_t)n * 3, GATE_TRAJ_BOUND);
    GATE_CHECK("final xt latents", "gen/xLl", xLl, (size_t)n * latent, GATE_TRAJ_BOUND);
  }

  // ---- the structure -------------------------------------------------------------------------------------
  // 🔴 DECODED FROM `x`, NOT FROM THE LAST x1. At the end of the trajectory t is 1 and the two agree, and the
  // reference decodes x - taking x1 instead would be decoding the second-to-last step's estimate.
  if (!outPath.empty()) {
    sync();
    const int* tokenType = M.hostI("in/tokenType");
    const int* condSeq = M.hostI("g/condSeq");
    std::vector<int> seq((size_t)n);
    const int filled = sequenceForDecoder(n, (const float*)host(logits), condSeq, tokenType, emitLl, seq.data());

    std::vector<int> attn;
    const float* pad = M.hostF("in/mask");
    for (int i = 0; i < n; ++i) if (pad[i] != 0 && emitBb[i] != 0) attn.push_back(i);
    const int na = (int)attn.size();
    int* attnIdx = uploadNew(attn.data(), attn.size());
    int* seqDev = uploadNew(seq.data(), seq.size());

    float* feats = allocT<float>((size_t)n * DD_.feat);
    decoderFeatures((const float*)host(xBb), (const float*)host(xLl), tokenType, (const int*)host(seqDev), pad, feats);
    float* sd = allocT<float>((size_t)na * DD_.D);
    float* dLogits = allocT<float>((size_t)na * DD_.types);
    float* dCoors = allocT<float>((size_t)na * 111);
    float* dPresence = allocT<float>((size_t)na * 37);
    decode(feats, attnIdx, na, sd, dLogits, dCoors, dPresence);
    sync();

    std::vector<float> coors((size_t)n * 111, 0.f);
    std::vector<uint8_t> atomMask((size_t)n * 37, 0);
    std::vector<int> resType((size_t)n, -1);
    decoderPostprocess(n, (const float*)host(xBb), seq.data(), (const float*)host(dLogits),
                       (const float*)host(dCoors), (const float*)host(dPresence), emitBb,
                       M.has("g/coordGroup") ? M.hostI("g/coordGroup") : nullptr, attn.data(), na, coors.data(),
                       atomMask.data(), resType.data());
    double ca[3] = {0, 0, 0};
    const int atoms = writePdb(outPath, n, coors.data(), atomMask.data(), resType.data(),
                               M.has("g/outResId") ? M.hostI("g/outResId") : nullptr,
                               M.has("g/outChain") ? M.hostI("g/outChain") : nullptr,
                               M.has("g/chainIid") ? M.hostI("g/chainIid") : nullptr, ca);
    int emitted = 0;
    for (int i = 0; i < n; ++i) if (emitBb[i] != 0) emitted += 1;
    printf("generated %d residues, %d atoms over %d steps (%d sequence positions read out) -> %s\n", emitted, atoms,
           steps, filled, outPath.c_str());
    printf("  consecutive CA-CA: median %.2f A, min %.2f, max %.2f\n", ca[0], ca[1], ca[2]);
  }
  return steps;
}
