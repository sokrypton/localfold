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
// 🔴 THE SCHEDULES AND THE RANDOM SOURCE COME FROM cuda/rfd4p's OWN HOST HEADER, not from a second transcription
// of flow.js. `cuda/rfd4p/src/schedule.h` is plain C++ - no CUDA in it - and carries the time schedules, the
// noise schedule, the tsr ratio, the stochastic-step rule, and an xorshift128+ whose Box-Muller goes through
// V8's own fdlibm (`cuda/featurise/jsmath.h`): bit-exact against the page, where glibc's log differs on 2.6% of
// arguments and one ulp in one draw is everything in a sixty-step trajectory. Two ports reading one header
// cannot drift apart; two transcriptions of one specification can, and the first draft of this file was one.
#include "rfd4p/src/schedule.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <set>
#include <vector>

namespace {

// Python's `round`, half to even (`pyRound`): the expanded step count goes through it, and
// `round(2.5)` being 2 rather than 3 moves a whole schedule by one step.
int pyRound(double x) {
  const double fl = std::floor(x);
  const double diff = x - fl;
  if (std::abs(diff - 0.5) < 1e-12) return (int)(std::fmod(fl, 2.0) == 0 ? fl : fl + 1);
  return (int)std::round(x);
}

// 🔴 THE FEATURE'S OFFSET IS FOUND, NOT ASSUMED, AND NOT TRANSCRIBED EITHER. `coord_group_seq` is 37 of the 897
// sequence columns and nothing in the export says where: the featuriser writes `meta/chirOff` for the chirality
// and nothing for this one. Its place is a function of `config.nn.feats_seq` and of every earlier feature's
// width - a table this port could copy and then watch go stale, which is what `cuda/featurise/tables.inc` exists
// to avoid. So the block is LOCATED instead: the labels the featuriser wrote at step 0 are reconstructible from
// `g/coordGroup` and the first row of `g/crdPerm`, and the 37 columns that hold them are searched for. A match
// over every token and slot is the offset AND the proof that this port and the featuriser agree about the
// feature; no match is a refusal, because a guessed offset would overwrite some other feature with group labels.
int coordGroupOffset(int n, int seqDim, const float* seq, const int* groups, const std::vector<int>& uniq,
                     const int* perm0) {
  std::vector<float> want((size_t)n * 37);
  for (size_t k = 0; k < want.size(); ++k) {
    const int g = groups[k];
    if (g == -1) { want[k] = -1.f; continue; }
    const auto at = std::find(uniq.begin(), uniq.end(), g);
    want[k] = at == uniq.end() ? -1.f : (float)perm0[at - uniq.begin()];
  }
  for (int off = 0; off + 37 <= seqDim; ++off) {
    bool all = true;
    for (int i = 0; i < n && all; ++i)
      for (int a = 0; a < 37; ++a)
        if (seq[(size_t)i * seqDim + off + a] != want[(size_t)i * 37 + a]) { all = false; break; }
    if (all) return off;
  }
  return -1;
}

}  // namespace

bool stochasticStep(double t, const Modality& m) {
  rfd4p::StepCfg c;
  c.mode = m.mode == 0 ? "vf_tsr" : (m.mode == 1 ? "vf" : "sc");
  c.tLimOde = m.odeLim;
  c.tLimOdeBelow = m.odeBelow;
  return rfd4p::isStochastic(t, c);
}

// the exported sampling metadata (`samplingMeta`'s fields) as cuda/rfd4p's own config struct
rfd4p::ModalityCfg modalityCfg(const char* tag) {
  auto meta = [&](const char* field, double fallback) {
    return M.meta(std::string("meta/samp") + tag + field, fallback);
  };
  static const char* SCHED[] = {"uniform", "power", "log"};
  static const char* GT[] = {"1/t", "1-t/t", "tan", "ode"};
  static const char* MODE[] = {"vf_tsr", "vf", "sc"};
  const int sched = (int)meta("Sched", 2), gt = (int)meta("Gt", 0), mode = (int)meta("Mode", 0);
  if (sched < 0 || sched > 2 || gt < 0 || gt > 3 || mode < 0 || mode > 2)
    die("rfd4p: this job's %s sampler is schedule %d, gt %d, mode %d - one of them is not a value this port "
        "knows", tag, sched, gt, mode);
  rfd4p::ModalityCfg c;
  c.schedule = {SCHED[sched], meta("P", 2), (int)meta("Tail", 0), (int)meta("TailMult", 1)};
  c.gt = {GT[gt], meta("GtP", 1), meta("HasClamp", 0) != 0, meta("Clamp", 0)};
  c.step.mode = MODE[mode];
  c.step.scScaleNoise = meta("Noise", 1);
  c.step.tLimOde = meta("Ode", 1);
  c.step.tLimOdeBelow = meta("OdeBelow", 0);
  c.step.tsrK = meta("TsrK", 1);
  c.step.tsrSigma = meta("TsrSigma", 1);
  c.step.center = meta("Center", 0) != 0;
  return c;
}

Modality modalitySchedule(const char* tag, int nsteps) {
  const rfd4p::ModalityCfg c = modalityCfg(tag);
  const rfd4p::Schedule sch = rfd4p::buildSchedule(c, nsteps);
  Modality m;
  m.ts = sch.ts;
  m.gt = sch.gt;
  m.mode = c.step.mode == "vf_tsr" ? 0 : (c.step.mode == "vf" ? 1 : 2);
  m.tsrK = c.step.tsrK;
  m.tsrSigma = c.step.tsrSigma;
  m.scn = c.step.scScaleNoise;
  m.odeLim = c.step.tLimOde;
  m.odeBelow = c.step.tLimOdeBelow;
  m.center = c.step.center;
  return m;
}

// `buildPartialSchedule`: a partial run does not run its own schedule, it runs the TAIL of the full one - so the
// times it steps through are the same times an unconditional run would reach, and the corruption time has to be
// one of them. Two cases: with a step FRACTION the tail of the base schedule; otherwise a schedule expanded so
// that `nsteps` of its tail start at the corruption time.
Modality partialSchedule(const char* tag, int nsteps, int fullNsteps, double pt) {
  Modality m = modalitySchedule(tag, nsteps);        // the step parameters; the times are rebuilt below
  auto tailOf = [&](int steps) {
    const Modality full = modalitySchedule(tag, steps);
    Modality out = m;
    out.ts.assign(full.ts.end() - (nsteps + 1), full.ts.end());
    out.gt.assign(full.gt.end() - nsteps, full.gt.end());
    return out;
  };
  if (fullNsteps > 0) {
    if (!(nsteps >= 1 && nsteps <= fullNsteps))
      die("rfd4p: a partial step count must be in [1, %d], got %d", fullNsteps, nsteps);
    Modality out = tailOf(fullNsteps);
    if (std::abs((float)out.ts[0] - (float)pt) > 1e-6)
      die("rfd4p: %s's corruption time %g is not the first time of the full schedule's tail (%g)", tag, pt,
          out.ts[0]);
    return out;
  }
  const int dense = std::max(nsteps * 100, 1000);
  const Modality d = modalitySchedule(tag, dense);
  int iStart = dense;
  for (int i = 0; i <= dense; ++i)
    if ((float)d.ts[(size_t)i] >= pt - 1e-6) { iStart = i; break; }
  const int nAbove = std::max(dense - iStart, 1);
  return tailOf(pyRound((double)nsteps * dense / nAbove));
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

// `zeroComBbNoise`: n*3 normals zero-centred over the emitted tokens
size_t noisedBackbone(const float* noise, int n, const float* emitBb, float* out) {
  int cnt = 0;
  float mean[3] = {0, 0, 0};
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) out[i * 3 + k] = emitBb[i] != 0 ? noise[(size_t)i * 3 + k] : 0.f;
    if (emitBb[i] == 0) continue;
    cnt += 1;
    for (int k = 0; k < 3; ++k) mean[k] += out[i * 3 + k];
  }
  for (int k = 0; k < 3; ++k) mean[k] = (float)(mean[k] / std::max(cnt, 1));
  for (int i = 0; i < n; ++i)
    if (emitBb[i] != 0)
      for (int k = 0; k < 3; ++k) out[i * 3 + k] = (float)(out[i * 3 + k] - mean[k]);
  return (size_t)n * 3;
}

// `applyPerGroupOffset`: one N(0, scale^2) offset a molecule instance, for tokens with no absolute coordinate
// conditioning, in ascending instance order - which is the draw order and so part of the specification
size_t groupOffsets(const float* noise, int n, const float* emitBb, const int* coordCond, const int* moleculeIid,
                    float scale, float* out) {
  std::set<int> groups;
  for (int i = 0; i < n; ++i) if (emitBb[i] != 0 && !coordCond[i]) groups.insert(moleculeIid[i]);
  size_t at = 0;
  for (int g : groups) {
    const float off[3] = {noise[at], noise[at + 1], noise[at + 2]};
    at += 3;
    for (int i = 0; i < n; ++i) {
      if (!(emitBb[i] != 0 && !coordCond[i]) || moleculeIid[i] != g) continue;
      for (int k = 0; k < 3; ++k) out[i * 3 + k] = (float)(out[i * 3 + k] + (float)(scale * off[k]));
    }
  }
  for (int i = 0; i < n; ++i)
    if (emitBb[i] == 0)
      for (int k = 0; k < 3; ++k) out[i * 3 + k] = 0.f;
  return at;
}

// 🔴 A SECOND DESIGN DRAWS ITS OWN STREAM, IN THE EXPORTER'S ORDER. The first design replays what the export
// recorded, so a trajectory gate compares two sides that drew the same numbers; every later one is seeded
// `seed + design` and draws for itself - which is only safe because `rfd4p::Rng` is bit-exact against the page,
// integers and Gaussians alike. The ORDER is the specification: the per-step `randperm` goes BEFORE that step's
// noise and consumes RAW uniforms where everything else goes through Box-Muller, so modelling it as a normal
// advances the stream by a different amount and moves every later draw of the run.
struct Stream {
  std::vector<float> noise;
  std::vector<int> perm;      // [steps][groups]
};
Stream drawStream(uint64_t seed, int n, int latent, int steps, int groups, int crdGroups, bool partial,
                  bool inverse, const Modality& bb, const Modality& ll) {
  rfd4p::Rng rng(seed);
  Stream out;
  auto take = [&](size_t count) {
    const size_t at = out.noise.size();
    out.noise.resize(at + count);
    rng.randn(out.noise.data() + at, count);
  };
  if (partial) {
    take((size_t)n * latent);                       // the encoder's reparameterisation
    take((size_t)n * 3);                            // corrupt_batch: the zero-COM backbone noise
    take((size_t)n * latent);                       // corrupt_batch: the latent noise
    for (int g = 0; g < groups; ++g) take(3);
    if (inverse) {
      take((size_t)n * 3);                          // the target backbone at ts[0]
      for (int g = 0; g < groups; ++g) take(3);
    }
  } else {
    take((size_t)n * 3);
    for (int g = 0; g < groups; ++g) take(3);
    take((size_t)n * latent);
  }
  for (int st = 0; st < steps; ++st) {
    const std::vector<int> p = rng.randperm(crdGroups);
    out.perm.insert(out.perm.end(), p.begin(), p.end());
    for (int dm = 0; dm < 2; ++dm) {
      const Modality& m = dm == 0 ? bb : ll;
      if (stochasticStep((float)m.ts[(size_t)st], m)) take((size_t)n * (dm == 0 ? 3 : latent));
    }
    if (inverse && st < steps - 1) {
      take((size_t)n * 3);
      for (int g = 0; g < groups; ++g) take(3);
    }
  }
  return out;
}

}  // namespace

double (*GATE_CHECK)(const char*, const std::string&, const float*, size_t, double) = nullptr;
double GATE_TRAJ_BOUND = 1e-4;
double GATE_TRAJ_XT_BOUND = 1e-4;

int generate(const std::string& genDir, const std::string& outPath, int designs, uint64_t seed) {
  (void)genDir;
  const int n = D_.n, latent = DD_.latent;
  const size_t nn = (size_t)n * n;
  const int steps = (int)M.meta("meta/steps");
  const bool partial = M.meta("meta/partial", 0) != 0;
  const bool inverse = M.meta("meta/inverseFolding", 0) != 0;
  const double ptBb = M.meta("meta/partialTBb", 0), ptLl = M.meta("meta/partialTLl", 0);
  const int fullNsteps = (int)M.meta("meta/fullNsteps", 0);
  // the molecule instances that take a stochastic offset: how many 3-normal draws the stream holds for them
  const int nGroups = (int)M.meta("meta/nGroups", 0);
  // 🔴 A PER-STEP FEATURE IS RECOMPUTED OR REPLAYED AND NEVER SILENTLY FROZEN, which is the export's own rule
  // and the reason these are refusals rather than omissions. The chirality alignment fits a frame against the
  // noisy alpha carbons every step and rotates the clean vectors (`cuda/rfd4p/src/chirality.cuh`), and a
  // coordinate group's LABELS are permuted from a replayed `randperm` at the top of every step. Freezing either
  // is a plausible structure generated from a feature the model was supposed to see move - so a job that needs
  // one stops here by name instead.
  const bool chiral = M.meta("meta/alignChirality", 0) != 0;
  const int chirOff = (int)M.meta("meta/chirOff", -1);
  const int frameWidth = (int)M.meta("meta/chiralFrameWidth", 0);
  if (chiral && (chirOff < 0 || frameWidth <= 2 || !M.has("g/chir") || !M.has("g/chiralFrames")))
    die("rfd4p: this job aligns chirality but its export carries no frames to align with");
  // 🔴 THE COORDINATE GROUPS' LABELS ARE PERMUTED EVERY STEP, from the replayed `randperm` the export carries -
  // `g/crdUniq` and `g/crdPerm`, written for a consumer that until now existed in neither port (cuda/rfd4p
  // freezes them silently, which is a note for upstream rather than a thing to copy). One group makes the
  // permutation the identity; two or more make it the difference between the feature the model was trained on
  // and a constant.
  const int nCrdGroups = (int)M.meta("meta/nCrdGroups", 0);
  int grpOff = -1;
  std::vector<int> crdUniq, crdPerm;
  if (nCrdGroups > 0) {
    if (!M.has("g/crdUniq") || !M.has("g/crdPerm") || !M.has("g/coordGroup"))
      die("rfd4p: this job has %d coordinate groups and its export carries no permutation for them", nCrdGroups);
    crdUniq.assign(M.hostI("g/crdUniq"), M.hostI("g/crdUniq") + nCrdGroups);
    crdPerm.assign(M.hostI("g/crdPerm"), M.hostI("g/crdPerm") + (size_t)nCrdGroups * steps);
    grpOff = coordGroupOffset(n, D_.seqDim, M.hostF("in/seq"), M.hostI("g/coordGroup"), crdUniq, crdPerm.data());
    if (grpOff < 0)
      die("rfd4p: the 37 coord_group_seq columns are not in this input's %d sequence columns under the first "
          "step's permutation - the feature's place cannot be assumed and this port will not guess it",
          D_.seqDim);
  }
  // 🔴 A PARTIAL RUN RUNS THE TAIL OF THE FULL SCHEDULE, NOT ITS OWN: its first time is the corruption time and
  // every later one is a time an unconditional run would also reach.
  Modality bbM = partial ? partialSchedule("Bb", steps, fullNsteps, ptBb) : modalitySchedule("Bb", steps);
  Modality llM = partial ? partialSchedule("Ll", steps, fullNsteps, ptLl) : modalitySchedule("Ll", steps);


  const float* replayed = M.hostF("g/noise");
  const float* emitBb = M.hostF("g/emitBb");
  const float* emitLl = M.hostF("g/emitLl");
  const int* coordCond = M.hostI("g/coordCond");
  const int* moleculeIid = M.hostI("g/moleculeIid");
  const int* depths = M.hostI("g/depths");
  const int* stoch = M.hostI("g/stochastic");

  const float scale = (float)M.meta("meta/centeringScale", 0.2);

  // the activations and the features, allocated ONCE: a campaign's second design pays for neither, which is
  // what makes it a campaign rather than N runs (cuda/rfd4p measures 0.43 s for the first and 0.27 after)
  float* seqF = allocT<float>((size_t)n * D_.seqDim);
  float* pairF = allocT<float>(nn * D_.pairDim);
  float* sAct = allocT<float>((size_t)n * D_.D);
  float* zAct = allocT<float>(nn * D_.P);
  float* vBb = allocT<float>((size_t)n * 3);
  float* vLl = allocT<float>((size_t)n * latent);
  float* logits = allocT<float>((size_t)n * 147);
  float* insertion = allocT<float>(nn);
  float* x1Bb = allocT<float>((size_t)n * 3);
  float* x1Ll = allocT<float>((size_t)n * latent);
  // 🔴 THE STEP'S CONDITIONING IS COPIED INTO A BUFFER OF ITS OWN, NOT POINTED AT INSIDE THE BIG ONE, AND THAT
  // IS A CORRECTNESS REQUIREMENT RATHER THAN TIDINESS. metal/core's GEMM chooses its VECTOR EPILOGUE by the
  // operands' ADDRESS alignment (`al4(p, ld) = p % 16 == 0 && ld % 4 == 0`, and the bias 16-byte aligned), so
  // which kernel runs - and therefore the order the sums are stored in - depends on where a buffer happens to
  // live. `g/cond + step * n * 609 * 4 bytes` is 8 modulo 16 on odd steps and whatever the tensor's base is
  // modulo 16 on top of that, and Metal assigns that base per run: so the same input folded twice took
  // different kernels from step 3 on and the designs differed in the third decimal. It showed only once the
  // featuriser ran first in the same process, which shifts every later allocation - a difference between runs
  // that no gate over one run could see. An allocation of its own is 256-byte aligned by construction.
  float* cond0Buf = allocT<float>((size_t)n * D_.condDim);
  float* cond1Buf = allocT<float>((size_t)n * D_.condDim);
  float* pairCondBuf = allocT<float>((size_t)M.len("g/pairCond") / std::max(steps, 1));
  float* xBb = allocT<float>((size_t)n * 3);
  float* xLl = allocT<float>((size_t)n * latent);
  float* eBb = uploadNew(emitBb, (size_t)n);
  float* eLl = uploadNew(emitLl, (size_t)n);

  for (int design = 0; design < std::max(designs, 1); ++design) {
  // 🔴 DESIGN 0 REPLAYS THE EXPORT'S STREAM AND EVERY LATER ONE DRAWS ITS OWN. The replay is what lets a
  // trajectory be compared against the reference's; a campaign's point is different proteins, so design d is
  // seeded `seed + d` and draws in the exporter's own order.
  Stream own;
  const float* noise = replayed;
  const int* perm = crdPerm.empty() ? nullptr : crdPerm.data();
  if (design > 0) {
    own = drawStream(seed + (uint64_t)design, n, latent, steps, nGroups, nCrdGroups, partial, inverse, bbM, llM);
    noise = own.noise.data();
    perm = own.perm.empty() ? nullptr : own.perm.data();
  }
  std::vector<float> bb0((size_t)n * 3), ll0((size_t)n * latent), target;
  size_t noiseAt = 0;
  if (!partial) {
    noiseAt = initialNoise(noise, bb0.data(), ll0.data(), n, latent, emitBb, emitLl, coordCond, moleculeIid, scale);
  } else {
    // 🔴 A PARTIAL RUN STARTS FROM THE INPUT, CORRUPTED - `corrupt_batch`, whose DRAW ORDER is not
    // `sample_noise`'s: the encoder's reparameterisation, then the backbone's noise, then the latents', then the
    // per-molecule offsets, where an unconditional run draws the backbone, the offsets, then the latents. One
    // transposition shifts every later draw of the whole run, which is invisible in any one stage.
    if (!M.has("e/feats") && !M.has("e/coorsNm"))
      die("rfd4p: this job is partial diffusion and needs its encoder input (--encoder=<work>-enc)");
    checkEncoderDims();
    // the clean latents: the encoder's mean and log scale, reparameterised with this run's own eps
    std::vector<int> attn, emitIdx, within;
    const float* pad = M.hostF("in/mask");
    for (int i = 0; i < n; ++i) if (M.hostF("in/attnMask")[i] != 0) attn.push_back(i);
    {
      int k = 0;
      for (int i = 0; i < n; ++i) {
        if (M.hostF("in/attnMask")[i] == 0) continue;
        if (emitLl[i] != 0) { emitIdx.push_back(i); within.push_back(k); }
        ++k;
      }
    }
    const int na = (int)attn.size();
    int* attnDev = uploadNew(attn.data(), attn.size());
    float* ef = allocT<float>((size_t)n * ED_.feat);
    encoderFeatures(n, M.hostF("e/coorsNm"), M.hostF("e/atomMask"), M.hostF("e/chainIid"), M.hostF("e/bondCounts"),
                    M.hostI("e/element"), M.hostI("e/charge"), ef);
    float* meanDev = allocT<float>((size_t)na * latent);
    float* lsDev = allocT<float>((size_t)na * latent);
    encode(ef, attnDev, na, meanDev, lsDev);
    sync();
    const float* mean = (const float*)host(meanDev);
    const float* ls = (const float*)host(lsDev);
    std::vector<float> x1Ll((size_t)n * latent, 0.f);
    for (size_t e = 0; e < emitIdx.size(); ++e) {
      const int tok = emitIdx[e], row = within[e];
      for (int k = 0; k < latent; ++k) {
        const float m = mean[(size_t)row * latent + k], l = ls[(size_t)row * latent + k];
        x1Ll[(size_t)tok * latent + k] = (float)(m + (float)(noise[noiseAt + (size_t)tok * latent + k] * (float)std::exp(l)));
      }
    }
    noiseAt += (size_t)n * latent;

    // the clean backbone is the input's own alpha carbons, emit-masked
    const float* caNm = M.hostF("in/caNm");
    std::vector<float> x1Bb((size_t)n * 3, 0.f);
    for (int i = 0; i < n; ++i)
      if (emitBb[i] != 0)
        for (int k = 0; k < 3; ++k) x1Bb[(size_t)i * 3 + k] = caNm[i * 3 + k];

    // and the corruption: (1 - t) x0 + t x1, the backbone's x0 offset per molecule first
    std::vector<float> x0Bb((size_t)n * 3), x0Ll((size_t)n * latent);
    noiseAt += noisedBackbone(noise + noiseAt, n, emitBb, x0Bb.data());
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < latent; ++k)
        x0Ll[(size_t)i * latent + k] = emitLl[i] != 0 ? noise[noiseAt + (size_t)i * latent + k] : 0.f;
    noiseAt += (size_t)n * latent;
    noiseAt += groupOffsets(noise + noiseAt, n, emitBb, coordCond, moleculeIid, scale, x0Bb.data());
    const float tB = (float)ptBb, tL = (float)ptLl;
    for (int i = 0; i < n; ++i) {
      if (emitBb[i] != 0)
        for (int k = 0; k < 3; ++k)
          bb0[(size_t)i * 3 + k] = (float)((float)((1.f - tB) * x0Bb[(size_t)i * 3 + k]) +
                                           (float)(tB * x1Bb[(size_t)i * 3 + k]));
      if (emitLl[i] != 0)
        for (int k = 0; k < latent; ++k)
          ll0[(size_t)i * latent + k] = (float)((float)((1.f - tL) * x0Ll[(size_t)i * latent + k]) +
                                                (float)(tL * x1Ll[(size_t)i * latent + k]));
    }
    // 🔴 INVERSE FOLDING KEEPS THE CLEAN BACKBONE AND RE-NOISES IT EVERY STEP (the resampled forward marginal):
    // the structure is given and only the sequence is designed, so the backbone is never integrated - it is
    // drawn afresh at each step's time and handed back clean at the last.
    if (inverse) {
      target = x1Bb;
      std::vector<float> xb((size_t)n * 3);
      noiseAt += noisedBackbone(noise + noiseAt, n, emitBb, xb.data());
      noiseAt += groupOffsets(noise + noiseAt, n, emitBb, coordCond, moleculeIid, scale, xb.data());
      const float t0 = (float)bbM.ts[0];
      for (int i = 0; i < n; ++i)
        if (emitBb[i] != 0)
          for (int k = 0; k < 3; ++k)
            bb0[(size_t)i * 3 + k] = (float)((float)((1.f - t0) * xb[(size_t)i * 3 + k]) +
                                             (float)(t0 * target[(size_t)i * 3 + k]));
    }
    (void)pad;
  }
  upload(xBb, bb0.data(), bb0.size() * 4);
  upload(xLl, ll0.data(), ll0.size() * 4);
  // the whole replayed stream on the device once, so no step issues an upload of its own
  const size_t noiseLen = design > 0 ? own.noise.size() : M.len("g/noise");
  float* noiseDev = uploadNew(noise, noiseLen);

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

  // LOCALFOLD_PROFILE=1: every labelled dispatch in its own command buffer, its GPU time summed by label - an
  // upper bound, for proportions. One step is profiled rather than all sixty: they are the same work.
  const int profileStep = profiling() ? (int)(getenv("LOCALFOLD_RFD4P_PROFILE_STEP")
                                                  ? atoi(getenv("LOCALFOLD_RFD4P_PROFILE_STEP"))
                                                  : steps - 1)
                                      : -1;
  for (int st = 0; st < steps; ++st) {
    if (st == profileStep) profileStart();
    // the step's own noised state into the sequence features - three columns at 0 and thirty-two at 3
    copy2d(seqF, (size_t)seqDim * 4, xBb, 3 * 4, 3 * 4, (size_t)n);
    copy2d(seqF + 3, (size_t)seqDim * 4, xLl, (size_t)latent * 4, (size_t)latent * 4, (size_t)n);
    // this step's coordinate-group labels, from the step's own row of the replayed permutation
    if (grpOff >= 0) {
      std::vector<float> rows((size_t)n * 37);
      const int* p = perm + (size_t)st * nCrdGroups;
      const int* groups = M.hostI("g/coordGroup");
      for (size_t k = 0; k < rows.size(); ++k) {
        const int g = groups[k];
        const auto at = std::find(crdUniq.begin(), crdUniq.end(), g);
        rows[k] = (g == -1 || at == crdUniq.end()) ? -1.f : (float)p[at - crdUniq.begin()];
      }
      float* dev = uploadNew(rows.data(), rows.size());
      copy2d(seqF + grpOff, (size_t)seqDim * 4, dev, 37 * 4, 37 * 4, (size_t)n);
    }

    // 🔴 AND THE CHIRALITY, FITTED AGAINST THESE COORDINATES. The clean vectors rotated onto the directions
    // their frame's neighbours have now: 18 columns at `chirOff`, recomputed because they describe the state the
    // model is being shown rather than the structure the job came from.
    if (chiral) {
      sync();
      std::vector<float> chirOut(M.len("g/chir"));
      const float* clean = M.hostF("g/chir");
      for (size_t k = 0; k < chirOut.size(); ++k) chirOut[k] = clean[k];
      alignChirality(n, (const float*)host(xBb), clean, M.hostI("g/chiralFrames"), M.hostI("g/chiralOwners"),
                     M.hostF("in/mask"), frameWidth, chirOut.data());
      // into the 18 columns the featuriser put them at, a row at a time
      std::vector<float> rows((size_t)n * 18);
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < 18; ++k) rows[(size_t)i * 18 + k] = chirOut[(size_t)i * 18 + k];
      float* dev = uploadNew(rows.data(), rows.size());
      copy2d(seqF + chirOff, (size_t)seqDim * 4, dev, 18 * 4, 18 * 4, (size_t)n);
    }
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
    const size_t pairCondDim = M.len("g/pairCond") / (size_t)std::max(steps, 1);
    copy(cond0Buf, M.f("g/cond") + (size_t)st * n * condDim, (size_t)n * condDim * 4);
    copy(cond1Buf, M.f("g/cond1") + (size_t)st * n * condDim, (size_t)n * condDim * 4);
    copy(pairCondBuf, M.f("g/pairCond") + (size_t)st * pairCondDim, pairCondDim * 4);
    trunkForward(seqF, cond0Buf, cond1Buf, pairF, pairCondBuf, depths[st], s, z, vBb, vLl, logits,
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
        // 🔴 A VIEW INTO THE WHOLE STREAM, UPLOADED ONCE - NOT AN UPLOAD A STEP. A per-step `uploadNew` was a
        // host-to-device copy issued while sixty steps of work were queued, and it made the fold
        // NONDETERMINISTIC: six runs of one input gave four different designs, differing in the third decimal.
        // One upload before the loop has nothing to race with, and the step reads its own slice.
        a.eps = noiseDev + noiseAt;
        noiseAt += (size_t)n * dim;
      }
      run1d("rfd4p_sim_step", a.n, a);
      // 🔴 AND THE RE-CENTRING AFTER IT, NOT INSIDE IT: `_force_zero_com` is a reduction over every emitted
      // token, which a per-element step cannot see. The released sampler leaves it off and a job may ask for it.
      if (m.center) {
        RfdCenterArgs ca2{};
        ca2.x = a.x; ca2.emit = a.emit; ca2.tokens = (size_t)n; ca2.dim = (uint)dim;
        dispatch("rfd4p_center", &ca2, sizeof ca2, Grid{(uint32_t)dim, 1, 1}, 256, 0, "rfd4p_center");
      }
    }
    if (st == profileStep) profileReport("one sampler step", 24);
    // LOCALFOLD_RFD4P_TRACE=1: a checksum of the state after every step, which is how a divergence between two
    // runs of one input is located - the step it first appears at says whether the difference came in with the
    // features or grew in the trunk.
    if (st == 0 && getenv("LOCALFOLD_RFD4P_TRACE")) {
      // the addresses themselves, because metal/core's GEMM chooses its epilogue by their alignment: a run whose
      // buffers land differently takes different kernels, and this is how that is seen rather than inferred
      printf("  trace addr: seqF %%16=%zu pairF %%16=%zu cond0 %%16=%zu s %%16=%zu z %%16=%zu "
             "in/seq %%16=%zu w/first %%16=%zu\n",
             (size_t)((uintptr_t)seqF % 16), (size_t)((uintptr_t)pairF % 16), (size_t)((uintptr_t)cond0Buf % 16),
             (size_t)((uintptr_t)s % 16), (size_t)((uintptr_t)z % 16),
             (size_t)((uintptr_t)M.f("in/seq") % 16),
             (size_t)((uintptr_t)M.f("w/nn.cond_factory.linear_out.weight") % 16));
    }
    if (getenv("LOCALFOLD_RFD4P_TRACE")) {
      sync();
      const float* xb = (const float*)host(xBb);
      const float* xl = (const float*)host(xLl);
      double a = 0, b = 0;
      for (size_t k = 0; k < (size_t)n * 3; ++k) a += (double)xb[k] * (k + 1);
      for (size_t k = 0; k < (size_t)n * latent; ++k) b += (double)xl[k] * (k + 1);
      printf("  trace step %d: bb %.17g ll %.17g\n", st, a, b);
    }

    // the resampled target comes AFTER both modalities have stepped, and not after the last step, where the
    // clean target is used as it is
    if (inverse) {
      sync();
      std::vector<float> xb((size_t)n * 3);
      float* x1b = (float*)host(x1Bb);
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) x1b[(size_t)i * 3 + k] = target[(size_t)i * 3 + k];
      if (st + 1 == steps) {
        upload(xBb, target.data(), target.size() * 4);
      } else {
        noiseAt += noisedBackbone(noise + noiseAt, n, emitBb, xb.data());
        noiseAt += groupOffsets(noise + noiseAt, n, emitBb, coordCond, moleculeIid, scale, xb.data());
        const float t1 = (float)bbM.ts[(size_t)st + 1];
        std::vector<float> xt((size_t)n * 3, 0.f);
        for (int i = 0; i < n; ++i)
          if (emitBb[i] != 0)
            for (int k = 0; k < 3; ++k)
              xt[(size_t)i * 3 + k] = (float)((float)((1.f - t1) * xb[(size_t)i * 3 + k]) +
                                              (float)(t1 * target[(size_t)i * 3 + k]));
        upload(xBb, xt.data(), xt.size() * 4);
      }
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
      // 🔴 THE REPLAYED INPUTS ARE HELD EXACT IN BOTH ARITHMETICS AND THE TRAJECTORY IS NOT, which is the split
      // that matters: the noise and the features are the same bytes whatever the GEMMs are told to do (measured
      // 4.5e-8 and exactly 0.0), while a step DIVIDES - x1 by nothing but a near-cancelling sum, xt's score by
      // (1 - t), which at the last step of a two-step run is 0.091, the worst-conditioned point a trajectory
      // has. In half those read 4e-3 and 1.5e-2 where float32 reads 1e-5 and 6e-6.
      for (int k = 0; k < 4; ++k)
        GATE_CHECK(("step " + tag + " " + names[k]).c_str(), "gen/s" + tag + keys[k], ours[k], counts[k],
                   GATE_TRAJ_XT_BOUND);
    }
  }
  if (GATE_CHECK) {
    GATE_CHECK("final xt bb_ca", "gen/xBb", xBb, (size_t)n * 3, GATE_TRAJ_XT_BOUND);
    GATE_CHECK("final xt latents", "gen/xLl", xLl, (size_t)n * latent, GATE_TRAJ_XT_BOUND);
  }

  // ---- the structure -------------------------------------------------------------------------------------
  // 🔴 DECODED FROM `x`, NOT FROM THE LAST x1. At the end of the trajectory t is 1 and the two agree, and the
  // reference decodes x - taking x1 instead would be decoding the second-to-last step's estimate.
  if (!outPath.empty()) {
    sync();
    // 🔴 ONE FILE A DESIGN, NUMBERED FROM ONE, and the plain name kept for a single design: a campaign that
    // overwrote its own output would report N designs and leave the last.
    std::string path = outPath;
    if (std::max(designs, 1) > 1) {
      const size_t dot = outPath.find_last_of('.');
      const std::string stem = dot == std::string::npos ? outPath : outPath.substr(0, dot);
      const std::string ext = dot == std::string::npos ? std::string(".pdb") : outPath.substr(dot);
      path = stem + "_" + std::to_string(design + 1) + ext;
    }
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
    OutTables in;
    if (M.has("g/outAtomize") && M.has("g/inName")) {
      in.present = true;
      in.atomize = M.hostI("g/outAtomize");
      in.hetero = M.hostI("g/outHetero");
      in.resName = M.hostI("g/outResName");
      in.inName = M.hostI("g/inName");
      in.inElem = M.hostI("g/inElem");
      in.inCharge = M.hostI("g/inCharge");
      in.insCode = M.hostI("g/outInsCode");
      in.entity = M.hostI("g/outEntity");
      in.inB = M.hostI("g/inB");
      in.inOcc = M.hostI("g/inOcc");
    }
    const int atoms = writeStructure(path, n, coors.data(), atomMask.data(), resType.data(),
                               M.has("g/outResId") ? M.hostI("g/outResId") : nullptr,
                               M.has("g/outChain") ? M.hostI("g/outChain") : nullptr,
                               M.has("g/chainIid") ? M.hostI("g/chainIid") : nullptr, ca, in);
    int emitted = 0;
    for (int i = 0; i < n; ++i) if (emitBb[i] != 0) emitted += 1;
    printf("generated %d residues, %d atoms over %d steps (%d sequence positions read out) -> %s\n", emitted, atoms,
           steps, filled, path.c_str());
    printf("  consecutive CA-CA: median %.2f A, min %.2f, max %.2f\n", ca[0], ca[1], ca[2]);
  }
  }
  return steps;
}
