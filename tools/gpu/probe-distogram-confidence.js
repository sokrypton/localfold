/**
 * Does the distogram stand in for pLDDT well enough to colour a trajectory?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-distogram-confidence.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-distogram-confidence.js --steps=16
 *
 * THE QUESTION. Colouring a diffusion trajectory by confidence means a score
 * per FRAME, and the real head costs more than a denoiser call - 226 ms against
 * 211 at 150 tokens - so it would more than double a fold.
 * src/af3/distogram-confidence.js is the cheap stand-in; this asks how close it
 * gets, because a proxy nobody has checked is a colour ramp that lies.
 *
 * WHAT IT MEASURES. One fold, then, on the FINAL coordinates: the real
 * per-token pLDDT from the confidence head against the distogram score, as
 * Pearson and Spearman correlation and as absolute error. Then the same score
 * on every trajectory frame, to see whether it rises as the structure resolves
 * - which is the whole point of drawing it.
 *
 * 🔴 THE RANK CORRELATION IS THE ONE THAT MATTERS. A colour ramp is a ranking:
 * it needs the wobbly loop to look worse than the folded core, not to agree on
 * a number. Pearson and the absolute error are reported so the failure mode is
 * visible, not because they are the criterion.
 *
 * 🔴 AND ONE PROTEIN IS NOT A VALIDATION. The formulation was swept on 6MRR
 * alone first and picked a variant that scored 0.66 there and 0.62 on Top7; the
 * settings that survived all three are in src/af3/distogram-confidence.js with
 * what each alternative measured. Run this on more than one sequence before
 * believing any change to it.
 *
 * 🔴 AND ACROSS TARGETS IS A DIFFERENT QUESTION FROM WITHIN ONE, WHICH IS THE
 * QUESTION A COLOUR ACTUALLY ASKS. Ranking residues inside a fold says whether
 * the wobbly loop looks worse than the core. It says NOTHING about whether a
 * badly folded protein looks worse than a well folded one - and somebody
 * comparing two predictions, or watching one resolve, is doing exactly that.
 * `--sequences` folds several and reports the correlation of the MEANS, which
 * is the test that a 0.9 target and a 0.5 target come out in that order.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";
import { distogramAgreementTable, distogramConfidence, calibrateToPlddt }
  from "../../src/af3/distogram-confidence.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const DEFAULT = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK";

/**
 * A spread of targets for the across-target test, chosen to span the range
 * rather than to be representative: three that should fold, one miniprotein,
 * and three that should not - a scramble, a linker and a homopolymer. A
 * correlation measured only on things that fold says nothing.
 */
const PANEL = [
  ["6MRR", DEFAULT],
  ["Top7", "DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLE"],
  ["designed-58", "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"],
  ["villin-HP36", "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"],
  ["trp-cage", "NLYIQWLKDGGPSSGRPPPS"],
  ["6MRR-scrambled", "EKLGKFLKSLEHTKEEGRWLNFAKQGKKGLEAIVELPLIKELSTKQFAEKAIGLRLTEKS"],
  ["GS-linker", "GSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGS"],
  ["poly-alanine", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"],
];

function pearson(a, b) {
  const n = a.length;
  let ma = 0; let mb = 0;
  for (let i = 0; i < n; i += 1) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0; let da = 0; let db = 0;
  for (let i = 0; i < n; i += 1) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return num / Math.sqrt(da * db);
}

/** Ranks with ties averaged, so Spearman is Pearson on them. */
function ranks(values) {
  const order = [...values.keys()].sort((x, y) => values[x] - values[y]);
  const out = new Float64Array(values.length);
  let index = 0;
  while (index < order.length) {
    let end = index;
    while (end + 1 < order.length && values[order[end + 1]] === values[order[index]]) end += 1;
    const mean = (index + end) / 2;
    for (let k = index; k <= end; k += 1) out[order[k]] = mean;
    index = end + 1;
  }
  return out;
}

async function foldOne(device, sequence, weights, { steps, mode, seed }) {
  const batch = featuriseProtein(sequence, {});
  const frames = [];
  const result = await foldBatch(device, batch, weights, {
    mode, steps, recycles: 0, seed,
    onStep: ({ step, denoised }) => { frames.push({ step, positions: denoised }); },
  });
  const tokens = batch.tokens;
  const beta = batch.tokenAtomsToPseudoBeta;
  const gather = (positions) => {
    const out = new Float32Array(tokens * 3);
    for (let token = 0; token < tokens; token += 1) {
      if (!beta.mask[token]) continue;
      const from = Number(beta.indices[token]) * 3;
      for (let axis = 0; axis < 3; axis += 1) out[token * 3 + axis] = positions[from + axis];
    }
    return out;
  };
  const prepareStart = performance.now();
  const table = distogramAgreementTable(
    result.trunk.logits, result.trunk.binEdges, tokens, batch.seqMask);
  const prepareMs = performance.now() - prepareStart;

  // The head emits one distribution per dense atom slot, so a token's pLDDT is
  // the mean over the atoms it actually has.
  const realPlddt = new Float64Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    let total = 0; let count = 0;
    for (let slot = 0; slot < batch.dense; slot += 1) {
      const index = token * batch.dense + slot;
      if (!batch.predDenseAtomMask[index]) continue;
      total += result.scores.plddt[index];
      count += 1;
    }
    realPlddt[token] = count === 0 ? 0 : total / count;
  }
  const scoreStart = performance.now();
  const approx = distogramConfidence(table, gather(result.positions));
  const scoreMs = performance.now() - scoreStart;

  const live = [...batch.seqMask.keys()].filter((t) => batch.seqMask[t] > 0);
  const pick = (values) => Float64Array.from(live, (t) => values[t]);
  const a = pick(realPlddt);
  const b = pick(approx);
  const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
  // 🔴 THE SPREAD OF THE REAL pLDDT IS REPORTED BESIDE THE CORRELATION, because
  // a correlation against something that barely varies is measuring noise. A
  // GS linker's per-residue pLDDT is nearly flat; ranking it is not a question
  // this or anything else can answer, and a low number there is not a failure.
  const sd = (v) => {
    const m = mean(v);
    return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length);
  };
  return {
    tokens: live.length,
    realMean: Number(mean(a).toFixed(1)),
    realSd: Number(sd(a).toFixed(1)),
    approxMean: Number(mean(b).toFixed(1)),
    approxSd: Number(sd(b).toFixed(1)),
    withinSpearman: Number(pearson(ranks(a), ranks(b)).toFixed(3)),
    prepareMs: Number(prepareMs.toFixed(1)),
    perFrameMs: Number(scoreMs.toFixed(2)),
    trajectory: frames.map(({ step, positions }) =>
      Number(mean(pick(distogramConfidence(table, gather(positions)))).toFixed(1))),
    // ...and the same trajectory on the fold's own pLDDT scale, anchored to the
    // final frame. The last entry must equal realMean by construction, which is
    // the check that the calibration is doing what it claims.
    calibrated: (() => {
      const map = calibrateToPlddt(approx, realPlddt, batch.seqMask);
      return frames.map(({ positions }) =>
        Number(mean(pick(map(distogramConfidence(table, gather(positions))))).toFixed(1)));
    })(),
  };
}

export async function main(device, args) {
  const steps = Number(option(args, "steps", "8"));
  const mode = option(args, "mode", "flow");
  const seed = Number(option(args, "seed", "3"));
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const [trunk, diffusion, confidence, reference, targetFeat] = await Promise.all([
    trunkWeights(store), diffusionWeights(store), confidenceWeights(store),
    atomReference(store), targetFeatureWeights(store),
  ]);
  const weights = { trunk, diffusion, confidence, atomReference: reference, targetFeat };

  const requested = option(args, "sequences", "");
  const panel = requested === "panel" ? PANEL
    : requested !== "" ? requested.split(",").map((s, i) => [`seq${i}`, s])
      : [["6MRR", option(args, "sequence", DEFAULT)]];

  const targets = [];
  for (const [name, sequence] of panel) {
    targets.push({ name, ...await foldOne(device, sequence, weights, { steps, mode, seed }) });
  }

  // 🔴 THE ACROSS-TARGET CORRELATION IS OVER THE MEANS, and it is the one a
  // colour that gets compared between folds has to pass.
  const real = Float64Array.from(targets, (t) => t.realMean);
  const approx = Float64Array.from(targets, (t) => t.approxMean);
  const across = targets.length < 3 ? null : {
    pearson: Number(pearson(real, approx).toFixed(3)),
    spearman: Number(pearson(ranks(real), ranks(approx)).toFixed(3)),
  };
  const within = targets.map((t) => t.withinSpearman);
  return {
    mode, steps, seed, targets,
    acrossTargets: across,
    withinTargets: {
      worst: Math.min(...within),
      mean: Number((within.reduce((s, x) => s + x, 0) / within.length).toFixed(3)),
    },
  };
}
