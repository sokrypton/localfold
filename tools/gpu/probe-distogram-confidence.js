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
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";
import { distogramAgreementTable, distogramConfidence }
  from "../../src/af3/distogram-confidence.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const DEFAULT = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK";

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

export async function main(device, args) {
  const sequence = option(args, "sequence", DEFAULT);
  const steps = Number(option(args, "steps", "8"));
  const mode = option(args, "mode", "flow");
  const batch = featuriseProtein(sequence, {});
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const [trunk, diffusion, confidence, reference, targetFeat] = await Promise.all([
    trunkWeights(store), diffusionWeights(store), confidenceWeights(store),
    atomReference(store), targetFeatureWeights(store),
  ]);

  // Every frame's pseudo-beta, kept as the sampler produces them.
  const frames = [];
  const result = await foldBatch(device, batch,
    { trunk, diffusion, confidence, atomReference: reference, targetFeat },
    {
      mode, steps, recycles: 0, seed: 3,
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

  // The real per-token pLDDT: the head emits one distribution per dense atom
  // slot, so a token's score is the mean over the atoms it actually has.
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

  const finalBeta = gather(result.positions);
  const scoreStart = performance.now();
  const approx = distogramConfidence(table, finalBeta);
  const scoreMs = performance.now() - scoreStart;

  const live = [...batch.seqMask.keys()].filter((t) => batch.seqMask[t] > 0);
  const pick = (values) => Float64Array.from(live, (t) => values[t]);
  const a = pick(realPlddt);
  const b = pick(approx);

  const trajectory = frames.map(({ step, positions }) => {
    const scores = distogramConfidence(table, gather(positions));
    const chosen = pick(scores);
    let mean = 0;
    for (const value of chosen) mean += value;
    return {
      step,
      meanScore: Number((mean / chosen.length).toFixed(1)),
      spearmanVsFinalPlddt: Number(pearson(ranks(a), ranks(chosen)).toFixed(3)),
    };
  });

  // --dump returns the raw arrays so the SCORING can be iterated on without
  // re-folding: one fold, then any number of formulations offline.
  if (args.includes("--dump")) {
    return {
      tokens, dense: batch.dense,
      binEdges: [...result.trunk.binEdges],
      seqMask: [...batch.seqMask],
      logits: [...result.trunk.logits],
      realPlddt: [...realPlddt],
      finalBeta: [...finalBeta],
      frames: frames.map(({ step, positions }) => ({ step, beta: [...gather(positions)] })),
    };
  }

  let absolute = 0;
  for (let i = 0; i < a.length; i += 1) absolute += Math.abs(a[i] - b[i]);
  return {
    tokens, live: live.length, mode, steps,
    cost: {
      prepareMs: Number(prepareMs.toFixed(1)),
      perFrameMs: Number(scoreMs.toFixed(2)),
      confidenceHeadMs: "226 at 150 tokens, for scale",
    },
    finalFrame: {
      pearson: Number(pearson(a, b).toFixed(3)),
      spearman: Number(pearson(ranks(a), ranks(b)).toFixed(3)),
      meanAbsoluteError: Number((absolute / a.length).toFixed(1)),
      meanRealPlddt: Number((a.reduce((s, v) => s + v, 0) / a.length).toFixed(1)),
      meanApprox: Number((b.reduce((s, v) => s + v, 0) / b.length).toFixed(1)),
    },
    trajectory,
  };
}
