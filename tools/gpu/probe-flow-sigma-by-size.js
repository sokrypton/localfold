/**
 * Does the flow's starting sigma need to grow with the complex?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-flow-sigma-by-size.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-flow-sigma-by-size.js --seeds=1,2,3,4
 *
 * THE HYPOTHESIS, and it has a mechanism rather than a hunch behind it. The
 * flow starts from one draw at sigma0 = 160 A while AF3's own sampler starts at
 * 2560 A. 160 A is an ABSOLUTE length, and a complex's extent is not: a 64
 * token dimer spans some 40 A and the starting cloud is four times its size,
 * while a 400 token one spans past 100 A and the cloud is barely wider than the
 * answer. If the two chains have to find each other, the scale of the initial
 * cloud is what decides how freely they can, so the flow should lose ground to
 * the sampler as the complex grows - and the fix would be a sigma0 that scales
 * rather than a larger constant.
 *
 * 🔴 THE EXISTING SWEEP CANNOT SEE THIS. tools/gpu/probe-sigma0.js settled
 * sigma0 on two MONOMERS of 68 and 92 residues, where 160 A is four to five
 * times the structure and every candidate looks alike. It measured RMSD to a
 * crystal, which is the right metric for a fold and says nothing about whether
 * two chains were placed against each other.
 *
 * WHAT IT MEASURES. ipTM from the real confidence head, per seed, against the
 * flow's sigma0 and against AF3's sampler on the same trunk. ipTM is the
 * model's own opinion and not a crystal - but it is the quantity that differs
 * when a complex comes out differently, and this is asking whether the SAMPLER
 * changes the answer, which is a question about the model's own output.
 *
 * WHAT IT FOUND, AND THE HYPOTHESIS LOST. Stacking copies of a chain that
 * folds drives the ratio from 3.27 down to 2.03, and sigma0 does nothing over
 * that whole range - mean ipTM over two seeds:
 *
 *     Top7 x2  186 tok  49 A  ratio 3.27   0.698  0.700  0.706
 *     Top7 x3  279 tok  63 A  ratio 2.54   0.162  0.170  0.178
 *     Top7 x4  372 tok  67 A  ratio 2.39   0.152  0.147  0.147
 *     Top7 x6  558 tok  79 A  ratio 2.03   0.132  0.134  0.133
 *
 * (columns are sigma0 = 160, 640, 2560 A.) Sixteen times the starting noise
 * moves ipTM by at most 0.016, with no consistent sign, mostly inside the seed
 * range. Whatever separates the two samplers on a complex, it is not where
 * they start.
 *
 * 🔴 WHAT IT IS INSTEAD IS THE STEP COUNT, and `--panel=churn` measures it.
 * AF3's sampler injects noise at EVERY step and is specified at 200 of them;
 * the page offers 20. On Top7 x3, mean ipTM over two seeds:
 *
 *     diffusion 20   0.077      diffusion 160  0.133 [0.083-0.182]
 *     diffusion 40   0.136      flow 16        0.162
 *     diffusion 80   0.176
 *
 * A walk that adds noise each step needs steps left to descend out of it, and
 * at 20 it does not have them - it scores less than half what it scores at 80,
 * where it matches the flow. pTM tracks the same shape. The 80-against-160
 * ordering is NOT resolved at two seeds; 160's range swallows the difference.
 * What holds is that 20 is too few.
 *
 * 🔴 SO THE PAGE'S DEFAULTS COMPARE AN UNCONVERGED SAMPLER WITH A CONVERGED
 * ONE: flow at 16 against diffusion at 20. That is a defaults question, not a
 * sampler question, and it wants a wider panel before anything moves.
 *
 * 🔴 ONE TRUNK PER TARGET, REUSED. The trunk is fixed for a sequence, so every
 * arm here differs only in the sampler - which is the comparison - and the
 * trunk is not paid for twelve times.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights }
  from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const GCN4 = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER";
const MRR = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK";
const DES58 = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const TOP7 = "DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLE";

/**
 * 🔴 THE FIRST PANEL COULD NOT TEST THE HYPOTHESIS AND IS KEPT TO SHOW WHY.
 * Its five complexes run 64 to 210 tokens and their extents came back 45, 40,
 * 51, 49 and 53 A - flat. Size grows as N^(1/3), so a 3.3x in tokens is a 1.5x
 * in length, and packing ate even that. The ratio the hypothesis is about
 * stayed between 3.0 and 4.0 across the whole panel, so every arm was asking
 * the same question.
 *
 * `--panel=copies` is the one that varies it: the same chain that folds, in
 * more and more copies, so extent grows while everything else is held.
 */
const PANEL = [
  ["GCN4 dimer", `${GCN4}:${GCN4}`],
  ["MRR dimer", `${MRR}:${MRR}`],
  ["Top7+MRR", `${TOP7}:${MRR}`],
  ["Top7 dimer", `${TOP7}:${TOP7}`],
  ["Top7+MRR+DES58", `${TOP7}:${MRR}:${DES58}`],
];

/** sigma0 in angstroms; sigmaMax is in units of sigmaData, which is 16 A. */
const SIGMAS = [160, 640, 2560];

const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;

/** The structure's own size, so sigma0 can be read against something. */
function extent(positions, batch) {
  const beta = batch.tokenAtomsToPseudoBeta;
  const points = [];
  for (let token = 0; token < batch.tokens; token += 1) {
    if (!beta.mask[token] || batch.seqMask[token] <= 0) continue;
    const from = Number(beta.indices[token]) * 3;
    points.push([positions[from], positions[from + 1], positions[from + 2]]);
  }
  let worst = 0;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      let squared = 0;
      for (let axis = 0; axis < 3; axis += 1) {
        squared += (points[i][axis] - points[j][axis]) ** 2;
      }
      if (squared > worst) worst = squared;
    }
  }
  return Math.sqrt(worst);
}

const COPIES = [2, 3, 4, 6].map((count) => [
  `Top7 x${count}`, Array.from({ length: count }, () => TOP7).join(":")]);

/**
 * 🔴 THE OTHER DIFFERENCE BETWEEN THE TWO SAMPLERS, AND THE ONE THAT SHOWED UP.
 * AF3's sampler injects noise at EVERY step and is specified at 200 of them;
 * the page offers it at 20. A walk that adds noise each step needs enough steps
 * left to descend out of it again, so this asks whether the gap against the
 * flow is the churn being under-stepped rather than anything about sigma0.
 */
const STEP_SWEEP = [20, 40, 80, 160];

async function churnSweep(device, batch, weights, { steps, seeds, arm }) {
  const out = {};
  for (const count of STEP_SWEEP) {
    const runs = [];
    for (const seed of seeds) runs.push(await arm("diffusion", seed, undefined, count));
    out[`diffusion ${count} steps`] = {
      iptm: Number(mean(runs.map((r) => r.iptm)).toFixed(3)),
      range: [Math.min(...runs.map((r) => r.iptm)), Math.max(...runs.map((r) => r.iptm))],
      ptm: Number(mean(runs.map((r) => r.ptm)).toFixed(3)),
    };
  }
  const flow = [];
  for (const seed of seeds) flow.push(await arm("flow", seed, undefined, steps));
  out[`flow ${steps} steps`] = {
    iptm: Number(mean(flow.map((r) => r.iptm)).toFixed(3)),
    range: [Math.min(...flow.map((r) => r.iptm)), Math.max(...flow.map((r) => r.iptm))],
    ptm: Number(mean(flow.map((r) => r.ptm)).toFixed(3)),
  };
  return out;
}

export async function main(device, args) {
  const steps = Number(option(args, "steps", "16"));
  const diffusionSteps = Number(option(args, "diffusion-steps", "20"));
  const seeds = option(args, "seeds", "1,2,3").split(",").map(Number);
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const [trunk, diffusion, confidence, reference, targetFeat] = await Promise.all([
    trunkWeights(store), diffusionWeights(store), confidenceWeights(store),
    atomReference(store), targetFeatureWeights(store),
  ]);
  const weights = { trunk, diffusion, confidence, atomReference: reference, targetFeat };

  const which = option(args, "panel", "pairs");
  const panel = which === "copies" ? COPIES
    : which === "churn" ? COPIES.filter(([name]) => name.endsWith(`x${option(args, "copies", "3")}`))
      : PANEL;
  const rows = [];
  for (const [name, sequence] of panel) {
    const batch = featuriseProtein(sequence, {});
    // The first fold pays for the trunk; every arm after it reuses that trunk,
    // so the only thing that differs between arms is the sampler.
    let reuse;
    const arm = async (mode, seed, schedule, count) => {
      const result = await foldBatch(device, batch, weights, {
        mode, steps: count ?? (mode === "diffusion" ? diffusionSteps : steps),
        recycles: 0, seed, reuse, schedule,
      });
      reuse = result.reusable;
      return {
        iptm: Number(result.iptm.toFixed(3)),
        ptm: Number(result.ptm.toFixed(3)),
        extent: Math.round(extent(result.positions, batch)),
      };
    };
    const row = { name, tokens: batch.tokens, arms: {} };
    if (which === "churn") {
      row.arms = await churnSweep(device, batch, weights, { steps, seeds, arm });
      rows.push(row);
      continue;
    }
    for (const sigma of SIGMAS) {
      const runs = [];
      for (const seed of seeds) {
        runs.push(await arm("flow", seed, { sigmaMax: sigma / 16 }));
      }
      row.arms[`flow sigma0=${sigma}`] = {
        iptm: Number(mean(runs.map((r) => r.iptm)).toFixed(3)),
        range: [Math.min(...runs.map((r) => r.iptm)), Math.max(...runs.map((r) => r.iptm))],
        ptm: Number(mean(runs.map((r) => r.ptm)).toFixed(3)),
      };
      if (sigma === SIGMAS[0]) row.extent = Math.round(mean(runs.map((r) => r.extent)));
    }
    const sampled = [];
    for (const seed of seeds) sampled.push(await arm("diffusion", seed, undefined));
    row.arms["diffusion (AF3, sigma0=2560 + churn)"] = {
      iptm: Number(mean(sampled.map((r) => r.iptm)).toFixed(3)),
      range: [Math.min(...sampled.map((r) => r.iptm)), Math.max(...sampled.map((r) => r.iptm))],
      ptm: Number(mean(sampled.map((r) => r.ptm)).toFixed(3)),
    };
    // 🔴 THE RATIO IS THE POINT. sigma0 against the structure's own extent is
    // what the hypothesis says should matter, and it is the column that varies
    // across this panel while sigma0 alone does not.
    row.cloudOverExtent = Number((160 / row.extent).toFixed(2));
    rows.push(row);
  }
  return { steps, diffusionSteps, seeds, rows };
}
