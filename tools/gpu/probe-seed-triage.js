/**
 * Does the FIRST denoiser call already say which seed is worth finishing?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-seed-triage.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-seed-triage.js --seeds=8 --steps=20
 *
 * THE IDEA. A sample costs 16 to 200 denoiser calls and a seed names a
 * structure, so folding a complex several times and keeping the best is the
 * obvious way to use a diffusion model - and it costs N times as much. If the
 * output of ONE call already ranked the seeds, a page could screen twenty of
 * them for the price of a single extra sample and finish only the ones worth
 * finishing.
 *
 * 🔴 IT MIGHT WELL NOT, AND THE REASON IS IN THE ARITHMETIC. The first call
 * sees noise at sigma0 - 160 A for the flow, 2560 for the sampler - which is
 * several times the structure's own extent, so its input carries almost no
 * information and its output is close to the conditional MEAN over the whole
 * posterior. That mean is the same for every seed. See the note in
 * diffusion-sampler-webgpu.js: from zeros the bond lengths come back at 73-87%
 * of ideal, which is the signature of an average rather than a structure. The
 * question this answers is whether the noise breaks that symmetry ENOUGH by
 * step one to carry a ranking.
 *
 * WHAT IT MEASURES, per target, over N seeds:
 *
 *   - the spread among the step-1 structures themselves, because a screen over
 *     something that does not vary cannot rank anything;
 *   - Spearman between three cheap step-1 scores and the FINAL ipTM, which is
 *     what the screen would have to predict.
 *
 * 🔴 SPEARMAN AND NOT PEARSON. A screen only has to ORDER the seeds; it never
 * has to get the value right.
 *
 * WHAT IT FOUND: THE PREMISE HOLDS AND THE SCREEN DOES NOT. Over eight seeds
 * of AF3's sampler at 20 steps -
 *
 *                    step-1 spread   final ipTM   settled  cross  Rg
 *     MRR dimer      11.8 A RMSD     0.092-0.203   -0.31   0.024  0.024
 *     Top7 x3        20.9 A RMSD     0.070-0.163   -0.23   0.524  -0.554
 *
 * The step-1 guesses differ by 12 to 21 A, so the noise DOES break the
 * conditional mean's symmetry and there is something to rank - the worry above
 * was wrong about the spread. But no step-1 score ranks the seeds: the best
 * showing is cross-chain agreement at 0.524 on one target and 0.024 on the
 * other, which at eight seeds is not a signal, it is a coin.
 *
 * 🔴 WHICH SEED WINS IS NOT DECIDED AT STEP ONE. The seeds start apart and the
 * REST OF THE WALK decides where they end up, so a screen has to look later.
 * The open question this leaves is how much later: capture at steps 1, 2, 4 and
 * 8 and correlate each against the final, and the answer is the price of a
 * screen. That sweep has not been run.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights }
  from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";
import { distogramAgreementTable, distogramConfidence }
  from "../../src/af3/distogram-confidence.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const MRR = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK";
const TOP7 = "DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLE";

/**
 * 🔴 TARGETS WITH REAL SEED SPREAD, or there is nothing to rank. The MRR dimer
 * ran 0.106 to 0.203 in ipTM over two seeds of AF3's sampler in
 * probe-flow-sigma-by-size.js, which is the widest on that panel.
 */
const PANEL = [
  ["MRR dimer", `${MRR}:${MRR}`],
  ["Top7 x3", `${TOP7}:${TOP7}:${TOP7}`],
];

function pearson(a, b) {
  const n = a.length;
  if (n < 2) return 0;
  const ma = a.reduce((x, y) => x + y, 0) / n;
  const mb = b.reduce((x, y) => x + y, 0) / n;
  let num = 0; let da = 0; let db = 0;
  for (let i = 0; i < n; i += 1) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return da === 0 || db === 0 ? 0 : num / Math.sqrt(da * db);
}

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
const spearman = (a, b) => Number(pearson([...ranks(a)], [...ranks(b)]).toFixed(3));

/** Token pseudo-beta from a dense atom array. */
function pseudoBetaOf(batch, positions) {
  const beta = batch.tokenAtomsToPseudoBeta;
  const out = new Float32Array(batch.tokens * 3);
  for (let token = 0; token < batch.tokens; token += 1) {
    if (!beta.mask[token]) continue;
    const from = Number(beta.indices[token]) * 3;
    for (let axis = 0; axis < 3; axis += 1) out[token * 3 + axis] = positions[from + axis];
  }
  return out;
}

/** Root mean square deviation with no superposition - these share a frame. */
function rmsd(a, b, batch) {
  let total = 0;
  let count = 0;
  for (let token = 0; token < batch.tokens; token += 1) {
    if (batch.seqMask[token] <= 0) continue;
    let squared = 0;
    for (let axis = 0; axis < 3; axis += 1) {
      squared += (a[token * 3 + axis] - b[token * 3 + axis]) ** 2;
    }
    total += squared;
    count += 1;
  }
  return count === 0 ? 0 : Math.sqrt(total / count);
}

/** The radius of gyration, as a plain shape statistic of a step-1 guess. */
function radiusOfGyration(beta, batch) {
  const centre = [0, 0, 0];
  let count = 0;
  for (let token = 0; token < batch.tokens; token += 1) {
    if (batch.seqMask[token] <= 0) continue;
    for (let axis = 0; axis < 3; axis += 1) centre[axis] += beta[token * 3 + axis];
    count += 1;
  }
  for (let axis = 0; axis < 3; axis += 1) centre[axis] /= Math.max(count, 1);
  let total = 0;
  for (let token = 0; token < batch.tokens; token += 1) {
    if (batch.seqMask[token] <= 0) continue;
    for (let axis = 0; axis < 3; axis += 1) {
      total += (beta[token * 3 + axis] - centre[axis]) ** 2;
    }
  }
  return Math.sqrt(total / Math.max(count, 1));
}

/** How well a frame's CROSS-CHAIN distances match the trunk's distogram. */
function crossChainAgreement(table, beta, batch) {
  const { tokens, expected, spread } = table;
  let total = 0;
  let count = 0;
  for (let i = 0; i < tokens; i += 1) {
    if (batch.seqMask[i] <= 0) continue;
    for (let j = 0; j < tokens; j += 1) {
      if (batch.seqMask[j] <= 0 || batch.asymId[i] === batch.asymId[j]) continue;
      let squared = 0;
      for (let axis = 0; axis < 3; axis += 1) {
        squared += (beta[i * 3 + axis] - beta[j * 3 + axis]) ** 2;
      }
      const deviation = Math.abs(Math.sqrt(squared) - expected[i * tokens + j]);
      total += 1 / (1 + (deviation / Math.max(spread[i * tokens + j], 0.1)) ** 2);
      count += 1;
    }
  }
  return count === 0 ? Number.NaN : total / count;
}

export async function main(device, args) {
  const steps = Number(option(args, "steps", "20"));
  const mode = option(args, "mode", "diffusion");
  const seedCount = Number(option(args, "seeds", "8"));
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const [trunk, diffusion, confidence, reference, targetFeat] = await Promise.all([
    trunkWeights(store), diffusionWeights(store), confidenceWeights(store),
    atomReference(store), targetFeatureWeights(store),
  ]);
  const weights = { trunk, diffusion, confidence, atomReference: reference, targetFeat };

  const rows = [];
  for (const [name, sequence] of PANEL) {
    const batch = featuriseProtein(sequence, {});
    let reuse;
    let table;
    const seeds = [];
    for (let seed = 1; seed <= seedCount; seed += 1) {
      let first = null;
      const result = await foldBatch(device, batch, weights, {
        mode, steps, recycles: 0, seed, reuse,
        // 🔴 THE FIRST CALL'S OUTPUT, WHICH IS THE WHOLE POINT. `denoised` is
        // the model's predicted structure at that call, not the noisy walk.
        onStep: ({ step, denoised }) => {
          if (first === null && step >= 1) first = Float32Array.from(denoised);
        },
      });
      reuse = result.reusable;
      if (table === undefined) {
        table = distogramAgreementTable(
          result.trunk.logits, result.trunk.binEdges, batch.tokens, batch.seqMask);
      }
      const beta = pseudoBetaOf(batch, first);
      const perToken = distogramConfidence(table, beta);
      let settled = 0;
      let live = 0;
      for (let token = 0; token < batch.tokens; token += 1) {
        if (batch.seqMask[token] <= 0) continue;
        settled += perToken[token];
        live += 1;
      }
      seeds.push({
        seed,
        firstBeta: beta,
        step1Settled: Number((settled / Math.max(live, 1)).toFixed(2)),
        step1Cross: Number(crossChainAgreement(table, beta, batch).toFixed(4)),
        step1Rg: Number(radiusOfGyration(beta, batch).toFixed(1)),
        iptm: Number(result.iptm.toFixed(3)),
        ptm: Number(result.ptm.toFixed(3)),
      });
    }

    // 🔴 DO THE STEP-1 GUESSES EVEN DIFFER? A screen over something constant
    // ranks nothing, and this is the number that decides whether the rest of
    // the table means anything.
    const spread = [];
    for (let i = 0; i < seeds.length; i += 1) {
      for (let j = i + 1; j < seeds.length; j += 1) {
        spread.push(rmsd(seeds[i].firstBeta, seeds[j].firstBeta, batch));
      }
    }
    const iptm = seeds.map((s) => s.iptm);
    rows.push({
      name,
      tokens: batch.tokens,
      step1SpreadRmsd: {
        mean: Number((spread.reduce((a, b) => a + b, 0) / spread.length).toFixed(2)),
        min: Number(Math.min(...spread).toFixed(2)),
        max: Number(Math.max(...spread).toFixed(2)),
      },
      finalIptm: {
        min: Math.min(...iptm), max: Math.max(...iptm),
        spread: Number((Math.max(...iptm) - Math.min(...iptm)).toFixed(3)),
      },
      predicts: {
        settled: spearman(iptm, seeds.map((s) => s.step1Settled)),
        crossChain: spearman(iptm, seeds.map((s) => s.step1Cross)),
        radiusOfGyration: spearman(iptm, seeds.map((s) => s.step1Rg)),
      },
      seeds: seeds.map(({ firstBeta, ...rest }) => rest),
    });
  }
  return { mode, steps, seedCount, rows };
}
