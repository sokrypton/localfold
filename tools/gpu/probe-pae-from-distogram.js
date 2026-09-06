/**
 * Can a PAE matrix be predicted from a distogram and a structure?
 *
 * WHY IT IS WORTH ASKING. EF2-fast has no confidence head at all - no pLDDT,
 * no PAE - and its distogram is the only thing it says about its own
 * reliability. A PAE is the score people actually read off a complex, because
 * it is what says whether two domains are placed relative to each other. If one
 * can be estimated from a distogram, this checkpoint gets the score it lacks.
 *
 * WHY IT CAN BE CHECKED RATHER THAN ARGUED. AlphaFold 3 produces BOTH: its
 * trunk's distogram head and its confidence head's PAE come out of the same
 * fold. So an estimator can be fitted and scored against a real PAE here, and
 * only then carried to the model that has none.
 *
 * 🔴 PAE IS NOT A DISTANCE ERROR, WHICH IS THE WHOLE DIFFICULTY. It is the
 * expected error in token j's POSITION after superposing on token i's FRAME -
 * so it is asymmetric, and it is large exactly when two rigid parts are each
 * confidently folded and badly placed against each other. A distogram knows
 * only about distances, and about pairs one at a time. What is being tested is
 * whether the distances CARRY the rigid-body information.
 *
 * 🔴 AND THE BASELINE HAS TO BE THE GEOMETRY, NOT ZERO. PAE grows with distance
 * and with sequence separation whatever the model thinks, so an estimator that
 * reports the distance would already correlate well. Anything proposed here
 * must beat `d_ij` and `|i - j|`, which are computed alongside for exactly that
 * reason.
 *
 * This dumps per-pair features beside the true PAE rather than deciding
 * anything, so the analysis can be iterated without re-folding.
 */
import { confidenceWeights, openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { atomReference, diffusionWeights, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch } from "../../src/af3/fold.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * Per-pair moments of the distogram: the mean distance it expects, the spread
 * it allows, and its entropy.
 *
 * 🔴 THE LAST BIN IS OPEN-ENDED AND ITS CENTRE IS A FICTION. 63 breaks describe
 * 64 bins, so the first and last have no finite centre; they are given the
 * edge plus/minus half a spacing, which is what every reader of an AlphaFold
 * distogram does and is worth saying out loud because the moments below are
 * sensitive to it at exactly the pairs that matter - a pair the model thinks is
 * far apart is one it is unsure about.
 */
function distogramMoments(logits, tokens, bins, breaks) {
  const centres = new Float64Array(bins);
  const step = breaks[1] - breaks[0];
  centres[0] = breaks[0] - step / 2;
  for (let b = 1; b < bins - 1; b += 1) centres[b] = (breaks[b - 1] + breaks[b]) / 2;
  centres[bins - 1] = breaks[bins - 2] + step / 2;

  const pairs = tokens * tokens;
  const mean = new Float32Array(pairs);
  const sigma = new Float32Array(pairs);
  const entropy = new Float32Array(pairs);
  const probability = new Float64Array(bins);
  for (let index = 0; index < pairs; index += 1) {
    const base = index * bins;
    let largest = -Infinity;
    for (let b = 0; b < bins; b += 1) {
      if (logits[base + b] > largest) largest = logits[base + b];
    }
    let total = 0;
    for (let b = 0; b < bins; b += 1) {
      probability[b] = Math.exp(logits[base + b] - largest);
      total += probability[b];
    }
    let m = 0, m2 = 0, h = 0;
    for (let b = 0; b < bins; b += 1) {
      const p = probability[b] / total;
      m += p * centres[b];
      m2 += p * centres[b] * centres[b];
      if (p > 1e-12) h -= p * Math.log(p);
    }
    mean[index] = m;
    sigma[index] = Math.sqrt(Math.max(0, m2 - m * m));
    entropy[index] = h;
  }
  return { mean, sigma, entropy, centres };
}

export async function main(device, args = []) {
  const sequence = option(args, "sequence",
    "GSMKQIEDKIEEILSKIYHIENEIARIKKLIGEA");
  const steps = Number(option(args, "steps", "50"));
  const recycles = Number(option(args, "recycles", "0"));
  const seed = Number(option(args, "seed", "20260831"));
  const model = option(args, "model", "/model-af3-int5/manifest.json");
  const blocks = Number(option(args, "blocks", "48"));

  const batch = featuriseProtein(sequence, {});
  const store = await openAf3Store(model, null);
  const weights = {
    trunk: await trunkWeights(store, blocks, 4),
    diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  const result = await foldBatch(device, batch, weights, {
    mode: "diffusion", steps, stopAfter: steps, recycles, seed,
  });

  const tokens = batch.tokens;
  const bins = result.trunk.binEdges.length + 1;
  const { mean, sigma, entropy } = distogramMoments(
    result.trunk.logits, tokens, bins, result.trunk.binEdges);

  // ...the points the PAE is about, taken from the fold rather than recomputed.
  const beta = result.pseudoBeta;
  const observed = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      observed[i * tokens + j] = Math.hypot(
        beta[i * 3] - beta[j * 3], beta[i * 3 + 1] - beta[j * 3 + 1],
        beta[i * 3 + 2] - beta[j * 3 + 2]);
    }
  }

  const pae = result.scores.pae;
  const asymId = batch.features?.asymId ?? new Int32Array(tokens);
  const residue = batch.features?.residueIndex ?? Int32Array.from(
    { length: tokens }, (_, i) => i);

  return {
    ok: true, sequence: sequence.length, tokens, bins, steps, recycles, seed,
    meanPlddt: result.meanPlddt, ptm: result.ptm,
    // Flat row-major (i * tokens + j), so the analysis can reshape once.
    data: {
      pae: [...pae].map((v) => Number(v.toFixed(3))),
      sigma: [...sigma].map((v) => Number(v.toFixed(3))),
      mean: [...mean].map((v) => Number(v.toFixed(3))),
      entropy: [...entropy].map((v) => Number(v.toFixed(4))),
      observed: [...observed].map((v) => Number(v.toFixed(3))),
      asymId: [...asymId],
      residue: [...residue],
      plddt: [...result.scores.plddt].map((v) => Number(v.toFixed(2))),
    },
  };
}
