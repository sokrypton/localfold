/**
 * AF3's triangle (grid) self-attention in WebGL2, against TWO references.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-grid-attention.js [--falsify]
 *
 * `cpu/af3/trunk/pairformer.js`'s `gridSelfAttention` is the f64 reference and
 * `Af3GridSelfAttentionGpu` the WGSL one, both in this page, with the two also
 * held to each other.
 *
 * 🔴 BOTH DIRECTIONS, AND THE TRANSPOSED ONE IS WHERE THE BUGS ARE. AF3 gets the
 * column direction by transposing the normalised ACTIVATION rather than writing
 * a second kernel - so the row arm exercises almost none of what can go wrong
 * there. Three things differ in that direction alone, and each returns a finite
 * tensor when got wrong:
 *
 *   - the MASK is over the KEY position and is the TRANSPOSED pair mask. The
 *     existing WGSL gate checks this "with a RAGGED MASK so that a kernel using
 *     the query's mask instead of the key's fails here rather than in a fold",
 *     and so does this one.
 *   - the BIAS comes from the UNTRANSPOSED activation in every dialect, and the
 *     openfold3 lineage then swaps its indices where stock AF3 does not. Both
 *     dialects are arms.
 *   - and the output is transposed BACK, so the residual lands on the
 *     orientation it came from.
 *
 * 🔴 AND THE RAGGED MASK IS THE POINT OF THE SHAPE HERE. A mask that is all ones
 * cannot tell a key-position mask from a query-position one, and neither can a
 * mask that is symmetric: this one zeroes entries whose transpose is live.
 *
 * 🔴 AND rosettafold3's TWO OPTIONAL BIASES ARE THEIR OWN ARM, because absent is
 * not zero: rf3's gate bias "initialises to 1.0 against a ZERO-initialised
 * weight, so the gate is bias-DOMINATED: dropping it roughly halves the gate".
 * A port that defaulted them to zeros would be wrong by a factor of two on one
 * checkpoint and right everywhere else.
 *
 * 🔴 AND THAT ARM RUNS AT ITS OWN LENGTH, BECAUSE THE WGSL `af3-grid` PIPELINE
 * KEY DOES NOT NAME THE OPTIONAL BIASES. Its eleven components are the shape,
 * the epsilon, the variance, the dialect, the precision and five attend knobs -
 * none of them says whether `gatingQueryBias` exists - while the generated
 * source embeds `const W_GATE_BIAS: u32 = 5312u;` only when it does. Running
 * both cases at one shape in one page therefore makes `ComputePipelineCache`
 * refuse, by name and correctly: "key collision for af3-grid:7:32:4:8:true:...
 * - line 17 of 102".
 *
 * It is a LATENT gap rather than a live bug, and that is worth stating as
 * carefully as the gap: rosettafold3 is the only family with those biases, and
 * `check-pipeline-key-collisions.js --models=af3,rosettafold3` reports
 * `collision: null`, so no two shipping checkpoints reach it. What stops it from
 * ever being silent is that the cache indexes by SOURCE as well as by key - the
 * same design CLAUDE.md credits with catching the msa-attention collision.
 */
import { gridSelfAttention as cpuGrid } from "../../cpu/af3/trunk/pairformer.js";
import { Af3GridSelfAttentionGpu } from "../../webgpu/af3/trunk/grid-attention.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  gridSelfAttention, releaseGridAttentionWeights, uploadGridAttentionWeights,
} from "../../webgl/af3/trunk/grid-attention.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "5e-6"));
  const length = Number(option(args, "length", "7"));
  const channels = Number(option(args, "channels", "32"));
  const heads = Number(option(args, "heads", "4"));
  const dimension = Number(option(args, "dimension", "8"));
  const width = heads * dimension;
  const pairs = length * length;

  const random = generator(161803);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  const base = {
    heads, dimension,
    actNormScale: noise(channels, 0.2, 1),
    actNormOffset: noise(channels, 0.1),
    pairBiasProjection: noise(channels * heads, 1 / Math.sqrt(channels)),
    qProjection: noise(width * channels, 1 / Math.sqrt(channels)),
    kProjection: noise(width * channels, 1 / Math.sqrt(channels)),
    vProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    gatingQuery: noise(width * channels, 1 / Math.sqrt(channels)),
    outputProjection: noise(width * channels, 1 / Math.sqrt(width)),
  };
  // rf3's two, the gate's deliberately large against a small weight - which is
  // the regime that makes dropping it halve the gate.
  const withBiases = {
    ...base,
    gatingQuery: noise(width * channels, 0.02 / Math.sqrt(channels)),
    gatingQueryBias: noise(width, 0.3, 1),
    outputProjectionBias: noise(channels, 0.2),
  };

  const pair = noise(pairs * channels, 1.2);
  // 🔴 A RAGGED, ASYMMETRIC MASK. A pair (i, j) is live where its transpose is
  // not, so a kernel reading the query's mask instead of the key's, or failing
  // to transpose it for the column direction, lands somewhere else.
  const mask = new Float32Array(pairs).fill(1);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      if ((i * 3 + j) % 7 === 0 && i !== j) mask[i * length + j] = 0;
    }
  }

  setDeviceTuning(device, { allowHostWeightPacking: true });
  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const maskTensor = execution.upload(mask, "af3.grid.mask");
  // The control skews half of each pair row's channels: the pair meets a
  // LayerNorm first, and a whole-row affine change is invisible to one.
  const pairTensor = execution.upload(falsify
    ? Float32Array.from(pair, (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v))
    : pair, "af3.grid.pair");

  const arms = [];
  const failures = [];
  /** @type {{name: string, transpose: boolean, swap: boolean, weights: any}[]} */
  const CASES = [];
  for (const transpose of [false, true]) {
    for (const swap of [false, true]) {
      // The swap only applies to the transposed direction, so the row arm needs
      // only one of the two dialects.
      if (!transpose && swap) continue;
      CASES.push({
        name: `${transpose ? "column" : "row"}${swap ? ", openfold3 bias swap" : ""}`,
        transpose, swap, weights: base,
      });
    }
  }
  CASES.push({
    name: "column, rosettafold3's two biases", transpose: true, swap: false,
    weights: withBiases,
    // ...at a different length, so the WGSL side's key differs. See the header.
    length: length + 1,
  });

  for (const { name, transpose, swap, weights, length: caseLength = length } of CASES) {
    const casePairs = caseLength * caseLength;
    const casePair = caseLength === length ? pair : noise(casePairs * channels, 1.2);
    const caseMask = caseLength === length ? mask : (() => {
      const out = new Float32Array(casePairs).fill(1);
      for (let i = 0; i < caseLength; i += 1) {
        for (let j = 0; j < caseLength; j += 1) {
          if ((i * 3 + j) % 7 === 0 && i !== j) out[i * caseLength + j] = 0;
        }
      }
      return out;
    })();
    const casePairTensor = caseLength === length ? pairTensor : execution.upload(falsify
      ? Float32Array.from(casePair,
        (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v))
      : casePair, `af3.grid.pair.${caseLength}`);
    const caseMaskTensor = caseLength === length
      ? maskTensor : execution.upload(caseMask, `af3.grid.mask.${caseLength}`);
    const uploaded = uploadGridAttentionWeights(execution, weights);
    const update = gridSelfAttention(execution, {
      pair: casePairTensor, mask: caseMaskTensor, weights: uploaded,
      length: caseLength, channels, transpose, swapTransposedBias: swap,
    });
    const ours = Float32Array.from(execution.read(update));
    execution.release(update);
    releaseGridAttentionWeights(execution, uploaded);

    const dialect = { swapTransposedBias: swap };
    const expected = cpuGrid(
      casePair, caseMask, caseLength, channels, transpose, weights, dialect);
    const theirs = await new Af3GridSelfAttentionGpu(device).run(
      casePair, caseMask, { n: caseLength, channels, transpose }, weights, dialect,
      { stagedPrecision: "f32" });

    for (const [label, want] of [["f64 reference", expected], ["WGSL", theirs.output]]) {
      const armName = `${name} vs the ${label}`;
      const relative = relativeRms(ours, want, armName);
      const ok = relative <= bar;
      arms.push({ arm: armName, relativeRms: Number(relative.toPrecision(4)), bar, ok });
      if (!ok) failures.push(`${armName}: relRMS ${relative.toExponential(3)} over ${bar}`);
    }
    const between = relativeRms(expected, theirs.output, `between.${name}`);
    arms.push({
      arm: `${name}: the two references agree`,
      relativeRms: Number(between.toPrecision(4)), bar, ok: between <= bar,
      falsifiable: false,
      reason: "both references read the unperturbed pair",
    });
    if (!(between <= bar)) {
      failures.push(`${name}: the references disagree by ${between.toExponential(3)}`);
    }
    if (casePairTensor !== pairTensor) execution.release(casePairTensor);
    if (caseMaskTensor !== maskTensor) execution.release(caseMaskTensor);
  }

  execution.release(pairTensor);
  execution.release(maskTensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: { length, channels, heads, dimension },
    maskedPairs: mask.length - mask.reduce((a, b) => a + b, 0),
    bar,
    falsify,
    arms,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms over ${CASES.length} cases, both references agreeing`
        : `FAILED: ${failures.join("; ")}`),
  };
}
