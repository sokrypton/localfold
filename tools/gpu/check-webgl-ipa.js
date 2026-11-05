/**
 * Invariant point attention against the WGSL one.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-ipa.js [--falsify]
 *
 * 🔴 THE ONE ATTENTION IN THE MODEL WHOSE LOGITS ARE A DISTANCE, and the stage
 * with the most ways to be plausibly wrong of anything in AF2. The point term is
 * SUBTRACTED and is a SQUARED distance, scaled by a softplus of a trained
 * per-head parameter; the three scale factors are `sqrt(1/(3*scalarQk))`,
 * `sqrt(1/(3*pointQk*4.5))` and `sqrt(1/3)`, and the 4.5 cannot be guessed; the
 * local point layout is `[residue][component][head][point]` where the global one
 * is `[residue][head][point][xyz]`; and the point feature comes back into the
 * QUERY's frame with a fourth plane holding the vector's norm.
 *
 * Every one of those produces a finite update of the right shape when wrong,
 * which is why this is checked against a second implementation rather than
 * against a reading of the shader.
 *
 * 🔴 AND THE AFFINE IS NOT A UNIT QUATERNION HERE EITHER. The reference rotates
 * with it as given, so a port that normalised would agree on unit input and
 * differ on what the iteration actually produces.
 */
import { InvariantPointAttentionGpu } from "../../webgpu/af2/structure/ipa.js";
import { setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  invariantPointAttention, normalisePair, releaseIpaWeights, uploadIpaWeights,
} from "../../webgl/af2/structure/ipa.js";
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
  const bar = Number(option(args, "bar", "1e-5"));
  const shape = {
    length: Number(option(args, "length", "7")),
    channels: 32,
    pairChannels: 16,
    heads: 4,
    scalarQk: 8,
    scalarV: 8,
    pointQk: 2,
    pointV: 4,
  };
  const { length, channels, pairChannels, heads, scalarQk, scalarV, pointQk, pointV } = shape;
  const featureChannels = heads * scalarV + 4 * heads * pointV + heads * pairChannels;

  const random = generator(1618);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const activations = noise(length * channels, 1.5);
  const pair = noise(length * length * pairChannels, 1.5);
  // Not a unit quaternion, on purpose - see the header.
  const affine = noise(length * 7, 0.7, 0.3);
  const mask = new Float32Array(length).fill(1);
  if (length > 2) mask[length - 2] = 0;

  const weights = {
    pairNormScale: noise(pairChannels, 0.2, 1),
    pairNormOffset: noise(pairChannels, 0.1),
    queryScalarWeight: noise(channels * heads * scalarQk, 1 / Math.sqrt(channels)),
    queryScalarBias: noise(heads * scalarQk, 0.1),
    keyValueScalarWeight: noise(channels * heads * (scalarQk + scalarV), 1 / Math.sqrt(channels)),
    keyValueScalarBias: noise(heads * (scalarQk + scalarV), 0.1),
    queryPointWeight: noise(channels * heads * pointQk * 3, 1 / Math.sqrt(channels)),
    queryPointBias: noise(heads * pointQk * 3, 0.1),
    keyValuePointWeight: noise(channels * heads * (pointQk + pointV) * 3, 1 / Math.sqrt(channels)),
    keyValuePointBias: noise(heads * (pointQk + pointV) * 3, 0.1),
    trainablePointWeights: noise(heads, 0.5),
    attention2dWeight: noise(pairChannels * heads, 1 / Math.sqrt(pairChannels)),
    attention2dBias: noise(heads, 0.1),
    outputWeight: noise(featureChannels * channels, 1 / Math.sqrt(featureChannels)),
    outputBias: noise(channels, 0.05),
  };

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const uploaded = uploadIpaWeights(execution, weights);
  const pairTensor = execution.upload(pair, "pair");
  const normalisedPair = normalisePair(execution, {
    pair: pairTensor, weights: uploaded, length, pairChannels,
  });
  const ours = invariantPointAttention(execution, {
    activations: execution.upload(falsify
      ? Float32Array.from(activations, (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v))
      : activations, "activations"),
    normalisedPair,
    mask: execution.upload(mask, "mask"),
    affine: execution.upload(affine, "affine"),
    weights: uploaded,
    ...shape,
  });

  setHalfPrecision(device, false);
  const theirs = await new InvariantPointAttentionGpu(device).run({
    activations, pair, mask, affine, weights, ...shape,
  });

  const got = execution.read(ours);
  const want = theirs.output ?? theirs;
  const relative = relativeRms(got, want, "ipa");
  const ok = relative <= bar;
  const finite = [...got].every((value) => Number.isFinite(value));

  releaseIpaWeights(execution, uploaded);
  const stats = execution.stats();
  return {
    shape: { ...shape, featureChannels },
    bar,
    falsify,
    arms: [{ arm: "ipa update", relativeRms: Number(relative.toPrecision(4)), bar, ok }],
    everyValueFinite: finite,
    draws: stats.draws,
    programs: stats.programs,
    verdict: falsify
      ? (ok ? "VACUOUS: the arm did not notice a 1% perturbation"
        : "ok: the arm goes red on a 1% perturbation")
      : (ok && finite
        ? `ok: the update matches at ${relative.toExponential(2)}, every value finite`
        : `FAILED: relRMS ${relative.toExponential(3)} over ${bar}`
          + `${finite ? "" : ", and some values are not finite"}`),
  };
}
