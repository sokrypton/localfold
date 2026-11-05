/**
 * The input embedder against the WGSL one - three embeddings and the recycle.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-input-embedder.js [--falsify]
 *
 * 🔴 FOUR CONVENTIONS, EVERY ONE OF WHICH PRODUCES A PLAUSIBLE EMBEDDING WHEN
 * WRONG:
 *
 *   - the recycled MSA enters on ROW ZERO only, not on every row
 *   - the pseudo-beta is CB (atom 3) except for GLYCINE (aatype 7), which uses
 *     CA (atom 1)
 *   - a pseudo-beta distance BELOW the first bin's lower edge lands in NO bin,
 *     and the last bin's upper edge is `maxBin + binWidth` rather than infinity
 *   - the relative position is CLAMPED, so every separation past 32 shares one
 *     bin - which is what lets a chain break be expressed as an index jump
 *
 * The arms include a glycine at a known residue and residue indices with a jump
 * of more than 32, because neither branch is reached by a plain sequence - and a
 * gate that never reaches a branch says nothing about it.
 *
 * 🔴 AND THE PERTURBATION HAS TO REACH ALL THREE EMBEDDINGS, which the first
 * version did not: it skewed the TARGET features only, and the extra-MSA
 * embedding does not read them - that arm came back identical and the inverse
 * verdict said "only 2 of 3". The extra-MSA path takes a code, a deletion flag
 * and a deletion value and nothing else, so the control skews the deletion value
 * too.
 */
import { InputEmbedderGpu } from "../../webgpu/af2/evoformer/input-embedder.js";
import { setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import {
  inputEmbedder, releaseInputEmbedderWeights, uploadInputEmbedderWeights,
} from "../../webgl/af2/evoformer/input-embedder.js";
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
  const bar = Number(option(args, "bar", "1e-6"));
  const shape = {
    length: Number(option(args, "length", "9")),
    msaSequences: 3,
    extraSequences: 4,
    targetChannels: 22,
    msaFeatureChannels: 49,
    msaChannels: 32,
    pairChannels: 16,
    extraMsaChannels: 16,
  };
  const {
    length, msaSequences, extraSequences, targetChannels, msaFeatureChannels,
    msaChannels, pairChannels, extraMsaChannels,
  } = shape;

  const random = generator(8086);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const integers = (count, limit) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = Math.floor(random() * limit);
    return out;
  };

  const aatype = integers(length, 21);
  // ...with a glycine planted, so the CA branch of the pseudo-beta is reached.
  aatype[Math.floor(length / 2)] = 7;
  // ...and an index jump past the clamp, which a plain sequence never produces.
  const residueIndex = new Float32Array(length);
  for (let residue = 0; residue < length; residue += 1) {
    residueIndex[residue] = residue + (residue > length / 2 ? 100 : 0);
  }

  const input = {
    ...shape,
    targetFeatures: noise(length * targetChannels, 1),
    msaFeatures: noise(msaSequences * length * msaFeatureChannels, 1),
    extraMsa: integers(extraSequences * length, 23),
    extraHasDeletion: integers(extraSequences * length, 2),
    extraDeletionValue: noise(extraSequences * length, 0.5, 0.5),
    previousMsaFirstRow: noise(length * msaChannels, 0.5),
    previousPair: noise(length * length * pairChannels, 0.5),
    // Positions spread widely enough that the distogram reaches several bins,
    // including distances under the first edge.
    previousPositions: noise(length * 37 * 3, 12),
    aatype,
    residueIndex,
    weights: {
      preprocess1dWeight: noise(targetChannels * msaChannels, 1 / Math.sqrt(targetChannels)),
      preprocess1dBias: noise(msaChannels, 0.1),
      preprocessMsaWeight: noise(msaFeatureChannels * msaChannels,
        1 / Math.sqrt(msaFeatureChannels)),
      preprocessMsaBias: noise(msaChannels, 0.1),
      leftSingleWeight: noise(targetChannels * pairChannels, 1 / Math.sqrt(targetChannels)),
      leftSingleBias: noise(pairChannels, 0.1),
      rightSingleWeight: noise(targetChannels * pairChannels, 1 / Math.sqrt(targetChannels)),
      rightSingleBias: noise(pairChannels, 0.1),
      previousPositionWeight: noise(15 * pairChannels, 0.3),
      previousPositionBias: noise(pairChannels, 0.1),
      previousMsaNormScale: noise(msaChannels, 0.2, 1),
      previousMsaNormOffset: noise(msaChannels, 0.1),
      previousPairNormScale: noise(pairChannels, 0.2, 1),
      previousPairNormOffset: noise(pairChannels, 0.1),
      relativePositionWeight: noise(65 * pairChannels, 0.3),
      relativePositionBias: noise(pairChannels, 0.1),
      extraMsaWeight: noise(25 * extraMsaChannels, 0.3),
      extraMsaBias: noise(extraMsaChannels, 0.1),
    },
  };

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const uploaded = uploadInputEmbedderWeights(execution, input.weights);
  const upload = (values, label) => execution.upload(values, label);
  const ours = inputEmbedder(execution, {
    ...shape,
    targetFeatures: upload(falsify
      ? Float32Array.from(input.targetFeatures,
        (v, i) => (i % targetChannels < targetChannels / 2 ? v * 1.01 + 0.01 : v))
      : input.targetFeatures, "target"),
    msaFeatures: upload(input.msaFeatures, "msa-features"),
    extraMsa: upload(input.extraMsa, "extra-codes"),
    extraHasDeletion: upload(input.extraHasDeletion, "has-deletion"),
    extraDeletionValue: upload(falsify
      ? Float32Array.from(input.extraDeletionValue, (v) => v * 1.01 + 0.01)
      : input.extraDeletionValue, "deletion-value"),
    previousMsaFirstRow: upload(input.previousMsaFirstRow, "previous-msa"),
    previousPair: upload(input.previousPair, "previous-pair"),
    previousPositions: upload(input.previousPositions, "previous-positions"),
    aatype: upload(aatype, "aatype"),
    residueIndex: upload(residueIndex, "residue-index"),
    weights: uploaded,
  });

  setHalfPrecision(device, false);
  const theirs = await new InputEmbedderGpu(device).run(input);

  const rows = [];
  const failures = [];
  const arm = (name, got, want) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };
  arm("msa", execution.read(ours.msa), theirs.msa);
  arm("extra msa", execution.read(ours.extraMsa), theirs.extraMsa);
  // The reference's name for it: the template stack adds to this afterwards.
  arm("pair without templates",
    execution.read(ours.pairWithoutTemplates), theirs.pairWithoutTemplates);

  releaseInputEmbedderWeights(execution, uploaded);
  const red = rows.filter((row) => !row.ok).length;
  return {
    shape,
    bar,
    falsify,
    glycineAt: Math.floor(length / 2),
    indexJump: 100,
    arms: rows,
    verdict: falsify
      ? (red === rows.length
        ? `ok: all ${rows.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${rows.length} arms noticed a 1% perturbation`)
      : (failures.length === 0
        ? `ok: ${rows.length} arms`
        : `FAILED: ${failures.join("; ")}`),
  };
}
