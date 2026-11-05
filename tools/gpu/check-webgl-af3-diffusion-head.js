/**
 * AF3's whole diffusion head - one denoising step - in WebGL2 against
 * `cpu/af3/diffusion/diffusion.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-diffusion-head.js [--falsify]
 *
 * This is the last piece of the denoiser: the conditioning, the atom encoder,
 * the 24-block token transformer, the atom decoder and the blend, composed. Each
 * part has its own gate; what only this one can see is the WIRING between them,
 * and the reference's header says exactly where that goes wrong.
 *
 * 🔴 THE OUTPUT IS A BLEND, NOT A PREDICTION, AND THAT IS AN ARM RATHER THAN A
 * COMMENT. "AF3 returns `skip * positions_noisy + out * update`... Returning the
 * update alone type-checks, runs, and produces a structure that is wrong in a way
 * that looks like a bad model rather than a bug." So the gate runs THREE noise
 * levels - 0.02, 4.5 and 160 - and asserts at each that the output differs from
 * both halves taken alone. At sigma 0.02 the skip coefficient is 0.9999 and the
 * update's is 0.02, so a head that returned the update alone would be red at low
 * noise and nearly right at high; one noise level cannot see that.
 *
 * 🔴 AND THE TRUNK'S SINGLE CONDITIONS THE ATOMS WHILE THE HEAD'S OWN
 * CONDITIONING CONDITIONS THE TOKEN TRANSFORMER - "they are different tensors and
 * AF3 uses both", and they are deliberately DIFFERENT WIDTHS here (12 against 16)
 * so that swapping them cannot be a shape-compatible mistake in this shape even
 * if it is one in the real model.
 *
 * 🔴 AND THE DECODER'S SKIP CONNECTION IS THE ENCODER'S OUTPUT, NOT ITS INPUT,
 * which the `encoder.skipConnection` tap is what localises: the reference records
 * it, and the decoder reading the encoder's INPUT instead would leave every tap
 * before the decoder green.
 *
 * 🔴 AND THE POSITION UPDATE IS THREE CHANNELS, through `linear`'s ragged-output
 * path. Projecting four and dropping one reads a weight column that does not
 * exist - the tensor is `channels * 3` floats - and the first version of this
 * port did exactly that.
 *
 * 🔴 AND THE LAYOUT IS SYNTHETIC AND IRREGULAR, the limitation
 * `check-webgl-af3-atom-encoder.js` states: "a hand-built layout would be regular
 * in ways a real one is not, and the padding is where the interesting
 * disagreements live". The real batch's gathers are not on this box.
 */
import {
  atomCrossAttentionEncoder as cpuEncoder,
} from "../../cpu/af3/diffusion/atom-encoder.js";
import { diffusionHead as cpuHead, scalings } from "../../cpu/af3/diffusion/diffusion.js";
import { diffusionHead } from "../../webgl/af3/diffusion/head.js";
import { RELATIVE_WIDTH, tokenIdentity } from "../../webgl/af3/trunk/embedder.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { syntheticHead } from "./af3-synthetic-head.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "5e-6"));
  const padDna = flag(args, "pad-unknown-dna");
  // 🔴 ONE BUILDER, SHARED WITH THE SAMPLER GATE - see af3-synthetic-head.js,
  // which is also where the square-atom-width refusal lives: the f64 reference's
  // `adaptiveZeroInit` reads the gathered attention at `channels`, so
  // `--channels=12` with two heads of eight is the reference being asked the
  // wrong question (1.2e-2 on `encoder.stackOut` with every earlier tap exact).
  const fixture = syntheticHead({
    tokens: Number(option(args, "tokens", "4")),
    dense: Number(option(args, "dense", "4")),
    subsets: Number(option(args, "subsets", "2")),
    queries: Number(option(args, "queries", "8")),
    keys: Number(option(args, "keys", "12")),
    channels: Number(option(args, "channels", "16")),
    pairChannels: Number(option(args, "pair-channels", "8")),
    perTokenChannels: Number(option(args, "per-token", "24")),
    seqChannels: Number(option(args, "seq-channels", "16")),
    diffusionPairChannels: Number(option(args, "cond-pair-channels", "16")),
    trunkSingleChannels: Number(option(args, "trunk-single-channels", "12")),
    trunkPairChannels: Number(option(args, "trunk-pair-channels", "8")),
    targetFeatWidth: Number(option(args, "target-feat-width", "68")),
    blocks: Number(option(args, "blocks", "2")),
    superBlocks: Number(option(args, "super-blocks", "2")),
    perSuper: Number(option(args, "per-super", "2")),
    padDna,
  });
  const { tokens, dense, subsets, queries, keys } = fixture.shape;
  const {
    channels, pairChannels, perTokenChannels, seqChannels, diffusionPairChannels,
    trunkSingleChannels, trunkPairChannels, targetFeatWidth, heads, dimension,
    blocks, superBlocks, perSuper,
  } = fixture.widths;
  const {
    atomMask, refSpaceUid, features, tokenAtomsToQueries, queriesToKeys,
    queriesToTokenAtoms, tokensToQueries, tokensToKeys,
  } = fixture.layout;
  const {
    conditioning, refPos, trunkSingle, trunkPair, targetFeat, positionsNoisy, seqMask,
  } = fixture.values;
  const { weights: headWeights, fourierWeight, fourierBias } = fixture;
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;
  const atoms = tokens * dense;
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploadTree = (value, label) => {
    if (value instanceof Float32Array) return upload(value, label);
    if (Array.isArray(value)) return value.map((v, i) => uploadTree(v, `${label}.${i}`));
    if (typeof value === "number") return value;
    if (typeof value === "boolean") return value;
    return Object.fromEntries(Object.entries(value).map(
      ([key, v]) => [key, uploadTree(v, `${label}.${key}`)]));
  };
  const uploaded = uploadTree(headWeights, "w.af3.head");

  // 🔴 FOUR SKEWS, BECAUSE NO ONE OF THEM REACHES EVERY ARM. The per-atom
  // conditioning reaches the atom stack and not the conditioning module; the
  // trunk pair reaches the pair track and not the single one (which read "3 of
  // 6" on the conditioning's own gate with one control); the target feature
  // reaches the single track; and only the noisy positions reach `scaled
  // positions` and the blend. Fed to the GPU alone - a control that feeds both
  // sides is not a control.
  const skew = (values, period) => Float32Array.from(values,
    (v, i) => (i % period < period / 2 ? v * 1.01 + 0.01 : v));
  const inputs = {
    conditioning: upload(
      falsify ? skew(conditioning, channels) : conditioning, "af3.head.cond"),
    atomMask: upload(atomMask, "af3.head.atom-mask"),
    seqMask: upload(seqMask, "af3.head.seq-mask"),
    refPos: upload(refPos, "af3.head.ref-pos"),
    refSpaceUid: upload(refSpaceUid, "af3.head.ref-uid"),
    trunkSingle: upload(trunkSingle, "af3.head.trunk-single"),
    trunkPair: upload(
      falsify ? skew(trunkPair, trunkPairChannels) : trunkPair, "af3.head.trunk-pair"),
    targetFeat: upload(
      falsify ? skew(targetFeat, targetFeatWidth) : targetFeat, "af3.head.target"),
    positionsNoisy: upload(
      falsify ? skew(positionsNoisy, 3) : positionsNoisy, "af3.head.noisy"),
    identity: upload(tokenIdentity(features, tokens), "af3.head.identity"),
  };
  const uploadGather = (g, name) => ({
    indices: upload(Float32Array.from(g.indices), `${name}.indices`),
    mask: upload(g.mask, `${name}.mask`),
  });
  const gathers = {
    tokenAtomsToQueries: uploadGather(tokenAtomsToQueries, "af3.head.ta2q"),
    queriesToKeys: uploadGather(queriesToKeys, "af3.head.q2k"),
    queriesToTokenAtoms: uploadGather(queriesToTokenAtoms, "af3.head.q2ta"),
    tokensToQueries: uploadGather(tokensToQueries, "af3.head.t2q"),
    tokensToKeys: uploadGather(tokensToKeys, "af3.head.t2k"),
  };

  const arms = [];
  const failures = [];
  const blendChecks = [];
  // 🔴 THREE NOISE LEVELS: at 0.02 the skip coefficient is 0.9999 and the
  // update's 0.02, at 160 they are 0.0099 and 15.9. A head that returned the
  // update alone would be red at one end and nearly right at the other.
  const LEVELS = [0.02, 4.5, 160].map(Number);
  const stageLevel = 4.5;
  for (const noiseLevel of LEVELS) {
    const ourStages = {};
    const theirStages = {};
    const out = diffusionHead(execution, {
      ...inputs,
      gathers,
      noiseLevel,
      shape: { tokens, dense, subsets, queries, keys },
      weights: uploaded,
      relativeWidth: RELATIVE_WIDTH,
      fourierWeight, fourierBias,
      padUnknownDna: padDna,
      maskPaddedKeys: false,
      keysUidFromQueriesLayout: false,
      keyMaskedAtomAttention: false,
      chainedAtomLayerNorm: false,
      diffusionNoResidual: false,
      pairNormPerBlock: false,
      maskAtomActPerBlock: false,
      onStage: noiseLevel === stageLevel
        ? (stage, tensor) => { ourStages[stage] = Float32Array.from(execution.read(tensor)); }
        : undefined,
    });
    const mine = Float32Array.from(execution.read(out));
    execution.release(out);

    const dialect = {
      maskPaddedKeys: false, keysUidFromQueriesLayout: false,
      padSingleCondUnknownDna: padDna,
    };
    const expected = cpuHead({
      shape: { tokens, dense, subsets, queries, keys },
      dialect, features,
      positionsNoisy, noiseLevel, atomMask, seqMask,
      trunkSingle, trunkPair, targetFeat,
      conditioning, refPos, refSpaceUid,
      tokenAtomsToQueries, queriesToKeys, queriesToTokenAtoms,
      tokensToQueries, tokensToKeys,
    }, headWeights, cpuEncoder,
    noiseLevel === stageLevel
      ? (stage, values) => { theirStages[stage] = Float32Array.from(values); }
      : undefined);

    const armName = `sigma ${noiseLevel}`;
    const relative = relativeRms(mine, expected, armName);
    const ok = relative <= bar;
    arms.push({ arm: armName, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${armName}: relRMS ${relative.toExponential(3)} over ${bar}`);

    // 🔴 THE BLEND, ASSERTED RATHER THAN COMMENTED. Both halves taken alone must
    // be far from the answer, at every level.
    const scale = scalings(noiseLevel);
    const update = theirStages["decoder"] ?? null;
    if (noiseLevel === stageLevel && update !== null) {
      const updateOnly = new Float32Array(mine.length);
      const skipOnly = new Float32Array(mine.length);
      for (let atom = 0; atom < atoms; atom += 1) {
        for (let axis = 0; axis < 3; axis += 1) {
          const index = atom * 3 + axis;
          updateOnly[index] = scale.out * update[index] * atomMask[atom];
          skipOnly[index] = scale.skip * positionsNoisy[index] * atomMask[atom];
        }
      }
      for (const [half, values] of [["the update alone", updateOnly],
        ["the noisy input alone", skipOnly]]) {
        const distance = relativeRms(mine, values, half);
        const moves = distance > 1e-2;
        blendChecks.push({ half, relativeRms: Number(distance.toPrecision(3)), moves });
        if (!moves) {
          failures.push(`the blend is indistinguishable from ${half}: ${distance}`);
        }
      }
    }
    blendChecks.push({
      sigma: noiseLevel,
      skip: Number(scale.skip.toPrecision(4)),
      out: Number(scale.out.toPrecision(4)),
      in: Number(scale.input.toPrecision(4)),
    });

    if (noiseLevel !== stageLevel) continue;
    // 🔴 THE STAGE TAPS, which is what turns "1.7e-3 somewhere" into a line
    // number: the reference records nine seams and this port now records the
    // same ones.
    for (const stage of Object.keys(ourStages)) {
      const theirs = theirStages[stage];
      if (theirs === undefined) continue;
      const stageRms = relativeRms(ourStages[stage], theirs, `stage.${stage}`);
      // Three taps are pure index gathers of the LAYOUT and cannot move under a
      // perturbation of any input value - declaring that beats inventing a
      // control that appears to move them.
      const indexOnly = ["encoder.queriesUid", "encoder.keysUid", "encoder.keysMask",
        "encoder.queriesMask"].includes(stage);
      arms.push({
        arm: `stage ${stage}`,
        relativeRms: Number(stageRms.toPrecision(4)), bar, ok: stageRms <= bar,
        ...(indexOnly ? {
          falsifiable: false,
          reason: "a gather of the layout alone; no input value reaches it",
        } : {}),
      });
      if (!(stageRms <= bar)) {
        failures.push(`stage ${stage}: relRMS ${stageRms.toExponential(3)} over ${bar}`);
      }
    }
  }

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, dense, subsets, queries, keys, channels, pairChannels,
      perTokenChannels, seqChannels, diffusionPairChannels,
      trunkSingleChannels, trunkPairChannels, targetFeatWidth,
      heads, dimension, blocks, superBlocks, perSuper,
    },
    noiseLevels: LEVELS,
    padUnknownDna: padDna,
    paddedAtoms: atoms - atomMask.reduce((a, b) => a + b, 0),
    paddedKeySlots: keyRows - queriesToKeys.mask.reduce((a, b) => a + b, 0),
    layout: "synthetic and irregular; the real batch's gathers are not on this box",
    bar,
    falsify,
    arms,
    blend: blendChecks,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms over three noise levels against the f64 reference`
        : `FAILED: ${failures.join("; ")}`),
  };
}
