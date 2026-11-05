/**
 * AF3's diffusion conditioning in WebGL2 against `cpu/af3/diffusion/diffusion.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-diffusion-conditioning.js [--falsify]
 *
 * 🔴 THREE PAIR SHAPES, AND THE LAYERNORM'S LENGTH IS WHAT SAYS WHICH. AF3
 * concatenates the trunk pair with the RAW relative encoding and normalises the
 * lot; OpenDDE compresses each to the pair width separately; protenix2 and
 * boltz2 project the relative encoding alone. "The joint LayerNorm over the
 * widened concatenation couples the two terms, so this is a different function
 * and not a re-association." All three are arms.
 *
 * 🔴 AND THE REASON THIS PORT REFUSES RATHER THAN COMPUTING IS WRITTEN IN THE
 * REFERENCE. boltz2's wrong width "is LONGER than its scale, so the LayerNorm
 * read past the end and every one of 73728 elements came out NaN. protenix2's
 * 395 is SHORTER than its 512, so it read a prefix, stayed finite, and the GPU
 * made the identical mistake - so check-af3-diffusion-conditioning compared two
 * wrong computations and passed at 3.20e-7. A checker agreeing with itself is
 * the failure this repository keeps finding, and the only reason it surfaced is
 * that a second model rounded the other way." So a `--wrong-width` arm asserts
 * this port REFUSES where the width and the norm disagree, rather than reading a
 * prefix and being quietly wrong.
 *
 * 🔴 AND THE SINGLE IT READS IS NOT THE WIDTH IT WRITES: "AF3's projection is
 * [831, 384] and its trunk single is also 384, so `seqChannels` served as both
 * and this line was right by coincidence. boltz2's is [768, 768] - 768 out, 384
 * in." The arms give the two different values.
 *
 * 🔴 AND THE 139 RELATIVE COLUMNS ARE MATERIALISED HERE, where `trunk/embedder.js`
 * folds them into a projection - because AF3 concatenates them with the trunk
 * pair and normalises the lot, so they have to exist as columns. Two kernels for
 * one encoding, and the cross-check is what says they agree about the bins.
 */
import {
  diffusionConditioning as cpuConditioning,
} from "../../cpu/af3/diffusion/diffusion.js";
import {
  diffusionConditioning, noiseEmbedding, scalings,
} from "../../webgl/af3/diffusion/conditioning.js";
import { RELATIVE_WIDTH, tokenIdentity } from "../../webgl/af3/trunk/embedder.js";
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
  const tokens = Number(option(args, "tokens", "5"));
  const pairChannels = Number(option(args, "pair-channels", "16"));
  const trunkPairChannels = Number(option(args, "trunk-pair-channels", "24"));
  const seqChannels = Number(option(args, "seq-channels", "16"));
  const trunkSingleChannels = Number(option(args, "trunk-single-channels", "12"));
  // 🔴 WIDE ENOUGH FOR THE PADDING TO LAND INSIDE IT. The openfold3 inserts sit
  // at `trunkSingle + 31` and `trunkSingle + 63` - one after the restype block
  // and one after the profile block - so a target feature narrower than about 64
  // puts them past the end of the concatenation, and the mapping then reads two
  // columns beyond `targetFeat` and returns NaN. The real model is 384 + 447,
  // where they land at 415 and 447.
  const targetFeatWidth = Number(option(args, "target-feat-width", "68"));
  const noiseChannels = 8;
  const noiseLevel = Number(option(args, "noise", "4.5"));
  // 🔴 THE openfold3 LINEAGE INSERTS TWO ZERO COLUMNS into the single
  // concatenation, one after the restype block and one after the profile block.
  // Neither side has a default for it, and the LayerNorm's length is what
  // asserts the dialect and the weights agree.
  const padDna = flag(args, "pad-unknown-dna");
  const pairs = tokens * tokens;

  const random = generator(1414213);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const transitionGroup = (channels) => ({
    ffwLayerNormScale: noise(channels, 0.2, 1),
    ffwLayerNormOffset: noise(channels, 0.1),
    ffwTransition1: noise(channels * channels * 2 * 2, 1 / Math.sqrt(channels)),
    ffwTransition2: noise(channels * 2 * channels, 0.3 / Math.sqrt(channels * 2)),
  });

  const chains = 2;
  const perChain = Math.ceil(tokens / chains);
  const residueIndex = new Int32Array(tokens);
  const tokenIndex = new Int32Array(tokens);
  const asymId = new Int32Array(tokens);
  const entityId = new Int32Array(tokens);
  const symId = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) {
    const chain = Math.min(Math.floor(t / perChain), chains - 1);
    asymId[t] = chain;
    entityId[t] = 0;
    symId[t] = chain;
    residueIndex[t] = t - chain * perChain;
    tokenIndex[t] = residueIndex[t];
  }
  const features = { residueIndex, tokenIndex, asymId, entityId, symId };
  const trunkSingle = noise(tokens * trunkSingleChannels, 1);
  const trunkPair = noise(pairs * trunkPairChannels, 1);
  const targetFeat = noise(tokens * targetFeatWidth, 1);
  const fourierWeight = noise(noiseChannels, 1);
  const fourierBias = noise(noiseChannels, 0.5);

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
    return Object.fromEntries(Object.entries(value).map(
      ([key, v]) => [key, uploadTree(v, `${label}.${key}`)]));
  };

  const arms = [];
  const failures = [];
  const refusals = [];
  /** @type {{name: string, extra: object, width: number}[]} */
  const CASES = [
    {
      name: "AF3 (raw relative encoding)",
      width: trunkPairChannels + RELATIVE_WIDTH,
      extra: {},
    },
    {
      name: "OpenDDE (both compressed)",
      width: 2 * pairChannels,
      extra: {
        zTrunkNormScale: noise(trunkPairChannels, 0.2, 1),
        zTrunkNormOffset: noise(trunkPairChannels, 0.1),
        zTrunkProjection: noise(
          trunkPairChannels * pairChannels, 1 / Math.sqrt(trunkPairChannels)),
        relpeProjection: noise(
          RELATIVE_WIDTH * pairChannels, 1 / Math.sqrt(RELATIVE_WIDTH)),
      },
    },
    {
      name: "protenix2/boltz2 (projected relative encoding)",
      width: trunkPairChannels + pairChannels,
      extra: {
        relpeProjection: noise(
          RELATIVE_WIDTH * pairChannels, 1 / Math.sqrt(RELATIVE_WIDTH)),
      },
    },
  ];

  const padColumns = padDna ? 2 : 0;
  const singleWidth = trunkSingleChannels + targetFeatWidth + padColumns;
  for (const { name, extra, width } of CASES) {
    const weights = {
      pairChannels, seqChannels, trunkPairChannels, trunkSingleChannels,
      targetFeatWidth, relativeWidth: RELATIVE_WIDTH,
      pairCondInitialNormScale: noise(width, 0.2, 1),
      pairCondInitialNormOffset: noise(width, 0.1),
      pairCondInitialProjection: noise(width * pairChannels, 1 / Math.sqrt(width)),
      pairTransitions: [transitionGroup(pairChannels), transitionGroup(pairChannels)],
      singleCondInitialNormScale: noise(singleWidth, 0.2, 1),
      singleCondInitialNormOffset: noise(singleWidth, 0.1),
      singleCondInitialProjection: noise(
        singleWidth * seqChannels, 1 / Math.sqrt(singleWidth)),
      singleCondInitialProjectionBias: noise(seqChannels, 0.1),
      noiseEmbeddingInitialNormScale: noise(noiseChannels, 0.2, 1),
      noiseEmbeddingInitialNormOffset: noise(noiseChannels, 0.1),
      noiseEmbeddingInitialProjection: noise(
        noiseChannels * seqChannels, 1 / Math.sqrt(noiseChannels)),
      singleTransitions: [transitionGroup(seqChannels), transitionGroup(seqChannels)],
      fourierWeight, fourierBias,
      ...extra,
    };
    const uploaded = uploadTree(weights, `w.af3.diffcond.${name}`);
    const out = diffusionConditioning(execution, {
      trunkSingle: upload(trunkSingle, "af3.diffcond.trunk-single"),
      trunkPair: upload(falsify
        ? Float32Array.from(trunkPair,
          (v, i) => (i % trunkPairChannels < trunkPairChannels / 2 ? v * 1.01 + 0.01 : v))
        : trunkPair, "af3.diffcond.trunk-pair"),
      // 🔴 THE TARGET FEATURE NEEDS ITS OWN SKEW: the trunk pair reaches the
      // pair track and nothing else, so with one control the three single arms
      // reported "3 of 6". The fifth time in this campaign that a control had to
      // be widened rather than trusted.
      targetFeat: upload(falsify
        ? Float32Array.from(targetFeat,
          (v, i) => (i % targetFeatWidth < targetFeatWidth / 2 ? v * 1.01 + 0.01 : v))
        : targetFeat, "af3.diffcond.target"),
      identity: upload(tokenIdentity(features, tokens), "af3.diffcond.identity"),
      noiseLevel, tokens, weights: uploaded, relativeWidth: RELATIVE_WIDTH,
      fourierWeight, fourierBias, padUnknownDna: padDna,
    });
    const mine = {
      single: Float32Array.from(execution.read(out.single)),
      pair: Float32Array.from(execution.read(out.pair)),
    };
    execution.release(out.single);
    execution.release(out.pair);

    const expected = cpuConditioning({
      tokens, trunkSingle, trunkPair, targetFeat, noiseLevel, features,
      dialect: { padSingleCondUnknownDna: padDna },
    }, weights);
    for (const track of ["single", "pair"]) {
      const armName = `${name}, ${track}`;
      const relative = relativeRms(mine[track], expected[track], armName);
      const ok = relative <= bar;
      arms.push({ arm: armName, relativeRms: Number(relative.toPrecision(4)), bar, ok });
      if (!ok) failures.push(`${armName}: relRMS ${relative.toExponential(3)} over ${bar}`);
    }
  }

  // 🔴 THE REFUSAL ARM: a norm whose length disagrees with the derived width.
  // The reference computes anyway and reads a prefix; this port raises, because
  // "a short read stays finite and two implementations can make the same
  // mistake".
  {
    const wrongWidth = trunkPairChannels + RELATIVE_WIDTH + 8;
    const weights = uploadTree({
      pairChannels, seqChannels, trunkPairChannels, trunkSingleChannels,
      targetFeatWidth,
      pairCondInitialNormScale: noise(wrongWidth, 0.2, 1),
      pairCondInitialNormOffset: noise(wrongWidth, 0.1),
      pairCondInitialProjection: noise(wrongWidth * pairChannels, 0.1),
      pairTransitions: [transitionGroup(pairChannels)],
      singleCondInitialNormScale: noise(singleWidth, 0.2, 1),
      singleCondInitialNormOffset: noise(singleWidth, 0.1),
      singleCondInitialProjection: noise(singleWidth * seqChannels, 0.1),
      noiseEmbeddingInitialNormScale: noise(noiseChannels, 0.2, 1),
      noiseEmbeddingInitialNormOffset: noise(noiseChannels, 0.1),
      noiseEmbeddingInitialProjection: noise(noiseChannels * seqChannels, 0.1),
      singleTransitions: [transitionGroup(seqChannels)],
    }, "w.af3.diffcond.wrong");
    let message = null;
    try {
      diffusionConditioning(execution, {
        trunkSingle: upload(trunkSingle, "af3.diffcond.ts2"),
        trunkPair: upload(trunkPair, "af3.diffcond.tp2"),
        targetFeat: upload(targetFeat, "af3.diffcond.tf2"),
        identity: upload(tokenIdentity(features, tokens), "af3.diffcond.id2"),
        noiseLevel, tokens, weights, relativeWidth: RELATIVE_WIDTH,
        fourierWeight, fourierBias, padUnknownDna: padDna,
      });
    } catch (error) {
      message = String(error.message);
    }
    const ok = message !== null && message.includes("LayerNorm scale");
    refusals.push({
      refusal: "a pair concatenation whose width disagrees with its LayerNorm",
      ok, said: message ?? "did not throw",
    });
    if (!ok) {
      failures.push("a mismatched pair width was NOT refused - which is how a short read"
        + " stays finite and two implementations make the same mistake");
    }
  }

  // The two host-side helpers, which have no GPU part at all.
  {
    const embedded = noiseEmbedding(noiseLevel / 16, fourierWeight, fourierBias);
    let ok = embedded.every((v) => Number.isFinite(v) && Math.abs(v) <= 1);
    const s = scalings(noiseLevel);
    // 🔴 THE IDENTITIES AT sigma = sigma_data, which are the only ones that
    // pin all three coefficients: skip is exactly 1/2, out is
    // sigma_data / sqrt(2) and in is 1 / (sigma_data * sqrt(2)). My first
    // version asserted `out == 16 / sqrt(512)` and was off by a factor of
    // sixteen - the assertion was wrong, not the code, which is why it is
    // written from the formula here rather than from a remembered number.
    const equal = scalings(16);
    ok = ok && Math.abs(equal.skip - 0.5) < 1e-6
      && Math.abs(equal.out - 16 / Math.SQRT2) < 1e-6
      && Math.abs(equal.in - 1 / (16 * Math.SQRT2)) < 1e-9
      && s.skip > 0 && s.out > 0 && s.in > 0;
    refusals.push({
      refusal: "the Fourier embedding is bounded and the scalings agree at sigma_data",
      ok, said: ok ? "ok" : `skip ${equal.skip}, out ${equal.out}`,
    });
    if (!ok) failures.push("the host-side noise helpers are out of range");
  }

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const red = arms.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, pairChannels, trunkPairChannels, seqChannels, trunkSingleChannels,
      targetFeatWidth, noiseChannels, relativeWidth: RELATIVE_WIDTH,
    },
    noiseLevel,
    padUnknownDna: padDna,
    bar,
    falsify,
    arms,
    refusals,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms over three pair shapes, ${refusals.length} refusals`
        : `FAILED: ${failures.join("; ")}`),
  };
}
