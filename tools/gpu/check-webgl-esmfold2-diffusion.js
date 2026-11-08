/**
 * ESMFold2's diffusion head in WebGL2 against `cpu/esmfold2/diffusion.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-esmfold2-diffusion.js [--falsify]
 *
 * 🔴 THE WHOLE STEP, AND THE CONDITIONING'S TWO HALVES SEPARATELY, because they
 * fail separately: `z` does not depend on the noise level and `s` is nothing but
 * the noise level and `s_inputs`. The gate runs TWO noise levels and asserts
 * that the single moves between them and the pair does NOT - which is what says
 * the cache upstream keeps is sound, and what a port that cached both would
 * fail.
 *
 * 🔴 AND THE TWO TRANSITIONS ARE PACKED THE TWO DIFFERENT WAYS IN ONE MODULE:
 * the conditioning's has `a_proj` and `b_proj` as separate Linears and the
 * denoiser's `lin_swish` is one Linear split in half, gate first. Both are arms
 * here because reading either as the other "indexes one matrix at half its
 * stride" and still returns a tensor.
 *
 * 🔴 AND THE EDM PRECONDITIONING IS ASSERTED AS ARITHMETIC, NOT TRUSTED: at a
 * large noise level the answer is almost all network and at a small one almost
 * all input, so the gate runs sigma 256 and sigma 0.1 and reports both
 * coefficients. "Writing it as `x_noisy + r` runs, converges to something, and
 * is a different sampler."
 */
import {
  denoiseStep as cpuStep, diffusionConditioning as cpuConditioning,
} from "../../cpu/esmfold2/diffusion.js";
import {
  atomDecoder as cpuDecoder, inputsEmbedder as cpuEmbedder,
} from "../../cpu/esmfold2/atom-transformer.js";
import { ATOM_FEATURES } from "../../cpu/esmfold2/atom-transformer.js";
import { denoiseStep, diffusionConditioning } from "../../webgl/esmfold2/diffusion.js";
import {
  atomDecoder, inputsEmbedder,
} from "../../webgl/esmfold2/atom-transformer.js";
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
  const tokens = Number(option(args, "tokens", "6"));
  const perToken = Number(option(args, "per-token", "4"));
  const pairChannels = Number(option(args, "pair-channels", "16"));
  const tokenChannels = Number(option(args, "token-channels", "32"));
  const singleInputs = Number(option(args, "single-inputs", "28"));
  const tokenHeads = Number(option(args, "token-heads", "4"));
  const multiplier = Number(option(args, "multiplier", "2"));
  const tokenBlocks = Number(option(args, "token-blocks", "2"));
  const atomChannels = Number(option(args, "atom-channels", "32"));
  const atomHeads = Number(option(args, "atom-heads", "4"));
  const atomBlocks = Number(option(args, "atom-blocks", "2"));
  const atomHidden = Number(option(args, "atom-hidden", "64"));
  const atoms = tokens * perToken;
  const pairs = tokens * tokens;

  const random = generator(3141592);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  const transitionGroup = (channels, hidden) => ({
    normScale: noise(channels, 0.2, 1),
    normOffset: noise(channels, 0.1),
    aProjection: noise(channels * hidden, 1 / Math.sqrt(channels)),
    bProjection: noise(channels * hidden, 1 / Math.sqrt(channels)),
    outProjection: noise(hidden * channels, 0.3 / Math.sqrt(hidden)),
  });
  const adalnGroup = (channels) => ({
    singleScale: noise(channels, 0.2, 1),
    gateWeights: noise(channels * channels, 1 / Math.sqrt(channels)),
    shiftWeights: noise(channels * channels, 0.3 / Math.sqrt(channels)),
    gateBias: noise(channels, 0.3, -2),
  });
  const noiseChannels = 16;
  const conditioningWeights = {
    zInputNormScale: noise(pairChannels * 2, 0.2, 1),
    zInputNormOffset: noise(pairChannels * 2, 0.1),
    zProjection: noise(pairChannels * 2 * pairChannels, 1 / Math.sqrt(pairChannels * 2)),
    zTransitions: [transitionGroup(pairChannels, pairChannels * multiplier),
      transitionGroup(pairChannels, pairChannels * multiplier)],
    sInputNormScale: noise(singleInputs, 0.2, 1),
    sInputNormOffset: noise(singleInputs, 0.1),
    sProjection: noise(singleInputs * tokenChannels, 1 / Math.sqrt(singleInputs)),
    fourierWeights: noise(noiseChannels, 1),
    fourierOffsets: noise(noiseChannels, 0.5),
    noiseNormScale: noise(noiseChannels, 0.2, 1),
    noiseNormOffset: noise(noiseChannels, 0.1),
    noiseProjection: noise(noiseChannels * tokenChannels, 1 / Math.sqrt(noiseChannels)),
    sTransitions: [transitionGroup(tokenChannels, tokenChannels * multiplier),
      transitionGroup(tokenChannels, tokenChannels * multiplier)],
  };
  const tokenBlockWeights = () => ({
    attention: {
      ...{ adaln: adalnGroup(tokenChannels) },
      queryWeights: noise(tokenChannels * tokenChannels, 1 / Math.sqrt(tokenChannels)),
      queryBias: noise(tokenChannels, 0.1),
      kvWeights: noise(tokenChannels * tokenChannels * 2, 1 / Math.sqrt(tokenChannels)),
      gateWeights: noise(tokenChannels * tokenChannels, 1 / Math.sqrt(tokenChannels)),
      pairNormScale: noise(pairChannels, 0.2, 1),
      pairNormOffset: noise(pairChannels, 0.1),
      pairBiasWeights: noise(pairChannels * tokenHeads, 1 / Math.sqrt(pairChannels)),
      outWeights: noise(tokenChannels * tokenChannels, 0.3 / Math.sqrt(tokenChannels)),
      outGateWeights: noise(tokenChannels * tokenChannels, 1 / Math.sqrt(tokenChannels)),
      outGateBias: noise(tokenChannels, 0.3, -2),
    },
    transition: {
      adaln: adalnGroup(tokenChannels),
      swishWeights: noise(
        tokenChannels * tokenChannels * multiplier * 2, 1 / Math.sqrt(tokenChannels)),
      outWeights: noise(
        tokenChannels * multiplier * tokenChannels, 0.3 / Math.sqrt(tokenChannels)),
      outGateWeights: noise(tokenChannels * tokenChannels, 1 / Math.sqrt(tokenChannels)),
      outGateBias: noise(tokenChannels, 0.3, -2),
    },
  });
  const atomBlockWeights = () => ({
    adaln: noise(atomChannels * atomChannels * 6, 1 / Math.sqrt(atomChannels)),
    qkv: noise(atomChannels * atomChannels * 3, 1 / Math.sqrt(atomChannels)),
    attnGate: noise(atomChannels * atomChannels, 1 / Math.sqrt(atomChannels)),
    attnOut: noise(atomChannels * atomChannels, 1 / Math.sqrt(atomChannels)),
    ffnUp: noise(atomChannels * atomHidden * 2, 1 / Math.sqrt(atomChannels)),
    ffnDown: noise(atomHidden * atomChannels, 1 / Math.sqrt(atomHidden)),
  });
  const encoderWeights = {
    atomLinear: noise(ATOM_FEATURES * atomChannels, 1 / Math.sqrt(ATOM_FEATURES)),
    atomNormScale: noise(atomChannels, 0.2, 1),
    atomNormOffset: noise(atomChannels, 0.1),
    coordsLinear: noise(6 * atomChannels, 0.3),
    atomToToken: noise(atomChannels * tokenChannels, 1 / Math.sqrt(atomChannels)),
    blocks: Array.from({ length: atomBlocks }, atomBlockWeights),
  };
  const decoderWeights = {
    tokenToAtom: noise(tokenChannels * atomChannels, 1 / Math.sqrt(tokenChannels)),
    normScale: noise(atomChannels, 0.2, 1),
    normOffset: noise(atomChannels, 0.1),
    outputLinear: noise(atomChannels * 3, 0.3 / Math.sqrt(atomChannels)),
    blocks: Array.from({ length: atomBlocks }, atomBlockWeights),
  };
  const headWeights = {
    conditioning: conditioningWeights,
    stepNormScale: noise(tokenChannels, 0.2, 1),
    stepNormOffset: noise(tokenChannels, 0.1),
    singleToToken: noise(tokenChannels * tokenChannels, 1 / Math.sqrt(tokenChannels)),
    tokenBlocks: Array.from({ length: tokenBlocks }, tokenBlockWeights),
    tokenNormScale: noise(tokenChannels, 0.2, 1),
    tokenNormOffset: noise(tokenChannels, 0.1),
  };

  // The atom layout, with padding throughout: the window is over rank.
  const mask = new Float32Array(atoms).fill(1);
  for (let atom = 0; atom < atoms; atom += 1) if (atom % 7 === 5) mask[atom] = 0;
  const atomToToken = Int32Array.from({ length: atoms }, (_, i) => Math.floor(i / perToken));
  const refPos = noise(atoms * 3, 4);
  const refCharge = noise(atoms, 1);
  const refElement = Int32Array.from({ length: atoms }, (_, i) => (i * 7 + 3) % 128);
  const refAtomNameChars = Int32Array.from({ length: atoms * 4 }, (_, i) => (i * 11 + 5) % 64);
  const refSpaceUid = Float32Array.from({ length: atoms }, (_, i) => Math.floor(i / perToken));
  const headDim = atomChannels / atomHeads;
  const spatialPairs = 1;
  const uidPairs = (headDim >> 1) - 3 * spatialPairs;
  const ropeOptions = { spatialPairs, uidPairs, precision: "bf16" };

  const zTrunk = noise(pairs * pairChannels, 1);
  const relPos = noise(pairs * pairChannels, 1);
  const sInputs = noise(tokens * singleInputs, 1);
  const noisy = noise(atoms * 3, 20);

  const shape = {
    tokens, atoms, pairChannels, singleInputs, tokenChannels, multiplier,
    tokenHeads, sigmaData: 16,
    // the atom stack's own shape, which the encoder and decoder read
    channels: atomChannels, heads: atomHeads, blocks: atomBlocks, hidden: atomHidden,
    rope: ropeOptions, attentionPrecision: "f32",
  };

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
    if (typeof value === "number" || typeof value === "boolean") return value;
    return Object.fromEntries(Object.entries(value).map(
      ([key, v]) => [key, uploadTree(v, `${label}.${key}`)]));
  };
  // 🔴 THE SKEW GOES TO THE DEVICE ALONE, and it has to reach BOTH halves of the
  // conditioning: the trunk pair feeds `z` and nothing else, `s_inputs` feeds
  // `s`, and the noisy coordinates feed the atom encoder and the blend.
  const skew = (values, period) => Float32Array.from(values,
    (v, i) => (i % period < period / 2 ? v * 1.01 + 0.01 : v));
  const uploadedHead = uploadTree(headWeights, "w.ef2.head");
  const uploadedEncoder = uploadTree(encoderWeights, "w.ef2.encoder");
  const uploadedDecoder = uploadTree(decoderWeights, "w.ef2.decoder");
  const zTrunkTensor = upload(falsify ? skew(zTrunk, pairChannels) : zTrunk, "ef2.z-trunk");
  const relPosTensor = upload(relPos, "ef2.rel-pos");
  const sInputsTensor = upload(
    falsify ? skew(sInputs, singleInputs) : sInputs, "ef2.s-inputs");
  const noisyTensor = upload(falsify ? skew(noisy, 3) : noisy, "ef2.noisy");
  const features = {
    refPos: upload(refPos, "ef2.ref-pos"),
    refCharge: upload(refCharge, "ef2.ref-charge"),
    mask: upload(mask, "ef2.mask"),
    refElement: upload(Float32Array.from(refElement), "ef2.ref-element"),
    refAtomNameChars: upload(Float32Array.from(refAtomNameChars), "ef2.ref-names"),
    atomToToken: upload(Float32Array.from(atomToToken), "ef2.atom-to-token"),
  };
  const host = { refPos, refSpaceUid, mask };
  const hostFeatures = {
    refPos, refCharge, refElement, refAtomNameChars, mask, atomToToken, refSpaceUid,
  };
  const hostFourier = {
    weights: conditioningWeights.fourierWeights,
    offsets: conditioningWeights.fourierOffsets,
  };

  const arms = [];
  const failures = [];
  const separations = {};
  const report = (name, mine, theirs) => {
    const relative = relativeRms(mine, theirs, name);
    const ok = relative <= bar;
    arms.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };

  // --- the conditioning, at two noise levels ----------------------------
  const conditioningAt = (tHat) => {
    const out = diffusionConditioning(execution, {
      zTrunk: zTrunkTensor, relPos: relPosTensor, sInputs: sInputsTensor, tHat,
      shape, weights: uploadedHead.conditioning, hostFourier,
    });
    const values = {
      single: Float32Array.from(execution.read(out.single)),
      pair: Float32Array.from(execution.read(out.pair)),
    };
    execution.release(out.single);
    execution.release(out.pair);
    return values;
  };
  const high = conditioningAt(256);
  const low = conditioningAt(0.1);
  for (const [name, tHat, mine] of [["sigma 256", 256, high], ["sigma 0.1", 0.1, low]]) {
    const theirs = cpuConditioning(zTrunk, relPos, sInputs, tHat, shape,
      conditioningWeights);
    report(`the conditioning's single, ${name}`, mine.single, theirs.single);
    report(`the conditioning's pair, ${name}`, mine.pair, theirs.pair);
  }
  // 🔴 THE SINGLE MOVES WITH THE NOISE LEVEL AND THE PAIR DOES NOT, which is
  // what makes the cache sound - and a port that cached both would pass every
  // arm above and fail this one.
  separations["the single across noise levels"] = Number(
    relativeRms(low.single, high.single, "single").toPrecision(3));
  separations["the pair across noise levels"] = Number(
    relativeRms(low.pair, high.pair, "pair").toPrecision(3));
  if (!(separations["the single across noise levels"] > 1e-3)) {
    failures.push("the single does not move with the noise level");
  }
  if (separations["the pair across noise levels"] !== 0) {
    failures.push("the pair moved with the noise level, so it is not cacheable");
  }

  // --- a whole denoise step ---------------------------------------------
  const embed = (scaled) => inputsEmbedder(execution, {
    features: { ...features, noisyCoordinates: scaled },
    host, shape, weights: uploadedEncoder,
  });
  const decode = (act, embedded) => atomDecoder(execution, {
    tokenAct: act, skip: embedded.atomState, conditioning: embedded.conditioning,
    rope: embedded.rope, valid: embedded.valid, rank: embedded.rank,
    index: embedded.index, shape, weights: uploadedDecoder,
  });
  const live = [];
  const embedTracked = (scaled) => {
    const embedded = embed(scaled);
    live.push(embedded);
    return embedded;
  };
  const stepAt = (tHat, cachedPair) => {
    const out = denoiseStep(execution, {
      noisy: noisyTensor, tHat, shape, weights: uploadedHead,
      sInputs: sInputsTensor, zTrunk: zTrunkTensor, relPos: relPosTensor,
      hostFourier, embed: embedTracked, decode, cachedPair,
    });
    const values = Float32Array.from(execution.read(out.coordinates));
    execution.release(out.coordinates);
    for (const embedded of live.splice(0)) {
      for (const tensor of [embedded.atomState, embedded.conditioning,
        embedded.tokenAct, embedded.rope.cos, embedded.rope.sin, embedded.rank]) {
        execution.release(tensor);
      }
    }
    return { values, pair: out.pair, borrowed: out.pairIsBorrowed };
  };
  const cpuEncoderShim = {
    embed: (f) => cpuEmbedder(f, shape, encoderWeights),
    decode: (act, embedded) => cpuDecoder(act, embedded.atomState, embedded.conditioning,
      embedded.rope, hostFeatures, shape, decoderWeights),
  };
  for (const tHat of [256, 0.1]) {
    const mine = stepAt(tHat, undefined);
    const theirs = cpuStep(noisy, tHat, hostFeatures, sInputs, zTrunk, relPos,
      shape, headWeights, cpuEncoderShim);
    report(`a whole denoise step, sigma ${tHat}`, mine.values, theirs.coordinates);
    if (!mine.borrowed) execution.release(mine.pair);
    // 🔴 THE EDM COEFFICIENTS, REPORTED: at 256 the answer is almost all
    // network and at 0.1 almost all input.
    separations[`sigma ${tHat}: keep`] = Number(
      (16 * 16 / (16 * 16 + tHat * tHat)).toPrecision(4));
    separations[`sigma ${tHat}: take`] = Number(
      ((16 * tHat) / Math.sqrt(16 * 16 + tHat * tHat)).toPrecision(4));
  }

  // 🔴 AND A CACHED PAIR MUST CHANGE NOTHING, which is exact: the same draws on
  // the same inputs, so the step's output is bit-identical or the cache is not a
  // cache.
  {
    const first = stepAt(256, undefined);
    const second = stepAt(0.1, first.pair);
    const plain = stepAt(0.1, undefined);
    const difference = relativeRms(second.values, plain.values, "cache");
    arms.push({
      arm: "a cached pair conditioning, against recomputing it",
      relativeRms: difference, bar: 0, ok: difference === 0,
      falsifiable: false,
      reason: "both arms are this port's own step, so a skewed input moves both;"
        + " what it asserts is that the cache is a no-op, which is exact or wrong",
    });
    if (difference !== 0) {
      failures.push(`a cached pair conditioning moved the answer by ${difference}`);
    }
    execution.release(first.pair);
    if (!second.borrowed) execution.release(second.pair);
    if (!plain.borrowed) execution.release(plain.pair);
  }

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, atoms, pairChannels, tokenChannels, singleInputs, tokenHeads,
      tokenBlocks, atomChannels, atomHeads, atomBlocks, multiplier,
      paddedAtoms: atoms - mask.reduce((a, b) => a + b, 0),
    },
    bar,
    falsify,
    arms,
    separations,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms against the f64 reference, two noise levels`
        : `FAILED: ${failures.join("; ")}`),
  };
}
