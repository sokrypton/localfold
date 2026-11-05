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
  const tokens = Number(option(args, "tokens", "4"));
  const dense = Number(option(args, "dense", "4"));
  const subsets = Number(option(args, "subsets", "2"));
  const queries = Number(option(args, "queries", "8"));
  const keys = Number(option(args, "keys", "12"));
  const channels = Number(option(args, "channels", "16"));
  const pairChannels = Number(option(args, "pair-channels", "8"));
  // The transformer's activation width, which is also what the encoder
  // aggregates to and what the decoder broadcasts from.
  const perTokenChannels = Number(option(args, "per-token", "24"));
  // ...and the conditioning's single width, which is the transformer's
  // CONDITIONING width. Deliberately different from the activation's: assuming
  // square reads the diffusion weights at the wrong stride.
  const seqChannels = Number(option(args, "seq-channels", "16"));
  const diffusionPairChannels = Number(option(args, "cond-pair-channels", "16"));
  const trunkSingleChannels = Number(option(args, "trunk-single-channels", "12"));
  const trunkPairChannels = Number(option(args, "trunk-pair-channels", "8"));
  const targetFeatWidth = Number(option(args, "target-feat-width", "68"));
  const heads = 2;
  const dimension = 8;
  const blocks = Number(option(args, "blocks", "2"));
  const superBlocks = Number(option(args, "super-blocks", "2"));
  const perSuper = Number(option(args, "per-super", "2"));
  const noiseChannels = 8;
  const padDna = flag(args, "pad-unknown-dna");
  // 🔴 THE ATOM STACK'S WIDTH MUST BE SQUARE, BECAUSE THE REFERENCE'S
  // `adaptiveZeroInit` READS THE GATHERED ATTENTION AT `channels`. At
  // `--channels=12` with two heads of eight every stage up to the pair MLP is
  // exact and `encoder.stackOut` reads 1.2e-2 - the reference being asked the
  // wrong question, not a wrong kernel, which `check-webgl-af3-atom-block.js`
  // refuses by name for the same reason. The real model is square here (128 =
  // 4 x 32); only the TOKEN transformer's conditioning differs from its
  // activation, and that axis is `seq-channels` against `per-token`.
  if (heads * dimension !== channels) {
    throw new RangeError(
      `af3 diffusion head gate: heads * dimension is ${heads * dimension} and the atom`
      + ` channels ${channels}. The f64 reference's \`adaptiveZeroInit\` reads the`
      + " gathered attention at `channels`, so the two must match for it to be"
      + " answering the same question.");
  }
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;
  const atoms = tokens * dense;
  const pairs = tokens * tokens;
  const width = heads * dimension;

  const random = generator(1618033);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const transitionGroup = (c) => ({
    ffwLayerNormScale: noise(c, 0.2, 1),
    ffwLayerNormOffset: noise(c, 0.1),
    ffwTransition1: noise(c * c * 2 * 2, 1 / Math.sqrt(c)),
    ffwTransition2: noise(c * 2 * c, 0.3 / Math.sqrt(c * 2)),
  });
  const adaptiveGroup = (prefix, cond, act) => ({
    [`${prefix}SingleCondLayerNormScale`]: noise(cond, 0.2, 1),
    [`${prefix}SingleCondScaleWeights`]: noise(cond * act, 1 / Math.sqrt(cond)),
    [`${prefix}SingleCondScaleBias`]: noise(act, 0.1),
    [`${prefix}SingleCondBias`]: noise(cond * act, 0.2 / Math.sqrt(cond)),
  });
  const attentionBlock = (cond, act) => {
    const intermediate = act * 2;
    return {
      ...adaptiveGroup("q", cond, act),
      ...adaptiveGroup("k", cond, act),
      ...adaptiveGroup("ffw", cond, act),
      qProjection: noise(act * width, 1 / Math.sqrt(act)),
      qBias: noise(width, 0.1),
      kProjection: noise(act * width, 1 / Math.sqrt(act)),
      vProjection: noise(act * width, 1 / Math.sqrt(act)),
      gatingQuery: noise(act * width, 1 / Math.sqrt(act)),
      Transition2: noise(width * act, 0.3 / Math.sqrt(width)),
      AdaptiveZeroCondWeights: noise(cond * act, 1 / Math.sqrt(cond)),
      AdaptiveZeroCondBias: noise(act, 0.1),
      ffwTransition1: noise(act * intermediate * 2, 1 / Math.sqrt(act)),
      ffwTransition2: noise(intermediate * act, 0.3 / Math.sqrt(intermediate)),
      ffwAdaptiveZeroCondWeights: noise(cond * act, 1 / Math.sqrt(cond)),
      ffwAdaptiveZeroCondBias: noise(act, 0.1),
      keyMaskedAtomAttention: false,
      chainedAtomLayerNorm: false,
      diffusionNoResidual: false,
      maskAtomActPerBlock: false,
    };
  };
  // The atom stack's blocks are conditioned by the ATOM channels, and the token
  // transformer's by the head's own single conditioning.
  const atomBlock = () => attentionBlock(channels, channels);
  const tokenBlock = () => {
    const block = attentionBlock(seqChannels, perTokenChannels);
    // The token transformer's own adaptive group has no prefix.
    const renamed = { ...block, ...adaptiveGroup("", seqChannels, perTokenChannels) };
    for (const key of Object.keys(block)) {
      if (key.startsWith("qSingleCond") || key.startsWith("kSingleCond")) {
        delete renamed[key];
      }
    }
    return renamed;
  };

  const condWidth = trunkPairChannels + RELATIVE_WIDTH;
  const padColumns = padDna ? 2 : 0;
  const singleWidth = trunkSingleChannels + targetFeatWidth + padColumns;
  const fourierWeight = noise(noiseChannels, 1);
  const fourierBias = noise(noiseChannels, 0.5);

  const conditioningWeights = {
    pairChannels: diffusionPairChannels, seqChannels,
    trunkPairChannels, trunkSingleChannels, targetFeatWidth,
    relativeWidth: RELATIVE_WIDTH,
    pairCondInitialNormScale: noise(condWidth, 0.2, 1),
    pairCondInitialNormOffset: noise(condWidth, 0.1),
    pairCondInitialProjection: noise(
      condWidth * diffusionPairChannels, 1 / Math.sqrt(condWidth)),
    // TWO of each: the reference's loops are `index < 2` rather than over the
    // array, so a shorter list is a shape neither side admits.
    pairTransitions: [
      transitionGroup(diffusionPairChannels), transitionGroup(diffusionPairChannels)],
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
  };
  const encoderWeights = {
    channels, pairChannels, perTokenChannels, heads, dimension,
    trunkSingleChannels,
    // 🔴 THE ENCODER'S TRUNK PAIR IS THE HEAD'S OWN CONDITIONING, not the
    // trunk's: `diffusionHead` passes `cond.pair`. So its width is the
    // conditioning's and not `trunkPairChannels`.
    trunkPairChannels: diffusionPairChannels,
    singleToPairCondRow: noise(channels * pairChannels, 1 / Math.sqrt(channels)),
    singleToPairCondCol: noise(channels * pairChannels, 1 / Math.sqrt(channels)),
    embedPairOffsets: noise(3 * pairChannels, 0.3),
    embedPairDistances: noise(pairChannels, 0.3),
    embedPairOffsetsValid: noise(pairChannels, 0.3),
    pairMlp1: noise(pairChannels * pairChannels, 1 / Math.sqrt(pairChannels)),
    pairMlp2: noise(pairChannels * pairChannels, 1 / Math.sqrt(pairChannels)),
    pairMlp3: noise(pairChannels * pairChannels, 1 / Math.sqrt(pairChannels)),
    pairInputLayerNormScale: noise(pairChannels, 0.2, 1),
    pairLogitsProjection: noise(
      pairChannels * blocks * heads, 1 / Math.sqrt(pairChannels)),
    projectAtomFeaturesForAggr: noise(
      channels * perTokenChannels, 1 / Math.sqrt(channels)),
    atomPositionsToFeatures: noise(3 * channels, 0.3),
    lnormTrunkSingleCondScale: noise(trunkSingleChannels, 0.2, 1),
    lnormTrunkSingleCondOffset: noise(trunkSingleChannels, 0.1),
    embedTrunkSingleCond: noise(
      trunkSingleChannels * channels, 1 / Math.sqrt(trunkSingleChannels)),
    lnormTrunkPairCondScale: noise(diffusionPairChannels, 0.2, 1),
    lnormTrunkPairCondOffset: noise(diffusionPairChannels, 0.1),
    embedTrunkPairCond: noise(
      diffusionPairChannels * pairChannels, 1 / Math.sqrt(diffusionPairChannels)),
    pairNormPerBlock: false,
    keyMaskedAtomAttention: false,
    chainedAtomLayerNorm: false,
    diffusionNoResidual: false,
    blocks: Array.from({ length: blocks }, atomBlock),
  };
  const transformerWeights = {
    channels: perTokenChannels, condChannels: seqChannels,
    pairChannels: diffusionPairChannels, heads, dimension,
    blocksPerSuperBlock: perSuper, transitionFactor: 2,
    pairInputLayerNormScale: noise(diffusionPairChannels, 0.2, 1),
    superBlocks: Array.from({ length: superBlocks }, () => ({
      pairLogitsProjection: noise(
        diffusionPairChannels * perSuper * heads, 1 / Math.sqrt(diffusionPairChannels)),
      blocks: Array.from({ length: perSuper }, tokenBlock),
    })),
  };
  const decoderWeights = {
    channels, pairChannels, perTokenChannels, heads, dimension,
    projectTokenFeaturesForBroadcast: noise(
      perTokenChannels * channels, 1 / Math.sqrt(perTokenChannels)),
    pairInputLayerNormScale: noise(pairChannels, 0.2, 1),
    pairLogitsProjection: noise(
      pairChannels * blocks * heads, 1 / Math.sqrt(pairChannels)),
    atomFeaturesLayerNormScale: noise(channels, 0.2, 1),
    atomFeaturesLayerNormOffset: noise(channels, 0.1),
    // 🔴 THREE COLUMNS, NOT FOUR. See the note at the top.
    atomFeaturesToPositionUpdate: noise(channels * 3, 0.3 / Math.sqrt(channels)),
    pairNormPerBlock: false,
    keyMaskedAtomAttention: false,
    chainedAtomLayerNorm: false,
    diffusionNoResidual: false,
    blocks: Array.from({ length: blocks }, atomBlock),
  };
  const headWeights = {
    seqChannels, perTokenChannels,
    singleCondEmbeddingNormScale: noise(seqChannels, 0.2, 1),
    singleCondEmbeddingNormOffset: noise(seqChannels, 0.1),
    singleCondEmbeddingProjection: noise(
      seqChannels * perTokenChannels, 1 / Math.sqrt(seqChannels)),
    outputNormScale: noise(perTokenChannels, 0.2, 1),
    outputNormOffset: noise(perTokenChannels, 0.1),
    conditioning: conditioningWeights,
    encoder: encoderWeights,
    transformer: transformerWeights,
    decoder: decoderWeights,
  };

  // 🔴 FIVE IRREGULAR GATHERS, as the encoder's own gate builds them: padding on
  // both layouts and a reference space that is not constant, so the conventions
  // have something to disagree about.
  const atomMask = new Float32Array(atoms).fill(1);
  for (let index = 0; index < atoms; index += 1) {
    if (index % 5 === 4) atomMask[index] = 0;
  }
  const refSpaceUid = new Float32Array(atoms);
  for (let index = 0; index < atoms; index += 1) refSpaceUid[index] = Math.floor(index / dense);
  const tokenAtomsToQueries = {
    indices: new Int32Array(queryRows), mask: new Float32Array(queryRows), count: queryRows,
  };
  for (let index = 0; index < queryRows; index += 1) {
    tokenAtomsToQueries.indices[index] = (index * 7 + 2) % atoms;
    tokenAtomsToQueries.mask[index] = index < atoms ? 1 : 0;
  }
  const queriesToKeys = {
    indices: new Int32Array(keyRows), mask: new Float32Array(keyRows), count: keyRows,
  };
  for (let subset = 0; subset < subsets; subset += 1) {
    for (let key = 0; key < keys; key += 1) {
      const index = subset * keys + key;
      queriesToKeys.indices[index] = (subset * queries + (key * 5 + 3) % queries) % queryRows;
      queriesToKeys.mask[index] = key % 3 === 0 ? 1 : 0;
    }
  }
  const queriesToTokenAtoms = {
    indices: new Int32Array(atoms), mask: new Float32Array(atoms), count: atoms,
  };
  for (let index = 0; index < atoms; index += 1) {
    queriesToTokenAtoms.indices[index] = index % queryRows;
    queriesToTokenAtoms.mask[index] = atomMask[index];
  }
  const tokensToQueries = {
    indices: new Int32Array(queryRows), mask: new Float32Array(queryRows), count: queryRows,
  };
  for (let index = 0; index < queryRows; index += 1) {
    tokensToQueries.indices[index] = Math.floor(tokenAtomsToQueries.indices[index] / dense);
    tokensToQueries.mask[index] = index < atoms ? 1 : 0;
  }
  const tokensToKeys = {
    indices: new Int32Array(keyRows), mask: new Float32Array(keyRows), count: keyRows,
  };
  for (let index = 0; index < keyRows; index += 1) {
    tokensToKeys.indices[index] = index % tokens;
    tokensToKeys.mask[index] = queriesToKeys.mask[index];
  }

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

  const conditioning = noise(atoms * channels, 1);
  const refPos = noise(atoms * 3, 3);
  const trunkSingle = noise(tokens * trunkSingleChannels, 1);
  const trunkPair = noise(pairs * trunkPairChannels, 1);
  const targetFeat = noise(tokens * targetFeatWidth, 1);
  const positionsNoisy = noise(atoms * 3, 6);
  const seqMask = new Float32Array(tokens).fill(1);
  if (tokens > 2) seqMask[tokens - 1] = 0;

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
