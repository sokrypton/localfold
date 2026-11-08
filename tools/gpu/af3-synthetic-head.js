/**
 * A synthetic AF3 diffusion head - weights, layout and inputs - in the shape
 * both `webgl/af3/diffusion/head.js` and `cpu/af3/diffusion/diffusion.js` read.
 *
 * 🔴 ONE BUILDER, BECAUSE THE HEAD GATE AND THE SAMPLER GATE NEED THE SAME
 * MODEL. The sampler's question is what a WALK does to the head's 2e-7, which
 * only means something if the two gates are denoising the identical synthetic
 * model; a second copy of 200 lines of weight shapes would drift from the first
 * at the next convention.
 *
 * 🔴 AND THE LAYOUT IS IRREGULAR ON PURPOSE - padding on both atom layouts and a
 * reference space that is not constant - which is
 * `check-webgl-af3-atom-encoder.js`'s own statement: "a hand-built layout would
 * be regular in ways a real one is not, and the padding is where the interesting
 * disagreements live".
 *
 * 🔴 AND THE ATOM WIDTH MUST BE SQUARE. The f64 reference's `adaptiveZeroInit`
 * reads the gathered attention at `channels`, so `heads * dimension != channels`
 * is the reference being asked the wrong question rather than a wrong kernel -
 * measured 1.2e-2 on `encoder.stackOut` with every earlier tap exact. Refused
 * here by name, as `check-webgl-af3-atom-block.js` refuses it.
 *
 * These are NOISE, not a checkpoint: for differentials between two
 * implementations of the same arithmetic, and they say nothing about folding.
 */
import { RELATIVE_WIDTH } from "../../webgl/af3/trunk/embedder.js";
import { generator } from "./af2-synthetic-block.js";

/**
 * @param {{tokens?: number, dense?: number, subsets?: number, queries?: number,
 *   keys?: number, channels?: number, pairChannels?: number,
 *   perTokenChannels?: number, seqChannels?: number,
 *   diffusionPairChannels?: number, trunkSingleChannels?: number,
 *   trunkPairChannels?: number, targetFeatWidth?: number, blocks?: number,
 *   superBlocks?: number, perSuper?: number, padDna?: boolean,
 *   seed?: number}} [options]
 */
export function syntheticHead(options = {}) {
  const tokens = options.tokens ?? 4;
  const dense = options.dense ?? 4;
  const subsets = options.subsets ?? 2;
  const queries = options.queries ?? 8;
  const keys = options.keys ?? 12;
  const channels = options.channels ?? 16;
  const pairChannels = options.pairChannels ?? 8;
  const perTokenChannels = options.perTokenChannels ?? 24;
  const seqChannels = options.seqChannels ?? 16;
  const diffusionPairChannels = options.diffusionPairChannels ?? 16;
  const trunkSingleChannels = options.trunkSingleChannels ?? 12;
  const trunkPairChannels = options.trunkPairChannels ?? 8;
  const targetFeatWidth = options.targetFeatWidth ?? 68;
  const blocks = options.blocks ?? 2;
  const superBlocks = options.superBlocks ?? 2;
  const perSuper = options.perSuper ?? 2;
  const padDna = options.padDna === true;
  const heads = 2;
  const dimension = 8;
  const noiseChannels = 8;
  if (heads * dimension !== channels) {
    throw new RangeError(
      `af3 synthetic head: heads * dimension is ${heads * dimension} and the atom`
      + " channels " + channels + ". The f64 reference's `adaptiveZeroInit` reads"
      + " the gathered attention at `channels`, so the two must match for it to be"
      + " answering the same question.");
  }
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;
  const atoms = tokens * dense;
  const pairs = tokens * tokens;
  const width = heads * dimension;

  const random = generator(options.seed ?? 1618033);
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

  return {
    shape: { tokens, dense, subsets, queries, keys },
    widths: {
      channels, pairChannels, perTokenChannels, seqChannels, diffusionPairChannels,
      trunkSingleChannels, trunkPairChannels, targetFeatWidth, noiseChannels,
      heads, dimension, blocks, superBlocks, perSuper,
      relativeWidth: RELATIVE_WIDTH,
    },
    padDna,
    weights: headWeights,
    fourierWeight,
    fourierBias,
    layout: {
      atomMask, refSpaceUid, features,
      tokenAtomsToQueries, queriesToKeys, queriesToTokenAtoms,
      tokensToQueries, tokensToKeys,
    },
    values: {
      conditioning, refPos, trunkSingle, trunkPair, targetFeat, positionsNoisy, seqMask,
    },
  };
}
