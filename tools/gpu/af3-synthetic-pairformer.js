/**
 * Synthetic AF3 pairformer weights, in the structure both backends read.
 *
 * 🔴 ONE STATEMENT OF THE WEIGHT LIST, BECAUSE FOUR STACKS SHARE IT. The pair
 * half - two triangle multiplications, two grid attentions and a transition - is
 * what the trunk's pairformer, the trunk's MSA blocks, the template stack and
 * the CONFIDENCE head's four blocks are all built from, at four different
 * channel counts and two different transition factors. A second copy of it in
 * the confidence gate would drift from the trunk gate's at the next convention.
 *
 * 🔴 AND THE DRAW ORDER IS THE CALLER'S, which is why `noise` is a parameter
 * rather than a generator built here: a gate's arms are reproducible only if the
 * same sequence of draws goes to the same tensors, so moving a builder into this
 * file must not move where the stream is consumed.
 *
 * These are NOISE, not a checkpoint: for differentials between two
 * implementations of the same arithmetic, and they say nothing about folding.
 */

/** @typedef {(n: number, spread: number, centre?: number) => Float32Array} Noise */

/** @param {Noise} noise */
export function triangleGroup(noise, channels) {
  return {
    leftNormInputScale: noise(channels, 0.2, 1),
    leftNormInputOffset: noise(channels, 0.1),
    projection: noise(channels * channels * 2, 0.3 / Math.sqrt(channels)),
    gate: noise(channels * channels * 2, 0.3 / Math.sqrt(channels)),
    centerNormScale: noise(channels, 0.2, 1),
    centerNormOffset: noise(channels, 0.1),
    outputProjection: noise(channels * channels, 0.3 / Math.sqrt(channels)),
    gatingLinear: noise(channels * channels, 0.3 / Math.sqrt(channels)),
  };
}

/** @param {Noise} noise */
export function gridGroup(noise, channels, heads, dimension) {
  const width = heads * dimension;
  return {
    heads, dimension,
    actNormScale: noise(channels, 0.2, 1),
    actNormOffset: noise(channels, 0.1),
    pairBiasProjection: noise(channels * heads, 1 / Math.sqrt(channels)),
    qProjection: noise(width * channels, 1 / Math.sqrt(channels)),
    kProjection: noise(width * channels, 1 / Math.sqrt(channels)),
    vProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    gatingQuery: noise(width * channels, 1 / Math.sqrt(channels)),
    outputProjection: noise(width * channels, 0.3 / Math.sqrt(width)),
  };
}

/** @param {Noise} noise */
export function transitionGroup(noise, channels, factor) {
  return {
    inputLayerNormScale: noise(channels, 0.2, 1),
    inputLayerNormOffset: noise(channels, 0.1),
    transition1: noise(channels * channels * factor * 2, 1 / Math.sqrt(channels)),
    transition2: noise(channels * factor * channels, 0.3 / Math.sqrt(channels * factor)),
  };
}

/** @param {Noise} noise */
export function pairHalfGroups(noise, channels, factor) {
  return {
    triangleMultiplicationOutgoing: triangleGroup(noise, channels),
    triangleMultiplicationIncoming: triangleGroup(noise, channels),
    pairAttention1: gridGroup(noise, channels, 4, 8),
    pairAttention2: gridGroup(noise, channels, 4, 8),
    pairTransition: transitionGroup(noise, channels, factor),
  };
}

/**
 * A whole pairformer block: the pair half, the single track's pair bias, its
 * attention and its transition.
 *
 * @param {Noise} noise
 */
export function pairformerBlockWeights(noise, {
  pairChannels, singleChannels, factor = 4, heads = 4, dimension = 16,
}) {
  const width = heads * dimension;
  return {
    pairChannels, singleChannels,
    ...pairHalfGroups(noise, pairChannels, factor),
    singlePairLogitsNormScale: noise(pairChannels, 0.2, 1),
    singlePairLogitsNormOffset: noise(pairChannels, 0.1),
    singlePairLogitsProjection: noise(pairChannels * 4, 1 / Math.sqrt(pairChannels)),
    singleAttention: {
      heads: 4, dimension,
      layerNormScale: noise(singleChannels, 0.2, 1),
      layerNormOffset: noise(singleChannels, 0.1),
      qProjection: noise(singleChannels * width, 1 / Math.sqrt(singleChannels)),
      qBias: noise(width, 0.1),
      kProjection: noise(singleChannels * width, 1 / Math.sqrt(singleChannels)),
      vProjection: noise(singleChannels * width, 1 / Math.sqrt(singleChannels)),
      gatingQuery: noise(singleChannels * width, 1 / Math.sqrt(singleChannels)),
      outputProjection: noise(width * singleChannels, 0.3 / Math.sqrt(width)),
    },
    singleTransition: transitionGroup(noise, singleChannels, factor),
  };
}
