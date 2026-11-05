/**
 * Synthetic AF2 Evoformer weights, in the structure both backends read.
 *
 * 🔴 ONE BUILDER, BECAUSE TWO GATES NEED THE SAME EIGHTY TENSORS AND A SECOND
 * COPY WOULD DRIFT. The block gate and the stack gate both hand ONE weight set
 * to two implementations; if they built it differently they would stop
 * comparing the same model, and the failure would look like a port bug. The
 * structure is `shared/bundles/alphafold-fixture.js`'s - `{heads, attention,
 * pairBias}` for an attention, a flat group for a transition, the outer product
 * mean and a triangle multiplication - because that is what
 * `webgpu/af2/evoformer/block.js` reads.
 *
 * These are NOISE, not a checkpoint. They are for differentials between two
 * implementations of the same arithmetic and say nothing about folding: see
 * docs/AF2.md on how far synthetic activations go, where a packing that moved
 * the real model 4e-4 read 6e-7 on noise.
 */

/** The repository's own deterministic generator, so a gate is reproducible. */
export function generator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}

/**
 * @param {{sequences: number, length: number, cM: number, cZ: number,
 *   cOuter: number, heads: number, triangleHidden: number, msaHidden: number,
 *   pairHidden: number, seed?: number, global?: boolean}} shape
 */
export function syntheticBlockWeights(shape) {
  const {
    cM, cZ, cOuter, heads, triangleHidden, msaHidden, pairHidden, seed = 1,
  } = shape;
  const random = generator(seed);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const attentionGroup = (channels) => ({
    heads,
    attention: {
      queryNormScale: noise(channels, 0.2, 1),
      queryNormOffset: noise(channels, 0.1),
      queryWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      keyWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      valueWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      gatingWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      gatingBias: noise(channels, 0.1, 1),
      outputWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      outputBias: noise(channels, 0.05),
    },
  });
  const transitionGroup = (channels, hidden) => ({
    layerNormScale: noise(channels, 0.2, 1),
    layerNormOffset: noise(channels, 0.1),
    firstWeight: noise(channels * hidden, 1 / Math.sqrt(channels)),
    firstBias: noise(hidden, 0.1),
    secondWeight: noise(hidden * channels, 1 / Math.sqrt(hidden)),
    secondBias: noise(channels, 0.05),
  });
  const triangleGroup = () => ({
    layerNormInWeight: noise(cZ, 0.2, 1),
    layerNormInBias: noise(cZ, 0.1),
    linearAPWeight: noise(triangleHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearAPBias: noise(triangleHidden, 0.03),
    linearAGWeight: noise(triangleHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearAGBias: noise(triangleHidden, 0.03, 1),
    linearBPWeight: noise(triangleHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearBPBias: noise(triangleHidden, 0.03),
    linearBGWeight: noise(triangleHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearBGBias: noise(triangleHidden, 0.03, 1),
    layerNormOutWeight: noise(triangleHidden, 0.2, 1),
    layerNormOutBias: noise(triangleHidden, 0.1),
    linearZWeight: noise(cZ * triangleHidden, 0.15 / Math.sqrt(triangleHidden)),
    linearZBias: noise(cZ, 0.03),
    linearGWeight: noise(cZ * cZ, 0.15 / Math.sqrt(cZ)),
    linearGBias: noise(cZ, 0.03, 1),
  });
  /** Global attention: a flat group, and single-headed keys and values. */
  const globalAttentionGroup = (channels) => {
    const headDim = channels / heads;
    const gates = heads * headDim;
    return Object.assign({
      queryNormScale: noise(channels, 0.2, 1),
      queryNormOffset: noise(channels, 0.1),
      queryWeight: noise(channels * gates, 1 / Math.sqrt(channels)),
      keyWeight: noise(channels * headDim, 1 / Math.sqrt(channels)),
      valueWeight: noise(channels * headDim, 1 / Math.sqrt(channels)),
      gatingWeight: noise(channels * gates, 1 / Math.sqrt(channels)),
      gatingBias: noise(gates, 0.1, 1),
      outputWeight: noise(gates * channels, 1 / Math.sqrt(gates)),
      outputBias: noise(channels, 0.05),
    }, { heads });
  };

  const weights = {
    msaRowAttention: {
      ...attentionGroup(cM),
      pairBias: {
        layerNormScale: noise(cZ, 0.2, 1),
        layerNormOffset: noise(cZ, 0.1),
        projectionWeight: noise(cZ * heads, 1 / Math.sqrt(cZ)),
      },
    },
    msaTransition: transitionGroup(cM, msaHidden),
    outerProductMean: {
      layerNormScale: noise(cM, 0.2, 1),
      layerNormOffset: noise(cM, 0.1),
      leftWeight: noise(cM * cOuter, 1 / Math.sqrt(cM)),
      leftBias: noise(cOuter, 0.1),
      rightWeight: noise(cM * cOuter, 1 / Math.sqrt(cM)),
      rightBias: noise(cOuter, 0.1),
      outputWeight: noise(cOuter * cOuter * cZ, 1 / cOuter),
      outputBias: noise(cZ, 0.05),
    },
    triangleMultiplicationOutgoing: triangleGroup(),
    triangleMultiplicationIncoming: triangleGroup(),
    triangleAttentionStarting: {
      ...attentionGroup(cZ),
      pairBias: { projectionWeight: noise(cZ * heads, 1 / Math.sqrt(cZ)) },
    },
    triangleAttentionEnding: {
      ...attentionGroup(cZ),
      pairBias: { projectionWeight: noise(cZ * heads, 1 / Math.sqrt(cZ)) },
    },
    pairTransition: transitionGroup(cZ, pairHidden),
  };
  return shape.global === true
    ? { ...weights, msaColumnGlobalAttention: globalAttentionGroup(cM) }
    : { ...weights, msaColumnAttention: attentionGroup(cM) };
}

/** Activations and masks for one of these shapes, with a few mask zeros. */
export function syntheticActivations(shape, seed = 9) {
  const { sequences, length, cM, cZ } = shape;
  const random = generator(seed);
  const noise = (count, spread) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = (random() * 2 - 1) * spread;
    return out;
  };
  const msaMask = new Float32Array(sequences * length).fill(1);
  if (sequences > 1 && length > 1) msaMask[(sequences - 1) * length + 1] = 0;
  const pairMask = new Float32Array(length * length).fill(1);
  if (length > 1) {
    pairMask[2 % (length * length)] = 0;
    pairMask[length * length - 3 >= 0 ? length * length - 3 : 0] = 0;
  }
  return {
    msa: noise(sequences * length * cM, 1.5),
    pair: noise(length * length * cZ, 1.5),
    msaMask,
    pairMask,
  };
}
