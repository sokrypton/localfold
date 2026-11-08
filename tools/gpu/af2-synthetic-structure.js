/**
 * Synthetic AF2 structure-module weights and geometry tables, in the structure
 * both backends read.
 *
 * 🔴 ONE BUILDER FOR THE WHOLE MODULE, BECAUSE A COMPOSITION GATE NEEDS ALL FOUR
 * STAGES AT ONCE. The five per-stage gates each build only their own stage's
 * tensors, with their own shapes and their own deliberately awkward arms - a
 * non-unit quaternion, a zeroed angle weight - and they stay as they are. This
 * is the set `structureModule` and `StructureModuleGpu` both take whole, in
 * `shared/bundles/alphafold-fixture.js`'s own spelling: `initialize`, `ipa`,
 * `postAttention` and `sidechain`, with the geometry tables beside them because
 * the module reads those as a separate argument.
 *
 * These are NOISE, not a checkpoint. They are for differentials between two
 * implementations of the same arithmetic and say nothing about folding.
 */
import { generator } from "./af2-synthetic-block.js";

/**
 * @param {{length: number, msaChannels: number, structureChannels: number,
 *   pairChannels: number, sidechainHidden: number, heads: number,
 *   scalarQk: number, scalarV: number, pointQk: number, pointV: number,
 *   seed?: number}} shape
 */
export function syntheticStructureWeights(shape) {
  const {
    msaChannels, structureChannels: channels, pairChannels, sidechainHidden: hidden,
    heads, scalarQk, scalarV, pointQk, pointV, seed = 7,
  } = shape;
  const random = generator(seed);
  const noise = (count, spread, centre = 0) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  // What the IPA's output projection reads: the scalar features, the point
  // features as (x, y, z, norm) a point, and the pair features.
  const featureChannels = heads * scalarV + 4 * heads * pointV + heads * pairChannels;
  return {
    initialize: {
      singleProjectionWeight: noise(msaChannels * channels, 1 / Math.sqrt(msaChannels)),
      singleProjectionBias: noise(channels, 0.1),
      singleNormScale: noise(channels, 0.2, 1),
      singleNormOffset: noise(channels, 0.1),
      initialProjectionWeight: noise(channels * channels, 1 / Math.sqrt(channels)),
      initialProjectionBias: noise(channels, 0.1),
    },
    ipa: {
      pairNormScale: noise(pairChannels, 0.2, 1),
      pairNormOffset: noise(pairChannels, 0.1),
      queryScalarWeight: noise(channels * heads * scalarQk, 1 / Math.sqrt(channels)),
      queryScalarBias: noise(heads * scalarQk, 0.1),
      keyValueScalarWeight: noise(
        channels * heads * (scalarQk + scalarV), 1 / Math.sqrt(channels)),
      keyValueScalarBias: noise(heads * (scalarQk + scalarV), 0.1),
      queryPointWeight: noise(channels * heads * pointQk * 3, 1 / Math.sqrt(channels)),
      queryPointBias: noise(heads * pointQk * 3, 0.1),
      keyValuePointWeight: noise(
        channels * heads * (pointQk + pointV) * 3, 1 / Math.sqrt(channels)),
      keyValuePointBias: noise(heads * (pointQk + pointV) * 3, 0.1),
      trainablePointWeights: noise(heads, 0.5),
      attention2dWeight: noise(pairChannels * heads, 1 / Math.sqrt(pairChannels)),
      attention2dBias: noise(heads, 0.1),
      outputWeight: noise(featureChannels * channels, 1 / Math.sqrt(featureChannels)),
      outputBias: noise(channels, 0.05),
    },
    postAttention: {
      attentionNormScale: noise(channels, 0.2, 1),
      attentionNormOffset: noise(channels, 0.1),
      transitionWeights: [0, 1, 2].map(
        () => noise(channels * channels, 1 / Math.sqrt(channels))),
      transitionBiases: [0, 1, 2].map(() => noise(channels, 0.1)),
      transitionNormScale: noise(channels, 0.2, 1),
      transitionNormOffset: noise(channels, 0.1),
      // 🔴 SMALL ON PURPOSE. The affine update is applied eight times and each
      // one rotates and translates the frame it is added to, so a weight at the
      // usual 1/sqrt(channels) scale winds the frames around several times over
      // eight iterations and the comparison measures a chaotic trajectory rather
      // than the arithmetic. The checkpoint's own is small for the same reason.
      affineWeight: noise(channels * 6, 0.05 / Math.sqrt(channels)),
      affineBias: noise(6, 0.01),
    },
    sidechain: {
      inputWeight: noise(channels * hidden, 1 / Math.sqrt(channels)),
      inputBias: noise(hidden, 0.1),
      initialInputWeight: noise(channels * hidden, 1 / Math.sqrt(channels)),
      initialInputBias: noise(hidden, 0.1),
      residual1Weights: [noise(hidden * hidden, 1 / Math.sqrt(hidden)),
        noise(hidden * hidden, 1 / Math.sqrt(hidden))],
      residual1Biases: [noise(hidden, 0.1), noise(hidden, 0.1)],
      residual2Weights: [noise(hidden * hidden, 1 / Math.sqrt(hidden)),
        noise(hidden * hidden, 1 / Math.sqrt(hidden))],
      residual2Biases: [noise(hidden, 0.1), noise(hidden, 0.1)],
      angleWeight: noise(hidden * 14, 1 / Math.sqrt(hidden)),
      angleBias: noise(14, 0.1),
    },
  };
}

/**
 * The geometry tables and the three per-residue index features the atom stage
 * reads. They are constants in a real fold - `shared/af2/geometry/` builds them
 * from the residue dictionary - and noise here, because this stage's arithmetic
 * has to satisfy whatever the tables say.
 */
export function syntheticGeometryTables(length, seed = 11) {
  const random = generator(seed);
  const noise = (count, spread) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = (random() * 2 - 1) * spread;
    return out;
  };
  const integers = (count, limit) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = Math.floor(random() * limit);
    return out;
  };
  return {
    tables: {
      defaultFrames: noise(21 * 8 * 16, 1),
      atom14ToGroup: integers(21 * 14, 8),
      atom14Positions: noise(21 * 14 * 3, 2),
      atom14Mask: integers(21 * 14, 2),
    },
    aatype: integers(length, 21),
    atom37ToAtom14: integers(length * 37, 14),
    atom37Mask: integers(length * 37, 2),
  };
}
