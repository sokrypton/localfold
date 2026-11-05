/**
 * The whole structure module: four stages, the second of which iterates eight
 * times.
 *
 *     single, frames = initialize(msaFirstRow)
 *     normalised     = LayerNorm(pair)                     ONCE, not eight times
 *     repeat 8:  update = invariantPointAttention(single, normalised, frames)
 *                single, frames = iteration(single, update, frames)
 *     angles         = sidechainAngles(single, initialRepresentation)
 *     atoms          = atomGeometry(frames, angles)
 *
 * 🔴 THE PAIR LAYERNORM IS HOISTED, AND THAT IS THE REFERENCE'S CHOICE RATHER
 * THAN AN OPTIMISATION TAKEN HERE. Every iteration reads the same pair track
 * with the same weights, so `webgpu/af2/structure/core.js` prepares it once -
 * "eight of them used to push L*L*128 floats across the bus and layer-normalise
 * them to produce a buffer identical to the last". `normalisePair` is exported
 * separately by `ipa.js` for exactly this reason, and a port that normalised
 * inside the loop would be correct and eight times as expensive.
 *
 * 🔴 AND THE SIDECHAIN HEAD READS THE FIRST REPRESENTATION, NOT THE EIGHTH.
 * `sidechainAngles` takes `core.activations` AND
 * `initialized.initialRepresentation` - the LayerNorm output from before the
 * first iteration, which has to survive all eight. Handing it the current
 * activations twice is a plausible head that produces plausible torsions, and
 * nothing downstream would complain.
 *
 * 🔴 AND EVERY ITERATION'S INPUTS ARE RELEASED, INCLUDING THE INITIALISER'S.
 * The WebGPU core ping-pongs between two buffers and writes in place; a
 * fragment shader cannot, so each iteration allocates its successor and the
 * predecessor is released as soon as it has been read. The initialiser's own
 * activations and frames are released by the first iteration on the same rule,
 * which is why the loop releases unconditionally rather than tracking
 * ownership.
 */
import { atomGeometry } from "./geometry.js";
import {
  invariantPointAttention, normalisePair, releaseIpaWeights, uploadIpaWeights,
} from "./ipa.js";
import {
  releaseIterationWeights, structureIteration, uploadIterationWeights,
} from "./iteration.js";
import {
  releaseSidechainWeights, sidechainAngles, uploadSidechainWeights,
} from "./sidechain.js";
import {
  releaseStructureInitializeWeights, structureInitialize, uploadStructureInitializeWeights,
} from "./initialize.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/** The six geometry tables, which are constants rather than learned weights. */
export const GEOMETRY_TABLES = [
  "defaultFrames", "atom14ToGroup", "atom14Positions", "atom14Mask",
];

/**
 * AF2's own shapes for this module. They are stated once here because four
 * stages have to agree about them and `webgpu/af2/structure/core.js` is where
 * the IPA half comes from - 12 heads, 16 scalar channels each for the query and
 * the value, 4 query points and 8 value points.
 */
export const STRUCTURE_SHAPE = {
  msaChannels: 256,
  structureChannels: 384,
  pairChannels: 128,
  sidechainHidden: 128,
  iterations: 8,
  heads: 12,
  scalarQk: 16,
  scalarV: 16,
  pointQk: 4,
  pointV: 8,
};

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{initialize: object, ipa: object, postAttention: object, sidechain: object}} weights
 *   the structure section of `shared/bundles/alphafold-fixture.js`, unchanged
 */
export function uploadStructureModuleWeights(execution, weights) {
  return {
    initialize: uploadStructureInitializeWeights(execution, weights.initialize),
    ipa: uploadIpaWeights(execution, weights.ipa),
    // The reference calls this section `postAttention`; `iteration.js` is what
    // consumes it, and `uploadIterationWeights` already reads that spelling.
    iteration: uploadIterationWeights(execution, weights.postAttention),
    sidechain: uploadSidechainWeights(execution, weights.sidechain),
  };
}

export function releaseStructureModuleWeights(execution, weights) {
  releaseStructureInitializeWeights(execution, weights.initialize);
  releaseIpaWeights(execution, weights.ipa);
  releaseIterationWeights(execution, weights.iteration);
  releaseSidechainWeights(execution, weights.sidechain);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadGeometryTables(execution, tables) {
  /** @type {any} */
  const out = {};
  for (const name of GEOMETRY_TABLES) {
    const values = tables[name];
    if (values === undefined) throw new Error(`geometry tables: ${name} is missing`);
    out[name] = execution.upload(values, `w.geometry.${name}`);
  }
  return out;
}

export function releaseGeometryTables(execution, tables) {
  for (const name of GEOMETRY_TABLES) execution.release(tables[name]);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{msaFirstRow: Tensor, pair: Tensor, mask: Tensor, aatype: Tensor,
 *   atom37ToAtom14: Tensor, atom37Mask: Tensor, length: number,
 *   weights: ReturnType<typeof uploadStructureModuleWeights>,
 *   tables: {defaultFrames: Tensor, atom14ToGroup: Tensor, atom14Positions: Tensor,
 *     atom14Mask: Tensor},
 *   shape?: Partial<typeof STRUCTURE_SHAPE>, epsilon?: number}} call
 * @returns {{atom14: Tensor, atom37: Tensor, finalRepresentation: Tensor,
 *   affine: Tensor, angles: Tensor, unnormalizedAngles: Tensor}} the caller owns
 *   all six; nothing else this function allocated is still live
 */
export function structureModule(execution, call) {
  const {
    msaFirstRow, pair, mask, aatype, atom37ToAtom14, atom37Mask,
    length, weights: w, tables, epsilon = 1e-5,
  } = call;
  const {
    msaChannels, structureChannels, pairChannels, sidechainHidden, iterations,
    heads, scalarQk, scalarV, pointQk, pointV,
  } = { ...STRUCTURE_SHAPE, ...(call.shape ?? {}) };

  const initialized = structureInitialize(execution, {
    msaFirstRow, weights: w.initialize, length, msaChannels, structureChannels, epsilon,
  });
  // The single projection is the LayerNorm's input and nothing else reads it.
  execution.release(initialized.single);

  const normalisedPair = normalisePair(execution, {
    pair, weights: w.ipa, length, pairChannels, epsilon,
  });

  let activations = initialized.activations;
  let affine = initialized.affine;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const update = invariantPointAttention(execution, {
      activations, normalisedPair, mask, affine, weights: w.ipa,
      length, channels: structureChannels, pairChannels,
      heads, scalarQk, scalarV, pointQk, pointV,
    });
    const next = structureIteration(execution, {
      activations, update, affine, weights: w.iteration,
      length, channels: structureChannels, epsilon,
    });
    execution.release(update);
    execution.release(activations);
    execution.release(affine);
    activations = next.activations;
    affine = next.affine;
  }
  execution.release(normalisedPair);

  const sidechain = sidechainAngles(execution, {
    activations, initialActivations: initialized.initialRepresentation,
    weights: w.sidechain, length,
    inputChannels: structureChannels, hiddenChannels: sidechainHidden,
  });
  execution.release(initialized.initialRepresentation);

  const geometry = atomGeometry(execution, {
    affine, angles: sidechain.angles, aatype, atom37ToAtom14, atom37Mask, length, tables,
  });
  // The composed frames are an intermediate of the geometry stage; the
  // reference's module returns the coordinates and not the frames.
  execution.release(geometry.frames);

  activations.label = "structure.final";
  return {
    atom14: geometry.atom14,
    atom37: geometry.atom37,
    finalRepresentation: activations,
    affine,
    angles: sidechain.angles,
    unnormalizedAngles: sidechain.unnormalizedAngles,
  };
}
