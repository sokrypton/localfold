/**
 * The input embedder: what the Evoformer starts from, and where recycling
 * re-enters.
 *
 *     msa[s][i]   = Linear(target[i]) + Linear(msaFeatures[s][i])
 *                   + (s == 0 ? previousMsa[i] : 0)
 *     extra[s][i] = Linear(one-hot code, deletion flag, deletion value)
 *     pair[i][j]  = Linear(target[i]) + Linear(target[j]) + previousPair[i][j]
 *                   + one-hot(distance(pseudo-beta i, j)) + one-hot(relative index)
 *
 * 🔴 AND BOTH RECYCLED TENSORS ARE LAYER-NORMALISED FIRST, WHICH THIS PORT
 * MISSED UNTIL THE REFERENCE'S WEIGHT LIST SAID SO. `previousMsaNormScale`,
 * `previousMsaNormOffset`, `previousPairNormScale` and `previousPairNormOffset`
 * are four tensors with nowhere else to go, and the embedder compiles a
 * normalise pipeline it would otherwise never use. Adding the recycled tensors
 * raw is a plausible embedding whose scale drifts with every pass - the kind of
 * error that degrades a long fold and leaves a short one looking fine.
 *
 * It was found by the cross-check refusing to run: `packWeights` reads all
 * eighteen tensors and threw on the four that were absent. A port written
 * against a reading of the shaders alone would have had three kernels that each
 * looked complete.
 *
 * 🔴 THE RECYCLED MSA GOES INTO ROW ZERO ONLY. `if (sequence == 0u)` - the
 * previous pass's single representation re-enters on the query row and nowhere
 * else, and adding it to every row is a plausible embedding that drowns the
 * alignment.
 *
 * 🔴 THE PSEUDO-BETA IS CB EXCEPT FOR GLYCINE, WHICH HAS NONE. Atom 3 normally
 * and atom 1 (CA) when `aatype == 7`, read out of the previous pass's atom37
 * positions. Using CA throughout, or CB for glycine, moves every distance in the
 * recycled distogram.
 *
 * 🔴 AND A DISTANCE BELOW `minBin` LANDS IN NO BIN AT ALL. The test is
 * `d2 > lower * lower && (last || d2 < upper * upper)`, with the first bin's
 * lower edge at 3.25 - so a contact closer than that contributes NOTHING rather
 * than falling into bin zero, and the last bin's upper edge is `maxBin +
 * binWidth` rather than infinity. A clamped one-hot is the obvious reading and
 * is wrong at both ends.
 *
 * 🔴 AND THE RELATIVE POSITION IS CLAMPED, NOT WRAPPED:
 * `clamp(residueIndex[i] - residueIndex[j] + maxRelative, 0, 2 * maxRelative)`,
 * so every separation past 32 shares one bin - which is what lets a chain break
 * be expressed as a large index jump.
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const INPUT_EMBEDDER_WEIGHTS = [
  "preprocess1dWeight", "preprocess1dBias", "preprocessMsaWeight", "preprocessMsaBias",
  "leftSingleWeight", "leftSingleBias", "rightSingleWeight", "rightSingleBias",
  "previousPositionWeight", "previousPositionBias",
  "previousMsaNormScale", "previousMsaNormOffset",
  "previousPairNormScale", "previousPairNormOffset",
  "relativePositionWeight", "relativePositionBias",
  "extraMsaWeight", "extraMsaBias",
];

/**
 * The per-residue identity the multimer relative encoding reads: `[4, length]`
 * holding the residue index, then the asymmetric-unit, entity and symmetry ids.
 *
 * All three default to zero, which is what a monomer IS: one chain, one entity,
 * one symmetry copy - so a monomer's identity lights none of the eight trailing
 * features and the chains kernel reproduces the monomer one exactly.
 */
export function residueIdentity(input) {
  const { length } = input;
  const identity = new Float32Array(length * 4);
  identity.set(Float32Array.from(input.residueIndex).subarray(0, length), 0);
  const lanes = [input.asymId, input.entityId, input.symId];
  for (const [index, lane] of lanes.entries()) {
    if (lane === undefined) continue;
    if (lane.length !== length) {
      throw new RangeError(`residueIdentity: chain identity must have one value per residue`);
    }
    identity.set(Float32Array.from(lane), length * (index + 1));
  }
  return identity;
}

/**
 * The relative-position table, widened from the monomer's 65 rows to the
 * multimer graph's 73.
 *
 * Widening here rather than at export means the shipped monomer weights stay
 * the bytes on disk: no re-quantisation, no new manifest, no re-download. The
 * reference does exactly this and says so.
 */
export function widenRelativePositionWeight(weight, pairChannels) {
  const rows = weight.length / pairChannels;
  if (rows === MULTIMER_RELATIVE_ROWS) return weight;
  if (rows !== MONOMER_RELATIVE_ROWS) {
    throw new RangeError(
      `relative position table has ${rows} rows; expected `
      + `${MONOMER_RELATIVE_ROWS} or ${MULTIMER_RELATIVE_ROWS}`);
  }
  const widened = new Float32Array(MULTIMER_RELATIVE_ROWS * pairChannels);
  widened.set(weight);
  return widened;
}

export const DGRAM_BINS = 15;
export const MAX_RELATIVE = 32;
/** AF2-multimer's relative-CHAIN clamp, which the monomer graph has no use for. */
export const MAX_RELATIVE_CHAIN = 2;
/** 65 rows for the monomer lookup, 73 for the multimer one. */
export const MONOMER_RELATIVE_ROWS = 2 * MAX_RELATIVE + 1;
// 66 offset bins (the last meaning "a different chain") + 1 entity-same + 6
// relative-chain bins (the last meaning "a different entity") = 73.
export const MULTIMER_RELATIVE_ROWS =
  (2 * MAX_RELATIVE + 2) + 1 + (2 * MAX_RELATIVE_CHAIN + 2);
export const MIN_BIN = 3.25;
export const MAX_BIN = 20.75;

export const msaEmbedSource = (shape) => {
  const { length, sequences, targetChannels, msaFeatureChannels, msaChannels } = shape;
  return `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int TARGET_C = ${targetChannels};
const int MSA_FEATURE_C = ${msaFeatureChannels};
const int MSA_C = ${msaChannels};
const int MSA_C4 = ${msaChannels / 4};
const int ELEMENTS = ${sequences * length * msaChannels};
uniform sampler2D targetTex;
uniform sampler2D msaFeatureTex;
uniform sampler2D previousMsaTex;
uniform sampler2D targetWeightTex;
uniform sampler2D targetBiasTex;
uniform sampler2D msaWeightTex;
uniform sampler2D msaBiasTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % MSA_C4;
  int row = texel / MSA_C4;
  int residue = row % L;
  int sequence = row / L;

  vec4 total = fetch4(targetBiasTex, group) + fetch4(msaBiasTex, group);
  for (int c = 0; c < TARGET_C; ++c) {
    total += fetch1(targetTex, residue * TARGET_C + c)
      * fetch4(targetWeightTex, c * MSA_C4 + group);
  }
  for (int c = 0; c < MSA_FEATURE_C; ++c) {
    total += fetch1(msaFeatureTex, row * MSA_FEATURE_C + c)
      * fetch4(msaWeightTex, c * MSA_C4 + group);
  }
  // Row zero only: the previous pass re-enters on the query row.
  if (sequence == 0) total += fetch4(previousMsaTex, residue * MSA_C4 + group);
  result = total;
}`;
};

export const extraMsaEmbedSource = (shape) => {
  const { length, sequences, extraChannels } = shape;
  return `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int EXTRA_C4 = ${extraChannels / 4};
const int ELEMENTS = ${sequences * length * extraChannels};
uniform sampler2D codeTex;
uniform sampler2D deletionFlagTex;
uniform sampler2D deletionValueTex;
uniform sampler2D weightTex;
uniform sampler2D biasTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % EXTRA_C4;
  int row = texel / EXTRA_C4;
  int code = int(fetch1(codeTex, row));
  // Rows 23 and 24 of the same weight carry the deletion flag and value.
  vec4 total = fetch4(biasTex, group)
    + fetch4(weightTex, code * EXTRA_C4 + group)
    + fetch1(deletionFlagTex, row) * fetch4(weightTex, 23 * EXTRA_C4 + group)
    + fetch1(deletionValueTex, row) * fetch4(weightTex, 24 * EXTRA_C4 + group);
  result = total;
}`;
};

/**
 * 🔴 TWO RELATIVE ENCODINGS, AND THE MULTIMER ONE IS A SUPERSET - WHICH THE
 * REFERENCE SAYS IS THE END STATE. `webgpu/af2/multimer/input-embedder.js`: "THE
 * MULTIMER GRAPH IS THE ONE GRAPH, and monomer is its single-chain case. 73 = 66
 * offset bins (the last meaning 'different chain') + 1 entity-same + 6
 * relative-chain bins. A monomer never lights any of the 8 trailing features:
 * one chain makes asym_same and entity_same true everywhere and the symmetry
 * delta zero. So zero rows there reproduce the 65-row result EXACTLY, and the
 * shader needs no monomer branch."
 *
 * Both are kept here for now, as the reference keeps them, and
 * `check-webgl-multimer-embedder.js` asserts the claim rather than quoting it:
 * one arm runs a monomer's features through the CHAINS kernel with widened
 * weights and holds it BIT-IDENTICAL to the monomer kernel. Unifying without
 * that measurement would change a path that folds today.
 *
 * The chains form reads a `[4][length]` identity - the residue index, then the
 * asymmetric-unit, entity and symmetry ids - where the monomer form reads the
 * residue index alone.
 */
const relativeBlockFor = (chains) => (chains
  ? `  // The multimer relative encoding: the offset, whether the two residues are
  // the same ENTITY, and how far apart their symmetry copies are.
  bool asymSame = fetch1(identityTex, L + i) == fetch1(identityTex, L + j);
  bool entitySame = fetch1(identityTex, 2 * L + i) == fetch1(identityTex, 2 * L + j);
  int rawOffset = int(fetch1(identityTex, i)) - int(fetch1(identityTex, j)) + MAX_RELATIVE;
  // Clamped, not wrapped: every separation past MAX_RELATIVE shares one bin.
  int clipped = clamp(rawOffset, 0, 2 * MAX_RELATIVE);
  // ...and the bin past the last offset means "a different chain entirely".
  int offsetRow = asymSame ? clipped : 2 * MAX_RELATIVE + 1;
  int entityRow = 2 * MAX_RELATIVE + 2;
  int symmetryDelta = int(fetch1(identityTex, 3 * L + i)) - int(fetch1(identityTex, 3 * L + j));
  int clippedChain = clamp(symmetryDelta + MAX_RELATIVE_CHAIN, 0, 2 * MAX_RELATIVE_CHAIN);
  int chainRow = entitySame ? clippedChain : 2 * MAX_RELATIVE_CHAIN + 1;
  total += fetch4(relativeBiasTex, group)
    + fetch4(relativeWeightTex, offsetRow * PAIR_C4 + group)
    + (entitySame ? fetch4(relativeWeightTex, entityRow * PAIR_C4 + group) : vec4(0.0))
    + fetch4(relativeWeightTex, (entityRow + 1 + chainRow) * PAIR_C4 + group);`
  : `  int rawOffset = int(fetch1(residueIndexTex, i)) - int(fetch1(residueIndexTex, j)) + MAX_RELATIVE;
  // Clamped, not wrapped: every separation past MAX_RELATIVE shares one bin.
  int relative = clamp(rawOffset, 0, 2 * MAX_RELATIVE);
  total += fetch4(relativeBiasTex, group) + fetch4(relativeWeightTex, relative * PAIR_C4 + group);`);

export const pairEmbedSource = (shape) => {
  const { length, targetChannels, pairChannels, chains = false } = shape;
  const relativeBlock = relativeBlockFor(chains);
  return `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int TARGET_C = ${targetChannels};
const int PAIR_C4 = ${pairChannels / 4};
const int ELEMENTS = ${length * length * pairChannels};
const int DGRAM_BINS = ${DGRAM_BINS};
const int MAX_RELATIVE = ${MAX_RELATIVE};
const int MAX_RELATIVE_CHAIN = ${MAX_RELATIVE_CHAIN};
const float MIN_BIN = ${MIN_BIN};
const float MAX_BIN = ${MAX_BIN};
uniform sampler2D targetTex;
uniform sampler2D previousPairTex;
uniform sampler2D previousPositionsTex;
uniform sampler2D aatypeTex;
uniform sampler2D ${chains ? "identityTex" : "residueIndexTex"};
uniform sampler2D leftWeightTex;
uniform sampler2D leftBiasTex;
uniform sampler2D rightWeightTex;
uniform sampler2D rightBiasTex;
uniform sampler2D positionWeightTex;
uniform sampler2D positionBiasTex;
uniform sampler2D relativeWeightTex;
uniform sampler2D relativeBiasTex;

// CB normally, CA for glycine, which has none.
float pseudoBeta(int residue, int axis) {
  int atom = int(fetch1(aatypeTex, residue)) == 7 ? 1 : 3;
  return fetch1(previousPositionsTex, (residue * 37 + atom) * 3 + axis);
}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % PAIR_C4;
  int pair = texel / PAIR_C4;
  int j = pair % L;
  int i = pair / L;

  vec4 total = fetch4(leftBiasTex, group) + fetch4(rightBiasTex, group);
  for (int c = 0; c < TARGET_C; ++c) {
    total += fetch1(targetTex, i * TARGET_C + c) * fetch4(leftWeightTex, c * PAIR_C4 + group);
    total += fetch1(targetTex, j * TARGET_C + c) * fetch4(rightWeightTex, c * PAIR_C4 + group);
  }

  total += fetch4(positionBiasTex, group);
  float d2 = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    float delta = pseudoBeta(i, axis) - pseudoBeta(j, axis);
    d2 += delta * delta;
  }
  float binWidth = (MAX_BIN - MIN_BIN) / float(DGRAM_BINS - 1);
  for (int bin = 0; bin < DGRAM_BINS; ++bin) {
    float lower = MIN_BIN + float(bin) * binWidth;
    bool last = bin + 1 == DGRAM_BINS;
    float upper = last ? MAX_BIN + binWidth : MIN_BIN + float(bin + 1) * binWidth;
    // A distance under MIN_BIN lands in no bin at all.
    if (d2 > lower * lower && (last || d2 < upper * upper)) {
      total += fetch4(positionWeightTex, bin * PAIR_C4 + group);
    }
  }

  total += fetch4(previousPairTex, texel);

${relativeBlock}
  result = total;
}`;
};

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadInputEmbedderWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of INPUT_EMBEDDER_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`input embedder weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.embedder.${name}`);
  }
  return out;
}

export function releaseInputEmbedderWeights(execution, weights) {
  for (const name of INPUT_EMBEDDER_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{targetFeatures: Tensor, msaFeatures: Tensor, extraMsa: Tensor,
 *   extraHasDeletion: Tensor, extraDeletionValue: Tensor, previousMsaFirstRow: Tensor,
 *   previousPair: Tensor, previousPositions: Tensor, aatype: Tensor,
 *   residueIndex: Tensor, residueIdentity?: Tensor, weights: Record<string, Tensor>,
 *   length: number, msaSequences: number, extraSequences: number,
 *   targetChannels: number, msaFeatureChannels: number, msaChannels: number,
 *   pairChannels: number, extraMsaChannels: number}} call
 * @returns {{msa: Tensor, extraMsa: Tensor, pairWithoutTemplates: Tensor}} the caller
 *   owns all three. The pair is named as the reference names it: the template
 *   stack adds to it afterwards, and a caller that treats this as the finished
 *   pair has skipped a stage.
 */
export function inputEmbedder(execution, call) {
  const {
    targetFeatures, msaFeatures, previousMsaFirstRow, previousPair, previousPositions,
    aatype, residueIndex, weights: w, length, msaSequences, extraSequences,
    targetChannels, msaFeatureChannels, msaChannels, pairChannels, extraMsaChannels,
  } = call;
  /** @type {[string, number][]} */
  const widths = [["msaChannels", msaChannels], ["pairChannels", pairChannels],
    ["extraMsaChannels", extraMsaChannels]];
  for (const [name, value] of widths) {
    if (value % 4 !== 0) {
      throw new RangeError(`inputEmbedder: ${name} must be a multiple of 4, got ${value}`);
    }
  }

  // Both recycled tensors are normalised before they are added.
  const normalisedMsa = layerNorm(execution, {
    x: previousMsaFirstRow, weight: w.previousMsaNormScale, bias: w.previousMsaNormOffset,
    rows: length, channels: msaChannels,
  });
  const normalisedPair = layerNorm(execution, {
    x: previousPair, weight: w.previousPairNormScale, bias: w.previousPairNormOffset,
    rows: length * length, channels: pairChannels,
  });

  const msa = execution.allocate(msaSequences * length * msaChannels, "embed.msa");
  execution.dispatch({
    source: msaEmbedSource({
      length, sequences: msaSequences, targetChannels, msaFeatureChannels, msaChannels,
    }),
    output: msa,
    inputs: {
      targetTex: targetFeatures, msaFeatureTex: msaFeatures, previousMsaTex: normalisedMsa,
      targetWeightTex: w.preprocess1dWeight, targetBiasTex: w.preprocess1dBias,
      msaWeightTex: w.preprocessMsaWeight, msaBiasTex: w.preprocessMsaBias,
    },
    label: `embed.msa:${msaSequences}x${length}x${msaChannels}`,
  });

  const extraMsa = execution.allocate(
    extraSequences * length * extraMsaChannels, "embed.extra-msa");
  execution.dispatch({
    source: extraMsaEmbedSource({
      length, sequences: extraSequences, extraChannels: extraMsaChannels,
    }),
    output: extraMsa,
    inputs: {
      codeTex: call.extraMsa, deletionFlagTex: call.extraHasDeletion,
      deletionValueTex: call.extraDeletionValue,
      weightTex: w.extraMsaWeight, biasTex: w.extraMsaBias,
    },
    label: `embed.extra:${extraSequences}x${length}x${extraMsaChannels}`,
  });

  const chains = call.residueIdentity !== undefined;
  const pair = execution.allocate(length * length * pairChannels, "embed.pair");
  execution.dispatch({
    source: pairEmbedSource({ length, targetChannels, pairChannels, chains }),
    output: pair,
    inputs: {
      targetTex: targetFeatures, previousPairTex: normalisedPair,
      previousPositionsTex: previousPositions, aatypeTex: aatype,
      // 🔴 THE PRESENCE OF THE IDENTITY IS THE SWITCH, and `dispatch` refuses a
      // binding the compiled source does not declare by name - so passing the
      // identity to the monomer kernel, or the residue index alone to the chains
      // kernel, raises rather than silently reading the wrong lane.
      ...(chains
        ? { identityTex: call.residueIdentity }
        : { residueIndexTex: residueIndex }),
      leftWeightTex: w.leftSingleWeight, leftBiasTex: w.leftSingleBias,
      rightWeightTex: w.rightSingleWeight, rightBiasTex: w.rightSingleBias,
      positionWeightTex: w.previousPositionWeight, positionBiasTex: w.previousPositionBias,
      relativeWeightTex: w.relativePositionWeight, relativeBiasTex: w.relativePositionBias,
    },
    label: `embed.pair:${length}x${pairChannels}`,
  });
  execution.release(normalisedMsa);
  execution.release(normalisedPair);
  return { msa, extraMsa, pairWithoutTemplates: pair };
}
