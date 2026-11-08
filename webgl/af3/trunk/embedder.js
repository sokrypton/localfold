/**
 * AF3's trunk embedder: what the two stacks are handed before they run.
 *
 *     pair    = left_single[i] + right_single[j]        from target_feat
 *     pair   += prev_embedding(LayerNorm(recycled pair))
 *     pair   += relative encoding
 *     pair   += bond embedding                          zero for a bond-free input,
 *                                                       a real term for ligands
 *     pair   += template embedding                      the CALLER adds this
 *     msa     = msa_activations(msa features) + extra_msa_target_feat(target_feat)
 *     single  = single_activations(target_feat)
 *     single += prev_single_embedding(LayerNorm(recycled single))
 *
 * 🔴 THE RECYCLED TERM IS NOT ZERO ON THE FIRST PASS. "AF3 starts from a zero
 * pair, but the LayerNorm ahead of the projection turns a zero input into its
 * OFFSET, and the projection of that offset is a constant this graph adds every
 * time. Skipping the whole branch 'because there is nothing to recycle' drops a
 * term that is present on pass one." Both recycled branches always run here.
 *
 * 🔴 AND THE TEMPLATE EMBEDDING IS NOT OPTIONAL, WHICH IS THE TRAP. "On a de
 * novo protein with FOUR EMPTY TEMPLATE SLOTS its output still has std 13.1
 * against a pair whose own std is about 55... AF2-multimer had the identical
 * trap and it cost this project a week: 'no templates' does not mean 'no
 * template contribution'." This module does NOT add it - the template stack
 * reads the pair as it is at this point, after the relative encoding and the
 * bonds - so the caller composes the two and `trunk` is where that order lives.
 *
 * 🔴 AND THE 139 ONE-HOT COLUMNS ARE NEVER MATERIALISED. The reference builds
 * `tokens^2 * 139` floats, which is 9.1 M at 256 tokens and 36 MiB; four of
 * those columns are 1 and the rest are 0, so the kernel here sums four weight
 * ROWS instead - the same trick AF2's relative encoding uses, and the reason
 * this stage costs one draw.
 *
 * 🔴 AND THE MSA FEATURE WIDTH IS THE WEIGHT'S, NOT A CONSTANT. "boltz2's is 35
 * and everyone else's is 34, and reading 34 off a [35, 64] matrix is a silent
 * prefix of the matrix, correct in its strides and missing a term." On a
 * single-sequence batch that missing term is the whole of what the MSA stack was
 * given, which is why it read 3.16e-1 with the z-init exact at 5.05e-8.
 *
 * 🔴 AND OpenDDE INITIALISES THE PAIR FROM THE SINGLE EMBEDDING, so
 * `pairInitFromSingle` has no default on either side: "AlphaFold 3's
 * `left_single` is [447, 128] and OpenDDE's is [384, 384]", and a bundle whose
 * flag and weights disagree would multiply a 447-wide feature by a 384-wide
 * matrix. Where it is set, the single embedding is the SAME tensor rather than a
 * second projection - "sharing is what says the two really are one embedding".
 */
import { ONE_TARGET, PRELUDE, sized, memoSource } from "../../runtime/execution.js";
import { add } from "../../runtime/elementwise.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const MAX_RELATIVE_IDX = 32;
export const MAX_RELATIVE_CHAIN = 2;
/** 66 position bins, twice, then same-entity, then 6 chain bins. */
export const POSITION_BINS = 2 * MAX_RELATIVE_IDX + 2;
export const CHAIN_BINS = 2 * MAX_RELATIVE_CHAIN + 2;
export const RELATIVE_WIDTH = POSITION_BINS * 2 + 1 + CHAIN_BINS;

export const EMBEDDER_WEIGHTS = [
  "leftSingle", "rightSingle", "bondEmbedding",
  "prevEmbeddingNormScale", "prevEmbeddingNormOffset", "prevEmbedding",
  "positionActivations", "msaActivations", "extraMsaTargetFeat", "singleActivations",
  "prevSingleEmbeddingNormScale", "prevSingleEmbeddingNormOffset", "prevSingleEmbedding",
];

/**
 * The per-token identity the relative encoding reads: `[5][tokens]` holding the
 * residue index, the token index, then the asym, entity and symmetry ids.
 */
export function tokenIdentity(features, tokens) {
  const lanes = ["residueIndex", "tokenIndex", "asymId", "entityId", "symId"];
  const out = new Float32Array(lanes.length * tokens);
  for (const [index, name] of lanes.entries()) {
    const lane = features[name];
    if (lane === undefined) throw new Error(`token identity: ${name} is missing`);
    if (lane.length !== tokens) {
      throw new RangeError(`token identity: ${name} has ${lane.length} values, not ${tokens}`);
    }
    out.set(Float32Array.from(lane), index * tokens);
  }
  return out;
}

/**
 * `pair[i][j][c] += bondMatrix[i][j] * bondEmbedding[c]` - the bond embedding.
 *
 * 🔴 IT IS A ONE-COLUMN Linear, WHICH IS WHY IT IS NOT `kernels/linear.js`. That
 * kernel reads whole texels of the input and refuses an input width that is not
 * a multiple of four; here the width is ONE. The term is bias-free, so it is
 * exactly zero for a polymer with no covalent links - "measured, not assumed:
 * the oracle's bond_embedding output is 0.0000 on this input. It is a real term
 * for ligands."
 */
const bondEmbeddingSourceUncached = (tokens, pairChannels) => sized({ ELEMENTS: tokens * tokens * pairChannels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${pairChannels / 4};
uniform sampler2D pairTex;
uniform sampler2D bondTex;
uniform sampler2D weightTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  result = fetch4(pairTex, texel) + fetch1(bondTex, pair) * fetch4(weightTex, group);
}`);

/** `pair[i][j][c] = left[i][c] + right[j][c]`. */
const pairInitSourceUncached = (tokens, pairChannels) => sized({ L: tokens, ELEMENTS: tokens * tokens * pairChannels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${pairChannels / 4};
uniform sampler2D leftTex;
uniform sampler2D rightTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  result = fetch4(leftTex, i * C4 + group) + fetch4(rightTex, j * C4 + group);
}`);

/**
 * The relative encoding, as four weight rows summed rather than 139 columns
 * multiplied.
 *
 * The four are: the chain-relative position (or one extra bin for anything
 * inter-chain), the residue-relative token offset (or its own extra bin), the
 * same-entity flag, and the relative symmetry copy (or an extra bin for a
 * different entity). Everything else in the 139 is zero.
 */
const relativeEncodingSourceUncached = (tokens, pairChannels) => sized({ L: tokens, ELEMENTS: tokens * tokens * pairChannels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${pairChannels / 4};
const int MAX_IDX = ${MAX_RELATIVE_IDX};
const int MAX_CHAIN = ${MAX_RELATIVE_CHAIN};
const int POSITION_BINS = ${POSITION_BINS};
uniform sampler2D identityTex;
uniform sampler2D weightTex;

int clampBin(int value, int high) { return clamp(value, 0, high); }

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;

  float residueI = fetch1(identityTex, i);
  float residueJ = fetch1(identityTex, j);
  float tokenI = fetch1(identityTex, L + i);
  float tokenJ = fetch1(identityTex, L + j);
  bool sameChain = fetch1(identityTex, 2 * L + i) == fetch1(identityTex, 2 * L + j);
  bool sameEntity = fetch1(identityTex, 3 * L + i) == fetch1(identityTex, 3 * L + j);
  float symI = fetch1(identityTex, 4 * L + i);
  float symJ = fetch1(identityTex, 4 * L + j);

  // ...distance along the chain, with everything inter-chain sharing ONE extra
  // bin rather than being clipped into the far end of the range.
  int offset = clampBin(int(residueI) - int(residueJ) + MAX_IDX, 2 * MAX_IDX);
  int positionRow = sameChain ? offset : 2 * MAX_IDX + 1;
  // ...distance along the RESIDUE, which means anything only for two tokens of
  // the same residue of the same chain.
  bool sameResidue = sameChain && residueI == residueJ;
  int tokenOffset = clampBin(int(tokenI) - int(tokenJ) + MAX_IDX, 2 * MAX_IDX);
  int tokenRow = POSITION_BINS + (sameResidue ? tokenOffset : 2 * MAX_IDX + 1);
  // ...and which copy of a repeated chain this is, within its symmetry class.
  int relativeChain = clampBin(int(symI) - int(symJ) + MAX_CHAIN, 2 * MAX_CHAIN);
  int chainRow = POSITION_BINS * 2 + 1
    + (sameEntity ? relativeChain : 2 * MAX_CHAIN + 1);

  vec4 total = fetch4(weightTex, positionRow * C4 + group)
    + fetch4(weightTex, tokenRow * C4 + group)
    + fetch4(weightTex, chainRow * C4 + group);
  if (sameEntity) total += fetch4(weightTex, (POSITION_BINS * 2) * C4 + group);
  result = total;
}`);

/**
 * The MSA embedding, with the 34 (or 35) feature columns folded in.
 *
 *     code       a 32-way one-hot, so one weight row
 *     column 32  the deletion count, clamped to [0, 1]
 *     column 33  `atan(deletions / 3) * 2 / pi` - "arctan-squashed rather than
 *                clipped, so a column with many deletions stays distinguishable
 *                from one with a few instead of saturating"
 *     column 34  boltz2's `is_paired`, 1 on the QUERY ROW where it exists
 */
const msaEmbedSourceUncached = (
  sequences, tokens, width, msaChannels, pairedQueryRow,
) => sized({ L: tokens, ELEMENTS: sequences * tokens * msaChannels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${msaChannels / 4};
const int WIDTH = ${width};
uniform sampler2D codeTex;
uniform sampler2D deletionTex;
uniform sampler2D weightTex;
uniform sampler2D fromTargetTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  int token = row % L;
  int sequence = row / L;

  vec4 total = fetch4(fromTargetTex, token * C4 + group);
  int code = int(fetch1(codeTex, row));
  if (code >= 0 && code < 32) total += fetch4(weightTex, code * C4 + group);
  float deletions = fetch1(deletionTex, row);
  total += clamp(deletions, 0.0, 1.0) * fetch4(weightTex, 32 * C4 + group);
  total += atan(deletions / 3.0) * (2.0 / 3.14159265358979323846)
    * fetch4(weightTex, 33 * C4 + group);
${width > 34 && pairedQueryRow
    ? "  if (sequence == 0) total += fetch4(weightTex, 34 * C4 + group);"
    : "  // no is_paired column, or the query row does not carry it."}
  result = total;
}`);

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadEmbedderWeights(execution, weights) {
  /** @type {any} */
  const out = {
    pairChannels: weights.pairChannels,
    singleChannels: weights.singleChannels,
    msaChannels: weights.msaChannels,
    targetFeatWidth: weights.targetFeatWidth,
  };
  for (const name of EMBEDDER_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`af3 embedder weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.af3.embed.${name}`);
  }
  return out;
}

export function releaseEmbedderWeights(execution, weights) {
  for (const name of EMBEDDER_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{targetFeat: Tensor, identity: Tensor, msaCodes: Tensor, deletions: Tensor,
 *   previousPair: Tensor, previousSingle: Tensor,
 *   weights: ReturnType<typeof uploadEmbedderWeights>,
 *   bondMatrix?: Tensor, tokens: number, sequences: number,
 *   pairInitFromSingle: boolean, msaPairedQueryRow?: boolean, epsilon?: number}} call
 * @returns {{pair: Tensor, msa: Tensor, single: Tensor}} the caller owns all
 *   three - and must add the TEMPLATE embedding to the pair before the stacks
 */
export function embed(execution, call) {
  const {
    targetFeat, identity, msaCodes, deletions, previousPair, previousSingle,
    weights: w, tokens, sequences, epsilon = 1e-5,
  } = call;
  if (typeof call.pairInitFromSingle !== "boolean") {
    throw new Error(
      "af3 embed: pairInitFromSingle has no default - stock AF3 builds the pair from"
      + " target_feat and OpenDDE from s_init, and their weight shapes differ");
  }
  const pairChannels = w.pairChannels;
  const singleChannels = w.singleChannels;
  const msaChannels = w.msaChannels;
  const featureWidth = w.targetFeatWidth;
  const pairs = tokens * tokens;
  const rows = sequences * tokens;

  // The single embedding, computed FIRST where the pair init reads it - and then
  // shared rather than recomputed.
  const singleEmbedding = linear(execution, {
    x: targetFeat, weight: w.singleActivations, rows: tokens,
    inChannels: featureWidth, outChannels: singleChannels, weightLayout: "in-major",
  });
  const pairSource = call.pairInitFromSingle ? singleEmbedding : targetFeat;
  const pairSourceWidth = call.pairInitFromSingle ? singleChannels : featureWidth;
  const left = linear(execution, {
    x: pairSource, weight: w.leftSingle, rows: tokens,
    inChannels: pairSourceWidth, outChannels: pairChannels, weightLayout: "in-major",
  });
  const right = linear(execution, {
    x: pairSource, weight: w.rightSingle, rows: tokens,
    inChannels: pairSourceWidth, outChannels: pairChannels, weightLayout: "in-major",
  });
  let pair = execution.allocate(pairs * pairChannels, "af3.embed.pair");
  execution.dispatch({
    source: pairInitSource(tokens, pairChannels),
    output: pair,
    inputs: { leftTex: left, rightTex: right },
    label: `af3.embed.pair-init:${tokens}x${pairChannels}`,
  });
  execution.release(left);
  execution.release(right);

  /** Thread a term into the running pair and release what it replaces. */
  const addPair = (update) => {
    const next = add(execution, pair, update);
    execution.release(update);
    execution.release(pair);
    pair = next;
  };

  // The recycled pair - which contributes on pass one too, through the norm's
  // offset.
  const normalisedPrevious = layerNorm(execution, {
    x: previousPair, weight: w.prevEmbeddingNormScale, bias: w.prevEmbeddingNormOffset,
    rows: pairs, channels: pairChannels, epsilon, variance: "fast",
  });
  addPair(linear(execution, {
    x: normalisedPrevious, weight: w.prevEmbedding, rows: pairs,
    inChannels: pairChannels, outChannels: pairChannels, weightLayout: "in-major",
  }));
  execution.release(normalisedPrevious);

  // The bond embedding, where the caller has a contact matrix. It reads the
  // pair and writes a new one rather than producing an update, because a
  // one-column Linear is cheaper fused than as an add.
  if (call.bondMatrix !== undefined) {
    const bonded = execution.allocate(pairs * pairChannels, "af3.embed.bonded");
    execution.dispatch({
      source: bondEmbeddingSource(tokens, pairChannels),
      output: bonded,
      inputs: { pairTex: pair, bondTex: call.bondMatrix, weightTex: w.bondEmbedding },
      label: `af3.embed.bonds:${tokens}x${pairChannels}`,
    });
    execution.release(pair);
    pair = bonded;
  }

  const relative = execution.allocate(pairs * pairChannels, "af3.embed.relative");
  execution.dispatch({
    source: relativeEncodingSource(tokens, pairChannels),
    output: relative,
    inputs: { identityTex: identity, weightTex: w.positionActivations },
    label: `af3.embed.relative:${tokens}x${pairChannels}`,
  });
  addPair(relative);

  // The MSA track. The feature width is the WEIGHT's; 34 for every model but
  // boltz2, whose 35th column is `is_paired`.
  const msaFeatureWidth = w.msaActivations.elements / msaChannels;
  if (!Number.isInteger(msaFeatureWidth) || msaFeatureWidth < 34) {
    throw new RangeError(
      `af3 embed: msaActivations hold ${w.msaActivations.elements} floats, which is not`
      + ` at least 34 x ${msaChannels}`);
  }
  if (msaFeatureWidth > 34 && call.msaPairedQueryRow === undefined) {
    throw new Error(
      "af3 embed: this bundle carries an is_paired column and only the dialect says"
      + " whether the query row is in it - boltz2 says yes, rosettafold3 no");
  }
  const fromTarget = linear(execution, {
    x: targetFeat, weight: w.extraMsaTargetFeat, rows: tokens,
    inChannels: featureWidth, outChannels: msaChannels, weightLayout: "in-major",
  });
  const msa = execution.allocate(rows * msaChannels, "af3.embed.msa");
  execution.dispatch({
    source: msaEmbedSource(
      sequences, tokens, msaFeatureWidth, msaChannels, call.msaPairedQueryRow === true),
    output: msa,
    inputs: {
      codeTex: msaCodes, deletionTex: deletions,
      weightTex: w.msaActivations, fromTargetTex: fromTarget,
    },
    label: `af3.embed.msa:${sequences}x${tokens}x${msaChannels}:${msaFeatureWidth}`,
  });
  execution.release(fromTarget);

  // The single track, plus its own recycled term.
  const normalisedPreviousSingle = layerNorm(execution, {
    x: previousSingle, weight: w.prevSingleEmbeddingNormScale,
    bias: w.prevSingleEmbeddingNormOffset,
    rows: tokens, channels: singleChannels, epsilon, variance: "fast",
  });
  const recycledSingle = linear(execution, {
    x: normalisedPreviousSingle, weight: w.prevSingleEmbedding, rows: tokens,
    inChannels: singleChannels, outChannels: singleChannels, weightLayout: "in-major",
  });
  execution.release(normalisedPreviousSingle);
  const single = add(execution, singleEmbedding, recycledSingle);
  execution.release(recycledSingle);
  execution.release(singleEmbedding);

  pair.label = "af3.embed.pair";
  msa.label = "af3.embed.msa";
  single.label = "af3.embed.single";
  return { pair, msa, single };
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const bondEmbeddingSource = memoSource(bondEmbeddingSourceUncached);
export const pairInitSource = memoSource(pairInitSourceUncached);
export const relativeEncodingSource = memoSource(relativeEncodingSourceUncached);
export const msaEmbedSource = memoSource(msaEmbedSourceUncached);
