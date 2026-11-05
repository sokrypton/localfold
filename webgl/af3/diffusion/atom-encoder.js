/**
 * AF3's atom cross-attention encoder: per-atom conditioning in, per-token
 * features out.
 *
 *     queriesCond = gather(conditioning) + trunkSingle, MASKED, then gathered
 *                   onto the keys
 *     pair        = rowCond + columnCond
 *                   + valid * (offset . W + W_dist / (1 + d^2))
 *                   + valid * W_validFlag
 *                   + trunkPair[tokenOf(query)][tokenOf(key)]
 *     pair       += Linear(relu(Linear(relu(Linear(relu(pair))))))
 *     act         = crossAttentionBlock(act, ...) x blocks
 *     out[token]  = mean over the token's REAL atoms of relu(Linear(act))
 *
 * 🔴 THE CONDITIONING IS MASKED BEFORE THE KEYS GATHER FROM IT, NOT AFTER.
 * "This is a no-op for the trunk's encoder - _per_atom_conditioning already
 * masked its output, so the two orders agree - and it is NOT a no-op once the
 * trunk conditioning is added above, because that puts non-zero values into
 * padded atom slots. Gathering first carries them into the keys, where two
 * thirds of the slots are padding: measured 8.4e-2 on the encoder's output,
 * which reads like a subtly wrong kernel rather than a line in the wrong order."
 *
 * 🔴 AND THE VALIDITY RULE LETS PADDED KEYS IN, WHICH IS A PROPERTY OF THE
 * WEIGHTS AND NOT A BUG TO FIX. Two atoms have a meaningful offset only if they
 * came from the same reference conformer - but "a padded key slot gathers a zero
 * reference space, and the FIRST reference conformer's uid is also zero - so
 * every padded key reads as a valid neighbour of token 0, and the N-terminus is
 * conditioned on a hundred atoms that do not exist. AF3 was released this way
 * and OpenFold3 was trained with the mask, so this is a property of the WEIGHTS
 * rather than a bug either side is free to fix: upstream measures the released
 * graph putting CA-C at ~0.85 A against an ideal 1.52." `maskPaddedKeys` is that
 * switch and it has no default.
 *
 * 🔴 AND THE VALIDITY FLAG ITSELF IS NOT GATED BY VALIDITY: "'these two atoms
 * are unrelated' is information the model uses", so `valid * W_validFlag` adds
 * the weight where the pair IS valid and nothing where it is not - which is the
 * opposite of what gating the flag would do.
 *
 * 🔴 AND `tokens_to_keys` IS IN THE BATCH; DO NOT DERIVE IT. "Carrying
 * tokens_to_queries through the queries-to-keys gather looks equivalent and is a
 * second source of truth for something the featuriser already computed - and its
 * MASK is not the same, because a derived one folds in the query mask where
 * AF3's is the key's own."
 *
 * 🔴 AND THE AGGREGATION AVERAGES OVER A TOKEN'S REAL ATOMS ONLY, after a relu -
 * so a token with three atoms of a 24-slot row divides by three.
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";
import {
  atomPairLogits, crossAttentionBlock, gather, sumSource,
} from "./atom-attention.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import { relu } from "../../runtime/elementwise.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/** `out[row][c] = x[row][c] * mask[row]` - the mask-before-gather step. */
export const maskRowsSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D xTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int row = texel / C4;
  result = fetch4(xTex, texel) * fetch1(maskTex, row);
}`;

/**
 * The atom pair conditioning: five terms, each with its own masking rule.
 *
 * One draw rather than five, because every one of them is indexed by the same
 * (subset, query, key) and four of the five are weight rows scaled by a scalar.
 */
export const atomPairSource = (shape, maskPaddedKeys, hasTrunkPair) => {
  const { subsets, queries, keys, pairChannels, tokens } = shape;
  return `${PRELUDE}${ONE_TARGET}
const int QUERIES = ${queries};
const int KEYS = ${keys};
const int P4 = ${pairChannels / 4};
const int TOKENS = ${tokens};
const int ELEMENTS = ${subsets * queries * keys * pairChannels};
uniform sampler2D rowTex;
uniform sampler2D columnTex;
uniform sampler2D queryPosTex;
uniform sampler2D keyPosTex;
uniform sampler2D queryUidTex;
uniform sampler2D keyUidTex;
${maskPaddedKeys ? "uniform sampler2D keyMaskTex;" : ""}
uniform sampler2D offsetWeightTex;
uniform sampler2D distanceWeightTex;
uniform sampler2D validWeightTex;
${hasTrunkPair ? `uniform sampler2D trunkPairTex;
uniform sampler2D tokensToQueriesTex;
uniform sampler2D tokensToQueriesMaskTex;
uniform sampler2D tokensToKeysTex;
uniform sampler2D tokensToKeysMaskTex;` : ""}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % P4;
  int rest = texel / P4;
  int key = rest % KEYS;
  int rest2 = rest / KEYS;
  int query = rest2 % QUERIES;
  int subset = rest2 / QUERIES;
  int queryIndex = subset * QUERIES + query;
  int keyIndex = subset * KEYS + key;

  vec4 total = fetch4(rowTex, queryIndex * P4 + group)
    + fetch4(columnTex, keyIndex * P4 + group);

  // The two atoms share a reference conformer - and under stock AF3 a PADDED
  // key passes this test, because its gathered uid is zero and so is the first
  // conformer's. See the note at the top: that is the released graph.
  bool sameSpace = fetch1(queryUidTex, queryIndex) == fetch1(keyUidTex, keyIndex);
  ${maskPaddedKeys
    ? "bool valid = sameSpace && fetch1(keyMaskTex, keyIndex) != 0.0;"
    : "bool valid = sameSpace;"}
  if (valid) {
    // 🔴 FETCHED AS A WHOLE TEXEL, BECAUSE THE GATHER STORES AT STRIDE FOUR.
    // The vec3 gather writes three floats into an RGBA texel and leaves the
    // fourth zero, so the row stride is 4 and not 3 - and reading at stride 3
    // here walked three quarters of the way through the array, which measured
    // relRMS 0.716 on pairBeforeMlp while the row, the column, both uids and
    // the key mask were all exact. The taps are what turned that into one line.
    vec3 offset = fetch4(queryPosTex, queryIndex).xyz
      - fetch4(keyPosTex, keyIndex).xyz;
    total += offset.x * fetch4(offsetWeightTex, group)
      + offset.y * fetch4(offsetWeightTex, P4 + group)
      + offset.z * fetch4(offsetWeightTex, 2 * P4 + group);
    total += fetch4(distanceWeightTex, group) / (1.0 + dot(offset, offset));
    // ...and the validity flag itself, added where the pair IS valid.
    total += fetch4(validWeightTex, group);
  }
${hasTrunkPair ? `  // The trunk's pair, indexed by the two atoms' TOKENS - both ends real.
  if (fetch1(tokensToQueriesMaskTex, queryIndex) != 0.0
      && fetch1(tokensToKeysMaskTex, keyIndex) != 0.0) {
    int from = int(fetch1(tokensToQueriesTex, queryIndex)) * TOKENS
      + int(fetch1(tokensToKeysTex, keyIndex));
    total += fetch4(trunkPairTex, from * P4 + group);
  }` : "  // The trunk's encoder runs before there is a pair to condition on."}
  result = total;
}`;
};

/**
 * `out[token][c] = mean over the token's REAL atoms of relu(projected[...][c])`
 *
 * The divisor is the token's own atom count, not the dense width - "so a token
 * with three atoms of a 24-slot row divides by three" - and a token with none
 * stays zero rather than dividing by zero.
 */
export const aggregateSource = (tokens, dense, channels) => `${PRELUDE}${ONE_TARGET}
const int DENSE = ${dense};
const int C4 = ${channels / 4};
const int ELEMENTS = ${tokens * channels};
uniform sampler2D projectedTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int token = texel / C4;
  vec4 total = vec4(0.0);
  float count = 0.0;
  for (int atom = 0; atom < DENSE; ++atom) {
    int slot = token * DENSE + atom;
    if (fetch1(maskTex, slot) == 0.0) continue;
    count += 1.0;
    total += max(fetch4(projectedTex, slot * C4 + group), vec4(0.0));
  }
  result = count > 0.0 ? total / count : vec4(0.0);
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{conditioning: Tensor, atomMask: Tensor, refPos: Tensor, refSpaceUid: Tensor,
 *   trunkSingleCond?: Tensor, trunkPairCond?: Tensor,
 *   tokenAtomsAct?: Tensor,
 *   gathers: {tokenAtomsToQueries: {indices: Tensor, mask: Tensor},
 *     queriesToKeys: {indices: Tensor, mask: Tensor},
 *     queriesToTokenAtoms: {indices: Tensor, mask: Tensor},
 *     tokensToQueries: {indices: Tensor, mask: Tensor},
 *     tokensToKeys: {indices: Tensor, mask: Tensor}},
 *   shape: {tokens: number, dense: number, subsets: number, queries: number,
 *     keys: number},
 *   weights: any, maskPaddedKeys: boolean, keysUidFromQueriesLayout: boolean,
 *   keyMaskedAtomAttention: boolean, chainedAtomLayerNorm: boolean,
 *   diffusionNoResidual: boolean, pairNormPerBlock: boolean,
 *   maskAtomActPerBlock: boolean,
 *   preTrunkQuery?: boolean, epsilon?: number,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {{tokenAct: Tensor, skipConnection: Tensor, queriesMask: Tensor,
 *   keysMask: Tensor, queriesCond: Tensor, keysCond: Tensor, pairCond: Tensor}}
 *   the caller owns all seven - "the decoder needs everything the encoder
 *   computed, not just its output"
 */
export function atomCrossAttentionEncoder(execution, call) {
  const { conditioning, atomMask, refPos, refSpaceUid, gathers, weights: w, epsilon = 1e-5 } = call;
  const { tokens, dense, subsets, queries, keys } = call.shape;
  for (const name of ["maskPaddedKeys", "keysUidFromQueriesLayout"]) {
    if (typeof call[name] !== "boolean") {
      throw new Error(
        `af3 atom encoder: ${name} has no default - see the note at the top of this`
        + " module for why the padded-key rule is a property of the weights");
    }
  }
  const channels = w.channels;
  const pairChannels = w.pairChannels;
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;

  // The per-atom conditioning onto the query layout, and its mask.
  let queriesCond = gather(execution, {
    source: conditioning, indices: gathers.tokenAtomsToQueries.indices,
    mask: gathers.tokenAtomsToQueries.mask, count: queryRows, channels,
  });
  // ...the mask is ONE channel, which `gather` cannot do - its channels must be
  // a multiple of four - so it takes its own kernel.
  const queriesMaskTensor = execution.allocate(queryRows, "af3.encoder.queries-mask");
  execution.dispatch({
    source: scalarGatherSource(queryRows),
    output: queriesMaskTensor,
    inputs: {
      sourceTex: atomMask, indicesTex: gathers.tokenAtomsToQueries.indices,
      maskTex: gathers.tokenAtomsToQueries.mask,
    },
    label: `af3.encoder.queries-mask:${queryRows}`,
  });

  // 🔴 THE TRUNK'S SINGLE CONDITIONING IS BROADCAST PER TOKEN, NOT PER ATOM,
  // and it goes in BEFORE the keys are gathered - so every key sees it too.
  // Only the diffusion head passes it.
  const preTrunkQuery = call.preTrunkQuery === true;
  let queriesBeforeTrunk;
  if (preTrunkQuery) {
    queriesBeforeTrunk = execution.allocate(queryRows * channels, "af3.encoder.pre-trunk");
    execution.dispatch({
      source: copySource(queryRows * channels),
      output: queriesBeforeTrunk,
      inputs: { xTex: queriesCond },
      label: `af3.encoder.pre-trunk:${queryRows}x${channels}`,
    });
  }
  if (call.trunkSingleCond !== undefined) {
    const normalised = layerNorm(execution, {
      x: call.trunkSingleCond, weight: w.lnormTrunkSingleCondScale,
      bias: w.lnormTrunkSingleCondOffset,
      rows: tokens, channels: w.trunkSingleChannels, epsilon,
    });
    const projected = linear(execution, {
      x: normalised, weight: w.embedTrunkSingleCond, rows: tokens,
      inChannels: w.trunkSingleChannels, outChannels: channels, weightLayout: "in-major",
    });
    execution.release(normalised);
    const perQuery = gather(execution, {
      source: projected, indices: gathers.tokensToQueries.indices,
      mask: gathers.tokensToQueries.mask, count: queryRows, channels,
    });
    execution.release(projected);
    const summed = execution.allocate(queryRows * channels, "af3.encoder.cond-sum");
    execution.dispatch({
      source: sumSource(queryRows * channels),
      output: summed,
      inputs: { aTex: queriesCond, bTex: perQuery },
      label: `af3.encoder.cond-sum:${queryRows}x${channels}`,
    });
    execution.release(perQuery);
    execution.release(queriesCond);
    queriesCond = summed;
  }

  // 🔴 MASKED BEFORE THE KEYS ARE GATHERED FROM IT - see the note at the top.
  const maskedCond = execution.allocate(queryRows * channels, "af3.encoder.cond-masked");
  execution.dispatch({
    source: maskRowsSource(queryRows, channels),
    output: maskedCond,
    inputs: { xTex: queriesCond, maskTex: queriesMaskTensor },
    label: `af3.encoder.cond-mask:${queryRows}x${channels}`,
  });
  execution.release(queriesCond);
  queriesCond = maskedCond;

  const keysMaskTensor = execution.allocate(keyRows, "af3.encoder.keys-mask");
  execution.dispatch({
    source: scalarGatherSource(keyRows),
    output: keysMaskTensor,
    inputs: {
      sourceTex: queriesMaskTensor, indicesTex: gathers.queriesToKeys.indices,
      maskTex: gathers.queriesToKeys.mask,
    },
    label: `af3.encoder.keys-mask:${keyRows}`,
  });
  const keysCond = gather(execution, {
    source: queriesCond, indices: gathers.queriesToKeys.indices,
    mask: gathers.queriesToKeys.mask, count: keyRows, channels,
  });

  // The query activation: the conditioning, plus the noisy positions where the
  // diffusion head supplies them.
  let queriesAct;
  if (preTrunkQuery) {
    queriesAct = execution.allocate(queryRows * channels, "af3.encoder.act");
    execution.dispatch({
      source: maskRowsSource(queryRows, channels),
      output: queriesAct,
      inputs: { xTex: queriesBeforeTrunk, maskTex: queriesMaskTensor },
      label: `af3.encoder.act-mask:${queryRows}x${channels}`,
    });
    execution.release(queriesBeforeTrunk);
  } else {
    queriesAct = execution.allocate(queryRows * channels, "af3.encoder.act");
    execution.dispatch({
      source: copySource(queryRows * channels),
      output: queriesAct,
      inputs: { xTex: queriesCond },
      label: `af3.encoder.act:${queryRows}x${channels}`,
    });
  }
  if (call.tokenAtomsAct !== undefined) {
    const gatheredPositions = execution.allocate(queryRows * 4, "af3.encoder.positions");
    execution.dispatch({
      source: vec3GatherSource(queryRows, 3),
      output: gatheredPositions,
      inputs: {
        sourceTex: call.tokenAtomsAct, indicesTex: gathers.tokenAtomsToQueries.indices,
        maskTex: gathers.tokenAtomsToQueries.mask,
      },
      label: `af3.encoder.gather-positions:${queryRows}`,
    });
    const positional = execution.allocate(queryRows * channels, "af3.encoder.positional");
    execution.dispatch({
      source: positionToFeaturesSource(queryRows, channels),
      output: positional,
      inputs: {
        positionsTex: gatheredPositions, weightTex: w.atomPositionsToFeatures,
        maskTex: queriesMaskTensor, actTex: queriesAct,
      },
      label: `af3.encoder.positional:${queryRows}x${channels}`,
    });
    execution.release(gatheredPositions);
    execution.release(queriesAct);
    queriesAct = positional;
  }

  call.onStage?.("encoder.queriesCond", queriesCond);

  // The pair conditioning's row and column terms, off the RECTIFIED cond.
  const rectifiedQueries = relu(execution, queriesCond);
  const rectifiedKeys = relu(execution, keysCond);
  const row = linear(execution, {
    x: rectifiedQueries, weight: w.singleToPairCondRow, rows: queryRows,
    inChannels: channels, outChannels: pairChannels, weightLayout: "in-major",
  });
  const column = linear(execution, {
    x: rectifiedKeys, weight: w.singleToPairCondCol, rows: keyRows,
    inChannels: channels, outChannels: pairChannels, weightLayout: "in-major",
  });
  execution.release(rectifiedQueries);
  execution.release(rectifiedKeys);
  call.onStage?.("encoder.pairCondRow", row);
  call.onStage?.("encoder.pairCondCol", column);

  // The trunk's pair, projected once if it is there at all.
  let trunkPair;
  if (call.trunkPairCond !== undefined) {
    const normalised = layerNorm(execution, {
      x: call.trunkPairCond, weight: w.lnormTrunkPairCondScale,
      bias: w.lnormTrunkPairCondOffset,
      rows: tokens * tokens, channels: w.trunkPairChannels, epsilon,
    });
    trunkPair = linear(execution, {
      x: normalised, weight: w.embedTrunkPairCond, rows: tokens * tokens,
      inChannels: w.trunkPairChannels, outChannels: pairChannels,
      weightLayout: "in-major",
    });
    execution.release(normalised);
  }

  // The reference positions and spaces, on both layouts.
  const queriesRefPos = execution.allocate(queryRows * 4, "af3.encoder.query-pos");
  execution.dispatch({
    source: vec3GatherSource(queryRows, 3),
    output: queriesRefPos,
    inputs: {
      sourceTex: refPos, indicesTex: gathers.tokenAtomsToQueries.indices,
      maskTex: gathers.tokenAtomsToQueries.mask,
    },
    label: `af3.encoder.query-pos:${queryRows}`,
  });
  const keysRefPos = execution.allocate(keyRows * 4, "af3.encoder.key-pos");
  execution.dispatch({
    // ...stride FOUR, because its source is the gather above.
    source: vec3GatherSource(keyRows, 4),
    output: keysRefPos,
    inputs: {
      sourceTex: queriesRefPos, indicesTex: gathers.queriesToKeys.indices,
      maskTex: gathers.queriesToKeys.mask,
    },
    label: `af3.encoder.key-pos:${keyRows}`,
  });
  const queriesUid = execution.allocate(queryRows, "af3.encoder.query-uid");
  execution.dispatch({
    source: scalarGatherSource(queryRows),
    output: queriesUid,
    inputs: {
      sourceTex: refSpaceUid, indicesTex: gathers.tokenAtomsToQueries.indices,
      maskTex: gathers.tokenAtomsToQueries.mask,
    },
    label: `af3.encoder.query-uid:${queryRows}`,
  });
  // 🔴 STOCK AF3 GATHERS THE KEYS' UID FROM THE **TOKEN** LAYOUT with indices
  // written for the QUERIES layout, clamped because a masked entry may point
  // past the end. `keysUidFromQueriesLayout` is the other reading, and the two
  // differ wherever the two layouts do.
  const keysUid = execution.allocate(keyRows, "af3.encoder.key-uid");
  execution.dispatch({
    source: call.keysUidFromQueriesLayout
      ? scalarGatherSource(keyRows)
      : clampedGatherSource(keyRows, refSpaceUid.elements),
    output: keysUid,
    inputs: {
      sourceTex: call.keysUidFromQueriesLayout ? queriesUid : refSpaceUid,
      indicesTex: gathers.queriesToKeys.indices,
      maskTex: gathers.queriesToKeys.mask,
    },
    label: `af3.encoder.key-uid:${keyRows}`
      + `${call.keysUidFromQueriesLayout ? ":queries" : ":tokens"}`,
  });

  call.onStage?.("encoder.queriesUid", queriesUid);
  call.onStage?.("encoder.keysUid", keysUid);
  call.onStage?.("encoder.keysMask", keysMaskTensor);
  let pair = execution.allocate(
    subsets * queries * keys * pairChannels, "af3.encoder.pair");
  execution.dispatch({
    source: atomPairSource(
      { subsets, queries, keys, pairChannels, tokens },
      call.maskPaddedKeys, trunkPair !== undefined),
    output: pair,
    inputs: {
      rowTex: row, columnTex: column,
      queryPosTex: queriesRefPos, keyPosTex: keysRefPos,
      queryUidTex: queriesUid, keyUidTex: keysUid,
      // ...bound only where the source reads it: with the padded-key rule off
      // the compiler strips the sampler and `dispatch` refuses it by name.
      ...(call.maskPaddedKeys ? { keyMaskTex: keysMaskTensor } : {}),
      offsetWeightTex: w.embedPairOffsets, distanceWeightTex: w.embedPairDistances,
      validWeightTex: w.embedPairOffsetsValid,
      ...(trunkPair === undefined ? {} : {
        trunkPairTex: trunkPair,
        tokensToQueriesTex: gathers.tokensToQueries.indices,
        tokensToQueriesMaskTex: gathers.tokensToQueries.mask,
        tokensToKeysTex: gathers.tokensToKeys.indices,
        tokensToKeysMaskTex: gathers.tokensToKeys.mask,
      }),
    },
    label: `af3.encoder.pair:${subsets}x${queries}x${keys}x${pairChannels}`,
  });
  for (const tensor of [row, column, queriesRefPos, keysRefPos, queriesUid, keysUid]) {
    execution.release(tensor);
  }
  if (trunkPair !== undefined) execution.release(trunkPair);

  call.onStage?.("encoder.pairBeforeMlp", pair);

  // The three-layer pair MLP, each layer reading a relu of the last, added back.
  const pairRows = subsets * queries * keys;
  let hidden = pair;
  let mlpIndex = 0;
  for (const weight of [w.pairMlp1, w.pairMlp2, w.pairMlp3]) {
    const rectified = relu(execution, hidden);
    const next = linear(execution, {
      x: rectified, weight, rows: pairRows,
      inChannels: pairChannels, outChannels: pairChannels, weightLayout: "in-major",
    });
    execution.release(rectified);
    if (hidden !== pair) execution.release(hidden);
    hidden = next;
    call.onStage?.(`encoder.pairMlp${["1", "2", "3"][mlpIndex]}`, hidden);
    mlpIndex += 1;
  }
  const withResidual = execution.allocate(
    pairRows * pairChannels, "af3.encoder.pair-residual");
  execution.dispatch({
    source: sumSource(pairRows * pairChannels),
    output: withResidual,
    inputs: { aTex: pair, bTex: hidden },
    label: `af3.encoder.pair-residual:${pairRows}x${pairChannels}`,
  });
  execution.release(pair);
  execution.release(hidden);
  pair = withResidual;

  const logits = atomPairLogits(execution, {
    pair, subsets, queries, keys, pairChannels, heads: w.heads,
    blocks: w.blocks.length, weights: w, pairNormPerBlock: call.pairNormPerBlock,
    epsilon,
  });

  // 🔴 intellifold2 AND chai1 PAD THE FLAT ATOM AXIS INSIDE EVERY ATTENTION
  // CALL, "so a block starts from a freshly zeroed padding rather than from
  // whatever the previous block's residual left there" - the same mask this
  // function applies after the stack, applied before each block as well. No
  // default: AF3 pads once.
  if (typeof call.maskAtomActPerBlock !== "boolean") {
    throw new Error(
      "af3 atom encoder: maskAtomActPerBlock has no default - AF3 pads the flat atom"
      + " axis once and intellifold2 and chai1 per block");
  }
  let act = queriesAct;
  for (const [index, block] of w.blocks.entries()) {
    if (call.maskAtomActPerBlock) {
      const padded = execution.allocate(queryRows * channels, "af3.encoder.repadded");
      execution.dispatch({
        source: maskRowsSource(queryRows, channels),
        output: padded,
        inputs: { xTex: act, maskTex: queriesMaskTensor },
        label: `af3.encoder.repad:${queryRows}x${channels}`,
      });
      execution.release(act);
      act = padded;
    }
    const next = crossAttentionBlock(execution, {
      queriesAct: act, queriesToKeys: gathers.queriesToKeys.indices,
      queriesMask: queriesMaskTensor, keysMask: keysMaskTensor,
      queriesCond, keysCond, pairLogits: logits[index],
      subsets, queries, keys, channels, heads: w.heads, dimension: w.dimension,
      weights: block,
      keyMaskedAtomAttention: call.keyMaskedAtomAttention,
      chainedAtomLayerNorm: call.chainedAtomLayerNorm,
      diffusionNoResidual: call.diffusionNoResidual,
      epsilon,
    });
    execution.release(act);
    act = next;
  }
  for (const tensor of logits) execution.release(tensor);

  const masked = execution.allocate(queryRows * channels, "af3.encoder.stack-out");
  execution.dispatch({
    source: maskRowsSource(queryRows, channels),
    output: masked,
    inputs: { xTex: act, maskTex: queriesMaskTensor },
    label: `af3.encoder.stack-mask:${queryRows}x${channels}`,
  });
  execution.release(act);
  call.onStage?.("encoder.stackOut", masked);

  const perToken = w.perTokenChannels;
  const projected = linear(execution, {
    x: masked, weight: w.projectAtomFeaturesForAggr, rows: queryRows,
    inChannels: channels, outChannels: perToken, weightLayout: "in-major",
  });
  call.onStage?.("encoder.projectForAggr", projected);
  const tokenAtoms = gather(execution, {
    source: projected, indices: gathers.queriesToTokenAtoms.indices,
    mask: gathers.queriesToTokenAtoms.mask, count: tokens * dense, channels: perToken,
  });
  execution.release(projected);
  const tokenAct = execution.allocate(tokens * perToken, "af3.encoder.token-act");
  execution.dispatch({
    source: aggregateSource(tokens, dense, perToken),
    output: tokenAct,
    inputs: { projectedTex: tokenAtoms, maskTex: atomMask },
    label: `af3.encoder.aggregate:${tokens}x${dense}x${perToken}`,
  });
  execution.release(tokenAtoms);

  return {
    tokenAct,
    skipConnection: masked,
    queriesMask: queriesMaskTensor,
    keysMask: keysMaskTensor,
    queriesCond,
    keysCond,
    pairCond: pair,
  };
}

/** A one-channel gather; `gather` needs a multiple of four. */
export const scalarGatherSource = (count) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${count};
uniform sampler2D sourceTex;
uniform sampler2D indicesTex;
uniform sampler2D maskTex;

float at(int row) {
  if (fetch1(maskTex, row) == 0.0) return 0.0;
  return fetch1(sourceTex, int(fetch1(indicesTex, row)));
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`;

/**
 * The same, with the index CLAMPED to the source - "a gather whose indices are
 * written for another array", which is what stock AF3's key uid is.
 */
export const clampedGatherSource = (count, sourceLength) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${count};
const int SOURCE = ${sourceLength};
uniform sampler2D sourceTex;
uniform sampler2D indicesTex;
uniform sampler2D maskTex;

float at(int row) {
  if (fetch1(maskTex, row) == 0.0) return 0.0;
  return fetch1(sourceTex, min(int(fetch1(indicesTex, row)), SOURCE - 1));
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`;

/**
 * A three-channel gather, written four to a texel so the reads stay aligned.
 *
 * 🔴 THE SOURCE'S STRIDE IS A PARAMETER, AND IT HAS TO BE. The first of these
 * reads a plain `[rows][3]` array and the second reads the FIRST ONE'S OUTPUT,
 * which is `[rows][4]` - so one kernel with a hard-coded stride of 3 reads the
 * second source three quarters of the way along. That cost two rounds here: the
 * pair's own read at stride 3 (relRMS 0.716) and then this one (0.450), each
 * found by the stage taps while every neighbouring stage stayed exact.
 */
export const vec3GatherSource = (count, sourceStride) => `${PRELUDE}${ONE_TARGET}
const int STRIDE = ${sourceStride};
uniform sampler2D sourceTex;
uniform sampler2D indicesTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel >= ${count}) { result = vec4(0.0); return; }
  if (fetch1(maskTex, texel) == 0.0) { result = vec4(0.0); return; }
  int from = int(fetch1(indicesTex, texel)) * STRIDE;
  result = vec4(fetch1(sourceTex, from), fetch1(sourceTex, from + 1),
    fetch1(sourceTex, from + 2), 0.0);
}`;

/** `act[row][c] += Linear(position)[row][c] * mask[row]`. */
export const positionToFeaturesSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${rows * channels};
uniform sampler2D positionsTex;
uniform sampler2D weightTex;
uniform sampler2D maskTex;
uniform sampler2D actTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  vec4 p = fetch4(positionsTex, row);
  vec4 projected = p.x * fetch4(weightTex, group)
    + p.y * fetch4(weightTex, C4 + group)
    + p.z * fetch4(weightTex, 2 * C4 + group);
  result = fetch4(actTex, texel) + projected * fetch1(maskTex, row);
}`;

/** `out[i] = x[i]` - a copy, because a fragment cannot read its own target. */
export const copySource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(xTex, texel);
}`;
