/**
 * AF3's atom cross-attention encoder: atoms in, 384 token channels out.
 *
 * This is the last piece of the trunk, and the only one that is not a stack of
 * token operations. It runs a three-block transformer over ATOMS, in windows,
 * and then averages each token's atoms into a single vector - which becomes 384
 * of `target_feat`'s 447 columns and so reaches everything downstream.
 *
 * THE LAYOUTS, and why there are three of them. 12 tokens x 24 dense atom slots
 * is 288 atoms, which AF3 reshapes into 9 SUBSETS of 32 QUERIES; each subset
 * attends over 128 KEYS drawn from its own neighbourhood. So an atom attends to
 * a window of about 128 atoms rather than to all of them, which is what keeps
 * the cost linear in a ligand's size. The three gathers between those layouts
 * are precomputed in the batch, not derived here.
 *
 * 🔴 THE LAYERNORMS HERE ARE NOT THE TRUNK'S. Every LayerNorm in the atom stack
 * sets use_fast_variance=False, so the variance is the two-pass mean((x-u)^2)
 * and not the trunk's E[x^2]-E[x]^2. The two are algebraically equal, so
 * borrowing the trunk's implementation runs, agrees to about six digits, and is
 * wrong in the last ones - which is exactly the size of error that gets blamed
 * on quantisation for a week.
 *
 * 🔴 THE CONDITIONING IS ADAPTIVE, NOT ADDITIVE. Each block normalises its
 * activation WITHOUT a learned scale or offset and then takes both from the
 * per-atom conditioning: `sigmoid(scale(cond)) * x + bias(cond)`. There is a
 * second gate at the end of each branch, `sigmoid(zero_cond(cond))`, whose bias
 * initialises to -2 so the branch starts at about a tenth of its weight. A
 * reading that treats either gate as a plain residual add produces finite,
 * plausible, wrong atoms.
 */
import { linear } from "./pairformer-reference.js";

const sigmoid = (value) => 1 / (1 + Math.exp(-value));
const swish = (value) => value * sigmoid(value);
const EPSILON = 1e-5;

/**
 * LayerNorm with AF3's SLOW variance, and optional affine parts.
 *
 * See the note at the top: the atom stack uses this everywhere and the trunk
 * uses the fast form everywhere, and they are not interchangeable.
 */
export function layerNormSlow(input, rows, channels, scale, offset) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    const base = row * channels;
    let sum = 0;
    for (let c = 0; c < channels; c += 1) sum += input[base + c];
    const mean = sum / channels;
    let variance = 0;
    for (let c = 0; c < channels; c += 1) {
      const centred = input[base + c] - mean;
      variance += centred * centred;
    }
    const inverse = 1 / Math.sqrt(variance / channels + EPSILON);
    for (let c = 0; c < channels; c += 1) {
      const normalised = (input[base + c] - mean) * inverse;
      output[base + c] = normalised * (scale === null ? 1 : scale[c])
        + (offset === null ? 0 : offset[c]);
    }
  }
  return output;
}

/**
 * One of AF3's atom-layout gathers.
 *
 * A gather flattens the source over its own layout axes and indexes it; where
 * the mask is false the destination is ZERO rather than whatever index 0 holds,
 * which matters because padded slots index 0 legitimately.
 *
 * @param {{indices: ArrayLike<number>, mask: ArrayLike<number>, count: number}} gather
 * @param {Float32Array} source
 * @param {number} channels  trailing channels carried through unchanged
 */
export function convert(gather, source, channels) {
  const output = new Float32Array(gather.count * channels);
  for (let index = 0; index < gather.count; index += 1) {
    if (!gather.mask[index]) continue;
    const from = gather.indices[index] * channels;
    const to = index * channels;
    for (let c = 0; c < channels; c += 1) output[to + c] = source[from + c];
  }
  return output;
}

/** Adaptive LayerNorm: normalise, then scale and shift from the conditioning. */
export function adaptiveLayerNorm(x, cond, rows, channels, weights, prefix,
                                  condChannels = channels) {
  // 🔴 THE CONDITIONING IS NOT THE SAME WIDTH AS THE ACTIVATION. In the atom
  // stack both are 128 and the distinction is invisible; in the diffusion
  // transformer the conditioning is 384 and the activation 768, so the two
  // projections are 384->768 rather than square. Assuming square runs on the
  // atom stack and reads the diffusion weights at the wrong stride.
  const normalised = layerNormSlow(x, rows, channels, null, null);
  // ...the conditioning gets a scale but NO offset before it is projected.
  const condNormalised = layerNormSlow(cond, rows, condChannels,
                                       weights[`${prefix}SingleCondLayerNormScale`], null);
  const scale = linear(condNormalised, rows, condChannels, channels,
                       weights[`${prefix}SingleCondScaleWeights`],
                       weights[`${prefix}SingleCondScaleBias`]);
  const shift = linear(condNormalised, rows, condChannels, channels,
                       weights[`${prefix}SingleCondBias`]);
  const output = new Float32Array(x.length);
  for (let index = 0; index < x.length; index += 1) {
    output[index] = sigmoid(scale[index]) * normalised[index] + shift[index];
  }
  return output;
}

/** The AdaLN-zero output gate: project, then gate on the conditioning. */
export function adaptiveZeroInit(x, cond, rows, channels, weights, prefix,
                                 condChannels = channels, inChannels = channels) {
  const projected = linear(x, rows, inChannels, channels,
                           weights[`${prefix}Transition2`]);
  const gate = linear(cond, rows, condChannels, channels,
                      weights[`${prefix}AdaptiveZeroCondWeights`],
                      weights[`${prefix}AdaptiveZeroCondBias`]);
  for (let index = 0; index < projected.length; index += 1) {
    projected[index] *= sigmoid(gate[index]);
  }
  return projected;
}

/**
 * One block of the atom cross-attention transformer.
 *
 * @param {object} shape {subsets, queries, keys, channels, heads, dimension}
 */
/** The one dialect flag a block's attention reads, required rather than assumed. */
function dialectOf(weights) {
  if (weights.keyMaskedAtomAttention === undefined) {
    throw new Error("weights.keyMaskedAtomAttention has no default: AF3 masks a "
      + "pair only when query AND key are padded, OpenDDE when either is");
  }
  return weights;
}

export function crossAttentionBlock(queriesAct, state, shape, weights) {
  const { subsets, queries, keys, channels, heads, dimension } = shape;
  const { queriesToKeys, queriesMask, keysMask, queriesCond, keysCond, pairLogits } = state;
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;
  const scale = 1 / Math.sqrt(dimension);

  // 🔴 THE TWO ADAPTIVE LAYERNORMS ARE PARALLEL OR CHAINED, AND THAT IS THE
  // MODEL. AlphaFold 3 normalises the RAW activation twice, once per side.
  // OpenDDE reassigns: its `AttentionPairBias` in cross-attention mode runs
  //
  //     a  = layernorm_a(a, s)     // the queries
  //     kv = layernorm_kv(a, s)    // <- the ALREADY-NORMALISED a
  //
  // so the gather onto the key layout happens BETWEEN the two, and the second
  // norm re-centres and re-scales a tensor whose statistics the first already
  // fixed - while seeing the first one's learned scale. Two chained LayerNorms
  // are not one, and neither arrangement changes a shape.
  if (weights.chainedAtomLayerNorm === undefined) {
    throw new Error("weights.chainedAtomLayerNorm has no default: AF3 "
      + "normalises the raw activation on both sides, OpenDDE chains them");
  }
  const xq = adaptiveLayerNorm(queriesAct, queriesCond, queryRows, channels, weights, "q");
  const keysAct = convert(queriesToKeys,
                          weights.chainedAtomLayerNorm ? xq : queriesAct, channels);
  const xk = adaptiveLayerNorm(keysAct, keysCond, keyRows, channels, weights, "k");

  const width = heads * dimension;
  const q = linear(xq, queryRows, channels, width, weights.qProjection, weights.qBias);
  const k = linear(xk, keyRows, channels, width, weights.kProjection);
  const v = linear(xk, keyRows, channels, width, weights.vProjection);

  const gathered = new Float32Array(queryRows * width);
  const logits = new Float32Array(keys);
  for (let subset = 0; subset < subsets; subset += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let query = 0; query < queries; query += 1) {
        const queryIndex = subset * queries + query;
        const queryBase = queryIndex * width + head * dimension;
        for (let key = 0; key < keys; key += 1) {
          const keyIndex = subset * keys + key;
          let dot = 0;
          for (let d = 0; d < dimension; d += 1) {
            dot += q[queryBase + d] * k[keyIndex * width + head * dimension + d];
          }
          // 🔴 THE MASK BIAS IS A PRODUCT, NOT A SUM - UNDER AlphaFold 3. It
          // penalises a pair only when the query AND the key are padded, so a
          // real query can still attend to a padded key. OpenDDE, Protenix and
          // RoseTTAFold3 ADD the two instead, which is an OR: a real query
          // cannot attend to a padded key at all. The difference is large in a
          // mostly-empty window - a lone ligand's 16 atoms sit in a 32-query,
          // 128-key window - and it is nothing at all when every key is real,
          // which is every batch THIS featuriser produces (the window is
          // `min(128, atomCount)`; see CLAUDE.md). It is implemented because
          // the differential checkers feed AF3's own gathers out of an oracle
          // dump, where padded keys do occur.
          const maskBias = dialectOf(weights).keyMaskedAtomAttention
            ? -1e9 * ((1 - queriesMask[queryIndex]) + (1 - keysMask[keyIndex]))
            : 1e9 * (queriesMask[queryIndex] - 1) * (keysMask[keyIndex] - 1);
          logits[key] = dot * scale + maskBias
            + pairLogits[((subset * heads + head) * queries + query) * keys + key];
        }
        let largest = -Infinity;
        for (let key = 0; key < keys; key += 1) {
          if (logits[key] > largest) largest = logits[key];
        }
        let total = 0;
        for (let key = 0; key < keys; key += 1) {
          logits[key] = Math.exp(logits[key] - largest);
          total += logits[key];
        }
        for (let d = 0; d < dimension; d += 1) {
          let sum = 0;
          for (let key = 0; key < keys; key += 1) {
            sum += logits[key] * v[(subset * keys + key) * width + head * dimension + d];
          }
          gathered[queryIndex * width + head * dimension + d] = sum / total;
        }
      }
    }
  }

  const gate = linear(xq, queryRows, channels, width, weights.gatingQuery);
  for (let index = 0; index < gathered.length; index += 1) {
    gathered[index] *= sigmoid(gate[index]);
  }
  const attention = adaptiveZeroInit(gathered, queriesCond, queryRows, channels,
                                     weights, "");

  // ...the transition reads the POST-attention activation, threaded rather than
  // parallel. (chai-1 reads the block input for both branches instead.)
  //
  // 🔴 AND rosettafold3 IS PARALLEL TOO: its
  // `no_residual_connection_between_attention_and_transition` hands the
  // transition the PRE-attention activation and adds both through ONE residual.
  // The two are not a reordering - the transition sees a different tensor - and
  // it is worth the atom encoder's whole skip connection.
  if (weights.diffusionNoResidual === undefined) {
    throw new Error("an atom block carries no diffusionNoResidual: AF3 adds the "
      + "attention and the transition through two residuals, rosettafold3 one");
  }
  const afterAttention = new Float32Array(queriesAct.length);
  for (let index = 0; index < queriesAct.length; index += 1) {
    afterAttention[index] = queriesAct[index] + attention[index];
  }

  const transitionInput = weights.diffusionNoResidual ? queriesAct : afterAttention;
  const normalised = adaptiveLayerNorm(transitionInput, queriesCond, queryRows,
                                       channels, weights, "ffw");
  const intermediate = channels * 2;
  const wide = linear(normalised, queryRows, channels, intermediate * 2,
                      weights.ffwTransition1);
  const gated = new Float32Array(queryRows * intermediate);
  for (let row = 0; row < queryRows; row += 1) {
    for (let i = 0; i < intermediate; i += 1) {
      gated[row * intermediate + i] = swish(wide[row * intermediate * 2 + i])
        * wide[row * intermediate * 2 + intermediate + i];
    }
  }
  // 🔴 boltz2's EXTRA UP-GATE. `SwiGLU(a) * a_to_b(a)`, reading the SAME
  // adaptive-LayerNorm output the SwiGLU reads - not the block input. Present
  // only where the bundle carries the weight; see `atomBlock`.
  if (weights.ffwAToB !== undefined && weights.ffwAToB !== null) {
    const upGate = linear(normalised, queryRows, channels, intermediate,
                          weights.ffwAToB);
    for (let index = 0; index < gated.length; index += 1) gated[index] *= upGate[index];
  }
  const projected = linear(gated, queryRows, intermediate, channels,
                           weights.ffwTransition2);
  const transitionGate = linear(queriesCond, queryRows, channels, channels,
                                weights.ffwAdaptiveZeroCondWeights,
                                weights.ffwAdaptiveZeroCondBias);
  const output = new Float32Array(queriesAct.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = queriesAct[index] + attention[index]
      + projected[index] * sigmoid(transitionGate[index]);
  }
  return output;
}

/**
 * The per-block attention biases, for every block of an atom stack at once.
 *
 * 🔴 ONE LAYERNORM AND ONE PROJECTION SERVE THE WHOLE STACK, which is why the
 * projection's output is (blocks, heads) rather than heads. Recomputing it per
 * block would read the same weights and give the same answer; splitting it the
 * wrong way round gives every block the biases meant for another.
 *
 * 🔴 ...UNDER AlphaFold 3. OpenDDE NORMALISES AND PROJECTS PER BLOCK, with its
 * own scale and its own matrix each time - `pair_input_layer_norm/scale` is
 * [3, 16] there and [16] here, and the projection [3, 16, 4] against [16, 12].
 * The MEAN and VARIANCE are the same either way (they come from the pair, which
 * does not change), so the difference is entirely in which scale and which
 * matrix a block reads. That makes it invisible to every shape check and to any
 * test that runs one block.
 */
export function atomPairLogits(pair, shape, weights, onStage) {
  const { subsets, queries, keys, pairChannels, heads, blocks } = shape;
  const pairRows = subsets * queries * keys;
  if (weights.pairNormPerBlock === undefined) {
    throw new Error("weights.pairNormPerBlock has no default: AF3 normalises "
      + "the atom-pair conditioning once for the stack, OpenDDE once per block");
  }
  // Per block: its own scale, its own matrix, `heads` outputs. Shared: one
  // scale, one matrix, `blocks * heads` outputs read at a block's offset. The
  // loop below indexes `flat` identically in both cases, which is what keeps
  // the two arms one function.
  let flat;
  if (weights.pairNormPerBlock) {
    flat = new Float32Array(pairRows * blocks * heads);
    for (let block = 0; block < blocks; block += 1) {
      const normalised = layerNormSlow(pair, pairRows, pairChannels,
                                       weights.pairInputLayerNormScales[block], null);
      const projected = linear(normalised, pairRows, pairChannels, heads,
                               weights.pairLogitsProjections[block]);
      for (let row = 0; row < pairRows; row += 1) {
        for (let head = 0; head < heads; head += 1) {
          flat[row * blocks * heads + block * heads + head] = projected[row * heads + head];
        }
      }
    }
  } else {
    const normalised = layerNormSlow(pair, pairRows, pairChannels,
                                     weights.pairInputLayerNormScale, null);
    flat = linear(normalised, pairRows, pairChannels, blocks * heads,
                  weights.pairLogitsProjection);
  }
  // In [subsets, queries, keys, blocks, heads] order, which is the reference's
  // own `pair_logits_projection` layout - so a checker can compare it without
  // a transpose it could get wrong.
  onStage?.("encoder.pairLogits", flat);
  const output = [];
  for (let block = 0; block < blocks; block += 1) {
    const perBlock = new Float32Array(subsets * heads * queries * keys);
    for (let subset = 0; subset < subsets; subset += 1) {
      for (let query = 0; query < queries; query += 1) {
        for (let key = 0; key < keys; key += 1) {
          const source = (((subset * queries + query) * keys) + key) * blocks * heads;
          for (let head = 0; head < heads; head += 1) {
            perBlock[((subset * heads + head) * queries + query) * keys + key] =
              flat[source + block * heads + head];
          }
        }
      }
    }
    output.push(perBlock);
  }
  return output;
}

/**
 * The whole encoder: per-atom conditioning in, per-token features out.
 *
 * @returns {Float32Array} tokens * perTokenChannels
 */
export function atomCrossAttentionEncoder(input, weights, onStage) {
  const { tokens, dense, subsets, queries, keys } = input.shape;
  // 🔴 THE DIALECT RIDES IN THE INPUT, because this encoder is INJECTED into
  // the diffusion head as a bare function value (see diffusion-reference.js)
  // and a fourth positional argument would not survive that. No default: see
  // `maskPaddedKeys` at the offset validity below for what silently changes.
  const dialect = input.dialect;
  if (dialect?.maskPaddedKeys === undefined) {
    throw new Error("input.dialect.maskPaddedKeys has no default: stock AF3 is "
      + "false, the openfold3 lineage true");
  }
  const channels = weights.channels;
  const pairChannels = weights.pairChannels;
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;

  const queriesCond = convert(input.tokenAtomsToQueries, input.conditioning, channels);
  const queriesMask = convert(input.tokenAtomsToQueries, input.atomMask, 1);

  // 🔴 THE TRUNK'S SINGLE CONDITIONING IS BROADCAST PER TOKEN, NOT PER ATOM,
  // and it goes in BEFORE the keys are gathered - so every key sees it too.
  // Only the diffusion head passes it; the trunk's own encoder has no trunk to
  // condition on yet.
  // 🔴 THE QUERY ACTIVATION AND THE CONDITIONING ARE ONE ARRAY UNDER AF3 AND
  // TWO UNDER boltz2. Snapshotted here, before the trunk single goes in: under
  // `preTrunkQuery` the queries read the per-atom features alone while every
  // adaptive LayerNorm below still reads the full conditioning.
  const preTrunkQuery = input.dialect?.preTrunkQuery === true;
  const queriesBeforeTrunk = preTrunkQuery ? Float32Array.from(queriesCond) : null;
  if (input.trunkSingleCond !== undefined) {
    const projected = linear(
      layerNormSlow(input.trunkSingleCond, tokens, weights.trunkSingleChannels,
                    weights.lnormTrunkSingleCondScale,
                    weights.lnormTrunkSingleCondOffset ?? null),
      tokens, weights.trunkSingleChannels, channels, weights.embedTrunkSingleCond);
    const perQuery = convert(input.tokensToQueries, projected, channels);
    for (let index = 0; index < queriesCond.length; index += 1) {
      queriesCond[index] += perQuery[index];
    }
  }

  // 🔴 MASKED BEFORE THE KEYS ARE GATHERED FROM IT, NOT AFTER. This is a no-op
  // for the trunk's encoder - _per_atom_conditioning already masked its output,
  // so the two orders agree - and it is NOT a no-op once the trunk conditioning
  // is added above, because that puts non-zero values into padded atom slots.
  // Gathering first carries them into the keys, where two thirds of the slots
  // are padding: measured 8.4e-2 on the encoder's output, which reads like a
  // subtly wrong kernel rather than a line in the wrong order.
  for (let row = 0; row < queryRows; row += 1) {
    for (let c = 0; c < channels; c += 1) queriesCond[row * channels + c] *= queriesMask[row];
  }
  const keysCond = convert(input.queriesToKeys, queriesCond, channels);
  const keysMask = convert(input.queriesToKeys, queriesMask, 1);

  // ...the query starts as the conditioning, and the diffusion head then adds
  // the NOISY POSITIONS to it. The trunk's encoder has no positions to add, so
  // its query is the conditioning alone.
  const queriesAct = Float32Array.from(queriesBeforeTrunk ?? queriesCond);
  if (queriesBeforeTrunk !== null) {
    // ...masked the same way the conditioning was, for the same reason.
    for (let row = 0; row < queryRows; row += 1) {
      for (let c = 0; c < channels; c += 1) {
        queriesAct[row * channels + c] *= queriesMask[row];
      }
    }
  }
  if (input.tokenAtomsAct !== undefined) {
    const gatheredPositions = convert(input.tokenAtomsToQueries, input.tokenAtomsAct, 3);
    const positional = linear(gatheredPositions, queryRows, 3, channels,
                              weights.atomPositionsToFeatures);
    for (let row = 0; row < queryRows; row += 1) {
      for (let c = 0; c < channels; c += 1) {
        queriesAct[row * channels + c] += positional[row * channels + c] * queriesMask[row];
      }
    }
  }

  const rectifiedQueries = Float32Array.from(queriesCond, (v) => (v > 0 ? v : 0));
  const rectifiedKeys = Float32Array.from(keysCond, (v) => (v > 0 ? v : 0));
  const row = linear(rectifiedQueries, queryRows, channels, pairChannels,
                     weights.singleToPairCondRow);
  const column = linear(rectifiedKeys, keyRows, channels, pairChannels,
                        weights.singleToPairCondCol);
  onStage?.("encoder.queriesCond", queriesCond);
  onStage?.("encoder.pairCondRow", row);
  onStage?.("encoder.pairCondCol", column);

  // ...the trunk pair conditioning, projected once and then gathered per atom
  // pair below.
  let trunkPair = null;
  let tokensToKeys = null;
  let keysTokenMask = null;
  if (input.trunkPairCond !== undefined) {
    trunkPair = linear(
      layerNormSlow(input.trunkPairCond, tokens * tokens, weights.trunkPairChannels,
                    weights.lnormTrunkPairCondScale,
                    weights.lnormTrunkPairCondOffset ?? null),
      tokens * tokens, weights.trunkPairChannels, pairChannels,
      weights.embedTrunkPairCond);
    onStage?.("encoder.trunkPair", trunkPair);
    // 🔴 tokens_to_keys IS IN THE BATCH; DO NOT DERIVE IT. Carrying
    // tokens_to_queries through the queries-to-keys gather looks equivalent and
    // is a second source of truth for something the featuriser already
    // computed - and its MASK is not the same, because a derived one folds in
    // the query mask where AF3's is the key's own.
    tokensToKeys = input.tokensToKeys.indices;
    keysTokenMask = input.tokensToKeys.mask;
  }

  const queriesRefPos = convert(input.tokenAtomsToQueries, input.refPos, 3);
  const queriesSpaceUid = convert(input.tokenAtomsToQueries, input.refSpaceUid, 1);
  const keysRefPos = convert(input.queriesToKeys, queriesRefPos, 3);
  const keysSpaceUid = convert(input.queriesToKeys, queriesSpaceUid, 1);

  const pair = new Float32Array(subsets * queries * keys * pairChannels);
  // 🔴 THE THREE PAIR TERMS UNMASKED, because the reference records each Linear's
  // output BEFORE multiplying it by `offsets_valid` - so a checker comparing the
  // masked sum cannot say which of the three moved. Built only when a caller
  // asks; `pairTerms` is null otherwise and the loop below skips the stores.
  const pairTerms = onStage === undefined ? null : {
    offsets: new Float32Array(subsets * queries * keys * pairChannels),
    distances: new Float32Array(subsets * queries * keys * pairChannels),
    valid: new Float32Array(subsets * queries * keys * pairChannels),
    validFlag: new Float32Array(subsets * queries * keys),
  };
  const offsets = new Float32Array(3);
  for (let subset = 0; subset < subsets; subset += 1) {
    for (let query = 0; query < queries; query += 1) {
      const queryIndex = subset * queries + query;
      for (let key = 0; key < keys; key += 1) {
        const keyIndex = subset * keys + key;
        const base = (queryIndex * keys + key) * pairChannels;
        for (let c = 0; c < pairChannels; c += 1) {
          pair[base + c] = row[queryIndex * pairChannels + c]
            + column[keyIndex * pairChannels + c];
        }
        // 🔴 VALIDITY IS "SAME REFERENCE SPACE", not "both atoms real". Two
        // atoms only have a meaningful offset if they came from the same
        // reference conformer; across residues the offset is arbitrary.
        //
        // 🔴 ...WHICH LETS PADDED KEYS IN, AND ONE DIALECT SAYS SO. A padded key
        // slot gathers a zero reference space, and the FIRST reference
        // conformer's uid is also zero - so every padded key reads as a valid
        // neighbour of token 0, and the N-terminus is conditioned on a hundred
        // atoms that do not exist. AF3 was released this way and OpenFold3 was
        // trained with the mask, so this is a property of the WEIGHTS rather
        // than a bug either side is free to fix: upstream measures the released
        // graph putting CA-C at ~0.85 A against an ideal 1.52. See
        // ../alphafold3 `atom_cross_attention.py`, gated on OPENFOLD3_LINEAGE.
        const valid = queriesSpaceUid[queryIndex] === keysSpaceUid[keyIndex]
          && (!dialect.maskPaddedKeys || keysMask[keyIndex] !== 0) ? 1 : 0;
        let squared = 0;
        for (let axis = 0; axis < 3; axis += 1) {
          const difference = queriesRefPos[queryIndex * 3 + axis]
            - keysRefPos[keyIndex * 3 + axis];
          offsets[axis] = difference;
          squared += difference * difference;
        }
        for (let c = 0; c < pairChannels; c += 1) {
          let offsetTerm = 0;
          for (let axis = 0; axis < 3; axis += 1) {
            offsetTerm += offsets[axis] * weights.embedPairOffsets[axis * pairChannels + c];
          }
          if (pairTerms !== null) {
            pairTerms.validFlag[queryIndex * keys + key] = valid;
            pairTerms.offsets[base + c] = offsetTerm;
            pairTerms.distances[base + c] = weights.embedPairDistances[c] / (1 + squared);
            pairTerms.valid[base + c] = weights.embedPairOffsetsValid[c];
          }
          pair[base + c] += valid * (offsetTerm
            + weights.embedPairDistances[c] / (1 + squared))
            // ...and the validity flag itself, which is NOT gated by validity:
            // "these two atoms are unrelated" is information the model uses.
            + valid * weights.embedPairOffsetsValid[c];
        }
        // 🔴 THE TRUNK'S PAIR REPRESENTATION, INDEXED BY THE TWO ATOMS' TOKENS.
        // Only the diffusion head supplies it; the trunk's own encoder runs
        // before there is a pair to condition on. Both ends must be real, so
        // the mask is the AND of the query's token and the key's.
        if (trunkPair !== null
            && input.tokensToQueries.mask[queryIndex] && keysTokenMask[keyIndex]) {
          const from = (input.tokensToQueries.indices[queryIndex] * tokens
            + tokensToKeys[keyIndex]) * pairChannels;
          for (let c = 0; c < pairChannels; c += 1) pair[base + c] += trunkPair[from + c];
        }
      }
    }
  }

  if (pairTerms !== null) {
    onStage?.("encoder.pairValid", pairTerms.validFlag);
    onStage?.("encoder.queriesMask", queriesMask);
    onStage?.("encoder.keysMask", keysMask);
    onStage?.("encoder.queriesUid", Float32Array.from(queriesSpaceUid));
    onStage?.("encoder.keysUid", Float32Array.from(keysSpaceUid));
    onStage?.("encoder.embedPairOffsets", pairTerms.offsets);
    onStage?.("encoder.embedPairDistances", pairTerms.distances);
    onStage?.("encoder.embedPairOffsetsValid", pairTerms.valid);
    onStage?.("encoder.pairBeforeMlp", pair);
  }
  const pairRows = subsets * queries * keys;
  const relu = (values) => Float32Array.from(values, (v) => (v > 0 ? v : 0));
  let hidden = linear(relu(pair), pairRows, pairChannels, pairChannels, weights.pairMlp1);
  onStage?.("encoder.pairMlp1", hidden);
  hidden = linear(relu(hidden), pairRows, pairChannels, pairChannels, weights.pairMlp2);
  onStage?.("encoder.pairMlp2", hidden);
  const residual = linear(relu(hidden), pairRows, pairChannels, pairChannels,
                          weights.pairMlp3);
  onStage?.("encoder.pairMlp3", residual);
  for (let index = 0; index < pair.length; index += 1) pair[index] += residual[index];

  const heads = weights.heads;
  const pairLogits = atomPairLogits(pair, { subsets, queries, keys, pairChannels,
                                            heads, blocks: weights.blocks.length },
                                    weights, onStage);

  const shape = { subsets, queries, keys, channels, heads,
                  dimension: weights.dimension };
  let act = queriesAct;
  // 🔴 intellifold2 AND chai1 PAD THE FLAT ATOM AXIS INSIDE EVERY ATTENTION
  // CALL, so a block starts from a freshly zeroed padding rather than from
  // whatever the previous block's residual left there. See the flag's note in
  // src/af3/dialect.js; the mask below is the same one this function already
  // applies AFTER the stack.
  const maskPerBlock = weights.blocks[0]?.maskAtomActPerBlock;
  if (maskPerBlock === undefined) {
    throw new Error("weights.blocks[].maskAtomActPerBlock has no default: AF3 "
      + "pads the flat atom axis once, intellifold2 and chai1 per block");
  }
  const zeroPadding = (rows) => {
    for (let index = 0; index < queryRows; index += 1) {
      if (queriesMask[index] !== 0) continue;
      for (let c = 0; c < channels; c += 1) rows[index * channels + c] = 0;
    }
    return rows;
  };
  for (let block = 0; block < weights.blocks.length; block += 1) {
    if (maskPerBlock) act = zeroPadding(Float32Array.from(act));
    act = crossAttentionBlock(act, {
      queriesToKeys: input.queriesToKeys, queriesMask, keysMask,
      queriesCond, keysCond, pairLogits: pairLogits[block],
    }, shape, weights.blocks[block]);
  }
  for (let index = 0; index < queryRows; index += 1) {
    for (let c = 0; c < channels; c += 1) act[index * channels + c] *= queriesMask[index];
  }
  onStage?.("encoder.stackOut", act);
  const skipConnection = Float32Array.from(act);

  // ...back to token-atom layout, rectified, and averaged over each token's
  // REAL atoms only.
  const perToken = weights.perTokenChannels;
  const projected = linear(act, queryRows, channels, perToken,
                           weights.projectAtomFeaturesForAggr);
  onStage?.("encoder.projectForAggr", projected);
  const tokenAtoms = convert(input.queriesToTokenAtoms, projected, perToken);
  const output = new Float32Array(tokens * perToken);
  for (let token = 0; token < tokens; token += 1) {
    let count = 0;
    for (let atom = 0; atom < dense; atom += 1) {
      count += input.atomMask[token * dense + atom];
    }
    for (let atom = 0; atom < dense; atom += 1) {
      if (!input.atomMask[token * dense + atom]) continue;
      const base = (token * dense + atom) * perToken;
      for (let c = 0; c < perToken; c += 1) {
        const value = tokenAtoms[base + c];
        output[token * perToken + c] += value > 0 ? value : 0;
      }
    }
    if (count > 0) {
      for (let c = 0; c < perToken; c += 1) output[token * perToken + c] /= count;
    }
  }
  // The decoder needs everything the encoder computed, not just its output.
  return { tokenAct: output, skipConnection, queriesMask, keysMask,
           queriesCond, keysCond, pairCond: pair };
}

/** boltz2's mol_type class: protein 0, DNA 1, RNA 2, ligand 3. */
function molTypeOf(input, token) {
  if (input.isLigand?.[token]) return 3;
  if (input.isRna?.[token]) return 2;
  if (input.isDna?.[token]) return 1;
  return 0;
}

/**
 * `target_feat`: the 447 columns everything in the trunk is built from.
 *
 * 🔴 THE ORDER IS AF3'S, AND OPENFOLD3 STORES IT DIFFERENTLY. AF3 lays the
 * columns out as [restype 31 | profile 31 | deletion mean 1 | atoms 384];
 * OpenFold3's checkpoint has the atom block FIRST and 32 restypes rather than
 * 31, and its converter permutes both to land here. Anything reading a
 * checkpoint directly rather than through that converter has to do the same.
 *
 * @param {{aatype: ArrayLike<number>, profile: ArrayLike<number>,
 *          deletionMean: ArrayLike<number>, atomFeatures: Float32Array}} input
 * @param {number} tokens
 */
export function targetFeatures(input, tokens, dialect) {
  // 🔴 boltz2's IS THE ATOM ENCODER'S COLUMNS ALONE. Everything else here
  // prepends a restype one-hot, a profile and a deletion mean - 31 + 31 + 1 -
  // for 447; boltz2 takes the 384 and concatenates s_trunk with THAT. Built
  // AF3's way it is 447 wide and the fold reads "targetFeat has 30396 elements;
  // expected 26112".
  const restypes = 31;
  if (dialect?.targetFeatAtomOnly === true) {
    const output = Float32Array.from(input.atomFeatures.subarray(0, tokens * 384));
    const sum = input.sum;
    if (sum == null) return output;
    // 🔴 SIX MORE TERMS, ADDED. boltz2's InputEmbedder is a SUM at
    // seq_channel, not a concatenation: the atom encoder's token activation
    // plus six bias-free projections. Four of them are constant on an ordinary
    // monomer and are trained NON-ZERO - the method row alone is a learned
    // 384-vector added to every token - so leaving them out is not "omitting a
    // feature nothing uses". Measured: target_feat 1.00e+0 against
    // af3-any-model without them.
    const channels = 384;
    const add = (weights, width, value) => {
      for (let token = 0; token < tokens; token += 1) {
        for (let c = 0; c < channels; c += 1) {
          let total = 0;
          for (let k = 0; k < width; k += 1) {
            const v = value(token, k);
            if (v !== 0) total += v * weights[k * channels + c];
          }
          output[token * channels + c] += total;
        }
      }
    };
    // restype one-hot, 31 classes
    add(sum.resType, restypes, (token, k) => (input.aatype[token] === k ? 1 : 0));
    // [profile (31) | deletion mean (1)]
    add(sum.msaProfile, restypes + 1, (token, k) => (k < restypes
      ? input.profile[token * restypes + k] : input.deletionMean[token]));
    // mol_type: protein 0, DNA 1, RNA 2, ligand 3
    add(sum.molType, 4, (token, k) => (molTypeOf(input, token) === k ? 1 : 0));
    // the cyclic flag: PRESENT or absent, not the period. Zero on every
    // ordinary input, so the trained weight only ever acts on a cyclic one.
    add(sum.cyclic, 1, (token) => Math.min(1, Math.max(0, input.cyclicPeriod?.[token] ?? 0)));
    // `method` is the experimental method the prediction asks for, and the
    // default is x-ray diffraction: class 1 of 12, the same for every token.
    add(sum.method, 12, (_token, k) => (k === 1 ? 1 : 0));
    // `modified` marks the tokens of a modified residue; row 0 is a learned
    // non-zero vector, so this term is present even when nothing is modified.
    add(sum.modified, 2, (token, k) => ((input.isModified?.[token] ? 1 : 0) === k ? 1 : 0));
    return output;
  }
  const width = restypes * 2 + 1 + 384;
  const output = new Float32Array(tokens * width);
  for (let token = 0; token < tokens; token += 1) {
    const base = token * width;
    const code = input.aatype[token];
    if (code >= 0 && code < restypes) output[base + code] = 1;
    for (let c = 0; c < restypes; c += 1) {
      output[base + restypes + c] = input.profile[token * restypes + c];
    }
    output[base + restypes * 2] = input.deletionMean[token];
    for (let c = 0; c < 384; c += 1) {
      output[base + restypes * 2 + 1 + c] = input.atomFeatures[token * 384 + c];
    }
  }
  return output;
}
