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
function adaptiveLayerNorm(x, cond, rows, channels, weights, prefix) {
  const normalised = layerNormSlow(x, rows, channels, null, null);
  // ...the conditioning gets a scale but NO offset before it is projected.
  const condNormalised = layerNormSlow(cond, rows, channels,
                                       weights[`${prefix}SingleCondLayerNormScale`], null);
  const scale = linear(condNormalised, rows, channels, channels,
                       weights[`${prefix}SingleCondScaleWeights`],
                       weights[`${prefix}SingleCondScaleBias`]);
  const shift = linear(condNormalised, rows, channels, channels,
                       weights[`${prefix}SingleCondBias`]);
  const output = new Float32Array(x.length);
  for (let index = 0; index < x.length; index += 1) {
    output[index] = sigmoid(scale[index]) * normalised[index] + shift[index];
  }
  return output;
}

/** The AdaLN-zero output gate: project, then gate on the conditioning. */
function adaptiveZeroInit(x, cond, rows, channels, weights, prefix) {
  const projected = linear(x, rows, channels, channels,
                           weights[`${prefix}Transition2`]);
  const gate = linear(cond, rows, channels, channels,
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
function crossAttentionBlock(queriesAct, state, shape, weights) {
  const { subsets, queries, keys, channels, heads, dimension } = shape;
  const { queriesToKeys, queriesMask, keysMask, queriesCond, keysCond, pairLogits } = state;
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;
  const scale = 1 / Math.sqrt(dimension);

  const keysAct = convert(queriesToKeys, queriesAct, channels);
  const xq = adaptiveLayerNorm(queriesAct, queriesCond, queryRows, channels, weights, "q");
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
          // 🔴 THE MASK BIAS IS A PRODUCT, NOT A SUM. AF3 penalises a pair only
          // when the query AND the key are padded, so a real query can still
          // attend to a padded key. (RoseTTAFold3 adds them instead, which is
          // an OR; the difference is large in a mostly-empty window.)
          const maskBias = 1e9 * (queriesMask[queryIndex] - 1)
            * (keysMask[keyIndex] - 1);
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
  const afterAttention = new Float32Array(queriesAct.length);
  for (let index = 0; index < queriesAct.length; index += 1) {
    afterAttention[index] = queriesAct[index] + attention[index];
  }

  const normalised = adaptiveLayerNorm(afterAttention, queriesCond, queryRows,
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
 * The whole encoder: per-atom conditioning in, per-token features out.
 *
 * @returns {Float32Array} tokens * perTokenChannels
 */
export function atomCrossAttentionEncoder(input, weights) {
  const { tokens, dense, subsets, queries, keys } = input.shape;
  const channels = weights.channels;
  const pairChannels = weights.pairChannels;
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;

  const queriesCond = convert(input.tokenAtomsToQueries, input.conditioning, channels);
  const queriesMask = convert(input.tokenAtomsToQueries, input.atomMask, 1);
  const keysCond = convert(input.queriesToKeys, queriesCond, channels);
  const keysMask = convert(input.queriesToKeys, queriesMask, 1);
  for (let row = 0; row < queryRows; row += 1) {
    for (let c = 0; c < channels; c += 1) queriesCond[row * channels + c] *= queriesMask[row];
  }
  const queriesAct = Float32Array.from(queriesCond);

  const rectifiedQueries = Float32Array.from(queriesCond, (v) => (v > 0 ? v : 0));
  const rectifiedKeys = Float32Array.from(keysCond, (v) => (v > 0 ? v : 0));
  const row = linear(rectifiedQueries, queryRows, channels, pairChannels,
                     weights.singleToPairCondRow);
  const column = linear(rectifiedKeys, keyRows, channels, pairChannels,
                        weights.singleToPairCondCol);

  const queriesRefPos = convert(input.tokenAtomsToQueries, input.refPos, 3);
  const queriesSpaceUid = convert(input.tokenAtomsToQueries, input.refSpaceUid, 1);
  const keysRefPos = convert(input.queriesToKeys, queriesRefPos, 3);
  const keysSpaceUid = convert(input.queriesToKeys, queriesSpaceUid, 1);

  const pair = new Float32Array(subsets * queries * keys * pairChannels);
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
        const valid = queriesSpaceUid[queryIndex] === keysSpaceUid[keyIndex] ? 1 : 0;
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
          pair[base + c] += valid * (offsetTerm
            + weights.embedPairDistances[c] / (1 + squared))
            // ...and the validity flag itself, which is NOT gated by validity:
            // "these two atoms are unrelated" is information the model uses.
            + valid * weights.embedPairOffsetsValid[c];
        }
      }
    }
  }

  const pairRows = subsets * queries * keys;
  const relu = (values) => Float32Array.from(values, (v) => (v > 0 ? v : 0));
  let hidden = linear(relu(pair), pairRows, pairChannels, pairChannels, weights.pairMlp1);
  hidden = linear(relu(hidden), pairRows, pairChannels, pairChannels, weights.pairMlp2);
  const residual = linear(relu(hidden), pairRows, pairChannels, pairChannels,
                          weights.pairMlp3);
  for (let index = 0; index < pair.length; index += 1) pair[index] += residual[index];

  // ...the pair logits for every block at once, which is why the projection's
  // output is (blocks, heads) rather than heads.
  const blocks = weights.blocks.length;
  const heads = weights.heads;
  const normalisedPair = layerNormSlow(pair, pairRows, pairChannels,
                                       weights.pairInputLayerNormScale, null);
  const flatLogits = linear(normalisedPair, pairRows, pairChannels, blocks * heads,
                            weights.pairLogitsProjection);
  const pairLogits = [];
  for (let block = 0; block < blocks; block += 1) {
    const perBlock = new Float32Array(subsets * heads * queries * keys);
    for (let subset = 0; subset < subsets; subset += 1) {
      for (let query = 0; query < queries; query += 1) {
        for (let key = 0; key < keys; key += 1) {
          const source = (((subset * queries + query) * keys) + key) * blocks * heads;
          for (let head = 0; head < heads; head += 1) {
            perBlock[((subset * heads + head) * queries + query) * keys + key] =
              flatLogits[source + block * heads + head];
          }
        }
      }
    }
    pairLogits.push(perBlock);
  }

  const shape = { subsets, queries, keys, channels, heads,
                  dimension: weights.dimension };
  let act = queriesAct;
  for (let block = 0; block < blocks; block += 1) {
    act = crossAttentionBlock(act, {
      queriesToKeys: input.queriesToKeys, queriesMask, keysMask,
      queriesCond, keysCond, pairLogits: pairLogits[block],
    }, shape, weights.blocks[block]);
  }
  for (let index = 0; index < queryRows; index += 1) {
    for (let c = 0; c < channels; c += 1) act[index * channels + c] *= queriesMask[index];
  }

  // ...back to token-atom layout, rectified, and averaged over each token's
  // REAL atoms only.
  const perToken = weights.perTokenChannels;
  const projected = linear(act, queryRows, channels, perToken,
                           weights.projectAtomFeaturesForAggr);
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
  return output;
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
export function targetFeatures(input, tokens) {
  const restypes = 31;
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
