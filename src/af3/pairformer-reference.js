/**
 * One AF3 pairformer block, on the CPU, as plainly as it can be written.
 *
 * This is the independent reference every AF3 GPU kernel is tested against, in
 * the sense AGENTS.md means: it shares no code with the shaders, so agreement
 * between them is evidence rather than a tautology. It is also where the wiring
 * gets to be wrong cheaply - the block is ten operations and about thirty
 * shape conventions, and it is the conventions that bite.
 *
 * THE BLOCK (config `openfold3`/`alphafold3`, no dropout). Each update reads the
 * running activation, not the block input - AF3 threads them sequentially where
 * chai-1 sums parallel deltas, which is why the order below is load-bearing:
 *
 *     pair   += triangleMultiplication(pair, "outgoing")
 *     pair   += triangleMultiplication(pair, "incoming")
 *     pair   += gridSelfAttention(pair, transpose = false)
 *     pair   += gridSelfAttention(pair, transpose = true)
 *     pair   += transition(pair)
 *     single += singleAttention(single, pairLogits(pair))
 *     single += transition(single)
 *
 * 🔴 TWO SPLITS OF A DOUBLE-WIDTH PROJECTION, AND THEY DISAGREE. Both the
 * triangle multiplication and the transitions project to twice the width they
 * need and split the result in half, they sit forty lines apart in AF3's
 * modules.py, and they are cut along DIFFERENT axes:
 *
 *   transition       BLOCKED.      transition1 is (in, 2 * intermediate) and is
 *                    reshaped (in, 2, intermediate), so the gate half is the
 *                    FIRST `intermediate` outputs and the value half the second.
 *
 *   triangle mul     INTERLEAVED.  projection is (in, 2c), but the result is
 *                    TRANSPOSED to (2c, i, j) BEFORE being reshaped to
 *                    (c, 2, i, j) - so channel `ch`'s two halves are outputs
 *                    `ch * 2` and `ch * 2 + 1`, not `ch` and `ch + c`.
 *
 * Getting either backwards costs no error and no NaN. It permutes channels and
 * returns a plausible tensor, which is why both are spelled out at their use
 * sites rather than left to a shared "split in half" helper.
 *
 * 🔴 NEARLY NOTHING HERE HAS A BIAS. AF3 writes `bias_init=1.0` on both gates,
 * which reads as "this gate starts open" and is inert: those Linears are
 * bias-free, so the initialiser never materialises a parameter. The exported
 * tensor table agrees - one `bias` in the whole block, on the single track's
 * query projection. A reference that helpfully added the missing biases would
 * open every gate and be wrong everywhere.
 */

const EPSILON = 1e-5;

const sigmoid = (value) => 1 / (1 + Math.exp(-value));
const swish = (value) => value * sigmoid(value);

/**
 * LayerNorm over the last axis, with AF3's variance.
 *
 * 🔴 `use_fast_variance` IS THE DEFAULT AND IT IS NOT THE TWO-PASS FORMULA:
 * AF3 takes E[x^2] - E[x]^2, which is algebraically the same and numerically
 * is not. The trunk's single representation reaches 1.7e5, where the two
 * expectations agree to about six digits and their difference does not.
 *
 * @param {Float32Array} input   rows x channels
 * @param {number} rows
 * @param {number} channels
 * @param {Float32Array} scale   channels
 * @param {Float32Array} offset  channels
 * @returns {Float32Array} rows x channels
 */
export function layerNorm(input, rows, channels, scale, offset) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    const base = row * channels;
    let sum = 0;
    let sumSquares = 0;
    for (let c = 0; c < channels; c += 1) {
      const value = input[base + c];
      sum += value;
      sumSquares += value * value;
    }
    const mean = sum / channels;
    const variance = sumSquares / channels - mean * mean;
    const inverse = 1 / Math.sqrt(variance + EPSILON);
    for (let c = 0; c < channels; c += 1) {
      output[base + c] = (input[base + c] - mean) * inverse * scale[c] + offset[c];
    }
  }
  return output;
}

/**
 * rows x inChannels  ->  rows x outChannels.
 *
 * @param {Float32Array} input
 * @param {number} rows
 * @param {number} inChannels
 * @param {number} outChannels
 * @param {Float32Array} weights  (in, out), or (out, in) when `transposed`
 * @param {Float32Array|null} bias
 * @param {boolean} transposed    AF3's `transpose_weights`
 */
export function linear(input, rows, inChannels, outChannels, weights, bias = null,
                       transposed = false) {
  const output = new Float32Array(rows * outChannels);
  for (let row = 0; row < rows; row += 1) {
    const inputBase = row * inChannels;
    const outputBase = row * outChannels;
    for (let out = 0; out < outChannels; out += 1) {
      let total = bias === null ? 0 : bias[out];
      if (transposed) {
        const weightBase = out * inChannels;
        for (let c = 0; c < inChannels; c += 1) {
          total += input[inputBase + c] * weights[weightBase + c];
        }
      } else {
        for (let c = 0; c < inChannels; c += 1) {
          total += input[inputBase + c] * weights[c * outChannels + out];
        }
      }
      output[outputBase + out] = total;
    }
  }
  return output;
}

/** Softmax over the last axis, in place, on `rows` rows of `width`. */
function softmaxRows(values, rows, width) {
  for (let row = 0; row < rows; row += 1) {
    const base = row * width;
    let largest = -Infinity;
    for (let i = 0; i < width; i += 1) {
      if (values[base + i] > largest) largest = values[base + i];
    }
    let total = 0;
    for (let i = 0; i < width; i += 1) {
      const value = Math.exp(values[base + i] - largest);
      values[base + i] = value;
      total += value;
    }
    for (let i = 0; i < width; i += 1) values[base + i] /= total;
  }
  return values;
}

/**
 * AF3's gated transition: LayerNorm, then SwiGLU, then project back.
 *
 * The BLOCKED split - see the note at the top. `transition1` widens to twice
 * the intermediate size and the first half is the one that goes through swish.
 *
 * @param {Float32Array} input rows x channels
 * @param {number} rows
 * @param {number} channels
 * @param {{inputLayerNormScale: Float32Array, inputLayerNormOffset: Float32Array,
 *          transition1: Float32Array, transition2: Float32Array}} weights
 */
export function transition(input, rows, channels, weights, factor = 4) {
  // 🔴 THE WIDENING FACTOR IS NOT ALWAYS FOUR. The trunk's transitions use 4
  // (128 -> 512, so transition1 is 128x1024), and the TEMPLATE stack's use 2
  // (64 -> 128, transition1 64x256). Both are "a transition block" and both
  // read `transition1` and `transition2`; the only thing that says which is the
  // shape of the weights, so a wrong factor here reads them at the wrong stride
  // rather than failing.
  const intermediate = channels * factor;
  const normalised = layerNorm(input, rows, channels, weights.inputLayerNormScale,
                               weights.inputLayerNormOffset);
  const wide = linear(normalised, rows, channels, intermediate * 2, weights.transition1);
  const gated = new Float32Array(rows * intermediate);
  for (let row = 0; row < rows; row += 1) {
    const wideBase = row * intermediate * 2;
    const gatedBase = row * intermediate;
    for (let i = 0; i < intermediate; i += 1) {
      // ...[gate half | value half], in that order.
      gated[gatedBase + i] = swish(wide[wideBase + i]) * wide[wideBase + intermediate + i];
    }
  }
  return linear(gated, rows, intermediate, channels, weights.transition2);
}

/**
 * Triangle multiplication, outgoing (`ikc,jkc->ijc`) or incoming (`kjc,kic->ijc`).
 *
 * @param {Float32Array} pair  n*n*channels
 * @param {Float32Array} mask  n*n
 * @param {number} n
 * @param {number} channels
 * @param {"outgoing"|"incoming"} direction
 * @param {object} weights
 */
export function triangleMultiplication(pair, mask, n, channels, direction, weights) {
  const pairs = n * n;
  const normalised = layerNorm(pair, pairs, channels, weights.leftNormInputScale,
                               weights.leftNormInputOffset);
  const projection = linear(normalised, pairs, channels, channels * 2, weights.projection);
  const gate = linear(normalised, pairs, channels, channels * 2, weights.gate);

  // The INTERLEAVED split - see the note at the top. `a` and `b` are laid out
  // as (channel, i, j) because that is the axis order the einsum contracts in.
  const a = new Float32Array(channels * pairs);
  const b = new Float32Array(channels * pairs);
  for (let index = 0; index < pairs; index += 1) {
    const wide = index * channels * 2;
    const maskValue = mask[index];
    for (let c = 0; c < channels; c += 1) {
      const gated = maskValue * sigmoid(gate[wide + c * 2]);
      a[c * pairs + index] = projection[wide + c * 2] * gated;
      b[c * pairs + index] = projection[wide + c * 2 + 1]
        * maskValue * sigmoid(gate[wide + c * 2 + 1]);
    }
  }

  const product = new Float32Array(channels * pairs);
  for (let c = 0; c < channels; c += 1) {
    const plane = c * pairs;
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        let total = 0;
        if (direction === "outgoing") {
          // out[c,i,j] = sum_k a[c,i,k] * b[c,j,k]
          for (let k = 0; k < n; k += 1) {
            total += a[plane + i * n + k] * b[plane + j * n + k];
          }
        } else {
          // out[c,i,j] = sum_k a[c,k,j] * b[c,k,i]
          for (let k = 0; k < n; k += 1) {
            total += a[plane + k * n + j] * b[plane + k * n + i];
          }
        }
        product[plane + i * n + j] = total;
      }
    }
  }

  // 🔴 `center_norm` NORMALISES OVER THE CHANNEL AXIS OF A (c, i, j) ARRAY.
  // AF3 spells it LayerNorm(axis=0, param_axis=0), which reads like "normalise
  // over channels" only once you know the array is channel-major here. Doing it
  // over i or j instead still runs and still returns finite numbers.
  const centred = new Float32Array(channels * pairs);
  for (let index = 0; index < pairs; index += 1) {
    let sum = 0;
    let sumSquares = 0;
    for (let c = 0; c < channels; c += 1) {
      const value = product[c * pairs + index];
      sum += value;
      sumSquares += value * value;
    }
    const mean = sum / channels;
    const inverse = 1 / Math.sqrt(sumSquares / channels - mean * mean + EPSILON);
    for (let c = 0; c < channels; c += 1) {
      centred[index * channels + c] =
        (product[c * pairs + index] - mean) * inverse * weights.centerNormScale[c]
        + weights.centerNormOffset[c];
    }
  }

  const projected = linear(centred, pairs, channels, channels, weights.outputProjection);
  const outputGate = linear(normalised, pairs, channels, channels, weights.gatingLinear);
  for (let index = 0; index < projected.length; index += 1) {
    projected[index] *= sigmoid(outputGate[index]);
  }
  return projected;
}

/**
 * Triangle ("grid") self-attention over the pair representation.
 *
 * `transpose` picks the direction: false attends within a row, true within a
 * column, and AF3 gets the second by transposing the activation rather than by
 * writing a second kernel.
 *
 * 🔴 THE OPENFOLD3 LINEAGE ALSO TRANSPOSES THE PAIR BIAS, AND STOCK AF3 DOES
 * NOT. Same weights, same shapes, silently different answer - so the dialect
 * has to be passed in rather than assumed. (openfold3, opendde, boltz2 and
 * protenix2 swap it; alphafold3 and rosettafold3 do not.)
 *
 * @param {Float32Array} pair n*n*channels
 * @param {Float32Array} mask n*n
 * @param {number} n
 * @param {number} channels
 * @param {boolean} transpose
 * @param {object} weights
 * @param {{swapTransposedBias: boolean}} dialect
 */
export function gridSelfAttention(pair, mask, n, channels, transpose, weights, dialect) {
  const pairs = n * n;
  const heads = weights.heads;
  const dimension = weights.dimension;
  const scale = 1 / Math.sqrt(dimension);
  const normalised = layerNorm(pair, pairs, channels, weights.actNormScale,
                               weights.actNormOffset);

  // bias[h][i][j], from the UNTRANSPOSED activation in every dialect.
  const rawBias = linear(normalised, pairs, channels, heads, weights.pairBiasProjection);
  const swap = transpose && dialect.swapTransposedBias;
  const bias = new Float32Array(heads * pairs);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      for (let h = 0; h < heads; h += 1) {
        const source = swap ? (j * n + i) : (i * n + j);
        bias[h * pairs + i * n + j] = rawBias[source * heads + h];
      }
    }
  }

  // The activation itself, transposed for the column direction.
  //
  // 🔴 THIS TRANSPOSES THE NORMALISED ACTIVATION, NOT THE PAIR. AF3 normalises
  // once at the top and every reader below - q, k, v, the gate - sees that.
  // Reading the raw pair here instead costs no error: it multiplies the
  // attention inputs by the ~450x the LayerNorm was removing, and the op
  // returns a finite tensor about eighty times too large.
  const act = new Float32Array(pairs * channels);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const from = (transpose ? (j * n + i) : (i * n + j)) * channels;
      const to = (i * n + j) * channels;
      for (let c = 0; c < channels; c += 1) act[to + c] = normalised[from + c];
    }
  }

  const width = heads * dimension;
  // q and k carry `transpose_weights`, v does not. The exported shapes say so:
  // (heads, dimension, channels) against (channels, heads, dimension).
  const q = linear(act, pairs, channels, width, weights.qProjection, null, true);
  const k = linear(act, pairs, channels, width, weights.kProjection, null, true);
  const v = linear(act, pairs, channels, width, weights.vProjection);

  const gathered = new Float32Array(pairs * width);
  const logits = new Float32Array(n);
  for (let row = 0; row < n; row += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let i = 0; i < n; i += 1) {
        const queryBase = (row * n + i) * width + head * dimension;
        for (let j = 0; j < n; j += 1) {
          const keyBase = (row * n + j) * width + head * dimension;
          let dot = 0;
          for (let d = 0; d < dimension; d += 1) {
            dot += q[queryBase + d] * k[keyBase + d];
          }
          // ...the mask is over the KEY position, and it is the transposed
          // pair mask: AF3 swaps its axes before slicing a row out of it.
          const masked = mask[transpose ? (j * n + row) : (row * n + j)];
          logits[j] = dot * scale + bias[head * pairs + i * n + j]
            + (masked > 0 ? 0 : -1e9);
        }
        softmaxRows(logits, 1, n);
        const outBase = (row * n + i) * width + head * dimension;
        for (let d = 0; d < dimension; d += 1) {
          let total = 0;
          for (let j = 0; j < n; j += 1) {
            total += logits[j] * v[(row * n + j) * width + head * dimension + d];
          }
          gathered[outBase + d] = total;
        }
      }
    }
  }

  const gate = linear(act, pairs, channels, width, weights.gatingQuery, null, true);
  for (let index = 0; index < gathered.length; index += 1) {
    gathered[index] *= sigmoid(gate[index]);
  }
  const projected = linear(gathered, pairs, width, channels, weights.outputProjection);

  if (!transpose) return projected;
  // ...and back, so the residual lands on the orientation it came from.
  const restored = new Float32Array(projected.length);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const from = (j * n + i) * channels;
      const to = (i * n + j) * channels;
      for (let c = 0; c < channels; c += 1) restored[to + c] = projected[from + c];
    }
  }
  return restored;
}

/**
 * The single track's attention, biased by the pair representation.
 *
 * Structurally this is AF2's row attention with a pair bias, applied to one row.
 * The only genuine bias parameter in the block is on this module's queries.
 *
 * @param {Float32Array} single n*channels
 * @param {Float32Array} pairLogits heads*n*n
 * @param {Float32Array} seqMask n
 * @param {number} n
 * @param {number} channels
 * @param {object} weights
 */
export function singleAttention(single, pairLogits, seqMask, n, channels, weights) {
  const heads = weights.heads;
  const dimension = weights.dimension;
  const scale = 1 / Math.sqrt(dimension);
  const width = heads * dimension;
  const normalised = layerNorm(single, n, channels, weights.layerNormScale,
                               weights.layerNormOffset);
  const q = linear(normalised, n, channels, width, weights.qProjection, weights.qBias);
  const k = linear(normalised, n, channels, width, weights.kProjection);
  const v = linear(normalised, n, channels, width, weights.vProjection);

  const gathered = new Float32Array(n * width);
  const logits = new Float32Array(n);
  for (let head = 0; head < heads; head += 1) {
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        let dot = 0;
        for (let d = 0; d < dimension; d += 1) {
          dot += q[i * width + head * dimension + d] * k[j * width + head * dimension + d];
        }
        logits[j] = dot * scale + pairLogits[head * n * n + i * n + j]
          + 1e9 * (seqMask[j] - 1);
      }
      softmaxRows(logits, 1, n);
      for (let d = 0; d < dimension; d += 1) {
        let total = 0;
        for (let j = 0; j < n; j += 1) {
          total += logits[j] * v[j * width + head * dimension + d];
        }
        gathered[i * width + head * dimension + d] = total;
      }
    }
  }

  const gate = linear(normalised, n, channels, width, weights.gatingQuery);
  for (let index = 0; index < gathered.length; index += 1) {
    gathered[index] *= sigmoid(gate[index]);
  }
  return linear(gathered, n, width, channels, weights.outputProjection);
}

/**
 * One pairformer block: pair and single in, pair and single out.
 *
 * @param {{pair: Float32Array, single: Float32Array, pairMask: Float32Array,
 *          seqMask: Float32Array, tokens: number}} state
 * @param {object} weights  one block's tensors, already sliced out of the stack
 * @param {{swapTransposedBias: boolean}} dialect
 */
export function pairformerBlock(state, weights, dialect) {
  const { pairMask, seqMask, tokens } = state;
  const pairChannels = weights.pairChannels;
  const singleChannels = weights.singleChannels;
  let pair = Float32Array.from(state.pair);

  const addPair = (delta) => {
    for (let index = 0; index < pair.length; index += 1) pair[index] += delta[index];
  };

  addPair(triangleMultiplication(pair, pairMask, tokens, pairChannels, "outgoing",
                                 weights.triangleMultiplicationOutgoing));
  addPair(triangleMultiplication(pair, pairMask, tokens, pairChannels, "incoming",
                                 weights.triangleMultiplicationIncoming));
  addPair(gridSelfAttention(pair, pairMask, tokens, pairChannels, false,
                            weights.pairAttention1, dialect));
  addPair(gridSelfAttention(pair, pairMask, tokens, pairChannels, true,
                            weights.pairAttention2, dialect));
  addPair(transition(pair, tokens * tokens, pairChannels, weights.pairTransition));

  // The single track reads the pair AFTER all five pair updates.
  const logitsSource = layerNorm(pair, tokens * tokens, pairChannels,
                                 weights.singlePairLogitsNormScale,
                                 weights.singlePairLogitsNormOffset);
  const heads = weights.singleAttention.heads;
  const flat = linear(logitsSource, tokens * tokens, pairChannels, heads,
                      weights.singlePairLogitsProjection);
  const pairLogits = new Float32Array(heads * tokens * tokens);
  for (let index = 0; index < tokens * tokens; index += 1) {
    for (let head = 0; head < heads; head += 1) {
      pairLogits[head * tokens * tokens + index] = flat[index * heads + head];
    }
  }

  let single = Float32Array.from(state.single);
  const attended = singleAttention(single, pairLogits, seqMask, tokens, singleChannels,
                                   weights.singleAttention);
  for (let index = 0; index < single.length; index += 1) single[index] += attended[index];
  const transitioned = transition(single, tokens, singleChannels, weights.singleTransition);
  for (let index = 0; index < single.length; index += 1) single[index] += transitioned[index];

  return { pair, single };
}
