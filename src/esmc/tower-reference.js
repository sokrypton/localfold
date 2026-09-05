// ESM-C's tower and ESMFold2's language-model shim, on the CPU.
//
// The reference the WebGPU tower is checked against, and the place the
// conventions get pinned. Written from the block layout
// ../alphafold3/converters/esmc.py documents:
//
//     h    = LN(x; attn_norm) @ qkv            -> [q | k | v]
//     q, k = LN(q; q_norm), LN(k; k_norm)      // over the FULL d_model
//     q, k = RoPE(q, k)                        // head_dim 64, base 10000
//     x    = x + attn(q, k, v) @ attn_out / residualScale
//     x    = x + swiglu(LN(x; ffn_norm) @ fc1) @ fc2 / residualScale
//
// 🔴 THE QK LayerNorm IS OVER THE WHOLE d_model, NOT PER HEAD. It runs on the
// 1152-wide q and k BEFORE they are split into 18 heads of 64, so normalising
// per head is a different model that still runs, still folds, and is wrong by an
// amount no shape check can see.
//
// 🔴 AND RoPE HERE IS THE SPLIT-HALVES CONVENTION, not the interleaved one.
// `[x0..x31, x32..x63]` rotates as halves against each other; interleaving
// adjacent pairs is the other common reading and is also silent. Both of these
// are what `tools/check-esmc-reference.js` exists to catch, against a dump from
// a reference that agrees with transformers' own ESMC to 2.1e-6.
//
// 🔴 AND ONLY THE LAST STATE IS FINAL-NORMED. ESMFold2 mixes all 37 - the
// embedding plus 36 blocks - and the last one is post the stack's final
// LayerNorm while the other 36 are the raw residual stream. Returning the
// pre-norm value there reads corr 0.909 against native on that layer where every
// other layer is >= 0.9998.

// 🔴 EXPORTS ARE PREFIXED WHERE THE BARE NAME WOULD BE A PLAUSIBLE LOCAL.
// test/module-references.test.js guards against calling an exported name a file
// cannot see, and it builds ONE set of exported names across all of src/ - so
// exporting `block` made every function that takes a `block` PARAMETER look
// like it was calling ours. `src/af3/confidence-reference.js` takes exactly
// that parameter. The guard is right and the export name was wrong.

// ESM-C's alphabet, in id order. 0/1/2/32 are BOS/PAD/EOS/MASK.
export const VOCAB = ("<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y "
  + "M H W C X B U Z O . - | <mask>").split(" ");
const TOKEN = new Map(VOCAB.map((name, index) => [name, index]));
export const BOS = 0, PAD = 1, EOS = 2, UNK = TOKEN.get("<unk>"), MASK = 32;

/** One-letter sequence -> token ids, with BOS and EOS attached. */
export function sequenceIds(sequence) {
  const ids = new Int32Array(sequence.length + 2);
  ids[0] = BOS;
  for (let i = 0; i < sequence.length; i += 1) {
    const code = TOKEN.get(sequence[i].toUpperCase());
    ids[i + 1] = code === undefined ? UNK : code;
  }
  ids[ids.length - 1] = EOS;
  return ids;
}

/** In-place LayerNorm over the last axis. `offset` may be null. */
export function layerNorm(values, rows, channels, scale, offset, eps = 1e-5) {
  for (let row = 0; row < rows; row += 1) {
    const base = row * channels;
    let mean = 0;
    for (let c = 0; c < channels; c += 1) mean += values[base + c];
    mean /= channels;
    let variance = 0;
    for (let c = 0; c < channels; c += 1) {
      const d = values[base + c] - mean;
      variance += d * d;
    }
    const inverse = 1 / Math.sqrt(variance / channels + eps);
    for (let c = 0; c < channels; c += 1) {
      const normalised = (values[base + c] - mean) * inverse * scale[c];
      values[base + c] = offset === null ? normalised : normalised + offset[c];
    }
  }
  return values;
}

/**
 * (rows, inner) @ weights, with `weights` stored (inner, outer) row-major.
 *
 * 🔴 THIS IS NOT torch's LAYOUT AND THE EXPORT TRANSPOSES INTO IT. torch stores
 * a Linear as (out, in) and computes x @ W.T; the tiled kernel this port uses on
 * the GPU - src/evoformer/transition.js, measured at 1140-1550 GFLOP/s - indexes
 * `weights[k * columns + column]`, which is (in, out). One layout for the
 * bundle, the reference and the shader, decided at export; the alternative is a
 * transpose per fold or a second kernel.
 */
export function linear(input, rows, inner, weights, outer) {
  const out = new Float32Array(rows * outer);
  for (let row = 0; row < rows; row += 1) {
    const source = row * inner;
    const destination = row * outer;
    for (let i = 0; i < inner; i += 1) {
      const value = input[source + i];
      if (value === 0) continue;
      const w = i * outer;
      for (let o = 0; o < outer; o += 1) out[destination + o] += value * weights[w + o];
    }
  }
  return out;
}

/**
 * Rotary embedding, in place, on a (rows, heads, headDim) tensor.
 *
 * The rotation pairs channel `d` with `d + headDim / 2`, which is the
 * split-halves convention llama and ESM-C share.
 */
export function applyRope(values, rows, heads, headDim, base = 10000) {
  const half = headDim >> 1;
  for (let row = 0; row < rows; row += 1) {
    for (let d = 0; d < half; d += 1) {
      const frequency = row / Math.pow(base, (2 * d) / headDim);
      const cos = Math.cos(frequency), sin = Math.sin(frequency);
      for (let head = 0; head < heads; head += 1) {
        const slot = (row * heads + head) * headDim + d;
        const first = values[slot], second = values[slot + half];
        values[slot] = first * cos - second * sin;
        values[slot + half] = first * sin + second * cos;
      }
    }
  }
  return values;
}

/** Full self-attention. q/k/v are (rows, heads, headDim); returns (rows, heads*headDim). */
export function esmcAttention(query, key, value, rows, heads, headDim) {
  const out = new Float32Array(rows * heads * headDim);
  const scale = 1 / Math.sqrt(headDim);
  const weights = new Float32Array(rows);
  for (let head = 0; head < heads; head += 1) {
    for (let i = 0; i < rows; i += 1) {
      const qBase = (i * heads + head) * headDim;
      let largest = -Infinity;
      for (let j = 0; j < rows; j += 1) {
        const kBase = (j * heads + head) * headDim;
        let dot = 0;
        for (let d = 0; d < headDim; d += 1) dot += query[qBase + d] * key[kBase + d];
        const logit = dot * scale;
        weights[j] = logit;
        if (logit > largest) largest = logit;
      }
      let total = 0;
      for (let j = 0; j < rows; j += 1) {
        const w = Math.exp(weights[j] - largest);
        weights[j] = w;
        total += w;
      }
      const outBase = (i * heads + head) * headDim;
      for (let j = 0; j < rows; j += 1) {
        const w = weights[j] / total;
        if (w === 0) continue;
        const vBase = (j * heads + head) * headDim;
        for (let d = 0; d < headDim; d += 1) out[outBase + d] += w * value[vBase + d];
      }
    }
  }
  return out;
}

const silu = (x) => x / (1 + Math.exp(-x));

/** One ESM-C block. `x` is (rows, model) and is not modified. */
export function esmcBlock(x, rows, weights, dims, layer) {
  const { model, heads, residualScale } = dims;
  const headDim = model / heads;
  const at = (leaf) => weights[`blocks/${layer}/${leaf}`];

  const normed = layerNorm(x.slice(), rows, model,
    at("attn_norm/scale"), at("attn_norm/offset"));
  const projected = linear(normed, rows, model, at("qkv/weights"), 3 * model);

  const query = new Float32Array(rows * model);
  const key = new Float32Array(rows * model);
  const value = new Float32Array(rows * model);
  for (let row = 0; row < rows; row += 1) {
    const source = row * 3 * model;
    query.set(projected.subarray(source, source + model), row * model);
    key.set(projected.subarray(source + model, source + 2 * model), row * model);
    value.set(projected.subarray(source + 2 * model, source + 3 * model), row * model);
  }
  // Over the FULL width, before the heads exist. No offset on either.
  layerNorm(query, rows, model, at("q_norm/scale"), null);
  layerNorm(key, rows, model, at("k_norm/scale"), null);
  applyRope(query, rows, heads, headDim);
  applyRope(key, rows, heads, headDim);

  const context = esmcAttention(query, key, value, rows, heads, headDim);
  const attention = linear(context, rows, model, at("attn_out/weights"), model);

  const afterAttention = new Float32Array(rows * model);
  for (let i = 0; i < afterAttention.length; i += 1) {
    afterAttention[i] = x[i] + attention[i] / residualScale;
  }

  const ffnNormed = layerNorm(afterAttention.slice(), rows, model,
    at("ffn_norm/scale"), at("ffn_norm/offset"));
  const hidden = linear(ffnNormed, rows, model, at("fc1/weights"), 2 * dims.ffn);
  const gated = new Float32Array(rows * dims.ffn);
  for (let row = 0; row < rows; row += 1) {
    const source = row * 2 * dims.ffn;
    const destination = row * dims.ffn;
    for (let c = 0; c < dims.ffn; c += 1) {
      gated[destination + c] = silu(hidden[source + c]) * hidden[source + dims.ffn + c];
    }
  }
  const projectedOut = linear(gated, rows, dims.ffn, at("fc2/weights"), model);

  const out = new Float32Array(rows * model);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = afterAttention[i] + projectedOut[i] / residualScale;
  }
  return out;
}

/** Token ids -> the layers + 1 hidden states ESMFold2 is allowed to mix. */
export function towerStates(ids, weights, dims) {
  const { model, layers } = dims;
  const rows = ids.length;
  const table = weights["embed/weights"];
  let x = new Float32Array(rows * model);
  for (let row = 0; row < rows; row += 1) {
    x.set(table.subarray(ids[row] * model, (ids[row] + 1) * model), row * model);
  }
  const states = [x];
  for (let layer = 0; layer < layers; layer += 1) {
    x = esmcBlock(x, rows, weights, dims, layer);
    states.push(x);
  }
  states[states.length - 1] = layerNorm(states[states.length - 1].slice(),
    rows, model, weights["final_norm/scale"], null);
  return states;
}

/** softmax over the mix logits; a constant, so it folds once. */
export function layerMix(combine) {
  let largest = -Infinity;
  for (const value of combine) if (value > largest) largest = value;
  const out = new Float32Array(combine.length);
  let total = 0;
  for (let i = 0; i < combine.length; i += 1) {
    out[i] = Math.exp(combine[i] - largest);
    total += out[i];
  }
  for (let i = 0; i < out.length; i += 1) out[i] /= total;
  return out;
}

/**
 * The 37 states -> the (rows, pair) single ESMFold2 mixes them into.
 *
 * 🔴 THE STATES NEVER HAVE TO EXIST AT ONCE. The mix is constant and the norm
 * and projection are shared across k, so this is a running accumulator - which
 * is what lets the WebGPU tower stream one block's weights at a time instead of
 * holding a (37, tokens, 1152) tensor. `towerStates` materialises them because
 * a REFERENCE should be obvious; the accumulation is the shipping shape.
 */
export function shimSingle(states, rows, weights, dims) {
  const { model, pair } = dims;
  const mix = layerMix(weights["lm/combine"]);
  const accumulator = new Float32Array(rows * pair);
  for (let k = 0; k < states.length; k += 1) {
    const normed = layerNorm(states[k].slice(), rows, model,
      weights["lm/norm/scale"], weights["lm/norm/offset"]);
    const projected = linear(normed, rows, model, weights["lm/projection/weights"], pair);
    for (let i = 0; i < accumulator.length; i += 1) accumulator[i] += mix[k] * projected[i];
  }
  // The downprojection is affine and sits OUTSIDE the sum: its bias must not be
  // added once per state.
  const down = linear(accumulator, rows, pair, weights["lm/downproject/weights"], pair);
  const bias = weights["lm/downproject/bias"];
  for (let row = 0; row < rows; row += 1) {
    for (let c = 0; c < pair; c += 1) down[row * pair + c] += bias[c];
  }
  return down;
}

const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

/** Abramowitz-Stegun 7.1.26, to about 1.5e-7 - below the tolerance any arm here uses. */
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const z = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * z);
  const series = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
    + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - series * Math.exp(-z * z));
}

/**
 * (rows, pair) -> (rows, rows, pair). The outer product carries BOTH a product
 * and a difference, so the pair representation sees magnitude and direction.
 */
export function shimPair(single, rows, weights, dims) {
  const { pair } = dims;
  const joined = new Float32Array(rows * rows * 2 * pair);
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < rows; j += 1) {
      const destination = (i * rows + j) * 2 * pair;
      for (let c = 0; c < pair; c += 1) {
        const a = single[i * pair + c], b = single[j * pair + c];
        joined[destination + c] = a * b;
        joined[destination + pair + c] = a - b;
      }
    }
  }
  const cells = rows * rows;
  const first = linear(joined, cells, 2 * pair, weights["lm/pair_mlp_1/weights"], pair);
  const bias1 = weights["lm/pair_mlp_1/bias"];
  for (let cell = 0; cell < cells; cell += 1) {
    for (let c = 0; c < pair; c += 1) {
      first[cell * pair + c] = gelu(first[cell * pair + c] + bias1[c]);
    }
  }
  const second = linear(first, cells, pair, weights["lm/pair_mlp_2/weights"], pair);
  const bias2 = weights["lm/pair_mlp_2/bias"];
  for (let cell = 0; cell < cells; cell += 1) {
    for (let c = 0; c < pair; c += 1) second[cell * pair + c] += bias2[c];
  }
  return layerNorm(second, cells, pair,
    weights["lm/pair_norm/scale"], weights["lm/pair_norm/offset"]);
}
