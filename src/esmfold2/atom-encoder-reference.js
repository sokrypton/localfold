/**
 * ESMFold2's inputs embedder: a sliding-window atom transformer with 3D RoPE.
 *
 *     c0 = LayerNorm(atomFeatures @ atom_linear)
 *     q  = 3 x swaBlock(c0, conditioned on c0)
 *     a  = scatterMean(relu(q @ atom_to_token), atom_to_token_index)
 *     s_inputs = [a | one_hot(res_type) | profile | deletion_mean]   (451)
 *
 * 🔴 THIS IS THE ONE PRIMITIVE WITH NO AF3 ANALOGUE, AND REUSING AF3's ATOM
 * ENCODER WOULD HAVE BEEN THE OBVIOUS WRONG MOVE. AF3 runs 32-query/128-key
 * windowed attention biased by a pair representation. This runs plain
 * sliding-window self-attention whose only positional signal is a rotary
 * embedding built from the REFERENCE CONFORMER's coordinates. Nothing about
 * the shapes says so - both are "an atom transformer" at 128 channels.
 *
 * 🔴 AND THE WINDOW IS OVER RANK AMONG VALID ATOMS, NOT OVER RAW INDEX. Two
 * atoms 64 apart in the array can be adjacent in rank if the atoms between
 * them are padding. The diagonal is always allowed, so a masked atom still
 * attends to itself and the softmax has something to normalise.
 *
 * 🔴 AND EVERY NORM HERE IS AFFINE-FREE RMS, WITH torch's eps AND NOT 1e-5.
 * `F.rms_norm(eps=None)` uses `finfo(float32).eps`, which is 1.19e-7 - about a
 * hundred times smaller than the LayerNorm epsilon used everywhere else in
 * this repository. Only `atom_norm` is a real LayerNorm with a scale and an
 * offset; the four inside a block are `x * rsqrt(mean(x^2) + eps)` and nothing
 * else.
 *
 * The spec is ../alphafold3's `converters/oracles/esmfold2_reference.py`,
 * which is an independently written JAX ESMFold2; this agrees with it term for
 * term and is checked against the native model's own recorded output.
 */

/** One-hot widths, from ESMFold2's own constants. */
export const MAX_ELEMENT = 128;
export const NAME_CHARS = 64;
export const NAME_LENGTH = 4;
/** 3 + charge + mask + elements + name characters. */
export const ATOM_FEATURES = 3 + 1 + 1 + MAX_ELEMENT + NAME_LENGTH * NAME_CHARS;

/**
 * 🔴 torch's F.rms_norm(eps=None), NOT this repository's 1e-5. See the note
 * above; at 128 channels the difference is small and it is not zero, and a
 * checker with a 1e-6 bound sees it.
 */
const RMS_EPSILON = 1.1920928955078125e-7;

function rmsNorm(values, rows, channels) {
  const out = new Float32Array(values.length);
  for (let row = 0; row < rows; row += 1) {
    const base = row * channels;
    let sum = 0;
    for (let c = 0; c < channels; c += 1) sum += values[base + c] * values[base + c];
    const scale = 1 / Math.sqrt(sum / channels + RMS_EPSILON);
    for (let c = 0; c < channels; c += 1) out[base + c] = values[base + c] * scale;
  }
  return out;
}

function layerNorm(values, rows, channels, scale, offset, epsilon = 1e-5) {
  const out = new Float32Array(values.length);
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
    const by = 1 / Math.sqrt(variance / channels + epsilon);
    for (let c = 0; c < channels; c += 1) {
      out[base + c] = (values[base + c] - mean) * by * scale[c] + offset[c];
    }
  }
  return out;
}

/** rows x inChannels against (inChannels, outChannels). */
function linear(input, rows, inChannels, outChannels, weights) {
  const out = new Float32Array(rows * outChannels);
  for (let row = 0; row < rows; row += 1) {
    const inBase = row * inChannels;
    const outBase = row * outChannels;
    for (let c = 0; c < inChannels; c += 1) {
      const value = input[inBase + c];
      if (value === 0) continue;
      const weightBase = c * outChannels;
      for (let o = 0; o < outChannels; o += 1) out[outBase + o] += value * weights[weightBase + o];
    }
  }
  return out;
}

/**
 * Round to bfloat16 and back, round-to-nearest-even.
 *
 * 🔴 THE ATTENTION RUNS IN bfloat16 WHATEVER THE MODEL'S DTYPE, WHICH IS A
 * PROPERTY OF THE MODULE AND NOT OF HOW IT WAS LOADED:
 *
 *     if q.dtype not in (torch.float16, torch.bfloat16):
 *         q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
 *
 * So a float32 checkpoint still computes this attention at eight mantissa bits,
 * and an f32 port CANNOT agree with it below about 1e-4. Matching it to 1e-7
 * needs the downcast reproduced; a checker with a tighter bound and no downcast
 * chases a convention bug that is not there. Everything else in the block -
 * the norms, the modulation, the SwiGLU, the gate - stays float32.
 */
const bfloat16 = (() => {
  const floats = new Float32Array(1);
  const bits = new Uint32Array(floats.buffer);
  return (value) => {
    floats[0] = value;
    const word = bits[0];
    // ...round to nearest, ties to even, on the low 16 bits.
    const rounded = (word + 0x7fff + ((word >>> 16) & 1)) & 0xffff0000;
    bits[0] = rounded;
    return floats[0];
  };
})();

const roundToBfloat16 = (values) => {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) out[i] = bfloat16(values[i]);
  return out;
};

const silu = (x) => x / (1 + Math.exp(-x));
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** `1 / base ** (i / n)` - ESMFold2's spacing, which divides by n and not n-1. */
export function ropeInverseFrequencies(pairs, base) {
  const out = new Float32Array(pairs);
  for (let i = 0; i < pairs; i += 1) out[i] = 1 / Math.pow(base, i / pairs);
  return out;
}

/**
 * The rotary table: three spatial axes plus the reference-space uid.
 *
 * 🔴 `3 * spatialPairs + uidPairs` MUST FILL `headDim / 2` EXACTLY, and here it
 * does: 3 x 2 + 10 = 16 = 32 / 2. The upstream code zero-pads a short table
 * rather than failing, so a wrong pair count is a silently smaller rotation.
 */
export function buildRope(refPos, spaceUid, atoms, headDim,
                          { spatialPairs = 2, uidPairs = 10,
                            spatialBase = 20, uidBase = 10000,
                            precision = "bf16" } = {}) {
  const half = headDim >> 1;
  const spatial = ropeInverseFrequencies(spatialPairs, spatialBase);
  const uid = ropeInverseFrequencies(uidPairs, uidBase);
  const cos = new Float32Array(atoms * half);
  const sin = new Float32Array(atoms * half);
  for (let atom = 0; atom < atoms; atom += 1) {
    const base = atom * half;
    let at = 0;
    // ...axis-major: x's pairs, then y's, then z's. The flattening upstream is
    // `(ref_pos[..., None] * inv).reshape(atoms, -1)`, which is this order.
    for (let axis = 0; axis < 3; axis += 1) {
      const position = refPos[atom * 3 + axis];
      for (let i = 0; i < spatialPairs; i += 1, at += 1) {
        const angle = position * spatial[i];
        cos[base + at] = Math.cos(angle);
        sin[base + at] = Math.sin(angle);
      }
    }
    for (let i = 0; i < uidPairs && at < half; i += 1, at += 1) {
      const angle = spaceUid[atom] * uid[i];
      cos[base + at] = Math.cos(angle);
      sin[base + at] = Math.sin(angle);
    }
    // A table shorter than the half is zero-padded, which is cos = 1, sin = 0.
    for (; at < half; at += 1) { cos[base + at] = 1; sin[base + at] = 0; }
  }
  // 🔴 THE TABLE IS STORED IN bfloat16, IN A float32 MODEL, AND THAT IS WORTH
  // 2.4e-3 THROUGH THE ATTENTION. `build_3d_rope` computes in float32 and the
  // table arrives at the attention at eight mantissa bits; cos alone is 1.05e-3
  // away from the exact one. Everything else about the table - axis-major with
  // the pairs inner, the two bases, the frequency spacing that divides by n and
  // not n-1 - was already exact: rounding this reference's own table to
  // bfloat16 reproduces the native one on ALL 5120 entries, bit for bit.
  //
  // That is why it is here and not in a comment. Three days of a residual that
  // looks exactly like a convention bug, is not one, and does not move when
  // every convention is swept, is what this line costs to omit.
  return precision === "bf16"
    ? { cos: roundToBfloat16(cos), sin: roundToBfloat16(sin) }
    : { cos, sin };
}

/**
 * Rotate in place, per head.
 *
 * 🔴 THE TABLE IS TILED `[c | c]`, NOT INTERLEAVED, because it pairs with a
 * rotate_half that SPLITS INTO HALVES: element i pairs with i + half, not with
 * i + 1. Interleaving instead is the same shapes and reads corr 0.88 - a
 * plausible tensor, measured by ../alphafold3 - and it is the same trap
 * tools/check-esmc-reference.js sweeps for the ESM-C tower.
 */
function applyRope(values, atoms, heads, headDim, cos, sin) {
  const half = headDim >> 1;
  const out = new Float32Array(values.length);
  for (let atom = 0; atom < atoms; atom += 1) {
    const table = atom * half;
    for (let head = 0; head < heads; head += 1) {
      const base = (atom * heads + head) * headDim;
      for (let i = 0; i < half; i += 1) {
        const a = values[base + i];
        const b = values[base + half + i];
        const c = cos[table + i];
        const s = sin[table + i];
        out[base + i] = a * c - b * s;
        out[base + half + i] = b * c + a * s;
      }
    }
  }
  return out;
}

/**
 * Sliding-window self-attention over atoms.
 *
 * @param {Float32Array} valid one per atom, 1 or 0
 */
export function slidingWindowAttention(input, atoms, channels, heads, weights,
                                       rope, valid, halfWindow, precision = "bf16") {
  const headDim = channels / heads;
  const narrow = precision === "bf16" ? roundToBfloat16 : (values) => values;
  const qkv = linear(input, atoms, channels, channels * 3, weights.qkv);
  // (atoms, 3, heads, headDim): q for every head, then k, then v.
  const part = (which) => {
    const out = new Float32Array(atoms * channels);
    for (let atom = 0; atom < atoms; atom += 1) {
      const from = atom * channels * 3 + which * channels;
      out.set(qkv.subarray(from, from + channels), atom * channels);
    }
    return out;
  };
  // 🔴 THE RMS NORM IS PER HEAD, NOT PER ATOM, which is what `rms(q)` means on
  // a tensor already reshaped to (atoms, heads, headDim). Normalising the whole
  // row instead conforms in shape and is a different model.
  const rotate = (values) => applyRope(
    rmsNorm(values, atoms * heads, headDim), atoms, heads, headDim, rope.cos, rope.sin);
  // ...narrowed AFTER the rotation, which is where the module does it.
  const query = narrow(rotate(part(0)));
  const key = narrow(rotate(part(1)));
  const value = narrow(part(2));

  // 🔴 RANK AMONG VALID ATOMS - see the note at the top of this file.
  const rank = new Int32Array(atoms);
  let seen = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    seen += valid[atom] !== 0 ? 1 : 0;
    rank[atom] = seen - 1;
  }

  const scale = 1 / Math.sqrt(headDim);
  const context = new Float32Array(atoms * channels);
  const logits = new Float32Array(atoms);
  for (let head = 0; head < heads; head += 1) {
    for (let i = 0; i < atoms; i += 1) {
      const queryBase = (i * heads + head) * headDim;
      let largest = -Infinity;
      for (let j = 0; j < atoms; j += 1) {
        const allowed = i === j
          || (Math.abs(rank[i] - rank[j]) <= halfWindow && valid[i] !== 0 && valid[j] !== 0);
        if (!allowed) { logits[j] = -Infinity; continue; }
        const keyBase = (j * heads + head) * headDim;
        let total = 0;
        for (let d = 0; d < headDim; d += 1) total += query[queryBase + d] * key[keyBase + d];
        logits[j] = total * scale;
        if (logits[j] > largest) largest = logits[j];
      }
      let sum = 0;
      for (let j = 0; j < atoms; j += 1) {
        logits[j] = logits[j] === -Infinity ? 0 : Math.exp(logits[j] - largest);
        sum += logits[j];
      }
      const outBase = (i * heads + head) * headDim;
      for (let j = 0; j < atoms; j += 1) {
        if (logits[j] === 0) continue;
        const weight = logits[j] / sum;
        const valueBase = (j * heads + head) * headDim;
        for (let d = 0; d < headDim; d += 1) {
          context[outBase + d] += weight * value[valueBase + d];
        }
      }
    }
  }

  // ...gated by the BLOCK'S INPUT, not by the attention's output.
  const gate = linear(input, atoms, channels, channels, weights.attnGate);
  for (let atom = 0; atom < atoms; atom += 1) {
    const base = atom * channels;
    const live = valid[atom] !== 0 ? 1 : 0;
    for (let c = 0; c < channels; c += 1) {
      context[base + c] *= live * sigmoid(gate[base + c]);
    }
  }
  return linear(context, atoms, channels, channels, weights.attnOut);
}

/** SwiGLU: one widening projection split in half, gate first. */
function swiglu(input, rows, channels, hidden, up, down) {
  const wide = linear(input, rows, channels, hidden * 2, up);
  const gated = new Float32Array(rows * hidden);
  for (let row = 0; row < rows; row += 1) {
    const wideBase = row * hidden * 2;
    const base = row * hidden;
    for (let i = 0; i < hidden; i += 1) {
      gated[base + i] = silu(wide[wideBase + i]) * wide[wideBase + hidden + i];
    }
  }
  return linear(gated, rows, hidden, channels, down);
}

/**
 * One adaLN-Zero block.
 *
 * 🔴 THE CONDITIONING IS THE BLOCK STACK'S INPUT, HELD FIXED FOR ALL THREE
 * BLOCKS. `atom_stack(c0, c0, ...)` passes c0 as both the running activation
 * and the conditioning; the activation is updated per block and the
 * conditioning is not. Feeding each block its own input instead is a different
 * model that runs.
 */
export function swaBlock(activation, conditioning, atoms, channels, heads, weights,
                         rope, valid, halfWindow, hidden, precision = "bf16") {
  const modulation = linear(
    conditioning.map(silu), atoms, channels, channels * 6, weights.adaln);
  const at = (index, atom, c) => modulation[atom * channels * 6 + index * channels + c];
  const modulated = (offsetIndex, scaleIndex) => {
    const normalised = rmsNorm(activation, atoms, channels);
    const out = new Float32Array(normalised.length);
    for (let atom = 0; atom < atoms; atom += 1) {
      for (let c = 0; c < channels; c += 1) {
        const i = atom * channels + c;
        out[i] = normalised[i] * (1 + at(scaleIndex, atom, c)) + at(offsetIndex, atom, c);
      }
    }
    return out;
  };
  // The six are shift_a, scale_a, gate_a, shift_f, scale_f, gate_f, in order.
  const attended = slidingWindowAttention(
    modulated(0, 1), atoms, channels, heads, weights, rope, valid, halfWindow, precision);
  const afterAttention = new Float32Array(activation.length);
  for (let atom = 0; atom < atoms; atom += 1) {
    for (let c = 0; c < channels; c += 1) {
      const i = atom * channels + c;
      afterAttention[i] = activation[i] + at(2, atom, c) * attended[i];
    }
  }

  const normalisedForFfn = rmsNorm(afterAttention, atoms, channels);
  const forFfn = new Float32Array(normalisedForFfn.length);
  for (let atom = 0; atom < atoms; atom += 1) {
    for (let c = 0; c < channels; c += 1) {
      const i = atom * channels + c;
      forFfn[i] = normalisedForFfn[i] * (1 + at(4, atom, c)) + at(3, atom, c);
    }
  }
  const transitioned = swiglu(
    forFfn, atoms, channels, hidden, weights.ffnUp, weights.ffnDown);
  const out = new Float32Array(activation.length);
  for (let atom = 0; atom < atoms; atom += 1) {
    for (let c = 0; c < channels; c += 1) {
      const i = atom * channels + c;
      out[i] = afterAttention[i] + at(5, atom, c) * transitioned[i];
    }
  }
  return out;
}

/** The 389 per-atom features, one-hots masked by the atom mask. */
export function atomFeatures(features, atoms) {
  const { refPos, refCharge, refElement, refAtomNameChars, mask } = features;
  // 🔴 INDICES, NOT ONE-HOTS, AND THE LENGTH IS THE ONLY THING THAT SAYS SO.
  // The MODEL is handed `ref_element` already one-hot at (atoms, 128) and
  // `ref_atom_name_chars` at (atoms, 4, 64); the FEATURISER produces them as
  // indices at (atoms,) and (atoms, 4). Both arrive here as a flat typed array,
  // and passing the one-hot makes this read its first `atoms` entries - all 0
  // or 1 - as element indices. Every shape conforms, nothing throws, and the
  // encoder's output comes back at relRMS 0.89 with corr 0.72, which reads
  // exactly like a wrong convention somewhere in three SWA blocks.
  if (refElement.length !== atoms) {
    throw new Error(`ref_element has ${refElement.length} entries for ${atoms} atoms; `
      + "this wants INDICES, not the one-hot the model is handed");
  }
  if (refAtomNameChars.length !== atoms * NAME_LENGTH) {
    throw new Error(`ref_atom_name_chars has ${refAtomNameChars.length} entries for `
      + `${atoms} atoms; this wants ${atoms * NAME_LENGTH} INDICES`);
  }
  const out = new Float32Array(atoms * ATOM_FEATURES);
  const elementBase = 5;
  const charBase = elementBase + MAX_ELEMENT;
  for (let atom = 0; atom < atoms; atom += 1) {
    const base = atom * ATOM_FEATURES;
    const live = mask[atom] !== 0 ? 1 : 0;
    // 🔴 ref_pos AND ref_charge ARE NOT MASKED, AND THE ONE-HOTS ARE. Upstream
    // multiplies only the two one-hot blocks by the mask, and passes the mask
    // itself through as a feature; a uniform masking conforms in shape.
    out[base] = refPos[atom * 3];
    out[base + 1] = refPos[atom * 3 + 1];
    out[base + 2] = refPos[atom * 3 + 2];
    out[base + 3] = refCharge[atom];
    out[base + 4] = mask[atom];
    if (live) {
      out[base + elementBase + refElement[atom]] = 1;
      for (let i = 0; i < NAME_LENGTH; i += 1) {
        out[base + charBase + i * NAME_CHARS + refAtomNameChars[atom * NAME_LENGTH + i]] = 1;
      }
    }
  }
  return out;
}

/** Mean over each token's atoms, weighted by the atom mask. */
export function scatterMean(values, atomToToken, mask, atoms, tokens, channels) {
  const out = new Float32Array(tokens * channels);
  const weight = new Float32Array(tokens);
  for (let atom = 0; atom < atoms; atom += 1) {
    const w = mask[atom];
    if (w === 0) continue;
    const token = atomToToken[atom];
    weight[token] += w;
    const from = atom * channels;
    const to = token * channels;
    for (let c = 0; c < channels; c += 1) out[to + c] += values[from + c] * w;
  }
  for (let token = 0; token < tokens; token += 1) {
    const by = 1 / Math.max(weight[token], 1e-9);
    const base = token * channels;
    for (let c = 0; c < channels; c += 1) out[base + c] *= by;
  }
  return out;
}

/**
 * The whole inputs embedder: features in, one per-token vector out.
 *
 * @returns {{atomState: Float32Array, tokenAct: Float32Array}} the atom
 *   transformer's output and the per-token pooling of it. `s_inputs` is that
 *   pooling concatenated with the residue one-hot, the profile and the
 *   deletion mean, which the caller assembles because only it knows the MSA.
 */
export function inputsEmbedder(features, shape, weights) {
  const { atoms, tokens, channels, heads, blocks, hidden, tokenChannels } = shape;
  const halfWindow = (shape.windowSize ?? 128) >> 1;
  const mask = features.mask;
  const embedded = linear(
    atomFeatures(features, atoms), atoms, ATOM_FEATURES, channels, weights.atomLinear);
  const start = layerNorm(embedded, atoms, channels,
                          weights.atomNormScale, weights.atomNormOffset);
  // 🔴 THE DIFFUSION'S COPY OF THIS ENCODER TAKES THE NOISY COORDINATES, AND
  // THE CONDITIONING DOES NOT MOVE. `q` starts at `c_base` plus a projection of
  // `[r_l | pred_r1]`, six channels, while `c` stays `c_base` - so the noisy
  // structure enters the ACTIVATION and never the conditioning. Adding it to
  // both is the natural-looking symmetry and a different model.
  //
  // 🔴 AND `pred_r1` IS ZEROS WHEN THERE IS NO PREVIOUS PREDICTION, not absent.
  // The projection is 6 channels wide either way; a port that fed it three
  // would read the second half of the matrix at the wrong offset.
  let activation = start;
  if (features.noisyCoordinates !== undefined) {
    const previous = features.previousCoordinates;
    const joined = new Float32Array(atoms * 6);
    for (let atom = 0; atom < atoms; atom += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        joined[atom * 6 + axis] = features.noisyCoordinates[atom * 3 + axis];
        joined[atom * 6 + 3 + axis] = previous === undefined ? 0
          : previous[atom * 3 + axis];
      }
    }
    const projected = linear(joined, atoms, 6, channels, weights.coordsLinear);
    activation = new Float32Array(start.length);
    for (let i = 0; i < start.length; i += 1) activation[i] = start[i] + projected[i];
  }
  const rope = buildRope(features.refPos, features.refSpaceUid, atoms, channels / heads,
                         shape.rope);
  let state = activation;
  for (let block = 0; block < blocks; block += 1) {
    state = swaBlock(state, start, atoms, channels, heads, weights.blocks[block],
                     rope, mask, halfWindow, hidden, shape.attentionPrecision ?? "f32");
  }
  const projected = linear(state, atoms, channels, tokenChannels, weights.atomToToken);
  for (let i = 0; i < projected.length; i += 1) {
    if (projected[i] < 0) projected[i] = 0;
  }
  // 🔴 THE SCATTER INDEX IS MASKED, so a padded atom lands on token 0 rather
  // than wherever its stale index pointed - and then contributes nothing,
  // because its weight is its mask.
  const index = new Int32Array(atoms);
  for (let atom = 0; atom < atoms; atom += 1) {
    index[atom] = mask[atom] !== 0 ? features.atomToToken[atom] : 0;
  }
  return {
    // 🔴 `atomState` IS THE SKIP THE DECODER ADDS TO, and it is the
    // transformer's OUTPUT rather than its input. `conditioning` is `c_base`,
    // which the decoder reuses unchanged, and the rope table is shared too - so
    // the decoder builds none of the three for itself.
    atomState: state,
    conditioning: start,
    rope,
    tokenAct: scatterMean(projected, index, mask, atoms, tokens, tokenChannels),
  };
}

/**
 * The atom decoder: a token representation back onto atoms, and out as a
 * coordinate update.
 *
 *     q = q_skip + gather(token_to_atom(a))
 *     q = 3 x swaBlock(q, conditioned on c_skip)
 *     r = output_linear(norm(q))
 *
 * 🔴 IT REUSES THE ENCODER'S SKIP, CONDITIONING AND ROPE TABLE. `q_l`, `c_l`
 * and the attention parameters all come back from the encoder; rebuilding any
 * of them here would be the same arithmetic done twice and, for the rope, a
 * second chance to get the bfloat16 table wrong.
 */
export function atomDecoder(tokenAct, skip, conditioning, rope, features, shape, weights) {
  const { atoms, tokens, channels, heads, blocks, hidden, tokenChannels } = shape;
  const halfWindow = (shape.windowSize ?? 128) >> 1;
  const mask = features.mask;
  const perToken = linear(tokenAct, tokens, tokenChannels, channels, weights.tokenToAtom);
  const state0 = new Float32Array(atoms * channels);
  for (let atom = 0; atom < atoms; atom += 1) {
    const from = features.atomToToken[atom] * channels;
    const to = atom * channels;
    for (let c = 0; c < channels; c += 1) state0[to + c] = skip[to + c] + perToken[from + c];
  }
  let state = state0;
  for (let block = 0; block < blocks; block += 1) {
    state = swaBlock(state, conditioning, atoms, channels, heads, weights.blocks[block],
                     rope, mask, halfWindow, hidden, shape.attentionPrecision ?? "f32");
  }
  const normalised = layerNorm(state, atoms, channels,
                               weights.normScale, weights.normOffset);
  return linear(normalised, atoms, channels, 3, weights.outputLinear);
}
