/**
 * ESMFold2's inputs embedder and atom decoder in WebGL2: a sliding-window atom
 * transformer with 3D RoPE.
 *
 *     c0 = LayerNorm(atomFeatures @ atom_linear)
 *     q  = 3 x swaBlock(c0, conditioned on c0)
 *     a  = scatterMean(relu(q @ atom_to_token), atom_to_token_index)
 *
 * 🔴 THIS IS THE ONE PRIMITIVE WITH NO AF3 ANALOGUE, AND REUSING AF3's ATOM
 * ENCODER WOULD HAVE BEEN THE OBVIOUS WRONG MOVE. AF3 runs 32-query/128-key
 * windowed attention biased by a pair representation; this runs plain
 * sliding-window self-attention whose only positional signal is a rotary
 * embedding built from the REFERENCE CONFORMER's coordinates. "Nothing about the
 * shapes says so - both are 'an atom transformer' at 128 channels."
 *
 * 🔴 AND THE WINDOW IS OVER RANK AMONG VALID ATOMS, NOT OVER RAW INDEX. Two
 * atoms 64 apart in the array can be adjacent in rank if the atoms between them
 * are padding. The diagonal is always allowed, so a masked atom still attends to
 * itself and the softmax has something to normalise.
 *
 * 🔴 AND IT IS DENSE UNLESS A CALLER ASKS FOR A WINDOW. The reference's
 * `SWA3DRoPEAttention` defaults to dense and nothing on the fold path sets
 * otherwise, so `swa_window_size: 128` is in the config and unused - "a window
 * of 128 was this port's own reading... it cost 2.5 pLDDT". A window wider than
 * the molecule IS dense.
 *
 * 🔴 AND EVERY NORM HERE IS AFFINE-FREE RMS, WITH torch's eps AND NOT 1e-5.
 * `F.rms_norm(eps=None)` uses `finfo(float32).eps`, 1.19e-7 - about a hundred
 * times smaller than the LayerNorm epsilon used everywhere else in this
 * repository. Only `atom_norm` is a real LayerNorm with a scale and an offset.
 *
 * 🔴 AND THE RMS NORM INSIDE THE ATTENTION IS PER HEAD, NOT PER ATOM, which is
 * what `rms(q)` means on a tensor already reshaped to (atoms, heads, headDim).
 * "Normalising the whole row instead conforms in shape and is a different
 * model."
 *
 * 🔴 AND THE ROTARY TABLE IS STORED IN bfloat16 IN A float32 MODEL, which is
 * worth 2.4e-3 through the attention: "cos alone is 1.05e-3 away from the exact
 * one... Three days of a residual that looks exactly like a convention bug, is
 * not one, and does not move when every convention is swept, is what this line
 * costs to omit." The table is built on the HOST here for that reason - the
 * rounding is round-to-nearest-even on a known bit pattern, and the table is
 * `atoms * headDim / 2` numbers either way.
 *
 * 🔴 AND THE TABLE IS TILED `[c | c]`, NOT INTERLEAVED, because it pairs with a
 * rotate_half that SPLITS INTO HALVES: element i pairs with i + half. "
 * Interleaving instead is the same shapes and reads corr 0.88."
 *
 * 🔴 AND THE CONDITIONING IS THE BLOCK STACK'S INPUT, HELD FIXED FOR ALL THREE
 * BLOCKS: `atom_stack(c0, c0, ...)` updates the activation per block and not the
 * conditioning. "Feeding each block its own input instead is a different model
 * that runs."
 *
 * 🔴 AND THE ATTENTION IS GATED BY THE BLOCK'S INPUT, not by its own output.
 *
 * 🔴 AND THE DIFFUSION'S COPY TAKES THE NOISY COORDINATES INTO THE ACTIVATION
 * AND NEVER INTO THE CONDITIONING, with `pred_r1` ZEROS rather than absent when
 * there is no previous prediction - "the projection is 6 channels wide either
 * way; a port that fed it three would read the second half of the matrix at the
 * wrong offset".
 */
import { ONE_TARGET, PRELUDE } from "../runtime/execution.js";
import { layerNorm } from "../kernels/layer-norm.js";
import { linear } from "../kernels/linear.js";
import { relu } from "../runtime/elementwise.js";
// The [q | k | v] slice and the blocked SwiGLU are the ESM-C tower's, which is
// this model's own language model - one spelling of each for both halves.
import { qkvSliceSource, swiGluSource } from "../esmc/tower.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/** One-hot widths, from ESMFold2's own constants. */
export const MAX_ELEMENT = 128;
export const NAME_CHARS = 64;
export const NAME_LENGTH = 4;
/** 3 + charge + mask + elements + name characters = 389. */
export const ATOM_FEATURES = 3 + 1 + 1 + MAX_ELEMENT + NAME_LENGTH * NAME_CHARS;
/** torch's `F.rms_norm(eps=None)`, NOT this repository's 1e-5. */
export const RMS_EPSILON = 1.1920928955078125e-7;

/** `1 / base ** (i / n)` - ESMFold2's spacing, which divides by n and not n-1. */
export function ropeInverseFrequencies(pairs, base) {
  const out = new Float32Array(pairs);
  for (let i = 0; i < pairs; i += 1) out[i] = 1 / Math.pow(base, i / pairs);
  return out;
}

const bfloat16 = (() => {
  const floats = new Float32Array(1);
  const bits = new Uint32Array(floats.buffer);
  return (value) => {
    floats[0] = value;
    const word = bits[0];
    const rounded = (word + 0x7fff + ((word >>> 16) & 1)) & 0xffff0000;
    bits[0] = rounded;
    return floats[0];
  };
})();

/** Round to bfloat16 and back, round-to-nearest-even. */
export function roundToBfloat16(values) {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) out[i] = bfloat16(values[i]);
  return out;
}

/**
 * The rotary table: three spatial axes plus the reference-space uid, on the
 * host and in bfloat16.
 *
 * 🔴 `3 * spatialPairs + uidPairs` MUST FILL `headDim / 2` EXACTLY - here
 * 3 x 2 + 10 = 16 = 32 / 2. "The upstream code zero-pads a short table rather
 * than failing, so a wrong pair count is a silently smaller rotation."
 */
export function buildRope(refPos, spaceUid, atoms, headDim, options = {}) {
  const {
    spatialPairs = 2, uidPairs = 10, spatialBase = 20, uidBase = 10000,
    precision = "bf16",
  } = options;
  const half = headDim >> 1;
  const spatial = ropeInverseFrequencies(spatialPairs, spatialBase);
  const uid = ropeInverseFrequencies(uidPairs, uidBase);
  const cos = new Float32Array(atoms * half);
  const sin = new Float32Array(atoms * half);
  for (let atom = 0; atom < atoms; atom += 1) {
    const base = atom * half;
    let at = 0;
    // ...axis-major: x's pairs, then y's, then z's.
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
    for (; at < half; at += 1) { cos[base + at] = 1; sin[base + at] = 0; }
  }
  return precision === "bf16"
    ? { cos: roundToBfloat16(cos), sin: roundToBfloat16(sin) }
    : { cos, sin };
}

/** The 389 per-atom features, one-hots masked by the atom mask. */
export const atomFeaturesSource = (atoms) => `${PRELUDE}${ONE_TARGET}
const int WIDTH = ${ATOM_FEATURES};
const int ELEMENT_BASE = 5;
const int CHAR_BASE = ${5 + MAX_ELEMENT};
const int NAME_CHARS = ${NAME_CHARS};
const int NAME_LENGTH = ${NAME_LENGTH};
const int ELEMENTS = ${atoms * ATOM_FEATURES};
uniform sampler2D posTex;
uniform sampler2D chargeTex;
uniform sampler2D maskTex;
uniform sampler2D elementTex;
uniform sampler2D nameTex;

float at(int element) {
  int atom = element / WIDTH;
  int c = element - atom * WIDTH;
  // 🔴 ref_pos AND ref_charge ARE NOT MASKED, AND THE ONE-HOTS ARE. Upstream
  // multiplies only the two one-hot blocks by the mask and passes the mask
  // itself through as a feature; a uniform masking conforms in shape.
  if (c < 3) return fetch1(posTex, atom * 3 + c);
  if (c == 3) return fetch1(chargeTex, atom);
  if (c == 4) return fetch1(maskTex, atom);
  if (!(fetch1(maskTex, atom) != 0.0)) return 0.0;
  if (c < CHAR_BASE) {
    return int(fetch1(elementTex, atom)) == c - ELEMENT_BASE ? 1.0 : 0.0;
  }
  int offset = c - CHAR_BASE;
  int slot = offset / NAME_CHARS;
  int code = offset - slot * NAME_CHARS;
  if (slot >= NAME_LENGTH) return 0.0;
  return int(fetch1(nameTex, atom * NAME_LENGTH + slot)) == code ? 1.0 : 0.0;
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

/** Affine-free RMS over `channels`, at torch's epsilon. */
export const rmsNormSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int CHANNELS = ${channels};
const int ELEMENTS = ${rows * channels};
const float EPSILON = ${RMS_EPSILON};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;
  float total = 0.0;
  for (int i = 0; i < C4; ++i) {
    vec4 v = fetch4(xTex, row * C4 + i);
    total += dot(v, v);
  }
  result = fetch4(xTex, texel) * inversesqrt(total / float(CHANNELS) + EPSILON);
}`;

/**
 * Rotate per head against a tiled `[c | c]` table.
 *
 * The fragment owns four channels of one head and they are all on one side of
 * the split, because `headDim / 2` is a multiple of four for every shape here -
 * which the caller asserts.
 */
export const ropeApplySource = (atoms, heads, headDim) => `${PRELUDE}${ONE_TARGET}
const int HEAD_DIM = ${headDim};
const int HALF = ${headDim >> 1};
const int HEADS = ${heads};
const int ELEMENTS = ${atoms * heads * headDim};
uniform sampler2D xTex;
uniform sampler2D cosTex;
uniform sampler2D sinTex;

float at(int element) {
  int d = element % HEAD_DIM;
  int atom = element / (HEAD_DIM * HEADS);
  bool low = d < HALF;
  int i = low ? d : d - HALF;
  float c = fetch1(cosTex, atom * HALF + i);
  float s = fetch1(sinTex, atom * HALF + i);
  float a = fetch1(xTex, element + (low ? 0 : -HALF));
  float b = fetch1(xTex, element + (low ? HALF : 0));
  return low ? a * c - b * s : b * c + a * s;
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

/** Round to bfloat16 on the device, for the arm that matches the native model. */
export const narrowSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;

float narrow(float value) {
  uint word = floatBitsToUint(value);
  uint rounded = (word + 0x7fffu + ((word >> 16) & 1u)) & 0xffff0000u;
  return uintBitsToFloat(rounded);
}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 x = fetch4(xTex, texel);
  result = vec4(narrow(x.x), narrow(x.y), narrow(x.z), narrow(x.w));
}`;

/**
 * Sliding-window self-attention over atoms, by RANK among valid ones.
 *
 * The rank is a prefix sum over the mask, which the host supplies as a tensor
 * because it is one scan of `atoms` numbers and a fragment that recomputed it
 * would sweep the mask once per output element.
 */
export const attentionSource = (atoms, heads, headDim, halfWindow) => `${PRELUDE}${ONE_TARGET}
const int ATOMS = ${atoms};
const int HEADS = ${heads};
const int HEAD_DIM = ${headDim};
const int WIDTH = ${heads * headDim};
const int HALF_WINDOW = ${halfWindow};
const int ELEMENTS = ${atoms * heads * headDim};
const float SCALE = ${(1 / Math.sqrt(headDim)).toFixed(10)};
uniform sampler2D qTex;
uniform sampler2D kTex;
uniform sampler2D vTex;
uniform sampler2D validTex;
uniform sampler2D rankTex;

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  int head = (base % WIDTH) / HEAD_DIM;
  int i = base / WIDTH;
  int qBase = (i * HEADS + head) * HEAD_DIM;
  float validI = fetch1(validTex, i);
  int rankI = int(fetch1(rankTex, i));

  float largest = -1e30;
  for (int j = 0; j < ATOMS; ++j) {
    bool allowed = i == j
      || (abs(rankI - int(fetch1(rankTex, j))) <= HALF_WINDOW
        && validI != 0.0 && fetch1(validTex, j) != 0.0);
    if (!allowed) continue;
    int kBase = (j * HEADS + head) * HEAD_DIM;
    float total = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      total += fetch1(qTex, qBase + d) * fetch1(kTex, kBase + d);
    }
    largest = max(largest, total * SCALE);
  }
  float sum = 0.0;
  vec4 gathered = vec4(0.0);
  for (int j = 0; j < ATOMS; ++j) {
    bool allowed = i == j
      || (abs(rankI - int(fetch1(rankTex, j))) <= HALF_WINDOW
        && validI != 0.0 && fetch1(validTex, j) != 0.0);
    if (!allowed) continue;
    int kBase = (j * HEADS + head) * HEAD_DIM;
    float total = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      total += fetch1(qTex, qBase + d) * fetch1(kTex, kBase + d);
    }
    float weight = exp(total * SCALE - largest);
    sum += weight;
    gathered += weight * fetch4(vTex, (kBase + base % HEAD_DIM) / 4);
  }
  result = gathered / sum;
}`;

/** `out = context * valid * sigmoid(gate)`, the gate being of the block's INPUT. */
export const gateSource = (atoms, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${atoms * channels};
uniform sampler2D contextTex;
uniform sampler2D gateTex;
uniform sampler2D validTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int atom = texel / C4;
  float live = fetch1(validTex, atom) != 0.0 ? 1.0 : 0.0;
  vec4 gate = fetch4(gateTex, texel);
  result = fetch4(contextTex, texel) * live / (vec4(1.0) + exp(-gate));
}`;

/** `out = silu(x)`, the modulation's input. */
export const siluSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 x = fetch4(xTex, texel);
  result = x / (vec4(1.0) + exp(-x));
}`;

/**
 * `out = rms(x) * (1 + modulation[scaleIndex]) + modulation[offsetIndex]`.
 *
 * The six slices are shift_a, scale_a, gate_a, shift_f, scale_f, gate_f, in
 * order, and this reads two of them by index.
 */
export const modulateSource = (atoms, channels, offsetIndex, scaleIndex) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${atoms * channels};
const int SIX = ${6 * (channels / 4)};
const int OFFSET = ${offsetIndex * (channels / 4)};
const int SCALE = ${scaleIndex * (channels / 4)};
uniform sampler2D normalisedTex;
uniform sampler2D modulationTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int atom = texel / C4;
  vec4 scale = fetch4(modulationTex, atom * SIX + SCALE + group);
  vec4 shift = fetch4(modulationTex, atom * SIX + OFFSET + group);
  result = fetch4(normalisedTex, texel) * (vec4(1.0) + scale) + shift;
}`;

/** `out = x + modulation[gateIndex] * branch`, the adaLN-Zero residual. */
export const gatedResidualSource = (atoms, channels, gateIndex) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${atoms * channels};
const int SIX = ${6 * (channels / 4)};
const int GATE = ${gateIndex * (channels / 4)};
uniform sampler2D xTex;
uniform sampler2D branchTex;
uniform sampler2D modulationTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int atom = texel / C4;
  vec4 gate = fetch4(modulationTex, atom * SIX + GATE + group);
  result = fetch4(xTex, texel) + gate * fetch4(branchTex, texel);
}`;

/** `out[token][c] = sum over the token's atoms of values * mask, over the mask. */
export const scatterMeanSource = (atoms, tokens, channels) => `${PRELUDE}${ONE_TARGET}
const int ATOMS = ${atoms};
const int C4 = ${channels / 4};
const int ELEMENTS = ${tokens * channels};
uniform sampler2D valuesTex;
uniform sampler2D indexTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int token = texel / C4;
  vec4 total = vec4(0.0);
  float weight = 0.0;
  for (int atom = 0; atom < ATOMS; ++atom) {
    float w = fetch1(maskTex, atom);
    if (w == 0.0) continue;
    if (int(fetch1(indexTex, atom)) != token) continue;
    weight += w;
    total += w * fetch4(valuesTex, atom * C4 + group);
  }
  result = total / max(weight, 1e-9);
}`;

/** `out[atom][c] = skip[atom][c] + perToken[index[atom]][c]`. */
export const broadcastToAtomsSource = (atoms, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${atoms * channels};
uniform sampler2D skipTex;
uniform sampler2D perTokenTex;
uniform sampler2D indexTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int atom = texel / C4;
  int token = int(fetch1(indexTex, atom));
  result = fetch4(skipTex, texel) + fetch4(perTokenTex, token * C4 + group);
}`;

/** `[noisy | previous]`, six channels: zeros where there is no previous. */
export const joinCoordinatesSource = (atoms, hasPrevious) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${atoms * 6};
uniform sampler2D noisyTex;
${hasPrevious ? "uniform sampler2D previousTex;" : ""}

float at(int element) {
  int atom = element / 6;
  int c = element - atom * 6;
  if (c < 3) return fetch1(noisyTex, atom * 3 + c);
${hasPrevious ? "  return fetch1(previousTex, atom * 3 + c - 3);" : "  return 0.0;"}
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

/** The prefix sum over the mask: an atom's rank among the valid ones. */
export function rankAmongValid(mask) {
  const out = new Float32Array(mask.length);
  let seen = 0;
  for (let atom = 0; atom < mask.length; atom += 1) {
    seen += mask[atom] !== 0 ? 1 : 0;
    out[atom] = seen - 1;
  }
  return out;
}

/**
 * Sliding-window self-attention, with its projections and its gate.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, atoms: number, channels: number, heads: number,
 *   weights: any, rope: {cos: Tensor, sin: Tensor}, valid: Tensor, rank: Tensor,
 *   halfWindow: number, precision?: "f32" | "bf16"}} call
 * @returns {Tensor} `atoms * channels`
 */
export function slidingWindowAttention(execution, call) {
  const {
    x, atoms, channels, heads, weights: w, rope, valid, rank, halfWindow,
    precision = "f32",
  } = call;
  const headDim = channels / heads;
  if ((headDim >> 1) % 4 !== 0) {
    throw new RangeError(
      `esmfold2 atom attention: head_dim ${headDim} - the RoPE kernel owns four`
      + " channels a fragment and needs both halves aligned to four");
  }
  const projected = linear(execution, {
    x, weight: w.qkv, rows: atoms,
    inChannels: channels, outChannels: 3 * channels, weightLayout: "in-major",
  });
  const narrow = (tensor) => {
    if (precision !== "bf16") return tensor;
    const out = execution.allocate(tensor.elements, "ef2.atom.narrowed");
    execution.dispatch({
      source: narrowSource(tensor.elements),
      output: out,
      inputs: { xTex: tensor },
      label: `ef2.atom.bf16:${tensor.elements}`,
    });
    execution.release(tensor);
    return out;
  };
  const part = (which, label) => {
    const out = execution.allocate(atoms * channels, `ef2.atom.${label}`);
    execution.dispatch({
      source: qkvSliceSource(atoms, channels, which),
      output: out,
      inputs: { projectedTex: projected },
      label: `ef2.atom.qkv:${atoms}x${channels}:${label}`,
    });
    return out;
  };
  // The RMS norm is PER HEAD, then the rotation, then - where the native model
  // narrows - bfloat16.
  const prepare = (which, label) => {
    const raw = part(which, label);
    const normalised = execution.allocate(atoms * channels, `ef2.atom.${label}-rms`);
    execution.dispatch({
      source: rmsNormSource(atoms * heads, headDim),
      output: normalised,
      inputs: { xTex: raw },
      label: `ef2.atom.rms:${atoms * heads}x${headDim}`,
    });
    execution.release(raw);
    const rotated = execution.allocate(atoms * channels, `ef2.atom.${label}-rope`);
    execution.dispatch({
      source: ropeApplySource(atoms, heads, headDim),
      output: rotated,
      inputs: { xTex: normalised, cosTex: rope.cos, sinTex: rope.sin },
      label: `ef2.atom.rope:${atoms}x${heads}x${headDim}`,
    });
    execution.release(normalised);
    return narrow(rotated);
  };
  const query = prepare(0, "q");
  const key = prepare(1, "k");
  const value = narrow(part(2, "v"));
  execution.release(projected);

  const context = execution.allocate(atoms * channels, "ef2.atom.context");
  execution.dispatch({
    source: attentionSource(atoms, heads, headDim, halfWindow),
    output: context,
    inputs: {
      qTex: query, kTex: key, vTex: value, validTex: valid, rankTex: rank,
    },
    label: `ef2.atom.attend:${atoms}x${heads}x${headDim}:w${halfWindow}`,
  });
  for (const tensor of [query, key, value]) execution.release(tensor);

  // ...gated by the BLOCK'S INPUT, not by the attention's output.
  const gate = linear(execution, {
    x, weight: w.attnGate, rows: atoms,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  const gated = execution.allocate(atoms * channels, "ef2.atom.gated");
  execution.dispatch({
    source: gateSource(atoms, channels),
    output: gated,
    inputs: { contextTex: context, gateTex: gate, validTex: valid },
    label: `ef2.atom.gate:${atoms}x${channels}`,
  });
  execution.release(context);
  execution.release(gate);
  const out = linear(execution, {
    x: gated, weight: w.attnOut, rows: atoms,
    inChannels: channels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(gated);
  return out;
}

/**
 * One adaLN-Zero block.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{activation: Tensor, modulation: Tensor, atoms: number,
 *   channels: number, heads: number, hidden: number, weights: any,
 *   rope: {cos: Tensor, sin: Tensor}, valid: Tensor, rank: Tensor,
 *   halfWindow: number, precision?: "f32" | "bf16"}} call
 *   `modulation` is `silu(conditioning) @ adaln`, computed ONCE for the stack
 *   because the conditioning is the stack's input and does not move.
 * @returns {Tensor} `atoms * channels`
 */
export function swaBlock(execution, call) {
  const {
    activation, modulation, atoms, channels, heads, hidden, weights: w,
    rope, valid, rank, halfWindow, precision = "f32",
  } = call;
  const modulated = (offsetIndex, scaleIndex, label) => {
    const normalised = execution.allocate(atoms * channels, `ef2.atom.${label}-rms`);
    execution.dispatch({
      source: rmsNormSource(atoms, channels),
      output: normalised,
      inputs: { xTex: activation },
      label: `ef2.atom.rms:${atoms}x${channels}`,
    });
    const out = execution.allocate(atoms * channels, `ef2.atom.${label}`);
    execution.dispatch({
      source: modulateSource(atoms, channels, offsetIndex, scaleIndex),
      output: out,
      inputs: { normalisedTex: normalised, modulationTex: modulation },
      label: `ef2.atom.modulate:${atoms}x${channels}:${offsetIndex}/${scaleIndex}`,
    });
    execution.release(normalised);
    return out;
  };
  const forAttention = modulated(0, 1, "attn-in");
  const attended = slidingWindowAttention(execution, {
    x: forAttention, atoms, channels, heads, weights: w, rope, valid, rank,
    halfWindow, precision,
  });
  execution.release(forAttention);
  const afterAttention = execution.allocate(atoms * channels, "ef2.atom.after-attention");
  execution.dispatch({
    source: gatedResidualSource(atoms, channels, 2),
    output: afterAttention,
    inputs: { xTex: activation, branchTex: attended, modulationTex: modulation },
    label: `ef2.atom.attn-residual:${atoms}x${channels}`,
  });
  execution.release(attended);

  // The FFN branch reads the POST-ATTENTION activation, through its own two
  // modulation slices.
  const normalisedForFfn = execution.allocate(atoms * channels, "ef2.atom.ffn-rms");
  execution.dispatch({
    source: rmsNormSource(atoms, channels),
    output: normalisedForFfn,
    inputs: { xTex: afterAttention },
    label: `ef2.atom.rms:${atoms}x${channels}`,
  });
  const forFfn = execution.allocate(atoms * channels, "ef2.atom.ffn-in");
  execution.dispatch({
    source: modulateSource(atoms, channels, 3, 4),
    output: forFfn,
    inputs: { normalisedTex: normalisedForFfn, modulationTex: modulation },
    label: `ef2.atom.modulate:${atoms}x${channels}:3/4`,
  });
  execution.release(normalisedForFfn);
  const wide = linear(execution, {
    x: forFfn, weight: w.ffnUp, rows: atoms,
    inChannels: channels, outChannels: 2 * hidden, weightLayout: "in-major",
  });
  execution.release(forFfn);
  const gated = execution.allocate(atoms * hidden, "ef2.atom.swiglu");
  execution.dispatch({
    source: swiGluSource(atoms, hidden),
    output: gated,
    inputs: { wideTex: wide },
    label: `ef2.atom.swiglu:${atoms}x${hidden}`,
  });
  execution.release(wide);
  const transitioned = linear(execution, {
    x: gated, weight: w.ffnDown, rows: atoms,
    inChannels: hidden, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(gated);
  const out = execution.allocate(atoms * channels, "ef2.atom.block");
  execution.dispatch({
    source: gatedResidualSource(atoms, channels, 5),
    output: out,
    inputs: { xTex: afterAttention, branchTex: transitioned, modulationTex: modulation },
    label: `ef2.atom.ffn-residual:${atoms}x${channels}`,
  });
  execution.release(afterAttention);
  execution.release(transitioned);
  return out;
}

/** `silu(conditioning) @ adaln`, the six slices a block reads. */
function modulationFor(execution, { conditioning, atoms, channels, weight }) {
  const activated = execution.allocate(atoms * channels, "ef2.atom.silu");
  execution.dispatch({
    source: siluSource(atoms * channels),
    output: activated,
    inputs: { xTex: conditioning },
    label: `ef2.atom.silu:${atoms}x${channels}`,
  });
  const out = linear(execution, {
    x: activated, weight, rows: atoms,
    inChannels: channels, outChannels: 6 * channels, weightLayout: "in-major",
  });
  execution.release(activated);
  return out;
}

/**
 * The whole inputs embedder: features in, one per-token vector out.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{features: {refPos: Tensor, refCharge: Tensor, mask: Tensor,
 *     refElement: Tensor, refAtomNameChars: Tensor, atomToToken: Tensor,
 *     noisyCoordinates?: Tensor, previousCoordinates?: Tensor},
 *   host: {refPos: Float32Array, refSpaceUid: Float32Array, mask: Float32Array},
 *   shape: any, weights: any, rope?: {cos: Tensor, sin: Tensor}, rank?: Tensor,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {{atomState: Tensor, conditioning: Tensor,
 *   rope: {cos: Tensor, sin: Tensor}, tokenAct: Tensor, valid: Tensor,
 *   rank: Tensor, index: Tensor}} the caller owns all of them - the decoder
 *   takes the skip, the conditioning, the rank and the rope table back
 */
export function inputsEmbedder(execution, call) {
  const { features, shape, weights: w } = call;
  const { atoms, tokens, channels, heads, blocks, hidden, tokenChannels } = shape;
  const halfWindow = (shape.windowSize ?? atoms * 2) >> 1;
  const precision = shape.attentionPrecision ?? "f32";

  const featureTensor = execution.allocate(atoms * ATOM_FEATURES, "ef2.atom.features");
  execution.dispatch({
    source: atomFeaturesSource(atoms),
    output: featureTensor,
    inputs: {
      posTex: features.refPos, chargeTex: features.refCharge, maskTex: features.mask,
      elementTex: features.refElement, nameTex: features.refAtomNameChars,
    },
    label: `ef2.atom.features:${atoms}x${ATOM_FEATURES}`,
  });
  call.onStage?.("atom-features", featureTensor);
  const embedded = linear(execution, {
    x: featureTensor, weight: w.atomLinear, rows: atoms,
    inChannels: ATOM_FEATURES, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(featureTensor);
  const start = layerNorm(execution, {
    x: embedded, weight: w.atomNormScale, bias: w.atomNormOffset,
    rows: atoms, channels, epsilon: shape.epsilon ?? 1e-5,
  });
  execution.release(embedded);
  call.onStage?.("c0", start);

  // 🔴 THE NOISY STRUCTURE ENTERS THE ACTIVATION AND NEVER THE CONDITIONING.
  let activation = start;
  if (features.noisyCoordinates !== undefined) {
    const joined = execution.allocate(atoms * 6, "ef2.atom.coordinates");
    execution.dispatch({
      source: joinCoordinatesSource(atoms, features.previousCoordinates !== undefined),
      output: joined,
      inputs: {
        noisyTex: features.noisyCoordinates,
        ...(features.previousCoordinates === undefined
          ? {} : { previousTex: features.previousCoordinates }),
      },
      label: `ef2.atom.coordinates:${atoms}`,
    });
    const projected = linear(execution, {
      x: joined, weight: w.coordsLinear, rows: atoms,
      inChannels: 6, outChannels: channels, weightLayout: "in-major",
    });
    execution.release(joined);
    activation = execution.allocate(atoms * channels, "ef2.atom.activation");
    execution.dispatch({
      source: sumSource(atoms * channels),
      output: activation,
      inputs: { aTex: start, bTex: projected },
      label: `ef2.atom.coords-add:${atoms}x${channels}`,
    });
    execution.release(projected);
  }

  // 🔴 THE ROPE TABLE AND THE RANK COME FROM THE HOST, ONCE. The table is
  // bfloat16-rounded - see the note at the top - and the rank is a prefix sum
  // over the mask; a fragment that recomputed either would sweep the mask once
  // per output element. `call.rope`/`call.rank` let the decoder reuse the
  // encoder's rather than build a second copy.
  const built = call.rope === undefined
    ? buildRope(call.host.refPos, call.host.refSpaceUid, atoms, channels / heads,
      shape.rope)
    : null;
  const rope = call.rope ?? {
    cos: execution.upload(/** @type {any} */ (built).cos, "ef2.atom.rope-cos"),
    sin: execution.upload(/** @type {any} */ (built).sin, "ef2.atom.rope-sin"),
  };
  const valid = features.mask;
  const rank = call.rank ?? execution.upload(
    rankAmongValid(call.host.mask), "ef2.atom.rank");

  let state = activation;
  for (let block = 0; block < blocks; block += 1) {
    // 🔴 EACH BLOCK HAS ITS OWN adaLN PROJECTION, so the modulation is per
    // block even though the CONDITIONING is not - c0 for all three.
    const modulation = modulationFor(execution, {
      conditioning: start, atoms, channels, weight: w.blocks[block].adaln,
    });
    const next = swaBlock(execution, {
      activation: state, modulation, atoms, channels, heads, hidden,
      weights: w.blocks[block], rope, valid, rank, halfWindow, precision,
    });
    execution.release(modulation);
    // `start` is the conditioning and the decoder's skip, so it is never
    // released here; every other intermediate is.
    if (state !== start) execution.release(state);
    state = next;
    call.onStage?.(`block${block}`, state);
  }

  const projected = linear(execution, {
    x: state, weight: w.atomToToken, rows: atoms,
    inChannels: channels, outChannels: tokenChannels, weightLayout: "in-major",
  });
  const rectified = relu(execution, projected);
  execution.release(projected);
  const tokenAct = execution.allocate(tokens * tokenChannels, "ef2.atom.token-act");
  execution.dispatch({
    source: scatterMeanSource(atoms, tokens, tokenChannels),
    output: tokenAct,
    inputs: { valuesTex: rectified, indexTex: features.atomToToken, maskTex: valid },
    label: `ef2.atom.scatter:${atoms}->${tokens}x${tokenChannels}`,
  });
  execution.release(rectified);

  return {
    atomState: state, conditioning: start, rope, rank, tokenAct, valid,
    index: features.atomToToken,
  };
}

/** `out = a + b`. */
export const sumSource = (elements) => `${PRELUDE}${ONE_TARGET}
const int ELEMENTS = ${elements};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(aTex, texel) + fetch4(bTex, texel);
}`;

/**
 * The atom decoder: a token representation back onto atoms, and out as a
 * coordinate update.
 *
 * 🔴 IT REUSES THE ENCODER'S SKIP, CONDITIONING AND ROPE TABLE. "Rebuilding any
 * of them here would be the same arithmetic done twice and, for the rope, a
 * second chance to get the bfloat16 table wrong."
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @returns {Tensor} `atoms * 3`
 */
export function atomDecoder(execution, call) {
  const { tokenAct, skip, conditioning, rope, valid, rank, index, shape, weights: w } = call;
  const { atoms, tokens, channels, heads, blocks, hidden, tokenChannels } = shape;
  const halfWindow = (shape.windowSize ?? atoms * 2) >> 1;
  const precision = shape.attentionPrecision ?? "f32";
  const perToken = linear(execution, {
    x: tokenAct, weight: w.tokenToAtom, rows: tokens,
    inChannels: tokenChannels, outChannels: channels, weightLayout: "in-major",
  });
  let state = execution.allocate(atoms * channels, "ef2.decoder.state");
  execution.dispatch({
    source: broadcastToAtomsSource(atoms, channels),
    output: state,
    inputs: { skipTex: skip, perTokenTex: perToken, indexTex: index },
    label: `ef2.decoder.broadcast:${atoms}x${channels}`,
  });
  execution.release(perToken);
  for (let block = 0; block < blocks; block += 1) {
    const modulation = modulationFor(execution, {
      conditioning, atoms, channels, weight: w.blocks[block].adaln,
    });
    const next = swaBlock(execution, {
      activation: state, modulation, atoms, channels, heads, hidden,
      weights: w.blocks[block], rope, valid, rank, halfWindow, precision,
    });
    execution.release(modulation);
    execution.release(state);
    state = next;
    call.onStage?.(`decoder.block${block}`, state);
  }
  const normalised = layerNorm(execution, {
    x: state, weight: w.normScale, bias: w.normOffset,
    rows: atoms, channels, epsilon: shape.epsilon ?? 1e-5,
  });
  execution.release(state);
  const update = linear(execution, {
    x: normalised, weight: w.outputLinear, rows: atoms,
    inChannels: channels, outChannels: 3, weightLayout: "in-major",
  });
  execution.release(normalised);
  update.label = "ef2.decoder.update";
  return update;
}
