/**
 * ESMFold2's sliding-window atom transformer, on the GPU.
 *
 * 🔴 THIS IS THE ONE STACK WITH NO AF3 KERNEL TO REUSE, AND THE REASON IS THE
 * POSITIONAL SIGNAL RATHER THAN THE SHAPE. AF3's atom encoder is 32-query /
 * 128-key windowed attention biased by a pair representation; this is plain
 * sliding-window self-attention whose only positional signal is a rotary
 * embedding built from the REFERENCE CONFORMER's coordinates. Both are "an
 * atom transformer at 128 channels", so nothing in the shapes says so - see
 * src/esmfold2/atom-encoder-reference.js, which is what this is checked
 * against.
 *
 * 🔴 EVERYTHING THAT IS A PLAIN PROJECTION COMES FROM src/esmc/block-webgpu.js.
 * `createLinearShader` is a tuned tiled GEMM (row tile 8, a lane owning four
 * columns as one vec4, the k loop unrolled by four) and `createSwigluShader`
 * computes `silu(wide[c]) * wide[hidden + c]` out of ONE fused widening matrix,
 * gate first - which is exactly what `ffnUp` at [128, 512] is. Writing a second
 * pair of those would be two more kernels to tune and one more place for the
 * gate half to be read second.
 *
 * What is new here is four shaders:
 *
 *   modulate    rmsNorm(act) * (1 + scale) + shift, reading one of the six
 *               slices of the adaLN modulation
 *   prepare     split the fused qkv, RMS-norm q and k PER HEAD, rotate both by
 *               the 3D rope table, and round all three to bfloat16
 *   attend      the sliding window itself, gated by a projection of its own
 *               input and written straight out
 *   gated       act += modulation[gate] * delta
 *
 * 🔴 THE WINDOW IS OVER RANK AMONG VALID ATOMS AND THE HOST RESOLVES IT. Two
 * atoms 64 apart in the array are adjacent in rank if everything between them
 * is padding, so the allowed set is not `|i - j| <= 64`. Rank is monotonic, so
 * the allowed set IS a contiguous range - and `atomWindows` computes those
 * bounds once on the host rather than making every invocation scan for them.
 * The validity test stays in the shader because a padded atom can sit INSIDE a
 * live range, and the diagonal is allowed unconditionally so that a masked
 * atom still has something for its softmax to normalise.
 */
import { GRID_WIDTH, LANES, createLinearShader, createSwigluShader, linearGrid, swigluGrid }
  from "../esmc/block-webgpu.js";

/** torch's `F.rms_norm(eps=None)`: `finfo(float32).eps`, not this tree's 1e-5. */
export const RMS_EPSILON = 1.1920928955078125e-7;

/** The six adaLN slices, in the order the fused projection writes them. */
export const SHIFT_ATTENTION = 0, SCALE_ATTENTION = 1, GATE_ATTENTION = 2;
export const SHIFT_FFN = 3, SCALE_FFN = 4, GATE_FFN = 5;

/**
 * Round to bfloat16 and back, in WGSL.
 *
 * 🔴 THE ATTENTION DOWNCASTS WHATEVER THE MODEL'S DTYPE IS, and it is a
 * property of the module rather than of how it was loaded:
 * `if q.dtype not in (float16, bfloat16): q, k, v = q.bfloat16(), ...`. An f32
 * port cannot agree with the shipping model below about 2e-4 however right it
 * is, so this is not an accuracy trade taken here - it is the model.
 */
const BFLOAT16 = `
fn bf16(value: f32) -> f32 {
  let word = bitcast<u32>(value);
  return bitcast<f32>((word + 0x7fffu + ((word >> 16u) & 1u)) & 0xffff0000u);
}`;

/**
 * Where each atom's window starts and ends, as ranks resolve it.
 *
 * @returns {Int32Array} two entries an atom: `[start, end)` over atom indices.
 */
export function atomWindows(valid, atoms, halfWindow) {
  const rank = new Int32Array(atoms);
  let seen = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    seen += valid[atom] !== 0 ? 1 : 0;
    rank[atom] = seen - 1;
  }
  const bounds = new Int32Array(atoms * 2);
  for (let atom = 0; atom < atoms; atom += 1) {
    if (valid[atom] === 0) {
      // ...only the diagonal, which is always allowed.
      bounds[atom * 2] = atom;
      bounds[atom * 2 + 1] = atom + 1;
      continue;
    }
    let start = atom, end = atom + 1;
    while (start > 0 && rank[atom] - rank[start - 1] <= halfWindow) start -= 1;
    while (end < atoms && rank[end] - rank[atom] <= halfWindow) end += 1;
    bounds[atom * 2] = start;
    bounds[atom * 2 + 1] = end;
  }
  return bounds;
}

/** The widest `[start, end)` any atom has, which is what the shader stages. */
export function widestWindow(bounds, atoms) {
  let widest = 1;
  for (let atom = 0; atom < atoms; atom += 1) {
    widest = Math.max(widest, bounds[atom * 2 + 1] - bounds[atom * 2]);
  }
  return Math.ceil(widest / 32) * 32;
}

/** `out = silu(input)`, elementwise. The conditioning's, computed once a stack. */
export function createSiluShader(elements) {
  return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${elements}u) { return; }
  let v = input[i];
  output[i] = v / (1.0 + exp(-v));
}`;
}

/**
 * `out = rmsNorm(activation) * (1 + modulation[scale]) + modulation[shift]`.
 *
 * 🔴 THE NORM IS AFFINE-FREE RMS AND ITS EPSILON IS torch's, NOT THIS TREE'S.
 * `F.rms_norm(eps=None)` is 1.19e-7, about a hundred times smaller than the
 * LayerNorm epsilon every other kernel here uses; at 128 channels the
 * difference is small and a 1e-6 bound sees it.
 */
export function createModulateShader({ atoms, channels }, shiftIndex, scaleIndex) {
  return `
@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> modulation: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> squares: array<f32, ${LANES}>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let atom = group.x + group.y * ${GRID_WIDTH}u;
  if (atom >= ${atoms}u) { return; }
  let base = atom * ${channels}u;
  var square = 0.0;
  for (var c = local.x; c < ${channels}u; c += ${LANES}u) {
    let v = activation[base + c];
    square += v * v;
  }
  squares[local.x] = square;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) { squares[local.x] += squares[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse = inverseSqrt(squares[0] / ${channels}.0 + ${RMS_EPSILON});
  let modulationBase = atom * ${channels * 6}u;
  for (var c = local.x; c < ${channels}u; c += ${LANES}u) {
    let scale = modulation[modulationBase + ${scaleIndex}u * ${channels}u + c];
    let shift = modulation[modulationBase + ${shiftIndex}u * ${channels}u + c];
    output[base + c] = activation[base + c] * inverse * (1.0 + scale) + shift;
  }
}`;
}

/**
 * `activation += modulation[gate] * delta`, elementwise per channel.
 *
 * 🔴 THE GATE IS PER CHANNEL, NOT PER ATOM. adaLN-Zero's third slice is a
 * whole channel vector; reading one scalar an atom conforms in shape.
 */
export function createGatedResidualShader({ atoms, channels }, gateIndex) {
  return `
@group(0) @binding(0) var<storage, read> delta: array<f32>;
@group(0) @binding(1) var<storage, read> modulation: array<f32>;
@group(0) @binding(2) var<storage, read_write> activation: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${atoms * channels}u) { return; }
  let atom = i / ${channels}u;
  let c = i % ${channels}u;
  activation[i] += modulation[atom * ${channels * 6}u + ${gateIndex}u * ${channels}u + c] * delta[i];
}`;
}

/**
 * Split the fused qkv, RMS-norm q and k PER HEAD, rotate both, narrow all three.
 *
 * 🔴 THE RMS NORM IS PER HEAD, NOT PER ATOM. `rms(q)` on a tensor already
 * reshaped to (atoms, heads, headDim) normalises each head's 32 channels
 * separately; over the whole 128-channel row instead conforms in shape and is a
 * different model.
 *
 * 🔴 AND THE ROPE TABLE IS TILED `[c | c]`, NOT INTERLEAVED, because it pairs
 * with a rotate_half that splits into halves: element i pairs with i + half.
 * Interleaving is the same shapes and reads corr 0.88 - a plausible tensor.
 *
 * 🔴 AND THE NARROWING HAPPENS AFTER THE ROTATION, which is where the module
 * does it. Rounding before would leave the rotation exact and the operand
 * coarse, which is a different sixth of a millionth.
 */
export function createSwaPrepareShader({ atoms, channels, heads }, precision) {
  const headDim = channels / heads;
  const half = headDim >> 1;
  const narrow = precision === "bf16" ? "bf16" : "";
  return `${precision === "bf16" ? BFLOAT16 : ""}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> cosine: array<f32>;
@group(0) @binding(2) var<storage, read> sine: array<f32>;
@group(0) @binding(3) var<storage, read_write> query: array<f32>;
@group(0) @binding(4) var<storage, read_write> key: array<f32>;
@group(0) @binding(5) var<storage, read_write> value: array<f32>;

var<workgroup> staged: array<f32, ${headDim}>;
var<workgroup> squares: array<f32, ${headDim}>;

@compute @workgroup_size(${headDim})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let slot = group.x + group.y * ${GRID_WIDTH}u;
  let atom = slot / ${heads}u;
  let head = slot % ${heads}u;
  if (atom >= ${atoms}u) { return; }
  let d = local.x;
  let out_base = (atom * ${heads}u + head) * ${headDim}u;
  // (atoms, 3, heads, headDim): every head's q, then every head's k, then v.
  let qkv_base = atom * ${channels * 3}u + head * ${headDim}u;

  value[out_base + d] = ${narrow}(qkv[qkv_base + ${channels * 2}u + d]);

  let cos_base = atom * ${half}u;
  for (var which = 0u; which < 2u; which += 1u) {
    let v = qkv[qkv_base + which * ${channels}u + d];
    squares[d] = v * v;
    workgroupBarrier();
    for (var stride = ${headDim / 2}u; stride > 0u; stride >>= 1u) {
      if (d < stride) { squares[d] += squares[d + stride]; }
      workgroupBarrier();
    }
    staged[d] = v * inverseSqrt(squares[0] / ${headDim}.0 + ${RMS_EPSILON});
    workgroupBarrier();
    // rotate_half: element i pairs with i + half, and the table is tiled.
    let table = select(d - ${half}u, d, d < ${half}u);
    let c = cosine[cos_base + table];
    let s = sine[cos_base + table];
    let rotated = select(staged[d - ${half}u] * s + staged[d] * c,
                         staged[d] * c - staged[d + ${half}u] * s,
                         d < ${half}u);
    if (which == 0u) { query[out_base + d] = ${narrow}(rotated); }
    else { key[out_base + d] = ${narrow}(rotated); }
    workgroupBarrier();
  }
}`;
}

/**
 * The sliding window itself, gated and projected out in one pass.
 *
 * One workgroup an atom, `channels` lanes, so lane `t` owns head `t / headDim`
 * and dimension `t % headDim`. The logits for all heads are computed a key
 * block at a time - 32 keys by `heads` at once - and the weighted sum is then
 * one global read per key per lane, with adjacent lanes on adjacent addresses.
 *
 * 🔴 THE GATE COMES FROM THE ATTENTION'S INPUT, NOT ITS OUTPUT, and it is
 * multiplied by the atom's own mask. Gating the output instead is the natural
 * reading and a different model.
 */
export function createSwaAttendShader({ atoms, channels, heads, window }) {
  const headDim = channels / heads;
  const blocks = window / 32;
  return `
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> bounds: array<i32>;
@group(0) @binding(4) var<storage, read> valid: array<f32>;
@group(0) @binding(5) var<storage, read> gate: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

var<workgroup> logits: array<f32, ${heads * window}>;
var<workgroup> reduce: array<f32, ${channels}>;
var<workgroup> live: array<f32, ${window}>;

@compute @workgroup_size(${channels})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let atom = group.x + group.y * ${GRID_WIDTH}u;
  if (atom >= ${atoms}u) { return; }
  let head = local.x / ${headDim}u;
  let d = local.x % ${headDim}u;
  let start = u32(bounds[atom * 2u]);
  let end = u32(bounds[atom * 2u + 1u]);
  let count = end - start;
  let mine = valid[atom];

  // Which keys are allowed at all: the diagonal always, otherwise both live.
  for (var slot = local.x; slot < ${window}u; slot += ${channels}u) {
    let j = start + slot;
    var allowed = 0.0;
    if (slot < count) {
      allowed = select(mine * valid[j], 1.0, j == atom);
    }
    live[slot] = allowed;
  }
  workgroupBarrier();

  // Logits, a key block of 32 across every head at a time.
  let q_base = (atom * ${heads}u + head) * ${headDim}u;
  for (var block = 0u; block < ${blocks}u; block += 1u) {
    let slot = block * 32u + d % 32u;
    // ...every head's lanes cover 32 keys; the rest of the head's lanes idle.
    if (d < 32u && slot < ${window}u) {
      var total = 0.0;
      if (live[slot] != 0.0) {
        let k_base = ((start + slot) * ${heads}u + head) * ${headDim}u;
        for (var e = 0u; e < ${headDim}u; e += 1u) {
          total += query[q_base + e] * key[k_base + e];
        }
      }
      logits[head * ${window}u + slot] = total * ${(1 / Math.sqrt(headDim)).toFixed(10)};
    }
  }
  workgroupBarrier();

  // Softmax per head, over its own slice.
  var largest = -3.0e38;
  for (var slot = d; slot < ${window}u; slot += ${headDim}u) {
    if (live[slot] != 0.0) { largest = max(largest, logits[head * ${window}u + slot]); }
  }
  reduce[local.x] = largest;
  workgroupBarrier();
  for (var stride = ${headDim / 2}u; stride > 0u; stride >>= 1u) {
    if (d < stride) { reduce[local.x] = max(reduce[local.x], reduce[local.x + stride]); }
    workgroupBarrier();
  }
  let peak = reduce[head * ${headDim}u];
  workgroupBarrier();
  var total = 0.0;
  for (var slot = d; slot < ${window}u; slot += ${headDim}u) {
    let weight = select(0.0, exp(logits[head * ${window}u + slot] - peak), live[slot] != 0.0);
    logits[head * ${window}u + slot] = weight;
    total += weight;
  }
  reduce[local.x] = total;
  workgroupBarrier();
  for (var stride = ${headDim / 2}u; stride > 0u; stride >>= 1u) {
    if (d < stride) { reduce[local.x] += reduce[local.x + stride]; }
    workgroupBarrier();
  }
  let sum = max(reduce[head * ${headDim}u], 1.0e-30);
  workgroupBarrier();

  // The weighted sum: one read a key a lane, adjacent lanes adjacent addresses.
  var context = 0.0;
  for (var slot = 0u; slot < count; slot += 1u) {
    let weight = logits[head * ${window}u + slot];
    if (weight == 0.0) { continue; }
    context += weight * value[((start + slot) * ${heads}u + head) * ${headDim}u + d];
  }
  let at = atom * ${channels}u + local.x;
  let g = gate[at];
  output[at] = (context / sum) * mine / (1.0 + exp(-g));
}`;
}

/** `out[atom] = skip[atom] + perToken[atomToToken[atom]]`, the decoder's start. */
export function createBroadcastShader({ atoms, channels }) {
  return `
@group(0) @binding(0) var<storage, read> skip: array<f32>;
@group(0) @binding(1) var<storage, read> perToken: array<f32>;
@group(0) @binding(2) var<storage, read> atomToToken: array<i32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${atoms * channels}u) { return; }
  let atom = i / ${channels}u;
  let c = i % ${channels}u;
  output[i] = skip[i] + perToken[u32(atomToToken[atom]) * ${channels}u + c];
}`;
}

/**
 * relu, then the mask-weighted mean over each token's atoms.
 *
 * 🔴 THE SCATTER INDEX IS MASKED, so a padded atom lands on token 0 rather than
 * wherever its stale index pointed - and then contributes nothing, because its
 * weight is its mask. One workgroup a TOKEN, walking its own atoms, so there
 * is no atomic and no ordering to be non-deterministic about.
 */
export function createPoolShader({ tokens, channels }, ranges) {
  return `
@group(0) @binding(0) var<storage, read> projected: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> ranges: array<i32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let token = group.x + group.y * ${GRID_WIDTH}u;
  if (token >= ${tokens}u) { return; }
  let start = u32(ranges[token * 2u]);
  let end = u32(ranges[token * 2u + 1u]);
  var weight = 0.0;
  for (var atom = start; atom < end; atom += 1u) { weight += mask[atom]; }
  let by = 1.0 / max(weight, 1.0e-9);
  for (var c = local.x; c < ${channels}u; c += ${LANES}u) {
    var total = 0.0;
    for (var atom = start; atom < end; atom += 1u) {
      let w = mask[atom];
      if (w == 0.0) { continue; }
      total += max(projected[atom * ${channels}u + c], 0.0) * w;
    }
    output[token * ${channels}u + c] = total * by;
  }
}`;
}

/**
 * The atoms belonging to each token, as `[start, end)`.
 *
 * 🔴 IT REQUIRES A TOKEN'S ATOMS TO BE CONTIGUOUS, and it checks rather than
 * assuming: ESMFold2's featurisation packs a residue's heavy atoms in order and
 * pads at the end, so they are - but a pooling kernel that walked a range on a
 * layout that interleaved would silently average the wrong atoms.
 */
export function tokenRanges(atomToToken, mask, atoms, tokens) {
  const ranges = new Int32Array(tokens * 2);
  const first = new Int32Array(tokens).fill(-1);
  const last = new Int32Array(tokens).fill(-1);
  for (let atom = 0; atom < atoms; atom += 1) {
    if (mask[atom] === 0) continue;
    const token = atomToToken[atom];
    if (token < 0 || token >= tokens) {
      throw new Error(`atom ${atom} names token ${token} of ${tokens}`);
    }
    if (first[token] === -1) first[token] = atom;
    last[token] = atom;
  }
  for (let token = 0; token < tokens; token += 1) {
    ranges[token * 2] = first[token] === -1 ? 0 : first[token];
    ranges[token * 2 + 1] = first[token] === -1 ? 0 : last[token] + 1;
  }
  // 🔴 A TOKEN'S ATOMS MUST BE CONTIGUOUS, AND THIS CHECKS RATHER THAN ASSUMES.
  // ESMFold2's featurisation packs a residue's heavy atoms in order and pads at
  // the end, so they are; a kernel walking a range over a layout that
  // interleaved would average the wrong atoms and still return a tensor.
  for (let atom = 0; atom < atoms; atom += 1) {
    if (mask[atom] === 0) continue;
    const token = atomToToken[atom];
    for (let other = 0; other < tokens; other += 1) {
      if (other === token) continue;
      if (atom >= ranges[other * 2] && atom < ranges[other * 2 + 1]) {
        throw new Error(`atom ${atom} of token ${token} falls inside token ${other}'s range; `
          + "this pooling wants a token's atoms contiguous");
      }
    }
  }
  return ranges;
}

/**
 * Compile the whole stack once for a shape.
 *
 * @param shape { atoms, tokens, channels, heads, blocks, hidden, window,
 *                precision, tokenChannels }
 */
export async function compileAtomStack(pipelines, shape) {
  const { atoms, channels, heads, hidden, window } = shape;
  const precision = shape.precision ?? "bf16";
  const base = `esmfold2-atom:${atoms}:${channels}:${heads}:${hidden}:${window}:${precision}`;
  const linear = (label, inner, outer) => pipelines.get(
    `${base}:linear:${label}:${inner}:${outer}`,
    createLinearShader({ rows: atoms, inner, outer }, false));
  const wanted = {
    silu: pipelines.get(`${base}:silu`, createSiluShader(atoms * channels)),
    modulation: linear("adaln", channels, channels * 6),
    modulateAttention: pipelines.get(`${base}:mod-a`,
      createModulateShader({ atoms, channels }, SHIFT_ATTENTION, SCALE_ATTENTION)),
    qkv: linear("qkv", channels, channels * 3),
    gate: linear("gate", channels, channels),
    prepare: pipelines.get(`${base}:prepare`,
      createSwaPrepareShader({ atoms, channels, heads }, precision)),
    attend: pipelines.get(`${base}:attend`,
      createSwaAttendShader({ atoms, channels, heads, window })),
    attentionOut: linear("attn-out", channels, channels),
    residualAttention: pipelines.get(`${base}:res-a`,
      createGatedResidualShader({ atoms, channels }, GATE_ATTENTION)),
    modulateFfn: pipelines.get(`${base}:mod-f`,
      createModulateShader({ atoms, channels }, SHIFT_FFN, SCALE_FFN)),
    swiglu: pipelines.get(`${base}:swiglu`,
      createSwigluShader({ rows: atoms, model: channels, ffn: hidden })),
    ffnDown: linear("ffn-down", hidden, channels),
    residualFfn: pipelines.get(`${base}:res-f`,
      createGatedResidualShader({ atoms, channels }, GATE_FFN)),
  };
  const resolved = {};
  for (const [name, promise] of Object.entries(wanted)) resolved[name] = await promise;
  return { ...resolved, shape: { ...shape, precision } };
}

/** The scratch a stack needs, in elements. */
export function atomStackScratch({ atoms, channels, heads, hidden }) {
  return {
    siluCond: atoms * channels,
    modulation: atoms * channels * 6,
    normalised: atoms * channels,
    qkv: atoms * channels * 3,
    query: atoms * channels,
    key: atoms * channels,
    value: atoms * channels,
    gate: atoms * channels,
    context: atoms * channels,
    delta: atoms * channels,
    hidden: atoms * hidden,
  };
}

/**
 * Encode `blocks` SWA blocks over `activation`, in place.
 *
 * 🔴 THE CONDITIONING IS THE STACK'S, HELD FIXED FOR EVERY BLOCK. Upstream
 * passes `c0` as both the running activation and the conditioning; only the
 * activation is updated. Feeding each block its own input is a different model
 * that runs - so `silu(conditioning)` is computed once here, outside the loop,
 * which is both the correct reading and one dispatch instead of `blocks`.
 *
 * @param run    (label, pipeline, buffers, x, y, z) => void
 * @param state  { activation, conditioning, cos, sin, bounds, valid }
 * @param weights per-block { adaln, qkv, attnGate, attnOut, ffnUp, ffnDown }
 */
export function encodeAtomStack({ run, pipelines, state, scratch, weights }) {
  const { atoms, channels, heads, hidden } = pipelines.shape;
  const rows = (elements) => {
    const groups = Math.ceil(elements / LANES);
    return [Math.min(GRID_WIDTH, groups), Math.ceil(groups / GRID_WIDTH)];
  };
  const perAtom = [Math.min(GRID_WIDTH, atoms), Math.ceil(atoms / GRID_WIDTH)];
  const perHead = [Math.min(GRID_WIDTH, atoms * heads), Math.ceil(atoms * heads / GRID_WIDTH)];

  run("esmfold2.atom.silu", pipelines.silu,
      [state.conditioning, scratch.siluCond], ...rows(atoms * channels));

  for (const block of weights) {
    run("esmfold2.atom.adaln", pipelines.modulation,
        [scratch.siluCond, block.adaln, scratch.modulation],
        ...linearGrid(atoms, channels * 6));
    run("esmfold2.atom.modulate-a", pipelines.modulateAttention,
        [state.activation, scratch.modulation, scratch.normalised], ...perAtom);
    run("esmfold2.atom.qkv", pipelines.qkv,
        [scratch.normalised, block.qkv, scratch.qkv], ...linearGrid(atoms, channels * 3));
    run("esmfold2.atom.gate", pipelines.gate,
        [scratch.normalised, block.attnGate, scratch.gate], ...linearGrid(atoms, channels));
    run("esmfold2.atom.prepare", pipelines.prepare,
        [scratch.qkv, state.cos, state.sin, scratch.query, scratch.key, scratch.value],
        ...perHead);
    run("esmfold2.atom.attend", pipelines.attend,
        [scratch.query, scratch.key, scratch.value, state.bounds, state.valid,
         scratch.gate, scratch.context], ...perAtom);
    run("esmfold2.atom.attn-out", pipelines.attentionOut,
        [scratch.context, block.attnOut, scratch.delta], ...linearGrid(atoms, channels));
    run("esmfold2.atom.residual-a", pipelines.residualAttention,
        [scratch.delta, scratch.modulation, state.activation], ...rows(atoms * channels));

    run("esmfold2.atom.modulate-f", pipelines.modulateFfn,
        [state.activation, scratch.modulation, scratch.normalised], ...perAtom);
    run("esmfold2.atom.swiglu", pipelines.swiglu,
        [scratch.normalised, block.ffnUp, scratch.hidden], ...swigluGrid(atoms, hidden));
    run("esmfold2.atom.ffn-down", pipelines.ffnDown,
        [scratch.hidden, block.ffnDown, scratch.delta], ...linearGrid(atoms, channels));
    run("esmfold2.atom.residual-f", pipelines.residualFfn,
        [scratch.delta, scratch.modulation, state.activation], ...rows(atoms * channels));
  }
}
