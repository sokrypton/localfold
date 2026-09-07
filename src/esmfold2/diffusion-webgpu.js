/**
 * ESMFold2's diffusion module on the GPU: `structure_head`, one denoise step.
 *
 *     s, z   = conditioning(t_hat, s_inputs, z_trunk, rel_pos)
 *     a,q,c  = atom_encoder(features, r_l = x_noisy / sqrt(t^2 + sigma^2))
 *     a      = a + s_to_token(s_step_norm(s))
 *     a      = token_norm(token_transformer(a, s, z))
 *     r      = atom_decoder(a, q, c)
 *     out    = sigma^2/(sigma^2+t^2) * x_noisy + sigma*t/sqrt(...) * r
 *
 * src/esmfold2/diffusion-reference.js is the specification and the oracle; what
 * is here is the same arithmetic arranged so that a step is a handful of
 * submits rather than four host round trips.
 *
 * 🔴 THREE THINGS ARE CONSTANT ACROSS THE WHOLE SAMPLER AND ONLY ONE OF THEM
 * LOOKS IT. The pair conditioning `z` does not depend on the noise level, which
 * upstream says out loud by caching it. Less obviously, so does the pair BIAS
 * every token block reads - twelve `(n, n, heads)` tensors derived from `z`
 * alone - and so does `s_proj(s_input_norm(s_inputs))`, since the noise enters
 * as a broadcast vector ADDED after that projection. All three are built once
 * in `prepare` and read by every step. At 200 steps and 300 tokens that is the
 * difference between 12 GFLOP a step and 0.
 *
 * 🔴 AND THE ATOM CONDITIONING IS A HOST COMPUTATION, ONCE. `c_base` is
 * `LayerNorm(atomFeatures @ atom_linear)` over a 389-wide one-hot with about
 * twelve non-zeros a row, so it is a GATHER of twelve weight rows rather than a
 * 389-deep matmul - a few million multiply-adds for the whole molecule, done
 * once for a fold and shared by every sampler step. Uploading it beats teaching
 * a kernel to build a one-hot it will immediately contract away.
 */
import { GRID_WIDTH, LANES, createLayerNormShader, createLinearShader,
         createSwigluShader, linearGrid, swigluGrid } from "../esmc/block-webgpu.js";
import {
  atomStackScratch, atomWindows, compileAtomStack, createBroadcastShader,
  createPoolShader, encodeAtomStack, tokenRanges, widestWindow,
} from "./atom-transformer-webgpu.js";
import { buildRope } from "./atom-encoder-reference.js";

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** `out = silu(a) * b`, the unfused transition's gate. */
export function createGatedProductShader(elements) {
  return `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${elements}u) { return; }
  let v = a[i];
  output[i] = (v / (1.0 + exp(-v))) * b[i];
}`;
}

/**
 * `[z | rel]` normalised as one 2c-wide row.
 *
 * 🔴 CONCATENATED, NOT ADDED. The two are the same shape and adding them
 * conforms in every dimension; `z_input_norm` being 512 wide where the pair is
 * 256 is what says so.
 */
export function createJoinedLayerNormShader({ rows, channels }, epsilon) {
  const width = channels * 2;
  return `
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> offset: array<f32>;
@group(0) @binding(4) var<storage, read_write> destination: array<f32>;

var<workgroup> sums: array<f32, ${LANES}>;
var<workgroup> squares: array<f32, ${LANES}>;

fn joined(row: u32, c: u32) -> f32 {
  if (c < ${channels}u) { return left[row * ${channels}u + c]; }
  return right[row * ${channels}u + c - ${channels}u];
}

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x + group.y * ${GRID_WIDTH}u;
  if (row >= ${rows}u) { return; }
  var total = 0.0;
  var square = 0.0;
  for (var c = local.x; c < ${width}u; c += ${LANES}u) {
    let value = joined(row, c);
    total += value;
    square += value * value;
  }
  sums[local.x] = total;
  squares[local.x] = square;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      sums[local.x] += sums[local.x + stride];
      squares[local.x] += squares[local.x + stride];
    }
    workgroupBarrier();
  }
  let mean = sums[0] / ${width}.0;
  let inverse = inverseSqrt(max(squares[0] / ${width}.0 - mean * mean, 0.0) + ${epsilon});
  for (var c = local.x; c < ${width}u; c += ${LANES}u) {
    destination[row * ${width}u + c] = (joined(row, c) - mean) * inverse * scale[c] + offset[c];
  }
}`;
}

/**
 * adaLN-Zero, as one pass over an already-normalised pair of tensors.
 *
 * 🔴 THE TWO LayerNorms ARE NOT THE SAME KIND, and that is settled before this
 * kernel: the activation's is affine-FREE and the conditioning's has a learned
 * SCALE and no bias. There is one weight vector between them and it belongs to
 * `s`. This takes both already normalised and only combines them, so the
 * mistake cannot be made here - it is made in what the caller binds.
 *
 * `gate` and `shift` arrive as ONE (rows, 2c) tensor because they are two
 * projections of the same normalised single: concatenating the matrices on the
 * host makes them one dispatch.
 */
export function createAdaptiveCombineShader({ rows, channels }) {
  return `
@group(0) @binding(0) var<storage, read> normalisedActivation: array<f32>;
@group(0) @binding(1) var<storage, read> gateShift: array<f32>;
@group(0) @binding(2) var<storage, read> gateBias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${rows * channels}u) { return; }
  let row = i / ${channels}u;
  let c = i % ${channels}u;
  let gate = gateShift[row * ${channels * 2}u + c] + gateBias[c];
  let shift = gateShift[row * ${channels * 2}u + ${channels}u + c];
  output[i] = normalisedActivation[i] / (1.0 + exp(-gate)) + shift;
}`;
}

/**
 * `activation += delta * sigmoid(outGate + bias)`.
 *
 * 🔴 THE OUT GATE COMES FROM THE CONDITIONING SINGLE AND THE OTHER ONE DOES
 * NOT. `g_proj` gates the per-head context from the adaLN-MODULATED activation;
 * `out_gate` gates the whole output from `s`, and only the second has a bias -
 * initialised to -2 upstream, so a fresh block starts nearly closed.
 */
export function createGatedAddShader({ rows, channels }) {
  return `
@group(0) @binding(0) var<storage, read> delta: array<f32>;
@group(0) @binding(1) var<storage, read> outGate: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> activation: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${rows * channels}u) { return; }
  activation[i] += delta[i] / (1.0 + exp(-(outGate[i] + bias[i % ${channels}u])));
}`;
}

/** `a += b`, elementwise. */
export function createAddShader(elements) {
  return `
@group(0) @binding(0) var<storage, read> delta: array<f32>;
@group(0) @binding(1) var<storage, read_write> accumulator: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${elements}u) { return; }
  accumulator[i] += delta[i];
}`;
}

/** `out = a * sigmoid(gate)`, elementwise. */
export function createSigmoidGateShader(elements) {
  return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${elements}u) { return; }
  output[i] = input[i] / (1.0 + exp(-gate[i]));
}`;
}

/** `out[i] = a[i] + b[i % channels]`, a vector broadcast down the rows. */
export function createBroadcastAddShader({ rows, channels }) {
  return `
@group(0) @binding(0) var<storage, read> vector: array<f32>;
@group(0) @binding(1) var<storage, read_write> activation: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${rows * channels}u) { return; }
  activation[i] += vector[i % ${channels}u];
}`;
}

/**
 * The token attention, biased by the pair and gated by the modulated input.
 *
 * One workgroup per (query token, head), walking the keys a block of `LANES` at
 * a time with an online softmax, so nothing of size `tokens` is ever staged and
 * a long chain needs no more workgroup memory than a short one.
 *
 * 🔴 THE SOFTMAX IS OVER THE KEY AXIS OF AN (i, j, head) TENSOR - `dim=-2`
 * upstream, not the last axis. Over the heads instead it still sums to one and
 * returns a plausible tensor.
 *
 * 🔴 AND THE VALUE IS THE SECOND HALF OF A FUSED kv ROW, so it starts a whole
 * `channels` further along rather than at `head * headDim`.
 */
export function createTokenAttentionShader({ tokens, channels, heads }) {
  const headDim = channels / heads;
  return `
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> kv: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> context: array<f32>;

var<workgroup> staged_query: array<f32, ${headDim}>;
var<workgroup> logits: array<f32, ${LANES}>;
var<workgroup> reduce: array<f32, ${LANES}>;
var<workgroup> accumulator: array<f32, ${headDim}>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let slot = group.x + group.y * ${GRID_WIDTH}u;
  let token = slot / ${heads}u;
  let head = slot % ${heads}u;
  if (token >= ${tokens}u) { return; }
  let lane = local.x;

  for (var d = lane; d < ${headDim}u; d += ${LANES}u) {
    staged_query[d] = query[token * ${channels}u + head * ${headDim}u + d];
    accumulator[d] = 0.0;
  }
  workgroupBarrier();

  var running_max = -3.0e38;
  var running_sum = 0.0;
  let bias_row = token * ${tokens}u;
  for (var block = 0u; block < ${tokens}u; block += ${LANES}u) {
    let j = block + lane;
    var logit = -3.0e38;
    if (j < ${tokens}u) {
      var total = 0.0;
      let key_base = j * ${channels * 2}u + head * ${headDim}u;
      for (var d = 0u; d < ${headDim}u; d += 1u) {
        total += staged_query[d] * kv[key_base + d];
      }
      logit = total * ${(1 / Math.sqrt(headDim)).toFixed(10)}
        + bias[(bias_row + j) * ${heads}u + head];
    }
    logits[lane] = logit;
    reduce[lane] = logit;
    workgroupBarrier();
    for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
      if (lane < stride) { reduce[lane] = max(reduce[lane], reduce[lane + stride]); }
      workgroupBarrier();
    }
    let block_max = reduce[0];
    workgroupBarrier();
    let next_max = max(running_max, block_max);
    let rescale = select(exp(running_max - next_max), 0.0, running_max == -3.0e38);
    // ...the weights for this block, and the running sum rescaled onto the new
    // maximum. Both halves must use the SAME maximum or the merge is silent.
    let weight = select(0.0, exp(logits[lane] - next_max), j < ${tokens}u);
    logits[lane] = weight;
    reduce[lane] = weight;
    workgroupBarrier();
    for (var stride = ${LANES / 2}u; stride > 0u; stride >>= 1u) {
      if (lane < stride) { reduce[lane] += reduce[lane + stride]; }
      workgroupBarrier();
    }
    running_sum = running_sum * rescale + reduce[0];
    running_max = next_max;
    workgroupBarrier();
    for (var d = lane; d < ${headDim}u; d += ${LANES}u) {
      var total = 0.0;
      for (var u = 0u; u < ${LANES}u; u += 1u) {
        let key = block + u;
        if (key >= ${tokens}u) { break; }
        total += logits[u] * kv[key * ${channels * 2}u + ${channels}u + head * ${headDim}u + d];
      }
      accumulator[d] = accumulator[d] * rescale + total;
    }
    workgroupBarrier();
  }

  let by = 1.0 / max(running_sum, 1.0e-30);
  for (var d = lane; d < ${headDim}u; d += ${LANES}u) {
    context[token * ${channels}u + head * ${headDim}u + d] = accumulator[d] * by;
  }
}`;
}

/** The decoder's last projection: 128 channels to three coordinates. */
export function createCoordinateShader({ atoms, channels }) {
  return `
@group(0) @binding(0) var<storage, read> normalised: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${atoms * 3}u) { return; }
  let atom = i / 3u;
  let axis = i % 3u;
  var total = 0.0;
  for (var c = 0u; c < ${channels}u; c += 1u) {
    total += normalised[atom * ${channels}u + c] * weights[c * 3u + axis];
  }
  output[i] = total;
}`;
}

/** `joined = [noisy / denominator | previous]`, the six channels the encoder takes. */
export function noisyCoordinateFeature(noisy, previous, atoms, denominator) {
  const joined = new Float32Array(atoms * 6);
  for (let atom = 0; atom < atoms; atom += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      joined[atom * 6 + axis] = noisy[atom * 3 + axis] / denominator;
      // 🔴 `pred_r1` IS ZEROS WHEN THERE IS NO PREVIOUS PREDICTION, NOT ABSENT.
      // The projection is six channels wide either way; feeding it three reads
      // the second half of the matrix at the wrong offset.
      joined[atom * 6 + 3 + axis] = previous === undefined ? 0 : previous[atom * 3 + axis];
    }
  }
  return joined;
}

/**
 * `LayerNorm(atomFeatures @ atom_linear)`, on the host, as a gather.
 *
 * A row of `atomFeatures` is three coordinates, a charge, a mask and two
 * one-hots - so the 389-deep contraction is five multiply-adds plus five weight
 * ROWS added in. Densifying it first would be 389 times the work for the same
 * answer, and it is the same answer to the last bit because the omitted terms
 * are multiplications by zero.
 */
export function atomConditioning(features, atoms, channels, weights,
                                 { elements = 128, nameChars = 64, nameLength = 4 } = {}) {
  const { refPos, refCharge, refElement, refAtomNameChars, mask } = features;
  const elementBase = 5;
  const charBase = elementBase + elements;
  const out = new Float32Array(atoms * channels);
  const row = (feature, weight, to) => {
    const from = feature * channels;
    for (let c = 0; c < channels; c += 1) out[to + c] += weight * weights.atomLinear[from + c];
  };
  for (let atom = 0; atom < atoms; atom += 1) {
    const to = atom * channels;
    const live = mask[atom] !== 0 ? 1 : 0;
    // 🔴 ref_pos AND ref_charge ARE NOT MASKED AND THE ONE-HOTS ARE. Upstream
    // multiplies only the two one-hot blocks by the mask and passes the mask
    // itself through as a feature; masking uniformly conforms in shape.
    for (let axis = 0; axis < 3; axis += 1) row(axis, refPos[atom * 3 + axis], to);
    row(3, refCharge[atom], to);
    row(4, mask[atom], to);
    if (live) {
      row(elementBase + refElement[atom], 1, to);
      for (let i = 0; i < nameLength; i += 1) {
        row(charBase + i * nameChars + refAtomNameChars[atom * nameLength + i], 1, to);
      }
    }
  }
  // ...and the LayerNorm, which is a real one with a scale and an offset.
  for (let atom = 0; atom < atoms; atom += 1) {
    const base = atom * channels;
    let mean = 0;
    for (let c = 0; c < channels; c += 1) mean += out[base + c];
    mean /= channels;
    let variance = 0;
    for (let c = 0; c < channels; c += 1) {
      const d = out[base + c] - mean;
      variance += d * d;
    }
    const by = 1 / Math.sqrt(variance / channels + 1e-5);
    for (let c = 0; c < channels; c += 1) {
      out[base + c] = (out[base + c] - mean) * by * weights.atomNormScale[c]
        + weights.atomNormOffset[c];
    }
  }
  return out;
}

/**
 * The noise level's own 768 channels, on the host: one row of arithmetic.
 *
 * 🔴 THE NOISE LEVEL IS LOGGED AND QUARTERED, AND THE CLAMP IS NOT DECORATION.
 * The sampler's last step takes `t` to about 6e-3 and a schedule reaching zero
 * would give -Infinity here.
 *
 * 🔴 AND THE FOURIER TABLE IS A `register_buffer`, WHICH IS STILL TRAINED IN.
 * `randn(c)` drawn once at construction and saved with the checkpoint, so a
 * port that redraws it gets a different model that runs.
 */
export function noiseEmbedding(tHat, sigmaData, weights, tokenChannels) {
  const t = 0.25 * Math.log(Math.max(tHat / sigmaData, 1e-20));
  const width = weights.fourierWeights.length;
  const fourier = new Float32Array(width);
  for (let i = 0; i < width; i += 1) {
    fourier[i] = Math.cos(2 * Math.PI * (t * weights.fourierWeights[i] + weights.fourierOffsets[i]));
  }
  let mean = 0;
  for (const value of fourier) mean += value;
  mean /= width;
  let variance = 0;
  for (const value of fourier) variance += (value - mean) * (value - mean);
  const by = 1 / Math.sqrt(variance / width + 1e-5);
  const out = new Float32Array(tokenChannels);
  for (let i = 0; i < width; i += 1) {
    const normalised = (fourier[i] - mean) * by * weights.noiseNormScale[i]
      + weights.noiseNormOffset[i];
    if (normalised === 0) continue;
    const base = i * tokenChannels;
    for (let c = 0; c < tokenChannels; c += 1) out[c] += normalised * weights.noiseProjection[base + c];
  }
  return out;
}

/** How many pair rows the conditioning processes at once. */
export const PAIR_CHUNK = 8192;

const perRow = (rows) => [Math.min(GRID_WIDTH, rows), Math.ceil(rows / GRID_WIDTH)];
const elementwise = (elements) => {
  const groups = Math.ceil(elements / LANES);
  return [Math.min(GRID_WIDTH, groups), Math.ceil(groups / GRID_WIDTH)];
};

/** Two (in, out) matrices side by side, so one dispatch produces both. */
export function concatenateColumns(left, right, inner, outer) {
  const out = new Float32Array(inner * outer * 2);
  for (let i = 0; i < inner; i += 1) {
    out.set(left.subarray(i * outer, (i + 1) * outer), i * outer * 2);
    out.set(right.subarray(i * outer, (i + 1) * outer), i * outer * 2 + outer);
  }
  return out;
}

/**
 * One `structure_head` on the device, prepared once and stepped many times.
 *
 * 🔴 THE STEP IS RECORDED, NOT REBUILT. Every buffer a denoise step reads is
 * allocated in `prepare` and never reallocated, so the pipelines, the bind
 * groups and the dispatch sizes are all constants of the fold - and a step is
 * two `writeBuffer`s, one submit of a recorded program and one readback. AF3's
 * head learned this the expensive way: rebuilding the per-atom conditioning,
 * the gathers and the pair logits on the host once per step was most of an
 * 86 ms call, and a 200-step fold did it two hundred times over things that
 * never moved.
 *
 * 🔴 AND THE PAIR CONDITIONING IS BUILT IN ROW CHUNKS. Its transitions widen
 * 256 channels to 512 and hold four tensors of that shape at once, which at 300
 * tokens would be 640 MiB of scratch for arithmetic that is purely row-wise.
 * Eight thousand rows at a time costs nothing measurable and bounds it.
 */
export class Esmfold2DenoiserGpu {
  constructor(device, allocator, pipelineCache, options = {}) {
    this.device = device;
    this.allocator = allocator;
    this.cache = pipelineCache;
    this.options = options;
    this.program = [];
    this.allocations = [];
    this.buffers = {};
    this.shape = undefined;
  }

  #keep(allocation) { this.allocations.push(allocation); return allocation; }

  #upload(label, data, usage = GPUBufferUsage.STORAGE) {
    return this.#keep(this.allocator.upload(label, data, usage));
  }

  #alloc(label, elements, usage = GPUBufferUsage.STORAGE) {
    return this.#keep(this.allocator.allocate(label, Math.max(16, elements * 4), usage));
  }

  static #view(entry) {
    if (entry.byteOffset !== undefined) {
      return { buffer: entry.buffer, offset: entry.byteOffset, size: entry.byteSize };
    }
    return { buffer: entry.buffer };
  }

  #bind(pipeline, buffers) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((entry, binding) =>
        ({ binding, resource: Esmfold2DenoiserGpu.#view(entry) })),
    });
  }

  /** A slice of a buffer, in rows of `channels`. Offsets stay 256-byte aligned. */
  static slice(allocation, row, rows, channels) {
    const byteOffset = row * channels * 4;
    if (byteOffset % 256 !== 0) {
      throw new Error(`a binding offset of ${byteOffset} is not 256-byte aligned`);
    }
    return { buffer: allocation.buffer, byteOffset, byteSize: rows * channels * 4 };
  }

  /** Encode and submit a list of passes right now, outside the recorded step. */
  async #now(label, passes) {
    const encoder = this.device.createCommandEncoder({ label });
    for (const [name, pipeline, buffers, x, y] of passes) {
      const pass = encoder.beginComputePass({ label: name });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.#bind(pipeline, buffers));
      pass.dispatchWorkgroups(x, y ?? 1, 1);
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  #record(label, pipeline, buffers, x, y = 1) {
    this.program.push({ label, pipeline, bindGroup: this.#bind(pipeline, buffers), x, y });
  }


  /**
   * Every pipeline a fold of this shape needs, resolved once.
   *
   * 🔴 SHAPES THAT COINCIDE SHARE A PIPELINE AND THAT IS THE POINT OF THE
   * NAMES. `wide` is 768 -> 1536 and serves the conditioning's `a_proj`, its
   * `b_proj`, both adaLN projections and the attention's fused kv; `square` is
   * 768 -> 768 and serves the query, the gate, the output and both out-gates.
   * The pipeline cache would deduplicate them anyway - it is keyed on the
   * generated WGSL - but naming them by shape rather than by role is what stops
   * a reader believing there are twenty kernels here.
   */
  async #compile(shape) {
    const { tokens, atoms, pairChannels, singleInputs, tokenChannels, tokenHeads,
            multiplier, atomChannels, atomHeads, atomBlocks, atomHidden } = shape;
    const window = shape.stagedWindow;
    const pairs = tokens * tokens;
    const chunk = Math.min(PAIR_CHUNK, pairs);
    const tail = pairs % chunk;
    const epsilon = 1e-5;
    const key = `esmfold2-diff:${tokens}:${atoms}:${pairChannels}:${tokenChannels}:`
      + `${tokenHeads}:${multiplier}:${atomChannels}:${atomHeads}:${window}`;
    const get = (name, code) => this.cache.get(`${key}:${name}`, code);
    const hidden = tokenChannels * multiplier;

    const wanted = {
      sInputNorm: get("s-input-norm",
        createLayerNormShader({ rows: tokens, channels: singleInputs }, true, epsilon)),
      sProject: get("s-project",
        createLinearShader({ rows: tokens, inner: singleInputs, outer: tokenChannels }, false)),
      broadcastAdd: get("broadcast-add",
        createBroadcastAddShader({ rows: tokens, channels: tokenChannels })),
      normOffset: get("token-norm-offset",
        createLayerNormShader({ rows: tokens, channels: tokenChannels }, true, epsilon)),
      normScaleOnly: get("token-norm-scale",
        createLayerNormShader({ rows: tokens, channels: tokenChannels }, false, epsilon)),
      wide: get("wide",
        createLinearShader({ rows: tokens, inner: tokenChannels, outer: hidden }, false)),
      square: get("square",
        createLinearShader({ rows: tokens, inner: tokenChannels, outer: tokenChannels }, false)),
      squareResidual: get("square-residual",
        createLinearShader({ rows: tokens, inner: tokenChannels, outer: tokenChannels }, true)),
      narrow: get("narrow",
        createLinearShader({ rows: tokens, inner: hidden, outer: tokenChannels }, false)),
      swishWide: get("swish-wide",
        createLinearShader({ rows: tokens, inner: tokenChannels, outer: hidden * 2 }, false)),
      gatedSingle: get("gated-single", createGatedProductShader(tokens * hidden)),
      addSingle: get("add-single", createAddShader(tokens * tokenChannels)),
      adaptive: get("adaptive",
        createAdaptiveCombineShader({ rows: tokens, channels: tokenChannels })),
      attention: get("attention",
        createTokenAttentionShader({ tokens, channels: tokenChannels, heads: tokenHeads })),
      sigmoidGate: get("sigmoid-gate", createSigmoidGateShader(tokens * tokenChannels)),
      gatedAdd: get("gated-add",
        createGatedAddShader({ rows: tokens, channels: tokenChannels })),
      swiglu: get("swiglu",
        createSwigluShader({ rows: tokens, model: tokenChannels, ffn: hidden })),
      tokenToAtom: get("token-to-atom",
        createLinearShader({ rows: tokens, inner: tokenChannels, outer: atomChannels }, false)),
      coordsProject: get("coords",
        createLinearShader({ rows: atoms, inner: 6, outer: atomChannels }, true)),
      toToken: get("to-token",
        createLinearShader({ rows: atoms, inner: atomChannels, outer: tokenChannels }, false)),
      pool: get("pool", createPoolShader({ tokens, channels: tokenChannels })),
      broadcast: get("broadcast",
        createBroadcastShader({ atoms, channels: atomChannels })),
      atomNorm: get("atom-norm",
        createLayerNormShader({ rows: atoms, channels: atomChannels }, true, epsilon)),
      coordinate: get("coordinate",
        createCoordinateShader({ atoms, channels: atomChannels })),
    };
    // ...and the pair-shaped ones, once per distinct chunk height.
    const chunks = tail === 0 ? [chunk] : [chunk, tail];
    const pairPipelines = {};
    for (const rows of chunks) {
      pairPipelines[rows] = {
        rows,
        joinedNorm: get(`joined-norm:${rows}`,
          createJoinedLayerNormShader({ rows, channels: pairChannels }, epsilon)),
        project: get(`z-project:${rows}`,
          createLinearShader({ rows, inner: pairChannels * 2, outer: pairChannels }, false)),
        norm: get(`pair-norm:${rows}`,
          createLayerNormShader({ rows, channels: pairChannels }, true, epsilon)),
        widen: get(`pair-widen:${rows}`,
          createLinearShader({ rows, inner: pairChannels, outer: pairChannels * multiplier }, false)),
        gated: get(`pair-gated:${rows}`,
          createGatedProductShader(rows * pairChannels * multiplier)),
        narrow: get(`pair-narrow:${rows}`,
          createLinearShader({ rows, inner: pairChannels * multiplier, outer: pairChannels }, false)),
        add: get(`pair-add:${rows}`, createAddShader(rows * pairChannels)),
        bias: get(`pair-bias:${rows}`,
          createLinearShader({ rows, inner: pairChannels, outer: tokenHeads }, false)),
      };
    }

    const resolved = {};
    for (const [name, promise] of Object.entries(wanted)) resolved[name] = await promise;
    for (const rows of chunks) {
      const group = pairPipelines[rows];
      for (const [name, promise] of Object.entries(group)) {
        if (name !== "rows") group[name] = await promise;
      }
    }
    resolved.pair = pairPipelines;
    resolved.chunks = chunks;
    resolved.atomStack = await compileAtomStack(this.cache, {
      atoms, tokens, channels: atomChannels, heads: atomHeads, blocks: atomBlocks,
      hidden: atomHidden, window, precision: shape.attentionPrecision ?? "bf16",
    });
    return resolved;
  }


  /**
   * Upload everything that does not move, build what does not depend on the
   * noise level, and record the program a step runs.
   *
   * @param shape    { tokens, atoms, pairChannels, singleInputs, tokenChannels,
   *                   tokenHeads, multiplier, sigmaData, atomChannels,
   *                   atomHeads, atomBlocks, atomHidden, window }
   * @param weights  the denoiser's, plus `encoder` and `decoder` atom stacks
   * @param features the featuriser's per-atom arrays
   * @param sInputs  the 451 channels the inputs embedder produced
   * @param pair     the trunk's final pair, as a device allocation
   * @param relPos   the same relative-position encoding z_init used, on device
   *
   * 🔴 THE CONDITIONING IS WRITTEN INTO `relPos` ITSELF. The pair conditioning walks the pair in ROW
   * CHUNKS, and within a chunk `joined-norm` is the last pass to read `relPos`
   * and it runs before `z-project` writes the output - so chunk k's output can
   * go where chunk k's input was, and chunk k+1 reads rows chunk k never
   * touched. It is the same aliasing the two models already do where an
   * attention writes into its own normalised input; here it is worth a third
   * pair-sized tensor, 87.9 MiB at 300 tokens, at the fold's fullest moment.
   *
   * 🔴 THE ORDER IS LOAD-BEARING AND NOTHING VALIDATES IT. Two tensors of the
   * same element count alias without complaint, so the gate is the STRUCTURE:
   * a 300-token fold's PDB is sha256 83c0530b02f867ac with and without this.
   */
  async prepare({ shape, weights, features, sInputs, pair, relPos }) {
    const { tokens, atoms, pairChannels, singleInputs, tokenChannels, tokenHeads,
            multiplier, atomChannels, atomHeads, atomBlocks, atomHidden, window } = shape;
    const pairs = tokens * tokens;
    const hidden = tokenChannels * multiplier;
    const storage = GPUBufferUsage.STORAGE;
    this.weights = weights;

    // ---- what the host computes once, because it is a gather or one row.
    const halfWindow = window >> 1;
    const bounds = atomWindows(features.mask, atoms, halfWindow);
    const ranges = tokenRanges(features.atomToToken, features.mask, atoms, tokens);
    // 🔴 THE STAGED WINDOW IS WHAT THE RANKS GIVE, NOT `window`. A half-window
    // of 64 admits 129 keys, and fewer than that when the molecule is smaller
    // than the window - so a kernel built for 129 on a 32-atom ligand would
    // walk four times the keys that exist. It is part of the shape and
    // therefore part of the pipeline key.
    this.shape = { ...shape, stagedWindow: widestWindow(bounds, atoms) };
    const pipelines = await this.#compile(this.shape);
    this.pipelines = pipelines;
    // 🔴 THE ROPE TABLE IS bfloat16 IN A float32 MODEL, and that is worth 2.4e-3
    // through the attention. `build_3d_rope` computes in float32 and the table
    // reaches the attention at eight mantissa bits; rounding this reference's
    // own table reproduces the native one on all 5120 entries bit for bit. It
    // is the DEFAULT of `buildRope` for that reason - see the note there.
    const rope = buildRope(features.refPos, features.refSpaceUid, atoms,
                           atomChannels / atomHeads, shape.rope);
    const cBase = atomConditioning(features, atoms, atomChannels, weights.encoder);

    const b = {};
    b.cBase = this.#upload("esmfold2.diff.c-base", cBase);
    b.cos = this.#upload("esmfold2.diff.rope-cos", rope.cos);
    b.sin = this.#upload("esmfold2.diff.rope-sin", rope.sin);
    b.bounds = this.#upload("esmfold2.diff.bounds", bounds);
    b.valid = this.#upload("esmfold2.diff.valid", features.mask);
    b.ranges = this.#upload("esmfold2.diff.ranges", ranges);
    b.atomToToken = this.#upload("esmfold2.diff.atom-to-token",
      Int32Array.from(features.atomToToken));
    b.ones = this.#upload("esmfold2.diff.ones", new Float32Array(tokenChannels).fill(1));
    b.sInputs = this.#upload("esmfold2.diff.s-inputs", sInputs);
    b.joined = this.#alloc("esmfold2.diff.joined-coords", atoms * 6,
                           storage | GPUBufferUsage.COPY_DST);
    b.noise = this.#alloc("esmfold2.diff.noise", tokenChannels,
                          storage | GPUBufferUsage.COPY_DST);

    const upload = (label, data) => this.#upload(`w.esmfold2.${label}`, data);
    const stack = (blocks, label) => blocks.map((block, index) => ({
      adaln: upload(`${label}.${index}.adaln`, block.adaln),
      qkv: upload(`${label}.${index}.qkv`, block.qkv),
      attnGate: upload(`${label}.${index}.attn-gate`, block.attnGate),
      attnOut: upload(`${label}.${index}.attn-out`, block.attnOut),
      ffnUp: upload(`${label}.${index}.ffn-up`, block.ffnUp),
      ffnDown: upload(`${label}.${index}.ffn-down`, block.ffnDown),
    }));
    const encoderBlocks = stack(weights.encoder.blocks, "atom-encoder");
    const decoderBlocks = stack(weights.decoder.blocks, "atom-decoder");
    const w = {
      coordsLinear: upload("coords", weights.encoder.coordsLinear),
      toToken: upload("to-token", weights.encoder.atomToToken),
      tokenToAtom: upload("token-to-atom", weights.decoder.tokenToAtom),
      decoderNormScale: upload("dec-norm-scale", weights.decoder.normScale),
      decoderNormOffset: upload("dec-norm-offset", weights.decoder.normOffset),
      outputLinear: upload("output-linear", weights.decoder.outputLinear),
      stepNormScale: upload("step-norm-scale", weights.stepNormScale),
      stepNormOffset: upload("step-norm-offset", weights.stepNormOffset),
      singleToToken: upload("single-to-token", weights.singleToToken),
      tokenNormScale: upload("token-norm-scale", weights.tokenNormScale),
      tokenNormOffset: upload("token-norm-offset", weights.tokenNormOffset),
    };
    const condition = weights.conditioning;
    const sTransitions = condition.sTransitions.map((block, index) => ({
      normScale: upload(`s-trans.${index}.norm-scale`, block.normScale),
      normOffset: upload(`s-trans.${index}.norm-offset`, block.normOffset),
      aProjection: upload(`s-trans.${index}.a`, block.aProjection),
      bProjection: upload(`s-trans.${index}.b`, block.bProjection),
      outProjection: upload(`s-trans.${index}.out`, block.outProjection),
    }));
    // 🔴 THE adaLN GATE AND SHIFT ARE ONE MATRIX HERE AND TWO IN THE
    // CHECKPOINT. They are two projections of the SAME normalised single, so
    // side by side they are one dispatch; `concatenateColumns` is where the two
    // halves are decided, and `createAdaptiveCombineShader` reads the gate
    // first because that is the order it is written in.
    const tokenBlocks = weights.tokenBlocks.map((block, index) => {
      const adaln = (kind, source) => ({
        singleScale: upload(`b${index}.${kind}.single-scale`, source.adaln.singleScale),
        gateShift: upload(`b${index}.${kind}.gate-shift`, concatenateColumns(
          source.adaln.gateWeights, source.adaln.shiftWeights, tokenChannels, tokenChannels)),
        gateBias: upload(`b${index}.${kind}.gate-bias`, source.adaln.gateBias),
      });
      return {
        attention: {
          adaln: adaln("attn", block.attention),
          queryWeights: upload(`b${index}.query`, block.attention.queryWeights),
          queryBias: upload(`b${index}.query-bias`, block.attention.queryBias),
          kvWeights: upload(`b${index}.kv`, block.attention.kvWeights),
          gateWeights: upload(`b${index}.gate`, block.attention.gateWeights),
          outWeights: upload(`b${index}.out`, block.attention.outWeights),
          outGateWeights: upload(`b${index}.out-gate`, block.attention.outGateWeights),
          outGateBias: upload(`b${index}.out-gate-bias`, block.attention.outGateBias),
        },
        transition: {
          adaln: adaln("ffn", block.transition),
          swishWeights: upload(`b${index}.swish`, block.transition.swishWeights),
          outWeights: upload(`b${index}.ffn-out`, block.transition.outWeights),
          outGateWeights: upload(`b${index}.ffn-out-gate`, block.transition.outGateWeights),
          outGateBias: upload(`b${index}.ffn-out-gate-bias`, block.transition.outGateBias),
        },
      };
    });

    // ---- the pair conditioning, in row chunks, and the twelve biases from it.
    // ...into `relPos`, which the conditioning is the last reader of; see the
    // note on prepare. A caller that does not hand one over gets its own.
    b.pairCond = relPos ?? this.#alloc("esmfold2.diff.pair-cond", pairs * pairChannels);
    const chunk = pipelines.chunks[0];
    const scratchNorm = this.#alloc("esmfold2.diff.pair-joined", chunk * pairChannels * 2);
    const scratchA = this.#alloc("esmfold2.diff.pair-a", chunk * pairChannels * multiplier);
    const scratchB = this.#alloc("esmfold2.diff.pair-b", chunk * pairChannels * multiplier);
    const scratchG = this.#alloc("esmfold2.diff.pair-g", chunk * pairChannels * multiplier);
    const scratchD = this.#alloc("esmfold2.diff.pair-d", chunk * pairChannels);
    const zInputNormScale = upload("z-input-norm-scale", condition.zInputNormScale);
    const zInputNormOffset = upload("z-input-norm-offset", condition.zInputNormOffset);
    const zProjection = upload("z-projection", condition.zProjection);
    const zTransitions = condition.zTransitions.map((block, index) => ({
      normScale: upload(`z-trans.${index}.norm-scale`, block.normScale),
      normOffset: upload(`z-trans.${index}.norm-offset`, block.normOffset),
      aProjection: upload(`z-trans.${index}.a`, block.aProjection),
      bProjection: upload(`z-trans.${index}.b`, block.bProjection),
      outProjection: upload(`z-trans.${index}.out`, block.outProjection),
    }));

    const slice = Esmfold2DenoiserGpu.slice;
    for (let start = 0; start < pairs; start += chunk) {
      const rows = Math.min(chunk, pairs - start);
      const p = pipelines.pair[rows];
      const inPair = slice(pair, start, rows, pairChannels);
      const inRel = slice(relPos, start, rows, pairChannels);
      const outCond = slice(b.pairCond, start, rows, pairChannels);
      const passes = [
        ["joined-norm", p.joinedNorm,
         [inPair, inRel, zInputNormScale, zInputNormOffset, scratchNorm], ...perRow(rows)],
        ["z-project", p.project, [scratchNorm, zProjection, outCond],
         ...linearGrid(rows, pairChannels)],
      ];
      for (const block of zTransitions) {
        passes.push(["z-norm", p.norm,
          [outCond, block.normScale, block.normOffset, scratchNorm], ...perRow(rows)]);
        passes.push(["z-a", p.widen, [scratchNorm, block.aProjection, scratchA],
                     ...linearGrid(rows, pairChannels * multiplier)]);
        passes.push(["z-b", p.widen, [scratchNorm, block.bProjection, scratchB],
                     ...linearGrid(rows, pairChannels * multiplier)]);
        passes.push(["z-gate", p.gated, [scratchA, scratchB, scratchG],
                     ...elementwise(rows * pairChannels * multiplier)]);
        passes.push(["z-out", p.narrow, [scratchG, block.outProjection, scratchD],
                     ...linearGrid(rows, pairChannels)]);
        passes.push(["z-add", p.add, [scratchD, outCond],
                     ...elementwise(rows * pairChannels)]);
      }
      await this.#now("esmfold2.diff.pair-conditioning", passes);
    }
    // 🔴 THE WIDENED SCRATCH GOES BACK BEFORE THE BIASES ARE ALLOCATED, NOT
    // AFTER. Only `scratchNorm` is read again (the bias loop normalises the
    // conditioning through it); the transition's four are dead here, and they
    // are 56 MiB of chunk-sized buffers standing beside the twelve per-block
    // pair biases that are allocated next. This allocator does not pool -
    // release DESTROYS - so where a release sits is where the peak is.
    for (const allocation of [scratchA, scratchB, scratchG, scratchD]) {
      allocation.release();
      this.allocations.splice(this.allocations.indexOf(allocation), 1);
    }

    b.bias = weights.tokenBlocks.map((_, index) =>
      this.#alloc(`esmfold2.diff.bias.${index}`, pairs * tokenHeads));
    for (let index = 0; index < weights.tokenBlocks.length; index += 1) {
      const block = weights.tokenBlocks[index].attention;
      const scale = upload(`b${index}.pair-norm-scale`, block.pairNormScale);
      const offset = upload(`b${index}.pair-norm-offset`, block.pairNormOffset);
      const projection = upload(`b${index}.pair-bias`, block.pairBiasWeights);
      for (let start = 0; start < pairs; start += chunk) {
        const rows = Math.min(chunk, pairs - start);
        const p = pipelines.pair[rows];
        await this.#now(`esmfold2.diff.bias.${index}`, [
          ["bias-norm", p.norm,
           [slice(b.pairCond, start, rows, pairChannels), scale, offset, scratchNorm],
           ...perRow(rows)],
          ["bias-project", p.bias,
           [scratchNorm, projection, slice(b.bias[index], start, rows, tokenHeads)],
           ...linearGrid(rows, tokenHeads)],
        ]);
      }
    }
    scratchNorm.release();
    this.allocations.splice(this.allocations.indexOf(scratchNorm), 1);

    // ---- the single conditioning's noise-independent half, once.
    b.singleBase = this.#alloc("esmfold2.diff.single-base", tokens * tokenChannels,
                               storage | GPUBufferUsage.COPY_SRC);
    const sNormScratch = this.#alloc("esmfold2.diff.s-input-norm", tokens * singleInputs);
    await this.#now("esmfold2.diff.single-base", [
      ["s-input-norm", pipelines.sInputNorm,
       [b.sInputs, upload("s-input-norm-scale", condition.sInputNormScale),
        upload("s-input-norm-offset", condition.sInputNormOffset), sNormScratch],
       ...perRow(tokens)],
      ["s-project", pipelines.sProject,
       [sNormScratch, upload("s-projection", condition.sProjection), b.singleBase],
       ...linearGrid(tokens, tokenChannels)],
    ]);
    sNormScratch.release();
    this.allocations.splice(this.allocations.indexOf(sNormScratch), 1);

    // ---- the per-step scratch, allocated once so the program is a constant.
    const scratch = (label, elements) => this.#alloc(`esmfold2.diff.${label}`, elements);
    b.single = this.#alloc("esmfold2.diff.single", tokens * tokenChannels,
                           storage | GPUBufferUsage.COPY_DST);
    b.sNorm = scratch("s-norm", tokens * tokenChannels);
    b.wideA = scratch("wide-a", tokens * hidden);
    b.wideB = scratch("wide-b", tokens * hidden);
    b.wideG = scratch("wide-g", tokens * hidden);
    b.delta = scratch("delta", tokens * tokenChannels);
    b.normAct = scratch("norm-act", tokens * tokenChannels);
    b.normSingle = scratch("norm-single", tokens * tokenChannels);
    b.gateShift = scratch("gate-shift", tokens * tokenChannels * 2);
    b.modulated = scratch("modulated", tokens * tokenChannels);
    b.query = scratch("query", tokens * tokenChannels);
    b.kv = scratch("kv", tokens * tokenChannels * 2);
    b.gate = scratch("gate", tokens * tokenChannels);
    b.context = scratch("context", tokens * tokenChannels);
    b.gatedContext = scratch("gated-context", tokens * tokenChannels);
    b.outGate = scratch("out-gate", tokens * tokenChannels);
    b.act = scratch("act", tokens * tokenChannels);
    b.tokenAct = scratch("token-act", tokens * tokenChannels);
    b.projected = scratch("projected", atoms * tokenChannels);
    b.perToken = scratch("per-token", tokens * atomChannels);
    b.atomAct = scratch("atom-act", atoms * atomChannels);
    b.decoderAct = scratch("decoder-act", atoms * atomChannels);
    b.atomNormed = scratch("atom-normed", atomChannels * atoms);
    b.update = this.#alloc("esmfold2.diff.update", atoms * 3,
                           storage | GPUBufferUsage.COPY_SRC);
    b.readback = this.#alloc("esmfold2.diff.readback", atoms * 3,
                             GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const atomScratchSizes = atomStackScratch(
      { atoms, channels: atomChannels, heads: atomHeads, hidden: atomHidden });
    const atomScratch = {};
    for (const [name, elements] of Object.entries(atomScratchSizes)) {
      atomScratch[name] = scratch(`atom.${name}`, elements);
    }
    this.buffers = b;

    // ---- and the program itself.
    const record = (label, pipeline, buffers, x, y) =>
      this.#record(label, pipeline, buffers, x, y ?? 1);
    this.program = [];
    this.program.push({ kind: "copy", from: b.singleBase, to: b.single,
                        bytes: tokens * tokenChannels * 4 });
    record("esmfold2.diff.noise-add", pipelines.broadcastAdd, [b.noise, b.single],
           ...elementwise(tokens * tokenChannels));
    for (const block of sTransitions) {
      record("esmfold2.diff.s-norm", pipelines.normOffset,
             [b.single, block.normScale, block.normOffset, b.sNorm], ...perRow(tokens));
      record("esmfold2.diff.s-a", pipelines.wide, [b.sNorm, block.aProjection, b.wideA],
             ...linearGrid(tokens, hidden));
      record("esmfold2.diff.s-b", pipelines.wide, [b.sNorm, block.bProjection, b.wideB],
             ...linearGrid(tokens, hidden));
      record("esmfold2.diff.s-gate", pipelines.gatedSingle, [b.wideA, b.wideB, b.wideG],
             ...elementwise(tokens * hidden));
      record("esmfold2.diff.s-out", pipelines.narrow, [b.wideG, block.outProjection, b.delta],
             ...linearGrid(tokens, tokenChannels));
      record("esmfold2.diff.s-add", pipelines.addSingle, [b.delta, b.single],
             ...elementwise(tokens * tokenChannels));
    }

    // 🔴 THE NOISY COORDINATES ENTER THE ACTIVATION AND NEVER THE CONDITIONING.
    // `q` starts at `c_base + coords_linear([r_l | pred_r1])` while `c` stays
    // `c_base`, so the atom stack is conditioned on the reference conformer
    // alone. Adding the projection to both is the natural-looking symmetry and
    // a different model.
    record("esmfold2.diff.coords", pipelines.coordsProject,
           [b.joined, w.coordsLinear, b.cBase, b.atomAct], ...linearGrid(atoms, atomChannels));
    encodeAtomStack({
      run: (label, pipeline, buffers, x, y) => record(label, pipeline, buffers, x, y),
      pipelines: pipelines.atomStack,
      state: { activation: b.atomAct, conditioning: b.cBase, cos: b.cos, sin: b.sin,
               bounds: b.bounds, valid: b.valid },
      scratch: atomScratch, weights: encoderBlocks,
    });
    record("esmfold2.diff.to-token", pipelines.toToken, [b.atomAct, w.toToken, b.projected],
           ...linearGrid(atoms, tokenChannels));
    record("esmfold2.diff.pool", pipelines.pool,
           [b.projected, b.valid, b.ranges, b.tokenAct], ...perRow(tokens));
    record("esmfold2.diff.step-norm", pipelines.normOffset,
           [b.single, w.stepNormScale, w.stepNormOffset, b.sNorm], ...perRow(tokens));
    record("esmfold2.diff.step-project", pipelines.squareResidual,
           [b.sNorm, w.singleToToken, b.tokenAct, b.act], ...linearGrid(tokens, tokenChannels));

    for (let index = 0; index < tokenBlocks.length; index += 1) {
      const block = tokenBlocks[index];
      const attention = block.attention;
      record("esmfold2.diff.attn-norm-act", pipelines.normScaleOnly,
             [b.act, b.ones, b.normAct], ...perRow(tokens));
      record("esmfold2.diff.attn-norm-single", pipelines.normScaleOnly,
             [b.single, attention.adaln.singleScale, b.normSingle], ...perRow(tokens));
      record("esmfold2.diff.attn-gate-shift", pipelines.wide,
             [b.normSingle, attention.adaln.gateShift, b.gateShift],
             ...linearGrid(tokens, tokenChannels * 2));
      record("esmfold2.diff.attn-adaln", pipelines.adaptive,
             [b.normAct, b.gateShift, attention.adaln.gateBias, b.modulated],
             ...elementwise(tokens * tokenChannels));
      record("esmfold2.diff.attn-query", pipelines.square,
             [b.modulated, attention.queryWeights, b.query], ...linearGrid(tokens, tokenChannels));
      record("esmfold2.diff.attn-query-bias", pipelines.broadcastAdd,
             [attention.queryBias, b.query], ...elementwise(tokens * tokenChannels));
      record("esmfold2.diff.attn-kv", pipelines.wide,
             [b.modulated, attention.kvWeights, b.kv], ...linearGrid(tokens, tokenChannels * 2));
      record("esmfold2.diff.attn-gate", pipelines.square,
             [b.modulated, attention.gateWeights, b.gate], ...linearGrid(tokens, tokenChannels));
      record("esmfold2.diff.attend", pipelines.attention,
             [b.query, b.kv, b.bias[index], b.context],
             ...perRow(tokens * tokenHeads));
      record("esmfold2.diff.attn-context-gate", pipelines.sigmoidGate,
             [b.context, b.gate, b.gatedContext], ...elementwise(tokens * tokenChannels));
      record("esmfold2.diff.attn-out", pipelines.square,
             [b.gatedContext, attention.outWeights, b.delta], ...linearGrid(tokens, tokenChannels));
      record("esmfold2.diff.attn-out-gate", pipelines.square,
             [b.single, attention.outGateWeights, b.outGate], ...linearGrid(tokens, tokenChannels));
      record("esmfold2.diff.attn-add", pipelines.gatedAdd,
             [b.delta, b.outGate, attention.outGateBias, b.act],
             ...elementwise(tokens * tokenChannels));

      const transition = block.transition;
      record("esmfold2.diff.ffn-norm-act", pipelines.normScaleOnly,
             [b.act, b.ones, b.normAct], ...perRow(tokens));
      record("esmfold2.diff.ffn-norm-single", pipelines.normScaleOnly,
             [b.single, transition.adaln.singleScale, b.normSingle], ...perRow(tokens));
      record("esmfold2.diff.ffn-gate-shift", pipelines.wide,
             [b.normSingle, transition.adaln.gateShift, b.gateShift],
             ...linearGrid(tokens, tokenChannels * 2));
      record("esmfold2.diff.ffn-adaln", pipelines.adaptive,
             [b.normAct, b.gateShift, transition.adaln.gateBias, b.modulated],
             ...elementwise(tokens * tokenChannels));
      record("esmfold2.diff.ffn-swiglu", pipelines.swiglu,
             [b.modulated, transition.swishWeights, b.wideG], ...swigluGrid(tokens, hidden));
      record("esmfold2.diff.ffn-out", pipelines.narrow,
             [b.wideG, transition.outWeights, b.delta], ...linearGrid(tokens, tokenChannels));
      record("esmfold2.diff.ffn-out-gate", pipelines.square,
             [b.single, transition.outGateWeights, b.outGate], ...linearGrid(tokens, tokenChannels));
      record("esmfold2.diff.ffn-add", pipelines.gatedAdd,
             [b.delta, b.outGate, transition.outGateBias, b.act],
             ...elementwise(tokens * tokenChannels));
    }

    record("esmfold2.diff.token-norm", pipelines.normOffset,
           [b.act, w.tokenNormScale, w.tokenNormOffset, b.normAct], ...perRow(tokens));
    record("esmfold2.diff.token-to-atom", pipelines.tokenToAtom,
           [b.normAct, w.tokenToAtom, b.perToken], ...linearGrid(tokens, atomChannels));
    record("esmfold2.diff.decoder-start", pipelines.broadcast,
           [b.atomAct, b.perToken, b.atomToToken, b.decoderAct],
           ...elementwise(atoms * atomChannels));
    encodeAtomStack({
      run: (label, pipeline, buffers, x, y) => record(label, pipeline, buffers, x, y),
      pipelines: pipelines.atomStack,
      state: { activation: b.decoderAct, conditioning: b.cBase, cos: b.cos, sin: b.sin,
               bounds: b.bounds, valid: b.valid },
      scratch: atomScratch, weights: decoderBlocks,
    });
    record("esmfold2.diff.decoder-norm", pipelines.atomNorm,
           [b.decoderAct, w.decoderNormScale, w.decoderNormOffset, b.atomNormed],
           ...perRow(atoms));
    record("esmfold2.diff.coordinate", pipelines.coordinate,
           [b.atomNormed, w.outputLinear, b.update], ...elementwise(atoms * 3));
    return this;
  }

  /**
   * One denoise step: noisy coordinates and a noise level in, coordinates out.
   *
   * 🔴 THE LAST LINE IS EDM PRECONDITIONING, NOT A RESIDUAL.
   * `sigma^2/(sigma^2+t^2) * x_noisy + sigma*t/sqrt(sigma^2+t^2) * r` - at a
   * large noise level the first term is nearly zero and the answer is almost
   * all network, at a small one almost all input. Writing it as
   * `x_noisy + r_update` runs and converges to something.
   */
  async denoise(noisy, tHat, previous = undefined) {
    const { atoms, tokenChannels, sigmaData } = this.shape;
    const b = this.buffers;
    const denominator = Math.sqrt(tHat * tHat + sigmaData * sigmaData);
    this.device.queue.writeBuffer(b.joined.buffer, 0,
      noisyCoordinateFeature(noisy, previous, atoms, denominator));
    this.device.queue.writeBuffer(b.noise.buffer, 0,
      noiseEmbedding(tHat, sigmaData, this.weights.conditioning, tokenChannels));

    const encoder = this.device.createCommandEncoder({ label: "esmfold2.denoise" });
    for (const step of this.program) {
      if (step.kind === "copy") {
        encoder.copyBufferToBuffer(step.from.buffer, 0, step.to.buffer, 0, step.bytes);
        continue;
      }
      const pass = encoder.beginComputePass({ label: step.label });
      pass.setPipeline(step.pipeline);
      pass.setBindGroup(0, step.bindGroup);
      pass.dispatchWorkgroups(step.x, step.y, 1);
      pass.end();
    }
    encoder.copyBufferToBuffer(b.update.buffer, 0, b.readback.buffer, 0, atoms * 3 * 4);
    this.device.queue.submit([encoder.finish()]);
    await b.readback.buffer.mapAsync(GPUMapMode.READ);
    const update = new Float32Array(b.readback.buffer.getMappedRange().slice(0));
    b.readback.buffer.unmap();

    const sigma2 = sigmaData * sigmaData;
    const t2 = tHat * tHat;
    const keep = sigma2 / (sigma2 + t2);
    const take = (sigmaData * tHat) / Math.sqrt(sigma2 + t2);
    const out = new Float32Array(noisy.length);
    for (let i = 0; i < noisy.length; i += 1) out[i] = keep * noisy[i] + take * update[i];
    return out;
  }

  release() {
    for (let at = this.allocations.length - 1; at >= 0; at -= 1) this.allocations[at].release();
    this.allocations = [];
    this.program = [];
    this.buffers = {};
  }
}
