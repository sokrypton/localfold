/**
 * AF3's diffusion token transformer: 24 blocks, AdaLN-conditioned, pair-biased.
 *
 * The bulk of the diffusion head's parameters. Each block is
 *
 *     act += adaptiveZeroInit(attention(adaptiveLayerNorm(act, cond)))
 *     act += adaptiveZeroInit(swiglu(adaptiveLayerNorm(act, cond)))
 *
 * where AdaLN is `sigmoid(scale(cond)) * layerNorm(act) + bias(cond)` and the
 * zero-init gate is `sigmoid(zeroCond(cond))`, whose bias is initialised at -2
 * so an untrained block starts near the identity.
 *
 * 🔴 THE BLOCKS ARE NESTED SIX BY FOUR AND THE PAIR LOGITS FOLLOW THAT NESTING.
 * The LayerNorm over the pair conditioning is computed ONCE and shared, but each
 * of the six SUPER-BLOCKS projects it to its own four blocks' worth of head
 * biases - so there are six projections, not one and not twenty-four. The
 * checkpoint says so: pair_logits_projection is (6, 128, 4, 16). A flat reading
 * of the stack indexes the wrong weights for every block after the fourth.
 *
 * 🔴 THE CONDITIONING IS NARROWER THAN THE ACTIVATION. cond is 384 and act is
 * 768, so every AdaLN projection is 384->768 rather than square. In the atom
 * stacks both are 128 and the distinction is invisible, which is exactly how a
 * square assumption survives to here and then reads at the wrong stride.
 *
 * 🔴 THE ATTENTION SCALE IS THE PER-HEAD DIMENSION, taken AFTER dividing by the
 * head count: AF3 writes `key_dim = key_dim // num_head` and only then
 * `key_dim ** -0.5`. That is 48, not 768. Using the full width is a factor of
 * four on every logit, which softmax turns into much flatter attention - and
 * flatter attention still folds proteins, just worse.
 *
 * 🔴 THE LayerNorms HERE ARE TWO-PASS, not the trunk's fast variance. AF3 sets
 * use_fast_variance=False for the diffusion and atom stacks. See
 * src/triangle/shaders.js for why that cannot be a global.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { DeferredValidation } from "../runtime/validation.js";

const GRID_WIDTH = 32_768;

const BLOCK_ORDER = [
  "SingleCondLayerNormScale", "SingleCondScaleWeights", "SingleCondScaleBias", "SingleCondBias",
  "qProjection", "qBias", "kProjection", "vProjection", "gatingQuery",
  "Transition2", "AdaptiveZeroCondWeights", "AdaptiveZeroCondBias",
  "ffwSingleCondLayerNormScale", "ffwSingleCondScaleWeights", "ffwSingleCondScaleBias",
  "ffwSingleCondBias", "ffwTransition1", "ffwTransition2",
  "ffwAdaptiveZeroCondWeights", "ffwAdaptiveZeroCondBias",
];

/**
 * 🔴 PACKED ONCE PER BLOCK OBJECT, NOT ONCE PER CALL. A block is about 6.5M
 * floats, so packing twenty-four of them allocates and memcpies ~630 MB of
 * CPU-side Float32Array on every denoiser call - and a 200-step fold makes 200
 * of those calls with the SAME weights each time. The keys are the weight
 * objects the loader built, which live as long as the model does, so a WeakMap
 * lets the cache go when the model does.
 */
const packed = new WeakMap();

export function packBlockWeightsCached(block) {
  let entry = packed.get(block);
  if (entry === undefined) {
    entry = packBlockWeights(block);
    packed.set(block, entry);
  }
  return entry;
}

/**
 * Block weights uploaded once and left on the device.
 *
 * 🔴 THE UPLOAD WAS THE FLOOR, NOT THE ARITHMETIC. A block is ~26 MB, so the
 * loop wrote ~630 MB to the device per call - and at eight tokens, where the
 * matmuls are nothing, twenty-four blocks still cost 174 ms, which is that
 * write at about 3.6 GB/s. A 200-step fold did it two hundred times over
 * weights that never change.
 *
 * 🔴 SO THIS TRADES DEVICE MEMORY FOR IT, DELIBERATELY. The stack stays
 * resident - ~630 MB at f32, the same order as the checkpoint itself - held as
 * long as the weight objects the loader built are reachable, which is as long
 * as the model is loaded. A denoiser called two hundred times should upload its
 * weights once.
 *
 * Keyed device-first so two devices cannot hand each other a buffer, and by the
 * block OBJECT so the buffers go when the model does.
 */
const residentBlocks = new WeakMap();

function residentBlockBuffer(device, block, packedBlock) {
  let byDevice = residentBlocks.get(device);
  if (byDevice === undefined) {
    byDevice = new WeakMap();
    residentBlocks.set(device, byDevice);
  }
  let buffer = byDevice.get(block);
  if (buffer === undefined) {
    // 🔴 NOT THROUGH allocator.upload, WHOSE ALLOCATIONS ARE POOLED AND
    // RECYCLED at the end of the run that made them. This one has to outlive
    // every run, so it is created directly and never released.
    buffer = device.createBuffer({
      label: "difftx.block.resident",
      size: Math.ceil(packedBlock.data.byteLength / 4) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, packedBlock.data.buffer,
                             packedBlock.data.byteOffset, packedBlock.data.byteLength);
    byDevice.set(block, buffer);
  }
  return buffer;
}

export function packBlockWeights(block) {
  const offsets = {};
  let total = 0;
  for (const name of BLOCK_ORDER) {
    if (block[name] === undefined) throw new Error(`diffusion block missing ${name}`);
    offsets[name] = total;
    total += block[name].length;
  }
  const data = new Float32Array(total);
  for (const name of BLOCK_ORDER) data.set(block[name], offsets[name]);
  return { data, offsets };
}

export function createDiffusionTransformerShaders(shape, offsets) {
  const { tokens, channels, condChannels, pairChannels, heads, dimension, factor } = shape;
  // 🔴 THE FOUR PER-TOKEN KERNELS RUN ONE WORKGROUP PER TOKEN, so the token
  // count IS the occupancy: a 59-residue chain launched 59 workgroups of 64
  // threads, which is under four thousand threads for a GPU that wants tens of
  // thousands, and each of those threads then walked a 768-long dot product.
  // Widening the workgroup is the cheap half of fixing that - the same work,
  // more lanes over it - and it costs only workgroup memory, which the
  // transition's 1536-wide scratch dominates anyway.
  const lanes = shape.lanes ?? 256;
  const width = heads * dimension;
  const intermediate = channels * factor;
  const pairs = tokens * tokens;

  const common = `
const TOKENS: u32 = ${tokens}u;
const PAIRS: u32 = ${pairs}u;
const C: u32 = ${channels}u;
const C_COND: u32 = ${condChannels}u;
const C_PAIR: u32 = ${pairChannels}u;
const HEADS: u32 = ${heads}u;
const DIMENSION: u32 = ${dimension}u;
const WIDTH: u32 = ${width}u;
const INTERMEDIATE: u32 = ${intermediate}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1.0e-5;
const SCALE: f32 = ${1 / Math.sqrt(dimension)};
${Object.entries(offsets).map(([name, value]) => `const W_${name}: u32 = ${value}u;`).join("\n")}

fn logistic(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
fn swish(value: f32) -> f32 { return value / (1.0 + exp(-value)); }
`;

  // The shared LayerNorm over the pair conditioning. Two-pass, no offset.
  const normalisePair = `
const PAIRS: u32 = ${pairs}u;
const C_PAIR: u32 = ${pairChannels}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1.0e-5;
@group(0) @binding(0) var<storage, read> pair_cond: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> normalized: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * C_PAIR;
  var total = 0.0;
  for (var c = 0u; c < C_PAIR; c += 1u) { total += pair_cond[base + c]; }
  let mean = total / f32(C_PAIR);
  var variance = 0.0;
  for (var c = 0u; c < C_PAIR; c += 1u) {
    let d = pair_cond[base + c] - mean;
    variance += d * d;
  }
  let inverse_std = inverseSqrt(variance / f32(C_PAIR) + EPSILON);
  for (var c = 0u; c < C_PAIR; c += 1u) {
    normalized[base + c] = (pair_cond[base + c] - mean) * inverse_std * scale[c];
  }
}`;

  // One super-block's projection, unpacked into per-block head-major logits.
  // 🔴 THE PROJECTION IS (pair, blocksPerSuper, heads) and the attention wants
  // (head, i, j) - so this pass is where the six-by-four nesting is resolved.
  // 🔴 ONE SHADER PER POSITION IN THE SUPER-BLOCK. The inner index selects a
  // column group of the projection, and the pipeline cache here takes no
  // override constants - so it is baked in rather than passed. Four sources,
  // compiled once each, not twenty-four.
  const pairLogitsFor = (inner, perSuper) => `${common}
const INNER: u32 = ${inner}u;
const PER_SUPER: u32 = ${perSuper}u;
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> projection: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * C_PAIR;
  for (var head = 0u; head < HEADS; head += 1u) {
    var total = 0.0;
    let column = INNER * HEADS + head;
    for (var c = 0u; c < C_PAIR; c += 1u) {
      total += normalized[base + c] * projection[c * PER_SUPER * HEADS + column];
    }
    logits[head * PAIRS + row] = total;
  }
}`;

  // AdaLN, then q/k/v/gate. One workgroup per token.
  const project = `${common}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read_write> k: array<f32>;
@group(0) @binding(5) var<storage, read_write> v: array<f32>;
@group(0) @binding(6) var<storage, read_write> gate: array<f32>;

var<workgroup> x: array<f32, ${channels}>;
var<workgroup> cond_norm: array<f32, ${condChannels}>;
var<workgroup> reduce_a: array<f32, ${lanes}>;

fn reduce_sum(local: u32, value: f32) -> f32 {
  reduce_a[local] = value;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_a[local] += reduce_a[local + stride]; }
    workgroupBarrier();
  }
  return reduce_a[0];
}

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x;
  if (token >= TOKENS) { return; }
  let local = local_id.x;

  // 🔴 TWO-PASS VARIANCE, and no scale or offset on the activation's own norm.
  var total = 0.0;
  for (var c = local; c < C; c += ${lanes}u) { total += act[token * C + c]; }
  let act_mean = reduce_sum(local, total) / f32(C);
  workgroupBarrier();
  var centred = 0.0;
  for (var c = local; c < C; c += ${lanes}u) {
    let d = act[token * C + c] - act_mean;
    centred += d * d;
  }
  let act_inverse = inverseSqrt(reduce_sum(local, centred) / f32(C) + EPSILON);
  workgroupBarrier();

  // ...the conditioning gets a scale but NO offset before it is projected.
  var cond_total = 0.0;
  for (var c = local; c < C_COND; c += ${lanes}u) { cond_total += cond[token * C_COND + c]; }
  let cond_mean = reduce_sum(local, cond_total) / f32(C_COND);
  workgroupBarrier();
  var cond_centred = 0.0;
  for (var c = local; c < C_COND; c += ${lanes}u) {
    let d = cond[token * C_COND + c] - cond_mean;
    cond_centred += d * d;
  }
  let cond_inverse = inverseSqrt(reduce_sum(local, cond_centred) / f32(C_COND) + EPSILON);
  workgroupBarrier();
  for (var c = local; c < C_COND; c += ${lanes}u) {
    cond_norm[c] = (cond[token * C_COND + c] - cond_mean) * cond_inverse
      * weights[W_SingleCondLayerNormScale + c];
  }
  workgroupBarrier();

  for (var c = local; c < C; c += ${lanes}u) {
    var scale_value = weights[W_SingleCondScaleBias + c];
    var shift = 0.0;
    for (var d = 0u; d < C_COND; d += 1u) {
      scale_value += cond_norm[d] * weights[W_SingleCondScaleWeights + d * C + c];
      shift += cond_norm[d] * weights[W_SingleCondBias + d * C + c];
    }
    let normalized = (act[token * C + c] - act_mean) * act_inverse;
    x[c] = logistic(scale_value) * normalized + shift;
  }
  workgroupBarrier();

  for (var out = local; out < WIDTH; out += ${lanes}u) {
    var q_total = weights[W_qBias + out];   // only q has a bias
    var k_total = 0.0;
    var v_total = 0.0;
    var gate_total = 0.0;
    for (var c = 0u; c < C; c += 1u) {
      let value = x[c];
      q_total += value * weights[W_qProjection + c * WIDTH + out];
      k_total += value * weights[W_kProjection + c * WIDTH + out];
      v_total += value * weights[W_vProjection + c * WIDTH + out];
      gate_total += value * weights[W_gatingQuery + c * WIDTH + out];
    }
    let index = token * WIDTH + out;
    q[index] = q_total;
    k[index] = k_total;
    v[index] = v_total;
    gate[index] = gate_total;
  }
}`;

  const attend = `${common}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> pair_logits: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> gathered: array<f32>;

var<workgroup> logits: array<f32, ${tokens}>;
var<workgroup> reduce: array<f32, ${lanes}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let slot = group.x + group.y * GRID_WIDTH;
  if (slot >= TOKENS * HEADS) { return; }
  let head = slot % HEADS;
  let i = slot / HEADS;
  let local = local_id.x;
  let query_base = i * WIDTH + head * DIMENSION;

  for (var j = local; j < TOKENS; j += ${lanes}u) {
    var dot = 0.0;
    for (var d = 0u; d < DIMENSION; d += 1u) {
      dot += q[query_base + d] * k[j * WIDTH + head * DIMENSION + d];
    }
    logits[j] = dot * SCALE + 1.0e9 * (mask[j] - 1.0)
      + pair_logits[(head * TOKENS + i) * TOKENS + j];
  }
  workgroupBarrier();

  var local_max = -3.0e38;
  for (var j = local; j < TOKENS; j += ${lanes}u) { local_max = max(local_max, logits[j]); }
  reduce[local] = local_max;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
    workgroupBarrier();
  }
  let largest = reduce[0];
  workgroupBarrier();

  var local_sum = 0.0;
  for (var j = local; j < TOKENS; j += ${lanes}u) {
    let value = exp(logits[j] - largest);
    logits[j] = value;
    local_sum += value;
  }
  reduce[local] = local_sum;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let total = reduce[0];
  workgroupBarrier();

  for (var d = local; d < DIMENSION; d += ${lanes}u) {
    var sum = 0.0;
    for (var j = 0u; j < TOKENS; j += 1u) {
      sum += logits[j] * v[j * WIDTH + head * DIMENSION + d];
    }
    gathered[i * WIDTH + head * DIMENSION + d] = sum / total;
  }
}`;

  // Gate, project back, apply the zero-init gate, and add to the residual.
  const attentionOutput = `${common}
@group(0) @binding(0) var<storage, read> gathered: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> cond: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> act: array<f32>;

var<workgroup> gated: array<f32, ${width}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x;
  if (token >= TOKENS) { return; }
  let local = local_id.x;
  for (var w = local; w < WIDTH; w += ${lanes}u) {
    let index = token * WIDTH + w;
    gated[w] = gathered[index] * logistic(gate[index]);
  }
  workgroupBarrier();

  for (var c = local; c < C; c += ${lanes}u) {
    var projected = 0.0;
    for (var w = 0u; w < WIDTH; w += 1u) {
      projected += gated[w] * weights[W_Transition2 + w * C + c];
    }
    // 🔴 THE ZERO-INIT GATE READS THE RAW CONDITIONING, not the normalised one.
    var zero_gate = weights[W_AdaptiveZeroCondBias + c];
    for (var d = 0u; d < C_COND; d += 1u) {
      zero_gate += cond[token * C_COND + d] * weights[W_AdaptiveZeroCondWeights + d * C + c];
    }
    act[token * C + c] = act[token * C + c] + projected * logistic(zero_gate);
  }
}`;

  // The conditioned SwiGLU transition, whole, one workgroup per token. The
  // widened tensor stays in workgroup memory.
  const transition = `${common}
@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> act: array<f32>;

var<workgroup> x: array<f32, ${channels}>;
var<workgroup> cond_norm: array<f32, ${condChannels}>;
var<workgroup> gated: array<f32, ${intermediate}>;
var<workgroup> reduce_a: array<f32, ${lanes}>;

fn reduce_sum(local: u32, value: f32) -> f32 {
  reduce_a[local] = value;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_a[local] += reduce_a[local + stride]; }
    workgroupBarrier();
  }
  return reduce_a[0];
}

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x;
  if (token >= TOKENS) { return; }
  let local = local_id.x;

  var total = 0.0;
  for (var c = local; c < C; c += ${lanes}u) { total += act[token * C + c]; }
  let act_mean = reduce_sum(local, total) / f32(C);
  workgroupBarrier();
  var centred = 0.0;
  for (var c = local; c < C; c += ${lanes}u) {
    let d = act[token * C + c] - act_mean;
    centred += d * d;
  }
  let act_inverse = inverseSqrt(reduce_sum(local, centred) / f32(C) + EPSILON);
  workgroupBarrier();

  var cond_total = 0.0;
  for (var c = local; c < C_COND; c += ${lanes}u) { cond_total += cond[token * C_COND + c]; }
  let cond_mean = reduce_sum(local, cond_total) / f32(C_COND);
  workgroupBarrier();
  var cond_centred = 0.0;
  for (var c = local; c < C_COND; c += ${lanes}u) {
    let d = cond[token * C_COND + c] - cond_mean;
    cond_centred += d * d;
  }
  let cond_inverse = inverseSqrt(reduce_sum(local, cond_centred) / f32(C_COND) + EPSILON);
  workgroupBarrier();
  for (var c = local; c < C_COND; c += ${lanes}u) {
    cond_norm[c] = (cond[token * C_COND + c] - cond_mean) * cond_inverse
      * weights[W_ffwSingleCondLayerNormScale + c];
  }
  workgroupBarrier();

  for (var c = local; c < C; c += ${lanes}u) {
    var scale_value = weights[W_ffwSingleCondScaleBias + c];
    var shift = 0.0;
    for (var d = 0u; d < C_COND; d += 1u) {
      scale_value += cond_norm[d] * weights[W_ffwSingleCondScaleWeights + d * C + c];
      shift += cond_norm[d] * weights[W_ffwSingleCondBias + d * C + c];
    }
    x[c] = logistic(scale_value) * ((act[token * C + c] - act_mean) * act_inverse) + shift;
  }
  workgroupBarrier();

  // 🔴 BLOCKED, gate half first - the same convention as the trunk's transition
  // and the opposite of triangle multiplication's interleave.
  let wide = INTERMEDIATE * 2u;
  for (var i = local; i < INTERMEDIATE; i += ${lanes}u) {
    var gate_value = 0.0;
    var value = 0.0;
    for (var c = 0u; c < C; c += 1u) {
      let column = W_ffwTransition1 + c * wide;
      gate_value += x[c] * weights[column + i];
      value += x[c] * weights[column + INTERMEDIATE + i];
    }
    gated[i] = swish(gate_value) * value;
  }
  workgroupBarrier();

  for (var c = local; c < C; c += ${lanes}u) {
    var projected = 0.0;
    for (var i = 0u; i < INTERMEDIATE; i += 1u) {
      projected += gated[i] * weights[W_ffwTransition2 + i * C + c];
    }
    var zero_gate = weights[W_ffwAdaptiveZeroCondBias + c];
    for (var d = 0u; d < C_COND; d += 1u) {
      zero_gate += cond[token * C_COND + d] * weights[W_ffwAdaptiveZeroCondWeights + d * C + c];
    }
    act[token * C + c] = act[token * C + c] + projected * logistic(zero_gate);
  }
}`;

  return { normalisePair, pairLogitsFor, project, attend, attentionOutput, transition };
}

export class Af3DiffusionTransformerGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {Float32Array} act tokens * channels
   * @param {Float32Array} cond tokens * condChannels
   * @param {Float32Array} pairCond tokens * tokens * pairChannels
   * @param {Float32Array} mask tokens
   * @param {number} tokens
   * @param {object} weights channels, condChannels, heads, dimension,
   *   transitionFactor, blocksPerSuperBlock, pairInputLayerNormScale, superBlocks
   */
  async run(act, cond, pairCond, mask, tokens, weights) {
    const channels = weights.channels;
    const condChannels = weights.condChannels;
    const pairChannels = weights.pairChannels;
    const heads = weights.heads;
    const dimension = weights.dimension;
    const perSuper = weights.blocksPerSuperBlock;
    const width = heads * dimension;
    const pairs = tokens * tokens;
    if (act.length !== tokens * channels) {
      throw new Error(`act has ${act.length} elements; expected ${tokens * channels}`);
    }

    const sample = packBlockWeights(weights.superBlocks[0].blocks[0]);
    const shape = { tokens, channels, condChannels, pairChannels, heads, dimension,
                    factor: weights.transitionFactor, lanes: weights.lanes };
    const sources = createDiffusionTransformerShaders(shape, sample.offsets);
    // 🔴 THE LANE COUNT IS PART OF THE KEY. It is baked into every one of these
    // sources as a workgroup size, so a cache that ignored it would hand a
    // later run the pipeline compiled for a different width.
    const base = `af3-difftx:${tokens}:${channels}:${condChannels}:${pairChannels}`
      + `:${heads}:${dimension}:${weights.transitionFactor}:${perSuper}`
      + `:${shape.lanes ?? "default"}`;
    const compiled = {};
    for (const [name, source] of Object.entries(sources)) {
      if (name === "pairLogitsFor") continue;
      compiled[name] = await this.pipelines.get(`${base}:${name}`, source);
    }
    compiled.pairLogits = [];
    for (let inner = 0; inner < perSuper; inner += 1) {
      compiled.pairLogits.push(await this.pipelines.get(
        `${base}:pair-logits:${inner}`, sources.pairLogitsFor(inner, perSuper)));
    }

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const actBuffer = keep(this.allocator.upload("difftx.act", Float32Array.from(act),
        storage | GPUBufferUsage.COPY_SRC));
      const condBuffer = keep(this.allocator.upload("difftx.cond", cond, storage));
      const pairBuffer = keep(this.allocator.upload("difftx.pair", pairCond, storage));
      const maskBuffer = keep(this.allocator.upload("difftx.mask", mask, storage));
      const pairScale = keep(this.allocator.upload("difftx.pair-scale",
        weights.pairInputLayerNormScale, storage));
      const normalized = keep(this.allocator.allocate("difftx.pair-norm",
        pairs * pairChannels * 4, storage));
      const logits = keep(this.allocator.allocate("difftx.logits", heads * pairs * 4, storage));
      const q = keep(this.allocator.allocate("difftx.q", tokens * width * 4, storage));
      const k = keep(this.allocator.allocate("difftx.k", tokens * width * 4, storage));
      const v = keep(this.allocator.allocate("difftx.v", tokens * width * 4, storage));
      const gate = keep(this.allocator.allocate("difftx.gate", tokens * width * 4, storage));
      const gathered = keep(this.allocator.allocate("difftx.gathered", tokens * width * 4, storage));
      const readback = keep(this.allocator.allocate("difftx.readback", tokens * channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      const start = performance.now();
      // 🔴 ONE VALIDATION SCOPE PER BLOCK, NONE OF THEM AWAITED IN THE LOOP.
      // See src/runtime/validation.js: `await popErrorScope()` between blocks
      // puts a host-device synchronisation in the middle of the stack, which is
      // what the pairformer found first. This loop also awaited
      // `onSubmittedWorkDone()` per block on top of that - two round trips a
      // block, twenty-four blocks, every denoiser call - and a denoiser call
      // happens up to 200 times a fold.
      const validation = new DeferredValidation(this.device, "diffusion transformer");
      // The shared pair LayerNorm, once for the whole stack.
      {
        validation.begin();
        const encoder = this.device.createCommandEncoder({ label: "difftx.pair-norm" });
        const pass = encoder.beginComputePass({ label: "pair-norm" });
        pass.setPipeline(compiled.normalisePair);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: compiled.normalisePair.getBindGroupLayout(0),
          entries: [pairBuffer, pairScale, normalized].map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        const groups = Math.ceil(pairs / 64);
        pass.dispatchWorkgroups(Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        validation.end("pair layer norm");
      }

      // 🔴 ONE ENCODER AND ONE SUBMIT FOR ALL TWENTY-FOUR BLOCKS. Every block
      // used to finish and submit its own command buffer, which at eight tokens
      // - where the matmuls are nothing at all - was most of what the stack
      // cost. The blocks are strictly sequential on the same buffers and WebGPU
      // orders passes within an encoder, so batching them changes nothing about
      // what runs, only how many times the CPU asks the driver to run it.
      validation.begin();
      const encoder = this.device.createCommandEncoder({ label: "difftx.stack" });
      const run = (label, pipeline, buffers, x, y = 1) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(x, y);
        pass.end();
      };
      const projections = [];
      for (const group of weights.superBlocks) {
        const projection = this.allocator.upload("difftx.pair-projection",
          group.pairLogitsProjection, storage);
        projections.push(projection);
        for (let inner = 0; inner < group.blocks.length; inner += 1) {
          const block = group.blocks[inner];
          const blockWeights = {
            buffer: residentBlockBuffer(this.device, block, packBlockWeightsCached(block)),
          };
          const pairGroups = Math.ceil(pairs / 64);
          run("pair-logits", compiled.pairLogits[inner], [normalized, projection, logits],
              Math.min(pairGroups, GRID_WIDTH), Math.ceil(pairGroups / GRID_WIDTH));
          run("project", compiled.project,
              [actBuffer, condBuffer, blockWeights, q, k, v, gate], tokens);
          const slots = tokens * heads;
          run("attend", compiled.attend, [q, k, v, logits, maskBuffer, gathered],
              Math.min(slots, GRID_WIDTH), Math.ceil(slots / GRID_WIDTH));
          run("attention-output", compiled.attentionOutput,
              [gathered, gate, condBuffer, blockWeights, actBuffer], tokens);
          run("transition", compiled.transition, [condBuffer, blockWeights, actBuffer], tokens);
        }
      }
      // ...and the readback rides the same submit.
      encoder.copyBufferToBuffer(actBuffer.buffer, 0, readback.buffer, 0, tokens * channels * 4);
      this.device.queue.submit([encoder.finish()]);
      validation.end("block stack");
      // 🔴 THE PAIR PROJECTIONS ARE RELEASED AFTER THE SUBMIT, NOT INSIDE THE
      // LOOP. They are pooled, so releasing one while a later block's encoded
      // pass still refers to it would hand the same buffer to that block's
      // upload.
      for (const projection of projections) projection.release();
      // ...the boundary that already synchronises, so the deferred scopes cost
      // nothing to read here.
      await validation.settle();
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return {
        output: result,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
