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
  // How many tokens one workgroup projects at once, and how many ways its
  // output range is split. `splits` must divide heads*dimension.
  // 🔴 NO DEFAULTS HERE. These used to fall back to their own constants, and a
  // caller that resolved them from the device limits and forgot to pass them
  // down got shaders tiling by four under a dispatch that divided the token
  // count by eight - half the tokens never projected, reported as a speedup.
  // A shape that does not say is a bug, so say so.
  const { tile, splits, outTile } = shape;
  for (const [name, value] of Object.entries({ tile, splits, outTile })) {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`the diffusion transformer's ${name} must be a positive integer,`
        + ` not ${value}: the caller resolves it from the device's workgroup storage`);
    }
  }
  if ((heads * dimension) % splits !== 0 || (channels * factor) % splits !== 0) {
    throw new Error(`splits ${splits} must divide both ${heads * dimension} and`
      + ` ${channels * factor}`);
  }
  const width = heads * dimension;
  const intermediate = channels * factor;
  const pairs = tokens * tokens;
  const outLanes = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[outTile];
  if (outLanes === undefined) {
    throw new Error(`outTile ${outTile} is not 1, 2 or 4`);
  }
  const outChunk = Math.min(intermediate, shape.outChunk ?? 384);
  if (intermediate % outChunk !== 0) {
    throw new Error(`outChunk ${outChunk} does not divide the intermediate ${intermediate}`);
  }
  const outPerLane = Math.ceil((channels / splits) / lanes);
  // 🔴 THE TOKEN TILE IS A VECTOR IN THE TWO WIDE PROJECTIONS TOO. Both read
  // one activation per token of the tile and multiply it by the same weight, so
  // one workgroup read and one vector multiply-add replace TILE of each - qkvg
  // went from 24 instructions a channel to nine.
  const tileWidth = Math.min(4, tile);
  const tileGroups = tile / tileWidth;
  const tileLanes = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[tileWidth];
  if (tileLanes === undefined || !Number.isInteger(tileGroups)) {
    throw new Error(`tile ${tile} is not 1, 2 or a multiple of 4`);
  }
  // Token t of the tile lives in group t/tileWidth, lane t%tileWidth. The
  // staged activations are indexed by group and channel; a register array only
  // by group.
  const lane = (t) => tileWidth === 1 ? "" : `.${"xyzw"[t % tileWidth]}`;
  const group = (t) => Math.floor(t / tileWidth);
  const stagedAt = (t) => `xt[${group(t)}u * C + c]${lane(t)}`;
  const accAt = (name, t) => `${name}[${group(t)}]${lane(t)}`;
  const overTile = (body) =>
    Array.from({ length: tile }, (_, t) => body(t)).join("\n      ");
  const overGroups = (body) =>
    Array.from({ length: tileGroups }, (_, g) => body(g)).join("\n      ");
  const stageTokens = `  for (var c = local; c < C; c += ${lanes}u) {
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      var value = 0.0;
      if (token < TOKENS) { value = xbuf[token * C + c]; }
      ${stagedAt(t)} = value;
    }`)}
  }
  workgroupBarrier();`;

  const overOutTile = (body) =>
    Array.from({ length: outTile }, (_, t) =>
      body(t, outTile === 1 ? "" : `.${"xyzw"[t]}`)).join("\n      ");


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

  // 🔴 THE WEIGHTS WERE THE BANDWIDTH, NOT THE ARITHMETIC. One workgroup per
  // token meant every workgroup read the block's whole weight set: 5.9M floats
  // for 2.4M multiply-adds, an arithmetic intensity of a quarter of a MAC per
  // byte where this device needs about twelve to be compute bound. A call read
  // 33 GB of weights that way, which at ~350 GB/s is the 107 ms it took.
  //
  // So the projection now runs over a TILE of tokens at once and each weight it
  // reads serves all of them. The AdaLN that produces the tile's input is
  // per-token and cannot tile - it is a different shape of work - so it moves
  // into its own pass and hands `x` over through a buffer.
  const adaln = `${common}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> xbuf: array<f32>;

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
    xbuf[token * C + c] = logistic(scale_value) * normalized + shift;
  }
}`;

  // 🔴 TILED OVER TOKENS AND SPLIT OVER OUTPUTS. Tiling alone would divide the
  // workgroup count by the tile size, and this GPU has already shown that
  // occupancy is the other half of the problem - so the output range is split
  // as well, which costs no extra weight traffic because each workgroup then
  // reads only its own slice of every matrix.
  const qkvg = `${common}
const TILE: u32 = ${tile}u;
const SPLITS: u32 = ${splits}u;
const SLICE: u32 = ${width / splits}u;
@group(0) @binding(0) var<storage, read> xbuf: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> gate: array<f32>;

var<workgroup> xt: array<${tileLanes}, ${tileGroups * channels}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  let out_begin = group.y * SLICE;
  let local = local_id.x;

${stageTokens}

  for (var o = local; o < SLICE; o += ${lanes}u) {
    let out = out_begin + o;
    var q_acc: array<${tileLanes}, ${tileGroups}>;
    var k_acc: array<${tileLanes}, ${tileGroups}>;
    var v_acc: array<${tileLanes}, ${tileGroups}>;
    var g_acc: array<${tileLanes}, ${tileGroups}>;
    ${overGroups((g) => `q_acc[${g}] = ${tileLanes}(weights[W_qBias + out]);
      k_acc[${g}] = ${tileLanes}(0.0);
      v_acc[${g}] = ${tileLanes}(0.0);
      g_acc[${g}] = ${tileLanes}(0.0);`)}
    // 🔴 FOUR WEIGHTS READ ONCE, USED TILE TIMES. That ratio is the whole point
    // of this kernel - and this stack reads all 566 MB of its weights once per
    // TILE of tokens, so it is also the whole of its cost. See the note above.
    for (var c = 0u; c < C; c += 1u) {
      let column = c * WIDTH + out;
      let wq = weights[W_qProjection + column];
      let wk = weights[W_kProjection + column];
      let wv = weights[W_vProjection + column];
      let wg = weights[W_gatingQuery + column];
      ${overGroups((g) => `{
        let value = xt[${g}u * C + c];
        q_acc[${g}] += value * wq;
        k_acc[${g}] += value * wk;
        v_acc[${g}] += value * wv;
        g_acc[${g}] += value * wg;
      }`)}
    }
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      if (token < TOKENS) {
        let index = token * WIDTH + out;
        q[index] = ${accAt("q_acc", t)};
        k[index] = ${accAt("k_acc", t)};
        v[index] = ${accAt("v_acc", t)};
        gate[index] = ${accAt("g_acc", t)};
      }
    }`)}
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
  // 🔴 ONE WORKGROUP A TOKEN, AND TILING IT WAS TRIED AND LOST. It reads the
  // whole 768x768 output projection to project one token - 2.4 MB each, 3.3 GB
  // a call, which is most of its 8.9 ms - so a tile of tokens looks like the
  // same halving that paid in ffw-out. Tiled by two it measured 69 ms against
  // 65 for the transformer as a whole, and adding a split of the output range
  // to restore the workgroup count left it at 70. Whatever ffw-out's traffic
  // was costing, this kernel's is not costing the same.
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

  // 🔴 THE TRANSITION IS THREE PASSES NOW, FOR THE SAME REASON THE PROJECTION
  // IS TWO. It carries 3.5M of the block's 5.9M weights, and one workgroup per
  // token read all of them per token. Tiling needs the tile's activations in
  // workgroup memory, and x plus the 1536-wide intermediate does not fit for
  // any tile at all - so the widening and the projection back become separate
  // kernels, each tiled as wide as its own scratch allows, and the intermediate
  // travels between them through a buffer. That buffer costs 360 KB of traffic
  // a block against the gigabytes of weights it saves.
  const ffwAdaln = `${common}
@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> xbuf: array<f32>;

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
    xbuf[token * C + c] =
      logistic(scale_value) * ((act[token * C + c] - act_mean) * act_inverse) + shift;
  }
}`;

  // The widening half: x (C) -> gate and value (INTERMEDIATE each) -> SwiGLU.
  const ffwWide = `${common}
const TILE: u32 = ${tile}u;
const SLICE: u32 = ${intermediate / splits}u;
@group(0) @binding(0) var<storage, read> xbuf: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> gated: array<f32>;

var<workgroup> xt: array<${tileLanes}, ${tileGroups * channels}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  let out_begin = group.y * SLICE;
  let local = local_id.x;
${stageTokens}

  let wide = INTERMEDIATE * 2u;
  for (var o = local; o < SLICE; o += ${lanes}u) {
    let i = out_begin + o;
    var gate_acc: array<${tileLanes}, ${tileGroups}>;
    var value_acc: array<${tileLanes}, ${tileGroups}>;
    ${overGroups((g) => `gate_acc[${g}] = ${tileLanes}(0.0);
      value_acc[${g}] = ${tileLanes}(0.0);`)}
    // 🔴 BLOCKED, gate half first - the same convention as the trunk's
    // transition and the opposite of triangle multiplication's interleave.
    for (var c = 0u; c < C; c += 1u) {
      let column = W_ffwTransition1 + c * wide;
      let wg = weights[column + i];
      let wv = weights[column + INTERMEDIATE + i];
      ${overGroups((g) => `{
        let value = xt[${g}u * C + c];
        gate_acc[${g}] += value * wg;
        value_acc[${g}] += value * wv;
      }`)}
    }
    ${overGroups((g) => `let swished${g} =
      gate_acc[${g}] / (${tileLanes}(1.0) + exp(-gate_acc[${g}])) * value_acc[${g}];`)}
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      if (token < TOKENS) {
        gated[token * INTERMEDIATE + i] = swished${group(t)}${lane(t)};
      }
    }`)}
  }
}`;

  // ...and the way back, INTERMEDIATE -> C, gated by the zero-init conditioning
  // and added to the residual.
  //
  // 🔴 THE LARGEST KERNEL OF THE DENOISER, AND IT IS ITS WEIGHT READS. Every
  // workgroup reads INTERMEDIATE x SLICE of transition2 to produce outTile
  // tokens - 2.4 MB for two of them, 3.4 GB a call, which at the 445 GB/s
  // tools/gpu/probe-alu.js measures for cached global reads is most of the
  // 20 ms it took. A bigger tile divides that traffic and used to cost
  // workgroup memory: outTile 4 wants 4 x INTERMEDIATE floats, 24 KB, and
  // measured 85 ms against outTile 2's 74. Staging a CHUNK of the intermediate
  // instead unties them - the accumulator survives the chunks - so the tile
  // buys its traffic back without spending residency for it.
  //
  // 🔴 AND THE TILE IS THE VECTOR. The inner loop multiplies one weight against
  // every token of the tile, so those tokens are exactly the axis to vectorise:
  // one workgroup read and one vector multiply-add where there were outTile of
  // each. See src/af3/transition-webgpu.js, which is the same kernel shape.
  const ffwOut = `${common}
const TILE: u32 = ${outTile}u;
const SLICE: u32 = ${channels / splits}u;
const OUT_CHUNK: u32 = ${outChunk}u;
@group(0) @binding(0) var<storage, read> gated: array<f32>;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;

// One chunk of the intermediate, holding the tile's tokens as a vector.
var<workgroup> wt: array<${outLanes}, ${outChunk}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  let out_begin = group.y * SLICE;
  let local = local_id.x;

  var acc: array<${outLanes}, ${outPerLane}>;
  for (var slot = 0u; slot < ${outPerLane}u; slot += 1u) { acc[slot] = ${outLanes}(0.0); }

  for (var chunk0 = 0u; chunk0 < INTERMEDIATE; chunk0 += OUT_CHUNK) {
    // ...before overwriting the chunk the previous iteration is still reading.
    workgroupBarrier();
    for (var i = local; i < OUT_CHUNK; i += ${lanes}u) {
      var staged: ${outLanes};
      ${overOutTile((t, at) => `{
        let token = base_token + ${t}u;
        var value = 0.0;
        if (token < TOKENS) { value = gated[token * INTERMEDIATE + chunk0 + i]; }
        ${outTile === 1 ? "staged" : `staged${at}`} = value;
      }`)}
      wt[i] = staged;
    }
    workgroupBarrier();

    var slot = 0u;
    for (var o = local; o < SLICE; o += ${lanes}u) {
      let c = out_begin + o;
      for (var i = 0u; i < OUT_CHUNK; i += 1u) {
        // ...read once, used by every token of the tile.
        acc[slot] += wt[i] * weights[W_ffwTransition2 + (chunk0 + i) * C + c];
      }
      slot += 1u;
    }
  }

  var slot = 0u;
  for (var o = local; o < SLICE; o += ${lanes}u) {
    let c = out_begin + o;
    // 🔴 THE ZERO-INIT GATE READS THE RAW CONDITIONING, not the normalised one.
    // Its weights are shared across the tile like every other matrix here; the
    // conditioning itself is 384 floats a token and stays in cache.
    var zero_gate = ${outLanes}(weights[W_ffwAdaptiveZeroCondBias + c]);
    for (var d = 0u; d < C_COND; d += 1u) {
      let w = weights[W_ffwAdaptiveZeroCondWeights + d * C + c];
      ${overOutTile((t, at) => `{
        let token = base_token + ${t}u;
        if (token < TOKENS) {
          ${outTile === 1 ? "zero_gate" : `zero_gate${at}`} += cond[token * C_COND + d] * w;
        }
      }`)}
    }
    // ...componentwise, so the gate is applied to every token of the tile at
    // once rather than through the scalar logistic the other kernels use.
    let contribution = acc[slot] / (${outLanes}(1.0) + exp(-zero_gate));
    ${overOutTile((t, at) => `{
      let token = base_token + ${t}u;
      if (token < TOKENS) {
        act[token * C + c] = act[token * C + c]
          + ${outTile === 1 ? "contribution" : `contribution${at}`};
      }
    }`)}
    slot += 1u;
  }
}`;

  return { normalisePair, pairLogitsFor, adaln, qkvg, attend, attentionOutput,
           ffwAdaln, ffwWide, ffwOut };
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
    // 🔴 FOUR AND TWO ARE MEASURED, AND MORE IS WORSE. Each kernel holds its
    // tile of activations in workgroup storage - the projection and the
    // widening keep `channels` floats a token, the way back `intermediate`,
    // twice as many - so a bigger tile buys weight traffic and costs
    // workgroups, and past here the workgroups are worth more:
    //
    //     tile     2   4   4   4   4   6   8      outTile 1 2 3 4 for tile 4
    //     outTile  2   1   2   3   4   2   2
    //     ms      91  75  74  79  85  79  83
    //
    // 🔴 SO RAISING maxComputeWorkgroupStorageSize BOUGHT NOTHING. It is asked
    // for in src/runtime/device.js and it does lift the ceiling from four
    // tokens to ten - but the ceiling was never what bound this, occupancy was.
    // The limit stays requested because it costs nothing and the cap below is
    // then real rather than notional; the numbers above are why the defaults do
    // not use the room.
    //
    // 🔴 AND THEY ARE RESOLVED BEFORE THE SHAPE, NOT AFTER. Leaving them to
    // default a second time inside the shader factory meant the SHADERS tiled
    // by four while the DISPATCH divided the token count by eight: every second
    // tile of tokens was simply never projected, and the bench reported it as a
    // 30% speedup. One resolution, passed down.
    const workgroupStorage = this.device.limits?.maxComputeWorkgroupStorageSize ?? 16384;
    const intermediate = channels * weights.transitionFactor;
    const fits = (perToken) => Math.max(1, Math.floor(workgroupStorage / (perToken * 4)));
    const tile = weights.tile ?? Math.min(4, fits(channels));
    const splits = weights.splits ?? 2;
    const outTile = weights.outTile ?? Math.min(2, tile, fits(intermediate));
    const shape = { tokens, channels, condChannels, pairChannels, heads, dimension,
                    factor: weights.transitionFactor, lanes: weights.lanes,
                    tile, splits, outTile, outChunk: weights.outChunk };
    const sources = createDiffusionTransformerShaders(shape, sample.offsets);
    // 🔴 THE LANE COUNT IS PART OF THE KEY. It is baked into every one of these
    // sources as a workgroup size, so a cache that ignored it would hand a
    // later run the pipeline compiled for a different width.
    const base = `af3-difftx:${tokens}:${channels}:${condChannels}:${pairChannels}`
      + `:${heads}:${dimension}:${weights.transitionFactor}:${perSuper}`
      + `:${shape.lanes ?? "default"}:${tile}:${splits}:${outTile}:${weights.outChunk ?? "d"}`;
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
      // The AdaLN pass hands the projection its input through this.
      const xBuffer = keep(this.allocator.allocate("difftx.x", tokens * channels * 4,
        storage));
      const gatedBuffer = keep(this.allocator.allocate("difftx.gated",
        tokens * channels * weights.transitionFactor * 4, storage));
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
          run("adaln", compiled.adaln, [actBuffer, condBuffer, blockWeights, xBuffer], tokens);
          run("qkvg", compiled.qkvg, [xBuffer, blockWeights, q, k, v, gate],
              Math.ceil(tokens / tile), splits);
          const slots = tokens * heads;
          run("attend", compiled.attend, [q, k, v, logits, maskBuffer, gathered],
              Math.min(slots, GRID_WIDTH), Math.ceil(slots / GRID_WIDTH));
          run("attention-output", compiled.attentionOutput,
              [gathered, gate, condBuffer, blockWeights, actBuffer], tokens);
          run("ffw-adaln", compiled.ffwAdaln,
              [condBuffer, blockWeights, actBuffer, xBuffer], tokens);
          run("ffw-wide", compiled.ffwWide, [xBuffer, blockWeights, gatedBuffer],
              Math.ceil(tokens / tile), splits);
          run("ffw-out", compiled.ffwOut,
              [gatedBuffer, condBuffer, blockWeights, actBuffer],
              Math.ceil(tokens / outTile), splits);
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
