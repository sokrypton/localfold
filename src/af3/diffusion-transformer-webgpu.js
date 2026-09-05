import { concatenateAs } from "../runtime/float16.js";
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
import { noteAllocation, noteDestroy } from "../runtime/device-memory.js";
import { GpuMemoryBudgetError, noteResidencyRefused, residencyAllowed }
  from "../runtime/device-memory.js";
import { releaseResidentWeights, residentWeightBuffer } from "../runtime/resident.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { DeferredValidation } from "../runtime/validation.js";
import { releaseWeights } from "./weights.js";
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
 * 🔴 THE UPLOAD WAS THE FLOOR, NOT THE ARITHMETIC. A block is ~26 MB, so the
 * loop wrote ~630 MB to the device per call - and at eight tokens, where the
 * matmuls are nothing, twenty-four blocks still cost 174 ms, which is that
 * write at about 3.6 GB/s. A 200-step fold did it two hundred times over
 * weights that never change.
 *
 * 🔴 SO THIS TRADES DEVICE MEMORY FOR IT, DELIBERATELY - ~630 MB at f32, the
 * same order as the checkpoint itself. It goes through src/runtime/resident.js
 * rather than a WeakMap of its own so that ONE call hands back every weight
 * buffer on a device, which is what the budget fallback below needs.
 */
function residentBlockBuffer(device, block, pack, variant = "") {
  return residentWeightBuffer(device, block, "difftx.block.resident", () => pack().data, variant);
}

/**
 * @param {"f32"|"f16"} precision the element the packed buffer holds. Offsets
 *   are in elements and do not depend on it; the shader must be built for the
 *   same word or it reads half the values at twice the stride.
 */
/**
 * The packing offsets alone, without building the buffer.
 *
 * 🔴 THE SHADERS NEED THE OFFSETS AND NOTHING ELSE, and `packBlockWeights` was
 * being called on a sample block to get them - concatenating 31.5 MiB, and in
 * f16 converting it, to read a dozen numbers that are a running sum of lengths.
 */
export function blockWeightOffsets(block) {
  const offsets = {};
  let total = 0;
  for (const name of BLOCK_ORDER) {
    if (block[name] === undefined) throw new Error(`diffusion block missing ${name}`);
    offsets[name] = total;
    total += block[name].length;
  }
  return offsets;
}

export function packBlockWeights(block, precision = "f32") {
  const offsets = {};
  let total = 0;
  for (const name of BLOCK_ORDER) {
    if (block[name] === undefined) throw new Error(`diffusion block missing ${name}`);
    offsets[name] = total;
    total += block[name].length;
  }
  const data = concatenateAs(precision, total, (target) => {
    for (const name of BLOCK_ORDER) target.set(block[name], offsets[name]);
  });
  return { data, offsets };
}

/**
 * How much of the intermediate `ffw-out` stages at once.
 *
 * 🔴 RESOLVED IN ONE PLACE BECAUSE TWO PLACES NEED IT AND THEY MUST AGREE. The
 * factory sizes its workgroup array from this; the caller sizes `outTile` from
 * it, since the tile only fits if the chunk is what is staged. Defaulting it
 * twice is the shape of mistake the note on `tile` and `splits` below records
 * as half the tokens going unprojected.
 */
export const DEFAULT_OUT_CHUNK = 384;
export const resolveOutChunk = (intermediate, requested) =>
  Math.min(intermediate, requested ?? DEFAULT_OUT_CHUNK);

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
  const outChunk = resolveOutChunk(intermediate, shape.outChunk);
  if (intermediate % outChunk !== 0) {
    throw new Error(`outChunk ${outChunk} does not divide the intermediate ${intermediate}`);
  }
  const outPerLane = Math.ceil((channels / splits) / lanes);
  // 🔴 THE TOKEN TILE IS A VECTOR IN THE TWO WIDE PROJECTIONS TOO. Both read
  // one activation per token of the tile and multiply it by the same weight, so
  // one workgroup read and one vector multiply-add replace TILE of each - qkvg
  // went from 24 instructions a channel to nine.
  // 🔴 THE WEIGHT BUFFER IS A STORAGE FORMAT, AND THIS IS THE BIGGEST ONE THERE
  // IS. `difftx.block.resident` was 756 MiB when it was f32 - 68% of what a
  // fold then held, more than the trunk's entire pairformer - because 24 blocks
  // of a 768-channel transformer each keep 31.5 MiB resident for the model's
  // lifetime. In f16 it is 378 MiB, and it is still the largest single tensor a
  // fold holds; a fold now holds 798 MiB in total.
  //
  // 🔴 AND HERE IT BUYS TIME AS WELL, WHICH IS TRUE OF NOWHERE ELSE IN EITHER
  // MODEL. This comment said "it buys no time" for a while, on the strength of
  // AF3's trunk, where halving resident weight bytes measured 377 ms against
  // 378. That reasoning does not transfer: this stack STREAMS its whole weight
  // set once per token tile rather than keeping it in cache, so the bytes are
  // the cost. Measured on bench-diffusion-transformer.js: 48 -> 41 ms at 59
  // tokens and 103 -> 89 at 150, and a denoiser call 104-114 -> 86-91.
  //
  // Reads are widened at the point of use and the arithmetic is f32 throughout.
  const weightPrecision = shape.weightPrecision ?? "f32";
  if (!["f32", "f16"].includes(weightPrecision)) {
    throw new RangeError(`unknown diffusion transformer weight precision ${weightPrecision}`);
  }
  const weight16 = weightPrecision === "f16";
  const wf = (e) => (weight16 ? `f32(${e})` : e);

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
  const accAt = (name, t) => `${name}[${group(t)}]${lane(t)}`;
  const overTile = (body) =>
    Array.from({ length: tile }, (_, t) => body(t)).join("\n      ");
  const overGroups = (body) =>
    Array.from({ length: tileGroups }, (_, g) => body(g)).join("\n      ");
  // 🔴 THE CHANNELS ARE STAGED IN CHUNKS SO THAT THE TOKEN TILE CAN GROW, and
  // the tile is the only thing that matters here: this stack streams all 566 MB
  // of its weights once per tile, so at four tokens a 240-token call makes
  // sixty passes over them. Holding TILE x C activations is what capped it -
  // 12 KB at four tokens, 24 at eight, where residency collapses and tile 8
  // measured 343 ms against 320. A chunk unties the two.
  const channelChunk = Math.min(channels, shape.channelChunk ?? 256);
  if (channels % channelChunk !== 0) {
    throw new Error(`channelChunk ${channelChunk} does not divide ${channels}`);
  }
  const stagedAt = (t) => `xt[${group(t)}u * CHANNEL_CHUNK + cc]${lane(t)}`;
  const stageChunk = `    workgroupBarrier();
    for (var cc = local; cc < CHANNEL_CHUNK; cc += ${lanes}u) {
      ${overTile((t) => `{
        let token = base_token + ${t}u;
        var value = 0.0;
        if (token < TOKENS) { value = xbuf[token * C + c0 + cc]; }
        ${stagedAt(t)} = value;
      }`)}
    }
    workgroupBarrier();`;

  /**
   * 🔴 ONE OUTPUT AN INVOCATION, WHICH IS WHAT MAKES THE TOKEN TILE AFFORDABLE.
   * The accumulators are (matrices x tile groups x outputs a lane) vectors, and
   * every one of them is live across the whole channel loop - so a second
   * output a lane doubles the registers, and at eight tokens that was enough to
   * spill: tile 8 measured 542 ms against tile 4's 332 at 240 tokens with two
   * outputs a lane, and the traffic model says it should have been ~200.
   *
   * So each kernel splits its own output range to exactly `lanes` wide rather
   * than sharing one `splits`. Their ranges differ - the attention projection
   * is heads*dimension, the widening is the doubled intermediate - so one
   * number cannot make both exact.
   */
  const splitFor = (range, name) => {
    if (range % lanes !== 0) {
      throw new Error(`${name} ${range} is not a multiple of ${lanes} lanes`);
    }
    return range / lanes;
  };
  const qkvgSplits = splitFor(width, "the attention projection's width");
  const wideSplits = splitFor(intermediate, "the transition's intermediate");
  const outSplits = splitFor(channels, "the channel count");



  const common = `${weight16 ? "enable f16;\n" : ""}
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
  // 🔴 THE HEADS ARE CONTIGUOUS IN THE PROJECTION, SO THEY ARE THE VECTOR. This
  // is the largest kernel of the stack once a protein is any size - 27 of 108
  // ms at 240 tokens, because it is quadratic in tokens where the token
  // projections are linear. It used to loop heads outside channels, re-reading
  // the normalised pair row for every one of the sixteen and reading one weight
  // per multiply-add: 48 instructions to buy 16. Channels outside, heads as
  // four vec4s, it is nine.
  const headVectors = heads / 4;
  const overHeadVectors = (body) =>
    Array.from({ length: headVectors }, (_, h) => body(h)).join("\n    ");
  const pairLogitsFor = (inner, perSuper) => `${common}
const INNER: u32 = ${inner}u;
const PER_SUPER: u32 = ${perSuper}u;
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
// ...as vec4, which is why the column base must be a multiple of four: it is
// c * PER_SUPER * HEADS + INNER * HEADS, and HEADS is sixteen.
@group(0) @binding(1) var<storage, read> projection: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * C_PAIR;
  ${overHeadVectors((h) => `var total${h} = vec4<f32>(0.0);`)}
  for (var c = 0u; c < C_PAIR; c += 1u) {
    // ...read once, used by every head.
    let x = normalized[base + c];
    let column = (c * PER_SUPER * HEADS + INNER * HEADS) / 4u;
    ${overHeadVectors((h) => `total${h} += x * projection[column + ${h}u];`)}
  }
  ${overHeadVectors((h) => Array.from({ length: 4 }, (_, l) =>
    `logits[(${h * 4 + l}u) * PAIRS + row] = total${h}.${"xyzw"[l]};`).join("\n    "))}
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
  /**
   * AdaLN: normalise the activation, normalise and project the conditioning
   * into a scale and a shift, and apply them. Both halves of a block use it,
   * with different weights, so it is generated twice from here.
   *
   * 🔴 ONE WEIGHT READ SERVES THE WHOLE TILE, which is the point of tiling it.
   * One workgroup a token read both 384x768 conditioning matrices per token -
   * 2.4 MB each way, 13.6 GB a call at 240 tokens - and the two AdaLN passes
   * were 16.5 ms of a 104 ms stack there. The LayerNorms stay per token,
   * sequential over the tile, because their reductions are not shared.
   */
  const conditionedNorm = (prefix, bindings) => `${common}
const TILE: u32 = ${tile}u;
${bindings}

var<workgroup> cond_norm: array<${tileLanes}, ${tileGroups * condChannels}>;
var<workgroup> reduce_a: array<f32, ${lanes}>;
var<workgroup> act_means: array<f32, ${tile}>;
var<workgroup> act_inverses: array<f32, ${tile}>;

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
  let base_token = group.x * TILE;
  if (base_token >= TOKENS) { return; }
  let local = local_id.x;

  // 🔴 TWO-PASS VARIANCE, and no scale or offset on the activation's own norm.
  // A token past the end is clamped rather than skipped: every lane has to
  // reach the barriers, and its lane of the vector is dropped at the write.
  for (var t = 0u; t < TILE; t += 1u) {
    let token = min(base_token + t, TOKENS - 1u);
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
    if (local == 0u) {
      act_means[t] = act_mean;
      act_inverses[t] = act_inverse;
    }
    workgroupBarrier();
    for (var c = local; c < C_COND; c += ${lanes}u) {
      let value = (cond[token * C_COND + c] - cond_mean) * cond_inverse
        * ${wf(`weights[W_${prefix}SingleCondLayerNormScale + c]`)};
      ${Array.from({ length: tile }, (_, t) =>
        `if (t == ${t}u) { cond_norm[${group(t)}u * C_COND + c]${lane(t)} = value; }`)
        .join("\n      ")}
    }
    workgroupBarrier();
  }

  for (var c = local; c < C; c += ${lanes}u) {
    ${overGroups((g) =>
      `var scale${g} = ${tileLanes}(${wf(`weights[W_${prefix}SingleCondScaleBias + c]`)});
    var shift${g} = ${tileLanes}(0.0);`)}
    for (var d = 0u; d < C_COND; d += 1u) {
      let ws = ${wf(`weights[W_${prefix}SingleCondScaleWeights + d * C + c]`)};
      let wb = ${wf(`weights[W_${prefix}SingleCondBias + d * C + c]`)};
      ${overGroups((g) => `{
        let cn = cond_norm[${g}u * C_COND + d];
        scale${g} += cn * ws;
        shift${g} += cn * wb;
      }`)}
    }
    ${overGroups((g) => `let gated${g} =
      ${tileLanes}(1.0) / (${tileLanes}(1.0) + exp(-scale${g}));`)}
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      if (token < TOKENS) {
        let normalized = (act[token * C + c] - act_means[${t}u]) * act_inverses[${t}u];
        xbuf[token * C + c] =
          gated${group(t)}${lane(t)} * normalized + shift${group(t)}${lane(t)};
      }
    }`)}
  }
}`;

  const adaln = conditionedNorm("", `@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(3) var<storage, read_write> xbuf: array<f32>;`);

  const qkvg = `${common}
const TILE: u32 = ${tile}u;
const SPLITS: u32 = ${splits}u;
@group(0) @binding(0) var<storage, read> xbuf: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> gate: array<f32>;

const CHANNEL_CHUNK: u32 = ${channelChunk}u;
var<workgroup> xt: array<${tileLanes}, ${tileGroups * channelChunk}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  let local = local_id.x;
  let out = group.y * ${lanes}u + local;

  ${overGroups((g) => `var q${g} = ${tileLanes}(${wf(`weights[W_qBias + out]`)});   // only q has a bias
  var k${g} = ${tileLanes}(0.0);
  var v${g} = ${tileLanes}(0.0);
  var g${g} = ${tileLanes}(0.0);`)}

  for (var c0 = 0u; c0 < C; c0 += CHANNEL_CHUNK) {
${stageChunk}
    // 🔴 FOUR WEIGHTS READ ONCE, USED TILE TIMES. That ratio is the whole point
    // of this kernel - and this stack streams all 566 MB of its weights once
    // per TILE of tokens, so it is also the whole of its cost.
    for (var cc = 0u; cc < CHANNEL_CHUNK; cc += 1u) {
      let column = (c0 + cc) * WIDTH + out;
      let wq = ${wf(`weights[W_qProjection + column]`)};
      let wk = ${wf(`weights[W_kProjection + column]`)};
      let wv = ${wf(`weights[W_vProjection + column]`)};
      let wg = ${wf(`weights[W_gatingQuery + column]`)};
      ${overGroups((g) => `{
        let x = xt[${g}u * CHANNEL_CHUNK + cc];
        q${g} += x * wq;
        k${g} += x * wk;
        v${g} += x * wv;
        g${g} += x * wg;
      }`)}
    }
  }

  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < TOKENS) {
      let index = token * WIDTH + out;
      q[index] = q${group(t)}${lane(t)};
      k[index] = k${group(t)}${lane(t)};
      v[index] = v${group(t)}${lane(t)};
      gate[index] = g${group(t)}${lane(t)};
    }
  }`)}
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
  // 🔴 A TILE OF TOKENS AND ONE OUTPUT AN INVOCATION. This reads the whole
  // 768x768 output projection to project a token - 2.4 MB each, 3.3 GB a call -
  // so the tile is what divides it. Tiled ALONE it measured worse (69 ms
  // against 65 for the stack), because it also divides the workgroups and this
  // kernel had one per token; splitting the output range to exactly `lanes`
  // wide puts them back, and leaves each invocation a single accumulator.
  const attentionOutput = `${common}
const TILE: u32 = ${tile}u;
@group(0) @binding(0) var<storage, read> gathered: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> cond: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(4) var<storage, read_write> act: array<f32>;

// The tile's tokens as one vector, so one weight read serves all of them - and
// then, once the projection loop is done with it, the CONDITIONING.
//
// 🔴 THE ZERO-GATE LOOP BELOW READ cond FROM GLOBAL, ONCE PER TOKEN PER
// CONDITIONING CHANNEL, ON EVERY LANE. It is indexed by (token, d) and not by
// the channel a lane owns, so all 256 lanes wanted the same TILE x C_COND
// floats: one weight read, TILE global reads and TILE scalar multiply-adds a
// step, which at width 768 and C_COND 384 is about 60% of this kernel. Staged
// as a vector over the tile it is one weight read, one workgroup read and one
// vector multiply-add.
//
// 🔴 AND IT REUSES THESE SLOTS RATHER THAN TAKING MORE. gated is dead once
// the projection loop has read it, and it is the larger of the two uses at
// WIDTH against C_COND - so the conditioning costs nothing and this kernel's
// residency is unchanged. ffw-out does the same thing for the same reason.
var<workgroup> gated: array<${tileLanes}, ${tileGroups * Math.max(width, condChannels)}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  if (base_token >= TOKENS) { return; }
  let local = local_id.x;
  let c = group.y * ${lanes}u + local;

  for (var w = local; w < WIDTH; w += ${lanes}u) {
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      var value = 0.0;
      if (token < TOKENS) {
        let index = token * WIDTH + w;
        value = gathered[index] * logistic(gate[index]);
      }
      gated[${group(t)}u * WIDTH + w]${lane(t)} = value;
    }`)}
  }
  workgroupBarrier();

  ${overGroups((g) => `var projected${g} = ${tileLanes}(0.0);`)}
  for (var w = 0u; w < WIDTH; w += 1u) {
    // ...read once, used by every token of the tile.
    let weight = ${wf(`weights[W_Transition2 + w * C + c]`)};
    ${overGroups((g) => `projected${g} += gated[${g}u * WIDTH + w] * weight;`)}
  }
  // 🔴 THE ZERO-INIT GATE READS THE RAW CONDITIONING, not the normalised one.
  // ...staged into the slots the projection loop has finished with. The barrier
  // before is what makes reusing them safe.
  workgroupBarrier();
  for (var d = local; d < C_COND; d += ${lanes}u) {
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      var value = 0.0;
      if (token < TOKENS) { value = cond[token * C_COND + d]; }
      gated[${group(t)}u * C_COND + d]${lane(t)} = value;
    }`)}
  }
  workgroupBarrier();
  ${overGroups((g) => `var zero${g} = ${tileLanes}(${wf(`weights[W_AdaptiveZeroCondBias + c]`)});`)}
  for (var d = 0u; d < C_COND; d += 1u) {
    let w = ${wf(`weights[W_AdaptiveZeroCondWeights + d * C + c]`)};
    ${overGroups((g) => `zero${g} += gated[${g}u * C_COND + d] * w;`)}
  }
  ${overGroups((g) => `let contribution${g} = projected${g}
    / (${tileLanes}(1.0) + exp(-zero${g}));`)}
  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < TOKENS) {
      act[token * C + c] = act[token * C + c] + contribution${group(t)}${lane(t)};
    }
  }`)}
}`;

  const ffwAdaln = conditionedNorm("ffw",
    `@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> xbuf: array<f32>;`);

  // The widening half: x (C) -> gate and value (INTERMEDIATE each) -> SwiGLU.
  const ffwWide = `${common}
const TILE: u32 = ${tile}u;
@group(0) @binding(0) var<storage, read> xbuf: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read_write> gated: array<f32>;

const CHANNEL_CHUNK: u32 = ${channelChunk}u;
var<workgroup> xt: array<${tileLanes}, ${tileGroups * channelChunk}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  let local = local_id.x;
  let i = group.y * ${lanes}u + local;

  ${overGroups((g) => `var gate_acc${g} = ${tileLanes}(0.0);
  var value_acc${g} = ${tileLanes}(0.0);`)}

  let wide = INTERMEDIATE * 2u;
  for (var c0 = 0u; c0 < C; c0 += CHANNEL_CHUNK) {
${stageChunk}
    // 🔴 BLOCKED, gate half first - the same convention as the trunk's
    // transition and the opposite of triangle multiplication's interleave.
    for (var cc = 0u; cc < CHANNEL_CHUNK; cc += 1u) {
      let column = W_ffwTransition1 + (c0 + cc) * wide;
      let wg = ${wf(`weights[column + i]`)};
      let wv = ${wf(`weights[column + INTERMEDIATE + i]`)};
      ${overGroups((g) => `{
        let x = xt[${g}u * CHANNEL_CHUNK + cc];
        gate_acc${g} += x * wg;
        value_acc${g} += x * wv;
      }`)}
    }
  }

  ${overGroups((g) => `let swished${g} = gate_acc${g}
    / (${tileLanes}(1.0) + exp(-gate_acc${g})) * value_acc${g};`)}
  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < TOKENS) {
      gated[token * INTERMEDIATE + i] = swished${group(t)}${lane(t)};
    }
  }`)}
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
  const outVector = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[Math.min(4, outTile)];
  const outWidth = Math.min(4, outTile);
  const outGroups = outTile / outWidth;
  if (outVector === undefined || !Number.isInteger(outGroups)) {
    throw new Error(`outTile ${outTile} is not 1, 2 or a multiple of 4`);
  }
  const outLane = (t) => outWidth === 1 ? "" : `.${"xyzw"[t % outWidth]}`;
  const outGroup = (t) => Math.floor(t / outWidth);
  const overOutTile = (body) =>
    Array.from({ length: outTile }, (_, t) => body(t)).join("\n    ");
  const overOutGroups = (body) =>
    Array.from({ length: outGroups }, (_, g) => body(g)).join("\n    ");
  const ffwOut = `${common}
const TILE: u32 = ${outTile}u;
const OUT_CHUNK: u32 = ${outChunk}u;
@group(0) @binding(0) var<storage, read> gated: array<f32>;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;

// One chunk of the intermediate, holding the tile's tokens as a vector - and
// then, once the chunk loop is done with it, the CONDITIONING.
//
// 🔴 THE ZERO-GATE LOOP READ cond FROM GLOBAL, ONCE PER TOKEN PER CHANNEL,
// ON EVERY LANE. It is indexed by (token, d) and not by c, so all 256 lanes
// of a workgroup want the same 4 x C_COND values - and at C_COND 384 that loop
// was one weight read, four global conditioning reads and four scalar
// multiply-adds a step, about 43% of this kernel's instructions. Staged as a
// vector over the tile it is one weight read, one workgroup read and one vector
// multiply-add: 384 x 3 where it was 384 x 9.
//
// 🔴 AND IT COSTS NO WORKGROUP MEMORY, because wt is dead by then. The chunk
// loop has finished reading it before the gate is computed, so the same slots
// carry the conditioning; the array is sized for whichever use is larger. A
// second array would have taken this kernel from 6 KiB to 12 - two workgroups
// a core against five - which is the trade this repository has lost to four
// times.
var<workgroup> wt: array<${outVector}, ${outGroups * Math.max(outChunk, condChannels)}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  if (base_token >= TOKENS) { return; }
  let local = local_id.x;
  let c = group.y * ${lanes}u + local;

  ${overOutGroups((g) => `var acc${g} = ${outVector}(0.0);`)}

  for (var chunk0 = 0u; chunk0 < INTERMEDIATE; chunk0 += OUT_CHUNK) {
    // ...before overwriting the chunk the previous iteration is still reading.
    workgroupBarrier();
    for (var i = local; i < OUT_CHUNK; i += ${lanes}u) {
      ${overOutTile((t) => `{
        let token = base_token + ${t}u;
        var value = 0.0;
        if (token < TOKENS) { value = gated[token * INTERMEDIATE + chunk0 + i]; }
        wt[${outGroup(t)}u * OUT_CHUNK + i]${outLane(t)} = value;
      }`)}
    }
    workgroupBarrier();

    for (var i = 0u; i < OUT_CHUNK; i += 1u) {
      // ...read once, used by every token of the tile.
      let weight = ${wf(`weights[W_ffwTransition2 + (chunk0 + i) * C + c]`)};
      ${overOutGroups((g) => `acc${g} += wt[${g}u * OUT_CHUNK + i] * weight;`)}
    }
  }

  // 🔴 THE ZERO-INIT GATE READS THE RAW CONDITIONING, not the normalised one.
  // ...the conditioning into the slots the chunk loop has finished with. The
  // barrier before is what makes reusing them safe; the one after is the
  // ordinary staging barrier.
  workgroupBarrier();
  for (var d = local; d < C_COND; d += ${lanes}u) {
    ${overOutTile((t) => `{
      let token = base_token + ${t}u;
      var value = 0.0;
      if (token < TOKENS) { value = cond[token * C_COND + d]; }
      wt[${outGroup(t)}u * C_COND + d]${outLane(t)} = value;
    }`)}
  }
  workgroupBarrier();

  ${overOutGroups((g) => `var zero${g} = ${outVector}(${wf(`weights[W_ffwAdaptiveZeroCondBias + c]`)});`)}
  for (var d = 0u; d < C_COND; d += 1u) {
    let w = ${wf(`weights[W_ffwAdaptiveZeroCondWeights + d * C + c]`)};
    ${overOutGroups((g) => `zero${g} += wt[${g}u * C_COND + d] * w;`)}
  }
  ${overOutGroups((g) => `let contribution${g} = acc${g}
    / (${outVector}(1.0) + exp(-zero${g}));`)}
  ${overOutTile((t) => `{
    let token = base_token + ${t}u;
    if (token < TOKENS) {
      act[token * C + c] = act[token * C + c] + contribution${outGroup(t)}${outLane(t)};
    }
  }`)}
}`;

  return { normalisePair, pairLogitsFor, adaln, qkvg, attend, attentionOutput,
           ffwAdaln, ffwWide, ffwOut, qkvgSplits, wideSplits, outSplits };
}

export class Af3DiffusionTransformerGpu {
  /**
   * The normalised pair conditioning, kept across calls.
   *
   * 🔴 NEITHER THE PAIR CONDITIONING NOR ITS LAYERNORM READS THE NOISE LEVEL.
   * The stack's twenty-four blocks all read one normalised pair tensor, built
   * by a single pass at the top of the call - and a sampler was uploading the
   * unnormalised tokens^2 x 128 tensor and running that pass again on every
   * step, for the identical bytes. At 59 tokens that is 1.8 MB across the bus
   * and a 3481 x 128 layer norm, two hundred times; at 256 tokens, 34 MB.
   *
   * Keyed on the array's identity, which is the same question the diffusion
   * head asks of its own pair cache: a new fold brings a new array. The buffer
   * lives outside the pooled allocator, because a pooled one is recycled when
   * the call that took it ends.
   */
  #pairNorm;

  /**
   * @param {{residentWeights?: boolean}} [options] whether the 24 blocks' packed
   *   weights stay on the device between calls. See the note at the upload; a
   *   budget refusal turns this off on its own.
   */
  constructor(device, options = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
    this.residentWeights = (options.residentWeights ?? true) && residencyAllowed(device);
  }

  /** Give back the normalised pair tensor. Callers that keep an instance own this. */
  dispose() {
    if (this.#pairNorm === undefined) return;
    this.#pairNorm.buffer.destroy();
    noteDestroy(this.device, this.#pairNorm.bytes, "difftx.pair-norm");
    this.#pairNorm = undefined;
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
    try {
      return await this.#runBlocks(act, cond, pairCond, mask, tokens, weights);
    } catch (error) {
      if (!(error instanceof GpuMemoryBudgetError) || !this.residentWeights) throw error;
      // The pairformer's reasoning exactly; see the note on its run(). The
      // refusal arrives with some blocks already resident and their command
      // buffers in flight, so the call is abandoned rather than patched up.
      await this.device.queue.onSubmittedWorkDone();
      const reclaimed = releaseResidentWeights(this.device);
      this.residentWeights = false;
      noteResidencyRefused(this.device);
      this.degradedTo = `uploading weights per call (${(reclaimed / (1024 * 1024)).toFixed(0)}`
        + ` MiB reclaimed): ${error.message}`;
      return await this.#runBlocks(act, cond, pairCond, mask, tokens, weights);
    }
  }

  /**
   * Everything about a run that depends only on the shape: the tiles, the
   * shaders and their pipelines.
   *
   * 🔴 SEPARATED BECAUSE IT WAS DOING PER-CALL WORK THAT DEPENDS ON NOTHING.
   * It read the packing offsets by calling `packBlockWeights` on a sample block
   * - concatenating 31.5 MiB, and in f16 converting it, to obtain a dozen
   * numbers that are a running sum of lengths - and it did that on EVERY
   * denoiser call, eight times a fold. `blockWeightOffsets` is the same numbers
   * without the buffer: a steady call goes 86-89 ms to 83-85, its transformer
   * 43 to 40.
   *
   * Compiling this early was tried too and is not worth keeping: measured from
   * a fold, the pipelines are ready in 6 ms, so compilation is not what makes
   * the first call several times a steady one. That is the weight conversion,
   * and docs/AF3.md records why it cannot be moved either.
   */
  async #compile(tokens, weights) {
    const channels = weights.channels;
    const condChannels = weights.condChannels;
    const pairChannels = weights.pairChannels;
    const heads = weights.heads;
    const dimension = weights.dimension;
    const perSuper = weights.blocksPerSuperBlock;
    const width = heads * dimension;
    const pairs = tokens * tokens;
    const sampleOffsets = blockWeightOffsets(weights.superBlocks[0].blocks[0]);
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
    // 🔴 FOUR, AND RE-MEASURED AFTER THE CONDITIONING WAS STAGED. Eight used to
    // lose partly because two kernels' zero-gate loops were TILE global reads a
    // step; staged, those loops cost the same whatever the tile, so the reason
    // for the old answer had gone even though the answer had not. Re-swept on
    // the whole transformer at 150 tokens: 4 -> 103, 105 ms; 8 -> 106, 113.
    const tile = weights.tile ?? Math.min(4, fits(channels));
    const splits = weights.splits ?? 2;
    // 🔴 THE CHUNK IS WHAT `ffw-out` STAGES, NOT THE WHOLE INTERMEDIATE, AND
    // THIS RULE STILL SAID OTHERWISE. It read `Math.min(2, tile,
    // fits(intermediate))`: a hard cap of two, under a sizing term that assumed
    // a workgroup held outTile x 1536 floats. Chunking removed that - it holds
    // outTile x outChunk, 6 KiB at four - and the kernel's own comment says so
    // ("staging a CHUNK of the intermediate instead unties them"), but the cap
    // was never lifted, so the measurement that set it (outTile 4 at 85 ms
    // against 2's 74) was left standing against a kernel that no longer
    // existed. Re-measured on the whole transformer with the chunking in place,
    // as medians of repeated runs: at 150 tokens **2 -> 138 ms, 4 -> 128, 8 ->
    // 132**; at 59 tokens they tie (63-65 either way). Four it is.
    const outChunk = resolveOutChunk(intermediate, weights.outChunk);
    const outTile = weights.outTile ?? Math.min(4, tile, fits(outChunk));
    // 🔴 f16 WHEREVER THE DEVICE HAS IT, FOR THE MEMORY. See the note in the
    // shader factory: this is the largest resident tensor a fold holds.
    const weightPrecision = weights.weightPrecision
      ?? (this.device.features?.has("shader-f16") ? "f16" : "f32");
    const shape = { tokens, channels, condChannels, pairChannels, heads, dimension,
                    factor: weights.transitionFactor, lanes: weights.lanes,
                    tile, splits, outTile, outChunk, weightPrecision,
                    channelChunk: weights.channelChunk };
    const sources = createDiffusionTransformerShaders(shape, sampleOffsets);
    // 🔴 THE LANE COUNT IS PART OF THE KEY. It is baked into every one of these
    // sources as a workgroup size, so a cache that ignored it would hand a
    // later run the pipeline compiled for a different width.
    const base = `af3-difftx:${tokens}:${channels}:${condChannels}:${pairChannels}`
      + `:${heads}:${dimension}:${weights.transitionFactor}:${perSuper}`
      + `:${shape.lanes ?? "default"}:${tile}:${splits}:${outTile}:${outChunk}`
      + `:${weights.channelChunk ?? "d"}:${weightPrecision}`;
    // 🔴 AWAITED TOGETHER, NOT ONE AT A TIME. `createComputePipelineAsync`
    // compiles off the main thread, so a loop that awaits each one in turn
    // serialises eleven compilations that could overlap - and this stack's
    // shaders are the largest in the model. It is paid once per process and
    // lands inside the FIRST denoiser call, which bench-head.js reports at 606
    // ms against a steady 86. The cache stores the promise, not the pipeline,
    // so asking for the same key twice is still one compilation.
    const compiled = { pairLogits: [] };
    const pending = [];
    for (const [name, source] of Object.entries(sources)) {
      // ...the factory also returns the split counts the dispatch needs, which
      // are numbers rather than shaders.
      if (name === "pairLogitsFor" || typeof source !== "string") continue;
      pending.push(this.pipelines.get(`${base}:${name}`, source)
        .then((pipeline) => { compiled[name] = pipeline; }));
    }
    for (let inner = 0; inner < perSuper; inner += 1) {
      const at = inner;
      pending.push(this.pipelines.get(
        `${base}:pair-logits:${at}`, sources.pairLogitsFor(at, perSuper))
        .then((pipeline) => { compiled.pairLogits[at] = pipeline; }));
    }
    await Promise.all(pending);
    return { channels, condChannels, pairChannels, heads, dimension, perSuper,
             width, pairs, shape, sources, compiled, tile, splits, outTile, outChunk,
             weightPrecision };
  }

  async #runBlocks(act, cond, pairCond, mask, tokens, weights) {
    const {
      channels, condChannels, pairChannels, heads, dimension, perSuper,
      width, pairs, shape, sources, compiled, tile, splits, outTile, outChunk,
      weightPrecision,
    } = await this.#compile(tokens, weights);
    if (act.length !== tokens * channels) {
      throw new Error(`act has ${act.length} elements; expected ${tokens * channels}`);
    }

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const actBuffer = keep(this.allocator.upload("difftx.act", Float32Array.from(act),
        storage | GPUBufferUsage.COPY_SRC));
      const condBuffer = keep(this.allocator.upload("difftx.cond", cond, storage));
      const maskBuffer = keep(this.allocator.upload("difftx.mask", mask, storage));
      // See #pairNorm: everything on this line and the two below it is the
      // trunk's, not the step's, and is skipped outright when the caller keeps
      // this instance across a schedule.
      const normBytes = pairs * pairChannels * 4;
      const buildPairNorm = this.#pairNorm?.pairCond !== pairCond
        || this.#pairNorm?.bytes !== normBytes;
      if (buildPairNorm) {
        this.dispose();
        noteAllocation(this.device, "difftx.pair-norm", normBytes);
        // 🔴 `pairCond` IS NOT SET UNTIL THE PASS THAT FILLS THIS HAS BEEN
        // SUBMITTED. An allocation between here and there can refuse on
        // budget, and run() retries the whole call - which would find a cache
        // that matches and skip the norm, handing twenty-four blocks an
        // uninitialised buffer.
        this.#pairNorm = {
          pairCond: undefined, bytes: normBytes,
          buffer: this.device.createBuffer({
            label: "difftx.pair-norm", size: normBytes, usage: storage,
          }),
        };
      }
      const normalized = { buffer: this.#pairNorm.buffer };
      const pairBuffer = buildPairNorm
        ? keep(this.allocator.upload("difftx.pair", pairCond, storage)) : undefined;
      const pairScale = buildPairNorm
        ? keep(this.allocator.upload("difftx.pair-scale",
                                     weights.pairInputLayerNormScale, storage))
        : undefined;
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
      // The shared pair LayerNorm, once for the whole stack - and once for the
      // whole SCHEDULE, since nothing it reads moves with the noise level.
      if (buildPairNorm) {
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
        this.#pairNorm.pairCond = pairCond;
      }

      // 🔴 ONE ENCODER AND ONE SUBMIT FOR ALL TWENTY-FOUR BLOCKS. Every block
      // used to finish and submit its own command buffer, which at eight tokens
      // - where the matmuls are nothing at all - was most of what the stack
      // cost. The blocks are strictly sequential on the same buffers and WebGPU
      // orders passes within an encoder, so batching them changes nothing about
      // what runs, only how many times the CPU asks the driver to run it.
      validation.begin();
      let encoder = this.device.createCommandEncoder({ label: "difftx.stack" });
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
      // The uploads a super-block owns, released once its commands are queued.
      const pending = [];
      let submits = 0;
      /**
       * Queue what has been encoded and let its weight uploads be reused.
       *
       * 🔴 RELEASING AFTER THE SUBMIT IS SAFE, AND RELEASING BEFORE IT IS NOT.
       * The allocator RECYCLES a released buffer, so handing one back while a
       * later block's pass is still being encoded against it would give that
       * block the same memory. Once the commands are queued the ordering does
       * the rest: allocator.upload writes through device.queue.writeBuffer,
       * which is ordered against work already submitted, so a recycled buffer
       * is only overwritten after the passes reading it have run. This is the
       * pairformer stack's idiom, for the same reason.
       */
      const flush = (label) => {
        this.device.queue.submit([encoder.finish()]);
        validation.end(label);
        for (let at = pending.length - 1; at >= 0; at -= 1) pending[at].release();
        pending.length = 0;
        submits += 1;
        validation.begin();
        encoder = this.device.createCommandEncoder({ label: "difftx.stack" });
      };
      for (const [groupIndex, group] of weights.superBlocks.entries()) {
        const projection = this.allocator.upload("difftx.pair-projection",
          group.pairLogitsProjection, storage);
        projections.push(projection);
        if (!this.residentWeights) pending.push(projection);
        for (let inner = 0; inner < group.blocks.length; inner += 1) {
          const block = group.blocks[inner];
          // 🔴 THE SAME TRADE AS THE PAIRFORMER'S, AND THE SAME ANSWER: TRY IT
          // AND LET THE BUDGET DECIDE. The 24 blocks are about 630 MB resident,
          // which is what makes a 200-step fold affordable - the upload alone
          // was 174 ms a call at eight tokens - but it is also half of what a
          // fold holds on the device. Over budget, this drops to uploading the
          // block per call, which is slow and finishes, rather than a
          // createBuffer the driver accepts on its way to freezing the machine.
          let blockWeights;
          if (this.residentWeights) {
            // A refusal here is not caught: run() restarts the call.
            blockWeights = {
              buffer: residentBlockBuffer(
                this.device, block, () => packBlockWeights(block, weightPrecision),
                weightPrecision),
            };
            // ...and the host's float32 goes: every buffer this block needs is
            // on the device for the model's lifetime. A lazily loaded weight
            // object decodes again if anything reads it after this.
            releaseWeights(block);
          } else {
            // ...held only until this super-block is submitted; see flush().
            blockWeights = keep(this.allocator.upload("difftx.block",
              packBlockWeights(block, weightPrecision).data, storage));
            pending.push(blockWeights);
          }
          const pairGroups = Math.ceil(pairs / 64);
          run("pair-logits", compiled.pairLogits[inner], [normalized, projection, logits],
              Math.min(pairGroups, GRID_WIDTH), Math.ceil(pairGroups / GRID_WIDTH));
          run("adaln", compiled.adaln, [actBuffer, condBuffer, blockWeights, xBuffer],
              Math.ceil(tokens / tile));
          run("qkvg", compiled.qkvg, [xBuffer, blockWeights, q, k, v, gate],
              Math.ceil(tokens / tile), sources.qkvgSplits);
          const slots = tokens * heads;
          run("attend", compiled.attend, [q, k, v, logits, maskBuffer, gathered],
              Math.min(slots, GRID_WIDTH), Math.ceil(slots / GRID_WIDTH));
          run("attention-output", compiled.attentionOutput,
              [gathered, gate, condBuffer, blockWeights, actBuffer],
              Math.ceil(tokens / tile), sources.outSplits);
          run("ffw-adaln", compiled.ffwAdaln,
              [condBuffer, blockWeights, actBuffer, xBuffer], Math.ceil(tokens / tile));
          run("ffw-wide", compiled.ffwWide, [xBuffer, blockWeights, gatedBuffer],
              Math.ceil(tokens / tile), sources.wideSplits);
          run("ffw-out", compiled.ffwOut,
              [gatedBuffer, condBuffer, blockWeights, actBuffer],
              Math.ceil(tokens / outTile), sources.outSplits);
        }
        // 🔴 ONE SUBMIT FOR THE WHOLE STACK WHEN THE WEIGHTS ARE RESIDENT, AND
        // ONE PER SUPER-BLOCK WHEN THEY ARE NOT. Batching all twenty-four
        // blocks into one command buffer is worth a lot at small token counts,
        // where the driver call is most of what the stack costs - but it also
        // means no upload can be released until the end, and the uploading path
        // then holds all twenty-four at once: 24 x 31.5 MiB is 756 MiB, MORE
        // than the 630 of residency it was called in to avoid. Four blocks at a
        // time bounds that at a super-block, which is the whole point of
        // falling back. Resident runs are untouched and still submit once.
        if (!this.residentWeights) flush(`super-block ${groupIndex}`);
      }
      // ...and the readback rides the same submit.
      encoder.copyBufferToBuffer(actBuffer.buffer, 0, readback.buffer, 0, tokens * channels * 4);
      this.device.queue.submit([encoder.finish()]);
      validation.end(submits === 0 ? "block stack" : "readback");
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
