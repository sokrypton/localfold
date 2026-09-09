/**
 * OpenDDE's structural-token expansion, on the device.
 *
 * 🔴 THE PAIR PROJECTION IS THE ONLY EXPENSIVE PART AND IT IS n^2 c^2. Every
 * structural pair is projected through one of 49 matrices chosen by its ordered
 * role pair, so the work is `tokens^2 * channels^2` multiply-adds - 1.2 GFLOP
 * at 91 tokens and 13 GFLOP at 300, which is seconds on the host and
 * milliseconds here. The CPU reference measures 1.8 s at 91 tokens; that is why
 * this exists.
 *
 * 🔴 AND THE MATRIX IS SELECTED PER PAIR, NOT PER BLOCK, so this is not a
 * batched GEMM with a uniform operand. One workgroup owns one pair and reads
 * the matrix its two roles name; the 49 matrices are one buffer and the role
 * pair is an offset into it.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { GRID_WIDTH } from "./pair-track-gpu.js";
import { ROLES } from "./structural-tokens.js";

/**
 * z_struct[i][j] = z[parent i][parent j] (1 + W[role i, role j]) + five biases.
 *
 * One workgroup per pair, 64 lanes over the output channels. The source row is
 * staged once because every output channel reads all of it.
 */
export function createPairExpandShader(tokens, residueTokens, channels) {
  return `
const TOKENS: u32 = ${tokens}u;
const RESIDUES: u32 = ${residueTokens}u;
const C: u32 = ${channels}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const ROLES: u32 = ${ROLES}u;

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> parent: array<i32>;
@group(0) @binding(2) var<storage, read> role: array<i32>;
// The five boolean features and the role-pair type, packed one byte-worth each
// into an i32 per pair: [samePar, twin, prevBb, nextBb, rolePairType].
@group(0) @binding(3) var<storage, read> features: array<i32>;
@group(0) @binding(4) var<storage, read> projection: array<f32>;
@group(0) @binding(5) var<storage, read> embeddings: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

var<workgroup> source: array<f32, ${channels}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= TOKENS * TOKENS) { return; }
  let local = local_id.x;
  let i = row / TOKENS;
  let j = row % TOKENS;
  let source_base = (u32(parent[i]) * RESIDUES + u32(parent[j])) * C;

  for (var c = local; c < C; c += 64u) { source[c] = pair[source_base + c]; }
  workgroupBarrier();

  let matrix = (u32(role[i]) * ROLES + u32(role[j])) * C * C;
  let f = features[row];
  // ...unpacked in the order packStructuralFeatures packed them.
  let same_parent = u32(f & 1);
  let twin = u32((f >> 1u) & 1);
  let prev_bb = u32((f >> 2u) & 1);
  let next_bb = u32((f >> 3u) & 1);
  let role_pair = u32((f >> 4u) & 15);

  // The five embedding tables, laid out back to back in the embeddings buffer:
  // sameParent[2], twin[2], prevBb[2], nextBb[2], rolePairType[8].
  let E_SAME: u32 = 0u;
  let E_TWIN: u32 = 2u * C;
  let E_PREV: u32 = 4u * C;
  let E_NEXT: u32 = 6u * C;
  let E_ROLE: u32 = 8u * C;

  for (var d = local; d < C; d += 64u) {
    var total = source[d];
    for (var c = 0u; c < C; c += 1u) {
      total += source[c] * projection[matrix + c * C + d];
    }
    total += embeddings[E_SAME + same_parent * C + d]
      + embeddings[E_TWIN + twin * C + d]
      + embeddings[E_PREV + prev_bb * C + d]
      + embeddings[E_NEXT + next_bb * C + d]
      + embeddings[E_ROLE + role_pair * C + d];
    out[row * C + d] = total;
  }
}`;
}

/** The single track: gather the parent, run the split MLP, add the role. */
export function createSingleExpandShader(tokens, residueTokens, channels, hidden) {
  return `
const TOKENS: u32 = ${tokens}u;
const C: u32 = ${channels}u;
const HIDDEN: u32 = ${hidden}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1e-5;

@group(0) @binding(0) var<storage, read> single: array<f32>;
@group(0) @binding(1) var<storage, read> parent: array<i32>;
@group(0) @binding(2) var<storage, read> role: array<i32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

// weights: normScale[C] normOffset[C] split1[C*HIDDEN] split2[HIDDEN*C] roleEmb[ROLES*C]
const W_SCALE: u32 = 0u;
const W_OFFSET: u32 = ${channels}u;
const W_SPLIT1: u32 = ${2 * channels}u;
const W_SPLIT2: u32 = ${2 * channels + channels * hidden}u;
const W_ROLE: u32 = ${2 * channels + channels * hidden + hidden * channels}u;

var<workgroup> parent_single: array<f32, ${channels}>;
var<workgroup> normalised: array<f32, ${channels}>;
var<workgroup> wide: array<f32, ${hidden}>;
var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x + group.y * GRID_WIDTH;
  if (token >= TOKENS) { return; }
  let local = local_id.x;
  let source_base = u32(parent[token]) * C;

  for (var c = local; c < C; c += 64u) { parent_single[c] = single[source_base + c]; }
  workgroupBarrier();

  var total = 0.0;
  for (var c = local; c < C; c += 64u) { total += parent_single[c]; }
  reduce[local] = total;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let mean = reduce[0] / f32(C);
  workgroupBarrier();
  var variance = 0.0;
  for (var c = local; c < C; c += 64u) {
    let d = parent_single[c] - mean;
    variance += d * d;
  }
  reduce[local] = variance;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let inverse = inverseSqrt(reduce[0] / f32(C) + EPSILON);
  workgroupBarrier();

  for (var c = local; c < C; c += 64u) {
    normalised[c] = (parent_single[c] - mean) * inverse * weights[W_SCALE + c]
      + weights[W_OFFSET + c];
  }
  workgroupBarrier();

  // ...silu, not a gate: single_split_1 is ONE matrix of 2 * C and the whole
  // of it goes through the nonlinearity. A fused SwiGLU would split it in half.
  for (var h = local; h < HIDDEN; h += 64u) {
    var value = 0.0;
    for (var c = 0u; c < C; c += 1u) { value += normalised[c] * weights[W_SPLIT1 + c * HIDDEN + h]; }
    wide[h] = value / (1.0 + exp(-value));
  }
  workgroupBarrier();

  let role_base = W_ROLE + u32(role[token]) * C;
  for (var c = local; c < C; c += 64u) {
    var value = 0.0;
    for (var h = 0u; h < HIDDEN; h += 1u) { value += wide[h] * weights[W_SPLIT2 + h * C + c]; }
    out[token * C + c] = parent_single[c] + value + weights[role_base + c];
  }
}`;
}

/** The five boolean features and the role-pair type, one i32 per pair. */
export function packStructuralFeatures(features, tokens) {
  const packed = new Int32Array(tokens * tokens);
  for (let at = 0; at < tokens * tokens; at += 1) {
    packed[at] = (features.sameParent[at] & 1)
      | ((features.twin[at] & 1) << 1)
      | ((features.prevBackbone[at] & 1) << 2)
      | ((features.nextBackbone[at] & 1) << 3)
      | ((features.rolePairType[at] & 15) << 4);
  }
  return packed;
}

export class Af3StructuralExpanderGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {object} layout from structuralLayout
   * @param {object} embeddings {single, pair, targetFeat, asymId} on residues
   * @param {object} weights from structuralExpanderWeights
   * @param {object} features from structuralPairFeatures
   */
  async run(layout, embeddings, weights, features, residueTokens, options = {}) {
    const tokens = layout.tokens;
    const channels = weights.pairChannels;
    const singleChannels = weights.singleChannels;
    const hidden = weights.singleSplit1.length / singleChannels;
    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (a) => { allocations.push(a); return a; };
    const up = (label, data) => keep(this.allocator.upload(label, data, storage));

    try {
      const [pairPipeline, singlePipeline] = await Promise.all([
        this.pipelines.get(`opendde-expand-pair:${tokens}:${residueTokens}:${channels}`,
                           createPairExpandShader(tokens, residueTokens, channels)),
        this.pipelines.get(`opendde-expand-single:${tokens}:${singleChannels}:${hidden}`,
                           createSingleExpandShader(tokens, residueTokens, singleChannels, hidden)),
      ]);

      const parent = up("expand.parent", layout.parent);
      const role = up("expand.role", layout.role);
      const packed = up("expand.features", packStructuralFeatures(features, tokens));
      const projection = up("expand.projection", weights.pairBlockProj);
      // The five tables, back to back in the order the shader indexes them.
      const tables = new Float32Array(16 * channels);
      tables.set(weights.sameParentEmbedding, 0);
      tables.set(weights.sameResidueTwinEmbedding, 2 * channels);
      tables.set(weights.prevBbChainEmbedding, 4 * channels);
      tables.set(weights.nextBbChainEmbedding, 6 * channels);
      tables.set(weights.rolePairTypeEmbedding, 8 * channels);
      const embeddingBuffer = up("expand.embeddings", tables);
      const pairIn = up("expand.pair-in", embeddings.pair);
      const pairOut = keep(this.allocator.allocate(
        "expand.pair-out", tokens * tokens * channels * 4,
        storage | GPUBufferUsage.COPY_SRC));

      const singleWeights = new Float32Array(
        2 * singleChannels + singleChannels * hidden + hidden * singleChannels
        + weights.singleRoleEmbedding.length);
      let at = 0;
      for (const part of [weights.singleSplitNormScale, weights.singleSplitNormOffset,
                          weights.singleSplit1, weights.singleSplit2,
                          weights.singleRoleEmbedding]) {
        singleWeights.set(part, at); at += part.length;
      }
      const singleIn = up("expand.single-in", embeddings.single);
      const singleWeightBuffer = up("expand.single-weights", singleWeights);
      const singleOut = keep(this.allocator.allocate(
        "expand.single-out", tokens * singleChannels * 4,
        storage | GPUBufferUsage.COPY_SRC));

      const encoder = this.device.createCommandEncoder({ label: "opendde-expand" });
      const dispatch = (pipeline, buffers, groups) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH));
        pass.end();
      };
      dispatch(pairPipeline,
               [pairIn, parent, role, packed, projection, embeddingBuffer, pairOut],
               tokens * tokens);
      dispatch(singlePipeline, [singleIn, parent, role, singleWeightBuffer, singleOut], tokens);

      // 🔴 THE EXPANDED PAIR MAY STAY WHERE IT IS. Its only reader is the
      // structural refiner, which is a pairformer stack that takes a buffer -
      // so this was `tokens^2 x 384` read back and uploaded again between two
      // GPU stages. Measured in a 200-residue OpenDDE fold: `expand.read-pair`
      // 343 ms for 216 MiB, at 0.6 GB/s. The caller owns what comes back.
      const keepPair = options.keepPair === true;
      const readPair = keepPair ? undefined : keep(this.allocator.allocate("expand.read-pair",
        tokens * tokens * channels * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      const readSingle = keep(this.allocator.allocate("expand.read-single",
        tokens * singleChannels * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      if (!keepPair) {
        encoder.copyBufferToBuffer(pairOut.buffer, 0, readPair.buffer, 0,
                                   tokens * tokens * channels * 4);
      }
      encoder.copyBufferToBuffer(singleOut.buffer, 0, readSingle.buffer, 0,
                                 tokens * singleChannels * 4);
      this.device.queue.submit([encoder.finish()]);

      const read = async (allocation) => {
        await allocation.buffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(allocation.buffer.getMappedRange().slice(0));
        allocation.buffer.unmap();
        return copy;
      };
      if (keepPair) {
        const owned = allocations.indexOf(pairOut);
        if (owned >= 0) allocations.splice(owned, 1);
        return { pair: undefined, pairAllocation: pairOut, single: await read(readSingle) };
      }
      return { pair: await read(readPair), single: await read(readSingle) };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
