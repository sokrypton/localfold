/**
 * AF3's embedder: target features, the relative encoding, and the recycled
 * representations, assembled into the pair, MSA and single tracks the trunk
 * starts from.
 *
 * 🔴 THE RELATIVE ENCODING IS NEVER MATERIALISED. Its 139 columns hold exactly
 * three one-hot bins and one binary flag, so projecting them is a sum of at
 * most four rows of position_activations rather than a 139x128 product, and the
 * tokens^2 x 139 tensor (50 MB at 300 tokens, 200 MB at 600) never exists. The
 * MSA features are the same shape of thing: a one-hot residue code plus two
 * scalars, so 34 columns become one row plus two scaled rows.
 *
 * THE ARITHMETIC SAVING IS 139x AND THE MEASURED SAVING IS NOT - it is 1.14x at
 * 96 tokens, 1.28x at 192 and 1.36x at 384, on the whole embedder
 * (tools/gpu/bench-relative-encoding.js, interleaved). Two reasons, and both
 * are the general lesson rather than a fact about this kernel. The 139 MACs
 * this removes share a pass with the recycled projection's 128, which stays -
 * so the pass loses half its work, not all of it. And the embedder is a small
 * slice of a trunk whose cost is triangle multiplication. Operation count is a
 * bad proxy for time; the reason to keep the gather is mostly the 50 MB.
 *
 * 🔴 THE RECYCLED TERM IS NOT ZERO ON THE FIRST PASS. AF3 starts from a zero
 * pair, but the LayerNorm ahead of the projection turns a zero input into its
 * OFFSET, and projecting that offset gives a constant this graph adds every
 * time. Skipping the branch "because there is nothing to recycle yet" silently
 * drops a term that is present on pass one. Both recycled branches are always
 * encoded here for that reason; there is no first-pass fast path.
 *
 * 🔴 THE TEMPLATE EMBEDDING IS NOT ADDED HERE. It reads the pair as it is after
 * the relative encoding and the bonds, so it is a separate step the caller
 * sequences - and it contributes even with no templates at all (std 13.1 with
 * four empty slots), so leaving it out is not the same as adding zeros.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const GRID_WIDTH = 32_768;
const MAX_RELATIVE_IDX = 32;
const MAX_RELATIVE_CHAIN = 2;
const POSITION_BINS = 2 * MAX_RELATIVE_IDX + 2;   // 66
const RELATIVE_WIDTH = POSITION_BINS * 2 + 1 + (2 * MAX_RELATIVE_CHAIN + 2);  // 139
const MSA_FEATURE_WIDTH = 34;

const ORDER = [
  "leftSingle", "rightSingle", "prevEmbeddingNormScale", "prevEmbeddingNormOffset",
  "prevEmbedding", "positionActivations", "msaActivations", "extraMsaTargetFeat",
  "singleActivations", "prevSingleEmbeddingNormScale", "prevSingleEmbeddingNormOffset",
  "prevSingleEmbedding",
];

export function packEmbedderWeights(weights) {
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    if (weights[name] === undefined) throw new Error(`embedder weights missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of ORDER) data.set(weights[name], offsets[name]);
  return { data, offsets };
}

export function createEmbedderShaders(shape, offsets, epsilon, variance,
                                      relative = "gather") {
  const { tokens, sequences, featureWidth, pairChannels, singleChannels, msaChannels } = shape;

  const common = `
const TOKENS: u32 = ${tokens}u;
const SEQUENCES: u32 = ${sequences}u;
const PAIRS: u32 = ${tokens * tokens}u;
const FEATURE_WIDTH: u32 = ${featureWidth}u;
const C_Z: u32 = ${pairChannels}u;
const C_S: u32 = ${singleChannels}u;
const C_M: u32 = ${msaChannels}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
const MAX_RELATIVE_IDX: i32 = ${MAX_RELATIVE_IDX};
const MAX_RELATIVE_CHAIN: i32 = ${MAX_RELATIVE_CHAIN};
const POSITION_BINS: u32 = ${POSITION_BINS}u;
const RELATIVE_WIDTH: u32 = ${RELATIVE_WIDTH}u;
const W_LEFT: u32 = ${offsets.leftSingle}u;
const W_RIGHT: u32 = ${offsets.rightSingle}u;
const W_PREV_SCALE: u32 = ${offsets.prevEmbeddingNormScale}u;
const W_PREV_OFFSET: u32 = ${offsets.prevEmbeddingNormOffset}u;
const W_PREV: u32 = ${offsets.prevEmbedding}u;
const W_POSITION: u32 = ${offsets.positionActivations}u;
const W_MSA: u32 = ${offsets.msaActivations}u;
const W_MSA_TARGET: u32 = ${offsets.extraMsaTargetFeat}u;
const W_SINGLE: u32 = ${offsets.singleActivations}u;
const W_PREV_SINGLE_SCALE: u32 = ${offsets.prevSingleEmbeddingNormScale}u;
const W_PREV_SINGLE_OFFSET: u32 = ${offsets.prevSingleEmbeddingNormOffset}u;
const W_PREV_SINGLE: u32 = ${offsets.prevSingleEmbedding}u;

fn clamp_bin(value: i32, high: i32) -> i32 { return min(max(value, 0), high); }
`;

  // Per token: the two pair projections and the MSA-from-target projection.
  const projectTokens = `${common}
@group(0) @binding(0) var<storage, read> target_feat: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> left: array<f32>;
@group(0) @binding(3) var<storage, read_write> right: array<f32>;
@group(0) @binding(4) var<storage, read_write> msa_from_target: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let token = id.x;
  if (token >= TOKENS) { return; }
  let base = token * FEATURE_WIDTH;
  for (var c = 0u; c < C_Z; c += 1u) {
    var left_total = 0.0;
    var right_total = 0.0;
    for (var f = 0u; f < FEATURE_WIDTH; f += 1u) {
      let value = target_feat[base + f];
      left_total += value * weights[W_LEFT + f * C_Z + c];
      right_total += value * weights[W_RIGHT + f * C_Z + c];
    }
    left[token * C_Z + c] = left_total;
    right[token * C_Z + c] = right_total;
  }
  for (var c = 0u; c < C_M; c += 1u) {
    var total = 0.0;
    for (var f = 0u; f < FEATURE_WIDTH; f += 1u) {
      total += target_feat[base + f] * weights[W_MSA_TARGET + f * C_M + c];
    }
    msa_from_target[token * C_M + c] = total;
  }
}`;

  // One workgroup per token pair: left_i + right_j + the relative encoding's
  // gathered rows + the recycled pair. No pairs x 139 tensor anywhere.
  const assemblePair = `${common}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read> previous: array<f32>;
@group(0) @binding(3) var<storage, read> features: array<i32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> pair: array<f32>;

// features is five rows of TOKENS: residueIndex, tokenIndex, asymId, entityId, symId.
fn residue_index(t: u32) -> i32 { return features[t]; }
fn token_index(t: u32) -> i32 { return features[TOKENS + t]; }
fn asym_id(t: u32) -> i32 { return features[2u * TOKENS + t]; }
fn entity_id(t: u32) -> i32 { return features[3u * TOKENS + t]; }
fn sym_id(t: u32) -> i32 { return features[4u * TOKENS + t]; }

var<workgroup> normalized: array<f32, ${pairChannels}>;
var<workgroup> reduce_a: array<f32, 64>;
var<workgroup> reduce_b: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= PAIRS) { return; }
  let local = local_id.x;
  let i = row / TOKENS;
  let j = row % TOKENS;
  let base = row * C_Z;

  // The recycled pair, whose LayerNorm makes it nonzero even on pass one.
  var total = 0.0;
  var squares = 0.0;
  for (var c = local; c < C_Z; c += 64u) {
    let value = previous[base + c];
    total += value;
    squares += value * value;
  }
  reduce_a[local] = total;
  reduce_b[local] = squares;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) {
      reduce_a[local] += reduce_a[local + stride];
      reduce_b[local] += reduce_b[local + stride];
    }
    workgroupBarrier();
  }
  let mean = reduce_a[0] / f32(C_Z);
  ${variance === "fast"
    ? "let variance = reduce_b[0] / f32(C_Z) - mean * mean;"
    : `var centered = 0.0;
  for (var c = local; c < C_Z; c += 64u) {
    let d = previous[base + c] - mean;
    centered += d * d;
  }
  // 🔴 A BARRIER BEFORE REUSING THE REDUCTION BUFFER. Every invocation has
  // just read reduce_a[0] for the mean; writing reduce_a[local] without a
  // barrier lets a fast lane clobber slot 0 while a slow one is still reading
  // it. The result is a WRONG MEAN in some rows, some of the time - which
  // reads as a numerical problem, not a race.
  workgroupBarrier();
  reduce_a[local] = centered;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_a[local] += reduce_a[local + stride]; }
    workgroupBarrier();
  }
  let variance = reduce_a[0] / f32(C_Z);`}
  let inverse_std = inverseSqrt(variance + EPSILON);
  workgroupBarrier();
  for (var c = local; c < C_Z; c += 64u) {
    normalized[c] = (previous[base + c] - mean) * inverse_std * weights[W_PREV_SCALE + c]
      + weights[W_PREV_OFFSET + c];
  }
  workgroupBarrier();

  // The relative encoding's four active columns, resolved once per pair.
  let same_chain = asym_id(i) == asym_id(j);
  let same_entity = entity_id(i) == entity_id(j);
  var bin_a = u32(2 * MAX_RELATIVE_IDX + 1);
  if (same_chain) {
    bin_a = u32(clamp_bin(residue_index(i) - residue_index(j) + MAX_RELATIVE_IDX,
                          2 * MAX_RELATIVE_IDX));
  }
  var bin_b = u32(2 * MAX_RELATIVE_IDX + 1);
  if (same_chain && residue_index(i) == residue_index(j)) {
    bin_b = u32(clamp_bin(token_index(i) - token_index(j) + MAX_RELATIVE_IDX,
                          2 * MAX_RELATIVE_IDX));
  }
  var bin_c = u32(2 * MAX_RELATIVE_CHAIN + 1);
  if (same_entity) {
    bin_c = u32(clamp_bin(sym_id(i) - sym_id(j) + MAX_RELATIVE_CHAIN,
                          2 * MAX_RELATIVE_CHAIN));
  }
  let row_a = W_POSITION + bin_a * C_Z;
  let row_b = W_POSITION + (POSITION_BINS + bin_b) * C_Z;
  let row_entity = W_POSITION + POSITION_BINS * 2u * C_Z;
  let row_c = W_POSITION + (POSITION_BINS * 2u + 1u + bin_c) * C_Z;
  var entity_flag = 0.0;
  if (same_entity) { entity_flag = 1.0; }

  for (var c = local; c < C_Z; c += 64u) {
    var recycled = 0.0;
    for (var k = 0u; k < C_Z; k += 1u) {
      recycled += normalized[k] * weights[W_PREV + k * C_Z + c];
    }
    ${relative === "gather"
      ? `// The four active rows, summed directly.
    let relative_total = weights[row_a + c] + weights[row_b + c]
      + entity_flag * weights[row_entity + c] + weights[row_c + c];`
      : `// The same thing as a dense 139-wide projection, for the A/B in
    // tools/gpu/bench-relative-encoding.js. The one-hot is still built in
    // registers rather than materialised, so this is the BEST case for dense.
    var relative_total = 0.0;
    for (var f = 0u; f < RELATIVE_WIDTH; f += 1u) {
      var onehot = 0.0;
      if (f == bin_a) { onehot = 1.0; }
      if (f == POSITION_BINS + bin_b) { onehot = 1.0; }
      if (f == POSITION_BINS * 2u && same_entity) { onehot = 1.0; }
      if (f == POSITION_BINS * 2u + 1u + bin_c) { onehot = 1.0; }
      relative_total += onehot * weights[W_POSITION + f * C_Z + c];
    }`}
    pair[base + c] = left[i * C_Z + c] + right[j * C_Z + c] + recycled + relative_total;
  }
}`;

  // The MSA track: a one-hot residue code plus two deletion scalars.
  const assembleMsa = `${common}
@group(0) @binding(0) var<storage, read> codes: array<i32>;
@group(0) @binding(1) var<storage, read> deletions: array<f32>;
@group(0) @binding(2) var<storage, read> msa_from_target: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> msa: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= SEQUENCES * TOKENS) { return; }
  let token = row % TOKENS;
  let code = codes[row];
  let raw = deletions[row];
  let clamped = min(max(raw, 0.0), 1.0);
  // ...arctan-squashed, not clipped, so a deep deletion column stays distinct.
  let squashed = atan(raw / 3.0) * (2.0 / ${Math.PI});
  for (var c = 0u; c < C_M; c += 1u) {
    var total = clamped * weights[W_MSA + 32u * C_M + c]
      + squashed * weights[W_MSA + 33u * C_M + c];
    if (code >= 0 && code < 32) {
      total += weights[W_MSA + u32(code) * C_M + c];
    }
    msa[row * C_M + c] = total + msa_from_target[token * C_M + c];
  }
}`;

  const assembleSingle = `${common}
@group(0) @binding(0) var<storage, read> target_feat: array<f32>;
@group(0) @binding(1) var<storage, read> previous: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> single: array<f32>;

var<workgroup> normalized: array<f32, ${singleChannels}>;
var<workgroup> reduce_a: array<f32, 64>;
var<workgroup> reduce_b: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x;
  if (token >= TOKENS) { return; }
  let local = local_id.x;
  let base = token * C_S;

  var total = 0.0;
  var squares = 0.0;
  for (var c = local; c < C_S; c += 64u) {
    let value = previous[base + c];
    total += value;
    squares += value * value;
  }
  reduce_a[local] = total;
  reduce_b[local] = squares;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) {
      reduce_a[local] += reduce_a[local + stride];
      reduce_b[local] += reduce_b[local + stride];
    }
    workgroupBarrier();
  }
  let mean = reduce_a[0] / f32(C_S);
  ${variance === "fast"
    ? "let variance = reduce_b[0] / f32(C_S) - mean * mean;"
    : `var centered = 0.0;
  for (var c = local; c < C_S; c += 64u) {
    let d = previous[base + c] - mean;
    centered += d * d;
  }
  // 🔴 A BARRIER BEFORE REUSING THE REDUCTION BUFFER. Every invocation has
  // just read reduce_a[0] for the mean; writing reduce_a[local] without a
  // barrier lets a fast lane clobber slot 0 while a slow one is still reading
  // it. The result is a WRONG MEAN in some rows, some of the time - which
  // reads as a numerical problem, not a race.
  workgroupBarrier();
  reduce_a[local] = centered;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_a[local] += reduce_a[local + stride]; }
    workgroupBarrier();
  }
  let variance = reduce_a[0] / f32(C_S);`}
  let inverse_std = inverseSqrt(variance + EPSILON);
  workgroupBarrier();
  for (var c = local; c < C_S; c += 64u) {
    normalized[c] = (previous[base + c] - mean) * inverse_std
      * weights[W_PREV_SINGLE_SCALE + c] + weights[W_PREV_SINGLE_OFFSET + c];
  }
  workgroupBarrier();

  for (var c = local; c < C_S; c += 64u) {
    var value = 0.0;
    for (var f = 0u; f < FEATURE_WIDTH; f += 1u) {
      value += target_feat[token * FEATURE_WIDTH + f] * weights[W_SINGLE + f * C_S + c];
    }
    for (var k = 0u; k < C_S; k += 1u) {
      value += normalized[k] * weights[W_PREV_SINGLE + k * C_S + c];
    }
    single[base + c] = value;
  }
}`;

  return { projectTokens, assemblePair, assembleMsa, assembleSingle };
}

export class Af3EmbedderGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {{tokens: number, sequences: number, targetFeat: Float32Array,
   *          features: {residueIndex, tokenIndex, asymId, entityId, symId},
   *          msaRows: ArrayLike<number>, deletionMatrix: ArrayLike<number>,
   *          previousPair?: Float32Array, previousSingle?: Float32Array}} input
   * @param {object} weights the twelve tensors in ORDER, plus the channel counts
   */
  async run(input, weights, options = {}) {
    const { tokens, sequences } = input;
    const pairChannels = weights.pairChannels;
    const singleChannels = weights.singleChannels;
    const msaChannels = weights.msaChannels;
    const featureWidth = weights.targetFeatWidth;
    const pairs = tokens * tokens;
    const rows = sequences * tokens;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    if (input.targetFeat.length !== tokens * featureWidth) {
      throw new Error(`targetFeat has ${input.targetFeat.length} elements; `
        + `expected ${tokens * featureWidth}`);
    }

    const packed = packEmbedderWeights(weights);
    const shape = { tokens, sequences, featureWidth, pairChannels, singleChannels, msaChannels };
    const sources = createEmbedderShaders(shape, packed.offsets, epsilon, variance,
                                          options.relative ?? "gather");
    const key = `af3-embed:${tokens}:${sequences}:${featureWidth}:${pairChannels}`
      + `:${singleChannels}:${msaChannels}:${epsilon}:${variance}`
      + `:${options.relative ?? "gather"}`;
    const compiled = {};
    for (const [name, source] of Object.entries(sources)) {
      compiled[name] = await this.pipelines.get(`${key}:${name}`, source);
    }

    // The five integer feature rows, packed in the order the shader indexes.
    const featureData = new Int32Array(5 * tokens);
    const names = ["residueIndex", "tokenIndex", "asymId", "entityId", "symId"];
    names.forEach((name, index) => {
      const source = input.features[name];
      if (source === undefined) throw new Error(`features.${name} is required`);
      for (let t = 0; t < tokens; t += 1) featureData[index * tokens + t] = source[t];
    });

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const targetFeat = keep(this.allocator.upload("af3-embed.target", input.targetFeat, storage));
      const weightBuffer = keep(this.allocator.upload("af3-embed.weights", packed.data, storage));
      const features = keep(this.allocator.upload("af3-embed.features", featureData, storage));
      const codes = keep(this.allocator.upload("af3-embed.codes",
        Int32Array.from(input.msaRows), storage));
      const deletions = keep(this.allocator.upload("af3-embed.deletions",
        Float32Array.from(input.deletionMatrix), storage));
      // 🔴 ALWAYS PRESENT, even on pass one - see the note at the top.
      const previousPair = keep(this.allocator.upload("af3-embed.previous-pair",
        input.previousPair ?? new Float32Array(pairs * pairChannels), storage));
      const previousSingle = keep(this.allocator.upload("af3-embed.previous-single",
        input.previousSingle ?? new Float32Array(tokens * singleChannels), storage));

      const left = keep(this.allocator.allocate("af3-embed.left", tokens * pairChannels * 4, storage));
      const right = keep(this.allocator.allocate("af3-embed.right", tokens * pairChannels * 4, storage));
      const msaFromTarget = keep(this.allocator.allocate(
        "af3-embed.msa-from-target", tokens * msaChannels * 4, storage));
      const pair = keep(this.allocator.allocate(
        "af3-embed.pair", pairs * pairChannels * 4, storage | GPUBufferUsage.COPY_SRC));
      const msa = keep(this.allocator.allocate(
        "af3-embed.msa", rows * msaChannels * 4, storage | GPUBufferUsage.COPY_SRC));
      const single = keep(this.allocator.allocate(
        "af3-embed.single", tokens * singleChannels * 4, storage | GPUBufferUsage.COPY_SRC));
      const readback = {
        pair: keep(this.allocator.allocate("af3-embed.rb-pair", pairs * pairChannels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        msa: keep(this.allocator.allocate("af3-embed.rb-msa", rows * msaChannels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
        single: keep(this.allocator.allocate("af3-embed.rb-single", tokens * singleChannels * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)),
      };

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-embedder" });
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
      const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];

      run("embed.project-tokens", compiled.projectTokens,
          [targetFeat, weightBuffer, left, right, msaFromTarget], Math.ceil(tokens / 64));
      const perPair = spread(pairs);
      run("embed.assemble-pair", compiled.assemblePair,
          [left, right, previousPair, features, weightBuffer, pair], perPair[0], perPair[1]);
      const perRow = spread(Math.ceil(rows / 64));
      run("embed.assemble-msa", compiled.assembleMsa,
          [codes, deletions, msaFromTarget, weightBuffer, msa], perRow[0], perRow[1]);
      run("embed.assemble-single", compiled.assembleSingle,
          [targetFeat, previousSingle, weightBuffer, single], tokens);

      encoder.copyBufferToBuffer(pair.buffer, 0, readback.pair.buffer, 0, pairs * pairChannels * 4);
      encoder.copyBufferToBuffer(msa.buffer, 0, readback.msa.buffer, 0, rows * msaChannels * 4);
      encoder.copyBufferToBuffer(single.buffer, 0, readback.single.buffer, 0,
                                 tokens * singleChannels * 4);

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      const read = async (allocation) => {
        await allocation.buffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(allocation.buffer.getMappedRange().slice(0));
        allocation.buffer.unmap();
        return copy;
      };
      return {
        pair: await read(readback.pair),
        msa: await read(readback.msa),
        single: await read(readback.single),
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
