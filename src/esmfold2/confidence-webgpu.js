/**
 * Synthyra's ESMFold2 confidence head, on the device.
 *
 * 🔴 ITS FOUR BLOCKS ARE THE TRUNK'S BLOCK, SO THEY ARE NOT HERE. They go
 * through `Esmfold2TrunkGpu` - the same class that runs the trunk's 24, at the
 * same widths, with the same tensors under the same names. What this file owns
 * is what surrounds them: the pair the blocks are handed, the single the
 * pooling makes out of their answer, and the two readouts.
 *
 * 🔴 AND THE PAIR-SIZED WORK IS THE ONLY WORK THAT HAD TO COME HERE. A head
 * runs ONCE a fold, not once a sampler step, so the per-token and per-atom
 * arithmetic stays on the host where it is readable: the four small
 * projections of `s_inputs` (tokens x 451), the pLDDT einsum, and both
 * categorical means. What cannot is anything shaped `tokens^2 x 256` - at 300
 * tokens the rank-1 projection alone is 5.9 GFLOP, which is minutes of
 * JavaScript and milliseconds of this.
 *
 * Three kernels, and two of them are general:
 *   pairInit      LayerNorm a pair row, add the two broadcast terms and the
 *                 rank-1 projection, then GATHER the distance embedding.
 *   projectRows   one `[in, out]` matrix over `rows` pair rows. Used for the
 *                 rank-1 projection (256 -> 256), the PAE logits (256 -> 64)
 *                 and the pooling's scalar (256 -> 1).
 *   poolRows      a masked softmax along j and the weighted sum of the row.
 */
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";

const GRID_WIDTH = 32_768;
const LANES = 64;

/**
 * LayerNorm the pair, add `rows[i]`, `cols[j]` and `extra[slot]`, then the
 * distance embedding's row.
 *
 * 🔴 THE BUCKET IS A COUNT AGAINST EXPLICIT EDGES AND NOT A DIVISION. Their
 * `(d.unsqueeze(-1) > boundaries).sum(-1)` is arithmetically a uniform bin only
 * because `linspace(2, 52, 127)` happens to be evenly spaced, and deriving a
 * `floor((d - 2) / step)` from that is one off-by-one away from reading the
 * neighbouring row of a 128-row table - which is a confidence that moves a few
 * percent and looks fine. The edges are in the bundle; this compares against
 * them.
 */
export function createPairInitShader({ tokens, channels, edges }) {
  // 🔴 THE SMALL VECTORS TRAVEL IN ONE BUFFER BECAUSE WebGPU PROMISES EIGHT.
  // `maxStorageBuffersPerShaderStage` is 8 at the guaranteed minimum - this
  // A100 offers 16 and would have hidden it - and the first version of this
  // kernel bound ten, so it could not create its pipeline on a conforming
  // device. `npm run test:spec-floor` exists for exactly that class. The scale,
  // the offset, the 127 edges and the rep-atom coordinates are one `constants`
  // array with the offsets baked in below.
  const scaleAt = 0;
  const offsetAt = channels;
  const edgesAt = channels * 2;
  const coordsAt = channels * 2 + edges;
  return `
const TOKENS: u32 = ${tokens}u;
const CHANNELS: u32 = ${channels}u;
const EDGES: u32 = ${edges}u;
const SCALE_AT: u32 = ${scaleAt}u;
const OFFSET_AT: u32 = ${offsetAt}u;
const EDGES_AT: u32 = ${edgesAt}u;
const COORDS_AT: u32 = ${coordsAt}u;
const LANES: u32 = ${LANES}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1.0e-5;

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> constants: array<f32>;
@group(0) @binding(2) var<storage, read> rows: array<f32>;
@group(0) @binding(3) var<storage, read> cols: array<f32>;
@group(0) @binding(4) var<storage, read> extra: array<f32>;
@group(0) @binding(5) var<storage, read> embedding: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

var<workgroup> reduce: array<f32, LANES>;

fn total_of(lane: u32, value: f32) -> f32 {
  reduce[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { reduce[lane] += reduce[lane + stride]; }
    workgroupBarrier();
  }
  return reduce[0];
}

@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= TOKENS * TOKENS) { return; }
  let lane = local.x;
  let i = row / TOKENS;
  let j = row % TOKENS;

  var partial = 0.0;
  for (var c = lane; c < CHANNELS; c += LANES) { partial += pair[row * CHANNELS + c]; }
  let mean = total_of(lane, partial) / f32(CHANNELS);
  workgroupBarrier();
  var squares = 0.0;
  for (var c = lane; c < CHANNELS; c += LANES) {
    let d = pair[row * CHANNELS + c] - mean;
    squares += d * d;
  }
  let inverse = inverseSqrt(total_of(lane, squares) / f32(CHANNELS) + EPSILON);
  workgroupBarrier();

  var squared = 0.0;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let delta = constants[COORDS_AT + i * 3u + axis] - constants[COORDS_AT + j * 3u + axis];
    squared += delta * delta;
  }
  let length = sqrt(squared);
  var bucket = 0u;
  for (var edge = 0u; edge < EDGES; edge += 1u) {
    if (length > constants[EDGES_AT + edge]) { bucket += 1u; }
  }

  for (var c = lane; c < CHANNELS; c += LANES) {
    let normed = (pair[row * CHANNELS + c] - mean) * inverse
      * constants[SCALE_AT + c] + constants[OFFSET_AT + c];
    out[row * CHANNELS + c] = normed + rows[i * CHANNELS + c] + cols[j * CHANNELS + c]
      + extra[row * CHANNELS + c] + embedding[bucket * CHANNELS + c];
  }
}`;
}

/** `out[row, o] = sum_i in[row, i] * weight[i, o]`, the weight already transposed. */
export function createProjectRowsShader({ rows, inChannels, outChannels }) {
  return `
const ROWS: u32 = ${rows}u;
const IN: u32 = ${inChannels}u;
const OUT: u32 = ${outChannels}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x + id.y * GRID_WIDTH * 64u;
  if (slot >= ROWS * OUT) { return; }
  let row = slot / OUT;
  let o = slot % OUT;
  var sum = 0.0;
  for (var c = 0u; c < IN; c += 1u) {
    sum += source[row * IN + c] * weight[c * OUT + o];
  }
  out[slot] = sum;
}`;
}

/**
 * Row-attention pooling: a masked softmax over j, then the weighted sum.
 *
 * 🔴 THE MASK IS A BIAS AND NOT A ZEROED WEIGHT, which is their line: a padded
 * column enters the softmax at -1e9 and leaves at zero, where dropping it from
 * the sum instead would renormalise the row over a different denominator.
 */
export function createPoolShader({ tokens, channels }) {
  return `
const TOKENS: u32 = ${tokens}u;
const CHANNELS: u32 = ${channels}u;
const LANES: u32 = ${LANES}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> scores: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> reduce: array<f32, LANES>;

fn largest_of(lane: u32, value: f32) -> f32 {
  reduce[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { reduce[lane] = max(reduce[lane], reduce[lane + stride]); }
    workgroupBarrier();
  }
  return reduce[0];
}

fn total_of(lane: u32, value: f32) -> f32 {
  reduce[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { reduce[lane] += reduce[lane + stride]; }
    workgroupBarrier();
  }
  return reduce[0];
}

fn biased(i: u32, j: u32) -> f32 {
  if (mask[j] > 0.5) { return scores[i * TOKENS + j]; }
  return -1.0e9;
}

@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let i = group.x + group.y * GRID_WIDTH;
  if (i >= TOKENS) { return; }
  let lane = local.x;

  var top = -3.4e38;
  for (var j = lane; j < TOKENS; j += LANES) { top = max(top, biased(i, j)); }
  let peak = largest_of(lane, top);
  workgroupBarrier();
  var partial = 0.0;
  for (var j = lane; j < TOKENS; j += LANES) { partial += exp(biased(i, j) - peak); }
  let denominator = total_of(lane, partial);
  workgroupBarrier();

  for (var c = lane; c < CHANNELS; c += LANES) {
    var sum = 0.0;
    for (var j = 0u; j < TOKENS; j += 1u) {
      sum += exp(biased(i, j) - peak) * pair[(i * TOKENS + j) * CHANNELS + c];
    }
    out[i * CHANNELS + c] = sum / denominator;
  }
}`;
}

const storage = () => GPUBufferUsage.STORAGE;

/**
 * The pair the confidence blocks are handed, and the single their answer makes.
 *
 * Returns the finished pair (after the blocks, which the caller runs) is NOT
 * this function's job: it returns the INITIAL pair, so the caller can hand it
 * to `Esmfold2TrunkGpu` and hand the result back for the readouts.
 */
export async function esmfold2ConfidencePairInit(device, input, weights, options = {}) {
  const { tokens } = input;
  const channels = weights.pairChannels;
  const pairs = tokens * tokens;
  const pipelines = pipelineCacheForDevice(device);
  const allocator = options.allocator ?? new GpuBufferAllocator(device);
  const allocations = [];
  const keep = (a) => { allocations.push(a); return a; };
  try {
    const project = await pipelines.get(
      `ef2-conf-project:${pairs}:${channels}:${channels}`,
      createProjectRowsShader({ rows: pairs, inChannels: channels, outChannels: channels }));
    const init = await pipelines.get(
      `ef2-conf-pair-init:${tokens}:${channels}:${weights.boundaries.length}`,
      createPairInitShader({ tokens, channels, edges: weights.boundaries.length }));

    const product = keep(allocator.upload("ef2-conf.product", input.product, storage()));
    const projected = keep(allocator.allocate("ef2-conf.projected", pairs * channels * 4,
                                              storage()));
    const pair = keep(allocator.upload("ef2-conf.pair", input.pair, storage()));
    const out = keep(allocator.allocate("ef2-conf.init", pairs * channels * 4,
                                        storage() | GPUBufferUsage.COPY_SRC));
    const up = (name, values) => keep(allocator.upload(`ef2-conf.${name}`, values, storage()));
    const bind = (pipeline, buffers) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((allocation, binding) => ({
        binding, resource: { buffer: allocation.buffer },
      })),
    });

    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder({ label: "ef2-confidence-pair-init" });
    let pass = encoder.beginComputePass({ label: "ef2-conf.project" });
    pass.setPipeline(project);
    pass.setBindGroup(0, bind(project, [product, up("prodOut", weights.sToZProdOut), projected]));
    const cells = pairs * channels;
    pass.dispatchWorkgroups(Math.min(Math.ceil(cells / 64), GRID_WIDTH),
                            Math.ceil(Math.ceil(cells / 64) / GRID_WIDTH));
    pass.end();
    pass = encoder.beginComputePass({ label: "ef2-conf.init" });
    pass.setPipeline(init);
    const constants = new Float32Array(channels * 2 + weights.boundaries.length + tokens * 3);
    constants.set(weights.zNormScale, 0);
    constants.set(weights.zNormOffset, channels);
    constants.set(weights.boundaries, channels * 2);
    constants.set(input.repCoordinates, channels * 2 + weights.boundaries.length);
    pass.setBindGroup(0, bind(init, [
      pair, up("constants", constants), up("rows", input.rows), up("cols", input.cols),
      projected, up("embedding", weights.distanceEmbedding), out]));
    pass.dispatchWorkgroups(Math.min(pairs, GRID_WIDTH), Math.ceil(pairs / GRID_WIDTH));
    pass.end();
    const readback = keep(allocator.allocate("ef2-conf.rb-init", pairs * channels * 4,
                                             GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    encoder.copyBufferToBuffer(out.buffer, 0, readback.buffer, 0, pairs * channels * 4);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
    await readback.buffer.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(readback.buffer.getMappedRange().slice(0));
    readback.buffer.unmap();
    return copy;
  } finally {
    for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
  }
}

/** The pooled single and the PAE logits, off the blocks' finished pair. */
export async function esmfold2ConfidenceReadouts(device, input, weights, options = {}) {
  const { tokens } = input;
  const channels = weights.pairChannels;
  const single = weights.singleChannels;
  const paeBins = weights.paeBins;
  const pairs = tokens * tokens;
  const pipelines = pipelineCacheForDevice(device);
  const allocator = options.allocator ?? new GpuBufferAllocator(device);
  const allocations = [];
  const keep = (a) => { allocations.push(a); return a; };
  try {
    const scoreProject = await pipelines.get(
      `ef2-conf-project:${pairs}:${channels}:1`,
      createProjectRowsShader({ rows: pairs, inChannels: channels, outChannels: 1 }));
    const paeProject = await pipelines.get(
      `ef2-conf-project:${pairs}:${channels}:${paeBins}`,
      createProjectRowsShader({ rows: pairs, inChannels: channels, outChannels: paeBins }));
    const outProject = await pipelines.get(
      `ef2-conf-project:${tokens}:${channels}:${single}`,
      createProjectRowsShader({ rows: tokens, inChannels: channels, outChannels: single }));
    const pool = await pipelines.get(`ef2-conf-pool:${tokens}:${channels}`,
                                     createPoolShader({ tokens, channels }));

    const up = (name, values) => keep(allocator.upload(`ef2-conf.${name}`, values, storage()));
    const pair = up("finished", input.pair);
    const scores = keep(allocator.allocate("ef2-conf.scores", pairs * 4, storage()));
    const pooled = keep(allocator.allocate("ef2-conf.pooled", tokens * channels * 4, storage()));
    const singleOut = keep(allocator.allocate("ef2-conf.single", tokens * single * 4,
                                              storage() | GPUBufferUsage.COPY_SRC));
    const paeOut = keep(allocator.allocate("ef2-conf.pae", pairs * paeBins * 4,
                                           storage() | GPUBufferUsage.COPY_SRC));
    const bind = (pipeline, buffers) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((allocation, binding) => ({
        binding, resource: { buffer: allocation.buffer },
      })),
    });
    const rowsOf = (cells) => {
      const groups = Math.ceil(cells / 64);
      return [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
    };

    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder({ label: "ef2-confidence-readouts" });
    let pass = encoder.beginComputePass({ label: "ef2-conf.pool-scores" });
    pass.setPipeline(scoreProject);
    pass.setBindGroup(0, bind(scoreProject,
      [pair, up("poolAttn", weights.poolingAttention), scores]));
    pass.dispatchWorkgroups(...rowsOf(pairs));
    pass.end();
    pass = encoder.beginComputePass({ label: "ef2-conf.pool" });
    pass.setPipeline(pool);
    pass.setBindGroup(0, bind(pool, [pair, scores, up("mask", input.tokenMask), pooled]));
    pass.dispatchWorkgroups(Math.min(tokens, GRID_WIDTH), Math.ceil(tokens / GRID_WIDTH));
    pass.end();
    pass = encoder.beginComputePass({ label: "ef2-conf.pool-output" });
    pass.setPipeline(outProject);
    pass.setBindGroup(0, bind(outProject,
      [pooled, up("poolOut", weights.poolingOutput), singleOut]));
    pass.dispatchWorkgroups(...rowsOf(tokens * single));
    pass.end();
    pass = encoder.beginComputePass({ label: "ef2-conf.pae" });
    pass.setPipeline(paeProject);
    pass.setBindGroup(0, bind(paeProject, [pair, up("paeWeight", weights.pae), paeOut]));
    pass.dispatchWorkgroups(...rowsOf(pairs * paeBins));
    pass.end();

    const readSingle = keep(allocator.allocate("ef2-conf.rb-single", tokens * single * 4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    const readPae = keep(allocator.allocate("ef2-conf.rb-pae", pairs * paeBins * 4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    encoder.copyBufferToBuffer(singleOut.buffer, 0, readSingle.buffer, 0, tokens * single * 4);
    encoder.copyBufferToBuffer(paeOut.buffer, 0, readPae.buffer, 0, pairs * paeBins * 4);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
    const read = async (allocation) => {
      await allocation.buffer.mapAsync(GPUMapMode.READ);
      const copy = new Float32Array(allocation.buffer.getMappedRange().slice(0));
      allocation.buffer.unmap();
      return copy;
    };
    return { single: await read(readSingle), paeLogits: await read(readPae) };
  } finally {
    for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
  }
}
