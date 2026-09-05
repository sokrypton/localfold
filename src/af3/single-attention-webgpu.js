import { concatenateAs, writeInto } from "../runtime/float16.js";
/**
 * AF3's single-track attention, biased by the pair representation.
 *
 * One row per token rather than per pair, so this is the cheap kernel of the
 * pairformer - 384 channels over N tokens, where everything else in the block
 * is over N^2. It matters anyway: it is the only path by which the pair
 * representation reaches the single one.
 *
 *     single -> LayerNorm -> q (with bias), k, v, gate
 *     logits[h][i][j] = q.k / sqrt(d) + pairLogits[h][i][j] + 1e9*(seqMask[j]-1)
 *     out = outputProjection(softmax(logits) @ v * sigmoid(gate))
 *
 * 🔴 ONLY q HAS A BIAS. k, v and the gate have none, and the checkpoint has no
 * tensor for them - so a kernel that assumes a bias per projection reads past
 * the end of the packed block or, worse, reads the next weight as one.
 *
 * 🔴 NONE OF THESE CARRY `transpose_weights`, unlike grid attention's q and k.
 * The exported shapes say it: (384, 16, 24), channels first. The two attention
 * modules of the same block disagree, so neither convention can be assumed from
 * the other.
 *
 * 🔴 THE MASK TERM IS ADDITIVE AND FINITE: `1e9 * (seqMask[j] - 1)`, which is
 * 0 or -1e9. Using -inf instead produces NaN for a fully padded row rather than
 * a uniform distribution over nothing.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

export const SINGLE_ATTENTION_ORDER = [
  "layerNormScale", "layerNormOffset", "qProjection", "qBias",
  "kProjection", "vProjection", "gatingQuery", "outputProjection",
];
const ORDER = SINGLE_ATTENTION_ORDER;

/**
 * @param {"f32"|"f16"} precision the element the packed buffer holds. Offsets
 *   are in elements and do not depend on it; the shader has to be built for the
 *   same word or it reads half the values at twice the stride.
 */
export function packSingleAttentionWeights(weights, precision = "f32") {
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    if (weights[name] === undefined) throw new Error(`single attention weights missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = concatenateAs(precision, total, (target) => {
    for (const name of ORDER) writeInto(target, weights[name], offsets[name]);
  });
  return { data, offsets };
}

/**
 * How many workgroups share one token's projection, given how many tokens there
 * are.
 *
 * 🔴 THE TOKEN COUNT IS THE OCCUPANCY HERE, AND IT IS TINY. One workgroup a
 * token launches 59 of them on a 59-residue chain - 3,776 invocations, on a
 * part that wants tens of thousands - and the kernel measured 163 GFLOP/s, a
 * twentieth of what the pair track's projections reach. There are no more
 * tokens to hand out, so the only axis left is the output width: each split
 * repeats the token's LayerNorm (384 reads, against the 590k weight reads the
 * projection does) and takes a slice of q, k, v and the gate.
 *
 * 🔴 AND IT IS A FUNCTION OF n, NOT A CONSTANT, because splitting costs the
 * register blocking it buys occupancy with - each invocation then accumulates
 * fewer outputs from the same activation read. Measured (tools/gpu/
 * bench-single-project.js), in milliseconds, best in each row marked:
 *
 *     n=30    1: 0.275   2: 0.150   *3: 0.119   6: 0.150
 *     n=59    1: 0.306  *2: 0.175    3: 0.238   6: 0.263
 *     n=100   1: 0.363  *2: 0.325    3: 0.425   6: 0.425
 *     n=128  *1: 0.369   2: 0.519    3: 0.525   6: 0.538
 *     n=200  *1: 0.675   2: 0.781    3: 0.813
 *     n=500  *1: 1.481   2: 2.000    3: 2.019
 *
 * Every one of those is the smallest split that reaches about 110 workgroups,
 * which is what this returns.
 */
export function singleProjectSplits(n, width) {
  for (const splits of [1, 2, 3]) {
    if (width % (64 * splits) !== 0) continue;
    if (n * splits >= 110) return splits;
  }
  return width % (64 * 3) === 0 ? 3 : 1;
}

export function createSingleAttentionShaders(shape, offsets, epsilon, variance) {
  const { n, channels, heads, dimension } = shape;
  const width = heads * dimension;
  // 🔴 A STORAGE FORMAT FOR THE WEIGHTS, FOR MEMORY AND NOT FOR TIME. `w.single`
  // is 135 MiB of the 567 an AF3 TRUNK keeps resident (a whole fold held 1406,
  // of which this is one tensor of three) - the single
  // track runs 384 channels where the pair track runs 128 - and every read here
  // is a scalar, so halving the bytes halves the residency and changes the time
  // by nothing. Reads are widened at the point of use; the arithmetic is f32
  // throughout. See the note in src/af3/pairformer-block-webgpu.js.
  const weightPrecision = shape.weightPrecision ?? "f32";
  if (!["f32", "f16"].includes(weightPrecision)) {
    throw new RangeError(`unknown single attention weight precision ${weightPrecision}`);
  }
  const weight16 = weightPrecision === "f16";
  const wf = (e) => (weight16 ? `f32(${e})` : e);
  const enableF16 = weight16 ? "enable f16;\n" : "";
  // 🔴 RESOLVED ONCE AND RETURNED, because the dispatch multiplies by it. A
  // caller reading the constant while the shader was generated from something
  // else would project a slice of the width and leave the rest as it found it.
  const splits = shape.projectSplits ?? singleProjectSplits(n, width);
  if (width % (64 * splits) !== 0) {
    throw new Error(`width ${width} is not a multiple of 64 * ${splits} splits`);
  }
  const perSplit = width / splits;
  const perThread = perSplit / 64;

  const common = `${enableF16}
const N: u32 = ${n}u;
const CHANNELS: u32 = ${channels}u;
const HEADS: u32 = ${heads}u;
const DIMENSION: u32 = ${dimension}u;
const WIDTH: u32 = ${width}u;
const EPSILON: f32 = ${epsilon};
const SCALE: f32 = ${1 / Math.sqrt(dimension)};
const W_NORM_SCALE: u32 = ${offsets.layerNormScale}u;
const W_NORM_OFFSET: u32 = ${offsets.layerNormOffset}u;
const W_Q: u32 = ${offsets.qProjection}u;
const W_QBIAS: u32 = ${offsets.qBias}u;
const W_K: u32 = ${offsets.kProjection}u;
const W_V: u32 = ${offsets.vProjection}u;
const W_GATE: u32 = ${offsets.gatingQuery}u;
const W_OUT: u32 = ${offsets.outputProjection}u;

fn logistic(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
`;

  // One workgroup per token: normalise, then q/k/v/gate for that token.
  const project = `${common}
@group(0) @binding(0) var<storage, read> single: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> gate: array<f32>;

var<workgroup> act: array<f32, ${channels}>;
var<workgroup> reduce_a: array<f32, 64>;
var<workgroup> reduce_b: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x / ${splits}u;
  let split = group.x % ${splits}u;
  if (token >= N) { return; }
  let local = local_id.x;
  let base = token * CHANNELS;

  var total = 0.0;
  var squares = 0.0;
  for (var c = local; c < CHANNELS; c += 64u) {
    let value = single[base + c];
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
  let mean = reduce_a[0] / f32(CHANNELS);
  ${variance === "fast"
    ? "let variance = reduce_b[0] / f32(CHANNELS) - mean * mean;"
    : `var centered = 0.0;
  for (var c = local; c < CHANNELS; c += 64u) {
    let d = single[base + c] - mean;
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
  let variance = reduce_a[0] / f32(CHANNELS);`}
  let inverse_std = inverseSqrt(variance + EPSILON);
  workgroupBarrier();

  for (var c = local; c < CHANNELS; c += 64u) {
    act[c] = (single[base + c] - mean) * inverse_std * ${wf("weights[W_NORM_SCALE + c]")}
      + ${wf("weights[W_NORM_OFFSET + c]")};
  }
  workgroupBarrier();

  // All four are (channels, width): channels first, no transpose_weights.
  //
  // 🔴 EVERY OUTPUT THIS INVOCATION OWNS IS ACCUMULATED AT ONCE, so the
  // normalised token is read from workgroup memory once for all of them rather
  // than once each. Output-outer, channel-inner, it read CHANNELS activations
  // per output; blocked, it reads them once and buys PER_THREAD * 4 times the
  // multiply-adds from them.
  let out0 = split * ${perSplit}u + local;
  var q_total: array<f32, ${perThread}>;
  var k_total: array<f32, ${perThread}>;
  var v_total: array<f32, ${perThread}>;
  var gate_total: array<f32, ${perThread}>;
  for (var b = 0u; b < ${perThread}u; b += 1u) {
    q_total[b] = ${wf("weights[W_QBIAS + out0 + b * 64u]")};   // ...and only q has one.
    k_total[b] = 0.0;
    v_total[b] = 0.0;
    gate_total[b] = 0.0;
  }
  for (var c = 0u; c < CHANNELS; c += 1u) {
    let x = act[c];
    let row = c * WIDTH;
    for (var b = 0u; b < ${perThread}u; b += 1u) {
      let out = row + out0 + b * 64u;
      q_total[b] += x * ${wf("weights[W_Q + out]")};
      k_total[b] += x * ${wf("weights[W_K + out]")};
      v_total[b] += x * ${wf("weights[W_V + out]")};
      gate_total[b] += x * ${wf("weights[W_GATE + out]")};
    }
  }
  for (var b = 0u; b < ${perThread}u; b += 1u) {
    let index = token * WIDTH + out0 + b * 64u;
    q[index] = q_total[b];
    k[index] = k_total[b];
    v[index] = v_total[b];
    gate[index] = gate_total[b];
  }
}`;

  // One workgroup per (i, head).
  const attend = `${common}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> pair_logits: array<f32>;
@group(0) @binding(4) var<storage, read> seq_mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> gathered: array<f32>;

var<workgroup> logits: array<f32, ${n}>;
var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let slot = group.x;
  if (slot >= N * HEADS) { return; }
  let head = slot % HEADS;
  let i = slot / HEADS;
  let local = local_id.x;
  let query_base = i * WIDTH + head * DIMENSION;

  for (var j = local; j < N; j += 64u) {
    let key_base = j * WIDTH + head * DIMENSION;
    var dot = 0.0;
    for (var d = 0u; d < DIMENSION; d += 1u) {
      dot += q[query_base + d] * k[key_base + d];
    }
    logits[j] = dot * SCALE + pair_logits[head * N * N + i * N + j]
      + 1.0e9 * (seq_mask[j] - 1.0);
  }
  workgroupBarrier();

  var local_max = -3.0e38;
  for (var j = local; j < N; j += 64u) { local_max = max(local_max, logits[j]); }
  reduce[local] = local_max;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
    workgroupBarrier();
  }
  let largest = reduce[0];
  workgroupBarrier();

  var local_sum = 0.0;
  for (var j = local; j < N; j += 64u) {
    let weight = exp(logits[j] - largest);
    logits[j] = weight;
    local_sum += weight;
  }
  reduce[local] = local_sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let total = reduce[0];
  workgroupBarrier();

  for (var d = local; d < DIMENSION; d += 64u) {
    var sum = 0.0;
    for (var j = 0u; j < N; j += 1u) {
      sum += logits[j] * v[j * WIDTH + head * DIMENSION + d];
    }
    gathered[i * WIDTH + head * DIMENSION + d] = sum / total;
  }
}`;

  const project_out = `${common}
@group(0) @binding(0) var<storage, read> gathered: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> gated: array<f32, ${width}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x;
  if (token >= N) { return; }
  let local = local_id.x;
  for (var w = local; w < WIDTH; w += 64u) {
    let index = token * WIDTH + w;
    gated[w] = gathered[index] * logistic(gate[index]);
  }
  workgroupBarrier();

  for (var c = local; c < CHANNELS; c += 64u) {
    var sum = 0.0;
    for (var wi = 0u; wi < WIDTH; wi += 1u) {
      sum += gated[wi] * ${wf("weights[W_OUT + wi * CHANNELS + c]")};
    }
    output[token * CHANNELS + c] = sum;
  }
}`;

  return { project, attend, project_out, projectSplits: splits };
}

export class Af3SingleAttentionGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {Float32Array} single n*channels
   * @param {Float32Array} pairLogits heads*n*n, head-major
   * @param {Float32Array} seqMask n
   * @param {{n: number, channels: number}} shape
   * @param {object} weights heads, dimension, and the eight tensors in ORDER
   * @param {{epsilon?: number, variance?: "fast"|"two-pass"}} options
   */
  async run(single, pairLogits, seqMask, shape, weights, options = {}) {
    const { n, channels } = shape;
    const { heads, dimension } = weights;
    if (!Number.isInteger(heads) || !Number.isInteger(dimension)) {
      throw new Error("weights.heads and weights.dimension must be integers");
    }
    const width = heads * dimension;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    if (single.length !== n * channels) {
      throw new Error(`single has ${single.length} elements; expected ${n * channels}`);
    }
    if (pairLogits.length !== heads * n * n) {
      throw new Error(`pairLogits has ${pairLogits.length} elements; expected ${heads * n * n}`);
    }

    const packed = packSingleAttentionWeights(weights);
    const sources = createSingleAttentionShaders(
      { n, channels, heads, dimension }, packed.offsets, epsilon, variance);
    const key = `af3-single:${n}:${channels}:${heads}:${dimension}:${epsilon}:${variance}`;
    const [project, attend, projectOut] = await Promise.all([
      this.pipelines.get(`${key}:project`, sources.project),
      this.pipelines.get(`${key}:attend`, sources.attend),
      this.pipelines.get(`${key}:project-out`, sources.project_out),
    ]);

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const singleBuffer = keep(this.allocator.upload("af3-single.input", single, storage));
      const logitsBuffer = keep(this.allocator.upload("af3-single.pair-logits", pairLogits, storage));
      const maskBuffer = keep(this.allocator.upload("af3-single.mask", seqMask, storage));
      const weightBuffer = keep(this.allocator.upload("af3-single.weights", packed.data, storage));
      const q = keep(this.allocator.allocate("af3-single.q", n * width * 4, storage));
      const k = keep(this.allocator.allocate("af3-single.k", n * width * 4, storage));
      const v = keep(this.allocator.allocate("af3-single.v", n * width * 4, storage));
      const gate = keep(this.allocator.allocate("af3-single.gate", n * width * 4, storage));
      const gathered = keep(this.allocator.allocate("af3-single.gathered", n * width * 4, storage));
      const output = keep(this.allocator.allocate(
        "af3-single.output", n * channels * 4, storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "af3-single.readback", n * channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-single-attention" });
      const runPass = (label, pipeline, buffers, groups) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(groups);
        pass.end();
      };

      runPass("project", project, [singleBuffer, weightBuffer, q, k, v, gate],
              n * sources.projectSplits);
      runPass("attend", attend, [q, k, v, logitsBuffer, maskBuffer, gathered], n * heads);
      runPass("project-out", projectOut, [gathered, gate, weightBuffer, output], n);
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, n * channels * 4);

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
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
