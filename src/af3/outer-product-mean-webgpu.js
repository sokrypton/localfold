/**
 * AF3's outer product mean: the MSA's only write into the pair representation.
 *
 *     msa -> LayerNorm -> left (32), right (32), both masked AFTER projection
 *     P[i][j][c][e] = sum over sequences of left[s,i,c] * right[s,j,e]
 *     pair[i][j][f] = (output_b[f] + sum_ce P[c][e] * output_w[c,e,f]) / (1e-3 + n_ij)
 *
 * 🔴 THE MEAN'S DENOMINATOR IS THE NUMBER OF SEQUENCES COVERING BOTH TOKENS,
 * plus 1e-3 - not the sequence count, and not the count of either token alone.
 * On an MSA with no gaps every version of this agrees, which is exactly the
 * input a hand-written test uses.
 *
 * 🔴 AND IT DIVIDES AFTER THE PROJECTION, not before. Dividing the accumulated
 * product instead changes the result whenever output_b is nonzero, because the
 * bias then gets scaled too.
 *
 * 🔴 THE MASK IS APPLIED AFTER THE PROJECTIONS, on both sides. The product is
 * bilinear, so a masked row contributes nothing to either factor - masking the
 * MSA before the LayerNorm instead would change the normalisation statistics of
 * every row that survives.
 *
 * WHY NOT AF2's KERNEL. AF2's outer product mean is the same operation, masks
 * in the same place, and even shares the 1e-3 epsilon - but it carries tiling
 * for large MSA depth, an alternate outer-first contraction, and a uniform
 * parameter block, and it takes the two-pass LayerNorm variance where AF3's
 * trunk takes the fast one. Adapting it is more work than this, and this can
 * grow the tiling later if AF3's depths need it.
 *
 * COST. Per token pair the contraction is c_outer^2 * c_z = 32*32*128 = 131,072
 * multiply-adds, and the accumulation is c_outer^2 per sequence. Both are
 * quadratic in tokens, so this is the second most expensive thing in the trunk
 * after triangle multiplication, and it runs four times rather than 48.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const GRID_WIDTH = 32_768;

const ORDER = [
  "layerNormInputScale", "layerNormInputOffset",
  "leftProjection", "rightProjection", "outputW", "outputB",
];

export function packOuterProductMeanWeights(weights) {
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    if (weights[name] === undefined) throw new Error(`outer product mean missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of ORDER) data.set(weights[name], offsets[name]);
  return { data, offsets };
}

export function createOuterProductMeanShaders(shape, offsets, epsilon, variance) {
  const { sequences, tokens, msaChannels, outerChannels, pairChannels } = shape;
  const rows = sequences * tokens;
  const products = outerChannels * outerChannels;

  const common = `
const SEQUENCES: u32 = ${sequences}u;
const TOKENS: u32 = ${tokens}u;
const ROWS: u32 = ${rows}u;
const C_M: u32 = ${msaChannels}u;
const C_OUTER: u32 = ${outerChannels}u;
const C_Z: u32 = ${pairChannels}u;
const PRODUCTS: u32 = ${products}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
const NORM_EPSILON: f32 = 1.0e-3;
const W_SCALE: u32 = ${offsets.layerNormInputScale}u;
const W_OFFSET: u32 = ${offsets.layerNormInputOffset}u;
const W_LEFT: u32 = ${offsets.leftProjection}u;
const W_RIGHT: u32 = ${offsets.rightProjection}u;
const W_OUT: u32 = ${offsets.outputW}u;
const W_OUT_BIAS: u32 = ${offsets.outputB}u;
`;

  // LayerNorm and both projections, masked on the way out.
  const project = `${common}
@group(0) @binding(0) var<storage, read> msa: array<f32>;
@group(0) @binding(1) var<storage, read> msa_mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> left: array<f32>;
@group(0) @binding(4) var<storage, read_write> right: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= ROWS) { return; }
  let base = row * C_M;
  var total = 0.0;
  var squares = 0.0;
  for (var c = 0u; c < C_M; c += 1u) {
    let value = msa[base + c];
    total += value;
    squares += value * value;
  }
  let mean = total / f32(C_M);
  ${variance === "fast"
    ? "let variance = squares / f32(C_M) - mean * mean;"
    : `var variance = 0.0;
  for (var c = 0u; c < C_M; c += 1u) {
    let d = msa[base + c] - mean;
    variance += d * d;
  }
  variance /= f32(C_M);`}
  let inverse_std = inverseSqrt(variance + EPSILON);
  let keep = msa_mask[row];
  for (var o = 0u; o < C_OUTER; o += 1u) {
    var left_total = 0.0;
    var right_total = 0.0;
    for (var c = 0u; c < C_M; c += 1u) {
      let value = (msa[base + c] - mean) * inverse_std * weights[W_SCALE + c]
        + weights[W_OFFSET + c];
      left_total += value * weights[W_LEFT + c * C_OUTER + o];
      right_total += value * weights[W_RIGHT + c * C_OUTER + o];
    }
    // ...masked here, after the projection, on both sides.
    left[row * C_OUTER + o] = keep * left_total;
    right[row * C_OUTER + o] = keep * right_total;
  }
}`;

  // One workgroup per token pair: accumulate the 32x32 product over sequences
  // in workgroup memory, then map it through output_w.
  //
  // 🔴 THIS IS THE HOT KERNEL AT DEPTH, AND STAGING THE ROWS WAS TRIED AND LOST.
  // The MSA stack costs about 72 ms fixed plus 0.38 ms a row, so at AF3's own
  // num_msa of 1024 it is 469 ms - 43% of a trunk pass - and this contraction is
  // half of that.
  //
  // The redundancy is real and looks damning: each of the 64 lanes owns
  // PRODUCTS/64 cells and walks all SEQUENCES rows for every one of them, so a
  // workgroup reads 2 x PRODUCTS x SEQUENCES floats to consume 2 x C_OUTER x
  // SEQUENCES distinct ones - 32-fold. Staging each chunk of rows' two 32-float
  // slices in workgroup memory removes exactly that, and measured at 1024 rows
  // it went 469 ms to 673, 44% WORSE. Two reasons, both structural: the tiled
  // form needs a per-lane accumulator ARRAY indexed by a loop variable, which
  // WGSL puts in spillable local memory rather than registers, where the
  // straight version keeps one scalar `total` live; and it pays two workgroup
  // barriers per chunk. The reads it saves were being served by cache anyway -
  // one workgroup's slice of `left` is 32 floats a row, 128 KiB at 1024 rows.
  //
  // Staging ALL the rows instead, so the cell loop could keep its scalar, wants
  // 2 x SEQUENCES x C_OUTER floats - 256 KiB at 1024 rows against a 32 KiB
  // limit - so the two ways out of the redundancy are mutually exclusive here.
  const contract = `${common}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read> msa_mask: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> product: array<f32, ${products}>;
var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let slot = group.x + group.y * GRID_WIDTH;
  if (slot >= TOKENS * TOKENS) { return; }
  let i = slot / TOKENS;
  let j = slot % TOKENS;
  let local = local_id.x;

  for (var index = local; index < PRODUCTS; index += 64u) { product[index] = 0.0; }
  workgroupBarrier();

  // Each invocation owns a fixed set of (c, e) cells for the whole sweep, so
  // the accumulator stays in registers and only lands in workgroup memory once.
  var count = 0.0;
  for (var index = local; index < PRODUCTS; index += 64u) {
    let c = index / C_OUTER;
    let e = index % C_OUTER;
    var total = 0.0;
    for (var s = 0u; s < SEQUENCES; s += 1u) {
      total += left[(s * TOKENS + i) * C_OUTER + c] * right[(s * TOKENS + j) * C_OUTER + e];
    }
    product[index] = total;
  }
  // 🔴 THE DENOMINATOR COUNTS SEQUENCES COVERING BOTH TOKENS.
  for (var s = local; s < SEQUENCES; s += 64u) {
    count += msa_mask[s * TOKENS + i] * msa_mask[s * TOKENS + j];
  }
  reduce[local] = count;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let scale = 1.0 / (NORM_EPSILON + reduce[0]);

  for (var f = local; f < C_Z; f += 64u) {
    var total = weights[W_OUT_BIAS + f];
    for (var index = 0u; index < PRODUCTS; index += 1u) {
      total += product[index] * weights[W_OUT + index * C_Z + f];
    }
    // ...scaled after the projection, so the bias is scaled with it.
    output[slot * C_Z + f] = total * scale;
  }
}`;

  return { project, contract };
}

export class Af3OuterProductMeanGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {Float32Array} msa sequences*tokens*msaChannels
   * @param {Float32Array} msaMask sequences*tokens
   * @param {{sequences: number, tokens: number, msaChannels: number,
   *          pairChannels: number}} shape
   * @param {object} weights outerChannels and the six tensors in ORDER
   */
  async run(msa, msaMask, shape, weights, options = {}) {
    const { sequences, tokens, msaChannels, pairChannels } = shape;
    const outerChannels = weights.outerChannels;
    if (!Number.isInteger(outerChannels)) {
      throw new Error("weights.outerChannels must be an integer");
    }
    const rows = sequences * tokens;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    if (msa.length !== rows * msaChannels) {
      throw new Error(`msa has ${msa.length} elements; expected ${rows * msaChannels}`);
    }

    const packed = packOuterProductMeanWeights(weights);
    const full = { sequences, tokens, msaChannels, outerChannels, pairChannels };
    const sources = createOuterProductMeanShaders(full, packed.offsets, epsilon, variance);
    const key = `af3-opm:${sequences}:${tokens}:${msaChannels}:${outerChannels}`
      + `:${pairChannels}:${epsilon}:${variance}`;
    const [project, contract] = await Promise.all([
      this.pipelines.get(`${key}:project`, sources.project),
      this.pipelines.get(`${key}:contract`, sources.contract),
    ]);

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const msaBuffer = keep(this.allocator.upload("af3-opm.msa", msa, storage));
      const maskBuffer = keep(this.allocator.upload("af3-opm.mask", msaMask, storage));
      const weightBuffer = keep(this.allocator.upload("af3-opm.weights", packed.data, storage));
      const left = keep(this.allocator.allocate("af3-opm.left", rows * outerChannels * 4, storage));
      const right = keep(this.allocator.allocate("af3-opm.right", rows * outerChannels * 4, storage));
      const output = keep(this.allocator.allocate(
        "af3-opm.output", tokens * tokens * pairChannels * 4,
        storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "af3-opm.readback", tokens * tokens * pairChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-opm" });
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
      const projectGroups = spread(Math.ceil(rows / 64));
      run("opm.project", project, [msaBuffer, maskBuffer, weightBuffer, left, right],
          projectGroups[0], projectGroups[1]);
      const contractGroups = spread(tokens * tokens);
      run("opm.contract", contract, [left, right, maskBuffer, weightBuffer, output],
          contractGroups[0], contractGroups[1]);
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0,
                                 tokens * tokens * pairChannels * 4);

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
