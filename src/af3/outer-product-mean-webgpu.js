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

/**
 * How many token pairs one contraction workgroup carries. Four is a vec4, which
 * is what keeps the accumulation one instruction; see the note on the kernel.
 */
export const OPM_PAIRS_PER_GROUP = 4;

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

  // 🔴 EVERY WORKGROUP READ THE WHOLE OUTPUT PROJECTION TO PRODUCE ONE PAIR.
  // The projection is PRODUCTS x C_Z - 131,072 floats, half a megabyte - and
  // with a workgroup per token pair that is 11.8 GB a call at 150 tokens, which
  // at the 445 GB/s tools/gpu/probe-alu.js measures for cached global reads is
  // 26 ms: the whole of what this kernel cost there. A tile of pairs divides it
  // by the tile, and the pairs share nothing else, so the arithmetic is
  // untouched.
  //
  // 🔴 AND THE TILE IS THE VECTOR, so the accumulation stays one instruction:
  // a cell's four pairs are one vec4, one weight read serves all of them, and
  // the workgroup memory holding them is a CHUNK of cells rather than all of
  // them - PRODUCTS x PAIRS floats would be 16 KB and this way it is 4.
  const pairsPerGroup = shape.pairsPerGroup ?? OPM_PAIRS_PER_GROUP;
  const cellChunk = Math.min(products, 256);
  if (products % cellChunk !== 0) {
    throw new Error(`PRODUCTS ${products} is not a multiple of the cell chunk ${cellChunk}`);
  }
  const pairVector = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[pairsPerGroup];
  if (pairVector === undefined) {
    throw new Error(`pairsPerGroup ${pairsPerGroup} is not 1, 2 or 4`);
  }
  // 🔴 AND SPLITTING THAT LOOP ACROSS THE IDLE LANES WAS TRIED AND LOST. It
  // walks C_Z output channels with 256 invocations and C_Z is 128, so half of
  // them sit out the loop this kernel spends its time in - which looks like a
  // free doubling. Giving two partitions half the cells each and adding their
  // partials once at the end measured 86.3 ms against 58.8 at 150 tokens. The
  // idle half was costing nothing to begin with: those are whole subgroups
  // taking a uniform branch, so they were never issuing, and the split bought
  // no instructions while paying a barrier and a 4 KB round trip through
  // workgroup memory.
  const pairAt = (name, p) => pairsPerGroup === 1 ? name : `${name}.${"xyzw"[p]}`;
  const overPairs = (body) =>
    Array.from({ length: pairsPerGroup }, (_, p) => body(p)).join("\n      ");

  // One workgroup per tile of token pairs: accumulate the 32x32 product over
  // sequences in workgroup memory, then map it through output_w.
  //
  // 🔴 256 LANES, NOT 64, AND THAT IS THE ONLY THING HERE THAT PAID. A lane
  // owns PRODUCTS/lanes cells and sweeps every sequence for each, so widening
  // the workgroup divides the cells a lane carries rather than the work it
  // does - the same trick the diffusion transformer's per-token kernels wanted.
  // Measured at 1024 rows, on the msa-stack as a whole:
  //     64 lanes 474 ms    128 lanes 401    256 lanes 386
  //
  // 🔴 STAGING THE ROWS WAS TRIED AND LOST, and that is a different redundancy
  // from the one above. Each lane owns PRODUCTS/256 cells and walks all
  // SEQUENCES rows for every one, so a workgroup reads 2 x PRODUCTS x SEQUENCES
  // floats to consume 2 x C_OUTER x SEQUENCES distinct ones. Staging each chunk
  // of rows' two 32-float slices removes exactly that, and at 1024 rows it went
  // 469 ms to 673, 44% WORSE: the tiled form needs a per-lane accumulator ARRAY
  // indexed by a loop variable, which WGSL puts in spillable local memory, and
  // it pays two barriers a chunk. Those reads were cache-served anyway.
  const contract = `${common}
const PAIRS_PER_GROUP: u32 = ${pairsPerGroup}u;
const CELL_CHUNK: u32 = ${cellChunk}u;

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read> msa_mask: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> product: array<${pairVector}, ${cellChunk}>;
var<workgroup> reduce: array<${pairVector}, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let first = (group.x + group.y * GRID_WIDTH) * PAIRS_PER_GROUP;
  if (first >= TOKENS * TOKENS) { return; }
  let local = local_id.x;

  // ...a slot past the end is clamped rather than skipped, so its lane of every
  // vector below holds a real number; it is dropped at the write.
  ${overPairs((p) => `let slot${p} = min(first + ${p}u, TOKENS * TOKENS - 1u);
  let i${p} = slot${p} / TOKENS;
  let j${p} = slot${p} % TOKENS;`)}

  // 🔴 THE DENOMINATOR COUNTS SEQUENCES COVERING BOTH TOKENS.
  var count: ${pairVector};
  ${overPairs((p) => `${pairAt("count", p)} = 0.0;`)}
  for (var s = local; s < SEQUENCES; s += 256u) {
    ${overPairs((p) =>
      `${pairAt("count", p)} += msa_mask[s * TOKENS + i${p}] * msa_mask[s * TOKENS + j${p}];`)}
  }
  reduce[local] = count;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let scale = ${pairVector}(1.0) / (${pairVector}(NORM_EPSILON) + reduce[0]);

  var totals: ${pairVector};
  let has_f = local < C_Z;
  let f = select(0u, local, has_f);
  ${overPairs((p) => `${pairAt("totals", p)} = weights[W_OUT_BIAS + f];`)}

  for (var chunk0 = 0u; chunk0 < PRODUCTS; chunk0 += CELL_CHUNK) {
    // ...before overwriting the chunk the previous iteration is still reading.
    workgroupBarrier();
    // Each invocation owns one cell of the chunk for the whole sweep, so the
    // accumulator stays in registers and lands in workgroup memory once.
    for (var index = local; index < CELL_CHUNK; index += 256u) {
      let cell = chunk0 + index;
      let c = cell / C_OUTER;
      let e = cell % C_OUTER;
      var total: ${pairVector};
      ${overPairs((p) => `${pairAt("total", p)} = 0.0;`)}
      for (var s = 0u; s < SEQUENCES; s += 1u) {
        let row = s * TOKENS;
        ${overPairs((p) =>
          `${pairAt("total", p)} += left[(row + i${p}) * C_OUTER + c]
            * right[(row + j${p}) * C_OUTER + e];`)}
      }
      product[index] = total;
    }
    workgroupBarrier();

    // 🔴 ONE WEIGHT READ SERVES EVERY PAIR IN THE TILE. This loop is what the
    // kernel spends its time in at shallow depth - PRODUCTS x C_Z of it - and
    // the tile is the only thing that divides it.
    if (has_f) {
      for (var t = 0u; t < CELL_CHUNK; t += 1u) {
        totals += product[t] * weights[W_OUT + (chunk0 + t) * C_Z + f];
      }
    }
  }

  if (has_f) {
    // ...scaled after the projection, so the bias is scaled with it.
    let scaled = totals * scale;
    ${overPairs((p) => `if (first + ${p}u < TOKENS * TOKENS) {
      output[slot${p} * C_Z + f] = ${pairAt("scaled", p)};
    }`)}
  }
}`;

  return { project, contract, pairsPerGroup };
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
      // ...one workgroup per TILE of pairs; the shaders report the tile so the
      // dispatch cannot divide by a different one.
      const contractGroups = spread(Math.ceil(tokens * tokens / sources.pairsPerGroup));
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
