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
 * The block of (i, j) token pairs one contraction workgroup carries. The j side
 * is a vector, so it is 1, 2 or 4; see the note on the kernel.
 *
 * 🔴 TWO BY FOUR, NOT FOUR BY FOUR, because the workgroup count still wins at
 * these sizes: measured at 59 tokens and 1024 rows, 2x4 is 15.0 ms against
 * 4x4's 16.3, 8x4's 43 and 1x4's 23.4; at 150 tokens and 512 rows, 55.0
 * against 56.3 and 75.6. The block is what divides the sweep over sequences -
 * BLOCK_I + BLOCK_J reads buy BLOCK_I * BLOCK_J multiply-adds - and the
 * workgroups are what fill the machine.
 */
export const OPM_BLOCK_I = 2;
export const OPM_BLOCK_J = 4;

/**
 * How many of the 1024 products are held in workgroup memory at once.
 *
 * 🔴 THIS IS WORTH MORE THAN THE BLOCK, WHICH IS NOT WHAT I EXPECTED. At 59
 * tokens and 1024 rows: 2x4 at a chunk of 128 is 24.8 ms, at 256 it is 15.0,
 * at 512 it is 19.5. The same shape at 150 tokens and 512 rows: 101.8, 55.0,
 * 61.1. It sets how many sequences' worth of sweep a workgroup does between
 * barriers and how much of the output projection it can amortise, against the
 * residency that 256 x BLOCK_I vec4 of workgroup memory costs.
 */
export const OPM_CELL_CHUNK = 256;

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

  // 🔴 A BLOCK OF (i, j) PAIRS, NOT A RUN OF SLOTS, AND THAT IS THE WHOLE
  // KERNEL. Two things cost here and the block divides both.
  //
  // The output projection is PRODUCTS x C_Z - 131,072 floats, half a megabyte -
  // and with a workgroup per pair that was 11.8 GB a call at 150 tokens, which
  // at the 445 GB/s tools/gpu/probe-alu.js measures for cached global reads is
  // 26 ms: the whole of what this kernel cost at shallow depth. Any tile of
  // pairs divides that.
  //
  // The sweep over sequences is what costs at DEPTH - the MSA stack is about
  // 72 ms plus 0.38 ms a row - and only a two-dimensional block divides that,
  // because a cell's product is left[i][c] * right[j][e]: an i-by-j block of
  // pairs reads BLOCK_I values of left and BLOCK_J of right and multiplies them
  // into BLOCK_I * BLOCK_J products. A run of consecutive SLOTS shares nothing
  // the compiler can see, since a slot's i is slot / TOKENS.
  //
  // 4x4 reads eight values to buy sixteen multiply-adds, against the two reads
  // for one that a workgroup per pair did.
  const blockI = shape.blockI ?? OPM_BLOCK_I;
  const blockJ = shape.blockJ ?? OPM_BLOCK_J;
  const cellChunk = Math.min(products, shape.cellChunk ?? OPM_CELL_CHUNK);
  if (products % cellChunk !== 0) {
    throw new Error(`PRODUCTS ${products} is not a multiple of the cell chunk ${cellChunk}`);
  }
  const jVector = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[blockJ];
  if (jVector === undefined) throw new Error(`blockJ ${blockJ} is not 1, 2 or 4`);
  const overI = (body) => Array.from({ length: blockI }, (_, i) => body(i)).join("\n      ");
  const overJ = (body) => Array.from({ length: blockJ }, (_, j) => body(j)).join("\n      ");
  const jAt = (name, j) => blockJ === 1 ? name : `${name}.${"xyzw"[j]}`;
  const blocksPerRow = Math.ceil(tokens / blockJ);

  // 🔴 THE DENOMINATOR IS ITS OWN PASS NOW. It counts sequences covering both
  // tokens, which is a reduction over SEQUENCES per pair - and with sixteen
  // pairs a workgroup, doing it inside the contraction wanted sixteen partial
  // sums per invocation through workgroup memory. It is PAIRS x SEQUENCES
  // multiply-adds in total, a thousandth of the contraction's, so it is
  // cheaper to compute once into a buffer than to carry.
  const counts = `${common}
@group(0) @binding(0) var<storage, read> msa_mask: array<f32>;
@group(0) @binding(1) var<storage, read_write> counts: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x + id.y * GRID_WIDTH * 64u;
  if (slot >= TOKENS * TOKENS) { return; }
  let i = slot / TOKENS;
  let j = slot % TOKENS;
  var total = 0.0;
  for (var s = 0u; s < SEQUENCES; s += 1u) {
    total += msa_mask[s * TOKENS + i] * msa_mask[s * TOKENS + j];
  }
  counts[slot] = total;
}`;

  // 256 lanes, not 64: a lane owns PRODUCTS/lanes cells and sweeps every
  // sequence for each, so widening the workgroup divides the cells a lane
  // carries rather than the work it does. Measured at 1024 rows, on the
  // msa-stack as a whole: 64 lanes 474 ms, 128 lanes 401, 256 lanes 386.
  //
  // 🔴 STAGING THE ROWS WAS TRIED AND LOST, and it is a different redundancy
  // from the one the block above fixes. Each lane owns PRODUCTS/256 cells and
  // walks all SEQUENCES rows for every one, so a workgroup reads 2 x PRODUCTS x
  // SEQUENCES floats to consume 2 x C_OUTER x SEQUENCES distinct ones. Staging
  // each chunk of rows' two 32-float slices removes exactly that, and at 1024
  // rows it went 469 ms to 673, 44% WORSE: the tiled form needs a per-lane
  // accumulator ARRAY indexed by a loop variable, which WGSL puts in spillable
  // local memory, and it pays two barriers a chunk. Those reads were
  // cache-served anyway.
  //
  // 🔴 AND THERE IS A RESIDENCY REASON IT CANNOT WIN, WHICH IS WORTH KNOWING
  // BEFORE ANYONE TRIES A THIRD TIME. `product` is already CELL_CHUNK x BLOCK_I
  // vectors - 8 KiB at 256 cells and a 2x4 block - so this kernel holds four
  // workgroups a core on a device granting 32 KiB. Staging the rows needs the
  // chunk's left and right slices on top: about 4.5 KiB at eight sequences,
  // 9 KiB at sixteen, which takes it to two workgroups a core or one. The
  // instruction count only improves from 1.0 useful operations to about 1.45
  // once the staging loop is paid for, and this device has traded four
  // workgroups for a 45% instruction cut before and lost.
  //
  // Measured elsewhere on the same day: the kernels that DID answer to staging
  // - the atom blocks' conditioning, ffw-out, attention-output - all reused an
  // array that was already dead, and cost no residency at all. That is the
  // distinction, not whether the reads look redundant.
  const contract = `${common}
const CELL_CHUNK: u32 = ${cellChunk}u;
const BLOCKS_PER_ROW: u32 = ${blocksPerRow}u;

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read> counts: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// One chunk of cells, each holding the block's BLOCK_I x BLOCK_J products.
var<workgroup> product: array<${jVector}, ${cellChunk * blockI}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let block = group.x + group.y * GRID_WIDTH;
  let base_i = (block / BLOCKS_PER_ROW) * ${blockI}u;
  let base_j = (block % BLOCKS_PER_ROW) * ${blockJ}u;
  if (base_i >= TOKENS) { return; }
  let local = local_id.x;

  // ...a token past the end is clamped rather than skipped; it is dropped at
  // the write.
  ${overI((i) => `let i${i} = min(base_i + ${i}u, TOKENS - 1u);`)}
  ${overJ((j) => `let j${j} = min(base_j + ${j}u, TOKENS - 1u);`)}

  let f = local;
  let has_f = f < C_Z;
  ${overI((i) => `var totals${i} = ${jVector}(weights[W_OUT_BIAS + select(0u, f, has_f)]);`)}

  for (var chunk0 = 0u; chunk0 < PRODUCTS; chunk0 += CELL_CHUNK) {
    // ...before overwriting the chunk the previous iteration is still reading.
    workgroupBarrier();
    for (var index = local; index < CELL_CHUNK; index += 256u) {
      let cell = chunk0 + index;
      let c = cell / C_OUTER;
      let e = cell % C_OUTER;
      ${overI((i) => `var acc${i} = ${jVector}(0.0);`)}
      for (var s = 0u; s < SEQUENCES; s += 1u) {
        let row = s * TOKENS;
        // BLOCK_I + BLOCK_J reads buy BLOCK_I * BLOCK_J multiply-adds.
        ${overI((i) => `let l${i} = left[(row + i${i}) * C_OUTER + c];`)}
        var r: ${jVector};
        ${overJ((j) => `${jAt("r", j)} = right[(row + j${j}) * C_OUTER + e];`)}
        ${overI((i) => `acc${i} += l${i} * r;`)}
      }
      ${overI((i) => `product[${i}u * CELL_CHUNK + index] = acc${i};`)}
    }
    workgroupBarrier();

    if (has_f) {
      for (var t = 0u; t < CELL_CHUNK; t += 1u) {
        let w = weights[W_OUT + (chunk0 + t) * C_Z + f];
        ${overI((i) => `totals${i} += product[${i}u * CELL_CHUNK + t] * w;`)}
      }
    }
  }

  if (has_f) {
    ${overI((i) => overJ((j) => `{
      let slot = i${i} * TOKENS + j${j};
      if (base_i + ${i}u < TOKENS && base_j + ${j}u < TOKENS) {
        // ...scaled after the projection, so the bias is scaled with it.
        output[slot * C_Z + f] = ${jAt(`totals${i}`, j)}
          / (NORM_EPSILON + counts[slot]);
      }
    }`))}
  }
}`;

  return { project, counts, contract, blockI, blockJ, blocksPerRow };
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
    const [project, counts, contract] = await Promise.all([
      this.pipelines.get(`${key}:project`, sources.project),
      this.pipelines.get(`${key}:counts`, sources.counts),
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
      const countBuffer = keep(this.allocator.allocate(
        "af3-opm.counts", tokens * tokens * 4, storage));
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
      const countGroups = spread(Math.ceil(tokens * tokens / 64));
      run("opm.counts", counts, [maskBuffer, countBuffer], countGroups[0], countGroups[1]);
      // ...one workgroup per (i, j) BLOCK of pairs; the shaders report the block
      // so the dispatch cannot divide by a different one.
      const contractGroups = spread(
        Math.ceil(tokens / sources.blockI) * sources.blocksPerRow);
      run("opm.contract", contract, [left, right, countBuffer, weightBuffer, output],
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
