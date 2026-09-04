import { concatenateAs } from "../runtime/float16.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const ceilDivide = (value, divisor) => Math.ceil(value / divisor);
const GRID_WIDTH = 32_768;

/**
 * The register block one invocation of the `linear` kernel owns.
 *
 * 🔴 THIS IS THE ONLY PLACE THE TILE IS STATED, AND THAT IS DELIBERATE. Every
 * caller dispatches with TRANSITION_TILE_ROWS/COLUMNS, which are derived from
 * it below, so a tile the dispatch does not match cannot be constructed - the
 * failure that once read as a 30% speedup while half the rows went unprojected.
 *
 * `step` is pinned to `lanesY` because a lane stages the weight row at its own
 * `local.y`; `rowsPerLane` and `columnsPerLane` must be multiples of 4, because
 * both operands are read as vec4 in the inner loop.
 */
export const LINEAR_PRECISIONS = new Set(["f32", "f16", "mixed"]);

export const LINEAR_TILE = {
  lanesX: 8, lanesY: 8, rowsPerLane: 4, columnsPerLane: 4,
};

/**
 * The same tile, twice as wide.
 *
 * 🔴 THE TWO DIFFER ONLY IN COLUMNS, AND THAT IS LOAD-BEARING. Both are 32
 * rows, so TRANSITION_TILE_ROWS is one number whatever the choice - which is
 * what keeps `transitionChunkRows` honest, since a chunk is aligned to the row
 * tile and is computed before any shader exists to ask.
 */
export const LINEAR_TILE_WIDE = { ...LINEAR_TILE, columnsPerLane: 8 };

export const linearTileRows = (tile = LINEAR_TILE) => tile.lanesY * tile.rowsPerLane;
export const linearTileColumns = (tile = LINEAR_TILE) => tile.lanesX * tile.columnsPerLane;

/**
 * Which of the two tiles a dispatch of this shape wants.
 *
 * 🔴 THE AXIS IS OCCUPANCY, NOT SIZE, WHICH IS WHY THE RULE COUNTS WORKGROUPS.
 * The wide tile halves the weight traffic and reads two vec4 of weights for one
 * of source, so it is the better kernel wherever the device is full - 1.18x
 * over the two-row scalar-source kernel at 30,208 rows. It is the WORSE kernel
 * when it is not: at 59 rows and 384 columns it launches twelve workgroups
 * against the narrow tile's twenty-four and measures 0.150 ms against 0.113.
 * Measured either side of the crossing (tools/gpu/bench-evoformer-linear.js,
 * 512 columns): at 512 rows the two tie, at 2,048 the wide tile leads by 7%,
 * at 8,192 by 8%, and that is 512 workgroups.
 *
 * Halving the traffic AGAIN, with a 64-row tile of 128 or 256 lanes, is slower
 * at every shape measured - so this is not a bandwidth story and a third tile
 * is not the next move.
 */
export function chooseLinearTile({ rows, columns }) {
  const wide = Math.ceil(rows / linearTileRows(LINEAR_TILE_WIDE))
    * Math.ceil(columns / linearTileColumns(LINEAR_TILE_WIDE));
  return wide >= 512 ? LINEAR_TILE_WIDE : LINEAR_TILE;
}

/**
 * The tile AND the element its k loop works in, which is one choice.
 *
 * 🔴 IN f16 THE NARROW TILE WINS EVERYWHERE THE WIDE ONE DID, and that inverts
 * the rule above rather than adding to it. The wide tile exists to halve the
 * weight traffic at the cost of workgroups; f16 halves the traffic too, and
 * halves the accumulators, so the narrow tile keeps its occupancy AND gets the
 * cheaper reads. Measured on the transition's own shapes
 * (bench-evoformer-linear.js), best f32 arm against 32x32 in f16:
 *
 *     first,  8192 rows   4.425 -> 3.725 ms      second   15.55 -> 13.45
 *     pair, 150 tokens    3.225 -> 2.850         first, 4096 rows  2.225 -> 1.925
 *
 * (a later run of the same two arms read 4.338 -> 3.675, which is this
 * machine's ~2% between-session drift and not a different result; the arrow is
 * what reproduces, the endpoints are not.)
 *
 * 🔴 AND IT LOSES ON SMALL SHAPES, WHICH IS WHY THIS IS A THRESHOLD AND NOT A
 * SWITCH. At 24 workgroups - a 59-residue structure module's single track -
 * f32 measures 0.188 ms against f16's 0.237, because there is not enough work
 * to hide the conversion. Swept: 32 and 64 workgroups favour f32, 128 and up
 * favour f16, so the crossing is put at 128.
 *
 * The structure module and the confidence head do not come through here at all
 * - they build the default shader once, at module level - so this only ever
 * decides for a block's transitions, which is where the rows are.
 */
export function chooseLinearKernel({ rows, columns, device, requested = "auto" }) {
  if (requested === "f32") {
    return {
      tile: chooseLinearTile({ rows, columns }), precision: "f32", weightPrecision: "f32",
    };
  }
  const narrow = Math.ceil(rows / linearTileRows(LINEAR_TILE))
    * Math.ceil(columns / linearTileColumns(LINEAR_TILE));
  if (requested === "f16") {
    if (device?.features?.has("shader-f16") !== true) {
      throw new Error("the f16 linear kernel requires the shader-f16 feature");
    }
    return { tile: LINEAR_TILE, precision: "f16", weightPrecision: "f16" };
  }
  if (device?.features?.has("shader-f16") === true && narrow >= 128) {
    // 🔴 THE WEIGHT BUFFER NARROWS WITH THE k LOOP, and it is a separate win
    // from it: this kernel re-reads the whole weight set once per row tile -
    // 944 times for a 512-row alignment, about 2 GB against a 2 MiB working
    // set - so halving those bytes is worth 3.675 -> 3.375 ms on top of the
    // 4.338 -> 3.675 the k loop already bought (bench-evoformer-linear.js at
    // 8192 rows), and 13.50 -> 12.24 on the second half. It also halves what
    // AF2 uploads per block, which it does on every pass of every recycle.
    return { tile: LINEAR_TILE, precision: "f16", weightPrecision: "f16" };
  }
  return { tile: chooseLinearTile({ rows, columns }), precision: "f32", weightPrecision: "f32" };
}

export const TRANSITION_TILE_COLUMNS = linearTileColumns();

/**
 * How large a single transition binding is allowed to get.
 *
 * 🔴 A CEILING, NOT A WORKAROUND, AND THAT CHANGED. This used to apply only
 * when the hidden activation could not be BOUND - so a fold that bound fine
 * allocated whatever it liked, and `msa-transition.hidden` was 118 MiB of an
 * AF2 fold's 681 MiB peak at 512 MSA rows: the largest tensor on the device by
 * a factor of four, for a scratch buffer that is read once and thrown away.
 * Chunking it always costs dispatches and nothing else, because the work per
 * row and the weight traffic per workgroup are identical either way.
 *
 * 🔴 AND 32 MiB IS THE KNEE, MEASURED. On a 59-residue fold at 512 MSA rows,
 * as device peak and wall clock (fold-af2.js, one run each, so the times are
 * inside this machine's noise floor and the peaks are exact - they are
 * accounting, not timing):
 *
 *     no cap   681 MiB   5256 ms        32 MiB   573 MiB   5294 ms
 *     64 MiB   613 MiB   5266 ms        16 MiB   553 MiB   5354 ms
 *
 * Every arm folds to the same pLDDT of 67.131. Past 32 the saving flattens and
 * the loop keeps getting longer.
 */
export const TRANSITION_CHUNK_TARGET_BYTES = 32 * 1024 * 1024;

const gcd = (left, right) => {
  let a = left; let b = right;
  while (b !== 0) { const remainder = a % b; a = b; b = remainder; }
  return a;
};

/**
 * How many rows one transition chunk may cover.
 *
 * 🔴 A TENSOR CAN FIT IN A BUFFER AND STILL NOT BE BINDABLE. The transition's
 * hidden activation is rows * hiddenChannels floats - at 508 MSA rows of a
 * 291-residue alignment that is 147,828 * 1024 * 4 bytes, 578 MiB, which
 * allocates on any modern adapter and then exceeds maxStorageBufferBindingSize
 * when it is bound. Splitting the rows is what makes long sequences possible.
 *
 * 🔴 THE CHUNK IS ALIGNED TWICE OVER, and both alignments are load-bearing:
 *   - to TRANSITION_TILE_ROWS, because the linear kernels tile rows by 16 and a
 *     chunk that is not a whole number of tiles would leave a ragged edge;
 *   - to minStorageBufferOffsetAlignment (256 bytes), because each chunk BINDS
 *     at a row offset, and a binding offset that is not a multiple of 256 is a
 *     validation error rather than a slow path.
 * The least common multiple of the two satisfies both at once. `channels * 4`
 * is the row stride the offset is measured in, so how many rows it takes to
 * reach a 256-byte boundary depends on it - hence the gcd.
 *
 * THE FULL PATH IS PRESERVED EXACTLY for anything under the ceiling: this
 * returns `rows` and the caller runs its single-dispatch branch unchanged. What
 * changed is where the ceiling is - see TRANSITION_CHUNK_TARGET_BYTES.
 */
export function transitionChunkRows(
  rows,
  channels,
  hiddenChannels,
  maxStorageBufferBindingSize,
  minStorageBufferOffsetAlignment = 256,
) {
  if (![rows, channels, hiddenChannels, maxStorageBufferBindingSize, minStorageBufferOffsetAlignment]
    .every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("transition chunk dimensions and limits must be positive safe integers");
  }
  const rowBytes = Math.max(channels, hiddenChannels) * Float32Array.BYTES_PER_ELEMENT;
  const ceiling = Math.min(maxStorageBufferBindingSize, TRANSITION_CHUNK_TARGET_BYTES);
  if (rows * rowBytes <= ceiling) return rows;
  const capacity = Math.floor(ceiling / rowBytes);
  if (capacity < 1) throw new RangeError("WebGPU storage binding is too small for one transition row");
  const sourceRowBytes = channels * Float32Array.BYTES_PER_ELEMENT;
  const offsetRowAlignment = minStorageBufferOffsetAlignment
    / gcd(sourceRowBytes, minStorageBufferOffsetAlignment);
  const rowAlignment = TRANSITION_TILE_ROWS * offsetRowAlignment
    / gcd(TRANSITION_TILE_ROWS, offsetRowAlignment);
  if (rows <= capacity) return rows;
  if (capacity < rowAlignment) {
    throw new RangeError("WebGPU storage binding cannot hold one aligned transition chunk");
  }
  return Math.min(rows, Math.floor(capacity / rowAlignment) * rowAlignment);
}
export const TRANSITION_TILE_ROWS = linearTileRows();

function validate(input) {
  const { rows, channels, hiddenChannels, activations, weights } = input;
  if (![rows, channels, hiddenChannels].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("transition dimensions must be positive safe integers");
  }
  const lengths = [
    ["activations", activations, rows * channels],
    ["layerNormScale", weights.layerNormScale, channels],
    ["layerNormOffset", weights.layerNormOffset, channels],
    ["firstWeight", weights.firstWeight, channels * hiddenChannels],
    ["firstBias", weights.firstBias, hiddenChannels],
    ["secondWeight", weights.secondWeight, hiddenChannels * channels],
    ["secondBias", weights.secondBias, channels],
  ];
  for (const [name, value, expected] of lengths) {
    if (value.length !== expected) throw new RangeError(`${name} has ${value.length} values; expected ${expected}`);
  }
}

/**
 * @param {"f32"|"f16"} weightPrecision the element the packed buffer holds. The
 *   offsets are in ELEMENTS and do not depend on it; a caller that packs one way
 *   and builds the shaders the other reads half the values at twice the stride.
 */
export function packTransitionWeights(input, weightPrecision = "f32") {
  const values = [
    input.weights.layerNormScale,
    input.weights.layerNormOffset,
    input.weights.firstWeight,
    input.weights.firstBias,
    input.weights.secondWeight,
    input.weights.secondBias,
  ];
  const offsets = [];
  let length = 0;
  for (const value of values) {
    offsets.push(length);
    length += value.length;
  }
  const data = concatenateAs(weightPrecision, length, (target) => {
    for (let index = 0; index < values.length; index += 1) {
      target.set(values[index], offsets[index]);
    }
  });
  return { data, offsets };
}


/**
 * The register-blocked projection that serves every dense layer in AF2.
 *
 * A workgroup computes a `lanesY*rowsPerLane` by `lanesX*columnsPerLane` output
 * tile; each invocation holds `rowsPerLane * columnsPerLane / 4` vec4
 * accumulators and the k loop reads BOTH operands as vec4.
 *
 * 🔴 THE SOURCE SIDE WAS THE SCALAR HALF, AND THAT WAS THE WHOLE KERNEL. The
 * weight tile has been staged per-thread-as-vec4 for a while; the source tile
 * was still read one float at a time, so an invocation owning two rows spent
 * two workgroup reads and two vec4 weight reads - four reads - to buy four vec4
 * multiply-adds. Staging the source TRANSPOSED, four rows to a vec4, lets one
 * read serve four rows: at eight rows and eight columns a lane now issues four
 * reads for sixteen multiply-adds, 3.2 useful operations an instruction where
 * it was 2.0. tools/gpu/probe-alu.js measures this device at about 640 billion
 * instructions a second whatever their width, so that ratio IS the speed.
 *
 * 🔴 TWO MORE THINGS WERE TRIED ON IT AND BOTH LOST, which puts this kernel at
 * its optimum rather than merely un-examined. Measured as the two MSA
 * transitions of a 512-row block against the f32 baseline of the time, 15.70
 * and 15.62 ms - the same two are 12.17 and 12.58 now, in f16 and split across
 * four chunks each - in runs where the untouched kernels matched to 0.05 ms:
 *
 *   - HOISTING THE BIAS AND THE ACTIVATION out of the store. A lane owns
 *     `rowsPerLane` rows of the same columns, so `weights[bias_offset + column]`
 *     is read from global memory `rowsPerLane` times over and the activation
 *     uniform re-tested with it - 32 reads and 32 branches a lane where four
 *     registers and one branch would do. **16.53 and 16.51**, and 16.61/16.54
 *     with the bias vector's writes unrolled to rule out a dynamic index. The
 *     four extra vectors live across the whole k loop and cost more than the
 *     reads, which are cached and few.
 *   - UNROLLING THE STAGING LOOPS' vec4 COMPONENT WRITES. `staged[j]` with `j`
 *     a loop variable is a dynamically indexed vector, which WGSL is entitled
 *     to put in spillable local memory - the trap src/af3/
 *     outer-product-mean-webgpu.js documents for accumulator arrays, and these
 *     sit in the hot k loop. **16.69 and 16.39.** A four-iteration loop over a
 *     constant bound is not that case: the compiler already unrolls it and
 *     keeps the vector in registers, and writing it out by hand only loses the
 *     hoisting it was doing.
 *
 * 🔴 A LANE'S COLUMNS STAY STRIDED BY `lanesX`, WHICH LOOKS WRONG AND IS NOT.
 * Contiguous columns per lane would make the staged weights one flat vec4 read,
 * but the OUTPUT store then goes out at stride `columnsPerLane` across the
 * lanes of a row, where strided ownership makes each store instruction a
 * consecutive run. The weight tile is laid out per thread instead, which costs
 * the staging loop nothing and keeps both ends coalesced.
 *
 * @param {{lanesX: number, lanesY: number, rowsPerLane: number, columnsPerLane: number}} tile
 * @param {boolean} residual whether the store accumulates into the output
 */
export function createLinearShader(
  tile = LINEAR_TILE, residual = false, precision = "f32", weightPrecision = "f32",
) {
  if (!["f32", "f16"].includes(weightPrecision)) {
    throw new RangeError(`unknown linear weight precision ${weightPrecision}`);
  }
  // 🔴 THE WEIGHT BUFFER IS A FOURTH FORMAT, AND FOR AF2 IT IS A BANDWIDTH
  // QUESTION RATHER THAN A REGISTER ONE. This kernel reads the whole weight set
  // once per ROW TILE - 944 of them for a 512-row alignment - which is about
  // 2 GB of traffic per transition per block against a 2 MiB working set that
  // lives in cache. AF3's trunk kernels do not care, because their weights are
  // resident and their reads are instruction-bound (measured 377 ms against
  // 378); AF2 uploads its weights per block and re-reads them far more times.
  const weight16 = weightPrecision === "f16";
  const wf = (e) => (weight16 ? `f32(${e})` : e);
  if (!LINEAR_PRECISIONS.has(precision)) throw new RangeError(`unknown linear precision ${precision}`);
  // The element the staged operands and the k loop work in. `mixed` multiplies
  // in f16 and folds into an f32 accumulator once a k tile, so a long inner
  // dimension does not accumulate in ten mantissa bits.
  const element = precision === "f32" ? "f32" : "f16";
  const vector = `vec4<${element}>`;
  const folds = precision === "mixed";
  const cast = element === "f32" ? (e) => e : (e) => `f16(${e})`;
  const { lanesX, lanesY, rowsPerLane, columnsPerLane } = tile;
  for (const [name, value] of Object.entries(tile)) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new RangeError(`linear tile ${name} must be a positive integer; got ${value}`);
    }
  }
  if (rowsPerLane % 4 !== 0 || columnsPerLane % 4 !== 0) {
    throw new RangeError("linear tile rowsPerLane and columnsPerLane must be multiples of 4");
  }
  const lanes = lanesX * lanesY;
  const step = lanesY;
  const tileRows = lanesY * rowsPerLane;
  const tileColumns = lanesX * columnsPerLane;
  const rowVectors = tileRows / 4;
  const columnVectors = columnsPerLane / 4;
  // Each lane stages whole vec4s of both operands. A wide, shallow tile can
  // have fewer source vectors than lanes, so the last pass is guarded rather
  // than forbidden - the alternative is a tile shape the sweep cannot reach.
  const sourceTasks = step * rowVectors;
  const sourcePerLane = Math.ceil(sourceTasks / lanes);

  const accumulator = (r, v) => `acc_${r}_${v}`;
  const block = (r, v) => `blk_${r}_${v}`;
  const target = folds ? block : accumulator;
  const declare = [];
  for (let r = 0; r < rowsPerLane; r += 1) {
    for (let v = 0; v < columnVectors; v += 1) {
      declare.push(`  var ${accumulator(r, v)} = ${folds ? "vec4<f32>" : vector}(0.0);`);
      if (folds) declare.push(`  var ${block(r, v)} = ${vector}(0.0);`);
    }
  }

  // The k loop, fully unrolled: the reads are hoisted so each is issued once.
  const inner = [];
  for (let k = 0; k < step; k += 1) {
    for (let g = 0; g < rowsPerLane / 4; g += 1) {
      inner.push(`    let s_${k}_${g} = tile_source[${k * rowVectors}u + local.y * ${rowsPerLane / 4}u + ${g}u];`);
    }
    for (let v = 0; v < columnVectors; v += 1) {
      inner.push(`    let w_${k}_${v} = tile_weight[${k * lanesX * columnVectors}u + local.x * ${columnVectors}u + ${v}u];`);
    }
    for (let r = 0; r < rowsPerLane; r += 1) {
      for (let v = 0; v < columnVectors; v += 1) {
        inner.push(`    ${target(r, v)} += s_${k}_${Math.floor(r / 4)}[${r % 4}u] * w_${k}_${v};`);
      }
    }
  }

  // Folding a k tile's f16 partial into the f32 accumulator. One conversion and
  // one add per accumulator per EIGHT k, so the error stays that of a sum of
  // eight rather than of the whole inner dimension, and the arithmetic stays
  // f16 where all of the instructions are.
  const fold = [];
  if (folds) {
    for (let r = 0; r < rowsPerLane; r += 1) {
      for (let v = 0; v < columnVectors; v += 1) {
        fold.push(`    ${accumulator(r, v)} += vec4<f32>(${block(r, v)});`);
        fold.push(`    ${block(r, v)} = ${vector}(0.0);`);
      }
    }
  }

  // Staging the source: a lane takes four rows at one k, so its four global
  // reads are `inner` apart and CONSECUTIVE LANES take consecutive k - which is
  // what makes each of the four a coalesced run.
  const stageSource = [];
  for (let n = 0; n < sourcePerLane; n += 1) {
    const task = n === 0 ? "linear_lane" : `(linear_lane + ${n * lanes}u)`;
    const guard = (n + 1) * lanes > sourceTasks ? `if (${task} < ${sourceTasks}u) ` : "";
    stageSource.push(`    ${guard}{
      let task = ${task};
      let row_group = task / ${step}u;
      let k_local = task % ${step}u;
      let k = k0 + k_local;
      let row_base = group.y * ${tileRows}u + row_group * 4u;
      var staged = ${vector}(0.0);
      if (k < parameters.inner) {
        for (var j = 0u; j < 4u; j += 1u) {
          let row = row_base + j;
          if (row < parameters.rows) { staged[j] = ${cast('source[row * parameters.inner + k]')}; }
        }
      }
      tile_source[k_local * ${rowVectors}u + row_group] = staged;
    }`);
  }

  const stageWeight = [];
  for (let v = 0; v < columnVectors; v += 1) {
    stageWeight.push(`    {
      var staged = ${vector}(0.0);
      if (weight_k < parameters.inner) {
        for (var j = 0u; j < 4u; j += 1u) {
          let output_column = column_origin + (${v}u * 4u + j) * ${lanesX}u;
          if (output_column < parameters.columns) {
            staged[j] = ${cast(wf('weights[parameters.weight_offset + weight_k * parameters.columns + output_column]'))};
          }
        }
      }
      tile_weight[local.y * ${lanesX * columnVectors}u + local.x * ${columnVectors}u + ${v}u] = staged;
    }`);
  }

  const store = [];
  for (let r = 0; r < rowsPerLane; r += 1) {
    const body = [];
    for (let v = 0; v < columnVectors; v += 1) {
      for (let c = 0; c < 4; c += 1) {
        body.push(`      {
        let output_column = column_origin + ${(v * 4 + c) * lanesX}u;
        if (output_column < parameters.columns) {
          var value = f32(${accumulator(r, v)}[${c}u])
            + ${wf("weights[parameters.bias_offset + output_column]")};
          if (parameters.activation == 1u) { value = max(value, 0.0); }
          output[row_${r} * parameters.columns + output_column] ${residual ? "+=" : "="} value;
        }
      }`);
      }
    }
    store.push(`  if (row_${r} < parameters.rows) {\n${body.join("\n")}\n  }`);
  }

  const rowNames = [];
  for (let r = 0; r < rowsPerLane; r += 1) {
    rowNames.push(`  let row_${r} = group.y * ${tileRows}u + local.y * ${rowsPerLane}u + ${r}u;`);
  }

  return `${element === "f16" || weight16 ? "enable f16;\n" : ""}
struct MatmulParameters {
  rows: u32,
  inner: u32,
  columns: u32,
  weight_offset: u32,
  bias_offset: u32,
  activation: u32,
  padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<uniform> parameters: MatmulParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Transposed: four ROWS to a vector, so one read serves four accumulators.
var<workgroup> tile_source: array<${vector}, ${step * rowVectors}>;
// Laid out per thread: a lane's own strided columns, contiguous where it reads.
var<workgroup> tile_weight: array<${vector}, ${step * lanesX * columnVectors}>;

@compute @workgroup_size(${lanesX}, ${lanesY}, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let linear_lane = local.y * ${lanesX}u + local.x;
  let column_origin = group.x * ${tileColumns}u + local.x;
${rowNames.join("\n")}
${declare.join("\n")}

  for (var k0 = 0u; k0 < parameters.inner; k0 += ${step}u) {
    let weight_k = k0 + local.y;
${stageSource.join("\n")}
${stageWeight.join("\n")}
    workgroupBarrier();
${inner.join("\n")}
${fold.join("\n")}
    workgroupBarrier();
  }

${store.join("\n")}
}`;
}

export function createTransitionShaders(
  input, offsets, tile = LINEAR_TILE, precision = "f32", weightPrecision = "f32",
) {
  void input;
  void offsets;
  // The normalize pass binds the SAME buffer as the two linear passes, so it
  // narrows with them or reads half the values at twice the stride.
  const weight16 = weightPrecision === "f16";
  const wf = (e) => (weight16 ? `f32(${e})` : e);
  const normalize = `${weight16 ? "enable f16;\n" : ""}
struct NormalizeParameters {
  rows: u32,
  channels: u32,
  scale_offset: u32,
  offset_offset: u32,
  epsilon: f32,
  padding_0: u32,
  padding_1: u32,
  padding_2: u32,
};
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<uniform> parameters: NormalizeParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> row_mean: array<f32, 1>;

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  // ONE WORKGROUP PER ROW, folded across two dimensions: a transition runs over
  // msaSequences * length rows, which is 147,828 at 508 rows of a 291-residue
  // alignment - well past the 65535 a dispatch may be wide. The row guard below
  // is what makes the fold safe: the grid is rounded up, and the extra
  // workgroups return before touching anything.
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= parameters.rows) { return; }
  let base = row * parameters.channels;
  var sum = 0.0;
  for (var c = local.x; c < parameters.channels; c += 64u) {
    sum += source[base + c];
  }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { row_mean[0] = partial[0] / f32(parameters.channels); }
  workgroupBarrier();

  var sum_squared = 0.0;
  for (var c = local.x; c < parameters.channels; c += 64u) {
    let centered = source[base + c] - row_mean[0];
    sum_squared += centered * centered;
  }
  partial[local.x] = sum_squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_std = inverseSqrt(partial[0] / f32(parameters.channels) + parameters.epsilon);
  for (var c = local.x; c < parameters.channels; c += 64u) {
    output[base + c] = (source[base + c] - row_mean[0]) * inverse_std
      * ${wf("weights[parameters.scale_offset + c]")}
      + ${wf("weights[parameters.offset_offset + c]")};
  }
}`;
  const [linear, linearResidual] = [false, true]
    .map((residual) => createLinearShader(tile, residual, precision, weightPrecision));
  return [normalize, linear, linearResidual];
}

export function createTransitionNormalizeParameters(input, offsets) {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  view.setUint32(0, input.rows, true);
  view.setUint32(4, input.channels, true);
  view.setUint32(8, offsets[0], true);
  view.setUint32(12, offsets[1], true);
  view.setFloat32(16, input.epsilon ?? 1e-5, true);
  return new Uint8Array(buffer);
}

export class TransitionGpu {
  device;
  allocator;
  pipelines;

  constructor(device, options = {}) {
    this.device = device;
    // So the differential checker can drive the same choice a block makes, and
    // force either arm of it.
    this.options = options;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(input) {
    validate(input);
    // The same choice the block encoders make, so this path - and the
    // differential checker that drives it - exercises whichever tile a fold of
    // this shape would actually run.
    const { tile, precision, weightPrecision } = chooseLinearKernel({
      rows: input.rows, columns: Math.max(input.channels, input.hiddenChannels),
      device: this.device, requested: this.options?.precision ?? "auto",
    });
    const packed = packTransitionWeights(input, weightPrecision);
    const tileColumns = linearTileColumns(tile);
    const code = createTransitionShaders(
      input, packed.offsets, tile, precision, weightPrecision);
    const key = `transition:${input.rows}:${input.channels}:${input.hiddenChannels}`
      + `:${input.epsilon ?? 1e-5}:${tileColumns}:${precision}:${weightPrecision}`;
    const pipelines = [];
    for (let index = 0; index < code.length; index += 1) {
      pipelines.push(await this.pipelines.get(`${key}:${index}`, code[index]));
    }
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const storage = GPUBufferUsage.STORAGE;
    try {
      const source = keep(this.allocator.upload("transition.source", input.activations, storage));
      const weights = keep(this.allocator.upload("transition.weights", packed.data, storage));
      const layerNormParameters = keep(this.allocator.upload(
        "transition.normalize.parameters", createTransitionNormalizeParameters(input, packed.offsets), GPUBufferUsage.UNIFORM,
      ));
      const firstParameters = keep(this.allocator.upload("transition.first.parameters", new Uint32Array([
        input.rows, input.channels, input.hiddenChannels, packed.offsets[2], packed.offsets[3], 1, 0, 0,
      ]), GPUBufferUsage.UNIFORM));
      const secondParameters = keep(this.allocator.upload("transition.second.parameters", new Uint32Array([
        input.rows, input.hiddenChannels, input.channels, packed.offsets[4], packed.offsets[5], 0, 0, 0,
      ]), GPUBufferUsage.UNIFORM));
      const normalized = keep(this.allocator.allocate("transition.normalized", input.rows * input.channels * 4, storage));
      const hidden = keep(this.allocator.allocate("transition.hidden", input.rows * input.hiddenChannels * 4, storage));
      const output = keep(this.allocator.allocate(
        "transition.output", input.rows * input.channels * 4, storage | GPUBufferUsage.COPY_SRC,
      ));
      const readback = keep(this.allocator.allocate(
        "transition.readback", input.rows * input.channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      ));
      const encoder = this.device.createCommandEncoder({ label: "transition" });
      this.device.pushErrorScope("validation");
      const pass = (pipeline, buffers, x, y = 1) => {
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }));
        compute.dispatchWorkgroups(x, y);
        compute.end();
      };
      pass(pipelines[0], [source.buffer, weights.buffer, layerNormParameters.buffer, normalized.buffer],
        Math.min(input.rows, GRID_WIDTH), ceilDivide(input.rows, GRID_WIDTH));
      pass(pipelines[1], [normalized.buffer, weights.buffer, firstParameters.buffer, hidden.buffer],
        ceilDivide(input.hiddenChannels, tileColumns), ceilDivide(input.rows, TRANSITION_TILE_ROWS));
      pass(pipelines[1], [hidden.buffer, weights.buffer, secondParameters.buffer, output.buffer],
        ceilDivide(input.channels, tileColumns), ceilDivide(input.rows, TRANSITION_TILE_ROWS));
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, input.rows * input.channels * 4);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return {
        output: result,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
    }
  }
}
