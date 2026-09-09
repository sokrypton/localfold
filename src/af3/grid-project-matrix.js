/**
 * AF3's grid attention q/k/v/gate projection on the SUBGROUP MATRIX UNITS.
 *
 * 🔴 THIS IS THE LARGEST KERNEL LEFT IN TWO TRUNKS. At 256 tokens `grid.project`
 * is 58.4 ms of AF3's 463 (13%) and at OpenDDE's widths it is 386 of 1876
 * (21%) - the single biggest pass in that model, ahead of `grid.attend`. The
 * vector kernel reads one interleaved vec4 of weights per output channel and
 * reuses it over a tile of eight pair rows; measured against this device's
 * ceiling it runs at about 16 TFLOP/s where a staged GEMM now reaches 39.
 *
 * 🔴 AND THE WEIGHTS ARE ALREADY IN THE LAYOUT THE MATRIX PATH WANTS, which is
 * why this is a shader and not a migration. `packGridAttentionWeights` lays the
 * four projections out as `(channels, out, 4)` - q, k, v and the gate of one
 * output channel in four ADJACENT columns - because the vector kernel wanted
 * one vec4 read a cell. That is exactly `outputGroup`: a group of four columns
 * is one channel, and each of the four is its own output binding. AF3's
 * triangle projection had to have its weights re-interleaved to reach this;
 * grid attention arrives already there.
 *
 * 🔴 THE SOURCE ROW IS NOT THE OUTPUT ROW IN THE TRANSPOSED DIRECTION. The
 * second grid attention reads `normalized[(row % n) * n + row / n]` and writes
 * at `row` - a pair transpose, not a matrix one - which is `sourceRowIndex`
 * rather than `sourceTransposed`. Getting that wrong returns a plausible tensor
 * of the right shape, so check-grid-project-matrix.js runs both directions.
 *
 * 🔴 AND IT REFUSES A PACKED q/k/v/gate, for the reason the triangle projection
 * refuses a packed a/b: the vector kernel gives one lane a PAIR of adjacent
 * channels so it can own the whole word they share, and this epilogue owns one
 * channel of one row. The packed path keeps the vector kernel.
 */
import {
  createStagedMatrixShader, directWeightsAllowed, stagedMatrixStorage,
} from "../runtime/matrix-linear.js";
import { deviceMatrixConfig, deviceTuning } from "../runtime/device-profile.js";

/**
 * The block this kernel takes when nothing overrides it. The same one the other
 * four staged GEMMs share - see `stagedMatrixBlock` in
 * src/runtime/device-profile.js - because a geometry nobody swept in situ is a
 * guess, and this one is swept there.
 */
export const GRID_PROJECT_MATRIX_GEOMETRY = {
  blockRows: 64, blockColumns: 128, blockInner: 16,
  subgroupRows: 1, subgroupColumns: 8,
};

/**
 * @param {{n: number, channels: number, width: number, transpose: boolean}} shape
 *   `width` is heads * dimension, the width of ONE of the four outputs.
 * @param {{normalized?: "f32"|"f16", qkvg?: "f32"|"f16", weight?: "f32"|"f16"}} storage
 * @param {object} [matrix] the device's tile and result type, plus any geometry
 *   override.
 */
export function createGridProjectMatrixShader(shape, storage = {}, matrix = {}) {
  const { n, channels, width, transpose } = shape;
  for (const [name, value] of Object.entries({ n, channels, width })) {
    if (!(value > 0)) throw new RangeError(`grid project matrix wants a positive ${name}`);
  }
  const qkvg = storage.qkvg ?? "f32";
  if (qkvg !== "f32") {
    throw new RangeError("the matrix grid projection writes f32 q/k/v/gate; "
      + "a packed pair is two channels to a word and needs the vector kernel");
  }
  const geometry = { ...GRID_PROJECT_MATRIX_GEOMETRY, ...matrix };
  const weightPrecision = storage.weight ?? "f32";
  // 🔴 THE REQUEST IS ASKED, NOT SPREAD. `directWeights` needs the weight
  // BUFFER to hold the matrix element and the contracted extent to divide the
  // K panel; spreading the geometry's request straight through is how a
  // caller with f32 weights gets a compile error at best and a wrong panel at
  // worst. See directWeightsAllowed.
  const direct = directWeightsAllowed(geometry, { inner: channels, weightPrecision });
  return createStagedMatrixShader({
    ...geometry,
    directWeights: direct,
    // The interleaved block is [k][4w + role], so a vec4 read of the WEIGHTS
    // needs the column count divisible by four, which 4 * width always is.
    // The source is read one channel at a time along k, so it needs `channels`.
    vectorStaging: { source: channels % 4 === 0, weights: !direct },
    sourcePrecision: storage.normalized ?? "f32",
    weightPrecision,
    outputPrecision: qkvg,
    // 🔴 NO BIAS. The four grid projections have none - the gate's bias, which
    // AF2's equivalent kernel carries in the w lane, does not exist here - and
    // a kernel that added one would read the NEXT tensor in the packed block.
    bias: false,
    ...(transpose ? { sourceRowIndex: `($row % ${n}u) * ${n}u + $row / ${n}u` } : {}),
    outputGroup: { size: 4, stores: [0, 1, 2, 3] },
    outputIndex: `row * ${width}u + channel`,
  });
}

/**
 * ...and grid attention's OUTPUT projection, which is the same operation two
 * kernels later: `(gathered * logistic(gate)) @ outputProjection`, added into
 * the pair.
 *
 * 🔴 IT IS 9.5% OF AN OpenDDE TRUNK. 128.5 ms of 1355 at 256 tokens, and 22.4
 * of AF3's 432, running at about 16 and 10 TFLOP/s where a staged GEMM reaches
 * 39. The gate is the only thing that makes it not a plain projection, and it
 * is applied AT THE STAGING - `sourceModulate` - which costs one extra read in
 * a loop that already runs and saves an elementwise pass over `rows x width`.
 *
 * 🔴 AND THE TRANSPOSED DIRECTION WRITES BACK UNTRANSPOSED, so the residual
 * lands on the pair the right way round. Same `(row % n) * n + row / n` the
 * projection reads with, applied to the DESTINATION instead of the source.
 */
export function createGridProjectOutMatrixShader(shape, storage = {}, matrix = {}, residual = true) {
  const { n, channels, width, transpose } = shape;
  for (const [name, value] of Object.entries({ n, channels, width })) {
    if (!(value > 0)) throw new RangeError(`grid project-out matrix wants a positive ${name}`);
  }
  const geometry = { ...GRID_PROJECT_MATRIX_GEOMETRY, ...matrix };
  const weightPrecision = storage.weight ?? "f32";
  const direct = directWeightsAllowed(geometry, { inner: width, weightPrecision });
  return createStagedMatrixShader({
    ...geometry,
    directWeights: direct,
    // `gathered` is row-major [row][width] and the gate is the same shape, so a
    // vec4 read needs the width divisible by four; every bundle's is.
    vectorStaging: { source: width % 4 === 0, weights: !direct && channels % 4 === 0 },
    sourcePrecision: storage.gathered ?? "f32",
    sourceModulate: true,
    sourceModulatePrecision: storage.gate ?? "f32",
    weightPrecision,
    outputPrecision: "f32",
    bias: false,
    residual,
    outputIndex: transpose
      ? `((row % ${n}u) * ${n}u + row / ${n}u) * parameters.columns + column`
      : "row * parameters.columns + column",
  });
}

/** Whether this device can stage the geometry at all. */
export function gridProjectMatrixFits(matrix = {}, limitBytes = 49152) {
  return stagedMatrixStorage({ ...GRID_PROJECT_MATRIX_GEOMETRY, ...matrix }) <= limitBytes;
}

/**
 * The uniform this shader reads, and the dispatch it wants.
 *
 * 🔴 THE TILE TRAVELS WITH THE SHADER, for the reason recorded on
 * triangleProjectMatrixDispatch: a caller dividing by a constant the shader was
 * not generated from has twice left rows unprocessed here and once reported it
 * as a speedup.
 */
export function gridProjectOutMatrixDispatch(shape, matrix = {}) {
  const geometry = { ...GRID_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return {
    x: Math.ceil(shape.channels / geometry.blockColumns),
    y: Math.ceil(shape.rows / geometry.blockRows),
    // In binding order: the optional source gate is LAST, after the output.
    bindings: ["gathered", "weights", "parameters", "output", "gate"],
  };
}

export function gridProjectMatrixDispatch(shape, matrix = {}) {
  const geometry = { ...GRID_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return {
    x: Math.ceil(4 * shape.width / geometry.blockColumns),
    y: Math.ceil(shape.rows / geometry.blockRows),
    // In binding order, which is the order a caller's buffer array must be in.
    bindings: ["normalized", "weights", "parameters", "q", "k", "v", "gate"],
  };
}

/**
 * One uniform buffer for every block and both directions, because the shape and
 * the offsets are the stack's. `encodePairTrack` has no allocator.
 */
export function allocateGridProjectMatrix(allocator, shape, keep = (a) => a) {
  const { rows, channels, width, weightOffset, outWeightOffset, label = "grid-project" } = shape;
  return {
    project: keep(allocator.upload(`${label}.parameters`, new Uint32Array([
      rows, channels, 4 * width, weightOffset, 0, 0, 0, 0,
    ]), GPUBufferUsage.UNIFORM)),
    // ...and the output projection's, which contracts the WIDTH and produces
    // the channels - the other way round from the one above, which is exactly
    // the kind of pair this repository has swapped before.
    out: keep(allocator.upload(`${label}.out-parameters`, new Uint32Array([
      rows, width, channels, outWeightOffset, 0, 0, 0, 0,
    ]), GPUBufferUsage.UNIFORM)),
  };
}

/**
 * This device's answer for this kernel: a geometry, or false.
 *
 * 🔴 IT LIVES HERE AND NOT IN A STACK, because THREE stacks compile this pair
 * track - the pairformer, the MSA stack and the template embedder - and
 * `grid.project`'s 108 passes in an AF3 trunk are 96, 8 and 4 of them. A rule
 * written into one of the three leaves the other two on the vector kernel and
 * reads as "the knob is worth less than it is". `resolveGridAttendMatrix` is
 * the same shape for the same reason.
 *
 * There is no width rule: this costs no memory, and the layout it reads is the
 * one the vector kernel's pack already writes.
 */
export function gridProjectMatrixConfig(device) {
  if (deviceTuning(device).gridProjectMatrix !== true) return false;
  const config = deviceMatrixConfig(device, { element: "f16" });
  if (config === null) return false;
  const tuning = deviceTuning(device);
  return {
    result: tuning.stagedMatrixResult ?? config.resultComponentType,
    matrixElement: config.componentType,
    tile: { M: config.M, N: config.N, K: config.K },
    prefetch: tuning.stagedMatrixPrefetch === true,
    directWeights: tuning.stagedMatrixDirectWeights === true,
  };
}
