/**
 * AF3's triangle projection on the subgroup matrix units.
 *
 * 🔴 IT IS THE LARGEST KERNEL LEFT IN AN ESMFold2 TRUNK. With the pair
 * transition split onto the units, `tri.project` is 130.6 ms of 494 - 26%, the
 * top of the profile - and `tri.project-out` another 106.5. Both are dense
 * projections over a materialised source, which is the shape docs/A100.md's
 * GEMM rule actually governs; the one that rule ruled OUT is `tri.contract`,
 * whose K is the protein's length.
 *
 * 🔴 AND IT NEEDS NO NEW MEMORY, which the transition's split did. Its source
 * is the layer-normed pair and its two outputs are `a` and `b`, all three
 * already pair-sized scratch the track holds. Nothing is materialised that was
 * not materialised before.
 *
 * 🔴 THE FOUR MATRICES ARE ONE GEMM WITH INTERLEAVED COLUMNS. `projectAB`
 * contracts a, a's gate, b and b's gate over one read of the source and gates
 * them pairwise: `a = mask * ap * logistic(ag)`. A staged matmul's epilogue can
 * only reach the columns inside its own subgroup's flush block, so the four
 * roles of channel `h` are packed as columns `4h..4h+3` - see AB_INTERLEAVED in
 * src/triangle/weights.js, which transposes and interleaves in one pass at pack
 * time. The pack REPLACES the four separate matrices, so a bundle costs no more
 * bytes and only one of the two kernels is ever compiled for a block.
 *
 * 🔴 THE STORE IS CHANNEL-MAJOR AND STAYS THAT WAY. `tri.contract` reads
 * `a[h * PAIRS + row]`, so this writes what it reads; the staged shader's
 * `outputIndex` hook is what expresses it, with `channel` in scope.
 */
import {
  createStagedMatrixShader, stagedMatrixStorage,
} from "../runtime/matrix-linear.js";

/**
 * 🔴 SWEPT PER KERNEL AND NEVER INHERITED - see docs/A100.md on AF2's tile
 * moving when its body changed, and on `grid.attend` not ranking by the metric
 * AF2's did. This is the geometry the transition's two GEMMs wanted at the same
 * widths; `tools/gpu/bench-triangle-project.js --matrix-arms=` re-asks.
 */
export const TRIANGLE_PROJECT_MATRIX_GEOMETRY = {
  blockRows: 128, blockColumns: 128, blockInner: 32,
  subgroupRows: 2, subgroupColumns: 4,
};

/**
 * The channel width at which the matrix projection is worth its precision.
 *
 * 🔴 IT COSTS ACCURACY AND THE UNITS ARE WHY. `tri.project` contracts over the
 * channel count, and the units multiply in f16 - BOTH operands, where the
 * vector kernel's f16 mode narrows only the staged activation and keeps the
 * weights in f32. Measured on `check-af3-block-any.js` at 40 tokens, one
 * block, uniform-noise input, as the pair relRMS against the CPU reference:
 *
 *     vector, staged f32                    1.4e-5
 *     vector, staged f16 (the shipped path) 2.7e-3
 *     matrix                                2.0e-2
 *
 * flat in the token count (2.2e-2 at 16, 1.6e-2 at 64), so it is the operand
 * rounding and not the contraction's length. There is no mitigation: this
 * device offers no f32 matrix configuration, so f16 operands are the kernel.
 *
 * 🔴 AND IT BUYS ALMOST NOTHING AT 128 CHANNELS. `tri.project` over 8 blocks:
 *
 *     channels    vector    matrix   speedup   share of trunk
 *          128     33.96     31.01      1.10   0.7%   AlphaFold 3
 *          256    130.55     85.59      1.53   9%     ESMFold2
 *
 * So the default declines it below 192, the same shape of rule
 * TRANSITION_SPLIT_MIN_CHANNELS carries and for a different reason: that one
 * trades memory, this one trades precision.
 *
 * 🔴 THE FOLDS ARE WHAT JUSTIFY SHIPPING IT WHERE IT DOES APPLY, not the block
 * figure - `check-af3-block-any.js` feeds uniform noise, which docs/AF3.md
 * already records as harsher than a real pair representation. With it on:
 * ESMFold2's CA-CA median moves 3.80480 -> 3.80483 with the contact map's
 * precision still 1.000, and OpenDDE's 3.6705 -> 3.6708 at pLDDT 92.0503 ->
 * 92.0508.
 */
export const TRIANGLE_PROJECT_MATRIX_MIN_CHANNELS = 192;

/** Whether a device can stage this geometry at all. */
export function triangleProjectMatrixFits(geometry, limit) {
  return stagedMatrixStorage(geometry) <= limit;
}

/**
 * @param {{cZ: number, cHidden: number}} shape
 * @param {{normalized?: "f32"|"f16", ab?: "f32"|"f16", weight?: "f32"|"f16"}} storage
 * @param {object} [matrix] the device's tile and result type, plus any geometry
 *   override.
 */
export function createTriangleProjectMatrixShader(shape, storage = {}, matrix = {}) {
  const { cZ, cHidden } = shape;
  if (!(cZ > 0) || !(cHidden > 0)) throw new RangeError("cZ and cHidden are required");
  const abStorage = storage.ab ?? "f32";
  // 🔴 A PACKED a/b IS A DIFFERENT OWNERSHIP RULE, NOT A NARROWER STORE. The
  // vector kernel puts two adjacent CHANNELS of a row in one word, which needs
  // one invocation to own both; this epilogue owns one channel of one row and
  // could not write half a word. Refused rather than made quietly wrong - the
  // packed path keeps the vector kernel.
  if (abStorage !== "f32") {
    throw new RangeError("the matrix triangle projection writes f32 a and b; "
      + "a packed pair is two channels to a word and needs the vector kernel");
  }
  const columns = 4 * cHidden;
  const geometry = { ...TRIANGLE_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return createStagedMatrixShader({
    ...geometry,
    // The interleaved block is [k][4h + role], so a vec4 read needs both the
    // contracted extent and the column count divisible by four. `columns` is
    // 4 * cHidden and always is; cZ is the one to check.
    vectorStaging: cZ % 4 === 0,
    sourcePrecision: storage.normalized ?? "f32",
    weightPrecision: storage.weight ?? "f32",
    outputPrecision: abStorage,
    bias: true,
    rowMask: true,
    outputGroup: { size: 4, stores: [[0, 1], [2, 3]] },
    // Channel-major, which is what tri.contract reads.
    outputIndex: "channel * parameters.rows + row",
  });
}

/**
 * The triangle OUTPUT projection, as two staged GEMMs.
 *
 * 🔴 IT IS NOT ONE GEMM AND CANNOT BE MADE INTO ONE. `projectOutput` computes
 * `logistic(z . Wg) * (x . Wz)` - two contractions over two DIFFERENT sources,
 * the layer-normed pair and the normalised contraction. Interleaving the
 * columns, which is what made the a/b projection one GEMM, cannot help: the
 * two halves would still need two sources staged at once.
 *
 * 🔴 SO THE GATE IS ITS OWN PASS AND THE PROJECTION READS IT. Two staged
 * sources with two accumulator sets fits the workgroup budget and then spends
 * about 128 registers a lane; `outputModulate` costs one global read an output
 * cell instead. And the gate needs no new memory either: at this point in the
 * track `scratch[2]` and `scratch[3]` are both dead - `tri.contract` was the
 * last reader of `b`, and `tri.normalize-hidden` has already consumed the
 * contraction into `scratch[1]` - so the gate lands in one of them.
 *
 * @returns {{gate: string, project: string}} in the order they must run.
 */
export function createTriangleProjectOutMatrixShaders(shape, storage = {}, matrix = {}) {
  const { cZ, cHidden } = shape;
  if (!(cZ > 0) || !(cHidden > 0)) throw new RangeError("cZ and cHidden are required");
  const geometry = { ...TRIANGLE_PROJECT_MATRIX_GEOMETRY, ...matrix };
  const weightPrecision = storage.weight ?? "f32";
  const gateStorage = storage.gate ?? "f32";
  return {
    // z . Wg + bias, into the dead scratch buffer. No logistic here: the
    // projection applies it, so this pass is a plain GEMM.
    gate: createStagedMatrixShader({
      ...geometry,
      vectorStaging: cZ % 4 === 0,
      sourcePrecision: storage.normalized ?? "f32",
      weightPrecision,
      outputPrecision: gateStorage,
      bias: true,
    }),
    // x . Wz + bias, times the gate, added into the pair.
    project: createStagedMatrixShader({
      ...geometry,
      vectorStaging: cHidden % 4 === 0 && cZ % 4 === 0,
      sourcePrecision: storage.hidden ?? "f32",
      weightPrecision,
      outputPrecision: "f32",
      bias: true,
      residual: true,
      outputModulate: true,
      modulatePrecision: gateStorage,
    }),
  };
}

/**
 * The two uniforms the output projection needs, and its dispatch.
 *
 * 🔴 THE GATE CONTRACTS OVER cZ AND THE PROJECTION OVER cHidden, which are the
 * same number for every bundle measured and are not the same field. Reading one
 * for the other is a kernel that contracts the wrong depth and returns a
 * plausible tensor.
 */
export function allocateTriangleProjectOutMatrix(allocator, shape, keep = (a) => a) {
  const { rows, cZ, cHidden, offsets, label = "tri-project-out" } = shape;
  const uniform = (name, inner, weight, bias) => keep(allocator.upload(
    `${label}.${name}`,
    new Uint32Array([rows, inner, cZ, weight, bias, 0, 0, 0]), GPUBufferUsage.UNIFORM));
  return {
    gate: uniform("gate", cZ, offsets.linearGWeight, offsets.linearGBias),
    project: uniform("project", cHidden, offsets.linearZWeight, offsets.linearZBias),
  };
}

/** Both passes take the same grid: cZ columns by `rows` rows. */
export function triangleProjectOutMatrixDispatch(shape, matrix = {}) {
  const geometry = { ...TRIANGLE_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return {
    x: Math.ceil(shape.cZ / geometry.blockColumns),
    y: Math.ceil(shape.rows / geometry.blockRows),
  };
}

/**
 * The one uniform this kernel needs, which the caller allocates.
 *
 * 🔴 encodePairTrack HAS NO ALLOCATOR AND SHOULD NOT GROW ONE - the same call
 * `allocateTransitionSplit` makes. One buffer serves every block and both
 * directions, because the shape and the offsets are the stack's.
 */
export function allocateTriangleProjectMatrix(allocator, shape, keep = (a) => a) {
  const { rows, cZ, cHidden, offsets, label = "tri-project" } = shape;
  const out = allocateTriangleProjectOutMatrix(allocator, shape, keep);
  return {
    parameters: keep(allocator.upload(`${label}.parameters`, new Uint32Array([
      rows, cZ, 4 * cHidden, offsets.linearABWeight, offsets.linearABBias, 0, 0, 0,
    ]), GPUBufferUsage.UNIFORM)),
    // ...the output projection's two, from the same call, because the three
    // kernels are one knob and a caller that allocated only part of it would
    // fail at the dispatch rather than here.
    outGate: out.gate,
    outProject: out.project,
    contract: allocateTriangleContractMatrix(
      allocator, { length: Math.round(Math.sqrt(rows)), label }, keep).parameters,
  };
}

/**
 * The dispatch this shader wants, and the bindings in order.
 *
 * 🔴 THE TILE TRAVELS WITH THE SHADER. This repository has twice had a kernel
 * process a fraction of its rows because a caller divided by a constant the
 * shader was not built from, and once reported it as a 30% speedup.
 */
export function triangleProjectMatrixDispatch(shape, matrix = {}) {
  const geometry = { ...TRIANGLE_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return {
    x: Math.ceil(4 * shape.cHidden / geometry.blockColumns),
    y: Math.ceil(shape.rows / geometry.blockRows),
    // In binding order, which is the order a caller's buffer array must be in.
    bindings: ["normalized", "weights", "parameters", "a", "mask", "b"],
  };
}

/**
 * AF3's triangle multiplication as a BATCHED matrix GEMM.
 *
 * 🔴 THIS IS THE KERNEL docs/A100.md's GEMM RULE ACTUALLY RULED OUT, AND THE
 * RULE WAS RIGHT ABOUT THE WRONG THING. The rule's fourth condition is that the
 * win is almost all in `vectorStaging`, which needs the inner extent divisible
 * by four - and here the inner extent is the PROTEIN'S LENGTH, which at 825 is
 * not. The conclusion drawn from that was "the matrix units do not help the
 * triangle multiplication"; what it actually says is that ONE of the two
 * operands cannot be vector-staged. The other can, and the contraction is
 * 10.4 TFLOP/s against this device's 28 for a staged GEMM.
 *
 * 🔴 IT IS 256 SMALL PRODUCTS, NOT ONE BIG ONE. `out[h][i][j] = sum_k
 * a[h][i][k] * b[h][j][k]` is an n x n by n x n product per CHANNEL, all the
 * same shape and the same stride - which is `batchStride` on `group.z`, the
 * dimension the vector kernel already dispatches over.
 *
 * 🔴 AND THE TWO DIRECTIONS TRANSPOSE DIFFERENT OPERANDS. Outgoing is
 * `a[i][k] . b[j][k]`, so the RIGHT operand is transposed; incoming is
 * `b[k][i] . a[k][j]`, so the LEFT one is, and the right is read plainly. That
 * is `weightsTransposed` and `sourceTransposed`, one each, and getting them the
 * wrong way round returns a finite tensor of the same shape.
 *
 * @param {{length: number, channels: number}} shape
 * @param {"outgoing"|"incoming"} direction
 */
export function createTriangleContractMatrixShader(shape, direction, storage = {}, matrix = {}) {
  const { length, channels } = shape;
  if (!(length > 0) || !(channels > 0)) throw new RangeError("length and channels are required");
  if (direction !== "outgoing" && direction !== "incoming") {
    throw new RangeError(`unknown triangle direction ${direction}`);
  }
  const geometry = { ...TRIANGLE_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return createStagedMatrixShader({
    ...geometry,
    // 🔴 ONE OPERAND EACH, WHICH IS WHAT THE RULE ACTUALLY ALLOWS. The
    // transposed one walks the axis it is not stored on and cannot be
    // vec4-read; the other is stored with k fastest and can, whenever the
    // protein's length divides by four. Turning BOTH off because one of them
    // has to be off is what made this kernel look like a wash.
    vectorStaging: length % 4 !== 0 ? false
      : (direction === "outgoing" ? { source: true } : { weights: true }),
    batchStride: length * length,
    sourceTransposed: direction === "incoming",
    weightsTransposed: direction === "outgoing",
    sourcePrecision: storage.ab ?? "f32",
    weightPrecision: storage.ab ?? "f32",
    outputPrecision: "f32",
    bias: false,
    outputIndex: "batch_base + row * parameters.columns + column",
  });
}

/**
 * Its uniform and its dispatch. The two operands are the SAME buffer pair in
 * both directions and only the roles swap, which the caller does at the
 * binding rather than here.
 */
export function allocateTriangleContractMatrix(allocator, shape, keep = (a) => a) {
  const { length, label = "tri-contract" } = shape;
  return {
    parameters: keep(allocator.upload(`${label}.parameters`,
      // rows, inner, columns are all the protein's length; no weight offset and
      // no bias, because both operands are activations.
      new Uint32Array([length, length, length, 0, 0, 0, 0, 0]), GPUBufferUsage.UNIFORM)),
  };
}

export function triangleContractMatrixDispatch(shape, matrix = {}) {
  const geometry = { ...TRIANGLE_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return {
    x: Math.ceil(shape.length / geometry.blockColumns),
    y: Math.ceil(shape.length / geometry.blockRows),
    z: shape.channels,
  };
}
