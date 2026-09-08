/**
 * A dense projection on the matrix units, with the operands STAGED IN
 * WORKGROUP MEMORY and shared by several subgroups.
 *
 * 🔴 chromium_experimental_subgroup_matrix IS NOT STANDARDS-COMPLIANT WGSL, and
 * AGENTS.md asks for standards compliance. This is here anyway on the same
 * terms as the subgroup attention kernels: a CAPABILITY-GATED FAST PATH that a
 * device without the feature never compiles. chooseLinearKernel is the gate,
 * and a device that fails it gets createLinearShader exactly as before.
 *
 * 🔴 THIS IS THE SHAPE gemm-matrix.js SAYS IS IMPOSSIBLE, AND IT WAS, ONCE.
 * That file is built around "one subgroup per workgroup, because the store must
 * be uniform" - WGSL's uniformity analysis works at workgroup scope, so an
 * offset derived from WHICH SUBGROUP YOU ARE could not be proven uniform even
 * when it was. On the Dawn this A100 runs, a subgroup_id builtin in a
 * subgroupMatrixStore offset COMPILES (tools/gpu/probe-staged-gemm-parts.js
 * grew out of checking that). Sharing staged operands between subgroups is the
 * whole difference between a memory-bound kernel and a compute-bound one.
 *
 * The arithmetic, at 128x128x16 with eight subgroups: one K panel costs 4096
 * halves of staging and feeds 8 subgroups x 8 accumulators = 64
 * multiply-accumulates. gemm-matrix.js's straight-line form reads its operands
 * from GLOBAL memory every k step, reaches about 16 flop per byte, and measures
 * 19 TFLOP/s against units that will do 310 - it is bound on operand traffic
 * and nothing else. Staging raises that by the number of subgroups sharing the
 * panel, and this measures 28.2.
 *
 * 🔴 THE PANEL IS ALWAYS f16 AND THE BUFFERS NEED NOT BE. The matrix units take
 * halves and nothing else on this device, but the conversion is free: it
 * happens in the staging copy, which every version of this kernel already
 * does. So `sourcePrecision`, `weightPrecision` and `outputPrecision` are
 * independent of the units, and this can stand in for createLinearShader
 * wherever that reads an f32 activation and writes an f16 hidden one. Without
 * that the kernel would only serve callers that already hold halves, which in
 * AF2's transition is the second pass and not the first.
 *
 * 🔴 THE EPILOGUE FLUSHES A ROW-BLOCK, NOT A TILE. One 16x16 tile is 16 f32 to
 * a row - 64 bytes, half a transaction - so flushing tiles one at a time writes
 * the output in half-width bursts; it measured 0.70 ms against 0.575 for the
 * row-block form. A region-sized result buffer is not an option either: 128x128
 * in f32 is 64 KiB against a 48 KiB limit. A row-block per subgroup is the
 * middle, and the bias and activation are applied by the lanes as they copy it
 * out.
 *
 * 🔴 EVERY EDGE IS ZERO-PADDED, AND NONE OF THEM SLIDES BACK. gemm-matrix.js
 * slides its last region back to end on the final row, because an
 * out-of-bounds subgroupMatrixLoad returns an entirely zero matrix rather than
 * a clamped one - so a region hanging over the edge cannot be masked
 * afterwards. That works, and it costs two restrictions this kernel cannot
 * afford: the shape must be at least one whole region on each axis, and a
 * RESIDUAL is wrong on a ragged shape because the overlap adds twice
 * (check-staged-matrix.js measured relRms 0.61).
 *
 * Because this kernel already copies its operands through workgroup memory, it
 * can pad instead. A staged element past the end is written as zero, so the
 * matrix load is always in range and always reads a real panel; the epilogue
 * bounds-checks the store. That makes rows, columns and K all the same case,
 * lets a 59-row tensor use a 128-row block, and makes the residual safe
 * everywhere - which is what a transition's third shader needs.
 */

/**
 * Whether a geometry can serve a shape at all.
 *
 * 🔴 IT ALWAYS CAN, NOW, and this stays as a named fact rather than being
 * deleted: the previous version refused a shape smaller than one region and
 * refused a residual on a ragged one, and both restrictions were real until
 * the edges became zero-padded rather than slid back. A caller that used to
 * consult this does not need to any more, and one reading the history should
 * see why.
 */
export function stagedMatrixFits(shape, geometry = {}) {
  void shape;
  void geometry;
  return true;
}

/**
 * How much workgroup storage a geometry needs, so a caller can refuse it
 * before the pipeline does.
 */
export function stagedMatrixStorage(geometry = {}) {
  const {
    blockRows = 128, blockColumns = 128, blockInner = 16,
    subgroupRows = 1, subgroupColumns = 8,
    tile = { M: 16, N: 16, K: 16 }, result = "f32",
  } = geometry;
  const panels = (blockRows * blockInner + blockInner * blockColumns) * 2;
  const scratch = subgroupRows * subgroupColumns * tile.M * (blockColumns / subgroupColumns)
    * (result === "f32" ? 4 : 2);
  return panels + scratch;
}

/**
 * @param {object} [options]
 * @param {number} [options.blockRows] rows of the output one workgroup owns.
 * @param {number} [options.blockColumns] columns of it.
 * @param {number} [options.blockInner] how much of K is staged at a time. 16
 *   beat 32 and 64 on every shape measured - a smaller panel is more barriers
 *   and less workgroup storage, and the storage buys occupancy.
 * @param {number} [options.subgroupRows] subgroups down.
 * @param {number} [options.subgroupColumns] subgroups across.
 * @param {number} [options.subgroupSize] lanes per subgroup; 32 on NVIDIA.
 * @param {{M: number, N: number, K: number}} [options.tile] the device's config.
 * @param {"f16"|"f32"} [options.matrixElement] what the units multiply in -
 *   the componentType of the device's chosen config. f16 everywhere on NVIDIA;
 *   an M2 also offers f32.
 * @param {"f16"|"f32"} [options.result] accumulator width. f32 is free on an
 *   A100 and strictly more accurate - see docs/A100.md.
 * @param {"f16"|"f32"} [options.sourcePrecision] the activation BUFFER's type.
 * @param {"f16"|"f32"} [options.weightPrecision] the weight BUFFER's type.
 * @param {"f16"|"f32"} [options.outputPrecision] what to write.
 * @param {boolean} [options.residual] add the existing output rather than
 *   replacing it, which is what the transition's third shader wants. It
 *   requires a block-aligned shape - see stagedMatrixFits.
 * @param {boolean} [options.vectorStaging] fetch four elements of global memory
 *   at a time. It was worth 1.24 -> 0.70 ms and needs every offset it forms to
 *   divide by four.
 *
 *   🔴 THAT INCLUDES `weight_offset`, WHICH IS A RUNTIME VALUE. A vec4 read at
 *   element `i` returns elements `i & ~3 .. i & ~3 + 3`, so an unaligned base
 *   silently shifts the whole weight panel. The caller knows the offset and
 *   this shader does not, so the caller decides - see createTransitionShaders,
 *   which asks per pass because a transition's two projections sit at
 *   different offsets in one buffer.
 */
export function createStagedMatrixShader(options = {}) {
  const BM = options.blockRows ?? 128;
  const BN = options.blockColumns ?? 128;
  const BK = options.blockInner ?? 16;
  const SGY = options.subgroupRows ?? 1;
  const SGX = options.subgroupColumns ?? 8;
  const lanes = options.subgroupSize ?? 32;
  const tile = options.tile ?? { M: 16, N: 16, K: 16 };
  const { M, N, K } = tile;
  const result = options.result ?? "f32";
  const sourcePrecision = options.sourcePrecision ?? "f16";
  const weightPrecision = options.weightPrecision ?? "f16";
  const outputPrecision = options.outputPrecision ?? "f32";
  const residual = options.residual ?? false;
  const vectorStaging = options.vectorStaging ?? false;

  const subgroups = SGY * SGX;
  const threads = subgroups * lanes;
  const rowsPerSubgroup = BM / SGY;
  const columnsPerSubgroup = BN / SGX;
  if (BM % (SGY * M) !== 0) throw new RangeError("blockRows must divide by subgroupRows * tile.M");
  if (BN % (SGX * N) !== 0) throw new RangeError("blockColumns must divide by subgroupColumns * tile.N");
  if (BK % K !== 0) throw new RangeError("blockInner must divide by tile.K");
  if (vectorStaging && (BK % 4 !== 0 || BN % 4 !== 0)) {
    throw new RangeError("vector staging needs blockInner and blockColumns divisible by four");
  }
  const accRows = rowsPerSubgroup / M;
  const accColumns = columnsPerSubgroup / N;
  const flushWidth = accColumns * N;

  const stagedSource = BM * BK;
  const stagedWeights = BK * BN;
  const scratch = subgroups * M * flushWidth;

  // The half the units multiply in. The buffers are whatever the caller holds.
  // 🔴 THE PANEL'S ELEMENT IS THE DEVICE'S, NOT ALWAYS f16. This A100 offers no
  // f32 component type so f16 was the only choice and got written down as one;
  // an M2 offers BOTH at 8x8x8, and upstream measures the f32 tiles as EXACT -
  // the same error as the scalar kernel to the digit - where its f16 config
  // accumulates in f16. So a device that has f32 units should be able to ask
  // for them, and hardcoding the half made that untestable.
  const half = options.matrixElement ?? "f16";
  if (half !== "f16" && half !== "f32") {
    throw new RangeError(`unknown matrix element ${half}`);
  }
  const sourceElement = vectorStaging ? `vec4<${sourcePrecision}>` : sourcePrecision;
  const weightElement = vectorStaging ? `vec4<${weightPrecision}>` : weightPrecision;

  const declarations = [];
  const stores = [];
  const biasRead = vectorStaging
    ? "weights[(parameters.bias_offset + column) / 4u][(parameters.bias_offset + column) % 4u]"
    : "weights[parameters.bias_offset + column]";

  for (let m = 0; m < accRows; m += 1) {
    const block = [];
    for (let n = 0; n < accColumns; n += 1) {
      declarations.push(`  var acc_${m}_${n} = subgroup_matrix_result<${result}, ${N}, ${M}>();`);
      block.push(`  subgroupMatrixStore(&scratch, sg * ${M * flushWidth}u + ${n * N}u,`
        + ` acc_${m}_${n}, false, ${flushWidth}u);`);
    }
    stores.push(`${block.join("\n")}
  workgroupBarrier();
  for (var i = lane; i < ${M * flushWidth}u; i += ${lanes}u) {
    let row = row_origin + sg_row + ${m * M}u + i / ${flushWidth}u;
    let column = column_origin + sg_column + i % ${flushWidth}u;
    if (row >= parameters.rows || column >= parameters.columns) { continue; }
    var value = f32(scratch[sg * ${M * flushWidth}u + i]);
    value += f32(${biasRead});
    if (parameters.activation == 1u) { value = max(value, 0.0); }
    let at = row * parameters.columns + column;
${residual ? `    value += f32(output[at]);\n` : ""}    output[at] = ${outputPrecision}(value);
  }
  workgroupBarrier();`);
  }

  // The inner k step: each subgroup reads its own rows and columns out of the
  // staged panel. Both offsets are subgroup-derived, which is the point.
  const inner = [];
  for (let m = 0; m < accRows; m += 1) {
    inner.push(`      let left_${m} = subgroupMatrixLoad<subgroup_matrix_left<${half}, ${K}, ${M}>>(`
      + `&staged_source, (sg_row + ${m * M}u) * ${BK}u + kk, false, ${BK}u);`);
  }
  for (let n = 0; n < accColumns; n += 1) {
    inner.push(`      let right_${n} = subgroupMatrixLoad<subgroup_matrix_right<${half}, ${N}, ${K}>>(`
      + `&staged_weights, kk * ${BN}u + sg_column + ${n * N}u, false, ${BN}u);`);
  }
  for (let m = 0; m < accRows; m += 1) {
    for (let n = 0; n < accColumns; n += 1) {
      inner.push(`      acc_${m}_${n} = subgroupMatrixMultiplyAccumulate(left_${m}, right_${n}, acc_${m}_${n});`);
    }
  }

  // Consecutive threads take consecutive columns, which is the coalesced
  // direction in both tensors, and with vector staging each takes four.
  const stagingLoops = vectorStaging
    ? `    for (var i = tid; i < ${stagedSource / 4}u; i += ${threads}u) {
      let k = k0 + (i % ${BK / 4}u) * 4u;
      let r = row_origin + i / ${BK / 4}u;
      let v = select(vec4<${sourcePrecision}>(0),
        source[(r * parameters.inner + k) / 4u], k < parameters.inner && r < parameters.rows);
      let at = i * 4u;
      staged_source[at] = ${half}(v.x);
      staged_source[at + 1u] = ${half}(v.y);
      staged_source[at + 2u] = ${half}(v.z);
      staged_source[at + 3u] = ${half}(v.w);
    }
    for (var i = tid; i < ${stagedWeights / 4}u; i += ${threads}u) {
      let k = k0 + i / ${BN / 4}u;
      let c = column_origin + (i % ${BN / 4}u) * 4u;
      let v = select(vec4<${weightPrecision}>(0),
        weights[(parameters.weight_offset + k * parameters.columns + c) / 4u],
        k < parameters.inner && c < parameters.columns);
      let at = i * 4u;
      staged_weights[at] = ${half}(v.x);
      staged_weights[at + 1u] = ${half}(v.y);
      staged_weights[at + 2u] = ${half}(v.z);
      staged_weights[at + 3u] = ${half}(v.w);
    }`
    : `    for (var i = tid; i < ${stagedSource}u; i += ${threads}u) {
      let k = k0 + i % ${BK}u;
      let r = row_origin + i / ${BK}u;
      staged_source[i] = select(${half}(0),
        ${half}(source[r * parameters.inner + k]), k < parameters.inner && r < parameters.rows);
    }
    for (var i = tid; i < ${stagedWeights}u; i += ${threads}u) {
      let k = k0 + i / ${BN}u;
      let c = column_origin + i % ${BN}u;
      staged_weights[i] = select(${half}(0),
        ${half}(weights[parameters.weight_offset + k * parameters.columns + c]),
        k < parameters.inner && c < parameters.columns);
    }`;

  const usesHalf = [half, result, sourcePrecision, weightPrecision, outputPrecision]
    .includes("f16");
  return `${usesHalf ? "enable f16;\n" : ""}enable chromium_experimental_subgroup_matrix;
enable subgroups;
struct MatmulParameters {
  rows: u32,
  inner: u32,
  columns: u32,
  weight_offset: u32,
  bias_offset: u32,
  activation: u32,
  padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<${sourceElement}>;
@group(0) @binding(1) var<storage, read> weights: array<${weightElement}>;
@group(0) @binding(2) var<uniform> parameters: MatmulParameters;
@group(0) @binding(3) var<storage, read_write> output: array<${outputPrecision}>;

var<workgroup> staged_source: array<${half}, ${stagedSource}>;
var<workgroup> staged_weights: array<${half}, ${stagedWeights}>;
// One ${M}-row block per subgroup, which is all the result staging this needs.
var<workgroup> scratch: array<${result}, ${scratch}>;

@compute @workgroup_size(${threads})
fn main(
  @builtin(local_invocation_index) tid: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let lane = tid % ${lanes}u;
  // No slide-back: the edges are padded instead, so these are plain origins
  // and a partial region is normal rather than forbidden.
  let row_origin = group.y * ${BM}u;
  let column_origin = group.x * ${BN}u;
  let sg_row = (sg / ${SGX}u) * ${rowsPerSubgroup}u;
  let sg_column = (sg % ${SGX}u) * ${columnsPerSubgroup}u;

${declarations.join("\n")}

  // 🔴 THE LAST PANEL IS ZERO-PADDED RATHER THAN HANDLED SEPARATELY. A tail
  // loop that did read-modify-write on the output would DOUBLE-ADD wherever
  // the slide-back makes two regions overlap, and would apply the activation
  // twice. Staging a zero for k past the end costs one compare per staged
  // element and makes the ragged case the same code as the whole one.
  //
  // It also keeps k0 from ever reaching the BIAS: the weights are bound as
  // [weight | bias], and a panel reading past the inner extent would otherwise
  // pick the bias up as if it were a weight row.
  let panels = (parameters.inner + ${BK}u - 1u) / ${BK}u;
  for (var p = 0u; p < panels; p += 1u) {
    let k0 = p * ${BK}u;
    workgroupBarrier();
${stagingLoops}
    workgroupBarrier();
    for (var kk = 0u; kk < ${BK}u; kk += ${K}u) {
${inner.join("\n")}
    }
  }
  workgroupBarrier();
${stores.join("\n")}
}`;
}
