/**
 * A dense projection built on WebGPU's subgroup matrix units.
 *
 * 🔴 NOTHING IN src/ USES THIS. It is a bench candidate, kept beside the
 * benches that measure it, because `chromium_experimental_subgroup_matrix` is
 * not standards-compliant WGSL and AGENTS.md asks for standards compliance. If
 * it ever ships it has to be a capability-gated fast path, the way the subgroup
 * attention kernels already are.
 *
 * It computes what createLinearShader computes - `source @ weights + bias`,
 * optionally through a ReLU - and takes the same four bindings in the same
 * order, so a bench can swap one for the other and compare outputs.
 *
 * 🔴 ONE SUBGROUP PER WORKGROUP, BECAUSE THE STORE MUST BE UNIFORM. WGSL's
 * uniformity analysis works at workgroup scope, so an offset derived from
 * WHICH SUBGROUP YOU ARE cannot be proven uniform even when it is:
 *
 *     error: 'subgroupMatrixStore' requires argument 1 to be uniform
 *
 * Every offset here comes from the workgroup id instead. That costs nothing,
 * because the reuse this kernel needs is in registers, not in staging.
 *
 * 🔴 AND THE REUSE IS THE WHOLE POINT. One subgroup owning ONE 8x8 tile walks
 * all of K from storage for one multiply-accumulate per two tile loads, which
 * is memory-bound by construction - upstream measured that shape at 0.60x to
 * 0.72x, a number about the arrangement and not about the hardware. A `blocks`
 * of 4 gives a subgroup a 32x32 region: four left tiles and four right tiles
 * feed SIXTEEN multiply-accumulates, 2.0 per load rather than 0.5.
 *
 * 🔴 AND A RAGGED EDGE MUST NOT BE A SCALAR FALLBACK. The row counts here are
 * products of a sequence count and a residue count - 30208, 22500 - so a kernel
 * needing M and N divisible by 32 cannot serve them, and a scalar path for the
 * partial regions is correct and catastrophic: upstream measured one shape at
 * 0.09x that way, because its inner loop reads the weights with a stride of a
 * whole row.
 *
 * 🔴 AND AN OUT-OF-BOUNDS subgroupMatrixLoad RETURNS AN ENTIRELY ZERO MATRIX,
 * NOT A CLAMPED ONE. Upstream's kernel runs the matrix path everywhere and
 * bounds-checks only in the store, reasoning that WGSL's robustness rules clamp
 * a load past the end of a binding so a partial region computes garbage exactly
 * in the rows that do not exist - which are the ones never written. That is not
 * what this device does. tools/gpu/check-subgroup-matrix.js loads an 8x8 tile
 * from a buffer holding five rows and every row of the result comes back zero,
 * the three that were missing and the five that were not: relRms 1.0 across the
 * whole tile. A scalar read of the same buffer is clamped; a matrix read of it
 * is refused wholesale.
 *
 * So the load itself has to stay in range. The last region on each axis SLIDES
 * BACK to end on the final row and column rather than hanging over the edge,
 * which costs one partly recomputed region per axis and nothing else - the
 * overlap writes the same values twice. That leaves this kernel needing at
 * least one whole region on each axis, which is the documented restriction
 * below rather than a silent wrong answer.
 */

/**
 * @param {object} [options]
 * @param {number} [options.blocks] 8x8 tiles per side, so the region is
 *   `blocks * 8` square. 4 is upstream's `matrix-bounded-f32`.
 * @param {number} [options.columnBlocks] tiles across, when the region is not
 *   square. Defaults to `blocks`, and `blocks=8, columnBlocks=16` is the
 *   shipped 64x128 geometry, which is what a caller needs if it wants the
 *   existing dispatch grid.
 * @param {"f32"|"f16"} [options.element] what the units multiply in. A device
 *   reports its configs through adapter.info.subgroupMatrixConfigs; this M2
 *   offers f32/f32 and f16/f16 at 8x8x8, and the f16 one ACCUMULATES in f16,
 *   which is a different accuracy question from f16 storage.
 */
export function createMatrixLinearShader(options = {}) {
  const blocks = options.blocks ?? 4;
  const columnBlocks = options.columnBlocks ?? blocks;
  const element = options.element ?? "f32";
  if (!Number.isInteger(blocks) || blocks < 1) throw new RangeError("blocks must be a positive integer");
  if (!Number.isInteger(columnBlocks) || columnBlocks < 1) {
    throw new RangeError("columnBlocks must be a positive integer");
  }
  if (element !== "f32" && element !== "f16") throw new RangeError(`unknown element ${element}`);
  const rowsPerGroup = blocks * 8;
  const columnsPerGroup = columnBlocks * 8;
  const stage = rowsPerGroup * columnsPerGroup;
  // The epilogue walks the staged region with 32 lanes, so it needs a whole
  // number of elements each; every geometry here is a multiple of 8x8.
  if (stage % 32 !== 0) throw new RangeError("the staged region must divide by the subgroup size");

  const enables = element === "f16"
    ? "enable f16;\nenable chromium_experimental_subgroup_matrix;"
    : "enable chromium_experimental_subgroup_matrix;";
  // The staged result is always f32: the epilogue adds a bias and may clamp,
  // and an f16 accumulator's narrowness is a question about the MULTIPLY, not
  // about what is written out.
  const result = element === "f16" ? "f16" : "f32";

  const loads = [];
  for (let m = 0; m < blocks; m += 1) {
    loads.push(`      let left_${m} = subgroupMatrixLoad<subgroup_matrix_left<${element}, 8, 8>>(`
      + `&source, row_base + ${m * 8}u * parameters.inner + k0, false, parameters.inner);`);
  }
  for (let n = 0; n < columnBlocks; n += 1) {
    loads.push(`      let right_${n} = subgroupMatrixLoad<subgroup_matrix_right<${element}, 8, 8>>(`
      + `&weights, k0 * parameters.columns + column_base + ${n * 8}u, false, parameters.columns);`);
  }
  const macs = [];
  for (let m = 0; m < blocks; m += 1) {
    for (let n = 0; n < columnBlocks; n += 1) {
      macs.push(`      acc_${m}_${n} = subgroupMatrixMultiplyAccumulate(left_${m}, right_${n}, acc_${m}_${n});`);
    }
  }
  const declarations = [];
  const stores = [];
  for (let m = 0; m < blocks; m += 1) {
    for (let n = 0; n < columnBlocks; n += 1) {
      declarations.push(`  var acc_${m}_${n} = subgroup_matrix_result<${result}, 8, 8>();`);
      stores.push(`  subgroupMatrixStore(&staged, ${m * 8 * columnsPerGroup + n * 8}u,`
        + ` acc_${m}_${n}, false, ${columnsPerGroup}u);`);
    }
  }

  return `${enables}
struct MatmulParameters {
  rows: u32,
  inner: u32,
  columns: u32,
  weight_offset: u32,
  bias_offset: u32,
  activation: u32,
  padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<${element}>;
@group(0) @binding(1) var<storage, read> weights: array<${element}>;
@group(0) @binding(2) var<uniform> parameters: MatmulParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> staged: array<${result}, ${stage}>;

@compute @workgroup_size(32)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  // 🔴 THE LAST REGION SLIDES BACK RATHER THAN HANGING OVER THE EDGE. See the
  // note above: a matrix load that runs past a binding yields zeros for the
  // WHOLE tile, so a region overlapping the end cannot be masked afterwards.
  // The overlap recomputes rows the previous region already wrote, with the
  // same inputs, so the duplicate write is the same value.
  let row_origin = min(group.y * ${rowsPerGroup}u, parameters.rows - ${rowsPerGroup}u);
  let column_origin = min(group.x * ${columnsPerGroup}u, parameters.columns - ${columnsPerGroup}u);
  // 🔴 EVERY OFFSET IS DERIVED FROM THE WORKGROUP ID. See the note above: a
  // subgroup-derived offset does not pass uniformity analysis.
  let row_base = row_origin * parameters.inner;
  let column_base = column_origin;
${declarations.join("\n")}
  // The weights are bound as [weight | bias], so K runs to inner and the
  // bias sits past it. A tail k0 past inner would read the BIAS as if it
  // were a weight row, so the loop stops on a whole tile and the remainder is
  // handled below.
  let whole = parameters.inner - (parameters.inner % 8u);
  for (var k0 = 0u; k0 < whole; k0 += 8u) {
${loads.join("\n")}
${macs.join("\n")}
  }
${stores.join("\n")}
  workgroupBarrier();
  // The epilogue: bias, activation, bounds. It runs per invocation over the
  // staged region, which is the same shape as an ordinary kernel's tail and is
  // why the matrix path can serve a caller that wants one.
  for (var i = lane; i < ${stage}u; i += 32u) {
    let local_row = i / ${columnsPerGroup}u;
    let local_column = i % ${columnsPerGroup}u;
    let row = row_origin + local_row;
    let column = column_origin + local_column;
    if (row < parameters.rows && column < parameters.columns) {
      var value = f32(staged[i]);
      // The ragged K tail, if the inner dimension is not a multiple of eight.
      for (var k = whole; k < parameters.inner; k += 1u) {
        value += f32(source[row * parameters.inner + k]) * f32(weights[k * parameters.columns + column]);
      }
      value += f32(weights[parameters.bias_offset + column]);
      if (parameters.activation == 1u) { value = max(value, 0.0); }
      output[row * parameters.columns + column] = value;
    }
  }
}`;
}
