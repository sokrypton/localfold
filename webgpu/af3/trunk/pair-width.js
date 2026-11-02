/**
 * Mol-1's width adapters (PairformerWidthAdapter): every pair row through a bias-free linear into a pair of another
 * width - `trunk_pair_in` 448 -> 320 before its pairformer and `trunk_pair_out` 320 -> 448 after it. The output IS the
 * new pair (written, not added), and no LayerNorm precedes either (shared/af3/dialect.js MOL1).
 *
 * A register-tiled f32 GEMM: 256 lanes in a 16 x 16 grid, each owning a 4 x 4 block of a 64 x 64 output tile, the
 * inner dimension staged 16 at a time. f32 throughout, because these two run once a pass over the whole pair and the
 * port's trunk seams are held at the f32 floor.
 */
import { GRID_WIDTH } from "./pair-track.js";

const TILE = 64;
const INNER = 16;

/** The adapter's shader, for `rows` pair rows of `inputs` channels into `outputs`. */
export function createPairWidthShader(rows, inputs, outputs) {
  if (inputs % INNER !== 0 || outputs % TILE !== 0) {
    throw new Error(`pair width adapter: ${inputs} -> ${outputs} is not a multiple of ${INNER} -> ${TILE}`);
  }
  return /* wgsl */ `
const ROWS: u32 = ${rows}u;
const IN: u32 = ${inputs}u;
const OUT: u32 = ${outputs}u;
const GRID_Y: u32 = ${GRID_WIDTH}u;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> xs: array<array<f32, ${INNER}>, ${TILE}>;
var<workgroup> ws: array<array<f32, ${TILE}>, ${INNER}>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let row0 = (group.y + group.z * GRID_Y) * ${TILE}u;
  let col0 = group.x * ${TILE}u;
  let lane = local.y * 16u + local.x;
  var acc: array<array<f32, 4>, 4>;
  for (var k0 = 0u; k0 < IN; k0 += ${INNER}u) {
    // 64 rows x 16 inner and 16 inner x 64 columns: four elements of each a lane
    for (var e = 0u; e < 4u; e += 1u) {
      let at = lane * 4u + e;
      let r = at / ${INNER}u;
      let c = at % ${INNER}u;
      let row = row0 + r;
      xs[r][c] = select(0.0, x[row * IN + k0 + c], row < ROWS);
      let kr = at / ${TILE}u;
      let kc = at % ${TILE}u;
      ws[kr][kc] = w[(k0 + kr) * OUT + col0 + kc];
    }
    workgroupBarrier();
    for (var k = 0u; k < ${INNER}u; k += 1u) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i += 1u) { a[i] = xs[local.y * 4u + i][k]; b[i] = ws[k][local.x * 4u + i]; }
      for (var i = 0u; i < 4u; i += 1u) {
        for (var j = 0u; j < 4u; j += 1u) { acc[i][j] = fma(a[i], b[j], acc[i][j]); }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i += 1u) {
    let row = row0 + local.y * 4u + i;
    if (row >= ROWS) { continue; }
    for (var j = 0u; j < 4u; j += 1u) { y[row * OUT + col0 + local.x * 4u + j] = acc[i][j]; }
  }
}`;
}

/** The workgroups `createPairWidthShader` dispatches: columns by x, row tiles folded over y and z. */
export function pairWidthDispatch(rows, outputs) {
  const tiles = Math.ceil(rows / TILE);
  return [outputs / TILE, Math.min(tiles, GRID_WIDTH), Math.ceil(tiles / GRID_WIDTH)];
}
