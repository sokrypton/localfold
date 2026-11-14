/**
 * The block a register-tiled VECTOR GEMM takes - the split transition's (webgpu/af3/trunk/transition.js) and the outer
 * product mean's two (webgpu/kernels/outer-product-mean.js), which share one layout: 256 lanes in a 16 x 16 grid, a
 * lane owning rows / 16 adjacent rows and columns / 16 adjacent columns, the source panel padded four words.
 *
 * 🔴 THE A100'S 64 x 64 x 16 IS NOT AN APPLE PART'S. A lane's 4 x 4 tile issues five workgroup reads for four vec4
 * multiply-adds; at 8 x 8 it is two. Measured on an M5 under stock flags (bench-transition.js, 256 channels at 255
 * tokens, the whole split): 69.5-70.9 ms -> 60.6 at 128 x 128 x 8 in f32 and 42.7 at 64 x 128 x 32 in f16, where the
 * A100 measured 128 x 128 slower. So a device names its block, per arithmetic, in `vectorGemmBlock`.
 *
 * 🔴 AND THE LANE GRID IS THE DEVICE'S TOO: a fourth number names its side (8 or 16; `32x64x8x8` is 64 lanes, a lane's
 * tile 4 x 8). An M2 runs the f16 transition 1.3x faster on 64 lanes and an M5 slower on every such block
 * (docs/M2-TO-M5.md, 3D-3); the startup race answers it (webgpu/af3/trunk/grid-attend-race.js).
 */
import { deviceTuning } from "../runtime/device-profile.js";

export const DEFAULT_VECTOR_GEMM_BLOCK = Object.freeze({ rows: 64, columns: 64, inner: 16 });

/** Workgroup bytes the block stages: the padded source panel and the weight panel. */
export function vectorGemmBlockStorage({ rows, columns, inner }, arithmetic = "f32") {
  return ((rows + 4) * inner + inner * columns) * (arithmetic === "f16" ? 2 : 4);
}

/**
 * The device's block for this arithmetic, or the default where it names none, cannot stage it, or `fits` (the
 * caller's extents) refuses it.
 */
export function vectorGemmBlock(device, arithmetic = "f32", fits = () => true, role = "") {
  // (`role` names a kernel that may have a block of its own - "Contract", the outer product mean's contraction, whose
  // spec is `f16Contract` - and takes the arithmetic's where the device names none for it)
  const named = deviceTuning(device).vectorGemmBlock;
  const spec = named?.[arithmetic + role] ?? named?.[arithmetic];
  if (!spec) return DEFAULT_VECTOR_GEMM_BLOCK;
  // (a fourth number is the lane grid's side, 8 or 16; three numbers are the 16 x 16 grid every block had)
  const [rows, columns, inner, lanes = 16] = String(spec).split("x").map(Number);
  const block = { rows, columns, inner, ...(lanes === 16 ? {} : { lanes }) };
  if (![rows, columns, inner].every((v) => Number.isInteger(v) && v > 0) || (lanes !== 8 && lanes !== 16)
      || rows % 16 || rows % (4 * lanes) || columns % (4 * lanes) || inner % 4
      || ((rows / 4) * inner) % (lanes * lanes) || ((columns / 4) * inner) % (lanes * lanes)) {
    throw new RangeError(`vectorGemmBlock: no block ${spec}`);
  }
  const limit = device.limits?.maxComputeWorkgroupStorageSize ?? 16384;
  return vectorGemmBlockStorage(block, arithmetic) <= limit && fits(block) ? block : DEFAULT_VECTOR_GEMM_BLOCK;
}

/** A key part naming the block, for a pipeline cache. */
export const vectorGemmBlockKey = ({ rows, columns, inner, lanes }) =>
  `${rows}x${columns}x${inner}${lanes === undefined || lanes === 16 ? "" : `x${lanes}`}`;
