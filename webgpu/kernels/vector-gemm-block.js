/**
 * The block a register-tiled VECTOR GEMM takes - the split transition's (webgpu/af3/trunk/transition.js) and the outer
 * product mean's two (webgpu/kernels/outer-product-mean.js), which share one layout: 256 lanes in a 16 x 16 grid, a
 * lane owning rows / 16 adjacent rows and columns / 16 adjacent columns, the source panel padded four words.
 *
 * 🔴 THE A100'S 64 x 64 x 16 IS NOT AN APPLE PART'S. A lane's 4 x 4 tile issues five workgroup reads for four vec4
 * multiply-adds; at 8 x 8 it is two. Measured on an M5 under stock flags (bench-transition.js, 256 channels at 255
 * tokens, the whole split): 69.5-70.9 ms -> 60.6 at 128 x 128 x 8 in f32 and 42.7 at 64 x 128 x 32 in f16, where the
 * A100 measured 128 x 128 slower. So a device names its block, per arithmetic, in `vectorGemmBlock`.
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
export function vectorGemmBlock(device, arithmetic = "f32", fits = () => true) {
  const spec = deviceTuning(device).vectorGemmBlock?.[arithmetic];
  if (!spec) return DEFAULT_VECTOR_GEMM_BLOCK;
  const [rows, columns, inner] = String(spec).split("x").map(Number);
  const block = { rows, columns, inner };
  if (![rows, columns, inner].every((v) => Number.isInteger(v) && v > 0) || rows % 16 || columns % 64 || inner % 4) {
    throw new RangeError(`vectorGemmBlock: no block ${spec}`);
  }
  const limit = device.limits?.maxComputeWorkgroupStorageSize ?? 16384;
  return vectorGemmBlockStorage(block, arithmetic) <= limit && fits(block) ? block : DEFAULT_VECTOR_GEMM_BLOCK;
}

/** A key part naming the block, for a pipeline cache. */
export const vectorGemmBlockKey = ({ rows, columns, inner }) => `${rows}x${columns}x${inner}`;
