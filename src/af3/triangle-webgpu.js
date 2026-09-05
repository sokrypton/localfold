/**
 * AF3 triangle multiplication on the GPU, by way of AF2's kernel.
 *
 * WHY THERE IS NO NEW SHADER HERE. AF3's triangle multiplication is AF2's,
 * operation for operation and layout for layout: the same input LayerNorm, the
 * same gated a/b projections, the same einsum, the same channel-major
 * intermediate that `center_norm` reduces over, the same gated output
 * projection. Two things differ, and neither is arithmetic:
 *
 *   1. AF3 FUSES a and b. Where AF2 has four weights (linearAP, linearAG,
 *      linearBP, linearBG) AF3 has two of double width - `projection` and
 *      `gate` - with a and b INTERLEAVED along the output axis, `c*2` and
 *      `c*2+1`. That split is interleaved because the checkpoint was produced
 *      by transposing and reshaping; the other double-width weight in the
 *      pairformer, transition1, is BLOCKED instead. Splitting this one in
 *      halves runs fine and returns garbage that still looks like a pair
 *      representation.
 *   2. AF3 HAS NO BIASES on any of the four, and its trunk uses the fast
 *      LayerNorm variance. Both are passed through rather than assumed.
 *
 * So this file is a weight adapter. It gets a kernel already differentially
 * tested against OpenFold rather than a second implementation of the same
 * einsum, and tools/gpu/check-af3-triangle.js pins it to
 * src/af3/pairformer-reference.js.
 *
 * 🔴 THE TWO REPOS TRANSPOSE WEIGHTS OPPOSITELY. AF2's shaders index
 * `weight[out * in + c]`; AF3 stores `(in, out)` and its reference reads
 * `weights[c * outChannels + out]`. Every projection below is transposed on the
 * way through. Skipping that runs at these shapes - c_z and c_hidden are both
 * 128, so the matrices are square - and quietly computes a different function.
 */
import { TriangleMultiplicationIncomingGpu, TriangleMultiplicationOutgoingGpu }
  from "../triangle/webgpu.js";

/**
 * Split one of AF3's double-width weights into AF2's two, transposing as it
 * goes.
 *
 * @param {Float32Array} fused (channels, channels * 2), a at `c*2`, b at `c*2+1`
 * @param {number} channels
 * @returns {{a: Float32Array, b: Float32Array}} each (channels, channels) as
 *   `[out * channels + in]`
 */
function splitInterleaved(fused, channels) {
  const expected = channels * channels * 2;
  if (fused.length !== expected) {
    throw new Error(`fused weight has ${fused.length} elements; expected ${expected}`);
  }
  const a = new Float32Array(channels * channels);
  const b = new Float32Array(channels * channels);
  for (let input = 0; input < channels; input += 1) {
    const row = input * channels * 2;
    for (let out = 0; out < channels; out += 1) {
      a[out * channels + input] = fused[row + out * 2];
      b[out * channels + input] = fused[row + out * 2 + 1];
    }
  }
  return { a, b };
}

/** Transpose an (in, out) matrix into the (out, in) the AF2 shaders index. */
function transpose(source, inChannels, outChannels) {
  const expected = inChannels * outChannels;
  if (source.length !== expected) {
    throw new Error(`weight has ${source.length} elements; expected ${expected}`);
  }
  const output = new Float32Array(expected);
  for (let input = 0; input < inChannels; input += 1) {
    for (let out = 0; out < outChannels; out += 1) {
      output[out * inChannels + input] = source[input * outChannels + out];
    }
  }
  return output;
}

/**
 * AF3's triangle weights in the layout AF2's kernel packs.
 *
 * @param {object} weights as `blockWeights` builds them: leftNormInput{Scale,
 *   Offset}, projection, gate, centerNorm{Scale,Offset}, outputProjection,
 *   gatingLinear.
 * @param {number} channels c_z, which equals c_hidden throughout AF3.
 */
export function af3TriangleWeights(weights, channels) {
  const projection = splitInterleaved(weights.projection, channels);
  const gate = splitInterleaved(weights.gate, channels);
  // AF3 has no biases here; the kernel's weight block is fixed-shape, so they
  // are present and zero rather than absent.
  const zeros = () => new Float32Array(channels);
  return {
    layerNormInWeight: weights.leftNormInputScale,
    layerNormInBias: weights.leftNormInputOffset,
    linearAPWeight: projection.a, linearAPBias: zeros(),
    linearAGWeight: gate.a, linearAGBias: zeros(),
    linearBPWeight: projection.b, linearBPBias: zeros(),
    linearBGWeight: gate.b, linearBGBias: zeros(),
    layerNormOutWeight: weights.centerNormScale,
    layerNormOutBias: weights.centerNormOffset,
    linearZWeight: transpose(weights.outputProjection, channels, channels),
    linearZBias: zeros(),
    linearGWeight: transpose(weights.gatingLinear, channels, channels),
    linearGBias: zeros(),
  };
}

/**
 * Run AF3 triangle multiplication.
 *
 * @param {GPUDevice} device
 * @param {Float32Array} pair n*n*channels
 * @param {Float32Array} mask n*n
 * @param {number} n
 * @param {number} channels
 * @param {"outgoing"|"incoming"} direction
 * @param {object} weights AF3's block weights for this direction
 * @param {{precision?: "f32"|"f16"}} options
 */
export async function af3TriangleMultiplication(
  device, pair, mask, n, channels, direction, weights, options = {},
) {
  const Runner = direction === "outgoing"
    ? TriangleMultiplicationOutgoingGpu : TriangleMultiplicationIncomingGpu;
  const input = {
    shape: { length: n, cZ: channels, cHidden: channels },
    epsilon: 1e-5,
    z: pair,
    mask,
    weights: af3TriangleWeights(weights, channels),
  };
  // 🔴 "fast" IS NOT A PERFORMANCE KNOB. It is AF3's trunk LayerNorm formula
  // (use_fast_variance=True). The atom and diffusion stacks want "two-pass".
  return new Runner(device).run(input, {
    precision: options.precision ?? "f32", variance: "fast",
    // ...forceable, so the row tile's fold over group.z can be checked at a
    // size a CPU reference can follow. See PROJECT_GRID_WIDTH in shaders.js.
    ...(options.projectGridWidth === undefined
      ? {} : { projectGridWidth: options.projectGridWidth }),
  });
}
