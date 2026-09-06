/**
 * ESM-C's per-residue single into ESMFold2's pair term, on the device.
 *
 *     joined[i, j] = [ a_i * a_j | a_i - a_j ]        (rows, rows, 2c)
 *     lm_z         = pair_norm(mlp_2(gelu(mlp_1(joined))))
 *
 * 🔴 THE OUTER PRODUCT CARRIES BOTH A PRODUCT AND A DIFFERENCE, so the pair
 * representation sees magnitude and direction. Either half alone conforms in
 * shape at half the width, and `pair_mlp_1` being 2c wide is what says so.
 * src/esmc/tower-reference.js's `shimPair` is the specification and the oracle.
 *
 * 🔴 AND IT IS CHUNKED FOR THE SAME REASON THE DIFFUSION CONDITIONING IS. The
 * join is twice the pair representation - 184 MiB at 300 tokens - for
 * arithmetic that is purely per-cell, so it is built eight thousand rows at a
 * time and the chunk is 16 MiB. The chunk's START is a one-element buffer
 * rather than a shader constant, because otherwise every chunk is a different
 * pipeline and a 300-token fold compiles eleven of them.
 */
import { GRID_WIDTH, LANES, createLayerNormShader, createLinearShader, linearGrid }
  from "../esmc/block-webgpu.js";

/** How many pair rows the shim builds at once. */
export const SHIM_CHUNK = 8192;

/** `[a_i * a_j | a_i - a_j]` for one chunk of pair rows. */
export function createOuterJoinShader({ tokens, channels, rows }) {
  return `
@group(0) @binding(0) var<storage, read> single: array<f32>;
@group(0) @binding(1) var<storage, read> start: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${rows * channels}u) { return; }
  let local_row = i / ${channels}u;
  let c = i % ${channels}u;
  let cell = start[0] + local_row;
  let a = single[(cell / ${tokens}u) * ${channels}u + c];
  let b = single[(cell % ${tokens}u) * ${channels}u + c];
  output[local_row * ${channels * 2}u + c] = a * b;
  output[local_row * ${channels * 2}u + ${channels}u + c] = a - b;
}`;
}

/** `x = gelu(x + bias)`, in place, with the exact erf rather than the tanh fit. */
export function createBiasGeluShader({ rows, channels }) {
  return `
@group(0) @binding(0) var<storage, read> bias: array<f32>;
@group(0) @binding(1) var<storage, read_write> activation: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${rows * channels}u) { return; }
  let x = activation[i] + bias[i % ${channels}u];
  // 🔴 THE EXACT GELU, NOT THE tanh APPROXIMATION. torch's default nn.GELU()
  // is the erf form; the tanh one differs by up to 1e-3 and is a different
  // model at the fourth digit, which is where this port's bounds are.
  // Abramowitz-Stegun 7.1.26, good to about 1.5e-7.
  let z = abs(x) * ${(1 / Math.SQRT2).toFixed(12)};
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let series = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
    + t * (-1.453152027 + t * 1.061405429))));
  let erf = sign(x) * (1.0 - series * exp(-z * z));
  activation[i] = 0.5 * x * (1.0 + erf);
}`;
}

/** `x += bias`, broadcast down the rows. */
export function createBiasShader({ rows, channels }) {
  return `
@group(0) @binding(0) var<storage, read> bias: array<f32>;
@group(0) @binding(1) var<storage, read_write> activation: array<f32>;

@compute @workgroup_size(${LANES})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * ${GRID_WIDTH * LANES}u;
  if (i >= ${rows * channels}u) { return; }
  activation[i] += bias[i % ${channels}u];
}`;
}

/** The tensors the shim's pair half reads, under the names the bundle uses. */
export const SHIM_PAIR_TENSORS = ["lm/pair_mlp_1/weights", "lm/pair_mlp_1/bias",
  "lm/pair_mlp_2/weights", "lm/pair_mlp_2/bias",
  "lm/pair_norm/scale", "lm/pair_norm/offset"];

/**
 * Build `lm_z` into `destination`, a (tokens^2, channels) allocation.
 *
 * @param context { device, allocator, cache, submit } - `submit` encodes and
 *   awaits a list of `[label, pipeline, buffers, x, y]` passes.
 */
export async function encodeLanguagePair(context, { tokens, channels, single, weights,
                                                    destination, chunk = SHIM_CHUNK }) {
  const { device, allocator, cache, submit } = context;
  const pairs = tokens * tokens;
  const height = Math.min(chunk, pairs);
  const storage = GPUBufferUsage.STORAGE;
  const key = `esmfold2-shim:${tokens}:${channels}`;
  const heights = [...new Set([height, pairs % height].filter((h) => h > 0))];
  const pipelines = {};
  for (const rows of heights) {
    pipelines[rows] = {
      join: await cache.get(`${key}:join:${rows}`,
        createOuterJoinShader({ tokens, channels, rows })),
      first: await cache.get(`${key}:mlp1:${rows}`,
        createLinearShader({ rows, inner: channels * 2, outer: channels }, false)),
      gelu: await cache.get(`${key}:gelu:${rows}`,
        createBiasGeluShader({ rows, channels })),
      second: await cache.get(`${key}:mlp2:${rows}`,
        createLinearShader({ rows, inner: channels, outer: channels }, false)),
      bias: await cache.get(`${key}:bias:${rows}`, createBiasShader({ rows, channels })),
      norm: await cache.get(`${key}:norm:${rows}`,
        createLayerNormShader({ rows, channels }, true, 1e-5)),
    };
  }

  const singleBuffer = allocator.upload("esmfold2.shim.single", single, storage);
  const uploads = SHIM_PAIR_TENSORS.map((name) =>
    allocator.upload(`esmfold2.shim.${name}`, weights[name], storage));
  const [firstWeights, firstBias, secondWeights, secondBias, normScale, normOffset] = uploads;
  const joined = allocator.allocate("esmfold2.shim.joined", height * channels * 2 * 4, storage);
  const hidden = allocator.allocate("esmfold2.shim.hidden", height * channels * 4, storage);
  const start = allocator.allocate("esmfold2.shim.start", 4,
    storage | GPUBufferUsage.COPY_DST);

  const elementwise = (elements) => {
    const groups = Math.ceil(elements / LANES);
    return [Math.min(GRID_WIDTH, groups), Math.ceil(groups / GRID_WIDTH)];
  };
  try {
    for (let offset = 0; offset < pairs; offset += height) {
      const rows = Math.min(height, pairs - offset);
      const p = pipelines[rows];
      device.queue.writeBuffer(start.buffer, 0, Uint32Array.of(offset));
      const out = { buffer: destination.buffer, byteOffset: offset * channels * 4,
                    byteSize: rows * channels * 4 };
      await submit("esmfold2.shim", [
        ["join", p.join, [singleBuffer, start, joined], ...elementwise(rows * channels)],
        ["mlp1", p.first, [joined, firstWeights, hidden], ...linearGrid(rows, channels)],
        ["gelu", p.gelu, [firstBias, hidden], ...elementwise(rows * channels)],
        // 🔴 THE SECOND MATMUL WRITES BACK INTO `joined`, NOT INTO THE OUTPUT.
        // The LayerNorm reads its source and writes its destination, and WebGPU
        // refuses a buffer bound writable and readable in one dispatch - so the
        // output cannot be both. `joined` is twice as wide as it needs to be
        // and is finished with by here, which is what makes it free.
        ["mlp2", p.second, [hidden, secondWeights, joined], ...linearGrid(rows, channels)],
        ["bias", p.bias, [secondBias, joined], ...elementwise(rows * channels)],
        ["norm", p.norm, [joined, normScale, normOffset, out],
         Math.min(GRID_WIDTH, rows), Math.ceil(rows / GRID_WIDTH)],
      ]);
    }
  } finally {
    for (const allocation of [singleBuffer, joined, hidden, start, ...uploads]) {
      allocation.release();
    }
  }
  return destination;
}
