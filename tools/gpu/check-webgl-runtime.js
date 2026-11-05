/**
 * The WebGL2 backend's runtime and its first three AF2 primitives, against CPU
 * references - and the expressibility question the headline ceiling gets wrong.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-runtime.js [--falsify]
 *
 * 🔴 THE ARM THAT MATTERS MOST IS THE BIG TENSOR, AND IT CONTRADICTS A NUMBER
 * THIS REPOSITORY HAS WRITTEN DOWN TWICE. docs/A100.md concludes that the pair
 * track "stops fitting a texture axis at 181 tokens", from a GEMM whose operand
 * is a 2D texture of `(K/4) x M` with `M = tokens^2`. Laid out LINEARLY - the
 * layout `webgl/runtime/storage.js` uses and the one the triangle kernel has
 * always used - the bound is the total element count instead, so a 400-token
 * 128-channel pair tensor (20,480,000 floats) is addressable and is checked
 * here. Both numbers are right about different layouts, and the one that got
 * quoted is the one that does not apply to this tree's kernels.
 *
 * 🔴 AND THE BANDED DRAW IS FORCED, because nothing on this machine reaches it.
 * A band is capped by `MAX_VIEWPORT_DIMS`, which here equals `MAX_TEXTURE_SIZE`,
 * so a legal tensor is never tall enough to need a second draw and the whole
 * path would ship untested. The arm sets `maxBandRows` small and requires the
 * result to be **bit-identical** to the unbanded one, which is how the WebGPU
 * side checks its own chunked paths: "forcing the chunked path on an input that
 * does not need it reproduces the full path bit for bit".
 *
 * 🔴 AND THREE ARMS ARE REFUSALS, not comparisons. A fragment shader cannot
 * read the texture its framebuffer is attached to, so `add(x, y, x)` - the
 * WebGPU side's ordinary in-place residual - must THROW rather than produce
 * undefined output, and an input bound to a sampler the kernel does not declare
 * must throw rather than silently read whatever was on unit 0.
 */
import { Execution } from "../../webgl/runtime/execution.js";
import { add, multiply, scale } from "../../webgl/runtime/elementwise.js";
import { layerNorm } from "../../webgl/kernels/layer-norm.js";
import { transposeLeading } from "../../webgl/runtime/transpose.js";
import { linear } from "../../webgl/kernels/linear.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);

function generator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}
const noise = (count, scale_, random) => {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = (random() * 2 - 1) * scale_;
  return out;
};

/** `normalizeRows` from webgpu/kernels/triangle/cpu-reference.js, independently. */
function layerNormReference(x, rows, channels, weight, bias, epsilon) {
  const out = new Float32Array(x.length);
  for (let row = 0; row < rows; row += 1) {
    const at = row * channels;
    let mean = 0;
    for (let c = 0; c < channels; c += 1) mean += x[at + c];
    mean /= channels;
    let spread = 0;
    for (let c = 0; c < channels; c += 1) {
      const centred = x[at + c] - mean;
      spread += centred * centred;
    }
    const inverseStd = 1 / Math.sqrt(spread / channels + epsilon);
    for (let c = 0; c < channels; c += 1) {
      out[at + c] = (x[at + c] - mean) * inverseStd * weight[c] + bias[c];
    }
  }
  return out;
}

/** `out[row][o] = sum_i x[row][i] * weight[o * inChannels + i] + bias[o]`. */
function linearReference(x, weight, bias, rows, inChannels, outChannels, activation) {
  const out = new Float32Array(rows * outChannels);
  for (let row = 0; row < rows; row += 1) {
    for (let o = 0; o < outChannels; o += 1) {
      let sum = 0;
      for (let i = 0; i < inChannels; i += 1) {
        sum += x[row * inChannels + i] * weight[o * inChannels + i];
      }
      const value = sum + (bias === undefined ? 0 : bias[o]);
      out[row * outChannels + o] = activation === "relu" ? Math.max(value, 0)
        : activation === "sigmoid" ? 1 / (1 + Math.exp(-value)) : value;
    }
  }
  return out;
}

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const skew = (values) => (falsify
    ? Float32Array.from(values, (v, i) => (i === (values.length >> 1) ? v * 1.01 + 0.01 : v))
    : values);

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const rows = [];
  const failures = [];
  const arm = (name, got, want, bar) => {
    const relative = relativeRms(got, want, name);
    const ok = relative <= bar;
    rows.push({ arm: name, relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  };
  // A refusal is a two-sided arm: it must throw, and it must throw for the
  // stated reason. `--falsify` cannot move these, so they are reported apart
  // from the numerical arms and are not counted in its verdict.
  const refusals = [];
  const refuses = (name, expect, body) => {
    let message = null;
    try { body(); } catch (error) { message = String(error.message); }
    const ok = message !== null && message.includes(expect);
    refusals.push({ refusal: name, ok, said: message ?? "did not throw" });
    if (!ok) failures.push(`${name}: expected a refusal naming "${expect}", got ${message}`);
  };

  // --- storage: a tensor is a linear index ---------------------------------
  const roundTrip = [];
  for (const elements of [1, 7, 2048 * 4, 400 * 400 * 128]) {
    const random = generator(elements);
    const values = noise(elements, 1, random);
    const tensor = execution.upload(values, `round-trip-${elements}`);
    const back = execution.read(tensor);
    let differing = 0;
    for (let i = 0; i < elements; i += 1) if (back[i] !== values[i]) differing += 1;
    roundTrip.push({ elements, height: tensor.height, megabytes: Number((tensor.bytes / 2 ** 20).toFixed(1)), differing });
    if (differing !== 0) failures.push(`round-trip ${elements}: ${differing} differing`);
    execution.release(tensor);
  }

  // --- the capacity refusal names the chain it would hold ------------------
  refuses("allocate past capacity", "tokens here",
    () => execution.allocate(execution.storage.capacity + 1, "too-big"));

  // --- elementwise: no reordering, so bit-exact ---------------------------
  {
    const random = generator(11);
    const elements = 4096 + 12;
    const a = noise(elements, 1, random);
    const b = noise(elements, 1, random);
    const aT = execution.upload(skew(a), "a");
    const bT = execution.upload(b, "b");
    const want = (f) => Float32Array.from({ length: elements }, (_, i) => f(a[i], b[i]));
    arm("elementwise add", execution.read(add(execution, aT, bT)), want((x, y) => x + y), 0);
    arm("elementwise multiply", execution.read(multiply(execution, aT, bT)), want((x, y) => x * y), 0);
    arm("elementwise scale", execution.read(scale(execution, aT, 0.375)),
      want((x) => Math.fround(x * 0.375)), 0);
    refuses("add into its own input", "cannot read the texture it draws into",
      () => add(execution, aT, bT, aT));
  }

  // --- LayerNorm ----------------------------------------------------------
  for (const [n, channels] of [[16, 128], [37, 32], [8, 256]]) {
    const random = generator(n * 100 + channels);
    const x = noise(n * channels, 1.5, random);
    const weight = noise(channels, 0.3, random);
    const bias = noise(channels, 0.1, random);
    const got = execution.read(layerNorm(execution, {
      x: execution.upload(skew(x), "x"),
      weight: execution.upload(weight, "w"),
      bias: execution.upload(bias, "b"),
      rows: n, channels,
    }));
    arm(`layerNorm ${n}x${channels}`, got,
      layerNormReference(x, n, channels, weight, bias, 1e-5), 1e-6);
  }
  refuses("layerNorm on a width that is not a multiple of four", "multiple of 4",
    () => layerNorm(execution, {
      x: execution.allocate(7 * 7, "x7"), weight: execution.allocate(7, "w7"),
      bias: execution.allocate(7, "b7"), rows: 7, channels: 7,
    }));

  // --- the dense projection ------------------------------------------------
  for (const [n, inChannels, outChannels] of [[16, 32, 64], [13, 128, 32], [5, 64, 128]]) {
    const random = generator(n + inChannels + outChannels);
    const x = noise(n * inChannels, 1, random);
    const weight = noise(inChannels * outChannels, 1 / Math.sqrt(inChannels), random);
    const bias = noise(outChannels, 0.1, random);
    const xT = execution.upload(skew(x), "x");
    const wT = execution.upload(weight, "w");
    const bT = execution.upload(bias, "b");
    for (const activation of ["none", "relu", "sigmoid"]) {
      const got = execution.read(linear(execution, {
        x: xT, weight: wT, bias: bT, rows: n, inChannels, outChannels, activation,
      }));
      arm(`linear ${n}x${inChannels}x${outChannels} ${activation}`, got,
        linearReference(x, weight, bias, n, inChannels, outChannels, activation), 1e-6);
    }
    // ...and without a bias, which is a different compiled kernel.
    const noBias = execution.read(linear(execution, {
      x: xT, weight: wT, rows: n, inChannels, outChannels,
    }));
    arm(`linear ${n}x${inChannels}x${outChannels} no bias`, noBias,
      linearReference(x, weight, undefined, n, inChannels, outChannels, "none"), 1e-6);
  }
  // --- a ragged output width, which AF2's pLDDT head has ------------------
  //
  // 🔴 50 BINS MEANS A TEXEL STRADDLES TWO ROWS, so `linear` compiles a
  // different kernel for it: row r occupies elements [50r, 50r + 50), and a
  // fragment's four consecutive elements can belong to two rows with two
  // different input vectors. The aligned kernel shares one input sweep across
  // its four outputs and cannot. Both layouts are checked at a ragged width
  // because the index arithmetic differs between them.
  for (const [n, inChannels, outChannels] of [[5, 32, 50], [3, 16, 7], [9, 8, 13]]) {
    const random = generator(n * 7919 + outChannels);
    const x = noise(n * inChannels, 1, random);
    const weight = noise(inChannels * outChannels, 1 / Math.sqrt(inChannels), random);
    const bias = noise(outChannels, 0.1, random);
    const xT = execution.upload(skew(x), "x");
    const wT = execution.upload(weight, "w");
    const bT = execution.upload(bias, "b");
    for (const weightLayout of ["in-major", "out-major"]) {
      const got = execution.read(linear(execution, {
        x: xT, weight: wT, bias: bT, rows: n, inChannels, outChannels, weightLayout,
      }));
      const want = new Float32Array(n * outChannels);
      for (let row = 0; row < n; row += 1) {
        for (let o = 0; o < outChannels; o += 1) {
          let sum = bias[o];
          for (let i = 0; i < inChannels; i += 1) {
            sum += x[row * inChannels + i] * (weightLayout === "in-major"
              ? weight[i * outChannels + o] : weight[o * inChannels + i]);
          }
          want[row * outChannels + o] = sum;
        }
      }
      arm(`linear ragged ${n}x${inChannels}x${outChannels} ${weightLayout}`, got, want, 1e-6);
    }
  }

  refuses("a sampler the kernel does not declare", "no sampler named",
    () => execution.dispatch({
      source: `${require_nothing()}`, output: execution.allocate(4, "out"),
      inputs: { absentTex: execution.allocate(4, "in") },
    }));

  // --- the leading-axis transpose, which is what column attention is -------
  for (const [outer, inner, channels] of [[4, 6, 8], [7, 3, 32], [1, 5, 16]]) {
    const random = generator(outer * 61 + inner * 13 + channels);
    const values = noise(outer * inner * channels, 1, random);
    const want = new Float32Array(values.length);
    for (let o = 0; o < outer; o += 1) {
      for (let i = 0; i < inner; i += 1) {
        for (let c = 0; c < channels; c += 1) {
          want[(i * outer + o) * channels + c] = values[(o * inner + i) * channels + c];
        }
      }
    }
    const x = execution.upload(skew(values), "x");
    const got = execution.read(transposeLeading(execution, { x, outer, inner, channels }));
    // A permutation moves no value, so this is bit-exact or the index is wrong.
    arm(`transpose ${outer}x${inner}x${channels}`, got, want, 0);
  }

  // --- the banded draw, forced -------------------------------------------
  {
    const random = generator(99);
    const n = 64;
    const channels = 128;
    const x = noise(n * channels, 1.5, random);
    const weight = noise(channels, 0.3, random);
    const bias = noise(channels, 0.1, random);
    const inputs = {
      x: execution.upload(x, "x"), weight: execution.upload(weight, "w"),
      bias: execution.upload(bias, "b"), rows: n, channels,
    };
    const whole = execution.read(layerNorm(execution, inputs)).slice();
    const was = execution.maxBandRows;
    execution.maxBandRows = 3;
    // ...the perturbation goes to the BANDED side only: this arm compares two
    // runs of the same kernel, so a skewed input reaches both equally and the
    // difference stays exactly 0. Same trap as the layout arm in
    // check-webgl-kernels.js, and the same one-sided answer.
    const banded = execution.read(layerNorm(execution, {
      ...inputs, x: execution.upload(skew(x), "x-banded"),
    })).slice();
    execution.maxBandRows = was;
    arm("layerNorm banded at 3 rows a draw", banded, whole, 0);
  }

  const stats = execution.stats();
  const numerical = rows.length;
  const red = rows.filter((row) => !row.ok).length;
  return {
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    maxViewportDims: [...gl.getParameter(gl.MAX_VIEWPORT_DIMS)],
    capacityFloats: stats.capacityFloats,
    pairTokensAt128Channels: Math.floor(Math.sqrt(stats.capacityFloats / 128)),
    falsify,
    roundTrip,
    arms: rows,
    refusals,
    stats,
    verdict: falsify
      ? (red === numerical
        ? `ok: all ${numerical} numerical arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${numerical} numerical arms noticed a 1% perturbation`)
      : (failures.length === 0 ? `ok: ${numerical} arms, ${refusals.length} refusals`
        : `FAILED: ${failures.join("; ")}`),
  };
}

/** A kernel that declares no sampler at all, for the undeclared-input refusal. */
function require_nothing() {
  return `#version 300 es
precision highp float;
out vec4 result;
void main() { result = vec4(1.0); }`;
}
