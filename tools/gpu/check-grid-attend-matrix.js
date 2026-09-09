/**
 * AF3's `grid.attend` on the matrix units, against the CPU reference and
 * against the scalar kernel it replaces.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-grid-attend-matrix.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-grid-attend-matrix.js --tile=2x32
 *     node tools/gpu-chrome.mjs tools/gpu/check-grid-attend-matrix.js --dimension=8
 *
 * 🔴 IT TAKES NO BUNDLE, DELIBERATELY. check-af3-grid-attention.js is the
 * oracle gate and needs `/model-af3-full-f32`; this one synthesises weights of
 * the right SHAPES so it runs anywhere, and so that the head widths the other
 * two models use - 8 for OpenDDE, 16 for the template embedder - can be checked
 * on a box that has only AF3's weights. Neither replaces the other: this says
 * the matrix kernel computes the scalar kernel's answer, and the oracle checker
 * says the scalar kernel computes AF3's.
 *
 * 🔴 AND IT SWEEPS THE TOKEN COUNT ACROSS BOTH TAILS. The kernel emits a
 * partial-key body only when the count does not divide the key tile and a
 * partial-row guard only when it does not divide the row tile, so a single size
 * checks at most one of the four combinations. `--lengths=` defaults to a set
 * that hits all of them for the default 4x32 geometry: 64 (both exact), 68
 * (rows exact, keys ragged), 40 (neither), 24 (shorter than one tile).
 *
 * 🔴 THE MATRIX ARM'S BAR IS f16, NOT THE f32 ARM'S. The units multiply in f16
 * with an f32 accumulator, and the probabilities go back through f16 as well -
 * so it is held where check-af3-grid-attention.js holds its staged-f16 arm, and
 * for the same reason: on this input the softmax turns an absolute logit error
 * into a relative weight error, and uniform noise makes worse-conditioned
 * logits than a pair representation ever does.
 */
import { gridSelfAttention } from "../../src/af3/pairformer-reference.js";
import { Af3GridSelfAttentionGpu } from "../../src/af3/grid-attention-webgpu.js";
import { supportsGridAttendMatrix } from "../../src/af3/grid-attention-matrix.js";

const DIALECT = { swapTransposedBias: false };

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function deterministic(length, seed) {
  let state = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    output[index] = (((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000) * 2 - 1;
  }
  return output;
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const difference = actual[index] - expected[index];
    error += difference * difference;
    scale += expected[index] * expected[index];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const heads = Number(option(args, "heads", "4"));
  const dimension = Number(option(args, "dimension", "32"));
  const channels = Number(option(args, "channels", String(heads * dimension)));
  const tile = option(args, "tile", null);
  const lengths = option(args, "lengths", "64,68,40,24").split(",").map(Number);
  const width = heads * dimension;

  if (!supportsGridAttendMatrix(device, dimension, tile ?? undefined)) {
    return { skipped: "this device has no subgroup matrix units at this shape",
             dimension, tile };
  }

  // The shapes the reference and the packer want; the values are noise, which
  // is what makes this a differential and not an oracle.
  const weights = {
    heads, dimension,
    actNormScale: deterministic(channels, 11).map((v) => 1 + 0.1 * v),
    actNormOffset: deterministic(channels, 12).map((v) => 0.1 * v),
    pairBiasProjection: deterministic(channels * heads, 13),
    qProjection: deterministic(width * channels, 14),
    kProjection: deterministic(width * channels, 15),
    vProjection: deterministic(width * channels, 16),
    gatingQuery: deterministic(width * channels, 17),
    outputProjection: deterministic(width * channels, 18),
  };

  const runner = new Af3GridSelfAttentionGpu(device);
  // 🔴 THE f32 ARM IS WHAT THE MATRIX ARM IS COMPARED TO AS WELL AS THE CPU.
  // Agreeing with the reference to the f16 bar leaves room for a kernel that is
  // wrong in the same direction as f16 rounding; agreeing with the scalar GPU
  // kernel to the same bar, on the same input, does not.
  const bounds = { f32: 1e-5, f16: 3e-2, matrix: 3e-2 };
  const rows = [];
  let failed = 0;

  for (const n of lengths) {
    // Ragged, so a kernel reading the query's mask instead of the key's fails
    // here rather than in a fold.
    const sequence = new Float32Array(n);
    for (let i = 0; i < n; i += 1) sequence[i] = i < Math.ceil(n * 0.75) ? 1 : 0;
    const mask = new Float32Array(n * n);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) mask[i * n + j] = sequence[i] * sequence[j];
    }
    const pair = deterministic(n * n * channels, 991 + n);

    for (const transpose of [false, true]) {
      const expected = gridSelfAttention(pair, mask, n, channels, transpose, weights, DIALECT);
      const shape = { n, channels, transpose };
      const arm = async (name, options) => {
        const run = await runner.run(pair, mask, shape, weights, DIALECT, options);
        return { output: run.output, ms: run.elapsedMilliseconds };
      };
      const scalar = await arm("f32", { stagedPrecision: "f32" });
      const matrix = await arm("matrix",
        { attendMatrix: tile === null ? true : tile, stagedPrecision: "f32" });

      const vsReference = relativeRms(matrix.output, expected);
      const vsScalar = relativeRms(matrix.output, scalar.output);
      const scalarVsReference = relativeRms(scalar.output, expected);
      const ok = vsReference <= bounds.matrix && vsScalar <= bounds.matrix
        && scalarVsReference <= bounds.f32;
      if (!ok) failed += 1;
      rows.push({
        n, transpose, ok,
        matrixVsReference: vsReference.toExponential(2),
        matrixVsScalar: vsScalar.toExponential(2),
        scalarVsReference: scalarVsReference.toExponential(2),
        scalarMs: Number(scalar.ms.toFixed(2)),
        matrixMs: Number(matrix.ms.toFixed(2)),
      });
      console.log(`n=${n}\ttranspose=${transpose}\tmatrix vs ref `
        + `${vsReference.toExponential(2)}\tvs scalar ${vsScalar.toExponential(2)}`
        + `\tscalar vs ref ${scalarVsReference.toExponential(2)}\t${ok ? "ok" : "FAIL"}`);
    }
  }

  if (failed > 0) throw new Error(`${failed} grid attend matrix arm(s) outside tolerance`);
  return { heads, dimension, channels, tile: tile ?? "default", bounds, rows };
}
