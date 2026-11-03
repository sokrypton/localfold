// What the CPU fold's multiply is worth, against the reference loop.
//
//     node tools/bench-cpu-backend.mjs [threads]
//
// One trunk-shaped projection and one triangle, small enough for a laptop. Prints
// GFLOP/s, counting a multiply-add as two.
import { linear, triangleMultiplication } from "../cpu/af3/trunk/pairformer.js";
import { CpuBackend } from "../cpu/runtime/backend.js";
import { setLinearBackend } from "../cpu/runtime/linear-backend.js";

const threads = Number(process.argv[2] ?? 1);
const rows = 4096, inner = 256, outer = 512;
const input = Float32Array.from({ length: rows * inner }, () => Math.random() - 0.5);
const weights = Float32Array.from({ length: inner * outer }, () => Math.random() - 0.5);

function time(label, flops, run) {
  run();
  const at = performance.now();
  run();
  const ms = performance.now() - at;
  console.log(`${label.padEnd(26)} ${ms.toFixed(1).padStart(8)} ms  ${(flops / ms / 1e6).toFixed(1).padStart(7)} GFLOP/s`);
}

const flops = 2 * rows * inner * outer;
setLinearBackend(undefined);
time("projection, reference", flops, () => linear(input, rows, inner, outer, weights));
const backend = await CpuBackend.create({ threads, gibibytes: 1 });
setLinearBackend(backend);
time(`projection, ${backend.threads} thread(s)`, flops, () => linear(input, rows, inner, outer, weights));

const n = 64, channels = 64;
const pair = Float32Array.from({ length: n * n * channels }, () => Math.random() - 0.5);
const mask = new Float32Array(n * n).fill(1);
const random = (length) => Float32Array.from({ length }, () => Math.random() - 0.5);
const triangle = {
  leftNormInputScale: random(channels), leftNormInputOffset: random(channels),
  centerNormScale: random(channels), centerNormOffset: random(channels),
  projection: random(channels * channels * 2), gate: random(channels * channels * 2),
  outputProjection: random(channels * channels), gatingLinear: random(channels * channels),
};
const triangleFlops = 2 * n * n * channels * (channels * 6 + n);
setLinearBackend(undefined);
time("triangle, reference", triangleFlops,
     () => triangleMultiplication(pair, mask, n, channels, "outgoing", triangle));
setLinearBackend(backend);
time(`triangle, ${backend.threads} thread(s)`, triangleFlops,
     () => triangleMultiplication(pair, mask, n, channels, "outgoing", triangle));
backend.close();
process.exit(0);
