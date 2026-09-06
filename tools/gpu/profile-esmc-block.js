// Where does an ESM-C block's time actually go?
//
//     node tools/gpu-chrome.mjs tools/gpu/profile-esmc-block.js --tokens=1500
//
// 🔴 EVERY ATTRIBUTION IN THIS PORT SO FAR CAME FROM ARITHMETIC, AND THE
// ARITHMETIC HAS BEEN WRONG THREE TIMES. It said the projections were bandwidth
// bound (the fix was coalescing, not traffic); it said staging attention's keys
// would divide its cost by eight (it was 0.8x, because the re-reads were cache
// hits); it said tiling the SwiGLU was worth 3.4x (it was 1.2x). This measures.
//
// 🔴 THE TOTALS ARE NOT THE SHIPPED TOTALS. Profiling puts every dispatch in its
// own compute pass so that tools/gpu/profile.js can time it; the shipped block
// runs them in one. The SHARES are what this is for.
import { EsmcBlockGpu } from "../../src/esmc/block-webgpu.js";
import { profileDevice } from "./profile.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const deterministic = (count, seed) => {
  const out = new Float32Array(count);
  let state = seed >>> 0;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = ((state >>> 8) / 8388608 - 1) * 0.05;
  }
  return out;
};

export async function main(device, args = []) {
  const tokens = Number(option(args, "tokens", "300"));
  const rounds = Number(option(args, "rounds", "5"));
  const model = 1152, heads = 18, ffn = 3072;
  const profiler = profileDevice(device);
  if (profiler === null) throw new Error("this device has no timestamp-query");

  const weights = {
    "attn_norm/scale": deterministic(model, 1),
    "attn_norm/offset": deterministic(model, 2),
    "qkv/weights": deterministic(3 * model * model, 3),
    "q_norm/scale": deterministic(model, 4),
    "k_norm/scale": deterministic(model, 5),
    "attn_out/weights": deterministic(model * model, 6),
    "ffn_norm/scale": deterministic(model, 7),
    "ffn_norm/offset": deterministic(model, 8),
    "fc1/weights": deterministic(2 * ffn * model, 9),
    "fc2/weights": deterministic(model * ffn, 10),
  };
  const block = new EsmcBlockGpu(device);
  const input = deterministic(tokens * model, 11);
  const shape = { rows: tokens, model, heads, ffn, residualScale: 1 };

  await block.run(input, shape, weights, { profile: true });   // warm the cache
  profiler.reset();
  for (let round = 0; round < rounds; round += 1) {
    await block.run(input, shape, weights, { profile: true });
  }
  const report = await profiler.report();
  profiler.restore();

  const total = report.reduce((sum, row) => sum + row.ms, 0);
  return {
    tokens,
    rounds,
    totalMilliseconds: Number(total.toFixed(2)),
    perBlockMilliseconds: Number((total / rounds).toFixed(2)),
    passes: report
      .map((row) => ({
        label: row.label,
        millisecondsPerBlock: Number((row.ms / rounds).toFixed(3)),
        share: Number((100 * row.ms / total).toFixed(1)),
        passes: row.passes,
      }))
      .sort((a, b) => b.share - a.share),
  };
}
