// What does an ESM-C block actually cost, and what is the tower's overhead?
//
//     node tools/gpu-chrome.mjs tools/gpu/bench-esmc-tower.js --tokens=61,150,300
//
// 🔴 THE WEIGHTS ARE UPLOADED ONCE AND THE BLOCK IS RUN MANY TIMES. A block is
// 64 MiB of float32; timing a call that uploads them measures the bus, and the
// tower's own 3.7 s at 61 tokens is mostly 36 submits and a shard fetch each.
// Those are real costs and they are reported separately - what they are not is
// the arithmetic.
//
// 🔴 AND THE ARMS ARE MEDIANS, INTERLEAVED. This M2 drifts by up to 3.2x
// between runs, so a single timing of each shape is not a comparison; see
// CLAUDE.md's note on measuring.
//
// 🔴 AND THE OUTPUT IS CHECKED, BECAUSE A DISPATCH THAT LEAVES ROWS
// UNPROCESSED READS AS A SPEEDUP. Every round's result is compared to the
// first: a shape whose grid does not cover its rows is faster and wrong, and
// this repository has been fooled by exactly that.
import { EsmcBlockGpu } from "../../src/esmc/block-webgpu.js";

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

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1];
};

export async function main(device, args = []) {
  const tokenCounts = option(args, "tokens", "61,150,300").split(",").map(Number);
  const rounds = Number(option(args, "rounds", "9"));
  const layers = Number(option(args, "layers", "36"));
  const model = 1152, heads = 18, ffn = 3072;

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
  const rows = [];
  for (const tokens of tokenCounts) {
    const input = deterministic(tokens * model, 11 + tokens);
    const shape = { rows: tokens, model, heads, ffn, residualScale: 1 };

    let reference = null;
    const timings = [];
    for (let round = 0; round < rounds; round += 1) {
      const result = await block.run(input, shape, weights);
      timings.push(result.elapsedMilliseconds);
      if (reference === null) {
        reference = result.output;
        let finite = true;
        for (let i = 0; i < reference.length; i += 1) {
          if (!Number.isFinite(reference[i])) { finite = false; break; }
        }
        if (!finite) throw new Error(`${tokens} tokens produced a non-finite output`);
      } else {
        let worst = 0;
        for (let i = 0; i < reference.length; i += 1) {
          worst = Math.max(worst, Math.abs(result.output[i] - reference[i]));
        }
        if (worst !== 0) {
          throw new Error(`round ${round} at ${tokens} tokens differs by ${worst}`
            + " - a dispatch that does not cover its rows reads as a speedup");
        }
      }
    }

    // Matmuls only: qkv, out, fc1, fc2, plus the two attention products.
    const perToken = 2 * (model * 3 * model + model * model
      + model * 2 * ffn + ffn * model);
    const attention = 2 * 2 * tokens * tokens * model;
    const flops = perToken * tokens + attention;
    const blockMs = median(timings);
    // 🔴 THE TRAFFIC IS WHAT THIS KERNEL IS ACTUALLY SPENDING. One workgroup a
    // ROW means every row re-reads the whole weight matrix from global memory,
    // so the weight traffic is rows x 63.7 MB rather than 63.7 MB - and at 300
    // tokens that is 19.1 GB in 218 ms, which is 87.6 GB/s against this M2's
    // ~100. The kernel is not slow at arithmetic; it is at the bandwidth limit
    // doing 300 times the reads it needs. Row tiling is the fix, and
    // transition-webgpu.js already records what it is worth there: tiling by
    // four took that kernel from 241 ms to 85.
    const weightBytes = (model * 3 * model + model * model
      + 2 * ffn * model + model * ffn) * 4;
    const trafficGb = (weightBytes * tokens) / 1e9;
    rows.push({
      tokens,
      blockMilliseconds: Number(blockMs.toFixed(3)),
      blockGflops: Number((flops / (blockMs / 1000) / 1e9).toFixed(1)),
      towerSeconds: Number((blockMs * layers / 1000).toFixed(2)),
      towerGflop: Number((flops * layers / 1e9).toFixed(1)),
      weightTrafficGb: Number(trafficGb.toFixed(2)),
      effectiveGbPerSecond: Number((trafficGb / (blockMs / 1000)).toFixed(1)),
      spread: Number((Math.max(...timings) / Math.min(...timings)).toFixed(2)),
    });
  }

  return {
    note: "one block, weights resident, median of " + rounds
      + " rounds; tower figures are that block x " + layers,
    layers,
    rows,
  };
}
