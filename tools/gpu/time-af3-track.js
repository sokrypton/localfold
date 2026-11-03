/**
 * What the sequential track costs in the AlphaFold 3 lineage: a pairformer block's time against what the stack holds.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/time-af3-track.js --family=intellifold2 --lengths=300 --blocks=4 \
 *       --arms=whole,compact,sequential,g2,g4,g16
 *
 * `whole` is the whole-tensor track, `compact` the channel-chunked one, `sequential` what the page runs, and `gN` that
 * with N channel groups. Each arm runs twice over one random pair and single with real weights and the second run is the
 * one reported, so a pipeline's compile is not in its time.
 */
import { Af3PairformerStackGpu } from "../../webgpu/af3/trunk/pairformer-block.js";
import { HttpTensorStore } from "../../shared/bundles/http-tensor-store.js";
import { bundleBaseUrl, loadManifest } from "../../shared/bundles/manifests/index.js";
import { af3Dialect, trunkWeights } from "../../shared/af3/weights/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function random(length, seed) {
  const output = new Float32Array(length);
  let state = seed;
  for (let i = 0; i < length; i += 1) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    output[i] = (state / 0x7fffffff) * 2 - 1;
  }
  return output;
}

function armOptions(arm) {
  if (arm === "whole") return { forceCompact: false, sequentialTriangle: false };
  if (arm === "compact") return { forceCompact: true, sequentialTriangle: false };
  if (arm === "sequential") return { forceCompact: true, sequentialTriangle: true };
  return { forceCompact: true, sequentialTriangle: true, sequentialGroups: Number(arm.slice(1)) };
}

export async function main(device, args) {
  const family = option(args, "family", "protenix2");
  const lengths = option(args, "lengths", "300").split(",").map(Number);
  const blockCount = Number(option(args, "blocks", "4"));
  const arms = option(args, "arms", "whole,compact,sequential,g2,g4,g16").split(",");
  const store = await HttpTensorStore.fromManifest(bundleBaseUrl(family), await loadManifest(family));
  const dialect = af3Dialect(store);
  const { pairformerBlocks: blocks } = await trunkWeights(store, blockCount, 0, { allowPrefix: true });
  const report = [];
  for (const n of lengths) {
    const pairs = n * n;
    const pair = random(pairs * blocks[0].pairChannels, 12345);
    const single = random(n * blocks[0].singleChannels, 54321);
    const pairMask = new Float32Array(pairs).fill(1);
    const seqMask = new Float32Array(n).fill(1);
    for (const arm of arms) {
      const stack = new Af3PairformerStackGpu(device, { residentWeights: false });
      let last;
      for (let repeat = 0; repeat < 2; repeat += 1) {
        const started = performance.now();
        // eslint-disable-next-line no-await-in-loop
        const out = await stack.run({ pair: pair.slice(), single: single.slice(), pairMask, seqMask, tokens: n }, blocks, dialect, armOptions(arm));
        last = { ms: performance.now() - started, peak: out.memory?.peakBytes ?? 0 };
      }
      const row = { family, n, arm, msPerBlock: Math.round(last.ms / blockCount), peakMiB: Math.round(last.peak / 1048576) };
      console.log(JSON.stringify(row));
      report.push(row);
    }
  }
  return report;
}
