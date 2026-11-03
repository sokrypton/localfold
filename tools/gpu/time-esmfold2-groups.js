/**
 * What a channel group costs: the pair track's time per block against what it holds, one row per arm.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/time-esmfold2-groups.js \
 *       --bundle=/model-ef2-fast-300m-int5 --lengths=300,600 --blocks=4 --arms=auto,compact,g2,g4,g8,g16
 *
 * `auto` is what the track chooses with the sequential path off, `compact` the channel-chunked track, and `gN` the
 * sequential one with N groups. Each arm runs twice over one random pair with real weights and the second run is the one
 * reported, so a pipeline's compile is not in its time. Each block's pair goes through the same weights as the fold's.
 */
import { Esmfold2TrunkGpu } from "../../webgpu/esmfold2/trunk.js";
import { reader } from "./fold-esmfold2.js";
import { trunkBlockWeights } from "../../shared/esmfold2/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const bundle = option(args, "bundle", "/model-ef2-fast-300m-int5");
  const lengths = option(args, "lengths", "300,600").split(",").map(Number);
  const blockCount = Number(option(args, "blocks", "4"));
  const arms = option(args, "arms", "auto,compact,g2,g4,g8,g16").split(",");
  const { read } = reader(bundle);
  const blocks = [];
  for (let layer = 0; layer < blockCount; layer += 1) {
    // eslint-disable-next-line no-await-in-loop
    blocks.push(await trunkBlockWeights(read, layer));
  }
  const channels = 256;
  const report = [];
  for (const n of lengths) {
    const pairs = n * n;
    const pair = new Float32Array(pairs * channels);
    let seed = 12345;
    for (let i = 0; i < pair.length; i += 1) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      pair[i] = (seed / 0x7fffffff) * 2 - 1;
    }
    const pairMask = new Float32Array(pairs).fill(1);
    const optionsOf = (arm) => {
      if (arm === "auto") return { forceCompact: false };
      if (arm === "compact") return { forceCompact: true };
      return { forceCompact: true, sequentialTriangle: true, sequentialGroups: Number(arm.slice(1)) };
    };
    for (const arm of arms) {
      const trunk = new Esmfold2TrunkGpu(device, {});
      const options = { n, channels, leanPair: true, submissionWindow: 1, ...optionsOf(arm) };
      let last;
      for (let repeat = 0; repeat < 2; repeat += 1) {
        const started = performance.now();
        // eslint-disable-next-line no-await-in-loop
        const out = await trunk.run({ pair, pairMask }, blocks, options);
        last = { ms: performance.now() - started, peak: out.memory.peakBytes };
      }
      trunk.dispose?.();
      const row = { n, arm, msPerBlock: Math.round(last.ms / blockCount), peakMiB: Math.round(last.peak / 1048576) };
      console.log(JSON.stringify(row));
      report.push(row);
    }
  }
  return report;
}
