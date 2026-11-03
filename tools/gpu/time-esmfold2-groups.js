/**
 * What a channel group costs: the pair track's time per block against what it holds, one row per arm.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/time-esmfold2-groups.js \
 *       --bundle=/model-ef2-fast-300m-int5 --lengths=300,600 --blocks=12 --arms=auto,compact,g2,g4,g8,g16
 *
 * `auto` is what the track chooses with the sequential path off, `compact` the channel-chunked track, `gN` the sequential
 * one with N groups and `gNcM` that with row chunks of M MiB. Each arm runs one random pair through real weights.
 *
 * 🔴 THE TIME IS THE MEDIAN GAP BETWEEN BLOCKS FINISHING WITH FOUR QUEUED, AND NEVER A WHOLE RUN OR A DRAINED BLOCK. A run's
 * first block pays for a device's lazy clear of every new buffer and its last for the readback, so a run's mean favours
 * whichever arm holds least; and a drained block counts the device waiting on the host. Both have read a slower track as
 * faster than a fold finds it. tools/gpu/hash-af3-fold.js has the AF3 lineage's folds, where this cannot reach.
 */
import { Esmfold2TrunkGpu } from "../../webgpu/esmfold2/trunk.js";
import { setDeviceTuning } from "../../webgpu/runtime/device-profile.js";
import { reader } from "./fold-esmfold2.js";
import { trunkBlockWeights } from "../../shared/esmfold2/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

function armOptions(arm) {
  if (arm === "auto") return { forceCompact: false };
  if (arm === "compact") return { forceCompact: true };
  const [, groups, chunkMiB] = /^g(\d+)(?:c(\d+))?$/.exec(arm);
  return { forceCompact: true, sequentialTriangle: true, sequentialGroups: Number(groups),
           ...(chunkMiB === undefined ? {} : { sequentialChunkBytes: Number(chunkMiB) * 1024 * 1024 }) };
}

export async function main(device, args) {
  // The kernels a warm fold runs, as tools/gpu-chrome.mjs gives a bench-* tool, or an arm's time depends on what has been upgraded.
  setDeviceTuning(device, { runtimeLoopBounds: false });
  const bundle = option(args, "bundle", "/model-ef2-fast-300m-int5");
  const lengths = option(args, "lengths", "300,600").split(",").map(Number);
  const blockCount = Number(option(args, "blocks", "12"));
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
    for (const arm of arms) {
      const trunk = new Esmfold2TrunkGpu(device, {});
      const stamps = [];
      // eslint-disable-next-line no-await-in-loop
      const out = await trunk.run({ pair, pairMask }, blocks, {
        n, channels, leanPair: true, submissionWindow: 4, ...armOptions(arm),
        onBlockDone: () => { stamps.push(performance.now()); },
      });
      trunk.dispose?.();
      const blockTimes = stamps.slice(4).map((at, index) => at - stamps[index + 3]);
      const row = { n, arm, msPerBlock: Math.round(median(blockTimes)), fastest: Math.round(Math.min(...blockTimes)),
                    peakMiB: Math.round(out.memory.peakBytes / 1048576) };
      console.log(JSON.stringify(row));
      report.push(row);
    }
  }
  return report;
}
