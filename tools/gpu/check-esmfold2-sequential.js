/**
 * Does the sequential triangle compute the pair track's own answer, to the bit?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-esmfold2-sequential.js \
 *       --bundle=/model-ef2-fast-300m-int5 --lengths=130,257 --blocks=2
 *
 * The sequential track holds the pair, the contraction's output and a and b for one channel group, and takes the
 * normalised pair and hidden a chunk of rows at a time. It adds no arithmetic: LayerNorm(pair) is computed again from a
 * pair that has not changed, and the projections read the same values. So the bar is ZERO differing elements against
 * the whole-tensor track, not a tolerance, and any difference at all is a mapping: a row base, a chunk that does not
 * cover the rows, a tail tile that reads past its chunk.
 *
 * The lengths are chosen for that tail. 130 and 257 tokens square to 16,900 and 66,049 pair rows, neither a multiple of
 * the kernels' 32-row tile or of the chunk, so the last chunk has a short tile and the mask's rows are offset. A
 * deliberately small chunk and two group counts walk the kernels through more chunks than a real fold at these sizes.
 *
 * Both arms run in one process over one input, so a second process's random pair cannot be what differs.
 */
import { Esmfold2TrunkGpu } from "../../webgpu/esmfold2/trunk.js";
import { reader } from "./fold-esmfold2.js";
import { GpuBufferAllocator } from "../../webgpu/runtime/allocator.js";
import { SegmentedAllocation } from "../../webgpu/runtime/segmented.js";
import { trunkBlockWeights } from "../../shared/esmfold2/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const bundle = option(args, "bundle", "/model-ef2-fast-300m-int5");
  const lengths = option(args, "lengths", "130,257").split(",").map(Number);
  const blockCount = Number(option(args, "blocks", "2"));
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
    // A mask with holes, because a row base that is wrong there is wrong only where the mask is not all ones.
    const pairMask = new Float32Array(pairs).fill(1);
    for (let i = 0; i < pairs; i += 97) pairMask[i] = 0;

    const run = async (label, options) => {
      const started = performance.now();
      const trunk = new Esmfold2TrunkGpu(device, { residentWeights: false });
      const out = await trunk.run({ pair: pair.slice(), pairMask }, blocks,
                                  { n, channels, leanPair: true, submissionWindow: 1, ...options });
      return { label, pair: out.pair, ms: performance.now() - started, peak: out.memory.peakBytes };
    };
    // The same track over a pair of several buffers, written and read back a segment at a time.
    const runSegmented = async (label, segmentRows, options) => {
      const allocator = new GpuBufferAllocator(device);
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      const held = new SegmentedAllocation(allocator, "pair", pairs, channels * 4, usage, segmentRows);
      held.segments.forEach((segment, at) => device.queue.writeBuffer(
        segment.buffer, 0, pair, at * segmentRows * channels, Math.min(segmentRows, pairs - at * segmentRows) * channels));
      const trunk = new Esmfold2TrunkGpu(device, { residentWeights: false, allocator });
      await trunk.run({ buffer: held, pairMask }, blocks,
                      { n, channels, leanPair: true, submissionWindow: 1, readback: false, ...options });
      const back = new Float32Array(pair.length);
      for (const [at, segment] of held.segments.entries()) {
        const bytes = Math.min(segmentRows, pairs - at * segmentRows) * channels * 4;
        const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(segment.buffer, 0, staging, 0, bytes);
        device.queue.submit([encoder.finish()]);
        // eslint-disable-next-line no-await-in-loop
        await staging.mapAsync(GPUMapMode.READ);
        back.set(new Float32Array(staging.getMappedRange()), at * segmentRows * channels);
        staging.destroy();
      }
      held.release();
      return { label, pair: back, ms: 0 };
    };
    // eslint-disable-next-line no-await-in-loop
    const whole = await run("whole", { forceCompact: false });
    const arms = [
      // eslint-disable-next-line no-await-in-loop
      await run("compact", { forceCompact: true }),
      // eslint-disable-next-line no-await-in-loop
      await run("sequential g8", { forceCompact: true, sequentialTriangle: true, sequentialGroups: 8,
                                   sequentialChunkBytes: 1048576 }),
      // eslint-disable-next-line no-await-in-loop
      await run("sequential g4", { forceCompact: true, sequentialTriangle: true, sequentialGroups: 4,
                                   sequentialChunkBytes: 4194304 }),
      // eslint-disable-next-line no-await-in-loop
      await run("sequential g16", { forceCompact: true, sequentialTriangle: true, sequentialGroups: 16,
                                    sequentialChunkBytes: 262144 }),
      // eslint-disable-next-line no-await-in-loop
      await run("sequential, output in two buffers", {
        forceCompact: true, sequentialTriangle: true, sequentialGroups: 8, sequentialHiddenParts: 2,
        sequentialChunkBytes: 1048576,
      }),
      // eslint-disable-next-line no-await-in-loop
      await runSegmented("sequential, pair in segments", 16384, {
        forceCompact: true, sequentialTriangle: true, sequentialGroups: 8, sequentialHiddenParts: 2,
        sequentialChunkBytes: 1048576,
      }),
    ];
    for (const arm of arms) {
      let differing = 0;
      for (let i = 0; i < whole.pair.length; i += 1) {
        if (Object.is(whole.pair[i], arm.pair[i])) continue;
        differing += 1;
      }
      report.push({ n, arm: arm.label, differing, ms: Math.round(arm.ms), wholeMs: Math.round(whole.ms) });
      if (differing !== 0) {
        throw new Error(`${arm.label} at ${n} tokens differs from the whole-tensor track in ${differing} elements`);
      }
    }
    // Finite, or two arms agreeing on NaN would pass.
    if (!whole.pair.every(Number.isFinite)) throw new Error(`the pair at ${n} tokens has non-finite elements`);
  }
  return report;
}
