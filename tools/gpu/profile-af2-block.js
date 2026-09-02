/**
 * Per-kernel GPU time inside one AF2 evoformer block.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/profile-af2-block.js
 *     node tools/gpu-chrome.mjs tools/gpu/profile-af2-block.js --length=200 --sequences=32
 *
 * 🔴 THE PROFILER THIS DRIVES WAS ALREADY BUILT AND NOTHING RAN IT.
 * src/runtime/execution.js has beginTimestampProfile/finishTimestampProfile/
 * readTimestampProfile, dispatch() already threads a label through every
 * kernel, and src/evoformer/stack.js already honours `profileBlock` - it gives
 * the chosen block a pass per dispatch instead of the batched one, so the
 * labels survive. All of that existed with no caller. This is the caller.
 *
 * 🔴 AND IT IS WHY tools/gpu/profile.js CANNOT SEE INTO AF2. That wrapper times
 * whole compute passes, and outside profiling mode this stack deliberately
 * batches every dispatch of a block into ONE pass called "localfold.compute" -
 * so the wrapper reports 1218 ms against a single label and attributes nothing.
 * The batching is load-bearing for speed; the switch is the right seam.
 *
 * WHAT IT FOUND, on 59 residues, as block time and the 48-block stack it implies:
 *
 *      5 rows    12.6 ms    603 ms    opm.accumulate 1.21, tri-attention 0.98 x2
 *    128 rows    61.1 ms   2930 ms    msa-column-attention.flash 7.7, opm.contract 7.5
 *    512 rows   294.9 ms  14154 ms    msa-column-attention.flash 125.6, opm.contract 29.3
 *
 * 🔴 THE SINGLE-SEQUENCE BLOCK HAS NO HOT KERNEL, AND THE DEEP ONE IS ALMOST
 * ALL ONE. At 5 rows the largest kernel is under 10% of the block and the tail
 * is flat - there is no pair-transition here, which was 38% of AF3's pairformer
 * and worth 2.8x. At 512 rows msa-column-attention.flash is 43% of the block on
 * its own, and it grows 16x for 4x the rows: column attention is over the
 * SEQUENCE axis, so it is quadratic in depth where everything else is linear.
 * At that depth it runs about 63 GFLOP/s, a couple of percent of this device,
 * so it is the one AF2 kernel that looks worth attacking - and only for folds
 * with a real alignment.
 *
 * 🔴 ONE BLOCK, NOT THE STACK. Profiling forces a submission window of 1 and a
 * device sync around the profiled block, so a whole-stack profile would measure
 * a differently-scheduled machine. Every block has identical shapes, so one
 * block times them all; multiply by blockWeights.length for a stack.
 */
import { AlphaFoldFixture } from "../../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { EvoformerStackGpu } from "../../src/evoformer/stack.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const length = Number(option(args, "length", "59"));
  const sequences = Number(option(args, "sequences", "5"));
  const block = Number(option(args, "block", "1"));
  const cM = 256;

  const { MODEL_BUNDLES, loadManifest } = await import("../../src/reference/manifests/index.js");
  const fixture = AlphaFoldFixture.fromStore(await HttpTensorStore.fromManifest(
    MODEL_BUNDLES.monomer.directory, await loadManifest("monomer")));
  const blockWeights = await fixture.mainStackWeights();

  // 🔴 SYNTHETIC ACTIVATIONS, WHICH IS SOUND FOR TIMING AND ONLY FOR TIMING.
  // Every dispatch is the same size whatever the numbers are and nothing here
  // branches on data. The WEIGHTS are the real exported bundle, because their
  // layout and precision do affect memory traffic. bench-blocks.js says the
  // same thing at more length.
  const noise = (count, seed) => {
    const values = new Float32Array(count);
    let state = seed >>> 0;
    for (let index = 0; index < count; index += 1) {
      state = (state * 1664525 + 1013904223) >>> 0;
      values[index] = (state / 4294967296) - 0.5;
    }
    return values;
  };

  const result = await new EvoformerStackGpu(device).run({
    msa: noise(sequences * length * cM, 1),
    pair: noise(length * length * 128, 2),
    msaMask: new Float32Array(sequences * length).fill(1),
    pairMask: new Float32Array(length * length).fill(1),
    sequences, length, cM, cZ: 128, cOuter: 32, triangleHidden: 128, blockWeights,
    // ...block 1, not 0: the first compiles every pipeline.
    profileBlock: block,
  });

  const profile = result.timestampProfile ?? [];
  const totals = new Map();
  for (const { label, nanoseconds } of profile) {
    const found = totals.get(label) ?? { label, ms: 0, dispatches: 0 };
    found.ms += nanoseconds / 1e6;
    found.dispatches += 1;
    totals.set(label, found);
  }
  const kernels = [...totals.values()]
    .map((row) => ({ ...row, ms: Number(row.ms.toFixed(3)) }))
    .sort((a, b) => b.ms - a.ms);
  const blockMs = kernels.reduce((sum, row) => sum + row.ms, 0);
  return {
    length, sequences, block, blocks: blockWeights.length,
    dispatches: profile.length,
    blockMs: Number(blockMs.toFixed(2)),
    stackMs: Number((blockMs * blockWeights.length).toFixed(0)),
    kernels: kernels.slice(0, 20),
  };
}
