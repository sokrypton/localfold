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
 * WHAT IT FINDS, on 59 residues, as block time and the 48-block stack it
 * implies:
 *
 *      5 rows    13.5 ms    646 ms    opm.accumulate 2.04, opm.intermediate 1.07
 *    128 rows    32.8 ms   1574 ms    the two attention projections 4.07 each
 *    512 rows    90.1 ms   4326 ms    msa-column-attention.flash 16.7
 *
 * 🔴 THE DEEP BLOCK IS NO LONGER ALMOST ALL ONE KERNEL, AND THIS DOCSTRING
 * SAID IT WAS FOR A WHILE AFTER IT STOPPED BEING TRUE. It recorded 294.9 ms at
 * 512 rows with msa-column-attention.flash at 125.6 - 43% of the block, "the
 * one AF2 kernel that looks worth attacking". Two things happened to that.
 * `selectAttentionFlashKernel` changed its default to the register-resident
 * kernel, which is 3.7x faster here and took column attention to 21 ms on its
 * own; and the three kernels that were then left leading - the transition's
 * dense projection, the outer product mean's contraction and the attention
 * output projection - each turned out to be reading one operand a float at a
 * time, and were given vector operands.
 *
 * What is left is FLAT. At 512 rows the top five kernels are within 16.7 to 13.6
 * ms of each other and every one of them runs between 900 and 1150 GFLOP/s, which
 * tools/gpu/probe-alu.js puts at 70-89% of this device's scalar multiply-add
 * ceiling. There is no outlier to attack: the next thing here is a different
 * algorithm, not a better tile, and the two benches that exist for these
 * kernels (bench-evoformer-linear.js, bench-msa-attention.js) both report their
 * current shape as the optimum of everything tried.
 *
 * 🔴 WHAT MOVED AFTERWARDS WAS NOT A TILE EITHER - IT WAS THE ELEMENT. The
 * block went 109.25 -> 90.12 ms, and every step of that was half precision
 * somewhere, for two different reasons:
 *
 *   - COLUMN ATTENTION 21.0 -> 16.7 and row attention 3.9 -> 2.9, by staging
 *     the key and value in f16. Not arithmetic: the surgery arms in
 *     bench-msa-attention.js price those staged reads at 8.7 ms of 20.8,
 *     against 0.4 for both exponentials. Halving the workgroup memory buys the
 *     occupancy. Everything tried that ADDED registers lost, by 2.3x and 4.7x.
 *   - THE THREE DENSE KERNELS 13.6/13.6 -> 9.7/9.7 (the q/k/v/gate
 *     projections), 4.2/4.2 -> 3.5/3.5 (the outputs) and the transitions with
 *     them, by putting the ACCUMULATORS in f16 - which is what let each of them
 *     take a tile that spilled in f32. The register budget was the ceiling and
 *     half precision moved it.
 *
 * So "no outlier to attack, the next thing is a different algorithm" was right
 * about tiles and wrong about the kernel as a whole: what was left was a
 * storage format, and it was worth 17.5%. The benches for these are
 * bench-msa-attention.js, bench-evoformer-linear.js and the new
 * bench-attention-project.js.
 *
 * 🔴 AND THE BLOCK TOTAL IS NOT COMPARABLE ACROSS PROCESSES. This machine
 * drifts up to 3.2x between them - measured in this repo's own bench, where one
 * arm read 32.1 ms and 15.9 ms an hour apart. The per-dispatch numbers within a
 * run are stable to about 1%, and a claim about a change belongs in a bench
 * that interleaves its arms, or in tools/gpu/fold-af2.js's end-to-end time.
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
