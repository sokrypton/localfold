import { describe, expect, it } from "./harness.js";
import { DEFAULT_CHUNK_BYTES, chunkPlan } from "../webgpu/rfd4p/trunk.js";

/**
 * 🔴 THE CHUNK DECISION IS THE ONE THAT DECIDES THE ARITHMETIC, and it has no GPU in it.
 *
 * `webgpu/rfd4p/trunk.js` slices the pair track when its scratch will not fit a bound, and
 * a sliced pair track runs a DIFFERENT set of kernels in f32 - the arm that never got the
 * f16 narrowing or the fused gate. So where that threshold falls is worth more than a
 * kernel: at n = 400 it was the difference between 1815.8 ms and 3523.7 ms a trunk pass.
 *
 * It used to fall in the wrong place for a reason nothing could see without a GPU: `triF16`
 * tested `R === nn` while R was computed from a bound in f32 terms, so f16 could never buy
 * the length it paid for. That is arithmetic, which means it is gateable here rather than
 * only in a browser on a machine with the right adapter - and these are the numbers the
 * comment in that file quotes.
 */
describe("the rfd4p trunk's pair-track chunk plan", () => {
  it("keeps one chunk to n = 362 in f32, which is where it always fell", () => {
    // 🔴 THE f32 ARM IS UNCHANGED ON PURPOSE. Restating the bound in bytes was meant to let
    // f16 reach further, not to move the strict path, so 362/363 is the assertion that the
    // restatement was faithful rather than a retune wearing its clothes.
    expect(chunkPlan(362 * 362, { wantF16: false }).chunks).toBe(1);
    expect(chunkPlan(363 * 363, { wantF16: false }).chunks).toBe(2);
  });

  it("keeps one chunk to n = 512 in f16, because half the bytes is twice the rows", () => {
    expect(chunkPlan(512 * 512, { wantF16: true }).chunks).toBe(1);
    expect(chunkPlan(512 * 512, { wantF16: true }).triF16).toBe(true);
    expect(chunkPlan(513 * 513, { wantF16: true }).chunks).toBeGreaterThan(1);
  });

  it("keeps f16 when it gives up the single chunk, which it did not used to", () => {
    // 🔴 THIS ASSERTION IS THE REVERSE OF THE ONE IT REPLACES, and deliberately. It used to
    // read `expect(plan.triF16).toBe(plan.chunks === 1)`, because the chunked kernels were
    // f32 only and a plan promising f16 on a sliced track would have sized buffers for one
    // arithmetic and run the other. They read and write halves now, so the arithmetic is a
    // property of the DEVICE and the request and the slicing is a property of the length -
    // two independent questions that were one.
    //
    // What it cost to have them coupled, profiled at the residue where they met: crossing
    // n = 512 to 513 widened the pair track back to f32 at the same moment the slicing
    // changed, so memory rose 46% (2056 -> 3001 MiB) and the pass took 2.02x as long
    // (2914 -> 5882 ms). A cliff of that size in one residue is what a coupling looks like
    // from the outside.
    for (const n of [100, 200, 362, 400, 512, 513, 700, 1000]) {
      expect(chunkPlan(n * n, { wantF16: true }).triF16).toBe(true);
    }
    // And the slicing still moves where it always did.
    expect(chunkPlan(512 * 512, { wantF16: true }).chunks).toBe(1);
    expect(chunkPlan(513 * 513, { wantF16: true }).chunks).toBeGreaterThan(1);
  });

  it("never asks f16 of a device that cannot do it", () => {
    for (const n of [100, 400, 1000]) {
      expect(chunkPlan(n * n, { wantF16: false }).triF16).toBe(false);
    }
  });

  it("honours the byte bound in BOTH shapes, which an earlier draft did not", () => {
    // 🔴 THIS IS THE REGRESSION TEST FOR A BUG THAT WAS WRITTEN AND CAUGHT. Sizing R from
    // f16's row cost and then allocating f32 buffers against it put 1.88 GB behind a 896 MB
    // bound at n = 520 - it ran only because the machine it was written on has 4 GB
    // bindings. The bound has to hold whichever shape is chosen.
    for (const n of [54, 100, 200, 362, 363, 400, 512, 513, 520, 700, 1200]) {
      for (const wantF16 of [true, false]) {
        const plan = chunkPlan(n * n, { wantF16 });
        // Integers, so `< bound + 1` is `<= bound` and the harness has no inclusive matcher.
        expect(plan.scratchBytes).toBeLessThan(DEFAULT_CHUNK_BYTES + 1);
      }
    }
  });

  it("covers every pair row, with chunks that tile the tensor", () => {
    for (const n of [54, 100, 363, 400, 513, 1000]) {
      const nn = n * n;
      const plan = chunkPlan(nn, { wantF16: true });
      expect(plan.R * plan.chunks).toBeGreaterThan(nn - 1);
      // One chunk short of the count must NOT cover it, or the count is padded.
      expect(plan.R * (plan.chunks - 1)).toBeLessThan(nn);
    }
  });

  it("still returns a usable plan when the bound is tiny", () => {
    // A bound smaller than one row is still a plan - and the floor is FOUR rows, not one,
    // because `transposeLnOut` reads four pair rows a word and a chunk must start on that
    // boundary. Zero rows is not a plan and one row is a wrong one.
    const plan = chunkPlan(400 * 400, { wantF16: true, chunkBytes: 16 });
    expect(plan.R).toBe(4);
    // f16 survives a bound this small now: narrowing no longer depends on the slicing.
    expect(plan.triF16).toBe(true);
    expect(plan.chunks).toBe(40000);
  });

  it("starts every chunk on a multiple of four pair rows", () => {
    // 🔴 THE INVARIANT `transposeLnOut` DEPENDS ON, and the one this file exists to keep: a
    // chunk begins at k * R, the kernel reads `x` as vec4 along the row axis, and a start
    // off that boundary reads every row of every channel from the wrong word. Measured at
    // n = 60 against the single-chunk arm: R = 450 and 225 divide 3600 EXACTLY and come
    // back at relRMS 0.89 and 1.04 on the pair track, where 1800, 1208, 1200, 900 and 512
    // all agree at 1.3e-6. So raggedness is fine and a misaligned start is not.
    //
    // Swept over bounds rather than a chosen few, because the default bound happens to
    // give R = 101944 = 4 * 25486 and has hidden this since the chunking was written.
    for (let mb = 1; mb <= 2048; mb += 1) {
      for (const nn of [400 * 400, 513 * 513, 1000 * 1000]) {
        const { R, chunks } = chunkPlan(nn, { wantF16: true, chunkBytes: mb * 1024 * 1024 });
        if (chunks === 1) continue;
        expect(R % 4).toBe(0);
      }
    }
  });

  it("pins the default plan's rows, four-aligned", () => {
    // The bound the trunk ships with, stated so a change to the row cost has to be
    // deliberate. It was 101944 while gz was R-indexed and is 122332 now that gz is held
    // whole - the bound describes the R-indexed scratch, so taking a buffer out of that set
    // buys rows rather than bytes.
    const { R, chunks } = chunkPlan(600 * 600, {});
    expect(R).toBe(122332);
    expect(R % 4).toBe(0);
    expect(chunks).toBe(3);
  });

  it("gives a bigger bound more rows, monotonically", () => {
    let previous = 0;
    for (const mb of [16, 64, 256, 896, 2048]) {
      const { R } = chunkPlan(1000 * 1000, { wantF16: true, chunkBytes: mb * 1024 * 1024 });
      expect(R).toBeGreaterThan(previous - 1);
      previous = R;
    }
  });
});
