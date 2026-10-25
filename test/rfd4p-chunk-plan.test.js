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

  it("gives up f16 exactly when it gives up the single chunk", () => {
    // 🔴 THE CHUNKED KERNELS ARE f32 ONLY, so a plan that promised f16 on a sliced track
    // would size buffers for one arithmetic and run the other.
    for (const n of [100, 200, 362, 400, 512, 513, 700, 1000]) {
      const plan = chunkPlan(n * n, { wantF16: true });
      expect(plan.triF16).toBe(plan.chunks === 1);
    }
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
    // A bound smaller than one row is still one row; zero rows is not a plan.
    const plan = chunkPlan(400 * 400, { wantF16: true, chunkBytes: 16 });
    expect(plan.R).toBe(1);
    expect(plan.triF16).toBe(false);
    expect(plan.chunks).toBe(160000);
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
