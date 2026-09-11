/**
 * The outer product mean's pair blocks must start on a residue whose view of
 * `left` lands on a 256-byte boundary.
 *
 * 🔴 AN ODD MSA DEPTH KILLED EVERY FOLD LONG ENOUGH TO BLOCK, and nothing on
 * the CPU side could see it. The matrix contraction binds `left` as a view
 * starting at the block's first residue - `residuesBefore * cOuter * sequences`
 * elements - and WebGPU refuses a bound range that does not start on 256 bytes.
 * With cOuter 32 that offset is `residuesBefore * 128 * sequences` bytes, a
 * multiple of 256 only when the depth is EVEN. Measured on a 400-residue chain:
 * depths 38 and 40 folded, 37 and 39 died inside WebGPU naming a buffer they
 * had nothing to do with, because the allocator pools by size and a pooled
 * buffer keeps its creation label.
 *
 * The cluster count is `min(508, depth)`, so a real alignment with an odd
 * number of rows reached this - half of all shallow ones.
 *
 * This is a CPU test because the arithmetic is where the bug was; the GPU-side
 * guard is the 256 check in src/runtime/execution.js, which names the binding
 * rather than letting WebGPU name the wrong tensor.
 */
import test from "node:test";
import assert from "node:assert/strict";

import { outerFirstPairBlocks } from "../src/evoformer/outer-product-mean.js";

/** The rule block.js applies, restated here so the test fails if it changes. */
function residueMultiple(cOuter, sequences) {
  const rowBytes = cOuter * sequences * 4;
  const commonFactor = (a, b) => (b === 0 ? a : commonFactor(b, a % b));
  return 256 / commonFactor(rowBytes % 256 === 0 ? 256 : rowBytes % 256, 256);
}

// 🔴 THE BUDGET DECIDES WHETHER THE BUG APPEARS, so it is swept. A first
// version of this test passed a 64 MiB limit, which at 400 residues gives 40
// residues a block - an even number, aligned by luck - and it caught nothing.
// The shipped ampere budget is 256 MiB, which gives 163, and 163 is odd. A test
// that does not vary the budget is testing one accident.
const BUDGETS = [16, 32, 64, 128, 256, 512].map((mib) => mib * 1024 * 1024);

test("a pair block's view of left starts on 256 bytes at every MSA depth", () => {
  const cOuter = 32;
  for (const length of [59, 128, 400, 401, 825]) {
    for (let sequences = 1; sequences <= 130; sequences += 1) {
    for (const budget of BUDGETS) {
      const input = { length, cOuter, sequences };
      const blocks = outerFirstPairBlocks(
        input, 4 * 1024 * 1024 * 1024, budget,
        length * residueMultiple(cOuter, sequences),
      );
      let seen = 0;
      for (const [offset, count] of blocks) {
        assert.equal(offset % length, 0,
          `depth ${sequences}, length ${length}: block at ${offset} is not a whole `
          + "number of i rows");
        const first = (offset / length) * cOuter * sequences;
        assert.equal((first * 4) % 256, 0,
          `depth ${sequences}, length ${length}: the view of left for the block at `
          + `${offset} starts at byte ${first * 4}, which is not a multiple of 256`);
        seen += count;
      }
      assert.equal(seen, length * length,
        `depth ${sequences}, length ${length}: the blocks do not cover every pair`);
    }
    }
  }
});

test("and the old rule, which used the residue count alone, does NOT", () => {
  // 🔴 A GATE THAT CANNOT FAIL IS NOT A GATE. These are the exact parameters
  // that killed a real fold: 400 residues, 37 rows, the ampere prior's 256 MiB
  // block. Under the old `multiple = length` the third block starts at residue
  // 163, and 163 * 32 * 37 * 4 is byte 771968, which is 128 past a boundary.
  const cOuter = 32;
  const length = 400;
  const sequences = 37;
  const blocks = outerFirstPairBlocks(
    { length, cOuter, sequences }, 4 * 1024 * 1024 * 1024, 256 * 1024 * 1024, length);
  const misaligned = blocks.filter(([offset]) =>
    (((offset / length) * cOuter * sequences) * 4) % 256 !== 0);
  assert.ok(misaligned.length > 0,
    "the old rule should still produce a misaligned view, or this test proves nothing");
  assert.equal((((misaligned[0][0] / length) * cOuter * sequences) * 4), 771968);
});

test("an even depth is unchanged, so no alignment that worked before moves", () => {
  const cOuter = 32;
  for (const sequences of [2, 38, 40, 128, 508, 510, 512]) {
    assert.equal(residueMultiple(cOuter, sequences), 1,
      `depth ${sequences} should need no extra residue alignment`);
  }
  for (const sequences of [1, 37, 39, 61, 507]) {
    assert.equal(residueMultiple(cOuter, sequences), 2,
      `depth ${sequences} is odd and needs two residues a block`);
  }
});
