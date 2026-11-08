import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { GROUPS_MARGIN_BYTES, MOST_SEQUENTIAL_GROUPS, SEQUENTIAL_CHUNK_BYTES, fewestSequentialGroups, rowPieces }
  from "../webgpu/af3/trunk/sequential-groups.js";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

describe("the channel groups the sequential triangle takes", () => {
  const pairBytes = GIB;
  const residentBytes = 2 * GIB;
  // What everything but a and b adds up to.
  const beside = residentBytes + pairBytes + 2 * SEQUENTIAL_CHUNK_BYTES + GROUPS_MARGIN_BYTES;

  it("takes the fewest with no budget to answer to", () => {
    assert.equal(fewestSequentialGroups({ pairBytes, residentBytes }), 2);
  });

  it("takes two wherever a and b for a half fit, to the byte", () => {
    assert.equal(fewestSequentialGroups({ pairBytes, residentBytes, budgetBytes: beside + pairBytes }), 2);
    assert.equal(fewestSequentialGroups({ pairBytes, residentBytes, budgetBytes: beside + pairBytes - 1 }), 4);
  });

  it("takes four, and then eight, as the room runs out", () => {
    assert.equal(fewestSequentialGroups({ pairBytes, residentBytes, budgetBytes: beside + pairBytes / 2 }), 4);
    assert.equal(fewestSequentialGroups({ pairBytes, residentBytes, budgetBytes: beside + pairBytes / 2 - 1 }), 8);
    assert.equal(fewestSequentialGroups({ pairBytes, residentBytes, budgetBytes: beside + pairBytes / 4 }), 8);
  });

  it("takes the most where nothing fits, and leaves saying so to the budget", () => {
    assert.equal(fewestSequentialGroups({ pairBytes, residentBytes, budgetBytes: GIB }), MOST_SEQUENTIAL_GROUPS);
  });

  it("keeps a and b inside one buffer whatever the budget", () => {
    const large = 3 * GIB;
    assert.equal(fewestSequentialGroups({ pairBytes: large, maxBufferBytes: 2 * GIB }), 4);
    assert.equal(fewestSequentialGroups({ pairBytes: large, maxBufferBytes: 4 * GIB }), 2);
  });

  it("refuses a pair whose a and b fit no buffer at the most groups, by name", () => {
    assert.equal(fewestSequentialGroups({ pairBytes: 16 * GIB, maxBufferBytes: 4 * GIB }), MOST_SEQUENTIAL_GROUPS);
    assert.throws(() => fewestSequentialGroups({ pairBytes: 16 * GIB + 8, maxBufferBytes: 4 * GIB }),
      /past this device's largest buffer/);
  });

  it("does not treat a missing budget as none left", () => {
    assert.equal(fewestSequentialGroups({ pairBytes, residentBytes, budgetBytes: null }), 2);
  });
});

describe("the pieces a kernel over rows is dispatched in", () => {
  it("covers every row once, in order", () => {
    for (const [rows, rowWork, stepWork, tile] of [[1_000_000, 1_048_576, 100e9, 8], [179_776, 262_144, 100e9, 16], [100, 1, 1e12, 1],
      [4_190_209, 32_768, 8e9, 1], [7, 1, 1, 8]]) {
      let next = 0;
      for (const { first, count } of rowPieces(rows, rowWork, stepWork, tile)) {
        assert.equal(first, next);
        assert.ok(count > 0);
        next += count;
      }
      assert.equal(next, rows);
    }
  });

  it("keeps a piece within its work and whole tiles, except the last", () => {
    const pieces = rowPieces(1_000_000, 1_048_576, 100e9, 8);
    assert.ok(pieces.length > 1);
    for (const { first, count } of pieces) {
      assert.equal(first % 8, 0);
      assert.ok(count * 1_048_576 <= 100e9);
    }
    for (const { count } of pieces.slice(0, -1)) assert.equal(count % 8, 0);
  });

  it("is one dispatch where the whole fits in one", () => {
    assert.deepEqual(rowPieces(90_000, 65_536, 100e9, 8), [{ first: 0, count: 90_000 }]);
  });

  it("takes a whole tile however little work a step allows", () => {
    assert.deepEqual(rowPieces(16, 1e9, 1, 8), [{ first: 0, count: 8 }, { first: 8, count: 8 }]);
  });
});
