import { describe, expect, it } from "./harness.js";
import {
  transitionChunkRows, TRANSITION_CHUNK_TARGET_BYTES, TRANSITION_TILE_ROWS,
} from "../src/evoformer/transition.js";

/**
 * WHY A TENSOR THAT FITS CAN STILL NOT BE BOUND.
 *
 * `maxBufferSize` and `maxStorageBufferBindingSize` are different limits, and
 * the second is the smaller one. A transition's hidden activation is
 * rows * hiddenChannels floats - 508 MSA rows of a 291-residue alignment is
 * 147,828 * 1024 * 4 bytes, 578 MiB - which allocates on any modern adapter and
 * then fails validation when it is bound. Chunking the rows is what makes long
 * sequences possible at all, and this file pins the arithmetic that decides how.
 */
const MiB = 1024 * 1024;

describe("transition chunking", () => {
  it("leaves short inputs alone, so nothing about them changes", () => {
    // ...THE FULL PATH IS THE DEFAULT. 508 rows of a 59-residue alignment is
    // 29,972 rows, and at 1024 hidden channels that is 117 MiB - under the
    // limit, so the caller keeps its single-dispatch branch.
    expect(transitionChunkRows(508 * 59, 256, 1024, 128 * MiB, 256)).toBe(508 * 59);
    expect(transitionChunkRows(59, 128, 512, 128 * MiB, 256)).toBe(59);
  });

  it("splits an input that cannot be bound whole", () => {
    expect(transitionChunkRows(508 * 291, 256, 1024, 256 * MiB, 256)).toBe(24_576);
  });

  it("aligns a chunk to the row tile AND to the binding offset", () => {
    // 🔴 BOTH ALIGNMENTS ARE LOAD-BEARING. The linear kernels tile rows by 16,
    // so a ragged chunk leaves an incomplete tile; and each chunk BINDS at a row
    // offset, so an offset that is not a multiple of 256 bytes is a validation
    // error rather than a slow path.
    for (const [rows, channels, hidden, limit] of [
      [508 * 291, 256, 1024, 256 * MiB],
      [508 * 291, 256, 1024, 128 * MiB],
      [1024 * 400, 64, 256, 64 * MiB],
      [508 * 512, 256, 1024, 96 * MiB],
    ]) {
      const chunk = transitionChunkRows(rows, channels, hidden, limit, 256);
      expect(chunk % TRANSITION_TILE_ROWS).toBe(0);
      expect((chunk * channels * 4) % 256).toBe(0);
      // ...and it actually fits what it was sized against
      expect(chunk * Math.max(channels, hidden) * 4 <= Math.min(limit, TRANSITION_CHUNK_TARGET_BYTES))
        .toBe(true);
      expect(chunk >= 1 && chunk <= rows).toBe(true);
    }
  });

  it("never returns more rows than it was given", () => {
    expect(transitionChunkRows(100, 256, 1024, 1 * MiB, 256) <= 100).toBe(true);
  });

  it("refuses limits it cannot honour rather than returning a bad chunk", () => {
    // one row of hidden activation larger than the whole binding
    expect(() => transitionChunkRows(1000, 256, 1024, 1024, 256)).toThrow();
    // a binding that cannot hold even one aligned chunk
    expect(() => transitionChunkRows(1_000_000, 256, 1_000_000, 8 * MiB, 256)).toThrow();
    expect(() => transitionChunkRows(0, 256, 1024, 128 * MiB, 256)).toThrow();
    expect(() => transitionChunkRows(10, -1, 1024, 128 * MiB, 256)).toThrow();
  });

  it("covers every row when the caller walks it", () => {
    // the loop shape the caller uses: the last chunk is usually short
    const rows = 508 * 291;
    const chunk = transitionChunkRows(rows, 256, 1024, 256 * MiB, 256);
    let covered = 0;
    for (let offset = 0; offset < rows; offset += chunk) {
      covered += Math.min(chunk, rows - offset);
    }
    expect(covered).toBe(rows);
  });
});
