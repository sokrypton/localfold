// The transition's staged block, at every shape in the tree.
//
// 🔴 THE TILE AND THE CHUNK ARE A SCHEDULING CHOICE, SO NOTHING ELSE CATCHES
// THEM. Every arm of tools/gpu/bench-transition.js is bit-identical - relRMS 0
// across eight tile/chunk pairs - so a rule that re-tunes a stack changes only
// its SPEED, silently, and no differential checker anywhere will say so. This
// pins the table in transition-webgpu.js's STAGED_TILE_FLOATS note: the rule
// generalised from the row count to the channel count to fix ESMFold2's trunk,
// and the point of the generalisation is that only that row moves.
import { describe, expect, it } from "./harness.js";
import { transitionChunk, transitionRowTile } from "../src/af3/transition-webgpu.js";

// stack, rows, channels, factor, expected tile, expected chunk
const SHAPES = [
  ["AF3 pair track, 200 tokens", 40_000, 128, 4, 8, 128],
  ["AF3 pair track, 300 tokens", 90_000, 128, 4, 8, 128],
  ["AF3 MSA stack", 32 * 200, 64, 4, 8, 128],
  ["AF3 template stack", 40_000, 64, 2, 8, 128],
  ["diffusion conditioning, pair", 40_000, 128, 2, 8, 128],
  ["diffusion conditioning, single", 200, 384, 2, 1, 768],
  ["pairformer single track", 200, 384, 4, 1, 1536],
  ["ESMFold2 trunk, 300 tokens", 90_000, 256, 4, 4, 256],
  ["ESMFold2 trunk, 40 tokens", 1_600, 256, 4, 4, 256],
];

describe("the transition's staged block", () => {
  for (const [label, rows, channels, factor, tile, chunk] of SHAPES) {
    it(`is ${tile}x${chunk} for ${label}`, () => {
      expect(transitionRowTile(rows, channels)).toBe(tile);
      expect(transitionChunk(channels * factor, tile, 128)).toBe(chunk);
    });
  }

  it("keeps the staged block near one target, whatever the channels", () => {
    // What the rule is FOR: `tile * channels` and `tile * chunk` are the two
    // halves of the workgroup's storage, and neither should grow with the
    // model's width. Both were 1024 floats at AF3's 128 channels; at 256 the
    // row rule alone made them 2048 each, which is the 1.71x.
    for (const [, rows, channels, factor] of SHAPES) {
      const tile = transitionRowTile(rows, channels);
      expect(tile * channels <= 1024).toBe(true);
      const intermediate = channels * factor;
      const staged = tile * transitionChunk(intermediate, tile, 128);
      // ...unless the intermediate itself is the floor, which is the single
      // track's case: tile 1 cannot stage less than one chunk.
      expect(staged <= Math.max(1024, intermediate)).toBe(true);
    }
  });

  it("still refuses a tile the rows cannot spare", () => {
    // The original rule, unchanged: 59 single-track rows tiled by four leaves
    // fifteen workgroups, which measured 30.6 -> 43.7 ms.
    expect(transitionRowTile(59, 384)).toBe(1);
    expect(transitionRowTile(59, 128)).toBe(1);
  });

  it("defaults to the old behaviour when no channels are given", () => {
    // Every caller passes them now; this is what says a caller that forgets
    // gets what the tree did before, not a silently different dispatch.
    expect(transitionRowTile(40_000)).toBe(8);
    expect(transitionRowTile(1_000)).toBe(4);
    expect(transitionRowTile(59)).toBe(1);
  });
});
