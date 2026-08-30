import { describe, expect, it } from "./harness.js";
import { alphaCarbons, anchoredStart, morphFrames } from "../web/morph.js";

const CA = 1;
const CB = 3;
const CG = 5;

/** A structure of `length` residues with the named atom37 slots present. */
const build = (length, slots, place) => {
  const atom37 = new Float32Array(length * 37 * 3);
  const atom37Mask = new Float32Array(length * 37);
  for (let residue = 0; residue < length; residue += 1) {
    for (const slot of slots) {
      const index = residue * 37 + slot;
      atom37Mask[index] = 1;
      const [x, y, z] = place(residue, slot);
      atom37[index * 3] = x; atom37[index * 3 + 1] = y; atom37[index * 3 + 2] = z;
    }
  }
  return { atom37, atom37Mask };
};
const at = (structure, residue, slot) => {
  const i = (residue * 37 + slot) * 3;
  return [structure.atom37[i], structure.atom37[i + 1], structure.atom37[i + 2]];
};

describe("morphing one prediction into the next", () => {
  it("names one alpha carbon per residue, in atom37 slots", () => {
    expect(alphaCarbons(3)).toEqual([CA, 37 + CA, 74 + CA]);
    expect(alphaCarbons(0)).toEqual([]);
  });

  // 🔴 THE MUTATED RESIDUE IS THE ONE THE TWO STRUCTURES DISAGREE ABOUT, which
  // is where a pairwise interpolation is undefined and where the reader is
  // looking. An atom the old structure did not have starts at that residue's
  // anchor - its old CB - so a new side chain grows out of the stub that was
  // there rather than being present, finished, from the first frame.
  it("starts an atom the old structure lacked at the old residue's CB", () => {
    const from = build(2, [CA, CB], (r, s) => [r, s, 0]);
    const to = build(2, [CA, CB, CG], (r, s) => [r + 10, s + 10, 10]);
    const start = anchoredStart(from, to, 2);
    // ...the shared atoms start where they were
    expect(start[0 * 37 + CA]).toEqual([0, CA, 0]);
    expect(start[1 * 37 + CB]).toEqual([1, CB, 0]);
    // ...and the new one starts at the old CB of ITS OWN residue, not at its
    // own final position and not at residue 0's
    expect(start[0 * 37 + CG]).toEqual([0, CB, 0]);
    expect(start[1 * 37 + CG]).toEqual([1, CB, 0]);
  });

  // GLYCINE HAS NO CB, and mutating away from a glycine is the very case that
  // grows a side chain from nothing - so the fallback is not an edge case, it
  // is the common one.
  it("falls back to CA where the old residue had no CB", () => {
    const from = build(1, [CA], () => [5, 5, 5]);
    const to = build(1, [CA, CB, CG], (r, s) => [s, 0, 0]);
    const start = anchoredStart(from, to, 1);
    expect(start[CB]).toEqual([5, 5, 5]);
    expect(start[CG]).toEqual([5, 5, 5]);
  });

  it("runs from one structure to the other, both ends exact", () => {
    const from = build(3, [CA], (r) => [r, 0, 0]);
    const to = build(3, [CA], (r) => [r, 10, 0]);
    const plddt = Float32Array.of(50, 50, 50);
    const high = Float32Array.of(90, 90, 90);
    const frames = morphFrames(from, to, plddt, high, 3, 5);
    expect(frames.length).toBe(5);
    // ...the first frame IS the old structure and the last IS the new one.
    // An eased ramp that does not reach its ends leaves the fold sitting a
    // fraction short of the answer it just computed.
    expect(at(frames[0].structure, 1, CA)).toEqual([1, 0, 0]);
    expect(at(frames[4].structure, 1, CA)).toEqual([1, 10, 0]);
    expect(frames[0].confidence.plddt[0]).toBe(50);
    expect(frames[4].confidence.plddt[0]).toBe(90);
    // ...and it is monotonic in between, which a bad easing is not
    const ys = frames.map((f) => at(f.structure, 1, CA)[1]);
    for (let i = 1; i < ys.length; i += 1) expect(ys[i] >= ys[i - 1]).toBe(true);
    // SMOOTHSTEP, so the middle frame is halfway and the ends are slower than
    // a straight ramp would make them
    expect(Math.abs(ys[2] - 5) < 1e-5).toBe(true);
    expect(ys[1] < 2.5).toBe(true);
  });

  // EVERY FRAME CARRIES THE NEW MASK, which is what keeps them all the same
  // length - py2Dmol aligns and caches per length, and frames that disagree
  // about how many positions they have cannot be either.
  it("gives every frame the same atom set, the new one", () => {
    const from = build(2, [CA, CB], (r, s) => [r, s, 0]);
    const to = build(2, [CA, CB, CG], (r, s) => [r, s, 5]);
    const frames = morphFrames(from, to, Float32Array.of(1, 1), Float32Array.of(2, 2), 2, 4);
    for (const frame of frames) {
      expect(frame.structure.atom37Mask).toBe(to.atom37Mask);
      expect(frame.structure.atom37.length).toBe(2 * 37 * 3);
    }
  });

  it("refuses a morph with fewer than its two ends", () => {
    const one = build(1, [CA], () => [0, 0, 0]);
    expect(() => morphFrames(one, one, Float32Array.of(1), Float32Array.of(1), 1, 1)).toThrow();
    expect(() => morphFrames(one, one, Float32Array.of(1), Float32Array.of(1), 1, 0)).toThrow();
  });
});
