import { describe, expect, it } from "./harness.js";
import { superposeApi } from "../tools/gpu/superpose.js";

/**
 * 🔴 A STREAMED AlphaFold 3 TRAJECTORY TUMBLES WITHOUT THIS. The sampler calls
 * randomAugmentation at the top of every step - a fresh rotation and
 * translation of the whole system - so consecutive frames differ by a rigid
 * motion far larger than anything the denoiser did. `fittedPdb` undoes it, and
 * it asks a VIEWER LIBRARY for the superposition: py2Dmol's
 * `superpose(mobile, reference, {from, to})`. A runtime that folds in a
 * process has no viewer, so it passes this adapter over the Kabsch this
 * repository already folds AF2 with.
 *
 * The contract has two halves and only one of them is obvious: fit on the
 * points NAMED, and move EVERY point of the frame.
 */
describe("superposeApi", () => {
  const random = (seed) => {
    let state = seed;
    return () => (state = (state * 1103515245 + 12345) % 2147483648) / 2147483648;
  };
  const rmsd = (a, b) => Math.sqrt(a.reduce((sum, p, i) =>
    sum + (p[0] - b[i][0]) ** 2 + (p[1] - b[i][1]) ** 2 + (p[2] - b[i][2]) ** 2,
    0) / a.length);

  const truth = (() => {
    const next = random(7);
    return Array.from({ length: 40 }, () => [next() * 30, next() * 30, next() * 30]);
  })();
  // The shape of what the sampler does: a rotation and a translation, nothing else.
  const angle = 40 * Math.PI / 180;
  const shifted = truth.map(([x, y, z]) => [
    x * Math.cos(angle) - y * Math.sin(angle) + 11,
    x * Math.sin(angle) + y * Math.cos(angle) - 4,
    z + 7,
  ]);

  it("is measuring a real displacement, or it proves nothing", () => {
    // The control. An adapter that returned its input unchanged would pass
    // every check below if the frames had not moved in the first place.
    expect(rmsd(shifted, truth)).toBeGreaterThan(5);
  });

  it("undoes a rigid motion", () => {
    expect(rmsd(superposeApi(shifted, truth), truth)).toBeLessThan(1e-6);
  });

  it("fits on the named slots and moves everything", () => {
    // 🔴 THE HALF THAT IS EASY TO GET WRONG. `fittedPdb` fits on the alpha
    // carbons and writes every atom, so an adapter that returned only the
    // fitted subset would produce a PDB missing most of its atoms.
    //
    // 🔴 AND THE POINTS OFF THE SLOTS MUST NOT SHARE THE MOTION, or the check
    // cannot see the slots being ignored: under a PURE rigid motion, fitting
    // on twelve points and fitting on forty give the same transform, and a
    // mutation that drops the slot arguments passes. Measured - it did. The
    // rest of the frame is displaced as well, which is also the honest case:
    // a denoiser moves side chains while the backbone is what gets fitted.
    const slots = Array.from({ length: 12 }, (_, i) => i * 3);
    const onSlots = new Set(slots);
    const noisy = shifted.map((p, i) => (onSlots.has(i) ? p
      : [p[0] + (i % 5) - 2, p[1] + (i % 3) - 1, p[2] + (i % 7) - 3]));

    const placed = superposeApi(noisy, truth, { from: slots, to: slots });
    expect(placed.length).toBe(40);
    // The named points land exactly; the others carry their own displacement.
    const pick = (points) => slots.map((i) => points[i]);
    expect(rmsd(pick(placed), pick(truth))).toBeLessThan(1e-6);
  });
});
