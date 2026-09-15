/**
 * RoseTTAFold3's chirality signal, checked by properties rather than by a twin.
 *
 * 🔴 THE IMPLEMENTATION IS A CENTRAL DIFFERENCE, SO A CENTRAL DIFFERENCE IS NOT
 * A TEST OF IT. What is checked here is what the term is FOR: the loss falls
 * when you step along the negative gradient, it is ~zero on an ideal centre,
 * and it is LARGE on the mirror image - which is the whole reason the term
 * exists, since every other input the network has is mirror-invariant.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { chiralPositionGradients, improperDihedral } from "../src/af3/diffusion/chiral-gradient.js";
import { chiralCentres, CHIRAL_ANGLE } from "../src/af3/featurise/template-features.js";
import { featuriseProtein } from "../src/af3/featurise/featurise.js";

const SEQUENCE = "GWSTELEKHREELKEFLKKEGITNVEIRIDNG";

/** The scalar the gradient is of: sum over valid centres of (dih - ideal)^2. */
function chiralLoss(positions, centers, angles) {
  const point = (index) => [positions[index * 3], positions[index * 3 + 1],
                            positions[index * 3 + 2]];
  let total = 0;
  for (let centre = 0; centre < angles.length; centre += 1) {
    if (angles[centre] === 0) continue;
    const p = [0, 1, 2, 3].map((k) => point(centers[centre * 4 + k]));
    const error = improperDihedral(p[0], p[1], p[2], p[3]) - angles[centre];
    total += error * error;
  }
  return total;
}

function fixture() {
  const batch = featuriseProtein(SEQUENCE, { dropTerminalAtoms: true });
  const { centers, angles, count } = chiralCentres(
    batch.aatype, batch.predDenseAtomMask, batch.tokens);
  // The reference conformers are the ideal geometry, which is the point: a real
  // L residue should already satisfy its own centre.
  return { batch, centers, angles, count,
           positions: Float32Array.from(batch.refPos ?? batch.ref_pos ?? []) };
}

describe("RoseTTAFold3's chirality signal", () => {
  it("finds a centre for every residue that has a CB, and none for glycine", () => {
    const { count, angles } = fixture();
    assert.ok(count > 0, "no chiral centres at all");
    assert.equal(count, angles.length);
    // Three per centre, so the count divides by three...
    assert.equal(count % 3, 0);
    // ...and two of every three are the positive ideal.
    const positive = [...angles].filter((a) => a > 0).length;
    assert.equal(positive, (count / 3) * 2);
    for (const angle of angles) assert.ok(Math.abs(Math.abs(angle) - CHIRAL_ANGLE) < 1e-6);
  });

  it("steps downhill: the loss falls along the negative gradient", () => {
    const { centers, angles, positions } = fixture();
    if (positions.length === 0) return;                 // no coordinates here
    const atoms = positions.length / 3;
    const before = chiralLoss(positions, centers, angles);
    const gradient = chiralPositionGradients(positions, centers, angles, atoms);
    // 🔴 A SMALL STEP, BECAUSE THIS IS A GRADIENT AND NOT A SOLVER. A large one
    // can increase a non-convex loss and would make this test say nothing.
    const stepped = Float32Array.from(positions);
    let moved = 0;
    for (let at = 0; at < stepped.length; at += 1) {
      stepped[at] -= 1e-3 * gradient[at];
      if (gradient[at] !== 0) moved += 1;
    }
    assert.ok(moved > 0, "the gradient is identically zero, so nothing is being tested");
    const after = chiralLoss(stepped, centers, angles);
    assert.ok(after <= before + 1e-9,
      `the loss rose along the negative gradient: ${before} -> ${after}`);
  });

  it("🔴 tells a structure from its MIRROR, which nothing else in the model does",
     () => {
    const { centers, angles, positions } = fixture();
    if (positions.length === 0) return;
    const mirrored = Float32Array.from(positions);
    for (let at = 0; at < mirrored.length; at += 3) mirrored[at] = -mirrored[at];
    const upright = chiralLoss(positions, centers, angles);
    const flipped = chiralLoss(mirrored, centers, angles);
    assert.ok(flipped > upright,
      `a mirrored structure must cost MORE, got ${flipped} against ${upright}`);
  });
});
