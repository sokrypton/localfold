/**
 * align.js
 * --------
 * Put every cycle of a run in the same place, so the play bar shows the design
 * moving rather than the camera.
 *
 * 🔴 A FOLD HAS NO FRAME OF REFERENCE. Each cycle is an independent prediction,
 * so its coordinates sit wherever the model put them: played back, a design
 * tumbles through space and the thing that actually changed - a loop closing, a
 * helix straightening - is invisible underneath the tumbling.
 *
 * 🔴 ALIGNED ON THE TARGET, NOT ON EVERYTHING. The target is in every cycle and
 * nothing designed it, so holding it still is what makes the binder the only
 * thing that moves. Superposing the whole complex averages the two and the
 * target drifts to meet the binder, which is the opposite of what the viewer is
 * for. With no target - a monomer design - there is nothing to hold but itself,
 * and the whole chain is the reference.
 *
 * Pure: PDB strings in, a PDB string out. It is in `cpu/` because nothing here
 * touches a GPU or a DOM, and `test/design-align.test.js` holds it.
 */
import { superimposeCoords3D } from "../../shared/rfd4p/output.js";
import { alphaCarbons, coordinateAtoms, rewriteCoordinates } from "../../shared/heads/superpose-pdb.js";

/**
 * `pdb` moved onto `reference`, by whichever chains they share besides `designed`.
 *
 * @param {string} pdb the cycle to move
 * @param {string} reference the cycle to move it onto - the first, normally
 * @param {string|string[]} [designed] the chain(s) being designed, which are NOT used to fit
 * @returns {string} the same PDB with every coordinate rewritten, or the input
 *   unchanged when there is nothing to fit on
 */
export function alignToReference(pdb, reference, designed) {
  const moving = coordinateAtoms(pdb);
  const fixed = coordinateAtoms(reference);

  // The chains both structures have, minus the one being designed.
  const held = Array.isArray(designed) ? designed : [designed];
  const onto = (chain) => !held.includes(chain);
  let movingIndices = alphaCarbons(moving, onto);
  let fixedIndices = alphaCarbons(fixed, onto);

  // 🔴 A MONOMER DESIGN HAS NOTHING ELSE TO HOLD, so it is fitted on itself.
  // That still removes the tumbling, which is the point; what it cannot do is
  // show the chain moving relative to something, because there is nothing.
  if (movingIndices.length === 0 || fixedIndices.length === 0) {
    movingIndices = alphaCarbons(moving, () => true);
    fixedIndices = alphaCarbons(fixed, () => true);
  }

  // 🔴 AND A FIT NEEDS THE SAME ATOMS ON BOTH SIDES. A cycle whose target lost a
  // residue would otherwise be fitted on a different set and land anywhere; the
  // frame is left where it is instead, which looks like a jump and IS one.
  const count = Math.min(movingIndices.length, fixedIndices.length);
  if (count < 3) return pdb;

  const movingCa = new Float32Array(count * 3);
  const fixedCa = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const from = moving.points[movingIndices[index]];
    const to = fixed.points[fixedIndices[index]];
    movingCa[index * 3] = from[0];
    movingCa[index * 3 + 1] = from[1];
    movingCa[index * 3 + 2] = from[2];
    fixedCa[index * 3] = to[0];
    fixedCa[index * 3 + 1] = to[1];
    fixedCa[index * 3 + 2] = to[2];
  }

  const { rotation, fixedCenter, mobileCenter } = superimposeCoords3D(fixedCa, movingCa);

  // 🔴 EVERY ATOM MOVES, NOT ONLY THE ONES FITTED ON. The transform is found
  // from the target's alpha carbons and applied to the whole file - the binder
  // included, which is the half nobody fitted and the half worth watching.
  const moved = moving.points.map((point) => {
    const x = point[0] - mobileCenter[0];
    const y = point[1] - mobileCenter[1];
    const z = point[2] - mobileCenter[2];
    return [
      rotation[0] * x + rotation[1] * y + rotation[2] * z + fixedCenter[0],
      rotation[3] * x + rotation[4] * y + rotation[5] * z + fixedCenter[1],
      rotation[6] * x + rotation[7] * y + rotation[8] * z + fixedCenter[2],
    ];
  });

  return rewriteCoordinates(moving, moved);
}

/**
 * How far the designed chain moved between two cycles, IN THE TARGET'S FRAME: the C-alpha RMSD after
 * `alignToReference` has fitted everything but it - so a binder that kept its shape and slid round
 * its target counts the slide.
 *
 * 🔴 THE FIRST MEASURE SUPERPOSED THE BINDER ON ITSELF (2026-10-28), which measures its SHAPE only:
 * the viewer, fitted on the target, showed a binder wandering across its target while the Moved
 * column read 1-2 A. A monomer has nothing else to fit on and is measured on itself, as the viewer
 * fits it.
 *
 * @param {string} previous the cycle before
 * @param {string} pdb this cycle
 * @param {string} designed the designed chain
 * @returns {number} angstroms
 */
export function designedDrift(previous, pdb, designed) {
  const placed = coordinateAtoms(alignToReference(pdb, previous, designed));
  const fixed = coordinateAtoms(previous);
  // (the designed chain by its C-alphas, or a designed strand by its C4's)
  const traced = (atoms) => atoms.names.flatMap((name, index) =>
    ((name === "CA" || name === "C4'") && atoms.chains[index] === designed ? [index] : []));
  const others = alphaCarbons(fixed, (chain) => chain !== designed).length > 0;
  const a = traced(placed);
  const b = traced(fixed);
  if (a.length !== b.length || a.length === 0) {
    throw new Error(`chain ${designed} has ${b.length} residues in the cycle before and ${a.length} in this one`);
  }
  if (!others) {
    // ...a monomer: on itself, which alignToReference has already done
    const moving = new Float32Array(a.flatMap((i) => placed.points[i]));
    const still = new Float32Array(b.flatMap((i) => fixed.points[i]));
    return superimposeCoords3D(still, moving).rmsd;
  }
  let sum = 0;
  for (let k = 0; k < a.length; k += 1) {
    const p = placed.points[a[k]];
    const q = fixed.points[b[k]];
    sum += (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
  }
  return Math.sqrt(sum / a.length);
}
