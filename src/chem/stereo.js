/**
 * Tetrahedral chirality: from a SMILES `@` to a sign the embedder can enforce.
 *
 * 🔴 DISTANCE GEOMETRY CANNOT SEE A MIRROR IMAGE. Every pairwise distance in a
 * molecule is identical to every pairwise distance in its enantiomer - that is
 * what makes them enantiomers - so the embedding returns whichever one the
 * random draw landed nearest, and half of all runs would hand the model the
 * wrong isomer. For a drug that is not a rounding error: thalidomide's two
 * enantiomers are a sedative and a teratogen. The signed volume of the
 * tetrahedron on a centre's four neighbours is the one quantity that differs,
 * and `refineCoordinates` drives it to the right sign.
 *
 * 🔴 AND THE NEIGHBOUR ORDER IS THE SMILES ORDER, WHICH IS NOT THE BOND ORDER.
 * `@` means: looking from the FIRST neighbour towards the centre, the other
 * three appear anticlockwise in the order written. The first neighbour is the
 * atom that PRECEDED this one in the string, and an implicit hydrogen written
 * inside the brackets takes the place immediately after it - or the first
 * place, when the centre opens the string. Getting that order wrong inverts
 * every centre in the molecule, and nothing about the result looks wrong
 * except that it is the other enantiomer.
 *
 * 🔴 AND A RING CLOSURE TAKES ITS PLACE WHERE THE DIGIT IS, not where the ring
 * closes. In `[C@@H]1CCCCO1` the ring-closure neighbour is the second
 * neighbour of the centre because the `1` is written there, even though the
 * atom it bonds to is read five atoms later. Ordering by bond index instead
 * puts it last and flips the centre.
 */

import { adjacency } from "./smiles.js";

/**
 * 🔴 CALIBRATED AGAINST RDKit RATHER THAN DERIVED, AND SAID SO OUT LOUD. The
 * OpenSMILES text fixes the geometric meaning of `@` - anticlockwise from the
 * first neighbour - but which SIGN of a determinant that is depends on a
 * handedness convention this file would otherwise be asserting from memory.
 * `tools/check-stereo-vs-rdkit.mjs` measures it: it takes every centre RDKit
 * assigns a CIP code to, reads RDKit's own 3D conformer, and checks that this
 * sign convention reproduces the R/S it assigned. Both alanine enantiomers and
 * all eight of cholesterol's centres are in that gate.
 */
const ANTICLOCKWISE = 1;

/**
 * The chiral centres of a graph, as the embedder wants them.
 *
 * @param {{atoms: object[], bonds: object[]}} graph
 * @returns {{atom: number, neighbours: number[], sign: number}[]} `neighbours`
 *   is four atom indices; where the centre has only three heavy neighbours and
 *   a hydrogen, the CENTRE ITSELF stands in for the hydrogen - see below.
 */
export function chiralCentres(graph) {
  const lists = adjacency(graph);
  const centres = [];

  graph.atoms.forEach((atom, index) => {
    if (atom.chirality !== "@" && atom.chirality !== "@@") return;
    const ordered = neighbourOrder(graph, lists, index);
    if (ordered === null) return;

    // 🔴 THE CENTRE STANDS IN FOR THE HYDROGEN, AND THAT IS EXACT RATHER THAN
    // AN APPROXIMATION. This port drops hydrogens before the featuriser sees
    // them, so a centre with three heavy neighbours has no fourth point. The
    // centre lies INSIDE the tetrahedron of its four neighbours, so it is on
    // the same side of the plane through the other three as the hydrogen is,
    // and the determinant's SIGN - which is all this term reads - is
    // unchanged. Its magnitude is not, which is why the target below is a
    // sign test with a floor rather than a value to match.
    const neighbours = ordered.map((entry) => (entry === "H" ? index : entry));
    if (neighbours.length !== 4) return;

    centres.push({
      atom: index,
      neighbours,
      sign: atom.chirality === "@" ? ANTICLOCKWISE : -ANTICLOCKWISE,
    });
  });
  return centres;
}

/**
 * The centre's neighbours in the order SMILES wrote them, with "H" standing
 * for an implicit hydrogen in its correct place.
 */
function neighbourOrder(graph, lists, centre) {
  const atom = graph.atoms[centre];
  // 🔴 THE PARSER'S WRITTEN ORDER, NOT THE BOND ORDER. See the note on
  // `writtenNeighbours` in smiles.js: a ring-closure bond is written as a
  // digit on the opening atom and CREATED when the ring closes, so sorting by
  // bond index moves it to the end of the list and inverts the centre. That
  // was 9 of 29 corpus centres coming out as the wrong enantiomer - biotin's
  // two, ATP's ribose, glucose, three of cholesterol's, penicillin's and
  // NAD's - every one of them a ring opening, with the geometry otherwise
  // perfect. Measured by `tools/check-stereo-vs-rdkit.mjs`.
  const written = (graph.writtenNeighbours?.[centre] ?? [])
    .filter((entry) => typeof entry === "number");
  if (written.length !== lists[centre].length) return null;
  const neighbourOf = (bond) =>
    (graph.bonds[bond].from === centre ? graph.bonds[bond].to : graph.bonds[bond].from);

  const order = written.map(neighbourOf);
  // The implicit hydrogen takes the place right after the preceding atom, or
  // the very first place when the centre opens the string.
  if (atom.hydrogens > 0) {
    const hasPreceding = order.length > 0 && order[0] < centre;
    order.splice(hasPreceding ? 1 : 0, 0, "H");
  }

  // A centre needs exactly four things to be tetrahedral. Three means one
  // implicit hydrogen that the bracket did not declare, which for a chiral
  // centre is a contradiction rather than something to infer: `[C@](C)(N)O`
  // has three neighbours and no `H`, and OpenSMILES leaves it undefined.
  if (order.length !== 4) return null;
  // ...and more than one hydrogen on a centre makes it not a centre at all.
  if (atom.hydrogens > 1) return null;
  return order;
}

/**
 * Which double bonds carry a `/` `\` arrangement, and whether it is cis.
 *
 * 🔴 KEPT SEPARATE FROM THE TETRAHEDRAL CENTRES BECAUSE IT IS ENFORCED
 * DIFFERENTLY. A double bond's geometry is already pinned by the 1-4 distance
 * bounds - cis and trans are simply two different distances between the
 * substituents - so this returns a DISTANCE to fix rather than a sign to
 * drive, and `component.js` narrows the bound instead of adding a term.
 *
 * @returns {{from: number, to: number, first: number, second: number,
 *            cis: boolean}[]}
 */
export function doubleBondStereo(graph) {
  const lists = adjacency(graph);
  const found = [];
  graph.bonds.forEach((bond, index) => {
    if ((bond.geometryOrder ?? bond.order) !== 2) return;
    const first = directedNeighbour(graph, lists, bond.from, index);
    const second = directedNeighbour(graph, lists, bond.to, index);
    if (first === null || second === null) return;
    // 🔴 THE TWO MARKS ARE READ RELATIVE TO THE ATOM THAT CARRIES THEM, so the
    // same character on both ends means OPPOSITE sides. `F/C=C/F` is trans and
    // `F/C=C\F` is cis, which is the reverse of how the slashes look.
    // 🔴 AND THE SENSE IS THE REVERSE OF HOW THE SLASHES LOOK, WHICH IS WHY
    // THIS IS MEASURED RATHER THAN REASONED. `F/C=C/F` is TRANS and
    // `F/C=C\\F` is cis; the first version here had both backwards and scored
    // 0 of 2 against RDKit's own conformers, which is the useful kind of
    // wrong - a convention inverted everywhere shows up immediately, where one
    // inverted half the time would not.
    const sameMark = first.direction === second.direction;
    const flipped = (first.before ? 1 : 0) + (second.before ? 1 : 0) === 1;
    found.push({
      from: bond.from, to: bond.to,
      first: first.atom, second: second.atom,
      cis: flipped ? !sameMark : sameMark,
    });
  });
  return found;
}

function directedNeighbour(graph, lists, atom, skipBond) {
  for (const step of lists[atom]) {
    if (step.bond === skipBond) continue;
    const bond = graph.bonds[step.bond];
    if (bond.direction === null) continue;
    return {
      atom: step.atom,
      direction: bond.direction,
      // Whether this atom is the bond's `from`, which is what says whether the
      // mark points towards the double bond or away from it.
      before: bond.from === atom,
    };
  }
  return null;
}
