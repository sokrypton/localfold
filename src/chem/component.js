/**
 * A SMILES string, as the object `parseCcdComponent` would have returned.
 *
 * 🔴 THIS IS THE WHOLE INTEGRATION AND IT IS DELIBERATELY ONE FUNCTION. A
 * ligand reaches AF3's featuriser as
 * `{code, atoms: [{name, element, charge, x, y, z, leaving}], bonds: [{from,
 * to, order}]}` and as nothing else - the atomised tokens, `ref_pos`, the bond
 * matrix, the bond-order plane boltz2 reads, the chirality centres and
 * `bond-geometry.js`'s scoring all come off that one object. So SMILES support
 * is not a second featuriser or a second code path through the model: it is a
 * second way of producing this object, and everything downstream cannot tell
 * which one it got. `smilesComponent` is that producer.
 *
 * 🔴 AND ONLY AS MUCH CHEMISTRY AS THE MODELS READ. There is no canonical
 * SMILES here, no fingerprints, no descriptors, no substructure search, no
 * conformer ensemble and no force field - none of it reaches a model input.
 * What does: the element, the formal charge, the bonds with their orders, one
 * reference conformer, and the right enantiomer.
 */

import { parseSmiles } from "./smiles.js";
import { chiralCentres, doubleBondStereo } from "./stereo.js";
import {
  distanceBounds, smoothBounds, embedBounds, refineCoordinates, planarQuadruples,
  linearTriples,
} from "./conformer.js";
import { ELEMENT_SYMBOLS } from "../af3/featurise/ccd-component.js";

/**
 * Build a component from a SMILES string.
 *
 * @param {string} smiles
 * @param {object} [options]
 * @param {string} [options.code] what to call it; defaults to `LIG`
 * @param {number} [options.seed] overrides the seed derived from the string
 * @param {number} [options.attempts] how many embeddings to try
 * @param {(bounds: object, seed: number) => Float64Array} [options.embed]
 *   an alternative embedder
 * @param {(bounds: object, starts: Float64Array[], options: object) =>
 *   Promise<{coordinates: Float64Array, error: number}[]>} [options.refine]
 *   an alternative refiner - this is the seam `conformer-webgpu.js` uses, so
 *   the GPU path is a parameter rather than a branch inside the chemistry, and
 *   the differential gate can run both over one set of bounds
 * @returns {{code: string, atoms: object[], bonds: object[], smiles: string,
 *            conformerError: number, attempts: number}}
 */
export async function smilesComponent(smiles, options = {}) {
  const graph = parseSmiles(smiles);
  const code = options.code ?? "LIG";

  const bounds = distanceBounds(graph);
  const contradictions = smoothBounds(bounds);
  if (contradictions > 0) {
    // 🔴 REPORTED, NOT CLAMPED. A lower bound above its upper means the
    // geometry rules contradict each other for that pair, which is a bug in
    // the rules - quietly swapping them places the atom somewhere arbitrary
    // and the conformer comes out merely odd rather than obviously wrong.
    throw new Error(`${code}: ${contradictions} distance bound`
      + `${contradictions === 1 ? "" : "s"} contradict themselves after`
      + " smoothing; this molecule's geometry rules disagree");
  }

  const centres = chiralCentres(graph);
  const planar = planarQuadruples(graph, bounds.rings);
  const linear = linearTriples(graph, bounds.sigma, bounds.pi);
  const seed = options.seed ?? hashOf(smiles);
  const embed = options.embed ?? embedBounds;

  // 🔴 SEVERAL ATTEMPTS, BECAUSE ONE EMBEDDING CAN LAND IN A LOCAL MINIMUM IT
  // CANNOT DESCEND OUT OF - most often with a ring threaded through another
  // ring, which no amount of gradient will undo. Each attempt is a different
  // draw from the same bounds, and the best is kept. The seed makes the SET of
  // attempts reproducible, which is what matters: `ref_pos` is an input
  // feature and a ligand whose reference conformer moves between runs makes
  // the whole fold irreproducible.
  // 🔴 THE ATTEMPT COUNT IS ADAPTIVE, BECAUSE A FIXED ONE IS EITHER WASTEFUL OR
  // WRONG AND CANNOT BE BOTH RIGHT. Glycerol satisfies its bounds on the first
  // start and ATP does not: at four attempts ATP came back with an error of
  // 0.314 and a purine ring bent to 176 degrees where it should be 118, while
  // six attempts reach 0.013 and the cost of the extra two is milliseconds. So
  // the loop stops when the bounds are met rather than after a number chosen
  // for the average molecule - small ones pay for one start, hard ones get as
  // many as the budget allows.
  const attempts = options.attempts ?? 16;
  const enough = options.enough ?? 1e-3;
  const starts = [];
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    starts.push(embed(bounds, seed + attempt * 7919));
  }

  // 🔴 THE REFINEMENT IS THE SEAM, NOT THE EMBEDDING, AND THE MEASUREMENT IS
  // WHAT MOVED IT. At 100 atoms the CPU path spends 27.6 ms refining against
  // 7.9 embedding, 3.4 building bounds and 2.6 smoothing them - so the O(N^3)
  // triangle smoothing, which is the part that LOOKS like the expensive one,
  // is 6% of it. `conformer-webgpu.js` refines every attempt in one dispatch.
  let best = null;
  if (options.refine !== undefined) {
    const refined = await options.refine(bounds, starts, { planar, linear, chiral: centres });
    for (const one of refined) if (best === null || one.error < best.error) best = one;
  } else {
    for (const start of starts) {
      const one = refineCoordinates(start, bounds, centres, { ...options, planar, linear });
      if (best === null || one.error < best.error) best = one;
      // ...and an attempt that satisfies the bounds ends it, which the device
      // path cannot do because its attempts are one dispatch. The threshold is
      // "good enough" rather than "converged": an error of 1e-3 spread over
      // hundreds of pairs is thousandths of an angstrom each, and holding out
      // for 1e-6 makes every large molecule run the whole budget for a
      // difference nothing downstream can see.
      if (best.error < enough) break;
    }
  }

  const atoms = graph.atoms.map((atom, index) => ({
    name: "",                                        // filled in below
    element: atom.element,
    charge: atom.charge,
    x: round(best.coordinates[index * 3]),
    y: round(best.coordinates[index * 3 + 1]),
    z: round(best.coordinates[index * 3 + 2]),
    // 🔴 A SMILES LIGAND HAS NO LEAVING ATOMS. `leaving` marks the atom a
    // residue gives up on forming a peptide bond, which is a property of a
    // dictionary entry describing a free amino acid. A ligand drawn as a
    // structure is already the thing being folded. `polymerResidue` reads this
    // field, so it is present and false rather than absent.
    leaving: false,
  }));
  nameAtoms(atoms, graph);

  return {
    code,
    atoms,
    bonds: graph.bonds.map((bond) => ({
      from: bond.from, to: bond.to, order: Math.round(bond.order),
    })),
    smiles,
    conformerError: best.error,
    attempts,
    stereo: {
      centres: centres.length,
      doubleBonds: doubleBondStereo(graph).length,
    },
  };
}

const round = (value) => Math.round(value * 1000) / 1000;

/**
 * Give every atom a name, the way a CCD component does.
 *
 * 🔴 THE NAMES MATTER MORE THAN THEY LOOK. `bond-geometry.js` scores a ligand
 * by atom ORDER rather than by name for exactly one reason - rosettafold3
 * renames atomised atoms to their element and keying on "C1" then reports a
 * missing ligand - but the PDB this port writes out carries these names, and
 * that file is what a reader opens. Element-plus-counter is the dictionary's
 * own convention (`C1`, `C2`, `N1`, `O1`), so a SMILES ligand's output looks
 * like every other ligand's.
 *
 * 🔴 AND A NAME IS FOUR CHARACTERS IN A PDB. An atom past the 999th of one
 * element would overflow the column; a ligand that large is refused in
 * `component.js`'s caller rather than silently renamed.
 */
function nameAtoms(atoms, graph) {
  const counts = new Map();
  atoms.forEach((atom, index) => {
    const symbol = ELEMENT_SYMBOLS[graph.atoms[index].element - 1] ?? "X";
    const next = (counts.get(symbol) ?? 0) + 1;
    counts.set(symbol, next);
    // A component with ONE carbon calls it C, not C1, which is what the
    // dictionary does for a monatomic ion and for a lone substituent.
    atom.name = `${symbol}${next}`;
  });
  for (const [symbol, total] of counts) {
    if (total !== 1) continue;
    const only = atoms.find((atom) => atom.name === `${symbol}1`);
    if (only !== undefined) only.name = symbol;
  }
}

/**
 * A stable hash of the SMILES, used as the conformer seed.
 *
 * 🔴 DERIVED FROM THE STRING SO IT TRAVELS. The same ligand must give the same
 * reference conformer on every machine and in every session, or a fold is not
 * reproducible and the difference surfaces as an unexplained seed effect
 * somewhere far away. FNV-1a, because it needs to be stable rather than good.
 */
export function hashOf(text) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash || 1;
}

/**
 * A residue name for the nth DISTINCT SMILES ligand in one job.
 *
 * 🔴 THREE CHARACTERS, BECAUSE THAT IS THE PDB'S COLUMN. `src/af3/fold.js`
 * writes the residue name with `.padEnd(3)` into a fixed-width field, so a
 * four-character name runs into the chain id - and `LIG2` truncated back to
 * `LIG` would put the collision straight back. LIG, LG2 to LG9, then L10
 * upwards: ninety-nine distinct structures in one job.
 *
 * 🔴 AND IT LIVES HERE BECAUSE TWO CALLERS NEEDED IT AND ONLY ONE HAD IT. The
 * page assigned distinct names and `tools/gpu/fold.js` took a single
 * `--smiles-code` for every ligand, so a CLI fold of a benzene and an
 * ethylene glycol wrote ten atoms into ONE residue with `C1` and `C2`
 * appearing twice - a file no reader and no bond checker can make sense of.
 * One rule, one definition; the repository's standing complaint about a
 * convention written down in two places.
 */
export function ligandName(index) {
  if (index === 0) return "LIG";
  if (index < 9) return `LG${index + 1}`;
  if (index < 99) return `L${index + 1}`;
  throw new Error("at most 99 distinct SMILES ligands in one job");
}

/**
 * Names for a list of SMILES, sharing a name between identical strings.
 *
 * Identical structures are genuinely ONE entity and must share a name;
 * `featuriseProtein` then gives them successive sym_ids, which is the same
 * rule it applies to a repeated sequence.
 */
export function nameSmilesLigands(strings) {
  const named = new Map();
  return strings.map((text) => {
    if (!named.has(text)) named.set(text, ligandName(named.size));
    return named.get(text);
  });
}
