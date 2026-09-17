/**
 * Do a fold's atoms overlap each other? The question bonds cannot answer.
 *
 * 🔴 NOTHING IN THIS REPOSITORY MEASURED THIS, AND THE ARCHIVE SAYS SO OUT
 * LOUD. `web/fold-archive.js` omits AlphaFold 3's `has_clash` field under its
 * own rule that "a field we do not compute is left out, not filled in" - which
 * was honest and left a whole failure mode unwatched. A side chain can have
 * every bond at its ideal length, every angle right, and still be driven
 * THROUGH its neighbour: bond geometry is blind to it by construction, because
 * every pair it scores is bonded.
 *
 * The instruments that existed and what each cannot see:
 *
 *   chain-geometry.js   consecutive alpha carbons. Steps over side chains by
 *                       design, and a clash between two of them moves no CA.
 *   bond-geometry.js    bonded pairs against the reference conformer's ideals.
 *   this file           NON-bonded pairs, which is the complement of that and
 *                       is where "the ends are overlapping" lives.
 *
 * 🔴 THE THRESHOLD IS MolProbity'S AND THE UNITS ARE ITS CLASHSCORE, so the
 * number means something outside this repository. An overlap is
 * `(r_i + r_j) - d` on van der Waals radii, a CLASH is an overlap past 0.4 A,
 * and the score is clashes per thousand atoms. A well-refined crystal
 * structure scores in the low single digits; anything past ~20 is visibly
 * wrong.
 *
 * 🔴 WHAT IT SAID WHEN IT WAS FIRST POINTED AT THE SAMPLER DEFAULT, because
 * the question that prompted it was "are the page's defaults squashing side
 * chains". 5CAJ chain A, 261 residues, diffusion at the page's 25 steps, the
 * deposited crystal scoring 6.32 for comparison:
 *
 *     af3            14.7    boltz2        63.6     intellifold2  116.8
 *     rosettafold3  191.8    protenix2    246.3
 *
 * 🔴 AND rosettafold3 READS pLDDT 64.5 THERE - the MOST confident of the five
 * - with 347 carbon-on-carbon overlaps. That is this repository's recurring
 * lesson in a new place: pLDDT is blind to it, and reads 64.5 at 25 steps and
 * 64.5 at 200 while the clashes halve.
 *
 * 🔴 BUT THE SAMPLER DEFAULT IS NOT THE CAUSE, AND THE CONTROL IS WHAT SAYS
 * SO. Given a TEMPLATE, so the fold is determined, every model sits at the
 * crystal's level and the step count stops mattering: af3 6.2 at 25 steps and
 * 8.1 at 200, rosettafold3 11.9 and 13.3, against the crystal's 6.32. The
 * clashes track how well-determined the fold is, not how it was sampled. More
 * steps does help an UNDER-determined fold - 25 is the worst setting measured
 * for every model and 50 roughly halves it - and it does not close the gap:
 * protenix2 is still 68 at 200 steps, eleven times a crystal.
 *
 * 🔴 AND ONE CONFOUND IS NAMED RATHER THAN ASSUMED AWAY. On the worst case the
 * radius of gyration also grows with the step count, 22.0 A at 25 to 24.5 at
 * 200, so part of "fewer clashes" is a looser structure rather than a
 * resolved one. An 11% expansion does not explain a 23-fold drop, but it is
 * there, and a recommendation to raise the default should not rest on this one
 * target.
 *
 * 🔴 AND THE EXCLUSIONS ARE THE WHOLE DIFFICULTY. A pair three bonds apart or
 * closer is held where it is by the BONDS, not by sterics - counting it would
 * report every tetrahedral carbon as clashing with its own substituents - so
 * pairs within three bonds are skipped, which needs the bond graph rather than
 * a distance cutoff. Hydrogen bonds are the other case: an N and an O that are
 * donor and acceptor sit at 2.8-3.0 A, well inside their 3.07 A van der Waals
 * sum, and that is chemistry rather than a clash.
 */

import { idealBonds, oneLetter, parsePdbResidues } from "./bond-geometry.js";
import { vanDerWaalsRadius } from "../../src/chem/geometry-tables.js";

/** Overlap past this, in angstroms, is a clash. MolProbity's cutoff. */
const CLASH = 0.4;

/**
 * 🔴 A DONOR AND AN ACCEPTOR ARE ALLOWED CLOSER, OR EVERY HELIX IS A CLASH.
 * A backbone hydrogen bond puts N and O at about 2.9 A where their van der
 * Waals radii sum to 3.07, so a rule that does not know about it reports one
 * clash per residue of helix - which would drown the signal this exists to
 * find. MolProbity does the same thing by modelling hydrogens; with none in
 * the structure the honest approximation is to allow an N/O pair the distance
 * a hydrogen bond actually takes.
 */
const HYDROGEN_BONDING = new Set(["N", "O", "S"]);
const HYDROGEN_BOND_FLOOR = 2.6;

/** The element a PDB atom name implies, which is its first alphabetic letter. */
function elementOf(name) {
  const letter = name.replace(/[^A-Za-z]/g, "")[0];
  return letter === undefined ? "C" : letter.toUpperCase();
}

/**
 * Every atom of a parsed structure, flattened, with the bond graph that says
 * which pairs sterics does not govern.
 */
function atomsAndBonds(residues, conformers) {
  const atoms = [];
  const index = new Map();
  let skipped = 0;
  residues.forEach((residue, at) => {
    // 🔴 A RESIDUE WHOSE BONDS ARE UNKNOWN CONTRIBUTES NO ATOMS EITHER, or its
    // own bonded pairs read as clashes with each other. Leaving the atoms in
    // while skipping the bonds is what made a selenomethionine and a
    // benzamidine the worst "clashes" in two crystals - `MSE37:C .. MSE37:O`
    // at 1.23 A, which is a carbonyl. Counted, not silently dropped, because
    // "0 clashes" over a structure most of which was skipped is worse than a
    // number with a caveat.
    if (conformers[oneLetter(residue.code)] === undefined) { skipped += 1; return; }
    for (const [name, position] of residue.atoms) {
      // 🔴 HEAVY ATOMS ONLY, AND THAT IS NOT A SHORTCUT. This port's folds
      // carry no hydrogens - AF3 tokenises heavy atoms and the reference
      // conformers have none - so there is no hydrogen bond graph to exclude
      // against, and every C-H in a structure that HAS them reads as an
      // overlap of 1.87 A. A crystal with hydrogens scored 1586 that way. It
      // does mean this is a heavy-atom clash score rather than MolProbity's,
      // which places hydrogens first; the threshold is still theirs and the
      // calibration below is against structures scored the same way.
      if (elementOf(name) === "H" || elementOf(name) === "D") continue;
      index.set(`${at}:${name}`, atoms.length);
      atoms.push({ residue: at, name, position, element: elementOf(name),
                   code: residue.code, chain: residue.chain, number: residue.number });
    }
  });

  const neighbours = atoms.map(() => new Set());
  const bondCache = new Map();
  const join = (a, b) => {
    if (a === undefined || b === undefined) return;
    neighbours[a].add(b);
    neighbours[b].add(a);
  };
  residues.forEach((residue, at) => {
    // 🔴 THE BONDS COME FROM THE SAME CONFORMER SET bond-geometry.js SCORES
    // AGAINST, so the two instruments cannot disagree about what a bond is. A
    // residue this port has no conformer for contributes no bonds, which makes
    // its atoms LOOK non-bonded to each other - so it is skipped entirely
    // rather than counted as a cage of clashes.
    // 🔴 KEYED BY THE ONE-LETTER CODE AND DERIVED, NOT STORED. The conformer
    // set is `{A: {...}, C: {...}}` and a PDB says `ALA`, and a conformer
    // carries POSITIONS rather than a bond list - `idealBonds` is what turns
    // one into the other. Looked up by the three-letter code this found
    // nothing, every atom looked non-bonded to every other, and a refined
    // CRYSTAL scored 2745 clashes per thousand atoms where the truth is single
    // digits. A calibration control is what caught it; without one the number
    // would have been believed.
    const conformer = conformers[oneLetter(residue.code)];
    if (conformer === undefined) return;
    // 🔴 ALL THREE FORMS OF THE RESIDUE, UNIONED. `idealBonds` reads
    // `internal`, and OXT lives only in `cTerminal` - so a chain's last
    // residue had no C-OXT bond and its own carboxyl terminus was the worst
    // clash in the structure, at 1.23 A. A bond that appears in any form of a
    // residue is a bond; taking the union can only make the exclusion more
    // generous, and every pair it adds is one sterics does not govern anyway.
    for (const form of ["internal", "nTerminal", "cTerminal"]) {
      if (conformer[form] === undefined) continue;
      for (const bond of idealBonds({ internal: conformer[form] }, bondCache,
                                    `${residue.code}:${form}`)) {
        join(index.get(`${at}:${bond.a}`), index.get(`${at}:${bond.b}`));
      }
    }
    // ...and the peptide bond, which belongs to no single residue.
    const next = residues[at + 1];
    if (next !== undefined && next.chain === residue.chain && !residue.hetatm) {
      join(index.get(`${at}:C`), index.get(`${at + 1}:N`));
    }
  });
  return { atoms, neighbours, skipped };
}

/** Which pairs are within `hops` bonds of each other. */
function nearInBonds(neighbours, hops) {
  return neighbours.map((_, start) => {
    const seen = new Set([start]);
    let front = [start];
    for (let step = 0; step < hops; step += 1) {
      const next = [];
      for (const atom of front) {
        for (const other of neighbours[atom]) {
          if (seen.has(other)) continue;
          seen.add(other);
          next.push(other);
        }
      }
      front = next;
    }
    return seen;
  });
}

/**
 * Score a structure for steric clashes.
 *
 * @param {string} pdb
 * @param {object} conformers the reference conformer set, for the bond graph
 * @returns {{score: number, clashes: number, atoms: number,
 *            worst: object[], bySidechain: number, byBackbone: number}}
 *   `score` is clashes per thousand atoms, MolProbity's unit.
 */
export function clashScore(pdb, conformers, options = {}) {
  const cutoff = options.cutoff ?? CLASH;
  const residues = parsePdbResidues(pdb);
  const { atoms, neighbours, skipped } = atomsAndBonds(residues, conformers);
  const near = nearInBonds(neighbours, 3);

  const BACKBONE = new Set(["N", "CA", "C", "O", "OXT"]);
  const found = [];
  for (let i = 0; i < atoms.length; i += 1) {
    for (let j = i + 1; j < atoms.length; j += 1) {
      // Three bonds or fewer apart is geometry, not sterics.
      if (near[i].has(j)) continue;
      const a = atoms[i];
      const b = atoms[j];
      const dx = a.position[0] - b.position[0];
      const dy = a.position[1] - b.position[1];
      const dz = a.position[2] - b.position[2];
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const sum = vanDerWaalsRadius(a.element) + vanDerWaalsRadius(b.element);
      if (distance >= sum - cutoff) continue;
      if (HYDROGEN_BONDING.has(a.element) && HYDROGEN_BONDING.has(b.element)
        && distance >= HYDROGEN_BOND_FLOOR) continue;
      found.push({
        overlap: sum - distance, distance,
        a: `${a.code}${a.number}${a.chain}:${a.name}`,
        b: `${b.code}${b.number}${b.chain}:${b.name}`,
        sidechain: !BACKBONE.has(a.name) || !BACKBONE.has(b.name),
        // 🔴 A CARBON OVERLAP IS UNAMBIGUOUS AND AN N/O ONE IS NOT. Two
        // carbons at 2.9 A have no business being there; a nitrogen and an
        // oxygen at 2.9 A are a hydrogen bond, and an arginine against a
        // glutamate is a salt bridge that any real structure is full of. With
        // no hydrogens to place, the honest thing is to report the two
        // separately rather than pick a floor and pretend it settles them -
        // "the ends are overlapping" is a claim about the first kind.
        polar: HYDROGEN_BONDING.has(a.element) && HYDROGEN_BONDING.has(b.element),
      });
    }
  }
  found.sort((x, y) => y.overlap - x.overlap);
  return {
    score: atoms.length === 0 ? 0
      : Number(((1000 * found.length) / atoms.length).toFixed(2)),
    clashes: found.length,
    atoms: atoms.length,
    // Residues with no conformer - ligands, modified residues - are not
    // scored, and saying how many is the difference between a clean
    // number and a misleading one.
    skippedResidues: skipped,
    bySidechain: found.filter((one) => one.sidechain).length,
    byBackbone: found.filter((one) => !one.sidechain).length,
    // The unambiguous ones: nothing a hydrogen bond could explain.
    nonPolar: found.filter((one) => !one.polar).length,
    polar: found.filter((one) => one.polar).length,
    worst: found.slice(0, 5).map((one) => ({
      ...one, overlap: Number(one.overlap.toFixed(2)),
      distance: Number(one.distance.toFixed(2)),
    })),
  };
}
